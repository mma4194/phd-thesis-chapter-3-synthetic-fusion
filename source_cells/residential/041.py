# ==========================================================
# CELL 10.9 — Materialize controlled protocol variants — v2.1-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Materialize controlled protocol-level synthetic TEST variants:
#
#     A0_PROTOCOL_TEST:
#       TRAIN-only baseline values under real TEST masks, already produced by
#       Cell 10.1. Diagnostic/control only. Not used for selection.
#
#     A1_PROTOCOL_TEST:
#       TRAIN-only baseline values under synthetic TEST masks, already produced
#       by Cell 10.1. Selection baseline for Cell 10.8.
#
#     A2_PROTOCOL_TEST:
#       VAL-locked selected protocol portfolio from Cell 10.8.
#
# Scientific contract:
# - This is a pure materialization cell.
# - It does not fit, refit, select, retune, repair, or compute TEST-vs-real metrics.
# - It does not read real TEST target values.
# - It does not materialize downstream rescue/generator logic.
# - Selected non-A1 columns must have valid registry-backed synthetic TEST
#   candidate paths. Missing TEST paths fail closed.
# - A2 is built by starting from A1_TEST and replacing only the VAL-selected
#   non-A1 columns using their synthetic TEST candidate parquet columns.
#
# Inputs expected:
# - portfolio_candidates/A0_TEST_baseline.parquet
# - portfolio_candidates/A1_TEST_baseline.parquet
# - artifacts/cell10_selected_generator_by_col.json
# - artifacts/cell10_selected_generator_by_col_rich.json
# - artifacts/cell10_portfolio_selection_summary.json
# - artifacts/cell10_portfolio_candidate_registry_v3.json
# - artifacts/contracts/cell10_8_val_only_selector_contract_v3_6_THESIS.json
#
# Outputs:
# - synthetic/A0_PROTOCOL_TEST.parquet
# - synthetic/A1_PROTOCOL_TEST.parquet
# - synthetic/A2_PROTOCOL_TEST.parquet
# - reports/cell10_9_variant_materialization_audit.csv
# - reports/cell10_9_a2_materialization_by_col.csv
# - reports/cell10_9_missing_test_materialization_required.csv, only on fail
# - artifacts/cell10_9_protocol_variant_manifest.json
# - artifacts/contracts/cell10_9_protocol_variant_materialization_contract_v2_1_THESIS.json
# ==========================================================

log("--- START: Cell 10.9 — Materialize A0/A1/A2 protocol variants (v2.1-THESIS pure materializer) ---")

import os
import re
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd


# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
need = [
    "CFG", "log",
    "df_te",
    "PROTO_VALUE_COLS",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.9] Missing required globals: {missing}. Run Cells 1–10.8 first.")

# Clean-run leakage guard.
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[Cell10.9] Clean protocol materializer forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.9] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
REPDIR = os.path.join(OUTDIR, "reports")
CANDDIR = os.path.join(OUTDIR, "portfolio_candidates")
SYNDIR = os.path.join(OUTDIR, "synthetic")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

for d in [ARTDIR, REPDIR, CANDDIR, SYNDIR, CONTRACT_DIR]:
    os.makedirs(d, exist_ok=True)

N_TEST = int(len(df_te))
if N_TEST <= 0:
    raise RuntimeError("[Cell10.9] df_te is empty.")


# ----------------------------------------------------------
# 1) Required paths
# ----------------------------------------------------------
a0_test_path = os.path.join(CANDDIR, "A0_TEST_baseline.parquet")
a1_test_path = os.path.join(CANDDIR, "A1_TEST_baseline.parquet")

selection_json_path = os.path.join(ARTDIR, "cell10_selected_generator_by_col.json")
selection_rich_json_path = os.path.join(ARTDIR, "cell10_selected_generator_by_col_rich.json")
selection_summary_path = os.path.join(ARTDIR, "cell10_portfolio_selection_summary.json")
candidate_registry_v3_path = os.path.join(ARTDIR, "cell10_portfolio_candidate_registry_v3.json")
baseline_manifest_path = os.path.join(ARTDIR, "cell10_a0_a1_baseline_manifest.json")
selector_contract_path = os.path.join(CONTRACT_DIR, "cell10_8_val_only_selector_contract_v3_6_THESIS.json")

required_paths = [
    (a0_test_path, "A0 TEST baseline"),
    (a1_test_path, "A1 TEST baseline"),
    (selection_json_path, "Cell 10.8 selected-generator registry"),
    (selection_summary_path, "Cell 10.8 selection summary"),
    (candidate_registry_v3_path, "Cell 10.8 candidate registry v3"),
    (selector_contract_path, "Cell 10.8 v3.6 selector contract"),
]

for p, label in required_paths:
    if not os.path.exists(p):
        raise RuntimeError(f"[Cell10.9] Missing {label}: {p}. Run Cells 10.1–10.8 first.")


# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_jsonable(obj) -> str:
    payload = json.dumps(_json_sanitize(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parquet_num_rows(path: str):
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _parquet_columns(path: str):
    try:
        import pyarrow.parquet as pq
        return list(map(str, pq.ParquetFile(path).schema.names))
    except Exception:
        return None


def _realpath(path: str) -> str:
    return os.path.realpath(os.path.abspath(str(path)))


def _tier_of(col: str):
    c = str(col)
    if c.startswith("router__"):
        return "router"
    if c.startswith("ota__"):
        return "ota"
    if c.startswith("zigbee__"):
        return "zigbee"
    if c.startswith("zwave__"):
        return "zwave"
    return "unknown"


COUNTLIKE_RX = re.compile(
    r"(_total$|_pkt$|_pkts$|_packets$|_bytes$|_syn$|_ack$|_rst$|_fin$|"
    r"_query$|_queries$|_response$|_responses$|_req$|_reply$|_unique$|"
    r"_count$|_events$|_present$|_obs_present$|_rcode\d+_.*$)",
    re.IGNORECASE,
)


def _is_countlike(col: str) -> bool:
    return bool(COUNTLIKE_RX.search(str(col)))


def _postprocess_protocol_col(col: str, x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.nan
    finite = np.isfinite(arr)
    arr[finite] = np.maximum(arr[finite], 0.0)
    if _is_countlike(col):
        arr[finite] = np.rint(arr[finite])
    return arr.astype(np.float32, copy=False)


def _load_parquet_required(path: str, label: str) -> pd.DataFrame:
    row_meta = _parquet_num_rows(path)
    if row_meta is not None and int(row_meta) != N_TEST:
        raise RuntimeError(
            f"[Cell10.9] {label} row-count metadata mismatch: {row_meta} vs df_te={N_TEST}"
        )

    df = pd.read_parquet(path)

    if len(df) != N_TEST:
        raise RuntimeError(f"[Cell10.9] {label} row-count mismatch: {len(df)} vs df_te={N_TEST}")

    return df


def _load_candidate_test_column(path: str, col: str) -> pd.Series:
    if not path or not os.path.exists(str(path)):
        raise RuntimeError(f"[Cell10.9] Missing TEST candidate path for selected column {col}: {path}")

    path = str(path)

    row_meta = _parquet_num_rows(path)
    if row_meta is not None and int(row_meta) != N_TEST:
        raise RuntimeError(
            f"[Cell10.9] Selected TEST candidate row-count mismatch for {col}: "
            f"path={path} rows={row_meta} expected={N_TEST}"
        )

    cols_meta = _parquet_columns(path)
    if cols_meta is not None and col not in set(map(str, cols_meta)):
        raise RuntimeError(
            f"[Cell10.9] Selected TEST candidate path does not contain required column {col}: {path}"
        )

    try:
        s = pd.read_parquet(path, columns=[col])[col]
    except Exception as e:
        raise RuntimeError(
            f"[Cell10.9] Could not read selected TEST candidate column {col} from {path}: "
            f"{type(e).__name__}: {e}"
        )

    if len(s) != N_TEST:
        raise RuntimeError(
            f"[Cell10.9] Selected TEST candidate column length mismatch for {col}: "
            f"{len(s)} vs {N_TEST}"
        )

    return s


def _candidate_registry_lookup(candidate_registry):
    out = {}

    if not isinstance(candidate_registry, dict):
        return out

    # v3 selector registry format.
    for key in ["candidate_files", "candidates", "generators"]:
        items = candidate_registry.get(key, {})
        if isinstance(items, dict):
            for uid, meta in items.items():
                if isinstance(meta, dict):
                    out[str(uid)] = dict(meta)

    return out


def _selector_for_col(col: str, selected_by_col: dict, selected_rich_by_col: dict) -> dict:
    if col not in selected_by_col:
        raise RuntimeError(f"[Cell10.9] Missing selection for protocol column: {col}")

    raw = selected_by_col[col]
    rich = selected_rich_by_col.get(col, {}) if isinstance(selected_rich_by_col, dict) else {}

    if isinstance(raw, dict):
        sel = dict(raw)
    else:
        # Backward-compatible direct registry fallback.
        sel = {"selected_generator": str(raw)}

    if isinstance(rich, dict):
        for k, v in rich.items():
            if k not in sel or sel.get(k) in [None, "", [], {}]:
                sel[k] = v

    return sel


def _resolve_selected_test_path(col: str, sel: dict, candidate_lookup: dict) -> tuple:
    gen = str(sel.get("selected_generator", ""))
    cid = str(sel.get("selected_candidate_id", ""))
    uid = str(sel.get("selected_candidate_uid", ""))

    test_path = sel.get("test_path", None)

    registry_meta = None
    if uid and uid in candidate_lookup:
        registry_meta = candidate_lookup[uid]
    else:
        # Fallback by generator/candidate_id if uid was absent in a direct registry.
        for uid2, meta2 in candidate_lookup.items():
            if str(meta2.get("generator", "")) == gen and str(meta2.get("candidate_id", "")) == cid:
                registry_meta = meta2
                uid = uid2
                break

    if registry_meta is None:
        raise RuntimeError(
            f"[Cell10.9] Selected non-A1 column is not present in current Cell10.8 candidate registry: "
            f"col={col}, generator={gen}, candidate_id={cid}, uid={uid}"
        )

    reg_test_path = registry_meta.get("test_path", registry_meta.get("TEST_path", None))

    if not test_path:
        test_path = reg_test_path

    if not test_path:
        raise RuntimeError(
            f"[Cell10.9] Selected non-A1 column has no TEST candidate path: "
            f"col={col}, generator={gen}, candidate_id={cid}, uid={uid}"
        )

    if reg_test_path and _realpath(test_path) != _realpath(reg_test_path):
        raise RuntimeError(
            f"[Cell10.9] Selected TEST path disagrees with current registry for {col}: "
            f"selected={test_path} registry={reg_test_path}"
        )

    return str(test_path), str(uid), registry_meta


def _validate_variant_frame(df_variant: pd.DataFrame, variant_name: str, proto_cols: list) -> list:
    audit_rows = []

    missing_cols = [c for c in proto_cols if c not in df_variant.columns]
    if missing_cols:
        raise RuntimeError(
            f"[Cell10.9] {variant_name} missing protocol columns: {missing_cols[:20]} "
            f"(n={len(missing_cols)})"
        )

    if len(df_variant) != N_TEST:
        raise RuntimeError(
            f"[Cell10.9] {variant_name} row-count mismatch: {len(df_variant)} vs {N_TEST}"
        )

    for c in proto_cols:
        x = pd.to_numeric(df_variant[c], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite_total = int(np.isfinite(x).sum())

        audit_rows.append({
            "variant": variant_name,
            "col": c,
            "tier": _tier_of(c),
            "rows": int(len(df_variant)),
            "finite_total": int(finite_total),
            "nan_total": int(len(x) - finite_total),
            "min": float(np.nanmin(x)) if finite_total else np.nan,
            "max": float(np.nanmax(x)) if finite_total else np.nan,
            "mean": float(np.nanmean(x)) if finite_total else np.nan,
        })

    return audit_rows


# ----------------------------------------------------------
# 3) Load and validate selection artifacts
# ----------------------------------------------------------
selected_by_col = _read_json(selection_json_path)
selected_rich_by_col = _read_json(selection_rich_json_path) if os.path.exists(selection_rich_json_path) else {}
selection_summary = _read_json(selection_summary_path)
candidate_registry_v3 = _read_json(candidate_registry_v3_path)
selector_contract = _read_json(selector_contract_path)
candidate_lookup = _candidate_registry_lookup(candidate_registry_v3)

if not isinstance(selected_by_col, dict):
    raise RuntimeError("[Cell10.9] selected-generator registry must be a dict.")
if not isinstance(selection_summary, dict):
    raise RuntimeError("[Cell10.9] selection summary must be a dict.")
if not isinstance(candidate_lookup, dict) or not candidate_lookup:
    raise RuntimeError("[Cell10.9] Candidate registry lookup is empty; cannot safely materialize A2.")

# Selector purity checks.
if bool(selection_summary.get("test_used_for_selection", True)):
    raise RuntimeError("[Cell10.9] Refusing to materialize: Cell 10.8 summary says TEST was used for selection.")
if bool(selection_summary.get("test_values_read", True)):
    raise RuntimeError("[Cell10.9] Refusing to materialize: Cell 10.8 summary says TEST values were read.")
if bool(selection_summary.get("test_rescue_materialization", False)):
    raise RuntimeError("[Cell10.9] Refusing to materialize: Cell 10.8 summary indicates TEST rescue materialization.")
if int(selection_summary.get("needs_test_materialization", 0) or 0) != 0:
    raise RuntimeError("[Cell10.9] Refusing downstream TEST materialization in canonical v2.1 pure materializer.")

if str(selection_summary.get("selection_split", "")).upper() != "VAL":
    raise RuntimeError(
        f"[Cell10.9] Expected Cell 10.8 selection_split='VAL', got "
        f"{selection_summary.get('selection_split')!r}"
    )

if bool(selection_summary.get("controlled_variants", {}).get("A0", {}).get("used_for_selection", True)):
    raise RuntimeError("[Cell10.9] A0 must be diagnostic only and not used for A2 selection.")

if str(selection_summary.get("selection_baseline", "")) not in {
    "A1_VAL_baseline",
    "A1",
    "A1_VAL",
}:
    raise RuntimeError(
        "[Cell10.9] Unexpected selection baseline. Expected A1_VAL_baseline/A1/A1_VAL, "
        f"got {selection_summary.get('selection_baseline')!r}"
    )

# v3.6 selector contract checks. Use fail-closed for known purity fields.
for key in [
    "test_values_read",
    "test_used_for_fitting",
    "test_used_for_thresholding",
    "test_used_for_selection",
    "test_rescue_materialization",
    "in_selector_candidate_generation",
]:
    if bool(selector_contract.get(key, False)):
        raise RuntimeError(f"[Cell10.9] Selector contract purity violation: {key}=True")

if bool(selector_contract.get("a0_used_for_selection", False)):
    raise RuntimeError("[Cell10.9] Selector contract says A0 was used for selection.")

if selector_contract.get("unregistered_file_scan", False):
    raise RuntimeError("[Cell10.9] Selector contract indicates unregistered file scan was enabled.")

log(
    "[Cell10.9] Loaded Cell10.8 selection artifacts | "
    f"selected_cols={len(selected_by_col)} | "
    f"summary_version={selection_summary.get('version', 'unknown')} | "
    f"selector_contract={selector_contract_path}"
)


# ----------------------------------------------------------
# 4) Resolve protocol columns
# ----------------------------------------------------------
proto_cols = [str(c) for c in list(PROTO_VALUE_COLS)]
selected_cols = list(map(str, selected_by_col.keys()))

missing_selected = sorted(set(proto_cols) - set(selected_cols))
extra_selected = sorted(set(selected_cols) - set(proto_cols))

if missing_selected or extra_selected:
    raise RuntimeError(
        "[Cell10.9] selected_by_col is not a complete partition of PROTO_VALUE_COLS. "
        f"missing={missing_selected[:20]} extra={extra_selected[:20]}"
    )

# Preserve selector registry order where possible.
proto_cols = selected_cols


# ----------------------------------------------------------
# 5) Load A0/A1 TEST baselines
# ----------------------------------------------------------
A0_TEST = _load_parquet_required(a0_test_path, "A0_TEST_baseline")
A1_TEST = _load_parquet_required(a1_test_path, "A1_TEST_baseline")

for label, df_base in [("A0_TEST", A0_TEST), ("A1_TEST", A1_TEST)]:
    missing = [c for c in proto_cols if c not in df_base.columns]
    if missing:
        raise RuntimeError(f"[Cell10.9] {label} missing protocol columns: {missing[:20]} n={len(missing)}")

A0_TEST = A0_TEST.loc[:, proto_cols].copy()
A1_TEST = A1_TEST.loc[:, proto_cols].copy()

log(
    "[Cell10.9] Loaded baselines | "
    f"A0_TEST={A0_TEST.shape} | A1_TEST={A1_TEST.shape} | proto_cols={len(proto_cols)}"
)


# ----------------------------------------------------------
# 6) Build A2 TEST from VAL-locked selected registry
# ----------------------------------------------------------
A2_TEST = A1_TEST.copy()

materialization_rows = []
selected_counts = Counter()
non_a1_selected_cols = []
missing_test_materialization_cols = []

for c in proto_cols:
    sel = _selector_for_col(c, selected_by_col, selected_rich_by_col)

    gen = str(sel.get("selected_generator", ""))
    cid = str(sel.get("selected_candidate_id", ""))
    uid = str(sel.get("selected_candidate_uid", ""))

    selected_counts[gen] += 1

    if gen == "A1_temporal_block_bootstrap":
        materialization_rows.append({
            "col": c,
            "tier": _tier_of(c),
            "selected_generator": gen,
            "selected_candidate_id": cid,
            "selected_candidate_uid": uid,
            "materialized_from": "A1_TEST_baseline",
            "test_path": a1_test_path,
            "requires_downstream_test_materialization": False,
            "status": "ok_a1_default",
            "test_values_used": False,
            "registry_backed": True,
        })
        continue

    non_a1_selected_cols.append(c)

    requires_downstream = bool(sel.get("requires_downstream_test_materialization", False))

    if requires_downstream:
        missing_test_materialization_cols.append({
            "col": c,
            "selected_generator": gen,
            "selected_candidate_id": cid,
            "selected_candidate_uid": uid,
            "test_path": sel.get("test_path", None),
            "requires_downstream_test_materialization": requires_downstream,
            "error_type": "DownstreamMaterializationDisabled",
            "error_message": (
                "Canonical Cell 10.9 v2.1 is a pure materializer. "
                "Selected non-A1 columns must provide registry-backed synthetic TEST candidate paths."
            ),
        })
        continue

    try:
        test_path, resolved_uid, registry_meta = _resolve_selected_test_path(c, sel, candidate_lookup)
        s = _load_candidate_test_column(test_path, c)
    except Exception as e:
        missing_test_materialization_cols.append({
            "col": c,
            "selected_generator": gen,
            "selected_candidate_id": cid,
            "selected_candidate_uid": uid,
            "test_path": sel.get("test_path", None),
            "requires_downstream_test_materialization": requires_downstream,
            "error_type": type(e).__name__,
            "error_message": str(e),
        })
        continue

    A2_TEST[c] = _postprocess_protocol_col(c, pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32))

    materialization_rows.append({
        "col": c,
        "tier": _tier_of(c),
        "selected_generator": gen,
        "selected_candidate_id": cid,
        "selected_candidate_uid": resolved_uid,
        "materialized_from": "selected_registry_backed_TEST_candidate_path",
        "test_path": str(test_path),
        "requires_downstream_test_materialization": False,
        "status": "ok_selected_non_a1",
        "test_values_used": False,
        "registry_backed": True,
    })

if missing_test_materialization_cols:
    missing_path = os.path.join(REPDIR, "cell10_9_missing_test_materialization_required.csv")
    pd.DataFrame(missing_test_materialization_cols).to_csv(missing_path, index=False)

    raise RuntimeError(
        "[Cell10.9] Some selected non-A1 columns could not be materialized from registry-backed "
        "synthetic TEST candidate paths. This is fail-closed. "
        f"See: {missing_path}"
    )

if not non_a1_selected_cols:
    log("[Cell10.9] All protocol columns selected A1; A2_TEST is exactly A1_TEST by VAL-locked selection.")


# ----------------------------------------------------------
# 7) Validate and save variants
# ----------------------------------------------------------
a0_out_path = os.path.join(SYNDIR, "A0_PROTOCOL_TEST.parquet")
a1_out_path = os.path.join(SYNDIR, "A1_PROTOCOL_TEST.parquet")
a2_out_path = os.path.join(SYNDIR, "A2_PROTOCOL_TEST.parquet")

A0_TEST.to_parquet(a0_out_path, index=False)
A1_TEST.to_parquet(a1_out_path, index=False)
A2_TEST.to_parquet(a2_out_path, index=False)

audit_rows = []
audit_rows.extend(_validate_variant_frame(A0_TEST, "A0_PROTOCOL_TEST", proto_cols))
audit_rows.extend(_validate_variant_frame(A1_TEST, "A1_PROTOCOL_TEST", proto_cols))
audit_rows.extend(_validate_variant_frame(A2_TEST, "A2_PROTOCOL_TEST", proto_cols))

variant_audit = pd.DataFrame(audit_rows)

mat_by_col = {
    r["col"]: r
    for r in materialization_rows
}

for k in [
    "selected_generator",
    "selected_candidate_id",
    "selected_candidate_uid",
    "materialized_from",
    "test_path",
    "status",
    "requires_downstream_test_materialization",
    "test_values_used",
    "registry_backed",
]:
    variant_audit[k] = None

for idx, row in variant_audit.iterrows():
    if row["variant"] != "A2_PROTOCOL_TEST":
        continue

    c = str(row["col"])
    meta = mat_by_col.get(c, {})
    for k in [
        "selected_generator",
        "selected_candidate_id",
        "selected_candidate_uid",
        "materialized_from",
        "test_path",
        "status",
        "requires_downstream_test_materialization",
        "test_values_used",
        "registry_backed",
    ]:
        variant_audit.loc[idx, k] = meta.get(k, None)

variant_audit_path = os.path.join(REPDIR, "cell10_9_variant_materialization_audit.csv")
variant_audit.to_csv(variant_audit_path, index=False)

materialization_audit_path = os.path.join(REPDIR, "cell10_9_a2_materialization_by_col.csv")
materialization_audit = pd.DataFrame(materialization_rows)
materialization_audit.to_csv(materialization_audit_path, index=False)

# Kept for backward compatibility with downstream cells; intentionally empty in v2.1.
downstream_rescue_audit_path = os.path.join(REPDIR, "cell10_9_downstream_markov_rescue_audit.csv")
downstream_rescue_audit = pd.DataFrame(columns=[
    "col",
    "selected_generator",
    "selected_candidate_id",
    "materializer",
    "status",
])
downstream_rescue_audit.to_csv(downstream_rescue_audit_path, index=False)


# ----------------------------------------------------------
# 8) Identity checks
# ----------------------------------------------------------
all_a1_selected = bool(len(non_a1_selected_cols) == 0)

a2_equals_a1 = bool(A2_TEST.equals(A1_TEST))
a2_equals_a0 = bool(A2_TEST.equals(A0_TEST))

if all_a1_selected and not a2_equals_a1:
    raise RuntimeError("[Cell10.9] Internal error: all columns selected A1 but A2_TEST != A1_TEST.")

if not all_a1_selected and a2_equals_a1:
    raise RuntimeError(
        "[Cell10.9] Non-A1 columns were selected, but A2_TEST equals A1_TEST. "
        "This suggests selected candidates were not materialized correctly."
    )


# ----------------------------------------------------------
# 9) Manifest and contract
# ----------------------------------------------------------
manifest_path = os.path.join(ARTDIR, "cell10_9_protocol_variant_manifest.json")
contract_path = os.path.join(CONTRACT_DIR, "cell10_9_protocol_variant_materialization_contract_v2_1_THESIS.json")

manifest = {
    "version": "cell10_9_protocol_variant_materialization_v2_1_THESIS",
    "rows_test": int(N_TEST),
    "proto_cols_n": int(len(proto_cols)),
    "proto_cols": list(proto_cols),

    "variant_outputs": {
        "A0_PROTOCOL_TEST": a0_out_path,
        "A1_PROTOCOL_TEST": a1_out_path,
        "A2_PROTOCOL_TEST": a2_out_path,
    },

    "source_inputs": {
        "A0_TEST_baseline": a0_test_path,
        "A1_TEST_baseline": a1_test_path,
        "selection_json": selection_json_path,
        "selection_rich_json": selection_rich_json_path if os.path.exists(selection_rich_json_path) else None,
        "selection_summary": selection_summary_path,
        "candidate_registry_v3": candidate_registry_v3_path,
        "cell10_8_selector_contract": selector_contract_path,
        "baseline_manifest": baseline_manifest_path if os.path.exists(baseline_manifest_path) else None,
    },

    "controlled_variant_contract": {
        "A0_role": "diagnostic_control_real_TEST_masks_as_materialized_by_Cell10_1",
        "A1_role": "synthetic_mask_baseline",
        "A2_role": "VAL_locked_selected_generator_portfolio",
        "A0_used_for_selection": False,
        "A1_used_for_A2_selection_baseline": True,
        "A2_selection_split": "VAL",
        "test_values_read": False,
        "test_metrics_computed": False,
        "test_usage": "materialization_and_downstream_final_QA_only",
        "downstream_test_rescue_materialization": False,
        "registry_backed_non_a1_materialization_only": True,
    },

    "selection_summary": {
        "selection_split": selection_summary.get("selection_split"),
        "selection_baseline": selection_summary.get("selection_baseline"),
        "selected_counts_from_summary": selection_summary.get("selected_counts"),
        "selected_counts_recomputed": dict(selected_counts),
        "selected_non_A1_count_from_summary": selection_summary.get("selected_non_A1_count"),
        "selected_non_A1_count_recomputed": int(len(non_a1_selected_cols)),
        "needs_test_materialization_from_summary": int(selection_summary.get("needs_test_materialization", 0) or 0),
        "all_optional_generators_rejected_by_VAL": bool(selection_summary.get("all_optional_generators_rejected_by_VAL", False)),
        "all_A1_selection_is_valid": bool(selection_summary.get("all_A1_selection_is_valid", False)),
    },

    "materialization": {
        "all_a1_selected": bool(all_a1_selected),
        "non_a1_selected_cols": list(non_a1_selected_cols),
        "non_a1_selected_cols_n": int(len(non_a1_selected_cols)),
        "selected_counts": dict(selected_counts),
        "selected_non_a1_materialized_from_test_path_n": int(
            sum(1 for r in materialization_rows if r.get("status") == "ok_selected_non_a1")
        ),
        "selected_non_a1_materialized_from_downstream_markov_rescue_n": 0,
        "a2_equals_a1": bool(a2_equals_a1),
        "a2_equals_a0": bool(a2_equals_a0),
        "missing_test_materialization_cols_n": 0,
        "materialization_audit_path": materialization_audit_path,
        "downstream_markov_rescue_audit_path": downstream_rescue_audit_path,
        "variant_audit_path": variant_audit_path,
    },

    "artifact_hashes": {
        "A0_PROTOCOL_TEST_sha256": _sha256_file(a0_out_path),
        "A1_PROTOCOL_TEST_sha256": _sha256_file(a1_out_path),
        "A2_PROTOCOL_TEST_sha256": _sha256_file(a2_out_path),
        "variant_audit_sha256": _sha256_file(variant_audit_path),
        "materialization_audit_sha256": _sha256_file(materialization_audit_path),
        "downstream_markov_rescue_audit_sha256": _sha256_file(downstream_rescue_audit_path),
        "selected_by_col_sha256": _sha256_jsonable(selected_by_col),
    },

    "methodological_note": (
        "Cell 10.9 v2.1 materializes protocol-level A0/A1/A2 TEST variants from artifacts "
        "created by Cells 10.1 and 10.8. It does not read real TEST target values, does not "
        "compute TEST metrics, and does not perform generator selection. A0 and A1 are loaded "
        "from controlled baselines. A2 starts from A1 and replaces only VAL-selected non-A1 "
        "columns using registry-backed synthetic TEST candidate parquet files. Missing TEST "
        "paths or downstream materialization requirements fail closed."
    ),
}

contract = {
    "version": "cell10_9_protocol_variant_materialization_contract_v2_1_THESIS",
    "component": "protocol_variant_materializer",
    "rows_test": int(N_TEST),
    "proto_cols_n": int(len(proto_cols)),
    "inputs": manifest["source_inputs"],
    "outputs": manifest["variant_outputs"],
    "selection_split": "VAL",
    "selection_baseline": "A1_VAL_baseline",
    "a0_used_for_selection": False,
    "test_values_read": False,
    "test_metrics_computed": False,
    "test_used_for_fitting": False,
    "test_used_for_thresholding": False,
    "test_used_for_selection": False,
    "test_used_for_repair": False,
    "downstream_test_rescue_materialization": False,
    "registry_backed_non_a1_materialization_only": True,
    "selected_counts": dict(selected_counts),
    "non_a1_selected_cols_n": int(len(non_a1_selected_cols)),
    "non_a1_selected_cols": list(non_a1_selected_cols),
    "a2_equals_a1": bool(a2_equals_a1),
    "a2_equals_a0": bool(a2_equals_a0),
    "manifest_path": manifest_path,
    "variant_audit_path": variant_audit_path,
    "materialization_audit_path": materialization_audit_path,
    "paper_claim_status": (
        "Cell 10.9 materializes VAL-locked protocol variants only. It is not a TEST QA cell "
        "and does not provide final realism evidence."
    ),
}

_write_json(manifest_path, manifest)
_write_json(contract_path, contract)


# ----------------------------------------------------------
# 10) Publish globals
# ----------------------------------------------------------
globals()["CELL10_A0_PROTOCOL_TEST"] = A0_TEST
globals()["CELL10_A1_PROTOCOL_TEST"] = A1_TEST
globals()["CELL10_A2_PROTOCOL_TEST"] = A2_TEST

globals()["CELL10_A0_PROTOCOL_TEST_PATH"] = a0_out_path
globals()["CELL10_A1_PROTOCOL_TEST_PATH"] = a1_out_path
globals()["CELL10_A2_PROTOCOL_TEST_PATH"] = a2_out_path

globals()["CELL10_9_PROTOCOL_VARIANT_MANIFEST"] = manifest
globals()["CELL10_9_PROTOCOL_VARIANT_CONTRACT"] = contract
globals()["CELL10_9_PROTOCOL_VARIANT_CONTRACT_PATH"] = contract_path

globals()["CELL10_9_VARIANT_AUDIT"] = variant_audit
globals()["CELL10_9_A2_MATERIALIZATION_AUDIT"] = materialization_audit
globals()["CELL10_9_DOWNSTREAM_MARKOV_RESCUE_AUDIT"] = downstream_rescue_audit

globals()["CELL10_SELECTED_PROTOCOL_TEST"] = A2_TEST
globals()["CELL10_SELECTED_PROTOCOL_TEST_PATH"] = a2_out_path

if "RUN_META" in globals():
    RUN_META.setdefault("protocol_variant_materialization", {})
    RUN_META["protocol_variant_materialization"]["cell10_9"] = contract


# ----------------------------------------------------------
# 11) Logs
# ----------------------------------------------------------
log(f"[Cell10.9] Saved A0_PROTOCOL_TEST: {a0_out_path} | shape={A0_TEST.shape}")
log(f"[Cell10.9] Saved A1_PROTOCOL_TEST: {a1_out_path} | shape={A1_TEST.shape}")
log(f"[Cell10.9] Saved A2_PROTOCOL_TEST: {a2_out_path} | shape={A2_TEST.shape}")
log(f"[Cell10.9] Saved variant audit: {variant_audit_path}")
log(f"[Cell10.9] Saved A2 materialization audit: {materialization_audit_path}")
log(f"[Cell10.9] Saved downstream Markov rescue audit: {downstream_rescue_audit_path}")
log(f"[Cell10.9] Saved manifest: {manifest_path}")
log(f"[Cell10.9] Saved contract: {contract_path}")

log(
    "[Cell10.9] Materialization summary | "
    f"selected_counts={dict(selected_counts)} | "
    f"all_a1_selected={all_a1_selected} | "
    f"non_A1_cols={len(non_a1_selected_cols)} | "
    f"A2_equals_A1={a2_equals_a1} | "
    f"A2_equals_A0={a2_equals_a0} | "
    f"downstream_markov_rescue_n=0"
)

log("--- END: Cell 10.9 — Materialize A0/A1/A2 protocol variants (v2.1-THESIS pure materializer) ---")
