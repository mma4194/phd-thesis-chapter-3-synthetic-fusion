# ==========================================================
# PRE-CELL11 — A0/A1/A2 protocol variant runtime guard — v4.1-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Validate that Cells 10.1, 10.8, and 10.9 completed correctly.
# - Lock the A0/A1/A2 protocol-variant contract before Cell 11.
# - Ensure Cell 11 consumes materialized protocol variants rather than
#   re-selecting generators, re-running DDPM, generating protocol candidates,
#   or applying TEST-informed changes.
#
# Scientific contract:
# - Cell 10.8 is the only protocol generator-selection authority.
# - Cell 10.8 selection is VAL-only.
# - Cell 10.9 is the only A0/A1/A2 protocol-variant materialization authority.
# - Cell 10.9 must be the v2.1 pure materializer.
# - TEST is final QA only.
# - This guard does not compute TEST-vs-real metrics.
# - Cell 11 must only consume materialized controlled variants and perform
#   assembly/evaluation.
# ==========================================================

log("--- START: PRE-CELL11 — A0/A1/A2 protocol variant runtime guard (v4.1-THESIS) ---")

import os
import json
import hashlib
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
required_globals = [
    "CFG",
    "log",
    "df_te",
    "PROTO_VALUE_COLS",
]

missing = [k for k in required_globals if k not in globals()]
if missing:
    raise RuntimeError(
        f"[PRE-CELL11] Missing required globals: {missing}. "
        "Run Cells 1–10.9 first."
    )

# Clean-run leakage guard.
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[PRE-CELL11] Clean runtime guard forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[PRE-CELL11] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
REPDIR = os.path.join(OUTDIR, "reports")
SYNDIR = os.path.join(OUTDIR, "synthetic")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

for d in [ARTDIR, REPDIR, SYNDIR, CONTRACT_DIR]:
    os.makedirs(d, exist_ok=True)

N_TEST = int(len(df_te))
if N_TEST <= 0:
    raise RuntimeError("[PRE-CELL11] df_te is empty.")

PROTO_COLS = [str(c) for c in list(PROTO_VALUE_COLS)]
if not PROTO_COLS:
    raise RuntimeError("[PRE-CELL11] PROTO_VALUE_COLS is empty.")

# ----------------------------------------------------------
# 1) Canonical artifact paths
# ----------------------------------------------------------
cell10_1_manifest_path = os.path.join(ARTDIR, "cell10_a0_a1_baseline_manifest.json")

cell10_8_selection_summary_path = os.path.join(ARTDIR, "cell10_portfolio_selection_summary.json")
cell10_8_selection_path = os.path.join(ARTDIR, "cell10_selected_generator_by_col.json")
cell10_8_selector_contract_path = os.path.join(
    CONTRACT_DIR,
    "cell10_8_val_only_selector_contract_v3_6_THESIS.json",
)

cell10_9_manifest_path = os.path.join(ARTDIR, "cell10_9_protocol_variant_manifest.json")
cell10_9_contract_path = os.path.join(
    CONTRACT_DIR,
    "cell10_9_protocol_variant_materialization_contract_v2_1_THESIS.json",
)

a0_protocol_test_path = os.path.join(SYNDIR, "A0_PROTOCOL_TEST.parquet")
a1_protocol_test_path = os.path.join(SYNDIR, "A1_PROTOCOL_TEST.parquet")
a2_protocol_test_path = os.path.join(SYNDIR, "A2_PROTOCOL_TEST.parquet")

required_paths = {
    "cell10_1_manifest": cell10_1_manifest_path,
    "cell10_8_selection_summary": cell10_8_selection_summary_path,
    "cell10_8_selection": cell10_8_selection_path,
    "cell10_8_selector_contract_v3_6": cell10_8_selector_contract_path,
    "cell10_9_manifest": cell10_9_manifest_path,
    "cell10_9_contract_v2_1": cell10_9_contract_path,
    "A0_PROTOCOL_TEST": a0_protocol_test_path,
    "A1_PROTOCOL_TEST": a1_protocol_test_path,
    "A2_PROTOCOL_TEST": a2_protocol_test_path,
}

missing_paths = {k: p for k, p in required_paths.items() if not os.path.exists(p)}
if missing_paths:
    raise RuntimeError(
        "[PRE-CELL11] Missing required Cell 10 artifacts/contracts. "
        f"missing={missing_paths}. Run Cells 10.1, 10.8 v3.6, and 10.9 v2.1 first."
    )

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _read_json_dict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"[PRE-CELL11] Expected JSON dict at {path}, got {type(obj).__name__}")
    return obj


def _sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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


def _realpath(path: str) -> str:
    return os.path.realpath(os.path.abspath(str(path)))


def _parquet_num_rows(path: str):
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _parquet_columns(path: str):
    try:
        import pyarrow.parquet as pq
        return [str(c) for c in pq.ParquetFile(path).schema.names]
    except Exception:
        return None


def _load_protocol_variant(path: str, label: str) -> pd.DataFrame:
    row_meta = _parquet_num_rows(path)
    if row_meta is not None and int(row_meta) != N_TEST:
        raise RuntimeError(
            f"[PRE-CELL11] {label} row metadata mismatch: got={row_meta} expected={N_TEST}"
        )

    cols_meta = _parquet_columns(path)
    if cols_meta is None:
        df = pd.read_parquet(path)
    else:
        read_cols = [c for c in PROTO_COLS if c in set(cols_meta)]
        if not read_cols:
            raise RuntimeError(f"[PRE-CELL11] {label} contains no protocol columns: {path}")
        df = pd.read_parquet(path, columns=read_cols)

    if len(df) != N_TEST:
        raise RuntimeError(
            f"[PRE-CELL11] {label} row mismatch: got={len(df)} expected={N_TEST}"
        )

    missing_cols = [c for c in PROTO_COLS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"[PRE-CELL11] {label} missing protocol columns: {missing_cols[:20]} "
            f"(n={len(missing_cols)})"
        )

    return df.loc[:, PROTO_COLS].copy()


def _frame_basic_audit(df: pd.DataFrame, label: str) -> dict:
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(arr)

    return {
        "label": label,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "finite_rate": float(finite.mean()) if arr.size else 0.0,
        "nan_rate": float(np.isnan(arr).mean()) if arr.size else 0.0,
        "negative_finite_count": int(np.sum(arr[finite] < 0.0)) if finite.any() else 0,
        "columns": list(df.columns),
    }


def _variant_outputs_from_manifest(manifest: dict) -> dict:
    # v2.1 Cell 10.9 uses "variant_outputs"; older versions sometimes used "outputs".
    if isinstance(manifest.get("variant_outputs"), dict):
        return dict(manifest["variant_outputs"])
    if isinstance(manifest.get("outputs"), dict):
        return dict(manifest["outputs"])
    return {}


# ----------------------------------------------------------
# 3) Load manifests/contracts/selections
# ----------------------------------------------------------
cell10_1_manifest = _read_json_dict(cell10_1_manifest_path)
cell10_8_summary = _read_json_dict(cell10_8_selection_summary_path)
cell10_8_selection = _read_json_dict(cell10_8_selection_path)
cell10_8_selector_contract = _read_json_dict(cell10_8_selector_contract_path)
cell10_9_manifest = _read_json_dict(cell10_9_manifest_path)
cell10_9_contract = _read_json_dict(cell10_9_contract_path)

# ----------------------------------------------------------
# 4) Contract checks: selection/materialization authority
# ----------------------------------------------------------
# Cell 10.8 summary checks.
if bool(cell10_8_summary.get("test_used_for_selection", True)) is not False:
    raise RuntimeError("[PRE-CELL11] Cell 10.8 summary says TEST was used for selection.")

if bool(cell10_8_summary.get("test_values_read", True)) is not False:
    raise RuntimeError("[PRE-CELL11] Cell 10.8 summary says TEST values were read.")

if str(cell10_8_summary.get("selection_split", "")).upper() != "VAL":
    raise RuntimeError(
        f"[PRE-CELL11] Cell 10.8 selection split must be VAL. "
        f"Got {cell10_8_summary.get('selection_split')!r}"
    )

if int(cell10_8_summary.get("needs_test_materialization", 0) or 0) != 0:
    raise RuntimeError("[PRE-CELL11] Cell 10.8 selected candidates requiring downstream TEST materialization.")

# Cell 10.8 v3.6 selector contract checks.
selector_forbidden_true = [
    "test_values_read",
    "test_used_for_fitting",
    "test_used_for_thresholding",
    "test_used_for_selection",
    "test_rescue_materialization",
    "in_selector_candidate_generation",
    "a0_used_for_selection",
    "unregistered_file_scan",
]
for key in selector_forbidden_true:
    if bool(cell10_8_selector_contract.get(key, False)):
        raise RuntimeError(f"[PRE-CELL11] Cell 10.8 selector contract violation: {key}=True")

if str(cell10_8_selector_contract.get("selection_baseline", "")) not in {
    "A1_VAL_baseline",
    "A1",
    "A1_VAL",
}:
    raise RuntimeError(
        "[PRE-CELL11] Unexpected selector baseline in Cell 10.8 contract: "
        f"{cell10_8_selector_contract.get('selection_baseline')!r}"
    )

# Cell 10.9 v2.1 materialization contract checks.
materializer_forbidden_true = [
    "test_values_read",
    "test_metrics_computed",
    "test_used_for_fitting",
    "test_used_for_thresholding",
    "test_used_for_selection",
    "test_used_for_repair",
    "a0_used_for_selection",
    "downstream_test_rescue_materialization",
]
for key in materializer_forbidden_true:
    if bool(cell10_9_contract.get(key, False)):
        raise RuntimeError(f"[PRE-CELL11] Cell 10.9 materializer contract violation: {key}=True")

if not bool(cell10_9_contract.get("registry_backed_non_a1_materialization_only", False)):
    raise RuntimeError(
        "[PRE-CELL11] Cell 10.9 contract must require registry-backed non-A1 materialization only."
    )

if str(cell10_9_contract.get("selection_split", "")).upper() != "VAL":
    raise RuntimeError(
        f"[PRE-CELL11] Cell 10.9 contract selection_split must be VAL. "
        f"Got {cell10_9_contract.get('selection_split')!r}"
    )

if set(cell10_8_selection.keys()) != set(PROTO_COLS):
    missing_sel = sorted(set(PROTO_COLS) - set(cell10_8_selection.keys()))
    extra_sel = sorted(set(cell10_8_selection.keys()) - set(PROTO_COLS))
    raise RuntimeError(
        "[PRE-CELL11] Cell 10.8 selected-generator registry is not a complete "
        f"protocol-column partition. missing={missing_sel[:20]} extra={extra_sel[:20]}"
    )

# Validate Cell 10.9 outputs if listed in manifest/contract.
cell10_9_outputs = _variant_outputs_from_manifest(cell10_9_manifest)
if not cell10_9_outputs:
    raise RuntimeError(
        "[PRE-CELL11] Cell 10.9 manifest must expose variant outputs via 'variant_outputs' "
        "or legacy 'outputs'."
    )

expected_outputs = {
    "A0_PROTOCOL_TEST": a0_protocol_test_path,
    "A1_PROTOCOL_TEST": a1_protocol_test_path,
    "A2_PROTOCOL_TEST": a2_protocol_test_path,
}
for key, expected_path in expected_outputs.items():
    actual = cell10_9_outputs.get(key)
    if actual is None:
        raise RuntimeError(f"[PRE-CELL11] Cell 10.9 manifest missing output key: {key}")
    if _realpath(actual) != _realpath(expected_path):
        raise RuntimeError(
            f"[PRE-CELL11] Cell 10.9 manifest output path mismatch for {key}: "
            f"manifest={actual} expected={expected_path}"
        )

contract_outputs = cell10_9_contract.get("outputs", {})
if isinstance(contract_outputs, dict):
    for key, expected_path in expected_outputs.items():
        actual = contract_outputs.get(key)
        if actual is not None and _realpath(actual) != _realpath(expected_path):
            raise RuntimeError(
                f"[PRE-CELL11] Cell 10.9 contract output path mismatch for {key}: "
                f"contract={actual} expected={expected_path}"
            )

# ----------------------------------------------------------
# 5) Load materialized protocol variants
# ----------------------------------------------------------
A0_PROTOCOL_TEST = _load_protocol_variant(a0_protocol_test_path, "A0_PROTOCOL_TEST")
A1_PROTOCOL_TEST = _load_protocol_variant(a1_protocol_test_path, "A1_PROTOCOL_TEST")
A2_PROTOCOL_TEST = _load_protocol_variant(a2_protocol_test_path, "A2_PROTOCOL_TEST")

# ----------------------------------------------------------
# 6) Variant relationship checks
# ----------------------------------------------------------
selected_counts = dict(
    cell10_9_contract.get(
        "selected_counts",
        cell10_8_summary.get("selected_counts", {}),
    )
)
selected_non_A1_count = int(
    cell10_9_contract.get(
        "non_a1_selected_cols_n",
        cell10_8_summary.get("selected_non_A1_count", 0),
    )
)
all_a1_selected = bool(selected_non_A1_count == 0)

a2_equals_a1 = bool(A2_PROTOCOL_TEST.equals(A1_PROTOCOL_TEST))
a2_equals_a0 = bool(A2_PROTOCOL_TEST.equals(A0_PROTOCOL_TEST))
a0_equals_a1 = bool(A0_PROTOCOL_TEST.equals(A1_PROTOCOL_TEST))

if all_a1_selected and not a2_equals_a1:
    raise RuntimeError(
        "[PRE-CELL11] Cell 10.8 selected A1 for all protocol columns, "
        "but A2_PROTOCOL_TEST is not exactly equal to A1_PROTOCOL_TEST."
    )

if not all_a1_selected and a2_equals_a1:
    raise RuntimeError(
        "[PRE-CELL11] Cell 10.8 selected at least one non-A1 generator, "
        "but A2_PROTOCOL_TEST is exactly equal to A1_PROTOCOL_TEST."
    )

if a0_equals_a1:
    log(
        "[PRE-CELL11][WARN] A0_PROTOCOL_TEST equals A1_PROTOCOL_TEST exactly. "
        "This is not automatically invalid, but it means real TEST masks and synthetic TEST masks "
        "produced identical protocol baseline frames."
    )

# ----------------------------------------------------------
# 7) Publish Cell 11 runtime contract
# ----------------------------------------------------------
CELL11_VARIANT_RUNTIME_CONTRACT = {
    "version": "pre_cell11_a0_a1_a2_variant_runtime_contract_v4_1_THESIS",
    "cell11_role": "consume_materialized_controlled_protocol_variants_for_assembly_and_QA",
    "selection_authority": "Cell10.8_VAL_only_registry_only_selector_v3_6",
    "materialization_authority": "Cell10.9_pure_materializer_v2_1",
    "test_usage": "final_QA_only",
    "model_shopping_policy": "forbidden",
    "cell11_must_not_select_generators": True,
    "cell11_must_not_register_ddpm_for_protocol_generation": True,
    "cell11_must_not_read_test_for_selection": True,
    "cell11_must_not_materialize_protocol_candidates": True,
    "cell11_must_not_apply_test_informed_rescue_or_repair": True,
    "proto_cols_n": int(len(PROTO_COLS)),
    "N_TEST": int(N_TEST),
    "paths": {
        "A0_PROTOCOL_TEST": a0_protocol_test_path,
        "A1_PROTOCOL_TEST": a1_protocol_test_path,
        "A2_PROTOCOL_TEST": a2_protocol_test_path,
        "cell10_1_manifest": cell10_1_manifest_path,
        "cell10_8_selection_summary": cell10_8_selection_summary_path,
        "cell10_8_selection": cell10_8_selection_path,
        "cell10_8_selector_contract": cell10_8_selector_contract_path,
        "cell10_9_manifest": cell10_9_manifest_path,
        "cell10_9_contract": cell10_9_contract_path,
    },
    "cell10_9_manifest_variant_outputs": cell10_9_outputs,
    "selected_counts": selected_counts,
    "selected_non_A1_count": int(selected_non_A1_count),
    "all_a1_selected": bool(all_a1_selected),
    "variant_equalities": {
        "A2_equals_A1": bool(a2_equals_a1),
        "A2_equals_A0": bool(a2_equals_a0),
        "A0_equals_A1": bool(a0_equals_a1),
    },
    "selector_contract_summary": {
        "selection_split": cell10_8_selector_contract.get("selection_split", "VAL"),
        "selection_baseline": cell10_8_selector_contract.get("selection_baseline", None),
        "unregistered_file_scan": bool(cell10_8_selector_contract.get("unregistered_file_scan", False)),
        "in_selector_candidate_generation": bool(cell10_8_selector_contract.get("in_selector_candidate_generation", False)),
        "test_values_read": bool(cell10_8_selector_contract.get("test_values_read", False)),
        "test_used_for_selection": bool(cell10_8_selector_contract.get("test_used_for_selection", False)),
    },
    "materializer_contract_summary": {
        "selection_split": cell10_9_contract.get("selection_split", "VAL"),
        "registry_backed_non_a1_materialization_only": bool(
            cell10_9_contract.get("registry_backed_non_a1_materialization_only", False)
        ),
        "downstream_test_rescue_materialization": bool(
            cell10_9_contract.get("downstream_test_rescue_materialization", False)
        ),
        "test_values_read": bool(cell10_9_contract.get("test_values_read", False)),
        "test_metrics_computed": bool(cell10_9_contract.get("test_metrics_computed", False)),
    },
    "variant_audits": {
        "A0_PROTOCOL_TEST": _frame_basic_audit(A0_PROTOCOL_TEST, "A0_PROTOCOL_TEST"),
        "A1_PROTOCOL_TEST": _frame_basic_audit(A1_PROTOCOL_TEST, "A1_PROTOCOL_TEST"),
        "A2_PROTOCOL_TEST": _frame_basic_audit(A2_PROTOCOL_TEST, "A2_PROTOCOL_TEST"),
    },
    "artifact_hashes": {
        "A0_PROTOCOL_TEST_sha256": _sha256_file(a0_protocol_test_path),
        "A1_PROTOCOL_TEST_sha256": _sha256_file(a1_protocol_test_path),
        "A2_PROTOCOL_TEST_sha256": _sha256_file(a2_protocol_test_path),
        "cell10_1_manifest_sha256": _sha256_file(cell10_1_manifest_path),
        "cell10_8_selection_summary_sha256": _sha256_file(cell10_8_selection_summary_path),
        "cell10_8_selection_sha256": _sha256_file(cell10_8_selection_path),
        "cell10_8_selector_contract_sha256": _sha256_file(cell10_8_selector_contract_path),
        "cell10_9_manifest_sha256": _sha256_file(cell10_9_manifest_path),
        "cell10_9_contract_sha256": _sha256_file(cell10_9_contract_path),
    },
    "methodological_note": (
        "PRE-CELL11 v4.1 validates the controlled A0/A1/A2 protocol-variant contract. "
        "A0 is the diagnostic control loaded from Cell 10.1, A1 is the synthetic-mask "
        "baseline, and A2 is the VAL-selected protocol portfolio materialized by Cell 10.9. "
        "Cell 11 is not a generator-selection or candidate-materialization cell. It must "
        "consume these materialized variants and perform final assembly/evaluation only. "
        "TEST remains final QA only."
    ),
}

contract_path = os.path.join(CONTRACT_DIR, "pre_cell11_a0_a1_a2_variant_runtime_contract_v4_1_THESIS.json")
audit_path = os.path.join(REPDIR, "pre_cell11_a0_a1_a2_variant_runtime_audit_v4_1_THESIS.json")

_write_json(contract_path, CELL11_VARIANT_RUNTIME_CONTRACT)
_write_json(audit_path, CELL11_VARIANT_RUNTIME_CONTRACT)

# Legacy paths for downstream compatibility.
legacy_contract_path = os.path.join(ARTDIR, "pre_cell11_a0_a1_a2_variant_runtime_contract.json")
legacy_audit_path = os.path.join(REPDIR, "pre_cell11_a0_a1_a2_variant_runtime_audit.json")
_write_json(legacy_contract_path, CELL11_VARIANT_RUNTIME_CONTRACT)
_write_json(legacy_audit_path, CELL11_VARIANT_RUNTIME_CONTRACT)

# Publish globals for Cell 11.
globals()["CELL11_VARIANT_RUNTIME_CONTRACT"] = CELL11_VARIANT_RUNTIME_CONTRACT
globals()["CELL11_VARIANT_RUNTIME_CONTRACT_PATH"] = contract_path
globals()["PRECELL11_A0_A1_A2_VARIANT_AUDIT_PATH"] = audit_path

globals()["CELL11_A0_PROTOCOL_TEST"] = A0_PROTOCOL_TEST
globals()["CELL11_A1_PROTOCOL_TEST"] = A1_PROTOCOL_TEST
globals()["CELL11_A2_PROTOCOL_TEST"] = A2_PROTOCOL_TEST

globals()["CELL11_A0_PROTOCOL_TEST_PATH"] = a0_protocol_test_path
globals()["CELL11_A1_PROTOCOL_TEST_PATH"] = a1_protocol_test_path
globals()["CELL11_A2_PROTOCOL_TEST_PATH"] = a2_protocol_test_path

# Backward-compatible names, but now explicitly variant-based.
globals()["CELL10_A0_PROTOCOL_TEST_PATH"] = a0_protocol_test_path
globals()["CELL10_A1_PROTOCOL_TEST_PATH"] = a1_protocol_test_path
globals()["CELL10_A2_PROTOCOL_TEST_PATH"] = a2_protocol_test_path

if "RUN_META" in globals():
    RUN_META.setdefault("pre_cell11_runtime_guards", {})
    RUN_META["pre_cell11_runtime_guards"]["a0_a1_a2_protocol_variants"] = CELL11_VARIANT_RUNTIME_CONTRACT

# Strong runtime flags consumed by rewritten Cell 11.
CFG["cell11_selection_authority"] = "Cell10.8_VAL_only_registry_only_selector_v3_6"
CFG["cell11_protocol_variants_materialized_by"] = "Cell10.9_pure_materializer_v2_1"
CFG["cell11_test_usage"] = "final_QA_only"
CFG["cell11_must_not_select_generators"] = True
CFG["cell11_must_not_register_ddpm_for_protocol_generation"] = True
CFG["cell11_must_not_materialize_protocol_candidates"] = True
CFG["cell11_must_not_apply_test_informed_rescue_or_repair"] = True
CFG["cell11_use_materialized_protocol_variants"] = True

log(
    "[PRE-CELL11] A0/A1/A2 protocol variant contract locked | "
    f"A0={A0_PROTOCOL_TEST.shape} | "
    f"A1={A1_PROTOCOL_TEST.shape} | "
    f"A2={A2_PROTOCOL_TEST.shape} | "
    f"selected_counts={selected_counts} | "
    f"all_a1_selected={all_a1_selected} | "
    f"A2_equals_A1={a2_equals_a1} | "
    f"A2_equals_A0={a2_equals_a0}"
)

log(f"[PRE-CELL11] Saved runtime contract: {contract_path}")
log(f"[PRE-CELL11] Saved runtime audit: {audit_path}")
log(f"[PRE-CELL11] Saved legacy runtime contract: {legacy_contract_path}")
log(f"[PRE-CELL11] Saved legacy runtime audit: {legacy_audit_path}")
log("--- END: PRE-CELL11 — A0/A1/A2 protocol variant runtime guard (v4.1-THESIS) ---")
