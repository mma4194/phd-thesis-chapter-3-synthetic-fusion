# ==========================================================
# CELL 4 — taxonomy / obs / driver-contract — v5
#
# Purpose:
# - freeze the protocol tier contract used by downstream modeling
# - validate protocol observability masks without inferring them from values
# - freeze TRUE IoT driver bases from Cell 3 explicit registries only
# - write deterministic driver/obs contract artifacts for later cells
#
# Audit policy:
# - no silent driver reconstruction
# - no silent driver merging
# - no value-based observability inference
# - no silent fabrication of required obs masks
# - no NaN/inf/negative clipping in event drivers
# - no silent binarization of malformed masks
#
# Hardening in v5:
# - strict binary mask validation: finite and exactly 0/1-like only
# - strict driver validation: finite, non-negative, integer-like event counts
# - logical OTA obs rate uses row-wise OR, not max of marginal rates
# - Z-Wave all-zero placeholder assumption is enforced when configured
# - validates Cell 3 registry and split contracts against current globals
# - writes protocol_obs_contract.json and driver_contract.json
# ==========================================================

log("--- START: Cell 4 — taxonomy / obs / driver-contract ---")

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Sequence, Tuple

need = [
    "df_tr", "df_va", "df_te",
    "CFG", "log", "TIERS",
    "NUMERIC_COLS_PROTO", "time_col",
    "EVENT_DRIVER_COLS", "PROTO_NAMESPACE_COLS",
    "CANON_TIME_COL",
    "assert_no_duplicate_columns",
    "assert_no_modeling_forbidden_columns",
    "assert_no_abs_time_leakage_columns",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell4] Missing prerequisites: {missing}. Run Cells 1–3 first.")

ARTDIR = os.path.join(str(CFG["outdir"]), "artifacts")
os.makedirs(ARTDIR, exist_ok=True)

# ------------------------------------------------------------
# Local helpers
# ------------------------------------------------------------
def _assert_ordered_subset(subset: Sequence[str], full: Sequence[str], label: str) -> None:
    full = list(full)
    x = list(subset)

    if len(x) != len(set(x)):
        raise RuntimeError(f"[Cell4] Duplicate entries in {label}: {x}")

    for item in x:
        if item not in full:
            raise RuntimeError(f"[Cell4] Unknown {label} entry '{item}'. Allowed={full}")

    idx = [full.index(item) for item in x]
    if idx != sorted(idx):
        raise RuntimeError(f"[Cell4] {label} order mismatch. Expected order from {full}, got {x}")


def _require_same_columns(
    cols: Sequence[str],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    label: str,
) -> None:
    miss_a = [c for c in cols if c not in df_a.columns]
    miss_b = [c for c in cols if c not in df_b.columns]

    if miss_a or miss_b:
        raise RuntimeError(
            f"[Cell4] Schema drift in {label}.\n"
            f"missing_in_{name_a}={miss_a[:20]}\n"
            f"missing_in_{name_b}={miss_b[:20]}"
        )


def _is_events_in_sec_driver(c: str) -> bool:
    return isinstance(c, str) and c.startswith("events_in_sec__")


def _as_numeric_array_strict(s: pd.Series, *, col: str, where: str) -> np.ndarray:
    try:
        arr = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.float64, copy=False)
    except Exception as e:
        raise RuntimeError(f"[Cell4] {where}: column '{col}' is not strictly numeric.") from e

    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(f"[Cell4] {where}: column '{col}' contains {n_bad} NaN/inf values.")

    return arr


def _validate_binary_mask_series(s: pd.Series, *, col: str, split_name: str) -> np.ndarray:
    """
    Strict observability mask validation.

    Allowed:
    - 0 / 1
    - bool
    - float/int values numerically equal to 0 or 1 within tiny tolerance

    Forbidden:
    - NaN
    - inf
    - negative values
    - fractional mask values such as 0.5
    - counts such as 2
    """
    arr = _as_numeric_array_strict(s, col=col, where=f"{split_name}/obs_mask")

    rounded = np.rint(arr)
    max_abs_err = float(np.max(np.abs(arr - rounded))) if len(arr) else 0.0

    if max_abs_err > 1e-6:
        bad_idx = int(np.where(np.abs(arr - rounded) > 1e-6)[0][0])
        raise RuntimeError(
            f"[Cell4] {split_name}: obs mask '{col}' is not integer-like. "
            f"first_bad_idx={bad_idx} value={arr[bad_idx]} max_abs_err={max_abs_err}"
        )

    vals = np.unique(rounded.astype(np.int64))
    bad_vals = [int(v) for v in vals if int(v) not in (0, 1)]

    if bad_vals:
        raise RuntimeError(
            f"[Cell4] {split_name}: obs mask '{col}' contains non-binary values: {bad_vals[:20]}"
        )

    return rounded.astype(np.int8, copy=False)


def _validate_driver_series(s: pd.Series, *, col: str, split_name: str) -> np.ndarray:
    """
    Strict event-driver validation.

    events_in_sec__* columns are event counts per second. They must be:
    - numeric
    - finite
    - non-negative
    - integer-like

    No clipping or zero-filling is allowed here.
    """
    arr = _as_numeric_array_strict(s, col=col, where=f"{split_name}/driver")

    if np.any(arr < 0.0):
        bad_idx = int(np.where(arr < 0.0)[0][0])
        raise RuntimeError(
            f"[Cell4] {split_name}: driver '{col}' contains negative value at row {bad_idx}: {arr[bad_idx]}"
        )

    rounded = np.rint(arr)
    max_abs_err = float(np.max(np.abs(arr - rounded))) if len(arr) else 0.0

    if max_abs_err > 1e-5:
        bad_idx = int(np.where(np.abs(arr - rounded) > 1e-5)[0][0])
        raise RuntimeError(
            f"[Cell4] {split_name}: driver '{col}' is not integer-like event count. "
            f"first_bad_idx={bad_idx} value={arr[bad_idx]} max_abs_err={max_abs_err}"
        )

    return rounded.astype(np.float32, copy=False)


def _mask_rate(mask: np.ndarray) -> float:
    if len(mask) == 0:
        return 0.0
    return float(np.asarray(mask, dtype=np.float32).mean())


def _mask_transition_rate(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=np.int8)
    if len(mask) <= 1:
        return 0.0
    return float(np.mean(mask[1:] != mask[:-1]))


def _logical_or_rate(mask_dict: Dict[str, np.ndarray], cols: Sequence[str]) -> float:
    if not cols:
        return 0.0

    mats = [np.asarray(mask_dict[c], dtype=bool) for c in cols]
    if not mats:
        return 0.0

    out = mats[0].copy()
    for m in mats[1:]:
        out |= m
    return float(out.mean())


def _driver_stats(df_: pd.DataFrame, cols: Sequence[str], split_name: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}

    for c in cols:
        x = df_[c].to_numpy(dtype=np.float32, copy=False)
        out[c] = {
            "mean": float(np.mean(x)) if len(x) else 0.0,
            "std": float(np.std(x)) if len(x) else 0.0,
            "max": float(np.max(x)) if len(x) else 0.0,
            "sum": float(np.sum(x)) if len(x) else 0.0,
            "nz_rate": float(np.mean(x > 0.0)) if len(x) else 0.0,
        }

    return out


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# ------------------------------------------------------------
# Base frame validation
# ------------------------------------------------------------
for split_name, df_part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    assert_no_duplicate_columns(df_part, f"Cell4/{split_name}")

# ------------------------------------------------------------
# Config / contract knobs
# ------------------------------------------------------------
PROTOCOL_TIERS_USED = list(
    CFG.get(
        "protocol_tiers_used",
        CFG.get("expected_ddpm_tiers_used", ["router", "ota", "zigbee"]),
    )
)
_assert_ordered_subset(PROTOCOL_TIERS_USED, TIERS, "PROTOCOL_TIERS_USED")

PROTOCOL_TIERS_ALL = list(TIERS)

PROTOCOL_OBS_COLS_REQUIRED = [
    "router__obs_present",
    "ota__obs_present",
    "ota24__obs_present",
    "ota5__obs_present",
    "zigbee__obs_present",
    "zwave__obs_present",
]

expected_runtime_obs_cols = [f"{t}__obs_present" for t in PROTOCOL_TIERS_USED]
missing_runtime_obs_cols = [c for c in expected_runtime_obs_cols if c not in PROTOCOL_OBS_COLS_REQUIRED]

if missing_runtime_obs_cols:
    raise RuntimeError(
        f"[Cell4] PROTOCOL_OBS_COLS_REQUIRED missing canonical runtime-tier masks: {missing_runtime_obs_cols}"
    )

bad_obs_names = [
    c for c in PROTOCOL_OBS_COLS_REQUIRED
    if not (isinstance(c, str) and c.lower().endswith("__obs_present"))
]
if bad_obs_names:
    raise RuntimeError(
        f"[Cell4] PROTOCOL_OBS_COLS_REQUIRED contains non-obs_present entries: {bad_obs_names}"
    )

EXPECT_EVENT_BASES = bool(CFG.get("expect_events_in_sec_bases", True))
RECONSTRUCT_IF_MISSING = bool(CFG.get("driver_reconstruct_if_missing", False))

if RECONSTRUCT_IF_MISSING:
    raise RuntimeError(
        "[Cell4] driver_reconstruct_if_missing=True is not allowed for this canonical dataset."
    )

ALLOW_DUPLICATE_DRIVER_MERGE = False

SPARSE_NZ_THR = float(CFG.get("other_rest_sparse_nz_threshold", 0.05))
if not np.isfinite(SPARSE_NZ_THR) or SPARSE_NZ_THR <= 0.0 or SPARSE_NZ_THR > 0.5:
    raise RuntimeError(
        f"[Cell4] Invalid other_rest_sparse_nz_threshold={SPARSE_NZ_THR}. "
        "Expected finite value in (0, 0.5]."
    )

CANON_TIME_COL = str(globals()["time_col"])
if CANON_TIME_COL != str(globals().get("CANON_TIME_COL", CANON_TIME_COL)):
    raise RuntimeError(
        f"[Cell4] time_col and CANON_TIME_COL disagree: {CANON_TIME_COL} vs {globals().get('CANON_TIME_COL')}"
    )

globals()["PROTOCOL_TIERS_USED"] = list(PROTOCOL_TIERS_USED)
globals()["PROTOCOL_TIERS_ALL"] = list(PROTOCOL_TIERS_ALL)

log(f"[Cell4] PROTOCOL_TIERS_USED={PROTOCOL_TIERS_USED}")
log(f"[Cell4] PROTOCOL_OBS_COLS_REQUIRED={PROTOCOL_OBS_COLS_REQUIRED}")
log(f"[Cell4] SPARSE_NZ_THR={SPARSE_NZ_THR:.6f}")
log(f"[Cell4] CANON_TIME_COL={CANON_TIME_COL} | EXPECT_EVENT_BASES={EXPECT_EVENT_BASES}")

# ------------------------------------------------------------
# Read split + routing contracts from Cell 3
# ------------------------------------------------------------
split_contract_path = os.path.join(ARTDIR, "split_contract.json")
routing_manifest_path = os.path.join(ARTDIR, "routing_manifest.json")

if not os.path.exists(split_contract_path):
    raise RuntimeError(f"[Cell4] Missing split contract from Cell 3: {split_contract_path}")

if not os.path.exists(routing_manifest_path):
    raise RuntimeError(f"[Cell4] Missing routing manifest from Cell 3: {routing_manifest_path}")

with open(split_contract_path, "r", encoding="utf-8") as f:
    split_contract = json.load(f)

with open(routing_manifest_path, "r", encoding="utf-8") as f:
    routing_manifest = json.load(f)

raw_time_detected = split_contract.get("raw_time_col_detected", None)
provenance_time_col = split_contract.get("provenance_time_col", None)

if split_contract.get("canon_time_col", None) != CANON_TIME_COL:
    raise RuntimeError(
        f"[Cell4] Canonical time mismatch vs split contract: "
        f"{split_contract.get('canon_time_col')} != {CANON_TIME_COL}"
    )

if int(split_contract.get("N_train", -1)) != len(df_tr):
    raise RuntimeError("[Cell4] TRAIN row count disagrees with split contract.")
if int(split_contract.get("N_val", -1)) != len(df_va):
    raise RuntimeError("[Cell4] VAL row count disagrees with split contract.")
if int(split_contract.get("N_test", -1)) != len(df_te):
    raise RuntimeError("[Cell4] TEST row count disagrees with split contract.")

if int(routing_manifest.get("n_event_driver_cols", -1)) != len(EVENT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell4] EVENT_DRIVER_COLS count disagrees with routing manifest: "
        f"{len(EVENT_DRIVER_COLS)} vs {routing_manifest.get('n_event_driver_cols')}"
    )

log(f"[Cell4] Loaded split contract: {split_contract_path}")
log(f"[Cell4] Loaded routing manifest: {routing_manifest_path}")

# ------------------------------------------------------------
# Validate protocol obs_present contract
# ------------------------------------------------------------
log("[Cell4] --- Validate protocol obs_present columns (strict; no fabrication, no value inference) ---")

for df_name, df_part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    missing_obs = [c for c in PROTOCOL_OBS_COLS_REQUIRED if c not in df_part.columns]
    if missing_obs:
        raise RuntimeError(
            f"[Cell4] {df_name}: missing required observability columns: {missing_obs}"
        )

_require_same_columns(PROTOCOL_OBS_COLS_REQUIRED, df_tr, df_va, "TRAIN", "VAL", "PROTOCOL_OBS_COLS_REQUIRED")
_require_same_columns(PROTOCOL_OBS_COLS_REQUIRED, df_tr, df_te, "TRAIN", "TEST", "PROTOCOL_OBS_COLS_REQUIRED")

# Validate and normalize existing masks to clean int8 binary form only after validation.
OBS_MASKS: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "val": {}, "test": {}}

for split_key, split_name, df_part in [
    ("train", "TRAIN", df_tr),
    ("val", "VAL", df_va),
    ("test", "TEST", df_te),
]:
    for c in PROTOCOL_OBS_COLS_REQUIRED:
        m = _validate_binary_mask_series(df_part[c], col=c, split_name=split_name)
        df_part[c] = m
        OBS_MASKS[split_key][c] = m

# Enforce current Z-Wave placeholder assumption when configured.
if bool(CFG.get("drop_all_zero_zwave_obs_from_modeling", True)):
    for split_key, split_name in [("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")]:
        z = OBS_MASKS[split_key]["zwave__obs_present"]
        if np.any(z != 0):
            raise RuntimeError(
                f"[Cell4] {split_name}: zwave__obs_present is nonzero, but "
                "drop_all_zero_zwave_obs_from_modeling=True assumes all-zero Z-Wave coverage."
            )

# Optional consistency audit: ota aggregate should match row-wise OR of ota24/ota5 if all three are present.
OTA_AGGREGATE_AUDIT: Dict[str, Dict[str, float]] = {}
for split_key, split_name in [("train", "TRAIN"), ("val", "VAL"), ("test", "TEST")]:
    ota = OBS_MASKS[split_key]["ota__obs_present"].astype(bool)
    ota_phys = (
        OBS_MASKS[split_key]["ota24__obs_present"].astype(bool)
        | OBS_MASKS[split_key]["ota5__obs_present"].astype(bool)
    )

    mismatch = ota != ota_phys
    OTA_AGGREGATE_AUDIT[split_key] = {
        "mismatch_count": int(mismatch.sum()),
        "mismatch_rate": float(mismatch.mean()) if len(mismatch) else 0.0,
        "ota_rate": float(ota.mean()) if len(ota) else 0.0,
        "ota24_or_ota5_rate": float(ota_phys.mean()) if len(ota_phys) else 0.0,
    }

    if int(mismatch.sum()) > 0:
        log(
            f"[WARN] [Cell4] {split_name}: ota__obs_present differs from "
            f"row-wise OR(ota24, ota5): mismatch_count={int(mismatch.sum())} "
            f"mismatch_rate={float(mismatch.mean()):.6f}. "
            "This is audited, not repaired."
        )

# ------------------------------------------------------------
# Validate no time/provenance leakage in Cell 3 model registries
# ------------------------------------------------------------
for nm, part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    model_view = part.loc[:, list(globals().get("ALL_NUMERIC_MODEL_COLS", []))]
    assert_no_modeling_forbidden_columns(model_view, where=f"Cell4/model_view/{nm}")
    assert_no_abs_time_leakage_columns(model_view, where=f"Cell4/model_view/{nm}")

# ------------------------------------------------------------
# Freeze driver contract from Cell 3 explicit registry
# ------------------------------------------------------------
log("[Cell4] --- Freeze event-driver bases from explicit Cell 3 registry ---")

IOT_DRIVER_COLS = list(globals()["EVENT_DRIVER_COLS"])
IOT_DRIVER_MODE = "events_in_sec"

if EXPECT_EVENT_BASES and not IOT_DRIVER_COLS:
    raise RuntimeError(
        "[Cell4] Expected true events_in_sec driver bases from Cell 3, but EVENT_DRIVER_COLS is empty."
    )

if len(IOT_DRIVER_COLS) != len(set(IOT_DRIVER_COLS)):
    raise RuntimeError("[Cell4] Duplicate driver columns detected in EVENT_DRIVER_COLS.")

bad_driver_names = [c for c in IOT_DRIVER_COLS if not _is_events_in_sec_driver(c)]
if bad_driver_names:
    raise RuntimeError(
        f"[Cell4] EVENT_DRIVER_COLS contains non-events_in_sec entries. first20={bad_driver_names[:20]}"
    )

for df_name, df_part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    missing_drv = [c for c in IOT_DRIVER_COLS if c not in df_part.columns]
    if missing_drv:
        raise RuntimeError(
            f"[Cell4] {df_name}: missing required driver columns: {missing_drv[:20]}"
            + (" ..." if len(missing_drv) > 20 else "")
        )

# No duplicate semantic merging during audit mode.
semantic_keys: Dict[str, List[str]] = {}
for c in IOT_DRIVER_COLS:
    # Keep exact events_in_sec namespace as the semantic key. Do not strip/merge.
    k = str(c)
    semantic_keys.setdefault(k, []).append(c)

dup_groups = {k: v for k, v in semantic_keys.items() if len(v) > 1}
if dup_groups:
    raise RuntimeError(
        f"[Cell4] Duplicate semantic driver aliases detected; audit mode forbids silent merge. "
        f"Examples: {list(dup_groups.items())[:10]}"
    )

# ------------------------------------------------------------
# Strictly validate driver columns, then normalize dtype only
# ------------------------------------------------------------
for split_name, df_part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    for c in IOT_DRIVER_COLS:
        df_part[c] = _validate_driver_series(df_part[c], col=c, split_name=split_name)

# ------------------------------------------------------------
# Validate driver cardinality against known dataset facts
# ------------------------------------------------------------
entity_driver_cols = [c for c in IOT_DRIVER_COLS if "__entity__" in c]
feat_driver_cols = [c for c in IOT_DRIVER_COLS if "__feat__" in c]

expected_entity = int(CFG.get("expected_events_entity_cols", 12))
expected_feat = int(CFG.get("expected_events_feat_cols", 14))
expected_total = expected_entity + expected_feat

if len(IOT_DRIVER_COLS) != expected_total:
    raise RuntimeError(
        f"[Cell4] Driver contract mismatch: detected {len(IOT_DRIVER_COLS)} true event bases, "
        f"expected {expected_total} (= {expected_entity} entity + {expected_feat} feat)."
    )

if len(entity_driver_cols) != expected_entity:
    raise RuntimeError(
        f"[Cell4] Entity-driver count mismatch: detected {len(entity_driver_cols)}, expected {expected_entity}."
    )

if len(feat_driver_cols) != expected_feat:
    raise RuntimeError(
        f"[Cell4] Feat-driver count mismatch: detected {len(feat_driver_cols)}, expected {expected_feat}."
    )

# ------------------------------------------------------------
# Sparse / dense split for OTHER_REST bookkeeping
# ------------------------------------------------------------
globals()["OTHER_REST_BASES_ALL"] = list(IOT_DRIVER_COLS)
globals()["OTHER_REST_BASES_SPARSE"] = []
globals()["OTHER_REST_BASES_DENSE"] = []

driver_stats_train = _driver_stats(df_tr, IOT_DRIVER_COLS, "TRAIN")
driver_stats_val = _driver_stats(df_va, IOT_DRIVER_COLS, "VAL")
driver_stats_test = _driver_stats(df_te, IOT_DRIVER_COLS, "TEST")

sparse_bases = [
    c for c in IOT_DRIVER_COLS
    if float(driver_stats_train[c]["nz_rate"]) <= SPARSE_NZ_THR
]
dense_bases = [
    c for c in IOT_DRIVER_COLS
    if float(driver_stats_train[c]["nz_rate"]) > SPARSE_NZ_THR
]

globals()["OTHER_REST_BASES_SPARSE"] = list(sparse_bases)
globals()["OTHER_REST_BASES_DENSE"] = list(dense_bases)

log(f"[Cell4] OTHER_REST SPARSE bases count={len(sparse_bases)} (nz_thr<={SPARSE_NZ_THR})")
log(f"[Cell4] OTHER_REST DENSE  bases count={len(dense_bases)}")

# ------------------------------------------------------------
# Observability reports
# ------------------------------------------------------------
obs_rates_physical = {
    "train": {c: _mask_rate(OBS_MASKS["train"][c]) for c in PROTOCOL_OBS_COLS_REQUIRED},
    "val": {c: _mask_rate(OBS_MASKS["val"][c]) for c in PROTOCOL_OBS_COLS_REQUIRED},
    "test": {c: _mask_rate(OBS_MASKS["test"][c]) for c in PROTOCOL_OBS_COLS_REQUIRED},
}

obs_transition_rates_physical = {
    "train": {c: _mask_transition_rate(OBS_MASKS["train"][c]) for c in PROTOCOL_OBS_COLS_REQUIRED},
    "val": {c: _mask_transition_rate(OBS_MASKS["val"][c]) for c in PROTOCOL_OBS_COLS_REQUIRED},
    "test": {c: _mask_transition_rate(OBS_MASKS["test"][c]) for c in PROTOCOL_OBS_COLS_REQUIRED},
}

logical_obs_rates = {
    "train": {
        "router": obs_rates_physical["train"]["router__obs_present"],
        "ota": _logical_or_rate(
            OBS_MASKS["train"],
            ["ota__obs_present", "ota24__obs_present", "ota5__obs_present"],
        ),
        "zigbee": obs_rates_physical["train"]["zigbee__obs_present"],
        "zwave": obs_rates_physical["train"]["zwave__obs_present"],
    },
    "val": {
        "router": obs_rates_physical["val"]["router__obs_present"],
        "ota": _logical_or_rate(
            OBS_MASKS["val"],
            ["ota__obs_present", "ota24__obs_present", "ota5__obs_present"],
        ),
        "zigbee": obs_rates_physical["val"]["zigbee__obs_present"],
        "zwave": obs_rates_physical["val"]["zwave__obs_present"],
    },
    "test": {
        "router": obs_rates_physical["test"]["router__obs_present"],
        "ota": _logical_or_rate(
            OBS_MASKS["test"],
            ["ota__obs_present", "ota24__obs_present", "ota5__obs_present"],
        ),
        "zigbee": obs_rates_physical["test"]["zigbee__obs_present"],
        "zwave": obs_rates_physical["test"]["zwave__obs_present"],
    },
}

log(
    "[Cell4] TRAIN obs rates | "
    + " | ".join([f"{c}={obs_rates_physical['train'][c]:.6f}" for c in PROTOCOL_OBS_COLS_REQUIRED])
)
log(
    "[Cell4] VAL   obs rates | "
    + " | ".join([f"{c}={obs_rates_physical['val'][c]:.6f}" for c in PROTOCOL_OBS_COLS_REQUIRED])
)
log(
    "[Cell4] TEST  obs rates | "
    + " | ".join([f"{c}={obs_rates_physical['test'][c]:.6f}" for c in PROTOCOL_OBS_COLS_REQUIRED])
)

log(
    "[Cell4] Logical obs rates | "
    f"train={logical_obs_rates['train']} | "
    f"val={logical_obs_rates['val']} | "
    f"test={logical_obs_rates['test']}"
)

# ------------------------------------------------------------
# Contract artifacts
# ------------------------------------------------------------
protocol_obs_contract = {
    "version": "protocol_obs_contract_v5_strict_binary",
    "canon_time_col": CANON_TIME_COL,
    "protocol_tiers_used": list(PROTOCOL_TIERS_USED),
    "protocol_tiers_all": list(PROTOCOL_TIERS_ALL),
    "protocol_obs_cols_required": list(PROTOCOL_OBS_COLS_REQUIRED),
    "expected_runtime_obs_cols": list(expected_runtime_obs_cols),
    "obs_rates_physical": obs_rates_physical,
    "obs_transition_rates_physical": obs_transition_rates_physical,
    "obs_rates_logical": logical_obs_rates,
    "ota_aggregate_audit": OTA_AGGREGATE_AUDIT,
    "zwave_all_zero_enforced": bool(CFG.get("drop_all_zero_zwave_obs_from_modeling", True)),
}

protocol_obs_contract_path = os.path.join(ARTDIR, "protocol_obs_contract.json")
_write_json(protocol_obs_contract_path, protocol_obs_contract)
log(f"[Cell4] Wrote protocol obs contract: {protocol_obs_contract_path}")

driver_contract = {
    "version": "driver_contract_v7_strict_registry_no_repair",
    "canon_time_col": CANON_TIME_COL,
    "raw_time_col_detected": raw_time_detected,
    "provenance_time_col": provenance_time_col,
    "protocol_tiers_used": list(PROTOCOL_TIERS_USED),
    "protocol_tiers_all": list(PROTOCOL_TIERS_ALL),
    "protocol_obs_cols_required": list(PROTOCOL_OBS_COLS_REQUIRED),
    "expected_runtime_obs_cols": list(expected_runtime_obs_cols),
    "driver_mode": IOT_DRIVER_MODE,
    "driver_reconstructed": False,
    "driver_alias_merge_applied": False,
    "driver_nan_zero_filled": False,
    "driver_negative_clipped": False,
    "iot_driver_cols": list(IOT_DRIVER_COLS),
    "event_bases_count": int(len(IOT_DRIVER_COLS)),
    "entity_driver_count": int(len(entity_driver_cols)),
    "feat_driver_count": int(len(feat_driver_cols)),
    "expected_entity_driver_count": int(expected_entity),
    "expected_feat_driver_count": int(expected_feat),
    "sparse_nz_thr": float(SPARSE_NZ_THR),
    "other_rest_sparse_bases": list(globals()["OTHER_REST_BASES_SPARSE"]),
    "other_rest_dense_bases": list(globals()["OTHER_REST_BASES_DENSE"]),
    "driver_stats_train": driver_stats_train,
    "driver_stats_val": driver_stats_val,
    "driver_stats_test": driver_stats_test,
    "obs_rates_physical": obs_rates_physical,
    "obs_rates_logical": logical_obs_rates,
}

driver_contract_path = os.path.join(ARTDIR, "driver_contract.json")
_write_json(driver_contract_path, driver_contract)
log(f"[Cell4] Wrote driver contract: {driver_contract_path}")

# Optional compact driver-rate artifact for quick inspection.
driver_rates_compact = {
    "version": "driver_rates_compact_v5",
    "train": {
        c: {
            "nz_rate": driver_stats_train[c]["nz_rate"],
            "sum": driver_stats_train[c]["sum"],
            "max": driver_stats_train[c]["max"],
        }
        for c in IOT_DRIVER_COLS
    },
    "val": {
        c: {
            "nz_rate": driver_stats_val[c]["nz_rate"],
            "sum": driver_stats_val[c]["sum"],
            "max": driver_stats_val[c]["max"],
        }
        for c in IOT_DRIVER_COLS
    },
    "test": {
        c: {
            "nz_rate": driver_stats_test[c]["nz_rate"],
            "sum": driver_stats_test[c]["sum"],
            "max": driver_stats_test[c]["max"],
        }
        for c in IOT_DRIVER_COLS
    },
}

driver_rates_path = os.path.join(ARTDIR, "driver_rates_cell4.json")
_write_json(driver_rates_path, driver_rates_compact)
log(f"[Cell4] Wrote driver rates audit: {driver_rates_path}")

# ------------------------------------------------------------
# Expose globals
# ------------------------------------------------------------
globals()["IOT_DRIVER_COLS"] = list(IOT_DRIVER_COLS)
globals()["IOT_DRIVER_MODE"] = IOT_DRIVER_MODE
globals()["PROTOCOL_OBS_COLS_REQUIRED"] = list(PROTOCOL_OBS_COLS_REQUIRED)
globals()["OBS_RATES_PHYSICAL"] = obs_rates_physical
globals()["OBS_RATES_LOGICAL"] = logical_obs_rates
globals()["OBS_TRANSITION_RATES_PHYSICAL"] = obs_transition_rates_physical
globals()["PROTOCOL_OBS_CONTRACT"] = protocol_obs_contract
globals()["DRIVER_CONTRACT"] = driver_contract
globals()["DRIVER_STATS_TRAIN"] = driver_stats_train
globals()["DRIVER_STATS_VAL"] = driver_stats_val
globals()["DRIVER_STATS_TEST"] = driver_stats_test

log(f"[Cell4] Broad protocol/meta matrix contract from Cell 3: D_proto={len(NUMERIC_COLS_PROTO)}")
log(
    f"[Cell4] IOT_DRIVER_MODE={IOT_DRIVER_MODE} | "
    f"IOT_DRIVER_COLS(n)={len(IOT_DRIVER_COLS)} | first 10: {IOT_DRIVER_COLS[:10]}"
)
log("--- END:   Cell 4 — taxonomy / obs / driver-contract ---")