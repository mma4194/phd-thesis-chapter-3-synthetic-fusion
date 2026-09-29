# ==========================================================
# CELL 5 — tiers / protocol contract / logical obs / regimes / TOD — v9
#
# Purpose:
# - Freeze the logical protocol value contract from Cell 4.4
# - Audit protocol observability masks without mutating them
# - Export canonical logical protocol-observation matrices for downstream cells
# - Reuse/validate canonical TOD arrays from Cell 3 when available
# - Fit regime bins on TRAIN time-of-day only, then apply to VAL/TEST
# - Declare the downstream protocol synthesis policy as portfolio/hybrid-based
#
# Critical correction:
# - DO NOT fit regime bins on absolute TRAIN timestamps.
# - Regimes here represent recurring daily context, not absolute chronology.
# - Protocol logical observability must be frozen here so later cells do not
#   re-infer masks inconsistently.
#
# Scientific policy:
# - no value-based observability inference
# - no mutation/repair of obs masks
# - no TEST-informed regime fitting
# - no absolute-time leakage into model features
# - PROTO_COLS_CONTRACT comes only from Cell 4.4 PROTO_VALUE_COLS
# - downstream generators must treat DDPM as one candidate family, not as the
#   universal protocol generator
#
# Downstream invariant:
# - PROTO_COLS_CONTRACT is the full protocol value universe.
# - DDPM_VALUE_COLS, if later defined, must be a subset of PROTO_COLS_CONTRACT.
# - The old invariant DDPM_VALUE_COLS == PROTO_COLS_CONTRACT is invalid.
# ==========================================================

log("--- START: Cell 5 — tiers / protocol contract / logical obs / regimes / TOD v9 ---")

import os
import json
import hashlib
import numpy as np
import pandas as pd

need = [
    "CFG", "df_tr", "df_va", "df_te", "time_col", "log",
    "PROTO_VALUE_COLS",
    "PROTOCOL_TIERS_USED",
    "PROTOCOL_OBS_COLS_REQUIRED",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell5] Missing prerequisites: {missing}. Run Cells 1–4.4 first.")

OUTDIR = str(CFG["outdir"])
OUT_ART = os.path.join(OUTDIR, "artifacts")
OUT_REP = os.path.join(OUTDIR, "reports")
os.makedirs(OUT_ART, exist_ok=True)
os.makedirs(OUT_REP, exist_ok=True)

# ----------------------------------------------------------
# 0) Helpers
# ----------------------------------------------------------
def _write_json(path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

def _sha_list(xs) -> str:
    return hashlib.sha256(("||".join(map(str, xs))).encode("utf-8")).hexdigest()

def _ordered_subset_check(subset, full, label: str) -> None:
    subset = list(subset)
    full = list(full)

    if len(subset) != len(set(subset)):
        raise RuntimeError(f"[Cell5] Duplicate entries in {label}: {subset}")

    unknown = [x for x in subset if x not in full]
    if unknown:
        raise RuntimeError(f"[Cell5] Unknown entries in {label}: {unknown}. Allowed={full}")

    idx = [full.index(x) for x in subset]
    if idx != sorted(idx):
        raise RuntimeError(
            f"[Cell5] {label} order mismatch. Expected subset order of {full}, got {subset}"
        )

def _numeric_array_strict(df_: pd.DataFrame, col: str, split_name: str) -> np.ndarray:
    if col not in df_.columns:
        raise RuntimeError(f"[Cell5] Missing column {split_name}:{col}")

    try:
        arr = pd.to_numeric(df_[col], errors="raise").to_numpy(dtype=np.float64, copy=False)
    except Exception as e:
        raise RuntimeError(f"[Cell5] {split_name}:{col} is not strictly numeric.") from e

    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(f"[Cell5] Non-finite values in {split_name}:{col} | bad={bad}")

    return arr

def _audit_binary_mask_strict(df_: pd.DataFrame, col: str, split_name: str) -> np.ndarray:
    arr = _numeric_array_strict(df_, col, split_name)
    rounded = np.rint(arr)
    max_abs_err = float(np.max(np.abs(arr - rounded))) if len(arr) else 0.0

    if max_abs_err > 1e-6:
        bad_idx = int(np.where(np.abs(arr - rounded) > 1e-6)[0][0])
        raise RuntimeError(
            f"[Cell5] Non-integer-like observability mask in {split_name}:{col}. "
            f"first_bad_idx={bad_idx} value={arr[bad_idx]} max_abs_err={max_abs_err}"
        )

    vals = np.unique(rounded.astype(np.int64))
    bad_vals = [int(v) for v in vals if int(v) not in (0, 1)]
    if bad_vals:
        raise RuntimeError(
            f"[Cell5] Non-binary values detected in {split_name}:{col} | bad_values={bad_vals[:20]}"
        )

    return rounded.astype(np.int8, copy=False)

def _obs_rate(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=np.int8)
    if len(mask) == 0:
        return 0.0
    return float(mask.mean())

def _transition_rate(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=np.int8)
    if len(mask) <= 1:
        return 0.0
    return float(np.mean(mask[1:] != mask[:-1]))

def _rowwise_or(masks) -> np.ndarray:
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if not masks:
        raise ValueError("[Cell5] _rowwise_or received no masks.")

    out = masks[0].copy()
    for m in masks[1:]:
        if m.shape != out.shape:
            raise RuntimeError("[Cell5] Mask shape mismatch inside rowwise OR.")
        out |= m

    return out.astype(np.int8, copy=False)

def _sec_of_day(sec_arr: np.ndarray) -> np.ndarray:
    day = 86400.0
    sod = np.mod(sec_arr, day).astype(np.float64, copy=False)
    sod[sod < 0.0] += day
    return sod

def _tod_sin_cos(sec_arr: np.ndarray) -> np.ndarray:
    day = 86400.0
    phase = (sec_arr % day) / day * (2.0 * np.pi)
    out = np.column_stack([np.sin(phase), np.cos(phase)]).astype(np.float32, copy=False)

    if not np.isfinite(out).all():
        raise RuntimeError("[Cell5] TOD contains non-finite values.")

    radius = np.sqrt(np.sum(out.astype(np.float64) ** 2, axis=1))
    if not np.allclose(radius, 1.0, atol=1e-5):
        raise RuntimeError("[Cell5] TOD sin/cos radius check failed.")

    return out

def _reg_counts(x: np.ndarray, K: int) -> list:
    bc = np.bincount(x.astype(np.int64, copy=False), minlength=int(K))
    return [int(v) for v in bc.tolist()]

def _looks_mask_or_meta(c: str) -> bool:
    cl = str(c).lower()
    return (
        cl.endswith("__obs_present")
        or cl.endswith("__present")
        or cl.endswith("__mask")
        or cl.endswith("__staleness_s")
        or cl.endswith("__stale_flag")
        or ("traffic_present" in cl)
        or ("traffic_obs_present" in cl)
    )

def _require_cols_in_all_splits(cols, label: str) -> None:
    cols = list(cols)
    split_cols = {
        "TRAIN": set(df_tr.columns),
        "VAL": set(df_va.columns),
        "TEST": set(df_te.columns),
    }

    miss = {
        split: [c for c in cols if c not in split_set]
        for split, split_set in split_cols.items()
    }
    bad = {split: xs for split, xs in miss.items() if xs}
    if bad:
        first = {split: xs[:20] for split, xs in bad.items()}
        raise RuntimeError(
            f"[Cell5] Required columns missing across splits for {label}: {first}"
        )

# ----------------------------------------------------------
# 1) Tier definitions
# ----------------------------------------------------------
ALLOWED_PROTOCOL_TIERS = ["router", "ota", "zigbee", "zwave"]

TIERS = list(globals().get("TIERS", ALLOWED_PROTOCOL_TIERS))
if not TIERS:
    raise RuntimeError("[Cell5] TIERS is empty.")

_ordered_subset_check(TIERS, ALLOWED_PROTOCOL_TIERS, "TIERS")

PROTOCOL_TIERS_USED = list(globals().get("PROTOCOL_TIERS_USED") or [])
if not PROTOCOL_TIERS_USED:
    raise RuntimeError("[Cell5] PROTOCOL_TIERS_USED missing or empty.")

_ordered_subset_check(PROTOCOL_TIERS_USED, TIERS, "PROTOCOL_TIERS_USED")


PROTOCOL_OBS_COLS_REQUIRED = list(globals().get("PROTOCOL_OBS_COLS_REQUIRED") or [])
if not PROTOCOL_OBS_COLS_REQUIRED:
    raise RuntimeError("[Cell5] PROTOCOL_OBS_COLS_REQUIRED missing or empty.")

if len(PROTOCOL_OBS_COLS_REQUIRED) != len(set(PROTOCOL_OBS_COLS_REQUIRED)):
    raise RuntimeError(
        f"[Cell5] Duplicate entries in PROTOCOL_OBS_COLS_REQUIRED: {PROTOCOL_OBS_COLS_REQUIRED}"
    )

bad_obs_names = [
    c for c in PROTOCOL_OBS_COLS_REQUIRED
    if not (isinstance(c, str) and c.lower().endswith("__obs_present"))
]
if bad_obs_names:
    raise RuntimeError(
        f"[Cell5] PROTOCOL_OBS_COLS_REQUIRED contains non-obs_present columns: {bad_obs_names}"
    )

expected_runtime_obs_cols = [f"{t}__obs_present" for t in PROTOCOL_TIERS_USED]
missing_runtime_obs_cols = [
    c for c in expected_runtime_obs_cols
    if c not in PROTOCOL_OBS_COLS_REQUIRED
]
if missing_runtime_obs_cols:
    raise RuntimeError(
        f"[Cell5] PROTOCOL_OBS_COLS_REQUIRED missing runtime tier masks: {missing_runtime_obs_cols}"
    )

# Stronger OTA validation: logical OTA requires aggregate and physical audit masks.
if "ota" in PROTOCOL_TIERS_USED:
    required_ota_logical_inputs = [
        "ota__obs_present",
        "ota24__obs_present",
        "ota5__obs_present",
    ]
    missing_ota_inputs = [
        c for c in required_ota_logical_inputs
        if c not in PROTOCOL_OBS_COLS_REQUIRED
    ]
    if missing_ota_inputs:
        raise RuntimeError(
            "[Cell5] Logical OTA mask requires aggregate + physical OTA masks. "
            f"Missing from PROTOCOL_OBS_COLS_REQUIRED: {missing_ota_inputs}"
        )

_require_cols_in_all_splits(PROTOCOL_OBS_COLS_REQUIRED, "PROTOCOL_OBS_COLS_REQUIRED")

log(f"TIERS (superset): {TIERS}")
log(f"PROTOCOL_TIERS_USED (runtime): {PROTOCOL_TIERS_USED}")
log(f"PROTOCOL_OBS_COLS_REQUIRED: {PROTOCOL_OBS_COLS_REQUIRED}")

# ----------------------------------------------------------
# 2) Audit obs_present masks and freeze physical/logical masks
# ----------------------------------------------------------
OBS_MASKS_PHYSICAL = {"train": {}, "val": {}, "test": {}}

for split_key, split_name, df_part in [
    ("train", "TRAIN", df_tr),
    ("val", "VAL", df_va),
    ("test", "TEST", df_te),
]:
    for c in PROTOCOL_OBS_COLS_REQUIRED:
        OBS_MASKS_PHYSICAL[split_key][c] = _audit_binary_mask_strict(df_part, c, split_name)

obs_rates_physical = {
    split: {
        c: _obs_rate(OBS_MASKS_PHYSICAL[split][c])
        for c in PROTOCOL_OBS_COLS_REQUIRED
    }
    for split in ["train", "val", "test"]
}

obs_transition_rates_physical = {
    split: {
        c: _transition_rate(OBS_MASKS_PHYSICAL[split][c])
        for c in PROTOCOL_OBS_COLS_REQUIRED
    }
    for split in ["train", "val", "test"]
}

def _logical_mask_for_tier(split_key: str, tier: str) -> np.ndarray:
    m = OBS_MASKS_PHYSICAL[split_key]

    if tier == "router":
        return m["router__obs_present"].copy()

    if tier == "ota":
        return _rowwise_or([
            m["ota__obs_present"],
            m["ota24__obs_present"],
            m["ota5__obs_present"],
        ])

    if tier == "zigbee":
        return m["zigbee__obs_present"].copy()

    if tier == "zwave":
        return m["zwave__obs_present"].copy()

    raise RuntimeError(f"[Cell5] Unsupported logical tier: {tier}")

OBS_MASKS_LOGICAL = {"train": {}, "val": {}, "test": {}}
for split in ["train", "val", "test"]:
    for tier in PROTOCOL_TIERS_USED:
        OBS_MASKS_LOGICAL[split][tier] = _logical_mask_for_tier(split, tier)

PROTO_OBS_TR = np.stack(
    [OBS_MASKS_LOGICAL["train"][t] for t in PROTOCOL_TIERS_USED],
    axis=1,
).astype(np.float32)

PROTO_OBS_VA = np.stack(
    [OBS_MASKS_LOGICAL["val"][t] for t in PROTOCOL_TIERS_USED],
    axis=1,
).astype(np.float32)

PROTO_OBS_TE = np.stack(
    [OBS_MASKS_LOGICAL["test"][t] for t in PROTOCOL_TIERS_USED],
    axis=1,
).astype(np.float32)

for nm, arr, expected_n in [
    ("PROTO_OBS_TR", PROTO_OBS_TR, len(df_tr)),
    ("PROTO_OBS_VA", PROTO_OBS_VA, len(df_va)),
    ("PROTO_OBS_TE", PROTO_OBS_TE, len(df_te)),
]:
    expected_shape = (expected_n, len(PROTOCOL_TIERS_USED))
    if arr.shape != expected_shape:
        raise RuntimeError(
            f"[Cell5] {nm} shape mismatch: got={arr.shape}, expected={expected_shape}"
        )
    if not np.isfinite(arr).all():
        raise RuntimeError(f"[Cell5] {nm} contains non-finite values.")
    uniq = set(np.unique(arr).round(6).tolist())
    if not uniq.issubset({0.0, 1.0}):
        raise RuntimeError(f"[Cell5] {nm} is not binary. unique={sorted(uniq)}")

obs_rates_logical = {
    split: {
        t: _obs_rate(OBS_MASKS_LOGICAL[split][t])
        for t in PROTOCOL_TIERS_USED
    }
    for split in ["train", "val", "test"]
}

obs_transition_rates_logical = {
    split: {
        t: _transition_rate(OBS_MASKS_LOGICAL[split][t])
        for t in PROTOCOL_TIERS_USED
    }
    for split in ["train", "val", "test"]
}

log("Obs rates physical (TRAIN): " + str(obs_rates_physical["train"]))
log("Obs rates physical (VAL):   " + str(obs_rates_physical["val"]))
log("Obs rates physical (TEST):  " + str(obs_rates_physical["test"]))
log("Obs rates logical  (TRAIN): " + str(obs_rates_logical["train"]))
log("Obs rates logical  (VAL):   " + str(obs_rates_logical["val"]))
log("Obs rates logical  (TEST):  " + str(obs_rates_logical["test"]))

# ----------------------------------------------------------
# 3) Protocol column contract — value cols only
# ----------------------------------------------------------
proto_value_base = list(globals()["PROTO_VALUE_COLS"])
if not proto_value_base:
    raise RuntimeError("[Cell5] PROTO_VALUE_COLS is empty. Taxonomy split likely failed.")

if len(proto_value_base) != len(set(proto_value_base)):
    dup = pd.Series(proto_value_base).value_counts()
    dup = dup[dup > 1].index.tolist()
    raise RuntimeError(f"[Cell5] Duplicate entries in PROTO_VALUE_COLS: {dup[:20]}")

_require_cols_in_all_splits(proto_value_base, "PROTO_VALUE_COLS")

PROTO_COLS_CONTRACT = list(proto_value_base)

bad_in_contract = [c for c in PROTO_COLS_CONTRACT if _looks_mask_or_meta(c)]
if bad_in_contract:
    raise RuntimeError(
        "[Cell5] Leakage risk: protocol value contract contains mask/meta-like columns. "
        f"n_bad={len(bad_in_contract)} first20={bad_in_contract[:20]}"
    )

if any(c.startswith("iot__") for c in PROTO_COLS_CONTRACT):
    raise RuntimeError("[Cell5] Leakage risk: IoT columns entered PROTO_COLS_CONTRACT.")

if any(c.startswith("events_in_sec__") for c in PROTO_COLS_CONTRACT):
    raise RuntimeError("[Cell5] Leakage risk: event-driver columns entered PROTO_COLS_CONTRACT.")

if any(c.startswith("telemetry_in_sec__") for c in PROTO_COLS_CONTRACT):
    raise RuntimeError("[Cell5] Leakage risk: telemetry helper columns entered PROTO_COLS_CONTRACT.")

if any(c.startswith(("ota24__", "ota5__", "zwave__")) for c in PROTO_COLS_CONTRACT):
    raise RuntimeError("[Cell5] Physical-only protocol namespace leaked into logical value contract.")

expected_core = int(CFG.get("expected_proto_value_cols_core", 54))
if expected_core > 0 and len(PROTO_COLS_CONTRACT) != expected_core:
    raise RuntimeError(
        f"[Cell5] Protocol contract size mismatch: got={len(PROTO_COLS_CONTRACT)} expected={expected_core}"
    )

proto_sig = _sha_list(PROTO_COLS_CONTRACT)

proto_contract_path = os.path.join(OUT_ART, "proto_cols_contract.json")
_write_json(proto_contract_path, PROTO_COLS_CONTRACT)

log(
    f"[Cell5] Saved proto contract: {proto_contract_path} | "
    f"n={len(PROTO_COLS_CONTRACT)} | sha256={proto_sig[:12]}..."
)

# ----------------------------------------------------------
# 4) TOD arrays and regime IDs
# ----------------------------------------------------------
SEC_COL = str(globals().get("SEC_COL", globals().get("time_col", "sec_epoch_s__canon")))

for split_name, df_part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    if SEC_COL not in df_part.columns:
        raise RuntimeError(f"[Cell5] SEC_COL '{SEC_COL}' missing from {split_name}.")

t_tr = _numeric_array_strict(df_tr, SEC_COL, "TRAIN")
t_va = _numeric_array_strict(df_va, SEC_COL, "VAL")
t_te = _numeric_array_strict(df_te, SEC_COL, "TEST")

for name, arr in [("TRAIN", t_tr), ("VAL", t_va), ("TEST", t_te)]:
    if arr.size > 1 and np.any(arr[1:] <= arr[:-1]):
        raise RuntimeError(f"[Cell5] Canonical time is not strictly increasing in {name}.")

tod_tr_new = _tod_sin_cos(t_tr)
tod_va_new = _tod_sin_cos(t_va)
tod_te_new = _tod_sin_cos(t_te)

# Validate against Cell 3 TOD globals if they exist.
for nm, new_arr, old_name in [
    ("TRAIN", tod_tr_new, "tod_tr"),
    ("VAL", tod_va_new, "tod_va"),
    ("TEST", tod_te_new, "tod_te"),
]:
    if old_name in globals():
        old_arr = np.asarray(globals()[old_name], dtype=np.float32)
        if old_arr.shape == new_arr.shape:
            max_abs = float(np.max(np.abs(old_arr - new_arr))) if new_arr.size else 0.0
            if max_abs > 1e-5:
                raise RuntimeError(
                    f"[Cell5] Recomputed TOD disagrees with existing {old_name}: "
                    f"split={nm} max_abs_diff={max_abs}"
                )

tod_tr = tod_tr_new
tod_va = tod_va_new
tod_te = tod_te_new

K_reg_req = int(CFG.get("K_reg", 8))
if K_reg_req <= 0:
    raise RuntimeError(f"[Cell5] K_reg must be positive. Got {K_reg_req}")

sod_tr = _sec_of_day(t_tr)
sod_va = _sec_of_day(t_va)
sod_te = _sec_of_day(t_te)

# Fit bins on TRAIN sec-of-day only, then apply cyclically to all splits.
qs = np.linspace(0.0, 1.0, K_reg_req + 1)
edges_sod = np.quantile(sod_tr, qs).astype(np.float64)
edges_sod = np.unique(edges_sod)

if edges_sod.size < 2:
    raise RuntimeError(
        "[Cell5] TRAIN sec-of-day regime fit collapsed to fewer than 2 unique edges. "
        "This means sec-of-day variation is insufficient for regime construction."
    )

def _digitize_reg_sod(sec_of_day_arr: np.ndarray, edges_: np.ndarray) -> np.ndarray:
    r = np.searchsorted(edges_[1:], sec_of_day_arr, side="right").astype(np.int64)
    return np.clip(r, 0, max(0, edges_.size - 2)).astype(np.int64, copy=False)

reg_tr = _digitize_reg_sod(sod_tr, edges_sod)
reg_va = _digitize_reg_sod(sod_va, edges_sod)
reg_te = _digitize_reg_sod(sod_te, edges_sod)

K_reg_eff = int(max(1, edges_sod.size - 1))

for nm, reg, expected_n in [
    ("TRAIN", reg_tr, len(df_tr)),
    ("VAL", reg_va, len(df_va)),
    ("TEST", reg_te, len(df_te)),
]:
    if reg.shape != (expected_n,):
        raise RuntimeError(f"[Cell5] {nm} regime shape mismatch: {reg.shape} expected={(expected_n,)}")
    if reg.min() < 0 or reg.max() >= K_reg_eff:
        raise RuntimeError(f"[Cell5] {nm} regime IDs outside [0, {K_reg_eff - 1}].")

reg_counts = {
    "train": _reg_counts(reg_tr, K_reg_eff),
    "val": _reg_counts(reg_va, K_reg_eff),
    "test": _reg_counts(reg_te, K_reg_eff),
}

for split, counts in reg_counts.items():
    active = int(np.sum(np.asarray(counts) > 0))
    if active < max(1, min(2, K_reg_eff)):
        log(f"[WARN] {split.upper()} regimes show low support spread: active={active}/{K_reg_eff}")

# ----------------------------------------------------------
# 5) Downstream protocol synthesis policy
# ----------------------------------------------------------
protocol_synthesis_policy = {
    "version": "protocol_synthesis_policy_v1_hybrid_portfolio",
    "declared_in_cell": "Cell5",
    "proto_cols_contract_source": "Cell4.4: PROTO_VALUE_COLS",
    "proto_cols_contract_sha256": proto_sig,
    "proto_cols_contract_n": int(len(PROTO_COLS_CONTRACT)),
    "policy": {
        "synthesis_architecture": "validation_selected_hybrid_portfolio",
        "baseline": "TRAIN_only_temporal_block_bootstrap_A1",
        "A2_definition": "VAL_selected_hybrid_protocol_output",
        "ddpm_role": "candidate_generator_only",
        "ddpm_may_model_subset_only": True,
        "ddpm_equal_proto_contract_allowed": False,
        "candidate_selection_split": "VAL_only",
        "test_usage": "final_QA_only",
        "observability_policy": "frozen_logical_masks_from_Cell5",
        "regime_policy": "TRAIN_sec_of_day_quantiles_only",
    },
    "required_downstream_invariants": [
        "set(DDPM_VALUE_COLS).issubset(set(PROTO_COLS_CONTRACT))",
        "DDPM_VALUE_COLS must match Cell6 PROTOCOL_GENERATOR_ROLE_CONTRACT['ddpm_candidate_cols'] when that artifact exists",
        "No downstream cell may infer protocol observability from protocol values",
        "No downstream candidate-selection logic may use TEST metrics",
        "A1 must remain available as the TRAIN-only temporal baseline for all protocol columns",
        "A2 must preserve protocol schema, logical mask semantics, and NaN-under-unobserved semantics",
    ],
}

policy_path = os.path.join(OUT_ART, "protocol_synthesis_policy.json")
_write_json(policy_path, protocol_synthesis_policy)

# ----------------------------------------------------------
# 6) Artifacts and globals
# ----------------------------------------------------------
globals()["PROTOCOL_TIERS"] = list(TIERS)
globals()["PROTOCOL_TIERS_USED"] = list(PROTOCOL_TIERS_USED)
globals()["PROTOCOL_OBS_COLS_REQUIRED"] = list(PROTOCOL_OBS_COLS_REQUIRED)
globals()["PROTO_COLS_CONTRACT"] = list(PROTO_COLS_CONTRACT)
globals()["PROTO_CONTRACT_SIG"] = proto_sig

globals()["OBS_MASKS_PHYSICAL"] = OBS_MASKS_PHYSICAL
globals()["OBS_MASKS_LOGICAL"] = OBS_MASKS_LOGICAL
globals()["PROTO_OBS_TR"] = PROTO_OBS_TR
globals()["PROTO_OBS_VA"] = PROTO_OBS_VA
globals()["PROTO_OBS_TE"] = PROTO_OBS_TE

# Backward-compatible common aliases.
globals()["obs_tr"] = PROTO_OBS_TR
globals()["obs_va"] = PROTO_OBS_VA
globals()["obs_te"] = PROTO_OBS_TE

globals()["tod_tr"] = tod_tr
globals()["tod_va"] = tod_va
globals()["tod_te"] = tod_te
globals()["reg_tr"] = reg_tr
globals()["reg_va"] = reg_va
globals()["reg_te"] = reg_te
globals()["K_reg"] = K_reg_eff
globals()["SEC_COL"] = SEC_COL
globals()["PROTOCOL_SYNTHESIS_POLICY"] = protocol_synthesis_policy

proto_manifest = {
    "version": "proto_contract_manifest_v9",
    "TIERS": list(TIERS),
    "PROTOCOL_TIERS_USED": list(PROTOCOL_TIERS_USED),
    "PROTOCOL_OBS_COLS_REQUIRED": list(PROTOCOL_OBS_COLS_REQUIRED),
    "n_proto_cols_contract": int(len(PROTO_COLS_CONTRACT)),
    "proto_contract_sha256": proto_sig,
    "source_of_truth": "Cell4.4: PROTO_VALUE_COLS",
    "MODEL_DERIVED_VALUES": bool(CFG.get("model_derived_protocol_values", False)),
    "n_core": int(len(globals().get("PROTO_VALUE_COLS_CORE", []))),
    "n_derived": int(len(globals().get("PROTO_VALUE_COLS_DERIVED", []))),
    "expected_proto_value_cols_core": int(expected_core),
    "downstream_synthesis_policy": {
        "architecture": "validation_selected_hybrid_portfolio",
        "baseline": "TRAIN_only_temporal_block_bootstrap_A1",
        "ddpm_role": "candidate_generator_only",
        "ddpm_value_cols_invariant": "subset_of_PROTO_COLS_CONTRACT_not_equal_required",
        "candidate_selection": "VAL_only",
        "test_usage": "final_QA_only",
    },
}
_write_json(os.path.join(OUT_ART, "proto_contract_manifest.json"), proto_manifest)

obs_contract = {
    "version": "protocol_logical_obs_contract_v9",
    "protocol_tiers_used": list(PROTOCOL_TIERS_USED),
    "protocol_obs_cols_required": list(PROTOCOL_OBS_COLS_REQUIRED),
    "logical_mask_policy": {
        "router": "router__obs_present",
        "ota": "rowwise_or(ota__obs_present, ota24__obs_present, ota5__obs_present)",
        "zigbee": "zigbee__obs_present",
        "zwave": "zwave__obs_present",
    },
    "no_value_based_observability_inference": True,
    "no_mask_mutation_or_repair": True,
    "obs_rates_physical": obs_rates_physical,
    "obs_transition_rates_physical": obs_transition_rates_physical,
    "obs_rates_logical": obs_rates_logical,
    "obs_transition_rates_logical": obs_transition_rates_logical,
    "proto_obs_shapes": {
        "train": list(PROTO_OBS_TR.shape),
        "val": list(PROTO_OBS_VA.shape),
        "test": list(PROTO_OBS_TE.shape),
    },
}
obs_contract_path = os.path.join(OUT_ART, "protocol_logical_obs_contract.json")
_write_json(obs_contract_path, obs_contract)

reg_spec = {
    "version": "regime_spec_v9_sec_of_day_train_quantiles",
    "SEC_COL": SEC_COL,
    "regime_basis": "sec_of_day_train_quantiles",
    "fit_split": "TRAIN_only",
    "applied_to": ["TRAIN", "VAL", "TEST"],
    "absolute_time_used_for_modeling": False,
    "K_reg_requested": int(K_reg_req),
    "K_reg_effective": int(K_reg_eff),
    "edges_sec_of_day": edges_sod.tolist(),
    "regime_counts": reg_counts,
}
reg_path = os.path.join(OUT_ART, "regime_spec.json")
_write_json(reg_path, reg_spec)

# Compact CSV audit for quick inspection.
obs_rows = []
for split in ["train", "val", "test"]:
    for c in PROTOCOL_OBS_COLS_REQUIRED:
        obs_rows.append({
            "split": split,
            "kind": "physical",
            "name": c,
            "obs_rate": float(obs_rates_physical[split][c]),
            "transition_rate": float(obs_transition_rates_physical[split][c]),
        })
    for t in PROTOCOL_TIERS_USED:
        obs_rows.append({
            "split": split,
            "kind": "logical",
            "name": t,
            "obs_rate": float(obs_rates_logical[split][t]),
            "transition_rate": float(obs_transition_rates_logical[split][t]),
        })

obs_audit_path = os.path.join(OUT_REP, "cell5_protocol_obs_audit.csv")
pd.DataFrame(obs_rows).to_csv(obs_audit_path, index=False)

log(f"[Cell5] Saved protocol logical obs contract: {obs_contract_path}")
log(f"[Cell5] Saved regime spec: {reg_path}")
log(f"[Cell5] Saved protocol synthesis policy: {policy_path}")
log(f"[Cell5] Saved obs audit: {obs_audit_path}")
log(f"[Cell5] TOD shapes: tr={tod_tr.shape} va={tod_va.shape} te={tod_te.shape}")
log(f"[Cell5] REG shapes: tr={reg_tr.shape} va={reg_va.shape} te={reg_te.shape} | K_reg={K_reg_eff}")
log(f"[Cell5] PROTO_OBS shapes: tr={PROTO_OBS_TR.shape} va={PROTO_OBS_VA.shape} te={PROTO_OBS_TE.shape}")
log(f"[Cell5] REG counts TRAIN={reg_counts['train']} | VAL={reg_counts['val']} | TEST={reg_counts['test']}")
log(
    "[Cell5] Downstream protocol synthesis policy: "
    "hybrid VAL-selected portfolio; DDPM=candidate-only; TEST=final-QA-only"
)

log("--- END:   Cell 5 ---")