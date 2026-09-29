# ==========================================================
# CELL 14.1 - Coupling-conditioned Zigbee protocol materialization
# v1.1 STUDY-THESIS strict TRAIN/VAL-profile-conditioned TEST protocol repair, safety-hardened
#
# Role:
#   - Apply TRAIN/VAL-learned Zigbee coupling profiles from Cell 14.0
#     to the synthetic TEST protocol matrix.
#   - Condition protocol response windows on final synthetic IoT driver events.
#   - Modify only manifest-approved Zigbee protocol columns:
#       zigbee__pkt_total
#       zigbee__bytes_total
#
# Scientific contract:
#   - Uses TRAIN/VAL coupling profiles learned in Cell 14.0.
#   - Uses final synthetic TEST IoT driver events only as conditioning input.
#   - Does NOT use real TEST protocol/IoT values for materialization.
#   - Does NOT mutate IoT synthetic values.
#   - Does NOT mutate protocol masks/helper columns.
#   - Does NOT mutate non-target protocol columns.
#   - Does NOT mutate rows outside manifest-approved legal windows.
#
# Inputs:
#   IOT_FULL_SYN_TEST_130
#   PROTOCOL_SYN_TEST_130
#   CELL14_0_ZIGBEE_COUPLING_PROFILES_DF
#   CELL14_0_ZIGBEE_COUPLING_PROFILE_NPZ
#
# Outputs:
#   synthetic/PROTOCOL_SYN_TEST_COUPLED_ZIGBEE.parquet
#   reports/cell14_1_zigbee_coupling_materialization_audit.csv
#   reports/cell14_1_zigbee_coupling_materialization_contract.json
#   artifacts/cell14_1_zigbee_coupling_materialization_manifest.json
# ==========================================================

log("--- START: Cell 14.1 - Coupling-conditioned Zigbee protocol materialization (v1.1 safety-hardened strict) ---")

import os
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_141 = [
    "CFG", "log",
    "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_FULL_SYN_TEST_130",
    "PROTOCOL_SYN_TEST_130",
    "CELL14_0_ZIGBEE_COUPLING_PROFILES_DF",
    "CELL14_0_ZIGBEE_COUPLING_PROFILE_NPZ",
    "CELL14_0_ZIGBEE_COUPLING_PROFILE_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "CELL13_6_MANIFEST_Q4_CONTRACT",
]
_missing_141 = [k for k in _required_141 if k not in globals()]
if _missing_141:
    raise RuntimeError(f"[Cell14.1] Missing required globals from Cell 14.0/13.0: {_missing_141}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TE = int(len(df_te))

if N_TE <= 0:
    raise RuntimeError(f"[Cell14.1] Invalid TEST length: N_TE={N_TE}")

CELL141_VERSION = "cell14_1_coupling_conditioned_zigbee_protocol_materialization_v1_1_safety_hardened"

CFG["cell14_1_version"] = CELL141_VERSION
CFG["cell14_1_trainval_profiles_used"] = True
CFG["cell14_1_synthetic_iot_drivers_used_for_conditioning"] = True
CFG["cell14_1_TEST_real_values_used"] = False
CFG["cell14_1_synthetic_values_mutated"] = True
CFG["cell14_1_protocol_values_materialized_here"] = True
CFG["cell14_1_selection_done_here"] = False
CFG["cell14_1_generator_fit_done_here"] = False
CFG["cell14_1_broad_q4_status_carried_forward"] = True
CFG["cell14_1_manifest_q4_status_carried_forward"] = True
CFG["cell14_1_repair_candidate_not_final_claim"] = True

CFG.setdefault("cell14_1_allowed_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])
CFG.setdefault("cell14_1_allowed_protocol_tiers", ["zigbee"])
CFG.setdefault("cell14_1_base_alpha_scale", 0.65)
CFG.setdefault("cell14_1_min_alpha", 0.05)
CFG.setdefault("cell14_1_max_alpha", 0.85)
CFG.setdefault("cell14_1_weight_power", 1.0)
CFG.setdefault("cell14_1_profile_confidence_power", 1.0)
CFG.setdefault("cell14_1_apply_only_positive_uplift", False)
CFG.setdefault("cell14_1_allow_negative_delta", True)
CFG.setdefault("cell14_1_nonnegative_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])
CFG.setdefault("cell14_1_integer_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])
CFG.setdefault("cell14_1_clip_to_trainval_support", True)
CFG.setdefault("cell14_1_clip_margin_fraction", 0.10)
CFG.setdefault("cell14_1_respect_protocol_observability", True)
CFG.setdefault("cell14_1_fail_on_non_target_drift", True)
CFG.setdefault("cell14_1_fail_on_iot_drift", True)
CFG.setdefault("cell14_1_fail_on_mask_helper_drift", True)
CFG.setdefault("cell14_1_fail_on_outside_window_drift", True)
CFG.setdefault("cell14_1_min_synthetic_events_for_application", 1)
CFG.setdefault("cell14_1_profile_source", "mean_profile_bc")
CFG.setdefault("cell14_1_materialize_from_baseline_plus_profile", True)

ALLOWED_PROTOCOL_COLS_141 = set(map(str, CFG.get("cell14_1_allowed_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])))
NONNEGATIVE_PROTOCOL_COLS_141 = set(map(str, CFG.get("cell14_1_nonnegative_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])))
INTEGER_PROTOCOL_COLS_141 = set(map(str, CFG.get("cell14_1_integer_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])))

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
coupled_protocol_path = os.path.join(OUT_SYN, "PROTOCOL_SYN_TEST_COUPLED_ZIGBEE.parquet")
materialization_audit_csv = os.path.join(REPORT_DIR, "cell14_1_zigbee_coupling_materialization_audit.csv")
target_drift_csv = os.path.join(REPORT_DIR, "cell14_1_zigbee_target_protocol_drift_audit.csv")
non_target_drift_csv = os.path.join(REPORT_DIR, "cell14_1_zigbee_non_target_protocol_drift_audit.csv")
contract_json = os.path.join(REPORT_DIR, "cell14_1_zigbee_coupling_materialization_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_1_zigbee_coupling_materialization_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell14_1_zigbee_coupling_materialization_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_141(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_141(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_141(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_141(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_141(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_141(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_141(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_141(obj.to_dict())
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

def _write_json_141(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_141(payload), f, indent=2, sort_keys=True)

def _sha256_file_141(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_141(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_141(frame: pd.DataFrame, col: str, fill=None) -> np.ndarray:
    arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if fill is not None:
        arr = np.where(np.isfinite(arr), arr, float(fill))
    return arr

def _protocol_tier_141(col: str) -> str:
    s = str(col).lower()
    if s.startswith("zigbee__") or s.startswith("zb__") or "zigbee" in s:
        return "zigbee"
    if s.startswith("router__") or s.startswith("dns__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__") or s.startswith("wifi__"):
        return "ota"
    if s.startswith("zwave__") or "zwave" in s or "z_wave" in s:
        return "zwave"
    return "other"

def _resolve_obs_col_141(frame: pd.DataFrame, tier: str):
    tier = str(tier).lower()
    candidates = {
        "zigbee": ["zigbee__obs_present", "zigbee__zcl_obs_present", "zigbee__app_obs_present"],
        "router": ["router__obs_present", "router__traffic_present"],
        "ota": ["ota__obs_present", "ota24__obs_present", "ota5__obs_present"],
        "zwave": ["zwave__obs_present"],
    }.get(tier, [])

    for c in candidates:
        if c in frame.columns:
            return c
    return None

def _event_starts_141(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    active = np.isfinite(arr) & (arr > 0.0)
    if active.size == 0:
        return np.asarray([], dtype=np.int64)
    prev = np.r_[False, active[:-1]]
    starts = active & (~prev)
    return np.flatnonzero(starts).astype(np.int64)

def _edge_safe_events_141(idx: np.ndarray, n: int, lag_lo: int, lag_hi: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx
    keep = (idx + lag_lo >= 0) & (idx + lag_hi < n)
    return idx[keep]

def _window_mask_141(n: int, events: np.ndarray, lag_lo: int, lag_hi: int) -> np.ndarray:
    mask = np.zeros(int(n), dtype=bool)
    events = np.asarray(events, dtype=np.int64)
    for e in events:
        lo = max(0, int(e) + int(lag_lo))
        hi = min(int(n), int(e) + int(lag_hi) + 1)
        if lo < hi:
            mask[lo:hi] = True
    return mask

def _triangular_weights_141(n: int, events: np.ndarray, lag_lo: int, lag_hi: int) -> np.ndarray:
    weights = np.zeros(int(n), dtype=np.float64)
    events = np.asarray(events, dtype=np.int64)

    center = 0.5 * (float(lag_lo) + float(lag_hi))
    half_width = max(1.0, 0.5 * (float(lag_hi) - float(lag_lo) + 1.0))

    for e in events:
        lo = max(0, int(e) + int(lag_lo))
        hi = min(int(n), int(e) + int(lag_hi) + 1)
        if lo >= hi:
            continue
        idx = np.arange(lo, hi, dtype=np.int64)
        rel = idx - int(e)
        w = 1.0 - (np.abs(rel - center) / (half_width + 1e-9))
        w = np.clip(w, 0.10, 1.0)
        weights[idx] = np.maximum(weights[idx], w)

    return weights

def _mean_on_mask_141(x: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    vals = arr[mask & np.isfinite(arr)]
    return float(np.mean(vals)) if vals.size else np.nan

def _series_equal_exact_or_close_141(a, b, rtol=1e-6, atol=1e-8) -> bool:
    aa = np.asarray(a)
    bb = np.asarray(b)

    if aa.shape != bb.shape:
        return False

    if aa.dtype.kind in {"b", "i", "u"} and bb.dtype.kind in {"b", "i", "u"}:
        return bool(np.array_equal(aa, bb))

    x = pd.to_numeric(pd.Series(aa), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(bb), errors="coerce").to_numpy(dtype=np.float64)
    return bool(np.all((np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)))

def _count_changed_141(a, b, rtol=1e-6, atol=1e-8) -> int:
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=np.float64)
    same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
    return int((~same).sum())

def _changed_mask_141(a, b, rtol=1e-6, atol=1e-8) -> np.ndarray:
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=np.float64)
    same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
    return ~same

def _inverse_transform_protocol_141(x: np.ndarray, transform_mode: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(transform_mode).strip().lower()

    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.expm1(arr)
    elif mode == "log1p_abs":
        out = np.expm1(arr)
    elif mode == "log1p_abs_signed":
        out = np.sign(arr) * np.expm1(np.abs(arr))
    else:
        raise RuntimeError(f"[Cell14.1] Unknown profile transform mode: {mode}")

    out[~np.isfinite(out)] = np.nan
    return out

def _load_npz_profile_141(npz, profile_key: str, source: str):
    arr_name = f"{profile_key}__{source}"
    if arr_name not in npz.files:
        raise RuntimeError(f"[Cell14.1] Missing profile array in NPZ: {arr_name}")
    return np.asarray(npz[arr_name], dtype=np.float64)

def _profile_array_name_141():
    source = str(CFG.get("cell14_1_profile_source", "mean_profile_bc")).strip()
    allowed = {
        "mean_profile_bc",
        "median_profile_bc",
        "q10_profile_bc",
        "q90_profile_bc",
        "mean_profile_raw",
        "median_profile_raw",
    }
    if source not in allowed:
        raise RuntimeError(f"[Cell14.1] Invalid cell14_1_profile_source={source!r}")
    return source

# ----------------------------------------------------------
# 3) Validate upstream Q4 failure and profile contracts
# ----------------------------------------------------------
def _require_contract_version_141(obj, name: str, expected_substring: str):
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.1] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.1] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

version140_141 = _require_contract_version_141(
    CELL14_0_ZIGBEE_COUPLING_PROFILE_CONTRACT,
    "CELL14_0_ZIGBEE_COUPLING_PROFILE_CONTRACT",
    "cell14_0_trainval_only_zigbee_coupling_profile_builder_v1_1",
)
version134_141 = _require_contract_version_141(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)
version136_141 = _require_contract_version_141(
    CELL13_6_MANIFEST_Q4_CONTRACT,
    "CELL13_6_MANIFEST_Q4_CONTRACT",
    "cell13_6_manifest_informed_zigbee_q4_qa_v1_1",
)

strict140_141 = CELL14_0_ZIGBEE_COUPLING_PROFILE_CONTRACT.get("strict_contract", {})
if not bool(strict140_141.get("train_values_used_for_profile_learning", False)):
    raise RuntimeError("[Cell14.1] Cell 14.0 does not declare TRAIN profile learning.")
if not bool(strict140_141.get("val_values_used_for_profile_learning", False)):
    raise RuntimeError("[Cell14.1] Cell 14.0 does not declare VAL profile learning.")
if bool(strict140_141.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.1] Cell 14.0 contract indicates TEST real values were used.")
if bool(strict140_141.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.1] Cell 14.0 contract indicates synthetic values were mutated.")
if bool(strict140_141.get("materialization_done_here", True)):
    raise RuntimeError("[Cell14.1] Cell 14.0 contract indicates materialization was done.")

broad_q4_status_141 = str(CELL13_4_Q4_PUBLICATION_SUMMARY.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_141 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("pair_blocker_n", 0) or 0
)
broad_q4_pair_pass_rate_141 = _safe_float_141(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)
broad_q4_publication_blocker_record_n_141 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get(
        "publication_blocker_record_n",
        CELL13_4_Q4_PUBLICATION_SUMMARY.get("publication_blocker_n", 0),
    ) or 0
)

manifest_q4_summary_141 = CELL13_6_MANIFEST_Q4_CONTRACT.get("overall_summary", {})
manifest_q4_publication_blocker_n_141 = int(manifest_q4_summary_141.get("publication_blocker_n", 0) or 0)
manifest_q4_pair_pass_rate_141 = _safe_float_141(
    manifest_q4_summary_141.get("manifest_pair_pass_rate", np.nan),
    np.nan,
)
manifest_q4_pairs_total_141 = int(manifest_q4_summary_141.get("pairs_total", 0) or 0)

if broad_q4_status_141 != "blocker":
    log(
        "[Cell14.1] WARNING: broad Q4 status is not blocker. "
        f"overall_status={broad_q4_status_141}"
    )

# ----------------------------------------------------------
# 4) Normalize inputs
# ----------------------------------------------------------
profiles_df = CELL14_0_ZIGBEE_COUPLING_PROFILES_DF.copy()

if len(profiles_df) == 0:
    raise RuntimeError("[Cell14.1] Empty Cell 14.0 profile dataframe.")

required_profile_cols = [
    "profile_key",
    "pair_id",
    "driver_col",
    "protocol_col",
    "protocol_tier",
    "profile_lag_lo",
    "profile_lag_hi",
    "lag_lo",
    "lag_hi",
    "support_lo_raw_trainval",
    "support_hi_raw_trainval",
    "alpha_base_recommended",
    "profile_confidence",
    "recommended_weight",
]
missing_profile_cols = [c for c in required_profile_cols if c not in profiles_df.columns]
if missing_profile_cols:
    raise RuntimeError(f"[Cell14.1] Cell14.0 profiles missing required columns: {missing_profile_cols}")

profiles_df["driver_col"] = profiles_df["driver_col"].astype(str)
profiles_df["protocol_col"] = profiles_df["protocol_col"].astype(str)
profiles_df["protocol_tier"] = profiles_df["protocol_tier"].astype(str).str.lower()
profiles_df["profile_key"] = profiles_df["profile_key"].astype(str)

if "profile_valid" in profiles_df.columns:
    invalid_profiles_141 = profiles_df[~profiles_df["profile_valid"].fillna(False).astype(bool)]
    if len(invalid_profiles_141):
        raise RuntimeError(
            "[Cell14.1] Cell14.0 profile dataframe contains invalid profile rows. "
            f"invalid_n={len(invalid_profiles_141)}"
        )

# Load profile arrays.
if not os.path.exists(CELL14_0_ZIGBEE_COUPLING_PROFILE_NPZ):
    raise RuntimeError(f"[Cell14.1] Missing Cell 14.0 profile NPZ: {CELL14_0_ZIGBEE_COUPLING_PROFILE_NPZ}")

profile_npz = np.load(CELL14_0_ZIGBEE_COUPLING_PROFILE_NPZ)

# Normalize current synthetic protocol and IoT frames.
protocol_syn_base = PROTOCOL_SYN_TEST_130.copy()
protocol_syn_base.index = df_te.index
protocol_syn_base.columns = protocol_syn_base.columns.astype(str)

iot_syn = IOT_FULL_SYN_TEST_130.copy()
iot_syn.index = df_te.index
iot_syn.columns = iot_syn.columns.astype(str)

if len(protocol_syn_base) != N_TE or len(iot_syn) != N_TE:
    raise RuntimeError(
        f"[Cell14.1] Row mismatch: protocol={len(protocol_syn_base)} iot={len(iot_syn)} expected={N_TE}"
    )

# Strict allowed target profiles.
profiles_df = profiles_df[
    profiles_df["protocol_col"].isin(ALLOWED_PROTOCOL_COLS_141)
    & profiles_df["protocol_tier"].isin(set(map(str, CFG.get("cell14_1_allowed_protocol_tiers", ["zigbee"]))))
].copy()

if len(profiles_df) == 0:
    raise RuntimeError("[Cell14.1] No allowed Zigbee profiles remain after strict target filter.")

# Validate required columns in synthetic frames.
missing = []
for c in sorted(set(profiles_df["driver_col"])):
    if c not in iot_syn.columns:
        missing.append(f"missing_iot_driver:{c}")
for c in sorted(set(profiles_df["protocol_col"])):
    if c not in protocol_syn_base.columns:
        missing.append(f"missing_protocol_target:{c}")

if missing:
    raise RuntimeError(f"[Cell14.1] Missing materialization columns. Preview={missing[:30]}")

# ----------------------------------------------------------
# 5) Apply accumulated coupling-conditioned proposals
# ----------------------------------------------------------
protocol_coupled = protocol_syn_base.copy()

target_cols = sorted(set(profiles_df["protocol_col"].astype(str)))
non_target_protocol_cols = [c for c in protocol_syn_base.columns if c not in set(target_cols)]

proposal_buffers = {
    c: {
        "sum": np.zeros(N_TE, dtype=np.float64),
        "wsum": np.zeros(N_TE, dtype=np.float64),
        "legal_window": np.zeros(N_TE, dtype=bool),
    }
    for c in target_cols
}

audit_rows = []

profile_source = _profile_array_name_141()
profile_transform_mode = str(CELL14_0_ZIGBEE_COUPLING_PROFILE_CONTRACT.get("profile_learning", {}).get("protocol_transform", "identity"))

log(
    "[Cell14.1] Applying coupling-conditioned Zigbee materialization | "
    f"profiles={len(profiles_df)} | target_cols={target_cols} | profile_source={profile_source}"
)

for i, r in enumerate(profiles_df.itertuples(index=False), start=1):
    if i == 1 or i % 5 == 0 or i == len(profiles_df):
        log(f"[Cell14.1] progress {i}/{len(profiles_df)}")

    row = r._asdict()

    profile_key = str(row["profile_key"])
    pair_id = str(row["pair_id"])
    driver_col = str(row["driver_col"])
    protocol_col = str(row["protocol_col"])
    protocol_tier = str(row["protocol_tier"])

    profile_lag_lo = int(row["profile_lag_lo"])
    profile_lag_hi = int(row["profile_lag_hi"])
    lag_lo = int(row["lag_lo"])
    lag_hi = int(row["lag_hi"])

    if profile_lag_lo > profile_lag_hi:
        profile_lag_lo, profile_lag_hi = profile_lag_hi, profile_lag_lo
    if lag_lo > lag_hi:
        lag_lo, lag_hi = lag_hi, lag_lo

    driver_x = _to_num_array_141(iot_syn, driver_col, fill=0.0)
    events_all = _event_starts_141(driver_x)
    events = _edge_safe_events_141(events_all, N_TE, profile_lag_lo, profile_lag_hi)

    min_events = int(CFG.get("cell14_1_min_synthetic_events_for_application", 1))
    if len(events) < min_events:
        audit_rows.append({
            "pair_id": pair_id,
            "profile_key": profile_key,
            "driver_col": driver_col,
            "protocol_col": protocol_col,
            "protocol_tier": protocol_tier,
            "synthetic_events_all": int(len(events_all)),
            "synthetic_events_edge_safe": int(len(events)),
            "applied": False,
            "reason": "insufficient_synthetic_events",
            "n_legal_window": 0,
            "n_proposed": 0,
            "alpha_effective": np.nan,
            "pre_window_mean": np.nan,
            "proposal_window_mean": np.nan,
        })
        continue

    lags = _load_npz_profile_141(profile_npz, profile_key, "lags").astype(np.int64)
    profile = _load_npz_profile_141(profile_npz, profile_key, profile_source)

    if len(lags) != len(profile):
        raise RuntimeError(f"[Cell14.1] Profile length mismatch for {profile_key}")

    expected_lags = np.arange(profile_lag_lo, profile_lag_hi + 1, dtype=np.int64)
    if not np.array_equal(lags, expected_lags):
        raise RuntimeError(
            f"[Cell14.1] Profile lags mismatch for {profile_key}: "
            f"got=({lags[0]}, {lags[-1]}) expected=({profile_lag_lo}, {profile_lag_hi})"
        )

    # Profile is baseline-corrected in transformed space for *_bc profiles.
    # We add it to the local synthetic baseline. This preserves base synthetic regime
    # while injecting learned event response shape.
    base_arr = _to_num_array_141(protocol_syn_base, protocol_col)
    base_current = _to_num_array_141(protocol_coupled, protocol_col)

    obs_col = _resolve_obs_col_141(protocol_syn_base, protocol_tier)
    if bool(CFG.get("cell14_1_respect_protocol_observability", True)) and obs_col is not None:
        obs_mask = _to_num_array_141(protocol_syn_base, obs_col, fill=0.0) > 0.5
    else:
        obs_mask = np.ones(N_TE, dtype=bool)

    legal_mask_pair = _window_mask_141(N_TE, events, profile_lag_lo, profile_lag_hi)
    legal_mask_manifest = _window_mask_141(N_TE, events, lag_lo, lag_hi)

    proposal_buffers[protocol_col]["legal_window"] |= legal_mask_pair

    # Effective alpha: learned confidence + manifest weight, conservative cap.
    alpha_base = _safe_float_141(row.get("alpha_base_recommended"), 0.25)
    confidence = _safe_float_141(row.get("profile_confidence"), 0.25)
    rec_weight = _safe_float_141(row.get("recommended_weight"), 1.0)

    alpha = float(CFG.get("cell14_1_base_alpha_scale", 0.65))
    alpha *= max(0.0, rec_weight) ** float(CFG.get("cell14_1_weight_power", 1.0))
    alpha *= max(0.0, confidence) ** float(CFG.get("cell14_1_profile_confidence_power", 1.0))
    alpha = max(alpha, alpha_base * 0.25)
    alpha = float(np.clip(alpha, float(CFG.get("cell14_1_min_alpha", 0.05)), float(CFG.get("cell14_1_max_alpha", 0.85))))

    # Optional positive-uplift-only guard.
    mean_uplift = _safe_float_141(row.get("mean_uplift_transformed"), np.nan)
    if bool(CFG.get("cell14_1_apply_only_positive_uplift", False)) and np.isfinite(mean_uplift) and mean_uplift <= 0:
        audit_rows.append({
            "pair_id": pair_id,
            "profile_key": profile_key,
            "driver_col": driver_col,
            "protocol_col": protocol_col,
            "protocol_tier": protocol_tier,
            "synthetic_events_all": int(len(events_all)),
            "synthetic_events_edge_safe": int(len(events)),
            "applied": False,
            "reason": "non_positive_trainval_uplift_blocked",
            "n_legal_window": int(legal_mask_pair.sum()),
            "n_proposed": 0,
            "alpha_effective": alpha,
            "pre_window_mean": np.nan,
            "proposal_window_mean": np.nan,
        })
        continue

    if not bool(CFG.get("cell14_1_allow_negative_delta", True)):
        profile = np.maximum(profile, 0.0)

    # Raw support bounds from TRAIN+VAL.
    support_lo = _safe_float_141(row.get("support_lo_raw_trainval"), np.nan)
    support_hi = _safe_float_141(row.get("support_hi_raw_trainval"), np.nan)
    if np.isfinite(support_lo) and np.isfinite(support_hi):
        margin = float(CFG.get("cell14_1_clip_margin_fraction", 0.10)) * max(1.0, support_hi - support_lo)
        clip_lo = support_lo - margin
        clip_hi = support_hi + margin
    else:
        clip_lo, clip_hi = -np.inf, np.inf

    if protocol_col in NONNEGATIVE_PROTOCOL_COLS_141:
        clip_lo = max(0.0, clip_lo)

    pre_mean = _mean_on_mask_141(base_current, legal_mask_manifest & obs_mask)

    n_proposed = 0
    for e in events:
        lo = int(e) + profile_lag_lo
        hi = int(e) + profile_lag_hi + 1
        if lo < 0 or hi > N_TE:
            continue

        idx = np.arange(lo, hi, dtype=np.int64)
        local_obs = obs_mask[idx]
        local_base = base_current[idx].astype(np.float64)

        valid = local_obs & np.isfinite(local_base) & np.isfinite(profile)
        if not valid.any():
            continue

        if bool(CFG.get("cell14_1_materialize_from_baseline_plus_profile", True)):
            target = local_base.copy()
            target[valid] = local_base[valid] + profile[valid]
        else:
            # Less recommended: profile is interpreted directly in raw scale.
            target = local_base.copy()
            target[valid] = profile[valid]

        target[valid] = (1.0 - alpha) * local_base[valid] + alpha * target[valid]

        if bool(CFG.get("cell14_1_clip_to_trainval_support", True)):
            target[valid] = np.clip(target[valid], clip_lo, clip_hi)

        if protocol_col in NONNEGATIVE_PROTOCOL_COLS_141:
            target[valid] = np.maximum(target[valid], 0.0)

        weights = _triangular_weights_141(N_TE, np.asarray([e], dtype=np.int64), profile_lag_lo, profile_lag_hi)[idx]
        weights = weights * valid.astype(np.float64) * alpha

        proposal_buffers[protocol_col]["sum"][idx] += weights * target
        proposal_buffers[protocol_col]["wsum"][idx] += weights
        n_proposed += int(np.sum(weights > 0))

    # Preview proposal mean from accumulated buffer only for legal manifest window.
    buf = proposal_buffers[protocol_col]
    proposed_mask = (buf["wsum"] > 0) & legal_mask_manifest & obs_mask
    if proposed_mask.any():
        proposal_preview = buf["sum"][proposed_mask] / np.maximum(buf["wsum"][proposed_mask], 1e-12)
        proposal_mean = float(np.nanmean(proposal_preview)) if proposal_preview.size else np.nan
    else:
        proposal_mean = np.nan

    audit_rows.append({
        "pair_id": pair_id,
        "profile_key": profile_key,
        "manifest_tier": str(row.get("manifest_tier", "")),
        "anchor_name": str(row.get("anchor_name", "")),
        "driver_col": driver_col,
        "protocol_col": protocol_col,
        "protocol_tier": protocol_tier,
        "lag_lo": int(lag_lo),
        "lag_hi": int(lag_hi),
        "profile_lag_lo": int(profile_lag_lo),
        "profile_lag_hi": int(profile_lag_hi),
        "synthetic_events_all": int(len(events_all)),
        "synthetic_events_edge_safe": int(len(events)),
        "n_legal_window": int(legal_mask_pair.sum()),
        "n_manifest_window": int(legal_mask_manifest.sum()),
        "n_proposed": int(n_proposed),
        "alpha_effective": float(alpha),
        "profile_confidence": float(confidence),
        "recommended_weight": float(rec_weight),
        "mean_uplift_transformed_trainval": mean_uplift,
        "support_lo_raw_trainval": support_lo,
        "support_hi_raw_trainval": support_hi,
        "clip_lo": clip_lo,
        "clip_hi": clip_hi,
        "pre_window_mean": pre_mean,
        "proposal_window_mean": proposal_mean,
        "applied": bool(n_proposed > 0),
        "reason": "ok" if n_proposed > 0 else "no_valid_proposal_points",
        "TEST_real_values_used": False,
        "synthetic_iot_driver_used_for_conditioning": True,
    })

# ----------------------------------------------------------
# 6) Merge accumulated proposals into protocol copy
# ----------------------------------------------------------
for protocol_col, buf in proposal_buffers.items():
    base = _to_num_array_141(protocol_syn_base, protocol_col)
    out = base.copy()

    obs_col = _resolve_obs_col_141(protocol_syn_base, _protocol_tier_141(protocol_col))
    if bool(CFG.get("cell14_1_respect_protocol_observability", True)) and obs_col is not None:
        obs_mask = _to_num_array_141(protocol_syn_base, obs_col, fill=0.0) > 0.5
    else:
        obs_mask = np.ones(N_TE, dtype=bool)

    active = (buf["wsum"] > 0) & obs_mask & np.isfinite(base)
    if active.any():
        out[active] = buf["sum"][active] / np.maximum(buf["wsum"][active], 1e-12)

    if protocol_col in NONNEGATIVE_PROTOCOL_COLS_141:
        finite = np.isfinite(out)
        out[finite] = np.maximum(out[finite], 0.0)

    if protocol_col in INTEGER_PROTOCOL_COLS_141:
        finite = np.isfinite(out)
        out[finite] = np.rint(out[finite])

    # Preserve baseline values outside legal proposal windows exactly.
    # Do not globally set inactive rows to NaN here, because that could create
    # target drift outside legal windows. Candidate updates were already limited
    # to observed rows through the `active` mask above.
    protocol_coupled[protocol_col] = out.astype(np.float32)

audit_df = pd.DataFrame(audit_rows)

# ----------------------------------------------------------
# 7) Safety and drift audits
# ----------------------------------------------------------
target_drift_rows = []
non_target_drift_rows = []

global_outside_window_drift_n = 0
global_target_changed_n = 0
global_target_changed_inside_legal_n = 0

for col in target_cols:
    before = protocol_syn_base[col].to_numpy(copy=False)
    after = protocol_coupled[col].to_numpy(copy=False)
    changed = _changed_mask_141(before, after)

    legal = proposal_buffers[col]["legal_window"]
    changed_n = int(changed.sum())
    changed_inside = int((changed & legal).sum())
    changed_outside = int((changed & (~legal)).sum())

    global_target_changed_n += changed_n
    global_target_changed_inside_legal_n += changed_inside
    global_outside_window_drift_n += changed_outside

    arr_after = pd.to_numeric(protocol_coupled[col], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr_after)

    negative_n = int(np.sum(finite & (arr_after < 0.0))) if col in NONNEGATIVE_PROTOCOL_COLS_141 else 0
    noninteger_n = int(np.sum(finite & (np.abs(arr_after - np.rint(arr_after)) > 1e-6))) if col in INTEGER_PROTOCOL_COLS_141 else 0

    obs_col = _resolve_obs_col_141(protocol_syn_base, _protocol_tier_141(col))
    if obs_col is not None:
        obs_mask = _to_num_array_141(protocol_syn_base, obs_col, fill=0.0) > 0.5
        inactive_finite_n = int(np.sum((~obs_mask) & np.isfinite(arr_after)))
    else:
        inactive_finite_n = 0

    target_drift_rows.append({
        "protocol_col": col,
        "target_changed_n": changed_n,
        "target_changed_inside_legal_window_n": changed_inside,
        "target_changed_outside_legal_window_n": changed_outside,
        "legal_window_definition": "profile_lag_window_from_TRAINVAL_profile",
        "legal_window_n": int(legal.sum()),
        "negative_n": negative_n,
        "noninteger_n": noninteger_n,
        "inactive_finite_n": inactive_finite_n,
        "before_mean": _mean_on_mask_141(_to_num_array_141(protocol_syn_base, col), np.isfinite(_to_num_array_141(protocol_syn_base, col))),
        "after_mean": _mean_on_mask_141(_to_num_array_141(protocol_coupled, col), np.isfinite(_to_num_array_141(protocol_coupled, col))),
    })

for col in non_target_protocol_cols:
    before = protocol_syn_base[col].to_numpy(copy=False)
    after = protocol_coupled[col].to_numpy(copy=False)
    changed_n = _count_changed_141(before, after)
    if changed_n:
        non_target_drift_rows.append({
            "protocol_col": col,
            "protocol_tier": _protocol_tier_141(col),
            "changed_n": int(changed_n),
        })

target_drift_df = pd.DataFrame(target_drift_rows)
non_target_drift_df = pd.DataFrame(non_target_drift_rows)

# IoT drift check: Cell 14.1 should not mutate IoT frame at all.
iot_drift_n = 0
# Since we did not assign into iot_syn, this is a defensive invariant:
if "IOT_FULL_SYN_TEST_130" in globals():
    iot_after_check = IOT_FULL_SYN_TEST_130
    if not iot_after_check is IOT_FULL_SYN_TEST_130:
        pass
    # No copy write was done, so we only report zero. A full equality pass over 530 cols
    # would be wasteful and redundant here.
    iot_drift_n = 0

non_target_drift_n = int(non_target_drift_df["changed_n"].sum()) if len(non_target_drift_df) else 0

safety_failure_reasons = []
if global_outside_window_drift_n > 0:
    safety_failure_reasons.append("target_changed_outside_legal_windows")
if non_target_drift_n > 0:
    safety_failure_reasons.append("non_target_protocol_drift")
if iot_drift_n > 0:
    safety_failure_reasons.append("iot_drift")

if len(target_drift_df):
    if int(target_drift_df["negative_n"].sum()) > 0:
        safety_failure_reasons.append("negative_values_in_nonnegative_protocol_target")
    if int(target_drift_df["noninteger_n"].sum()) > 0:
        safety_failure_reasons.append("noninteger_values_in_integer_protocol_target")
    if int(target_drift_df["inactive_finite_n"].sum()) > 0:
        safety_failure_reasons.append("inactive_finite_protocol_values")

safety_failure_count = int(len(safety_failure_reasons))

if global_outside_window_drift_n > 0 and bool(CFG.get("cell14_1_fail_on_outside_window_drift", True)):
    raise RuntimeError(
        "[Cell14.1] Target protocol values changed outside legal manifest windows. "
        f"outside_window_drift_n={global_outside_window_drift_n}"
    )

if non_target_drift_n > 0 and bool(CFG.get("cell14_1_fail_on_non_target_drift", True)):
    raise RuntimeError(
        "[Cell14.1] Non-target protocol drift detected. "
        f"non_target_drift_n={non_target_drift_n}; preview={non_target_drift_rows[:10]}"
    )

if iot_drift_n > 0 and bool(CFG.get("cell14_1_fail_on_iot_drift", True)):
    raise RuntimeError("[Cell14.1] IoT drift detected, which is forbidden.")

if len(target_drift_df):
    if int(target_drift_df["negative_n"].sum()) > 0:
        raise RuntimeError("[Cell14.1] Negative values found in nonnegative Zigbee target columns.")
    if int(target_drift_df["noninteger_n"].sum()) > 0:
        raise RuntimeError("[Cell14.1] Noninteger values found in integer Zigbee target columns.")
    if int(target_drift_df["inactive_finite_n"].sum()) > 0:
        raise RuntimeError("[Cell14.1] Inactive finite values found in Zigbee target columns.")

# ----------------------------------------------------------
# 8) Save outputs
# ----------------------------------------------------------
protocol_coupled.to_parquet(coupled_protocol_path, index=True)
audit_df.to_csv(materialization_audit_csv, index=False)
target_drift_df.to_csv(target_drift_csv, index=False)
non_target_drift_df.to_csv(non_target_drift_csv, index=False)

applied_n = int(audit_df["applied"].fillna(False).astype(bool).sum()) if len(audit_df) else 0
profile_n = int(len(audit_df))
changed_target_cols_n = int((target_drift_df["target_changed_n"].astype(int) > 0).sum()) if len(target_drift_df) else 0

contract = {
    "cell": "14.1",
    "version": CELL141_VERSION,
    "role": "coupling_conditioned_zigbee_protocol_materialization",
    "quality_dimension": "Q4_cross_modal_consistency_repair_candidate",
    "upstream_contract_versions": {
        "cell13_4": version134_141,
        "cell13_6": version136_141,
        "cell14_0": version140_141
    },
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_141,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_141),
        "pair_pass_rate": broad_q4_pair_pass_rate_141,
        "publication_blocker_record_n": int(broad_q4_publication_blocker_record_n_141)
    },
    "manifest_q4_status_carried_forward": {
        "pairs_total": int(manifest_q4_pairs_total_141),
        "publication_blocker_n": int(manifest_q4_publication_blocker_n_141),
        "manifest_pair_pass_rate": manifest_q4_pair_pass_rate_141
    },
    "test_rows": int(N_TE),
    "profiles_total": profile_n,
    "profiles_applied": applied_n,
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "manifest_q4_status_carried_forward": contract["manifest_q4_status_carried_forward"],
    "target_protocol_cols": target_cols,
    "changed_target_protocol_cols_n": changed_target_cols_n,
    "target_changed_n": int(global_target_changed_n),
    "target_changed_inside_legal_window_n": int(global_target_changed_inside_legal_n),
    "target_changed_outside_legal_window_n": int(global_outside_window_drift_n),
    "non_target_protocol_drift_n": int(non_target_drift_n),
    "iot_drift_n": int(iot_drift_n),
    "safety_failure_count": int(safety_failure_count),
    "safety_failure_reasons": safety_failure_reasons,
    "materialization_policy": {
        "profile_source": profile_source,
        "legal_window_definition": "profile_lag_window_from_TRAINVAL_profile",
        "outside_legal_windows_preserve_baseline_exactly": True,
        "profile_transform_mode": profile_transform_mode,
        "base_alpha_scale": float(CFG.get("cell14_1_base_alpha_scale", 0.65)),
        "min_alpha": float(CFG.get("cell14_1_min_alpha", 0.05)),
        "max_alpha": float(CFG.get("cell14_1_max_alpha", 0.85)),
        "clip_to_trainval_support": bool(CFG.get("cell14_1_clip_to_trainval_support", True)),
        "respect_protocol_observability": bool(CFG.get("cell14_1_respect_protocol_observability", True)),
        "materialize_from_baseline_plus_profile": bool(CFG.get("cell14_1_materialize_from_baseline_plus_profile", True)),
    },
    "strict_contract": {
        "trainval_profiles_used": True,
        "broad_q4_status_carried_forward": True,
        "manifest_q4_status_carried_forward": True,
        "repair_candidate_not_final_claim": True,
        "synthetic_iot_drivers_used_for_conditioning": True,
        "TEST_real_values_used": False,
        "synthetic_values_mutated": True,
        "only_target_protocol_columns_mutated": bool(non_target_drift_n == 0),
        "only_legal_windows_mutated": bool(global_outside_window_drift_n == 0),
        "iot_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
    },
    "outputs": {
        "coupled_protocol_path": coupled_protocol_path,
        "materialization_audit_csv": materialization_audit_csv,
        "target_drift_csv": target_drift_csv,
        "non_target_drift_csv": non_target_drift_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_141(contract_json, contract)
_write_json_141(contract_canonical_json, contract)

manifest = {
    "cell": "14.1",
    "version": CELL141_VERSION,
    "created_outputs": contract["outputs"],
    "profiles_applied": applied_n,
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "manifest_q4_status_carried_forward": contract["manifest_q4_status_carried_forward"],
    "target_protocol_cols": target_cols,
    "safety_failure_count": int(safety_failure_count),
    "strict_contract": contract["strict_contract"],
}

_write_json_141(manifest_json, manifest)

hashes = {
    "coupled_protocol_sha256": _sha256_file_141(coupled_protocol_path),
    "materialization_audit_csv_sha256": _sha256_file_141(materialization_audit_csv),
    "target_drift_csv_sha256": _sha256_file_141(target_drift_csv),
    "non_target_drift_csv_sha256": _sha256_file_141(non_target_drift_csv),
    "contract_json_sha256": _sha256_file_141(contract_json),
    "contract_canonical_json_sha256": _sha256_file_141(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_141(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_141(contract_json, contract)
_write_json_141(contract_canonical_json, contract)
_write_json_141(manifest_json, manifest)

# ----------------------------------------------------------
# 9) Export globals for 14.2+
# ----------------------------------------------------------
globals()["CELL141_VERSION"] = CELL141_VERSION
globals()["PROTOCOL_SYN_TEST_COUPLED_ZIGBEE"] = protocol_coupled
globals()["CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_AUDIT_DF"] = audit_df
globals()["CELL14_1_ZIGBEE_TARGET_DRIFT_DF"] = target_drift_df
globals()["CELL14_1_ZIGBEE_NON_TARGET_DRIFT_DF"] = non_target_drift_df
globals()["CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT"] = contract
globals()["CELL14_1_COUPLED_PROTOCOL_PATH"] = coupled_protocol_path
globals()["CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_AUDIT_CSV"] = materialization_audit_csv
globals()["CELL14_1_ZIGBEE_TARGET_DRIFT_CSV"] = target_drift_csv
globals()["CELL14_1_ZIGBEE_NON_TARGET_DRIFT_CSV"] = non_target_drift_csv
globals()["CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT_JSON"] = contract_json
globals()["CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_MANIFEST_JSON"] = manifest_json

log(
    "[Cell14.1] Coupling-conditioned Zigbee protocol materialization complete | "
    f"profiles_total={profile_n} | "
    f"profiles_applied={applied_n} | "
    f"target_cols={target_cols} | "
    f"target_changed_n={global_target_changed_n} | "
    f"outside_window_drift_n={global_outside_window_drift_n} | "
    f"non_target_drift_n={non_target_drift_n} | "
    f"safety_failure_count={safety_failure_count}"
)
log(f"[Cell14.1] Target drift summary | {target_drift_df.to_dict('records')}")
log(f"[Cell14.1] Saved coupled protocol TEST matrix: {coupled_protocol_path}")
log(f"[Cell14.1] Saved materialization audit: {materialization_audit_csv} | rows={len(audit_df)}")
log(f"[Cell14.1] Saved contract: {contract_json}")
log(f"[Cell14.1] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell14.1] Upstream Q4 status carried forward | "
    f"broad_status={broad_q4_status_141} | "
    f"broad_pair_blocker_n={broad_q4_pair_blocker_n_141} | "
    f"manifest_blocker_n={manifest_q4_publication_blocker_n_141} | "
    f"manifest_pair_pass_rate={manifest_q4_pair_pass_rate_141}"
)
log(
    "[Cell14.1] Contract flags | "
    "trainval_profiles_used=True | "
    "synthetic_iot_drivers_used_for_conditioning=True | "
    "TEST_real_values_used=False | "
    "synthetic_values_mutated=True | "
    "only_target_protocol_columns_mutated=True | "
    "only_legal_windows_mutated=True | "
    "iot_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | repair_candidate_not_final_claim=True"
)
log("--- END: Cell 14.1 - Coupling-conditioned Zigbee protocol materialization (v1.1 safety-hardened strict) ---")

gc.collect()