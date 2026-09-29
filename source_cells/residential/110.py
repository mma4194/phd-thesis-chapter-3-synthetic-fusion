# ==========================================================
# CELL 14.4 - Two-stage TRAIN/VAL IoT↔protocol coupling discovery and tiering
# v2.1 STUDY-THESIS strict staged cross-modal coupling evidence audit, repair-status-aware
#
# Role:
#   - Discover and rank IoT→protocol coupling candidates across:
#       router
#       OTA / Wi-Fi
#       Zigbee
#   - Stage 1: fast broad screening over many pairs, no null baselines.
#   - Stage 2: deep null-baseline replication only over shortlisted pairs.
#
# Scientific contract:
#   - TRAIN+VAL real data may be used for discovery/tiering.
#   - Real TEST values are not used.
#   - Synthetic values are not mutated.
#   - No generator fitting or materialization happens here.
#
# Outputs:
#   reports/cell14_4_stage1_fast_screen_metrics.csv
#   reports/cell14_4_stage1_fast_screen_shortlist.csv
#   reports/cell14_4_stage2_deep_candidate_metrics.csv
#   reports/cell14_4_deep_coupling_replication_table.csv
#   reports/cell14_4_deep_coupling_tier_summary.csv
#   reports/cell14_4_deep_coupling_tierA_candidates.csv
#   reports/cell14_4_deep_coupling_tierB_candidates.csv
#   reports/cell14_4_deep_coupling_tierC_candidates.csv
#   reports/cell14_4_deep_coupling_contract.json
#   artifacts/cell14_4_deep_coupling_manifest.json
# ==========================================================

log("--- START: Cell 14.4 - Two-stage TRAIN/VAL IoT↔protocol coupling discovery and tiering (v2.1 repair-status-aware strict) ---")

import os
import re
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_144 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "CELL13_6_MANIFEST_Q4_CONTRACT",
    "CELL14_3_Q4_REPAIR_CONTRACT",
]
_missing_144 = [k for k in _required_144 if k not in globals()]
if _missing_144:
    raise RuntimeError(f"[Cell14.4] Missing required globals: {_missing_144}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

if N_TR <= 0 or N_VAL <= 0 or N_TE <= 0:
    raise RuntimeError(
        f"[Cell14.4] Invalid split lengths: train={N_TR}, val={N_VAL}, test={N_TE}"
    )

CELL144_VERSION = "cell14_4_two_stage_trainval_iot_protocol_coupling_tiering_v2_1_repair_status_aware"

CFG["cell14_4_version"] = CELL144_VERSION
CFG["cell14_4_train_values_used_for_discovery"] = True
CFG["cell14_4_val_values_used_for_replication"] = True
CFG["cell14_4_TEST_real_values_used"] = False
CFG["cell14_4_synthetic_values_mutated"] = False
CFG["cell14_4_selection_done_here"] = False
CFG["cell14_4_generator_fit_done_here"] = False
CFG["cell14_4_materialization_done_here"] = False
CFG["cell14_4_coupling_candidate_discovery_done_here"] = True
CFG["cell14_4_coupling_candidate_shortlisting_done_here"] = True
CFG["cell14_4_repair_status_carried_forward"] = True
CFG["cell14_4_broad_q4_status_carried_forward"] = True
CFG["cell14_4_TEST_activity_used_for_discovery"] = False

# ---------------------------
# Discovery scope
# ---------------------------
CFG.setdefault("cell14_4_protocol_tiers", ["router", "ota", "zigbee"])
CFG.setdefault("cell14_4_include_event_driver_anchors", True)
CFG.setdefault("cell14_4_include_binary_state_anchors", True)
CFG.setdefault("cell14_4_max_anchors", 160)
CFG.setdefault("cell14_4_max_protocol_signals", 120)

# ---------------------------
# Event support
# ---------------------------
CFG.setdefault("cell14_4_min_train_anchor_events", 20)
CFG.setdefault("cell14_4_min_val_anchor_events", 5)
CFG.setdefault("cell14_4_max_anchor_events", 30000)

# ---------------------------
# Stage 1 fast screening
# ---------------------------
CFG.setdefault("cell14_4_stage1_lag_min", -10)
CFG.setdefault("cell14_4_stage1_lag_max", 20)
CFG.setdefault("cell14_4_stage1_pre_window", [-10, -1])
CFG.setdefault("cell14_4_stage1_immediate_window", [0, 5])
CFG.setdefault("cell14_4_stage1_delayed_window", [6, 20])
CFG.setdefault("cell14_4_stage1_outer_left_window", [-10, -6])
CFG.setdefault("cell14_4_stage1_outer_right_window", [6, 20])

CFG.setdefault("cell14_4_stage1_min_modality_coverage", 0.20)
CFG.setdefault("cell14_4_stage1_min_nonzero_signal_fraction", 0.005)
CFG.setdefault("cell14_4_stage1_min_protocol_finite_fraction", 0.50)
CFG.setdefault("cell14_4_stage1_min_delta_core", 0.0)
CFG.setdefault("cell14_4_stage1_min_sharpness_core", -np.inf)
CFG.setdefault("cell14_4_stage1_max_lag_abs_diff", 15)

# Shortlist controls.
CFG.setdefault("cell14_4_stage1_top_k_total", 500)
CFG.setdefault("cell14_4_stage1_top_k_per_protocol_tier", 200)
CFG.setdefault("cell14_4_stage1_top_k_per_protocol_col", 50)
CFG.setdefault("cell14_4_stage1_force_include_existing_manifest_pairs", True)

# ---------------------------
# Stage 2 deep replication
# ---------------------------
CFG.setdefault("cell14_4_stage2_lag_min", -30)
CFG.setdefault("cell14_4_stage2_lag_max", 60)
CFG.setdefault("cell14_4_stage2_pre_window", [-30, -1])
CFG.setdefault("cell14_4_stage2_immediate_window", [0, 5])
CFG.setdefault("cell14_4_stage2_delayed_window", [6, 30])
CFG.setdefault("cell14_4_stage2_outer_left_window", [-30, -6])
CFG.setdefault("cell14_4_stage2_outer_right_window", [6, 30])

CFG.setdefault("cell14_4_stage2_n_null_baselines", 12)
CFG.setdefault("cell14_4_stage2_max_null_events", 512)
CFG.setdefault("cell14_4_stage2_null_exclusion_radius", 60)
CFG.setdefault("cell14_4_stage2_null_local_radius", 3600)

# ---------------------------
# Tier thresholds
# ---------------------------
CFG.setdefault("cell14_4_tierA_min_z_immediate", 3.0)
CFG.setdefault("cell14_4_tierA_min_z_sharpness", 2.0)
CFG.setdefault("cell14_4_tierA_min_coverage", 0.50)
CFG.setdefault("cell14_4_tierA_max_lag_abs_diff", 2)
CFG.setdefault("cell14_4_tierA_lag_lo", 0)
CFG.setdefault("cell14_4_tierA_lag_hi", 5)

CFG.setdefault("cell14_4_tierB_min_z_immediate", 2.5)
CFG.setdefault("cell14_4_tierB_min_z_sharpness", 1.5)
CFG.setdefault("cell14_4_tierB_min_coverage", 0.35)
CFG.setdefault("cell14_4_tierB_max_lag_abs_diff", 5)
CFG.setdefault("cell14_4_tierB_lag_lo", 0)
CFG.setdefault("cell14_4_tierB_lag_hi", 10)

CFG.setdefault("cell14_4_tierC_min_z_immediate", 2.0)
CFG.setdefault("cell14_4_tierC_min_coverage", 0.25)
CFG.setdefault("cell14_4_tierC_max_lag_abs_diff", 10)

CFG.setdefault("cell14_4_fail_if_no_candidates", False)

RNG_144 = np.random.default_rng(SEED + 14400)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
stage1_metrics_csv = os.path.join(REPORT_DIR, "cell14_4_stage1_fast_screen_metrics.csv")
stage1_shortlist_csv = os.path.join(REPORT_DIR, "cell14_4_stage1_fast_screen_shortlist.csv")
stage1_rejects_csv = os.path.join(REPORT_DIR, "cell14_4_stage1_fast_screen_rejects.csv")
anchor_support_csv = os.path.join(REPORT_DIR, "cell14_4_anchor_support_audit.csv")
stage2_metrics_csv = os.path.join(REPORT_DIR, "cell14_4_stage2_deep_candidate_metrics.csv")
stage2_rejects_csv = os.path.join(REPORT_DIR, "cell14_4_stage2_deep_candidate_rejects.csv")
replication_table_csv = os.path.join(REPORT_DIR, "cell14_4_deep_coupling_replication_table.csv")
tier_summary_csv = os.path.join(REPORT_DIR, "cell14_4_deep_coupling_tier_summary.csv")
tierA_csv = os.path.join(REPORT_DIR, "cell14_4_deep_coupling_tierA_candidates.csv")
tierB_csv = os.path.join(REPORT_DIR, "cell14_4_deep_coupling_tierB_candidates.csv")
tierC_csv = os.path.join(REPORT_DIR, "cell14_4_deep_coupling_tierC_candidates.csv")
contract_json = os.path.join(REPORT_DIR, "cell14_4_deep_coupling_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_4_deep_coupling_contract_v2_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell14_4_deep_coupling_manifest.json")

# ----------------------------------------------------------
# 2) General helpers
# ----------------------------------------------------------
def _json_sanitize_144(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_144(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_144(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_144(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_144(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_144(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_144(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_144(obj.to_dict())
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

def _write_json_144(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_144(payload), f, indent=2, sort_keys=True)

def _sha256_file_144(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_144(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_144(frame: pd.DataFrame, col: str, fill=None) -> np.ndarray:
    arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if fill is not None:
        arr = np.where(np.isfinite(arr), arr, float(fill))
    return arr

def _tier_of_protocol_144(col: str) -> str:
    s = str(col).lower()
    if s.startswith("router__") or s.startswith("dns__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__") or s.startswith("wifi__") or s.startswith("wlan__"):
        return "ota"
    if s.startswith("zigbee__") or s.startswith("zb__") or "zigbee" in s:
        return "zigbee"
    if s.startswith("zwave__") or "zwave" in s or "z_wave" in s:
        return "zwave"
    return "other"

def _is_protocol_helper_144(col: str) -> bool:
    s = str(col).lower()
    helper_tokens = [
        "__obs_present", "__present", "__traffic_present", "__traffic_obs_present",
        "__stale", "__staleness", "__mask", "__availability", "__available",
    ]
    if any(tok in s for tok in helper_tokens):
        return True
    if s.endswith("_present") or s.endswith("__flag"):
        return True
    return False

def _resolve_obs_col_144(frame: pd.DataFrame, tier: str):
    tier = str(tier).lower()
    candidates = {
        "router": ["router__obs_present", "router__traffic_present"],
        "ota": ["ota__obs_present", "ota24__obs_present", "ota5__obs_present"],
        "zigbee": ["zigbee__obs_present", "zigbee__zcl_obs_present", "zigbee__app_obs_present"],
        "zwave": ["zwave__obs_present"],
    }.get(tier, [])
    for c in candidates:
        if c in frame.columns:
            return c
    return None

def _anchor_name_144(col: str) -> str:
    s = str(col)
    if s.startswith("events_in_sec__entity__"):
        return s.split("events_in_sec__entity__", 1)[1]
    if s.startswith("events_in_sec__feat__"):
        return s.split("events_in_sec__feat__", 1)[1]
    if s.startswith("iot__"):
        s2 = s[5:]
        return s2.split("__", 1)[0]
    return s

def _anchor_type_144(col: str) -> str:
    s = str(col)
    if s.startswith("events_in_sec__entity__"):
        return "event_driver_entity"
    if s.startswith("events_in_sec__feat__"):
        return "event_driver_feature"
    if s.startswith("iot__") and "__binary_sensor__" in s:
        return "iot_binary_sensor_state"
    if s.startswith("iot__") and ("__switch__" in s or "__light__" in s or "__lock__" in s):
        return "iot_binary_state"
    return "iot_other_binary_like"

def _event_starts_144(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    active = np.isfinite(arr) & (arr > 0.0)
    if active.size == 0:
        return np.asarray([], dtype=np.int64)
    prev = np.r_[False, active[:-1]]
    starts = active & (~prev)
    return np.flatnonzero(starts).astype(np.int64)

def _edge_safe_events_144(events: np.ndarray, n: int, lag_min: int, lag_max: int) -> np.ndarray:
    events = np.asarray(events, dtype=np.int64)
    if events.size == 0:
        return events
    keep = (events + lag_min >= 0) & (events + lag_max < n)
    return events[keep]

def _window_mean_per_event_144(signal: np.ndarray, centers: np.ndarray, lo: int, hi: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.int64)

    vals = np.full(len(centers), np.nan, dtype=np.float64)
    n = len(signal)

    for i, c in enumerate(centers):
        a = int(c) + int(lo)
        b = int(c) + int(hi)
        if a < 0 or b >= n:
            continue
        seg = signal[a:b + 1]
        if np.isfinite(seg).any():
            vals[i] = float(np.nanmean(seg))

    return vals

def _coverage_near_144(obs: np.ndarray, centers: np.ndarray, lo: int, hi: int) -> float:
    obs = np.asarray(obs, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.int64)
    vals = []
    n = len(obs)
    for c in centers:
        a = int(c) + int(lo)
        b = int(c) + int(hi)
        if a < 0 or b >= n:
            continue
        seg = obs[a:b + 1]
        vals.append(float(np.nanmean(seg > 0.5)) if len(seg) else np.nan)
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan

def _frac_nonzero_near_144(signal: np.ndarray, centers: np.ndarray, lo=-5, hi=5) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.int64)
    vals = []
    n = len(signal)
    for c in centers:
        a = int(c) + int(lo)
        b = int(c) + int(hi)
        if a < 0 or b >= n:
            continue
        seg = signal[a:b + 1]
        seg = seg[np.isfinite(seg)]
        vals.append(float(np.mean(seg > 0)) if seg.size else np.nan)
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan

def _lag_profile_144(signal: np.ndarray, centers: np.ndarray, lags: np.ndarray) -> dict:
    signal = np.asarray(signal, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.int64)
    out = {}
    n = len(signal)

    for lag in lags:
        idx = centers + int(lag)
        keep = (idx >= 0) & (idx < n)
        if not keep.any():
            out[int(lag)] = np.nan
            continue
        vals = signal[idx[keep]]
        out[int(lag)] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

    return out

def _peak_lag_144(profile: dict):
    items = [(k, v) for k, v in profile.items() if np.isfinite(v)]
    if not items:
        return np.nan, np.nan
    k, v = max(items, key=lambda kv: abs(kv[1]))
    return int(k), float(v)

def _robust_median_144(x) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan

def _robust_mean_144(x, trim=0.10) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size < 10:
        return float(np.mean(arr))
    arr = np.sort(arr)
    lo = int(np.floor(trim * arr.size))
    hi = int(np.ceil((1.0 - trim) * arr.size))
    arr = arr[lo:hi] if hi > lo else arr
    return float(np.mean(arr)) if arr.size else np.nan

def _z_vs_null_144(value, baseline) -> float:
    b = np.asarray(baseline, dtype=np.float64)
    b = b[np.isfinite(b)]
    if b.size < 6:
        return np.nan
    mu = float(np.mean(b))
    sd = float(np.std(b))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.nan
    return float((float(value) - mu) / sd)

def _eligible_null_centers_144(n: int, lag_min: int, lag_max: int) -> np.ndarray:
    return np.arange(max(0, -lag_min), n - max(0, lag_max), dtype=np.int64)

def _remove_event_neighborhoods_144(base_idx: np.ndarray, events: np.ndarray, radius: int) -> np.ndarray:
    if len(events) == 0 or len(base_idx) == 0:
        return base_idx
    keep = np.ones(len(base_idx), dtype=bool)
    for e in np.asarray(events, dtype=np.int64):
        keep &= ~((base_idx >= e - radius) & (base_idx <= e + radius))
    return base_idx[keep]

def _sample_null_centers_144(pool: np.ndarray, n_needed: int, ref_centers: np.ndarray) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    ref_centers = np.asarray(ref_centers, dtype=np.int64)

    if pool.size == 0 or n_needed <= 0:
        return np.asarray([], dtype=np.int64)

    local_radius = int(CFG.get("cell14_4_stage2_null_local_radius", 3600))
    out = []

    for t in ref_centers[:n_needed]:
        local = pool[(pool >= int(t) - local_radius) & (pool <= int(t) + local_radius)]
        src = local if local.size else pool
        out.append(int(src[RNG_144.integers(0, len(src))]))

    return np.asarray(sorted(out), dtype=np.int64)

def _event_scores_144(signal: np.ndarray, centers: np.ndarray, pre, imm, dly, outer_left, outer_right):
    pre_v = _window_mean_per_event_144(signal, centers, *pre)
    imm_v = _window_mean_per_event_144(signal, centers, *imm)
    dly_v = _window_mean_per_event_144(signal, centers, *dly)
    ol_v = _window_mean_per_event_144(signal, centers, *outer_left)
    or_v = _window_mean_per_event_144(signal, centers, *outer_right)

    outer = np.nanmean(np.vstack([ol_v, or_v]), axis=0)

    return {
        "pre": pre_v,
        "immediate": imm_v,
        "delayed": dly_v,
        "outer": outer,
        "delta_immediate": imm_v - pre_v,
        "delta_delayed": dly_v - pre_v,
        "sharpness": imm_v - outer,
    }

# ----------------------------------------------------------
# 3) Validate upstream Q4 / repair status contracts
# ----------------------------------------------------------
def _require_contract_version_144(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.4] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.4] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

version134_144 = _require_contract_version_144(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)
version136_144 = _require_contract_version_144(
    CELL13_6_MANIFEST_Q4_CONTRACT,
    "CELL13_6_MANIFEST_Q4_CONTRACT",
    "cell13_6_manifest_informed_zigbee_q4_qa_v1_1",
)
version143_144 = _require_contract_version_144(
    CELL14_3_Q4_REPAIR_CONTRACT,
    "CELL14_3_Q4_REPAIR_CONTRACT",
    "cell14_3_coupling_conditioned_q4_re_evaluation_v1_1",
)

strict143_144 = CELL14_3_Q4_REPAIR_CONTRACT.get("strict_contract", {})
if bool(strict143_144.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.4] Cell 14.3 contract indicates synthetic values were mutated.")
if bool(strict143_144.get("selection_done_here", True)):
    raise RuntimeError("[Cell14.4] Cell 14.3 contract indicates selection was done.")
if bool(strict143_144.get("generator_fit_done_here", True)):
    raise RuntimeError("[Cell14.4] Cell 14.3 contract indicates generator fitting was done.")
if bool(strict143_144.get("materialization_done_here", True)):
    raise RuntimeError("[Cell14.4] Cell 14.3 contract indicates materialization was done.")
if not bool(strict143_144.get("repair_candidate_not_final_claim", False)):
    raise RuntimeError("[Cell14.4] Cell 14.3 did not declare repair_candidate_not_final_claim.")

broad_q4_status_144 = str(CELL13_4_Q4_PUBLICATION_SUMMARY.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_144 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("pair_blocker_n", 0) or 0
)
broad_q4_pair_pass_rate_144 = _safe_float_144(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)

pre_manifest_summary_144 = CELL13_6_MANIFEST_Q4_CONTRACT.get("overall_summary", {})
pre_manifest_blocker_n_144 = int(pre_manifest_summary_144.get("publication_blocker_n", 0) or 0)
pre_manifest_pass_rate_144 = _safe_float_144(
    pre_manifest_summary_144.get("manifest_pair_pass_rate", np.nan),
    np.nan,
)

repair_status_144 = str(CELL14_3_Q4_REPAIR_CONTRACT.get("repair_candidate_publication_status", ""))
repair_reasons_144 = CELL14_3_Q4_REPAIR_CONTRACT.get("repair_candidate_publication_reasons", [])
repair_post_summary_144 = CELL14_3_Q4_REPAIR_CONTRACT.get("post_coupling_summary_14_3", {})
repair_delta_summary_144 = CELL14_3_Q4_REPAIR_CONTRACT.get("delta_summary", {})
repair_post_blocker_n_144 = int(repair_post_summary_144.get("publication_blocker_n", 0) or 0)
repair_post_pass_rate_144 = _safe_float_144(repair_post_summary_144.get("pass_rate", np.nan), np.nan)

if broad_q4_status_144 != "blocker":
    log(
        "[Cell14.4] WARNING: broad Q4 status is not blocker. "
        f"overall_status={broad_q4_status_144}"
    )

# ----------------------------------------------------------
# 4) Discovery helpers
# ----------------------------------------------------------
def _discover_anchor_cols_144():
    cols = []

    if bool(CFG.get("cell14_4_include_event_driver_anchors", True)):
        cols.extend([c for c in df_tr.columns if str(c).startswith("events_in_sec__entity__")])
        cols.extend([c for c in df_tr.columns if str(c).startswith("events_in_sec__feat__")])

    if bool(CFG.get("cell14_4_include_binary_state_anchors", True)):
        if "IOT_BINARY_COLS" in globals():
            try:
                cols.extend([str(c) for c in list(IOT_BINARY_COLS)])
            except Exception:
                pass

        for c in df_tr.columns:
            s = str(c)
            if not s.startswith("iot__"):
                continue
            if not (s.endswith("__state") or s.endswith("__value")):
                continue
            if any(tok in s for tok in ["__binary_sensor__", "__switch__", "__light__", "__lock__", "__remote__", "__media_player__"]):
                cols.append(s)

    schema_common = set(map(str, df_tr.columns)) & set(map(str, df_val.columns))
    cols = [str(c) for c in dict.fromkeys(cols) if str(c) in schema_common]

    kept_rows = []
    min_tr = int(CFG.get("cell14_4_min_train_anchor_events", 20))
    min_val = int(CFG.get("cell14_4_min_val_anchor_events", 5))
    max_ev = int(CFG.get("cell14_4_max_anchor_events", 30000))

    for c in cols:
        xtr = _to_num_array_144(df_tr, c, fill=0.0)
        xva = _to_num_array_144(df_val, c, fill=0.0)

        evtr = _event_starts_144(xtr)
        evva = _event_starts_144(xva)

        ntr = int(len(evtr))
        nva = int(len(evva))

        keep = bool(ntr >= min_tr and nva >= min_val and ntr <= max_ev and nva <= max_ev)

        kept_rows.append({
            "anchor_col": c,
            "anchor_type": _anchor_type_144(c),
            "anchor_name": _anchor_name_144(c),
            "train_events": ntr,
            "val_events": nva,
            "kept": keep,
        })

    support_df = pd.DataFrame(kept_rows)

    if len(support_df):
        support_df["rank_type"] = support_df["anchor_type"].map({
            "event_driver_entity": 0,
            "event_driver_feature": 1,
            "iot_binary_sensor_state": 2,
            "iot_binary_state": 3,
        }).fillna(9)

        support_df = support_df.sort_values(
            ["kept", "rank_type", "train_events", "val_events"],
            ascending=[False, True, False, False],
        )

        max_anchors = int(CFG.get("cell14_4_max_anchors", 160))
        anchors = support_df.loc[support_df["kept"].astype(bool), "anchor_col"].astype(str).head(max_anchors).tolist()
    else:
        anchors = []

    return anchors, support_df

def _is_likely_protocol_signal_144(frame: pd.DataFrame, col: str) -> bool:
    if col not in frame.columns:
        return False

    tier = _tier_of_protocol_144(col)
    if tier not in set(map(str, CFG.get("cell14_4_protocol_tiers", ["router", "ota", "zigbee"]))):
        return False

    if _is_protocol_helper_144(col):
        return False

    try:
        x = pd.to_numeric(frame[col], errors="coerce")
        finite_frac = float(x.notna().mean())
        return finite_frac >= float(CFG.get("cell14_4_stage1_min_protocol_finite_fraction", 0.50))
    except Exception:
        return False

def _discover_protocol_cols_144():
    schema_common = set(map(str, df_tr.columns)) & set(map(str, df_val.columns))
    protocol_cols = []

    for c in sorted(schema_common):
        if _is_likely_protocol_signal_144(df_tr, c) and _is_likely_protocol_signal_144(df_val, c):
            protocol_cols.append(c)

    protocol_cols = sorted(
        protocol_cols,
        key=lambda c: (
            {"zigbee": 0, "router": 1, "ota": 2}.get(_tier_of_protocol_144(c), 9),
            c,
        )
    )

    max_signals = int(CFG.get("cell14_4_max_protocol_signals", 120))
    return protocol_cols[:max_signals]

# ----------------------------------------------------------
# 5) Stage 1 fast analysis
# ----------------------------------------------------------
def _fast_split_pair_144(split_name, frame, anchor_col, protocol_col):
    lag_min = int(CFG.get("cell14_4_stage1_lag_min", -10))
    lag_max = int(CFG.get("cell14_4_stage1_lag_max", 20))
    lags = np.arange(lag_min, lag_max + 1, dtype=np.int64)

    pre = tuple(map(int, CFG.get("cell14_4_stage1_pre_window", [-10, -1])))
    imm = tuple(map(int, CFG.get("cell14_4_stage1_immediate_window", [0, 5])))
    dly = tuple(map(int, CFG.get("cell14_4_stage1_delayed_window", [6, 20])))
    outer_left = tuple(map(int, CFG.get("cell14_4_stage1_outer_left_window", [-10, -6])))
    outer_right = tuple(map(int, CFG.get("cell14_4_stage1_outer_right_window", [6, 20])))

    tier = _tier_of_protocol_144(protocol_col)
    obs_col = _resolve_obs_col_144(frame, tier)

    if obs_col is None:
        return None, "missing_obs_col"

    anchor_x = _to_num_array_144(frame, anchor_col, fill=0.0)
    signal = _to_num_array_144(frame, protocol_col, fill=0.0)
    obs = _to_num_array_144(frame, obs_col, fill=0.0)

    events_all = _event_starts_144(anchor_x)
    events = _edge_safe_events_144(events_all, len(frame), lag_min, lag_max)

    min_events = int(CFG.get("cell14_4_min_train_anchor_events", 20)) if split_name == "train" else int(CFG.get("cell14_4_min_val_anchor_events", 5))
    if len(events) < min_events:
        return None, "insufficient_events"

    cov = _coverage_near_144(obs, events, lag_min, lag_max)
    if np.isfinite(cov) and cov < float(CFG.get("cell14_4_stage1_min_modality_coverage", 0.20)):
        return None, "low_modality_coverage"

    frac_nonzero = _frac_nonzero_near_144(signal, events, -5, 5)
    if np.isfinite(frac_nonzero) and frac_nonzero < float(CFG.get("cell14_4_stage1_min_nonzero_signal_fraction", 0.005)):
        return None, "low_signal_activity_near_anchor"

    ev = _event_scores_144(signal, events, pre, imm, dly, outer_left, outer_right)
    delta_imm_med = _robust_median_144(ev["delta_immediate"])
    delta_dly_med = _robust_median_144(ev["delta_delayed"])
    sharp_med = _robust_median_144(ev["sharpness"])

    delta_imm_mean = _robust_mean_144(ev["delta_immediate"])
    sharp_mean = _robust_mean_144(ev["sharpness"])

    profile = _lag_profile_144(signal, events, lags)
    peak_lag, peak_value = _peak_lag_144(profile)

    return {
        "split": split_name,
        "anchor_col": anchor_col,
        "anchor_name": _anchor_name_144(anchor_col),
        "anchor_type": _anchor_type_144(anchor_col),
        "protocol_col": protocol_col,
        "protocol_tier": tier,
        "obs_col": obs_col,
        "n_anchor_events_all": int(len(events_all)),
        "n_anchor_events_edge_safe": int(len(events)),
        "coverage_modality_near": cov,
        "frac_nonzero_signal_near": frac_nonzero,
        "peak_lag_sec": peak_lag,
        "peak_lag_value": peak_value,
        "delta_immediate_med": delta_imm_med,
        "delta_delayed_med": delta_dly_med,
        "sharpness_med": sharp_med,
        "delta_immediate_mean": delta_imm_mean,
        "sharpness_mean": sharp_mean,
        "rejection_reason": "",
    }, ""

def _stage1_score_144(row):
    dcore = _safe_float_144(row.get("delta_immediate_core"), 0.0)
    score = _safe_float_144(row.get("stage1_replication_score"), 0.0)
    cov = _safe_float_144(row.get("coverage_core"), 0.0)
    sharp = _safe_float_144(row.get("sharpness_core"), 0.0)
    lagdiff = _safe_float_144(row.get("lag_agreement_abs_diff"), 99.0)
    return float(score + 0.30 * np.log1p(max(0.0, dcore)) + 0.20 * cov + 0.10 * max(0.0, sharp) - 0.05 * lagdiff)

def _build_stage1_replication_144(stage1_df):
    if len(stage1_df) == 0:
        return pd.DataFrame()

    key_cols = ["anchor_col", "anchor_name", "anchor_type", "protocol_col", "protocol_tier"]

    train = stage1_df[stage1_df["split"].eq("train")].copy()
    val = stage1_df[stage1_df["split"].eq("val")].copy()

    train_r = train.rename(columns={c: f"{c}_train" for c in train.columns if c not in key_cols and c != "split"}).drop(columns=["split"], errors="ignore")
    val_r = val.rename(columns={c: f"{c}_val" for c in val.columns if c not in key_cols and c != "split"}).drop(columns=["split"], errors="ignore")

    rep = train_r.merge(val_r, on=key_cols, how="inner")

    rep["present_in_both"] = 1
    rep["lag_agreement_abs_diff"] = np.abs(
        pd.to_numeric(rep["peak_lag_sec_train"], errors="coerce")
        - pd.to_numeric(rep["peak_lag_sec_val"], errors="coerce")
    )

    def _min_pair(row, a, b):
        vals = [_safe_float_144(row.get(a), np.nan), _safe_float_144(row.get(b), np.nan)]
        vals = [v for v in vals if np.isfinite(v)]
        return min(vals) if vals else np.nan

    rep["delta_immediate_core"] = rep.apply(lambda r: _min_pair(r, "delta_immediate_med_train", "delta_immediate_med_val"), axis=1)
    rep["sharpness_core"] = rep.apply(lambda r: _min_pair(r, "sharpness_med_train", "sharpness_med_val"), axis=1)
    rep["coverage_core"] = rep.apply(lambda r: _min_pair(r, "coverage_modality_near_train", "coverage_modality_near_val"), axis=1)

    rep["positive_delta_both"] = (
        pd.to_numeric(rep["delta_immediate_med_train"], errors="coerce").gt(float(CFG.get("cell14_4_stage1_min_delta_core", 0.0)))
        & pd.to_numeric(rep["delta_immediate_med_val"], errors="coerce").gt(float(CFG.get("cell14_4_stage1_min_delta_core", 0.0)))
    )

    rep["positive_sharpness_both"] = (
        pd.to_numeric(rep["sharpness_med_train"], errors="coerce").gt(float(CFG.get("cell14_4_stage1_min_sharpness_core", -np.inf)))
        & pd.to_numeric(rep["sharpness_med_val"], errors="coerce").gt(float(CFG.get("cell14_4_stage1_min_sharpness_core", -np.inf)))
    )

    rep["stage1_lag_ok"] = pd.to_numeric(rep["lag_agreement_abs_diff"], errors="coerce").le(float(CFG.get("cell14_4_stage1_max_lag_abs_diff", 15)))

    rep["stage1_candidate"] = (
        rep["positive_delta_both"].astype(bool)
        & rep["positive_sharpness_both"].astype(bool)
        & rep["stage1_lag_ok"].astype(bool)
    )

    rep["stage1_replication_score"] = (
        0.45 * pd.to_numeric(rep["delta_immediate_core"], errors="coerce").clip(lower=0).fillna(0.0)
        + 0.25 * pd.to_numeric(rep["sharpness_core"], errors="coerce").clip(lower=0).fillna(0.0)
        + 0.20 * pd.to_numeric(rep["coverage_core"], errors="coerce").clip(lower=0).fillna(0.0)
        - 0.05 * pd.to_numeric(rep["lag_agreement_abs_diff"], errors="coerce").fillna(99.0)
    )

    rep["stage1_rank_score"] = rep.apply(_stage1_score_144, axis=1)

    return rep.sort_values(["stage1_candidate", "stage1_rank_score"], ascending=[False, False]).reset_index(drop=True)

def _shortlist_stage1_144(rep):
    if len(rep) == 0:
        return rep.copy()

    cand = rep[rep["stage1_candidate"].astype(bool)].copy()
    if len(cand) == 0:
        cand = rep.copy()

    selected_parts = []

    total_k = int(CFG.get("cell14_4_stage1_top_k_total", 500))
    selected_parts.append(cand.sort_values("stage1_rank_score", ascending=False).head(total_k))

    per_tier_k = int(CFG.get("cell14_4_stage1_top_k_per_protocol_tier", 200))
    for tier, sub in cand.groupby("protocol_tier"):
        selected_parts.append(sub.sort_values("stage1_rank_score", ascending=False).head(per_tier_k))

    per_col_k = int(CFG.get("cell14_4_stage1_top_k_per_protocol_col", 50))
    for col, sub in cand.groupby("protocol_col"):
        selected_parts.append(sub.sort_values("stage1_rank_score", ascending=False).head(per_col_k))

    shortlist = pd.concat(selected_parts, ignore_index=True).drop_duplicates(
        subset=["anchor_col", "protocol_col"]
    )

    # Force include current 13.5 manifest pairs, if available.
    if bool(CFG.get("cell14_4_stage1_force_include_existing_manifest_pairs", True)):
        if "CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF" in globals() and isinstance(CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF, pd.DataFrame):
            m = CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF.copy()
            if {"driver_col", "protocol_col"}.issubset(m.columns):
                keys = set(zip(m["driver_col"].astype(str), m["protocol_col"].astype(str)))
                forced = rep[
                    rep.apply(lambda r: (str(r["anchor_col"]), str(r["protocol_col"])) in keys, axis=1)
                ].copy()
                if len(forced):
                    forced["forced_existing_manifest_pair"] = True
                    shortlist["forced_existing_manifest_pair"] = False
                    shortlist = pd.concat([shortlist, forced], ignore_index=True).drop_duplicates(
                        subset=["anchor_col", "protocol_col"],
                        keep="last",
                    )

    if "forced_existing_manifest_pair" not in shortlist.columns:
        shortlist["forced_existing_manifest_pair"] = False

    return shortlist.sort_values(
        ["forced_existing_manifest_pair", "stage1_rank_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

# ----------------------------------------------------------
# 6) Stage 2 deep analysis
# ----------------------------------------------------------
def _classify_split_label_144(delta_imm, delta_dly, sharp, peak_lag, z_imm, z_sharp):
    if not np.isfinite(delta_imm):
        return "unclear"

    positive_shape = (
        np.isfinite(delta_imm) and delta_imm > 0
        and np.isfinite(sharp) and sharp > 0
        and (
            not np.isfinite(delta_dly)
            or delta_imm >= delta_dly
        )
    )

    if (
        positive_shape
        and np.isfinite(z_imm) and z_imm >= float(CFG.get("cell14_4_tierA_min_z_immediate", 3.0))
        and np.isfinite(z_sharp) and z_sharp >= float(CFG.get("cell14_4_tierA_min_z_sharpness", 2.0))
        and np.isfinite(peak_lag)
        and int(CFG.get("cell14_4_tierA_lag_lo", 0)) <= int(peak_lag) <= int(CFG.get("cell14_4_tierA_lag_hi", 5))
    ):
        return "event_locked_positive"

    if (
        np.isfinite(delta_imm) and delta_imm > 0
        and np.isfinite(z_imm) and z_imm >= float(CFG.get("cell14_4_tierC_min_z_immediate", 2.0))
    ):
        if np.isfinite(peak_lag) and peak_lag < 0:
            return "pre_leading_or_shifted"
        if np.isfinite(delta_dly) and delta_dly > delta_imm:
            return "delayed_broad_response"
        return "suggestive_positive"

    if np.isfinite(delta_dly) and np.isfinite(delta_imm) and delta_dly > delta_imm and delta_dly > 0:
        return "delayed_broad_response"

    return "weak_or_unclear"

def _deep_split_pair_144(split_name, frame, anchor_col, protocol_col):
    lag_min = int(CFG.get("cell14_4_stage2_lag_min", -30))
    lag_max = int(CFG.get("cell14_4_stage2_lag_max", 60))
    lags = np.arange(lag_min, lag_max + 1, dtype=np.int64)

    pre = tuple(map(int, CFG.get("cell14_4_stage2_pre_window", [-30, -1])))
    imm = tuple(map(int, CFG.get("cell14_4_stage2_immediate_window", [0, 5])))
    dly = tuple(map(int, CFG.get("cell14_4_stage2_delayed_window", [6, 30])))
    outer_left = tuple(map(int, CFG.get("cell14_4_stage2_outer_left_window", [-30, -6])))
    outer_right = tuple(map(int, CFG.get("cell14_4_stage2_outer_right_window", [6, 30])))

    tier = _tier_of_protocol_144(protocol_col)
    obs_col = _resolve_obs_col_144(frame, tier)

    if obs_col is None:
        return None, "missing_obs_col"

    anchor_x = _to_num_array_144(frame, anchor_col, fill=0.0)
    signal = _to_num_array_144(frame, protocol_col, fill=0.0)
    obs = _to_num_array_144(frame, obs_col, fill=0.0)

    events_all = _event_starts_144(anchor_x)
    events = _edge_safe_events_144(events_all, len(frame), lag_min, lag_max)

    min_events = int(CFG.get("cell14_4_min_train_anchor_events", 20)) if split_name == "train" else int(CFG.get("cell14_4_min_val_anchor_events", 5))
    if len(events) < min_events:
        return None, "insufficient_events"

    coverage_near = _coverage_near_144(obs, events, lag_min, lag_max)
    if np.isfinite(coverage_near) and coverage_near < float(CFG.get("cell14_4_stage1_min_modality_coverage", 0.20)):
        return None, "low_modality_coverage"

    frac_nonzero = _frac_nonzero_near_144(signal, events, -5, 5)
    if np.isfinite(frac_nonzero) and frac_nonzero < float(CFG.get("cell14_4_stage1_min_nonzero_signal_fraction", 0.005)):
        return None, "low_signal_activity_near_anchor"

    ev = _event_scores_144(signal, events, pre, imm, dly, outer_left, outer_right)

    delta_imm_med = _robust_median_144(ev["delta_immediate"])
    delta_dly_med = _robust_median_144(ev["delta_delayed"])
    sharp_med = _robust_median_144(ev["sharpness"])

    delta_imm_mean = _robust_mean_144(ev["delta_immediate"])
    delta_dly_mean = _robust_mean_144(ev["delta_delayed"])
    sharp_mean = _robust_mean_144(ev["sharpness"])

    profile = _lag_profile_144(signal, events, lags)
    peak_lag, peak_value = _peak_lag_144(profile)

    # Null baselines only in stage 2.
    null_base = _eligible_null_centers_144(len(frame), lag_min, lag_max)
    null_pool = _remove_event_neighborhoods_144(
        null_base,
        events,
        radius=int(CFG.get("cell14_4_stage2_null_exclusion_radius", 60)),
    )

    base_delta = []
    base_sharp = []

    n_null = int(CFG.get("cell14_4_stage2_n_null_baselines", 12))
    max_null_events = int(CFG.get("cell14_4_stage2_max_null_events", 512))
    n_null_events = min(len(events), max_null_events)

    for _ in range(n_null):
        null_centers = _sample_null_centers_144(null_pool, n_null_events, ref_centers=events[:n_null_events])
        if len(null_centers) == 0:
            continue
        ev0 = _event_scores_144(signal, null_centers, pre, imm, dly, outer_left, outer_right)
        base_delta.append(_robust_median_144(ev0["delta_immediate"]))
        base_sharp.append(_robust_median_144(ev0["sharpness"]))

    z_imm = _z_vs_null_144(delta_imm_med, base_delta)
    z_sharp = _z_vs_null_144(sharp_med, base_sharp)

    label = _classify_split_label_144(
        delta_imm=delta_imm_med,
        delta_dly=delta_dly_med,
        sharp=sharp_med,
        peak_lag=peak_lag,
        z_imm=z_imm,
        z_sharp=z_sharp,
    )

    return {
        "split": split_name,
        "anchor_col": anchor_col,
        "anchor_name": _anchor_name_144(anchor_col),
        "anchor_type": _anchor_type_144(anchor_col),
        "protocol_col": protocol_col,
        "protocol_tier": tier,
        "obs_col": obs_col,
        "n_anchor_events_all": int(len(events_all)),
        "n_anchor_events_edge_safe": int(len(events)),
        "coverage_modality_near": coverage_near,
        "frac_nonzero_signal_near": frac_nonzero,
        "peak_lag_sec": peak_lag,
        "peak_lag_value": peak_value,
        "delta_immediate_med": delta_imm_med,
        "delta_delayed_med": delta_dly_med,
        "sharpness_med": sharp_med,
        "delta_immediate_mean": delta_imm_mean,
        "delta_delayed_mean": delta_dly_mean,
        "sharpness_mean": sharp_mean,
        "z_immediate_vs_null": z_imm,
        "z_sharpness_vs_null": z_sharp,
        "baseline_delta_immediate_mean": float(np.nanmean(base_delta)) if len(base_delta) else np.nan,
        "baseline_delta_immediate_std": float(np.nanstd(base_delta)) if len(base_delta) else np.nan,
        "baseline_sharpness_mean": float(np.nanmean(base_sharp)) if len(base_sharp) else np.nan,
        "baseline_sharpness_std": float(np.nanstd(base_sharp)) if len(base_sharp) else np.nan,
        "alignment_label": label,
        "rejection_reason": "",
    }, ""

def _build_stage2_replication_144(stage2_df, shortlist_df):
    if len(stage2_df) == 0:
        return pd.DataFrame()

    key_cols = ["anchor_col", "anchor_name", "anchor_type", "protocol_col", "protocol_tier"]

    train = stage2_df[stage2_df["split"].eq("train")].copy()
    val = stage2_df[stage2_df["split"].eq("val")].copy()

    train_r = train.rename(columns={c: f"{c}_train" for c in train.columns if c not in key_cols and c != "split"}).drop(columns=["split"], errors="ignore")
    val_r = val.rename(columns={c: f"{c}_val" for c in val.columns if c not in key_cols and c != "split"}).drop(columns=["split"], errors="ignore")

    rep = train_r.merge(val_r, on=key_cols, how="outer")

    rep["present_in_train"] = rep["alignment_label_train"].notna().astype(int)
    rep["present_in_val"] = rep["alignment_label_val"].notna().astype(int)
    rep["present_in_both"] = ((rep["present_in_train"] == 1) & (rep["present_in_val"] == 1)).astype(int)

    rep["lag_agreement_abs_diff"] = np.abs(
        pd.to_numeric(rep.get("peak_lag_sec_train"), errors="coerce")
        - pd.to_numeric(rep.get("peak_lag_sec_val"), errors="coerce")
    )

    def _min_pair(row, a, b):
        vals = [_safe_float_144(row.get(a), np.nan), _safe_float_144(row.get(b), np.nan)]
        vals = [v for v in vals if np.isfinite(v)]
        return min(vals) if vals else np.nan

    rep["delta_immediate_core"] = rep.apply(lambda r: _min_pair(r, "delta_immediate_med_train", "delta_immediate_med_val"), axis=1)
    rep["sharpness_core"] = rep.apply(lambda r: _min_pair(r, "sharpness_med_train", "sharpness_med_val"), axis=1)
    rep["z_immediate_core"] = rep.apply(lambda r: _min_pair(r, "z_immediate_vs_null_train", "z_immediate_vs_null_val"), axis=1)
    rep["z_sharpness_core"] = rep.apply(lambda r: _min_pair(r, "z_sharpness_vs_null_train", "z_sharpness_vs_null_val"), axis=1)
    rep["coverage_core"] = rep.apply(lambda r: _min_pair(r, "coverage_modality_near_train", "coverage_modality_near_val"), axis=1)

    # Bring stage1 metadata back.
    s1_cols = [
        "anchor_col", "protocol_col",
        "stage1_rank_score", "stage1_replication_score",
        "forced_existing_manifest_pair",
    ]
    s1_cols = [c for c in s1_cols if c in shortlist_df.columns]
    if len(s1_cols) >= 2:
        rep = rep.merge(shortlist_df[s1_cols].drop_duplicates(["anchor_col", "protocol_col"]), on=["anchor_col", "protocol_col"], how="left")

    tier_out = rep.apply(lambda r: _tier_candidate_144(r), axis=1, result_type="expand")
    rep["coupling_tier"] = tier_out[0]
    rep["coupling_tier_reason"] = tier_out[1]
    rep["replication_score"] = rep.apply(_score_replication_144, axis=1)

    return rep.sort_values(
        ["coupling_tier", "replication_score", "z_immediate_core", "z_sharpness_core"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

def _tier_candidate_144(row):
    present_both = bool(row.get("present_in_both", 0) == 1)
    label_tr = str(row.get("alignment_label_train", ""))
    label_va = str(row.get("alignment_label_val", ""))

    delta_tr = _safe_float_144(row.get("delta_immediate_med_train"), np.nan)
    delta_va = _safe_float_144(row.get("delta_immediate_med_val"), np.nan)
    sharp_tr = _safe_float_144(row.get("sharpness_med_train"), np.nan)
    sharp_va = _safe_float_144(row.get("sharpness_med_val"), np.nan)

    zimm_tr = _safe_float_144(row.get("z_immediate_vs_null_train"), np.nan)
    zimm_va = _safe_float_144(row.get("z_immediate_vs_null_val"), np.nan)
    zsharp_tr = _safe_float_144(row.get("z_sharpness_vs_null_train"), np.nan)
    zsharp_va = _safe_float_144(row.get("z_sharpness_vs_null_val"), np.nan)

    cov_tr = _safe_float_144(row.get("coverage_modality_near_train"), np.nan)
    cov_va = _safe_float_144(row.get("coverage_modality_near_val"), np.nan)

    lag_tr = _safe_float_144(row.get("peak_lag_sec_train"), np.nan)
    lag_va = _safe_float_144(row.get("peak_lag_sec_val"), np.nan)
    lag_diff = abs(lag_tr - lag_va) if np.isfinite(lag_tr) and np.isfinite(lag_va) else np.nan

    pos_both = bool(
        np.isfinite(delta_tr) and delta_tr > 0
        and np.isfinite(delta_va) and delta_va > 0
        and np.isfinite(sharp_tr) and sharp_tr > 0
        and np.isfinite(sharp_va) and sharp_va > 0
    )

    event_locked_both = bool(label_tr == "event_locked_positive" and label_va == "event_locked_positive")

    zimm_core = min([v for v in [zimm_tr, zimm_va] if np.isfinite(v)], default=np.nan)
    zsharp_core = min([v for v in [zsharp_tr, zsharp_va] if np.isfinite(v)], default=np.nan)
    cov_core = min([v for v in [cov_tr, cov_va] if np.isfinite(v)], default=np.nan)

    if not present_both:
        return "Reject", "not_present_in_both_train_val"

    if not pos_both:
        return "Reject", "non_positive_or_non_sharp_train_val"

    if (
        event_locked_both
        and np.isfinite(zimm_core) and zimm_core >= float(CFG.get("cell14_4_tierA_min_z_immediate", 3.0))
        and np.isfinite(zsharp_core) and zsharp_core >= float(CFG.get("cell14_4_tierA_min_z_sharpness", 2.0))
        and np.isfinite(cov_core) and cov_core >= float(CFG.get("cell14_4_tierA_min_coverage", 0.50))
        and np.isfinite(lag_diff) and lag_diff <= float(CFG.get("cell14_4_tierA_max_lag_abs_diff", 2))
        and int(CFG.get("cell14_4_tierA_lag_lo", 0)) <= lag_tr <= int(CFG.get("cell14_4_tierA_lag_hi", 5))
        and int(CFG.get("cell14_4_tierA_lag_lo", 0)) <= lag_va <= int(CFG.get("cell14_4_tierA_lag_hi", 5))
    ):
        return "A", "replicated_high_confidence_event_locked"

    if (
        label_tr in {"event_locked_positive", "suggestive_positive"}
        and label_va in {"event_locked_positive", "suggestive_positive"}
        and np.isfinite(zimm_core) and zimm_core >= float(CFG.get("cell14_4_tierB_min_z_immediate", 2.5))
        and np.isfinite(zsharp_core) and zsharp_core >= float(CFG.get("cell14_4_tierB_min_z_sharpness", 1.5))
        and np.isfinite(cov_core) and cov_core >= float(CFG.get("cell14_4_tierB_min_coverage", 0.35))
        and np.isfinite(lag_diff) and lag_diff <= float(CFG.get("cell14_4_tierB_max_lag_abs_diff", 5))
        and int(CFG.get("cell14_4_tierB_lag_lo", 0)) <= lag_tr <= int(CFG.get("cell14_4_tierB_lag_hi", 10))
        and int(CFG.get("cell14_4_tierB_lag_lo", 0)) <= lag_va <= int(CFG.get("cell14_4_tierB_lag_hi", 10))
    ):
        return "B", "replicated_positive_coupling"

    if (
        label_tr in {"event_locked_positive", "suggestive_positive", "delayed_broad_response"}
        and label_va in {"event_locked_positive", "suggestive_positive", "delayed_broad_response"}
        and np.isfinite(zimm_core) and zimm_core >= float(CFG.get("cell14_4_tierC_min_z_immediate", 2.0))
        and np.isfinite(cov_core) and cov_core >= float(CFG.get("cell14_4_tierC_min_coverage", 0.25))
        and np.isfinite(lag_diff) and lag_diff <= float(CFG.get("cell14_4_tierC_max_lag_abs_diff", 10))
    ):
        return "C", "suggestive_replicated_needs_review"

    return "D", "weak_or_unstable_replicated_signal"

def _score_replication_144(row):
    zimm = _safe_float_144(row.get("z_immediate_core"), 0.0)
    zsharp = _safe_float_144(row.get("z_sharpness_core"), 0.0)
    cov = _safe_float_144(row.get("coverage_core"), 0.0)
    gain = _safe_float_144(row.get("delta_immediate_core"), 0.0)
    lag_diff = _safe_float_144(row.get("lag_agreement_abs_diff"), 99.0)

    return float(
        0.40 * max(0.0, zimm)
        + 0.30 * max(0.0, zsharp)
        + 0.20 * max(0.0, cov)
        + 0.10 * np.log1p(max(0.0, gain))
        - 0.15 * max(0.0, lag_diff)
    )

# ----------------------------------------------------------
# 7) Run Stage 1
# ----------------------------------------------------------
anchor_cols, anchor_support_df = _discover_anchor_cols_144()
protocol_cols = _discover_protocol_cols_144()

if not anchor_cols:
    raise RuntimeError("[Cell14.4] No eligible IoT anchor columns discovered.")
if not protocol_cols:
    raise RuntimeError("[Cell14.4] No eligible protocol signal columns discovered.")

total_pairs = len(anchor_cols) * len(protocol_cols)

log(
    "[Cell14.4] Stage 1 discovery scope | "
    f"anchors={len(anchor_cols)} | protocol_signals={len(protocol_cols)} | "
    f"pairs={total_pairs} | protocol_tiers={dict(Counter(_tier_of_protocol_144(c) for c in protocol_cols))}"
)

stage1_rows = []
stage1_reject_rows = []

pair_idx = 0
for anchor_col in anchor_cols:
    for protocol_col in protocol_cols:
        pair_idx += 1
        if pair_idx == 1 or pair_idx % 250 == 0 or pair_idx == total_pairs:
            log(f"[Cell14.4] Stage 1 progress {pair_idx}/{total_pairs}")

        for split_name, frame in [("train", df_tr), ("val", df_val)]:
            result, rejection = _fast_split_pair_144(split_name, frame, anchor_col, protocol_col)
            if result is not None:
                stage1_rows.append(result)
            else:
                stage1_reject_rows.append({
                    "split": split_name,
                    "anchor_col": anchor_col,
                    "anchor_name": _anchor_name_144(anchor_col),
                    "anchor_type": _anchor_type_144(anchor_col),
                    "protocol_col": protocol_col,
                    "protocol_tier": _tier_of_protocol_144(protocol_col),
                    "rejection_reason": rejection,
                })

stage1_df = pd.DataFrame(stage1_rows)
stage1_rep = _build_stage1_replication_144(stage1_df)
stage1_shortlist = _shortlist_stage1_144(stage1_rep)

stage1_df.to_csv(stage1_metrics_csv, index=False)
anchor_support_df.to_csv(anchor_support_csv, index=False)
pd.DataFrame(stage1_reject_rows).to_csv(stage1_rejects_csv, index=False)
stage1_shortlist.to_csv(stage1_shortlist_csv, index=False)

log(
    "[Cell14.4] Stage 1 complete | "
    f"split_metric_rows={len(stage1_df)} | "
    f"replication_rows={len(stage1_rep)} | "
    f"shortlist_rows={len(stage1_shortlist)}"
)

if len(stage1_shortlist) == 0 and bool(CFG.get("cell14_4_fail_if_no_candidates", False)):
    raise RuntimeError("[Cell14.4] Stage 1 produced no shortlist candidates.")

# ----------------------------------------------------------
# 8) Run Stage 2 only on shortlist
# ----------------------------------------------------------
stage2_rows = []
stage2_reject_rows = []

if len(stage1_shortlist) and {"anchor_col", "protocol_col"}.issubset(stage1_shortlist.columns):
    stage2_pairs = stage1_shortlist[["anchor_col", "protocol_col"]].drop_duplicates().to_dict("records")
else:
    stage2_pairs = []

log(
    "[Cell14.4] Stage 2 deep replication | "
    f"shortlisted_pairs={len(stage2_pairs)} | "
    f"null_baselines={int(CFG.get('cell14_4_stage2_n_null_baselines', 12))}"
)

for i, p in enumerate(stage2_pairs, start=1):
    if i == 1 or i % 50 == 0 or i == len(stage2_pairs):
        log(f"[Cell14.4] Stage 2 progress {i}/{len(stage2_pairs)}")

    anchor_col = str(p["anchor_col"])
    protocol_col = str(p["protocol_col"])

    for split_name, frame in [("train", df_tr), ("val", df_val)]:
        result, rejection = _deep_split_pair_144(split_name, frame, anchor_col, protocol_col)
        if result is not None:
            stage2_rows.append(result)
        else:
            stage2_reject_rows.append({
                "split": split_name,
                "anchor_col": anchor_col,
                "anchor_name": _anchor_name_144(anchor_col),
                "anchor_type": _anchor_type_144(anchor_col),
                "protocol_col": protocol_col,
                "protocol_tier": _tier_of_protocol_144(protocol_col),
                "rejection_reason": rejection,
            })

stage2_df = pd.DataFrame(stage2_rows)
rep = _build_stage2_replication_144(stage2_df, stage1_shortlist)

stage2_df.to_csv(stage2_metrics_csv, index=False)
pd.DataFrame(stage2_reject_rows).to_csv(stage2_rejects_csv, index=False)
rep.to_csv(replication_table_csv, index=False)

# ----------------------------------------------------------
# 9) Tier outputs and summaries
# ----------------------------------------------------------
tierA = rep[rep["coupling_tier"].eq("A")].copy() if len(rep) else pd.DataFrame()
tierB = rep[rep["coupling_tier"].eq("B")].copy() if len(rep) else pd.DataFrame()
tierC = rep[rep["coupling_tier"].eq("C")].copy() if len(rep) else pd.DataFrame()

tierA.to_csv(tierA_csv, index=False)
tierB.to_csv(tierB_csv, index=False)
tierC.to_csv(tierC_csv, index=False)

summary_rows = []
if len(rep):
    for tier, sub in rep.groupby("coupling_tier", dropna=False):
        summary_rows.append({
            "coupling_tier": tier,
            "n_pairs": int(len(sub)),
            "n_anchor_cols": int(sub["anchor_col"].nunique()),
            "n_protocol_cols": int(sub["protocol_col"].nunique()),
            "mean_replication_score": float(pd.to_numeric(sub["replication_score"], errors="coerce").mean()),
            "mean_z_immediate_core": float(pd.to_numeric(sub["z_immediate_core"], errors="coerce").mean()),
            "mean_z_sharpness_core": float(pd.to_numeric(sub["z_sharpness_core"], errors="coerce").mean()),
            "mean_coverage_core": float(pd.to_numeric(sub["coverage_core"], errors="coerce").mean()),
            "mean_lag_abs_diff": float(pd.to_numeric(sub["lag_agreement_abs_diff"], errors="coerce").mean()),
        })

    for (tier, ptier), sub in rep.groupby(["coupling_tier", "protocol_tier"], dropna=False):
        summary_rows.append({
            "coupling_tier": f"{tier}::{ptier}",
            "n_pairs": int(len(sub)),
            "n_anchor_cols": int(sub["anchor_col"].nunique()),
            "n_protocol_cols": int(sub["protocol_col"].nunique()),
            "mean_replication_score": float(pd.to_numeric(sub["replication_score"], errors="coerce").mean()),
            "mean_z_immediate_core": float(pd.to_numeric(sub["z_immediate_core"], errors="coerce").mean()),
            "mean_z_sharpness_core": float(pd.to_numeric(sub["z_sharpness_core"], errors="coerce").mean()),
            "mean_coverage_core": float(pd.to_numeric(sub["coverage_core"], errors="coerce").mean()),
            "mean_lag_abs_diff": float(pd.to_numeric(sub["lag_agreement_abs_diff"], errors="coerce").mean()),
        })

tier_summary_df = pd.DataFrame(summary_rows)
tier_summary_df.to_csv(tier_summary_csv, index=False)

tier_counts = rep["coupling_tier"].astype(str).value_counts().sort_index().to_dict() if len(rep) else {}
tier_protocol_counts = (
    rep.groupby(["coupling_tier", "protocol_tier"]).size().reset_index(name="n").to_dict("records")
    if len(rep)
    else []
)

# ----------------------------------------------------------
# 10) Save contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "14.4",
    "version": CELL144_VERSION,
    "role": "two_stage_trainval_iot_protocol_coupling_discovery_and_tiering",
    "quality_dimension": "Q4_cross_modal_consistency_manifest_expansion",
    "upstream_contract_versions": {
        "cell13_4": version134_144,
        "cell13_6": version136_144,
        "cell14_3": version143_144
    },
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_144,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_144),
        "pair_pass_rate": broad_q4_pair_pass_rate_144
    },
    "manifest_q4_status_carried_forward": {
        "pre_repair_publication_blocker_n": int(pre_manifest_blocker_n_144),
        "pre_repair_pass_rate": pre_manifest_pass_rate_144,
        "post_repair_publication_blocker_n": int(repair_post_blocker_n_144),
        "post_repair_pass_rate": repair_post_pass_rate_144,
        "repair_candidate_status": repair_status_144,
        "repair_candidate_reasons": repair_reasons_144,
        "repair_delta_summary": repair_delta_summary_144
    },
    "stage1_fast_screen": {
        "anchors_discovered": int(len(anchor_cols)),
        "protocol_signals_discovered": int(len(protocol_cols)),
        "possible_pairs": int(total_pairs),
        "split_metric_rows": int(len(stage1_df)),
        "reject_rows": int(len(stage1_reject_rows)),
        "replication_rows": int(len(stage1_rep)),
        "shortlist_rows": int(len(stage1_shortlist)),
        "lag_min": int(CFG.get("cell14_4_stage1_lag_min", -10)),
        "lag_max": int(CFG.get("cell14_4_stage1_lag_max", 20)),
    },
    "stage2_deep_replication": {
        "shortlisted_pairs": int(len(stage2_pairs)),
        "split_metric_rows": int(len(stage2_df)),
        "reject_rows": int(len(stage2_reject_rows)),
        "replication_rows": int(len(rep)),
        "null_baselines": int(CFG.get("cell14_4_stage2_n_null_baselines", 12)),
        "lag_min": int(CFG.get("cell14_4_stage2_lag_min", -30)),
        "lag_max": int(CFG.get("cell14_4_stage2_lag_max", 60)),
    },
    "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
    "tier_protocol_counts": tier_protocol_counts,
    "thresholds": {
        "tierA": {
            "min_z_immediate": float(CFG.get("cell14_4_tierA_min_z_immediate", 3.0)),
            "min_z_sharpness": float(CFG.get("cell14_4_tierA_min_z_sharpness", 2.0)),
            "min_coverage": float(CFG.get("cell14_4_tierA_min_coverage", 0.50)),
            "max_lag_abs_diff": float(CFG.get("cell14_4_tierA_max_lag_abs_diff", 2)),
            "lag_range": [int(CFG.get("cell14_4_tierA_lag_lo", 0)), int(CFG.get("cell14_4_tierA_lag_hi", 5))],
        },
        "tierB": {
            "min_z_immediate": float(CFG.get("cell14_4_tierB_min_z_immediate", 2.5)),
            "min_z_sharpness": float(CFG.get("cell14_4_tierB_min_z_sharpness", 1.5)),
            "min_coverage": float(CFG.get("cell14_4_tierB_min_coverage", 0.35)),
            "max_lag_abs_diff": float(CFG.get("cell14_4_tierB_max_lag_abs_diff", 5)),
            "lag_range": [int(CFG.get("cell14_4_tierB_lag_lo", 0)), int(CFG.get("cell14_4_tierB_lag_hi", 10))],
        },
        "tierC": {
            "min_z_immediate": float(CFG.get("cell14_4_tierC_min_z_immediate", 2.0)),
            "min_coverage": float(CFG.get("cell14_4_tierC_min_coverage", 0.25)),
            "max_lag_abs_diff": float(CFG.get("cell14_4_tierC_max_lag_abs_diff", 10)),
        },
    },
    "strict_contract": {
        "train_values_used_for_discovery": True,
        "coupling_candidate_discovery_done_here": True,
        "coupling_candidate_shortlisting_done_here": True,
        "TEST_activity_used_for_discovery": False,
        "repair_status_carried_forward": True,
        "val_values_used_for_replication": True,
        "TEST_real_values_used": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
    },
    "outputs": {
        "stage1_metrics_csv": stage1_metrics_csv,
        "stage1_shortlist_csv": stage1_shortlist_csv,
        "stage1_rejects_csv": stage1_rejects_csv,
        "anchor_support_csv": anchor_support_csv,
        "stage2_metrics_csv": stage2_metrics_csv,
        "stage2_rejects_csv": stage2_rejects_csv,
        "replication_table_csv": replication_table_csv,
        "tier_summary_csv": tier_summary_csv,
        "tierA_csv": tierA_csv,
        "tierB_csv": tierB_csv,
        "tierC_csv": tierC_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_144(contract_json, contract)
_write_json_144(contract_canonical_json, contract)

manifest = {
    "cell": "14.4",
    "version": CELL144_VERSION,
    "created_outputs": contract["outputs"],
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "manifest_q4_status_carried_forward": contract["manifest_q4_status_carried_forward"],
    "tier_counts": contract["tier_counts"],
    "tier_protocol_counts": tier_protocol_counts,
    "strict_contract": contract["strict_contract"],
}

_write_json_144(manifest_json, manifest)

hashes = {
    "stage1_metrics_csv_sha256": _sha256_file_144(stage1_metrics_csv),
    "stage1_shortlist_csv_sha256": _sha256_file_144(stage1_shortlist_csv),
    "stage1_rejects_csv_sha256": _sha256_file_144(stage1_rejects_csv),
    "anchor_support_csv_sha256": _sha256_file_144(anchor_support_csv),
    "stage2_metrics_csv_sha256": _sha256_file_144(stage2_metrics_csv),
    "stage2_rejects_csv_sha256": _sha256_file_144(stage2_rejects_csv),
    "replication_table_csv_sha256": _sha256_file_144(replication_table_csv),
    "tier_summary_csv_sha256": _sha256_file_144(tier_summary_csv),
    "tierA_csv_sha256": _sha256_file_144(tierA_csv),
    "tierB_csv_sha256": _sha256_file_144(tierB_csv),
    "tierC_csv_sha256": _sha256_file_144(tierC_csv),
    "contract_json_sha256": _sha256_file_144(contract_json),
    "contract_canonical_json_sha256": _sha256_file_144(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_144(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_144(contract_json, contract)
_write_json_144(contract_canonical_json, contract)
_write_json_144(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL144_VERSION"] = CELL144_VERSION
globals()["CELL14_4_STAGE1_FAST_SCREEN_DF"] = stage1_df
globals()["CELL14_4_STAGE1_SHORTLIST_DF"] = stage1_shortlist
globals()["CELL14_4_ANCHOR_SUPPORT_DF"] = anchor_support_df
globals()["CELL14_4_STAGE1_REJECTS_DF"] = pd.DataFrame(stage1_reject_rows)
globals()["CELL14_4_STAGE2_DEEP_METRICS_DF"] = stage2_df
globals()["CELL14_4_STAGE2_REJECTS_DF"] = pd.DataFrame(stage2_reject_rows)
globals()["CELL14_4_DEEP_COUPLING_REPLICATION_DF"] = rep
globals()["CELL14_4_DEEP_COUPLING_TIER_SUMMARY_DF"] = tier_summary_df
globals()["CELL14_4_DEEP_COUPLING_TIERA_DF"] = tierA
globals()["CELL14_4_DEEP_COUPLING_TIERB_DF"] = tierB
globals()["CELL14_4_DEEP_COUPLING_TIERC_DF"] = tierC
globals()["CELL14_4_DEEP_COUPLING_CONTRACT"] = contract
globals()["CELL14_4_STAGE1_FAST_SCREEN_CSV"] = stage1_metrics_csv
globals()["CELL14_4_STAGE1_SHORTLIST_CSV"] = stage1_shortlist_csv
globals()["CELL14_4_STAGE1_REJECTS_CSV"] = stage1_rejects_csv
globals()["CELL14_4_ANCHOR_SUPPORT_CSV"] = anchor_support_csv
globals()["CELL14_4_STAGE2_DEEP_METRICS_CSV"] = stage2_metrics_csv
globals()["CELL14_4_STAGE2_REJECTS_CSV"] = stage2_rejects_csv
globals()["CELL14_4_DEEP_COUPLING_REPLICATION_CSV"] = replication_table_csv
globals()["CELL14_4_DEEP_COUPLING_TIER_SUMMARY_CSV"] = tier_summary_csv
globals()["CELL14_4_DEEP_COUPLING_TIERA_CSV"] = tierA_csv
globals()["CELL14_4_DEEP_COUPLING_TIERB_CSV"] = tierB_csv
globals()["CELL14_4_DEEP_COUPLING_TIERC_CSV"] = tierC_csv
globals()["CELL14_4_DEEP_COUPLING_CONTRACT_JSON"] = contract_json
globals()["CELL14_4_DEEP_COUPLING_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL14_4_DEEP_COUPLING_MANIFEST_JSON"] = manifest_json

log(
    "[Cell14.4] Two-stage coupling discovery complete | "
    f"anchors={len(anchor_cols)} | protocol_signals={len(protocol_cols)} | "
    f"stage1_pairs={total_pairs} | "
    f"stage1_shortlist={len(stage1_shortlist)} | "
    f"stage2_replication_rows={len(rep)} | "
    f"tier_counts={tier_counts}"
)
log(f"[Cell14.4] Tier/protocol counts | {tier_protocol_counts}")
log(f"[Cell14.4] Saved Stage 1 shortlist: {stage1_shortlist_csv} | rows={len(stage1_shortlist)}")
log(f"[Cell14.4] Saved Stage 2 replication table: {replication_table_csv} | rows={len(rep)}")
log(f"[Cell14.4] Saved Tier A candidates: {tierA_csv} | rows={len(tierA)}")
log(f"[Cell14.4] Saved Tier B candidates: {tierB_csv} | rows={len(tierB)}")
log(f"[Cell14.4] Saved Tier C candidates: {tierC_csv} | rows={len(tierC)}")
log(f"[Cell14.4] Saved contract: {contract_json}")
log(f"[Cell14.4] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell14.4] Upstream Q4/repair status carried forward | "
    f"broad_status={broad_q4_status_144} | "
    f"broad_pair_blocker_n={broad_q4_pair_blocker_n_144} | "
    f"repair_status={repair_status_144} | "
    f"repair_post_blocker_n={repair_post_blocker_n_144} | "
    f"repair_post_pass_rate={repair_post_pass_rate_144}"
)
log(
    "[Cell14.4] Contract flags | "
    "train_values_used_for_discovery=True | "
    "val_values_used_for_replication=True | "
    "TEST_real_values_used=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | coupling_candidate_discovery_done_here=True | TEST_activity_used_for_discovery=False"
)
log("--- END: Cell 14.4 - Two-stage TRAIN/VAL IoT↔protocol coupling discovery and tiering (v2.1 repair-status-aware strict) ---")

gc.collect()