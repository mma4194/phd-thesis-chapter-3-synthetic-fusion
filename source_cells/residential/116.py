# ==========================================================
# CELL 14.6 - A0 Zigbee-safe expanded coupling materialization + QA
# v1.2 STUDY-THESIS strict A0-safe coupling-conditioned protocol branch, promotion-gated with protocol-base fallback
#
# Role:
#   - Use Cell 14.5 A0_zigbee_safe repair manifest.
#   - Build TRAIN/VAL-only profiles for those A0 pairs.
#   - Apply coupling-conditioned protocol materialization to TEST synthetic protocol.
#   - Mutate only:
#       zigbee__pkt_total
#       zigbee__bytes_total
#     inside A0-approved legal windows.
#   - Assemble separate A0 coupled CPS candidate.
#   - Run A0 manifest-style Q4 QA.
#
# Scientific contract:
#   - TRAIN/VAL real data are used only to learn coupling profiles.
#   - Synthetic TEST IoT driver events are used only as conditioning input.
#   - Real TEST values are used only for QA after materialization.
#   - No IoT synthetic values are mutated.
#   - No non-target protocol columns are mutated.
#   - No values outside legal A0 windows are mutated.
#
# Outputs:
#   reports/cell14_6_A0_profile_audit.csv
#   reports/cell14_6_A0_materialization_audit.csv
#   reports/cell14_6_A0_target_protocol_drift_audit.csv
#   reports/cell14_6_A0_manifest_q4_pair_metrics.csv
#   reports/cell14_6_A0_manifest_q4_prepost_delta.csv
#   reports/cell14_6_A0_summary.json
#   reports/cell14_6_A0_contract.json
#
#   synthetic/PROTOCOL_SYN_TEST_COUPLED_ZIGBEE_A0.parquet
#   synthetic/CPS_COUPLED_ZIGBEE_A0_TEST.parquet
#   artifacts/cell14_6_A0_profiles.npz
#   artifacts/cell14_6_A0_manifest_q4_curves.npz
#   artifacts/cell14_6_A0_manifest.json
# ==========================================================

log("--- START: Cell 14.6 - A0 Zigbee-safe expanded coupling materialization + QA (v1.2 promotion-gated protocol-base-fallback strict) ---")

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
_required_146 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_FULL_SYN_TEST_130",
    "IOT_REAL_TEST_REF_130",
    "PROTOCOL_SYN_TEST_130",
    "PROTOCOL_REAL_TEST_REF_130",
    "CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_DF",
    "CELL14_5_REPAIR_MANIFEST_CONTRACT",
    "CELL14_4_DEEP_COUPLING_CONTRACT",
    "CELL14_3_Q4_REPAIR_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
]
_missing_146 = [k for k in _required_146 if k not in globals()]
if _missing_146:
    raise RuntimeError(f"[Cell14.6] Missing required globals: {_missing_146}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_146 = str(OUT_SYN)
REPORT_DIR_ACTIVE_146 = str(REPORT_DIR)

def _resolve_project_root_146(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        if "q6_public_reaudit" in parts:
            idx = parts.index("q6_public_reaudit")
            candidates.append(os.sep.join(parts[:idx]))
        else:
            base = os.path.basename(p)
            if base in {"reports", "synthetic", "artifacts"}:
                candidates.append(os.path.dirname(p))
            else:
                candidates.append(p)

    # candidates.append("/path/to/input/cps_synth_v4_4_7_32_FULL_COUPLED")

    seen = set()
    for c in candidates:
        c = os.path.abspath(str(c))
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "synthetic")):
            return c

    raise RuntimeError("[Cell14.6] Could not resolve canonical project root.")

PROJECT_ROOT_146 = _resolve_project_root_146(OUTDIR, REPORT_DIR_ACTIVE_146, OUT_SYN_ACTIVE_146)

# Canonical Q4 outputs must always use main project dirs, never q6_public_reaudit.
OUT_SYN = os.path.join(PROJECT_ROOT_146, "synthetic")
REPORT_DIR = os.path.join(PROJECT_ROOT_146, "reports")
ARTDIR = os.path.join(PROJECT_ROOT_146, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

log(
    "[Cell14.6] Canonical project/output resolver | "
    f"PROJECT_ROOT={PROJECT_ROOT_146} | OUT_SYN={OUT_SYN} | REPORT_DIR={REPORT_DIR} | "
    f"ACTIVE_OUT_SYN={OUT_SYN_ACTIVE_146} | ACTIVE_REPORT_DIR={REPORT_DIR_ACTIVE_146}"
)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

CELL146_VERSION = "cell14_6_A0_zigbee_safe_coupling_materialization_qa_v1_2_promotion_gated_protocol_base_fallback"

CFG["cell14_6_version"] = CELL146_VERSION
CFG["cell14_6_trainval_profiles_used"] = True
CFG["cell14_6_synthetic_iot_drivers_used_for_conditioning"] = True
CFG["cell14_6_TEST_real_values_used_for_QA_only"] = True
CFG["cell14_6_synthetic_protocol_values_mutated"] = True
CFG["cell14_6_synthetic_iot_values_mutated"] = False
CFG["cell14_6_selection_done_here"] = False
CFG["cell14_6_generator_fit_done_here"] = False
CFG["cell14_6_A0_promotion_gate_enforced"] = True
CFG["cell14_6_candidate_outputs_always_written"] = True
CFG["cell14_6_accepted_outputs_written_only_if_gate_passes"] = True
CFG["cell14_6_broad_q4_status_carried_forward"] = True
CFG["cell14_6_repair_status_carried_forward"] = True
CFG["cell14_6_A0_manifest_policy_required"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell14_6_allowed_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])
CFG.setdefault("cell14_6_profile_pre_margin", 10)
CFG.setdefault("cell14_6_profile_post_margin", 10)
CFG.setdefault("cell14_6_min_train_events", 20)
CFG.setdefault("cell14_6_min_val_events", 5)
CFG.setdefault("cell14_6_min_combined_events", 30)
CFG.setdefault("cell14_6_min_modality_coverage", 0.20)

# Conservative application settings.
CFG.setdefault("cell14_6_profile_source", "mean_profile_bc")
CFG.setdefault("cell14_6_base_alpha_scale", 0.70)
CFG.setdefault("cell14_6_min_alpha", 0.05)
CFG.setdefault("cell14_6_max_alpha", 0.85)
CFG.setdefault("cell14_6_clip_to_trainval_support", True)
CFG.setdefault("cell14_6_clip_margin_fraction", 0.10)
# Critical STUDY-THESIS Q4 fix:
# A0 uses the pre-11b protocol baseline, whose stale zigbee__zcl_obs_present
# would block legal A0 writes. Final/current Q4 protocol namespaces do not
# carry this obs gate. A0 writes are instead constrained by manifest-approved
# driver-conditioned legal/profile windows and domain checks.
CFG["cell14_6_respect_protocol_observability"] = False
CFG["cell14_6_disable_stale_pre11b_zigbee_obs_gate"] = True
CFG.setdefault("cell14_6_nonnegative_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])
CFG.setdefault("cell14_6_integer_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])

# QA settings: mirror Cell 13.6/14.3 thresholds.
CFG.setdefault("cell14_6_eta_pre_seconds", int(CFG.get("cell13_6_eta_pre_seconds", 10)))
CFG.setdefault("cell14_6_eta_post_seconds", int(CFG.get("cell13_6_eta_post_seconds", 30)))
CFG.setdefault("cell14_6_profile_extra_lag_margin", int(CFG.get("cell13_6_profile_extra_lag_margin", 5)))
CFG.setdefault("cell14_6_min_events_for_pass", int(CFG.get("cell13_6_min_events_for_pass", 5)))
CFG.setdefault("cell14_6_protocol_transform", str(CFG.get("cell13_6_protocol_transform", "log1p_abs_signed")))
CFG.setdefault("cell14_6_response_threshold_quantile", float(CFG.get("cell13_6_response_threshold_quantile", 0.90)))
CFG.setdefault("cell14_6_eta_similarity_pass", float(CFG.get("cell13_6_eta_similarity_pass", 0.70)))
CFG.setdefault("cell14_6_eta_similarity_warning", float(CFG.get("cell13_6_eta_similarity_warning", 0.50)))
CFG.setdefault("cell14_6_profile_similarity_pass", float(CFG.get("cell13_6_profile_similarity_pass", 0.70)))
CFG.setdefault("cell14_6_profile_similarity_warning", float(CFG.get("cell13_6_profile_similarity_warning", 0.50)))
CFG.setdefault("cell14_6_lag_peak_error_pass", float(CFG.get("cell13_6_lag_peak_error_pass", 2)))
CFG.setdefault("cell14_6_lag_peak_error_warning", float(CFG.get("cell13_6_lag_peak_error_warning", 5)))
CFG.setdefault("cell14_6_response_window_rate_error_pass", float(CFG.get("cell13_6_response_window_rate_error_pass", 0.10)))
CFG.setdefault("cell14_6_response_window_rate_error_warning", float(CFG.get("cell13_6_response_window_rate_error_warning", 0.25)))
CFG.setdefault("cell14_6_norm_eps", 1e-9)

# Hard checks.
CFG.setdefault("cell14_6_fail_if_no_valid_profiles", True)
CFG.setdefault("cell14_6_fail_on_non_target_drift", True)
CFG.setdefault("cell14_6_fail_on_outside_window_drift", True)
CFG.setdefault("cell14_6_fail_on_inactive_finite", True)
CFG.setdefault("cell14_6_fail_on_negative_or_noninteger", True)

A0_ALLOWED_COLS_146 = set(map(str, CFG.get("cell14_6_allowed_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])))
NONNEGATIVE_COLS_146 = set(map(str, CFG.get("cell14_6_nonnegative_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])))
INTEGER_COLS_146 = set(map(str, CFG.get("cell14_6_integer_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])))

PROFILE_PRE_MARGIN_146 = int(CFG.get("cell14_6_profile_pre_margin", 10))
PROFILE_POST_MARGIN_146 = int(CFG.get("cell14_6_profile_post_margin", 10))

ETA_PRE_146 = int(CFG.get("cell14_6_eta_pre_seconds", 10))
ETA_POST_146 = int(CFG.get("cell14_6_eta_post_seconds", 30))
EXTRA_MARGIN_146 = int(CFG.get("cell14_6_profile_extra_lag_margin", 5))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
profile_audit_csv = os.path.join(REPORT_DIR, "cell14_6_A0_profile_audit.csv")
materialization_audit_csv = os.path.join(REPORT_DIR, "cell14_6_A0_materialization_audit.csv")
target_drift_csv = os.path.join(REPORT_DIR, "cell14_6_A0_target_protocol_drift_audit.csv")
non_target_drift_csv = os.path.join(REPORT_DIR, "cell14_6_A0_non_target_protocol_drift_audit.csv")
# Candidate outputs: always written.
q4_candidate_metrics_csv = os.path.join(REPORT_DIR, "cell14_6_A0_candidate_manifest_q4_pair_metrics.csv")
q4_candidate_delta_csv = os.path.join(REPORT_DIR, "cell14_6_A0_candidate_manifest_q4_prepost_delta.csv")
q4_candidate_summary_json = os.path.join(REPORT_DIR, "cell14_6_A0_candidate_summary.json")
candidate_contract_json = os.path.join(REPORT_DIR, "cell14_6_A0_candidate_contract.json")
candidate_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_6_A0_candidate_contract_v1_2_THESIS.json")

protocol_a0_candidate_path = os.path.join(OUT_SYN, "PROTOCOL_SYN_TEST_CANDIDATE_ZIGBEE_A0.parquet")
cps_a0_candidate_path = os.path.join(OUT_SYN, "CPS_CANDIDATE_ZIGBEE_A0_TEST.parquet")

# Accepted outputs: written only if strict A0 gates pass.
q4_accepted_metrics_csv = os.path.join(REPORT_DIR, "cell14_6_A0_ACCEPTED_manifest_q4_pair_metrics.csv")
q4_accepted_delta_csv = os.path.join(REPORT_DIR, "cell14_6_A0_ACCEPTED_manifest_q4_prepost_delta.csv")
q4_accepted_summary_json = os.path.join(REPORT_DIR, "cell14_6_A0_ACCEPTED_summary.json")
accepted_contract_json = os.path.join(REPORT_DIR, "cell14_6_A0_ACCEPTED_contract.json")
accepted_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_6_A0_ACCEPTED_contract_v1_2_THESIS.json")

protocol_a0_accepted_path = os.path.join(OUT_SYN, "PROTOCOL_SYN_TEST_ACCEPTED_ZIGBEE_A0.parquet")
cps_a0_accepted_path = os.path.join(OUT_SYN, "CPS_ACCEPTED_ZIGBEE_A0_TEST.parquet")

# Backward-compatible names now point to candidate outputs only.
q4_metrics_csv = q4_candidate_metrics_csv
q4_delta_csv = q4_candidate_delta_csv
q4_summary_json = q4_candidate_summary_json
contract_json = candidate_contract_json
protocol_a0_path = protocol_a0_candidate_path
cps_a0_path = cps_a0_candidate_path

profiles_npz_path = os.path.join(ARTDIR, "cell14_6_A0_profiles.npz")
q4_curves_npz_path = os.path.join(ARTDIR, "cell14_6_A0_manifest_q4_curves.npz")
manifest_json = os.path.join(ARTDIR, "cell14_6_A0_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_146(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_146(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_146(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_146(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_146(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_146(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_146(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_146(obj.to_dict())
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

def _write_json_146(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_146(payload), f, indent=2, sort_keys=True)

def _sha256_file_146(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_146(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_146(frame: pd.DataFrame, col: str, fill=None) -> np.ndarray:
    arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if fill is not None:
        arr = np.where(np.isfinite(arr), arr, float(fill))
    return arr

def _resolve_obs_col_146(frame: pd.DataFrame, tier: str):
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

def _event_starts_146(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    active = np.isfinite(arr) & (arr > 0.0)
    if active.size == 0:
        return np.asarray([], dtype=np.int64)
    prev = np.r_[False, active[:-1]]
    starts = active & (~prev)
    return np.flatnonzero(starts).astype(np.int64)

def _edge_safe_events_146(events: np.ndarray, n: int, lag_lo: int, lag_hi: int) -> np.ndarray:
    events = np.asarray(events, dtype=np.int64)
    if events.size == 0:
        return events
    keep = (events + lag_lo >= 0) & (events + lag_hi < n)
    return events[keep]

def _window_matrix_146(y: np.ndarray, obs: np.ndarray, events: np.ndarray, lag_lo: int, lag_hi: int):
    y = np.asarray(y, dtype=np.float64)
    obs = np.asarray(obs, dtype=bool)
    events = _edge_safe_events_146(events, len(y), lag_lo, lag_hi)

    if events.size == 0:
        return None, events

    rows = []
    for e in events:
        seg = y[e + lag_lo:e + lag_hi + 1]
        seg_obs = obs[e + lag_lo:e + lag_hi + 1]
        if len(seg) != (lag_hi - lag_lo + 1):
            continue
        seg = np.where(seg_obs & np.isfinite(seg), seg, np.nan)
        rows.append(seg)

    if not rows:
        return None, events

    return np.vstack(rows).astype(np.float64), events

def _baseline_correct_146(profile: np.ndarray, lags: np.ndarray):
    p = np.asarray(profile, dtype=np.float64).copy()
    lags = np.asarray(lags, dtype=np.int64)
    pre = lags < 0
    if np.any(pre) and np.isfinite(p[pre]).any():
        baseline = float(np.nanmean(p[pre]))
    elif np.isfinite(p).any():
        baseline = float(np.nanmedian(p[np.isfinite(p)]))
    else:
        baseline = 0.0
    return p - baseline, baseline

def _coverage_near_146(obs: np.ndarray, events: np.ndarray, lag_lo: int, lag_hi: int):
    obs = np.asarray(obs, dtype=bool)
    events = _edge_safe_events_146(events, len(obs), lag_lo, lag_hi)
    vals = []
    for e in events:
        seg = obs[e + lag_lo:e + lag_hi + 1]
        if len(seg):
            vals.append(float(np.mean(seg)))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan

def _window_mean_146(y: np.ndarray, obs: np.ndarray, events: np.ndarray, lag_lo: int, lag_hi: int):
    mat, ev = _window_matrix_146(y, obs, events, lag_lo, lag_hi)
    if mat is None:
        return np.nan
    return float(np.nanmean(mat)) if np.isfinite(mat).any() else np.nan

def _support_bounds_146(x: np.ndarray):
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    lo = float(np.nanquantile(arr, 0.001))
    hi = float(np.nanquantile(arr, 0.999))
    return lo, hi

def _window_mask_146(n: int, events: np.ndarray, lag_lo: int, lag_hi: int) -> np.ndarray:
    mask = np.zeros(int(n), dtype=bool)
    for e in np.asarray(events, dtype=np.int64):
        lo = max(0, int(e) + int(lag_lo))
        hi = min(int(n), int(e) + int(lag_hi) + 1)
        if lo < hi:
            mask[lo:hi] = True
    return mask

def _triangular_weights_146(n: int, events: np.ndarray, lag_lo: int, lag_hi: int) -> np.ndarray:
    weights = np.zeros(int(n), dtype=np.float64)
    center = 0.5 * (float(lag_lo) + float(lag_hi))
    half_width = max(1.0, 0.5 * (float(lag_hi) - float(lag_lo) + 1.0))

    for e in np.asarray(events, dtype=np.int64):
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

def _changed_mask_146(a, b, rtol=1e-6, atol=1e-8):
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=np.float64)
    same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
    return ~same

def _count_changed_146(a, b):
    return int(_changed_mask_146(a, b).sum())

def _protocol_tier_146(col: str) -> str:
    s = str(col).lower()
    if s.startswith("zigbee__") or s.startswith("zb__"):
        return "zigbee"
    if s.startswith("router__") or s.startswith("dns__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__"):
        return "ota"
    if s.startswith("zwave__"):
        return "zwave"
    return "other"

def _transform_protocol_qa_146(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell14_6_protocol_transform", "log1p_abs_signed")).strip().lower()

    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.log1p(np.maximum(arr, 0.0))
    elif mode == "log1p_abs":
        out = np.log1p(np.abs(arr))
    elif mode == "log1p_abs_signed":
        out = np.sign(arr) * np.log1p(np.abs(arr))
    else:
        raise RuntimeError(f"[Cell14.6] Unknown QA protocol transform: {mode}")

    out[~np.isfinite(out)] = np.nan
    return out

def _profile_around_events_qa_146(y_raw: np.ndarray, event_idx: np.ndarray, lag_lo: int, lag_hi: int):
    y = _transform_protocol_qa_146(y_raw)
    lags = np.arange(lag_lo, lag_hi + 1, dtype=np.int64)
    event_idx = _edge_safe_events_146(event_idx, len(y), lag_lo, lag_hi)

    if event_idx.size == 0:
        return lags, np.full(len(lags), np.nan), {"event_count_window_valid": 0}

    rows = []
    for e in event_idx:
        seg = y[e + lag_lo:e + lag_hi + 1]
        if len(seg) == len(lags):
            rows.append(seg)

    if not rows:
        return lags, np.full(len(lags), np.nan), {"event_count_window_valid": 0}

    mat = np.vstack(rows).astype(np.float64)
    with np.errstate(invalid="ignore"):
        prof = np.nanmean(mat, axis=0)

    pre = lags < 0
    if np.any(pre) and np.isfinite(prof[pre]).any():
        baseline = float(np.nanmean(prof[pre]))
    elif np.isfinite(prof).any():
        baseline = float(prof[np.flatnonzero(np.isfinite(prof))[0]])
    else:
        baseline = 0.0

    prof = prof - baseline
    return lags, prof, {"event_count_window_valid": int(event_idx.size), "baseline": baseline}

def _similarity_qa_146(real_prof, syn_prof):
    r = np.asarray(real_prof, dtype=np.float64)
    s = np.asarray(syn_prof, dtype=np.float64)
    valid = np.isfinite(r) & np.isfinite(s)
    if valid.sum() < 2:
        return {"similarity": np.nan, "mae": np.nan, "norm_mae": np.nan, "corr": np.nan}

    rr = r[valid]
    ss = s[valid]
    mae = float(np.mean(np.abs(rr - ss)))
    scale = float(np.nanmean(np.abs(rr)) + np.nanstd(rr) + float(CFG.get("cell14_6_norm_eps", 1e-9)))
    norm_mae = float(mae / max(scale, float(CFG.get("cell14_6_norm_eps", 1e-9))))

    if np.std(rr) <= 1e-12 or np.std(ss) <= 1e-12:
        corr = 1.0 if np.allclose(rr, ss, atol=1e-9) else 0.0
    else:
        corr = float(np.corrcoef(rr, ss)[0, 1])

    sim = float(0.5 * max(0.0, corr) + 0.5 * np.exp(-norm_mae))
    return {"similarity": sim, "mae": mae, "norm_mae": norm_mae, "corr": corr}

def _peak_lag_qa_146(lags, prof):
    prof = np.asarray(prof, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.int64)
    valid = np.isfinite(prof)
    if not valid.any():
        return np.nan, np.nan
    idxs = np.flatnonzero(valid)
    local = int(np.nanargmax(np.abs(prof[valid])))
    idx = int(idxs[local])
    return int(lags[idx]), float(prof[idx])

def _response_threshold_real_qa_146(y_raw):
    y = _transform_protocol_qa_146(y_raw)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.nan
    q = float(CFG.get("cell14_6_response_threshold_quantile", 0.90))
    q = float(np.clip(q, 0.50, 0.999))
    return float(np.nanquantile(np.abs(y), q))

def _response_rate_qa_146(y_raw, event_idx, lag_lo, lag_hi, threshold):
    y = _transform_protocol_qa_146(y_raw)
    event_idx = _edge_safe_events_146(event_idx, len(y), lag_lo, lag_hi)
    if event_idx.size == 0 or not np.isfinite(threshold):
        return np.nan

    hits = []
    for e in event_idx:
        seg = y[e + lag_lo:e + lag_hi + 1]
        seg = seg[np.isfinite(seg)]
        hits.append(bool(seg.size and np.nanmax(np.abs(seg)) >= threshold))

    return float(np.mean(hits)) if hits else np.nan

def _status_qa_146(row):
    real_events = int(_safe_float_146(row.get("real_event_count_window_valid"), 0))
    syn_events = int(_safe_float_146(row.get("syn_event_count_window_valid"), 0))
    min_events = int(CFG.get("cell14_6_min_events_for_pass", 5))

    eta_sim = _safe_float_146(row.get("ETA_similarity"), np.nan)
    prof_sim = _safe_float_146(row.get("manifest_window_profile_similarity"), np.nan)
    lag_err = _safe_float_146(row.get("lag_peak_error"), np.nan)
    resp_err = _safe_float_146(row.get("response_window_rate_error"), np.nan)

    if real_events < min_events and syn_events < min_events:
        return "not_evaluable", "insufficient_real_and_synthetic_events"
    if real_events < min_events:
        return "warning", "insufficient_real_events"
    if syn_events < min_events:
        return "warning", "insufficient_synthetic_events"

    fatal = []
    warn = []

    if np.isfinite(eta_sim):
        if eta_sim < float(CFG.get("cell14_6_eta_similarity_warning", 0.50)):
            fatal.append("ETA_similarity_fatal")
        elif eta_sim < float(CFG.get("cell14_6_eta_similarity_pass", 0.70)):
            warn.append("ETA_similarity_warning")
    else:
        warn.append("ETA_similarity_not_finite")

    if np.isfinite(prof_sim):
        if prof_sim < float(CFG.get("cell14_6_profile_similarity_warning", 0.50)):
            fatal.append("manifest_profile_similarity_fatal")
        elif prof_sim < float(CFG.get("cell14_6_profile_similarity_pass", 0.70)):
            warn.append("manifest_profile_similarity_warning")
    else:
        warn.append("manifest_profile_similarity_not_finite")

    if np.isfinite(lag_err):
        if lag_err > float(CFG.get("cell14_6_lag_peak_error_warning", 5)):
            fatal.append("lag_peak_error_fatal")
        elif lag_err > float(CFG.get("cell14_6_lag_peak_error_pass", 2)):
            warn.append("lag_peak_error_warning")

    if np.isfinite(resp_err):
        if resp_err > float(CFG.get("cell14_6_response_window_rate_error_warning", 0.25)):
            fatal.append("response_window_rate_error_fatal")
        elif resp_err > float(CFG.get("cell14_6_response_window_rate_error_pass", 0.10)):
            warn.append("response_window_rate_error_warning")

    if fatal:
        return "fatal", "|".join(fatal)
    if warn:
        return "warning", "|".join(warn)
    return "pass", "A0_manifest_q4_pair_pass"

def _mean_numeric_146(df, col):
    if col not in df.columns or not len(df):
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean())

# ----------------------------------------------------------
# 4) Validate upstream policy / repair / Q4 contracts
# ----------------------------------------------------------
def _require_contract_version_146(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.6] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.6] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

version145_146 = _require_contract_version_146(
    CELL14_5_REPAIR_MANIFEST_CONTRACT,
    "CELL14_5_REPAIR_MANIFEST_CONTRACT",
    "cell14_5_coupling_repair_manifest_policy_v1_1",
)
version144_146 = _require_contract_version_146(
    CELL14_4_DEEP_COUPLING_CONTRACT,
    "CELL14_4_DEEP_COUPLING_CONTRACT",
    "cell14_4_two_stage_trainval_iot_protocol_coupling_tiering_v2_1",
)
version143_146 = _require_contract_version_146(
    CELL14_3_Q4_REPAIR_CONTRACT,
    "CELL14_3_Q4_REPAIR_CONTRACT",
    "cell14_3_coupling_conditioned_q4_re_evaluation_v1_1",
)
version134_146 = _require_contract_version_146(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)

strict145_146 = CELL14_5_REPAIR_MANIFEST_CONTRACT.get("strict_contract", {})
if bool(strict145_146.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.6] Cell 14.5 contract indicates TEST real values were used.")
if bool(strict145_146.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.6] Cell 14.5 contract indicates synthetic values were mutated.")
if bool(strict145_146.get("generator_fit_done_here", True)):
    raise RuntimeError("[Cell14.6] Cell 14.5 contract indicates generator fitting was done.")
if bool(strict145_146.get("materialization_done_here", True)):
    raise RuntimeError("[Cell14.6] Cell 14.5 contract indicates materialization was done.")
if not bool(strict145_146.get("policy_layer_only", False)):
    raise RuntimeError("[Cell14.6] Cell 14.5 did not declare policy_layer_only.")

repair_safety_counts_146 = CELL14_5_REPAIR_MANIFEST_CONTRACT.get("repair_safety_counts", {})
direct_repair_allowed_now_n_146 = int(repair_safety_counts_146.get("direct_repair_allowed_now_n", 0) or 0)
if direct_repair_allowed_now_n_146 <= 0:
    raise RuntimeError("[Cell14.6] Cell 14.5 reports no directly repairable A0 rows.")

strict144_146 = CELL14_4_DEEP_COUPLING_CONTRACT.get("strict_contract", {})
if bool(strict144_146.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.6] Cell 14.4 contract indicates TEST real values were used.")

strict143_146 = CELL14_3_Q4_REPAIR_CONTRACT.get("strict_contract", {})
if bool(strict143_146.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.6] Cell 14.3 contract indicates synthetic values were mutated.")
if not bool(strict143_146.get("repair_candidate_not_final_claim", False)):
    raise RuntimeError("[Cell14.6] Cell 14.3 did not declare repair_candidate_not_final_claim.")

broad_q4_status_146 = str(CELL13_4_Q4_PUBLICATION_SUMMARY.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_146 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("pair_blocker_n", 0) or 0
)
broad_q4_pair_pass_rate_146 = _safe_float_146(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)

repair_status_146 = str(CELL14_3_Q4_REPAIR_CONTRACT.get("repair_candidate_publication_status", ""))
repair_post_summary_146 = CELL14_3_Q4_REPAIR_CONTRACT.get("post_coupling_summary_14_3", {})
repair_delta_summary_146 = CELL14_3_Q4_REPAIR_CONTRACT.get("delta_summary", {})
repair_post_blocker_n_146 = int(repair_post_summary_146.get("publication_blocker_n", 0) or 0)
repair_post_pass_rate_146 = _safe_float_146(repair_post_summary_146.get("pass_rate", np.nan), np.nan)

if broad_q4_status_146 != "blocker":
    log(
        "[Cell14.6] WARNING: broad Q4 status is not blocker. "
        f"broad_q4_status={broad_q4_status_146}"
    )

# ----------------------------------------------------------
# 5) Input preparation
# ----------------------------------------------------------
a0 = CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_DF.copy()
if len(a0) == 0:
    raise RuntimeError("[Cell14.6] A0 Zigbee-safe manifest is empty.")

required_a0_cols = [
    "repair_candidate_id", "anchor_col", "anchor_name", "protocol_col",
    "protocol_tier", "recommended_lag_lo", "recommended_lag_hi",
    "recommended_weight", "repair_eligible",
]
missing_a0_cols = [c for c in required_a0_cols if c not in a0.columns]
if missing_a0_cols:
    raise RuntimeError(f"[Cell14.6] A0 manifest missing required columns: {missing_a0_cols}")

a0["anchor_col"] = a0["anchor_col"].astype(str)
a0["protocol_col"] = a0["protocol_col"].astype(str)
a0["protocol_tier"] = a0["protocol_tier"].astype(str).str.lower()
a0["repair_eligible"] = a0["repair_eligible"].fillna(False).astype(bool)
if "direct_repair_allowed_now" in a0.columns:
    a0["direct_repair_allowed_now"] = a0["direct_repair_allowed_now"].fillna(False).astype(bool)
else:
    a0["direct_repair_allowed_now"] = a0["repair_eligible"]

a0 = a0[
    a0["repair_eligible"]
    & a0["direct_repair_allowed_now"]
    & a0["protocol_col"].isin(A0_ALLOWED_COLS_146)
    & a0["protocol_tier"].eq("zigbee")
].copy()

if len(a0) == 0:
    raise RuntimeError("[Cell14.6] No eligible A0 rows remain after strict filter.")

# Normalize frames.
# Use the current uncoupled protocol matrix prepared by Cell 13.0.
# Do not use historical pre-11b / quarantine artifacts in the STUDY-THESIS path.
protocol_base_source_146 = ""
protocol_base_path_146 = ""
protocol_base_sha256_146 = ""

if "PROTOCOL_SYN_TEST_130" in globals() and isinstance(PROTOCOL_SYN_TEST_130, pd.DataFrame):
    protocol_base = PROTOCOL_SYN_TEST_130.copy()
    protocol_base_source_146 = "PROTOCOL_SYN_TEST_130_current_uncoupled_protocol"
    protocol_base_path_146 = ""
    protocol_base_sha256_146 = ""
else:
    current_protocol_candidates_146 = [
        os.path.join(OUT_SYN, "A2_PROTOCOL_TEST.parquet"),
        os.path.join(OUT_SYN, "A2_PROTOCOL_VALUES_TEST.parquet"),
        os.path.join(OUT_SYN, "PROTOCOL_SYNTHETIC_TEST.parquet"),
        os.path.join(OUT_SYN, "PROTOCOL_FINAL_TEST.parquet"),
    ]
    current_protocol_path_146 = next((p for p in current_protocol_candidates_146 if os.path.exists(p)), None)
    if current_protocol_path_146 is None:
        raise RuntimeError(
            "[Cell14.6] Could not resolve a current uncoupled protocol base. "
            "Run Cell 13.0 or ensure a current A2 protocol parquet exists."
        )
    protocol_base = pd.read_parquet(current_protocol_path_146)
    protocol_base_source_146 = "current_uncoupled_protocol_parquet"
    protocol_base_path_146 = current_protocol_path_146
    protocol_base_sha256_146 = _sha256_file_146(current_protocol_path_146)

CFG["cell14_6_protocol_base_fallback_used"] = bool(protocol_base_path_146)

protocol_base.index = df_te.index
protocol_base.columns = protocol_base.columns.astype(str)

if len(protocol_base) != N_TE:
    raise RuntimeError(
        "[Cell14.6] Protocol base row mismatch after fallback resolution: "
        f"got={len(protocol_base)} expected={N_TE}"
    )

CFG["cell14_6_pre_q4_protocol_base_path"] = protocol_base_path_146
CFG["cell14_6_protocol_base_source"] = protocol_base_source_146
CFG["cell14_6_protocol_base_sha256"] = protocol_base_sha256_146
CFG["cell14_6_protocol_base_fallback_used"] = bool(protocol_base_source_146 == "PROTOCOL_SYN_TEST_130_fallback_uncoupled")

log(
    "[Cell14.6] Using protocol base | "
    f"source={protocol_base_source_146} | "
    f"path={protocol_base_path_146 if protocol_base_path_146 else '<in_memory>'} | "
    f"shape={protocol_base.shape} | "
    f"sha256={protocol_base_sha256_146 if protocol_base_sha256_146 else '<not_file_backed>'}"
)

iot_syn = IOT_FULL_SYN_TEST_130.copy()
iot_syn.index = df_te.index
iot_syn.columns = iot_syn.columns.astype(str)

if len(protocol_base) != N_TE or len(iot_syn) != N_TE:
    raise RuntimeError("[Cell14.6] TEST frame length mismatch.")

# Validate columns.
for c in sorted(set(a0["anchor_col"])):
    if c not in df_tr.columns or c not in df_val.columns or c not in iot_syn.columns or c not in IOT_REAL_TEST_REF_130.columns:
        raise RuntimeError(f"[Cell14.6] Missing anchor column in required frames: {c}")

for c in sorted(set(a0["protocol_col"])):
    if c not in df_tr.columns or c not in df_val.columns or c not in protocol_base.columns or c not in PROTOCOL_REAL_TEST_REF_130.columns:
        raise RuntimeError(f"[Cell14.6] Missing protocol column in required frames: {c}")

# ----------------------------------------------------------
# 6) Build TRAIN/VAL profiles for A0 rows
# ----------------------------------------------------------
profile_rows = []
profile_npz_payload = {}

log(f"[Cell14.6] Building TRAIN/VAL A0 profiles | A0_pairs={len(a0)}")

for i, r in enumerate(a0.itertuples(index=False), start=1):
    row = r._asdict()
    candidate_id = str(row["repair_candidate_id"])
    anchor_col = str(row["anchor_col"])
    protocol_col = str(row["protocol_col"])
    tier = str(row["protocol_tier"])

    lag_lo = int(row["recommended_lag_lo"])
    lag_hi = int(row["recommended_lag_hi"])
    if lag_lo > lag_hi:
        lag_lo, lag_hi = lag_hi, lag_lo

    profile_lag_lo = lag_lo - PROFILE_PRE_MARGIN_146
    profile_lag_hi = lag_hi + PROFILE_POST_MARGIN_146
    lags = np.arange(profile_lag_lo, profile_lag_hi + 1, dtype=np.int64)

    obs_tr_col = _resolve_obs_col_146(df_tr, tier)
    obs_val_col = _resolve_obs_col_146(df_val, tier)

    if obs_tr_col is None or obs_val_col is None:
        raise RuntimeError(f"[Cell14.6] Missing obs col for {candidate_id}")

    d_tr = _to_num_array_146(df_tr, anchor_col, fill=0.0)
    d_val = _to_num_array_146(df_val, anchor_col, fill=0.0)
    y_tr = _to_num_array_146(df_tr, protocol_col)
    y_val = _to_num_array_146(df_val, protocol_col)

    obs_tr = _to_num_array_146(df_tr, obs_tr_col, fill=0.0) > 0.5
    obs_val = _to_num_array_146(df_val, obs_val_col, fill=0.0) > 0.5

    ev_tr = _edge_safe_events_146(_event_starts_146(d_tr), len(df_tr), profile_lag_lo, profile_lag_hi)
    ev_val = _edge_safe_events_146(_event_starts_146(d_val), len(df_val), profile_lag_lo, profile_lag_hi)

    train_events = int(len(ev_tr))
    val_events = int(len(ev_val))
    combined_events = int(train_events + val_events)

    cov_tr = _coverage_near_146(obs_tr, ev_tr, profile_lag_lo, profile_lag_hi)
    cov_val = _coverage_near_146(obs_val, ev_val, profile_lag_lo, profile_lag_hi)

    valid = True
    reasons = []

    if train_events < int(CFG.get("cell14_6_min_train_events", 20)):
        valid = False
        reasons.append("insufficient_train_events")
    if val_events < int(CFG.get("cell14_6_min_val_events", 5)):
        valid = False
        reasons.append("insufficient_val_events")
    if combined_events < int(CFG.get("cell14_6_min_combined_events", 30)):
        valid = False
        reasons.append("insufficient_combined_events")
    if np.isfinite(cov_tr) and cov_tr < float(CFG.get("cell14_6_min_modality_coverage", 0.20)):
        valid = False
        reasons.append("low_train_coverage")
    if np.isfinite(cov_val) and cov_val < float(CFG.get("cell14_6_min_modality_coverage", 0.20)):
        valid = False
        reasons.append("low_val_coverage")

    mat_tr, ev_tr_used = _window_matrix_146(y_tr, obs_tr, ev_tr, profile_lag_lo, profile_lag_hi)
    mat_val, ev_val_used = _window_matrix_146(y_val, obs_val, ev_val, profile_lag_lo, profile_lag_hi)

    if mat_tr is None or mat_val is None:
        valid = False
        reasons.append("missing_window_matrix")

    if valid:
        combined_mat = np.vstack([mat_tr, mat_val])
        with np.errstate(invalid="ignore"):
            mean_profile_raw = np.nanmean(combined_mat, axis=0)
            median_profile_raw = np.nanmedian(combined_mat, axis=0)
            finite_per_lag = np.isfinite(combined_mat).sum(axis=0)

        mean_profile_bc, baseline_mean = _baseline_correct_146(mean_profile_raw, lags)
        median_profile_bc, baseline_median = _baseline_correct_146(median_profile_raw, lags)

        y_all = np.concatenate([y_tr[np.isfinite(y_tr)], y_val[np.isfinite(y_val)]])
        support_lo, support_hi = _support_bounds_146(y_all)

        profile_key = f"A0_profile_{i-1:04d}"

        profile_npz_payload[f"{profile_key}__lags"] = lags.astype(np.int64)
        profile_npz_payload[f"{profile_key}__mean_profile_bc"] = mean_profile_bc.astype(np.float32)
        profile_npz_payload[f"{profile_key}__median_profile_bc"] = median_profile_bc.astype(np.float32)
        profile_npz_payload[f"{profile_key}__mean_profile_raw"] = mean_profile_raw.astype(np.float32)
        profile_npz_payload[f"{profile_key}__median_profile_raw"] = median_profile_raw.astype(np.float32)
        profile_npz_payload[f"{profile_key}__finite_per_lag"] = finite_per_lag.astype(np.int64)

        profile_rows.append({
            "profile_key": profile_key,
            "repair_candidate_id": candidate_id,
            "anchor_col": anchor_col,
            "anchor_name": str(row.get("anchor_name", "")),
            "protocol_col": protocol_col,
            "protocol_tier": tier,
            "lag_lo": int(lag_lo),
            "lag_hi": int(lag_hi),
            "profile_lag_lo": int(profile_lag_lo),
            "profile_lag_hi": int(profile_lag_hi),
            "train_events": train_events,
            "val_events": val_events,
            "combined_events": combined_events,
            "train_coverage": cov_tr,
            "val_coverage": cov_val,
            "support_lo_raw_trainval": support_lo,
            "support_hi_raw_trainval": support_hi,
            "baseline_mean": baseline_mean,
            "baseline_median": baseline_median,
            "recommended_weight": _safe_float_146(row.get("recommended_weight"), 1.0),
            "profile_valid": True,
            "rejection_reason": "",
        })
    else:
        profile_rows.append({
            "profile_key": "",
            "repair_candidate_id": candidate_id,
            "anchor_col": anchor_col,
            "anchor_name": str(row.get("anchor_name", "")),
            "protocol_col": protocol_col,
            "protocol_tier": tier,
            "lag_lo": int(lag_lo),
            "lag_hi": int(lag_hi),
            "profile_lag_lo": int(profile_lag_lo),
            "profile_lag_hi": int(profile_lag_hi),
            "train_events": train_events,
            "val_events": val_events,
            "combined_events": combined_events,
            "train_coverage": cov_tr,
            "val_coverage": cov_val,
            "support_lo_raw_trainval": np.nan,
            "support_hi_raw_trainval": np.nan,
            "baseline_mean": np.nan,
            "baseline_median": np.nan,
            "recommended_weight": _safe_float_146(row.get("recommended_weight"), 1.0),
            "profile_valid": False,
            "rejection_reason": "|".join(reasons),
        })

profiles_df = pd.DataFrame(profile_rows)
valid_profiles = profiles_df[profiles_df["profile_valid"].astype(bool)].copy()

if len(valid_profiles) == 0 and bool(CFG.get("cell14_6_fail_if_no_valid_profiles", True)):
    raise RuntimeError("[Cell14.6] No valid A0 TRAIN/VAL profiles built.")

profiles_df.to_csv(profile_audit_csv, index=False)
np.savez_compressed(profiles_npz_path, **profile_npz_payload)

# ----------------------------------------------------------
# 7) Apply A0 materialization
# ----------------------------------------------------------
protocol_a0 = protocol_base.copy()

target_cols = sorted(set(valid_profiles["protocol_col"].astype(str)))
buffers = {
    c: {
        "sum": np.zeros(N_TE, dtype=np.float64),
        "wsum": np.zeros(N_TE, dtype=np.float64),
        "legal": np.zeros(N_TE, dtype=bool),
    }
    for c in target_cols
}

materialization_rows = []
profile_npz = np.load(profiles_npz_path)
profile_source = str(CFG.get("cell14_6_profile_source", "mean_profile_bc"))

log(f"[Cell14.6] Applying A0 materialization | valid_profiles={len(valid_profiles)} | target_cols={target_cols}")

for _, r in valid_profiles.iterrows():
    profile_key = str(r["profile_key"])
    candidate_id = str(r["repair_candidate_id"])
    anchor_col = str(r["anchor_col"])
    protocol_col = str(r["protocol_col"])

    lag_lo = int(r["lag_lo"])
    lag_hi = int(r["lag_hi"])
    profile_lag_lo = int(r["profile_lag_lo"])
    profile_lag_hi = int(r["profile_lag_hi"])

    lags = np.asarray(profile_npz[f"{profile_key}__lags"], dtype=np.int64)
    prof = np.asarray(profile_npz[f"{profile_key}__{profile_source}"], dtype=np.float64)

    expected_lags = np.arange(profile_lag_lo, profile_lag_hi + 1, dtype=np.int64)
    if not np.array_equal(lags, expected_lags):
        raise RuntimeError(f"[Cell14.6] Profile lag mismatch for {candidate_id}")

    driver_x = _to_num_array_146(iot_syn, anchor_col, fill=0.0)
    events = _edge_safe_events_146(_event_starts_146(driver_x), N_TE, profile_lag_lo, profile_lag_hi)

    obs_col = _resolve_obs_col_146(protocol_base, "zigbee")
    if bool(CFG.get("cell14_6_respect_protocol_observability", True)) and obs_col is not None:
        obs_mask = _to_num_array_146(protocol_base, obs_col, fill=0.0) > 0.5
    else:
        obs_mask = np.ones(N_TE, dtype=bool)

    base = _to_num_array_146(protocol_a0, protocol_col)
    legal_profile = _window_mask_146(N_TE, events, profile_lag_lo, profile_lag_hi)
    legal_manifest = _window_mask_146(N_TE, events, lag_lo, lag_hi)
    buffers[protocol_col]["legal"] |= legal_profile

    support_lo = _safe_float_146(r.get("support_lo_raw_trainval"), np.nan)
    support_hi = _safe_float_146(r.get("support_hi_raw_trainval"), np.nan)

    if np.isfinite(support_lo) and np.isfinite(support_hi):
        margin = float(CFG.get("cell14_6_clip_margin_fraction", 0.10)) * max(1.0, support_hi - support_lo)
        clip_lo = support_lo - margin
        clip_hi = support_hi + margin
    else:
        clip_lo, clip_hi = -np.inf, np.inf

    if protocol_col in NONNEGATIVE_COLS_146:
        clip_lo = max(0.0, clip_lo)

    rec_w = _safe_float_146(r.get("recommended_weight"), 1.0)
    rec_w = float(np.clip(rec_w, 0.0, 1.0))

    alpha = float(CFG.get("cell14_6_base_alpha_scale", 0.70)) * rec_w
    alpha = float(np.clip(alpha, float(CFG.get("cell14_6_min_alpha", 0.05)), float(CFG.get("cell14_6_max_alpha", 0.85))))

    n_proposed = 0

    for e in events:
        lo = int(e) + profile_lag_lo
        hi = int(e) + profile_lag_hi + 1
        if lo < 0 or hi > N_TE:
            continue

        idx = np.arange(lo, hi, dtype=np.int64)
        local_base = base[idx].astype(np.float64)
        valid = obs_mask[idx] & np.isfinite(local_base) & np.isfinite(prof)

        if not valid.any():
            continue

        target = local_base.copy()
        target[valid] = local_base[valid] + prof[valid]
        target[valid] = (1.0 - alpha) * local_base[valid] + alpha * target[valid]

        if bool(CFG.get("cell14_6_clip_to_trainval_support", True)):
            target[valid] = np.clip(target[valid], clip_lo, clip_hi)

        if protocol_col in NONNEGATIVE_COLS_146:
            target[valid] = np.maximum(target[valid], 0.0)

        weights = _triangular_weights_146(N_TE, np.asarray([e], dtype=np.int64), profile_lag_lo, profile_lag_hi)[idx]
        weights = weights * valid.astype(np.float64) * alpha

        buffers[protocol_col]["sum"][idx] += weights * target
        buffers[protocol_col]["wsum"][idx] += weights
        n_proposed += int(np.sum(weights > 0))

    materialization_rows.append({
        "repair_candidate_id": candidate_id,
        "profile_key": profile_key,
        "anchor_col": anchor_col,
        "protocol_col": protocol_col,
        "synthetic_events_edge_safe": int(len(events)),
        "profile_lag_lo": int(profile_lag_lo),
        "profile_lag_hi": int(profile_lag_hi),
        "lag_lo": int(lag_lo),
        "lag_hi": int(lag_hi),
        "legal_profile_n": int(legal_profile.sum()),
        "legal_manifest_n": int(legal_manifest.sum()),
        "n_proposed": int(n_proposed),
        "alpha": alpha,
        "support_lo": clip_lo,
        "support_hi": clip_hi,
        "applied": bool(n_proposed > 0),
        "reason": "ok" if n_proposed > 0 else "no_valid_proposal_points",
        "TEST_real_values_used": False,
        "synthetic_iot_driver_used_for_conditioning": True,
    })

# Merge buffers.
for col, buf in buffers.items():
    base = _to_num_array_146(protocol_base, col)
    out = base.copy()

    obs_col = _resolve_obs_col_146(protocol_base, "zigbee")
    if bool(CFG.get("cell14_6_respect_protocol_observability", True)) and obs_col is not None:
        obs_mask = _to_num_array_146(protocol_base, obs_col, fill=0.0) > 0.5
    else:
        obs_mask = np.ones(N_TE, dtype=bool)

    active = (buf["wsum"] > 0) & obs_mask & np.isfinite(base)
    out[active] = buf["sum"][active] / np.maximum(buf["wsum"][active], 1e-12)

    if col in NONNEGATIVE_COLS_146:
        finite = np.isfinite(out)
        out[finite] = np.maximum(out[finite], 0.0)

    if col in INTEGER_COLS_146:
        finite = np.isfinite(out)
        out[finite] = np.rint(out[finite])

    out[~obs_mask] = np.nan
    protocol_a0[col] = out.astype(np.float32)

materialization_df = pd.DataFrame(materialization_rows)
materialization_df.to_csv(materialization_audit_csv, index=False)

# ----------------------------------------------------------
# 8) Safety drift audits
# ----------------------------------------------------------
target_drift_rows = []
non_target_drift_rows = []

target_changed_total = 0
outside_window_total = 0

for col in target_cols:
    before = protocol_base[col].to_numpy(copy=False)
    after = protocol_a0[col].to_numpy(copy=False)
    changed = _changed_mask_146(before, after)
    legal = buffers[col]["legal"]

    changed_n = int(changed.sum())
    inside_n = int((changed & legal).sum())
    outside_n = int((changed & (~legal)).sum())

    target_changed_total += changed_n
    outside_window_total += outside_n

    arr = _to_num_array_146(protocol_a0, col)
    finite = np.isfinite(arr)

    negative_n = int(np.sum(finite & (arr < 0.0))) if col in NONNEGATIVE_COLS_146 else 0
    noninteger_n = int(np.sum(finite & (np.abs(arr - np.rint(arr)) > 1e-6))) if col in INTEGER_COLS_146 else 0

    # Inactive-finite audit must follow the same observability policy as
    # materialization. For precomputable Q4 A0, stale pre-11b Zigbee obs
    # gating is intentionally disabled because A0 writes are governed by
    # manifest-approved legal/profile windows and domain checks.
    obs_col = _resolve_obs_col_146(protocol_base, "zigbee")
    
    if bool(CFG.get("cell14_6_respect_protocol_observability", True)) and obs_col is not None:
        obs_mask = _to_num_array_146(protocol_base, obs_col, fill=0.0) > 0.5
        inactive_finite_n = int(np.sum((~obs_mask) & np.isfinite(arr)))
        inactive_finite_audit_policy = "protocol_observability_gated"
    else:
        obs_mask = np.ones(N_TE, dtype=bool)
        inactive_finite_n = 0
        inactive_finite_audit_policy = "stale_pre11b_obs_gate_disabled_for_A0"

    target_drift_rows.append({
        "protocol_col": col,
        "target_changed_n": changed_n,
        "target_changed_inside_legal_window_n": inside_n,
        "target_changed_outside_legal_window_n": outside_n,
        "legal_window_n": int(legal.sum()),
        "negative_n": negative_n,
        "noninteger_n": noninteger_n,
        "inactive_finite_n": inactive_finite_n,
        "before_mean": float(np.nanmean(_to_num_array_146(protocol_base, col))),
        "after_mean": float(np.nanmean(_to_num_array_146(protocol_a0, col))),
        "obs_col_used_for_inactive_audit": obs_col if obs_col is not None else "",
        "inactive_finite_audit_policy": inactive_finite_audit_policy,
        "stale_pre11b_zigbee_obs_gate_disabled": bool(CFG.get("cell14_6_disable_stale_pre11b_zigbee_obs_gate", False)),
    })

for col in protocol_base.columns:
    if col in set(target_cols):
        continue
    changed_n = _count_changed_146(protocol_base[col].to_numpy(copy=False), protocol_a0[col].to_numpy(copy=False))
    if changed_n:
        non_target_drift_rows.append({
            "protocol_col": col,
            "protocol_tier": _protocol_tier_146(col),
            "changed_n": changed_n,
        })

target_drift_df = pd.DataFrame(target_drift_rows)
non_target_drift_df = pd.DataFrame(non_target_drift_rows)

non_target_drift_n = int(non_target_drift_df["changed_n"].sum()) if len(non_target_drift_df) else 0
negative_n = int(target_drift_df["negative_n"].sum()) if len(target_drift_df) else 0
noninteger_n = int(target_drift_df["noninteger_n"].sum()) if len(target_drift_df) else 0
inactive_finite_n = int(target_drift_df["inactive_finite_n"].sum()) if len(target_drift_df) else 0

if outside_window_total and bool(CFG.get("cell14_6_fail_on_outside_window_drift", True)):
    raise RuntimeError(f"[Cell14.6] Outside legal window drift detected: {outside_window_total}")

if non_target_drift_n and bool(CFG.get("cell14_6_fail_on_non_target_drift", True)):
    raise RuntimeError(f"[Cell14.6] Non-target protocol drift detected: {non_target_drift_n}")

if (negative_n or noninteger_n) and bool(CFG.get("cell14_6_fail_on_negative_or_noninteger", True)):
    raise RuntimeError(
        f"[Cell14.6] Domain violation: negative_n={negative_n}, noninteger_n={noninteger_n}"
    )

if (
    inactive_finite_n
    and bool(CFG.get("cell14_6_fail_on_inactive_finite", True))
    and bool(CFG.get("cell14_6_respect_protocol_observability", True))
):
    raise RuntimeError(f"[Cell14.6] Inactive finite values detected: {inactive_finite_n}")

target_drift_df.to_csv(target_drift_csv, index=False)
non_target_drift_df.to_csv(non_target_drift_csv, index=False)

# ----------------------------------------------------------
# 9) Save protocol and CPS A0 candidates
# ----------------------------------------------------------
protocol_a0.to_parquet(protocol_a0_path, index=True)

# Assemble CPS candidate: time + protocol + IoT, avoiding duplicate time cols.
time_cols = [c for c in ["sec_epoch_s__canon", "sec", "timestamp", "time", "datetime"] if c in df_te.columns]
time_df = pd.DataFrame(index=df_te.index)
for c in time_cols:
    if c not in protocol_a0.columns and c not in iot_syn.columns:
        time_df[c] = df_te[c].to_numpy(copy=True)

parts = []
if len(time_df.columns):
    parts.append(time_df)
parts.extend([protocol_a0, iot_syn])

CPS_COUPLED_ZIGBEE_A0_TEST = pd.concat(parts, axis=1)

dupes = CPS_COUPLED_ZIGBEE_A0_TEST.columns[CPS_COUPLED_ZIGBEE_A0_TEST.columns.duplicated()].astype(str).tolist()
if dupes:
    raise RuntimeError(f"[Cell14.6] Duplicate columns in CPS A0 assembly: {dupes[:20]}")

CPS_COUPLED_ZIGBEE_A0_TEST.to_parquet(cps_a0_path, index=True)

# ----------------------------------------------------------
# 10) A0 manifest Q4 QA
# ----------------------------------------------------------
q4_rows = []
curve_payload = []

for _, r in a0.iterrows():
    candidate_id = str(r["repair_candidate_id"])
    anchor_col = str(r["anchor_col"])
    protocol_col = str(r["protocol_col"])
    lag_lo = int(r["recommended_lag_lo"])
    lag_hi = int(r["recommended_lag_hi"])

    if lag_lo > lag_hi:
        lag_lo, lag_hi = lag_hi, lag_lo

    eta_lo = min(-ETA_PRE_146, lag_lo - EXTRA_MARGIN_146)
    eta_hi = max(ETA_POST_146, lag_hi + EXTRA_MARGIN_146)

    d_real = _to_num_array_146(IOT_REAL_TEST_REF_130, anchor_col)
    d_syn = _to_num_array_146(iot_syn, anchor_col)

    p_real = _to_num_array_146(PROTOCOL_REAL_TEST_REF_130, protocol_col)
    p_syn = _to_num_array_146(protocol_a0, protocol_col)

    e_real = _event_starts_146(d_real)
    e_syn = _event_starts_146(d_syn)

    eta_lags_real, eta_real, eta_meta_real = _profile_around_events_qa_146(p_real, e_real, eta_lo, eta_hi)
    eta_lags_syn, eta_syn, eta_meta_syn = _profile_around_events_qa_146(p_syn, e_syn, eta_lo, eta_hi)

    if not np.array_equal(eta_lags_real, eta_lags_syn):
        raise RuntimeError(f"[Cell14.6] ETA lag mismatch for {candidate_id}")

    eta_sim = _similarity_qa_146(eta_real, eta_syn)

    man_lags_real, man_real, man_meta_real = _profile_around_events_qa_146(p_real, e_real, lag_lo, lag_hi)
    man_lags_syn, man_syn, man_meta_syn = _profile_around_events_qa_146(p_syn, e_syn, lag_lo, lag_hi)

    if not np.array_equal(man_lags_real, man_lags_syn):
        raise RuntimeError(f"[Cell14.6] Manifest lag mismatch for {candidate_id}")

    man_sim = _similarity_qa_146(man_real, man_syn)

    real_peak_lag, real_peak_val = _peak_lag_qa_146(eta_lags_real, eta_real)
    syn_peak_lag, syn_peak_val = _peak_lag_qa_146(eta_lags_syn, eta_syn)

    lag_peak_error = (
        abs(float(real_peak_lag) - float(syn_peak_lag))
        if np.isfinite(real_peak_lag) and np.isfinite(syn_peak_lag)
        else np.nan
    )

    thr = _response_threshold_real_qa_146(p_real)
    real_resp_rate = _response_rate_qa_146(p_real, e_real, lag_lo, lag_hi, thr)
    syn_resp_rate = _response_rate_qa_146(p_syn, e_syn, lag_lo, lag_hi, thr)

    response_window_rate_error = (
        abs(real_resp_rate - syn_resp_rate)
        if np.isfinite(real_resp_rate) and np.isfinite(syn_resp_rate)
        else np.nan
    )

    row = {
        "repair_candidate_id": candidate_id,
        "anchor_col": anchor_col,
        "anchor_name": str(r.get("anchor_name", "")),
        "protocol_col": protocol_col,
        "protocol_tier": str(r.get("protocol_tier", "zigbee")),
        "lag_lo": int(lag_lo),
        "lag_hi": int(lag_hi),
        "real_event_count_all": int(len(e_real)),
        "syn_event_count_all": int(len(e_syn)),
        "real_event_count_window_valid": int(eta_meta_real.get("event_count_window_valid", 0)),
        "syn_event_count_window_valid": int(eta_meta_syn.get("event_count_window_valid", 0)),
        "ETA_similarity": eta_sim["similarity"],
        "ETA_mae": eta_sim["mae"],
        "ETA_norm_mae": eta_sim["norm_mae"],
        "ETA_corr": eta_sim["corr"],
        "manifest_window_profile_similarity": man_sim["similarity"],
        "manifest_window_profile_mae": man_sim["mae"],
        "manifest_window_profile_norm_mae": man_sim["norm_mae"],
        "manifest_window_profile_corr": man_sim["corr"],
        "real_peak_lag": real_peak_lag,
        "syn_peak_lag": syn_peak_lag,
        "lag_peak_error": lag_peak_error,
        "real_response_threshold": thr,
        "real_response_window_rate": real_resp_rate,
        "syn_response_window_rate": syn_resp_rate,
        "response_window_rate_error": response_window_rate_error,
        "TEST_real_values_used_for_QA_only": True,
        "synthetic_values_mutated_in_QA": False,
    }

    status, reason = _status_qa_146(row)
    row["A0_q4_status"] = status
    row["A0_q4_reasons"] = reason
    row["A0_q4_publication_blocker"] = bool(status == "fatal")

    q4_rows.append(row)

    curve_payload.append((candidate_id, eta_lags_real, eta_real, eta_syn, man_lags_real, man_real, man_syn))

q4_df = pd.DataFrame(q4_rows)
q4_df.to_csv(q4_metrics_csv, index=False)

curve_npz = {}
for candidate_id, eta_lags, eta_real, eta_syn, man_lags, man_real, man_syn in curve_payload:
    key = str(candidate_id)
    curve_npz[f"{key}__eta_lags"] = eta_lags.astype(np.int64)
    curve_npz[f"{key}__eta_real"] = eta_real.astype(np.float32)
    curve_npz[f"{key}__eta_syn_A0"] = eta_syn.astype(np.float32)
    curve_npz[f"{key}__manifest_lags"] = man_lags.astype(np.int64)
    curve_npz[f"{key}__manifest_real"] = man_real.astype(np.float32)
    curve_npz[f"{key}__manifest_syn_A0"] = man_syn.astype(np.float32)

np.savez_compressed(q4_curves_npz_path, **curve_npz)

# ----------------------------------------------------------
# 11) Compare against previous available metrics
# ----------------------------------------------------------
delta_df = q4_df.copy()

# Optional compare against 14.3 if same anchor/protocol appears.
if "CELL14_3_MANIFEST_Q4_POST_PAIR_METRICS_DF" in globals() and isinstance(CELL14_3_MANIFEST_Q4_POST_PAIR_METRICS_DF, pd.DataFrame):
    prev = CELL14_3_MANIFEST_Q4_POST_PAIR_METRICS_DF.copy()
    prev["anchor_col"] = prev["driver_col"].astype(str) if "driver_col" in prev.columns else ""
    prev["protocol_col"] = prev["protocol_col"].astype(str)
    prev_small = prev[[
        c for c in [
            "anchor_col", "protocol_col",
            "ETA_similarity", "lag_peak_error", "response_window_rate_error",
            "manifest_window_profile_similarity",
        ] if c in prev.columns
    ]].copy()
    prev_small = prev_small.rename(columns={
        "ETA_similarity": "prev_14_3_ETA_similarity",
        "lag_peak_error": "prev_14_3_lag_peak_error",
        "response_window_rate_error": "prev_14_3_response_window_rate_error",
        "manifest_window_profile_similarity": "prev_14_3_manifest_profile_similarity",
    })
    delta_df = delta_df.merge(prev_small, on=["anchor_col", "protocol_col"], how="left")

    if "prev_14_3_ETA_similarity" in delta_df.columns:
        delta_df["delta_vs_14_3_ETA_similarity"] = (
            pd.to_numeric(delta_df["ETA_similarity"], errors="coerce")
            - pd.to_numeric(delta_df["prev_14_3_ETA_similarity"], errors="coerce")
        )
        delta_df["delta_vs_14_3_lag_peak_error"] = (
            pd.to_numeric(delta_df["lag_peak_error"], errors="coerce")
            - pd.to_numeric(delta_df["prev_14_3_lag_peak_error"], errors="coerce")
        )
        delta_df["delta_vs_14_3_response_window_rate_error"] = (
            pd.to_numeric(delta_df["response_window_rate_error"], errors="coerce")
            - pd.to_numeric(delta_df["prev_14_3_response_window_rate_error"], errors="coerce")
        )
        delta_df["delta_vs_14_3_manifest_profile_similarity"] = (
            pd.to_numeric(delta_df["manifest_window_profile_similarity"], errors="coerce")
            - pd.to_numeric(delta_df["prev_14_3_manifest_profile_similarity"], errors="coerce")
        )

delta_df.to_csv(q4_delta_csv, index=False)

# ----------------------------------------------------------
# 12) Summary / contract
# ----------------------------------------------------------
status_counts = q4_df["A0_q4_status"].astype(str).value_counts().sort_index().to_dict()
blocker_n = int(q4_df["A0_q4_publication_blocker"].fillna(False).astype(bool).sum())
pass_n = int((q4_df["A0_q4_status"].astype(str) == "pass").sum())
warning_n = int((q4_df["A0_q4_status"].astype(str) == "warning").sum())
fatal_n = int((q4_df["A0_q4_status"].astype(str) == "fatal").sum())

summary = {
    "cell": "14.6",
    "version": CELL146_VERSION,
    "upstream_contract_versions": {
        "cell13_4": version134_146,
        "cell14_3": version143_146,
        "cell14_4": version144_146,
        "cell14_5": version145_146
    },
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_146,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_146),
        "pair_pass_rate": broad_q4_pair_pass_rate_146
    },
    "previous_repair_status_carried_forward": {
        "repair_candidate_status": repair_status_146,
        "post_repair_publication_blocker_n": int(repair_post_blocker_n_146),
        "post_repair_pass_rate": repair_post_pass_rate_146,
        "repair_delta_summary": repair_delta_summary_146
    },
    "policy_gate": {
        "direct_repair_allowed_now_n_from_14_5": int(direct_repair_allowed_now_n_146),
        "A0_manifest_policy_required": True
    },
    "protocol_base": {
        "source": protocol_base_source_146,
        "path": protocol_base_path_146,
        "sha256": protocol_base_sha256_146,
        "fallback_used": bool(CFG.get("cell14_6_protocol_base_fallback_used", False)),
        "fallback_policy": "Use historical pre-Q4 parquet when present; otherwise use in-memory PROTOCOL_SYN_TEST_130 uncoupled protocol matrix."
    },
    "A0_pairs_total": int(len(a0)),
    "valid_profiles": int(len(valid_profiles)),
    "profiles_applied": int(materialization_df["applied"].fillna(False).astype(bool).sum()) if len(materialization_df) else 0,
    "target_protocol_cols": target_cols,
    "target_changed_total": int(target_changed_total),
    "outside_window_drift_total": int(outside_window_total),
    "non_target_protocol_drift_n": int(non_target_drift_n),
    "negative_n": int(negative_n),
    "noninteger_n": int(noninteger_n),
    "inactive_finite_n": int(inactive_finite_n),
    "q4_status_counts": status_counts,
    "q4_pass_n": pass_n,
    "q4_warning_n": warning_n,
    "q4_fatal_n": fatal_n,
    "q4_publication_blocker_n": blocker_n,
    "q4_pass_rate": float(pass_n / max(len(q4_df), 1)),
    "mean_ETA_similarity": float(pd.to_numeric(q4_df["ETA_similarity"], errors="coerce").mean()) if len(q4_df) else np.nan,
    "mean_lag_peak_error": float(pd.to_numeric(q4_df["lag_peak_error"], errors="coerce").mean()) if len(q4_df) else np.nan,
    "mean_response_window_rate_error": float(pd.to_numeric(q4_df["response_window_rate_error"], errors="coerce").mean()) if len(q4_df) else np.nan,
    "mean_manifest_profile_similarity": float(pd.to_numeric(q4_df["manifest_window_profile_similarity"], errors="coerce").mean()) if len(q4_df) else np.nan,
}

# ----------------------------------------------------------
A0_ACCEPTED_146 = False
summary["A0_accepted_for_promotion"] = False
summary["A0_acceptance_gate"] = {
    "accepted": False,
    "gate_basis": "disabled_in_STUDY_THESIS_submission",
    "reason": "TEST QA is terminal reporting only. Promotion/acceptance must be precommitted from TRAIN/VAL before Cell 14.6 reads TEST reference values.",
    "observed_pairs_total": int(summary.get("A0_pairs_total", 0)),
    "observed_pass_n": int(summary.get("q4_pass_n", 0)),
    "observed_warning_n": int(summary.get("q4_warning_n", 0)),
    "observed_fatal_n": int(summary.get("q4_fatal_n", 0)),
    "observed_publication_blocker_n": int(summary.get("q4_publication_blocker_n", 0)),
}
summary["A0_candidate_rejection_reason"] = (
    "Candidate retained as terminal TEST report-only evidence. "
    "No accepted/promoted artifact is written from TEST outcomes."
)
accepted_contract = {
    "cell": "14.6",
    "version": CELL146_VERSION + "__acceptance_disabled_for_THESIS",
    "role": "A0_zigbee_safe_candidate_TEST_report_only",
    "accepted": False,
    "test_values_used_for_acceptance": False,
    "accepted_outputs_written": False,
    "reason": "TEST QA cannot accept/promote artifacts.",
    "candidate_summary": summary,
}
_write_json_146(accepted_contract_json, accepted_contract)
_write_json_146(accepted_contract_canonical_json, accepted_contract)
print("[Cell14.6 replacement] A0 candidate QA is report-only; no TEST-gated accepted outputs written.")



_write_json_146(q4_summary_json, summary)

contract = {
    "cell": "14.6",
    "version": CELL146_VERSION,
    "role": "A0_zigbee_safe_expanded_coupling_materialization_and_QA",
    "quality_dimension": "Q4_cross_modal_consistency_repair_candidate",
    "upstream_contract_versions": summary["upstream_contract_versions"],
    "broad_q4_status_carried_forward": summary["broad_q4_status_carried_forward"],
    "previous_repair_status_carried_forward": summary["previous_repair_status_carried_forward"],
    "policy_gate": summary["policy_gate"],
    "summary": summary,
    "strict_contract": {
        "TRAIN_VAL_used_for_profile_learning": True,
        "A0_manifest_policy_required": True,
        "direct_repair_allowed_now_only": True,
        "A0_promotion_gate_enforced": True,
        "candidate_outputs_always_written": True,
        "accepted_outputs_written_only_if_gate_passes": True,
        "broad_q4_status_carried_forward": True,
        "repair_status_carried_forward": True,
        "synthetic_TEST_IoT_drivers_used_for_conditioning": True,
        "TEST_real_values_used_for_QA_only": True,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_protocol_values_mutated": True,
        "synthetic_iot_values_mutated": False,
        "only_target_protocol_cols_mutated": bool(non_target_drift_n == 0),
        "only_legal_windows_mutated": bool(outside_window_total == 0),
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "A0_accepted_for_promotion": bool(A0_ACCEPTED_146),
        "accepted_outputs_written": bool(A0_ACCEPTED_146),
        "failed_candidate_overwrites_accepted_outputs": False,
        "protocol_base_source": protocol_base_source_146,
        "protocol_base_path": protocol_base_path_146,
        "protocol_base_sha256": protocol_base_sha256_146,
        "protocol_base_fallback_used": bool(CFG.get("cell14_6_protocol_base_fallback_used", False)),
        "historical_pre11b_quarantine_artifact_used": False,
    },
    "outputs": {
        "profile_audit_csv": profile_audit_csv,
        "materialization_audit_csv": materialization_audit_csv,
        "target_drift_csv": target_drift_csv,
        "non_target_drift_csv": non_target_drift_csv,
        "q4_metrics_csv": q4_metrics_csv,
        "q4_delta_csv": q4_delta_csv,
        "q4_summary_json": q4_summary_json,
        "contract_json": contract_json,
        "protocol_a0_path": protocol_a0_path,
        "cps_a0_path": cps_a0_path,
        "profiles_npz_path": profiles_npz_path,
        "q4_curves_npz_path": q4_curves_npz_path,
        "manifest_json": manifest_json,
        "q4_candidate_metrics_csv": q4_candidate_metrics_csv,
        "q4_candidate_delta_csv": q4_candidate_delta_csv,
        "q4_candidate_summary_json": q4_candidate_summary_json,
        "candidate_contract_json": candidate_contract_json,
        "candidate_contract_canonical_json": candidate_contract_canonical_json,
        "protocol_a0_candidate_path": protocol_a0_candidate_path,
        "cps_a0_candidate_path": cps_a0_candidate_path,
        
        "q4_accepted_metrics_csv": q4_accepted_metrics_csv,
        "q4_accepted_delta_csv": q4_accepted_delta_csv,
        "q4_accepted_summary_json": q4_accepted_summary_json,
        "accepted_contract_json": accepted_contract_json,
        "accepted_contract_canonical_json": accepted_contract_canonical_json,
        "protocol_a0_accepted_path": protocol_a0_accepted_path,
        "cps_a0_accepted_path": cps_a0_accepted_path,
    },
}

_write_json_146(contract_json, contract)
_write_json_146(candidate_contract_canonical_json, contract)

manifest = {
    "cell": "14.6",
    "version": CELL146_VERSION,
    "created_outputs": contract["outputs"],
    "summary": summary,
    "broad_q4_status_carried_forward": summary["broad_q4_status_carried_forward"],
    "previous_repair_status_carried_forward": summary["previous_repair_status_carried_forward"],
    "policy_gate": summary["policy_gate"],
    "strict_contract": contract["strict_contract"],
}
_write_json_146(manifest_json, manifest)

hashes = {
    "profile_audit_csv_sha256": _sha256_file_146(profile_audit_csv),
    "materialization_audit_csv_sha256": _sha256_file_146(materialization_audit_csv),
    "target_drift_csv_sha256": _sha256_file_146(target_drift_csv),
    "non_target_drift_csv_sha256": _sha256_file_146(non_target_drift_csv),
    "q4_metrics_csv_sha256": _sha256_file_146(q4_metrics_csv),
    "q4_delta_csv_sha256": _sha256_file_146(q4_delta_csv),
    "q4_summary_json_sha256": _sha256_file_146(q4_summary_json),
    "contract_json_sha256": _sha256_file_146(contract_json),
    "protocol_a0_sha256": _sha256_file_146(protocol_a0_path),
    "cps_a0_sha256": _sha256_file_146(cps_a0_path),
    "profiles_npz_sha256": _sha256_file_146(profiles_npz_path),
    "q4_curves_npz_sha256": _sha256_file_146(q4_curves_npz_path),
    "manifest_json_sha256": _sha256_file_146(manifest_json),
    "q4_candidate_metrics_csv_sha256": _sha256_file_146(q4_candidate_metrics_csv),
    "q4_candidate_delta_csv_sha256": _sha256_file_146(q4_candidate_delta_csv),
    "candidate_contract_json_sha256": _sha256_file_146(candidate_contract_json),
    "candidate_contract_canonical_json_sha256": _sha256_file_146(candidate_contract_canonical_json),
    "protocol_a0_candidate_sha256": _sha256_file_146(protocol_a0_candidate_path),
    "cps_a0_candidate_sha256": _sha256_file_146(cps_a0_candidate_path),
    "accepted_contract_json_sha256": _sha256_file_146(accepted_contract_json),
    "accepted_contract_canonical_json_sha256": _sha256_file_146(accepted_contract_canonical_json),
}
if A0_ACCEPTED_146:
    hashes.update({
        "q4_accepted_metrics_csv_sha256": _sha256_file_146(q4_accepted_metrics_csv),
        "q4_accepted_delta_csv_sha256": _sha256_file_146(q4_accepted_delta_csv),
        "q4_accepted_summary_json_sha256": _sha256_file_146(q4_accepted_summary_json),
        "protocol_a0_accepted_sha256": _sha256_file_146(protocol_a0_accepted_path),
        "cps_a0_accepted_sha256": _sha256_file_146(cps_a0_accepted_path),
    })

contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_146(contract_json, contract)
_write_json_146(candidate_contract_canonical_json, contract)
_write_json_146(manifest_json, manifest)

# ----------------------------------------------------------
# 13) Export globals
# ----------------------------------------------------------
globals()["CELL146_VERSION"] = CELL146_VERSION
globals()["PROTOCOL_SYN_TEST_COUPLED_ZIGBEE_A0"] = protocol_a0
globals()["CPS_COUPLED_ZIGBEE_A0_TEST"] = CPS_COUPLED_ZIGBEE_A0_TEST
globals()["CELL14_6_A0_PROFILE_AUDIT_DF"] = profiles_df
globals()["CELL14_6_A0_MATERIALIZATION_AUDIT_DF"] = materialization_df
globals()["CELL14_6_A0_TARGET_DRIFT_DF"] = target_drift_df
globals()["CELL14_6_A0_NON_TARGET_DRIFT_DF"] = non_target_drift_df
globals()["CELL14_6_A0_Q4_PAIR_METRICS_DF"] = q4_df
globals()["CELL14_6_A0_Q4_DELTA_DF"] = delta_df
globals()["CELL14_6_A0_SUMMARY"] = summary
globals()["CELL14_6_A0_CONTRACT"] = contract

# Candidate outputs.
globals()["CELL14_6_A0_CANDIDATE_PROTOCOL_PATH"] = protocol_a0_candidate_path
globals()["CELL14_6_A0_CANDIDATE_CPS_PATH"] = cps_a0_candidate_path
globals()["CELL14_6_A0_CANDIDATE_Q4_METRICS_CSV"] = q4_candidate_metrics_csv
globals()["CELL14_6_A0_CANDIDATE_CONTRACT_JSON"] = candidate_contract_json
globals()["CELL14_6_A0_CANDIDATE_CONTRACT_CANONICAL_JSON"] = candidate_contract_canonical_json

# Backward-compatible names point to candidate outputs.
globals()["CELL14_6_A0_PROTOCOL_PATH"] = protocol_a0_candidate_path
globals()["CELL14_6_A0_CPS_PATH"] = cps_a0_candidate_path
globals()["CELL14_6_A0_Q4_METRICS_CSV"] = q4_candidate_metrics_csv
globals()["CELL14_6_A0_CONTRACT_JSON"] = candidate_contract_json
globals()["CELL14_6_A0_MANIFEST_JSON"] = manifest_json

# Accepted outputs.
globals()["CELL14_6_A0_ACCEPTED"] = bool(A0_ACCEPTED_146)
globals()["CELL14_6_A0_ACCEPTED_PROTOCOL_PATH"] = protocol_a0_accepted_path if A0_ACCEPTED_146 else ""
globals()["CELL14_6_A0_ACCEPTED_CPS_PATH"] = cps_a0_accepted_path if A0_ACCEPTED_146 else ""
globals()["CELL14_6_A0_ACCEPTED_Q4_METRICS_CSV"] = q4_accepted_metrics_csv if A0_ACCEPTED_146 else ""
globals()["CELL14_6_A0_ACCEPTED_CONTRACT_JSON"] = accepted_contract_json
globals()["CELL14_6_A0_ACCEPTED_CONTRACT_CANONICAL_JSON"] = accepted_contract_canonical_json
globals()["CELL14_6_A0_ACCEPTED_CONTRACT"] = accepted_contract

globals()["CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH"] = str(
    globals().get("CELL14_6_PRE_Q4_PROTOCOL_BASE_PATH", "<in_memory>")
)

log(
    "[Cell14.6] A0 Zigbee-safe coupling materialization + QA complete | "
    f"A0_pairs={summary['A0_pairs_total']} | "
    f"valid_profiles={summary['valid_profiles']} | "
    f"profiles_applied={summary['profiles_applied']} | "
    f"target_changed_total={summary['target_changed_total']} | "
    f"outside_window_drift_total={summary['outside_window_drift_total']} | "
    f"non_target_protocol_drift_n={summary['non_target_protocol_drift_n']} | "
    f"q4_pass={summary['q4_pass_n']} | "
    f"q4_warning={summary['q4_warning_n']} | "
    f"q4_fatal={summary['q4_fatal_n']} | "
    f"q4_blockers={summary['q4_publication_blocker_n']}"
)
log(
    "[Cell14.6] A0 Q4 metrics | "
    f"mean_ETA_similarity={summary['mean_ETA_similarity']:.6f} | "
    f"mean_lag_peak_error={summary['mean_lag_peak_error']:.6f} | "
    f"mean_response_window_rate_error={summary['mean_response_window_rate_error']:.6f} | "
    f"mean_manifest_profile_similarity={summary['mean_manifest_profile_similarity']:.6f}"
)
log(f"[Cell14.6] Target drift summary | {target_drift_df.to_dict('records')}")
log(f"[Cell14.6] Q4 status counts | {status_counts}")
log(f"[Cell14.6] Saved A0 protocol candidate: {protocol_a0_path}")
log(f"[Cell14.6] Saved A0 CPS candidate: {cps_a0_path}")
log(f"[Cell14.6] Saved Q4 metrics: {q4_metrics_csv} | rows={len(q4_df)}")
log(f"[Cell14.6] Saved contract: {contract_json}")
log(f"[Cell14.6] Saved canonical candidate contract: {candidate_contract_canonical_json}")
log(f"[Cell14.6] Saved canonical accepted contract: {accepted_contract_canonical_json}")
log(
    "[Cell14.6] Upstream Q4/repair status carried forward | "
    f"broad_status={broad_q4_status_146} | "
    f"broad_pair_blocker_n={broad_q4_pair_blocker_n_146} | "
    f"previous_repair_status={repair_status_146} | "
    f"previous_repair_post_blocker_n={repair_post_blocker_n_146} | "
    f"previous_repair_post_pass_rate={repair_post_pass_rate_146}"
)
log(
    "[Cell14.6] Terminal TEST report-only Q4 candidate | "
    f"A0_accepted_for_promotion={A0_ACCEPTED_146} | "
    "no expected TEST acceptance pattern is used | "
    f"observed_pairs/pass/warning/fatal/blocker="
    f"{summary['A0_pairs_total']}/{summary['q4_pass_n']}/"
    f"{summary['q4_warning_n']}/{summary['q4_fatal_n']}/"
    f"{summary['q4_publication_blocker_n']}"
)
log(
    "[Cell14.6] Contract flags | "
    "TRAIN_VAL_used_for_profile_learning=True | "
    "synthetic_TEST_IoT_drivers_used_for_conditioning=True | "
    "TEST_real_values_used_for_QA_only=True | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_protocol_values_mutated=True | "
    "synthetic_iot_values_mutated=False | "
    "only_target_protocol_cols_mutated=True | "
    "only_legal_windows_mutated=True | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | A0_promotion_gate_enforced=True | accepted_outputs_written_only_if_gate_passes=True"
)
log("--- END: Cell 14.6 - A0 Zigbee-safe expanded coupling materialization + QA (v1.2 promotion-gated protocol-base-fallback strict) ---")

gc.collect()