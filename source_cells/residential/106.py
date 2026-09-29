# ==========================================================
# CELL 14.0 - TRAIN/VAL-only Zigbee coupling profile builder
# v1.1 STUDY-THESIS strict coupling-conditioned protocol profiles, Q4-failure-aware
#
# Role:
#   - Learn replicated Zigbee IoT->protocol coupling profiles from TRAIN+VAL only.
#   - Use manifest-informed Tier-A/Tier-B pairs from Cell 13.5.
#   - Build donor response profiles for later TEST materialization in Cell 14.1.
#
# Scientific contract:
#   - TRAIN+VAL real data may be used to learn coupling profiles.
#   - Real TEST protocol/IoT values must NOT be used for fitting/profile construction.
#   - Final synthetic TEST driver events will be used later only as conditioning input.
#   - No synthetic values are mutated here.
#
# Inputs:
#   CFG, log, df_tr, df_val, df_te
#   OUTDIR, OUT_SYN, REPORT_DIR, SEED
#   CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF
#
# Outputs:
#   reports/cell14_0_zigbee_coupling_profiles.csv
#   reports/cell14_0_zigbee_coupling_profile_audit.csv
#   reports/cell14_0_zigbee_coupling_profile_contract.json
#   artifacts/cell14_0_zigbee_coupling_profiles.npz
#   artifacts/cell14_0_zigbee_coupling_profile_manifest.json
# ==========================================================

log("--- START: Cell 14.0 - TRAIN/VAL-only Zigbee coupling profile builder (v1.1 Q4-failure-aware strict) ---")

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
_required_140 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF",
    "CELL13_5_MANIFEST_Q4_CONTRACT",
    "CELL13_6_MANIFEST_Q4_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
]
_missing_140 = [k for k in _required_140 if k not in globals()]
if _missing_140:
    raise RuntimeError(f"[Cell14.0] Missing required globals: {_missing_140}")

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
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

if N_TR <= 0 or N_VAL <= 0 or N_TE <= 0:
    raise RuntimeError(
        f"[Cell14.0] Invalid split lengths: train={N_TR}, val={N_VAL}, test={N_TE}"
    )

CELL140_VERSION = "cell14_0_trainval_only_zigbee_coupling_profile_builder_v1_1_q4_failure_aware"

CFG["cell14_0_version"] = CELL140_VERSION
CFG["cell14_0_train_values_used_for_profile_learning"] = True
CFG["cell14_0_val_values_used_for_profile_learning"] = True
CFG["cell14_0_TEST_real_values_used"] = False
CFG["cell14_0_synthetic_values_mutated"] = False
CFG["cell14_0_selection_done_here"] = False
CFG["cell14_0_generator_fit_done_here"] = False
CFG["cell14_0_materialization_done_here"] = False
CFG["cell14_0_broad_q4_status_carried_forward"] = True
CFG["cell14_0_manifest_q4_status_carried_forward"] = True
CFG["cell14_0_profile_learning_is_repair_prerequisite"] = True

CFG.setdefault("cell14_0_allowed_protocol_tiers", ["zigbee"])
CFG.setdefault("cell14_0_allowed_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])
CFG.setdefault("cell14_0_profile_pre_margin", 10)
CFG.setdefault("cell14_0_profile_post_margin", 10)
CFG.setdefault("cell14_0_min_train_events", 20)
CFG.setdefault("cell14_0_min_val_events", 5)
CFG.setdefault("cell14_0_min_combined_events", 30)
CFG.setdefault("cell14_0_min_modality_coverage", 0.20)
CFG.setdefault("cell14_0_response_threshold_quantile", 0.90)
CFG.setdefault("cell14_0_protocol_transform", "identity")
CFG.setdefault("cell14_0_clip_low_quantile", 0.001)
CFG.setdefault("cell14_0_clip_high_quantile", 0.999)
CFG.setdefault("cell14_0_max_profile_abs_multiplier", 5.0)
CFG.setdefault("cell14_0_fail_if_no_valid_profiles", True)
CFG.setdefault("cell14_0_event_deduplicate_adjacent", True)

PROFILE_PRE_MARGIN_140 = int(CFG.get("cell14_0_profile_pre_margin", 10))
PROFILE_POST_MARGIN_140 = int(CFG.get("cell14_0_profile_post_margin", 10))

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
profile_csv = os.path.join(REPORT_DIR, "cell14_0_zigbee_coupling_profiles.csv")
profile_audit_csv = os.path.join(REPORT_DIR, "cell14_0_zigbee_coupling_profile_audit.csv")
profile_contract_json = os.path.join(REPORT_DIR, "cell14_0_zigbee_coupling_profile_contract.json")
profile_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_0_zigbee_coupling_profile_contract_v1_1_THESIS.json")
profile_npz = os.path.join(ARTDIR, "cell14_0_zigbee_coupling_profiles.npz")
profile_manifest_json = os.path.join(ARTDIR, "cell14_0_zigbee_coupling_profile_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_140(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_140(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_140(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_140(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_140(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_140(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_140(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_140(obj.to_dict())
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

def _write_json_140(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_140(payload), f, indent=2, sort_keys=True)

def _sha256_file_140(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_140(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_140(frame: pd.DataFrame, col: str, fill=None) -> np.ndarray:
    arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if fill is not None:
        arr = np.where(np.isfinite(arr), arr, float(fill))
    return arr

def _protocol_tier_140(col: str) -> str:
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

def _resolve_obs_col_140(frame: pd.DataFrame, tier: str):
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

def _transform_protocol_140(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell14_0_protocol_transform", "identity")).strip().lower()

    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.log1p(np.maximum(arr, 0.0))
    elif mode == "log1p_abs":
        out = np.log1p(np.abs(arr))
    elif mode == "log1p_abs_signed":
        out = np.sign(arr) * np.log1p(np.abs(arr))
    else:
        raise RuntimeError(f"[Cell14.0] Unknown protocol transform: {mode}")

    out[~np.isfinite(out)] = np.nan
    return out

def _inverse_transform_protocol_140(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell14_0_protocol_transform", "identity")).strip().lower()

    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.expm1(arr)
    elif mode == "log1p_abs":
        out = np.expm1(arr)
    elif mode == "log1p_abs_signed":
        out = np.sign(arr) * np.expm1(np.abs(arr))
    else:
        raise RuntimeError(f"[Cell14.0] Unknown protocol transform: {mode}")

    out[~np.isfinite(out)] = np.nan
    return out

def _event_starts_140(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    active = np.isfinite(arr) & (arr > 0.0)
    if active.size == 0:
        return np.asarray([], dtype=np.int64)

    if not bool(CFG.get("cell14_0_event_deduplicate_adjacent", True)):
        return np.flatnonzero(active).astype(np.int64)

    prev = np.r_[False, active[:-1]]
    starts = active & (~prev)
    return np.flatnonzero(starts).astype(np.int64)

def _edge_safe_events_140(idx: np.ndarray, n: int, lag_lo: int, lag_hi: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx
    keep = (idx + lag_lo >= 0) & (idx + lag_hi < n)
    return idx[keep]

def _window_matrix_140(y: np.ndarray, obs: np.ndarray, events: np.ndarray, lag_lo: int, lag_hi: int):
    y = np.asarray(y, dtype=np.float64)
    obs = np.asarray(obs, dtype=bool)
    events = _edge_safe_events_140(events, len(y), lag_lo, lag_hi)

    if events.size == 0:
        return None, events

    rows = []
    for e in events:
        seg = y[e + lag_lo : e + lag_hi + 1]
        seg_obs = obs[e + lag_lo : e + lag_hi + 1]
        if len(seg) != (lag_hi - lag_lo + 1):
            continue
        seg = np.where(seg_obs & np.isfinite(seg), seg, np.nan)
        rows.append(seg)

    if not rows:
        return None, events

    return np.vstack(rows).astype(np.float64), events

def _robust_profile_stats_140(mat: np.ndarray):
    if mat is None or not isinstance(mat, np.ndarray) or mat.size == 0:
        return None

    with np.errstate(invalid="ignore"):
        mean_profile = np.nanmean(mat, axis=0)
        median_profile = np.nanmedian(mat, axis=0)
        q10_profile = np.nanquantile(mat, 0.10, axis=0)
        q90_profile = np.nanquantile(mat, 0.90, axis=0)
        finite_per_lag = np.isfinite(mat).sum(axis=0)

    return {
        "mean": mean_profile.astype(np.float64),
        "median": median_profile.astype(np.float64),
        "q10": q10_profile.astype(np.float64),
        "q90": q90_profile.astype(np.float64),
        "finite_per_lag": finite_per_lag.astype(np.int64),
    }

def _baseline_correct_140(profile: np.ndarray, lags: np.ndarray):
    p = np.asarray(profile, dtype=np.float64).copy()
    lags = np.asarray(lags, dtype=np.int64)

    pre_mask = lags < 0
    if np.any(pre_mask) and np.isfinite(p[pre_mask]).any():
        baseline = float(np.nanmean(p[pre_mask]))
    elif np.isfinite(p).any():
        baseline = float(np.nanmedian(p[np.isfinite(p)]))
    else:
        baseline = 0.0

    return p - baseline, baseline

def _window_mean_140(y: np.ndarray, obs: np.ndarray, events: np.ndarray, lag_lo: int, lag_hi: int) -> float:
    mat, ev = _window_matrix_140(y, obs, events, lag_lo, lag_hi)
    if mat is None:
        return np.nan
    return float(np.nanmean(mat)) if np.isfinite(mat).any() else np.nan

def _coverage_near_events_140(obs: np.ndarray, events: np.ndarray, lag_lo: int, lag_hi: int) -> float:
    obs = np.asarray(obs, dtype=bool)
    events = _edge_safe_events_140(events, len(obs), lag_lo, lag_hi)
    if events.size == 0:
        return np.nan
    vals = []
    for e in events:
        seg = obs[e + lag_lo : e + lag_hi + 1]
        vals.append(float(np.mean(seg)) if len(seg) else np.nan)
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan

def _quantile_clip_bounds_140(x: np.ndarray):
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan

    lo_q = float(CFG.get("cell14_0_clip_low_quantile", 0.001))
    hi_q = float(CFG.get("cell14_0_clip_high_quantile", 0.999))
    lo_q = float(np.clip(lo_q, 0.0, 0.25))
    hi_q = float(np.clip(hi_q, 0.75, 1.0))

    return float(np.nanquantile(arr, lo_q)), float(np.nanquantile(arr, hi_q))

def _profile_confidence_140(train_events, val_events, train_cov, val_cov, zimm_train, zimm_val, zsharp_train, zsharp_val):
    # Conservative confidence: support + coverage + replicated z evidence.
    ev_core = min(float(train_events), float(val_events))
    support_score = min(1.0, ev_core / max(float(CFG.get("cell14_0_min_combined_events", 30)), 1.0))

    cov_vals = [v for v in [train_cov, val_cov] if np.isfinite(v)]
    cov_score = min(cov_vals) if cov_vals else 0.0
    cov_score = float(np.clip(cov_score, 0.0, 1.0))

    z_vals = [
        _safe_float_140(zimm_train, 0.0),
        _safe_float_140(zimm_val, 0.0),
        _safe_float_140(zsharp_train, 0.0),
        _safe_float_140(zsharp_val, 0.0),
    ]
    z_core = min(z_vals)
    z_score = float(np.clip(z_core / 5.0, 0.0, 1.0))

    return float(0.40 * support_score + 0.30 * cov_score + 0.30 * z_score)

# ----------------------------------------------------------
# 3) Validate upstream Q4 failure / manifest contracts
# ----------------------------------------------------------
def _require_contract_version_140(obj, name: str, expected_substring: str):
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.0] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.0] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

version135_140 = _require_contract_version_140(
    CELL13_5_MANIFEST_Q4_CONTRACT,
    "CELL13_5_MANIFEST_Q4_CONTRACT",
    "cell13_5_manifest_informed_zigbee_q4_registry_v1_1",
)
version136_140 = _require_contract_version_140(
    CELL13_6_MANIFEST_Q4_CONTRACT,
    "CELL13_6_MANIFEST_Q4_CONTRACT",
    "cell13_6_manifest_informed_zigbee_q4_qa_v1_1",
)
version134_140 = _require_contract_version_140(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)

strict135_140 = CELL13_5_MANIFEST_Q4_CONTRACT.get("strict_contract", {})
strict136_140 = CELL13_6_MANIFEST_Q4_CONTRACT.get("strict_contract", {})

for _name, _strict in [("13.5", strict135_140), ("13.6", strict136_140)]:
    if bool(_strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell14.0] Upstream Cell {_name} indicates synthetic values were mutated.")
    if bool(_strict.get("selection_done_here", True)):
        raise RuntimeError(f"[Cell14.0] Upstream Cell {_name} indicates selection was done.")
    if bool(_strict.get("generator_fit_done_here", True)):
        raise RuntimeError(f"[Cell14.0] Upstream Cell {_name} indicates generator fitting was done.")
    if bool(_strict.get("materialization_done_here", True)):
        raise RuntimeError(f"[Cell14.0] Upstream Cell {_name} indicates materialization was done.")

broad_q4_status_140 = str(CELL13_4_Q4_PUBLICATION_SUMMARY.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_140 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("pair_blocker_n", 0) or 0
)
broad_q4_pair_pass_rate_140 = _safe_float_140(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)
broad_q4_publication_blocker_record_n_140 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get(
        "publication_blocker_record_n",
        CELL13_4_Q4_PUBLICATION_SUMMARY.get("publication_blocker_n", 0),
    ) or 0
)

manifest_q4_summary_140 = CELL13_6_MANIFEST_Q4_CONTRACT.get("overall_summary", {})
manifest_q4_publication_blocker_n_140 = int(manifest_q4_summary_140.get("publication_blocker_n", 0) or 0)
manifest_q4_pair_pass_rate_140 = _safe_float_140(
    manifest_q4_summary_140.get("manifest_pair_pass_rate", np.nan),
    np.nan,
)
manifest_q4_pairs_total_140 = int(manifest_q4_summary_140.get("pairs_total", 0) or 0)

if broad_q4_status_140 != "blocker":
    log(
        "[Cell14.0] WARNING: broad Q4 status is not blocker. "
        f"overall_status={broad_q4_status_140}"
    )

if manifest_q4_publication_blocker_n_140 <= 0:
    log(
        "[Cell14.0] WARNING: manifest Q4 does not report blockers. "
        f"publication_blocker_n={manifest_q4_publication_blocker_n_140}"
    )

# ----------------------------------------------------------
# 4) Validate manifest pairs and schemas
# ----------------------------------------------------------
pairs = CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF.copy()
if len(pairs) == 0:
    raise RuntimeError("[Cell14.0] Empty Cell 13.5 manifest pair registry.")

required_pair_cols = [
    "pair_id",
    "manifest_tier",
    "anchor_name",
    "driver_col",
    "protocol_col",
    "protocol_tier",
    "lag_lo",
    "lag_hi",
]
missing_pair_cols = [c for c in required_pair_cols if c not in pairs.columns]
if missing_pair_cols:
    raise RuntimeError(f"[Cell14.0] Manifest pair registry missing required columns: {missing_pair_cols}")

pairs["driver_col"] = pairs["driver_col"].astype(str)
pairs["protocol_col"] = pairs["protocol_col"].astype(str)
pairs["protocol_tier"] = pairs["protocol_tier"].astype(str).str.lower()
pairs["manifest_tier"] = pairs["manifest_tier"].astype(str).str.upper()

allowed_tiers = set(str(x).lower() for x in CFG.get("cell14_0_allowed_protocol_tiers", ["zigbee"]))
allowed_protocol_cols = set(str(x) for x in CFG.get("cell14_0_allowed_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"]))

# This builder is intentionally narrow.
pairs = pairs[
    pairs["protocol_tier"].isin(allowed_tiers)
    & pairs["protocol_col"].isin(allowed_protocol_cols)
].copy()

if len(pairs) == 0:
    raise RuntimeError("[Cell14.0] No allowed Zigbee manifest pairs remain after strict filter.")

# Required train/val columns only. No TEST values used for profiles.
missing = []
for c in sorted(set(pairs["driver_col"]) | set(pairs["protocol_col"])):
    if c not in df_tr.columns:
        missing.append(f"train_missing:{c}")
    if c not in df_val.columns:
        missing.append(f"val_missing:{c}")

if missing:
    raise RuntimeError(f"[Cell14.0] Required TRAIN/VAL columns missing. Preview={missing[:30]}")

# df_te schema may be checked for future compatibility only, not values.
schema_missing_te = []
for c in sorted(set(pairs["driver_col"]) | set(pairs["protocol_col"])):
    if c not in df_te.columns:
        schema_missing_te.append(c)

if schema_missing_te:
    raise RuntimeError(
        "[Cell14.0] Required columns absent from TEST schema for later materialization. "
        f"Preview={schema_missing_te[:30]}"
    )

# ----------------------------------------------------------
# 5) Build TRAIN/VAL profiles
# ----------------------------------------------------------
profile_rows = []
audit_rows = []
npz_payload = {}

log(
    "[Cell14.0] Building TRAIN/VAL-only Zigbee coupling profiles | "
    f"manifest_pairs={len(pairs)}"
)

for i, r in enumerate(pairs.itertuples(index=False), start=1):
    row = r._asdict()

    pair_id = str(row["pair_id"])
    manifest_tier = str(row["manifest_tier"])
    anchor_name = str(row.get("anchor_name", ""))
    driver_col = str(row["driver_col"])
    protocol_col = str(row["protocol_col"])
    protocol_tier = str(row["protocol_tier"])

    lag_lo = int(row["lag_lo"])
    lag_hi = int(row["lag_hi"])
    if lag_lo > lag_hi:
        lag_lo, lag_hi = lag_hi, lag_lo

    # Expanded profile window around manifest lag window.
    profile_lag_lo = int(lag_lo - PROFILE_PRE_MARGIN_140)
    profile_lag_hi = int(lag_hi + PROFILE_POST_MARGIN_140)
    lags = np.arange(profile_lag_lo, profile_lag_hi + 1, dtype=np.int64)

    obs_col_tr = _resolve_obs_col_140(df_tr, protocol_tier)
    obs_col_val = _resolve_obs_col_140(df_val, protocol_tier)

    if obs_col_tr is None or obs_col_val is None:
        audit_rows.append({
            "pair_id": pair_id,
            "driver_col": driver_col,
            "protocol_col": protocol_col,
            "profile_valid": False,
            "rejection_reason": "missing_protocol_observability_column",
        })
        continue

    d_tr = _to_num_array_140(df_tr, driver_col, fill=0.0)
    d_val = _to_num_array_140(df_val, driver_col, fill=0.0)

    y_tr_raw = _to_num_array_140(df_tr, protocol_col)
    y_val_raw = _to_num_array_140(df_val, protocol_col)

    y_tr = _transform_protocol_140(y_tr_raw)
    y_val = _transform_protocol_140(y_val_raw)

    obs_tr = _to_num_array_140(df_tr, obs_col_tr, fill=0.0) > 0.5
    obs_val = _to_num_array_140(df_val, obs_col_val, fill=0.0) > 0.5

    ev_tr_all = _event_starts_140(d_tr)
    ev_val_all = _event_starts_140(d_val)

    ev_tr = _edge_safe_events_140(ev_tr_all, len(df_tr), profile_lag_lo, profile_lag_hi)
    ev_val = _edge_safe_events_140(ev_val_all, len(df_val), profile_lag_lo, profile_lag_hi)

    train_events = int(len(ev_tr))
    val_events = int(len(ev_val))
    combined_events = int(train_events + val_events)

    train_cov = _coverage_near_events_140(obs_tr, ev_tr, profile_lag_lo, profile_lag_hi)
    val_cov = _coverage_near_events_140(obs_val, ev_val, profile_lag_lo, profile_lag_hi)

    train_event_mean = _window_mean_140(y_tr, obs_tr, ev_tr, lag_lo, lag_hi)
    val_event_mean = _window_mean_140(y_val, obs_val, ev_val, lag_lo, lag_hi)

    train_pre_mean = _window_mean_140(y_tr, obs_tr, ev_tr, profile_lag_lo, min(-1, lag_lo - 1))
    val_pre_mean = _window_mean_140(y_val, obs_val, ev_val, profile_lag_lo, min(-1, lag_lo - 1))

    train_uplift = (
        train_event_mean - train_pre_mean
        if np.isfinite(train_event_mean) and np.isfinite(train_pre_mean)
        else np.nan
    )
    val_uplift = (
        val_event_mean - val_pre_mean
        if np.isfinite(val_event_mean) and np.isfinite(val_pre_mean)
        else np.nan
    )

    # Profile matrices.
    mat_tr, ev_tr_used = _window_matrix_140(y_tr, obs_tr, ev_tr, profile_lag_lo, profile_lag_hi)
    mat_val, ev_val_used = _window_matrix_140(y_val, obs_val, ev_val, profile_lag_lo, profile_lag_hi)

    valid = True
    rejection_reasons = []

    if train_events < int(CFG.get("cell14_0_min_train_events", 20)):
        valid = False
        rejection_reasons.append("insufficient_train_events")

    if val_events < int(CFG.get("cell14_0_min_val_events", 5)):
        valid = False
        rejection_reasons.append("insufficient_val_events")

    if combined_events < int(CFG.get("cell14_0_min_combined_events", 30)):
        valid = False
        rejection_reasons.append("insufficient_combined_events")

    min_cov = float(CFG.get("cell14_0_min_modality_coverage", 0.20))
    if np.isfinite(train_cov) and train_cov < min_cov:
        valid = False
        rejection_reasons.append("low_train_modality_coverage")
    if np.isfinite(val_cov) and val_cov < min_cov:
        valid = False
        rejection_reasons.append("low_val_modality_coverage")

    if mat_tr is None or mat_val is None:
        valid = False
        rejection_reasons.append("missing_train_or_val_window_matrix")

    if valid:
        combined_mat = np.vstack([mat_tr, mat_val])
        stats = _robust_profile_stats_140(combined_mat)

        if stats is None:
            valid = False
            rejection_reasons.append("profile_stats_failed")
        else:
            mean_profile_raw = stats["mean"]
            median_profile_raw = stats["median"]
            q10_profile_raw = stats["q10"]
            q90_profile_raw = stats["q90"]

            mean_profile_bc, baseline_mean = _baseline_correct_140(mean_profile_raw, lags)
            median_profile_bc, baseline_median = _baseline_correct_140(median_profile_raw, lags)
            q10_profile_bc, _ = _baseline_correct_140(q10_profile_raw, lags)
            q90_profile_bc, _ = _baseline_correct_140(q90_profile_raw, lags)

            finite_per_lag = stats["finite_per_lag"]

            # Bound profile amplitude based on TRAIN+VAL observed support.
            y_combined_raw = np.concatenate([y_tr_raw[np.isfinite(y_tr_raw)], y_val_raw[np.isfinite(y_val_raw)]])
            support_lo_raw, support_hi_raw = _quantile_clip_bounds_140(y_combined_raw)

            y_combined_trans = np.concatenate([y_tr[np.isfinite(y_tr)], y_val[np.isfinite(y_val)]])
            trans_lo, trans_hi = _quantile_clip_bounds_140(y_combined_trans)

            profile_abs_cap = float(CFG.get("cell14_0_max_profile_abs_multiplier", 5.0)) * max(
                1e-9,
                float(np.nanstd(y_combined_trans)) if y_combined_trans.size else 1.0,
            )

            mean_profile_bc = np.clip(mean_profile_bc, -profile_abs_cap, profile_abs_cap)
            median_profile_bc = np.clip(median_profile_bc, -profile_abs_cap, profile_abs_cap)
            q10_profile_bc = np.clip(q10_profile_bc, -profile_abs_cap, profile_abs_cap)
            q90_profile_bc = np.clip(q90_profile_bc, -profile_abs_cap, profile_abs_cap)

            zimm_train = _safe_float_140(row.get("z_immediate_train"), np.nan)
            zimm_val = _safe_float_140(row.get("z_immediate_val"), np.nan)
            zsharp_train = _safe_float_140(row.get("z_sharpness_train"), np.nan)
            zsharp_val = _safe_float_140(row.get("z_sharpness_val"), np.nan)

            confidence = _profile_confidence_140(
                train_events=train_events,
                val_events=val_events,
                train_cov=train_cov,
                val_cov=val_cov,
                zimm_train=zimm_train,
                zimm_val=zimm_val,
                zsharp_train=zsharp_train,
                zsharp_val=zsharp_val,
            )

            recommended_weight = _safe_float_140(row.get("recommended_weight"), 1.0)
            if not np.isfinite(recommended_weight):
                recommended_weight = 1.0
            recommended_weight = float(np.clip(recommended_weight, 0.0, 1.0))

            alpha_base = float(np.clip(0.20 + 0.50 * confidence + 0.30 * recommended_weight, 0.05, 0.95))

            response_threshold_raw = np.nan
            if y_combined_raw.size:
                q = float(CFG.get("cell14_0_response_threshold_quantile", 0.90))
                q = float(np.clip(q, 0.50, 0.999))
                response_threshold_raw = float(np.nanquantile(np.abs(y_combined_raw), q))

            profile_key = f"profile_{i-1:05d}"

            npz_payload[f"{profile_key}__lags"] = lags.astype(np.int64)
            npz_payload[f"{profile_key}__mean_profile_bc"] = mean_profile_bc.astype(np.float32)
            npz_payload[f"{profile_key}__median_profile_bc"] = median_profile_bc.astype(np.float32)
            npz_payload[f"{profile_key}__q10_profile_bc"] = q10_profile_bc.astype(np.float32)
            npz_payload[f"{profile_key}__q90_profile_bc"] = q90_profile_bc.astype(np.float32)
            npz_payload[f"{profile_key}__mean_profile_raw"] = mean_profile_raw.astype(np.float32)
            npz_payload[f"{profile_key}__median_profile_raw"] = median_profile_raw.astype(np.float32)
            npz_payload[f"{profile_key}__finite_per_lag"] = finite_per_lag.astype(np.int64)

            profile_rows.append({
                "profile_key": profile_key,
                "pair_id": pair_id,
                "manifest_tier": manifest_tier,
                "anchor_name": anchor_name,
                "driver_col": driver_col,
                "protocol_col": protocol_col,
                "protocol_tier": protocol_tier,
                "obs_col_train": obs_col_tr,
                "obs_col_val": obs_col_val,
                "lag_lo": int(lag_lo),
                "lag_hi": int(lag_hi),
                "profile_lag_lo": int(profile_lag_lo),
                "profile_lag_hi": int(profile_lag_hi),
                "profile_lag_count": int(len(lags)),
                "train_events": int(train_events),
                "val_events": int(val_events),
                "combined_events": int(combined_events),
                "train_coverage_near": train_cov,
                "val_coverage_near": val_cov,
                "train_event_mean_transformed": train_event_mean,
                "val_event_mean_transformed": val_event_mean,
                "train_pre_mean_transformed": train_pre_mean,
                "val_pre_mean_transformed": val_pre_mean,
                "train_uplift_transformed": train_uplift,
                "val_uplift_transformed": val_uplift,
                "mean_uplift_transformed": float(np.nanmean([train_uplift, val_uplift])),
                "baseline_mean_transformed": baseline_mean,
                "baseline_median_transformed": baseline_median,
                "support_lo_raw_trainval": support_lo_raw,
                "support_hi_raw_trainval": support_hi_raw,
                "support_lo_transformed_trainval": trans_lo,
                "support_hi_transformed_trainval": trans_hi,
                "response_threshold_raw_trainval": response_threshold_raw,
                "profile_abs_cap_transformed": profile_abs_cap,
                "recommended_weight": recommended_weight,
                "profile_confidence": confidence,
                "alpha_base_recommended": alpha_base,
                "replication_score": _safe_float_140(row.get("replication_score"), np.nan),
                "consensus_lag": _safe_float_140(row.get("consensus_lag"), np.nan),
                "peak_lag_sec_train": _safe_float_140(row.get("peak_lag_sec_train"), np.nan),
                "peak_lag_sec_val": _safe_float_140(row.get("peak_lag_sec_val"), np.nan),
                "z_immediate_train": zimm_train,
                "z_immediate_val": zimm_val,
                "z_sharpness_train": zsharp_train,
                "z_sharpness_val": zsharp_val,
                "profile_valid": True,
                "rejection_reason": "",
                "TEST_real_values_used": False,
            })

    audit_rows.append({
        "pair_id": pair_id,
        "manifest_tier": manifest_tier,
        "anchor_name": anchor_name,
        "driver_col": driver_col,
        "protocol_col": protocol_col,
        "protocol_tier": protocol_tier,
        "lag_lo": int(lag_lo),
        "lag_hi": int(lag_hi),
        "profile_lag_lo": int(profile_lag_lo),
        "profile_lag_hi": int(profile_lag_hi),
        "train_events": int(train_events),
        "val_events": int(val_events),
        "combined_events": int(combined_events),
        "train_coverage_near": train_cov,
        "val_coverage_near": val_cov,
        "profile_valid": bool(valid),
        "rejection_reason": "|".join(rejection_reasons),
        "TEST_real_values_used": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
    })

profiles_df = pd.DataFrame(profile_rows)
audit_df = pd.DataFrame(audit_rows)

valid_profiles_n = int(len(profiles_df))

if valid_profiles_n == 0 and bool(CFG.get("cell14_0_fail_if_no_valid_profiles", True)):
    raise RuntimeError(
        "[Cell14.0] No valid TRAIN/VAL Zigbee coupling profiles were built. "
        f"See audit path: {profile_audit_csv}"
    )

# ----------------------------------------------------------
# 6) Save artifacts
# ----------------------------------------------------------
profiles_df.to_csv(profile_csv, index=False)
audit_df.to_csv(profile_audit_csv, index=False)

np.savez_compressed(profile_npz, **npz_payload)

profile_counts_by_protocol = (
    profiles_df["protocol_col"].astype(str).value_counts().sort_index().to_dict()
    if len(profiles_df)
    else {}
)
profile_counts_by_manifest_tier = (
    profiles_df["manifest_tier"].astype(str).value_counts().sort_index().to_dict()
    if len(profiles_df)
    else {}
)

contract = {
    "cell": "14.0",
    "version": CELL140_VERSION,
    "role": "TRAIN_VAL_only_zigbee_coupling_profile_builder",
    "quality_dimension": "Q4_cross_modal_consistency_repair_prerequisite",
    "upstream_contract_versions": {
        "cell13_4": version134_140,
        "cell13_5": version135_140,
        "cell13_6": version136_140
    },
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_140,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_140),
        "pair_pass_rate": broad_q4_pair_pass_rate_140,
        "publication_blocker_record_n": int(broad_q4_publication_blocker_record_n_140)
    },
    "manifest_q4_status_carried_forward": {
        "pairs_total": int(manifest_q4_pairs_total_140),
        "publication_blocker_n": int(manifest_q4_publication_blocker_n_140),
        "manifest_pair_pass_rate": manifest_q4_pair_pass_rate_140,
        "interpretation": "Cell 14.0 builds TRAIN/VAL-only repair profiles because both broad Q4 and manifest Zigbee Q4 are currently blocked."
    },
    "input_manifest_pairs": int(len(pairs)),
    "valid_profiles": int(valid_profiles_n),
    "audit_rows": int(len(audit_df)),
    "profile_counts_by_protocol": profile_counts_by_protocol,
    "profile_counts_by_manifest_tier": profile_counts_by_manifest_tier,
    "profile_learning": {
        "splits_used": ["TRAIN", "VAL"],
        "TEST_real_values_used": False,
        "protocol_transform": str(CFG.get("cell14_0_protocol_transform", "identity")),
        "profile_pre_margin": int(PROFILE_PRE_MARGIN_140),
        "profile_post_margin": int(PROFILE_POST_MARGIN_140),
        "allowed_protocol_tiers": list(sorted(allowed_tiers)),
        "allowed_protocol_cols": list(sorted(allowed_protocol_cols)),
    },
    "thresholds": {
        "min_train_events": int(CFG.get("cell14_0_min_train_events", 20)),
        "min_val_events": int(CFG.get("cell14_0_min_val_events", 5)),
        "min_combined_events": int(CFG.get("cell14_0_min_combined_events", 30)),
        "min_modality_coverage": float(CFG.get("cell14_0_min_modality_coverage", 0.20)),
    },
    "strict_contract": {
        "train_values_used_for_profile_learning": True,
        "profile_learning_is_repair_prerequisite": True,
        "broad_q4_status_carried_forward": True,
        "manifest_q4_status_carried_forward": True,
        "val_values_used_for_profile_learning": True,
        "TEST_real_values_used": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
    },
    "outputs": {
        "profile_csv": profile_csv,
        "profile_audit_csv": profile_audit_csv,
        "profile_npz": profile_npz,
        "profile_contract_json": profile_contract_json,
        "profile_contract_canonical_json": profile_contract_canonical_json,
        "profile_manifest_json": profile_manifest_json,
    },
}

_write_json_140(profile_contract_json, contract)
_write_json_140(profile_contract_canonical_json, contract)

manifest = {
    "cell": "14.0",
    "version": CELL140_VERSION,
    "created_outputs": contract["outputs"],
    "valid_profiles": int(valid_profiles_n),
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "manifest_q4_status_carried_forward": contract["manifest_q4_status_carried_forward"],
    "profile_counts_by_protocol": profile_counts_by_protocol,
    "strict_contract": contract["strict_contract"],
}

_write_json_140(profile_manifest_json, manifest)

hashes = {
    "profile_csv_sha256": _sha256_file_140(profile_csv),
    "profile_audit_csv_sha256": _sha256_file_140(profile_audit_csv),
    "profile_npz_sha256": _sha256_file_140(profile_npz),
    "profile_contract_json_sha256": _sha256_file_140(profile_contract_json),
    "profile_contract_canonical_json_sha256": _sha256_file_140(profile_contract_canonical_json),
    "profile_manifest_json_sha256": _sha256_file_140(profile_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_140(profile_contract_json, contract)
_write_json_140(profile_contract_canonical_json, contract)
_write_json_140(profile_manifest_json, manifest)

# ----------------------------------------------------------
# 7) Export globals for 14.1
# ----------------------------------------------------------
globals()["CELL140_VERSION"] = CELL140_VERSION
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILES_DF"] = profiles_df
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILE_AUDIT_DF"] = audit_df
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILE_CONTRACT"] = contract
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILE_CSV"] = profile_csv
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILE_AUDIT_CSV"] = profile_audit_csv
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILE_NPZ"] = profile_npz
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILE_CONTRACT_JSON"] = profile_contract_json
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILE_CONTRACT_CANONICAL_JSON"] = profile_contract_canonical_json
globals()["CELL14_0_ZIGBEE_COUPLING_PROFILE_MANIFEST_JSON"] = profile_manifest_json

log(
    "[Cell14.0] TRAIN/VAL Zigbee coupling profiles built | "
    f"manifest_pairs={len(pairs)} | "
    f"valid_profiles={valid_profiles_n} | "
    f"profile_counts_by_protocol={profile_counts_by_protocol}"
)
log(f"[Cell14.0] Saved profiles: {profile_csv} | rows={len(profiles_df)}")
log(f"[Cell14.0] Saved audit: {profile_audit_csv} | rows={len(audit_df)}")
log(f"[Cell14.0] Saved profile NPZ: {profile_npz}")
log(f"[Cell14.0] Saved contract: {profile_contract_json}")
log(f"[Cell14.0] Saved canonical contract: {profile_contract_canonical_json}")
log(
    "[Cell14.0] Upstream Q4 status carried forward | "
    f"broad_status={broad_q4_status_140} | "
    f"broad_pair_blocker_n={broad_q4_pair_blocker_n_140} | "
    f"manifest_blocker_n={manifest_q4_publication_blocker_n_140} | "
    f"manifest_pair_pass_rate={manifest_q4_pair_pass_rate_140}"
)
log(
    "[Cell14.0] Contract flags | "
    "train_values_used_for_profile_learning=True | "
    "val_values_used_for_profile_learning=True | "
    "TEST_real_values_used=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | profile_learning_is_repair_prerequisite=True"
)
log("--- END: Cell 14.0 - TRAIN/VAL-only Zigbee coupling profile builder (v1.1 Q4-failure-aware strict) ---")

gc.collect()