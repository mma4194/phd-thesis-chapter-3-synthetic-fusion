# ==========================================================
# CELL 14.3 - Coupling-conditioned Q4 re-evaluation
# v1.1 STUDY-THESIS strict manifest Q4 post-coupling QA + broad impact audit, repair-candidate/status-aware
#
# Role:
#   - Re-run manifest-informed Zigbee Q4 QA after Cell 14.1/14.2.
#   - Compare pre-coupling Cell 13.6 metrics vs post-coupling metrics.
#   - Audit whether broad Cell 13.1-13.4 pair registry is affected by the
#     Zigbee-targeted repair.
#
# Scientific contract:
#   - TEST real values are used for QA only.
#   - No synthetic values are mutated.
#   - No model fitting.
#   - No generator selection.
#   - No materialization.
#
# Inputs:
#   IOT_FULL_SYN_TEST_130
#   IOT_REAL_TEST_REF_130
#   PROTOCOL_REAL_TEST_REF_130
#   PROTOCOL_SYN_TEST_COUPLED_ZIGBEE
#   CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF
#   CELL13_6_MANIFEST_Q4_PAIR_METRICS_DF
#   CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF
#   CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT
#
# Outputs:
#   reports/cell14_3_manifest_q4_post_pair_metrics.csv
#   reports/cell14_3_manifest_q4_pre_post_delta.csv
#   reports/cell14_3_manifest_q4_summary_by_tier.csv
#   reports/cell14_3_broad_q4_impact_audit.csv
#   reports/cell14_3_q4_repair_contract.json
#   artifacts/cell14_3_manifest_q4_curves.npz
#   artifacts/cell14_3_q4_repair_manifest.json
# ==========================================================

log("--- START: Cell 14.3 - Coupling-conditioned Q4 re-evaluation (v1.1 repair-candidate/status-aware strict) ---")

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
_required_143 = [
    "CFG", "log",
    "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_FULL_SYN_TEST_130",
    "IOT_REAL_TEST_REF_130",
    "PROTOCOL_REAL_TEST_REF_130",
    "PROTOCOL_SYN_TEST_COUPLED_ZIGBEE",
    "CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF",
    "CELL13_6_MANIFEST_Q4_PAIR_METRICS_DF",
    "CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF",
    "CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT",
    "CELL14_2_COUPLED_CPS_CONTRACT",
    "CELL13_6_MANIFEST_Q4_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
]
_missing_143 = [k for k in _required_143 if k not in globals()]
if _missing_143:
    raise RuntimeError(f"[Cell14.3] Missing required globals: {_missing_143}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TE = int(len(df_te))

CELL143_VERSION = "cell14_3_coupling_conditioned_q4_re_evaluation_v1_1_repair_candidate_status_aware"

CFG["cell14_3_version"] = CELL143_VERSION
CFG["cell14_3_TEST_real_values_used_for_QA_reference"] = True
CFG["cell14_3_synthetic_values_mutated"] = False
CFG["cell14_3_selection_done_here"] = False
CFG["cell14_3_generator_fit_done_here"] = False
CFG["cell14_3_materialization_done_here"] = False
CFG["cell14_3_QA_done_here"] = True
CFG["cell14_3_repair_candidate_not_final_claim"] = True
CFG["cell14_3_broad_q4_status_carried_forward"] = True
CFG["cell14_3_manifest_q4_status_carried_forward"] = True
CFG["cell14_3_does_not_update_broad_q4_metrics"] = True

CFG.setdefault("cell14_3_eta_pre_seconds", int(CFG.get("cell13_6_eta_pre_seconds", 10)))
CFG.setdefault("cell14_3_eta_post_seconds", int(CFG.get("cell13_6_eta_post_seconds", 30)))
CFG.setdefault("cell14_3_profile_extra_lag_margin", int(CFG.get("cell13_6_profile_extra_lag_margin", 5)))
CFG.setdefault("cell14_3_min_events_for_pass", int(CFG.get("cell13_6_min_events_for_pass", 5)))
CFG.setdefault("cell14_3_protocol_transform", str(CFG.get("cell13_6_protocol_transform", "log1p_abs_signed")))
CFG.setdefault("cell14_3_response_threshold_quantile", float(CFG.get("cell13_6_response_threshold_quantile", 0.90)))
CFG.setdefault("cell14_3_eta_similarity_pass", float(CFG.get("cell13_6_eta_similarity_pass", 0.70)))
CFG.setdefault("cell14_3_eta_similarity_warning", float(CFG.get("cell13_6_eta_similarity_warning", 0.50)))
CFG.setdefault("cell14_3_profile_similarity_pass", float(CFG.get("cell13_6_profile_similarity_pass", 0.70)))
CFG.setdefault("cell14_3_profile_similarity_warning", float(CFG.get("cell13_6_profile_similarity_warning", 0.50)))
CFG.setdefault("cell14_3_lag_peak_error_pass", float(CFG.get("cell13_6_lag_peak_error_pass", 2)))
CFG.setdefault("cell14_3_lag_peak_error_warning", float(CFG.get("cell13_6_lag_peak_error_warning", 5)))
CFG.setdefault("cell14_3_response_window_rate_error_pass", float(CFG.get("cell13_6_response_window_rate_error_pass", 0.10)))
CFG.setdefault("cell14_3_response_window_rate_error_warning", float(CFG.get("cell13_6_response_window_rate_error_warning", 0.25)))
CFG.setdefault("cell14_3_store_curves", True)
CFG.setdefault("cell14_3_norm_eps", 1e-9)

ETA_PRE_143 = int(CFG.get("cell14_3_eta_pre_seconds", 10))
ETA_POST_143 = int(CFG.get("cell14_3_eta_post_seconds", 30))
EXTRA_MARGIN_143 = int(CFG.get("cell14_3_profile_extra_lag_margin", 5))

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
post_pair_metrics_csv = os.path.join(REPORT_DIR, "cell14_3_manifest_q4_post_pair_metrics.csv")
pre_post_delta_csv = os.path.join(REPORT_DIR, "cell14_3_manifest_q4_pre_post_delta.csv")
summary_by_tier_csv = os.path.join(REPORT_DIR, "cell14_3_manifest_q4_summary_by_tier.csv")
blockers_csv = os.path.join(REPORT_DIR, "cell14_3_manifest_q4_post_blockers.csv")
broad_impact_csv = os.path.join(REPORT_DIR, "cell14_3_broad_q4_impact_audit.csv")
contract_json = os.path.join(REPORT_DIR, "cell14_3_q4_repair_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_3_q4_repair_contract_v1_1_THESIS.json")
curves_npz = os.path.join(ARTDIR, "cell14_3_manifest_q4_curves.npz")
manifest_json = os.path.join(ARTDIR, "cell14_3_q4_repair_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_143(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_143(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_143(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_143(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_143(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_143(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_143(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_143(obj.to_dict())
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

def _write_json_143(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_143(payload), f, indent=2, sort_keys=True)

def _sha256_file_143(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_143(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_143(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _transform_protocol_143(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell14_3_protocol_transform", "log1p_abs_signed")).strip().lower()

    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.log1p(np.maximum(arr, 0.0))
    elif mode == "log1p_abs":
        out = np.log1p(np.abs(arr))
    elif mode == "log1p_abs_signed":
        out = np.sign(arr) * np.log1p(np.abs(arr))
    else:
        raise RuntimeError(f"[Cell14.3] Unknown protocol transform: {mode}")

    out[~np.isfinite(out)] = np.nan
    return out

def _event_starts_143(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    active = np.isfinite(arr) & (arr > 0.5)
    if active.size == 0:
        return np.asarray([], dtype=np.int64)
    prev = np.r_[False, active[:-1]]
    starts = active & (~prev)
    return np.flatnonzero(starts).astype(np.int64)

def _edge_safe_events_143(idx: np.ndarray, n: int, lo: int, hi: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx
    keep = (idx + lo >= 0) & (idx + hi < n)
    return idx[keep]

def _profile_around_events_143(y_raw: np.ndarray, event_idx: np.ndarray, lag_lo: int, lag_hi: int):
    y = _transform_protocol_143(y_raw)
    lags = np.arange(lag_lo, lag_hi + 1, dtype=np.int64)

    event_idx = _edge_safe_events_143(event_idx, len(y), lag_lo, lag_hi)
    if event_idx.size == 0:
        return lags, np.full(len(lags), np.nan), {
            "event_count_window_valid": 0,
            "profile_valid": False,
            "profile_reason": "no_edge_safe_events",
        }

    rows = []
    for e in event_idx:
        seg = y[e + lag_lo : e + lag_hi + 1]
        if len(seg) == len(lags):
            rows.append(seg)

    if not rows:
        return lags, np.full(len(lags), np.nan), {
            "event_count_window_valid": 0,
            "profile_valid": False,
            "profile_reason": "no_complete_segments",
        }

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

    return lags, prof, {
        "event_count_window_valid": int(event_idx.size),
        "profile_valid": bool(np.isfinite(prof).sum() >= 2),
        "profile_reason": "",
        "profile_baseline": baseline,
    }

def _similarity_143(real_prof: np.ndarray, syn_prof: np.ndarray):
    r = np.asarray(real_prof, dtype=np.float64)
    s = np.asarray(syn_prof, dtype=np.float64)
    valid = np.isfinite(r) & np.isfinite(s)

    if int(valid.sum()) < 2:
        return {
            "similarity": np.nan,
            "mae": np.nan,
            "norm_mae": np.nan,
            "corr": np.nan,
            "compare_valid": False,
            "compare_reason": "insufficient_finite_lags",
        }

    rr = r[valid]
    ss = s[valid]

    mae = float(np.mean(np.abs(rr - ss)))
    scale = float(np.nanmean(np.abs(rr)) + np.nanstd(rr) + float(CFG.get("cell14_3_norm_eps", 1e-9)))
    norm_mae = float(mae / max(scale, float(CFG.get("cell14_3_norm_eps", 1e-9))))

    if np.std(rr) <= 1e-12 or np.std(ss) <= 1e-12:
        corr = 1.0 if np.allclose(rr, ss, atol=1e-9) else 0.0
    else:
        corr = float(np.corrcoef(rr, ss)[0, 1])

    sim = float(0.5 * max(0.0, corr) + 0.5 * np.exp(-norm_mae))

    return {
        "similarity": sim,
        "mae": mae,
        "norm_mae": norm_mae,
        "corr": corr,
        "compare_valid": True,
        "compare_reason": "",
    }

def _peak_lag_143(lags: np.ndarray, prof: np.ndarray):
    prof = np.asarray(prof, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.int64)
    valid = np.isfinite(prof)
    if not valid.any():
        return np.nan, np.nan
    valid_idx = np.flatnonzero(valid)
    local_i = int(np.nanargmax(np.abs(prof[valid])))
    idx = int(valid_idx[local_i])
    return int(lags[idx]), float(prof[idx])

def _response_threshold_from_real_143(y_raw: np.ndarray) -> float:
    y = _transform_protocol_143(y_raw)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.nan
    q = float(CFG.get("cell14_3_response_threshold_quantile", 0.90))
    q = float(np.clip(q, 0.50, 0.999))
    return float(np.nanquantile(np.abs(y), q))

def _response_rate_143(y_raw: np.ndarray, event_idx: np.ndarray, lag_lo: int, lag_hi: int, threshold: float) -> float:
    y = _transform_protocol_143(y_raw)
    event_idx = _edge_safe_events_143(event_idx, len(y), lag_lo, lag_hi)
    if event_idx.size == 0 or not np.isfinite(threshold):
        return np.nan

    hits = []
    for e in event_idx:
        seg = y[e + lag_lo : e + lag_hi + 1]
        seg = seg[np.isfinite(seg)]
        hits.append(bool(seg.size and np.nanmax(np.abs(seg)) >= threshold))

    return float(np.mean(hits)) if hits else np.nan

def _status_143(row: dict):
    real_events = int(_safe_float_143(row.get("real_event_count_window_valid"), 0))
    syn_events = int(_safe_float_143(row.get("syn_event_count_window_valid"), 0))
    min_events = int(CFG.get("cell14_3_min_events_for_pass", 5))

    eta_sim = _safe_float_143(row.get("ETA_similarity"), np.nan)
    profile_sim = _safe_float_143(row.get("manifest_window_profile_similarity"), np.nan)
    lag_peak_error = _safe_float_143(row.get("lag_peak_error"), np.nan)
    resp_err = _safe_float_143(row.get("response_window_rate_error"), np.nan)

    if real_events < min_events and syn_events < min_events:
        return "not_evaluable", "insufficient_real_and_synthetic_events"

    if real_events < min_events:
        return "warning", "insufficient_real_events"

    if syn_events < min_events:
        return "warning", "insufficient_synthetic_events"

    fatal = []
    warn = []

    if np.isfinite(eta_sim):
        if eta_sim < float(CFG.get("cell14_3_eta_similarity_warning", 0.50)):
            fatal.append("ETA_similarity_fatal")
        elif eta_sim < float(CFG.get("cell14_3_eta_similarity_pass", 0.70)):
            warn.append("ETA_similarity_warning")
    else:
        warn.append("ETA_similarity_not_finite")

    if np.isfinite(profile_sim):
        if profile_sim < float(CFG.get("cell14_3_profile_similarity_warning", 0.50)):
            fatal.append("manifest_profile_similarity_fatal")
        elif profile_sim < float(CFG.get("cell14_3_profile_similarity_pass", 0.70)):
            warn.append("manifest_profile_similarity_warning")
    else:
        warn.append("manifest_profile_similarity_not_finite")

    if np.isfinite(lag_peak_error):
        if lag_peak_error > float(CFG.get("cell14_3_lag_peak_error_warning", 5)):
            fatal.append("lag_peak_error_fatal")
        elif lag_peak_error > float(CFG.get("cell14_3_lag_peak_error_pass", 2)):
            warn.append("lag_peak_error_warning")

    if np.isfinite(resp_err):
        if resp_err > float(CFG.get("cell14_3_response_window_rate_error_warning", 0.25)):
            fatal.append("response_window_rate_error_fatal")
        elif resp_err > float(CFG.get("cell14_3_response_window_rate_error_pass", 0.10)):
            warn.append("response_window_rate_error_warning")

    if fatal:
        return "fatal", "|".join(fatal)
    if warn:
        return "warning", "|".join(warn)
    return "pass", "post_coupling_manifest_q4_pair_pass"

def _mean_numeric_143(df, col):
    return float(pd.to_numeric(df[col], errors="coerce").mean()) if col in df.columns and len(df) else np.nan

def _status_counts_143(df, col):
    if col not in df.columns or not len(df):
        return {}
    return df[col].astype(str).value_counts().sort_index().to_dict()

# ----------------------------------------------------------
# 3) Validate upstream repair and pre-repair contracts
# ----------------------------------------------------------
def _require_contract_version_143(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.3] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.3] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

version141_143 = _require_contract_version_143(
    CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT,
    "CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT",
    "cell14_1_coupling_conditioned_zigbee_protocol_materialization_v1_1",
)
version142_143 = _require_contract_version_143(
    CELL14_2_COUPLED_CPS_CONTRACT,
    "CELL14_2_COUPLED_CPS_CONTRACT",
    "cell14_2_coupling_conditioned_cps_assembly_v1_1",
)
version136_143 = _require_contract_version_143(
    CELL13_6_MANIFEST_Q4_CONTRACT,
    "CELL13_6_MANIFEST_Q4_CONTRACT",
    "cell13_6_manifest_informed_zigbee_q4_qa_v1_1",
)
version134_143 = _require_contract_version_143(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)

strict141_143 = CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get("strict_contract", {})
if bool(strict141_143.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.3] Cell 14.1 contract indicates TEST real values were used for materialization.")
if int(CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get("safety_failure_count", -1)) != 0:
    raise RuntimeError(
        "[Cell14.3] Cell 14.1 safety failures are nonzero: "
        f"{CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get('safety_failure_count')}"
    )
if not bool(strict141_143.get("only_target_protocol_columns_mutated", False)):
    raise RuntimeError("[Cell14.3] Cell 14.1 did not guarantee target-only protocol mutation.")
if not bool(strict141_143.get("only_legal_windows_mutated", False)):
    raise RuntimeError("[Cell14.3] Cell 14.1 did not guarantee legal-window-only mutation.")
if bool(strict141_143.get("iot_values_mutated", True)):
    raise RuntimeError("[Cell14.3] Cell 14.1 indicates IoT values were mutated.")

strict142_143 = CELL14_2_COUPLED_CPS_CONTRACT.get("strict_contract", {})
if bool(strict142_143.get("TEST_target_values_used", True)):
    raise RuntimeError("[Cell14.3] Cell 14.2 contract indicates TEST target values were used.")
if not bool(strict142_143.get("assembly_done_here", False)):
    raise RuntimeError("[Cell14.3] Cell 14.2 did not declare assembly_done_here.")
if not bool(strict142_143.get("repair_candidate_not_final_claim", False)):
    raise RuntimeError("[Cell14.3] Cell 14.2 did not declare repair_candidate_not_final_claim.")
if bool(strict142_143.get("synthetic_values_mutated_here", True)):
    raise RuntimeError("[Cell14.3] Cell 14.2 contract indicates synthetic values were mutated during assembly.")
if not bool(strict142_143.get("protocol_values_from_14_1", False)):
    raise RuntimeError("[Cell14.3] Cell 14.2 did not declare protocol_values_from_14_1.")
if not bool(strict142_143.get("iot_values_preserved_from_12g", False)):
    raise RuntimeError("[Cell14.3] Cell 14.2 did not declare IoT preservation from 12.g/13.0.")
if not bool(strict142_143.get("target_protocol_drift_matches_14_1", False)):
    raise RuntimeError("[Cell14.3] Cell 14.2 target drift does not match Cell 14.1.")

broad_q4_status_143 = str(CELL13_4_Q4_PUBLICATION_SUMMARY.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_143 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("pair_blocker_n", 0) or 0
)
broad_q4_pair_pass_rate_143 = _safe_float_143(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)
broad_q4_publication_blocker_record_n_143 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get(
        "publication_blocker_record_n",
        CELL13_4_Q4_PUBLICATION_SUMMARY.get("publication_blocker_n", 0),
    ) or 0
)

pre_manifest_summary_143 = CELL13_6_MANIFEST_Q4_CONTRACT.get("overall_summary", {})
pre_manifest_publication_blocker_n_143 = int(pre_manifest_summary_143.get("publication_blocker_n", 0) or 0)
pre_manifest_pair_pass_rate_143 = _safe_float_143(
    pre_manifest_summary_143.get("manifest_pair_pass_rate", np.nan),
    np.nan,
)
pre_manifest_pairs_total_143 = int(pre_manifest_summary_143.get("pairs_total", 0) or 0)

if broad_q4_status_143 != "blocker":
    log(
        "[Cell14.3] WARNING: broad Q4 summary is not blocker. "
        f"overall_status={broad_q4_status_143}"
    )

# ----------------------------------------------------------
# 4) Validate inputs
# ----------------------------------------------------------
pairs = CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF.copy()
pre = CELL13_6_MANIFEST_Q4_PAIR_METRICS_DF.copy()

if len(pairs) == 0:
    raise RuntimeError("[Cell14.3] Empty manifest registry from Cell 13.5.")

if len(pre) == 0:
    raise RuntimeError("[Cell14.3] Empty pre-coupling manifest Q4 metrics from Cell 13.6.")

for frame_name, frame in [
    ("IOT_FULL_SYN_TEST_130", IOT_FULL_SYN_TEST_130),
    ("IOT_REAL_TEST_REF_130", IOT_REAL_TEST_REF_130),
    ("PROTOCOL_REAL_TEST_REF_130", PROTOCOL_REAL_TEST_REF_130),
    ("PROTOCOL_SYN_TEST_COUPLED_ZIGBEE", PROTOCOL_SYN_TEST_COUPLED_ZIGBEE),
]:
    if len(frame) != N_TE:
        raise RuntimeError(f"[Cell14.3] {frame_name} row mismatch: got={len(frame)} expected={N_TE}")
    frame.columns = frame.columns.astype(str)

required_pair_cols = [
    "pair_id", "manifest_tier", "anchor_name", "driver_col",
    "protocol_col", "protocol_tier", "lag_lo", "lag_hi",
]
missing_pair_cols = [c for c in required_pair_cols if c not in pairs.columns]
if missing_pair_cols:
    raise RuntimeError(f"[Cell14.3] Pair registry missing columns: {missing_pair_cols}")

pairs = pairs[pairs.get("eligible_for_13_6_manifest_q4", True).astype(bool)].copy()

non_zigbee_pairs_143 = pairs[pairs["protocol_tier"].astype(str).str.lower().ne("zigbee")]
if len(non_zigbee_pairs_143):
    raise RuntimeError(
        "[Cell14.3] This repair re-evaluation is scoped to manifest Zigbee pairs only. "
        f"Non-Zigbee rows={len(non_zigbee_pairs_143)}"
    )

missing = []
for c in sorted(set(pairs["driver_col"].astype(str))):
    if c not in IOT_FULL_SYN_TEST_130.columns:
        missing.append(f"missing_syn_driver:{c}")
    if c not in IOT_REAL_TEST_REF_130.columns:
        missing.append(f"missing_real_driver:{c}")
for c in sorted(set(pairs["protocol_col"].astype(str))):
    if c not in PROTOCOL_SYN_TEST_COUPLED_ZIGBEE.columns:
        missing.append(f"missing_coupled_protocol:{c}")
    if c not in PROTOCOL_REAL_TEST_REF_130.columns:
        missing.append(f"missing_real_protocol:{c}")

if missing:
    raise RuntimeError(f"[Cell14.3] Missing QA columns. Preview={missing[:30]}")

# ----------------------------------------------------------
# 5) Re-run manifest Q4 QA using coupled protocol matrix
# ----------------------------------------------------------
metric_rows = []
curve_store = {}

log(
    "[Cell14.3] Re-running manifest-informed Q4 QA after coupling | "
    f"pairs={len(pairs)}"
)

for i, r in enumerate(pairs.itertuples(index=False), start=1):
    if i == 1 or i % 10 == 0 or i == len(pairs):
        log(f"[Cell14.3] progress {i}/{len(pairs)}")

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

    eta_lo = min(-ETA_PRE_143, lag_lo - EXTRA_MARGIN_143)
    eta_hi = max(ETA_POST_143, lag_hi + EXTRA_MARGIN_143)

    profile_lo = lag_lo
    profile_hi = lag_hi

    d_real = _to_num_array_143(IOT_REAL_TEST_REF_130, driver_col)
    d_syn = _to_num_array_143(IOT_FULL_SYN_TEST_130, driver_col)

    p_real = _to_num_array_143(PROTOCOL_REAL_TEST_REF_130, protocol_col)
    p_syn_coupled = _to_num_array_143(PROTOCOL_SYN_TEST_COUPLED_ZIGBEE, protocol_col)

    e_real = _event_starts_143(d_real)
    e_syn = _event_starts_143(d_syn)

    eta_lags_real, eta_real, eta_real_meta = _profile_around_events_143(p_real, e_real, eta_lo, eta_hi)
    eta_lags_syn, eta_syn, eta_syn_meta = _profile_around_events_143(p_syn_coupled, e_syn, eta_lo, eta_hi)

    if not np.array_equal(eta_lags_real, eta_lags_syn):
        raise RuntimeError(f"[Cell14.3] ETA lag mismatch for pair {pair_id}")

    eta_sim = _similarity_143(eta_real, eta_syn)

    man_lags_real, man_real, man_real_meta = _profile_around_events_143(p_real, e_real, profile_lo, profile_hi)
    man_lags_syn, man_syn, man_syn_meta = _profile_around_events_143(p_syn_coupled, e_syn, profile_lo, profile_hi)

    if not np.array_equal(man_lags_real, man_lags_syn):
        raise RuntimeError(f"[Cell14.3] Manifest lag mismatch for pair {pair_id}")

    man_sim = _similarity_143(man_real, man_syn)

    real_peak_lag, real_peak_value = _peak_lag_143(eta_lags_real, eta_real)
    syn_peak_lag, syn_peak_value = _peak_lag_143(eta_lags_syn, eta_syn)

    lag_peak_error = (
        abs(float(real_peak_lag) - float(syn_peak_lag))
        if np.isfinite(real_peak_lag) and np.isfinite(syn_peak_lag)
        else np.nan
    )

    real_thr = _response_threshold_from_real_143(p_real)
    real_resp_rate = _response_rate_143(p_real, e_real, profile_lo, profile_hi, real_thr)
    syn_resp_rate = _response_rate_143(p_syn_coupled, e_syn, profile_lo, profile_hi, real_thr)

    response_window_rate_error = (
        abs(real_resp_rate - syn_resp_rate)
        if np.isfinite(real_resp_rate) and np.isfinite(syn_resp_rate)
        else np.nan
    )

    manifest_real_mean = float(np.nanmean(man_real)) if np.isfinite(man_real).any() else np.nan
    manifest_syn_mean = float(np.nanmean(man_syn)) if np.isfinite(man_syn).any() else np.nan
    manifest_window_mean_error = (
        abs(manifest_real_mean - manifest_syn_mean)
        if np.isfinite(manifest_real_mean) and np.isfinite(manifest_syn_mean)
        else np.nan
    )

    metric_row = {
        "pair_id": pair_id,
        "manifest_tier": manifest_tier,
        "anchor_name": anchor_name,
        "driver_col": driver_col,
        "protocol_col": protocol_col,
        "protocol_tier": protocol_tier,
        "lag_lo": int(lag_lo),
        "lag_hi": int(lag_hi),
        "recommended_weight": _safe_float_143(row.get("recommended_weight"), np.nan),
        "replication_score": _safe_float_143(row.get("replication_score"), np.nan),
        "real_event_count_all": int(len(e_real)),
        "syn_event_count_all": int(len(e_syn)),
        "real_event_count_window_valid": int(eta_real_meta.get("event_count_window_valid", 0)),
        "syn_event_count_window_valid": int(eta_syn_meta.get("event_count_window_valid", 0)),
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
        "real_peak_value": real_peak_value,
        "syn_peak_value": syn_peak_value,
        "real_response_threshold": real_thr,
        "real_response_window_rate": real_resp_rate,
        "syn_response_window_rate": syn_resp_rate,
        "response_window_rate_error": response_window_rate_error,
        "manifest_real_mean": manifest_real_mean,
        "manifest_syn_mean": manifest_syn_mean,
        "manifest_window_mean_error": manifest_window_mean_error,
        "TEST_real_values_used_for_QA_reference": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
    }

    status, reason = _status_143(metric_row)
    metric_row["post_manifest_q4_status"] = status
    metric_row["post_manifest_q4_reasons"] = reason
    metric_row["post_manifest_q4_publication_blocker"] = bool(status == "fatal")

    metric_rows.append(metric_row)

    if bool(CFG.get("cell14_3_store_curves", True)):
        curve_store[f"{pair_id}__eta_lags"] = eta_lags_real.astype(np.int64)
        curve_store[f"{pair_id}__eta_real"] = eta_real.astype(np.float32)
        curve_store[f"{pair_id}__eta_syn_coupled"] = eta_syn.astype(np.float32)
        curve_store[f"{pair_id}__manifest_lags"] = man_lags_real.astype(np.int64)
        curve_store[f"{pair_id}__manifest_real"] = man_real.astype(np.float32)
        curve_store[f"{pair_id}__manifest_syn_coupled"] = man_syn.astype(np.float32)

post_df = pd.DataFrame(metric_rows)

if len(post_df) != len(pairs):
    raise RuntimeError(f"[Cell14.3] Post metrics row mismatch: got={len(post_df)} expected={len(pairs)}")

# ----------------------------------------------------------
# 6) Pre/post delta
# ----------------------------------------------------------
pre_key_cols = [
    "pair_id",
    "ETA_similarity",
    "lag_peak_error",
    "response_window_rate_error",
    "manifest_window_profile_similarity",
    "manifest_window_mean_error",
    "manifest_q4_status",
    "manifest_q4_reasons",
    "manifest_q4_publication_blocker",
]
pre_key_cols = [c for c in pre_key_cols if c in pre.columns]

pre_s = pre[pre_key_cols].copy()
rename_pre = {
    "ETA_similarity": "pre_ETA_similarity",
    "lag_peak_error": "pre_lag_peak_error",
    "response_window_rate_error": "pre_response_window_rate_error",
    "manifest_window_profile_similarity": "pre_manifest_window_profile_similarity",
    "manifest_window_mean_error": "pre_manifest_window_mean_error",
    "manifest_q4_status": "pre_manifest_q4_status",
    "manifest_q4_reasons": "pre_manifest_q4_reasons",
    "manifest_q4_publication_blocker": "pre_manifest_q4_publication_blocker",
}
pre_s = pre_s.rename(columns=rename_pre)

post_s = post_df.copy().rename(columns={
    "ETA_similarity": "post_ETA_similarity",
    "lag_peak_error": "post_lag_peak_error",
    "response_window_rate_error": "post_response_window_rate_error",
    "manifest_window_profile_similarity": "post_manifest_window_profile_similarity",
    "manifest_window_mean_error": "post_manifest_window_mean_error",
})

delta = post_s.merge(pre_s, on="pair_id", how="left")

for pre_col, post_col, delta_col, direction in [
    ("pre_ETA_similarity", "post_ETA_similarity", "delta_ETA_similarity", "higher_better"),
    ("pre_manifest_window_profile_similarity", "post_manifest_window_profile_similarity", "delta_manifest_profile_similarity", "higher_better"),
    ("pre_lag_peak_error", "post_lag_peak_error", "delta_lag_peak_error", "lower_better"),
    ("pre_response_window_rate_error", "post_response_window_rate_error", "delta_response_window_rate_error", "lower_better"),
    ("pre_manifest_window_mean_error", "post_manifest_window_mean_error", "delta_manifest_window_mean_error", "lower_better"),
]:
    if pre_col in delta.columns and post_col in delta.columns:
        delta[delta_col] = pd.to_numeric(delta[post_col], errors="coerce") - pd.to_numeric(delta[pre_col], errors="coerce")
        if direction == "higher_better":
            delta[f"{delta_col}_improved"] = delta[delta_col] > 0
        else:
            delta[f"{delta_col}_improved"] = delta[delta_col] < 0

delta["post_publication_blocker"] = delta["post_manifest_q4_publication_blocker"].fillna(False).astype(bool)
if "pre_manifest_q4_publication_blocker" in delta.columns:
    delta["pre_publication_blocker"] = delta["pre_manifest_q4_publication_blocker"].fillna(False).astype(bool)
else:
    delta["pre_publication_blocker"] = np.nan

# ----------------------------------------------------------
# 7) Summary by tier / manifest tier
# ----------------------------------------------------------
summary_rows = []

for (tier, mtier), sub in post_df.groupby(["protocol_tier", "manifest_tier"], dropna=False):
    sub = sub.copy()
    evaluable = sub[~sub["post_manifest_q4_status"].astype(str).eq("not_evaluable")]

    summary_rows.append({
        "protocol_tier": tier,
        "manifest_tier": mtier,
        "pairs_total": int(len(sub)),
        "pairs_evaluable": int(len(evaluable)),
        "pass_n": int((sub["post_manifest_q4_status"].astype(str) == "pass").sum()),
        "warning_n": int((sub["post_manifest_q4_status"].astype(str) == "warning").sum()),
        "fatal_n": int((sub["post_manifest_q4_status"].astype(str) == "fatal").sum()),
        "not_evaluable_n": int((sub["post_manifest_q4_status"].astype(str) == "not_evaluable").sum()),
        "post_manifest_pair_pass_rate": float((evaluable["post_manifest_q4_status"].astype(str) == "pass").mean()) if len(evaluable) else np.nan,
        "mean_post_ETA_similarity": float(pd.to_numeric(evaluable["ETA_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_post_lag_peak_error": float(pd.to_numeric(evaluable["lag_peak_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_post_response_window_rate_error": float(pd.to_numeric(evaluable["response_window_rate_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_post_manifest_profile_similarity": float(pd.to_numeric(evaluable["manifest_window_profile_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "post_publication_blocker_n": int(sub["post_manifest_q4_publication_blocker"].fillna(False).astype(bool).sum()),
    })

summary_df = pd.DataFrame(summary_rows)

blockers_df = post_df[post_df["post_manifest_q4_publication_blocker"].fillna(False).astype(bool)].copy()
status_counts = _status_counts_143(post_df, "post_manifest_q4_status")
evaluable = post_df[~post_df["post_manifest_q4_status"].astype(str).eq("not_evaluable")].copy()

pre_summary = {
    "pairs_total": int(len(pre)),
    "publication_blocker_n": int(pre["manifest_q4_publication_blocker"].fillna(False).astype(bool).sum()) if "manifest_q4_publication_blocker" in pre.columns else np.nan,
    "pass_rate": float((pre["manifest_q4_status"].astype(str) == "pass").mean()) if "manifest_q4_status" in pre.columns and len(pre) else np.nan,
    "mean_ETA_similarity": _mean_numeric_143(pre, "ETA_similarity"),
    "mean_lag_peak_error": _mean_numeric_143(pre, "lag_peak_error"),
    "mean_response_window_rate_error": _mean_numeric_143(pre, "response_window_rate_error"),
    "mean_manifest_profile_similarity": _mean_numeric_143(pre, "manifest_window_profile_similarity"),
    "mean_manifest_window_mean_error": _mean_numeric_143(pre, "manifest_window_mean_error"),
}

post_summary = {
    "pairs_total": int(len(post_df)),
    "pairs_evaluable": int(len(evaluable)),
    "status_counts": status_counts,
    "publication_blocker_n": int(len(blockers_df)),
    "pass_rate": float((evaluable["post_manifest_q4_status"].astype(str) == "pass").mean()) if len(evaluable) else np.nan,
    "mean_ETA_similarity": _mean_numeric_143(evaluable, "ETA_similarity"),
    "mean_lag_peak_error": _mean_numeric_143(evaluable, "lag_peak_error"),
    "mean_response_window_rate_error": _mean_numeric_143(evaluable, "response_window_rate_error"),
    "mean_manifest_profile_similarity": _mean_numeric_143(evaluable, "manifest_window_profile_similarity"),
    "mean_manifest_window_mean_error": _mean_numeric_143(evaluable, "manifest_window_mean_error"),
}

delta_summary = {
    "delta_publication_blocker_n": (
        post_summary["publication_blocker_n"] - pre_summary["publication_blocker_n"]
        if np.isfinite(pre_summary["publication_blocker_n"]) else np.nan
    ),
    "delta_pass_rate": (
        post_summary["pass_rate"] - pre_summary["pass_rate"]
        if np.isfinite(post_summary["pass_rate"]) and np.isfinite(pre_summary["pass_rate"]) else np.nan
    ),
    "delta_mean_ETA_similarity": (
        post_summary["mean_ETA_similarity"] - pre_summary["mean_ETA_similarity"]
        if np.isfinite(post_summary["mean_ETA_similarity"]) and np.isfinite(pre_summary["mean_ETA_similarity"]) else np.nan
    ),
    "delta_mean_lag_peak_error": (
        post_summary["mean_lag_peak_error"] - pre_summary["mean_lag_peak_error"]
        if np.isfinite(post_summary["mean_lag_peak_error"]) and np.isfinite(pre_summary["mean_lag_peak_error"]) else np.nan
    ),
    "delta_mean_response_window_rate_error": (
        post_summary["mean_response_window_rate_error"] - pre_summary["mean_response_window_rate_error"]
        if np.isfinite(post_summary["mean_response_window_rate_error"]) and np.isfinite(pre_summary["mean_response_window_rate_error"]) else np.nan
    ),
    "delta_mean_manifest_profile_similarity": (
        post_summary["mean_manifest_profile_similarity"] - pre_summary["mean_manifest_profile_similarity"]
        if np.isfinite(post_summary["mean_manifest_profile_similarity"]) and np.isfinite(pre_summary["mean_manifest_profile_similarity"]) else np.nan
    ),
    "delta_mean_manifest_window_mean_error": (
        post_summary["mean_manifest_window_mean_error"] - pre_summary["mean_manifest_window_mean_error"]
        if np.isfinite(post_summary["mean_manifest_window_mean_error"]) and np.isfinite(pre_summary["mean_manifest_window_mean_error"]) else np.nan
    ),
}

repair_improved_any_143 = bool(
    (
        np.isfinite(delta_summary.get("delta_mean_ETA_similarity", np.nan))
        and delta_summary["delta_mean_ETA_similarity"] > 0
    )
    or (
        np.isfinite(delta_summary.get("delta_mean_manifest_profile_similarity", np.nan))
        and delta_summary["delta_mean_manifest_profile_similarity"] > 0
    )
    or (
        np.isfinite(delta_summary.get("delta_mean_lag_peak_error", np.nan))
        and delta_summary["delta_mean_lag_peak_error"] < 0
    )
    or (
        np.isfinite(delta_summary.get("delta_mean_response_window_rate_error", np.nan))
        and delta_summary["delta_mean_response_window_rate_error"] < 0
    )
)

repair_publication_status_143 = (
    "pass" if post_summary["publication_blocker_n"] == 0 and post_summary["pass_rate"] >= 0.70
    else "warning" if repair_improved_any_143 and post_summary["publication_blocker_n"] < pre_summary["publication_blocker_n"]
    else "blocker"
)

repair_publication_reasons_143 = []
if post_summary["publication_blocker_n"] > 0:
    repair_publication_reasons_143.append("post_coupling_manifest_q4_blockers_remain")
if post_summary["pass_rate"] < 0.70:
    repair_publication_reasons_143.append("post_coupling_manifest_q4_pass_rate_below_claim_threshold")
if repair_improved_any_143:
    repair_publication_reasons_143.append("at_least_one_mean_metric_improved")
else:
    repair_publication_reasons_143.append("no_mean_metric_improvement_detected")

# ----------------------------------------------------------
# 8) Broad Q4 impact audit
# ----------------------------------------------------------
broad_pairs = CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF.copy()
if len(broad_pairs):
    broad_pairs["protocol_col"] = broad_pairs["protocol_col"].astype(str)
else:
    broad_pairs["protocol_col"] = []

target_cols = list(CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get("target_protocol_cols", []))
target_cols = [str(c) for c in target_cols]

broad_affected = broad_pairs[broad_pairs["protocol_col"].isin(target_cols)].copy() if len(broad_pairs) else pd.DataFrame()

broad_rows = [
    {
        "metric": "broad_pair_registry_rows",
        "value": int(len(broad_pairs)),
        "interpretation": "Total broad 13.x driver-protocol pair registry rows.",
    },
    {
        "metric": "coupled_zigbee_target_protocol_cols",
        "value": "|".join(target_cols),
        "interpretation": "Protocol columns changed by Cell 14.1.",
    },
    {
        "metric": "broad_pairs_using_coupled_target_cols",
        "value": int(len(broad_affected)),
        "interpretation": "Broad 13.1-13.4 metrics are not recomputed here. This count only identifies broad-registry rows whose protocol columns were changed by the Zigbee repair candidate.",
    },
]

if len(broad_affected):
    for tier, sub in broad_affected.groupby("protocol_tier", dropna=False):
        broad_rows.append({
            "metric": f"affected_broad_pairs_by_tier::{tier}",
            "value": int(len(sub)),
            "interpretation": "Broad Q4 pairs affected by coupled target protocol columns.",
        })

broad_impact_df = pd.DataFrame(broad_rows)

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
post_df.to_csv(post_pair_metrics_csv, index=False)
delta.to_csv(pre_post_delta_csv, index=False)
summary_df.to_csv(summary_by_tier_csv, index=False)
blockers_df.to_csv(blockers_csv, index=False)
broad_impact_df.to_csv(broad_impact_csv, index=False)

if bool(CFG.get("cell14_3_store_curves", True)):
    np.savez_compressed(curves_npz, **curve_store)

contract = {
    "cell": "14.3",
    "version": CELL143_VERSION,
    "role": "coupling_conditioned_q4_re_evaluation",
    "quality_dimension": "Q4_cross_modal_consistency_repair_evaluation",
    "upstream_contract_versions": {
        "cell13_4": version134_143,
        "cell13_6": version136_143,
        "cell14_1": version141_143,
        "cell14_2": version142_143
    },
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_143,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_143),
        "pair_pass_rate": broad_q4_pair_pass_rate_143,
        "publication_blocker_record_n": int(broad_q4_publication_blocker_record_n_143),
        "not_recomputed_here": True
    },
    "pre_manifest_q4_status_carried_forward": {
        "pairs_total": int(pre_manifest_pairs_total_143),
        "publication_blocker_n": int(pre_manifest_publication_blocker_n_143),
        "manifest_pair_pass_rate": pre_manifest_pair_pass_rate_143
    },
    "repair_candidate_publication_status": repair_publication_status_143,
    "repair_candidate_publication_reasons": repair_publication_reasons_143,
    "pre_coupling_summary_13_6": pre_summary,
    "post_coupling_summary_14_3": post_summary,
    "repair_candidate_publication_status": repair_publication_status_143,
    "repair_candidate_publication_reasons": repair_publication_reasons_143,
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "delta_summary": delta_summary,
    "summary_by_tier": summary_df.to_dict("records"),
    "broad_q4_impact": {
        "broad_pair_registry_rows": int(len(broad_pairs)),
        "broad_metrics_recomputed_here": False,
        "broad_q4_status_not_overridden": True,
        "broad_pairs_using_coupled_target_cols": int(len(broad_affected)),
        "coupled_target_protocol_cols": target_cols,
        "interpretation": (
            "Broad 13.1-13.4 metrics are affected only if their pair registry includes the "
            "Zigbee target protocol columns changed by Cell 14.1."
        ),
    },
    "strict_contract": {
        "TEST_real_values_used_for_QA_reference": True,
        "repair_candidate_not_final_claim": True,
        "broad_q4_status_carried_forward": True,
        "broad_metrics_recomputed_here": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "QA_done_here": True,
    },
    "outputs": {
        "post_pair_metrics_csv": post_pair_metrics_csv,
        "pre_post_delta_csv": pre_post_delta_csv,
        "summary_by_tier_csv": summary_by_tier_csv,
        "blockers_csv": blockers_csv,
        "broad_impact_csv": broad_impact_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "curves_npz": curves_npz if bool(CFG.get("cell14_3_store_curves", True)) else "",
        "manifest_json": manifest_json,
    },
}

_write_json_143(contract_json, contract)
_write_json_143(contract_canonical_json, contract)

manifest = {
    "cell": "14.3",
    "version": CELL143_VERSION,
    "created_outputs": contract["outputs"],
    "pre_coupling_summary_13_6": pre_summary,
    "post_coupling_summary_14_3": post_summary,
    "repair_candidate_publication_status": repair_publication_status_143,
    "repair_candidate_publication_reasons": repair_publication_reasons_143,
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "delta_summary": delta_summary,
    "broad_q4_impact": contract["broad_q4_impact"],
    "strict_contract": contract["strict_contract"],
}

_write_json_143(manifest_json, manifest)

hashes = {
    "post_pair_metrics_csv_sha256": _sha256_file_143(post_pair_metrics_csv),
    "pre_post_delta_csv_sha256": _sha256_file_143(pre_post_delta_csv),
    "summary_by_tier_csv_sha256": _sha256_file_143(summary_by_tier_csv),
    "blockers_csv_sha256": _sha256_file_143(blockers_csv),
    "broad_impact_csv_sha256": _sha256_file_143(broad_impact_csv),
    "contract_json_sha256": _sha256_file_143(contract_json),
    "contract_canonical_json_sha256": _sha256_file_143(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_143(manifest_json),
}

if bool(CFG.get("cell14_3_store_curves", True)) and os.path.exists(curves_npz):
    hashes["curves_npz_sha256"] = _sha256_file_143(curves_npz)

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_143(contract_json, contract)
_write_json_143(contract_canonical_json, contract)
_write_json_143(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals for 14.4
# ----------------------------------------------------------
globals()["CELL143_VERSION"] = CELL143_VERSION
globals()["CELL14_3_MANIFEST_Q4_POST_PAIR_METRICS_DF"] = post_df
globals()["CELL14_3_MANIFEST_Q4_PRE_POST_DELTA_DF"] = delta
globals()["CELL14_3_MANIFEST_Q4_SUMMARY_BY_TIER_DF"] = summary_df
globals()["CELL14_3_MANIFEST_Q4_BLOCKERS_DF"] = blockers_df
globals()["CELL14_3_BROAD_Q4_IMPACT_AUDIT_DF"] = broad_impact_df
globals()["CELL14_3_Q4_REPAIR_CONTRACT"] = contract
globals()["CELL14_3_MANIFEST_Q4_POST_PAIR_METRICS_CSV"] = post_pair_metrics_csv
globals()["CELL14_3_MANIFEST_Q4_PRE_POST_DELTA_CSV"] = pre_post_delta_csv
globals()["CELL14_3_MANIFEST_Q4_SUMMARY_BY_TIER_CSV"] = summary_by_tier_csv
globals()["CELL14_3_MANIFEST_Q4_BLOCKERS_CSV"] = blockers_csv
globals()["CELL14_3_BROAD_Q4_IMPACT_AUDIT_CSV"] = broad_impact_csv
globals()["CELL14_3_Q4_REPAIR_CONTRACT_JSON"] = contract_json
globals()["CELL14_3_Q4_REPAIR_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL14_3_MANIFEST_Q4_CURVES_NPZ"] = curves_npz
globals()["CELL14_3_Q4_REPAIR_MANIFEST_JSON"] = manifest_json

log(
    "[Cell14.3] Coupling-conditioned Q4 re-evaluation complete | "
    f"pre_pass_rate={pre_summary['pass_rate']:.6f} | "
    f"post_pass_rate={post_summary['pass_rate']:.6f} | "
    f"pre_blockers={pre_summary['publication_blocker_n']} | "
    f"post_blockers={post_summary['publication_blocker_n']} | "
    f"delta_ETA={delta_summary['delta_mean_ETA_similarity']:.6f} | "
    f"delta_lag_peak_error={delta_summary['delta_mean_lag_peak_error']:.6f} | "
    f"delta_profile_similarity={delta_summary['delta_mean_manifest_profile_similarity']:.6f}"
)
log(f"[Cell14.3] Post-coupling manifest Q4 status counts | {status_counts}")
log(f"[Cell14.3] Post-coupling manifest Q4 summary by tier | {summary_df.to_dict('records')}")
log(
    "[Cell14.3] Broad Q4 impact audit | "
    f"broad_pairs_total={len(broad_pairs)} | "
    f"broad_pairs_using_coupled_targets={len(broad_affected)} | "
    f"target_cols={target_cols}"
)
log(f"[Cell14.3] Saved post metrics: {post_pair_metrics_csv} | rows={len(post_df)}")
log(f"[Cell14.3] Saved pre/post delta: {pre_post_delta_csv} | rows={len(delta)}")
log(f"[Cell14.3] Saved broad impact audit: {broad_impact_csv}")
log(f"[Cell14.3] Saved contract: {contract_json}")
log(f"[Cell14.3] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell14.3] Repair candidate status | "
    f"status={repair_publication_status_143} | "
    f"reasons={repair_publication_reasons_143} | "
    f"broad_q4_status_carried_forward={broad_q4_status_143}"
)
log(
    "[Cell14.3] Contract flags | "
    "TEST_real_values_used_for_QA_reference=True | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "QA_done_here=True | repair_candidate_not_final_claim=True | broad_metrics_recomputed_here=False"
)
log("--- END: Cell 14.3 - Coupling-conditioned Q4 re-evaluation (v1.1 repair-candidate/status-aware strict) ---")

gc.collect()