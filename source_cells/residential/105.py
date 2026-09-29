# ==========================================================
# CELL 13.6 - Manifest-informed Zigbee Q4 QA
# v1.1 STUDY-THESIS strict manifest-subset coupling QA, broad-Q4-aware
#
# Role:
#   - Evaluate only replicated manifest-informed driver↔Zigbee pairs
#     from Cell 13.5.
#   - Compare real TEST coupling vs synthetic TEST coupling using:
#       ETA_similarity
#       lag_peak_error
#       response_window_rate_error
#       manifest_window_mean_error
#       manifest_window_profile_similarity
#
# Scientific stance:
#   - This is not broad all-pair Q4.
#   - This asks: does current synthetic data preserve the replicated
#     TRAIN/VAL Zigbee coupling manifest?
#
# Strict rules:
#   - TEST real values are used for QA only.
#   - Do NOT mutate synthetic values.
#   - Do NOT fit generators.
#   - Do NOT select generators.
#   - Do NOT materialize new synthetic values.
#
# Outputs:
#   reports/cell13_6_manifest_q4_pair_metrics.csv
#   reports/cell13_6_manifest_q4_summary_by_tier.csv
#   reports/cell13_6_manifest_q4_blockers.csv
#   reports/cell13_6_manifest_q4_contract.json
#   artifacts/cell13_6_manifest_q4_curves.npz
#   artifacts/cell13_6_manifest_q4_manifest.json
# ==========================================================

log("--- START: Cell 13.6 - Manifest-informed Zigbee Q4 QA (v1.1 broad-Q4-aware strict) ---")

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
_required_136 = [
    "CFG", "log",
    "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_FULL_SYN_TEST_130",
    "IOT_REAL_TEST_REF_130",
    "PROTOCOL_SYN_TEST_130",
    "PROTOCOL_REAL_TEST_REF_130",
    "CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF",
    "CELL13_5_MANIFEST_Q4_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
]
_missing_136 = [k for k in _required_136 if k not in globals()]
if _missing_136:
    raise RuntimeError(f"[Cell13.6] Missing required globals from 13.0/13.5: {_missing_136}")

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

CELL136_VERSION = "cell13_6_manifest_informed_zigbee_q4_qa_v1_1_broad_q4_aware"

CFG["cell13_6_version"] = CELL136_VERSION
CFG["cell13_6_QA_only"] = True
CFG["cell13_6_TEST_real_values_used_for_QA_reference"] = True
CFG["cell13_6_synthetic_values_mutated"] = False
CFG["cell13_6_selection_done_here"] = False
CFG["cell13_6_generator_fit_done_here"] = False
CFG["cell13_6_materialization_done_here"] = False
CFG["cell13_6_broad_q4_status_carried_forward"] = True
CFG["cell13_6_manifest_subset_does_not_override_broad_q4"] = True

CFG.setdefault("cell13_6_eta_pre_seconds", 10)
CFG.setdefault("cell13_6_eta_post_seconds", 30)
CFG.setdefault("cell13_6_profile_extra_lag_margin", 5)
CFG.setdefault("cell13_6_min_events_for_pass", 5)
CFG.setdefault("cell13_6_protocol_transform", "log1p_abs_signed")
CFG.setdefault("cell13_6_response_threshold_quantile", 0.90)
CFG.setdefault("cell13_6_eta_similarity_pass", 0.70)
CFG.setdefault("cell13_6_eta_similarity_warning", 0.50)
CFG.setdefault("cell13_6_profile_similarity_pass", 0.70)
CFG.setdefault("cell13_6_profile_similarity_warning", 0.50)
CFG.setdefault("cell13_6_lag_peak_error_pass", 2)
CFG.setdefault("cell13_6_lag_peak_error_warning", 5)
CFG.setdefault("cell13_6_response_window_rate_error_pass", 0.10)
CFG.setdefault("cell13_6_response_window_rate_error_warning", 0.25)
CFG.setdefault("cell13_6_store_curves", True)
CFG.setdefault("cell13_6_norm_eps", 1e-9)

ETA_PRE_136 = int(CFG.get("cell13_6_eta_pre_seconds", 10))
ETA_POST_136 = int(CFG.get("cell13_6_eta_post_seconds", 30))
EXTRA_MARGIN_136 = int(CFG.get("cell13_6_profile_extra_lag_margin", 5))

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
pair_metrics_csv = os.path.join(REPORT_DIR, "cell13_6_manifest_q4_pair_metrics.csv")
summary_by_tier_csv = os.path.join(REPORT_DIR, "cell13_6_manifest_q4_summary_by_tier.csv")
blockers_csv = os.path.join(REPORT_DIR, "cell13_6_manifest_q4_blockers.csv")
contract_json = os.path.join(REPORT_DIR, "cell13_6_manifest_q4_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell13_6_manifest_q4_contract_v1_1_THESIS.json")
curves_npz = os.path.join(ARTDIR, "cell13_6_manifest_q4_curves.npz")
manifest_json = os.path.join(ARTDIR, "cell13_6_manifest_q4_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_136(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_136(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_136(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_136(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_136(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_136(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_136(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_136(obj.to_dict())
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

def _write_json_136(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_136(payload), f, indent=2, sort_keys=True)

def _sha256_file_136(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_136(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_136(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _transform_protocol_136(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell13_6_protocol_transform", "log1p_abs_signed")).strip().lower()

    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.log1p(np.maximum(arr, 0.0))
    elif mode == "log1p_abs":
        out = np.log1p(np.abs(arr))
    elif mode == "log1p_abs_signed":
        out = np.sign(arr) * np.log1p(np.abs(arr))
    else:
        raise RuntimeError(f"[Cell13.6] Unknown protocol transform: {mode}")

    out[~np.isfinite(out)] = np.nan
    return out

def _event_starts_136(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    active = np.isfinite(arr) & (arr > 0.5)
    if active.size == 0:
        return np.asarray([], dtype=np.int64)
    prev = np.r_[False, active[:-1]]
    starts = active & (~prev)
    return np.flatnonzero(starts).astype(np.int64)

def _edge_safe_events_136(idx: np.ndarray, n: int, lo: int, hi: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx
    keep = (idx + lo >= 0) & (idx + hi < n)
    return idx[keep]

def _profile_around_events_136(y_raw: np.ndarray, event_idx: np.ndarray, lag_lo: int, lag_hi: int) -> tuple:
    y = _transform_protocol_136(y_raw)
    lags = np.arange(lag_lo, lag_hi + 1, dtype=np.int64)

    event_idx = _edge_safe_events_136(event_idx, len(y), lag_lo, lag_hi)
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

    # Baseline-correct by pre-event region if available; else by first lag.
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

def _similarity_136(real_prof: np.ndarray, syn_prof: np.ndarray) -> dict:
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
    scale = float(np.nanmean(np.abs(rr)) + np.nanstd(rr) + float(CFG.get("cell13_6_norm_eps", 1e-9)))
    norm_mae = float(mae / max(scale, float(CFG.get("cell13_6_norm_eps", 1e-9))))

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

def _peak_lag_136(lags: np.ndarray, prof: np.ndarray):
    prof = np.asarray(prof, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.int64)
    valid = np.isfinite(prof)
    if not valid.any():
        return np.nan, np.nan
    idx = np.flatnonzero(valid)[int(np.nanargmax(np.abs(prof[valid])))]
    return int(lags[idx]), float(prof[idx])

def _response_rate_136(y_raw: np.ndarray, event_idx: np.ndarray, lag_lo: int, lag_hi: int, threshold: float) -> float:
    y = _transform_protocol_136(y_raw)
    event_idx = _edge_safe_events_136(event_idx, len(y), lag_lo, lag_hi)
    if event_idx.size == 0 or not np.isfinite(threshold):
        return np.nan

    hits = []
    for e in event_idx:
        seg = y[e + lag_lo : e + lag_hi + 1]
        seg = seg[np.isfinite(seg)]
        hits.append(bool(seg.size and np.nanmax(np.abs(seg)) >= threshold))

    return float(np.mean(hits)) if hits else np.nan

def _response_threshold_from_real_136(y_raw: np.ndarray) -> float:
    y = _transform_protocol_136(y_raw)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.nan
    q = float(CFG.get("cell13_6_response_threshold_quantile", 0.90))
    q = float(np.clip(q, 0.50, 0.999))
    return float(np.nanquantile(np.abs(y), q))

def _status_136(row: dict) -> tuple:
    real_events = int(_safe_float_136(row.get("real_event_count_window_valid"), 0))
    syn_events = int(_safe_float_136(row.get("syn_event_count_window_valid"), 0))
    min_events = int(CFG.get("cell13_6_min_events_for_pass", 5))

    eta_sim = _safe_float_136(row.get("ETA_similarity"), np.nan)
    profile_sim = _safe_float_136(row.get("manifest_window_profile_similarity"), np.nan)
    lag_peak_error = _safe_float_136(row.get("lag_peak_error"), np.nan)
    resp_err = _safe_float_136(row.get("response_window_rate_error"), np.nan)

    if real_events < min_events and syn_events < min_events:
        return "not_evaluable", "insufficient_real_and_synthetic_events"

    if real_events < min_events:
        return "warning", "insufficient_real_events"

    if syn_events < min_events:
        return "warning", "insufficient_synthetic_events"

    fatal = []
    warn = []

    if np.isfinite(eta_sim):
        if eta_sim < float(CFG.get("cell13_6_eta_similarity_warning", 0.50)):
            fatal.append("ETA_similarity_fatal")
        elif eta_sim < float(CFG.get("cell13_6_eta_similarity_pass", 0.70)):
            warn.append("ETA_similarity_warning")
    else:
        warn.append("ETA_similarity_not_finite")

    if np.isfinite(profile_sim):
        if profile_sim < float(CFG.get("cell13_6_profile_similarity_warning", 0.50)):
            fatal.append("manifest_profile_similarity_fatal")
        elif profile_sim < float(CFG.get("cell13_6_profile_similarity_pass", 0.70)):
            warn.append("manifest_profile_similarity_warning")
    else:
        warn.append("manifest_profile_similarity_not_finite")

    if np.isfinite(lag_peak_error):
        if lag_peak_error > float(CFG.get("cell13_6_lag_peak_error_warning", 5)):
            fatal.append("lag_peak_error_fatal")
        elif lag_peak_error > float(CFG.get("cell13_6_lag_peak_error_pass", 2)):
            warn.append("lag_peak_error_warning")

    if np.isfinite(resp_err):
        if resp_err > float(CFG.get("cell13_6_response_window_rate_error_warning", 0.25)):
            fatal.append("response_window_rate_error_fatal")
        elif resp_err > float(CFG.get("cell13_6_response_window_rate_error_pass", 0.10)):
            warn.append("response_window_rate_error_warning")

    if fatal:
        return "fatal", "|".join(fatal)
    if warn:
        return "warning", "|".join(warn)
    return "pass", "manifest_q4_pair_pass"

# ----------------------------------------------------------
# 3) Validate upstream manifest registry and broad-Q4 summary contracts
# ----------------------------------------------------------
contract135_136 = CELL13_5_MANIFEST_Q4_CONTRACT
if not isinstance(contract135_136, dict):
    raise RuntimeError("[Cell13.6] CELL13_5_MANIFEST_Q4_CONTRACT is not a dict.")

version135_136 = str(contract135_136.get("version", ""))
if "cell13_5_manifest_informed_zigbee_q4_registry_v1_1" not in version135_136:
    raise RuntimeError(
        "[Cell13.6] Unexpected Cell 13.5 contract version. "
        f"Expected v1.1 broad-Q4-aware registry, got: {version135_136}"
    )

strict135_136 = contract135_136.get("strict_contract", {})
if bool(strict135_136.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell13.6] Cell 13.5 contract indicates synthetic values were mutated.")
if bool(strict135_136.get("selection_done_here", True)):
    raise RuntimeError("[Cell13.6] Cell 13.5 contract indicates selection was done.")
if bool(strict135_136.get("generator_fit_done_here", True)):
    raise RuntimeError("[Cell13.6] Cell 13.5 contract indicates generator fitting was done.")
if bool(strict135_136.get("materialization_done_here", True)):
    raise RuntimeError("[Cell13.6] Cell 13.5 contract indicates materialization was done.")

broad_from_135_136 = contract135_136.get("broad_q4_status_carried_forward", {})
if not isinstance(broad_from_135_136, dict):
    broad_from_135_136 = {}

q4_summary_136 = CELL13_4_Q4_PUBLICATION_SUMMARY
if not isinstance(q4_summary_136, dict):
    raise RuntimeError("[Cell13.6] CELL13_4_Q4_PUBLICATION_SUMMARY is not a dict.")

version134_136 = str(q4_summary_136.get("version", ""))
if "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1" not in version134_136:
    raise RuntimeError(
        "[Cell13.6] Unexpected Cell 13.4 summary version. "
        f"Expected v1.1 contract-hardened Q4 summary, got: {version134_136}"
    )

broad_q4_status_136 = str(q4_summary_136.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_136 = int(q4_summary_136.get("pair_summary", {}).get("pair_blocker_n", 0) or 0)
broad_q4_pair_pass_rate_136 = _safe_float_136(
    q4_summary_136.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)
broad_q4_publication_blocker_record_n_136 = int(
    q4_summary_136.get("publication_blocker_record_n", q4_summary_136.get("publication_blocker_n", 0)) or 0
)

if broad_q4_status_136 != "blocker":
    log(
        "[Cell13.6] WARNING: broad Q4 summary is not blocker. "
        f"status={broad_q4_status_136}"
    )

# ----------------------------------------------------------
# 4) Validate inputs
# ----------------------------------------------------------
pairs = CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF.copy()

if len(pairs) == 0 and bool(CFG.get("cell13_6_allow_empty_manifest_q4_subset", True)):
    log("[Cell13.6] No manifest-informed Q4 pairs available. Writing no-claim empty manifest-subset QA outputs.")
    metrics_df = pd.DataFrame(columns=[
        "pair_id", "manifest_tier", "anchor_name", "driver_col", "protocol_col",
        "protocol_tier", "manifest_q4_status", "manifest_q4_publication_blocker"
    ])
    summary_df = pd.DataFrame(columns=[
        "protocol_tier", "manifest_tier", "pairs_total", "pairs_evaluable", "pass_n",
        "warning_n", "fatal_n", "not_evaluable_n", "manifest_pair_pass_rate",
        "publication_blocker_n"
    ])
    blockers_df = metrics_df.copy()

    metrics_df.to_csv(pair_metrics_csv, index=False)
    summary_df.to_csv(summary_by_tier_csv, index=False)
    blockers_df.to_csv(blockers_csv, index=False)

    overall_summary = {
        "pairs_total": 0,
        "pairs_evaluable": 0,
        "status_counts": {},
        "publication_blocker_n": 0,
        "manifest_pair_pass_rate": None,
        "manifest_q4_status": "not_evaluated_no_manifest_pairs",
    }

    contract = {
        "cell": "13.6",
        "version": CELL136_VERSION,
        "role": "manifest_informed_zigbee_q4_qa_no_claim_empty_registry",
        "quality_dimension": "Q4_cross_modal_consistency_manifest_subset",
        "empty_manifest_registry": True,
        "reason": "Cell 13.5 produced zero manifest-informed pairs; no manifest-subset Q4 claim is made.",
        "overall_summary": overall_summary,
        "strict_contract": {
            "TEST_real_values_used_for_QA_reference": False,
            "synthetic_values_mutated": False,
            "selection_done_here": False,
            "generator_fit_done_here": False,
            "materialization_done_here": False,
            "manifest_subset_does_not_override_broad_q4": True,
        },
        "outputs": {
            "pair_metrics_csv": pair_metrics_csv,
            "summary_by_tier_csv": summary_by_tier_csv,
            "blockers_csv": blockers_csv,
            "contract_json": contract_json,
            "contract_canonical_json": contract_canonical_json,
            "manifest_json": manifest_json,
        },
    }

    _write_json_136(contract_json, contract)
    _write_json_136(contract_canonical_json, contract)
    manifest = {
        "cell": "13.6",
        "version": CELL136_VERSION,
        "created_outputs": contract["outputs"],
        "empty_manifest_registry": True,
        "overall_summary": overall_summary,
        "strict_contract": contract["strict_contract"],
    }
    _write_json_136(manifest_json, manifest)

    CELL13_6_MANIFEST_Q4_PAIR_METRICS_DF = metrics_df
    CELL13_6_MANIFEST_Q4_SUMMARY_DF = summary_df
    CELL13_6_MANIFEST_Q4_BLOCKERS_DF = blockers_df
    CELL13_6_MANIFEST_Q4_CONTRACT = contract
    CELL13_6_MANIFEST_Q4_MANIFEST = manifest
    globals()["CELL13_6_MANIFEST_Q4_PAIR_METRICS_DF"] = metrics_df
    globals()["CELL13_6_MANIFEST_Q4_SUMMARY_DF"] = summary_df
    globals()["CELL13_6_MANIFEST_Q4_BLOCKERS_DF"] = blockers_df
    globals()["CELL13_6_MANIFEST_Q4_CONTRACT"] = contract
    globals()["CELL13_6_MANIFEST_Q4_MANIFEST"] = manifest

    log("[Cell13.6] Empty manifest subset handled as no-claim evidence; broad Q4 status carried separately.")
    log("--- END: Cell 13.6 - Manifest-informed Zigbee Q4 QA (empty no-claim mode) ---")
    gc.collect()
else:

    if len(pairs) == 0:
        raise RuntimeError("[Cell13.6] Empty manifest-informed pair registry from Cell 13.5.")

    for frame_name, frame in [
        ("IOT_FULL_SYN_TEST_130", IOT_FULL_SYN_TEST_130),
        ("IOT_REAL_TEST_REF_130", IOT_REAL_TEST_REF_130),
        ("PROTOCOL_SYN_TEST_130", PROTOCOL_SYN_TEST_130),
        ("PROTOCOL_REAL_TEST_REF_130", PROTOCOL_REAL_TEST_REF_130),
    ]:
        if len(frame) != N_TE:
            raise RuntimeError(f"[Cell13.6] {frame_name} row mismatch: got={len(frame)} expected={N_TE}")
        frame.columns = frame.columns.astype(str)

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
        raise RuntimeError(f"[Cell13.6] Manifest pair registry missing required columns: {missing_pair_cols}")

    pairs = pairs[pairs.get("eligible_for_13_6_manifest_q4", True).astype(bool)].copy()

    if len(pairs) == 0:
        raise RuntimeError("[Cell13.6] No eligible manifest-informed Q4 pairs.")

    # ----------------------------------------------------------
    # 5) Compute manifest-informed Q4 metrics
    # ----------------------------------------------------------
    metric_rows = []
    curve_store = {}

    log(
        "[Cell13.6] Computing manifest-informed Zigbee Q4 QA | "
        f"pairs={len(pairs)}"
    )

    for i, r in enumerate(pairs.itertuples(index=False), start=1):
        if i == 1 or i % 10 == 0 or i == len(pairs):
            log(f"[Cell13.6] progress {i}/{len(pairs)}")

        row = r._asdict()
        pair_id = str(row["pair_id"])
        driver_col = str(row["driver_col"])
        protocol_col = str(row["protocol_col"])
        manifest_tier = str(row["manifest_tier"])
        protocol_tier = str(row["protocol_tier"])

        lag_lo = int(row["lag_lo"])
        lag_hi = int(row["lag_hi"])
        if lag_lo > lag_hi:
            lag_lo, lag_hi = lag_hi, lag_lo

        # ETA profile over wider window.
        eta_lo = min(-ETA_PRE_136, lag_lo - EXTRA_MARGIN_136)
        eta_hi = max(ETA_POST_136, lag_hi + EXTRA_MARGIN_136)

        # Manifest-window focused profile.
        profile_lo = lag_lo
        profile_hi = lag_hi

        d_real = _to_num_array_136(IOT_REAL_TEST_REF_130, driver_col)
        d_syn = _to_num_array_136(IOT_FULL_SYN_TEST_130, driver_col)
        p_real = _to_num_array_136(PROTOCOL_REAL_TEST_REF_130, protocol_col)
        p_syn = _to_num_array_136(PROTOCOL_SYN_TEST_130, protocol_col)

        e_real = _event_starts_136(d_real)
        e_syn = _event_starts_136(d_syn)

        eta_lags_real, eta_real, eta_real_meta = _profile_around_events_136(p_real, e_real, eta_lo, eta_hi)
        eta_lags_syn, eta_syn, eta_syn_meta = _profile_around_events_136(p_syn, e_syn, eta_lo, eta_hi)

        if not np.array_equal(eta_lags_real, eta_lags_syn):
            raise RuntimeError(f"[Cell13.6] Internal lag mismatch for pair {pair_id}")

        eta_sim = _similarity_136(eta_real, eta_syn)

        man_lags_real, man_real, man_real_meta = _profile_around_events_136(p_real, e_real, profile_lo, profile_hi)
        man_lags_syn, man_syn, man_syn_meta = _profile_around_events_136(p_syn, e_syn, profile_lo, profile_hi)

        if not np.array_equal(man_lags_real, man_lags_syn):
            raise RuntimeError(f"[Cell13.6] Internal manifest lag mismatch for pair {pair_id}")

        man_sim = _similarity_136(man_real, man_syn)

        real_peak_lag, real_peak_value = _peak_lag_136(eta_lags_real, eta_real)
        syn_peak_lag, syn_peak_value = _peak_lag_136(eta_lags_syn, eta_syn)
        lag_peak_error = (
            abs(float(real_peak_lag) - float(syn_peak_lag))
            if np.isfinite(real_peak_lag) and np.isfinite(syn_peak_lag)
            else np.nan
        )

        real_thr = _response_threshold_from_real_136(p_real)
        real_resp_rate = _response_rate_136(p_real, e_real, profile_lo, profile_hi, real_thr)
        syn_resp_rate = _response_rate_136(p_syn, e_syn, profile_lo, profile_hi, real_thr)

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
            "anchor_name": str(row.get("anchor_name", "")),
            "driver_col": driver_col,
            "protocol_col": protocol_col,
            "protocol_tier": protocol_tier,
            "lag_lo": int(lag_lo),
            "lag_hi": int(lag_hi),
            "recommended_weight": _safe_float_136(row.get("recommended_weight"), np.nan),
            "replication_score": _safe_float_136(row.get("replication_score"), np.nan),
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

        status, reason = _status_136(metric_row)
        metric_row["manifest_q4_status"] = status
        metric_row["manifest_q4_reasons"] = reason
        metric_row["manifest_q4_publication_blocker"] = bool(status == "fatal")

        metric_rows.append(metric_row)

        if bool(CFG.get("cell13_6_store_curves", True)):
            curve_store[f"{pair_id}__eta_lags"] = eta_lags_real.astype(np.int64)
            curve_store[f"{pair_id}__eta_real"] = eta_real.astype(np.float32)
            curve_store[f"{pair_id}__eta_syn"] = eta_syn.astype(np.float32)
            curve_store[f"{pair_id}__manifest_lags"] = man_lags_real.astype(np.int64)
            curve_store[f"{pair_id}__manifest_real"] = man_real.astype(np.float32)
            curve_store[f"{pair_id}__manifest_syn"] = man_syn.astype(np.float32)

    metrics_df = pd.DataFrame(metric_rows)

    if len(metrics_df) != len(pairs):
        raise RuntimeError(f"[Cell13.6] Metrics row mismatch: got={len(metrics_df)} expected={len(pairs)}")

    # ----------------------------------------------------------
    # 6) Summaries and blockers
    # ----------------------------------------------------------
    summary_rows = []

    for (tier, mtier), sub in metrics_df.groupby(["protocol_tier", "manifest_tier"], dropna=False):
        sub = sub.copy()
        evaluable = sub[~sub["manifest_q4_status"].astype(str).eq("not_evaluable")]

        summary_rows.append({
            "protocol_tier": tier,
            "manifest_tier": mtier,
            "pairs_total": int(len(sub)),
            "pairs_evaluable": int(len(evaluable)),
            "pass_n": int((sub["manifest_q4_status"].astype(str) == "pass").sum()),
            "warning_n": int((sub["manifest_q4_status"].astype(str) == "warning").sum()),
            "fatal_n": int((sub["manifest_q4_status"].astype(str) == "fatal").sum()),
            "not_evaluable_n": int((sub["manifest_q4_status"].astype(str) == "not_evaluable").sum()),
            "manifest_pair_pass_rate": float((evaluable["manifest_q4_status"].astype(str) == "pass").mean()) if len(evaluable) else np.nan,
            "mean_ETA_similarity": float(pd.to_numeric(evaluable["ETA_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
            "mean_lag_peak_error": float(pd.to_numeric(evaluable["lag_peak_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
            "mean_response_window_rate_error": float(pd.to_numeric(evaluable["response_window_rate_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
            "mean_manifest_profile_similarity": float(pd.to_numeric(evaluable["manifest_window_profile_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
            "publication_blocker_n": int(sub["manifest_q4_publication_blocker"].fillna(False).astype(bool).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)

    blockers_df = metrics_df[metrics_df["manifest_q4_publication_blocker"].fillna(False).astype(bool)].copy()

    status_counts = metrics_df["manifest_q4_status"].astype(str).value_counts().sort_index().to_dict()
    evaluable = metrics_df[~metrics_df["manifest_q4_status"].astype(str).eq("not_evaluable")].copy()

    overall_summary = {
        "pairs_total": int(len(metrics_df)),
        "pairs_evaluable": int(len(evaluable)),
        "status_counts": status_counts,
        "publication_blocker_n": int(len(blockers_df)),
        "manifest_pair_pass_rate": float((evaluable["manifest_q4_status"].astype(str) == "pass").mean()) if len(evaluable) else np.nan,
        "mean_ETA_similarity": float(pd.to_numeric(evaluable["ETA_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_lag_peak_error": float(pd.to_numeric(evaluable["lag_peak_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_response_window_rate_error": float(pd.to_numeric(evaluable["response_window_rate_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_manifest_profile_similarity": float(pd.to_numeric(evaluable["manifest_window_profile_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_manifest_window_mean_error": float(pd.to_numeric(evaluable["manifest_window_mean_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
    }

    # ----------------------------------------------------------
    # 7) Save outputs
    # ----------------------------------------------------------
    metrics_df.to_csv(pair_metrics_csv, index=False)
    summary_df.to_csv(summary_by_tier_csv, index=False)
    blockers_df.to_csv(blockers_csv, index=False)

    if bool(CFG.get("cell13_6_store_curves", True)):
        np.savez_compressed(curves_npz, **curve_store)

    contract = {
        "cell": "13.6",
        "version": CELL136_VERSION,
        "role": "manifest_informed_zigbee_q4_qa",
        "quality_dimension": "Q4_cross_modal_consistency_manifest_subset",
        "cell13_5_contract_version_seen": version135_136,
        "cell13_4_summary_version_seen": version134_136,
        "broad_q4_status_carried_forward": {
            "overall_q4_status": broad_q4_status_136,
            "pair_blocker_n": int(broad_q4_pair_blocker_n_136),
            "pair_pass_rate": broad_q4_pair_pass_rate_136,
            "publication_blocker_record_n": int(broad_q4_publication_blocker_record_n_136),
            "does_not_repair_or_override_broad_q4": True,
            "interpretation": "This manifest subset QA is a narrow follow-up evaluation and does not replace the broad 624-pair Q4 blocker."
        },
        "overall_summary": overall_summary,
        "summary_by_tier": summary_df.to_dict("records"),
        "thresholds": {
            "ETA_similarity_pass": float(CFG.get("cell13_6_eta_similarity_pass", 0.70)),
            "ETA_similarity_warning": float(CFG.get("cell13_6_eta_similarity_warning", 0.50)),
            "profile_similarity_pass": float(CFG.get("cell13_6_profile_similarity_pass", 0.70)),
            "profile_similarity_warning": float(CFG.get("cell13_6_profile_similarity_warning", 0.50)),
            "lag_peak_error_pass": float(CFG.get("cell13_6_lag_peak_error_pass", 2)),
            "lag_peak_error_warning": float(CFG.get("cell13_6_lag_peak_error_warning", 5)),
            "response_window_rate_error_pass": float(CFG.get("cell13_6_response_window_rate_error_pass", 0.10)),
            "response_window_rate_error_warning": float(CFG.get("cell13_6_response_window_rate_error_warning", 0.25)),
        },
        "interpretation": {
            "purpose": (
                "Evaluates whether the current synthetic TEST data preserves the replicated TRAIN/VAL "
                "Zigbee coupling manifest, independent of broad all-pair Q4 results."
            ),
            "if_blocked": (
                "A blocker here means the real data contains replicated Zigbee coupling evidence, "
                "but the current synthetic protocol/IoT output does not preserve those manifest-defined pairs."
            ),
            "if_pass_or_warning": (
                "A pass/warning here supports only a narrow manifest-defined Zigbee subset claim; "
                "it does not repair or override the broad 624-pair Q4 blocker from Cell 13.4."
            ),
        },
        "strict_contract": {
            "TEST_real_values_used_for_QA_reference": True,
            "broad_q4_status_carried_forward": True,
            "manifest_subset_does_not_override_broad_q4": True,
            "synthetic_values_mutated": False,
            "selection_done_here": False,
            "generator_fit_done_here": False,
            "materialization_done_here": False,
        },
        "outputs": {
            "pair_metrics_csv": pair_metrics_csv,
            "summary_by_tier_csv": summary_by_tier_csv,
            "blockers_csv": blockers_csv,
            "contract_json": contract_json,
            "contract_canonical_json": contract_canonical_json,
            "curves_npz": curves_npz if bool(CFG.get("cell13_6_store_curves", True)) else "",
            "manifest_json": manifest_json,
        },
    }

    _write_json_136(contract_json, contract)
    _write_json_136(contract_canonical_json, contract)

    manifest = {
        "cell": "13.6",
        "version": CELL136_VERSION,
        "created_outputs": contract["outputs"],
        "overall_summary": overall_summary,
        "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
        "strict_contract": contract["strict_contract"],
    }

    _write_json_136(manifest_json, manifest)

    hashes = {
        "pair_metrics_csv_sha256": _sha256_file_136(pair_metrics_csv),
        "summary_by_tier_csv_sha256": _sha256_file_136(summary_by_tier_csv),
        "blockers_csv_sha256": _sha256_file_136(blockers_csv),
        "contract_json_sha256": _sha256_file_136(contract_json),
        "contract_canonical_json_sha256": _sha256_file_136(contract_canonical_json),
        "manifest_json_sha256": _sha256_file_136(manifest_json),
    }
    if bool(CFG.get("cell13_6_store_curves", True)) and os.path.exists(curves_npz):
        hashes["curves_npz_sha256"] = _sha256_file_136(curves_npz)

    contract["hashes"] = hashes
    manifest["hashes"] = hashes

    _write_json_136(contract_json, contract)
    _write_json_136(contract_canonical_json, contract)
    _write_json_136(manifest_json, manifest)

    # ----------------------------------------------------------
    # 8) Export globals
    # ----------------------------------------------------------
    globals()["CELL136_VERSION"] = CELL136_VERSION
    globals()["CELL13_6_MANIFEST_Q4_PAIR_METRICS_DF"] = metrics_df
    globals()["CELL13_6_MANIFEST_Q4_SUMMARY_BY_TIER_DF"] = summary_df
    globals()["CELL13_6_MANIFEST_Q4_BLOCKERS_DF"] = blockers_df
    globals()["CELL13_6_MANIFEST_Q4_CONTRACT"] = contract
    globals()["CELL13_6_MANIFEST_Q4_PAIR_METRICS_CSV"] = pair_metrics_csv
    globals()["CELL13_6_MANIFEST_Q4_SUMMARY_BY_TIER_CSV"] = summary_by_tier_csv
    globals()["CELL13_6_MANIFEST_Q4_BLOCKERS_CSV"] = blockers_csv
    globals()["CELL13_6_MANIFEST_Q4_CONTRACT_JSON"] = contract_json
    globals()["CELL13_6_MANIFEST_Q4_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
    globals()["CELL13_6_MANIFEST_Q4_CURVES_NPZ"] = curves_npz
    globals()["CELL13_6_MANIFEST_Q4_MANIFEST_JSON"] = manifest_json

    log(
        "[Cell13.6] Manifest-informed Q4 QA complete | "
        f"pairs_total={overall_summary['pairs_total']} | "
        f"pairs_evaluable={overall_summary['pairs_evaluable']} | "
        f"pass_rate={overall_summary['manifest_pair_pass_rate']:.6f} | "
        f"mean_ETA_similarity={overall_summary['mean_ETA_similarity']:.6f} | "
        f"mean_lag_peak_error={overall_summary['mean_lag_peak_error']:.6f} | "
        f"mean_response_window_rate_error={overall_summary['mean_response_window_rate_error']:.6f} | "
        f"mean_manifest_profile_similarity={overall_summary['mean_manifest_profile_similarity']:.6f} | "
        f"publication_blocker_n={overall_summary['publication_blocker_n']}"
    )
    log(f"[Cell13.6] Manifest Q4 status counts | {status_counts}")
    log(f"[Cell13.6] Manifest Q4 summary by tier | {summary_df.to_dict('records')}")
    log(f"[Cell13.6] Saved pair metrics: {pair_metrics_csv} | rows={len(metrics_df)}")
    log(f"[Cell13.6] Saved blockers: {blockers_csv} | rows={len(blockers_df)}")
    log(f"[Cell13.6] Saved contract: {contract_json}")
    log(f"[Cell13.6] Saved canonical contract: {contract_canonical_json}")
    log(
        "[Cell13.6] Broad Q4 status carried forward | "
        f"overall_status={broad_q4_status_136} | "
        f"pair_blocker_n={broad_q4_pair_blocker_n_136} | "
        f"pair_pass_rate={broad_q4_pair_pass_rate_136}"
    )
    log(
        "[Cell13.6] Contract flags | "
        "TEST_real_values_used_for_QA_reference=True | "
        "synthetic_values_mutated=False | "
        "selection_done_here=False | "
        "generator_fit_done_here=False | "
        "materialization_done_here=False | broad_q4_status_carried_forward=True | manifest_subset_does_not_override_broad_q4=True"
    )
    log("--- END: Cell 13.6 - Manifest-informed Zigbee Q4 QA (v1.1 broad-Q4-aware strict) ---")

    gc.collect()
