# ==========================================================
# CELL 13.3 - Q4 cross-correlation preservation by protocol tier
# v1.1 STUDY-THESIS strict cross-correlation QA, contract-hardened
#
# Role:
#   - For each IoT driver↔protocol pair:
#       * compute real TEST cross-correlation over bounded lags
#       * compute synthetic TEST cross-correlation over bounded lags
#       * compare cross-correlation curves
#   - Summarize preservation by protocol tier.
#
# Metrics:
#   cross_corr_mae
#   cross_corr_rmse
#   cross_corr_similarity
#   cross_corr_peak_lag_error
#   cross_corr_peak_value_error
#   cross_corr_pair_pass_rate
#
# Strict rules:
#   - TEST real values are used for QA only.
#   - Do NOT mutate synthetic values.
#   - Do NOT fit generators.
#   - Do NOT select generators.
#   - Do NOT materialize new synthetic values.
#
# Inputs from Cell 13.0:
#   IOT_FULL_SYN_TEST_130
#   IOT_REAL_TEST_REF_130
#   PROTOCOL_SYN_TEST_130
#   PROTOCOL_REAL_TEST_REF_130
#   CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF
#
# Optional inputs:
#   CELL13_1_ETA_PAIR_METRICS_DF
#   CELL13_2_LAG_RESPONSE_PAIR_METRICS_DF
#
# Outputs:
#   reports/cell13_3_cross_corr_pair_metrics.csv
#   reports/cell13_3_cross_corr_tier_summary.csv
#   reports/cell13_3_cross_corr_contract.json
#   artifacts/cell13_3_cross_corr_curves.npz
#   artifacts/cell13_3_cross_corr_manifest.json
# ==========================================================

log("--- START: Cell 13.3 - Q4 cross-correlation preservation QA (v1.1 contract-hardened strict) ---")

import os
import gc
import json
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_133 = [
    "CFG", "log",
    "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_FULL_SYN_TEST_130",
    "IOT_REAL_TEST_REF_130",
    "PROTOCOL_SYN_TEST_130",
    "PROTOCOL_REAL_TEST_REF_130",
    "CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF",
    "CELL130_Q4_COUPLING_INPUT_CONTRACT",
    "CELL13_1_ETA_CONTRACT",
    "CELL13_2_LAG_RESPONSE_CONTRACT",
]
_missing_133 = [k for k in _required_133 if k not in globals()]
if _missing_133:
    raise RuntimeError(f"[Cell13.3] Missing required globals from Cell 13.0: {_missing_133}")

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

CELL133_VERSION = "cell13_3_q4_cross_correlation_preservation_qa_v1_1_contract_hardened"

CFG["cell13_3_version"] = CELL133_VERSION
CFG["cell13_3_QA_only"] = True
CFG["cell13_3_TEST_real_values_used_for_QA_reference"] = True
CFG["cell13_3_synthetic_values_mutated"] = False
CFG["cell13_3_selection_done_here"] = False
CFG["cell13_3_generator_fit_done_here"] = False
CFG["cell13_3_materialization_done_here"] = False
CFG["cell13_3_requires_schema_driven_pair_registry"] = True
CFG["cell13_3_TEST_activity_used_for_pair_selection"] = False

CFG.setdefault("cell13_3_max_lag_seconds", int(CFG.get("cell13_max_lag_seconds", 60)))
CFG.setdefault("cell13_3_protocol_transform", "log1p_abs_signed")
CFG.setdefault("cell13_3_driver_transform", "binary_centered")
CFG.setdefault("cell13_3_min_pair_events_for_pass", 5)
CFG.setdefault("cell13_3_min_finite_overlap_rate", 0.95)
CFG.setdefault("cell13_3_cross_corr_similarity_pass", 0.75)
CFG.setdefault("cell13_3_cross_corr_similarity_warning", 0.50)
CFG.setdefault("cell13_3_cross_corr_mae_pass", 0.10)
CFG.setdefault("cell13_3_cross_corr_mae_warning", 0.25)
CFG.setdefault("cell13_3_peak_lag_error_pass_seconds", 5)
CFG.setdefault("cell13_3_peak_lag_error_warning_seconds", 15)
CFG.setdefault("cell13_3_norm_eps", 1e-9)
CFG.setdefault("cell13_3_store_cross_corr_curves", True)

MAX_LAG_133 = int(CFG.get("cell13_3_max_lag_seconds", 60))
if MAX_LAG_133 < 1:
    raise RuntimeError(f"[Cell13.3] Invalid max lag: {MAX_LAG_133}")

LAGS_133 = np.arange(-MAX_LAG_133, MAX_LAG_133 + 1, dtype=np.int64)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
cross_corr_pair_metrics_csv = os.path.join(REPORT_DIR, "cell13_3_cross_corr_pair_metrics.csv")
cross_corr_tier_summary_csv = os.path.join(REPORT_DIR, "cell13_3_cross_corr_tier_summary.csv")
cross_corr_curve_npz = os.path.join(ARTDIR, "cell13_3_cross_corr_curves.npz")
cross_corr_contract_json = os.path.join(REPORT_DIR, "cell13_3_cross_corr_contract.json")
cross_corr_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell13_3_cross_corr_contract_v1_1_THESIS.json")
cross_corr_manifest_json = os.path.join(ARTDIR, "cell13_3_cross_corr_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_133(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_133(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_133(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_133(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_133(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_133(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_133(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_133(obj.to_dict())
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

def _write_json_133(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_133(payload), f, indent=2, sort_keys=True)

def _sha256_file_133(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_133(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_133(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _transform_driver_133(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell13_3_driver_transform", "binary_centered")).strip().lower()

    out = arr.copy()
    out[~np.isfinite(out)] = np.nan

    if mode == "identity":
        return out

    if mode == "binary":
        finite = np.isfinite(out)
        out[finite] = np.where(out[finite] > 0.5, 1.0, 0.0)
        return out

    if mode == "binary_centered":
        finite = np.isfinite(out)
        out[finite] = np.where(out[finite] > 0.5, 1.0, 0.0)
        if finite.any():
            out[finite] = out[finite] - float(np.nanmean(out[finite]))
        return out

    if mode == "zscore":
        finite = np.isfinite(out)
        if finite.any():
            mu = float(np.nanmean(out[finite]))
            sd = float(np.nanstd(out[finite]))
            out[finite] = (out[finite] - mu) / max(sd, float(CFG.get("cell13_3_norm_eps", 1e-9)))
        return out

    raise RuntimeError(f"[Cell13.3] Unknown driver transform: {mode}")

def _transform_protocol_133(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell13_3_protocol_transform", "log1p_abs_signed")).strip().lower()

    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.log1p(np.maximum(arr, 0.0))
    elif mode == "log1p_abs":
        out = np.log1p(np.abs(arr))
    elif mode == "log1p_abs_signed":
        out = np.sign(arr) * np.log1p(np.abs(arr))
    elif mode == "zscore":
        out = arr.copy()
        finite = np.isfinite(out)
        if finite.any():
            mu = float(np.nanmean(out[finite]))
            sd = float(np.nanstd(out[finite]))
            out[finite] = (out[finite] - mu) / max(sd, float(CFG.get("cell13_3_norm_eps", 1e-9)))
    else:
        raise RuntimeError(f"[Cell13.3] Unknown protocol transform: {mode}")

    out[~np.isfinite(out)] = np.nan
    return out

def _event_count_133(x: np.ndarray) -> int:
    arr = np.asarray(x, dtype=np.float64)
    return int(np.sum(np.isfinite(arr) & (arr > 0.5)))

def _corr_safe_133(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 5:
        return np.nan

    aa = a[finite]
    bb = b[finite]

    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        # If both are effectively constant and equal after centering, correlation is not meaningful.
        return 0.0

    return float(np.corrcoef(aa, bb)[0, 1])

def _cross_corr_curve_133(driver_x: np.ndarray, protocol_y: np.ndarray) -> tuple:
    d = _transform_driver_133(driver_x)
    p = _transform_protocol_133(protocol_y)

    n = min(len(d), len(p))
    d = d[:n]
    p = p[:n]

    finite_overlap_rate = float((np.isfinite(d) & np.isfinite(p)).mean()) if n else np.nan

    corrs = np.full(len(LAGS_133), np.nan, dtype=np.float64)
    overlap_counts = np.zeros(len(LAGS_133), dtype=np.int64)

    for i, lag in enumerate(LAGS_133):
        lag = int(lag)

        if lag < 0:
            # driver leads protocol by abs(lag)? align d[t] with p[t+abs(lag)]
            dd = d[:lag]
            pp = p[-lag:]
        elif lag > 0:
            dd = d[lag:]
            pp = p[:-lag]
        else:
            dd = d
            pp = p

        finite = np.isfinite(dd) & np.isfinite(pp)
        overlap_counts[i] = int(finite.sum())

        if overlap_counts[i] >= 5:
            corrs[i] = _corr_safe_133(dd, pp)

    return corrs, {
        "finite_overlap_rate": finite_overlap_rate,
        "min_overlap_count": int(np.min(overlap_counts)) if overlap_counts.size else 0,
        "mean_overlap_count": float(np.mean(overlap_counts)) if overlap_counts.size else np.nan,
        "curve_valid": bool(np.isfinite(corrs).sum() >= 3),
        "curve_finite_lags": int(np.isfinite(corrs).sum()),
    }

def _cross_corr_similarity_133(real_curve: np.ndarray, syn_curve: np.ndarray) -> dict:
    r = np.asarray(real_curve, dtype=np.float64)
    s = np.asarray(syn_curve, dtype=np.float64)

    valid = np.isfinite(r) & np.isfinite(s)
    if int(valid.sum()) < 3:
        return {
            "cross_corr_similarity": np.nan,
            "cross_corr_mae": np.nan,
            "cross_corr_rmse": np.nan,
            "cross_corr_corr": np.nan,
            "cross_corr_peak_lag_error": np.nan,
            "cross_corr_peak_value_error": np.nan,
            "real_cross_corr_peak_lag": np.nan,
            "syn_cross_corr_peak_lag": np.nan,
            "real_cross_corr_peak_value": np.nan,
            "syn_cross_corr_peak_value": np.nan,
            "cross_corr_compare_valid": False,
            "cross_corr_compare_reason": "insufficient_finite_lags",
        }

    rr = r[valid]
    ss = s[valid]
    lags = LAGS_133[valid]

    mae = float(np.mean(np.abs(rr - ss)))
    rmse = float(np.sqrt(np.mean((rr - ss) ** 2)))

    if np.std(rr) <= 1e-12 or np.std(ss) <= 1e-12:
        corr = 1.0 if np.allclose(rr, ss, atol=1e-9) else 0.0
    else:
        corr = float(np.corrcoef(rr, ss)[0, 1])

    real_peak_i = int(np.nanargmax(np.abs(rr)))
    syn_peak_i = int(np.nanargmax(np.abs(ss)))

    real_peak_lag = int(lags[real_peak_i])
    syn_peak_lag = int(lags[syn_peak_i])
    peak_lag_error = abs(real_peak_lag - syn_peak_lag)

    real_peak_value = float(rr[real_peak_i])
    syn_peak_value = float(ss[syn_peak_i])
    peak_value_error = abs(real_peak_value - syn_peak_value)

    # Similarity: combine curve-shape correlation and absolute error.
    corr_part = max(0.0, corr)
    mae_part = float(np.exp(-mae / 0.25))  # corr values live in [-1, 1]; 0.25 MAE is already large.
    sim = float(0.5 * corr_part + 0.5 * mae_part)

    return {
        "cross_corr_similarity": sim,
        "cross_corr_mae": mae,
        "cross_corr_rmse": rmse,
        "cross_corr_corr": corr,
        "cross_corr_peak_lag_error": float(peak_lag_error),
        "cross_corr_peak_value_error": float(peak_value_error),
        "real_cross_corr_peak_lag": int(real_peak_lag),
        "syn_cross_corr_peak_lag": int(syn_peak_lag),
        "real_cross_corr_peak_value": real_peak_value,
        "syn_cross_corr_peak_value": syn_peak_value,
        "cross_corr_compare_valid": True,
        "cross_corr_compare_reason": "",
    }

def _cross_corr_status_133(row: dict) -> tuple:
    real_events = int(_safe_float_133(row.get("real_driver_event_count"), 0))
    syn_events = int(_safe_float_133(row.get("syn_driver_event_count"), 0))
    min_events = int(CFG.get("cell13_3_min_pair_events_for_pass", 5))

    sim = _safe_float_133(row.get("cross_corr_similarity"), np.nan)
    mae = _safe_float_133(row.get("cross_corr_mae"), np.nan)
    peak_lag_error = _safe_float_133(row.get("cross_corr_peak_lag_error"), np.nan)

    if real_events < min_events and syn_events < min_events:
        return "not_evaluable", "insufficient_real_and_synthetic_driver_events"

    if real_events < min_events:
        return "warning", "insufficient_real_driver_events"

    if syn_events < min_events:
        return "warning", "insufficient_synthetic_driver_events"

    fatal_reasons = []
    warning_reasons = []

    if not np.isfinite(sim):
        warning_reasons.append("cross_corr_similarity_not_finite")
    elif sim < float(CFG.get("cell13_3_cross_corr_similarity_warning", 0.50)):
        fatal_reasons.append("cross_corr_similarity_fatal")
    elif sim < float(CFG.get("cell13_3_cross_corr_similarity_pass", 0.75)):
        warning_reasons.append("cross_corr_similarity_warning")

    if np.isfinite(mae):
        if mae > float(CFG.get("cell13_3_cross_corr_mae_warning", 0.25)):
            fatal_reasons.append("cross_corr_mae_fatal")
        elif mae > float(CFG.get("cell13_3_cross_corr_mae_pass", 0.10)):
            warning_reasons.append("cross_corr_mae_warning")

    if np.isfinite(peak_lag_error):
        if peak_lag_error > float(CFG.get("cell13_3_peak_lag_error_warning_seconds", 15)):
            fatal_reasons.append("cross_corr_peak_lag_error_fatal")
        elif peak_lag_error > float(CFG.get("cell13_3_peak_lag_error_pass_seconds", 5)):
            warning_reasons.append("cross_corr_peak_lag_error_warning")

    if fatal_reasons:
        return "fatal", "|".join(fatal_reasons)

    if warning_reasons:
        return "warning", "|".join(warning_reasons)

    return "pass", "cross_corr_metrics_pass"

# ----------------------------------------------------------
# 3) Validate upstream Cell 13.0 / 13.1 / 13.2 contracts
# ----------------------------------------------------------
contract130_133 = CELL130_Q4_COUPLING_INPUT_CONTRACT
if not isinstance(contract130_133, dict):
    raise RuntimeError("[Cell13.3] CELL130_Q4_COUPLING_INPUT_CONTRACT is not a dict.")

version130_133 = str(contract130_133.get("version", ""))
if "cell13_0_q4_cross_modal_coupling_input_contract_v1_1" not in version130_133:
    raise RuntimeError(
        "[Cell13.3] Unexpected Cell 13.0 contract version. "
        f"Expected v1.1 schema-driven pair registry, got: {version130_133}"
    )

strict130_133 = contract130_133.get("strict_contract", {})
if bool(strict130_133.get("TEST_activity_used_for_pair_selection", True)):
    raise RuntimeError("[Cell13.3] Cell 13.0 contract indicates TEST activity was used for pair selection.")
if not bool(strict130_133.get("pair_registry_schema_driven", False)):
    raise RuntimeError("[Cell13.3] Cell 13.0 contract does not declare schema-driven pair registry.")
if bool(strict130_133.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell13.3] Cell 13.0 contract indicates synthetic values were mutated.")

contract131_133 = CELL13_1_ETA_CONTRACT
if not isinstance(contract131_133, dict):
    raise RuntimeError("[Cell13.3] CELL13_1_ETA_CONTRACT is not a dict.")

version131_133 = str(contract131_133.get("version", ""))
if "cell13_1_q4_event_triggered_average_eta_qa_v1_1" not in version131_133:
    raise RuntimeError(
        "[Cell13.3] Unexpected Cell 13.1 ETA contract version. "
        f"Expected v1.1 contract-hardened ETA QA, got: {version131_133}"
    )

strict131_133 = contract131_133.get("strict_contract", {})
if bool(strict131_133.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell13.3] Cell 13.1 contract indicates synthetic values were mutated.")
if bool(strict131_133.get("TEST_activity_used_for_pair_selection", True)):
    raise RuntimeError("[Cell13.3] Cell 13.1 contract indicates TEST activity was used for pair selection.")

contract132_133 = CELL13_2_LAG_RESPONSE_CONTRACT
if not isinstance(contract132_133, dict):
    raise RuntimeError("[Cell13.3] CELL13_2_LAG_RESPONSE_CONTRACT is not a dict.")

version132_133 = str(contract132_133.get("version", ""))
if "cell13_2_q4_lag_response_window_qa_v1_1" not in version132_133:
    raise RuntimeError(
        "[Cell13.3] Unexpected Cell 13.2 lag/response contract version. "
        f"Expected v1.1 contract-hardened lag/response QA, got: {version132_133}"
    )

strict132_133 = contract132_133.get("strict_contract", {})
if bool(strict132_133.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell13.3] Cell 13.2 contract indicates synthetic values were mutated.")
if bool(strict132_133.get("TEST_activity_used_for_pair_selection", True)):
    raise RuntimeError("[Cell13.3] Cell 13.2 contract indicates TEST activity was used for pair selection.")

eta_summary_133 = contract131_133.get("metric_summary", {})
lag_summary_133 = contract132_133.get("metric_summary", {})
eta_publication_blocker_n_133 = int(eta_summary_133.get("publication_blocker_n", 0) or 0)
lag_publication_blocker_n_133 = int(lag_summary_133.get("publication_blocker_n", 0) or 0)

q3_status_133 = (
    contract130_133.get("quality_status_carried_forward", {})
    .get("q3_observability", {})
    if isinstance(contract130_133.get("quality_status_carried_forward", {}), dict)
    else {}
)
q3_publication_blocker_n_133 = int(
    contract130_133.get("quality_status_carried_forward", {})
    .get("q3_publication_blocker_n", 0)
    if isinstance(contract130_133.get("quality_status_carried_forward", {}), dict)
    else 0
)

# ----------------------------------------------------------
# 4) Normalize inputs
# ----------------------------------------------------------
IOT_FULL_SYN_TEST_130 = IOT_FULL_SYN_TEST_130.copy()
IOT_REAL_TEST_REF_130 = IOT_REAL_TEST_REF_130.copy()
PROTOCOL_SYN_TEST_130 = PROTOCOL_SYN_TEST_130.copy()
PROTOCOL_REAL_TEST_REF_130 = PROTOCOL_REAL_TEST_REF_130.copy()
pairs_df = CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF.copy()

for frame_name, frame in [
    ("IOT_FULL_SYN_TEST_130", IOT_FULL_SYN_TEST_130),
    ("IOT_REAL_TEST_REF_130", IOT_REAL_TEST_REF_130),
    ("PROTOCOL_SYN_TEST_130", PROTOCOL_SYN_TEST_130),
    ("PROTOCOL_REAL_TEST_REF_130", PROTOCOL_REAL_TEST_REF_130),
]:
    if len(frame) != N_TE:
        raise RuntimeError(
            f"[Cell13.3] {frame_name} row mismatch: got={len(frame)} expected={N_TE}"
        )
    frame.index = df_te.index
    frame.columns = frame.columns.astype(str)

pairs_df["driver_col"] = pairs_df["driver_col"].astype(str)
pairs_df["protocol_col"] = pairs_df["protocol_col"].astype(str)

required_pair_cols = ["pair_id", "driver_col", "protocol_col", "protocol_tier"]
missing_pair_cols = [c for c in required_pair_cols if c not in pairs_df.columns]
if missing_pair_cols:
    raise RuntimeError(f"[Cell13.3] Pair registry missing required columns: {missing_pair_cols}")

pairs_df = pairs_df[pairs_df.get("eligible_for_13_3_cross_corr", True).astype(bool)].copy()

if len(pairs_df) == 0:
    raise RuntimeError("[Cell13.3] No cross-correlation-eligible driver/protocol pairs.")

if "activity_stats_used_for_pair_selection" in pairs_df.columns:
    bad_activity_selected = pairs_df[
        pairs_df["activity_stats_used_for_pair_selection"].fillna(False).astype(bool)
    ]
    if len(bad_activity_selected):
        raise RuntimeError(
            "[Cell13.3] Pair registry contains pairs selected using TEST activity statistics. "
            f"Rows={len(bad_activity_selected)}"
        )

missing_driver_syn = sorted(set(pairs_df["driver_col"]) - set(IOT_FULL_SYN_TEST_130.columns))
missing_driver_real = sorted(set(pairs_df["driver_col"]) - set(IOT_REAL_TEST_REF_130.columns))
missing_protocol_syn = sorted(set(pairs_df["protocol_col"]) - set(PROTOCOL_SYN_TEST_130.columns))
missing_protocol_real = sorted(set(pairs_df["protocol_col"]) - set(PROTOCOL_REAL_TEST_REF_130.columns))

if missing_driver_syn or missing_driver_real or missing_protocol_syn or missing_protocol_real:
    raise RuntimeError(
        "[Cell13.3] Missing columns for cross-correlation QA. "
        f"missing_driver_syn={missing_driver_syn[:10]} | "
        f"missing_driver_real={missing_driver_real[:10]} | "
        f"missing_protocol_syn={missing_protocol_syn[:10]} | "
        f"missing_protocol_real={missing_protocol_real[:10]}"
    )

# Optional merge with 13.1/13.2 statuses.
eta_status_map = {}
lag_status_map = {}

if "CELL13_1_ETA_PAIR_METRICS_DF" in globals() and isinstance(CELL13_1_ETA_PAIR_METRICS_DF, pd.DataFrame):
    tmp = CELL13_1_ETA_PAIR_METRICS_DF.copy()
    if "pair_id" in tmp.columns and "eta_status" in tmp.columns:
        eta_status_map = dict(zip(tmp["pair_id"].astype(str), tmp["eta_status"].astype(str)))

if "CELL13_2_LAG_RESPONSE_PAIR_METRICS_DF" in globals() and isinstance(CELL13_2_LAG_RESPONSE_PAIR_METRICS_DF, pd.DataFrame):
    tmp = CELL13_2_LAG_RESPONSE_PAIR_METRICS_DF.copy()
    if "pair_id" in tmp.columns and "lag_response_status" in tmp.columns:
        lag_status_map = dict(zip(tmp["pair_id"].astype(str), tmp["lag_response_status"].astype(str)))

# ----------------------------------------------------------
# 5) Compute cross-correlation metrics
# ----------------------------------------------------------
metric_rows = []
curve_store = {}

driver_real_cache = {}
driver_syn_cache = {}
protocol_real_cache = {}
protocol_syn_cache = {}
cross_real_cache = {}
cross_syn_cache = {}

def _get_driver_real_133(col):
    if col not in driver_real_cache:
        driver_real_cache[col] = _to_num_array_133(IOT_REAL_TEST_REF_130, col)
    return driver_real_cache[col]

def _get_driver_syn_133(col):
    if col not in driver_syn_cache:
        driver_syn_cache[col] = _to_num_array_133(IOT_FULL_SYN_TEST_130, col)
    return driver_syn_cache[col]

def _get_protocol_real_133(col):
    if col not in protocol_real_cache:
        protocol_real_cache[col] = _to_num_array_133(PROTOCOL_REAL_TEST_REF_130, col)
    return protocol_real_cache[col]

def _get_protocol_syn_133(col):
    if col not in protocol_syn_cache:
        protocol_syn_cache[col] = _to_num_array_133(PROTOCOL_SYN_TEST_130, col)
    return protocol_syn_cache[col]

log(
    "[Cell13.3] Computing cross-correlation preservation metrics | "
    f"pairs={len(pairs_df)} | lags=[-{MAX_LAG_133}, +{MAX_LAG_133}]"
)

for i, r in enumerate(pairs_df.itertuples(index=False), start=1):
    if i == 1 or i % 50 == 0 or i == len(pairs_df):
        log(f"[Cell13.3] progress {i}/{len(pairs_df)}")

    row = r._asdict()

    pair_id = str(row.get("pair_id", f"q4_pair_{i:05d}"))
    dcol = str(row["driver_col"])
    pcol = str(row["protocol_col"])
    tier = str(row["protocol_tier"])

    d_real_raw = _get_driver_real_133(dcol)
    d_syn_raw = _get_driver_syn_133(dcol)
    p_real_raw = _get_protocol_real_133(pcol)
    p_syn_raw = _get_protocol_syn_133(pcol)

    real_key = (dcol, pcol, "real")
    syn_key = (dcol, pcol, "syn")

    if real_key not in cross_real_cache:
        real_curve, real_meta = _cross_corr_curve_133(d_real_raw, p_real_raw)
        cross_real_cache[real_key] = (real_curve, real_meta)
    else:
        real_curve, real_meta = cross_real_cache[real_key]

    if syn_key not in cross_syn_cache:
        syn_curve, syn_meta = _cross_corr_curve_133(d_syn_raw, p_syn_raw)
        cross_syn_cache[syn_key] = (syn_curve, syn_meta)
    else:
        syn_curve, syn_meta = cross_syn_cache[syn_key]

    sim = _cross_corr_similarity_133(real_curve, syn_curve)

    real_events = _event_count_133(d_real_raw)
    syn_events = _event_count_133(d_syn_raw)

    metric_row = {
        "pair_id": pair_id,
        "driver_col": dcol,
        "driver_entity": str(row.get("driver_entity", "")),
        "protocol_col": pcol,
        "protocol_tier": tier,
        "semantic_hint": str(row.get("semantic_hint", "")),
        "max_lag_seconds": int(MAX_LAG_133),
        "real_driver_event_count": int(real_events),
        "syn_driver_event_count": int(syn_events),
        "real_finite_overlap_rate": real_meta.get("finite_overlap_rate", np.nan),
        "syn_finite_overlap_rate": syn_meta.get("finite_overlap_rate", np.nan),
        "real_curve_valid": bool(real_meta.get("curve_valid", False)),
        "syn_curve_valid": bool(syn_meta.get("curve_valid", False)),
        "real_curve_finite_lags": int(real_meta.get("curve_finite_lags", 0)),
        "syn_curve_finite_lags": int(syn_meta.get("curve_finite_lags", 0)),
        "eta_status_from_13_1": eta_status_map.get(pair_id, ""),
        "lag_response_status_from_13_2": lag_status_map.get(pair_id, ""),
        **sim,
        "TEST_real_values_used_for_QA_reference": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
    }

    status, reason = _cross_corr_status_133(metric_row)
    metric_row["cross_corr_status"] = status
    metric_row["cross_corr_status_reason"] = reason
    metric_row["cross_corr_publication_blocker"] = bool(status == "fatal")

    metric_rows.append(metric_row)

    if bool(CFG.get("cell13_3_store_cross_corr_curves", True)):
        curve_store[f"{pair_id}__real"] = real_curve.astype(np.float32)
        curve_store[f"{pair_id}__syn"] = syn_curve.astype(np.float32)

cross_corr_metrics_df = pd.DataFrame(metric_rows)

if len(cross_corr_metrics_df) != len(pairs_df):
    raise RuntimeError(
        f"[Cell13.3] Cross-correlation metrics row mismatch: "
        f"got={len(cross_corr_metrics_df)} expected={len(pairs_df)}"
    )

# ----------------------------------------------------------
# 6) Summaries
# ----------------------------------------------------------
tier_summary_rows = []

for tier, sub in cross_corr_metrics_df.groupby("protocol_tier"):
    sub = sub.copy()
    evaluable = sub[sub["cross_corr_status"].astype(str).ne("not_evaluable")]

    tier_summary_rows.append({
        "protocol_tier": tier,
        "pairs_total": int(len(sub)),
        "pairs_evaluable": int(len(evaluable)),
        "pass_n": int((sub["cross_corr_status"].astype(str) == "pass").sum()),
        "warning_n": int((sub["cross_corr_status"].astype(str) == "warning").sum()),
        "fatal_n": int((sub["cross_corr_status"].astype(str) == "fatal").sum()),
        "not_evaluable_n": int((sub["cross_corr_status"].astype(str) == "not_evaluable").sum()),
        "cross_corr_pair_pass_rate": float(
            (evaluable["cross_corr_status"].astype(str) == "pass").mean()
        ) if len(evaluable) else np.nan,
        "mean_cross_corr_similarity": float(pd.to_numeric(evaluable["cross_corr_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "median_cross_corr_similarity": float(pd.to_numeric(evaluable["cross_corr_similarity"], errors="coerce").median()) if len(evaluable) else np.nan,
        "mean_cross_corr_mae": float(pd.to_numeric(evaluable["cross_corr_mae"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_cross_corr_rmse": float(pd.to_numeric(evaluable["cross_corr_rmse"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_cross_corr_peak_lag_error": float(pd.to_numeric(evaluable["cross_corr_peak_lag_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "publication_blocker_n": int(sub["cross_corr_publication_blocker"].fillna(False).astype(bool).sum()),
    })

cross_corr_tier_summary_df = pd.DataFrame(tier_summary_rows)

status_counts = (
    cross_corr_metrics_df["cross_corr_status"].astype(str).value_counts().sort_index().to_dict()
)

evaluable_df = cross_corr_metrics_df[cross_corr_metrics_df["cross_corr_status"].astype(str).ne("not_evaluable")].copy()

metric_summary = {
    "pairs_total": int(len(cross_corr_metrics_df)),
    "pairs_evaluable": int(len(evaluable_df)),
    "status_counts": status_counts,
    "publication_blocker_n": int(cross_corr_metrics_df["cross_corr_publication_blocker"].fillna(False).astype(bool).sum()),
    "cross_corr_pair_pass_rate": float(
        (evaluable_df["cross_corr_status"].astype(str) == "pass").mean()
    ) if len(evaluable_df) else np.nan,
    "mean_cross_corr_similarity": float(pd.to_numeric(evaluable_df["cross_corr_similarity"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "median_cross_corr_similarity": float(pd.to_numeric(evaluable_df["cross_corr_similarity"], errors="coerce").median()) if len(evaluable_df) else np.nan,
    "mean_cross_corr_mae": float(pd.to_numeric(evaluable_df["cross_corr_mae"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_cross_corr_rmse": float(pd.to_numeric(evaluable_df["cross_corr_rmse"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_cross_corr_peak_lag_error": float(pd.to_numeric(evaluable_df["cross_corr_peak_lag_error"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_cross_corr_peak_value_error": float(pd.to_numeric(evaluable_df["cross_corr_peak_value_error"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
}

# ----------------------------------------------------------
# 7) Save outputs
# ----------------------------------------------------------
cross_corr_metrics_df.to_csv(cross_corr_pair_metrics_csv, index=False)
cross_corr_tier_summary_df.to_csv(cross_corr_tier_summary_csv, index=False)

if bool(CFG.get("cell13_3_store_cross_corr_curves", True)):
    curve_store["lags"] = LAGS_133.astype(np.int64)
    np.savez_compressed(cross_corr_curve_npz, **curve_store)

contract = {
    "cell": "13.3",
    "version": CELL133_VERSION,
    "role": "Q4_cross_correlation_preservation_by_protocol_tier_QA",
    "quality_dimension": "Q4_cross_modal_consistency",
    "test_rows": int(N_TE),
    "cross_corr_window": {
        "max_lag_seconds": int(MAX_LAG_133),
        "lags": LAGS_133.tolist(),
    },
    "transforms": {
        "driver_transform": str(CFG.get("cell13_3_driver_transform", "binary_centered")),
        "protocol_transform": str(CFG.get("cell13_3_protocol_transform", "log1p_abs_signed")),
    },
    "metric_summary": metric_summary,
    "tier_summary": cross_corr_tier_summary_df.to_dict("records"),
    "cell13_0_contract_version_seen": version130_133,
    "cell13_1_contract_version_seen": version131_133,
    "cell13_2_contract_version_seen": version132_133,
    "eta_publication_blocker_n_carried_forward": int(eta_publication_blocker_n_133),
    "lag_response_publication_blocker_n_carried_forward": int(lag_publication_blocker_n_133),
    "q3_publication_blocker_n_carried_forward": int(q3_publication_blocker_n_133),
    "q3_observability_status_carried_forward": q3_status_133,
    "thresholds": {
        "cross_corr_similarity_pass": float(CFG.get("cell13_3_cross_corr_similarity_pass", 0.75)),
        "cross_corr_similarity_warning": float(CFG.get("cell13_3_cross_corr_similarity_warning", 0.50)),
        "cross_corr_mae_pass": float(CFG.get("cell13_3_cross_corr_mae_pass", 0.10)),
        "cross_corr_mae_warning": float(CFG.get("cell13_3_cross_corr_mae_warning", 0.25)),
        "peak_lag_error_pass_seconds": float(CFG.get("cell13_3_peak_lag_error_pass_seconds", 5)),
        "peak_lag_error_warning_seconds": float(CFG.get("cell13_3_peak_lag_error_warning_seconds", 15)),
        "min_pair_events_for_pass": int(CFG.get("cell13_3_min_pair_events_for_pass", 5)),
    },
    "strict_contract": {
        "TEST_real_values_used_for_QA_reference": True,
        "TEST_activity_used_for_pair_selection": False,
        "pair_registry_schema_driven": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
    },
    "outputs": {
        "cross_corr_pair_metrics_csv": cross_corr_pair_metrics_csv,
        "cross_corr_tier_summary_csv": cross_corr_tier_summary_csv,
        "cross_corr_curve_npz": cross_corr_curve_npz if bool(CFG.get("cell13_3_store_cross_corr_curves", True)) else "",
        "cross_corr_contract_json": cross_corr_contract_json,
        "cross_corr_contract_canonical_json": cross_corr_contract_canonical_json,
        "cross_corr_manifest_json": cross_corr_manifest_json,
    },
}

_write_json_133(cross_corr_contract_json, contract)
_write_json_133(cross_corr_contract_canonical_json, contract)

manifest = {
    "cell": "13.3",
    "version": CELL133_VERSION,
    "created_outputs": contract["outputs"],
    "quality_dimension": "Q4_cross_modal_consistency",
    "metric_summary": metric_summary,
    "eta_publication_blocker_n_carried_forward": int(eta_publication_blocker_n_133),
    "lag_response_publication_blocker_n_carried_forward": int(lag_publication_blocker_n_133),
    "q3_publication_blocker_n_carried_forward": int(q3_publication_blocker_n_133),
    "strict_contract": contract["strict_contract"],
}

_write_json_133(cross_corr_manifest_json, manifest)

hashes = {
    "cross_corr_pair_metrics_csv_sha256": _sha256_file_133(cross_corr_pair_metrics_csv),
    "cross_corr_tier_summary_csv_sha256": _sha256_file_133(cross_corr_tier_summary_csv),
    "cross_corr_contract_json_sha256": _sha256_file_133(cross_corr_contract_json),
    "cross_corr_contract_canonical_json_sha256": _sha256_file_133(cross_corr_contract_canonical_json),
    "cross_corr_manifest_json_sha256": _sha256_file_133(cross_corr_manifest_json),
}

if bool(CFG.get("cell13_3_store_cross_corr_curves", True)) and os.path.exists(cross_corr_curve_npz):
    hashes["cross_corr_curve_npz_sha256"] = _sha256_file_133(cross_corr_curve_npz)

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_133(cross_corr_contract_json, contract)
_write_json_133(cross_corr_contract_canonical_json, contract)
_write_json_133(cross_corr_manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals for 13.4
# ----------------------------------------------------------
globals()["CELL133_VERSION"] = CELL133_VERSION
globals()["CELL13_3_CROSS_CORR_PAIR_METRICS_DF"] = cross_corr_metrics_df
globals()["CELL13_3_CROSS_CORR_TIER_SUMMARY_DF"] = cross_corr_tier_summary_df
globals()["CELL13_3_CROSS_CORR_CONTRACT"] = contract
globals()["CELL13_3_CROSS_CORR_PAIR_METRICS_CSV"] = cross_corr_pair_metrics_csv
globals()["CELL13_3_CROSS_CORR_TIER_SUMMARY_CSV"] = cross_corr_tier_summary_csv
globals()["CELL13_3_CROSS_CORR_CURVE_NPZ"] = cross_corr_curve_npz
globals()["CELL13_3_CROSS_CORR_CONTRACT_JSON"] = cross_corr_contract_json
globals()["CELL13_3_CROSS_CORR_CONTRACT_CANONICAL_JSON"] = cross_corr_contract_canonical_json
globals()["CELL13_3_CROSS_CORR_MANIFEST_JSON"] = cross_corr_manifest_json

log(
    "[Cell13.3] Cross-correlation coupling QA complete | "
    f"pairs_total={metric_summary['pairs_total']} | "
    f"pairs_evaluable={metric_summary['pairs_evaluable']} | "
    f"pass_rate={metric_summary['cross_corr_pair_pass_rate']:.6f} | "
    f"mean_cross_corr_similarity={metric_summary['mean_cross_corr_similarity']:.6f} | "
    f"mean_cross_corr_mae={metric_summary['mean_cross_corr_mae']:.6f} | "
    f"mean_cross_corr_peak_lag_error={metric_summary['mean_cross_corr_peak_lag_error']:.6f} | "
    f"publication_blocker_n={metric_summary['publication_blocker_n']}"
)
log(f"[Cell13.3] Cross-correlation status counts | {status_counts}")
log(
    "[Cell13.3] Cross-correlation tier summary | "
    f"{cross_corr_tier_summary_df.to_dict('records')}"
)
log(f"[Cell13.3] Saved cross-correlation pair metrics: {cross_corr_pair_metrics_csv} | rows={len(cross_corr_metrics_df)}")
log(f"[Cell13.3] Saved cross-correlation tier summary: {cross_corr_tier_summary_csv} | rows={len(cross_corr_tier_summary_df)}")
log(f"[Cell13.3] Saved canonical cross-correlation contract: {cross_corr_contract_canonical_json}")
if bool(CFG.get("cell13_3_store_cross_corr_curves", True)):
    log(f"[Cell13.3] Saved cross-correlation curves: {cross_corr_curve_npz}")
log(
    "[Cell13.3] Contract flags | "
    "TEST_real_values_used_for_QA_reference=True | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | TEST_activity_used_for_pair_selection=False | pair_registry_schema_driven=True"
)
log("--- END: Cell 13.3 - Q4 cross-correlation preservation QA (v1.1 contract-hardened strict) ---")

gc.collect()