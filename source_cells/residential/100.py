# ==========================================================
# CELL 13.2 - Q4 lag distribution and response-window QA
# v1.1 STUDY-THESIS strict lag/response coupling QA, contract-hardened
#
# Role:
#   - For each driver↔protocol pair:
#       * estimate real driver→protocol response lag distribution
#       * estimate synthetic driver→protocol response lag distribution
#       * compare lag peak timing
#       * compare response-window activation rate
#
# Metrics:
#   lag_peak_error
#   lag_similarity
#   lag_distribution_mae
#   response_window_rate_error
#   response_window_real_rate
#   response_window_syn_rate
#
# Strict rules:
#   - TEST real values are used for QA only.
#   - Do NOT mutate synthetic values.
#   - Do NOT fit generators.
#   - Do NOT select generators.
#   - Do NOT materialize new synthetic values.
#
# Inputs from Cell 13.0 / 13.1:
#   IOT_FULL_SYN_TEST_130
#   IOT_REAL_TEST_REF_130
#   PROTOCOL_SYN_TEST_130
#   PROTOCOL_REAL_TEST_REF_130
#   CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF
#   CELL13_1_ETA_PAIR_METRICS_DF
#
# Outputs:
#   reports/cell13_2_lag_response_pair_metrics.csv
#   reports/cell13_2_lag_response_tier_summary.csv
#   reports/cell13_2_lag_response_contract.json
#   artifacts/cell13_2_lag_response_manifest.json
# ==========================================================

log("--- START: Cell 13.2 - Q4 lag distribution and response-window QA (v1.1 contract-hardened strict) ---")

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
_required_132 = [
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
]
_missing_132 = [k for k in _required_132 if k not in globals()]
if _missing_132:
    raise RuntimeError(f"[Cell13.2] Missing required globals from Cell 13.0: {_missing_132}")

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

CELL132_VERSION = "cell13_2_q4_lag_response_window_qa_v1_1_contract_hardened"

CFG["cell13_2_version"] = CELL132_VERSION
CFG["cell13_2_QA_only"] = True
CFG["cell13_2_TEST_real_values_used_for_QA_reference"] = True
CFG["cell13_2_synthetic_values_mutated"] = False
CFG["cell13_2_selection_done_here"] = False
CFG["cell13_2_generator_fit_done_here"] = False
CFG["cell13_2_materialization_done_here"] = False
CFG["cell13_2_requires_schema_driven_pair_registry"] = True
CFG["cell13_2_TEST_activity_used_for_pair_selection"] = False
CFG["cell13_2_response_threshold_policy"] = "real_TEST_protocol_threshold_applied_to_real_and_synthetic_QA_only"

CFG.setdefault("cell13_2_max_lag_seconds", int(CFG.get("cell13_max_lag_seconds", 60)))
CFG.setdefault("cell13_2_response_window_seconds", int(CFG.get("cell13_lag_window_seconds", 10)))
CFG.setdefault("cell13_2_min_pair_events_for_pass", 5)
CFG.setdefault("cell13_2_protocol_transform", "log1p_abs_signed")
CFG.setdefault("cell13_2_response_threshold_quantile", 0.90)
CFG.setdefault("cell13_2_response_threshold_min_abs", 1e-9)
CFG.setdefault("cell13_2_lag_similarity_pass", 0.75)
CFG.setdefault("cell13_2_lag_similarity_warning", 0.50)
CFG.setdefault("cell13_2_lag_peak_error_pass_seconds", 5)
CFG.setdefault("cell13_2_lag_peak_error_warning_seconds", 15)
CFG.setdefault("cell13_2_response_window_rate_error_pass", 0.10)
CFG.setdefault("cell13_2_response_window_rate_error_warning", 0.25)
CFG.setdefault("cell13_2_norm_eps", 1e-9)
CFG.setdefault("cell13_2_event_deduplicate_adjacent", True)
CFG.setdefault("cell13_2_store_lag_distributions", True)

MAX_LAG_132 = int(CFG.get("cell13_2_max_lag_seconds", 60))
RESP_WINDOW_132 = int(CFG.get("cell13_2_response_window_seconds", 10))

if MAX_LAG_132 < 1:
    raise RuntimeError(f"[Cell13.2] Invalid max lag: {MAX_LAG_132}")
if RESP_WINDOW_132 < 1:
    raise RuntimeError(f"[Cell13.2] Invalid response window: {RESP_WINDOW_132}")

LAGS_132 = np.arange(0, MAX_LAG_132 + 1, dtype=np.int64)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
lag_pair_metrics_csv = os.path.join(REPORT_DIR, "cell13_2_lag_response_pair_metrics.csv")
lag_tier_summary_csv = os.path.join(REPORT_DIR, "cell13_2_lag_response_tier_summary.csv")
lag_distribution_npz = os.path.join(ARTDIR, "cell13_2_lag_distributions.npz")
lag_contract_json = os.path.join(REPORT_DIR, "cell13_2_lag_response_contract.json")
lag_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell13_2_lag_response_contract_v1_1_THESIS.json")
lag_manifest_json = os.path.join(ARTDIR, "cell13_2_lag_response_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_132(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_132(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_132(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_132(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_132(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_132(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_132(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_132(obj.to_dict())
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

def _write_json_132(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_132(payload), f, indent=2, sort_keys=True)

def _sha256_file_132(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_132(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_132(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _transform_protocol_132(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell13_2_protocol_transform", "log1p_abs_signed")).strip().lower()

    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.log1p(np.maximum(arr, 0.0))
    elif mode == "log1p_abs":
        out = np.log1p(np.abs(arr))
    elif mode == "log1p_abs_signed":
        out = np.sign(arr) * np.log1p(np.abs(arr))
    else:
        raise RuntimeError(f"[Cell13.2] Unknown protocol transform: {mode}")

    out[~np.isfinite(out)] = np.nan
    return out

def _event_indices_132(driver_x: np.ndarray) -> np.ndarray:
    x = np.asarray(driver_x, dtype=np.float64)
    idx = np.flatnonzero(np.isfinite(x) & (x > 0.5)).astype(np.int64)

    if not bool(CFG.get("cell13_2_event_deduplicate_adjacent", True)):
        return idx

    if idx.size <= 1:
        return idx

    keep = np.ones(idx.size, dtype=bool)
    keep[1:] = np.diff(idx) > 1
    return idx[keep]

def _response_threshold_132(y: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    yy = yy[np.isfinite(yy)]

    if yy.size == 0:
        return np.nan

    abs_y = np.abs(yy)
    q = float(CFG.get("cell13_2_response_threshold_quantile", 0.90))
    q = float(np.clip(q, 0.50, 0.999))
    thr = float(np.nanquantile(abs_y, q))
    return max(thr, float(CFG.get("cell13_2_response_threshold_min_abs", 1e-9)))

def _lag_response_profile_132(driver_x: np.ndarray, protocol_y: np.ndarray, response_threshold: float = None) -> tuple:
    """
    For each lag L in [0, MAX_LAG], estimate the average transformed protocol
    response at driver_event + L.

    Also computes response-window rate: fraction of events where any protocol
    response exceeds threshold within [0, RESP_WINDOW].
    """
    d = np.asarray(driver_x, dtype=np.float64)
    y = _transform_protocol_132(protocol_y)

    event_idx_all = _event_indices_132(d)
    event_count_all = int(event_idx_all.size)

    valid_idx = event_idx_all[event_idx_all + MAX_LAG_132 < len(y)]
    event_count_valid = int(valid_idx.size)

    if event_count_valid == 0:
        return np.full(len(LAGS_132), np.nan, dtype=np.float64), {
            "event_count_all": event_count_all,
            "event_count_lag_valid": event_count_valid,
            "lag_valid": False,
            "lag_reason": "no_lag_valid_events",
            "response_window_rate": np.nan,
            "response_threshold": np.nan,
        }

    profile = np.full(len(LAGS_132), np.nan, dtype=np.float64)
    finite_per_lag = []

    for j, lag in enumerate(LAGS_132):
        vals = y[valid_idx + int(lag)]
        finite = np.isfinite(vals)
        finite_per_lag.append(int(finite.sum()))
        if finite.any():
            profile[j] = float(np.nanmean(vals[finite]))

    # Baseline-correct using lag 0 as a weak local reference? For response lag,
    # keep raw transformed response profile but center by global finite mean to
    # reduce scale differences.
    y_finite = y[np.isfinite(y)]
    baseline = float(np.nanmean(y_finite)) if y_finite.size else 0.0
    profile = profile - baseline

    # Response-window rate.
    # v1.1: use a shared threshold per real/synthetic pair, normally derived
    # from the real TEST protocol reference and then applied to both real and
    # synthetic profiles. This is QA-only and avoids comparing each side against
    # its own distribution-specific threshold.
    thr = _safe_float_132(response_threshold, np.nan)
    if not np.isfinite(thr):
        thr = _response_threshold_132(y)

    response_hits = []
    if np.isfinite(thr):
        w = int(RESP_WINDOW_132)
        for event_i in valid_idx:
            vals = y[event_i : event_i + w + 1]
            vals = vals[np.isfinite(vals)]
            response_hits.append(bool(vals.size and np.nanmax(np.abs(vals)) >= thr))

    response_window_rate = (
        float(np.mean(response_hits))
        if response_hits
        else np.nan
    )

    return profile.astype(np.float64), {
        "event_count_all": event_count_all,
        "event_count_lag_valid": event_count_valid,
        "lag_valid": bool(np.isfinite(profile).any()),
        "lag_reason": "",
        "response_window_rate": response_window_rate,
        "response_threshold": thr,
        "lag_baseline": baseline,
        "lag_min_finite_per_lag": int(np.nanmin(finite_per_lag)) if finite_per_lag else 0,
        "lag_mean_finite_per_lag": float(np.nanmean(finite_per_lag)) if finite_per_lag else np.nan,
    }

def _lag_similarity_132(real_profile: np.ndarray, syn_profile: np.ndarray) -> dict:
    r = np.asarray(real_profile, dtype=np.float64)
    s = np.asarray(syn_profile, dtype=np.float64)

    valid = np.isfinite(r) & np.isfinite(s)

    if int(valid.sum()) < 3:
        return {
            "lag_similarity": np.nan,
            "lag_distribution_mae": np.nan,
            "lag_distribution_norm_mae": np.nan,
            "lag_corr": np.nan,
            "lag_peak_error": np.nan,
            "real_lag_peak": np.nan,
            "syn_lag_peak": np.nan,
            "lag_compare_valid": False,
            "lag_compare_reason": "insufficient_finite_lags",
        }

    rr = r[valid]
    ss = s[valid]
    lags = LAGS_132[valid]

    mae = float(np.mean(np.abs(rr - ss)))
    scale = float(np.nanmean(np.abs(rr)) + np.nanstd(rr) + float(CFG.get("cell13_2_norm_eps", 1e-9)))
    norm_mae = float(mae / max(scale, float(CFG.get("cell13_2_norm_eps", 1e-9))))

    if np.std(rr) <= 1e-12 or np.std(ss) <= 1e-12:
        corr = 1.0 if np.allclose(rr, ss, equal_nan=True) else 0.0
    else:
        corr = float(np.corrcoef(rr, ss)[0, 1])

    corr_part = max(0.0, corr)
    amp_part = float(np.exp(-norm_mae))
    lag_similarity = float(0.5 * corr_part + 0.5 * amp_part)

    real_peak = int(lags[int(np.nanargmax(np.abs(rr)))])
    syn_peak = int(lags[int(np.nanargmax(np.abs(ss)))])
    peak_error = abs(real_peak - syn_peak)

    return {
        "lag_similarity": lag_similarity,
        "lag_distribution_mae": mae,
        "lag_distribution_norm_mae": norm_mae,
        "lag_corr": corr,
        "lag_peak_error": float(peak_error),
        "real_lag_peak": int(real_peak),
        "syn_lag_peak": int(syn_peak),
        "lag_compare_valid": True,
        "lag_compare_reason": "",
    }

def _lag_status_132(row: dict) -> tuple:
    real_events = int(_safe_float_132(row.get("real_event_count_lag_valid"), 0))
    syn_events = int(_safe_float_132(row.get("syn_event_count_lag_valid"), 0))
    min_events = int(CFG.get("cell13_2_min_pair_events_for_pass", 5))

    sim = _safe_float_132(row.get("lag_similarity"), np.nan)
    peak_err = _safe_float_132(row.get("lag_peak_error"), np.nan)
    resp_err = _safe_float_132(row.get("response_window_rate_error"), np.nan)

    if real_events < min_events and syn_events < min_events:
        return "not_evaluable", "insufficient_real_and_synthetic_driver_events"

    if real_events < min_events:
        return "warning", "insufficient_real_driver_events"

    if syn_events < min_events:
        return "warning", "insufficient_synthetic_driver_events"

    fatal_reasons = []
    warning_reasons = []

    if not np.isfinite(sim):
        warning_reasons.append("lag_similarity_not_finite")
    elif sim < float(CFG.get("cell13_2_lag_similarity_warning", 0.50)):
        fatal_reasons.append("lag_similarity_fatal")
    elif sim < float(CFG.get("cell13_2_lag_similarity_pass", 0.75)):
        warning_reasons.append("lag_similarity_warning")

    if np.isfinite(peak_err):
        if peak_err > float(CFG.get("cell13_2_lag_peak_error_warning_seconds", 15)):
            fatal_reasons.append("lag_peak_error_fatal")
        elif peak_err > float(CFG.get("cell13_2_lag_peak_error_pass_seconds", 5)):
            warning_reasons.append("lag_peak_error_warning")

    if np.isfinite(resp_err):
        if resp_err > float(CFG.get("cell13_2_response_window_rate_error_warning", 0.25)):
            fatal_reasons.append("response_window_rate_error_fatal")
        elif resp_err > float(CFG.get("cell13_2_response_window_rate_error_pass", 0.10)):
            warning_reasons.append("response_window_rate_error_warning")

    if fatal_reasons:
        return "fatal", "|".join(fatal_reasons)

    if warning_reasons:
        return "warning", "|".join(warning_reasons)

    return "pass", "lag_response_metrics_pass"

# ----------------------------------------------------------
# 3) Validate upstream Cell 13.0 / 13.1 contracts
# ----------------------------------------------------------
contract130_132 = CELL130_Q4_COUPLING_INPUT_CONTRACT
if not isinstance(contract130_132, dict):
    raise RuntimeError("[Cell13.2] CELL130_Q4_COUPLING_INPUT_CONTRACT is not a dict.")

version130_132 = str(contract130_132.get("version", ""))
if "cell13_0_q4_cross_modal_coupling_input_contract_v1_1" not in version130_132:
    raise RuntimeError(
        "[Cell13.2] Unexpected Cell 13.0 contract version. "
        f"Expected v1.1 schema-driven pair registry, got: {version130_132}"
    )

strict130_132 = contract130_132.get("strict_contract", {})
if bool(strict130_132.get("TEST_activity_used_for_pair_selection", True)):
    raise RuntimeError("[Cell13.2] Cell 13.0 contract indicates TEST activity was used for pair selection.")
if not bool(strict130_132.get("pair_registry_schema_driven", False)):
    raise RuntimeError("[Cell13.2] Cell 13.0 contract does not declare schema-driven pair registry.")
if bool(strict130_132.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell13.2] Cell 13.0 contract indicates synthetic values were mutated.")

contract131_132 = CELL13_1_ETA_CONTRACT
if not isinstance(contract131_132, dict):
    raise RuntimeError("[Cell13.2] CELL13_1_ETA_CONTRACT is not a dict.")

version131_132 = str(contract131_132.get("version", ""))
if "cell13_1_q4_event_triggered_average_eta_qa_v1_1" not in version131_132:
    raise RuntimeError(
        "[Cell13.2] Unexpected Cell 13.1 ETA contract version. "
        f"Expected v1.1 contract-hardened ETA QA, got: {version131_132}"
    )

strict131_132 = contract131_132.get("strict_contract", {})
if bool(strict131_132.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell13.2] Cell 13.1 contract indicates synthetic values were mutated.")
if bool(strict131_132.get("TEST_activity_used_for_pair_selection", True)):
    raise RuntimeError("[Cell13.2] Cell 13.1 contract indicates TEST activity was used for pair selection.")

eta_summary_132 = contract131_132.get("metric_summary", {})
eta_publication_blocker_n_132 = int(eta_summary_132.get("publication_blocker_n", 0) or 0)

q3_status_132 = (
    contract130_132.get("quality_status_carried_forward", {})
    .get("q3_observability", {})
    if isinstance(contract130_132.get("quality_status_carried_forward", {}), dict)
    else {}
)
q3_publication_blocker_n_132 = int(
    contract130_132.get("quality_status_carried_forward", {})
    .get("q3_publication_blocker_n", 0)
    if isinstance(contract130_132.get("quality_status_carried_forward", {}), dict)
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
            f"[Cell13.2] {frame_name} row mismatch: got={len(frame)} expected={N_TE}"
        )
    frame.index = df_te.index
    frame.columns = frame.columns.astype(str)

pairs_df["driver_col"] = pairs_df["driver_col"].astype(str)
pairs_df["protocol_col"] = pairs_df["protocol_col"].astype(str)

required_pair_cols = ["pair_id", "driver_col", "protocol_col", "protocol_tier"]
missing_pair_cols = [c for c in required_pair_cols if c not in pairs_df.columns]
if missing_pair_cols:
    raise RuntimeError(f"[Cell13.2] Pair registry missing required columns: {missing_pair_cols}")

pairs_df = pairs_df[pairs_df.get("eligible_for_13_2_lag", True).astype(bool)].copy()

if len(pairs_df) == 0:
    raise RuntimeError("[Cell13.2] No lag-eligible driver/protocol pairs.")

if "activity_stats_used_for_pair_selection" in pairs_df.columns:
    bad_activity_selected = pairs_df[
        pairs_df["activity_stats_used_for_pair_selection"].fillna(False).astype(bool)
    ]
    if len(bad_activity_selected):
        raise RuntimeError(
            "[Cell13.2] Pair registry contains pairs selected using TEST activity statistics. "
            f"Rows={len(bad_activity_selected)}"
        )

# Validate columns.
missing_driver_syn = sorted(set(pairs_df["driver_col"]) - set(IOT_FULL_SYN_TEST_130.columns))
missing_driver_real = sorted(set(pairs_df["driver_col"]) - set(IOT_REAL_TEST_REF_130.columns))
missing_protocol_syn = sorted(set(pairs_df["protocol_col"]) - set(PROTOCOL_SYN_TEST_130.columns))
missing_protocol_real = sorted(set(pairs_df["protocol_col"]) - set(PROTOCOL_REAL_TEST_REF_130.columns))

if missing_driver_syn or missing_driver_real or missing_protocol_syn or missing_protocol_real:
    raise RuntimeError(
        "[Cell13.2] Missing columns for lag QA. "
        f"missing_driver_syn={missing_driver_syn[:10]} | "
        f"missing_driver_real={missing_driver_real[:10]} | "
        f"missing_protocol_syn={missing_protocol_syn[:10]} | "
        f"missing_protocol_real={missing_protocol_real[:10]}"
    )

# ----------------------------------------------------------
# 5) Compute lag/response metrics
# ----------------------------------------------------------
metric_rows = []
dist_store = {}

driver_real_cache = {}
driver_syn_cache = {}
protocol_real_cache = {}
protocol_syn_cache = {}
lag_real_cache = {}
lag_syn_cache = {}

def _get_driver_real_132(col):
    if col not in driver_real_cache:
        driver_real_cache[col] = _to_num_array_132(IOT_REAL_TEST_REF_130, col)
    return driver_real_cache[col]

def _get_driver_syn_132(col):
    if col not in driver_syn_cache:
        driver_syn_cache[col] = _to_num_array_132(IOT_FULL_SYN_TEST_130, col)
    return driver_syn_cache[col]

def _get_protocol_real_132(col):
    if col not in protocol_real_cache:
        protocol_real_cache[col] = _to_num_array_132(PROTOCOL_REAL_TEST_REF_130, col)
    return protocol_real_cache[col]

def _get_protocol_syn_132(col):
    if col not in protocol_syn_cache:
        protocol_syn_cache[col] = _to_num_array_132(PROTOCOL_SYN_TEST_130, col)
    return protocol_syn_cache[col]

log(
    "[Cell13.2] Computing lag/response-window coupling metrics | "
    f"pairs={len(pairs_df)} | max_lag={MAX_LAG_132}s | response_window={RESP_WINDOW_132}s"
)

for i, r in enumerate(pairs_df.itertuples(index=False), start=1):
    if i == 1 or i % 50 == 0 or i == len(pairs_df):
        log(f"[Cell13.2] progress {i}/{len(pairs_df)}")

    row = r._asdict()

    pair_id = str(row.get("pair_id", f"q4_pair_{i:05d}"))
    dcol = str(row["driver_col"])
    pcol = str(row["protocol_col"])
    tier = str(row["protocol_tier"])

    d_real = _get_driver_real_132(dcol)
    d_syn = _get_driver_syn_132(dcol)
    p_real = _get_protocol_real_132(pcol)
    p_syn = _get_protocol_syn_132(pcol)

    real_key = (dcol, pcol, "real")
    syn_key = (dcol, pcol, "syn")

    # Shared response threshold is derived from real TEST protocol reference
    # and applied to both real and synthetic response-window rates. This is
    # evaluation-only and avoids per-side threshold drift.
    shared_response_threshold = _response_threshold_132(_transform_protocol_132(p_real))

    if real_key not in lag_real_cache:
        real_profile, real_meta = _lag_response_profile_132(
            d_real, p_real, response_threshold=shared_response_threshold
        )
        lag_real_cache[real_key] = (real_profile, real_meta)
    else:
        real_profile, real_meta = lag_real_cache[real_key]

    if syn_key not in lag_syn_cache:
        syn_profile, syn_meta = _lag_response_profile_132(
            d_syn, p_syn, response_threshold=shared_response_threshold
        )
        lag_syn_cache[syn_key] = (syn_profile, syn_meta)
    else:
        syn_profile, syn_meta = lag_syn_cache[syn_key]

    sim = _lag_similarity_132(real_profile, syn_profile)

    real_resp_rate = _safe_float_132(real_meta.get("response_window_rate"), np.nan)
    syn_resp_rate = _safe_float_132(syn_meta.get("response_window_rate"), np.nan)
    response_window_rate_error = (
        abs(real_resp_rate - syn_resp_rate)
        if np.isfinite(real_resp_rate) and np.isfinite(syn_resp_rate)
        else np.nan
    )

    metric_row = {
        "pair_id": pair_id,
        "driver_col": dcol,
        "driver_entity": str(row.get("driver_entity", "")),
        "protocol_col": pcol,
        "protocol_tier": tier,
        "semantic_hint": str(row.get("semantic_hint", "")),
        "max_lag_seconds": int(MAX_LAG_132),
        "response_window_seconds": int(RESP_WINDOW_132),
        "real_event_count_all": int(real_meta.get("event_count_all", 0)),
        "real_event_count_lag_valid": int(real_meta.get("event_count_lag_valid", 0)),
        "syn_event_count_all": int(syn_meta.get("event_count_all", 0)),
        "syn_event_count_lag_valid": int(syn_meta.get("event_count_lag_valid", 0)),
        "real_lag_valid": bool(real_meta.get("lag_valid", False)),
        "syn_lag_valid": bool(syn_meta.get("lag_valid", False)),
        "real_lag_reason": str(real_meta.get("lag_reason", "")),
        "syn_lag_reason": str(syn_meta.get("lag_reason", "")),
        "real_response_window_rate": real_resp_rate,
        "syn_response_window_rate": syn_resp_rate,
        "response_window_rate_error": response_window_rate_error,
        "real_response_threshold": real_meta.get("response_threshold", np.nan),
        "syn_response_threshold": syn_meta.get("response_threshold", np.nan),
        "shared_response_threshold": shared_response_threshold,
        "response_threshold_policy": "real_TEST_protocol_threshold_applied_to_real_and_synthetic_QA_only",
        **sim,
        "TEST_real_values_used_for_QA_reference": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
    }

    status, reason = _lag_status_132(metric_row)
    metric_row["lag_response_status"] = status
    metric_row["lag_response_status_reason"] = reason
    metric_row["lag_response_publication_blocker"] = bool(status == "fatal")

    metric_rows.append(metric_row)

    if bool(CFG.get("cell13_2_store_lag_distributions", True)):
        dist_store[f"{pair_id}__real"] = real_profile.astype(np.float32)
        dist_store[f"{pair_id}__syn"] = syn_profile.astype(np.float32)

lag_metrics_df = pd.DataFrame(metric_rows)

if len(lag_metrics_df) != len(pairs_df):
    raise RuntimeError(
        f"[Cell13.2] Lag metrics row mismatch: got={len(lag_metrics_df)} expected={len(pairs_df)}"
    )

# ----------------------------------------------------------
# 6) Summaries
# ----------------------------------------------------------
tier_summary_rows = []

for tier, sub in lag_metrics_df.groupby("protocol_tier"):
    sub = sub.copy()
    evaluable = sub[sub["lag_response_status"].astype(str).ne("not_evaluable")]

    tier_summary_rows.append({
        "protocol_tier": tier,
        "pairs_total": int(len(sub)),
        "pairs_evaluable": int(len(evaluable)),
        "pass_n": int((sub["lag_response_status"].astype(str) == "pass").sum()),
        "warning_n": int((sub["lag_response_status"].astype(str) == "warning").sum()),
        "fatal_n": int((sub["lag_response_status"].astype(str) == "fatal").sum()),
        "not_evaluable_n": int((sub["lag_response_status"].astype(str) == "not_evaluable").sum()),
        "lag_pair_pass_rate": float(
            (evaluable["lag_response_status"].astype(str) == "pass").mean()
        ) if len(evaluable) else np.nan,
        "mean_lag_similarity": float(pd.to_numeric(evaluable["lag_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "median_lag_similarity": float(pd.to_numeric(evaluable["lag_similarity"], errors="coerce").median()) if len(evaluable) else np.nan,
        "mean_lag_peak_error": float(pd.to_numeric(evaluable["lag_peak_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_response_window_rate_error": float(pd.to_numeric(evaluable["response_window_rate_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "publication_blocker_n": int(sub["lag_response_publication_blocker"].fillna(False).astype(bool).sum()),
    })

lag_tier_summary_df = pd.DataFrame(tier_summary_rows)

status_counts = (
    lag_metrics_df["lag_response_status"].astype(str).value_counts().sort_index().to_dict()
)

evaluable_df = lag_metrics_df[lag_metrics_df["lag_response_status"].astype(str).ne("not_evaluable")].copy()

metric_summary = {
    "pairs_total": int(len(lag_metrics_df)),
    "pairs_evaluable": int(len(evaluable_df)),
    "status_counts": status_counts,
    "publication_blocker_n": int(lag_metrics_df["lag_response_publication_blocker"].fillna(False).astype(bool).sum()),
    "lag_pair_pass_rate": float(
        (evaluable_df["lag_response_status"].astype(str) == "pass").mean()
    ) if len(evaluable_df) else np.nan,
    "mean_lag_similarity": float(pd.to_numeric(evaluable_df["lag_similarity"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "median_lag_similarity": float(pd.to_numeric(evaluable_df["lag_similarity"], errors="coerce").median()) if len(evaluable_df) else np.nan,
    "mean_lag_distribution_norm_mae": float(pd.to_numeric(evaluable_df["lag_distribution_norm_mae"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_lag_peak_error": float(pd.to_numeric(evaluable_df["lag_peak_error"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_response_window_rate_error": float(pd.to_numeric(evaluable_df["response_window_rate_error"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_real_response_window_rate": float(pd.to_numeric(evaluable_df["real_response_window_rate"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_syn_response_window_rate": float(pd.to_numeric(evaluable_df["syn_response_window_rate"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
}

# ----------------------------------------------------------
# 7) Save outputs
# ----------------------------------------------------------
lag_metrics_df.to_csv(lag_pair_metrics_csv, index=False)
lag_tier_summary_df.to_csv(lag_tier_summary_csv, index=False)

if bool(CFG.get("cell13_2_store_lag_distributions", True)):
    dist_store["lags"] = LAGS_132.astype(np.int64)
    np.savez_compressed(lag_distribution_npz, **dist_store)

contract = {
    "cell": "13.2",
    "version": CELL132_VERSION,
    "role": "Q4_lag_distribution_and_response_window_QA",
    "quality_dimension": "Q4_cross_modal_consistency",
    "test_rows": int(N_TE),
    "lag_window": {
        "max_lag_seconds": int(MAX_LAG_132),
        "lags": LAGS_132.tolist(),
    },
    "response_window": {
        "seconds": int(RESP_WINDOW_132),
        "threshold_quantile": float(CFG.get("cell13_2_response_threshold_quantile", 0.90)),
        "threshold_policy": "real_TEST_protocol_threshold_applied_to_real_and_synthetic_QA_only",
    },
    "protocol_transform": str(CFG.get("cell13_2_protocol_transform", "log1p_abs_signed")),
    "metric_summary": metric_summary,
    "tier_summary": lag_tier_summary_df.to_dict("records"),
    "cell13_0_contract_version_seen": version130_132,
    "cell13_1_contract_version_seen": version131_132,
    "eta_publication_blocker_n_carried_forward": int(eta_publication_blocker_n_132),
    "q3_publication_blocker_n_carried_forward": int(q3_publication_blocker_n_132),
    "q3_observability_status_carried_forward": q3_status_132,
    "thresholds": {
        "lag_similarity_pass": float(CFG.get("cell13_2_lag_similarity_pass", 0.75)),
        "lag_similarity_warning": float(CFG.get("cell13_2_lag_similarity_warning", 0.50)),
        "lag_peak_error_pass_seconds": float(CFG.get("cell13_2_lag_peak_error_pass_seconds", 5)),
        "lag_peak_error_warning_seconds": float(CFG.get("cell13_2_lag_peak_error_warning_seconds", 15)),
        "response_window_rate_error_pass": float(CFG.get("cell13_2_response_window_rate_error_pass", 0.10)),
        "response_window_rate_error_warning": float(CFG.get("cell13_2_response_window_rate_error_warning", 0.25)),
        "min_pair_events_for_pass": int(CFG.get("cell13_2_min_pair_events_for_pass", 5)),
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
        "lag_pair_metrics_csv": lag_pair_metrics_csv,
        "lag_tier_summary_csv": lag_tier_summary_csv,
        "lag_distribution_npz": lag_distribution_npz if bool(CFG.get("cell13_2_store_lag_distributions", True)) else "",
        "lag_contract_json": lag_contract_json,
        "lag_contract_canonical_json": lag_contract_canonical_json,
        "lag_manifest_json": lag_manifest_json,
    },
}

_write_json_132(lag_contract_json, contract)
_write_json_132(lag_contract_canonical_json, contract)

manifest = {
    "cell": "13.2",
    "version": CELL132_VERSION,
    "created_outputs": contract["outputs"],
    "quality_dimension": "Q4_cross_modal_consistency",
    "metric_summary": metric_summary,
    "eta_publication_blocker_n_carried_forward": int(eta_publication_blocker_n_132),
    "q3_publication_blocker_n_carried_forward": int(q3_publication_blocker_n_132),
    "strict_contract": contract["strict_contract"],
}

_write_json_132(lag_manifest_json, manifest)

hashes = {
    "lag_pair_metrics_csv_sha256": _sha256_file_132(lag_pair_metrics_csv),
    "lag_tier_summary_csv_sha256": _sha256_file_132(lag_tier_summary_csv),
    "lag_contract_json_sha256": _sha256_file_132(lag_contract_json),
    "lag_contract_canonical_json_sha256": _sha256_file_132(lag_contract_canonical_json),
    "lag_manifest_json_sha256": _sha256_file_132(lag_manifest_json),
}

if bool(CFG.get("cell13_2_store_lag_distributions", True)) and os.path.exists(lag_distribution_npz):
    hashes["lag_distribution_npz_sha256"] = _sha256_file_132(lag_distribution_npz)

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_132(lag_contract_json, contract)
_write_json_132(lag_contract_canonical_json, contract)
_write_json_132(lag_manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals for 13.3+
# ----------------------------------------------------------
globals()["CELL132_VERSION"] = CELL132_VERSION
globals()["CELL13_2_LAG_RESPONSE_PAIR_METRICS_DF"] = lag_metrics_df
globals()["CELL13_2_LAG_RESPONSE_TIER_SUMMARY_DF"] = lag_tier_summary_df
globals()["CELL13_2_LAG_RESPONSE_CONTRACT"] = contract
globals()["CELL13_2_LAG_RESPONSE_PAIR_METRICS_CSV"] = lag_pair_metrics_csv
globals()["CELL13_2_LAG_RESPONSE_TIER_SUMMARY_CSV"] = lag_tier_summary_csv
globals()["CELL13_2_LAG_DISTRIBUTION_NPZ"] = lag_distribution_npz
globals()["CELL13_2_LAG_RESPONSE_CONTRACT_JSON"] = lag_contract_json
globals()["CELL13_2_LAG_RESPONSE_CONTRACT_CANONICAL_JSON"] = lag_contract_canonical_json
globals()["CELL13_2_LAG_RESPONSE_MANIFEST_JSON"] = lag_manifest_json

log(
    "[Cell13.2] Lag/response coupling QA complete | "
    f"pairs_total={metric_summary['pairs_total']} | "
    f"pairs_evaluable={metric_summary['pairs_evaluable']} | "
    f"pass_rate={metric_summary['lag_pair_pass_rate']:.6f} | "
    f"mean_lag_similarity={metric_summary['mean_lag_similarity']:.6f} | "
    f"mean_lag_peak_error={metric_summary['mean_lag_peak_error']:.6f} | "
    f"mean_response_window_rate_error={metric_summary['mean_response_window_rate_error']:.6f} | "
    f"publication_blocker_n={metric_summary['publication_blocker_n']}"
)
log(f"[Cell13.2] Lag/response status counts | {status_counts}")
log(
    "[Cell13.2] Lag/response tier summary | "
    f"{lag_tier_summary_df.to_dict('records')}"
)
log(f"[Cell13.2] Saved lag/response pair metrics: {lag_pair_metrics_csv} | rows={len(lag_metrics_df)}")
log(f"[Cell13.2] Saved lag/response tier summary: {lag_tier_summary_csv} | rows={len(lag_tier_summary_df)}")
log(f"[Cell13.2] Saved canonical lag/response contract: {lag_contract_canonical_json}")
if bool(CFG.get("cell13_2_store_lag_distributions", True)):
    log(f"[Cell13.2] Saved lag distributions: {lag_distribution_npz}")
log(
    "[Cell13.2] Contract flags | "
    "TEST_real_values_used_for_QA_reference=True | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | TEST_activity_used_for_pair_selection=False | pair_registry_schema_driven=True | shared_response_threshold_policy=True"
)
log("--- END: Cell 13.2 - Q4 lag distribution and response-window QA (v1.1 contract-hardened strict) ---")

gc.collect()