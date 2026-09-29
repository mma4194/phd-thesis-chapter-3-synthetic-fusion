# ==========================================================
# CELL 13.1 - Q4 IoT-driver-to-network event-triggered average QA
# v1.1 STUDY-THESIS strict ETA coupling QA, contract-hardened
#
# Role:
#   - For each driver↔protocol pair from Cell 13.0:
#       * compute real TEST event-triggered protocol response curve
#       * compute synthetic TEST event-triggered protocol response curve
#       * compare curves using ETA_similarity
#   - This implements Q4 cross-modal consistency:
#       IoT event drivers should induce bounded network/protocol responses.
#
# Metrics:
#   ETA_similarity
#   eta_mae
#   eta_norm_mae
#   eta_peak_error
#   eta_area_error
#   real/syn event counts
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
# Outputs:
#   reports/cell13_1_eta_pair_metrics.csv
#   reports/cell13_1_eta_tier_summary.csv
#   reports/cell13_1_eta_contract.json
#   artifacts/cell13_1_eta_manifest.json
# ==========================================================

log("--- START: Cell 13.1 - Q4 event-triggered average QA (v1.1 contract-hardened strict) ---")

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
_required_131 = [
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
]
_missing_131 = [k for k in _required_131 if k not in globals()]
if _missing_131:
    raise RuntimeError(f"[Cell13.1] Missing required globals from Cell 13.0: {_missing_131}")

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

CELL131_VERSION = "cell13_1_q4_event_triggered_average_eta_qa_v1_1_contract_hardened"

CFG["cell13_1_version"] = CELL131_VERSION
CFG["cell13_1_QA_only"] = True
CFG["cell13_1_TEST_real_values_used_for_QA_reference"] = True
CFG["cell13_1_synthetic_values_mutated"] = False
CFG["cell13_1_selection_done_here"] = False
CFG["cell13_1_generator_fit_done_here"] = False
CFG["cell13_1_materialization_done_here"] = False
CFG["cell13_1_requires_schema_driven_pair_registry"] = True
CFG["cell13_1_TEST_activity_used_for_pair_selection"] = False

CFG.setdefault("cell13_1_eta_pre_window_seconds", int(CFG.get("cell13_eta_pre_window_seconds", 10)))
CFG.setdefault("cell13_1_eta_post_window_seconds", int(CFG.get("cell13_eta_post_window_seconds", 30)))
CFG.setdefault("cell13_1_min_real_driver_events", int(CFG.get("cell13_min_driver_events_for_pair", 5)))
CFG.setdefault("cell13_1_min_syn_driver_events", int(CFG.get("cell13_min_driver_events_for_pair", 5)))
CFG.setdefault("cell13_1_min_pair_events_for_pass", 5)
CFG.setdefault("cell13_1_eta_similarity_pass", 0.75)
CFG.setdefault("cell13_1_eta_similarity_warning", 0.50)
CFG.setdefault("cell13_1_norm_eps", 1e-9)
CFG.setdefault("cell13_1_event_deduplicate_adjacent", True)
CFG.setdefault("cell13_1_protocol_transform", "log1p_abs_signed")
CFG.setdefault("cell13_1_store_eta_curves", True)

ETA_PRE_131 = int(CFG.get("cell13_1_eta_pre_window_seconds", 10))
ETA_POST_131 = int(CFG.get("cell13_1_eta_post_window_seconds", 30))

if ETA_PRE_131 < 0 or ETA_POST_131 < 0:
    raise RuntimeError(
        f"[Cell13.1] Invalid ETA windows: pre={ETA_PRE_131}, post={ETA_POST_131}"
    )

ETA_LAGS_131 = np.arange(-ETA_PRE_131, ETA_POST_131 + 1, dtype=np.int64)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
eta_pair_metrics_csv = os.path.join(REPORT_DIR, "cell13_1_eta_pair_metrics.csv")
eta_tier_summary_csv = os.path.join(REPORT_DIR, "cell13_1_eta_tier_summary.csv")
eta_curve_npz = os.path.join(ARTDIR, "cell13_1_eta_curves.npz")
eta_contract_json = os.path.join(REPORT_DIR, "cell13_1_eta_contract.json")
eta_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell13_1_eta_contract_v1_1_THESIS.json")
eta_manifest_json = os.path.join(ARTDIR, "cell13_1_eta_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_131(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_131(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_131(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_131(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_131(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_131(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_131(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_131(obj.to_dict())
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

def _write_json_131(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_131(payload), f, indent=2, sort_keys=True)

def _sha256_file_131(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_131(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_131(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _transform_protocol_131(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell13_1_protocol_transform", "log1p_abs_signed")).strip().lower()

    if mode == "identity":
        return arr

    if mode == "log1p":
        out = np.log1p(np.maximum(arr, 0.0))
        out[~np.isfinite(out)] = np.nan
        return out

    if mode == "log1p_abs":
        out = np.log1p(np.abs(arr))
        out[~np.isfinite(out)] = np.nan
        return out

    if mode == "log1p_abs_signed":
        out = np.sign(arr) * np.log1p(np.abs(arr))
        out[~np.isfinite(out)] = np.nan
        return out

    raise RuntimeError(f"[Cell13.1] Unknown protocol transform: {mode}")

def _event_indices_131(driver_x: np.ndarray) -> np.ndarray:
    x = np.asarray(driver_x, dtype=np.float64)
    idx = np.flatnonzero(np.isfinite(x) & (x > 0.5)).astype(np.int64)

    if not bool(CFG.get("cell13_1_event_deduplicate_adjacent", True)):
        return idx

    if idx.size <= 1:
        return idx

    # Keep starts of adjacent event-runs only.
    keep = np.ones(idx.size, dtype=bool)
    keep[1:] = np.diff(idx) > 1
    return idx[keep]

def _valid_event_indices_for_window_131(idx: np.ndarray, n: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx
    valid = (idx - ETA_PRE_131 >= 0) & (idx + ETA_POST_131 < int(n))
    return idx[valid]

def _eta_curve_131(driver_x: np.ndarray, protocol_y: np.ndarray) -> tuple:
    y = _transform_protocol_131(protocol_y)
    idx = _event_indices_131(driver_x)
    idx_all_n = int(idx.size)
    idx = _valid_event_indices_for_window_131(idx, len(y))
    idx_valid_n = int(idx.size)

    if idx_valid_n == 0:
        return np.full(len(ETA_LAGS_131), np.nan, dtype=np.float64), {
            "event_count_all": idx_all_n,
            "event_count_window_valid": idx_valid_n,
            "eta_valid": False,
            "eta_reason": "no_window_valid_events",
        }

    windows = []
    for event_i in idx:
        w = y[event_i - ETA_PRE_131 : event_i + ETA_POST_131 + 1]
        if w.size == len(ETA_LAGS_131):
            windows.append(w)

    if not windows:
        return np.full(len(ETA_LAGS_131), np.nan, dtype=np.float64), {
            "event_count_all": idx_all_n,
            "event_count_window_valid": idx_valid_n,
            "eta_valid": False,
            "eta_reason": "no_complete_windows",
        }

    W = np.vstack(windows).astype(np.float64)

    finite_per_lag = np.isfinite(W).sum(axis=0)
    with np.errstate(invalid="ignore"):
        curve = np.nanmean(W, axis=0)

    # Baseline-correct using pre-event lags if possible.
    pre_mask = ETA_LAGS_131 < 0
    if np.any(pre_mask) and np.isfinite(curve[pre_mask]).any():
        baseline = float(np.nanmean(curve[pre_mask]))
        curve = curve - baseline
    else:
        baseline = 0.0

    return curve.astype(np.float64), {
        "event_count_all": idx_all_n,
        "event_count_window_valid": idx_valid_n,
        "eta_valid": bool(np.isfinite(curve).any()),
        "eta_reason": "",
        "eta_baseline": baseline,
        "eta_min_finite_per_lag": int(np.nanmin(finite_per_lag)) if finite_per_lag.size else 0,
        "eta_mean_finite_per_lag": float(np.nanmean(finite_per_lag)) if finite_per_lag.size else np.nan,
    }

def _eta_similarity_131(real_curve: np.ndarray, syn_curve: np.ndarray) -> dict:
    r = np.asarray(real_curve, dtype=np.float64)
    s = np.asarray(syn_curve, dtype=np.float64)

    valid = np.isfinite(r) & np.isfinite(s)
    if int(valid.sum()) < 3:
        return {
            "eta_similarity": np.nan,
            "eta_mae": np.nan,
            "eta_norm_mae": np.nan,
            "eta_corr": np.nan,
            "eta_peak_error": np.nan,
            "eta_area_error": np.nan,
            "eta_compare_valid": False,
            "eta_compare_reason": "insufficient_finite_lags",
        }

    rr = r[valid]
    ss = s[valid]

    mae = float(np.mean(np.abs(rr - ss)))
    scale = float(np.nanmean(np.abs(rr)) + np.nanstd(rr) + float(CFG.get("cell13_1_norm_eps", 1e-9)))
    norm_mae = float(mae / max(scale, float(CFG.get("cell13_1_norm_eps", 1e-9))))

    if np.std(rr) <= 1e-12 or np.std(ss) <= 1e-12:
        corr = 1.0 if np.allclose(rr, ss, equal_nan=True) else 0.0
    else:
        corr = float(np.corrcoef(rr, ss)[0, 1])

    # Similarity in [0,1-ish], with correlation and normalized amplitude error.
    corr_part = max(0.0, corr)
    amp_part = float(np.exp(-norm_mae))
    eta_similarity = float(0.5 * corr_part + 0.5 * amp_part)

    post_mask = ETA_LAGS_131 >= 0
    post_valid = valid & post_mask

    if np.any(post_valid):
        rpost = r[post_valid]
        spost = s[post_valid]
        lpost = ETA_LAGS_131[post_valid]

        r_peak_lag = int(lpost[int(np.nanargmax(np.abs(rpost)))])
        s_peak_lag = int(lpost[int(np.nanargmax(np.abs(spost)))])
        peak_error = abs(r_peak_lag - s_peak_lag)

        r_area = float(np.nansum(np.abs(rpost)))
        s_area = float(np.nansum(np.abs(spost)))
        area_error = abs(r_area - s_area) / max(r_area, float(CFG.get("cell13_1_norm_eps", 1e-9)))
    else:
        r_peak_lag = np.nan
        s_peak_lag = np.nan
        peak_error = np.nan
        area_error = np.nan

    return {
        "eta_similarity": eta_similarity,
        "eta_mae": mae,
        "eta_norm_mae": norm_mae,
        "eta_corr": corr,
        "eta_peak_error": peak_error,
        "eta_area_error": area_error,
        "real_eta_peak_lag": r_peak_lag,
        "syn_eta_peak_lag": s_peak_lag,
        "eta_compare_valid": True,
        "eta_compare_reason": "",
    }

def _eta_status_131(row: dict) -> tuple:
    sim = _safe_float_131(row.get("eta_similarity"), np.nan)
    real_events = int(_safe_float_131(row.get("real_event_count_window_valid"), 0))
    syn_events = int(_safe_float_131(row.get("syn_event_count_window_valid"), 0))
    min_events = int(CFG.get("cell13_1_min_pair_events_for_pass", 5))

    if real_events < min_events and syn_events < min_events:
        return "not_evaluable", "insufficient_real_and_synthetic_driver_events"

    if real_events < min_events:
        return "warning", "insufficient_real_driver_events"

    if syn_events < min_events:
        return "warning", "insufficient_synthetic_driver_events"

    if not np.isfinite(sim):
        return "warning", "eta_similarity_not_finite"

    if sim >= float(CFG.get("cell13_1_eta_similarity_pass", 0.75)):
        return "pass", "eta_similarity_pass"

    if sim >= float(CFG.get("cell13_1_eta_similarity_warning", 0.50)):
        return "warning", "eta_similarity_warning"

    return "fatal", "eta_similarity_fatal"

# ----------------------------------------------------------
# 3) Validate upstream Cell 13.0 contract
# ----------------------------------------------------------
contract130_131 = CELL130_Q4_COUPLING_INPUT_CONTRACT
if not isinstance(contract130_131, dict):
    raise RuntimeError("[Cell13.1] CELL130_Q4_COUPLING_INPUT_CONTRACT is not a dict.")

version130_131 = str(contract130_131.get("version", ""))
if "cell13_0_q4_cross_modal_coupling_input_contract_v1_1" not in version130_131:
    raise RuntimeError(
        "[Cell13.1] Unexpected Cell 13.0 contract version. "
        f"Expected v1.1 schema-driven pair registry, got: {version130_131}"
    )

strict130_131 = contract130_131.get("strict_contract", {})
if not bool(strict130_131.get("TEST_real_values_used_for_QA_reference", False)):
    raise RuntimeError("[Cell13.1] Cell 13.0 contract does not declare TEST references as QA-only.")

if bool(strict130_131.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell13.1] Cell 13.0 contract indicates synthetic values were mutated.")

if bool(strict130_131.get("selection_done_here", True)):
    raise RuntimeError("[Cell13.1] Cell 13.0 contract indicates selection was done.")

if bool(strict130_131.get("generator_fit_done_here", True)):
    raise RuntimeError("[Cell13.1] Cell 13.0 contract indicates generator fitting was done.")

if bool(strict130_131.get("TEST_activity_used_for_pair_selection", True)):
    raise RuntimeError(
        "[Cell13.1] Cell 13.0 contract indicates TEST activity was used for pair selection."
    )

if not bool(strict130_131.get("pair_registry_schema_driven", False)):
    raise RuntimeError("[Cell13.1] Cell 13.0 contract does not declare schema-driven pair registry.")

q3_status_131 = (
    contract130_131.get("quality_status_carried_forward", {})
    .get("q3_observability", {})
    if isinstance(contract130_131.get("quality_status_carried_forward", {}), dict)
    else {}
)
q3_publication_blocker_n_131 = int(
    contract130_131.get("quality_status_carried_forward", {})
    .get("q3_publication_blocker_n", 0)
    if isinstance(contract130_131.get("quality_status_carried_forward", {}), dict)
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
            f"[Cell13.1] {frame_name} row mismatch: got={len(frame)} expected={N_TE}"
        )
    frame.index = df_te.index
    frame.columns = frame.columns.astype(str)

pairs_df["driver_col"] = pairs_df["driver_col"].astype(str)
pairs_df["protocol_col"] = pairs_df["protocol_col"].astype(str)

# Pair eligibility.
required_pair_cols = ["pair_id", "driver_col", "protocol_col", "protocol_tier"]
missing_pair_cols = [c for c in required_pair_cols if c not in pairs_df.columns]
if missing_pair_cols:
    raise RuntimeError(f"[Cell13.1] Pair registry missing required columns: {missing_pair_cols}")

pairs_df = pairs_df[pairs_df.get("eligible_for_13_1_eta", True).astype(bool)].copy()

if len(pairs_df) == 0:
    raise RuntimeError("[Cell13.1] No ETA-eligible driver/protocol pairs.")

if "TEST_real_values_used_for_QA_reference" in pairs_df.columns:
    # Expected: pair registry acknowledges QA reference use, but does not imply selection by TEST activity.
    pass

if "activity_stats_used_for_pair_selection" in pairs_df.columns:
    bad_activity_selected = pairs_df[
        pairs_df["activity_stats_used_for_pair_selection"].fillna(False).astype(bool)
    ]
    if len(bad_activity_selected):
        raise RuntimeError(
            "[Cell13.1] Pair registry contains pairs selected using TEST activity statistics. "
            f"Rows={len(bad_activity_selected)}"
        )

# Validate pair columns.
missing_driver_syn = sorted(set(pairs_df["driver_col"]) - set(IOT_FULL_SYN_TEST_130.columns))
missing_driver_real = sorted(set(pairs_df["driver_col"]) - set(IOT_REAL_TEST_REF_130.columns))
missing_protocol_syn = sorted(set(pairs_df["protocol_col"]) - set(PROTOCOL_SYN_TEST_130.columns))
missing_protocol_real = sorted(set(pairs_df["protocol_col"]) - set(PROTOCOL_REAL_TEST_REF_130.columns))

if missing_driver_syn or missing_driver_real or missing_protocol_syn or missing_protocol_real:
    raise RuntimeError(
        "[Cell13.1] Missing columns for ETA QA. "
        f"missing_driver_syn={missing_driver_syn[:10]} | "
        f"missing_driver_real={missing_driver_real[:10]} | "
        f"missing_protocol_syn={missing_protocol_syn[:10]} | "
        f"missing_protocol_real={missing_protocol_real[:10]}"
    )

# ----------------------------------------------------------
# 5) Compute ETA metrics
# ----------------------------------------------------------
metric_rows = []
curve_store = {}

log(
    "[Cell13.1] Computing ETA coupling metrics | "
    f"pairs={len(pairs_df)} | lags=[{-ETA_PRE_131}, +{ETA_POST_131}]"
)

# Cache arrays and ETA curves to avoid repeated driver/protocol conversions.
driver_real_cache = {}
driver_syn_cache = {}
protocol_real_cache = {}
protocol_syn_cache = {}
eta_real_cache = {}
eta_syn_cache = {}

def _get_driver_real_131(col):
    if col not in driver_real_cache:
        driver_real_cache[col] = _to_num_array_131(IOT_REAL_TEST_REF_130, col)
    return driver_real_cache[col]

def _get_driver_syn_131(col):
    if col not in driver_syn_cache:
        driver_syn_cache[col] = _to_num_array_131(IOT_FULL_SYN_TEST_130, col)
    return driver_syn_cache[col]

def _get_protocol_real_131(col):
    if col not in protocol_real_cache:
        protocol_real_cache[col] = _to_num_array_131(PROTOCOL_REAL_TEST_REF_130, col)
    return protocol_real_cache[col]

def _get_protocol_syn_131(col):
    if col not in protocol_syn_cache:
        protocol_syn_cache[col] = _to_num_array_131(PROTOCOL_SYN_TEST_130, col)
    return protocol_syn_cache[col]

for i, r in enumerate(pairs_df.itertuples(index=False), start=1):
    if i == 1 or i % 50 == 0 or i == len(pairs_df):
        log(f"[Cell13.1] progress {i}/{len(pairs_df)}")

    row = r._asdict()

    pair_id = str(row.get("pair_id", f"q4_pair_{i:05d}"))
    dcol = str(row["driver_col"])
    pcol = str(row["protocol_col"])
    tier = str(row["protocol_tier"])

    d_real = _get_driver_real_131(dcol)
    d_syn = _get_driver_syn_131(dcol)
    p_real = _get_protocol_real_131(pcol)
    p_syn = _get_protocol_syn_131(pcol)

    real_key = (dcol, pcol, "real")
    syn_key = (dcol, pcol, "syn")

    if real_key not in eta_real_cache:
        real_curve, real_meta = _eta_curve_131(d_real, p_real)
        eta_real_cache[real_key] = (real_curve, real_meta)
    else:
        real_curve, real_meta = eta_real_cache[real_key]

    if syn_key not in eta_syn_cache:
        syn_curve, syn_meta = _eta_curve_131(d_syn, p_syn)
        eta_syn_cache[syn_key] = (syn_curve, syn_meta)
    else:
        syn_curve, syn_meta = eta_syn_cache[syn_key]

    sim = _eta_similarity_131(real_curve, syn_curve)

    metric_row = {
        "pair_id": pair_id,
        "driver_col": dcol,
        "driver_entity": str(row.get("driver_entity", "")),
        "protocol_col": pcol,
        "protocol_tier": tier,
        "semantic_hint": str(row.get("semantic_hint", "")),
        "eta_pre_window_seconds": int(ETA_PRE_131),
        "eta_post_window_seconds": int(ETA_POST_131),
        "real_event_count_all": int(real_meta.get("event_count_all", 0)),
        "real_event_count_window_valid": int(real_meta.get("event_count_window_valid", 0)),
        "syn_event_count_all": int(syn_meta.get("event_count_all", 0)),
        "syn_event_count_window_valid": int(syn_meta.get("event_count_window_valid", 0)),
        "real_eta_valid": bool(real_meta.get("eta_valid", False)),
        "syn_eta_valid": bool(syn_meta.get("eta_valid", False)),
        "real_eta_reason": str(real_meta.get("eta_reason", "")),
        "syn_eta_reason": str(syn_meta.get("eta_reason", "")),
        "real_eta_baseline": real_meta.get("eta_baseline", np.nan),
        "syn_eta_baseline": syn_meta.get("eta_baseline", np.nan),
        **sim,
        "TEST_real_values_used_for_QA_reference": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
    }

    status, reason = _eta_status_131(metric_row)
    metric_row["eta_status"] = status
    metric_row["eta_status_reason"] = reason
    metric_row["eta_publication_blocker"] = bool(status == "fatal")

    metric_rows.append(metric_row)

    if bool(CFG.get("cell13_1_store_eta_curves", True)):
        curve_store[f"{pair_id}__real"] = real_curve.astype(np.float32)
        curve_store[f"{pair_id}__syn"] = syn_curve.astype(np.float32)

eta_metrics_df = pd.DataFrame(metric_rows)

if len(eta_metrics_df) != len(pairs_df):
    raise RuntimeError(
        f"[Cell13.1] ETA metrics row mismatch: got={len(eta_metrics_df)} expected={len(pairs_df)}"
    )

# ----------------------------------------------------------
# 6) Summaries
# ----------------------------------------------------------
tier_summary_rows = []

for tier, sub in eta_metrics_df.groupby("protocol_tier"):
    sub = sub.copy()
    evaluable = sub[sub["eta_status"].astype(str).ne("not_evaluable")]

    tier_summary_rows.append({
        "protocol_tier": tier,
        "pairs_total": int(len(sub)),
        "pairs_evaluable": int(len(evaluable)),
        "pass_n": int((sub["eta_status"].astype(str) == "pass").sum()),
        "warning_n": int((sub["eta_status"].astype(str) == "warning").sum()),
        "fatal_n": int((sub["eta_status"].astype(str) == "fatal").sum()),
        "not_evaluable_n": int((sub["eta_status"].astype(str) == "not_evaluable").sum()),
        "eta_pair_pass_rate": float(
            (evaluable["eta_status"].astype(str) == "pass").mean()
        ) if len(evaluable) else np.nan,
        "mean_eta_similarity": float(pd.to_numeric(evaluable["eta_similarity"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "median_eta_similarity": float(pd.to_numeric(evaluable["eta_similarity"], errors="coerce").median()) if len(evaluable) else np.nan,
        "mean_eta_norm_mae": float(pd.to_numeric(evaluable["eta_norm_mae"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_eta_peak_error": float(pd.to_numeric(evaluable["eta_peak_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "publication_blocker_n": int(sub["eta_publication_blocker"].fillna(False).astype(bool).sum()),
    })

eta_tier_summary_df = pd.DataFrame(tier_summary_rows)

status_counts = (
    eta_metrics_df["eta_status"].astype(str).value_counts().sort_index().to_dict()
)

evaluable_df = eta_metrics_df[eta_metrics_df["eta_status"].astype(str).ne("not_evaluable")].copy()

metric_summary = {
    "pairs_total": int(len(eta_metrics_df)),
    "pairs_evaluable": int(len(evaluable_df)),
    "status_counts": status_counts,
    "publication_blocker_n": int(eta_metrics_df["eta_publication_blocker"].fillna(False).astype(bool).sum()),
    "eta_pair_pass_rate": float(
        (evaluable_df["eta_status"].astype(str) == "pass").mean()
    ) if len(evaluable_df) else np.nan,
    "mean_eta_similarity": float(pd.to_numeric(evaluable_df["eta_similarity"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "median_eta_similarity": float(pd.to_numeric(evaluable_df["eta_similarity"], errors="coerce").median()) if len(evaluable_df) else np.nan,
    "mean_eta_norm_mae": float(pd.to_numeric(evaluable_df["eta_norm_mae"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_eta_peak_error": float(pd.to_numeric(evaluable_df["eta_peak_error"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
    "mean_eta_area_error": float(pd.to_numeric(evaluable_df["eta_area_error"], errors="coerce").mean()) if len(evaluable_df) else np.nan,
}

# ----------------------------------------------------------
# 7) Save outputs
# ----------------------------------------------------------
eta_metrics_df.to_csv(eta_pair_metrics_csv, index=False)
eta_tier_summary_df.to_csv(eta_tier_summary_csv, index=False)

if bool(CFG.get("cell13_1_store_eta_curves", True)):
    curve_store["lags"] = ETA_LAGS_131.astype(np.int64)
    np.savez_compressed(eta_curve_npz, **curve_store)

contract = {
    "cell": "13.1",
    "version": CELL131_VERSION,
    "role": "Q4_event_triggered_average_driver_to_protocol_QA",
    "quality_dimension": "Q4_cross_modal_consistency",
    "test_rows": int(N_TE),
    "eta_window": {
        "pre_seconds": int(ETA_PRE_131),
        "post_seconds": int(ETA_POST_131),
        "lags": ETA_LAGS_131.tolist(),
    },
    "protocol_transform": str(CFG.get("cell13_1_protocol_transform", "log1p_abs_signed")),
    "metric_summary": metric_summary,
    "tier_summary": eta_tier_summary_df.to_dict("records"),
    "cell13_0_contract_version_seen": version130_131,
    "pair_registry_selection_policy_seen": contract130_131.get("pair_registry", {}).get("selection_policy", ""),
    "TEST_activity_used_for_pair_selection": False,
    "q3_publication_blocker_n_carried_forward": int(q3_publication_blocker_n_131),
    "q3_observability_status_carried_forward": q3_status_131,
    "thresholds": {
        "eta_similarity_pass": float(CFG.get("cell13_1_eta_similarity_pass", 0.75)),
        "eta_similarity_warning": float(CFG.get("cell13_1_eta_similarity_warning", 0.50)),
        "min_pair_events_for_pass": int(CFG.get("cell13_1_min_pair_events_for_pass", 5)),
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
        "eta_pair_metrics_csv": eta_pair_metrics_csv,
        "eta_tier_summary_csv": eta_tier_summary_csv,
        "eta_curve_npz": eta_curve_npz if bool(CFG.get("cell13_1_store_eta_curves", True)) else "",
        "eta_contract_json": eta_contract_json,
        "eta_contract_canonical_json": eta_contract_canonical_json,
        "eta_manifest_json": eta_manifest_json,
    },
}

_write_json_131(eta_contract_json, contract)
_write_json_131(eta_contract_canonical_json, contract)

manifest = {
    "cell": "13.1",
    "version": CELL131_VERSION,
    "created_outputs": contract["outputs"],
    "quality_dimension": "Q4_cross_modal_consistency",
    "metric_summary": metric_summary,
    "q3_publication_blocker_n_carried_forward": int(q3_publication_blocker_n_131),
    "strict_contract": contract["strict_contract"],
}

_write_json_131(eta_manifest_json, manifest)

hashes = {
    "eta_pair_metrics_csv_sha256": _sha256_file_131(eta_pair_metrics_csv),
    "eta_tier_summary_csv_sha256": _sha256_file_131(eta_tier_summary_csv),
    "eta_contract_json_sha256": _sha256_file_131(eta_contract_json),
    "eta_contract_canonical_json_sha256": _sha256_file_131(eta_contract_canonical_json),
    "eta_manifest_json_sha256": _sha256_file_131(eta_manifest_json),
}

if bool(CFG.get("cell13_1_store_eta_curves", True)) and os.path.exists(eta_curve_npz):
    hashes["eta_curve_npz_sha256"] = _sha256_file_131(eta_curve_npz)

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_131(eta_contract_json, contract)
_write_json_131(eta_contract_canonical_json, contract)
_write_json_131(eta_manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals for 13.2+
# ----------------------------------------------------------
globals()["CELL131_VERSION"] = CELL131_VERSION
globals()["CELL13_1_ETA_PAIR_METRICS_DF"] = eta_metrics_df
globals()["CELL13_1_ETA_TIER_SUMMARY_DF"] = eta_tier_summary_df
globals()["CELL13_1_ETA_CONTRACT"] = contract
globals()["CELL13_1_ETA_PAIR_METRICS_CSV"] = eta_pair_metrics_csv
globals()["CELL13_1_ETA_TIER_SUMMARY_CSV"] = eta_tier_summary_csv
globals()["CELL13_1_ETA_CURVE_NPZ"] = eta_curve_npz
globals()["CELL13_1_ETA_CONTRACT_JSON"] = eta_contract_json
globals()["CELL13_1_ETA_CONTRACT_CANONICAL_JSON"] = eta_contract_canonical_json
globals()["CELL13_1_ETA_MANIFEST_JSON"] = eta_manifest_json

log(
    "[Cell13.1] ETA coupling QA complete | "
    f"pairs_total={metric_summary['pairs_total']} | "
    f"pairs_evaluable={metric_summary['pairs_evaluable']} | "
    f"pass_rate={metric_summary['eta_pair_pass_rate']:.6f} | "
    f"mean_eta_similarity={metric_summary['mean_eta_similarity']:.6f} | "
    f"publication_blocker_n={metric_summary['publication_blocker_n']}"
)
log(f"[Cell13.1] ETA status counts | {status_counts}")
log(
    "[Cell13.1] ETA tier summary | "
    f"{eta_tier_summary_df.to_dict('records')}"
)
log(f"[Cell13.1] Saved ETA pair metrics: {eta_pair_metrics_csv} | rows={len(eta_metrics_df)}")
log(f"[Cell13.1] Saved ETA tier summary: {eta_tier_summary_csv} | rows={len(eta_tier_summary_df)}")
log(f"[Cell13.1] Saved canonical ETA contract: {eta_contract_canonical_json}")
if bool(CFG.get("cell13_1_store_eta_curves", True)):
    log(f"[Cell13.1] Saved ETA curves: {eta_curve_npz}")
log(
    "[Cell13.1] Contract flags | "
    "TEST_real_values_used_for_QA_reference=True | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | TEST_activity_used_for_pair_selection=False | pair_registry_schema_driven=True"
)
log("--- END: Cell 13.1 - Q4 event-triggered average QA (v1.1 contract-hardened strict) ---")

gc.collect()