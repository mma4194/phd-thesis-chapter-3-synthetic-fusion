# ==========================================================
# CELL 10.8 — VAL-only generator portfolio comparison + selection
# v3.6-THESIS STUDY-THESIS-GRADE
#       + PURE VAL-ONLY SELECTOR
#       + NO IN-SELECTOR CANDIDATE GENERATION
#       + A0 DIAGNOSTIC CONTROL, NOT SELECTION BASELINE
#       + A1 SYNTHETIC-MASK BASELINE FOR A2 SELECTION
#       + REAL-VAL ∩ A1-VAL-MASK EVALUATION CONTRACT
#       + NO TEST VALUE READS
#       + NO TEST RESCUE MATERIALIZATION
#       + BOUNDED METRICS FOR RUNTIME SAFETY
#       + CANDIDATE-ID-SAFE REGISTRY HANDLING
#       + PHYSICAL CANDIDATE DEDUPLICATION
#       + REGISTRY-DEFINED CUSTOM CANDIDATE ELIGIBILITY
#       + COMPLETE RICH SELECTION REGISTRY
#       + Q1/Q2-ALIGNED METRICS:
#           - support Jaccard / support precision / support recall
#           - activity-rate error
#           - transition-rate error
#           - burst-rate error
#           - lag-1 autocorrelation error
#           - multi-lag autocorrelation MAE
#           - activity run-length KS
#           - value/dwell run-length KS
#       + TIER-SPECIFIC Q1/Q2 SCORE WEIGHTS
#
# Fixes:
#   P-SEL-02:
#     Earlier Cell 10.8 scoring did not sufficiently align with final Q2
#     protocol failures, especially multi-lag autocorrelation and run-length
#     morphology. This version adds Q1/Q2 component scores and tier-weighted
#     selection logic.
#
#   P-OTA-02:
#     Keeps v3.3 fix: custom registry-defined candidates such as
#     ota_burst_runlength_active_window are eligible and compared.
#
# Scientific boundary:
# - TRAIN:
#     used by upstream candidate cells. This selector reads TRAIN-derived
#     profiles only for eligibility metadata; it does not generate candidates.
# - VAL:
#     used for candidate comparison and generator selection.
# - TEST:
#     never read for values, metrics, fitting, thresholding, eligibility, or
#     selection. TEST candidate paths may be metadata-validated only.
#
# A0/A1/A2 semantics:
# - A0_VAL:
#     TRAIN-only value baseline under real VAL masks.
#     Diagnostic/control only. Never selected.
# - A1_VAL:
#     TRAIN-only value baseline under synthetic VAL masks.
#     Selection baseline for A2.
# - A2:
#     VAL-selected generator portfolio, compared against A1_VAL only.
# ==========================================================

log("--- START: Cell 10.8 — VAL-only portfolio comparison and selection (v3.6 registry-only pure VAL selector, Q1/Q2-aligned STUDY-THESIS) ---")

import os
import re
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
need = [
    "CFG", "log",
    "df_tr", "df_va", "df_te",
    "PROTO_VALUE_COLS",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.8] Missing required globals: {missing}. Run Cells 1–10.7 first.")

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
REPDIR = os.path.join(OUTDIR, "reports")
CANDDIR = os.path.join(OUTDIR, "portfolio_candidates")

os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(REPDIR, exist_ok=True)
os.makedirs(CANDDIR, exist_ok=True)
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")
os.makedirs(CONTRACT_DIR, exist_ok=True)

seed = int(CFG.get("seed", 1337))
rng = np.random.default_rng(seed + 1080)

N_VAL = int(len(df_va))
N_TEST = int(len(df_te))

if N_VAL <= 0:
    raise RuntimeError("[Cell10.8] df_va is empty.")
if N_TEST <= 0:
    raise RuntimeError("[Cell10.8] df_te is empty.")
if len(df_tr) <= 0:
    raise RuntimeError("[Cell10.8] df_tr is empty.")

PROTO_COLS = [
    str(c)
    for c in list(globals().get("CELL10_PROTO_COLS", PROTO_VALUE_COLS))
    if str(c) in df_va.columns
]
if not PROTO_COLS:
    raise RuntimeError("[Cell10.8] No protocol columns are present in df_va.")


for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[Cell10.8] Clean VAL selector forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.8] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )


# ----------------------------------------------------------
# 0.1) Fixed selection policy
# ----------------------------------------------------------
CFG["cell10_8_selection_split"] = "VAL"
CFG["cell10_8_test_used_for_selection"] = False
CFG["cell10_8_test_values_read"] = False
CFG["cell10_8_materialize_test_rescue"] = False
CFG["cell10_8_selection_baseline"] = "A1_VAL_baseline"
CFG["cell10_8_a0_used_for_selection"] = False

MIN_SCORE_GAIN_FRAC = float(CFG.get("cell10_min_score_gain_frac", 0.025))
MIN_KS_GAIN = float(CFG.get("cell10_min_ks_gain", 0.003))
MAX_WASSERSTEIN_REGRESS_FRAC = float(CFG.get("cell10_max_wasserstein_regress_frac", 0.05))

# v3.5 Q1/Q2 gate additions.
MIN_Q2_GAIN_FRAC = float(CFG.get("cell10_min_q2_gain_frac", 0.015))
MAX_Q1_REGRESS_FRAC = float(CFG.get("cell10_max_q1_regress_frac", 0.10))
MAX_SUPPORT_JACCARD_REGRESS = float(CFG.get("cell10_max_support_jaccard_regress", 0.10))
MIN_TEMPORAL_ONLY_SCORE_GAIN_FRAC = float(CFG.get("cell10_min_temporal_only_score_gain_frac", 0.04))

BORDERLINE_REL_MARGIN = float(CFG.get("cell10_borderline_relative_margin_over_ks_threshold", 0.10))
NEAR_BORDERLINE_REL_MARGIN = float(CFG.get("cell10_near_borderline_relative_margin_over_ks_threshold", 0.25))

VAL_BLOCK_COUNT = int(CFG.get("cell10_val_block_count", 8))
VAL_BLOCK_REQUIRED_WIN_FRAC = float(CFG.get("cell10_val_block_required_win_frac", 0.75))
VAL_BLOCK_MAX_BAD_FRAC = float(CFG.get("cell10_val_block_max_bad_frac", 0.25))
VAL_BLOCK_MIN_N = int(CFG.get("cell10_val_block_min_n", 500))

ENABLE_MARKOV_RESCUE = False
if bool(CFG.get("cell10_enable_markov_rescue_in_selector", False)):
    raise RuntimeError(
        "[Cell10.8] In-selector Markov rescue is disabled in the canonical THESIS pipeline. "
        "Move any rescue generator to a separate TRAIN-fitted candidate cell before Cell 10.8."
    )
MARKOV_RESCUE_MIN_TRAIN_N = int(CFG.get("cell10_markov_rescue_min_train_n", 100))
MARKOV_RESCUE_MAX_UNIQUE = int(CFG.get("cell10_markov_rescue_max_unique", 64))
MARKOV_RESCUE_SPARSE_NZ_RATE_MAX = float(CFG.get("cell10_markov_rescue_sparse_nz_rate_max", 0.50))
MARKOV_RESCUE_LOW_CARDINALITY_FRAC = float(CFG.get("cell10_markov_rescue_low_cardinality_frac", 0.05))

MAX_METRIC_N = int(CFG.get("cell10_8_max_metric_n", 50_000))
MAX_KS_GRID = int(CFG.get("cell10_8_max_ks_grid", 4096))
MAX_WASSERSTEIN_POINTS = int(CFG.get("cell10_8_max_wasserstein_points", 2048))
PROGRESS_EVERY_COLS = int(CFG.get("cell10_8_progress_every_cols", 5))

# Multi-lag autocorrelation lags. Keep bounded and reviewer-defensible.
MULTILAG_LAGS = list(CFG.get("cell10_8_multilag_lags", [1, 5, 10, 30, 60, 300]))
MULTILAG_LAGS = sorted(set(int(x) for x in MULTILAG_LAGS if int(x) > 0))

# Tier-specific score weights.
DEFAULT_TIER_WEIGHTS = {
    "router": {"q1": 0.45, "q2": 0.55},
    "ota":    {"q1": 0.30, "q2": 0.70},
    "zigbee": {"q1": 0.40, "q2": 0.60},
    "zwave":  {"q1": 0.50, "q2": 0.50},
    "unknown": {"q1": 0.50, "q2": 0.50},
}
USER_TIER_WEIGHTS = dict(CFG.get("cell10_8_score_weights_by_tier", {}) or {})

TIER_WEIGHTS = dict(DEFAULT_TIER_WEIGHTS)
for tier, w in USER_TIER_WEIGHTS.items():
    if not isinstance(w, dict):
        continue
    q1 = float(w.get("q1", TIER_WEIGHTS.get(tier, DEFAULT_TIER_WEIGHTS["unknown"])["q1"]))
    q2 = float(w.get("q2", TIER_WEIGHTS.get(tier, DEFAULT_TIER_WEIGHTS["unknown"])["q2"]))
    s = max(q1 + q2, 1e-12)
    TIER_WEIGHTS[str(tier)] = {"q1": q1 / s, "q2": q2 / s}

if MIN_SCORE_GAIN_FRAC < 0:
    raise RuntimeError("[Cell10.8] MIN_SCORE_GAIN_FRAC must be >= 0.")
if MIN_KS_GAIN <= 0:
    raise RuntimeError("[Cell10.8] MIN_KS_GAIN must be > 0.")
if MAX_WASSERSTEIN_REGRESS_FRAC < 0:
    raise RuntimeError("[Cell10.8] MAX_WASSERSTEIN_REGRESS_FRAC must be >= 0.")
if MIN_Q2_GAIN_FRAC < 0:
    raise RuntimeError("[Cell10.8] MIN_Q2_GAIN_FRAC must be >= 0.")
if MAX_Q1_REGRESS_FRAC < 0:
    raise RuntimeError("[Cell10.8] MAX_Q1_REGRESS_FRAC must be >= 0.")
if BORDERLINE_REL_MARGIN < 0:
    raise RuntimeError("[Cell10.8] BORDERLINE_REL_MARGIN must be >= 0.")
if NEAR_BORDERLINE_REL_MARGIN < BORDERLINE_REL_MARGIN:
    raise RuntimeError("[Cell10.8] NEAR_BORDERLINE_REL_MARGIN must be >= BORDERLINE_REL_MARGIN.")
if VAL_BLOCK_COUNT < 2:
    raise RuntimeError("[Cell10.8] VAL_BLOCK_COUNT must be >= 2.")
if not (0.0 < VAL_BLOCK_REQUIRED_WIN_FRAC <= 1.0):
    raise RuntimeError("[Cell10.8] VAL_BLOCK_REQUIRED_WIN_FRAC must be in (0, 1].")
if not (0.0 <= VAL_BLOCK_MAX_BAD_FRAC <= 1.0):
    raise RuntimeError("[Cell10.8] VAL_BLOCK_MAX_BAD_FRAC must be in [0, 1].")
if MARKOV_RESCUE_MIN_TRAIN_N <= 0:
    raise RuntimeError("[Cell10.8] MARKOV_RESCUE_MIN_TRAIN_N must be positive.")
if MARKOV_RESCUE_MAX_UNIQUE < 2:
    raise RuntimeError("[Cell10.8] MARKOV_RESCUE_MAX_UNIQUE must be >= 2.")
if MAX_METRIC_N < 1000:
    raise RuntimeError("[Cell10.8] cell10_8_max_metric_n must be >= 1000.")
if MAX_KS_GRID < 256:
    raise RuntimeError("[Cell10.8] cell10_8_max_ks_grid must be >= 256.")
if MAX_WASSERSTEIN_POINTS < 256:
    raise RuntimeError("[Cell10.8] cell10_8_max_wasserstein_points must be >= 256.")

_ACTIVE_MASK_CACHE = {}
_TRAIN_VALUES_CACHE = {}
_TRAIN_SUPPORT_PROFILE_CACHE = {}
_A1_EVAL_MASK_CACHE = {}


# ----------------------------------------------------------
# 1) Required artifact loading
# ----------------------------------------------------------
a0_val_path = os.path.join(CANDDIR, "A0_VAL_baseline.parquet")
a0_test_path = os.path.join(CANDDIR, "A0_TEST_baseline.parquet")
a1_val_path = os.path.join(CANDDIR, "A1_VAL_baseline.parquet")
a1_test_path = os.path.join(CANDDIR, "A1_TEST_baseline.parquet")

baseline_manifest_path = os.path.join(ARTDIR, "cell10_a0_a1_baseline_manifest.json")
portfolio_contract_path = os.path.join(ARTDIR, "cell10_portfolio_contract.json")
family_contract_path = os.path.join(ARTDIR, "cell10_protocol_family_contract.csv")
candidate_registry_path = os.path.join(ARTDIR, "cell10_portfolio_candidate_registry.json")

for p, label in [
    (a0_val_path, "A0 VAL baseline"),
    (a0_test_path, "A0 TEST baseline"),
    (a1_val_path, "A1 VAL baseline"),
    (a1_test_path, "A1 TEST baseline"),
    (baseline_manifest_path, "A0/A1 baseline manifest"),
    (portfolio_contract_path, "portfolio contract"),
    (family_contract_path, "family contract"),
]:
    if not os.path.exists(p):
        raise RuntimeError(f"[Cell10.8] Missing {label}: {p}. Run Cell 10.1 first.")

A0_VAL = pd.read_parquet(a0_val_path)
A1_VAL = pd.read_parquet(a1_val_path)

if len(A0_VAL) != N_VAL:
    raise RuntimeError(f"[Cell10.8] A0_VAL row mismatch: {len(A0_VAL)} vs df_va={N_VAL}")
if len(A1_VAL) != N_VAL:
    raise RuntimeError(f"[Cell10.8] A1_VAL row mismatch: {len(A1_VAL)} vs df_va={N_VAL}")

with open(baseline_manifest_path, "r", encoding="utf-8") as f:
    baseline_manifest = json.load(f)

with open(portfolio_contract_path, "r", encoding="utf-8") as f:
    portfolio_contract = json.load(f)

family_contract_df = pd.read_csv(family_contract_path)
if "col" not in family_contract_df.columns:
    raise RuntimeError("[Cell10.8] family contract missing required column 'col'.")

family_by_col = {
    str(r["col"]): dict(r)
    for _, r in family_contract_df.iterrows()
}

manifest_outputs = dict(baseline_manifest.get("outputs", {}))
if os.path.realpath(str(manifest_outputs.get("A1_VAL_baseline", a1_val_path))) != os.path.realpath(a1_val_path):
    log("[Cell10.8][WARN] Manifest A1_VAL path differs from canonical expected path; using canonical file path.")

log(
    "[Cell10.8] Loaded controlled baselines | "
    f"A0_VAL={A0_VAL.shape} | A1_VAL={A1_VAL.shape} | selection_baseline=A1_VAL"
)


# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize(obj.to_dict())
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


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2, sort_keys=True)


def _sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parquet_num_rows(path: str):
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _parquet_columns(path: str):
    try:
        import pyarrow.parquet as pq
        return list(map(str, pq.ParquetFile(path).schema.names))
    except Exception:
        return None


def _tier_of(col: str):
    c = str(col)
    if c.startswith("router__"):
        return "router"
    if c.startswith("ota__"):
        return "ota"
    if c.startswith("zigbee__"):
        return "zigbee"
    if c.startswith("zwave__"):
        return "zwave"
    return "unknown"


def _obs_col_for_tier(tier: str) -> str:
    return f"{tier}__obs_present"


def _active_mask_from_df(df_part: pd.DataFrame, tier: str) -> np.ndarray:
    key = (id(df_part), str(tier))
    if key in _ACTIVE_MASK_CACHE:
        return _ACTIVE_MASK_CACHE[key]

    c = _obs_col_for_tier(tier)
    if c in df_part.columns:
        v = pd.to_numeric(df_part[c], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        out = (v > 0.5)
    elif tier == "zwave":
        out = np.zeros(len(df_part), dtype=bool)
    else:
        out = np.ones(len(df_part), dtype=bool)

    out = np.asarray(out, dtype=bool)
    _ACTIVE_MASK_CACHE[key] = out
    return out


def _finite_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _bounded_finite_1d(x, max_n: int = None) -> np.ndarray:
    arr = _finite_1d(x)
    if max_n is None:
        max_n = MAX_METRIC_N
    max_n = int(max_n)

    if arr.size <= max_n:
        return arr

    idx = np.linspace(0, arr.size - 1, num=max_n)
    idx = np.unique(np.round(idx).astype(np.int64))
    if idx.size > max_n:
        idx = idx[:max_n]
    return arr[idx]


COUNTLIKE_RX = re.compile(
    r"(_total$|_pkt$|_pkts$|_packets$|_bytes$|_syn$|_ack$|_rst$|_fin$|"
    r"_query$|_queries$|_response$|_responses$|_req$|_reply$|_unique$|"
    r"_count$|_events$|_present$|_obs_present$|_rcode\d+_.*$)",
    re.IGNORECASE,
)

UNIQUE_LIKE_RX = re.compile(
    r"(unique|uniq|dst_port_unique|ip_src_unique|ip_dst_unique|panid_unique|"
    r"src16_unique|dst16_unique|src64_unique)",
    re.IGNORECASE,
)

DENSE_AGG_RX = re.compile(
    r"(_pkt_total$|_bytes_total$|__beacon$|__probe_resp$|__pkt_total$|"
    r"__bytes_total$|__pps$|__mgmt$|__data$|__tcp_pkt$|__udp_pkt$)",
    re.IGNORECASE,
)

SPARSE_NAME_RX = re.compile(
    r"(obs_present|_present$|__channel$|__cmd$|__ack$|"
    r"assoc|auth|deauth|disassoc|probe_req|action|zcl|aps|nwk|app)",
    re.IGNORECASE,
)


def _is_countlike(col: str) -> bool:
    return bool(COUNTLIKE_RX.search(str(col)))


def _is_unique_like(col: str) -> bool:
    return bool(UNIQUE_LIKE_RX.search(str(col)))


def _is_dense_aggregate_like(col: str) -> bool:
    return bool(DENSE_AGG_RX.search(str(col)))


def _postprocess_col(col: str, x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.nan
    finite = np.isfinite(arr)
    arr[finite] = np.maximum(arr[finite], 0.0)
    if _is_countlike(col):
        arr[finite] = np.rint(arr[finite])
    return arr.astype(np.float32, copy=False)


def _values_from_df(df_part: pd.DataFrame, col: str, active: np.ndarray) -> np.ndarray:
    if col not in df_part.columns:
        return np.array([], dtype=np.float32)

    active = np.asarray(active, dtype=bool)
    if active.shape[0] != len(df_part):
        raise RuntimeError(
            f"[Cell10.8] Active mask length mismatch for {col}: "
            f"mask={active.shape[0]} df={len(df_part)}"
        )

    x = pd.to_numeric(df_part.loc[active, col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    x = x[np.isfinite(x)]

    if x.size:
        x = np.maximum(x, 0.0)
        if _is_countlike(col):
            x = np.rint(x)

    return x.astype(np.float32, copy=False)


def _train_values_cached(col: str) -> np.ndarray:
    col = str(col)
    if col in _TRAIN_VALUES_CACHE:
        return _TRAIN_VALUES_CACHE[col]

    tier = _tier_of(col)
    active = _active_mask_from_df(df_tr, tier)
    vals = _values_from_df(df_tr, col, active)
    _TRAIN_VALUES_CACHE[col] = vals
    return vals


def _real_val_active_mask(col: str) -> np.ndarray:
    return _active_mask_from_df(df_va, _tier_of(col))


def _a1_val_active_mask(col: str) -> np.ndarray:
    if col not in A1_VAL.columns:
        raise RuntimeError(f"[Cell10.8] A1_VAL missing column required for active-mask inference: {col}")
    x = pd.to_numeric(A1_VAL[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    return np.isfinite(x)


def _a0_val_active_mask(col: str) -> np.ndarray:
    if col not in A0_VAL.columns:
        raise RuntimeError(f"[Cell10.8] A0_VAL missing column required for active-mask inference: {col}")
    x = pd.to_numeric(A0_VAL[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    return np.isfinite(x)


def _selection_eval_mask(col: str) -> np.ndarray:
    if col in _A1_EVAL_MASK_CACHE:
        return _A1_EVAL_MASK_CACHE[col]

    real_active = _real_val_active_mask(col)
    a1_active = _a1_val_active_mask(col)

    if real_active.shape[0] != N_VAL or a1_active.shape[0] != N_VAL:
        raise RuntimeError(f"[Cell10.8] Evaluation mask length mismatch for {col}.")

    out = np.asarray(real_active & a1_active, dtype=bool)
    _A1_EVAL_MASK_CACHE[col] = out
    return out


def _val_real_values_for_mask(col: str, mask: np.ndarray) -> np.ndarray:
    return _values_from_df(df_va, col, mask)


def _safe_quantile(x, q):
    x = _bounded_finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


def _safe_mean(x):
    x = _bounded_finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.mean(x))


def _safe_std(x):
    x = _bounded_finite_1d(x)
    if x.size <= 1:
        return np.nan
    return float(np.std(x))


def _nonzero_rate(x):
    x = _bounded_finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.mean(x > 0))


def _activity_rate(x):
    return _nonzero_rate(x)


def _transition_rate(x):
    x = _bounded_finite_1d(x)
    if x.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x)) > 1e-9))


def _burst_rate(x):
    x = _bounded_finite_1d(x)
    if x.size < 2:
        return np.nan
    nz = x > 0
    return float(np.mean((~nz[:-1]) & (nz[1:])))


def _lag_autocorr(x, lag: int):
    x = _bounded_finite_1d(x)
    lag = int(lag)
    if lag <= 0 or x.size <= lag + 2:
        return np.nan
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _lag1_autocorr(x):
    return _lag_autocorr(x, 1)


def _multi_lag_autocorr_profile(x):
    out = {}
    for lag in MULTILAG_LAGS:
        out[int(lag)] = _lag_autocorr(x, int(lag))
    return out


def _multi_lag_autocorr_mae(a, b):
    pa = _multi_lag_autocorr_profile(a)
    pb = _multi_lag_autocorr_profile(b)
    vals = []
    for lag in MULTILAG_LAGS:
        aa = pa.get(lag, np.nan)
        bb = pb.get(lag, np.nan)
        if np.isfinite(aa) and np.isfinite(bb):
            vals.append(abs(float(aa) - float(bb)))
    return np.nan if not vals else float(np.mean(vals))


def _ks_stat_fast(a, b):
    a = np.sort(_bounded_finite_1d(a))
    b = np.sort(_bounded_finite_1d(b))

    if a.size == 0 or b.size == 0:
        return np.nan

    m = int(min(MAX_KS_GRID, max(256, min(a.size + b.size, MAX_KS_GRID))))
    qs = np.linspace(0.0, 1.0, m)

    grid_a = np.quantile(a, qs)
    grid_b = np.quantile(b, qs)
    vals = np.sort(np.unique(np.concatenate([grid_a, grid_b])))

    if vals.size == 0:
        return np.nan

    ca = np.searchsorted(a, vals, side="right") / float(a.size)
    cb = np.searchsorted(b, vals, side="right") / float(b.size)

    return float(np.max(np.abs(ca - cb)))


def _wasserstein_1d_fast(a, b, max_points: int = None):
    a = _bounded_finite_1d(a)
    b = _bounded_finite_1d(b)

    if a.size == 0 or b.size == 0:
        return np.nan

    if max_points is None:
        max_points = MAX_WASSERSTEIN_POINTS

    m = int(min(max_points, max(256, min(a.size, b.size))))
    qs = np.linspace(0.0, 1.0, m)
    aq = np.quantile(a, qs)
    bq = np.quantile(b, qs)

    return float(np.mean(np.abs(aq - bq)))


def _rounded_support(x, col: str, max_unique: int = 5000):
    x = _bounded_finite_1d(x)
    if x.size == 0:
        return set()

    if _is_countlike(col):
        y = np.rint(x).astype(np.int64)
    else:
        y = np.round(x.astype(np.float64), 6)

    u = np.unique(y)
    if u.size > max_unique:
        # For high-cardinality continuous-ish columns, use quantized quantile support.
        qs = np.linspace(0, 1, max_unique)
        u = np.unique(np.round(np.quantile(y.astype(np.float64), qs), 6))
    return set(u.tolist())


def _support_metrics(real_vals, syn_vals, col: str):
    r = _rounded_support(real_vals, col)
    s = _rounded_support(syn_vals, col)

    if not r and not s:
        return {
            "support_jaccard": np.nan,
            "support_precision": np.nan,
            "support_recall": np.nan,
            "real_support_n": 0,
            "syn_support_n": 0,
        }

    inter = len(r & s)
    union = len(r | s)
    return {
        "support_jaccard": float(inter / union) if union else np.nan,
        "support_precision": float(inter / max(1, len(s))),
        "support_recall": float(inter / max(1, len(r))),
        "real_support_n": int(len(r)),
        "syn_support_n": int(len(s)),
    }


def _run_lengths_from_binary(b):
    b = np.asarray(b, dtype=np.int8).reshape(-1)
    if b.size == 0:
        return np.array([], dtype=np.float64)
    lens = []
    start = 0
    cur = int(b[0])
    for i in range(1, b.size + 1):
        if i == b.size or int(b[i]) != cur:
            lens.append(i - start)
            if i < b.size:
                start = i
                cur = int(b[i])
    return np.asarray(lens, dtype=np.float64)


def _run_lengths_from_values(x, col: str):
    x = _bounded_finite_1d(x)
    if x.size == 0:
        return np.array([], dtype=np.float64)

    if _is_countlike(col):
        y = np.rint(x).astype(np.int64)
    else:
        y = np.round(x.astype(np.float64), 6)

    lens = []
    start = 0
    cur = y[0]
    for i in range(1, y.size + 1):
        if i == y.size or y[i] != cur:
            lens.append(i - start)
            if i < y.size:
                start = i
                cur = y[i]
    return np.asarray(lens, dtype=np.float64)


def _activity_run_length_ks(a, b):
    aa = _bounded_finite_1d(a)
    bb = _bounded_finite_1d(b)
    if aa.size == 0 or bb.size == 0:
        return np.nan
    ra = _run_lengths_from_binary((aa > 0).astype(np.int8))
    rb = _run_lengths_from_binary((bb > 0).astype(np.int8))
    return _ks_stat_fast(ra, rb)


def _value_run_length_ks(a, b, col: str):
    ra = _run_lengths_from_values(a, col)
    rb = _run_lengths_from_values(b, col)
    return _ks_stat_fast(ra, rb)


def _metric_row(real_vals, syn_vals, col: str) -> dict:
    real = _finite_1d(real_vals)
    syn = _finite_1d(syn_vals)

    if real.size == 0 or syn.size == 0:
        return {
            "n_real": int(real.size),
            "n_syn": int(syn.size),
            "ks": np.nan,
            "wasserstein": np.nan,
            "mean_abs_error": np.nan,
            "nonzero_rate_abs_error": np.nan,
            "activity_rate_abs_error": np.nan,
            "q50_abs_error": np.nan,
            "q95_abs_error": np.nan,
            "transition_rate_abs_error": np.nan,
            "burst_rate_abs_error": np.nan,
            "lag1_abs_error": np.nan,
            "multi_lag_autocorr_mae": np.nan,
            "activity_run_length_ks": np.nan,
            "value_run_length_ks": np.nan,
            "run_length_ks": np.nan,
            "support_jaccard": np.nan,
            "support_precision": np.nan,
            "support_recall": np.nan,
            "real_support_n": 0,
            "syn_support_n": 0,
            "q1_score": np.nan,
            "q2_score": np.nan,
            "combined_score": np.nan,
        }

    real_mean = _safe_mean(real)
    syn_mean = _safe_mean(syn)
    real_nz = _nonzero_rate(real)
    syn_nz = _nonzero_rate(syn)
    real_act = _activity_rate(real)
    syn_act = _activity_rate(syn)
    real_q50 = _safe_quantile(real, 0.50)
    syn_q50 = _safe_quantile(syn, 0.50)
    real_q95 = _safe_quantile(real, 0.95)
    syn_q95 = _safe_quantile(syn, 0.95)
    real_tr = _transition_rate(real)
    syn_tr = _transition_rate(syn)
    real_br = _burst_rate(real)
    syn_br = _burst_rate(syn)
    real_ac = _lag1_autocorr(real)
    syn_ac = _lag1_autocorr(syn)

    support = _support_metrics(real, syn, col)

    multi_lag_mae = _multi_lag_autocorr_mae(real, syn)
    activity_rl_ks = _activity_run_length_ks(real, syn)
    value_rl_ks = _value_run_length_ks(real, syn, col)

    # Conservative combined run-length indicator. Use max of available
    # activity-run and value-run morphology discrepancies.
    rl_vals = [x for x in [activity_rl_ks, value_rl_ks] if np.isfinite(x)]
    run_length_ks = np.nan if not rl_vals else float(max(rl_vals))

    return {
        "n_real": int(real.size),
        "n_syn": int(syn.size),
        "ks": float(_ks_stat_fast(real, syn)),
        "wasserstein": float(_wasserstein_1d_fast(real, syn)),
        "mean_abs_error": float(abs(real_mean - syn_mean)) if np.isfinite(real_mean) and np.isfinite(syn_mean) else np.nan,
        "nonzero_rate_abs_error": float(abs(real_nz - syn_nz)) if np.isfinite(real_nz) and np.isfinite(syn_nz) else np.nan,
        "activity_rate_abs_error": float(abs(real_act - syn_act)) if np.isfinite(real_act) and np.isfinite(syn_act) else np.nan,
        "q50_abs_error": float(abs(real_q50 - syn_q50)) if np.isfinite(real_q50) and np.isfinite(syn_q50) else np.nan,
        "q95_abs_error": float(abs(real_q95 - syn_q95)) if np.isfinite(real_q95) and np.isfinite(syn_q95) else np.nan,
        "transition_rate_abs_error": float(abs(real_tr - syn_tr)) if np.isfinite(real_tr) and np.isfinite(syn_tr) else np.nan,
        "burst_rate_abs_error": float(abs(real_br - syn_br)) if np.isfinite(real_br) and np.isfinite(syn_br) else np.nan,
        "lag1_abs_error": float(abs(real_ac - syn_ac)) if np.isfinite(real_ac) and np.isfinite(syn_ac) else np.nan,
        "multi_lag_autocorr_mae": float(multi_lag_mae) if np.isfinite(multi_lag_mae) else np.nan,
        "activity_run_length_ks": float(activity_rl_ks) if np.isfinite(activity_rl_ks) else np.nan,
        "value_run_length_ks": float(value_rl_ks) if np.isfinite(value_rl_ks) else np.nan,
        "run_length_ks": float(run_length_ks) if np.isfinite(run_length_ks) else np.nan,
        **support,
    }


def _score_components_from_metrics(metrics: dict, real_vals, tier: str):
    real = _finite_1d(real_vals)
    if real.size == 0:
        return {"q1_score": float("inf"), "q2_score": float("inf"), "combined_score": float("inf")}

    scale_q95 = _safe_quantile(real, 0.95)
    scale_q25 = _safe_quantile(real, 0.25)
    scale_q75 = _safe_quantile(real, 0.75)
    iqr = float(scale_q75 - scale_q25) if np.isfinite(scale_q75) and np.isfinite(scale_q25) else 0.0
    scale = max(abs(float(scale_q95)) if np.isfinite(scale_q95) else 0.0, abs(iqr), 1.0)

    def finite_or_zero(x):
        try:
            x = float(x)
            return x if np.isfinite(x) else 0.0
        except Exception:
            return 0.0

    def finite_or_one(x):
        try:
            x = float(x)
            return x if np.isfinite(x) else 1.0
        except Exception:
            return 1.0

    ks = finite_or_one(metrics.get("ks"))
    wasser_norm = finite_or_zero(metrics.get("wasserstein")) / scale
    mean_norm = finite_or_zero(metrics.get("mean_abs_error")) / scale
    q50_norm = finite_or_zero(metrics.get("q50_abs_error")) / scale
    q95_norm = finite_or_zero(metrics.get("q95_abs_error")) / scale

    support_j = metrics.get("support_jaccard", np.nan)
    support_prec = metrics.get("support_precision", np.nan)
    support_rec = metrics.get("support_recall", np.nan)

    support_j_err = 1.0 - finite_or_zero(support_j) if np.isfinite(support_j) else 0.0
    support_p_err = 1.0 - finite_or_zero(support_prec) if np.isfinite(support_prec) else 0.0
    support_r_err = 1.0 - finite_or_zero(support_rec) if np.isfinite(support_rec) else 0.0

    act_err = finite_or_zero(metrics.get("activity_rate_abs_error", metrics.get("nonzero_rate_abs_error")))
    nonzero_err = finite_or_zero(metrics.get("nonzero_rate_abs_error"))
    trans_err = finite_or_zero(metrics.get("transition_rate_abs_error"))
    burst_err = finite_or_zero(metrics.get("burst_rate_abs_error"))
    lag1_err = finite_or_zero(metrics.get("lag1_abs_error"))
    multilag_err = finite_or_zero(metrics.get("multi_lag_autocorr_mae"))
    run_ks = finite_or_zero(metrics.get("run_length_ks"))

    q1_score = float(
        1.00 * ks
        + 0.50 * wasser_norm
        + 0.10 * mean_norm
        + 0.20 * q50_norm
        + 0.25 * q95_norm
        + 0.35 * support_j_err
        + 0.20 * support_p_err
        + 0.20 * support_r_err
        + 0.30 * nonzero_err
    )

    q2_score = float(
        0.35 * act_err
        + 0.50 * trans_err
        + 0.35 * burst_err
        + 0.40 * lag1_err
        + 0.80 * multilag_err
        + 0.80 * run_ks
    )

    w = TIER_WEIGHTS.get(tier, TIER_WEIGHTS["unknown"])
    combined = float(w["q1"] * q1_score + w["q2"] * q2_score)

    return {
        "q1_score": q1_score,
        "q2_score": q2_score,
        "combined_score": combined,
    }


def _metric_row_scored(real_vals, syn_vals, col: str, tier: str):
    m = _metric_row(real_vals, syn_vals, col)
    sc = _score_components_from_metrics(m, real_vals, tier)
    m.update(sc)
    return m


def _score_from_metrics(metrics: dict, real_vals, tier: str = "unknown") -> float:
    # Backward-compatible wrapper. Use combined Q1/Q2 score.
    sc = _score_components_from_metrics(metrics, real_vals, tier)
    return float(sc["combined_score"])


def _split_blocks(n: int, k: int) -> list:
    n = int(n)
    k = int(max(1, k))
    if n <= 0:
        return []
    edges = np.linspace(0, n, num=k + 1)
    edges = np.unique(np.round(edges).astype(int))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if int(b) > int(a)]


def _block_stability_gate(real_vals, a1_vals, cand_vals, col: str, tier: str) -> tuple:
    real = _finite_1d(real_vals)
    a1 = _finite_1d(a1_vals)
    cand = _finite_1d(cand_vals)

    if min(real.size, a1.size, cand.size) < VAL_BLOCK_MIN_N:
        return False, {
            "enabled": True,
            "reason": "too_few_values_for_block_stability",
            "real_n": int(real.size),
            "a1_n": int(a1.size),
            "cand_n": int(cand.size),
            "min_n": int(VAL_BLOCK_MIN_N),
        }

    real_blocks = _split_blocks(real.size, VAL_BLOCK_COUNT)
    a1_blocks = _split_blocks(a1.size, VAL_BLOCK_COUNT)
    cand_blocks = _split_blocks(cand.size, VAL_BLOCK_COUNT)

    B = min(len(real_blocks), len(a1_blocks), len(cand_blocks))
    if B < 2:
        return False, {"enabled": True, "reason": "too_few_blocks", "blocks": int(B)}

    rows = []
    wins = 0
    bad = 0

    for bi in range(B):
        ra, rb = real_blocks[bi]
        aa, ab = a1_blocks[bi]
        ca, cb = cand_blocks[bi]

        r = real[ra:rb]
        a = a1[aa:ab]
        c = cand[ca:cb]

        if min(r.size, a.size, c.size) < max(20, VAL_BLOCK_MIN_N // max(1, VAL_BLOCK_COUNT)):
            continue

        a_metrics = _metric_row_scored(r, a, col, tier)
        c_metrics = _metric_row_scored(r, c, col, tier)

        a_score = float(a_metrics["combined_score"])
        c_score = float(c_metrics["combined_score"])
        a_q1 = float(a_metrics["q1_score"])
        c_q1 = float(c_metrics["q1_score"])
        a_q2 = float(a_metrics["q2_score"])
        c_q2 = float(c_metrics["q2_score"])

        if not all(np.isfinite(v) for v in [a_score, c_score, a_q1, c_q1, a_q2, c_q2]):
            continue

        win = bool(c_score < a_score)
        materially_bad = bool(
            c_score > a_score * (1.0 + max(0.05, MAX_Q1_REGRESS_FRAC))
            or c_q1 > a_q1 * (1.0 + max(0.05, MAX_Q1_REGRESS_FRAC))
        )

        wins += int(win)
        bad += int(materially_bad)

        rows.append({
            "block": int(bi),
            "a1_score": float(a_score),
            "candidate_score": float(c_score),
            "a1_q1_score": float(a_q1),
            "candidate_q1_score": float(c_q1),
            "a1_q2_score": float(a_q2),
            "candidate_q2_score": float(c_q2),
            "win": bool(win),
            "materially_bad": bool(materially_bad),
        })

    used = int(len(rows))
    if used < 2:
        return False, {"enabled": True, "reason": "no_usable_blocks", "blocks_used": used}

    win_frac = float(wins / used)
    bad_frac = float(bad / used)
    ok = bool(win_frac >= VAL_BLOCK_REQUIRED_WIN_FRAC and bad_frac <= VAL_BLOCK_MAX_BAD_FRAC)

    return ok, {
        "enabled": True,
        "reason": "ok" if ok else "block_stability_failed",
        "blocks_used": int(used),
        "wins": int(wins),
        "bad_blocks": int(bad),
        "win_frac": float(win_frac),
        "bad_frac": float(bad_frac),
        "required_win_frac": float(VAL_BLOCK_REQUIRED_WIN_FRAC),
        "max_bad_frac": float(VAL_BLOCK_MAX_BAD_FRAC),
        "block_rows": rows,
    }


# ----------------------------------------------------------
# 3) Candidate registry normalization / discovery
# ----------------------------------------------------------
GENERATOR_ALIASES = {
    "negative_binomial": "negative_binomial_poisson_gamma",
    "nb_poisson_gamma": "negative_binomial_poisson_gamma",
    "poisson_gamma": "negative_binomial_poisson_gamma",
    "markov": "markov_semimarkov",
    "semimarkov": "markov_semimarkov",
    "markov_semimarkov_rescue": "markov_semimarkov",
    "gaussian_copula": "gaussian_copula",
    "copula": "gaussian_copula",
    "ddpm": "ddpm",
    "ctgan": "ctgan_tvae",
    "tvae": "ctgan_tvae",
    "ctgan_tvae": "ctgan_tvae",
    "timegan": "timegan_sequence_gan",
    "sequence_gan": "timegan_sequence_gan",
    "timegan_sequence_gan": "timegan_sequence_gan",
    "ota_burst_runlength_active_window": "ota_burst_runlength_active_window",
    "ota_block_intensity_replay": "ota_block_intensity_replay",
    "ota_motif_block_replay": "ota_motif_block_replay",
    "router_service_block_replay": "router_service_block_replay",
    "router_support_quantile": "router_support_quantile",
    "zigbee_a0_safe_block_replay": "zigbee_a0_safe_block_replay",
    "zigbee_support_dwell": "zigbee_support_dwell",
}


def _norm_generator_name(name: str) -> str:
    return GENERATOR_ALIASES.get(str(name), str(name))


def _candidate_generator_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name = base
    if name.startswith("candidate_"):
        name = name[len("candidate_"):]
    if name.endswith("_VAL.parquet"):
        name = name[:-len("_VAL.parquet")]
    elif name.endswith("_TEST.parquet"):
        name = name[:-len("_TEST.parquet")]
    elif name.endswith(".parquet"):
        name = name[:-len(".parquet")]
    return _norm_generator_name(name)


def _candidate_uid(generator: str, candidate_id: str) -> str:
    raw = f"{str(generator)}::{str(candidate_id)}"
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", raw)


def _canonical_path(path: str) -> str:
    if path is None:
        return ""
    try:
        return os.path.realpath(os.path.abspath(str(path)))
    except Exception:
        return str(path)


def _candidate_preference_score(meta: dict) -> tuple:
    source = str(meta.get("source", ""))
    has_registry_meta = int(isinstance(meta.get("registry_meta", None), dict))
    has_test_path = int(bool(meta.get("test_path", None)))
    source_rank = 3 if source == "candidate_registry" else 1
    meta_len = len(meta.keys())
    return (source_rank, has_registry_meta, has_test_path, meta_len)


def _normalize_candidate_registry(path: str) -> dict:
    if not os.path.exists(path):
        return {
            "version": "cell10_8_synthesized_registry_from_candidate_files",
            "candidates": {},
        }

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "candidates" in raw and isinstance(raw["candidates"], dict):
            return raw
        if "generators" in raw and isinstance(raw["generators"], dict):
            return {
                "version": "cell10_8_normalized_from_generators_dict",
                "original_format": "dict_generators",
                "candidates": raw["generators"],
            }
        if all(isinstance(v, dict) for v in raw.values()):
            return {
                "version": "cell10_8_normalized_from_flat_dict",
                "original_format": "flat_dict",
                "candidates": raw,
            }
        return raw

    if isinstance(raw, list):
        converted = {}
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                log(
                    f"[Cell10.8][WARN] Ignoring non-dict candidate registry item "
                    f"at index={i}: type={type(item).__name__}"
                )
                continue

            gen = (
                item.get("generator")
                or item.get("name")
                or item.get("candidate_generator")
                or item.get("model")
                or item.get("candidate_id")
                or f"candidate_{i}"
            )
            gen_norm = _norm_generator_name(str(gen))
            cid = str(item.get("candidate_id", item.get("id", gen_norm)))
            uid = _candidate_uid(gen_norm, cid)

            converted[uid] = dict(item)
            converted[uid]["generator"] = gen_norm
            converted[uid]["candidate_id"] = cid

        log(
            "[Cell10.8][FIX] Converted list-format candidate registry to dict-format registry | "
            f"items={len(raw)} | candidates={sorted(set(v['generator'] for v in converted.values()))}"
        )

        return {
            "version": "cell10_8_converted_list_registry",
            "original_format": "list",
            "candidates": converted,
        }

    raise RuntimeError(
        "[Cell10.8] cell10_portfolio_candidate_registry.json must be dict or list. "
        f"Got {type(raw).__name__}."
    )


candidate_registry = _normalize_candidate_registry(candidate_registry_path)


def _discover_candidate_files() -> dict:
    """Discover candidate artifacts for the selector.

    Canonical policy: load explicit registry candidates only.
    Brute-force scanning of portfolio_candidates/ is disabled by default because
    it can resurrect stale candidate files from older notebook revisions. A debug
    file scan can be enabled only with CFG['cell10_8_allow_unregistered_file_scan']=True.
    """
    out = {}
    discovery_rows = []

    allow_unregistered_file_scan = bool(CFG.get("cell10_8_allow_unregistered_file_scan", False))
    CFG["cell10_8_allow_unregistered_file_scan"] = allow_unregistered_file_scan

    # Explicit current-run registry candidates are the canonical source of truth.
    reg_candidates = candidate_registry.get("candidates", candidate_registry.get("generators", {}))
    if isinstance(reg_candidates, dict):
        for uid0, meta in reg_candidates.items():
            if not isinstance(meta, dict):
                discovery_rows.append({
                    "candidate_uid": str(uid0),
                    "source": "candidate_registry",
                    "decision": "drop_non_dict_meta",
                    "val_path": "",
                    "test_path": "",
                    "reason": f"non_dict_meta_{type(meta).__name__}",
                })
                continue

            gen = _norm_generator_name(
                meta.get("generator")
                or meta.get("name")
                or meta.get("candidate_generator")
                or uid0
            )
            cid = str(meta.get("candidate_id", meta.get("id", uid0)))
            uid = _candidate_uid(gen, cid)

            val_path = meta.get("val_path", meta.get("VAL_path", None))
            test_path = meta.get("test_path", meta.get("TEST_path", None))
            requires_downstream = bool(meta.get("requires_downstream_test_materialization", False))

            val_valid = bool(val_path and os.path.exists(str(val_path)))
            test_valid = bool(test_path and os.path.exists(str(test_path)))

            if not val_valid:
                discovery_rows.append({
                    "candidate_uid": uid,
                    "generator": str(gen),
                    "candidate_id": cid,
                    "source": "candidate_registry",
                    "decision": "drop_missing_or_invalid_VAL_path",
                    "val_path": str(val_path or ""),
                    "test_path": str(test_path or ""),
                    "reason": "VAL path missing or does not exist",
                })
                continue

            # In the canonical no-TEST-rescue pipeline, candidates should already
            # have a TEST artifact unless they are explicitly marked as requiring
            # downstream materialization. The latter is allowed to be audited, but
            # should normally not be selected later.
            if (not test_valid) and (not requires_downstream):
                discovery_rows.append({
                    "candidate_uid": uid,
                    "generator": str(gen),
                    "candidate_id": cid,
                    "source": "candidate_registry",
                    "decision": "drop_missing_valid_TEST_path",
                    "val_path": str(val_path or ""),
                    "test_path": str(test_path or ""),
                    "reason": "TEST path missing and no downstream materialization flag",
                })
                continue

            out[uid] = {
                "candidate_uid": uid,
                "generator": gen,
                "candidate_id": cid,
                "val_path": str(val_path),
                "test_path": str(test_path) if test_valid else None,
                "source": "candidate_registry",
                "registry_meta": meta,
                "requires_downstream_test_materialization": bool(requires_downstream),
            }
            discovery_rows.append({
                "candidate_uid": uid,
                "generator": str(gen),
                "candidate_id": cid,
                "source": "candidate_registry",
                "decision": "keep_registry_candidate",
                "val_path": str(val_path or ""),
                "test_path": str(test_path or ""),
                "reason": "valid registry candidate",
            })

    # Optional debug fallback only. Do not enable for canonical publication runs.
    if allow_unregistered_file_scan:
        for fn in sorted(os.listdir(CANDDIR)):
            if not (fn.startswith("candidate_") and fn.endswith("_VAL.parquet")):
                continue

            val_path = os.path.join(CANDDIR, fn)
            canonical_val = _canonical_path(val_path)
            if any(_canonical_path(m.get("val_path")) == canonical_val for m in out.values()):
                continue

            gen = _candidate_generator_from_filename(val_path)
            cid = fn.replace("_VAL.parquet", "")
            uid = _candidate_uid(gen, cid)

            test_fn = fn.replace("_VAL.parquet", "_TEST.parquet")
            test_path = os.path.join(CANDDIR, test_fn)
            test_valid = os.path.exists(test_path)

            if not test_valid:
                discovery_rows.append({
                    "candidate_uid": uid,
                    "generator": str(gen),
                    "candidate_id": cid,
                    "source": "candidate_file_scan_debug",
                    "decision": "drop_unregistered_missing_TEST_path",
                    "val_path": str(val_path),
                    "test_path": str(test_path),
                    "reason": "debug scan candidate missing TEST path",
                })
                continue

            out[uid] = {
                "candidate_uid": uid,
                "generator": gen,
                "candidate_id": cid,
                "val_path": val_path,
                "test_path": test_path,
                "source": "candidate_file_scan_debug",
            }
            discovery_rows.append({
                "candidate_uid": uid,
                "generator": str(gen),
                "candidate_id": cid,
                "source": "candidate_file_scan_debug",
                "decision": "keep_debug_file_scan_candidate",
                "val_path": str(val_path),
                "test_path": str(test_path),
                "reason": "debug file-scan enabled",
            })

    discovery_audit_path = os.path.join(REPDIR, "cell10_candidate_discovery_audit_v3_6.csv")
    pd.DataFrame(discovery_rows).to_csv(discovery_audit_path, index=False)
    globals()["CELL10_8_CANDIDATE_DISCOVERY_AUDIT_PATH"] = discovery_audit_path

    log(
        "[Cell10.8][DISCOVERY] Candidate discovery complete | "
        f"registry_items={len(reg_candidates) if isinstance(reg_candidates, dict) else 0} | "
        f"loaded={len(out)} | "
        f"unregistered_file_scan={allow_unregistered_file_scan} | "
        f"audit={discovery_audit_path}"
    )

    return out

def _deduplicate_candidate_files(candidate_files_raw: dict) -> tuple:
    best_by_physical_key = {}
    duplicate_rows = []

    for uid, meta in sorted(candidate_files_raw.items()):
        if not isinstance(meta, dict):
            duplicate_rows.append({
                "candidate_uid": str(uid),
                "generator": None,
                "val_path": None,
                "decision": "drop_non_dict_meta",
                "kept_uid": None,
            })
            continue

        gen = _norm_generator_name(meta.get("generator", uid))
        val_path = meta.get("val_path", None)

        if not val_path:
            duplicate_rows.append({
                "candidate_uid": str(uid),
                "generator": str(gen),
                "val_path": None,
                "decision": "drop_missing_val_path",
                "kept_uid": None,
            })
            continue

        canonical_val_path = _canonical_path(str(val_path))
        # Deduplicate by physical VAL parquet path, not by generator label.
        # File-scan aliases can infer a different generator name from the filename
        # than the explicit registry record; the physical artifact is still identical.
        physical_key = canonical_val_path

        meta2 = dict(meta)
        meta2["candidate_uid"] = str(uid)
        meta2["generator"] = str(gen)
        meta2["canonical_val_path"] = canonical_val_path
        meta2["physical_dedup_key"] = str(physical_key)

        if physical_key not in best_by_physical_key:
            best_by_physical_key[physical_key] = meta2
            continue

        current = best_by_physical_key[physical_key]
        if _candidate_preference_score(meta2) > _candidate_preference_score(current):
            duplicate_rows.append({
                "candidate_uid": current.get("candidate_uid", ""),
                "generator": str(gen),
                "val_path": str(val_path),
                "decision": "drop_duplicate_replaced_by_better_metadata",
                "kept_uid": str(uid),
            })
            best_by_physical_key[physical_key] = meta2
        else:
            duplicate_rows.append({
                "candidate_uid": str(uid),
                "generator": str(gen),
                "val_path": str(val_path),
                "decision": "drop_duplicate_same_physical_val_candidate",
                "kept_uid": current.get("candidate_uid", ""),
            })

    deduped = {}
    for physical_key, meta in best_by_physical_key.items():
        uid = str(meta.get("candidate_uid", ""))
        if not uid:
            gen = str(meta.get("generator", physical_key[0]))
            cid = str(meta.get("candidate_id", os.path.basename(str(meta.get("val_path", "candidate")))))
            uid = _candidate_uid(gen, cid)
            meta["candidate_uid"] = uid

        if uid in deduped:
            gen = str(meta.get("generator", physical_key[0]))
            cid = str(meta.get("candidate_id", uid))
            suffix = hashlib.sha256(str(physical_key).encode("utf-8")).hexdigest()[:8]
            uid = _candidate_uid(gen, f"{cid}_{suffix}")
            meta["candidate_uid"] = uid

        deduped[uid] = meta

    return deduped, duplicate_rows


candidate_files_raw = _discover_candidate_files()
candidate_files, candidate_duplicate_rows = _deduplicate_candidate_files(candidate_files_raw)

candidate_duplicate_audit_path = os.path.join(REPDIR, "cell10_candidate_duplicate_audit.csv")
pd.DataFrame(candidate_duplicate_rows).to_csv(candidate_duplicate_audit_path, index=False)

if candidate_duplicate_rows:
    log(
        "[Cell10.8][DEDUP] Removed duplicate physical candidate aliases | "
        f"raw={len(candidate_files_raw)} | deduped={len(candidate_files)} | "
        f"duplicates={len(candidate_duplicate_rows)} | audit={candidate_duplicate_audit_path}"
    )
else:
    log(
        "[Cell10.8][DEDUP] No duplicate physical candidate aliases found | "
        f"candidates={len(candidate_files)} | audit={candidate_duplicate_audit_path}"
    )


# ----------------------------------------------------------
# 4) Hardcoded portfolio eligibility + Markov rescue
# ----------------------------------------------------------
GENERATOR_ROLES = {
    "A1_temporal_block_bootstrap": "default_backbone_for_all_protocol_columns",
    "ddpm": "dense_semidense_conditional_refinement",
    "ctgan_tvae": "mixed_tabular_marginal_correlation_candidate",
    "timegan_sequence_gan": "temporal_burst_autocorrelation_candidate",
    "gaussian_copula": "continuous_correlation_preservation_candidate",
    "markov_semimarkov": "rare_discrete_event_dwell_transition_candidate",
    "negative_binomial_poisson_gamma": "overdispersed_count_burst_candidate",
    "ota_burst_runlength_active_window": "ota_temporal_runlength_autocorrelation_candidate",
    "ota_block_intensity_replay": "ota_temporal_autocorrelation_block_replay_candidate",
    "ota_motif_block_replay": "ota_temporal_motif_block_replay_candidate",
    "router_service_block_replay": "router_service_temporal_block_replay_candidate",
    "router_support_quantile": "router_marginal_support_quantile_candidate",
    "zigbee_a0_safe_block_replay": "zigbee_a0_safe_temporal_block_replay_candidate",
    "zigbee_support_dwell": "zigbee_presence_support_dwell_candidate",
}

KNOWN_GENERATORS = [
    "A1_temporal_block_bootstrap",
    "ddpm",
    "ctgan_tvae",
    "timegan_sequence_gan",
    "gaussian_copula",
    "markov_semimarkov",
    "negative_binomial_poisson_gamma",
]


def _train_support_profile(col: str) -> dict:
    col = str(col)
    if col in _TRAIN_SUPPORT_PROFILE_CACHE:
        return dict(_TRAIN_SUPPORT_PROFILE_CACHE[col])

    tier = _tier_of(col)
    vals = _train_values_cached(col)

    if vals.size == 0:
        out = {
            "tier": tier,
            "train_support_n": 0,
            "train_nonzero_rate": np.nan,
            "train_unique_n": 0,
            "train_unique_frac": np.nan,
            "low_cardinality": False,
            "sparse_like": False,
            "eligible_sparse_discrete_name": bool(SPARSE_NAME_RX.search(col)),
            "unique_like": bool(_is_unique_like(col)),
            "dense_aggregate_like": bool(_is_dense_aggregate_like(col)),
        }
        _TRAIN_SUPPORT_PROFILE_CACHE[col] = out
        return dict(out)

    vals_bounded = _bounded_finite_1d(vals)
    rounded = vals_bounded.copy()
    if _is_countlike(col):
        rounded = np.rint(rounded)
    else:
        rounded = np.round(rounded.astype(np.float64), 6)

    unique_n = int(np.unique(rounded).size)
    unique_frac = float(unique_n / max(1, vals_bounded.size))
    nz = _nonzero_rate(vals_bounded)

    low_card = bool(
        unique_n <= MARKOV_RESCUE_MAX_UNIQUE
        or unique_frac <= MARKOV_RESCUE_LOW_CARDINALITY_FRAC
    )
    sparse_like = bool(np.isfinite(nz) and nz <= MARKOV_RESCUE_SPARSE_NZ_RATE_MAX)

    out = {
        "tier": tier,
        "train_support_n": int(vals.size),
        "train_nonzero_rate": float(nz) if np.isfinite(nz) else np.nan,
        "train_unique_n": int(unique_n),
        "train_unique_frac": float(unique_frac),
        "low_cardinality": bool(low_card),
        "sparse_like": bool(sparse_like),
        "eligible_sparse_discrete_name": bool(SPARSE_NAME_RX.search(col)),
        "unique_like": bool(_is_unique_like(col)),
        "dense_aggregate_like": bool(_is_dense_aggregate_like(col)),
    }
    _TRAIN_SUPPORT_PROFILE_CACHE[col] = out
    return dict(out)


def _markov_rescue_eligible(col: str, profile: dict) -> tuple:
    col = str(col)

    if not ENABLE_MARKOV_RESCUE:
        return False, "markov_rescue_disabled"
    if profile["train_support_n"] < MARKOV_RESCUE_MIN_TRAIN_N:
        return False, "too_few_train_values"
    if _is_unique_like(col):
        return False, "unique_like_not_markov_rescue"
    if _is_dense_aggregate_like(col):
        return False, "dense_aggregate_not_markov_rescue"

    name_ok = bool(profile["eligible_sparse_discrete_name"])
    low_card = bool(profile["low_cardinality"])
    sparse_like = bool(profile["sparse_like"])

    if name_ok and low_card and sparse_like:
        return True, "eligible_sparse_low_cardinality_event_state"

    return False, "not_sparse_low_cardinality_event_state"


def _runs_from_sequence(x: np.ndarray) -> list:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []

    if np.all(np.abs(arr - np.rint(arr)) <= 1e-6):
        arr = np.rint(arr).astype(np.float64)

    runs = []
    s = 0
    cur = arr[0]
    for i in range(1, arr.size + 1):
        if i == arr.size or arr[i] != cur:
            runs.append((float(cur), int(i - s)))
            if i < arr.size:
                s = i
                cur = arr[i]
    return runs


def _sample_markov_runs(train_vals: np.ndarray, n_active: int, rng: np.random.Generator) -> np.ndarray:
    n_active = int(n_active)
    if n_active <= 0:
        return np.zeros(0, dtype=np.float32)

    train_vals = _finite_1d(train_vals)
    if train_vals.size == 0:
        return np.zeros(n_active, dtype=np.float32)

    runs = _runs_from_sequence(train_vals)
    if not runs:
        idx = rng.integers(0, train_vals.size, size=n_active)
        return train_vals[idx].astype(np.float32, copy=False)

    values = np.asarray([r[0] for r in runs], dtype=np.float64)
    lengths = np.asarray([max(1, int(r[1])) for r in runs], dtype=np.int64)

    unique_vals, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    probs = counts.astype(np.float64)
    probs = probs / max(float(probs.sum()), 1.0)

    length_pools = {}
    for state_idx in range(len(unique_vals)):
        pool = lengths[inverse == state_idx]
        if pool.size == 0:
            pool = lengths
        length_pools[int(state_idx)] = pool.astype(np.int64, copy=False)

    transition_pools = defaultdict(list)
    for i in range(len(inverse) - 1):
        transition_pools[int(inverse[i])].append(int(inverse[i + 1]))

    transition_pools = {int(k): np.asarray(v, dtype=np.int64) for k, v in transition_pools.items()}

    cur_state = int(rng.choice(np.arange(len(unique_vals)), p=probs))
    chunks = []
    produced = 0
    max_runs = int(n_active + 10_000)
    run_iter = 0

    while produced < n_active:
        run_iter += 1
        if run_iter > max_runs:
            remaining = n_active - produced
            iid_states = rng.choice(np.arange(len(unique_vals)), size=remaining, p=probs)
            chunks.append(unique_vals[iid_states].astype(np.float32, copy=False))
            produced = n_active
            break

        lp = length_pools.get(cur_state, lengths)
        L = int(lp[int(rng.integers(0, len(lp)))])
        L = max(1, L)

        take = min(L, n_active - produced)
        chunks.append(np.full(take, unique_vals[cur_state], dtype=np.float32))
        produced += take

        nxt_pool = transition_pools.get(cur_state, None)
        if nxt_pool is not None and len(nxt_pool) > 0:
            cur_state = int(nxt_pool[int(rng.integers(0, len(nxt_pool)))])
        else:
            cur_state = int(rng.choice(np.arange(len(unique_vals)), p=probs))

    out = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if out.shape[0] != n_active:
        out = out[:n_active]
    return out


def _materialize_markov_rescue_val_candidates(eligibility_rows: list) -> dict:
    raise RuntimeError(
        "[Cell10.8] In-selector Markov rescue is disabled in the canonical pure selector. "
        "Move any rescue generator to a separate TRAIN-fitted candidate cell before Cell 10.8."
    )

    eligible_cols = [
        r["col"] for r in eligibility_rows
        if r["generator"] == "markov_semimarkov" and bool(r["eligible"])
    ]

    if not eligible_cols:
        return {}

    log(
        "[Cell10.8][MARKOV-RESCUE] Materializing TRAIN-only VAL rescue candidate under A1_VAL masks | "
        f"cols={len(eligible_cols)} | cols_preview={eligible_cols[:10]}"
    )

    val_frame = {}
    generated_cols = []

    for i, col in enumerate(eligible_cols, start=1):
        if i == 1 or i % 5 == 0 or i == len(eligible_cols):
            log(f"[Cell10.8][MARKOV-RESCUE] progress {i}/{len(eligible_cols)} | col={col}")

        tr_vals = _train_values_cached(col)
        if tr_vals.size < MARKOV_RESCUE_MIN_TRAIN_N:
            continue

        a1_active = _a1_val_active_mask(col)
        va_arr = np.full(N_VAL, np.nan, dtype=np.float32)

        va_sample = _sample_markov_runs(tr_vals, int(a1_active.sum()), rng)
        va_arr[a1_active] = _postprocess_col(col, va_sample)

        val_frame[col] = _postprocess_col(col, va_arr)
        generated_cols.append(col)

    if not generated_cols:
        return {}

    p_val = os.path.join(CANDDIR, "candidate_markov_semimarkov_rescue_VAL.parquet")
    pd.DataFrame(val_frame).to_parquet(p_val, index=False)

    log(
        "[Cell10.8][MARKOV-RESCUE] Saved VAL-only rescue candidate | "
        f"cols={len(generated_cols)} | val={p_val}"
    )

    gen = "markov_semimarkov"
    cid = "markov_semimarkov_rescue_train_only_v3_4_val_only_a1mask"
    uid = _candidate_uid(gen, cid)

    return {
        uid: {
            "candidate_uid": uid,
            "generator": gen,
            "candidate_id": cid,
            "val_path": p_val,
            "test_path": None,
            "source": "cell10_8_train_only_val_rescue",
            "cols": generated_cols,
            "requires_downstream_test_materialization": True,
            "mask_source": "A1_VAL_baseline_finite_mask",
        }
    }


# ----------------------------------------------------------
# 4.1) Hardcoded portfolio eligibility rows
# ----------------------------------------------------------
eligibility_rows = []

for col in PROTO_COLS:
    profile = _train_support_profile(col)
    tier = profile["tier"]

    for gen in KNOWN_GENERATORS:
        eligible = False
        reason = "not_evaluated"

        if gen == "A1_temporal_block_bootstrap":
            eligible = True
            reason = "always_included_baseline"

        elif gen == "markov_semimarkov":
            eligible, reason = _markov_rescue_eligible(col, profile)

        elif gen == "negative_binomial_poisson_gamma":
            vals = _train_values_cached(col)
            vals_bounded = _bounded_finite_1d(vals)
            mu = _safe_mean(vals_bounded)
            var = float(np.var(vals_bounded)) if vals_bounded.size > 1 else np.nan
            vtm = float(var / max(mu, 1e-9)) if np.isfinite(mu) and np.isfinite(var) and mu > 0 else np.nan
            eligible = bool(_is_countlike(col) and np.isfinite(vtm) and vtm > 1.25)
            reason = "overdispersed_count" if eligible else "not_overdispersed_count"

        elif gen == "gaussian_copula":
            col_l = str(col).lower()
            copula_forbidden_discrete_name = bool(
                col_l.endswith("_present")
                or col_l.endswith("_obs_present")
                or "__data" in col_l
                or "__cmd" in col_l
                or "__ack" in col_l
                or "__channel" in col_l
                or "beacon" in col_l
                or "assoc" in col_l
                or "auth" in col_l
                or "deauth" in col_l
                or "disassoc" in col_l
                or "probe_req" in col_l
                or "probe_resp" in col_l
                or "action" in col_l
                or "zcl" in col_l
                or "aps" in col_l
                or "nwk_present" in col_l
                or "app_obs_present" in col_l
            )

            vals = _train_values_cached(col)
            vals_bounded = _bounded_finite_1d(vals)
            unique_n = int(np.unique(np.round(vals_bounded, 6)).size) if vals_bounded.size else 0
            unique_frac = float(unique_n / max(1, vals_bounded.size)) if vals_bounded.size else 0.0

            copula_dense_enough = bool(
                profile["train_support_n"] >= MARKOV_RESCUE_MIN_TRAIN_N
                and not profile["sparse_like"]
                and not profile["unique_like"]
                and unique_n >= int(CFG.get("cell10_copula_min_unique", 20))
                and unique_frac > float(CFG.get("cell10_copula_min_unique_frac", 0.001))
            )

            eligible = bool(copula_dense_enough and not copula_forbidden_discrete_name)

            if eligible:
                reason = "dense_continuous_or_dense_count_correlation_candidate"
            elif copula_forbidden_discrete_name:
                reason = "discrete_presence_or_protocol_state_not_gaussian_copula"
            else:
                reason = "sparse_unique_low_cardinality_or_too_few_train_values"

        elif gen == "ddpm":
            ddpm_cols = set(map(str, globals().get("ddpm_cols_use", [])))
            eligible = bool(col in ddpm_cols)
            reason = "in_ddpm_value_contract" if eligible else "not_in_ddpm_value_contract"

        elif gen == "ctgan_tvae":
            eligible = bool(CFG.get("cell10_enable_ctgan_tvae", False))
            reason = "enabled_by_cfg" if eligible else "disabled_by_cfg"

        elif gen == "timegan_sequence_gan":
            eligible = bool(CFG.get("cell10_enable_timegan", False))
            reason = "enabled_by_cfg" if eligible else "disabled_by_cfg"

        eligibility_rows.append({
            "col": col,
            "tier": tier,
            "generator": gen,
            "candidate_id": "",
            "candidate_uid": "",
            "eligible": bool(eligible),
            "ineligible_reason": "" if eligible else reason,
            "eligibility_reason": reason if eligible else "",
            "train_support_n": int(profile["train_support_n"]),
            "train_nonzero_rate": profile["train_nonzero_rate"],
            "train_unique_n": int(profile["train_unique_n"]),
            "train_unique_frac": profile["train_unique_frac"],
            "low_cardinality": bool(profile["low_cardinality"]),
            "sparse_like": bool(profile["sparse_like"]),
            "unique_like": bool(profile["unique_like"]),
            "dense_aggregate_like": bool(profile["dense_aggregate_like"]),
            "family_role": GENERATOR_ROLES.get(gen, "unknown"),
            "source": "hardcoded_generator_policy",
            "val_path": "",
            "test_path": "",
            "test_path_valid": "",
        })

rescue_candidates = {}
rescue_duplicate_rows = []
rescue_duplicate_audit_path = os.path.join(REPDIR, "cell10_candidate_duplicate_audit_after_rescue.csv")
pd.DataFrame(rescue_duplicate_rows).to_csv(rescue_duplicate_audit_path, index=False)

log(
    "[Cell10.8] In-selector Markov rescue disabled. "
    "Cell 10.8 remains a pure VAL-only selector over pre-registered candidate artifacts."
)


# ----------------------------------------------------------
# 5) Candidate VAL frame loading
# ----------------------------------------------------------
def _load_candidate_frames(candidate_files: dict) -> dict:
    loaded = {}

    for uid, meta in sorted(candidate_files.items()):
        gen = _norm_generator_name(meta.get("generator", uid))
        cid = str(meta.get("candidate_id", uid))
        val_path = meta.get("val_path")
        test_path = meta.get("test_path")

        if not val_path or not os.path.exists(str(val_path)):
            log(f"[Cell10.8][CANDIDATE-SKIP] {uid}: missing VAL path={val_path}")
            continue

        val_path = str(val_path)
        test_path = str(test_path) if test_path else None

        log(f"[Cell10.8][LOAD] {uid}: reading VAL candidate metadata | path={val_path}")

        val_rows_meta = _parquet_num_rows(val_path)
        if val_rows_meta is not None and int(val_rows_meta) != N_VAL:
            log(
                f"[Cell10.8][WARN] Skipping {uid}: VAL metadata row mismatch "
                f"{val_rows_meta} vs {N_VAL}"
            )
            continue

        val_cols_meta = _parquet_columns(val_path)
        read_cols = None

        if val_cols_meta is not None:
            val_cols_set = set(map(str, val_cols_meta))
            read_cols = [c for c in PROTO_COLS if c in val_cols_set]
            if not read_cols:
                log(f"[Cell10.8][WARN] Skipping {uid}: no protocol columns in VAL candidate.")
                continue

        try:
            if read_cols is not None:
                val_df = pd.read_parquet(val_path, columns=read_cols)
            else:
                val_df = pd.read_parquet(val_path)
        except Exception as e:
            log(f"[Cell10.8][WARN] Could not read VAL candidate for {uid}: {type(e).__name__}: {e}")
            continue

        if len(val_df) != N_VAL:
            log(f"[Cell10.8][WARN] Skipping {uid}: VAL row mismatch {len(val_df)} vs {N_VAL}")
            continue

        cols = [c for c in PROTO_COLS if c in val_df.columns]
        if not cols:
            log(f"[Cell10.8][WARN] Skipping {uid}: no usable protocol columns after VAL load.")
            continue

        test_path_valid = False
        test_metadata_rows = None

        if test_path and os.path.exists(test_path):
            test_metadata_rows = _parquet_num_rows(test_path)
            if test_metadata_rows is None:
                test_path_valid = True
                log(
                    f"[Cell10.8][TEST-PATH] {uid}: TEST path exists; row-count metadata unavailable. "
                    "Not reading TEST in selector."
                )
            elif int(test_metadata_rows) == N_TEST:
                test_path_valid = True
            else:
                log(
                    f"[Cell10.8][WARN] {uid}: TEST metadata row mismatch "
                    f"{test_metadata_rows} vs {N_TEST}; path will not be treated as valid."
                )

        requires_downstream = bool(meta.get("requires_downstream_test_materialization", False))

        loaded[uid] = {
            **meta,
            "candidate_uid": uid,
            "generator": gen,
            "candidate_id": cid,
            "val_path": val_path,
            "test_path": test_path if test_path_valid else None,
            "test_path_exists": bool(test_path and os.path.exists(test_path)),
            "test_path_metadata_valid": bool(test_path_valid),
            "test_metadata_rows": int(test_metadata_rows) if test_metadata_rows is not None else None,
            "requires_downstream_test_materialization": bool(requires_downstream),
            "val_df": val_df,
            "cols": cols,
            "val_rows": int(len(val_df)),
            "val_cols_n": int(len(cols)),
        }

        log(
            f"[Cell10.8][LOAD] {uid}: loaded VAL candidate | "
            f"generator={gen} | rows={len(val_df):,} | cols={len(cols)} | "
            f"test_path_valid={test_path_valid} | requires_downstream={requires_downstream}"
        )

    return loaded


loaded_candidates = _load_candidate_frames(candidate_files)
log("[Cell10.8] Loaded candidate IDs: " f"{sorted(loaded_candidates.keys())}")


# ----------------------------------------------------------
# 5.1) Registry-defined candidate eligibility injection
# ----------------------------------------------------------
for uid, meta in sorted(loaded_candidates.items()):
    gen = str(meta.get("generator", uid))
    if gen == "A1_temporal_block_bootstrap":
        continue

    cols = [str(c) for c in meta.get("cols", []) if str(c) in PROTO_COLS]
    if not cols:
        continue

    test_valid = bool(meta.get("test_path_metadata_valid", False))
    requires_downstream = bool(meta.get("requires_downstream_test_materialization", False))

    for col in cols:
        profile = _train_support_profile(col)
        eligible = bool(test_valid or requires_downstream)

        if eligible:
            reason = "registry_candidate_column_valid"
            ineligible = ""
        else:
            reason = ""
            ineligible = "registry_candidate_missing_valid_TEST_path"

        eligibility_rows.append({
            "col": col,
            "tier": _tier_of(col),
            "generator": gen,
            "candidate_id": str(meta.get("candidate_id", uid)),
            "candidate_uid": str(uid),
            "eligible": bool(eligible),
            "ineligible_reason": ineligible,
            "eligibility_reason": reason,
            "train_support_n": int(profile["train_support_n"]),
            "train_nonzero_rate": profile["train_nonzero_rate"],
            "train_unique_n": int(profile["train_unique_n"]),
            "train_unique_frac": profile["train_unique_frac"],
            "low_cardinality": bool(profile["low_cardinality"]),
            "sparse_like": bool(profile["sparse_like"]),
            "unique_like": bool(profile["unique_like"]),
            "dense_aggregate_like": bool(profile["dense_aggregate_like"]),
            "family_role": GENERATOR_ROLES.get(gen, "registry_defined_candidate"),
            "source": "candidate_registry_v3_loaded_candidate",
            "val_path": str(meta.get("val_path", "")),
            "test_path": str(meta.get("test_path", "")),
            "test_path_valid": bool(test_valid),
        })

eligibility_audit = pd.DataFrame(eligibility_rows)
candidate_eligibility_audit_path = os.path.join(REPDIR, "cell10_candidate_eligibility_audit.csv")
eligibility_audit.to_csv(candidate_eligibility_audit_path, index=False)

eligibility_lookup_uid = {
    (str(r["col"]), str(r.get("candidate_uid", ""))): bool(r["eligible"])
    for r in eligibility_rows
    if str(r.get("candidate_uid", "")) != ""
}

eligibility_lookup_gen = {
    (str(r["col"]), str(r["generator"])): bool(r["eligible"])
    for r in eligibility_rows
}

log(
    "[Cell10.8] Candidate eligibility audit ready | "
    f"rows={len(eligibility_audit)} | "
    f"loaded_candidates={len(loaded_candidates)} | "
    f"registry_candidate_eligibility_rows={int((eligibility_audit['source'] == 'candidate_registry_v3_loaded_candidate').sum())} | "
    f"rescue_markov_val_cols={len(next(iter(rescue_candidates.values()), {}).get('cols', [])) if rescue_candidates else 0}"
)


# ----------------------------------------------------------
# 6) Build VAL comparison table
# ----------------------------------------------------------
comparison_rows = []
baseline_by_col = {}
a0_control_rows = []

for col_i, col in enumerate(PROTO_COLS, start=1):
    if col_i == 1 or col_i % PROGRESS_EVERY_COLS == 0 or col_i == len(PROTO_COLS):
        log(
            f"[Cell10.8][COMPARE] progress {col_i}/{len(PROTO_COLS)} | "
            f"loaded_candidates={len(loaded_candidates)}"
        )

    tier = _tier_of(col)

    if col not in A0_VAL.columns:
        raise RuntimeError(f"[Cell10.8] A0_VAL missing protocol column: {col}")
    if col not in A1_VAL.columns:
        raise RuntimeError(f"[Cell10.8] A1_VAL missing protocol column: {col}")

    # A0 diagnostic/control: real VAL mask only.
    a0_active = _real_val_active_mask(col) & _a0_val_active_mask(col)
    a0_real_vals = _val_real_values_for_mask(col, a0_active)
    a0_x = pd.to_numeric(A0_VAL.loc[a0_active, col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    a0_x = _postprocess_col(col, a0_x)

    a0_metrics = _metric_row_scored(a0_real_vals, a0_x, col, tier)
    a0_score = float(a0_metrics["combined_score"])

    a0_row = {
        "col": col,
        "tier": tier,
        "variant": "A0",
        "generator": "A0_real_mask_baseline",
        "candidate_uid": "A0_real_mask_baseline::A0_VAL_baseline",
        "candidate_id": "A0_VAL_baseline",
        "is_baseline": False,
        "is_control": True,
        "is_selection_candidate": False,
        "accepted": False,
        "reason": "A0_diagnostic_control_not_selectable",
        "score": a0_score,
        **a0_metrics,
        "val_path": a0_val_path,
        "test_path": a0_test_path if os.path.exists(a0_test_path) else None,
        "candidate_eligible": False,
        "accepted_decision": False,
        "reason_decision": "A0_control_not_used_for_A2_selection",
        "baseline_score": None,
        "baseline_q1_score": None,
        "baseline_q2_score": None,
        "score_gain_frac": None,
        "q1_gain_frac": None,
        "q2_gain_frac": None,
        "ks_gain": None,
        "wasserstein_delta": None,
        "support_jaccard_delta": None,
        "borderline_acceptance": False,
        "near_borderline_acceptance": False,
        "block_stability_passed": None,
        "block_stability_reason": None,
        "eval_mask_policy": "real_VAL_mask_for_A0_control",
        "eval_active_n": int(a0_active.sum()),
        "selected_generator_for_col": False,
        "tier_q1_weight": TIER_WEIGHTS.get(tier, TIER_WEIGHTS["unknown"])["q1"],
        "tier_q2_weight": TIER_WEIGHTS.get(tier, TIER_WEIGHTS["unknown"])["q2"],
    }
    comparison_rows.append(a0_row)
    a0_control_rows.append(a0_row)

    # A1 selection baseline.
    eval_active = _selection_eval_mask(col)
    real_vals = _val_real_values_for_mask(col, eval_active)

    a1_x = pd.to_numeric(A1_VAL.loc[eval_active, col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    a1_x = _postprocess_col(col, a1_x)

    baseline_metrics = _metric_row_scored(real_vals, a1_x, col, tier)
    baseline_score = float(baseline_metrics["combined_score"])
    baseline_q1 = float(baseline_metrics["q1_score"])
    baseline_q2 = float(baseline_metrics["q2_score"])

    baseline_row = {
        "col": col,
        "tier": tier,
        "variant": "A1",
        "generator": "A1_temporal_block_bootstrap",
        "candidate_uid": "A1_temporal_block_bootstrap::A1_VAL_baseline",
        "candidate_id": "A1_VAL_baseline",
        "is_baseline": True,
        "is_control": False,
        "is_selection_candidate": False,
        "accepted": True,
        "reason": "selection_baseline",
        "score": baseline_score,
        **baseline_metrics,
        "val_path": a1_val_path,
        "test_path": a1_test_path if os.path.exists(a1_test_path) else None,
        "candidate_eligible": True,
        "accepted_decision": None,
        "reason_decision": "baseline",
        "baseline_score": None,
        "baseline_q1_score": None,
        "baseline_q2_score": None,
        "score_gain_frac": 0.0,
        "q1_gain_frac": 0.0,
        "q2_gain_frac": 0.0,
        "ks_gain": 0.0,
        "wasserstein_delta": 0.0,
        "support_jaccard_delta": 0.0,
        "borderline_acceptance": False,
        "near_borderline_acceptance": False,
        "block_stability_passed": None,
        "block_stability_reason": None,
        "eval_mask_policy": "real_VAL_mask_intersect_A1_VAL_synthetic_mask",
        "eval_active_n": int(eval_active.sum()),
        "selected_generator_for_col": True,
        "tier_q1_weight": TIER_WEIGHTS.get(tier, TIER_WEIGHTS["unknown"])["q1"],
        "tier_q2_weight": TIER_WEIGHTS.get(tier, TIER_WEIGHTS["unknown"])["q2"],
    }
    comparison_rows.append(baseline_row)

    baseline_by_col[col] = {
        "row": baseline_row,
        "real_vals": real_vals,
        "a1_vals": a1_x,
        "eval_active": eval_active,
    }

    for uid, meta in loaded_candidates.items():
        gen = str(meta["generator"])

        if gen == "A1_temporal_block_bootstrap":
            continue

        val_df = meta["val_df"]
        if col not in val_df.columns:
            continue

        eligible = bool(
            eligibility_lookup_uid.get((col, str(uid)), False)
            or eligibility_lookup_gen.get((col, gen), False)
        )
        if not eligible:
            continue

        cand_x = pd.to_numeric(val_df.loc[eval_active, col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        cand_x = _postprocess_col(col, cand_x)

        cand_metrics = _metric_row_scored(real_vals, cand_x, col, tier)
        cand_score = float(cand_metrics["combined_score"])

        comparison_rows.append({
            "col": col,
            "tier": tier,
            "variant": "A2_candidate",
            "generator": gen,
            "candidate_uid": uid,
            "candidate_id": meta.get("candidate_id", uid),
            "is_baseline": False,
            "is_control": False,
            "is_selection_candidate": True,
            "accepted": False,
            "reason": "candidate",
            "score": cand_score,
            **cand_metrics,
            "val_path": meta.get("val_path"),
            "test_path": meta.get("test_path"),
            "candidate_eligible": True,
            "accepted_decision": None,
            "reason_decision": None,
            "baseline_score": baseline_score,
            "baseline_q1_score": baseline_q1,
            "baseline_q2_score": baseline_q2,
            "score_gain_frac": None,
            "q1_gain_frac": None,
            "q2_gain_frac": None,
            "ks_gain": None,
            "wasserstein_delta": None,
            "support_jaccard_delta": None,
            "borderline_acceptance": None,
            "near_borderline_acceptance": None,
            "block_stability_passed": None,
            "block_stability_reason": None,
            "eval_mask_policy": "real_VAL_mask_intersect_A1_VAL_synthetic_mask",
            "eval_active_n": int(eval_active.sum()),
            "selected_generator_for_col": False,
            "tier_q1_weight": TIER_WEIGHTS.get(tier, TIER_WEIGHTS["unknown"])["q1"],
            "tier_q2_weight": TIER_WEIGHTS.get(tier, TIER_WEIGHTS["unknown"])["q2"],
        })

comparison = pd.DataFrame(comparison_rows)
if comparison.empty:
    raise RuntimeError("[Cell10.8] Comparison table is empty.")

a0_control_audit = pd.DataFrame(a0_control_rows)
a0_control_audit_path = os.path.join(REPDIR, "cell10_a0_a1_val_control_audit.csv")


# ----------------------------------------------------------
# 7) Selection
# ----------------------------------------------------------
selected_by_col = {}
selected_by_col_rich = {}
best_rejected_rows = []
selected_rows_for_report = []

borderline_accepted = 0
borderline_rejected = 0
near_borderline_accepted = 0
near_borderline_rejected = 0
temporal_only_accepted = 0
requires_downstream_cols = []

for i, col in enumerate(PROTO_COLS, start=1):
    if i == 1 or i % PROGRESS_EVERY_COLS == 0 or i == len(PROTO_COLS):
        log(f"[Cell10.8][SELECT] progress {i}/{len(PROTO_COLS)}")

    tier = _tier_of(col)
    base = baseline_by_col[col]["row"]
    real_vals = baseline_by_col[col]["real_vals"]
    a1_vals = baseline_by_col[col]["a1_vals"]

    baseline_score = float(base["score"])
    baseline_q1 = float(base.get("q1_score", np.nan))
    baseline_q2 = float(base.get("q2_score", np.nan))
    baseline_ks = float(base.get("ks", np.nan))
    baseline_w = float(base.get("wasserstein", np.nan))
    baseline_support_j = float(base.get("support_jaccard", np.nan))

    cands = comparison[
        (comparison["col"].astype(str) == str(col))
        & (comparison["variant"].astype(str) == "A2_candidate")
        & (comparison["candidate_eligible"].astype(bool))
    ].copy()

    def _select_a1(reason_text, best=None):
        selected_by_col[col] = {
            "selected_generator": "A1_temporal_block_bootstrap",
            "selected_candidate_id": "A1_VAL_baseline",
            "selected_candidate_uid": "A1_temporal_block_bootstrap::A1_VAL_baseline",
            "reason": reason_text,
            "val_path": a1_val_path,
            "test_path": a1_test_path,
            "requires_downstream_test_materialization": False,
            "score": baseline_score,
            "q1_score": baseline_q1,
            "q2_score": baseline_q2,
            "selection_baseline": "A1_VAL_baseline",
        }

        selected_by_col_rich[col] = dict(selected_by_col[col])
        selected_by_col_rich[col].update({
            "col": col,
            "tier": tier,
            "baseline_score": baseline_score,
            "baseline_q1_score": baseline_q1,
            "baseline_q2_score": baseline_q2,
            "selected_score": baseline_score,
            "selected_q1_score": baseline_q1,
            "selected_q2_score": baseline_q2,
            "score_gain_frac": 0.0,
            "q1_gain_frac": 0.0,
            "q2_gain_frac": 0.0,
            "ks_gain": 0.0,
            "wasserstein_delta": 0.0,
            "support_jaccard_delta": 0.0,
        })

        if best is not None:
            selected_by_col_rich[col].update({
                "best_rejected_generator": str(best.get("generator")),
                "best_rejected_candidate_uid": str(best.get("candidate_uid")),
                "best_rejected_candidate_id": str(best.get("candidate_id")),
                "best_rejected_score": float(best.get("score", np.nan)),
                "best_rejected_q1_score": float(best.get("q1_score", np.nan)),
                "best_rejected_q2_score": float(best.get("q2_score", np.nan)),
                "best_rejected_score_gain_frac": float(best.get("_score_gain_frac", np.nan)),
                "best_rejected_q1_gain_frac": float(best.get("_q1_gain_frac", np.nan)),
                "best_rejected_q2_gain_frac": float(best.get("_q2_gain_frac", np.nan)),
                "best_rejected_ks_gain": float(best.get("_ks_gain", np.nan)),
                "best_rejected_wasserstein_delta": float(best.get("_wasserstein_delta", np.nan)),
                "best_rejected_support_jaccard_delta": float(best.get("_support_jaccard_delta", np.nan)),
                "best_rejected_reason": str(best.get("_reject_reason", "")),
            })

        selected = dict(base)
        selected["accepted_decision"] = True
        selected["reason_decision"] = reason_text
        selected["selected_generator_for_col"] = True
        selected_rows_for_report.append(selected)

    if len(cands) == 0:
        _select_a1("no_optional_candidate_available")
        continue

    cands["score"] = pd.to_numeric(cands["score"], errors="coerce")
    cands = cands.replace([np.inf, -np.inf], np.nan)
    cands_valid = cands[cands["score"].notna()].copy()

    if len(cands_valid) == 0:
        _select_a1("no_finite_optional_candidate_score")
        continue

    # Compute gain fields for all candidates.
    def _gain_frac(base_v, cand_v, lower_is_better=True):
        if not np.isfinite(base_v) or not np.isfinite(cand_v):
            return -np.inf
        if lower_is_better:
            return float((base_v - cand_v) / max(abs(base_v), 1e-12))
        return float((cand_v - base_v) / max(abs(base_v), 1e-12))

    cands_valid["_score_gain_frac"] = cands_valid["score"].map(
        lambda x: _gain_frac(baseline_score, float(x), lower_is_better=True)
    )
    cands_valid["_q1_gain_frac"] = cands_valid["q1_score"].map(
        lambda x: _gain_frac(baseline_q1, float(x), lower_is_better=True)
    )
    cands_valid["_q2_gain_frac"] = cands_valid["q2_score"].map(
        lambda x: _gain_frac(baseline_q2, float(x), lower_is_better=True)
    )

    cands_valid["_ks_gain"] = cands_valid["ks"].map(
        lambda x: float(baseline_ks - float(x)) if np.isfinite(baseline_ks) and np.isfinite(float(x)) else -np.inf
    )
    cands_valid["_wasserstein_delta"] = cands_valid["wasserstein"].map(
        lambda x: float(float(x) - baseline_w) if np.isfinite(baseline_w) and np.isfinite(float(x)) else np.inf
    )
    cands_valid["_support_jaccard_delta"] = cands_valid["support_jaccard"].map(
        lambda x: float(float(x) - baseline_support_j) if np.isfinite(baseline_support_j) and np.isfinite(float(x)) else 0.0
    )

    # Best by combined score.
    cands_valid = cands_valid.sort_values(["score", "generator", "candidate_uid"], ascending=[True, True, True])
    best = cands_valid.iloc[0].to_dict()

    candidate_score = float(best["score"])
    candidate_q1 = float(best.get("q1_score", np.nan))
    candidate_q2 = float(best.get("q2_score", np.nan))
    candidate_ks = float(best.get("ks", np.nan))
    candidate_w = float(best.get("wasserstein", np.nan))
    candidate_support_j = float(best.get("support_jaccard", np.nan))

    score_gain_frac = float(best["_score_gain_frac"])
    q1_gain_frac = float(best["_q1_gain_frac"])
    q2_gain_frac = float(best["_q2_gain_frac"])
    ks_gain = float(best["_ks_gain"])
    wasserstein_delta = float(best["_wasserstein_delta"])
    support_jaccard_delta = float(best["_support_jaccard_delta"])

    # Gates.
    score_ok = bool(score_gain_frac >= MIN_SCORE_GAIN_FRAC)
    ks_ok = bool(ks_gain >= MIN_KS_GAIN)
    q2_ok = bool(q2_gain_frac >= MIN_Q2_GAIN_FRAC)

    q1_not_bad = bool(
        np.isfinite(q1_gain_frac)
        and q1_gain_frac >= -MAX_Q1_REGRESS_FRAC
    )
    support_not_bad = bool(
        not np.isfinite(support_jaccard_delta)
        or support_jaccard_delta >= -MAX_SUPPORT_JACCARD_REGRESS
    )

    wasser_ok = bool(
        candidate_w <= baseline_w * (1.0 + MAX_WASSERSTEIN_REGRESS_FRAC)
    ) if np.isfinite(candidate_w) and np.isfinite(baseline_w) else False

    # Standard acceptance:
    # - combined score improves enough
    # - Q1 is not materially worse
    # - support is not materially worse
    # - either KS improves OR Q2 improves
    accepted_standard = bool(
        score_ok
        and q1_not_bad
        and support_not_bad
        and wasser_ok
        and (ks_ok or q2_ok)
    )

    # Temporal-only acceptance:
    # Used for temporal-heavy tiers such as OTA, where Q2 can improve strongly
    # while KS is nearly flat/slightly worse. Still requires no bad Q1/support
    # and non-regressive Wasserstein.
    temporal_only = bool(
        tier == "ota"
        and score_gain_frac >= MIN_TEMPORAL_ONLY_SCORE_GAIN_FRAC
        and q2_ok
        and q1_not_bad
        and support_not_bad
        and wasser_ok
    )

    borderline = bool(
        score_gain_frac >= MIN_SCORE_GAIN_FRAC
        and (ks_gain >= MIN_KS_GAIN * max(0.0, 1.0 - BORDERLINE_REL_MARGIN) or q2_gain_frac >= MIN_Q2_GAIN_FRAC)
        and q1_not_bad
        and support_not_bad
        and wasser_ok
    )

    near_borderline = bool(
        score_gain_frac >= MIN_SCORE_GAIN_FRAC
        and (ks_gain >= MIN_KS_GAIN * max(0.0, 1.0 - NEAR_BORDERLINE_REL_MARGIN) or q2_gain_frac >= MIN_Q2_GAIN_FRAC * 0.75)
        and q1_not_bad
        and support_not_bad
        and wasser_ok
    )

    accepted = bool(accepted_standard or temporal_only)
    reason = (
        "candidate_materially_beat_A1_on_VAL_Q1Q2_aligned"
        if accepted_standard
        else "candidate_temporal_Q2_beat_A1_on_VAL_with_guardrails"
        if temporal_only
        else "candidate_did_not_materially_beat_A1_on_VAL_Q1Q2_aligned"
    )

    block_ok = None
    block_reason = None

    if accepted or borderline or near_borderline:
        uid = str(best.get("candidate_uid", ""))
        cand_meta = loaded_candidates.get(uid, {})
        cand_df = cand_meta.get("val_df", None)
        if isinstance(cand_df, pd.DataFrame) and col in cand_df.columns:
            eval_active = baseline_by_col[col]["eval_active"]
            cand_vals = pd.to_numeric(cand_df.loc[eval_active, col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            cand_vals = _postprocess_col(col, cand_vals)
            block_ok, block_info = _block_stability_gate(real_vals, a1_vals, cand_vals, col, tier)
            block_reason = str(block_info.get("reason"))
        else:
            block_ok = False
            block_reason = "candidate_frame_unavailable_for_block_stability"

        if (borderline or near_borderline or temporal_only) and not block_ok:
            accepted = False
            reason = "candidate_failed_block_stability_guard"

    if borderline:
        if accepted:
            borderline_accepted += 1
        else:
            borderline_rejected += 1

    if near_borderline:
        if accepted:
            near_borderline_accepted += 1
        else:
            near_borderline_rejected += 1

    if temporal_only and accepted:
        temporal_only_accepted += 1

    # Update best row fields for comparison report.
    idx = comparison[
        (comparison["col"].astype(str) == str(col))
        & (comparison["candidate_uid"].astype(str) == str(best.get("candidate_uid", "")))
    ].index

    if len(idx):
        comparison.loc[idx, "baseline_score"] = baseline_score
        comparison.loc[idx, "baseline_q1_score"] = baseline_q1
        comparison.loc[idx, "baseline_q2_score"] = baseline_q2
        comparison.loc[idx, "score_gain_frac"] = score_gain_frac
        comparison.loc[idx, "q1_gain_frac"] = q1_gain_frac
        comparison.loc[idx, "q2_gain_frac"] = q2_gain_frac
        comparison.loc[idx, "ks_gain"] = ks_gain
        comparison.loc[idx, "wasserstein_delta"] = wasserstein_delta
        comparison.loc[idx, "support_jaccard_delta"] = support_jaccard_delta
        comparison.loc[idx, "accepted_decision"] = bool(accepted)
        comparison.loc[idx, "reason_decision"] = reason
        comparison.loc[idx, "borderline_acceptance"] = bool(borderline)
        comparison.loc[idx, "near_borderline_acceptance"] = bool(near_borderline)
        comparison.loc[idx, "block_stability_passed"] = block_ok
        comparison.loc[idx, "block_stability_reason"] = block_reason

    best["_score_gain_frac"] = score_gain_frac
    best["_q1_gain_frac"] = q1_gain_frac
    best["_q2_gain_frac"] = q2_gain_frac
    best["_ks_gain"] = ks_gain
    best["_wasserstein_delta"] = wasserstein_delta
    best["_support_jaccard_delta"] = support_jaccard_delta
    best["_reject_reason"] = reason

    if accepted:
        uid = str(best.get("candidate_uid", ""))
        cand_meta = loaded_candidates.get(uid, {})
        requires_downstream = bool(cand_meta.get("requires_downstream_test_materialization", False))

        if requires_downstream:
            requires_downstream_cols.append(col)

        selected_by_col[col] = {
            "selected_generator": str(best.get("generator")),
            "selected_candidate_id": str(best.get("candidate_id")),
            "selected_candidate_uid": uid,
            "reason": reason,
            "val_path": str(best.get("val_path")),
            "test_path": str(best.get("test_path")) if pd.notna(best.get("test_path")) else None,
            "requires_downstream_test_materialization": requires_downstream,
            "score": candidate_score,
            "q1_score": candidate_q1,
            "q2_score": candidate_q2,
            "selection_baseline": "A1_VAL_baseline",
        }

        selected_by_col_rich[col] = dict(selected_by_col[col])
        selected_by_col_rich[col].update({
            "col": col,
            "tier": tier,
            "baseline_score": baseline_score,
            "baseline_q1_score": baseline_q1,
            "baseline_q2_score": baseline_q2,
            "selected_score": candidate_score,
            "selected_q1_score": candidate_q1,
            "selected_q2_score": candidate_q2,
            "score_gain_frac": score_gain_frac,
            "q1_gain_frac": q1_gain_frac,
            "q2_gain_frac": q2_gain_frac,
            "ks_gain": ks_gain,
            "wasserstein_delta": wasserstein_delta,
            "support_jaccard_delta": support_jaccard_delta,
            "borderline_acceptance": bool(borderline),
            "near_borderline_acceptance": bool(near_borderline),
            "temporal_only_acceptance": bool(temporal_only),
            "block_stability_passed": block_ok,
            "block_stability_reason": block_reason,
        })

        selected = dict(best)
        selected["accepted"] = True
        selected["accepted_decision"] = True
        selected["reason_decision"] = reason
        selected["selected_generator_for_col"] = True
        selected_rows_for_report.append(selected)

        if len(idx):
            comparison.loc[idx, "selected_generator_for_col"] = True

    else:
        best_rejected_rows.append({
            "col": col,
            "tier": tier,
            "has_optional_candidate": True,
            "best_rejected_generator": str(best.get("generator")),
            "best_rejected_candidate_uid": str(best.get("candidate_uid")),
            "best_rejected_candidate_id": str(best.get("candidate_id")),
            "best_rejected_score": candidate_score,
            "best_rejected_q1_score": candidate_q1,
            "best_rejected_q2_score": candidate_q2,
            "best_rejected_ks": candidate_ks,
            "best_rejected_wasserstein": candidate_w,
            "best_rejected_support_jaccard": candidate_support_j,
            "best_rejected_score_gain_frac": score_gain_frac,
            "best_rejected_q1_gain_frac": q1_gain_frac,
            "best_rejected_q2_gain_frac": q2_gain_frac,
            "best_rejected_ks_gain": ks_gain,
            "best_rejected_wasserstein_delta": wasserstein_delta,
            "best_rejected_support_jaccard_delta": support_jaccard_delta,
            "best_rejected_accepted_decision": False,
            "best_rejected_reason": reason,
        })

        _select_a1("no_candidate_materially_beat_A1_on_VAL_Q1Q2_aligned", best=best)

# Ensure every column has a rich record.
missing_rich_cols = [c for c in PROTO_COLS if c not in selected_by_col_rich]
if missing_rich_cols:
    raise RuntimeError(f"[Cell10.8] Internal error: missing rich selection rows for cols={missing_rich_cols[:20]}")


# ----------------------------------------------------------
# 8) Save outputs
# ----------------------------------------------------------
comparison_report_path = os.path.join(REPDIR, "cell10_portfolio_val_comparison.csv")
best_rejected_path = os.path.join(REPDIR, "cell10_best_rejected_candidate_by_col.csv")
selected_json_path = os.path.join(ARTDIR, "cell10_selected_generator_by_col.json")
selected_rich_json_path = os.path.join(ARTDIR, "cell10_selected_generator_by_col_rich.json")
selection_summary_path = os.path.join(ARTDIR, "cell10_portfolio_selection_summary.json")
candidate_registry_v3_path = os.path.join(ARTDIR, "cell10_portfolio_candidate_registry_v3.json")

comparison.to_csv(comparison_report_path, index=False)
a0_control_audit.to_csv(a0_control_audit_path, index=False)
pd.DataFrame(best_rejected_rows).to_csv(best_rejected_path, index=False)
_write_json(selected_json_path, selected_by_col)
_write_json(selected_rich_json_path, selected_by_col_rich)

selected_counts = Counter([v["selected_generator"] for v in selected_by_col.values()])
selected_non_a1_cols = [
    c for c, v in selected_by_col.items()
    if v["selected_generator"] != "A1_temporal_block_bootstrap"
]

candidate_registry_v3 = {
    "version": "cell10_portfolio_candidate_registry_v3_6",
    "candidate_files": {
        uid: {
            k: v for k, v in meta.items()
            if k not in {"val_df"}
        }
        for uid, meta in loaded_candidates.items()
    },
    "candidate_duplicate_initial_count": int(len(candidate_duplicate_rows)),
    "candidate_duplicate_after_rescue_count": int(len(rescue_duplicate_rows)),
    "candidate_count_raw_before_dedup": int(len(candidate_files_raw)),
    "candidate_count_after_final_dedup": int(len(candidate_files)),
    "loaded_candidate_count": int(len(loaded_candidates)),
}
_write_json(candidate_registry_v3_path, candidate_registry_v3)

no_optional_candidate_count = int(
    sum(
        1
        for c in PROTO_COLS
        if len(
            comparison[
                (comparison["col"].astype(str) == str(c))
                & (comparison["variant"].astype(str) == "A2_candidate")
            ]
        ) == 0
    )
)

custom_registry_rows_n = int((eligibility_audit["source"] == "candidate_registry_v3_loaded_candidate").sum())

summary = {
    "version": "cell10_8_val_only_portfolio_selection_summary_v3_5_q1q2_aligned_pure_selector",
    "selection_split": "VAL",
    "selection_baseline": "A1_VAL_baseline",
    "controlled_variants": {
        "A0": {
            "role": "diagnostic_control_only",
            "used_for_selection": False,
            "val_path": a0_val_path,
            "test_path_recorded_only": a0_test_path,
        },
        "A1": {
            "role": "selection_baseline_for_A2",
            "used_for_selection": True,
            "val_path": a1_val_path,
            "test_path_recorded_only": a1_test_path,
        },
        "A2": {
            "role": "selected_generator_portfolio",
            "selection_rule": "candidate_must_materially_beat_A1_on_VAL_using_Q1_Q2_aligned_metrics",
        },
    },
    "test_used_for_selection": False,
    "test_values_read": False,
    "test_used_for_candidate_fitting": False,
    "test_used_for_threshold_tuning": False,
    "test_rescue_materialized": False,
    "test_usage": "final_QA_only_downstream",
    "eval_mask_policy": {
        "A0_control": "real_VAL_mask",
        "A1_and_candidates": "real_VAL_observed_mask_intersect_A1_VAL_synthetic_mask",
        "rationale": "A1/A2 selection must respect synthetic capture semantics, but scoring can only use real VAL rows where observed real values exist.",
    },
    "policy": {
        "A1_always_included": True,
        "A0_diagnostic_only": True,
        "optional_generators_keep_only_if_VAL_beats_A1": True,
        "custom_registry_candidates_require_valid_VAL_and_TEST_metadata": True,
        "q1_q2_aligned_selector": True,
        "tier_score_weights": TIER_WEIGHTS,
        "multi_lag_lags": MULTILAG_LAGS,
        "min_score_gain_frac": MIN_SCORE_GAIN_FRAC,
        "min_ks_gain": MIN_KS_GAIN,
        "min_q2_gain_frac": MIN_Q2_GAIN_FRAC,
        "max_q1_regress_frac": MAX_Q1_REGRESS_FRAC,
        "max_wasserstein_regress_frac": MAX_WASSERSTEIN_REGRESS_FRAC,
        "max_support_jaccard_regress": MAX_SUPPORT_JACCARD_REGRESS,
        "min_temporal_only_score_gain_frac": MIN_TEMPORAL_ONLY_SCORE_GAIN_FRAC,
        "borderline_relative_margin_over_ks_threshold": BORDERLINE_REL_MARGIN,
        "near_borderline_relative_margin_over_ks_threshold": NEAR_BORDERLINE_REL_MARGIN,
        "borderline_or_near_borderline_requires_block_stability": True,
        "val_block_count": VAL_BLOCK_COUNT,
        "val_block_required_win_frac": VAL_BLOCK_REQUIRED_WIN_FRAC,
        "val_block_max_bad_frac": VAL_BLOCK_MAX_BAD_FRAC,
        "bounded_metric_n": MAX_METRIC_N,
        "bounded_ks_grid": MAX_KS_GRID,
        "bounded_wasserstein_points": MAX_WASSERSTEIN_POINTS,
        "markov_rescue_scope": "disabled_in_pure_selector",
        "markov_rescue_excludes_unique_like": None,
        "markov_rescue_excludes_dense_aggregate_like": None,
        "markov_rescue_mask_source": None,
        "in_selector_candidate_generation": False,
    },
    "selected_counts": dict(selected_counts),
    "selected_non_A1_cols": selected_non_a1_cols,
    "selected_non_A1_count": int(len(selected_non_a1_cols)),
    "all_optional_generators_rejected_by_VAL": bool(len(selected_non_a1_cols) == 0),
    "all_A1_selection_is_valid": bool(len(selected_non_a1_cols) == 0),
    "requires_downstream_test_materialization_cols": requires_downstream_cols,
    "requires_downstream_test_materialization_count": int(len(requires_downstream_cols)),
    "no_optional_candidate_available_count": int(no_optional_candidate_count),
    "best_rejected_candidate_audit_path": best_rejected_path,
    "candidate_duplicate_audit_path": candidate_duplicate_audit_path,
    "candidate_duplicate_audit_after_rescue_path": rescue_duplicate_audit_path,
    "candidate_discovery_audit_path": globals().get("CELL10_8_CANDIDATE_DISCOVERY_AUDIT_PATH", ""),
    "unregistered_file_scan_enabled": bool(CFG.get("cell10_8_allow_unregistered_file_scan", False)),
    "a0_a1_val_control_audit_path": a0_control_audit_path,
    "candidate_eligibility_audit_path": candidate_eligibility_audit_path,
    "candidate_duplicate_initial_count": int(len(candidate_duplicate_rows)),
    "candidate_duplicate_after_rescue_count": int(len(rescue_duplicate_rows)),
    "candidate_count_raw_before_dedup": int(len(candidate_files_raw)),
    "candidate_count_after_final_dedup": int(len(candidate_files)),
    "loaded_candidate_count": int(len(loaded_candidates)),
    "custom_registry_eligibility_rows_n": int(custom_registry_rows_n),
    "borderline_acceptance_count": int(borderline_accepted),
    "borderline_rejected_count": int(borderline_rejected),
    "near_borderline_acceptance_count": int(near_borderline_accepted),
    "near_borderline_rejected_count": int(near_borderline_rejected),
    "temporal_only_acceptance_count": int(temporal_only_accepted),
    "markov_rescue_enabled": bool(ENABLE_MARKOV_RESCUE),
    "markov_rescue_val_only": False,
    "markov_rescue_generated_cols": next(iter(rescue_candidates.values()), {}).get("cols", []) if rescue_candidates else [],
    "markov_rescue_generated_cols_n": len(next(iter(rescue_candidates.values()), {}).get("cols", [])) if rescue_candidates else 0,
    "candidate_ids_loaded": sorted(loaded_candidates.keys()),
    "reports": {
        "comparison": comparison_report_path,
        "eligibility": candidate_eligibility_audit_path,
        "best_rejected": best_rejected_path,
        "a0_a1_control": a0_control_audit_path,
    },
}
_write_json(selection_summary_path, summary)

selection_contract_path = os.path.join(CONTRACT_DIR, "cell10_8_val_only_selector_contract_v3_6_THESIS.json")
selector_contract = {
    "version": "cell10_8_val_only_selector_contract_v3_6_THESIS",
    "component": "VAL_only_generator_portfolio_selector",
    "selection_split": "VAL",
    "selection_baseline": "A1_VAL_baseline",
    "a0_used_for_selection": False,
    "test_values_read": False,
    "test_used_for_fitting": False,
    "test_used_for_thresholding": False,
    "test_used_for_selection": False,
    "test_rescue_materialization": False,
    "in_selector_candidate_generation": False,
    "candidate_registry_path": candidate_registry_path,
    "comparison_report_path": comparison_report_path,
    "a0_control_audit_path": a0_control_audit_path,
    "selected_json_path": selected_json_path,
    "selected_rich_json_path": selected_rich_json_path,
    "selection_summary_path": selection_summary_path,
    "candidate_registry_v3_path": candidate_registry_v3_path,
    "candidate_duplicate_audit_path": candidate_duplicate_audit_path,
    "candidate_duplicate_audit_after_rescue_path": rescue_duplicate_audit_path,
    "candidate_discovery_audit_path": globals().get("CELL10_8_CANDIDATE_DISCOVERY_AUDIT_PATH", ""),
    "unregistered_file_scan_enabled": bool(CFG.get("cell10_8_allow_unregistered_file_scan", False)),
    "candidate_eligibility_audit_path": candidate_eligibility_audit_path,
    "best_rejected_path": best_rejected_path,
    "selected_counts": dict(selected_counts),
    "selected_non_a1_cols_n": int(len(selected_non_a1_cols)),
    "requires_downstream_test_materialization_count": int(len(requires_downstream_cols)),
    "loaded_candidate_count": int(len(loaded_candidates)),
    "candidate_count_raw_before_dedup": int(len(candidate_files_raw)),
    "candidate_count_after_final_dedup": int(len(candidate_files)),
    "custom_registry_eligibility_rows_n": int(custom_registry_rows_n),
    "markov_rescue_enabled": False,
    "markov_rescue_generated_cols_n": 0,
    "leakage_status": {
        "uses_train_values_for_fitting": False,
        "uses_train_profiles_for_eligibility_metadata": True,
        "uses_val_values_for_selection": True,
        "uses_test_values_for_fitting": False,
        "uses_test_values_for_thresholding": False,
        "uses_test_values_for_selection": False,
        "uses_test_values_for_repair": False,
        "materializes_test_rescue": False,
        "overwrites_final_artifact": False,
    },
    "paper_claim_status": (
        "Cell 10.8 performs VAL-only generator selection against A1. "
        "It does not read TEST values and does not generate candidates."
    ),
}
_write_json(selection_contract_path, selector_contract)
globals()["CELL10_8_SELECTOR_CONTRACT_PATH"] = selection_contract_path

if "RUN_META" in globals():
    RUN_META.setdefault("selector_contracts", {})
    RUN_META["selector_contracts"]["cell10_8_val_only_selector"] = selector_contract


log(f"[Cell10.8] Saved VAL comparison report: {comparison_report_path}")
log(f"[Cell10.8] Saved A0/A1 VAL control audit: {a0_control_audit_path}")
log(f"[Cell10.8] Saved candidate eligibility audit: {candidate_eligibility_audit_path}")
log(f"[Cell10.8] Saved selected-generator direct registry: {selected_json_path}")
log(f"[Cell10.8] Saved selected-generator rich registry: {selected_rich_json_path}")
log(f"[Cell10.8] Saved selection summary: {selection_summary_path}")
log(f"[Cell10.8] Saved candidate registry v3: {candidate_registry_v3_path}")
log(f"[Cell10.8] Saved selector contract: {selection_contract_path}")

log(
    "[Cell10.8] Selection counts: "
    f"{dict(selected_counts)} | selection_baseline=A1 | A0_used_for_selection=False | "
    f"non_A1={len(selected_non_a1_cols)} | "
    f"all_optional_rejected_by_VAL={len(selected_non_a1_cols) == 0} | "
    f"needs_test_materialization={len(requires_downstream_cols)} | "
    f"no_optional_candidate_available={no_optional_candidate_count} | "
    f"borderline_accepted={borderline_accepted} | borderline_rejected={borderline_rejected} | "
    f"near_borderline_accepted={near_borderline_accepted} | near_borderline_rejected={near_borderline_rejected} | "
    f"temporal_only_accepted={temporal_only_accepted} | "
    f"markov_rescue_val_cols={summary['markov_rescue_generated_cols_n']} | "
    f"raw_candidates={len(candidate_files_raw)} | initial_duplicates={len(candidate_duplicate_rows)} | "
    f"after_rescue_duplicates={len(rescue_duplicate_rows)} | deduped_candidates={len(candidate_files)} | "
    f"custom_registry_eligibility_rows={custom_registry_rows_n}"
)

log("--- END: Cell 10.8 — VAL-only portfolio comparison and selection (v3.6 registry-only pure VAL selector, Q1/Q2-aligned STUDY-THESIS) ---")