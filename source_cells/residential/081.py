# ==========================================================
# CELL 12.f.2 - Observability mask model candidates
# v1.2 STUDY-THESIS strict Q3 mask VAL candidates, no VAL-anchored candidate design
#
# Role:
#   - Generate VAL-length mask candidates for direct observability targets.
#   - Fit all non-A1 candidates on TRAIN only.
#   - Evaluate candidates on VAL only.
#   - Carry A1 mask baseline from Cell 12.f.1 as selectable reference.
#   - Do NOT include A0 real-VAL diagnostic as selectable candidate.
#
# Candidate families:
#   1) A1_train_fitted_mask_baseline_reference
#   2) ConstantRateMaskFallback
#   3) RegimeConditionedMarkovMask
#   4) SemiMarkovMask
#   5) LogisticAutoregressiveMaskModel
#   6) RareEventZeroInflatedMask
#   7) ValRateAnchoredPlateauConstantMask
#   8) ValRateAnchoredSegmentPlateauMask
#   9) PlateauPreservingTwoStateMask
#
# Paper 2-style logistic observation model:
#   previous observation + regime + TOD + event count
#
# Strict rules:
#   - Do NOT read real TEST mask values.
#   - Do NOT generate TEST-length masks.
#   - Do NOT select final mask model.
#   - Do NOT mutate A1 outputs.
#
# Outputs:
#   synthetic/IOT_OBSERVABILITY_VAL_CANDIDATES_ALL.parquet
#   reports/cell12f2_mask_val_candidate_metrics.csv
#   reports/cell12f2_mask_candidate_inventory.csv
#   reports/cell12f2_mask_candidate_audit.csv
#   reports/cell12f2_mask_candidate_coverage.csv
#   reports/cell12f2_mask_contract.json
#   artifacts/cell12f2_mask_val_candidate_manifest.json
# ==========================================================

log("--- START: Cell 12.f.2 - Observability mask VAL candidates (v1.2 no VAL-anchored candidate design) ---")

import os
import re
import gc
import json
import math
import hashlib
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_12f2 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_OBSERVABILITY_COLS",
    "IOT_OBSERVABILITY_TARGET_CONTRACT_DF",
    "IOT_OBSERVABILITY_FAMILY_CONTRACT_DF",
    "CELL12F_OBSERVABILITY_CONTRACT",
    "A1_OBSERVABILITY_MASKS_VAL",
    "CELL12F1_MASK_BASELINE_METRICS_DF",
    "CELL12F1_MASK_BASELINE_AUDIT_DF",
    "CELL12F1_MASK_BASELINE_CONTRACT",
]
_missing_12f2 = [k for k in _required_12f2 if k not in globals()]
if _missing_12f2:
    raise RuntimeError(f"[Cell12.f.2] Missing required globals from prior cells: {_missing_12f2}")

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

IOT_OBSERVABILITY_COLS = list(map(str, IOT_OBSERVABILITY_COLS))

if len(IOT_OBSERVABILITY_COLS) <= 0:
    raise RuntimeError("[Cell12.f.2] IOT_OBSERVABILITY_COLS is empty.")

missing_tr = sorted([c for c in IOT_OBSERVABILITY_COLS if c not in df_tr.columns])
missing_val = sorted([c for c in IOT_OBSERVABILITY_COLS if c not in df_val.columns])
missing_te_schema = sorted([c for c in IOT_OBSERVABILITY_COLS if c not in df_te.columns])
missing_a1 = sorted([c for c in IOT_OBSERVABILITY_COLS if c not in A1_OBSERVABILITY_MASKS_VAL.columns])

if missing_tr or missing_val or missing_te_schema or missing_a1:
    raise RuntimeError(
        "[Cell12.f.2] Observability targets missing from required schemas/artifacts. "
        f"missing_train={missing_tr[:10]} | missing_val={missing_val[:10]} | "
        f"missing_test_schema={missing_te_schema[:10]} | missing_a1={missing_a1[:10]}"
    )

CELL12F2_VERSION = "cell12f2_observability_mask_val_candidates_strict_v1_2_no_val_anchored_candidate_design"

CFG["cell12f2_version"] = CELL12F2_VERSION
CFG["cell12f2_TEST_real_values_used"] = False
CFG["cell12f2_selection_done_here"] = False
CFG["cell12f2_generator_fit_done_here"] = True
CFG["cell12f2_train_values_used_for_fitting"] = True
CFG["cell12f2_val_values_used_for_evaluation"] = True
CFG["cell12f2_test_length_materialization_done_here"] = False
CFG["cell12f2_quality_dimension"] = "Q3_observability"
CFG["cell12f2_VAL_values_used_for_targeted_candidate_design"] = False
CFG["cell12f2_uses_val_rate_anchor"] = False
CFG["cell12f2_targeted_val_anchored_families_disabled"] = True

CFG.setdefault("cell12f2_mask_regime_bins", 24)
CFG.setdefault("cell12f2_mask_markov_eps", 1e-6)
CFG.setdefault("cell12f2_min_train_observed_n", 1)
CFG.setdefault("cell12f2_min_val_observed_n", 1)
CFG.setdefault("cell12f2_semimarkov_max_run_len", 250000)
CFG.setdefault("cell12f2_logistic_max_train_rows", 200000)
CFG.setdefault("cell12f2_logistic_min_positive", 10)
CFG.setdefault("cell12f2_logistic_min_negative", 10)
CFG.setdefault("cell12f2_mask_c2st_sample_cap", 50000)

# Candidate scoring weights. Selection happens in 12.f.3.
CFG.setdefault("cell12f2_score_obs_rate_weight", 2.0)
CFG.setdefault("cell12f2_score_p11_weight", 1.0)
CFG.setdefault("cell12f2_score_p00_weight", 1.0)
CFG.setdefault("cell12f2_score_run_ks_weight", 0.75)
CFG.setdefault("cell12f2_score_regime_rate_weight", 1.0)
CFG.setdefault("cell12f2_score_c2st_weight", 0.50)

MASK_CANDIDATE_FAMILIES_12F2 = [
    "A1_train_fitted_mask_baseline_reference",
    "ConstantRateMaskFallback",
    "RegimeConditionedMarkovMask",
    "SemiMarkovMask",
    "LogisticAutoregressiveMaskModel",
    "RareEventZeroInflatedMask",
    "ValRateAnchoredPlateauConstantMask",
    "ValRateAnchoredSegmentPlateauMask",
    "PlateauPreservingTwoStateMask",
]

NON_A1_MASK_CANDIDATE_FAMILIES_12F2 = [
    "ConstantRateMaskFallback",
    "RegimeConditionedMarkovMask",
    "SemiMarkovMask",
    "LogisticAutoregressiveMaskModel",
    "RareEventZeroInflatedMask",
    "ValRateAnchoredPlateauConstantMask",
    "ValRateAnchoredSegmentPlateauMask",
    "PlateauPreservingTwoStateMask",
]

TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2 = [
    "RareEventZeroInflatedMask",
    "ValRateAnchoredPlateauConstantMask",
    "ValRateAnchoredSegmentPlateauMask",
    "PlateauPreservingTwoStateMask",
]

CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP = {
    "iot__entity_obs__smart_plug_cloud_connection": "RareEventZeroInflatedMask",
    "iot__entity_stale__coffee_maker": "RareEventZeroInflatedMask",
    "iot__entity_stale__coffee_maker_local_control": "RareEventZeroInflatedMask",
    "iot__entity_stale__smart_plug_current": "RareEventZeroInflatedMask",
    "iot__entity_stale__smart_plug_ps5": "RareEventZeroInflatedMask",
    "iot__entity_stale__smart_plug_ps5_current": "RareEventZeroInflatedMask",
    "iot__entity_stale__coffee_maker_child_lock": "ValRateAnchoredPlateauConstantMask",
    "iot__entity_stale__coffee_maker_connectivity": "ValRateAnchoredPlateauConstantMask",
    "iot__entity_stale__coffee_maker_remote_start": "ValRateAnchoredPlateauConstantMask",
    "iot__entity_stale__smart_plug_cloud_connection": "ValRateAnchoredSegmentPlateauMask",
    "iot__entity_stale__smart_plug_ps5_led": "PlateauPreservingTwoStateMask",
}

# VAL-anchored targeted families are retained as rows for schema compatibility only.
# They are diagnostic-invalid in v1.2 and must never be selectable.
CELL12F2_VAL_ANCHORED_TARGETED_FAMILIES_DISABLED = True

# ----------------------------------------------------------
# 1) Validate upstream Cell 12.f.1 contract
# ----------------------------------------------------------
contract1 = CELL12F1_MASK_BASELINE_CONTRACT
if not isinstance(contract1, dict):
    raise RuntimeError("[Cell12.f.2] CELL12F1_MASK_BASELINE_CONTRACT is not a dict.")

version1 = str(contract1.get("version", ""))
if "cell12f1_observability_mask_baseline_strict_v1_1" not in version1 and "cell12f1_observability_mask_baseline_strict_v1_0" not in version1:
    raise RuntimeError(
        "[Cell12.f.2] Unexpected Cell 12.f.1 contract version. "
        f"Expected v1.0/v1.1 compatible contract, got: {version1}"
    )

if bool(contract1.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell12.f.2] Cell 12.f.1 contract indicates TEST values were used.")

if bool(contract1.get("a0_selectable", True)):
    raise RuntimeError("[Cell12.f.2] Refusing to run because A0 diagnostic baseline appears selectable.")


# ----------------------------------------------------------
# 2) Generic helpers
# ----------------------------------------------------------
def _json_sanitize_12f2(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12f2(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12f2(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12f2(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12f2(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12f2(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12f2(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12f2(obj.to_dict())
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

def _write_json_12f2(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12f2(payload), f, indent=2, sort_keys=True)

def _sha256_file_12f2(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _stable_seed_12f2(name: str, offset: int = 0) -> int:
    h = hashlib.sha256(f"{SEED}|12f2|{offset}|{name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)

def _rng_12f2(name: str, offset: int = 0):
    return np.random.default_rng(_stable_seed_12f2(name, offset=offset))

def _safe_float_12f2(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_bool_12f2(x, default=False) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return bool(default)
    try:
        if pd.isna(x):
            return bool(default)
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", "", "nan", "none", "null"}:
        return False
    return bool(default)

def _to_num_array_12f2(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _coerce_mask_nan_12f2(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    arr[finite & (arr < 0.5)] = 0.0
    arr[finite & (arr >= 0.5)] = 1.0
    return arr

def _mask_observed_values_12f2(frame: pd.DataFrame, col: str) -> tuple:
    raw = _to_num_array_12f2(frame, col)
    arr = _coerce_mask_nan_12f2(raw)
    obs = np.isfinite(arr)
    return arr, obs

def _entity_from_obs_col_12f2(col: str) -> str:
    s = str(col)
    if s.startswith("iot__entity_obs__"):
        return s.split("iot__entity_obs__", 1)[1]
    if s.startswith("iot__entity_stale__"):
        return s.split("iot__entity_stale__", 1)[1]
    if s.startswith("iot__"):
        s = s[5:]
    return s.split("__")[0] if "__" in s else s

# ----------------------------------------------------------
# 3) Contract lookup
# ----------------------------------------------------------
contract_df = IOT_OBSERVABILITY_TARGET_CONTRACT_DF.copy()
contract_df["col"] = contract_df["col"].astype(str)

contract_map = {str(r["col"]): r.to_dict() for _, r in contract_df.iterrows()}

def _obs_kind_for_col_12f2(col: str) -> str:
    return str(contract_map.get(str(col), {}).get("obs_kind", ""))

def _mask_role_for_col_12f2(col: str) -> str:
    return str(contract_map.get(str(col), {}).get("mask_role", ""))

def _recommended_for_col_12f2(col: str) -> str:
    return str(contract_map.get(str(col), {}).get("recommended_12f2_candidate", ""))

def _eligible_constant_12f2(col: str) -> bool:
    return _safe_bool_12f2(contract_map.get(str(col), {}).get("eligible_constant_rate_fallback", True), True)

def _eligible_markov_12f2(col: str) -> bool:
    return _safe_bool_12f2(contract_map.get(str(col), {}).get("eligible_regime_conditioned_markov_mask", False), False)

def _eligible_semimarkov_12f2(col: str) -> bool:
    return _safe_bool_12f2(contract_map.get(str(col), {}).get("eligible_semimarkov_mask", False), False)

def _eligible_logistic_12f2(col: str) -> bool:
    return _safe_bool_12f2(contract_map.get(str(col), {}).get("eligible_logistic_autoregressive_mask_model", False), False)

# ----------------------------------------------------------
# 4) VAL mask and time/regime/event-count features
# ----------------------------------------------------------
def _build_val_observed_mask_12f2() -> pd.DataFrame:
    data = {}
    for col in IOT_OBSERVABILITY_COLS:
        _, obs = _mask_observed_values_12f2(df_val, col)
        data[col] = obs.astype(np.float32)
    return pd.DataFrame(data, index=df_val.index, dtype=np.float32)

IOT_OBSERVABILITY_AVAIL_VAL_DF_12F2 = _build_val_observed_mask_12f2()

def _detect_time_col_12f2(frame: pd.DataFrame):
    preferred = [
        "sec_epoch_s__canon",
        "sec",
        "timestamp",
        "time",
        "datetime",
        "date_time",
    ]
    for c in preferred:
        if c in frame.columns:
            return c
    for c in frame.columns:
        s = str(c).lower()
        if s in {"sec", "time", "timestamp"} or "epoch" in s or s.endswith("__sec"):
            return c
    return None

def _time_regime_bins_12f2(frame: pd.DataFrame, n_bins: int) -> np.ndarray:
    n = int(len(frame))
    n_bins = int(max(1, n_bins))

    tcol = _detect_time_col_12f2(frame)
    if tcol is not None:
        raw = pd.to_numeric(frame[tcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite_n = int(np.isfinite(raw).sum())
        if finite_n >= max(10, int(0.50 * n)):
            sec_day = np.mod(raw, 86400.0)
            bins = np.floor(sec_day / (86400.0 / n_bins)).astype(np.float64)
            bins[~np.isfinite(bins)] = 0.0
            return np.clip(bins.astype(np.int64), 0, n_bins - 1)

    pos = np.arange(n, dtype=np.float64)
    bins = np.floor((pos / max(n, 1)) * n_bins)
    return np.clip(bins.astype(np.int64), 0, n_bins - 1)

def _tod_features_12f2(frame: pd.DataFrame) -> tuple:
    n = int(len(frame))
    tcol = _detect_time_col_12f2(frame)
    if tcol is not None:
        raw = pd.to_numeric(frame[tcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        if np.isfinite(raw).sum() >= max(10, int(0.50 * n)):
            sec_day = np.mod(raw, 86400.0)
            theta = 2.0 * np.pi * sec_day / 86400.0
            sinv = np.sin(theta)
            cosv = np.cos(theta)
            sinv[~np.isfinite(sinv)] = 0.0
            cosv[~np.isfinite(cosv)] = 1.0
            return sinv.astype(np.float32), cosv.astype(np.float32)

    pos = np.arange(n, dtype=np.float64)
    theta = 2.0 * np.pi * (pos / max(n, 1))
    return np.sin(theta).astype(np.float32), np.cos(theta).astype(np.float32)

def _event_count_feature_12f2(frame: pd.DataFrame) -> np.ndarray:
    candidates = [
        "iot__events_total",
        "events_total",
        "iot__events_entity_unique",
        "iot__any_update_raw",
    ]
    for c in candidates:
        if c in frame.columns:
            x = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            x[~np.isfinite(x)] = 0.0
            return x.astype(np.float32)

    # Fallback: count available events_in_sec columns if present.
    event_cols = [c for c in frame.columns if str(c).startswith("events_in_sec__")]
    if event_cols:
        acc = np.zeros(len(frame), dtype=np.float32)
        for c in event_cols[:200]:
            x = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            acc += np.nan_to_num(x, nan=0.0).astype(np.float32)
        return acc

    return np.zeros(len(frame), dtype=np.float32)

REGIME_BINS_12F2 = int(CFG.get("cell12f2_mask_regime_bins", 24))
TRAIN_REGIME_12F2 = _time_regime_bins_12f2(df_tr, REGIME_BINS_12F2)
VAL_REGIME_12F2 = _time_regime_bins_12f2(df_val, REGIME_BINS_12F2)

TRAIN_TOD_SIN_12F2, TRAIN_TOD_COS_12F2 = _tod_features_12f2(df_tr)
VAL_TOD_SIN_12F2, VAL_TOD_COS_12F2 = _tod_features_12f2(df_val)

TRAIN_EVENT_COUNT_12F2 = _event_count_feature_12f2(df_tr)
VAL_EVENT_COUNT_12F2 = _event_count_feature_12f2(df_val)

# ----------------------------------------------------------
# 5) Metric helpers
# ----------------------------------------------------------
def _transition_probs_12f2(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "p00": np.nan,
            "p01": np.nan,
            "p10": np.nan,
            "p11": np.nan,
            "transition_rate": np.nan,
            "n_pairs": 0,
        }

    arr = np.where(arr >= 0.5, 1, 0).astype(np.int8)
    a = arr[:-1]
    b = arr[1:]

    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))

    n0 = n00 + n01
    n1 = n10 + n11
    total = n0 + n1

    def div(x, y):
        return float(x / y) if y else np.nan

    return {
        "p00": div(n00, n0),
        "p01": div(n01, n0),
        "p10": div(n10, n1),
        "p11": div(n11, n1),
        "transition_rate": div(n01 + n10, total),
        "n_pairs": int(total),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }

def _run_lengths_12f2(x: np.ndarray, state=None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)

    arr = np.where(arr >= 0.5, 1, 0).astype(np.int8)
    lengths = []
    cur = int(arr[0])
    run = 1

    for v in arr[1:]:
        v = int(v)
        if v == cur:
            run += 1
        else:
            if state is None or cur == int(state):
                lengths.append(run)
            cur = v
            run = 1

    if state is None or cur == int(state):
        lengths.append(run)

    return np.asarray(lengths, dtype=np.float64)

def _empirical_ks_12f2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 1.0

    vals = np.unique(np.concatenate([a, b]))
    if vals.size == 0:
        return 0.0

    aa = np.sort(a)
    bb = np.sort(b)

    ca = np.searchsorted(aa, vals, side="right") / float(aa.size)
    cb = np.searchsorted(bb, vals, side="right") / float(bb.size)

    return float(np.max(np.abs(ca - cb)))

def _regime_obs_rate_error_12f2(real_full: np.ndarray, syn_full: np.ndarray, obs: np.ndarray, val_regime: np.ndarray) -> float:
    real_full = np.asarray(real_full, dtype=np.float64)
    syn_full = np.asarray(syn_full, dtype=np.float64)
    obs = np.asarray(obs, dtype=bool)
    val_regime = np.asarray(val_regime, dtype=np.int64)

    errs = []
    weights = []

    for r in np.unique(val_regime):
        idx = obs & (val_regime == int(r))
        n = int(idx.sum())
        if n <= 0:
            continue

        rr = real_full[idx]
        ss = syn_full[idx]
        rr = rr[np.isfinite(rr)]
        ss = ss[np.isfinite(ss)]

        if rr.size == 0 or ss.size == 0:
            continue

        errs.append(abs(float(np.mean(rr)) - float(np.mean(ss))))
        weights.append(n)

    if not errs:
        return np.nan

    return float(np.average(np.asarray(errs, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))

def _mask_c2st_auc_proxy_12f2(real: np.ndarray, syn: np.ndarray, rng: np.random.Generator) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return np.nan

    real = np.asarray(real, dtype=np.float64)
    syn = np.asarray(syn, dtype=np.float64)
    real = real[np.isfinite(real)]
    syn = syn[np.isfinite(syn)]

    n = int(min(real.size, syn.size))
    if n < 20:
        return np.nan

    cap = int(CFG.get("cell12f2_mask_c2st_sample_cap", 50000))
    n = int(min(n, cap))

    real_idx = rng.choice(real.size, size=n, replace=False) if real.size > n else np.arange(real.size)
    syn_idx = rng.choice(syn.size, size=n, replace=False) if syn.size > n else np.arange(syn.size)

    r = real[real_idx].astype(np.float64)
    s = syn[syn_idx].astype(np.float64)

    def features(x):
        x = np.asarray(x, dtype=np.float64)
        prev = np.concatenate([[x[0]], x[:-1]])
        nxt = np.concatenate([x[1:], [x[-1]]])
        change_prev = (x != prev).astype(np.float64)
        change_next = (x != nxt).astype(np.float64)
        return np.column_stack([x, prev, nxt, change_prev, change_next])

    X = np.vstack([features(r), features(s)])
    y = np.concatenate([np.zeros(r.size, dtype=np.int8), np.ones(s.size, dtype=np.int8)])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.35,
            random_state=int(rng.integers(0, 2**31 - 1)),
            stratify=y,
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            solver="liblinear",
            max_iter=500,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, p))
        return float(max(auc, 1.0 - auc))
    except Exception:
        return np.nan

def _mask_metrics_12f2(real: np.ndarray, syn: np.ndarray, real_full: np.ndarray, syn_full: np.ndarray, obs: np.ndarray, rng: np.random.Generator) -> dict:
    real = np.asarray(real, dtype=np.float64)
    syn = np.asarray(syn, dtype=np.float64)

    real = real[np.isfinite(real)]
    syn = syn[np.isfinite(syn)]

    real_n = int(real.size)
    syn_n = int(syn.size)

    if real_n == 0 or syn_n == 0:
        return {
            "eval_ok": False,
            "eval_reason": "empty_real_or_synthetic",
            "real_n": real_n,
            "syn_n": syn_n,
            "obs_rate_error": np.nan,
            "p11_error": np.nan,
            "p00_error": np.nan,
            "run_length_ks": np.nan,
            "regime_obs_rate_error": np.nan,
            "mask_c2st_auc": np.nan,
            "score": np.inf,
        }

    real = np.where(real >= 0.5, 1.0, 0.0)
    syn = np.where(syn >= 0.5, 1.0, 0.0)

    real_rate = float(np.mean(real))
    syn_rate = float(np.mean(syn))
    obs_rate_error = abs(real_rate - syn_rate)

    rt = _transition_probs_12f2(real)
    st = _transition_probs_12f2(syn)

    p11_error = abs(_safe_float_12f2(rt["p11"], np.nan) - _safe_float_12f2(st["p11"], np.nan))
    p00_error = abs(_safe_float_12f2(rt["p00"], np.nan) - _safe_float_12f2(st["p00"], np.nan))

    real_runs = _run_lengths_12f2(real)
    syn_runs = _run_lengths_12f2(syn)
    run_ks = _empirical_ks_12f2(real_runs, syn_runs)

    regime_err = _regime_obs_rate_error_12f2(
        real_full=real_full,
        syn_full=syn_full,
        obs=obs,
        val_regime=VAL_REGIME_12F2,
    )

    c2st_auc = _mask_c2st_auc_proxy_12f2(real, syn, rng)
    c2st_excess = abs(_safe_float_12f2(c2st_auc, 0.5) - 0.5)

    components = {
        "obs_rate_error": obs_rate_error,
        "p11_error": p11_error,
        "p00_error": p00_error,
        "run_length_ks": run_ks,
        "regime_obs_rate_error": regime_err,
        "c2st_excess": c2st_excess,
    }
    for k, v in list(components.items()):
        if not np.isfinite(v):
            components[k] = 1.0

    score = (
        float(CFG.get("cell12f2_score_obs_rate_weight", 2.0)) * components["obs_rate_error"]
        + float(CFG.get("cell12f2_score_p11_weight", 1.0)) * components["p11_error"]
        + float(CFG.get("cell12f2_score_p00_weight", 1.0)) * components["p00_error"]
        + float(CFG.get("cell12f2_score_run_ks_weight", 0.75)) * components["run_length_ks"]
        + float(CFG.get("cell12f2_score_regime_rate_weight", 1.0)) * components["regime_obs_rate_error"]
        + float(CFG.get("cell12f2_score_c2st_weight", 0.50)) * components["c2st_excess"]
    )

    return {
        "eval_ok": True,
        "eval_reason": "",
        "real_n": real_n,
        "syn_n": syn_n,
        "real_obs_rate": real_rate,
        "syn_obs_rate": syn_rate,
        "obs_rate_error": float(obs_rate_error),
        "real_p11": rt["p11"],
        "syn_p11": st["p11"],
        "p11_error": float(p11_error) if np.isfinite(p11_error) else np.nan,
        "real_p00": rt["p00"],
        "syn_p00": st["p00"],
        "p00_error": float(p00_error) if np.isfinite(p00_error) else np.nan,
        "real_transition_rate": rt["transition_rate"],
        "syn_transition_rate": st["transition_rate"],
        "real_run_count": int(real_runs.size),
        "syn_run_count": int(syn_runs.size),
        "real_run_median": float(np.median(real_runs)) if real_runs.size else np.nan,
        "syn_run_median": float(np.median(syn_runs)) if syn_runs.size else np.nan,
        "run_length_ks": float(run_ks),
        "regime_obs_rate_error": float(regime_err) if np.isfinite(regime_err) else np.nan,
        "mask_c2st_auc": c2st_auc,
        "mask_c2st_excess": c2st_excess,
        "score": float(score),
    }

# ----------------------------------------------------------
# 6) Candidate generators
# ----------------------------------------------------------
def _apply_val_mask_12f2(syn: np.ndarray, val_obs: np.ndarray) -> np.ndarray:
    out = np.asarray(syn, dtype=np.float64).copy()
    val_obs = np.asarray(val_obs, dtype=bool)

    if out.size != val_obs.size:
        raise RuntimeError(
            f"[Cell12.f.2] VAL mask length mismatch: syn={out.size} mask={val_obs.size}"
        )

    out[~val_obs] = np.nan
    out[val_obs] = np.where(out[val_obs] >= 0.5, 1.0, 0.0)
    return out

def _fit_constant_rate_12f2(train_x: np.ndarray) -> dict:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"valid": False, "reason": "empty_train"}
    x = np.where(x >= 0.5, 1.0, 0.0)
    return {
        "valid": True,
        "reason": "",
        "rate": float(np.mean(x)),
        "constant": bool(len(np.unique(x)) < 2),
    }

def _generate_constant_rate_12f2(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    if not bool(model.get("valid", False)):
        return np.full(n, np.nan, dtype=np.float64)
    rate = float(np.clip(model.get("rate", 0.0), 0.0, 1.0))
    if bool(model.get("constant", False)):
        return np.full(n, 1.0 if rate >= 0.5 else 0.0, dtype=np.float64)
    return (rng.random(n) < rate).astype(np.float64)

def _estimate_markov_12f2(train_x: np.ndarray) -> dict:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]
    eps = float(CFG.get("cell12f2_mask_markov_eps", 1e-6))

    if x.size == 0:
        return {"valid": False, "reason": "empty_train"}

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))

    if x.size < 2:
        return {
            "valid": True,
            "reason": "",
            "rate": rate,
            "constant": True,
            "p01": rate,
            "p10": 1.0 - rate,
            "init_p1": rate,
            "n_pairs": 0,
        }

    a = x[:-1].astype(np.int8)
    b = x[1:].astype(np.int8)

    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))

    p01 = (n01 + eps) / max(n00 + n01 + 2.0 * eps, eps)
    p10 = (n10 + eps) / max(n10 + n11 + 2.0 * eps, eps)

    return {
        "valid": True,
        "reason": "",
        "rate": rate,
        "constant": bool(len(np.unique(x)) < 2),
        "p01": float(np.clip(p01, 0.0, 1.0)),
        "p10": float(np.clip(p10, 0.0, 1.0)),
        "init_p1": rate,
        "n_pairs": int(a.size),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }

def _generate_markov_from_params_12f2(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    if not bool(model.get("valid", False)):
        return np.full(n, np.nan, dtype=np.float64)

    rate = float(np.clip(model.get("rate", 0.0), 0.0, 1.0))

    if bool(model.get("constant", False)):
        return np.full(n, 1.0 if rate >= 0.5 else 0.0, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    state = int(rng.random() < rate)
    out[0] = float(state)

    p01 = float(np.clip(model.get("p01", rate), 0.0, 1.0))
    p10 = float(np.clip(model.get("p10", 1.0 - rate), 0.0, 1.0))

    for i in range(1, n):
        if state == 0:
            state = int(rng.random() < p01)
        else:
            state = int(not (rng.random() < p10))
        out[i] = float(state)

    return out

def _fit_regime_markov_12f2(train_full: np.ndarray, train_obs: np.ndarray, train_regime: np.ndarray) -> dict:
    train_full = np.asarray(train_full, dtype=np.float64)
    train_obs = np.asarray(train_obs, dtype=bool)
    train_regime = np.asarray(train_regime, dtype=np.int64)

    global_model = _estimate_markov_12f2(train_full[train_obs])
    if not bool(global_model.get("valid", False)):
        return {
            "valid": False,
            "reason": "empty_global_train",
            "global_model": global_model,
            "regime_models": {},
        }

    regime_models = {}
    for r in np.unique(train_regime):
        idx = train_obs & (train_regime == int(r))
        if int(idx.sum()) < 5:
            continue
        m = _estimate_markov_12f2(train_full[idx])
        if bool(m.get("valid", False)):
            regime_models[int(r)] = m

    return {
        "valid": True,
        "reason": "",
        "global_model": global_model,
        "regime_models": regime_models,
    }

def _generate_regime_markov_12f2(model: dict, n: int, val_regime: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    if not bool(model.get("valid", False)):
        return np.full(n, np.nan, dtype=np.float64)

    global_model = model["global_model"]
    regime_models = model.get("regime_models", {})

    out = np.empty(n, dtype=np.float64)
    state = int(rng.random() < float(np.clip(global_model.get("rate", 0.0), 0.0, 1.0)))

    for i in range(n):
        r = int(val_regime[i]) if i < len(val_regime) else 0
        m = regime_models.get(r, global_model)

        if i == 0:
            out[i] = float(state)
            continue

        if bool(m.get("constant", False)):
            state = int(float(m.get("rate", 0.0)) >= 0.5)
        else:
            p01 = float(np.clip(m.get("p01", global_model.get("p01", 0.0)), 0.0, 1.0))
            p10 = float(np.clip(m.get("p10", global_model.get("p10", 1.0)), 0.0, 1.0))
            if state == 0:
                state = int(rng.random() < p01)
            else:
                state = int(not (rng.random() < p10))

        out[i] = float(state)

    return out

def _fit_semimarkov_12f2(train_x: np.ndarray) -> dict:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {"valid": False, "reason": "empty_train"}

    x = np.where(x >= 0.5, 1.0, 0.0)
    runs = []
    cur = int(x[0])
    run = 1

    for v in x[1:]:
        v = int(v)
        if v == cur:
            run += 1
        else:
            runs.append((cur, run))
            cur = v
            run = 1
    runs.append((cur, run))

    states = np.asarray([r[0] for r in runs], dtype=np.int8)
    lengths = np.asarray([r[1] for r in runs], dtype=np.int64)

    len_by_state = {}
    max_run = int(CFG.get("cell12f2_semimarkov_max_run_len", 250000))
    for st in [0, 1]:
        pool = lengths[states == st]
        if pool.size:
            pool = np.clip(pool, 1, max_run)
            len_by_state[st] = pool.astype(np.int64)

    return {
        "valid": True,
        "reason": "",
        "rate": float(np.mean(x)),
        "runs_n": int(len(runs)),
        "unique_states": sorted([int(v) for v in np.unique(states)]),
        "len_by_state": len_by_state,
    }

def _generate_semimarkov_12f2(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    if not bool(model.get("valid", False)):
        return np.full(n, np.nan, dtype=np.float64)

    rate = float(np.clip(model.get("rate", 0.0), 0.0, 1.0))
    len_by_state = model.get("len_by_state", {})
    unique_states = model.get("unique_states", [])

    if not len_by_state:
        return np.full(n, 1.0 if rate >= 0.5 else 0.0, dtype=np.float64)

    if len(unique_states) == 1:
        return np.full(n, float(unique_states[0]), dtype=np.float64)

    state = int(rng.random() < rate)
    out = np.empty(n, dtype=np.float64)
    pos = 0

    while pos < n:
        pool = len_by_state.get(state, None)
        if pool is None or len(pool) == 0:
            pool = len_by_state.get(1 - state, np.asarray([1], dtype=np.int64))

        run_len = int(rng.choice(pool))
        run_len = max(1, run_len)
        take = min(run_len, n - pos)
        out[pos:pos + take] = float(state)
        pos += take
        state = 1 - state

    return out

def _fit_logistic_autoregressive_12f2(col: str, train_full: np.ndarray, train_obs: np.ndarray) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return {"valid": False, "reason": f"sklearn_unavailable:{e}"}

    y_full = np.asarray(train_full, dtype=np.float64)
    train_obs = np.asarray(train_obs, dtype=bool)

    y = y_full.copy()
    y[~np.isfinite(y)] = 0.0
    y = np.where(y >= 0.5, 1.0, 0.0)

    idx = np.flatnonzero(train_obs)
    if idx.size < int(CFG.get("cell12f2_min_train_observed_n", 1)):
        return {"valid": False, "reason": "insufficient_train_observed"}

    # Need previous observation, so use all rows with observed current.
    prev = np.concatenate([[y[0]], y[:-1]])

    regime = TRAIN_REGIME_12F2.astype(np.float64)
    regime_norm = regime / max(float(np.nanmax(regime)), 1.0)

    ev = TRAIN_EVENT_COUNT_12F2.astype(np.float64)
    ev = np.nan_to_num(ev, nan=0.0)
    ev_log = np.log1p(np.maximum(ev, 0.0))

    X = np.column_stack([
        prev,
        regime_norm,
        TRAIN_TOD_SIN_12F2,
        TRAIN_TOD_COS_12F2,
        ev_log,
    ]).astype(np.float64)

    X = X[idx]
    yy = y[idx].astype(np.int8)

    pos_n = int(np.sum(yy == 1))
    neg_n = int(np.sum(yy == 0))

    if pos_n < int(CFG.get("cell12f2_logistic_min_positive", 10)) or neg_n < int(CFG.get("cell12f2_logistic_min_negative", 10)):
        return {
            "valid": False,
            "reason": f"insufficient_classes:pos={pos_n},neg={neg_n}",
            "pos_n": pos_n,
            "neg_n": neg_n,
        }

    cap = int(CFG.get("cell12f2_logistic_max_train_rows", 200000))
    rng = _rng_12f2(col, offset=909)

    if X.shape[0] > cap:
        take = rng.choice(X.shape[0], size=cap, replace=False)
        X_fit = X[take]
        y_fit = yy[take]
    else:
        X_fit = X
        y_fit = yy

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=500,
        class_weight="balanced",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_fit_s, y_fit)

    return {
        "valid": True,
        "reason": "",
        "model": clf,
        "scaler": scaler,
        "train_rows_used": int(X_fit.shape[0]),
        "pos_n": pos_n,
        "neg_n": neg_n,
        "feature_names": ["prev_obs", "regime_norm", "tod_sin", "tod_cos", "event_count_log1p"],
    }

def _generate_logistic_autoregressive_12f2(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    if not bool(model.get("valid", False)):
        return np.full(n, np.nan, dtype=np.float64)

    clf = model["model"]
    scaler = model["scaler"]

    regime = VAL_REGIME_12F2.astype(np.float64)
    regime_norm = regime / max(float(np.nanmax(regime)), 1.0)

    ev = VAL_EVENT_COUNT_12F2.astype(np.float64)
    ev = np.nan_to_num(ev, nan=0.0)
    ev_log = np.log1p(np.maximum(ev, 0.0))

    out = np.zeros(n, dtype=np.float64)
    prev = 0.0

    # Autoregressive simulation.
    for i in range(n):
        X = np.asarray([[prev, regime_norm[i], VAL_TOD_SIN_12F2[i], VAL_TOD_COS_12F2[i], ev_log[i]]], dtype=np.float64)
        Xs = scaler.transform(X)
        p = float(clf.predict_proba(Xs)[0, 1])
        state = float(rng.random() < p)
        out[i] = state
        prev = state

    return out


def _norm_targeted_family_12f2(x):
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    if "rare" in s or "zero" in s:
        return "RareEventZeroInflatedMask"
    if "plateau" in s and "constant" in s:
        return "ValRateAnchoredPlateauConstantMask"
    if "segment" in s and "plateau" in s:
        return "ValRateAnchoredSegmentPlateauMask"
    if "two" in s and "state" in s:
        return "PlateauPreservingTwoStateMask"
    return str(x)


def _generate_rare_event_zero_inflated_12f2(val_real, n, rng):
    val_bin = np.where(np.asarray(val_real, dtype=np.float64) >= 0.5, 1, 0).astype(np.int8)
    k = int(val_bin.sum())
    n = int(n)
    out = np.zeros(n, dtype=np.float64)
    if k <= 0:
        return out, True, "val_zero_events_all_zero"
    if k >= n:
        return np.ones(n, dtype=np.float64), True, "val_all_events_all_one"

    if k == 1:
        positions = np.asarray([n // 2], dtype=int)
    else:
        positions = np.linspace(1, max(1, n - 2), k).round().astype(int)
        positions = np.clip(positions, 0, n - 1)
        positions = np.unique(positions)
        if len(positions) < k:
            available = np.setdiff1d(np.arange(n), positions)
            extra = rng.choice(available, size=k - len(positions), replace=False)
            positions = np.sort(np.r_[positions, extra])

    out[positions[:k]] = 1.0
    return out, True, f"val_count_evenly_spaced_events:k={k}"


def _generate_val_rate_anchored_plateau_constant_12f2(val_real, n):
    val_bin = np.where(np.asarray(val_real, dtype=np.float64) >= 0.5, 1, 0).astype(np.int8)
    p = float(val_bin.mean()) if len(val_bin) else np.nan
    if not np.isfinite(p):
        return None, False, "invalid_val_rate"
    if p >= 0.98:
        return np.ones(int(n), dtype=np.float64), True, "val_rate_ge_0_98_constant_one"
    if p <= 0.02:
        return np.zeros(int(n), dtype=np.float64), True, "val_rate_le_0_02_constant_zero"
    return None, False, f"val_rate_not_plateau:p={p:.6f}"


def _generate_val_rate_anchored_segment_plateau_12f2(val_real, n):
    val_bin = np.where(np.asarray(val_real, dtype=np.float64) >= 0.5, 1, 0).astype(np.int8)
    n = int(n)
    p = float(val_bin.mean()) if len(val_bin) else 0.0
    k = int(round(p * n))
    out = np.zeros(n, dtype=np.float64)
    if k <= 0:
        return out, True, "val_rate_zero_segment_all_zero"
    if k >= n:
        return np.ones(n, dtype=np.float64), True, "val_rate_one_segment_all_one"
    start = max(0, (n - k) // 2)
    out[start:start + k] = 1.0
    return out, True, f"central_one_plateau:k={k},p={p:.6f}"


def _generate_plateau_preserving_two_state_12f2(val_real, n):
    val_bin = np.where(np.asarray(val_real, dtype=np.float64) >= 0.5, 1, 0).astype(np.int8)
    n = int(n)
    p = float(val_bin.mean()) if len(val_bin) else 0.0
    ones = int(round(p * n))
    zeros = n - ones

    if ones <= 0:
        return np.zeros(n, dtype=np.float64), True, "all_zero"
    if zeros <= 0:
        return np.ones(n, dtype=np.float64), True, "all_one"

    if ones >= zeros:
        out = np.ones(n, dtype=np.float64)
        out[-zeros:] = 0.0
    else:
        out = np.zeros(n, dtype=np.float64)
        out[-ones:] = 1.0

    return out, True, f"two_state_plateau:ones={ones},zeros={zeros}"


def _generate_targeted_observability_candidate_12f2(col, family, val_real, n, rng):
    """
    v1.2 safety policy:
    The previous targeted families used VAL mask values to construct candidates
    (for example, VAL-rate anchoring). That makes the candidate itself VAL-
    designed rather than TRAIN-fitted. The rows are retained for schema/audit
    continuity but are diagnostic-invalid and non-selectable.
    """
    expected_family = CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP.get(str(col), "")
    if expected_family != family:
        return None, False, "not_targeted_for_this_family", {
            "targeted_blocker_family": False,
            "targeted_family": family,
            "targeted_reason": "not_targeted_for_this_family",
            "VAL_values_used_for_targeted_candidate_design": False,
            "uses_val_rate_anchor": False,
            "uses_val_positions": False,
            "TEST_real_values_used": False,
        }

    fit_summary = {
        "targeted_blocker_family": True,
        "targeted_family": family,
        "targeted_reason": "disabled_in_v1_2_val_anchored_candidate_design_not_canonical",
        "VAL_values_used_for_targeted_candidate_design": False,
        "uses_val_rate_anchor": False,
        "targeted_val_anchored_families_disabled": True,
        "uses_val_positions": False,
        "TEST_real_values_used": False,
    }
    return None, False, "disabled_val_anchored_candidate_design_not_train_fitted", fit_summary

# ----------------------------------------------------------
# 7) Generate candidates
# ----------------------------------------------------------
candidate_val_data = {}
metrics_rows = []
inventory_rows = []
audit_rows = []

min_train_obs = int(CFG.get("cell12f2_min_train_observed_n", 1))
min_val_obs = int(CFG.get("cell12f2_min_val_observed_n", 1))

log(
    "[Cell12.f.2] Generating observability mask VAL candidates | "
    f"targets={len(IOT_OBSERVABILITY_COLS)} | N_VAL={N_VAL} | families={MASK_CANDIDATE_FAMILIES_12F2}"
)

def _add_candidate_12f2(
    col,
    family,
    candidate_id,
    syn_full,
    real_full,
    val_obs,
    train_x,
    candidate_valid,
    valid_for_selection,
    invalid_reason,
    fit_summary,
    is_a1,
    extra_fields=None,
):
    if extra_fields is None:
        extra_fields = {}

    real_eval = real_full[val_obs]
    syn_eval = syn_full[val_obs] if syn_full is not None else np.asarray([], dtype=np.float64)

    if candidate_valid and syn_full is not None:
        metrics = _mask_metrics_12f2(
            real=real_eval,
            syn=syn_eval,
            real_full=real_full,
            syn_full=syn_full,
            obs=val_obs,
            rng=_rng_12f2(candidate_id, offset=404),
        )
    else:
        metrics = {
            "eval_ok": False,
            "eval_reason": invalid_reason,
            "real_n": int(real_eval.size),
            "syn_n": 0,
            "obs_rate_error": np.nan,
            "p11_error": np.nan,
            "p00_error": np.nan,
            "run_length_ks": np.nan,
            "regime_obs_rate_error": np.nan,
            "mask_c2st_auc": np.nan,
            "score": np.inf,
        }

    train_rate = float(np.mean(train_x)) if train_x.size else np.nan
    val_rate = float(np.mean(real_eval)) if real_eval.size else np.nan

    row = {
        "col": col,
        "entity": _entity_from_obs_col_12f2(col),
        "obs_kind": _obs_kind_for_col_12f2(col),
        "mask_role": _mask_role_for_col_12f2(col),
        "candidate_id": candidate_id,
        "candidate_family": family,
        "candidate_source": "A1" if is_a1 else "12f2_non_a1",
        "is_a1": bool(is_a1),
        "candidate_valid": bool(candidate_valid),
        "valid_for_selection": bool(valid_for_selection),
        "invalid_reason": str(invalid_reason),
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(real_eval.size),
        "train_obs_rate": train_rate,
        "val_obs_rate": val_rate,
        "abs_train_val_obs_rate_delta": (
            abs(train_rate - val_rate)
            if np.isfinite(train_rate) and np.isfinite(val_rate)
            else np.nan
        ),
        "recommended_12f2_candidate": _recommended_for_col_12f2(col),
        "fit_summary": json.dumps(_json_sanitize_12f2(fit_summary), sort_keys=True),
        "TEST_real_values_used": False,
        "test_length_materialization_done_here": False,
        **metrics,
    }
    row.update(extra_fields)

    metrics_rows.append(row)

    inventory_row = {
        "col": col,
        "candidate_id": candidate_id,
        "candidate_family": family,
        "is_a1": bool(is_a1),
        "candidate_valid": bool(candidate_valid),
        "valid_for_selection": bool(valid_for_selection),
        "invalid_reason": str(invalid_reason),
        "saved_in_candidate_matrix": bool(candidate_valid and syn_full is not None),
        "TEST_real_values_used": False,
    }
    inventory_row.update(extra_fields)
    inventory_rows.append(inventory_row)


def _add_targeted_candidate_row_12f2(col, fam, val_real, val_full, val_obs, train_x):
    cid = f"{col}__{fam}"
    expected_family = CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP.get(str(col), "")
    family_match = bool(expected_family == fam)
    fit_summary = {}
    reason = ""

    try:
        rng = _rng_12f2(cid, offset=505)
        syn_raw, ok, reason, fit_summary = _generate_targeted_observability_candidate_12f2(
            col=col,
            family=fam,
            val_real=val_real,
            n=N_VAL,
            rng=rng,
        )

        if ok and syn_raw is not None:
            syn = _apply_val_mask_12f2(syn_raw, val_obs)
            valid = bool(np.isfinite(syn[val_obs]).all())
            if not valid:
                reason = "nonfinite_after_val_mask"
        else:
            syn = None
            valid = False

        if valid:
            candidate_val_data[cid] = syn.astype(np.float32)

        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=syn if valid else None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=valid,
            valid_for_selection=valid,
            invalid_reason="" if valid else reason,
            fit_summary=fit_summary,
            is_a1=False,
            extra_fields={
                "targeted_blocker_family": bool(col in CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP),
                "targeted_candidate_expected_family": expected_family,
                "targeted_candidate_family_match": family_match,
                "VAL_values_used_for_targeted_candidate_design": bool(fit_summary.get("VAL_values_used_for_targeted_candidate_design", False)),
                "uses_val_rate_anchor": bool(fit_summary.get("uses_val_rate_anchor", False)),
                "uses_val_positions": bool(fit_summary.get("uses_val_positions", False)),
                "TEST_real_values_used": False,
            },
        )
    except Exception as e:
        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=False,
            valid_for_selection=False,
            invalid_reason=str(e),
            fit_summary={
                "valid": False,
                "exception": str(e),
                "targeted_blocker_family": bool(col in CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP),
                "targeted_candidate_expected_family": expected_family,
                "targeted_candidate_family_match": family_match,
                "TEST_real_values_used": False,
            },
            is_a1=False,
            extra_fields={
                "targeted_blocker_family": bool(col in CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP),
                "targeted_candidate_expected_family": expected_family,
                "targeted_candidate_family_match": family_match,
                "VAL_values_used_for_targeted_candidate_design": False,
                "uses_val_rate_anchor": False,
                "uses_val_positions": False,
                "TEST_real_values_used": False,
            },
        )


for j, col in enumerate(IOT_OBSERVABILITY_COLS, start=1):
    if j == 1 or j % 25 == 0 or j == len(IOT_OBSERVABILITY_COLS):
        log(f"[Cell12.f.2] progress {j}/{len(IOT_OBSERVABILITY_COLS)} | col={col}")

    tr_full, tr_obs = _mask_observed_values_12f2(df_tr, col)
    val_full, val_obs = _mask_observed_values_12f2(df_val, col)

    train_x = tr_full[tr_obs]
    val_real = val_full[val_obs]

    if val_real.size < min_val_obs:
        raise RuntimeError(
            f"[Cell12.f.2] Insufficient VAL observed mask values for {col}: val_observed_n={val_real.size}"
        )

    # Candidate 1: A1 reference.
    a1_syn = pd.to_numeric(A1_OBSERVABILITY_MASKS_VAL[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    a1_syn = _coerce_mask_nan_12f2(a1_syn)
    a1_syn = _apply_val_mask_12f2(a1_syn, val_obs)

    a1_cid = f"{col}__A1_train_fitted_mask_baseline_reference"
    candidate_val_data[a1_cid] = a1_syn.astype(np.float32)

    _add_candidate_12f2(
        col=col,
        family="A1_train_fitted_mask_baseline_reference",
        candidate_id=a1_cid,
        syn_full=a1_syn,
        real_full=val_full,
        val_obs=val_obs,
        train_x=train_x,
        candidate_valid=True,
        valid_for_selection=True,
        invalid_reason="",
        fit_summary={
            "source": "CELL12F1_A1_OBSERVABILITY_MASKS_VAL",
            "baseline": "A1_train_fitted_mask_baseline",
        },
        is_a1=True,
    )

    # Non-A1 candidates require TRAIN support.
    if train_x.size < min_train_obs:
        for fam in NON_A1_MASK_CANDIDATE_FAMILIES_12F2:
            cid = f"{col}__{fam}"
            if fam in TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2:
                _add_targeted_candidate_row_12f2(
                    col=col,
                    fam=fam,
                    val_real=val_real,
                    val_full=val_full,
                    val_obs=val_obs,
                    train_x=train_x,
                )
                continue

            _add_candidate_12f2(
                col=col,
                family=fam,
                candidate_id=cid,
                syn_full=None,
                real_full=val_full,
                val_obs=val_obs,
                train_x=train_x,
                candidate_valid=False,
                valid_for_selection=False,
                invalid_reason="empty_train_no_non_a1_fit_allowed",
                fit_summary={"valid": False, "reason": "empty_train"},
                is_a1=False,
            )

        sub_inv = [r for r in inventory_rows if r["col"] == col]
        targeted_inv = [
            r for r in sub_inv
            if str(r["candidate_family"]) in set(TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2)
        ]
        targeted_valid = [r for r in targeted_inv if r["candidate_valid"]]
        targeted_valid_sel = [r for r in targeted_inv if r["valid_for_selection"]]
        targeted_best = sorted(
            [
                r for r in metrics_rows
                if r["col"] == col
                and str(r["candidate_family"]) in set(TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2)
                and bool(r.get("valid_for_selection", False))
            ],
            key=lambda r: _safe_float_12f2(r.get("score"), np.inf),
        )
        audit_rows.append({
            "col": col,
            "entity": _entity_from_obs_col_12f2(col),
            "obs_kind": _obs_kind_for_col_12f2(col),
            "train_observed_n": int(train_x.size),
            "val_observed_n": int(val_real.size),
            "non_a1_candidates_generated_n": 0,
            "non_a1_candidates_invalid_n": int(len(NON_A1_MASK_CANDIDATE_FAMILIES_12F2)),
            "targeted_candidate_expected_family": CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP.get(col, ""),
            "targeted_candidate_generated_n": int(len(targeted_valid)),
            "targeted_candidate_valid_n": int(len(targeted_valid)),
            "targeted_candidate_valid_for_selection_n": int(len(targeted_valid_sel)),
            "targeted_candidate_best_family": str(targeted_best[0]["candidate_family"]) if targeted_best else "",
            "targeted_candidate_best_score": _safe_float_12f2(targeted_best[0].get("score"), np.nan) if targeted_best else np.nan,
            "audit_reason": "empty_train_non_a1_candidates_invalid_by_contract",
            "TEST_real_values_used": False,
        })
        continue

    # Candidate 2: ConstantRateMaskFallback.
    fam = "ConstantRateMaskFallback"
    cid = f"{col}__{fam}"
    try:
        if not _eligible_constant_12f2(col):
            raise RuntimeError("not_eligible_constant_rate_fallback")

        model = _fit_constant_rate_12f2(train_x)
        syn = _generate_constant_rate_12f2(
            model=model,
            n=N_VAL,
            rng=_rng_12f2(cid, offset=101),
        )
        syn = _apply_val_mask_12f2(syn, val_obs)

        valid = bool(model.get("valid", False)) and np.isfinite(syn[val_obs]).all()
        reason = "" if valid else str(model.get("reason", "invalid_constant_rate"))

        if valid:
            candidate_val_data[cid] = syn.astype(np.float32)

        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=syn if valid else None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=valid,
            valid_for_selection=valid,
            invalid_reason=reason,
            fit_summary={
                "valid": bool(model.get("valid", False)),
                "rate": model.get("rate", np.nan),
                "constant": model.get("constant", False),
            },
            is_a1=False,
        )
    except Exception as e:
        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=False,
            valid_for_selection=False,
            invalid_reason=str(e),
            fit_summary={"valid": False, "exception": str(e)},
            is_a1=False,
        )

    # Candidate 3: RegimeConditionedMarkovMask.
    fam = "RegimeConditionedMarkovMask"
    cid = f"{col}__{fam}"
    try:
        if not _eligible_markov_12f2(col):
            raise RuntimeError("not_eligible_regime_conditioned_markov_mask")

        model = _fit_regime_markov_12f2(
            train_full=tr_full,
            train_obs=tr_obs,
            train_regime=TRAIN_REGIME_12F2,
        )
        syn = _generate_regime_markov_12f2(
            model=model,
            n=N_VAL,
            val_regime=VAL_REGIME_12F2,
            rng=_rng_12f2(cid, offset=202),
        )
        syn = _apply_val_mask_12f2(syn, val_obs)

        valid = bool(model.get("valid", False)) and np.isfinite(syn[val_obs]).all()
        reason = "" if valid else str(model.get("reason", "invalid_regime_markov"))

        if valid:
            candidate_val_data[cid] = syn.astype(np.float32)

        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=syn if valid else None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=valid,
            valid_for_selection=valid,
            invalid_reason=reason,
            fit_summary={
                "valid": bool(model.get("valid", False)),
                "global_rate": model.get("global_model", {}).get("rate", np.nan),
                "regime_models_n": int(len(model.get("regime_models", {}))),
            },
            is_a1=False,
        )
    except Exception as e:
        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=False,
            valid_for_selection=False,
            invalid_reason=str(e),
            fit_summary={"valid": False, "exception": str(e)},
            is_a1=False,
        )

    # Candidate 4: SemiMarkovMask.
    fam = "SemiMarkovMask"
    cid = f"{col}__{fam}"
    try:
        if not _eligible_semimarkov_12f2(col):
            raise RuntimeError("not_eligible_semimarkov_mask")

        model = _fit_semimarkov_12f2(train_x)
        syn = _generate_semimarkov_12f2(
            model=model,
            n=N_VAL,
            rng=_rng_12f2(cid, offset=303),
        )
        syn = _apply_val_mask_12f2(syn, val_obs)

        valid = bool(model.get("valid", False)) and np.isfinite(syn[val_obs]).all()
        reason = "" if valid else str(model.get("reason", "invalid_semimarkov"))

        if valid:
            candidate_val_data[cid] = syn.astype(np.float32)

        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=syn if valid else None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=valid,
            valid_for_selection=valid,
            invalid_reason=reason,
            fit_summary={
                "valid": bool(model.get("valid", False)),
                "rate": model.get("rate", np.nan),
                "runs_n": model.get("runs_n", 0),
                "unique_states": model.get("unique_states", []),
            },
            is_a1=False,
        )
    except Exception as e:
        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=False,
            valid_for_selection=False,
            invalid_reason=str(e),
            fit_summary={"valid": False, "exception": str(e)},
            is_a1=False,
        )

    # Candidate 5: LogisticAutoregressiveMaskModel.
    fam = "LogisticAutoregressiveMaskModel"
    cid = f"{col}__{fam}"
    try:
        if not _eligible_logistic_12f2(col):
            raise RuntimeError("not_eligible_logistic_autoregressive_mask_model")

        model = _fit_logistic_autoregressive_12f2(
            col=col,
            train_full=tr_full,
            train_obs=tr_obs,
        )
        syn = _generate_logistic_autoregressive_12f2(
            model=model,
            n=N_VAL,
            rng=_rng_12f2(cid, offset=404),
        )
        syn = _apply_val_mask_12f2(syn, val_obs)

        valid = bool(model.get("valid", False)) and np.isfinite(syn[val_obs]).all()
        reason = "" if valid else str(model.get("reason", "invalid_logistic_autoregressive"))

        if valid:
            candidate_val_data[cid] = syn.astype(np.float32)

        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=syn if valid else None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=valid,
            valid_for_selection=valid,
            invalid_reason=reason,
            fit_summary={
                "valid": bool(model.get("valid", False)),
                "train_rows_used": model.get("train_rows_used", 0),
                "pos_n": model.get("pos_n", 0),
                "neg_n": model.get("neg_n", 0),
                "features": model.get("feature_names", []),
            },
            is_a1=False,
        )
    except Exception as e:
        _add_candidate_12f2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=False,
            valid_for_selection=False,
            invalid_reason=str(e),
            fit_summary={"valid": False, "exception": str(e)},
            is_a1=False,
        )

    for fam in TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2:
        _add_targeted_candidate_row_12f2(
            col=col,
            fam=fam,
            val_real=val_real,
            val_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
        )

    sub_inv = [r for r in inventory_rows if r["col"] == col]
    targeted_inv = [
        r for r in sub_inv
        if str(r["candidate_family"]) in set(TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2)
    ]
    targeted_valid = [r for r in targeted_inv if r["candidate_valid"]]
    targeted_valid_sel = [r for r in targeted_inv if r["valid_for_selection"]]
    targeted_best = sorted(
        [
            r for r in metrics_rows
            if r["col"] == col
            and str(r["candidate_family"]) in set(TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2)
            and bool(r.get("valid_for_selection", False))
        ],
        key=lambda r: _safe_float_12f2(r.get("score"), np.inf),
    )
    audit_rows.append({
        "col": col,
        "entity": _entity_from_obs_col_12f2(col),
        "obs_kind": _obs_kind_for_col_12f2(col),
        "mask_role": _mask_role_for_col_12f2(col),
        "recommended_12f2_candidate": _recommended_for_col_12f2(col),
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(val_real.size),
        "train_obs_rate": float(np.mean(train_x)) if train_x.size else np.nan,
        "val_obs_rate": float(np.mean(val_real)) if val_real.size else np.nan,
        "eligible_constant_rate_fallback": bool(_eligible_constant_12f2(col)),
        "eligible_regime_conditioned_markov_mask": bool(_eligible_markov_12f2(col)),
        "eligible_semimarkov_mask": bool(_eligible_semimarkov_12f2(col)),
        "eligible_logistic_autoregressive_mask_model": bool(_eligible_logistic_12f2(col)),
        "candidate_rows_total": int(len(sub_inv)),
        "non_a1_candidates_generated_n": int(sum((not r["is_a1"]) and r["candidate_valid"] for r in sub_inv)),
        "non_a1_candidates_invalid_n": int(sum((not r["is_a1"]) and (not r["candidate_valid"]) for r in sub_inv)),
        "targeted_candidate_expected_family": CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP.get(col, ""),
        "targeted_candidate_generated_n": int(len(targeted_valid)),
        "targeted_candidate_valid_n": int(len(targeted_valid)),
        "targeted_candidate_valid_for_selection_n": int(len(targeted_valid_sel)),
        "targeted_candidate_best_family": str(targeted_best[0]["candidate_family"]) if targeted_best else "",
        "targeted_candidate_best_score": _safe_float_12f2(targeted_best[0].get("score"), np.nan) if targeted_best else np.nan,
        "TEST_real_values_used": False,
    })

# ----------------------------------------------------------
# 7) Assemble outputs
# ----------------------------------------------------------
IOT_OBSERVABILITY_VAL_CANDIDATES_ALL = pd.DataFrame(
    candidate_val_data,
    index=df_val.index,
    dtype=np.float32,
)

metrics_df = pd.DataFrame(metrics_rows)
inventory_df = pd.DataFrame(inventory_rows)
audit_df = pd.DataFrame(audit_rows)

# Safety invariant: no selectable candidate may be designed from VAL values.
if "VAL_values_used_for_targeted_candidate_design" in metrics_df.columns:
    _val_designed_selectable = metrics_df[
        metrics_df["VAL_values_used_for_targeted_candidate_design"].fillna(False).astype(bool)
        & metrics_df["valid_for_selection"].fillna(False).astype(bool)
    ].copy()
    if len(_val_designed_selectable):
        raise RuntimeError(
            "[Cell12.f.2] VAL-designed candidates are marked valid_for_selection. "
            f"Rows={len(_val_designed_selectable)} preview="
            f"{_val_designed_selectable[['col','candidate_family','candidate_id']].head(20).to_dict('records')}"
        )

if "uses_val_rate_anchor" in metrics_df.columns:
    _val_rate_selectable = metrics_df[
        metrics_df["uses_val_rate_anchor"].fillna(False).astype(bool)
        & metrics_df["valid_for_selection"].fillna(False).astype(bool)
    ].copy()
    if len(_val_rate_selectable):
        raise RuntimeError(
            "[Cell12.f.2] VAL-rate-anchored candidates are marked valid_for_selection. "
            f"Rows={len(_val_rate_selectable)} preview="
            f"{_val_rate_selectable[['col','candidate_family','candidate_id']].head(20).to_dict('records')}"
        )

expected_metric_rows = int(len(IOT_OBSERVABILITY_COLS) * len(MASK_CANDIDATE_FAMILIES_12F2))
if len(metrics_df) != expected_metric_rows:
    raise RuntimeError(
        "[Cell12.f.2] Metrics row count mismatch: "
        f"got={len(metrics_df)} expected={expected_metric_rows}"
    )

if len(inventory_df) != len(metrics_df):
    raise RuntimeError(
        f"[Cell12.f.2] Inventory row mismatch: inventory={len(inventory_df)} metrics={len(metrics_df)}"
    )

if len(audit_df) != len(IOT_OBSERVABILITY_COLS):
    raise RuntimeError(
        f"[Cell12.f.2] Audit row count mismatch: got={len(audit_df)} expected={len(IOT_OBSERVABILITY_COLS)}"
    )

if IOT_OBSERVABILITY_VAL_CANDIDATES_ALL.shape[0] != N_VAL:
    raise RuntimeError(
        "[Cell12.f.2] Candidate matrix row mismatch: "
        f"got={IOT_OBSERVABILITY_VAL_CANDIDATES_ALL.shape[0]} expected={N_VAL}"
    )

# Validate binary domain.
bad_candidate_cols = []
for c in IOT_OBSERVABILITY_VAL_CANDIDATES_ALL.columns:
    arr = pd.to_numeric(IOT_OBSERVABILITY_VAL_CANDIDATES_ALL[c], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    bad = int(np.sum(finite & ~((arr == 0.0) | (arr == 1.0))))
    if bad:
        bad_candidate_cols.append((c, bad))

if bad_candidate_cols:
    raise RuntimeError(
        "[Cell12.f.2] Non-binary finite values found in candidate matrix. "
        f"Preview={bad_candidate_cols[:20]}"
    )

# Coverage.
coverage_rows = []
for col in IOT_OBSERVABILITY_COLS:
    sub = metrics_df[metrics_df["col"].astype(str) == col]
    non_a1 = sub[~sub["is_a1"].astype(bool)]
    valid_non_a1 = non_a1[non_a1["valid_for_selection"].fillna(False).astype(bool)]

    coverage_rows.append({
        "col": col,
        "entity": _entity_from_obs_col_12f2(col),
        "obs_kind": _obs_kind_for_col_12f2(col),
        "mask_role": _mask_role_for_col_12f2(col),
        "candidate_rows_total": int(len(sub)),
        "a1_candidate_rows": int(sub["is_a1"].fillna(False).astype(bool).sum()),
        "non_a1_candidate_rows": int(len(non_a1)),
        "valid_non_a1_candidate_rows": int(len(valid_non_a1)),
        "has_valid_non_a1_candidate": bool(len(valid_non_a1) > 0),
        "TEST_real_values_used": False,
    })

coverage_df = pd.DataFrame(coverage_rows)

# ----------------------------------------------------------
# 8) Save outputs
# ----------------------------------------------------------
mask_candidates_val_path = os.path.join(OUT_SYN, "IOT_OBSERVABILITY_VAL_CANDIDATES_ALL.parquet")
mask_metrics_csv = os.path.join(REPORT_DIR, "cell12f2_mask_val_candidate_metrics.csv")
mask_inventory_csv = os.path.join(REPORT_DIR, "cell12f2_mask_candidate_inventory.csv")
mask_audit_csv = os.path.join(REPORT_DIR, "cell12f2_mask_candidate_audit.csv")
mask_coverage_csv = os.path.join(REPORT_DIR, "cell12f2_mask_candidate_coverage.csv")
targeted_audit_csv = os.path.join(REPORT_DIR, "cell12f2_targeted_observability_candidate_audit.csv")
targeted_recommendations_csv = os.path.join(REPORT_DIR, "cell12f2_targeted_observability_candidate_recommendations.csv")
targeted_summary_json = os.path.join(REPORT_DIR, "cell12f2_targeted_observability_candidate_summary.json")
mask_contract_json = os.path.join(REPORT_DIR, "cell12f2_mask_contract.json")
mask_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12f2_mask_contract_v1_2_THESIS.json")
mask_manifest_json = os.path.join(ARTDIR, "cell12f2_mask_val_candidate_manifest.json")

IOT_OBSERVABILITY_VAL_CANDIDATES_ALL.to_parquet(mask_candidates_val_path, index=True)
metrics_df.to_csv(mask_metrics_csv, index=False)
inventory_df.to_csv(mask_inventory_csv, index=False)
audit_df.to_csv(mask_audit_csv, index=False)
coverage_df.to_csv(mask_coverage_csv, index=False)

candidate_family_counts = (
    metrics_df["candidate_family"].astype(str).value_counts().sort_index().to_dict()
)

valid_family_counts = (
    metrics_df.loc[
        metrics_df["valid_for_selection"].fillna(False).astype(bool),
        "candidate_family",
    ]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

valid_non_a1_cols_n = int(coverage_df["has_valid_non_a1_candidate"].fillna(False).astype(bool).sum())
valid_metrics = metrics_df[metrics_df["valid_for_selection"].fillna(False).astype(bool)].copy()

targeted_metrics_df = metrics_df[
    metrics_df["candidate_family"].isin(TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2)
].copy()
targeted_expected_df = targeted_metrics_df[
    targeted_metrics_df["targeted_candidate_family_match"].fillna(False).astype(bool)
].copy()

targeted_publication_blocker_like = (
    (pd.to_numeric(targeted_expected_df["obs_rate_error"], errors="coerce") >= 0.15)
    | (pd.to_numeric(targeted_expected_df["p11_error"], errors="coerce") >= 0.30)
    | (pd.to_numeric(targeted_expected_df["p00_error"], errors="coerce") >= 0.30)
    | (pd.to_numeric(targeted_expected_df["run_length_ks"], errors="coerce") >= 0.85)
    | (pd.to_numeric(targeted_expected_df["regime_obs_rate_error"], errors="coerce") >= 0.25)
    | (pd.to_numeric(targeted_expected_df["mask_c2st_auc"], errors="coerce") >= 0.85)
)

targeted_summary = {
    "targeted_candidate_family_rows_total": int(len(targeted_metrics_df)),
    "targeted_expected_targets_n": int(len(CELL12F2_TARGETED_OBS_CANDIDATE_FAMILY_MAP)),
    "targeted_candidates_generated_n": int(targeted_expected_df["candidate_valid"].fillna(False).astype(bool).sum()),
    "targeted_candidates_valid_n": int(targeted_expected_df["candidate_valid"].fillna(False).astype(bool).sum()),
    "targeted_candidates_valid_for_selection_n": int(targeted_expected_df["valid_for_selection"].fillna(False).astype(bool).sum()),
    "targeted_candidates_publication_blocker_like_n": int(targeted_publication_blocker_like.fillna(False).astype(bool).sum()),
    "counts_by_candidate_family": targeted_metrics_df["candidate_family"].astype(str).value_counts().sort_index().to_dict(),
    "valid_counts_by_candidate_family": (
        targeted_metrics_df.loc[
            targeted_metrics_df["candidate_valid"].fillna(False).astype(bool),
            "candidate_family",
        ]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    ),
    "TEST_real_values_used": False,
    "uses_val_rate_anchor": True,
    "uses_val_positions": False,
}

targeted_recommendations_df = targeted_expected_df.copy()
targeted_recommendations_df["recommended_12f2_candidate"] = targeted_recommendations_df["candidate_family"].astype(str)
targeted_recommendations_df["recommended_family_norm"] = targeted_recommendations_df["candidate_family"].map(_norm_targeted_family_12f2)
targeted_recommendation_cols = [
    "col",
    "recommended_12f2_candidate",
    "recommended_family_norm",
    "candidate_id",
    "candidate_family",
    "candidate_valid",
    "valid_for_selection",
    "obs_rate_error",
    "p11_error",
    "p00_error",
    "run_length_ks",
    "regime_obs_rate_error",
    "mask_c2st_auc",
    "score",
    "uses_val_rate_anchor",
    "uses_val_positions",
    "TEST_real_values_used",
]
targeted_recommendations_df = targeted_recommendations_df[targeted_recommendation_cols].copy()

targeted_metrics_df.to_csv(targeted_audit_csv, index=False)
targeted_recommendations_df.to_csv(targeted_recommendations_csv, index=False)
_write_json_12f2(targeted_summary_json, targeted_summary)

metric_summary = {
    "candidate_rows_total": int(len(metrics_df)),
    "candidate_matrix_cols": int(IOT_OBSERVABILITY_VAL_CANDIDATES_ALL.shape[1]),
    "valid_for_selection_rows_total": int(len(valid_metrics)),
    "valid_non_a1_cols_n": int(valid_non_a1_cols_n),
    "mean_obs_rate_error_valid": float(pd.to_numeric(valid_metrics["obs_rate_error"], errors="coerce").mean()),
    "mean_p11_error_valid": float(pd.to_numeric(valid_metrics["p11_error"], errors="coerce").mean()),
    "mean_p00_error_valid": float(pd.to_numeric(valid_metrics["p00_error"], errors="coerce").mean()),
    "mean_run_length_ks_valid": float(pd.to_numeric(valid_metrics["run_length_ks"], errors="coerce").mean()),
    "mean_regime_obs_rate_error_valid": float(pd.to_numeric(valid_metrics["regime_obs_rate_error"], errors="coerce").mean()),
    "mean_mask_c2st_auc_valid": float(pd.to_numeric(valid_metrics["mask_c2st_auc"], errors="coerce").mean()),
}

contract = {
    "cell": "12.f.2",
    "version": CELL12F2_VERSION,
    "role": "Q3_observability_mask_VAL_candidate_generation",
    "quality_dimension": "Q3_observability",
    "observability_targets_total": int(len(IOT_OBSERVABILITY_COLS)),
    "train_rows": int(N_TR),
    "val_rows": int(N_VAL),
    "test_rows_schema_only": int(N_TE),
    "candidate_families": MASK_CANDIDATE_FAMILIES_12F2,
    "non_a1_candidate_families": NON_A1_MASK_CANDIDATE_FAMILIES_12F2,
    "targeted_observability_candidate_families": TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2,
    "targeted_observability_candidate_summary": targeted_summary,
    "targeted_recommendations_csv": targeted_recommendations_csv,
    "targeted_audit_csv": targeted_audit_csv,
    "targeted_summary_json": targeted_summary_json,
    "candidate_family_counts": candidate_family_counts,
    "valid_family_counts": valid_family_counts,
    "valid_non_a1_cols_n": int(valid_non_a1_cols_n),
    "metric_summary": metric_summary,
    "paper2_observation_model": {
        "candidate": "LogisticAutoregressiveMaskModel",
        "conditioning": [
            "previous_observation",
            "regime",
            "time_of_day",
            "event_count",
        ],
    },
    "TEST_real_values_used": False,
    "test_length_materialization_done_here": False,
    "selection_done_here": False,
    "generator_fit_done_here": True,
    "train_values_used_for_fitting": True,
    "val_values_used_for_evaluation": True,
    "VAL_values_used_for_targeted_candidate_design": False,
    "uses_val_rate_anchor": False,
    "uses_val_positions": False,
    "df_te_used_for_index_length_schema_only": True,
    "cell12f1_contract_version_seen": version1,
    "A0_selectable_seen": bool(contract1.get("a0_selectable", False)),
    "targeted_val_anchored_families_disabled": True,
    "targeted_family_policy": "Rows are retained for audit/schema continuity only; VAL-anchored targeted candidates are diagnostic-invalid and never valid_for_selection.",
    "outputs": {
        "mask_candidates_val_path": mask_candidates_val_path,
        "mask_metrics_csv": mask_metrics_csv,
        "mask_inventory_csv": mask_inventory_csv,
        "mask_audit_csv": mask_audit_csv,
        "mask_coverage_csv": mask_coverage_csv,
        "targeted_recommendations_csv": targeted_recommendations_csv,
        "targeted_audit_csv": targeted_audit_csv,
        "targeted_summary_json": targeted_summary_json,
        "mask_contract_json": mask_contract_json,
        "mask_contract_canonical_json": mask_contract_canonical_json,
        "mask_manifest_json": mask_manifest_json,
    },
}

_write_json_12f2(mask_contract_json, contract)
_write_json_12f2(mask_contract_canonical_json, contract)

manifest = {
    "cell": "12.f.2",
    "version": CELL12F2_VERSION,
    "created_outputs": contract["outputs"],
    "quality_dimension": "Q3_observability",
    "observability_targets_total": int(len(IOT_OBSERVABILITY_COLS)),
    "candidate_families": MASK_CANDIDATE_FAMILIES_12F2,
    "targeted_observability_candidate_families": TARGETED_OBSERVABILITY_CANDIDATE_FAMILIES_12F2,
    "targeted_observability_candidate_summary": targeted_summary,
    "targeted_recommendations_csv": targeted_recommendations_csv,
    "targeted_audit_csv": targeted_audit_csv,
    "targeted_summary_json": targeted_summary_json,
    "candidate_family_counts": candidate_family_counts,
    "valid_family_counts": valid_family_counts,
    "metric_summary": metric_summary,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "test_length_materialization_done_here": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": True,
        "VAL_values_used_for_targeted_candidate_design": False,
        "uses_val_rate_anchor": False,
        "targeted_val_anchored_families_disabled": True,
        "uses_val_positions": False,
    },
}

_write_json_12f2(mask_manifest_json, manifest)

hashes = {
    "mask_candidates_val_sha256": _sha256_file_12f2(mask_candidates_val_path),
    "mask_metrics_csv_sha256": _sha256_file_12f2(mask_metrics_csv),
    "mask_inventory_csv_sha256": _sha256_file_12f2(mask_inventory_csv),
    "mask_audit_csv_sha256": _sha256_file_12f2(mask_audit_csv),
    "mask_coverage_csv_sha256": _sha256_file_12f2(mask_coverage_csv),
    "targeted_recommendations_csv_sha256": _sha256_file_12f2(targeted_recommendations_csv),
    "targeted_audit_csv_sha256": _sha256_file_12f2(targeted_audit_csv),
    "targeted_summary_json_sha256": _sha256_file_12f2(targeted_summary_json),
    "mask_contract_json_sha256": _sha256_file_12f2(mask_contract_json),
    "mask_contract_canonical_json_sha256": _sha256_file_12f2(mask_contract_canonical_json),
    "mask_manifest_json_sha256": _sha256_file_12f2(mask_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12f2(mask_contract_json, contract)
_write_json_12f2(mask_contract_canonical_json, contract)
_write_json_12f2(mask_manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals for 12.f.3+
# ----------------------------------------------------------
globals()["CELL12F2_VERSION"] = CELL12F2_VERSION
globals()["IOT_OBSERVABILITY_VAL_CANDIDATES_ALL"] = IOT_OBSERVABILITY_VAL_CANDIDATES_ALL
globals()["CELL12F2_MASK_CANDIDATE_METRICS_DF"] = metrics_df
globals()["CELL12F2_MASK_CANDIDATE_INVENTORY_DF"] = inventory_df
globals()["CELL12F2_MASK_CANDIDATE_AUDIT_DF"] = audit_df
globals()["CELL12F2_MASK_CANDIDATE_COVERAGE_DF"] = coverage_df
globals()["CELL12F2_MASK_CONTRACT"] = contract
globals()["CELL12F2_MASK_VAL_CANDIDATES_PATH"] = mask_candidates_val_path
globals()["CELL12F2_MASK_METRICS_CSV"] = mask_metrics_csv
globals()["CELL12F2_MASK_INVENTORY_CSV"] = mask_inventory_csv
globals()["CELL12F2_MASK_AUDIT_CSV"] = mask_audit_csv
globals()["CELL12F2_MASK_COVERAGE_CSV"] = mask_coverage_csv
globals()["CELL12F2_TARGETED_OBSERVABILITY_CANDIDATE_AUDIT_CSV"] = targeted_audit_csv
globals()["CELL12F2_TARGETED_OBSERVABILITY_CANDIDATE_RECOMMENDATIONS_CSV"] = targeted_recommendations_csv
globals()["CELL12F2_TARGETED_OBSERVABILITY_CANDIDATE_SUMMARY_JSON"] = targeted_summary_json
globals()["CELL12F2_TARGETED_OBSERVABILITY_CANDIDATE_SUMMARY"] = targeted_summary
globals()["CELL12F2_MASK_CONTRACT_JSON"] = mask_contract_json
globals()["CELL12F2_MASK_CONTRACT_CANONICAL_JSON"] = mask_contract_canonical_json
globals()["CELL12F2_MASK_MANIFEST_JSON"] = mask_manifest_json

log(
    "[Cell12.f.2] Mask VAL candidates complete | "
    f"targets={len(IOT_OBSERVABILITY_COLS)} | "
    f"candidate_rows={len(metrics_df)} | "
    f"candidate_matrix_shape={IOT_OBSERVABILITY_VAL_CANDIDATES_ALL.shape} | "
    f"valid_non_a1_cols={valid_non_a1_cols_n}"
)
log(f"[Cell12.f.2] Candidate family counts | {candidate_family_counts}")
log(f"[Cell12.f.2] Valid family counts | {valid_family_counts}")
log(
    "[Cell12.f.2] Targeted observability candidates | "
    f"expected_targets={targeted_summary['targeted_expected_targets_n']} | "
    f"valid={targeted_summary['targeted_candidates_valid_n']} | "
    f"valid_for_selection={targeted_summary['targeted_candidates_valid_for_selection_n']} | "
    f"blocker_like={targeted_summary['targeted_candidates_publication_blocker_like_n']}"
)
log(
    "[Cell12.f.2] Metric summary | "
    f"mean_obs_rate_error_valid={metric_summary['mean_obs_rate_error_valid']:.6f} | "
    f"mean_p11_error_valid={metric_summary['mean_p11_error_valid']:.6f} | "
    f"mean_p00_error_valid={metric_summary['mean_p00_error_valid']:.6f} | "
    f"mean_run_length_ks_valid={metric_summary['mean_run_length_ks_valid']:.6f} | "
    f"mean_regime_obs_rate_error_valid={metric_summary['mean_regime_obs_rate_error_valid']:.6f} | "
    f"mean_mask_c2st_auc_valid={metric_summary['mean_mask_c2st_auc_valid']:.6f}"
)
log(f"[Cell12.f.2] Saved candidates: {mask_candidates_val_path}")
log(f"[Cell12.f.2] Saved metrics: {mask_metrics_csv} | rows={len(metrics_df)}")
log(f"[Cell12.f.2] Saved inventory: {mask_inventory_csv} | rows={len(inventory_df)}")
log(f"[Cell12.f.2] Saved audit: {mask_audit_csv} | rows={len(audit_df)}")
log(f"[Cell12.f.2] Saved coverage: {mask_coverage_csv} | rows={len(coverage_df)}")
log(f"[Cell12.f.2] Saved targeted candidate audit: {targeted_audit_csv} | rows={len(targeted_metrics_df)}")
log(f"[Cell12.f.2] Saved targeted recommendations: {targeted_recommendations_csv} | rows={len(targeted_recommendations_df)}")
log(f"[Cell12.f.2] Saved canonical contract: {mask_contract_canonical_json}")
log(
    "[Cell12.f.2] Contract flags | "
    "TEST_real_values_used=False | "
    "test_length_materialization_done_here=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=True | "
    "df_te_used_for_index_length_schema_only=True"
)
log("--- END: Cell 12.f.2 - Observability mask VAL candidates (v1.2 no VAL-anchored candidate design) ---")

gc.collect()
