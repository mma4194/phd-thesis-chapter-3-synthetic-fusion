# ==========================================================
# CELL 12.e.2 - Driver VAL candidates
# v1.1 STUDY-THESIS strict sparse binary driver VAL candidates, contract-hardened
#
# Role:
#   - Generate VAL-length candidates for 26 sparse binary event-driver columns.
#   - Fit all candidate generators on TRAIN only.
#   - Evaluate candidates on VAL only.
#   - Carry A1 driver baseline from Cell 12.e.1 as reference.
#
# Current driver scope from Cell 12.e.0:
#   - 26/26 drivers are sparse binary event indicators:
#       events_in_sec__...
#
# Candidate families:
#   1) A1_driver_event_block_bootstrap_reference
#   2) RegimeConditionedDriverMarkov
#   3) RareEventBernoulliMarkovDriver
#   4) SparseEventGapRenewalDriver
#
# Not applicable in current data:
#   - Negative Binomial / Poisson-Gamma
#   - AR(1)+Student-t
#   - continuous intensity models
#   - countlike models
#
# Strict rules:
#   - Do NOT read real TEST target values.
#   - Do NOT generate TEST-length driver values.
#   - Do NOT select final generator.
#   - Do NOT mutate A1 outputs.
#
# Outputs:
#   synthetic/IOT_DRIVER_VAL_CANDIDATES_ALL.parquet
#   reports/cell12e2_driver_val_candidate_metrics.csv
#   reports/cell12e2_driver_candidate_inventory.csv
#   reports/cell12e2_driver_candidate_audit.csv
#   reports/cell12e2_driver_candidate_coverage.csv
#   reports/cell12e2_driver_contract.json
#   artifacts/cell12e2_driver_val_candidate_manifest.json
# ==========================================================

log("--- START: Cell 12.e.2 - Driver VAL candidates (v1.1 strict sparse binary, contract-hardened) ---")

import os
import re
import gc
import json
import math
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_12e2 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_DRIVER_COLS",
    "IOT_DRIVER_TARGET_CONTRACT_DF",
    "CELL12E_DRIVER_CONTRACT",
    "A1_DRIVER_VALUES_VAL",
    "CELL12E1_A1_DRIVER_METRICS_DF",
    "CELL12E1_A1_DRIVER_AUDIT_DF",
    "CELL12E1_A1_DRIVER_CONTRACT",
]
_missing_12e2 = [k for k in _required_12e2 if k not in globals()]
if _missing_12e2:
    raise RuntimeError(f"[Cell12.e.2] Missing required globals from prior cells: {_missing_12e2}")

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

IOT_DRIVER_COLS = list(map(str, IOT_DRIVER_COLS))
EXPECTED_DRIVER_TARGET_COUNT_12E = int(CFG.get("cell12e_expected_driver_target_count", 26))

if len(IOT_DRIVER_COLS) != EXPECTED_DRIVER_TARGET_COUNT_12E:
    raise RuntimeError(
        "[Cell12.e.2] Driver target count mismatch from Cell 12.e.0: "
        f"got={len(IOT_DRIVER_COLS)} expected={EXPECTED_DRIVER_TARGET_COUNT_12E}"
    )

missing_tr = sorted([c for c in IOT_DRIVER_COLS if c not in df_tr.columns])
missing_val = sorted([c for c in IOT_DRIVER_COLS if c not in df_val.columns])
missing_te_schema = sorted([c for c in IOT_DRIVER_COLS if c not in df_te.columns])
missing_a1 = sorted([c for c in IOT_DRIVER_COLS if c not in A1_DRIVER_VALUES_VAL.columns])

if missing_tr or missing_val or missing_te_schema or missing_a1:
    raise RuntimeError(
        "[Cell12.e.2] Driver targets missing from required schemas/artifacts. "
        f"missing_train={missing_tr[:10]} | missing_val={missing_val[:10]} | "
        f"missing_test_schema={missing_te_schema[:10]} | missing_a1={missing_a1[:10]}"
    )

CELL12E2_VERSION = "cell12e2_sparse_binary_driver_val_candidates_strict_v1_1_contract_hardened"

CFG["cell12e2_version"] = CELL12E2_VERSION
CFG["cell12e2_TEST_real_values_used"] = False
CFG["cell12e2_selection_done_here"] = False
CFG["cell12e2_generator_fit_done_here"] = True
CFG["cell12e2_train_values_used_for_fitting"] = True
CFG["cell12e2_val_values_used_for_evaluation"] = True
CFG["cell12e2_test_length_materialization_done_here"] = False

CFG.setdefault("cell12e2_driver_regime_bins", 24)
CFG.setdefault("cell12e2_driver_markov_eps", 1e-6)
CFG.setdefault("cell12e2_driver_sparse_event_rate_threshold", 0.02)
CFG.setdefault("cell12e2_driver_min_train_observed_n", 1)
CFG.setdefault("cell12e2_driver_min_val_observed_n", 1)
CFG.setdefault("cell12e2_driver_copy_risk_duplicate_rate_warn", 0.98)
CFG.setdefault("cell12e2_driver_exact_event_index_overlap_warn", 0.20)
CFG.setdefault("cell12e2_driver_score_event_rate_weight", 2.0)
CFG.setdefault("cell12e2_driver_score_burst_count_weight", 2.0)
CFG.setdefault("cell12e2_driver_score_interarrival_weight", 1.0)
CFG.setdefault("cell12e2_driver_score_duration_weight", 0.75)
CFG.setdefault("cell12e2_driver_score_overdispersion_weight", 0.50)
CFG.setdefault("cell12e2_driver_score_regime_rate_weight", 1.0)

DRIVER_CANDIDATE_FAMILIES_12E2 = [
    "A1_driver_event_block_bootstrap_reference",
    "RegimeConditionedDriverMarkov",
    "RareEventBernoulliMarkovDriver",
    "SparseEventGapRenewalDriver",
]

NON_A1_DRIVER_CANDIDATE_FAMILIES_12E2 = [
    "RegimeConditionedDriverMarkov",
    "RareEventBernoulliMarkovDriver",
    "SparseEventGapRenewalDriver",
]

# ----------------------------------------------------------
# 1) Generic helpers
# ----------------------------------------------------------
def _json_sanitize_12e2(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12e2(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12e2(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12e2(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12e2(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12e2(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12e2(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12e2(obj.to_dict())
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

def _write_json_12e2(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12e2(payload), f, indent=2, sort_keys=True)

def _sha256_file_12e2(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _stable_seed_12e2(name: str, offset: int = 0) -> int:
    h = hashlib.sha256(f"{SEED}|12e2|{offset}|{name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)

def _rng_12e2(name: str, offset: int = 0):
    return np.random.default_rng(_stable_seed_12e2(name, offset=offset))

def _safe_float_12e2(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_12e2(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _coerce_binary_nan_12e2(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    arr[finite & (np.abs(arr - 0.0) <= tol)] = 0.0
    arr[finite & (np.abs(arr - 1.0) <= tol)] = 1.0
    bad = finite & ~((arr == 0.0) | (arr == 1.0))
    arr[bad] = np.nan
    return arr

def _driver_observed_values_12e2(frame: pd.DataFrame, col: str) -> tuple:
    raw = _to_num_array_12e2(frame, col)
    arr = _coerce_binary_nan_12e2(
        raw,
        tol=float(CFG.get("cell12d0_binary_value_tolerance", 1e-6)),
    )
    obs = np.isfinite(arr)
    return arr, obs

def _entity_from_driver_col_12e2(col: str) -> str:
    """Parse entity names robustly for events_in_sec driver namespaces."""
    parts = [p for p in str(col).split("__") if p != ""]
    if not parts:
        return str(col)

    if parts[0] == "events_in_sec":
        if len(parts) >= 3 and parts[1] in {"entity", "device", "iot"}:
            return parts[2]
        if len(parts) >= 3 and parts[1] == "feat":
            return parts[2]
        return parts[1] if len(parts) > 1 else str(col)

    if parts[0] == "iot":
        return parts[1] if len(parts) > 1 else str(col)

    if "__feat__" in str(col):
        tail = str(col).split("__feat__", 1)[1]
        return tail.split("__", 1)[0]

    if "__entity__" in str(col):
        tail = str(col).split("__entity__", 1)[1]
        return tail.split("__", 1)[0]

    return str(col)

def _driver_measurement_12e2(col: str) -> str:
    """Parse measurement/driver semantics without collapsing the entity into the measurement."""
    parts = [p for p in str(col).split("__") if p != ""]
    if not parts:
        return str(col)

    if parts[0] == "events_in_sec":
        if len(parts) >= 2 and parts[1] == "entity":
            return "entity_event"
        if len(parts) >= 4 and parts[1] == "feat":
            return "__".join(parts[3:])
        if len(parts) >= 3:
            return "__".join(parts[2:])
        return str(col)

    if parts[0] == "iot":
        if len(parts) >= 4:
            return "__".join(parts[2:])
        return "__".join(parts[1:])

    if "__feat__" in str(col):
        tail = str(col).split("__feat__", 1)[1]
        bits = tail.split("__")
        return "__".join(bits[1:]) if len(bits) > 1 else tail

    if "__entity__" in str(col):
        return "entity_event"

    return str(col)

def _event_indices_12e2(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    return np.flatnonzero(finite & (arr >= 0.5)).astype(np.int64)

def _run_lengths_12e2(x: np.ndarray, state=None) -> np.ndarray:
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

def _interarrival_12e2(event_idx: np.ndarray) -> np.ndarray:
    event_idx = np.asarray(event_idx, dtype=np.int64)
    if event_idx.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(event_idx).astype(np.float64)

def _empirical_ks_12e2(a: np.ndarray, b: np.ndarray) -> float:
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

def _overdispersion_12e2(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    mean = float(np.mean(arr))
    var = float(np.var(arr))
    if mean <= 0:
        return np.nan
    return float(var / max(mean, 1e-12))

def _duplicate_rate_12e2(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    vc = pd.Series(arr).value_counts(normalize=True)
    return float(vc.iloc[0]) if len(vc) else np.nan

# ----------------------------------------------------------
# 2) VAL observed mask and regime bins
# ----------------------------------------------------------
def _build_val_observed_mask_12e2() -> pd.DataFrame:
    if "IOT_DRIVER_AVAIL_VAL_DF" in globals() and isinstance(IOT_DRIVER_AVAIL_VAL_DF, pd.DataFrame):
        mask_df = IOT_DRIVER_AVAIL_VAL_DF.copy()
        mask_df.columns = mask_df.columns.astype(str)
        missing = sorted(set(IOT_DRIVER_COLS) - set(mask_df.columns))
        if missing:
            raise RuntimeError(f"[Cell12.e.2] IOT_DRIVER_AVAIL_VAL_DF missing columns: {missing[:20]}")
        return mask_df[IOT_DRIVER_COLS].astype(np.float32)

    data = {}
    for col in IOT_DRIVER_COLS:
        _, obs = _driver_observed_values_12e2(df_val, col)
        data[col] = obs.astype(np.float32)
    return pd.DataFrame(data, index=df_val.index, dtype=np.float32)

IOT_DRIVER_AVAIL_VAL_DF_12E2 = _build_val_observed_mask_12e2()

def _detect_time_col_12e2(frame: pd.DataFrame):
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

def _time_regime_bins_12e2(frame: pd.DataFrame, n_bins: int) -> np.ndarray:
    n = int(len(frame))
    n_bins = int(max(1, n_bins))

    tcol = _detect_time_col_12e2(frame)
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

REGIME_BINS_12E2 = int(CFG.get("cell12e2_driver_regime_bins", 24))
TRAIN_REGIME_12E2 = _time_regime_bins_12e2(df_tr, REGIME_BINS_12E2)
VAL_REGIME_12E2 = _time_regime_bins_12e2(df_val, REGIME_BINS_12E2)

# ----------------------------------------------------------
# 3) Metrics
# ----------------------------------------------------------
def _regime_event_rate_error_12e2(real_full: np.ndarray, syn_full: np.ndarray, obs: np.ndarray, val_regime: np.ndarray) -> float:
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

def _copy_risk_metrics_12e2(train_x: np.ndarray, syn_x: np.ndarray) -> dict:
    train = np.asarray(train_x, dtype=np.float64)
    syn = np.asarray(syn_x, dtype=np.float64)
    train = train[np.isfinite(train)]
    syn = syn[np.isfinite(syn)]

    if train.size == 0 or syn.size == 0:
        return {
            "copy_risk_eval_ok": False,
            "train_duplicate_rate": np.nan,
            "syn_duplicate_rate": np.nan,
            "exact_event_index_overlap_rate": np.nan,
            "sparse_duplicate_rate_expected": True,
            "copy_risk_high": True,
            "copy_risk_reason": "empty_train_or_synthetic",
        }

    train = np.where(train >= 0.5, 1.0, 0.0)
    syn = np.where(syn >= 0.5, 1.0, 0.0)

    train_dup = _duplicate_rate_12e2(train)
    syn_dup = _duplicate_rate_12e2(syn)

    train_events = set(_event_indices_12e2(train).tolist())
    syn_events = set(_event_indices_12e2(syn).tolist())

    if len(syn_events) == 0:
        overlap = 0.0
    else:
        # Coordinate overlap is only a diagnostic proxy. TRAIN and VAL are different
        # temporal segments, so this must not be interpreted as direct row copying.
        overlap = len(train_events.intersection(syn_events)) / max(len(syn_events), 1)

    sparse_duplicate_rate_expected = bool(
        np.isfinite(train_dup)
        and train_dup >= float(CFG.get("cell12e2_driver_copy_risk_duplicate_rate_warn", 0.98))
    )

    high = bool(
        np.isfinite(overlap)
        and overlap >= float(CFG.get("cell12e2_driver_exact_event_index_overlap_warn", 0.20))
    )

    reason = "exact_event_coordinate_overlap_high" if high else (
        "sparse_duplicate_rate_expected" if sparse_duplicate_rate_expected else "no_high_copy_risk_signal"
    )

    return {
        "copy_risk_eval_ok": True,
        "train_duplicate_rate": train_dup,
        "syn_duplicate_rate": syn_dup,
        "exact_event_index_overlap_rate": float(overlap),
        "sparse_duplicate_rate_expected": sparse_duplicate_rate_expected,
        "copy_risk_high": high,
        "copy_risk_reason": reason,
    }

def _driver_metrics_12e2(real: np.ndarray, syn: np.ndarray, real_full: np.ndarray, syn_full: np.ndarray, obs: np.ndarray) -> dict:
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
            "event_rate_error": np.nan,
            "burst_count_error": np.nan,
            "interarrival_ks": np.nan,
            "duration_ks": np.nan,
            "overdispersion_error": np.nan,
            "regime_event_rate_error": np.nan,
            "score": np.inf,
        }

    real = np.where(real >= 0.5, 1.0, 0.0)
    syn = np.where(syn >= 0.5, 1.0, 0.0)

    real_rate = float(np.mean(real))
    syn_rate = float(np.mean(syn))
    event_rate_error = abs(real_rate - syn_rate)

    real_idx = _event_indices_12e2(real)
    syn_idx = _event_indices_12e2(syn)

    real_event_count = int(real_idx.size)
    syn_event_count = int(syn_idx.size)

    burst_count_error = abs(real_event_count - syn_event_count) / max(real_n, 1)

    real_inter = _interarrival_12e2(real_idx)
    syn_inter = _interarrival_12e2(syn_idx)
    interarrival_ks = _empirical_ks_12e2(real_inter, syn_inter)

    real_dur = _run_lengths_12e2(real, state=1)
    syn_dur = _run_lengths_12e2(syn, state=1)
    duration_ks = _empirical_ks_12e2(real_dur, syn_dur)

    real_over = _overdispersion_12e2(real)
    syn_over = _overdispersion_12e2(syn)
    overdispersion_error = abs(
        _safe_float_12e2(real_over, 0.0)
        - _safe_float_12e2(syn_over, 0.0)
    )

    regime_err = _regime_event_rate_error_12e2(
        real_full=real_full,
        syn_full=syn_full,
        obs=obs,
        val_regime=VAL_REGIME_12E2,
    )

    components = {
        "event_rate_error": event_rate_error,
        "burst_count_error": burst_count_error,
        "interarrival_ks": interarrival_ks,
        "duration_ks": duration_ks,
        "overdispersion_error": min(overdispersion_error, 10.0) / 10.0,
        "regime_event_rate_error": regime_err,
    }
    for k, v in list(components.items()):
        if not np.isfinite(v):
            components[k] = 1.0

    score = (
        float(CFG.get("cell12e2_driver_score_event_rate_weight", 2.0)) * components["event_rate_error"]
        + float(CFG.get("cell12e2_driver_score_burst_count_weight", 2.0)) * components["burst_count_error"]
        + float(CFG.get("cell12e2_driver_score_interarrival_weight", 1.0)) * components["interarrival_ks"]
        + float(CFG.get("cell12e2_driver_score_duration_weight", 0.75)) * components["duration_ks"]
        + float(CFG.get("cell12e2_driver_score_overdispersion_weight", 0.50)) * components["overdispersion_error"]
        + float(CFG.get("cell12e2_driver_score_regime_rate_weight", 1.0)) * components["regime_event_rate_error"]
    )

    return {
        "eval_ok": True,
        "eval_reason": "",
        "real_n": real_n,
        "syn_n": syn_n,
        "real_event_rate": real_rate,
        "syn_event_rate": syn_rate,
        "event_rate_error": float(event_rate_error),
        "real_event_count": real_event_count,
        "syn_event_count": syn_event_count,
        "burst_count_error": float(burst_count_error),
        "real_interarrival_median": float(np.median(real_inter)) if real_inter.size else np.nan,
        "syn_interarrival_median": float(np.median(syn_inter)) if syn_inter.size else np.nan,
        "interarrival_ks": float(interarrival_ks),
        "real_duration_median": float(np.median(real_dur)) if real_dur.size else np.nan,
        "syn_duration_median": float(np.median(syn_dur)) if syn_dur.size else np.nan,
        "duration_ks": float(duration_ks),
        "real_overdispersion": real_over,
        "syn_overdispersion": syn_over,
        "overdispersion_error": float(overdispersion_error) if np.isfinite(overdispersion_error) else np.nan,
        "regime_event_rate_error": float(regime_err) if np.isfinite(regime_err) else np.nan,
        "score": float(score),
    }

# ----------------------------------------------------------
# 4) Candidate generators
# ----------------------------------------------------------
def _apply_val_mask_12e2(syn: np.ndarray, val_obs: np.ndarray) -> np.ndarray:
    out = np.asarray(syn, dtype=np.float64).copy()
    val_obs = np.asarray(val_obs, dtype=bool)

    if out.size != val_obs.size:
        raise RuntimeError(
            f"[Cell12.e.2] VAL mask length mismatch: syn={out.size} mask={val_obs.size}"
        )

    out[~val_obs] = np.nan
    out[val_obs] = np.where(out[val_obs] >= 0.5, 1.0, 0.0)
    return out

def _estimate_markov_12e2(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    eps = float(CFG.get("cell12e2_driver_markov_eps", 1e-6))

    if arr.size == 0:
        return {"valid": False, "reason": "empty_train"}

    arr = np.where(arr >= 0.5, 1.0, 0.0)
    rate = float(np.mean(arr))

    if arr.size < 2:
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

    a = arr[:-1].astype(np.int8)
    b = arr[1:].astype(np.int8)

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
        "constant": bool(len(np.unique(arr)) < 2),
        "p01": float(np.clip(p01, 0.0, 1.0)),
        "p10": float(np.clip(p10, 0.0, 1.0)),
        "init_p1": rate,
        "n_pairs": int(a.size),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }

def _generate_markov_12e2(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
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

def _fit_regime_conditioned_markov_12e2(train_full: np.ndarray, train_obs: np.ndarray, train_regime: np.ndarray) -> dict:
    train_full = np.asarray(train_full, dtype=np.float64)
    train_obs = np.asarray(train_obs, dtype=bool)
    train_regime = np.asarray(train_regime, dtype=np.int64)

    global_model = _estimate_markov_12e2(train_full[train_obs])

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
        m = _estimate_markov_12e2(train_full[idx])
        if bool(m.get("valid", False)):
            regime_models[int(r)] = m

    return {
        "valid": True,
        "reason": "",
        "global_model": global_model,
        "regime_models": regime_models,
    }

def _generate_regime_conditioned_markov_12e2(model: dict, n: int, val_regime: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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

def _fit_rare_event_bernoulli_markov_driver_12e2(train_x: np.ndarray) -> dict:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {"valid": False, "reason": "empty_train"}

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))

    event_idx = _event_indices_12e2(x)
    one_runs = _run_lengths_12e2(x, state=1)
    inter = _interarrival_12e2(event_idx)

    return {
        "valid": True,
        "reason": "",
        "rate": rate,
        "event_n": int(event_idx.size),
        "one_run_lengths": one_runs,
        "interarrival": inter,
        "markov": _estimate_markov_12e2(x),
        "constant": bool(len(np.unique(x)) < 2),
    }

def _generate_rare_event_bernoulli_markov_driver_12e2(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)

    if not bool(model.get("valid", False)):
        return np.full(n, np.nan, dtype=np.float64)

    rate = float(np.clip(model.get("rate", 0.0), 0.0, 1.0))
    if rate <= 0.0:
        return np.zeros(n, dtype=np.float64)
    if rate >= 1.0:
        return np.ones(n, dtype=np.float64)

    one_runs = np.asarray(model.get("one_run_lengths", []), dtype=np.float64)
    if one_runs.size == 0:
        expected_events = rate * n
        n_events = int(rng.poisson(expected_events))
        n_events = min(n_events, n)
        out = np.zeros(n, dtype=np.float64)
        if n_events > 0:
            starts = rng.choice(n, size=n_events, replace=False)
            out[starts] = 1.0
        return out

    median_run = max(1.0, float(np.median(one_runs)))
    expected_events = rate * n / median_run
    n_events = int(rng.poisson(max(0.0, expected_events)))
    n_events = min(n_events, n)

    out = np.zeros(n, dtype=np.float64)
    if n_events <= 0:
        return out

    starts = rng.choice(n, size=n_events, replace=True)
    starts = np.sort(starts)

    for st in starts:
        run_len = int(rng.choice(one_runs))
        run_len = max(1, run_len)
        end = min(n, int(st) + run_len)
        out[int(st):end] = 1.0

    # TRAIN-only marginal guard.
    max_rate = min(1.0, max(rate * 3.0, rate + 0.002))
    cur_rate = float(np.mean(out))
    if cur_rate > max_rate:
        one_idx = np.flatnonzero(out >= 0.5)
        keep_n = int(np.floor(max_rate * n))
        keep_n = max(0, min(keep_n, one_idx.size))
        out[:] = 0.0
        if keep_n > 0:
            keep_idx = rng.choice(one_idx, size=keep_n, replace=False)
            out[keep_idx] = 1.0

    return out

def _fit_sparse_event_gap_renewal_driver_12e2(train_x: np.ndarray) -> dict:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {"valid": False, "reason": "empty_train"}

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))
    event_idx = _event_indices_12e2(x)

    if event_idx.size == 0:
        return {
            "valid": True,
            "reason": "no_train_events",
            "rate": rate,
            "event_n": 0,
            "gaps": np.asarray([], dtype=np.float64),
            "one_runs": np.asarray([], dtype=np.float64),
        }

    gaps = _interarrival_12e2(event_idx)
    one_runs = _run_lengths_12e2(x, state=1)

    return {
        "valid": True,
        "reason": "",
        "rate": rate,
        "event_n": int(event_idx.size),
        "gaps": gaps,
        "one_runs": one_runs,
    }

def _generate_sparse_event_gap_renewal_driver_12e2(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)

    if not bool(model.get("valid", False)):
        return np.full(n, np.nan, dtype=np.float64)

    rate = float(np.clip(model.get("rate", 0.0), 0.0, 1.0))
    gaps = np.asarray(model.get("gaps", []), dtype=np.float64)
    one_runs = np.asarray(model.get("one_runs", []), dtype=np.float64)

    if rate <= 0.0 or int(model.get("event_n", 0)) == 0:
        return np.zeros(n, dtype=np.float64)

    out = np.zeros(n, dtype=np.float64)

    if gaps.size == 0:
        expected_events = max(0.0, rate * n)
        n_events = int(rng.poisson(expected_events))
        n_events = min(n_events, n)
        if n_events > 0:
            starts = rng.choice(n, size=n_events, replace=False)
            out[starts] = 1.0
        return out

    if one_runs.size == 0:
        one_runs = np.asarray([1.0], dtype=np.float64)

    start_cap = max(1, int(np.median(gaps)))
    pos = int(rng.integers(0, start_cap))
    event_points = 0

    while pos < n:
        run_len = int(rng.choice(one_runs))
        run_len = max(1, run_len)
        end = min(n, pos + run_len)
        out[pos:end] = 1.0
        event_points += int(end - pos)

        gap = int(rng.choice(gaps))
        jitter_radius = max(1, int(abs(gap) // 10))
        jitter = int(rng.integers(-jitter_radius, jitter_radius + 1))
        gap = max(1, gap + jitter)
        pos = end + gap

    # TRAIN-only marginal guard.
    max_rate = min(1.0, max(rate * 3.0, rate + 0.002))
    cur_rate = float(np.mean(out))
    if cur_rate > max_rate:
        one_idx = np.flatnonzero(out >= 0.5)
        keep_n = int(np.floor(max_rate * n))
        keep_n = max(0, min(keep_n, one_idx.size))
        out[:] = 0.0
        if keep_n > 0:
            keep_idx = rng.choice(one_idx, size=keep_n, replace=False)
            out[keep_idx] = 1.0

    return out

# ----------------------------------------------------------
# 5) Generate candidates
# ----------------------------------------------------------
candidate_val_data = {}
metrics_rows = []
inventory_rows = []
audit_rows = []

min_train_obs = int(CFG.get("cell12e2_driver_min_train_observed_n", 1))
min_val_obs = int(CFG.get("cell12e2_driver_min_val_observed_n", 1))

log(
    "[Cell12.e.2] Generating sparse binary driver VAL candidates | "
    f"cols={len(IOT_DRIVER_COLS)} | N_VAL={N_VAL} | families={DRIVER_CANDIDATE_FAMILIES_12E2}"
)

def _add_candidate_12e2(
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
):
    real_eval = real_full[val_obs]
    syn_eval = syn_full[val_obs] if syn_full is not None else np.asarray([], dtype=np.float64)

    if candidate_valid and syn_full is not None:
        metrics = _driver_metrics_12e2(
            real=real_eval,
            syn=syn_eval,
            real_full=real_full,
            syn_full=syn_full,
            obs=val_obs,
        )
        copy_metrics = _copy_risk_metrics_12e2(train_x=train_x, syn_x=syn_eval)
    else:
        metrics = {
            "eval_ok": False,
            "eval_reason": invalid_reason,
            "real_n": int(real_eval.size),
            "syn_n": 0,
            "event_rate_error": np.nan,
            "burst_count_error": np.nan,
            "interarrival_ks": np.nan,
            "duration_ks": np.nan,
            "overdispersion_error": np.nan,
            "regime_event_rate_error": np.nan,
            "score": np.inf,
        }
        copy_metrics = {
            "copy_risk_eval_ok": False,
            "train_duplicate_rate": np.nan,
            "syn_duplicate_rate": np.nan,
            "exact_event_index_overlap_rate": np.nan,
            "copy_risk_high": True,
        }

    train_event_rate = float(np.mean(train_x)) if train_x.size else np.nan
    val_event_rate = float(np.mean(real_eval)) if real_eval.size else np.nan

    row = {
        "col": col,
        "entity": _entity_from_driver_col_12e2(col),
        "measurement_name": _driver_measurement_12e2(col),
        "candidate_id": candidate_id,
        "candidate_family": family,
        "candidate_source": "A1" if is_a1 else "12e2_non_a1",
        "is_a1": bool(is_a1),
        "candidate_valid": bool(candidate_valid),
        "valid_for_selection": bool(valid_for_selection),
        "invalid_reason": str(invalid_reason),
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(real_eval.size),
        "train_event_rate": train_event_rate,
        "val_event_rate": val_event_rate,
        "abs_train_val_event_rate_delta": (
            abs(train_event_rate - val_event_rate)
            if np.isfinite(train_event_rate) and np.isfinite(val_event_rate)
            else np.nan
        ),
        "train_event_count": int(np.sum(train_x >= 0.5)) if train_x.size else 0,
        "val_event_count": int(np.sum(real_eval >= 0.5)) if real_eval.size else 0,
        "fit_summary": json.dumps(_json_sanitize_12e2(fit_summary), sort_keys=True),
        "TEST_real_values_used": False,
        "test_length_materialization_done_here": False,
        **metrics,
        **copy_metrics,
    }

    metrics_rows.append(row)

    inventory_rows.append({
        "col": col,
        "candidate_id": candidate_id,
        "candidate_family": family,
        "is_a1": bool(is_a1),
        "candidate_valid": bool(candidate_valid),
        "valid_for_selection": bool(valid_for_selection),
        "invalid_reason": str(invalid_reason),
        "saved_in_candidate_matrix": bool(candidate_valid and syn_full is not None),
        "copy_risk_high": bool(copy_metrics.get("copy_risk_high", False)),
        "TEST_real_values_used": False,
    })

for j, col in enumerate(IOT_DRIVER_COLS, start=1):
    if j == 1 or j % 5 == 0 or j == len(IOT_DRIVER_COLS):
        log(f"[Cell12.e.2] progress {j}/{len(IOT_DRIVER_COLS)} | col={col}")

    tr_full, tr_obs = _driver_observed_values_12e2(df_tr, col)
    val_full, val_obs = _driver_observed_values_12e2(df_val, col)

    train_x = tr_full[tr_obs]
    val_real = val_full[val_obs]

    if val_real.size < min_val_obs:
        raise RuntimeError(
            f"[Cell12.e.2] Insufficient VAL observed driver values for {col}: "
            f"val_observed_n={val_real.size}"
        )

    # Candidate 1: A1 reference.
    a1_syn = pd.to_numeric(A1_DRIVER_VALUES_VAL[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    a1_syn = _coerce_binary_nan_12e2(a1_syn)
    a1_syn = _apply_val_mask_12e2(a1_syn, val_obs)

    a1_cid = f"{col}__A1_driver_event_block_bootstrap_reference"
    candidate_val_data[a1_cid] = a1_syn.astype(np.float32)

    _add_candidate_12e2(
        col=col,
        family="A1_driver_event_block_bootstrap_reference",
        candidate_id=a1_cid,
        syn_full=a1_syn,
        real_full=val_full,
        val_obs=val_obs,
        train_x=train_x,
        candidate_valid=True,
        valid_for_selection=True,
        invalid_reason="",
        fit_summary={
            "source": "CELL12E1_A1_DRIVER_VALUES_VAL",
            "baseline": "A1_driver_event_block_bootstrap",
        },
        is_a1=True,
    )

    if train_x.size < min_train_obs:
        for fam in NON_A1_DRIVER_CANDIDATE_FAMILIES_12E2:
            cid = f"{col}__{fam}"
            _add_candidate_12e2(
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

        audit_rows.append({
            "col": col,
            "entity": _entity_from_driver_col_12e2(col),
            "measurement_name": _driver_measurement_12e2(col),
            "train_observed_n": int(train_x.size),
            "val_observed_n": int(val_real.size),
            "train_event_rate": np.nan,
            "val_event_rate": float(np.mean(val_real)) if val_real.size else np.nan,
            "train_event_count": 0,
            "val_event_count": int(np.sum(val_real >= 0.5)),
            "non_a1_candidates_generated_n": 0,
            "non_a1_candidates_invalid_n": int(len(NON_A1_DRIVER_CANDIDATE_FAMILIES_12E2)),
            "audit_reason": "empty_train_non_a1_candidates_invalid_by_contract",
            "TEST_real_values_used": False,
        })
        continue

    # Candidate 2: RegimeConditionedDriverMarkov.
    fam = "RegimeConditionedDriverMarkov"
    cid = f"{col}__{fam}"
    try:
        model = _fit_regime_conditioned_markov_12e2(
            train_full=tr_full,
            train_obs=tr_obs,
            train_regime=TRAIN_REGIME_12E2,
        )
        syn = _generate_regime_conditioned_markov_12e2(
            model=model,
            n=N_VAL,
            val_regime=VAL_REGIME_12E2,
            rng=_rng_12e2(cid, offset=101),
        )
        syn = _apply_val_mask_12e2(syn, val_obs)

        valid = bool(model.get("valid", False)) and np.isfinite(syn[val_obs]).all()
        reason = "" if valid else str(model.get("reason", "invalid_regime_conditioned_markov"))

        if valid:
            candidate_val_data[cid] = syn.astype(np.float32)

        _add_candidate_12e2(
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
        _add_candidate_12e2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=False,
            valid_for_selection=False,
            invalid_reason=f"exception:{type(e).__name__}:{e}",
            fit_summary={"valid": False, "exception": str(e)},
            is_a1=False,
        )

    # Candidate 3: RareEventBernoulliMarkovDriver.
    fam = "RareEventBernoulliMarkovDriver"
    cid = f"{col}__{fam}"
    try:
        model = _fit_rare_event_bernoulli_markov_driver_12e2(train_x)
        syn = _generate_rare_event_bernoulli_markov_driver_12e2(
            model=model,
            n=N_VAL,
            rng=_rng_12e2(cid, offset=202),
        )
        syn = _apply_val_mask_12e2(syn, val_obs)

        valid = bool(model.get("valid", False)) and np.isfinite(syn[val_obs]).all()
        reason = "" if valid else str(model.get("reason", "invalid_rare_event_driver"))

        if valid:
            candidate_val_data[cid] = syn.astype(np.float32)

        _add_candidate_12e2(
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
                "event_n": model.get("event_n", 0),
            },
            is_a1=False,
        )
    except Exception as e:
        _add_candidate_12e2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=False,
            valid_for_selection=False,
            invalid_reason=f"exception:{type(e).__name__}:{e}",
            fit_summary={"valid": False, "exception": str(e)},
            is_a1=False,
        )

    # Candidate 4: SparseEventGapRenewalDriver.
    fam = "SparseEventGapRenewalDriver"
    cid = f"{col}__{fam}"
    try:
        model = _fit_sparse_event_gap_renewal_driver_12e2(train_x)
        syn = _generate_sparse_event_gap_renewal_driver_12e2(
            model=model,
            n=N_VAL,
            rng=_rng_12e2(cid, offset=303),
        )
        syn = _apply_val_mask_12e2(syn, val_obs)

        valid = bool(model.get("valid", False)) and np.isfinite(syn[val_obs]).all()
        reason = "" if valid else str(model.get("reason", "invalid_sparse_event_gap_renewal"))

        if valid:
            candidate_val_data[cid] = syn.astype(np.float32)

        _add_candidate_12e2(
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
                "event_n": model.get("event_n", 0),
                "gap_n": int(len(model.get("gaps", []))) if isinstance(model.get("gaps", []), np.ndarray) else 0,
            },
            is_a1=False,
        )
    except Exception as e:
        _add_candidate_12e2(
            col=col,
            family=fam,
            candidate_id=cid,
            syn_full=None,
            real_full=val_full,
            val_obs=val_obs,
            train_x=train_x,
            candidate_valid=False,
            valid_for_selection=False,
            invalid_reason=f"exception:{type(e).__name__}:{e}",
            fit_summary={"valid": False, "exception": str(e)},
            is_a1=False,
        )

    # Audit.
    train_event_rate = float(np.mean(train_x)) if train_x.size else np.nan
    val_event_rate = float(np.mean(val_real)) if val_real.size else np.nan

    audit_rows.append({
        "col": col,
        "entity": _entity_from_driver_col_12e2(col),
        "measurement_name": _driver_measurement_12e2(col),
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(val_real.size),
        "train_event_rate": train_event_rate,
        "val_event_rate": val_event_rate,
        "abs_train_val_event_rate_delta": (
            abs(train_event_rate - val_event_rate)
            if np.isfinite(train_event_rate) and np.isfinite(val_event_rate)
            else np.nan
        ),
        "train_event_count": int(np.sum(train_x >= 0.5)),
        "val_event_count": int(np.sum(val_real >= 0.5)),
        "train_duplicate_rate": _duplicate_rate_12e2(train_x),
        "train_copy_risk_high": bool(
            _duplicate_rate_12e2(train_x) >= float(CFG.get("cell12e2_driver_copy_risk_duplicate_rate_warn", 0.98))
        ),
        "non_a1_candidates_generated_n": int(
            sum(
                1
                for fam_name in NON_A1_DRIVER_CANDIDATE_FAMILIES_12E2
                if f"{col}__{fam_name}" in candidate_val_data
            )
        ),
        "non_a1_candidates_invalid_n": int(
            sum(
                1
                for r in inventory_rows
                if r["col"] == col
                and not bool(r["is_a1"])
                and not bool(r["candidate_valid"])
            )
        ),
        "audit_reason": "",
        "TEST_real_values_used": False,
    })

# ----------------------------------------------------------
# 6) Assemble outputs and validate
# ----------------------------------------------------------
IOT_DRIVER_VAL_CANDIDATES_ALL = pd.DataFrame(
    candidate_val_data,
    index=df_val.index,
    dtype=np.float32,
)

metrics_df = pd.DataFrame(metrics_rows)
inventory_df = pd.DataFrame(inventory_rows)
audit_df = pd.DataFrame(audit_rows)

expected_metric_rows = int(len(IOT_DRIVER_COLS) * len(DRIVER_CANDIDATE_FAMILIES_12E2))
if len(metrics_df) != expected_metric_rows:
    raise RuntimeError(
        "[Cell12.e.2] Metrics row count mismatch: "
        f"got={len(metrics_df)} expected={expected_metric_rows}"
    )

if len(inventory_df) != len(metrics_df):
    raise RuntimeError(
        f"[Cell12.e.2] Inventory row mismatch: inventory={len(inventory_df)} metrics={len(metrics_df)}"
    )

if len(audit_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.2] Audit row count mismatch: got={len(audit_df)} expected={len(IOT_DRIVER_COLS)}"
    )

if IOT_DRIVER_VAL_CANDIDATES_ALL.shape[0] != N_VAL:
    raise RuntimeError(
        "[Cell12.e.2] Candidate matrix row mismatch: "
        f"got={IOT_DRIVER_VAL_CANDIDATES_ALL.shape[0]} expected={N_VAL}"
    )

# Validate candidate matrix binary domain.
bad_candidate_cols = []
for c in IOT_DRIVER_VAL_CANDIDATES_ALL.columns:
    arr = pd.to_numeric(IOT_DRIVER_VAL_CANDIDATES_ALL[c], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    bad = int(np.sum(finite & ~((arr == 0.0) | (arr == 1.0))))
    if bad:
        bad_candidate_cols.append((c, bad))

if bad_candidate_cols:
    raise RuntimeError(
        "[Cell12.e.2] Non-binary finite values found in candidate matrix. "
        f"Preview={bad_candidate_cols[:20]}"
    )

# Coverage.
coverage_rows = []
for col in IOT_DRIVER_COLS:
    sub = metrics_df[metrics_df["col"].astype(str) == col]
    non_a1 = sub[~sub["is_a1"].astype(bool)]
    valid_non_a1 = non_a1[non_a1["valid_for_selection"].fillna(False).astype(bool)]

    coverage_rows.append({
        "col": col,
        "entity": _entity_from_driver_col_12e2(col),
        "measurement_name": _driver_measurement_12e2(col),
        "candidate_rows_total": int(len(sub)),
        "a1_candidate_rows": int(sub["is_a1"].fillna(False).astype(bool).sum()),
        "non_a1_candidate_rows": int(len(non_a1)),
        "valid_non_a1_candidate_rows": int(len(valid_non_a1)),
        "has_valid_non_a1_candidate": bool(len(valid_non_a1) > 0),
        "TEST_real_values_used": False,
    })

coverage_df = pd.DataFrame(coverage_rows)

# ----------------------------------------------------------
# 7) Save outputs
# ----------------------------------------------------------
driver_candidates_val_path = os.path.join(OUT_SYN, "IOT_DRIVER_VAL_CANDIDATES_ALL.parquet")
driver_metrics_csv = os.path.join(REPORT_DIR, "cell12e2_driver_val_candidate_metrics.csv")
driver_inventory_csv = os.path.join(REPORT_DIR, "cell12e2_driver_candidate_inventory.csv")
driver_audit_csv = os.path.join(REPORT_DIR, "cell12e2_driver_candidate_audit.csv")
driver_coverage_csv = os.path.join(REPORT_DIR, "cell12e2_driver_candidate_coverage.csv")
driver_contract_json = os.path.join(REPORT_DIR, "cell12e2_driver_contract.json")
driver_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12e2_driver_contract_v1_1_THESIS.json")
driver_manifest_json = os.path.join(ARTDIR, "cell12e2_driver_val_candidate_manifest.json")

IOT_DRIVER_VAL_CANDIDATES_ALL.to_parquet(driver_candidates_val_path, index=True)
metrics_df.to_csv(driver_metrics_csv, index=False)
inventory_df.to_csv(driver_inventory_csv, index=False)
audit_df.to_csv(driver_audit_csv, index=False)
coverage_df.to_csv(driver_coverage_csv, index=False)

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

metric_summary = {
    "candidate_rows_total": int(len(metrics_df)),
    "candidate_matrix_cols": int(IOT_DRIVER_VAL_CANDIDATES_ALL.shape[1]),
    "valid_for_selection_rows_total": int(len(valid_metrics)),
    "valid_non_a1_cols_n": int(valid_non_a1_cols_n),
    "mean_event_rate_error_valid": float(pd.to_numeric(valid_metrics["event_rate_error"], errors="coerce").mean()),
    "mean_burst_count_error_valid": float(pd.to_numeric(valid_metrics["burst_count_error"], errors="coerce").mean()),
    "mean_interarrival_ks_valid": float(pd.to_numeric(valid_metrics["interarrival_ks"], errors="coerce").mean()),
    "mean_duration_ks_valid": float(pd.to_numeric(valid_metrics["duration_ks"], errors="coerce").mean()),
    "mean_overdispersion_error_valid": float(pd.to_numeric(valid_metrics["overdispersion_error"], errors="coerce").mean()),
    "mean_regime_event_rate_error_valid": float(pd.to_numeric(valid_metrics["regime_event_rate_error"], errors="coerce").mean()),
    "copy_risk_high_valid_n": int(valid_metrics["copy_risk_high"].fillna(False).astype(bool).sum()),
}

contract = {
    "cell": "12.e.2",
    "version": CELL12E2_VERSION,
    "role": "sparse_binary_driver_VAL_candidate_generation",
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "expected_driver_targets_total": int(EXPECTED_DRIVER_TARGET_COUNT_12E),
    "driver_type_scope": "sparse_binary_event_drivers",
    "train_rows": int(N_TR),
    "val_rows": int(N_VAL),
    "test_rows_schema_only": int(N_TE),
    "candidate_families": DRIVER_CANDIDATE_FAMILIES_12E2,
    "non_a1_candidate_families": NON_A1_DRIVER_CANDIDATE_FAMILIES_12E2,
    "not_applicable_current_scope": [
        "Negative Binomial / Poisson-Gamma",
        "AR(1)+Student-t",
        "continuous intensity models",
        "countlike driver models",
    ],
    "candidate_family_counts": candidate_family_counts,
    "valid_family_counts": valid_family_counts,
    "valid_non_a1_cols_n": int(valid_non_a1_cols_n),
    "metric_summary": metric_summary,
    "TEST_real_values_used": False,
    "test_length_materialization_done_here": False,
    "selection_done_here": False,
    "generator_fit_done_here": True,
    "train_values_used_for_fitting": True,
    "val_values_used_for_evaluation": True,
    "df_te_used_for_index_length_schema_only": True,
    "copy_risk_interpretation": "For sparse binary drivers, high duplicate rate is expected from dominant zeros; copy_risk_high is based on event-coordinate overlap diagnostics, not duplicate zeros alone.",
    "outputs": {
        "driver_candidates_val_path": driver_candidates_val_path,
        "driver_metrics_csv": driver_metrics_csv,
        "driver_inventory_csv": driver_inventory_csv,
        "driver_audit_csv": driver_audit_csv,
        "driver_coverage_csv": driver_coverage_csv,
        "driver_contract_json": driver_contract_json,
        "driver_contract_canonical_json": driver_contract_canonical_json,
        "driver_manifest_json": driver_manifest_json,
    },
}

_write_json_12e2(driver_contract_json, contract)
_write_json_12e2(driver_contract_canonical_json, contract)

manifest = {
    "cell": "12.e.2",
    "version": CELL12E2_VERSION,
    "created_outputs": contract["outputs"],
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "candidate_families": DRIVER_CANDIDATE_FAMILIES_12E2,
    "candidate_family_counts": candidate_family_counts,
    "valid_family_counts": valid_family_counts,
    "metric_summary": metric_summary,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "test_length_materialization_done_here": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": True,
    },
}

_write_json_12e2(driver_manifest_json, manifest)

hashes = {
    "driver_candidates_val_sha256": _sha256_file_12e2(driver_candidates_val_path),
    "driver_metrics_csv_sha256": _sha256_file_12e2(driver_metrics_csv),
    "driver_inventory_csv_sha256": _sha256_file_12e2(driver_inventory_csv),
    "driver_audit_csv_sha256": _sha256_file_12e2(driver_audit_csv),
    "driver_coverage_csv_sha256": _sha256_file_12e2(driver_coverage_csv),
    "driver_contract_json_sha256": _sha256_file_12e2(driver_contract_json),
    "driver_contract_canonical_json_sha256": _sha256_file_12e2(driver_contract_canonical_json),
    "driver_manifest_json_sha256": _sha256_file_12e2(driver_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12e2(driver_contract_json, contract)
_write_json_12e2(driver_contract_canonical_json, contract)
_write_json_12e2(driver_manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals for 12.e.3+
# ----------------------------------------------------------
globals()["CELL12E2_VERSION"] = CELL12E2_VERSION
globals()["IOT_DRIVER_VAL_CANDIDATES_ALL"] = IOT_DRIVER_VAL_CANDIDATES_ALL
globals()["CELL12E2_DRIVER_CANDIDATE_METRICS_DF"] = metrics_df
globals()["CELL12E2_DRIVER_CANDIDATE_INVENTORY_DF"] = inventory_df
globals()["CELL12E2_DRIVER_CANDIDATE_AUDIT_DF"] = audit_df
globals()["CELL12E2_DRIVER_CANDIDATE_COVERAGE_DF"] = coverage_df
globals()["CELL12E2_DRIVER_CONTRACT"] = contract
globals()["CELL12E2_DRIVER_VAL_CANDIDATES_PATH"] = driver_candidates_val_path
globals()["CELL12E2_DRIVER_METRICS_CSV"] = driver_metrics_csv
globals()["CELL12E2_DRIVER_INVENTORY_CSV"] = driver_inventory_csv
globals()["CELL12E2_DRIVER_AUDIT_CSV"] = driver_audit_csv
globals()["CELL12E2_DRIVER_COVERAGE_CSV"] = driver_coverage_csv
globals()["CELL12E2_DRIVER_CONTRACT_JSON"] = driver_contract_json
globals()["CELL12E2_DRIVER_CONTRACT_CANONICAL_JSON"] = driver_contract_canonical_json
globals()["CELL12E2_DRIVER_MANIFEST_JSON"] = driver_manifest_json

log(
    "[Cell12.e.2] Driver VAL candidates complete | "
    f"targets={len(IOT_DRIVER_COLS)} | "
    f"candidate_rows={len(metrics_df)} | "
    f"candidate_matrix_shape={IOT_DRIVER_VAL_CANDIDATES_ALL.shape} | "
    f"valid_non_a1_cols={valid_non_a1_cols_n}"
)
log(f"[Cell12.e.2] Candidate family counts | {candidate_family_counts}")
log(f"[Cell12.e.2] Valid family counts | {valid_family_counts}")
log(
    "[Cell12.e.2] Metric summary | "
    f"mean_event_rate_error_valid={metric_summary['mean_event_rate_error_valid']:.8f} | "
    f"mean_burst_count_error_valid={metric_summary['mean_burst_count_error_valid']:.8f} | "
    f"mean_interarrival_ks_valid={metric_summary['mean_interarrival_ks_valid']:.6f} | "
    f"mean_duration_ks_valid={metric_summary['mean_duration_ks_valid']:.6f} | "
    f"mean_overdispersion_error_valid={metric_summary['mean_overdispersion_error_valid']:.6f} | "
    f"mean_regime_event_rate_error_valid={metric_summary['mean_regime_event_rate_error_valid']:.8f}"
)
log(f"[Cell12.e.2] Saved candidates: {driver_candidates_val_path}")
log(f"[Cell12.e.2] Saved metrics: {driver_metrics_csv} | rows={len(metrics_df)}")
log(f"[Cell12.e.2] Saved inventory: {driver_inventory_csv} | rows={len(inventory_df)}")
log(f"[Cell12.e.2] Saved audit: {driver_audit_csv} | rows={len(audit_df)}")
log(f"[Cell12.e.2] Saved coverage: {driver_coverage_csv} | rows={len(coverage_df)}")
log(f"[Cell12.e.2] Saved canonical contract: {driver_contract_canonical_json}")
log(
    "[Cell12.e.2] Contract flags | "
    "TEST_real_values_used=False | "
    "test_length_materialization_done_here=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=True | "
    "df_te_used_for_index_length_schema_only=True"
)
log("--- END: Cell 12.e.2 - Driver VAL candidates (v1.1 strict sparse binary, contract-hardened) ---")

gc.collect()