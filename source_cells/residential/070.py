# ==========================================================
# CELL 12.e.1 - Driver baseline
# v1.1 STUDY-THESIS strict sparse binary driver A1 baseline, contract-hardened
#
# Role:
#   - Build TRAIN-fitted, VAL-length A1 baseline for 26 driver columns.
#   - Baseline family:
#       A1_driver_event_block_bootstrap
#   - Drivers are currently sparse binary event indicators:
#       events_in_sec__...
#   - Evaluate against real VAL driver values only.
#   - Run copy-risk diagnostics because all drivers are highly sparse.
#
# Strict rules:
#   - Do NOT read real TEST values.
#   - Do NOT generate TEST-length driver values.
#   - Do NOT select final production generator.
#   - Do NOT fit non-A1 generators.
#
# Outputs:
#   synthetic/A1_DRIVER_VALUES_VAL.parquet
#   reports/cell12e1_a1_driver_baseline_metrics.csv
#   reports/cell12e1_a1_driver_baseline_audit.csv
#   reports/cell12e1_a1_driver_contract.json
#   artifacts/cell12e1_a1_driver_manifest.json
# ==========================================================

log("--- START: Cell 12.e.1 - Driver baseline (v1.1 strict sparse binary A1, contract-hardened) ---")

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
_required_12e1 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_DRIVER_COLS",
    "IOT_DRIVER_TARGET_CONTRACT_DF",
    "CELL12E_DRIVER_CONTRACT",
]
_missing_12e1 = [k for k in _required_12e1 if k not in globals()]
if _missing_12e1:
    raise RuntimeError(f"[Cell12.e.1] Missing required globals from prior cells: {_missing_12e1}")

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
        "[Cell12.e.1] Driver target count mismatch from Cell 12.e.0: "
        f"got={len(IOT_DRIVER_COLS)} expected={EXPECTED_DRIVER_TARGET_COUNT_12E}"
    )

missing_tr = sorted([c for c in IOT_DRIVER_COLS if c not in df_tr.columns])
missing_val = sorted([c for c in IOT_DRIVER_COLS if c not in df_val.columns])
missing_te_schema = sorted([c for c in IOT_DRIVER_COLS if c not in df_te.columns])

if missing_tr or missing_val or missing_te_schema:
    raise RuntimeError(
        "[Cell12.e.1] Driver targets missing from split schemas. "
        f"missing_train={missing_tr[:10]} | missing_val={missing_val[:10]} | "
        f"missing_test_schema={missing_te_schema[:10]}"
    )

CELL12E1_VERSION = "cell12e1_a1_sparse_binary_driver_baseline_strict_v1_1_contract_hardened"

CFG["cell12e1_version"] = CELL12E1_VERSION
CFG["cell12e1_TEST_real_values_used"] = False
CFG["cell12e1_selection_done_here"] = False
CFG["cell12e1_generator_fit_done_here"] = False
CFG["cell12e1_A1_baseline_done_here"] = True
CFG["cell12e1_train_values_used_for_fitting"] = True
CFG["cell12e1_val_values_used_for_evaluation"] = True
CFG["cell12e1_test_length_materialization_done_here"] = False

CFG.setdefault("cell12e1_driver_block_len", 4096)
CFG.setdefault("cell12e1_driver_event_window_radius", 4)
CFG.setdefault("cell12e1_driver_sparse_event_rate_threshold", 0.02)
CFG.setdefault("cell12e1_driver_copy_risk_duplicate_rate_warn", 0.98)
CFG.setdefault("cell12e1_driver_exact_event_index_overlap_warn", 0.20)
CFG.setdefault("cell12e1_driver_min_val_observed_n", 1)

A1_DRIVER_BASELINE_FAMILY_12E1 = "A1_driver_event_block_bootstrap"

# ----------------------------------------------------------
# 1) Generic helpers
# ----------------------------------------------------------
def _json_sanitize_12e1(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12e1(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12e1(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12e1(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12e1(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12e1(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12e1(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12e1(obj.to_dict())
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

def _write_json_12e1(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12e1(payload), f, indent=2, sort_keys=True)

def _sha256_file_12e1(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _stable_seed_12e1(name: str, offset: int = 0) -> int:
    h = hashlib.sha256(f"{SEED}|12e1|{offset}|{name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)

def _rng_12e1(name: str, offset: int = 0):
    return np.random.default_rng(_stable_seed_12e1(name, offset=offset))

def _safe_float_12e1(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_12e1(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _coerce_binary_nan_12e1(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    arr[finite & (np.abs(arr - 0.0) <= tol)] = 0.0
    arr[finite & (np.abs(arr - 1.0) <= tol)] = 1.0
    bad = finite & ~((arr == 0.0) | (arr == 1.0))
    arr[bad] = np.nan
    return arr

def _driver_observed_values_12e1(frame: pd.DataFrame, col: str) -> tuple:
    raw = _to_num_array_12e1(frame, col)
    arr = _coerce_binary_nan_12e1(
        raw,
        tol=float(CFG.get("cell12d0_binary_value_tolerance", 1e-6)),
    )
    obs = np.isfinite(arr)
    return arr, obs

def _finite_driver_values_12e1(frame: pd.DataFrame, col: str) -> np.ndarray:
    arr, obs = _driver_observed_values_12e1(frame, col)
    return arr[obs].astype(np.float64, copy=False)

def _entity_from_driver_col_12e1(col: str) -> str:
    """Parse driver entity robustly for events_in_sec__entity__<entity>__... and iot__<entity>__... columns."""
    s = str(col)
    parts = [p for p in s.split("__") if p != ""]
    if not parts:
        return s
    if parts[0] == "events_in_sec":
        if len(parts) > 2 and parts[1] in {"entity", "device", "iot"}:
            return parts[2]
        return parts[1] if len(parts) > 1 else s
    if parts[0] == "iot":
        return parts[1] if len(parts) > 1 else s
    if "__feat__" in s:
        tail = s.split("__feat__", 1)[1]
        return tail.split("__", 1)[0]
    return parts[0]

def _driver_measurement_12e1(col: str) -> str:
    """Parse measurement tail robustly for events_in_sec and iot driver namespaces."""
    s = str(col)
    parts = [p.strip().lower() for p in s.split("__") if p.strip()]
    if not parts:
        return s
    if parts[0] == "events_in_sec":
        rest = parts[1:]
        if rest and rest[0] in {"entity", "device", "iot"}:
            rest = rest[1:]
        # Drop entity token if present.
        if rest:
            rest = rest[1:]
        return "__".join(rest) if rest else s
    if parts[0] == "iot":
        rest = parts[2:] if len(parts) > 2 else parts[1:]
        return "__".join(rest) if rest else s
    if "__feat__" in s:
        return s.split("__feat__", 1)[1]
    if "__entity__" in s:
        return s.split("__entity__", 1)[1]
    return s

def _event_indices_12e1(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    return np.flatnonzero(finite & (arr >= 0.5)).astype(np.int64)

def _run_lengths_12e1(x: np.ndarray, state=None) -> np.ndarray:
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

def _interarrival_12e1(event_idx: np.ndarray) -> np.ndarray:
    event_idx = np.asarray(event_idx, dtype=np.int64)
    if event_idx.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(event_idx).astype(np.float64)

def _empirical_ks_12e1(a: np.ndarray, b: np.ndarray) -> float:
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

def _overdispersion_12e1(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    mean = float(np.mean(arr))
    var = float(np.var(arr))
    if mean <= 0:
        return np.nan
    return float(var / max(mean, 1e-12))

def _duplicate_rate_12e1(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    vc = pd.Series(arr).value_counts(normalize=True)
    return float(vc.iloc[0]) if len(vc) else np.nan

# ----------------------------------------------------------
# 2) VAL observed mask
# ----------------------------------------------------------
def _build_val_observed_mask_12e1() -> pd.DataFrame:
    data = {}
    for col in IOT_DRIVER_COLS:
        _, obs = _driver_observed_values_12e1(df_val, col)
        data[col] = obs.astype(np.float32)
    return pd.DataFrame(data, index=df_val.index, dtype=np.float32)

IOT_DRIVER_AVAIL_VAL_DF = _build_val_observed_mask_12e1()

# ----------------------------------------------------------
# 3) Sparse-aware A1 driver event block bootstrap
# ----------------------------------------------------------
def _generate_a1_driver_event_block_bootstrap_12e1(
    train_x: np.ndarray,
    n: int,
    rng: np.random.Generator,
    block_len: int,
) -> tuple:
    """
    TRAIN-only sparse binary driver baseline.

    It is deliberately copy-risk-aware:
    - For very sparse drivers, it samples TRAIN event interarrival gaps and
      short event runs rather than copying long raw blocks of zeros.
    - If events are too few, it uses a Bernoulli event-rate baseline.
    - It never uses VAL or TEST values for generation.
    """
    n = int(n)
    block_len = int(max(1, block_len))

    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.zeros(n, dtype=np.float64), {
            "method": "empty_train_zero_baseline",
            "train_event_n": 0,
            "copy_risk_mode": True,
        }

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))
    event_idx = _event_indices_12e1(x)
    event_n = int(event_idx.size)

    sparse_threshold = float(CFG.get("cell12e1_driver_sparse_event_rate_threshold", 0.02))

    # Constant cases.
    if rate <= 0.0 or event_n == 0:
        return np.zeros(n, dtype=np.float64), {
            "method": "no_train_events_zero_baseline",
            "train_event_n": event_n,
            "train_event_rate": rate,
            "copy_risk_mode": True,
        }

    if rate >= 1.0:
        return np.ones(n, dtype=np.float64), {
            "method": "all_train_events_one_baseline",
            "train_event_n": event_n,
            "train_event_rate": rate,
            "copy_risk_mode": False,
        }

    # Sparse event-gap generator.
    if rate <= sparse_threshold:
        one_runs = _run_lengths_12e1(x, state=1)
        gaps = _interarrival_12e1(event_idx)

        out = np.zeros(n, dtype=np.float64)

        if gaps.size == 0:
            expected_events = max(0.0, rate * n)
            n_events = int(rng.poisson(expected_events))
            n_events = min(n_events, n)
            if n_events > 0:
                starts = rng.choice(n, size=n_events, replace=False)
                out[starts] = 1.0
            return out, {
                "method": "sparse_bernoulli_poisson_event_rate",
                "train_event_n": event_n,
                "train_event_rate": rate,
                "expected_events": float(expected_events),
                "generated_event_n": int(np.sum(out >= 0.5)),
                "copy_risk_mode": True,
            }

        # Gap replay with jitter avoids exact block copy.
        pos = int(rng.integers(0, max(1, int(np.median(gaps)))))
        generated_events = 0

        while pos < n:
            run_len = int(rng.choice(one_runs)) if one_runs.size else 1
            run_len = max(1, run_len)

            end = min(n, pos + run_len)
            out[pos:end] = 1.0
            generated_events += int(end - pos)

            gap = int(rng.choice(gaps))
            jitter = int(rng.integers(-max(1, gap // 10), max(2, gap // 10 + 1)))
            gap = max(1, gap + jitter)
            pos = end + gap

        # Marginal guard: no TEST/VAL involved. Keep generated rate near TRAIN.
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

        return out, {
            "method": "sparse_event_gap_run_bootstrap_with_jitter",
            "train_event_n": event_n,
            "train_event_rate": rate,
            "train_gap_n": int(gaps.size),
            "train_one_run_n": int(one_runs.size),
            "generated_event_n": int(np.sum(out >= 0.5)),
            "generated_event_rate": float(np.mean(out)),
            "copy_risk_mode": True,
        }

    # Less sparse fallback: block bootstrap.
    out = np.empty(n, dtype=np.float64)
    pos = 0
    max_start = max(0, x.size - 1)

    while pos < n:
        start = int(rng.integers(0, max_start + 1))
        end = min(x.size, start + block_len)
        block = x[start:end]
        if block.size == 0:
            block = x
        take = min(block.size, n - pos)
        out[pos:pos + take] = block[:take]
        pos += take

    return out, {
        "method": "raw_event_block_bootstrap",
        "train_event_n": event_n,
        "train_event_rate": rate,
        "block_len": block_len,
        "copy_risk_mode": False,
    }

def _apply_val_mask_12e1(syn: np.ndarray, val_obs: np.ndarray) -> np.ndarray:
    out = np.asarray(syn, dtype=np.float64).copy()
    val_obs = np.asarray(val_obs, dtype=bool)

    if out.size != val_obs.size:
        raise RuntimeError(
            f"[Cell12.e.1] VAL mask length mismatch: syn={out.size} mask={val_obs.size}"
        )

    out[~val_obs] = np.nan
    out[val_obs] = np.where(out[val_obs] >= 0.5, 1.0, 0.0)
    return out

# ----------------------------------------------------------
# 4) Metrics
# ----------------------------------------------------------
def _driver_metrics_12e1(real: np.ndarray, syn: np.ndarray) -> dict:
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
            "score": np.inf,
        }

    real = np.where(real >= 0.5, 1.0, 0.0)
    syn = np.where(syn >= 0.5, 1.0, 0.0)

    real_rate = float(np.mean(real))
    syn_rate = float(np.mean(syn))
    event_rate_error = abs(real_rate - syn_rate)

    real_idx = _event_indices_12e1(real)
    syn_idx = _event_indices_12e1(syn)

    real_event_count = int(real_idx.size)
    syn_event_count = int(syn_idx.size)

    burst_count_error = abs(real_event_count - syn_event_count) / max(real_n, 1)

    real_inter = _interarrival_12e1(real_idx)
    syn_inter = _interarrival_12e1(syn_idx)
    interarrival_ks = _empirical_ks_12e1(real_inter, syn_inter)

    real_dur = _run_lengths_12e1(real, state=1)
    syn_dur = _run_lengths_12e1(syn, state=1)
    duration_ks = _empirical_ks_12e1(real_dur, syn_dur)

    real_over = _overdispersion_12e1(real)
    syn_over = _overdispersion_12e1(syn)
    overdispersion_error = abs(
        _safe_float_12e1(real_over, 0.0)
        - _safe_float_12e1(syn_over, 0.0)
    )

    # Sparse driver score: prioritize event-rate and burst-count, then timing.
    components = {
        "event_rate_error": event_rate_error,
        "burst_count_error": burst_count_error,
        "interarrival_ks": interarrival_ks,
        "duration_ks": duration_ks,
        "overdispersion_error": min(overdispersion_error, 10.0) / 10.0,
    }
    for k, v in list(components.items()):
        if not np.isfinite(v):
            components[k] = 1.0

    score = (
        2.0 * components["event_rate_error"]
        + 2.0 * components["burst_count_error"]
        + 0.75 * components["interarrival_ks"]
        + 0.75 * components["duration_ks"]
        + 0.50 * components["overdispersion_error"]
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
        "score": float(score),
    }

def _copy_risk_metrics_12e1(train_x: np.ndarray, syn_x: np.ndarray) -> dict:
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
            "index_coordinate_overlap_rate": np.nan,
            "copy_risk_high": True,
            "copy_risk_reason": "empty_train_or_synthetic",
        }

    train = np.where(train >= 0.5, 1.0, 0.0)
    syn = np.where(syn >= 0.5, 1.0, 0.0)

    train_dup = _duplicate_rate_12e1(train)
    syn_dup = _duplicate_rate_12e1(syn)

    train_events = set(_event_indices_12e1(train).tolist())
    syn_events = set(_event_indices_12e1(syn).tolist())

    if len(syn_events) == 0:
        overlap = 0.0
    else:
        overlap = len(train_events.intersection(syn_events)) / max(len(syn_events), 1)

    high = bool(
        (
            np.isfinite(train_dup)
            and train_dup >= float(CFG.get("cell12e1_driver_copy_risk_duplicate_rate_warn", 0.98))
        )
        or (
            np.isfinite(overlap)
            and overlap >= float(CFG.get("cell12e1_driver_exact_event_index_overlap_warn", 0.20))
        )
    )

    # This is an index-coordinate overlap diagnostic only. TRAIN and VAL have
    # different temporal positions, so overlap is not interpreted as direct row-copy
    # evidence by itself. High duplicate rate is expected for sparse binary drivers.
    reasons = []
    if np.isfinite(train_dup) and train_dup >= float(CFG.get("cell12e1_driver_copy_risk_duplicate_rate_warn", 0.98)):
        reasons.append("sparse_duplicate_rate_high_expected_for_event_drivers")
    if np.isfinite(overlap) and overlap >= float(CFG.get("cell12e1_driver_exact_event_index_overlap_warn", 0.20)):
        reasons.append("index_coordinate_overlap_high_diagnostic_only")

    return {
        "copy_risk_eval_ok": True,
        "train_duplicate_rate": train_dup,
        "syn_duplicate_rate": syn_dup,
        "exact_event_index_overlap_rate": float(overlap),
        "index_coordinate_overlap_rate": float(overlap),
        "copy_risk_high": high,
        "copy_risk_reason": "|".join(reasons) if reasons else "none",
    }

# ----------------------------------------------------------
# 5) Generate A1 baseline
# ----------------------------------------------------------
metrics_rows = []
audit_rows = []
selected_val_data = {}

block_len = int(CFG.get("cell12e1_driver_block_len", 4096))
min_val_obs = int(CFG.get("cell12e1_driver_min_val_observed_n", 1))

log(
    "[Cell12.e.1] Building A1 driver VAL baseline | "
    f"cols={len(IOT_DRIVER_COLS)} | N_VAL={N_VAL} | baseline={A1_DRIVER_BASELINE_FAMILY_12E1}"
)

for j, col in enumerate(IOT_DRIVER_COLS, start=1):
    if j == 1 or j % 5 == 0 or j == len(IOT_DRIVER_COLS):
        log(f"[Cell12.e.1] progress {j}/{len(IOT_DRIVER_COLS)} | col={col}")

    tr_full, tr_obs = _driver_observed_values_12e1(df_tr, col)
    val_full, val_obs = _driver_observed_values_12e1(df_val, col)

    train_x = tr_full[tr_obs]
    val_real = val_full[val_obs]

    if val_real.size < min_val_obs:
        raise RuntimeError(
            f"[Cell12.e.1] Insufficient VAL observed driver values for {col}: "
            f"val_observed_n={val_real.size}"
        )

    rng = _rng_12e1(col, offset=101)
    syn_raw, fit_summary = _generate_a1_driver_event_block_bootstrap_12e1(
        train_x=train_x,
        n=N_VAL,
        rng=rng,
        block_len=block_len,
    )

    syn_full = _apply_val_mask_12e1(syn_raw, val_obs)
    syn_eval = syn_full[val_obs]

    m = _driver_metrics_12e1(real=val_real, syn=syn_eval)
    copy_m = _copy_risk_metrics_12e1(train_x=train_x, syn_x=syn_eval)

    selected_val_data[col] = syn_full.astype(np.float32)

    train_event_rate = float(np.mean(train_x)) if train_x.size else np.nan
    val_event_rate = float(np.mean(val_real)) if val_real.size else np.nan

    row = {
        "col": col,
        "entity": _entity_from_driver_col_12e1(col),
        "measurement_name": _driver_measurement_12e1(col),
        "candidate_id": f"{col}__{A1_DRIVER_BASELINE_FAMILY_12E1}",
        "candidate_family": A1_DRIVER_BASELINE_FAMILY_12E1,
        "candidate_source": "A1",
        "is_a1": True,
        "candidate_valid": True,
        "valid_for_selection": True,
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(val_real.size),
        "train_event_rate": train_event_rate,
        "val_event_rate": val_event_rate,
        "abs_train_val_event_rate_delta": (
            abs(train_event_rate - val_event_rate)
            if np.isfinite(train_event_rate) and np.isfinite(val_event_rate)
            else np.nan
        ),
        "train_event_count": int(np.sum(train_x >= 0.5)) if train_x.size else 0,
        "val_event_count": int(np.sum(val_real >= 0.5)) if val_real.size else 0,
        "fit_summary": json.dumps(_json_sanitize_12e1(fit_summary), sort_keys=True),
        "TEST_real_values_used": False,
        "test_length_materialization_done_here": False,
        **m,
        **copy_m,
    }
    metrics_rows.append(row)

    audit_rows.append({
        "col": col,
        "entity": _entity_from_driver_col_12e1(col),
        "measurement_name": _driver_measurement_12e1(col),
        "baseline_family": A1_DRIVER_BASELINE_FAMILY_12E1,
        "fit_method": fit_summary.get("method", ""),
        "copy_risk_mode": bool(fit_summary.get("copy_risk_mode", False)),
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(val_real.size),
        "train_event_rate": train_event_rate,
        "val_event_rate": val_event_rate,
        "syn_event_rate": m.get("syn_event_rate", np.nan),
        "event_rate_error": m.get("event_rate_error", np.nan),
        "burst_count_error": m.get("burst_count_error", np.nan),
        "interarrival_ks": m.get("interarrival_ks", np.nan),
        "duration_ks": m.get("duration_ks", np.nan),
        "overdispersion_error": m.get("overdispersion_error", np.nan),
        "copy_risk_high": bool(copy_m.get("copy_risk_high", False)),
        "exact_event_index_overlap_rate": copy_m.get("exact_event_index_overlap_rate", np.nan),
        "diagnostic_only_if_copy_risk_high": bool(copy_m.get("copy_risk_high", False)),
        "TEST_real_values_used": False,
    })

# ----------------------------------------------------------
# 6) Assemble and validate
# ----------------------------------------------------------
A1_DRIVER_VALUES_VAL = pd.DataFrame(
    selected_val_data,
    index=df_val.index,
    columns=IOT_DRIVER_COLS,
    dtype=np.float32,
)

if A1_DRIVER_VALUES_VAL.shape != (N_VAL, len(IOT_DRIVER_COLS)):
    raise RuntimeError(
        "[Cell12.e.1] A1_DRIVER_VALUES_VAL shape mismatch: "
        f"got={A1_DRIVER_VALUES_VAL.shape}, expected={(N_VAL, len(IOT_DRIVER_COLS))}"
    )

# Hard binary/mask validation.
for col in IOT_DRIVER_COLS:
    mask = IOT_DRIVER_AVAIL_VAL_DF[col].to_numpy(dtype=np.float32) > 0.5
    arr = pd.to_numeric(A1_DRIVER_VALUES_VAL[col], errors="coerce").to_numpy(dtype=np.float64)

    inactive_finite = int(np.isfinite(arr[~mask]).sum())
    if inactive_finite != 0:
        raise RuntimeError(
            f"[Cell12.e.1] A1 VAL inactive finite violation for {col}: {inactive_finite}"
        )

    active = arr[mask]
    active_nonfinite = int(np.sum(~np.isfinite(active)))
    bad_binary = int(np.sum(np.isfinite(active) & ~((active == 0.0) | (active == 1.0))))

    if active_nonfinite != 0 or bad_binary != 0:
        raise RuntimeError(
            f"[Cell12.e.1] A1 VAL binary violation for {col}: "
            f"active_nonfinite={active_nonfinite}, bad_binary={bad_binary}"
        )

metrics_df = pd.DataFrame(metrics_rows)
audit_df = pd.DataFrame(audit_rows)

if len(metrics_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.1] Metrics row count mismatch: got={len(metrics_df)} expected={len(IOT_DRIVER_COLS)}"
    )

if len(audit_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.1] Audit row count mismatch: got={len(audit_df)} expected={len(IOT_DRIVER_COLS)}"
    )

copy_risk_high_n = int(audit_df["copy_risk_high"].fillna(False).astype(bool).sum())

# ----------------------------------------------------------
# 7) Save artifacts
# ----------------------------------------------------------
a1_values_val_path = os.path.join(OUT_SYN, "A1_DRIVER_VALUES_VAL.parquet")
a1_metrics_csv = os.path.join(REPORT_DIR, "cell12e1_a1_driver_baseline_metrics.csv")
a1_audit_csv = os.path.join(REPORT_DIR, "cell12e1_a1_driver_baseline_audit.csv")
a1_contract_json = os.path.join(REPORT_DIR, "cell12e1_a1_driver_contract.json")
a1_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12e1_a1_driver_contract_v1_1_THESIS.json")
a1_manifest_json = os.path.join(ARTDIR, "cell12e1_a1_driver_manifest.json")

A1_DRIVER_VALUES_VAL.to_parquet(a1_values_val_path, index=True)
metrics_df.to_csv(a1_metrics_csv, index=False)
audit_df.to_csv(a1_audit_csv, index=False)

metric_summary = {
    "mean_event_rate_error": float(pd.to_numeric(metrics_df["event_rate_error"], errors="coerce").mean()),
    "mean_burst_count_error": float(pd.to_numeric(metrics_df["burst_count_error"], errors="coerce").mean()),
    "mean_interarrival_ks": float(pd.to_numeric(metrics_df["interarrival_ks"], errors="coerce").mean()),
    "mean_duration_ks": float(pd.to_numeric(metrics_df["duration_ks"], errors="coerce").mean()),
    "mean_overdispersion_error": float(pd.to_numeric(metrics_df["overdispersion_error"], errors="coerce").mean()),
    "mean_score": float(pd.to_numeric(metrics_df["score"], errors="coerce").mean()),
    "copy_risk_high_n": int(copy_risk_high_n),
}

fit_method_counts = (
    audit_df["fit_method"].astype(str).value_counts().sort_index().to_dict()
)

contract = {
    "cell": "12.e.1",
    "version": CELL12E1_VERSION,
    "role": "A1_driver_VAL_baseline",
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "expected_driver_targets_total": int(EXPECTED_DRIVER_TARGET_COUNT_12E),
    "driver_type_scope": "sparse_binary_event_drivers",
    "baseline_family": A1_DRIVER_BASELINE_FAMILY_12E1,
    "train_rows": int(N_TR),
    "val_rows": int(N_VAL),
    "test_rows_schema_only": int(N_TE),
    "fit_method_counts": fit_method_counts,
    "copy_risk_high_n": int(copy_risk_high_n),
    "metric_summary": metric_summary,
    "TEST_real_values_used": False,
    "test_length_materialization_done_here": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "A1_baseline_done_here": True,
    "train_values_used_for_fitting": True,
    "val_values_used_for_evaluation": True,
    "df_te_used_for_index_length_schema_only": True,
    "copy_risk_interpretation": "Duplicate-rate warnings are expected for sparse binary event drivers; index-coordinate overlap is diagnostic only, not direct copy evidence across TRAIN/VAL splits.",
    "outputs": {
        "a1_values_val_path": a1_values_val_path,
        "a1_metrics_csv": a1_metrics_csv,
        "a1_audit_csv": a1_audit_csv,
        "a1_contract_json": a1_contract_json,
        "a1_contract_canonical_json": a1_contract_canonical_json,
        "a1_manifest_json": a1_manifest_json,
    },
}

_write_json_12e1(a1_contract_json, contract)
_write_json_12e1(a1_contract_canonical_json, contract)

manifest = {
    "cell": "12.e.1",
    "version": CELL12E1_VERSION,
    "created_outputs": contract["outputs"],
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "baseline_family": A1_DRIVER_BASELINE_FAMILY_12E1,
    "metric_summary": metric_summary,
    "fit_method_counts": fit_method_counts,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "test_length_materialization_done_here": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": False,
    },
}

_write_json_12e1(a1_manifest_json, manifest)

hashes = {
    "a1_values_val_sha256": _sha256_file_12e1(a1_values_val_path),
    "a1_metrics_csv_sha256": _sha256_file_12e1(a1_metrics_csv),
    "a1_audit_csv_sha256": _sha256_file_12e1(a1_audit_csv),
    "a1_contract_json_sha256": _sha256_file_12e1(a1_contract_json),
    "a1_contract_canonical_json_sha256": _sha256_file_12e1(a1_contract_canonical_json),
    "a1_manifest_json_sha256": _sha256_file_12e1(a1_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12e1(a1_contract_json, contract)
_write_json_12e1(a1_contract_canonical_json, contract)
_write_json_12e1(a1_manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals for 12.e.2+
# ----------------------------------------------------------
globals()["CELL12E1_VERSION"] = CELL12E1_VERSION
globals()["A1_DRIVER_VALUES_VAL"] = A1_DRIVER_VALUES_VAL
globals()["CELL12E1_A1_DRIVER_METRICS_DF"] = metrics_df
globals()["CELL12E1_A1_DRIVER_AUDIT_DF"] = audit_df
globals()["CELL12E1_A1_DRIVER_CONTRACT"] = contract
globals()["CELL12E1_A1_DRIVER_VALUES_VAL_PATH"] = a1_values_val_path
globals()["CELL12E1_A1_DRIVER_METRICS_CSV"] = a1_metrics_csv
globals()["CELL12E1_A1_DRIVER_AUDIT_CSV"] = a1_audit_csv
globals()["CELL12E1_A1_DRIVER_CONTRACT_JSON"] = a1_contract_json
globals()["CELL12E1_A1_DRIVER_CONTRACT_CANONICAL_JSON"] = a1_contract_canonical_json
globals()["CELL12E1_A1_DRIVER_MANIFEST_JSON"] = a1_manifest_json
globals()["IOT_DRIVER_AVAIL_VAL_DF"] = IOT_DRIVER_AVAIL_VAL_DF

log(
    "[Cell12.e.1] A1 driver VAL baseline complete | "
    f"targets={len(IOT_DRIVER_COLS)} | "
    f"baseline={A1_DRIVER_BASELINE_FAMILY_12E1} | "
    f"shape={A1_DRIVER_VALUES_VAL.shape} | "
    f"copy_risk_high_n={copy_risk_high_n}"
)
log(f"[Cell12.e.1] Fit method counts | {fit_method_counts}")
log(
    "[Cell12.e.1] Metric summary | "
    f"mean_event_rate_error={metric_summary['mean_event_rate_error']:.8f} | "
    f"mean_burst_count_error={metric_summary['mean_burst_count_error']:.8f} | "
    f"mean_interarrival_ks={metric_summary['mean_interarrival_ks']:.6f} | "
    f"mean_duration_ks={metric_summary['mean_duration_ks']:.6f} | "
    f"mean_overdispersion_error={metric_summary['mean_overdispersion_error']:.6f}"
)
log(f"[Cell12.e.1] Saved A1 driver VAL values: {a1_values_val_path}")
log(f"[Cell12.e.1] Saved metrics: {a1_metrics_csv} | rows={len(metrics_df)}")
log(f"[Cell12.e.1] Saved audit: {a1_audit_csv} | rows={len(audit_df)}")
log(
    "[Cell12.e.1] Contract flags | "
    "TEST_real_values_used=False | "
    "test_length_materialization_done_here=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "df_te_used_for_index_length_schema_only=True"
)
log("--- END: Cell 12.e.1 - Driver baseline (v1.1 strict sparse binary A1, contract-hardened) ---")

gc.collect()