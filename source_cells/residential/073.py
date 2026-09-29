# ==========================================================
# CELL 12.e.4 - Driver TEST materialization
# v1.1 STUDY-THESIS strict sparse binary driver TEST materialization, contract-hardened
#
# Role:
#   - Materialize TEST-length sparse binary driver columns using locked
#     VAL-only selection from Cell 12.e.3.
#   - Refit selected generators on TRAIN only.
#   - Use df_te only for index/length/schema/time-regime alignment.
#   - Use IOT_DRIVER_AVAIL_SYN from Cell 12.e.0 for TEST active mask.
#
# Driver scope:
#   - 26 sparse binary events_in_sec__... driver targets.
#
# Strict rules:
#   - Do NOT read real TEST driver values.
#   - Do NOT evaluate against TEST real values.
#   - Do NOT change locked generator choices.
#   - Do NOT select generators here.
#
# Outputs:
#   synthetic/IOT_SELECTED_DRIVER_TEST.parquet
#   reports/cell12e4_driver_test_materialization_audit.csv
#   reports/cell12e4_driver_test_materialization_contract.json
#   artifacts/cell12e4_driver_test_materialization_manifest.json
# ==========================================================

log("--- START: Cell 12.e.4 - Driver TEST materialization (v1.1 strict sparse binary, contract-hardened) ---")

import os
import re
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_12e4 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_DRIVER_COLS",
    "IOT_DRIVER_TARGET_CONTRACT_DF",
    "IOT_DRIVER_AVAIL_SYN",
    "CELL12E_DRIVER_CONTRACT",
    "CELL12E3_LOCKED_DRIVER_SELECTION_DF",
    "CELL12E3_DRIVER_CONTRACT",
]
_missing_12e4 = [k for k in _required_12e4 if k not in globals()]
if _missing_12e4:
    raise RuntimeError(f"[Cell12.e.4] Missing required globals from prior cells: {_missing_12e4}")

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
        "[Cell12.e.4] Driver target count mismatch: "
        f"got={len(IOT_DRIVER_COLS)} expected={EXPECTED_DRIVER_TARGET_COUNT_12E}"
    )

missing_tr = sorted([c for c in IOT_DRIVER_COLS if c not in df_tr.columns])
missing_te_schema = sorted([c for c in IOT_DRIVER_COLS if c not in df_te.columns])
if missing_tr or missing_te_schema:
    raise RuntimeError(
        "[Cell12.e.4] Driver targets missing from TRAIN or TEST schema. "
        f"missing_train={missing_tr[:20]} | missing_test_schema={missing_te_schema[:20]}"
    )

CELL12E4_VERSION = "cell12e4_sparse_binary_driver_test_materialization_strict_v1_1_contract_hardened"

CFG["cell12e4_version"] = CELL12E4_VERSION
CFG["cell12e4_TEST_real_values_used"] = False
CFG["cell12e4_selection_done_here"] = False
CFG["cell12e4_generator_fit_done_here"] = True
CFG["cell12e4_train_values_used_for_fitting"] = True
CFG["cell12e4_test_length_materialization_done_here"] = True
CFG["cell12e4_df_te_used_for_index_length_schema_time_only"] = True

CFG.setdefault("cell12e4_driver_regime_bins", int(CFG.get("cell12e2_driver_regime_bins", 24)))
CFG.setdefault("cell12e4_driver_markov_eps", float(CFG.get("cell12e2_driver_markov_eps", 1e-6)))
CFG.setdefault("cell12e4_driver_sparse_event_rate_threshold", float(CFG.get("cell12e2_driver_sparse_event_rate_threshold", 0.02)))
CFG.setdefault("cell12e4_driver_block_len", int(CFG.get("cell12e1_driver_block_len", 4096)))
CFG.setdefault("cell12e4_fail_on_unknown_locked_generator", True)
CFG.setdefault("cell12e4_fail_on_binary_domain_violation", True)

LOCKED_DRIVER_GENERATORS_12E4 = {
    "A1_driver_event_block_bootstrap_reference",
    "RegimeConditionedDriverMarkov",
    "RareEventBernoulliMarkovDriver",
    "SparseEventGapRenewalDriver",
}

# ----------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------
def _json_sanitize_12e4(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12e4(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12e4(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12e4(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12e4(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12e4(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12e4(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12e4(obj.to_dict())
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

def _write_json_12e4(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12e4(payload), f, indent=2, sort_keys=True)

def _sha256_file_12e4(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _stable_seed_12e4(name: str, offset: int = 0) -> int:
    h = hashlib.sha256(f"{SEED}|12e4|{offset}|{name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)

def _rng_12e4(name: str, offset: int = 0):
    return np.random.default_rng(_stable_seed_12e4(name, offset=offset))

def _safe_float_12e4(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_bool_12e4(x, default=False) -> bool:
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

def _to_num_array_12e4(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _coerce_binary_nan_12e4(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    arr[finite & (np.abs(arr - 0.0) <= tol)] = 0.0
    arr[finite & (np.abs(arr - 1.0) <= tol)] = 1.0
    bad = finite & ~((arr == 0.0) | (arr == 1.0))
    arr[bad] = np.nan
    return arr

def _driver_observed_values_12e4(frame: pd.DataFrame, col: str) -> tuple:
    raw = _to_num_array_12e4(frame, col)
    arr = _coerce_binary_nan_12e4(
        raw,
        tol=float(CFG.get("cell12d0_binary_value_tolerance", 1e-6)),
    )
    obs = np.isfinite(arr)
    return arr, obs

def _entity_from_driver_col_12e4(col: str) -> str:
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

def _driver_measurement_12e4(col: str) -> str:
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

def _event_indices_12e4(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    return np.flatnonzero(finite & (arr >= 0.5)).astype(np.int64)

def _run_lengths_12e4(x: np.ndarray, state=None) -> np.ndarray:
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

def _interarrival_12e4(event_idx: np.ndarray) -> np.ndarray:
    event_idx = np.asarray(event_idx, dtype=np.int64)
    if event_idx.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(event_idx).astype(np.float64)

def _transition_rate_12e4(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.mean(arr[1:] != arr[:-1]))

# ----------------------------------------------------------
# 2) TEST time/regime helpers
# ----------------------------------------------------------
def _detect_time_col_12e4(frame: pd.DataFrame):
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

def _time_regime_bins_12e4(frame: pd.DataFrame, n_bins: int) -> np.ndarray:
    n = int(len(frame))
    n_bins = int(max(1, n_bins))

    tcol = _detect_time_col_12e4(frame)
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

REGIME_BINS_12E4 = int(CFG.get("cell12e4_driver_regime_bins", 24))
TRAIN_REGIME_12E4 = _time_regime_bins_12e4(df_tr, REGIME_BINS_12E4)
TEST_REGIME_12E4 = _time_regime_bins_12e4(df_te, REGIME_BINS_12E4)

TIME_COL_TR_12E4 = _detect_time_col_12e4(df_tr)
TIME_COL_TE_12E4 = _detect_time_col_12e4(df_te)

# ----------------------------------------------------------
# 3) Availability mask
# ----------------------------------------------------------
def _normalise_driver_test_availability_12e4(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        mask_df = obj.copy()
    elif isinstance(obj, dict):
        mask_df = pd.DataFrame(obj)
    else:
        arr = np.asarray(obj)
        if arr.ndim != 2:
            raise RuntimeError(
                f"[Cell12.e.4] IOT_DRIVER_AVAIL_SYN cannot be normalized: shape={arr.shape}"
            )
        mask_df = pd.DataFrame(arr)

    if len(mask_df) != N_TE:
        raise RuntimeError(
            f"[Cell12.e.4] IOT_DRIVER_AVAIL_SYN row mismatch: got={len(mask_df)} expected={N_TE}"
        )

    mask_df.index = df_te.index
    mask_df.columns = mask_df.columns.astype(str)

    missing = sorted(set(IOT_DRIVER_COLS) - set(mask_df.columns))
    if missing:
        raise RuntimeError(
            f"[Cell12.e.4] IOT_DRIVER_AVAIL_SYN missing driver columns: {missing[:30]}"
        )

    out = mask_df[IOT_DRIVER_COLS].copy()
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = (out > 0.5).astype(np.float32)

    return out

IOT_DRIVER_AVAIL_SYN_DF_12E4 = _normalise_driver_test_availability_12e4(IOT_DRIVER_AVAIL_SYN)

if IOT_DRIVER_AVAIL_SYN_DF_12E4.shape != (N_TE, len(IOT_DRIVER_COLS)):
    raise RuntimeError(
        "[Cell12.e.4] TEST driver availability shape mismatch: "
        f"got={IOT_DRIVER_AVAIL_SYN_DF_12E4.shape} expected={(N_TE, len(IOT_DRIVER_COLS))}"
    )

# ----------------------------------------------------------
# 4) Generator implementations
# ----------------------------------------------------------
def _apply_test_mask_12e4(syn: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(syn, dtype=np.float64).copy()
    active_mask = np.asarray(active_mask, dtype=bool)

    if out.size != active_mask.size:
        raise RuntimeError(
            f"[Cell12.e.4] TEST mask length mismatch: syn={out.size} mask={active_mask.size}"
        )

    out[~active_mask] = np.nan
    out[active_mask] = np.where(out[active_mask] >= 0.5, 1.0, 0.0)
    return out

def _estimate_markov_12e4(x: np.ndarray) -> dict:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    eps = float(CFG.get("cell12e4_driver_markov_eps", 1e-6))

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

def _generate_markov_12e4(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
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

def _fit_regime_conditioned_markov_12e4(train_full: np.ndarray, train_obs: np.ndarray, train_regime: np.ndarray) -> dict:
    train_full = np.asarray(train_full, dtype=np.float64)
    train_obs = np.asarray(train_obs, dtype=bool)
    train_regime = np.asarray(train_regime, dtype=np.int64)

    global_model = _estimate_markov_12e4(train_full[train_obs])

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
        m = _estimate_markov_12e4(train_full[idx])
        if bool(m.get("valid", False)):
            regime_models[int(r)] = m

    return {
        "valid": True,
        "reason": "",
        "global_model": global_model,
        "regime_models": regime_models,
    }

def _generate_regime_conditioned_markov_12e4(model: dict, n: int, test_regime: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = int(n)

    if not bool(model.get("valid", False)):
        return np.full(n, np.nan, dtype=np.float64)

    global_model = model["global_model"]
    regime_models = model.get("regime_models", {})

    out = np.empty(n, dtype=np.float64)
    state = int(rng.random() < float(np.clip(global_model.get("rate", 0.0), 0.0, 1.0)))

    for i in range(n):
        r = int(test_regime[i]) if i < len(test_regime) else 0
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

def _fit_rare_event_bernoulli_markov_driver_12e4(train_x: np.ndarray) -> dict:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {"valid": False, "reason": "empty_train"}

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))

    event_idx = _event_indices_12e4(x)
    one_runs = _run_lengths_12e4(x, state=1)
    inter = _interarrival_12e4(event_idx)

    return {
        "valid": True,
        "reason": "",
        "rate": rate,
        "event_n": int(event_idx.size),
        "one_run_lengths": one_runs,
        "interarrival": inter,
        "markov": _estimate_markov_12e4(x),
        "constant": bool(len(np.unique(x)) < 2),
    }

def _generate_rare_event_bernoulli_markov_driver_12e4(
    model: dict,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
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

def _fit_sparse_event_gap_renewal_driver_12e4(train_x: np.ndarray) -> dict:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {"valid": False, "reason": "empty_train"}

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))
    event_idx = _event_indices_12e4(x)

    if event_idx.size == 0:
        return {
            "valid": True,
            "reason": "no_train_events",
            "rate": rate,
            "event_n": 0,
            "gaps": np.asarray([], dtype=np.float64),
            "one_runs": np.asarray([], dtype=np.float64),
        }

    gaps = _interarrival_12e4(event_idx)
    one_runs = _run_lengths_12e4(x, state=1)

    return {
        "valid": True,
        "reason": "",
        "rate": rate,
        "event_n": int(event_idx.size),
        "gaps": gaps,
        "one_runs": one_runs,
    }

def _generate_sparse_event_gap_renewal_driver_12e4(model: dict, n: int, rng: np.random.Generator) -> np.ndarray:
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

    while pos < n:
        run_len = int(rng.choice(one_runs))
        run_len = max(1, run_len)
        end = min(n, pos + run_len)
        out[pos:end] = 1.0

        gap = int(rng.choice(gaps))
        jitter_radius = max(1, int(abs(gap) // 10))
        jitter = int(rng.integers(-jitter_radius, jitter_radius + 1))
        gap = max(1, gap + jitter)
        pos = end + gap

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

def _generate_a1_driver_event_block_bootstrap_12e4(
    train_x: np.ndarray,
    n: int,
    rng: np.random.Generator,
    block_len: int,
) -> np.ndarray:
    """
    Same sparse-aware A1 baseline logic as 12.e.1, refit on TRAIN only.
    """
    n = int(n)
    block_len = int(max(1, block_len))

    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.zeros(n, dtype=np.float64)

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))
    event_idx = _event_indices_12e4(x)
    event_n = int(event_idx.size)

    sparse_threshold = float(CFG.get("cell12e4_driver_sparse_event_rate_threshold", 0.02))

    if rate <= 0.0 or event_n == 0:
        return np.zeros(n, dtype=np.float64)

    if rate >= 1.0:
        return np.ones(n, dtype=np.float64)

    if rate <= sparse_threshold:
        one_runs = _run_lengths_12e4(x, state=1)
        gaps = _interarrival_12e4(event_idx)

        out = np.zeros(n, dtype=np.float64)

        if gaps.size == 0:
            expected_events = max(0.0, rate * n)
            n_events = int(rng.poisson(expected_events))
            n_events = min(n_events, n)
            if n_events > 0:
                starts = rng.choice(n, size=n_events, replace=False)
                out[starts] = 1.0
            return out

        pos = int(rng.integers(0, max(1, int(np.median(gaps)))))

        while pos < n:
            run_len = int(rng.choice(one_runs)) if one_runs.size else 1
            run_len = max(1, run_len)
            end = min(n, pos + run_len)
            out[pos:end] = 1.0

            gap = int(rng.choice(gaps))
            jitter = int(rng.integers(-max(1, gap // 10), max(2, gap // 10 + 1)))
            gap = max(1, gap + jitter)
            pos = end + gap

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

    return out

# ----------------------------------------------------------
# 5) Locked selection validation
# ----------------------------------------------------------
locked_df = CELL12E3_LOCKED_DRIVER_SELECTION_DF.copy()
locked_df["col"] = locked_df["col"].astype(str)

if len(locked_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.4] Locked selection row count mismatch: got={len(locked_df)} expected={len(IOT_DRIVER_COLS)}"
    )

if locked_df["col"].duplicated().any():
    dupes = locked_df.loc[locked_df["col"].duplicated(), "col"].astype(str).tolist()
    raise RuntimeError(f"[Cell12.e.4] Duplicate locked driver selection rows: {dupes[:20]}")

missing_locked = sorted(set(IOT_DRIVER_COLS) - set(locked_df["col"].astype(str)))
if missing_locked:
    raise RuntimeError(f"[Cell12.e.4] Missing locked driver selections: {missing_locked[:20]}")

locked_map = {str(r["col"]): r.to_dict() for _, r in locked_df.iterrows()}

unknown_locked = sorted(
    set(locked_df["selected_driver_generator"].astype(str)) - LOCKED_DRIVER_GENERATORS_12E4
)
if unknown_locked and bool(CFG.get("cell12e4_fail_on_unknown_locked_generator", True)):
    raise RuntimeError(f"[Cell12.e.4] Unknown locked driver generators: {unknown_locked}")

# ----------------------------------------------------------
# 6) Materialize TEST selected drivers
# ----------------------------------------------------------
selected_test_data = {}
audit_rows = []

log(
    "[Cell12.e.4] Materializing TEST-length sparse binary driver values | "
    f"targets={len(IOT_DRIVER_COLS)} | N_TE={N_TE}"
)

for j, col in enumerate(IOT_DRIVER_COLS, start=1):
    if j == 1 or j % 5 == 0 or j == len(IOT_DRIVER_COLS):
        log(f"[Cell12.e.4] progress {j}/{len(IOT_DRIVER_COLS)} | col={col}")

    locked = locked_map[col]

    selected_generator = str(locked.get("selected_driver_generator", ""))
    selected_candidate_id = str(locked.get("selected_candidate_id", ""))

    tr_full, tr_obs = _driver_observed_values_12e4(df_tr, col)
    train_x = tr_full[tr_obs]

    active_mask = (
        pd.to_numeric(IOT_DRIVER_AVAIL_SYN_DF_12E4[col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        > 0.5
    )

    fit_valid = True
    materializer = ""
    materialization_reason = ""
    materialization_warning = ""
    syn = None

    try:
        if selected_generator == "A1_driver_event_block_bootstrap_reference":
            if train_x.size == 0:
                syn = np.zeros(N_TE, dtype=np.float64)
                materialization_reason = "a1_driver_empty_train_zero_baseline"
            else:
                syn = _generate_a1_driver_event_block_bootstrap_12e4(
                    train_x=train_x,
                    n=N_TE,
                    rng=_rng_12e4(col, offset=101),
                    block_len=int(CFG.get("cell12e4_driver_block_len", 4096)),
                )
                materialization_reason = "a1_sparse_driver_event_block_bootstrap"
            materializer = "a1_driver_event_block_bootstrap_locked_materializer"

        elif selected_generator == "RegimeConditionedDriverMarkov":
            if train_x.size == 0:
                raise RuntimeError("RegimeConditionedDriverMarkov selected but TRAIN is empty")
            model = _fit_regime_conditioned_markov_12e4(
                train_full=tr_full,
                train_obs=tr_obs,
                train_regime=TRAIN_REGIME_12E4,
            )
            if not bool(model.get("valid", False)):
                raise RuntimeError(f"invalid regime-conditioned driver markov fit: {model.get('reason', '')}")
            syn = _generate_regime_conditioned_markov_12e4(
                model=model,
                n=N_TE,
                test_regime=TEST_REGIME_12E4,
                rng=_rng_12e4(col, offset=202),
            )
            materializer = "regime_conditioned_driver_markov_locked_materializer"
            materialization_reason = "selected_non_a1_regime_conditioned_driver_markov"

        elif selected_generator == "RareEventBernoulliMarkovDriver":
            if train_x.size == 0:
                raise RuntimeError("RareEventBernoulliMarkovDriver selected but TRAIN is empty")
            model = _fit_rare_event_bernoulli_markov_driver_12e4(train_x)
            if not bool(model.get("valid", False)):
                raise RuntimeError(f"invalid rare-event driver fit: {model.get('reason', '')}")
            syn = _generate_rare_event_bernoulli_markov_driver_12e4(
                model=model,
                n=N_TE,
                rng=_rng_12e4(col, offset=303),
            )
            materializer = "rare_event_bernoulli_markov_driver_locked_materializer"
            materialization_reason = "selected_non_a1_rare_event_bernoulli_markov_driver"

        elif selected_generator == "SparseEventGapRenewalDriver":
            if train_x.size == 0:
                raise RuntimeError("SparseEventGapRenewalDriver selected but TRAIN is empty")
            model = _fit_sparse_event_gap_renewal_driver_12e4(train_x)
            if not bool(model.get("valid", False)):
                raise RuntimeError(f"invalid sparse-event gap renewal driver fit: {model.get('reason', '')}")
            syn = _generate_sparse_event_gap_renewal_driver_12e4(
                model=model,
                n=N_TE,
                rng=_rng_12e4(col, offset=404),
            )
            materializer = "sparse_event_gap_renewal_driver_locked_materializer"
            materialization_reason = "selected_non_a1_sparse_event_gap_renewal_driver"

        else:
            raise RuntimeError(f"Unknown locked selected_driver_generator={selected_generator}")

    except Exception as e:
        fit_valid = False
        materialization_warning = f"materialization_exception:{type(e).__name__}:{e}"
        if bool(CFG.get("cell12e4_fail_on_unknown_locked_generator", True)):
            raise RuntimeError(
                f"[Cell12.e.4] Failed materializing {col} with {selected_generator}: {e}"
            )
        syn = np.full(N_TE, np.nan, dtype=np.float64)

    if syn is None:
        raise RuntimeError(f"[Cell12.e.4] Internal error: syn is None for {col}")

    syn = _coerce_binary_nan_12e4(syn)
    syn = _apply_test_mask_12e4(syn, active_mask)

    active = active_mask
    inactive = ~active

    arr = np.asarray(syn, dtype=np.float64)
    finite = np.isfinite(arr)

    inactive_finite_n = int(np.isfinite(arr[inactive]).sum())
    active_nonfinite_n = int(np.sum(active & ~np.isfinite(arr)))
    nonbinary_finite_n = int(np.sum(finite & ~((arr == 0.0) | (arr == 1.0))))

    if inactive_finite_n != 0:
        raise RuntimeError(
            f"[Cell12.e.4] Inactive finite driver values for {col}: {inactive_finite_n}"
        )

    if active_nonfinite_n != 0:
        raise RuntimeError(
            f"[Cell12.e.4] Active nonfinite driver values for {col}: {active_nonfinite_n}"
        )

    if nonbinary_finite_n != 0 and bool(CFG.get("cell12e4_fail_on_binary_domain_violation", True)):
        raise RuntimeError(
            f"[Cell12.e.4] Non-binary finite driver values for {col}: {nonbinary_finite_n}"
        )

    selected_test_data[col] = arr.astype(np.float32)

    active_vals = arr[active & np.isfinite(arr)]
    train_rate = float(np.mean(train_x)) if train_x.size else np.nan
    syn_rate = float(np.mean(active_vals)) if active_vals.size else np.nan

    audit_rows.append({
        "col": col,
        "entity": _entity_from_driver_col_12e4(col),
        "measurement_name": _driver_measurement_12e4(col),
        "selected_driver_generator": selected_generator,
        "selected_candidate_id": selected_candidate_id,
        "materializer": materializer,
        "materialization_reason": materialization_reason,
        "materialization_warning": materialization_warning,
        "fit_valid": bool(fit_valid),
        "publication_status_after_12e3": str(locked.get("driver_publication_status_after_12e3", "")),
        "publication_reasons_after_12e3": str(locked.get("driver_publication_reasons_after_12e3", "")),
        "publication_blocker_after_12e3": _safe_bool_12e4(locked.get("driver_publication_blocker_after_12e3"), False),
        "train_observed_n": int(train_x.size),
        "train_event_rate": train_rate,
        "train_event_count": int(np.sum(train_x >= 0.5)) if train_x.size else 0,
        "test_rows": int(N_TE),
        "test_active_n": int(active.sum()),
        "test_inactive_n": int(inactive.sum()),
        "test_active_rate": float(active.mean()) if active.size else np.nan,
        "syn_active_n": int(active_vals.size),
        "syn_event_rate_active": syn_rate,
        "syn_event_count_active": int(np.sum(active_vals >= 0.5)) if active_vals.size else 0,
        "syn_transition_rate_active": _transition_rate_12e4(active_vals),
        "syn_unique_values_active": sorted([float(v) for v in np.unique(active_vals)]) if active_vals.size else [],
        "inactive_finite_n": inactive_finite_n,
        "active_nonfinite_n": active_nonfinite_n,
        "nonbinary_finite_n": nonbinary_finite_n,
        "feeds_cell13_coupling_qa": True,
        "TEST_real_values_used": False,
        "df_te_used_for_index_length_schema_time_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": True,
    })

# ----------------------------------------------------------
# 7) Assemble and validate selected TEST driver matrix
# ----------------------------------------------------------
IOT_SELECTED_DRIVER_TEST = pd.DataFrame(
    selected_test_data,
    index=df_te.index,
    columns=IOT_DRIVER_COLS,
    dtype=np.float32,
)

if IOT_SELECTED_DRIVER_TEST.shape != (N_TE, len(IOT_DRIVER_COLS)):
    raise RuntimeError(
        "[Cell12.e.4] IOT_SELECTED_DRIVER_TEST shape mismatch: "
        f"got={IOT_SELECTED_DRIVER_TEST.shape} expected={(N_TE, len(IOT_DRIVER_COLS))}"
    )

audit_df = pd.DataFrame(audit_rows)

if len(audit_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.4] Audit row count mismatch: got={len(audit_df)} expected={len(IOT_DRIVER_COLS)}"
    )

global_inactive_finite = 0
global_active_nonfinite = 0
global_nonbinary_finite = 0

for col in IOT_DRIVER_COLS:
    arr = pd.to_numeric(IOT_SELECTED_DRIVER_TEST[col], errors="coerce").to_numpy(dtype=np.float64)
    mask = (
        pd.to_numeric(IOT_DRIVER_AVAIL_SYN_DF_12E4[col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        > 0.5
    )

    finite = np.isfinite(arr)
    global_inactive_finite += int(np.sum(~mask & finite))
    global_active_nonfinite += int(np.sum(mask & ~finite))
    global_nonbinary_finite += int(np.sum(finite & ~((arr == 0.0) | (arr == 1.0))))

if global_inactive_finite != 0:
    raise RuntimeError(f"[Cell12.e.4] Global inactive finite violation: {global_inactive_finite}")

if global_active_nonfinite != 0:
    raise RuntimeError(f"[Cell12.e.4] Global active nonfinite violation: {global_active_nonfinite}")

if global_nonbinary_finite != 0:
    raise RuntimeError(f"[Cell12.e.4] Global non-binary finite violation: {global_nonbinary_finite}")

# ----------------------------------------------------------
# 8) Save outputs
# ----------------------------------------------------------
selected_driver_test_path = os.path.join(OUT_SYN, "IOT_SELECTED_DRIVER_TEST.parquet")
materialization_audit_csv = os.path.join(REPORT_DIR, "cell12e4_driver_test_materialization_audit.csv")
contract_json = os.path.join(REPORT_DIR, "cell12e4_driver_test_materialization_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12e4_driver_test_materialization_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell12e4_driver_test_materialization_manifest.json")

IOT_SELECTED_DRIVER_TEST.to_parquet(selected_driver_test_path, index=True)
audit_df.to_csv(materialization_audit_csv, index=False)

selected_counts = (
    audit_df["selected_driver_generator"].astype(str).value_counts().sort_index().to_dict()
)
materializer_counts = (
    audit_df["materializer"].astype(str).value_counts().sort_index().to_dict()
)
publication_counts_carried = (
    audit_df["publication_status_after_12e3"].astype(str).value_counts().sort_index().to_dict()
)

safety_summary = {
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "test_rows": int(N_TE),
    "test_active_total": int(audit_df["test_active_n"].sum()),
    "test_inactive_total": int(audit_df["test_inactive_n"].sum()),
    "inactive_finite_total": int(audit_df["inactive_finite_n"].sum()),
    "active_nonfinite_total": int(audit_df["active_nonfinite_n"].sum()),
    "nonbinary_finite_total": int(audit_df["nonbinary_finite_n"].sum()),
    "safety_failure_count": int(
        (
            (audit_df["inactive_finite_n"].astype(int) > 0)
            | (audit_df["active_nonfinite_n"].astype(int) > 0)
            | (audit_df["nonbinary_finite_n"].astype(int) > 0)
        ).sum()
    ),
    "publication_blocker_carried_from_12e3_n": int(
        audit_df["publication_blocker_after_12e3"].fillna(False).astype(bool).sum()
    ),
}

contract = {
    "cell": "12.e.4",
    "version": CELL12E4_VERSION,
    "role": "sparse_binary_driver_TEST_length_materialization_from_locked_VAL_selection",
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "expected_driver_targets_total": int(EXPECTED_DRIVER_TARGET_COUNT_12E),
    "driver_type_scope": "sparse_binary_event_drivers",
    "train_rows": int(N_TR),
    "val_rows_schema_only": int(N_VAL),
    "test_rows": int(N_TE),
    "time_columns_used_for_regime_alignment": {
        "train_time_col": TIME_COL_TR_12E4,
        "test_time_col": TIME_COL_TE_12E4,
        "regime_bins": int(REGIME_BINS_12E4),
    },
    "selected_counts": selected_counts,
    "materializer_counts": materializer_counts,
    "publication_counts_carried_from_12e3": publication_counts_carried,
    "safety_summary": safety_summary,
    "feeds_cell13_coupling_qa": True,
    "TEST_real_values_used": False,
    "test_length_materialization_done_here": True,
    "selection_done_here": False,
    "generator_fit_done_here": True,
    "train_values_used_for_fitting": True,
    "df_te_used_for_index_length_schema_time_only": True,
    "raw_TEST_driver_values_read": False,
    "time_regime_alignment_note": "df_te is used only to derive TEST-length/time-regime bins for materialization; real TEST driver target values are not read or evaluated here.",
    "outputs": {
        "selected_driver_test_path": selected_driver_test_path,
        "materialization_audit_csv": materialization_audit_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_12e4(contract_json, contract)
_write_json_12e4(contract_canonical_json, contract)

manifest = {
    "cell": "12.e.4",
    "version": CELL12E4_VERSION,
    "created_outputs": contract["outputs"],
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "selected_counts": selected_counts,
    "materializer_counts": materializer_counts,
    "safety_summary": safety_summary,
    "feeds_cell13_coupling_qa": True,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "df_te_used_for_index_length_schema_time_only": True,
        "test_length_materialization_done_here": True,
        "selection_done_here": False,
        "generator_fit_done_here": True,
    },
}

_write_json_12e4(manifest_json, manifest)

hashes = {
    "selected_driver_test_sha256": _sha256_file_12e4(selected_driver_test_path),
    "materialization_audit_csv_sha256": _sha256_file_12e4(materialization_audit_csv),
    "contract_json_sha256": _sha256_file_12e4(contract_json),
    "contract_canonical_json_sha256": _sha256_file_12e4(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_12e4(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12e4(contract_json, contract)
_write_json_12e4(contract_canonical_json, contract)
_write_json_12e4(manifest_json, manifest)

# ----------------------------------------------------------
# 9) Export globals for 12.e.5+
# ----------------------------------------------------------
globals()["CELL12E4_VERSION"] = CELL12E4_VERSION
globals()["IOT_SELECTED_DRIVER_TEST"] = IOT_SELECTED_DRIVER_TEST
globals()["CELL12E4_DRIVER_TEST_MATERIALIZATION_AUDIT_DF"] = audit_df
globals()["CELL12E4_DRIVER_TEST_MATERIALIZATION_CONTRACT"] = contract
globals()["CELL12E4_SELECTED_DRIVER_TEST_PATH"] = selected_driver_test_path
globals()["CELL12E4_DRIVER_TEST_MATERIALIZATION_AUDIT_CSV"] = materialization_audit_csv
globals()["CELL12E4_DRIVER_TEST_MATERIALIZATION_CONTRACT_JSON"] = contract_json
globals()["CELL12E4_DRIVER_TEST_MATERIALIZATION_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL12E4_DRIVER_TEST_MATERIALIZATION_MANIFEST_JSON"] = manifest_json

log(
    "[Cell12.e.4] Driver TEST materialization complete | "
    f"targets={len(IOT_DRIVER_COLS)} | "
    f"shape={IOT_SELECTED_DRIVER_TEST.shape} | "
    f"safety_failure_count={safety_summary['safety_failure_count']} | "
    f"publication_blocker_carried_from_12e3_n={safety_summary['publication_blocker_carried_from_12e3_n']}"
)
log(f"[Cell12.e.4] Selected generator counts | {selected_counts}")
log(f"[Cell12.e.4] Materializer counts | {materializer_counts}")
log(
    "[Cell12.e.4] Safety summary | "
    f"active_total={safety_summary['test_active_total']} | "
    f"inactive_total={safety_summary['test_inactive_total']} | "
    f"inactive_finite_total={safety_summary['inactive_finite_total']} | "
    f"active_nonfinite_total={safety_summary['active_nonfinite_total']} | "
    f"nonbinary_finite_total={safety_summary['nonbinary_finite_total']}"
)
log(f"[Cell12.e.4] Saved selected driver TEST values: {selected_driver_test_path}")
log(f"[Cell12.e.4] Saved materialization audit: {materialization_audit_csv} | rows={len(audit_df)}")
log(f"[Cell12.e.4] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell12.e.4] Contract flags | "
    "TEST_real_values_used=False | "
    "test_length_materialization_done_here=True | "
    "selection_done_here=False | "
    "generator_fit_done_here=True | "
    "df_te_used_for_index_length_schema_time_only=True"
)
log("--- END: Cell 12.e.4 - Driver TEST materialization (v1.1 strict sparse binary, contract-hardened) ---")

gc.collect()