# ==========================================================
# CELL 12.f.1 - Observability mask baseline
# v1.1 STUDY-THESIS strict Q3 mask baseline, contract-hardened
#
# Role:
#   - Build A0/A1 VAL baselines for direct observability targets.
#   - A0: diagnostic real-mask upper bound on VAL only.
#   - A1: TRAIN-fitted baseline mask model, VAL-length.
#
# Scope from 12.f.0:
#   - entity_obs
#   - entity_stale
#   - global_update_indicator
#   - observed_indicator
#   - tier_present_indicator
#
# Strict rules:
#   - A0 uses real VAL masks only as diagnostic upper bound.
#   - A0 must not be selected or materialized.
#   - A1 fits on TRAIN only.
#   - Do NOT read real TEST mask values.
#   - Do NOT generate TEST-length masks.
#   - Do NOT select final mask model.
#
# Outputs:
#   synthetic/A0_OBSERVABILITY_MASKS_VAL.parquet
#   synthetic/A1_OBSERVABILITY_MASKS_VAL.parquet
#   reports/cell12f1_mask_baseline_metrics.csv
#   reports/cell12f1_mask_baseline_audit.csv
#   reports/cell12f1_mask_baseline_contract.json
#   artifacts/cell12f1_mask_baseline_manifest.json
# ==========================================================

log("--- START: Cell 12.f.1 - Observability mask baseline (v1.1 strict Q3, contract-hardened) ---")

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
_required_12f1 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_OBSERVABILITY_COLS",
    "IOT_OBSERVABILITY_TARGET_CONTRACT_DF",
    "IOT_OBSERVABILITY_FAMILY_CONTRACT_DF",
    "CELL12F_OBSERVABILITY_CONTRACT",
]
_missing_12f1 = [k for k in _required_12f1 if k not in globals()]
if _missing_12f1:
    raise RuntimeError(f"[Cell12.f.1] Missing required globals from prior cells: {_missing_12f1}")

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
    raise RuntimeError("[Cell12.f.1] IOT_OBSERVABILITY_COLS is empty.")

missing_tr = sorted([c for c in IOT_OBSERVABILITY_COLS if c not in df_tr.columns])
missing_val = sorted([c for c in IOT_OBSERVABILITY_COLS if c not in df_val.columns])
missing_te_schema = sorted([c for c in IOT_OBSERVABILITY_COLS if c not in df_te.columns])

if missing_tr or missing_val or missing_te_schema:
    raise RuntimeError(
        "[Cell12.f.1] Observability targets missing from split schemas. "
        f"missing_train={missing_tr[:10]} | missing_val={missing_val[:10]} | "
        f"missing_test_schema={missing_te_schema[:10]}"
    )

_bad_direct_targets_12f1 = [
    c for c in IOT_OBSERVABILITY_COLS
    if not str(c).startswith("iot__")
]
if _bad_direct_targets_12f1:
    raise RuntimeError(
        "[Cell12.f.1] Direct observability mask targets must be iot__ observability columns. "
        f"Preview={_bad_direct_targets_12f1[:30]}"
    )

CELL12F1_VERSION = "cell12f1_observability_mask_baseline_strict_v1_1_contract_hardened"

CFG["cell12f1_version"] = CELL12F1_VERSION
CFG["cell12f1_TEST_real_values_used"] = False
CFG["cell12f1_A0_real_val_used_for_diagnostic_only"] = True
CFG["cell12f1_A0_selectable"] = False
CFG["cell12f1_A1_baseline_done_here"] = True
CFG["cell12f1_A0_real_val_artifact_is_diagnostic_only"] = True
CFG["cell12f1_selection_done_here"] = False
CFG["cell12f1_generator_fit_done_here"] = False  # A1 baseline only; non-A1 generator fitting starts in 12.f.2
CFG["cell12f1_test_length_materialization_done_here"] = False
CFG["cell12f1_quality_dimension"] = "Q3_observability"

CFG.setdefault("cell12f1_mask_block_len", 4096)
CFG.setdefault("cell12f1_mask_markov_eps", 1e-6)
CFG.setdefault("cell12f1_near_constant_threshold", float(CFG.get("cell12f0_mask_near_constant_threshold", 0.995)))
CFG.setdefault("cell12f1_sparse_threshold", float(CFG.get("cell12f0_mask_sparse_threshold", 0.05)))
CFG.setdefault("cell12f1_min_val_observed_n", 1)

A0_MASK_FAMILY_12F1 = "A0_real_VAL_mask_upper_bound_diagnostic"
A1_MASK_FAMILY_12F1 = "A1_train_fitted_mask_baseline"

# ----------------------------------------------------------
# 1) Validate upstream Cell 12.f.0 contract
# ----------------------------------------------------------
contract0 = CELL12F_OBSERVABILITY_CONTRACT
if not isinstance(contract0, dict):
    raise RuntimeError("[Cell12.f.1] CELL12F_OBSERVABILITY_CONTRACT is not a dict.")

version0 = str(contract0.get("version", ""))
if "cell12f0_observability_contract_strict_v1_1" not in version0 and "cell12f0_observability_contract_strict_v1_0" not in version0:
    raise RuntimeError(
        "[Cell12.f.1] Unexpected Cell 12.f.0 contract version. "
        f"Expected v1.0/v1.1 compatible contract, got: {version0}"
    )

if int(contract0.get("generated_target_direct_leak_n", -1)) != 0:
    raise RuntimeError(
        "[Cell12.f.1] Refusing to build mask baseline because 12.f.0 reported generated target leakage: "
        f"{contract0.get('generated_target_direct_leak_n')}"
    )

if bool(contract0.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell12.f.1] Cell 12.f.0 contract indicates TEST values were used.")


# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_12f1(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12f1(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12f1(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12f1(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12f1(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12f1(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12f1(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12f1(obj.to_dict())
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

def _write_json_12f1(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12f1(payload), f, indent=2, sort_keys=True)

def _sha256_file_12f1(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _stable_seed_12f1(name: str, offset: int = 0) -> int:
    h = hashlib.sha256(f"{SEED}|12f1|{offset}|{name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)

def _rng_12f1(name: str, offset: int = 0):
    return np.random.default_rng(_stable_seed_12f1(name, offset=offset))

def _safe_float_12f1(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_12f1(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _coerce_mask_nan_12f1(x: np.ndarray, binary_required: bool = True) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(arr)

    if binary_required:
        arr[finite & (arr < 0.5)] = 0.0
        arr[finite & (arr >= 0.5)] = 1.0
    else:
        # For stale/age-like values, convert to observed/stale indicator for Q3 mask QA.
        # 0 = not stale/false, 1 = stale/true if value >= 0.5
        arr[finite & (arr < 0.5)] = 0.0
        arr[finite & (arr >= 0.5)] = 1.0

    return arr

def _mask_observed_values_12f1(frame: pd.DataFrame, col: str, binary_required: bool = True) -> tuple:
    raw = _to_num_array_12f1(frame, col)
    arr = _coerce_mask_nan_12f1(raw, binary_required=binary_required)
    obs = np.isfinite(arr)
    return arr, obs

def _obs_kind_for_col_12f1(col: str) -> str:
    if "IOT_OBSERVABILITY_TARGET_CONTRACT_DF" in globals():
        d = IOT_OBSERVABILITY_TARGET_CONTRACT_DF
        if isinstance(d, pd.DataFrame) and "col" in d.columns and "obs_kind" in d.columns:
            m = d[d["col"].astype(str).eq(str(col))]
            if len(m):
                return str(m["obs_kind"].iloc[0])
    s = str(col).lower()
    if s.startswith("iot__entity_obs__"):
        return "entity_obs"
    if s.startswith("iot__entity_stale__"):
        return "entity_stale"
    if s == "iot__obs_present":
        return "observed_indicator"
    if s == "iot__tier_present":
        return "tier_present_indicator"
    if s == "iot__any_update_raw":
        return "global_update_indicator"
    return "observability_indicator"

def _entity_from_obs_col_12f1(col: str) -> str:
    s = str(col)
    if s.startswith("iot__entity_obs__"):
        return s.split("iot__entity_obs__", 1)[1]
    if s.startswith("iot__entity_stale__"):
        return s.split("iot__entity_stale__", 1)[1]
    if s.startswith("iot__"):
        s = s[5:]
    return s.split("__")[0] if "__" in s else s

def _transition_probs_12f1(x: np.ndarray) -> dict:
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

def _run_lengths_12f1(x: np.ndarray, state=None) -> np.ndarray:
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

def _empirical_ks_12f1(a: np.ndarray, b: np.ndarray) -> float:
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

def _mask_metrics_12f1(real: np.ndarray, syn: np.ndarray) -> dict:
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
            "score": np.inf,
        }

    real = np.where(real >= 0.5, 1.0, 0.0)
    syn = np.where(syn >= 0.5, 1.0, 0.0)

    real_rate = float(np.mean(real))
    syn_rate = float(np.mean(syn))
    obs_rate_error = abs(real_rate - syn_rate)

    rt = _transition_probs_12f1(real)
    st = _transition_probs_12f1(syn)

    p11_error = abs(_safe_float_12f1(rt["p11"], np.nan) - _safe_float_12f1(st["p11"], np.nan))
    p00_error = abs(_safe_float_12f1(rt["p00"], np.nan) - _safe_float_12f1(st["p00"], np.nan))

    real_runs = _run_lengths_12f1(real)
    syn_runs = _run_lengths_12f1(syn)
    run_ks = _empirical_ks_12f1(real_runs, syn_runs)

    components = {
        "obs_rate_error": obs_rate_error,
        "p11_error": p11_error,
        "p00_error": p00_error,
        "run_length_ks": run_ks,
    }
    for k, v in list(components.items()):
        if not np.isfinite(v):
            components[k] = 1.0

    score = (
        2.0 * components["obs_rate_error"]
        + 1.0 * components["p11_error"]
        + 1.0 * components["p00_error"]
        + 0.75 * components["run_length_ks"]
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
        "score": float(score),
    }

# ----------------------------------------------------------
# 3) A1 baseline generators
# ----------------------------------------------------------
def _generate_constant_rate_mask_12f1(train_x: np.ndarray, n: int, rng: np.random.Generator) -> tuple:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.zeros(n, dtype=np.float64), {
            "method": "empty_train_constant_zero",
            "train_obs_rate": np.nan,
        }

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))

    near_constant = float(CFG.get("cell12f1_near_constant_threshold", 0.995))
    if rate >= near_constant:
        return np.ones(n, dtype=np.float64), {
            "method": "near_always_observed_constant_one",
            "train_obs_rate": rate,
        }

    if rate <= 1.0 - near_constant:
        return np.zeros(n, dtype=np.float64), {
            "method": "near_never_observed_constant_zero",
            "train_obs_rate": rate,
        }

    out = (rng.random(n) < rate).astype(np.float64)
    return out, {
        "method": "bernoulli_constant_rate",
        "train_obs_rate": rate,
    }

def _generate_markov_mask_12f1(train_x: np.ndarray, n: int, rng: np.random.Generator) -> tuple:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]
    eps = float(CFG.get("cell12f1_mask_markov_eps", 1e-6))

    if x.size == 0:
        return np.zeros(n, dtype=np.float64), {
            "method": "empty_train_markov_zero",
            "train_obs_rate": np.nan,
        }

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))

    if x.size < 2 or len(np.unique(x)) < 2:
        out = np.full(n, 1.0 if rate >= 0.5 else 0.0, dtype=np.float64)
        return out, {
            "method": "constant_from_degenerate_markov",
            "train_obs_rate": rate,
        }

    a = x[:-1].astype(np.int8)
    b = x[1:].astype(np.int8)

    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))

    p01 = (n01 + eps) / max(n00 + n01 + 2 * eps, eps)
    p10 = (n10 + eps) / max(n10 + n11 + 2 * eps, eps)

    out = np.empty(n, dtype=np.float64)
    state = int(rng.random() < rate)
    out[0] = float(state)

    for i in range(1, n):
        if state == 0:
            state = int(rng.random() < p01)
        else:
            state = int(not (rng.random() < p10))
        out[i] = float(state)

    return out, {
        "method": "train_markov_mask",
        "train_obs_rate": rate,
        "p01": float(p01),
        "p10": float(p10),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }

def _generate_block_bootstrap_mask_12f1(train_x: np.ndarray, n: int, rng: np.random.Generator, block_len: int) -> tuple:
    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.zeros(n, dtype=np.float64), {
            "method": "empty_train_block_zero",
            "train_obs_rate": np.nan,
        }

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))

    if x.size == 1:
        return np.full(n, float(x[0]), dtype=np.float64), {
            "method": "single_train_value_constant",
            "train_obs_rate": rate,
        }

    out = np.empty(n, dtype=np.float64)
    pos = 0
    max_start = max(0, x.size - 1)
    block_len = int(max(1, block_len))

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
        "method": "train_block_bootstrap_mask",
        "train_obs_rate": rate,
        "block_len": block_len,
    }

def _generate_a1_mask_baseline_12f1(col: str, train_x: np.ndarray, n: int, rng: np.random.Generator) -> tuple:
    obs_kind = _obs_kind_for_col_12f1(col)

    x = np.asarray(train_x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.zeros(n, dtype=np.float64), {
            "method": "empty_train_zero_baseline",
            "obs_kind": obs_kind,
        }

    x = np.where(x >= 0.5, 1.0, 0.0)
    rate = float(np.mean(x))
    near_constant = float(CFG.get("cell12f1_near_constant_threshold", 0.995))
    sparse_threshold = float(CFG.get("cell12f1_sparse_threshold", 0.05))

    if rate >= near_constant or rate <= (1.0 - near_constant):
        return _generate_constant_rate_mask_12f1(x, n, rng)

    if obs_kind in {"entity_stale", "stale_or_age"}:
        return _generate_block_bootstrap_mask_12f1(
            train_x=x,
            n=n,
            rng=rng,
            block_len=int(CFG.get("cell12f1_mask_block_len", 4096)),
        )

    if rate <= sparse_threshold or rate >= 1.0 - sparse_threshold:
        return _generate_markov_mask_12f1(x, n, rng)

    return _generate_markov_mask_12f1(x, n, rng)

# ----------------------------------------------------------
# 4) Generate A0/A1 VAL baselines
# ----------------------------------------------------------
a0_data = {}
a1_data = {}
metrics_rows = []
audit_rows = []

log(
    "[Cell12.f.1] Building A0/A1 mask VAL baselines | "
    f"targets={len(IOT_OBSERVABILITY_COLS)} | N_VAL={N_VAL}"
)

for j, col in enumerate(IOT_OBSERVABILITY_COLS, start=1):
    if j == 1 or j % 25 == 0 or j == len(IOT_OBSERVABILITY_COLS):
        log(f"[Cell12.f.1] progress {j}/{len(IOT_OBSERVABILITY_COLS)} | col={col}")

    obs_kind = _obs_kind_for_col_12f1(col)
    binary_required = True

    tr_full, tr_obs = _mask_observed_values_12f1(df_tr, col, binary_required=binary_required)
    val_full, val_obs = _mask_observed_values_12f1(df_val, col, binary_required=binary_required)

    train_x = tr_full[tr_obs]
    val_real = val_full[val_obs]

    if val_real.size < int(CFG.get("cell12f1_min_val_observed_n", 1)):
        raise RuntimeError(
            f"[Cell12.f.1] Insufficient VAL observed mask values for {col}: val_observed_n={val_real.size}"
        )

    # A0 diagnostic real VAL upper bound.
    a0_syn = val_full.copy()
    a0_syn[~val_obs] = np.nan
    a0_data[col] = a0_syn.astype(np.float32)

    a0_metrics = _mask_metrics_12f1(real=val_real, syn=a0_syn[val_obs])
    metrics_rows.append({
        "col": col,
        "entity": _entity_from_obs_col_12f1(col),
        "obs_kind": obs_kind,
        "candidate_id": f"{col}__{A0_MASK_FAMILY_12F1}",
        "candidate_family": A0_MASK_FAMILY_12F1,
        "candidate_source": "A0_diagnostic",
        "is_a0_diagnostic": True,
        "is_a1": False,
        "selectable": False,
        "candidate_valid": True,
        "valid_for_selection": False,
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(val_real.size),
        "train_obs_rate": float(np.mean(train_x)) if train_x.size else np.nan,
        "val_obs_rate": float(np.mean(val_real)) if val_real.size else np.nan,
        "TEST_real_values_used": False,
        "A0_real_VAL_used_for_diagnostic_only": True,
        **a0_metrics,
    })

    # A1 train-fitted baseline.
    rng = _rng_12f1(col, offset=101)
    a1_raw, fit_summary = _generate_a1_mask_baseline_12f1(
        col=col,
        train_x=train_x,
        n=N_VAL,
        rng=rng,
    )

    a1_syn = _coerce_mask_nan_12f1(a1_raw, binary_required=True)
    a1_syn[~val_obs] = np.nan
    a1_data[col] = a1_syn.astype(np.float32)

    a1_metrics = _mask_metrics_12f1(real=val_real, syn=a1_syn[val_obs])

    metrics_rows.append({
        "col": col,
        "entity": _entity_from_obs_col_12f1(col),
        "obs_kind": obs_kind,
        "candidate_id": f"{col}__{A1_MASK_FAMILY_12F1}",
        "candidate_family": A1_MASK_FAMILY_12F1,
        "candidate_source": "A1",
        "is_a0_diagnostic": False,
        "is_a1": True,
        "selectable": True,
        "candidate_valid": True,
        "valid_for_selection": True,
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(val_real.size),
        "train_obs_rate": float(np.mean(train_x)) if train_x.size else np.nan,
        "val_obs_rate": float(np.mean(val_real)) if val_real.size else np.nan,
        "fit_summary": json.dumps(_json_sanitize_12f1(fit_summary), sort_keys=True),
        "TEST_real_values_used": False,
        "A0_real_VAL_used_for_diagnostic_only": False,
        **a1_metrics,
    })

    audit_rows.append({
        "col": col,
        "entity": _entity_from_obs_col_12f1(col),
        "obs_kind": obs_kind,
        "train_observed_n": int(train_x.size),
        "val_observed_n": int(val_real.size),
        "train_obs_rate": float(np.mean(train_x)) if train_x.size else np.nan,
        "val_obs_rate": float(np.mean(val_real)) if val_real.size else np.nan,
        "abs_train_val_obs_rate_delta": (
            abs(float(np.mean(train_x)) - float(np.mean(val_real)))
            if train_x.size and val_real.size
            else np.nan
        ),
        "a1_fit_method": fit_summary.get("method", ""),
        "a1_obs_rate_error": a1_metrics.get("obs_rate_error", np.nan),
        "a1_p11_error": a1_metrics.get("p11_error", np.nan),
        "a1_p00_error": a1_metrics.get("p00_error", np.nan),
        "a1_run_length_ks": a1_metrics.get("run_length_ks", np.nan),
        "A0_diagnostic_only": True,
        "A0_selectable": False,
        "TEST_real_values_used": False,
    })

# ----------------------------------------------------------
# 5) Assemble and validate outputs
# ----------------------------------------------------------
A0_OBSERVABILITY_MASKS_VAL = pd.DataFrame(
    a0_data,
    index=df_val.index,
    columns=IOT_OBSERVABILITY_COLS,
    dtype=np.float32,
)

A1_OBSERVABILITY_MASKS_VAL = pd.DataFrame(
    a1_data,
    index=df_val.index,
    columns=IOT_OBSERVABILITY_COLS,
    dtype=np.float32,
)

if A0_OBSERVABILITY_MASKS_VAL.shape != (N_VAL, len(IOT_OBSERVABILITY_COLS)):
    raise RuntimeError(
        "[Cell12.f.1] A0_OBSERVABILITY_MASKS_VAL shape mismatch: "
        f"got={A0_OBSERVABILITY_MASKS_VAL.shape} expected={(N_VAL, len(IOT_OBSERVABILITY_COLS))}"
    )

if A1_OBSERVABILITY_MASKS_VAL.shape != (N_VAL, len(IOT_OBSERVABILITY_COLS)):
    raise RuntimeError(
        "[Cell12.f.1] A1_OBSERVABILITY_MASKS_VAL shape mismatch: "
        f"got={A1_OBSERVABILITY_MASKS_VAL.shape} expected={(N_VAL, len(IOT_OBSERVABILITY_COLS))}"
    )

# Binary domain validation.
for frame_name, frame in [
    ("A0_OBSERVABILITY_MASKS_VAL", A0_OBSERVABILITY_MASKS_VAL),
    ("A1_OBSERVABILITY_MASKS_VAL", A1_OBSERVABILITY_MASKS_VAL),
]:
    for col in IOT_OBSERVABILITY_COLS:
        arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(arr)
        bad = int(np.sum(finite & ~((arr == 0.0) | (arr == 1.0))))
        if bad:
            raise RuntimeError(
                f"[Cell12.f.1] Non-binary finite values in {frame_name}.{col}: {bad}"
            )

metrics_df = pd.DataFrame(metrics_rows)
audit_df = pd.DataFrame(audit_rows)

expected_metric_rows = int(len(IOT_OBSERVABILITY_COLS) * 2)
if len(metrics_df) != expected_metric_rows:
    raise RuntimeError(
        f"[Cell12.f.1] Metrics row count mismatch: got={len(metrics_df)} expected={expected_metric_rows}"
    )

if len(audit_df) != len(IOT_OBSERVABILITY_COLS):
    raise RuntimeError(
        f"[Cell12.f.1] Audit row count mismatch: got={len(audit_df)} expected={len(IOT_OBSERVABILITY_COLS)}"
    )

# ----------------------------------------------------------
# 6) Save artifacts
# ----------------------------------------------------------
a0_val_path = os.path.join(OUT_SYN, "A0_OBSERVABILITY_MASKS_VAL.parquet")
a1_val_path = os.path.join(OUT_SYN, "A1_OBSERVABILITY_MASKS_VAL.parquet")
metrics_csv = os.path.join(REPORT_DIR, "cell12f1_mask_baseline_metrics.csv")
audit_csv = os.path.join(REPORT_DIR, "cell12f1_mask_baseline_audit.csv")
contract_json = os.path.join(REPORT_DIR, "cell12f1_mask_baseline_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12f1_mask_baseline_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell12f1_mask_baseline_manifest.json")

A0_OBSERVABILITY_MASKS_VAL.to_parquet(a0_val_path, index=True)
A1_OBSERVABILITY_MASKS_VAL.to_parquet(a1_val_path, index=True)
metrics_df.to_csv(metrics_csv, index=False)
audit_df.to_csv(audit_csv, index=False)

a1_metrics = metrics_df[metrics_df["candidate_family"].astype(str).eq(A1_MASK_FAMILY_12F1)].copy()

fit_method_counts = (
    audit_df["a1_fit_method"].astype(str).value_counts().sort_index().to_dict()
)

metric_summary = {
    "observability_targets_total": int(len(IOT_OBSERVABILITY_COLS)),
    "a1_mean_obs_rate_error": float(pd.to_numeric(a1_metrics["obs_rate_error"], errors="coerce").mean()),
    "a1_mean_p11_error": float(pd.to_numeric(a1_metrics["p11_error"], errors="coerce").mean()),
    "a1_mean_p00_error": float(pd.to_numeric(a1_metrics["p00_error"], errors="coerce").mean()),
    "a1_mean_run_length_ks": float(pd.to_numeric(a1_metrics["run_length_ks"], errors="coerce").mean()),
    "a1_mean_score": float(pd.to_numeric(a1_metrics["score"], errors="coerce").mean()),
}

contract = {
    "cell": "12.f.1",
    "version": CELL12F1_VERSION,
    "role": "Q3_observability_A0_A1_mask_baseline",
    "quality_dimension": "Q3_observability",
    "observability_targets_total": int(len(IOT_OBSERVABILITY_COLS)),
    "train_rows": int(N_TR),
    "val_rows": int(N_VAL),
    "test_rows_schema_only": int(N_TE),
    "a0_family": A0_MASK_FAMILY_12F1,
    "a0_diagnostic_only": True,
    "a0_selectable": False,
    "a1_family": A1_MASK_FAMILY_12F1,
    "fit_method_counts": fit_method_counts,
    "obs_kind_counts": audit_df["obs_kind"].astype(str).value_counts().sort_index().to_dict(),
    "metric_summary": metric_summary,
    "TEST_real_values_used": False,
    "A0_real_VAL_used_for_diagnostic_only": True,
    "test_length_materialization_done_here": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "A1_baseline_done_here": True,
    "train_values_used_for_fitting": True,
    "val_values_used_for_evaluation": True,
    "df_te_used_for_index_length_schema_only": True,
    "cell12f0_contract_version_seen": version0,
    "cell12f0_generated_target_direct_leak_n_seen": int(contract0.get("generated_target_direct_leak_n", -1)),
    "A0_artifact_policy": "A0_OBSERVABILITY_MASKS_VAL stores real VAL masks only as a diagnostic upper bound; it is not selectable, not materializable, and must not feed 12.f.2/12.f.3 selection.",
    "direct_target_policy": "Only true iot__ observability/mask/stale columns are direct Q3 mask targets; generated 12.c/12.d/12.e target columns are excluded from direct mask baselines.",
    "outputs": {
        "a0_val_path": a0_val_path,
        "a1_val_path": a1_val_path,
        "metrics_csv": metrics_csv,
        "audit_csv": audit_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_12f1(contract_json, contract)
_write_json_12f1(contract_canonical_json, contract)

manifest = {
    "cell": "12.f.1",
    "version": CELL12F1_VERSION,
    "created_outputs": contract["outputs"],
    "observability_targets_total": int(len(IOT_OBSERVABILITY_COLS)),
    "fit_method_counts": fit_method_counts,
    "obs_kind_counts": audit_df["obs_kind"].astype(str).value_counts().sort_index().to_dict(),
    "metric_summary": metric_summary,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "A0_real_VAL_used_for_diagnostic_only": True,
        "A0_selectable": False,
        "test_length_materialization_done_here": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": False,
    },
}

_write_json_12f1(manifest_json, manifest)

hashes = {
    "a0_val_sha256": _sha256_file_12f1(a0_val_path),
    "a1_val_sha256": _sha256_file_12f1(a1_val_path),
    "metrics_csv_sha256": _sha256_file_12f1(metrics_csv),
    "audit_csv_sha256": _sha256_file_12f1(audit_csv),
    "contract_json_sha256": _sha256_file_12f1(contract_json),
    "contract_canonical_json_sha256": _sha256_file_12f1(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_12f1(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12f1(contract_json, contract)
_write_json_12f1(contract_canonical_json, contract)
_write_json_12f1(manifest_json, manifest)

# ----------------------------------------------------------
# 7) Export globals for 12.f.2+
# ----------------------------------------------------------
globals()["CELL12F1_VERSION"] = CELL12F1_VERSION
globals()["A0_OBSERVABILITY_MASKS_VAL"] = A0_OBSERVABILITY_MASKS_VAL
globals()["A1_OBSERVABILITY_MASKS_VAL"] = A1_OBSERVABILITY_MASKS_VAL
globals()["CELL12F1_MASK_BASELINE_METRICS_DF"] = metrics_df
globals()["CELL12F1_MASK_BASELINE_AUDIT_DF"] = audit_df
globals()["CELL12F1_MASK_BASELINE_CONTRACT"] = contract
globals()["CELL12F1_A0_OBSERVABILITY_MASKS_VAL_PATH"] = a0_val_path
globals()["CELL12F1_A1_OBSERVABILITY_MASKS_VAL_PATH"] = a1_val_path
globals()["CELL12F1_MASK_BASELINE_METRICS_CSV"] = metrics_csv
globals()["CELL12F1_MASK_BASELINE_AUDIT_CSV"] = audit_csv
globals()["CELL12F1_MASK_BASELINE_CONTRACT_JSON"] = contract_json
globals()["CELL12F1_MASK_BASELINE_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL12F1_MASK_BASELINE_MANIFEST_JSON"] = manifest_json

log(
    "[Cell12.f.1] Mask baseline complete | "
    f"targets={len(IOT_OBSERVABILITY_COLS)} | "
    f"A0_shape={A0_OBSERVABILITY_MASKS_VAL.shape} | "
    f"A1_shape={A1_OBSERVABILITY_MASKS_VAL.shape}"
)
log(f"[Cell12.f.1] A1 fit method counts | {fit_method_counts}")
log(
    "[Cell12.f.1] A1 metric summary | "
    f"mean_obs_rate_error={metric_summary['a1_mean_obs_rate_error']:.6f} | "
    f"mean_p11_error={metric_summary['a1_mean_p11_error']:.6f} | "
    f"mean_p00_error={metric_summary['a1_mean_p00_error']:.6f} | "
    f"mean_run_length_ks={metric_summary['a1_mean_run_length_ks']:.6f}"
)
log(f"[Cell12.f.1] Saved A0 diagnostic VAL masks: {a0_val_path}")
log(f"[Cell12.f.1] Saved A1 VAL masks: {a1_val_path}")
log(f"[Cell12.f.1] Saved metrics: {metrics_csv} | rows={len(metrics_df)}")
log(f"[Cell12.f.1] Saved audit: {audit_csv} | rows={len(audit_df)}")
log(f"[Cell12.f.1] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell12.f.1] Contract flags | "
    "TEST_real_values_used=False | "
    "A0_real_VAL_used_for_diagnostic_only=True | "
    "A0_selectable=False | "
    "test_length_materialization_done_here=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "df_te_used_for_index_length_schema_only=True"
)
log("--- END: Cell 12.f.1 - Observability mask baseline (v1.1 strict Q3, contract-hardened) ---")

gc.collect()