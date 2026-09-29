# %% CELL 12.f.V5.0 — Observability-mask V5 semantic setup
# Purpose:
#   Development-only V5 branch for Q3 observability masks.
#   V5 keeps V3/V4's correct semantic interpretation:
#       indicator columns are parsed as actual 0/1 states, not pd.notna(column).
#
# Key V5 idea:
#   Use TRAIN→VAL backtesting to choose mask materializers and claim scope.
#   TEST is used exactly once at the terminal V5.4 QA cell.
#
# Safety:
#   - Does not read TEST values except length/schema.
#   - Does not use V2 all-observed artifacts.
#   - Does not use V3/V4 TEST failures for per-column selection.
#   - Same-df_te V5 is development-only unless later frozen and evaluated on a fresh holdout.

import os
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = None


def _f5_log(msg):
    print(f"[Cell12.f.V5] {msg}")


def _f5_path(x):
    if x is None:
        return None
    try:
        return Path(str(x)).expanduser().resolve()
    except Exception:
        return None


OUTDIR_P = _f5_path(globals().get("OUTDIR", None)) or _f5_path(os.environ.get("CPS_OUTDIR", ""))
if OUTDIR_P is None:
    raise RuntimeError("[12.f.V5.0] OUTDIR is unresolved.")

OUT_SYN_P = _f5_path(globals().get("OUT_SYN", None)) or OUTDIR_P / "synthetic"
REPORT_DIR_P = _f5_path(globals().get("REPORT_DIR", None)) or OUTDIR_P / "reports"
CONTRACT_DIR_P = _f5_path(globals().get("CONTRACT_DIR", None)) or OUTDIR_P / "artifacts" / "contracts"
ARTIFACT_DIR_P = _f5_path(globals().get("ARTDIR", None)) or _f5_path(globals().get("ARTIFACT_DIR", None)) or OUTDIR_P / "artifacts"

for p in [OUT_SYN_P, REPORT_DIR_P, CONTRACT_DIR_P, ARTIFACT_DIR_P]:
    p.mkdir(parents=True, exist_ok=True)


def _require_df_12f5(names, label):
    for name in names:
        obj = globals().get(name)
        if isinstance(obj, pd.DataFrame):
            return name, obj
    raise RuntimeError(f"[12.f.V5.0] Could not find {label} dataframe. Tried {names}")


DF_TR_NAME, DF_TR = _require_df_12f5(["df_tr", "DF_TR", "train_df", "DF_TRAIN", "df_train"], "TRAIN")
DF_VA_NAME, DF_VA = _require_df_12f5(["df_va", "DF_VA", "val_df", "DF_VAL", "df_val"], "VAL")
DF_TE_NAME, DF_TE = _require_df_12f5(["df_te", "DF_TE", "test_df", "DF_TEST", "df_test"], "TEST length/schema")

globals()["DF_TR"] = DF_TR
globals()["DF_VA"] = DF_VA
globals()["DF_TE"] = DF_TE

N_TEST_12F5 = int(len(DF_TE))


def _pick_col_12f5(df, candidates):
    cols = list(df.columns)
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_q3_targets_12f5():
    target_contract = REPORT_DIR_P / "cell12f0_observability_target_contract.csv"
    if not target_contract.exists():
        raise RuntimeError(f"[12.f.V5.0] Missing target contract: {target_contract}")
    tdf = pd.read_csv(target_contract)
    c = _pick_col_12f5(tdf, ["column", "col", "target", "target_col", "mask_col"])
    if c is None:
        raise RuntimeError(f"[12.f.V5.0] Cannot find target column in {target_contract}. Columns={list(tdf.columns)}")
    targets = [str(x) for x in tdf[c].dropna().astype(str).tolist()]
    targets = [x for x in dict.fromkeys(targets) if x in DF_TR.columns or x in DF_VA.columns]
    return targets, str(target_contract)


MASK_TARGETS_12F5, MASK_TARGET_SOURCE_12F5 = _load_q3_targets_12f5()

_TRUE_12F5 = {
    "1", "true", "t", "yes", "y", "observed", "present", "active", "available",
    "online", "valid", "fresh"
}
_FALSE_12F5 = {
    "0", "false", "f", "no", "n", "missing", "absent", "inactive", "unavailable",
    "offline", "invalid", "stale", "unknown", "none"
}


def _is_indicator_name_12f5(col):
    name = str(col).lower()
    return any(tok in name for tok in [
        "__entity_obs__",
        "__entity_stale__",
        "__tier_present",
        "__any_update",
        "observability",
        "presence",
        "present",
        "stale",
        "_obs__",
    ])


def _binary_like_trainval_12f5(col):
    vals = []
    for df in [DF_TR, DF_VA]:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce").dropna()
        else:
            low = s.astype("string").str.lower().str.strip()
            mapped = low.map(lambda z: 1.0 if z in _TRUE_12F5 else 0.0 if z in _FALSE_12F5 else np.nan)
            num = pd.to_numeric(low, errors="coerce")
            x = mapped.fillna(num).dropna()
        if len(x):
            vals.extend(pd.Series(x).astype(float).unique().tolist())
        if len(vals) > 20:
            break
    if not vals:
        return False
    uniq = set(float(v) for v in vals if np.isfinite(float(v)))
    return len(uniq) > 0 and uniq.issubset({0.0, 1.0})


def _semantic_mode_12f5(col):
    if _is_indicator_name_12f5(col) or _binary_like_trainval_12f5(col):
        return "indicator_value_column"
    return "raw_value_missingness_column"


def _parse_indicator_state_12f5(series):
    if series is None:
        return np.asarray([], dtype=np.uint8)
    s = series
    if pd.api.types.is_bool_dtype(s):
        arr = s.astype("float64").to_numpy()
    elif pd.api.types.is_numeric_dtype(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    else:
        low = s.astype("string").str.lower().str.strip()
        mapped = low.map(lambda z: 1.0 if z in _TRUE_12F5 else 0.0 if z in _FALSE_12F5 else np.nan)
        numeric = pd.to_numeric(low, errors="coerce")
        arr = mapped.fillna(numeric).to_numpy(dtype="float64")
    # Indicator columns are state columns; unparseable values are conservatively state 0.
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return (arr > 0.5).astype(np.uint8)


def _state_values_12f5(df, col, mode):
    if df is None or col not in df.columns:
        return np.asarray([], dtype=np.uint8)
    if mode == "indicator_value_column":
        return _parse_indicator_state_12f5(df[col])
    return pd.notna(df[col]).to_numpy(dtype=np.uint8)


def _run_lengths_12f5(vals):
    vals = np.asarray(vals, dtype=np.uint8)
    n = len(vals)
    if n == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.uint8)
    changes = np.flatnonzero(np.diff(vals) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, n]
    return (ends - starts).astype(np.int64), vals[starts].astype(np.uint8)


def _q_12f5(x, q, default=np.nan):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return default
    return float(np.quantile(x, q))


def _stats_12f5(vals):
    vals = np.asarray(vals, dtype=np.uint8)
    n = len(vals)
    if n == 0:
        return {
            "n": 0, "state_one_rate": np.nan, "state_zero_rate": np.nan, "transition_rate": np.nan,
            "support": "empty", "run_count": 0, "one_run_count": 0, "zero_run_count": 0,
            "all_max": 0, "one_max": 0, "zero_max": 0,
            "all_p95": np.nan, "one_p95": np.nan, "zero_p95": np.nan,
        }
    p = float(vals.mean())
    trans = float(np.mean(vals[1:] != vals[:-1])) if n > 1 else 0.0
    lens, states = _run_lengths_12f5(vals)
    one_lens = lens[states == 1]
    zero_lens = lens[states == 0]
    if p == 0.0:
        support = "constant_0"
    elif p == 1.0:
        support = "constant_1"
    elif p < 0.001:
        support = "ultra_rare_1"
    elif p < 0.02:
        support = "rare_1"
    elif p > 0.999:
        support = "ultra_rare_0"
    elif p > 0.98:
        support = "rare_0"
    else:
        support = "dynamic"
    return {
        "n": int(n),
        "state_one_rate": p,
        "state_zero_rate": 1.0 - p,
        "transition_rate": trans,
        "support": support,
        "run_count": int(len(lens)),
        "one_run_count": int(len(one_lens)),
        "zero_run_count": int(len(zero_lens)),
        "all_max": int(lens.max()) if len(lens) else 0,
        "one_max": int(one_lens.max()) if len(one_lens) else 0,
        "zero_max": int(zero_lens.max()) if len(zero_lens) else 0,
        "all_p95": _q_12f5(lens, 0.95),
        "one_p95": _q_12f5(one_lens, 0.95),
        "zero_p95": _q_12f5(zero_lens, 0.95),
    }


def _seed_12f5(s):
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _tail_replay_12f5(vals, n):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(n, dtype=np.uint8)
    if len(vals) >= n:
        return vals[-n:].astype(np.uint8)
    reps = int(math.ceil(n / len(vals)))
    return np.tile(vals, reps)[-n:].astype(np.uint8)


def _head_replay_12f5(vals, n):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(n, dtype=np.uint8)
    if len(vals) >= n:
        return vals[:n].astype(np.uint8)
    reps = int(math.ceil(n / len(vals)))
    return np.tile(vals, reps)[:n].astype(np.uint8)


def _circular_replay_12f5(vals, n, seed):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(n, dtype=np.uint8)
    idx = (np.arange(n) + int(seed % len(vals))) % len(vals)
    return vals[idx].astype(np.uint8)


def _block_bootstrap_12f5(vals, n, seed, block=32768):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(n, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    block = int(max(128, min(block, len(vals))))
    chunks = []
    while sum(len(x) for x in chunks) < n:
        start = int(rng.integers(0, max(1, len(vals) - block + 1)))
        chunks.append(vals[start:start + block])
    return np.concatenate(chunks)[:n].astype(np.uint8)


def _run_bootstrap_12f5(vals, n, seed):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(n, dtype=np.uint8)
    lens, states = _run_lengths_12f5(vals)
    if len(lens) == 0:
        return np.zeros(n, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    out = []
    while sum(len(x) for x in out) < n:
        j = int(rng.integers(0, len(lens)))
        out.append(np.full(int(lens[j]), int(states[j]), dtype=np.uint8))
    return np.concatenate(out)[:n].astype(np.uint8)


def _rare_episode_bootstrap_12f5(vals, n, seed):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(n, dtype=np.uint8)
    p = float(vals.mean())
    if p == 0.0:
        return np.zeros(n, dtype=np.uint8)
    if p == 1.0:
        return np.ones(n, dtype=np.uint8)
    rare_state = 1 if p <= 0.5 else 0
    modal_state = 1 - rare_state
    lens, states = _run_lengths_12f5(vals)
    rare_lens = lens[states == rare_state]
    gap_lens = lens[states == modal_state]
    if len(rare_lens) == 0:
        return np.full(n, modal_state, dtype=np.uint8)
    if len(gap_lens) == 0:
        return np.full(n, rare_state, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    out = []
    state = modal_state if bool(rng.integers(0, 2)) else rare_state
    while sum(len(x) for x in out) < n:
        pool = gap_lens if state == modal_state else rare_lens
        L = int(pool[int(rng.integers(0, len(pool)))])
        out.append(np.full(max(1, L), state, dtype=np.uint8))
        state = 1 - state
    return np.concatenate(out)[:n].astype(np.uint8)


def _exact_rate_spread_12f5(vals, n, seed):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(n, dtype=np.uint8)
    p = float(vals.mean())
    k = int(round(p * n))
    out = np.zeros(n, dtype=np.uint8)
    if k <= 0:
        return out
    if k >= n:
        return np.ones(n, dtype=np.uint8)
    idx = np.linspace(0, n - 1, k).round().astype(int)
    out[np.unique(idx)] = 1
    if out.sum() < k:
        missing = k - int(out.sum())
        rng = np.random.default_rng(seed)
        zeros = np.flatnonzero(out == 0)
        fill = rng.choice(zeros, size=missing, replace=False)
        out[fill] = 1
    return out


def _boolish_12f5(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


MASK_CACHE_12F5 = {}
sem_rows = []
for col in MASK_TARGETS_12F5:
    mode = _semantic_mode_12f5(col)
    tr = _state_values_12f5(DF_TR, col, mode)
    va = _state_values_12f5(DF_VA, col, mode)
    tv = np.r_[tr, va].astype(np.uint8)
    tr_s = _stats_12f5(tr)
    va_s = _stats_12f5(va)
    tv_s = _stats_12f5(tv)
    MASK_CACHE_12F5[col] = {"mode": mode, "tr": tr, "va": va, "tv": tv, "tr_stats": tr_s, "va_stats": va_s, "tv_stats": tv_s}
    sem_rows.append({
        "col": col,
        "semantic_mode_v5": mode,
        "train_state_one_rate": tr_s["state_one_rate"],
        "val_state_one_rate": va_s["state_one_rate"],
        "train_support": tr_s["support"],
        "val_support": va_s["support"],
        "train_runs": tr_s["run_count"],
        "val_runs": va_s["run_count"],
        "TEST_real_values_used": False,
    })

semantics = pd.DataFrame(sem_rows)
sem_path = REPORT_DIR_P / "cell12f_v5_semantic_target_registry_trainval_only.csv"
semantics.to_csv(sem_path, index=False)

setup = {
    "cell": "12.f.V5.0",
    "role": "observability_mask_v5_semantic_setup",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "targets": int(len(MASK_TARGETS_12F5)),
    "target_source": MASK_TARGET_SOURCE_12F5,
    "semantic_mode_counts_trainval_only": semantics["semantic_mode_v5"].value_counts().to_dict(),
    "N_TEST_length_only": int(N_TEST_12F5),
    "TEST_real_values_used": False,
    "TEST_length_schema_only": True,
    "selection_done_here": False,
    "synthetic_values_mutated": False,
    "same_df_te_v5_status": "development_only_unless_fresh_holdout_used",
}
setup_path = CONTRACT_DIR_P / "cell12f_v5_setup_semantic_target_resolver_contract.json"
setup_path.write_text(json.dumps(setup, indent=2, sort_keys=True), encoding="utf-8")

CELL12F_V5_TARGETS = MASK_TARGETS_12F5
CELL12F_V5_MASK_CACHE = MASK_CACHE_12F5
CELL12F_V5_SETUP = setup

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}
CFG["cell12f_v5_active"] = True
CFG["cell12f_v5_targets"] = MASK_TARGETS_12F5
CFG["cell12f_v5_test_values_used"] = False
CFG["active_q3_branch"] = "12.f.V5"

_f5_log(f"V5 setup complete | targets={len(MASK_TARGETS_12F5)} | semantic_counts={setup['semantic_mode_counts_trainval_only']}")
_f5_log(f"Contract: {setup_path}")

if display is not None:
    display(semantics["semantic_mode_v5"].value_counts().rename_axis("semantic_mode_v5").reset_index(name="n"))
    display(semantics.head(20))
