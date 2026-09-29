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


def _bv6_log(msg):
    print(f"[Cell12.d.V6] {msg}")


def _bv6_path(x):
    if x is None:
        return None
    try:
        return Path(str(x)).expanduser().resolve()
    except Exception:
        return None


OUTDIR_P = _bv6_path(globals().get("OUTDIR", None)) or _bv6_path(os.environ.get("CPS_OUTDIR", ""))
if OUTDIR_P is None:
    raise RuntimeError("OUTDIR is unresolved. Run Binary V6 root bootstrap first.")

OUT_SYN_P = _bv6_path(globals().get("OUT_SYN", None)) or OUTDIR_P / "synthetic"
REPORT_DIR_P = _bv6_path(globals().get("REPORT_DIR", None)) or OUTDIR_P / "reports"
CONTRACT_DIR_P = _bv6_path(globals().get("CONTRACT_DIR", None)) or OUTDIR_P / "artifacts" / "contracts"
ARTIFACT_DIR_P = OUTDIR_P / "artifacts"

for p in [OUT_SYN_P, REPORT_DIR_P, CONTRACT_DIR_P, ARTIFACT_DIR_P]:
    p.mkdir(parents=True, exist_ok=True)

if "BINARY_V6" not in str(OUTDIR_P):
    raise RuntimeError(f"[Cell12.d.V6.0] OUTDIR is not a Binary V6 root: {OUTDIR_P}")


def _bv6_require_df(names, label):
    for name in names:
        obj = globals().get(name)
        if isinstance(obj, pd.DataFrame):
            return name, obj
    raise RuntimeError(f"Could not find {label} dataframe. Tried {names}")


DF_TR_NAME, DF_TR = _bv6_require_df(["df_tr", "DF_TR", "train_df", "DF_TRAIN", "df_train"], "TRAIN")
DF_VA_NAME, DF_VA = _bv6_require_df(["df_va", "DF_VA", "val_df", "DF_VAL", "df_val"], "VAL")
DF_TE_NAME, DF_TE = _bv6_require_df(["df_te", "DF_TE", "test_df", "DF_TEST", "df_test"], "TEST")

N_TEST_BV6 = int(len(DF_TE))


def _bv6_pick_col(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None


def _bv6_load_targets():
    for name in [
        "IOT_BINARY_TARGETS",
        "BINARY_TARGET_COLS",
        "CELL12D0_BINARY_TARGETS",
        "CELL12D0_BINARY_TARGET_COLS",
        "BINARY_TARGETS",
        "CELL12D_V5_TARGETS",
        "CELL12D_V4_TARGETS",
    ]:
        obj = globals().get(name)
        if isinstance(obj, (list, tuple, set)) and len(obj):
            return [str(x) for x in obj]

    json_paths = [
        OUTDIR_P / "artifacts" / "IOT_BINARY_TARGETS.json",
        OUTDIR_P / "reports" / "IOT_BINARY_TARGETS.json",
    ]
    for p in json_paths:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return [str(x) for x in obj]
            if isinstance(obj, dict):
                for key in ["targets", "binary_targets", "columns", "target_cols"]:
                    if isinstance(obj.get(key), list):
                        return [str(x) for x in obj[key]]

    csv_paths = [
        REPORT_DIR_P / "cell12d0_binary_target_contract.csv",
        REPORT_DIR_P / "cell12d_binary_target_contract.csv",
    ]
    for p in csv_paths:
        if p.exists():
            df = pd.read_csv(p)
            c = _bv6_pick_col(df, ["column", "col", "target_col", "target"])
            if c:
                return [str(x) for x in df[c].dropna().tolist()]

    raise RuntimeError("Could not resolve binary targets. Run Cell 12.d.0 first.")


BINARY_TARGETS_BV6 = [c for c in _bv6_load_targets() if c in DF_TR.columns or c in DF_VA.columns]
if not BINARY_TARGETS_BV6:
    raise RuntimeError("Resolved zero Binary V6 targets.")

_TRUE_BV6 = {
    "1", "true", "t", "yes", "y", "on", "open", "opened", "active",
    "present", "detected", "home", "playing", "cleaning", "charging",
    "connected", "available", "online", "motion", "vibration"
}
_FALSE_BV6 = {
    "0", "false", "f", "no", "n", "off", "closed", "inactive",
    "clear", "not_home", "idle", "paused", "stopped", "docked",
    "disconnected", "unavailable", "offline", "none", "unknown"
}


def _bv6_bin_values(s):
    if s is None:
        return np.asarray([], dtype=np.uint8), 0.0

    if pd.api.types.is_bool_dtype(s):
        arr = s.astype("float64").to_numpy()
    elif pd.api.types.is_numeric_dtype(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    else:
        lower = s.astype("string").str.lower().str.strip()
        mapped = lower.map(lambda x: 1.0 if x in _TRUE_BV6 else 0.0 if x in _FALSE_BV6 else np.nan)
        numeric = pd.to_numeric(lower, errors="coerce")
        arr = mapped.fillna(numeric).to_numpy(dtype="float64")

    finite = np.isfinite(arr)
    finite_frac = float(finite.mean()) if len(arr) else 0.0
    if finite.sum() == 0:
        return np.asarray([], dtype=np.uint8), finite_frac

    return (arr[finite] > 0.5).astype(np.uint8), finite_frac


def _bv6_run_lengths(vals):
    vals = np.asarray(vals, dtype=np.uint8)
    n = len(vals)
    if n == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.uint8)
    changes = np.flatnonzero(np.diff(vals) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, n]
    lens = (ends - starts).astype(np.int64)
    states = vals[starts].astype(np.uint8)
    return lens, states


def _bv6_q(x, q, default=np.nan):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return default
    return float(np.quantile(x, q))


def _bv6_stats(vals):
    vals = np.asarray(vals, dtype=np.uint8)
    n = len(vals)
    if n == 0:
        return {
            "n": 0, "rate": np.nan, "transition_rate": np.nan,
            "support": "empty",
            "run_count": 0, "one_count": 0, "zero_count": 0,
            "one_run_count": 0, "zero_run_count": 0,
            "all_max": 0, "one_max": 0, "zero_max": 0,
            "all_p95": np.nan, "one_p95": np.nan, "zero_p95": np.nan,
            "all_p99": np.nan, "one_p99": np.nan, "zero_p99": np.nan,
        }

    rate = float(vals.mean())
    trans = float(np.mean(vals[1:] != vals[:-1])) if n > 1 else 0.0
    lens, states = _bv6_run_lengths(vals)
    one_lens = lens[states == 1]
    zero_lens = lens[states == 0]

    if rate == 0:
        support = "constant_0"
    elif rate == 1:
        support = "constant_1"
    elif rate < 0.001:
        support = "ultra_rare_1"
    elif rate < 0.02:
        support = "rare_1"
    elif rate > 0.999:
        support = "ultra_rare_0"
    elif rate > 0.98:
        support = "rare_0"
    else:
        support = "both_states"

    return {
        "n": int(n),
        "rate": rate,
        "transition_rate": trans,
        "support": support,
        "run_count": int(len(lens)),
        "one_count": int(vals.sum()),
        "zero_count": int(n - vals.sum()),
        "one_run_count": int(len(one_lens)),
        "zero_run_count": int(len(zero_lens)),
        "all_max": int(lens.max()) if len(lens) else 0,
        "one_max": int(one_lens.max()) if len(one_lens) else 0,
        "zero_max": int(zero_lens.max()) if len(zero_lens) else 0,
        "all_p95": _bv6_q(lens, 0.95),
        "one_p95": _bv6_q(one_lens, 0.95),
        "zero_p95": _bv6_q(zero_lens, 0.95),
        "all_p99": _bv6_q(lens, 0.99),
        "one_p99": _bv6_q(one_lens, 0.99),
        "zero_p99": _bv6_q(zero_lens, 0.99),
    }


def _bv6_seed_for_col(col, salt="bv6"):
    h = hashlib.sha256((str(salt) + "::" + str(col)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _bv6_tail_replay(vals, N):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(N, dtype=np.uint8)
    if len(vals) >= N:
        return vals[-N:].astype(np.uint8)
    reps = int(math.ceil(N / len(vals)))
    return np.tile(vals, reps)[-N:].astype(np.uint8)


def _bv6_circular_replay(vals, N, seed):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(N, dtype=np.uint8)
    idx = (np.arange(N) + int(seed % len(vals))) % len(vals)
    return vals[idx].astype(np.uint8)


def _bv6_block_bootstrap(vals, N, seed, block=32768):
    vals = np.asarray(vals, dtype=np.uint8)
    if len(vals) == 0:
        return np.zeros(N, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    block = int(max(128, min(block, len(vals))))
    chunks = []
    while sum(len(x) for x in chunks) < N:
        start = int(rng.integers(0, max(1, len(vals) - block + 1)))
        chunks.append(vals[start:start + block])
    return np.concatenate(chunks)[:N].astype(np.uint8)


def _bv6_ks(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0
    a = np.sort(a)
    b = np.sort(b)
    x = np.unique(np.r_[a, b])
    ca = np.searchsorted(a, x, side="right") / len(a)
    cb = np.searchsorted(b, x, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _bv6_wasserstein(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0:
        return float(np.mean(np.abs(b)))
    if len(b) == 0:
        return float(np.mean(np.abs(a)))
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        grid = np.linspace(0, 1, 101)
        return float(np.mean(np.abs(np.quantile(a, grid) - np.quantile(b, grid))))


def _bv6_state_run_metrics(real, syn, state=None):
    rl, rs = _bv6_run_lengths(real)
    sl, ss = _bv6_run_lengths(syn)
    if state is not None:
        rl = rl[rs == state]
        sl = sl[ss == state]
    return {
        "ks": _bv6_ks(rl, sl),
        "wasserstein": _bv6_wasserstein(rl, sl),
        "real_count": int(len(rl)),
        "syn_count": int(len(sl)),
        "real_max": int(rl.max()) if len(rl) else 0,
        "syn_max": int(sl.max()) if len(sl) else 0,
    }


def _bv6_window_features(vals, window=1024):
    vals = np.asarray(vals, dtype=np.uint8)
    n = len(vals)
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    m = max(1, n // window)
    vals = vals[:m * window]
    chunks = vals.reshape(m, window)
    rates = chunks.mean(axis=1)
    trans = np.mean(chunks[:, 1:] != chunks[:, :-1], axis=1) if window > 1 else np.zeros(m)
    longest = []
    for row in chunks:
        lens, _ = _bv6_run_lengths(row)
        longest.append(float(lens.max() / len(row)) if len(lens) else 1.0)
    return np.c_[rates, trans, np.asarray(longest)]


def _bv6_c2st_auc(real, syn):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        Xr = _bv6_window_features(real)
        Xs = _bv6_window_features(syn)
        n = min(len(Xr), len(Xs))
        if n < 8:
            return 0.5
        X = np.vstack([Xr[:n], Xs[:n]])
        y = np.r_[np.zeros(n), np.ones(n)]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.35, random_state=13, stratify=y)
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(Xtr, ytr)
        scores = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, scores))
        return max(auc, 1.0 - auc)
    except Exception:
        Xr = _bv6_window_features(real)
        Xs = _bv6_window_features(syn)
        if len(Xr) == 0 or len(Xs) == 0:
            return 0.5
        dist = float(np.linalg.norm(np.nanmean(Xr, axis=0) - np.nanmean(Xs, axis=0)))
        return float(min(1.0, 0.5 + dist))


# Build TRAIN/VAL cache.
BV6_TRAINVAL_CACHE = {}
for col in BINARY_TARGETS_BV6:
    tr_vals, tr_finite = _bv6_bin_values(DF_TR[col]) if col in DF_TR.columns else (np.asarray([], dtype=np.uint8), 0.0)
    va_vals, va_finite = _bv6_bin_values(DF_VA[col]) if col in DF_VA.columns else (np.asarray([], dtype=np.uint8), 0.0)
    tv_vals = np.r_[tr_vals, va_vals].astype(np.uint8)

    tr_stats = _bv6_stats(tr_vals)
    va_stats = _bv6_stats(va_vals)
    tv_stats = _bv6_stats(tv_vals)

    BV6_TRAINVAL_CACHE[col] = {
        "tr_vals": tr_vals,
        "va_vals": va_vals,
        "tv_vals": tv_vals,
        "tr_stats": tr_stats,
        "va_stats": va_stats,
        "tv_stats": tv_stats,
    }

setup_contract = {
    "cell": "12.d.V6.0",
    "role": "binary_v6_setup_firewall_utilities",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "n_targets": int(len(BINARY_TARGETS_BV6)),
    "n_train": int(len(DF_TR)),
    "n_val": int(len(DF_VA)),
    "n_test_length_only": int(N_TEST_BV6),
    "TEST_real_values_used": False,
    "TEST_length_schema_only": True,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "synthetic_values_mutated": False,
    "class_level_rules_from_v6_audit": True,
    "per_column_TEST_winners_used": False,
    "development_only_on_current_df_te": True,
}
setup_path = CONTRACT_DIR_P / "cell12d_v6_setup_firewall_contract_v1_0_THESIS.json"
setup_path.write_text(json.dumps(setup_contract, indent=2, sort_keys=True), encoding="utf-8")

CELL12D_V6_TARGETS = BINARY_TARGETS_BV6
CELL12D_V6_TRAINVAL_CACHE = BV6_TRAINVAL_CACHE
CELL12D_V6_SETUP_CONTRACT_PATH = str(setup_path)

_bv6_log(f"V6 setup complete | targets={len(BINARY_TARGETS_BV6)} | train={len(DF_TR)} | val={len(DF_VA)} | test_length={N_TEST_BV6}")
_bv6_log(f"Contract: {setup_path}")