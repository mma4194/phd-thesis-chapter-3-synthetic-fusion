C2ST_FALLBACK_EVENTS_174 = []
# ==========================================================
# CELL 17.4 - External baseline metric computation
# v1.2 STUDY-THESIS strict fair-scope baseline evaluation, temporal-blocked C2ST, no-Q4-promotion aware
#
# Role:
#   - Compute fair metrics for external baselines produced by Cells 17.1–17.3.
#   - Compare baselines only within scopes defined by Cell 17.0.
#   - Include same-scope pipeline comparator artifacts where available.
#
# Fairness rules:
#   - CTGAN / TabDDPM are evaluated only on tabular scopes:
#       protocol_core_tabular
#       public_candidate_core_tabular
#       iot_continuous_compact_tabular
#   - TimeGAN is evaluated only on temporal scopes:
#       sequence_core_temporal
#       protocol_sequence_temporal
#   - Metrics are scope-specific.
#   - Do NOT compare baselines against full role-aware pipeline globally.
#   - Do NOT score baselines on Q3 observability, Q4 coupling repair, or Q6 release safety.
#
# Outputs:
#   reports/cell17_4_baseline_metric_by_artifact.csv
#   reports/cell17_4_baseline_metric_by_column.csv
#   reports/cell17_4_baseline_scope_summary.csv
#   reports/cell17_4_baseline_comparison_ledger.csv
#   reports/cell17_4_baseline_metric_contract.json
#   artifacts/cell17_4_baseline_metric_manifest.json
# ==========================================================

log("--- START: Cell 17.4 - External baseline metric computation (v1.2 temporal-blocked-C2ST no-Q4-promotion strict) ---")

import os
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_174 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL17_0_BASELINE_SCOPE_REGISTRY_DF",
    "CELL17_0_BASELINE_COLUMN_REGISTRY_DF",
    "CELL17_0_BASELINE_FAIRNESS_CONTRACT",
    "CELL17_0_BASELINE_PLAN",
    "CELL17_1_CTGAN_ARTIFACT_REGISTRY_DF",
    "CELL17_1_CTGAN_BASELINE_CONTRACT",
    "CELL17_2_TABDDPM_ARTIFACT_REGISTRY_DF",
    "CELL17_2_TABDDPM_BASELINE_CONTRACT",
    "CELL17_3_TIMEGAN_ARTIFACT_REGISTRY_DF",
    "CELL17_3_TIMEGAN_BASELINE_CONTRACT",
    "CELL16_0_SCIENTIFIC_NO_Q4_PROTOCOL_PATH",
    "CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH",
    "CELL16_0_PUBLIC_Q6_CPS_PATH",
]
_missing_174 = [k for k in _required_174 if k not in globals()]
if _missing_174:
    raise RuntimeError(f"[Cell17.4] Missing required globals: {_missing_174}")

ORIGINAL_OUTDIR_174 = str(OUTDIR)
ORIGINAL_OUT_SYN_174 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_174 = str(REPORT_DIR)

def _resolve_project_root_174(outdir, report_dir, out_syn):
    """
    Final STUDY-safe resolver.

    Do not fall back to old FULL_COUPLED roots. Cell 17.4 must write and read
    from the active clean root only.
    """
    candidates = []

    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue

        pp = os.path.abspath(str(p))
        base = os.path.basename(pp)

        if base in {"reports", "synthetic", "artifacts"}:
            candidates.append(os.path.dirname(pp))
        elif base == "contracts":
            candidates.append(os.path.dirname(os.path.dirname(pp)))
        else:
            candidates.append(pp)

    seen = set()

    for c in candidates:
        c = os.path.abspath(str(c))

        if c in seen:
            continue

        seen.add(c)

        if (
            os.path.isdir(os.path.join(c, "reports"))
            and os.path.isdir(os.path.join(c, "synthetic"))
        ):
            # Hard guard against accidental old-root fallback.
            if (
                "FULL_COUPLED" in c
                and "STUDY_FINAL" not in c
                and "BINARY_V6_Q3V5_STUDY_FINAL" not in c
            ):
                raise RuntimeError(
                    "[Cell17.4] Refusing old FULL_COUPLED root for final baseline metrics: "
                    + c
                )

            return c

    raise RuntimeError(
        "[Cell17.4] Could not resolve active clean project root from current OUTDIR/REPORT_DIR/OUT_SYN. "
        "Do not fall back to old FULL_COUPLED roots."
    )

PROJECT_ROOT_174 = _resolve_project_root_174(
    ORIGINAL_OUTDIR_174,
    ORIGINAL_REPORT_DIR_174,
    ORIGINAL_OUT_SYN_174,
)

OUT_SYN_BASE_174 = os.path.join(PROJECT_ROOT_174, "synthetic")
REPORT_DIR_BASE_174 = os.path.join(PROJECT_ROOT_174, "reports")
ARTDIR_BASE_174 = os.path.join(PROJECT_ROOT_174, "artifacts")
CONTRACT_DIR_BASE_174 = os.path.join(ARTDIR_BASE_174, "contracts")
BASELINE_DIR = os.path.join(OUT_SYN_BASE_174, "baselines")

os.makedirs(REPORT_DIR_BASE_174, exist_ok=True)
os.makedirs(ARTDIR_BASE_174, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_174, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)

os.makedirs(REPORT_DIR_BASE_174, exist_ok=True)
os.makedirs(ARTDIR_BASE_174, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_174, exist_ok=True)

SEED = int(SEED)
N_TE = int(len(df_te))

CELL174_VERSION = "cell17_4_external_baseline_metric_computation_v1_2_temporal_blocked_c2st_no_q4_promotion"

CFG["cell17_4_version"] = CELL174_VERSION
CFG["cell17_4_Q4_final_status"] = "blocked_no_promotion"
CFG["cell17_4_Q4_coupled_artifacts_used"] = False
CFG["cell17_4_TEST_real_values_used_for_reference_only"] = True
CFG["cell17_4_TEST_real_values_used_for_materialization"] = False
CFG["cell17_4_synthetic_values_mutated"] = False
CFG["cell17_4_selection_done_here"] = False
CFG["cell17_4_generator_fit_done_here"] = False
CFG["cell17_4_materialization_done_here"] = False
CFG["cell17_4_metric_computation_done_here"] = True
CFG["cell17_4_c2st_split_policy"] = "temporal_blocked_same_positions"

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell17_4_hist_bins", 50)
CFG.setdefault("cell17_4_low_card_unique_threshold", 30)
CFG.setdefault("cell17_4_lags", [1, 5, 10, 30, 60])
CFG.setdefault("cell17_4_activity_eps", 1e-12)
CFG.setdefault("cell17_4_correlation_min_cols", 3)
CFG.setdefault("cell17_4_max_rows_for_c2st", 8000)
CFG.setdefault("cell17_4_c2st_random_seed", SEED + 174)
CFG.setdefault("cell17_4_pipeline_prefer_public_q6", True)
CFG.setdefault("cell17_4_c2st_split_policy", "temporal_blocked_same_positions")
CFG.setdefault("cell17_4_c2st_train_fraction", 0.65)

HIST_BINS_174 = int(CFG.get("cell17_4_hist_bins", 50))
LOW_CARD_UNIQUE_174 = int(CFG.get("cell17_4_low_card_unique_threshold", 30))
LAGS_174 = [int(x) for x in CFG.get("cell17_4_lags", [1, 5, 10, 30, 60])]
ACTIVITY_EPS_174 = float(CFG.get("cell17_4_activity_eps", 1e-12))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
artifact_metrics_csv = os.path.join(REPORT_DIR_BASE_174, "cell17_4_baseline_metric_by_artifact.csv")
column_metrics_csv = os.path.join(REPORT_DIR_BASE_174, "cell17_4_baseline_metric_by_column.csv")
scope_summary_csv = os.path.join(REPORT_DIR_BASE_174, "cell17_4_baseline_scope_summary.csv")
comparison_ledger_csv = os.path.join(REPORT_DIR_BASE_174, "cell17_4_baseline_comparison_ledger.csv")
contract_json = os.path.join(REPORT_DIR_BASE_174, "cell17_4_baseline_metric_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_174, "cell17_4_baseline_metric_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_174, "cell17_4_baseline_metric_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_174(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_174(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_174(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_174(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_174(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_174(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_174(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_174(obj.to_dict())
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

def _write_json_174(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_174(payload), f, indent=2, sort_keys=True)

def _sha256_file_174(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_174(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _boolish_174(x, default=False):
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

    if s in {"false", "0", "no", "n", "f", ""}:
        return False

    return bool(default)


def _require_baseline_contract_174(contract: dict, name: str, expected_substrings):
    """
    Validate no-Q4-promotion baseline contracts.

    expected_substrings may be:
      - a string
      - a list/tuple of accepted version substrings

    This allows Cell 17.4 to accept both the original CTGAN v1.1 contract
    and the kernel-safe CTGAN 17.1R subprocess v1.2 contract.
    """
    if not isinstance(contract, dict):
        raise RuntimeError(f"[Cell17.4] {name} is not a dict.")

    version = str(contract.get("version", ""))

    if isinstance(expected_substrings, str):
        expected_substrings = [expected_substrings]

    ok_version = any(str(s) in version for s in expected_substrings)

    if not ok_version:
        raise RuntimeError(
            f"[Cell17.4] Unexpected {name} version: {version}. "
            f"Accepted substrings: {expected_substrings}"
        )

    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})

    final_q4_status = str(
        q4.get("final_q4_status", strict.get("Q4_final_status", ""))
    )

    if final_q4_status != "blocked_no_promotion":
        raise RuntimeError(
            f"[Cell17.4] {name} does not carry Q4 blocked_no_promotion governance. "
            f"Observed: {final_q4_status}"
        )

    q4_used = _boolish_174(
        q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True)),
        default=True,
    )

    if q4_used:
        raise RuntimeError(f"[Cell17.4] {name} indicates Q4-coupled artifacts were used.")

    return version

def _read_parquet_optional_174(path: str):
    if not _exists_174(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None

def _safe_float_174(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _finite_174(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]

def _num_arr_174(df: pd.DataFrame, col: str):
    if not isinstance(df, pd.DataFrame) or col not in df.columns:
        return np.asarray([], dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)

def _scope_cols_174(scope_id: str):
    reg = CELL17_0_BASELINE_COLUMN_REGISTRY_DF.copy()
    reg["scope_id"] = reg["scope_id"].astype(str)
    reg["col"] = reg["col"].astype(str)
    return reg.loc[reg["scope_id"].eq(str(scope_id)), "col"].drop_duplicates().tolist()

def _role_group_174(c: str) -> str:
    s = str(c)
    if s.startswith("router__"):
        return "protocol_router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__"):
        return "protocol_ota"
    if s.startswith("zigbee__") or s.startswith("zb__"):
        return "protocol_zigbee"
    if s.startswith("zwave__"):
        return "protocol_zwave"
    if s.startswith("events_in_sec__"):
        return "iot_event_drivers"
    if s.startswith("iot__") and (s.endswith("__state") or "__binary_sensor__" in s or "__switch__" in s):
        return "iot_binary_states"
    if s.startswith("iot__") and s.endswith("__value"):
        return "iot_continuous_values"
    if s.startswith("iot__"):
        return "iot_other"
    return "other"

def _ks_2samp_174(a, b):
    a = np.sort(_finite_174(a))
    b = np.sort(_finite_174(b))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    vals = np.sort(np.unique(np.concatenate([a, b])))
    if len(vals) == 0:
        return np.nan
    cdf_a = np.searchsorted(a, vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, vals, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))

def _hist_tv_174(a, b, bins=50):
    a = _finite_174(a)
    b = _finite_174(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan

    combined = np.concatenate([a, b])
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan

    if hi <= lo:
        av = float(np.nanmedian(a)) if len(a) else np.nan
        bv = float(np.nanmedian(b)) if len(b) else np.nan
        return 0.0 if np.isfinite(av) and np.isfinite(bv) and np.isclose(av, bv) else 1.0

    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    pa = ha / max(1, ha.sum())
    pb = hb / max(1, hb.sum())
    return float(0.5 * np.sum(np.abs(pa - pb)))

def _wasserstein_1d_174(a, b):
    a = np.sort(_finite_174(a))
    b = np.sort(_finite_174(b))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    q = np.linspace(0.0, 1.0, 101)
    aq = np.quantile(a, q)
    bq = np.quantile(b, q)
    return float(np.mean(np.abs(aq - bq)))

def _normalized_wasserstein_174(a, b):
    w = _wasserstein_1d_174(a, b)
    af = _finite_174(a)
    if len(af) == 0 or not np.isfinite(w):
        return np.nan
    scale = float(np.nanstd(af) + np.nanmean(np.abs(af)) + 1e-9)
    return float(w / max(scale, 1e-9))

def _support_jaccard_174(a, b):
    a = _finite_174(a)
    b = _finite_174(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan

    au = set(np.unique(np.round(a, 6)).tolist())
    bu = set(np.unique(np.round(b, 6)).tolist())
    if not au and not bu:
        return 1.0
    return float(len(au & bu) / max(1, len(au | bu)))

def _autocorr_174(x, lag: int):
    x = np.asarray(x, dtype=np.float64)
    lag = int(lag)
    if lag <= 0 or len(x) <= lag:
        return np.nan
    a = x[:-lag]
    b = x[lag:]
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 5:
        return np.nan
    aa = a[valid]
    bb = b[valid]
    if np.nanstd(aa) <= 1e-12 or np.nanstd(bb) <= 1e-12:
        return 1.0 if np.allclose(aa, bb, equal_nan=True) else 0.0
    return float(np.corrcoef(aa, bb)[0, 1])

def _transition_rate_174(x):
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.nan
    vals = x[idx]
    return float(np.mean(vals[1:] != vals[:-1]))

def _activity_rate_174(x):
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(x[valid]) > ACTIVITY_EPS_174))

def _run_lengths_174(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    vals = x[finite]
    if len(vals) == 0:
        return np.asarray([], dtype=np.float64)
    low_card = len(np.unique(vals)) <= LOW_CARD_UNIQUE_174

    if low_card:
        use = vals
    else:
        use = (np.abs(vals) > ACTIVITY_EPS_174).astype(float)

    runs = []
    cur = use[0]
    length = 1
    for v in use[1:]:
        if v == cur:
            length += 1
        else:
            runs.append(length)
            cur = v
            length = 1
    runs.append(length)
    return np.asarray(runs, dtype=np.float64)

def _run_length_ks_174(a, b):
    return _ks_2samp_174(_run_lengths_174(a), _run_lengths_174(b))

def _corr_mae_174(real_df: pd.DataFrame, syn_df: pd.DataFrame, cols):
    cols = [c for c in cols if c in real_df.columns and c in syn_df.columns]
    if len(cols) < int(CFG.get("cell17_4_correlation_min_cols", 3)):
        return np.nan
    r = real_df[cols].apply(pd.to_numeric, errors="coerce")
    s = syn_df[cols].apply(pd.to_numeric, errors="coerce")
    rc = r.corr().to_numpy(dtype=np.float64)
    sc = s.corr().to_numpy(dtype=np.float64)
    mask = np.isfinite(rc) & np.isfinite(sc)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(rc[mask] - sc[mask])))

def _temporal_block_indices_174(n_rows: int, train_fraction: float, max_total_rows: int):
    n_rows = int(n_rows)
    if n_rows < 100:
        return np.arange(0), np.arange(0)

    train_end = int(np.floor(n_rows * float(train_fraction)))
    train_end = max(50, min(train_end, n_rows - 50))

    train_idx = np.arange(0, train_end, dtype=np.int64)
    test_idx = np.arange(train_end, n_rows, dtype=np.int64)

    # Cap row count deterministically while preserving chronological coverage within each block.
    max_total_rows = int(max_total_rows)
    max_each = max(50, max_total_rows // 2)

    if len(train_idx) > max_each:
        train_idx = train_idx[np.linspace(0, len(train_idx) - 1, max_each).round().astype(int)]
    if len(test_idx) > max_each:
        test_idx = test_idx[np.linspace(0, len(test_idx) - 1, max_each).round().astype(int)]

    return train_idx, test_idx

def _c2st_auc_proxy_174(real_df: pd.DataFrame, syn_df: pd.DataFrame, cols):
    """
    Temporal-blocked C2ST:
      - No random row train_test_split.
      - Train discriminator on early temporal block from real/synthetic.
      - Test discriminator on later temporal block from real/synthetic.
      - Deterministic thinning only, preserving chronological coverage.
    """
    cols = [c for c in cols if c in real_df.columns and c in syn_df.columns]
    if len(cols) == 0:
        return np.nan

    n = min(len(real_df), len(syn_df))
    if n < 100:
        return np.nan

    max_rows = int(CFG.get("cell17_4_max_rows_for_c2st", 8000))
    train_fraction = float(CFG.get("cell17_4_c2st_train_fraction", 0.65))

    train_idx, test_idx = _temporal_block_indices_174(n, train_fraction, max_rows)
    if len(train_idx) < 50 or len(test_idx) < 50:
        return np.nan

    r = real_df.iloc[:n][cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = syn_df.iloc[:n][cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Median impute using real TRAIN block medians only.
    med = r.iloc[train_idx].median(numeric_only=True).fillna(0.0)
    r = r.fillna(med)
    s = s.fillna(med)

    X_train = np.vstack([
        r.iloc[train_idx].to_numpy(dtype=np.float64),
        s.iloc[train_idx].to_numpy(dtype=np.float64),
    ])
    y_train = np.concatenate([np.zeros(len(train_idx)), np.ones(len(train_idx))])

    X_test = np.vstack([
        r.iloc[test_idx].to_numpy(dtype=np.float64),
        s.iloc[test_idx].to_numpy(dtype=np.float64),
    ])
    y_test = np.concatenate([np.zeros(len(test_idx)), np.ones(len(test_idx))])

    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs")
        )
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, p))
        return float(max(auc, 1.0 - auc))
    except Exception as c2st_error:
        C2ST_FALLBACK_EVENTS_174.append({"columns": list(cols), "error": str(c2st_error)})
        # fallback: deterministic univariate rank AUC proxy on held-out temporal block only
        aucs = []
        for c in cols:
            rv = _finite_174(r.iloc[test_idx][c])
            sv = _finite_174(s.iloc[test_idx][c])
            if len(rv) == 0 or len(sv) == 0:
                continue
            vals = np.concatenate([rv, sv])
            labels = np.concatenate([np.zeros(len(rv)), np.ones(len(sv))])
            if np.nanmax(vals) <= np.nanmin(vals):
                aucs.append(0.5)
                continue
            order = np.argsort(vals)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(vals) + 1, dtype=np.float64)
            pos = ranks[labels == 1]
            npos = len(sv)
            nneg = len(rv)
            auc = (np.sum(pos) - npos * (npos + 1) / 2.0) / max(1.0, npos * nneg)
            aucs.append(max(float(auc), float(1.0 - auc)))
        return float(np.mean(aucs)) if aucs else np.nan


def _artifact_score_tabular_174(row):
    vals = [
        _safe_float_174(row.get("mean_ks"), np.nan),
        _safe_float_174(row.get("mean_hist_tv"), np.nan),
        _safe_float_174(row.get("mean_wasserstein_norm"), np.nan),
        _safe_float_174(row.get("correlation_mae"), np.nan),
        max(0.0, _safe_float_174(row.get("c2st_auc"), 0.5) - 0.5),
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan

def _artifact_score_temporal_174(row):
    vals = [
        _safe_float_174(row.get("mean_lag1_error"), np.nan),
        _safe_float_174(row.get("mean_multi_lag_mae"), np.nan),
        _safe_float_174(row.get("mean_transition_rate_error"), np.nan),
        _safe_float_174(row.get("mean_run_length_ks"), np.nan),
        _safe_float_174(row.get("mean_activity_rate_error"), np.nan),
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan

def _is_q4_coupled_path_174(path: str) -> bool:
    s = str(path).lower()
    return (
        "final_q4" in s
        or "q4_coupled" in s
        or "accepted" in s
        or "promoted" in s
        or "promotion" in s
    )


def _resolve_pipeline_comparator_path_174(scope_id: str):
    """
    Same-scope pipeline comparator.

    STUDY-THESIS strict rule:
      - Do not use Q4-coupled artifacts as baseline comparators.
      - Prefer scientific no-Q4 artifacts.
      - Public Q6 artifact is allowed only for public candidate scope.
    """
    scope_id = str(scope_id)

    candidates = []

    if scope_id in {"protocol_core_tabular", "protocol_sequence_temporal"}:
        candidates.extend([
            globals().get("CELL16_0_SCIENTIFIC_NO_Q4_PROTOCOL_PATH", ""),
            os.path.join(OUT_SYN, "PROTOCOL_SYN_TEST_SCIENTIFIC_NO_Q4.parquet"),
            os.path.join(OUT_SYN, "PROTOCOL_SYN_TEST_NO_Q4.parquet"),
            os.path.join(OUT_SYN, "A2_PROTOCOL_FINAL.parquet"),
        ])

    elif scope_id == "public_candidate_core_tabular":
        candidates.extend([
            globals().get("CELL16_0_PUBLIC_Q6_CPS_PATH", ""),
            os.path.join(OUT_SYN, "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet"),
            globals().get("CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH", ""),
        ])

    elif scope_id == "sequence_core_temporal":
        candidates.extend([
            globals().get("CELL16_0_PUBLIC_Q6_CPS_PATH", ""),
            os.path.join(OUT_SYN, "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet"),
            globals().get("CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH", ""),
        ])

    elif scope_id == "iot_continuous_compact_tabular":
        candidates.extend([
            globals().get("CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH", ""),
            os.path.join(OUT_SYN, "CPS_SYNTHETIC_TEST_SCIENTIFIC_NO_Q4.parquet"),
            os.path.join(OUT_SYN, "IOT_FULL_SYNTHETIC_TEST.parquet"),
            os.path.join(OUT_SYN, "IOT_FINAL_VALUES_TEST.parquet"),
        ])

    else:
        candidates.extend([
            globals().get("CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH", ""),
            globals().get("CELL16_0_PUBLIC_Q6_CPS_PATH", ""),
        ])

    for p in candidates:
        p = str(p)

        if not p:
            continue

        if _is_q4_coupled_path_174(p):
            continue

        if _exists_174(p):
            return p

    return ""

def _load_pipeline_comparator_174(scope_id: str):
    p = _resolve_pipeline_comparator_path_174(scope_id)
    if not p:
        return None, ""
    d = _read_parquet_optional_174(p)
    return d, p

# ----------------------------------------------------------
# 4) Validate upstream no-Q4 baseline contracts
# ----------------------------------------------------------
CELL17_0_VERSION_174 = _require_baseline_contract_174(
    CELL17_0_BASELINE_FAIRNESS_CONTRACT,
    "CELL17_0_BASELINE_FAIRNESS_CONTRACT",
    "cell17_0_external_baseline_scope_contract_v1_1",
)

CELL17_1_VERSION_174 = _require_baseline_contract_174(
    CELL17_1_CTGAN_BASELINE_CONTRACT,
    "CELL17_1_CTGAN_BASELINE_CONTRACT",
    [
        "cell17_1_ctgan_external_baseline_v1_1",
        "cell17_1R_ctgan_external_baseline_subprocess_v1_2_THESIS",
    ],
)

CELL17_2_VERSION_174 = _require_baseline_contract_174(
    CELL17_2_TABDDPM_BASELINE_CONTRACT,
    "CELL17_2_TABDDPM_BASELINE_CONTRACT",
    [
        "cell17_2_tabddpm_external_baseline_v1_1",
        "cell17_2R_tabddpm_external_baseline",
    ],
)

CELL17_3_VERSION_174 = _require_baseline_contract_174(
    CELL17_3_TIMEGAN_BASELINE_CONTRACT,
    "CELL17_3_TIMEGAN_BASELINE_CONTRACT",
    [
        "cell17_3_timegan_external_baseline_v1_1",
        "cell17_3R_timegan_external_baseline",
    ],
)

# ----------------------------------------------------------
# 5) Build artifact registry for baselines + same-scope pipeline comparators
# ----------------------------------------------------------
baseline_regs = []

for name, df in [
    ("CTGAN", CELL17_1_CTGAN_ARTIFACT_REGISTRY_DF),
    ("TabDDPM", CELL17_2_TABDDPM_ARTIFACT_REGISTRY_DF),
    ("TimeGAN", CELL17_3_TIMEGAN_ARTIFACT_REGISTRY_DF),
]:
    if isinstance(df, pd.DataFrame) and len(df):
        d = df.copy()
        d["baseline"] = name
        baseline_regs.append(d)

baseline_artifacts_df = pd.concat(baseline_regs, axis=0, ignore_index=True) if baseline_regs else pd.DataFrame()

artifact_rows = []
column_rows = []

real_test = df_te.copy()
real_test.columns = real_test.columns.astype(str)

log("[Cell17.4] Computing fair-scope baseline metrics")

# ----------------------------------------------------------
# 6) Evaluate successful external baselines
# ----------------------------------------------------------
for _, a in baseline_artifacts_df.iterrows():
    baseline = str(a.get("baseline", ""))
    scope_id = str(a.get("scope_id", ""))
    path = str(a.get("path", ""))
    exists = _boolish_174(a.get("exists", False), default=False) and _exists_174(path)
    artifact_status = str(a.get("status", ""))

    cols = _scope_cols_174(scope_id)
    metric_family = (
        "temporal" if scope_id in {"sequence_core_temporal", "protocol_sequence_temporal"} else "tabular"
    )

    base_row = {
        "artifact_type": "external_baseline",
        "baseline": baseline,
        "scope_id": scope_id,
        "metric_family": metric_family,
        "path": path,
        "artifact_available": bool(exists),
        "source_status": artifact_status,
        "cols_n_scope": int(len(cols)),
        "TEST_real_values_used_for_reference_only": True,
        "synthetic_values_mutated": False,
        "c2st_split_policy": "temporal_blocked_same_positions",
    }

    if not exists:
        artifact_rows.append({
            **base_row,
            "metric_status": "not_evaluable",
            "metric_reasons": str(a.get("reason", "artifact_unavailable")),
        })
        continue

    syn = _read_parquet_optional_174(path)
    if not isinstance(syn, pd.DataFrame):
        artifact_rows.append({
            **base_row,
            "metric_status": "not_evaluable",
            "metric_reasons": "artifact_read_failed",
        })
        continue

    syn.columns = syn.columns.astype(str)

    eval_cols = [c for c in cols if c in syn.columns and c in real_test.columns]
    if len(eval_cols) == 0:
        artifact_rows.append({
            **base_row,
            "metric_status": "not_evaluable",
            "metric_reasons": "no_common_scope_columns",
        })
        continue

    # Column-level metrics
    per_col = []
    for c in eval_cols:
        rv = _num_arr_174(real_test, c)
        sv = _num_arr_174(syn, c)

        col_row = {
            "artifact_type": "external_baseline",
            "baseline": baseline,
            "scope_id": scope_id,
            "metric_family": metric_family,
            "col": c,
            "role_group": _role_group_174(c),
            "real_finite_rate": float(np.isfinite(rv).mean()) if len(rv) else 0.0,
            "syn_finite_rate": float(np.isfinite(sv).mean()) if len(sv) else 0.0,
            "real_mean": float(np.nanmean(_finite_174(rv))) if len(_finite_174(rv)) else np.nan,
            "syn_mean": float(np.nanmean(_finite_174(sv))) if len(_finite_174(sv)) else np.nan,
            "real_std": float(np.nanstd(_finite_174(rv))) if len(_finite_174(rv)) else np.nan,
            "syn_std": float(np.nanstd(_finite_174(sv))) if len(_finite_174(sv)) else np.nan,
            "ks": _ks_2samp_174(rv, sv),
            "hist_tv": _hist_tv_174(rv, sv, bins=HIST_BINS_174),
            "wasserstein_norm": _normalized_wasserstein_174(rv, sv),
            "support_jaccard": _support_jaccard_174(rv, sv),
            "TEST_real_values_used_for_reference_only": True,
            "synthetic_values_mutated": False,
        }

        if metric_family == "temporal":
            ac_errors = []
            for lag in LAGS_174:
                rac = _autocorr_174(rv, lag)
                sac = _autocorr_174(sv, lag)
                col_row[f"real_lag{lag}_autocorr"] = rac
                col_row[f"syn_lag{lag}_autocorr"] = sac
                if np.isfinite(rac) and np.isfinite(sac):
                    ac_errors.append(abs(sac - rac))

            col_row["lag1_error"] = (
                abs(col_row.get("syn_lag1_autocorr", np.nan) - col_row.get("real_lag1_autocorr", np.nan))
                if np.isfinite(col_row.get("syn_lag1_autocorr", np.nan)) and np.isfinite(col_row.get("real_lag1_autocorr", np.nan))
                else np.nan
            )
            col_row["multi_lag_mae"] = float(np.mean(ac_errors)) if ac_errors else np.nan

            rt = _transition_rate_174(rv)
            st = _transition_rate_174(sv)
            col_row["real_transition_rate"] = rt
            col_row["syn_transition_rate"] = st
            col_row["transition_rate_error"] = abs(st - rt) if np.isfinite(st) and np.isfinite(rt) else np.nan

            ra = _activity_rate_174(rv)
            sa = _activity_rate_174(sv)
            col_row["real_activity_rate"] = ra
            col_row["syn_activity_rate"] = sa
            col_row["activity_rate_error"] = abs(sa - ra) if np.isfinite(sa) and np.isfinite(ra) else np.nan
            col_row["run_length_ks"] = _run_length_ks_174(rv, sv)

        per_col.append(col_row)
        column_rows.append(col_row)

    per_col_df = pd.DataFrame(per_col)

    if metric_family == "tabular":
        corr_mae = _corr_mae_174(real_test, syn, eval_cols)
        c2st_auc = _c2st_auc_proxy_174(real_test, syn, eval_cols)

        row = {
            **base_row,
            "metric_status": "evaluated",
            "metric_reasons": "fair_scope_tabular_metrics",
            "eval_cols_n": int(len(eval_cols)),
            "rows_syn": int(len(syn)),
            "mean_ks": float(pd.to_numeric(per_col_df["ks"], errors="coerce").mean()),
            "median_ks": float(pd.to_numeric(per_col_df["ks"], errors="coerce").median()),
            "mean_hist_tv": float(pd.to_numeric(per_col_df["hist_tv"], errors="coerce").mean()),
            "mean_wasserstein_norm": float(pd.to_numeric(per_col_df["wasserstein_norm"], errors="coerce").mean()),
            "mean_support_jaccard": float(pd.to_numeric(per_col_df["support_jaccard"], errors="coerce").mean()),
            "correlation_mae": corr_mae,
            "c2st_auc": c2st_auc,
        }
        row["fair_scope_penalty_score"] = _artifact_score_tabular_174(row)
        artifact_rows.append(row)

    else:
        row = {
            **base_row,
            "metric_status": "evaluated",
            "metric_reasons": "fair_scope_temporal_metrics",
            "eval_cols_n": int(len(eval_cols)),
            "rows_syn": int(len(syn)),
            "mean_ks": float(pd.to_numeric(per_col_df["ks"], errors="coerce").mean()),
            "mean_hist_tv": float(pd.to_numeric(per_col_df["hist_tv"], errors="coerce").mean()),
            "mean_wasserstein_norm": float(pd.to_numeric(per_col_df["wasserstein_norm"], errors="coerce").mean()),
            "mean_lag1_error": float(pd.to_numeric(per_col_df["lag1_error"], errors="coerce").mean()),
            "mean_multi_lag_mae": float(pd.to_numeric(per_col_df["multi_lag_mae"], errors="coerce").mean()),
            "mean_transition_rate_error": float(pd.to_numeric(per_col_df["transition_rate_error"], errors="coerce").mean()),
            "mean_run_length_ks": float(pd.to_numeric(per_col_df["run_length_ks"], errors="coerce").mean()),
            "mean_activity_rate_error": float(pd.to_numeric(per_col_df["activity_rate_error"], errors="coerce").mean()),
        }
        row["fair_scope_penalty_score"] = _artifact_score_temporal_174(row)
        artifact_rows.append(row)

    del syn
    gc.collect()

# ----------------------------------------------------------
# 7) Evaluate same-scope pipeline comparators
# ----------------------------------------------------------
for scope_id in CELL17_0_BASELINE_SCOPE_REGISTRY_DF["scope_id"].astype(str).tolist():
    cols = _scope_cols_174(scope_id)
    comparator_df, comparator_path = _load_pipeline_comparator_174(scope_id)

    metric_family = (
        "temporal" if scope_id in {"sequence_core_temporal", "protocol_sequence_temporal"} else "tabular"
    )

    base_row = {
        "artifact_type": "pipeline_same_scope_comparator",
        "baseline": "RoleAwarePipeline",
        "scope_id": scope_id,
        "metric_family": metric_family,
        "path": comparator_path,
        "artifact_available": bool(isinstance(comparator_df, pd.DataFrame)),
        "source_status": "comparator",
        "cols_n_scope": int(len(cols)),
        "TEST_real_values_used_for_reference_only": True,
        "synthetic_values_mutated": False,
        "c2st_split_policy": "temporal_blocked_same_positions",
    }

    if not isinstance(comparator_df, pd.DataFrame):
        artifact_rows.append({
            **base_row,
            "metric_status": "not_evaluable",
            "metric_reasons": "pipeline_comparator_unavailable",
        })
        continue

    comparator_df.columns = comparator_df.columns.astype(str)
    eval_cols = [c for c in cols if c in comparator_df.columns and c in real_test.columns]

    if len(eval_cols) == 0:
        artifact_rows.append({
            **base_row,
            "metric_status": "not_evaluable",
            "metric_reasons": "no_common_scope_columns",
        })
        continue

    per_col = []
    for c in eval_cols:
        rv = _num_arr_174(real_test, c)
        sv = _num_arr_174(comparator_df, c)

        col_row = {
            "artifact_type": "pipeline_same_scope_comparator",
            "baseline": "RoleAwarePipeline",
            "scope_id": scope_id,
            "metric_family": metric_family,
            "col": c,
            "role_group": _role_group_174(c),
            "real_finite_rate": float(np.isfinite(rv).mean()) if len(rv) else 0.0,
            "syn_finite_rate": float(np.isfinite(sv).mean()) if len(sv) else 0.0,
            "real_mean": float(np.nanmean(_finite_174(rv))) if len(_finite_174(rv)) else np.nan,
            "syn_mean": float(np.nanmean(_finite_174(sv))) if len(_finite_174(sv)) else np.nan,
            "real_std": float(np.nanstd(_finite_174(rv))) if len(_finite_174(rv)) else np.nan,
            "syn_std": float(np.nanstd(_finite_174(sv))) if len(_finite_174(sv)) else np.nan,
            "ks": _ks_2samp_174(rv, sv),
            "hist_tv": _hist_tv_174(rv, sv, bins=HIST_BINS_174),
            "wasserstein_norm": _normalized_wasserstein_174(rv, sv),
            "support_jaccard": _support_jaccard_174(rv, sv),
            "TEST_real_values_used_for_reference_only": True,
            "synthetic_values_mutated": False,
        }

        if metric_family == "temporal":
            ac_errors = []
            for lag in LAGS_174:
                rac = _autocorr_174(rv, lag)
                sac = _autocorr_174(sv, lag)
                col_row[f"real_lag{lag}_autocorr"] = rac
                col_row[f"syn_lag{lag}_autocorr"] = sac
                if np.isfinite(rac) and np.isfinite(sac):
                    ac_errors.append(abs(sac - rac))

            col_row["lag1_error"] = (
                abs(col_row.get("syn_lag1_autocorr", np.nan) - col_row.get("real_lag1_autocorr", np.nan))
                if np.isfinite(col_row.get("syn_lag1_autocorr", np.nan)) and np.isfinite(col_row.get("real_lag1_autocorr", np.nan))
                else np.nan
            )
            col_row["multi_lag_mae"] = float(np.mean(ac_errors)) if ac_errors else np.nan

            rt = _transition_rate_174(rv)
            st = _transition_rate_174(sv)
            col_row["real_transition_rate"] = rt
            col_row["syn_transition_rate"] = st
            col_row["transition_rate_error"] = abs(st - rt) if np.isfinite(st) and np.isfinite(rt) else np.nan

            ra = _activity_rate_174(rv)
            sa = _activity_rate_174(sv)
            col_row["real_activity_rate"] = ra
            col_row["syn_activity_rate"] = sa
            col_row["activity_rate_error"] = abs(sa - ra) if np.isfinite(sa) and np.isfinite(ra) else np.nan
            col_row["run_length_ks"] = _run_length_ks_174(rv, sv)

        per_col.append(col_row)
        column_rows.append(col_row)

    per_col_df = pd.DataFrame(per_col)

    if metric_family == "tabular":
        corr_mae = _corr_mae_174(real_test, comparator_df, eval_cols)
        c2st_auc = _c2st_auc_proxy_174(real_test, comparator_df, eval_cols)

        row = {
            **base_row,
            "metric_status": "evaluated",
            "metric_reasons": "same_scope_pipeline_tabular_comparator",
            "eval_cols_n": int(len(eval_cols)),
            "rows_syn": int(len(comparator_df)),
            "mean_ks": float(pd.to_numeric(per_col_df["ks"], errors="coerce").mean()),
            "median_ks": float(pd.to_numeric(per_col_df["ks"], errors="coerce").median()),
            "mean_hist_tv": float(pd.to_numeric(per_col_df["hist_tv"], errors="coerce").mean()),
            "mean_wasserstein_norm": float(pd.to_numeric(per_col_df["wasserstein_norm"], errors="coerce").mean()),
            "mean_support_jaccard": float(pd.to_numeric(per_col_df["support_jaccard"], errors="coerce").mean()),
            "correlation_mae": corr_mae,
            "c2st_auc": c2st_auc,
        }
        row["fair_scope_penalty_score"] = _artifact_score_tabular_174(row)
        artifact_rows.append(row)

    else:
        row = {
            **base_row,
            "metric_status": "evaluated",
            "metric_reasons": "same_scope_pipeline_temporal_comparator",
            "eval_cols_n": int(len(eval_cols)),
            "rows_syn": int(len(comparator_df)),
            "mean_ks": float(pd.to_numeric(per_col_df["ks"], errors="coerce").mean()),
            "mean_hist_tv": float(pd.to_numeric(per_col_df["hist_tv"], errors="coerce").mean()),
            "mean_wasserstein_norm": float(pd.to_numeric(per_col_df["wasserstein_norm"], errors="coerce").mean()),
            "mean_lag1_error": float(pd.to_numeric(per_col_df["lag1_error"], errors="coerce").mean()),
            "mean_multi_lag_mae": float(pd.to_numeric(per_col_df["multi_lag_mae"], errors="coerce").mean()),
            "mean_transition_rate_error": float(pd.to_numeric(per_col_df["transition_rate_error"], errors="coerce").mean()),
            "mean_run_length_ks": float(pd.to_numeric(per_col_df["run_length_ks"], errors="coerce").mean()),
            "mean_activity_rate_error": float(pd.to_numeric(per_col_df["activity_rate_error"], errors="coerce").mean()),
        }
        row["fair_scope_penalty_score"] = _artifact_score_temporal_174(row)
        artifact_rows.append(row)

    del comparator_df
    gc.collect()

artifact_metrics_df = pd.DataFrame(artifact_rows)
column_metrics_df = pd.DataFrame(column_rows)

if len(artifact_metrics_df) == 0:
    raise RuntimeError("[Cell17.4] No artifact metrics were produced.")

# ----------------------------------------------------------
# 8) Scope summary and comparison ledger
# ----------------------------------------------------------
evaluated = artifact_metrics_df[artifact_metrics_df["metric_status"].astype(str).eq("evaluated")].copy()

scope_summary_rows = []
ledger_rows = []

for scope_id, g in artifact_metrics_df.groupby("scope_id", dropna=False):
    ge = g[g["metric_status"].astype(str).eq("evaluated")].copy()
    if len(ge):
        ge["fair_scope_penalty_score"] = pd.to_numeric(ge["fair_scope_penalty_score"], errors="coerce")
        best = ge.sort_values("fair_scope_penalty_score", ascending=True, na_position="last").iloc[0]
        best_baseline = str(best["baseline"])
        best_score = _safe_float_174(best["fair_scope_penalty_score"], np.nan)
    else:
        best_baseline = ""
        best_score = np.nan

    scope_summary_rows.append({
        "scope_id": str(scope_id),
        "artifacts_total": int(len(g)),
        "artifacts_evaluated": int(len(ge)),
        "external_baselines_evaluated": int((ge["artifact_type"].astype(str) == "external_baseline").sum()) if len(ge) else 0,
        "pipeline_comparator_evaluated": bool((ge["artifact_type"].astype(str) == "pipeline_same_scope_comparator").any()) if len(ge) else False,
        "best_artifact": best_baseline,
        "best_fair_scope_penalty_score": best_score,
        "metric_family": str(g["metric_family"].dropna().iloc[0]) if "metric_family" in g.columns and len(g["metric_family"].dropna()) else "",
    })

    if len(ge):
        pipeline_rows = ge[ge["artifact_type"].astype(str).eq("pipeline_same_scope_comparator")]
        if len(pipeline_rows):
            pipeline_score = _safe_float_174(pipeline_rows.iloc[0].get("fair_scope_penalty_score"), np.nan)
            for _, r in ge[ge["artifact_type"].astype(str).eq("external_baseline")].iterrows():
                bscore = _safe_float_174(r.get("fair_scope_penalty_score"), np.nan)
                ledger_rows.append({
                    "scope_id": str(scope_id),
                    "baseline": str(r.get("baseline", "")),
                    "external_baseline_score": bscore,
                    "pipeline_same_scope_score": pipeline_score,
                    "delta_external_minus_pipeline": bscore - pipeline_score if np.isfinite(bscore) and np.isfinite(pipeline_score) else np.nan,
                    "winner": (
                        "external_baseline"
                        if np.isfinite(bscore) and np.isfinite(pipeline_score) and bscore < pipeline_score
                        else "pipeline_same_scope"
                        if np.isfinite(bscore) and np.isfinite(pipeline_score)
                        else "not_comparable"
                    ),
                    "interpretation": (
                        "Lower penalty is better. This is a same-scope comparison only, "
                        "not a full-pipeline comparison."
                    ),
                    "TEST_real_values_used_for_reference_only": True,
                    "synthetic_values_mutated": False,
                })

scope_summary_df = pd.DataFrame(scope_summary_rows)
comparison_ledger_df = pd.DataFrame(ledger_rows)

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
artifact_metrics_df.to_csv(artifact_metrics_csv, index=False)
column_metrics_df.to_csv(column_metrics_csv, index=False)
scope_summary_df.to_csv(scope_summary_csv, index=False)
comparison_ledger_df.to_csv(comparison_ledger_csv, index=False)

# ----------------------------------------------------------
# 10) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "17.4",
    "version": CELL174_VERSION,
    "role": "external_baseline_metric_computation_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "pipeline_comparator_uses_no_q4_or_public_q6_only": True,
        "q4_coupled_comparator_paths_forbidden": True,
    },
    "q6_public_context": CELL17_0_BASELINE_FAIRNESS_CONTRACT.get("q6_public_context", {}),
    "upstream_contract_versions": {
        "cell17_0": CELL17_0_VERSION_174,
        "cell17_1": CELL17_1_VERSION_174,
        "cell17_2": CELL17_2_VERSION_174,
        "cell17_3": CELL17_3_VERSION_174
    },
    "summary": {
        "artifact_metric_rows": int(len(artifact_metrics_df)),
        "column_metric_rows": int(len(column_metrics_df)),
        "scope_summary_rows": int(len(scope_summary_df)),
        "comparison_ledger_rows": int(len(comparison_ledger_df)),
        "evaluated_artifacts": int(len(evaluated)),
        "external_baselines_evaluated": int((evaluated["artifact_type"].astype(str) == "external_baseline").sum()) if len(evaluated) else 0,
        "pipeline_comparators_evaluated": int((evaluated["artifact_type"].astype(str) == "pipeline_same_scope_comparator").sum()) if len(evaluated) else 0,
    },
    "fairness_contract": {
        "tabular_scopes": [
            "protocol_core_tabular",
            "public_candidate_core_tabular",
            "iot_continuous_compact_tabular",
        ],
        "temporal_scopes": [
            "sequence_core_temporal",
            "protocol_sequence_temporal",
        ],
        "not_allowed_claims": [
            "external baselines are full CPS replacements",
            "external baselines are evaluated on Q3 observability",
            "external baselines are evaluated on Q4 coupling repair",
            "external baselines are evaluated on Q6 release safety",
        ],
        "same_scope_only": True,
        "q4_coupled_artifacts_not_used": True,
        "public_candidate_scope_is_not_public_release_pass": True,
        "baseline_c2st_uses_temporal_blocked_split": True,
        "q4_coupled_comparator_paths_forbidden": True,
        "ctgan_kernel_safe_v1_2_contract_accepted": True,
    },
    "metrics": {
        "tabular": [
            "mean KS",
            "histogram total variation",
            "normalized Wasserstein",
            "support Jaccard",
            "correlation MAE",
            "temporal-blocked row-level C2ST AUC",
        ],
        "temporal": [
            "mean KS",
            "histogram total variation",
            "normalized Wasserstein",
            "lag autocorrelation error",
            "transition-rate error",
            "run-length KS",
            "activity-rate error",
        ],
        "penalty_score": "lower is better; scope-specific aggregate only",
        "c2st_split_policy": "temporal_blocked_same_positions; train on early block and test on later block; no random train_test_split",
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_for_reference_only": True,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "metric_computation_done_here": True,
        "c2st_split_policy": "temporal_blocked_same_positions",
    },
    "outputs": {
        "artifact_metrics_csv": artifact_metrics_csv,
        "column_metrics_csv": column_metrics_csv,
        "scope_summary_csv": scope_summary_csv,
        "comparison_ledger_csv": comparison_ledger_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_174(contract_json, contract)
_write_json_174(contract_canonical_json, contract)

manifest = {
    "cell": "17.4",
    "version": CELL174_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "summary": contract["summary"],
    "fairness_contract": contract["fairness_contract"],
    "strict_contract": contract["strict_contract"],
}

_write_json_174(manifest_json, manifest)

hashes = {
    "artifact_metrics_csv_sha256": _sha256_file_174(artifact_metrics_csv),
    "column_metrics_csv_sha256": _sha256_file_174(column_metrics_csv),
    "scope_summary_csv_sha256": _sha256_file_174(scope_summary_csv),
    "comparison_ledger_csv_sha256": _sha256_file_174(comparison_ledger_csv),
    "contract_json_sha256": _sha256_file_174(contract_json),
    "contract_canonical_json_sha256": _sha256_file_174(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_174(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_174(contract_json, contract)
_write_json_174(contract_canonical_json, contract)
_write_json_174(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL174_VERSION"] = CELL174_VERSION
globals()["CELL17_4_BASELINE_ARTIFACT_METRICS_DF"] = artifact_metrics_df
globals()["CELL17_4_BASELINE_COLUMN_METRICS_DF"] = column_metrics_df
globals()["CELL17_4_BASELINE_SCOPE_SUMMARY_DF"] = scope_summary_df
globals()["CELL17_4_BASELINE_COMPARISON_LEDGER_DF"] = comparison_ledger_df
globals()["CELL17_4_BASELINE_METRIC_CONTRACT"] = contract

globals()["CELL17_4_BASELINE_ARTIFACT_METRICS_CSV"] = artifact_metrics_csv
globals()["CELL17_4_BASELINE_COLUMN_METRICS_CSV"] = column_metrics_csv
globals()["CELL17_4_BASELINE_SCOPE_SUMMARY_CSV"] = scope_summary_csv
globals()["CELL17_4_BASELINE_COMPARISON_LEDGER_CSV"] = comparison_ledger_csv
globals()["CELL17_4_BASELINE_CONTRACT_JSON"] = contract_json
globals()["CELL17_4_BASELINE_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL17_4_BASELINE_MANIFEST_JSON"] = manifest_json

log(
    "[Cell17.4] Baseline metric computation complete | "
    f"artifact_rows={len(artifact_metrics_df)} | "
    f"column_rows={len(column_metrics_df)} | "
    f"scope_rows={len(scope_summary_df)} | "
    f"ledger_rows={len(comparison_ledger_df)} | q4_status=blocked_no_promotion"
)
if len(scope_summary_df):
    log(f"[Cell17.4] Scope summary | {scope_summary_df.to_dict('records')}")
log(f"[Cell17.4] Saved artifact metrics: {artifact_metrics_csv}")
log(f"[Cell17.4] Saved column metrics: {column_metrics_csv}")
log(f"[Cell17.4] Saved scope summary: {scope_summary_csv}")
log(f"[Cell17.4] Saved comparison ledger: {comparison_ledger_csv}")
log(f"[Cell17.4] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell17.4] Contract flags | "
    "TEST_real_values_used_for_reference_only=True | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "metric_computation_done_here=True | c2st_split_policy=temporal_blocked_same_positions | Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False"
)
log("--- END: Cell 17.4 - External baseline metric computation (v1.2 temporal-blocked-C2ST no-Q4-promotion strict) ---")

gc.collect()
from pathlib import Path as _AuditPath174
(_AuditPath174(REPORT_DIR_BASE_174) / "baseline_c2st_fallback_events.json").write_text(json.dumps({"events": C2ST_FALLBACK_EVENTS_174}, indent=2))
