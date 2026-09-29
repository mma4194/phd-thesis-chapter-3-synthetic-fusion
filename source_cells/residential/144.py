# ==========================================================
# CELL 16.2 - Q2 temporal decomposition
# v1.1 STUDY-THESIS strict no-Q4-promotion A0/A1/A2 temporal-quality decomposition
#
# Role:
#   - Decompose temporal quality across A0/A1/A2/no-Q4 scientific/public branches.
#   - Explain where Q2 temporal gains/failures come from:
#       autocorrelation preservation
#       transition/event-rate preservation
#       run-length / dwell structure
#       burst/activity-window consistency
#
# Scientific contract:
#   - No generation.
#   - No selection.
#   - No repair.
#   - No synthetic mutation.
#   - TEST real values are used only for decomposition/reference metrics.
#   - This cell reports Q2 decomposition; it does not change the pipeline.
#
# Outputs:
#   reports/cell16_2_q2_temporal_decomposition_by_variant.csv
#   reports/cell16_2_q2_temporal_decomposition_by_role.csv
#   reports/cell16_2_q2_temporal_decomposition_by_column.csv
#   reports/cell16_2_q2_temporal_gain_loss_ledger.csv
#   reports/cell16_2_q2_temporal_decomposition_contract.json
#   artifacts/cell16_2_q2_temporal_decomposition_manifest.json
# ==========================================================

log("--- START: Cell 16.2 - Q2 temporal decomposition (v1.1 no-Q4-promotion strict) ---")

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
_required_162 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL16_0_ARTIFACT_REGISTRY_DF",
    "CELL16_0_METRIC_SOURCE_REGISTRY_DF",
    "CELL16_0_DECOMPOSITION_INPUT_CONTRACT",
    "CELL16_0_A0_PROTOCOL_PATH",
    "CELL16_0_A1_PROTOCOL_PATH",
    "CELL16_0_A2_PROTOCOL_PATH",
    "CELL16_0_SCIENTIFIC_NO_Q4_PROTOCOL_PATH",
    "CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH",
    "CELL16_0_PUBLIC_Q6_CPS_PATH",
    "CELL16_0_DECOMPOSITION_INPUT_CONTRACT",
    "CELL16_1_Q1_VARIANT_SUMMARY_DF",
    "CELL16_1_Q1_DECOMPOSITION_CONTRACT",
]
_missing_162 = [k for k in _required_162 if k not in globals()]
if _missing_162:
    raise RuntimeError(f"[Cell16.2] Missing required globals: {_missing_162}")

ORIGINAL_OUTDIR_162 = str(OUTDIR)
ORIGINAL_OUT_SYN_162 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_162 = str(REPORT_DIR)

def _resolve_project_root_162(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        for marker in ["q6_public_reaudit_v1", "q6_public_reaudit"]:
            if marker in parts:
                candidates.append(os.sep.join(parts[:parts.index(marker)]))
        if os.path.basename(p) in {"reports", "synthetic", "artifacts"}:
            candidates.append(os.path.dirname(p))
        else:
            candidates.append(p)
    _env_project_root = os.environ.get("CPS_CANONICAL_PROJECT_ROOT", "").strip()
    if _env_project_root:
        candidates.append(_env_project_root)

    seen = set()
    for c in candidates:
        c = os.path.abspath(str(c))
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "synthetic")):
            return c

    raise RuntimeError("[Cell16.2] Could not resolve canonical project root.")

PROJECT_ROOT_162 = _resolve_project_root_162(ORIGINAL_OUTDIR_162, ORIGINAL_REPORT_DIR_162, ORIGINAL_OUT_SYN_162)
OUTDIR_BASE_162 = PROJECT_ROOT_162
OUT_SYN_BASE_162 = os.path.join(PROJECT_ROOT_162, "synthetic")
REPORT_DIR_BASE_162 = os.path.join(PROJECT_ROOT_162, "reports")
ARTDIR_BASE_162 = os.path.join(PROJECT_ROOT_162, "artifacts")
CONTRACT_DIR_BASE_162 = os.path.join(ARTDIR_BASE_162, "contracts")

os.makedirs(REPORT_DIR_BASE_162, exist_ok=True)
os.makedirs(ARTDIR_BASE_162, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_162, exist_ok=True)

SEED = int(SEED)
N_TE = int(len(df_te))

CELL162_VERSION = "cell16_2_q2_temporal_decomposition_v1_1_no_q4_promotion"

CFG["cell16_2_version"] = CELL162_VERSION
CFG["cell16_2_TEST_real_values_used_for_reference_only"] = True
CFG["cell16_2_TEST_real_values_used_for_materialization"] = False
CFG["cell16_2_synthetic_values_mutated"] = False
CFG["cell16_2_selection_done_here"] = False
CFG["cell16_2_generator_fit_done_here"] = False
CFG["cell16_2_materialization_done_here"] = False
CFG["cell16_2_decomposition_done_here"] = True
CFG["cell16_2_Q4_final_status"] = "blocked_no_promotion"
CFG["cell16_2_Q4_coupled_artifacts_used"] = False

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell16_2_max_columns_per_variant", 250)
CFG.setdefault("cell16_2_numeric_min_finite_rate", 0.001)
CFG.setdefault("cell16_2_low_cardinality_unique_threshold", 30)
CFG.setdefault("cell16_2_activity_eps", 1e-12)
CFG.setdefault("cell16_2_max_run_lengths_to_compare", 5000)
CFG.setdefault("cell16_2_lag_list", [1, 5, 10, 30, 60])
CFG.setdefault("cell16_2_lag1_error_pass", 0.10)
CFG.setdefault("cell16_2_lag1_error_warning", 0.30)
CFG.setdefault("cell16_2_multi_lag_mae_pass", 0.15)
CFG.setdefault("cell16_2_multi_lag_mae_warning", 0.40)
CFG.setdefault("cell16_2_transition_rate_error_pass", 0.05)
CFG.setdefault("cell16_2_transition_rate_error_warning", 0.20)
CFG.setdefault("cell16_2_run_length_ks_pass", 0.15)
CFG.setdefault("cell16_2_run_length_ks_warning", 0.35)
CFG.setdefault("cell16_2_activity_rate_error_pass", 0.05)
CFG.setdefault("cell16_2_activity_rate_error_warning", 0.20)

MAX_COLS_162 = int(CFG.get("cell16_2_max_columns_per_variant", 250))
MIN_FINITE_RATE_162 = float(CFG.get("cell16_2_numeric_min_finite_rate", 0.001))
LOW_CARD_UNIQUE_162 = int(CFG.get("cell16_2_low_cardinality_unique_threshold", 30))
ACTIVITY_EPS_162 = float(CFG.get("cell16_2_activity_eps", 1e-12))
MAX_RUNS_162 = int(CFG.get("cell16_2_max_run_lengths_to_compare", 5000))
LAGS_162 = [int(x) for x in CFG.get("cell16_2_lag_list", [1, 5, 10, 30, 60])]

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
variant_summary_csv = os.path.join(REPORT_DIR_BASE_162, "cell16_2_q2_temporal_decomposition_by_variant.csv")
role_summary_csv = os.path.join(REPORT_DIR_BASE_162, "cell16_2_q2_temporal_decomposition_by_role.csv")
column_metrics_csv = os.path.join(REPORT_DIR_BASE_162, "cell16_2_q2_temporal_decomposition_by_column.csv")
gain_loss_ledger_csv = os.path.join(REPORT_DIR_BASE_162, "cell16_2_q2_temporal_gain_loss_ledger.csv")
contract_json = os.path.join(REPORT_DIR_BASE_162, "cell16_2_q2_temporal_decomposition_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_162, "cell16_2_q2_temporal_decomposition_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_162, "cell16_2_q2_temporal_decomposition_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_162(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_162(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_162(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_162(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_162(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_162(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_162(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_162(obj.to_dict())
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

def _write_json_162(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_162(payload), f, indent=2, sort_keys=True)

def _sha256_file_162(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_162(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _require_contract_governance_162(contract: dict, name: str, expected_version_substring: str):
    if not isinstance(contract, dict):
        raise RuntimeError(f"[Cell16.2] {name} is not a dict.")
    version = str(contract.get("version", ""))
    if expected_version_substring not in version:
        raise RuntimeError(
            f"[Cell16.2] Unexpected {name} version. "
            f"Expected substring={expected_version_substring}, got={version}"
        )
    q4 = contract.get("q4_governance", {})
    if str(q4.get("final_q4_status", contract.get("strict_contract", {}).get("Q4_final_status", ""))) != "blocked_no_promotion":
        raise RuntimeError(f"[Cell16.2] {name} does not carry Q4 blocked_no_promotion governance.")
    if bool(q4.get("q4_coupled_artifacts_used", contract.get("strict_contract", {}).get("Q4_coupled_artifacts_used", True))):
        raise RuntimeError(f"[Cell16.2] {name} indicates Q4-coupled artifacts were used.")
    return version

def _read_parquet_optional_162(path: str, columns=None):
    if not _exists_162(path):
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return None

def _safe_float_162(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _finite_162(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]

def _numeric_array_162(frame: pd.DataFrame, col: str):
    if not isinstance(frame, pd.DataFrame) or col not in frame.columns:
        return np.asarray([], dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)

def _is_time_col_162(c: str) -> bool:
    s = str(c).lower()
    return s in {"sec", "sec_epoch_s__canon", "timestamp", "time", "datetime"} or s.endswith("__time")

def _role_group_162(col: str) -> str:
    s = str(col)
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
    if s.startswith("iot__entity_obs__") or s.startswith("iot__entity_stale__") or "obs_present" in s or "stale" in s:
        return "iot_observability_masks"
    if s.startswith("iot__") and (s.endswith("__state") or "__binary_sensor__" in s or "__switch__" in s):
        return "iot_binary_states"
    if s.startswith("iot__") and s.endswith("__value"):
        return "iot_continuous_values"
    if s.startswith("iot__"):
        return "iot_other"
    if _is_time_col_162(s):
        return "time"
    return "other"

def _ks_2samp_manual_162(a, b):
    a = np.sort(_finite_162(a))
    b = np.sort(_finite_162(b))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    vals = np.sort(np.unique(np.concatenate([a, b])))
    if len(vals) == 0:
        return np.nan
    cdf_a = np.searchsorted(a, vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, vals, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))

def _autocorr_162(x, lag: int):
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

def _multi_lag_autocorrs_162(x, lags):
    return {int(l): _autocorr_162(x, int(l)) for l in lags}

def _transition_rate_162(x):
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    if valid.sum() < 2:
        return np.nan

    # For low-cardinality/discrete signals, exact changes.
    xf = x.copy()
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.nan

    vals = xf[idx]
    return float(np.mean(vals[1:] != vals[:-1]))

def _activity_rate_162(x):
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(x[valid]) > ACTIVITY_EPS_162))

def _run_lengths_binary_162(active):
    active = np.asarray(active, dtype=bool)
    if active.size == 0:
        return np.asarray([], dtype=np.float64)

    runs = []
    current = active[0]
    length = 1

    for v in active[1:]:
        if v == current:
            length += 1
        else:
            runs.append(length)
            current = v
            length = 1
    runs.append(length)

    arr = np.asarray(runs, dtype=np.float64)
    if len(arr) > MAX_RUNS_162:
        # deterministic thinning
        idx = np.linspace(0, len(arr) - 1, MAX_RUNS_162).astype(int)
        arr = arr[idx]
    return arr

def _value_run_lengths_162(x):
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    if valid.sum() == 0:
        return np.asarray([], dtype=np.float64)

    xv = x[valid]
    if len(xv) == 0:
        return np.asarray([], dtype=np.float64)

    runs = []
    cur = xv[0]
    length = 1
    for v in xv[1:]:
        if v == cur:
            length += 1
        else:
            runs.append(length)
            cur = v
            length = 1
    runs.append(length)

    arr = np.asarray(runs, dtype=np.float64)
    if len(arr) > MAX_RUNS_162:
        idx = np.linspace(0, len(arr) - 1, MAX_RUNS_162).astype(int)
        arr = arr[idx]
    return arr

def _run_length_ks_162(real_x, syn_x):
    r = np.asarray(real_x, dtype=np.float64)
    s = np.asarray(syn_x, dtype=np.float64)
    rf = _finite_162(r)
    sf = _finite_162(s)

    if len(rf) == 0 or len(sf) == 0:
        return np.nan

    low_card = bool(len(np.unique(rf)) <= LOW_CARD_UNIQUE_162 and len(np.unique(sf)) <= LOW_CARD_UNIQUE_162)

    if low_card:
        rr = _value_run_lengths_162(r)
        ss = _value_run_lengths_162(s)
    else:
        rr = _run_lengths_binary_162(np.isfinite(r) & (np.abs(r) > ACTIVITY_EPS_162))
        ss = _run_lengths_binary_162(np.isfinite(s) & (np.abs(s) > ACTIVITY_EPS_162))

    return _ks_2samp_manual_162(rr, ss)

def _q2_status_162(row):
    lag1_err = _safe_float_162(row.get("lag1_autocorr_abs_error"), np.nan)
    multi_lag = _safe_float_162(row.get("multi_lag_autocorr_mae"), np.nan)
    trans_err = _safe_float_162(row.get("transition_rate_abs_error"), np.nan)
    run_ks = _safe_float_162(row.get("run_length_ks"), np.nan)
    act_err = _safe_float_162(row.get("activity_rate_abs_error"), np.nan)

    fatal = []
    warn = []

    if np.isfinite(lag1_err):
        if lag1_err > float(CFG.get("cell16_2_lag1_error_warning", 0.30)):
            fatal.append("lag1_autocorr_error_fatal")
        elif lag1_err > float(CFG.get("cell16_2_lag1_error_pass", 0.10)):
            warn.append("lag1_autocorr_error_warning")

    if np.isfinite(multi_lag):
        if multi_lag > float(CFG.get("cell16_2_multi_lag_mae_warning", 0.40)):
            fatal.append("multi_lag_autocorr_error_fatal")
        elif multi_lag > float(CFG.get("cell16_2_multi_lag_mae_pass", 0.15)):
            warn.append("multi_lag_autocorr_error_warning")

    if np.isfinite(trans_err):
        if trans_err > float(CFG.get("cell16_2_transition_rate_error_warning", 0.20)):
            fatal.append("transition_rate_error_fatal")
        elif trans_err > float(CFG.get("cell16_2_transition_rate_error_pass", 0.05)):
            warn.append("transition_rate_error_warning")

    if np.isfinite(run_ks):
        if run_ks > float(CFG.get("cell16_2_run_length_ks_warning", 0.35)):
            fatal.append("run_length_ks_fatal")
        elif run_ks > float(CFG.get("cell16_2_run_length_ks_pass", 0.15)):
            warn.append("run_length_ks_warning")

    if np.isfinite(act_err):
        if act_err > float(CFG.get("cell16_2_activity_rate_error_warning", 0.20)):
            fatal.append("activity_rate_error_fatal")
        elif act_err > float(CFG.get("cell16_2_activity_rate_error_pass", 0.05)):
            warn.append("activity_rate_error_warning")

    if fatal:
        return "fatal", "|".join(fatal)
    if warn:
        return "warning", "|".join(warn)
    return "pass", "within_q2_temporal_gates"

def _variant_score_162(g: pd.DataFrame):
    if not isinstance(g, pd.DataFrame) or len(g) == 0:
        return np.nan
    evaluable = g[~g["q2_status"].astype(str).eq("not_evaluable")].copy()
    if len(evaluable) == 0:
        return np.nan

    score = (
        pd.to_numeric(evaluable["lag1_autocorr_abs_error"], errors="coerce").fillna(0.5)
        + pd.to_numeric(evaluable["multi_lag_autocorr_mae"], errors="coerce").fillna(0.5)
        + pd.to_numeric(evaluable["transition_rate_abs_error"], errors="coerce").fillna(0.5)
        + pd.to_numeric(evaluable["run_length_ks"], errors="coerce").fillna(0.5)
        + pd.to_numeric(evaluable["activity_rate_abs_error"], errors="coerce").fillna(0.5)
    )
    return float(np.nanmean(score))

def _choose_candidate_cols_162(syn_df, real_df, artifact_role):
    syn_cols = list(map(str, syn_df.columns))
    real_cols = set(map(str, real_df.columns))

    if artifact_role in {
        "protocol_baseline",
        "protocol_backbone",
        "protocol_hybrid",
        "scientific_protocol_no_q4",
        "legacy_q4_protocol_not_authoritative",
    }:
        cols = [c for c in syn_cols if _role_group_162(c).startswith("protocol_") and c in real_cols]
    elif artifact_role == "public_q6_mitigated_cps":
        cols = [
            c for c in syn_cols
            if _role_group_162(c) in {"protocol_router", "protocol_zigbee", "iot_event_drivers"}
            and c in real_cols
        ]
    else:
        cols = [c for c in syn_cols if c in real_cols and not _is_time_col_162(c)]

    if len(cols) <= MAX_COLS_162:
        return cols

    scored = []
    for c in cols:
        sx = _numeric_array_162(syn_df, c)
        rx = _numeric_array_162(real_df, c)
        sf = np.isfinite(sx)
        rf = np.isfinite(rx)
        finite_rate = min(float(sf.mean()) if len(sf) else 0.0, float(rf.mean()) if len(rf) else 0.0)
        activity = max(_activity_rate_162(sx), _activity_rate_162(rx))
        trans = max(_transition_rate_162(sx), _transition_rate_162(rx))
        score = finite_rate + activity + trans
        scored.append((c, score))

    return [c for c, _ in sorted(scored, key=lambda x: (-x[1], x[0]))[:MAX_COLS_162]]

def _compute_q2_column_metrics_162(variant_name, artifact_role, syn_df, real_df, candidate_cols):
    rows = []

    syn_df = syn_df.copy()
    syn_df.columns = syn_df.columns.astype(str)

    for col in candidate_cols:
        if col not in syn_df.columns or col not in real_df.columns or _is_time_col_162(col):
            continue

        s = _numeric_array_162(syn_df, col)
        r = _numeric_array_162(real_df, col)

        if len(s) == 0 or len(r) == 0:
            continue

        syn_finite_rate = float(np.isfinite(s).mean())
        real_finite_rate = float(np.isfinite(r).mean())

        if syn_finite_rate < MIN_FINITE_RATE_162 or real_finite_rate < MIN_FINITE_RATE_162:
            rows.append({
                "variant": variant_name,
                "artifact_role": artifact_role,
                "role_group": _role_group_162(col),
                "col": col,
                "metric_scope": "Q2_temporal",
                "q2_status": "not_evaluable",
                "q2_reasons": "low_finite_rate",
                "syn_finite_rate": syn_finite_rate,
                "real_finite_rate": real_finite_rate,
                "TEST_real_values_used_for_reference_only": True,
                "synthetic_values_mutated": False,
            })
            continue

        r_ac = _multi_lag_autocorrs_162(r, LAGS_162)
        s_ac = _multi_lag_autocorrs_162(s, LAGS_162)

        ac_errors = []
        for lag in LAGS_162:
            rv = r_ac.get(lag, np.nan)
            sv = s_ac.get(lag, np.nan)
            if np.isfinite(rv) and np.isfinite(sv):
                ac_errors.append(abs(sv - rv))

        lag1_err = (
            abs(s_ac.get(1, np.nan) - r_ac.get(1, np.nan))
            if np.isfinite(s_ac.get(1, np.nan)) and np.isfinite(r_ac.get(1, np.nan))
            else np.nan
        )
        multi_lag_mae = float(np.mean(ac_errors)) if ac_errors else np.nan

        real_transition_rate = _transition_rate_162(r)
        syn_transition_rate = _transition_rate_162(s)
        transition_rate_abs_error = (
            abs(syn_transition_rate - real_transition_rate)
            if np.isfinite(syn_transition_rate) and np.isfinite(real_transition_rate)
            else np.nan
        )

        real_activity_rate = _activity_rate_162(r)
        syn_activity_rate = _activity_rate_162(s)
        activity_rate_abs_error = (
            abs(syn_activity_rate - real_activity_rate)
            if np.isfinite(syn_activity_rate) and np.isfinite(real_activity_rate)
            else np.nan
        )

        run_length_ks = _run_length_ks_162(r, s)

        real_unique_n = int(len(np.unique(_finite_162(r))))
        syn_unique_n = int(len(np.unique(_finite_162(s))))
        low_cardinality = bool(real_unique_n <= LOW_CARD_UNIQUE_162 and syn_unique_n <= LOW_CARD_UNIQUE_162)

        row = {
            "variant": variant_name,
            "artifact_role": artifact_role,
            "role_group": _role_group_162(col),
            "col": col,
            "metric_scope": "Q2_temporal",
            "syn_finite_rate": syn_finite_rate,
            "real_finite_rate": real_finite_rate,
            "real_unique_n": real_unique_n,
            "syn_unique_n": syn_unique_n,
            "low_cardinality": low_cardinality,

            "real_lag1_autocorr": r_ac.get(1, np.nan),
            "syn_lag1_autocorr": s_ac.get(1, np.nan),
            "lag1_autocorr_abs_error": lag1_err,
            "multi_lag_autocorr_mae": multi_lag_mae,

            "real_transition_rate": real_transition_rate,
            "syn_transition_rate": syn_transition_rate,
            "transition_rate_abs_error": transition_rate_abs_error,

            "real_activity_rate": real_activity_rate,
            "syn_activity_rate": syn_activity_rate,
            "activity_rate_abs_error": activity_rate_abs_error,

            "run_length_ks": run_length_ks,

            "lags_evaluated": "|".join(map(str, LAGS_162)),
            "TEST_real_values_used_for_reference_only": True,
            "synthetic_values_mutated": False,
        }

        for lag in LAGS_162:
            row[f"real_lag{lag}_autocorr"] = r_ac.get(lag, np.nan)
            row[f"syn_lag{lag}_autocorr"] = s_ac.get(lag, np.nan)

        status, reasons = _q2_status_162(row)
        row["q2_status"] = status
        row["q2_reasons"] = reasons
        rows.append(row)

    return rows

# ----------------------------------------------------------
# 4) Validate upstream no-Q4 governance contracts
# ----------------------------------------------------------
CELL16_0_CONTRACT_VERSION_162 = _require_contract_governance_162(
    CELL16_0_DECOMPOSITION_INPUT_CONTRACT,
    "CELL16_0_DECOMPOSITION_INPUT_CONTRACT",
    "cell16_0_a0_a1_a2_decomposition_input_contract_v1_1",
)
CELL16_1_CONTRACT_VERSION_162 = _require_contract_governance_162(
    CELL16_1_Q1_DECOMPOSITION_CONTRACT,
    "CELL16_1_Q1_DECOMPOSITION_CONTRACT",
    "cell16_1_q1_marginal_decomposition_v1_1",
)

# ----------------------------------------------------------
# 5) Load real TEST reference and synthetic variants
# ----------------------------------------------------------
real_test_df_162 = df_te.copy()
real_test_df_162.columns = real_test_df_162.columns.astype(str)

variant_specs = []

def _add_variant_spec_162(variant, artifact_role, path, required=False):
    if _exists_162(path):
        variant_specs.append({
            "variant": variant,
            "artifact_role": artifact_role,
            "path": path,
            "exists": True,
            "required": bool(required),
        })
    elif required:
        raise RuntimeError(f"[Cell16.2] Missing required variant artifact: {variant} | {path}")
    else:
        variant_specs.append({
            "variant": variant,
            "artifact_role": artifact_role,
            "path": path,
            "exists": False,
            "required": bool(required),
        })

_add_variant_spec_162("A0", "protocol_baseline", CELL16_0_A0_PROTOCOL_PATH, required=False)
_add_variant_spec_162("A1", "protocol_backbone", CELL16_0_A1_PROTOCOL_PATH, required=False)
_add_variant_spec_162("A2", "protocol_hybrid", CELL16_0_A2_PROTOCOL_PATH, required=False)

# Authoritative scientific variants after Q4 was blocked/no-promoted.
_add_variant_spec_162("scientific_no_q4_protocol", "scientific_protocol_no_q4", CELL16_0_SCIENTIFIC_NO_Q4_PROTOCOL_PATH, required=True)
_add_variant_spec_162("scientific_no_q4_cps", "scientific_cps_no_q4", CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH, required=True)

# Optional non-authoritative legacy Q4 context only, if registered by Cell 16.0.
if isinstance(globals().get("CELL16_0_LEGACY_Q4_PROTOCOL_PATH", ""), str) and _exists_162(globals().get("CELL16_0_LEGACY_Q4_PROTOCOL_PATH", "")):
    _add_variant_spec_162("legacy_q4_protocol_context", "legacy_q4_protocol_not_authoritative", globals()["CELL16_0_LEGACY_Q4_PROTOCOL_PATH"], required=False)
if isinstance(globals().get("CELL16_0_LEGACY_Q4_CPS_PATH", ""), str) and _exists_162(globals().get("CELL16_0_LEGACY_Q4_CPS_PATH", "")):
    _add_variant_spec_162("legacy_q4_cps_context", "legacy_q4_cps_not_authoritative", globals()["CELL16_0_LEGACY_Q4_CPS_PATH"], required=False)

# Public Q6 role-restricted release candidate.
_add_variant_spec_162("Q6_public_cps", "public_q6_mitigated_cps", CELL16_0_PUBLIC_Q6_CPS_PATH, required=False)

log(
    "[Cell16.2] Computing Q2 temporal decomposition | "
    f"variants={[v['variant'] for v in variant_specs if v['exists']]} | lags={LAGS_162}"
)

all_column_rows = []
variant_load_rows = []

for spec in variant_specs:
    variant = str(spec["variant"])
    artifact_role = str(spec["artifact_role"])
    path = str(spec["path"])

    if not bool(spec["exists"]):
        variant_load_rows.append({
            "variant": variant,
            "artifact_role": artifact_role,
            "path": path,
            "loaded": False,
            "rows": 0,
            "cols": 0,
            "reason": "missing_optional_artifact",
        })
        continue

    syn_df = _read_parquet_optional_162(path)
    if not isinstance(syn_df, pd.DataFrame):
        variant_load_rows.append({
            "variant": variant,
            "artifact_role": artifact_role,
            "path": path,
            "loaded": False,
            "rows": 0,
            "cols": 0,
            "reason": "read_failed",
        })
        continue

    syn_df.columns = syn_df.columns.astype(str)

    if len(syn_df) != N_TE:
        raise RuntimeError(
            f"[Cell16.2] Row mismatch for variant={variant}: got={len(syn_df)} expected={N_TE}"
        )

    candidate_cols = _choose_candidate_cols_162(
        syn_df=syn_df,
        real_df=real_test_df_162,
        artifact_role=artifact_role,
    )

    rows = _compute_q2_column_metrics_162(
        variant_name=variant,
        artifact_role=artifact_role,
        syn_df=syn_df,
        real_df=real_test_df_162,
        candidate_cols=candidate_cols,
    )

    all_column_rows.extend(rows)

    variant_load_rows.append({
        "variant": variant,
        "artifact_role": artifact_role,
        "path": path,
        "loaded": True,
        "rows": int(syn_df.shape[0]),
        "cols": int(syn_df.shape[1]),
        "evaluated_candidate_cols": int(len(candidate_cols)),
        "evaluated_metric_rows": int(len(rows)),
        "reason": "ok",
    })

    del syn_df
    gc.collect()

column_metrics_df = pd.DataFrame(all_column_rows)
variant_load_df = pd.DataFrame(variant_load_rows)

if len(column_metrics_df) == 0:
    raise RuntimeError("[Cell16.2] No Q2 temporal decomposition metrics were produced.")

# ----------------------------------------------------------
# 6) Summaries by variant and role
# ----------------------------------------------------------
def _summary_from_group_162(g: pd.DataFrame):
    status_counts = g["q2_status"].astype(str).value_counts().to_dict()
    evaluable = g[~g["q2_status"].astype(str).eq("not_evaluable")].copy()

    return pd.Series({
        "cols_total": int(len(g)),
        "cols_evaluable": int(len(evaluable)),
        "pass_n": int(status_counts.get("pass", 0)),
        "warning_n": int(status_counts.get("warning", 0)),
        "fatal_n": int(status_counts.get("fatal", 0)),
        "not_evaluable_n": int(status_counts.get("not_evaluable", 0)),
        "pass_rate_evaluable": float((evaluable["q2_status"].astype(str).eq("pass")).mean()) if len(evaluable) else np.nan,
        "fatal_rate_evaluable": float((evaluable["q2_status"].astype(str).eq("fatal")).mean()) if len(evaluable) else np.nan,
        "mean_lag1_autocorr_abs_error": float(pd.to_numeric(evaluable["lag1_autocorr_abs_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "median_lag1_autocorr_abs_error": float(pd.to_numeric(evaluable["lag1_autocorr_abs_error"], errors="coerce").median()) if len(evaluable) else np.nan,
        "mean_multi_lag_autocorr_mae": float(pd.to_numeric(evaluable["multi_lag_autocorr_mae"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_transition_rate_abs_error": float(pd.to_numeric(evaluable["transition_rate_abs_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_run_length_ks": float(pd.to_numeric(evaluable["run_length_ks"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_activity_rate_abs_error": float(pd.to_numeric(evaluable["activity_rate_abs_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "q2_temporal_penalty_score": _variant_score_162(evaluable),
    })

variant_summary_df = (
    column_metrics_df
    .groupby(["variant", "artifact_role"], dropna=False)
    .apply(_summary_from_group_162)
    .reset_index()
    .sort_values(["q2_temporal_penalty_score", "variant"], ascending=[True, True])
    .reset_index(drop=True)
)

role_summary_df = (
    column_metrics_df
    .groupby(["variant", "artifact_role", "role_group"], dropna=False)
    .apply(_summary_from_group_162)
    .reset_index()
    .sort_values(["variant", "role_group"])
    .reset_index(drop=True)
)

# ----------------------------------------------------------
# 7) Gain/loss ledger
# ----------------------------------------------------------
ledger_rows = []

protocol_variants = ["A0", "A1", "A2", "scientific_no_q4_protocol", "legacy_q4_protocol_context"]
protocol_summary = variant_summary_df[variant_summary_df["variant"].isin(protocol_variants)].copy()

if len(protocol_summary):
    best_protocol = protocol_summary.sort_values("q2_temporal_penalty_score", ascending=True).iloc[0]
    worst_protocol = protocol_summary.sort_values("q2_temporal_penalty_score", ascending=False).iloc[0]

    ledger_rows.append({
        "ledger_item": "best_protocol_q2_variant",
        "variant": str(best_protocol["variant"]),
        "artifact_role": str(best_protocol["artifact_role"]),
        "q2_temporal_penalty_score": _safe_float_162(best_protocol["q2_temporal_penalty_score"], np.nan),
        "mean_lag1_autocorr_abs_error": _safe_float_162(best_protocol["mean_lag1_autocorr_abs_error"], np.nan),
        "mean_run_length_ks": _safe_float_162(best_protocol["mean_run_length_ks"], np.nan),
        "pass_rate_evaluable": _safe_float_162(best_protocol["pass_rate_evaluable"], np.nan),
        "interpretation": "Lowest temporal penalty among protocol variants.",
    })

    ledger_rows.append({
        "ledger_item": "worst_protocol_q2_variant",
        "variant": str(worst_protocol["variant"]),
        "artifact_role": str(worst_protocol["artifact_role"]),
        "q2_temporal_penalty_score": _safe_float_162(worst_protocol["q2_temporal_penalty_score"], np.nan),
        "mean_lag1_autocorr_abs_error": _safe_float_162(worst_protocol["mean_lag1_autocorr_abs_error"], np.nan),
        "mean_run_length_ks": _safe_float_162(worst_protocol["mean_run_length_ks"], np.nan),
        "pass_rate_evaluable": _safe_float_162(worst_protocol["pass_rate_evaluable"], np.nan),
        "interpretation": "Highest temporal penalty among protocol variants.",
    })

    score_map = dict(zip(protocol_summary["variant"].astype(str), pd.to_numeric(protocol_summary["q2_temporal_penalty_score"], errors="coerce")))
    for a, b in [("A0", "A1"), ("A1", "A2"), ("A2", "scientific_no_q4_protocol"), ("A0", "scientific_no_q4_protocol")]:
        if a in score_map and b in score_map and np.isfinite(score_map[a]) and np.isfinite(score_map[b]):
            delta = float(score_map[b] - score_map[a])
            ledger_rows.append({
                "ledger_item": f"q2_penalty_delta::{a}_to_{b}",
                "variant": f"{a}->{b}",
                "artifact_role": "protocol_delta",
                "q2_temporal_penalty_score": delta,
                "interpretation": (
                    "negative means later variant improved temporal penalty; "
                    "positive means later variant worsened temporal penalty."
                ),
            })

# Worst final temporal columns.
final_cols = column_metrics_df[
    column_metrics_df["variant"].astype(str).eq("scientific_no_q4_cps")
    & column_metrics_df["q2_status"].astype(str).isin(["fatal", "warning"])
].copy()

if len(final_cols):
    final_cols["column_temporal_penalty"] = (
        pd.to_numeric(final_cols["lag1_autocorr_abs_error"], errors="coerce").fillna(0.5)
        + pd.to_numeric(final_cols["multi_lag_autocorr_mae"], errors="coerce").fillna(0.5)
        + pd.to_numeric(final_cols["transition_rate_abs_error"], errors="coerce").fillna(0.5)
        + pd.to_numeric(final_cols["run_length_ks"], errors="coerce").fillna(0.5)
        + pd.to_numeric(final_cols["activity_rate_abs_error"], errors="coerce").fillna(0.5)
    )

    for _, r in final_cols.sort_values("column_temporal_penalty", ascending=False).head(25).iterrows():
        ledger_rows.append({
            "ledger_item": "worst_final_q2_column",
            "variant": str(r["variant"]),
            "artifact_role": str(r["artifact_role"]),
            "role_group": str(r["role_group"]),
            "col": str(r["col"]),
            "q2_status": str(r["q2_status"]),
            "q2_reasons": str(r["q2_reasons"]),
            "q2_temporal_penalty_score": _safe_float_162(r["column_temporal_penalty"], np.nan),
            "lag1_autocorr_abs_error": _safe_float_162(r["lag1_autocorr_abs_error"], np.nan),
            "multi_lag_autocorr_mae": _safe_float_162(r["multi_lag_autocorr_mae"], np.nan),
            "transition_rate_abs_error": _safe_float_162(r["transition_rate_abs_error"], np.nan),
            "run_length_ks": _safe_float_162(r["run_length_ks"], np.nan),
            "activity_rate_abs_error": _safe_float_162(r["activity_rate_abs_error"], np.nan),
            "interpretation": "High-priority temporal mismatch in authoritative no-Q4 scientific artifact.",
        })

gain_loss_ledger_df = pd.DataFrame(ledger_rows)

# ----------------------------------------------------------
# 8) Save outputs
# ----------------------------------------------------------
column_metrics_df.to_csv(column_metrics_csv, index=False)
variant_summary_df.to_csv(variant_summary_csv, index=False)
role_summary_df.to_csv(role_summary_csv, index=False)
gain_loss_ledger_df.to_csv(gain_loss_ledger_csv, index=False)

# ----------------------------------------------------------
# 9) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "16.2",
    "version": CELL162_VERSION,
    "role": "q2_temporal_decomposition_no_q4_promotion",
    "quality_dimension": "Q2_temporal_dynamics_quality",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "legacy_q4_variants_non_authoritative": True
    },
    "upstream_contract_versions": {
        "cell16_0": CELL16_0_CONTRACT_VERSION_162,
        "cell16_1": CELL16_1_CONTRACT_VERSION_162
    },
    "summary": {
        "variants_loaded": variant_load_df.to_dict("records"),
        "column_metric_rows": int(len(column_metrics_df)),
        "variant_summary_rows": int(len(variant_summary_df)),
        "role_summary_rows": int(len(role_summary_df)),
        "gain_loss_ledger_rows": int(len(gain_loss_ledger_df)),
        "best_variant_by_q2_temporal_penalty": (
            variant_summary_df.iloc[0].to_dict()
            if len(variant_summary_df)
            else {}
        ),
    },
    "method": {
        "metrics": [
            "lag-1 autocorrelation absolute error",
            "multi-lag autocorrelation MAE",
            "transition-rate absolute error",
            "run-length KS",
            "activity-rate absolute error",
        ],
        "lags": LAGS_162,
        "penalty_score": "lower is better; average of temporal mismatch penalties",
        "scope": "A0/A1/A2 protocol variants, authoritative no-Q4 scientific artifact, non-authoritative legacy Q4 context if present, and Q6 public candidate where present",
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
        "decomposition_done_here": True,
    },
    "outputs": {
        "variant_summary_csv": variant_summary_csv,
        "role_summary_csv": role_summary_csv,
        "column_metrics_csv": column_metrics_csv,
        "gain_loss_ledger_csv": gain_loss_ledger_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_162(contract_json, contract)
_write_json_162(contract_canonical_json, contract)

manifest = {
    "cell": "16.2",
    "version": CELL162_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
}

_write_json_162(manifest_json, manifest)

hashes = {
    "variant_summary_csv_sha256": _sha256_file_162(variant_summary_csv),
    "role_summary_csv_sha256": _sha256_file_162(role_summary_csv),
    "column_metrics_csv_sha256": _sha256_file_162(column_metrics_csv),
    "gain_loss_ledger_csv_sha256": _sha256_file_162(gain_loss_ledger_csv),
    "contract_json_sha256": _sha256_file_162(contract_json),
    "contract_canonical_json_sha256": _sha256_file_162(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_162(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_162(contract_json, contract)
_write_json_162(contract_canonical_json, contract)
_write_json_162(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals
# ----------------------------------------------------------
globals()["CELL162_VERSION"] = CELL162_VERSION
globals()["CELL16_2_Q2_COLUMN_METRICS_DF"] = column_metrics_df
globals()["CELL16_2_Q2_VARIANT_SUMMARY_DF"] = variant_summary_df
globals()["CELL16_2_Q2_ROLE_SUMMARY_DF"] = role_summary_df
globals()["CELL16_2_Q2_GAIN_LOSS_LEDGER_DF"] = gain_loss_ledger_df
globals()["CELL16_2_Q2_DECOMPOSITION_CONTRACT"] = contract

globals()["CELL16_2_Q2_COLUMN_METRICS_CSV"] = column_metrics_csv
globals()["CELL16_2_Q2_VARIANT_SUMMARY_CSV"] = variant_summary_csv
globals()["CELL16_2_Q2_ROLE_SUMMARY_CSV"] = role_summary_csv
globals()["CELL16_2_Q2_GAIN_LOSS_LEDGER_CSV"] = gain_loss_ledger_csv
globals()["CELL16_2_Q2_CONTRACT_JSON"] = contract_json
globals()["CELL16_2_Q2_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL16_2_Q2_MANIFEST_JSON"] = manifest_json

status_counts = column_metrics_df["q2_status"].astype(str).value_counts().sort_index().to_dict()

log(
    "[Cell16.2] Q2 temporal decomposition complete | "
    f"column_metrics={len(column_metrics_df)} | "
    f"variants={variant_summary_df['variant'].nunique() if len(variant_summary_df) else 0} | "
    f"status_counts={status_counts}"
)
if len(variant_summary_df):
    log(
        "[Cell16.2] Variant summary | "
        f"{variant_summary_df[['variant', 'artifact_role', 'cols_evaluable', 'pass_n', 'warning_n', 'fatal_n', 'q2_temporal_penalty_score']].to_dict('records')}"
    )
log(f"[Cell16.2] Saved variant summary: {variant_summary_csv}")
log(f"[Cell16.2] Saved role summary: {role_summary_csv}")
log(f"[Cell16.2] Saved column metrics: {column_metrics_csv}")
log(f"[Cell16.2] Saved gain/loss ledger: {gain_loss_ledger_csv}")
log(f"[Cell16.2] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell16.2] Contract flags | "
    "TEST_real_values_used_for_reference_only=True | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "decomposition_done_here=True | Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False"
)
log("--- END: Cell 16.2 - Q2 temporal decomposition (v1.1 no-Q4-promotion strict) ---")

gc.collect()