# ==========================================================
# CELL 16.3 - Q3 observability decomposition
# v1.1 STUDY-THESIS strict no-Q4-promotion observability-quality decomposition
#
# Role:
#   - Decompose Q3 observability/mask/staleness quality.
#   - Explain how observability realism contributes to full scientific artifact
#     and how public Q6 mitigation changes release scope.
#
# Scientific contract:
#   - No generation.
#   - No selection.
#   - No repair.
#   - No synthetic mutation.
#   - TEST real values are used only for decomposition/reference metrics.
#   - This cell reports Q3 decomposition; it does not change the pipeline.
#
# Outputs:
#   reports/cell16_3_q3_observability_decomposition_by_variant.csv
#   reports/cell16_3_q3_observability_decomposition_by_role.csv
#   reports/cell16_3_q3_observability_decomposition_by_column.csv
#   reports/cell16_3_q3_observability_gain_loss_ledger.csv
#   reports/cell16_3_q3_observability_decomposition_contract.json
#   artifacts/cell16_3_q3_observability_decomposition_manifest.json
# ==========================================================

log("--- START: Cell 16.3 - Q3 observability decomposition (v1.1 no-Q4-promotion strict) ---")

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
_required_163 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL16_0_ARTIFACT_REGISTRY_DF",
    "CELL16_0_METRIC_SOURCE_REGISTRY_DF",
    "CELL16_0_DECOMPOSITION_INPUT_CONTRACT",
    "CELL16_0_IOT_FULL_PATH",
    "CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH",
    "CELL16_0_PUBLIC_Q6_CPS_PATH",
    "CELL16_1_Q1_VARIANT_SUMMARY_DF",
    "CELL16_1_Q1_DECOMPOSITION_CONTRACT",
    "CELL16_2_Q2_VARIANT_SUMMARY_DF",
    "CELL16_2_Q2_DECOMPOSITION_CONTRACT",
]
_missing_163 = [k for k in _required_163 if k not in globals()]
if _missing_163:
    raise RuntimeError(f"[Cell16.3] Missing required globals: {_missing_163}")

ORIGINAL_OUTDIR_163 = str(OUTDIR)
ORIGINAL_OUT_SYN_163 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_163 = str(REPORT_DIR)

def _resolve_project_root_163(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell16.3] Could not resolve canonical project root.")

PROJECT_ROOT_163 = _resolve_project_root_163(ORIGINAL_OUTDIR_163, ORIGINAL_REPORT_DIR_163, ORIGINAL_OUT_SYN_163)
OUTDIR_BASE_163 = PROJECT_ROOT_163
OUT_SYN_BASE_163 = os.path.join(PROJECT_ROOT_163, "synthetic")
REPORT_DIR_BASE_163 = os.path.join(PROJECT_ROOT_163, "reports")
ARTDIR_BASE_163 = os.path.join(PROJECT_ROOT_163, "artifacts")
CONTRACT_DIR_BASE_163 = os.path.join(ARTDIR_BASE_163, "contracts")

os.makedirs(REPORT_DIR_BASE_163, exist_ok=True)
os.makedirs(ARTDIR_BASE_163, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_163, exist_ok=True)

SEED = int(SEED)
N_TE = int(len(df_te))

CELL163_VERSION = "cell16_3_q3_observability_decomposition_v1_1_no_q4_promotion"

CFG["cell16_3_version"] = CELL163_VERSION
CFG["cell16_3_TEST_real_values_used_for_reference_only"] = True
CFG["cell16_3_TEST_real_values_used_for_materialization"] = False
CFG["cell16_3_synthetic_values_mutated"] = False
CFG["cell16_3_selection_done_here"] = False
CFG["cell16_3_generator_fit_done_here"] = False
CFG["cell16_3_materialization_done_here"] = False
CFG["cell16_3_decomposition_done_here"] = True
CFG["cell16_3_Q4_final_status"] = "blocked_no_promotion"
CFG["cell16_3_Q4_coupled_artifacts_used"] = False

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell16_3_obs_rate_error_pass", 0.03)
CFG.setdefault("cell16_3_obs_rate_error_warning", 0.10)
CFG.setdefault("cell16_3_transition_error_pass", 0.02)
CFG.setdefault("cell16_3_transition_error_warning", 0.08)
CFG.setdefault("cell16_3_run_length_ks_pass", 0.15)
CFG.setdefault("cell16_3_run_length_ks_warning", 0.35)
CFG.setdefault("cell16_3_c2st_auc_pass", 0.60)
CFG.setdefault("cell16_3_c2st_auc_warning", 0.75)
CFG.setdefault("cell16_3_min_finite_rate", 0.001)
CFG.setdefault("cell16_3_max_columns_per_variant", 250)

OBS_RATE_PASS_163 = float(CFG.get("cell16_3_obs_rate_error_pass", 0.03))
OBS_RATE_WARN_163 = float(CFG.get("cell16_3_obs_rate_error_warning", 0.10))
TRANS_PASS_163 = float(CFG.get("cell16_3_transition_error_pass", 0.02))
TRANS_WARN_163 = float(CFG.get("cell16_3_transition_error_warning", 0.08))
RUN_KS_PASS_163 = float(CFG.get("cell16_3_run_length_ks_pass", 0.15))
RUN_KS_WARN_163 = float(CFG.get("cell16_3_run_length_ks_warning", 0.35))
C2ST_PASS_163 = float(CFG.get("cell16_3_c2st_auc_pass", 0.60))
C2ST_WARN_163 = float(CFG.get("cell16_3_c2st_auc_warning", 0.75))
MIN_FINITE_RATE_163 = float(CFG.get("cell16_3_min_finite_rate", 0.001))
MAX_COLS_163 = int(CFG.get("cell16_3_max_columns_per_variant", 250))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
variant_summary_csv = os.path.join(REPORT_DIR_BASE_163, "cell16_3_q3_observability_decomposition_by_variant.csv")
role_summary_csv = os.path.join(REPORT_DIR_BASE_163, "cell16_3_q3_observability_decomposition_by_role.csv")
column_metrics_csv = os.path.join(REPORT_DIR_BASE_163, "cell16_3_q3_observability_decomposition_by_column.csv")
gain_loss_ledger_csv = os.path.join(REPORT_DIR_BASE_163, "cell16_3_q3_observability_gain_loss_ledger.csv")
contract_json = os.path.join(REPORT_DIR_BASE_163, "cell16_3_q3_observability_decomposition_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_163, "cell16_3_q3_observability_decomposition_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_163, "cell16_3_q3_observability_decomposition_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_163(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_163(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_163(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_163(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_163(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_163(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_163(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_163(obj.to_dict())
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

def _write_json_163(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_163(payload), f, indent=2, sort_keys=True)

def _sha256_file_163(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_163(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _require_contract_governance_163(contract: dict, name: str, expected_version_substring: str):
    if not isinstance(contract, dict):
        raise RuntimeError(f"[Cell16.3] {name} is not a dict.")
    version = str(contract.get("version", ""))
    if expected_version_substring not in version:
        raise RuntimeError(
            f"[Cell16.3] Unexpected {name} version. "
            f"Expected substring={expected_version_substring}, got={version}"
        )
    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})
    status = str(q4.get("final_q4_status", strict.get("Q4_final_status", "")))
    used = bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True)))
    if status != "blocked_no_promotion":
        raise RuntimeError(f"[Cell16.3] {name} does not carry Q4 blocked_no_promotion governance.")
    if used:
        raise RuntimeError(f"[Cell16.3] {name} indicates Q4-coupled artifacts were used.")
    return version

def _read_parquet_optional_163(path: str, columns=None):
    if not _exists_163(path):
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return None

def _read_csv_optional_163(path: str):
    if not _exists_163(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _safe_float_163(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _is_time_col_163(c: str) -> bool:
    s = str(c).lower()
    return s in {"sec", "sec_epoch_s__canon", "timestamp", "time", "datetime"} or s.endswith("__time")

def _is_observability_col_163(c: str) -> bool:
    s = str(c).lower()
    if _is_time_col_163(s):
        return False
    return bool(
        "obs_present" in s
        or "traffic_present" in s
        or "entity_obs" in s
        or "entity_stale" in s
        or "stale" in s
        or s.endswith("__present")
        or s.endswith("__stale_flag")
        or s.endswith("__staleness_s")
        or s in {
            "iot__any_update_raw",
            "iot__tier_present",
            "router__obs_present",
            "zigbee__obs_present",
            "ota__obs_present",
            "ota24__obs_present",
            "ota5__obs_present",
            "zwave__obs_present",
        }
    )

def _obs_role_163(c: str) -> str:
    s = str(c)
    if s.startswith("router__"):
        return "protocol_router_observability"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__"):
        return "protocol_ota_observability"
    if s.startswith("zigbee__") or s.startswith("zb__"):
        return "protocol_zigbee_observability"
    if s.startswith("zwave__"):
        return "protocol_zwave_observability"
    if s.startswith("iot__entity_obs__"):
        return "iot_entity_observability"
    if s.startswith("iot__entity_stale__"):
        return "iot_entity_staleness"
    if s.startswith("iot__"):
        return "iot_global_observability"
    return "other_observability"

def _numeric_array_163(frame: pd.DataFrame, col: str):
    if not isinstance(frame, pd.DataFrame) or col not in frame.columns:
        return np.asarray([], dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)

def _binary_view_163(x):
    arr = np.asarray(x, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(arr)
    out[finite] = (arr[finite] > 0.5).astype(float)
    return out

def _finite_163(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]

def _ks_2samp_manual_163(a, b):
    a = np.sort(_finite_163(a))
    b = np.sort(_finite_163(b))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    vals = np.sort(np.unique(np.concatenate([a, b])))
    if len(vals) == 0:
        return np.nan
    cdf_a = np.searchsorted(a, vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, vals, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))

def _transition_rate_163(x):
    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.nan
    vals = x[idx]
    return float(np.mean(vals[1:] != vals[:-1]))

def _run_lengths_163(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    vals = x[finite]
    if vals.size == 0:
        return np.asarray([], dtype=np.float64)
    runs = []
    cur = vals[0]
    length = 1
    for v in vals[1:]:
        if v == cur:
            length += 1
        else:
            runs.append(length)
            cur = v
            length = 1
    runs.append(length)
    return np.asarray(runs, dtype=np.float64)

def _c2st_proxy_auc_163(real_x, syn_x):
    """
    Lightweight univariate C2ST proxy via normalized rank separation.
    AUC=0.5 means indistinguishable, 1.0 means separable.
    """
    r = _finite_163(real_x)
    s = _finite_163(syn_x)
    if len(r) == 0 or len(s) == 0:
        return np.nan
    vals = np.concatenate([r, s])
    labels = np.concatenate([np.zeros(len(r)), np.ones(len(s))])

    # Degenerate constant case.
    if np.nanmax(vals) <= np.nanmin(vals):
        return 0.5

    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(vals) + 1, dtype=np.float64)

    pos_ranks = ranks[labels == 1]
    n_pos = len(s)
    n_neg = len(r)
    if n_pos == 0 or n_neg == 0:
        return np.nan

    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / max(1.0, n_pos * n_neg)
    auc = float(auc)
    auc = max(auc, 1.0 - auc)
    return auc

def _q3_status_163(row):
    obs_err = _safe_float_163(row.get("obs_rate_abs_error"), np.nan)
    trans_err = _safe_float_163(row.get("transition_rate_abs_error"), np.nan)
    run_ks = _safe_float_163(row.get("run_length_ks"), np.nan)
    auc = _safe_float_163(row.get("c2st_proxy_auc"), np.nan)

    fatal = []
    warn = []

    if np.isfinite(obs_err):
        if obs_err > OBS_RATE_WARN_163:
            fatal.append("obs_rate_error_fatal")
        elif obs_err > OBS_RATE_PASS_163:
            warn.append("obs_rate_error_warning")

    if np.isfinite(trans_err):
        if trans_err > TRANS_WARN_163:
            fatal.append("transition_rate_error_fatal")
        elif trans_err > TRANS_PASS_163:
            warn.append("transition_rate_error_warning")

    if np.isfinite(run_ks):
        if run_ks > RUN_KS_WARN_163:
            fatal.append("run_length_ks_fatal")
        elif run_ks > RUN_KS_PASS_163:
            warn.append("run_length_ks_warning")

    if np.isfinite(auc):
        if auc > C2ST_WARN_163:
            fatal.append("c2st_proxy_auc_fatal")
        elif auc > C2ST_PASS_163:
            warn.append("c2st_proxy_auc_warning")

    if fatal:
        return "fatal", "|".join(fatal)
    if warn:
        return "warning", "|".join(warn)
    return "pass", "within_q3_observability_gates"

def _variant_score_163(g: pd.DataFrame):
    if not isinstance(g, pd.DataFrame) or len(g) == 0:
        return np.nan
    evaluable = g[~g["q3_status"].astype(str).eq("not_evaluable")].copy()
    if len(evaluable) == 0:
        return np.nan

    score = (
        pd.to_numeric(evaluable["obs_rate_abs_error"], errors="coerce").fillna(0.5)
        + pd.to_numeric(evaluable["transition_rate_abs_error"], errors="coerce").fillna(0.5)
        + pd.to_numeric(evaluable["run_length_ks"], errors="coerce").fillna(0.5)
        + (pd.to_numeric(evaluable["c2st_proxy_auc"], errors="coerce").fillna(0.75) - 0.5).clip(lower=0)
    )
    return float(np.nanmean(score))

def _choose_obs_cols_163(syn_df, real_df):
    syn_cols = list(map(str, syn_df.columns))
    real_cols = set(map(str, real_df.columns))
    cols = [c for c in syn_cols if c in real_cols and _is_observability_col_163(c)]
    if len(cols) <= MAX_COLS_163:
        return cols
    return sorted(cols)[:MAX_COLS_163]

def _compute_q3_column_metrics_163(variant_name, artifact_role, syn_df, real_df, candidate_cols):
    rows = []

    if syn_df is None or not isinstance(syn_df, pd.DataFrame):
        return rows

    syn_df = syn_df.copy()
    syn_df.columns = syn_df.columns.astype(str)

    for col in candidate_cols:
        if col not in syn_df.columns or col not in real_df.columns:
            continue

        s_raw = _numeric_array_163(syn_df, col)
        r_raw = _numeric_array_163(real_df, col)

        s = _binary_view_163(s_raw)
        r = _binary_view_163(r_raw)

        syn_finite_rate = float(np.isfinite(s).mean()) if len(s) else 0.0
        real_finite_rate = float(np.isfinite(r).mean()) if len(r) else 0.0

        if syn_finite_rate < MIN_FINITE_RATE_163 or real_finite_rate < MIN_FINITE_RATE_163:
            rows.append({
                "variant": variant_name,
                "artifact_role": artifact_role,
                "role_group": _obs_role_163(col),
                "col": col,
                "metric_scope": "Q3_observability",
                "q3_status": "not_evaluable",
                "q3_reasons": "low_finite_rate",
                "syn_finite_rate": syn_finite_rate,
                "real_finite_rate": real_finite_rate,
                "TEST_real_values_used_for_reference_only": True,
                "synthetic_values_mutated": False,
            })
            continue

        rf = _finite_163(r)
        sf = _finite_163(s)

        real_obs_rate = float(np.nanmean(rf)) if len(rf) else np.nan
        syn_obs_rate = float(np.nanmean(sf)) if len(sf) else np.nan
        obs_rate_abs_error = abs(syn_obs_rate - real_obs_rate) if np.isfinite(real_obs_rate) and np.isfinite(syn_obs_rate) else np.nan

        real_transition_rate = _transition_rate_163(r)
        syn_transition_rate = _transition_rate_163(s)
        transition_rate_abs_error = (
            abs(syn_transition_rate - real_transition_rate)
            if np.isfinite(real_transition_rate) and np.isfinite(syn_transition_rate)
            else np.nan
        )

        rruns = _run_lengths_163(r)
        sruns = _run_lengths_163(s)
        run_length_ks = _ks_2samp_manual_163(rruns, sruns)

        c2st_proxy_auc = _c2st_proxy_auc_163(r, s)

        row = {
            "variant": variant_name,
            "artifact_role": artifact_role,
            "role_group": _obs_role_163(col),
            "col": col,
            "metric_scope": "Q3_observability",
            "syn_finite_rate": syn_finite_rate,
            "real_finite_rate": real_finite_rate,
            "real_obs_rate": real_obs_rate,
            "syn_obs_rate": syn_obs_rate,
            "obs_rate_abs_error": obs_rate_abs_error,
            "real_transition_rate": real_transition_rate,
            "syn_transition_rate": syn_transition_rate,
            "transition_rate_abs_error": transition_rate_abs_error,
            "run_length_ks": run_length_ks,
            "c2st_proxy_auc": c2st_proxy_auc,
            "real_run_count": int(len(rruns)),
            "syn_run_count": int(len(sruns)),
            "TEST_real_values_used_for_reference_only": True,
            "synthetic_values_mutated": False,
        }

        status, reasons = _q3_status_163(row)
        row["q3_status"] = status
        row["q3_reasons"] = reasons
        rows.append(row)

    return rows

# ----------------------------------------------------------
# 4) Validate upstream no-Q4 governance contracts
# ----------------------------------------------------------
CELL16_0_CONTRACT_VERSION_163 = _require_contract_governance_163(
    CELL16_0_DECOMPOSITION_INPUT_CONTRACT,
    "CELL16_0_DECOMPOSITION_INPUT_CONTRACT",
    "cell16_0_a0_a1_a2_decomposition_input_contract_v1_1",
)
CELL16_1_CONTRACT_VERSION_163 = _require_contract_governance_163(
    CELL16_1_Q1_DECOMPOSITION_CONTRACT,
    "CELL16_1_Q1_DECOMPOSITION_CONTRACT",
    "cell16_1_q1_marginal_decomposition_v1_1",
)
CELL16_2_CONTRACT_VERSION_163 = _require_contract_governance_163(
    CELL16_2_Q2_DECOMPOSITION_CONTRACT,
    "CELL16_2_Q2_DECOMPOSITION_CONTRACT",
    "cell16_2_q2_temporal_decomposition_v1_1",
)

# ----------------------------------------------------------
# 5) Load optional upstream Q3 reports
# ----------------------------------------------------------
# These reports are optional. If absent, Cell 16.3 must still compute
# artifact-level Q3 decomposition from the available artifacts.
#
# Important fix:
# _read_csv_optional_163() returns None when a file is missing/unreadable.
# Normalize None -> empty DataFrame before any len(...) or column access.

q3_report_candidates_163 = [
    os.path.join(REPORT_DIR_BASE_163, "cell12f6_observability_final_test_qa_metrics.csv"),
    os.path.join(REPORT_DIR_BASE_163, "cell12f6_mask_final_test_qa_metrics.csv"),
    os.path.join(REPORT_DIR_BASE_163, "cell12f5_mask_final_test_qa_metrics.csv"),
    os.path.join(REPORT_DIR_BASE_163, "cell12f5_observability_final_test_qa_metrics.csv"),
    os.path.join(REPORT_DIR_BASE_163, "cell12f4_mask_final_test_qa_metrics.csv"),
]

q3_status_candidates_163 = [
    os.path.join(REPORT_DIR_BASE_163, "cell12f6_observability_publication_status.csv"),
    os.path.join(REPORT_DIR_BASE_163, "cell12f6_mask_publication_status.csv"),
    os.path.join(REPORT_DIR_BASE_163, "cell12f5_mask_publication_status.csv"),
    os.path.join(REPORT_DIR_BASE_163, "cell12f5_observability_publication_status.csv"),
    os.path.join(REPORT_DIR_BASE_163, "cell12f4_mask_publication_status.csv"),
]

def _read_first_existing_csv_163(paths):
    audit = []
    for p in paths:
        exists = _exists_163(p)
        df = _read_csv_optional_163(p) if exists else None
        loaded = isinstance(df, pd.DataFrame)
        audit.append({
            "path": p,
            "exists": bool(exists),
            "loaded": bool(loaded),
            "rows": int(len(df)) if loaded else 0,
            "cols": int(df.shape[1]) if loaded else 0,
        })
        if loaded:
            return df, p, audit
    return pd.DataFrame(), "", audit

q3_report_df, q3_report_loaded_path_163, q3_report_load_audit_163 = _read_first_existing_csv_163(
    q3_report_candidates_163
)
q3_status_df, q3_status_loaded_path_163, q3_status_load_audit_163 = _read_first_existing_csv_163(
    q3_status_candidates_163
)

# Defensive normalization. Keep this even after _read_first_existing_csv_163.
if not isinstance(q3_report_df, pd.DataFrame):
    q3_report_df = pd.DataFrame()
if not isinstance(q3_status_df, pd.DataFrame):
    q3_status_df = pd.DataFrame()

q3_report_loaded_163 = bool(len(q3_report_df) > 0)
q3_status_loaded_163 = bool(len(q3_status_df) > 0)

log(
    "[Cell16.3] Optional upstream Q3 reports | "
    f"report_loaded={q3_report_loaded_163} | report_path={q3_report_loaded_path_163 or 'none'} | "
    f"status_loaded={q3_status_loaded_163} | status_path={q3_status_loaded_path_163 or 'none'}"
)

# ----------------------------------------------------------
# 6) Compute artifact-level Q3 decomposition
# ----------------------------------------------------------
real_test_df_163 = df_te.copy()
real_test_df_163.columns = real_test_df_163.columns.astype(str)

variant_specs = []

def _add_variant_spec_163(variant, artifact_role, path, required=False):
    if _exists_163(path):
        variant_specs.append({
            "variant": variant,
            "artifact_role": artifact_role,
            "path": path,
            "exists": True,
            "required": bool(required),
        })
    elif required:
        raise RuntimeError(f"[Cell16.3] Missing required variant artifact: {variant} | {path}")
    else:
        variant_specs.append({
            "variant": variant,
            "artifact_role": artifact_role,
            "path": path,
            "exists": False,
            "required": bool(required),
        })

_add_variant_spec_163("IOT_full_12g", "iot_full_namespace", CELL16_0_IOT_FULL_PATH, required=False)
_add_variant_spec_163("scientific_no_q4_cps", "scientific_cps_no_q4", CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH, required=True)

# Optional non-authoritative legacy Q4 context only.
if isinstance(globals().get("CELL16_0_LEGACY_Q4_CPS_PATH", ""), str) and _exists_163(globals().get("CELL16_0_LEGACY_Q4_CPS_PATH", "")):
    _add_variant_spec_163("legacy_q4_cps_context", "legacy_q4_cps_not_authoritative", globals()["CELL16_0_LEGACY_Q4_CPS_PATH"], required=False)

_add_variant_spec_163("Q6_public_cps", "public_q6_mitigated_cps", CELL16_0_PUBLIC_Q6_CPS_PATH, required=False)

log(
    "[Cell16.3] Computing Q3 observability decomposition | "
    f"variants={[v['variant'] for v in variant_specs if v['exists']]}"
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

    syn_df = _read_parquet_optional_163(path)
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
            f"[Cell16.3] Row mismatch for variant={variant}: got={len(syn_df)} expected={N_TE}"
        )

    candidate_cols = _choose_obs_cols_163(syn_df, real_test_df_163)

    rows = _compute_q3_column_metrics_163(
        variant_name=variant,
        artifact_role=artifact_role,
        syn_df=syn_df,
        real_df=real_test_df_163,
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
        "candidate_observability_cols": int(len(candidate_cols)),
        "evaluated_metric_rows": int(len(rows)),
        "reason": "ok",
    })

    del syn_df
    gc.collect()

column_metrics_df = pd.DataFrame(all_column_rows)
variant_load_df = pd.DataFrame(variant_load_rows)

if len(column_metrics_df) == 0:
    # Public candidate may intentionally have no Q3 observability cols, but full/Q4 should not.
    raise RuntimeError("[Cell16.3] No Q3 observability decomposition metrics were produced.")

# ----------------------------------------------------------
# 7) Summaries
# ----------------------------------------------------------
def _summary_from_group_163(g: pd.DataFrame):
    status_counts = g["q3_status"].astype(str).value_counts().to_dict()
    evaluable = g[~g["q3_status"].astype(str).eq("not_evaluable")].copy()

    return pd.Series({
        "cols_total": int(len(g)),
        "cols_evaluable": int(len(evaluable)),
        "pass_n": int(status_counts.get("pass", 0)),
        "warning_n": int(status_counts.get("warning", 0)),
        "fatal_n": int(status_counts.get("fatal", 0)),
        "not_evaluable_n": int(status_counts.get("not_evaluable", 0)),
        "pass_rate_evaluable": float((evaluable["q3_status"].astype(str).eq("pass")).mean()) if len(evaluable) else np.nan,
        "fatal_rate_evaluable": float((evaluable["q3_status"].astype(str).eq("fatal")).mean()) if len(evaluable) else np.nan,
        "mean_obs_rate_abs_error": float(pd.to_numeric(evaluable["obs_rate_abs_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "median_obs_rate_abs_error": float(pd.to_numeric(evaluable["obs_rate_abs_error"], errors="coerce").median()) if len(evaluable) else np.nan,
        "mean_transition_rate_abs_error": float(pd.to_numeric(evaluable["transition_rate_abs_error"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_run_length_ks": float(pd.to_numeric(evaluable["run_length_ks"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_c2st_proxy_auc": float(pd.to_numeric(evaluable["c2st_proxy_auc"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "q3_observability_penalty_score": _variant_score_163(evaluable),
    })

variant_summary_df = (
    column_metrics_df
    .groupby(["variant", "artifact_role"], dropna=False)
    .apply(_summary_from_group_163)
    .reset_index()
    .sort_values(["q3_observability_penalty_score", "variant"], ascending=[True, True])
    .reset_index(drop=True)
)

role_summary_df = (
    column_metrics_df
    .groupby(["variant", "artifact_role", "role_group"], dropna=False)
    .apply(_summary_from_group_163)
    .reset_index()
    .sort_values(["variant", "role_group"])
    .reset_index(drop=True)
)

# Add explicit public candidate no-observability row if applicable.
public_loaded = variant_load_df[
    variant_load_df["variant"].astype(str).eq("Q6_public_cps")
    & variant_load_df["loaded"].astype(bool)
]
if len(public_loaded) and int(public_loaded.iloc[0].get("evaluated_metric_rows", 0)) == 0:
    variant_summary_df = pd.concat([
        variant_summary_df,
        pd.DataFrame([{
            "variant": "Q6_public_cps",
            "artifact_role": "public_q6_mitigated_cps",
            "cols_total": 0,
            "cols_evaluable": 0,
            "pass_n": 0,
            "warning_n": 0,
            "fatal_n": 0,
            "not_evaluable_n": 0,
            "pass_rate_evaluable": np.nan,
            "fatal_rate_evaluable": np.nan,
            "mean_obs_rate_abs_error": np.nan,
            "median_obs_rate_abs_error": np.nan,
            "mean_transition_rate_abs_error": np.nan,
            "mean_run_length_ks": np.nan,
            "mean_c2st_proxy_auc": np.nan,
            "q3_observability_penalty_score": np.nan,
            "interpretation": "Public Q6 candidate intentionally excludes fine-grained observability columns.",
        }])
    ], axis=0, ignore_index=True)

# ----------------------------------------------------------
# 8) Gain/loss ledger
# ----------------------------------------------------------
ledger_rows = []

# Optional upstream Q3 summary. Absence is not fatal because this cell computes
# its own artifact-level Q3 decomposition.
upstream_q3_source_df_163 = pd.DataFrame()
upstream_q3_source_name_163 = ""

if isinstance(q3_report_df, pd.DataFrame) and len(q3_report_df) > 0:
    upstream_q3_source_df_163 = q3_report_df
    upstream_q3_source_name_163 = "upstream_q3_report"
elif isinstance(q3_status_df, pd.DataFrame) and len(q3_status_df) > 0:
    upstream_q3_source_df_163 = q3_status_df
    upstream_q3_source_name_163 = "upstream_q3_status"

if len(upstream_q3_source_df_163) > 0:
    q3_status_col = None
    for c in [
        "mask_test_qa_status",
        "publication_status",
        "q3_status",
        "mask_publication_status",
        "observability_publication_status",
        "status",
    ]:
        if c in upstream_q3_source_df_163.columns:
            q3_status_col = c
            break

    if q3_status_col:
        counts = upstream_q3_source_df_163[q3_status_col].astype(str).value_counts().to_dict()
        ledger_rows.append({
            "ledger_item": "upstream_q3_status_counts",
            "variant": "IOT_full_12g",
            "artifact_role": upstream_q3_source_name_163,
            "source_path": (
                q3_report_loaded_path_163
                if upstream_q3_source_name_163 == "upstream_q3_report"
                else q3_status_loaded_path_163
            ),
            "status_column": q3_status_col,
            "pass_n": int(counts.get("pass", 0)),
            "warning_n": int(counts.get("warning", 0)),
            "fatal_n": int(counts.get("fatal", 0)),
            "blocker_n": int(counts.get("blocker", 0)),
            "not_evaluable_n": int(counts.get("not_evaluable", 0)),
            "interpretation": "Status counts from optional upstream Q3 observability QA/status report.",
        })
    else:
        ledger_rows.append({
            "ledger_item": "upstream_q3_status_counts_unavailable",
            "variant": "IOT_full_12g",
            "artifact_role": upstream_q3_source_name_163,
            "source_path": (
                q3_report_loaded_path_163
                if upstream_q3_source_name_163 == "upstream_q3_report"
                else q3_status_loaded_path_163
            ),
            "interpretation": "Upstream Q3 report/status file was found, but no recognized status column was available.",
        })
else:
    ledger_rows.append({
        "ledger_item": "upstream_q3_report_absent",
        "variant": "IOT_full_12g",
        "artifact_role": "upstream_q3_report_optional",
        "pass_n": np.nan,
        "warning_n": np.nan,
        "fatal_n": np.nan,
        "blocker_n": np.nan,
        "interpretation": (
            "Optional upstream Q3 report/status files were not found. "
            "Cell 16.3 computed artifact-level Q3 decomposition directly from the available artifacts."
        ),
    })

for _, r in variant_summary_df.iterrows():
    ledger_rows.append({
        "ledger_item": "q3_variant_summary",
        "variant": str(r.get("variant", "")),
        "artifact_role": str(r.get("artifact_role", "")),
        "cols_evaluable": _safe_float_163(r.get("cols_evaluable"), np.nan),
        "pass_n": _safe_float_163(r.get("pass_n"), np.nan),
        "warning_n": _safe_float_163(r.get("warning_n"), np.nan),
        "fatal_n": _safe_float_163(r.get("fatal_n"), np.nan),
        "q3_observability_penalty_score": _safe_float_163(r.get("q3_observability_penalty_score"), np.nan),
        "interpretation": str(r.get("interpretation", "Q3 observability decomposition variant summary.")),
    })

# Worst observability columns in full scientific artifact.
worst = column_metrics_df[
    column_metrics_df["variant"].astype(str).isin(["IOT_full_12g", "scientific_no_q4_cps", "legacy_q4_cps_context"])
    & column_metrics_df["q3_status"].astype(str).isin(["fatal", "warning"])
].copy()

if len(worst):
    worst["q3_column_penalty"] = (
        pd.to_numeric(worst["obs_rate_abs_error"], errors="coerce").fillna(0.5)
        + pd.to_numeric(worst["transition_rate_abs_error"], errors="coerce").fillna(0.5)
        + pd.to_numeric(worst["run_length_ks"], errors="coerce").fillna(0.5)
        + (pd.to_numeric(worst["c2st_proxy_auc"], errors="coerce").fillna(0.75) - 0.5).clip(lower=0)
    )

    for _, r in worst.sort_values("q3_column_penalty", ascending=False).head(30).iterrows():
        ledger_rows.append({
            "ledger_item": "worst_q3_observability_column",
            "variant": str(r["variant"]),
            "artifact_role": str(r["artifact_role"]),
            "role_group": str(r["role_group"]),
            "col": str(r["col"]),
            "q3_status": str(r["q3_status"]),
            "q3_reasons": str(r["q3_reasons"]),
            "q3_column_penalty": _safe_float_163(r["q3_column_penalty"], np.nan),
            "obs_rate_abs_error": _safe_float_163(r["obs_rate_abs_error"], np.nan),
            "transition_rate_abs_error": _safe_float_163(r["transition_rate_abs_error"], np.nan),
            "run_length_ks": _safe_float_163(r["run_length_ks"], np.nan),
            "c2st_proxy_auc": _safe_float_163(r["c2st_proxy_auc"], np.nan),
            "interpretation": "High-priority Q3 observability mismatch.",
        })

gain_loss_ledger_df = pd.DataFrame(ledger_rows)

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
column_metrics_df.to_csv(column_metrics_csv, index=False)
variant_summary_df.to_csv(variant_summary_csv, index=False)
role_summary_df.to_csv(role_summary_csv, index=False)
gain_loss_ledger_df.to_csv(gain_loss_ledger_csv, index=False)

# ----------------------------------------------------------
# 10) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "16.3",
    "version": CELL163_VERSION,
    "role": "q3_observability_decomposition_no_q4_promotion",
    "quality_dimension": "Q3_observability_mask_staleness_quality",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "legacy_q4_variants_non_authoritative": True
    },
    "upstream_contract_versions": {
        "cell16_0": CELL16_0_CONTRACT_VERSION_163,
        "cell16_1": CELL16_1_CONTRACT_VERSION_163,
        "cell16_2": CELL16_2_CONTRACT_VERSION_163
    },
    "summary": {
        "variants_loaded": variant_load_df.to_dict("records"),
        "column_metric_rows": int(len(column_metrics_df)),
        "variant_summary_rows": int(len(variant_summary_df)),
        "role_summary_rows": int(len(role_summary_df)),
        "gain_loss_ledger_rows": int(len(gain_loss_ledger_df)),
        "upstream_q3_report_loaded": bool(q3_report_loaded_163),
        "upstream_q3_report_loaded_path": q3_report_loaded_path_163,
        "upstream_q3_status_loaded": bool(q3_status_loaded_163),
        "upstream_q3_status_loaded_path": q3_status_loaded_path_163,
        "upstream_q3_report_load_audit": q3_report_load_audit_163,
        "upstream_q3_status_load_audit": q3_status_load_audit_163,
        "best_variant_by_q3_observability_penalty": (
            variant_summary_df.sort_values("q3_observability_penalty_score", na_position="last").iloc[0].to_dict()
            if len(variant_summary_df)
            else {}
        ),
    },
    "method": {
        "metrics": [
            "observability rate absolute error",
            "transition-rate absolute error",
            "run-length KS",
            "univariate C2ST proxy AUC",
        ],
        "public_candidate_policy": "absence of observability columns is interpreted as role restriction, not Q3 failure",
        "penalty_score": "lower is better; average of observability mismatch penalties",
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

_write_json_163(contract_json, contract)
_write_json_163(contract_canonical_json, contract)

manifest = {
    "cell": "16.3",
    "version": CELL163_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
}

_write_json_163(manifest_json, manifest)

hashes = {
    "variant_summary_csv_sha256": _sha256_file_163(variant_summary_csv),
    "role_summary_csv_sha256": _sha256_file_163(role_summary_csv),
    "column_metrics_csv_sha256": _sha256_file_163(column_metrics_csv),
    "gain_loss_ledger_csv_sha256": _sha256_file_163(gain_loss_ledger_csv),
    "contract_json_sha256": _sha256_file_163(contract_json),
    "contract_canonical_json_sha256": _sha256_file_163(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_163(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_163(contract_json, contract)
_write_json_163(contract_canonical_json, contract)
_write_json_163(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL163_VERSION"] = CELL163_VERSION
globals()["CELL16_3_Q3_COLUMN_METRICS_DF"] = column_metrics_df
globals()["CELL16_3_Q3_VARIANT_SUMMARY_DF"] = variant_summary_df
globals()["CELL16_3_Q3_ROLE_SUMMARY_DF"] = role_summary_df
globals()["CELL16_3_Q3_GAIN_LOSS_LEDGER_DF"] = gain_loss_ledger_df
globals()["CELL16_3_Q3_DECOMPOSITION_CONTRACT"] = contract

globals()["CELL16_3_Q3_COLUMN_METRICS_CSV"] = column_metrics_csv
globals()["CELL16_3_Q3_VARIANT_SUMMARY_CSV"] = variant_summary_csv
globals()["CELL16_3_Q3_ROLE_SUMMARY_CSV"] = role_summary_csv
globals()["CELL16_3_Q3_GAIN_LOSS_LEDGER_CSV"] = gain_loss_ledger_csv
globals()["CELL16_3_Q3_CONTRACT_JSON"] = contract_json
globals()["CELL16_3_Q3_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL16_3_Q3_MANIFEST_JSON"] = manifest_json

status_counts = column_metrics_df["q3_status"].astype(str).value_counts().sort_index().to_dict() if len(column_metrics_df) else {}

log(
    "[Cell16.3] Q3 observability decomposition complete | "
    f"column_metrics={len(column_metrics_df)} | "
    f"variants={variant_summary_df['variant'].nunique() if len(variant_summary_df) else 0} | "
    f"status_counts={status_counts}"
)
if len(variant_summary_df):
    cols_show = [
        "variant", "artifact_role", "cols_evaluable", "pass_n",
        "warning_n", "fatal_n", "q3_observability_penalty_score"
    ]
    show = [c for c in cols_show if c in variant_summary_df.columns]
    log(f"[Cell16.3] Variant summary | {variant_summary_df[show].to_dict('records')}")
log(f"[Cell16.3] Saved variant summary: {variant_summary_csv}")
log(f"[Cell16.3] Saved role summary: {role_summary_csv}")
log(f"[Cell16.3] Saved column metrics: {column_metrics_csv}")
log(f"[Cell16.3] Saved gain/loss ledger: {gain_loss_ledger_csv}")
log(f"[Cell16.3] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell16.3] Contract flags | "
    "TEST_real_values_used_for_reference_only=True | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "decomposition_done_here=True | Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False"
)
log("--- END: Cell 16.3 - Q3 observability decomposition (v1.1 no-Q4-promotion strict) ---")

gc.collect()