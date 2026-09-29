# ==========================================================
# CELL 16.1 - Q1 marginal decomposition
# v1.1 STUDY-THESIS strict no-Q4-promotion marginal-quality decomposition
#
# Role:
#   - Decompose marginal/distributional quality across A0/A1/A2/final branches.
#   - Explain where Q1 gains/failures come from:
#       protocol A0/A1/A2 variants
#       IoT continuous/value branch
#       IoT binary branch
#       IoT observability/masks where applicable
#       authoritative no-Q4 scientific artifact
#       public Q6-mitigated artifact as release-context variant
#
# Scientific contract:
#   - No generation.
#   - No selection.
#   - No repair.
#   - No synthetic mutation.
#   - TEST real values may be used only for decomposition/reference metrics.
#   - This cell reports Q1 decomposition; it does not change the pipeline.
#
# Outputs:
#   reports/cell16_1_q1_marginal_decomposition_by_variant.csv
#   reports/cell16_1_q1_marginal_decomposition_by_role.csv
#   reports/cell16_1_q1_marginal_decomposition_by_column.csv
#   reports/cell16_1_q1_marginal_gain_loss_ledger.csv
#   reports/cell16_1_q1_marginal_decomposition_contract.json
#   artifacts/cell16_1_q1_marginal_decomposition_manifest.json
# ==========================================================

log("--- START: Cell 16.1 - Q1 marginal decomposition (v1.1 no-Q4-promotion strict) ---")

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
_required_161 = [
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
]
_missing_161 = [k for k in _required_161 if k not in globals()]
if _missing_161:
    raise RuntimeError(f"[Cell16.1] Missing required globals: {_missing_161}")

ORIGINAL_OUTDIR_161 = str(OUTDIR)
ORIGINAL_OUT_SYN_161 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_161 = str(REPORT_DIR)

def _resolve_project_root_161(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell16.1] Could not resolve canonical project root.")

PROJECT_ROOT_161 = _resolve_project_root_161(ORIGINAL_OUTDIR_161, ORIGINAL_REPORT_DIR_161, ORIGINAL_OUT_SYN_161)
OUTDIR_BASE_161 = PROJECT_ROOT_161
OUT_SYN_BASE_161 = os.path.join(PROJECT_ROOT_161, "synthetic")
REPORT_DIR_BASE_161 = os.path.join(PROJECT_ROOT_161, "reports")
ARTDIR_BASE_161 = os.path.join(PROJECT_ROOT_161, "artifacts")
CONTRACT_DIR_BASE_161 = os.path.join(ARTDIR_BASE_161, "contracts")

os.makedirs(REPORT_DIR_BASE_161, exist_ok=True)
os.makedirs(ARTDIR_BASE_161, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_161, exist_ok=True)

SEED = int(SEED)
N_TE = int(len(df_te))

CELL161_VERSION = "cell16_1_q1_marginal_decomposition_v1_1_no_q4_promotion"

CFG["cell16_1_version"] = CELL161_VERSION
CFG["cell16_1_TEST_real_values_used_for_reference_only"] = True
CFG["cell16_1_TEST_real_values_used_for_materialization"] = False
CFG["cell16_1_synthetic_values_mutated"] = False
CFG["cell16_1_selection_done_here"] = False
CFG["cell16_1_generator_fit_done_here"] = False
CFG["cell16_1_materialization_done_here"] = False
CFG["cell16_1_decomposition_done_here"] = True
CFG["cell16_1_Q4_final_status"] = "blocked_no_promotion"
CFG["cell16_1_Q4_coupled_artifacts_used"] = False

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell16_1_max_columns_per_variant", 250)
CFG.setdefault("cell16_1_numeric_min_finite_rate", 0.001)
CFG.setdefault("cell16_1_hist_bins", 50)
CFG.setdefault("cell16_1_ks_pass", 0.10)
CFG.setdefault("cell16_1_ks_warning", 0.25)
CFG.setdefault("cell16_1_wass_norm_pass", 0.25)
CFG.setdefault("cell16_1_wass_norm_warning", 0.75)
CFG.setdefault("cell16_1_mean_error_norm_pass", 0.25)
CFG.setdefault("cell16_1_mean_error_norm_warning", 0.75)
CFG.setdefault("cell16_1_std_ratio_log_pass", 0.25)
CFG.setdefault("cell16_1_std_ratio_log_warning", 0.75)
CFG.setdefault("cell16_1_support_jaccard_pass", 0.70)
CFG.setdefault("cell16_1_support_jaccard_warning", 0.40)
CFG.setdefault("cell16_1_low_cardinality_unique_threshold", 30)
CFG.setdefault("cell16_1_load_large_variants", True)

MAX_COLS_161 = int(CFG.get("cell16_1_max_columns_per_variant", 250))
MIN_FINITE_RATE_161 = float(CFG.get("cell16_1_numeric_min_finite_rate", 0.001))
HIST_BINS_161 = int(CFG.get("cell16_1_hist_bins", 50))
LOW_CARD_UNIQUE_161 = int(CFG.get("cell16_1_low_cardinality_unique_threshold", 30))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
variant_summary_csv = os.path.join(REPORT_DIR_BASE_161, "cell16_1_q1_marginal_decomposition_by_variant.csv")
role_summary_csv = os.path.join(REPORT_DIR_BASE_161, "cell16_1_q1_marginal_decomposition_by_role.csv")
column_metrics_csv = os.path.join(REPORT_DIR_BASE_161, "cell16_1_q1_marginal_decomposition_by_column.csv")
gain_loss_ledger_csv = os.path.join(REPORT_DIR_BASE_161, "cell16_1_q1_marginal_gain_loss_ledger.csv")
contract_json = os.path.join(REPORT_DIR_BASE_161, "cell16_1_q1_marginal_decomposition_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_161, "cell16_1_q1_marginal_decomposition_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_161, "cell16_1_q1_marginal_decomposition_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_161(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_161(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_161(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_161(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_161(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_161(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_161(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_161(obj.to_dict())
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

def _write_json_161(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_161(payload), f, indent=2, sort_keys=True)

def _sha256_file_161(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_161(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _require_contract_governance_161(contract: dict):
    if not isinstance(contract, dict):
        raise RuntimeError("[Cell16.1] CELL16_0_DECOMPOSITION_INPUT_CONTRACT is not a dict.")
    version = str(contract.get("version", ""))
    if "cell16_0_a0_a1_a2_decomposition_input_contract_v1_1" not in version:
        raise RuntimeError(f"[Cell16.1] Unexpected Cell16.0 contract version: {version}")
    q4 = contract.get("q4_governance", {})
    if str(q4.get("final_q4_status", "")) != "blocked_no_promotion":
        raise RuntimeError("[Cell16.1] Cell16.0 does not carry Q4 blocked_no_promotion governance.")
    if bool(q4.get("q4_coupled_artifacts_used", True)):
        raise RuntimeError("[Cell16.1] Cell16.0 indicates Q4-coupled artifacts were used.")
    return version

def _read_parquet_optional_161(path: str, columns=None):
    if not _exists_161(path):
        return None
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return None

def _read_csv_optional_161(path: str):
    if not _exists_161(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _safe_float_161(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_int_161(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _numeric_array_161(frame: pd.DataFrame, col: str):
    if not isinstance(frame, pd.DataFrame) or col not in frame.columns:
        return np.asarray([], dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)

def _finite_161(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]

def _is_time_col_161(c: str) -> bool:
    s = str(c).lower()
    return s in {"sec", "sec_epoch_s__canon", "timestamp", "time", "datetime"} or s.endswith("__time")

def _tier_or_role_161(col: str) -> str:
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
    if _is_time_col_161(s):
        return "time"
    return "other"

def _ks_2samp_manual_161(a, b):
    a = np.sort(_finite_161(a))
    b = np.sort(_finite_161(b))
    if len(a) == 0 or len(b) == 0:
        return np.nan
    vals = np.sort(np.unique(np.concatenate([a, b])))
    if len(vals) == 0:
        return np.nan
    cdf_a = np.searchsorted(a, vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, vals, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))

def _wasserstein_1d_161(a, b):
    a = _finite_161(a)
    b = _finite_161(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    qs = np.linspace(0.01, 0.99, 99)
    qa = np.nanquantile(a, qs)
    qb = np.nanquantile(b, qs)
    return float(np.nanmean(np.abs(qa - qb)))

def _support_jaccard_161(a, b):
    a = _finite_161(a)
    b = _finite_161(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan

    # Low-card exact support.
    au = np.unique(a)
    bu = np.unique(b)
    if len(au) <= LOW_CARD_UNIQUE_161 and len(bu) <= LOW_CARD_UNIQUE_161:
        sa = set(map(float, au))
        sb = set(map(float, bu))
        if not sa and not sb:
            return np.nan
        return float(len(sa & sb) / max(1, len(sa | sb)))

    # Continuous range-overlap proxy.
    amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
    bmin, bmax = float(np.nanmin(b)), float(np.nanmax(b))
    inter = max(0.0, min(amax, bmax) - max(amin, bmin))
    union = max(1e-12, max(amax, bmax) - min(amin, bmin))
    return float(inter / union)

def _hist_tv_161(a, b, bins=50):
    a = _finite_161(a)
    b = _finite_161(b)

    if len(a) == 0 or len(b) == 0:
        return np.nan

    combined = np.concatenate([a, b])
    lo = float(np.nanmin(combined))
    hi = float(np.nanmax(combined))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan

    # Degenerate constant/range-zero case.
    # Do NOT call np.allclose(a, b), because real/synthetic arrays can have different lengths.
    if hi <= lo:
        a_val = float(np.nanmedian(a)) if len(a) else np.nan
        b_val = float(np.nanmedian(b)) if len(b) else np.nan
        if np.isfinite(a_val) and np.isfinite(b_val) and np.isclose(a_val, b_val, rtol=1e-8, atol=1e-10):
            return 0.0
        return 1.0

    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)

    pa = ha / max(1, ha.sum())
    pb = hb / max(1, hb.sum())

    return float(0.5 * np.sum(np.abs(pa - pb)))

def _q1_status_161(row):
    ks = _safe_float_161(row.get("ks"), np.nan)
    wn = _safe_float_161(row.get("wasserstein_norm"), np.nan)
    me = _safe_float_161(row.get("mean_error_norm"), np.nan)
    sl = _safe_float_161(row.get("std_ratio_log_abs"), np.nan)
    sj = _safe_float_161(row.get("support_jaccard"), np.nan)

    fatal = []
    warn = []

    if np.isfinite(ks):
        if ks > float(CFG.get("cell16_1_ks_warning", 0.25)):
            fatal.append("ks_fatal")
        elif ks > float(CFG.get("cell16_1_ks_pass", 0.10)):
            warn.append("ks_warning")

    if np.isfinite(wn):
        if wn > float(CFG.get("cell16_1_wass_norm_warning", 0.75)):
            fatal.append("wasserstein_norm_fatal")
        elif wn > float(CFG.get("cell16_1_wass_norm_pass", 0.25)):
            warn.append("wasserstein_norm_warning")

    if np.isfinite(me):
        if me > float(CFG.get("cell16_1_mean_error_norm_warning", 0.75)):
            fatal.append("mean_error_norm_fatal")
        elif me > float(CFG.get("cell16_1_mean_error_norm_pass", 0.25)):
            warn.append("mean_error_norm_warning")

    if np.isfinite(sl):
        if sl > float(CFG.get("cell16_1_std_ratio_log_warning", 0.75)):
            fatal.append("std_ratio_fatal")
        elif sl > float(CFG.get("cell16_1_std_ratio_log_pass", 0.25)):
            warn.append("std_ratio_warning")

    if np.isfinite(sj):
        if sj < float(CFG.get("cell16_1_support_jaccard_warning", 0.40)):
            fatal.append("support_jaccard_fatal")
        elif sj < float(CFG.get("cell16_1_support_jaccard_pass", 0.70)):
            warn.append("support_jaccard_warning")

    if fatal:
        return "fatal", "|".join(fatal)
    if warn:
        return "warning", "|".join(warn)
    return "pass", "within_q1_marginal_gates"

def _variant_score_161(g: pd.DataFrame):
    if not isinstance(g, pd.DataFrame) or len(g) == 0:
        return np.nan
    ks = pd.to_numeric(g.get("ks"), errors="coerce")
    wn = pd.to_numeric(g.get("wasserstein_norm"), errors="coerce")
    me = pd.to_numeric(g.get("mean_error_norm"), errors="coerce")
    sl = pd.to_numeric(g.get("std_ratio_log_abs"), errors="coerce")
    sj = pd.to_numeric(g.get("support_jaccard"), errors="coerce")

    penalty = (
        ks.fillna(0.5)
        + wn.fillna(1.0)
        + me.fillna(1.0)
        + sl.fillna(1.0)
        + (1.0 - sj.fillna(0.0))
    )
    return float(np.nanmean(penalty))

def _compute_q1_column_metrics_161(variant_name, artifact_role, syn_df, real_df, candidate_cols):
    rows = []
    if syn_df is None or not isinstance(syn_df, pd.DataFrame):
        return rows

    syn_df = syn_df.copy()
    syn_df.columns = syn_df.columns.astype(str)

    common = [
        c for c in candidate_cols
        if c in syn_df.columns and c in real_df.columns and not _is_time_col_161(c)
    ]

    if len(common) > MAX_COLS_161:
        # Deterministic selection: prioritize finite/variable columns.
        scored = []
        for c in common:
            sx = _numeric_array_161(syn_df, c)
            rx = _numeric_array_161(real_df, c)
            score = (
                min(float(np.isfinite(sx).mean()) if len(sx) else 0.0,
                    float(np.isfinite(rx).mean()) if len(rx) else 0.0)
                * np.log1p(max(np.nanstd(_finite_161(sx)) if len(_finite_161(sx)) else 0.0,
                               np.nanstd(_finite_161(rx)) if len(_finite_161(rx)) else 0.0))
            )
            scored.append((c, score))
        common = [c for c, _ in sorted(scored, key=lambda x: (-x[1], x[0]))[:MAX_COLS_161]]

    for col in common:
        s = _numeric_array_161(syn_df, col)
        r = _numeric_array_161(real_df, col)

        sf = _finite_161(s)
        rf = _finite_161(r)

        syn_finite_rate = float(np.isfinite(s).mean()) if len(s) else 0.0
        real_finite_rate = float(np.isfinite(r).mean()) if len(r) else 0.0

        if syn_finite_rate < MIN_FINITE_RATE_161 or real_finite_rate < MIN_FINITE_RATE_161:
            rows.append({
                "variant": variant_name,
                "artifact_role": artifact_role,
                "role_group": _tier_or_role_161(col),
                "col": col,
                "metric_scope": "Q1_marginal",
                "q1_status": "not_evaluable",
                "q1_reasons": "low_finite_rate",
                "syn_finite_rate": syn_finite_rate,
                "real_finite_rate": real_finite_rate,
                "TEST_real_values_used_for_reference_only": True,
                "synthetic_values_mutated": False,
            })
            continue

        real_mean = float(np.nanmean(rf)) if len(rf) else np.nan
        syn_mean = float(np.nanmean(sf)) if len(sf) else np.nan
        real_std = float(np.nanstd(rf)) if len(rf) else np.nan
        syn_std = float(np.nanstd(sf)) if len(sf) else np.nan
        real_iqr = float(np.nanquantile(rf, 0.75) - np.nanquantile(rf, 0.25)) if len(rf) else np.nan
        scale = max(abs(real_mean) if np.isfinite(real_mean) else 0.0,
                    real_std if np.isfinite(real_std) else 0.0,
                    real_iqr if np.isfinite(real_iqr) else 0.0,
                    1e-9)

        ks = _ks_2samp_manual_161(rf, sf)
        wass = _wasserstein_1d_161(rf, sf)
        wass_norm = float(wass / scale) if np.isfinite(wass) else np.nan
        mean_error_norm = float(abs(syn_mean - real_mean) / scale) if np.isfinite(syn_mean) and np.isfinite(real_mean) else np.nan

        if np.isfinite(real_std) and real_std > 1e-12 and np.isfinite(syn_std):
            std_ratio = float(syn_std / real_std)
            std_ratio_log_abs = float(abs(np.log(max(std_ratio, 1e-12))))
        else:
            std_ratio = np.nan
            std_ratio_log_abs = np.nan

        support_jaccard = _support_jaccard_161(rf, sf)
        hist_tv = _hist_tv_161(rf, sf, bins=HIST_BINS_161)

        low_cardinality = bool(
            len(np.unique(rf)) <= LOW_CARD_UNIQUE_161
            and len(np.unique(sf)) <= LOW_CARD_UNIQUE_161
        )

        row = {
            "variant": variant_name,
            "artifact_role": artifact_role,
            "role_group": _tier_or_role_161(col),
            "col": col,
            "metric_scope": "Q1_marginal",
            "syn_finite_rate": syn_finite_rate,
            "real_finite_rate": real_finite_rate,
            "real_n": int(len(rf)),
            "syn_n": int(len(sf)),
            "real_mean": real_mean,
            "syn_mean": syn_mean,
            "real_std": real_std,
            "syn_std": syn_std,
            "real_iqr": real_iqr,
            "scale": scale,
            "ks": ks,
            "wasserstein": wass,
            "wasserstein_norm": wass_norm,
            "mean_error_norm": mean_error_norm,
            "std_ratio": std_ratio,
            "std_ratio_log_abs": std_ratio_log_abs,
            "support_jaccard": support_jaccard,
            "histogram_tv": hist_tv,
            "low_cardinality": low_cardinality,
            "TEST_real_values_used_for_reference_only": True,
            "synthetic_values_mutated": False,
        }

        status, reasons = _q1_status_161(row)
        row["q1_status"] = status
        row["q1_reasons"] = reasons
        rows.append(row)

    return rows

# ----------------------------------------------------------
# 4) Validate Cell 16.0 no-Q4 governance
# ----------------------------------------------------------
CELL16_0_CONTRACT_VERSION_161 = _require_contract_governance_161(CELL16_0_DECOMPOSITION_INPUT_CONTRACT)

# ----------------------------------------------------------
# 5) Load real TEST reference and synthetic variants
# ----------------------------------------------------------
real_test_df_161 = df_te.copy()
real_test_df_161.columns = real_test_df_161.columns.astype(str)

variant_specs = []

def _add_variant_spec_161(variant, artifact_role, path, required=False):
    if _exists_161(path):
        variant_specs.append({
            "variant": variant,
            "artifact_role": artifact_role,
            "path": path,
            "exists": True,
            "required": bool(required),
        })
    elif required:
        raise RuntimeError(f"[Cell16.1] Missing required variant artifact: {variant} | {path}")
    else:
        variant_specs.append({
            "variant": variant,
            "artifact_role": artifact_role,
            "path": path,
            "exists": False,
            "required": bool(required),
        })

_add_variant_spec_161("A0", "protocol_baseline", CELL16_0_A0_PROTOCOL_PATH, required=False)
_add_variant_spec_161("A1", "protocol_backbone", CELL16_0_A1_PROTOCOL_PATH, required=False)
_add_variant_spec_161("A2", "protocol_hybrid", CELL16_0_A2_PROTOCOL_PATH, required=False)

# Authoritative scientific variants after Q4 was blocked/no-promoted.
_add_variant_spec_161("scientific_no_q4_protocol", "scientific_protocol_no_q4", CELL16_0_SCIENTIFIC_NO_Q4_PROTOCOL_PATH, required=True)
_add_variant_spec_161("scientific_no_q4_cps", "scientific_cps_no_q4", CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH, required=True)

# Optional non-authoritative legacy Q4 context only, if Cell16.0 registered paths.
if isinstance(globals().get("CELL16_0_LEGACY_Q4_PROTOCOL_PATH", ""), str) and _exists_161(globals().get("CELL16_0_LEGACY_Q4_PROTOCOL_PATH", "")):
    _add_variant_spec_161("legacy_q4_protocol_context", "legacy_q4_protocol_not_authoritative", globals()["CELL16_0_LEGACY_Q4_PROTOCOL_PATH"], required=False)
if isinstance(globals().get("CELL16_0_LEGACY_Q4_CPS_PATH", ""), str) and _exists_161(globals().get("CELL16_0_LEGACY_Q4_CPS_PATH", "")):
    _add_variant_spec_161("legacy_q4_cps_context", "legacy_q4_cps_not_authoritative", globals()["CELL16_0_LEGACY_Q4_CPS_PATH"], required=False)

# Public Q6 role-restricted release candidate.
_add_variant_spec_161("Q6_public_cps", "public_q6_mitigated_cps", CELL16_0_PUBLIC_Q6_CPS_PATH, required=False)

log(
    "[Cell16.1] Computing Q1 marginal decomposition | "
    f"variants={[v['variant'] for v in variant_specs if v['exists']]}"
)

# Candidate columns by variant: use common with real df; protocol variants are protocol-only.
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

    syn_df = _read_parquet_optional_161(path)
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
            f"[Cell16.1] Row mismatch for variant={variant}: got={len(syn_df)} expected={N_TE}"
        )

    if artifact_role in {
        "protocol_baseline",
        "protocol_backbone",
        "protocol_hybrid",
        "scientific_protocol_no_q4",
        "legacy_q4_protocol_not_authoritative",
    }:
        candidate_cols = [
            c for c in syn_df.columns
            if _tier_or_role_161(c).startswith("protocol_")
        ]
    elif artifact_role == "public_q6_mitigated_cps":
        candidate_cols = [
            c for c in syn_df.columns
            if _tier_or_role_161(c) in {"protocol_router", "protocol_zigbee", "iot_event_drivers"}
        ]
    else:
        candidate_cols = [c for c in syn_df.columns if not _is_time_col_161(c)]

    rows = _compute_q1_column_metrics_161(
        variant_name=variant,
        artifact_role=artifact_role,
        syn_df=syn_df,
        real_df=real_test_df_161,
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
    raise RuntimeError("[Cell16.1] No Q1 marginal decomposition metrics were produced.")

# ----------------------------------------------------------
# 6) Summaries by variant and role
# ----------------------------------------------------------
def _summary_from_group_161(g: pd.DataFrame):
    status_counts = g["q1_status"].astype(str).value_counts().to_dict()
    evaluable = g[~g["q1_status"].astype(str).eq("not_evaluable")].copy()

    return pd.Series({
        "cols_total": int(len(g)),
        "cols_evaluable": int(len(evaluable)),
        "pass_n": int(status_counts.get("pass", 0)),
        "warning_n": int(status_counts.get("warning", 0)),
        "fatal_n": int(status_counts.get("fatal", 0)),
        "not_evaluable_n": int(status_counts.get("not_evaluable", 0)),
        "pass_rate_evaluable": float((evaluable["q1_status"].astype(str).eq("pass")).mean()) if len(evaluable) else np.nan,
        "fatal_rate_evaluable": float((evaluable["q1_status"].astype(str).eq("fatal")).mean()) if len(evaluable) else np.nan,
        "mean_ks": float(pd.to_numeric(evaluable["ks"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "median_ks": float(pd.to_numeric(evaluable["ks"], errors="coerce").median()) if len(evaluable) else np.nan,
        "mean_wasserstein_norm": float(pd.to_numeric(evaluable["wasserstein_norm"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "median_wasserstein_norm": float(pd.to_numeric(evaluable["wasserstein_norm"], errors="coerce").median()) if len(evaluable) else np.nan,
        "mean_mean_error_norm": float(pd.to_numeric(evaluable["mean_error_norm"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_std_ratio_log_abs": float(pd.to_numeric(evaluable["std_ratio_log_abs"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_support_jaccard": float(pd.to_numeric(evaluable["support_jaccard"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "mean_histogram_tv": float(pd.to_numeric(evaluable["histogram_tv"], errors="coerce").mean()) if len(evaluable) else np.nan,
        "q1_penalty_score": _variant_score_161(evaluable),
    })

variant_summary_df = (
    column_metrics_df
    .groupby(["variant", "artifact_role"], dropna=False)
    .apply(_summary_from_group_161)
    .reset_index()
    .sort_values(["q1_penalty_score", "variant"], ascending=[True, True])
    .reset_index(drop=True)
)

role_summary_df = (
    column_metrics_df
    .groupby(["variant", "artifact_role", "role_group"], dropna=False)
    .apply(_summary_from_group_161)
    .reset_index()
    .sort_values(["variant", "role_group"])
    .reset_index(drop=True)
)

# ----------------------------------------------------------
# 7) Gain/loss ledger
# ----------------------------------------------------------
ledger_rows = []

# Compare A0/A1/A2 protocol variants where available.
protocol_variants = ["A0", "A1", "A2", "scientific_no_q4_protocol", "legacy_q4_protocol_context"]
protocol_summary = variant_summary_df[variant_summary_df["variant"].isin(protocol_variants)].copy()

if len(protocol_summary):
    best_protocol = protocol_summary.sort_values("q1_penalty_score", ascending=True).iloc[0]
    worst_protocol = protocol_summary.sort_values("q1_penalty_score", ascending=False).iloc[0]

    ledger_rows.append({
        "ledger_item": "best_protocol_q1_variant",
        "variant": str(best_protocol["variant"]),
        "artifact_role": str(best_protocol["artifact_role"]),
        "q1_penalty_score": _safe_float_161(best_protocol["q1_penalty_score"], np.nan),
        "mean_ks": _safe_float_161(best_protocol["mean_ks"], np.nan),
        "mean_wasserstein_norm": _safe_float_161(best_protocol["mean_wasserstein_norm"], np.nan),
        "pass_rate_evaluable": _safe_float_161(best_protocol["pass_rate_evaluable"], np.nan),
        "interpretation": "Lowest Q1 penalty among protocol variants.",
    })

    ledger_rows.append({
        "ledger_item": "worst_protocol_q1_variant",
        "variant": str(worst_protocol["variant"]),
        "artifact_role": str(worst_protocol["artifact_role"]),
        "q1_penalty_score": _safe_float_161(worst_protocol["q1_penalty_score"], np.nan),
        "mean_ks": _safe_float_161(worst_protocol["mean_ks"], np.nan),
        "mean_wasserstein_norm": _safe_float_161(worst_protocol["mean_wasserstein_norm"], np.nan),
        "pass_rate_evaluable": _safe_float_161(worst_protocol["pass_rate_evaluable"], np.nan),
        "interpretation": "Highest Q1 penalty among protocol variants.",
    })

    # Pairwise deltas if present.
    score_map = dict(zip(protocol_summary["variant"].astype(str), pd.to_numeric(protocol_summary["q1_penalty_score"], errors="coerce")))
    for a, b in [("A0", "A1"), ("A1", "A2"), ("A2", "scientific_no_q4_protocol"), ("A0", "scientific_no_q4_protocol")]:
        if a in score_map and b in score_map and np.isfinite(score_map[a]) and np.isfinite(score_map[b]):
            delta = float(score_map[b] - score_map[a])
            ledger_rows.append({
                "ledger_item": f"q1_penalty_delta::{a}_to_{b}",
                "variant": f"{a}->{b}",
                "artifact_role": "protocol_delta",
                "q1_penalty_score": delta,
                "mean_ks": np.nan,
                "mean_wasserstein_norm": np.nan,
                "pass_rate_evaluable": np.nan,
                "interpretation": (
                    "negative means later variant improved Q1 marginal penalty; "
                    "positive means later variant worsened Q1 marginal penalty."
                ),
            })

# Identify worst columns in final scientific artifact.
final_cols = column_metrics_df[
    column_metrics_df["variant"].astype(str).eq("scientific_no_q4_cps")
    & column_metrics_df["q1_status"].astype(str).isin(["fatal", "warning"])
].copy()

if len(final_cols):
    final_cols["column_penalty"] = (
        pd.to_numeric(final_cols["ks"], errors="coerce").fillna(0.5)
        + pd.to_numeric(final_cols["wasserstein_norm"], errors="coerce").fillna(1.0)
        + pd.to_numeric(final_cols["mean_error_norm"], errors="coerce").fillna(1.0)
        + pd.to_numeric(final_cols["std_ratio_log_abs"], errors="coerce").fillna(1.0)
        + (1.0 - pd.to_numeric(final_cols["support_jaccard"], errors="coerce").fillna(0.0))
    )
    for _, r in final_cols.sort_values("column_penalty", ascending=False).head(25).iterrows():
        ledger_rows.append({
            "ledger_item": "worst_final_q1_column",
            "variant": str(r["variant"]),
            "artifact_role": str(r["artifact_role"]),
            "role_group": str(r["role_group"]),
            "col": str(r["col"]),
            "q1_status": str(r["q1_status"]),
            "q1_reasons": str(r["q1_reasons"]),
            "q1_penalty_score": _safe_float_161(r["column_penalty"], np.nan),
            "ks": _safe_float_161(r["ks"], np.nan),
            "wasserstein_norm": _safe_float_161(r["wasserstein_norm"], np.nan),
            "support_jaccard": _safe_float_161(r["support_jaccard"], np.nan),
            "interpretation": "High-priority marginal mismatch in authoritative no-Q4 scientific artifact.",
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
    "cell": "16.1",
    "version": CELL161_VERSION,
    "role": "q1_marginal_decomposition_no_q4_promotion",
    "quality_dimension": "Q1_marginal_distribution_quality",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "legacy_q4_variants_non_authoritative": True
    },
    "upstream_contract_versions": {
        "cell16_0": CELL16_0_CONTRACT_VERSION_161
    },
    "summary": {
        "variants_loaded": variant_load_df.to_dict("records"),
        "column_metric_rows": int(len(column_metrics_df)),
        "variant_summary_rows": int(len(variant_summary_df)),
        "role_summary_rows": int(len(role_summary_df)),
        "gain_loss_ledger_rows": int(len(gain_loss_ledger_df)),
        "best_variant_by_q1_penalty": (
            variant_summary_df.iloc[0].to_dict()
            if len(variant_summary_df)
            else {}
        ),
    },
    "method": {
        "metrics": [
            "KS distance",
            "Wasserstein distance normalized by real scale",
            "mean absolute error normalized by real scale",
            "log absolute standard-deviation ratio",
            "support Jaccard / range-overlap proxy",
            "histogram total variation",
        ],
        "penalty_score": "lower is better; average of marginal mismatch penalties",
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

_write_json_161(contract_json, contract)
_write_json_161(contract_canonical_json, contract)

manifest = {
    "cell": "16.1",
    "version": CELL161_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
}

_write_json_161(manifest_json, manifest)

hashes = {
    "variant_summary_csv_sha256": _sha256_file_161(variant_summary_csv),
    "role_summary_csv_sha256": _sha256_file_161(role_summary_csv),
    "column_metrics_csv_sha256": _sha256_file_161(column_metrics_csv),
    "gain_loss_ledger_csv_sha256": _sha256_file_161(gain_loss_ledger_csv),
    "contract_json_sha256": _sha256_file_161(contract_json),
    "contract_canonical_json_sha256": _sha256_file_161(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_161(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_161(contract_json, contract)
_write_json_161(contract_canonical_json, contract)
_write_json_161(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals
# ----------------------------------------------------------
globals()["CELL161_VERSION"] = CELL161_VERSION
globals()["CELL16_1_Q1_COLUMN_METRICS_DF"] = column_metrics_df
globals()["CELL16_1_Q1_VARIANT_SUMMARY_DF"] = variant_summary_df
globals()["CELL16_1_Q1_ROLE_SUMMARY_DF"] = role_summary_df
globals()["CELL16_1_Q1_GAIN_LOSS_LEDGER_DF"] = gain_loss_ledger_df
globals()["CELL16_1_Q1_DECOMPOSITION_CONTRACT"] = contract

globals()["CELL16_1_Q1_COLUMN_METRICS_CSV"] = column_metrics_csv
globals()["CELL16_1_Q1_VARIANT_SUMMARY_CSV"] = variant_summary_csv
globals()["CELL16_1_Q1_ROLE_SUMMARY_CSV"] = role_summary_csv
globals()["CELL16_1_Q1_GAIN_LOSS_LEDGER_CSV"] = gain_loss_ledger_csv
globals()["CELL16_1_Q1_CONTRACT_JSON"] = contract_json
globals()["CELL16_1_Q1_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL16_1_Q1_MANIFEST_JSON"] = manifest_json

status_counts = column_metrics_df["q1_status"].astype(str).value_counts().sort_index().to_dict()

log(
    "[Cell16.1] Q1 marginal decomposition complete | "
    f"column_metrics={len(column_metrics_df)} | "
    f"variants={variant_summary_df['variant'].nunique() if len(variant_summary_df) else 0} | "
    f"status_counts={status_counts}"
)
if len(variant_summary_df):
    log(
        "[Cell16.1] Variant summary | "
        f"{variant_summary_df[['variant', 'artifact_role', 'cols_evaluable', 'pass_n', 'warning_n', 'fatal_n', 'q1_penalty_score']].to_dict('records')}"
    )
log(f"[Cell16.1] Saved variant summary: {variant_summary_csv}")
log(f"[Cell16.1] Saved role summary: {role_summary_csv}")
log(f"[Cell16.1] Saved column metrics: {column_metrics_csv}")
log(f"[Cell16.1] Saved gain/loss ledger: {gain_loss_ledger_csv}")
log(f"[Cell16.1] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell16.1] Contract flags | "
    "TEST_real_values_used_for_reference_only=True | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "decomposition_done_here=True | Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False"
)
log("--- END: Cell 16.1 - Q1 marginal decomposition (v1.1 no-Q4-promotion strict) ---")

gc.collect()