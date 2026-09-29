# ==========================================================
# CELL 13.4 - Q4 cross-modal coupling summary and blockers
# v1.1 STUDY-THESIS strict Q4 publication summary, contract-hardened
#
# Role:
#   - Summarize cross-modal IoT↔network/protocol coupling QA from:
#       13.1 ETA event-triggered average QA
#       13.2 lag distribution / response-window QA
#       13.3 cross-correlation preservation QA
#   - Produce tier-level and pair-level Q4 publication decisions.
#   - Attribute Q4 blockers honestly.
#
# Metrics:
#   ETA_similarity
#   lag_peak_error
#   response_window_rate_error
#   cross_corr_mae
#   driver_protocol_pair_pass_rate
#
# Strict rules:
#   - TEST real values were used only in upstream QA cells.
#   - Do NOT mutate synthetic values.
#   - Do NOT fit generators.
#   - Do NOT select generators.
#   - Do NOT materialize new synthetic values.
#
# Outputs:
#   reports/cell13_4_q4_pair_summary.csv
#   reports/cell13_4_q4_tier_summary.csv
#   reports/cell13_4_q4_publication_blockers.csv
#   reports/cell13_4_q4_publication_summary.json
#   artifacts/cell13_4_q4_manifest.json
# ==========================================================

log("--- START: Cell 13.4 - Q4 coupling summary and publication blockers (v1.1 contract-hardened strict) ---")

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
_required_134 = [
    "CFG", "log",
    "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL13_1_ETA_PAIR_METRICS_DF",
    "CELL13_1_ETA_TIER_SUMMARY_DF",
    "CELL13_1_ETA_CONTRACT",
    "CELL13_2_LAG_RESPONSE_PAIR_METRICS_DF",
    "CELL13_2_LAG_RESPONSE_TIER_SUMMARY_DF",
    "CELL13_2_LAG_RESPONSE_CONTRACT",
    "CELL13_3_CROSS_CORR_PAIR_METRICS_DF",
    "CELL13_3_CROSS_CORR_TIER_SUMMARY_DF",
    "CELL13_3_CROSS_CORR_CONTRACT",
    "CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF",
    "CELL130_Q4_COUPLING_INPUT_CONTRACT",
]
_missing_134 = [k for k in _required_134 if k not in globals()]
if _missing_134:
    raise RuntimeError(f"[Cell13.4] Missing required globals from prior Cell 13.x cells: {_missing_134}")

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
N_TE = int(len(df_te))

CELL134_VERSION = "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1_contract_hardened"

CFG["cell13_4_version"] = CELL134_VERSION
CFG["cell13_4_QA_summary_only"] = True
CFG["cell13_4_TEST_real_values_used_only_in_upstream_QA"] = True
CFG["cell13_4_synthetic_values_mutated"] = False
CFG["cell13_4_selection_done_here"] = False
CFG["cell13_4_generator_fit_done_here"] = False
CFG["cell13_4_materialization_done_here"] = False
CFG["cell13_4_contract_hardened"] = True
CFG["cell13_4_TEST_activity_used_for_pair_selection"] = False
CFG["cell13_4_pair_registry_schema_driven_required"] = True

CFG.setdefault("cell13_4_pair_pass_requires_all_metrics_pass", True)
CFG.setdefault("cell13_4_pair_warning_allowed", True)
CFG.setdefault("cell13_4_tier_pass_rate_pass", 0.70)
CFG.setdefault("cell13_4_tier_pass_rate_warning", 0.40)
CFG.setdefault("cell13_4_overall_pass_rate_pass", 0.70)
CFG.setdefault("cell13_4_overall_pass_rate_warning", 0.40)
CFG.setdefault("cell13_4_eta_similarity_blocker_mean", 0.50)
CFG.setdefault("cell13_4_lag_similarity_blocker_mean", 0.50)
CFG.setdefault("cell13_4_cross_corr_similarity_blocker_mean", 0.50)
CFG.setdefault("cell13_4_lag_peak_error_blocker_mean", 15.0)
CFG.setdefault("cell13_4_response_window_error_blocker_mean", 0.25)
CFG.setdefault("cell13_4_zigbee_missing_is_warning", True)
CFG.setdefault("cell13_4_zwave_missing_is_warning", True)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
q4_pair_summary_csv = os.path.join(REPORT_DIR, "cell13_4_q4_pair_summary.csv")
q4_tier_summary_csv = os.path.join(REPORT_DIR, "cell13_4_q4_tier_summary.csv")
q4_publication_blockers_csv = os.path.join(REPORT_DIR, "cell13_4_q4_publication_blockers.csv")
q4_publication_summary_json = os.path.join(REPORT_DIR, "cell13_4_q4_publication_summary.json")
q4_publication_summary_canonical_json = os.path.join(CONTRACT_DIR, "cell13_4_q4_publication_summary_v1_1_THESIS.json")
q4_manifest_json = os.path.join(ARTDIR, "cell13_4_q4_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_134(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_134(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_134(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_134(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_134(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_134(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_134(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_134(obj.to_dict())
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

def _write_json_134(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_134(payload), f, indent=2, sort_keys=True)

def _sha256_file_134(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_134(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _status_rank_134(status: str) -> int:
    s = str(status).strip().lower()
    if s == "pass":
        return 0
    if s == "warning":
        return 1
    if s == "not_evaluable":
        return 2
    if s == "fatal":
        return 3
    return 2

def _worst_status_134(statuses) -> str:
    statuses = [str(s) for s in statuses if str(s).strip()]
    if not statuses:
        return "not_evaluable"
    worst = max(statuses, key=_status_rank_134)
    return worst

def _pair_publication_status_134(eta_status, lag_status, cc_status) -> tuple:
    statuses = [eta_status, lag_status, cc_status]
    worst = _worst_status_134(statuses)

    reasons = []

    if str(eta_status) == "fatal":
        reasons.append("eta_fatal")
    elif str(eta_status) == "warning":
        reasons.append("eta_warning")
    elif str(eta_status) == "not_evaluable":
        reasons.append("eta_not_evaluable")

    if str(lag_status) == "fatal":
        reasons.append("lag_response_fatal")
    elif str(lag_status) == "warning":
        reasons.append("lag_response_warning")
    elif str(lag_status) == "not_evaluable":
        reasons.append("lag_response_not_evaluable")

    if str(cc_status) == "fatal":
        reasons.append("cross_corr_fatal")
    elif str(cc_status) == "warning":
        reasons.append("cross_corr_warning")
    elif str(cc_status) == "not_evaluable":
        reasons.append("cross_corr_not_evaluable")

    if worst == "fatal":
        return "blocker", "|".join(reasons) if reasons else "q4_metric_fatal"

    if worst == "warning":
        return "warning", "|".join(reasons) if reasons else "q4_metric_warning"

    if worst == "not_evaluable":
        return "not_evaluable", "|".join(reasons) if reasons else "q4_not_evaluable"

    return "pass", "all_q4_pair_metrics_pass"

def _tier_status_134(row: dict) -> tuple:
    pass_rate = _safe_float_134(row.get("driver_protocol_pair_pass_rate"), np.nan)
    blocker_n = int(_safe_float_134(row.get("publication_blocker_n"), 0))
    pairs_evaluable = int(_safe_float_134(row.get("pairs_evaluable"), 0))

    reasons = []

    if pairs_evaluable <= 0:
        return "not_evaluable", "no_evaluable_pairs"

    if np.isfinite(pass_rate):
        if pass_rate >= float(CFG["cell13_4_tier_pass_rate_pass"]):
            status = "pass"
        elif pass_rate >= float(CFG["cell13_4_tier_pass_rate_warning"]):
            status = "warning"
            reasons.append("tier_pair_pass_rate_warning")
        else:
            status = "blocker"
            reasons.append("tier_pair_pass_rate_blocker")
    else:
        status = "not_evaluable"
        reasons.append("tier_pair_pass_rate_not_finite")

    if blocker_n > 0 and status == "pass":
        status = "warning"
        reasons.append("tier_has_pair_blockers")

    if not reasons:
        reasons.append("tier_q4_pass")

    return status, "|".join(reasons)

def _protocol_tier_from_pair_134(row):
    if "protocol_tier" in row and str(row["protocol_tier"]).strip():
        return str(row["protocol_tier"])
    p = str(row.get("protocol_col", "")).lower()
    if p.startswith("router__") or p.startswith("dns__"):
        return "router"
    if p.startswith("ota__") or p.startswith("ota24__") or p.startswith("ota5__") or p.startswith("wifi__"):
        return "ota"
    if p.startswith("zigbee__") or "zigbee" in p:
        return "zigbee"
    if p.startswith("zwave__") or "zwave" in p or "z_wave" in p:
        return "zwave"
    return "other"

# ----------------------------------------------------------
# 3) Validate upstream Q4 contracts
# ----------------------------------------------------------
def _require_contract_version_134(obj, name: str, expected_substring: str):
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell13.4] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell13.4] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

version130_134 = _require_contract_version_134(
    CELL130_Q4_COUPLING_INPUT_CONTRACT,
    "CELL130_Q4_COUPLING_INPUT_CONTRACT",
    "cell13_0_q4_cross_modal_coupling_input_contract_v1_1",
)
version131_134 = _require_contract_version_134(
    CELL13_1_ETA_CONTRACT,
    "CELL13_1_ETA_CONTRACT",
    "cell13_1_q4_event_triggered_average_eta_qa_v1_1",
)
version132_134 = _require_contract_version_134(
    CELL13_2_LAG_RESPONSE_CONTRACT,
    "CELL13_2_LAG_RESPONSE_CONTRACT",
    "cell13_2_q4_lag_response_window_qa_v1_1",
)
version133_134 = _require_contract_version_134(
    CELL13_3_CROSS_CORR_CONTRACT,
    "CELL13_3_CROSS_CORR_CONTRACT",
    "cell13_3_q4_cross_correlation_preservation_qa_v1_1",
)

strict130_134 = CELL130_Q4_COUPLING_INPUT_CONTRACT.get("strict_contract", {})
strict131_134 = CELL13_1_ETA_CONTRACT.get("strict_contract", {})
strict132_134 = CELL13_2_LAG_RESPONSE_CONTRACT.get("strict_contract", {})
strict133_134 = CELL13_3_CROSS_CORR_CONTRACT.get("strict_contract", {})

for _name, _strict in [
    ("13.0", strict130_134),
    ("13.1", strict131_134),
    ("13.2", strict132_134),
    ("13.3", strict133_134),
]:
    if bool(_strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell13.4] Upstream Cell {_name} indicates synthetic values were mutated.")
    if bool(_strict.get("selection_done_here", True)):
        raise RuntimeError(f"[Cell13.4] Upstream Cell {_name} indicates selection was done.")
    if bool(_strict.get("generator_fit_done_here", True)):
        raise RuntimeError(f"[Cell13.4] Upstream Cell {_name} indicates generator fitting was done.")
    if bool(_strict.get("materialization_done_here", True)):
        raise RuntimeError(f"[Cell13.4] Upstream Cell {_name} indicates materialization was done.")

if bool(strict130_134.get("TEST_activity_used_for_pair_selection", True)):
    raise RuntimeError("[Cell13.4] Cell 13.0 contract indicates TEST activity was used for pair selection.")
if not bool(strict130_134.get("pair_registry_schema_driven", False)):
    raise RuntimeError("[Cell13.4] Cell 13.0 contract does not declare a schema-driven pair registry.")

for _name, _strict in [("13.1", strict131_134), ("13.2", strict132_134), ("13.3", strict133_134)]:
    if bool(_strict.get("TEST_activity_used_for_pair_selection", True)):
        raise RuntimeError(f"[Cell13.4] Cell {_name} contract indicates TEST activity was used for pair selection.")
    if not bool(_strict.get("pair_registry_schema_driven", False)):
        raise RuntimeError(f"[Cell13.4] Cell {_name} contract does not declare schema-driven pair registry.")

q3_status_134 = (
    CELL130_Q4_COUPLING_INPUT_CONTRACT.get("quality_status_carried_forward", {})
    .get("q3_observability", {})
    if isinstance(CELL130_Q4_COUPLING_INPUT_CONTRACT.get("quality_status_carried_forward", {}), dict)
    else {}
)
q3_publication_blocker_n_134 = int(
    CELL130_Q4_COUPLING_INPUT_CONTRACT.get("quality_status_carried_forward", {})
    .get("q3_publication_blocker_n", 0)
    if isinstance(CELL130_Q4_COUPLING_INPUT_CONTRACT.get("quality_status_carried_forward", {}), dict)
    else 0
)

eta_publication_blocker_n_134 = int(
    CELL13_1_ETA_CONTRACT.get("metric_summary", {}).get("publication_blocker_n", 0) or 0
)
lag_publication_blocker_n_134 = int(
    CELL13_2_LAG_RESPONSE_CONTRACT.get("metric_summary", {}).get("publication_blocker_n", 0) or 0
)
cross_corr_publication_blocker_n_134 = int(
    CELL13_3_CROSS_CORR_CONTRACT.get("metric_summary", {}).get("publication_blocker_n", 0) or 0
)

# ----------------------------------------------------------
# 4) Load and normalize upstream metrics
# ----------------------------------------------------------
eta = CELL13_1_ETA_PAIR_METRICS_DF.copy()
lag = CELL13_2_LAG_RESPONSE_PAIR_METRICS_DF.copy()
cc = CELL13_3_CROSS_CORR_PAIR_METRICS_DF.copy()
pairs = CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF.copy()

if "activity_stats_used_for_pair_selection" in pairs.columns:
    bad_activity_selected_134 = pairs[
        pairs["activity_stats_used_for_pair_selection"].fillna(False).astype(bool)
    ]
    if len(bad_activity_selected_134):
        raise RuntimeError(
            "[Cell13.4] Pair registry contains pairs selected using TEST activity statistics. "
            f"Rows={len(bad_activity_selected_134)}"
        )

for name, df in [("eta", eta), ("lag", lag), ("cc", cc), ("pairs", pairs)]:
    if "pair_id" not in df.columns:
        raise RuntimeError(f"[Cell13.4] Missing pair_id in {name} dataframe.")
    df["pair_id"] = df["pair_id"].astype(str)

# Keep key columns and avoid duplicate metric name collisions.
eta_cols = [
    "pair_id", "driver_col", "driver_entity", "protocol_col", "protocol_tier",
    "eta_similarity", "eta_mae", "eta_norm_mae", "eta_corr",
    "eta_peak_error", "eta_area_error", "real_event_count_window_valid",
    "syn_event_count_window_valid", "eta_status", "eta_status_reason",
    "eta_publication_blocker",
]
eta_cols = [c for c in eta_cols if c in eta.columns]
eta_s = eta[eta_cols].copy()

lag_cols = [
    "pair_id",
    "lag_similarity", "lag_distribution_mae", "lag_distribution_norm_mae",
    "lag_corr", "lag_peak_error", "real_lag_peak", "syn_lag_peak",
    "real_response_window_rate", "syn_response_window_rate",
    "response_window_rate_error", "lag_response_status",
    "lag_response_status_reason", "lag_response_publication_blocker",
]
lag_cols = [c for c in lag_cols if c in lag.columns]
lag_s = lag[lag_cols].copy()

cc_cols = [
    "pair_id",
    "cross_corr_similarity", "cross_corr_mae", "cross_corr_rmse",
    "cross_corr_corr", "cross_corr_peak_lag_error",
    "cross_corr_peak_value_error", "real_cross_corr_peak_lag",
    "syn_cross_corr_peak_lag", "cross_corr_status",
    "cross_corr_status_reason", "cross_corr_publication_blocker",
]
cc_cols = [c for c in cc_cols if c in cc.columns]
cc_s = cc[cc_cols].copy()

pair_summary = eta_s.merge(lag_s, on="pair_id", how="outer")
pair_summary = pair_summary.merge(cc_s, on="pair_id", how="outer")

# Fill identity columns from registry if needed.
pairs_cols = [
    "pair_id", "driver_col", "driver_entity", "protocol_col", "protocol_tier", "semantic_hint"
]
pairs_cols = [c for c in pairs_cols if c in pairs.columns]
pairs_s = pairs[pairs_cols].copy()

pair_summary = pairs_s.merge(pair_summary, on="pair_id", how="left", suffixes=("", "_metric"))

# Reconcile duplicated identity columns from ETA.
for base in ["driver_col", "driver_entity", "protocol_col", "protocol_tier"]:
    metric_col = f"{base}_metric"
    if metric_col in pair_summary.columns:
        pair_summary[base] = pair_summary[base].fillna(pair_summary[metric_col])
        pair_summary.drop(columns=[metric_col], inplace=True)

# Numeric conversions.
numeric_cols = [
    "eta_similarity", "eta_mae", "eta_norm_mae", "eta_corr",
    "eta_peak_error", "eta_area_error",
    "lag_similarity", "lag_distribution_mae", "lag_distribution_norm_mae",
    "lag_corr", "lag_peak_error", "real_lag_peak", "syn_lag_peak",
    "real_response_window_rate", "syn_response_window_rate",
    "response_window_rate_error",
    "cross_corr_similarity", "cross_corr_mae", "cross_corr_rmse",
    "cross_corr_corr", "cross_corr_peak_lag_error",
    "cross_corr_peak_value_error",
]
for c in numeric_cols:
    if c in pair_summary.columns:
        pair_summary[c] = pd.to_numeric(pair_summary[c], errors="coerce")

for c in ["eta_status", "lag_response_status", "cross_corr_status"]:
    if c not in pair_summary.columns:
        pair_summary[c] = "not_evaluable"
    pair_summary[c] = pair_summary[c].fillna("not_evaluable").astype(str)

pair_status_rows = []
for _, r in pair_summary.iterrows():
    status, reasons = _pair_publication_status_134(
        r.get("eta_status", "not_evaluable"),
        r.get("lag_response_status", "not_evaluable"),
        r.get("cross_corr_status", "not_evaluable"),
    )
    pair_status_rows.append((status, reasons))

pair_summary["q4_pair_publication_status"] = [x[0] for x in pair_status_rows]
pair_summary["q4_pair_publication_reasons"] = [x[1] for x in pair_status_rows]
pair_summary["q4_pair_publication_blocker"] = pair_summary["q4_pair_publication_status"].eq("blocker")

pair_summary["protocol_tier"] = pair_summary.apply(_protocol_tier_from_pair_134, axis=1)

if len(pair_summary) == 0:
    raise RuntimeError("[Cell13.4] Empty Q4 pair summary after merging metrics.")

# ----------------------------------------------------------
# 5) Tier summary
# ----------------------------------------------------------
tier_rows = []

for tier, sub in pair_summary.groupby("protocol_tier"):
    sub = sub.copy()
    evaluable = sub[~sub["q4_pair_publication_status"].astype(str).eq("not_evaluable")].copy()

    pairs_total = int(len(sub))
    pairs_evaluable = int(len(evaluable))
    pass_n = int((sub["q4_pair_publication_status"].astype(str) == "pass").sum())
    warning_n = int((sub["q4_pair_publication_status"].astype(str) == "warning").sum())
    blocker_n = int((sub["q4_pair_publication_status"].astype(str) == "blocker").sum())
    not_eval_n = int((sub["q4_pair_publication_status"].astype(str) == "not_evaluable").sum())

    pass_rate = float(pass_n / max(pairs_evaluable, 1)) if pairs_evaluable else np.nan

    row = {
        "protocol_tier": tier,
        "pairs_total": pairs_total,
        "pairs_evaluable": pairs_evaluable,
        "pass_n": pass_n,
        "warning_n": warning_n,
        "blocker_n": blocker_n,
        "not_evaluable_n": not_eval_n,
        "driver_protocol_pair_pass_rate": pass_rate,
        "mean_ETA_similarity": float(pd.to_numeric(evaluable.get("eta_similarity"), errors="coerce").mean()) if pairs_evaluable else np.nan,
        "median_ETA_similarity": float(pd.to_numeric(evaluable.get("eta_similarity"), errors="coerce").median()) if pairs_evaluable else np.nan,
        "mean_lag_peak_error": float(pd.to_numeric(evaluable.get("lag_peak_error"), errors="coerce").mean()) if pairs_evaluable else np.nan,
        "mean_response_window_rate_error": float(pd.to_numeric(evaluable.get("response_window_rate_error"), errors="coerce").mean()) if pairs_evaluable else np.nan,
        "mean_cross_corr_mae": float(pd.to_numeric(evaluable.get("cross_corr_mae"), errors="coerce").mean()) if pairs_evaluable else np.nan,
        "mean_cross_corr_similarity": float(pd.to_numeric(evaluable.get("cross_corr_similarity"), errors="coerce").mean()) if pairs_evaluable else np.nan,
        "mean_cross_corr_peak_lag_error": float(pd.to_numeric(evaluable.get("cross_corr_peak_lag_error"), errors="coerce").mean()) if pairs_evaluable else np.nan,
    }

    tier_status, tier_reasons = _tier_status_134(row)
    row["q4_tier_publication_status"] = tier_status
    row["q4_tier_publication_reasons"] = tier_reasons
    row["q4_tier_publication_blocker"] = bool(tier_status == "blocker")

    tier_rows.append(row)

tier_summary = pd.DataFrame(tier_rows).sort_values("protocol_tier").reset_index(drop=True)

# Explicit unavailable tiers.
evaluated_tiers = set(tier_summary["protocol_tier"].astype(str))
unavailable_rows = []

for tier in ["router", "ota", "zigbee", "zwave"]:
    if tier not in evaluated_tiers:
        reason = f"{tier}_protocol_coupling_not_evaluated_no_selected_protocol_columns"
        unavailable_rows.append({
            "protocol_tier": tier,
            "pairs_total": 0,
            "pairs_evaluable": 0,
            "pass_n": 0,
            "warning_n": 0,
            "blocker_n": 0,
            "not_evaluable_n": 0,
            "driver_protocol_pair_pass_rate": np.nan,
            "mean_ETA_similarity": np.nan,
            "median_ETA_similarity": np.nan,
            "mean_lag_peak_error": np.nan,
            "mean_response_window_rate_error": np.nan,
            "mean_cross_corr_mae": np.nan,
            "mean_cross_corr_similarity": np.nan,
            "mean_cross_corr_peak_lag_error": np.nan,
            "q4_tier_publication_status": "warning" if (
                (tier == "zwave" and bool(CFG["cell13_4_zwave_missing_is_warning"]))
                or (tier == "zigbee" and bool(CFG["cell13_4_zigbee_missing_is_warning"]))
            ) else "not_evaluable",
            "q4_tier_publication_reasons": reason,
            "q4_tier_publication_blocker": False,
        })

if unavailable_rows:
    tier_summary = pd.concat([tier_summary, pd.DataFrame(unavailable_rows)], ignore_index=True)
    tier_summary = tier_summary.sort_values("protocol_tier").reset_index(drop=True)

# ----------------------------------------------------------
# 6) Overall Q4 summary and blockers
# ----------------------------------------------------------
evaluable_pairs = pair_summary[
    ~pair_summary["q4_pair_publication_status"].astype(str).eq("not_evaluable")
].copy()

pair_pass_n = int((pair_summary["q4_pair_publication_status"].astype(str) == "pass").sum())
pair_warning_n = int((pair_summary["q4_pair_publication_status"].astype(str) == "warning").sum())
pair_blocker_n = int(pair_summary["q4_pair_publication_blocker"].fillna(False).astype(bool).sum())
pair_not_eval_n = int((pair_summary["q4_pair_publication_status"].astype(str) == "not_evaluable").sum())
pair_evaluable_n = int(len(evaluable_pairs))
pair_total_n = int(len(pair_summary))

driver_protocol_pair_pass_rate = (
    float(pair_pass_n / max(pair_evaluable_n, 1))
    if pair_evaluable_n
    else np.nan
)

overall_reasons = []

if not np.isfinite(driver_protocol_pair_pass_rate):
    q4_overall_status = "not_evaluable"
    overall_reasons.append("q4_pair_pass_rate_not_finite")
elif driver_protocol_pair_pass_rate >= float(CFG["cell13_4_overall_pass_rate_pass"]):
    q4_overall_status = "pass"
    overall_reasons.append("q4_pair_pass_rate_pass")
elif driver_protocol_pair_pass_rate >= float(CFG["cell13_4_overall_pass_rate_warning"]):
    q4_overall_status = "warning"
    overall_reasons.append("q4_pair_pass_rate_warning")
else:
    q4_overall_status = "blocker"
    overall_reasons.append("q4_pair_pass_rate_blocker")

mean_eta_similarity = float(pd.to_numeric(evaluable_pairs.get("eta_similarity"), errors="coerce").mean()) if pair_evaluable_n else np.nan
mean_lag_similarity = float(pd.to_numeric(evaluable_pairs.get("lag_similarity"), errors="coerce").mean()) if pair_evaluable_n else np.nan
mean_cross_corr_similarity = float(pd.to_numeric(evaluable_pairs.get("cross_corr_similarity"), errors="coerce").mean()) if pair_evaluable_n else np.nan
mean_lag_peak_error = float(pd.to_numeric(evaluable_pairs.get("lag_peak_error"), errors="coerce").mean()) if pair_evaluable_n else np.nan
mean_response_window_rate_error = float(pd.to_numeric(evaluable_pairs.get("response_window_rate_error"), errors="coerce").mean()) if pair_evaluable_n else np.nan
mean_cross_corr_mae = float(pd.to_numeric(evaluable_pairs.get("cross_corr_mae"), errors="coerce").mean()) if pair_evaluable_n else np.nan

if np.isfinite(mean_eta_similarity) and mean_eta_similarity < float(CFG["cell13_4_eta_similarity_blocker_mean"]):
    overall_reasons.append("mean_eta_similarity_below_blocker_threshold")
    q4_overall_status = "blocker"

if np.isfinite(mean_lag_similarity) and mean_lag_similarity < float(CFG["cell13_4_lag_similarity_blocker_mean"]):
    overall_reasons.append("mean_lag_similarity_below_blocker_threshold")
    q4_overall_status = "blocker"

if np.isfinite(mean_lag_peak_error) and mean_lag_peak_error > float(CFG["cell13_4_lag_peak_error_blocker_mean"]):
    overall_reasons.append("mean_lag_peak_error_above_blocker_threshold")
    q4_overall_status = "blocker"

if np.isfinite(mean_response_window_rate_error) and mean_response_window_rate_error > float(CFG["cell13_4_response_window_error_blocker_mean"]):
    overall_reasons.append("mean_response_window_rate_error_above_blocker_threshold")
    q4_overall_status = "blocker"

# cross_corr_similarity is around 0.52 in current run, but peak lag error matters more.
if np.isfinite(mean_cross_corr_similarity) and mean_cross_corr_similarity < float(CFG["cell13_4_cross_corr_similarity_blocker_mean"]):
    overall_reasons.append("mean_cross_corr_similarity_below_blocker_threshold")
    q4_overall_status = "blocker"

tier_blockers_n = int(tier_summary["q4_tier_publication_blocker"].fillna(False).astype(bool).sum())
if tier_blockers_n > 0:
    overall_reasons.append("one_or_more_protocol_tiers_blocked")
    q4_overall_status = "blocker"

if not overall_reasons:
    overall_reasons.append("q4_overall_pass")

# Publication blockers table.
blocker_rows = []

# Overall blocker.
if q4_overall_status == "blocker":
    blocker_rows.append({
        "blocker_scope": "overall_q4",
        "identifier": "Q4_cross_modal_coupling",
        "status": q4_overall_status,
        "reason": "|".join(sorted(set(overall_reasons))),
        "driver_col": "",
        "protocol_col": "",
        "protocol_tier": "",
        "eta_similarity": mean_eta_similarity,
        "lag_peak_error": mean_lag_peak_error,
        "response_window_rate_error": mean_response_window_rate_error,
        "cross_corr_mae": mean_cross_corr_mae,
        "cross_corr_similarity": mean_cross_corr_similarity,
    })

# Tier blockers.
for _, r in tier_summary.iterrows():
    if bool(r.get("q4_tier_publication_blocker", False)):
        blocker_rows.append({
            "blocker_scope": "protocol_tier",
            "identifier": str(r.get("protocol_tier", "")),
            "status": str(r.get("q4_tier_publication_status", "")),
            "reason": str(r.get("q4_tier_publication_reasons", "")),
            "driver_col": "",
            "protocol_col": "",
            "protocol_tier": str(r.get("protocol_tier", "")),
            "eta_similarity": r.get("mean_ETA_similarity", np.nan),
            "lag_peak_error": r.get("mean_lag_peak_error", np.nan),
            "response_window_rate_error": r.get("mean_response_window_rate_error", np.nan),
            "cross_corr_mae": r.get("mean_cross_corr_mae", np.nan),
            "cross_corr_similarity": r.get("mean_cross_corr_similarity", np.nan),
        })

# Pair blockers.
pair_blockers = pair_summary[pair_summary["q4_pair_publication_blocker"].fillna(False).astype(bool)].copy()
for _, r in pair_blockers.iterrows():
    blocker_rows.append({
        "blocker_scope": "driver_protocol_pair",
        "identifier": str(r.get("pair_id", "")),
        "status": str(r.get("q4_pair_publication_status", "")),
        "reason": str(r.get("q4_pair_publication_reasons", "")),
        "driver_col": str(r.get("driver_col", "")),
        "protocol_col": str(r.get("protocol_col", "")),
        "protocol_tier": str(r.get("protocol_tier", "")),
        "eta_similarity": r.get("eta_similarity", np.nan),
        "lag_peak_error": r.get("lag_peak_error", np.nan),
        "response_window_rate_error": r.get("response_window_rate_error", np.nan),
        "cross_corr_mae": r.get("cross_corr_mae", np.nan),
        "cross_corr_similarity": r.get("cross_corr_similarity", np.nan),
    })

blockers_df = pd.DataFrame(blocker_rows)

# ----------------------------------------------------------
# 7) Save outputs
# ----------------------------------------------------------
pair_summary.to_csv(q4_pair_summary_csv, index=False)
tier_summary.to_csv(q4_tier_summary_csv, index=False)
blockers_df.to_csv(q4_publication_blockers_csv, index=False)

summary_payload = {
    "cell": "13.4",
    "version": CELL134_VERSION,
    "role": "Q4_cross_modal_coupling_publication_summary",
    "quality_dimension": "Q4_cross_modal_consistency",
    "test_rows": int(N_TE),
    "overall_q4_status": q4_overall_status,
    "overall_q4_reasons": sorted(set(overall_reasons)),
    "publication_blocker_record_n": int(len(blockers_df)),
    "publication_blocker_n": int(len(blockers_df)),
    "blocker_count_semantics": {
        "publication_blocker_n": "number of blocker records in q4_publication_blockers.csv; includes overall, tier, and pair-level records",
        "pair_publication_blocker_n": int(pair_blocker_n),
        "tier_publication_blocker_n": int(tier_blockers_n),
        "overall_publication_blocker_n": int(1 if q4_overall_status == "blocker" else 0)
    },
    "upstream_quality_status_carried_forward": {
        "q3_publication_blocker_n": int(q3_publication_blocker_n_134),
        "eta_publication_blocker_n": int(eta_publication_blocker_n_134),
        "lag_response_publication_blocker_n": int(lag_publication_blocker_n_134),
        "cross_corr_publication_blocker_n": int(cross_corr_publication_blocker_n_134),
        "q3_observability_status": q3_status_134
    },
    "upstream_contract_versions": {
        "cell13_0": version130_134,
        "cell13_1": version131_134,
        "cell13_2": version132_134,
        "cell13_3": version133_134
    },
    "pair_summary": {
        "pairs_total": pair_total_n,
        "pairs_evaluable": pair_evaluable_n,
        "pair_pass_n": pair_pass_n,
        "pair_warning_n": pair_warning_n,
        "pair_blocker_n": pair_blocker_n,
        "pair_not_evaluable_n": pair_not_eval_n,
        "driver_protocol_pair_pass_rate": driver_protocol_pair_pass_rate,
    },
    "metric_summary": {
        "mean_ETA_similarity": mean_eta_similarity,
        "mean_lag_similarity": mean_lag_similarity,
        "mean_lag_peak_error": mean_lag_peak_error,
        "mean_response_window_rate_error": mean_response_window_rate_error,
        "mean_cross_corr_mae": mean_cross_corr_mae,
        "mean_cross_corr_similarity": mean_cross_corr_similarity,
    },
    "tier_summary": tier_summary.to_dict("records"),
    "interpretation": {
        "q4_main_conclusion": (
            "Cross-modal IoT-to-network/protocol coupling is not publication-clean. "
            "Event-triggered response shape, lag/response-window timing, and cross-correlation peak timing "
            "are poorly preserved under the current synthetic pipeline."
        ),
        "likely_root_cause": (
            "The protocol/network synthetic matrix is not sufficiently conditioned on the final IoT driver "
            "event timing produced by Cell 12.e / assembled in Cell 12.g."
        ),
        "recommended_future_fix": (
            "Add an explicit coupling correction/materialization layer that conditions or aligns protocol "
            "responses to final synthetic IoT driver events before final Q4 QA."
        ),
        "zwave_note": (
            "Z-Wave coupling was not evaluated because Cell 13.0 found no selected Z-Wave synthetic protocol columns."
        ) if "zwave" not in set(pair_summary["protocol_tier"].astype(str)) else "",
    },
    "strict_contract": {
        "TEST_real_values_used_only_in_upstream_QA": True,
        "TEST_activity_used_for_pair_selection": False,
        "pair_registry_schema_driven": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
    },
    "outputs": {
        "q4_pair_summary_csv": q4_pair_summary_csv,
        "q4_tier_summary_csv": q4_tier_summary_csv,
        "q4_publication_blockers_csv": q4_publication_blockers_csv,
        "q4_publication_summary_json": q4_publication_summary_json,
        "q4_publication_summary_canonical_json": q4_publication_summary_canonical_json,
        "q4_manifest_json": q4_manifest_json,
    },
}

_write_json_134(q4_publication_summary_json, summary_payload)
_write_json_134(q4_publication_summary_canonical_json, summary_payload)

manifest = {
    "cell": "13.4",
    "version": CELL134_VERSION,
    "created_outputs": summary_payload["outputs"],
    "quality_dimension": "Q4_cross_modal_consistency",
    "overall_q4_status": q4_overall_status,
    "publication_blocker_record_n": int(len(blockers_df)),
    "publication_blocker_n": int(len(blockers_df)),
    "pair_publication_blocker_n": int(pair_blocker_n),
    "tier_publication_blocker_n": int(tier_blockers_n),
    "q3_publication_blocker_n_carried_forward": int(q3_publication_blocker_n_134),
    "eta_publication_blocker_n_carried_forward": int(eta_publication_blocker_n_134),
    "lag_response_publication_blocker_n_carried_forward": int(lag_publication_blocker_n_134),
    "cross_corr_publication_blocker_n_carried_forward": int(cross_corr_publication_blocker_n_134),
    "driver_protocol_pair_pass_rate": driver_protocol_pair_pass_rate,
    "metric_summary": summary_payload["metric_summary"],
    "strict_contract": summary_payload["strict_contract"],
}

_write_json_134(q4_manifest_json, manifest)

hashes = {
    "q4_pair_summary_csv_sha256": _sha256_file_134(q4_pair_summary_csv),
    "q4_tier_summary_csv_sha256": _sha256_file_134(q4_tier_summary_csv),
    "q4_publication_blockers_csv_sha256": _sha256_file_134(q4_publication_blockers_csv),
    "q4_publication_summary_json_sha256": _sha256_file_134(q4_publication_summary_json),
    "q4_publication_summary_canonical_json_sha256": _sha256_file_134(q4_publication_summary_canonical_json),
    "q4_manifest_json_sha256": _sha256_file_134(q4_manifest_json),
}

summary_payload["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_134(q4_publication_summary_json, summary_payload)
_write_json_134(q4_publication_summary_canonical_json, summary_payload)
_write_json_134(q4_manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals
# ----------------------------------------------------------
globals()["CELL134_VERSION"] = CELL134_VERSION
globals()["CELL13_4_Q4_PAIR_SUMMARY_DF"] = pair_summary
globals()["CELL13_4_Q4_TIER_SUMMARY_DF"] = tier_summary
globals()["CELL13_4_Q4_PUBLICATION_BLOCKERS_DF"] = blockers_df
globals()["CELL13_4_Q4_PUBLICATION_SUMMARY"] = summary_payload
globals()["CELL13_4_Q4_PAIR_SUMMARY_CSV"] = q4_pair_summary_csv
globals()["CELL13_4_Q4_TIER_SUMMARY_CSV"] = q4_tier_summary_csv
globals()["CELL13_4_Q4_PUBLICATION_BLOCKERS_CSV"] = q4_publication_blockers_csv
globals()["CELL13_4_Q4_PUBLICATION_SUMMARY_JSON"] = q4_publication_summary_json
globals()["CELL13_4_Q4_PUBLICATION_SUMMARY_CANONICAL_JSON"] = q4_publication_summary_canonical_json
globals()["CELL13_4_Q4_MANIFEST_JSON"] = q4_manifest_json

log(
    "[Cell13.4] Q4 coupling summary complete | "
    f"overall_status={q4_overall_status} | "
    f"pairs_total={pair_total_n} | "
    f"pairs_evaluable={pair_evaluable_n} | "
    f"pair_pass_rate={driver_protocol_pair_pass_rate:.6f} | "
    f"pair_blocker_n={pair_blocker_n} | "
    f"publication_blocker_records={len(blockers_df)}"
)
log(
    "[Cell13.4] Q4 metric summary | "
    f"mean_ETA_similarity={mean_eta_similarity:.6f} | "
    f"mean_lag_similarity={mean_lag_similarity:.6f} | "
    f"mean_lag_peak_error={mean_lag_peak_error:.6f} | "
    f"mean_response_window_rate_error={mean_response_window_rate_error:.6f} | "
    f"mean_cross_corr_mae={mean_cross_corr_mae:.6f} | "
    f"mean_cross_corr_similarity={mean_cross_corr_similarity:.6f}"
)
log(
    "[Cell13.4] Q4 tier summary | "
    f"{tier_summary.to_dict('records')}"
)
log(
    "[Cell13.4] Q4 conclusion | "
    "Generic full-scope cross-modal coupling is a publication blocker under current pipeline; "
    "protocol synthetic generation is not sufficiently conditioned on final IoT driver timing."
)
log(
    "[Cell13.4] Upstream blocker counts carried forward | "
    f"Q3={q3_publication_blocker_n_134} | "
    f"ETA={eta_publication_blocker_n_134} | "
    f"lag_response={lag_publication_blocker_n_134} | "
    f"cross_corr={cross_corr_publication_blocker_n_134}"
)
log(f"[Cell13.4] Saved Q4 pair summary: {q4_pair_summary_csv} | rows={len(pair_summary)}")
log(f"[Cell13.4] Saved Q4 tier summary: {q4_tier_summary_csv} | rows={len(tier_summary)}")
log(f"[Cell13.4] Saved Q4 blockers: {q4_publication_blockers_csv} | rows={len(blockers_df)}")
log(f"[Cell13.4] Saved Q4 publication summary: {q4_publication_summary_json}")
log(f"[Cell13.4] Saved canonical Q4 publication summary: {q4_publication_summary_canonical_json}")
log(
    "[Cell13.4] Contract flags | "
    "TEST_real_values_used_only_in_upstream_QA=True | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | TEST_activity_used_for_pair_selection=False | pair_registry_schema_driven=True"
)
log("--- END: Cell 13.4 - Q4 coupling summary and publication blockers (v1.1 contract-hardened strict) ---")

gc.collect()