# ==========================================================
# CELL 15.5 - Q6 mitigation planning audit
# v1.1 STUDY-THESIS strict mitigation policy planning only, no-Q4/forensic-aware
#
# Role:
#   - Convert Cell 15.4 Q6 release blockers/warnings into a mitigation plan.
#   - Identify role-specific mitigation options:
#       suppress
#       aggregate
#       coarsen
#       disclose
#       improve upstream realism
#       defer public release
#   - Produce a defensible release-risk mitigation table.
#
# Scientific contract:
#   - No fitting.
#   - No materialization.
#   - No synthetic mutation.
#   - No suppression/redaction applied here.
#   - TEST real values are not newly read here; this cell uses prior Q6 audit outputs.
#   - This is a planning/audit cell only.
#
# Outputs:
#   reports/cell15_5_q6_mitigation_plan.csv
#   reports/cell15_5_q6_mitigation_priority_summary.csv
#   reports/cell15_5_q6_public_release_options.csv
#   reports/cell15_5_q6_mitigation_statement.txt
#   reports/cell15_5_q6_mitigation_contract.json
#   artifacts/cell15_5_q6_mitigation_manifest.json
# ==========================================================

log("--- START: Cell 15.5 - Q6 mitigation planning audit (v1.1 no-Q4 forensic-aware strict) ---")

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
_required_155 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF",
    "CELL15_4_Q6_PRIVACY_FINDINGS_DF",
    "CELL15_4_Q6_RELEASE_SUMMARY_DF",
    "CELL15_4_Q6_RELEASE_SAFETY_CONTRACT",
    "CELL15_0_PRIVACY_ROLE_GROUPS",
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT",
    "CELL15_3_MIA_CONTRACT",
]
_missing_155 = [k for k in _required_155 if k not in globals()]
if _missing_155:
    raise RuntimeError(f"[Cell15.5] Missing required globals: {_missing_155}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)

CELL155_VERSION = "cell15_5_q6_mitigation_planning_audit_v1_1_no_q4_forensic_aware"

CFG["cell15_5_version"] = CELL155_VERSION
CFG["cell15_5_TEST_real_values_used"] = False
CFG["cell15_5_TEST_real_values_used_for_materialization"] = False
CFG["cell15_5_synthetic_values_mutated"] = False
CFG["cell15_5_selection_done_here"] = False
CFG["cell15_5_generator_fit_done_here"] = False
CFG["cell15_5_materialization_done_here"] = False
CFG["cell15_5_mitigation_plan_done_here"] = True
CFG["cell15_5_redaction_or_suppression_applied"] = False
CFG["cell15_5_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_5_Q4_coupled_artifacts_used"] = False
CFG["cell15_5_forensics_carried_forward"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell15_5_public_release_default_policy", "blocked_until_mitigated_or_disclosed")
CFG.setdefault("cell15_5_allow_internal_research_use_with_caveats", True)
CFG.setdefault("cell15_5_create_mutation_plan_only", True)
CFG.setdefault("cell15_5_recommend_suppression_for_ota_blocker", True)
CFG.setdefault("cell15_5_recommend_binary_coarsening", True)
CFG.setdefault("cell15_5_recommend_observability_aggregation", True)
CFG.setdefault("cell15_5_recommend_continuous_upstream_realism_rework", True)

# Severity ordering for planning.
SEVERITY_ORDER_155 = {
    "release_blocker": 3,
    "release_warning": 2,
    "release_pass": 1,
    "blocker": 3,
    "warning": 2,
    "pass": 1,
    "not_evaluable": 2,
}

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
mitigation_plan_csv = os.path.join(REPORT_DIR, "cell15_5_q6_mitigation_plan.csv")
priority_summary_csv = os.path.join(REPORT_DIR, "cell15_5_q6_mitigation_priority_summary.csv")
release_options_csv = os.path.join(REPORT_DIR, "cell15_5_q6_public_release_options.csv")
mitigation_statement_txt = os.path.join(REPORT_DIR, "cell15_5_q6_mitigation_statement.txt")
contract_json = os.path.join(REPORT_DIR, "cell15_5_q6_mitigation_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell15_5_q6_mitigation_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell15_5_q6_mitigation_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_155(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_155(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_155(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_155(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_155(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_155(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_155(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_155(obj.to_dict())
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

def _write_json_155(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_155(payload), f, indent=2, sort_keys=True)

def _sha256_file_155(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_155(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.5] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.5] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

def _safe_float_155(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_int_155(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _severity_rank_155(status: str) -> int:
    return int(SEVERITY_ORDER_155.get(str(status), 0))

def _split_reasons_155(x):
    if x is None:
        return []
    s = str(x)
    if not s or s.lower() in {"nan", "none"}:
        return []
    return [r.strip() for r in s.split("|") if r.strip()]

def _role_family_155(role: str) -> str:
    r = str(role)
    if r.startswith("protocol_"):
        return "protocol"
    if r.startswith("iot_"):
        return "iot"
    if r == "full_cps_mixed_sampled":
        return "mixed"
    return "other"

def _risk_type_from_role_reason_155(role: str, reason: str) -> str:
    role = str(role)
    reason = str(reason)

    if "no_copy" in reason or "copy" in reason:
        return "copy_or_near_copy_risk"
    if "DCR" in reason or "NNDR" in reason or "close_neighbor" in reason:
        return "nearest_neighbor_privacy_risk"
    if "synthetic_distinguishability" in reason or "MIA" in reason:
        return "synthetic_distinguishability_risk"
    if "temporal_drift" in reason:
        return "temporal_drift_context"
    if "calibration" in reason:
        return "calibration_uncertainty"
    if role == "full_cps_mixed_sampled":
        return "inherited_mixed_role_risk"
    return "general_release_risk"

def _recommended_action_155(role: str, risk_type: str, release_status: str) -> tuple:
    """
    Returns:
      mitigation_action
      mitigation_priority
      mitigation_rationale
      possible_next_cell
      public_release_handling
    """
    role = str(role)
    risk_type = str(risk_type)
    release_status = str(release_status)

    if role == "protocol_ota":
        return (
            "suppress_or_aggregate_ota_active_windows",
            "P0",
            "OTA active-window DCR/NNDR indicates near-identical active OTA windows; suppress or aggregate OTA features before public release.",
            "15.6_optional_ota_suppression_or_aggregation",
            "exclude_or_aggregate_ota_from_public_release",
        )

    if role == "iot_binary_states":
        return (
            "coarsen_or_suppress_repeated_binary_state_windows",
            "P0",
            "Binary-state windows show no-copy/near-copy risk and high synthetic distinguishability; coarsen temporal resolution or suppress high-risk binary state columns/windows.",
            "15.6_optional_binary_state_coarsening",
            "coarsen_or_remove_binary_state_streams_from_public_release",
        )

    if role == "iot_observability_masks":
        return (
            "aggregate_or_remove_fine_grained_observability_masks",
            "P1",
            "Observability/staleness masks are highly distinguishable and may reveal capture/device availability routines.",
            "15.7_optional_observability_aggregation",
            "aggregate_or_remove_observability_masks_from_public_release",
        )

    if role == "iot_continuous_values":
        return (
            "improve_upstream_iot_continuous_realism_or_disclose_distinguishability",
            "P1",
            "Continuous IoT values are highly separable from real training data; this is primarily realism/distinguishability risk rather than direct copy evidence.",
            "future_upstream_iot_value_realism_rework",
            "release_only_with_explicit_distinguishability_limitation_or_after_rework",
        )

    if role == "full_cps_mixed_sampled":
        return (
            "inherits_iot_role_mitigations",
            "P1",
            "Mixed CPS sample risk is inherited from IoT continuous/binary/observability role risks; mitigate source roles first.",
            "after_role_specific_mitigation_reaudit_full_cps_mixed",
            "do_not_use_mixed_full_cps_release_claim_until_source_roles_mitigated",
        )

    if role == "iot_event_drivers":
        return (
            "document_warning_and_monitor_event_driver_DCR_calibration",
            "P2",
            "Event drivers are distinguishability-clean but carry DCR/NNDR calibration warning; document and re-audit if release policy is strict.",
            "optional_event_driver_privacy_recheck",
            "release_with_caveat_if_no_other_blockers",
        )

    if role == "protocol_router":
        return (
            "document_temporal_drift_warning",
            "P2",
            "Router synthetic-vs-real distinguishability and DCR are acceptable; warning is temporal drift context only.",
            "no_immediate_mitigation_required",
            "release_with_temporal_drift_caveat_if_no_other_blockers",
        )

    if role == "protocol_zigbee":
        return (
            "document_temporal_drift_warning",
            "P2",
            "Zigbee synthetic-vs-real distinguishability and DCR are acceptable; warning is temporal drift context only.",
            "no_immediate_mitigation_required",
            "release_with_temporal_drift_caveat_if_no_other_blockers",
        )

    if release_status == "release_blocker":
        return (
            "manual_review_required",
            "P1",
            "Release blocker requires manual mitigation planning.",
            "manual_q6_mitigation_review",
            "do_not_release_without_review",
        )

    if release_status == "release_warning":
        return (
            "document_warning",
            "P2",
            "Warning-level privacy risk should be disclosed and monitored.",
            "no_immediate_mitigation_required",
            "release_with_caveat_if_no_blockers",
        )

    return (
        "no_mitigation_required",
        "P3",
        "Role passed current Q6 release-safety gates.",
        "none",
        "release_eligible_under_current_q6_audits",
    )

def _suppression_impact_155(role: str, role_cols_n: int) -> str:
    role = str(role)
    n = int(role_cols_n)

    if role == "protocol_ota":
        return f"Low dimensional impact: {n} OTA columns; suppressing/aggregating may reduce wireless-layer fidelity but protects OTA privacy risk."
    if role == "iot_binary_states":
        return f"Moderate impact: {n} binary-state columns; coarsening may reduce state-transition utility but protects household routine leakage."
    if role == "iot_observability_masks":
        return f"High interpretability impact: {n} observability columns; aggregation protects availability routines but may reduce missingness realism analysis."
    if role == "iot_continuous_values":
        return f"High utility impact: {n} continuous/value columns; upstream realism improvement preferred over blanket suppression."
    if role == "full_cps_mixed_sampled":
        return "No direct suppression recommended; mixed risk should be resolved by mitigating source role groups."
    return f"Limited direct impact estimate for {n} columns."

# ----------------------------------------------------------
# 4) Validate upstream no-Q4 and forensic-aware Q6 contracts
# ----------------------------------------------------------
version150_155 = _require_contract_version_155(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
version151a_155 = _require_contract_version_155(
    CELL15_1A_NO_COPY_FORENSIC_CONTRACT,
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "cell15_1a_q6_no_copy_blocker_forensic_v1_0",
)
version152a_155 = _require_contract_version_155(
    CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT,
    "CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT",
    "cell15_2a_q6_dcr_nndr_blocker_forensic_v1_0",
)
version153_155 = _require_contract_version_155(
    CELL15_3_MIA_CONTRACT,
    "CELL15_3_MIA_CONTRACT",
    "cell15_3_q6_mia_random_temporal_privacy_v1_2",
)
version154_155 = _require_contract_version_155(
    CELL15_4_Q6_RELEASE_SAFETY_CONTRACT,
    "CELL15_4_Q6_RELEASE_SAFETY_CONTRACT",
    "cell15_4_q6_privacy_release_safety_summary_v1_2",
)

strict150_155 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
if str(strict150_155.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.5] Cell 15.0 does not carry Q4_final_status=blocked_no_promotion.")
if bool(strict150_155.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.5] Cell 15.0 used Q4-coupled artifacts unexpectedly.")

for name, contract in [
    ("Cell15.1a", CELL15_1A_NO_COPY_FORENSIC_CONTRACT),
    ("Cell15.2a", CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT),
]:
    strict = contract.get("strict_contract", {})
    if bool(strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell15.5] {name} indicates synthetic mutation.")
    if not bool(strict.get("diagnostic_only", False)):
        raise RuntimeError(f"[Cell15.5] {name} did not declare diagnostic_only.")

strict154_155 = CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("strict_contract", {})
if str(strict154_155.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.5] Cell 15.4 does not carry Q4_final_status=blocked_no_promotion.")
if bool(strict154_155.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.5] Cell 15.4 used Q4-coupled artifacts unexpectedly.")
if bool(strict154_155.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell15.5] Cell 15.4 indicates synthetic mutation.")

# ----------------------------------------------------------
# 5) Load Q6 role-level evidence
# ----------------------------------------------------------
role_release_df = CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF.copy()
findings_df = CELL15_4_Q6_PRIVACY_FINDINGS_DF.copy()
summary_df = CELL15_4_Q6_RELEASE_SUMMARY_DF.copy()

if not isinstance(role_release_df, pd.DataFrame) or "role_group" not in role_release_df.columns:
    raise RuntimeError("[Cell15.5] Invalid Cell 15.4 role release safety table.")

if not isinstance(findings_df, pd.DataFrame):
    findings_df = pd.DataFrame()

log(
    "[Cell15.5] Building Q6 mitigation plan | "
    f"roles={len(role_release_df)} | findings={len(findings_df)}"
)

# ----------------------------------------------------------
# 6) Build mitigation plan
# ----------------------------------------------------------
plan_rows = []

for _, row in role_release_df.iterrows():
    role = str(row.get("role_group", ""))
    release_status = str(row.get("release_status", ""))
    release_reasons = _split_reasons_155(row.get("release_reasons", ""))

    role_cols_n = _safe_int_155(row.get("role_cols_n"), default=len(CELL15_0_PRIVACY_ROLE_GROUPS.get(role, [])))
    role_family = _role_family_155(role)

    if not release_reasons:
        release_reasons = ["within_Q6_release_safety_gates"]

    # One row per reason/risk.
    for reason in release_reasons:
        risk_type = _risk_type_from_role_reason_155(role, reason)
        action, priority, rationale, next_cell, public_handling = _recommended_action_155(
            role=role,
            risk_type=risk_type,
            release_status=release_status,
        )

        plan_rows.append({
            "role_group": role,
            "role_family": role_family,
            "role_cols_n": int(role_cols_n),
            "release_status": release_status,
            "release_reason": reason,
            "risk_type": risk_type,
            "mitigation_priority": priority,
            "mitigation_action": action,
            "mitigation_rationale": rationale,
            "possible_next_cell": next_cell,
            "public_release_handling": public_handling,
            "suppression_or_aggregation_impact": _suppression_impact_155(role, role_cols_n),

            # Key metrics carried forward when present.
            "no_copy_exact_nonzero_match_rate": _safe_float_155(row.get("no_copy_exact_nonzero_match_rate"), np.nan),
            "no_copy_near_copy_rate": _safe_float_155(row.get("no_copy_near_copy_rate"), np.nan),
            "syn_train_dcr_p01_ratio_active": _safe_float_155(row.get("syn_train_dcr_p01_ratio_active"), np.nan),
            "close_neighbor_rate_active": _safe_float_155(row.get("close_neighbor_rate_active"), np.nan),
            "syn_to_train_nndr_active_p01": _safe_float_155(row.get("syn_to_train_nndr_active_p01"), np.nan),
            "synthetic_distinguishability_max_auc": _safe_float_155(row.get("synthetic_distinguishability_max_auc"), np.nan),
            "synthetic_distinguishability_max_balanced_accuracy": _safe_float_155(row.get("synthetic_distinguishability_max_balanced_accuracy"), np.nan),
            "temporal_drift_max_auc": _safe_float_155(row.get("temporal_drift_max_auc"), np.nan),
            "temporal_drift_max_balanced_accuracy": _safe_float_155(row.get("temporal_drift_max_balanced_accuracy"), np.nan),

            "plan_only_no_mutation": True,
            "TEST_real_values_used_here": False,
        })

mitigation_plan_df = pd.DataFrame(plan_rows)

# Enforce deterministic priority ordering.
priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
mitigation_plan_df["_priority_rank"] = mitigation_plan_df["mitigation_priority"].map(priority_rank).fillna(9).astype(int)
mitigation_plan_df["_severity_rank"] = mitigation_plan_df["release_status"].map(lambda x: _severity_rank_155(str(x))).astype(int)
mitigation_plan_df = (
    mitigation_plan_df
    .sort_values(["_priority_rank", "_severity_rank", "role_group", "risk_type"], ascending=[True, False, True, True])
    .drop(columns=["_priority_rank", "_severity_rank"])
    .reset_index(drop=True)
)

# ----------------------------------------------------------
# 7) Priority summary
# ----------------------------------------------------------
priority_summary_rows = []

if len(mitigation_plan_df):
    for priority, g in mitigation_plan_df.groupby("mitigation_priority", dropna=False):
        priority_summary_rows.append({
            "mitigation_priority": str(priority),
            "rows_n": int(len(g)),
            "roles_n": int(g["role_group"].nunique()),
            "roles": "|".join(sorted(g["role_group"].astype(str).unique().tolist())),
            "actions": "|".join(sorted(g["mitigation_action"].astype(str).unique().tolist())),
        })

    for action, g in mitigation_plan_df.groupby("mitigation_action", dropna=False):
        priority_summary_rows.append({
            "mitigation_priority": "by_action",
            "rows_n": int(len(g)),
            "roles_n": int(g["role_group"].nunique()),
            "roles": "|".join(sorted(g["role_group"].astype(str).unique().tolist())),
            "actions": str(action),
        })

priority_summary_df = pd.DataFrame(priority_summary_rows)

# ----------------------------------------------------------
# 8) Public release option table
# ----------------------------------------------------------
blocker_roles = sorted(
    role_release_df.loc[
        role_release_df["release_status"].astype(str).eq("release_blocker"),
        "role_group",
    ]
    .astype(str)
    .tolist()
)

warning_roles = sorted(
    role_release_df.loc[
        role_release_df["release_status"].astype(str).eq("release_warning"),
        "role_group",
    ]
    .astype(str)
    .tolist()
)

release_options_rows = [
    {
        "option_id": "Q6_RELEASE_OPTION_0",
        "option_name": "do_not_publicly_release_current_artifact",
        "recommended": True,
        "description": "Keep current final Q4-coupled dataset as internal/research artifact only until Q6 blockers are mitigated or explicitly disclosed.",
        "required_actions": "none_if_internal_only",
        "residual_risk": "blockers_remain_for_public_release",
        "scientific_defensibility": "high",
    },
    {
        "option_id": "Q6_RELEASE_OPTION_1",
        "option_name": "release_with_explicit_q6_risk_disclosure",
        "recommended": False,
        "description": "Publicly release current artifact but disclose Q6 blockers for OTA, binary states, continuous IoT values, observability masks, and mixed CPS samples.",
        "required_actions": "include_q6_release_statement_and_role_level_risk_table",
        "residual_risk": "high",
        "scientific_defensibility": "medium_if_disclosure_is_prominent",
    },
    {
        "option_id": "Q6_RELEASE_OPTION_2",
        "option_name": "mitigated_public_release",
        "recommended": True,
        "description": "Create a separate public-release artifact with OTA aggregation/suppression, binary-state coarsening, and observability aggregation/removal.",
        "required_actions": "implement_mitigation_cell_then_rerun_15_1_to_15_4",
        "residual_risk": "lower_but_requires_reaudit",
        "scientific_defensibility": "high_after_reaudit",
    },
    {
        "option_id": "Q6_RELEASE_OPTION_3",
        "option_name": "role-restricted_release",
        "recommended": True,
        "description": "Release only role groups with warning/pass-level Q6 status, e.g., protocol_router, protocol_zigbee, and event drivers, while withholding high-risk IoT roles and OTA.",
        "required_actions": "construct_role_restricted_artifact_and_reaudit",
        "residual_risk": "moderate",
        "scientific_defensibility": "medium_to_high_after_reaudit",
    },
]

release_options_df = pd.DataFrame(release_options_rows)

# ----------------------------------------------------------
# 9) Statement
# ----------------------------------------------------------
q6_status = str(CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("overall", {}).get("q6_overall_status", "unknown"))
q6_recommendation = str(CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("overall", {}).get("final_recommendation", "unknown"))

statement_lines = [
    "Q6 Mitigation Planning Statement",
    "================================",
    "",
    f"Current Q6 status: {q6_status}",
    f"Current Q6 recommendation: {q6_recommendation}",
    "",
    "This cell is a mitigation planning audit only. It does not mutate, suppress, redact, or materialize any synthetic data.",
    "",
    "Priority P0 mitigations:",
]

p0 = mitigation_plan_df[mitigation_plan_df["mitigation_priority"].astype(str).eq("P0")].copy()
if len(p0):
    for _, r in p0.iterrows():
        statement_lines.append(
            f"- {r['role_group']}: {r['mitigation_action']} "
            f"({r['release_reason']})"
        )
else:
    statement_lines.append("- None")

statement_lines.extend([
    "",
    "Priority P1 mitigations:",
])

p1 = mitigation_plan_df[mitigation_plan_df["mitigation_priority"].astype(str).eq("P1")].copy()
if len(p1):
    for _, r in p1.iterrows():
        statement_lines.append(
            f"- {r['role_group']}: {r['mitigation_action']} "
            f"({r['release_reason']})"
        )
else:
    statement_lines.append("- None")

statement_lines.extend([
    "",
    "Recommended release path:",
    "- Do not claim global Q6 privacy pass for the current artifact.",
    "- Keep the current no-Q4 synthetic artifact for internal scientific evaluation; Q4 remains blocked_no_promotion.",
    "- For public release, either disclose Q6 blockers prominently or create a mitigated/restricted release artifact and rerun Q6 audits.",
    "",
    "Most defensible next technical step:",
    "Create an optional mitigation cell that produces a separate public-release candidate, not an overwrite of the current final no-Q4 artifact.",
    "",
    "Candidate mitigation sequence:",
    "1. OTA: suppress or aggregate active-window OTA features.",
    "2. Binary states: coarsen temporal resolution or suppress repeated binary-state windows.",
    "3. Observability masks: aggregate/remove fine-grained observability and staleness columns.",
    "4. Re-run Cells 15.1–15.4 on the mitigated artifact.",
])

mitigation_statement = "\n".join(statement_lines)

with open(mitigation_statement_txt, "w", encoding="utf-8") as f:
    f.write(mitigation_statement)

# ----------------------------------------------------------
# 10) Save CSV outputs
# ----------------------------------------------------------
mitigation_plan_df.to_csv(mitigation_plan_csv, index=False)
priority_summary_df.to_csv(priority_summary_csv, index=False)
release_options_df.to_csv(release_options_csv, index=False)

# ----------------------------------------------------------
# 11) Contract / manifest
# ----------------------------------------------------------
p_counts = mitigation_plan_df["mitigation_priority"].astype(str).value_counts().sort_index().to_dict() if len(mitigation_plan_df) else {}
action_counts = mitigation_plan_df["mitigation_action"].astype(str).value_counts().sort_index().to_dict() if len(mitigation_plan_df) else {}

contract = {
    "cell": "15.5",
    "version": CELL155_VERSION,
    "role": "q6_mitigation_planning_audit",
    "quality_dimension": "Q6_privacy_no_copy_release_safety",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False
    },
    "upstream_contract_versions": {
        "cell15_0": version150_155,
        "cell15_1a": version151a_155,
        "cell15_2a": version152a_155,
        "cell15_3": version153_155,
        "cell15_4": version154_155
    },
    "plan_scope": {
        "plan_only": True,
        "redaction_or_suppression_applied": False,
        "synthetic_values_mutated": False,
        "new_release_artifact_created": False,
    },
    "input_q6_status": {
        "q6_overall_status": q6_status,
        "q6_recommendation": q6_recommendation,
        "blocker_roles": blocker_roles,
        "warning_roles": warning_roles,
    },
    "mitigation_summary": {
        "plan_rows_n": int(len(mitigation_plan_df)),
        "priority_counts": p_counts,
        "action_counts": action_counts,
        "p0_roles": sorted(p0["role_group"].astype(str).unique().tolist()) if len(p0) else [],
        "p1_roles": sorted(p1["role_group"].astype(str).unique().tolist()) if len(p1) else [],
    },
    "release_options": release_options_df.to_dict("records"),
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "forensics_carried_forward": True,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "redaction_or_suppression_applied": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "mitigation_plan_done_here": True,
    },
    "outputs": {
        "mitigation_plan_csv": mitigation_plan_csv,
        "priority_summary_csv": priority_summary_csv,
        "release_options_csv": release_options_csv,
        "mitigation_statement_txt": mitigation_statement_txt,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_155(contract_json, contract)
_write_json_155(contract_canonical_json, contract)

manifest = {
    "cell": "15.5",
    "version": CELL155_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "plan_scope": contract["plan_scope"],
    "input_q6_status": contract["input_q6_status"],
    "mitigation_summary": contract["mitigation_summary"],
    "strict_contract": contract["strict_contract"],
}

_write_json_155(manifest_json, manifest)

hashes = {
    "mitigation_plan_csv_sha256": _sha256_file_155(mitigation_plan_csv),
    "priority_summary_csv_sha256": _sha256_file_155(priority_summary_csv),
    "release_options_csv_sha256": _sha256_file_155(release_options_csv),
    "mitigation_statement_txt_sha256": _sha256_file_155(mitigation_statement_txt),
    "contract_json_sha256": _sha256_file_155(contract_json),
    "contract_canonical_json_sha256": _sha256_file_155(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_155(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_155(contract_json, contract)
_write_json_155(contract_canonical_json, contract)
_write_json_155(manifest_json, manifest)

# ----------------------------------------------------------
# 12) Export globals
# ----------------------------------------------------------
globals()["CELL155_VERSION"] = CELL155_VERSION
globals()["CELL15_5_Q6_MITIGATION_PLAN_DF"] = mitigation_plan_df
globals()["CELL15_5_Q6_MITIGATION_PRIORITY_SUMMARY_DF"] = priority_summary_df
globals()["CELL15_5_Q6_PUBLIC_RELEASE_OPTIONS_DF"] = release_options_df
globals()["CELL15_5_Q6_MITIGATION_STATEMENT"] = mitigation_statement
globals()["CELL15_5_Q6_MITIGATION_CONTRACT"] = contract

globals()["CELL15_5_Q6_MITIGATION_PLAN_CSV"] = mitigation_plan_csv
globals()["CELL15_5_Q6_MITIGATION_PRIORITY_SUMMARY_CSV"] = priority_summary_csv
globals()["CELL15_5_Q6_PUBLIC_RELEASE_OPTIONS_CSV"] = release_options_csv
globals()["CELL15_5_Q6_MITIGATION_STATEMENT_TXT"] = mitigation_statement_txt
globals()["CELL15_5_Q6_MITIGATION_CONTRACT_JSON"] = contract_json
globals()["CELL15_5_Q6_MITIGATION_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL15_5_Q6_MITIGATION_MANIFEST_JSON"] = manifest_json

log(
    "[Cell15.5] Q6 mitigation planning audit complete | "
    f"input_q6_status={q6_status} | "
    f"plan_rows={len(mitigation_plan_df)} | "
    f"priority_counts={p_counts}"
)
log(f"[Cell15.5] P0 roles | {sorted(p0['role_group'].astype(str).unique().tolist()) if len(p0) else []}")
log(f"[Cell15.5] P1 roles | {sorted(p1['role_group'].astype(str).unique().tolist()) if len(p1) else []}")
log(f"[Cell15.5] Saved mitigation plan: {mitigation_plan_csv} | rows={len(mitigation_plan_df)}")
log(f"[Cell15.5] Saved public release options: {release_options_csv} | rows={len(release_options_df)}")
log(f"[Cell15.5] Saved mitigation statement: {mitigation_statement_txt}")
log(f"[Cell15.5] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell15.5] Contract flags | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "redaction_or_suppression_applied=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "mitigation_plan_done_here=True | Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False | forensics_carried_forward=True"
)
log("--- END: Cell 15.5 - Q6 mitigation planning audit (v1.1 no-Q4 forensic-aware strict) ---")

gc.collect()