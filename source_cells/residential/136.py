# ==========================================================
# CELL 15.4 - Q6 privacy summary and release-safety manifest
# v1.2 STUDY-THESIS strict no-Q4 / forensic-aware privacy release decision
#
# Role:
#   - Consolidate Q6 privacy/no-copy evidence from:
#       15.1  no-copy window audit
#       15.1a no-copy blocker forensic diagnostic
#       15.2  active-window DCR / NNDR audit
#       15.2a DCR / NNDR blocker forensic diagnostic
#       15.3  separated MIA / distinguishability / temporal-drift audit
#   - Preserve Q4 governance:
#       final_q4_status = blocked_no_promotion
#       q4_coupled_artifacts_used = False
#   - Assign role-level release safety status:
#       release_pass
#       release_warning
#       release_blocker
#   - Produce final Q6 release-safety manifest.
#
# Scientific contract:
#   - No fitting.
#   - No materialization.
#   - No synthetic mutation.
#   - TEST real values are used only through upstream Q6 privacy reference audits.
#   - This cell summarizes release risk; it does not claim formal privacy proof.
#
# Outputs:
#   reports/cell15_4_q6_role_release_safety.csv
#   reports/cell15_4_q6_privacy_findings.csv
#   reports/cell15_4_q6_release_safety_summary.csv
#   reports/cell15_4_q6_release_statement.txt
#   reports/cell15_4_q6_release_safety_contract.json
#   artifacts/contracts/cell15_4_q6_release_safety_contract_v1_2_THESIS.json
#   artifacts/cell15_4_q6_release_safety_manifest.json
# ==========================================================

log("--- START: Cell 15.4 - Q6 privacy summary and release-safety manifest (v1.2 forensic-aware strict) ---")

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
_required_154 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL15_0_PRIVACY_ROLE_GROUPS",
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "CELL15_1_NO_COPY_WINDOW_METRICS_DF",
    "CELL15_1_NO_COPY_CONTRACT",
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "CELL15_1A_NO_COPY_BLOCKER_FORENSIC_DF",
    "CELL15_2_DCR_NNDR_ROLE_METRICS_DF",
    "CELL15_2_DCR_NNDR_CONTRACT",
    "CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT",
    "CELL15_2A_DCR_NNDR_BLOCKER_FORENSIC_DF",
    "CELL15_3_MIA_ROLE_METRICS_DF",
    "CELL15_3_MIA_CONTRACT",
]
_missing_154 = [k for k in _required_154 if k not in globals()]
if _missing_154:
    raise RuntimeError(f"[Cell15.4] Missing required globals: {_missing_154}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_154 = str(OUT_SYN)
REPORT_DIR_ACTIVE_154 = str(REPORT_DIR)

def _resolve_project_root_154(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        if "q6_public_reaudit" in parts:
            candidates.append(os.sep.join(parts[:parts.index("q6_public_reaudit")]))
        elif os.path.basename(p) in {"reports", "synthetic", "artifacts"}:
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
    raise RuntimeError("[Cell15.4] Could not resolve canonical project root.")

PROJECT_ROOT_154 = _resolve_project_root_154(OUTDIR, REPORT_DIR_ACTIVE_154, OUT_SYN_ACTIVE_154)
REPORT_DIR = os.path.join(PROJECT_ROOT_154, "reports")
OUT_SYN = os.path.join(PROJECT_ROOT_154, "synthetic")
ARTDIR = os.path.join(PROJECT_ROOT_154, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)

CELL154_VERSION = "cell15_4_q6_privacy_release_safety_summary_v1_2_forensic_aware_no_q4"

CFG["cell15_4_version"] = CELL154_VERSION
CFG["cell15_4_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_4_Q4_coupled_artifacts_used"] = False
CFG["cell15_4_TEST_real_values_used_for_privacy_reference_only"] = True
CFG["cell15_4_synthetic_values_mutated"] = False
CFG["cell15_4_selection_done_here"] = False
CFG["cell15_4_generator_fit_done_here"] = False
CFG["cell15_4_materialization_done_here"] = False
CFG["cell15_4_release_decision_done_here"] = True
CFG["cell15_4_forensics_carried_forward"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell15_4_fail_on_missing_role_from_audit", False)
CFG.setdefault("cell15_4_release_requires_no_blockers", True)
CFG.setdefault("cell15_4_treat_synthetic_distinguishability_blocker_as_release_blocker", True)
CFG.setdefault("cell15_4_treat_temporal_drift_warning_as_release_warning", True)
CFG.setdefault("cell15_4_allow_release_with_warnings", True)
CFG.setdefault("cell15_4_final_recommendation_if_blockers", "not_release_ready_without_privacy_mitigation_or_scope_suppression")
CFG.setdefault("cell15_4_final_recommendation_if_warnings_only", "release_with_privacy_caveats_and_documented_warnings")
CFG.setdefault("cell15_4_final_recommendation_if_pass", "release_ready_under_current_q6_audits")

# Keep this optional but conservative: mixed composite can be downgraded only
# if its source roles are not blockers and synthetic distinguishability is not a blocker.
CFG.setdefault("cell15_4_downgrade_mixed_composite_degenerate_dcr", True)
CFG.setdefault("cell15_4_mixed_composite_source_roles", [
    "protocol_router",
    "protocol_zigbee",
    "iot_event_drivers",
])

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
role_release_csv = os.path.join(REPORT_DIR, "cell15_4_q6_role_release_safety.csv")
privacy_findings_csv = os.path.join(REPORT_DIR, "cell15_4_q6_privacy_findings.csv")
release_summary_csv = os.path.join(REPORT_DIR, "cell15_4_q6_release_safety_summary.csv")
release_statement_txt = os.path.join(REPORT_DIR, "cell15_4_q6_release_statement.txt")
contract_json = os.path.join(REPORT_DIR, "cell15_4_q6_release_safety_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell15_4_q6_release_safety_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell15_4_q6_release_safety_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_154(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_154(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_154(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_154(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_154(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_154(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_154(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_154(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_154(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_154(payload), f, indent=2, sort_keys=True)

def _sha256_file_154(path: str) -> str:
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_154(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.4] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.4] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

def _safe_float_154(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_int_154(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _get_row_by_role_154(df: pd.DataFrame, role: str):
    if not isinstance(df, pd.DataFrame) or "role_group" not in df.columns:
        return None
    d = df[df["role_group"].astype(str).eq(str(role))]
    if len(d) == 0:
        return None
    return d.iloc[0].to_dict()

def _as_status_154(x, default="not_evaluable"):
    s = str(x if x is not None else "").strip()
    return s if s else default

def _combine_release_status_154(
    no_copy_status,
    dcr_status,
    mia_final_status,
    synthetic_dist_status,
    temporal_status,
    forensic_no_copy_blocker,
    forensic_dcr_blocker,
):
    reasons = []
    warnings = []

    if str(no_copy_status) == "blocker":
        reasons.append("no_copy_window_blocker")
    elif str(no_copy_status) == "warning":
        warnings.append("no_copy_window_warning")

    if bool(forensic_no_copy_blocker):
        reasons.append("forensic_no_copy_blocker")

    if str(dcr_status) == "blocker":
        reasons.append("active_DCR_NNDR_blocker")
    elif str(dcr_status) == "warning":
        warnings.append("active_DCR_NNDR_warning")

    if bool(forensic_dcr_blocker):
        reasons.append("forensic_DCR_NNDR_blocker")

    if str(mia_final_status) == "blocker":
        reasons.append("MIA_release_blocker")
    elif str(mia_final_status) == "warning":
        warnings.append("MIA_release_warning")

    if (
        bool(CFG.get("cell15_4_treat_synthetic_distinguishability_blocker_as_release_blocker", True))
        and str(synthetic_dist_status) == "blocker"
    ):
        reasons.append("synthetic_distinguishability_blocker")
    elif str(synthetic_dist_status) == "warning":
        warnings.append("synthetic_distinguishability_warning")

    if (
        bool(CFG.get("cell15_4_treat_temporal_drift_warning_as_release_warning", True))
        and str(temporal_status) == "warning"
    ):
        warnings.append("temporal_drift_warning_context_only")

    if reasons:
        return "release_blocker", "|".join(sorted(set(reasons)))

    if warnings:
        return "release_warning", "|".join(sorted(set(warnings)))

    return "release_pass", "within_Q6_release_safety_gates"

def _finding_rows_for_role_154(role, audit_name, status, reasons, severity_override=None, extra=None):
    status = str(status)
    reasons = str(reasons or "")
    if severity_override is not None:
        severity = str(severity_override)
    elif status in {"blocker", "release_blocker"}:
        severity = "blocker"
    elif status in {"warning", "release_warning", "not_evaluable"}:
        severity = "warning"
    else:
        severity = "pass"

    if not reasons:
        reasons = "none"

    out = []
    for reason in reasons.split("|"):
        reason = reason.strip()
        if not reason:
            continue
        rec = {
            "role_group": str(role),
            "audit": str(audit_name),
            "severity": severity,
            "status": status,
            "finding": reason,
        }
        if isinstance(extra, dict):
            rec.update(extra)
        out.append(rec)
    return out

# ----------------------------------------------------------
# 4) Validate upstream contracts
# ----------------------------------------------------------
version150_154 = _require_contract_version_154(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
version151_154 = _require_contract_version_154(
    CELL15_1_NO_COPY_CONTRACT,
    "CELL15_1_NO_COPY_CONTRACT",
    "cell15_1_q6_no_copy_window_audit_v1_1",
)
version151a_154 = _require_contract_version_154(
    CELL15_1A_NO_COPY_FORENSIC_CONTRACT,
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "cell15_1a_q6_no_copy_blocker_forensic_v1_0",
)
version152_154 = _require_contract_version_154(
    CELL15_2_DCR_NNDR_CONTRACT,
    "CELL15_2_DCR_NNDR_CONTRACT",
    "cell15_2_q6_dcr_nndr_role_specific_privacy_v1_2",
)
version152a_154 = _require_contract_version_154(
    CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT,
    "CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT",
    "cell15_2a_q6_dcr_nndr_blocker_forensic_v1_0",
)
version153_154 = _require_contract_version_154(
    CELL15_3_MIA_CONTRACT,
    "CELL15_3_MIA_CONTRACT",
    "cell15_3_q6_mia_random_temporal_privacy_v1_2",
)

strict150 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
if str(strict150.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.4] Cell 15.0 does not carry Q4_final_status=blocked_no_promotion.")
if bool(strict150.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.4] Cell 15.0 used Q4-coupled artifacts unexpectedly.")

for name, contract, key in [
    ("Cell15.1", CELL15_1_NO_COPY_CONTRACT, "no_copy_window_audit_done_here"),
    ("Cell15.2", CELL15_2_DCR_NNDR_CONTRACT, "dcr_nndr_audit_done_here"),
    ("Cell15.3", CELL15_3_MIA_CONTRACT, "mia_audit_done_here"),
]:
    strict = contract.get("strict_contract", {})
    if bool(strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell15.4] {name} indicates synthetic mutation.")
    if not bool(strict.get(key, False)):
        raise RuntimeError(f"[Cell15.4] {name} did not complete required audit.")

for name, contract in [
    ("Cell15.1a", CELL15_1A_NO_COPY_FORENSIC_CONTRACT),
    ("Cell15.2a", CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT),
]:
    strict = contract.get("strict_contract", {})
    if bool(strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell15.4] {name} indicates synthetic mutation.")
    if not bool(strict.get("diagnostic_only", False)):
        raise RuntimeError(f"[Cell15.4] {name} did not declare diagnostic_only.")

# ----------------------------------------------------------
# 5) Load upstream audit tables
# ----------------------------------------------------------
role_groups = CELL15_0_PRIVACY_ROLE_GROUPS
no_copy_df = CELL15_1_NO_COPY_WINDOW_METRICS_DF.copy()
no_copy_forensic_df = CELL15_1A_NO_COPY_BLOCKER_FORENSIC_DF.copy()
dcr_df = CELL15_2_DCR_NNDR_ROLE_METRICS_DF.copy()
dcr_forensic_df = CELL15_2A_DCR_NNDR_BLOCKER_FORENSIC_DF.copy()
mia_df = CELL15_3_MIA_ROLE_METRICS_DF.copy()

for d, name in [
    (no_copy_df, "15.1 no-copy"),
    (no_copy_forensic_df, "15.1a no-copy forensic"),
    (dcr_df, "15.2 DCR/NNDR"),
    (dcr_forensic_df, "15.2a DCR/NNDR forensic"),
    (mia_df, "15.3 MIA"),
]:
    if not isinstance(d, pd.DataFrame) or "role_group" not in d.columns:
        raise RuntimeError(f"[Cell15.4] Invalid upstream audit table: {name}")

all_roles = sorted(set(map(str, role_groups.keys())) - {"iot_placeholders_or_excluded"})

log(
    "[Cell15.4] Consolidating Q6 privacy evidence | "
    f"roles={len(all_roles)} | audits=['15.1','15.1a','15.2','15.2a','15.3']"
)

# ----------------------------------------------------------
# 6) Consolidate role-level release status
# ----------------------------------------------------------
role_rows = []
finding_rows = []
missing_audit_rows = []

for role in all_roles:
    no_copy = _get_row_by_role_154(no_copy_df, role)
    no_copy_f = _get_row_by_role_154(no_copy_forensic_df, role)
    dcr = _get_row_by_role_154(dcr_df, role)
    dcr_f = _get_row_by_role_154(dcr_forensic_df, role)
    mia = _get_row_by_role_154(mia_df, role)

    if no_copy is None:
        missing_audit_rows.append({"role_group": role, "missing_audit": "15.1_no_copy"})
        no_copy = {"status": "not_evaluable", "reasons": "missing_no_copy_audit"}

    if no_copy_f is None:
        missing_audit_rows.append({"role_group": role, "missing_audit": "15.1a_no_copy_forensic"})
        no_copy_f = {"status": "not_evaluable", "reasons": "missing_no_copy_forensic"}

    if dcr is None:
        missing_audit_rows.append({"role_group": role, "missing_audit": "15.2_dcr_nndr"})
        dcr = {"status": "not_evaluable", "reasons": "missing_dcr_nndr_audit"}

    if dcr_f is None:
        missing_audit_rows.append({"role_group": role, "missing_audit": "15.2a_dcr_nndr_forensic"})
        dcr_f = {"status": "not_evaluable", "reasons": "missing_dcr_nndr_forensic"}

    if mia is None:
        missing_audit_rows.append({"role_group": role, "missing_audit": "15.3_mia"})
        mia = {
            "status": "not_evaluable",
            "reasons": "missing_mia_audit",
            "synthetic_distinguishability_status": "not_evaluable",
            "synthetic_distinguishability_reasons": "missing_mia_audit",
            "temporal_drift_status": "not_evaluable",
            "temporal_drift_reasons": "missing_mia_audit",
            "carried_privacy_status": "not_evaluable",
            "carried_privacy_reasons": "missing_mia_audit",
            "final_mia_release_status": "not_evaluable",
            "final_mia_release_reasons": "missing_mia_audit",
        }

    no_copy_status = _as_status_154(no_copy.get("status"))
    no_copy_reasons = str(no_copy.get("reasons", ""))

    no_copy_forensic_status = _as_status_154(no_copy_f.get("status"))
    no_copy_forensic_reasons = str(no_copy_f.get("reasons", no_copy_f.get("root_cause_class", "")))
    forensic_no_copy_blocker = bool(no_copy_forensic_status == "blocker")

    dcr_status = _as_status_154(dcr.get("status"))
    dcr_reasons = str(dcr.get("reasons", ""))

    dcr_forensic_status = _as_status_154(dcr_f.get("status"))
    dcr_forensic_reasons = str(dcr_f.get("reasons", dcr_f.get("root_cause_class", "")))
    forensic_dcr_blocker = bool(dcr_forensic_status == "blocker")

    mia_status = _as_status_154(mia.get("final_mia_release_status", mia.get("status")))
    mia_reasons = str(mia.get("final_mia_release_reasons", mia.get("reasons", "")))

    synth_status = _as_status_154(mia.get("synthetic_distinguishability_status"))
    synth_reasons = str(mia.get("synthetic_distinguishability_reasons", ""))

    temporal_status = _as_status_154(mia.get("temporal_drift_status"))
    temporal_reasons = str(mia.get("temporal_drift_reasons", ""))

    carried_status = _as_status_154(mia.get("carried_privacy_status"))
    carried_reasons = str(mia.get("carried_privacy_reasons", ""))

    release_status, release_reasons = _combine_release_status_154(
        no_copy_status=no_copy_status,
        dcr_status=dcr_status,
        mia_final_status=mia_status,
        synthetic_dist_status=synth_status,
        temporal_status=temporal_status,
        forensic_no_copy_blocker=forensic_no_copy_blocker,
        forensic_dcr_blocker=forensic_dcr_blocker,
    )

    role_rows.append({
        "role_group": role,
        "initial_release_status": release_status,
        "initial_release_reasons": release_reasons,
        "release_status": release_status,
        "release_reasons": release_reasons,
        "release_policy_override_applied": False,
        "release_policy_override_reason": "",

        "no_copy_status": no_copy_status,
        "no_copy_reasons": no_copy_reasons,
        "no_copy_exact_nonzero_match_rate": _safe_float_154(no_copy.get("exact_nonzero_match_rate"), np.nan),
        "no_copy_near_copy_rate_used_for_gate": _safe_float_154(no_copy.get("near_copy_rate_used_for_gate", no_copy.get("near_copy_nonzero_rate")), np.nan),
        "no_copy_p99_similarity_used_for_gate": _safe_float_154(no_copy.get("p99_similarity_used_for_gate", no_copy.get("p99_nonzero_similarity")), np.nan),

        "no_copy_forensic_status": no_copy_forensic_status,
        "no_copy_forensic_reasons": no_copy_forensic_reasons,
        "no_copy_forensic_root_cause": str(no_copy_f.get("root_cause_class", "")),

        "dcr_nndr_status": dcr_status,
        "dcr_nndr_reasons": dcr_reasons,
        "dcr_representation": str(dcr.get("representation", "")),
        "syn_train_dcr_p01_ratio_active": _safe_float_154(dcr.get("syn_train_dcr_p01_ratio_active", dcr.get("syn_train_dcr_p01_ratio")), np.nan),
        "close_neighbor_rate_active": _safe_float_154(dcr.get("close_neighbor_rate_active", dcr.get("close_neighbor_rate")), np.nan),
        "syn_to_train_nndr_active_p01": _safe_float_154(dcr.get("syn_to_train_nndr_active_p01", dcr.get("syn_to_train_nndr_p01")), np.nan),
        "calibration_degenerate_active": bool(dcr.get("calibration_degenerate_active", False)),

        "dcr_forensic_status": dcr_forensic_status,
        "dcr_forensic_reasons": dcr_forensic_reasons,
        "dcr_forensic_root_cause": str(dcr_f.get("root_cause_class", "")),

        "mia_release_status": mia_status,
        "mia_release_reasons": mia_reasons,
        "synthetic_distinguishability_status": synth_status,
        "synthetic_distinguishability_reasons": synth_reasons,
        "synthetic_distinguishability_max_auc": _safe_float_154(mia.get("synthetic_distinguishability_max_auc", mia.get("max_mia_auc")), np.nan),
        "synthetic_distinguishability_max_balanced_accuracy": _safe_float_154(
            mia.get("synthetic_distinguishability_max_balanced_accuracy", mia.get("max_mia_balanced_accuracy")),
            np.nan,
        ),
        "temporal_drift_status": temporal_status,
        "temporal_drift_reasons": temporal_reasons,
        "temporal_drift_max_auc": _safe_float_154(mia.get("temporal_drift_max_auc"), np.nan),
        "temporal_drift_max_balanced_accuracy": _safe_float_154(mia.get("temporal_drift_max_balanced_accuracy"), np.nan),
        "carried_privacy_status": carried_status,
        "carried_privacy_reasons": carried_reasons,

        "role_cols_n": int(len(role_groups.get(role, []))),
        "TEST_real_values_used_for_privacy_reference_only": True,
        "synthetic_values_mutated": False,
    })

    # Findings
    finding_rows.extend(
        _finding_rows_for_role_154(
            role,
            "15.1_no_copy_window",
            no_copy_status,
            no_copy_reasons,
            extra={
                "metric_1": "exact_nonzero_match_rate",
                "value_1": _safe_float_154(no_copy.get("exact_nonzero_match_rate"), np.nan),
                "metric_2": "near_copy_rate_used_for_gate",
                "value_2": _safe_float_154(no_copy.get("near_copy_rate_used_for_gate", no_copy.get("near_copy_nonzero_rate")), np.nan),
            },
        )
    )

    finding_rows.extend(
        _finding_rows_for_role_154(
            role,
            "15.1a_no_copy_forensic",
            no_copy_forensic_status,
            no_copy_forensic_reasons,
            extra={
                "metric_1": "root_cause_class",
                "value_1": str(no_copy_f.get("root_cause_class", "")),
                "metric_2": "suggested_next_action",
                "value_2": str(no_copy_f.get("suggested_next_action", "")),
            },
        )
    )

    finding_rows.extend(
        _finding_rows_for_role_154(
            role,
            "15.2_active_DCR_NNDR",
            dcr_status,
            dcr_reasons,
            extra={
                "metric_1": "syn_train_dcr_p01_ratio_active",
                "value_1": _safe_float_154(dcr.get("syn_train_dcr_p01_ratio_active", dcr.get("syn_train_dcr_p01_ratio")), np.nan),
                "metric_2": "syn_to_train_nndr_active_p01",
                "value_2": _safe_float_154(dcr.get("syn_to_train_nndr_active_p01", dcr.get("syn_to_train_nndr_p01")), np.nan),
            },
        )
    )

    finding_rows.extend(
        _finding_rows_for_role_154(
            role,
            "15.2a_DCR_NNDR_forensic",
            dcr_forensic_status,
            dcr_forensic_reasons,
            extra={
                "metric_1": "root_cause_class",
                "value_1": str(dcr_f.get("root_cause_class", "")),
                "metric_2": "suggested_next_action",
                "value_2": str(dcr_f.get("suggested_next_action", "")),
            },
        )
    )

    finding_rows.extend(
        _finding_rows_for_role_154(
            role,
            "15.3_synthetic_distinguishability",
            synth_status,
            synth_reasons,
            extra={
                "metric_1": "synthetic_distinguishability_max_auc",
                "value_1": _safe_float_154(mia.get("synthetic_distinguishability_max_auc", mia.get("max_mia_auc")), np.nan),
                "metric_2": "synthetic_distinguishability_max_balanced_accuracy",
                "value_2": _safe_float_154(
                    mia.get("synthetic_distinguishability_max_balanced_accuracy", mia.get("max_mia_balanced_accuracy")),
                    np.nan,
                ),
            },
        )
    )

    finding_rows.extend(
        _finding_rows_for_role_154(
            role,
            "15.3_temporal_drift_calibration",
            temporal_status,
            temporal_reasons,
            severity_override="warning" if temporal_status == "warning" else temporal_status,
            extra={
                "metric_1": "temporal_drift_max_auc",
                "value_1": _safe_float_154(mia.get("temporal_drift_max_auc"), np.nan),
                "metric_2": "temporal_drift_max_balanced_accuracy",
                "value_2": _safe_float_154(mia.get("temporal_drift_max_balanced_accuracy"), np.nan),
            },
        )
    )

role_release_df = pd.DataFrame(role_rows)

# Optional mixed composite override, but never if its synthetic distinguishability is blocker.
mixed_composite_source_roles_154 = list(map(
    str,
    CFG.get("cell15_4_mixed_composite_source_roles", [
        "protocol_router", "protocol_zigbee", "iot_event_drivers"
    ]),
))
mixed_composite_override_roles = []

if (
    len(role_release_df)
    and bool(CFG.get("cell15_4_downgrade_mixed_composite_degenerate_dcr", True))
    and "role_group" in role_release_df.columns
):
    initial_status_by_role = dict(zip(
        role_release_df["role_group"].astype(str),
        role_release_df["initial_release_status"].astype(str),
    ))
    source_roles_not_blocked = all(
        initial_status_by_role.get(str(src), "release_blocker") != "release_blocker"
        for src in mixed_composite_source_roles_154
    )

    mixed_mask = role_release_df["role_group"].astype(str).eq("full_cps_mixed_sampled")
    if bool(mixed_mask.any()):
        mixed_idx = role_release_df.index[mixed_mask][0]
        mixed_row = role_release_df.loc[mixed_idx]
        should_override_mixed = bool(
            source_roles_not_blocked
            and str(mixed_row.get("initial_release_status", "")) == "release_blocker"
            and str(mixed_row.get("no_copy_status", "")) == "pass"
            and str(mixed_row.get("synthetic_distinguishability_status", "")) != "blocker"
            and bool(mixed_row.get("calibration_degenerate_active", False))
        )
        if should_override_mixed:
            override_reason = "mixed_composite_degenerate_DCR_NNDR_warning_source_roles_not_blocked"
            role_release_df.loc[mixed_idx, "release_status"] = "release_warning"
            role_release_df.loc[mixed_idx, "release_reasons"] = override_reason
            role_release_df.loc[mixed_idx, "release_policy_override_applied"] = True
            role_release_df.loc[mixed_idx, "release_policy_override_reason"] = override_reason
            mixed_composite_override_roles.append("full_cps_mixed_sampled")
            finding_rows.append({
                "role_group": "full_cps_mixed_sampled",
                "audit": "15.4_release_policy_override",
                "severity": "warning",
                "status": "release_warning",
                "finding": "mixed_composite_degenerate_DCR_NNDR_downgraded",
                "source_roles": "|".join(mixed_composite_source_roles_154),
                "override_reason": override_reason,
            })

privacy_findings_df = pd.DataFrame(finding_rows)

if missing_audit_rows and bool(CFG.get("cell15_4_fail_on_missing_role_from_audit", False)):
    raise RuntimeError(f"[Cell15.4] Missing upstream audit rows: {missing_audit_rows[:10]}")

# ----------------------------------------------------------
# 7) Final Q6 summary and recommendation
# ----------------------------------------------------------
release_counts = (
    role_release_df["release_status"].astype(str).value_counts().sort_index().to_dict()
    if len(role_release_df) else {}
)

release_blocker_n = int((role_release_df["release_status"].astype(str) == "release_blocker").sum())
release_warning_n = int((role_release_df["release_status"].astype(str) == "release_warning").sum())
release_pass_n = int((role_release_df["release_status"].astype(str) == "release_pass").sum())

blocker_roles = role_release_df.loc[
    role_release_df["release_status"].astype(str).eq("release_blocker"), "role_group"
].astype(str).tolist()
warning_roles = role_release_df.loc[
    role_release_df["release_status"].astype(str).eq("release_warning"), "role_group"
].astype(str).tolist()
pass_roles = role_release_df.loc[
    role_release_df["release_status"].astype(str).eq("release_pass"), "role_group"
].astype(str).tolist()

mixed_composite_overrides_applied_n = int(len(mixed_composite_override_roles))

if release_blocker_n > 0:
    q6_overall_status = "release_blocker"
    final_recommendation = str(CFG.get(
        "cell15_4_final_recommendation_if_blockers",
        "not_release_ready_without_privacy_mitigation_or_scope_suppression",
    ))
elif release_warning_n > 0:
    q6_overall_status = "release_warning"
    final_recommendation = str(CFG.get(
        "cell15_4_final_recommendation_if_warnings_only",
        "release_with_privacy_caveats_and_documented_warnings",
    ))
else:
    q6_overall_status = "release_pass"
    final_recommendation = str(CFG.get(
        "cell15_4_final_recommendation_if_pass",
        "release_ready_under_current_q6_audits",
    ))

release_ready_under_current_q6 = bool(q6_overall_status == "release_pass")
release_ready_with_caveats = bool(q6_overall_status == "release_warning")
release_blocked_by_q6 = bool(q6_overall_status == "release_blocker")

finding_counts = (
    privacy_findings_df.groupby(["audit", "severity"], dropna=False)
    .size()
    .reset_index(name="n")
    .sort_values(["severity", "audit"])
    if len(privacy_findings_df)
    else pd.DataFrame(columns=["audit", "severity", "n"])
)

release_summary_rows = [
    {"metric": "q4_final_status", "value": "blocked_no_promotion"},
    {"metric": "q4_coupled_artifacts_used", "value": False},
    {"metric": "q6_overall_status", "value": q6_overall_status},
    {"metric": "final_recommendation", "value": final_recommendation},
    {"metric": "role_groups_total", "value": int(len(role_release_df))},
    {"metric": "release_pass_n", "value": int(release_pass_n)},
    {"metric": "release_warning_n", "value": int(release_warning_n)},
    {"metric": "release_blocker_n", "value": int(release_blocker_n)},
    {"metric": "release_ready_under_current_q6", "value": bool(release_ready_under_current_q6)},
    {"metric": "release_ready_with_caveats", "value": bool(release_ready_with_caveats)},
    {"metric": "release_blocked_by_q6", "value": bool(release_blocked_by_q6)},
    {"metric": "blocker_roles", "value": "|".join(blocker_roles)},
    {"metric": "warning_roles", "value": "|".join(warning_roles)},
    {"metric": "pass_roles", "value": "|".join(pass_roles)},
    {"metric": "mixed_composite_overrides_applied_n", "value": int(mixed_composite_overrides_applied_n)},
    {"metric": "mixed_composite_override_roles", "value": "|".join(mixed_composite_override_roles)},
    {"metric": "formal_privacy_proof_claimed", "value": False},
    {"metric": "TEST_real_values_used_for_privacy_reference_only", "value": True},
    {"metric": "synthetic_values_mutated", "value": False},
]

release_summary_df = pd.DataFrame(release_summary_rows)

# ----------------------------------------------------------
# 8) Write publication/release statement
# ----------------------------------------------------------
statement_lines = [
    "Q6 Privacy / No-Copy Release-Safety Summary",
    "===========================================",
    "",
    "Q4 governance carried into Q6:",
    "- final_q4_status: blocked_no_promotion",
    "- q4_coupled_artifacts_used: False",
    "",
    f"Overall Q6 status: {q6_overall_status}",
    f"Final recommendation: {final_recommendation}",
    "",
    "Interpretation:",
    "These audits provide release-risk evidence, not a formal privacy proof.",
    "The checks are role-specific across protocol, IoT continuous values, binary states, event drivers, observability masks, and mixed CPS samples.",
    "",
    "Role-level release counts:",
    f"- release_pass: {release_pass_n}",
    f"- release_warning: {release_warning_n}",
    f"- release_blocker: {release_blocker_n}",
    "",
]

if blocker_roles:
    statement_lines.extend([
        "Release-blocker role groups:",
        *[f"- {r}" for r in blocker_roles],
        "",
    ])

if warning_roles:
    statement_lines.extend([
        "Warning-level role groups:",
        *[f"- {r}" for r in warning_roles],
        "",
    ])

if pass_roles:
    statement_lines.extend([
        "Pass-level role groups:",
        *[f"- {r}" for r in pass_roles],
        "",
    ])

statement_lines.extend([
    "Main conclusions:",
    "- Q6 is not release-ready under the current strict role-level policy because release blockers remain.",
    "- The carried forensic blockers from no-copy/DCR are preserved; MIA evidence does not clear them.",
    "- OTA and binary-state risks require mitigation, suppression, aggregation, or explicit non-release scoping before a strict public release.",
    "- Synthetic distinguishability is high for several role groups and is treated as release risk under the current policy.",
    "- Temporal drift is reported as calibration/context unless separately configured as a release blocker.",
    "",
    "Recommended next actions if strict public release is required:",
    "1. Add a mitigation/suppression/redaction cell for high-risk OTA active windows.",
    "2. Add binary-state suppression/aggregation or regenerate binary states with stronger anti-copy constraints.",
    "3. Consider excluding high-risk role groups from public release while keeping them in internal evaluation.",
    "4. Report synthetic-vs-real distinguishability honestly as a limitation if releasing without mitigation.",
    "5. Do not claim formal privacy; claim role-specific privacy/no-copy release-risk auditing.",
    "",
    "No synthetic values were mutated in Cell 15.4.",
])

release_statement = "\n".join(statement_lines)

with open(release_statement_txt, "w", encoding="utf-8") as f:
    f.write(release_statement)

# ----------------------------------------------------------
# 9) Save CSV outputs
# ----------------------------------------------------------
role_release_df.to_csv(role_release_csv, index=False)
privacy_findings_df.to_csv(privacy_findings_csv, index=False)
release_summary_df.to_csv(release_summary_csv, index=False)

# ----------------------------------------------------------
# 10) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "15.4",
    "version": CELL154_VERSION,
    "role": "q6_privacy_summary_and_release_safety_manifest_forensic_aware",
    "quality_dimension": "Q6_privacy_no_copy_release_safety",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
    },
    "upstream_contract_versions": {
        "cell15_0": version150_154,
        "cell15_1": version151_154,
        "cell15_1a": version151a_154,
        "cell15_2": version152_154,
        "cell15_2a": version152a_154,
        "cell15_3": version153_154,
    },
    "overall": {
        "q6_overall_status": q6_overall_status,
        "final_recommendation": final_recommendation,
        "release_ready_under_current_q6": release_ready_under_current_q6,
        "release_ready_with_caveats": release_ready_with_caveats,
        "release_blocked_by_q6": release_blocked_by_q6,
        "formal_privacy_proof_claimed": False,
    },
    "role_counts": {
        "role_groups_total": int(len(role_release_df)),
        "release_pass_n": int(release_pass_n),
        "release_warning_n": int(release_warning_n),
        "release_blocker_n": int(release_blocker_n),
        "release_status_counts": release_counts,
        "blocker_roles": blocker_roles,
        "warning_roles": warning_roles,
        "pass_roles": pass_roles,
    },
    "mixed_composite_policy": {
        "enabled": bool(CFG.get("cell15_4_downgrade_mixed_composite_degenerate_dcr", True)),
        "source_roles": mixed_composite_source_roles_154,
        "override_applied_roles": mixed_composite_override_roles,
    },
    "audit_inputs": {
        "cell15_1_no_copy_summary": CELL15_1_NO_COPY_CONTRACT.get("summary", {}) if isinstance(CELL15_1_NO_COPY_CONTRACT, dict) else {},
        "cell15_1a_no_copy_forensic_summary": CELL15_1A_NO_COPY_FORENSIC_CONTRACT.get("summary", {}) if isinstance(CELL15_1A_NO_COPY_FORENSIC_CONTRACT, dict) else {},
        "cell15_2_dcr_nndr_summary": CELL15_2_DCR_NNDR_CONTRACT.get("summary", {}) if isinstance(CELL15_2_DCR_NNDR_CONTRACT, dict) else {},
        "cell15_2a_dcr_nndr_forensic_summary": CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT.get("summary", {}) if isinstance(CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT, dict) else {},
        "cell15_3_mia_summary": CELL15_3_MIA_CONTRACT.get("summary", {}) if isinstance(CELL15_3_MIA_CONTRACT, dict) else {},
    },
    "role_release_safety": role_release_df.to_dict("records"),
    "finding_counts": finding_counts.to_dict("records"),
    "methodological_position": {
        "privacy_claim": "release-risk evidence, not formal privacy proof",
        "temporal_drift_policy": "temporal TRAIN-vs-TEST separability is calibration/context unless explicitly configured as a release blocker",
        "synthetic_distinguishability_policy": "synthetic-vs-real distinguishability is treated as release risk under current policy",
        "no_copy_policy": "non-idle exact or near-copy windows are release blockers or warnings depending on rate",
        "DCR_NNDR_policy": "active-window DCR/NNDR is used for sparse protocol and binary roles",
        "MIA_policy": "MIA adds evidence but does not clear prior no-copy or DCR/NNDR blockers",
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_for_privacy_reference_only": True,
        "TEST_real_values_used_for_model_fitting": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "release_decision_done_here": True,
    },
    "outputs": {
        "role_release_csv": role_release_csv,
        "privacy_findings_csv": privacy_findings_csv,
        "release_summary_csv": release_summary_csv,
        "release_statement_txt": release_statement_txt,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_154(contract_json, contract)
_write_json_154(contract_canonical_json, contract)

manifest = {
    "cell": "15.4",
    "version": CELL154_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "overall": contract["overall"],
    "role_counts": contract["role_counts"],
    "mixed_composite_policy": contract["mixed_composite_policy"],
    "strict_contract": contract["strict_contract"],
    "blocker_roles": blocker_roles,
    "warning_roles": warning_roles,
}

_write_json_154(manifest_json, manifest)

hashes = {
    "role_release_csv_sha256": _sha256_file_154(role_release_csv),
    "privacy_findings_csv_sha256": _sha256_file_154(privacy_findings_csv),
    "release_summary_csv_sha256": _sha256_file_154(release_summary_csv),
    "release_statement_txt_sha256": _sha256_file_154(release_statement_txt),
    "contract_json_sha256": _sha256_file_154(contract_json),
    "contract_canonical_json_sha256": _sha256_file_154(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_154(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_154(contract_json, contract)
_write_json_154(contract_canonical_json, contract)
_write_json_154(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL154_VERSION"] = CELL154_VERSION
globals()["CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF"] = role_release_df
globals()["CELL15_4_Q6_PRIVACY_FINDINGS_DF"] = privacy_findings_df
globals()["CELL15_4_Q6_RELEASE_SUMMARY_DF"] = release_summary_df
globals()["CELL15_4_Q6_RELEASE_STATEMENT"] = release_statement
globals()["CELL15_4_Q6_RELEASE_SAFETY_CONTRACT"] = contract

globals()["CELL15_4_Q6_ROLE_RELEASE_SAFETY_CSV"] = role_release_csv
globals()["CELL15_4_Q6_PRIVACY_FINDINGS_CSV"] = privacy_findings_csv
globals()["CELL15_4_Q6_RELEASE_SUMMARY_CSV"] = release_summary_csv
globals()["CELL15_4_Q6_RELEASE_STATEMENT_TXT"] = release_statement_txt
globals()["CELL15_4_Q6_RELEASE_SAFETY_CONTRACT_JSON"] = contract_json
globals()["CELL15_4_Q6_RELEASE_SAFETY_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL15_4_Q6_RELEASE_SAFETY_MANIFEST_JSON"] = manifest_json

log(
    "[Cell15.4] Q6 privacy release-safety summary complete | "
    f"overall_status={q6_overall_status} | "
    f"recommendation={final_recommendation} | "
    f"pass={release_pass_n} | warning={release_warning_n} | blocker={release_blocker_n}"
)
log(f"[Cell15.4] Blocker roles | {blocker_roles}")
log(f"[Cell15.4] Warning roles | {warning_roles}")
log(f"[Cell15.4] Pass roles | {pass_roles}")
log(
    "[Cell15.4] Mixed composite overrides | "
    f"mixed_composite_overrides_applied_n={mixed_composite_overrides_applied_n} | "
    f"roles={mixed_composite_override_roles}"
)
log(f"[Cell15.4] Saved role release safety: {role_release_csv} | rows={len(role_release_df)}")
log(f"[Cell15.4] Saved privacy findings: {privacy_findings_csv} | rows={len(privacy_findings_df)}")
log(f"[Cell15.4] Saved release statement: {release_statement_txt}")
log(f"[Cell15.4] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell15.4] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_for_privacy_reference_only=True | "
    "TEST_real_values_used_for_model_fitting=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "release_decision_done_here=True"
)
log("--- END: Cell 15.4 - Q6 privacy summary and release-safety manifest (v1.2 forensic-aware strict) ---")

gc.collect()