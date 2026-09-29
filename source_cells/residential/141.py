# ==========================================================
# CELL 15.4b - Q6 public-candidate policy-split summary
# v1.0 STUDY-THESIS strict direct-privacy vs distinguishability split
#
# Role:
#   - Summarize the public-candidate Q6 re-audit with two separated claims:
#
#       A) Direct privacy / no-copy release-risk status
#          Evidence sources:
#            15.1  no-copy window audit
#            15.1a no-copy forensic diagnostic
#            15.2  DCR / NNDR audit
#            15.2a DCR / NNDR forensic diagnostic
#
#       B) Synthetic-real distinguishability / realism-risk status
#          Evidence sources:
#            15.3 synthetic-vs-real distinguishability MIA protocol
#            15.3 temporal real TRAIN-vs-TEST drift calibration
#
#   - Preserve the strict combined Cell 15.4 result:
#       release_blocker remains release_blocker under the current strict
#       policy because synthetic distinguishability is treated as release risk.
#
# Scientific contract:
#   - No fitting.
#   - No materialization.
#   - No synthetic mutation.
#   - No release artifact mutation.
#   - TEST real values are not newly read here; this cell only summarizes
#     previous Q6 public-candidate audit outputs.
#   - This cell does not claim formal privacy proof.
#
# Outputs:
#   reports/cell15_4b_q6_policy_split_role_summary.csv
#   reports/cell15_4b_q6_policy_split_findings.csv
#   reports/cell15_4b_q6_policy_split_summary.csv
#   reports/cell15_4b_q6_policy_split_statement.txt
#   reports/cell15_4b_q6_policy_split_contract.json
#   artifacts/contracts/cell15_4b_q6_policy_split_contract_v1_0_THESIS.json
#   artifacts/cell15_4b_q6_policy_split_manifest.json
# ==========================================================

log("--- START: Cell 15.4b - Q6 public-candidate policy-split summary (v1.0 strict) ---")

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
_required_154b = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR", "SEED",
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
    "CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF",
    "CELL15_4_Q6_RELEASE_SAFETY_CONTRACT",
]
_missing_154b = [k for k in _required_154b if k not in globals()]
if _missing_154b:
    raise RuntimeError(f"[Cell15.4b] Missing required globals: {_missing_154b}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
CELL154B_VERSION = "cell15_4b_q6_public_policy_split_summary_v1_0"

CFG["cell15_4b_version"] = CELL154B_VERSION
CFG["cell15_4b_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_4b_Q4_coupled_artifacts_used"] = False
CFG["cell15_4b_TEST_real_values_used_here"] = False
CFG["cell15_4b_synthetic_values_mutated"] = False
CFG["cell15_4b_selection_done_here"] = False
CFG["cell15_4b_generator_fit_done_here"] = False
CFG["cell15_4b_materialization_done_here"] = False
CFG["cell15_4b_policy_split_summary_done_here"] = True
CFG["cell15_4b_formal_privacy_proof_claimed"] = False

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell15_4b_temporal_drift_is_context_not_direct_privacy", True)
CFG.setdefault("cell15_4b_synthetic_distinguishability_is_realism_risk", True)
CFG.setdefault("cell15_4b_strict_combined_policy_inherits_cell15_4", True)
CFG.setdefault("cell15_4b_direct_privacy_warning_if_DCR_calibration_degenerate", True)

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
role_summary_csv = os.path.join(REPORT_DIR, "cell15_4b_q6_policy_split_role_summary.csv")
findings_csv = os.path.join(REPORT_DIR, "cell15_4b_q6_policy_split_findings.csv")
summary_csv = os.path.join(REPORT_DIR, "cell15_4b_q6_policy_split_summary.csv")
statement_txt = os.path.join(REPORT_DIR, "cell15_4b_q6_policy_split_statement.txt")
contract_json = os.path.join(REPORT_DIR, "cell15_4b_q6_policy_split_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell15_4b_q6_policy_split_contract_v1_0_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell15_4b_q6_policy_split_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_154b(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_154b(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_154b(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_154b(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_154b(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_154b(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_154b(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_154b(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_154b(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_154b(payload), f, indent=2, sort_keys=True)

def _sha256_file_154b(path):
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_154b(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.4b] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.4b] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

def _safe_float_154b(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _status_or_default_154b(x, default="not_evaluable"):
    s = str(x if x is not None else "").strip()
    return s if s else default

def _get_role_row_154b(df, role):
    if not isinstance(df, pd.DataFrame) or "role_group" not in df.columns:
        return {}
    sub = df[df["role_group"].astype(str).eq(str(role))]
    if len(sub) == 0:
        return {}
    return sub.iloc[0].to_dict()

def _combine_direct_privacy_status_154b(no_copy_status, no_copy_forensic_status, dcr_status, dcr_forensic_status):
    blockers = []
    warnings = []

    if str(no_copy_status) == "blocker":
        blockers.append("no_copy_window_blocker")
    elif str(no_copy_status) == "warning":
        warnings.append("no_copy_window_warning")

    if str(no_copy_forensic_status) == "blocker":
        blockers.append("no_copy_forensic_blocker")
    elif str(no_copy_forensic_status) == "warning":
        warnings.append("no_copy_forensic_warning")

    if str(dcr_status) == "blocker":
        blockers.append("DCR_NNDR_blocker")
    elif str(dcr_status) == "warning":
        warnings.append("DCR_NNDR_warning")

    if str(dcr_forensic_status) == "blocker":
        blockers.append("DCR_NNDR_forensic_blocker")
    elif str(dcr_forensic_status) == "warning":
        warnings.append("DCR_NNDR_forensic_warning")

    if blockers:
        return "direct_privacy_blocker", "|".join(sorted(set(blockers)))
    if warnings:
        return "direct_privacy_warning", "|".join(sorted(set(warnings)))
    return "direct_privacy_pass", "no_copy_and_DCR_NNDR_pass"

def _combine_distinguishability_status_154b(synth_status, temporal_status):
    blockers = []
    warnings = []

    if str(synth_status) == "blocker":
        blockers.append("synthetic_real_distinguishability_blocker")
    elif str(synth_status) == "warning":
        warnings.append("synthetic_real_distinguishability_warning")

    if str(temporal_status) == "warning":
        warnings.append("temporal_drift_context_warning")
    elif str(temporal_status) == "blocker":
        # This should not normally happen because 15.3 treats temporal drift as context.
        warnings.append("temporal_drift_context_blocker_reported_as_context")

    if blockers:
        return "distinguishability_blocker", "|".join(sorted(set(blockers + warnings)))
    if warnings:
        return "distinguishability_warning", "|".join(sorted(set(warnings)))
    return "distinguishability_pass", "synthetic_distinguishability_and_temporal_drift_within_context_gates"

def _overall_direct_status_154b(statuses):
    statuses = list(map(str, statuses))
    if any(s == "direct_privacy_blocker" for s in statuses):
        return "release_blocker"
    if any(s == "direct_privacy_warning" for s in statuses):
        return "release_warning"
    if statuses:
        return "release_pass"
    return "not_evaluable"

def _overall_dist_status_154b(statuses):
    statuses = list(map(str, statuses))
    if any(s == "distinguishability_blocker" for s in statuses):
        return "release_blocker"
    if any(s == "distinguishability_warning" for s in statuses):
        return "release_warning"
    if statuses:
        return "release_pass"
    return "not_evaluable"

def _role_list_by_status_154b(df, col, status):
    if not isinstance(df, pd.DataFrame) or col not in df.columns:
        return []
    return df.loc[df[col].astype(str).eq(str(status)), "role_group"].astype(str).tolist()

# ----------------------------------------------------------
# 4) Validate upstream contracts
# ----------------------------------------------------------
version150_154b = _require_contract_version_154b(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
version151_154b = _require_contract_version_154b(
    CELL15_1_NO_COPY_CONTRACT,
    "CELL15_1_NO_COPY_CONTRACT",
    "cell15_1_q6_no_copy_window_audit_v1_1",
)
version151a_154b = _require_contract_version_154b(
    CELL15_1A_NO_COPY_FORENSIC_CONTRACT,
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "cell15_1a_q6_no_copy_blocker_forensic_v1_0",
)
version152_154b = _require_contract_version_154b(
    CELL15_2_DCR_NNDR_CONTRACT,
    "CELL15_2_DCR_NNDR_CONTRACT",
    "cell15_2_q6_dcr_nndr_role_specific_privacy_v1_2",
)
version152a_154b = _require_contract_version_154b(
    CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT,
    "CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT",
    "cell15_2a_q6_dcr_nndr_blocker_forensic_v1_0",
)
version153_154b = _require_contract_version_154b(
    CELL15_3_MIA_CONTRACT,
    "CELL15_3_MIA_CONTRACT",
    "cell15_3_q6_mia_random_temporal_privacy_v1_2",
)
version154_154b = _require_contract_version_154b(
    CELL15_4_Q6_RELEASE_SAFETY_CONTRACT,
    "CELL15_4_Q6_RELEASE_SAFETY_CONTRACT",
    "cell15_4_q6_privacy_release_safety_summary_v1_2",
)

strict150 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
if str(strict150.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.4b] Cell 15.0 does not carry Q4_final_status=blocked_no_promotion.")
if bool(strict150.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.4b] Q4-coupled artifacts were used unexpectedly.")

for name, contract in [
    ("Cell15.1", CELL15_1_NO_COPY_CONTRACT),
    ("Cell15.1a", CELL15_1A_NO_COPY_FORENSIC_CONTRACT),
    ("Cell15.2", CELL15_2_DCR_NNDR_CONTRACT),
    ("Cell15.2a", CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT),
    ("Cell15.3", CELL15_3_MIA_CONTRACT),
    ("Cell15.4", CELL15_4_Q6_RELEASE_SAFETY_CONTRACT),
]:
    strict = contract.get("strict_contract", {})
    if bool(strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell15.4b] {name} indicates synthetic mutation.")

# ----------------------------------------------------------
# 5) Load public-candidate Q6 audit tables
# ----------------------------------------------------------
no_copy_df = CELL15_1_NO_COPY_WINDOW_METRICS_DF.copy()
no_copy_forensic_df = CELL15_1A_NO_COPY_BLOCKER_FORENSIC_DF.copy()
dcr_df = CELL15_2_DCR_NNDR_ROLE_METRICS_DF.copy()
dcr_forensic_df = CELL15_2A_DCR_NNDR_BLOCKER_FORENSIC_DF.copy()
mia_df = CELL15_3_MIA_ROLE_METRICS_DF.copy()
strict_release_df = CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF.copy()

for name, df in [
    ("15.1 no_copy_df", no_copy_df),
    ("15.1a no_copy_forensic_df", no_copy_forensic_df),
    ("15.2 dcr_df", dcr_df),
    ("15.2a dcr_forensic_df", dcr_forensic_df),
    ("15.3 mia_df", mia_df),
    ("15.4 strict_release_df", strict_release_df),
]:
    if not isinstance(df, pd.DataFrame) or "role_group" not in df.columns:
        raise RuntimeError(f"[Cell15.4b] Invalid or empty role table: {name}")

roles = sorted(set(strict_release_df["role_group"].astype(str).tolist()))

log(
    "[Cell15.4b] Building policy-split Q6 summary | "
    f"roles={roles}"
)

# ----------------------------------------------------------
# 6) Role-level policy split
# ----------------------------------------------------------
role_rows = []
finding_rows = []

for role in roles:
    no_copy = _get_role_row_154b(no_copy_df, role)
    no_copy_f = _get_role_row_154b(no_copy_forensic_df, role)
    dcr = _get_role_row_154b(dcr_df, role)
    dcr_f = _get_role_row_154b(dcr_forensic_df, role)
    mia = _get_role_row_154b(mia_df, role)
    strict = _get_role_row_154b(strict_release_df, role)

    no_copy_status = _status_or_default_154b(no_copy.get("status"))
    no_copy_reasons = str(no_copy.get("reasons", ""))

    no_copy_f_status = _status_or_default_154b(no_copy_f.get("status"))
    no_copy_f_reasons = str(no_copy_f.get("reasons", no_copy_f.get("root_cause_class", "")))

    dcr_status = _status_or_default_154b(dcr.get("status"))
    dcr_reasons = str(dcr.get("reasons", ""))

    dcr_f_status = _status_or_default_154b(dcr_f.get("status"))
    dcr_f_reasons = str(dcr_f.get("reasons", dcr_f.get("root_cause_class", "")))

    synth_status = _status_or_default_154b(mia.get("synthetic_distinguishability_status"))
    synth_reasons = str(mia.get("synthetic_distinguishability_reasons", ""))

    temporal_status = _status_or_default_154b(mia.get("temporal_drift_status"))
    temporal_reasons = str(mia.get("temporal_drift_reasons", ""))

    strict_status = str(strict.get("release_status", "not_evaluable"))
    strict_reasons = str(strict.get("release_reasons", ""))

    direct_status, direct_reasons = _combine_direct_privacy_status_154b(
        no_copy_status,
        no_copy_f_status,
        dcr_status,
        dcr_f_status,
    )
    dist_status, dist_reasons = _combine_distinguishability_status_154b(
        synth_status,
        temporal_status,
    )

    role_rows.append({
        "role_group": role,

        "direct_privacy_status": direct_status,
        "direct_privacy_reasons": direct_reasons,
        "direct_privacy_interpretation": (
            "direct copy/nearest-neighbor privacy evidence"
        ),

        "no_copy_status": no_copy_status,
        "no_copy_reasons": no_copy_reasons,
        "no_copy_exact_nonzero_match_rate": _safe_float_154b(no_copy.get("exact_nonzero_match_rate"), np.nan),
        "no_copy_near_copy_rate_used_for_gate": _safe_float_154b(
            no_copy.get("near_copy_rate_used_for_gate", no_copy.get("near_copy_nonzero_rate")),
            np.nan,
        ),

        "no_copy_forensic_status": no_copy_f_status,
        "no_copy_forensic_reasons": no_copy_f_reasons,

        "dcr_nndr_status": dcr_status,
        "dcr_nndr_reasons": dcr_reasons,
        "syn_train_dcr_p01_ratio_active": _safe_float_154b(
            dcr.get("syn_train_dcr_p01_ratio_active", dcr.get("syn_train_dcr_p01_ratio")),
            np.nan,
        ),
        "close_neighbor_rate_active": _safe_float_154b(dcr.get("close_neighbor_rate_active", dcr.get("close_neighbor_rate")), np.nan),
        "syn_to_train_nndr_active_p01": _safe_float_154b(dcr.get("syn_to_train_nndr_active_p01", dcr.get("syn_to_train_nndr_p01")), np.nan),
        "calibration_degenerate_active": bool(dcr.get("calibration_degenerate_active", False)),

        "dcr_forensic_status": dcr_f_status,
        "dcr_forensic_reasons": dcr_f_reasons,

        "distinguishability_status": dist_status,
        "distinguishability_reasons": dist_reasons,
        "distinguishability_interpretation": (
            "synthetic-real separability / realism and disclosure caveat, not direct proof of copying"
        ),

        "synthetic_distinguishability_status": synth_status,
        "synthetic_distinguishability_reasons": synth_reasons,
        "synthetic_distinguishability_max_auc": _safe_float_154b(mia.get("synthetic_distinguishability_max_auc", mia.get("max_mia_auc")), np.nan),
        "synthetic_distinguishability_max_balanced_accuracy": _safe_float_154b(
            mia.get("synthetic_distinguishability_max_balanced_accuracy", mia.get("max_mia_balanced_accuracy")),
            np.nan,
        ),

        "temporal_drift_status": temporal_status,
        "temporal_drift_reasons": temporal_reasons,
        "temporal_drift_max_auc": _safe_float_154b(mia.get("temporal_drift_max_auc"), np.nan),
        "temporal_drift_max_balanced_accuracy": _safe_float_154b(mia.get("temporal_drift_max_balanced_accuracy"), np.nan),

        "strict_combined_release_status": strict_status,
        "strict_combined_release_reasons": strict_reasons,
        "strict_combined_interpretation": (
            "current Cell 15.4 policy: synthetic distinguishability is treated as release risk"
        ),
        "formal_privacy_proof_claimed": False,
        "synthetic_values_mutated": False,
    })

    # Findings table
    if direct_status != "direct_privacy_pass":
        finding_rows.append({
            "role_group": role,
            "evidence_family": "direct_privacy_no_copy_DCR_NNDR",
            "severity": "blocker" if direct_status == "direct_privacy_blocker" else "warning",
            "status": direct_status,
            "finding": direct_reasons,
            "interpretation": "direct release-risk evidence",
        })
    else:
        finding_rows.append({
            "role_group": role,
            "evidence_family": "direct_privacy_no_copy_DCR_NNDR",
            "severity": "pass",
            "status": direct_status,
            "finding": direct_reasons,
            "interpretation": "no direct privacy/no-copy blocker",
        })

    if dist_status != "distinguishability_pass":
        finding_rows.append({
            "role_group": role,
            "evidence_family": "synthetic_real_distinguishability",
            "severity": "blocker" if dist_status == "distinguishability_blocker" else "warning",
            "status": dist_status,
            "finding": dist_reasons,
            "interpretation": "realism/disclosure caveat; not direct proof of copying",
        })
    else:
        finding_rows.append({
            "role_group": role,
            "evidence_family": "synthetic_real_distinguishability",
            "severity": "pass",
            "status": dist_status,
            "finding": dist_reasons,
            "interpretation": "within synthetic-real distinguishability gates",
        })

role_summary_df = pd.DataFrame(role_rows)
findings_df = pd.DataFrame(finding_rows)

# ----------------------------------------------------------
# 7) Overall policy-split status
# ----------------------------------------------------------
direct_overall_status = _overall_direct_status_154b(role_summary_df["direct_privacy_status"].tolist())
dist_overall_status = _overall_dist_status_154b(role_summary_df["distinguishability_status"].tolist())

strict_overall_status = str(CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("overall", {}).get("q6_overall_status", "unknown"))
strict_final_recommendation = str(CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("overall", {}).get("final_recommendation", "unknown"))

direct_blocker_roles = _role_list_by_status_154b(role_summary_df, "direct_privacy_status", "direct_privacy_blocker")
direct_warning_roles = _role_list_by_status_154b(role_summary_df, "direct_privacy_status", "direct_privacy_warning")
direct_pass_roles = _role_list_by_status_154b(role_summary_df, "direct_privacy_status", "direct_privacy_pass")

dist_blocker_roles = _role_list_by_status_154b(role_summary_df, "distinguishability_status", "distinguishability_blocker")
dist_warning_roles = _role_list_by_status_154b(role_summary_df, "distinguishability_status", "distinguishability_warning")
dist_pass_roles = _role_list_by_status_154b(role_summary_df, "distinguishability_status", "distinguishability_pass")

strict_blocker_roles = _role_list_by_status_154b(role_summary_df, "strict_combined_release_status", "release_blocker")
strict_warning_roles = _role_list_by_status_154b(role_summary_df, "strict_combined_release_status", "release_warning")
strict_pass_roles = _role_list_by_status_154b(role_summary_df, "strict_combined_release_status", "release_pass")

summary_rows = [
    {"metric": "q4_final_status", "value": "blocked_no_promotion"},
    {"metric": "q4_coupled_artifacts_used", "value": False},
    {"metric": "public_candidate_policy_split", "value": True},

    {"metric": "direct_privacy_overall_status", "value": direct_overall_status},
    {"metric": "direct_privacy_blocker_roles", "value": "|".join(direct_blocker_roles)},
    {"metric": "direct_privacy_warning_roles", "value": "|".join(direct_warning_roles)},
    {"metric": "direct_privacy_pass_roles", "value": "|".join(direct_pass_roles)},
    {"metric": "direct_privacy_claim", "value": "no_copy_and_DCR_NNDR_release_risk_only_not_formal_privacy_proof"},

    {"metric": "distinguishability_overall_status", "value": dist_overall_status},
    {"metric": "distinguishability_blocker_roles", "value": "|".join(dist_blocker_roles)},
    {"metric": "distinguishability_warning_roles", "value": "|".join(dist_warning_roles)},
    {"metric": "distinguishability_pass_roles", "value": "|".join(dist_pass_roles)},
    {"metric": "distinguishability_claim", "value": "synthetic_real_separability_realism_and_disclosure_caveat_not_direct_copy_proof"},

    {"metric": "strict_combined_overall_status", "value": strict_overall_status},
    {"metric": "strict_combined_final_recommendation", "value": strict_final_recommendation},
    {"metric": "strict_combined_blocker_roles", "value": "|".join(strict_blocker_roles)},
    {"metric": "strict_combined_warning_roles", "value": "|".join(strict_warning_roles)},
    {"metric": "strict_combined_pass_roles", "value": "|".join(strict_pass_roles)},
    {"metric": "strict_combined_policy", "value": "inherits_Cell15_4_where_synthetic_distinguishability_is_release_risk"},

    {"metric": "formal_privacy_proof_claimed", "value": False},
    {"metric": "TEST_real_values_used_here", "value": False},
    {"metric": "synthetic_values_mutated", "value": False},
]

summary_df = pd.DataFrame(summary_rows)

# ----------------------------------------------------------
# 8) Statement
# ----------------------------------------------------------
statement_lines = [
    "Q6 Public-Candidate Policy-Split Summary",
    "========================================",
    "",
    "Q4 governance:",
    "- final_q4_status: blocked_no_promotion",
    "- q4_coupled_artifacts_used: False",
    "",
    f"Direct privacy / no-copy status: {direct_overall_status}",
    f"Direct privacy blocker roles: {direct_blocker_roles if direct_blocker_roles else 'none'}",
    f"Direct privacy warning roles: {direct_warning_roles if direct_warning_roles else 'none'}",
    "",
    f"Synthetic-real distinguishability status: {dist_overall_status}",
    f"Distinguishability blocker roles: {dist_blocker_roles if dist_blocker_roles else 'none'}",
    f"Distinguishability warning roles: {dist_warning_roles if dist_warning_roles else 'none'}",
    "",
    f"Strict combined release status inherited from Cell 15.4: {strict_overall_status}",
    f"Strict combined blocker roles: {strict_blocker_roles if strict_blocker_roles else 'none'}",
    f"Strict combined warning roles: {strict_warning_roles if strict_warning_roles else 'none'}",
    "",
    "Interpretation:",
    "The public candidate removes the original no-copy and active DCR/NNDR blockers.",
    "Therefore, under direct privacy/no-copy evidence alone, the public candidate is warning-level rather than blocker-level.",
    "The remaining strict release blockers are caused by synthetic-real distinguishability policy, especially roles that remain classifier-separable from real training data.",
    "This distinguishability evidence should be reported as realism/disclosure release risk, not as direct proof of memorization or copying.",
    "",
    "Recommended paper wording:",
    "For the role-restricted public candidate, direct no-copy and nearest-neighbor privacy checks found no blocker-level risks; remaining warnings were calibration/context warnings. However, under the stricter combined release policy, synthetic-real distinguishability remains a release-risk blocker for selected retained roles. We therefore report both the direct privacy/no-copy status and the stricter combined release-risk status, and do not claim a formal privacy guarantee.",
    "",
    "No synthetic values were mutated in Cell 15.4b.",
]
statement = "\n".join(statement_lines)

with open(statement_txt, "w", encoding="utf-8") as f:
    f.write(statement)

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
role_summary_df.to_csv(role_summary_csv, index=False)
findings_df.to_csv(findings_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

# ----------------------------------------------------------
# 10) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "15.4b",
    "version": CELL154B_VERSION,
    "role": "q6_public_candidate_policy_split_summary",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
    },
    "upstream_contract_versions": {
        "cell15_0_public_rebind": version150_154b,
        "cell15_1": version151_154b,
        "cell15_1a": version151a_154b,
        "cell15_2": version152_154b,
        "cell15_2a": version152a_154b,
        "cell15_3": version153_154b,
        "cell15_4": version154_154b,
    },
    "policy_split": {
        "direct_privacy_overall_status": direct_overall_status,
        "direct_privacy_blocker_roles": direct_blocker_roles,
        "direct_privacy_warning_roles": direct_warning_roles,
        "direct_privacy_pass_roles": direct_pass_roles,
        "distinguishability_overall_status": dist_overall_status,
        "distinguishability_blocker_roles": dist_blocker_roles,
        "distinguishability_warning_roles": dist_warning_roles,
        "distinguishability_pass_roles": dist_pass_roles,
        "strict_combined_overall_status": strict_overall_status,
        "strict_combined_final_recommendation": strict_final_recommendation,
        "strict_combined_blocker_roles": strict_blocker_roles,
        "strict_combined_warning_roles": strict_warning_roles,
        "strict_combined_pass_roles": strict_pass_roles,
    },
    "methodological_position": {
        "direct_privacy_claim": "no-copy and DCR/NNDR release-risk evidence; not formal privacy proof",
        "distinguishability_claim": "synthetic-real separability is realism/disclosure release-risk evidence, not direct copy proof",
        "strict_combined_policy": "inherits Cell 15.4 combined release status where synthetic distinguishability is treated as release risk",
        "temporal_drift_policy": "temporal drift is context/calibration unless separately configured as a direct release blocker",
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_model_fitting": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "release_artifact_mutated_here": False,
        "policy_split_summary_done_here": True,
        "formal_privacy_proof_claimed": False,
    },
    "outputs": {
        "role_summary_csv": role_summary_csv,
        "findings_csv": findings_csv,
        "summary_csv": summary_csv,
        "statement_txt": statement_txt,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_154b(contract_json, contract)
_write_json_154b(contract_canonical_json, contract)

manifest = {
    "cell": "15.4b",
    "version": CELL154B_VERSION,
    "q4_governance": contract["q4_governance"],
    "policy_split": contract["policy_split"],
    "strict_contract": contract["strict_contract"],
    "created_outputs": contract["outputs"],
}
_write_json_154b(manifest_json, manifest)

hashes = {
    "role_summary_csv_sha256": _sha256_file_154b(role_summary_csv),
    "findings_csv_sha256": _sha256_file_154b(findings_csv),
    "summary_csv_sha256": _sha256_file_154b(summary_csv),
    "statement_txt_sha256": _sha256_file_154b(statement_txt),
    "contract_json_sha256": _sha256_file_154b(contract_json),
    "contract_canonical_json_sha256": _sha256_file_154b(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_154b(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_154b(contract_json, contract)
_write_json_154b(contract_canonical_json, contract)
_write_json_154b(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL154B_VERSION"] = CELL154B_VERSION
globals()["CELL15_4B_Q6_POLICY_SPLIT_ROLE_SUMMARY_DF"] = role_summary_df
globals()["CELL15_4B_Q6_POLICY_SPLIT_FINDINGS_DF"] = findings_df
globals()["CELL15_4B_Q6_POLICY_SPLIT_SUMMARY_DF"] = summary_df
globals()["CELL15_4B_Q6_POLICY_SPLIT_STATEMENT"] = statement
globals()["CELL15_4B_Q6_POLICY_SPLIT_CONTRACT"] = contract

globals()["CELL15_4B_Q6_POLICY_SPLIT_ROLE_SUMMARY_CSV"] = role_summary_csv
globals()["CELL15_4B_Q6_POLICY_SPLIT_FINDINGS_CSV"] = findings_csv
globals()["CELL15_4B_Q6_POLICY_SPLIT_SUMMARY_CSV"] = summary_csv
globals()["CELL15_4B_Q6_POLICY_SPLIT_STATEMENT_TXT"] = statement_txt
globals()["CELL15_4B_Q6_POLICY_SPLIT_CONTRACT_JSON"] = contract_json
globals()["CELL15_4B_Q6_POLICY_SPLIT_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL15_4B_Q6_POLICY_SPLIT_MANIFEST_JSON"] = manifest_json

log(
    "[Cell15.4b] Q6 public policy-split summary complete | "
    f"direct_privacy={direct_overall_status} | "
    f"distinguishability={dist_overall_status} | "
    f"strict_combined={strict_overall_status}"
)
log(f"[Cell15.4b] Direct privacy blocker roles | {direct_blocker_roles}")
log(f"[Cell15.4b] Direct privacy warning roles | {direct_warning_roles}")
log(f"[Cell15.4b] Distinguishability blocker roles | {dist_blocker_roles}")
log(f"[Cell15.4b] Distinguishability warning roles | {dist_warning_roles}")
log(f"[Cell15.4b] Saved statement: {statement_txt}")
log(f"[Cell15.4b] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell15.4b] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "synthetic_values_mutated=False | "
    "materialization_done_here=False | "
    "policy_split_summary_done_here=True | "
    "formal_privacy_proof_claimed=False"
)
log("--- END: Cell 15.4b - Q6 public-candidate policy-split summary (v1.0 strict) ---")

gc.collect()
