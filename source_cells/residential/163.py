# ==========================================================
# CELL 20.1 — Final non-continuous publication evidence ledger
# v3.2 STUDY-THESIS claim-scoped / no-Q4-promotion canonical ledger, forbidden-claim-aware Q4 audit
#
# Purpose:
#   Build the final STUDY-THESIS non-continuous publication evidence ledger
#   from canonical Cell 18.3/19.0/19.1 outputs.
#
# Final governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative or promoted.
#   - Q4 evidence is report/ledger evidence only.
#   - Public Q6 candidate is role-restricted.
#   - Direct privacy/no-copy is warning-level.
#   - Strict public release remains distinguishability-blocked.
#
# Scientific contract:
#   - Reads existing reports only.
#   - Does not mutate synthetic values.
#   - Does not fit/select/materialize generators.
#   - Does not use TEST real values.
#
# Outputs:
#   reports/cell20_1_noncontinuous_canonical_evidence_table.csv
#   reports/cell20_1_noncontinuous_fix_priority_ledger.csv
#   reports/cell20_1_noncontinuous_fix_checklist.csv
#   reports/cell20_1_noncontinuous_claim_risk_audit.csv
#   reports/cell20_1_noncontinuous_publication_grade_summary.csv
#   reports/cell20_1_noncontinuous_reviewer_summary.txt
#   reports/cell20_1_noncontinuous_fix_manifest.json
#   artifacts/contracts/cell20_1_noncontinuous_evidence_contract_v3_0_THESIS.json
# ==========================================================

log("--- START: Cell 20.1 — Final non-continuous publication evidence ledger (v3.2 forbidden-claim-aware no-Q4 strict) ---")

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
_required_201 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL19_1_FULL_CPS_EVALUATION_CONTRACT",
]
_missing_201 = [k for k in _required_201 if k not in globals()]
if _missing_201:
    raise RuntimeError(f"[Cell20.1] Missing required globals: {_missing_201}")

ORIGINAL_OUTDIR_201 = str(OUTDIR)
ORIGINAL_OUT_SYN_201 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_201 = str(REPORT_DIR)

def _resolve_project_root_201(outdir, report_dir, out_syn):
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
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "artifacts")):
            return c

    raise RuntimeError("[Cell20.1] Could not resolve canonical project root.")

BASE_201 = _resolve_project_root_201(ORIGINAL_OUTDIR_201, ORIGINAL_REPORT_DIR_201, ORIGINAL_OUT_SYN_201)
REPORT_DIR_201 = os.path.join(BASE_201, "reports")
SYN_DIR_201 = os.path.join(BASE_201, "synthetic")
ARTDIR_201 = os.path.join(BASE_201, "artifacts")
CONTRACT_DIR_201 = os.path.join(ARTDIR_201, "contracts")

os.makedirs(REPORT_DIR_201, exist_ok=True)
os.makedirs(ARTDIR_201, exist_ok=True)
os.makedirs(CONTRACT_DIR_201, exist_ok=True)

SEED = int(SEED)

CELL201_VERSION = "cell20_1_noncontinuous_publication_evidence_ledger_v3_2_forbidden_claim_aware_q4_audit"

CFG["cell20_1_version"] = CELL201_VERSION
CFG["cell20_1_Q4_final_status"] = "blocked_no_promotion"
CFG["cell20_1_Q4_coupled_artifacts_used"] = False
CFG["cell20_1_TEST_real_values_used_here"] = False
CFG["cell20_1_TEST_real_values_used_for_materialization"] = False
CFG["cell20_1_synthetic_values_mutated"] = False
CFG["cell20_1_selection_done_here"] = False
CFG["cell20_1_generator_fit_done_here"] = False
CFG["cell20_1_materialization_done_here"] = False
CFG["cell20_1_noncontinuous_ledger_done_here"] = True

# ----------------------------------------------------------
# 1) Paths
# ----------------------------------------------------------
paths = {
    "claim_manifest_csv": os.path.join(REPORT_DIR_201, "claim_traceability_manifest.csv"),
    "claim_manifest_json": os.path.join(REPORT_DIR_201, "claim_traceability_manifest.json"),
    "readiness_dashboard": os.path.join(REPORT_DIR_201, "publication_readiness_dashboard.csv"),
    "publication_blockers": os.path.join(REPORT_DIR_201, "publication_blockers.csv"),

    "cell19_1_dimension_summary": os.path.join(REPORT_DIR_201, "cell19_1_full_cps_dimension_summary.csv"),
    "cell19_1_role_summary": os.path.join(REPORT_DIR_201, "cell19_1_full_cps_role_summary.csv"),
    "cell19_1_failures": os.path.join(REPORT_DIR_201, "cell19_1_full_cps_column_or_pair_failures.csv"),
    "cell19_1_metric_inventory": os.path.join(REPORT_DIR_201, "cell19_1_full_cps_metric_inventory.csv"),

    "cell19_0_artifact_registry": os.path.join(REPORT_DIR_201, "cell19_0_full_cps_eval_artifact_registry.csv"),
    "cell19_0_metric_source_registry": os.path.join(REPORT_DIR_201, "cell19_0_full_cps_eval_metric_source_registry.csv"),

    "master_ledger": os.path.join(REPORT_DIR_201, "MASTER_RESULTS_LEDGER_FOR_PAPER.csv"),
    "cell18_0_claim_registry": os.path.join(REPORT_DIR_201, "cell18_0_claim_registry.csv"),
    "cell18_1_traceability": os.path.join(REPORT_DIR_201, "cell18_1_claim_traceability_linked.csv"),
    "cell18_2_decision_grade": os.path.join(REPORT_DIR_201, "cell18_2_claim_decision_grade.csv"),

    # Optional branch-specific reports retained as supporting context only.
    "binary_qa": os.path.join(REPORT_DIR_201, "cell12d6_binary_final_test_qa_metrics.csv"),
    "binary_status": os.path.join(REPORT_DIR_201, "cell12d6_binary_publication_status.csv"),
    "binary_blockers": os.path.join(REPORT_DIR_201, "cell12d6_binary_publication_blocked_test_qa_metrics.csv"),
    "driver_qa": os.path.join(REPORT_DIR_201, "cell12e6_driver_final_test_qa_metrics.csv"),
    "driver_status": os.path.join(REPORT_DIR_201, "cell12e6_driver_publication_status.csv"),
    "mask_qa": os.path.join(REPORT_DIR_201, "cell12f5_mask_final_test_qa_metrics.csv"),
    "mask_status": os.path.join(REPORT_DIR_201, "cell12f5_mask_publication_status.csv"),
}

canonical_evidence_path = os.path.join(REPORT_DIR_201, "cell20_1_noncontinuous_canonical_evidence_table.csv")
ledger_path = os.path.join(REPORT_DIR_201, "cell20_1_noncontinuous_fix_priority_ledger.csv")
checklist_path = os.path.join(REPORT_DIR_201, "cell20_1_noncontinuous_fix_checklist.csv")
claim_risk_path = os.path.join(REPORT_DIR_201, "cell20_1_noncontinuous_claim_risk_audit.csv")
summary_path = os.path.join(REPORT_DIR_201, "cell20_1_noncontinuous_publication_grade_summary.csv")
reviewer_summary_path = os.path.join(REPORT_DIR_201, "cell20_1_noncontinuous_reviewer_summary.txt")
manifest_path = os.path.join(REPORT_DIR_201, "cell20_1_noncontinuous_fix_manifest.json")
contract_json = os.path.join(REPORT_DIR_201, "cell20_1_noncontinuous_evidence_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_201, "cell20_1_noncontinuous_evidence_contract_v3_2_THESIS.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_201(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_201(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_201(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_201(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_201(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_201(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_201(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_201(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_201(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_201(payload), f, indent=2, sort_keys=True)

def _sha256_file_201(path):
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_201(path):
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _read_csv_201(path):
    if not _exists_201(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _safe_int_201(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _safe_float_201(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _status_counts_201(df, col):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return {}
    return df[col].astype(str).value_counts().sort_index().to_dict()

def _first_col_201(df, candidates):
    if not isinstance(df, pd.DataFrame):
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _metric_from_dimension_201(df, dim, target_contains=None, status_contains=None):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    d = df[df["quality_dimension"].astype(str).eq(str(dim))].copy() if "quality_dimension" in df.columns else df.copy()
    if target_contains and "evaluation_target" in d.columns:
        d = d[d["evaluation_target"].astype(str).str.contains(str(target_contains), case=False, na=False)]
    if status_contains and "status" in d.columns:
        d = d[d["status"].astype(str).str.contains(str(status_contains), case=False, na=False)]
    if len(d) == 0:
        return {}
    return d.iloc[0].to_dict()

def _add_issue(rows, issue_id, priority, component, issue, evidence, target, fix_type,
               acceptance, risk_if_unfixed, expected_effort="low", patch_required=False,
               task_type="paper_reporting", reviewer_note="", status="paper_action_required"):
    rows.append({
        "priority": int(priority),
        "issue_id": str(issue_id),
        "component": str(component),
        "issue": str(issue),
        "evidence": str(evidence),
        "fix_target": str(target),
        "fix_type": str(fix_type),
        "task_type": str(task_type),
        "acceptance_criteria": str(acceptance),
        "risk_if_unfixed": str(risk_if_unfixed),
        "expected_effort": str(expected_effort),
        "patch_required": bool(patch_required),
        "reviewer_note": str(reviewer_note),
        "status": str(status),
    })

# ----------------------------------------------------------
# 3) Validate upstream Cell 19.1 governance
# ----------------------------------------------------------
cell19_1_version_201 = str(CELL19_1_FULL_CPS_EVALUATION_CONTRACT.get("version", ""))
if "cell19_1_full_cps_consolidated_metric_evaluation_v1_2" not in cell19_1_version_201:
    raise RuntimeError(f"[Cell20.1] Unexpected Cell 19.1 contract version: {cell19_1_version_201}")

q4 = CELL19_1_FULL_CPS_EVALUATION_CONTRACT.get("q4_governance", {})
strict = CELL19_1_FULL_CPS_EVALUATION_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell20.1] Cell 19.1 does not carry Q4 blocked_no_promotion governance.")
if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell20.1] Cell 19.1 indicates Q4-coupled artifacts were used.")

q6ctx = CELL19_1_FULL_CPS_EVALUATION_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6ctx.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6ctx.get("public_distinguishability_status", ""))
public_strict_status = str(q6ctx.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 4) Load reports
# ----------------------------------------------------------
claim_manifest = _read_csv_201(paths["claim_manifest_csv"])
readiness_dashboard = _read_csv_201(paths["readiness_dashboard"])
publication_blockers = _read_csv_201(paths["publication_blockers"])
dim_summary = _read_csv_201(paths["cell19_1_dimension_summary"])
role_summary = _read_csv_201(paths["cell19_1_role_summary"])
failure_rows = _read_csv_201(paths["cell19_1_failures"])
metric_inventory = _read_csv_201(paths["cell19_1_metric_inventory"])
artifact_registry = _read_csv_201(paths["cell19_0_artifact_registry"])
metric_source_registry = _read_csv_201(paths["cell19_0_metric_source_registry"])
master_ledger = _read_csv_201(paths["master_ledger"])
binary_qa = _read_csv_201(paths["binary_qa"])
binary_status = _read_csv_201(paths["binary_status"])
binary_blockers = _read_csv_201(paths["binary_blockers"])
driver_qa = _read_csv_201(paths["driver_qa"])
driver_status = _read_csv_201(paths["driver_status"])
mask_qa = _read_csv_201(paths["mask_qa"])
mask_status = _read_csv_201(paths["mask_status"])

# ----------------------------------------------------------
# 5) Canonical facts
# ----------------------------------------------------------
facts = {}

facts["claim_total"] = int(len(claim_manifest))
facts["claim_grade_counts"] = _status_counts_201(claim_manifest, "decision_grade")
facts["claim_final_status_counts"] = _status_counts_201(claim_manifest, "claim_final_status")
facts["dashboard_rows"] = int(len(readiness_dashboard))
facts["dashboard_status_counts"] = _status_counts_201(readiness_dashboard, "dashboard_status")
facts["publication_blockers_or_caveats_rows"] = int(len(publication_blockers))
facts["dim_summary_rows"] = int(len(dim_summary))
facts["dim_status_counts"] = _status_counts_201(dim_summary, "status")
facts["failure_rows"] = int(len(failure_rows))
facts["metric_inventory_rows"] = int(len(metric_inventory))
facts["required_artifacts_existing"] = int((artifact_registry.get("required", pd.Series(dtype=bool)).astype(bool) & artifact_registry.get("exists", pd.Series(dtype=bool)).astype(bool)).sum()) if len(artifact_registry) else 0
facts["required_artifacts_total"] = int(artifact_registry.get("required", pd.Series(dtype=bool)).astype(bool).sum()) if len(artifact_registry) else 0
facts["required_metric_sources_existing"] = int((metric_source_registry.get("required", pd.Series(dtype=bool)).astype(bool) & metric_source_registry.get("exists", pd.Series(dtype=bool)).astype(bool)).sum()) if len(metric_source_registry) else 0
facts["required_metric_sources_total"] = int(metric_source_registry.get("required", pd.Series(dtype=bool)).astype(bool).sum()) if len(metric_source_registry) else 0

# Branch-supporting context if present.
facts["binary_targets"] = int(len(binary_qa))
facts["binary_blocker_rows"] = int(len(binary_blockers))
facts["binary_status_counts"] = _status_counts_201(binary_status, "publication_status")
facts["driver_targets"] = int(len(driver_qa))
facts["driver_status_counts"] = _status_counts_201(driver_status, "publication_status")
facts["mask_targets"] = int(len(mask_qa))
facts["mask_status_counts"] = _status_counts_201(mask_status, "publication_status")

# Dimension-specific rows from consolidated summary.
q1_sci = _metric_from_dimension_201(dim_summary, "Q1_marginal", "scientific")
q1_pub = _metric_from_dimension_201(dim_summary, "Q1_marginal", "public")
q2_sci = _metric_from_dimension_201(dim_summary, "Q2_temporal", "scientific")
q2_pub = _metric_from_dimension_201(dim_summary, "Q2_temporal", "public")
q3_sci = _metric_from_dimension_201(dim_summary, "Q3_observability", "scientific")
q3_pub = _metric_from_dimension_201(dim_summary, "Q3_observability", "public")
q4_any = dim_summary[dim_summary["quality_dimension"].astype(str).eq("Q4_coupling")].copy() if len(dim_summary) and "quality_dimension" in dim_summary.columns else pd.DataFrame()
q6_row = _metric_from_dimension_201(dim_summary, "Q6_privacy_release", "public")
baseline_row = _metric_from_dimension_201(dim_summary, "external_baselines")
claim_row = _metric_from_dimension_201(dim_summary, "claim_traceability")

facts["q4_stage_rows"] = int(len(q4_any))
facts["q4_status_counts"] = _status_counts_201(q4_any, "status")
facts["q6_status"] = str(q6_row.get("status", public_strict_status))
facts["baseline_status"] = str(baseline_row.get("status", ""))
facts["claim_traceability_status"] = str(claim_row.get("status", ""))

# ----------------------------------------------------------
# 6) Canonical evidence table
# ----------------------------------------------------------
evidence_rows = [
    {
        "component": "claim_traceability_dashboard",
        "canonical_status": "claim_scoped_supported_with_limitations",
        "targets_or_pairs": facts["claim_total"],
        "pass_warning_fatal_or_equiv": json.dumps(facts["claim_grade_counts"], sort_keys=True),
        "blocker_n": 0,
        "key_metric_summary": f"claims={facts['claim_total']}; grades={facts['claim_grade_counts']}; dashboard_statuses={facts['dashboard_status_counts']}; blockers_or_caveats={facts['publication_blockers_or_caveats_rows']}",
        "canonical_claim": "The claim set is traceable and claim-scoped, but supported with limitations/caveats rather than global readiness.",
        "paper_allowed_claim": "Claim-scoped evidence ledger with explicit limitations and no Grade-D claims.",
        "paper_forbidden_claim": "Global publication-ready status or uncaveated full-artifact readiness.",
    },
    {
        "component": "scientific_no_q4_artifact_registry",
        "canonical_status": "required_artifacts_present_no_q4",
        "targets_or_pairs": facts["required_artifacts_existing"],
        "pass_warning_fatal_or_equiv": f"{facts['required_artifacts_existing']}/{facts['required_artifacts_total']} required artifacts exist",
        "blocker_n": 0 if facts["required_artifacts_existing"] == facts["required_artifacts_total"] else facts["required_artifacts_total"] - facts["required_artifacts_existing"],
        "key_metric_summary": f"required_metric_sources={facts['required_metric_sources_existing']}/{facts['required_metric_sources_total']}",
        "canonical_claim": "Required no-Q4 scientific and public Q6 artifacts are present.",
        "paper_allowed_claim": "No-Q4 scientific artifact and role-restricted public artifact are available for evidence/reporting.",
        "paper_forbidden_claim": "Final Q4-coupled artifact is authoritative.",
    },
    {
        "component": "Q1_marginal",
        "canonical_status": str(q1_sci.get("status", "not_evaluable")),
        "targets_or_pairs": q1_sci.get("cols_or_pairs_evaluable", q1_sci.get("claims_or_items_total", np.nan)),
        "pass_warning_fatal_or_equiv": f"sci_status={q1_sci.get('status','')}; public_status={q1_pub.get('status','')}",
        "blocker_n": _safe_int_201(q1_sci.get("blocker_n"), 0),
        "key_metric_summary": f"sci_penalty={q1_sci.get('primary_penalty_score', np.nan)}; public_penalty={q1_pub.get('primary_penalty_score', np.nan)}",
        "canonical_claim": "Q1 marginal quality is claim-scoped and role-dependent.",
        "paper_allowed_claim": "Q1 is reported dimension-wise with scientific/public scope separation.",
        "paper_forbidden_claim": "Uniform marginal realism across all CPS columns.",
    },
    {
        "component": "Q2_temporal",
        "canonical_status": str(q2_sci.get("status", "not_evaluable")),
        "targets_or_pairs": q2_sci.get("cols_or_pairs_evaluable", q2_sci.get("claims_or_items_total", np.nan)),
        "pass_warning_fatal_or_equiv": f"sci_status={q2_sci.get('status','')}; public_status={q2_pub.get('status','')}",
        "blocker_n": _safe_int_201(q2_sci.get("blocker_n"), 0),
        "key_metric_summary": f"sci_penalty={q2_sci.get('primary_penalty_score', np.nan)}; public_penalty={q2_pub.get('primary_penalty_score', np.nan)}",
        "canonical_claim": "Q2 temporal quality is claim-scoped and role-dependent.",
        "paper_allowed_claim": "Q2 is reported dimension-wise with scientific/public scope separation.",
        "paper_forbidden_claim": "Uniform temporal realism across all CPS roles.",
    },
    {
        "component": "Q3_observability",
        "canonical_status": str(q3_sci.get("status", "not_evaluable")),
        "targets_or_pairs": q3_sci.get("cols_or_pairs_evaluable", q3_sci.get("claims_or_items_total", np.nan)),
        "pass_warning_fatal_or_equiv": f"sci_status={q3_sci.get('status','')}; public_status={q3_pub.get('status','')}",
        "blocker_n": _safe_int_201(q3_sci.get("blocker_n"), 0),
        "key_metric_summary": f"sci_penalty={q3_sci.get('primary_penalty_score', np.nan)}; public_scope={q3_pub.get('status','')}",
        "canonical_claim": "Observability is a known limitation/boundary; public candidate intentionally excludes fine-grained observability.",
        "paper_allowed_claim": "Q3 is quality-tiered and public-scope excluded.",
        "paper_forbidden_claim": "Full observability branch is publication-clean.",
    },
    {
        "component": "Q4_coupling",
        "canonical_status": "blocked_no_promotion_report_only",
        "targets_or_pairs": facts["q4_stage_rows"],
        "pass_warning_fatal_or_equiv": json.dumps(facts["q4_status_counts"], sort_keys=True),
        "blocker_n": int((q4_any["status"].astype(str).str.contains("block", case=False, na=False)).sum()) if len(q4_any) and "status" in q4_any.columns else 0,
        "key_metric_summary": f"stage_rows={facts['q4_stage_rows']}; status_counts={facts['q4_status_counts']}",
        "canonical_claim": "Q4 remains blocked_no_promotion; evidence is report-only and no Q4-coupled artifact is authoritative.",
        "paper_allowed_claim": "Q4 failure/governance boundary is reported as limitation.",
        "paper_forbidden_claim": "A0 accepted/promoted or final Q4-coupled artifact is publication-ready.",
    },
    {
        "component": "Q6_public_release",
        "canonical_status": public_strict_status or facts["q6_status"],
        "targets_or_pairs": q6_row.get("claims_or_items_total", np.nan),
        "pass_warning_fatal_or_equiv": q6_row.get("metric_summary_json", ""),
        "blocker_n": _safe_int_201(q6_row.get("blocker_n"), 0),
        "key_metric_summary": f"direct={public_direct_privacy_status}; distinguishability={public_distinguishability_status}; strict={public_strict_status}",
        "canonical_claim": "Public Q6 candidate is role-restricted; direct privacy/no-copy is warning-level, but strict release remains distinguishability-blocked.",
        "paper_allowed_claim": "Direct privacy warning and strict distinguishability blocker must both be disclosed.",
        "paper_forbidden_claim": "Public artifact is unrestricted release-ready or has formal privacy guarantee.",
    },
    {
        "component": "external_baselines",
        "canonical_status": facts["baseline_status"],
        "targets_or_pairs": baseline_row.get("claims_or_items_total", np.nan),
        "pass_warning_fatal_or_equiv": baseline_row.get("metric_summary_json", ""),
        "blocker_n": 0,
        "key_metric_summary": "same-scope single-run comparison only",
        "canonical_claim": "External baselines are same-scope only and single-run; they are not full CPS replacements.",
        "paper_allowed_claim": "Fair-scope baseline comparison with limitations.",
        "paper_forbidden_claim": "Global outperformance over all external baselines or full-CPS replacement comparison.",
    },
]

canonical_evidence_df = pd.DataFrame(evidence_rows)

# ----------------------------------------------------------
# 7) Build issue/fix ledger
# ----------------------------------------------------------
issues = []

_add_issue(
    issues,
    "CLAIM-SCOPED-WORDING",
    0,
    "Final reporting / claim traceability",
    "All final claims are supported with caveats/limitations, not global readiness.",
    f"claim_grades={facts['claim_grade_counts']} | dashboard_statuses={facts['dashboard_status_counts']}",
    "manuscript claims, abstract, conclusion, artifact README",
    "claim_scope_fix",
    "Every claim explicitly declares scope and avoids global publication-ready language.",
    "Reviewers may reject unsupported full-readiness or uniform-realism claims.",
    expected_effort="low",
    patch_required=False,
)

_add_issue(
    issues,
    "Q4-NO-PROMOTION",
    1,
    "Cross-modal coupling / Q4",
    "Q4 remains blocked_no_promotion; no Q4-coupled artifact is authoritative.",
    f"q4_status_counts={facts['q4_status_counts']}",
    "Q4 results, limitations, artifact manifest",
    "claim_scope_fix",
    "State Q4 as report-only evidence and limitation/governance boundary.",
    "Claiming A0 accepted/promoted contradicts final governance.",
    expected_effort="low",
    patch_required=False,
)

_add_issue(
    issues,
    "Q6-STRICT-RELEASE-BLOCKER",
    2,
    "Public release governance",
    "Public candidate direct privacy/no-copy status and strict release status differ.",
    f"direct={public_direct_privacy_status}; distinguishability={public_distinguishability_status}; strict={public_strict_status}",
    "release section, public README, artifact card",
    "release_governance_fix",
    "Public artifact is described as direct-privacy-warning and strict-distinguishability-blocked, not unrestricted release-ready.",
    "Reviewers may interpret the public artifact as privacy-safe if strict release blocker is hidden.",
    expected_effort="low",
    patch_required=False,
)

_add_issue(
    issues,
    "Q1-Q2-Q3-LIMITATIONS",
    3,
    "Quality dimensions",
    "Q1/Q2/Q3 have fatal/limitation/scope-excluded statuses in the consolidated evidence.",
    f"dimension_status_counts={facts['dim_status_counts']}",
    "results tables and limitations",
    "quality_tier_reporting",
    "Report dimension-specific status and avoid uniform realism claims.",
    "Aggregating these into one quality score would be misleading.",
    expected_effort="low",
    patch_required=False,
)

_add_issue(
    issues,
    "BASELINE-FAIR-SCOPE-ONLY",
    4,
    "External baselines",
    "Baseline evidence is fair-scope and single-run only.",
    f"baseline_status={facts['baseline_status']}",
    "baseline comparison section",
    "fairness_reporting",
    "State compatible-scope comparison only and no statistical/multi-seed superiority claim.",
    "Unsupported baseline superiority claim would be vulnerable.",
    expected_effort="low",
    patch_required=False,
)

_add_issue(
    issues,
    "ARTIFACT-PACKAGE-FINALIZATION",
    5,
    "Artifact package",
    "Final package must align with no-Q4 governance and claim-scoped dashboard.",
    f"required_artifacts={facts['required_artifacts_existing']}/{facts['required_artifacts_total']} | required_metric_sources={facts['required_metric_sources_existing']}/{facts['required_metric_sources_total']}",
    "README, checksums, schema, artifact index",
    "artifact_packaging",
    "Package includes no-Q4 scientific track, role-restricted public track, claim manifest, dashboard, blockers/caveats, and checksums.",
    "Artifact review will be weaker without clear no-Q4/Q6 governance packaging.",
    expected_effort="medium",
    patch_required=False,
)

ledger = pd.DataFrame(issues).sort_values(["priority", "issue_id"]).reset_index(drop=True)

# ----------------------------------------------------------
# 8) Checklist
# ----------------------------------------------------------
checklist = ledger[[
    "priority", "issue_id", "component", "fix_target", "acceptance_criteria",
    "patch_required", "task_type", "status"
]].copy()
checklist.insert(0, "done", False)
checklist["confirmation_evidence_to_send"] = checklist["issue_id"].map({
    "CLAIM-SCOPED-WORDING": "claim_traceability_manifest.csv and publication_readiness_dashboard.csv",
    "Q4-NO-PROMOTION": "cell19_1_full_cps_dimension_summary.csv Q4 rows and Cell 19.1 contract",
    "Q6-STRICT-RELEASE-BLOCKER": "cell18/19 Q6 context and public release ledger",
    "Q1-Q2-Q3-LIMITATIONS": "cell19_1_full_cps_dimension_summary.csv",
    "BASELINE-FAIR-SCOPE-ONLY": "cell17_6_baseline_publication_safe_summary.csv and Cell 19.1 external baseline row",
    "ARTIFACT-PACKAGE-FINALIZATION": "artifact index/checksums after Cell 20.3",
}).fillna("Relevant report excerpt.")

# ----------------------------------------------------------
# 9) Claim risk audit
# ----------------------------------------------------------
risk_rows = []
if len(claim_manifest):
    for _, r in claim_manifest.iterrows():
        claim_id = str(r.get("claim_id", ""))
        claim_text = str(r.get("claim_text", ""))
        dim = str(r.get("quality_dimension", ""))
        grade = str(r.get("decision_grade", ""))
        status = str(r.get("claim_final_status", r.get("publication_status", "")))
        forbidden = str(r.get("forbidden_claim", ""))
        limitation = str(r.get("known_limitation", ""))

        blob = " ".join([claim_text, dim, grade, status, forbidden, limitation]).lower()
        reasons = []
        action = "none"

        if "q4" in blob or "coupling" in blob:
            reasons.append("requires_no_q4_promotion_wording")
            action = "state_Q4_blocked_no_promotion"
        if "release" in blob or "privacy" in blob or "q6" in blob:
            reasons.append("requires_Q6_direct_vs_strict_release_distinction")
            action = "state_direct_warning_and_strict_blocker"
        if grade in {"B", "C"}:
            reasons.append("requires_caveat_or_limitation_wording")
            if action == "none":
                action = "keep_claim_scoped_with_caveat"
        if "global" in blob or "publication_ready" in blob or "unrestricted" in blob:
            reasons.append("check_no_global_readiness_overclaim")
            action = "avoid_global_ready_language"

        risk_rows.append({
            "claim_id": claim_id,
            "quality_dimension": dim,
            "decision_grade": grade,
            "claim_final_status": status,
            "risk_flag": bool(reasons),
            "risk_reason": "|".join(sorted(set(reasons))),
            "required_action": action,
            "claim_text": claim_text,
        })
claim_risk = pd.DataFrame(risk_rows)
if len(claim_risk) == 0:
    claim_risk = pd.DataFrame(columns=[
        "claim_id", "quality_dimension", "decision_grade", "claim_final_status",
        "risk_flag", "risk_reason", "required_action", "claim_text"
    ])

# ----------------------------------------------------------
# 10) Publication-grade summary
# ----------------------------------------------------------
pub_summary = pd.DataFrame([
    {
        "item": "claim_traceability",
        "canonical_status": "claim_scoped_supported_with_limitations",
        "key_numbers": f"claims={facts['claim_total']}; grades={facts['claim_grade_counts']}; dashboard={facts['dashboard_status_counts']}",
        "paper_action": "Use claim-scoped wording only; no global publication-ready claim.",
    },
    {
        "item": "no_q4_scientific_artifact",
        "canonical_status": "primary_internal_scientific_track",
        "key_numbers": f"required_artifacts={facts['required_artifacts_existing']}/{facts['required_artifacts_total']}",
        "paper_action": "Describe as scientific/internal no-Q4 track.",
    },
    {
        "item": "q4_coupling",
        "canonical_status": "blocked_no_promotion_report_only",
        "key_numbers": f"q4_status_counts={facts['q4_status_counts']}",
        "paper_action": "Report Q4 as limitation/governance evidence only.",
    },
    {
        "item": "q6_public_release",
        "canonical_status": public_strict_status or "release_blocker",
        "key_numbers": f"direct={public_direct_privacy_status}; distinguishability={public_distinguishability_status}; strict={public_strict_status}",
        "paper_action": "Separate direct privacy warning from strict release blocker.",
    },
    {
        "item": "external_baselines",
        "canonical_status": facts["baseline_status"],
        "key_numbers": "single-run fair-scope evidence only",
        "paper_action": "Avoid global baseline superiority claim.",
    },
])

reviewer_statement = f"""Final non-continuous publication evidence ledger — no-Q4-promotion canonical version

This ledger confirms that remaining non-continuous work is paper/reporting and artifact-packaging work, not generator repair.

Canonical statuses:
- Claim traceability: {facts['claim_total']} claims; grades={facts['claim_grade_counts']}; dashboard_statuses={facts['dashboard_status_counts']}.
- Scientific artifact track: no-Q4 scientific/internal artifact; required artifacts present={facts['required_artifacts_existing']}/{facts['required_artifacts_total']}.
- Q4 coupling: blocked_no_promotion. Q4 evidence is report-only; no Q4-coupled artifact is authoritative.
- Public Q6 release: direct privacy/no-copy status={public_direct_privacy_status}; distinguishability status={public_distinguishability_status}; strict combined status={public_strict_status}.
- External baselines: same-scope, single-run evidence only; no global superiority claim.

Reviewer-facing conclusion:
The paper should not claim uniform full-artifact realism, unrestricted public-release safety, or promoted Q4-coupled artifact quality. It should claim a quality-stratified CPS synthesis/evaluation artifact with claim-scoped evidence, a no-Q4 scientific track, a role-restricted public candidate with Q6 caveats, and Q4 coupling as a documented no-promotion limitation.
"""

# ----------------------------------------------------------
# 11) Audits
# ----------------------------------------------------------
audit_rows = []
# Audit only positive Q4-promotion assertions.
# Safe labels such as scientific_no_q4_artifact_registry,
# required_artifacts_present_no_q4, and blocked_no_promotion_report_only
# are explicitly allowed.
def _unsafe_positive_q4_promotion_201(row):
    """
    Detect positive Q4-promotion claims only.

    Do not inspect paper_forbidden_claim. That field intentionally contains
    unsafe phrases to tell the paper writer what NOT to claim.
    """
    positive_cols = [
        "component",
        "canonical_status",
        "canonical_claim",
        "paper_allowed_claim",
        "key_metric_summary",
    ]
    blob = " ".join([str(row.get(c, "")) for c in positive_cols if c in row.index]).lower()
    blob_norm = blob.replace("_", " ").replace("-", " ")

    safe_patterns = [
        "no q4",
        "noq4",
        "blocked no promotion",
        "blocked/no promotion",
        "report only",
        "report-only",
        "not authoritative",
        "q4 evidence is report only",
        "q4 coupled artifacts used false",
        "q4 coupled artifacts used=false",
        "not promoted",
    ]
    if any(p in blob_norm for p in safe_patterns):
        return False

    positive_patterns = [
        "a0 accepted",
        "accepted final q4",
        "accepted q4",
        "final q4 coupled",
        "q4 coupled artifact is authoritative",
        "q4 promoted",
        "publication ready q4",
        "q4 publication ready",
    ]
    return any(p in blob_norm for p in positive_patterns)

unsafe_q4 = canonical_evidence_df[canonical_evidence_df.apply(_unsafe_positive_q4_promotion_201, axis=1)]
audit_rows.append({
    "audit_check": "no_positive_q4_promotion_language",
    "passed": bool(len(unsafe_q4) == 0),
    "details": f"unsafe_rows={unsafe_q4[['component','canonical_status']].to_dict('records') if len(unsafe_q4) else []}",
})
audit_rows.append({
    "audit_check": "outputs_nonempty",
    "passed": bool(len(canonical_evidence_df) > 0 and len(ledger) > 0 and len(checklist) > 0),
    "details": f"evidence_rows={len(canonical_evidence_df)}; ledger_rows={len(ledger)}; checklist_rows={len(checklist)}",
})
audit_df = pd.DataFrame(audit_rows)
audit_passed = bool(audit_df["passed"].astype(bool).all())
if not audit_passed:
    raise RuntimeError(f"[Cell20.1] Audit failed: {audit_df.loc[~audit_df['passed'].astype(bool)].to_dict('records')}")

# ----------------------------------------------------------
# 12) Save outputs
# ----------------------------------------------------------
canonical_evidence_df.to_csv(canonical_evidence_path, index=False)
ledger.to_csv(ledger_path, index=False)
checklist.to_csv(checklist_path, index=False)
claim_risk.to_csv(claim_risk_path, index=False)
pub_summary.to_csv(summary_path, index=False)

with open(reviewer_summary_path, "w", encoding="utf-8") as f:
    f.write(reviewer_statement)

contract = {
    "cell": "20.1",
    "version": CELL201_VERSION,
    "role": "final_noncontinuous_publication_evidence_ledger_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_evidence_is_report_only": True,
        "no_q4_coupled_artifact_authoritative": True,
        "safe_no_q4_and_blocked_no_promotion_wording_allowed": True,
        "paper_forbidden_claim_excluded_from_positive_q4_audit": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell19_1": cell19_1_version_201,
    },
    "summary": {
        "canonical_evidence_rows": int(len(canonical_evidence_df)),
        "ledger_rows": int(len(ledger)),
        "checklist_rows": int(len(checklist)),
        "claim_risk_rows": int(len(claim_risk)),
        "claim_risk_flagged_n": int(claim_risk["risk_flag"].astype(bool).sum()) if len(claim_risk) else 0,
        "audit_passed": bool(audit_passed),
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "safe_no_q4_and_blocked_no_promotion_wording_allowed": True,
        "paper_forbidden_claim_excluded_from_positive_q4_audit": True,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "diagnostic_and_reporting_only": True,
        "safe_no_q4_and_blocked_no_promotion_wording_allowed": True,
        "paper_forbidden_claim_excluded_from_positive_q4_audit": True,
        "paper_forbidden_claim_excluded_from_positive_q4_audit": True,
    },
    "outputs": {
        "canonical_evidence_table": canonical_evidence_path,
        "fix_priority_ledger": ledger_path,
        "fix_checklist": checklist_path,
        "claim_risk_audit": claim_risk_path,
        "publication_grade_summary": summary_path,
        "reviewer_summary": reviewer_summary_path,
        "manifest": manifest_path,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
    },
}
_write_json_201(contract_json, contract)
_write_json_201(contract_canonical_json, contract)

manifest = {
    "cell": "20.1",
    "version": CELL201_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "canonical_evidence": canonical_evidence_df.to_dict("records"),
    "ledger": ledger.to_dict("records"),
    "checklist": checklist.to_dict("records"),
    "claim_risk": claim_risk.to_dict("records"),
    "publication_summary": pub_summary.to_dict("records"),
    "facts": facts,
    "audit": audit_df.to_dict("records"),
    "strict_contract": contract["strict_contract"],
}
_write_json_201(manifest_path, manifest)

hashes = {
    "canonical_evidence_sha256": _sha256_file_201(canonical_evidence_path),
    "ledger_sha256": _sha256_file_201(ledger_path),
    "checklist_sha256": _sha256_file_201(checklist_path),
    "claim_risk_sha256": _sha256_file_201(claim_risk_path),
    "summary_sha256": _sha256_file_201(summary_path),
    "reviewer_summary_sha256": _sha256_file_201(reviewer_summary_path),
    "manifest_sha256": _sha256_file_201(manifest_path),
    "contract_json_sha256": _sha256_file_201(contract_json),
    "contract_canonical_json_sha256": _sha256_file_201(contract_canonical_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_201(contract_json, contract)
_write_json_201(contract_canonical_json, contract)
_write_json_201(manifest_path, manifest)

# ----------------------------------------------------------
# 13) Export globals
# ----------------------------------------------------------
globals()["CELL201_VERSION"] = CELL201_VERSION
globals()["CELL20_1_NONCONTINUOUS_CANONICAL_EVIDENCE_DF"] = canonical_evidence_df
globals()["CELL20_1_NONCONTINUOUS_FIX_PRIORITY_LEDGER_DF"] = ledger
globals()["CELL20_1_NONCONTINUOUS_FIX_CHECKLIST_DF"] = checklist
globals()["CELL20_1_NONCONTINUOUS_CLAIM_RISK_AUDIT_DF"] = claim_risk
globals()["CELL20_1_NONCONTINUOUS_PUBLICATION_GRADE_SUMMARY_DF"] = pub_summary
globals()["CELL20_1_NONCONTINUOUS_EVIDENCE_CONTRACT"] = contract

globals()["CELL20_1_NONCONTINUOUS_CANONICAL_EVIDENCE_CSV"] = canonical_evidence_path
globals()["CELL20_1_NONCONTINUOUS_FIX_PRIORITY_LEDGER_CSV"] = ledger_path
globals()["CELL20_1_NONCONTINUOUS_FIX_CHECKLIST_CSV"] = checklist_path
globals()["CELL20_1_NONCONTINUOUS_CLAIM_RISK_AUDIT_CSV"] = claim_risk_path
globals()["CELL20_1_NONCONTINUOUS_PUBLICATION_GRADE_SUMMARY_CSV"] = summary_path
globals()["CELL20_1_NONCONTINUOUS_REVIEWER_SUMMARY_TXT"] = reviewer_summary_path
globals()["CELL20_1_NONCONTINUOUS_FIX_MANIFEST_JSON"] = manifest_path
globals()["CELL20_1_NONCONTINUOUS_EVIDENCE_CONTRACT_JSON"] = contract_json
globals()["CELL20_1_NONCONTINUOUS_EVIDENCE_CONTRACT_CANONICAL_JSON"] = contract_canonical_json

log(
    "[Cell20.1] Final non-continuous evidence ledger complete | "
    f"evidence_rows={len(canonical_evidence_df)} | "
    f"ledger_rows={len(ledger)} | "
    f"claim_risk_flagged={int(claim_risk['risk_flag'].astype(bool).sum()) if len(claim_risk) else 0} | "
    "q4_status=blocked_no_promotion"
)
log(
    "[Cell20.1] Canonical statuses | "
    f"{canonical_evidence_df['canonical_status'].astype(str).value_counts().sort_index().to_dict()}"
)
log(f"[Cell20.1] Saved canonical evidence: {canonical_evidence_path}")
log(f"[Cell20.1] Saved fix ledger: {ledger_path}")
log(f"[Cell20.1] Saved checklist: {checklist_path}")
log(f"[Cell20.1] Saved reviewer summary: {reviewer_summary_path}")
log(f"[Cell20.1] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell20.1] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "diagnostic_and_reporting_only=True | safe_no_q4_and_blocked_no_promotion_wording_allowed=True | paper_forbidden_claim_excluded_from_positive_q4_audit=True"
)
log("--- END: Cell 20.1 — Final non-continuous publication evidence ledger (v3.2 forbidden-claim-aware no-Q4 strict) ---")

gc.collect()