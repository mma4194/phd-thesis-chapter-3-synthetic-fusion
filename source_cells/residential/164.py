# ==========================================================
# CELL 20.3 — STUDY-THESIS artifact package README and index builder
# v3.0 STUDY-THESIS no-Q4-promotion / claim-scoped artifact package
#
# Purpose:
#   Fix ARTIFACT-PACKAGE-COMPLETENESS by creating a reviewer-facing
#   artifact package index, checksums, release-scope table, claim-scope
#   table, README, and package contract.
#
# Final governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative or promoted.
#   - Q4 evidence is report/ledger evidence only.
#   - Primary scientific artifact = no-Q4 CPS artifact.
#   - Public artifact = role-restricted Q6 candidate.
#   - Public direct privacy/no-copy = release_warning.
#   - Public strict release = release_blocker because distinguishability remains blocked.
#
# Scientific contract:
#   - Does NOT mutate synthetic data.
#   - Does NOT fit, select, or materialize generators.
#   - Does NOT use TEST real values.
#   - Documentation/index/checksum generation only.
#
# Outputs:
#   reports/artifact_package_index.csv
#   reports/artifact_package_checksums.csv
#   reports/artifact_release_scope_table.csv
#   reports/artifact_claim_scope_table.csv
#   reports/ARTIFACT_README_STUDY_THESIS.md
#   reports/cell20_3_artifact_package_contract.json
#   artifacts/contracts/cell20_3_artifact_package_contract_v3_0_THESIS.json
#   artifacts/cell20_3_artifact_package_manifest.json
# ==========================================================

log("--- START: Cell 20.3 — STUDY-THESIS artifact package README and index builder (v3.0 no-Q4-promotion strict) ---")

import os
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_203 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL20_1_NONCONTINUOUS_EVIDENCE_CONTRACT",
]
_missing_203 = [k for k in _required_203 if k not in globals()]
if _missing_203:
    raise RuntimeError(f"[Cell20.3] Missing required globals: {_missing_203}")

ORIGINAL_OUTDIR_203 = str(OUTDIR)
ORIGINAL_OUT_SYN_203 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_203 = str(REPORT_DIR)

def _resolve_project_root_203(outdir, report_dir, out_syn):
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
        if os.path.isdir(os.path.join(c, "synthetic")) and os.path.isdir(os.path.join(c, "reports")):
            return c

    raise RuntimeError("[Cell20.3] Could not resolve canonical project root.")

BASE_203 = _resolve_project_root_203(ORIGINAL_OUTDIR_203, ORIGINAL_REPORT_DIR_203, ORIGINAL_OUT_SYN_203)
REPORT_DIR_203 = os.path.join(BASE_203, "reports")
SYN_DIR_203 = os.path.join(BASE_203, "synthetic")
ARTDIR_203 = os.path.join(BASE_203, "artifacts")
CONTRACT_DIR_203 = os.path.join(ARTDIR_203, "contracts")

PUBLIC_Q6_REPORT_DIRS_203 = [
    os.path.join(BASE_203, "q6_public_reaudit_v1", "reports"),
    os.path.join(BASE_203, "q6_public_reaudit", "reports"),
]

os.makedirs(REPORT_DIR_203, exist_ok=True)
os.makedirs(ARTDIR_203, exist_ok=True)
os.makedirs(CONTRACT_DIR_203, exist_ok=True)

SEED = int(SEED)

CELL203_VERSION = "cell20_3_study_thesis_artifact_package_v3_0_no_q4_promotion"

CFG["cell20_3_version"] = CELL203_VERSION
CFG["cell20_3_Q4_final_status"] = "blocked_no_promotion"
CFG["cell20_3_Q4_coupled_artifacts_used"] = False
CFG["cell20_3_TEST_real_values_used_here"] = False
CFG["cell20_3_TEST_real_values_used_for_materialization"] = False
CFG["cell20_3_synthetic_values_mutated"] = False
CFG["cell20_3_selection_done_here"] = False
CFG["cell20_3_generator_fit_done_here"] = False
CFG["cell20_3_materialization_done_here"] = False
CFG["cell20_3_artifact_package_done_here"] = True

# ----------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------
def _json_sanitize_203(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_203(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_203(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_203(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_203(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_203(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_203(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_203(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_203(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_203(payload), f, indent=2, sort_keys=True)

def _exists_203(path):
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _sha256_file_203(path):
    if not _exists_203(path) or not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _file_size_203(path):
    if not _exists_203(path):
        return None
    return int(os.path.getsize(path))

def _safe_read_csv_203(path):
    if not _exists_203(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _safe_shape_from_parquet_203(path):
    if not _exists_203(path):
        return None
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return [int(pf.metadata.num_rows), int(len(pf.schema_arrow.names))]
    except Exception:
        try:
            df = pd.read_parquet(path)
            return [int(df.shape[0]), int(df.shape[1])]
        except Exception:
            return None

def _first_existing_203(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0] if paths else ""

def _status_counts_203(df, cols=("release_status", "publication_status", "claim_final_status", "dashboard_status", "status")):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    for c in cols:
        if c in df.columns:
            return df[c].astype(str).value_counts().sort_index().to_dict()
    return {}

def _scope_path_203(rel):
    return os.path.join(REPORT_DIR_203, rel)

# ----------------------------------------------------------
# 2) Validate upstream Cell 20.1 governance
# ----------------------------------------------------------
cell20_1_version_203 = str(CELL20_1_NONCONTINUOUS_EVIDENCE_CONTRACT.get("version", ""))
if "cell20_1_noncontinuous_publication_evidence_ledger_v3_2" not in cell20_1_version_203:
    raise RuntimeError(f"[Cell20.3] Unexpected Cell 20.1 contract version: {cell20_1_version_203}")

q4 = CELL20_1_NONCONTINUOUS_EVIDENCE_CONTRACT.get("q4_governance", {})
strict = CELL20_1_NONCONTINUOUS_EVIDENCE_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell20.3] Cell 20.1 does not carry Q4 blocked_no_promotion governance.")
if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell20.3] Cell 20.1 indicates Q4-coupled artifacts were used.")

q6ctx = CELL20_1_NONCONTINUOUS_EVIDENCE_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6ctx.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6ctx.get("public_distinguishability_status", ""))
public_strict_status = str(q6ctx.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 3) Canonical paths — no-Q4/public first
# ----------------------------------------------------------
scientific_cps_path = _first_existing_203([
    os.path.join(SYN_DIR_203, "CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet"),
    os.path.join(SYN_DIR_203, "CPS_SYNTHETIC_TEST_FINAL_NO_Q4_COUPLED_REMOVED.parquet"),
])
scientific_protocol_path = _first_existing_203([
    os.path.join(SYN_DIR_203, "PROTOCOL_SYN_TEST_FINAL_NO_Q4.parquet"),
])
public_cps_path = os.path.join(SYN_DIR_203, "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet")
public_protocol_path = os.path.join(SYN_DIR_203, "PROTOCOL_SYN_TEST_PUBLIC_Q6_MITIGATED.parquet")
public_iot_path = os.path.join(SYN_DIR_203, "IOT_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet")

claim_manifest_path = os.path.join(REPORT_DIR_203, "claim_traceability_manifest.csv")
readiness_dashboard_path = os.path.join(REPORT_DIR_203, "publication_readiness_dashboard.csv")
publication_blockers_path = os.path.join(REPORT_DIR_203, "publication_blockers.csv")

dimension_summary_path = os.path.join(REPORT_DIR_203, "cell19_1_full_cps_dimension_summary.csv")
role_summary_path = os.path.join(REPORT_DIR_203, "cell19_1_full_cps_role_summary.csv")
metric_inventory_path = os.path.join(REPORT_DIR_203, "cell19_1_full_cps_metric_inventory.csv")
failure_rows_path = os.path.join(REPORT_DIR_203, "cell19_1_full_cps_column_or_pair_failures.csv")

artifact_registry_path = os.path.join(REPORT_DIR_203, "cell19_0_full_cps_eval_artifact_registry.csv")
metric_source_registry_path = os.path.join(REPORT_DIR_203, "cell19_0_full_cps_eval_metric_source_registry.csv")

noncontinuous_evidence_path = os.path.join(REPORT_DIR_203, "cell20_1_noncontinuous_canonical_evidence_table.csv")
noncontinuous_ledger_path = os.path.join(REPORT_DIR_203, "cell20_1_noncontinuous_fix_priority_ledger.csv")
noncontinuous_checklist_path = os.path.join(REPORT_DIR_203, "cell20_1_noncontinuous_fix_checklist.csv")
noncontinuous_reviewer_summary_path = os.path.join(REPORT_DIR_203, "cell20_1_noncontinuous_reviewer_summary.txt")

master_ledger_path = os.path.join(REPORT_DIR_203, "MASTER_RESULTS_LEDGER_FOR_PAPER.csv")
claim_registry_path = os.path.join(REPORT_DIR_203, "cell18_0_claim_registry.csv")

# Q4 report-only evidence.
q4_stage_summary_path = os.path.join(REPORT_DIR_203, "cell16_4_q4_coupling_decomposition_by_stage.csv")
q4_pair_summary_path = os.path.join(REPORT_DIR_203, "cell16_4_q4_coupling_decomposition_by_pair.csv")
q4_decision_scope_path = os.path.join(REPORT_DIR_203, "cell14_11_final_q4_scope_decision_table.csv")

# Q6 reports: prefer v1 re-audit if present.
q6_public_release_safety_path = _first_existing_203([
    os.path.join(PUBLIC_Q6_REPORT_DIRS_203[0], "cell15_4_q6_role_release_safety.csv"),
    os.path.join(PUBLIC_Q6_REPORT_DIRS_203[1], "cell15_4_q6_role_release_safety.csv"),
    os.path.join(REPORT_DIR_203, "cell15_4_q6_role_release_safety.csv"),
])
q6_public_findings_path = _first_existing_203([
    os.path.join(PUBLIC_Q6_REPORT_DIRS_203[0], "cell15_4_q6_privacy_findings.csv"),
    os.path.join(PUBLIC_Q6_REPORT_DIRS_203[1], "cell15_4_q6_privacy_findings.csv"),
])
q6_public_column_registry_path = os.path.join(REPORT_DIR_203, "cell15_6_q6_public_column_registry.csv")
q6_public_dropped_columns_path = os.path.join(REPORT_DIR_203, "cell15_6_q6_public_dropped_columns.csv")

baseline_fairness_summary_path = os.path.join(REPORT_DIR_203, "cell17_6_baseline_publication_safe_summary.csv")
baseline_headline_path = os.path.join(REPORT_DIR_203, "cell17_6_baseline_headline_comparable_results.csv")
baseline_partial_path = os.path.join(REPORT_DIR_203, "cell17_6_baseline_partial_limited_results.csv")

# ----------------------------------------------------------
# 4) Load selected facts
# ----------------------------------------------------------
q6_public_release = _safe_read_csv_203(q6_public_release_safety_path)
claim_manifest = _safe_read_csv_203(claim_manifest_path)
readiness_dashboard = _safe_read_csv_203(readiness_dashboard_path)
publication_blockers = _safe_read_csv_203(publication_blockers_path)
dimension_summary = _safe_read_csv_203(dimension_summary_path)
noncontinuous_evidence = _safe_read_csv_203(noncontinuous_evidence_path)

q6_public_status_counts = _status_counts_203(q6_public_release)
claim_grade_counts = _status_counts_203(claim_manifest, ("decision_grade",))
dashboard_status_counts = _status_counts_203(readiness_dashboard, ("dashboard_status",))
dimension_status_counts = _status_counts_203(dimension_summary, ("status",))

q6_public_blocker_roles = []
q6_public_warning_roles = []
if not q6_public_release.empty:
    role_col = "role_group" if "role_group" in q6_public_release.columns else ("role" if "role" in q6_public_release.columns else None)
    status_col = "release_status" if "release_status" in q6_public_release.columns else None
    if role_col and status_col:
        q6_public_blocker_roles = q6_public_release.loc[
            q6_public_release[status_col].astype(str).str.lower().eq("release_blocker"),
            role_col
        ].astype(str).tolist()
        q6_public_warning_roles = q6_public_release.loc[
            q6_public_release[status_col].astype(str).str.lower().eq("release_warning"),
            role_col
        ].astype(str).tolist()

# ----------------------------------------------------------
# 5) Artifact index
# ----------------------------------------------------------
artifact_items = [
    {
        "artifact_id": "scientific_no_q4_cps",
        "scope": "scientific_internal_reference",
        "path": scientific_cps_path,
        "description": "Primary no-Q4 scientific/internal CPS artifact. Used for controlled analysis and quality evidence.",
        "release_status": "scientific_internal_not_unrestricted_public",
        "reviewer_use": "Scientific/internal quality evidence track.",
        "required": True,
    },
    {
        "artifact_id": "scientific_no_q4_protocol",
        "scope": "scientific_internal_reference",
        "path": scientific_protocol_path,
        "description": "No-Q4 protocol component for scientific/internal analysis.",
        "release_status": "scientific_internal_not_unrestricted_public",
        "reviewer_use": "Protocol component of no-Q4 scientific track.",
        "required": True,
    },
    {
        "artifact_id": "public_cps_candidate",
        "scope": "public_candidate_restricted",
        "path": public_cps_path,
        "description": "Q6-mitigated, role-restricted public CPS candidate. Strict release remains distinguishability-blocked.",
        "release_status": "public_candidate_direct_warning_strict_blocker",
        "reviewer_use": "Public candidate with documented restrictions and caveats.",
        "required": True,
    },
    {
        "artifact_id": "public_protocol_candidate",
        "scope": "public_candidate_restricted",
        "path": public_protocol_path,
        "description": "Protocol subset of Q6-mitigated public candidate.",
        "release_status": "public_candidate_direct_warning_strict_blocker",
        "reviewer_use": "Public protocol subset.",
        "required": True,
    },
    {
        "artifact_id": "public_iot_driver_candidate",
        "scope": "public_candidate_restricted",
        "path": public_iot_path,
        "description": "Sparse IoT event-driver subset of Q6-mitigated public candidate.",
        "release_status": "public_candidate_direct_warning_strict_blocker",
        "reviewer_use": "Public sparse-driver IoT subset.",
        "required": True,
    },

    # Core claim/evidence docs.
    {"artifact_id": "master_results_ledger", "scope": "documentation", "path": master_ledger_path, "description": "Master ledger from which claim registry was generated.", "release_status": "documentation", "reviewer_use": "Trace paper claims to evidence.", "required": True},
    {"artifact_id": "claim_registry", "scope": "documentation", "path": claim_registry_path, "description": "Ledger-derived claim registry.", "release_status": "documentation", "reviewer_use": "Claim audit.", "required": True},
    {"artifact_id": "claim_traceability_manifest", "scope": "documentation", "path": claim_manifest_path, "description": "Final claim traceability manifest.", "release_status": "documentation", "reviewer_use": "Claim verification.", "required": True},
    {"artifact_id": "publication_readiness_dashboard", "scope": "documentation", "path": readiness_dashboard_path, "description": "Claim-scoped dashboard with limitations/caveats.", "release_status": "documentation", "reviewer_use": "Publication readiness evidence.", "required": True},
    {"artifact_id": "publication_blockers_caveats", "scope": "documentation", "path": publication_blockers_path, "description": "Claim caveats/limitations/blockers ledger.", "release_status": "documentation", "reviewer_use": "Caveat verification.", "required": True},

    # 19/20 consolidated evidence.
    {"artifact_id": "full_cps_dimension_summary", "scope": "documentation", "path": dimension_summary_path, "description": "Consolidated dimension-level metric evidence.", "release_status": "documentation", "reviewer_use": "Quality-stratified results.", "required": True},
    {"artifact_id": "full_cps_role_summary", "scope": "documentation", "path": role_summary_path, "description": "Role-level evidence summary.", "release_status": "documentation", "reviewer_use": "Role-level quality evidence.", "required": True},
    {"artifact_id": "full_cps_metric_inventory", "scope": "documentation", "path": metric_inventory_path, "description": "Metric inventory across evidence sources.", "release_status": "documentation", "reviewer_use": "Metric audit.", "required": True},
    {"artifact_id": "full_cps_failures_caveats", "scope": "documentation", "path": failure_rows_path, "description": "Failure/caveat rows from consolidated evidence.", "release_status": "documentation", "reviewer_use": "Limitations audit.", "required": True},
    {"artifact_id": "artifact_registry", "scope": "documentation", "path": artifact_registry_path, "description": "Cell 19.0 artifact registry.", "release_status": "documentation", "reviewer_use": "Artifact existence/shape evidence.", "required": True},
    {"artifact_id": "metric_source_registry", "scope": "documentation", "path": metric_source_registry_path, "description": "Cell 19.0 metric source registry.", "release_status": "documentation", "reviewer_use": "Source availability evidence.", "required": True},
    {"artifact_id": "noncontinuous_canonical_evidence", "scope": "documentation", "path": noncontinuous_evidence_path, "description": "Final non-continuous canonical evidence table.", "release_status": "documentation", "reviewer_use": "Non-continuous evidence.", "required": True},
    {"artifact_id": "noncontinuous_fix_ledger", "scope": "documentation", "path": noncontinuous_ledger_path, "description": "Final reporting/action ledger.", "release_status": "documentation", "reviewer_use": "Remaining action tracking.", "required": True},
    {"artifact_id": "noncontinuous_fix_checklist", "scope": "documentation", "path": noncontinuous_checklist_path, "description": "Final reporting/action checklist.", "release_status": "documentation", "reviewer_use": "Checklist.", "required": True},
    {"artifact_id": "noncontinuous_reviewer_summary", "scope": "documentation", "path": noncontinuous_reviewer_summary_path, "description": "Reviewer-facing non-continuous summary.", "release_status": "documentation", "reviewer_use": "Reviewer summary.", "required": True},

    # Report-only Q4 evidence.
    {"artifact_id": "q4_stage_summary_report_only", "scope": "documentation", "path": q4_stage_summary_path, "description": "Q4 stage evidence. Q4 remains blocked_no_promotion.", "release_status": "documentation", "reviewer_use": "Q4 limitation/governance evidence.", "required": True},
    {"artifact_id": "q4_pair_summary_report_only", "scope": "documentation", "path": q4_pair_summary_path, "description": "Q4 pair-level evidence if available. Report-only, not promoted artifact evidence.", "release_status": "documentation", "reviewer_use": "Pair-level Q4 evidence.", "required": False},
    {"artifact_id": "q4_scope_decision_report_only", "scope": "documentation", "path": q4_decision_scope_path, "description": "Q4 scope decision evidence if available. Report-only.", "release_status": "documentation", "reviewer_use": "Q4 claim-scope verification.", "required": False},

    # Q6/baseline docs.
    {"artifact_id": "q6_public_release_safety", "scope": "documentation", "path": q6_public_release_safety_path, "description": "Public Q6 role safety table.", "release_status": "documentation", "reviewer_use": "Q6 caveat evidence.", "required": False},
    {"artifact_id": "q6_public_findings", "scope": "documentation", "path": q6_public_findings_path, "description": "Public Q6 privacy findings.", "release_status": "documentation", "reviewer_use": "Q6 warning details.", "required": False},
    {"artifact_id": "q6_public_column_registry", "scope": "documentation", "path": q6_public_column_registry_path, "description": "Public artifact column registry.", "release_status": "documentation", "reviewer_use": "Public schema.", "required": False},
    {"artifact_id": "q6_public_dropped_columns", "scope": "documentation", "path": q6_public_dropped_columns_path, "description": "Dropped high-risk columns for public candidate.", "release_status": "documentation", "reviewer_use": "Release mitigation audit.", "required": False},
    {"artifact_id": "baseline_safe_summary", "scope": "documentation", "path": baseline_fairness_summary_path, "description": "Baseline claim sanitizer summary.", "release_status": "documentation", "reviewer_use": "Baseline scope evidence.", "required": False},
    {"artifact_id": "baseline_headline_results", "scope": "documentation", "path": baseline_headline_path, "description": "Headline fair-scope baseline results.", "release_status": "documentation", "reviewer_use": "Baseline evidence.", "required": False},
    {"artifact_id": "baseline_partial_results", "scope": "documentation", "path": baseline_partial_path, "description": "Partial/limited baseline scope results.", "release_status": "documentation", "reviewer_use": "Baseline limitations.", "required": False},
]

rows = []
for item in artifact_items:
    p = str(item["path"])
    rows.append({
        **item,
        "exists": bool(_exists_203(p)),
        "size_bytes": _file_size_203(p),
        "sha256": _sha256_file_203(p),
        "shape_if_parquet": json.dumps(_safe_shape_from_parquet_203(p)) if p.endswith(".parquet") and _exists_203(p) else "",
    })

artifact_index = pd.DataFrame(rows)
missing_required = artifact_index[artifact_index["required"].astype(bool) & (~artifact_index["exists"].astype(bool))].copy()

# ----------------------------------------------------------
# 6) Release scope and claim scope tables
# ----------------------------------------------------------
release_scope = pd.DataFrame([
    {
        "artifact_scope": "scientific_no_q4_artifact",
        "artifact": "CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet",
        "status": "scientific_internal_reference",
        "public_release_status": "not_unrestricted_public_release",
        "reason": "No-Q4 scientific/internal artifact for controlled quality analysis. Not a public-release-safety guarantee.",
        "allowed_use": "Controlled/internal scientific analysis and paper evidence.",
        "forbidden_use": "Do not describe as unrestricted public release or Q4-coupled final artifact.",
    },
    {
        "artifact_scope": "q6_mitigated_public_candidate",
        "artifact": "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet",
        "status": "role_restricted_public_candidate",
        "public_release_status": "direct_warning_strict_distinguishability_blocker",
        "reason": "Role-restricted schema; direct privacy/no-copy warning and strict release blocker from distinguishability.",
        "allowed_use": "Public candidate with documented caveats and release restrictions.",
        "forbidden_use": "Do not describe as privacy-proof, warning-free, or strict-release-ready.",
    },
    {
        "artifact_scope": "q4_evidence",
        "artifact": "reports only",
        "status": "blocked_no_promotion",
        "public_release_status": "not_applicable",
        "reason": "Q4 evidence is report-only. No Q4-coupled artifact is authoritative.",
        "allowed_use": "Report as limitation/governance evidence.",
        "forbidden_use": "Do not claim A0 accepted/promoted or final Q4-coupled artifact authoritative.",
    },
])
release_scope.to_csv(os.path.join(REPORT_DIR_203, "artifact_release_scope_table.csv"), index=False)

claim_scope = pd.DataFrame([
    {
        "claim_area": "scientific_artifact",
        "allowed_claim": "No-Q4 scientific/internal artifact for quality-stratified evaluation.",
        "forbidden_claim": "Final Q4-coupled artifact is authoritative or unrestricted public release-safe.",
        "evidence": "cell19_0_full_cps_eval_artifact_registry.csv; cell20_1_noncontinuous_canonical_evidence_table.csv",
    },
    {
        "claim_area": "public_q6_artifact",
        "allowed_claim": "Role-restricted public candidate with direct privacy warning and strict distinguishability blocker disclosed.",
        "forbidden_claim": "Public artifact is privacy-proof, warning-free, or strict-release-ready.",
        "evidence": "publication_readiness_dashboard.csv; q6_public release reports if available",
    },
    {
        "claim_area": "q4_coupling",
        "allowed_claim": "Q4 remains blocked_no_promotion; evidence is report-only.",
        "forbidden_claim": "A0 accepted/promoted, generic Q4 preserved, or final Q4-coupled artifact authoritative.",
        "evidence": "cell16_4_q4_coupling_decomposition_by_stage.csv; cell19_1_full_cps_dimension_summary.csv",
    },
    {
        "claim_area": "external_baselines",
        "allowed_claim": "Single-run, compatible-scope baseline comparison only.",
        "forbidden_claim": "Global superiority over all baselines or full-CPS replacement comparison.",
        "evidence": "cell17_6_baseline_publication_safe_summary.csv",
    },
    {
        "claim_area": "overall_quality",
        "allowed_claim": "Quality-stratified and claim-scoped evidence with limitations.",
        "forbidden_claim": "Uniform full-artifact realism or global publication-ready status.",
        "evidence": "claim_traceability_manifest.csv; publication_readiness_dashboard.csv; cell19_1_full_cps_dimension_summary.csv",
    },
])
claim_scope.to_csv(os.path.join(REPORT_DIR_203, "artifact_claim_scope_table.csv"), index=False)

# ----------------------------------------------------------
# 7) README
# ----------------------------------------------------------
generated_utc = datetime.now(timezone.utc).isoformat()
scientific_shape = _safe_shape_from_parquet_203(scientific_cps_path)
public_shape = _safe_shape_from_parquet_203(public_cps_path)

readme = f"""# STUDY-THESIS Artifact Package: Claim-Scoped Synthetic Smart-Home CPS Evidence

Generated: {generated_utc}

## 1. Package purpose

This package supports an STUDY-THESIS submission on quality-governed synthetic smart-home CPS data. The package is intentionally **claim-scoped** and **quality-stratified**. It must not be interpreted as a uniformly realistic or unrestricted-public-release-safe full CPS dataset.

## 2. Final governance

The final governance state is:

- `Q4_final_status = blocked_no_promotion`
- `Q4_coupled_artifacts_used = False`
- no Q4-coupled artifact is authoritative or promoted
- Q4 evidence is report-only
- public direct privacy/no-copy status: `{public_direct_privacy_status}`
- public distinguishability status: `{public_distinguishability_status}`
- public strict combined release status: `{public_strict_status}`

## 3. Primary artifacts

| Scope | Artifact | Shape | Release status |
|---|---|---:|---|
| Scientific/internal | `CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet` | {scientific_shape} | Scientific/internal no-Q4 reference; not unrestricted public release |
| Public candidate | `CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet` | {public_shape} | Role-restricted public candidate; direct warning and strict distinguishability blocker disclosed |
| Public protocol | `PROTOCOL_SYN_TEST_PUBLIC_Q6_MITIGATED.parquet` | {_safe_shape_from_parquet_203(public_protocol_path)} | Public protocol subset |
| Public IoT drivers | `IOT_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet` | {_safe_shape_from_parquet_203(public_iot_path)} | Public sparse-driver subset |

## 4. What this package does not claim

This package does **not** claim:

- a final Q4-coupled artifact is authoritative;
- A0/Q4 is accepted or promoted as a final artifact;
- generic all-pair or all-protocol CPS coupling is preserved;
- the full scientific artifact is unrestricted-public-release safe;
- the public artifact is privacy-proof or warning-free;
- all quality dimensions pass uniformly;
- external baselines are beaten globally across the full CPS artifact;
- multi-seed stability, unless a separate multi-seed evidence file is later added.

## 5. Reviewer-facing evidence files

The most important evidence files are:

- `MASTER_RESULTS_LEDGER_FOR_PAPER.csv`
- `cell18_0_claim_registry.csv`
- `claim_traceability_manifest.csv`
- `publication_readiness_dashboard.csv`
- `publication_blockers.csv`
- `cell19_0_full_cps_eval_artifact_registry.csv`
- `cell19_1_full_cps_dimension_summary.csv`
- `cell19_1_full_cps_role_summary.csv`
- `cell19_1_full_cps_metric_inventory.csv`
- `cell20_1_noncontinuous_canonical_evidence_table.csv`
- `cell20_1_noncontinuous_fix_priority_ledger.csv`
- `artifact_package_index.csv`
- `artifact_package_checksums.csv`
- `artifact_release_scope_table.csv`
- `artifact_claim_scope_table.csv`

## 6. Quality interpretation

The final evidence state is quality-stratified:

- all six dashboard dimensions are supported with limitations/caveats;
- Cell 18.2 grades are single-run: Grade B or C, not Grade A;
- Q4 remains blocked-no-promotion and report-only;
- Q6 public release is role-restricted and not strict-release-ready;
- the public artifact is not equivalent to the full scientific artifact.

## 7. Integrity verification

Checksums are provided in `artifact_package_checksums.csv`. Use these hashes to verify artifact integrity.

## 8. Reproduction and leakage discipline

This package is an index/documentation/checksum cell only. It does not generate, fit, select, materialize, or mutate any synthetic values. TEST values are not used here for fitting or repair.
"""

readme_path = os.path.join(REPORT_DIR_203, "ARTIFACT_README_STUDY_THESIS.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

# ----------------------------------------------------------
# 8) Save package index/checksums
# ----------------------------------------------------------
artifact_index_csv = os.path.join(REPORT_DIR_203, "artifact_package_index.csv")
checksum_csv = os.path.join(REPORT_DIR_203, "artifact_package_checksums.csv")
release_scope_csv = os.path.join(REPORT_DIR_203, "artifact_release_scope_table.csv")
claim_scope_csv = os.path.join(REPORT_DIR_203, "artifact_claim_scope_table.csv")
contract_json = os.path.join(REPORT_DIR_203, "cell20_3_artifact_package_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_203, "cell20_3_artifact_package_contract_v3_0_THESIS.json")
manifest_json = os.path.join(ARTDIR_203, "cell20_3_artifact_package_manifest.json")

artifact_index.to_csv(artifact_index_csv, index=False)
artifact_index[[
    "artifact_id", "scope", "path", "sha256", "size_bytes", "exists", "shape_if_parquet", "release_status", "required"
]].to_csv(checksum_csv, index=False)

# ----------------------------------------------------------
# 9) Audits
# ----------------------------------------------------------
audit_rows = []
unsafe_required_q4 = artifact_index[
    artifact_index["required"].astype(bool)
    & artifact_index["path"].astype(str).str.contains("FINAL_Q4|Q4_COUPLED|q4_coupled", case=False, regex=True, na=False)
]
audit_rows.append({
    "audit_check": "no_required_q4_coupled_artifacts",
    "passed": bool(len(unsafe_required_q4) == 0),
    "details": f"unsafe_required_q4={unsafe_required_q4[['artifact_id','path']].to_dict('records') if len(unsafe_required_q4) else []}",
})
audit_rows.append({
    "audit_check": "required_items_exist",
    "passed": bool(len(missing_required) == 0),
    "details": f"missing_required={missing_required[['artifact_id','path']].to_dict('records') if len(missing_required) else []}",
})
audit_df = pd.DataFrame(audit_rows)
audit_passed = bool(audit_df["passed"].astype(bool).all())
if not audit_passed:
    raise RuntimeError(f"[Cell20.3] Artifact package audit failed: {audit_df.loc[~audit_df['passed'].astype(bool)].to_dict('records')}")

# ----------------------------------------------------------
# 10) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "20.3",
    "version": CELL203_VERSION,
    "role": "artifact_package_readme_index_checksum_builder_no_q4_promotion",
    "created_utc": generated_utc,
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_evidence_is_report_only": True,
        "no_q4_coupled_artifact_authoritative": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
        "q6_public_status_counts": q6_public_status_counts,
        "q6_public_blocker_roles": q6_public_blocker_roles,
        "q6_public_warning_roles": q6_public_warning_roles,
    },
    "upstream_contract_versions": {
        "cell20_1": cell20_1_version_203,
    },
    "summary": {
        "artifact_index_rows": int(len(artifact_index)),
        "required_items_n": int(artifact_index["required"].astype(bool).sum()),
        "missing_required_items_n": int(len(missing_required)),
        "audit_passed": bool(audit_passed),
        "scientific_no_q4_shape": scientific_shape,
        "public_cps_shape": public_shape,
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "artifact_package_done_here": True,
    },
    "outputs": {
        "artifact_index_csv": artifact_index_csv,
        "checksum_csv": checksum_csv,
        "release_scope_csv": release_scope_csv,
        "claim_scope_csv": claim_scope_csv,
        "readme_path": readme_path,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}
_write_json_203(contract_json, contract)
_write_json_203(contract_canonical_json, contract)

manifest = {
    "cell": "20.3",
    "version": CELL203_VERSION,
    "created_outputs": contract["outputs"],
    "artifact_index": artifact_index.to_dict("records"),
    "release_scope": release_scope.to_dict("records"),
    "claim_scope": claim_scope.to_dict("records"),
    "audit": audit_df.to_dict("records"),
    "summary": contract["summary"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "strict_contract": contract["strict_contract"],
}
_write_json_203(manifest_json, manifest)

hashes = {
    "artifact_index_csv_sha256": _sha256_file_203(artifact_index_csv),
    "checksum_csv_sha256": _sha256_file_203(checksum_csv),
    "release_scope_csv_sha256": _sha256_file_203(release_scope_csv),
    "claim_scope_csv_sha256": _sha256_file_203(claim_scope_csv),
    "readme_sha256": _sha256_file_203(readme_path),
    "contract_json_sha256": _sha256_file_203(contract_json),
    "contract_canonical_json_sha256": _sha256_file_203(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_203(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_203(contract_json, contract)
_write_json_203(contract_canonical_json, contract)
_write_json_203(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL203_VERSION"] = CELL203_VERSION
globals()["CELL20_3_ARTIFACT_PACKAGE_INDEX_DF"] = artifact_index
globals()["CELL20_3_ARTIFACT_RELEASE_SCOPE_DF"] = release_scope
globals()["CELL20_3_ARTIFACT_CLAIM_SCOPE_DF"] = claim_scope
globals()["CELL20_3_ARTIFACT_PACKAGE_AUDIT_DF"] = audit_df
globals()["CELL20_3_ARTIFACT_PACKAGE_CONTRACT"] = contract

globals()["CELL20_3_ARTIFACT_PACKAGE_INDEX_CSV"] = artifact_index_csv
globals()["CELL20_3_ARTIFACT_PACKAGE_CHECKSUMS_CSV"] = checksum_csv
globals()["CELL20_3_ARTIFACT_RELEASE_SCOPE_CSV"] = release_scope_csv
globals()["CELL20_3_ARTIFACT_CLAIM_SCOPE_CSV"] = claim_scope_csv
globals()["CELL20_3_ARTIFACT_README_MD"] = readme_path
globals()["CELL20_3_ARTIFACT_PACKAGE_CONTRACT_JSON"] = contract_json
globals()["CELL20_3_ARTIFACT_PACKAGE_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL20_3_ARTIFACT_PACKAGE_MANIFEST_JSON"] = manifest_json

log(
    "[Cell20.3] Artifact package README/index complete | "
    f"items={len(artifact_index)} | "
    f"required_missing={len(missing_required)} | "
    f"public_shape={public_shape} | "
    f"scientific_shape={scientific_shape} | "
    "q4_status=blocked_no_promotion"
)
log(f"[Cell20.3] Q6 public status counts | {q6_public_status_counts}")
log(f"[Cell20.3] Saved artifact index: {artifact_index_csv}")
log(f"[Cell20.3] Saved checksums: {checksum_csv}")
log(f"[Cell20.3] Saved README: {readme_path}")
log(f"[Cell20.3] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell20.3] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "artifact_package_done_here=True"
)
log("--- END: Cell 20.3 — STUDY-THESIS artifact package README and index builder (v3.0 no-Q4-promotion strict) ---")