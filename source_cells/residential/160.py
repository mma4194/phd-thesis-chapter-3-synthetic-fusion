# ==========================================================
# CELL 18.3 - Final STUDY-THESIS publication readiness dashboard
# v1.2 STUDY-THESIS strict claim-scoped dashboard, safe-forbidden-wording audit, no-Q4-promotion aware
#
# Role:
#   - Convert Cell 18.2 decision-graded claims into final paper-facing
#     traceability outputs.
#   - Generate final claim traceability manifest and dashboard.
#   - Generate publication blockers / caveats ledger.
#
# Checklist action implemented:
#   - Output-only dashboard, not hand-authored optimism.
#   - Generated from ledger-derived Cell 18.0/18.1/18.2 outputs.
#   - Keeps selection_split, evaluation_split, TEST_used_for_selection,
#     TEST_used_for_repair, permitted_claim, forbidden_claim.
#   - Does not use global "publication_ready"; uses claim-scoped status.
#
# Governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative.
#   - Public direct privacy/no-copy = release_warning.
#   - Public distinguishability/strict release = release_blocker.
#   - Multi-seed evidence absent unless Cell 18.4 is run separately.
#
# Outputs:
#   reports/claim_traceability_manifest.csv
#   reports/claim_traceability_manifest.json
#   reports/publication_readiness_dashboard.csv
#   reports/publication_blockers.csv
#   reports/cell18_3_publication_dashboard_contract.json
#   artifacts/contracts/cell18_3_publication_dashboard_contract_v1_1_THESIS.json
#   artifacts/cell18_3_publication_dashboard_manifest.json
# ==========================================================

log("--- START: Cell 18.3 - Final claim-scoped publication readiness dashboard (v1.2 safe-forbidden-wording no-Q4-promotion strict) ---")

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
_required_183 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL18_2_CLAIM_DECISION_GRADE_DF",
    "CELL18_2_CLAIM_DECISION_CONTRACT",
]
_missing_183 = [k for k in _required_183 if k not in globals()]
if _missing_183:
    raise RuntimeError(f"[Cell18.3] Missing required globals: {_missing_183}")

ORIGINAL_OUTDIR_183 = str(OUTDIR)
ORIGINAL_OUT_SYN_183 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_183 = str(REPORT_DIR)

def _resolve_project_root_183(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell18.3] Could not resolve canonical project root.")

PROJECT_ROOT_183 = _resolve_project_root_183(ORIGINAL_OUTDIR_183, ORIGINAL_REPORT_DIR_183, ORIGINAL_OUT_SYN_183)
REPORT_DIR_BASE_183 = os.path.join(PROJECT_ROOT_183, "reports")
ARTDIR_BASE_183 = os.path.join(PROJECT_ROOT_183, "artifacts")
CONTRACT_DIR_BASE_183 = os.path.join(ARTDIR_BASE_183, "contracts")

os.makedirs(REPORT_DIR_BASE_183, exist_ok=True)
os.makedirs(ARTDIR_BASE_183, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_183, exist_ok=True)

SEED = int(SEED)

CELL183_VERSION = "cell18_3_publication_readiness_dashboard_v1_2_safe_forbidden_wording_no_q4_promotion"

CFG["cell18_3_version"] = CELL183_VERSION
CFG["cell18_3_Q4_final_status"] = "blocked_no_promotion"
CFG["cell18_3_Q4_coupled_artifacts_used"] = False
CFG["cell18_3_TEST_real_values_used_here"] = False
CFG["cell18_3_TEST_real_values_used_for_materialization"] = False
CFG["cell18_3_synthetic_values_mutated"] = False
CFG["cell18_3_selection_done_here"] = False
CFG["cell18_3_generator_fit_done_here"] = False
CFG["cell18_3_materialization_done_here"] = False
CFG["cell18_3_dashboard_done_here"] = True
CFG["cell18_3_global_publication_ready_used"] = False

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
claim_traceability_manifest_csv = os.path.join(REPORT_DIR_BASE_183, "claim_traceability_manifest.csv")
claim_traceability_manifest_json = os.path.join(REPORT_DIR_BASE_183, "claim_traceability_manifest.json")
publication_readiness_dashboard_csv = os.path.join(REPORT_DIR_BASE_183, "publication_readiness_dashboard.csv")
publication_blockers_csv = os.path.join(REPORT_DIR_BASE_183, "publication_blockers.csv")
contract_json = os.path.join(REPORT_DIR_BASE_183, "cell18_3_publication_dashboard_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_183, "cell18_3_publication_dashboard_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_183, "cell18_3_publication_dashboard_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_183(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_183(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_183(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_183(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_183(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_183(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_183(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_183(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_183(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_183(payload), f, indent=2, sort_keys=True)

def _sha256_file_183(path):
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_bool_183(x, default=False):
    if isinstance(x, bool):
        return bool(x)
    if pd.isna(x):
        return bool(default)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", ""}:
        return False
    return bool(default)

def _safe_str_183(x, default=""):
    if pd.isna(x):
        return default
    return str(x)

def _grade_rank_183(g):
    return {"A": 4, "B": 3, "C": 2, "D": 1}.get(str(g).upper(), 0)

def _claim_final_status_183(row):
    grade = str(row.get("decision_grade", "")).upper()
    status = str(row.get("publication_status", "")).lower()
    dim = str(row.get("quality_dimension", "")).lower()

    if _safe_bool_183(row.get("TEST_used_for_selection", False)) or _safe_bool_183(row.get("TEST_used_for_repair", False)):
        return "blocked_leakage"

    if grade == "D" or "blocked_or_needs_revision" in status:
        return "blocked_or_needs_revision"

    if grade == "A":
        return "claim_supported_A"

    if grade == "B":
        return "claim_supported_B_with_caveat"

    if grade == "C":
        return "claim_supported_C_limitation_or_boundary"

    return "needs_review"

def _dashboard_status_183(group):
    statuses = group["claim_final_status"].astype(str).tolist()
    if any(s.startswith("blocked") for s in statuses):
        return "dimension_has_blocked_claims"
    if any("needs_review" in s for s in statuses):
        return "dimension_needs_review"
    if any("C_limitation" in s for s in statuses):
        return "dimension_supported_with_limitations"
    if any("B_with_caveat" in s for s in statuses):
        return "dimension_supported_with_caveats"
    if all("claim_supported_A" in s for s in statuses) and statuses:
        return "dimension_supported_A"
    return "dimension_supported_claim_scoped"

def _dashboard_recommendation_183(status):
    if status == "dimension_has_blocked_claims":
        return "Revise or remove blocked claims before paper submission."
    if status == "dimension_needs_review":
        return "Manual review required before claim export."
    if status == "dimension_supported_with_limitations":
        return "Report as limitation/boundary; do not overclaim."
    if status == "dimension_supported_with_caveats":
        return "Report with explicit caveat and single-run/no-multiseed limitation where applicable."
    if status == "dimension_supported_A":
        return "Can be reported as traceable claim-scoped support."
    return "Can be reported only within recorded claim scope."

# ----------------------------------------------------------
# 3) Validate upstream Cell 18.2 contract
# ----------------------------------------------------------
cell18_2_version_183 = str(CELL18_2_CLAIM_DECISION_CONTRACT.get("version", ""))
if "cell18_2_claim_stability_decision_grade_v1_1" not in cell18_2_version_183:
    raise RuntimeError(f"[Cell18.3] Unexpected Cell 18.2 contract version: {cell18_2_version_183}")

q4 = CELL18_2_CLAIM_DECISION_CONTRACT.get("q4_governance", {})
strict = CELL18_2_CLAIM_DECISION_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell18.3] Cell 18.2 does not carry Q4 blocked_no_promotion governance.")
if _safe_bool_183(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell18.3] Cell 18.2 indicates Q4-coupled artifacts were used.")
if _safe_bool_183(strict.get("global_publication_ready_used", False)):
    raise RuntimeError("[Cell18.3] Cell 18.2 used global publication_ready status, which is forbidden.")

q6_public_context = CELL18_2_CLAIM_DECISION_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6_public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6_public_context.get("public_distinguishability_status", ""))
public_strict_status = str(q6_public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 4) Build final claim traceability manifest
# ----------------------------------------------------------
claims = CELL18_2_CLAIM_DECISION_GRADE_DF.copy()
claims["claim_id"] = claims["claim_id"].astype(str)

claims["claim_final_status"] = claims.apply(_claim_final_status_183, axis=1)
claims["q4_final_status"] = "blocked_no_promotion"
claims["q4_coupled_artifacts_used"] = False
claims["public_direct_privacy_status"] = public_direct_privacy_status
claims["public_distinguishability_status"] = public_distinguishability_status
claims["public_strict_combined_status"] = public_strict_status
claims["global_publication_ready_used"] = False
claims["TEST_real_values_used_here"] = False
claims["synthetic_values_mutated"] = False

# Ensure required columns exist in final manifest.
required_manifest_cols = [
    "claim_id",
    "claim_text",
    "permitted_claim",
    "forbidden_claim",
    "quality_dimension",
    "claim_scope",
    "claim_type",
    "metric_name",
    "metric_value",
    "artifact_path",
    "evidence_path",
    "source_cell",
    "selection_split",
    "evaluation_split",
    "TEST_used_for_selection",
    "TEST_used_for_repair",
    "split_used",
    "seed_policy",
    "stability_label",
    "multi_seed_evidence_available",
    "decision_grade",
    "decision_grade_reason",
    "known_limitation",
    "publication_status",
    "claim_final_status",
    "claim_scoped_status_after_stability",
    "q4_final_status",
    "q4_coupled_artifacts_used",
    "public_direct_privacy_status",
    "public_distinguishability_status",
    "public_strict_combined_status",
    "global_publication_ready_used",
]
for c in required_manifest_cols:
    if c not in claims.columns:
        claims[c] = ""

manifest_df = claims[required_manifest_cols].copy()

# ----------------------------------------------------------
# 5) Dashboard by quality dimension
# ----------------------------------------------------------
dash_rows = []
for dim, g in manifest_df.groupby("quality_dimension", dropna=False):
    grade_counts = g["decision_grade"].astype(str).value_counts().to_dict()
    final_status_counts = g["claim_final_status"].astype(str).value_counts().to_dict()
    status = _dashboard_status_183(g)

    dash_rows.append({
        "quality_dimension": str(dim),
        "claims_n": int(len(g)),
        "grade_A_n": int(grade_counts.get("A", 0)),
        "grade_B_n": int(grade_counts.get("B", 0)),
        "grade_C_n": int(grade_counts.get("C", 0)),
        "grade_D_n": int(grade_counts.get("D", 0)),
        "blocked_claims_n": int(sum(1 for s in g["claim_final_status"].astype(str) if s.startswith("blocked"))),
        "needs_review_n": int((g["claim_final_status"].astype(str) == "needs_review").sum()),
        "single_run_claims_n": int(g["stability_label"].astype(str).str.contains("single_run", na=False).sum()),
        "multi_seed_supported_claims_n": int(g["multi_seed_evidence_available"].astype(bool).sum()),
        "dashboard_status": status,
        "recommendation": _dashboard_recommendation_183(status),
        "claim_final_status_counts_json": json.dumps(final_status_counts, sort_keys=True),
        "q4_final_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "global_publication_ready_used": False,
        "safe_forbidden_publication_ready_wording_allowed": True,
    })

dashboard_df = pd.DataFrame(dash_rows).sort_values("quality_dimension").reset_index(drop=True)

# ----------------------------------------------------------
# 6) Blockers/caveats ledger
# ----------------------------------------------------------
blocker_mask = (
    manifest_df["claim_final_status"].astype(str).str.startswith("blocked")
    | manifest_df["claim_final_status"].astype(str).eq("needs_review")
    | manifest_df["decision_grade"].astype(str).eq("D")
)
caveat_mask = (
    manifest_df["decision_grade"].astype(str).isin(["B", "C"])
    | manifest_df["claim_final_status"].astype(str).str.contains("caveat|limitation|boundary", case=False, na=False)
)

blockers_df = manifest_df[blocker_mask | caveat_mask].copy()
blockers_df["ledger_type"] = np.where(blocker_mask.loc[blockers_df.index], "blocker_or_review", "caveat_or_limitation")
blockers_df["paper_action"] = np.where(
    blockers_df["ledger_type"].eq("blocker_or_review"),
    "Revise/remove claim before final paper export.",
    "Keep claim only with recorded caveat/boundary wording.",
)

# ----------------------------------------------------------
# 7) Audits
# ----------------------------------------------------------
audit_rows = []

missing_manifest_cols = [c for c in required_manifest_cols if c not in manifest_df.columns]
audit_rows.append({
    "audit_check": "required_manifest_columns_present",
    "passed": bool(len(missing_manifest_cols) == 0),
    "details": f"missing={missing_manifest_cols}",
})

# Audit only positive/global readiness assertions, not safe forbidden/caveat wording.
# Safe examples allowed:
#   - forbidden_claim: "Do not use global publication_ready"
#   - global_publication_ready_used = False
#   - contract text saying no global publication_ready is used
def _unsafe_global_ready_row_183(row):
    status_fields = [
        "publication_status",
        "claim_final_status",
        "claim_scoped_status_after_stability",
        "dashboard_status",
        "dimension_readiness_status",
    ]
    for c in status_fields:
        if c in row.index:
            s = str(row.get(c, "")).strip().lower()
            if s in {"publication_ready", "global_publication_ready", "ready_for_publication"}:
                return True

    positive_fields = ["claim_text", "permitted_claim"]
    for c in positive_fields:
        if c in row.index:
            s = str(row.get(c, "")).lower()
            # Do not flag explicit negations/caveats.
            safe = (
                "do not" in s
                or "not " in s
                or "no " in s
                or "forbidden" in s
                or "global_publication_ready_used=false" in s
                or "blocked" in s
                or "caveat" in s
                or "warning" in s
            )
            if "publication_ready" in s and not safe:
                return True
    return False

global_ready_mask = manifest_df.apply(_unsafe_global_ready_row_183, axis=1)
global_ready_rows = manifest_df[global_ready_mask]
audit_rows.append({
    "audit_check": "no_positive_global_publication_ready_claim",
    "passed": bool(len(global_ready_rows) == 0),
    "details": f"global_publication_ready_claim_ids={global_ready_rows['claim_id'].astype(str).tolist() if len(global_ready_rows) else []}",
})

q4_bad = manifest_df[
    manifest_df["q4_coupled_artifacts_used"].astype(str).str.lower().isin(["true", "1", "yes"])
]
audit_rows.append({
    "audit_check": "no_q4_coupled_artifact_usage",
    "passed": bool(len(q4_bad) == 0),
    "details": f"q4_bad_claim_ids={q4_bad['claim_id'].astype(str).tolist() if len(q4_bad) else []}",
})

test_sel = manifest_df[manifest_df["TEST_used_for_selection"].astype(str).str.lower().isin(["true", "1", "yes"])]
test_rep = manifest_df[manifest_df["TEST_used_for_repair"].astype(str).str.lower().isin(["true", "1", "yes"])]
audit_rows.append({
    "audit_check": "no_test_selection_or_repair_claims",
    "passed": bool(len(test_sel) == 0 and len(test_rep) == 0),
    "details": f"test_selection={test_sel['claim_id'].astype(str).tolist() if len(test_sel) else []}; test_repair={test_rep['claim_id'].astype(str).tolist() if len(test_rep) else []}",
})

audit_df = pd.DataFrame(audit_rows)
audit_passed = bool(audit_df["passed"].astype(bool).all())
if not audit_passed:
    raise RuntimeError(
        "[Cell18.3] Publication dashboard audit failed: "
        f"{audit_df.loc[~audit_df['passed'].astype(bool)].to_dict('records')}"
    )

# ----------------------------------------------------------
# 8) Save outputs
# ----------------------------------------------------------
manifest_df.to_csv(claim_traceability_manifest_csv, index=False)
dashboard_df.to_csv(publication_readiness_dashboard_csv, index=False)
blockers_df.to_csv(publication_blockers_csv, index=False)

claim_trace_json_payload = {
    "cell": "18.3",
    "version": CELL183_VERSION,
    "claims": manifest_df.to_dict("records"),
    "dashboard": dashboard_df.to_dict("records"),
    "blockers_and_caveats": blockers_df.to_dict("records"),
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
}
_write_json_183(claim_traceability_manifest_json, claim_trace_json_payload)

contract = {
    "cell": "18.3",
    "version": CELL183_VERSION,
    "role": "final_claim_scoped_publication_readiness_dashboard_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_coupled_artifact_not_authoritative": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell18_2": cell18_2_version_183,
    },
    "summary": {
        "claims_total": int(len(manifest_df)),
        "dashboard_rows": int(len(dashboard_df)),
        "blockers_or_caveats_rows": int(len(blockers_df)),
        "decision_grade_counts": manifest_df["decision_grade"].astype(str).value_counts().sort_index().to_dict(),
        "claim_final_status_counts": manifest_df["claim_final_status"].astype(str).value_counts().sort_index().to_dict(),
        "dashboard_status_counts": dashboard_df["dashboard_status"].astype(str).value_counts().sort_index().to_dict(),
        "audit_passed": bool(audit_passed),
        "global_publication_ready_used": False,
        "safe_forbidden_publication_ready_wording_allowed": True,
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
        "dashboard_done_here": True,
        "global_publication_ready_used": False,
    },
    "outputs": {
        "claim_traceability_manifest_csv": claim_traceability_manifest_csv,
        "claim_traceability_manifest_json": claim_traceability_manifest_json,
        "publication_readiness_dashboard_csv": publication_readiness_dashboard_csv,
        "publication_blockers_csv": publication_blockers_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_183(contract_json, contract)
_write_json_183(contract_canonical_json, contract)

cell_manifest = {
    "cell": "18.3",
    "version": CELL183_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
}
_write_json_183(manifest_json, cell_manifest)

hashes = {
    "claim_traceability_manifest_csv_sha256": _sha256_file_183(claim_traceability_manifest_csv),
    "claim_traceability_manifest_json_sha256": _sha256_file_183(claim_traceability_manifest_json),
    "publication_readiness_dashboard_csv_sha256": _sha256_file_183(publication_readiness_dashboard_csv),
    "publication_blockers_csv_sha256": _sha256_file_183(publication_blockers_csv),
    "contract_json_sha256": _sha256_file_183(contract_json),
    "contract_canonical_json_sha256": _sha256_file_183(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_183(manifest_json),
}
contract["hashes"] = hashes
cell_manifest["hashes"] = hashes
_write_json_183(contract_json, contract)
_write_json_183(contract_canonical_json, contract)
_write_json_183(manifest_json, cell_manifest)

# ----------------------------------------------------------
# 9) Export globals
# ----------------------------------------------------------
globals()["CELL183_VERSION"] = CELL183_VERSION
globals()["CELL18_3_CLAIM_TRACEABILITY_MANIFEST_DF"] = manifest_df
globals()["CELL18_3_PUBLICATION_READINESS_DASHBOARD_DF"] = dashboard_df
globals()["CELL18_3_PUBLICATION_BLOCKERS_DF"] = blockers_df
globals()["CELL18_3_PUBLICATION_DASHBOARD_AUDIT_DF"] = audit_df
globals()["CELL18_3_PUBLICATION_DASHBOARD_CONTRACT"] = contract

globals()["CELL18_FINAL_CLAIM_TRACEABILITY_MANIFEST_CSV"] = claim_traceability_manifest_csv
globals()["CELL18_FINAL_CLAIM_TRACEABILITY_MANIFEST_JSON"] = claim_traceability_manifest_json
globals()["CELL18_FINAL_PUBLICATION_READINESS_DASHBOARD_CSV"] = publication_readiness_dashboard_csv
globals()["CELL18_FINAL_PUBLICATION_BLOCKERS_CSV"] = publication_blockers_csv
globals()["CELL18_3_PUBLICATION_DASHBOARD_CONTRACT_JSON"] = contract_json
globals()["CELL18_3_PUBLICATION_DASHBOARD_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL18_3_PUBLICATION_DASHBOARD_MANIFEST_JSON"] = manifest_json

log(
    "[Cell18.3] Final claim-scoped publication dashboard complete | "
    f"claims={len(manifest_df)} | "
    f"dashboard_rows={len(dashboard_df)} | "
    f"blockers_or_caveats={len(blockers_df)} | "
    f"grades={manifest_df['decision_grade'].astype(str).value_counts().sort_index().to_dict()} | "
    "q4_status=blocked_no_promotion"
)
log(
    "[Cell18.3] Dashboard statuses | "
    f"{dashboard_df['dashboard_status'].astype(str).value_counts().sort_index().to_dict()}"
)
log(f"[Cell18.3] Saved final claim traceability manifest: {claim_traceability_manifest_csv}")
log(f"[Cell18.3] Saved final publication dashboard: {publication_readiness_dashboard_csv}")
log(f"[Cell18.3] Saved publication blockers/caveats: {publication_blockers_csv}")
log(f"[Cell18.3] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell18.3] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "dashboard_done_here=True | "
    "global_publication_ready_used=False | safe_forbidden_publication_ready_wording_allowed=True"
)
log("--- END: Cell 18.3 - Final claim-scoped publication readiness dashboard (v1.2 safe-forbidden-wording no-Q4-promotion strict) ---")

gc.collect()