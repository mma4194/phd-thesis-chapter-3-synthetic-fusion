# ==========================================================
# CELL 18.2 - Multi-seed stability and decision-grade labels
# v1.1 STUDY-THESIS strict ledger-derived decision grading, no-Q4-promotion aware
#
# Role:
#   - Assign decision-grade labels to each linked ledger-derived claim.
#   - Use available multi-seed evidence when present.
#   - If no multi-seed evidence exists, explicitly label empirical claims as
#     single-run evidence rather than pretending stability was proven.
#   - Convert evidence-linked claims into claim-scoped status candidates.
#
# Important governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative.
#   - Public direct privacy/no-copy = release_warning.
#   - Public distinguishability/strict release = release_blocker.
#   - Do not use global "publication_ready"; statuses are claim-scoped.
#
# Decision grades:
#   A = traceable deterministic/contractual claim or multi-seed-stable empirical claim.
#   B = traceable single-run empirical claim with caveat.
#   C = traceable limitation, boundary, negative result, blocker, or partial-scope claim.
#   D = blocked/not defensible: leakage, missing evidence, overclaim, or unstable claim.
#
# Outputs:
#   reports/cell18_2_claim_decision_grade.csv
#   reports/cell18_2_claim_stability_audit.csv
#   reports/cell18_2_claim_decision_contract.json
#   artifacts/contracts/cell18_2_claim_decision_contract_v1_1_THESIS.json
#   artifacts/cell18_2_claim_decision_manifest.json
# ==========================================================

log("--- START: Cell 18.2 - Claim stability and decision-grade labels (v1.1 ledger-derived no-Q4-promotion strict) ---")

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
_required_182 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL18_1_CLAIM_TRACEABILITY_LINKED_DF",
    "CELL18_1_CLAIM_TRACEABILITY_CONTRACT",
]
_missing_182 = [k for k in _required_182 if k not in globals()]
if _missing_182:
    raise RuntimeError(f"[Cell18.2] Missing required globals: {_missing_182}")

ORIGINAL_OUTDIR_182 = str(OUTDIR)
ORIGINAL_OUT_SYN_182 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_182 = str(REPORT_DIR)

def _resolve_project_root_182(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell18.2] Could not resolve canonical project root.")

PROJECT_ROOT_182 = _resolve_project_root_182(ORIGINAL_OUTDIR_182, ORIGINAL_REPORT_DIR_182, ORIGINAL_OUT_SYN_182)
REPORT_DIR_BASE_182 = os.path.join(PROJECT_ROOT_182, "reports")
ARTDIR_BASE_182 = os.path.join(PROJECT_ROOT_182, "artifacts")
CONTRACT_DIR_BASE_182 = os.path.join(ARTDIR_BASE_182, "contracts")

os.makedirs(REPORT_DIR_BASE_182, exist_ok=True)
os.makedirs(ARTDIR_BASE_182, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_182, exist_ok=True)

SEED = int(SEED)

CELL182_VERSION = "cell18_2_claim_stability_decision_grade_v1_1_ledger_no_q4_promotion"

CFG["cell18_2_version"] = CELL182_VERSION
CFG["cell18_2_Q4_final_status"] = "blocked_no_promotion"
CFG["cell18_2_Q4_coupled_artifacts_used"] = False
CFG["cell18_2_TEST_real_values_used_here"] = False
CFG["cell18_2_TEST_real_values_used_for_materialization"] = False
CFG["cell18_2_synthetic_values_mutated"] = False
CFG["cell18_2_selection_done_here"] = False
CFG["cell18_2_generator_fit_done_here"] = False
CFG["cell18_2_materialization_done_here"] = False
CFG["cell18_2_decision_grading_done_here"] = True
CFG["cell18_2_global_publication_ready_used"] = False

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell18_2_multiseed_manifest_candidates", [
    os.path.join(REPORT_DIR_BASE_182, "multi_seed_stability_summary.csv"),
    os.path.join(REPORT_DIR_BASE_182, "cell_multi_seed_stability_summary.csv"),
    os.path.join(REPORT_DIR_BASE_182, "cell18_2_external_multiseed_stability.csv"),
])
CFG.setdefault("cell18_2_require_multiseed_for_grade_A_empirical", True)
CFG.setdefault("cell18_2_empirical_single_run_max_grade", "B")
CFG.setdefault("cell18_2_contractual_single_run_max_grade", "A")
CFG.setdefault("cell18_2_limitation_claim_grade", "C")
CFG.setdefault("cell18_2_negative_or_blocker_claim_grade", "C")
CFG.setdefault("cell18_2_traceability_gap_grade", "D")
CFG.setdefault("cell18_2_high_overclaim_without_limitation_grade", "D")

EMPIRICAL_SINGLE_RUN_MAX_182 = str(CFG.get("cell18_2_empirical_single_run_max_grade", "B")).upper()
CONTRACTUAL_SINGLE_RUN_MAX_182 = str(CFG.get("cell18_2_contractual_single_run_max_grade", "A")).upper()
LIMITATION_GRADE_182 = str(CFG.get("cell18_2_limitation_claim_grade", "C")).upper()
NEGATIVE_GRADE_182 = str(CFG.get("cell18_2_negative_or_blocker_claim_grade", "C")).upper()
TRACE_GAP_GRADE_182 = str(CFG.get("cell18_2_traceability_gap_grade", "D")).upper()

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
decision_csv = os.path.join(REPORT_DIR_BASE_182, "cell18_2_claim_decision_grade.csv")
stability_audit_csv = os.path.join(REPORT_DIR_BASE_182, "cell18_2_claim_stability_audit.csv")
contract_json = os.path.join(REPORT_DIR_BASE_182, "cell18_2_claim_decision_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_182, "cell18_2_claim_decision_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_182, "cell18_2_claim_decision_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_182(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_182(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_182(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_182(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_182(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_182(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_182(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_182(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_182(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_182(payload), f, indent=2, sort_keys=True)

def _sha256_file_182(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_182(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _read_csv_optional_182(path: str):
    if not _exists_182(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _safe_float_182(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _safe_bool_182(x, default=False):
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

def _grade_rank_182(g):
    return {"A": 4, "B": 3, "C": 2, "D": 1}.get(str(g).upper(), 0)

def _min_grade_182(*grades):
    valid = [str(g).upper() for g in grades if str(g).upper() in {"A", "B", "C", "D"}]
    if not valid:
        return "D"
    return sorted(valid, key=lambda g: _grade_rank_182(g))[0]

def _has_limitation_182(row):
    return bool(str(row.get("known_limitation", "")).strip())

def _claim_text_blob_182(row):
    return " ".join([
        str(row.get("claim_text", "")),
        str(row.get("permitted_claim", "")),
        str(row.get("forbidden_claim", "")),
        str(row.get("known_limitation", "")),
        str(row.get("claim_scope", "")),
        str(row.get("claim_type", "")),
        str(row.get("quality_dimension", "")),
        str(row.get("claim_scoped_status", "")),
        str(row.get("publication_status", "")),
    ]).lower()

def _is_empirical_182(row):
    typ = str(row.get("claim_type", "")).lower()
    dim = str(row.get("quality_dimension", "")).lower()
    return (
        "empirical" in typ
        or dim in {"q1_marginal", "q2_temporal"}
        or ("metric_value" in row and np.isfinite(_safe_float_182(row.get("metric_value"), np.nan)))
    )

def _is_contractual_182(row):
    typ = str(row.get("claim_type", "")).lower()
    blob = _claim_text_blob_182(row)
    return (
        "contract" in typ
        or "artifact" in typ
        or "method" in typ
        or "scope_boundary" in typ
        or "baseline_unavailable" in typ
        or "fairness" in typ
        or "deterministic" in blob
    )

def _is_limitation_or_boundary_182(row):
    typ = str(row.get("claim_type", "")).lower()
    blob = _claim_text_blob_182(row)
    dim = str(row.get("quality_dimension", "")).lower()
    return (
        "limitation" in typ
        or "negative" in typ
        or "boundary" in typ
        or "partial" in typ
        or "limited" in blob
        or "blocker" in blob
        or dim in {"q3_observability", "q4_coupling", "q6_privacy_release"}
    )

def _metric_or_evidence_valid_182(row):
    if not _safe_bool_182(row.get("evidence_linked", False)):
        return False
    metric_name = str(row.get("metric_name", "")).strip()
    if not metric_name:
        return False
    metric_value = _safe_float_182(row.get("metric_value"), np.nan)
    if np.isfinite(metric_value):
        return True
    if str(row.get("artifact_path", "")).strip():
        return True
    if str(row.get("evidence_path", "")).strip():
        return True
    return False

def _status_is_blocked_182(row):
    s = str(row.get("claim_scoped_status", row.get("publication_status", ""))).lower()
    return s.startswith("claim_blocked")

def _status_has_caveat_182(row):
    s = str(row.get("claim_scoped_status", row.get("publication_status", ""))).lower()
    return "caveat" in s or "blocker" in s or "negative" in s

def _stability_lookup_182(multiseed_df, row):
    if not isinstance(multiseed_df, pd.DataFrame) or len(multiseed_df) == 0:
        return {}
    keys = [
        ("claim_id", row.get("claim_id", "")),
        ("metric_name", row.get("metric_name", "")),
        ("quality_dimension", row.get("quality_dimension", "")),
        ("claim_scope", row.get("claim_scope", "")),
    ]
    for col, val in keys:
        if col in multiseed_df.columns and str(val).strip():
            d = multiseed_df[multiseed_df[col].astype(str).eq(str(val))]
            if len(d):
                return d.iloc[0].to_dict()
    return {}

def _stability_label_182(row, multiseed_row):
    if multiseed_row:
        if "stability_status" in multiseed_row:
            st = str(multiseed_row.get("stability_status"))
            if st:
                return st
        if "stable" in multiseed_row:
            return "multi_seed_stable" if _safe_bool_182(multiseed_row.get("stable")) else "multi_seed_unstable"
        if "cv" in multiseed_row:
            cv = _safe_float_182(multiseed_row.get("cv"), np.nan)
            if np.isfinite(cv):
                return "multi_seed_stable" if cv <= 0.10 else "multi_seed_variable"

    if _is_contractual_182(row):
        return "deterministic_or_contractual_single_run"
    if _is_limitation_or_boundary_182(row):
        return "single_run_limitation_boundary_or_negative"
    if _is_empirical_182(row):
        return "single_run_empirical"
    return "single_run_claim"

def _decision_grade_182(row, stability_label):
    if _safe_bool_182(row.get("TEST_used_for_selection", False)) or _safe_bool_182(row.get("TEST_used_for_repair", False)):
        return "D", "test_used_for_selection_or_repair"

    if _status_is_blocked_182(row):
        return "D", "claim_blocked_by_registry"

    if not _metric_or_evidence_valid_182(row):
        return TRACE_GAP_GRADE_182, "traceability_or_metric_gap"

    overclaim_risk = str(row.get("overclaim_risk", "")).lower()
    if overclaim_risk == "high" and not _has_limitation_182(row):
        return str(CFG.get("cell18_2_high_overclaim_without_limitation_grade", "D")).upper(), "high_overclaim_risk_without_limitation"

    if stability_label in {"multi_seed_unstable", "unstable"}:
        return "D", "multi_seed_unstable"

    if stability_label == "multi_seed_variable":
        return "B", "multi_seed_variable"

    if _is_limitation_or_boundary_182(row):
        return LIMITATION_GRADE_182, "limitation_boundary_blocker_or_negative_result"

    if _is_contractual_182(row):
        return CONTRACTUAL_SINGLE_RUN_MAX_182, "contractual_or_scope_claim_traceable"

    if _is_empirical_182(row):
        if stability_label == "multi_seed_stable":
            return "A", "multi_seed_stable_empirical_claim"
        return EMPIRICAL_SINGLE_RUN_MAX_182, "single_run_empirical_grade_cap_no_multiseed_CI"

    return "B", "default_traceable_claim"

def _claim_scoped_publication_status_182(row, grade):
    if grade == "D":
        return "claim_blocked_or_needs_revision"
    if grade == "A":
        return "claim_supported_decision_grade_A"
    if grade == "B":
        return "claim_supported_decision_grade_B_with_caveat"
    if grade == "C":
        return "claim_supported_decision_grade_C_limitation_or_boundary"
    return "claim_needs_review"

# ----------------------------------------------------------
# 4) Validate upstream Cell 18.1 contract
# ----------------------------------------------------------
cell18_1_version_182 = str(CELL18_1_CLAIM_TRACEABILITY_CONTRACT.get("version", ""))
if "cell18_1_claim_traceability_linker_v1_2" not in cell18_1_version_182:
    raise RuntimeError(f"[Cell18.2] Unexpected Cell 18.1 contract version: {cell18_1_version_182}")

q4 = CELL18_1_CLAIM_TRACEABILITY_CONTRACT.get("q4_governance", {})
strict = CELL18_1_CLAIM_TRACEABILITY_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell18.2] Cell 18.1 does not carry Q4 blocked_no_promotion governance.")
if _safe_bool_182(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell18.2] Cell 18.1 indicates Q4-coupled artifacts were used.")

q6_public_context = CELL18_1_CLAIM_TRACEABILITY_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6_public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6_public_context.get("public_distinguishability_status", ""))
public_strict_status = str(q6_public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 5) Load optional multi-seed evidence
# ----------------------------------------------------------
multiseed_paths = [str(p) for p in CFG.get("cell18_2_multiseed_manifest_candidates", [])]
loaded_multiseed = []
for p in multiseed_paths:
    d = _read_csv_optional_182(p)
    if isinstance(d, pd.DataFrame):
        loaded_multiseed.append((p, d))

if loaded_multiseed:
    multiseed_path, multiseed_df = loaded_multiseed[0]
else:
    multiseed_path, multiseed_df = "", pd.DataFrame()

# ----------------------------------------------------------
# 6) Grade claims
# ----------------------------------------------------------
linked_df = CELL18_1_CLAIM_TRACEABILITY_LINKED_DF.copy()
linked_df["claim_id"] = linked_df["claim_id"].astype(str)

graded_rows = []
stability_rows = []

for _, r in linked_df.iterrows():
    row = r.to_dict()
    cid = str(row["claim_id"])

    ms_row = _stability_lookup_182(multiseed_df, row)
    stability_label = _stability_label_182(row, ms_row)
    grade, grade_reason = _decision_grade_182(row, stability_label)
    claim_status = _claim_scoped_publication_status_182(row, grade)

    metric_valid = _metric_or_evidence_valid_182(row)

    out = dict(row)
    out["stability_label"] = stability_label
    out["multi_seed_evidence_available"] = bool(ms_row)
    out["multi_seed_evidence_path"] = multiseed_path
    out["decision_grade"] = grade
    out["decision_grade_reason"] = grade_reason
    out["publication_status"] = claim_status
    out["claim_scoped_status_after_stability"] = claim_status
    out["metric_evidence_valid"] = bool(metric_valid)
    out["Q4_final_status"] = "blocked_no_promotion"
    out["Q4_coupled_artifacts_used"] = False
    out["public_direct_privacy_status"] = public_direct_privacy_status
    out["public_distinguishability_status"] = public_distinguishability_status
    out["public_strict_combined_status"] = public_strict_status
    out["TEST_real_values_used_here"] = False
    out["synthetic_values_mutated"] = False

    graded_rows.append(out)

    stability_rows.append({
        "claim_id": cid,
        "claim_type": str(row.get("claim_type", "")),
        "quality_dimension": str(row.get("quality_dimension", "")),
        "claim_scope": str(row.get("claim_scope", "")),
        "metric_name": str(row.get("metric_name", "")),
        "metric_value": row.get("metric_value", np.nan),
        "multi_seed_evidence_available": bool(ms_row),
        "multi_seed_evidence_path": multiseed_path,
        "stability_label": stability_label,
        "decision_grade": grade,
        "decision_grade_reason": grade_reason,
        "publication_status": claim_status,
        "metric_evidence_valid": bool(metric_valid),
        "known_limitation": str(row.get("known_limitation", "")),
        "TEST_real_values_used_here": False,
        "synthetic_values_mutated": False,
    })

graded_df = pd.DataFrame(graded_rows)
stability_audit_df = pd.DataFrame(stability_rows)

# ----------------------------------------------------------
# 7) Audit
# ----------------------------------------------------------
audit_rows = []

required_final_cols = [
    "claim_id",
    "claim_text",
    "quality_dimension",
    "metric_name",
    "metric_value",
    "artifact_path",
    "source_cell",
    "selection_split",
    "evaluation_split",
    "TEST_used_for_selection",
    "TEST_used_for_repair",
    "split_used",
    "seed_policy",
    "decision_grade",
    "known_limitation",
    "publication_status",
    "claim_scoped_status_after_stability",
]
missing_cols = [c for c in required_final_cols if c not in graded_df.columns]
audit_rows.append({
    "audit_check": "required_final_columns_present",
    "passed": bool(len(missing_cols) == 0),
    "details": f"missing={missing_cols}",
})

invalid_grades = graded_df.loc[
    ~graded_df["decision_grade"].astype(str).isin(["A", "B", "C", "D"]),
    "claim_id",
].astype(str).tolist()
audit_rows.append({
    "audit_check": "decision_grades_valid",
    "passed": bool(len(invalid_grades) == 0),
    "details": f"invalid_grade_claims={invalid_grades}",
})

# No global publication_ready labels.
global_ready = graded_df.loc[
    graded_df["publication_status"].astype(str).str.contains("publication_ready", case=False, na=False),
    "claim_id",
].astype(str).tolist()
audit_rows.append({
    "audit_check": "no_global_publication_ready_status",
    "passed": bool(len(global_ready) == 0),
    "details": f"global_publication_ready_claim_ids={global_ready}",
})

metric_invalid = graded_df.loc[
    ~graded_df["metric_evidence_valid"].astype(bool),
    "claim_id",
].astype(str).tolist()
audit_rows.append({
    "audit_check": "metric_evidence_valid",
    "passed": bool(len(metric_invalid) == 0),
    "details": f"invalid_metric_evidence_claims={metric_invalid}",
})

test_sel = graded_df.loc[graded_df["TEST_used_for_selection"].astype(bool), "claim_id"].astype(str).tolist()
test_rep = graded_df.loc[graded_df["TEST_used_for_repair"].astype(bool), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "no_test_selection_or_repair",
    "passed": bool(len(test_sel) == 0 and len(test_rep) == 0),
    "details": f"test_selection_claims={test_sel}; test_repair_claims={test_rep}",
})

empirical_single_run = graded_df.loc[
    graded_df["stability_label"].astype(str).eq("single_run_empirical"),
    "claim_id",
].astype(str).tolist()
audit_rows.append({
    "audit_check": "single_run_empirical_claims_identified",
    "passed": True,
    "details": f"single_run_empirical_claims={empirical_single_run}",
})

multi_seed_available_n = int(graded_df["multi_seed_evidence_available"].astype(bool).sum())
audit_rows.append({
    "audit_check": "multi_seed_evidence_presence",
    "passed": True,
    "details": f"multi_seed_evidence_available_n={multi_seed_available_n}; source={multiseed_path or 'none'}",
})

decision_audit_df = pd.DataFrame(audit_rows)
audit_passed = bool(decision_audit_df["passed"].astype(bool).all())

if not audit_passed:
    raise RuntimeError(
        "[Cell18.2] Claim decision-grade audit failed: "
        f"{decision_audit_df.loc[~decision_audit_df['passed'].astype(bool)].to_dict('records')}"
    )

# ----------------------------------------------------------
# 8) Save outputs
# ----------------------------------------------------------
graded_df.to_csv(decision_csv, index=False)
stability_audit_df.to_csv(stability_audit_csv, index=False)

contract = {
    "cell": "18.2",
    "version": CELL182_VERSION,
    "role": "ledger_derived_claim_stability_and_decision_grading_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_promotion_claims_not_reenabled": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell18_1": cell18_1_version_182,
    },
    "summary": {
        "claims_total": int(len(graded_df)),
        "decision_grade_counts": graded_df["decision_grade"].astype(str).value_counts().sort_index().to_dict(),
        "publication_status_counts": graded_df["publication_status"].astype(str).value_counts().sort_index().to_dict(),
        "stability_label_counts": graded_df["stability_label"].astype(str).value_counts().sort_index().to_dict(),
        "multi_seed_evidence_available_n": int(multi_seed_available_n),
        "multi_seed_evidence_path": multiseed_path,
        "audit_passed": bool(audit_passed),
    },
    "grade_policy": {
        "A": "traceable deterministic/contractual claim or multi-seed-stable empirical claim",
        "B": "traceable single-run empirical claim with caveat",
        "C": "traceable limitation, boundary, negative result, blocker, or partial-scope claim",
        "D": "blocked/not defensible: leakage, missing evidence, overclaim, or unstable claim",
        "empirical_single_run_max_grade": EMPIRICAL_SINGLE_RUN_MAX_182,
        "contractual_single_run_max_grade": CONTRACTUAL_SINGLE_RUN_MAX_182,
        "global_publication_ready_used": False,
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
        "decision_grading_done_here": True,
        "global_publication_ready_used": False,
    },
    "outputs": {
        "decision_csv": decision_csv,
        "stability_audit_csv": stability_audit_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_182(contract_json, contract)
_write_json_182(contract_canonical_json, contract)

manifest = {
    "cell": "18.2",
    "version": CELL182_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "claims": graded_df.to_dict("records"),
    "summary": contract["summary"],
    "grade_policy": contract["grade_policy"],
    "strict_contract": contract["strict_contract"],
}

_write_json_182(manifest_json, manifest)

hashes = {
    "decision_csv_sha256": _sha256_file_182(decision_csv),
    "stability_audit_csv_sha256": _sha256_file_182(stability_audit_csv),
    "contract_json_sha256": _sha256_file_182(contract_json),
    "contract_canonical_json_sha256": _sha256_file_182(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_182(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_182(contract_json, contract)
_write_json_182(contract_canonical_json, contract)
_write_json_182(manifest_json, manifest)

# ----------------------------------------------------------
# 9) Export globals
# ----------------------------------------------------------
globals()["CELL182_VERSION"] = CELL182_VERSION
globals()["CELL18_2_CLAIM_DECISION_GRADE_DF"] = graded_df
globals()["CELL18_2_CLAIM_STABILITY_AUDIT_DF"] = stability_audit_df
globals()["CELL18_2_CLAIM_DECISION_AUDIT_DF"] = decision_audit_df
globals()["CELL18_2_CLAIM_DECISION_CONTRACT"] = contract

globals()["CELL18_2_CLAIM_DECISION_GRADE_CSV"] = decision_csv
globals()["CELL18_2_CLAIM_STABILITY_AUDIT_CSV"] = stability_audit_csv
globals()["CELL18_2_CLAIM_DECISION_CONTRACT_JSON"] = contract_json
globals()["CELL18_2_CLAIM_DECISION_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL18_2_CLAIM_DECISION_MANIFEST_JSON"] = manifest_json

log(
    "[Cell18.2] Ledger-derived claim decision grading complete | "
    f"claims={len(graded_df)} | "
    f"grades={graded_df['decision_grade'].astype(str).value_counts().sort_index().to_dict()} | "
    f"statuses={graded_df['publication_status'].astype(str).value_counts().sort_index().to_dict()} | "
    "q4_status=blocked_no_promotion"
)
log(
    "[Cell18.2] Stability labels | "
    f"{graded_df['stability_label'].astype(str).value_counts().sort_index().to_dict()} | "
    f"multi_seed_evidence_available_n={multi_seed_available_n}"
)
if multi_seed_available_n == 0:
    log(
        "[Cell18.2] NOTE: No multi-seed evidence file found. "
        "Empirical claims are capped as single-run evidence rather than marked multi-seed stable."
    )
log(f"[Cell18.2] Saved decision grades: {decision_csv}")
log(f"[Cell18.2] Saved stability audit: {stability_audit_csv}")
log(f"[Cell18.2] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell18.2] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "decision_grading_done_here=True | "
    "global_publication_ready_used=False"
)
log("--- END: Cell 18.2 - Claim stability and decision-grade labels (v1.1 ledger-derived no-Q4-promotion strict) ---")

gc.collect()