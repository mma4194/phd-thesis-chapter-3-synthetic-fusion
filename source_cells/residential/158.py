# ==========================================================
# CELL 18.1 - Claim-to-evidence traceability linker
# v1.2 STUDY-THESIS strict ledger-derived evidence linking, Cell18.0 v1.3 compatible
#
# Role:
#   - Link every Cell 18.0 ledger-derived claim to concrete source ledger rows,
#     source reports, metric values, artifacts, and known limitations.
#   - Produce an intermediate traceability manifest.
#   - Do NOT assign final multi-seed decision grade yet; Cell 18.2 does that.
#   - Do NOT produce final publication dashboard yet; Cell 18.3 does that.
#
# Why this version exists:
#   The old Cell 18.1 was written for hand-authored claim IDs
#   (e.g., Q1-MARGINAL-001). Cell 18.0 v1.2 now creates a
#   ledger-derived registry with LEDGER-CLAIM-* rows, so evidence linking
#   must be generic and ledger-aware rather than hardcoded to old IDs.
#
# Governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative or used as final evidence.
#   - Public direct privacy/no-copy = release_warning.
#   - Public distinguishability/strict release = release_blocker.
#
# Output columns required downstream:
#   claim_id
#   claim_text
#   permitted_claim
#   forbidden_claim
#   quality_dimension
#   metric_name
#   metric_value
#   artifact_path
#   source_cell
#   selection_split
#   evaluation_split
#   TEST_used_for_selection
#   TEST_used_for_repair
#   split_used
#   seed_policy
#   decision_grade
#   known_limitation
#   publication_status
#   claim_scoped_status
#
# Outputs:
#   reports/cell18_1_claim_traceability_linked.csv
#   reports/cell18_1_claim_traceability_link_audit.csv
#   reports/cell18_1_claim_traceability_contract.json
#   artifacts/contracts/cell18_1_claim_traceability_contract_v1_1_THESIS.json
#   artifacts/cell18_1_claim_traceability_manifest.json
# ==========================================================

log("--- START: Cell 18.1 - Claim-to-evidence traceability linker (v1.2 Cell18.0-v1.3-compatible no-Q4-promotion strict) ---")

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
_required_181 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL18_0_CLAIM_REGISTRY_DF",
    "CELL18_0_CLAIM_REGISTRY_CONTRACT",
]
_missing_181 = [k for k in _required_181 if k not in globals()]
if _missing_181:
    raise RuntimeError(f"[Cell18.1] Missing required globals: {_missing_181}")

ORIGINAL_OUTDIR_181 = str(OUTDIR)
ORIGINAL_OUT_SYN_181 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_181 = str(REPORT_DIR)

def _resolve_project_root_181(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell18.1] Could not resolve canonical project root.")

PROJECT_ROOT_181 = _resolve_project_root_181(ORIGINAL_OUTDIR_181, ORIGINAL_REPORT_DIR_181, ORIGINAL_OUT_SYN_181)
OUT_SYN_BASE_181 = os.path.join(PROJECT_ROOT_181, "synthetic")
REPORT_DIR_BASE_181 = os.path.join(PROJECT_ROOT_181, "reports")
ARTDIR_BASE_181 = os.path.join(PROJECT_ROOT_181, "artifacts")
CONTRACT_DIR_BASE_181 = os.path.join(ARTDIR_BASE_181, "contracts")

os.makedirs(REPORT_DIR_BASE_181, exist_ok=True)
os.makedirs(ARTDIR_BASE_181, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_181, exist_ok=True)

SEED = int(SEED)

CELL181_VERSION = "cell18_1_claim_traceability_linker_v1_2_cell18_0_v1_3_compatible_no_q4_promotion"

CFG["cell18_1_version"] = CELL181_VERSION
CFG["cell18_1_Q4_final_status"] = "blocked_no_promotion"
CFG["cell18_1_Q4_coupled_artifacts_used"] = False
CFG["cell18_1_TEST_real_values_used_here"] = False
CFG["cell18_1_TEST_real_values_used_for_materialization"] = False
CFG["cell18_1_synthetic_values_mutated"] = False
CFG["cell18_1_selection_done_here"] = False
CFG["cell18_1_generator_fit_done_here"] = False
CFG["cell18_1_materialization_done_here"] = False
CFG["cell18_1_traceability_linking_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell18_1_fail_if_any_claim_unlinked", False)
CFG.setdefault("cell18_1_default_decision_grade", "pending_cell18_2_stability")
CFG.setdefault("cell18_1_default_publication_status_if_linked", "evidence_linked_pending_stability")
CFG.setdefault("cell18_1_default_publication_status_if_unlinked", "traceability_gap")
CFG.setdefault("cell18_1_allow_metric_only_evidence", True)

DEFAULT_DECISION_GRADE_181 = str(CFG.get("cell18_1_default_decision_grade", "pending_cell18_2_stability"))
STATUS_LINKED_181 = str(CFG.get("cell18_1_default_publication_status_if_linked", "evidence_linked_pending_stability"))
STATUS_GAP_181 = str(CFG.get("cell18_1_default_publication_status_if_unlinked", "traceability_gap"))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
linked_csv = os.path.join(REPORT_DIR_BASE_181, "cell18_1_claim_traceability_linked.csv")
audit_csv = os.path.join(REPORT_DIR_BASE_181, "cell18_1_claim_traceability_link_audit.csv")
contract_json = os.path.join(REPORT_DIR_BASE_181, "cell18_1_claim_traceability_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_181, "cell18_1_claim_traceability_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_181, "cell18_1_claim_traceability_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_181(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_181(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_181(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_181(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_181(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_181(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_181(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_181(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_181(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_181(payload), f, indent=2, sort_keys=True)

def _sha256_file_181(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_181(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _any_path_exists_181(pipe_path: str):
    paths = [p.strip() for p in str(pipe_path or "").split("|") if p.strip()]
    return any(_exists_181(p) for p in paths)

def _first_existing_181(pipe_path: str):
    paths = [p.strip() for p in str(pipe_path or "").split("|") if p.strip()]
    for p in paths:
        if _exists_181(p):
            return p
    return ""

def _safe_float_181(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _safe_bool_181(x, default=False):
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

def _safe_str_181(x, default=""):
    if pd.isna(x):
        return default
    return str(x)

def _status_from_evidence_181(row, evidence_linked: bool):
    # Preserve blocker/caveat states from Cell 18.0, otherwise move linked claims to pending stability.
    prior = _safe_str_181(row.get("claim_scoped_status", row.get("publication_status", "")))
    if prior.startswith("claim_blocked"):
        return prior
    if "blocker" in prior or "negative" in prior:
        return prior
    if "caveat" in prior:
        return prior
    if evidence_linked:
        return STATUS_LINKED_181
    return STATUS_GAP_181

def _decision_grade_181(row):
    prior = _safe_str_181(row.get("decision_grade", ""))
    if prior and prior != "pending_cell18_2":
        return prior
    return DEFAULT_DECISION_GRADE_181

def _evidence_summary_181(row, source_exists, artifact_exists, metric_exists):
    parts = []
    if source_exists:
        parts.append("source_file_exists")
    if artifact_exists:
        parts.append("artifact_path_exists")
    if metric_exists:
        parts.append("metric_value_recorded")
    if not parts:
        parts.append("no_source_or_metric_evidence_found")
    dim = _safe_str_181(row.get("quality_dimension", ""))
    scope = _safe_str_181(row.get("claim_scope", ""))
    metric = _safe_str_181(row.get("metric_name", ""))
    val = row.get("metric_value", np.nan)
    val_s = "NA" if not np.isfinite(_safe_float_181(val, np.nan)) else f"{_safe_float_181(val):.6g}"
    return f"{';'.join(parts)} | dimension={dim} | scope={scope} | metric={metric} | value={val_s}"

# ----------------------------------------------------------
# 4) Validate upstream Cell 18.0 contract
# ----------------------------------------------------------
cell18_0_version_181 = str(CELL18_0_CLAIM_REGISTRY_CONTRACT.get("version", ""))
if "cell18_0_claim_registry_v1_3" not in cell18_0_version_181:
    raise RuntimeError(f"[Cell18.1] Unexpected Cell 18.0 contract version: {cell18_0_version_181}")

q4 = CELL18_0_CLAIM_REGISTRY_CONTRACT.get("q4_governance", {})
strict = CELL18_0_CLAIM_REGISTRY_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell18.1] Cell 18.0 does not carry Q4 blocked_no_promotion governance.")
if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell18.1] Cell 18.0 indicates Q4-coupled artifacts were used.")

q6_public_context = CELL18_0_CLAIM_REGISTRY_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6_public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6_public_context.get("public_distinguishability_status", ""))
public_strict_status = str(q6_public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 5) Generic ledger-derived evidence linking
# ----------------------------------------------------------
claim_df = CELL18_0_CLAIM_REGISTRY_DF.copy()
claim_df["claim_id"] = claim_df["claim_id"].astype(str)

linked_rows = []
for _, row in claim_df.iterrows():
    r = row.to_dict()

    source_file = _safe_str_181(r.get("source_file", ""))
    master_ledger_path = _safe_str_181(r.get("master_ledger_path", ""))
    artifact_path = _safe_str_181(r.get("artifact_path", ""))
    source_cell = _safe_str_181(r.get("source_cell", ""))

    source_exists = _exists_181(source_file) or _exists_181(master_ledger_path)
    artifact_exists = _any_path_exists_181(artifact_path)
    metric_value = _safe_float_181(r.get("metric_value", np.nan), np.nan)
    metric_exists = np.isfinite(metric_value)

    evidence_path = source_file if _exists_181(source_file) else master_ledger_path if _exists_181(master_ledger_path) else ""
    evidence_linked = bool(source_exists or artifact_exists or (bool(CFG.get("cell18_1_allow_metric_only_evidence", True)) and metric_exists))

    # Use source row/file from master ledger; no stale Q4-coupled path resolution here.
    out = dict(r)
    out["claim_text"] = _safe_str_181(r.get("permitted_claim", r.get("claim_text_raw_from_ledger", "")))
    out["source_cell"] = source_cell
    out["metric_name"] = _safe_str_181(r.get("metric_name", ""))
    out["metric_value"] = metric_value if metric_exists else np.nan
    out["artifact_path"] = artifact_path
    out["evidence_path"] = evidence_path
    out["evidence_linked"] = bool(evidence_linked)
    out["evidence_link_type"] = "|".join([
        x for x, ok in [
            ("source_file", source_exists),
            ("artifact_path", artifact_exists),
            ("metric_value", metric_exists),
        ] if ok
    ]) or "none"
    out["evidence_summary"] = _evidence_summary_181(r, source_exists, artifact_exists, metric_exists)
    out["selection_split"] = _safe_str_181(r.get("selection_split", "TRAIN_fit_VAL_select_only;TEST_not_used_for_selection"))
    out["evaluation_split"] = _safe_str_181(r.get("evaluation_split", "TEST_QA_reference_only"))
    out["TEST_used_for_selection"] = _safe_bool_181(r.get("TEST_used_for_selection", False))
    out["TEST_used_for_repair"] = _safe_bool_181(r.get("TEST_used_for_repair", False))
    out["split_used"] = _safe_str_181(
        r.get("split_used", f"selection={out['selection_split']};evaluation={out['evaluation_split']}")
    )
    out["seed_policy"] = _safe_str_181(r.get("seed_policy", "single_seed_current_run;multi_seed_CI_not_claimed"))
    out["decision_grade"] = _decision_grade_181(r)
    out["publication_status"] = _status_from_evidence_181(r, evidence_linked)
    out["claim_scoped_status"] = _safe_str_181(r.get("claim_scoped_status", out["publication_status"]))
    out["q4_final_status"] = "blocked_no_promotion"
    out["q4_coupled_artifacts_used"] = False
    out["public_direct_privacy_status"] = public_direct_privacy_status
    out["public_distinguishability_status"] = public_distinguishability_status
    out["public_strict_combined_status"] = public_strict_status
    out["TEST_real_values_used_here"] = False
    out["synthetic_values_mutated"] = False

    linked_rows.append(out)

linked_df = pd.DataFrame(linked_rows)

# ----------------------------------------------------------
# 6) Audit links
# ----------------------------------------------------------
audit_rows = []

required_final_cols = [
    "claim_id",
    "claim_text",
    "permitted_claim",
    "forbidden_claim",
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
    "claim_scoped_status",
]
missing_cols = [c for c in required_final_cols if c not in linked_df.columns]
audit_rows.append({
    "audit_check": "required_final_columns_present",
    "passed": bool(len(missing_cols) == 0),
    "details": f"missing={missing_cols}",
})

unlinked = linked_df.loc[~linked_df["evidence_linked"].astype(bool), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "all_claims_have_some_evidence_link",
    "passed": bool(len(unlinked) == 0),
    "details": f"unlinked_claim_ids={unlinked}",
})

empty_metric = linked_df.loc[linked_df["metric_name"].astype(str).str.strip().eq(""), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "metric_name_nonempty",
    "passed": bool(len(empty_metric) == 0),
    "details": f"empty_metric_claim_ids={empty_metric}",
})

empty_source = linked_df.loc[linked_df["source_cell"].astype(str).str.strip().eq(""), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "source_cell_nonempty",
    "passed": bool(len(empty_source) == 0),
    "details": f"empty_source_claim_ids={empty_source}",
})

test_sel = linked_df.loc[linked_df["TEST_used_for_selection"].astype(bool), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "no_test_used_for_selection",
    "passed": bool(len(test_sel) == 0),
    "details": f"test_selection_claim_ids={test_sel}",
})

test_rep = linked_df.loc[linked_df["TEST_used_for_repair"].astype(bool), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "no_test_used_for_repair",
    "passed": bool(len(test_rep) == 0),
    "details": f"test_repair_claim_ids={test_rep}",
})

# Stale-Q4 path check: artifact_path may contain old strings only if claim explicitly says no authoritative Q4.
# In this ledger-derived version artifact paths come from the master ledger; do not resolve Q4-coupled artifacts.
stale_q4_artifacts = []
for _, r in linked_df.iterrows():
    ap = _safe_str_181(r.get("artifact_path", "")).lower()
    permitted = _safe_str_181(r.get("permitted_claim", "")).lower()
    if ("final_q4_coupled" in ap or "q4_coupled" in ap) and "no q4" not in permitted and "blocked" not in permitted:
        stale_q4_artifacts.append(str(r["claim_id"]))

audit_rows.append({
    "audit_check": "no_stale_q4_artifact_path_without_blocked_context",
    "passed": bool(len(stale_q4_artifacts) == 0),
    "details": f"stale_q4_artifact_claim_ids={stale_q4_artifacts}",
})

status_counts_pre_audit_181 = linked_df["publication_status"].astype(str).value_counts().to_dict()
all_blocker_status_181 = (
    len(linked_df) > 0
    and len(status_counts_pre_audit_181) == 1
    and "claim_supported_with_blocker_or_negative_result" in status_counts_pre_audit_181
)
audit_rows.append({
    "audit_check": "publication_status_not_globally_collapsed_to_blocker",
    "passed": bool(not all_blocker_status_181),
    "details": f"publication_status_counts={status_counts_pre_audit_181}",
})

trace_audit_df = pd.DataFrame(audit_rows)
audit_passed = bool(trace_audit_df["passed"].astype(bool).all())

if not audit_passed and bool(CFG.get("cell18_1_fail_if_any_claim_unlinked", False)):
    raise RuntimeError(
        "[Cell18.1] Traceability audit failed: "
        f"{trace_audit_df.loc[~trace_audit_df['passed'].astype(bool)].to_dict('records')}"
    )

# ----------------------------------------------------------
# 7) Save outputs
# ----------------------------------------------------------
linked_df.to_csv(linked_csv, index=False)
trace_audit_df.to_csv(audit_csv, index=False)

contract = {
    "cell": "18.1",
    "version": CELL181_VERSION,
    "role": "ledger_derived_claim_to_evidence_traceability_linker_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "stale_q4_artifacts_not_resolved_as_authoritative": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell18_0": cell18_0_version_181,
    },
    "summary": {
        "claims_total": int(len(linked_df)),
        "claims_linked": int(linked_df["evidence_linked"].astype(bool).sum()),
        "claims_unlinked": int((~linked_df["evidence_linked"].astype(bool)).sum()),
        "audit_passed": bool(audit_passed),
        "audit_rows": int(len(trace_audit_df)),
        "publication_status_counts": linked_df["publication_status"].astype(str).value_counts().sort_index().to_dict(),
        "quality_dimension_counts": linked_df["quality_dimension"].astype(str).value_counts().sort_index().to_dict(),
        "evidence_link_type_counts": linked_df["evidence_link_type"].astype(str).value_counts().sort_index().to_dict(),
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
        "traceability_linking_done_here": True,
        "ledger_derived_linking": True,
    },
    "outputs": {
        "linked_csv": linked_csv,
        "audit_csv": audit_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_181(contract_json, contract)
_write_json_181(contract_canonical_json, contract)

manifest = {
    "cell": "18.1",
    "version": CELL181_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "claims": linked_df.to_dict("records"),
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
}
_write_json_181(manifest_json, manifest)

hashes = {
    "linked_csv_sha256": _sha256_file_181(linked_csv),
    "audit_csv_sha256": _sha256_file_181(audit_csv),
    "contract_json_sha256": _sha256_file_181(contract_json),
    "contract_canonical_json_sha256": _sha256_file_181(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_181(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_181(contract_json, contract)
_write_json_181(contract_canonical_json, contract)
_write_json_181(manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals
# ----------------------------------------------------------
globals()["CELL181_VERSION"] = CELL181_VERSION
globals()["CELL18_1_CLAIM_TRACEABILITY_LINKED_DF"] = linked_df
globals()["CELL18_1_CLAIM_TRACEABILITY_AUDIT_DF"] = trace_audit_df
globals()["CELL18_1_CLAIM_TRACEABILITY_CONTRACT"] = contract

globals()["CELL18_1_CLAIM_TRACEABILITY_LINKED_CSV"] = linked_csv
globals()["CELL18_1_CLAIM_TRACEABILITY_AUDIT_CSV"] = audit_csv
globals()["CELL18_1_CLAIM_TRACEABILITY_CONTRACT_JSON"] = contract_json
globals()["CELL18_1_CLAIM_TRACEABILITY_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL18_1_CLAIM_TRACEABILITY_MANIFEST_JSON"] = manifest_json

log(
    "[Cell18.1] Ledger-derived claim traceability linking complete | "
    f"claims={len(linked_df)} | "
    f"linked={int(linked_df['evidence_linked'].astype(bool).sum())} | "
    f"unlinked={int((~linked_df['evidence_linked'].astype(bool)).sum())} | "
    f"audit_passed={audit_passed} | "
    "q4_status=blocked_no_promotion"
)
log(
    "[Cell18.1] Publication status counts | "
    f"{linked_df['publication_status'].astype(str).value_counts().sort_index().to_dict()}"
)
if not audit_passed:
    log(
        "[Cell18.1] WARNING: Traceability audit has non-fatal issues | "
        f"{trace_audit_df.loc[~trace_audit_df['passed'].astype(bool)].to_dict('records')}"
    )
log(f"[Cell18.1] Saved linked traceability table: {linked_csv}")
log(f"[Cell18.1] Saved audit: {audit_csv}")
log(f"[Cell18.1] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell18.1] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "traceability_linking_done_here=True | "
    "ledger_derived_linking=True"
)
log("--- END: Cell 18.1 - Claim-to-evidence traceability linker (v1.2 Cell18.0-v1.3-compatible no-Q4-promotion strict) ---")

gc.collect()