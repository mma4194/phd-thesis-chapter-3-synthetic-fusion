# ==========================================================
# CELL 14.10 - Final Q4 no-promotion decision ledger
# v1.1 STUDY-THESIS strict final coupling governance ledger
#
# Role:
#   - Record the final Q4 cross-modal coupling decision after:
#       Cell 13.4 broad-Q4 blocker summary
#       Cell 14.6 A0 candidate rejection
#       Cell 14.6b policy-legality diagnostic
#       Cell 14.7 no-promotion governance
#       Cell 14.8 A1/router audit-only review
#       Cell 14.9 A1a deferral
#   - Produce a defensible final audit trail for paper/reporting.
#
# Scientific contract:
#   - No fitting.
#   - No materialization.
#   - No synthetic mutation.
#   - No TEST real values used here.
#   - No artifact promotion/copy.
#   - Final Q4 status remains blocked_no_promotion.
#
# Outputs:
#   reports/cell14_10_final_q4_decision_ledger.csv
#   reports/cell14_10_final_q4_candidate_comparison.csv
#   reports/cell14_10_final_q4_publication_statement.txt
#   reports/cell14_10_final_q4_decision_contract.json
#   artifacts/contracts/cell14_10_final_q4_decision_contract_v1_1_THESIS.json
#   artifacts/cell14_10_final_q4_decision_manifest.json
# ==========================================================

log("--- START: Cell 14.10 - Final Q4 no-promotion decision ledger (v1.1 strict) ---")

import os
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_1410 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "CELL14_7_Q4_NO_PROMOTION_CONTRACT",
    "CELL14_8_A1_ROUTER_AUDIT_CONTRACT",
    "CELL14_9_A1A_DEFERRAL_CONTRACT",
]
_missing_1410 = [k for k in _required_1410 if k not in globals()]
if _missing_1410:
    raise RuntimeError(f"[Cell14.10] Missing required globals: {_missing_1410}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_1410 = str(OUT_SYN)
REPORT_DIR_ACTIVE_1410 = str(REPORT_DIR)

def _resolve_project_root_1410(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell14.10] Could not resolve canonical project root.")

PROJECT_ROOT_1410 = _resolve_project_root_1410(OUTDIR, REPORT_DIR_ACTIVE_1410, OUT_SYN_ACTIVE_1410)
OUT_SYN_1410 = os.path.join(PROJECT_ROOT_1410, "synthetic")
REPORT_DIR_1410 = os.path.join(PROJECT_ROOT_1410, "reports")
ARTDIR_1410 = os.path.join(PROJECT_ROOT_1410, "artifacts")
CONTRACT_DIR_1410 = os.path.join(ARTDIR_1410, "contracts")

os.makedirs(REPORT_DIR_1410, exist_ok=True)
os.makedirs(ARTDIR_1410, exist_ok=True)
os.makedirs(CONTRACT_DIR_1410, exist_ok=True)

SEED = int(SEED)
CELL1410_VERSION = "cell14_10_final_q4_no_promotion_decision_ledger_v1_1"

CFG["cell14_10_version"] = CELL1410_VERSION
CFG["cell14_10_final_q4_status"] = "blocked_no_promotion"
CFG["cell14_10_TEST_real_values_used"] = False
CFG["cell14_10_synthetic_values_mutated"] = False
CFG["cell14_10_selection_done_here"] = False
CFG["cell14_10_generator_fit_done_here"] = False
CFG["cell14_10_materialization_done_here"] = False
CFG["cell14_10_artifact_copy_done_here"] = False
CFG["cell14_10_promotion_done_here"] = False
CFG["cell14_10_decision_ledger_done_here"] = True

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
decision_ledger_csv = os.path.join(REPORT_DIR_1410, "cell14_10_final_q4_decision_ledger.csv")
candidate_comparison_csv = os.path.join(REPORT_DIR_1410, "cell14_10_final_q4_candidate_comparison.csv")
publication_statement_txt = os.path.join(REPORT_DIR_1410, "cell14_10_final_q4_publication_statement.txt")
decision_contract_json = os.path.join(REPORT_DIR_1410, "cell14_10_final_q4_decision_contract.json")
decision_contract_canonical_json = os.path.join(CONTRACT_DIR_1410, "cell14_10_final_q4_decision_contract_v1_1_THESIS.json")
decision_manifest_json = os.path.join(ARTDIR_1410, "cell14_10_final_q4_decision_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_1410(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_1410(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_1410(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_1410(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_1410(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_1410(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_1410(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_1410(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_1410(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_1410(payload), f, indent=2, sort_keys=True)

def _sha256_file_1410(path: str) -> str:
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_1410(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_int_1410(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _require_contract_version_1410(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.10] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.10] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

# ----------------------------------------------------------
# 3) Validate upstream final governance
# ----------------------------------------------------------
version134_1410 = _require_contract_version_1410(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)
version147_1410 = _require_contract_version_1410(
    CELL14_7_Q4_NO_PROMOTION_CONTRACT,
    "CELL14_7_Q4_NO_PROMOTION_CONTRACT",
    "cell14_7_q4_no_promotion_governance_v2_1",
)
version148_1410 = _require_contract_version_1410(
    CELL14_8_A1_ROUTER_AUDIT_CONTRACT,
    "CELL14_8_A1_ROUTER_AUDIT_CONTRACT",
    "cell14_8_A1_router_protocol_consistency_audit_v1_1",
)
version149_1410 = _require_contract_version_1410(
    CELL14_9_A1A_DEFERRAL_CONTRACT,
    "CELL14_9_A1A_DEFERRAL_CONTRACT",
    "cell14_9_A1a_expansion_deferral_v1_1",
)

strict147_1410 = CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("strict_contract", {})
if bool(CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("accepted", True)):
    raise RuntimeError("[Cell14.10] Cell 14.7 governance contract unexpectedly has accepted=True.")
if bool(CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("promoted", True)):
    raise RuntimeError("[Cell14.10] Cell 14.7 governance contract unexpectedly has promoted=True.")
if bool(strict147_1410.get("candidate_outputs_promoted_to_final", True)):
    raise RuntimeError("[Cell14.10] Cell 14.7 governance says candidate outputs were promoted.")
if bool(strict147_1410.get("synthetic_values_mutated_here", True)):
    raise RuntimeError("[Cell14.10] Cell 14.7 governance indicates synthetic mutation.")

strict148_1410 = CELL14_8_A1_ROUTER_AUDIT_CONTRACT.get("strict_contract", {})
if bool(strict148_1410.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.10] Cell 14.8 audit indicates TEST real values were used.")
if bool(strict148_1410.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.10] Cell 14.8 audit indicates synthetic mutation.")
if bool(strict148_1410.get("direct_repair_authorized_here", True)):
    raise RuntimeError("[Cell14.10] Cell 14.8 unexpectedly authorized direct repair.")

strict149_1410 = CELL14_9_A1A_DEFERRAL_CONTRACT.get("strict_contract", {})
if not bool(strict149_1410.get("A1a_deferred", False)):
    raise RuntimeError("[Cell14.10] Cell 14.9 did not declare A1a_deferred.")
if bool(strict149_1410.get("materialization_done_here", True)):
    raise RuntimeError("[Cell14.10] Cell 14.9 indicates materialization was done.")
if bool(strict149_1410.get("promotion_done_here", True)):
    raise RuntimeError("[Cell14.10] Cell 14.9 indicates promotion was done.")
if bool(strict149_1410.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.10] Cell 14.9 indicates synthetic mutation.")

q4_134_summary = CELL13_4_Q4_PUBLICATION_SUMMARY
q4_147_summary = CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("summary", {})
q4_governance = q4_147_summary.get("governance_decision", {})
audit_148_counts = CELL14_8_A1_ROUTER_AUDIT_CONTRACT.get("audit_counts", {})
a1a_149_summary = CELL14_9_A1A_DEFERRAL_CONTRACT.get("summary", {})

final_q4_status = str(q4_governance.get("final_q4_status", ""))
if final_q4_status != "blocked_no_promotion":
    raise RuntimeError(f"[Cell14.10] Expected blocked_no_promotion, got {final_q4_status}")

# ----------------------------------------------------------
# 4) Candidate comparison
# ----------------------------------------------------------
candidate_rows = [
    {
        "candidate_name": "broad_schema_driven_Q4",
        "source_cell": "13.4",
        "decision": "blocked",
        "status": str(q4_134_summary.get("overall_q4_status", "blocker")),
        "reason": "Broad 624-pair Q4 failed under ETA, lag/response, and cross-correlation QA.",
        "pairs_total": _safe_int_1410(q4_134_summary.get("pair_summary", {}).get("pairs_total", np.nan), -1),
        "q4_pass_n": _safe_int_1410(q4_134_summary.get("pair_summary", {}).get("pair_pass_n", np.nan), -1),
        "q4_warning_n": _safe_int_1410(q4_134_summary.get("pair_summary", {}).get("pair_warning_n", np.nan), -1),
        "q4_fatal_or_blocker_n": _safe_int_1410(q4_134_summary.get("pair_summary", {}).get("pair_blocker_n", np.nan), -1),
        "q4_publication_blocker_n": _safe_int_1410(q4_134_summary.get("pair_summary", {}).get("pair_blocker_n", np.nan), -1),
        "promoted": False,
        "artifact_protocol": "",
        "artifact_cps": "",
    },
    {
        "candidate_name": "A0_zigbee_safe_candidate",
        "source_cell": "14.6/14.7",
        "decision": "blocked_no_promotion",
        "status": "blocked",
        "reason": (
            "A0 improved coupling but failed strict promotion gate; 14.6b found no defensible simple TRAIN/VAL proxy "
            "for pruning fatal rows without TEST-outcome cherry-picking."
        ),
        "pairs_total": _safe_int_1410(q4_147_summary.get("pairs_total", 0), 0),
        "q4_pass_n": _safe_int_1410(q4_147_summary.get("q4_pass_n", 0), 0),
        "q4_warning_n": _safe_int_1410(q4_147_summary.get("q4_warning_n", 0), 0),
        "q4_fatal_or_blocker_n": _safe_int_1410(q4_147_summary.get("q4_fatal_n", 0), 0),
        "q4_publication_blocker_n": _safe_int_1410(q4_147_summary.get("publication_blocker_n", 0), 0),
        "promoted": False,
        "artifact_protocol": "",
        "artifact_cps": "",
    },
    {
        "candidate_name": "A1a_zigbee_semantic_counts",
        "source_cell": "14.8/14.9",
        "decision": "deferred_no_materialization",
        "status": "deferred",
        "reason": "A1a requires accepted A0 final base; A0 was not promoted, so A1a was deferred.",
        "pairs_total": _safe_int_1410(a1a_149_summary.get("cell14_8_A1_semantic_count_candidates", 0), 0),
        "q4_pass_n": -1,
        "q4_warning_n": -1,
        "q4_fatal_or_blocker_n": -1,
        "q4_publication_blocker_n": -1,
        "promoted": False,
        "artifact_protocol": "",
        "artifact_cps": "",
    },
    {
        "candidate_name": "A1b_zigbee_rates",
        "source_cell": "14.8",
        "decision": "deferred",
        "status": "deferred",
        "reason": "Rate/intensity columns should be recomputed from count features, not directly repaired.",
        "pairs_total": _safe_int_1410(audit_148_counts.get("A1_rate_recompute_candidates", 0), 0),
        "q4_pass_n": -1,
        "q4_warning_n": -1,
        "q4_fatal_or_blocker_n": -1,
        "q4_publication_blocker_n": -1,
        "promoted": False,
        "artifact_protocol": "",
        "artifact_cps": "",
    },
    {
        "candidate_name": "A1c_zigbee_cardinality",
        "source_cell": "14.8",
        "decision": "deferred",
        "status": "deferred",
        "reason": "Cardinality repair requires support projection and upper-bound constraints; no direct repair authorized.",
        "pairs_total": _safe_int_1410(audit_148_counts.get("A1_cardinality_candidates", 0), 0),
        "q4_pass_n": -1,
        "q4_warning_n": -1,
        "q4_fatal_or_blocker_n": -1,
        "q4_publication_blocker_n": -1,
        "promoted": False,
        "artifact_protocol": "",
        "artifact_cps": "",
    },
    {
        "candidate_name": "router_coupling_branch",
        "source_cell": "14.8",
        "decision": "deferred",
        "status": "diagnostic_only",
        "reason": "Router evidence remains audit-only; no router repair branch authorized.",
        "pairs_total": _safe_int_1410(audit_148_counts.get("Router_R0_future_candidates", 0), 0),
        "q4_pass_n": -1,
        "q4_warning_n": -1,
        "q4_fatal_or_blocker_n": -1,
        "q4_publication_blocker_n": -1,
        "promoted": False,
        "artifact_protocol": "",
        "artifact_cps": "",
    },
]
candidate_df = pd.DataFrame(candidate_rows)
candidate_df.to_csv(candidate_comparison_csv, index=False)

# ----------------------------------------------------------
# 5) Decision ledger
# ----------------------------------------------------------
ledger_rows = [
    {
        "decision_id": "Q4_FINAL_001",
        "decision_area": "final_publication_q4_status",
        "decision": "blocked_no_promotion",
        "status": "blocked",
        "rationale": (
            "Broad Q4 failed, A0 repair remained blocked, no TRAIN/VAL-legitimate pruning policy was found, "
            "and A1/router branches were audit-only/deferred."
        ),
        "artifact_protocol": "",
        "artifact_cps": "",
        "TEST_real_values_used": False,
        "synthetic_values_mutated_here": False,
    },
    {
        "decision_id": "Q4_FINAL_002",
        "decision_area": "A0_zigbee_safe",
        "decision": "retain_as_failed_diagnostic_candidate_do_not_promote",
        "status": "blocked",
        "rationale": "A0 candidate had remaining fatal blockers and was not promotable without TEST-outcome pruning.",
        "artifact_protocol": "",
        "artifact_cps": "",
        "TEST_real_values_used": False,
        "synthetic_values_mutated_here": False,
    },
    {
        "decision_id": "Q4_FINAL_003",
        "decision_area": "A1a_zigbee_semantic_counts",
        "decision": "defer_no_materialization",
        "status": "deferred",
        "rationale": "A1a requires accepted A0 final base; no accepted A0 artifact exists.",
        "artifact_protocol": "",
        "artifact_cps": "",
        "TEST_real_values_used": False,
        "synthetic_values_mutated_here": False,
    },
    {
        "decision_id": "Q4_FINAL_004",
        "decision_area": "A1b_A1c_router",
        "decision": "defer_or_audit_only",
        "status": "deferred",
        "rationale": "Rate/cardinality/router repairs require separate predeclared constraint branches and were not authorized.",
        "artifact_protocol": "",
        "artifact_cps": "",
        "TEST_real_values_used": False,
        "synthetic_values_mutated_here": False,
    },
]
ledger_df = pd.DataFrame(ledger_rows)
ledger_df.to_csv(decision_ledger_csv, index=False)

# ----------------------------------------------------------
# 6) Publication statement
# ----------------------------------------------------------
publication_statement = f"""Final Q4 Cross-Modal Coupling Decision
=====================================

Final Q4 status:
- blocked_no_promotion

No final Q4-coupled protocol or CPS artifact was promoted.

Rationale:
The broad schema-driven Q4 evaluation remained a publication blocker. A constrained A0 Zigbee-safe repair branch improved timing/profile metrics, but retained two fatal publication blockers. A follow-up legality diagnostic found no defensible simple TRAIN/VAL-only pruning rule that could remove the fatal rows while retaining a sufficiently large clean subset. Therefore, the A0 candidate was not promoted.

A1 and router expansion:
Cell 14.8 found additional TRAIN/VAL coupling evidence, including:
- A1a semantic-count candidates: {audit_148_counts.get("A1_semantic_count_candidates", "NA")}
- A1b rate/recompute candidates: {audit_148_counts.get("A1_rate_recompute_candidates", "NA")}
- A1c cardinality candidates: {audit_148_counts.get("A1_cardinality_candidates", "NA")}
- Router R0 future candidates: {audit_148_counts.get("Router_R0_future_candidates", "NA")}

However, these branches were not authorized for materialization. Cell 14.9 therefore deferred A1a because no accepted A0 final base exists.

Scientific statement:
Q4 is reported as a blocked quality dimension in the final publication ledger. The A0 and A1/router analyses are retained as diagnostic evidence showing that protocol-specific coupling repair can improve some metrics, but no final Q4-coupled artifact is released because the acceptance and governance gates did not pass.
"""

with open(publication_statement_txt, "w", encoding="utf-8") as f:
    f.write(publication_statement)

# ----------------------------------------------------------
# 7) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "14.10",
    "version": CELL1410_VERSION,
    "role": "final_q4_no_promotion_decision_ledger",
    "final_decision": {
        "final_q4_status": "blocked_no_promotion",
        "accepted_candidate": "",
        "accepted_protocol_path": "",
        "accepted_cps_path": "",
        "A0_promoted": False,
        "A1a_promoted": False,
        "A1b_promoted": False,
        "A1c_promoted": False,
        "router_promoted": False,
    },
    "upstream_contract_versions": {
        "cell13_4": version134_1410,
        "cell14_7_no_promotion": version147_1410,
        "cell14_8_audit": version148_1410,
        "cell14_9_deferral": version149_1410,
    },
    "candidate_comparison": candidate_df.to_dict("records"),
    "decision_ledger": ledger_df.to_dict("records"),
    "publication_statement": publication_statement,
    "strict_contract": {
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "artifact_copy_done_here": False,
        "promotion_done_here": False,
        "decision_ledger_done_here": True,
    },
    "outputs": {
        "decision_ledger_csv": decision_ledger_csv,
        "candidate_comparison_csv": candidate_comparison_csv,
        "publication_statement_txt": publication_statement_txt,
        "decision_contract_json": decision_contract_json,
        "decision_contract_canonical_json": decision_contract_canonical_json,
        "decision_manifest_json": decision_manifest_json,
    },
}

_write_json_1410(decision_contract_json, contract)
_write_json_1410(decision_contract_canonical_json, contract)

manifest = {
    "cell": "14.10",
    "version": CELL1410_VERSION,
    "created_outputs": contract["outputs"],
    "final_decision": contract["final_decision"],
    "strict_contract": contract["strict_contract"],
}
_write_json_1410(decision_manifest_json, manifest)

hashes = {
    "decision_ledger_csv_sha256": _sha256_file_1410(decision_ledger_csv),
    "candidate_comparison_csv_sha256": _sha256_file_1410(candidate_comparison_csv),
    "publication_statement_txt_sha256": _sha256_file_1410(publication_statement_txt),
    "decision_contract_json_sha256": _sha256_file_1410(decision_contract_json),
    "decision_contract_canonical_json_sha256": _sha256_file_1410(decision_contract_canonical_json),
    "decision_manifest_json_sha256": _sha256_file_1410(decision_manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_1410(decision_contract_json, contract)
_write_json_1410(decision_contract_canonical_json, contract)
_write_json_1410(decision_manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals
# ----------------------------------------------------------
globals()["CELL1410_VERSION"] = CELL1410_VERSION
globals()["CELL14_10_FINAL_Q4_DECISION_LEDGER_DF"] = ledger_df
globals()["CELL14_10_FINAL_Q4_CANDIDATE_COMPARISON_DF"] = candidate_df
globals()["CELL14_10_FINAL_Q4_DECISION_CONTRACT"] = contract
globals()["CELL14_10_FINAL_Q4_PUBLICATION_STATEMENT"] = publication_statement
globals()["CELL14_10_FINAL_Q4_STATUS"] = "blocked_no_promotion"

globals()["CELL14_10_FINAL_PROTOCOL_PATH"] = ""
globals()["CELL14_10_FINAL_CPS_PATH"] = ""
globals()["CELL14_10_DECISION_LEDGER_CSV"] = decision_ledger_csv
globals()["CELL14_10_CANDIDATE_COMPARISON_CSV"] = candidate_comparison_csv
globals()["CELL14_10_PUBLICATION_STATEMENT_TXT"] = publication_statement_txt
globals()["CELL14_10_DECISION_CONTRACT_JSON"] = decision_contract_json
globals()["CELL14_10_DECISION_CONTRACT_CANONICAL_JSON"] = decision_contract_canonical_json
globals()["CELL14_10_DECISION_MANIFEST_JSON"] = decision_manifest_json

log(
    "[Cell14.10] Final Q4 no-promotion decision ledger complete | "
    "final_q4_status=blocked_no_promotion | accepted_candidate=<none> | "
    f"broad_pair_blockers={q4_134_summary.get('pair_summary', {}).get('pair_blocker_n', 'NA')} | "
    f"A0_blockers={q4_147_summary.get('publication_blocker_n', 'NA')} | "
    f"A1a_deferred={bool(strict149_1410.get('A1a_deferred', False))}"
)
log("[Cell14.10] Decision summary | Q4 blocked; A0 not promoted; A1a/A1b/A1c/router deferred.")
log(f"[Cell14.10] Saved decision ledger: {decision_ledger_csv}")
log(f"[Cell14.10] Saved candidate comparison: {candidate_comparison_csv}")
log(f"[Cell14.10] Saved publication statement: {publication_statement_txt}")
log(f"[Cell14.10] Saved decision contract: {decision_contract_json}")
log(f"[Cell14.10] Saved canonical decision contract: {decision_contract_canonical_json}")
log(
    "[Cell14.10] Contract flags | "
    "TEST_real_values_used=False | TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | selection_done_here=False | generator_fit_done_here=False | "
    "materialization_done_here=False | artifact_copy_done_here=False | promotion_done_here=False | "
    "decision_ledger_done_here=True"
)
log("--- END: Cell 14.10 - Final Q4 no-promotion decision ledger (v1.1 strict) ---")