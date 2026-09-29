# ==========================================================
# CELL 14.9 - A1a expansion deferral ledger
# v1.1 STUDY-THESIS strict no-materialization governance
#
# Role:
#   - Record that A1a Zigbee semantic-count expansion is deferred.
#   - Reason: Cell 14.7 recorded Q4 as blocked_no_promotion, so there is
#     no accepted A0 final artifact on which A1a can safely build.
#   - Preserve Cell 14.8 A1/router audit evidence as future-work context.
#
# This cell does NOT:
#   - fit models,
#   - select generators,
#   - materialize protocol values,
#   - mutate synthetic values,
#   - use TEST values,
#   - promote any candidate.
#
# Outputs:
#   reports/cell14_9_A1a_deferral_summary.csv
#   reports/cell14_9_A1a_deferral_contract.json
#   artifacts/contracts/cell14_9_A1a_deferral_contract_v1_1_THESIS.json
#   artifacts/cell14_9_A1a_deferral_manifest.json
# ==========================================================

log("--- START: Cell 14.9 - A1a expansion deferral ledger (v1.1 strict no-materialization) ---")

import os
import json
import hashlib
import numpy as np
import pandas as pd

_required_149 = [
    "CFG", "log", "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "CELL14_7_Q4_NO_PROMOTION_CONTRACT",
    "CELL14_8_A1_ROUTER_AUDIT_CONTRACT",
]
_missing_149 = [k for k in _required_149 if k not in globals()]
if _missing_149:
    raise RuntimeError(f"[Cell14.9] Missing required globals: {_missing_149}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_149 = str(OUT_SYN)
REPORT_DIR_ACTIVE_149 = str(REPORT_DIR)

def _resolve_project_root_149(outdir, report_dir, out_syn):
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
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "synthetic")):
            return c
    raise RuntimeError("[Cell14.9] Could not resolve canonical project root.")

PROJECT_ROOT_149 = _resolve_project_root_149(OUTDIR, REPORT_DIR_ACTIVE_149, OUT_SYN_ACTIVE_149)
REPORT_DIR_149 = os.path.join(PROJECT_ROOT_149, "reports")
ARTDIR_149 = os.path.join(PROJECT_ROOT_149, "artifacts")
CONTRACT_DIR_149 = os.path.join(ARTDIR_149, "contracts")
os.makedirs(REPORT_DIR_149, exist_ok=True)
os.makedirs(ARTDIR_149, exist_ok=True)
os.makedirs(CONTRACT_DIR_149, exist_ok=True)

CELL149_VERSION = "cell14_9_A1a_expansion_deferral_v1_1_no_materialization"

CFG["cell14_9_version"] = CELL149_VERSION
CFG["cell14_9_A1a_deferred"] = True
CFG["cell14_9_no_materialization"] = True
CFG["cell14_9_TEST_real_values_used"] = False
CFG["cell14_9_synthetic_values_mutated"] = False
CFG["cell14_9_selection_done_here"] = False
CFG["cell14_9_generator_fit_done_here"] = False
CFG["cell14_9_materialization_done_here"] = False
CFG["cell14_9_promotion_done_here"] = False

def _require_contract_version_149(obj, name, expected_substring):
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.9] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.9] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

version147_149 = _require_contract_version_149(
    CELL14_7_Q4_NO_PROMOTION_CONTRACT,
    "CELL14_7_Q4_NO_PROMOTION_CONTRACT",
    "cell14_7_q4_no_promotion_governance_v2_1",
)
version148_149 = _require_contract_version_149(
    CELL14_8_A1_ROUTER_AUDIT_CONTRACT,
    "CELL14_8_A1_ROUTER_AUDIT_CONTRACT",
    "cell14_8_A1_router_protocol_consistency_audit_v1_1",
)

strict147 = CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("strict_contract", {})
if bool(CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("accepted", True)):
    raise RuntimeError("[Cell14.9] No-promotion governance contract unexpectedly has accepted=True.")
if bool(CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("promoted", True)):
    raise RuntimeError("[Cell14.9] No-promotion governance contract unexpectedly has promoted=True.")
if bool(strict147.get("candidate_outputs_promoted_to_final", True)):
    raise RuntimeError("[Cell14.9] Governance contract says candidate outputs were promoted.")

strict148 = CELL14_8_A1_ROUTER_AUDIT_CONTRACT.get("strict_contract", {})
if bool(strict148.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.9] Cell 14.8 audit contract indicates TEST real values were used.")
if bool(strict148.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.9] Cell 14.8 audit contract indicates synthetic values were mutated.")
if not bool(strict148.get("audit_only", False)):
    raise RuntimeError("[Cell14.9] Cell 14.8 did not declare audit_only.")
if bool(CELL14_8_A1_ROUTER_AUDIT_CONTRACT.get("expansion_authorization", {}).get("direct_repair_authorized_here", True)):
    raise RuntimeError("[Cell14.9] Cell 14.8 unexpectedly authorizes direct repair.")

q4_summary = CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("summary", {})
governance_decision = q4_summary.get("governance_decision", {})
audit_counts = CELL14_8_A1_ROUTER_AUDIT_CONTRACT.get("audit_counts", {})
recommended_next_steps = CELL14_8_A1_ROUTER_AUDIT_CONTRACT.get("recommended_next_steps", [])

summary = {
    "cell": "14.9",
    "version": CELL149_VERSION,
    "decision": "A1a_deferred_no_materialization",
    "reason": (
        "Q4 was closed by Cell 14.7 as blocked_no_promotion. "
        "A1a requires an accepted A0 final base, but A0 was not promoted."
    ),
    "q4_final_status_carried_forward": governance_decision.get("final_q4_status", ""),
    "A0_promoted_carried_forward": bool(governance_decision.get("A0_promoted", False)),
    "A0_candidate_status_carried_forward": governance_decision.get("A0_candidate_status", ""),
    "A0_pairs_total": int(q4_summary.get("pairs_total", 0) or 0),
    "A0_pass_n": int(q4_summary.get("q4_pass_n", 0) or 0),
    "A0_warning_n": int(q4_summary.get("q4_warning_n", 0) or 0),
    "A0_fatal_n": int(q4_summary.get("q4_fatal_n", 0) or 0),
    "A0_publication_blocker_n": int(q4_summary.get("q4_publication_blocker_n", 0) or 0),
    "cell14_8_A1_semantic_count_candidates": int(audit_counts.get("A1_semantic_count_candidates", 0) or 0),
    "cell14_8_A1_rate_recompute_candidates": int(audit_counts.get("A1_rate_recompute_candidates", 0) or 0),
    "cell14_8_A1_cardinality_candidates": int(audit_counts.get("A1_cardinality_candidates", 0) or 0),
    "cell14_8_Router_R0_future_candidates": int(audit_counts.get("Router_R0_future_candidates", 0) or 0),
    "recommended_future_work": recommended_next_steps,
}

rows = [
    {"metric": "decision", "value": summary["decision"]},
    {"metric": "q4_final_status_carried_forward", "value": summary["q4_final_status_carried_forward"]},
    {"metric": "A0_promoted_carried_forward", "value": summary["A0_promoted_carried_forward"]},
    {"metric": "A0_pairs_total", "value": summary["A0_pairs_total"]},
    {"metric": "A0_pass_n", "value": summary["A0_pass_n"]},
    {"metric": "A0_warning_n", "value": summary["A0_warning_n"]},
    {"metric": "A0_fatal_n", "value": summary["A0_fatal_n"]},
    {"metric": "A0_publication_blocker_n", "value": summary["A0_publication_blocker_n"]},
    {"metric": "A1_semantic_count_candidates_from_14_8", "value": summary["cell14_8_A1_semantic_count_candidates"]},
    {"metric": "A1_rate_recompute_candidates_from_14_8", "value": summary["cell14_8_A1_rate_recompute_candidates"]},
    {"metric": "A1_cardinality_candidates_from_14_8", "value": summary["cell14_8_A1_cardinality_candidates"]},
    {"metric": "Router_R0_future_candidates_from_14_8", "value": summary["cell14_8_Router_R0_future_candidates"]},
]

summary_csv = os.path.join(REPORT_DIR_149, "cell14_9_A1a_deferral_summary.csv")
contract_json = os.path.join(REPORT_DIR_149, "cell14_9_A1a_deferral_contract.json")
canonical_contract_json = os.path.join(CONTRACT_DIR_149, "cell14_9_A1a_deferral_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_149, "cell14_9_A1a_deferral_manifest.json")

pd.DataFrame(rows).to_csv(summary_csv, index=False)

contract = {
    "cell": "14.9",
    "version": CELL149_VERSION,
    "role": "A1a_expansion_deferral_no_materialization",
    "summary": summary,
    "upstream_contract_versions": {
        "cell14_7_no_promotion": version147_149,
        "cell14_8_audit": version148_149,
    },
    "strict_contract": {
        "A1a_deferred": True,
        "TEST_real_values_used": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "promotion_done_here": False,
        "artifact_copy_done_here": False,
        "no_final_A0_base_available": True,
        "expansion_not_authorized_here": True,
    },
    "outputs": {
        "summary_csv": summary_csv,
        "contract_json": contract_json,
        "canonical_contract_json": canonical_contract_json,
        "manifest_json": manifest_json,
    },
}

def _sanitize(o):
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    if isinstance(o, tuple):
        return [_sanitize(v) for v in o]
    if isinstance(o, np.ndarray):
        return _sanitize(o.tolist())
    if isinstance(o, pd.DataFrame):
        return _sanitize(o.to_dict("records"))
    if isinstance(o, pd.Series):
        return _sanitize(o.to_dict())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if not np.isfinite(x) else x
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o

with open(contract_json, "w", encoding="utf-8") as f:
    json.dump(_sanitize(contract), f, indent=2, sort_keys=True)
with open(canonical_contract_json, "w", encoding="utf-8") as f:
    json.dump(_sanitize(contract), f, indent=2, sort_keys=True)

manifest = {
    "cell": "14.9",
    "version": CELL149_VERSION,
    "summary": summary,
    "strict_contract": contract["strict_contract"],
    "created_outputs": contract["outputs"],
}
with open(manifest_json, "w", encoding="utf-8") as f:
    json.dump(_sanitize(manifest), f, indent=2, sort_keys=True)

globals()["CELL149_VERSION"] = CELL149_VERSION
globals()["CELL14_9_A1A_DEFERRED"] = True
globals()["CELL14_9_A1A_FINAL_DECISION"] = "deferred_no_materialization"
globals()["CELL14_9_A1A_DEFERRAL_SUMMARY"] = summary
globals()["CELL14_9_A1A_DEFERRAL_CONTRACT"] = contract
globals()["CELL14_9_A1A_DEFERRAL_SUMMARY_CSV"] = summary_csv
globals()["CELL14_9_A1A_DEFERRAL_CONTRACT_JSON"] = contract_json
globals()["CELL14_9_A1A_DEFERRAL_CONTRACT_CANONICAL_JSON"] = canonical_contract_json
globals()["CELL14_9_A1A_DEFERRAL_MANIFEST_JSON"] = manifest_json

log(
    "[Cell14.9] A1a expansion deferred | "
    f"q4_final_status={summary['q4_final_status_carried_forward']} | "
    f"A0_promoted={summary['A0_promoted_carried_forward']} | "
    f"A1_semantic_count_candidates={summary['cell14_8_A1_semantic_count_candidates']} | "
    "materialization_done_here=False"
)
log(f"[Cell14.9] Deferral summary: {summary_csv}")
log(f"[Cell14.9] Contract: {canonical_contract_json}")
log(
    "[Cell14.9] Contract flags | A1a_deferred=True | TEST_real_values_used=False | "
    "synthetic_values_mutated=False | selection_done_here=False | generator_fit_done_here=False | "
    "materialization_done_here=False | promotion_done_here=False"
)
log("--- END: Cell 14.9 - A1a expansion deferral ledger (v1.1 strict no-materialization) ---")