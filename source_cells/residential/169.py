# %% CELL Q6.3 — Final STUDY-THESIS governance gate
# Purpose:
#   Make the final submission status explicit and claim-scoped.
#   This cell fails if Q6.0–Q6.2 are missing or if invalid active branch evidence is detected.

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

OUTDIR_G = Path(str(globals().get("OUTDIR", os.environ.get("CPS_OUTDIR")))).expanduser().resolve()
REPORT_DIR_G = Path(str(globals().get("REPORT_DIR", OUTDIR_G / "reports"))).expanduser().resolve()
CONTRACT_DIR_G = Path(str(globals().get("CONTRACT_DIR", OUTDIR_G / "artifacts" / "contracts"))).expanduser().resolve()

required_contracts = [
    CONTRACT_DIR_G / "q6_active_branch_registry_contract_v1_0_THESIS.json",
    CONTRACT_DIR_G / "q6_active_artifact_hygiene_contract_v1_0_THESIS.json",
    CONTRACT_DIR_G / "q6_reproducibility_manifest_contract_v1_0_THESIS.json",
    CONTRACT_DIR_G / "q3_post_v5_observability_mask_post_qa_contract.json",
    CONTRACT_DIR_G / "cell12d6_binary_final_test_qa_contract_v6_0_THESIS.json",
    CONTRACT_DIR_G / "cell12c6R_final_qa_contract.json",
]
missing = [str(p) for p in required_contracts if not p.exists()]
if missing:
    raise RuntimeError("[Q6.3] Missing required Q6/active contracts:\n" + "\n".join(missing))

active = json.loads((CONTRACT_DIR_G / "q6_active_branch_registry_contract_v1_0_THESIS.json").read_text(encoding="utf-8"))
hygiene = json.loads((CONTRACT_DIR_G / "q6_active_artifact_hygiene_contract_v1_0_THESIS.json").read_text(encoding="utf-8"))
q3 = json.loads((CONTRACT_DIR_G / "q3_post_v5_observability_mask_post_qa_contract.json").read_text(encoding="utf-8"))

issues = []
if not hygiene.get("passed", False):
    issues.append("Q6.1 artifact hygiene did not pass")
if q3.get("cell") != "Q3.POST.V5":
    issues.append("Q3 active post contract is not Q3.POST.V5")
if q3.get("final_status") not in {"BLOCKED_PARTIAL", "PASS_PARTIAL_WITH_SCOPE_EXCLUSIONS", "PASS_FULL"}:
    issues.append(f"Unexpected Q3 final_status: {q3.get('final_status')}")
if q3.get("full_scope_ready") is True:
    issues.append("Q3 claims full_scope_ready=True; current vetted branch should be partial unless fresh holdout proves otherwise")

final_status = "SUBMISSION_READY_WITH_PARTIAL_CLAIMS" if not issues else "BLOCKED_Q6_GOVERNANCE"
summary = {
    "cell": "Q6.3",
    "role": "final_study_thesis_governance_gate",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "final_status": final_status,
    "issues": issues,
    "active_q3_branch": "Q3.POST.V5",
    "active_binary_branch": "12.d.V6",
    "active_continuous_branch": "12.c.R",
    "claim_scope_summary": {
        "Q3": "partial_admissible_only",
        "Q5_binary": "partial_best_development",
        "continuous_values": "lineage_safe_value_limited",
        "fresh_holdout_required_for_blind_final_claims": True,
    },
    "policy": {
        "reads_TEST_real_values": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "post_TEST_repair_done_here": False,
    },
}
summary_path = REPORT_DIR_G / "q6_final_study_thesis_governance_gate.json"
contract_path = CONTRACT_DIR_G / "q6_final_study_thesis_governance_gate_contract_v1_0_THESIS.json"
for p in [summary_path, contract_path]:
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

print("\n" + "=" * 72)
print("Q6 FINAL STUDY-THESIS GOVERNANCE GATE")
print("=" * 72)
print("Final status:", final_status)
print("Active continuous:", "12.c.R")
print("Active binary:", "12.d.V6")
print("Active Q3:", "Q3.POST.V5")
print("Q3 full-scope ready:", q3.get("full_scope_ready"))
print("Q3 claim scope:", q3.get("claim_scope"))
print("Issues:", issues)
print("Contract:", contract_path)
print("=" * 72)

if issues:
    raise RuntimeError("[Q6.3] Final governance gate failed: " + "; ".join(issues))

CELL_Q6_FINAL_GATE = summary
