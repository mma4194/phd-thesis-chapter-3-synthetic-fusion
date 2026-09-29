# %% CELL FINAL.Q6 — Artifact-focused pre-submission hygiene guard
# Purpose:
#   Final guard for submitted runs. This replaces brittle notebook-source scanning.
#   It validates active artifact contracts, not harmless source comments or local HPC paths.

import os
import json
from pathlib import Path

OUTDIR_F = Path(str(globals().get("OUTDIR", os.environ.get("CPS_OUTDIR")))).expanduser().resolve()
CONTRACT_DIR_F = Path(str(globals().get("CONTRACT_DIR", OUTDIR_F / "artifacts" / "contracts"))).expanduser().resolve()
REPORT_DIR_F = Path(str(globals().get("REPORT_DIR", OUTDIR_F / "reports"))).expanduser().resolve()

required = [
    CONTRACT_DIR_F / "q6_final_study_thesis_governance_gate_contract_v1_0_THESIS.json",
    CONTRACT_DIR_F / "q6_active_artifact_hygiene_contract_v1_0_THESIS.json",
    CONTRACT_DIR_F / "q6_active_branch_registry_contract_v1_0_THESIS.json",
    CONTRACT_DIR_F / "q3_post_v5_observability_mask_post_qa_contract.json",
]
issues = []
for p in required:
    if not p.exists():
        issues.append(f"missing required final contract: {p}")

if not issues:
    gate = json.loads((CONTRACT_DIR_F / "q6_final_study_thesis_governance_gate_contract_v1_0_THESIS.json").read_text(encoding="utf-8"))
    hygiene = json.loads((CONTRACT_DIR_F / "q6_active_artifact_hygiene_contract_v1_0_THESIS.json").read_text(encoding="utf-8"))
    if gate.get("final_status") != "SUBMISSION_READY_WITH_PARTIAL_CLAIMS":
        issues.append(f"unexpected Q6 final status: {gate.get('final_status')}")
    if hygiene.get("passed") is not True:
        issues.append("Q6 artifact hygiene did not pass")

contract = {
    "cell": "FINAL.Q6",
    "role": "artifact_focused_pre_submission_hygiene_guard",
    "passed": not issues,
    "issues": issues,
    "checks": [str(p) for p in required],
}
path = CONTRACT_DIR_F / "final_artifact_focused_pre_submission_hygiene_guard_THESIS.json"
path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
print(f"[FINAL.Q6] Contract: {path}")
if issues:
    raise RuntimeError("[FINAL.Q6] Pre-submission hygiene failed: " + "; ".join(issues))
print("[FINAL.Q6] PASS: artifact-focused pre-submission hygiene checks passed.")
