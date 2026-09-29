# %% CELL Q6.4 — Reviewer README addendum for active branch evidence
# Purpose:
#   Generate a concise reviewer-facing README describing active branches,
#   rejected/superseded branches, and claim limitations.
#
# Safety:
#   Documentation only.

import os
import json
from pathlib import Path
from datetime import datetime, timezone

OUTDIR_README = Path(str(globals().get("OUTDIR", os.environ.get("CPS_OUTDIR")))).expanduser().resolve()
REPORT_DIR_README = Path(str(globals().get("REPORT_DIR", OUTDIR_README / "reports"))).expanduser().resolve()
CONTRACT_DIR_README = Path(str(globals().get("CONTRACT_DIR", OUTDIR_README / "artifacts" / "contracts"))).expanduser().resolve()

q3_path = CONTRACT_DIR_README / "q3_post_v5_observability_mask_post_qa_contract.json"
q6_path = CONTRACT_DIR_README / "q6_final_study_thesis_governance_gate_contract_v1_0_THESIS.json"
q3 = json.loads(q3_path.read_text(encoding="utf-8")) if q3_path.exists() else {}
q6 = json.loads(q6_path.read_text(encoding="utf-8")) if q6_path.exists() else {}

readme = f"""# STUDY-THESIS active artifact evidence addendum

Generated: {datetime.now(timezone.utc).isoformat()}

## Active branches

| Dimension | Active branch | Claim scope |
|---|---|---|
| Continuous IoT values | `12.c.R` | lineage-safe, value-limited |
| Binary IoT states | `12.d.V6` | partial development branch, 39 included / 13 scope-excluded / 3 included blockers |
| Observability masks | `12.f.V5` / `Q3.POST.V5` | {q3.get('claim_scope', 'partial_admissible_only')} |

## Rejected and superseded branches

- `Q3.POST.V2` is rejected: it used missingness of indicator columns rather than the actual 0/1 indicator states.
- `Q3.POST.V3` and `Q3.POST.V4` are valid development baselines but superseded by `Q3.POST.V5`.
- Old `12.c.3–12.c.6` continuous artifacts are superseded by `12.c.R`.
- Old `12.f.3x–12.f.5` mask artifacts are superseded by `12.f.V5`.
- Binary V2–V5 are superseded by Binary V6.

## Current Q3 result

- Final status: `{q3.get('final_status', 'unknown')}`
- Full-scope ready: `{q3.get('full_scope_ready', 'unknown')}`
- Publication blockers: `{q3.get('publication_blocker_n', 'unknown')}`

## Claim limitation

This notebook produces a clean, claim-scoped STUDY-THESIS artifact. Current same-`df_te` results are development evidence unless the frozen design is evaluated once on a fresh untouched temporal holdout.

## Q6 governance gate

Final Q6 status: `{q6.get('final_status', 'unknown')}`
"""
readme_path = REPORT_DIR_README / "Q6_ACTIVE_EVIDENCE_README_STUDY_THESIS.md"
readme_path.write_text(readme, encoding="utf-8")
contract = {
    "cell": "Q6.4",
    "role": "reviewer_readme_addendum_for_active_branch_evidence",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "readme_path": str(readme_path),
    "policy": {"documentation_only": True, "synthetic_values_mutated": False, "reads_TEST_real_values": False},
}
contract_path = CONTRACT_DIR_README / "q6_active_evidence_readme_contract_v1_0_THESIS.json"
contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
print(f"[Q6.4] README written: {readme_path}")
print(f"[Q6.4] Contract: {contract_path}")
