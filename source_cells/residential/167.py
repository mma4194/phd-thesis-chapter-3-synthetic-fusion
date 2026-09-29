# %% CELL Q6.1 — Active artifact hygiene and superseded-branch guard
# Purpose:
#   Fail closed if a final clean root exposes invalid/superseded branch outputs as active artifacts.
#
# Safety:
#   - Reads filenames/contracts/reports only.
#   - Does not mutate data.
#   - Does not fit/select/repair.
#
# Important:
#   A clean STUDY run should not contain Q3 V2/V3/V4 post contracts or old continuous TEST-policy
#   selector artifacts. If this cell fails in a development root, use a fresh CPS_STUDY_FINAL_OUTDIR.

import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def _q6h_log(msg):
    print(f"[Q6.1] {msg}")

OUTDIR_H = Path(str(globals().get("OUTDIR", os.environ.get("CPS_OUTDIR")))).expanduser().resolve()
REPORT_DIR_H = Path(str(globals().get("REPORT_DIR", OUTDIR_H / "reports"))).expanduser().resolve()
CONTRACT_DIR_H = Path(str(globals().get("CONTRACT_DIR", OUTDIR_H / "artifacts" / "contracts"))).expanduser().resolve()
ARTIFACT_DIR_H = Path(str(globals().get("ARTIFACT_DIR", globals().get("ARTDIR", OUTDIR_H / "artifacts")))).expanduser().resolve()

ALLOW_SUPERSEDED_IN_DEV = bool(isinstance(globals().get("CFG"), dict) and globals()["CFG"].get("q6_allow_superseded_artifacts_in_dev_root", False))

active_required = {
    "q3_post_v5": CONTRACT_DIR_H / "q3_post_v5_observability_mask_post_qa_contract.json",
    "q3_terminal_v5": CONTRACT_DIR_H / "cell12f5_mask_final_test_qa_contract_v1_1_THESIS.json",
    "continuous_12cR": CONTRACT_DIR_H / "cell12c6R_final_qa_contract.json",
    "binary_v6": CONTRACT_DIR_H / "cell12d6_binary_final_test_qa_contract_v6_0_THESIS.json",
    "q6_active": CONTRACT_DIR_H / "q6_active_branch_registry_contract_v1_0_THESIS.json",
}
missing = [f"{k}: {p}" for k, p in active_required.items() if not p.exists()]

# Active contract semantic checks.
semantic_issues = []
try:
    q3_v5 = json.loads(active_required["q3_post_v5"].read_text(encoding="utf-8"))
    if q3_v5.get("cell") != "Q3.POST.V5":
        semantic_issues.append(f"q3_post_v5 contract cell is {q3_v5.get('cell')!r}, expected Q3.POST.V5")
    if q3_v5.get("full_scope_ready") is True:
        semantic_issues.append("q3_post_v5 unexpectedly claims full_scope_ready=True; expected partial scope for current evidence")
except Exception as e:
    semantic_issues.append(f"could not parse q3_post_v5 contract: {e}")

try:
    q3_terminal = json.loads(active_required["q3_terminal_v5"].read_text(encoding="utf-8"))
    if q3_terminal.get("cell") != "12.f.V5.4":
        semantic_issues.append(f"canonical 12.f.5 terminal contract cell is {q3_terminal.get('cell')!r}, expected 12.f.V5.4")
except Exception as e:
    semantic_issues.append(f"could not parse canonical 12.f.5 contract: {e}")

# File-pattern hygiene.
superseded_patterns = [
    r"q3_post_v2", r"cell12f_v2", r"Q3\.POST\.V2",
    r"q3_post_v3", r"cell12f_v3", r"Q3\.POST\.V3",
    r"q3_post_v4", r"cell12f_v4", r"Q3\.POST\.V4",
]
old_continuous_patterns = [
    r"cell12c_continuous_selector_test_policy",
    r"cell12c3_dense_temperature_selector_audit\.csv",
    r"cell12c3_temperature_v2_final_selection_policy",
    r"cell12c3_diagnostic_pool_not_selected_audit\.csv",
]

scan_roots = [REPORT_DIR_H, CONTRACT_DIR_H]
rows = []
for root in scan_roots:
    if not root.exists():
        continue
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(OUTDIR_H) if p.is_relative_to(OUTDIR_H) else p.name
        s = str(rel)
        if any("q6_" in part.lower() for part in p.parts):
            continue
        superseded_hit = [pat for pat in superseded_patterns if re.search(pat, s, flags=re.I)]
        old_cont_hit = [pat for pat in old_continuous_patterns if re.search(pat, s, flags=re.I)]
        content_hit = []
        if p.suffix.lower() in {".csv", ".json", ".txt", ".md"} and p.stat().st_size < 5_000_000:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                for term in ["final_selection_disabled_after_testqa", "after_testqa_regression_audit", "Q3.POST.V2", "MaskExactObservedConstantV2"]:
                    if term in txt:
                        content_hit.append(term)
            except Exception:
                pass
        if superseded_hit or old_cont_hit or content_hit:
            rows.append({
                "relative_path": str(rel),
                "path": str(p),
                "superseded_filename_hit": ";".join(superseded_hit),
                "old_continuous_filename_hit": ";".join(old_cont_hit),
                "content_hit": ";".join(content_hit),
            })

hygiene_df = pd.DataFrame(rows)
report_path = REPORT_DIR_H / "q6_active_artifact_hygiene_report.csv"
summary_path = REPORT_DIR_H / "q6_active_artifact_hygiene_summary.json"
contract_path = CONTRACT_DIR_H / "q6_active_artifact_hygiene_contract_v1_0_THESIS.json"
hygiene_df.to_csv(report_path, index=False)
summary = {
    "cell": "Q6.1",
    "role": "active_artifact_hygiene_and_superseded_branch_guard",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "missing_active_artifacts": missing,
    "semantic_issues": semantic_issues,
    "superseded_or_invalid_artifact_hits_n": int(len(hygiene_df)),
    "allow_superseded_in_dev_root": ALLOW_SUPERSEDED_IN_DEV,
    "passed": not missing and not semantic_issues and (len(hygiene_df) == 0 or ALLOW_SUPERSEDED_IN_DEV),
    "policy": {
        "reads_TEST_real_values": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "post_TEST_repair_done_here": False,
    },
}
for p in [summary_path, contract_path]:
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

_q6h_log(f"Hygiene report: {report_path}")
_q6h_log(f"Superseded/invalid artifact hits: {len(hygiene_df)}")
if missing or semantic_issues or (len(hygiene_df) and not ALLOW_SUPERSEDED_IN_DEV):
    if len(hygiene_df):
        print(hygiene_df.head(50).to_string(index=False))
    raise RuntimeError(
        "[Q6.1] Active artifact hygiene failed. Use a fresh clean OUTDIR for the final STUDY run, "
        "or set CFG['q6_allow_superseded_artifacts_in_dev_root']=True only for a non-submission development root."
    )

CELL_Q6_ARTIFACT_HYGIENE = summary
_q6h_log("PASS: active artifact hygiene checks passed.")
