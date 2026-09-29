# %% CELL Q3.POST.V5 — Observability-mask V5 post-QA attestation
# Purpose:
#   Summarize V5 using canonical 12.f.5 files. No repair or mutation.

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

metrics_path = REPORT_DIR_P / "cell12f5_mask_final_test_qa_metrics.csv"
status_path = REPORT_DIR_P / "cell12f5_mask_publication_status.csv"
blocker_path = REPORT_DIR_P / "cell12f5_mask_blocker_origin_audit.csv"
pre_contract = CONTRACT_DIR_P / "cell12f_pre_q3_v5_readiness_contract.json"
final_contract = CONTRACT_DIR_P / "cell12f5_mask_final_test_qa_contract_v1_1_THESIS.json"

for p in [metrics_path, status_path, blocker_path, pre_contract, final_contract]:
    if not p.exists():
        raise RuntimeError(f"[Q3.POST.V5] Missing required file: {p}")

metrics = pd.read_csv(metrics_path)
status = pd.read_csv(status_path)
blocker = pd.read_csv(blocker_path)
contract = json.loads(final_contract.read_text(encoding="utf-8"))

status_col = "mask_publication_status_after_12f5"
blocker_col = "mask_publication_blocker_after_12f5"

counts = status[status_col].fillna("NA").astype(str).value_counts().to_dict()
qa_counts = metrics["test_qa_status"].fillna("NA").astype(str).value_counts().to_dict()
blocker_n = int(status[blocker_col].astype(bool).sum())
blocker_origins = blocker["blocker_origin"].fillna("NA").astype(str).value_counts().to_dict()

full_scope_ready = blocker_n == 0 and int(qa_counts.get("fatal", 0)) == 0 and counts.get("scope_excluded", 0) == 0
if full_scope_ready:
    final_status = "PASS_FULL"
    claim_scope = "full_mask_scope"
elif blocker_n > 0:
    final_status = "BLOCKED_PARTIAL"
    claim_scope = "partial_admissible_only"
else:
    final_status = "PASS_PARTIAL_WITH_SCOPE_EXCLUSIONS"
    claim_scope = "trainval_backtested_partial_scope"

report = {
    "cell": "Q3.POST.V5",
    "role": "observability_mask_v5_semantic_trainval_backtest_post_qa_attestation",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "targets": int(len(metrics)),
    "qa_status_counts": qa_counts,
    "publication_status_counts": counts,
    "publication_blocker_n": blocker_n,
    "blocker_origin_counts": blocker_origins,
    "final_status": final_status,
    "claim_scope": claim_scope,
    "full_scope_ready": full_scope_ready,
    "mean_metrics": contract.get("mean_metrics", {}),
    "semantic_mode_counts": contract.get("semantic_mode_counts", {}),
    "policy": {
        "TEST_QA_reports_read_only": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "semantic_parser": "actual_indicator_values_not_notna",
        "selector": "TRAIN_to_VAL_backtest_only",
        "same_df_te_v5_status": "development_only_unless_fresh_holdout_used",
    },
    "sources": {
        "metrics": str(metrics_path),
        "status": str(status_path),
        "blocker": str(blocker_path),
        "pre_contract": str(pre_contract),
        "final_contract": str(final_contract),
    },
}

for p in [REPORT_DIR_P / "q3_post_v5_observability_mask_post_qa_contract.json", CONTRACT_DIR_P / "q3_post_v5_observability_mask_post_qa_contract.json"]:
    p.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

pd.DataFrame([{
    "Q": "Q3",
    "branch": "observability_mask_v5_semantic_trainval_backtest",
    "targets": report["targets"],
    "qa_status_counts": json.dumps(qa_counts, sort_keys=True),
    "publication_status_counts": json.dumps(counts, sort_keys=True),
    "publication_blocker_n": blocker_n,
    "final_status": final_status,
    "claim_scope": claim_scope,
    "full_scope_ready": full_scope_ready,
}]).to_csv(REPORT_DIR_P / "q3_post_v5_claim_ledger_row.csv", index=False)

CELL12F_V5_Q3_POST = report

print("\n========================================================================")
print("Q3.POST.V5 OBSERVABILITY-MASK SUMMARY")
print("========================================================================")
print(f"Targets:                {report['targets']}")
print(f"QA status counts:       {qa_counts}")
print(f"Publication statuses:   {counts}")
print(f"Publication blockers:   {blocker_n}")
print(f"Blocker origins:        {blocker_origins}")
print(f"Final status:           {final_status}")
print(f"Claim scope:            {claim_scope}")
print(f"Full-scope ready:       {full_scope_ready}")
print(f"Mean metrics:           {report['mean_metrics']}")
print(f"Post contract:          {CONTRACT_DIR_P / 'q3_post_v5_observability_mask_post_qa_contract.json'}")
print("========================================================================")