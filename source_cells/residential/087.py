# %% CELL 12.f.V5.4 — Observability-mask V5 terminal TEST QA only
# Purpose:
#   Terminal TEST QA. This is the only V5 cell that reads TEST real values.
#
# Safety:
#   - TEST real values are used for QA only.
#   - No mutation, repair, reselection, or threshold change is performed here.

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "CELL12F_V5_MASK_FINAL" not in globals():
    p = OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_FINAL_TEST.parquet"
    if not p.exists():
        raise RuntimeError("[12.f.V5.4] Run 12.f.V5.3 first.")
    CELL12F_V5_MASK_FINAL = pd.read_parquet(p)

if "CELL12F_V5_SELECTION" not in globals():
    p = REPORT_DIR_P / "cell12f_v5_locked_mask_selection.csv"
    if not p.exists():
        raise RuntimeError("[12.f.V5.4] Run 12.f.V5.1 first.")
    CELL12F_V5_SELECTION = pd.read_csv(p)

sel = CELL12F_V5_SELECTION.set_index("col", drop=False)
syn = CELL12F_V5_MASK_FINAL
DF_TE = globals().get("DF_TE", globals().get("df_te"))

rows = []
for i, col in enumerate(syn.columns, start=1):
    if i == 1 or i % 25 == 0 or i == len(syn.columns):
        _f5_log(f"V5.4 QA {i}/{len(syn.columns)} | {col}")

    mode = CELL12F_V5_MASK_CACHE[col]["mode"]
    real = _state_values_12f5(DF_TE, col, mode)
    fake = pd.to_numeric(syn[col], errors="coerce").fillna(0).to_numpy(dtype=np.uint8)

    rs, fs = _stats_12f5(real), _stats_12f5(fake)
    all_m = _state_metrics_12f5(real, fake, state=None)
    one_m = _state_metrics_12f5(real, fake, state=1)
    zero_m = _state_metrics_12f5(real, fake, state=0)
    c2st_auc = _c2st_auc_12f5(real, fake, seed=41)

    rate_error = abs(rs["state_one_rate"] - fs["state_one_rate"])
    transition_error = abs(rs["transition_rate"] - fs["transition_rate"])

    srow = sel.loc[col] if col in sel.index else {}
    claim_included = _boolish_12f5(srow.get("claim_included", False)) if hasattr(srow, "get") else False
    scope_excluded = _boolish_12f5(srow.get("scope_excluded", True)) if hasattr(srow, "get") else True

    reasons = []
    if rate_error > 0.05:
        reasons.append("state_rate_error_gt_0.05")
    elif rate_error > 0.02:
        reasons.append("state_rate_error_gt_0.02")
    if all_m["ks"] > 0.50:
        reasons.append("runlength_ks_gt_0.50")
    elif all_m["ks"] > 0.25:
        reasons.append("runlength_ks_gt_0.25")
    if all_m["wasserstein"] > 50000:
        reasons.append("dwell_wasserstein_gt_50000")
    elif all_m["wasserstein"] > 10000:
        reasons.append("dwell_wasserstein_gt_10000")
    elif all_m["wasserstein"] > 1000:
        reasons.append("dwell_wasserstein_gt_1000")
    if c2st_auc > 0.75:
        reasons.append("mask_c2st_auc_gt_0.75")
    elif c2st_auc > 0.65:
        reasons.append("mask_c2st_auc_gt_0.65")

    if not reasons:
        qa_status = "pass"
        reasons = ["within_thresholds"]
    elif any(x in reasons for x in ["state_rate_error_gt_0.05", "runlength_ks_gt_0.50", "dwell_wasserstein_gt_50000", "mask_c2st_auc_gt_0.75"]):
        qa_status = "fatal"
    else:
        qa_status = "warning"

    if scope_excluded:
        publication_status = "scope_excluded"
        publication_blocker = False
        blocker_origin = "scope_excluded_trainval_predeclared"
    else:
        publication_status = qa_status
        publication_blocker = bool(qa_status == "fatal")
        blocker_origin = "new_12f_v5_test_qa_blocker" if qa_status == "fatal" else ("test_qa_warning" if qa_status == "warning" else "pass")

    rows.append({
        "col": col, "column": col, "semantic_mode_v5": mode,
        "selected_generator": srow.get("selected_generator", "") if hasattr(srow, "get") else "",
        "materializer_class_v5": srow.get("materializer_class_v5", "") if hasattr(srow, "get") else "",
        "v5_claim_status_trainval_only": srow.get("v5_claim_status_trainval_only", "") if hasattr(srow, "get") else "",
        "best_val_backtest_materializer": srow.get("best_val_backtest_materializer", "") if hasattr(srow, "get") else "",
        "best_val_backtest_status": srow.get("best_val_backtest_status", "") if hasattr(srow, "get") else "",
        "claim_included": claim_included,
        "scope_excluded": scope_excluded,
        "test_qa_status": qa_status,
        "mask_publication_status_after_12f5": publication_status,
        "mask_publication_blocker_after_12f5": publication_blocker,
        "blocker_origin": blocker_origin,
        "qa_reasons": ";".join(reasons),
        "real_test_state_one_rate": rs["state_one_rate"],
        "synthetic_state_one_rate": fs["state_one_rate"],
        "mask_rate_error": rate_error,
        "state_one_rate_error": rate_error,
        "real_test_transition_rate": rs["transition_rate"],
        "synthetic_transition_rate": fs["transition_rate"],
        "transition_error": transition_error,
        "runlength_ks": all_m["ks"],
        "dwell_wasserstein": all_m["wasserstein"],
        "one_run_ks": one_m["ks"],
        "one_dwell_wasserstein": one_m["wasserstein"],
        "zero_run_ks": zero_m["ks"],
        "zero_dwell_wasserstein": zero_m["wasserstein"],
        "test_all_run_max": rs["all_max"],
        "synthetic_all_run_max": fs["all_max"],
        "mask_c2st_auc": c2st_auc,
        "TEST_real_values_used_for_QA_only": True,
        "synthetic_values_mutated": False,
    })

qa = pd.DataFrame(rows)

metrics_path = REPORT_DIR_P / "cell12f5_mask_final_test_qa_metrics.csv"
status_path = REPORT_DIR_P / "cell12f5_mask_publication_status.csv"
blocker_path = REPORT_DIR_P / "cell12f5_mask_blocker_origin_audit.csv"
metrics_v5_path = REPORT_DIR_P / "cell12f_v5_mask_final_test_qa_metrics.csv"
summary_path = REPORT_DIR_P / "cell12f_v5_terminal_test_qa_summary.json"

qa.to_csv(metrics_path, index=False)
qa.to_csv(metrics_v5_path, index=False)
qa[[
    "col", "column", "semantic_mode_v5", "selected_generator", "materializer_class_v5",
    "v5_claim_status_trainval_only", "best_val_backtest_materializer", "best_val_backtest_status",
    "claim_included", "scope_excluded", "test_qa_status", "mask_publication_status_after_12f5",
    "mask_publication_blocker_after_12f5", "qa_reasons",
]].to_csv(status_path, index=False)
qa[["col", "column", "blocker_origin", "mask_publication_blocker_after_12f5", "qa_reasons"]].to_csv(blocker_path, index=False)

status_counts = qa["test_qa_status"].value_counts().to_dict()
publication_counts = qa["mask_publication_status_after_12f5"].value_counts().to_dict()
blocker_origin_counts = qa["blocker_origin"].value_counts().to_dict()
publication_blocker_n = int(qa["mask_publication_blocker_after_12f5"].sum())

summary = {
    "cell": "12.f.V5.4",
    "role": "observability_mask_v5_terminal_TEST_QA_only",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "targets": int(len(qa)),
    "pass": int(status_counts.get("pass", 0)),
    "warning": int(status_counts.get("warning", 0)),
    "fatal": int(status_counts.get("fatal", 0)),
    "publication_status_counts": publication_counts,
    "publication_blocker_n": publication_blocker_n,
    "blocker_origin_counts": blocker_origin_counts,
    "mean_metrics": {
        "mean_mask_rate_error": float(qa["mask_rate_error"].mean()),
        "mean_runlength_ks": float(qa["runlength_ks"].mean()),
        "mean_dwell_wasserstein": float(qa["dwell_wasserstein"].mean()),
        "mean_mask_c2st_auc": float(qa["mask_c2st_auc"].mean()),
    },
    "semantic_mode_counts": qa["semantic_mode_v5"].value_counts().to_dict(),
    "TEST_real_values_used_for_QA_only": True,
    "synthetic_values_mutated": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "post_TEST_repair_done_here": False,
    "split_drift_is_attribution_only": True,
    "same_df_te_v5_status": "development_only_unless_fresh_holdout_used",
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

for p in [CONTRACT_DIR_P / "cell12f5_mask_final_test_qa_contract_v1_1_THESIS.json", CONTRACT_DIR_P / "cell12f_v5_mask_final_test_qa_contract.json"]:
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12F_V5_QA = qa
CELL12F_V5_QA_SUMMARY = summary

_f5_log(f"V5.4 terminal QA complete | targets={len(qa)} | pass/warning/fatal={summary['pass']}/{summary['warning']}/{summary['fatal']} | publication_blocker_n={publication_blocker_n}")

if display is not None:
    display(pd.DataFrame([summary["mean_metrics"]]))
    display(qa.sort_values(["mask_publication_blocker_after_12f5", "dwell_wasserstein", "mask_rate_error"], ascending=[False, False, False]).head(40))

