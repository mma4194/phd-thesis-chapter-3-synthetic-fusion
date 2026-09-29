import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "IOT_FINAL_BINARY_TEST" not in globals():
    p = OUT_SYN_P / "IOT_FINAL_BINARY_TEST.parquet"
    if not p.exists():
        raise RuntimeError("Run CELL 12.d.V6.3 first.")
    IOT_FINAL_BINARY_TEST = pd.read_parquet(p)

selection = CELL12D_V6_SELECTION_DF.set_index("column")

qa_rows = []

for i, col in enumerate(CELL12D_V6_TARGETS, start=1):
    if i == 1 or i % 10 == 0 or i == len(CELL12D_V6_TARGETS):
        _bv6_log(f"V6.5 progress {i}/{len(CELL12D_V6_TARGETS)} | col={col}")

    real_vals, real_finite = _bv6_bin_values(DF_TE[col]) if col in DF_TE.columns else (np.asarray([], dtype=np.uint8), 0.0)
    syn_vals, syn_finite = _bv6_bin_values(IOT_FINAL_BINARY_TEST[col]) if col in IOT_FINAL_BINARY_TEST.columns else (np.asarray([], dtype=np.uint8), 0.0)

    real_s = _bv6_stats(real_vals)
    syn_s = _bv6_stats(syn_vals)

    all_m = _bv6_state_run_metrics(real_vals, syn_vals, state=None)
    one_m = _bv6_state_run_metrics(real_vals, syn_vals, state=1)
    zero_m = _bv6_state_run_metrics(real_vals, syn_vals, state=0)

    rate_error = abs(real_s["rate"] - syn_s["rate"]) if np.isfinite(real_s["rate"]) and np.isfinite(syn_s["rate"]) else np.nan
    transition_error = abs(real_s["transition_rate"] - syn_s["transition_rate"]) if np.isfinite(real_s["transition_rate"]) and np.isfinite(syn_s["transition_rate"]) else np.nan
    c2st_auc = _bv6_c2st_auc(real_vals, syn_vals)

    selected_generator = str(selection.loc[col, "selected_generator"]) if col in selection.index else "unknown"
    materializer_class = str(selection.loc[col, "materializer_class_v6"]) if col in selection.index else "unknown"
    claim_status = str(selection.loc[col, "v6_claim_status_trainval_only"]) if col in selection.index else "unknown"
    claim_included = bool(selection.loc[col, "claim_included"]) if col in selection.index else False

    reasons = []
    if np.isfinite(rate_error) and rate_error > 0.10:
        reasons.append("rate_error_gt_0.10")
    elif np.isfinite(rate_error) and rate_error > 0.05:
        reasons.append("rate_error_gt_0.05")

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
        reasons.append("binary_c2st_auc_gt_0.75")
    elif c2st_auc > 0.65:
        reasons.append("binary_c2st_auc_gt_0.65")

    if not reasons:
        qa_status = "pass"
        reasons.append("within_thresholds")
    elif any(r in reasons for r in ["rate_error_gt_0.10", "runlength_ks_gt_0.50", "dwell_wasserstein_gt_50000", "binary_c2st_auc_gt_0.75"]):
        qa_status = "fatal"
    else:
        qa_status = "warning"

    if claim_included:
        publication_status = qa_status
        publication_scope = "included_claim"
    else:
        publication_status = "scope_excluded"
        publication_scope = "scope_excluded_trainval_predeclared"

    qa_rows.append({
        "column": col,
        "selected_generator": selected_generator,
        "materializer_class_v6": materializer_class,
        "v6_claim_status_trainval_only": claim_status,
        "claim_included": claim_included,
        "publication_scope": publication_scope,
        "test_qa_status": qa_status,
        "publication_status": publication_status,
        "qa_reasons": ";".join(reasons),
        "test_rate": real_s["rate"],
        "synthetic_rate": syn_s["rate"],
        "rate_error": rate_error,
        "test_transition_rate": real_s["transition_rate"],
        "synthetic_transition_rate": syn_s["transition_rate"],
        "transition_error": transition_error,
        "runlength_ks": all_m["ks"],
        "dwell_wasserstein": all_m["wasserstein"],
        "one_run_ks": one_m["ks"],
        "one_dwell_wasserstein": one_m["wasserstein"],
        "zero_run_ks": zero_m["ks"],
        "zero_dwell_wasserstein": zero_m["wasserstein"],
        "test_all_run_max": real_s["all_max"],
        "synthetic_all_run_max": syn_s["all_max"],
        "binary_c2st_auc": c2st_auc,
        "TEST_real_values_used_for_QA_only": True,
        "synthetic_values_mutated": False,
    })

qa = pd.DataFrame(qa_rows)

qa_path = REPORT_DIR_P / "cell12d6_binary_final_test_qa_metrics.csv"
qa_v6_path = REPORT_DIR_P / "cell12d_v6_terminal_test_qa_metrics.csv"
status_path = REPORT_DIR_P / "cell12d6_binary_publication_status.csv"
included_path = REPORT_DIR_P / "cell12d_v6_included_claim_terminal_test_qa_metrics.csv"
scope_excluded_path = REPORT_DIR_P / "cell12d_v6_scope_excluded_terminal_test_diagnostics.csv"

qa.to_csv(qa_path, index=False)
qa.to_csv(qa_v6_path, index=False)
qa.loc[qa["claim_included"]].to_csv(included_path, index=False)
qa.loc[~qa["claim_included"]].to_csv(scope_excluded_path, index=False)
qa[[
    "column",
    "selected_generator",
    "materializer_class_v6",
    "v6_claim_status_trainval_only",
    "claim_included",
    "publication_scope",
    "test_qa_status",
    "publication_status",
    "qa_reasons",
]].to_csv(status_path, index=False)

included = qa["claim_included"]
included_counts = qa.loc[included, "test_qa_status"].value_counts().to_dict()
all_counts = qa["test_qa_status"].value_counts().to_dict()
scope_counts = qa.loc[~included, "test_qa_status"].value_counts().to_dict()

included_blocker_n = int((qa.loc[included, "publication_status"] == "fatal").sum())
scope_excluded_n = int((~included).sum())

def _mean_subset(col, mask):
    vals = pd.to_numeric(qa.loc[mask, col], errors="coerce")
    return float(vals.mean()) if len(vals) else None

summary = {
    "cell": "12.d.V6.5",
    "role": "binary_v6_terminal_TEST_QA_only_claim_scoped",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "targets_total": int(len(qa)),
    "included_n_trainval_only": int(included.sum()),
    "scope_excluded_n_trainval_only": scope_excluded_n,

    "included_pass": int(included_counts.get("pass", 0)),
    "included_warning": int(included_counts.get("warning", 0)),
    "included_fatal": int(included_counts.get("fatal", 0)),
    "included_publication_blocker_n": included_blocker_n,

    "scope_excluded_diagnostic_counts": scope_counts,
    "all_columns_raw_test_qa_counts": all_counts,

    "selected_generator_counts_all": qa["selected_generator"].value_counts().to_dict(),
    "selected_generator_counts_included": qa.loc[included, "selected_generator"].value_counts().to_dict(),
    "materializer_class_counts_all": qa["materializer_class_v6"].value_counts().to_dict(),
    "materializer_class_counts_included": qa.loc[included, "materializer_class_v6"].value_counts().to_dict(),

    "mean_rate_error_included": _mean_subset("rate_error", included),
    "mean_transition_error_included": _mean_subset("transition_error", included),
    "mean_runlength_ks_included": _mean_subset("runlength_ks", included),
    "mean_dwell_wasserstein_included": _mean_subset("dwell_wasserstein", included),
    "mean_binary_c2st_auc_included": _mean_subset("binary_c2st_auc", included),

    "mean_rate_error_all": _mean_subset("rate_error", qa.index == qa.index),
    "mean_transition_error_all": _mean_subset("transition_error", qa.index == qa.index),
    "mean_runlength_ks_all": _mean_subset("runlength_ks", qa.index == qa.index),
    "mean_dwell_wasserstein_all": _mean_subset("dwell_wasserstein", qa.index == qa.index),
    "mean_binary_c2st_auc_all": _mean_subset("binary_c2st_auc", qa.index == qa.index),

    "TEST_real_values_used_for_QA_only": True,
    "synthetic_values_mutated": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "per_column_TEST_winners_used": False,
    "development_only_on_current_df_te": True,
}
summary_path = REPORT_DIR_P / "cell12d_v6_terminal_test_qa_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

contract = {
    **summary,
    "contract_version": "v6_0_THESIS",
    "safe_interpretation": (
        "Terminal TEST QA only. Scope-excluded columns were predeclared using TRAIN/VAL only. "
        "Class-level materializer rules were chosen from V6 diagnostic family-level evidence, not per-column TEST winners. "
        "Because this same df_te has already informed Binary V2/V3/V4/V5/V6 development context, "
        "results are development evidence unless rerun on a fresh holdout."
    ),
}
contract_path = CONTRACT_DIR_P / "cell12d6_binary_final_test_qa_contract_v6_0_THESIS.json"
contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

CELL12D_V6_TEST_QA = qa
CELL12D_V6_TEST_QA_SUMMARY = summary
CELL12D_V6_TEST_QA_PATH = str(qa_v6_path)

_bv6_log(
    f"V6.5 terminal TEST QA complete | total={summary['targets_total']} | "
    f"included={summary['included_n_trainval_only']} | scope_excluded={summary['scope_excluded_n_trainval_only']} | "
    f"included pass/warning/fatal={summary['included_pass']}/{summary['included_warning']}/{summary['included_fatal']} | "
    f"included_blockers={summary['included_publication_blocker_n']}"
)
_bv6_log(
    f"Included metric summary | mean_rate_error={summary['mean_rate_error_included']:.6f} | "
    f"mean_transition_error={summary['mean_transition_error_included']:.6f} | "
    f"mean_runlength_ks={summary['mean_runlength_ks_included']:.6f} | "
    f"mean_dwell_wasserstein={summary['mean_dwell_wasserstein_included']:.6f} | "
    f"mean_binary_c2st_auc={summary['mean_binary_c2st_auc_included']:.6f}"
)
_bv6_log(f"Contract: {contract_path}")

if display is not None:
    display(
        qa.sort_values(
            ["claim_included", "publication_status", "dwell_wasserstein", "rate_error"],
            ascending=[False, True, False, False],
        ).head(40)
    )