import json
from pathlib import Path
from datetime import datetime, timezone

if "CELL12D_V6_TEST_QA_SUMMARY" not in globals():
    summary_path = REPORT_DIR_P / "cell12d_v6_terminal_test_qa_summary.json"
    if not summary_path.exists():
        raise RuntimeError("Run CELL 12.d.V6.5 first.")
    CELL12D_V6_TEST_QA_SUMMARY = json.loads(summary_path.read_text(encoding="utf-8"))

freeze = {
    "cell": "12.d.V6.6",
    "role": "binary_v6_terminal_TEST_freeze_no_post_TEST_repair",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "branch": "Binary V6 development claim-scoped class-level hybrid",
    "TEST_real_values_have_been_inspected": True,
    "TEST_real_values_used_for_QA_only": True,
    "synthetic_values_mutated": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "post_TEST_repair_allowed_in_this_run": False,
    "post_TEST_threshold_change_allowed_in_this_run": False,
    "post_TEST_generator_family_change_allowed_in_this_run": False,
    "post_TEST_target_exclusion_allowed_in_this_run": False,
    "per_column_TEST_winner_selection_used": False,
    "class_level_materializer_rules_used": True,
    "result_interpretation": "development_only_on_current_df_te_unless_fresh_holdout_used",
    "claim_scope": {
        "included_n_trainval_only": CELL12D_V6_TEST_QA_SUMMARY.get("included_n_trainval_only"),
        "scope_excluded_n_trainval_only": CELL12D_V6_TEST_QA_SUMMARY.get("scope_excluded_n_trainval_only"),
        "scope_exclusion_predeclared_from_trainval_only": True,
    },
    "included_terminal_TEST_counts": {
        "pass": CELL12D_V6_TEST_QA_SUMMARY.get("included_pass"),
        "warning": CELL12D_V6_TEST_QA_SUMMARY.get("included_warning"),
        "fatal": CELL12D_V6_TEST_QA_SUMMARY.get("included_fatal"),
        "publication_blocker_n": CELL12D_V6_TEST_QA_SUMMARY.get("included_publication_blocker_n"),
    },
    "scope_excluded_diagnostic_counts": CELL12D_V6_TEST_QA_SUMMARY.get("scope_excluded_diagnostic_counts"),
    "all_columns_raw_test_qa_counts": CELL12D_V6_TEST_QA_SUMMARY.get("all_columns_raw_test_qa_counts"),
    "included_mean_metrics": {
        "mean_rate_error": CELL12D_V6_TEST_QA_SUMMARY.get("mean_rate_error_included"),
        "mean_transition_error": CELL12D_V6_TEST_QA_SUMMARY.get("mean_transition_error_included"),
        "mean_runlength_ks": CELL12D_V6_TEST_QA_SUMMARY.get("mean_runlength_ks_included"),
        "mean_dwell_wasserstein": CELL12D_V6_TEST_QA_SUMMARY.get("mean_dwell_wasserstein_included"),
        "mean_binary_c2st_auc": CELL12D_V6_TEST_QA_SUMMARY.get("mean_binary_c2st_auc_included"),
    },
}

freeze_path = CONTRACT_DIR_P / "cell12d_v6_freeze_no_post_TEST_repair_contract_v1_0_THESIS.json"
freeze_summary_path = REPORT_DIR_P / "cell12d_v6_freeze_no_post_TEST_repair_summary.json"
freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
freeze_summary_path.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")

CFG["cell12d_binary_v6_terminal_test_frozen"] = True
CFG["cell12d_binary_v6_post_test_repair_allowed"] = False
CFG["cell12d_binary_v6_claim_scoped"] = True

CELL12D_V6_FREEZE_CONTRACT = freeze
CELL12D_V6_FREEZE_CONTRACT_PATH = str(freeze_path)

_bv6_log("V6 terminal result frozen. No post-TEST binary repair allowed.")
_bv6_log(f"Contract: {freeze_path}")
print(json.dumps(freeze, indent=2, sort_keys=True))