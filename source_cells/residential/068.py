# %% CELL 12.d.V6.7 — Binary V6 decision: preferred development candidate, freeze
# Purpose:
#   Record that Binary V6 is the preferred binary development candidate so far,
#   but must not be claimed as blind final evidence on the already-inspected df_te.

import json
from pathlib import Path
from datetime import datetime, timezone

root = Path(OUTDIR).expanduser().resolve()
report_dir = Path(REPORT_DIR).expanduser().resolve()
contract_dir = root / "artifacts" / "contracts"
report_dir.mkdir(parents=True, exist_ok=True)
contract_dir.mkdir(parents=True, exist_ok=True)

decision = {
    "cell": "12.d.V6.7",
    "role": "binary_v6_preferred_development_candidate_decision",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "branch": "Binary V6 claim-scoped class-level hybrid",
    "scientific_status": "preferred_development_candidate_requires_fresh_holdout_for_final_claims",
    "merge_into_current_main_artifact_without_fresh_holdout": False,
    "reason": (
        "Binary V6 substantially improves the claim-scoped binary branch: "
        "39 TRAIN/VAL-included targets, 13 TRAIN/VAL-scope-excluded targets, "
        "and terminal current-df_te included QA of 27 pass, 9 warning, 3 fatal. "
        "It also improves included transition error, runlength KS, and C2ST relative to V5. "
        "However, V6 was developed after repeated inspection of the same df_te across V2–V6, "
        "so final STUDY-THESIS claims require a fresh untouched temporal holdout."
    ),
    "leakage_policy": {
        "TEST_real_values_used_for_selection_inside_V6_cells": False,
        "TEST_real_values_used_for_generator_fit_inside_V6_cells": False,
        "TEST_real_values_used_for_materialization_inside_V6_cells": False,
        "TEST_real_values_used_for_terminal_QA": True,
        "current_TEST_has_influenced_binary_development_context": True,
        "post_TEST_repair_allowed": False,
        "post_TEST_threshold_change_allowed": False,
        "post_TEST_target_exclusion_allowed": False,
        "post_TEST_merge_allowed_without_fresh_holdout": False,
    },
    "v6_terminal_current_df_te_result": {
        "included_n": 39,
        "scope_excluded_n": 13,
        "included_pass": 27,
        "included_warning": 9,
        "included_fatal": 3,
        "included_publication_blocker_n": 3,
        "mean_rate_error_included": 0.006508088452206795,
        "mean_transition_error_included": 0.00013114561970118267,
        "mean_runlength_ks_included": 0.1158421721781378,
        "mean_dwell_wasserstein_included": 10406.98462404232,
        "mean_binary_c2st_auc_included": 0.5069946815636471,
    },
    "comparison_to_v5": {
        "v5_included_pass": 19,
        "v5_included_warning": 8,
        "v5_included_fatal": 12,
        "v5_included_publication_blocker_n": 12,
        "v5_mean_runlength_ks_included": 0.31701388531601915,
        "v5_mean_binary_c2st_auc_included": 0.5857849047504221,
        "improvement_summary": (
            "V6 reduces included fatal/blocker count from 12 to 3, improves mean runlength KS "
            "from 0.317 to 0.116, and improves mean C2ST AUC from 0.586 to 0.507."
        ),
    },
    "decision": {
        "preferred_binary_development_candidate": "V6",
        "continue_iterating_on_same_df_te_for_final_claims": False,
        "fresh_holdout_required_for_final_binary_claim": True,
        "paper_safe_claim_before_fresh_holdout": (
            "Binary V6 is a development candidate showing strong improvement under a claim-scoped, "
            "TRAIN/VAL-predeclared binary subset. It is not blind final evidence until evaluated "
            "once on a fresh untouched temporal holdout."
        ),
    },
}

decision_path = contract_dir / "cell12d_v6_preferred_development_candidate_requires_fresh_holdout_contract.json"
summary_path = report_dir / "cell12d_v6_preferred_development_candidate_requires_fresh_holdout_summary.json"

decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
summary_path.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

CFG["cell12d_binary_v6_preferred_development_candidate"] = True
CFG["cell12d_binary_v6_requires_fresh_holdout_for_final_claims"] = True
CFG["cell12d_binary_v6_merge_without_fresh_holdout"] = False

print("[Cell12.d.V6.7] Binary V6 frozen as preferred development candidate. Fresh holdout required for final claims.")
print("Decision contract:", decision_path)
print("Decision summary:", summary_path)