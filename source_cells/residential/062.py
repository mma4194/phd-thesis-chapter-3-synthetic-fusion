import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "CELL12D_V6_TRAINVAL_CACHE" not in globals():
    raise RuntimeError("Run CELL 12.d.V6.0 first.")

rows = []

for col in CELL12D_V6_TARGETS:
    cache = CELL12D_V6_TRAINVAL_CACHE[col]
    tr = cache["tr_stats"]
    va = cache["va_stats"]
    tv = cache["tv_stats"]

    tr_rate = tr["rate"]
    va_rate = va["rate"]

    same_constant = tr["support"] in {"constant_0", "constant_1"} and tr["support"] == va["support"]
    train_or_val_empty = tr["support"] == "empty" or va["support"] == "empty"

    support_shift_constant_dynamic = (
        (tr["support"] in {"constant_0", "constant_1"} and va["support"] not in {tr["support"], "empty"})
        or (va["support"] in {"constant_0", "constant_1"} and tr["support"] not in {va["support"], "empty"})
    )

    support_shift_rare_to_opposite_constant = (
        (tr["support"] in {"rare_1", "ultra_rare_1"} and va["support"] == "constant_0")
        or (tr["support"] in {"rare_0", "ultra_rare_0"} and va["support"] == "constant_1")
    )

    rate_delta = abs(tr_rate - va_rate) if np.isfinite(tr_rate) and np.isfinite(va_rate) else np.nan
    rate_unstable = (
        np.isfinite(rate_delta)
        and rate_delta > max(0.02, 0.50 * max(tr_rate, va_rate, 1e-6))
    )

    enough_events = tv["one_count"] >= 10 and tv["zero_count"] >= 10
    enough_runs = tv["one_run_count"] >= 3 and tv["zero_run_count"] >= 3

    stable_dynamic = (
        tr["support"] == "both_states"
        and va["support"] == "both_states"
        and not rate_unstable
        and enough_events
        and enough_runs
    )

    stable_rare = (
        tr["support"] in {"rare_1", "ultra_rare_1", "rare_0", "ultra_rare_0"}
        and va["support"] in {"rare_1", "ultra_rare_1", "rare_0", "ultra_rare_0"}
        and not rate_unstable
        and enough_events
    )

    near_zero_consensus = (
        np.isfinite(tr_rate) and np.isfinite(va_rate)
        and tr_rate <= 0.001 and va_rate <= 0.001
        and tr["transition_rate"] <= 0.00005
        and va["transition_rate"] <= 0.00005
    )

    near_one_consensus = (
        np.isfinite(tr_rate) and np.isfinite(va_rate)
        and tr_rate >= 0.999 and va_rate >= 0.999
        and tr["transition_rate"] <= 0.00005
        and va["transition_rate"] <= 0.00005
    )

    if same_constant:
        family = "ExactConstantIncludedV6"
        materializer_class = "exact_constant"
        claim_status = "included_pass"
        reason = "TRAIN and VAL are exact same constant."
    elif train_or_val_empty:
        family = "ScopeExcludeInsufficientSupportV6"
        materializer_class = "scope_excluded_matrix_completion"
        claim_status = "scope_excluded"
        reason = "TRAIN or VAL has empty finite support."
    elif support_shift_constant_dynamic or support_shift_rare_to_opposite_constant:
        family = "ScopeExcludeSupportShiftV6"
        materializer_class = "scope_excluded_matrix_completion"
        claim_status = "scope_excluded"
        reason = "TRAIN/VAL support changes between constant/rare/dynamic states."
    elif near_zero_consensus:
        family = "NearZeroConstantIncludedWarningV6"
        materializer_class = "exact_zero_constant"
        claim_status = "included_warning"
        reason = "TRAIN/VAL near-zero consensus; exact zero materialization."
    elif near_one_consensus:
        family = "NearOneConstantIncludedWarningV6"
        materializer_class = "exact_one_constant"
        claim_status = "included_warning"
        reason = "TRAIN/VAL near-one consensus; exact one materialization."
    elif rate_unstable:
        family = "ScopeExcludeRateUnstableV6"
        materializer_class = "scope_excluded_matrix_completion"
        claim_status = "scope_excluded"
        reason = "TRAIN/VAL rate unstable; excluded from binary fidelity claim."
    elif stable_dynamic:
        family = "StableDynamicDwellIncludedV6"
        materializer_class = "trainval_tail_replay"
        claim_status = "included_pass"
        reason = "Stable dynamic TRAIN/VAL support; class-level V6 rule uses TRAIN/VAL tail replay."
    elif stable_rare:
        family = "StableRareEpisodeIncludedWarningV6"
        materializer_class = "trainval_circular_replay"
        claim_status = "included_warning"
        reason = "Stable rare TRAIN/VAL support; class-level V6 rule uses TRAIN/VAL circular replay."
    else:
        family = "ConservativeIncludedWarningOrA1V6"
        materializer_class = "trainval_block_bootstrap_32768"
        claim_status = "included_warning"
        reason = "Conservative warning-scoped fallback; class-level V6 rule uses 32k TRAIN/VAL block bootstrap."

    rows.append({
        "column": col,
        "selected_generator": family,
        "materializer_class_v6": materializer_class,
        "v6_claim_status_trainval_only": claim_status,
        "claim_included": bool(claim_status in {"included_pass", "included_warning"}),
        "scope_excluded": bool(claim_status == "scope_excluded"),
        "selection_reason_trainval_only": reason,
        "train_support": tr["support"],
        "val_support": va["support"],
        "train_rate": tr_rate,
        "val_rate": va_rate,
        "trainval_rate": tv["rate"],
        "train_transition_rate": tr["transition_rate"],
        "val_transition_rate": va["transition_rate"],
        "train_val_rate_delta": rate_delta,
        "rate_unstable_trainval_only": bool(rate_unstable),
        "support_shift_constant_dynamic_trainval_only": bool(support_shift_constant_dynamic),
        "support_shift_rare_to_opposite_constant_trainval_only": bool(support_shift_rare_to_opposite_constant),
        "near_zero_consensus_trainval_only": bool(near_zero_consensus),
        "near_one_consensus_trainval_only": bool(near_one_consensus),
        "stable_dynamic_trainval_only": bool(stable_dynamic),
        "stable_rare_trainval_only": bool(stable_rare),
        "TEST_real_values_used": False,
    })

selection_df = pd.DataFrame(rows)

selection_path = REPORT_DIR_P / "cell12d6_locked_binary_v6_claim_scoped_selection.csv"
legacy_selection_path = REPORT_DIR_P / "cell12d3_locked_binary_selection.csv"
selection_df.to_csv(selection_path, index=False)
selection_df.to_csv(legacy_selection_path, index=False)

registry_path = REPORT_DIR_P / "cell12d6_binary_v6_claim_scope_registry.csv"
selection_df[[
    "column",
    "selected_generator",
    "materializer_class_v6",
    "v6_claim_status_trainval_only",
    "claim_included",
    "scope_excluded",
    "selection_reason_trainval_only",
]].to_csv(registry_path, index=False)

summary = {
    "cell": "12.d.V6.1",
    "role": "binary_v6_trainval_only_claim_scope_and_class_level_materializer_selector",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "n_targets": int(len(selection_df)),
    "included_n_trainval_only": int(selection_df["claim_included"].sum()),
    "scope_excluded_n_trainval_only": int(selection_df["scope_excluded"].sum()),
    "selected_generator_counts": selection_df["selected_generator"].value_counts().to_dict(),
    "materializer_class_counts": selection_df["materializer_class_v6"].value_counts().to_dict(),
    "claim_status_counts_trainval_only": selection_df["v6_claim_status_trainval_only"].value_counts().to_dict(),
    "TEST_real_values_used": False,
    "selection_done_here": True,
    "generator_fit_done_here": False,
    "synthetic_values_mutated": False,
    "per_column_TEST_winners_used": False,
    "development_only_on_current_df_te": True,
}
summary_path = REPORT_DIR_P / "cell12d_v6_selection_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

contract = {
    **summary,
    "contract_version": "v6_0_THESIS",
    "policy": {
        "claim_scope_predeclared_from_trainval_only": True,
        "scope_excluded_columns_not_counted_as_publication_failures_for_included_claim": True,
        "full_matrix_still_materialized_for_pipeline_completeness": True,
        "class_level_materializer_rules_used": True,
        "per_column_test_winner_selection_forbidden": True,
    },
}
contract_path = CONTRACT_DIR_P / "cell12d6_binary_selection_contract_v6_0_THESIS.json"
contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

CELL12D_V6_SELECTION_DF = selection_df
CELL12D_V6_SELECTION_PATH = str(selection_path)
CELL12D_V6_CLAIM_SCOPE_REGISTRY_PATH = str(registry_path)
CELL12D_V6_SELECTION_SUMMARY = summary

_bv6_log(
    f"V6.1 selection/scope complete | included={summary['included_n_trainval_only']} | "
    f"scope_excluded={summary['scope_excluded_n_trainval_only']} | "
    f"materializer_classes={summary['materializer_class_counts']}"
)
_bv6_log(f"Selection: {selection_path}")
_bv6_log(f"Scope registry: {registry_path}")
_bv6_log(f"Contract: {contract_path}")

if display is not None:
    display(selection_df["selected_generator"].value_counts().rename_axis("selected_generator").reset_index(name="n"))
    display(selection_df["materializer_class_v6"].value_counts().rename_axis("materializer_class_v6").reset_index(name="n"))
    display(selection_df["v6_claim_status_trainval_only"].value_counts().rename_axis("v6_claim_status_trainval_only").reset_index(name="n"))