# ==========================================================
# MANIFEST — SequenceDataset helper contract
# ==========================================================
SEQUENCE_DATASET_HELPER_CONTRACT = {
    "component": "SequenceDataset",
    "component_type": "core_sequence_window_dataset_helper",
    "generator_family": "none_dataset_helper_only",
    "branch_scope": [
        "protocol_ddpm_candidate_generation",
        "continuous_ddpm_candidate_generation",
        "sequence_model_candidate_generation",
    ],
    "parameters": {
        "seq_len_source": "CFG['ddpm_seq_len'] or caller-provided",
        "stride_source": "CFG['ddpm_stride'] or caller-provided",
        "expected_tod_dim": 2,
        "x_obs_binary_required": True,
        "require_finite_X_default": True,
        "debug_reference_mask_check_available": True,
    },
    "seed": "caller-provided; default 1337",
    "train_fit_scope": "none_in_this_helper",
    "val_selection_metric": "none_in_this_helper",
    "test_only_qa": "none_in_this_helper",
    "contributes_to_final_artifact": False,
    "final_artifact_contribution_rule": (
        "This class may support training/evaluating candidate sequence generators, "
        "but it does not itself generate, select, repair, overwrite, or release any artifact."
    ),
    "leakage_status": {
        "reads_train_values": "caller_dependent",
        "reads_val_values": "caller_dependent",
        "reads_test_values": "caller_dependent",
        "uses_test_for_selection": False,
        "uses_test_for_repair": False,
        "mutates_synthetic_artifact": False,
    },
    "paper_claim_status": (
        "Not citable as quality evidence. Only cite downstream selected branch results "
        "whose TRAIN/VAL/TEST contracts are recorded separately."
    ),
}

if "RUN_META" in globals():
    RUN_META.setdefault("helper_contracts", {})
    RUN_META["helper_contracts"]["SequenceDataset"] = SEQUENCE_DATASET_HELPER_CONTRACT

log("[SequenceDataset] helper contract registered; class kept as non-claim-supporting infrastructure.")