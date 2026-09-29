# %% STRONG.2 — Locate canonical artifacts and fail-closed path audit

PATHS = {
    "real_train": first_existing([REAL_SPLIT_DIR / "REAL_TRAIN_SPLIT.parquet", SYN_DIR / "REAL_TRAIN_SPLIT.parquet"]),
    "real_val": first_existing([REAL_SPLIT_DIR / "REAL_VAL_SPLIT.parquet", SYN_DIR / "REAL_VAL_SPLIT.parquet"]),
    "real_test": first_existing([os.environ.get("CPS_REAL_TEST_PARQUET", ""), REAL_SPLIT_DIR / "REAL_TEST_SPLIT.parquet", SYN_DIR / "REAL_TEST_SPLIT.parquet"]),
    "syn_scientific": first_existing([SYN_DIR / "CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet", SYN_DIR / "CPS_SYNTHETIC_TEST_SCIENTIFIC.parquet"]),
    "syn_public_existing": first_existing([SYN_DIR / "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet"]),
    "q6_registry": first_existing([REPORT_DIR / "cell15_6_q6_public_column_registry.csv"]),
    "q6_dropped": first_existing([REPORT_DIR / "cell15_6_q6_public_dropped_columns.csv"]),
    "driver_ledger": first_existing([REPORT_DIR / "cell12e6_driver_corrected_release_ledger.csv"]),
    "paper_numbers": first_existing([REPORT_DIR / "paper_numbers_summary.json"]),
    "role_ownership": first_existing([ARTIFACT_DIR / "iot_role_ownership.csv", REPORT_DIR / "iot_role_ownership.csv"]),
    "continuous_selection": first_existing([REPORT_DIR / "cell12c3R_locked_iot_value_selection.csv"]),
    "continuous_qa": first_existing([REPORT_DIR / "cell12c6R_final_test_qa_metrics.csv", REPORT_DIR / "cell12c6_iot_value_final_test_qa_metrics.csv"]),
    "ext_c_windows": first_existing([EXT_OUT / "tables" / "EXT_C_window_metrics_v3.csv", EXT_OUT / "tables" / "EXT_C_window_metrics.csv"]),
    "ext_c_agreement": first_existing([EXT_OUT / "tables" / "EXT_C2_instrument_agreement.csv"]),
    "q5_5seed": first_existing([EXT_OUT / "tables" / "EXT_A_regime_consequence_5seed.csv"]),
    "q5_5seed_summary": first_existing([EXT_OUT / "tables" / "EXT_A_gap_summary_5seed.csv"]),
    "baseline_summary": first_existing([REPORT_DIR / "cell17_6_baseline_publication_safe_summary.csv", REPORT_DIR / "baseline_fairness_comparison_ledger.csv"]),
    "postrun_baseline_facts": first_existing([REPORT_DIR / "review_fix_outputs" / "baseline_budget_split_facts_from_contracts.csv"]),
    "postrun_patched_index": first_existing([REPORT_DIR / "artifact_package_index_REVIEW_FIXES_PATCHED.csv"]),
}

required = ["real_test", "syn_scientific", "syn_public_existing", "q6_registry", "driver_ledger"]
missing = [k for k in required if PATHS.get(k) is None]
if missing:
    raise FileNotFoundError(f"Missing required artifacts: {missing}. Run the final governed notebook before this notebook.")

rows = []
for k, p in PATHS.items():
    rows.append({
        "key": k,
        "path": rel_to_run(p) if p else "",
        "exists": bool(p and Path(p).exists()),
        "sha256": sha256_file(p) if p and Path(p).is_file() else "",
    })
path_audit = pd.DataFrame(rows)
write_csv(path_audit, STRONG_MANIFESTS / "strong_fix_path_resolution_audit.csv")
path_audit
