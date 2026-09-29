def run_small_self_checks():
    t = np.arange(5, dtype=np.int64) + 1750446300
    s = pd.Series([0.0, 1.0, 0.0000001, 1.0, 0.0])
    row, detail, samples, windows = profile_feature("precision_check", s, t, 0, CFG_INSPECT)
    assert row["observed_representation"] == "real_valued_numeric", "Do not round near-binary values."
    row = profile_feature("missing_check", pd.Series([1.0, None, 1.0, 0.0, 1.0]), t, 0, CFG_INSPECT)[0]
    assert row["finite_adjacent_pairs"] == 2 and row["positive_run_count"] == 3
    assert show_value(np.int64(9007199254740993)) == "9007199254740993"
    first = stable_rng("same_feature", "sample", SEED).integers(0, 100, 10)
    second = stable_rng("same_feature", "sample", SEED).integers(0, 100, 10)
    assert np.array_equal(first, second)
    print("PASS: precision, missing-gap/run handling, exact integer display and repeatable seeds.")
run_small_self_checks()
