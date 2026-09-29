# %% CELL EXT.DRV.REGRADE2 — final corrected ledger already materialized pre-Q6
led = CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF.copy()
assert len(led) == int(CFG.get("expected_iot_sparse_driver_cols", 26)), "driver ledger row-count mismatch"
assert int(led["driver_release_eligible_corrected"].sum()) == int(CFG.get("expected_public_sparse_driver_cols", 23)), "driver release count mismatch"
print("[REGRADE2] PRIMARY corrected battery:", led["driver_corrected_ratio_battery_status"].value_counts().to_dict())
print("[REGRADE2] ledger:", CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_CSV)
