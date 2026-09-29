# %% CELL EXT.DRV.FINAL — corrected sparse-driver release result for manuscript
led = CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF.copy()
print("[DRV.FINAL] corrected status counts:", led["driver_corrected_ratio_battery_status"].value_counts().to_dict())
print("[DRV.FINAL] release eligible cols:", len(DRIVER_RELEASE_ELIGIBLE_COLS_CORRECTED))
print("[DRV.FINAL] release excluded fatal cols:", DRIVER_RELEASE_EXCLUDED_FATAL_COLS_CORRECTED)
