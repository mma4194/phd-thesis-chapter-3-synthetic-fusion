# %% CELL EXT.DRV.GATE — corrected driver-gate summary, no legacy release claim
led = CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF.copy()
print("[DRV.GATE] Corrected ratio-scale status counts:", led["driver_corrected_ratio_battery_status"].value_counts().to_dict())
print("[DRV.GATE] Q6 release eligible:", int(led["driver_release_eligible_corrected"].sum()), "of", len(led))
print("[DRV.GATE] Legacy 12.e.6 status is diagnostic only and must not drive Q6.")
