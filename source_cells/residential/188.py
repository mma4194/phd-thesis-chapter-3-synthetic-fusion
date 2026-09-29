# %% CELL EXT.DRV.Q2TRUE — corrected burst and windowed-Fano metrics from authoritative ledger
led = CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF.copy()
cols = [c for c in ["col", "real_burst_count", "syn_burst_count", "burst_count_ratio", "real_windowed_fano", "syn_windowed_fano", "windowed_fano_ratio", "driver_corrected_ratio_battery_status"] if c in led.columns]
print(led[cols].sort_values("burst_count_ratio", ascending=False).head(10).round(3).to_string(index=False))
