# %% CELL EXT.DRV.GATE2 — threshold geometry audit for corrected sparse-driver gates
led = CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF.copy()
ratio_cols = [c for c in ["event_rate_ratio", "burst_count_ratio", "windowed_fano_ratio"] if c in led.columns]
print("[DRV.GATE2] ratio thresholds: warning >=1.5 or <=2/3; fatal >=3 or <=1/3")
for c in ratio_cols:
    x = pd.to_numeric(led[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    print(f"[DRV.GATE2] {c}: min={x.min():.3f} median={x.median():.3f} max={x.max():.3f}")
