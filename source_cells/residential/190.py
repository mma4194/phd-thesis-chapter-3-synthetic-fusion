# %% CELL EXT.DRV.FAM — generator families for corrected-fatal drivers
led = CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF.copy()
fatal_cols = set(led.loc[led["driver_corrected_ratio_battery_status"].eq("fatal"), "col"].astype(str))
reg_path = Path(str(REPORT_DIR)) / "cell12e5_driver_publication_column_registry.csv"
if reg_path.exists():
    reg = pd.read_csv(reg_path)
    key = next((c for c in reg.columns if c.lower() in {"col", "column", "target"}), None)
    gen = next((c for c in reg.columns if "generator" in c.lower()), None)
    if key and gen:
        print(reg[reg[key].astype(str).isin(fatal_cols)][[key, gen]].to_string(index=False))
else:
    print("[DRV.FAM] registry not found:", reg_path)
