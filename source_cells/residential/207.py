# %% FIG1-TRUTH — authoritative fitted-family counts from the locked selection registries
import pandas as pd
R = lambda p: ARTIFACT_ROOT / "reports" / p
cont = pd.read_csv(R("cell12c3R_locked_iot_value_selection.csv"))
gcol = next(c for c in ["candidate_generator","generator"] if c in cont.columns)
print("CONTINUOUS (148 targets) — selected families:")
print(cont[gcol].value_counts().to_string(), "\n")

drv = pd.read_csv(R("cell12e5_driver_publication_column_registry.csv"))
print("DRIVERS (26 targets) — selected families; release eligibility is governed by corrected Cell 12.e.6R ledger:")
print(drv["selected_driver_generator"].value_counts().to_string(), "\n")

import glob
bin_cands = glob.glob(str(ARTIFACT_ROOT/"reports"/"cell12d*select*.csv")) + \
            glob.glob(str(ARTIFACT_ROOT/"reports"/"cell12d*registry*.csv"))
print("BINARY registry candidates:", [p.split("/")[-1] for p in bin_cands])
if bin_cands:
    b = pd.read_csv(bin_cands[0])
    gc = [c for c in b.columns if "generator" in c.lower()]
    if gc: print(b[gc[0]].value_counts().to_string())

if "CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF" in globals():
    print("\nCORRECTED DRIVER RELEASE STATUS:")
    print(CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF["driver_corrected_ratio_battery_status"].value_counts().to_string())
    print("release eligible:", len(DRIVER_RELEASE_ELIGIBLE_COLS_CORRECTED), "excluded fatal:", len(DRIVER_RELEASE_EXCLUDED_FATAL_COLS_CORRECTED))
