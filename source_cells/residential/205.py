# %% FIG1-AUDIT v3 — fitted-family audit from locked selection registries, no hard-coded notebook path
from pathlib import Path
import pandas as pd
R = lambda p: Path(str(globals().get("REPORT_DIR", ARTIFACT_ROOT/"reports"))) / p
print("[FIG1-AUDIT] Registry-backed family summary. Figure 1 should describe branch responsibilities, not claim one generator family per branch.")
cont_p = R("cell12c3R_locked_iot_value_selection.csv")
if cont_p.exists():
    cont = pd.read_csv(cont_p)
    gcol = next((c for c in ["candidate_generator", "generator", "selected_generator"] if c in cont.columns), None)
    if gcol:
        print("CONTINUOUS selected families:")
        print(cont[gcol].value_counts().to_string())
drv_p = R("cell12e5_driver_publication_column_registry.csv")
if drv_p.exists():
    drv = pd.read_csv(drv_p)
    gcol = next((c for c in drv.columns if "generator" in c.lower()), None)
    if gcol:
        print("\nDRIVER selected families:")
        print(drv[gcol].value_counts().to_string())
print("\n[FIG1-AUDIT] Use schematic labels: protocol tier-aware support/count materialization; continuous target-specific portfolio; binary rate/transition/dwell models; sparse-driver event block/renewal models; observability reporting/persistence models.")
