# Reproduction adapter: fixed policy selection; original source retained in 016.py.
# ============================================================
# PATCH — Make strict absolute-Fano 6/14/6 sensitivity explicit
# Place after STRONG.4 / Fano diagnostic and before packaging
# ============================================================

from pathlib import Path
import os
import shutil
import pandas as pd

RUN_ROOT = Path(os.environ["CPS_CANONICAL_RUN"])
strong_tables = RUN_ROOT / "reports/thesis_strong_revision_fixes/tables"

composition_path = strong_tables / "q6_release_composition_by_policy.csv"
grid_path = strong_tables / "q6_sparse_driver_absolute_fano_sensitivity_grid.csv"

composition = pd.read_csv(composition_path)
grid = pd.read_csv(grid_path)

# Use the paper's declared thresholds directly. Observed counts are outputs.
# Never search the grid for a row matching expected pass/warning/fatal counts.
strict = grid[
    (grid["abs_fano_warn_threshold"].round(10) == 0.1) &
    (grid["abs_fano_fatal_threshold"].round(10) == 0.5)
].copy()
if len(strict) != 1:
    raise RuntimeError("Expected exactly one predeclared Fano policy: warn=0.1, fatal=0.5")
strict_row = strict.iloc[0]

strict_public_sparse = int(strict_row["release_eligible_drivers"])
strict_public_total = int(strict_row["public_total_if_only_fatal_excluded"])

new_row = {
    "policy": (
        "absolute_fano_sensitivity_strict_"
        f"warn_{strict_row['abs_fano_warn_threshold']:g}_"
        f"fatal_{strict_row['abs_fano_fatal_threshold']:g}"
    ),
    "sparse_driver_pass": int(strict_row["pass"]),
    "sparse_driver_warning": int(strict_row["warning"]),
    "sparse_driver_fatal": int(strict_row["fatal"]),
    "sparse_driver_public_kept": strict_public_sparse,
    "timestamp_public_kept": 1,
    "router_public_kept": 18,
    "zigbee_public_kept": 4,
    "public_logical_columns": strict_public_total,
    "excluded_logical_columns": 555 - strict_public_total,
    "claim_interpretation": (
        "fixed strict absolute-Fano policy (warn=0.1, fatal=0.5); measured counts "
        "must be compared with the paper after computation; shown for sensitivity, "
        "not recommended as the public-release policy after the sequence-risk gate"
    ),
}

# Remove old inserted strict rows if the cell is rerun.
composition = composition[
    ~composition["policy"].astype(str).str.contains(
        "absolute_fano_sensitivity_strict_", regex=False, na=False
    )
].copy()

# Insert strict row immediately after existing ratio-scale policy.
rows = []
inserted = False
for _, r in composition.iterrows():
    rows.append(r.to_dict())
    if r["policy"] == "existing_ratio_scale_policy":
        rows.append(new_row)
        inserted = True

if not inserted:
    rows.insert(1, new_row)

composition_patched = pd.DataFrame(rows)

# Keep a stable, reviewer-readable order.
preferred_order = {
    "existing_ratio_scale_policy": 0,
    new_row["policy"]: 1,
    "absolute_fano_sensitivity_warn_0.5_fatal_1.0": 2,
    "conservative_sequence_risk_gate_recommended": 3,
}
composition_patched["_order"] = composition_patched["policy"].map(preferred_order).fillna(99)
composition_patched = (
    composition_patched
    .sort_values(["_order", "policy"])
    .drop(columns=["_order"])
    .reset_index(drop=True)
)

# Back up and overwrite the original table so existing package-index references remain valid.
backup_path = composition_path.with_suffix(".pre_strict_fano_patch.csv")
if not backup_path.exists():
    shutil.copy2(composition_path, backup_path)

composition_patched.to_csv(composition_path, index=False)

# Also write a named manuscript-facing copy.
manuscript_path = strong_tables / "q6_release_composition_by_policy_MANUSCRIPT_READY.csv"
composition_patched.to_csv(manuscript_path, index=False)

print("Patched:", composition_path)
print("Backup:", backup_path)
print("Manuscript copy:", manuscript_path)
display(composition_patched)