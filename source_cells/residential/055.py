# ==========================================================
# CLEANUP BEFORE RERUNNING CELL 12.c.3
# Purpose:
#   Force Cell 12.c.3 to use the canonical disk Cell 12.c.2 metrics
#   instead of stale in-memory globals.
# ==========================================================

_STALE_12C3_INPUT_GLOBALS = [
    "CELL12C2_ALL_VAL_CANDIDATE_METRICS_DF",
    "CANONICAL_CANDIDATE_METRICS_DF",
    "CELL12C2_PUBLICATION_ELIGIBILITY_AUDIT_DF",
    "CELL12C2_VAL_CANDIDATE_AUDIT_DF",
    "CELL12C2_CANDIDATE_INVENTORY_DF",
    "CELL12C3_SELECTION_DF",
    "CELL12C3_SELECTED_GENERATORS_DF",
    "CELL12C3_SELECTION_SUMMARY",
]

_removed = []

for _name in _STALE_12C3_INPUT_GLOBALS:
    if _name in globals():
        del globals()[_name]
        _removed.append(_name)
        print(f"Deleted stale global: {_name}")

print(f"Cleanup done. Removed {len(_removed)} stale globals.")
print("Now rerun Cell 12.c.3.")