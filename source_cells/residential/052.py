# ==========================================================
# FRESH CONTINUOUS CANDIDATE CONFIGURATION
# Regenerate every candidate and save current-run intermediate arrays.
# ==========================================================

from pathlib import Path
import os

CFG = dict(CFG)

# Final scientifically valid mode.
# Do NOT use route_only/materialize_subset as final evidence.
CFG["cell12c2_mode"] = "full"

# The historical flag enables writing new arrays. The loader is disabled.
CFG["cell12c2_use_candidate_cache"] = True
CFG["cell12c2_force_recompute"] = True

# Keep generated intermediate arrays inside this new scratch run.
CFG["cell12c2_cache_dir"] = str(Path(ARTDIR) / "cell12c2_candidate_cache_v8_14")

# Keep final evaluation strength unchanged.
CFG.setdefault("cell12c2_audit_max_points", 50000)
CFG.setdefault("cell12c2_c2st_max_per_class", 4096)
CFG.setdefault("cell12c2_quantile_n", 401)

print("Cell 12.c.2 mode:", CFG["cell12c2_mode"])
print("Cell 12.c.2: all candidates are regenerated; previous arrays are not loaded.")
print("Cell 12.c.2 current-run intermediate folder:", CFG["cell12c2_cache_dir"])
assert not any(Path(CFG["cell12c2_cache_dir"]).rglob("*.npy")), "Candidate artifacts already exist: start a clean run."
