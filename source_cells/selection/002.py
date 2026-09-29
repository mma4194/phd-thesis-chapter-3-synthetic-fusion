# %% STRONG.0 — User configuration and policy switches
# Edit only this cell when moving between machines/runs.

from pathlib import Path
import os

# Required: canonical run root from THESIS_FINAL_NOTEBOOK.
# Prefer setting this in the shell/Jupyter environment. Example:
#   export CPS_CANONICAL_RUN=/shared/home1/.../cps_synth_v4_4_7_32_BINARY_V6_Q3V5_STUDY_FINAL_20260708_035653
CPS_CANONICAL_RUN = os.environ.get("CPS_CANONICAL_RUN", "").strip()

# Optional: real split directory if REAL_*_SPLIT.parquet is not inside <run>/synthetic.
CPS_REAL_SPLIT_DIR = os.environ.get("CPS_REAL_SPLIT_DIR", "").strip()

# Optional: completed Smart* smoke-test artifact root.
SMARTSTAR_ARTIFACT_ROOT = os.environ.get("SMARTSTAR_ARTIFACT_ROOT", "").strip()

# Q6 sequence-risk policy. Exact active-window replay is a blocker by default.
WINDOW_ROWS = int(os.environ.get("THESIS_Q6_SEQUENCE_WINDOW_ROWS", "60"))
MAX_NN_SYN_WINDOWS = int(os.environ.get("THESIS_Q6_MAX_NN_SYN_WINDOWS", "10000"))
SEQUENCE_GATE_BLOCKS_SPARSE_GROUP = os.environ.get("THESIS_Q6_BLOCK_SPARSE_GROUP", "1") != "0"
UNION_RATIO_WARN_HI = float(os.environ.get("THESIS_Q6_UNION_RATIO_WARN_HI", "1.50"))
UNION_RATIO_WARN_LO = float(os.environ.get("THESIS_Q6_UNION_RATIO_WARN_LO", "0.67"))
REPEATED_ACTIVE_RATE_WARN_HI = float(os.environ.get("THESIS_Q6_REPEATED_ACTIVE_RATE_WARN_HI", "0.80"))

# Absolute-Fano sensitivity policy. These are sensitivity values, not universal constants.
# If your Table S8 used different absolute-Fano thresholds, set these two env vars and rerun cells STRONG.3–STRONG.4.
ABS_FANO_WARN_THRESHOLD = float(os.environ.get("THESIS_ABS_FANO_WARN_THRESHOLD", "0.50"))
ABS_FANO_FATAL_THRESHOLD = float(os.environ.get("THESIS_ABS_FANO_FATAL_THRESHOLD", "1.00"))

# Q5: fast claim-scope downgrade runs by default. Extended seed rerun is optional because it can be expensive.
RUN_Q5_EXTENDED_SEEDS = os.environ.get("THESIS_RUN_Q5_EXTENDED_SEEDS", "0") == "1"
Q5_SEEDS = [int(x) for x in os.environ.get(
    "THESIS_Q5_SEEDS",
    ",".join(str(20260702 + i) for i in range(20))
).split(",") if x.strip()]
Q5_MODEL_MAX_ITER = int(os.environ.get("THESIS_Q5_MODEL_MAX_ITER", "200"))
Q5_USE_CONSERVATIVE_PUBLIC = os.environ.get("THESIS_Q5_USE_CONSERVATIVE_PUBLIC", "1") != "0"

# Packaging: this notebook writes a merged package index but does not copy files unless you ask.
MATERIALIZE_STRONG_PACKAGE = os.environ.get("THESIS_MATERIALIZE_STRONG_PACKAGE", "0") == "1"
STRONG_PACKAGE_ROOT = os.environ.get("THESIS_STRONG_PACKAGE_ROOT", "").strip()

print("CPS_CANONICAL_RUN        =", CPS_CANONICAL_RUN or "<not set>")
print("CPS_REAL_SPLIT_DIR       =", CPS_REAL_SPLIT_DIR or "<auto>")
print("SMARTSTAR_ARTIFACT_ROOT  =", SMARTSTAR_ARTIFACT_ROOT or "<optional/not set>")
print("WINDOW_ROWS              =", WINDOW_ROWS)
print("ABS_FANO thresholds      =", ABS_FANO_WARN_THRESHOLD, ABS_FANO_FATAL_THRESHOLD)
print("RUN_Q5_EXTENDED_SEEDS    =", RUN_Q5_EXTENDED_SEEDS, "n=", len(Q5_SEEDS))
