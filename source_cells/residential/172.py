import os
from pathlib import Path

def _bv6_guard_log(msg):
    print(f"[binary-v6-clean-root-guard] {msg}")

if "CFG" in globals() and isinstance(CFG, dict):
    if "OUTDIR" not in globals() and CFG.get("outdir"):
        OUTDIR = str(Path(CFG["outdir"]).expanduser().resolve())
    if "OUT_SYN" not in globals() and CFG.get("out_syn"):
        OUT_SYN = str(Path(CFG["out_syn"]).expanduser().resolve())
    if "REPORT_DIR" not in globals() and CFG.get("report_dir"):
        REPORT_DIR = str(Path(CFG["report_dir"]).expanduser().resolve())

for name in ["OUTDIR", "OUT_SYN", "REPORT_DIR"]:
    if name not in globals():
        raise RuntimeError(
            f"[binary-v6-clean-root-guard] Missing global {name}. "
            "Run CELL 0.y — Binary V6 clean output-root bootstrap after Cell 1."
        )

paths = {
    "OUTDIR": Path(str(OUTDIR)).expanduser().resolve(),
    "OUT_SYN": Path(str(OUT_SYN)).expanduser().resolve(),
    "REPORT_DIR": Path(str(REPORT_DIR)).expanduser().resolve(),
}

for name, p in paths.items():
    _bv6_guard_log(f"{name} = {p}")

    if not p.exists():
        raise RuntimeError(f"[binary-v6-clean-root-guard] {name} does not exist: {p}")

    if "BINARY_V6" not in str(p):
        raise RuntimeError(f"[binary-v6-clean-root-guard] {name} is not a Binary V6 root: {p}")

    for forbidden in ["BINARY_V2", "BINARY_V3", "BINARY_V4", "BINARY_V5", "q6_public_reaudit_v1", "FULL_COUPLED_STUDY_VERIFY"]:
        if forbidden in str(p):
            raise RuntimeError(
                f"[binary-v6-clean-root-guard] {name} points to forbidden marker '{forbidden}': {p}"
            )

env_outdir = os.environ.get("CPS_OUTDIR", "")
env_bv6_outdir = os.environ.get("CPS_BINARY_V6_OUTDIR", "")

_bv6_guard_log(f"CPS_OUTDIR = {env_outdir}")
_bv6_guard_log(f"CPS_BINARY_V6_OUTDIR = {env_bv6_outdir}")

if "BINARY_V6" not in env_outdir:
    raise RuntimeError("[binary-v6-clean-root-guard] CPS_OUTDIR is not Binary V6. Rerun CELL 0.y.")

if "BINARY_V6" not in env_bv6_outdir:
    raise RuntimeError("[binary-v6-clean-root-guard] CPS_BINARY_V6_OUTDIR is not Binary V6. Rerun CELL 0.y.")

_bv6_guard_log("PASS: clean Binary V6 development root is active.")