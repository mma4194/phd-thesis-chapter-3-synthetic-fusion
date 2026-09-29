import os
from pathlib import Path
from datetime import datetime, timezone

def _bv6_root_log(msg):
    print(f"[binary-v6-study-root] {msg}")

# Prefer a final clean artifact root. It must still contain BINARY_V6 so existing guards remain compatible.
env_study_final_outdir = os.environ.get("CPS_STUDY_FINAL_OUTDIR", "").strip()
env_binary_v6_outdir = os.environ.get("CPS_BINARY_V6_OUTDIR", "").strip()
env_cps_outdir = os.environ.get("CPS_OUTDIR", "").strip()

if env_study_final_outdir and "BINARY_V6" in env_study_final_outdir:
    root = Path(env_study_final_outdir).expanduser().resolve()
elif env_binary_v6_outdir and "BINARY_V6" in env_binary_v6_outdir:
    root = Path(env_binary_v6_outdir).expanduser().resolve()
elif env_cps_outdir and "BINARY_V6" in env_cps_outdir:
    root = Path(env_cps_outdir).expanduser().resolve()
else:
    default_parent = Path("/path/to/input").expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = default_parent / f"cps_synth_v4_4_7_32_BINARY_V6_Q3V5_STUDY_FINAL_{timestamp}"

root = root.expanduser().resolve()
root_str = str(root)

for forbidden in [
    "q6_public_reaudit_v1",
    "FULL_COUPLED_STUDY_VERIFY",
    "BINARY_V2",
    "BINARY_V3",
    "BINARY_V4",
    "BINARY_V5",
]:
    if forbidden in root_str:
        raise RuntimeError(f"[binary-v6-root] Refusing root containing '{forbidden}': {root}")

if root.name == "cps_synth_v4_4_7_32_FULL_COUPLED":
    raise RuntimeError(
        "[binary-v6-root] Refusing old contaminated FULL_COUPLED root. "
        "Set CPS_BINARY_V6_OUTDIR to a new clean directory."
    )

if "BINARY_V6" not in root_str:
    raise RuntimeError(f"[binary-v6-root] Root must visibly contain BINARY_V6: {root}")

out_syn = root / "synthetic"
report_dir = root / "reports"
artifact_dir = root / "artifacts"
contract_dir = artifact_dir / "contracts"

for p in [root, out_syn, report_dir, artifact_dir, contract_dir]:
    p.mkdir(parents=True, exist_ok=True)

OUTDIR = str(root)
OUT_SYN = str(out_syn)
REPORT_DIR = str(report_dir)
ARTIFACT_DIR = str(artifact_dir)
CONTRACT_DIR = str(contract_dir)
ARTDIR = ARTIFACT_DIR

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}

CFG["binary_v6_development_notebook"] = True
CFG["study_thesis_clean_final_notebook"] = True
CFG["active_q3_branch"] = "12.f.V5"
CFG["active_binary_branch"] = "12.d.V6"
CFG["active_continuous_branch"] = "12.c.R"
CFG["outdir"] = OUTDIR
CFG["OUTDIR"] = OUTDIR
CFG["out_syn"] = OUT_SYN
CFG["OUT_SYN"] = OUT_SYN
CFG["report_dir"] = REPORT_DIR
CFG["REPORT_DIR"] = REPORT_DIR
CFG["artifact_dir"] = ARTIFACT_DIR
CFG["ARTIFACT_DIR"] = ARTIFACT_DIR
CFG["contract_dir"] = CONTRACT_DIR
CFG["CONTRACT_DIR"] = CONTRACT_DIR
CFG["ARTDIR"] = ARTDIR

os.environ["CPS_OUTDIR"] = OUTDIR
os.environ["CPS_BINARY_V6_OUTDIR"] = OUTDIR
os.environ["CPS_STUDY_FINAL_OUTDIR"] = OUTDIR

assert Path(OUTDIR).exists()
assert Path(OUT_SYN).exists()
assert Path(REPORT_DIR).exists()
assert Path(ARTIFACT_DIR).exists()
assert Path(CONTRACT_DIR).exists()
assert "BINARY_V6" in str(Path(OUTDIR))
for marker in ["BINARY_V2", "BINARY_V3", "BINARY_V4", "BINARY_V5"]:
    assert marker not in str(Path(OUTDIR))

_bv6_root_log(f"OUTDIR       = {OUTDIR}")
_bv6_root_log(f"OUT_SYN      = {OUT_SYN}")
_bv6_root_log(f"REPORT_DIR   = {REPORT_DIR}")
_bv6_root_log(f"ARTIFACT_DIR = {ARTIFACT_DIR}")
_bv6_root_log(f"CONTRACT_DIR = {CONTRACT_DIR}")
_bv6_root_log("PASS: Binary V6 / Q3 V5 STUDY clean output root is active.")