# %% CELL Q6.2 — Reproducibility and checksum manifest
# Purpose:
#   Write a reviewer-facing reproducibility manifest and checksums for active branch artifacts.
#
# Safety:
#   - Reads artifact bytes only for hashes.
#   - Does not inspect raw TEST values.
#   - Does not mutate synthetic data.

import os
import sys
import json
import hashlib
import platform
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

OUTDIR_R = Path(str(globals().get("OUTDIR", os.environ.get("CPS_OUTDIR")))).expanduser().resolve()
REPORT_DIR_R = Path(str(globals().get("REPORT_DIR", OUTDIR_R / "reports"))).expanduser().resolve()
CONTRACT_DIR_R = Path(str(globals().get("CONTRACT_DIR", OUTDIR_R / "artifacts" / "contracts"))).expanduser().resolve()
OUT_SYN_R = Path(str(globals().get("OUT_SYN", OUTDIR_R / "synthetic"))).expanduser().resolve()

candidate_paths = [
    CONTRACT_DIR_R / "q6_active_branch_registry_contract_v1_0_THESIS.json",
    CONTRACT_DIR_R / "q6_active_artifact_hygiene_contract_v1_0_THESIS.json",
    CONTRACT_DIR_R / "cell12c6R_final_qa_contract.json",
    REPORT_DIR_R / "cell12c6R_final_publication_manifest.json",
    CONTRACT_DIR_R / "cell12d6_binary_final_test_qa_contract_v6_0_THESIS.json",
    REPORT_DIR_R / "cell12d_v6_terminal_test_qa_summary.json",
    CONTRACT_DIR_R / "q3_post_v5_observability_mask_post_qa_contract.json",
    CONTRACT_DIR_R / "cell12f5_mask_final_test_qa_contract_v1_1_THESIS.json",
    OUT_SYN_R / "IOT_FINAL_CONTINUOUS_TEST.parquet",
    OUT_SYN_R / "IOT_FINAL_BINARY_TEST.parquet",
    OUT_SYN_R / "IOT_FINAL_OBSERVABILITY_MASK_TEST.parquet",
]
rows = []
for p in candidate_paths:
    rows.append({
        "path": str(p),
        "exists": p.exists(),
        "size_bytes": int(p.stat().st_size) if p.exists() else None,
        "sha256": _sha256_file(p) if p.exists() else None,
    })
checksums = pd.DataFrame(rows)
checksums_path = REPORT_DIR_R / "q6_active_artifact_checksums.csv"
checksums.to_csv(checksums_path, index=False)

packages = {}
for name in ["numpy", "pandas", "pyarrow", "sklearn", "scipy", "nbformat"]:
    try:
        mod = __import__(name)
        packages[name] = getattr(mod, "__version__", "unknown")
    except Exception as e:
        packages[name] = f"unavailable: {e}"

cfg_snapshot = {}
if isinstance(globals().get("CFG"), dict):
    for k, v in globals()["CFG"].items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            cfg_snapshot[k] = v

manifest = {
    "cell": "Q6.2",
    "role": "reproducibility_and_checksum_manifest",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "packages": packages,
    "outdir": str(OUTDIR_R),
    "active_artifact_checksums": rows,
    "cfg_scalar_snapshot": cfg_snapshot,
    "policy": {
        "reads_TEST_real_values": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "post_TEST_repair_done_here": False,
    },
}
manifest_path = REPORT_DIR_R / "q6_reproducibility_manifest.json"
contract_path = CONTRACT_DIR_R / "q6_reproducibility_manifest_contract_v1_0_THESIS.json"
for p in [manifest_path, contract_path]:
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")

CELL_Q6_REPRODUCIBILITY_MANIFEST = manifest
print(f"[Q6.2] Checksums: {checksums_path}")
print(f"[Q6.2] Contract: {contract_path}")
