# %% CANONICAL-RUN v2 — write submission-run manifest for the CURRENT run root
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
RUN = Path(str(globals().get("OUTDIR", CFG.get("outdir", "")))).expanduser().resolve()
if not RUN.exists(): raise RuntimeError(f"[CANONICAL-RUN] Current OUTDIR does not exist: {RUN}")
PARENT = RUN.parent
def sha256(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()
key_artifacts = [
    Path(globals().get("CELL15_0_FINAL_CPS_NO_Q4_PATH", RUN / "synthetic/CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet")),
    Path(globals().get("CELL15_6_PUBLIC_CPS_PATH", RUN / "synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet")),
    RUN / "reports/cell12e6_driver_corrected_release_ledger.csv",
    RUN / "reports/cell15_6_q6_public_column_registry.csv",
    RUN / "reports/cell15_6_q6_public_dropped_columns.csv",
    RUN / "reports/paper_numbers_summary.json",
]
missing = [str(p) for p in key_artifacts if not Path(p).exists()]
if missing: raise RuntimeError(f"[CANONICAL-RUN] Missing load-bearing artifacts; rerun downstream cells: {missing}")
manifest = {
    "canonical_submission_run": RUN.name,
    "designated_utc": datetime.now(timezone.utc).isoformat(),
    "paper": "Synthetic Fusion (STUDY THESIS)",
    "public_logical_cols": int(CELL15_6_Q6_PUBLIC_MITIGATION_CONTRACT["column_counts"]["kept_cols"]),
    "dropped_logical_cols": int(CELL15_6_Q6_PUBLIC_MITIGATION_CONTRACT["column_counts"]["dropped_cols"]),
    "corrected_sparse_driver_release_eligible_n": int(len(DRIVER_RELEASE_ELIGIBLE_COLS_CORRECTED)),
    "corrected_sparse_driver_release_excluded_fatal_n": int(len(DRIVER_RELEASE_EXCLUDED_FATAL_COLS_CORRECTED)),
    "artifacts": [{"path": str(Path(p).relative_to(PARENT)), "sha256": sha256(p)} for p in key_artifacts],
}
(PARENT / "SUBMISSION_RUN.txt").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2, sort_keys=True))
print(f"[CANONICAL-RUN] Wrote {PARENT / 'SUBMISSION_RUN.txt'}")
