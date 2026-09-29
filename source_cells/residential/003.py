from pathlib import Path
import os
import json
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError("pyarrow is required for metadata-only parquet inspection.") from e

if "CFG" not in globals() or not isinstance(CFG, dict):
    raise RuntimeError("Run Cell 1 first so CFG is defined.")

_parquet_env = os.environ.get("CPS_INPUT_PARQUET", "").strip()
_parquet_cfg = str(CFG.get("input_parquet", "")).strip()
PARQUET_PATH = Path(_parquet_env or _parquet_cfg).expanduser().resolve()
if not PARQUET_PATH.exists():
    raise FileNotFoundError(
        f"Configured parquet does not exist: {PARQUET_PATH}. "
        "Set CPS_INPUT_PARQUET or CFG['input_parquet']; do not hard-code /shared/home1 paths."
    )

pf = pq.ParquetFile(PARQUET_PATH)
cols = list(map(str, pf.schema_arrow.names))

candidate_time_cols = [
    str(CFG.get("raw_time_col", "sec")),
    str(CFG.get("canonical_time_col", "sec_epoch_s__canon")),
    "timestamp", "time", "datetime", "frame.time_epoch", "frame_time_epoch",
]
time_col = next((c for c in candidate_time_cols if c in cols), None)
if time_col is None:
    for c in cols:
        lc = str(c).lower()
        if "time" in lc or "timestamp" in lc or "epoch" in lc or lc in {"sec", "seconds"}:
            time_col = c
            break
if time_col is None:
    raise RuntimeError("No time column found for duration audit. Configure CFG['raw_time_col'].")

t = pd.read_parquet(PARQUET_PATH, columns=[time_col])[time_col]
if pd.api.types.is_datetime64_any_dtype(t):
    dt = pd.to_datetime(t, errors="coerce").dropna()
    start, end = dt.min(), dt.max()
    duration_seconds = float((end - start).total_seconds())
else:
    x = pd.to_numeric(t, errors="coerce").dropna()
    if x.empty:
        raise RuntimeError(f"Time column {time_col!r} has no numeric/datetime values.")
    if float(x.median()) > 1_000_000_000:
        dt = pd.to_datetime(x, unit="s", errors="coerce").dropna()
        start, end = dt.min(), dt.max()
        duration_seconds = float((end - start).total_seconds())
    else:
        start, end = float(x.min()), float(x.max())
        duration_seconds = float(x.max() - x.min())

audit = {
    "cell": "portable_capture_duration_audit_replacement",
    "input_parquet": str(PARQUET_PATH),
    "time_col": str(time_col),
    "rows": int(len(t)),
    "start": str(start),
    "end": str(end),
    "duration_seconds": duration_seconds,
    "duration_days": duration_seconds / 86400.0,
    "hard_coded_local_path_used": False,
}
contract_dir = Path(CFG["outdir"]).expanduser().resolve() / "artifacts" / "contracts"
contract_dir.mkdir(parents=True, exist_ok=True)
(contract_dir / "portable_capture_duration_audit_contract.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
)
print("=" * 70)
print(f"Parquet file: {PARQUET_PATH}")
print(f"Time column: {time_col}")
print(f"Rows: {len(t):,}")
print(f"Start: {start}")
print(f"End: {end}")
print(f"Duration seconds: {duration_seconds:,.2f}")
print(f"Duration days: {duration_seconds / 86400.0:.4f}")
print("=" * 70)