# ==========================================================
# CELL 14.6D0.PRE - Discover available Q4/A0 diagnostic inputs
# ==========================================================

from pathlib import Path
import pandas as pd
import os

REPORT_DIR_PRE = Path(str(REPORT_DIR)).expanduser().resolve()
OUTDIR_PRE = Path(str(OUTDIR)).expanduser().resolve()

search_roots = []
for p in [
    REPORT_DIR_PRE,
    OUTDIR_PRE,
    OUTDIR_PRE / "reports",
    OUTDIR_PRE / "artifacts",
    OUTDIR_PRE / "artifacts" / "contracts",
    OUTDIR_PRE / "quarantine",
    OUTDIR_PRE / "archive",
    OUTDIR_PRE / "superseded",
]:
    if p.exists() and p.is_dir():
        search_roots.append(p)

# Also search one level below OUTDIR for quarantine/archive/report-like folders.
for p in OUTDIR_PRE.glob("*"):
    if p.is_dir() and any(tok in p.name.lower() for tok in ["report", "q6", "quarantine", "archive", "superseded", "artifact"]):
        search_roots.append(p)

search_roots = list(dict.fromkeys(search_roots))

hits = []
patterns = [
    "*14_6*A0*.csv",
    "*14.6*A0*.csv",
    "*146*A0*.csv",
    "*A0*q4*.csv",
    "*A0*Q4*.csv",
    "*q4*pair*metric*.csv",
    "*Q4*pair*metric*.csv",
    "*trainval*policy*.csv",
    "*14_6R0*.csv",
    "*14_6R1*.csv",
]

for root in search_roots:
    for pat in patterns:
        for f in root.rglob(pat):
            if f.is_file():
                try:
                    stat = f.stat()
                    hits.append({
                        "path": str(f),
                        "name": f.name,
                        "size_bytes": stat.st_size,
                        "mtime": pd.Timestamp(stat.st_mtime, unit="s"),
                    })
                except Exception:
                    hits.append({
                        "path": str(f),
                        "name": f.name,
                        "size_bytes": None,
                        "mtime": None,
                    })

hits_df = pd.DataFrame(hits).drop_duplicates("path") if hits else pd.DataFrame(
    columns=["path", "name", "size_bytes", "mtime"]
)

if len(hits_df):
    hits_df = hits_df.sort_values(["mtime", "name"], ascending=[False, True])

print("REPORT_DIR:", REPORT_DIR_PRE)
print("OUTDIR:", OUTDIR_PRE)
print("Search roots:")
for r in search_roots:
    print(" -", r)

print("\nCandidate A0/Q4 files:")
display(hits_df)

# Try previewing columns for likely metric/policy files.
preview_rows = []
for _, row in hits_df.head(80).iterrows():
    path = Path(row["path"])
    try:
        df_head = pd.read_csv(path, nrows=3)
        preview_rows.append({
            "path": str(path),
            "rows_previewed": len(df_head),
            "columns": list(df_head.columns),
        })
    except Exception as e:
        preview_rows.append({
            "path": str(path),
            "rows_previewed": None,
            "columns": f"READ_ERROR: {type(e).__name__}: {e}",
        })

preview_df = pd.DataFrame(preview_rows)
print("\nColumn preview:")
display(preview_df)