
# %% THESIS.REV.0 — path resolver and shared helpers
from pathlib import Path
import os, json, csv, hashlib, shutil, re, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# ---- Resolve the final governed run root. ----
def _first_existing_path(candidates):
    for x in candidates:
        if not x:
            continue
        try:
            p = Path(str(x)).expanduser().resolve()
            if p.exists():
                return p
        except Exception:
            pass
    return None

RUN_ROOT = _first_existing_path([
    os.environ.get("CPS_CANONICAL_RUN", ""),
    os.environ.get("CPS_STUDY_FINAL_OUTDIR", ""),
    globals().get("ARTIFACT_ROOT", ""),
    globals().get("OUTDIR", ""),
    globals().get("PROJECT_ROOT_170", ""),
    globals().get("PROJECT_ROOT_203", ""),
    (globals().get("CFG", {}) or {}).get("outdir", "") if isinstance(globals().get("CFG", {}), dict) else "",
])
if RUN_ROOT is None:
    raise RuntimeError("Could not resolve final run root. Set CPS_CANONICAL_RUN=/path/to/final/governed/run and rerun this cell.")

REPORT_DIR_FIX = RUN_ROOT / "reports"
ARTIFACT_DIR_FIX = RUN_ROOT / "artifacts"
SYN_DIR_FIX = RUN_ROOT / "synthetic"
for p in [REPORT_DIR_FIX, ARTIFACT_DIR_FIX, SYN_DIR_FIX]:
    p.mkdir(parents=True, exist_ok=True)

EXT_OUT_FIX = _first_existing_path([
    os.environ.get("CPS_EXT_OUT", ""),
    globals().get("EXT_OUT", ""),
    RUN_ROOT / "EXT_OUT",
])
if EXT_OUT_FIX is None:
    EXT_OUT_FIX = RUN_ROOT / "EXT_OUT"
    (EXT_OUT_FIX / "tables").mkdir(parents=True, exist_ok=True)
    (EXT_OUT_FIX / "manifests").mkdir(parents=True, exist_ok=True)

SMARTSTAR_ROOT_FIX = _first_existing_path([
    os.environ.get("SMARTSTAR_ARTIFACT_ROOT", ""),
    os.environ.get("SMARTSTAR_OUT", ""),
    globals().get("SMARTSTAR_ARTIFACT_ROOT", ""),
    RUN_ROOT / "smartstar_homeA_study_thesis_transfer_artifact",
    RUN_ROOT.parent / "smartstar_homeA_study_thesis_transfer_artifact",
    Path.cwd() / "smartstar_homeA_study_thesis_transfer_artifact",
])

PATCH_DIR = REPORT_DIR_FIX / "review_fix_outputs"
PATCH_DIR.mkdir(parents=True, exist_ok=True)

# ---- Helpers. ----
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def rel_to_run(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(RUN_ROOT.resolve()))
    except Exception:
        return str(path)

def find_files(patterns, roots=None):
    if isinstance(patterns, str):
        patterns = [patterns]
    roots = roots or [RUN_ROOT]
    if isinstance(roots, (str, Path)):
        roots = [roots]
    hits = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for pat in patterns:
            hits.extend(root.rglob(pat))
    hits = sorted(set([p.resolve() for p in hits if p.is_file()]), key=lambda p: str(p))
    return hits

def find_one(patterns, roots=None, required=True, label="file"):
    hits = find_files(patterns, roots=roots)
    if not hits:
        if required:
            raise FileNotFoundError(f"Could not find {label}; patterns={patterns}; roots={roots or [RUN_ROOT]}")
        return None
    # Prefer report files over old duplicates; prefer non-QUARANTINE.
    hits = [p for p in hits if "QUARANTINE" not in str(p).upper()] or hits
    hits = sorted(hits, key=lambda p: (0 if "/reports/" in str(p) else 1, len(str(p)), str(p)))
    return hits[0]

def read_csv_required(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {rel_to_run(path)} rows={len(df)}")
    return path

def write_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {rel_to_run(path)}")
    return path

def pick_col(df: pd.DataFrame, contains_any, exclude_any=(), required=True, label="column"):
    contains_any = [s.lower() for s in contains_any]
    exclude_any = [s.lower() for s in exclude_any]
    for c in df.columns:
        cl = c.lower()
        if all(x not in cl for x in exclude_any) and any(x in cl for x in contains_any):
            return c
    if required:
        raise KeyError(f"Could not find {label}. columns={list(df.columns)} contains={contains_any} exclude={exclude_any}")
    return None

print("THESIS.REV paths")
print("  RUN_ROOT       =", rel_to_run(RUN_ROOT))
print("  REPORT_DIR     =", rel_to_run(REPORT_DIR_FIX))
print("  EXT_OUT        =", rel_to_run(EXT_OUT_FIX))
print("  SMARTSTAR_ROOT =", str(SMARTSTAR_ROOT_FIX) if SMARTSTAR_ROOT_FIX else "<not resolved yet; set SMARTSTAR_ARTIFACT_ROOT>")
