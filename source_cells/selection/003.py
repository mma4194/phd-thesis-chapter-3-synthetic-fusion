# %% STRONG.1 — Imports, path resolver, and shared artifact helpers

import csv
import hashlib
import json
import math
import re
import shutil
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_path(x: Any) -> Optional[Path]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return Path(s).expanduser().resolve()
    except Exception:
        return None


def first_existing(candidates: Sequence[Any]) -> Optional[Path]:
    for c in candidates:
        p = _as_path(c)
        if p is not None and p.exists():
            return p
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return json_sanitize(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return json_sanitize(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "Infinity" if x > 0 else "-Infinity"
        return x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(payload: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, sort_keys=True)
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_text(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def read_json_any(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def find_files(patterns: Any, roots: Sequence[Path], recursive: bool = True) -> List[Path]:
    if isinstance(patterns, (str, Path)):
        patterns = [str(patterns)]
    hits: List[Path] = []
    for root in roots:
        if root is None:
            continue
        root = Path(root)
        if not root.exists():
            continue
        for pat in patterns:
            if recursive:
                hits.extend(root.rglob(str(pat)))
            else:
                hits.extend(root.glob(str(pat)))
    # drop duplicates and quarantine/superseded paths when possible
    uniq = []
    seen = set()
    for p in hits:
        rp = p.resolve()
        key = str(rp)
        if key in seen:
            continue
        if "QUARANTINE" in key.upper():
            continue
        seen.add(key)
        uniq.append(rp)
    return sorted(uniq, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def find_one(patterns: Any, roots: Sequence[Path], required: bool = True, label: str = "artifact") -> Optional[Path]:
    hits = find_files(patterns, roots)
    if not hits:
        if required:
            raise FileNotFoundError(f"Could not find {label} using patterns={patterns} roots={[str(r) for r in roots]}")
        return None
    return hits[0]


def pick_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True, label: str = "column") -> Optional[str]:
    if df is None or df.empty:
        if required:
            raise ValueError(f"Cannot pick {label}: empty DataFrame")
        return None
    norm = {str(c).lower().strip(): c for c in df.columns}
    for c in candidates:
        key = str(c).lower().strip()
        if key in norm:
            return norm[key]
    # contains fallback
    for c in df.columns:
        lc = str(c).lower()
        if any(str(k).lower() in lc for k in candidates):
            return c
    if required:
        raise KeyError(f"Could not find {label}; candidates={candidates}; columns={list(df.columns)}")
    return None


def bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


RUN_ROOT = _as_path(CPS_CANONICAL_RUN)
if RUN_ROOT is None or not RUN_ROOT.exists():
    raise RuntimeError("Set CPS_CANONICAL_RUN to the completed main notebook run root before running this notebook.")
if not (RUN_ROOT / "reports").exists() or not (RUN_ROOT / "synthetic").exists():
    raise RuntimeError(f"CPS_CANONICAL_RUN does not look like a completed run root: {RUN_ROOT}")

SYN_DIR = RUN_ROOT / "synthetic"
REPORT_DIR = RUN_ROOT / "reports"
ARTIFACT_DIR = RUN_ROOT / "artifacts"
EXT_OUT = RUN_ROOT / "EXT_OUT"
STRONG_DIR = REPORT_DIR / "thesis_strong_revision_fixes"
STRONG_TABLES = STRONG_DIR / "tables"
STRONG_MANIFESTS = STRONG_DIR / "manifests"
STRONG_SYNTH = SYN_DIR
for d in [STRONG_DIR, STRONG_TABLES, STRONG_MANIFESTS]:
    d.mkdir(parents=True, exist_ok=True)

REAL_SPLIT_DIR = first_existing([CPS_REAL_SPLIT_DIR, SYN_DIR]) or SYN_DIR
SMARTSTAR_ROOT = first_existing([SMARTSTAR_ARTIFACT_ROOT])


def rel_to_run(path: Any) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(RUN_ROOT))
    except Exception:
        return str(p)

print("RUN_ROOT  =", RUN_ROOT)
print("STRONG_DIR=", STRONG_DIR)
print("REAL_SPLIT_DIR=", REAL_SPLIT_DIR)
print("Smart*    =", SMARTSTAR_ROOT or "<not available>")
print("scipy/sklearn available:", SCIPY_AVAILABLE, SKLEARN_AVAILABLE)
