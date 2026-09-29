from __future__ import annotations

# Source cell 2


from pathlib import Path
import os

SMARTSTAR_ROOT = Path(os.environ.get("SMARTSTAR_ROOT", "./data/raw/smartstar")).expanduser()

SMARTSTAR_OUT = Path(os.environ.get("SMARTSTAR_OUT", "./runs/smartstar")).expanduser().resolve()

CORE_FOLDERS = [
    "homeA-circuit",
    "homeA-environmental",
    "homeA-switch",
    "homeA-furnace",
    "homeA-door",
    "homeA-motion",
]

INCLUDE_OPTIONAL_METER_PHASE = False
OPTIONAL_FOLDERS = ["homeA-meter", "homeA-phase"]

WINDOW_FREQ = "60s"
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20
RANDOM_SEED = 20260717

MIN_TRAIN_VAL_NONMISSING = 30
MIN_VAL_NONMISSING = 5
MAX_FEATURES_PER_ROLE_FOR_SMOKE = None

BINARY_FFILL_LIMIT_WINDOWS = 180

MAX_RAW_FILES_FOR_DRY_RUN = None
WRITE_SYNTHETIC_TEST_ARTIFACT = True

print("SMARTSTAR_ROOT =", f"<local_path>/{SMARTSTAR_ROOT.name}")
print("SMARTSTAR_OUT  =", f"<local_path>/{SMARTSTAR_OUT.name}")
print("Core folders   =", CORE_FOLDERS)
print("Optional folders included?", INCLUDE_OPTIONAL_METER_PHASE)

# Source cell 4

import csv
import datetime as dt
import hashlib
import json
import math
import platform
import random
import re
import shutil
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import sklearn
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import matplotlib.pyplot as plt

try:
    from IPython.display import display
except Exception:
    def display(obj):
        try:
            print(obj.to_string() if hasattr(obj, "to_string") else obj)
        except Exception:
            print(obj)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 180)

warnings.filterwarnings("ignore", message="Could not infer format", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="ks_2samp: Exact calculation unsuccessful.*", category=RuntimeWarning)

SMARTSTAR_OUT.mkdir(parents=True, exist_ok=True)
for sub in ["manifests", "tables", "ledgers", "features", "figures", "synthetic", "logs", "manuscript"]:
    (SMARTSTAR_OUT / sub).mkdir(parents=True, exist_ok=True)

RUN_STARTED_UTC = dt.datetime.now(dt.timezone.utc).isoformat()

ENV_INFO = {
    "run_started_utc": RUN_STARTED_UTC,
    "python": sys.version,
    "platform": platform.platform(),
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "scipy_available": SCIPY_AVAILABLE,
    "sklearn_available": SKLEARN_AVAILABLE,
    "sklearn_version": getattr(sys.modules.get("sklearn"), "__version__", None) if SKLEARN_AVAILABLE else None,
}

with open(SMARTSTAR_OUT / "manifests" / "environment_manifest.json", "w") as f:
    json.dump(ENV_INFO, f, indent=2, sort_keys=True)

STATUS_ORDER = {"pass": 0, "warning": 1, "fatal": 2, "blocker": 3, "unsupported": 4, "not_assessed": 5, "excluded": 6}

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))

def safe_name(x: Any, max_len: int = 96) -> str:
    s = str(x)
    s = s.replace("*", "star")
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    if not s:
        s = "unnamed"
    return s[:max_len]

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)

def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def save_table(df: pd.DataFrame, name: str, subdir: str = "tables") -> Path:
    path = SMARTSTAR_OUT / subdir / name
    write_csv(path, df)
    return path

def worst_status(statuses: Iterable[str]) -> str:
    vals = [s for s in statuses if isinstance(s, str) and s]
    if not vals:
        return "not_assessed"
    return max(vals, key=lambda s: STATUS_ORDER.get(s, 99))

def stable_rng(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256((str(seed) + "::" + key).encode("utf-8")).hexdigest()
    sub_seed = int(digest[:16], 16) % (2**32 - 1)
    return np.random.default_rng(sub_seed)

print(json.dumps(ENV_INFO, indent=2))

# Source cell 6

selected_folders = list(CORE_FOLDERS) + (list(OPTIONAL_FOLDERS) if INCLUDE_OPTIONAL_METER_PHASE else [])

FROZEN_POLICY = {
    "artifact_name": "smartstar_homeA_study_thesis_governance_transfer_smoke_test",
    "dataset": {
        "name": "UMass Smart* Home Dataset, 2013 Home A subset",
        "source_page": "https://traces.cs.umass.edu/docs/traces/smartstar/",
        "license_statement": "CC BY 4.0 as stated on the UMass Smart* trace page",
        "selected_folders": selected_folders,
        "excluded_optional_folders": [] if INCLUDE_OPTIONAL_METER_PHASE else OPTIONAL_FOLDERS,
    },
    "claim_scope": {
        "supported": "procedural governance-transfer smoke test over a public weaker smart-home dataset",
        "not_supported": [
            "empirical universality of the restricted residential CPS readiness counts",
            "full smart-home CPS validation",
            "Q4 physical-network coupling validation",
            "leakage-safe anomaly-detection benchmark",
            "formal privacy guarantee or routine non-inference proof",
        ],
        "q4_policy": "not_instantiated_no_independent_protocol_or_network_stream",
        "q5_policy": "not_assessed_unless_predeclared_compatible_task_exists",
        "q6_policy": "public-source release-manifest smoke test only; no privacy guarantee",
    },
    "split": {
        "type": "chronological",
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "test_use": "final audit only; no repair, selection, tuning, or threshold setting",
    },
    "feature_rules": {
        "window_freq": WINDOW_FREQ,
        "read_csv_header": None,
        "core_folders": CORE_FOLDERS,
        "optional_folders": OPTIONAL_FOLDERS,
        "include_optional_meter_phase": INCLUDE_OPTIONAL_METER_PHASE,
        "binary_ffill_limit_windows": BINARY_FFILL_LIMIT_WINDOWS,
        "min_train_val_nonmissing": MIN_TRAIN_VAL_NONMISSING,
        "min_val_nonmissing": MIN_VAL_NONMISSING,
        "max_features_per_role_for_smoke": MAX_FEATURES_PER_ROLE_FOR_SMOKE,
    },
    "candidate_registry": {
        "iot_continuous": ["empirical_iid", "minute_of_day", "block_bootstrap"],
        "iot_binary": ["bernoulli_iid", "minute_of_day_bernoulli", "markov_1"],
        "iot_driver": ["bernoulli_iid", "minute_of_day_bernoulli", "event_block_bootstrap"],
        "iot_observability": ["bernoulli_iid", "minute_of_day_bernoulli", "markov_1"],
    },
    "readiness_gates": {
        "continuous": {
            "ks_pass_max": 0.20,
            "ks_warning_max": 0.35,
            "nw_pass_max": 0.20,
            "nw_warning_max": 0.35,
            "acf1_absdiff_pass_max": 0.10,
            "acf1_absdiff_warning_max": 0.25,
        },
        "binary": {
            "rate_absdiff_pass_max": 0.05,
            "rate_absdiff_warning_max": 0.15,
            "transition_absdiff_pass_max": 0.05,
            "transition_absdiff_warning_max": 0.15,
            "run_ks_pass_max": 0.20,
            "run_ks_warning_max": 0.35,
        },
        "driver": {
            "rate_ratio_pass_range": [0.50, 2.00],
            "rate_ratio_warning_range": [0.25, 4.00],
            "burst_ratio_pass_range": [0.50, 2.00],
            "burst_ratio_warning_range": [0.25, 4.00],
            "interarrival_ks_pass_max": 0.20,
            "interarrival_ks_warning_max": 0.35,
        },
        "observability": {
            "rate_absdiff_pass_max": 0.05,
            "rate_absdiff_warning_max": 0.15,
            "transition_absdiff_pass_max": 0.05,
            "transition_absdiff_warning_max": 0.15,
            "run_ks_pass_max": 0.20,
            "run_ks_warning_max": 0.35,
        },
    },
    "random_seed": RANDOM_SEED,
    "manifest_created_utc": utc_now(),
}

policy_path = SMARTSTAR_OUT / "manifests" / "transfer_policy_manifest_v0.json"
write_json(policy_path, FROZEN_POLICY)
policy_sha_path = SMARTSTAR_OUT / "manifests" / "transfer_policy_manifest_v0.sha256"
policy_sha_path.write_text(sha256_file(policy_path) + "  transfer_policy_manifest_v0.json\n", encoding="utf-8")

print("Wrote frozen policy manifest:", f"<artifact_root>/manifests/{policy_path.name}")
print("SHA256:", sha256_file(policy_path))

# Source cell 8

if not SMARTSTAR_ROOT.exists():
    raise FileNotFoundError(f"SMARTSTAR_ROOT does not exist: {SMARTSTAR_ROOT}")

missing = [folder for folder in selected_folders if not (SMARTSTAR_ROOT / folder).exists()]
if missing:
    raise FileNotFoundError(f"Missing selected Smart* folders under {SMARTSTAR_ROOT}: {missing}")

all_raw_files = []
for folder in selected_folders:
    base = SMARTSTAR_ROOT / folder
    all_raw_files.extend(sorted([p for p in base.rglob("*") if p.is_file()]))

raw_manifest_rows = []
for p in all_raw_files:
    try:
        rel = str(p.relative_to(SMARTSTAR_ROOT))
    except Exception:
        rel = str(p)
    raw_manifest_rows.append({
        "relative_path": rel,
        "folder": rel.split("/")[0] if "/" in rel else p.parent.name,
        "filename": p.name,
        "suffix": p.suffix,
        "bytes": int(p.stat().st_size),
        "sha256": sha256_file(p),
    })
raw_file_manifest = pd.DataFrame(raw_manifest_rows).sort_values(["folder", "relative_path"])
save_table(raw_file_manifest, "raw_file_manifest.csv", "manifests")

folder_summary = raw_file_manifest.groupby("folder", dropna=False).agg(
    files=("relative_path", "count"),
    bytes=("bytes", "sum"),
).reset_index()
save_table(folder_summary, "raw_folder_summary.csv", "tables")

print(folder_summary)
print("Raw manifest written to", "<artifact_root>/manifests/raw_file_manifest.csv")

# Source cell 10

METADATA_NAME_TOKENS = {"format", "summary", "readme", "license", "metadata", "schema"}
DATE_NAME_RE = re.compile(r"^\d{4}[-_][A-Za-z]{3,9}[-_]\d{1,2}(\..*)?$")

def is_metadata_file(path: Path) -> bool:
    name = path.name.lower()
    return any(tok in name for tok in METADATA_NAME_TOKENS) or name.startswith(".")

def discover_data_files(folder_path: Path) -> List[Path]:
    candidates = []
    for p in sorted(folder_path.rglob("*")):
        if not p.is_file():
            continue
        if is_metadata_file(p):
            continue
        if p.stat().st_size <= 0:
            continue

        candidates.append(p)
    if MAX_RAW_FILES_FOR_DRY_RUN is not None:
        candidates = candidates[:int(MAX_RAW_FILES_FOR_DRY_RUN)]
    return candidates

def family_from_folder(folder: str) -> str:
    s = folder.lower()
    if "circuit" in s:
        return "circuit"
    if "environment" in s or "weather" in s:
        return "environmental"
    if "switch" in s:
        return "switch"
    if "furnace" in s or "hvac" in s or "thermostat" in s:
        return "furnace"
    if "door" in s:
        return "door"
    if "motion" in s:
        return "motion"
    if "phase" in s:
        return "phase"
    if "meter" in s:
        return "meter"
    return "other"

def read_noheader_csv(path: Path, max_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    attempts = [
        {"sep": ",", "engine": "python"},
        {"sep": None, "engine": "python"},
        {"sep": r"\s+", "engine": "python"},
        {"sep": "\t", "engine": "python"},
    ]
    for opts in attempts:
        try:
            df = pd.read_csv(
                path,
                header=None,
                nrows=max_rows,
                on_bad_lines="skip",
                dtype=str,
                **opts,
            )
            if df.shape[1] >= 2 and df.shape[0] > 0:
                df = df.dropna(axis=1, how="all")
                df.columns = list(range(df.shape[1]))
                return df
        except Exception:
            continue
    return None

def numeric_fraction(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    return pd.to_numeric(s, errors="coerce").notna().mean()

def timestamp_score(s: pd.Series) -> Tuple[float, str]:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() < 0.50:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(s, utc=True, errors="coerce")
        return float(parsed.notna().mean()), "datetime"
    sec = num.between(946684800, 1893456000).mean()
    ms = num.between(946684800000, 1893456000000).mean()
    if ms > sec:
        return float(ms), "ms"
    return float(sec), "s"

def find_timestamp_column(df: pd.DataFrame) -> Tuple[Optional[int], Optional[str], float]:
    best = (None, None, -1.0)
    for c in df.columns:
        score, unit = timestamp_score(df[c])
        if score > best[2]:
            best = (int(c), unit, float(score))
    if best[2] < 0.50:
        return None, None, float(best[2])
    return best

def parse_timestamp(s: pd.Series, unit: Optional[str]) -> pd.Series:
    if unit in {"s", "ms"}:
        return pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit=unit, utc=True, errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return pd.to_datetime(s, utc=True, errors="coerce")

def folder_value_columns(family: str, df: pd.DataFrame, time_col: int) -> List[int]:
    after = [int(c) for c in df.columns if int(c) > int(time_col)]

    if family in {"switch", "furnace", "door", "motion", "meter"}:
        return after[:1]

    numeric_after = [c for c in after if numeric_fraction(df[c]) >= 0.70]
    return numeric_after

def value_label_for(family: str, value_position: int) -> str:
    labels = {
        "circuit": ["real_power_w", "apparent_power_va"],
        "phase": ["frequency_hz", "voltage_v"],
        "switch": ["level_or_state"],
        "furnace": ["state"],
        "door": ["state"],
        "motion": ["state"],
        "meter": ["state_or_power"],
    }
    if family in labels and value_position < len(labels[family]):
        return labels[family][value_position]
    if family == "environmental":
        return f"env_v{value_position + 1:02d}"
    return f"v{value_position + 1:02d}"

def entity_from_row_columns(df: pd.DataFrame, time_col: int, family: str) -> pd.Series:
    if time_col == 0:
        return pd.Series([family] * len(df), index=df.index)
    entity_cols = [c for c in df.columns if int(c) < int(time_col)]

    if not entity_cols:
        return pd.Series([family] * len(df), index=df.index)
    parts = []
    for c in entity_cols:
        col = df[c].fillna("").astype(str).str.strip()

        if (col != "").mean() > 0.1:
            parts.append(col)
    if not parts:
        return pd.Series([family] * len(df), index=df.index)
    entity = parts[0]
    for part in parts[1:]:
        entity = entity + "_" + part
    return entity.map(lambda x: safe_name(x))

def role_for_family(family: str, value_position: int) -> str:
    if family in {"circuit", "environmental", "phase"}:
        return "iot_continuous"
    if family in {"switch", "furnace", "door", "motion", "meter"}:
        return "iot_binary"
    return "excluded"

def feature_name(role: str, family: str, entity: str, value_label: str) -> str:
    prefix = {
        "iot_continuous": "cont",
        "iot_binary": "bin",
        "iot_driver": "drv",
        "iot_observability": "obs",
    }.get(role, "x")
    return f"{prefix}__{safe_name(family)}__{safe_name(entity)}__{safe_name(value_label)}"

# Source cell 12

parse_ledger_rows: List[Dict[str, Any]] = []
feature_meta: Dict[str, Dict[str, Any]] = {}
series_parts: Dict[str, List[pd.Series]] = defaultdict(list)

def add_series(name: str, role: str, family: str, source_folder: str, source_file: Path, s: pd.Series, value_label: str, entity: str, aggregation: str) -> None:
    s = s.dropna()
    if s.empty:
        return
    s.name = name
    series_parts[name].append(s)
    feature_meta.setdefault(name, {
        "feature": name,
        "owner_role": role,
        "source_family": family,
        "value_label": value_label,
        "entity": entity,
        "aggregation": aggregation,
        "source_folders": set(),
        "source_file_count": 0,
    })
    feature_meta[name]["source_folders"].add(source_folder)
    feature_meta[name]["source_file_count"] += 1

def parse_one_file(path: Path, source_folder: str) -> None:
    family = family_from_folder(source_folder)
    df = read_noheader_csv(path)
    try:
        rel = str(path.relative_to(SMARTSTAR_ROOT))
    except Exception:
        rel = str(path)
    if df is None or df.empty:
        parse_ledger_rows.append({"relative_path": rel, "folder": source_folder, "family": family, "status": "parse_failed", "reason": "could_not_read_table"})
        return

    time_col, time_unit, score = find_timestamp_column(df)
    if time_col is None:
        parse_ledger_rows.append({"relative_path": rel, "folder": source_folder, "family": family, "status": "parse_failed", "reason": f"no_timestamp_column_score={score:.3f}", "n_cols": df.shape[1], "n_rows": df.shape[0]})
        return

    timestamps = parse_timestamp(df[time_col], time_unit)
    good_time = timestamps.notna()
    if good_time.mean() < 0.50:
        parse_ledger_rows.append({"relative_path": rel, "folder": source_folder, "family": family, "status": "parse_failed", "reason": "timestamp_parse_low_coverage", "time_col": time_col, "time_unit": time_unit, "time_score": score})
        return

    df = df.loc[good_time].copy()
    timestamps = timestamps.loc[good_time]
    windows = timestamps.dt.floor(WINDOW_FREQ)
    entity_series = entity_from_row_columns(df, time_col, family)
    val_cols = folder_value_columns(family, df, time_col)

    if not val_cols:
        parse_ledger_rows.append({"relative_path": rel, "folder": source_folder, "family": family, "status": "parse_failed", "reason": "no_value_columns", "time_col": time_col, "time_unit": time_unit, "n_cols": df.shape[1]})
        return

    added = 0
    for pos, val_col in enumerate(val_cols):
        raw_vals = pd.to_numeric(df[val_col], errors="coerce")
        if raw_vals.notna().sum() == 0:
            continue
        value_label = value_label_for(family, pos)
        role = role_for_family(family, pos)
        tmp = pd.DataFrame({
            "window": windows,
            "entity": entity_series.values,
            "value": raw_vals.values,
        }).dropna(subset=["window", "entity", "value"])
        if tmp.empty:
            continue

        for entity, g in tmp.groupby("entity", sort=False):
            if role == "iot_continuous":

                s = g.groupby("window")["value"].mean().sort_index()
                name = feature_name(role, family, entity, value_label)
                add_series(name, role, family, source_folder, path, s, value_label, entity, "window_mean")

                obs_name = f"obs__{safe_name(family)}__{safe_name(entity)}__{safe_name(value_label)}__present"
                obs_s = g.groupby("window")["value"].size().gt(0).astype(float).sort_index()
                add_series(obs_name, "iot_observability", family, source_folder, path, obs_s, "present", entity, "window_any_reported")
                added += 2
            elif role == "iot_binary":

                state = g.copy()
                state["active"] = (pd.to_numeric(state["value"], errors="coerce").fillna(0) > 0).astype(float)
                s = state.groupby("window")["active"].last().sort_index()
                name = feature_name(role, family, entity, "active")
                add_series(name, role, family, source_folder, path, s, "active", entity, "window_last_active")

                drv_event_name = f"drv__{safe_name(family)}__{safe_name(entity)}__reported_event"
                drv_event = state.groupby("window")["active"].size().gt(0).astype(float).sort_index()
                add_series(drv_event_name, "iot_driver", family, source_folder, path, drv_event, "reported_event", entity, "window_any_reported")

                obs_name = f"obs__{safe_name(family)}__{safe_name(entity)}__active__present"
                obs_s = state.groupby("window")["active"].size().gt(0).astype(float).sort_index()
                add_series(obs_name, "iot_observability", family, source_folder, path, obs_s, "present", entity, "window_any_reported")
                added += 3
            else:
                continue

    parse_ledger_rows.append({
        "relative_path": rel,
        "folder": source_folder,
        "family": family,
        "status": "parsed" if added else "no_features_added",
        "time_col": time_col,
        "time_unit": time_unit,
        "time_score": score,
        "value_cols": json.dumps(val_cols),
        "features_added_approx": added,
        "n_cols": df.shape[1],
        "n_rows_after_time_filter": len(df),
    })

for folder in selected_folders:
    folder_path = SMARTSTAR_ROOT / folder
    files = discover_data_files(folder_path)
    print(f"[{folder}] candidate data files: {len(files)}")
    for p in files:
        parse_one_file(p, folder)

parse_ledger = pd.DataFrame(parse_ledger_rows)
save_table(parse_ledger, "parse_ledger.csv", "ledgers")

if not series_parts:
    raise RuntimeError("No feature series were parsed. Check SMARTSTAR_ROOT and data file format.")

combined = {}
for name, parts in series_parts.items():
    role = feature_meta[name]["owner_role"]
    s = pd.concat(parts).sort_index()
    if role == "iot_continuous":
        s = s.groupby(level=0).mean()
    elif role in {"iot_binary", "iot_observability", "iot_driver"}:

        s = s.groupby(level=0).max()
    else:
        s = s.groupby(level=0).last()
    combined[name] = s

start = min(s.index.min() for s in combined.values())
end = max(s.index.max() for s in combined.values())
full_index = pd.date_range(start=start.floor(WINDOW_FREQ), end=end.ceil(WINDOW_FREQ), freq=WINDOW_FREQ, tz="UTC")

feature_table = pd.DataFrame(index=full_index)
for name, s in combined.items():
    feature_table[name] = s.reindex(full_index)

structural_zero_fill_cols_pre = []
for name, meta in feature_meta.items():
    if name not in feature_table.columns:
        continue
    if meta.get("owner_role") in {"iot_observability", "iot_driver"} and meta.get("aggregation") == "window_any_reported":
        structural_zero_fill_cols_pre.append(name)
if structural_zero_fill_cols_pre:
    feature_table[structural_zero_fill_cols_pre] = feature_table[structural_zero_fill_cols_pre].fillna(0.0)

bin_cols = [c for c in feature_table.columns if c.startswith("bin__")]
if bin_cols:
    feature_table[bin_cols] = feature_table[bin_cols].ffill(limit=BINARY_FFILL_LIMIT_WINDOWS)

for col in bin_cols:
    base = col.replace("bin__", "drv__", 1)
    state = feature_table[col]
    prev = state.shift(1)
    change = ((state.notna()) & (prev.notna()) & (state != prev)).astype(float)
    rising = ((state.notna()) & (prev.notna()) & (state > prev)).astype(float)
    change_name = f"{base}__state_change"
    rising_name = f"{base}__rising_edge"
    feature_table[change_name] = change
    feature_table[rising_name] = rising
    feature_meta[change_name] = {
        "feature": change_name,
        "owner_role": "iot_driver",
        "source_family": "derived_from_binary",
        "value_label": "state_change",
        "entity": col,
        "aggregation": "fixed_diff_after_limited_ffill",
        "source_folders": {"derived"},
        "source_file_count": 0,
    }
    feature_meta[rising_name] = {
        "feature": rising_name,
        "owner_role": "iot_driver",
        "source_family": "derived_from_binary",
        "value_label": "rising_edge",
        "entity": col,
        "aggregation": "fixed_diff_after_limited_ffill",
        "source_folders": {"derived"},
        "source_file_count": 0,
    }

structural_zero_fill_rows = []
for c in structural_zero_fill_cols_pre:
    structural_zero_fill_rows.append({
        "feature": c,
        "owner_role": feature_meta.get(c, {}).get("owner_role"),
        "source_family": feature_meta.get(c, {}).get("source_family"),
        "aggregation": feature_meta.get(c, {}).get("aggregation"),
        "zero_semantics": "no_report_or_no_event_in_fixed_60s_window",
    })
for c, meta in feature_meta.items():
    if c in feature_table.columns and meta.get("owner_role") == "iot_driver" and meta.get("source_family") == "derived_from_binary":
        structural_zero_fill_rows.append({
            "feature": c,
            "owner_role": "iot_driver",
            "source_family": "derived_from_binary",
            "aggregation": meta.get("aggregation"),
            "zero_semantics": "no_state_change_or_no_rising_edge_in_fixed_60s_window",
        })
structural_zero_fill_manifest = pd.DataFrame(structural_zero_fill_rows)
save_table(structural_zero_fill_manifest, "structural_zero_fill_manifest.csv", "manifests")

non_obs_cols = [c for c in feature_table.columns if not c.startswith("obs__")]
row_has_signal = feature_table[non_obs_cols].notna().any(axis=1)
feature_table = feature_table.loc[row_has_signal].copy()
feature_table.index.name = "timestamp"
feature_table = feature_table.sort_index()

role_rows = []
for feat, meta in feature_meta.items():
    if feat not in feature_table.columns:
        continue
    src_folders = meta.get("source_folders", set())
    if isinstance(src_folders, set):
        src_folders_out = ";".join(sorted(src_folders))
    else:
        src_folders_out = str(src_folders)
    role_rows.append({
        "feature": feat,
        "owner_role": meta["owner_role"],
        "source_family": meta.get("source_family"),
        "source_folders": src_folders_out,
        "value_label": meta.get("value_label"),
        "entity": meta.get("entity"),
        "aggregation": meta.get("aggregation"),
        "source_file_count": meta.get("source_file_count", 0),
        "structural_zero_semantics": bool(feat in set(structural_zero_fill_manifest["feature"].tolist()) if not structural_zero_fill_manifest.empty else False),
        "n_nonmissing_total": int(feature_table[feat].notna().sum()),
        "n_unique_nonmissing_total": int(pd.Series(feature_table[feat].dropna().unique()).shape[0]) if feature_table[feat].notna().any() else 0,
    })
role_owner_manifest = pd.DataFrame(role_rows).sort_values(["owner_role", "feature"])

feature_csv_path = SMARTSTAR_OUT / "features" / "smartstar_homeA_features_60s.csv"
feature_table.reset_index().to_csv(feature_csv_path, index=False)
try:
    feature_parquet_path = SMARTSTAR_OUT / "features" / "smartstar_homeA_features_60s.parquet"
    feature_table.reset_index().to_parquet(feature_parquet_path, index=False)
except Exception as e:
    feature_parquet_path = None
    print("Parquet write skipped:", repr(e))

save_table(role_owner_manifest, "role_owner_manifest.csv", "manifests")
save_table(parse_ledger, "parse_ledger.csv", "ledgers")

build_summary = {
    "feature_table_rows": int(feature_table.shape[0]),
    "feature_table_columns": int(feature_table.shape[1]),
    "start_utc": str(feature_table.index.min()),
    "end_utc": str(feature_table.index.max()),
    "feature_csv_path": "features/smartstar_homeA_features_60s.csv",
    "feature_csv_sha256": sha256_file(feature_csv_path),
    "feature_parquet_path": "features/smartstar_homeA_features_60s.parquet" if feature_parquet_path else None,
    "role_counts": role_owner_manifest["owner_role"].value_counts().to_dict(),
    "parse_status_counts": parse_ledger["status"].value_counts().to_dict() if not parse_ledger.empty else {},
}
write_json(SMARTSTAR_OUT / "manifests" / "feature_build_summary.json", build_summary)

print(json.dumps(build_summary, indent=2))
display(role_owner_manifest.groupby("owner_role").size().reset_index(name="feature_count"))
display(parse_ledger["status"].value_counts().reset_index(name="count").rename(columns={"index":"status"}))

# Source cell 14

n = len(feature_table)
if n < 50:
    raise RuntimeError(f"Feature table too short for a meaningful chronological split: {n} rows")

train_end = int(math.floor(n * TRAIN_FRAC))
val_end = int(math.floor(n * (TRAIN_FRAC + VAL_FRAC)))
if not (0 < train_end < val_end < n):
    raise RuntimeError("Invalid split boundaries; check TRAIN/VAL/TEST fractions and row count.")

split_label = pd.Series(index=feature_table.index, dtype="object")
split_label.iloc[:train_end] = "TRAIN"
split_label.iloc[train_end:val_end] = "VAL"
split_label.iloc[val_end:] = "TEST"

split_manifest = pd.DataFrame({
    "split": ["TRAIN", "VAL", "TEST"],
    "start_timestamp": [str(feature_table.index[0]), str(feature_table.index[train_end]), str(feature_table.index[val_end])],
    "end_timestamp": [str(feature_table.index[train_end - 1]), str(feature_table.index[val_end - 1]), str(feature_table.index[-1])],
    "row_start_inclusive": [0, train_end, val_end],
    "row_end_exclusive": [train_end, val_end, n],
    "rows": [train_end, val_end - train_end, n - val_end],
    "policy": ["fit/support only", "select/calibrate only", "final audit only"]
})
save_table(split_manifest, "split_manifest.csv", "manifests")

train_idx = split_label == "TRAIN"
val_idx = split_label == "VAL"
test_idx = split_label == "TEST"
train_val_idx = train_idx | val_idx

elig_rows = []
for _, r in role_owner_manifest.iterrows():
    col = r["feature"]
    s = feature_table[col]
    n_train_nonmissing = int(s[train_idx].notna().sum())
    n_val_nonmissing = int(s[val_idx].notna().sum())
    n_train_val_nonmissing = int(s[train_val_idx].notna().sum())
    n_test_nonmissing = int(s[test_idx].notna().sum())
    eligible = (n_train_val_nonmissing >= MIN_TRAIN_VAL_NONMISSING) and (n_val_nonmissing >= MIN_VAL_NONMISSING)
    reason = "eligible_train_val_coverage" if eligible else "excluded_train_val_low_coverage"
    elig_rows.append({
        "feature": col,
        "owner_role": r["owner_role"],
        "eligible": bool(eligible),
        "eligibility_reason": reason,
        "n_train_nonmissing": n_train_nonmissing,
        "n_val_nonmissing": n_val_nonmissing,
        "n_train_val_nonmissing": n_train_val_nonmissing,
        "n_test_nonmissing_report_only": n_test_nonmissing,
    })
feature_eligibility = pd.DataFrame(elig_rows)

if MAX_FEATURES_PER_ROLE_FOR_SMOKE is not None:
    capped = []
    for role, g in feature_eligibility.groupby("owner_role", sort=False):
        eligible_features = g[g["eligible"]].sort_values("n_train_val_nonmissing", ascending=False).head(int(MAX_FEATURES_PER_ROLE_FOR_SMOKE))["feature"].tolist()
        capped.append(g.assign(eligible=lambda x, keep=set(eligible_features): x["eligible"] & x["feature"].isin(keep)))
    feature_eligibility = pd.concat(capped, ignore_index=True)
    feature_eligibility.loc[~feature_eligibility["eligible"], "eligibility_reason"] = "excluded_predeclared_role_cap_or_low_coverage"

save_table(feature_eligibility, "feature_eligibility_train_val_only.csv", "ledgers")

print(split_manifest)
display(feature_eligibility.groupby(["owner_role", "eligible"]).size().reset_index(name="count"))

# Source cell 16

EPS = 1e-9

def clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def finite_values(s: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr

def ks_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if SCIPY_AVAILABLE:

        try:
            return float(stats.ks_2samp(a, b, method="asymp").statistic)
        except TypeError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return float(stats.ks_2samp(a, b).statistic)

    grid = np.unique(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))

def normalized_wasserstein(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if SCIPY_AVAILABLE:
        wd = float(stats.wasserstein_distance(a, b))
    else:
        aa = np.sort(a)
        bb = np.sort(b)
        m = min(len(aa), len(bb))
        wd = float(np.mean(np.abs(np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(aa)), aa) - np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(bb)), bb))))
    scale = float(np.nanpercentile(a, 75) - np.nanpercentile(a, 25))
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.nanmax(a) - np.nanmin(a)) if len(a) else np.nan
    if not np.isfinite(scale) or scale <= EPS:
        scale = max(float(np.nanstd(a)), EPS)
    return float(wd / max(scale, EPS))

def acf1(s: Sequence[float]) -> float:
    x = np.asarray(s, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    x0 = x[:-1]
    x1 = x[1:]
    if np.nanstd(x0) <= EPS or np.nanstd(x1) <= EPS:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])

def binary_run_lengths(x: Sequence[float], value: int = 1) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.array([], dtype=int)
    b = (arr > 0.5).astype(int)
    runs = []
    current = 0
    for v in b:
        if v == value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return np.asarray(runs, dtype=int)

def binary_transition_rate(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    b = (arr > 0.5).astype(int)
    return float(np.mean(np.abs(np.diff(b)) > 0))

def event_rate(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(arr > 0.5))

def burst_count(x: Sequence[float]) -> int:
    return int(len(binary_run_lengths(x, value=1)))

def interarrival_gaps(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    idx = np.where(arr > 0.5)[0]
    if len(idx) < 2:
        return np.array([], dtype=int)
    return np.diff(idx)

def ratio_status(ratio: float, pass_range: Sequence[float], warn_range: Sequence[float]) -> str:
    if not np.isfinite(ratio):
        return "fatal"
    if pass_range[0] <= ratio <= pass_range[1]:
        return "pass"
    if warn_range[0] <= ratio <= warn_range[1]:
        return "warning"
    return "fatal"

def threshold_status(value: float, pass_max: float, warning_max: float, larger_is_worse: bool = True) -> str:
    if not np.isfinite(value):
        return "fatal"
    if larger_is_worse:
        if value <= pass_max:
            return "pass"
        if value <= warning_max:
            return "warning"
        return "fatal"
    raise NotImplementedError

def eval_continuous(real: pd.Series, syn: pd.Series) -> Dict[str, Any]:
    mask = real.notna() & syn.notna()
    rv = finite_values(real[mask])
    sv = finite_values(syn[mask])
    out: Dict[str, Any] = {"n_eval": int(len(rv))}
    if len(rv) < 10 or len(sv) < 10:
        out.update({"ks": np.nan, "normalized_wasserstein": np.nan, "acf1_absdiff": np.nan, "status": "fatal", "status_reason": "insufficient_observed_intersection"})
        return out
    ks = ks_distance(rv, sv)
    nw = normalized_wasserstein(rv, sv)
    acf_diff = abs(acf1(rv) - acf1(sv))
    gates = FROZEN_POLICY["readiness_gates"]["continuous"]
    status = worst_status([
        threshold_status(ks, gates["ks_pass_max"], gates["ks_warning_max"]),
        threshold_status(nw, gates["nw_pass_max"], gates["nw_warning_max"]),
        threshold_status(acf_diff, gates["acf1_absdiff_pass_max"], gates["acf1_absdiff_warning_max"]),
    ])
    out.update({"ks": ks, "normalized_wasserstein": nw, "acf1_absdiff": acf_diff, "status": status, "status_reason": "worst_of_ks_nw_acf1"})
    return out

def eval_binary(real: pd.Series, syn: pd.Series, gates_key: str = "binary") -> Dict[str, Any]:
    mask = real.notna() & syn.notna()
    rv = finite_values(real[mask])
    sv = finite_values(syn[mask])
    out: Dict[str, Any] = {"n_eval": int(min(len(rv), len(sv)))}
    if len(rv) < 10 or len(sv) < 10:
        out.update({"rate_real": np.nan, "rate_syn": np.nan, "rate_absdiff": np.nan, "transition_absdiff": np.nan, "run_ks": np.nan, "status": "fatal", "status_reason": "insufficient_observed_intersection"})
        return out
    rv = (rv > 0.5).astype(float)
    sv = (sv > 0.5).astype(float)
    rate_real = float(np.mean(rv))
    rate_syn = float(np.mean(sv))
    rate_absdiff = abs(rate_real - rate_syn)
    tr_absdiff = abs(binary_transition_rate(rv) - binary_transition_rate(sv))
    run_ks = ks_distance(binary_run_lengths(rv), binary_run_lengths(sv))
    if not np.isfinite(run_ks):
        run_ks = 0.0 if len(binary_run_lengths(rv)) == len(binary_run_lengths(sv)) else 1.0
    gates = FROZEN_POLICY["readiness_gates"][gates_key]
    status = worst_status([
        threshold_status(rate_absdiff, gates["rate_absdiff_pass_max"], gates["rate_absdiff_warning_max"]),
        threshold_status(tr_absdiff, gates["transition_absdiff_pass_max"], gates["transition_absdiff_warning_max"]),
        threshold_status(run_ks, gates["run_ks_pass_max"], gates["run_ks_warning_max"]),
    ])
    out.update({
        "rate_real": rate_real,
        "rate_syn": rate_syn,
        "rate_absdiff": rate_absdiff,
        "transition_absdiff": tr_absdiff,
        "run_ks": run_ks,
        "status": status,
        "status_reason": "worst_of_rate_transition_run_ks",
    })
    return out

def eval_driver(real: pd.Series, syn: pd.Series) -> Dict[str, Any]:
    mask = real.notna() & syn.notna()
    rv = finite_values(real[mask])
    sv = finite_values(syn[mask])
    out: Dict[str, Any] = {"n_eval": int(min(len(rv), len(sv)))}
    if len(rv) < 10 or len(sv) < 10:
        out.update({"rate_ratio": np.nan, "burst_ratio": np.nan, "interarrival_ks": np.nan, "status": "fatal", "status_reason": "insufficient_observed_intersection"})
        return out
    rv = (rv > 0.5).astype(float)
    sv = (sv > 0.5).astype(float)
    rr = (event_rate(sv) + EPS) / (event_rate(rv) + EPS)
    br = (burst_count(sv) + EPS) / (burst_count(rv) + EPS)
    ia_ks = ks_distance(interarrival_gaps(rv), interarrival_gaps(sv))
    if not np.isfinite(ia_ks):
        ia_ks = 0.0 if len(interarrival_gaps(rv)) == len(interarrival_gaps(sv)) else 1.0
    gates = FROZEN_POLICY["readiness_gates"]["driver"]
    status = worst_status([
        ratio_status(rr, gates["rate_ratio_pass_range"], gates["rate_ratio_warning_range"]),
        ratio_status(br, gates["burst_ratio_pass_range"], gates["burst_ratio_warning_range"]),
        threshold_status(ia_ks, gates["interarrival_ks_pass_max"], gates["interarrival_ks_warning_max"]),
    ])
    out.update({
        "rate_real": event_rate(rv),
        "rate_syn": event_rate(sv),
        "rate_ratio": rr,
        "burst_real": burst_count(rv),
        "burst_syn": burst_count(sv),
        "burst_ratio": br,
        "interarrival_ks": ia_ks,
        "status": status,
        "status_reason": "worst_of_rate_ratio_burst_ratio_interarrival_ks",
    })
    return out

def sample_empirical(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.full(n, np.nan)
    return rng.choice(vals, size=n, replace=True)

def cont_generate(model: str, train_s: pd.Series, target_index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    train_vals = clean_numeric(train_s)
    if model == "empirical_iid":
        arr = sample_empirical(train_vals.dropna().to_numpy(), len(target_index), rng)
    elif model == "minute_of_day":
        train_df = pd.DataFrame({"v": train_vals.values}, index=train_s.index).dropna()
        if train_df.empty:
            arr = np.full(len(target_index), np.nan)
        else:
            train_df["minute_of_day"] = train_df.index.hour * 60 + train_df.index.minute
            groups = {k: g["v"].to_numpy(dtype=float) for k, g in train_df.groupby("minute_of_day")}
            global_vals = train_df["v"].to_numpy(dtype=float)
            out = []
            for ts in target_index:
                key = ts.hour * 60 + ts.minute
                vals = groups.get(key, global_vals)
                out.append(rng.choice(vals))
            arr = np.asarray(out, dtype=float)
    elif model == "block_bootstrap":
        vals = train_vals.dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            arr = np.full(len(target_index), np.nan)
        else:
            block = min(120, max(10, int(np.sqrt(len(vals)))))
            chunks = []
            while sum(len(c) for c in chunks) < len(target_index):
                if len(vals) <= block:
                    start = 0
                else:
                    start = int(rng.integers(0, len(vals) - block + 1))
                chunks.append(vals[start:start + block])
            arr = np.concatenate(chunks)[:len(target_index)]
    else:
        raise ValueError(model)
    return pd.Series(arr, index=target_index)

def bernoulli_by_p(p: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if not np.isfinite(p):
        p = 0.0
    p = min(max(float(p), 0.0), 1.0)
    return (rng.random(n) < p).astype(float)

def binary_generate(model: str, train_s: pd.Series, target_index: pd.DatetimeIndex, rng: np.random.Generator, driver_mode: bool = False) -> pd.Series:
    x = clean_numeric(train_s).dropna()
    x = (x > 0.5).astype(float)
    n = len(target_index)
    if len(x) == 0:
        return pd.Series(np.zeros(n), index=target_index)
    if model == "bernoulli_iid":
        arr = bernoulli_by_p(float(x.mean()), n, rng)
    elif model == "minute_of_day_bernoulli":
        train_df = pd.DataFrame({"v": x.values}, index=x.index)
        train_df["minute_of_day"] = train_df.index.hour * 60 + train_df.index.minute
        probs = train_df.groupby("minute_of_day")["v"].mean().to_dict()
        global_p = float(x.mean())
        arr = np.asarray([rng.random() < probs.get(ts.hour * 60 + ts.minute, global_p) for ts in target_index], dtype=float)
    elif model == "markov_1":
        b = x.to_numpy(dtype=int)
        if len(b) < 2:
            arr = np.full(n, float(b[-1] if len(b) else 0.0))
        else:
            prev = b[:-1]
            nxt = b[1:]
            p01 = (np.sum((prev == 0) & (nxt == 1)) + 1) / (np.sum(prev == 0) + 2)
            p11 = (np.sum((prev == 1) & (nxt == 1)) + 1) / (np.sum(prev == 1) + 2)
            arr = np.zeros(n, dtype=float)
            arr[0] = b[-1]
            for i in range(1, n):
                p = p11 if arr[i - 1] > 0.5 else p01
                arr[i] = 1.0 if rng.random() < p else 0.0
    elif model == "event_block_bootstrap":
        vals = x.to_numpy(dtype=float)
        block = min(120, max(10, int(np.sqrt(len(vals)))))
        chunks = []
        while sum(len(c) for c in chunks) < n:
            if len(vals) <= block:
                start = 0
            else:
                start = int(rng.integers(0, len(vals) - block + 1))
            chunks.append(vals[start:start + block])
        arr = np.concatenate(chunks)[:n]
    else:
        raise ValueError(model)
    return pd.Series(arr.astype(float), index=target_index)

def candidate_models_for_role(role: str) -> List[str]:
    return list(FROZEN_POLICY["candidate_registry"].get(role, []))

def evaluate_by_role(role: str, real: pd.Series, syn: pd.Series) -> Dict[str, Any]:
    if role == "iot_continuous":
        return eval_continuous(real, syn)
    if role == "iot_binary":
        return eval_binary(real, syn, "binary")
    if role == "iot_driver":
        return eval_driver(real, syn)
    if role == "iot_observability":
        return eval_binary(real, syn, "observability")
    return {"status": "excluded", "status_reason": "unsupported_role"}

def generate_by_role(role: str, model: str, train_s: pd.Series, target_index: pd.DatetimeIndex, key: str) -> pd.Series:
    rng = stable_rng(RANDOM_SEED, key + "::" + model + "::" + str(len(target_index)))
    if role == "iot_continuous":
        return cont_generate(model, train_s, target_index, rng)
    if role in {"iot_binary", "iot_observability"}:
        return binary_generate(model, train_s, target_index, rng)
    if role == "iot_driver":
        return binary_generate(model, train_s, target_index, rng, driver_mode=True)
    raise ValueError(role)

# Source cell 18

eligible_features = feature_eligibility[feature_eligibility["eligible"]]["feature"].tolist()
elig_role = dict(zip(feature_eligibility["feature"], feature_eligibility["owner_role"]))

train_index = feature_table.index[train_idx]
val_index = feature_table.index[val_idx]
test_index = feature_table.index[test_idx]

selection_rows = []
test_rows = []
syn_val_parts: Dict[str, pd.Series] = {}
syn_test_parts: Dict[str, pd.Series] = {}

role_order = ["iot_observability", "iot_continuous", "iot_binary", "iot_driver"]

def matching_obs_col(feature: str) -> Optional[str]:
    if feature.startswith("cont__"):
        suffix = feature.replace("cont__", "obs__", 1) + "__present"
        return suffix if suffix in feature_table.columns else None
    if feature.startswith("bin__"):
        suffix = feature.replace("bin__", "obs__", 1) + "__present"
        return suffix if suffix in feature_table.columns else None
    return None

for role in role_order:
    role_features = [f for f in eligible_features if elig_role.get(f) == role]
    print(f"Selecting role {role}: {len(role_features)} eligible features")
    for feature in role_features:
        real_train = feature_table.loc[train_index, feature]
        real_val = feature_table.loc[val_index, feature]
        real_test = feature_table.loc[test_index, feature]
        models = candidate_models_for_role(role)
        if not models:
            selection_rows.append({"feature": feature, "owner_role": role, "selected_model": None, "selection_status": "excluded", "reason": "no_models_for_role"})
            continue

        candidate_scores = []
        for model in models:
            try:
                syn_val = generate_by_role(role, model, real_train, val_index, f"{feature}::VAL")

                obs_col = matching_obs_col(feature)
                if obs_col and obs_col in syn_val_parts and role in {"iot_continuous", "iot_binary"}:
                    syn_val = syn_val.mask(syn_val_parts[obs_col].reindex(val_index).fillna(0) < 0.5)
                ev = evaluate_by_role(role, real_val, syn_val)

                if role == "iot_continuous":
                    score = STATUS_ORDER.get(ev["status"], 99) * 1000 + np.nan_to_num(ev.get("ks", np.nan), nan=10) + np.nan_to_num(ev.get("normalized_wasserstein", np.nan), nan=10) + np.nan_to_num(ev.get("acf1_absdiff", np.nan), nan=10)
                elif role in {"iot_binary", "iot_observability"}:
                    score = STATUS_ORDER.get(ev["status"], 99) * 1000 + np.nan_to_num(ev.get("rate_absdiff", np.nan), nan=10) + np.nan_to_num(ev.get("transition_absdiff", np.nan), nan=10) + np.nan_to_num(ev.get("run_ks", np.nan), nan=10)
                elif role == "iot_driver":

                    rr_pen = abs(np.log(np.nan_to_num(ev.get("rate_ratio", np.nan), nan=1e6)))
                    br_pen = abs(np.log(np.nan_to_num(ev.get("burst_ratio", np.nan), nan=1e6)))
                    score = STATUS_ORDER.get(ev["status"], 99) * 1000 + rr_pen + br_pen + np.nan_to_num(ev.get("interarrival_ks", np.nan), nan=10)
                else:
                    score = 1e9
                candidate_scores.append((score, model, ev, syn_val))
            except Exception as e:
                candidate_scores.append((1e12, model, {"status": "fatal", "status_reason": f"candidate_exception:{type(e).__name__}:{str(e)[:120]}"}, pd.Series(index=val_index, dtype=float)))

        candidate_scores.sort(key=lambda x: x[0])
        best_score, best_model, best_val_ev, best_syn_val = candidate_scores[0]
        syn_val_parts[feature] = best_syn_val

        syn_test = generate_by_role(role, best_model, real_train, test_index, f"{feature}::TEST")
        obs_col = matching_obs_col(feature)
        obs_applied = False
        if obs_col and obs_col in syn_test_parts and role in {"iot_continuous", "iot_binary"}:
            syn_test = syn_test.mask(syn_test_parts[obs_col].reindex(test_index).fillna(0) < 0.5)
            obs_applied = True
        syn_test_parts[feature] = syn_test
        test_ev = evaluate_by_role(role, real_test, syn_test)

        selection_rows.append({
            "feature": feature,
            "owner_role": role,
            "selected_model": best_model,
            "selection_status": best_val_ev.get("status"),
            "selection_score": float(best_score),
            "val_status_reason": best_val_ev.get("status_reason"),
            **{f"val_{k}": v for k, v in best_val_ev.items() if k not in {"status", "status_reason"}},
        })
        test_rows.append({
            "feature": feature,
            "owner_role": role,
            "selected_model": best_model,
            "test_status": test_ev.get("status"),
            "test_status_reason": test_ev.get("status_reason"),
            "synthetic_observability_mask_applied": bool(obs_applied),
            "matching_obs_col": obs_col,
            **{f"test_{k}": v for k, v in test_ev.items() if k not in {"status", "status_reason"}},
        })

selection_ledger = pd.DataFrame(selection_rows)
test_evidence_ledger = pd.DataFrame(test_rows)
save_table(selection_ledger, "candidate_selection_val_only_ledger.csv", "ledgers")
save_table(test_evidence_ledger, "test_evidence_q1_q2_q3_ledger.csv", "ledgers")

if syn_test_parts and WRITE_SYNTHETIC_TEST_ARTIFACT:
    synthetic_test = pd.concat(
        [s.reindex(test_index).rename(feat) for feat, s in syn_test_parts.items()],
        axis=1,
    )
    synthetic_test.index.name = "timestamp"
    syn_csv = SMARTSTAR_OUT / "synthetic" / "smartstar_synthetic_test_projection.csv"
    synthetic_test.reset_index().to_csv(syn_csv, index=False)
    try:
        synthetic_test.reset_index().to_parquet(SMARTSTAR_OUT / "synthetic" / "smartstar_synthetic_test_projection.parquet", index=False)
    except Exception as e:
        print("Synthetic parquet write skipped:", repr(e))
else:
    synthetic_test = pd.DataFrame(index=test_index)

print("VAL selection rows:", selection_ledger.shape)
print("TEST evidence rows:", test_evidence_ledger.shape)
display(test_evidence_ledger.groupby(["owner_role", "test_status"]).size().reset_index(name="count"))

# Source cell 20


branch_summary = (
    test_evidence_ledger
    .groupby(["owner_role", "test_status"], dropna=False)
    .size()
    .reset_index(name="count")
    .pivot(index="owner_role", columns="test_status", values="count")
    .fillna(0)
    .astype(int)
    .reset_index()
)
for col in ["pass", "warning", "fatal", "blocker", "unsupported", "not_assessed", "excluded"]:
    if col not in branch_summary.columns:
        branch_summary[col] = 0
branch_summary["total_audited"] = branch_summary[["pass", "warning", "fatal", "blocker", "unsupported", "not_assessed", "excluded"]].sum(axis=1)
branch_summary["dominant_claim_consequence"] = branch_summary.apply(lambda r: (
    "warning_governed_or_pass" if (r.get("fatal", 0) == 0 and r.get("blocker", 0) == 0 and r.get("total_audited", 0) > 0)
    else "scope_contains_fatal_or_blocked_evidence"
), axis=1)
save_table(branch_summary, "branch_readiness_summary.csv", "tables")

blocker_rows = []
for _, r in test_evidence_ledger.iterrows():
    if r.get("test_status") in {"fatal", "blocker"}:
        blocker_rows.append({
            "scope": r["feature"],
            "owner_role": r["owner_role"],
            "dimension": "Q1/Q2/Q3_by_role",
            "blocker_type": r.get("test_status"),
            "evidence": r.get("test_status_reason"),
            "claim_consequence": "feature cannot support a positive smoke-test readiness claim without warning/exclusion",
        })
blocker_rows.extend([
    {
        "scope": "Smart* Home A smoke test",
        "owner_role": "protocol",
        "dimension": "Q4_cross_modal_consistency",
        "blocker_type": "unsupported",
        "evidence": "selected Smart* traces contain electrical/environmental/operational signals but no independent router/Zigbee/Z-Wave protocol stream",
        "claim_consequence": "Q4 physical-network coupling is not instantiated and must not be claimed",
    },
    {
        "scope": "Smart* Home A smoke test",
        "owner_role": "all_roles",
        "dimension": "Q5_scope_compatible_utility",
        "blocker_type": "not_assessed",
        "evidence": "no predeclared downstream or anomaly-detection task is included in this smoke test",
        "claim_consequence": "no Q5 anomaly-detection or downstream utility claim is made",
    },
])
blocker_ledger = pd.DataFrame(blocker_rows)
save_table(blocker_ledger, "blocker_ledger.csv", "ledgers")

q4_scope_status = pd.DataFrame([
    {
        "dataset": "Smart* Home A 2013 smoke test",
        "q_dimension": "Q4 cross-modal consistency",
        "status": "unsupported",
        "reason": "no independent network/protocol stream in selected Smart* traces",
        "allowed_interpretation": "governance layer correctly marks Q4 absent",
        "blocked_interpretation": "physical-network event-response validation or transfer of original Q4 readiness",
    }
])
save_table(q4_scope_status, "q4_scope_status.csv", "tables")

q5_scope_status = pd.DataFrame([
    {
        "dataset": "Smart* Home A 2013 smoke test",
        "q_dimension": "Q5 scope-compatible utility",
        "status": "not_assessed",
        "reason": "no predeclared task and no leakage-safe anomaly benchmark in this smoke test",
        "allowed_interpretation": "no utility claim; only governance re-instantiation",
        "blocked_interpretation": "downstream/anomaly detection utility generalization",
    }
])
save_table(q5_scope_status, "q5_scope_status.csv", "tables")

release_rows = []
for _, r in test_evidence_ledger.iterrows():
    st = r.get("test_status")
    role = r.get("owner_role")
    release_decision = "keep_warning_governed" if st in {"pass", "warning"} else "exclude_from_smoke_release_projection"
    release_rows.append({
        "feature": r["feature"],
        "owner_role": role,
        "test_status": st,
        "selected_model": r.get("selected_model"),
        "release_decision": release_decision,
        "release_scope": "public_source_synthetic_projection",
        "release_warning": "Smart* source is public/CC-BY; this projection is not a formal privacy guarantee and does not transfer original Q6 residential-risk claims",
    })
q6_release_manifest = pd.DataFrame(release_rows)
save_table(q6_release_manifest, "q6_public_source_release_manifest.csv", "manifests")

claim_scope_table = pd.DataFrame([
    {
        "claim": "Governance contract can be re-instantiated on a public weaker smart-home dataset",
        "status": "supported_with_scope_limits",
        "evidence": "role_owner_manifest, split_manifest, VAL-only selection ledger, TEST evidence ledger, blocker ledger, release manifest",
        "not_claimed": "numerical transfer of original restricted-deployment readiness counts",
    },
    {
        "claim": "Smart* validates full smart-home CPS Synthetic Fusion readiness",
        "status": "unsupported",
        "evidence": "Smart* lacks independent protocol/router/Zigbee/Z-Wave streams and original restricted roles",
        "not_claimed": "full CPS empirical universality",
    },
    {
        "claim": "Q4 physical-network coupling transfers",
        "status": "unsupported",
        "evidence": "q4_scope_status.csv marks Q4 not instantiated",
        "not_claimed": "event-response coupling or protocol behavior",
    },
    {
        "claim": "Q5 anomaly-detection or downstream utility transfers",
        "status": "not_assessed",
        "evidence": "q5_scope_status.csv marks Q5 not assessed",
        "not_claimed": "anomaly benchmark or downstream task utility",
    },
    {
        "claim": "Public-source release-manifest logic can be executed",
        "status": "supported_as_governance_artifact_only",
        "evidence": "q6_public_source_release_manifest.csv and checksum manifest",
        "not_claimed": "formal privacy guarantee or routine non-inference",
    },
])
save_table(claim_scope_table, "claim_scope_table.csv", "tables")

print("Branch summary")
display(branch_summary)
print("Q4 status")
display(q4_scope_status)
print("Q6 release decisions")
display(q6_release_manifest["release_decision"].value_counts().reset_index(name="count").rename(columns={"index": "release_decision"}))

# Source cell 22

def branch_c2st(role: str, max_rows: int = 5000, max_cols: int = 80) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        return {"owner_role": role, "status": "missing_dependency", "auc_oriented": np.nan, "n_rows_used": 0, "n_cols_used": 0}
    cols = [f for f in test_evidence_ledger.loc[test_evidence_ledger["owner_role"] == role, "feature"].tolist() if f in synthetic_test.columns]
    if not cols:
        return {"owner_role": role, "status": "not_assessed_no_columns", "auc_oriented": np.nan, "n_rows_used": 0, "n_cols_used": 0}

    status_map = dict(zip(test_evidence_ledger["feature"], test_evidence_ledger["test_status"]))
    cols = [c for c in cols if status_map.get(c) in {"pass", "warning", "fatal"}]
    cols = cols[:max_cols]
    real = feature_table.loc[test_index, cols].copy()
    syn = synthetic_test.loc[:, cols].copy()

    fill_values = {}
    for c in cols:
        tr = feature_table.loc[train_index, c]
        if c.startswith(("bin__", "drv__", "obs__")):
            fill_values[c] = float((clean_numeric(tr).dropna() > 0.5).mean() > 0.5) if clean_numeric(tr).notna().any() else 0.0
        else:
            fill_values[c] = float(clean_numeric(tr).median()) if clean_numeric(tr).notna().any() else 0.0
    real = real.fillna(fill_values).astype(float)
    syn = syn.fillna(fill_values).astype(float)
    n_use = min(len(real), len(syn), max_rows)
    if n_use < 50 or len(cols) < 1:
        return {"owner_role": role, "status": "not_assessed_insufficient_rows", "auc_oriented": np.nan, "n_rows_used": n_use, "n_cols_used": len(cols)}
    rng = stable_rng(RANDOM_SEED, "c2st::" + role)
    real_idx = rng.choice(len(real), size=n_use, replace=False)
    syn_idx = rng.choice(len(syn), size=n_use, replace=False)
    X = np.vstack([real.iloc[real_idx].to_numpy(), syn.iloc[syn_idx].to_numpy()])
    y = np.r_[np.zeros(n_use), np.ones(n_use)]
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y)
    clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_eval)[:, 1]
    auc = float(roc_auc_score(y_eval, prob))
    auc_oriented = max(auc, 1 - auc)
    c2st_status = "pass" if auc_oriented <= 0.60 else "warning" if auc_oriented <= 0.70 else "fatal"
    return {"owner_role": role, "status": c2st_status, "auc_oriented": auc_oriented, "n_rows_used": n_use, "n_cols_used": len(cols)}

c2st_rows = [branch_c2st(role) for role in ["iot_continuous", "iot_binary", "iot_driver", "iot_observability"]]
c2st_summary = pd.DataFrame(c2st_rows)
save_table(c2st_summary, "branch_c2st_smoke_diagnostic.csv", "tables")
display(c2st_summary)

# Source cell 24


transfer_table = pd.DataFrame([
    {
        "component": "Dataset",
        "restricted_deployment": "Original residential CPS trace",
        "smartstar_smoke_test": "UMass Smart* Home A 2013 public subset",
        "interpretation": "Second instantiation is weaker/public; not numerical generalization",
    },
    {
        "component": "Roles instantiated",
        "restricted_deployment": "protocol, continuous, binary, drivers, observability",
        "smartstar_smoke_test": "continuous, binary, drivers, observability",
        "interpretation": "Role ownership transfers partially",
    },
    {
        "component": "Protocol/network",
        "restricted_deployment": "router, Zigbee, Z-Wave, OTA",
        "smartstar_smoke_test": "absent",
        "interpretation": "protocol role unsupported",
    },
    {
        "component": "Q4",
        "restricted_deployment": "evaluated but blocked/development-scoped",
        "smartstar_smoke_test": "not instantiated",
        "interpretation": "no independent protocol stream",
    },
    {
        "component": "Split discipline",
        "restricted_deployment": "chronological TRAIN/VAL/TEST",
        "smartstar_smoke_test": "chronological 60/20/20",
        "interpretation": "invariant re-instantiated",
    },
    {
        "component": "Q1/Q2/Q3",
        "restricted_deployment": "role-specific ledgers",
        "smartstar_smoke_test": "role-specific TEST evidence ledgers",
        "interpretation": "procedural transfer only",
    },
    {
        "component": "Q5",
        "restricted_deployment": "baseline fairness + diagnostic consequence evidence",
        "smartstar_smoke_test": "not assessed",
        "interpretation": "no task claim",
    },
    {
        "component": "Q6",
        "restricted_deployment": "warning-governed 46-column release candidate/baseline",
        "smartstar_smoke_test": "public-source release manifest only",
        "interpretation": "no privacy proof",
    },
])
save_table(transfer_table, "supplement_transfer_smoke_test_table.csv", "manuscript")

fig_data = test_evidence_ledger.groupby(["owner_role", "test_status"]).size().unstack(fill_value=0)
fig = plt.figure(figsize=(8, 4.5))
ax = fig.add_subplot(111)
fig_data.plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Smart* Home A smoke test: TEST readiness by role")
ax.set_xlabel("Owner role")
ax.set_ylabel("Feature count")
ax.legend(title="Status", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig_path = SMARTSTAR_OUT / "figures" / "smartstar_branch_readiness_by_role.png"
fig.savefig(fig_path, dpi=200)
plt.show()

main_text = f"""
### Manuscript paragraph: Smart* public-dataset smoke test

To separate procedural transfer from empirical universality, we added a public Smart* Home A smoke test. The selected Smart* traces provide residential electrical, environmental, and operational signals but no independent router, Zigbee, Z-Wave, or other network/protocol stream. We therefore re-instantiated role ownership, chronological split discipline, TRAIN-only fitting, VAL-only candidate selection, TEST-only final audit, Q1/Q2/Q3-style ledgers, unsupported-dimension declarations, and public-source release-manifest logic, while explicitly marking protocol roles and Q4 physical-network coupling as unsupported. The smoke test produced {int(feature_table.shape[0])} aligned {WINDOW_FREQ} windows and {int(feature_table.shape[1])} role-owned features before TRAIN+VAL eligibility filtering. It does not validate the original deployment's numerical readiness outcomes; it demonstrates that the evidence-to-claim contract can be re-applied and can produce a different, narrower readiness matrix.

### Claim sentence

The Smart* smoke test supports procedural re-instantiation of the governance contract, not empirical universality of Synthetic Fusion readiness counts.
""".strip()

(SMARTSTAR_OUT / "manuscript" / "smartstar_smoke_test_paragraph.md").write_text(main_text, encoding="utf-8")
print(main_text)

# Source cell 26


schema_rows = []
for csv_path in sorted(SMARTSTAR_OUT.rglob("*.csv")):
    try:
        df_head = pd.read_csv(csv_path, nrows=5)
        for col in df_head.columns:
            schema_rows.append({
                "file": str(csv_path.relative_to(SMARTSTAR_OUT)),
                "column": col,
                "observed_dtype_head": str(df_head[col].dtype),
                "description": "auto-catalogued; see notebook cell comments for construction rule",
            })
    except Exception as e:
        schema_rows.append({"file": str(csv_path.relative_to(SMARTSTAR_OUT)), "column": "<read_error>", "observed_dtype_head": type(e).__name__, "description": str(e)[:200]})
schema_catalog = pd.DataFrame(schema_rows)
save_table(schema_catalog, "schema_catalog.csv", "manifests")

checksum_rows = []
for p in sorted(SMARTSTAR_OUT.rglob("*")):
    if p.is_file() and p.name != "package_sha256_manifest.csv":
        checksum_rows.append({
            "relative_path": str(p.relative_to(SMARTSTAR_OUT)),
            "bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        })
package_sha256_manifest = pd.DataFrame(checksum_rows).sort_values("relative_path")
save_table(package_sha256_manifest, "package_sha256_manifest.csv", "manifests")

package_index = pd.DataFrame([
    {"group": "entry_point", "file": "ARTIFACT_README_STUDY_THESIS_SMARTSTAR.md", "purpose": "reviewer-facing run scope, claim limits, and audit path"},
    {"group": "integrity", "file": "manifests/package_sha256_manifest.csv", "purpose": "SHA256 checksums for generated artifacts"},
    {"group": "raw_integrity", "file": "manifests/raw_file_manifest.csv", "purpose": "raw Smart* selected-folder hashes"},
    {"group": "frozen_policy", "file": "manifests/transfer_policy_manifest_v0.json", "purpose": "predeclared policy, split, roles, gates, and unsupported claims"},
    {"group": "roles", "file": "manifests/role_owner_manifest.csv", "purpose": "unique owner role for each feature"},
    {"group": "structural_zero_semantics", "file": "manifests/structural_zero_fill_manifest.csv", "purpose": "documents which reporting/event indicators are dense 0/1 structural processes"},
    {"group": "split", "file": "manifests/split_manifest.csv", "purpose": "chronological TRAIN/VAL/TEST boundaries"},
    {"group": "selection", "file": "ledgers/candidate_selection_val_only_ledger.csv", "purpose": "VAL-only candidate selection"},
    {"group": "evidence", "file": "ledgers/test_evidence_q1_q2_q3_ledger.csv", "purpose": "TEST-only Q1/Q2/Q3-style evidence"},
    {"group": "blockers", "file": "ledgers/blocker_ledger.csv", "purpose": "fatal/blocker/unsupported/not-assessed claim consequences"},
    {"group": "q4", "file": "tables/q4_scope_status.csv", "purpose": "explicit Q4 unsupported declaration"},
    {"group": "q5", "file": "tables/q5_scope_status.csv", "purpose": "explicit Q5 not-assessed declaration"},
    {"group": "q6", "file": "manifests/q6_public_source_release_manifest.csv", "purpose": "public-source release-governance manifest"},
    {"group": "manuscript", "file": "manuscript/supplement_transfer_smoke_test_table.csv", "purpose": "supplement-ready transfer table"},
])
save_table(package_index, "package_index.csv", "manifests")

readme = f"""
# Smart* Home A STUDY-THESIS Governance-Transfer Smoke Test Artifact

## Scope

This package is a public-dataset smoke test for the Synthetic Fusion claim-governance contract. It supports only procedural re-instantiation on a weaker public smart-home dataset.

It does **not** claim:

- numerical transfer of the original restricted residential CPS readiness counts;
- full smart-home CPS empirical universality;
- Q4 physical-network coupling validation;
- Q5 anomaly-detection or downstream utility benchmarking;
- formal privacy guarantee or routine non-inference.

## Dataset

- Dataset: UMass Smart* Home Dataset, 2013 Home A subset.
- Selected folders: {', '.join(selected_folders)}.
- Windowing: {WINDOW_FREQ}.
- Split: chronological TRAIN={TRAIN_FRAC}, VAL={VAL_FRAC}, TEST={TEST_FRAC}.
- Run started UTC: {RUN_STARTED_UTC}.

## One-pass audit path

1. Read `manifests/transfer_policy_manifest_v0.json`.
2. Verify raw hashes in `manifests/raw_file_manifest.csv`.
3. Verify role ownership in `manifests/role_owner_manifest.csv`.
4. Verify structural-zero semantics in `manifests/structural_zero_fill_manifest.csv`.
5. Verify chronological split in `manifests/split_manifest.csv`.
5. Verify VAL-only selection in `ledgers/candidate_selection_val_only_ledger.csv`.
6. Verify TEST-only evidence in `ledgers/test_evidence_q1_q2_q3_ledger.csv`.
7. Verify unsupported Q4/Q5 declarations in `tables/q4_scope_status.csv` and `tables/q5_scope_status.csv`.
8. Verify Q6 public-source release manifest in `manifests/q6_public_source_release_manifest.csv`.
9. Verify checksums in `manifests/package_sha256_manifest.csv`.

## Generated artifact groups

See `manifests/package_index.csv` and `manifests/schema_catalog.csv`.
""".strip() + "\n"

readme_path = SMARTSTAR_OUT / "ARTIFACT_README_STUDY_THESIS_SMARTSTAR.md"
readme_path.write_text(readme, encoding="utf-8")

required_files = [
    "manifests/transfer_policy_manifest_v0.json",
    "manifests/raw_file_manifest.csv",
    "manifests/role_owner_manifest.csv",
    "manifests/structural_zero_fill_manifest.csv",
    "manifests/split_manifest.csv",
    "ledgers/candidate_selection_val_only_ledger.csv",
    "ledgers/test_evidence_q1_q2_q3_ledger.csv",
    "ledgers/blocker_ledger.csv",
    "tables/q4_scope_status.csv",
    "tables/q5_scope_status.csv",
    "manifests/q6_public_source_release_manifest.csv",
    "manuscript/supplement_transfer_smoke_test_table.csv",
    "manifests/package_sha256_manifest.csv",
    "ARTIFACT_README_STUDY_THESIS_SMARTSTAR.md",
]
missing_required = [f for f in required_files if not (SMARTSTAR_OUT / f).exists()]
if missing_required:
    raise RuntimeError("Missing required artifact files: " + repr(missing_required))

final_summary = {
    "artifact_root": f"<local_path>/{SMARTSTAR_OUT.name}",
    "required_files_present": True,
    "n_required_files": len(required_files),
    "feature_table_rows": int(feature_table.shape[0]),
    "feature_table_columns": int(feature_table.shape[1]),
    "test_evidence_rows": int(test_evidence_ledger.shape[0]),
    "branch_summary": branch_summary.to_dict(orient="records"),
    "q4_status": q4_scope_status.to_dict(orient="records"),
    "q5_status": q5_scope_status.to_dict(orient="records"),
    "release_decisions": q6_release_manifest["release_decision"].value_counts().to_dict(),
    "run_finished_utc": utc_now(),
}
write_json(SMARTSTAR_OUT / "manifests" / "final_run_summary.json", final_summary)

print("Artifact package complete:", f"<local_path>/{SMARTSTAR_OUT.name}")
print(json.dumps(final_summary, indent=2, default=str)[:4000])