from __future__ import annotations
def display(obj):
    print(obj.to_string(index=False) if hasattr(obj, "to_string") else obj)

# Source cell 1


import os
import re
import json
import math
import hashlib
import datetime as _dt
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

TONIOT_ROOT = Path(os.environ.get("TONIOT_ROOT", "./data/raw/toniot")).expanduser().resolve()
OUT_ROOT = Path(os.environ.get("TONIOT_OUT_ROOT", "./runs/toniot")).expanduser().resolve()

WINDOW_SECONDS = int(os.environ.get("TONIOT_WINDOW_SECONDS", "60"))
CHUNKSIZE = int(os.environ.get("TONIOT_CHUNKSIZE", "500000"))
MAX_ROWS_PER_FILE = int(os.environ.get("TONIOT_MAX_ROWS_PER_FILE", "0"))
MAX_Q4_PAIRS = int(os.environ.get("TONIOT_MAX_Q4_PAIRS", "50"))
Q4_LAG_WINDOWS = int(os.environ.get("TONIOT_Q4_LAG_WINDOWS", "5"))
MIN_EVENTS_PER_SPLIT = int(os.environ.get("TONIOT_MIN_EVENTS_PER_SPLIT", "8"))
MIN_TOTAL_EVENTS_TRAINVAL = int(os.environ.get("TONIOT_MIN_TOTAL_EVENTS_TRAINVAL", "20"))

HASH_SELECTED_SOURCES = os.environ.get("TONIOT_HASH_SELECTED_SOURCES", "0").strip() == "1"

MANIFEST_DIR = OUT_ROOT / "manifests"
LEDGER_DIR = OUT_ROOT / "ledgers"
TABLE_DIR = OUT_ROOT / "tables"
INTERMEDIATE_DIR = OUT_ROOT / "intermediate"
RELEASE_DIR = OUT_ROOT / "release_candidate"
README_DIR = OUT_ROOT
for d in [MANIFEST_DIR, LEDGER_DIR, TABLE_DIR, INTERMEDIATE_DIR, RELEASE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RUN_CREATED_UTC = _dt.datetime.now(_dt.timezone.utc).isoformat()

print("TONIOT_ROOT =", TONIOT_ROOT)
print("OUT_ROOT    =", OUT_ROOT)
print("WINDOW_SECONDS =", WINDOW_SECONDS)
print("CHUNKSIZE =", CHUNKSIZE)
print("MAX_ROWS_PER_FILE =", MAX_ROWS_PER_FILE if MAX_ROWS_PER_FILE else "ALL")

if not TONIOT_ROOT.exists():
    raise RuntimeError(f"TONIOT_ROOT does not exist: {TONIOT_ROOT}")

def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def sanitize_token(x, max_len: int = 80) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s

def std_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [sanitize_token(c) for c in out.columns]
    return out

def read_csv_head(path: Path, nrows: int = 1000) -> pd.DataFrame:
    try:
        return std_columns(pd.read_csv(path, nrows=nrows, low_memory=False))
    except UnicodeDecodeError:
        return std_columns(pd.read_csv(path, nrows=nrows, encoding="latin1", low_memory=False))

def read_csv_chunks(path: Path, chunksize: int = CHUNKSIZE):
    kwargs = dict(chunksize=chunksize, low_memory=False)
    try:
        yield from pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        yield from pd.read_csv(path, encoding="latin1", **kwargs)

def parse_timestamp(df: pd.DataFrame) -> pd.Series:
    """Parse TON-IoT network or IoT timestamps into UTC pandas timestamps."""
    cols = {c.lower(): c for c in df.columns}
    if "ts" in cols:
        s = df[cols["ts"]]
        numeric = pd.to_numeric(s, errors="coerce")

        if numeric.notna().mean() > 0.8:

            t = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
            valid = t.notna().mean()
            if valid > 0.5:
                return t
        return pd.to_datetime(s, errors="coerce", utc=True)
    if "timestamp" in cols:
        return pd.to_datetime(df[cols["timestamp"]], errors="coerce", utc=True)
    if "date" in cols and "time" in cols:
        return pd.to_datetime(df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str), errors="coerce", utc=True)
    if "date" in cols:
        return pd.to_datetime(df[cols["date"]], errors="coerce", utc=True)
    raise ValueError(f"No timestamp/date/time columns found in {list(df.columns)[:20]}")

def combine_add(acc: pd.DataFrame | None, part: pd.DataFrame) -> pd.DataFrame:
    if part is None or part.empty:
        return acc if acc is not None else pd.DataFrame()
    if acc is None or acc.empty:
        return part.copy()
    return acc.add(part, fill_value=0)

def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path

def save_frame(df: pd.DataFrame, path_base: Path, index: bool = False) -> Path:
    """Prefer parquet; fall back to CSV if parquet engine is missing."""
    path_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = path_base.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=index)
        return parquet_path
    except Exception as e:
        csv_path = path_base.with_suffix(".csv")
        warnings.warn(f"Parquet write failed for {parquet_path}: {e}. Falling back to CSV.")
        df.to_csv(csv_path, index=index)
        return csv_path

def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def truthy_to_float(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    m = s.astype(str).str.strip().str.lower().map({
        "true": 1.0, "t": 1.0, "yes": 1.0, "y": 1.0, "on": 1.0, "open": 1.0, "1": 1.0,
        "false": 0.0, "f": 0.0, "no": 0.0, "n": 0.0, "off": 0.0, "closed": 0.0, "0": 0.0,
    })
    return m

def split_indices(n: int, train_frac: float = 0.60, val_frac: float = 0.20):
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else n
    n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else 0
    return n_train, n_train + n_val

# Source cell 2


csv_files = sorted([p for p in TONIOT_ROOT.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"])
print(f"Discovered {len(csv_files)} CSV files under {TONIOT_ROOT}")

NETWORK_MARKERS = {"ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "duration", "conn_state", "src_bytes", "dst_bytes"}
IOT_MARKERS = {"date", "time", "label", "type"}
SECURITY_PATH_MARKERS = {"securityevents_groundtruth_datasets", "securityevents", "groundtruth", "ground_truth"}

manifest_rows = []
for p in csv_files:
    rel = str(p.relative_to(TONIOT_ROOT))
    parts_lower = {sanitize_token(part) for part in p.relative_to(TONIOT_ROOT).parts}
    try:
        head = read_csv_head(p, nrows=1000)
        cols = set(head.columns)
        error = ""
    except Exception as e:
        head = pd.DataFrame()
        cols = set()
        error = repr(e)
    network_score = len(cols & NETWORK_MARKERS)
    iot_score = len(cols & IOT_MARKERS)
    is_processed = "processed_datasets" in parts_lower
    is_train_test = "train_test_datasets" in parts_lower
    is_raw = "raw_datasets" in parts_lower
    is_security = bool(parts_lower & SECURITY_PATH_MARKERS)
    manifest_rows.append({
        "relative_path": rel,
        "path": str(p),
        "bytes": p.stat().st_size,
        "n_head_columns": len(cols),
        "head_columns": ";".join(list(head.columns)[:80]) if not head.empty else "",
        "network_score": network_score,
        "iot_score": iot_score,
        "is_processed": is_processed,
        "is_train_test": is_train_test,
        "is_raw": is_raw,
        "is_security_groundtruth_path": is_security,
        "read_error": error,
    })

file_manifest = pd.DataFrame(manifest_rows)
file_manifest_path = save_table(file_manifest, MANIFEST_DIR / "toniot_file_discovery_manifest.csv")

processed_network = file_manifest[(file_manifest.is_processed) & (file_manifest.network_score >= 5)].copy()
processed_iot = file_manifest[(file_manifest.is_processed) & (file_manifest.iot_score >= 3) & (file_manifest.network_score < 5)].copy()

fallback_network = file_manifest[(file_manifest.is_train_test) & (file_manifest.network_score >= 5)].copy()
fallback_iot = file_manifest[(file_manifest.is_train_test) & (file_manifest.iot_score >= 3) & (file_manifest.network_score < 5)].copy()
security_files = file_manifest[file_manifest.is_security_groundtruth_path].copy()

if len(processed_network) > 0:
    selected_network = processed_network.sort_values(["bytes", "relative_path"], ascending=[False, True]).copy()
    network_source_policy = "processed_network_primary"
else:
    selected_network = fallback_network.sort_values(["bytes", "relative_path"], ascending=[False, True]).copy()
    network_source_policy = "train_test_network_fallback_chronology_limited"

if len(processed_iot) > 0:
    selected_iot = processed_iot.sort_values(["relative_path"]).copy()
    iot_source_policy = "processed_iot_primary"
else:
    selected_iot = fallback_iot.sort_values(["relative_path"]).copy()
    iot_source_policy = "train_test_iot_fallback_chronology_limited"

if selected_network.empty:
    raise RuntimeError("No network CSV found with TON-IoT network columns. Check TONIOT_ROOT and dataset extraction.")
if selected_iot.empty:
    raise RuntimeError("No IoT CSVs found with TON-IoT IoT columns. Check TONIOT_ROOT and dataset extraction.")

selection_policy = {
    "created_utc": now_utc(),
    "toniot_root": str(TONIOT_ROOT),
    "out_root": str(OUT_ROOT),
    "primary_policy": "Use Processed_datasets for primary STUDY-THESIS transfer evidence; use Train_Test only as fallback.",
    "network_source_policy": network_source_policy,
    "iot_source_policy": iot_source_policy,
    "train_test_warning": "Train_Test_datasets are balanced extracts and should not be used for primary chronological Q4 claims unless processed streams are unavailable.",
    "raw_warning": "Raw_datasets are not used in this notebook because processed streams retain the needed telemetry/protocol columns with lower parsing burden.",
    "security_groundtruth_policy": "SecurityEvents_GroundTruth_datasets are discovered and indexed; timestamped records can be added as event-driver supplements if their schema is compatible.",
    "selected_network_files": selected_network["relative_path"].tolist(),
    "selected_iot_files": selected_iot["relative_path"].tolist(),
    "security_groundtruth_files_detected": security_files["relative_path"].tolist(),
}
write_json(MANIFEST_DIR / "toniot_source_selection_policy.json", selection_policy)
selected_network.to_csv(MANIFEST_DIR / "toniot_selected_network_files.csv", index=False)
selected_iot.to_csv(MANIFEST_DIR / "toniot_selected_iot_files.csv", index=False)
security_files.to_csv(MANIFEST_DIR / "toniot_security_groundtruth_file_manifest.csv", index=False)

print("Selected network policy:", network_source_policy)
display(selected_network[["relative_path", "bytes", "network_score", "head_columns"]].head(10))
print("Selected IoT policy:", iot_source_policy)
display(selected_iot[["relative_path", "bytes", "iot_score", "head_columns"]].head(20))
print("Security ground-truth files detected:", len(security_files))

# Source cell 3


NETWORK_NUMERIC_SUM_COLS = [
    "duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes",
    "dst_pkts", "dst_ip_bytes", "dns_qclass", "dns_qtype", "dns_rcode",
    "http_trans_depth", "http_request_body_len", "http_response_body_len", "http_status_code",
]
NETWORK_BOOL_COLS = ["dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_resumed", "ssl_established", "weird_notice"]
NETWORK_CATEGORICAL_COUNT_COLS = ["proto", "service", "conn_state"]

def aggregate_network_file(path: Path) -> tuple[pd.DataFrame, dict]:
    acc = None
    n_rows_seen = 0
    n_rows_used = 0
    n_bad_time = 0
    chunks = 0
    for raw_chunk in read_csv_chunks(path, CHUNKSIZE):
        chunks += 1
        if MAX_ROWS_PER_FILE and n_rows_seen >= MAX_ROWS_PER_FILE:
            break
        if MAX_ROWS_PER_FILE:
            remaining = MAX_ROWS_PER_FILE - n_rows_seen
            raw_chunk = raw_chunk.iloc[:remaining]
        n_rows_seen += len(raw_chunk)
        chunk = std_columns(raw_chunk)
        try:
            t = parse_timestamp(chunk)
        except Exception as e:
            raise RuntimeError(f"Could not parse timestamp for network file {path}: {e}")
        w = t.dt.floor(f"{WINDOW_SECONDS}s")
        ok = w.notna()
        n_bad_time += int((~ok).sum())
        if ok.sum() == 0:
            continue
        df = chunk.loc[ok].copy()
        df["window_start"] = w.loc[ok].values
        n_rows_used += len(df)
        g = df.groupby("window_start", observed=True)
        part = pd.DataFrame(index=g.size().index)
        part["net_flow_count"] = g.size().astype(float)

        for c in NETWORK_NUMERIC_SUM_COLS:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                part[f"net_sum_{c}"] = vals.groupby(df["window_start"]).sum().astype(float)

        for c in NETWORK_BOOL_COLS:
            if c in df.columns:
                vals = truthy_to_float(df[c]).fillna(0.0)
                part[f"net_count_{c}"] = vals.groupby(df["window_start"]).sum().astype(float)

        if "label" in df.columns:
            lab = pd.to_numeric(df["label"], errors="coerce").fillna(0.0)
            part["net_label_attack_count"] = (lab > 0).astype(float).groupby(df["window_start"]).sum().astype(float)
            part["net_label_observed_count"] = lab.groupby(df["window_start"]).count().astype(float)
        if "type" in df.columns:
            types = df["type"].astype(str).str.strip().str.lower().map(sanitize_token)
            type_dummies = pd.get_dummies(types, prefix="net_type")
            type_dummies["window_start"] = df["window_start"].values
            part = part.join(type_dummies.groupby("window_start", observed=True).sum().astype(float), how="outer")

        for c in NETWORK_CATEGORICAL_COUNT_COLS:
            if c in df.columns:
                cats = df[c].astype(str).str.strip().str.lower().map(sanitize_token)

                top = cats.value_counts(dropna=True).head(30).index
                cats = cats.where(cats.isin(top), other="other")
                dummies = pd.get_dummies(cats, prefix=f"net_{c}")
                dummies["window_start"] = df["window_start"].values
                part = part.join(dummies.groupby("window_start", observed=True).sum().astype(float), how="outer")
        acc = combine_add(acc, part.fillna(0.0))
    if acc is None or acc.empty:
        raise RuntimeError(f"No usable timestamped network rows in {path}")
    acc = acc.sort_index()

    if "net_label_attack_count" in acc.columns and "net_label_observed_count" in acc.columns:
        denom = acc["net_label_observed_count"].replace(0, np.nan)
        acc["net_label_attack_rate"] = (acc["net_label_attack_count"] / denom).fillna(0.0)
    meta = {
        "path": str(path),
        "rows_seen": int(n_rows_seen),
        "rows_used": int(n_rows_used),
        "bad_timestamp_rows": int(n_bad_time),
        "chunks": int(chunks),
        "windows": int(len(acc)),
        "columns": int(acc.shape[1]),
    }
    return acc, meta

network_acc = None
network_metas = []
for p in selected_network["path"].map(Path):
    print("Aggregating network:", p)
    part, meta = aggregate_network_file(p)
    network_metas.append(meta)
    network_acc = combine_add(network_acc, part.fillna(0.0))

network_windows = network_acc.sort_index().reset_index().rename(columns={"index": "window_start"})
network_path = save_frame(network_windows, INTERMEDIATE_DIR / "toniot_network_windows", index=False)
network_meta_path = save_table(pd.DataFrame(network_metas), MANIFEST_DIR / "toniot_network_aggregation_manifest.csv")

print("Network windows:", network_windows.shape)
print("Saved:", network_path)
display(network_windows.head())

# Source cell 4


IOT_EXCLUDE_BASE = {"date", "time", "label", "type"}

def infer_service_name(path: Path) -> str:
    stem = sanitize_token(path.stem)

    stem = re.sub(r"^(processed_|train_test_)?", "", stem)
    stem = re.sub(r"(_dataset|_data|_activity)$", "", stem)
    return stem or "iot_service"

def aggregate_iot_file(path: Path) -> tuple[pd.DataFrame, dict]:
    service = infer_service_name(path)
    acc = None
    n_rows_seen = 0
    n_rows_used = 0
    n_bad_time = 0
    chunks = 0
    for raw_chunk in read_csv_chunks(path, CHUNKSIZE):
        chunks += 1
        if MAX_ROWS_PER_FILE and n_rows_seen >= MAX_ROWS_PER_FILE:
            break
        if MAX_ROWS_PER_FILE:
            remaining = MAX_ROWS_PER_FILE - n_rows_seen
            raw_chunk = raw_chunk.iloc[:remaining]
        n_rows_seen += len(raw_chunk)
        chunk = std_columns(raw_chunk)
        try:
            t = parse_timestamp(chunk)
        except Exception as e:
            raise RuntimeError(f"Could not parse timestamp for IoT file {path}: {e}")
        w = t.dt.floor(f"{WINDOW_SECONDS}s")
        ok = w.notna()
        n_bad_time += int((~ok).sum())
        if ok.sum() == 0:
            continue
        df = chunk.loc[ok].copy()
        df["window_start"] = w.loc[ok].values
        n_rows_used += len(df)
        g = df.groupby("window_start", observed=True)
        part = pd.DataFrame(index=g.size().index)
        part[f"obs_{service}_record_count"] = g.size().astype(float)

        if "label" in df.columns:
            lab = pd.to_numeric(df["label"], errors="coerce").fillna(0.0)
            part[f"driver_{service}_attack_count"] = (lab > 0).astype(float).groupby(df["window_start"]).sum().astype(float)
            part[f"driver_{service}_label_observed_count"] = lab.groupby(df["window_start"]).count().astype(float)
        if "type" in df.columns:
            types = df["type"].astype(str).str.strip().str.lower().map(sanitize_token)
            type_dummies = pd.get_dummies(types, prefix=f"driver_{service}_type")
            type_dummies["window_start"] = df["window_start"].values
            part = part.join(type_dummies.groupby("window_start", observed=True).sum().astype(float), how="outer")

        for c in df.columns:
            if c in IOT_EXCLUDE_BASE or c == "window_start":
                continue

            b = truthy_to_float(df[c])
            b_valid = b.notna().mean()
            numeric = pd.to_numeric(df[c], errors="coerce")
            num_valid = numeric.notna().mean()
            nunique_head = df[c].dropna().astype(str).str.lower().nunique()
            if b_valid > 0.8 and nunique_head <= 4:
                vals = b.fillna(0.0)
                part[f"iot_bin_{service}_{c}_active_count"] = vals.groupby(df["window_start"]).sum().astype(float)
                part[f"iot_bin_{service}_{c}_observed_count"] = vals.groupby(df["window_start"]).count().astype(float)
            elif num_valid > 0.8:
                vals = numeric
                part[f"iot_sum_{service}_{c}"] = vals.fillna(0.0).groupby(df["window_start"]).sum().astype(float)
                part[f"iot_count_{service}_{c}"] = vals.groupby(df["window_start"]).count().astype(float)
            else:

                cats = df[c].astype(str).str.strip().str.lower().map(sanitize_token)
                top = cats.value_counts(dropna=True).head(20).index
                cats = cats.where(cats.isin(top), other="other")
                dummies = pd.get_dummies(cats, prefix=f"iot_cat_{service}_{c}")
                dummies["window_start"] = df["window_start"].values
                part = part.join(dummies.groupby("window_start", observed=True).sum().astype(float), how="outer")
        acc = combine_add(acc, part.fillna(0.0))
    if acc is None or acc.empty:
        raise RuntimeError(f"No usable timestamped IoT rows in {path}")
    acc = acc.sort_index()

    derived = pd.DataFrame(index=acc.index)
    for c in list(acc.columns):
        m_sum = re.match(rf"iot_sum_{re.escape(service)}_(.+)", c)
        if m_sum:
            base = m_sum.group(1)
            count_col = f"iot_count_{service}_{base}"
            if count_col in acc.columns:
                denom = acc[count_col].replace(0, np.nan)
                derived[f"iot_cont_{service}_{base}_mean"] = (acc[c] / denom).replace([np.inf, -np.inf], np.nan)
        m_bin = re.match(rf"iot_bin_{re.escape(service)}_(.+)_active_count", c)
        if m_bin:
            base = m_bin.group(1)
            count_col = f"iot_bin_{service}_{base}_observed_count"
            if count_col in acc.columns:
                denom = acc[count_col].replace(0, np.nan)
                derived[f"iot_binary_{service}_{base}_active_rate"] = (acc[c] / denom).replace([np.inf, -np.inf], np.nan)

    keep_cols = [c for c in acc.columns if c.startswith("driver_") or c.startswith("obs_") or c.startswith("iot_cat_")]
    final = acc[keep_cols].join(derived, how="outer").sort_index()
    meta = {
        "path": str(path),
        "service": service,
        "rows_seen": int(n_rows_seen),
        "rows_used": int(n_rows_used),
        "bad_timestamp_rows": int(n_bad_time),
        "chunks": int(chunks),
        "windows": int(len(final)),
        "columns": int(final.shape[1]),
    }
    return final, meta

iot_acc = None
iot_metas = []
for p in selected_iot["path"].map(Path):
    print("Aggregating IoT:", p)
    part, meta = aggregate_iot_file(p)
    iot_metas.append(meta)
    iot_acc = part if iot_acc is None else iot_acc.join(part, how="outer")

iot_windows = iot_acc.sort_index().reset_index().rename(columns={"index": "window_start"})
iot_path = save_frame(iot_windows, INTERMEDIATE_DIR / "toniot_iot_windows", index=False)
iot_meta_path = save_table(pd.DataFrame(iot_metas), MANIFEST_DIR / "toniot_iot_aggregation_manifest.csv")

print("IoT windows:", iot_windows.shape)
print("Saved:", iot_path)
display(iot_windows.head())

# Source cell 5


nw = network_windows.copy()
iw = iot_windows.copy()
nw["window_start"] = pd.to_datetime(nw["window_start"], utc=True, errors="coerce")
iw["window_start"] = pd.to_datetime(iw["window_start"], utc=True, errors="coerce")
nw = nw.dropna(subset=["window_start"]).set_index("window_start").sort_index()
iw = iw.dropna(subset=["window_start"]).set_index("window_start").sort_index()

network_index = nw.index.unique()
iot_index = iw.index.unique()
overlap_index = network_index.intersection(iot_index).sort_values()

alignment_summary = {
    "created_utc": now_utc(),
    "network_windows": int(len(network_index)),
    "iot_windows": int(len(iot_index)),
    "overlap_windows": int(len(overlap_index)),
    "network_start": str(network_index.min()) if len(network_index) else None,
    "network_end": str(network_index.max()) if len(network_index) else None,
    "iot_start": str(iot_index.min()) if len(iot_index) else None,
    "iot_end": str(iot_index.max()) if len(iot_index) else None,
    "overlap_start": str(overlap_index.min()) if len(overlap_index) else None,
    "overlap_end": str(overlap_index.max()) if len(overlap_index) else None,
    "q4_overlap_claim_status": "eligible" if len(overlap_index) >= 100 else "unsupported_insufficient_time_overlap",
}
write_json(MANIFEST_DIR / "toniot_stream_alignment_summary.json", alignment_summary)

if len(overlap_index) < 30:
    raise RuntimeError(
        "Insufficient timestamp overlap between processed network and IoT streams. "
        "Q4 cannot be audited chronologically without manual timestamp reconciliation. "
        f"Overlap windows: {len(overlap_index)}"
    )

aligned = nw.loc[overlap_index].join(iw.loc[overlap_index], how="inner")

count_like = [c for c in aligned.columns if any(s in c for s in ["count", "flow", "type_", "cat_", "attack", "obs_"])]
aligned[count_like] = aligned[count_like].fillna(0.0)
aligned = aligned.reset_index()

n = len(aligned)
train_end, val_end = split_indices(n, train_frac=0.60, val_frac=0.20)
aligned["split"] = "TEST"
aligned.loc[:train_end-1, "split"] = "TRAIN"
aligned.loc[train_end:val_end-1, "split"] = "VAL"

aligned_path = save_frame(aligned, INTERMEDIATE_DIR / "toniot_aligned_windows_with_split", index=False)

split_manifest = pd.DataFrame([
    {"split": "TRAIN", "row_start": 0, "row_end_exclusive": train_end, "n_windows": train_end,
     "start_time": str(aligned.loc[0, "window_start"]), "end_time": str(aligned.loc[train_end-1, "window_start"])},
    {"split": "VAL", "row_start": train_end, "row_end_exclusive": val_end, "n_windows": val_end-train_end,
     "start_time": str(aligned.loc[train_end, "window_start"]), "end_time": str(aligned.loc[val_end-1, "window_start"])},
    {"split": "TEST", "row_start": val_end, "row_end_exclusive": n, "n_windows": n-val_end,
     "start_time": str(aligned.loc[val_end, "window_start"]), "end_time": str(aligned.loc[n-1, "window_start"])},
])
split_manifest_path = save_table(split_manifest, MANIFEST_DIR / "toniot_chronological_split_manifest.csv")

print("Aligned windows:", aligned.shape)
print("Saved:", aligned_path)
display(pd.DataFrame([alignment_summary]))
display(split_manifest)

# Source cell 6


def infer_role_for_aligned_column(c: str, s: pd.Series | None = None) -> tuple[str, str]:
    cl = c.lower()
    if c in {"window_start", "split"}:
        return "excluded", "time/split control column; not a generated quality target"
    if cl.startswith("net_"):
        return "protocol", "network/protocol aggregate derived from TON-IoT network stream"
    if cl.startswith("driver_"):
        return "iot_driver", "attack/type/event-driver aggregate derived from TON-IoT IoT label/type stream"
    if cl.startswith("obs_"):
        return "iot_observability", "record-count/reporting-process aggregate"
    if cl.startswith("iot_binary_") or cl.startswith("iot_cat_"):
        return "iot_binary", "IoT binary/state/category occupancy aggregate"
    if cl.startswith("iot_cont_"):

        if any(tok in cl for tok in ["status", "state", "signal"]):
            return "iot_binary", "numeric/bool state-like IoT feature aggregated as active rate"
        return "iot_continuous", "numeric IoT telemetry value aggregate"
    if any(tok in cl for tok in ["ip", "uri", "query", "subject", "issuer", "user_agent"]):
        return "excluded", "identifier/high-cardinality protocol content excluded from release claims"
    return "excluded", "unrecognized derived column; excluded by conservative transfer role policy"

role_rows = []
for c in aligned.columns:
    role, reason = infer_role_for_aligned_column(c, aligned[c] if c in aligned.columns else None)
    role_rows.append({
        "column": c,
        "owner_role": role,
        "role_reason": reason,
        "release_default": "exclude" if role == "excluded" else "candidate_warning_governed",
    })
role_manifest = pd.DataFrame(role_rows)
role_manifest_path = save_table(role_manifest, MANIFEST_DIR / "toniot_role_owner_manifest.csv")

ambig = role_manifest[
    role_manifest["column"].str.contains("status|state|signal|type_|label|attack|obs_|cat_", case=False, regex=True)
    | (role_manifest["owner_role"] == "excluded")
].copy()
ambig["review_action"] = "confirm single owner role under TON-IoT transfer codebook"
ambig_path = save_table(ambig, LEDGER_DIR / "toniot_role_assignment_ambiguity_queue.csv")

numeric_cols = [c for c in aligned.columns if c not in {"window_start", "split"} and pd.api.types.is_numeric_dtype(aligned[c])]
role_map = dict(zip(role_manifest["column"], role_manifest["owner_role"]))

q_rows = []
for split_name, df in aligned.groupby("split", sort=False):
    for role, cols in role_manifest.groupby("owner_role"):
        role_cols = [c for c in cols["column"].tolist() if c in numeric_cols]
        if not role_cols:
            continue
        sub = df[role_cols]
        nonzero_rate = float((sub.fillna(0.0) != 0).mean().mean()) if sub.size else np.nan
        missing_rate = float(sub.isna().mean().mean()) if sub.size else np.nan
        q_rows.append({
            "split": split_name,
            "owner_role": role,
            "n_numeric_columns": len(role_cols),
            "mean_missing_rate": missing_rate,
            "mean_nonzero_rate": nonzero_rate,
            "q1_inventory_status": "pass" if len(role_cols) > 0 else "excluded",
            "claim_scope": "TON-IoT real-trace governance-transfer inventory; not synthetic fidelity",
        })
q1_inventory = pd.DataFrame(q_rows)
q1_path = save_table(q1_inventory, LEDGER_DIR / "toniot_q1_inventory_by_role.csv")

def lag1_autocorr(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(v) < 3 or np.nanstd(v) == 0:
        return np.nan
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])

q2_rows = []
for split_name, df in aligned.groupby("split", sort=False):
    for c in numeric_cols:
        role = role_map.get(c, "excluded")
        q2_rows.append({"split": split_name, "column": c, "owner_role": role, "lag1_autocorr": lag1_autocorr(df[c])})
q2_col = pd.DataFrame(q2_rows)
q2_summary = q2_col.groupby(["split", "owner_role"], dropna=False).agg(
    n_columns=("column", "count"),
    mean_lag1_autocorr=("lag1_autocorr", "mean"),
    median_lag1_autocorr=("lag1_autocorr", "median"),
).reset_index()
q2_col_path = save_table(q2_col, LEDGER_DIR / "toniot_q2_lag1_autocorr_by_column.csv")
q2_summary_path = save_table(q2_summary, LEDGER_DIR / "toniot_q2_lag1_autocorr_summary.csv")

obs_cols = role_manifest.loc[role_manifest.owner_role == "iot_observability", "column"].tolist()
q3_rows = []
for split_name, df in aligned.groupby("split", sort=False):
    if obs_cols:
        obs = df[obs_cols].fillna(0.0)
        q3_rows.append({
            "split": split_name,
            "n_observability_columns": len(obs_cols),
            "mean_record_count": float(obs.mean().mean()),
            "zero_observation_window_rate": float((obs.sum(axis=1) == 0).mean()),
            "claim_scope": "observability inventory for transfer trace; not release safety or synthetic realism",
        })
q3_inventory = pd.DataFrame(q3_rows)
q3_path = save_table(q3_inventory, LEDGER_DIR / "toniot_q3_observability_inventory.csv")

print("Role manifest:", role_manifest.shape)
display(role_manifest["owner_role"].value_counts().rename_axis("owner_role").reset_index(name="count"))
print("Ambiguity queue:", ambig.shape)
display(q1_inventory)

# Source cell 7


train_df = aligned[aligned["split"] == "TRAIN"].reset_index(drop=True)
val_df = aligned[aligned["split"] == "VAL"].reset_index(drop=True)
test_df = aligned[aligned["split"] == "TEST"].reset_index(drop=True)
trainval_df = aligned[aligned["split"].isin(["TRAIN", "VAL"])].reset_index(drop=True)

role_manifest = pd.read_csv(MANIFEST_DIR / "toniot_role_owner_manifest.csv")
role_map = dict(zip(role_manifest["column"], role_manifest["owner_role"]))

def as_binary_events(s: pd.Series) -> np.ndarray:
    v = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    active = v > 0
    starts = active & np.r_[True, ~active[:-1]]
    return starts.astype(int)

def robust_z_from_train(x_train: pd.Series, x: pd.Series) -> np.ndarray:
    tr = pd.to_numeric(x_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vals = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = float(tr.median(skipna=True)) if tr.notna().any() else 0.0
    q25 = float(tr.quantile(0.25)) if tr.notna().any() else 0.0
    q75 = float(tr.quantile(0.75)) if tr.notna().any() else 0.0
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 0:
        scale = float(tr.std(skipna=True)) if tr.notna().sum() > 1 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return ((vals.fillna(med) - med) / scale).to_numpy(dtype=float)

def event_profile(event_vec: np.ndarray, response_vec: np.ndarray, L: int = Q4_LAG_WINDOWS):
    event_idx = np.where(event_vec > 0)[0]
    usable = event_idx[(event_idx - L >= 0) & (event_idx + L < len(response_vec))]
    if len(usable) == 0:
        return None, 0
    mats = []
    for idx in usable:
        mats.append(response_vec[idx-L:idx+L+1])
    mat = np.vstack(mats)
    return np.nanmean(mat, axis=0), int(len(usable))

def profile_metrics(ref_profile: np.ndarray, cmp_profile: np.ndarray, L: int = Q4_LAG_WINDOWS):
    if ref_profile is None or cmp_profile is None:
        return {"eta_similarity": np.nan, "lag_error_windows": np.nan, "response_window_rel_error": np.nan, "status": "blocker"}
    if np.nanstd(ref_profile) == 0 or np.nanstd(cmp_profile) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(ref_profile, cmp_profile)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    mae = float(np.nanmean(np.abs(ref_profile - cmp_profile)))
    scale = float(np.nanmean(np.abs(ref_profile)))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    nmae = mae / scale
    eta = 0.5 * max(0.0, corr) + 0.5 * math.exp(-nmae)
    post = slice(L, 2*L + 1)
    lag_ref = int(np.nanargmax(ref_profile[post]))
    lag_cmp = int(np.nanargmax(cmp_profile[post]))
    lag_error = abs(lag_ref - lag_cmp)
    resp_ref = float(np.nanmean(ref_profile[post]))
    resp_cmp = float(np.nanmean(cmp_profile[post]))
    response_window_rel_error = abs(resp_ref - resp_cmp) / max(abs(resp_ref), 1e-9)
    if eta >= 0.70 and lag_error <= 2 and response_window_rel_error <= 0.25:
        status = "pass"
    elif eta >= 0.50 and lag_error <= 5 and response_window_rel_error <= 0.50:
        status = "warning"
    else:
        status = "blocker"
    return {
        "eta_similarity": float(eta),
        "lag_error_windows": int(lag_error),
        "response_window_rel_error": float(response_window_rel_error),
        "status": status,
    }

driver_cols = [c for c, r in role_map.items() if r == "iot_driver" and c in aligned.columns]
protocol_cols = [c for c, r in role_map.items() if r == "protocol" and c in aligned.columns and pd.api.types.is_numeric_dtype(aligned[c])]

candidate_drivers = []
for c in driver_cols:
    ev_tr = as_binary_events(train_df[c]) if c in train_df else np.array([])
    ev_val = as_binary_events(val_df[c]) if c in val_df else np.array([])
    ev_tv = as_binary_events(trainval_df[c]) if c in trainval_df else np.array([])
    if ev_tv.sum() >= MIN_TOTAL_EVENTS_TRAINVAL and ev_tr.sum() >= MIN_EVENTS_PER_SPLIT and ev_val.sum() >= MIN_EVENTS_PER_SPLIT:
        candidate_drivers.append(c)

candidate_responses = []
for c in protocol_cols:
    v = pd.to_numeric(trainval_df[c], errors="coerce").fillna(0.0)
    if len(v) and float(v.std()) > 0 and float((v != 0).mean()) > 0.01:
        candidate_responses.append(c)

response_var = {c: float(pd.to_numeric(trainval_df[c], errors="coerce").fillna(0.0).var()) for c in candidate_responses}
candidate_responses = sorted(candidate_responses, key=lambda c: response_var.get(c, 0.0), reverse=True)[:80]

pair_rows = []
for d in candidate_drivers:
    ev_train = as_binary_events(train_df[d])
    ev_val = as_binary_events(val_df[d])
    for z in candidate_responses:
        z_train_all = pd.to_numeric(train_df[z], errors="coerce").fillna(0.0)
        z_val_all = pd.to_numeric(val_df[z], errors="coerce").fillna(0.0)
        z_train = robust_z_from_train(z_train_all, z_train_all)
        z_val = robust_z_from_train(z_train_all, z_val_all)
        p_train, n_ev_train = event_profile(ev_train, z_train, Q4_LAG_WINDOWS)
        p_val, n_ev_val = event_profile(ev_val, z_val, Q4_LAG_WINDOWS)
        if n_ev_train < MIN_EVENTS_PER_SPLIT or n_ev_val < MIN_EVENTS_PER_SPLIT:
            continue
        m = profile_metrics(p_train, p_val, Q4_LAG_WINDOWS)
        pair_rows.append({
            "driver_column": d,
            "protocol_response_column": z,
            "train_events": n_ev_train,
            "val_events": n_ev_val,
            "trainval_status": m["status"],
            "trainval_eta_similarity": m["eta_similarity"],
            "trainval_lag_error_windows": m["lag_error_windows"],
            "trainval_response_window_rel_error": m["response_window_rel_error"],
        })

pair_scores = pd.DataFrame(pair_rows)
if pair_scores.empty:
    frozen_manifest = pd.DataFrame(columns=[
        "pair_id", "driver_column", "protocol_response_column", "train_events", "val_events",
        "trainval_status", "trainval_eta_similarity", "trainval_lag_error_windows",
        "trainval_response_window_rel_error", "freeze_rank", "freeze_reason"
    ])
    q4_freeze_status = "unsupported_no_trainval_eligible_pairs"
else:

    pair_scores["status_rank"] = pair_scores["trainval_status"].map({"pass": 0, "warning": 1, "blocker": 2}).fillna(3)
    pair_scores = pair_scores.sort_values(
        ["status_rank", "trainval_eta_similarity", "train_events", "val_events"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
    frozen_manifest = pair_scores.head(MAX_Q4_PAIRS).copy()
    frozen_manifest.insert(0, "pair_id", [f"TONIOT_Q4_PAIR_{i:03d}" for i in range(len(frozen_manifest))])
    frozen_manifest["freeze_rank"] = np.arange(1, len(frozen_manifest) + 1)
    frozen_manifest["freeze_reason"] = "selected using TRAIN/VAL only before TEST audit"
    q4_freeze_status = "frozen_trainval_pairs_for_test_audit"

pair_scores_path = save_table(pair_scores, LEDGER_DIR / "toniot_q4_trainval_pair_scores.csv")
frozen_manifest_path = save_table(frozen_manifest, MANIFEST_DIR / "toniot_q4_manifest_trainval_frozen.csv")

freeze_meta = {
    "created_utc": now_utc(),
    "q4_transfer_scope": "TON-IoT event-driver to network/protocol response stability; not synthetic Q4 realism",
    "selection_split": "TRAIN+VAL only",
    "test_use_before_freeze": False,
    "lag_windows": Q4_LAG_WINDOWS,
    "window_seconds": WINDOW_SECONDS,
    "min_events_per_split": MIN_EVENTS_PER_SPLIT,
    "min_total_events_trainval": MIN_TOTAL_EVENTS_TRAINVAL,
    "candidate_driver_columns": len(candidate_drivers),
    "candidate_protocol_response_columns": len(candidate_responses),
    "trainval_pairs_scored": int(len(pair_scores)),
    "frozen_pairs": int(len(frozen_manifest)),
    "freeze_status": q4_freeze_status,
    "gate_policy": {
        "pass": "ETA>=0.70 and lag_error<=2 windows and response_window_rel_error<=0.25",
        "warning": "ETA>=0.50 and lag_error<=5 windows and response_window_rel_error<=0.50",
        "blocker": "otherwise or insufficient evidence",
    },
}
write_json(MANIFEST_DIR / "toniot_q4_freeze_manifest.json", freeze_meta)

print("Candidate drivers:", len(candidate_drivers))
print("Candidate protocol responses:", len(candidate_responses))
print("Train/VAL pairs scored:", len(pair_scores))
print("Frozen pairs:", len(frozen_manifest))
print("Freeze status:", q4_freeze_status)
display(frozen_manifest.head(20))

# Source cell 8


frozen_manifest = pd.read_csv(MANIFEST_DIR / "toniot_q4_manifest_trainval_frozen.csv")
trainval_df = aligned[aligned["split"].isin(["TRAIN", "VAL"])].reset_index(drop=True)
test_df = aligned[aligned["split"] == "TEST"].reset_index(drop=True)

if frozen_manifest.empty:
    q4_test_audit = pd.DataFrame([{
        "pair_id": "NONE",
        "driver_column": None,
        "protocol_response_column": None,
        "test_status": "unsupported_no_frozen_pairs",
        "claim_consequence": "No ToN-IoT Q4 transfer pair could be frozen from TRAIN/VAL evidence.",
    }])
else:
    rows = []
    for _, r in frozen_manifest.iterrows():
        d = r["driver_column"]
        z = r["protocol_response_column"]
        ev_ref = as_binary_events(trainval_df[d])
        ev_test = as_binary_events(test_df[d])
        z_ref_raw = pd.to_numeric(trainval_df[z], errors="coerce").fillna(0.0)
        z_test_raw = pd.to_numeric(test_df[z], errors="coerce").fillna(0.0)
        z_ref = robust_z_from_train(z_ref_raw, z_ref_raw)
        z_test = robust_z_from_train(z_ref_raw, z_test_raw)
        p_ref, n_ev_ref = event_profile(ev_ref, z_ref, Q4_LAG_WINDOWS)
        p_test, n_ev_test = event_profile(ev_test, z_test, Q4_LAG_WINDOWS)
        if n_ev_test < MIN_EVENTS_PER_SPLIT:
            m = {"eta_similarity": np.nan, "lag_error_windows": np.nan, "response_window_rel_error": np.nan, "status": "blocker"}
            consequence = "Blocked: insufficient TEST event support for this frozen pair."
        else:
            m = profile_metrics(p_ref, p_test, Q4_LAG_WINDOWS)
            consequence = {
                "pass": "Supports scoped ToN-IoT Q4 transfer-stability evidence for this pair only.",
                "warning": "Warning-scoped ToN-IoT Q4 transfer-stability evidence for this pair only.",
                "blocker": "Blocked: TEST response profile does not match frozen TRAIN/VAL reference sufficiently.",
            }.get(m["status"], "Blocked or unsupported.")
        rows.append({
            "pair_id": r["pair_id"],
            "driver_column": d,
            "protocol_response_column": z,
            "trainval_status": r.get("trainval_status", None),
            "test_events": int(n_ev_test),
            "reference_events_trainval": int(n_ev_ref),
            "test_status": m["status"],
            "test_eta_similarity": m["eta_similarity"],
            "test_lag_error_windows": m["lag_error_windows"],
            "test_response_window_rel_error": m["response_window_rel_error"],
            "claim_consequence": consequence,
        })
    q4_test_audit = pd.DataFrame(rows)

q4_test_path = save_table(q4_test_audit, LEDGER_DIR / "toniot_q4_test_event_response_audit.csv")

if "test_status" in q4_test_audit.columns:
    status_counts = q4_test_audit["test_status"].value_counts(dropna=False).to_dict()
else:
    status_counts = {}
q4_summary = pd.DataFrame([{
    "q4_scope": "TON-IoT public second-deployment event-response transfer audit",
    "frozen_pairs": int(len(frozen_manifest)),
    "test_pairs_audited": int(len(q4_test_audit)) if not (len(q4_test_audit)==1 and q4_test_audit.iloc[0].get("pair_id") == "NONE") else 0,
    "pass": int(status_counts.get("pass", 0)),
    "warning": int(status_counts.get("warning", 0)),
    "blocker": int(status_counts.get("blocker", 0) + status_counts.get("unsupported_no_frozen_pairs", 0)),
    "claim_interpretation": "Q4 governance re-instantiation on a public trace; not synthetic realism or numerical transfer from the residential deployment.",
}])
q4_summary_path = save_table(q4_summary, TABLE_DIR / "toniot_q4_scope_summary.csv")

write_json(MANIFEST_DIR / "toniot_q4_test_audit_manifest.json", {
    "created_utc": now_utc(),
    "test_only_audit": True,
    "frozen_manifest": str(frozen_manifest_path),
    "test_audit_table": str(q4_test_path),
    "summary_table": str(q4_summary_path),
    "claim_scope": "Q4 transfer-governance smoke test only",
})

print("Q4 TEST audit saved:", q4_test_path)
display(q4_summary)
display(q4_test_audit.head(20))

# Source cell 9


role_manifest = pd.read_csv(MANIFEST_DIR / "toniot_role_owner_manifest.csv")
role_map = dict(zip(role_manifest["column"], role_manifest["owner_role"]))

IDENTIFIER_PATTERNS = re.compile(r"ip|uri|query|subject|issuer|user_agent|mime|identifier|addr|mac", re.IGNORECASE)

release_rows = []
release_cols = ["window_start"]
for c in aligned.columns:
    role = role_map.get(c, "excluded")
    if c in {"window_start", "split"}:
        status = "keep_alignment" if c == "window_start" else "exclude_split_control"
        reason = "alignment column" if c == "window_start" else "split control column"
    elif IDENTIFIER_PATTERNS.search(c):
        status = "exclude_identifier_or_high_cardinality_content"
        reason = "conservative Q6 excludes identifier/high-cardinality content"
    elif role == "excluded":
        status = "exclude_role"
        reason = "excluded role under transfer manifest"
    elif role in {"protocol", "iot_continuous", "iot_binary", "iot_observability"}:
        status = "keep_warning"
        reason = "windowed aggregate public-source transfer feature; warning-governed"
    elif role == "iot_driver":

        status = "keep_warning_public_source_label_driver"
        reason = "public-source attack/type event driver aggregate; warning-governed"
    else:
        status = "exclude_unknown"
        reason = "unknown role"
    keep = status.startswith("keep")
    if keep and c not in release_cols:
        release_cols.append(c)
    release_rows.append({
        "column": c,
        "owner_role": role,
        "q6_release_status": status,
        "q6_keep": bool(keep),
        "q6_reason": reason,
    })

q6_scope = pd.DataFrame(release_rows)
q6_scope_path = save_table(q6_scope, TABLE_DIR / "toniot_q6_release_scope_table.csv")
excluded_path = save_table(q6_scope[~q6_scope["q6_keep"]], TABLE_DIR / "toniot_q6_excluded_columns.csv")

release_df = aligned[release_cols].copy()
release_df_path = save_frame(release_df, RELEASE_DIR / "toniot_windowed_q6_release_candidate", index=False)

q6_summary = q6_scope.groupby(["owner_role", "q6_release_status"], dropna=False).agg(
    columns=("column", "count"), kept=("q6_keep", "sum")
).reset_index()
q6_summary_path = save_table(q6_summary, TABLE_DIR / "toniot_q6_release_summary_by_role.csv")

write_json(MANIFEST_DIR / "toniot_q6_release_manifest.json", {
    "created_utc": now_utc(),
    "release_candidate_path": str(release_df_path),
    "release_scope_table": str(q6_scope_path),
    "release_interpretation": "Windowed aggregate release-governance projection for a public TON-IoT transfer trace; not a privacy guarantee.",
    "kept_columns": int(len(release_cols)),
    "excluded_columns": int((~q6_scope.q6_keep).sum()),
    "identifier_policy": "exclude raw/high-cardinality identifier-like content if present; notebook uses processed aggregate windows as release unit.",
})

print("Q6 release candidate:", release_df.shape)
print("Saved:", release_df_path)
display(q6_summary)

# Source cell 10


q4_summary = pd.read_csv(TABLE_DIR / "toniot_q4_scope_summary.csv")
q6_scope = pd.read_csv(TABLE_DIR / "toniot_q6_release_scope_table.csv")
split_manifest = pd.read_csv(MANIFEST_DIR / "toniot_chronological_split_manifest.csv")
role_manifest = pd.read_csv(MANIFEST_DIR / "toniot_role_owner_manifest.csv")

claim_summary = pd.DataFrame([{
    "artifact": "TON-IoT public second-deployment governance transfer",
    "source_policy": f"network={network_source_policy}; iot={iot_source_policy}",
    "window_seconds": WINDOW_SECONDS,
    "aligned_windows": int(len(aligned)),
    "train_windows": int(split_manifest.loc[split_manifest.split == "TRAIN", "n_windows"].iloc[0]),
    "val_windows": int(split_manifest.loc[split_manifest.split == "VAL", "n_windows"].iloc[0]),
    "test_windows": int(split_manifest.loc[split_manifest.split == "TEST", "n_windows"].iloc[0]),
    "role_counts": json.dumps(role_manifest["owner_role"].value_counts().to_dict(), sort_keys=True),
    "q4_frozen_pairs": int(q4_summary["frozen_pairs"].iloc[0]),
    "q4_pass": int(q4_summary["pass"].iloc[0]),
    "q4_warning": int(q4_summary["warning"].iloc[0]),
    "q4_blocker": int(q4_summary["blocker"].iloc[0]),
    "q6_kept_columns": int(q6_scope["q6_keep"].sum()),
    "q6_excluded_columns": int((~q6_scope["q6_keep"]).sum()),
    "permitted_claim": "Q4/Q6 governance contract re-instantiated on a public trace with independent IoT and network streams.",
    "unsupported_claim": "No synthetic realism, anomaly-detection utility, or numerical transfer from the residential deployment is claimed.",
}])
claim_summary_path = save_table(claim_summary, TABLE_DIR / "toniot_transfer_claim_scope_summary.csv")

readme = f"""# TON-IoT STUDY-THESIS Governance Transfer Artifact

Created UTC: {now_utc()}

This artifact re-instantiates the Synthetic Fusion governance contract on TON-IoT as a public second trace.
It is a Q4/Q6 governance-transfer smoke test, not a full synthetic-data generator benchmark.

Primary source policy:
- Network stream: {network_source_policy}
- IoT stream: {iot_source_policy}
- Raw datasets are not used in the primary run.
- Train_Test datasets are fallback only because balanced extracts may weaken chronological Q4 interpretation.

Main reviewer entry points:
- manifests/toniot_source_selection_policy.json
- manifests/toniot_chronological_split_manifest.csv
- manifests/toniot_role_owner_manifest.csv
- manifests/toniot_q4_manifest_trainval_frozen.csv
- ledgers/toniot_q4_test_event_response_audit.csv
- tables/toniot_q4_scope_summary.csv
- tables/toniot_q6_release_scope_table.csv
- tables/toniot_transfer_claim_scope_summary.csv
- release_candidate/toniot_windowed_q6_release_candidate.*

Claim scope:
The supported claim is that the Q4 event-response manifest/audit and Q6 release-projection logic can be
re-instantiated on a public trace with independent telemetry and network/protocol streams. The artifact does
not claim synthetic realism or anomaly-detection utility.
"""
readme_path = README_DIR / "README_TONIOT_STUDY_THESIS_TRANSFER.md"
readme_path.write_text(readme)

artifact_files = []
for sub in [MANIFEST_DIR, LEDGER_DIR, TABLE_DIR, RELEASE_DIR]:
    artifact_files.extend([p for p in sub.rglob("*") if p.is_file()])
artifact_files.append(readme_path)
artifact_files = sorted(set(artifact_files))

index_rows = []
for p in artifact_files:
    rel = str(p.relative_to(OUT_ROOT))
    try:
        sha = sha256_file(p)
    except Exception as e:
        sha = f"ERROR:{e}"
    index_rows.append({
        "artifact_group": rel.split("/")[0],
        "relative_path": rel,
        "path": str(p),
        "filename": p.name,
        "bytes": p.stat().st_size,
        "sha256": sha,
        "created_utc": now_utc(),
        "claim_scope": "TON-IoT Q4/Q6 governance-transfer artifact",
    })
package_index = pd.DataFrame(index_rows)
package_index_path = save_table(package_index, OUT_ROOT / "TONIOT_STUDY_THESIS_PACKAGE_INDEX.csv")
sha_manifest_path = save_table(package_index[["relative_path", "sha256", "bytes"]], OUT_ROOT / "TONIOT_PACKAGE_SHA256_MANIFEST.csv")

print("Wrote package index:", package_index_path)
print("Wrote checksum manifest:", sha_manifest_path)
display(claim_summary)
display(package_index.head(20))