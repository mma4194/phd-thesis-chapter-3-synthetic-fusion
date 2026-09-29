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
from pathlib import Path

import numpy as np
import pandas as pd

TONIOT_ROOT = Path(os.environ.get("TONIOT_ROOT", "./data/raw/toniot")).expanduser().resolve()
OUT_ROOT = Path(os.environ.get("TONIOT_OUT_ROOT", "./runs/toniot")).expanduser().resolve()

MANIFEST_DIR = OUT_ROOT / "manifests"
LEDGER_DIR = OUT_ROOT / "ledgers"
TABLE_DIR = OUT_ROOT / "tables"
INTERMEDIATE_DIR = OUT_ROOT / "intermediate"
RELEASE_DIR = OUT_ROOT / "release_candidate"
PATCH_DIR = OUT_ROOT / "physical_q4"
PATCH_MANIFEST_DIR = PATCH_DIR / "manifests"
PATCH_LEDGER_DIR = PATCH_DIR / "ledgers"
PATCH_TABLE_DIR = PATCH_DIR / "tables"
PATCH_RELEASE_DIR = PATCH_DIR / "release_candidate"
for d in [PATCH_DIR, PATCH_MANIFEST_DIR, PATCH_LEDGER_DIR, PATCH_TABLE_DIR, PATCH_RELEASE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WINDOW_SECONDS = int(os.environ.get("TONIOT_WINDOW_SECONDS", "60"))
Q4_LAG_WINDOWS = int(os.environ.get("TONIOT_PHYS_Q4_LAG_WINDOWS", os.environ.get("TONIOT_Q4_LAG_WINDOWS", "5")))
MAX_Q4_PAIRS = int(os.environ.get("TONIOT_PHYS_MAX_Q4_PAIRS", os.environ.get("TONIOT_MAX_Q4_PAIRS", "50")))
MIN_EVENTS_PER_SPLIT = int(os.environ.get("TONIOT_PHYS_MIN_EVENTS_PER_SPLIT", "5"))
MIN_TOTAL_EVENTS_TRAINVAL = int(os.environ.get("TONIOT_PHYS_MIN_TOTAL_EVENTS_TRAINVAL", "15"))

OVERWRITE_CANONICAL = os.environ.get("TONIOT_PHYS_OVERWRITE_CANONICAL", "0").strip() == "1"

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

def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path

def save_frame(df: pd.DataFrame, path_base: Path, index: bool = False) -> Path:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = path_base.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=index)
        return parquet_path
    except Exception:
        csv_path = path_base.with_suffix(".csv")
        df.to_csv(csv_path, index=index)
        return csv_path

def load_frame_any(path_base: Path) -> pd.DataFrame:
    candidates = []
    if path_base.suffix:
        candidates.append(path_base)
    else:
        candidates.extend([path_base.with_suffix(".parquet"), path_base.with_suffix(".csv")])
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find any of: {candidates}")

aligned = load_frame_any(INTERMEDIATE_DIR / "toniot_aligned_windows_with_split")
if "split" not in aligned.columns or "window_start" not in aligned.columns:
    raise RuntimeError("Aligned TON-IoT window table must contain window_start and split columns.")

aligned["window_start"] = pd.to_datetime(aligned["window_start"], errors="coerce", utc=True)

role_manifest_path = MANIFEST_DIR / "toniot_role_owner_manifest.csv"
if not role_manifest_path.exists():
    raise FileNotFoundError(role_manifest_path)
role_manifest = pd.read_csv(role_manifest_path)
role_map = dict(zip(role_manifest["column"], role_manifest["owner_role"]))

train_df = aligned[aligned["split"] == "TRAIN"].reset_index(drop=True)
val_df = aligned[aligned["split"] == "VAL"].reset_index(drop=True)
test_df = aligned[aligned["split"] == "TEST"].reset_index(drop=True)
trainval_df = aligned[aligned["split"].isin(["TRAIN", "VAL"])].reset_index(drop=True)

print("Loaded aligned windows:", aligned.shape)
print("Splits:", aligned["split"].value_counts(dropna=False).to_dict())
print("OUT_ROOT:", OUT_ROOT)
print("PATCH_DIR:", PATCH_DIR)
print("Q4_LAG_WINDOWS:", Q4_LAG_WINDOWS)
print("MIN_EVENTS_PER_SPLIT:", MIN_EVENTS_PER_SPLIT)
print("MIN_TOTAL_EVENTS_TRAINVAL:", MIN_TOTAL_EVENTS_TRAINVAL)
print("OVERWRITE_CANONICAL:", OVERWRITE_CANONICAL)

# Source cell 2


LABEL_TOKENS = re.compile(r"(^|_)(label|type|attack|normal|backdoor|ddos|dos|injection|mitm|password|ransomware|scanning|xss)(_|$)", re.I)

def is_label_or_attack_col(c: str) -> bool:
    return bool(LABEL_TOKENS.search(str(c)))

def is_physical_iot_source_col(c: str) -> bool:
    s = str(c).lower()
    if is_label_or_attack_col(s):
        return False
    if s.startswith("iot_cont_") or s.startswith("iot_binary_") or s.startswith("iot_cat_"):
        return True
    return False

physical_source_cols = [
    c for c in aligned.columns
    if c not in {"window_start", "split"}
    and c in aligned.columns
    and is_physical_iot_source_col(c)
    and pd.api.types.is_numeric_dtype(aligned[c])
]

if not physical_source_cols:
    raise RuntimeError("No physical IoT source columns found. Check the upstream IoT aggregation naming.")

def numeric_series(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

def binary_rising_events(values: pd.Series, threshold: float = 0.5) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    active = v > threshold
    starts = active & np.r_[True, ~active[:-1]]
    return starts.astype(int)

def continuous_crossing_events(values: pd.Series, threshold: float, direction: str) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if direction == "high":
        active = v >= threshold
    elif direction == "low":
        active = v <= threshold
    else:
        raise ValueError(direction)
    active = np.where(np.isfinite(v), active, False)
    starts = active & np.r_[True, ~active[:-1]]
    return starts.astype(int)

def continuous_change_events(values: pd.Series, threshold: float) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    dif = np.abs(np.r_[np.nan, np.diff(v)])
    active = np.where(np.isfinite(dif), dif >= threshold, False)
    return active.astype(int)

threshold_rows = []
driver_frames = []

for c in physical_source_cols:
    role = role_map.get(c, "unknown")
    s_tv = numeric_series(trainval_df, c)
    finite_tv = s_tv[np.isfinite(s_tv)]
    if finite_tv.empty:
        continue

    if c.startswith("iot_binary_") or c.startswith("iot_cat_"):
        driver_name = f"physdrv_rise__{c}"
        ev_all = binary_rising_events(aligned[c], threshold=0.5)
        driver_frames.append(pd.Series(ev_all, name=driver_name))
        threshold_rows.append({
            "physical_driver": driver_name,
            "source_column": c,
            "source_role": role,
            "driver_rule": "rising_edge_active_rate_gt_0.5",
            "threshold": 0.5,
            "threshold_split": "TRAIN+VAL only",
            "train_events": int(binary_rising_events(train_df[c], 0.5).sum()),
            "val_events": int(binary_rising_events(val_df[c], 0.5).sum()),
            "test_events": int(binary_rising_events(test_df[c], 0.5).sum()),
        })
    elif c.startswith("iot_cont_"):
        q10 = float(finite_tv.quantile(0.10))
        q90 = float(finite_tv.quantile(0.90))
        dif_tv = np.abs(np.diff(finite_tv.to_numpy(dtype=float)))
        q90_change = float(np.nanquantile(dif_tv, 0.90)) if len(dif_tv) else np.nan
        if not np.isfinite(q90_change) or q90_change <= 0:
            q90_change = float(finite_tv.std())
        if not np.isfinite(q90_change) or q90_change <= 0:
            q90_change = 1.0

        specs = [
            (f"physdrv_high__{c}", "high_crossing_q90", q90, "high"),
            (f"physdrv_low__{c}", "low_crossing_q10", q10, "low"),
            (f"physdrv_change__{c}", "absolute_change_ge_trainval_q90", q90_change, "change"),
        ]
        for driver_name, rule, thr, direction in specs:
            if direction == "change":
                ev_all = continuous_change_events(aligned[c], thr)
                ev_train = continuous_change_events(train_df[c], thr)
                ev_val = continuous_change_events(val_df[c], thr)
                ev_test = continuous_change_events(test_df[c], thr)
            else:
                ev_all = continuous_crossing_events(aligned[c], thr, direction)
                ev_train = continuous_crossing_events(train_df[c], thr, direction)
                ev_val = continuous_crossing_events(val_df[c], thr, direction)
                ev_test = continuous_crossing_events(test_df[c], thr, direction)
            driver_frames.append(pd.Series(ev_all, name=driver_name))
            threshold_rows.append({
                "physical_driver": driver_name,
                "source_column": c,
                "source_role": role,
                "driver_rule": rule,
                "threshold": thr,
                "threshold_split": "TRAIN+VAL only",
                "train_events": int(np.nansum(ev_train)),
                "val_events": int(np.nansum(ev_val)),
                "test_events": int(np.nansum(ev_test)),
            })

physical_driver_matrix = pd.concat(driver_frames, axis=1) if driver_frames else pd.DataFrame(index=aligned.index)
physical_driver_catalog = pd.DataFrame(threshold_rows)
physical_driver_catalog["trainval_events"] = physical_driver_catalog["train_events"] + physical_driver_catalog["val_events"]
physical_driver_catalog["eligible_for_q4_freeze"] = (
    (physical_driver_catalog["train_events"] >= MIN_EVENTS_PER_SPLIT)
    & (physical_driver_catalog["val_events"] >= MIN_EVENTS_PER_SPLIT)
    & (physical_driver_catalog["trainval_events"] >= MIN_TOTAL_EVENTS_TRAINVAL)
)

matrix_path = save_frame(physical_driver_matrix, INTERMEDIATE_DIR / "toniot_physical_q4_driver_matrix", index=False)
catalog_path = save_table(physical_driver_catalog, PATCH_LEDGER_DIR / "toniot_physical_q4_driver_catalog.csv")

print("Physical source columns:", len(physical_source_cols))
print("Derived physical drivers:", len(physical_driver_catalog))
print("Q4-freeze eligible physical drivers:", int(physical_driver_catalog["eligible_for_q4_freeze"].sum()))
print("Saved driver matrix:", matrix_path)
print("Saved driver catalog:", catalog_path)
display(physical_driver_catalog.sort_values(["eligible_for_q4_freeze", "trainval_events"], ascending=[False, False]).head(30))

# Source cell 3


def is_primary_protocol_response(c: str) -> bool:
    s = str(c).lower()
    if not s.startswith("net_"):
        return False
    if s.startswith("net_label_") or s.startswith("net_type_"):
        return False
    if is_label_or_attack_col(s):
        return False
    if s.endswith("_nan") or s in {"net_proto_nan", "net_service_nan", "net_conn_state_nan", "net_service_-"}:
        return False

    if s in {"net_sum_src_ip_bytes", "net_sum_dst_ip_bytes"}:
        return True
    return pd.api.types.is_numeric_dtype(aligned[c])

protocol_response_cols = [c for c in aligned.columns if is_primary_protocol_response(c)]
if not protocol_response_cols:
    raise RuntimeError("No primary protocol response columns found.")

eligible_drivers = physical_driver_catalog.loc[physical_driver_catalog["eligible_for_q4_freeze"], "physical_driver"].tolist()
if not eligible_drivers:
    print("WARNING: No physical drivers satisfy TRAIN/VAL event-support thresholds. Q4 physical transfer will be unsupported.")

driver_train = physical_driver_matrix.loc[aligned["split"].eq("TRAIN")].reset_index(drop=True)
driver_val = physical_driver_matrix.loc[aligned["split"].eq("VAL")].reset_index(drop=True)
driver_trainval = physical_driver_matrix.loc[aligned["split"].isin(["TRAIN", "VAL"])].reset_index(drop=True)
driver_test = physical_driver_matrix.loc[aligned["split"].eq("TEST")].reset_index(drop=True)

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
    event_vec = np.asarray(event_vec, dtype=float)
    response_vec = np.asarray(response_vec, dtype=float)
    event_idx = np.where(event_vec > 0)[0]
    usable = event_idx[(event_idx - L >= 0) & (event_idx + L < len(response_vec))]
    if len(usable) == 0:
        return None, 0
    mats = [response_vec[idx-L:idx+L+1] for idx in usable]
    return np.nanmean(np.vstack(mats), axis=0), int(len(usable))

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
    post = slice(L, 2 * L + 1)
    try:
        lag_ref = int(np.nanargmax(ref_profile[post]))
        lag_cmp = int(np.nanargmax(cmp_profile[post]))
        lag_error = abs(lag_ref - lag_cmp)
    except Exception:
        lag_error = 999
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

response_stats = []
for c in protocol_response_cols:
    v = pd.to_numeric(trainval_df[c], errors="coerce").fillna(0.0)
    response_stats.append({
        "protocol_response_column": c,
        "trainval_variance": float(v.var()),
        "trainval_nonzero_rate": float((v != 0).mean()),
        "eligible": bool(float(v.var()) > 0 and float((v != 0).mean()) > 0.01),
    })
response_catalog = pd.DataFrame(response_stats).sort_values(["eligible", "trainval_variance"], ascending=[False, False])
candidate_responses = response_catalog.loc[response_catalog["eligible"], "protocol_response_column"].head(80).tolist()

pair_rows = []
for d in eligible_drivers:
    ev_train = pd.to_numeric(driver_train[d], errors="coerce").fillna(0).to_numpy(dtype=int)
    ev_val = pd.to_numeric(driver_val[d], errors="coerce").fillna(0).to_numpy(dtype=int)
    for z in candidate_responses:
        z_train_raw = pd.to_numeric(train_df[z], errors="coerce").fillna(0.0)
        z_val_raw = pd.to_numeric(val_df[z], errors="coerce").fillna(0.0)
        z_train = robust_z_from_train(z_train_raw, z_train_raw)
        z_val = robust_z_from_train(z_train_raw, z_val_raw)
        p_train, n_ev_train = event_profile(ev_train, z_train, Q4_LAG_WINDOWS)
        p_val, n_ev_val = event_profile(ev_val, z_val, Q4_LAG_WINDOWS)
        if n_ev_train < MIN_EVENTS_PER_SPLIT or n_ev_val < MIN_EVENTS_PER_SPLIT:
            continue
        m = profile_metrics(p_train, p_val, Q4_LAG_WINDOWS)
        src_col = physical_driver_catalog.loc[physical_driver_catalog["physical_driver"] == d, "source_column"].iloc[0]
        pair_rows.append({
            "driver_column": d,
            "physical_source_column": src_col,
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
        "pair_id", "driver_column", "physical_source_column", "protocol_response_column",
        "train_events", "val_events", "trainval_status", "trainval_eta_similarity",
        "trainval_lag_error_windows", "trainval_response_window_rel_error", "freeze_rank", "freeze_reason"
    ])
    freeze_status = "unsupported_no_trainval_eligible_physical_pairs"
else:
    pair_scores["status_rank"] = pair_scores["trainval_status"].map({"pass": 0, "warning": 1, "blocker": 2}).fillna(3)
    pair_scores = pair_scores.sort_values(
        ["status_rank", "trainval_eta_similarity", "train_events", "val_events"],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
    frozen_manifest = pair_scores.head(MAX_Q4_PAIRS).copy()
    frozen_manifest.insert(0, "pair_id", [f"TONIOT_PHYS_Q4_PAIR_{i:03d}" for i in range(len(frozen_manifest))])
    frozen_manifest["freeze_rank"] = np.arange(1, len(frozen_manifest) + 1)
    frozen_manifest["freeze_reason"] = "physical IoT driver and protocol response selected using TRAIN/VAL only before TEST audit"
    freeze_status = "frozen_physical_trainval_pairs_for_test_audit"

response_catalog_path = save_table(response_catalog, PATCH_LEDGER_DIR / "toniot_physical_q4_protocol_response_catalog.csv")
pair_scores_path = save_table(pair_scores, PATCH_LEDGER_DIR / "toniot_physical_q4_trainval_pair_scores.csv")
frozen_manifest_path = save_table(frozen_manifest, PATCH_MANIFEST_DIR / "toniot_physical_q4_manifest_trainval_frozen.csv")

freeze_meta = {
    "created_utc": now_utc(),
    "q4_transfer_scope": "TON-IoT physical-IoT-event to network/protocol response stability; not synthetic Q4 realism",
    "selection_split": "TRAIN+VAL only",
    "test_use_before_freeze": False,
    "physical_source_columns": len(physical_source_cols),
    "derived_physical_drivers": int(len(physical_driver_catalog)),
    "eligible_physical_drivers": int(len(eligible_drivers)),
    "candidate_protocol_response_columns": int(len(candidate_responses)),
    "trainval_pairs_scored": int(len(pair_scores)),
    "frozen_pairs": int(len(frozen_manifest)),
    "freeze_status": freeze_status,
    "excluded_from_primary_q4": "label/type/attack columns are excluded from both physical drivers and protocol responses",
    "lag_windows": Q4_LAG_WINDOWS,
    "window_seconds": WINDOW_SECONDS,
    "min_events_per_split": MIN_EVENTS_PER_SPLIT,
    "min_total_events_trainval": MIN_TOTAL_EVENTS_TRAINVAL,
    "gate_policy": {
        "pass": "ETA>=0.70 and lag_error<=2 windows and response_window_rel_error<=0.25",
        "warning": "ETA>=0.50 and lag_error<=5 windows and response_window_rel_error<=0.50",
        "blocker": "otherwise or insufficient evidence",
    },
}
write_json(PATCH_MANIFEST_DIR / "toniot_physical_q4_freeze_manifest.json", freeze_meta)

print("Primary protocol responses:", len(protocol_response_cols))
print("Eligible responses after variance/activity filter:", len(candidate_responses))
print("Train/VAL physical pairs scored:", len(pair_scores))
print("Frozen physical Q4 pairs:", len(frozen_manifest))
print("Freeze status:", freeze_status)
display(frozen_manifest.head(20))

# Source cell 4


frozen_manifest = pd.read_csv(PATCH_MANIFEST_DIR / "toniot_physical_q4_manifest_trainval_frozen.csv")

if frozen_manifest.empty:
    q4_test_audit = pd.DataFrame([{
        "pair_id": "NONE",
        "driver_column": None,
        "physical_source_column": None,
        "protocol_response_column": None,
        "test_status": "unsupported_no_frozen_physical_pairs",
        "claim_consequence": "No physical-IoT Q4 transfer pair could be frozen from TRAIN/VAL evidence.",
    }])
else:
    rows = []
    for _, r in frozen_manifest.iterrows():
        d = r["driver_column"]
        z = r["protocol_response_column"]
        ev_ref = pd.to_numeric(driver_trainval[d], errors="coerce").fillna(0).to_numpy(dtype=int)
        ev_test = pd.to_numeric(driver_test[d], errors="coerce").fillna(0).to_numpy(dtype=int)
        z_ref_raw = pd.to_numeric(trainval_df[z], errors="coerce").fillna(0.0)
        z_test_raw = pd.to_numeric(test_df[z], errors="coerce").fillna(0.0)
        z_ref = robust_z_from_train(z_ref_raw, z_ref_raw)
        z_test = robust_z_from_train(z_ref_raw, z_test_raw)
        p_ref, n_ev_ref = event_profile(ev_ref, z_ref, Q4_LAG_WINDOWS)
        p_test, n_ev_test = event_profile(ev_test, z_test, Q4_LAG_WINDOWS)
        if n_ev_test < MIN_EVENTS_PER_SPLIT:
            m = {"eta_similarity": np.nan, "lag_error_windows": np.nan, "response_window_rel_error": np.nan, "status": "blocker"}
            consequence = "Blocked: insufficient TEST physical-event support for this frozen pair."
        else:
            m = profile_metrics(p_ref, p_test, Q4_LAG_WINDOWS)
            consequence = {
                "pass": "Supports scoped TON-IoT physical-event/protocol Q4 transfer-stability evidence for this pair only.",
                "warning": "Warning-scoped TON-IoT physical-event/protocol Q4 transfer-stability evidence for this pair only.",
                "blocker": "Blocked: TEST physical-event/protocol response profile does not match frozen TRAIN/VAL reference sufficiently.",
            }.get(m["status"], "Blocked or unsupported.")
        rows.append({
            "pair_id": r["pair_id"],
            "driver_column": d,
            "physical_source_column": r.get("physical_source_column", None),
            "protocol_response_column": z,
            "trainval_status": r.get("trainval_status", None),
            "reference_events_trainval": int(n_ev_ref),
            "test_events": int(n_ev_test),
            "test_status": m["status"],
            "test_eta_similarity": m["eta_similarity"],
            "test_lag_error_windows": m["lag_error_windows"],
            "test_response_window_rel_error": m["response_window_rel_error"],
            "claim_consequence": consequence,
        })
    q4_test_audit = pd.DataFrame(rows)

q4_test_path = save_table(q4_test_audit, PATCH_LEDGER_DIR / "toniot_physical_q4_test_event_response_audit.csv")
status_counts = q4_test_audit.get("test_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
q4_physical_summary = pd.DataFrame([{
    "q4_scope": "TON-IoT physical-IoT-event to protocol-response second-deployment transfer audit",
    "frozen_pairs": int(len(frozen_manifest)),
    "test_pairs_audited": int(len(q4_test_audit)) if not (len(q4_test_audit) == 1 and q4_test_audit.iloc[0].get("pair_id") == "NONE") else 0,
    "pass": int(status_counts.get("pass", 0)),
    "warning": int(status_counts.get("warning", 0)),
    "blocker": int(status_counts.get("blocker", 0) + status_counts.get("unsupported_no_frozen_physical_pairs", 0)),
    "claim_interpretation": "Primary TON-IoT Q4 governance-transfer result: physical IoT drivers and protocol responses were frozen on TRAIN/VAL and audited on TEST only; not synthetic realism or numerical transfer from the residential deployment.",
}])
q4_summary_path = save_table(q4_physical_summary, PATCH_TABLE_DIR / "toniot_physical_q4_scope_summary.csv")
write_json(PATCH_MANIFEST_DIR / "toniot_physical_q4_test_audit_manifest.json", {
    "created_utc": now_utc(),
    "test_only_audit": True,
    "frozen_manifest": str(frozen_manifest_path),
    "test_audit_table": str(q4_test_path),
    "summary_table": str(q4_summary_path),
    "claim_scope": "Primary TON-IoT physical-Q4 transfer-governance audit",
})

print("Physical Q4 TEST audit saved:", q4_test_path)
display(q4_physical_summary)
display(q4_test_audit.head(20))

# Source cell 5


orig_q6_path = TABLE_DIR / "toniot_q6_release_scope_table.csv"
if not orig_q6_path.exists():
    raise FileNotFoundError(orig_q6_path)
orig_q6 = pd.read_csv(orig_q6_path)

RAW_IDENTIFIER_EXACT = {
    "src_ip", "dst_ip", "dns_query", "http_uri", "ssl_subject", "ssl_issuer", "http_user_agent",
    "net_src_ip", "net_dst_ip", "net_dns_query", "net_http_uri", "net_ssl_subject", "net_ssl_issuer", "net_http_user_agent",
}

AGGREGATE_IP_BYTE_COUNTS = {"net_sum_src_ip_bytes", "net_sum_dst_ip_bytes"}

patched_rows = []
for _, r in orig_q6.iterrows():
    c = str(r["column"])
    owner = r["owner_role"]
    keep = bool(r["q6_keep"])
    status = str(r["q6_release_status"])
    reason = str(r["q6_reason"])

    if c in AGGREGATE_IP_BYTE_COUNTS:
        keep = True
        status = "keep_warning"
        reason = "aggregate IP-byte count protocol feature; not a raw IP identifier; warning-governed"
    elif c in RAW_IDENTIFIER_EXACT:
        keep = False
        status = "exclude_identifier_or_high_cardinality_content"
        reason = "conservative Q6 excludes raw identifier/high-cardinality content"
    elif c == "split":
        keep = False
        status = "exclude_split_control"
        reason = "split control column"

    patched_rows.append({
        "column": c,
        "owner_role": owner,
        "q6_release_status": status,
        "q6_keep": keep,
        "q6_reason": reason,
    })

q6_patched = pd.DataFrame(patched_rows)
q6_patched_path = save_table(q6_patched, PATCH_TABLE_DIR / "toniot_q6_release_scope_table_physical_q4.csv")
release_cols = q6_patched.loc[q6_patched["q6_keep"], "column"].tolist()
release_df = aligned[release_cols].copy()
release_path = save_frame(release_df, PATCH_RELEASE_DIR / "toniot_windowed_q6_release_candidate_physical_q4", index=False)

q6_patched_summary = (
    q6_patched.groupby(["owner_role", "q6_release_status"], dropna=False)
    .agg(columns=("column", "count"), kept=("q6_keep", "sum"))
    .reset_index()
)
q6_summary_path = save_table(q6_patched_summary, PATCH_TABLE_DIR / "toniot_q6_release_summary_by_role_physical_q4.csv")
q6_excluded_path = save_table(q6_patched[~q6_patched["q6_keep"]], PATCH_TABLE_DIR / "toniot_q6_excluded_columns_physical_q4.csv")

write_json(PATCH_MANIFEST_DIR / "toniot_q6_release_manifest_physical_q4.json", {
    "created_utc": now_utc(),
    "source_q6_table": str(orig_q6_path),
    "patched_q6_table": str(q6_patched_path),
    "release_candidate_path": str(release_path),
    "kept_columns": int(q6_patched["q6_keep"].sum()),
    "excluded_columns": int((~q6_patched["q6_keep"]).sum()),
    "patch_reason": "Corrects identifier heuristic: src_ip_bytes/dst_ip_bytes are aggregate byte-count features, not raw IP identifiers.",
    "release_interpretation": "Windowed aggregate release-governance projection for a public TON-IoT transfer trace; not a privacy guarantee.",
})

print("Patched Q6 release candidate:", release_df.shape)
print("Saved:", release_path)
display(q6_patched_summary)
display(q6_patched[~q6_patched["q6_keep"]])

# Source cell 6


orig_claim_summary_path = TABLE_DIR / "toniot_transfer_claim_scope_summary.csv"
orig_claim_summary = pd.read_csv(orig_claim_summary_path) if orig_claim_summary_path.exists() else pd.DataFrame()
role_manifest = pd.read_csv(MANIFEST_DIR / "toniot_role_owner_manifest.csv")

physical_claim_summary = pd.DataFrame([{
    "artifact": "TON-IoT public second-deployment physical-Q4/Q6 governance transfer",
    "source_policy": "network=processed_network_primary; iot=processed_iot_primary",
    "window_seconds": int(WINDOW_SECONDS),
    "aligned_windows": int(len(aligned)),
    "train_windows": int((aligned["split"] == "TRAIN").sum()),
    "val_windows": int((aligned["split"] == "VAL").sum()),
    "test_windows": int((aligned["split"] == "TEST").sum()),
    "role_counts": json.dumps(role_manifest["owner_role"].value_counts().to_dict(), sort_keys=True),
    "physical_source_columns": int(len(physical_source_cols)),
    "derived_physical_drivers": int(len(physical_driver_catalog)),
    "eligible_physical_drivers": int(physical_driver_catalog["eligible_for_q4_freeze"].sum()),
    "q4_frozen_pairs": int(q4_physical_summary["frozen_pairs"].iloc[0]),
    "q4_pass": int(q4_physical_summary["pass"].iloc[0]),
    "q4_warning": int(q4_physical_summary["warning"].iloc[0]),
    "q4_blocker": int(q4_physical_summary["blocker"].iloc[0]),
    "q6_kept_columns": int(q6_patched["q6_keep"].sum()),
    "q6_excluded_columns": int((~q6_patched["q6_keep"]).sum()),
    "permitted_claim": "Physical-IoT-event/protocol Q4 and Q6 governance contract re-instantiated on a public trace with independent IoT and network streams.",
    "unsupported_claim": "No synthetic realism, anomaly-detection utility, privacy guarantee, or numerical transfer from the residential deployment is claimed.",
    "supersedes_note": "Use this physical-Q4 summary rather than the earlier label/type-driven Q4 smoke-test summary for manuscript-facing M2 evidence.",
}])
physical_summary_path = save_table(physical_claim_summary, PATCH_TABLE_DIR / "toniot_transfer_claim_scope_summary_physical_q4.csv")

index_rows = []
for p in sorted(PATCH_DIR.rglob("*")):
    if p.is_file():
        index_rows.append({
            "artifact_group": "physical_q4",
            "relative_path": str(p.relative_to(OUT_ROOT)),
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "claim_scope": "TON-IoT physical-Q4/Q6 governance-transfer patch",
        })

for p in [
    MANIFEST_DIR / "toniot_source_selection_policy.json",
    MANIFEST_DIR / "toniot_chronological_split_manifest.csv",
    MANIFEST_DIR / "toniot_file_discovery_manifest.csv",
    MANIFEST_DIR / "toniot_network_aggregation_manifest.csv",
    MANIFEST_DIR / "toniot_iot_aggregation_manifest.csv",
]:
    if p.exists():
        index_rows.append({
            "artifact_group": "prior_toniot_context",
            "relative_path": str(p.relative_to(OUT_ROOT)),
            "path": str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "claim_scope": "TON-IoT source/split context reused by physical-Q4 patch",
        })

patch_index = pd.DataFrame(index_rows)
patch_index_path = save_table(patch_index, OUT_ROOT / "TONIOT_STUDY_THESIS_PACKAGE_INDEX_PHYSICAL_Q4.csv")
patch_sha_path = save_table(patch_index[["relative_path", "sha256", "bytes"]], OUT_ROOT / "TONIOT_PACKAGE_SHA256_MANIFEST_PHYSICAL_Q4.csv")

if OVERWRITE_CANONICAL:

    save_table(q4_physical_summary, TABLE_DIR / "toniot_q4_scope_summary.csv")
    save_table(q4_test_audit, LEDGER_DIR / "toniot_q4_test_event_response_audit.csv")
    save_table(frozen_manifest, MANIFEST_DIR / "toniot_q4_manifest_trainval_frozen.csv")
    save_table(q6_patched, TABLE_DIR / "toniot_q6_release_scope_table.csv")
    save_table(q6_patched_summary, TABLE_DIR / "toniot_q6_release_summary_by_role.csv")
    save_table(q6_patched[~q6_patched["q6_keep"]], TABLE_DIR / "toniot_q6_excluded_columns.csv")
    save_table(physical_claim_summary, TABLE_DIR / "toniot_transfer_claim_scope_summary.csv")
    print("Canonical TON-IoT summary/Q4/Q6 files overwritten with physical-Q4 patched outputs.")
else:
    print("Canonical files were NOT overwritten. Use physical_q4/* outputs for review/manuscript until validated.")

print("Wrote physical-Q4 summary:", physical_summary_path)
print("Wrote patched package index:", patch_index_path)
print("Wrote patched SHA manifest:", patch_sha_path)
display(physical_claim_summary)
display(patch_index.head(20))

# Source cell 7


from pathlib import Path
import os
import pandas as pd

OUT_ROOT = Path(os.environ.get(
    "TONIOT_OUT_ROOT",
    "./runs/toniot"
)).expanduser().resolve()

PATCH_DIR = OUT_ROOT / "physical_q4"

audit_path = PATCH_DIR / "ledgers/toniot_physical_q4_test_event_response_audit.csv"
summary_path = PATCH_DIR / "tables/toniot_physical_q4_scope_summary.csv"
claim_path = PATCH_DIR / "tables/toniot_transfer_claim_scope_summary_physical_q4.csv"
q6_summary_path = PATCH_DIR / "tables/toniot_q6_release_summary_by_role_physical_q4.csv"

for p in [audit_path, summary_path, claim_path, q6_summary_path]:
    if not p.exists():
        raise FileNotFoundError(p)

audit = pd.read_csv(audit_path)
summary = pd.read_csv(summary_path)
claim = pd.read_csv(claim_path)
q6_summary = pd.read_csv(q6_summary_path)

driver_bad = (
    audit["driver_column"].astype(str).str.lower().str.contains(r"(^|_)label($|_)|(^|_)type($|_)|_type_", regex=True, na=False)
    | audit["physical_source_column"].astype(str).str.lower().str.contains(r"(^|_)label($|_)|(^|_)type($|_)|_type_", regex=True, na=False)
)

response_bad = audit["protocol_response_column"].astype(str).str.lower().str.contains(
    r"^net_label|^net_type",
    regex=True,
    na=False,
)

bad = audit[driver_bad | response_bad].copy()

print("Physical-Q4 summary:")
display(summary)

print("Transfer claim summary:")
display(claim)

print("Q6 summary by role:")
display(q6_summary)

if not bad.empty:
    print("Potential label/type leakage rows:")
    display(bad)
    raise RuntimeError(
        "Physical-Q4 patch still contains label/type-driven Q4 rows. "
        "Do not promote this patch until those rows are removed."
    )

warning_pairs = audit[audit["test_status"].astype(str).str.lower().eq("warning")].copy()
warning_pairs_path = PATCH_DIR / "tables/toniot_physical_q4_warning_pairs.csv"
warning_pairs.to_csv(warning_pairs_path, index=False)

print(f"Validation passed. Warning-level physical-Q4 pairs: {len(warning_pairs)}")
print("Wrote:", warning_pairs_path)
display(warning_pairs)