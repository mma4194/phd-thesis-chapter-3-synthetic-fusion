# ============================================================
# CELL 3 — Load + canonical time + fixed split + abs-time drop
#           + protected constant drop + explicit modeling registries — v7
#
# Purpose:
# - load the canonical FULLGRID Parquet dataset
# - materialize sec_epoch_s__canon from validated raw epoch-second source
# - quarantine raw time and provenance time from modeling
# - apply fixed chronological train/val/test split
# - remove absolute-time leakage while preserving events_in_sec* drivers
# - route all columns into explicit registries
# - drop constant/degenerate protocol columns using TRAIN only
# - expose explicit downstream globals and artifact manifests
#
# Alignment:
# - Cell 1 validates raw "sec" as strict 1 Hz Unix epoch seconds
# - Cell 2 provides canonical time, leakage, schema, and constant-drop helpers
# - Cell 3 performs dataset loading and registry construction only
#
# Scientific constraints:
# - no random split
# - no TEST-informed pruning
# - no silent time leakage
# - no silent missing driver/mask synthesis
# - no unrouted feature namespace
# - no object/category modeling columns
# ============================================================

log("--- START: Cell 3 (Load + canonical time + fixed split) ---")

import os
import time
import json
import hashlib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import List, Dict, Any, Sequence

# ------------------------------------------------------------
# 0) Resolve input path
# ------------------------------------------------------------
def _resolve_input_path(cfg: dict) -> str:
    for k in ["input_parquet", "input_path", "input", "data_path", "parquet_path"]:
        v = cfg.get(k, "")
        if v:
            return str(v)
    raise RuntimeError("[Cell3] Could not resolve input parquet path from CFG.")


in_path = _resolve_input_path(CFG)

if not os.path.exists(in_path):
    raise FileNotFoundError(f"[Cell3] Input parquet not found: {in_path}")

ARTDIR = os.path.join(str(CFG["outdir"]), "artifacts")
os.makedirs(ARTDIR, exist_ok=True)

# ------------------------------------------------------------
# 1) Require Cell 2 helpers
# ------------------------------------------------------------
_required_cell2_helpers = [
    "materialize_canonical_time_column",
    "validate_epoch_second_series",
    "collect_abs_time_features",
    "drop_constants_train_only",
    "quarantine_raw_time_column",
    "make_tod_features",
    "assert_no_duplicate_columns",
    "assert_no_unexpected_object_columns",
    "assert_no_modeling_forbidden_columns",
    "assert_no_abs_time_leakage_columns",
    "is_provenance_column",
]

for req in _required_cell2_helpers:
    if req not in globals():
        raise RuntimeError(f"Cell 2 must be executed before Cell 3 ({req} missing).")

# ------------------------------------------------------------
# 2) Load Parquet once
# ------------------------------------------------------------
t0 = time.time()
df0 = pd.read_parquet(in_path).reset_index(drop=True)
log(f"Loaded parquet: rows={len(df0):,} cols={df0.shape[1]} | elapsed={time.time() - t0:.2f}s")

if df0.empty:
    raise RuntimeError("[Cell3] Input parquet is empty.")

assert_no_duplicate_columns(df0, "Cell3/load")

if bool(CFG.get("forbid_object_dtype_features", True)):
    # At raw-load stage, allow only configured time candidates to be non-numeric
    # if they happen to be datetime/object parseable. Modeling columns must not
    # enter as object/category.
    allow_obj = list(CFG.get("time_col_candidates", []))
    assert_no_unexpected_object_columns(df0, allow_cols=allow_obj, where="Cell3/load")

expected_rows = int(CFG.get("expected_total_rows", len(df0)))
if len(df0) != expected_rows:
    raise RuntimeError(
        f"[Cell3] Row count mismatch for canonical dataset: "
        f"got={len(df0):,} expected={expected_rows:,}"
    )

expected_cols = int(CFG.get("expected_total_columns", CFG.get("expected_total_columns_hint", df0.shape[1])))
if int(df0.shape[1]) != expected_cols:
    msg = f"[Cell3] Column count mismatch: got={df0.shape[1]} expected={expected_cols}"
    if bool(CFG.get("strict_schema_checks", False)):
        raise RuntimeError(msg)
    log("[WARN] " + msg)

# ------------------------------------------------------------
# 3) Canonical time materialization + raw-time quarantine
# ------------------------------------------------------------
RAW_TIME_COL = str(CFG.get("raw_time_col", "sec"))
CANON_TIME_COL = str(CFG.get("canonical_time_col", "sec_epoch_s__canon"))

if RAW_TIME_COL not in df0.columns and CANON_TIME_COL not in df0.columns:
    raise RuntimeError(
        f"[Cell3] Neither raw nor canonical time column exists. "
        f"raw={RAW_TIME_COL} | canonical={CANON_TIME_COL}"
    )

df0, time_contract = materialize_canonical_time_column(
    df0,
    raw_time_col=RAW_TIME_COL,
    canonical_time_col=CANON_TIME_COL,
    expected_hz=int(CFG.get("expected_hz", 1)),
    require_monotonic=bool(CFG.get("require_monotonic_time", True)),
    require_unique=bool(CFG.get("require_unique_time", True)),
    overwrite=False,
    where="Cell3/materialize_canonical_time",
)

raw_time_col0 = RAW_TIME_COL if RAW_TIME_COL in df0.columns else CANON_TIME_COL

provenance_time_col = None
if RAW_TIME_COL in df0.columns and RAW_TIME_COL != CANON_TIME_COL:
    df0, provenance_time_col = quarantine_raw_time_column(
        df=df0,
        raw_time_col=RAW_TIME_COL,
        canonical_time_col=CANON_TIME_COL,
        keep_provenance=True,
    )

assert_no_duplicate_columns(df0, "Cell3/post_time_quarantine")

sec_all = df0[CANON_TIME_COL].to_numpy(dtype=np.int64, copy=False)

log(
    f"time_col(raw)={raw_time_col0} | canon={CANON_TIME_COL} | "
    f"range=[{int(sec_all.min())}, {int(sec_all.max())}] | "
    f"step={time_contract.get('median_step')}"
)

# ------------------------------------------------------------
# 4) Canonical fixed chronological split
# ------------------------------------------------------------
N_TRAIN = int(CFG.get("n_train", 766616))
N_VAL = int(CFG.get("n_val", 255538))
N_TEST = int(CFG.get("n_test", 255540))

if len(df0) != (N_TRAIN + N_VAL + N_TEST):
    raise RuntimeError(
        f"[Cell3] Fixed split sizes do not match loaded dataset: "
        f"rows={len(df0):,} vs split_sum={N_TRAIN + N_VAL + N_TEST:,}"
    )

df_tr = df0.iloc[:N_TRAIN].reset_index(drop=True).copy()
df_va = df0.iloc[N_TRAIN:N_TRAIN + N_VAL].reset_index(drop=True).copy()
df_te = df0.iloc[N_TRAIN + N_VAL:].reset_index(drop=True).copy()

SPLIT_N_TRAIN = int(len(df_tr))
SPLIT_N_VAL = int(len(df_va))
SPLIT_N_TEST = int(len(df_te))

split_time_contracts: Dict[str, Dict[str, Any]] = {}

for name, part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    if len(part) == 0:
        raise RuntimeError(f"[Cell3] {name} split is empty.")

    split_time_contracts[name] = validate_epoch_second_series(
        part[CANON_TIME_COL],
        where=f"Cell3/{name}_time",
        expected_hz=int(CFG.get("expected_hz", 1)),
        require_monotonic=bool(CFG.get("require_monotonic_time", True)),
        require_unique=bool(CFG.get("require_unique_time", True)),
    )

    if int(split_time_contracts[name]["duration_seconds_inclusive"]) != len(part):
        raise RuntimeError(
            f"[Cell3] {name} split is not a gap-free 1 Hz interval: "
            f"rows={len(part):,} duration={split_time_contracts[name]['duration_seconds_inclusive']:,}"
        )

    log(f"[fullgrid] {name}: rows == unique_seconds == duration_seconds == {len(part):,}")

SEC_TRAIN_END = int(df_tr[CANON_TIME_COL].iloc[-1])
SEC_VAL_END = int(df_va[CANON_TIME_COL].iloc[-1])

if not (
    int(df_tr[CANON_TIME_COL].max()) < int(df_va[CANON_TIME_COL].min())
    <= int(df_va[CANON_TIME_COL].max()) < int(df_te[CANON_TIME_COL].min())
):
    raise RuntimeError("[Cell3] Split temporal boundaries overlap or are not strictly ordered.")

log(
    f"split fixed train={SPLIT_N_TRAIN:,} val={SPLIT_N_VAL:,} test={SPLIT_N_TEST:,} | "
    f"sec_train_end={SEC_TRAIN_END} | sec_val_end={SEC_VAL_END}"
)

# ------------------------------------------------------------
# 5) Write split contract artifact
# ------------------------------------------------------------
def _sha_ints(xs: Sequence[int]) -> str:
    h = hashlib.sha256()
    for v in xs:
        h.update(str(int(v)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


split_contract = {
    "version": "split_contract_v7_fixed_counts_gapfree_time",
    "input_parquet": str(in_path),
    "canon_time_col": CANON_TIME_COL,
    "raw_time_col_configured": RAW_TIME_COL,
    "raw_time_col_detected": raw_time_col0,
    "provenance_time_col": provenance_time_col,
    "N_total": int(len(df0)),
    "N_train": SPLIT_N_TRAIN,
    "N_val": SPLIT_N_VAL,
    "N_test": SPLIT_N_TEST,
    "sec_train_end": SEC_TRAIN_END,
    "sec_val_end": SEC_VAL_END,
    "sec_range": [int(sec_all.min()), int(sec_all.max())],
    "sec_hash_ends": _sha_ints([int(sec_all.min()), SEC_TRAIN_END, SEC_VAL_END, int(sec_all.max())]),
    "time_contract_full": time_contract,
    "time_contract_splits": split_time_contracts,
    "cfg_sha1": RUN_META["cfg_sha1"] if "RUN_META" in globals() else None,
}

split_contract_path = os.path.join(ARTDIR, "split_contract.json")
with open(split_contract_path, "w", encoding="utf-8") as f:
    json.dump(split_contract, f, indent=2, default=str)

log(f"[Cell3] Wrote split contract: {split_contract_path}")

# ------------------------------------------------------------
# 6) Time-of-day features from canonical time
# ------------------------------------------------------------
tod_tr_sin, tod_tr_cos = make_tod_features(df_tr[CANON_TIME_COL])
tod_va_sin, tod_va_cos = make_tod_features(df_va[CANON_TIME_COL])
tod_te_sin, tod_te_cos = make_tod_features(df_te[CANON_TIME_COL])

tod_tr = np.stack([tod_tr_sin, tod_tr_cos], axis=1).astype(np.float32, copy=False)
tod_va = np.stack([tod_va_sin, tod_va_cos], axis=1).astype(np.float32, copy=False)
tod_te = np.stack([tod_te_sin, tod_te_cos], axis=1).astype(np.float32, copy=False)

for nm, arr, expected_n in [
    ("TRAIN", tod_tr, SPLIT_N_TRAIN),
    ("VAL", tod_va, SPLIT_N_VAL),
    ("TEST", tod_te, SPLIT_N_TEST),
]:
    if arr.shape != (expected_n, 2):
        raise RuntimeError(f"[Cell3] {nm} TOD shape invalid: got={arr.shape} expected={(expected_n, 2)}")
    if arr.dtype != np.float32:
        raise RuntimeError(f"[Cell3] {nm} TOD dtype invalid: {arr.dtype}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"[Cell3] {nm} TOD contains non-finite values.")
    radius = np.sqrt(np.sum(arr.astype(np.float64) ** 2, axis=1))
    if not np.allclose(radius, 1.0, atol=1e-5):
        raise RuntimeError(f"[Cell3] {nm} TOD sin/cos radius check failed.")

log(f"TOD shapes: tr={tod_tr.shape} va={tod_va.shape} te={tod_te.shape}")

# ------------------------------------------------------------
# 7) Remove absolute-time leakage columns from split dataframes
# ------------------------------------------------------------
log("--- START: Remove absolute-time features ---")

abs_time_cols = collect_abs_time_features(
    df_tr.columns.tolist(),
    raw_time_col=RAW_TIME_COL,
    canonical_time_col=CANON_TIME_COL,
    include_canonical=True,
)

# Keep canonical time only as bookkeeping. It must not enter modeling registries.
abs_time_cols = [c for c in abs_time_cols if c != CANON_TIME_COL]
abs_time_cols = sorted(set(abs_time_cols))

if any(c.startswith("events_in_sec__") for c in abs_time_cols):
    raise RuntimeError("[Cell3] Bug: events_in_sec__* driver columns were incorrectly classified as abs-time.")

def _drop_if_present(df_part: pd.DataFrame, cols: Sequence[str], *, where: str) -> pd.DataFrame:
    cols2 = [c for c in cols if c in df_part.columns]
    if not cols2:
        out = df_part.copy()
    else:
        out = df_part.drop(columns=cols2).copy()
    assert_no_duplicate_columns(out, where)
    return out


if abs_time_cols:
    df_tr = _drop_if_present(df_tr, abs_time_cols, where="Cell3/drop_abs_time/TRAIN")
    df_va = _drop_if_present(df_va, abs_time_cols, where="Cell3/drop_abs_time/VAL")
    df_te = _drop_if_present(df_te, abs_time_cols, where="Cell3/drop_abs_time/TEST")
    log(f"removed abs-time cols={len(abs_time_cols)} | first up to 30: {abs_time_cols[:30]}")
else:
    log("removed abs-time cols=0")

if provenance_time_col is not None:
    for nm, part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
        if provenance_time_col in part.columns:
            raise RuntimeError(f"[Cell3] Provenance time column survived abs-time removal in {nm}: {provenance_time_col}")

log("--- END:   Remove absolute-time features ---")

# ------------------------------------------------------------
# 8) Explicit namespace routing registries
# ------------------------------------------------------------
PROTOCOL_PREFIXES = (
    "router__",
    "ota__",
    "ota24__",
    "ota5__",
    "zigbee__",
    "zwave__",
)

AUX_META_EXACT = {
    "zb__app_event_flag",
    "zb__app_event_intensity",
    "net__obs_any",
    "net__obs_all",
}

def _is_protocol_namespace(col: str) -> bool:
    return isinstance(col, str) and col.lower().startswith(PROTOCOL_PREFIXES)


def _is_iot_namespace(col: str) -> bool:
    return isinstance(col, str) and col.lower().startswith("iot__")


def _is_events_driver_namespace(col: str) -> bool:
    return isinstance(col, str) and col.lower().startswith("events_in_sec__")


def _is_telemetry_driver_namespace(col: str) -> bool:
    return isinstance(col, str) and col.lower().startswith("telemetry_in_sec__")


def _is_aux_meta_namespace(col: str) -> bool:
    return isinstance(col, str) and col in AUX_META_EXACT


def _is_bookkeeping_only(col: str) -> bool:
    if col == CANON_TIME_COL:
        return True
    if isinstance(col, str) and is_provenance_column(col):
        return True
    return False


all_cols_tr = df_tr.columns.tolist()

BOOKKEEPING_COLS = [c for c in all_cols_tr if _is_bookkeeping_only(c)]
PROTO_NAMESPACE_COLS = [c for c in all_cols_tr if _is_protocol_namespace(c)]
IOT_NAMESPACE_COLS = [c for c in all_cols_tr if _is_iot_namespace(c)]
EVENT_DRIVER_COLS = [c for c in all_cols_tr if _is_events_driver_namespace(c)]
TELEMETRY_DRIVER_COLS = [c for c in all_cols_tr if _is_telemetry_driver_namespace(c)]
AUX_META_COLS = [c for c in all_cols_tr if _is_aux_meta_namespace(c)]

KNOWN_NONMODEL = set(BOOKKEEPING_COLS)
KNOWN_MODEL = (
    set(PROTO_NAMESPACE_COLS)
    | set(IOT_NAMESPACE_COLS)
    | set(EVENT_DRIVER_COLS)
    | set(TELEMETRY_DRIVER_COLS)
    | set(AUX_META_COLS)
)

UNROUTED_COLS = [c for c in all_cols_tr if c not in KNOWN_NONMODEL and c not in KNOWN_MODEL]

if UNROUTED_COLS:
    raise RuntimeError(
        f"[Cell3] Unrouted columns detected ({len(UNROUTED_COLS)}): {UNROUTED_COLS[:30]}"
    )

if CANON_TIME_COL not in BOOKKEEPING_COLS:
    raise RuntimeError(f"[Cell3] Canonical time column must be bookkeeping-only: {CANON_TIME_COL}")

log(
    f"[Cell3] registries | protocol={len(PROTO_NAMESPACE_COLS)} | "
    f"iot={len(IOT_NAMESPACE_COLS)} | event_drivers={len(EVENT_DRIVER_COLS)} | "
    f"telemetry_drivers={len(TELEMETRY_DRIVER_COLS)} | aux_meta={len(AUX_META_COLS)} | "
    f"bookkeeping={len(BOOKKEEPING_COLS)}"
)

# ------------------------------------------------------------
# 9) Protected constant drop on protocol namespace only
# ------------------------------------------------------------
log("--- START: Drop constants on TRAIN (protocol namespace only) ---")

for cols_name, cols_list in [
    ("PROTO_NAMESPACE_COLS", PROTO_NAMESPACE_COLS),
    ("IOT_NAMESPACE_COLS", IOT_NAMESPACE_COLS),
    ("EVENT_DRIVER_COLS", EVENT_DRIVER_COLS),
    ("TELEMETRY_DRIVER_COLS", TELEMETRY_DRIVER_COLS),
    ("AUX_META_COLS", AUX_META_COLS),
    ("BOOKKEEPING_COLS", BOOKKEEPING_COLS),
]:
    miss_va = [c for c in cols_list if c not in df_va.columns]
    miss_te = [c for c in cols_list if c not in df_te.columns]
    if miss_va or miss_te:
        raise RuntimeError(
            f"[Cell3] Schema drift in registry {cols_name}.\n"
            f"missing_va={miss_va[:20]}\n"
            f"missing_te={miss_te[:20]}"
        )

if not PROTO_NAMESPACE_COLS:
    raise RuntimeError("[Cell3] No protocol namespace columns found before constant drop.")

_tr_p, _va_p, _te_p, dropped_const_proto = drop_constants_train_only(
    df_tr.loc[:, PROTO_NAMESPACE_COLS].copy(),
    df_va.loc[:, PROTO_NAMESPACE_COLS].copy(),
    df_te.loc[:, PROTO_NAMESPACE_COLS].copy(),
    fail_on_object_cols=True,
    fail_on_forbidden_cols=True,
)

PROTO_NAMESPACE_COLS_KEPT = _tr_p.columns.tolist()

def _ordered_concat(parts: Sequence[pd.DataFrame], *, where: str) -> pd.DataFrame:
    lengths = {len(p) for p in parts}
    if len(lengths) != 1:
        raise RuntimeError(f"[{where}] concat length mismatch across parts: {sorted(lengths)}")

    used = set()
    ordered = []

    for p in parts:
        overlap = used.intersection(set(p.columns))
        if overlap:
            raise RuntimeError(f"[{where}] concat column overlap detected: {sorted(list(overlap))[:20]}")
        used.update(p.columns)
        ordered.append(p.reset_index(drop=True))

    out = pd.concat(ordered, axis=1)
    assert_no_duplicate_columns(out, where)
    return out


df_tr = _ordered_concat([
    _tr_p,
    df_tr.loc[:, IOT_NAMESPACE_COLS].copy(),
    df_tr.loc[:, EVENT_DRIVER_COLS].copy(),
    df_tr.loc[:, TELEMETRY_DRIVER_COLS].copy(),
    df_tr.loc[:, AUX_META_COLS].copy(),
    df_tr.loc[:, BOOKKEEPING_COLS].copy(),
], where="Cell3/reassemble/TRAIN")

df_va = _ordered_concat([
    _va_p,
    df_va.loc[:, IOT_NAMESPACE_COLS].copy(),
    df_va.loc[:, EVENT_DRIVER_COLS].copy(),
    df_va.loc[:, TELEMETRY_DRIVER_COLS].copy(),
    df_va.loc[:, AUX_META_COLS].copy(),
    df_va.loc[:, BOOKKEEPING_COLS].copy(),
], where="Cell3/reassemble/VAL")

df_te = _ordered_concat([
    _te_p,
    df_te.loc[:, IOT_NAMESPACE_COLS].copy(),
    df_te.loc[:, EVENT_DRIVER_COLS].copy(),
    df_te.loc[:, TELEMETRY_DRIVER_COLS].copy(),
    df_te.loc[:, AUX_META_COLS].copy(),
    df_te.loc[:, BOOKKEEPING_COLS].copy(),
], where="Cell3/reassemble/TEST")

log(f"Dropped constants/degenerate protocol cols (TRAIN-only) = {len(dropped_const_proto)}")
if dropped_const_proto:
    log(f"[Cell3] dropped_const_proto (first up to 30): {dropped_const_proto[:30]}")

dropped_path = os.path.join(ARTDIR, "dropped_constants_train_only_protocol.json")
with open(dropped_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "version": "dropped_constants_train_only_protocol_v7",
            "dropped": dropped_const_proto,
            "n_dropped": len(dropped_const_proto),
        },
        f,
        indent=2,
    )

log(f"[Cell3] Wrote dropped constants list: {dropped_path}")
log("--- END:   Drop constants (protocol namespace only) ---")

# ------------------------------------------------------------
# 10) Numeric registries for downstream cells
# ------------------------------------------------------------
def _numeric_cols(df_: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    out = []
    for c in cols:
        if c in df_.columns and is_numeric_dtype(df_[c]):
            out.append(c)
    return out


NUMERIC_COLS_PROTO = _numeric_cols(df_tr, PROTO_NAMESPACE_COLS_KEPT)
NUMERIC_COLS_IOT = _numeric_cols(df_tr, IOT_NAMESPACE_COLS)
NUMERIC_COLS_EVENT_DRIVERS = _numeric_cols(df_tr, EVENT_DRIVER_COLS)
NUMERIC_COLS_TELEMETRY_DRIVERS = _numeric_cols(df_tr, TELEMETRY_DRIVER_COLS)
NUMERIC_COLS_AUX_META = _numeric_cols(df_tr, AUX_META_COLS)

ALL_NUMERIC_MODEL_COLS = list(dict.fromkeys(
    NUMERIC_COLS_PROTO
    + NUMERIC_COLS_IOT
    + NUMERIC_COLS_EVENT_DRIVERS
    + NUMERIC_COLS_TELEMETRY_DRIVERS
    + NUMERIC_COLS_AUX_META
))

if not NUMERIC_COLS_PROTO:
    raise RuntimeError("[Cell3] No protocol numeric columns remain after preprocessing.")

if len(ALL_NUMERIC_MODEL_COLS) != len(set(ALL_NUMERIC_MODEL_COLS)):
    raise RuntimeError("[Cell3] Duplicate names inside ALL_NUMERIC_MODEL_COLS.")

for bad in [RAW_TIME_COL, CANON_TIME_COL]:
    if bad in ALL_NUMERIC_MODEL_COLS:
        raise RuntimeError(f"[Cell3] Time column leaked into ALL_NUMERIC_MODEL_COLS: {bad}")

if any(is_provenance_column(c) for c in ALL_NUMERIC_MODEL_COLS):
    bad = [c for c in ALL_NUMERIC_MODEL_COLS if is_provenance_column(c)]
    raise RuntimeError(f"[Cell3] Provenance columns leaked into ALL_NUMERIC_MODEL_COLS: {bad[:20]}")

for reg_name, reg_cols in [
    ("NUMERIC_COLS_PROTO", NUMERIC_COLS_PROTO),
    ("NUMERIC_COLS_IOT", NUMERIC_COLS_IOT),
    ("NUMERIC_COLS_EVENT_DRIVERS", NUMERIC_COLS_EVENT_DRIVERS),
    ("NUMERIC_COLS_TELEMETRY_DRIVERS", NUMERIC_COLS_TELEMETRY_DRIVERS),
    ("NUMERIC_COLS_AUX_META", NUMERIC_COLS_AUX_META),
    ("ALL_NUMERIC_MODEL_COLS", ALL_NUMERIC_MODEL_COLS),
]:
    miss_va = [c for c in reg_cols if c not in df_va.columns]
    miss_te = [c for c in reg_cols if c not in df_te.columns]
    if miss_va or miss_te:
        raise RuntimeError(
            f"[Cell3] Schema drift after preprocessing in {reg_name}.\n"
            f"missing_va={miss_va[:20]}\n"
            f"missing_te={miss_te[:20]}"
        )

# Validate model registries by constructing temporary frames from model cols.
for nm, part in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    model_view = part.loc[:, ALL_NUMERIC_MODEL_COLS]
    assert_no_duplicate_columns(model_view, f"Cell3/model_view/{nm}")
    assert_no_unexpected_object_columns(model_view, where=f"Cell3/model_view/{nm}")
    assert_no_modeling_forbidden_columns(model_view, where=f"Cell3/model_view/{nm}")
    assert_no_abs_time_leakage_columns(
        model_view,
        where=f"Cell3/model_view/{nm}",
        raw_time_col=RAW_TIME_COL,
        canonical_time_col=CANON_TIME_COL,
        include_canonical=True,
    )

log(
    f"[Cell3] df_tr cols={df_tr.shape[1]} | "
    f"numeric_proto={len(NUMERIC_COLS_PROTO)} | "
    f"numeric_iot={len(NUMERIC_COLS_IOT)} | "
    f"numeric_event_drivers={len(NUMERIC_COLS_EVENT_DRIVERS)} | "
    f"numeric_telemetry_drivers={len(NUMERIC_COLS_TELEMETRY_DRIVERS)} | "
    f"numeric_aux_meta={len(NUMERIC_COLS_AUX_META)} | "
    f"numeric_all_model={len(ALL_NUMERIC_MODEL_COLS)}"
)

# ------------------------------------------------------------
# 10.5) Routing manifest artifact
# ------------------------------------------------------------
routing_manifest = {
    "version": "routing_manifest_v7",
    "canon_time_col": CANON_TIME_COL,
    "raw_time_col_configured": RAW_TIME_COL,
    "raw_time_col_detected": raw_time_col0,
    "provenance_time_col": provenance_time_col,
    "protocol_prefixes": list(PROTOCOL_PREFIXES),
    "aux_meta_exact": sorted(list(AUX_META_EXACT)),
    "n_protocol_namespace_cols_pre_drop": int(len(PROTO_NAMESPACE_COLS)),
    "n_protocol_namespace_cols_kept": int(len(PROTO_NAMESPACE_COLS_KEPT)),
    "n_iot_namespace_cols": int(len(IOT_NAMESPACE_COLS)),
    "n_event_driver_cols": int(len(EVENT_DRIVER_COLS)),
    "n_telemetry_driver_cols": int(len(TELEMETRY_DRIVER_COLS)),
    "n_aux_meta_cols": int(len(AUX_META_COLS)),
    "n_bookkeeping_cols": int(len(BOOKKEEPING_COLS)),
    "n_all_numeric_model_cols": int(len(ALL_NUMERIC_MODEL_COLS)),
    "dropped_const_proto": list(dropped_const_proto),
    "abs_time_cols_removed": list(abs_time_cols),
    "unrouted_cols": list(UNROUTED_COLS),
}

routing_manifest_path = os.path.join(ARTDIR, "routing_manifest.json")
with open(routing_manifest_path, "w", encoding="utf-8") as f:
    json.dump(routing_manifest, f, indent=2)

log(f"[Cell3] Wrote routing manifest: {routing_manifest_path}")

# ------------------------------------------------------------
# 11) Broad protocol matrices only
# ------------------------------------------------------------
# Important:
# - Protocol value columns may contain NaN when the modality was not observed.
# - Cell 3 must preserve these NaNs because they encode observation semantics.
# - Do NOT impute here.
# - Later modeling cells must perform mask-aware imputation/scaling.
# - Inf/-Inf remains a hard error.

Xtr = df_tr.loc[:, NUMERIC_COLS_PROTO].to_numpy(dtype=np.float32, copy=True)
Xva = df_va.loc[:, NUMERIC_COLS_PROTO].to_numpy(dtype=np.float32, copy=True)
Xte = df_te.loc[:, NUMERIC_COLS_PROTO].to_numpy(dtype=np.float32, copy=True)

PROTOCOL_NAN_AUDIT = {}

for nm, arr, expected_n in [
    ("Xtr", Xtr, SPLIT_N_TRAIN),
    ("Xva", Xva, SPLIT_N_VAL),
    ("Xte", Xte, SPLIT_N_TEST),
]:
    if arr.shape != (expected_n, len(NUMERIC_COLS_PROTO)):
        raise RuntimeError(
            f"[Cell3] {nm} shape mismatch: got={arr.shape} "
            f"expected={(expected_n, len(NUMERIC_COLS_PROTO))}"
        )

    if arr.dtype != np.float32:
        raise RuntimeError(f"[Cell3] {nm} dtype mismatch: got={arr.dtype}")

    inf_mask = np.isinf(arr)
    if inf_mask.any():
        bad = int(inf_mask.sum())
        raise RuntimeError(f"[Cell3] {nm} contains +/-inf protocol values: {bad}")

    nan_mask = np.isnan(arr)
    n_nan = int(nan_mask.sum())
    n_total = int(arr.size)
    nan_rate = float(n_nan / max(n_total, 1))

    col_nan_counts = nan_mask.sum(axis=0).astype(np.int64)
    cols_with_nan = [
        {
            "col": str(c),
            "nan_count": int(col_nan_counts[j]),
            "nan_rate": float(col_nan_counts[j] / max(arr.shape[0], 1)),
        }
        for j, c in enumerate(NUMERIC_COLS_PROTO)
        if int(col_nan_counts[j]) > 0
    ]

    all_nan_cols = [
        str(NUMERIC_COLS_PROTO[j])
        for j, cnt in enumerate(col_nan_counts)
        if int(cnt) == arr.shape[0]
    ]

    PROTOCOL_NAN_AUDIT[nm] = {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "nan_count": n_nan,
        "total_values": n_total,
        "nan_rate": nan_rate,
        "n_cols_with_nan": int(len(cols_with_nan)),
        "n_all_nan_cols": int(len(all_nan_cols)),
        "cols_with_nan_first_50": cols_with_nan[:50],
        "all_nan_cols": all_nan_cols,
    }

    log(
        f"[Cell3] {nm} protocol matrix: shape={arr.shape} | "
        f"nan_count={n_nan:,} | nan_rate={nan_rate:.6f} | "
        f"cols_with_nan={len(cols_with_nan)} | all_nan_cols={len(all_nan_cols)}"
    )

    # A protocol column that is all-NaN after TRAIN-only constant dropping is
    # suspicious because it cannot support downstream model fitting.
    # Do not drop it here because Cell 3 is only routing/auditing, but fail
    # closed if it exists in TRAIN.
    if nm == "Xtr" and all_nan_cols:
        raise RuntimeError(
            "[Cell3] TRAIN contains protocol columns that are entirely NaN after constant-drop. "
            f"These cannot be fitted downstream: {all_nan_cols[:30]}"
        )

protocol_nan_audit_path = os.path.join(ARTDIR, "protocol_nan_audit_cell3.json")
with open(protocol_nan_audit_path, "w", encoding="utf-8") as f:
    json.dump(PROTOCOL_NAN_AUDIT, f, indent=2)

log(f"[Cell3] Wrote protocol NaN audit: {protocol_nan_audit_path}")

time_col = CANON_TIME_COL
log(f"Canonical time column set: time_col={time_col}")

# ------------------------------------------------------------
# 12) Expose globals
# ------------------------------------------------------------
globals()["df_tr"] = df_tr
globals()["df_va"] = df_va
globals()["df_te"] = df_te

globals()["time_col"] = time_col
globals()["CANON_TIME_COL"] = CANON_TIME_COL
globals()["RAW_TIME_COL"] = RAW_TIME_COL
globals()["RAW_TIME_COL_DETECTED"] = raw_time_col0
globals()["PROVENANCE_TIME_COL"] = provenance_time_col

globals()["tod_tr"] = tod_tr
globals()["tod_va"] = tod_va
globals()["tod_te"] = tod_te

globals()["Xtr"] = Xtr
globals()["Xva"] = Xva
globals()["Xte"] = Xte

globals()["BOOKKEEPING_COLS"] = list(BOOKKEEPING_COLS)
globals()["PROTO_NAMESPACE_COLS"] = list(PROTO_NAMESPACE_COLS_KEPT)
globals()["PROTO_NAMESPACE_COLS_PRE_DROP"] = list(PROTO_NAMESPACE_COLS)
globals()["IOT_NAMESPACE_COLS"] = list(IOT_NAMESPACE_COLS)
globals()["EVENT_DRIVER_COLS"] = list(EVENT_DRIVER_COLS)
globals()["TELEMETRY_DRIVER_COLS"] = list(TELEMETRY_DRIVER_COLS)
globals()["AUX_META_COLS"] = list(AUX_META_COLS)

globals()["ALL_NUMERIC_MODEL_COLS"] = list(ALL_NUMERIC_MODEL_COLS)
globals()["NUMERIC_COLS_PROTO"] = list(NUMERIC_COLS_PROTO)
globals()["NUMERIC_COLS_IOT"] = list(NUMERIC_COLS_IOT)
globals()["NUMERIC_COLS_EVENT_DRIVERS"] = list(NUMERIC_COLS_EVENT_DRIVERS)
globals()["NUMERIC_COLS_TELEMETRY_DRIVERS"] = list(NUMERIC_COLS_TELEMETRY_DRIVERS)
globals()["NUMERIC_COLS_AUX_META"] = list(NUMERIC_COLS_AUX_META)

globals()["SPLIT_N_TRAIN"] = SPLIT_N_TRAIN
globals()["SPLIT_N_VAL"] = SPLIT_N_VAL
globals()["SPLIT_N_TEST"] = SPLIT_N_TEST
globals()["SEC_TRAIN_END"] = SEC_TRAIN_END
globals()["SEC_VAL_END"] = SEC_VAL_END

globals()["PROTOCOL_NAN_AUDIT"] = PROTOCOL_NAN_AUDIT

globals()["SPLIT_CONTRACT"] = split_contract
globals()["ROUTING_MANIFEST"] = routing_manifest
globals()["TIME_CONTRACT_CELL3"] = time_contract
globals()["SPLIT_TIME_CONTRACTS"] = split_time_contracts

log(
    f"[Cell3] Done. df_tr={df_tr.shape} df_va={df_va.shape} df_te={df_te.shape} | "
    f"Xtr={Xtr.shape} Xva={Xva.shape} Xte={Xte.shape}"
)
log("--- END: Cell 3 ---")