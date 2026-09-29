# ==========================================================
# CELL 12.e.6 - Driver TEST QA only
# v1.1 STUDY-THESIS strict sparse binary driver final TEST QA, contract-hardened
#
# Role:
#   - Evaluate final synthetic sparse binary driver TEST values against real TEST.
#   - Use TEST real driver values ONLY for QA.
#   - Attribute failures to:
#       1) carried upstream 12.e.3 warning/blocker
#       2) TRAIN/VAL→TEST split drift
#       3) new 12.e.6 TEST QA blocker
#   - Export driver QA artifacts for Cell 13 coupling QA.
#
# Strict rules:
#   - Do NOT mutate synthetic values.
#   - Do NOT select generators.
#   - Do NOT fit generators.
#   - TEST real values are used only for final QA.
#
# Outputs:
#   reports/cell12e6_driver_final_test_qa_metrics.csv
#   reports/cell12e6_driver_publication_status.csv
#   reports/cell12e6_driver_blocker_origin_audit.csv
#   reports/cell12e6_driver_publication_blocked_test_qa_metrics.csv
#   reports/cell12e6_driver_cell13_handoff_manifest.json
#   reports/cell12e6_driver_contract.json
#   artifacts/cell12e6_driver_final_test_qa_manifest.json
# ==========================================================

log("--- START: Cell 12.e.6 - Driver TEST QA only (v1.1 strict sparse binary, contract-hardened) ---")

import os
import re
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_12e6 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_DRIVER_COLS",
    "IOT_DRIVER_TARGET_CONTRACT_DF",
    "IOT_FINAL_DRIVER_TEST",
    "CELL12E3_LOCKED_DRIVER_SELECTION_DF",
    "CELL12E5_DRIVER_FINAL_ENFORCEMENT_AUDIT_DF",
    "CELL12E5_DRIVER_PUBLICATION_COLUMN_REGISTRY_DF",
    "CELL12E5_DRIVER_FINAL_ENFORCEMENT_CONTRACT",
]
_missing_12e6 = [k for k in _required_12e6 if k not in globals()]
if _missing_12e6:
    raise RuntimeError(f"[Cell12.e.6] Missing required globals from prior cells: {_missing_12e6}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

IOT_DRIVER_COLS = list(map(str, IOT_DRIVER_COLS))
EXPECTED_DRIVER_TARGET_COUNT_12E = int(CFG.get("cell12e_expected_driver_target_count", 26))

if len(IOT_DRIVER_COLS) != EXPECTED_DRIVER_TARGET_COUNT_12E:
    raise RuntimeError(
        "[Cell12.e.6] Driver target count mismatch: "
        f"got={len(IOT_DRIVER_COLS)} expected={EXPECTED_DRIVER_TARGET_COUNT_12E}"
    )

CELL12E6_VERSION = "cell12e6_sparse_binary_driver_final_test_qa_only_strict_v1_1_contract_hardened"

CFG["cell12e6_version"] = CELL12E6_VERSION
CFG["cell12e6_TEST_real_values_used_for_QA_only"] = True
CFG["cell12e6_synthetic_values_mutated"] = False
CFG["cell12e6_selection_done_here"] = False
CFG["cell12e6_generator_fit_done_here"] = False
CFG["cell12e6_test_qa_done_here"] = True
CFG["cell12e6_feeds_cell13_coupling_qa"] = True

# QA thresholds.
CFG.setdefault("cell12e6_event_rate_error_warning", 0.001)
CFG.setdefault("cell12e6_event_rate_error_fatal", 0.005)
CFG.setdefault("cell12e6_burst_count_error_warning", 0.001)
CFG.setdefault("cell12e6_burst_count_error_fatal", 0.005)
CFG.setdefault("cell12e6_interarrival_ks_warning", 0.50)
CFG.setdefault("cell12e6_interarrival_ks_fatal", 0.80)
CFG.setdefault("cell12e6_duration_ks_warning", 0.20)
CFG.setdefault("cell12e6_duration_ks_fatal", 0.50)
CFG.setdefault("cell12e6_overdispersion_error_warning", 0.10)
CFG.setdefault("cell12e6_overdispersion_error_fatal", 0.50)
CFG.setdefault("cell12e6_train_test_event_rate_shift_warning", 0.002)
CFG.setdefault("cell12e6_train_test_event_rate_shift_fatal", 0.010)

# Coupling handoff config.
CFG.setdefault("cell12e6_cell13_lag_window_seconds", 10)
CFG.setdefault("cell12e6_driver_event_context_radius", 10)

# ----------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------
def _json_sanitize_12e6(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12e6(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12e6(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12e6(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12e6(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12e6(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12e6(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12e6(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj

def _write_json_12e6(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12e6(payload), f, indent=2, sort_keys=True)

def _sha256_file_12e6(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_12e6(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_bool_12e6(x, default=False) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return bool(default)
    try:
        if pd.isna(x):
            return bool(default)
    except Exception:
        pass
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", "", "nan", "none", "null"}:
        return False
    return bool(default)

def _to_num_array_12e6(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _coerce_binary_nan_12e6(x: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    arr[finite & (np.abs(arr - 0.0) <= tol)] = 0.0
    arr[finite & (np.abs(arr - 1.0) <= tol)] = 1.0
    bad = finite & ~((arr == 0.0) | (arr == 1.0))
    arr[bad] = np.nan
    return arr

def _driver_observed_values_12e6(frame: pd.DataFrame, col: str) -> tuple:
    raw = _to_num_array_12e6(frame, col)
    arr = _coerce_binary_nan_12e6(
        raw,
        tol=float(CFG.get("cell12d0_binary_value_tolerance", 1e-6)),
    )
    obs = np.isfinite(arr)
    return arr, obs

def _entity_from_driver_col_12e6(col: str) -> str:
    """Parse entity names robustly for events_in_sec driver namespaces."""
    parts = [p for p in str(col).split("__") if p != ""]
    if not parts:
        return str(col)

    if parts[0] == "events_in_sec":
        if len(parts) >= 3 and parts[1] in {"entity", "device", "iot"}:
            return parts[2]
        if len(parts) >= 3 and parts[1] == "feat":
            return parts[2]
        return parts[1] if len(parts) > 1 else str(col)

    if parts[0] == "iot":
        return parts[1] if len(parts) > 1 else str(col)

    if "__feat__" in str(col):
        tail = str(col).split("__feat__", 1)[1]
        return tail.split("__", 1)[0]

    if "__entity__" in str(col):
        tail = str(col).split("__entity__", 1)[1]
        return tail.split("__", 1)[0]

    return str(col)

def _driver_measurement_12e6(col: str) -> str:
    """Parse measurement/driver semantics without collapsing the entity into the measurement."""
    parts = [p for p in str(col).split("__") if p != ""]
    if not parts:
        return str(col)

    if parts[0] == "events_in_sec":
        if len(parts) >= 2 and parts[1] == "entity":
            return "entity_event"
        if len(parts) >= 4 and parts[1] == "feat":
            return "__".join(parts[3:])
        if len(parts) >= 3:
            return "__".join(parts[2:])
        return str(col)

    if parts[0] == "iot":
        if len(parts) >= 4:
            return "__".join(parts[2:])
        return "__".join(parts[1:])

    if "__feat__" in str(col):
        tail = str(col).split("__feat__", 1)[1]
        bits = tail.split("__")
        return "__".join(bits[1:]) if len(bits) > 1 else tail

    if "__entity__" in str(col):
        return "entity_event"

    return str(col)

def _event_indices_12e6(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    return np.flatnonzero(finite & (arr >= 0.5)).astype(np.int64)

def _run_lengths_12e6(x: np.ndarray, state=None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=np.float64)

    arr = np.where(arr >= 0.5, 1, 0).astype(np.int8)
    lengths = []
    cur = int(arr[0])
    run = 1

    for v in arr[1:]:
        v = int(v)
        if v == cur:
            run += 1
        else:
            if state is None or cur == int(state):
                lengths.append(run)
            cur = v
            run = 1

    if state is None or cur == int(state):
        lengths.append(run)

    return np.asarray(lengths, dtype=np.float64)

def _interarrival_12e6(event_idx: np.ndarray) -> np.ndarray:
    event_idx = np.asarray(event_idx, dtype=np.int64)
    if event_idx.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(event_idx).astype(np.float64)

def _empirical_ks_12e6(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 1.0

    vals = np.unique(np.concatenate([a, b]))
    if vals.size == 0:
        return 0.0

    aa = np.sort(a)
    bb = np.sort(b)

    ca = np.searchsorted(aa, vals, side="right") / float(aa.size)
    cb = np.searchsorted(bb, vals, side="right") / float(bb.size)

    return float(np.max(np.abs(ca - cb)))

def _overdispersion_12e6(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    mean = float(np.mean(arr))
    var = float(np.var(arr))
    if mean <= 0:
        return np.nan
    return float(var / max(mean, 1e-12))

def _status_from_driver_metrics_12e6(row: dict) -> tuple:
    reasons = []
    status = "pass"

    event_rate_error = _safe_float_12e6(row.get("event_rate_error"), np.nan)
    burst_count_error = _safe_float_12e6(row.get("burst_count_error"), np.nan)
    interarrival_ks = _safe_float_12e6(row.get("interarrival_ks"), np.nan)
    duration_ks = _safe_float_12e6(row.get("duration_ks"), np.nan)
    overdispersion_error = _safe_float_12e6(row.get("overdispersion_error"), np.nan)
    train_test_shift = _safe_float_12e6(row.get("abs_train_test_event_rate_shift"), np.nan)

    if np.isfinite(event_rate_error):
        if event_rate_error >= float(CFG["cell12e6_event_rate_error_fatal"]):
            status = "fatal"
            reasons.append("event_rate_error_fatal")
        elif event_rate_error >= float(CFG["cell12e6_event_rate_error_warning"]) and status != "fatal":
            status = "warning"
            reasons.append("event_rate_error_warning")

    if np.isfinite(burst_count_error):
        if burst_count_error >= float(CFG["cell12e6_burst_count_error_fatal"]):
            status = "fatal"
            reasons.append("burst_count_error_fatal")
        elif burst_count_error >= float(CFG["cell12e6_burst_count_error_warning"]) and status != "fatal":
            status = "warning"
            reasons.append("burst_count_error_warning")

    if np.isfinite(interarrival_ks):
        if interarrival_ks >= float(CFG["cell12e6_interarrival_ks_fatal"]):
            status = "fatal"
            reasons.append("interarrival_ks_fatal")
        elif interarrival_ks >= float(CFG["cell12e6_interarrival_ks_warning"]) and status != "fatal":
            status = "warning"
            reasons.append("interarrival_ks_warning")

    if np.isfinite(duration_ks):
        if duration_ks >= float(CFG["cell12e6_duration_ks_fatal"]):
            status = "fatal"
            reasons.append("duration_ks_fatal")
        elif duration_ks >= float(CFG["cell12e6_duration_ks_warning"]) and status != "fatal":
            status = "warning"
            reasons.append("duration_ks_warning")

    if np.isfinite(overdispersion_error):
        if overdispersion_error >= float(CFG["cell12e6_overdispersion_error_fatal"]):
            status = "fatal"
            reasons.append("overdispersion_error_fatal")
        elif overdispersion_error >= float(CFG["cell12e6_overdispersion_error_warning"]) and status != "fatal":
            status = "warning"
            reasons.append("overdispersion_error_warning")

    # Split/regime drift is not a standalone synthetic failure gate.
    # It is used below for attribution when synthetic-vs-real TEST QA fails.
    # Keeping it out of the status gate prevents penalizing the generator for
    # real TRAIN/VAL→TEST distribution shift when TEST synthetic fidelity is otherwise acceptable.

    if not reasons:
        reasons.append("within_driver_test_qa_gates")

    return status, "|".join(reasons)

def _driver_qa_metrics_12e6(real: np.ndarray, syn: np.ndarray) -> dict:
    real = np.asarray(real, dtype=np.float64)
    syn = np.asarray(syn, dtype=np.float64)

    real = real[np.isfinite(real)]
    syn = syn[np.isfinite(syn)]

    real_n = int(real.size)
    syn_n = int(syn.size)

    if real_n == 0 or syn_n == 0:
        return {
            "qa_eval_ok": False,
            "qa_eval_reason": "empty_real_or_synthetic",
            "real_n": real_n,
            "syn_n": syn_n,
            "event_rate_error": np.nan,
            "burst_count_error": np.nan,
            "interarrival_ks": np.nan,
            "duration_ks": np.nan,
            "overdispersion_error": np.nan,
        }

    real = np.where(real >= 0.5, 1.0, 0.0)
    syn = np.where(syn >= 0.5, 1.0, 0.0)

    real_rate = float(np.mean(real))
    syn_rate = float(np.mean(syn))
    event_rate_error = abs(real_rate - syn_rate)

    real_idx = _event_indices_12e6(real)
    syn_idx = _event_indices_12e6(syn)

    real_event_count = int(real_idx.size)
    syn_event_count = int(syn_idx.size)
    burst_count_error = abs(real_event_count - syn_event_count) / max(real_n, 1)

    real_inter = _interarrival_12e6(real_idx)
    syn_inter = _interarrival_12e6(syn_idx)
    interarrival_ks = _empirical_ks_12e6(real_inter, syn_inter)

    real_dur = _run_lengths_12e6(real, state=1)
    syn_dur = _run_lengths_12e6(syn, state=1)
    duration_ks = _empirical_ks_12e6(real_dur, syn_dur)

    real_over = _overdispersion_12e6(real)
    syn_over = _overdispersion_12e6(syn)
    overdispersion_error = abs(
        _safe_float_12e6(real_over, 0.0)
        - _safe_float_12e6(syn_over, 0.0)
    )

    return {
        "qa_eval_ok": True,
        "qa_eval_reason": "",
        "real_n": real_n,
        "syn_n": syn_n,
        "real_event_rate": real_rate,
        "syn_event_rate": syn_rate,
        "event_rate_error": float(event_rate_error),
        "real_event_count": real_event_count,
        "syn_event_count": syn_event_count,
        "burst_count_error": float(burst_count_error),
        "real_interarrival_median": float(np.median(real_inter)) if real_inter.size else np.nan,
        "syn_interarrival_median": float(np.median(syn_inter)) if syn_inter.size else np.nan,
        "interarrival_ks": float(interarrival_ks),
        "real_duration_median": float(np.median(real_dur)) if real_dur.size else np.nan,
        "syn_duration_median": float(np.median(syn_dur)) if syn_dur.size else np.nan,
        "duration_ks": float(duration_ks),
        "real_overdispersion": real_over,
        "syn_overdispersion": syn_over,
        "overdispersion_error": float(overdispersion_error) if np.isfinite(overdispersion_error) else np.nan,
    }

# ----------------------------------------------------------
# 2) Validate upstream Cell 12.e.5 contract
# ----------------------------------------------------------
contract5 = CELL12E5_DRIVER_FINAL_ENFORCEMENT_CONTRACT
if not isinstance(contract5, dict):
    raise RuntimeError("[Cell12.e.6] CELL12E5_DRIVER_FINAL_ENFORCEMENT_CONTRACT is not a dict.")

version5 = str(contract5.get("version", ""))
if "cell12e5_sparse_binary_driver_final_enforcement_strict_v1_1" not in version5 and "cell12e5_sparse_binary_driver_final_enforcement_strict_v1_0" not in version5:
    raise RuntimeError(
        "[Cell12.e.6] Unexpected Cell 12.e.5 contract version. "
        f"Expected v1.0/v1.1 compatible contract, got: {version5}"
    )

summary5 = contract5.get("enforcement_summary", {})
if int(summary5.get("enforcement_failure_count", -1)) != 0:
    raise RuntimeError(
        "[Cell12.e.6] Refusing QA because Cell 12.e.5 reported enforcement failures: "
        f"{summary5.get('enforcement_failure_count')}"
    )

if bool(contract5.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell12.e.6] Cell 12.e.5 contract indicates TEST driver values were used.")

# ----------------------------------------------------------
# 3) Normalize inputs
# ----------------------------------------------------------
final_syn = IOT_FINAL_DRIVER_TEST.copy()
final_syn.columns = final_syn.columns.astype(str)
final_syn = final_syn[IOT_DRIVER_COLS].copy()
final_syn.index = df_te.index

locked_df = CELL12E3_LOCKED_DRIVER_SELECTION_DF.copy()
locked_df["col"] = locked_df["col"].astype(str)
locked_map = {str(r["col"]): r.to_dict() for _, r in locked_df.iterrows()}

enforce_df = CELL12E5_DRIVER_FINAL_ENFORCEMENT_AUDIT_DF.copy()
enforce_df["col"] = enforce_df["col"].astype(str)
enforce_map = {str(r["col"]): r.to_dict() for _, r in enforce_df.iterrows()}

contract_df = IOT_DRIVER_TARGET_CONTRACT_DF.copy()
contract_df["col"] = contract_df["col"].astype(str)

driver_type_map = {}
if "driver_type" in contract_df.columns:
    driver_type_map = dict(zip(contract_df["col"], contract_df["driver_type"].astype(str)))

# ----------------------------------------------------------
# 3) Final TEST QA
# ----------------------------------------------------------
metrics_rows = []
status_rows = []
origin_rows = []
cell13_rows = []

log(
    "[Cell12.e.6] Running final sparse binary driver TEST QA | "
    f"targets={len(IOT_DRIVER_COLS)} | N_TE={N_TE} | TEST values used for QA only"
)

for j, col in enumerate(IOT_DRIVER_COLS, start=1):
    if j == 1 or j % 5 == 0 or j == len(IOT_DRIVER_COLS):
        log(f"[Cell12.e.6] progress {j}/{len(IOT_DRIVER_COLS)} | col={col}")

    if col not in df_te.columns:
        raise RuntimeError(f"[Cell12.e.6] Missing TEST real driver column for QA: {col}")

    tr_arr, tr_obs = _driver_observed_values_12e6(df_tr, col)
    va_arr, va_obs = _driver_observed_values_12e6(df_val, col)
    te_real_arr, te_obs = _driver_observed_values_12e6(df_te, col)

    syn_arr = pd.to_numeric(final_syn[col], errors="coerce").to_numpy(dtype=np.float64)
    syn_arr = _coerce_binary_nan_12e6(syn_arr)

    joint_obs = te_obs & np.isfinite(syn_arr)

    real_eval = te_real_arr[joint_obs]
    syn_eval = syn_arr[joint_obs]

    qa = _driver_qa_metrics_12e6(real_eval, syn_eval)

    train_vals = tr_arr[tr_obs]
    val_vals = va_arr[va_obs]
    test_vals = te_real_arr[te_obs]
    syn_vals = syn_arr[np.isfinite(syn_arr)]

    train_rate = float(np.mean(train_vals)) if train_vals.size else np.nan
    val_rate = float(np.mean(val_vals)) if val_vals.size else np.nan
    test_rate = float(np.mean(test_vals)) if test_vals.size else np.nan
    syn_rate = float(np.mean(syn_vals)) if syn_vals.size else np.nan

    abs_train_test_event_rate_shift = (
        abs(train_rate - test_rate)
        if np.isfinite(train_rate) and np.isfinite(test_rate)
        else np.nan
    )
    abs_val_test_event_rate_shift = (
        abs(val_rate - test_rate)
        if np.isfinite(val_rate) and np.isfinite(test_rate)
        else np.nan
    )

    locked = locked_map.get(col, {})
    enf = enforce_map.get(col, {})

    upstream_status = str(locked.get("driver_publication_status_after_12e3", ""))
    upstream_reasons = str(locked.get("driver_publication_reasons_after_12e3", ""))
    upstream_blocker = _safe_bool_12e6(locked.get("driver_publication_blocker_after_12e3"), False)

    metric_row = {
        "col": col,
        "entity": _entity_from_driver_col_12e6(col),
        "measurement_name": _driver_measurement_12e6(col),
        "driver_type": driver_type_map.get(col, "binary_driver"),
        "selected_driver_generator": str(locked.get("selected_driver_generator", "")),
        "selected_candidate_id": str(locked.get("selected_candidate_id", "")),
        "upstream_publication_status_after_12e3": upstream_status,
        "upstream_publication_reasons_after_12e3": upstream_reasons,
        "upstream_publication_blocker_after_12e3": upstream_blocker,
        "test_real_observed_n": int(te_obs.sum()),
        "test_syn_finite_n": int(np.isfinite(syn_arr).sum()),
        "test_joint_eval_n": int(joint_obs.sum()),
        "train_event_rate": train_rate,
        "val_event_rate": val_rate,
        "test_real_event_rate": test_rate,
        "syn_event_rate_all": syn_rate,
        "abs_train_test_event_rate_shift": abs_train_test_event_rate_shift,
        "abs_val_test_event_rate_shift": abs_val_test_event_rate_shift,
        "final_enforcement_passed": _safe_bool_12e6(enf.get("enforcement_passed"), False),
        "TEST_real_values_used_for_QA_only": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "feeds_cell13_coupling_qa": True,
        **qa,
    }

    qa_status, qa_reasons = _status_from_driver_metrics_12e6(metric_row)

    origin = "pass"
    publication_status = qa_status
    publication_blocker = bool(qa_status == "fatal")
    origin_reasons = []

    if upstream_blocker:
        origin = "carried_upstream_12e3_blocker"
        origin_reasons.append("already_blocked_before_test_qa")
        publication_blocker = True

    elif qa_status == "fatal":
        if (
            np.isfinite(abs_train_test_event_rate_shift)
            and abs_train_test_event_rate_shift >= float(CFG["cell12e6_train_test_event_rate_shift_warning"])
        ) or (
            np.isfinite(abs_val_test_event_rate_shift)
            and abs_val_test_event_rate_shift >= float(CFG["cell12e6_train_test_event_rate_shift_warning"])
        ):
            origin = "trainval_to_test_split_or_regime_drift"
            origin_reasons.append("large_trainval_to_test_event_rate_shift")
        else:
            origin = "new_12e6_test_qa_blocker"
            origin_reasons.append("driver_failed_test_qa_without_split_drift_attribution")

    elif qa_status == "warning":
        if (
            np.isfinite(abs_train_test_event_rate_shift)
            and abs_train_test_event_rate_shift >= float(CFG["cell12e6_train_test_event_rate_shift_warning"])
        ):
            origin = "test_qa_warning_with_split_drift"
            origin_reasons.append("train_to_test_event_rate_shift_warning")
        else:
            origin = "test_qa_warning"
            origin_reasons.append("warning_metric_without_fatal_failure")

    else:
        origin = "pass"
        origin_reasons.append("passed_driver_test_qa")

    metric_row["driver_test_qa_status"] = qa_status
    metric_row["driver_test_qa_reasons"] = qa_reasons
    metric_row["driver_publication_status_after_12e6"] = publication_status
    metric_row["driver_publication_blocker_after_12e6"] = bool(publication_blocker)
    metric_row["blocker_origin_after_12e6"] = origin
    metric_row["blocker_origin_reasons_after_12e6"] = "|".join(origin_reasons)

    metrics_rows.append(metric_row)

    status_rows.append({
        "col": col,
        "selected_driver_generator": metric_row["selected_driver_generator"],
        "driver_test_qa_status": qa_status,
        "driver_test_qa_reasons": qa_reasons,
        "driver_publication_status_after_12e6": publication_status,
        "driver_publication_blocker_after_12e6": bool(publication_blocker),
        "blocker_origin_after_12e6": origin,
        "blocker_origin_reasons_after_12e6": "|".join(origin_reasons),
        "upstream_publication_blocker_after_12e3": upstream_blocker,
        "event_rate_error": metric_row.get("event_rate_error", np.nan),
        "burst_count_error": metric_row.get("burst_count_error", np.nan),
        "interarrival_ks": metric_row.get("interarrival_ks", np.nan),
        "duration_ks": metric_row.get("duration_ks", np.nan),
        "overdispersion_error": metric_row.get("overdispersion_error", np.nan),
        "abs_train_test_event_rate_shift": abs_train_test_event_rate_shift,
        "abs_val_test_event_rate_shift": abs_val_test_event_rate_shift,
        "TEST_real_values_used_for_QA_only": True,
        "synthetic_values_mutated": False,
        "feeds_cell13_coupling_qa": True,
    })

    origin_rows.append({
        "col": col,
        "origin": origin,
        "origin_reasons": "|".join(origin_reasons),
        "upstream_status": upstream_status,
        "upstream_reasons": upstream_reasons,
        "qa_status": qa_status,
        "qa_reasons": qa_reasons,
        "publication_status_after_12e6": publication_status,
        "publication_blocker_after_12e6": bool(publication_blocker),
        "selected_driver_generator": metric_row["selected_driver_generator"],
        "event_rate_error": metric_row.get("event_rate_error", np.nan),
        "burst_count_error": metric_row.get("burst_count_error", np.nan),
        "interarrival_ks": metric_row.get("interarrival_ks", np.nan),
        "duration_ks": metric_row.get("duration_ks", np.nan),
        "overdispersion_error": metric_row.get("overdispersion_error", np.nan),
        "abs_train_test_event_rate_shift": abs_train_test_event_rate_shift,
        "abs_val_test_event_rate_shift": abs_val_test_event_rate_shift,
    })

    cell13_rows.append({
        "driver_col": col,
        "entity": _entity_from_driver_col_12e6(col),
        "measurement_name": _driver_measurement_12e6(col),
        "selected_driver_generator": metric_row["selected_driver_generator"],
        "driver_publication_status_after_12e6": publication_status,
        "driver_publication_blocker_after_12e6": bool(publication_blocker),
        "real_test_event_count": int(np.sum(test_vals >= 0.5)) if test_vals.size else 0,
        "synthetic_test_event_count": int(np.sum(syn_vals >= 0.5)) if syn_vals.size else 0,
        "real_test_event_rate": test_rate,
        "synthetic_test_event_rate": syn_rate,
        "event_rate_error": metric_row.get("event_rate_error", np.nan),
        "interarrival_ks": metric_row.get("interarrival_ks", np.nan),
        "duration_ks": metric_row.get("duration_ks", np.nan),
        "recommended_cell13_lag_window_seconds": int(CFG.get("cell12e6_cell13_lag_window_seconds", 10)),
        "recommended_driver_event_context_radius": int(CFG.get("cell12e6_driver_event_context_radius", 10)),
        "cell13_use": "driver_to_network_lag_proxy_and_coupling_QA",
        "TEST_real_values_used_for_QA_only": True,
        "synthetic_values_mutated": False,
    })

# ----------------------------------------------------------
# 4) Assemble reports
# ----------------------------------------------------------
metrics_df = pd.DataFrame(metrics_rows)
status_df = pd.DataFrame(status_rows)
origin_df = pd.DataFrame(origin_rows)
cell13_handoff_df = pd.DataFrame(cell13_rows)

if len(metrics_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.6] Metrics row count mismatch: got={len(metrics_df)} expected={len(IOT_DRIVER_COLS)}"
    )

blocked_metrics_df = metrics_df[
    metrics_df["driver_publication_blocker_after_12e6"].fillna(False).astype(bool)
].copy()

status_counts = (
    status_df["driver_publication_status_after_12e6"].astype(str).value_counts().sort_index().to_dict()
)
qa_status_counts = (
    status_df["driver_test_qa_status"].astype(str).value_counts().sort_index().to_dict()
)
origin_counts = (
    origin_df["origin"].astype(str).value_counts().sort_index().to_dict()
)

pass_n = int((status_df["driver_publication_status_after_12e6"].astype(str) == "pass").sum())
warning_n = int((status_df["driver_publication_status_after_12e6"].astype(str) == "warning").sum())
fatal_n = int((status_df["driver_publication_status_after_12e6"].astype(str) == "fatal").sum())
blocker_n = int(status_df["driver_publication_blocker_after_12e6"].fillna(False).astype(bool).sum())

upstream_blocker_n = int(status_df["upstream_publication_blocker_after_12e3"].fillna(False).astype(bool).sum())
split_drift_n = int((origin_df["origin"].astype(str) == "trainval_to_test_split_or_regime_drift").sum())
new_test_qa_blocker_n = int((origin_df["origin"].astype(str) == "new_12e6_test_qa_blocker").sum())

metric_summary = {
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "pass_n": pass_n,
    "warning_n": warning_n,
    "fatal_n": fatal_n,
    "publication_blocker_n": blocker_n,
    "upstream_publication_blocker_n": upstream_blocker_n,
    "split_drift_blocker_n": split_drift_n,
    "new_12e6_test_qa_blocker_n": new_test_qa_blocker_n,
    "mean_event_rate_error": float(pd.to_numeric(metrics_df["event_rate_error"], errors="coerce").mean()),
    "mean_burst_count_error": float(pd.to_numeric(metrics_df["burst_count_error"], errors="coerce").mean()),
    "mean_interarrival_ks": float(pd.to_numeric(metrics_df["interarrival_ks"], errors="coerce").mean()),
    "mean_duration_ks": float(pd.to_numeric(metrics_df["duration_ks"], errors="coerce").mean()),
    "mean_overdispersion_error": float(pd.to_numeric(metrics_df["overdispersion_error"], errors="coerce").mean()),
    "mean_abs_train_test_event_rate_shift": float(pd.to_numeric(metrics_df["abs_train_test_event_rate_shift"], errors="coerce").mean()),
    "mean_abs_val_test_event_rate_shift": float(pd.to_numeric(metrics_df["abs_val_test_event_rate_shift"], errors="coerce").mean()),
}

# ----------------------------------------------------------
# 5) Save outputs
# ----------------------------------------------------------
qa_metrics_csv = os.path.join(REPORT_DIR, "cell12e6_driver_final_test_qa_metrics.csv")
publication_status_csv = os.path.join(REPORT_DIR, "cell12e6_driver_publication_status.csv")
blocker_origin_csv = os.path.join(REPORT_DIR, "cell12e6_driver_blocker_origin_audit.csv")
blocked_metrics_csv = os.path.join(REPORT_DIR, "cell12e6_driver_publication_blocked_test_qa_metrics.csv")
cell13_handoff_csv = os.path.join(REPORT_DIR, "cell12e6_driver_cell13_handoff.csv")
cell13_handoff_manifest_json = os.path.join(REPORT_DIR, "cell12e6_driver_cell13_handoff_manifest.json")
contract_json = os.path.join(REPORT_DIR, "cell12e6_driver_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12e6_driver_final_test_qa_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell12e6_driver_final_test_qa_manifest.json")

metrics_df.to_csv(qa_metrics_csv, index=False)
status_df.to_csv(publication_status_csv, index=False)
origin_df.to_csv(blocker_origin_csv, index=False)
blocked_metrics_df.to_csv(blocked_metrics_csv, index=False)
cell13_handoff_df.to_csv(cell13_handoff_csv, index=False)

cell13_handoff_manifest = {
    "cell": "12.e.6",
    "version": CELL12E6_VERSION,
    "role": "driver_handoff_to_cell13_coupling_QA",
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "final_driver_test_global": "IOT_FINAL_DRIVER_TEST",
    "final_driver_test_path": globals().get("CELL12E5_FINAL_DRIVER_TEST_PATH", None),
    "driver_qa_metrics_csv": qa_metrics_csv,
    "driver_publication_status_csv": publication_status_csv,
    "driver_handoff_csv": cell13_handoff_csv,
    "recommended_cell13_metrics": [
        "driver_to_network_lag_proxy",
        "network_response_within_lag_window",
        "event_conditioned_protocol_burst_rate",
        "synthetic_vs_real_driver_network_lag_ks",
        "driver_event_context_window_similarity",
    ],
    "recommended_lag_window_seconds": int(CFG.get("cell12e6_cell13_lag_window_seconds", 10)),
    "recommended_driver_event_context_radius": int(CFG.get("cell12e6_driver_event_context_radius", 10)),
    "TEST_real_values_used_for_QA_only": True,
    "synthetic_values_mutated": False,
}

_write_json_12e6(cell13_handoff_manifest_json, cell13_handoff_manifest)

contract = {
    "cell": "12.e.6",
    "version": CELL12E6_VERSION,
    "role": "sparse_binary_driver_final_TEST_QA_only",
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "expected_driver_targets_total": int(EXPECTED_DRIVER_TARGET_COUNT_12E),
    "driver_type_scope": "sparse_binary_event_drivers",
    "train_rows_schema_and_split_context": int(N_TR),
    "val_rows_schema_and_split_context": int(N_VAL),
    "test_rows": int(N_TE),
    "metric_summary": metric_summary,
    "qa_status_counts": qa_status_counts,
    "publication_status_counts_after_12e6": status_counts,
    "blocker_origin_counts": origin_counts,
    "feeds_cell13_coupling_qa": True,
    "cell13_handoff_manifest": cell13_handoff_manifest,
    "thresholds": {
        "event_rate_error_warning": float(CFG["cell12e6_event_rate_error_warning"]),
        "event_rate_error_fatal": float(CFG["cell12e6_event_rate_error_fatal"]),
        "burst_count_error_warning": float(CFG["cell12e6_burst_count_error_warning"]),
        "burst_count_error_fatal": float(CFG["cell12e6_burst_count_error_fatal"]),
        "interarrival_ks_warning": float(CFG["cell12e6_interarrival_ks_warning"]),
        "interarrival_ks_fatal": float(CFG["cell12e6_interarrival_ks_fatal"]),
        "duration_ks_warning": float(CFG["cell12e6_duration_ks_warning"]),
        "duration_ks_fatal": float(CFG["cell12e6_duration_ks_fatal"]),
        "overdispersion_error_warning": float(CFG["cell12e6_overdispersion_error_warning"]),
        "overdispersion_error_fatal": float(CFG["cell12e6_overdispersion_error_fatal"]),
    },
    "TEST_real_values_used_for_QA_only": True,
    "synthetic_values_mutated": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "test_qa_done_here": True,
    "cell12e5_contract_version_seen": version5,
    "cell12e5_enforcement_summary_seen": summary5,
    "split_drift_policy": "TRAIN/VAL-to-TEST event-rate shift is attribution-only and does not by itself create a QA fatal/warning.",
    "outputs": {
        "qa_metrics_csv": qa_metrics_csv,
        "publication_status_csv": publication_status_csv,
        "blocker_origin_csv": blocker_origin_csv,
        "blocked_metrics_csv": blocked_metrics_csv,
        "cell13_handoff_csv": cell13_handoff_csv,
        "cell13_handoff_manifest_json": cell13_handoff_manifest_json,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_12e6(contract_json, contract)
_write_json_12e6(contract_canonical_json, contract)

manifest = {
    "cell": "12.e.6",
    "version": CELL12E6_VERSION,
    "created_outputs": contract["outputs"],
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "metric_summary": metric_summary,
    "qa_status_counts": qa_status_counts,
    "publication_status_counts_after_12e6": status_counts,
    "blocker_origin_counts": origin_counts,
    "feeds_cell13_coupling_qa": True,
    "qa_only_contract": {
        "TEST_real_values_used_for_QA_only": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "test_qa_done_here": True,
    },
}

_write_json_12e6(manifest_json, manifest)

hashes = {
    "qa_metrics_csv_sha256": _sha256_file_12e6(qa_metrics_csv),
    "publication_status_csv_sha256": _sha256_file_12e6(publication_status_csv),
    "blocker_origin_csv_sha256": _sha256_file_12e6(blocker_origin_csv),
    "blocked_metrics_csv_sha256": _sha256_file_12e6(blocked_metrics_csv),
    "cell13_handoff_csv_sha256": _sha256_file_12e6(cell13_handoff_csv),
    "cell13_handoff_manifest_json_sha256": _sha256_file_12e6(cell13_handoff_manifest_json),
    "contract_json_sha256": _sha256_file_12e6(contract_json),
    "contract_canonical_json_sha256": _sha256_file_12e6(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_12e6(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes
cell13_handoff_manifest["hashes"] = {
    "cell13_handoff_csv_sha256": hashes["cell13_handoff_csv_sha256"],
    "cell13_handoff_manifest_json_sha256": hashes["cell13_handoff_manifest_json_sha256"],
}

_write_json_12e6(contract_json, contract)
_write_json_12e6(contract_canonical_json, contract)
_write_json_12e6(manifest_json, manifest)
_write_json_12e6(cell13_handoff_manifest_json, cell13_handoff_manifest)

# ----------------------------------------------------------
# 6) Export globals
# ----------------------------------------------------------
globals()["CELL12E6_VERSION"] = CELL12E6_VERSION
globals()["CELL12E6_DRIVER_FINAL_TEST_QA_METRICS_DF"] = metrics_df
globals()["CELL12E6_DRIVER_PUBLICATION_STATUS_DF"] = status_df
globals()["CELL12E6_DRIVER_BLOCKER_ORIGIN_AUDIT_DF"] = origin_df
globals()["CELL12E6_DRIVER_PUBLICATION_BLOCKED_TEST_QA_METRICS_DF"] = blocked_metrics_df
globals()["CELL12E6_DRIVER_CELL13_HANDOFF_DF"] = cell13_handoff_df
globals()["CELL12E6_DRIVER_CONTRACT"] = contract
globals()["CELL12E6_DRIVER_FINAL_TEST_QA_METRICS_CSV"] = qa_metrics_csv
globals()["CELL12E6_DRIVER_PUBLICATION_STATUS_CSV"] = publication_status_csv
globals()["CELL12E6_DRIVER_BLOCKER_ORIGIN_AUDIT_CSV"] = blocker_origin_csv
globals()["CELL12E6_DRIVER_PUBLICATION_BLOCKED_TEST_QA_METRICS_CSV"] = blocked_metrics_csv
globals()["CELL12E6_DRIVER_CELL13_HANDOFF_CSV"] = cell13_handoff_csv
globals()["CELL12E6_DRIVER_CELL13_HANDOFF_MANIFEST_JSON"] = cell13_handoff_manifest_json
globals()["CELL12E6_DRIVER_CONTRACT_JSON"] = contract_json
globals()["CELL12E6_DRIVER_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL12E6_DRIVER_MANIFEST_JSON"] = manifest_json

# Cell 13 explicit handoff aliases.
globals()["CELL13_DRIVER_QA_METRICS_DF"] = metrics_df
globals()["CELL13_DRIVER_PUBLICATION_STATUS_DF"] = status_df
globals()["CELL13_DRIVER_HANDOFF_DF"] = cell13_handoff_df
globals()["CELL13_DRIVER_HANDOFF_CSV"] = cell13_handoff_csv
globals()["CELL13_DRIVER_HANDOFF_MANIFEST"] = cell13_handoff_manifest

log(
    "[Cell12.e.6] Driver final TEST QA complete | "
    f"driver_targets_total={len(IOT_DRIVER_COLS)} | "
    f"pass={pass_n} | warning={warning_n} | fatal={fatal_n} | "
    f"publication_blocker_n={blocker_n}"
)
log(
    "[Cell12.e.6] Metric summary | "
    f"mean_event_rate_error={metric_summary['mean_event_rate_error']:.8f} | "
    f"mean_burst_count_error={metric_summary['mean_burst_count_error']:.8f} | "
    f"mean_interarrival_ks={metric_summary['mean_interarrival_ks']:.6f} | "
    f"mean_duration_ks={metric_summary['mean_duration_ks']:.6f} | "
    f"mean_overdispersion_error={metric_summary['mean_overdispersion_error']:.6f}"
)
log(f"[Cell12.e.6] QA status counts | {qa_status_counts}")
log(f"[Cell12.e.6] Publication status counts after 12.e.6 | {status_counts}")
log(f"[Cell12.e.6] Blocker origin counts | {origin_counts}")
log(
    "[Cell12.e.6] Attribution summary | "
    f"upstream_publication_blocker_n={upstream_blocker_n} | "
    f"split_drift_blocker_n={split_drift_n} | "
    f"new_12e6_test_qa_blocker_n={new_test_qa_blocker_n}"
)
log(f"[Cell12.e.6] Saved QA metrics: {qa_metrics_csv} | rows={len(metrics_df)}")
log(f"[Cell12.e.6] Saved publication status: {publication_status_csv} | rows={len(status_df)}")
log(f"[Cell12.e.6] Saved blocker origin audit: {blocker_origin_csv} | rows={len(origin_df)}")
log(f"[Cell12.e.6] Saved Cell 13 driver handoff: {cell13_handoff_csv} | rows={len(cell13_handoff_df)}")
log(f"[Cell12.e.6] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell12.e.6] Contract flags | "
    "TEST_real_values_used_for_QA_only=True | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "test_qa_done_here=True | "
    "feeds_cell13_coupling_qa=True"
)
log("--- END: Cell 12.e.6 - Driver TEST QA only (v1.1 strict sparse binary, contract-hardened) ---")

gc.collect()