# ==========================================================
# CELL 12.e.5 - Driver final enforcement
# v1.1 STUDY-THESIS strict sparse binary driver enforcement, contract-hardened
#
# Role:
#   - Enforce final TEST-length sparse binary driver matrix safety.
#   - Preserve Cell 12.e.4 selected driver values unless mask/domain
#     enforcement is required.
#   - Enforce:
#       binary drivers ∈ {0, 1, NaN}
#       NaN under inactive driver availability mask
#       finite binary values under active mask
#       exact column order and TEST index
#   - Carry 12.e.3 publication attribution forward.
#   - Prepare final driver artifact for Cell 12.e.6 and Cell 13 coupling QA.
#
# Strict rules:
#   - Do NOT read real TEST driver values.
#   - Do NOT evaluate against TEST real values.
#   - Do NOT select generators.
#   - Do NOT fit generators.
#
# Outputs:
#   synthetic/IOT_FINAL_DRIVER_TEST.parquet
#   reports/cell12e5_driver_final_enforcement_audit.csv
#   reports/cell12e5_driver_publication_column_registry.csv
#   reports/cell12e5_driver_final_enforcement_contract.json
#   artifacts/cell12e5_driver_final_enforcement_manifest.json
# ==========================================================

log("--- START: Cell 12.e.5 - Driver final enforcement (v1.1 strict sparse binary, contract-hardened) ---")

import os
import re
import gc
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_12e5 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_DRIVER_COLS",
    "IOT_DRIVER_TARGET_CONTRACT_DF",
    "IOT_DRIVER_AVAIL_SYN",
    "IOT_SELECTED_DRIVER_TEST",
    "CELL12E3_LOCKED_DRIVER_SELECTION_DF",
    "CELL12E4_DRIVER_TEST_MATERIALIZATION_AUDIT_DF",
    "CELL12E4_DRIVER_TEST_MATERIALIZATION_CONTRACT",
]
_missing_12e5 = [k for k in _required_12e5 if k not in globals()]
if _missing_12e5:
    raise RuntimeError(f"[Cell12.e.5] Missing required globals from prior cells: {_missing_12e5}")

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
        "[Cell12.e.5] Driver target count mismatch: "
        f"got={len(IOT_DRIVER_COLS)} expected={EXPECTED_DRIVER_TARGET_COUNT_12E}"
    )

CELL12E5_VERSION = "cell12e5_sparse_binary_driver_final_enforcement_strict_v1_1_contract_hardened"

CFG["cell12e5_version"] = CELL12E5_VERSION
CFG["cell12e5_TEST_real_values_used"] = False
CFG["cell12e5_selection_done_here"] = False
CFG["cell12e5_generator_fit_done_here"] = False
CFG["cell12e5_test_length_materialization_done_here"] = False
CFG["cell12e5_final_enforcement_done_here"] = True
CFG["cell12e5_df_te_used_for_index_length_schema_only"] = True
CFG["cell12e5_feeds_cell13_coupling_qa"] = True

CFG.setdefault("cell12e5_fail_on_active_nonfinite", True)
CFG.setdefault("cell12e5_fail_on_nonbinary_finite", True)
CFG.setdefault("cell12e5_fail_on_inactive_finite_after_enforcement", True)
CFG.setdefault("cell12e5_fail_on_negative_driver", True)
CFG.setdefault("cell12e5_fail_on_noninteger_driver", True)

# ----------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------
def _json_sanitize_12e5(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12e5(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12e5(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12e5(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12e5(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12e5(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12e5(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12e5(obj.to_dict())
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

def _write_json_12e5(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12e5(payload), f, indent=2, sort_keys=True)

def _sha256_file_12e5(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_bool_12e5(x, default=False) -> bool:
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

def _safe_float_12e5(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _entity_from_driver_col_12e5(col: str) -> str:
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

def _driver_measurement_12e5(col: str) -> str:
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

def _normalise_driver_availability_12e5(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        mask_df = obj.copy()
    elif isinstance(obj, dict):
        mask_df = pd.DataFrame(obj)
    else:
        arr = np.asarray(obj)
        if arr.ndim != 2:
            raise RuntimeError(
                f"[Cell12.e.5] IOT_DRIVER_AVAIL_SYN cannot be normalized: shape={arr.shape}"
            )
        mask_df = pd.DataFrame(arr)

    if len(mask_df) != N_TE:
        raise RuntimeError(
            f"[Cell12.e.5] IOT_DRIVER_AVAIL_SYN row mismatch: got={len(mask_df)} expected={N_TE}"
        )

    mask_df.index = df_te.index
    mask_df.columns = mask_df.columns.astype(str)

    missing = sorted(set(IOT_DRIVER_COLS) - set(mask_df.columns))
    if missing:
        raise RuntimeError(
            f"[Cell12.e.5] IOT_DRIVER_AVAIL_SYN missing driver columns: {missing[:30]}"
        )

    out = mask_df[IOT_DRIVER_COLS].copy()
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = (out > 0.5).astype(np.float32)

    return out

def _normalise_selected_driver_test_12e5(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        x = obj.copy()
    else:
        x = pd.DataFrame(obj)

    if len(x) != N_TE:
        raise RuntimeError(
            f"[Cell12.e.5] IOT_SELECTED_DRIVER_TEST row mismatch: got={len(x)} expected={N_TE}"
        )

    x.index = df_te.index
    x.columns = x.columns.astype(str)

    missing = sorted(set(IOT_DRIVER_COLS) - set(x.columns))
    extra = sorted(set(x.columns) - set(IOT_DRIVER_COLS))

    if missing:
        raise RuntimeError(f"[Cell12.e.5] IOT_SELECTED_DRIVER_TEST missing columns: {missing[:30]}")

    if extra:
        log(f"[Cell12.e.5] Warning: IOT_SELECTED_DRIVER_TEST has extra columns; dropping preview={extra[:20]}")

    return x[IOT_DRIVER_COLS].copy()

def _transition_rate_12e5(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.mean(arr[1:] != arr[:-1]))

# ----------------------------------------------------------
# 2) Validate upstream 12.e.4 contract
# ----------------------------------------------------------
contract4 = CELL12E4_DRIVER_TEST_MATERIALIZATION_CONTRACT
if not isinstance(contract4, dict):
    raise RuntimeError("[Cell12.e.5] CELL12E4_DRIVER_TEST_MATERIALIZATION_CONTRACT is not a dict.")

version4 = str(contract4.get("version", ""))
if "cell12e4_sparse_binary_driver_test_materialization_strict_v1_1" not in version4 and "cell12e4_sparse_binary_driver_test_materialization_strict_v1_0" not in version4:
    raise RuntimeError(
        "[Cell12.e.5] Unexpected Cell 12.e.4 contract version. "
        f"Expected v1.0/v1.1 compatible contract, got: {version4}"
    )

summary4 = contract4.get("safety_summary", {})
if int(summary4.get("safety_failure_count", -1)) != 0:
    raise RuntimeError(
        "[Cell12.e.5] Refusing to enforce because Cell 12.e.4 reported safety failures: "
        f"{summary4.get('safety_failure_count')}"
    )

if bool(contract4.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell12.e.5] Cell 12.e.4 contract indicates TEST driver values were used.")

# ----------------------------------------------------------
# 3) Normalize inputs
# ----------------------------------------------------------
selected_in = _normalise_selected_driver_test_12e5(IOT_SELECTED_DRIVER_TEST)
avail_df = _normalise_driver_availability_12e5(IOT_DRIVER_AVAIL_SYN)

locked_df = CELL12E3_LOCKED_DRIVER_SELECTION_DF.copy()
locked_df["col"] = locked_df["col"].astype(str)

mat_audit_df = CELL12E4_DRIVER_TEST_MATERIALIZATION_AUDIT_DF.copy()
mat_audit_df["col"] = mat_audit_df["col"].astype(str)

contract_df = IOT_DRIVER_TARGET_CONTRACT_DF.copy()
contract_df["col"] = contract_df["col"].astype(str)

if len(locked_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.5] Locked selection row mismatch: got={len(locked_df)} expected={len(IOT_DRIVER_COLS)}"
    )

if len(mat_audit_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.5] Materialization audit row mismatch: got={len(mat_audit_df)} expected={len(IOT_DRIVER_COLS)}"
    )

locked_map = {str(r["col"]): r.to_dict() for _, r in locked_df.iterrows()}
mat_map = {str(r["col"]): r.to_dict() for _, r in mat_audit_df.iterrows()}

driver_type_map = {}
domain_contract_map = {}
integer_required_map = {}
binary_required_map = {}
nonnegative_required_map = {}

if "driver_type" in contract_df.columns:
    driver_type_map = dict(zip(contract_df["col"], contract_df["driver_type"].astype(str)))
if "domain_contract" in contract_df.columns:
    domain_contract_map = dict(zip(contract_df["col"], contract_df["domain_contract"].astype(str)))
if "integer_required" in contract_df.columns:
    integer_required_map = dict(zip(contract_df["col"], contract_df["integer_required"]))
if "binary_required" in contract_df.columns:
    binary_required_map = dict(zip(contract_df["col"], contract_df["binary_required"]))
if "nonnegative_required" in contract_df.columns:
    nonnegative_required_map = dict(zip(contract_df["col"], contract_df["nonnegative_required"]))

# ----------------------------------------------------------
# 3) Enforce driver domain and mask
# ----------------------------------------------------------
final_data = {}
audit_rows = []
registry_rows = []

log(
    "[Cell12.e.5] Enforcing final sparse binary driver TEST matrix | "
    f"targets={len(IOT_DRIVER_COLS)} | N_TE={N_TE}"
)

for j, col in enumerate(IOT_DRIVER_COLS, start=1):
    if j == 1 or j % 5 == 0 or j == len(IOT_DRIVER_COLS):
        log(f"[Cell12.e.5] progress {j}/{len(IOT_DRIVER_COLS)} | col={col}")

    locked = locked_map[col]
    mat = mat_map[col]

    raw = pd.to_numeric(selected_in[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    active_mask = (
        pd.to_numeric(avail_df[col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        > 0.5
    )

    if raw.size != N_TE or active_mask.size != N_TE:
        raise RuntimeError(
            f"[Cell12.e.5] Length mismatch for {col}: raw={raw.size} mask={active_mask.size} expected={N_TE}"
        )

    driver_type = str(driver_type_map.get(col, "binary_driver"))
    domain_contract = str(domain_contract_map.get(col, "{0,1,NaN}"))

    integer_required = _safe_bool_12e5(integer_required_map.get(col, True), True)
    binary_required = _safe_bool_12e5(binary_required_map.get(col, True), True)
    nonnegative_required = _safe_bool_12e5(nonnegative_required_map.get(col, True), True)

    before = raw.copy()
    before_finite = np.isfinite(before)

    before_inactive_finite_n = int(np.sum(~active_mask & before_finite))
    before_active_nonfinite_n = int(np.sum(active_mask & ~before_finite))
    before_negative_finite_n = int(np.sum(before_finite & (before < 0.0)))
    before_noninteger_finite_n = int(np.sum(before_finite & (np.abs(before - np.round(before)) > 1e-6)))
    before_nonbinary_finite_n = int(
        np.sum(before_finite & ~((before == 0.0) | (before == 1.0)))
    )

    # Enforce.
    out = raw.copy()

    finite = np.isfinite(out)

    if nonnegative_required:
        out[finite & (out < 0.0)] = 0.0

    if binary_required:
        # Sparse driver branch currently requires binary event indicators.
        finite = np.isfinite(out)
        out[finite & (out < 0.5)] = 0.0
        out[finite & (out >= 0.5)] = 1.0
    elif integer_required:
        finite = np.isfinite(out)
        out[finite] = np.round(out[finite])
        if nonnegative_required:
            out[finite & (out < 0.0)] = 0.0

    # Strict mask.
    out[~active_mask] = np.nan

    after_finite = np.isfinite(out)

    after_inactive_finite_n = int(np.sum(~active_mask & after_finite))
    after_active_nonfinite_n = int(np.sum(active_mask & ~after_finite))
    after_negative_finite_n = int(np.sum(after_finite & (out < 0.0)))
    after_noninteger_finite_n = int(np.sum(after_finite & (np.abs(out - np.round(out)) > 1e-6)))
    after_nonbinary_finite_n = int(
        np.sum(after_finite & ~((out == 0.0) | (out == 1.0)))
    )

    if after_inactive_finite_n != 0 and bool(CFG.get("cell12e5_fail_on_inactive_finite_after_enforcement", True)):
        raise RuntimeError(
            f"[Cell12.e.5] Inactive finite driver values remain after enforcement for {col}: "
            f"{after_inactive_finite_n}"
        )

    if after_active_nonfinite_n != 0 and bool(CFG.get("cell12e5_fail_on_active_nonfinite", True)):
        raise RuntimeError(
            f"[Cell12.e.5] Active nonfinite driver values remain after enforcement for {col}: "
            f"{after_active_nonfinite_n}"
        )

    if after_negative_finite_n != 0 and bool(CFG.get("cell12e5_fail_on_negative_driver", True)):
        raise RuntimeError(
            f"[Cell12.e.5] Negative finite driver values remain after enforcement for {col}: "
            f"{after_negative_finite_n}"
        )

    if integer_required and after_noninteger_finite_n != 0 and bool(CFG.get("cell12e5_fail_on_noninteger_driver", True)):
        raise RuntimeError(
            f"[Cell12.e.5] Non-integer finite driver values remain after enforcement for {col}: "
            f"{after_noninteger_finite_n}"
        )

    if binary_required and after_nonbinary_finite_n != 0 and bool(CFG.get("cell12e5_fail_on_nonbinary_finite", True)):
        raise RuntimeError(
            f"[Cell12.e.5] Non-binary finite driver values remain after enforcement for {col}: "
            f"{after_nonbinary_finite_n}"
        )

    changed_n = int(
        np.sum(
            (
                (np.isfinite(before) & np.isfinite(out) & (before != out))
                | (np.isfinite(before) & ~np.isfinite(out))
                | (~np.isfinite(before) & np.isfinite(out))
            )
        )
    )

    active_vals = out[active_mask & np.isfinite(out)]
    active_n = int(active_mask.sum())
    inactive_n = int((~active_mask).sum())

    final_data[col] = out.astype(np.float32)

    selected_generator = str(locked.get("selected_driver_generator", ""))
    selected_candidate_id = str(locked.get("selected_candidate_id", ""))

    publication_status = str(locked.get("driver_publication_status_after_12e3", ""))
    publication_reasons = str(locked.get("driver_publication_reasons_after_12e3", ""))
    blocker = _safe_bool_12e5(locked.get("driver_publication_blocker_after_12e3"), False)

    final_event_rate = float(np.mean(active_vals)) if active_vals.size else np.nan
    final_event_count = int(np.sum(active_vals >= 0.5)) if active_vals.size else 0

    enforcement_passed = bool(
        after_inactive_finite_n == 0
        and after_active_nonfinite_n == 0
        and after_negative_finite_n == 0
        and (not integer_required or after_noninteger_finite_n == 0)
        and (not binary_required or after_nonbinary_finite_n == 0)
    )

    audit_rows.append({
        "col": col,
        "entity": _entity_from_driver_col_12e5(col),
        "measurement_name": _driver_measurement_12e5(col),
        "driver_type": driver_type,
        "domain_contract": domain_contract,
        "integer_required": bool(integer_required),
        "binary_required": bool(binary_required),
        "nonnegative_required": bool(nonnegative_required),
        "selected_driver_generator": selected_generator,
        "selected_candidate_id": selected_candidate_id,
        "materializer_from_12e4": str(mat.get("materializer", "")),
        "publication_status_after_12e3": publication_status,
        "publication_reasons_after_12e3": publication_reasons,
        "publication_blocker_after_12e3": blocker,
        "test_rows": int(N_TE),
        "active_n": active_n,
        "inactive_n": inactive_n,
        "active_rate": float(active_n / max(N_TE, 1)),
        "before_inactive_finite_n": before_inactive_finite_n,
        "before_active_nonfinite_n": before_active_nonfinite_n,
        "before_negative_finite_n": before_negative_finite_n,
        "before_noninteger_finite_n": before_noninteger_finite_n,
        "before_nonbinary_finite_n": before_nonbinary_finite_n,
        "after_inactive_finite_n": after_inactive_finite_n,
        "after_active_nonfinite_n": after_active_nonfinite_n,
        "after_negative_finite_n": after_negative_finite_n,
        "after_noninteger_finite_n": after_noninteger_finite_n,
        "after_nonbinary_finite_n": after_nonbinary_finite_n,
        "finite_changed_by_enforcement_n": changed_n,
        "final_active_finite_n": int(active_vals.size),
        "final_event_rate_active": final_event_rate,
        "final_event_count_active": final_event_count,
        "final_transition_rate_active": _transition_rate_12e5(active_vals),
        "final_unique_values_active": sorted([float(v) for v in np.unique(active_vals)]) if active_vals.size else [],
        "enforcement_passed": enforcement_passed,
        "feeds_cell13_coupling_qa": True,
        "TEST_real_values_used": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "final_enforcement_done_here": True,
    })

    registry_rows.append({
        "col": col,
        "branch": "12.e",
        "target_role": "driver_target",
        "driver_type": driver_type,
        "output_artifact": "IOT_FINAL_DRIVER_TEST",
        "selected_driver_generator": selected_generator,
        "selected_candidate_id": selected_candidate_id,
        "publication_status_after_12e3": publication_status,
        "publication_reasons_after_12e3": publication_reasons,
        "publication_blocker_after_12e3": blocker,
        "domain_contract": domain_contract,
        "mask_contract": "NaN under inactive driver availability mask",
        "final_enforcement_passed": enforcement_passed,
        "feeds_cell13_coupling_qa": True,
        "TEST_real_values_used": False,
    })

# ----------------------------------------------------------
# 4) Assemble and global validation
# ----------------------------------------------------------
IOT_FINAL_DRIVER_TEST = pd.DataFrame(
    final_data,
    index=df_te.index,
    columns=IOT_DRIVER_COLS,
    dtype=np.float32,
)

if IOT_FINAL_DRIVER_TEST.shape != (N_TE, len(IOT_DRIVER_COLS)):
    raise RuntimeError(
        "[Cell12.e.5] IOT_FINAL_DRIVER_TEST shape mismatch: "
        f"got={IOT_FINAL_DRIVER_TEST.shape} expected={(N_TE, len(IOT_DRIVER_COLS))}"
    )

audit_df = pd.DataFrame(audit_rows)
registry_df = pd.DataFrame(registry_rows)

if len(audit_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.5] Audit row count mismatch: got={len(audit_df)} expected={len(IOT_DRIVER_COLS)}"
    )

if len(registry_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.5] Registry row count mismatch: got={len(registry_df)} expected={len(IOT_DRIVER_COLS)}"
    )

global_inactive_finite = 0
global_active_nonfinite = 0
global_negative_finite = 0
global_noninteger_finite = 0
global_nonbinary_finite = 0

for col in IOT_DRIVER_COLS:
    arr = pd.to_numeric(IOT_FINAL_DRIVER_TEST[col], errors="coerce").to_numpy(dtype=np.float64)
    mask = (
        pd.to_numeric(avail_df[col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
        > 0.5
    )

    finite = np.isfinite(arr)

    global_inactive_finite += int(np.sum(~mask & finite))
    global_active_nonfinite += int(np.sum(mask & ~finite))
    global_negative_finite += int(np.sum(finite & (arr < 0.0)))
    global_noninteger_finite += int(np.sum(finite & (np.abs(arr - np.round(arr)) > 1e-6)))
    global_nonbinary_finite += int(np.sum(finite & ~((arr == 0.0) | (arr == 1.0))))

if global_inactive_finite != 0:
    raise RuntimeError(f"[Cell12.e.5] Global inactive finite violation: {global_inactive_finite}")

if global_active_nonfinite != 0:
    raise RuntimeError(f"[Cell12.e.5] Global active nonfinite violation: {global_active_nonfinite}")

if global_negative_finite != 0:
    raise RuntimeError(f"[Cell12.e.5] Global negative finite driver violation: {global_negative_finite}")

if global_noninteger_finite != 0:
    raise RuntimeError(f"[Cell12.e.5] Global non-integer finite driver violation: {global_noninteger_finite}")

if global_nonbinary_finite != 0:
    raise RuntimeError(f"[Cell12.e.5] Global non-binary finite driver violation: {global_nonbinary_finite}")

# ----------------------------------------------------------
# 5) Save outputs
# ----------------------------------------------------------
final_driver_test_path = os.path.join(OUT_SYN, "IOT_FINAL_DRIVER_TEST.parquet")
enforcement_audit_csv = os.path.join(REPORT_DIR, "cell12e5_driver_final_enforcement_audit.csv")
publication_registry_csv = os.path.join(REPORT_DIR, "cell12e5_driver_publication_column_registry.csv")
contract_json = os.path.join(REPORT_DIR, "cell12e5_driver_final_enforcement_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12e5_driver_final_enforcement_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell12e5_driver_final_enforcement_manifest.json")

IOT_FINAL_DRIVER_TEST.to_parquet(final_driver_test_path, index=True)
audit_df.to_csv(enforcement_audit_csv, index=False)
registry_df.to_csv(publication_registry_csv, index=False)

selected_counts = (
    audit_df["selected_driver_generator"].astype(str).value_counts().sort_index().to_dict()
)

publication_counts = (
    audit_df["publication_status_after_12e3"].astype(str).value_counts().sort_index().to_dict()
)

enforcement_summary = {
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "test_rows": int(N_TE),
    "active_total": int(audit_df["active_n"].sum()),
    "inactive_total": int(audit_df["inactive_n"].sum()),
    "before_inactive_finite_total": int(audit_df["before_inactive_finite_n"].sum()),
    "before_active_nonfinite_total": int(audit_df["before_active_nonfinite_n"].sum()),
    "before_negative_finite_total": int(audit_df["before_negative_finite_n"].sum()),
    "before_noninteger_finite_total": int(audit_df["before_noninteger_finite_n"].sum()),
    "before_nonbinary_finite_total": int(audit_df["before_nonbinary_finite_n"].sum()),
    "after_inactive_finite_total": int(audit_df["after_inactive_finite_n"].sum()),
    "after_active_nonfinite_total": int(audit_df["after_active_nonfinite_n"].sum()),
    "after_negative_finite_total": int(audit_df["after_negative_finite_n"].sum()),
    "after_noninteger_finite_total": int(audit_df["after_noninteger_finite_n"].sum()),
    "after_nonbinary_finite_total": int(audit_df["after_nonbinary_finite_n"].sum()),
    "finite_changed_by_enforcement_total": int(audit_df["finite_changed_by_enforcement_n"].sum()),
    "enforcement_failure_count": int((~audit_df["enforcement_passed"].astype(bool)).sum()),
    "publication_blocker_carried_from_12e3_n": int(
        audit_df["publication_blocker_after_12e3"].fillna(False).astype(bool).sum()
    ),
    "feeds_cell13_coupling_qa": True,
}

contract = {
    "cell": "12.e.5",
    "version": CELL12E5_VERSION,
    "role": "sparse_binary_driver_final_mask_domain_schema_enforcement",
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "expected_driver_targets_total": int(EXPECTED_DRIVER_TARGET_COUNT_12E),
    "driver_type_scope": "sparse_binary_event_drivers",
    "test_rows": int(N_TE),
    "selected_counts": selected_counts,
    "publication_counts_carried_from_12e3": publication_counts,
    "enforcement_summary": enforcement_summary,
    "feeds_cell13_coupling_qa": True,
    "TEST_real_values_used": False,
    "test_length_materialization_done_here": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "final_enforcement_done_here": True,
    "df_te_used_for_index_length_schema_only": True,
    "raw_TEST_driver_values_read": False,
    "cell12e4_contract_version_seen": version4,
    "cell12e4_safety_summary_seen": summary4,
    "outputs": {
        "final_driver_test_path": final_driver_test_path,
        "enforcement_audit_csv": enforcement_audit_csv,
        "publication_registry_csv": publication_registry_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_12e5(contract_json, contract)
_write_json_12e5(contract_canonical_json, contract)

manifest = {
    "cell": "12.e.5",
    "version": CELL12E5_VERSION,
    "created_outputs": contract["outputs"],
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "selected_counts": selected_counts,
    "publication_counts_carried_from_12e3": publication_counts,
    "enforcement_summary": enforcement_summary,
    "feeds_cell13_coupling_qa": True,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "final_enforcement_done_here": True,
    },
}

_write_json_12e5(manifest_json, manifest)

hashes = {
    "final_driver_test_sha256": _sha256_file_12e5(final_driver_test_path),
    "enforcement_audit_csv_sha256": _sha256_file_12e5(enforcement_audit_csv),
    "publication_registry_csv_sha256": _sha256_file_12e5(publication_registry_csv),
    "contract_json_sha256": _sha256_file_12e5(contract_json),
    "contract_canonical_json_sha256": _sha256_file_12e5(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_12e5(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12e5(contract_json, contract)
_write_json_12e5(contract_canonical_json, contract)
_write_json_12e5(manifest_json, manifest)

# ----------------------------------------------------------
# 6) Export globals for 12.e.6 / Cell 13+
# ----------------------------------------------------------
globals()["CELL12E5_VERSION"] = CELL12E5_VERSION
globals()["IOT_FINAL_DRIVER_TEST"] = IOT_FINAL_DRIVER_TEST
globals()["CELL12E5_DRIVER_FINAL_ENFORCEMENT_AUDIT_DF"] = audit_df
globals()["CELL12E5_DRIVER_PUBLICATION_COLUMN_REGISTRY_DF"] = registry_df
globals()["CELL12E5_DRIVER_FINAL_ENFORCEMENT_CONTRACT"] = contract
globals()["CELL12E5_FINAL_DRIVER_TEST_PATH"] = final_driver_test_path
globals()["CELL12E5_DRIVER_FINAL_ENFORCEMENT_AUDIT_CSV"] = enforcement_audit_csv
globals()["CELL12E5_DRIVER_PUBLICATION_COLUMN_REGISTRY_CSV"] = publication_registry_csv
globals()["CELL12E5_DRIVER_FINAL_ENFORCEMENT_CONTRACT_JSON"] = contract_json
globals()["CELL12E5_DRIVER_FINAL_ENFORCEMENT_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL12E5_DRIVER_FINAL_ENFORCEMENT_MANIFEST_JSON"] = manifest_json

# Cell 13 explicit handoff aliases.
globals()["CELL13_DRIVER_SYN_TEST"] = IOT_FINAL_DRIVER_TEST
globals()["CELL13_DRIVER_SYN_TEST_PATH"] = final_driver_test_path
globals()["CELL13_DRIVER_CONTRACT_DF"] = registry_df
globals()["CELL13_DRIVER_ENFORCEMENT_AUDIT_DF"] = audit_df

log(
    "[Cell12.e.5] Driver final enforcement complete | "
    f"targets={len(IOT_DRIVER_COLS)} | "
    f"shape={IOT_FINAL_DRIVER_TEST.shape} | "
    f"enforcement_failure_count={enforcement_summary['enforcement_failure_count']} | "
    f"publication_blocker_carried_from_12e3_n={enforcement_summary['publication_blocker_carried_from_12e3_n']}"
)
log(f"[Cell12.e.5] Selected generator counts | {selected_counts}")
log(f"[Cell12.e.5] Publication counts carried from 12.e.3 | {publication_counts}")
log(
    "[Cell12.e.5] Enforcement summary | "
    f"active_total={enforcement_summary['active_total']} | "
    f"inactive_total={enforcement_summary['inactive_total']} | "
    f"after_inactive_finite_total={enforcement_summary['after_inactive_finite_total']} | "
    f"after_active_nonfinite_total={enforcement_summary['after_active_nonfinite_total']} | "
    f"after_negative_finite_total={enforcement_summary['after_negative_finite_total']} | "
    f"after_noninteger_finite_total={enforcement_summary['after_noninteger_finite_total']} | "
    f"after_nonbinary_finite_total={enforcement_summary['after_nonbinary_finite_total']} | "
    f"finite_changed_by_enforcement_total={enforcement_summary['finite_changed_by_enforcement_total']}"
)
log(f"[Cell12.e.5] Saved final driver TEST values: {final_driver_test_path}")
log(f"[Cell12.e.5] Saved enforcement audit: {enforcement_audit_csv} | rows={len(audit_df)}")
log(f"[Cell12.e.5] Saved publication registry: {publication_registry_csv} | rows={len(registry_df)}")
log(f"[Cell12.e.5] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell12.e.5] Contract flags | "
    "TEST_real_values_used=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "final_enforcement_done_here=True | "
    "df_te_used_for_index_length_schema_only=True | "
    "feeds_cell13_coupling_qa=True"
)
log("--- END: Cell 12.e.5 - Driver final enforcement (v1.1 strict sparse binary, contract-hardened) ---")

gc.collect()