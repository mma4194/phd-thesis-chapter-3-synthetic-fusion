# ==========================================================
# CELL 12.c.1 — IoT value A1 VAL backbone materialization
# v2.1-THESIS modular STUDY-THESIS
#
# Purpose:
#   Build the mandatory A1 support-preserving IoT value baseline
#   for VAL-only portfolio comparison.
#
# Revised architecture contract:
#   - A1 is mandatory/default for every IoT continuous value column.
#   - A1 is fitted from TRAIN donor values only.
#   - This cell materializes VAL-sized A1 candidates only.
#   - This cell does NOT materialize TEST-length values.
#   - TEST-length materialization happens later in Cell 12.c.4
#     after Cell 12.c.3 locks generator choices using VAL only.
#   - TEST real values are never used here.
#   - TEST synthetic masks / VALUE_AVAIL_SYN are not used here.
#   - VAL availability uses VAL observed mask only for fair candidate scoring.
#
# Inputs expected from 12.c.0:
#   - CONT_VALUE_COLS
#   - VALUE_CONTRACT_DF / CELL12C_VALUE_CONTRACT_DF / cell12c0_value_contract.csv
#   - IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF
#   - CELL12C_POLICY
#   - OUTDIR, OUT_SYN, REPORT_DIR
#
# Outputs:
#   - synthetic/A1_IOT_VALUES_VAL.parquet
#   - reports/cell12c1_a1_iot_value_backbone_audit.csv
#   - reports/cell12c1_a1_iot_value_backbone_contract.json
#   - artifacts/cell12c1_a1_iot_value_backbone_manifest.json
# ==========================================================

log("--- START: Cell 12.c.1 — IoT value A1 VAL backbone materialization (v2.1-THESIS contract-locked modular STUDY-THESIS) ---")

import os
import re
import gc
import json
import math
import hashlib
import warnings
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_12c1 = [
    "CFG", "log",
    "df_tr", "df_val",
    "CONT_VALUE_COLS",
    "IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
]

_missing_12c1 = [k for k in _required_12c1 if k not in globals()]
if _missing_12c1:
    raise RuntimeError(f"[Cell12.c.1] Missing required globals from 12.c.0/12.b: {_missing_12c1}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

def _read_json_dict_required_12c1(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"[Cell12.c.1] Missing required {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell12.c.1] Required {label} must be a JSON object: {path}")
    return obj

CELL12C0_CONTRACT_PATH_REQUIRED = os.path.join(
    CONTRACT_DIR,
    "cell12c0_foundation_contract_v1_2_THESIS.json",
)
CELL12B_SCAFFOLD_CONTRACT_PATH_REQUIRED = os.path.join(
    CONTRACT_DIR,
    "cell12b_iot_scaffold_contract_v27_1_1_THESIS.json",
)

CELL12C0_CONTRACT_REQUIRED = _read_json_dict_required_12c1(
    CELL12C0_CONTRACT_PATH_REQUIRED,
    "Cell 12.c.0 v1.2 foundation contract",
)
CELL12B_SCAFFOLD_CONTRACT_REQUIRED = _read_json_dict_required_12c1(
    CELL12B_SCAFFOLD_CONTRACT_PATH_REQUIRED,
    "Cell 12.b v27.1.1 scaffold contract",
)

_policy_required = CELL12C0_CONTRACT_REQUIRED.get("policy", {})
if bool(_policy_required.get("test_used_for_generator_selection", False)):
    raise RuntimeError("[Cell12.c.1] Cell 12.c.0 contract indicates TEST selection, which is forbidden.")
if str(_policy_required.get("selection_split", "VAL_only")) != "VAL_only":
    raise RuntimeError("[Cell12.c.1] Cell 12.c.0 contract selection_split must be VAL_only.")

_s12b_test_usage = CELL12B_SCAFFOLD_CONTRACT_REQUIRED.get("test_usage", {})
for _key in [
    "test_values_used_for_fitting",
    "test_values_used_for_selection",
    "test_values_used_for_calibration",
    "test_values_used_for_repair",
    "test_values_used_for_continuous_value_generation",
]:
    if bool(_s12b_test_usage.get(_key, False)):
        raise RuntimeError(f"[Cell12.c.1] Cell 12.b scaffold contract violation: {_key}=True")

log(
    "[Cell12.c.1] Upstream contracts locked | "
    "cell12c0_v1_2=True | cell12b_v27_1_1=True"
)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))

if N_TR <= 0 or N_VAL <= 0:
    raise RuntimeError(f"[Cell12.c.1] Invalid split lengths: N_TR={N_TR}, N_VAL={N_VAL}")

CONT_VALUE_COLS = list(CONT_VALUE_COLS)
if len(CONT_VALUE_COLS) == 0:
    raise RuntimeError("[Cell12.c.1] CONT_VALUE_COLS is empty.")

# ----------------------------------------------------------
# 1) Enforce revised architecture / no-TEST boundary
# ----------------------------------------------------------
CELL12C1_VERSION = "cell12c1_a1_val_backbone_v2_1_THESIS"

if "CELL12C_POLICY" in globals() and isinstance(globals()["CELL12C_POLICY"], dict):
    if bool(globals()["CELL12C_POLICY"].get("test_used_for_generator_selection", False)):
        raise RuntimeError("[Cell12.c.1] CELL12C_POLICY contradicts no-TEST-selection contract.")
    if str(globals()["CELL12C_POLICY"].get("selection_split", "VAL_only")) != "VAL_only":
        raise RuntimeError("[Cell12.c.1] CELL12C_POLICY selection_split must be VAL_only.")

CFG["cell12c1_version"] = CELL12C1_VERSION
CFG["cell12c1_generates_test_values"] = False
CFG["cell12c1_uses_test_values"] = False
CFG["cell12c1_fit_split"] = "TRAIN_only"
CFG["cell12c1_val_mask_policy"] = "VAL_observed_mask_only"

# ----------------------------------------------------------
# 2) Load / normalize 12.c.0 value contract
# ----------------------------------------------------------
if "VALUE_CONTRACT_DF" in globals() and isinstance(globals()["VALUE_CONTRACT_DF"], pd.DataFrame):
    VALUE_CONTRACT_DF = globals()["VALUE_CONTRACT_DF"].copy()
elif "CELL12C_VALUE_CONTRACT_DF" in globals() and isinstance(globals()["CELL12C_VALUE_CONTRACT_DF"], pd.DataFrame):
    VALUE_CONTRACT_DF = globals()["CELL12C_VALUE_CONTRACT_DF"].copy()
else:
    _value_contract_csv = os.path.join(REPORT_DIR, "cell12c0_value_contract.csv")
    if not os.path.exists(_value_contract_csv):
        raise RuntimeError(
            "[Cell12.c.1] Missing VALUE_CONTRACT_DF and cannot find canonical 12.c.0 CSV: "
            f"{_value_contract_csv}. Re-run Cell 12.c.0 first."
        )
    VALUE_CONTRACT_DF = pd.read_csv(_value_contract_csv)

if not isinstance(VALUE_CONTRACT_DF, pd.DataFrame) or len(VALUE_CONTRACT_DF) == 0:
    raise RuntimeError("[Cell12.c.1] VALUE_CONTRACT_DF is empty or invalid.")

if "col" not in VALUE_CONTRACT_DF.columns:
    for _cand in ["column", "value_col", "iot_col"]:
        if _cand in VALUE_CONTRACT_DF.columns:
            VALUE_CONTRACT_DF = VALUE_CONTRACT_DF.rename(columns={_cand: "col"})
            break

if "col" not in VALUE_CONTRACT_DF.columns:
    raise RuntimeError(
        "[Cell12.c.1] VALUE_CONTRACT_DF must contain a 'col' column after normalization. "
        f"Available columns={list(VALUE_CONTRACT_DF.columns)}"
    )

VALUE_CONTRACT_DF["col"] = VALUE_CONTRACT_DF["col"].astype(str)

_missing_contract_cols = sorted(set(CONT_VALUE_COLS) - set(VALUE_CONTRACT_DF["col"].tolist()))
if _missing_contract_cols:
    raise RuntimeError(
        "[Cell12.c.1] VALUE_CONTRACT_DF does not cover all CONT_VALUE_COLS. "
        f"missing={_missing_contract_cols[:30]} n_missing={len(_missing_contract_cols)}"
    )

globals()["VALUE_CONTRACT_DF"] = VALUE_CONTRACT_DF

log(
    "[Cell12.c.1] Loaded value contract from 12.c.0 | "
    f"rows={len(VALUE_CONTRACT_DF)} | cols={len(VALUE_CONTRACT_DF.columns)}"
)

# ----------------------------------------------------------
# 3) Config
# ----------------------------------------------------------
CFG12C1 = {
    "version": CELL12C1_VERSION,
    "role": "A1_VAL_backbone_only",
    "a1_fit_split": "TRAIN_only",
    "val_mask_policy": "VAL_observed_mask_for_candidate_evaluation",
    "test_materialization_policy": "deferred_to_Cell12c4_after_VAL_locked_selection",
    "min_train_values": 4,
    "piece_min": 64,
    "piece_max": 4096,
    "piece_draw_cap": 768,
    "block_len_default": 512,
    "block_len_sparse": 128,
    "block_len_counter": 256,
    "block_len_dense": 2048,
    "jitter_max": 128,
    "support_round_decimals": 6,
    "constant_std_tol": 1e-8,
    "count_integer_tol": 1e-6,
    "strict_fail_on_missing_a1": True,
    "preserve_train_support_for_low_cardinality": True,
    "low_cardinality_unique_max": 64,
    "audit_topk_log": 20,
}

globals()["CFG12C1"] = CFG12C1

# ----------------------------------------------------------
# 4) Utility helpers
# ----------------------------------------------------------
def _stable_seed(name: str, offset: int = 0) -> int:
    h = hashlib.sha256(f"{SEED}|{offset}|{name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)

def _rng_for(name: str, offset: int = 0):
    return np.random.default_rng(_stable_seed(name, offset=offset))

def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize(obj.to_dict())
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

def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2)

def _sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _finite_np(x, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    return arr[np.isfinite(arr)]

def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), np.nan, dtype=np.float32), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").astype(np.float32)

def _safe_mean(x: np.ndarray) -> float:
    x = _finite_np(x, dtype=np.float64)
    return float(np.mean(x)) if x.size else np.nan

def _safe_std(x: np.ndarray) -> float:
    x = _finite_np(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0 if x.size == 1 else np.nan
    return float(np.std(x))

def _safe_q(x: np.ndarray, q: float) -> float:
    x = _finite_np(x, dtype=np.float64)
    return float(np.quantile(x, q)) if x.size else np.nan

def _nonzero_rate(x: np.ndarray) -> float:
    x = _finite_np(x, dtype=np.float64)
    return float(np.mean(x > 0)) if x.size else np.nan

def _get_contract_row(col: str) -> dict:
    row = VALUE_CONTRACT_DF.loc[VALUE_CONTRACT_DF["col"].astype(str) == str(col)]
    if len(row) == 0:
        return {}
    return row.iloc[0].to_dict()

def _family_for_col(col: str) -> str:
    if "FAMILY_MAP" in globals() and isinstance(globals()["FAMILY_MAP"], dict):
        return str(globals()["FAMILY_MAP"].get(col, "other"))
    r = _get_contract_row(col)
    return str(r.get("family", "other"))

def _subfamily_for_col(col: str) -> str:
    if "SYNTHESIS_SUBFAMILY_MAP" in globals() and isinstance(globals()["SYNTHESIS_SUBFAMILY_MAP"], dict):
        return str(globals()["SYNTHESIS_SUBFAMILY_MAP"].get(col, ""))
    return ""

def _infer_entity(col: str) -> str:
    if "CANON_ENTITY_MAP" in globals() and isinstance(globals()["CANON_ENTITY_MAP"], dict):
        if col in globals()["CANON_ENTITY_MAP"]:
            return str(globals()["CANON_ENTITY_MAP"][col])
    if "ENTITY_MAP" in globals() and isinstance(globals()["ENTITY_MAP"], dict):
        if col in globals()["ENTITY_MAP"]:
            return str(globals()["ENTITY_MAP"][col])
    r = _get_contract_row(col)
    if "entity" in r and pd.notna(r["entity"]):
        return str(r["entity"])
    s = str(col).lower()
    if s.startswith("iot__"):
        s = s[5:]
    parts = s.split("__")
    return parts[0] if parts else "misc"

def _stage_kind_for_col(col: str) -> str:
    r = _get_contract_row(col)
    if "expected_stage12c_kind" in r and pd.notna(r["expected_stage12c_kind"]):
        return str(r["expected_stage12c_kind"])
    if "stage12c_kind" in r and pd.notna(r["stage12c_kind"]):
        return str(r["stage12c_kind"])
    return ""

_COUNTLIKE_RX = re.compile(
    r"(count|coffees|cups|total_cleaning_count|trigger_count|open_count|"
    r"close_count|motion_count|press_count|wakeup_count|episode_count)",
    re.IGNORECASE,
)

_COUNT_OR_ACCUM_RX = re.compile(
    r"(count|coffees|cups|total_cleaning|cleaning_time|cleaning_area|"
    r"consumption|energy|precipitation|events_total|events_entity_unique)",
    re.IGNORECASE,
)

_SIGNED_ALLOWED_RX = re.compile(
    r"(rssi|signal_strength|signal_level|x_axis|y_axis|z_axis|accel|acceleration)",
    re.IGNORECASE,
)

_CIRCULAR_ANGLE_RX = re.compile(
    r"(wind_angle|angle|azimuth|bearing|direction)",
    re.IGNORECASE,
)

def _is_signed_allowed_domain(col: str) -> bool:
    return bool(_SIGNED_ALLOWED_RX.search(str(col).lower()))

def _is_circular_angle_domain(col: str) -> bool:
    s = str(col).lower()
    return bool(_CIRCULAR_ANGLE_RX.search(s)) and not _is_signed_allowed_domain(col)

def _is_nonnegative_domain(col: str) -> bool:
    s = str(col).lower()

    if _is_signed_allowed_domain(col):
        return False

    if _is_circular_angle_domain(col):
        return True

    return any(k in s for k in [
        "battery", "voltage", "linkquality", "lqi",
        "humidity", "pressure", "illuminance",
        "brightness", "color_temp", "consumption", "current",
        "power", "energy", "count", "cups", "coffees",
        "cleaning", "precipitation", "purity", "co2",
        "carbon_dioxide", "snoring", "sleep", "wakeup",
        "programme_progress", "time_left", "volume_level",
        "heart_rate", "respiratory_rate", "distance", "steps",
        "calories",
    ])

def _is_integer_count_domain(col: str) -> bool:
    s = str(col).lower()
    if any(k in s for k in [
        "current", "power", "voltage", "temperature", "humidity",
        "pressure", "illuminance", "rssi", "signal_strength",
        "signal_level", "x_axis", "y_axis", "z_axis",
    ]):
        return False
    return bool(_COUNTLIKE_RX.search(s))

def _is_countlike_or_accumulator(col: str) -> bool:
    s = str(col).lower()
    fam = _family_for_col(col).lower()
    ssf = _subfamily_for_col(col).lower()
    return (
        fam in {"cumulative_count", "aux_global_counter"}
        or "count" in ssf
        or bool(_COUNT_OR_ACCUM_RX.search(s))
    )

def _domain_postprocess_col(col: str, arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return out.astype(np.float32, copy=False)

    s = str(col).lower()

    if _is_nonnegative_domain(col):
        out[finite] = np.maximum(out[finite], 0.0)

    if _is_circular_angle_domain(col):
        out[finite] = np.mod(out[finite], 360.0)

    if "humidity" in s:
        out[finite] = np.clip(out[finite], 0.0, 100.0)

    if "battery" in s and ("percent" in s or "__battery__value" in s or "battery_level" in s):
        out[finite] = np.clip(out[finite], 0.0, 100.0)

    if "programme_progress" in s:
        out[finite] = np.clip(out[finite], 0.0, 100.0)

    if "volume_level" in s:
        out[finite] = np.clip(out[finite], 0.0, 1.0)

    if "color_temp_kelvin" in s:
        out[finite] = np.clip(out[finite], 1000.0, 12000.0)
    elif "color_temp" in s:
        out[finite] = np.clip(out[finite], 0.0, 100000.0)

    if "linkquality" in s or "lqi" in s:
        out[finite] = np.clip(out[finite], 0.0, 255.0)

    if _is_integer_count_domain(col):
        out[finite] = np.rint(out[finite]).astype(np.float32)

    return out.astype(np.float32, copy=False)

def _train_values(col: str) -> np.ndarray:
    x = _numeric_series(df_tr, col).to_numpy(dtype=np.float32, copy=False)
    x = x[np.isfinite(x)]
    x = _domain_postprocess_col(col, x)
    return x.astype(np.float32, copy=False)

def _val_mask(col: str) -> np.ndarray:
    # VAL mask is used only to decide where the VAL candidate is evaluated.
    # Generated values still come from TRAIN donors only.
    return _numeric_series(df_val, col).notna().to_numpy(dtype=bool, copy=False)

def _is_low_cardinality_train(x: np.ndarray) -> bool:
    x = _finite_np(x, dtype=np.float64)
    if x.size == 0:
        return True
    rounded = np.round(x, int(CFG12C1["support_round_decimals"]))
    return int(np.unique(rounded).size) <= int(CFG12C1["low_cardinality_unique_max"])

def _is_constant_train(x: np.ndarray):
    x = _finite_np(x, dtype=np.float64)
    if x.size == 0:
        return True, np.nan
    rounded = np.round(x, int(CFG12C1["support_round_decimals"]))
    u = np.unique(rounded)
    if u.size <= 1 or float(np.nanstd(x)) <= float(CFG12C1["constant_std_tol"]):
        return True, float(np.nanmedian(x))
    return False, float(np.nanmedian(x))

def _support_project_low_cardinality(col: str, arr: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()

    if not bool(CFG12C1["preserve_train_support_for_low_cardinality"]):
        return out

    train_x = _finite_np(train_x, dtype=np.float64)
    if train_x.size == 0:
        return out

    if not _is_low_cardinality_train(train_x):
        return out

    support = np.unique(np.round(train_x, int(CFG12C1["support_round_decimals"]))).astype(np.float32)
    if support.size == 0:
        return out

    finite_idx = np.where(np.isfinite(out))[0]
    if finite_idx.size == 0:
        return out

    vals = out[finite_idx].astype(np.float32)

    # Safe because support is deliberately capped at <=64.
    dist = np.abs(vals[:, None] - support[None, :])
    out[finite_idx] = support[np.argmin(dist, axis=1)].astype(np.float32)

    return out.astype(np.float32, copy=False)

def _block_len_for_col(col: str, train_x: np.ndarray) -> int:
    fam = _family_for_col(col).lower()
    ssf = _subfamily_for_col(col).lower()
    x = _finite_np(train_x)

    if x.size <= 0:
        return int(CFG12C1["block_len_default"])

    if _is_countlike_or_accumulator(col) or "count" in ssf:
        return int(CFG12C1["block_len_counter"])

    nz = _nonzero_rate(x)
    if np.isfinite(nz) and nz < 0.10:
        return int(CFG12C1["block_len_sparse"])

    if fam in {
        "temperature", "humidity", "pressure", "environmental",
        "device_telemetry", "light", "activity_tracker",
    }:
        return int(CFG12C1["block_len_dense"])

    return int(CFG12C1["block_len_default"])

def _sample_iid_from_train(col: str, train_x: np.ndarray, n: int, rrng) -> np.ndarray:
    train_x = _finite_np(train_x, dtype=np.float32)
    if train_x.size == 0:
        return np.zeros(n, dtype=np.float32)
    idx = rrng.integers(0, train_x.size, size=int(n))
    out = train_x[idx].astype(np.float32)
    return _domain_postprocess_col(col, out)

def _sample_blocks_from_train(col: str, train_x: np.ndarray, n: int, rrng, block_len: int) -> np.ndarray:
    train_x = _finite_np(train_x, dtype=np.float32)
    n = int(n)

    if n <= 0:
        return np.empty(0, dtype=np.float32)

    if train_x.size == 0:
        return np.zeros(n, dtype=np.float32)

    if train_x.size < 4:
        return _sample_iid_from_train(col, train_x, n, rrng)

    L = int(max(1, min(block_len, max(1, train_x.size))))
    out_parts = []
    total = 0

    while total < n:
        if train_x.size <= L:
            seg = train_x.copy()
        else:
            s = int(rrng.integers(0, train_x.size - L + 1))
            seg = train_x[s:s + L].copy()

            jitter_max = int(CFG12C1.get("jitter_max", 0))
            if jitter_max > 0 and seg.size > 8:
                jitter = int(rrng.integers(-jitter_max, jitter_max + 1))
                if jitter != 0:
                    s2 = int(np.clip(s + jitter, 0, train_x.size - L))
                    seg = train_x[s2:s2 + L].copy()

        out_parts.append(seg)
        total += int(seg.size)

    out = np.concatenate(out_parts, axis=0)[:n].astype(np.float32)
    return _domain_postprocess_col(col, out)

def _generate_a1_val_for_col(col: str, n_out: int, active_mask: np.ndarray):
    rrng = _rng_for(f"a1_val::{col}", offset=12010)

    active_mask = np.asarray(active_mask, dtype=bool)
    if active_mask.shape[0] != int(n_out):
        raise RuntimeError(
            f"[Cell12.c.1] active_mask length mismatch for VAL.{col}: "
            f"{active_mask.shape[0]} vs {n_out}"
        )

    train_x = _train_values(col)
    const_like, const_val = _is_constant_train(train_x)

    out = np.full(int(n_out), np.nan, dtype=np.float32)
    n_active = int(active_mask.sum())

    if n_active == 0:
        return out, {
            "method": "zero_active_val_mask",
            "train_n": int(train_x.size),
            "active_n": int(n_active),
            "constant_like_train": bool(const_like),
        }

    if train_x.size < int(CFG12C1["min_train_values"]):
        fill = np.float32(0.0 if not np.isfinite(const_val) else const_val)
        vals = np.full(n_active, fill, dtype=np.float32)
        method = "low_train_support_constant_fill"

    elif const_like:
        fill = np.float32(0.0 if not np.isfinite(const_val) else const_val)
        vals = np.full(n_active, fill, dtype=np.float32)
        method = "train_constant_preserve"

    else:
        block_len = _block_len_for_col(col, train_x)
        vals = _sample_blocks_from_train(col, train_x, n_active, rrng, block_len=block_len)
        method = f"train_block_bootstrap_L{block_len}"

    vals = _domain_postprocess_col(col, vals)
    vals = _support_project_low_cardinality(col, vals, train_x)
    vals = _domain_postprocess_col(col, vals)

    out[active_mask] = vals.astype(np.float32)

    active_vals = out[active_mask]

    audit = {
        "method": method,
        "train_n": int(train_x.size),
        "active_n": int(n_active),
        "constant_like_train": bool(const_like),
        "train_mean": _safe_mean(train_x),
        "train_std": _safe_std(train_x),
        "train_q01": _safe_q(train_x, 0.01),
        "train_q05": _safe_q(train_x, 0.05),
        "train_q50": _safe_q(train_x, 0.50),
        "train_q95": _safe_q(train_x, 0.95),
        "train_q99": _safe_q(train_x, 0.99),
        "train_nonzero_rate": _nonzero_rate(train_x),
        "a1_val_mean": _safe_mean(active_vals),
        "a1_val_std": _safe_std(active_vals),
        "a1_val_q01": _safe_q(active_vals, 0.01),
        "a1_val_q05": _safe_q(active_vals, 0.05),
        "a1_val_q50": _safe_q(active_vals, 0.50),
        "a1_val_q95": _safe_q(active_vals, 0.95),
        "a1_val_q99": _safe_q(active_vals, 0.99),
        "a1_val_nonzero_rate": _nonzero_rate(active_vals),
        "low_cardinality_train": bool(_is_low_cardinality_train(train_x)),
        "countlike_or_accumulator": bool(_is_countlike_or_accumulator(col)),
        "integer_count_domain": bool(_is_integer_count_domain(col)),
        "nonnegative_domain": bool(_is_nonnegative_domain(col)),
        "signed_allowed_domain": bool(_is_signed_allowed_domain(col)),
        "circular_angle_domain": bool(_is_circular_angle_domain(col)),
    }

    return out.astype(np.float32), audit

# ----------------------------------------------------------
# 5) Validate portfolio A1 eligibility
# ----------------------------------------------------------
_port = IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF.copy()

if not {"col", "generator", "eligible"}.issubset(set(_port.columns)):
    raise RuntimeError(
        "[Cell12.c.1] IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF must be normalized long-form "
        "with columns {'col','generator','eligible'}. Run fixed Cell 12.c.0 first."
    )

_port["col"] = _port["col"].astype(str)
_port["generator"] = _port["generator"].astype(str)
_port["eligible"] = _port["eligible"].astype(bool)

_A1_GENERATORS = {"A1_temporal_block_bootstrap", "A1_support_preserving_replay"}

_a1_eligible = (
    _port.loc[_port["eligible"] & _port["generator"].isin(_A1_GENERATORS)]
    .groupby("col")["generator"]
    .apply(lambda s: sorted(set(s.astype(str))))
    .to_dict()
)

_missing_a1 = [c for c in CONT_VALUE_COLS if c not in _a1_eligible]
if _missing_a1:
    msg = (
        "[Cell12.c.1] A1 is not eligible for all value columns. "
        f"missing={_missing_a1[:30]} n_missing={len(_missing_a1)}"
    )
    if bool(CFG12C1["strict_fail_on_missing_a1"]):
        raise RuntimeError(msg)
    log("[Cell12.c.1][WARN] " + msg)

log(
    "[Cell12.c.1] A1 eligibility confirmed | "
    f"value_cols={len(CONT_VALUE_COLS)} | "
    f"A1_covered={len(_a1_eligible)}"
)

# ----------------------------------------------------------
# 6) Build A1 VAL only
# ----------------------------------------------------------
log("[Cell12.c.1] Building A1 IoT VAL backbone from TRAIN donors only...")

A1_VAL_DICT = {}
AUDIT_ROWS = []

for i, col in enumerate(CONT_VALUE_COLS, start=1):
    if col not in df_tr.columns:
        raise RuntimeError(f"[Cell12.c.1] TRAIN missing value column: {col}")
    if col not in df_val.columns:
        raise RuntimeError(f"[Cell12.c.1] VAL missing value column: {col}")

    m_val = _val_mask(col)
    arr_val, audit_val = _generate_a1_val_for_col(col, N_VAL, m_val)

    A1_VAL_DICT[col] = arr_val

    fam = _family_for_col(col)
    ssf = _subfamily_for_col(col)
    entity = _infer_entity(col)

    row = {
        "col": col,
        "entity": entity,
        "family": fam,
        "synthesis_subfamily": ssf,
        "stage12c_kind": _stage_kind_for_col(col),
        "a1_generators_eligible": "|".join(_a1_eligible.get(col, [])),
        "fit_split": "TRAIN_only",
        "val_mask_policy": CFG12C1["val_mask_policy"],
        "test_materialization_policy": CFG12C1["test_materialization_policy"],

        "train_value_n": int(audit_val["train_n"]),
        "val_active_n": int(m_val.sum()),
        "val_active_rate": float(m_val.mean()),

        "val_method": audit_val["method"],

        "train_mean": audit_val.get("train_mean", np.nan),
        "train_std": audit_val.get("train_std", np.nan),
        "train_q01": audit_val.get("train_q01", np.nan),
        "train_q05": audit_val.get("train_q05", np.nan),
        "train_q50": audit_val.get("train_q50", np.nan),
        "train_q95": audit_val.get("train_q95", np.nan),
        "train_q99": audit_val.get("train_q99", np.nan),
        "train_nonzero_rate": audit_val.get("train_nonzero_rate", np.nan),

        "a1_val_mean": audit_val.get("a1_val_mean", np.nan),
        "a1_val_std": audit_val.get("a1_val_std", np.nan),
        "a1_val_q01": audit_val.get("a1_val_q01", np.nan),
        "a1_val_q05": audit_val.get("a1_val_q05", np.nan),
        "a1_val_q50": audit_val.get("a1_val_q50", np.nan),
        "a1_val_q95": audit_val.get("a1_val_q95", np.nan),
        "a1_val_q99": audit_val.get("a1_val_q99", np.nan),
        "a1_val_nonzero_rate": audit_val.get("a1_val_nonzero_rate", np.nan),

        "countlike_or_accumulator": bool(audit_val.get("countlike_or_accumulator", False)),
        "integer_count_domain": bool(audit_val.get("integer_count_domain", False)),
        "low_cardinality_train": bool(audit_val.get("low_cardinality_train", False)),
        "constant_like_train": bool(audit_val.get("constant_like_train", False)),
        "nonnegative_domain": bool(audit_val.get("nonnegative_domain", False)),
        "signed_allowed_domain": bool(audit_val.get("signed_allowed_domain", False)),
        "circular_angle_domain": bool(audit_val.get("circular_angle_domain", False)),

        "test_real_values_used_for_generation": False,
        "test_synthetic_masks_used_for_generation": False,
        "test_length_values_materialized_here": False,
        "val_values_used_for_generation": False,
        "train_values_used_for_generation": True,
        "selection_done_here": False,
    }

    AUDIT_ROWS.append(row)

    if i == 1 or i % 25 == 0 or i == len(CONT_VALUE_COLS):
        log(f"[Cell12.c.1] progress {i}/{len(CONT_VALUE_COLS)} | col={col}")

A1_IOT_VALUES_VAL = pd.DataFrame(A1_VAL_DICT, index=df_val.index, copy=False)
A1_IOT_VALUE_BACKBONE_AUDIT_DF = pd.DataFrame(AUDIT_ROWS)
# ----------------------------------------------------------
# 6B) A1 candidate labeling and publication-safety gate
# ----------------------------------------------------------
def _a1_publication_safety_label(row: dict) -> Tuple[str, str, bool, str]:
    """
    Classify A1 as a VAL candidate and identify copy-risk.

    This does not reject A1 globally. It marks whether A1 is publication-safe
    enough to win automatically in Cell 12.c.3.

    A1 should remain eligible for every column, but Cell 12.c.3 must not allow
    A1 to dominate when publication_safety == 'unsafe'.
    """
    train_n = int(row.get("train_value_n", 0) or 0)
    val_active_n = int(row.get("val_active_n", 0) or 0)

    train_std = row.get("train_std", np.nan)
    a1_std = row.get("a1_val_std", np.nan)

    train_nonzero = row.get("train_nonzero_rate", np.nan)
    a1_nonzero = row.get("a1_val_nonzero_rate", np.nan)

    low_card = bool(row.get("low_cardinality_train", False))
    const_like = bool(row.get("constant_like_train", False))
    countlike = bool(row.get("countlike_or_accumulator", False))
    method = str(row.get("val_method", ""))

    if val_active_n == 0:
        return (
            "candidate",
            "controlled",
            True,
            "no active VAL rows; publication safety deferred to columns with observations",
        )

    if train_n < int(CFG12C1["min_train_values"]):
        return (
            "candidate",
            "high",
            False,
            "low TRAIN support; A1 is a fallback candidate only, not publication-safe dominant baseline",
        )

    if const_like:
        return (
            "candidate",
            "controlled",
            True,
            "TRAIN is constant/quasi-constant; preserving constant support is acceptable",
        )

    if low_card and train_n < 32:
        return (
            "candidate",
            "high",
            False,
            "low-cardinality column with small TRAIN support; elevated memorization/copy risk",
        )

    if "train_block_bootstrap" in method and train_n < max(128, int(row.get("val_active_n", 0) or 0) // 8):
        return (
            "candidate",
            "medium",
            False,
            "block bootstrap uses limited donor pool relative to VAL active length",
        )

    if countlike and low_card:
        return (
            "candidate",
            "controlled",
            True,
            "countlike low-cardinality support projection is expected but must be compared against alternatives",
        )

    if np.isfinite(train_std) and np.isfinite(a1_std):
        if float(train_std) > 1e-8 and float(a1_std) <= 1e-8:
            return (
                "candidate",
                "high",
                False,
                "A1 collapsed variance relative to non-constant TRAIN values",
            )

    if np.isfinite(train_nonzero) and np.isfinite(a1_nonzero):
        if abs(float(train_nonzero) - float(a1_nonzero)) > 0.20:
            return (
                "candidate",
                "medium",
                False,
                "A1 nonzero-rate drift is large relative to TRAIN donor distribution",
            )

    return (
        "candidate",
        "controlled",
        True,
        "A1 copy risk controlled by TRAIN-only donor use, VAL-only scoring, support projection, and no TEST access",
    )


_a1_safety_rows = []

for _idx, _r in A1_IOT_VALUE_BACKBONE_AUDIT_DF.iterrows():
    diagnostic_or_candidate, copy_risk, publication_safe, safety_reason = _a1_publication_safety_label(
        _r.to_dict()
    )

    A1_IOT_VALUE_BACKBONE_AUDIT_DF.loc[_idx, "A1_continuous_value_baseline"] = True
    A1_IOT_VALUE_BACKBONE_AUDIT_DF.loc[_idx, "diagnostic_or_candidate"] = diagnostic_or_candidate
    A1_IOT_VALUE_BACKBONE_AUDIT_DF.loc[_idx, "copy_risk"] = copy_risk
    A1_IOT_VALUE_BACKBONE_AUDIT_DF.loc[_idx, "publication_safe_candidate"] = bool(publication_safe)
    A1_IOT_VALUE_BACKBONE_AUDIT_DF.loc[_idx, "a1_auto_domination_allowed"] = bool(publication_safe)
    A1_IOT_VALUE_BACKBONE_AUDIT_DF.loc[_idx, "a1_safety_reason"] = safety_reason

    _a1_safety_rows.append({
        "col": str(_r["col"]),
        "A1_continuous_value_baseline": True,
        "diagnostic_or_candidate": diagnostic_or_candidate,
        "copy_risk": copy_risk,
        "publication_safe_candidate": bool(publication_safe),
        "a1_auto_domination_allowed": bool(publication_safe),
        "a1_safety_reason": safety_reason,
    })

A1_PUBLICATION_SAFETY_DF = pd.DataFrame(_a1_safety_rows)

A1_PUBLICATION_SAFETY_CSV = os.path.join(
    REPORT_DIR,
    "cell12c1_a1_publication_safety_gate.csv",
)
A1_PUBLICATION_SAFETY_DF.to_csv(A1_PUBLICATION_SAFETY_CSV, index=False)

_a1_unsafe_cols = A1_PUBLICATION_SAFETY_DF.loc[
    ~A1_PUBLICATION_SAFETY_DF["publication_safe_candidate"].astype(bool),
    "col",
].astype(str).tolist()

_a1_copy_risk_counts = (
    A1_PUBLICATION_SAFETY_DF["copy_risk"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

globals()["A1_PUBLICATION_SAFETY_DF"] = A1_PUBLICATION_SAFETY_DF
globals()["A1_PUBLICATION_SAFETY_CSV"] = A1_PUBLICATION_SAFETY_CSV
globals()["A1_UNSAFE_AUTO_DOMINATION_COLS"] = _a1_unsafe_cols

log(
    "[Cell12.c.1] A1 publication-safety gate complete | "
    f"A1_continuous_value_baseline=True | "
    f"diagnostic_or_candidate=candidate | "
    f"copy_risk_counts={_a1_copy_risk_counts} | "
    f"unsafe_auto_domination_cols={len(_a1_unsafe_cols)} | "
    f"csv={A1_PUBLICATION_SAFETY_CSV}"
)
# ----------------------------------------------------------
# 7) Contract validation
# ----------------------------------------------------------
def _validate_a1_val_frame(df_out: pd.DataFrame):
    if not isinstance(df_out, pd.DataFrame):
        raise RuntimeError("[Cell12.c.1] A1_IOT_VALUES_VAL is not a DataFrame.")
    if len(df_out) != N_VAL:
        raise RuntimeError(f"[Cell12.c.1] A1_IOT_VALUES_VAL row mismatch: {len(df_out)} vs {N_VAL}")
    if list(df_out.columns) != list(CONT_VALUE_COLS):
        raise RuntimeError("[Cell12.c.1] A1_IOT_VALUES_VAL column order/schema mismatch vs CONT_VALUE_COLS.")

    bad_rows = []

    for col in CONT_VALUE_COLS:
        x = pd.to_numeric(df_out[col], errors="coerce").to_numpy(dtype=np.float64)
        active = _val_mask(col)

        if active.shape[0] != N_VAL:
            raise RuntimeError(f"[Cell12.c.1] VAL mask length mismatch for {col}.")

        inactive_finite = int(np.isfinite(x[~active]).sum())
        active_finite = int(np.isfinite(x[active]).sum())

        if inactive_finite > 0:
            bad_rows.append({
                "col": col,
                "error": "inactive_finite_values",
                "inactive_finite": inactive_finite,
            })

        if int(active.sum()) > 0 and active_finite == 0:
            bad_rows.append({
                "col": col,
                "error": "no_active_values",
                "active_n": int(active.sum()),
            })

        if active_finite > 0:
            finite_x = x[active][np.isfinite(x[active])]

            if _is_nonnegative_domain(col):
                min_val = float(np.min(finite_x))
                if min_val < -1e-6:
                    bad_rows.append({
                        "col": col,
                        "error": "negative_values_in_nonnegative_domain",
                        "min_val": min_val,
                    })

            if _is_circular_angle_domain(col):
                min_val = float(np.min(finite_x))
                max_val = float(np.max(finite_x))
                if min_val < -1e-6 or max_val >= 360.0 + 1e-6:
                    bad_rows.append({
                        "col": col,
                        "error": "circular_angle_out_of_range",
                        "min_val": min_val,
                        "max_val": max_val,
                    })

            if _is_integer_count_domain(col):
                non_integer_frac = float(
                    np.mean(np.abs(finite_x - np.rint(finite_x)) > float(CFG12C1["count_integer_tol"]))
                )
                if non_integer_frac > 0.001:
                    bad_rows.append({
                        "col": col,
                        "error": "countlike_noninteger",
                        "non_integer_frac": non_integer_frac,
                    })

    if bad_rows:
        bad_path = os.path.join(REPORT_DIR, "cell12c1_a1_val_contract_failures.csv")
        pd.DataFrame(bad_rows).to_csv(bad_path, index=False)
        raise RuntimeError(
            "[Cell12.c.1] A1 VAL contract validation failed. "
            f"failures={len(bad_rows)} | details={bad_path}"
        )

_validate_a1_val_frame(A1_IOT_VALUES_VAL)

log(
    "[Cell12.c.1] A1 VAL contract validation passed | "
    f"VAL={A1_IOT_VALUES_VAL.shape}"
)

# ----------------------------------------------------------
# 8) Save artifacts
# ----------------------------------------------------------
p_a1_val = os.path.join(OUT_SYN, "A1_IOT_VALUES_VAL.parquet")

# Deliberately no A1_IOT_VALUES_TEST output in v2.0.
p_audit = os.path.join(REPORT_DIR, "cell12c1_a1_iot_value_backbone_audit.csv")
p_contract = os.path.join(REPORT_DIR, "cell12c1_a1_iot_value_backbone_contract.json")
p_contract_canonical = os.path.join(CONTRACT_DIR, "cell12c1_a1_iot_value_backbone_contract_v2_1_THESIS.json")
p_manifest = os.path.join(ARTDIR, "cell12c1_a1_iot_value_backbone_manifest.json")

A1_IOT_VALUES_VAL.to_parquet(p_a1_val, index=False)
A1_IOT_VALUE_BACKBONE_AUDIT_DF.to_csv(p_audit, index=False)

# ----------------------------------------------------------
# 9) Summaries / manifest
# ----------------------------------------------------------
_method_counts_val = (
    A1_IOT_VALUE_BACKBONE_AUDIT_DF["val_method"]
    .value_counts()
    .sort_index()
    .to_dict()
)

_family_counts = (
    A1_IOT_VALUE_BACKBONE_AUDIT_DF["family"]
    .value_counts()
    .sort_index()
    .to_dict()
)

_zero_active_val_cols = A1_IOT_VALUE_BACKBONE_AUDIT_DF.loc[
    pd.to_numeric(A1_IOT_VALUE_BACKBONE_AUDIT_DF["val_active_n"], errors="coerce").fillna(0).astype(int) == 0,
    "col"
].astype(str).tolist()

_low_support_cols = A1_IOT_VALUE_BACKBONE_AUDIT_DF.loc[
    pd.to_numeric(A1_IOT_VALUE_BACKBONE_AUDIT_DF["train_value_n"], errors="coerce").fillna(0).astype(int)
    < int(CFG12C1["min_train_values"]),
    "col"
].astype(str).tolist()

contract = {
    "version": CFG12C1["version"],
    "role": "mandatory_A1_VAL_iot_value_backbone",
    "upstream_contracts": {
        "cell12c0_foundation_contract": CELL12C0_CONTRACT_PATH_REQUIRED,
        "cell12b_scaffold_contract": CELL12B_SCAFFOLD_CONTRACT_PATH_REQUIRED,
    },
    "fit_split": "TRAIN_only",
    "val_artifact_role": "VAL_candidate_baseline_for_later_selector",
    "test_artifact_role": "not_materialized_here_deferred_to_Cell12c4",
    "test_real_values_used_for_generation": False,
    "test_synthetic_masks_used_for_generation": False,
    "test_length_values_materialized_here": False,
    "val_values_used_for_generation": False,
    "train_values_used_for_generation": True,
    "selection_done_here": False,
    "N_TR": int(N_TR),
    "N_VAL": int(N_VAL),
    "value_cols_n": int(len(CONT_VALUE_COLS)),
    "method_counts_val": _method_counts_val,
    "family_counts": _family_counts,
    "zero_active_val_cols_n": int(len(_zero_active_val_cols)),
    "zero_active_val_cols": _zero_active_val_cols,
    "low_train_support_cols_n": int(len(_low_support_cols)),
    "low_train_support_cols": _low_support_cols,
    "strict_contracts": {
        "inactive_val_rows_are_nan": True,
        "active_val_rows_have_values_when_active_exists": True,
        "nonnegative_domains_enforced": True,
        "signed_domains_preserved": True,
        "circular_angle_domains_wrapped": True,
        "countlike_values_rounded": True,
        "low_cardinality_support_projected": bool(CFG12C1["preserve_train_support_for_low_cardinality"]),
    },
    "downstream_contract": {
        "Cell12c2": "may use A1_IOT_VALUES_VAL as mandatory baseline candidate",
        "Cell12c3": "must select using VAL metrics only",
        "Cell12c4": "must materialize TEST-length selected choices after VAL lock",
    },
    "A1_continuous_value_baseline": True,
    "diagnostic_or_candidate": "candidate",
    "copy_risk": "controlled",
    "copy_risk_policy": {
        "default_copy_risk": "controlled",
        "unsafe_auto_domination_cols_n": int(len(_a1_unsafe_cols)),
        "unsafe_auto_domination_cols": _a1_unsafe_cols,
        "copy_risk_counts": _a1_copy_risk_counts,
        "publication_safety_csv": A1_PUBLICATION_SAFETY_CSV,
        "cell12c3_requirement": (
            "Cell 12.c.3 must not let A1 automatically dominate for columns where "
            "a1_auto_domination_allowed=False. A1 may still win only if VAL metrics and "
            "publication-safety checks justify it."
        ),
    },
}

_write_json(p_contract, contract)
_write_json(p_contract_canonical, contract)

manifest = {
    "version": CFG12C1["version"],
    "cell": "12.c.1",
    "created_artifacts": {
        "A1_IOT_VALUES_VAL": p_a1_val,
        "a1_backbone_audit": p_audit,
        "a1_backbone_contract": p_contract,
        "a1_backbone_contract_canonical": p_contract_canonical,
        "a1_publication_safety_gate": A1_PUBLICATION_SAFETY_CSV,
    },
    "intentionally_not_created": {
        "A1_IOT_VALUES_TEST": "deferred_to_Cell12c4_after_VAL_locked_selection",
    },
    "artifact_hashes": {
        "A1_IOT_VALUES_VAL_sha256": _sha256_file(p_a1_val),
        "a1_backbone_audit_sha256": _sha256_file(p_audit),
        "a1_backbone_contract_sha256": _sha256_file(p_contract),
        "a1_backbone_contract_canonical_sha256": _sha256_file(p_contract_canonical),
        "a1_publication_safety_gate_sha256": _sha256_file(A1_PUBLICATION_SAFETY_CSV),
    },
    "contract": contract,
    "next_cells": {
        "12.c.2": "VAL candidates for all eligible non-A1 generators, including DDPM_conditional_sequence and TimeGAN_sequence",
        "12.c.3": "VAL-only portfolio selection",
        "12.c.4": "TEST-length materialization from locked choices",
        "12.c.5": "final contract/mask enforcement",
        "12.c.6": "final TEST QA and publication artifacts",
    },
}

_write_json(p_manifest, manifest)

A1_IOT_VALUE_BACKBONE_MANIFEST = manifest

# ----------------------------------------------------------
# 10) Publish globals
# ----------------------------------------------------------
globals()["A1_IOT_VALUES_VAL"] = A1_IOT_VALUES_VAL
globals()["A1_IOT_VALUE_BACKBONE_AUDIT_DF"] = A1_IOT_VALUE_BACKBONE_AUDIT_DF
globals()["A1_IOT_VALUE_BACKBONE_CONTRACT"] = contract
globals()["A1_IOT_VALUE_BACKBONE_MANIFEST"] = A1_IOT_VALUE_BACKBONE_MANIFEST
globals()["A1_IOT_VALUES_VAL_PATH"] = p_a1_val
globals()["A1_IOT_VALUE_BACKBONE_AUDIT_CSV"] = p_audit
globals()["CELL12C1_A1_BACKBONE_MANIFEST_PATH"] = p_manifest
globals()["A1_IOT_VALUE_BACKBONE_CONTRACT_PATH"] = p_contract_canonical

# Remove stale same-name TEST artifact globals from older runs if present.
for _stale in [
    "A1_IOT_VALUES_TEST",
    "A1_IOT_VALUES_TEST_PATH",
]:
    if _stale in globals():
        del globals()[_stale]
        log(f"[Cell12.c.1][WARN] Removed stale global from older architecture: {_stale}")

log(f"[Cell12.c.1] Saved A1 VAL artifact: {p_a1_val} | shape={A1_IOT_VALUES_VAL.shape}")
log(f"[Cell12.c.1] Saved A1 audit: {p_audit}")
log(f"[Cell12.c.1] Saved A1 contract: {p_contract}")
log(f"[Cell12.c.1] Saved canonical A1 contract: {p_contract_canonical}")
log(f"[Cell12.c.1] Saved A1 manifest: {p_manifest}")



log(
    "[Cell12.c.1] Final summary | "
    "A1_continuous_value_baseline=True | "
    "diagnostic_or_candidate=candidate | "
    f"copy_risk_counts={_a1_copy_risk_counts} | "
    f"unsafe_auto_domination_cols={len(_a1_unsafe_cols)} | "
    f"value_cols={len(CONT_VALUE_COLS)} | "
    f"method_counts={_method_counts_val} | "
    f"zero_active_val_cols={len(_zero_active_val_cols)} | "
    f"low_train_support_cols={len(_low_support_cols)} | "
    "TEST_materialized=False"
)

log("--- END: Cell 12.c.1 — IoT value A1 VAL backbone materialization (v2.1-THESIS contract-locked modular STUDY-THESIS) ---")

gc.collect()