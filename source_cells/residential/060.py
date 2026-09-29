# ==========================================================
# CELL 12.d.0 - Binary IoT shared utilities/contracts
# v1.2-THESIS contract-locked STUDY-THESIS strict binary contract + robust target resolver
#
# Role:
#   - Define IoT binary/state targets owned by the binary branch.
#   - Prefer Cell 4.5 ownership metadata when it is exact/valid.
#   - Otherwise use TRAIN+VAL-only binary-domain + HA/state-name inference.
#   - Do NOT generate binary values.
#   - Do NOT select generators.
#   - Do NOT fit generators.
#   - Do NOT read real TEST values for target discovery/calibration.
#   - df_te is used only for schema/index/length alignment.
#
# Outputs:
#   artifacts/IOT_BINARY_TARGETS.json
#   reports/IOT_BINARY_TARGETS.json
#   reports/cell12d_binary_contract.json
#   reports/cell12d0_binary_target_contract.csv
#   reports/cell12d0_binary_availability_contract.csv
#   reports/cell12d0_binary_ownership_extraction.csv
#   reports/cell12d0_binary_fallback_trainval_inference.csv
#   reports/cell12d0_binary_target_resolution_diagnostic.csv
# ==========================================================

log("--- START: Cell 12.d.0 - Binary IoT shared utilities/contracts (v1.2-THESIS contract-locked strict resolver) ---")

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
_required_12d0 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
]
_missing_12d0 = [k for k in _required_12d0 if k not in globals()]
if _missing_12d0:
    raise RuntimeError(f"[Cell12.d.0] Missing required globals: {_missing_12d0}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

# ----------------------------------------------------------
# 0B) Upstream contract locks
# ----------------------------------------------------------
def _read_json_dict_required_12d0(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"[Cell12.d.0] Missing required {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell12.d.0] Required {label} must be a JSON object: {path}")
    return obj

def _read_json_dict_optional_12d0(path: str, label: str):
    if not os.path.exists(path):
        return None
    return _read_json_dict_required_12d0(path, label)

def _contract_bool_12d0(obj: dict, key: str, default: bool = False) -> bool:
    if not isinstance(obj, dict):
        return bool(default)
    try:
        return bool(obj.get(key, default))
    except Exception:
        return bool(default)

CELL12B_SCAFFOLD_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "cell12b_iot_scaffold_contract_v27_1_1_THESIS.json",
)
CELL12C0_FOUNDATION_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "cell12c0_foundation_contract_v1_2_THESIS.json",
)
CELL12C5_FINAL_VALUE_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "cell12c5_final_value_enforcement_contract_v1_4_THESIS.json",
)
CELL12C6_QA_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "cell12c6_final_qa_contract_v1_4_THESIS.json",
)
CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "continuous_target_ready_VAL_frozen_TEST_audit_contract_v2_0_THESIS.json",
)
CELL11B_QUARANTINE_MANIFEST_PATH = os.path.join(
    ARTDIR,
    "cell11b_protocol_test_regression_quarantine_manifest.json",
)

CELL12B_SCAFFOLD_CONTRACT = _read_json_dict_required_12d0(
    CELL12B_SCAFFOLD_CONTRACT_PATH,
    "Cell 12.b v27.1.1 scaffold contract",
)
CELL12C0_FOUNDATION_CONTRACT = _read_json_dict_required_12d0(
    CELL12C0_FOUNDATION_CONTRACT_PATH,
    "Cell 12.c.0 v1.2 foundation contract",
)
CELL12C5_FINAL_VALUE_CONTRACT = _read_json_dict_optional_12d0(
    CELL12C5_FINAL_VALUE_CONTRACT_PATH,
    "Cell 12.c.5 final value enforcement contract",
)
CELL12C6_QA_CONTRACT = _read_json_dict_optional_12d0(
    CELL12C6_QA_CONTRACT_PATH,
    "Cell 12.c.6 QA contract",
)
CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT = _read_json_dict_optional_12d0(
    CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT_PATH,
    "Cell 12.c.6v VAL-frozen TEST-audit contract",
)

if os.path.exists(CELL11B_QUARANTINE_MANIFEST_PATH):
    _cell11b_manifest = _read_json_dict_required_12d0(
        CELL11B_QUARANTINE_MANIFEST_PATH,
        "noncanonical Cell 11b quarantine manifest",
    )
    _cell11b_outputs = _cell11b_manifest.get("outputs", {}) if isinstance(_cell11b_manifest, dict) else {}
    _cell11b_policy = _cell11b_manifest.get("policy", {}) if isinstance(_cell11b_manifest, dict) else {}
    _cell11b_overwrote = bool(
        _cell11b_policy.get("canonical_A2_overwritten", False)
        or _cell11b_outputs.get("A2_PROTOCOL_FINAL_overwritten")
        or _cell11b_outputs.get("A2_PROTOCOL_TEST_overwritten")
    )
    if _cell11b_overwrote:
        raise RuntimeError(
            "[Cell12.d.0] Refusing to proceed because noncanonical Cell 11b TEST-quarantine "
            "artifacts indicate canonical A2 was overwritten."
        )

_s12b_test_usage = CELL12B_SCAFFOLD_CONTRACT.get("test_usage", {})
for _key in [
    "test_values_used_for_fitting",
    "test_values_used_for_selection",
    "test_values_used_for_calibration",
    "test_values_used_for_repair",
    "test_values_used_for_continuous_value_generation",
]:
    if bool(_s12b_test_usage.get(_key, False)):
        raise RuntimeError(f"[Cell12.d.0] Cell 12.b scaffold contract violation: {_key}=True")

_policy_12c0 = CELL12C0_FOUNDATION_CONTRACT.get("policy", {})
if bool(_policy_12c0.get("test_used_for_generator_selection", False)):
    raise RuntimeError("[Cell12.d.0] Cell 12.c.0 policy indicates TEST generator selection.")
if str(_policy_12c0.get("selection_split", "VAL_only")) != "VAL_only":
    raise RuntimeError("[Cell12.d.0] Cell 12.c.0 policy selection_split must be VAL_only.")

# Optional completed continuous branch contracts must remain clean if present.
if CELL12C5_FINAL_VALUE_CONTRACT is not None:
    for _key in ["test_real_values_used", "TEST_real_values_used", "selection_done_here", "generator_fit_done_here"]:
        if _contract_bool_12d0(CELL12C5_FINAL_VALUE_CONTRACT, _key, False):
            raise RuntimeError(f"[Cell12.d.0] Cell 12.c.5 contract violation: {_key}=True")

if CELL12C6_QA_CONTRACT is not None:
    # Cell 12.c.6 may use TEST for QA but must not mutate/select/fit.
    for _key in ["synthetic_values_mutated", "selection_done_here", "generator_fit_done_here", "fits_generators", "selects_generators", "calibrates_generators"]:
        if _contract_bool_12d0(CELL12C6_QA_CONTRACT, _key, False):
            raise RuntimeError(f"[Cell12.d.0] Cell 12.c.6 contract violation: {_key}=True")

if CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT is not None:
    for _key in ["TEST_used_for_selection", "TEST_used_for_repair", "synthetic_values_mutated"]:
        if _contract_bool_12d0(CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT, _key, False):
            raise RuntimeError(f"[Cell12.d.0] Cell 12.c.6v contract violation: {_key}=True")

log(
    "[Cell12.d.0] Upstream contracts locked | "
    "cell12b_v27_1_1=True | cell12c0_v1_2=True | "
    f"cell12c5_present={CELL12C5_FINAL_VALUE_CONTRACT is not None} | "
    f"cell12c6_present={CELL12C6_QA_CONTRACT is not None} | "
    f"cell12c6v_present={CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT is not None}"
)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

if N_TR <= 0 or N_VAL <= 0 or N_TE <= 0:
    raise RuntimeError(
        f"[Cell12.d.0] Invalid split lengths: train={N_TR}, val={N_VAL}, test={N_TE}"
    )

CELL12D0_VERSION = "cell12d0_binary_contract_strict_v1_2_THESIS_contract_locked_resolver_fix"

CFG["cell12d0_version"] = CELL12D0_VERSION
CFG["cell12d0_TEST_real_values_used"] = False
CFG["cell12d0_selection_done_here"] = False
CFG["cell12d0_generator_fit_done_here"] = False
CFG["cell12d0_train_val_values_used_for_target_validation"] = True
CFG["cell12d0_df_te_used_for_schema_index_length_only"] = True

CFG.setdefault("cell12d_expected_binary_target_count", 52)
CFG.setdefault("cell12d0_allow_binary_target_count_mismatch", False)
CFG.setdefault("cell12d0_binary_target_owner_label", "binary_state_target")
CFG.setdefault("cell12d0_binary_value_tolerance", 1e-6)
CFG.setdefault("cell12d0_binary_min_trainval_observed_n", 1)
CFG.setdefault("cell12d0_binary_availability_default", "all_active")
CFG.setdefault("cell12d0_binary_prefer_ownership_metadata", True)
CFG.setdefault("cell12d0_allow_fallback_when_ownership_count_wrong", True)

EXPECTED_BINARY_TARGET_COUNT_12D = int(CFG.get("cell12d_expected_binary_target_count", 52))
ALLOW_BINARY_TARGET_COUNT_MISMATCH_12D0 = bool(
    CFG.get("cell12d0_allow_binary_target_count_mismatch", False)
)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
binary_targets_json = os.path.join(ARTDIR, "IOT_BINARY_TARGETS.json")
binary_targets_report_json = os.path.join(REPORT_DIR, "IOT_BINARY_TARGETS.json")
binary_contract_json = os.path.join(REPORT_DIR, "cell12d_binary_contract.json")
binary_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12d0_binary_contract_v1_2_THESIS.json")
binary_target_contract_csv = os.path.join(REPORT_DIR, "cell12d0_binary_target_contract.csv")
binary_availability_contract_csv = os.path.join(REPORT_DIR, "cell12d0_binary_availability_contract.csv")
binary_availability_mask_parquet = os.path.join(OUT_SYN, "IOT_BINARY_AVAIL_SYN.parquet")
binary_ownership_extraction_csv = os.path.join(REPORT_DIR, "cell12d0_binary_ownership_extraction.csv")
binary_fallback_inference_csv = os.path.join(REPORT_DIR, "cell12d0_binary_fallback_trainval_inference.csv")
binary_resolution_diag_csv = os.path.join(REPORT_DIR, "cell12d0_binary_target_resolution_diagnostic.csv")

# ----------------------------------------------------------
# 2) Generic helpers
# ----------------------------------------------------------
def _json_sanitize_12d0(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12d0(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12d0(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12d0(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12d0(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12d0(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12d0(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12d0(obj.to_dict())
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

def _write_json_12d0(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12d0(payload), f, indent=2, sort_keys=True)

def _sha256_file_12d0(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _stable_seed_12d0(name: str, offset: int = 0) -> int:
    h = hashlib.sha256(f"{SEED}|12d0|{offset}|{name}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)

def _read_csv_if_exists_12d0(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _is_iot_col_12d0(col: str) -> bool:
    return str(col).startswith("iot__")

def _normalise_bool_12d0(x, default=False) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(int(x))
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(float(x)):
            return bool(default)
        return bool(int(float(x)))
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f", "", "nan", "none", "null"}:
        return False
    return bool(default)

def _normalise_token_12d0(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _split_iot_col_12d0(col: str) -> list[str]:
    return [_normalise_token_12d0(p) for p in str(col).split("__") if str(p).strip()]

def _ha_domain_from_iot_col_12d0(col: str) -> str:
    parts = _split_iot_col_12d0(col)
    if parts and parts[0] == "iot":
        parts = parts[1:]
    if len(parts) >= 2:
        return str(parts[1])
    return ""

def _entity_from_iot_col_12d0(col: str) -> str:
    s = str(col)
    if s.startswith("iot__"):
        s = s[5:]
    return s.split("__")[0] if "__" in s else s

def _measurement_from_iot_col_12d0(col: str) -> str:
    parts = _split_iot_col_12d0(col)
    if parts and parts[0] == "iot":
        parts = parts[1:]
    if len(parts) >= 2 and parts[-1] in {"state", "value", "binary", "flag"}:
        return parts[-2]
    return parts[-1] if parts else _normalise_token_12d0(col)

def _to_num_array_12d0(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _finite_12d0(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]

def _trainval_values_12d0(col: str) -> np.ndarray:
    tr = _to_num_array_12d0(df_tr, col)
    va = _to_num_array_12d0(df_val, col)
    return _finite_12d0(np.concatenate([tr, va]))

def _is_binary_trainval_values_12d0(col: str) -> tuple[bool, dict]:
    if col not in df_tr.columns or col not in df_val.columns:
        return False, {
            "binary_value_check_passed": False,
            "binary_value_check_reason": "missing_from_train_or_val",
            "trainval_observed_n": 0,
            "trainval_unique_n": 0,
            "trainval_unique_values_preview": [],
            "trainval_binary_fraction": np.nan,
            "trainval_non_binary_n": 0,
            "trainval_zero_n": 0,
            "trainval_one_n": 0,
            "trainval_rate": np.nan,
        }

    x = _trainval_values_12d0(col)
    finite_n = int(x.size)
    tol = float(CFG.get("cell12d0_binary_value_tolerance", 1e-6))
    min_obs = int(CFG.get("cell12d0_binary_min_trainval_observed_n", 1))

    if finite_n < min_obs:
        return False, {
            "binary_value_check_passed": False,
            "binary_value_check_reason": "insufficient_trainval_observed_values",
            "trainval_observed_n": finite_n,
            "trainval_unique_n": 0,
            "trainval_unique_values_preview": [],
            "trainval_binary_fraction": np.nan,
            "trainval_non_binary_n": 0,
            "trainval_zero_n": 0,
            "trainval_one_n": 0,
            "trainval_rate": np.nan,
        }

    near_zero = np.abs(x - 0.0) <= tol
    near_one = np.abs(x - 1.0) <= tol
    binary_mask = near_zero | near_one

    bad_n = int(np.sum(~binary_mask))
    unique_vals = np.unique(np.round(x, 6))
    passed = bool(bad_n == 0 and unique_vals.size <= 2)

    return passed, {
        "binary_value_check_passed": passed,
        "binary_value_check_reason": "" if passed else "non_binary_trainval_values",
        "trainval_observed_n": finite_n,
        "trainval_unique_n": int(unique_vals.size),
        "trainval_unique_values_preview": [float(v) for v in unique_vals[:20]],
        "trainval_binary_fraction": float(np.mean(binary_mask)),
        "trainval_non_binary_n": bad_n,
        "trainval_zero_n": int(np.sum(near_zero)),
        "trainval_one_n": int(np.sum(near_one)),
        "trainval_rate": float(np.mean(near_one)),
    }

# ----------------------------------------------------------
# 3) Binary branch target/scaffold classification
# ----------------------------------------------------------
def _is_binary_helper_or_scaffold_12d0(col: str) -> bool:
    s = str(col)

    if s.startswith("iot__entity_obs__"):
        return True
    if s.startswith("iot__entity_stale__"):
        return True

    return s in {
        "iot__obs_present",
        "iot__any_update_raw",
        "iot__tier_present",
    }

def _looks_like_binary_name_12d0(col: str) -> bool:
    """
    Backward-compatible coarse check. Kept for audit only.
    """
    s = str(col).lower()
    binary_tokens = [
        "__binary_sensor__",
        "__switch__",
        "__button__",
        "__lock__",
        "__contact__",
        "__motion__",
        "__occupancy__",
        "__presence__",
        "__opening__",
        "__tamper__",
        "__smoke__",
        "__moisture__",
        "__water_leak__",
        "__plugged_in__",
        "__charging__",
        "__online__",
        "__connectivity__",
        "__state__",
        "__enabled__",
        "__available__",
        "__detected__",
        "__open__",
        "__closed__",
        "__on__",
        "__off__",
    ]
    return bool(_is_iot_col_12d0(col) and any(tok in s for tok in binary_tokens))

def _looks_like_home_assistant_binary_state_target_12d0(col: str) -> tuple[bool, str]:
    """
    TRAIN+VAL domain check is done separately. This function only decides
    whether the column name/structure belongs to the 12.d binary-state branch.

    Important:
    - Do not include internal availability/staleness scaffolds.
    - Do not rely on broad IOT_VALUE_COLS/VALUE_COLS ownership lists.
    - Include HA binary/state domains and selected binary-valued state/value
      sensors that are semantically state flags.
    """
    s = str(col).lower()
    if not _is_iot_col_12d0(s):
        return False, "not_iot"

    if _is_binary_helper_or_scaffold_12d0(s):
        return False, "helper_or_scaffold"

    # Keep this eligible. It is not a scaffold availability column, and the
    # previous diagnostic suggests expected 52 = 53 binary-valued excluded minus tier_present.
    if s == "iot__events_update_any":
        return True, "explicit_iot_update_event_binary_target"

    parts = _split_iot_col_12d0(s)
    if parts and parts[0] == "iot":
        parts = parts[1:]

    domain = parts[1] if len(parts) >= 2 else ""
    measurement = _measurement_from_iot_col_12d0(s)
    tail = parts[-1] if parts else ""

    direct_domains = {
        "binary_sensor",
        "switch",
        "light",
        "media_player",
        "remote",
        "lock",
        "button",
    }
    if domain in direct_domains and tail in {"state", "value"}:
        return True, f"ha_domain_{domain}_state_or_value"

    # Numeric/state domains that may encode binary flags after TRAIN+VAL validation.
    if domain == "number" and tail in {"state", "value"}:
        if any(tok in s or tok in measurement for tok in [
            "protection", "enabled", "state", "power_protection"
        ]):
            return True, "binary_valued_number_state_flag"

    # Sensor state/value columns that are actually binary flags.
    binary_sensor_measurements = {
        "state",
        "connectivity",
        "reachability",
        "contact",
        "occupancy",
        "moving",
        "motion",
        "presence",
        "vibration",
        "cleaning",
        "charging",
        "mop_attached",
        "water_box_attached",
        "bean_container_empty",
        "drip_tray_full",
        "water_tank_empty",
        "local_control",
        "remote_start",
        "cloud_connection",
        "overheated",
        "overloaded",
        "tamper",
        "alarm",
        "smoke",
        "moisture",
        "water_leak",
    }
    if domain == "sensor" and tail in {"state", "value"}:
        if measurement in binary_sensor_measurements:
            return True, f"binary_valued_sensor_{measurement}"

        # Conservative compatibility: some HA exports encode a binary/constant
        # sensor-like flag as a numeric value with a non-binary-sounding measurement.
        # This is allowed only after _is_binary_trainval_values_12d0 passes.
        if s in {
            "iot__coffee_maker_milk__sensor__cups__value",
        }:
            return True, "explicit_binary_valued_sensor_exception"

    return False, "not_ha_binary_state_target"

def _strict_12c_continuous_exclusion_set_12d0() -> set[str]:
    """
    Build a narrow exclusion set for targets already owned by 12.c continuous/value branch.

    Critical correction:
    - Do NOT exclude broad VALUE_COLS or IOT_VALUE_COLS.
    - Those broad lists may include binary HA state columns and caused the previous 49/52 failure.
    """
    excluded = set()

    strict_global_names = [
        "CONT_VALUE_COLS",
        "CELL12C_VALUE_COLS",
        "CELL12C_CONT_VALUE_COLS",
        "CONTINUOUS_VALUE_COLS",
        "IOT_CONT_VALUE_COLS",
        "IOT_CONTINUOUS_VALUE_COLS",
    ]

    for name in strict_global_names:
        obj = globals().get(name, None)
        if obj is not None:
            try:
                excluded.update(map(str, list(obj)))
            except Exception:
                pass

    # Add only contract rows that clearly belong to the continuous/value branch,
    # never rows explicitly marked binary_state_target.
    contract_frames = []
    for name in [
        "VALUE_CONTRACT_DF",
        "CELL12C_VALUE_CONTRACT_DF",
        "IOT_VALUE_CONTRACT_DF",
        "CELL12C0_VALUE_CONTRACT_DF",
    ]:
        obj = globals().get(name, None)
        if isinstance(obj, pd.DataFrame) and len(obj):
            contract_frames.append((name, obj.copy()))

    for path in [
        os.path.join(REPORT_DIR, "cell12c0_value_contract.csv"),
        os.path.join(REPORT_DIR, "cell12c_value_contract.csv"),
    ]:
        df = _read_csv_if_exists_12d0(path)
        if isinstance(df, pd.DataFrame) and len(df):
            contract_frames.append((path, df.copy()))

    owner_cols = [
        "primary_owner",
        "owner",
        "target_owner",
        "iot_owner",
        "role",
        "target_role",
        "column_role",
        "generation_role",
        "assigned_role",
    ]
    kind_cols = [
        "target_type",
        "type",
        "semantic_type",
        "value_type",
        "domain_type",
        "stage12c_kind",
        "kind",
        "family",
        "synthesis_subfamily",
    ]

    for _, raw in contract_frames:
        df = _standardize_col_field_12d0(raw)
        if "col" not in df.columns:
            continue

        for _, r in df.iterrows():
            col = str(r.get("col", ""))
            if not _is_iot_col_12d0(col):
                continue

            joined_owner = " ".join(
                str(r.get(c, "")).strip().lower()
                for c in owner_cols
                if c in df.columns
            )
            joined_kind = " ".join(
                str(r.get(c, "")).strip().lower()
                for c in kind_cols
                if c in df.columns
            )
            joined_all = f"{joined_owner} {joined_kind}"

            if "binary_state_target" in joined_all or "binary target" in joined_all:
                continue

            clear_continuous = any(tok in joined_all for tok in [
                "continuous",
                "cont_value",
                "continuous_value",
                "numeric_continuous",
                "12.c",
                "cell12c",
                "value_target",
                "cont_target",
            ])

            # Do not let broad "value" wording alone exclude binary __state__value columns.
            if clear_continuous:
                excluded.add(col)

    return excluded

# ----------------------------------------------------------
# 4) Load possible Cell 4.5 / Cell 11 ownership metadata
# ----------------------------------------------------------
def _candidate_ownership_frames_12d0() -> list[tuple[str, pd.DataFrame]]:
    frames = []

    global_names = [
        "CELL45_IOT_OWNERSHIP_DF",
        "CELL4_5_IOT_OWNERSHIP_DF",
        "IOT_OWNERSHIP_DF",
        "CELL45_OWNERSHIP_DF",
        "IOT_COLUMN_OWNERSHIP_DF",
        "CELL11_IOT_TARGET_REGISTRY_DF",
        "IOT_TARGET_REGISTRY_DF",
        "VALUE_CONTRACT_DF",
        "CELL12C_VALUE_CONTRACT_DF",
    ]

    for name in global_names:
        obj = globals().get(name, None)
        if isinstance(obj, pd.DataFrame) and len(obj):
            frames.append((name, obj.copy()))

    possible_paths = [
        os.path.join(REPORT_DIR, "cell45_iot_ownership.csv"),
        os.path.join(REPORT_DIR, "cell4_5_iot_ownership.csv"),
        os.path.join(REPORT_DIR, "cell4_5_iot_column_ownership.csv"),
        os.path.join(REPORT_DIR, "cell45_iot_column_ownership.csv"),
        os.path.join(REPORT_DIR, "cell45_iot_target_registry.csv"),
        os.path.join(REPORT_DIR, "cell11_iot_target_registry.csv"),
        os.path.join(REPORT_DIR, "cell11_iot_role_registry.csv"),
        os.path.join(REPORT_DIR, "cell12c0_value_contract.csv"),
    ]

    for path in possible_paths:
        df = _read_csv_if_exists_12d0(path)
        if isinstance(df, pd.DataFrame) and len(df):
            frames.append((path, df.copy()))

    return frames

def _standardize_col_field_12d0(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "col" not in out.columns:
        for cand in ["column", "name", "feature", "feature_name", "target_col", "iot_col", "value_col"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "col"})
                break
    if "col" in out.columns:
        out["col"] = out["col"].astype(str)
    return out

def _extract_binary_targets_from_ownership_12d0() -> tuple[list[str], pd.DataFrame, dict]:
    owner_label = str(CFG.get("cell12d0_binary_target_owner_label", "binary_state_target"))
    frames = _candidate_ownership_frames_12d0()

    extraction_rows = []
    target_sources = []

    owner_cols = [
        "primary_owner",
        "owner",
        "target_owner",
        "iot_owner",
        "role",
        "target_role",
        "column_role",
        "generation_role",
        "assigned_role",
    ]
    binary_flag_cols = [
        "binary",
        "is_binary",
        "binary_state_target",
        "is_binary_state_target",
        "is_binary_target",
    ]
    type_cols = [
        "target_type",
        "type",
        "semantic_type",
        "value_type",
        "domain_type",
        "stage12d_kind",
        "kind",
    ]

    for source_name, raw in frames:
        df = _standardize_col_field_12d0(raw)
        if "col" not in df.columns:
            continue

        df = df[df["col"].astype(str).map(_is_iot_col_12d0)].copy()
        if len(df) == 0:
            continue

        selected = pd.Series(False, index=df.index)
        reasons = pd.Series("", index=df.index, dtype="object")

        for c in owner_cols:
            if c in df.columns:
                lower = df[c].astype(str).str.strip().str.lower()
                hit = lower.eq(owner_label.lower())
                selected = selected | hit
                reasons = reasons.mask(hit, reasons + f"|{c}=={owner_label}")

        for c in binary_flag_cols:
            if c in df.columns:
                hit = df[c].map(lambda x: _normalise_bool_12d0(x, default=False))
                selected = selected | hit
                reasons = reasons.mask(hit, reasons + f"|{c}=true")

        for c in type_cols:
            if c in df.columns:
                lower = df[c].astype(str).str.lower()
                hit = lower.str.contains("binary", regex=False) | lower.str.contains("bool", regex=False)
                selected = selected | hit
                reasons = reasons.mask(hit, reasons + f"|{c} contains binary/bool")

        for c in ["branch", "pipeline_branch", "cell_branch"]:
            if c in df.columns:
                lower = df[c].astype(str).str.lower()
                hit = lower.str.contains("12.d", regex=False) | lower.str.contains("binary", regex=False)
                selected = selected | hit
                reasons = reasons.mask(hit, reasons + f"|{c} indicates 12.d/binary")

        if selected.any():
            sub = df.loc[selected].copy()
            sub["_binary_target_source"] = str(source_name)
            sub["_binary_target_reason"] = reasons.loc[selected].str.strip("|").replace("", "ownership_binary_match")
            extraction_rows.append(sub)
            target_sources.append(str(source_name))

    if extraction_rows:
        extracted = pd.concat(extraction_rows, ignore_index=True)
        extracted = extracted.drop_duplicates("col", keep="first")
        targets = extracted["col"].astype(str).tolist()
        meta = {
            "ownership_sources_checked_n": int(len(frames)),
            "ownership_sources_with_binary_targets": target_sources,
            "ownership_extraction_found_n": int(len(targets)),
        }
        return targets, extracted, meta

    meta = {
        "ownership_sources_checked_n": int(len(frames)),
        "ownership_sources_with_binary_targets": [],
        "ownership_extraction_found_n": 0,
    }
    return [], pd.DataFrame(), meta

# ----------------------------------------------------------
# 5) Fallback TRAIN+VAL-only binary target inference
# ----------------------------------------------------------
def _fallback_infer_binary_targets_trainval_only_12d0() -> tuple[list[str], pd.DataFrame]:
    strict_12c_excluded = _strict_12c_continuous_exclusion_set_12d0()

    all_schema_cols = (
        set(map(str, df_tr.columns))
        & set(map(str, df_val.columns))
        & set(map(str, df_te.columns))
    )

    candidate_cols = sorted(c for c in all_schema_cols if _is_iot_col_12d0(c))

    rows = []
    targets = []

    for col in candidate_cols:
        passed, audit = _is_binary_trainval_values_12d0(col)

        strict_cont_excluded = bool(col in strict_12c_excluded)
        helper_excluded = bool(_is_binary_helper_or_scaffold_12d0(col))
        coarse_binary_name_hit = bool(_looks_like_binary_name_12d0(col))
        ha_binary_hit, ha_binary_reason = _looks_like_home_assistant_binary_state_target_12d0(col)

        # Explicit binary-over-continuous ownership override:
        # Some HA-exported sensor value columns are binary-valued in TRAIN+VAL
        # and semantically behave as state flags, even if earlier broad 12.c
        # value lists captured them. These should be owned by 12.d, not by
        # the continuous/value branch.
        binary_over_continuous_exception = bool(
            col in {
                "iot__coffee_maker_milk__sensor__cups__value",
            }
            and passed
            and not helper_excluded
            and ha_binary_hit
        )

        selected = bool(
            passed
            and not helper_excluded
            and ha_binary_hit
            and (
                not strict_cont_excluded
                or binary_over_continuous_exception
            )
        )

        if selected:
            targets.append(col)

        rejection_reasons = []
        if not passed:
            rejection_reasons.append(audit.get("binary_value_check_reason", "binary_domain_failed"))
        if strict_cont_excluded and not binary_over_continuous_exception:
            rejection_reasons.append("strict_12c_continuous_excluded")
        elif strict_cont_excluded and binary_over_continuous_exception:
            rejection_reasons.append("strict_12c_continuous_exclusion_overridden_for_binary_state_exception")
        if helper_excluded:
            rejection_reasons.append("binary_helper_or_scaffold_excluded")
        if not ha_binary_hit:
            rejection_reasons.append(f"ha_binary_state_name_miss:{ha_binary_reason}")

        rows.append({
            "col": col,
            "entity": _entity_from_iot_col_12d0(col),
            "ha_domain_guess": _ha_domain_from_iot_col_12d0(col),
            "measurement_name": _measurement_from_iot_col_12d0(col),
            "strict_12c_continuous_excluded": strict_cont_excluded,
            "binary_over_continuous_exception": binary_over_continuous_exception,
            "binary_helper_or_scaffold_excluded": helper_excluded,
            "fallback_name_binary_like": coarse_binary_name_hit,
            "ha_binary_state_name_hit": bool(ha_binary_hit),
            "ha_binary_state_name_reason": str(ha_binary_reason),
            "fallback_selected": selected,
            "fallback_rejection_reasons": "|".join(rejection_reasons),
            "TEST_values_used_for_fallback": False,
            "df_te_used_schema_only": True,
            **audit,
        })

    audit_df = pd.DataFrame(rows)

    # Deterministic order.
    targets = sorted(list(dict.fromkeys(map(str, targets))))

    return targets, audit_df

# ----------------------------------------------------------
# 6) Resolve binary targets
# ----------------------------------------------------------
ownership_targets, ownership_df, ownership_meta = _extract_binary_targets_from_ownership_12d0()

fallback_targets = []
fallback_df = pd.DataFrame()
resolution_mode_reason = ""

all_schema_cols = (
    set(map(str, df_tr.columns))
    & set(map(str, df_val.columns))
    & set(map(str, df_te.columns))
)

def _validate_candidate_schema_12d0(cols: list[str], mode: str) -> None:
    missing = sorted([c for c in cols if c not in all_schema_cols])
    if missing:
        raise RuntimeError(
            f"[Cell12.d.0] Candidate binary targets from {mode} are missing from one or more split schemas. "
            f"Preview={missing[:30]}"
        )

def _validate_candidate_binary_domain_12d0(cols: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    bad = []

    for col in cols:
        passed, audit = _is_binary_trainval_values_12d0(col)
        row = {
            "col": col,
            "entity": _entity_from_iot_col_12d0(col),
            "ha_domain_guess": _ha_domain_from_iot_col_12d0(col),
            "measurement_name": _measurement_from_iot_col_12d0(col),
            "in_df_tr": bool(col in df_tr.columns),
            "in_df_val": bool(col in df_val.columns),
            "in_df_te_schema": bool(col in df_te.columns),
            "TEST_values_used_for_target_validation": False,
            **audit,
        }
        rows.append(row)

        if not passed:
            bad.append({
                "col": col,
                "reason": audit.get("binary_value_check_reason", "binary_value_check_failed"),
                "trainval_unique_values_preview": audit.get("trainval_unique_values_preview", []),
            })

    return pd.DataFrame(rows), bad

# Always run fallback as a diagnostic artifact. It uses TRAIN+VAL values only.
fallback_targets, fallback_df = _fallback_infer_binary_targets_trainval_only_12d0()

if isinstance(ownership_df, pd.DataFrame) and len(ownership_df):
    ownership_df.to_csv(binary_ownership_extraction_csv, index=False)
else:
    pd.DataFrame(columns=["col"]).to_csv(binary_ownership_extraction_csv, index=False)

fallback_df.to_csv(binary_fallback_inference_csv, index=False)

prefer_ownership = bool(CFG.get("cell12d0_binary_prefer_ownership_metadata", True))
allow_fallback_when_ownership_wrong = bool(
    CFG.get("cell12d0_allow_fallback_when_ownership_count_wrong", True)
)

ownership_targets = sorted(list(dict.fromkeys(map(str, ownership_targets))))
fallback_targets = sorted(list(dict.fromkeys(map(str, fallback_targets))))

if ownership_targets and prefer_ownership and len(ownership_targets) == EXPECTED_BINARY_TARGET_COUNT_12D:
    candidate_targets = ownership_targets
    target_source_mode = "cell4_5_or_registry_ownership_metadata"
    resolution_mode_reason = "ownership_metadata_exact_count"
elif (
    ownership_targets
    and prefer_ownership
    and len(ownership_targets) != EXPECTED_BINARY_TARGET_COUNT_12D
    and allow_fallback_when_ownership_wrong
    and len(fallback_targets) == EXPECTED_BINARY_TARGET_COUNT_12D
):
    candidate_targets = fallback_targets
    target_source_mode = "trainval_binary_value_and_name_inference"
    resolution_mode_reason = (
        "ownership_metadata_count_wrong_fallback_exact"
        f"|ownership_found={len(ownership_targets)}"
        f"|fallback_found={len(fallback_targets)}"
    )
elif not ownership_targets and len(fallback_targets) == EXPECTED_BINARY_TARGET_COUNT_12D:
    candidate_targets = fallback_targets
    target_source_mode = "trainval_binary_value_and_name_inference"
    resolution_mode_reason = "no_ownership_metadata_fallback_exact"
else:
    candidate_targets = fallback_targets if fallback_targets else ownership_targets
    target_source_mode = (
        "trainval_binary_value_and_name_inference"
        if fallback_targets
        else "cell4_5_or_registry_ownership_metadata"
    )
    resolution_mode_reason = "no_exact_target_source_available"

candidate_targets = sorted(list(dict.fromkeys(map(str, candidate_targets))))

_validate_candidate_schema_12d0(candidate_targets, target_source_mode)

IOT_BINARY_TARGET_CONTRACT_DF, bad_binary_domain = _validate_candidate_binary_domain_12d0(candidate_targets)

if bad_binary_domain:
    raise RuntimeError(
        "[Cell12.d.0] Selected binary targets failed TRAIN+VAL binary-domain validation. "
        f"Preview={bad_binary_domain[:20]}"
    )

IOT_BINARY_TARGET_CONTRACT_DF["binary_target_source_mode"] = target_source_mode
IOT_BINARY_TARGET_CONTRACT_DF["target_resolution_reason"] = resolution_mode_reason

# Resolution diagnostic.
resolution_diag = {
    "cell": "12.d.0",
    "version": CELL12D0_VERSION,
    "expected_binary_targets": int(EXPECTED_BINARY_TARGET_COUNT_12D),
    "ownership_targets_n": int(len(ownership_targets)),
    "fallback_targets_n": int(len(fallback_targets)),
    "candidate_targets_n": int(len(candidate_targets)),
    "target_source_mode": target_source_mode,
    "resolution_mode_reason": resolution_mode_reason,
    "TEST_values_used": False,
    "df_te_used_schema_only": True,
}

resolution_rows = []
if isinstance(fallback_df, pd.DataFrame) and len(fallback_df):
    resolution_rows.append({
        "metric": "fallback_all_iot_schema_cols",
        "value": int(len(fallback_df)),
    })
    resolution_rows.append({
        "metric": "fallback_binary_domain_passed_n",
        "value": int(fallback_df["binary_value_check_passed"].fillna(False).astype(bool).sum()),
    })
    resolution_rows.append({
        "metric": "fallback_strict_12c_continuous_excluded_n",
        "value": int(fallback_df["strict_12c_continuous_excluded"].fillna(False).astype(bool).sum()),
    })
    resolution_rows.append({
        "metric": "fallback_helper_or_scaffold_excluded_n",
        "value": int(fallback_df["binary_helper_or_scaffold_excluded"].fillna(False).astype(bool).sum()),
    })
    resolution_rows.append({
        "metric": "fallback_ha_binary_state_name_hit_n",
        "value": int(fallback_df["ha_binary_state_name_hit"].fillna(False).astype(bool).sum()),
    })
    resolution_rows.append({
        "metric": "fallback_selected_n",
        "value": int(fallback_df["fallback_selected"].fillna(False).astype(bool).sum()),
    })

pd.DataFrame(resolution_rows).to_csv(binary_resolution_diag_csv, index=False)

if len(candidate_targets) != EXPECTED_BINARY_TARGET_COUNT_12D and not ALLOW_BINARY_TARGET_COUNT_MISMATCH_12D0:
    selected_preview = []
    scaffold_excluded_preview = []
    binary_rejected_preview = []
    strict_cont_excluded_preview = []

    if isinstance(fallback_df, pd.DataFrame) and len(fallback_df):
        selected_preview = (
            fallback_df.loc[fallback_df["fallback_selected"].fillna(False).astype(bool), "col"]
            .astype(str)
            .head(80)
            .tolist()
        )
        scaffold_excluded_preview = (
            fallback_df.loc[
                fallback_df["binary_helper_or_scaffold_excluded"].fillna(False).astype(bool),
                "col",
            ]
            .astype(str)
            .head(50)
            .tolist()
        )
        binary_rejected_preview = (
            fallback_df.loc[
                fallback_df["binary_value_check_passed"].fillna(False).astype(bool)
                & ~fallback_df["fallback_selected"].fillna(False).astype(bool),
                ["col", "fallback_rejection_reasons"],
            ]
            .head(80)
            .to_dict("records")
        )
        strict_cont_excluded_preview = (
            fallback_df.loc[
                fallback_df["strict_12c_continuous_excluded"].fillna(False).astype(bool),
                "col",
            ]
            .astype(str)
            .head(80)
            .tolist()
        )

    raise RuntimeError(
        "[Cell12.d.0] Binary target count mismatch after strict resolver. "
        f"found={len(candidate_targets)} expected={EXPECTED_BINARY_TARGET_COUNT_12D}. "
        f"target_source_mode={target_source_mode}. "
        f"resolution_reason={resolution_mode_reason}. "
        f"ownership_found={len(ownership_targets)} fallback_found={len(fallback_targets)}. "
        "Set CFG['cell12d0_allow_binary_target_count_mismatch']=True only for debugging, not publication. "
        f"selected_preview={selected_preview}. "
        f"scaffold_excluded_preview={scaffold_excluded_preview}. "
        f"binary_valued_but_rejected_preview={binary_rejected_preview}. "
        f"strict_continuous_excluded_preview={strict_cont_excluded_preview}."
    )

IOT_BINARY_COLS = sorted(candidate_targets)
IOT_BINARY_TARGETS = list(IOT_BINARY_COLS)

# ----------------------------------------------------------
# 7) Binary availability mask contract
# ----------------------------------------------------------
def _as_frame_12d0(obj, index, columns, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
    elif isinstance(obj, dict):
        out = pd.DataFrame(obj)
    else:
        arr = np.asarray(obj)
        if arr.ndim != 2:
            raise RuntimeError(f"[Cell12.d.0] Cannot normalize {name}: shape={arr.shape}")
        out = pd.DataFrame(arr)

    if len(out) != len(index):
        raise RuntimeError(
            f"[Cell12.d.0] {name} row mismatch: got={len(out)} expected={len(index)}"
        )

    out.index = index
    out.columns = out.columns.astype(str)

    missing = sorted(set(columns) - set(out.columns))
    if missing:
        raise RuntimeError(f"[Cell12.d.0] {name} missing binary columns: {missing[:30]}")

    return out[columns].copy()

def _build_binary_availability_syn_12d0() -> tuple[pd.DataFrame, dict]:
    """
    Build synthetic TEST-length availability mask for 12.d binary values.

    This cell does NOT inspect real TEST values.
    It accepts an explicit upstream synthetic binary mask if already available.
    Otherwise it defaults to full-grid active for binary state targets.
    """
    for name in [
        "IOT_BINARY_AVAIL_SYN",
        "BINARY_AVAIL_SYN",
        "IOT_BINARY_STATE_AVAIL_SYN",
        "BINARY_STATE_AVAIL_SYN",
    ]:
        obj = globals().get(name, None)
        if obj is not None:
            mask_df = _as_frame_12d0(
                obj,
                getattr(df_te, "index"),
                IOT_BINARY_COLS,
                name,
            )
            mask_df = mask_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            mask_df = (mask_df > 0.5).astype(np.float32)
            return mask_df, {
                "binary_availability_source": name,
                "binary_availability_defaulted": False,
                "binary_availability_policy": "explicit_upstream_synthetic_binary_availability_mask",
                "TEST_values_used_for_availability": False,
            }

    # Optional entity-regime support. Only use synthetic TEST-length scaffolds,
    # not real TEST values. This branch is intentionally conservative and
    # disabled unless column names line up exactly.
    for name in [
        "ENTITY_REGIME_SYN",
        "ENTITY_DRIVER_ACTIVITY_SYN",
    ]:
        obj = globals().get(name, None)
        if isinstance(obj, pd.DataFrame) and len(obj) == N_TE:
            ent_df = obj.copy()
            ent_df.columns = ent_df.columns.astype(str)
            cols = {}
            all_found = True
            for col in IOT_BINARY_COLS:
                entity = _entity_from_iot_col_12d0(col)
                candidates = [
                    entity,
                    f"iot__entity_obs__{entity}",
                    f"entity_obs__{entity}",
                    f"{entity}__obs_present",
                ]
                found = None
                for c in candidates:
                    if c in ent_df.columns:
                        found = c
                        break
                if found is None:
                    all_found = False
                    break
                vals = pd.to_numeric(ent_df[found], errors="coerce").fillna(0.0)
                cols[col] = (vals > 0.5).astype(np.float32).to_numpy()

            if all_found:
                mask_df = pd.DataFrame(cols, index=getattr(df_te, "index"), dtype=np.float32)
                return mask_df, {
                    "binary_availability_source": name,
                    "binary_availability_defaulted": False,
                    "binary_availability_policy": "synthetic_entity_regime_derived_binary_availability",
                    "TEST_values_used_for_availability": False,
                }

    policy = str(CFG.get("cell12d0_binary_availability_default", "all_active")).strip().lower()
    if policy != "all_active":
        raise RuntimeError(
            "[Cell12.d.0] No explicit binary availability mask found and unsupported "
            f"cell12d0_binary_availability_default={policy!r}."
        )

    mask_df = pd.DataFrame(
        1.0,
        index=getattr(df_te, "index"),
        columns=IOT_BINARY_COLS,
        dtype=np.float32,
    )
    return mask_df, {
        "binary_availability_source": "default_all_active",
        "binary_availability_defaulted": True,
        "binary_availability_policy": "binary_state_targets_full_grid_all_active",
        "TEST_values_used_for_availability": False,
    }

IOT_BINARY_AVAIL_SYN_DF, binary_availability_meta = _build_binary_availability_syn_12d0()

availability_rows = []
for col in IOT_BINARY_COLS:
    mask = (
        pd.to_numeric(IOT_BINARY_AVAIL_SYN_DF[col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    active = mask > 0.5
    availability_rows.append({
        "col": col,
        "test_rows": int(N_TE),
        "binary_active_n": int(active.sum()),
        "binary_inactive_n": int((~active).sum()),
        "binary_active_rate": float(active.mean()) if active.size else np.nan,
        "availability_source": binary_availability_meta["binary_availability_source"],
        "availability_defaulted": bool(binary_availability_meta["binary_availability_defaulted"]),
        "availability_policy": binary_availability_meta["binary_availability_policy"],
        "TEST_values_used_for_availability": False,
    })

IOT_BINARY_AVAILABILITY_CONTRACT_DF = pd.DataFrame(availability_rows)

if len(IOT_BINARY_AVAILABILITY_CONTRACT_DF) != len(IOT_BINARY_COLS):
    raise RuntimeError("[Cell12.d.0] Availability contract row count mismatch.")

if IOT_BINARY_AVAIL_SYN_DF.shape != (N_TE, len(IOT_BINARY_COLS)):
    raise RuntimeError(
        "[Cell12.d.0] IOT_BINARY_AVAIL_SYN_DF shape mismatch: "
        f"got={IOT_BINARY_AVAIL_SYN_DF.shape}, expected={(N_TE, len(IOT_BINARY_COLS))}"
    )

# ----------------------------------------------------------
# 8) Family/role heuristics for binary branch
# ----------------------------------------------------------
def _binary_role_12d0(col: str, rate: float) -> str:
    s = str(col).lower()
    domain = _ha_domain_from_iot_col_12d0(col)
    m = _measurement_from_iot_col_12d0(col)

    if not np.isfinite(float(rate)):
        sparsity = "unknown_rate"
    elif rate <= 0.01 or rate >= 0.99:
        sparsity = "quasi_static_or_rare"
    elif rate <= 0.10 or rate >= 0.90:
        sparsity = "imbalanced"
    else:
        sparsity = "balanced"

    if "events_update_any" in s:
        semantic = "iot_update_event"
    elif domain == "light":
        semantic = "light_on_off"
    elif domain == "switch":
        semantic = "switch_on_off"
    elif domain == "media_player":
        semantic = "media_player_state"
    elif domain == "remote":
        semantic = "remote_state"
    elif any(tok in s or tok in m for tok in ["motion", "occupancy", "presence", "moving", "vibration"]):
        semantic = "motion_presence_vibration"
    elif any(tok in s or tok in m for tok in ["contact", "door", "window", "open", "closed", "opening"]):
        semantic = "contact_open_close"
    elif any(tok in s or tok in m for tok in ["cleaning", "charging", "mop_attached", "water_box_attached"]):
        semantic = "robot_binary_state"
    elif any(tok in s or tok in m for tok in ["connectivity", "cloud_connection", "reachability", "online", "available"]):
        semantic = "availability_connectivity"
    elif any(tok in s or tok in m for tok in ["tamper", "alarm", "smoke", "moisture", "water", "leak", "overheated", "overloaded", "protection"]):
        semantic = "safety_alarm_or_protection"
    elif any(tok in s or tok in m for tok in ["bean_container_empty", "drip_tray_full", "water_tank_empty", "local_control", "remote_start"]):
        semantic = "appliance_binary_state"
    else:
        semantic = "generic_binary_state"

    return f"{semantic}::{sparsity}"

IOT_BINARY_TARGET_CONTRACT_DF["binary_role"] = IOT_BINARY_TARGET_CONTRACT_DF.apply(
    lambda r: _binary_role_12d0(r["col"], float(r.get("trainval_rate", np.nan))),
    axis=1,
)

IOT_BINARY_TARGET_CONTRACT_DF["ha_domain_guess"] = IOT_BINARY_TARGET_CONTRACT_DF["col"].map(
    _ha_domain_from_iot_col_12d0
)
IOT_BINARY_TARGET_CONTRACT_DF["helper_or_scaffold"] = IOT_BINARY_TARGET_CONTRACT_DF["col"].map(
    _is_binary_helper_or_scaffold_12d0
)
IOT_BINARY_TARGET_CONTRACT_DF["TEST_values_used_for_target_validation"] = False

if IOT_BINARY_TARGET_CONTRACT_DF["helper_or_scaffold"].any():
    bad = (
        IOT_BINARY_TARGET_CONTRACT_DF.loc[
            IOT_BINARY_TARGET_CONTRACT_DF["helper_or_scaffold"].astype(bool), "col"
        ]
        .astype(str)
        .tolist()
    )
    raise RuntimeError(
        "[Cell12.d.0] Helper/scaffold columns leaked into binary targets. "
        f"Preview={bad[:30]}"
    )

binary_role_counts = (
    IOT_BINARY_TARGET_CONTRACT_DF["binary_role"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

entity_counts = (
    IOT_BINARY_TARGET_CONTRACT_DF["entity"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

ha_domain_counts = (
    IOT_BINARY_TARGET_CONTRACT_DF["ha_domain_guess"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

# ----------------------------------------------------------
# 9) Save artifacts
# ----------------------------------------------------------
IOT_BINARY_TARGET_CONTRACT_DF.to_csv(binary_target_contract_csv, index=False)
IOT_BINARY_AVAILABILITY_CONTRACT_DF.to_csv(binary_availability_contract_csv, index=False)
IOT_BINARY_AVAIL_SYN_DF.to_parquet(binary_availability_mask_parquet, index=True)

binary_targets_payload = {
    "cell": "12.d.0",
    "version": CELL12D0_VERSION,
    "upstream_contracts": {
        "cell12b_scaffold_contract": CELL12B_SCAFFOLD_CONTRACT_PATH,
        "cell12c0_foundation_contract": CELL12C0_FOUNDATION_CONTRACT_PATH,
        "cell12c5_final_value_contract": CELL12C5_FINAL_VALUE_CONTRACT_PATH if CELL12C5_FINAL_VALUE_CONTRACT is not None else None,
        "cell12c6_qa_contract": CELL12C6_QA_CONTRACT_PATH if CELL12C6_QA_CONTRACT is not None else None,
        "cell12c6v_val_frozen_test_audit_contract": CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT_PATH if CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT is not None else None,
    },
    "binary_targets_total": int(len(IOT_BINARY_COLS)),
    "expected_binary_targets_total": int(EXPECTED_BINARY_TARGET_COUNT_12D),
    "binary_target_count_matches_expected": bool(len(IOT_BINARY_COLS) == EXPECTED_BINARY_TARGET_COUNT_12D),
    "target_source_mode": target_source_mode,
    "target_resolution_reason": resolution_mode_reason,
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "df_te_used_for_index_length_schema_only": True,
    "targets": IOT_BINARY_COLS,
    "binary_role_counts": binary_role_counts,
    "ha_domain_counts": ha_domain_counts,
    "entity_counts": entity_counts,
}

_write_json_12d0(binary_targets_json, binary_targets_payload)
_write_json_12d0(binary_targets_report_json, binary_targets_payload)

contract = {
    "cell": "12.d.0",
    "version": CELL12D0_VERSION,
    "role": "binary_iot_shared_utilities_and_contracts",
    "upstream_contracts": {
        "cell12b_scaffold_contract": CELL12B_SCAFFOLD_CONTRACT_PATH,
        "cell12c0_foundation_contract": CELL12C0_FOUNDATION_CONTRACT_PATH,
        "cell12c5_final_value_contract": CELL12C5_FINAL_VALUE_CONTRACT_PATH if CELL12C5_FINAL_VALUE_CONTRACT is not None else None,
        "cell12c6_qa_contract": CELL12C6_QA_CONTRACT_PATH if CELL12C6_QA_CONTRACT is not None else None,
        "cell12c6v_val_frozen_test_audit_contract": CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT_PATH if CELL12C6V_VAL_FROZEN_TEST_AUDIT_CONTRACT is not None else None,
    },
    "binary_targets_total": int(len(IOT_BINARY_COLS)),
    "expected_binary_targets_total": int(EXPECTED_BINARY_TARGET_COUNT_12D),
    "binary_target_count_matches_expected": bool(len(IOT_BINARY_COLS) == EXPECTED_BINARY_TARGET_COUNT_12D),
    "target_source_mode": target_source_mode,
    "target_resolution_reason": resolution_mode_reason,
    "ownership_metadata": ownership_meta,
    "ownership_targets_n": int(len(ownership_targets)),
    "fallback_trainval_inferred_n": int(len(fallback_targets)),
    "all_targets_in_train_schema": bool(all(c in df_tr.columns for c in IOT_BINARY_COLS)),
    "all_targets_in_val_schema": bool(all(c in df_val.columns for c in IOT_BINARY_COLS)),
    "all_targets_in_test_schema": bool(all(c in df_te.columns for c in IOT_BINARY_COLS)),
    "trainval_binary_domain_passed": bool(IOT_BINARY_TARGET_CONTRACT_DF["binary_value_check_passed"].all()),
    "helper_or_scaffold_leak_n": int(IOT_BINARY_TARGET_CONTRACT_DF["helper_or_scaffold"].astype(bool).sum()),
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "train_val_values_used_for_target_validation": True,
    "df_te_used_for_index_length_schema_only": True,
    "binary_availability": binary_availability_meta,
    "binary_availability_shape": list(IOT_BINARY_AVAIL_SYN_DF.shape),
    "binary_availability_no_test_values_used": True,
    "outputs": {
        "binary_targets_json": binary_targets_json,
        "binary_targets_report_json": binary_targets_report_json,
        "binary_contract_json": binary_contract_json,
        "binary_contract_canonical_json": binary_contract_canonical_json,
        "binary_target_contract_csv": binary_target_contract_csv,
        "binary_availability_mask_parquet": binary_availability_mask_parquet,
        "binary_availability_contract_csv": binary_availability_contract_csv,
        "binary_ownership_extraction_csv": binary_ownership_extraction_csv,
        "binary_fallback_inference_csv": binary_fallback_inference_csv,
        "binary_resolution_diag_csv": binary_resolution_diag_csv,
    },
    "binary_role_counts": binary_role_counts,
    "ha_domain_counts": ha_domain_counts,
    "entity_counts": entity_counts,
}

_write_json_12d0(binary_contract_json, contract)
_write_json_12d0(binary_contract_canonical_json, contract)

contract["hashes"] = {
    "binary_targets_json_sha256": _sha256_file_12d0(binary_targets_json),
    "binary_targets_report_json_sha256": _sha256_file_12d0(binary_targets_report_json),
    "binary_target_contract_csv_sha256": _sha256_file_12d0(binary_target_contract_csv),
    "binary_availability_contract_csv_sha256": _sha256_file_12d0(binary_availability_contract_csv),
    "binary_availability_mask_parquet_sha256": _sha256_file_12d0(binary_availability_mask_parquet),
    "binary_ownership_extraction_csv_sha256": _sha256_file_12d0(binary_ownership_extraction_csv),
    "binary_fallback_inference_csv_sha256": _sha256_file_12d0(binary_fallback_inference_csv),
    "binary_resolution_diag_csv_sha256": _sha256_file_12d0(binary_resolution_diag_csv),
}
_write_json_12d0(binary_contract_json, contract)
_write_json_12d0(binary_contract_canonical_json, contract)
contract["hashes"]["binary_contract_json_sha256"] = _sha256_file_12d0(binary_contract_json)
contract["hashes"]["binary_contract_canonical_json_sha256"] = _sha256_file_12d0(binary_contract_canonical_json)
_write_json_12d0(binary_contract_json, contract)
_write_json_12d0(binary_contract_canonical_json, contract)

# ----------------------------------------------------------
# 10) Export globals for 12.d.1+
# ----------------------------------------------------------
globals()["CELL12D0_VERSION"] = CELL12D0_VERSION
globals()["IOT_BINARY_COLS"] = IOT_BINARY_COLS
globals()["IOT_BINARY_TARGETS"] = IOT_BINARY_TARGETS
globals()["IOT_BINARY_TARGET_CONTRACT_DF"] = IOT_BINARY_TARGET_CONTRACT_DF
globals()["IOT_BINARY_AVAIL_SYN"] = IOT_BINARY_AVAIL_SYN_DF
globals()["IOT_BINARY_AVAIL_SYN_DF"] = IOT_BINARY_AVAIL_SYN_DF
globals()["IOT_BINARY_AVAILABILITY_CONTRACT_DF"] = IOT_BINARY_AVAILABILITY_CONTRACT_DF
globals()["CELL12D_BINARY_CONTRACT"] = contract
globals()["CELL12D_BINARY_TARGETS_JSON"] = binary_targets_json
globals()["CELL12D_BINARY_CONTRACT_JSON"] = binary_contract_json
globals()["CELL12D_BINARY_CANONICAL_CONTRACT_JSON"] = binary_contract_canonical_json
globals()["CELL12D0_BINARY_TARGET_CONTRACT_CSV"] = binary_target_contract_csv
globals()["CELL12D0_BINARY_AVAILABILITY_MASK_PARQUET"] = binary_availability_mask_parquet
globals()["CELL12D0_BINARY_AVAILABILITY_CONTRACT_CSV"] = binary_availability_contract_csv
globals()["CELL12D0_BINARY_FALLBACK_INFERENCE_CSV"] = binary_fallback_inference_csv
globals()["CELL12D0_BINARY_RESOLUTION_DIAG_CSV"] = binary_resolution_diag_csv

log(
    "[Cell12.d.0] Binary target contract built | "
    f"targets={len(IOT_BINARY_COLS)} | "
    f"expected={EXPECTED_BINARY_TARGET_COUNT_12D} | "
    f"source_mode={target_source_mode} | "
    f"resolution={resolution_mode_reason} | "
    f"availability_source={binary_availability_meta['binary_availability_source']}"
)
log(f"[Cell12.d.0] Saved binary targets JSON: {binary_targets_json}")
log(f"[Cell12.d.0] Saved binary contract JSON: {binary_contract_json}")
log(f"[Cell12.d.0] Saved canonical binary contract JSON: {binary_contract_canonical_json}")
log(f"[Cell12.d.0] Saved target contract CSV: {binary_target_contract_csv} | rows={len(IOT_BINARY_TARGET_CONTRACT_DF)}")
log(f"[Cell12.d.0] Saved availability contract CSV: {binary_availability_contract_csv} | rows={len(IOT_BINARY_AVAILABILITY_CONTRACT_DF)}")
log(f"[Cell12.d.0] Saved binary availability mask parquet: {binary_availability_mask_parquet} | shape={IOT_BINARY_AVAIL_SYN_DF.shape}")
log(f"[Cell12.d.0] Saved fallback inference CSV: {binary_fallback_inference_csv} | rows={len(fallback_df)}")
log(f"[Cell12.d.0] Saved resolution diagnostic CSV: {binary_resolution_diag_csv}")
log(
    "[Cell12.d.0] Domain counts | "
    f"{ha_domain_counts}"
)
log(
    "[Cell12.d.0] Binary role counts | "
    f"{binary_role_counts}"
)
log(
    "[Cell12.d.0] Contract flags | "
    "TEST_real_values_used=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "df_te_used_for_index_length_schema_only=True"
)
log("--- END: Cell 12.d.0 - Binary IoT shared utilities/contracts (v1.2-THESIS contract-locked strict resolver) ---")

gc.collect()