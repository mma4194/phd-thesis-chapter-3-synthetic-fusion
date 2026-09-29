# ==========================================================
# CELL 12.e.0 - Driver contracts
# v1.1 STUDY-THESIS strict driver/event contract resolver
#
# Why v1.1:
#   - The original v1.0 fallback resolver only considered columns
#     starting with "iot__", but this project’s driver branch is expected
#     to include sparse event-driver columns such as "events_in_sec__...".
#   - v1.1 explicitly supports both:
#       1) events_in_sec__... driver columns
#       2) iot__... event/driver/update/change columns
#   - Adds canonical contract under artifacts/contracts.
#   - Persists IOT_DRIVER_AVAIL_SYN.parquet for downstream/restart safety.
#
# Role:
#   - Define IoT/CPS driver columns for the 12.e driver branch.
#   - Prefer explicit upstream driver globals/registries when valid.
#   - Otherwise infer driver columns from TRAIN+VAL-only semantics.
#   - Build driver type/domain contract for 12.e.1–12.e.6.
#
# Strict rules:
#   - Does NOT generate driver values.
#   - Does NOT fit generators.
#   - Does NOT select generators.
#   - Does NOT read real TEST driver values for discovery/calibration.
#   - df_te is used only for schema/index/length alignment.
#
# Expected driver count:
#   26
# ==========================================================

log("--- START: Cell 12.e.0 - Driver contracts (v1.1 strict resolver, events_in_sec-aware) ---")

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
_required_12e0 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
]
_missing_12e0 = [k for k in _required_12e0 if k not in globals()]
if _missing_12e0:
    raise RuntimeError(f"[Cell12.e.0] Missing required globals: {_missing_12e0}")

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

if N_TR <= 0 or N_VAL <= 0 or N_TE <= 0:
    raise RuntimeError(
        f"[Cell12.e.0] Invalid split lengths: train={N_TR}, val={N_VAL}, test={N_TE}"
    )

CELL12E0_VERSION = "cell12e0_driver_contract_strict_v1_1_events_in_sec_aware"

CFG["cell12e0_version"] = CELL12E0_VERSION
CFG["cell12e0_TEST_real_values_used"] = False
CFG["cell12e0_selection_done_here"] = False
CFG["cell12e0_generator_fit_done_here"] = False
CFG["cell12e0_train_val_values_used_for_target_validation"] = True
CFG["cell12e0_df_te_used_for_schema_index_length_only"] = True
CFG["cell12e0_events_in_sec_aware"] = True

CFG.setdefault("cell12e_expected_driver_target_count", 26)
CFG.setdefault("cell12e0_allow_driver_target_count_mismatch", False)
CFG.setdefault("cell12e0_allow_fallback_when_explicit_count_wrong", True)
CFG.setdefault("cell12e0_prefer_explicit_driver_metadata", True)
CFG.setdefault("cell12e0_driver_availability_default", "all_active")
CFG.setdefault("cell12e0_driver_min_trainval_observed_n", 1)
CFG.setdefault("cell12e0_sparse_rate_threshold", 0.05)
CFG.setdefault("cell12e0_countlike_integer_tolerance", 1e-6)
CFG.setdefault("cell12e0_driver_copy_risk_duplicate_rate_warn", 0.98)

EXPECTED_DRIVER_TARGET_COUNT_12E = int(CFG.get("cell12e_expected_driver_target_count", 26))
ALLOW_DRIVER_TARGET_COUNT_MISMATCH_12E0 = bool(
    CFG.get("cell12e0_allow_driver_target_count_mismatch", False)
)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
driver_targets_json = os.path.join(ARTDIR, "IOT_DRIVER_TARGETS.json")
driver_targets_report_json = os.path.join(REPORT_DIR, "IOT_DRIVER_TARGETS.json")
driver_contract_json = os.path.join(REPORT_DIR, "cell12e_driver_contract.json")
driver_contract_canonical_json = os.path.join(
    CONTRACT_DIR, "cell12e0_driver_contract_v1_1_THESIS.json"
)
driver_target_contract_csv = os.path.join(REPORT_DIR, "cell12e0_driver_target_contract.csv")
driver_availability_contract_csv = os.path.join(REPORT_DIR, "cell12e0_driver_availability_contract.csv")
driver_resolution_diag_csv = os.path.join(REPORT_DIR, "cell12e0_driver_resolution_diagnostic.csv")
driver_fallback_inference_csv = os.path.join(REPORT_DIR, "cell12e0_driver_fallback_trainval_inference.csv")
driver_explicit_diag_csv = os.path.join(REPORT_DIR, "cell12e0_driver_explicit_metadata_diagnostic.csv")
driver_avail_parquet = os.path.join(OUT_SYN, "IOT_DRIVER_AVAIL_SYN.parquet")
driver_manifest_json = os.path.join(ARTDIR, "cell12e0_driver_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_12e0(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12e0(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12e0(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12e0(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12e0(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12e0(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12e0(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12e0(obj.to_dict())
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

def _write_json_12e0(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12e0(payload), f, indent=2, sort_keys=True)

def _sha256_file_12e0(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_csv_if_exists_12e0(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _normalise_token_12e0(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _safe_float_12e0(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_bool_12e0(x, default=False) -> bool:
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

def _is_iot_col_12e0(col: str) -> bool:
    return str(col).startswith("iot__")

def _is_events_in_sec_col_12e0(col: str) -> bool:
    return str(col).startswith("events_in_sec__")

def _is_driver_namespace_col_12e0(col: str) -> bool:
    return _is_events_in_sec_col_12e0(col) or _is_iot_col_12e0(col)

def _is_protocol_col_12e0(col: str) -> bool:
    s = str(col).lower()
    return bool(
        s.startswith("router__")
        or s.startswith("ota__")
        or s.startswith("ota24__")
        or s.startswith("ota5__")
        or s.startswith("zigbee__")
        or s.startswith("zwave__")
    )

def _is_observability_or_helper_col_12e0(col: str) -> bool:
    s = str(col).lower()
    helper_tokens = [
        "obs_present",
        "present_mask",
        "availability",
        "available",
        "is_available",
        "stale",
        "staleness",
        "stale_flag",
        "missing",
        "is_missing",
        "nan_mask",
        "__mask__",
        "capture",
        "coverage",
        "quality_flag",
        "valid_flag",
    ]
    if _is_events_in_sec_col_12e0(s):
        return False
    if any(tok in s for tok in helper_tokens):
        return True
    if s.startswith("iot__entity_obs__") or s.startswith("iot__entity_stale__"):
        return True
    if s in {"iot__obs_present", "iot__tier_present"}:
        return True
    return False

def _is_time_like_col_12e0(col: str) -> bool:
    s = str(col).lower()
    return bool(
        s == "sec"
        or s == "time"
        or s == "timestamp"
        or "datetime" in s
        or "epoch" in s
        or s.endswith("__sec")
        or s.endswith("_ts")
    )

def _entity_from_driver_col_12e0(col: str) -> str:
    s = str(col)
    parts = [p for p in s.split("__") if p != ""]
    if not parts:
        return s
    if parts[0] == "iot":
        return parts[1] if len(parts) > 1 else s
    if parts[0] == "events_in_sec":
        if len(parts) > 2 and parts[1] in {"entity", "iot", "device"}:
            return parts[2]
        return parts[1] if len(parts) > 1 else s
    return parts[0]

def _measurement_from_col_12e0(col: str) -> str:
    parts = [p.strip().lower() for p in str(col).split("__") if str(p).strip()]
    if not parts:
        return _normalise_token_12e0(col)
    if parts[0] == "iot":
        parts = parts[1:]
    elif parts[0] == "events_in_sec":
        # Strip prefix and optional "entity"/"device" marker plus entity name.
        parts = parts[1:]
        if parts and parts[0] in {"entity", "device", "iot"}:
            parts = parts[1:]
        if parts:
            parts = parts[1:]
    if len(parts) >= 2 and parts[-1] in {"state", "value", "count", "flag", "rate"}:
        return _normalise_token_12e0(parts[-2])
    return _normalise_token_12e0(parts[-1] if parts else col)

def _to_num_array_12e0(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _finite_trainval_values_12e0(col: str) -> np.ndarray:
    if col not in df_tr.columns or col not in df_val.columns:
        return np.asarray([], dtype=np.float64)
    tr = _to_num_array_12e0(df_tr, col)
    va = _to_num_array_12e0(df_val, col)
    x = np.concatenate([tr, va])
    return x[np.isfinite(x)]

def _integer_fraction_12e0(x: np.ndarray, tol: float = 1e-6) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(np.abs(x - np.round(x)) <= tol))

def _nonzero_rate_12e0(x: np.ndarray, tol: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(np.abs(x) > tol))

def _is_binary_values_12e0(x: np.ndarray, tol: float = 1e-6) -> bool:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    return bool(np.all((np.abs(x - 0.0) <= tol) | (np.abs(x - 1.0) <= tol)))

def _is_nonnegative_12e0(x: np.ndarray, tol: float = 1e-9) -> bool:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    return bool(np.nanmin(x) >= -tol)

def _value_summary_12e0(col: str) -> dict:
    x = _finite_trainval_values_12e0(col)
    if x.size == 0:
        return {
            "trainval_observed_n": 0,
            "trainval_nonzero_rate": np.nan,
            "trainval_mean": np.nan,
            "trainval_var": np.nan,
            "trainval_min": np.nan,
            "trainval_max": np.nan,
            "trainval_unique_n_capped": 0,
            "integer_fraction": np.nan,
            "is_binary_values": False,
            "is_nonnegative": False,
            "variance_to_mean": np.nan,
            "zero_inflated": False,
        }

    mean = float(np.mean(x))
    var = float(np.var(x))
    nz = _nonzero_rate_12e0(x)
    vtm = float(var / max(mean, 1e-12)) if mean > 0 else np.nan
    unique_n = int(min(len(np.unique(x[: min(len(x), 200000)])), 1000000))

    return {
        "trainval_observed_n": int(x.size),
        "trainval_nonzero_rate": nz,
        "trainval_mean": mean,
        "trainval_var": var,
        "trainval_min": float(np.min(x)),
        "trainval_max": float(np.max(x)),
        "trainval_unique_n_capped": unique_n,
        "integer_fraction": _integer_fraction_12e0(x),
        "is_binary_values": _is_binary_values_12e0(x),
        "is_nonnegative": _is_nonnegative_12e0(x),
        "variance_to_mean": vtm,
        "zero_inflated": bool(np.isfinite(nz) and nz <= float(CFG.get("cell12e0_sparse_rate_threshold", 0.05))),
    }

# ----------------------------------------------------------
# 3) Branch plan embedded into contract
# ----------------------------------------------------------
CELL12E_PLAN = {
    "12.e.0": {
        "name": "Driver contracts",
        "status": "create",
        "role": "define 26 driver columns; classify driver types; define domain/mask contracts; no TEST value leakage",
    },
    "12.e.1": {
        "name": "Driver baseline",
        "status": "create",
        "baseline": "A1_driver_event_block_bootstrap",
        "role": "TRAIN-fitted VAL-length driver baseline; copy-risk audited",
    },
    "12.e.2": {
        "name": "Driver VAL candidates",
        "status": "create",
        "candidate_families": {
            "binary_driver": "regime-conditioned Markov",
            "countlike_driver": "Negative Binomial / Poisson-Gamma",
            "continuous_intensity": "AR(1)+Student-t",
            "sparse_burst": "hurdle/zero-inflated Negative Binomial",
            "event_session": "session replay with no-copy audit",
        },
        "role": "generate VAL candidates from TRAIN-only fits",
    },
    "12.e.3": {
        "name": "Driver VAL selection",
        "status": "create",
        "role": "VAL-only portfolio selection against A1",
    },
    "12.e.4": {
        "name": "Driver TEST materialization",
        "status": "create",
        "role": "TEST-length driver generation from locked VAL selections; TRAIN-only refits; no TEST target values",
    },
    "12.e.5": {
        "name": "Driver enforcement",
        "status": "create",
        "role": "hard mask/domain/schema enforcement",
    },
    "12.e.6": {
        "name": "Driver QA",
        "status": "create",
        "role": "final TEST QA only; handoff to Cell 13 coupling QA",
    },
}

# ----------------------------------------------------------
# 4) Exclusion sets from previous branches
# ----------------------------------------------------------
def _collect_known_cols_12e0(names):
    out = set()
    for name in names:
        obj = globals().get(name, None)
        if obj is None:
            continue
        try:
            out.update(map(str, list(obj)))
        except Exception:
            pass
    return out

KNOWN_BINARY_TARGETS_12E0 = _collect_known_cols_12e0([
    "IOT_BINARY_COLS",
    "IOT_BINARY_TARGETS",
    "BINARY_TARGET_COLS",
    "CELL12D_BINARY_TARGET_COLS",
])

KNOWN_CONT_VALUE_TARGETS_12E0 = _collect_known_cols_12e0([
    "IOT_VALUE_COLS",
    "VALUE_COLS",
    "CONT_VALUE_COLS",
    "CELL12C_VALUE_COLS",
    "CELL12C_CONT_VALUE_COLS",
    "IOT_CONT_VALUE_COLS",
    "IOT_CONTINUOUS_VALUE_COLS",
])

# Add report-derived exclusions when available.
for path in [
    os.path.join(REPORT_DIR, "cell12c0_value_contract.csv"),
    os.path.join(REPORT_DIR, "cell12c5_publication_column_registry.csv"),
    os.path.join(REPORT_DIR, "cell12d0_binary_target_contract.csv"),
    os.path.join(REPORT_DIR, "cell12d5_binary_publication_column_registry.csv"),
]:
    df = _read_csv_if_exists_12e0(path)
    if not isinstance(df, pd.DataFrame) or not len(df):
        continue
    col_name = next((c for c in ["col", "column", "target", "target_col", "feature"] if c in df.columns), None)
    if col_name is None:
        continue
    for _, r in df.iterrows():
        c = str(r[col_name])
        row_text = " ".join(str(v).lower() for v in r.values)
        if "binary" in row_text or "12.d" in row_text:
            KNOWN_BINARY_TARGETS_12E0.add(c)
        elif "continuous" in row_text or "value_target" in row_text or "12.c" in row_text:
            KNOWN_CONT_VALUE_TARGETS_12E0.add(c)

log(
    "[Cell12.e.0] Loaded prior target exclusions | "
    f"continuous_value={len(KNOWN_CONT_VALUE_TARGETS_12E0)} | "
    f"binary={len(KNOWN_BINARY_TARGETS_12E0)}"
)

# ----------------------------------------------------------
# 5) Explicit metadata discovery
# ----------------------------------------------------------
def _standardize_col_field_12e0(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "col" not in out.columns:
        for cand in ["column", "name", "feature", "feature_name", "target_col", "iot_col", "value_col", "driver_col"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "col"})
                break
    if "col" in out.columns:
        out["col"] = out["col"].astype(str)
    return out

def _candidate_driver_frames_12e0() -> list:
    frames = []

    global_names = [
        "IOT_DRIVER_COLS",
        "IOT_DRIVER_TARGETS",
        "DRIVER_COLS",
        "DRIVER_TARGET_COLS",
        "CELL12E_DRIVER_COLS",
        "CELL12E_DRIVER_TARGETS",
        "CELL45_IOT_OWNERSHIP_DF",
        "CELL4_5_IOT_OWNERSHIP_DF",
        "IOT_OWNERSHIP_DF",
        "IOT_TARGET_REGISTRY_DF",
        "CELL11_IOT_TARGET_REGISTRY_DF",
    ]

    for name in global_names:
        obj = globals().get(name, None)
        if obj is None:
            continue
        if isinstance(obj, pd.DataFrame) and len(obj):
            frames.append((f"global:{name}", obj.copy()))
        elif isinstance(obj, (list, tuple, set, pd.Index)):
            frames.append((f"global:{name}", pd.DataFrame({"col": list(map(str, obj)), "explicit_driver_list": True})))

    possible_paths = [
        os.path.join(REPORT_DIR, "cell45_iot_ownership.csv"),
        os.path.join(REPORT_DIR, "cell4_5_iot_ownership.csv"),
        os.path.join(REPORT_DIR, "cell4_5_iot_column_ownership.csv"),
        os.path.join(REPORT_DIR, "cell45_iot_column_ownership.csv"),
        os.path.join(REPORT_DIR, "cell45_iot_target_registry.csv"),
        os.path.join(REPORT_DIR, "cell11_iot_target_registry.csv"),
        os.path.join(REPORT_DIR, "cell11_iot_role_registry.csv"),
        os.path.join(REPORT_DIR, "IOT_DRIVER_TARGETS.csv"),
        os.path.join(REPORT_DIR, "cell12e_driver_targets.csv"),
    ]

    for path in possible_paths:
        df = _read_csv_if_exists_12e0(path)
        if isinstance(df, pd.DataFrame) and len(df):
            frames.append((f"report:{os.path.basename(path)}", df.copy()))

    return frames

def _looks_driverish_by_name_12e0(col: str) -> tuple:
    s = str(col).lower()

    if _is_protocol_col_12e0(s):
        return False, "protocol_column"
    if _is_observability_or_helper_col_12e0(s):
        return False, "observability_or_helper"
    if _is_time_like_col_12e0(s):
        return False, "time_like"

    if _is_events_in_sec_col_12e0(s):
        return True, "events_in_sec_namespace"

    if not _is_iot_col_12e0(s):
        return False, "not_driver_namespace"

    if s in KNOWN_BINARY_TARGETS_12E0:
        return False, "known_binary_target"

    strong_patterns = [
        "events_in_sec",
        "__events__",
        "__event__",
        "event_count",
        "event_rate",
        "event_intensity",
        "driver",
        "activity",
        "trigger",
        "update_any",
        "any_update",
        "state_change",
        "changed",
        "change_count",
        "command_count",
    ]

    if any(tok in s for tok in strong_patterns):
        return True, "iot_strong_driver_name_pattern"

    return False, "no_driver_name_pattern"

def _extract_driver_targets_from_metadata_12e0() -> tuple:
    frames = _candidate_driver_frames_12e0()
    rows = []
    targets = []

    owner_cols = [
        "primary_owner", "owner", "target_owner", "iot_owner", "role",
        "target_role", "column_role", "generation_role", "assigned_role",
        "branch", "pipeline_branch",
    ]
    type_cols = [
        "target_type", "type", "semantic_type", "value_type",
        "domain_type", "kind", "family", "stage12e_kind",
    ]

    schema_cols = set(map(str, df_tr.columns)) & set(map(str, df_val.columns)) & set(map(str, df_te.columns))

    for source_name, raw in frames:
        df = _standardize_col_field_12e0(raw)
        if "col" not in df.columns:
            continue

        for _, r in df.iterrows():
            col = str(r.get("col", ""))
            if not col or col not in schema_cols:
                continue

            row_text = " ".join(str(v).strip().lower() for v in r.values)
            explicit_list_hit = _safe_bool_12e0(r.get("explicit_driver_list", False), False)

            owner_hit = any(
                str(r.get(c, "")).strip().lower() in {
                    "driver", "driver_target", "event_driver", "iot_driver",
                    "coupling_driver", "driver_variable",
                }
                for c in owner_cols
                if c in df.columns
            )

            type_hit = any(
                any(tok in str(r.get(c, "")).strip().lower() for tok in [
                    "driver", "event_driver", "coupling_driver",
                    "event_intensity", "event_count", "events_in_sec",
                ])
                for c in type_cols
                if c in df.columns
            )

            text_hit = any(tok in row_text for tok in [
                "driver_target", "event_driver", "coupling_driver",
                "driver variable", "driver_variable", "events_in_sec",
                "event_intensity",
            ])

            name_hit, name_reason = _looks_driverish_by_name_12e0(col)

            selected = bool(explicit_list_hit or owner_hit or type_hit or text_hit)

            if selected:
                targets.append(col)
                rows.append({
                    "col": col,
                    "source": source_name,
                    "explicit_list_hit": explicit_list_hit,
                    "owner_hit": owner_hit,
                    "type_hit": type_hit,
                    "text_hit": text_hit,
                    "name_hit": name_hit,
                    "name_reason": name_reason,
                    "reason": "explicit_or_registry_driver_match",
                    "TEST_values_used": False,
                })

    out_df = pd.DataFrame(rows)
    targets = sorted(list(dict.fromkeys(map(str, targets))))
    meta = {
        "driver_metadata_sources_checked_n": int(len(frames)),
        "driver_metadata_targets_n": int(len(targets)),
    }
    return targets, out_df, meta

# ----------------------------------------------------------
# 6) Fallback TRAIN+VAL-only inference
# ----------------------------------------------------------
def _classify_driver_type_12e0(col: str, summary: dict) -> tuple:
    s = str(col).lower()
    nz = _safe_float_12e0(summary.get("trainval_nonzero_rate"), np.nan)
    mean = _safe_float_12e0(summary.get("trainval_mean"), np.nan)
    vtm = _safe_float_12e0(summary.get("variance_to_mean"), np.nan)
    integer_fraction = _safe_float_12e0(summary.get("integer_fraction"), np.nan)
    is_binary = bool(summary.get("is_binary_values", False))
    nonneg = bool(summary.get("is_nonnegative", False))

    sparse = bool(np.isfinite(nz) and nz <= float(CFG.get("cell12e0_sparse_rate_threshold", 0.05)))
    countlike = bool(nonneg and np.isfinite(integer_fraction) and integer_fraction >= 0.999 and not is_binary)

    if is_binary:
        return "binary_driver", "binary domain {0,1}"

    if "session" in s or "duration" in s or "episode" in s:
        return "event_session", "session/duration-like driver"

    if countlike and sparse:
        return "sparse_burst", "sparse zero-inflated countlike driver"

    if countlike:
        if np.isfinite(vtm) and vtm > 1.25:
            return "countlike_driver", "overdispersed countlike driver"
        return "countlike_driver", "countlike integer nonnegative driver"

    if nonneg:
        if sparse:
            return "sparse_burst", "sparse nonnegative intensity driver"
        return "continuous_intensity", "continuous nonnegative intensity driver"

    return "continuous_intensity", "fallback continuous/intensity driver"

def _fallback_infer_driver_targets_trainval_only_12e0() -> tuple:
    all_schema_cols = set(map(str, df_tr.columns)) & set(map(str, df_val.columns)) & set(map(str, df_te.columns))

    candidate_cols = sorted(c for c in all_schema_cols if _is_driver_namespace_col_12e0(c))

    rows = []
    targets = []

    for col in candidate_cols:
        name_hit, name_reason = _looks_driverish_by_name_12e0(col)
        summary = _value_summary_12e0(col)
        observed_n = int(summary.get("trainval_observed_n", 0))
        driver_type, type_reason = _classify_driver_type_12e0(col, summary)

        finite_ok = bool(observed_n >= int(CFG.get("cell12e0_driver_min_trainval_observed_n", 1)))
        nonnegative_ok = bool(summary.get("is_nonnegative", False) or summary.get("is_binary_values", False))
        selected = bool(name_hit and finite_ok and nonnegative_ok)

        exclusion_reason = ""
        if col in KNOWN_BINARY_TARGETS_12E0:
            selected = False
            exclusion_reason = "known_12d_binary_target"
        elif col in KNOWN_CONT_VALUE_TARGETS_12E0 and not _is_events_in_sec_col_12e0(col):
            selected = False
            exclusion_reason = "known_12c_continuous_value_not_events_in_sec_driver"

        if selected:
            targets.append(col)

        rows.append({
            "col": col,
            "entity": _entity_from_driver_col_12e0(col),
            "measurement_name": _measurement_from_col_12e0(col),
            "namespace": "events_in_sec" if _is_events_in_sec_col_12e0(col) else "iot",
            "name_hit": bool(name_hit),
            "name_reason": name_reason,
            "driver_type_guess": driver_type,
            "driver_type_reason": type_reason,
            "known_binary_target": bool(col in KNOWN_BINARY_TARGETS_12E0),
            "known_continuous_value_target": bool(col in KNOWN_CONT_VALUE_TARGETS_12E0),
            "exclusion_reason": exclusion_reason,
            "fallback_selected": bool(selected),
            "TEST_values_used_for_fallback": False,
            "df_te_used_schema_only": True,
            **summary,
        })

    audit_df = pd.DataFrame(rows)
    targets = sorted(list(dict.fromkeys(map(str, targets))))

    return targets, audit_df

# ----------------------------------------------------------
# 7) Resolve driver targets
# ----------------------------------------------------------
explicit_targets, explicit_df, explicit_meta = _extract_driver_targets_from_metadata_12e0()
fallback_targets, fallback_df = _fallback_infer_driver_targets_trainval_only_12e0()

explicit_targets = sorted(list(dict.fromkeys(map(str, explicit_targets))))
fallback_targets = sorted(list(dict.fromkeys(map(str, fallback_targets))))

explicit_df.to_csv(driver_explicit_diag_csv, index=False)
fallback_df.to_csv(driver_fallback_inference_csv, index=False)

prefer_explicit = bool(CFG.get("cell12e0_prefer_explicit_driver_metadata", True))
allow_fallback_when_explicit_wrong = bool(CFG.get("cell12e0_allow_fallback_when_explicit_count_wrong", True))

explicit_exact = bool(len(explicit_targets) == EXPECTED_DRIVER_TARGET_COUNT_12E)
fallback_exact = bool(len(fallback_targets) == EXPECTED_DRIVER_TARGET_COUNT_12E)

if explicit_targets and prefer_explicit and explicit_exact:
    candidate_targets = explicit_targets
    target_source_mode = "explicit_driver_metadata"
    resolution_reason = "explicit_driver_metadata_exact_count"
elif (
    explicit_targets
    and prefer_explicit
    and not explicit_exact
    and allow_fallback_when_explicit_wrong
    and fallback_exact
):
    candidate_targets = fallback_targets
    target_source_mode = "trainval_driver_name_and_semantic_inference"
    resolution_reason = (
        "explicit_driver_metadata_count_wrong_fallback_exact"
        f"|explicit_found={len(explicit_targets)}"
        f"|fallback_found={len(fallback_targets)}"
    )
elif not explicit_targets and fallback_exact:
    candidate_targets = fallback_targets
    target_source_mode = "trainval_driver_name_and_semantic_inference"
    resolution_reason = "no_explicit_driver_metadata_fallback_exact"
else:
    candidate_targets = fallback_targets if fallback_targets else explicit_targets
    target_source_mode = "trainval_driver_name_and_semantic_inference" if fallback_targets else "explicit_driver_metadata"
    resolution_reason = "no_exact_driver_target_source_available"

candidate_targets = sorted(list(dict.fromkeys(map(str, candidate_targets))))

schema_cols = set(map(str, df_tr.columns)) & set(map(str, df_val.columns)) & set(map(str, df_te.columns))
missing_schema = sorted([c for c in candidate_targets if c not in schema_cols])
if missing_schema:
    raise RuntimeError(
        "[Cell12.e.0] Candidate driver targets missing from one or more split schemas. "
        f"Preview={missing_schema[:30]}"
    )

if len(candidate_targets) != EXPECTED_DRIVER_TARGET_COUNT_12E and not ALLOW_DRIVER_TARGET_COUNT_MISMATCH_12E0:
    fallback_selected_preview = []
    fallback_rejected_driver_like_preview = []

    if isinstance(fallback_df, pd.DataFrame) and len(fallback_df):
        fallback_selected_preview = (
            fallback_df.loc[fallback_df["fallback_selected"].fillna(False).astype(bool), "col"]
            .astype(str)
            .head(80)
            .tolist()
        )
        fallback_rejected_driver_like_preview = (
            fallback_df.loc[
                fallback_df["name_hit"].fillna(False).astype(bool)
                & ~fallback_df["fallback_selected"].fillna(False).astype(bool),
                ["col", "namespace", "name_reason", "exclusion_reason", "trainval_observed_n", "is_nonnegative"],
            ]
            .head(80)
            .to_dict("records")
        )

    raise RuntimeError(
        "[Cell12.e.0] Driver target count mismatch after strict resolver. "
        f"found={len(candidate_targets)} expected={EXPECTED_DRIVER_TARGET_COUNT_12E}. "
        f"target_source_mode={target_source_mode}. "
        f"resolution_reason={resolution_reason}. "
        f"explicit_found={len(explicit_targets)} fallback_found={len(fallback_targets)}. "
        f"selected_preview={candidate_targets[:80]}. "
        f"fallback_selected_preview={fallback_selected_preview}. "
        f"fallback_rejected_driver_like_preview={fallback_rejected_driver_like_preview}."
    )

IOT_DRIVER_COLS = sorted(candidate_targets)
IOT_DRIVER_TARGETS = list(IOT_DRIVER_COLS)

# ----------------------------------------------------------
# 8) Build/persist driver availability contract
# ----------------------------------------------------------
def _as_driver_mask_frame_12e0(obj, index, columns, name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
    elif isinstance(obj, dict):
        out = pd.DataFrame(obj)
    else:
        arr = np.asarray(obj)
        if arr.ndim != 2:
            raise RuntimeError(f"[Cell12.e.0] Cannot normalize {name}: shape={arr.shape}")
        out = pd.DataFrame(arr)

    if len(out) != len(index):
        raise RuntimeError(
            f"[Cell12.e.0] {name} row mismatch: got={len(out)} expected={len(index)}"
        )

    out.index = index
    out.columns = out.columns.astype(str)

    missing = sorted(set(columns) - set(out.columns))
    if missing:
        raise RuntimeError(f"[Cell12.e.0] {name} missing driver columns: {missing[:30]}")

    return out[columns].copy()

def _build_driver_availability_syn_12e0() -> tuple:
    for name in [
        "IOT_DRIVER_AVAIL_SYN",
        "DRIVER_AVAIL_SYN",
        "IOT_DRIVER_ACTIVITY_SYN",
        "ENTITY_DRIVER_ACTIVITY_SYN",
    ]:
        obj = globals().get(name, None)
        if obj is not None:
            try:
                mask_df = _as_driver_mask_frame_12e0(obj, df_te.index, IOT_DRIVER_COLS, name)
                mask_df = mask_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                mask_df = (mask_df > 0.5).astype(np.float32)
                return mask_df, {
                    "driver_availability_source": name,
                    "driver_availability_defaulted": False,
                    "driver_availability_policy": "explicit_upstream_synthetic_driver_availability_mask",
                    "TEST_values_used_for_availability": False,
                }
            except Exception:
                pass

    policy = str(CFG.get("cell12e0_driver_availability_default", "all_active")).strip().lower()
    if policy != "all_active":
        raise RuntimeError(
            "[Cell12.e.0] No explicit driver availability mask found and unsupported "
            f"cell12e0_driver_availability_default={policy!r}."
        )

    mask_df = pd.DataFrame(
        1.0,
        index=df_te.index,
        columns=IOT_DRIVER_COLS,
        dtype=np.float32,
    )
    return mask_df, {
        "driver_availability_source": "default_all_active",
        "driver_availability_defaulted": True,
        "driver_availability_policy": "driver_targets_full_grid_all_active",
        "TEST_values_used_for_availability": False,
    }

IOT_DRIVER_AVAIL_SYN_DF, driver_availability_meta = _build_driver_availability_syn_12e0()
IOT_DRIVER_AVAIL_SYN_DF.to_parquet(driver_avail_parquet, index=True)

availability_rows = []
for col in IOT_DRIVER_COLS:
    mask = pd.to_numeric(IOT_DRIVER_AVAIL_SYN_DF[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    active = mask > 0.5
    availability_rows.append({
        "col": col,
        "test_rows": int(N_TE),
        "driver_active_n": int(active.sum()),
        "driver_inactive_n": int((~active).sum()),
        "driver_active_rate": float(active.mean()) if active.size else np.nan,
        "availability_source": driver_availability_meta["driver_availability_source"],
        "availability_defaulted": bool(driver_availability_meta["driver_availability_defaulted"]),
        "availability_policy": driver_availability_meta["driver_availability_policy"],
        "TEST_values_used_for_availability": False,
    })

IOT_DRIVER_AVAILABILITY_CONTRACT_DF = pd.DataFrame(availability_rows)

# ----------------------------------------------------------
# 9) Build target contract
# ----------------------------------------------------------
def _driver_domain_contract_12e0(driver_type: str) -> dict:
    if driver_type == "binary_driver":
        return {
            "domain": "{0,1,NaN}",
            "nonnegative": True,
            "integer_required": True,
            "binary_required": True,
            "recommended_12e2_candidate": "regime-conditioned Markov",
        }
    if driver_type == "countlike_driver":
        return {
            "domain": "nonnegative integer count",
            "nonnegative": True,
            "integer_required": True,
            "binary_required": False,
            "recommended_12e2_candidate": "Negative Binomial / Poisson-Gamma",
        }
    if driver_type == "sparse_burst":
        return {
            "domain": "zero-inflated nonnegative count/intensity",
            "nonnegative": True,
            "integer_required": None,
            "binary_required": False,
            "recommended_12e2_candidate": "hurdle/zero-inflated Negative Binomial",
        }
    if driver_type == "event_session":
        return {
            "domain": "nonnegative session/event duration or count",
            "nonnegative": True,
            "integer_required": None,
            "binary_required": False,
            "recommended_12e2_candidate": "session replay with no-copy audit",
        }
    return {
        "domain": "nonnegative continuous intensity",
        "nonnegative": True,
        "integer_required": False,
        "binary_required": False,
        "recommended_12e2_candidate": "AR(1)+Student-t",
    }

contract_rows = []

for ordinal, col in enumerate(IOT_DRIVER_COLS):
    summary = _value_summary_12e0(col)
    driver_type, driver_type_reason = _classify_driver_type_12e0(col, summary)
    domain_contract = _driver_domain_contract_12e0(driver_type)

    train_vals = _to_num_array_12e0(df_tr, col)
    val_vals = _to_num_array_12e0(df_val, col)

    train_f = train_vals[np.isfinite(train_vals)]
    val_f = val_vals[np.isfinite(val_vals)]

    train_nonzero_rate = _nonzero_rate_12e0(train_f)
    val_nonzero_rate = _nonzero_rate_12e0(val_f)

    train_mean = float(np.mean(train_f)) if train_f.size else np.nan
    val_mean = float(np.mean(val_f)) if val_f.size else np.nan
    train_var = float(np.var(train_f)) if train_f.size else np.nan
    val_var = float(np.var(val_f)) if val_f.size else np.nan

    copy_risk_duplicate_rate = np.nan
    if train_f.size:
        vc = pd.Series(train_f).value_counts(normalize=True, dropna=True)
        copy_risk_duplicate_rate = float(vc.iloc[0]) if len(vc) else np.nan

    contract_rows.append({
        "ordinal": int(ordinal),
        "col": col,
        "namespace": "events_in_sec" if _is_events_in_sec_col_12e0(col) else "iot",
        "entity": _entity_from_driver_col_12e0(col),
        "measurement_name": _measurement_from_col_12e0(col),
        "target_role": "driver_target",
        "branch": "12.e",
        "driver_type": driver_type,
        "driver_type_reason": driver_type_reason,
        "domain_contract": domain_contract["domain"],
        "nonnegative_required": domain_contract["nonnegative"],
        "integer_required": domain_contract["integer_required"],
        "binary_required": domain_contract["binary_required"],
        "recommended_12e2_candidate": domain_contract["recommended_12e2_candidate"],
        "target_source_mode": target_source_mode,
        "target_resolution_reason": resolution_reason,
        "train_observed_n": int(train_f.size),
        "val_observed_n": int(val_f.size),
        "train_nonzero_rate": train_nonzero_rate,
        "val_nonzero_rate": val_nonzero_rate,
        "abs_train_val_nonzero_rate_delta": (
            abs(train_nonzero_rate - val_nonzero_rate)
            if np.isfinite(train_nonzero_rate) and np.isfinite(val_nonzero_rate)
            else np.nan
        ),
        "train_mean": train_mean,
        "val_mean": val_mean,
        "abs_train_val_mean_delta": (
            abs(train_mean - val_mean)
            if np.isfinite(train_mean) and np.isfinite(val_mean)
            else np.nan
        ),
        "train_var": train_var,
        "val_var": val_var,
        "train_variance_to_mean": (
            train_var / max(train_mean, 1e-12)
            if np.isfinite(train_var) and np.isfinite(train_mean) and train_mean > 0
            else np.nan
        ),
        "val_variance_to_mean": (
            val_var / max(val_mean, 1e-12)
            if np.isfinite(val_var) and np.isfinite(val_mean) and val_mean > 0
            else np.nan
        ),
        "trainval_observed_n": int(summary["trainval_observed_n"]),
        "trainval_nonzero_rate": summary["trainval_nonzero_rate"],
        "trainval_mean": summary["trainval_mean"],
        "trainval_var": summary["trainval_var"],
        "trainval_min": summary["trainval_min"],
        "trainval_max": summary["trainval_max"],
        "trainval_unique_n_capped": int(summary["trainval_unique_n_capped"]),
        "integer_fraction": summary["integer_fraction"],
        "is_binary_values": bool(summary["is_binary_values"]),
        "is_nonnegative": bool(summary["is_nonnegative"]),
        "variance_to_mean": summary["variance_to_mean"],
        "zero_inflated": bool(summary["zero_inflated"]),
        "copy_risk_duplicate_rate": copy_risk_duplicate_rate,
        "copy_risk_high": bool(
            np.isfinite(copy_risk_duplicate_rate)
            and copy_risk_duplicate_rate >= float(CFG.get("cell12e0_driver_copy_risk_duplicate_rate_warn", 0.98))
        ),
        "selection_criteria_sparsity": bool(summary["zero_inflated"] or (np.isfinite(summary["trainval_nonzero_rate"]) and summary["trainval_nonzero_rate"] < 0.50)),
        "selection_criteria_event_semantics": True,
        "selection_criteria_temporal_alignment_robustness": True,
        "selection_criteria_coupling_relevance": True,
        "fit_policy": "TRAIN_only",
        "selection_policy": "VAL_only",
        "test_policy": "TEST_schema_length_index_only_until_12e6_QA",
        "TEST_values_used_in_12e0": False,
    })

IOT_DRIVER_TARGET_CONTRACT_DF = pd.DataFrame(contract_rows)

if len(IOT_DRIVER_TARGET_CONTRACT_DF) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        "[Cell12.e.0] Driver contract row mismatch: "
        f"rows={len(IOT_DRIVER_TARGET_CONTRACT_DF)} targets={len(IOT_DRIVER_COLS)}"
    )

if IOT_DRIVER_TARGET_CONTRACT_DF["col"].duplicated().any():
    dupes = IOT_DRIVER_TARGET_CONTRACT_DF.loc[
        IOT_DRIVER_TARGET_CONTRACT_DF["col"].duplicated(), "col"
    ].astype(str).tolist()
    raise RuntimeError(f"[Cell12.e.0] Duplicate driver targets: {dupes[:20]}")

fatal_domain_rows = []
for _, r in IOT_DRIVER_TARGET_CONTRACT_DF.iterrows():
    col = str(r["col"])
    if not bool(r["is_nonnegative"]):
        fatal_domain_rows.append({"col": col, "reason": "driver_not_nonnegative_trainval"})
    if bool(r["integer_required"]) and float(r["integer_fraction"]) < 0.999:
        fatal_domain_rows.append({"col": col, "reason": "integer_required_but_trainval_not_integer"})
    if bool(r["binary_required"]) and not bool(r["is_binary_values"]):
        fatal_domain_rows.append({"col": col, "reason": "binary_required_but_trainval_not_binary"})

if fatal_domain_rows:
    fatal_csv = os.path.join(REPORT_DIR, "cell12e0_driver_fatal_domain_errors.csv")
    pd.DataFrame(fatal_domain_rows).to_csv(fatal_csv, index=False)
    raise RuntimeError(
        f"[Cell12.e.0] Fatal driver domain contract errors: n={len(fatal_domain_rows)}. "
        f"See {fatal_csv}"
    )

# ----------------------------------------------------------
# 10) Save diagnostics/artifacts
# ----------------------------------------------------------
driver_type_counts = (
    IOT_DRIVER_TARGET_CONTRACT_DF["driver_type"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)
namespace_counts = (
    IOT_DRIVER_TARGET_CONTRACT_DF["namespace"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

resolution_rows = [
    {"metric": "expected_driver_targets", "value": int(EXPECTED_DRIVER_TARGET_COUNT_12E)},
    {"metric": "explicit_driver_targets_n", "value": int(len(explicit_targets))},
    {"metric": "fallback_driver_targets_n", "value": int(len(fallback_targets))},
    {"metric": "selected_driver_targets_n", "value": int(len(IOT_DRIVER_COLS))},
    {"metric": "target_source_mode", "value": target_source_mode},
    {"metric": "resolution_reason", "value": resolution_reason},
    {"metric": "TEST_values_used", "value": False},
]
for k, v in driver_type_counts.items():
    resolution_rows.append({"metric": f"driver_type_count::{k}", "value": int(v)})
for k, v in namespace_counts.items():
    resolution_rows.append({"metric": f"namespace_count::{k}", "value": int(v)})

pd.DataFrame(resolution_rows).to_csv(driver_resolution_diag_csv, index=False)

IOT_DRIVER_TARGET_CONTRACT_DF.to_csv(driver_target_contract_csv, index=False)
IOT_DRIVER_AVAILABILITY_CONTRACT_DF.to_csv(driver_availability_contract_csv, index=False)

targets_payload = {
    "cell": "12.e.0",
    "version": CELL12E0_VERSION,
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "expected_driver_targets_total": int(EXPECTED_DRIVER_TARGET_COUNT_12E),
    "driver_target_count_matches_expected": bool(len(IOT_DRIVER_COLS) == EXPECTED_DRIVER_TARGET_COUNT_12E),
    "target_source_mode": target_source_mode,
    "target_resolution_reason": resolution_reason,
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "df_te_used_for_index_length_schema_only": True,
    "targets": IOT_DRIVER_COLS,
    "driver_type_counts": driver_type_counts,
    "namespace_counts": namespace_counts,
}

_write_json_12e0(driver_targets_json, targets_payload)
_write_json_12e0(driver_targets_report_json, targets_payload)

contract = {
    "cell": "12.e.0",
    "version": CELL12E0_VERSION,
    "role": "driver_contracts_and_12e_branch_plan",
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "expected_driver_targets_total": int(EXPECTED_DRIVER_TARGET_COUNT_12E),
    "driver_target_count_matches_expected": bool(len(IOT_DRIVER_COLS) == EXPECTED_DRIVER_TARGET_COUNT_12E),
    "target_source_mode": target_source_mode,
    "target_resolution_reason": resolution_reason,
    "explicit_metadata": explicit_meta,
    "explicit_targets_n": int(len(explicit_targets)),
    "fallback_trainval_inferred_n": int(len(fallback_targets)),
    "all_targets_in_train_schema": bool(all(c in df_tr.columns for c in IOT_DRIVER_COLS)),
    "all_targets_in_val_schema": bool(all(c in df_val.columns for c in IOT_DRIVER_COLS)),
    "all_targets_in_test_schema": bool(all(c in df_te.columns for c in IOT_DRIVER_COLS)),
    "driver_type_counts": driver_type_counts,
    "namespace_counts": namespace_counts,
    "driver_availability": driver_availability_meta,
    "driver_availability_shape": list(IOT_DRIVER_AVAIL_SYN_DF.shape),
    "driver_availability_no_test_values_used": True,
    "cell12e_plan": CELL12E_PLAN,
    "paper_selection_principles": [
        "sparsity",
        "event semantics",
        "temporal alignment robustness",
        "coupling relevance",
    ],
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "train_val_values_used_for_target_validation": True,
    "df_te_used_for_index_length_schema_only": True,
    "events_in_sec_namespace_supported": True,
    "outputs": {
        "driver_targets_json": driver_targets_json,
        "driver_targets_report_json": driver_targets_report_json,
        "driver_contract_json": driver_contract_json,
        "driver_contract_canonical_json": driver_contract_canonical_json,
        "driver_target_contract_csv": driver_target_contract_csv,
        "driver_availability_contract_csv": driver_availability_contract_csv,
        "driver_avail_parquet": driver_avail_parquet,
        "driver_resolution_diag_csv": driver_resolution_diag_csv,
        "driver_explicit_diag_csv": driver_explicit_diag_csv,
        "driver_fallback_inference_csv": driver_fallback_inference_csv,
        "driver_manifest_json": driver_manifest_json,
    },
}

_write_json_12e0(driver_contract_json, contract)
_write_json_12e0(driver_contract_canonical_json, contract)

manifest = {
    "cell": "12.e.0",
    "version": CELL12E0_VERSION,
    "created_outputs": contract["outputs"],
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "driver_type_counts": driver_type_counts,
    "namespace_counts": namespace_counts,
    "cell12e_plan": CELL12E_PLAN,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": False,
    },
}

_write_json_12e0(driver_manifest_json, manifest)

hashes = {
    "driver_targets_json_sha256": _sha256_file_12e0(driver_targets_json),
    "driver_targets_report_json_sha256": _sha256_file_12e0(driver_targets_report_json),
    "driver_target_contract_csv_sha256": _sha256_file_12e0(driver_target_contract_csv),
    "driver_availability_contract_csv_sha256": _sha256_file_12e0(driver_availability_contract_csv),
    "driver_avail_parquet_sha256": _sha256_file_12e0(driver_avail_parquet),
    "driver_resolution_diag_csv_sha256": _sha256_file_12e0(driver_resolution_diag_csv),
    "driver_explicit_diag_csv_sha256": _sha256_file_12e0(driver_explicit_diag_csv),
    "driver_fallback_inference_csv_sha256": _sha256_file_12e0(driver_fallback_inference_csv),
    "driver_contract_json_sha256": _sha256_file_12e0(driver_contract_json),
    "driver_contract_canonical_json_sha256": _sha256_file_12e0(driver_contract_canonical_json),
    "driver_manifest_json_sha256": _sha256_file_12e0(driver_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12e0(driver_contract_json, contract)
_write_json_12e0(driver_contract_canonical_json, contract)
_write_json_12e0(driver_manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals for 12.e.1+
# ----------------------------------------------------------
globals()["CELL12E0_VERSION"] = CELL12E0_VERSION
globals()["IOT_DRIVER_COLS"] = IOT_DRIVER_COLS
globals()["IOT_DRIVER_TARGETS"] = IOT_DRIVER_TARGETS
globals()["IOT_DRIVER_TARGET_CONTRACT_DF"] = IOT_DRIVER_TARGET_CONTRACT_DF
globals()["IOT_DRIVER_AVAIL_SYN"] = IOT_DRIVER_AVAIL_SYN_DF
globals()["IOT_DRIVER_AVAIL_SYN_DF"] = IOT_DRIVER_AVAIL_SYN_DF
globals()["IOT_DRIVER_AVAILABILITY_CONTRACT_DF"] = IOT_DRIVER_AVAILABILITY_CONTRACT_DF
globals()["CELL12E_DRIVER_CONTRACT"] = contract
globals()["CELL12E_PLAN"] = CELL12E_PLAN
globals()["CELL12E_DRIVER_TARGETS_JSON"] = driver_targets_json
globals()["CELL12E_DRIVER_CONTRACT_JSON"] = driver_contract_json
globals()["CELL12E_DRIVER_CONTRACT_CANONICAL_JSON"] = driver_contract_canonical_json
globals()["CELL12E0_DRIVER_TARGET_CONTRACT_CSV"] = driver_target_contract_csv
globals()["CELL12E0_DRIVER_AVAILABILITY_CONTRACT_CSV"] = driver_availability_contract_csv
globals()["CELL12E0_DRIVER_AVAIL_PARQUET"] = driver_avail_parquet
globals()["CELL12E0_DRIVER_RESOLUTION_DIAG_CSV"] = driver_resolution_diag_csv
globals()["CELL12E0_DRIVER_EXPLICIT_DIAG_CSV"] = driver_explicit_diag_csv
globals()["CELL12E0_DRIVER_FALLBACK_INFERENCE_CSV"] = driver_fallback_inference_csv
globals()["CELL12E0_DRIVER_MANIFEST_JSON"] = driver_manifest_json

log(
    "[Cell12.e.0] Driver contract built | "
    f"targets={len(IOT_DRIVER_COLS)} | expected={EXPECTED_DRIVER_TARGET_COUNT_12E} | "
    f"source_mode={target_source_mode} | resolution={resolution_reason} | "
    f"availability_source={driver_availability_meta['driver_availability_source']}"
)
log(f"[Cell12.e.0] Driver namespace counts | {namespace_counts}")
log(f"[Cell12.e.0] Driver type counts | {driver_type_counts}")
log(f"[Cell12.e.0] Saved driver targets JSON: {driver_targets_json}")
log(f"[Cell12.e.0] Saved driver contract JSON: {driver_contract_json}")
log(f"[Cell12.e.0] Saved canonical contract JSON: {driver_contract_canonical_json}")
log(f"[Cell12.e.0] Saved target contract CSV: {driver_target_contract_csv} | rows={len(IOT_DRIVER_TARGET_CONTRACT_DF)}")
log(f"[Cell12.e.0] Saved availability contract CSV: {driver_availability_contract_csv} | rows={len(IOT_DRIVER_AVAILABILITY_CONTRACT_DF)}")
log(f"[Cell12.e.0] Saved availability parquet: {driver_avail_parquet} | shape={IOT_DRIVER_AVAIL_SYN_DF.shape}")
log(f"[Cell12.e.0] Saved explicit diagnostic CSV: {driver_explicit_diag_csv} | rows={len(explicit_df)}")
log(f"[Cell12.e.0] Saved fallback inference CSV: {driver_fallback_inference_csv} | rows={len(fallback_df)}")
log(f"[Cell12.e.0] Saved resolution diagnostic CSV: {driver_resolution_diag_csv}")
log(
    "[Cell12.e.0] Contract flags | TEST_real_values_used=False | "
    "selection_done_here=False | generator_fit_done_here=False | "
    "df_te_used_for_index_length_schema_only=True | events_in_sec_namespace_supported=True"
)
log("--- END: Cell 12.e.0 - Driver contracts (v1.1 strict resolver, events_in_sec-aware) ---")

gc.collect()
