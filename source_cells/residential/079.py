# ==========================================================
# CELL 12.f.0 - IoT observability/mask contracts
# v1.1 STUDY-THESIS strict observability contract resolver, target-leak hardened
#
# Role:
#   - Define IoT observability / mask targets for Q3.
#   - Resolve:
#       entity_obs_cols
#       entity_stale_cols
#       value availability masks
#       feature/entity observed indicators
#   - Link observability masks to generated IoT target families:
#       12.c continuous/value IoT targets
#       12.d binary IoT state targets
#       12.e sparse event-driver targets
#   - Build contracts for 12.f.1-12.f.4.
#
# Strict rules:
#   - Do NOT generate masks here.
#   - Do NOT fit mask models here.
#   - Do NOT select mask models here.
#   - Do NOT read TEST mask/target values for calibration.
#   - df_te is used only for schema/index/length alignment.
#
# Outputs:
#   artifacts/IOT_OBSERVABILITY_TARGETS.json
#   reports/IOT_OBSERVABILITY_TARGETS.json
#   reports/cell12f_observability_contract.json
#   reports/cell12f0_observability_target_contract.csv
#   reports/cell12f0_observability_family_contract.csv
#   reports/cell12f0_observability_resolution_diagnostic.csv
#   reports/cell12f0_observability_fallback_schema_inference.csv
#   artifacts/cell12f0_observability_manifest.json
# ==========================================================

log("--- START: Cell 12.f.0 - IoT observability contracts (v1.1 strict resolver, target-leak hardened) ---")

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
_required_12f0 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
]
_missing_12f0 = [k for k in _required_12f0 if k not in globals()]
if _missing_12f0:
    raise RuntimeError(f"[Cell12.f.0] Missing required globals: {_missing_12f0}")

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
        f"[Cell12.f.0] Invalid split lengths: train={N_TR}, val={N_VAL}, test={N_TE}"
    )

CELL12F0_VERSION = "cell12f0_observability_contract_strict_v1_1_target_leak_hardened"

CFG["cell12f0_version"] = CELL12F0_VERSION
CFG["cell12f0_TEST_real_values_used"] = False
CFG["cell12f0_selection_done_here"] = False
CFG["cell12f0_generator_fit_done_here"] = False
CFG["cell12f0_mask_materialization_done_here"] = False
CFG["cell12f0_train_val_values_used_for_target_validation"] = True
CFG["cell12f0_df_te_used_for_schema_index_length_only"] = True
CFG["cell12f0_quality_dimension"] = "Q3_observability"

CFG.setdefault("cell12f0_allow_empty_observability_targets", False)
CFG.setdefault("cell12f0_min_trainval_observed_n", 1)
CFG.setdefault("cell12f0_binary_value_tolerance", 1e-6)
CFG.setdefault("cell12f0_mask_near_constant_threshold", 0.995)
CFG.setdefault("cell12f0_mask_sparse_threshold", 0.05)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
obs_targets_json = os.path.join(ARTDIR, "IOT_OBSERVABILITY_TARGETS.json")
obs_targets_report_json = os.path.join(REPORT_DIR, "IOT_OBSERVABILITY_TARGETS.json")
obs_contract_json = os.path.join(REPORT_DIR, "cell12f_observability_contract.json")
obs_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12f0_observability_contract_v1_1_THESIS.json")
obs_explicit_metadata_diag_csv = os.path.join(REPORT_DIR, "cell12f0_observability_explicit_metadata_diagnostic.csv")
obs_target_contract_csv = os.path.join(REPORT_DIR, "cell12f0_observability_target_contract.csv")
obs_family_contract_csv = os.path.join(REPORT_DIR, "cell12f0_observability_family_contract.csv")
obs_resolution_diag_csv = os.path.join(REPORT_DIR, "cell12f0_observability_resolution_diagnostic.csv")
obs_fallback_schema_csv = os.path.join(REPORT_DIR, "cell12f0_observability_fallback_schema_inference.csv")
obs_manifest_json = os.path.join(ARTDIR, "cell12f0_observability_manifest.json")

# ----------------------------------------------------------
# 2) Generic helpers
# ----------------------------------------------------------
def _json_sanitize_12f0(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12f0(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12f0(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12f0(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12f0(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12f0(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12f0(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12f0(obj.to_dict())
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

def _write_json_12f0(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12f0(payload), f, indent=2, sort_keys=True)

def _sha256_file_12f0(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_csv_if_exists_12f0(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _safe_float_12f0(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_bool_12f0(x, default=False) -> bool:
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

def _to_num_array_12f0(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _finite_trainval_values_12f0(col: str) -> np.ndarray:
    if col not in df_tr.columns or col not in df_val.columns:
        return np.asarray([], dtype=np.float64)
    tr = _to_num_array_12f0(df_tr, col)
    va = _to_num_array_12f0(df_val, col)
    x = np.concatenate([tr, va])
    return x[np.isfinite(x)]

def _is_binary_trainval_12f0(col: str) -> tuple:
    x = _finite_trainval_values_12f0(col)
    tol = float(CFG.get("cell12f0_binary_value_tolerance", 1e-6))

    if x.size == 0:
        return False, {
            "trainval_observed_n": 0,
            "trainval_obs_rate": np.nan,
            "trainval_unique_n": 0,
            "trainval_unique_preview": [],
            "binary_check_reason": "empty_trainval",
        }

    near0 = np.abs(x - 0.0) <= tol
    near1 = np.abs(x - 1.0) <= tol
    ok = bool(np.all(near0 | near1))
    unique_vals = np.unique(np.round(x, 6))

    return ok, {
        "trainval_observed_n": int(x.size),
        "trainval_obs_rate": float(np.mean(near1)),
        "trainval_unique_n": int(unique_vals.size),
        "trainval_unique_preview": [float(v) for v in unique_vals[:20]],
        "binary_check_reason": "" if ok else "non_binary_trainval_values",
    }

def _entity_from_iot_col_12f0(col: str) -> str:
    s = str(col)
    if s.startswith("iot__entity_obs__"):
        return s.split("iot__entity_obs__", 1)[1]
    if s.startswith("iot__entity_stale__"):
        return s.split("iot__entity_stale__", 1)[1]
    if s.startswith("iot__"):
        s = s[5:]
    return s.split("__")[0] if "__" in s else s

def _entity_from_target_or_obs_col_12f0(col: str) -> str:
    """Entity parser for both iot__... observability masks and events_in_sec__... driver targets."""
    s = str(col)
    parts = [p for p in s.split("__") if p != ""]
    if not parts:
        return s

    if s.startswith("iot__entity_obs__"):
        return s.split("iot__entity_obs__", 1)[1]
    if s.startswith("iot__entity_stale__"):
        return s.split("iot__entity_stale__", 1)[1]

    if parts[0] == "iot":
        return parts[1] if len(parts) > 1 else s

    if parts[0] == "events_in_sec":
        if len(parts) >= 3 and parts[1] in {"entity", "device", "iot"}:
            return parts[2]
        if len(parts) >= 3 and parts[1] == "feat":
            return parts[2]
        return parts[1] if len(parts) > 1 else s

    return parts[0]

def _is_iot_col_12f0(col: str) -> bool:
    return str(col).startswith("iot__")

def _is_observability_name_12f0(col: str) -> tuple:
    s = str(col).lower()

    if s.startswith("iot__entity_obs__"):
        return True, "entity_obs_col"

    if s.startswith("iot__entity_stale__"):
        return True, "entity_stale_col"

    if s in {
        "iot__obs_present",
        "iot__any_update_raw",
        "iot__tier_present",
    }:
        return True, "global_iot_observability_indicator"

    obs_tokens = [
        "obs_present",
        "observed",
        "is_observed",
        "present_mask",
        "availability",
        "available",
        "is_available",
        "mask_observed",
    ]

    stale_tokens = [
        "stale",
        "staleness",
        "stale_flag",
        "age_since_update",
        "seconds_since_update",
        "last_seen_age",
    ]

    if any(tok in s for tok in obs_tokens):
        return True, "feature_or_value_observed_indicator"

    if any(tok in s for tok in stale_tokens):
        return True, "feature_or_entity_stale_indicator"

    return False, "not_observability_name"

def _obs_kind_12f0(col: str) -> str:
    s = str(col).lower()

    if s.startswith("iot__entity_obs__"):
        return "entity_obs"

    if s.startswith("iot__entity_stale__"):
        return "entity_stale"

    if "stale" in s or "staleness" in s or "age_since_update" in s or "last_seen_age" in s:
        return "stale_or_age"

    if "availability" in s or "available" in s:
        return "availability_mask"

    if "obs_present" in s or "observed" in s or "present_mask" in s:
        return "observed_indicator"

    if s == "iot__any_update_raw":
        return "global_update_indicator"

    if s == "iot__tier_present":
        return "tier_present_indicator"

    return "observability_indicator"

def _recommended_mask_model_12f0(obs_rate: float, obs_kind: str) -> str:
    if not np.isfinite(obs_rate):
        return "constant-rate fallback"

    near_constant_threshold = float(CFG.get("cell12f0_mask_near_constant_threshold", 0.995))
    sparse_threshold = float(CFG.get("cell12f0_mask_sparse_threshold", 0.05))

    if obs_rate >= near_constant_threshold or obs_rate <= (1.0 - near_constant_threshold):
        return "constant-rate fallback"

    if obs_rate <= sparse_threshold or obs_rate >= 1.0 - sparse_threshold:
        return "regime-conditioned Markov mask"

    if obs_kind in {"entity_stale", "stale_or_age"}:
        return "semi-Markov mask"

    return "logistic autoregressive mask model"

def _mask_role_for_target_12f0(col: str) -> str:
    s = str(col).lower()

    if s.startswith("iot__entity_obs__"):
        return "entity_observed_indicator"

    if s.startswith("iot__entity_stale__"):
        return "entity_stale_indicator"

    if "stale" in s or "staleness" in s:
        return "stale_indicator"

    if "availability" in s or "available" in s:
        return "availability_mask"

    if "obs_present" in s or "observed" in s or "present_mask" in s:
        return "observed_indicator"

    if s == "iot__any_update_raw":
        return "global_update_indicator"

    if s == "iot__tier_present":
        return "tier_present_indicator"

    return "observability_indicator"

# ----------------------------------------------------------
# 3) Load generated target-family references
# ----------------------------------------------------------
def _load_target_family_cols_12f0() -> dict:
    fam = {
        "continuous_value_12c": [],
        "binary_state_12d": [],
        "driver_event_12e": [],
    }

    # 12.c
    for name in [
        "IOT_VALUE_COLS",
        "VALUE_COLS",
        "CONT_VALUE_COLS",
        "CELL12C_VALUE_COLS",
        "IOT_CONT_VALUE_COLS",
    ]:
        obj = globals().get(name, None)
        if obj is not None:
            try:
                fam["continuous_value_12c"].extend(map(str, list(obj)))
            except Exception:
                pass

    for path in [
        os.path.join(REPORT_DIR, "cell12c0_value_contract.csv"),
        os.path.join(REPORT_DIR, "cell12c5_publication_column_registry.csv"),
    ]:
        d = _read_csv_if_exists_12f0(path)
        if isinstance(d, pd.DataFrame) and len(d):
            for cand in ["col", "column", "target", "target_col", "value_col"]:
                if cand in d.columns:
                    fam["continuous_value_12c"].extend(d[cand].dropna().astype(str).tolist())
                    break

    # 12.d
    for name in [
        "IOT_BINARY_COLS",
        "IOT_BINARY_TARGETS",
        "BINARY_TARGET_COLS",
        "CELL12D_BINARY_TARGET_COLS",
    ]:
        obj = globals().get(name, None)
        if obj is not None:
            try:
                fam["binary_state_12d"].extend(map(str, list(obj)))
            except Exception:
                pass

    for path in [
        os.path.join(REPORT_DIR, "cell12d0_binary_target_contract.csv"),
        os.path.join(REPORT_DIR, "cell12d5_binary_publication_column_registry.csv"),
    ]:
        d = _read_csv_if_exists_12f0(path)
        if isinstance(d, pd.DataFrame) and len(d):
            for cand in ["col", "column", "target", "target_col"]:
                if cand in d.columns:
                    fam["binary_state_12d"].extend(d[cand].dropna().astype(str).tolist())
                    break

    # 12.e
    for name in [
        "IOT_DRIVER_COLS",
        "IOT_DRIVER_TARGETS",
        "DRIVER_TARGET_COLS",
        "CELL12E_DRIVER_TARGETS",
    ]:
        obj = globals().get(name, None)
        if obj is not None:
            try:
                fam["driver_event_12e"].extend(map(str, list(obj)))
            except Exception:
                pass

    for path in [
        os.path.join(REPORT_DIR, "cell12e0_driver_target_contract.csv"),
        os.path.join(REPORT_DIR, "cell12e5_driver_publication_column_registry.csv"),
    ]:
        d = _read_csv_if_exists_12f0(path)
        if isinstance(d, pd.DataFrame) and len(d):
            for cand in ["col", "column", "target", "target_col", "driver_col"]:
                if cand in d.columns:
                    fam["driver_event_12e"].extend(d[cand].dropna().astype(str).tolist())
                    break

    for k in list(fam.keys()):
        fam[k] = sorted(list(dict.fromkeys([c for c in fam[k] if isinstance(c, str) and c])))

    return fam

TARGET_FAMILIES_12F0 = _load_target_family_cols_12f0()

DIRECT_OBSERVABILITY_EXCLUDE_TARGET_COLS_12F0 = set()
for _fam_cols in TARGET_FAMILIES_12F0.values():
    DIRECT_OBSERVABILITY_EXCLUDE_TARGET_COLS_12F0.update(map(str, _fam_cols))

def _is_generated_target_col_12f0(col: str) -> bool:
    return str(col) in DIRECT_OBSERVABILITY_EXCLUDE_TARGET_COLS_12F0

log(
    "[Cell12.f.0] Loaded generated IoT target family refs | "
    f"12c_continuous={len(TARGET_FAMILIES_12F0['continuous_value_12c'])} | "
    f"12d_binary={len(TARGET_FAMILIES_12F0['binary_state_12d'])} | "
    f"12e_driver={len(TARGET_FAMILIES_12F0['driver_event_12e'])}"
)

# ----------------------------------------------------------
# 4) Explicit metadata discovery
# ----------------------------------------------------------
def _standardize_col_field_12f0(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "col" not in out.columns:
        for cand in ["column", "name", "feature", "feature_name", "target_col", "iot_col", "mask_col"]:
            if cand in out.columns:
                out = out.rename(columns={cand: "col"})
                break
    if "col" in out.columns:
        out["col"] = out["col"].astype(str)
    return out

def _candidate_observability_frames_12f0() -> list:
    frames = []

    global_names = [
        "IOT_OBSERVABILITY_COLS",
        "IOT_OBS_COLS",
        "IOT_MASK_COLS",
        "IOT_AVAILABILITY_COLS",
        "ENTITY_OBS_COLS",
        "ENTITY_STALE_COLS",
        "VALUE_AVAIL_COLS",
        "VALUE_AVAIL_SYN",
        "VALUE_AVAIL_RATE_MAP",
        "IOT_VALUE_AVAILABILITY_CONTRACT_DF",
        "IOT_BINARY_AVAILABILITY_CONTRACT_DF",
        "IOT_DRIVER_AVAILABILITY_CONTRACT_DF",
    ]

    for name in global_names:
        obj = globals().get(name, None)
        if obj is None:
            continue
        if isinstance(obj, pd.DataFrame) and len(obj):
            frames.append((f"global:{name}", obj.copy()))
        elif isinstance(obj, (list, tuple, set, pd.Index)):
            frames.append((f"global:{name}", pd.DataFrame({"col": list(map(str, obj)), "explicit_observability_list": True})))
        elif isinstance(obj, dict):
            frames.append((f"global:{name}", pd.DataFrame({"col": list(map(str, obj.keys())), "explicit_observability_dict": True})))

    possible_paths = [
        os.path.join(REPORT_DIR, "cell12c0_value_contract.csv"),
        os.path.join(REPORT_DIR, "cell12d0_binary_availability_contract.csv"),
        os.path.join(REPORT_DIR, "cell12e0_driver_availability_contract.csv"),
        os.path.join(REPORT_DIR, "cell12f_observability_targets.csv"),
        os.path.join(REPORT_DIR, "IOT_OBSERVABILITY_TARGETS.csv"),
    ]

    for path in possible_paths:
        d = _read_csv_if_exists_12f0(path)
        if isinstance(d, pd.DataFrame) and len(d):
            frames.append((f"report:{os.path.basename(path)}", d.copy()))

    return frames

def _extract_observability_targets_from_metadata_12f0() -> tuple:
    frames = _candidate_observability_frames_12f0()
    rows = []
    targets = []

    role_cols = [
        "primary_owner",
        "owner",
        "target_owner",
        "role",
        "target_role",
        "column_role",
        "generation_role",
        "branch",
        "pipeline_branch",
        "mask_role",
    ]
    type_cols = [
        "target_type",
        "type",
        "semantic_type",
        "value_type",
        "domain_type",
        "kind",
        "family",
        "stage12f_kind",
    ]

    for source_name, raw in frames:
        df = _standardize_col_field_12f0(raw)

        # Some availability-contract reports have mask_col instead of col.
        if "col" not in df.columns and "mask_col" in df.columns:
            df["col"] = df["mask_col"].astype(str)

        if "col" not in df.columns:
            continue

        for _, r in df.iterrows():
            col = str(r.get("col", ""))
            if not col:
                continue

            # Accept only real schema columns for direct observability targets.
            # Generated availability DataFrames may be target-column-shaped,
            # handled later as family contracts, not direct mask columns.
            if col not in df_tr.columns or col not in df_val.columns or col not in df_te.columns:
                continue

            true_obs_name, true_obs_reason = _is_observability_name_12f0(col)
            is_generated_target = _is_generated_target_col_12f0(col)

            if is_generated_target and true_obs_reason not in {
                "entity_obs_col",
                "entity_stale_col",
                "global_iot_observability_indicator",
                "feature_or_value_observed_indicator",
                "feature_or_entity_stale_indicator",
            }:
                continue

            name_hit, name_reason = true_obs_name, true_obs_reason
            row_text = " ".join(str(v).strip().lower() for v in r.values)
            explicit_hit = _safe_bool_12f0(r.get("explicit_observability_list", False), False) or _safe_bool_12f0(r.get("explicit_observability_dict", False), False)

            role_hit = any(
                any(tok in str(r.get(c, "")).strip().lower() for tok in [
                    "observability",
                    "obs",
                    "mask",
                    "availability",
                    "stale",
                    "present",
                ])
                for c in role_cols
                if c in df.columns
            )

            type_hit = any(
                any(tok in str(r.get(c, "")).strip().lower() for tok in [
                    "observability",
                    "obs",
                    "mask",
                    "availability",
                    "stale",
                    "present",
                ])
                for c in type_cols
                if c in df.columns
            )

            text_hit = any(tok in row_text for tok in [
                "observability",
                "obs_present",
                "mask",
                "availability",
                "stale",
                "present_mask",
            ])

            selected = bool(name_hit or explicit_hit or role_hit or type_hit or text_hit)

            if selected:
                targets.append(col)
                rows.append({
                    "col": col,
                    "source": source_name,
                    "name_hit": bool(name_hit),
                    "name_reason": name_reason,
                    "explicit_hit": bool(explicit_hit),
                    "role_hit": bool(role_hit),
                    "type_hit": bool(type_hit),
                    "text_hit": bool(text_hit),
                    "reason": "explicit_or_registry_observability_match",
                    "TEST_values_used": False,
                })

    out_df = pd.DataFrame(rows)
    targets = sorted(list(dict.fromkeys(map(str, targets))))
    meta = {
        "observability_metadata_sources_checked_n": int(len(frames)),
        "observability_metadata_targets_n": int(len(targets)),
    }
    return targets, out_df, meta

# ----------------------------------------------------------
# 5) Fallback schema inference
# ----------------------------------------------------------
def _fallback_infer_observability_targets_schema_trainval_12f0() -> tuple:
    all_schema_cols = (
        set(map(str, df_tr.columns))
        & set(map(str, df_val.columns))
        & set(map(str, df_te.columns))
    )

    candidate_cols = sorted(c for c in all_schema_cols if _is_iot_col_12f0(c))

    rows = []
    targets = []

    for col in candidate_cols:
        if _is_generated_target_col_12f0(col):
            rows.append({
                "col": col,
                "entity": _entity_from_iot_col_12f0(col),
                "obs_kind": "excluded_generated_target",
                "mask_role": "not_direct_observability_target",
                "name_hit": False,
                "name_reason": "generated_target_excluded_from_direct_observability",
                "fallback_selected": False,
                "is_binary_trainval": False,
                "is_age_like_nonnegative": False,
                "recommended_mask_model": "not_applicable_generated_target",
                "TEST_values_used_for_fallback": False,
                "df_te_used_schema_only": True,
                "trainval_observed_n": 0,
                "trainval_obs_rate": np.nan,
                "trainval_unique_n": 0,
                "trainval_unique_preview": [],
                "binary_check_reason": "generated_target_excluded",
            })
            continue

        name_hit, name_reason = _is_observability_name_12f0(col)

        binary_ok, binary_audit = _is_binary_trainval_12f0(col)
        obs_kind = _obs_kind_12f0(col)

        # Staleness/age columns may be non-binary; keep them as stale_or_age
        # only if explicitly named and nonnegative.
        x = _finite_trainval_values_12f0(col)
        nonnegative = bool(x.size > 0 and np.nanmin(x) >= -1e-9)

        is_age_like = bool(obs_kind == "stale_or_age" and not binary_ok)

        selected = bool(
            name_hit
            and int(binary_audit.get("trainval_observed_n", 0)) >= int(CFG.get("cell12f0_min_trainval_observed_n", 1))
            and (binary_ok or is_age_like)
        )

        if selected:
            targets.append(col)

        obs_rate = _safe_float_12f0(binary_audit.get("trainval_obs_rate"), np.nan)

        rows.append({
            "col": col,
            "entity": _entity_from_iot_col_12f0(col),
            "obs_kind": obs_kind,
            "mask_role": _mask_role_for_target_12f0(col),
            "name_hit": bool(name_hit),
            "name_reason": name_reason,
            "fallback_selected": bool(selected),
            "is_binary_trainval": bool(binary_ok),
            "is_age_like_nonnegative": bool(is_age_like and nonnegative),
            "recommended_mask_model": _recommended_mask_model_12f0(obs_rate, obs_kind),
            "TEST_values_used_for_fallback": False,
            "df_te_used_schema_only": True,
            **binary_audit,
        })

    audit_df = pd.DataFrame(rows)
    targets = sorted(list(dict.fromkeys(map(str, targets))))

    return targets, audit_df

# ----------------------------------------------------------
# 6) Resolve direct observability targets
# ----------------------------------------------------------
explicit_targets, explicit_df, explicit_meta = _extract_observability_targets_from_metadata_12f0()
fallback_targets, fallback_df = _fallback_infer_observability_targets_schema_trainval_12f0()

explicit_targets = sorted(list(dict.fromkeys(map(str, explicit_targets))))
fallback_targets = sorted(list(dict.fromkeys(map(str, fallback_targets))))

explicit_df.to_csv(obs_explicit_metadata_diag_csv, index=False)
fallback_df.to_csv(obs_fallback_schema_csv, index=False)

# Use union, because observability columns are often scattered across contracts.
candidate_targets = sorted(list(dict.fromkeys(explicit_targets + fallback_targets)))

_generated_leaks = sorted([c for c in candidate_targets if _is_generated_target_col_12f0(c)])
if _generated_leaks:
    raise RuntimeError(
        "[Cell12.f.0] Generated target columns leaked into direct observability targets. "
        f"Preview={_generated_leaks[:30]}"
    )

_non_iot_direct_targets = sorted([c for c in candidate_targets if not _is_iot_col_12f0(c)])
if _non_iot_direct_targets:
    raise RuntimeError(
        "[Cell12.f.0] Direct observability targets must be real iot__ mask/stale/observability columns, "
        "not protocol or events_in_sec/generated target columns. "
        f"Preview={_non_iot_direct_targets[:30]}"
    )

if len(candidate_targets) == 0 and not bool(CFG.get("cell12f0_allow_empty_observability_targets", False)):
    raise RuntimeError(
        "[Cell12.f.0] No observability targets found. "
        "Check whether entity_obs/entity_stale/mask columns exist in df_tr/df_val/df_te."
    )

target_source_mode = "metadata_plus_schema_observability_inference"
resolution_reason = (
    f"explicit_n={len(explicit_targets)}|fallback_n={len(fallback_targets)}|union_n={len(candidate_targets)}"
)

IOT_OBSERVABILITY_COLS = list(candidate_targets)
IOT_OBS_MASK_COLS = list(candidate_targets)

ENTITY_OBS_COLS_12F0 = sorted([c for c in IOT_OBSERVABILITY_COLS if _obs_kind_12f0(c) == "entity_obs"])
ENTITY_STALE_COLS_12F0 = sorted([c for c in IOT_OBSERVABILITY_COLS if _obs_kind_12f0(c) in {"entity_stale", "stale_or_age"}])
AVAILABILITY_MASK_COLS_12F0 = sorted([c for c in IOT_OBSERVABILITY_COLS if _obs_kind_12f0(c) == "availability_mask"])
OBSERVED_INDICATOR_COLS_12F0 = sorted([c for c in IOT_OBSERVABILITY_COLS if _obs_kind_12f0(c) in {"observed_indicator", "global_update_indicator", "tier_present_indicator", "observability_indicator"}])

# ----------------------------------------------------------
# 7) Build target contract
# ----------------------------------------------------------
contract_rows = []

for ordinal, col in enumerate(IOT_OBSERVABILITY_COLS):
    binary_ok, binary_audit = _is_binary_trainval_12f0(col)
    obs_kind = _obs_kind_12f0(col)
    obs_rate = _safe_float_12f0(binary_audit.get("trainval_obs_rate"), np.nan)

    train_arr = _to_num_array_12f0(df_tr, col)
    val_arr = _to_num_array_12f0(df_val, col)

    train_f = train_arr[np.isfinite(train_arr)]
    val_f = val_arr[np.isfinite(val_arr)]

    train_rate = float(np.mean(train_f >= 0.5)) if train_f.size and binary_ok else np.nan
    val_rate = float(np.mean(val_f >= 0.5)) if val_f.size and binary_ok else np.nan

    near_always = bool(np.isfinite(obs_rate) and obs_rate >= float(CFG.get("cell12f0_mask_near_constant_threshold", 0.995)))
    near_never = bool(np.isfinite(obs_rate) and obs_rate <= 1.0 - float(CFG.get("cell12f0_mask_near_constant_threshold", 0.995)))
    sparse = bool(np.isfinite(obs_rate) and obs_rate <= float(CFG.get("cell12f0_mask_sparse_threshold", 0.05)))

    contract_rows.append({
        "ordinal": int(ordinal),
        "col": col,
        "entity": _entity_from_iot_col_12f0(col),
        "target_role": "observability_mask_target",
        "branch": "12.f",
        "quality_dimension": "Q3_observability",
        "obs_kind": obs_kind,
        "mask_role": _mask_role_for_target_12f0(col),
        "is_binary_mask_trainval": bool(binary_ok),
        "domain_contract": "{0,1,NaN}" if binary_ok else "nonnegative stale/age or mask-derived scalar",
        "binary_required": bool(binary_ok),
        "nonnegative_required": True,
        "train_observed_n": int(train_f.size),
        "val_observed_n": int(val_f.size),
        "train_obs_rate": train_rate,
        "val_obs_rate": val_rate,
        "abs_train_val_obs_rate_delta": (
            abs(train_rate - val_rate)
            if np.isfinite(train_rate) and np.isfinite(val_rate)
            else np.nan
        ),
        "trainval_obs_rate": obs_rate,
        "trainval_unique_n": int(binary_audit.get("trainval_unique_n", 0)),
        "trainval_unique_preview": str(binary_audit.get("trainval_unique_preview", [])),
        "near_always_observed": near_always,
        "near_never_observed": near_never,
        "sparse_observed": sparse,
        "recommended_12f2_candidate": _recommended_mask_model_12f0(obs_rate, obs_kind),
        "eligible_logistic_autoregressive_mask_model": bool(binary_ok and not near_always and not near_never),
        "eligible_regime_conditioned_markov_mask": bool(binary_ok),
        "eligible_semimarkov_mask": bool(binary_ok),
        "eligible_constant_rate_fallback": True,
        "paper2_observation_model_features": "prev_observation|regime|TOD|event_count",
        "fit_policy": "TRAIN_only",
        "selection_policy": "VAL_only",
        "test_policy": "TEST_schema_length_index_only_until_12f4_QA",
        "TEST_values_used_in_12f0": False,
    })

IOT_OBSERVABILITY_TARGET_CONTRACT_DF = pd.DataFrame(contract_rows)

if len(IOT_OBSERVABILITY_TARGET_CONTRACT_DF) != len(IOT_OBSERVABILITY_COLS):
    raise RuntimeError(
        "[Cell12.f.0] Observability target contract row mismatch: "
        f"rows={len(IOT_OBSERVABILITY_TARGET_CONTRACT_DF)} targets={len(IOT_OBSERVABILITY_COLS)}"
    )

if IOT_OBSERVABILITY_TARGET_CONTRACT_DF["col"].duplicated().any():
    dupes = IOT_OBSERVABILITY_TARGET_CONTRACT_DF.loc[
        IOT_OBSERVABILITY_TARGET_CONTRACT_DF["col"].duplicated(), "col"
    ].astype(str).tolist()
    raise RuntimeError(f"[Cell12.f.0] Duplicate observability targets: {dupes[:20]}")

# ----------------------------------------------------------
# 8) Build target-family mask contract
# ----------------------------------------------------------
family_rows = []

for family_name, cols in TARGET_FAMILIES_12F0.items():
    for target_col in cols:
        entity = _entity_from_target_or_obs_col_12f0(target_col)

        # Heuristic linked masks using parsed entity, not raw prefix tokens.
        candidate_masks = []
        for obs_col in IOT_OBSERVABILITY_COLS:
            obs_entity = _entity_from_target_or_obs_col_12f0(obs_col)
            if obs_entity and entity and (obs_entity == entity or obs_entity in str(target_col) or entity in str(obs_col)):
                candidate_masks.append(obs_col)

        # Prefer entity obs over stale over global.
        linked_entity_obs = [c for c in candidate_masks if _obs_kind_12f0(c) == "entity_obs"]
        linked_stale = [c for c in candidate_masks if _obs_kind_12f0(c) in {"entity_stale", "stale_or_age"}]
        linked_observed = [c for c in candidate_masks if _obs_kind_12f0(c) in {"observed_indicator", "availability_mask"}]

        if linked_entity_obs:
            primary_mask = linked_entity_obs[0]
            primary_method = "entity_obs_name_match"
        elif linked_observed:
            primary_mask = linked_observed[0]
            primary_method = "observed_or_availability_name_match"
        elif linked_stale:
            primary_mask = linked_stale[0]
            primary_method = "stale_name_match"
        elif "iot__obs_present" in IOT_OBSERVABILITY_COLS:
            primary_mask = "iot__obs_present"
            primary_method = "global_iot_obs_present_fallback"
        else:
            primary_mask = ""
            primary_method = "no_direct_mask_link_found"

        family_rows.append({
            "target_family": family_name,
            "target_col": target_col,
            "entity_guess": entity,
            "primary_observability_col": primary_mask,
            "primary_observability_link_method": primary_method,
            "candidate_observability_cols": "|".join(candidate_masks[:20]),
            "candidate_observability_cols_n": int(len(candidate_masks)),
            "mask_materialization_required_in_12f3": True,
            "feeds_q3_observability_qa": True,
            "TEST_values_used": False,
        })

IOT_OBSERVABILITY_FAMILY_CONTRACT_DF = pd.DataFrame(family_rows)

# ----------------------------------------------------------
# 9) 12.f plan
# ----------------------------------------------------------
CELL12F_PLAN = {
    "12.f.0": {
        "name": "Observability contracts",
        "status": "create",
        "targets": [
            "entity_obs_cols",
            "entity_stale_cols",
            "value availability masks",
            "feature/entity observed indicators",
        ],
        "role": "define Q3 observability targets and family mask links",
    },
    "12.f.1": {
        "name": "Mask baseline",
        "status": "create",
        "baselines": {
            "A0": "diagnostic real-mask upper bound if needed",
            "A1": "TRAIN/VAL-fitted mask model",
        },
        "role": "build leakage-safe VAL mask baselines",
    },
    "12.f.2": {
        "name": "Mask model candidates",
        "status": "create",
        "candidate_families": {
            "logistic autoregressive mask model": "Paper 2-style observation model conditioned on previous observation, regime, TOD, and event count",
            "regime-conditioned Markov mask": "sparse sensors",
            "semi-Markov mask": "long online/offline periods",
            "constant-rate fallback": "near-always/near-never observed",
        },
    },
    "12.f.3": {
        "name": "Mask materialization",
        "status": "create",
        "role": "generate TEST-length IoT masks for all generated IoT target families",
    },
    "12.f.4": {
        "name": "Mask QA",
        "status": "create",
        "quality_dimension": "Q3",
        "metrics": [
            "obs_rate_error",
            "p11_error",
            "p00_error",
            "run_length_ks",
            "mask_c2st",
            "regime_obs_rate_error",
            "inactive_finite_violations",
        ],
    },
}

# ----------------------------------------------------------
# 10) Resolution diagnostic
# ----------------------------------------------------------
obs_kind_counts = (
    IOT_OBSERVABILITY_TARGET_CONTRACT_DF["obs_kind"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

recommended_counts = (
    IOT_OBSERVABILITY_TARGET_CONTRACT_DF["recommended_12f2_candidate"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

generated_target_direct_leak_n = int(
    sum(1 for c in IOT_OBSERVABILITY_COLS if _is_generated_target_col_12f0(c))
)
direct_observability_excluded_generated_target_n = int(
    sum(
        1
        for c in DIRECT_OBSERVABILITY_EXCLUDE_TARGET_COLS_12F0
        if c in df_tr.columns
        and c in df_val.columns
        and c in df_te.columns
        and c not in IOT_OBSERVABILITY_COLS
    )
)

resolution_rows = [
    {"metric": "observability_targets_total", "value": int(len(IOT_OBSERVABILITY_COLS))},
    {"metric": "explicit_observability_targets_n", "value": int(len(explicit_targets))},
    {"metric": "fallback_observability_targets_n", "value": int(len(fallback_targets))},
    {"metric": "generated_target_direct_leak_n", "value": int(generated_target_direct_leak_n)},
    {"metric": "direct_observability_excluded_generated_target_n", "value": int(direct_observability_excluded_generated_target_n)},
    {"metric": "entity_obs_cols_n", "value": int(len(ENTITY_OBS_COLS_12F0))},
    {"metric": "entity_stale_cols_n", "value": int(len(ENTITY_STALE_COLS_12F0))},
    {"metric": "availability_mask_cols_n", "value": int(len(AVAILABILITY_MASK_COLS_12F0))},
    {"metric": "observed_indicator_cols_n", "value": int(len(OBSERVED_INDICATOR_COLS_12F0))},
    {"metric": "family_contract_rows_n", "value": int(len(IOT_OBSERVABILITY_FAMILY_CONTRACT_DF))},
    {"metric": "target_source_mode", "value": target_source_mode},
    {"metric": "resolution_reason", "value": resolution_reason},
    {"metric": "TEST_values_used", "value": False},
]

for k, v in obs_kind_counts.items():
    resolution_rows.append({"metric": f"obs_kind_count::{k}", "value": int(v)})

for k, v in recommended_counts.items():
    resolution_rows.append({"metric": f"recommended_model_count::{k}", "value": int(v)})

pd.DataFrame(resolution_rows).to_csv(obs_resolution_diag_csv, index=False)

# ----------------------------------------------------------
# 11) Save artifacts
# ----------------------------------------------------------
IOT_OBSERVABILITY_TARGET_CONTRACT_DF.to_csv(obs_target_contract_csv, index=False)
IOT_OBSERVABILITY_FAMILY_CONTRACT_DF.to_csv(obs_family_contract_csv, index=False)

targets_payload = {
    "cell": "12.f.0",
    "version": CELL12F0_VERSION,
    "quality_dimension": "Q3_observability",
    "observability_targets_total": int(len(IOT_OBSERVABILITY_COLS)),
    "target_source_mode": target_source_mode,
    "target_resolution_reason": resolution_reason,
    "generated_target_direct_leak_n": int(generated_target_direct_leak_n),
    "direct_observability_excluded_generated_target_n": int(direct_observability_excluded_generated_target_n),
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "df_te_used_for_index_length_schema_only": True,
    "targets": IOT_OBSERVABILITY_COLS,
    "entity_obs_cols": ENTITY_OBS_COLS_12F0,
    "entity_stale_cols": ENTITY_STALE_COLS_12F0,
    "availability_mask_cols": AVAILABILITY_MASK_COLS_12F0,
    "observed_indicator_cols": OBSERVED_INDICATOR_COLS_12F0,
    "obs_kind_counts": obs_kind_counts,
    "recommended_model_counts": recommended_counts,
}

_write_json_12f0(obs_targets_json, targets_payload)
_write_json_12f0(obs_targets_report_json, targets_payload)

contract = {
    "cell": "12.f.0",
    "version": CELL12F0_VERSION,
    "role": "iot_observability_mask_contracts_and_12f_branch_plan",
    "quality_dimension": "Q3_observability",
    "observability_targets_total": int(len(IOT_OBSERVABILITY_COLS)),
    "target_source_mode": target_source_mode,
    "target_resolution_reason": resolution_reason,
    "explicit_metadata": explicit_meta,
    "explicit_targets_n": int(len(explicit_targets)),
    "fallback_schema_inferred_n": int(len(fallback_targets)),
    "generated_target_direct_leak_n": int(generated_target_direct_leak_n),
    "direct_observability_excluded_generated_target_n": int(direct_observability_excluded_generated_target_n),
    "obs_kind_counts": obs_kind_counts,
    "recommended_model_counts": recommended_counts,
    "entity_obs_cols_n": int(len(ENTITY_OBS_COLS_12F0)),
    "entity_stale_cols_n": int(len(ENTITY_STALE_COLS_12F0)),
    "availability_mask_cols_n": int(len(AVAILABILITY_MASK_COLS_12F0)),
    "observed_indicator_cols_n": int(len(OBSERVED_INDICATOR_COLS_12F0)),
    "target_family_refs": {
        "continuous_value_12c_n": int(len(TARGET_FAMILIES_12F0["continuous_value_12c"])),
        "binary_state_12d_n": int(len(TARGET_FAMILIES_12F0["binary_state_12d"])),
        "driver_event_12e_n": int(len(TARGET_FAMILIES_12F0["driver_event_12e"])),
        "family_contract_rows_n": int(len(IOT_OBSERVABILITY_FAMILY_CONTRACT_DF)),
    },
    "cell12f_plan": CELL12F_PLAN,
    "paper2_observation_model": {
        "primary_candidate": "logistic autoregressive mask model",
        "conditioning": [
            "previous_observation",
            "regime",
            "time_of_day",
            "event_count",
        ],
    },
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "mask_materialization_done_here": False,
    "train_val_values_used_for_target_validation": True,
    "df_te_used_for_index_length_schema_only": True,
    "direct_observability_target_policy": "direct targets must be true iot__ mask/stale/observability columns; 12.c/12.d/12.e generated target columns are only family refs, never direct 12.f mask targets.",
    "events_in_sec_driver_family_link_supported": True,
    "outputs": {
        "obs_targets_json": obs_targets_json,
        "obs_targets_report_json": obs_targets_report_json,
        "obs_contract_json": obs_contract_json,
        "obs_contract_canonical_json": obs_contract_canonical_json,
        "obs_explicit_metadata_diag_csv": obs_explicit_metadata_diag_csv,
        "obs_target_contract_csv": obs_target_contract_csv,
        "obs_family_contract_csv": obs_family_contract_csv,
        "obs_resolution_diag_csv": obs_resolution_diag_csv,
        "obs_fallback_schema_csv": obs_fallback_schema_csv,
        "obs_manifest_json": obs_manifest_json,
    },
}

_write_json_12f0(obs_contract_json, contract)
_write_json_12f0(obs_contract_canonical_json, contract)

manifest = {
    "cell": "12.f.0",
    "version": CELL12F0_VERSION,
    "created_outputs": contract["outputs"],
    "quality_dimension": "Q3_observability",
    "observability_targets_total": int(len(IOT_OBSERVABILITY_COLS)),
    "generated_target_direct_leak_n": int(generated_target_direct_leak_n),
    "direct_observability_excluded_generated_target_n": int(direct_observability_excluded_generated_target_n),
    "obs_kind_counts": obs_kind_counts,
    "recommended_model_counts": recommended_counts,
    "cell12f_plan": CELL12F_PLAN,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "mask_materialization_done_here": False,
    },
}

_write_json_12f0(obs_manifest_json, manifest)

hashes = {
    "obs_targets_json_sha256": _sha256_file_12f0(obs_targets_json),
    "obs_targets_report_json_sha256": _sha256_file_12f0(obs_targets_report_json),
    "obs_target_contract_csv_sha256": _sha256_file_12f0(obs_target_contract_csv),
    "obs_family_contract_csv_sha256": _sha256_file_12f0(obs_family_contract_csv),
    "obs_resolution_diag_csv_sha256": _sha256_file_12f0(obs_resolution_diag_csv),
    "obs_explicit_metadata_diag_csv_sha256": _sha256_file_12f0(obs_explicit_metadata_diag_csv),
    "obs_fallback_schema_csv_sha256": _sha256_file_12f0(obs_fallback_schema_csv),
    "obs_contract_json_sha256": _sha256_file_12f0(obs_contract_json),
    "obs_contract_canonical_json_sha256": _sha256_file_12f0(obs_contract_canonical_json),
    "obs_manifest_json_sha256": _sha256_file_12f0(obs_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12f0(obs_contract_json, contract)
_write_json_12f0(obs_contract_canonical_json, contract)
_write_json_12f0(obs_manifest_json, manifest)

# ----------------------------------------------------------
# 12) Export globals for 12.f.1+
# ----------------------------------------------------------
globals()["CELL12F0_VERSION"] = CELL12F0_VERSION
globals()["IOT_OBSERVABILITY_COLS"] = IOT_OBSERVABILITY_COLS
globals()["IOT_OBS_MASK_COLS"] = IOT_OBS_MASK_COLS
globals()["ENTITY_OBS_COLS_12F0"] = ENTITY_OBS_COLS_12F0
globals()["ENTITY_STALE_COLS_12F0"] = ENTITY_STALE_COLS_12F0
globals()["AVAILABILITY_MASK_COLS_12F0"] = AVAILABILITY_MASK_COLS_12F0
globals()["OBSERVED_INDICATOR_COLS_12F0"] = OBSERVED_INDICATOR_COLS_12F0
globals()["IOT_OBSERVABILITY_TARGET_CONTRACT_DF"] = IOT_OBSERVABILITY_TARGET_CONTRACT_DF
globals()["IOT_OBSERVABILITY_FAMILY_CONTRACT_DF"] = IOT_OBSERVABILITY_FAMILY_CONTRACT_DF
globals()["TARGET_FAMILIES_12F0"] = TARGET_FAMILIES_12F0
globals()["CELL12F_OBSERVABILITY_CONTRACT"] = contract
globals()["CELL12F_PLAN"] = CELL12F_PLAN
globals()["CELL12F_OBSERVABILITY_TARGETS_JSON"] = obs_targets_json
globals()["CELL12F_OBSERVABILITY_CONTRACT_JSON"] = obs_contract_json
globals()["CELL12F_OBSERVABILITY_CONTRACT_CANONICAL_JSON"] = obs_contract_canonical_json
globals()["CELL12F0_OBSERVABILITY_EXPLICIT_METADATA_DIAG_CSV"] = obs_explicit_metadata_diag_csv
globals()["CELL12F0_OBSERVABILITY_TARGET_CONTRACT_CSV"] = obs_target_contract_csv
globals()["CELL12F0_OBSERVABILITY_FAMILY_CONTRACT_CSV"] = obs_family_contract_csv
globals()["CELL12F0_OBSERVABILITY_RESOLUTION_DIAG_CSV"] = obs_resolution_diag_csv
globals()["CELL12F0_OBSERVABILITY_FALLBACK_SCHEMA_CSV"] = obs_fallback_schema_csv
globals()["CELL12F0_OBSERVABILITY_MANIFEST_JSON"] = obs_manifest_json

log(
    "[Cell12.f.0] Observability contract built | "
    f"targets={len(IOT_OBSERVABILITY_COLS)} | "
    f"entity_obs={len(ENTITY_OBS_COLS_12F0)} | "
    f"entity_stale={len(ENTITY_STALE_COLS_12F0)} | "
    f"availability_masks={len(AVAILABILITY_MASK_COLS_12F0)} | "
    f"observed_indicators={len(OBSERVED_INDICATOR_COLS_12F0)} | "
    f"family_contract_rows={len(IOT_OBSERVABILITY_FAMILY_CONTRACT_DF)}"
)
log(f"[Cell12.f.0] Obs kind counts | {obs_kind_counts}")
log(f"[Cell12.f.0] Recommended model counts | {recommended_counts}")
log(f"[Cell12.f.0] Saved observability targets JSON: {obs_targets_json}")
log(f"[Cell12.f.0] Saved observability contract JSON: {obs_contract_json}")
log(f"[Cell12.f.0] Saved canonical contract JSON: {obs_contract_canonical_json}")
log(f"[Cell12.f.0] Saved explicit metadata diagnostic CSV: {obs_explicit_metadata_diag_csv} | rows={len(explicit_df)}")
log(f"[Cell12.f.0] Saved target contract CSV: {obs_target_contract_csv} | rows={len(IOT_OBSERVABILITY_TARGET_CONTRACT_DF)}")
log(f"[Cell12.f.0] Saved family contract CSV: {obs_family_contract_csv} | rows={len(IOT_OBSERVABILITY_FAMILY_CONTRACT_DF)}")
log(f"[Cell12.f.0] Saved fallback schema inference CSV: {obs_fallback_schema_csv} | rows={len(fallback_df)}")
log(f"[Cell12.f.0] Saved resolution diagnostic CSV: {obs_resolution_diag_csv}")
log(
    "[Cell12.f.0] Contract flags | "
    "TEST_real_values_used=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "mask_materialization_done_here=False | "
    "df_te_used_for_index_length_schema_only=True"
)
log("--- END: Cell 12.f.0 - IoT observability contracts (v1.1 strict resolver, target-leak hardened) ---")

gc.collect()
