# ============================================================
# CELL 4.5 — IoT feature grouping / role ownership contract — v13
# (STUDY-THESIS GRADE / NO TEST-LEAKAGE TAXONOMY / FULL-IOT OWNERSHIP /
#  EXHAUSTIVE PRIMARY OWNER CONTRACT / ROLE-TO-CELL ROUTING /
#  EXPLICIT EXCLUSIONS / TELEMETRY-IN-SEC META QUARANTINE /
#  BINARY __VALUE STATE SUPPORT / HARMFUL RED-FLAG AUDIT)
#
# Purpose:
# - Freeze an exhaustive IoT taxonomy for downstream synthesis.
# - Classify over the FULL IoT-related namespace:
#     iot__*
#     events_in_sec__*
#     telemetry_in_sec__*
# - Assign every IoT-related column exactly one PRIMARY OWNER.
# - Keep observability/freshness/bookkeeping/helper columns OUT of modelable
#   IoT state targets.
# - Explicitly account for every IoT-related column:
#     continuous target / binary target / driver target / mask/meta /
#     staleness/meta / conditioning/meta / excluded.
# - Preserve subtype overlaps only inside continuous/value targets:
#     countlike / telemetry / step_progress / discrete_state / setting_like.
# - Allow semantically binary state fields that end with __value.
# - Avoid TEST leakage in taxonomy assignment.
#
# Critical policy:
# - TRAIN+VAL may be used to define taxonomy.
# - TEST may be used only for audit/red flags, never for assigning roles.
# - telemetry_in_sec__* columns are meta/conditioning helpers, not modeled
#   IoT state targets.
# - COUNTLIKE / TELEMETRY / STEP_PROGRESS / DISCRETE_STATE / SETTING_LIKE
#   are continuous-target subtypes, not separate top-level target groups.
#
# Primary owner labels:
#   continuous_value_target
#   binary_state_target
#   driver_event_target
#   mask_observability_meta
#   staleness_meta
#   conditioning_meta
#   derived_meta
#   excluded_no_dev_support
#   excluded_low_dev_support
#   excluded_non_numeric
#
# Artifacts:
#   artifacts/iot_feature_groups.json
#   artifacts/iot_feature_grouping_evidence.csv
#   artifacts/iot_feature_grouping_partition_audit.csv
#   artifacts/iot_feature_grouping_red_flags.csv
#   artifacts/iot_semantic_contract.json
#   artifacts/iot_role_ownership.csv
#   artifacts/iot_partition_contract.json
#   artifacts/iot_overlap_policy.json
#   artifacts/iot_cell_routing_manifest.json
# ============================================================

log("--- START: Cell 4.5 — IoT feature grouping / role ownership contract (v13) ---")

import os
import json
import re
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Sequence, Tuple, Optional

needed = [
    "CFG", "df_tr", "df_va", "df_te", "log", "time_col",
    "IOT_DRIVER_COLS",
]
missing = [k for k in needed if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell4.5] Missing prerequisites: {missing}. Run Cells 1–4.4 first.")

art_dir = os.path.join(str(CFG["outdir"]), "artifacts")
os.makedirs(art_dir, exist_ok=True)

# ------------------------------------------------------------
# 0) Small helpers
# ------------------------------------------------------------
def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in list(obj)]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj

def _dedup_keep_order(xs: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _lower(c: str) -> str:
    return str(c).lower()

# ------------------------------------------------------------
# 1) Resolve canonical seconds column
# ------------------------------------------------------------
SEC_COL = str(globals()["time_col"])

if SEC_COL not in df_tr.columns:
    raise RuntimeError(f"[Cell4.5] Canonical time column missing from TRAIN: {SEC_COL}")

globals()["SEC_COL"] = SEC_COL
log(f"[Cell4.5] SEC_COL resolved to: {SEC_COL}")

# ------------------------------------------------------------
# 2) Resolve driver columns from strict upstream contract
# ------------------------------------------------------------
driver_cols_raw = list(globals().get("IOT_DRIVER_COLS") or [])

driver_contract_path = os.path.join(art_dir, "driver_contract.json")
if not driver_cols_raw and os.path.exists(driver_contract_path):
    with open(driver_contract_path, "r", encoding="utf-8") as f:
        dc = json.load(f)
    driver_cols_raw = list(dc.get("iot_driver_cols") or [])
    log(f"[Cell4.5] Loaded drivers from contract: {driver_contract_path}")

if not driver_cols_raw:
    raise RuntimeError(
        "[Cell4.5] No IoT driver columns found from upstream contract. "
        "Do not recover heuristically here; fix upstream cells."
    )

if len(driver_cols_raw) != len(set(driver_cols_raw)):
    dupes = sorted([c for c in set(driver_cols_raw) if driver_cols_raw.count(c) > 1])
    raise RuntimeError(f"[Cell4.5] Duplicate driver columns detected: {dupes[:20]}")

IOT_DRIVER_COLS = _dedup_keep_order(driver_cols_raw)

bad_driver_names = [
    c for c in IOT_DRIVER_COLS
    if not (isinstance(c, str) and c.startswith("events_in_sec__"))
]
if bad_driver_names:
    raise RuntimeError(
        f"[Cell4.5] IOT_DRIVER_COLS contains non-events_in_sec entries: {bad_driver_names[:20]}"
    )

for nm, _df in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    miss = [c for c in IOT_DRIVER_COLS if c not in _df.columns]
    if miss:
        raise RuntimeError(f"[Cell4.5] Missing driver columns in {nm}: {miss[:20]}")

# ------------------------------------------------------------
# 3) Full IoT-related namespace discovery
# ------------------------------------------------------------
def _is_iot_namespace_col(c: str) -> bool:
    cl = _lower(c)
    return (
        cl.startswith("iot__")
        or cl.startswith("events_in_sec__")
        or cl.startswith("telemetry_in_sec__")
    )

cell3_iot_cols = list(globals().get("IOT_NAMESPACE_COLS") or [])
cell3_event_cols = list(globals().get("EVENT_DRIVER_COLS") or IOT_DRIVER_COLS)
cell3_telemetry_driver_cols = list(globals().get("TELEMETRY_DRIVER_COLS") or [])

all_iot_from_train_schema = [c for c in df_tr.columns if _is_iot_namespace_col(c)]

ALL_IOT_NAMESPACE_COLS = sorted(set(
    all_iot_from_train_schema
    + cell3_iot_cols
    + cell3_event_cols
    + cell3_telemetry_driver_cols
    + IOT_DRIVER_COLS
))

if not ALL_IOT_NAMESPACE_COLS:
    raise RuntimeError("[Cell4.5] No IoT-related namespace columns found in TRAIN.")

for nm, _df in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    miss = [c for c in ALL_IOT_NAMESPACE_COLS if c not in _df.columns]
    if miss:
        raise RuntimeError(
            f"[Cell4.5] Full IoT namespace is not split-stable in {nm}: missing={miss[:30]}"
        )

# ------------------------------------------------------------
# 4) Meta / helper / observability patterns
# ------------------------------------------------------------
META_HELPER_RX = re.compile(
    r"("
    r"__obs_present$|"
    r"__traffic_present$|"
    r"__traffic_obs_present$|"
    r"__present$|"
    r"__mask$|"
    r"__stale_flag$|"
    r"__staleness_s$|"
    r"^iot__entity_obs__|"
    r"^iot__entity_stale__"
    r")",
    re.IGNORECASE,
)

META_EXACT = {
    "iot__obs_present",
    "iot__any_update_raw",
}

def _meta_owner_reason(c: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Returns:
      is_meta, primary_owner, reason
    """
    cl = _lower(c)

    if c == SEC_COL:
        return True, "conditioning_meta", "canonical_time_column_conditioning_meta"

    if c in META_EXACT:
        if c == "iot__obs_present":
            return True, "mask_observability_meta", "global_iot_observability_mask"
        return True, "derived_meta", "exact_iot_bookkeeping_meta"

    if cl.startswith("telemetry_in_sec__"):
        return True, "conditioning_meta", "telemetry_in_sec_helper_conditioning_meta"

    if cl.startswith("iot__entity_obs__"):
        return True, "mask_observability_meta", "entity_observability_meta"

    if cl.startswith("iot__entity_stale__"):
        return True, "staleness_meta", "entity_staleness_meta"

    if META_HELPER_RX.search(cl):
        if "__stale_flag" in cl or "__staleness_s" in cl:
            return True, "staleness_meta", "feature_staleness_meta"
        return True, "mask_observability_meta", "feature_observability_mask_meta"

    return False, None, None

def _is_iot_meta(c: str) -> bool:
    return _meta_owner_reason(c)[0]

IOT_META_COLS = sorted([c for c in ALL_IOT_NAMESPACE_COLS if _is_iot_meta(c)])

# ------------------------------------------------------------
# 5) Numeric / support helpers
# ------------------------------------------------------------
def _to_num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32, copy=False)

def _finite_vals_from_df(df_: pd.DataFrame, c: str) -> np.ndarray:
    if c not in df_.columns:
        return np.array([], dtype=np.float32)
    x = _to_num(df_[c])
    return x[np.isfinite(x)].astype(np.float32, copy=False)

def _finite_vals_train(c: str) -> np.ndarray:
    return _finite_vals_from_df(df_tr, c)

def _finite_vals_val(c: str) -> np.ndarray:
    return _finite_vals_from_df(df_va, c)

def _finite_vals_test(c: str) -> np.ndarray:
    return _finite_vals_from_df(df_te, c)

def _finite_vals_dev(c: str) -> np.ndarray:
    # TRAIN + VAL only. Allowed for taxonomy assignment.
    xs = [_finite_vals_train(c), _finite_vals_val(c)]
    xs = [x for x in xs if len(x)]
    if not xs:
        return np.array([], dtype=np.float32)
    if len(xs) == 1:
        return xs[0].astype(np.float32, copy=False)
    return np.concatenate(xs).astype(np.float32, copy=False)

def _finite_vals_all_for_audit(c: str) -> np.ndarray:
    # TEST is audit-only.
    xs = [_finite_vals_train(c), _finite_vals_val(c), _finite_vals_test(c)]
    xs = [x for x in xs if len(x)]
    if not xs:
        return np.array([], dtype=np.float32)
    if len(xs) == 1:
        return xs[0].astype(np.float32, copy=False)
    return np.concatenate(xs).astype(np.float32, copy=False)

def _finite_rate(df_: pd.DataFrame, c: str) -> float:
    if c not in df_.columns or len(df_) == 0:
        return 0.0
    x = _to_num(df_[c])
    return float(np.isfinite(x).mean())

def _finite_rate_train(c: str) -> float:
    return _finite_rate(df_tr, c)

def _finite_rate_val(c: str) -> float:
    return _finite_rate(df_va, c)

def _finite_rate_test(c: str) -> float:
    return _finite_rate(df_te, c)

def _finite_rate_dev_max(c: str) -> float:
    return max(_finite_rate_train(c), _finite_rate_val(c))

def _is_numeric_like_dev(c: str, min_dev_rate: float = 0.001) -> bool:
    # Do not inspect TEST here.
    return _finite_rate_dev_max(c) >= float(min_dev_rate)

def _rounded_unique(vals: np.ndarray, decimals: int = 6) -> np.ndarray:
    if vals.size == 0:
        return np.array([], dtype=np.float64)
    return np.unique(np.round(vals.astype(np.float64), decimals))

def _binary01(vals: np.ndarray) -> bool:
    if vals.size == 0:
        return False
    u = _rounded_unique(vals, 6)
    return set(u.tolist()).issubset({0.0, 1.0})

def _binary01_train(c: str) -> bool:
    return _binary01(_finite_vals_train(c))

def _binary01_val(c: str) -> bool:
    return _binary01(_finite_vals_val(c))

def _binary01_test(c: str) -> bool:
    return _binary01(_finite_vals_test(c))

def _binary01_dev(c: str) -> bool:
    return _binary01(_finite_vals_dev(c))

def _nunique_from_vals(vals: np.ndarray) -> int:
    if len(vals) == 0:
        return 0
    return int(len(_rounded_unique(vals, 6)))

def _nunique_train(c: str) -> int:
    return _nunique_from_vals(_finite_vals_train(c))

def _nunique_val(c: str) -> int:
    return _nunique_from_vals(_finite_vals_val(c))

def _nunique_test(c: str) -> int:
    return _nunique_from_vals(_finite_vals_test(c))

def _nunique_dev(c: str) -> int:
    return _nunique_from_vals(_finite_vals_dev(c))

def _nunique_all_for_audit(c: str) -> int:
    return _nunique_from_vals(_finite_vals_all_for_audit(c))

def _const_status(vals: np.ndarray) -> str:
    if len(vals) == 0:
        return "all_nan"
    u = _rounded_unique(vals, 6)
    if len(u) == 1:
        return "constant"
    if len(u) <= 2:
        return "binary_or_two_point"
    return "variable"

def _const_status_dev(c: str) -> str:
    return _const_status(_finite_vals_dev(c))

def _const_status_all_for_audit(c: str) -> str:
    return _const_status(_finite_vals_all_for_audit(c))

# ------------------------------------------------------------
# 6) Semantic name helpers
# ------------------------------------------------------------
def _looks_value_like(c: str) -> bool:
    return _lower(c).endswith("__value")

def _looks_state_like(c: str) -> bool:
    return _lower(c).endswith("__state")

def _looks_state_value_like(c: str) -> bool:
    s = _lower(c)
    return s.endswith("__value") and "__state__" in s

def _is_countlike_name(c: str) -> bool:
    s = _lower(c)
    return any(k in s for k in [
        "count__value",
        "trigger_count",
        "open_count",
        "close_count",
        "motion_count",
        "press_count",
        "power_outage_count",
        "today_s_consumption",
        "this_month_s_consumption",
        "total_consumption",
        "total_cleaning_area",
        "total_cleaning_time",
        "total_cleaning_count",
        "cleaning_area",
        "cleaning_time",
        "coffees",
        "cups",
        "coffee_and_milk_cups",
        "hot_water_cups",
    ])

def _is_telemetry_name(c: str) -> bool:
    s = _lower(c)
    return any(k in s for k in [
        "battery",
        "voltage",
        "linkquality",
        "rssi",
        "lqi",
        "signal_strength",
        "signal_level",
        "device_temperature",
    ])

def _is_step_progress_name(c: str) -> bool:
    return "programme_progress" in _lower(c)

def _is_setting_like_name(c: str) -> bool:
    s = _lower(c)
    return any(k in s for k in [
        "__select__",
        "__number__",
        "__update__",
        "occupancy_timeout",
        "power_on_behavior",
        "keep_time",
        "auto_off",
        "auto_update",
        "led_indication",
        "power_protection",
        "turn_off_in",
        "fill_quantity",
        "sensitivity",
    ])

def _continuous_subtypes_for_col(c: str) -> List[str]:
    subtypes = []
    if _is_countlike_name(c):
        subtypes.append("countlike")
    if _is_telemetry_name(c):
        subtypes.append("telemetry")
    if _is_step_progress_name(c):
        subtypes.append("step_progress")
    if _is_setting_like_name(c):
        subtypes.append("setting_like")
    if (_looks_state_like(c) or _looks_state_value_like(c)) and _nunique_dev(c) <= 16:
        subtypes.append("discrete_state")
    return subtypes

# ------------------------------------------------------------
# 7) Full-schema classification with primary ownership
# ------------------------------------------------------------
driver_set = set(IOT_DRIVER_COLS)
meta_set = set(IOT_META_COLS)

IOT_BIN_COLS: List[str] = []
IOT_CONT_COLS: List[str] = []

IOT_EXCLUDED_COLS: List[str] = []
IOT_EXCLUDED_NON_NUMERIC_COLS: List[str] = []
IOT_EXCLUDED_ALL_NAN_COLS: List[str] = []
IOT_EXCLUDED_LOW_SIGNAL_COLS: List[str] = []

classification_rows: List[Dict[str, Any]] = []
role_ownership_rows: List[Dict[str, Any]] = []

MIN_DEV_NUMERIC_RATE = float(CFG.get("iot_taxonomy_min_dev_numeric_rate", 0.001))
if not np.isfinite(MIN_DEV_NUMERIC_RATE) or not (0.0 <= MIN_DEV_NUMERIC_RATE <= 1.0):
    raise RuntimeError(
        f"[Cell4.5] Invalid iot_taxonomy_min_dev_numeric_rate={MIN_DEV_NUMERIC_RATE}"
    )

def _target_cell_for_owner(primary_owner: str) -> str:
    return {
        "continuous_value_target": "Cell 12.c",
        "binary_state_target": "Cell 12.d",
        "driver_event_target": "Cell 12.e",
        "mask_observability_meta": "Cell 12.f / Cell 12.g",
        "staleness_meta": "Cell 12.f / Cell 12.g",
        "conditioning_meta": "Cell 12.g / conditioning only",
        "derived_meta": "Cell 12.g / derived only",
        "excluded_no_dev_support": "not generated",
        "excluded_low_dev_support": "not generated",
        "excluded_non_numeric": "not generated",
    }.get(str(primary_owner), "unknown")

for c in ALL_IOT_NAMESPACE_COLS:
    cl = _lower(c)
    numeric_like_dev = _is_numeric_like_dev(c, min_dev_rate=MIN_DEV_NUMERIC_RATE)

    looks_value = _looks_value_like(c)
    looks_state = _looks_state_like(c)
    looks_state_value = _looks_state_value_like(c)

    top_role = None
    primary_owner = None
    owner_reason = None
    generation_policy = None

    if c in driver_set:
        top_role = "driver"
        primary_owner = "driver_event_target"
        owner_reason = "explicit_upstream_event_driver_contract"
        generation_policy = "generate_as_driver_event_process"

    elif c in meta_set:
        is_meta, meta_owner, meta_reason = _meta_owner_reason(c)
        if not is_meta or meta_owner is None:
            raise RuntimeError(f"[Cell4.5] Internal meta owner resolution failure for: {c}")
        top_role = "meta"
        primary_owner = meta_owner
        owner_reason = meta_reason
        if meta_owner in {"mask_observability_meta", "staleness_meta"}:
            generation_policy = "generate_or_derive_as_observability_meta_not_value_target"
        elif meta_owner == "conditioning_meta":
            generation_policy = "conditioning_helper_not_primary_target"
        else:
            generation_policy = "derive_or_carry_not_primary_target"

    elif not numeric_like_dev:
        top_role = "excluded"
        IOT_EXCLUDED_COLS.append(c)

        if _finite_rate_dev_max(c) == 0.0:
            primary_owner = "excluded_no_dev_support"
            owner_reason = "excluded_all_nan_or_no_dev_numeric_support"
            IOT_EXCLUDED_ALL_NAN_COLS.append(c)
        elif _finite_rate_dev_max(c) < MIN_DEV_NUMERIC_RATE:
            primary_owner = "excluded_low_dev_support"
            owner_reason = "excluded_low_dev_numeric_support"
            IOT_EXCLUDED_LOW_SIGNAL_COLS.append(c)
        else:
            primary_owner = "excluded_non_numeric"
            owner_reason = "excluded_non_numeric_or_unparseable_dev"
            IOT_EXCLUDED_NON_NUMERIC_COLS.append(c)

        IOT_EXCLUDED_NON_NUMERIC_COLS.append(c)
        generation_policy = "not_generated_without_trainval_support"

    else:
        # Semantics-first modelable-state rules. TEST is not inspected here.
        if looks_state_value:
            if _binary01_dev(c):
                top_role = "binary"
                primary_owner = "binary_state_target"
                owner_reason = "state_value_like_and_binary01_dev"
                IOT_BIN_COLS.append(c)
                generation_policy = "generate_as_binary_state"
            else:
                top_role = "continuous"
                primary_owner = "continuous_value_target"
                owner_reason = "state_value_like_but_nonbinary_dev"
                IOT_CONT_COLS.append(c)
                generation_policy = "generate_as_continuous_or_ordinal_value"

        elif looks_value:
            top_role = "continuous"
            primary_owner = "continuous_value_target"
            owner_reason = "suffix::__value"
            IOT_CONT_COLS.append(c)
            generation_policy = "generate_as_continuous_value"

        elif looks_state:
            if _binary01_dev(c):
                top_role = "binary"
                primary_owner = "binary_state_target"
                owner_reason = "suffix::__state_and_binary01_dev"
                IOT_BIN_COLS.append(c)
                generation_policy = "generate_as_binary_state"
            else:
                top_role = "continuous"
                primary_owner = "continuous_value_target"
                owner_reason = "suffix::__state_but_nonbinary_dev"
                IOT_CONT_COLS.append(c)
                generation_policy = "generate_as_continuous_or_ordinal_state"

        else:
            if _binary01_dev(c):
                top_role = "binary"
                primary_owner = "binary_state_target"
                owner_reason = "residual_numeric_binary01_dev"
                IOT_BIN_COLS.append(c)
                generation_policy = "generate_as_binary_state"
            else:
                top_role = "continuous"
                primary_owner = "continuous_value_target"
                owner_reason = "residual_numeric_state_channel_dev"
                IOT_CONT_COLS.append(c)
                generation_policy = "generate_as_continuous_value"

    if top_role is None or primary_owner is None:
        raise RuntimeError(f"[Cell4.5] Internal classification failed for column: {c}")

    subtypes = _continuous_subtypes_for_col(c) if top_role == "continuous" else []

    classification_rows.append({
        "col": c,
        "assigned_top_role": top_role,
        "primary_owner": primary_owner,
        "assigned_subtypes": "|".join(subtypes),
        "reason": owner_reason,
        "generation_policy": generation_policy,
        "target_cell": _target_cell_for_owner(primary_owner),
        "finite_rate_train": _finite_rate_train(c),
        "finite_rate_val": _finite_rate_val(c),
        "finite_rate_test_audit": _finite_rate_test(c),
        "is_numeric_like_dev": bool(numeric_like_dev),
        "binary01_train": bool(_binary01_train(c)),
        "binary01_val": bool(_binary01_val(c)),
        "binary01_dev": bool(_binary01_dev(c)),
        "binary01_test_audit": bool(_binary01_test(c)),
        "nunique_train": int(_nunique_train(c)),
        "nunique_val": int(_nunique_val(c)),
        "nunique_dev": int(_nunique_dev(c)),
        "nunique_test_audit": int(_nunique_test(c)),
        "nunique_all_audit": int(_nunique_all_for_audit(c)),
    })

    role_ownership_rows.append({
        "col": c,
        "primary_owner": primary_owner,
        "top_role": top_role,
        "target_cell": _target_cell_for_owner(primary_owner),
        "generation_policy": generation_policy,
        "reason": owner_reason,
        "continuous_subtypes": "|".join(subtypes),
        "is_model_target": int(primary_owner in {
            "continuous_value_target",
            "binary_state_target",
            "driver_event_target",
        }),
        "is_generated_value_target": int(primary_owner in {
            "continuous_value_target",
            "binary_state_target",
        }),
        "is_generated_driver_target": int(primary_owner == "driver_event_target"),
        "is_meta_or_conditioning": int(primary_owner in {
            "mask_observability_meta",
            "staleness_meta",
            "conditioning_meta",
            "derived_meta",
        }),
        "is_excluded": int(str(primary_owner).startswith("excluded_")),
        "finite_rate_train": _finite_rate_train(c),
        "finite_rate_val": _finite_rate_val(c),
        "finite_rate_test_audit": _finite_rate_test(c),
        "nunique_dev": int(_nunique_dev(c)),
        "nunique_test_audit": int(_nunique_test(c)),
    })

IOT_BIN_COLS = sorted(set(IOT_BIN_COLS))
IOT_CONT_COLS = sorted(set(IOT_CONT_COLS))
IOT_STATE_COLS = sorted(set(IOT_BIN_COLS) | set(IOT_CONT_COLS))

IOT_EXCLUDED_COLS = sorted(set(IOT_EXCLUDED_COLS))
IOT_EXCLUDED_NON_NUMERIC_COLS = sorted(set(IOT_EXCLUDED_NON_NUMERIC_COLS))
IOT_EXCLUDED_ALL_NAN_COLS = sorted(set(IOT_EXCLUDED_ALL_NAN_COLS))
IOT_EXCLUDED_LOW_SIGNAL_COLS = sorted(set(IOT_EXCLUDED_LOW_SIGNAL_COLS))

IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS = sorted([
    c for c in IOT_EXCLUDED_COLS
    if (
        _finite_rate_dev_max(c) == 0.0
        and (
            _looks_value_like(c)
            or _looks_state_like(c)
            or _looks_state_value_like(c)
        )
    )
])

IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS = sorted([
    c for c in IOT_EXCLUDED_COLS
    if (
        0.0 < _finite_rate_dev_max(c) < MIN_DEV_NUMERIC_RATE
        and (
            _looks_value_like(c)
            or _looks_state_like(c)
            or _looks_state_value_like(c)
        )
    )
])

classification_reason_by_col = {
    r["col"]: r["reason"]
    for r in classification_rows
}

primary_owner_by_col = {
    r["col"]: r["primary_owner"]
    for r in role_ownership_rows
}

# ------------------------------------------------------------
# 8) Derived continuous subtypes
# ------------------------------------------------------------
IOT_STEP_PROGRESS_COLS = sorted([c for c in IOT_CONT_COLS if _is_step_progress_name(c)])
IOT_TELEMETRY_COLS = sorted([c for c in IOT_CONT_COLS if _is_telemetry_name(c)])
IOT_COUNTLIKE_COLS = sorted([c for c in IOT_CONT_COLS if _is_countlike_name(c)])
IOT_DISCRETE_STATE_COLS = sorted([
    c for c in IOT_CONT_COLS
    if (_looks_state_like(c) or _looks_state_value_like(c)) and _nunique_dev(c) <= 16
])
IOT_SETTINGLIKE_COLS = sorted([c for c in IOT_CONT_COLS if _is_setting_like_name(c)])

# ------------------------------------------------------------
# 9) Partition sanity checks
# ------------------------------------------------------------
top_blocks = {
    "driver": set(IOT_DRIVER_COLS),
    "meta": set(IOT_META_COLS),
    "binary": set(IOT_BIN_COLS),
    "continuous": set(IOT_CONT_COLS),
    "excluded": set(IOT_EXCLUDED_COLS),
}

partition_rows = []
for c in ALL_IOT_NAMESPACE_COLS:
    roles = [k for k, s in top_blocks.items() if c in s]
    partition_rows.append({
        "col": c,
        "n_top_roles": len(roles),
        "top_roles": "|".join(roles),
        "primary_owner": primary_owner_by_col.get(c, ""),
        "is_uncategorized": int(len(roles) == 0),
        "is_overlap": int(len(roles) > 1),
        "looks_value_like": int(_looks_value_like(c)),
        "looks_state_like": int(_looks_state_like(c)),
        "looks_state_value_like": int(_looks_state_value_like(c)),
        "is_numeric_like_dev": int(_is_numeric_like_dev(c, MIN_DEV_NUMERIC_RATE)),
        "finite_rate_train": _finite_rate_train(c),
        "finite_rate_val": _finite_rate_val(c),
        "finite_rate_test_audit": _finite_rate_test(c),
    })

partition_audit_df = pd.DataFrame(partition_rows).sort_values(
    ["is_uncategorized", "is_overlap", "col"],
    ascending=[False, False, True]
).reset_index(drop=True)

role_ownership_df = pd.DataFrame(role_ownership_rows).sort_values(
    ["primary_owner", "col"]
).reset_index(drop=True)

classification_df = pd.DataFrame(classification_rows).sort_values(
    ["assigned_top_role", "primary_owner", "col"]
).reset_index(drop=True)

uncategorized = partition_audit_df.loc[partition_audit_df["is_uncategorized"] == 1, "col"].tolist()
overlap = partition_audit_df.loc[partition_audit_df["is_overlap"] == 1, "col"].tolist()

if uncategorized:
    raise RuntimeError(f"[Cell4.5] Unassigned IoT columns remain after partition: {uncategorized[:25]}")
if overlap:
    raise RuntimeError(f"[Cell4.5] Overlapping top-level IoT columns remain after partition: {overlap[:25]}")

for a_name, a_set in top_blocks.items():
    for b_name, b_set in top_blocks.items():
        if a_name >= b_name:
            continue
        inter = sorted(a_set & b_set)
        if inter:
            raise RuntimeError(
                f"[Cell4.5] Top-level overlap {a_name} ∩ {b_name}: {inter[:20]}"
            )

primary_owner_counts_per_col = role_ownership_df.groupby("col")["primary_owner"].nunique()
owner_duplicate_cols = primary_owner_counts_per_col[primary_owner_counts_per_col != 1].index.tolist()
if owner_duplicate_cols:
    raise RuntimeError(
        f"[Cell4.5] Primary-owner assignment is not exactly one per column: {owner_duplicate_cols[:20]}"
    )

owner_assigned_n = int(role_ownership_df["col"].nunique())
if owner_assigned_n != len(ALL_IOT_NAMESPACE_COLS):
    missing_owner = sorted(set(ALL_IOT_NAMESPACE_COLS) - set(role_ownership_df["col"].tolist()))
    extra_owner = sorted(set(role_ownership_df["col"].tolist()) - set(ALL_IOT_NAMESPACE_COLS))
    raise RuntimeError(
        "[Cell4.5] Primary-owner coverage mismatch: "
        f"assigned={owner_assigned_n} namespace={len(ALL_IOT_NAMESPACE_COLS)} "
        f"missing={missing_owner[:20]} extra={extra_owner[:20]}"
    )

# Binary integrity on DEV; TEST is audited below.
bad_bin_dev_cols = [c for c in IOT_BIN_COLS if not _binary01_dev(c)]
if bad_bin_dev_cols:
    raise RuntimeError(f"[Cell4.5] Non-binary DEV columns remain in BIN: {bad_bin_dev_cols[:20]}")

# Split stability.
for nm, _df in [("VAL", df_va), ("TEST", df_te)]:
    for block_name, cols in [
        ("driver", IOT_DRIVER_COLS),
        ("meta", IOT_META_COLS),
        ("binary", IOT_BIN_COLS),
        ("continuous", IOT_CONT_COLS),
        ("excluded", IOT_EXCLUDED_COLS),
    ]:
        miss = [c for c in cols if c not in _df.columns]
        if miss:
            raise RuntimeError(f"[Cell4.5] Split schema drift in {nm} for {block_name}: {miss[:10]}")

# Guard against telemetry helper leakage.
telemetry_helper_leak = [
    c for c in IOT_BIN_COLS + IOT_CONT_COLS
    if _lower(c).startswith("telemetry_in_sec__")
]
if telemetry_helper_leak:
    raise RuntimeError(
        f"[Cell4.5] telemetry_in_sec helper columns leaked into modeled IoT state: {telemetry_helper_leak[:20]}"
    )

# Guard against event stream leakage.
event_stream_leak = [
    c for c in IOT_BIN_COLS + IOT_CONT_COLS + IOT_META_COLS + IOT_EXCLUDED_COLS
    if _lower(c).startswith("events_in_sec__")
]
if event_stream_leak:
    raise RuntimeError(
        f"[Cell4.5] events_in_sec columns leaked outside driver target role: {event_stream_leak[:20]}"
    )

# ------------------------------------------------------------
# 10) Embedded harmful red-flag audit
# ------------------------------------------------------------
red_flag_rows = []

for c in ALL_IOT_NAMESPACE_COLS:
    roles = [k for k, s in top_blocks.items() if c in s]
    subtypes = _continuous_subtypes_for_col(c) if c in IOT_CONT_COLS else []

    reasons = []

    if len(roles) == 0:
        reasons.append("uncategorized")

    if len(roles) > 1:
        reasons.append("top_role_overlap")

    if c in IOT_META_COLS and (_looks_value_like(c) or _looks_state_like(c) or _looks_state_value_like(c)):
        cl = _lower(c)
        if not (
            cl.startswith("telemetry_in_sec__")
            or cl.startswith("iot__entity_obs__")
            or cl.startswith("iot__entity_stale__")
            or "__obs_present" in cl
            or "__stale_flag" in cl
            or "__staleness_s" in cl
        ):
            reasons.append("state_or_value_like_but_meta")

    if c in IOT_BIN_COLS and not _binary01_dev(c):
        reasons.append("binary_role_but_dev_not_binary01")

    if c in IOT_BIN_COLS and len(_finite_vals_test(c)) > 0 and not _binary01_test(c):
        reasons.append("binary_role_but_test_not_binary01")

    if c in IOT_EXCLUDED_COLS and (
        _looks_value_like(c)
        or _looks_state_like(c)
        or _looks_state_value_like(c)
    ):
        dev_rate = float(_finite_rate_dev_max(c))

        if dev_rate == 0.0:
            pass
        elif dev_rate < MIN_DEV_NUMERIC_RATE:
            pass
        else:
            reasons.append("harmful_state_or_value_exclusion_with_dev_support")

    if _lower(c).startswith("telemetry_in_sec__") and c not in IOT_META_COLS:
        reasons.append("telemetry_in_sec_not_meta")

    if _lower(c).startswith("events_in_sec__") and c not in IOT_DRIVER_COLS:
        reasons.append("events_in_sec_not_driver")

    if c in IOT_CONT_COLS and c in IOT_BIN_COLS:
        reasons.append("continuous_binary_overlap")

    if c in IOT_DRIVER_COLS and not _lower(c).startswith("events_in_sec__"):
        reasons.append("driver_without_events_in_sec_prefix")

    if primary_owner_by_col.get(c) in {"continuous_value_target", "binary_state_target"} and c in IOT_META_COLS:
        reasons.append("model_target_also_meta")

    if reasons:
        red_flag_rows.append({
            "col": c,
            "top_roles": "|".join(roles),
            "primary_owner": primary_owner_by_col.get(c, ""),
            "subtypes": "|".join(subtypes),
            "reasons": "|".join(reasons),
            "const_status_dev": _const_status_dev(c),
            "const_status_all_audit": _const_status_all_for_audit(c),
            "binary01_train": int(_binary01_train(c)),
            "binary01_val": int(_binary01_val(c)),
            "binary01_dev": int(_binary01_dev(c)),
            "binary01_test_audit": int(_binary01_test(c)),
            "nunique_dev": int(_nunique_dev(c)),
            "nunique_test_audit": int(_nunique_test(c)),
            "finite_rate_train": float(_finite_rate_train(c)),
            "finite_rate_val": float(_finite_rate_val(c)),
            "finite_rate_test_audit": float(_finite_rate_test(c)),
        })

red_flag_df = (
    pd.DataFrame(red_flag_rows).sort_values(["col"]).reset_index(drop=True)
    if red_flag_rows
    else pd.DataFrame(columns=[
        "col", "top_roles", "primary_owner", "subtypes", "reasons",
        "const_status_dev", "const_status_all_audit",
        "binary01_train", "binary01_val", "binary01_dev", "binary01_test_audit",
        "nunique_dev", "nunique_test_audit",
        "finite_rate_train", "finite_rate_val", "finite_rate_test_audit",
    ])
)

# ------------------------------------------------------------
# 11) Evidence table
# ------------------------------------------------------------
def _binary_like_stats_for_split(df_: pd.DataFrame, c: str) -> Dict[str, Any]:
    x = _finite_vals_from_df(df_, c)
    if x.size == 0:
        return {
            "is_binary_01": False,
            "unique_n": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "non01_frac": np.nan,
            "support_preview": "",
        }

    u = np.unique(x)
    non01_frac = float(np.mean(~np.isin(np.round(x, 6), [0.0, 1.0])))
    return {
        "is_binary_01": bool(set(np.round(u, 6).tolist()).issubset({0.0, 1.0})),
        "unique_n": int(len(u)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "non01_frac": float(non01_frac),
        "support_preview": ",".join(map(str, np.round(u[:8], 4).tolist())),
    }

evidence_rows = []
for c in ALL_IOT_NAMESPACE_COLS:
    stats_tr = _binary_like_stats_for_split(df_tr, c)
    stats_va = _binary_like_stats_for_split(df_va, c)
    stats_te = _binary_like_stats_for_split(df_te, c)

    assigned_top_role = (
        "driver" if c in IOT_DRIVER_COLS else
        "meta" if c in IOT_META_COLS else
        "binary" if c in IOT_BIN_COLS else
        "continuous" if c in IOT_CONT_COLS else
        "excluded" if c in IOT_EXCLUDED_COLS else
        "unassigned"
    )

    subtypes = _continuous_subtypes_for_col(c) if c in IOT_CONT_COLS else []

    soft_flags = []

    if c in IOT_CONT_COLS and _binary01_dev(c):
        soft_flags.append("continuous_but_binary01_dev")

    if c in IOT_CONT_COLS and _binary01_dev(c) and _binary01_test(c):
        soft_flags.append("continuous_but_binary01_all_splits_audit")

    if c in IOT_CONT_COLS and _const_status_dev(c) == "constant":
        soft_flags.append("continuous_but_constant_dev")

    if c in IOT_BIN_COLS and _looks_state_value_like(c):
        soft_flags.append("binary_state_value_like_allowed")

    if c in IOT_EXCLUDED_COLS and not (_looks_value_like(c) or _looks_state_like(c) or _looks_state_value_like(c)):
        soft_flags.append("explicit_non_harmful_exclusion")

    if c in IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS:
        soft_flags.append("state_or_value_excluded_no_dev_support_no_test_leakage")

    if c in IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS:
        soft_flags.append("state_or_value_excluded_low_dev_support_no_test_leakage")

    if _lower(c).startswith("telemetry_in_sec__") and c in IOT_META_COLS:
        soft_flags.append("telemetry_in_sec_meta_helper")

    if primary_owner_by_col.get(c) in {"mask_observability_meta", "staleness_meta"}:
        soft_flags.append("observability_or_staleness_meta_not_value_target")

    evidence_rows.append({
        "col": c,
        "assigned_top_role": assigned_top_role,
        "primary_owner": primary_owner_by_col.get(c, ""),
        "assigned_subtypes": "|".join(subtypes),
        "reason": classification_reason_by_col.get(c, ""),
        "target_cell": _target_cell_for_owner(primary_owner_by_col.get(c, "")),
        "is_numeric_like_dev": int(_is_numeric_like_dev(c, MIN_DEV_NUMERIC_RATE)),
        "finite_rate_train": float(_finite_rate_train(c)),
        "finite_rate_val": float(_finite_rate_val(c)),
        "finite_rate_test_audit": float(_finite_rate_test(c)),
        "looks_value_like": int(_looks_value_like(c)),
        "looks_state_like": int(_looks_state_like(c)),
        "looks_state_value_like": int(_looks_state_value_like(c)),
        "looks_countlike_name": int(_is_countlike_name(c)),
        "looks_telemetry_name": int(_is_telemetry_name(c)),
        "looks_step_progress_name": int(_is_step_progress_name(c)),
        "looks_setting_like_name": int(_is_setting_like_name(c)),
        "is_binary_01_train": stats_tr["is_binary_01"],
        "is_binary_01_val": stats_va["is_binary_01"],
        "is_binary_01_test_audit": stats_te["is_binary_01"],
        "unique_n_train": stats_tr["unique_n"],
        "unique_n_val": stats_va["unique_n"],
        "unique_n_test_audit": stats_te["unique_n"],
        "min_train": stats_tr["min"],
        "max_train": stats_tr["max"],
        "mean_train": stats_tr["mean"],
        "non01_frac_train": stats_tr["non01_frac"],
        "support_preview_train": stats_tr["support_preview"],
        "soft_flags": "|".join(soft_flags),
    })

evidence_df = pd.DataFrame(evidence_rows).sort_values(
    ["primary_owner", "assigned_top_role", "col"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 12) Diagnostics
# ------------------------------------------------------------
entity_obs_cols = sorted([c for c in ALL_IOT_NAMESPACE_COLS if _lower(c).startswith("iot__entity_obs__")])
entity_stale_cols = sorted([c for c in ALL_IOT_NAMESPACE_COLS if _lower(c).startswith("iot__entity_stale__")])
feat_event_cols = sorted([c for c in IOT_DRIVER_COLS if "__feat__" in c])
entity_event_cols = sorted([c for c in IOT_DRIVER_COLS if "__entity__" in c])
telemetry_in_sec_cols = sorted([c for c in ALL_IOT_NAMESPACE_COLS if _lower(c).startswith("telemetry_in_sec__")])

primary_owner_counts = role_ownership_df["primary_owner"].value_counts(dropna=False).sort_index().to_dict()

log(
    f"[Cell4.5] Full IoT namespace: {len(ALL_IOT_NAMESPACE_COLS)} | "
    f"primary_owner_assigned={owner_assigned_n}/{len(ALL_IOT_NAMESPACE_COLS)} | "
    f"primary_owner_duplicates={len(owner_duplicate_cols)} | "
    f"unowned={len(set(ALL_IOT_NAMESPACE_COLS) - set(role_ownership_df['col'].tolist()))}"
)

log(
    f"[Cell4.5] IoT groups: drivers={len(IOT_DRIVER_COLS)} | "
    f"binary={len(IOT_BIN_COLS)} | continuous={len(IOT_CONT_COLS)} | "
    f"countlike={len(IOT_COUNTLIKE_COLS)} | telemetry={len(IOT_TELEMETRY_COLS)} | "
    f"step_progress={len(IOT_STEP_PROGRESS_COLS)} | discrete_state={len(IOT_DISCRETE_STATE_COLS)} | "
    f"setting_like={len(IOT_SETTINGLIKE_COLS)} | meta={len(IOT_META_COLS)} | "
    f"excluded={len(IOT_EXCLUDED_COLS)}"
)

log(f"[Cell4.5] Primary owner counts: {primary_owner_counts}")

log(
    f"[Cell4.5] Diagnostics: entity_obs={len(entity_obs_cols)} | "
    f"entity_stale={len(entity_stale_cols)} | "
    f"entity_driver_like={len(entity_event_cols)} | feat_driver_like={len(feat_event_cols)} | "
    f"telemetry_in_sec_meta={len(telemetry_in_sec_cols)}"
)

log(
    f"[Cell4.5] Exclusions: all_nan_or_no_dev_support={len(IOT_EXCLUDED_ALL_NAN_COLS)} | "
    f"low_dev_signal={len(IOT_EXCLUDED_LOW_SIGNAL_COLS)} | "
    f"non_numeric_total={len(IOT_EXCLUDED_NON_NUMERIC_COLS)} | "
    f"state_value_no_dev_support={len(IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS)} | "
    f"state_value_low_dev_support={len(IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS)}"
)

log(f"[Cell4.5] Harmful red flags={len(red_flag_df)}")

for _, r in red_flag_df.head(25).iterrows():
    log(
        f"[Cell4.5] red-flag | {r['col']} | top={r['top_roles']} | "
        f"owner={r['primary_owner']} | sub={r['subtypes']} | reasons={r['reasons']}"
    )

# ------------------------------------------------------------
# 13) Contract artifacts
# ------------------------------------------------------------
overlap_policy = {
    "version": "iot_overlap_policy_v13",
    "top_level_partition_policy": "driver/meta/binary/continuous/excluded must be exhaustive and disjoint",
    "allowed_subtype_overlaps": {
        "continuous∩countlike": "allowed; countlike is a continuous/value subtype",
        "continuous∩telemetry": "allowed; telemetry is a continuous/value subtype",
        "continuous∩step_progress": "allowed; step_progress is a continuous/value subtype",
        "continuous∩discrete_state": "allowed; discrete_state is a continuous/value subtype",
        "continuous∩setting_like": "allowed; setting_like is a continuous/value subtype",
    },
    "forbidden_top_level_overlaps": [
        "driver∩meta",
        "driver∩binary",
        "driver∩continuous",
        "driver∩excluded",
        "meta∩binary",
        "meta∩continuous",
        "meta∩excluded",
        "binary∩continuous",
        "binary∩excluded",
        "continuous∩excluded",
    ],
    "telemetry_in_sec_policy": "conditioning_meta_only_not_model_target",
    "events_in_sec_policy": "driver_event_target_only",
    "test_usage": "audit_only_never_taxonomy_assignment",
}

cell_routing_manifest = {
    "version": "iot_cell_routing_manifest_v13",
    "role_to_cell": {
        "continuous_value_target": {
            "cell": "12.c",
            "description": "continuous/value IoT target generation, materialization, and QA",
            "count": int((role_ownership_df["primary_owner"] == "continuous_value_target").sum()),
        },
        "binary_state_target": {
            "cell": "12.d",
            "description": "binary IoT state generation, materialization, and QA",
            "count": int((role_ownership_df["primary_owner"] == "binary_state_target").sum()),
        },
        "driver_event_target": {
            "cell": "12.e",
            "description": "IoT driver/event-process generation, materialization, and QA",
            "count": int((role_ownership_df["primary_owner"] == "driver_event_target").sum()),
        },
        "mask_observability_meta": {
            "cell": "12.f / 12.g",
            "description": "observability masks and mask realism, not value targets",
            "count": int((role_ownership_df["primary_owner"] == "mask_observability_meta").sum()),
        },
        "staleness_meta": {
            "cell": "12.f / 12.g",
            "description": "staleness metadata, derived or generated as observability metadata",
            "count": int((role_ownership_df["primary_owner"] == "staleness_meta").sum()),
        },
        "conditioning_meta": {
            "cell": "12.g / conditioning only",
            "description": "conditioning/helper metadata, not primary targets",
            "count": int((role_ownership_df["primary_owner"] == "conditioning_meta").sum()),
        },
        "derived_meta": {
            "cell": "12.g / derived only",
            "description": "derived bookkeeping metadata",
            "count": int((role_ownership_df["primary_owner"] == "derived_meta").sum()),
        },
        "excluded_no_dev_support": {
            "cell": "not generated",
            "description": "excluded because TRAIN+VAL have no numeric support",
            "count": int((role_ownership_df["primary_owner"] == "excluded_no_dev_support").sum()),
        },
        "excluded_low_dev_support": {
            "cell": "not generated",
            "description": "excluded because TRAIN+VAL support is below taxonomy threshold",
            "count": int((role_ownership_df["primary_owner"] == "excluded_low_dev_support").sum()),
        },
        "excluded_non_numeric": {
            "cell": "not generated",
            "description": "excluded because non-numeric or unparseable under TRAIN+VAL",
            "count": int((role_ownership_df["primary_owner"] == "excluded_non_numeric").sum()),
        },
    },
    "continuous_subtype_routes": {
        "countlike": "12.c counter/countlike generator family",
        "telemetry": "12.c telemetry generator family",
        "step_progress": "12.c episode/step-progress generator family",
        "discrete_state": "12.c ordinal/discrete-state generator family unless binary-owned",
        "setting_like": "12.c quasi-static or context generator family",
    },
}

partition_contract = {
    "version": "iot_partition_contract_v13",
    "taxonomy_assignment_source": "TRAIN_PLUS_VAL_ONLY",
    "test_usage": "AUDIT_ONLY",
    "full_iot_namespace_n": int(len(ALL_IOT_NAMESPACE_COLS)),
    "primary_owner_assigned_n": int(owner_assigned_n),
    "primary_owner_duplicates_n": int(len(owner_duplicate_cols)),
    "unowned_iot_cols_n": int(len(set(ALL_IOT_NAMESPACE_COLS) - set(role_ownership_df["col"].tolist()))),
    "top_level_counts": {
        "driver": int(len(IOT_DRIVER_COLS)),
        "meta": int(len(IOT_META_COLS)),
        "binary": int(len(IOT_BIN_COLS)),
        "continuous": int(len(IOT_CONT_COLS)),
        "excluded": int(len(IOT_EXCLUDED_COLS)),
    },
    "primary_owner_counts": {str(k): int(v) for k, v in primary_owner_counts.items()},
    "subtype_counts": {
        "countlike": int(len(IOT_COUNTLIKE_COLS)),
        "telemetry": int(len(IOT_TELEMETRY_COLS)),
        "step_progress": int(len(IOT_STEP_PROGRESS_COLS)),
        "discrete_state": int(len(IOT_DISCRETE_STATE_COLS)),
        "setting_like": int(len(IOT_SETTINGLIKE_COLS)),
    },
    "schema_stable_across_splits": True,
    "harmful_red_flags_n": int(len(red_flag_df)),
    "partition_audit_csv": "iot_feature_grouping_partition_audit.csv",
    "role_ownership_csv": "iot_role_ownership.csv",
}

# ------------------------------------------------------------
# 14) Export globals
# ------------------------------------------------------------
globals()["IOT_DRIVER_COLS"] = list(IOT_DRIVER_COLS)
globals()["IOT_BIN_COLS"] = list(IOT_BIN_COLS)
globals()["IOT_CONT_COLS"] = list(IOT_CONT_COLS)
globals()["IOT_STATE_COLS"] = list(IOT_STATE_COLS)
globals()["IOT_META_COLS"] = list(IOT_META_COLS)

globals()["IOT_COUNTLIKE_COLS"] = list(IOT_COUNTLIKE_COLS)
globals()["IOT_TELEMETRY_COLS"] = list(IOT_TELEMETRY_COLS)
globals()["IOT_STEP_PROGRESS_COLS"] = list(IOT_STEP_PROGRESS_COLS)
globals()["IOT_DISCRETE_STATE_COLS"] = list(IOT_DISCRETE_STATE_COLS)
globals()["IOT_SETTINGLIKE_COLS"] = list(IOT_SETTINGLIKE_COLS)

globals()["IOT_EXCLUDED_COLS"] = list(IOT_EXCLUDED_COLS)
globals()["IOT_EXCLUDED_NON_NUMERIC_COLS"] = list(IOT_EXCLUDED_NON_NUMERIC_COLS)
globals()["IOT_EXCLUDED_ALL_NAN_COLS"] = list(IOT_EXCLUDED_ALL_NAN_COLS)
globals()["IOT_EXCLUDED_LOW_SIGNAL_COLS"] = list(IOT_EXCLUDED_LOW_SIGNAL_COLS)
globals()["IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS"] = list(IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS)
globals()["IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS"] = list(IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS)

globals()["IOT_FULL_NAMESPACE_COLS"] = list(ALL_IOT_NAMESPACE_COLS)
globals()["IOT_TELEMETRY_IN_SEC_COLS"] = list(telemetry_in_sec_cols)

globals()["IOT_ROLE_OWNERSHIP_DF"] = role_ownership_df.copy()
globals()["IOT_FEATURE_GROUPING_EVIDENCE_DF"] = evidence_df.copy()
globals()["IOT_FEATURE_GROUPING_PARTITION_AUDIT_DF"] = partition_audit_df.copy()
globals()["IOT_FEATURE_GROUPING_RED_FLAG_DF"] = red_flag_df.copy()

# Backward-compatible aliases.
globals()["IOT_BINARY_COLS"] = list(IOT_BIN_COLS)
globals()["IOT_CONTINUOUS_COLS"] = list(IOT_CONT_COLS)
globals()["iot_drivers"] = list(IOT_DRIVER_COLS)
globals()["iot_binary"] = list(IOT_BIN_COLS)
globals()["iot_continuous"] = list(IOT_CONT_COLS)

# New explicit role aliases for Phase 1+ downstream cells.
globals()["IOT_CONTINUOUS_VALUE_TARGET_COLS"] = list(IOT_CONT_COLS)
globals()["IOT_BINARY_TARGET_COLS"] = list(IOT_BIN_COLS)
globals()["IOT_DRIVER_TARGET_COLS"] = list(IOT_DRIVER_COLS)
globals()["IOT_MASK_OBSERVABILITY_META_COLS"] = role_ownership_df.loc[
    role_ownership_df["primary_owner"] == "mask_observability_meta", "col"
].tolist()
globals()["IOT_STALENESS_META_COLS"] = role_ownership_df.loc[
    role_ownership_df["primary_owner"] == "staleness_meta", "col"
].tolist()
globals()["IOT_CONDITIONING_META_COLS"] = role_ownership_df.loc[
    role_ownership_df["primary_owner"] == "conditioning_meta", "col"
].tolist()
globals()["IOT_DERIVED_META_COLS"] = role_ownership_df.loc[
    role_ownership_df["primary_owner"] == "derived_meta", "col"
].tolist()

# ------------------------------------------------------------
# 15) Persist artifacts
# ------------------------------------------------------------
evidence_path = os.path.join(art_dir, "iot_feature_grouping_evidence.csv")
partition_audit_path = os.path.join(art_dir, "iot_feature_grouping_partition_audit.csv")
red_flags_path = os.path.join(art_dir, "iot_feature_grouping_red_flags.csv")
role_ownership_path = os.path.join(art_dir, "iot_role_ownership.csv")
partition_contract_path = os.path.join(art_dir, "iot_partition_contract.json")
overlap_policy_path = os.path.join(art_dir, "iot_overlap_policy.json")
cell_routing_manifest_path = os.path.join(art_dir, "iot_cell_routing_manifest.json")

evidence_df.to_csv(evidence_path, index=False)
partition_audit_df.to_csv(partition_audit_path, index=False)
red_flag_df.to_csv(red_flags_path, index=False)
role_ownership_df.to_csv(role_ownership_path, index=False)

with open(partition_contract_path, "w", encoding="utf-8") as f:
    json.dump(_json_safe(partition_contract), f, indent=2, ensure_ascii=False)

with open(overlap_policy_path, "w", encoding="utf-8") as f:
    json.dump(_json_safe(overlap_policy), f, indent=2, ensure_ascii=False)

with open(cell_routing_manifest_path, "w", encoding="utf-8") as f:
    json.dump(_json_safe(cell_routing_manifest), f, indent=2, ensure_ascii=False)

groups = {
    "version": "v13_no_test_leakage_full_iot_role_ownership",
    "SEC_COL": SEC_COL,
    "taxonomy_assignment_source": "TRAIN_PLUS_VAL_ONLY",
    "test_usage": "AUDIT_ONLY",
    "min_dev_numeric_rate": float(MIN_DEV_NUMERIC_RATE),
    "counts": {
        "full_iot_namespace": int(len(ALL_IOT_NAMESPACE_COLS)),
        "primary_owner_assigned": int(owner_assigned_n),
        "primary_owner_duplicates": int(len(owner_duplicate_cols)),
        "drivers": int(len(IOT_DRIVER_COLS)),
        "binary": int(len(IOT_BIN_COLS)),
        "continuous": int(len(IOT_CONT_COLS)),
        "state_total": int(len(IOT_STATE_COLS)),
        "meta": int(len(IOT_META_COLS)),
        "telemetry_in_sec_meta": int(len(telemetry_in_sec_cols)),
        "countlike": int(len(IOT_COUNTLIKE_COLS)),
        "telemetry": int(len(IOT_TELEMETRY_COLS)),
        "step_progress": int(len(IOT_STEP_PROGRESS_COLS)),
        "discrete_state": int(len(IOT_DISCRETE_STATE_COLS)),
        "setting_like": int(len(IOT_SETTINGLIKE_COLS)),
        "excluded_total": int(len(IOT_EXCLUDED_COLS)),
        "excluded_non_numeric": int(len(IOT_EXCLUDED_NON_NUMERIC_COLS)),
        "excluded_all_nan": int(len(IOT_EXCLUDED_ALL_NAN_COLS)),
        "excluded_low_signal": int(len(IOT_EXCLUDED_LOW_SIGNAL_COLS)),
        "excluded_state_value_no_dev_support": int(len(IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS)),
        "excluded_state_value_low_dev_support": int(len(IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS)),
        "harmful_red_flags": int(len(red_flag_df)),
        "entity_obs": int(len(entity_obs_cols)),
        "entity_stale": int(len(entity_stale_cols)),
        "entity_event_drivers": int(len(entity_event_cols)),
        "feat_event_drivers": int(len(feat_event_cols)),
    },
    "primary_owner_counts": {str(k): int(v) for k, v in primary_owner_counts.items()},
    "IOT_FULL_NAMESPACE_COLS": list(ALL_IOT_NAMESPACE_COLS),
    "IOT_DRIVER_COLS": list(IOT_DRIVER_COLS),
    "IOT_BIN_COLS": list(IOT_BIN_COLS),
    "IOT_CONT_COLS": list(IOT_CONT_COLS),
    "IOT_STATE_COLS": list(IOT_STATE_COLS),
    "IOT_META_COLS": list(IOT_META_COLS),
    "IOT_TELEMETRY_IN_SEC_COLS": list(telemetry_in_sec_cols),
    "IOT_COUNTLIKE_COLS": list(IOT_COUNTLIKE_COLS),
    "IOT_TELEMETRY_COLS": list(IOT_TELEMETRY_COLS),
    "IOT_STEP_PROGRESS_COLS": list(IOT_STEP_PROGRESS_COLS),
    "IOT_DISCRETE_STATE_COLS": list(IOT_DISCRETE_STATE_COLS),
    "IOT_SETTINGLIKE_COLS": list(IOT_SETTINGLIKE_COLS),
    "IOT_EXCLUDED_COLS": list(IOT_EXCLUDED_COLS),
    "IOT_EXCLUDED_NON_NUMERIC_COLS": list(IOT_EXCLUDED_NON_NUMERIC_COLS),
    "IOT_EXCLUDED_ALL_NAN_COLS": list(IOT_EXCLUDED_ALL_NAN_COLS),
    "IOT_EXCLUDED_LOW_SIGNAL_COLS": list(IOT_EXCLUDED_LOW_SIGNAL_COLS),
    "IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS": list(IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS),
    "IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS": list(IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS),
    "entity_obs_cols": list(entity_obs_cols),
    "entity_stale_cols": list(entity_stale_cols),
    "artifacts": {
        "evidence_csv": os.path.basename(evidence_path),
        "partition_audit_csv": os.path.basename(partition_audit_path),
        "red_flags_csv": os.path.basename(red_flags_path),
        "role_ownership_csv": os.path.basename(role_ownership_path),
        "partition_contract_json": os.path.basename(partition_contract_path),
        "overlap_policy_json": os.path.basename(overlap_policy_path),
        "cell_routing_manifest_json": os.path.basename(cell_routing_manifest_path),
    },
    "notes": [
        "Top-level IoT partition is exhaustive and disjoint over the full IoT-related namespace.",
        "Every IoT-related column has exactly one primary owner.",
        "Taxonomy assignment uses TRAIN+VAL only; TEST is audit-only.",
        "Binary __value state fields are allowed when strict 0/1 on DEV.",
        "telemetry_in_sec__ columns are conditioning/meta helpers, not modeled IoT state targets.",
        "events_in_sec__ columns are driver/event targets and route to Cell 12.e.",
        "COUNTLIKE / TELEMETRY / STEP_PROGRESS / DISCRETE_STATE / SETTINGLIKE are continuous-target subtypes.",
        "Only genuinely harmful taxonomy issues appear in red_flags.csv.",
    ],
}

out_path = os.path.join(art_dir, "iot_feature_groups.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(_json_safe(groups), f, indent=2, ensure_ascii=False)

log(f"[Cell4.5] Saved grouping evidence: {evidence_path}")
log(f"[Cell4.5] Saved partition audit: {partition_audit_path}")
log(f"[Cell4.5] Saved red flags: {red_flags_path}")
log(f"[Cell4.5] Saved role ownership: {role_ownership_path}")
log(f"[Cell4.5] Saved partition contract: {partition_contract_path}")
log(f"[Cell4.5] Saved overlap policy: {overlap_policy_path}")
log(f"[Cell4.5] Saved cell routing manifest: {cell_routing_manifest_path}")
log(f"[Cell4.5] Saved: {out_path}")

# ------------------------------------------------------------
# 16) Downstream semantic contract export
# ------------------------------------------------------------
IOT_SEMANTIC_CONTRACT = {
    "version": "iot_semantic_contract_v13_full_role_ownership_no_test_leakage",
    "taxonomy_assignment_source": "TRAIN_PLUS_VAL_ONLY",
    "test_usage": "AUDIT_ONLY",
    "full_iot_namespace_cols": list(ALL_IOT_NAMESPACE_COLS),
    "driver_cols": list(IOT_DRIVER_COLS),
    "meta_cols": list(IOT_META_COLS),
    "binary_cols": list(IOT_BIN_COLS),
    "continuous_cols": list(IOT_CONT_COLS),
    "state_cols": list(IOT_STATE_COLS),
    "telemetry_in_sec_meta_cols": list(telemetry_in_sec_cols),
    "excluded_cols": list(IOT_EXCLUDED_COLS),
    "excluded_state_value_no_dev_support_cols": list(IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS),
    "excluded_state_value_low_dev_support_cols": list(IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS),
    "primary_owner_counts": {str(k): int(v) for k, v in primary_owner_counts.items()},
    "role_ownership_csv": os.path.basename(role_ownership_path),
    "partition_contract_json": os.path.basename(partition_contract_path),
    "overlap_policy_json": os.path.basename(overlap_policy_path),
    "cell_routing_manifest_json": os.path.basename(cell_routing_manifest_path),
    "n_full_iot_namespace": int(len(ALL_IOT_NAMESPACE_COLS)),
    "n_primary_owner_assigned": int(owner_assigned_n),
    "n_driver": int(len(IOT_DRIVER_COLS)),
    "n_meta": int(len(IOT_META_COLS)),
    "n_binary": int(len(IOT_BIN_COLS)),
    "n_continuous": int(len(IOT_CONT_COLS)),
    "n_state": int(len(IOT_STATE_COLS)),
    "n_excluded": int(len(IOT_EXCLUDED_COLS)),
    "n_harmful_red_flags": int(len(red_flag_df)),
    "target_cells": {
        "continuous_value_target": "Cell 12.c",
        "binary_state_target": "Cell 12.d",
        "driver_event_target": "Cell 12.e",
        "mask_observability_meta": "Cell 12.f / Cell 12.g",
        "staleness_meta": "Cell 12.f / Cell 12.g",
        "conditioning_meta": "Cell 12.g / conditioning only",
        "derived_meta": "Cell 12.g / derived only",
        "excluded_*": "not generated",
    },
}

sem_contract_path = os.path.join(art_dir, "iot_semantic_contract.json")
with open(sem_contract_path, "w", encoding="utf-8") as f:
    json.dump(_json_safe(IOT_SEMANTIC_CONTRACT), f, indent=2, ensure_ascii=False)

globals()["IOT_SEMANTIC_CONTRACT"] = IOT_SEMANTIC_CONTRACT
globals()["IOT_SEMANTIC_CONTRACT_JSON"] = sem_contract_path
globals()["IOT_PARTITION_CONTRACT_JSON"] = partition_contract_path
globals()["IOT_OVERLAP_POLICY_JSON"] = overlap_policy_path
globals()["IOT_CELL_ROUTING_MANIFEST_JSON"] = cell_routing_manifest_path
globals()["IOT_ROLE_OWNERSHIP_CSV"] = role_ownership_path

log(f"[Cell4.5] Downstream semantic contract saved: {sem_contract_path}")

# ------------------------------------------------------------
# 17) Final hard gate
# ------------------------------------------------------------
if len(red_flag_df):
    raise RuntimeError(
        f"[Cell4.5] Harmful taxonomy red flags remain: {len(red_flag_df)}. "
        f"See {red_flags_path}"
    )

log(
    "[Cell4.5] DONE | "
    f"IOT_FULL_NAMESPACE_COLS={len(ALL_IOT_NAMESPACE_COLS)} | "
    f"IOT_CONTINUOUS_VALUE_TARGETS={len(IOT_CONT_COLS)} | "
    f"IOT_BINARY_TARGETS={len(IOT_BIN_COLS)} | "
    f"IOT_DRIVER_TARGETS={len(IOT_DRIVER_COLS)} | "
    f"IOT_META_COLS={len(IOT_META_COLS)} | "
    f"IOT_EXCLUDED_COLS={len(IOT_EXCLUDED_COLS)} | "
    f"PRIMARY_OWNER_ASSIGNED={owner_assigned_n}/{len(ALL_IOT_NAMESPACE_COLS)} | "
    f"PRIMARY_OWNER_DUPLICATES={len(owner_duplicate_cols)} | "
    f"HARMFUL_RED_FLAGS={len(red_flag_df)}"
)

log("--- END:   Cell 4.5 — IoT feature grouping / role ownership contract ---")