# ==========================================================
# CELL 6.5 — IoT group contract loader — v8
# CANONICAL / STRICT / v13 ROLE-OWNERSHIP / CPS-COUPLED CONTRACT
# ==========================================================
#
# Purpose:
# - Reload and validate the canonical IoT grouping contract from Cell 4.5 v13
# - Re-export stable globals for downstream IoT cells
# - Validate the full IoT role-ownership contract:
#     full namespace
#     primary ownership
#     drivers / binary / continuous / meta / excluded
#     semantic subtypes
#     cell routing manifest
#     overlap policy
# - Cross-check:
#     iot_feature_groups.json
#     iot_semantic_contract.json
#     iot_role_ownership.csv
#     iot_partition_contract.json
#     iot_overlap_policy.json
#     iot_cell_routing_manifest.json
# - Bind IoT synthesis to the frozen CPS/protocol timeline contract
#
# Policy:
# - Do not reconstruct groups heuristically here
# - Do not silently ignore missing v13 artifacts
# - Do not mutate df_tr/df_va/df_te
# - Do not reorder canonical groups from Cell 4.5
# - Fail closed on split schema drift, unowned columns, duplicate owners,
#   or partition inconsistency
# - IoT downstream synthesis must not be standalone; it must be conditioned on
#   the same CPS timeline, TOD, regimes, protocol masks, and protocol activity
# ==========================================================

log("--- START: Cell 6.5 — IoT group contract loader (v8 v13 role-ownership CPS-coupled) ---")

import os
import json
import hashlib
import itertools
import numpy as np
import pandas as pd

need = [
    "CFG", "df_tr", "df_va", "df_te", "log",
    "PROTO_COLS_CONTRACT",
    "PROTOCOL_TIERS_USED",
    "PROTO_OBS_TR", "PROTO_OBS_VA", "PROTO_OBS_TE",
    "tod_tr", "tod_va", "tod_te",
    "reg_tr", "reg_va", "reg_te",
    "K_reg",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(
        f"[Cell6.5] Missing prerequisites: {missing}. "
        "Run Cells 3–6 and Cell 4.5 first."
    )

if "PROTOCOL_GENERATOR_ROLE_CONTRACT" not in globals():
    raise RuntimeError(
        "[Cell6.5] Missing PROTOCOL_GENERATOR_ROLE_CONTRACT. "
        "Run the updated Cell 6 first. IoT synthesis must be coupled to the "
        "protocol generator-role contract, not treated as standalone IoT."
    )

art_dir = os.path.join(str(CFG["outdir"]), "artifacts")
rep_dir = os.path.join(str(CFG["outdir"]), "reports")
os.makedirs(art_dir, exist_ok=True)
os.makedirs(rep_dir, exist_ok=True)

groups_path = os.path.join(art_dir, "iot_feature_groups.json")
semantic_path = os.path.join(art_dir, "iot_semantic_contract.json")
role_ownership_path = os.path.join(art_dir, "iot_role_ownership.csv")
partition_contract_path = os.path.join(art_dir, "iot_partition_contract.json")
overlap_policy_path = os.path.join(art_dir, "iot_overlap_policy.json")
cell_routing_manifest_path = os.path.join(art_dir, "iot_cell_routing_manifest.json")

protocol_role_path = os.path.join(art_dir, "protocol_generator_role_contract.json")
protocol_policy_path = os.path.join(art_dir, "protocol_synthesis_policy.json")

required_paths = {
    "iot_feature_groups.json": groups_path,
    "iot_semantic_contract.json": semantic_path,
    "iot_role_ownership.csv": role_ownership_path,
    "iot_partition_contract.json": partition_contract_path,
    "iot_overlap_policy.json": overlap_policy_path,
    "iot_cell_routing_manifest.json": cell_routing_manifest_path,
}
missing_paths = [name for name, path in required_paths.items() if not os.path.exists(path)]
if missing_paths:
    raise RuntimeError(
        f"[Cell6.5] Missing required Cell 4.5 v13 artifacts: {missing_paths}. "
        "Re-run Cell 4.5 v13 before Cell 6.5."
    )

with open(groups_path, "r", encoding="utf-8") as f:
    obj = json.load(f)

with open(semantic_path, "r", encoding="utf-8") as f:
    sem_obj = json.load(f)

with open(partition_contract_path, "r", encoding="utf-8") as f:
    partition_contract = json.load(f)

with open(overlap_policy_path, "r", encoding="utf-8") as f:
    overlap_policy = json.load(f)

with open(cell_routing_manifest_path, "r", encoding="utf-8") as f:
    cell_routing_manifest = json.load(f)

role_ownership_df = pd.read_csv(role_ownership_path)

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _write_json(path, payload) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)

def _sha_list(xs) -> str:
    return hashlib.sha256(("||".join(map(str, list(xs)))).encode("utf-8")).hexdigest()

def _clean_unique_list(name, xs, *, required=True):
    if xs is None:
        if required:
            raise RuntimeError(f"[Cell6.5] Missing required list: {name}")
        return []

    if not isinstance(xs, list):
        raise RuntimeError(f"[Cell6.5] {name} must be a list, got {type(xs)}.")

    out = []
    seen = set()
    for x in xs:
        if not isinstance(x, str):
            raise RuntimeError(f"[Cell6.5] {name} contains non-string entry: {x!r}")
        if x in seen:
            raise RuntimeError(f"[Cell6.5] {name} contains duplicate entry: {x}")
        seen.add(x)
        out.append(x)

    return out

def _require_cols_in_splits(cols, block_name):
    cols = list(cols)
    for nm, _df in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
        miss = [c for c in cols if c not in _df.columns]
        if miss:
            raise RuntimeError(
                f"[Cell6.5] Split schema drift in {nm} for {block_name}: {miss[:20]}"
            )

def _assert_disjoint(name_a, a, name_b, b):
    inter = sorted(set(a).intersection(set(b)))
    if inter:
        raise RuntimeError(
            f"[Cell6.5] {name_a} ∩ {name_b} is not empty ({len(inter)}): {inter[:20]}"
        )

def _same_set(name_a, a, name_b, b):
    sa, sb = set(a), set(b)
    if sa != sb:
        only_a = sorted(sa - sb)
        only_b = sorted(sb - sa)
        raise RuntimeError(
            f"[Cell6.5] Set mismatch {name_a} vs {name_b}. "
            f"only_{name_a}={only_a[:20]} | only_{name_b}={only_b[:20]}"
        )

def _same_ordered_list(name_a, a, name_b, b):
    a = list(a)
    b = list(b)
    if a != b:
        first_bad = None
        for i, (x, y) in enumerate(itertools.zip_longest(a, b, fillvalue="<MISSING>")):
            if x != y:
                first_bad = (i, x, y)
                break
        raise RuntimeError(
            f"[Cell6.5] Ordered-list mismatch {name_a} vs {name_b}. "
            f"len_a={len(a)} len_b={len(b)} first_bad={first_bad}"
        )

def _count_check(counts, key, xs):
    if not isinstance(counts, dict):
        raise RuntimeError("[Cell6.5] groups['counts'] must be a dict.")
    expected = counts.get(key, None)
    if expected is None:
        raise RuntimeError(f"[Cell6.5] groups['counts'] missing key: {key}")
    if int(expected) != len(xs):
        raise RuntimeError(
            f"[Cell6.5] Count mismatch for {key}: json={expected} actual={len(xs)}"
        )

def _is_iot_related(c: str) -> bool:
    cl = str(c).lower()
    return (
        cl.startswith("iot__")
        or cl.startswith("events_in_sec__")
        or cl.startswith("telemetry_in_sec__")
    )

def _shape_list(x):
    return [int(v) for v in getattr(x, "shape", [])]

def _require_len(name, arr, expected):
    if len(arr) != int(expected):
        raise RuntimeError(
            f"[Cell6.5] Length mismatch for {name}: got={len(arr)} expected={int(expected)}"
        )

def _first_existing_key(d, keys, *, required_name=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return k
    if required_name is not None:
        raise RuntimeError(f"[Cell6.5] Missing required key for {required_name}; tried={keys}")
    return None

def _first_existing_col(df, cols, *, required_name=None):
    for c in cols:
        if c in df.columns:
            return c
    if required_name is not None:
        raise RuntimeError(
            f"[Cell6.5] Missing required column in role ownership CSV for {required_name}; "
            f"tried={cols}; available={list(df.columns)[:40]}"
        )
    return None

# ----------------------------------------------------------
# Required keys from Cell 4.5 v13
# ----------------------------------------------------------
required_group_keys = [
    "version",
    "SEC_COL",
    "taxonomy_assignment_source",
    "test_usage",
    "counts",
    "IOT_FULL_NAMESPACE_COLS",
    "IOT_DRIVER_COLS",
    "IOT_BIN_COLS",
    "IOT_CONT_COLS",
    "IOT_STATE_COLS",
    "IOT_META_COLS",
    "IOT_TELEMETRY_IN_SEC_COLS",
    "IOT_COUNTLIKE_COLS",
    "IOT_TELEMETRY_COLS",
    "IOT_STEP_PROGRESS_COLS",
    "IOT_DISCRETE_STATE_COLS",
    "IOT_SETTINGLIKE_COLS",
    "IOT_EXCLUDED_COLS",
    "IOT_EXCLUDED_NON_NUMERIC_COLS",
    "IOT_EXCLUDED_ALL_NAN_COLS",
    "IOT_EXCLUDED_LOW_SIGNAL_COLS",
    "IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS",
    "IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS",
]
missing_keys = [k for k in required_group_keys if k not in obj]
if missing_keys:
    raise RuntimeError(
        f"[Cell6.5] Missing required keys in iot_feature_groups.json: {missing_keys}"
    )

if str(obj.get("taxonomy_assignment_source", "")) != "TRAIN_PLUS_VAL_ONLY":
    raise RuntimeError(
        f"[Cell6.5] Unexpected taxonomy_assignment_source: {obj.get('taxonomy_assignment_source')}"
    )

if str(obj.get("test_usage", "")) != "AUDIT_ONLY":
    raise RuntimeError(
        f"[Cell6.5] Unexpected test_usage: {obj.get('test_usage')}"
    )

# ----------------------------------------------------------
# Load canonical groups
# ----------------------------------------------------------
SEC_COL = str(obj["SEC_COL"])

full_namespace = _clean_unique_list("IOT_FULL_NAMESPACE_COLS", obj["IOT_FULL_NAMESPACE_COLS"])
drv = _clean_unique_list("IOT_DRIVER_COLS", obj["IOT_DRIVER_COLS"])
bin_ = _clean_unique_list("IOT_BIN_COLS", obj["IOT_BIN_COLS"])
con = _clean_unique_list("IOT_CONT_COLS", obj["IOT_CONT_COLS"])
state = _clean_unique_list("IOT_STATE_COLS", obj["IOT_STATE_COLS"])
meta = _clean_unique_list("IOT_META_COLS", obj["IOT_META_COLS"])

telemetry_in_sec = _clean_unique_list("IOT_TELEMETRY_IN_SEC_COLS", obj["IOT_TELEMETRY_IN_SEC_COLS"])

countlike = _clean_unique_list("IOT_COUNTLIKE_COLS", obj["IOT_COUNTLIKE_COLS"])
telemetry = _clean_unique_list("IOT_TELEMETRY_COLS", obj["IOT_TELEMETRY_COLS"])
step_progress = _clean_unique_list("IOT_STEP_PROGRESS_COLS", obj["IOT_STEP_PROGRESS_COLS"])
discrete_state = _clean_unique_list("IOT_DISCRETE_STATE_COLS", obj["IOT_DISCRETE_STATE_COLS"])
settinglike = _clean_unique_list("IOT_SETTINGLIKE_COLS", obj["IOT_SETTINGLIKE_COLS"])

excluded = _clean_unique_list("IOT_EXCLUDED_COLS", obj["IOT_EXCLUDED_COLS"])
excluded_non_numeric = _clean_unique_list("IOT_EXCLUDED_NON_NUMERIC_COLS", obj["IOT_EXCLUDED_NON_NUMERIC_COLS"])
excluded_all_nan = _clean_unique_list("IOT_EXCLUDED_ALL_NAN_COLS", obj["IOT_EXCLUDED_ALL_NAN_COLS"])
excluded_low_signal = _clean_unique_list("IOT_EXCLUDED_LOW_SIGNAL_COLS", obj["IOT_EXCLUDED_LOW_SIGNAL_COLS"])
excluded_state_value_no_dev = _clean_unique_list(
    "IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS",
    obj["IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS"],
)
excluded_state_value_low_dev = _clean_unique_list(
    "IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS",
    obj["IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS"],
)

counts = obj["counts"]

# ----------------------------------------------------------
# Validate v13 role ownership CSV
# ----------------------------------------------------------
role_col = _first_existing_col(
    role_ownership_df,
    ["col", "column", "feature", "name"],
    required_name="role ownership column name",
)
owner_col = _first_existing_col(
    role_ownership_df,
    ["primary_owner", "primary_role", "owner", "assigned_primary_owner", "assigned_owner"],
    required_name="primary owner",
)

role_ownership_df[role_col] = role_ownership_df[role_col].astype(str)
role_ownership_df[owner_col] = role_ownership_df[owner_col].astype(str)

if role_ownership_df[role_col].duplicated().any():
    dup = role_ownership_df.loc[role_ownership_df[role_col].duplicated(), role_col].head(20).tolist()
    raise RuntimeError(f"[Cell6.5] Duplicate columns in iot_role_ownership.csv: {dup}")

role_cols = role_ownership_df[role_col].tolist()
_same_set("groups.full_namespace", full_namespace, "role_ownership.cols", role_cols)

owner_by_col = dict(zip(role_ownership_df[role_col], role_ownership_df[owner_col]))

allowed_primary_owners = {
    "continuous_value_target",
    "binary_state_target",
    "driver_event_target",
    "conditioning_meta",
    "derived_meta",
    "mask_observability_meta",
    "staleness_meta",
    "excluded_no_dev_support",
}

bad_owners = sorted(set(owner_by_col.values()) - allowed_primary_owners)
if bad_owners:
    raise RuntimeError(
        f"[Cell6.5] Unknown primary owners in role ownership contract: {bad_owners}"
    )

missing_owner = sorted([c for c in full_namespace if not str(owner_by_col.get(c, "")).strip()])
if missing_owner:
    raise RuntimeError(f"[Cell6.5] Unowned IoT columns in role ownership contract: {missing_owner[:20]}")

owner_counts = pd.Series(owner_by_col).value_counts().to_dict()

expected_owner_sets = {
    "driver_event_target": set(drv),
    "binary_state_target": set(bin_),
    "continuous_value_target": set(con),
    "excluded_no_dev_support": set(excluded),
    "conditioning_meta": set(telemetry_in_sec),
}

for owner_name, expected_set in expected_owner_sets.items():
    actual_set = {c for c, owner in owner_by_col.items() if owner == owner_name}
    if owner_name == "conditioning_meta":
        if not expected_set.issubset(actual_set):
            missing_conditioning = sorted(expected_set - actual_set)
            raise RuntimeError(
                f"[Cell6.5] telemetry_in_sec columns must be conditioning_meta. "
                f"Missing={missing_conditioning[:20]}"
            )
    else:
        if actual_set != expected_set:
            only_actual = sorted(actual_set - expected_set)
            only_expected = sorted(expected_set - actual_set)
            raise RuntimeError(
                f"[Cell6.5] Primary owner mismatch for {owner_name}. "
                f"only_actual={only_actual[:20]} | only_expected={only_expected[:20]}"
            )

meta_owner_set = {
    c for c, owner in owner_by_col.items()
    if owner in {"conditioning_meta", "derived_meta", "mask_observability_meta", "staleness_meta"}
}
if meta_owner_set != set(meta):
    only_meta_owner = sorted(meta_owner_set - set(meta))
    only_meta = sorted(set(meta) - meta_owner_set)
    raise RuntimeError(
        "[Cell6.5] Meta owner union does not equal IOT_META_COLS. "
        f"only_owner_meta={only_meta_owner[:20]} | only_IOT_META={only_meta[:20]}"
    )

# ----------------------------------------------------------
# CPS timeline/protocol context checks
# ----------------------------------------------------------
if SEC_COL not in df_tr.columns or SEC_COL not in df_va.columns or SEC_COL not in df_te.columns:
    raise RuntimeError(f"[Cell6.5] SEC_COL missing from one or more splits: {SEC_COL}")

_require_len("tod_tr", globals()["tod_tr"], len(df_tr))
_require_len("tod_va", globals()["tod_va"], len(df_va))
_require_len("tod_te", globals()["tod_te"], len(df_te))
_require_len("reg_tr", globals()["reg_tr"], len(df_tr))
_require_len("reg_va", globals()["reg_va"], len(df_va))
_require_len("reg_te", globals()["reg_te"], len(df_te))

if _shape_list(globals()["PROTO_OBS_TR"]) != [len(df_tr), len(globals()["PROTOCOL_TIERS_USED"])]:
    raise RuntimeError("[Cell6.5] PROTO_OBS_TR shape does not match protocol tier contract.")
if _shape_list(globals()["PROTO_OBS_VA"]) != [len(df_va), len(globals()["PROTOCOL_TIERS_USED"])]:
    raise RuntimeError("[Cell6.5] PROTO_OBS_VA shape does not match protocol tier contract.")
if _shape_list(globals()["PROTO_OBS_TE"]) != [len(df_te), len(globals()["PROTOCOL_TIERS_USED"])]:
    raise RuntimeError("[Cell6.5] PROTO_OBS_TE shape does not match protocol tier contract.")

proto_contract = list(globals()["PROTO_COLS_CONTRACT"])
protocol_generator_role_contract = globals()["PROTOCOL_GENERATOR_ROLE_CONTRACT"]

if not isinstance(protocol_generator_role_contract, dict):
    raise RuntimeError("[Cell6.5] PROTOCOL_GENERATOR_ROLE_CONTRACT must be a dict.")

role_proto_cols = list(protocol_generator_role_contract.get("proto_cols_contract", []))
if role_proto_cols and role_proto_cols != proto_contract:
    raise RuntimeError(
        "[Cell6.5] PROTOCOL_GENERATOR_ROLE_CONTRACT proto_cols_contract differs from "
        "global PROTO_COLS_CONTRACT. Re-run Cells 5 and 6 in order."
    )

# ----------------------------------------------------------
# Basic schema checks
# ----------------------------------------------------------
for c in full_namespace:
    if not _is_iot_related(c):
        raise RuntimeError(f"[Cell6.5] Non-IoT-related column inside IOT_FULL_NAMESPACE_COLS: {c}")

_require_cols_in_splits(full_namespace, "IOT_FULL_NAMESPACE_COLS")
_require_cols_in_splits(drv, "IOT_DRIVER_COLS")
_require_cols_in_splits(bin_, "IOT_BIN_COLS")
_require_cols_in_splits(con, "IOT_CONT_COLS")
_require_cols_in_splits(state, "IOT_STATE_COLS")
_require_cols_in_splits(meta, "IOT_META_COLS")
_require_cols_in_splits(excluded, "IOT_EXCLUDED_COLS")
_require_cols_in_splits(telemetry_in_sec, "IOT_TELEMETRY_IN_SEC_COLS")
_require_cols_in_splits(countlike, "IOT_COUNTLIKE_COLS")
_require_cols_in_splits(telemetry, "IOT_TELEMETRY_COLS")
_require_cols_in_splits(step_progress, "IOT_STEP_PROGRESS_COLS")
_require_cols_in_splits(discrete_state, "IOT_DISCRETE_STATE_COLS")
_require_cols_in_splits(settinglike, "IOT_SETTINGLIKE_COLS")

# ----------------------------------------------------------
# Partition validation
# ----------------------------------------------------------
top_blocks = [
    ("driver", drv),
    ("meta", meta),
    ("binary", bin_),
    ("continuous", con),
    ("excluded", excluded),
]

for i in range(len(top_blocks)):
    a_name, a = top_blocks[i]
    for j in range(i + 1, len(top_blocks)):
        b_name, b = top_blocks[j]
        _assert_disjoint(a_name, a, b_name, b)

partition_union = set(drv) | set(meta) | set(bin_) | set(con) | set(excluded)
if partition_union != set(full_namespace):
    missing_from_partition = sorted(set(full_namespace) - partition_union)
    extra_in_partition = sorted(partition_union - set(full_namespace))
    raise RuntimeError(
        "[Cell6.5] Top-level IoT partition does not equal IOT_FULL_NAMESPACE_COLS. "
        f"missing={missing_from_partition[:20]} | extra={extra_in_partition[:20]}"
    )

state_expected_set = set(bin_) | set(con)
if set(state) != state_expected_set:
    missing_from_state = sorted(state_expected_set - set(state))
    extra_in_state = sorted(set(state) - state_expected_set)
    raise RuntimeError(
        "[Cell6.5] IOT_STATE_COLS is not exactly BIN ∪ CONT as a set. "
        f"missing_from_state={missing_from_state[:20]} | extra_in_state={extra_in_state[:20]}"
    )

for name, cols in [
    ("IOT_COUNTLIKE_COLS", countlike),
    ("IOT_TELEMETRY_COLS", telemetry),
    ("IOT_STEP_PROGRESS_COLS", step_progress),
    ("IOT_DISCRETE_STATE_COLS", discrete_state),
    ("IOT_SETTINGLIKE_COLS", settinglike),
]:
    bad = sorted(set(cols) - set(con))
    if bad:
        raise RuntimeError(
            f"[Cell6.5] {name} must be a subset of IOT_CONT_COLS. Bad={bad[:20]}"
        )

bad_telemetry_meta = sorted(set(telemetry_in_sec) - set(meta))
if bad_telemetry_meta:
    raise RuntimeError(
        f"[Cell6.5] telemetry_in_sec columns must be meta. Bad={bad_telemetry_meta[:20]}"
    )

bad_events_driver = sorted(
    [c for c in full_namespace if c.startswith("events_in_sec__") and c not in set(drv)]
)
if bad_events_driver:
    raise RuntimeError(
        f"[Cell6.5] events_in_sec columns must be drivers. Bad={bad_events_driver[:20]}"
    )

bad_driver_names = [c for c in drv if not c.startswith("events_in_sec__")]
if bad_driver_names:
    raise RuntimeError(
        f"[Cell6.5] IOT_DRIVER_COLS contains non-events_in_sec columns: {bad_driver_names[:20]}"
    )

for name, cols in [
    ("IOT_EXCLUDED_NON_NUMERIC_COLS", excluded_non_numeric),
    ("IOT_EXCLUDED_ALL_NAN_COLS", excluded_all_nan),
    ("IOT_EXCLUDED_LOW_SIGNAL_COLS", excluded_low_signal),
    ("IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS", excluded_state_value_no_dev),
    ("IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS", excluded_state_value_low_dev),
]:
    bad = sorted(set(cols) - set(excluded))
    if bad:
        raise RuntimeError(f"[Cell6.5] {name} must be a subset of IOT_EXCLUDED_COLS. Bad={bad[:20]}")

# ----------------------------------------------------------
# Count consistency
# ----------------------------------------------------------
_count_check(counts, "full_iot_namespace", full_namespace)
_count_check(counts, "drivers", drv)
_count_check(counts, "binary", bin_)
_count_check(counts, "continuous", con)
_count_check(counts, "state_total", state)
_count_check(counts, "meta", meta)
_count_check(counts, "telemetry_in_sec_meta", telemetry_in_sec)
_count_check(counts, "countlike", countlike)
_count_check(counts, "telemetry", telemetry)
_count_check(counts, "step_progress", step_progress)
_count_check(counts, "discrete_state", discrete_state)
_count_check(counts, "setting_like", settinglike)
_count_check(counts, "excluded_total", excluded)
_count_check(counts, "excluded_non_numeric", excluded_non_numeric)
_count_check(counts, "excluded_all_nan", excluded_all_nan)
_count_check(counts, "excluded_low_signal", excluded_low_signal)
_count_check(counts, "excluded_state_value_no_dev_support", excluded_state_value_no_dev)
_count_check(counts, "excluded_state_value_low_dev_support", excluded_state_value_low_dev)

# ----------------------------------------------------------
# Cross-check semantic contract file
# ----------------------------------------------------------
sem_required = [
    "version",
    "taxonomy_assignment_source",
    "test_usage",
    "driver_cols",
    "meta_cols",
    "binary_cols",
    "continuous_cols",
    "state_cols",
    "telemetry_in_sec_meta_cols",
    "excluded_cols",
    "excluded_state_value_no_dev_support_cols",
    "excluded_state_value_low_dev_support_cols",
]
sem_missing = [k for k in sem_required if k not in sem_obj]
if sem_missing:
    raise RuntimeError(f"[Cell6.5] Missing keys in iot_semantic_contract.json: {sem_missing}")

if str(sem_obj["taxonomy_assignment_source"]) != "TRAIN_PLUS_VAL_ONLY":
    raise RuntimeError("[Cell6.5] Semantic contract taxonomy source is not TRAIN_PLUS_VAL_ONLY.")
if str(sem_obj["test_usage"]) != "AUDIT_ONLY":
    raise RuntimeError("[Cell6.5] Semantic contract test_usage is not AUDIT_ONLY.")

_same_set("groups.driver", drv, "semantic.driver", sem_obj["driver_cols"])
_same_set("groups.meta", meta, "semantic.meta", sem_obj["meta_cols"])
_same_set("groups.binary", bin_, "semantic.binary", sem_obj["binary_cols"])
_same_set("groups.continuous", con, "semantic.continuous", sem_obj["continuous_cols"])
_same_set("groups.state", state, "semantic.state", sem_obj["state_cols"])
_same_set("groups.telemetry_in_sec", telemetry_in_sec, "semantic.telemetry_in_sec", sem_obj["telemetry_in_sec_meta_cols"])
_same_set("groups.excluded", excluded, "semantic.excluded", sem_obj["excluded_cols"])
_same_set(
    "groups.excluded_state_value_no_dev",
    excluded_state_value_no_dev,
    "semantic.excluded_state_value_no_dev",
    sem_obj["excluded_state_value_no_dev_support_cols"],
)
_same_set(
    "groups.excluded_state_value_low_dev",
    excluded_state_value_low_dev,
    "semantic.excluded_state_value_low_dev",
    sem_obj["excluded_state_value_low_dev_support_cols"],
)

_same_ordered_list("groups.binary", bin_, "semantic.binary", sem_obj["binary_cols"])
_same_ordered_list("groups.continuous", con, "semantic.continuous", sem_obj["continuous_cols"])
_same_set("groups.state", state, "semantic.state", sem_obj["state_cols"])

# ----------------------------------------------------------
# Validate v13 partition contract / overlap policy / routing manifest
# ----------------------------------------------------------
partition_counts = partition_contract.get("counts", {}) if isinstance(partition_contract, dict) else {}
if partition_counts:
    for key, expected in [
        ("full_iot_namespace", len(full_namespace)),
        ("continuous", len(con)),
        ("binary", len(bin_)),
        ("drivers", len(drv)),
        ("meta", len(meta)),
        ("excluded_total", len(excluded)),
    ]:
        if key in partition_counts and int(partition_counts[key]) != int(expected):
            raise RuntimeError(
                f"[Cell6.5] partition_contract count mismatch for {key}: "
                f"json={partition_counts[key]} actual={expected}"
            )

overlap_policy_counts = overlap_policy.get("counts", {}) if isinstance(overlap_policy, dict) else {}
routing_counts = cell_routing_manifest.get("counts", {}) if isinstance(cell_routing_manifest, dict) else {}

# Routing manifest is allowed to contain more detail, but it must not contradict
# the canonical v13 ownership counts.
for key, expected in [
    ("continuous", len(con)),
    ("binary", len(bin_)),
    ("drivers", len(drv)),
    ("meta", len(meta)),
    ("excluded", len(excluded)),
    ("full_iot_namespace", len(full_namespace)),
]:
    if key in routing_counts and int(routing_counts[key]) != int(expected):
        raise RuntimeError(
            f"[Cell6.5] cell_routing_manifest count mismatch for {key}: "
            f"json={routing_counts[key]} actual={expected}"
        )

# ----------------------------------------------------------
# Derived downstream target aliases
# ----------------------------------------------------------
IOT_CONTINUOUS_VALUE_TARGETS = list(con)
IOT_BINARY_TARGETS = list(bin_)
IOT_DRIVER_TARGETS = list(drv)

IOT_CONDITIONING_META_COLS = sorted([
    c for c, owner in owner_by_col.items()
    if owner == "conditioning_meta"
])
IOT_MASK_OBSERVABILITY_META_COLS = sorted([
    c for c, owner in owner_by_col.items()
    if owner == "mask_observability_meta"
])
IOT_STALENESS_META_COLS = sorted([
    c for c, owner in owner_by_col.items()
    if owner == "staleness_meta"
])
IOT_DERIVED_META_COLS = sorted([
    c for c, owner in owner_by_col.items()
    if owner == "derived_meta"
])

if set(IOT_CONDITIONING_META_COLS) != set(telemetry_in_sec):
    raise RuntimeError(
        "[Cell6.5] Conditioning meta columns must equal telemetry_in_sec helper columns."
    )

# ----------------------------------------------------------
# CPS-coupling policy for downstream IoT cells
# ----------------------------------------------------------
IOT_CPS_COUPLING_POLICY = {
    "version": "iot_cps_coupling_policy_v2_role_ownership",
    "declared_in_cell": "Cell6.5",
    "policy": {
        "iot_synthesis_must_be_cps_coupled": True,
        "standalone_iot_synthesis_allowed": False,
        "test_usage": "AUDIT_ONLY_OR_FINAL_QA_ONLY",
        "candidate_selection_split": "VAL_only",
        "required_shared_context": [
            "same_synthetic_timeline_length_as_protocol_A2",
            "same_SEC_COL_basis",
            "same_tod_arrays_or_synthetic_tod_equivalent",
            "same_regime_contract_or_synthetic_regime_equivalent",
            "protocol_logical_observation_masks",
            "selected_protocol_A2_activity_summaries",
            "events_in_sec_driver_context",
            "device_or_entity_availability_state",
            "cross_modal_lag_windows",
        ],
        "target_ownership_policy": {
            "continuous_value_target": "generate_as_numeric_IoT_value_targets",
            "binary_state_target": "generate_as_binary_IoT_state_targets",
            "driver_event_target": "generate_or_materialize_as_event_driver_streams",
            "conditioning_meta": "do_not_generate_as_targets_use_for_conditioning_or_derive",
            "mask_observability_meta": "derive_from_generated_availability_or_event_observability",
            "staleness_meta": "derive_from generated_observability_and_update_times",
            "derived_meta": "derive_or_carry_only_when_contract_safe",
            "excluded_no_dev_support": "do_not_generate_without_explicit_no_leakage_rule",
        },
        "forbidden_patterns": [
            "generate_IoT_values_independently_of_protocol_activity",
            "fit_IoT_generator_using_TEST_statistics",
            "infer_protocol_observability_from_protocol_values",
            "overwrite_protocol_masks_inside_IoT_cells",
            "create_IoT_timeline_length_different_from_A2_protocol_timeline",
            "treat_meta_or_excluded_columns_as direct synthetic targets",
            "collapse full IoT namespace to continuous-only scope",
        ],
        "required_downstream_audits": [
            "iot_event_to_protocol_response_delay_distribution",
            "protocol_burst_to_iot_event_or_state_cooccurrence",
            "per_device_availability_vs_protocol_observability",
            "regime_conditioned_iot_state_distribution",
            "tod_conditioned_iot_state_distribution",
            "sensor_actuator_transition_rates",
            "lagged_cross_correlation_iot_events_vs_protocol_activity",
            "binary_state_transition_and_dwell_realism",
            "driver_event_count_overdispersion_realism",
            "mask_and_staleness_derivation_consistency",
            "A1_A2_protocol_impact_on_iot_conditional_realism",
        ],
    },
    "upstream_protocol_contract": {
        "proto_cols_contract_n": int(len(proto_contract)),
        "proto_cols_contract_sha256": _sha_list(proto_contract),
        "protocol_tiers_used": list(globals()["PROTOCOL_TIERS_USED"]),
        "protocol_generator_role_contract_version": protocol_generator_role_contract.get("version"),
        "protocol_generator_role_contract_path": protocol_role_path if os.path.exists(protocol_role_path) else None,
        "protocol_synthesis_policy_path": protocol_policy_path if os.path.exists(protocol_policy_path) else None,
    },
    "upstream_iot_contract": {
        "groups_version": obj.get("version"),
        "semantic_version": sem_obj.get("version"),
        "full_iot_namespace_n": int(len(full_namespace)),
        "state_n": int(len(state)),
        "continuous_value_target_n": int(len(con)),
        "binary_state_target_n": int(len(bin_)),
        "driver_event_target_n": int(len(drv)),
        "meta_n": int(len(meta)),
        "conditioning_meta_n": int(len(IOT_CONDITIONING_META_COLS)),
        "mask_observability_meta_n": int(len(IOT_MASK_OBSERVABILITY_META_COLS)),
        "staleness_meta_n": int(len(IOT_STALENESS_META_COLS)),
        "derived_meta_n": int(len(IOT_DERIVED_META_COLS)),
        "excluded_n": int(len(excluded)),
        "primary_owner_counts": {str(k): int(v) for k, v in owner_counts.items()},
    },
}

coupling_policy_path = os.path.join(art_dir, "iot_cps_coupling_policy.json")
_write_json(coupling_policy_path, IOT_CPS_COUPLING_POLICY)

# ----------------------------------------------------------
# Export globals
# ----------------------------------------------------------
globals()["SEC_COL"] = SEC_COL

globals()["IOT_FULL_NAMESPACE_COLS"] = list(full_namespace)
globals()["IOT_DRIVER_COLS"] = list(drv)
globals()["IOT_BIN_COLS"] = list(bin_)
globals()["IOT_CONT_COLS"] = list(con)
globals()["IOT_STATE_COLS"] = list(state)
globals()["IOT_META_COLS"] = list(meta)

globals()["IOT_COUNTLIKE_COLS"] = list(countlike)
globals()["IOT_TELEMETRY_COLS"] = list(telemetry)
globals()["IOT_STEP_PROGRESS_COLS"] = list(step_progress)
globals()["IOT_DISCRETE_STATE_COLS"] = list(discrete_state)
globals()["IOT_SETTINGLIKE_COLS"] = list(settinglike)
globals()["IOT_TELEMETRY_IN_SEC_COLS"] = list(telemetry_in_sec)

globals()["IOT_EXCLUDED_COLS"] = list(excluded)
globals()["IOT_EXCLUDED_NON_NUMERIC_COLS"] = list(excluded_non_numeric)
globals()["IOT_EXCLUDED_ALL_NAN_COLS"] = list(excluded_all_nan)
globals()["IOT_EXCLUDED_LOW_SIGNAL_COLS"] = list(excluded_low_signal)
globals()["IOT_EXCLUDED_STATE_VALUE_NO_DEV_SUPPORT_COLS"] = list(excluded_state_value_no_dev)
globals()["IOT_EXCLUDED_STATE_VALUE_LOW_DEV_SUPPORT_COLS"] = list(excluded_state_value_low_dev)

globals()["IOT_CONTINUOUS_VALUE_TARGETS"] = list(IOT_CONTINUOUS_VALUE_TARGETS)
globals()["IOT_BINARY_TARGETS"] = list(IOT_BINARY_TARGETS)
globals()["IOT_DRIVER_TARGETS"] = list(IOT_DRIVER_TARGETS)
globals()["IOT_CONDITIONING_META_COLS"] = list(IOT_CONDITIONING_META_COLS)
globals()["IOT_MASK_OBSERVABILITY_META_COLS"] = list(IOT_MASK_OBSERVABILITY_META_COLS)
globals()["IOT_STALENESS_META_COLS"] = list(IOT_STALENESS_META_COLS)
globals()["IOT_DERIVED_META_COLS"] = list(IOT_DERIVED_META_COLS)

globals()["IOT_PRIMARY_OWNER_BY_COL"] = dict(owner_by_col)
globals()["IOT_ROLE_OWNERSHIP_DF"] = role_ownership_df.copy()
globals()["IOT_PARTITION_CONTRACT"] = partition_contract
globals()["IOT_OVERLAP_POLICY"] = overlap_policy
globals()["IOT_CELL_ROUTING_MANIFEST"] = cell_routing_manifest

# Backward-compatible aliases.
globals()["IOT_BINARY_COLS"] = list(bin_)
globals()["IOT_CONTINUOUS_COLS"] = list(con)
globals()["iot_drivers"] = list(drv)
globals()["iot_binary"] = list(bin_)
globals()["iot_continuous"] = list(con)

IOT_GROUP_LOADER_CONTRACT = {
    "version": "iot_group_loader_contract_v8_v13_role_ownership_cps_coupled",
    "source_groups_json": groups_path,
    "source_semantic_contract_json": semantic_path,
    "source_role_ownership_csv": role_ownership_path,
    "source_partition_contract_json": partition_contract_path,
    "source_overlap_policy_json": overlap_policy_path,
    "source_cell_routing_manifest_json": cell_routing_manifest_path,
    "groups_version": obj.get("version"),
    "semantic_version": sem_obj.get("version"),
    "taxonomy_assignment_source": obj.get("taxonomy_assignment_source"),
    "test_usage": obj.get("test_usage"),
    "sha_full_namespace": _sha_list(full_namespace),
    "sha_driver": _sha_list(drv),
    "sha_meta": _sha_list(meta),
    "sha_binary": _sha_list(bin_),
    "sha_continuous": _sha_list(con),
    "sha_state": _sha_list(state),
    "sha_excluded": _sha_list(excluded),
    "sha_primary_owner_items": hashlib.sha256(
        json.dumps(owner_by_col, sort_keys=True).encode("utf-8")
    ).hexdigest(),
    "counts": {
        "full_iot_namespace": len(full_namespace),
        "drivers": len(drv),
        "binary": len(bin_),
        "continuous": len(con),
        "state_total": len(state),
        "meta": len(meta),
        "excluded": len(excluded),
        "telemetry_in_sec_meta": len(telemetry_in_sec),
        "countlike": len(countlike),
        "telemetry": len(telemetry),
        "step_progress": len(step_progress),
        "discrete_state": len(discrete_state),
        "setting_like": len(settinglike),
        "conditioning_meta": len(IOT_CONDITIONING_META_COLS),
        "mask_observability_meta": len(IOT_MASK_OBSERVABILITY_META_COLS),
        "staleness_meta": len(IOT_STALENESS_META_COLS),
        "derived_meta": len(IOT_DERIVED_META_COLS),
        "excluded_state_value_no_dev_support": len(excluded_state_value_no_dev),
        "excluded_state_value_low_dev_support": len(excluded_state_value_low_dev),
    },
    "primary_owner_counts": {str(k): int(v) for k, v in owner_counts.items()},
    "cps_coupling_policy_json": coupling_policy_path,
    "protocol_context": {
        "proto_cols_contract_n": len(proto_contract),
        "proto_cols_contract_sha256": _sha_list(proto_contract),
        "protocol_tiers_used": list(globals()["PROTOCOL_TIERS_USED"]),
        "proto_obs_shapes": {
            "train": _shape_list(globals()["PROTO_OBS_TR"]),
            "val": _shape_list(globals()["PROTO_OBS_VA"]),
            "test": _shape_list(globals()["PROTO_OBS_TE"]),
        },
        "tod_shapes": {
            "train": _shape_list(globals()["tod_tr"]),
            "val": _shape_list(globals()["tod_va"]),
            "test": _shape_list(globals()["tod_te"]),
        },
        "regime_lengths": {
            "train": len(globals()["reg_tr"]),
            "val": len(globals()["reg_va"]),
            "test": len(globals()["reg_te"]),
        },
        "K_reg": int(globals()["K_reg"]),
    },
}

loader_contract_path = os.path.join(art_dir, "iot_group_loader_contract.json")
_write_json(loader_contract_path, IOT_GROUP_LOADER_CONTRACT)

globals()["IOT_GROUP_LOADER_CONTRACT"] = IOT_GROUP_LOADER_CONTRACT
globals()["IOT_GROUP_LOADER_CONTRACT_JSON"] = loader_contract_path
globals()["IOT_CPS_COUPLING_POLICY"] = IOT_CPS_COUPLING_POLICY
globals()["IOT_CPS_COUPLING_POLICY_JSON"] = coupling_policy_path

log(
    f"[Cell6.5] Canonical globals set: "
    f"full_namespace={len(full_namespace)} | drivers={len(drv)} | "
    f"binary={len(bin_)} | continuous={len(con)} | state={len(state)} | "
    f"meta={len(meta)} | excluded={len(excluded)}"
)
log(
    f"[Cell6.5] Primary owner counts: "
    f"{dict(sorted({str(k): int(v) for k, v in owner_counts.items()}.items()))}"
)
log(
    f"[Cell6.5] Subtypes: countlike={len(countlike)} | telemetry={len(telemetry)} | "
    f"step_progress={len(step_progress)} | discrete_state={len(discrete_state)} | "
    f"setting_like={len(settinglike)} | telemetry_in_sec_meta={len(telemetry_in_sec)}"
)
log(
    f"[Cell6.5] Derived target aliases: "
    f"IOT_CONTINUOUS_VALUE_TARGETS={len(IOT_CONTINUOUS_VALUE_TARGETS)} | "
    f"IOT_BINARY_TARGETS={len(IOT_BINARY_TARGETS)} | "
    f"IOT_DRIVER_TARGETS={len(IOT_DRIVER_TARGETS)} | "
    f"IOT_CONDITIONING_META_COLS={len(IOT_CONDITIONING_META_COLS)} | "
    f"IOT_MASK_OBSERVABILITY_META_COLS={len(IOT_MASK_OBSERVABILITY_META_COLS)} | "
    f"IOT_STALENESS_META_COLS={len(IOT_STALENESS_META_COLS)} | "
    f"IOT_DERIVED_META_COLS={len(IOT_DERIVED_META_COLS)}"
)
log(
    f"[Cell6.5] Exclusions: no_dev_state_value={len(excluded_state_value_no_dev)} | "
    f"low_dev_state_value={len(excluded_state_value_low_dev)}"
)
log(
    "[Cell6.5] CPS coupling policy: IoT downstream synthesis must share protocol "
    "timeline/regime/TOD/masks/activity context; standalone IoT synthesis is forbidden."
)
log(f"[Cell6.5] Saved loader contract: {loader_contract_path}")
log(f"[Cell6.5] Saved CPS coupling policy: {coupling_policy_path}")
log("--- END: Cell 6.5 ---")