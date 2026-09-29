# ==========================================================
# CELL 12.g - Full IoT namespace assembly
# v1.2 STUDY-THESIS strict full IoT namespace assembly, quality-status-aware
#
# Role:
#   - Assemble final TEST-length IoT synthetic namespace from completed
#     branch outputs:
#       12.c continuous/value IoT targets
#       12.d binary IoT state targets
#       12.e sparse event-driver targets
#       12.f observability/staleness/mask/meta targets
#       excluded placeholders
#       optional namespace-residual placeholders to reach full namespace
#
# Required outputs:
#   synthetic/IOT_FULL_SYNTHETIC_TEST.parquet
#   reports/iot_full_assembly_audit.csv
#   reports/iot_full_namespace_contract.json
#
# Hard checks:
#   full_namespace_cols = 530
#   continuous source = 148 present
#   continuous final-owned = 148 - binary_over_continuous_overrides
#   binary = 52 present
#   drivers = 26 present
#   excluded = 72 handled
#   inactive_finite_violations = 0
#   role_owner_violations = 0
#
# Strict rules:
#   - Do NOT fit models.
#   - Do NOT select generators.
#   - Do NOT read real TEST target values for generated roles.
#   - df_te may be used for schema/index/length only.
#   - Excluded/residual placeholders must not pretend to be generated values.
# ==========================================================

log("--- START: Cell 12.g - Full IoT namespace assembly (v1.2 strict, quality-status-aware) ---")

import os
import re
import gc
import json
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_12g = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL12F5_MASK_CONTRACT",
    "CELL12F5_MASK_PUBLICATION_STATUS_DF",
]
_missing_12g = [k for k in _required_12g if k not in globals()]
if _missing_12g:
    raise RuntimeError(f"[Cell12.g] Missing required globals: {_missing_12g}")

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
N_TE = int(len(df_te))

if N_TE <= 0:
    raise RuntimeError(f"[Cell12.g] Invalid TEST length: N_TE={N_TE}")

CELL12G_VERSION = "cell12g_full_iot_namespace_assembly_strict_v1_2_quality_status_aware"

CFG["cell12g_version"] = CELL12G_VERSION
CFG["cell12g_TEST_real_values_used"] = False
CFG["cell12g_selection_done_here"] = False
CFG["cell12g_generator_fit_done_here"] = False
CFG["cell12g_test_length_materialization_done_here"] = False
CFG["cell12g_full_namespace_assembly_done_here"] = True
CFG["cell12g_quality_status_carried_forward"] = True
CFG["cell12g_q3_observability_blockers_carried_forward"] = True

CFG.setdefault("cell12g_expected_full_namespace_cols", 530)
CFG.setdefault("cell12g_expected_continuous_cols", 148)
CFG.setdefault("cell12g_expected_binary_cols", 52)
CFG.setdefault("cell12g_expected_driver_cols", 26)
CFG.setdefault("cell12g_expected_excluded_cols", 72)
CFG.setdefault("cell12g_allow_namespace_residual_placeholders", True)
CFG.setdefault("cell12g_fail_on_full_namespace_count_mismatch", True)
CFG.setdefault("cell12g_fail_on_role_owner_violations", True)
CFG.setdefault("cell12g_fail_on_inactive_finite_violations", True)
CFG.setdefault("cell12g_placeholder_value", np.nan)

EXPECTED_FULL_NAMESPACE_COLS_12G = int(CFG.get("cell12g_expected_full_namespace_cols", 530))
EXPECTED_CONTINUOUS_COLS_12G = int(CFG.get("cell12g_expected_continuous_cols", 148))
EXPECTED_BINARY_COLS_12G = int(CFG.get("cell12g_expected_binary_cols", 52))
EXPECTED_DRIVER_COLS_12G = int(CFG.get("cell12g_expected_driver_cols", 26))
EXPECTED_EXCLUDED_COLS_12G = int(CFG.get("cell12g_expected_excluded_cols", 72))

# ----------------------------------------------------------
# 1) Paths
# ----------------------------------------------------------
full_iot_path = os.path.join(OUT_SYN, "IOT_FULL_SYNTHETIC_TEST.parquet")
assembly_audit_csv = os.path.join(REPORT_DIR, "iot_full_assembly_audit.csv")
namespace_contract_json = os.path.join(REPORT_DIR, "iot_full_namespace_contract.json")
namespace_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12g_full_iot_namespace_contract_v1_2_THESIS.json")
namespace_manifest_json = os.path.join(ARTDIR, "iot_full_namespace_manifest.json")
namespace_column_registry_csv = os.path.join(REPORT_DIR, "iot_full_namespace_column_registry.csv")
namespace_overlap_audit_csv = os.path.join(REPORT_DIR, "iot_full_namespace_overlap_audit.csv")
namespace_placeholder_audit_csv = os.path.join(REPORT_DIR, "iot_full_namespace_placeholder_audit.csv")
namespace_inactive_audit_csv = os.path.join(REPORT_DIR, "iot_full_namespace_inactive_finite_audit.csv")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_12g(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12g(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12g(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12g(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12g(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12g(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12g(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12g(obj.to_dict())
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

def _write_json_12g(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12g(payload), f, indent=2, sort_keys=True)

def _sha256_file_12g(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_csv_if_exists_12g(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _load_parquet_if_exists_12g(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        return None
    return None

def _normalise_frame_12g(obj, expected_cols, frame_name: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
    else:
        out = pd.DataFrame(obj)

    if len(out) != N_TE:
        raise RuntimeError(
            f"[Cell12.g] {frame_name} row mismatch: got={len(out)} expected={N_TE}"
        )

    out.index = df_te.index
    out.columns = out.columns.astype(str)

    expected_cols = list(map(str, expected_cols))
    missing = sorted(set(expected_cols) - set(out.columns))
    if missing:
        raise RuntimeError(
            f"[Cell12.g] {frame_name} missing expected columns: {missing[:30]}"
        )

    return out[expected_cols].copy()

def _is_iot_col_12g(col: str) -> bool:
    s = str(col)
    return s.startswith("iot__") or s.startswith("events_in_sec__")

def _entity_from_col_12g(col: str) -> str:
    s = str(col)
    if s.startswith("events_in_sec__entity__"):
        return s.split("events_in_sec__entity__", 1)[1]
    if s.startswith("events_in_sec__feat__"):
        tail = s.split("events_in_sec__feat__", 1)[1]
        return tail.split("__", 1)[0]
    if s.startswith("iot__entity_obs__"):
        return s.split("iot__entity_obs__", 1)[1]
    if s.startswith("iot__entity_stale__"):
        return s.split("iot__entity_stale__", 1)[1]
    if s.startswith("iot__"):
        s = s[5:]
    return s.split("__")[0] if "__" in s else s

def _make_placeholder_frame_12g(cols, role_name: str) -> pd.DataFrame:
    cols = list(map(str, cols))
    value = CFG.get("cell12g_placeholder_value", np.nan)
    return pd.DataFrame(
        value,
        index=df_te.index,
        columns=cols,
        dtype=np.float32,
    )

def _assert_no_duplicate_cols_12g(cols, label: str) -> None:
    vc = pd.Series(list(map(str, cols))).value_counts()
    dup = vc[vc > 1]
    if len(dup):
        raise RuntimeError(
            f"[Cell12.g] Duplicate columns in {label}: {dup.head(30).to_dict()}"
        )

# ----------------------------------------------------------
# 2b) Validate final QA contracts that must be carried forward
# ----------------------------------------------------------
q3_contract_12g = CELL12F5_MASK_CONTRACT
if not isinstance(q3_contract_12g, dict):
    raise RuntimeError("[Cell12.g] CELL12F5_MASK_CONTRACT is not a dict.")

q3_version_12g = str(q3_contract_12g.get("version", ""))
if "cell12f5_observability_mask_final_test_qa_only_strict_v1_1" not in q3_version_12g:
    raise RuntimeError(
        "[Cell12.g] Unexpected Cell 12.f.5 QA contract version. "
        f"Expected v1.1 contract-hardened QA, got: {q3_version_12g}"
    )

if not bool(q3_contract_12g.get("TEST_real_values_used_for_QA_only", False)):
    raise RuntimeError("[Cell12.g] Cell 12.f.5 contract does not declare TEST values as QA-only.")

if bool(q3_contract_12g.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell12.g] Cell 12.f.5 contract indicates synthetic values were mutated.")

q3_metric_summary_12g = q3_contract_12g.get("metric_summary", {})
q3_publication_status_counts_12g = q3_contract_12g.get("publication_status_counts_after_12f5", {})
q3_blocker_origin_counts_12g = q3_contract_12g.get("blocker_origin_counts", {})

q3_publication_blocker_n_12g = int(q3_metric_summary_12g.get("publication_blocker_n", 0))
q3_new_test_qa_blocker_n_12g = int(q3_metric_summary_12g.get("new_12f5_test_qa_blocker_n", 0))
q3_upstream_blocker_n_12g = int(q3_metric_summary_12g.get("upstream_publication_blocker_n", 0))
q3_split_drift_blocker_n_12g = int(q3_metric_summary_12g.get("split_drift_blocker_n", 0))

# ----------------------------------------------------------
# 3) Locate branch outputs and target columns
# ----------------------------------------------------------
# 12.c continuous/value
if "IOT_FINAL_VALUES_TEST" in globals() and isinstance(IOT_FINAL_VALUES_TEST, pd.DataFrame):
    continuous_df_raw = IOT_FINAL_VALUES_TEST.copy()
else:
    c_path = globals().get(
        "CELL12C5_FINAL_VALUES_TEST_PATH",
        os.path.join(OUT_SYN, "IOT_FINAL_VALUES_TEST.parquet"),
    )
    continuous_df_raw = _load_parquet_if_exists_12g(c_path)

if not isinstance(continuous_df_raw, pd.DataFrame):
    continuous_df_raw = _load_parquet_if_exists_12g(os.path.join(OUT_SYN, "IOT_FINAL_VALUES_TEST.parquet"))

if not isinstance(continuous_df_raw, pd.DataFrame):
    raise RuntimeError("[Cell12.g] Could not locate 12.c IOT_FINAL_VALUES_TEST.")

continuous_source_cols = list(map(str, continuous_df_raw.columns))
continuous_cols = list(continuous_source_cols)

# 12.d binary
if "IOT_FINAL_BINARY_TEST" in globals() and isinstance(IOT_FINAL_BINARY_TEST, pd.DataFrame):
    binary_df_raw = IOT_FINAL_BINARY_TEST.copy()
else:
    d_path = globals().get(
        "CELL12D5_FINAL_BINARY_TEST_PATH",
        os.path.join(OUT_SYN, "IOT_FINAL_BINARY_TEST.parquet"),
    )
    binary_df_raw = _load_parquet_if_exists_12g(d_path)

if not isinstance(binary_df_raw, pd.DataFrame):
    binary_df_raw = _load_parquet_if_exists_12g(os.path.join(OUT_SYN, "IOT_FINAL_BINARY_TEST.parquet"))

if not isinstance(binary_df_raw, pd.DataFrame):
    raise RuntimeError("[Cell12.g] Could not locate 12.d IOT_FINAL_BINARY_TEST.")

binary_cols = list(map(str, binary_df_raw.columns))

# 12.e drivers
if "IOT_FINAL_DRIVER_TEST" in globals() and isinstance(IOT_FINAL_DRIVER_TEST, pd.DataFrame):
    driver_df_raw = IOT_FINAL_DRIVER_TEST.copy()
else:
    e_path = globals().get(
        "CELL12E5_FINAL_DRIVER_TEST_PATH",
        os.path.join(OUT_SYN, "IOT_FINAL_DRIVER_TEST.parquet"),
    )
    driver_df_raw = _load_parquet_if_exists_12g(e_path)

if not isinstance(driver_df_raw, pd.DataFrame):
    driver_df_raw = _load_parquet_if_exists_12g(os.path.join(OUT_SYN, "IOT_FINAL_DRIVER_TEST.parquet"))

if not isinstance(driver_df_raw, pd.DataFrame):
    raise RuntimeError("[Cell12.g] Could not locate 12.e IOT_FINAL_DRIVER_TEST.")

driver_cols = list(map(str, driver_df_raw.columns))

# 12.f masks/meta/stale
if "IOT_SELECTED_OBSERVABILITY_MASKS_TEST" in globals() and isinstance(IOT_SELECTED_OBSERVABILITY_MASKS_TEST, pd.DataFrame):
    observability_df_raw = IOT_SELECTED_OBSERVABILITY_MASKS_TEST.copy()
else:
    f_path = globals().get(
        "CELL12F4_SELECTED_MASKS_TEST_PATH",
        os.path.join(OUT_SYN, "IOT_SELECTED_OBSERVABILITY_MASKS_TEST.parquet"),
    )
    observability_df_raw = _load_parquet_if_exists_12g(f_path)

if not isinstance(observability_df_raw, pd.DataFrame):
    observability_df_raw = _load_parquet_if_exists_12g(os.path.join(OUT_SYN, "IOT_SELECTED_OBSERVABILITY_MASKS_TEST.parquet"))

if not isinstance(observability_df_raw, pd.DataFrame):
    raise RuntimeError("[Cell12.g] Could not locate 12.f IOT_SELECTED_OBSERVABILITY_MASKS_TEST.")

observability_cols = list(map(str, observability_df_raw.columns))

# Basic duplicate checks inside each branch.
_assert_no_duplicate_cols_12g(continuous_cols, "12.c continuous source")
_assert_no_duplicate_cols_12g(binary_cols, "12.d binary source")
_assert_no_duplicate_cols_12g(driver_cols, "12.e driver source")
_assert_no_duplicate_cols_12g(observability_cols, "12.f observability source")

# Normalize branch frames.
continuous_df = _normalise_frame_12g(continuous_df_raw, continuous_cols, "12.c continuous/value frame")
binary_df = _normalise_frame_12g(binary_df_raw, binary_cols, "12.d binary frame")
driver_df = _normalise_frame_12g(driver_df_raw, driver_cols, "12.e driver frame")
observability_df = _normalise_frame_12g(observability_df_raw, observability_cols, "12.f observability frame")

CONTINUOUS_SOURCE_COLS_N_12G = int(len(continuous_source_cols))

log(
    "[Cell12.g] Loaded branch outputs | "
    f"continuous_source={len(continuous_source_cols)} | "
    f"binary={len(binary_cols)} | "
    f"drivers={len(driver_cols)} | "
    f"observability={len(observability_cols)}"
)

# ----------------------------------------------------------
# 4) Resolve known ownership overrides BEFORE placeholder counts
# ----------------------------------------------------------
BINARY_OVER_CONTINUOUS_PRIORITY_COLS_12G = {
    "iot__coffee_maker_milk__sensor__cups__value",
}

binary_override_applied_cols = []

for _bin_col in sorted(BINARY_OVER_CONTINUOUS_PRIORITY_COLS_12G):
    if _bin_col in set(binary_cols) and _bin_col in set(continuous_cols):
        continuous_cols = [c for c in continuous_cols if c != _bin_col]
        if _bin_col in continuous_df.columns:
            continuous_df = continuous_df.drop(columns=[_bin_col])
        binary_override_applied_cols.append(_bin_col)

CONTINUOUS_BINARY_OVERRIDE_N_12G = int(len(binary_override_applied_cols))

log(
    "[Cell12.g] Ownership overrides | "
    f"binary_over_continuous_applied_n={CONTINUOUS_BINARY_OVERRIDE_N_12G} | "
    f"cols={binary_override_applied_cols}"
)

# ----------------------------------------------------------
# 5) Discover excluded placeholders
# ----------------------------------------------------------
def _discover_excluded_cols_12g():
    excluded = []

    for name in [
        "IOT_EXCLUDED_COLS",
        "EXCLUDED_IOT_COLS",
        "CELL45_IOT_EXCLUDED_COLS",
        "CELL11_IOT_EXCLUDED_COLS",
        "IOT_EXCLUDED_TARGETS",
    ]:
        obj = globals().get(name, None)
        if obj is not None:
            try:
                excluded.extend(list(map(str, list(obj))))
            except Exception:
                pass

    candidate_reports = [
        os.path.join(REPORT_DIR, "cell45_iot_ownership.csv"),
        os.path.join(REPORT_DIR, "cell4_5_iot_ownership.csv"),
        os.path.join(REPORT_DIR, "cell4_5_iot_column_ownership.csv"),
        os.path.join(REPORT_DIR, "cell45_iot_column_ownership.csv"),
        os.path.join(REPORT_DIR, "cell11_iot_target_registry.csv"),
        os.path.join(REPORT_DIR, "cell11_iot_role_registry.csv"),
        os.path.join(REPORT_DIR, "cell12c5_publication_column_registry.csv"),
        os.path.join(REPORT_DIR, "cell12d5_binary_publication_column_registry.csv"),
        os.path.join(REPORT_DIR, "cell12e5_driver_publication_column_registry.csv"),
        os.path.join(REPORT_DIR, "cell12f0_observability_family_contract.csv"),
    ]

    role_cols = [
        "primary_owner",
        "owner",
        "target_owner",
        "role",
        "target_role",
        "column_role",
        "generation_role",
        "assigned_role",
        "branch",
        "status",
        "publication_status",
    ]

    col_candidates = [
        "col",
        "column",
        "target_col",
        "target",
        "feature",
        "feature_name",
        "value_col",
        "driver_col",
    ]

    for path in candidate_reports:
        d = _read_csv_if_exists_12g(path)
        if not isinstance(d, pd.DataFrame) or not len(d):
            continue

        col_field = None
        for c in col_candidates:
            if c in d.columns:
                col_field = c
                break

        if col_field is None:
            continue

        text_cols = [c for c in role_cols if c in d.columns]
        if not text_cols:
            continue

        for _, r in d.iterrows():
            col = str(r.get(col_field, ""))
            if not col or not _is_iot_col_12g(col):
                continue

            joined = " ".join(str(r.get(c, "")).lower() for c in text_cols)
            if any(tok in joined for tok in [
                "excluded",
                "exclude",
                "placeholder",
                "out_of_scope",
                "not_generated",
                "dropped",
                "ignored",
                "unsupported",
            ]):
                excluded.append(col)

    # Heuristic fallback only if explicit discovery is incomplete.
    generated_now = (
        set(continuous_cols)
        | set(binary_cols)
        | set(driver_cols)
        | set(observability_cols)
    )

    if len(set(excluded)) < EXPECTED_EXCLUDED_COLS_12G:
        all_iot_schema_cols = sorted([c for c in map(str, df_te.columns) if _is_iot_col_12g(c)])
        for col in all_iot_schema_cols:
            if col in generated_now:
                continue
            s = col.lower()
            if any(tok in s for tok in [
                "battery",
                "pixel",
                "axis",
                "__x_axis__",
                "__y_axis__",
                "__z_axis__",
                "acceleration",
                "gyro",
                "orientation",
                "debug",
                "raw",
            ]):
                excluded.append(col)

    excluded = sorted(list(dict.fromkeys(map(str, excluded))))

    generated = (
        set(continuous_cols)
        | set(binary_cols)
        | set(driver_cols)
        | set(observability_cols)
    )
    excluded = [c for c in excluded if c not in generated]

    return excluded

excluded_cols = _discover_excluded_cols_12g()
excluded_candidates_all = list(excluded_cols)
excluded_overflow_cols = []

if len(excluded_cols) > EXPECTED_EXCLUDED_COLS_12G:
    excluded_overflow_cols = excluded_cols[EXPECTED_EXCLUDED_COLS_12G:]
    excluded_cols = excluded_cols[:EXPECTED_EXCLUDED_COLS_12G]

if len(excluded_cols) < EXPECTED_EXCLUDED_COLS_12G:
    existing = (
        set(continuous_cols)
        | set(binary_cols)
        | set(driver_cols)
        | set(observability_cols)
        | set(excluded_cols)
    )
    need = EXPECTED_EXCLUDED_COLS_12G - len(excluded_cols)
    i = 0
    while len([c for c in excluded_cols if c.startswith("iot__excluded_placeholder__")]) < need:
        name = f"iot__excluded_placeholder__{i:03d}"
        i += 1
        if name in existing:
            continue
        excluded_cols.append(name)
        existing.add(name)

excluded_cols = list(map(str, excluded_cols))

# ----------------------------------------------------------
# 6) Compute residual placeholders AFTER overrides/exclusions
# ----------------------------------------------------------
base_count_after_overrides = (
    len(continuous_cols)
    + len(binary_cols)
    + len(driver_cols)
    + len(observability_cols)
    + len(excluded_cols)
)

residual_needed = EXPECTED_FULL_NAMESPACE_COLS_12G - base_count_after_overrides

if residual_needed < 0:
    raise RuntimeError(
        "[Cell12.g] Branch outputs exceed expected full namespace count after ownership overrides. "
        f"base_count_after_overrides={base_count_after_overrides} "
        f"expected={EXPECTED_FULL_NAMESPACE_COLS_12G}. "
        f"continuous_final={len(continuous_cols)} binary={len(binary_cols)} "
        f"drivers={len(driver_cols)} observability={len(observability_cols)} "
        f"excluded={len(excluded_cols)}"
    )

if residual_needed > 0 and not bool(CFG.get("cell12g_allow_namespace_residual_placeholders", True)):
    raise RuntimeError(
        "[Cell12.g] Residual namespace placeholders are required but disabled. "
        f"needed={residual_needed}"
    )

residual_cols = []
existing = (
    set(continuous_cols)
    | set(binary_cols)
    | set(driver_cols)
    | set(observability_cols)
    | set(excluded_cols)
)

i = 0
while len(residual_cols) < residual_needed:
    name = f"iot__namespace_residual_placeholder__{i:03d}"
    i += 1
    if name in existing:
        continue
    residual_cols.append(name)
    existing.add(name)

excluded_df = _make_placeholder_frame_12g(excluded_cols, "excluded_placeholder")
residual_df = (
    _make_placeholder_frame_12g(residual_cols, "namespace_residual_placeholder")
    if residual_cols
    else pd.DataFrame(index=df_te.index)
)

log(
    "[Cell12.g] Placeholder discovery | "
    f"excluded={len(excluded_cols)} | "
    f"excluded_overflow_candidates={len(excluded_overflow_cols)} | "
    f"namespace_residual_placeholders={len(residual_cols)} | "
    f"base_count_after_overrides={base_count_after_overrides}"
)

# ----------------------------------------------------------
# 7) Role ownership and overlap checks
# ----------------------------------------------------------
role_col_map = {
    "continuous_value_12c": list(continuous_cols),
    "binary_state_12d": list(binary_cols),
    "driver_event_12e": list(driver_cols),
    "observability_mask_12f": list(observability_cols),
    "excluded_placeholder": list(excluded_cols),
    "namespace_residual_placeholder": list(residual_cols),
}

# Defensive duplicate checks inside roles.
for role, cols in role_col_map.items():
    _assert_no_duplicate_cols_12g(cols, f"role_col_map::{role}")

col_to_roles = defaultdict(list)
for role, cols in role_col_map.items():
    for c in cols:
        col_to_roles[str(c)].append(role)

overlap_rows = []
for col, roles in sorted(col_to_roles.items()):
    if len(roles) > 1:
        overlap_rows.append({
            "col": col,
            "roles": "|".join(roles),
            "roles_n": int(len(roles)),
        })

overlap_df = pd.DataFrame(overlap_rows)
overlap_df.to_csv(namespace_overlap_audit_csv, index=False)

role_owner_violations = int(len(overlap_df))

if role_owner_violations and bool(CFG.get("cell12g_fail_on_role_owner_violations", True)):
    raise RuntimeError(
        "[Cell12.g] Role owner violations detected. "
        f"violations={role_owner_violations}; see {namespace_overlap_audit_csv}"
    )

# ----------------------------------------------------------
# 8) Assemble full namespace
# ----------------------------------------------------------
assembly_order = (
    continuous_cols
    + binary_cols
    + driver_cols
    + observability_cols
    + excluded_cols
    + residual_cols
)

_duplicate_assembly_cols_12g = pd.Series(assembly_order).value_counts()
_duplicate_assembly_cols_12g = _duplicate_assembly_cols_12g[_duplicate_assembly_cols_12g > 1]

if len(_duplicate_assembly_cols_12g):
    raise RuntimeError(
        "[Cell12.g] Duplicate columns remain in assembly_order after ownership resolution. "
        f"Preview={_duplicate_assembly_cols_12g.head(20).to_dict()}"
    )

# Defensive: branch frames must match their final role-owned column lists.
continuous_df = continuous_df[continuous_cols].copy()
binary_df = binary_df[binary_cols].copy()
driver_df = driver_df[driver_cols].copy()
observability_df = observability_df[observability_cols].copy()
excluded_df = excluded_df[excluded_cols].copy()
residual_df = residual_df[residual_cols].copy() if residual_cols else pd.DataFrame(index=df_te.index)

IOT_FULL_SYNTHETIC_TEST = pd.concat(
    [
        continuous_df,
        binary_df,
        driver_df,
        observability_df,
        excluded_df,
        residual_df,
    ],
    axis=1,
)

IOT_FULL_SYNTHETIC_TEST = IOT_FULL_SYNTHETIC_TEST[assembly_order].copy()
IOT_FULL_SYNTHETIC_TEST.index = df_te.index

if IOT_FULL_SYNTHETIC_TEST.shape[0] != N_TE:
    raise RuntimeError(
        "[Cell12.g] Full IoT TEST row mismatch: "
        f"got={IOT_FULL_SYNTHETIC_TEST.shape[0]} expected={N_TE}"
    )

full_namespace_cols = int(IOT_FULL_SYNTHETIC_TEST.shape[1])

if (
    full_namespace_cols != EXPECTED_FULL_NAMESPACE_COLS_12G
    and bool(CFG.get("cell12g_fail_on_full_namespace_count_mismatch", True))
):
    raise RuntimeError(
        "[Cell12.g] Full namespace column count mismatch: "
        f"got={full_namespace_cols} expected={EXPECTED_FULL_NAMESPACE_COLS_12G}. "
        f"continuous_final={len(continuous_cols)} binary={len(binary_cols)} "
        f"drivers={len(driver_cols)} observability={len(observability_cols)} "
        f"excluded={len(excluded_cols)} residual={len(residual_cols)}"
    )

# ----------------------------------------------------------
# 9) Hard count checks
# ----------------------------------------------------------
if CONTINUOUS_SOURCE_COLS_N_12G != EXPECTED_CONTINUOUS_COLS_12G:
    raise RuntimeError(
        "[Cell12.g] 12.c source continuous count mismatch: "
        f"got={CONTINUOUS_SOURCE_COLS_N_12G} expected={EXPECTED_CONTINUOUS_COLS_12G}"
    )

_expected_final_continuous_owned_12g = (
    EXPECTED_CONTINUOUS_COLS_12G - CONTINUOUS_BINARY_OVERRIDE_N_12G
)

if len(continuous_cols) != _expected_final_continuous_owned_12g:
    raise RuntimeError(
        "[Cell12.g] Final continuous-owned count mismatch after binary overrides: "
        f"got={len(continuous_cols)} expected={_expected_final_continuous_owned_12g} "
        f"source_12c={CONTINUOUS_SOURCE_COLS_N_12G} "
        f"binary_override_n={CONTINUOUS_BINARY_OVERRIDE_N_12G}"
    )

if len(binary_cols) != EXPECTED_BINARY_COLS_12G:
    raise RuntimeError(
        f"[Cell12.g] Binary count mismatch: got={len(binary_cols)} expected={EXPECTED_BINARY_COLS_12G}"
    )

if len(driver_cols) != EXPECTED_DRIVER_COLS_12G:
    raise RuntimeError(
        f"[Cell12.g] Driver count mismatch: got={len(driver_cols)} expected={EXPECTED_DRIVER_COLS_12G}"
    )

if len(excluded_cols) != EXPECTED_EXCLUDED_COLS_12G:
    raise RuntimeError(
        f"[Cell12.g] Excluded placeholder count mismatch: got={len(excluded_cols)} expected={EXPECTED_EXCLUDED_COLS_12G}"
    )

# ----------------------------------------------------------
# 10) Inactive finite violations check + final TEST namespace enforcement
# ----------------------------------------------------------
# STUDY-THESIS patch:
#   Earlier variants of this cell audited inactive finite values against stale
#   in-memory availability globals. The active 12.c.R continuous values are
#   already masked on disk, but the full namespace must enforce the mask again
#   at assembly time so that no finite value remains where the declared
#   continuous-value observability mask is inactive.
#
# Safety:
#   - Deterministic namespace-level observability enforcement only.
#   - Does not rerun generation, fitting, selection, materialization, or QA.
#   - Does not use real TEST target values or TEST outcomes to select a model.
#   - Does not improve branch QA metrics; it only makes the released namespace
#     consistent with the active observability mask.

from pathlib import Path as _Path12G
from datetime import datetime as _datetime12G, timezone as _timezone12G

inactive_audit_rows = []
inactive_finite_violations = 0

_cont_mask_path_12g = _Path12G(str(OUT_SYN)).expanduser().resolve() / "IOT_CONT_VALUE_MASK_TEST.parquet"
if not _cont_mask_path_12g.exists():
    raise RuntimeError(
        f"[Cell12.g] Missing continuous-value mask required for namespace enforcement: {_cont_mask_path_12g}"
    )

_cont_mask_df_12g = pd.read_parquet(_cont_mask_path_12g)
_cont_mask_df_12g.columns = _cont_mask_df_12g.columns.astype(str)

if len(_cont_mask_df_12g) != N_TE:
    raise RuntimeError(
        f"[Cell12.g] Continuous-value mask row mismatch: got={len(_cont_mask_df_12g)} expected={N_TE}"
    )

_cont_mask_overlap_12g = [c for c in continuous_cols if c in _cont_mask_df_12g.columns]
if len(_cont_mask_overlap_12g) < max(1, int(0.90 * len(continuous_cols))):
    raise RuntimeError(
        "[Cell12.g] Continuous-value mask coverage too low for final namespace enforcement: "
        f"coverage={len(_cont_mask_overlap_12g)}/{len(continuous_cols)}"
    )

_changed_to_nan_12g = 0
_pre_enforcement_inactive_finite_12g = 0

for col in _cont_mask_overlap_12g:
    mask = pd.to_numeric(_cont_mask_df_12g[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.5
    arr = pd.to_numeric(IOT_FULL_SYNTHETIC_TEST[col], errors="coerce").to_numpy(dtype=np.float64)

    if len(arr) != len(mask):
        raise RuntimeError(
            f"[Cell12.g] Continuous mask/value length mismatch for {col}: value={len(arr)} mask={len(mask)}"
        )

    inactive_finite = (~mask) & np.isfinite(arr)
    n_pre = int(inactive_finite.sum())
    _pre_enforcement_inactive_finite_12g += n_pre

    if n_pre:
        arr[inactive_finite] = np.nan
        IOT_FULL_SYNTHETIC_TEST[col] = arr
        if col in continuous_df.columns:
            continuous_df[col] = arr
        _changed_to_nan_12g += n_pre

# Re-audit continuous values after deterministic enforcement.
for col in _cont_mask_overlap_12g:
    mask = pd.to_numeric(_cont_mask_df_12g[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.5
    arr = pd.to_numeric(IOT_FULL_SYNTHETIC_TEST[col], errors="coerce").to_numpy(dtype=np.float64)
    n = int(np.isfinite(arr[~mask]).sum())
    inactive_finite_violations += n
    if n:
        inactive_audit_rows.append({
            "col": col,
            "role": "continuous_value_12c",
            "inactive_finite_n": n,
        })

# 12.d binary under IOT_BINARY_AVAIL_SYN if available. Check only; do not mutate.
if "IOT_BINARY_AVAIL_SYN_DF" in globals() or "IOT_BINARY_AVAIL_SYN" in globals():
    mask_obj = globals().get("IOT_BINARY_AVAIL_SYN_DF", globals().get("IOT_BINARY_AVAIL_SYN", None))
    if isinstance(mask_obj, pd.DataFrame):
        mask_df = mask_obj.copy()
        mask_df.columns = mask_df.columns.astype(str)
        common = [c for c in binary_cols if c in mask_df.columns]
        if len(mask_df) == N_TE and common:
            mask_df.index = df_te.index
            for col in common:
                mask = pd.to_numeric(mask_df[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.5
                arr = pd.to_numeric(IOT_FULL_SYNTHETIC_TEST[col], errors="coerce").to_numpy(dtype=np.float64)
                n = int(np.isfinite(arr[~mask]).sum())
                inactive_finite_violations += n
                if n:
                    inactive_audit_rows.append({
                        "col": col,
                        "role": "binary_state_12d",
                        "inactive_finite_n": n,
                    })

# 12.e drivers under IOT_DRIVER_AVAIL_SYN if available. Check only; do not mutate.
if "IOT_DRIVER_AVAIL_SYN_DF" in globals() or "IOT_DRIVER_AVAIL_SYN" in globals():
    mask_obj = globals().get("IOT_DRIVER_AVAIL_SYN_DF", globals().get("IOT_DRIVER_AVAIL_SYN", None))
    if isinstance(mask_obj, pd.DataFrame):
        mask_df = mask_obj.copy()
        mask_df.columns = mask_df.columns.astype(str)
        common = [c for c in driver_cols if c in mask_df.columns]
        if len(mask_df) == N_TE and common:
            mask_df.index = df_te.index
            for col in common:
                mask = pd.to_numeric(mask_df[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.5
                arr = pd.to_numeric(IOT_FULL_SYNTHETIC_TEST[col], errors="coerce").to_numpy(dtype=np.float64)
                n = int(np.isfinite(arr[~mask]).sum())
                inactive_finite_violations += n
                if n:
                    inactive_audit_rows.append({
                        "col": col,
                        "role": "driver_event_12e",
                        "inactive_finite_n": n,
                    })

inactive_audit_df = pd.DataFrame(inactive_audit_rows, columns=["col", "role", "inactive_finite_n"])
inactive_audit_df.to_csv(namespace_inactive_audit_csv, index=False)

namespace_mask_enforcement_report = {
    "cell": "12.g",
    "substep": "10",
    "role": "final_namespace_continuous_observability_enforcement",
    "timestamp_utc": _datetime12G.now(_timezone12G.utc).isoformat(),
    "continuous_mask_path": str(_cont_mask_path_12g),
    "continuous_mask_overlap_cols": int(len(_cont_mask_overlap_12g)),
    "pre_enforcement_continuous_inactive_finite_n": int(_pre_enforcement_inactive_finite_12g),
    "changed_to_nan_total": int(_changed_to_nan_12g),
    "inactive_finite_violations_after": int(inactive_finite_violations),
    "policy": {
        "synthetic_values_mutated": True,
        "mutation_type": "deterministic_namespace_observability_enforcement",
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "TEST_real_values_read": False,
        "qa_metrics_improved_here": False,
        "branch_decisions_changed": False,
    },
}

namespace_mask_enforcement_report_json = os.path.join(REPORT_DIR, "cell12g_final_namespace_mask_enforcement_report.json")
namespace_mask_enforcement_contract_json = os.path.join(CONTRACT_DIR, "cell12g_final_namespace_mask_enforcement_contract.json")
_write_json_12g(namespace_mask_enforcement_report_json, namespace_mask_enforcement_report)
_write_json_12g(namespace_mask_enforcement_contract_json, namespace_mask_enforcement_report)

log(
    "[Cell12.g] Namespace observability enforcement | "
    f"continuous_mask_overlap={len(_cont_mask_overlap_12g)} | "
    f"pre_enforcement_continuous_inactive_finite={_pre_enforcement_inactive_finite_12g} | "
    f"changed_to_nan={_changed_to_nan_12g} | "
    f"inactive_finite_after={inactive_finite_violations}"
)

if inactive_finite_violations and bool(CFG.get("cell12g_fail_on_inactive_finite_violations", True)):
    raise RuntimeError(
        "[Cell12.g] Inactive finite violations detected after final namespace observability enforcement. "
        f"inactive_finite_violations={inactive_finite_violations}; see {namespace_inactive_audit_csv}"
    )
# ----------------------------------------------------------
# 11) Column registry and audits
# ----------------------------------------------------------
registry_rows = []

for role, cols in role_col_map.items():
    for ordinal, col in enumerate(cols):
        arr = pd.to_numeric(IOT_FULL_SYNTHETIC_TEST[col], errors="coerce").to_numpy(dtype=np.float64)
        finite_n = int(np.isfinite(arr).sum())
        registry_rows.append({
            "col": col,
            "role": role,
            "role_ordinal": int(ordinal),
            "entity": _entity_from_col_12g(col),
            "rows": int(N_TE),
            "finite_n": finite_n,
            "finite_rate": float(finite_n / max(N_TE, 1)),
            "all_nan": bool(finite_n == 0),
            "is_placeholder": bool(role in {"excluded_placeholder", "namespace_residual_placeholder"}),
            "TEST_real_values_used": False,
        })

registry_df = pd.DataFrame(registry_rows)

if len(registry_df) != full_namespace_cols:
    raise RuntimeError(
        f"[Cell12.g] Registry row mismatch: registry={len(registry_df)} full_cols={full_namespace_cols}"
    )

placeholder_rows = []
for col in excluded_cols:
    placeholder_rows.append({
        "col": col,
        "placeholder_role": "excluded_placeholder",
        "reason": "excluded_or_out_of_scope_iot_column_handled_as_placeholder",
        "all_nan_expected": True,
        "TEST_values_used": False,
    })

for col in residual_cols:
    placeholder_rows.append({
        "col": col,
        "placeholder_role": "namespace_residual_placeholder",
        "reason": "residual_namespace_slot_to_reach_canonical_530_columns",
        "all_nan_expected": True,
        "TEST_values_used": False,
    })

placeholder_df = pd.DataFrame(placeholder_rows)
placeholder_df.to_csv(namespace_placeholder_audit_csv, index=False)

# Main audit.
assembly_audit_rows = [
    {"metric": "full_namespace_cols", "value": int(full_namespace_cols)},
    {"metric": "expected_full_namespace_cols", "value": int(EXPECTED_FULL_NAMESPACE_COLS_12G)},
    {"metric": "continuous_source_present_n", "value": int(CONTINUOUS_SOURCE_COLS_N_12G)},
    {"metric": "continuous_final_owned_n", "value": int(len(continuous_cols))},
    {"metric": "continuous_binary_override_n", "value": int(CONTINUOUS_BINARY_OVERRIDE_N_12G)},
    {"metric": "binary_present_n", "value": int(len(binary_cols))},
    {"metric": "drivers_present_n", "value": int(len(driver_cols))},
    {"metric": "observability_present_n", "value": int(len(observability_cols))},
    {"metric": "excluded_handled_n", "value": int(len(excluded_cols))},
    {"metric": "namespace_residual_placeholder_n", "value": int(len(residual_cols))},
    {"metric": "inactive_finite_violations", "value": int(inactive_finite_violations)},
    {"metric": "role_owner_violations", "value": int(role_owner_violations)},
    {"metric": "test_rows", "value": int(N_TE)},
    {"metric": "TEST_real_values_used", "value": False},
    {"metric": "selection_done_here", "value": False},
    {"metric": "generator_fit_done_here", "value": False},
]

for role, cols in role_col_map.items():
    assembly_audit_rows.append({"metric": f"role_count::{role}", "value": int(len(cols))})

assembly_audit_df = pd.DataFrame(assembly_audit_rows)

# ----------------------------------------------------------
# 12) Save outputs
# ----------------------------------------------------------
IOT_FULL_SYNTHETIC_TEST.to_parquet(full_iot_path, index=True)
assembly_audit_df.to_csv(assembly_audit_csv, index=False)
registry_df.to_csv(namespace_column_registry_csv, index=False)

branch_status = {
    "12c_continuous": {
        "source_cols": int(CONTINUOUS_SOURCE_COLS_N_12G),
        "final_owned_cols": int(len(continuous_cols)),
        "binary_override_n": int(CONTINUOUS_BINARY_OVERRIDE_N_12G),
        "binary_override_cols": binary_override_applied_cols,
        "path_or_global": "IOT_FINAL_VALUES_TEST",
        "expected_source_cols": int(EXPECTED_CONTINUOUS_COLS_12G),
        "source_present": bool(CONTINUOUS_SOURCE_COLS_N_12G == EXPECTED_CONTINUOUS_COLS_12G),
    },
    "12d_binary": {
        "cols": int(len(binary_cols)),
        "path_or_global": "IOT_FINAL_BINARY_TEST",
        "expected": int(EXPECTED_BINARY_COLS_12G),
        "present": bool(len(binary_cols) == EXPECTED_BINARY_COLS_12G),
    },
    "12e_drivers": {
        "cols": int(len(driver_cols)),
        "path_or_global": "IOT_FINAL_DRIVER_TEST",
        "expected": int(EXPECTED_DRIVER_COLS_12G),
        "present": bool(len(driver_cols) == EXPECTED_DRIVER_COLS_12G),
    },
    "12f_observability": {
        "cols": int(len(observability_cols)),
        "path_or_global": "IOT_SELECTED_OBSERVABILITY_MASKS_TEST",
        "present": bool(len(observability_cols) > 0),
        "qa_contract": "CELL12F5_MASK_CONTRACT",
        "qa_version": q3_version_12g,
        "publication_status_counts_after_12f5": q3_publication_status_counts_12g,
        "blocker_origin_counts_after_12f5": q3_blocker_origin_counts_12g,
        "publication_blocker_n": int(q3_publication_blocker_n_12g),
        "upstream_publication_blocker_n": int(q3_upstream_blocker_n_12g),
        "split_drift_blocker_n": int(q3_split_drift_blocker_n_12g),
        "new_test_qa_blocker_n": int(q3_new_test_qa_blocker_n_12g),
        "quality_status": "quality_stratified_with_blockers" if q3_publication_blocker_n_12g > 0 else "publication_clean",
    },
    "excluded_placeholders": {
        "cols": int(len(excluded_cols)),
        "expected": int(EXPECTED_EXCLUDED_COLS_12G),
        "handled": bool(len(excluded_cols) == EXPECTED_EXCLUDED_COLS_12G),
    },
    "namespace_residual_placeholders": {
        "cols": int(len(residual_cols)),
        "reason": "fills canonical namespace to 530 columns after known generated/excluded branches",
    },
}

contract = {
    "cell": "12.g",
    "version": CELL12G_VERSION,
    "role": "full_iot_namespace_assembly",
    "test_rows": int(N_TE),
    "full_namespace_cols": int(full_namespace_cols),
    "expected_full_namespace_cols": int(EXPECTED_FULL_NAMESPACE_COLS_12G),
    "hard_checks": {
        "full_namespace_cols_eq_530": bool(full_namespace_cols == EXPECTED_FULL_NAMESPACE_COLS_12G),
        "continuous_148_source_present": bool(CONTINUOUS_SOURCE_COLS_N_12G == EXPECTED_CONTINUOUS_COLS_12G),
        "continuous_final_owned_after_binary_override": int(len(continuous_cols)),
        "continuous_binary_override_n": int(CONTINUOUS_BINARY_OVERRIDE_N_12G),
        "binary_52_present": bool(len(binary_cols) == EXPECTED_BINARY_COLS_12G),
        "drivers_26_present": bool(len(driver_cols) == EXPECTED_DRIVER_COLS_12G),
        "excluded_72_handled": bool(len(excluded_cols) == EXPECTED_EXCLUDED_COLS_12G),
        "inactive_finite_violations_eq_0": bool(inactive_finite_violations == 0),
        "role_owner_violations_eq_0": bool(role_owner_violations == 0),
    },
    "branch_status": branch_status,
    "role_counts": {k: int(len(v)) for k, v in role_col_map.items()},
    "inactive_finite_violations": int(inactive_finite_violations),
    "role_owner_violations": int(role_owner_violations),
    "quality_status_carried_forward": {
        "q3_observability": {
            "cell12f5_version": q3_version_12g,
            "publication_blocker_n": int(q3_publication_blocker_n_12g),
            "upstream_publication_blocker_n": int(q3_upstream_blocker_n_12g),
            "split_drift_blocker_n": int(q3_split_drift_blocker_n_12g),
            "new_test_qa_blocker_n": int(q3_new_test_qa_blocker_n_12g),
            "publication_status_counts_after_12f5": q3_publication_status_counts_12g,
            "blocker_origin_counts_after_12f5": q3_blocker_origin_counts_12g,
            "mean_obs_rate_error": q3_metric_summary_12g.get("mean_obs_rate_error", None),
            "mean_p11_error": q3_metric_summary_12g.get("mean_p11_error", None),
            "mean_p00_error": q3_metric_summary_12g.get("mean_p00_error", None),
            "mean_run_length_ks": q3_metric_summary_12g.get("mean_run_length_ks", None),
            "mean_regime_obs_rate_error": q3_metric_summary_12g.get("mean_regime_obs_rate_error", None),
            "mean_mask_c2st_auc": q3_metric_summary_12g.get("mean_mask_c2st_auc", None),
            "synthetic_values_mutated": False,
            "TEST_real_values_used_for_QA_only": True,
        }
    },
    "excluded_discovery": {
        "excluded_candidates_all_n": int(len(excluded_candidates_all)),
        "excluded_overflow_candidates_n": int(len(excluded_overflow_cols)),
        "excluded_final_n": int(len(excluded_cols)),
        "excluded_was_padded_with_synthetic_placeholders": bool(
            any(c.startswith("iot__excluded_placeholder__") for c in excluded_cols)
        ),
    },
    "namespace_residual": {
        "residual_placeholder_n": int(len(residual_cols)),
        "residual_placeholders_used": bool(len(residual_cols) > 0),
    },
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "test_length_materialization_done_here": False,
    "full_namespace_assembly_done_here": True,
    "outputs": {
        "full_iot_path": full_iot_path,
        "assembly_audit_csv": assembly_audit_csv,
        "namespace_contract_json": namespace_contract_json,
        "namespace_contract_canonical_json": namespace_contract_canonical_json,
        "namespace_column_registry_csv": namespace_column_registry_csv,
        "namespace_overlap_audit_csv": namespace_overlap_audit_csv,
        "namespace_placeholder_audit_csv": namespace_placeholder_audit_csv,
        "namespace_inactive_audit_csv": namespace_inactive_audit_csv,
        "namespace_manifest_json": namespace_manifest_json,
    },
}

_write_json_12g(namespace_contract_json, contract)
_write_json_12g(namespace_contract_canonical_json, contract)

manifest = {
    "cell": "12.g",
    "version": CELL12G_VERSION,
    "created_outputs": contract["outputs"],
    "full_namespace_cols": int(full_namespace_cols),
    "role_counts": contract["role_counts"],
    "hard_checks": contract["hard_checks"],
    "quality_status_carried_forward": contract["quality_status_carried_forward"],
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "full_namespace_assembly_done_here": True,
    },
}

_write_json_12g(namespace_manifest_json, manifest)

hashes = {
    "full_iot_sha256": _sha256_file_12g(full_iot_path),
    "assembly_audit_csv_sha256": _sha256_file_12g(assembly_audit_csv),
    "namespace_column_registry_csv_sha256": _sha256_file_12g(namespace_column_registry_csv),
    "namespace_overlap_audit_csv_sha256": _sha256_file_12g(namespace_overlap_audit_csv),
    "namespace_placeholder_audit_csv_sha256": _sha256_file_12g(namespace_placeholder_audit_csv),
    "namespace_inactive_audit_csv_sha256": _sha256_file_12g(namespace_inactive_audit_csv),
    "namespace_contract_json_sha256": _sha256_file_12g(namespace_contract_json),
    "namespace_contract_canonical_json_sha256": _sha256_file_12g(namespace_contract_canonical_json),
    "namespace_manifest_json_sha256": _sha256_file_12g(namespace_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12g(namespace_contract_json, contract)
_write_json_12g(namespace_contract_canonical_json, contract)
_write_json_12g(namespace_manifest_json, manifest)

# ----------------------------------------------------------
# 13) Export globals
# ----------------------------------------------------------
globals()["CELL12G_VERSION"] = CELL12G_VERSION
globals()["IOT_FULL_SYNTHETIC_TEST"] = IOT_FULL_SYNTHETIC_TEST
globals()["CELL12G_FULL_IOT_SYNTHETIC_TEST_PATH"] = full_iot_path
globals()["CELL12G_IOT_FULL_ASSEMBLY_AUDIT_DF"] = assembly_audit_df
globals()["CELL12G_IOT_FULL_NAMESPACE_COLUMN_REGISTRY_DF"] = registry_df
globals()["CELL12G_IOT_FULL_NAMESPACE_CONTRACT"] = contract
globals()["CELL12G_IOT_FULL_ASSEMBLY_AUDIT_CSV"] = assembly_audit_csv
globals()["CELL12G_IOT_FULL_NAMESPACE_CONTRACT_JSON"] = namespace_contract_json
globals()["CELL12G_IOT_FULL_NAMESPACE_CONTRACT_CANONICAL_JSON"] = namespace_contract_canonical_json
globals()["CELL12G_IOT_FULL_NAMESPACE_COLUMN_REGISTRY_CSV"] = namespace_column_registry_csv
globals()["CELL12G_IOT_FULL_NAMESPACE_MANIFEST_JSON"] = namespace_manifest_json

log(
    "[Cell12.g] Full IoT namespace assembly complete | "
    f"shape={IOT_FULL_SYNTHETIC_TEST.shape} | "
    f"full_namespace_cols={full_namespace_cols} | "
    f"continuous_source={CONTINUOUS_SOURCE_COLS_N_12G} | "
    f"continuous_final_owned={len(continuous_cols)} | "
    f"binary_override_n={CONTINUOUS_BINARY_OVERRIDE_N_12G} | "
    f"binary={len(binary_cols)} | "
    f"drivers={len(driver_cols)} | "
    f"observability={len(observability_cols)} | "
    f"excluded={len(excluded_cols)} | "
    f"residual_placeholders={len(residual_cols)}"
)
log(
    "[Cell12.g] Hard checks | "
    f"full_namespace_cols=530:{full_namespace_cols == 530} | "
    f"continuous_source=148:{CONTINUOUS_SOURCE_COLS_N_12G == 148} | "
    f"continuous_final_owned={len(continuous_cols)} | "
    f"binary_override_n={CONTINUOUS_BINARY_OVERRIDE_N_12G} | "
    f"binary=52:{len(binary_cols) == 52} | "
    f"drivers=26:{len(driver_cols) == 26} | "
    f"excluded=72:{len(excluded_cols) == 72} | "
    f"inactive_finite_violations={inactive_finite_violations} | "
    f"role_owner_violations={role_owner_violations}"
)
log(f"[Cell12.g] Saved full IoT synthetic TEST parquet: {full_iot_path}")
log(f"[Cell12.g] Saved assembly audit: {assembly_audit_csv}")
log(f"[Cell12.g] Saved namespace contract: {namespace_contract_json}")
log(f"[Cell12.g] Saved canonical namespace contract: {namespace_contract_canonical_json}")
log(
    "[Cell12.g] Q3 quality carried forward | "
    f"publication_blocker_n={q3_publication_blocker_n_12g} | "
    f"upstream_blocker_n={q3_upstream_blocker_n_12g} | "
    f"split_drift_blocker_n={q3_split_drift_blocker_n_12g} | "
    f"new_test_qa_blocker_n={q3_new_test_qa_blocker_n_12g}"
)
log(f"[Cell12.g] Saved namespace column registry: {namespace_column_registry_csv}")
log(
    "[Cell12.g] Contract flags | "
    "TEST_real_values_used=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "full_namespace_assembly_done_here=True | quality_status_carried_forward=True"
)
log("--- END: Cell 12.g - Full IoT namespace assembly (v1.2 strict, quality-status-aware) ---")

gc.collect()
