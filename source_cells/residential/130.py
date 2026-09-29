# ==========================================================
# CELL 15.0 - Q6 privacy input preparation by role group
# v1.1 STUDY-THESIS strict no-Q4-promotion privacy input contract
#
# Role:
#   - Prepare role-specific privacy inputs for Q6 after Q4 was closed as
#     blocked_no_promotion.
#   - DO NOT load or use stale/legacy Q4-coupled final artifacts.
#   - Assemble the authoritative Q6 synthetic TEST input from the accepted
#     non-Q4 components:
#         PROTOCOL_SYN_TEST_130
#         IOT_FULL_SYN_TEST_130
#     plus canonical TEST time column(s) when available.
#   - Build real TRAIN / VAL / TEST and synthetic TEST role-group matrices.
#
# Scientific contract:
#   - No fitting.
#   - No privacy decision here.
#   - No synthetic value mutation.
#   - No Q4 promotion.
#   - TEST real values are loaded as privacy reference only.
#   - Q4 final status remains blocked_no_promotion.
#
# Outputs:
#   synthetic/PROTOCOL_SYN_TEST_FINAL_NO_Q4.parquet
#   synthetic/CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet
#   reports/cell15_0_privacy_role_group_registry.csv
#   reports/cell15_0_privacy_input_audit.csv
#   reports/cell15_0_privacy_column_exclusion_audit.csv
#   reports/cell15_0_privacy_input_contract.json
#   artifacts/contracts/cell15_0_privacy_input_contract_v1_1_THESIS.json
#   artifacts/cell15_0_privacy_role_manifest.json
#
# Exports:
#   CELL15_0_PRIVACY_ROLE_GROUPS
#   CELL15_0_PRIVACY_ROLE_GROUP_REGISTRY_DF
#   CELL15_0_PRIVACY_INPUT_CONTRACT
#   CPS_SYNTHETIC_TEST_FINAL_NO_Q4
#   CPS_SYNTHETIC_TEST_Q6_INPUT
# ==========================================================

log("--- START: Cell 15.0 - Q6 privacy input preparation by role group (v1.1 no-Q4-promotion strict) ---")

import os
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_150 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "PROTOCOL_SYN_TEST_130",
    "IOT_FULL_SYN_TEST_130",
    "CELL14_10_FINAL_Q4_DECISION_CONTRACT",
    "CELL14_11_FINAL_Q4_SCOPE_AUDIT_CONTRACT",
    "CELL14_12_Q4_NO_PROMOTION_FRESHNESS_SUMMARY",
]
_missing_150 = [k for k in _required_150 if k not in globals()]
if _missing_150:
    raise RuntimeError(f"[Cell15.0] Missing required globals: {_missing_150}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_150 = str(OUT_SYN)
REPORT_DIR_ACTIVE_150 = str(REPORT_DIR)

def _resolve_project_root_150(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        if "q6_public_reaudit" in parts:
            candidates.append(os.sep.join(parts[:parts.index("q6_public_reaudit")]))
        elif os.path.basename(p) in {"reports", "synthetic", "artifacts"}:
            candidates.append(os.path.dirname(p))
        else:
            candidates.append(p)

    _env_project_root = os.environ.get("CPS_CANONICAL_PROJECT_ROOT", "").strip()
    if _env_project_root:
        candidates.append(_env_project_root)

    seen = set()
    for c in candidates:
        c = os.path.abspath(str(c))
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "synthetic")):
            return c

    raise RuntimeError("[Cell15.0] Could not resolve canonical project root.")

PROJECT_ROOT_150 = _resolve_project_root_150(OUTDIR, REPORT_DIR_ACTIVE_150, OUT_SYN_ACTIVE_150)
OUT_SYN = os.path.join(PROJECT_ROOT_150, "synthetic")
REPORT_DIR = os.path.join(PROJECT_ROOT_150, "reports")
ARTDIR = os.path.join(PROJECT_ROOT_150, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

CELL150_VERSION = "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion"

CFG["cell15_0_version"] = CELL150_VERSION
CFG["cell15_0_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_0_Q4_coupled_artifacts_used"] = False
CFG["cell15_0_TEST_real_values_used_for_privacy_reference"] = True
CFG["cell15_0_TEST_real_values_used_for_model_fitting"] = False
CFG["cell15_0_synthetic_values_mutated"] = False
CFG["cell15_0_selection_done_here"] = False
CFG["cell15_0_generator_fit_done_here"] = False
CFG["cell15_0_materialization_done_here"] = False
CFG["cell15_0_privacy_decision_done_here"] = False
CFG["cell15_0_input_preparation_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell15_0_expected_synthetic_rows", N_TE)
CFG.setdefault("cell15_0_fail_on_row_mismatch", True)
CFG.setdefault("cell15_0_fail_on_empty_core_role_group", True)

# Privacy dimensionality controls for later cells.
CFG.setdefault("cell15_0_full_cps_mixed_max_cols", 120)
CFG.setdefault("cell15_0_full_cps_mixed_include_protocol_n", 40)
CFG.setdefault("cell15_0_full_cps_mixed_include_iot_cont_n", 35)
CFG.setdefault("cell15_0_full_cps_mixed_include_iot_binary_n", 20)
CFG.setdefault("cell15_0_full_cps_mixed_include_driver_n", 15)
CFG.setdefault("cell15_0_full_cps_mixed_include_obs_n", 10)
CFG.setdefault("cell15_0_low_variance_eps", 1e-12)

# Expected counts are advisory; do not fail solely on these because the
# release namespace can change after governance.
CFG.setdefault("cell15_0_expected_iot_binary_states", 52)
CFG.setdefault("cell15_0_expected_iot_event_drivers", 26)
CFG.setdefault("cell15_0_expected_iot_full_namespace", 530)

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
final_protocol_noq4_path = os.path.join(OUT_SYN, "PROTOCOL_SYN_TEST_FINAL_NO_Q4.parquet")
final_cps_noq4_path = os.path.join(OUT_SYN, "CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet")

role_registry_csv = os.path.join(REPORT_DIR, "cell15_0_privacy_role_group_registry.csv")
input_audit_csv = os.path.join(REPORT_DIR, "cell15_0_privacy_input_audit.csv")
column_exclusion_audit_csv = os.path.join(REPORT_DIR, "cell15_0_privacy_column_exclusion_audit.csv")
contract_json = os.path.join(REPORT_DIR, "cell15_0_privacy_input_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell15_0_privacy_input_contract_v1_1_THESIS.json")
role_manifest_json = os.path.join(ARTDIR, "cell15_0_privacy_role_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_150(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_150(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_150(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_150(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_150(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_150(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_150(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_150(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_150(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_150(payload), f, indent=2, sort_keys=True)

def _sha256_file_150(path: str) -> str:
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_150(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.0] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.0] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

def _dedup_150(seq):
    seen = set()
    out = []
    for x in seq:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _protocol_tier_150(c: str) -> str:
    s = str(c)
    if s.startswith("router__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__"):
        return "ota"
    if s.startswith("zigbee__") or s.startswith("zb__"):
        return "zigbee"
    if s.startswith("zwave__"):
        return "zwave"
    return "other"

def _is_time_col_150(c: str) -> bool:
    s = str(c).lower()
    return s in {"sec", "sec_epoch_s__canon", "timestamp", "time", "datetime"} or "time_iso" in s

def _is_iot_col_150(c: str) -> bool:
    s = str(c)
    return s.startswith("iot__") or s.startswith("events_in_sec__")

def _is_driver_col_150(c: str) -> bool:
    return str(c).startswith("events_in_sec__")

def _is_entity_obs_col_150(c: str) -> bool:
    return str(c).startswith("iot__entity_obs__")

def _is_entity_stale_col_150(c: str) -> bool:
    return str(c).startswith("iot__entity_stale__")

def _is_global_obs_col_150(c: str) -> bool:
    s = str(c)
    return s in {"iot__obs_present", "iot__any_update_raw", "iot__tier_present"}

def _is_observability_col_150(c: str) -> bool:
    s = str(c)
    return (
        _is_entity_obs_col_150(s)
        or _is_entity_stale_col_150(s)
        or _is_global_obs_col_150(s)
        or ("__obs_present" in s)
        or ("__stale" in s)
        or s.endswith("__present")
        or "availability" in s.lower()
    )

def _is_placeholder_col_150(c: str) -> bool:
    s = str(c)
    return (
        "placeholder" in s.lower()
        or "excluded" in s.lower()
        or s.startswith("iot__namespace_residual_placeholder__")
        or s.startswith("iot__excluded_placeholder__")
    )

def _is_binary_like_name_150(c: str) -> bool:
    s = str(c).lower()
    return (
        "__binary_sensor__" in s
        or "__switch__" in s
        or s.endswith("__state")
        or "__state__state" in s
        or "enabled" in s
        or "overheated" in s
        or "overloaded" in s
        or "attached" in s
        or "connection" in s
    )

def _is_continuous_value_name_150(c: str) -> bool:
    s = str(c)
    return s.startswith("iot__") and s.endswith("__value") and not _is_observability_col_150(s)

def _numeric_stats_150(frame: pd.DataFrame, cols: list):
    rows = []
    for c in cols:
        if c not in frame.columns:
            rows.append({
                "col": c,
                "exists": False,
                "finite_rate": np.nan,
                "nonzero_rate": np.nan,
                "unique_n_sample": np.nan,
                "std": np.nan,
            })
            continue
        x = pd.to_numeric(frame[c], errors="coerce")
        finite = x.notna()
        finite_rate = float(finite.mean()) if len(x) else np.nan
        if finite.any():
            vals = x[finite].to_numpy(dtype=np.float64)
            nonzero_rate = float(np.mean(vals != 0))
            unique_n = int(pd.Series(vals).nunique(dropna=True))
            std = float(np.nanstd(vals)) if vals.size else np.nan
        else:
            nonzero_rate = np.nan
            unique_n = 0
            std = np.nan
        rows.append({
            "col": c,
            "exists": True,
            "finite_rate": finite_rate,
            "nonzero_rate": nonzero_rate,
            "unique_n_sample": unique_n,
            "std": std,
        })
    return pd.DataFrame(rows)

def _role_summary_150(role_name, cols, real_tr, real_val, real_te, syn_te):
    cols = _dedup_150(cols)
    common_tr = [c for c in cols if c in real_tr.columns]
    common_val = [c for c in cols if c in real_val.columns]
    common_te = [c for c in cols if c in real_te.columns]
    common_syn = [c for c in cols if c in syn_te.columns]
    common_all = sorted(set(common_tr) & set(common_val) & set(common_te) & set(common_syn))

    return {
        "role_group": role_name,
        "registered_cols_n": int(len(cols)),
        "common_all_n": int(len(common_all)),
        "real_train_cols_n": int(len(common_tr)),
        "real_val_cols_n": int(len(common_val)),
        "real_test_cols_n": int(len(common_te)),
        "synthetic_test_cols_n": int(len(common_syn)),
        "missing_real_train_n": int(len(set(cols) - set(common_tr))),
        "missing_real_val_n": int(len(set(cols) - set(common_val))),
        "missing_real_test_n": int(len(set(cols) - set(common_te))),
        "missing_synthetic_test_n": int(len(set(cols) - set(common_syn))),
        "real_train_rows": int(len(real_tr)),
        "real_val_rows": int(len(real_val)),
        "real_test_rows": int(len(real_te)),
        "synthetic_test_rows": int(len(syn_te)),
        "privacy_reference_real_TEST_used": True,
        "synthetic_values_mutated": False,
    }

def _select_full_mixed_cols_150(role_groups: dict, syn_te: pd.DataFrame):
    rng = np.random.default_rng(SEED + 1500)
    selected = []

    quotas = [
        ("protocol_router", int(CFG.get("cell15_0_full_cps_mixed_include_protocol_n", 40))),
        ("protocol_ota", max(0, int(CFG.get("cell15_0_full_cps_mixed_include_protocol_n", 40)) // 5)),
        ("protocol_zigbee", max(0, int(CFG.get("cell15_0_full_cps_mixed_include_protocol_n", 40)) // 5)),
        ("iot_continuous_values", int(CFG.get("cell15_0_full_cps_mixed_include_iot_cont_n", 35))),
        ("iot_binary_states", int(CFG.get("cell15_0_full_cps_mixed_include_iot_binary_n", 20))),
        ("iot_event_drivers", int(CFG.get("cell15_0_full_cps_mixed_include_driver_n", 15))),
        ("iot_observability_masks", int(CFG.get("cell15_0_full_cps_mixed_include_obs_n", 10))),
    ]

    for group, n in quotas:
        cols = [c for c in role_groups.get(group, []) if c in syn_te.columns]
        if not cols or n <= 0:
            continue

        stats = _numeric_stats_150(syn_te, cols)
        stats["std"] = pd.to_numeric(stats["std"], errors="coerce").fillna(0.0)
        stats["finite_rate"] = pd.to_numeric(stats["finite_rate"], errors="coerce").fillna(0.0)

        eligible = stats[
            stats["exists"].astype(bool)
            & (stats["finite_rate"] > 0.01)
            & (stats["std"] > float(CFG.get("cell15_0_low_variance_eps", 1e-12)))
        ]["col"].astype(str).tolist()

        if len(eligible) <= n:
            chosen = eligible
        else:
            chosen = rng.choice(np.asarray(eligible, dtype=object), size=n, replace=False).tolist()

        selected.extend(chosen)

    selected = _dedup_150(selected)
    max_cols = int(CFG.get("cell15_0_full_cps_mixed_max_cols", 120))
    return selected[:max_cols]

# ----------------------------------------------------------
# 4) Validate Q4 no-promotion governance
# ----------------------------------------------------------
version1410_150 = _require_contract_version_150(
    CELL14_10_FINAL_Q4_DECISION_CONTRACT,
    "CELL14_10_FINAL_Q4_DECISION_CONTRACT",
    "cell14_10_final_q4_no_promotion_decision_ledger_v1_1",
)
version1411_150 = _require_contract_version_150(
    CELL14_11_FINAL_Q4_SCOPE_AUDIT_CONTRACT,
    "CELL14_11_FINAL_Q4_SCOPE_AUDIT_CONTRACT",
    "cell14_11_final_q4_no_promotion_scope_evidence_audit_v1_1",
)

final_decision_1410 = CELL14_10_FINAL_Q4_DECISION_CONTRACT.get("final_decision", {})
if str(final_decision_1410.get("final_q4_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.0] Q4 final status is not blocked_no_promotion.")
if str(final_decision_1410.get("accepted_candidate", "")) != "":
    raise RuntimeError("[Cell15.0] Q4 accepted candidate is non-empty; this no-Q4 input cell should not be used.")
if bool(final_decision_1410.get("A0_promoted", True)):
    raise RuntimeError("[Cell15.0] Q4 A0 was promoted unexpectedly.")

if str(CELL14_11_FINAL_Q4_SCOPE_AUDIT_CONTRACT.get("final_q4_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.0] Cell 14.11 did not preserve blocked_no_promotion.")
if bool(CELL14_11_FINAL_Q4_SCOPE_AUDIT_CONTRACT.get("A0_promoted", True)):
    raise RuntimeError("[Cell15.0] Cell 14.11 indicates A0 was promoted.")

freshness = CELL14_12_Q4_NO_PROMOTION_FRESHNESS_SUMMARY

if not isinstance(freshness, dict):
    raise RuntimeError("[Cell15.0] Cell 14.12 freshness summary is not a dict.")

freshness_final_q4_status = str(freshness.get("final_q4_status", ""))
freshness_decision_severity = str(freshness.get("decision_severity", ""))
freshness_decision = str(freshness.get("decision", ""))

if freshness_final_q4_status != "blocked_no_promotion":
    raise RuntimeError(
        "[Cell15.0] Cell 14.12 final_q4_status is not blocked_no_promotion: "
        f"{freshness_final_q4_status}"
    )

# Cell 15.0 prepares Q6 inputs from the accepted no-Q4 components:
#   PROTOCOL_SYN_TEST_130
#   IOT_FULL_SYN_TEST_130
# It does not use promoted Q4 artifacts. Therefore, a Cell 14.12 freshness
# severity of "fail" must not promote or validate Q4, but it also should not
# block Q6 input preparation. Instead, record it as a no-Q4 freshness limitation.
if freshness_decision_severity not in {"pass", "warning"}:
    log(
        "[Cell15.0][Q4 FRESHNESS WARNING] Cell 14.12 freshness severity is "
        f"{freshness_decision_severity!r}. Continuing because Q4 remains "
        "blocked_no_promotion and Cell 15.0 does not use Q4-coupled artifacts. "
        "This run must not claim fresh positive Q4 evidence."
    )

CELL15_0_Q4_FRESHNESS_GOVERNANCE = {
    "cell14_12_final_q4_status": freshness_final_q4_status,
    "cell14_12_decision": freshness_decision,
    "cell14_12_decision_severity": freshness_decision_severity,
    "cell15_0_action": (
        "continue_no_q4_q6_input_preparation"
        if freshness_final_q4_status == "blocked_no_promotion"
        else "stop"
    ),
    "q4_positive_claim_allowed": False,
    "q4_coupled_artifacts_used": False,
    "q4_freshness_failure_interpretation": (
        "Q4 freshness failure is recorded as a no-promotion limitation. "
        "It does not validate any Q4 claim and does not block Q6 input preparation "
        "because Cell 15.0 assembles the no-Q4 synthetic CPS input only."
    ),
}

globals()["CELL15_0_Q4_FRESHNESS_GOVERNANCE"] = CELL15_0_Q4_FRESHNESS_GOVERNANCE

# ----------------------------------------------------------
# 5) Assemble authoritative no-Q4 synthetic TEST CPS input
# ----------------------------------------------------------
protocol_noq4 = PROTOCOL_SYN_TEST_130.copy()
iot_noq4 = IOT_FULL_SYN_TEST_130.copy()

protocol_noq4.index = df_te.index
iot_noq4.index = df_te.index
protocol_noq4.columns = protocol_noq4.columns.astype(str)
iot_noq4.columns = iot_noq4.columns.astype(str)

if len(protocol_noq4) != N_TE or len(iot_noq4) != N_TE:
    raise RuntimeError(
        "[Cell15.0] Component row mismatch: "
        f"protocol={len(protocol_noq4)} iot={len(iot_noq4)} expected={N_TE}"
    )

# Save protocol input for traceability.
protocol_noq4.to_parquet(final_protocol_noq4_path, index=True)

time_cols = [c for c in ["sec_epoch_s__canon", "sec", "timestamp", "time", "datetime"] if c in df_te.columns]
time_df = pd.DataFrame(index=df_te.index)
for c in time_cols:
    if c not in protocol_noq4.columns and c not in iot_noq4.columns:
        time_df[c] = df_te[c].to_numpy(copy=True)

parts = []
if len(time_df.columns):
    parts.append(time_df)
parts.extend([protocol_noq4, iot_noq4])

CPS_SYNTHETIC_TEST_FINAL_NO_Q4 = pd.concat(parts, axis=1)
dupes = CPS_SYNTHETIC_TEST_FINAL_NO_Q4.columns[CPS_SYNTHETIC_TEST_FINAL_NO_Q4.columns.duplicated()].astype(str).tolist()
if dupes:
    raise RuntimeError(f"[Cell15.0] Duplicate columns in no-Q4 CPS assembly: {dupes[:20]}")

if len(CPS_SYNTHETIC_TEST_FINAL_NO_Q4) != N_TE:
    raise RuntimeError(
        "[Cell15.0] Final no-Q4 synthetic CPS row mismatch: "
        f"got={len(CPS_SYNTHETIC_TEST_FINAL_NO_Q4)} expected={N_TE}"
    )

CPS_SYNTHETIC_TEST_FINAL_NO_Q4.to_parquet(final_cps_noq4_path, index=True)

# The real references are existing split frames.
REAL_TRAIN_REF_150 = df_tr.copy()
REAL_VAL_REF_150 = df_val.copy()
REAL_TEST_REF_150 = df_te.copy()
SYN_TEST_REF_150 = CPS_SYNTHETIC_TEST_FINAL_NO_Q4

for _df in [REAL_TRAIN_REF_150, REAL_VAL_REF_150, REAL_TEST_REF_150, SYN_TEST_REF_150]:
    _df.columns = _df.columns.astype(str)

log(
    "[Cell15.0] Prepared no-Q4 privacy references | "
    f"real_train={REAL_TRAIN_REF_150.shape} | "
    f"real_val={REAL_VAL_REF_150.shape} | "
    f"real_test={REAL_TEST_REF_150.shape} | "
    f"syn_test={SYN_TEST_REF_150.shape} | "
    f"q4_status=blocked_no_promotion"
)

# ----------------------------------------------------------
# 6) Build role groups
# ----------------------------------------------------------
syn_cols = list(SYN_TEST_REF_150.columns)

protocol_router_cols = [c for c in syn_cols if _protocol_tier_150(c) == "router"]
protocol_ota_cols = [c for c in syn_cols if _protocol_tier_150(c) == "ota"]
protocol_zigbee_cols = [c for c in syn_cols if _protocol_tier_150(c) == "zigbee"]

iot_event_driver_cols = [c for c in syn_cols if _is_driver_col_150(c)]

iot_observability_cols = [
    c for c in syn_cols
    if _is_iot_col_150(c)
    and _is_observability_col_150(c)
    and not _is_driver_col_150(c)
]

iot_placeholder_cols = [
    c for c in syn_cols
    if _is_iot_col_150(c)
    and _is_placeholder_col_150(c)
]

iot_continuous_value_cols = [
    c for c in syn_cols
    if _is_continuous_value_name_150(c)
    and not _is_driver_col_150(c)
    and not _is_observability_col_150(c)
    and not _is_placeholder_col_150(c)
    and not _is_binary_like_name_150(c)
]

iot_binary_state_cols = [
    c for c in syn_cols
    if _is_iot_col_150(c)
    and not _is_driver_col_150(c)
    and not _is_observability_col_150(c)
    and not _is_placeholder_col_150(c)
    and (
        _is_binary_like_name_150(c)
        or (c.startswith("iot__") and c.endswith("__state"))
    )
]

# If branch registries are available, prefer them for stable ownership.
if isinstance(globals().get("CELL12D_BINARY_TARGET_COLS", None), (list, tuple, set)):
    registry_binary = [str(c) for c in globals()["CELL12D_BINARY_TARGET_COLS"] if str(c) in syn_cols]
    if registry_binary:
        iot_binary_state_cols = _dedup_150(registry_binary)

if isinstance(globals().get("CELL12C_VALUE_COLS", None), (list, tuple, set)):
    registry_cont = [str(c) for c in globals()["CELL12C_VALUE_COLS"] if str(c) in syn_cols]
    if registry_cont:
        iot_continuous_value_cols = [
            c for c in registry_cont
            if c not in set(iot_binary_state_cols)
            and not _is_placeholder_col_150(c)
            and not _is_observability_col_150(c)
        ]

if isinstance(globals().get("CELL12E_DRIVER_TARGET_COLS", None), (list, tuple, set)):
    registry_driver = [str(c) for c in globals()["CELL12E_DRIVER_TARGET_COLS"] if str(c) in syn_cols]
    if registry_driver:
        iot_event_driver_cols = _dedup_150(registry_driver)

role_groups = {
    "protocol_router": _dedup_150(protocol_router_cols),
    "protocol_ota": _dedup_150(protocol_ota_cols),
    "protocol_zigbee": _dedup_150(protocol_zigbee_cols),
    "iot_continuous_values": _dedup_150(iot_continuous_value_cols),
    "iot_binary_states": _dedup_150(iot_binary_state_cols),
    "iot_event_drivers": _dedup_150(iot_event_driver_cols),
    "iot_observability_masks": _dedup_150(iot_observability_cols),
    "iot_placeholders_or_excluded": _dedup_150(iot_placeholder_cols),
}

role_groups["full_cps_mixed_sampled"] = _select_full_mixed_cols_150(role_groups, SYN_TEST_REF_150)

# ----------------------------------------------------------
# 7) Build registry and audits
# ----------------------------------------------------------
registry_rows = []
column_exclusion_rows = []

for role_name, cols in role_groups.items():
    summary = _role_summary_150(
        role_name=role_name,
        cols=cols,
        real_tr=REAL_TRAIN_REF_150,
        real_val=REAL_VAL_REF_150,
        real_te=REAL_TEST_REF_150,
        syn_te=SYN_TEST_REF_150,
    )

    stats_syn = _numeric_stats_150(SYN_TEST_REF_150, cols)
    if len(stats_syn):
        role_finite_mean = float(pd.to_numeric(stats_syn["finite_rate"], errors="coerce").mean())
        role_nonzero_mean = float(pd.to_numeric(stats_syn["nonzero_rate"], errors="coerce").mean())
        role_zero_variance_n = int(
            (pd.to_numeric(stats_syn["std"], errors="coerce").fillna(0.0)
             <= float(CFG.get("cell15_0_low_variance_eps", 1e-12))).sum()
        )
    else:
        role_finite_mean = np.nan
        role_nonzero_mean = np.nan
        role_zero_variance_n = 0

    registry_rows.append({
        **summary,
        "synthetic_finite_rate_mean": role_finite_mean,
        "synthetic_nonzero_rate_mean": role_nonzero_mean,
        "synthetic_zero_or_low_variance_cols_n": role_zero_variance_n,
        "intended_for_15_1_no_copy": bool(role_name != "iot_placeholders_or_excluded"),
        "intended_for_15_2_dcr_nndr": bool(role_name not in {"iot_placeholders_or_excluded"}),
        "intended_for_15_3_mia": bool(role_name not in {"iot_placeholders_or_excluded"}),
        "q4_coupled_artifacts_used": False,
        "notes": (
            "sampled mixed feature set for multivariate privacy smoke tests"
            if role_name == "full_cps_mixed_sampled"
            else "placeholder/excluded columns tracked but excluded from privacy risk metrics"
            if role_name == "iot_placeholders_or_excluded"
            else "role-specific privacy group"
        ),
    })

    missing_syn = sorted(set(cols) - set(SYN_TEST_REF_150.columns))
    missing_real_te = sorted(set(cols) - set(REAL_TEST_REF_150.columns))
    for c in missing_syn:
        column_exclusion_rows.append({
            "role_group": role_name,
            "col": c,
            "reason": "missing_from_synthetic_final_no_q4",
            "privacy_metric_exclusion": True,
        })
    for c in missing_real_te:
        column_exclusion_rows.append({
            "role_group": role_name,
            "col": c,
            "reason": "missing_from_real_test_reference",
            "privacy_metric_exclusion": True,
        })

registry_df = pd.DataFrame(registry_rows)
column_exclusion_df = pd.DataFrame(column_exclusion_rows)

registry_df.to_csv(role_registry_csv, index=False)
column_exclusion_df.to_csv(column_exclusion_audit_csv, index=False)

# ----------------------------------------------------------
# 8) Hard checks
# ----------------------------------------------------------
core_groups = [
    "protocol_router",
    "protocol_ota",
    "protocol_zigbee",
    "iot_continuous_values",
    "iot_binary_states",
    "iot_event_drivers",
    "iot_observability_masks",
]

empty_core = []
for g in core_groups:
    sub = registry_df[registry_df["role_group"].eq(g)]
    if len(sub) == 0 or int(sub["common_all_n"].iloc[0]) == 0:
        empty_core.append(g)

if empty_core and bool(CFG.get("cell15_0_fail_on_empty_core_role_group", True)):
    raise RuntimeError(f"[Cell15.0] Empty core privacy role groups: {empty_core}")

# ----------------------------------------------------------
# 9) Input audit
# ----------------------------------------------------------
input_audit_rows = [
    {"metric": "Q4_final_status", "value": "blocked_no_promotion"},
    {"metric": "Q4_coupled_artifacts_used", "value": False},
    {"metric": "real_train_rows", "value": int(N_TR)},
    {"metric": "real_val_rows", "value": int(N_VAL)},
    {"metric": "real_test_rows", "value": int(N_TE)},
    {"metric": "synthetic_test_rows", "value": int(len(SYN_TEST_REF_150))},
    {"metric": "synthetic_test_cols", "value": int(SYN_TEST_REF_150.shape[1])},
    {"metric": "final_no_q4_cps_path", "value": final_cps_noq4_path},
    {"metric": "final_no_q4_cps_exists", "value": bool(os.path.exists(final_cps_noq4_path))},
    {"metric": "final_no_q4_protocol_path", "value": final_protocol_noq4_path},
    {"metric": "final_no_q4_protocol_exists", "value": bool(os.path.exists(final_protocol_noq4_path))},
    {"metric": "role_groups_n", "value": int(len(role_groups))},
    {"metric": "TEST_real_values_used_for_privacy_reference", "value": True},
    {"metric": "synthetic_values_mutated", "value": False},
    {"metric": "privacy_decision_done_here", "value": False},
]

for role_name, cols in role_groups.items():
    input_audit_rows.append({
        "metric": f"role_group_cols::{role_name}",
        "value": int(len(cols)),
    })

input_audit_df = pd.DataFrame(input_audit_rows)
input_audit_df.to_csv(input_audit_csv, index=False)

# ----------------------------------------------------------
# 10) Contract / manifest
# ----------------------------------------------------------
role_manifest = {
    role_name: {
        "cols": cols,
        "cols_n": int(len(cols)),
    }
    for role_name, cols in role_groups.items()
}

final_cps_sha = _sha256_file_150(final_cps_noq4_path)
final_protocol_sha = _sha256_file_150(final_protocol_noq4_path)

contract = {
    "cell": "15.0",
    "version": CELL150_VERSION,
    "role": "q6_privacy_input_preparation_by_role_group_no_q4_promotion",
    "quality_dimension": "Q6_privacy_no_copy_release_safety",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_coupled_artifacts_required": False,
        "q4_positive_claim_allowed": False,
        "cell14_10_contract_version": version1410_150,
        "cell14_11_contract_version": version1411_150,
        "cell14_12_freshness_decision": freshness.get("decision", ""),
        "cell14_12_decision_severity": freshness.get("decision_severity", ""),
        "cell15_0_q4_freshness_governance": CELL15_0_Q4_FRESHNESS_GOVERNANCE,
    },
    "privacy_input_artifacts": {
        "final_no_q4_cps_path": final_cps_noq4_path,
        "final_no_q4_cps_sha256": final_cps_sha,
        "final_no_q4_protocol_path": final_protocol_noq4_path,
        "final_no_q4_protocol_sha256": final_protocol_sha,
        "source_protocol_component": "PROTOCOL_SYN_TEST_130",
        "source_iot_component": "IOT_FULL_SYN_TEST_130",
    },
    "split_shapes": {
        "real_train": [int(REAL_TRAIN_REF_150.shape[0]), int(REAL_TRAIN_REF_150.shape[1])],
        "real_val": [int(REAL_VAL_REF_150.shape[0]), int(REAL_VAL_REF_150.shape[1])],
        "real_test": [int(REAL_TEST_REF_150.shape[0]), int(REAL_TEST_REF_150.shape[1])],
        "synthetic_test": [int(SYN_TEST_REF_150.shape[0]), int(SYN_TEST_REF_150.shape[1])],
    },
    "role_group_counts": {k: int(len(v)) for k, v in role_groups.items()},
    "role_registry": registry_df.to_dict("records"),
    "empty_core_groups": empty_core,
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_for_privacy_reference": True,
        "TEST_real_values_used_for_model_fitting": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "privacy_decision_done_here": False,
        "input_preparation_done_here": True,
    },
    "next_cells": {
        "15.1": "no-copy window audit",
        "15.2": "DCR / NNDR role-specific privacy",
        "15.3": "MIA audit random split + temporal split",
        "15.4": "Q6 privacy summary and release-safety manifest",
    },
    "outputs": {
        "final_protocol_noq4_path": final_protocol_noq4_path,
        "final_cps_noq4_path": final_cps_noq4_path,
        "role_registry_csv": role_registry_csv,
        "input_audit_csv": input_audit_csv,
        "column_exclusion_audit_csv": column_exclusion_audit_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "role_manifest_json": role_manifest_json,
    },
}

_write_json_150(contract_json, contract)
_write_json_150(contract_canonical_json, contract)

manifest = {
    "cell": "15.0",
    "version": CELL150_VERSION,
    "created_outputs": contract["outputs"],
    "role_groups": role_manifest,
    "strict_contract": contract["strict_contract"],
    "privacy_input_artifacts": contract["privacy_input_artifacts"],
    "q4_governance": contract["q4_governance"],
}
_write_json_150(role_manifest_json, manifest)

hashes = {
    "final_protocol_noq4_sha256": _sha256_file_150(final_protocol_noq4_path),
    "final_cps_noq4_sha256": _sha256_file_150(final_cps_noq4_path),
    "role_registry_csv_sha256": _sha256_file_150(role_registry_csv),
    "input_audit_csv_sha256": _sha256_file_150(input_audit_csv),
    "column_exclusion_audit_csv_sha256": _sha256_file_150(column_exclusion_audit_csv),
    "contract_json_sha256": _sha256_file_150(contract_json),
    "contract_canonical_json_sha256": _sha256_file_150(contract_canonical_json),
    "role_manifest_json_sha256": _sha256_file_150(role_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_150(contract_json, contract)
_write_json_150(contract_canonical_json, contract)
_write_json_150(role_manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL150_VERSION"] = CELL150_VERSION
globals()["PROTOCOL_SYN_TEST_FINAL_NO_Q4"] = protocol_noq4
globals()["CPS_SYNTHETIC_TEST_FINAL_NO_Q4"] = CPS_SYNTHETIC_TEST_FINAL_NO_Q4
globals()["CPS_SYNTHETIC_TEST_Q6_INPUT"] = CPS_SYNTHETIC_TEST_FINAL_NO_Q4
globals()["REAL_TRAIN_REF_150"] = REAL_TRAIN_REF_150
globals()["REAL_VAL_REF_150"] = REAL_VAL_REF_150
globals()["REAL_TEST_REF_150"] = REAL_TEST_REF_150
globals()["SYN_TEST_REF_150"] = SYN_TEST_REF_150

globals()["CELL15_0_PRIVACY_ROLE_GROUPS"] = role_groups
globals()["CELL15_0_PRIVACY_ROLE_GROUP_REGISTRY_DF"] = registry_df
globals()["CELL15_0_PRIVACY_INPUT_AUDIT_DF"] = input_audit_df
globals()["CELL15_0_PRIVACY_COLUMN_EXCLUSION_AUDIT_DF"] = column_exclusion_df
globals()["CELL15_0_PRIVACY_INPUT_CONTRACT"] = contract

globals()["CELL15_0_FINAL_PROTOCOL_NO_Q4_PATH"] = final_protocol_noq4_path
globals()["CELL15_0_FINAL_CPS_NO_Q4_PATH"] = final_cps_noq4_path
globals()["CELL15_0_PRIVACY_ROLE_GROUP_REGISTRY_CSV"] = role_registry_csv
globals()["CELL15_0_PRIVACY_INPUT_AUDIT_CSV"] = input_audit_csv
globals()["CELL15_0_PRIVACY_COLUMN_EXCLUSION_AUDIT_CSV"] = column_exclusion_audit_csv
globals()["CELL15_0_PRIVACY_INPUT_CONTRACT_JSON"] = contract_json
globals()["CELL15_0_PRIVACY_INPUT_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL15_0_PRIVACY_ROLE_MANIFEST_JSON"] = role_manifest_json

log(
    "[Cell15.0] Q6 privacy input preparation complete | "
    f"q4_status=blocked_no_promotion | "
    f"role_groups={len(role_groups)} | "
    f"synthetic_shape={SYN_TEST_REF_150.shape} | "
    f"real_train_shape={REAL_TRAIN_REF_150.shape} | "
    f"real_val_shape={REAL_VAL_REF_150.shape} | "
    f"real_test_shape={REAL_TEST_REF_150.shape}"
)
log(
    "[Cell15.0] Role group counts | "
    f"{ {k: int(len(v)) for k, v in role_groups.items()} }"
)
log(f"[Cell15.0] Saved no-Q4 protocol: {final_protocol_noq4_path}")
log(f"[Cell15.0] Saved no-Q4 CPS: {final_cps_noq4_path}")
log(f"[Cell15.0] Saved role registry: {role_registry_csv} | rows={len(registry_df)}")
log(f"[Cell15.0] Saved input audit: {input_audit_csv}")
log(f"[Cell15.0] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell15.0] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_for_privacy_reference=True | "
    "TEST_real_values_used_for_model_fitting=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "privacy_decision_done_here=False | "
    "input_preparation_done_here=True"
)
log("--- END: Cell 15.0 - Q6 privacy input preparation by role group (v1.1 no-Q4-promotion strict) ---")

gc.collect()