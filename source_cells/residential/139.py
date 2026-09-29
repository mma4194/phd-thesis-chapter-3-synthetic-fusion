# ==========================================================
# CELL 15.6 - Q6 mitigated public-release candidate builder
# v1.1 STUDY-THESIS strict role-restricted public candidate, no-Q4-source aware
#
# Role:
#   - Build a separate Q6-mitigated public-release candidate.
#   - Does NOT overwrite the scientific no-Q4 synthetic artifact.
#   - Uses Cell 15.5 mitigation plan.
#   - Conservative first public candidate:
#       KEEP:
#         time columns
#         protocol_router
#         protocol_zigbee
#         iot_event_drivers
#       DROP:
#         protocol_ota
#         iot_binary_states
#         iot_continuous_values
#         iot_observability_masks
#         iot_placeholders_or_excluded
#         full_cps_mixed_sampled is not a namespace; it is audit-only
#
# Scientific contract:
#   - No fitting.
#   - No generation.
#   - No Q4 artifact promotion.
#   - No values changed inside kept columns.
#   - Only column-level restriction is applied.
#   - TEST real values are not read here.
#   - Public candidate must be re-audited by rerunning Cells 15.1–15.4
#     against the restricted public candidate input.
#
# Outputs:
#   synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet
#   synthetic/PROTOCOL_SYN_TEST_PUBLIC_Q6_MITIGATED.parquet
#   synthetic/IOT_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet
#
#   reports/cell15_6_q6_public_mitigation_audit.csv
#   reports/cell15_6_q6_public_column_registry.csv
#   reports/cell15_6_q6_public_dropped_columns.csv
#   reports/cell15_6_q6_public_release_reaudit_instructions.txt
#   reports/cell15_6_q6_public_mitigation_contract.json
#   artifacts/contracts/cell15_6_q6_public_mitigation_contract_v1_1_THESIS.json
#   artifacts/cell15_6_q6_public_mitigation_manifest.json
# ==========================================================

log("--- START: Cell 15.6 - Q6 mitigated public-release candidate builder (v1.1 no-Q4-source strict) ---")

import os
import gc
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_156 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL15_0_PRIVACY_ROLE_GROUPS",
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF",
    "CELL15_4_Q6_RELEASE_SAFETY_CONTRACT",
    "CELL15_5_Q6_MITIGATION_PLAN_DF",
    "CELL15_5_Q6_MITIGATION_CONTRACT",
]
_missing_156 = [k for k in _required_156 if k not in globals()]
if _missing_156:
    raise RuntimeError(f"[Cell15.6] Missing required globals: {_missing_156}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_156 = str(OUT_SYN)
REPORT_DIR_ACTIVE_156 = str(REPORT_DIR)

def _resolve_project_root_156(outdir, report_dir, out_syn):
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
    raise RuntimeError("[Cell15.6] Could not resolve canonical project root.")

PROJECT_ROOT_156 = _resolve_project_root_156(OUTDIR, REPORT_DIR_ACTIVE_156, OUT_SYN_ACTIVE_156)
OUT_SYN = os.path.join(PROJECT_ROOT_156, "synthetic")
REPORT_DIR = os.path.join(PROJECT_ROOT_156, "reports")
ARTDIR = os.path.join(PROJECT_ROOT_156, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
CELL156_VERSION = "cell15_6_q6_mitigated_public_release_candidate_v2_0_corrected_driver_release_no_q4_source"

CFG["cell15_6_version"] = CELL156_VERSION
CFG["cell15_6_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_6_Q4_coupled_artifacts_used"] = False
CFG["cell15_6_TEST_real_values_used"] = False
CFG["cell15_6_TEST_real_values_used_for_materialization"] = False
CFG["cell15_6_synthetic_values_mutated"] = False
CFG["cell15_6_column_restriction_applied"] = True
CFG["cell15_6_selection_done_here"] = False
CFG["cell15_6_generator_fit_done_here"] = False
CFG["cell15_6_materialization_done_here"] = False
CFG["cell15_6_public_release_candidate_created"] = True
CFG["cell15_6_overwrites_scientific_no_q4_artifact"] = False

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell15_6_public_policy", "role_restricted_release_v1")
CFG.setdefault("cell15_6_keep_roles", [
    "protocol_router",
    "protocol_zigbee",
    "iot_event_drivers",
])
CFG.setdefault("cell15_6_drop_roles", [
    "protocol_ota",
    "iot_binary_states",
    "iot_continuous_values",
    "iot_observability_masks",
    "iot_placeholders_or_excluded",
])
CFG.setdefault("cell15_6_include_time_columns", True)
CFG.setdefault("cell15_6_fail_if_kept_column_missing", True)
CFG.setdefault("cell15_6_fail_if_dropped_high_risk_column_remains", True)
CFG.setdefault("cell15_6_fail_if_kept_values_drift", True)
CFG.setdefault("cell15_6_min_public_cols", 1)

KEEP_ROLES_156 = list(map(str, CFG.get("cell15_6_keep_roles", [])))
DROP_ROLES_156 = list(map(str, CFG.get("cell15_6_drop_roles", [])))

# Router retention is not allowed by role name alone. If protocol_router is kept,
# a predeclared Q6.ROUTER.RELEASE_POLICY.V1 contract must already have downgraded
# router from blocker to warning-governed eligibility without mutating synthetic values.
ROUTER_RELEASE_POLICY_156 = globals().get("CELL_Q6_ROUTER_RELEASE_POLICY_CONTRACT", {})
if "protocol_router" in set(KEEP_ROLES_156):
    if not isinstance(ROUTER_RELEASE_POLICY_156, dict) or not bool(ROUTER_RELEASE_POLICY_156.get("router_release_eligible", False)):
        raise RuntimeError(
            "[Cell15.6] protocol_router is in keep roles but Q6.ROUTER.RELEASE_POLICY.V1 did not pass. "
            "Do not build a corrected public baseline by bypassing Q6."
        )
    strict_router_156 = ROUTER_RELEASE_POLICY_156.get("strict_contract", {})
    if bool(strict_router_156.get("synthetic_values_mutated", True)):
        raise RuntimeError("[Cell15.6] Router release policy mutated synthetic values; invalid for Q6 public candidate.")
    if not bool(strict_router_156.get("direct_identifier_or_hard_privacy_blocker_remains_fail_closed", False)):
        raise RuntimeError("[Cell15.6] Router release policy is missing hard-privacy fail-closed assertion.")


# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
public_cps_path = os.path.join(OUT_SYN, "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet")
public_protocol_path = os.path.join(OUT_SYN, "PROTOCOL_SYN_TEST_PUBLIC_Q6_MITIGATED.parquet")
public_iot_path = os.path.join(OUT_SYN, "IOT_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet")

mitigation_audit_csv = os.path.join(REPORT_DIR, "cell15_6_q6_public_mitigation_audit.csv")
public_column_registry_csv = os.path.join(REPORT_DIR, "cell15_6_q6_public_column_registry.csv")
dropped_columns_csv = os.path.join(REPORT_DIR, "cell15_6_q6_public_dropped_columns.csv")
reaudit_instructions_txt = os.path.join(REPORT_DIR, "cell15_6_q6_public_release_reaudit_instructions.txt")
contract_json = os.path.join(REPORT_DIR, "cell15_6_q6_public_mitigation_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell15_6_q6_public_mitigation_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell15_6_q6_public_mitigation_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_156(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_156(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_156(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_156(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_156(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_156(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_156(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_156(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_156(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_156(payload), f, indent=2, sort_keys=True)

def _sha256_file_156(path: str) -> str:
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_156(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _require_contract_version_156(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.6] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.6] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

def _dedup_156(seq):
    seen = set()
    out = []
    for x in seq:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _is_time_col_156(c: str) -> bool:
    s = str(c).lower()
    return (
        s in {"sec", "sec_epoch_s__canon", "timestamp", "time", "datetime"}
        or s.endswith("__time")
        or "time_iso" in s
    )

def _protocol_tier_156(c: str) -> str:
    s = str(c)
    if s.startswith("router__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__"):
        return "ota"
    if s.startswith("zigbee__") or s.startswith("zb__"):
        return "zigbee"
    if s.startswith("zwave__"):
        return "zwave"
    return ""

def _is_protocol_col_156(c: str) -> bool:
    return _protocol_tier_156(c) in {"router", "ota", "zigbee", "zwave"}

def _is_iot_col_156(c: str) -> bool:
    s = str(c)
    return s.startswith("iot__") or s.startswith("events_in_sec__")

def _role_for_col_156(c: str, role_groups: dict) -> str:
    c = str(c)
    matches = []
    for role, cols in role_groups.items():
        if c in set(map(str, cols)):
            matches.append(str(role))
    if matches:
        non_mixed = [m for m in matches if m != "full_cps_mixed_sampled"]
        return non_mixed[0] if non_mixed else matches[0]
    if _is_time_col_156(c):
        return "time"
    if _is_protocol_col_156(c):
        return f"protocol_{_protocol_tier_156(c)}"
    if _is_iot_col_156(c):
        return "iot_unmapped"
    return "other"

def _series_equal_156(a, b) -> bool:
    xa = pd.Series(a)
    xb = pd.Series(b)
    if xa.dtype == object or xb.dtype == object:
        return xa.astype(str).equals(xb.astype(str))
    va = pd.to_numeric(xa, errors="coerce").to_numpy(dtype=np.float64)
    vb = pd.to_numeric(xb, errors="coerce").to_numpy(dtype=np.float64)
    same = (np.isnan(va) & np.isnan(vb)) | np.isclose(va, vb, rtol=1e-8, atol=1e-10, equal_nan=True)
    return bool(np.all(same))

def _finite_rate_156(frame: pd.DataFrame, col: str):
    if col not in frame.columns:
        return np.nan
    x = pd.to_numeric(frame[col], errors="coerce")
    return float(x.notna().mean()) if len(x) else np.nan

# ----------------------------------------------------------
# 4) Validate upstream no-Q4 and Q6 mitigation contracts
# ----------------------------------------------------------
version150_156 = _require_contract_version_156(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
version154_156 = _require_contract_version_156(
    CELL15_4_Q6_RELEASE_SAFETY_CONTRACT,
    "CELL15_4_Q6_RELEASE_SAFETY_CONTRACT",
    "cell15_4_q6_privacy_release_safety_summary_v1_2",
)
version155_156 = _require_contract_version_156(
    CELL15_5_Q6_MITIGATION_CONTRACT,
    "CELL15_5_Q6_MITIGATION_CONTRACT",
    "cell15_5_q6_mitigation_planning_audit_v1_1",
)

strict150 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
strict154 = CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("strict_contract", {})
strict155 = CELL15_5_Q6_MITIGATION_CONTRACT.get("strict_contract", {})

for name, strict in [("Cell15.0", strict150), ("Cell15.4", strict154), ("Cell15.5", strict155)]:
    if str(strict.get("Q4_final_status", "")) != "blocked_no_promotion":
        raise RuntimeError(f"[Cell15.6] {name} does not carry Q4_final_status=blocked_no_promotion.")
    if bool(strict.get("Q4_coupled_artifacts_used", True)):
        raise RuntimeError(f"[Cell15.6] {name} used Q4-coupled artifacts unexpectedly.")
    if bool(strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell15.6] {name} indicates synthetic mutation.")

if str(CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("overall", {}).get("q6_overall_status", "")) != "release_blocker":
    raise RuntimeError("[Cell15.6] Cell 15.4 did not report release_blocker; mitigation candidate may be unnecessary.")
if not bool(CELL15_5_Q6_MITIGATION_CONTRACT.get("plan_scope", {}).get("plan_only", False)):
    raise RuntimeError("[Cell15.6] Cell 15.5 did not declare planning-only mitigation.")

# ----------------------------------------------------------
# 5) Load source no-Q4 synthetic artifact
# ----------------------------------------------------------
source_cps_path = str(globals().get("CELL15_0_FINAL_CPS_NO_Q4_PATH", ""))
source_protocol_path = str(globals().get("CELL15_0_FINAL_PROTOCOL_NO_Q4_PATH", ""))

if isinstance(globals().get("CPS_SYNTHETIC_TEST_FINAL_NO_Q4", None), pd.DataFrame):
    CPS_SOURCE_NO_Q4_156 = globals()["CPS_SYNTHETIC_TEST_FINAL_NO_Q4"].copy()
    if not source_cps_path:
        source_cps_path = str(CELL15_0_PRIVACY_INPUT_CONTRACT.get("privacy_input_artifacts", {}).get("final_no_q4_cps_path", ""))
elif _exists_156(source_cps_path):
    CPS_SOURCE_NO_Q4_156 = pd.read_parquet(source_cps_path)
else:
    source_cps_path = str(CELL15_0_PRIVACY_INPUT_CONTRACT.get("privacy_input_artifacts", {}).get("final_no_q4_cps_path", ""))
    if _exists_156(source_cps_path):
        CPS_SOURCE_NO_Q4_156 = pd.read_parquet(source_cps_path)
    else:
        raise RuntimeError(f"[Cell15.6] Missing source no-Q4 CPS artifact: {source_cps_path}")

CPS_SOURCE_NO_Q4_156.columns = CPS_SOURCE_NO_Q4_156.columns.astype(str)
N_ROWS_156 = int(len(CPS_SOURCE_NO_Q4_156))

if not source_protocol_path:
    source_protocol_path = str(CELL15_0_PRIVACY_INPUT_CONTRACT.get("privacy_input_artifacts", {}).get("final_no_q4_protocol_path", ""))

log(
    "[Cell15.6] Loaded no-Q4 scientific/privacy input artifact | "
    f"path={source_cps_path} | shape={CPS_SOURCE_NO_Q4_156.shape}"
)

role_groups = CELL15_0_PRIVACY_ROLE_GROUPS

# ----------------------------------------------------------
# 6) Determine keep/drop columns
# ----------------------------------------------------------
if bool(CFG.get("cell15_6_include_time_columns", True)):
    _time_allowlist_156 = list(map(str, CFG.get("cell15_6_time_column_allowlist", [])))
    if _time_allowlist_156:
        time_cols = [c for c in _time_allowlist_156 if c in CPS_SOURCE_NO_Q4_156.columns]
        missing_time_allowlist_156 = sorted(set(_time_allowlist_156) - set(time_cols))
        if missing_time_allowlist_156:
            raise RuntimeError(
                "[Cell15.6] Configured time-column allowlist not present in source artifact: "
                f"{missing_time_allowlist_156}"
            )
    else:
        time_cols = [c for c in CPS_SOURCE_NO_Q4_156.columns if _is_time_col_156(c)]
else:
    time_cols = []

keep_cols_by_role = {}
drop_cols_by_role = {}

for role in KEEP_ROLES_156:
    cols = [c for c in map(str, role_groups.get(role, [])) if c in CPS_SOURCE_NO_Q4_156.columns]
    keep_cols_by_role[role] = _dedup_156(cols)

# Corrected sparse-driver release eligibility is a hard Q6 gate. Do not keep
# corrected-fatal sparse drivers merely because their role is iot_event_drivers.
driver_release_ledger_156 = globals().get("CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF", pd.DataFrame())
if (not isinstance(driver_release_ledger_156, pd.DataFrame) or driver_release_ledger_156.empty):
    candidate_driver_ledger_csv_156 = str(globals().get("CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_CSV", ""))
    if not candidate_driver_ledger_csv_156:
        candidate_driver_ledger_csv_156 = os.path.join(REPORT_DIR, "cell12e6_driver_corrected_release_ledger.csv")
    if os.path.exists(candidate_driver_ledger_csv_156):
        driver_release_ledger_156 = pd.read_csv(candidate_driver_ledger_csv_156)
    else:
        driver_release_ledger_156 = pd.DataFrame()

driver_release_eligible_cols_156 = []
driver_release_excluded_cols_156 = []
if "iot_event_drivers" in set(KEEP_ROLES_156):
    if not isinstance(driver_release_ledger_156, pd.DataFrame) or driver_release_ledger_156.empty:
        raise RuntimeError(
            "[Cell15.6] Corrected sparse-driver release ledger is missing. "
            "Run Cell 12.e.6R before building the Q6 public artifact."
        )
    if "col" not in driver_release_ledger_156.columns or "driver_release_eligible_corrected" not in driver_release_ledger_156.columns:
        raise RuntimeError(
            "[Cell15.6] Corrected sparse-driver release ledger must contain 'col' and 'driver_release_eligible_corrected'."
        )
    tmp_driver_156 = driver_release_ledger_156.copy()
    tmp_driver_156["col"] = tmp_driver_156["col"].astype(str)
    tmp_driver_156["driver_release_eligible_corrected"] = tmp_driver_156["driver_release_eligible_corrected"].fillna(False).astype(bool)
    driver_release_eligible_cols_156 = sorted(tmp_driver_156.loc[tmp_driver_156["driver_release_eligible_corrected"], "col"].tolist())
    driver_release_excluded_cols_156 = sorted(tmp_driver_156.loc[~tmp_driver_156["driver_release_eligible_corrected"], "col"].tolist())
    requested_driver_cols_156 = list(keep_cols_by_role.get("iot_event_drivers", []))
    requested_driver_set_156 = set(requested_driver_cols_156)
    missing_eligible_156 = sorted(set(driver_release_eligible_cols_156) - set(CPS_SOURCE_NO_Q4_156.columns))
    if missing_eligible_156:
        raise RuntimeError(f"[Cell15.6] Corrected release-eligible driver columns missing from source artifact: {missing_eligible_156[:10]}")
    unexpected_eligible_156 = sorted(set(driver_release_eligible_cols_156) - requested_driver_set_156)
    if unexpected_eligible_156:
        raise RuntimeError(f"[Cell15.6] Corrected release ledger names drivers outside the driver role group: {unexpected_eligible_156[:10]}")
    keep_cols_by_role["iot_event_drivers"] = [c for c in requested_driver_cols_156 if c in set(driver_release_eligible_cols_156)]
    if set(keep_cols_by_role["iot_event_drivers"]) & set(driver_release_excluded_cols_156):
        raise RuntimeError("[Cell15.6] Corrected-fatal sparse-driver columns remain in the Q6 keep set.")
    expected_driver_public_156 = int(CFG.get("expected_public_sparse_driver_cols", len(driver_release_eligible_cols_156)))
    if len(keep_cols_by_role["iot_event_drivers"]) != expected_driver_public_156:
        raise RuntimeError(
            "[Cell15.6] Corrected sparse-driver public count mismatch: "
            f"got={len(keep_cols_by_role['iot_event_drivers'])} expected={expected_driver_public_156}"
        )

for role in DROP_ROLES_156:
    cols = [c for c in map(str, role_groups.get(role, [])) if c in CPS_SOURCE_NO_Q4_156.columns]
    drop_cols_by_role[role] = _dedup_156(cols)

keep_cols = _dedup_156(time_cols + [c for role in KEEP_ROLES_156 for c in keep_cols_by_role.get(role, [])])
drop_cols = _dedup_156([c for role in DROP_ROLES_156 for c in drop_cols_by_role.get(role, [])])

conflict_cols = sorted((set(keep_cols) & set(drop_cols)) - set(time_cols))
if conflict_cols:
    raise RuntimeError(
        "[Cell15.6] Columns are both keep and drop outside time columns. "
        f"sample={conflict_cols[:30]}"
    )

keep_cols = [c for c in keep_cols if c not in set(drop_cols) or c in set(time_cols)]

missing_keep_by_role = {}
for role in KEEP_ROLES_156:
    requested = list(map(str, role_groups.get(role, [])))
    missing = sorted([c for c in requested if c not in CPS_SOURCE_NO_Q4_156.columns])
    if missing:
        missing_keep_by_role[role] = missing

if missing_keep_by_role and bool(CFG.get("cell15_6_fail_if_kept_column_missing", True)):
    raise RuntimeError(
        "[Cell15.6] Some requested kept columns are missing from no-Q4 source artifact: "
        f"{ {k: v[:10] for k, v in missing_keep_by_role.items()} }"
    )

if len(keep_cols) < int(CFG.get("cell15_6_min_public_cols", 1)):
    raise RuntimeError(f"[Cell15.6] Public candidate would have too few columns: keep_cols={len(keep_cols)}")

high_risk_remaining = sorted([c for c in drop_cols if c in keep_cols])
if high_risk_remaining and bool(CFG.get("cell15_6_fail_if_dropped_high_risk_column_remains", True)):
    raise RuntimeError(
        "[Cell15.6] High-risk dropped columns remain in public candidate. "
        f"sample={high_risk_remaining[:30]}"
    )

# ----------------------------------------------------------
# 7) Build public role-restricted candidate
# ----------------------------------------------------------
CPS_PUBLIC_Q6_MITIGATED_156 = CPS_SOURCE_NO_Q4_156[keep_cols].copy()

expected_public_cols_156 = CFG.get("cell15_6_expected_public_cols", None)
if expected_public_cols_156 is not None:
    expected_public_cols_156 = int(expected_public_cols_156)
    if CPS_PUBLIC_Q6_MITIGATED_156.shape[1] != expected_public_cols_156:
        raise RuntimeError(
            "[Cell15.6] Public candidate logical column count does not match the precommitted release policy: "
            f"got={CPS_PUBLIC_Q6_MITIGATED_156.shape[1]} expected={expected_public_cols_156}. "
            "Do not patch the count post hoc; fix role eligibility or update the paper."
        )

protocol_public_cols = [c for c in keep_cols if _is_protocol_col_156(c)]
iot_public_cols = [c for c in keep_cols if _is_iot_col_156(c)]

PROTOCOL_PUBLIC_Q6_MITIGATED_156 = CPS_SOURCE_NO_Q4_156[protocol_public_cols].copy() if protocol_public_cols else pd.DataFrame(index=CPS_SOURCE_NO_Q4_156.index)
IOT_PUBLIC_Q6_MITIGATED_156 = CPS_SOURCE_NO_Q4_156[iot_public_cols].copy() if iot_public_cols else pd.DataFrame(index=CPS_SOURCE_NO_Q4_156.index)

# ----------------------------------------------------------
# 8) Safety checks: kept values must not drift
# ----------------------------------------------------------
value_drift_rows = []

if bool(CFG.get("cell15_6_fail_if_kept_values_drift", True)):
    for c in keep_cols:
        if c not in CPS_SOURCE_NO_Q4_156.columns or c not in CPS_PUBLIC_Q6_MITIGATED_156.columns:
            value_drift_rows.append({"col": c, "drift_detected": True, "reason": "missing_after_restriction"})
            continue
        same = _series_equal_156(CPS_SOURCE_NO_Q4_156[c], CPS_PUBLIC_Q6_MITIGATED_156[c])
        if not same:
            value_drift_rows.append({"col": c, "drift_detected": True, "reason": "values_changed"})

if value_drift_rows and bool(CFG.get("cell15_6_fail_if_kept_values_drift", True)):
    raise RuntimeError(
        "[Cell15.6] Kept column value drift detected. "
        f"sample={value_drift_rows[:10]}"
    )

# ----------------------------------------------------------
# 9) Build audits / registries
# ----------------------------------------------------------
registry_rows = []
for ordinal, c in enumerate(keep_cols):
    role = _role_for_col_156(c, role_groups)
    registry_rows.append({
        "col": c,
        "public_ordinal": int(ordinal),
        "role_group": role,
        "is_time": bool(_is_time_col_156(c)),
        "is_protocol": bool(_is_protocol_col_156(c)),
        "protocol_tier": _protocol_tier_156(c),
        "is_iot": bool(_is_iot_col_156(c)),
        "source_artifact": source_cps_path,
        "value_mutated": False,
        "finite_rate": _finite_rate_156(CPS_PUBLIC_Q6_MITIGATED_156, c),
        "public_release_candidate": True,
        "driver_release_eligible_corrected": bool(c in set(driver_release_eligible_cols_156)) if role == "iot_event_drivers" else np.nan,
        "driver_release_exclusion_reason": "" if c not in set(driver_release_excluded_cols_156) else "corrected_sparse_driver_fatal_release_excluded",
    })

public_registry_df = pd.DataFrame(registry_rows)

dropped_rows = []
for c in sorted(set(CPS_SOURCE_NO_Q4_156.columns) - set(keep_cols)):
    role = _role_for_col_156(c, role_groups)
    if role in DROP_ROLES_156:
        drop_reason = "dropped_by_q6_role_restriction_policy"
    elif role == "full_cps_mixed_sampled":
        drop_reason = "mixed_audit_role_not_a_physical_namespace"
    elif c in set(driver_release_excluded_cols_156):
        drop_reason = "corrected_sparse_driver_fatal_release_excluded"
    elif role == "iot_unmapped":
        drop_reason = "unmapped_iot_column_not_in_public_keep_roles"
    elif role == "other":
        drop_reason = "non_selected_other_column"
    else:
        drop_reason = "not_in_public_keep_roles"
    dropped_rows.append({
        "col": c,
        "role_group": role,
        "drop_reason": drop_reason,
        "was_high_risk_role": bool(role in DROP_ROLES_156),
        "is_time": bool(_is_time_col_156(c)),
        "is_protocol": bool(_is_protocol_col_156(c)),
        "protocol_tier": _protocol_tier_156(c),
        "is_iot": bool(_is_iot_col_156(c)),
        "value_mutated": False,
        "public_release_candidate": False,
    })

dropped_df = pd.DataFrame(dropped_rows)

audit_rows = [
    {"metric": "q4_final_status", "value": "blocked_no_promotion"},
    {"metric": "q4_coupled_artifacts_used", "value": False},
    {"metric": "source_no_q4_cps_path", "value": source_cps_path},
    {"metric": "source_no_q4_rows", "value": int(CPS_SOURCE_NO_Q4_156.shape[0])},
    {"metric": "source_no_q4_cols", "value": int(CPS_SOURCE_NO_Q4_156.shape[1])},
    {"metric": "public_cps_rows", "value": int(CPS_PUBLIC_Q6_MITIGATED_156.shape[0])},
    {"metric": "public_cps_cols", "value": int(CPS_PUBLIC_Q6_MITIGATED_156.shape[1])},
    {"metric": "public_protocol_cols", "value": int(PROTOCOL_PUBLIC_Q6_MITIGATED_156.shape[1])},
    {"metric": "public_iot_cols", "value": int(IOT_PUBLIC_Q6_MITIGATED_156.shape[1])},
    {"metric": "kept_roles", "value": "|".join(KEEP_ROLES_156)},
    {"metric": "dropped_roles", "value": "|".join(DROP_ROLES_156)},
    {"metric": "dropped_cols_total", "value": int(len(dropped_df))},
    {"metric": "value_drift_detected_n", "value": int(len(value_drift_rows))},
    {"metric": "synthetic_values_mutated", "value": False},
    {"metric": "column_restriction_applied", "value": True},
    {"metric": "overwrites_scientific_no_q4_artifact", "value": False},
    {"metric": "requires_q6_reaudit", "value": True},
    {"metric": "corrected_sparse_driver_release_eligible_n", "value": int(len(driver_release_eligible_cols_156))},
    {"metric": "corrected_sparse_driver_release_excluded_fatal_n", "value": int(len(driver_release_excluded_cols_156))},
]

for role in KEEP_ROLES_156:
    audit_rows.append({"metric": f"kept_cols::{role}", "value": int(len(keep_cols_by_role.get(role, [])))})
for role in DROP_ROLES_156:
    audit_rows.append({"metric": f"dropped_cols::{role}", "value": int(len(drop_cols_by_role.get(role, [])))})

mitigation_audit_df = pd.DataFrame(audit_rows)

# ----------------------------------------------------------
# 10) Save public candidates
# ----------------------------------------------------------
CPS_PUBLIC_Q6_MITIGATED_156.to_parquet(public_cps_path, index=True)
PROTOCOL_PUBLIC_Q6_MITIGATED_156.to_parquet(public_protocol_path, index=True)
IOT_PUBLIC_Q6_MITIGATED_156.to_parquet(public_iot_path, index=True)

mitigation_audit_df.to_csv(mitigation_audit_csv, index=False)
public_registry_df.to_csv(public_column_registry_csv, index=False)
dropped_df.to_csv(dropped_columns_csv, index=False)

# ----------------------------------------------------------
# 11) Re-audit instructions
# ----------------------------------------------------------
reaudit_instructions = f"""Q6 Public Candidate Re-Audit Instructions
========================================

Public Q6-mitigated candidate created:
- CPS:      {public_cps_path}
- Protocol: {public_protocol_path}
- IoT:      {public_iot_path}

Policy:
- {CFG.get("cell15_6_public_policy")}

Q4 governance:
- final_q4_status: blocked_no_promotion
- q4_coupled_artifacts_used: False

Kept roles:
- {", ".join(KEEP_ROLES_156)}

Dropped roles:
- {", ".join(DROP_ROLES_156)}

Important:
This cell created a separate role-restricted public-release candidate.
It did not overwrite the no-Q4 scientific/privacy input:
- {source_cps_path}

Next required step:
Re-run Q6 privacy audits against the public candidate before claiming public release safety.

Recommended approach:
1. For public-candidate re-audit, set:
   SYN_TEST_REF_150 = pd.read_parquet("{public_cps_path}")
   CELL15_0_PRIVACY_ROLE_GROUPS = role groups rebuilt/restricted from the public candidate,
   or run a dedicated Cell 15.6a public-candidate privacy input cell.

2. Re-run:
   Cell 15.1
   Cell 15.1a if needed
   Cell 15.2
   Cell 15.2a if needed
   Cell 15.3
   Cell 15.4

3. Treat the public candidate as release-eligible only if Cell 15.4 no longer reports release_blocker.

Expected effect:
- protocol_ota blocker should disappear because OTA is removed.
- iot_binary_states blocker should disappear because binary states are removed.
- iot_continuous_values and iot_observability_masks distinguishability blockers should disappear because those roles are removed.
- protocol_router, protocol_zigbee, and event-driver roles should remain warning/pass-level if no new issue appears.

Do not claim global Q6 pass until the re-audit confirms it.
"""

with open(reaudit_instructions_txt, "w", encoding="utf-8") as f:
    f.write(reaudit_instructions)

# ----------------------------------------------------------
# 12) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "15.6",
    "version": CELL156_VERSION,
    "role": "q6_mitigated_public_release_candidate_builder_no_q4_source",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
    },
    "upstream_contract_versions": {
        "cell15_0": version150_156,
        "cell15_4": version154_156,
        "cell15_5": version155_156,
    },
    "policy": {
        "public_policy": str(CFG.get("cell15_6_public_policy")),
        "router_release_policy": ROUTER_RELEASE_POLICY_156 if isinstance(ROUTER_RELEASE_POLICY_156, dict) else {},
        "expected_public_cols": CFG.get("cell15_6_expected_public_cols", None),
        "time_column_allowlist": list(map(str, CFG.get("cell15_6_time_column_allowlist", []))),
        "mitigation_type": "role_restricted_column_release",
        "kept_roles": KEEP_ROLES_156,
        "dropped_roles": DROP_ROLES_156,
        "column_restriction_applied": True,
        "values_changed_inside_kept_columns": False,
        "no_q4_scientific_artifact_overwritten": False,
        "requires_q6_reaudit": True,
        "corrected_sparse_driver_release_eligible_n": int(len(driver_release_eligible_cols_156)),
        "corrected_sparse_driver_release_excluded_fatal_n": int(len(driver_release_excluded_cols_156)),
        "corrected_sparse_driver_release_eligible_cols": list(driver_release_eligible_cols_156),
        "corrected_sparse_driver_release_excluded_fatal_cols": list(driver_release_excluded_cols_156),
    },
    "source_artifacts": {
        "source_no_q4_cps_path": source_cps_path,
        "source_no_q4_protocol_path": source_protocol_path,
        "source_no_q4_cps_sha256": _sha256_file_156(source_cps_path),
        "source_no_q4_protocol_sha256": _sha256_file_156(source_protocol_path) if _exists_156(source_protocol_path) else "",
    },
    "output_shapes": {
        "public_cps": [int(CPS_PUBLIC_Q6_MITIGATED_156.shape[0]), int(CPS_PUBLIC_Q6_MITIGATED_156.shape[1])],
        "public_protocol": [int(PROTOCOL_PUBLIC_Q6_MITIGATED_156.shape[0]), int(PROTOCOL_PUBLIC_Q6_MITIGATED_156.shape[1])],
        "public_iot": [int(IOT_PUBLIC_Q6_MITIGATED_156.shape[0]), int(IOT_PUBLIC_Q6_MITIGATED_156.shape[1])],
    },
    "column_counts": {
        "source_cols": int(CPS_SOURCE_NO_Q4_156.shape[1]),
        "kept_cols": int(len(keep_cols)),
        "dropped_cols": int(len(dropped_df)),
        "time_cols": int(len(time_cols)),
        "protocol_public_cols": int(len(protocol_public_cols)),
        "iot_public_cols": int(len(iot_public_cols)),
        "value_drift_detected_n": int(len(value_drift_rows)),
        "corrected_sparse_driver_release_eligible_n": int(len(driver_release_eligible_cols_156)),
        "corrected_sparse_driver_release_excluded_fatal_n": int(len(driver_release_excluded_cols_156)),
    },
    "role_counts": {
        "kept": {role: int(len(keep_cols_by_role.get(role, []))) for role in KEEP_ROLES_156},
        "dropped": {role: int(len(drop_cols_by_role.get(role, []))) for role in DROP_ROLES_156},
        "corrected_sparse_drivers": {
            "release_eligible": int(len(driver_release_eligible_cols_156)),
            "release_excluded_fatal": int(len(driver_release_excluded_cols_156)),
        },
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "column_restriction_applied": True,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "redaction_or_suppression_applied": False,
        "public_release_candidate_created": True,
        "overwrites_scientific_no_q4_artifact": False,
        "requires_q6_reaudit": True,
    },
    "outputs": {
        "public_cps_path": public_cps_path,
        "public_protocol_path": public_protocol_path,
        "public_iot_path": public_iot_path,
        "mitigation_audit_csv": mitigation_audit_csv,
        "public_column_registry_csv": public_column_registry_csv,
        "dropped_columns_csv": dropped_columns_csv,
        "reaudit_instructions_txt": reaudit_instructions_txt,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_156(contract_json, contract)
_write_json_156(contract_canonical_json, contract)

manifest = {
    "cell": "15.6",
    "version": CELL156_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "policy": contract["policy"],
    "output_shapes": contract["output_shapes"],
    "column_counts": contract["column_counts"],
    "strict_contract": contract["strict_contract"],
}
_write_json_156(manifest_json, manifest)

hashes = {
    "public_cps_sha256": _sha256_file_156(public_cps_path),
    "public_protocol_sha256": _sha256_file_156(public_protocol_path),
    "public_iot_sha256": _sha256_file_156(public_iot_path),
    "mitigation_audit_csv_sha256": _sha256_file_156(mitigation_audit_csv),
    "public_column_registry_csv_sha256": _sha256_file_156(public_column_registry_csv),
    "dropped_columns_csv_sha256": _sha256_file_156(dropped_columns_csv),
    "reaudit_instructions_txt_sha256": _sha256_file_156(reaudit_instructions_txt),
    "contract_json_sha256": _sha256_file_156(contract_json),
    "contract_canonical_json_sha256": _sha256_file_156(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_156(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_156(contract_json, contract)
_write_json_156(contract_canonical_json, contract)
_write_json_156(manifest_json, manifest)

# ----------------------------------------------------------
# 13) Export globals
# ----------------------------------------------------------
globals()["CELL156_VERSION"] = CELL156_VERSION
globals()["CPS_PUBLIC_Q6_MITIGATED_156"] = CPS_PUBLIC_Q6_MITIGATED_156
globals()["PROTOCOL_PUBLIC_Q6_MITIGATED_156"] = PROTOCOL_PUBLIC_Q6_MITIGATED_156
globals()["IOT_PUBLIC_Q6_MITIGATED_156"] = IOT_PUBLIC_Q6_MITIGATED_156

globals()["CELL15_6_PUBLIC_CPS_PATH"] = public_cps_path
globals()["CELL15_6_PUBLIC_PROTOCOL_PATH"] = public_protocol_path
globals()["CELL15_6_PUBLIC_IOT_PATH"] = public_iot_path
globals()["CELL15_6_Q6_PUBLIC_MITIGATION_AUDIT_DF"] = mitigation_audit_df
globals()["CELL15_6_Q6_PUBLIC_COLUMN_REGISTRY_DF"] = public_registry_df
globals()["CELL15_6_Q6_PUBLIC_DROPPED_COLUMNS_DF"] = dropped_df
globals()["CELL15_6_Q6_PUBLIC_MITIGATION_CONTRACT"] = contract

globals()["CELL15_6_Q6_PUBLIC_MITIGATION_AUDIT_CSV"] = mitigation_audit_csv
globals()["CELL15_6_Q6_PUBLIC_COLUMN_REGISTRY_CSV"] = public_column_registry_csv
globals()["CELL15_6_Q6_PUBLIC_DROPPED_COLUMNS_CSV"] = dropped_columns_csv
globals()["CELL15_6_Q6_REAUDIT_INSTRUCTIONS_TXT"] = reaudit_instructions_txt
globals()["CELL15_6_Q6_PUBLIC_MITIGATION_CONTRACT_JSON"] = contract_json
globals()["CELL15_6_Q6_PUBLIC_MITIGATION_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL15_6_Q6_PUBLIC_MITIGATION_MANIFEST_JSON"] = manifest_json

log(
    "[Cell15.6] Q6 public candidate built | "
    f"source_shape={CPS_SOURCE_NO_Q4_156.shape} | "
    f"public_shape={CPS_PUBLIC_Q6_MITIGATED_156.shape} | "
    f"protocol_public_cols={len(protocol_public_cols)} | "
    f"iot_public_cols={len(iot_public_cols)} | "
    f"dropped_cols={len(dropped_df)}"
)
log(f"[Cell15.6] Kept roles | { {r: len(keep_cols_by_role.get(r, [])) for r in KEEP_ROLES_156} }")
log(f"[Cell15.6] Corrected sparse-driver Q6 policy | release_eligible={len(driver_release_eligible_cols_156)} | excluded_fatal={len(driver_release_excluded_cols_156)}")
log(f"[Cell15.6] Dropped roles | { {r: len(drop_cols_by_role.get(r, [])) for r in DROP_ROLES_156} }")
log(f"[Cell15.6] Saved public CPS candidate: {public_cps_path}")
log(f"[Cell15.6] Saved public protocol candidate: {public_protocol_path}")
log(f"[Cell15.6] Saved public IoT candidate: {public_iot_path}")
log(f"[Cell15.6] Saved re-audit instructions: {reaudit_instructions_txt}")
log(f"[Cell15.6] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell15.6] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "column_restriction_applied=True | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "public_release_candidate_created=True | "
    "overwrites_scientific_no_q4_artifact=False | "
    "requires_q6_reaudit=True"
)
log("--- END: Cell 15.6 - Q6 mitigated public-release candidate builder (v1.1 no-Q4-source strict) ---")

gc.collect()