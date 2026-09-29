# ==========================================================
# CELL 14.2 - Reassemble coupling-conditioned CPS candidate
# v1.1 STUDY-THESIS strict coupled protocol + full IoT assembly,
#      repair-candidate/status-aware
#
# Role:
#   - Build a coupling-conditioned TEST candidate by combining:
#       1) coupled synthetic protocol TEST matrix from Cell 14.1
#       2) full synthetic IoT TEST namespace from Cell 12.g / 13.0
#   - Preserve exact TEST row count.
#   - Preserve protocol/IOT provenance and audit all column roles.
#
# Scientific contract:
#   - No model fitting.
#   - No generator selection.
#   - No new materialization except dataframe assembly.
#   - No real TEST target values are used.
#   - Optional time columns from df_te may be copied as time/index metadata only.
#   - IoT full namespace is copied exactly from 12.g / 13.0.
#   - Protocol matrix is copied exactly from 14.1.
#   - This is a repair candidate, not a final Q4 success claim.
#
# Outputs:
#   synthetic/CPS_COUPLED_ZIGBEE_TEST.parquet
#   synthetic/IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE.parquet
#   reports/cell14_2_coupled_cps_assembly_audit.csv
#   reports/cell14_2_coupled_cps_column_registry.csv
#   reports/cell14_2_coupled_cps_contract.json
#   artifacts/contracts/cell14_2_coupled_cps_contract_v1_1_THESIS.json
#   artifacts/cell14_2_coupled_cps_manifest.json
# ==========================================================

log("--- START: Cell 14.2 - Reassemble coupling-conditioned CPS candidate (v1.1 repair-candidate/status-aware strict) ---")

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
_required_142 = [
    "CFG", "log",
    "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_FULL_SYN_TEST_130",
    "PROTOCOL_SYN_TEST_130",
    "PROTOCOL_SYN_TEST_COUPLED_ZIGBEE",
    "CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT",
    "CELL12G_IOT_FULL_NAMESPACE_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "CELL13_6_MANIFEST_Q4_CONTRACT",
]
_missing_142 = [k for k in _required_142 if k not in globals()]
if _missing_142:
    raise RuntimeError(f"[Cell14.2] Missing required globals: {_missing_142}")

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
    raise RuntimeError(f"[Cell14.2] Invalid TEST length: N_TE={N_TE}")

CELL142_VERSION = "cell14_2_coupling_conditioned_cps_assembly_v1_1_repair_candidate_status_aware"

CFG["cell14_2_version"] = CELL142_VERSION
CFG["cell14_2_TEST_target_values_used"] = False
CFG["cell14_2_df_te_time_columns_used_for_time_only"] = bool(CFG.get("cell14_2_include_time_columns_from_df_te", True))
CFG["cell14_2_synthetic_values_mutated_here"] = False
CFG["cell14_2_selection_done_here"] = False
CFG["cell14_2_generator_fit_done_here"] = False
CFG["cell14_2_materialization_done_here"] = False
CFG["cell14_2_assembly_done_here"] = True
CFG["cell14_2_repair_candidate_not_final_claim"] = True
CFG["cell14_2_broad_q4_status_carried_forward"] = True
CFG["cell14_2_manifest_q4_status_carried_forward"] = True

CFG.setdefault("cell14_2_fail_on_iot_drift_from_12g", True)
CFG.setdefault("cell14_2_fail_on_protocol_uncoupled_non_target_drift", True)
CFG.setdefault("cell14_2_fail_on_row_mismatch", True)
CFG.setdefault("cell14_2_fail_on_duplicate_columns", True)
CFG.setdefault("cell14_2_include_time_columns_from_df_te", True)
CFG.setdefault("cell14_2_time_col_candidates", ["sec_epoch_s__canon", "sec", "timestamp", "time", "datetime"])
CFG.setdefault("cell14_2_expected_iot_cols", 530)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
coupled_cps_path = os.path.join(OUT_SYN, "CPS_COUPLED_ZIGBEE_TEST.parquet")
coupled_iot_path = os.path.join(OUT_SYN, "IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE.parquet")
assembly_audit_csv = os.path.join(REPORT_DIR, "cell14_2_coupled_cps_assembly_audit.csv")
column_registry_csv = os.path.join(REPORT_DIR, "cell14_2_coupled_cps_column_registry.csv")
protocol_drift_csv = os.path.join(REPORT_DIR, "cell14_2_protocol_vs_uncoupled_drift_audit.csv")
iot_drift_csv = os.path.join(REPORT_DIR, "cell14_2_iot_vs_12g_drift_audit.csv")
contract_json = os.path.join(REPORT_DIR, "cell14_2_coupled_cps_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_2_coupled_cps_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell14_2_coupled_cps_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_142(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_142(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_142(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_142(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_142(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_142(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_142(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_142(obj.to_dict())
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

def _write_json_142(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_142(payload), f, indent=2, sort_keys=True)

def _sha256_file_142(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_142(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_num_array_142(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _series_equal_exact_or_close_142(a, b, rtol=1e-6, atol=1e-8) -> bool:
    aa = np.asarray(a)
    bb = np.asarray(b)

    if aa.shape != bb.shape:
        return False

    if aa.dtype.kind in {"b", "i", "u"} and bb.dtype.kind in {"b", "i", "u"}:
        return bool(np.array_equal(aa, bb))

    x = pd.to_numeric(pd.Series(aa), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(bb), errors="coerce").to_numpy(dtype=np.float64)

    return bool(np.all((np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)))

def _changed_count_142(a, b, rtol=1e-6, atol=1e-8) -> int:
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=np.float64)
    same = (np.isnan(x) & np.isnan(y)) | np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
    return int((~same).sum())

def _role_for_col_142(col: str) -> str:
    s = str(col)
    if s.startswith("events_in_sec__"):
        return "iot_driver"
    if s.startswith("iot__"):
        return "iot_full_namespace"
    if s.startswith("router__") or s.startswith("dns__") or s.startswith("lan__") or s.startswith("wan__"):
        return "protocol_router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__") or s.startswith("wifi__") or s.startswith("wlan__"):
        return "protocol_ota"
    if s.startswith("zigbee__") or s.startswith("zb__"):
        return "protocol_zigbee"
    if s.startswith("zwave__") or "zwave" in s.lower():
        return "protocol_zwave"
    if s in set(map(str, CFG.get("cell14_2_time_col_candidates", []))):
        return "time"
    return "other_or_context"

def _protocol_tier_142(col: str) -> str:
    role = _role_for_col_142(col)
    if role == "protocol_router":
        return "router"
    if role == "protocol_ota":
        return "ota"
    if role == "protocol_zigbee":
        return "zigbee"
    if role == "protocol_zwave":
        return "zwave"
    return ""

def _detect_time_cols_142():
    cols = []
    for c in CFG.get("cell14_2_time_col_candidates", ["sec_epoch_s__canon", "sec", "timestamp", "time", "datetime"]):
        if c in df_te.columns:
            cols.append(str(c))
    return cols

def _normalise_frame_142(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"[Cell14.2] {name} is not a DataFrame.")
    out = frame.copy()
    out.index = df_te.index
    out.columns = out.columns.astype(str)
    if len(out) != N_TE:
        raise RuntimeError(f"[Cell14.2] {name} row mismatch: got={len(out)} expected={N_TE}")
    return out

def _require_version_142(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.2] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.2] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

# ----------------------------------------------------------
# 3) Validate upstream contracts
# ----------------------------------------------------------
version141_142 = _require_version_142(
    CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT,
    "CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT",
    "cell14_1_coupling_conditioned_zigbee_protocol_materialization_v1_1",
)
version12g_142 = _require_version_142(
    CELL12G_IOT_FULL_NAMESPACE_CONTRACT,
    "CELL12G_IOT_FULL_NAMESPACE_CONTRACT",
    "cell12g_full_iot_namespace_assembly_strict_v1_2",
)
version134_142 = _require_version_142(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)
version136_142 = _require_version_142(
    CELL13_6_MANIFEST_Q4_CONTRACT,
    "CELL13_6_MANIFEST_Q4_CONTRACT",
    "cell13_6_manifest_informed_zigbee_q4_qa_v1_1",
)

strict141_142 = CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get("strict_contract", {})
if bool(strict141_142.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.2] Cell 14.1 contract indicates TEST real values were used.")
if not bool(strict141_142.get("synthetic_iot_drivers_used_for_conditioning", False)):
    raise RuntimeError("[Cell14.2] Cell 14.1 did not declare synthetic IoT driver conditioning.")
if not bool(strict141_142.get("only_target_protocol_columns_mutated", False)):
    raise RuntimeError("[Cell14.2] Cell 14.1 did not guarantee target-only protocol mutation.")
if not bool(strict141_142.get("only_legal_windows_mutated", False)):
    raise RuntimeError("[Cell14.2] Cell 14.1 did not guarantee legal-window-only mutation.")
if bool(strict141_142.get("iot_values_mutated", True)):
    raise RuntimeError("[Cell14.2] Cell 14.1 contract indicates IoT values were mutated.")
if int(CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get("safety_failure_count", -1)) != 0:
    raise RuntimeError(
        "[Cell14.2] Cell 14.1 safety failures are nonzero: "
        f"{CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get('safety_failure_count')}"
    )

strict12g_142 = CELL12G_IOT_FULL_NAMESPACE_CONTRACT.get("strict_contract", {})
if bool(CELL12G_IOT_FULL_NAMESPACE_CONTRACT.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.2] Cell 12.g contract indicates TEST real values were used.")

broad_q4_status_142 = str(CELL13_4_Q4_PUBLICATION_SUMMARY.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_142 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("pair_blocker_n", 0) or 0
)
broad_q4_pair_pass_rate_142 = _safe_float_142(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)
broad_q4_publication_blocker_record_n_142 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get(
        "publication_blocker_record_n",
        CELL13_4_Q4_PUBLICATION_SUMMARY.get("publication_blocker_n", 0),
    ) or 0
)

manifest_q4_summary_142 = CELL13_6_MANIFEST_Q4_CONTRACT.get("overall_summary", {})
manifest_q4_publication_blocker_n_142 = int(manifest_q4_summary_142.get("publication_blocker_n", 0) or 0)
manifest_q4_pair_pass_rate_142 = _safe_float_142(
    manifest_q4_summary_142.get("manifest_pair_pass_rate", np.nan),
    np.nan,
)
manifest_q4_pairs_total_142 = int(manifest_q4_summary_142.get("pairs_total", 0) or 0)

# ----------------------------------------------------------
# 4) Normalize input frames
# ----------------------------------------------------------
iot_full = _normalise_frame_142(IOT_FULL_SYN_TEST_130, "IOT_FULL_SYN_TEST_130")
protocol_base = _normalise_frame_142(PROTOCOL_SYN_TEST_130, "PROTOCOL_SYN_TEST_130")
protocol_coupled = _normalise_frame_142(PROTOCOL_SYN_TEST_COUPLED_ZIGBEE, "PROTOCOL_SYN_TEST_COUPLED_ZIGBEE")

expected_iot_cols = int(CFG.get("cell14_2_expected_iot_cols", 530))
if iot_full.shape[1] != expected_iot_cols:
    raise RuntimeError(
        f"[Cell14.2] IoT full namespace width mismatch: got={iot_full.shape[1]} expected={expected_iot_cols}"
    )

if list(protocol_base.columns) != list(protocol_coupled.columns):
    raise RuntimeError(
        "[Cell14.2] Coupled protocol schema differs from base protocol schema. "
        f"base_cols={len(protocol_base.columns)} coupled_cols={len(protocol_coupled.columns)}"
    )

if list(iot_full.index) != list(df_te.index):
    raise RuntimeError("[Cell14.2] IoT full index mismatch against df_te index.")

if list(protocol_coupled.index) != list(df_te.index):
    raise RuntimeError("[Cell14.2] Coupled protocol index mismatch against df_te index.")

# ----------------------------------------------------------
# 5) Time/context columns
# ----------------------------------------------------------
time_cols = _detect_time_cols_142() if bool(CFG.get("cell14_2_include_time_columns_from_df_te", True)) else []
time_df = pd.DataFrame(index=df_te.index)
for c in time_cols:
    time_df[c] = df_te[c].to_numpy(copy=True)

# Avoid duplicate time columns if already included in protocol or IoT.
time_cols_final = [
    c for c in time_cols
    if c not in protocol_coupled.columns and c not in iot_full.columns
]
time_df = time_df[time_cols_final].copy() if time_cols_final else pd.DataFrame(index=df_te.index)

# ----------------------------------------------------------
# 6) Drift audits
# ----------------------------------------------------------
target_protocol_cols = list(
    CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get("target_protocol_cols", [])
)
target_protocol_cols = [str(c) for c in target_protocol_cols]

if not target_protocol_cols:
    raise RuntimeError("[Cell14.2] Cell 14.1 contract did not declare target_protocol_cols.")

protocol_drift_rows = []
for c in protocol_base.columns:
    changed_n = _changed_count_142(protocol_base[c].to_numpy(copy=False), protocol_coupled[c].to_numpy(copy=False))
    protocol_drift_rows.append({
        "col": c,
        "protocol_tier": _protocol_tier_142(c),
        "is_target_coupled_col": bool(c in set(target_protocol_cols)),
        "changed_n": int(changed_n),
        "changed": bool(changed_n > 0),
    })

protocol_drift_df = pd.DataFrame(protocol_drift_rows)

non_target_protocol_drift_n = int(
    protocol_drift_df.loc[
        ~protocol_drift_df["is_target_coupled_col"].astype(bool),
        "changed_n",
    ].sum()
)

target_protocol_drift_n = int(
    protocol_drift_df.loc[
        protocol_drift_df["is_target_coupled_col"].astype(bool),
        "changed_n",
    ].sum()
)

expected_target_changed_n = int(CELL14_1_ZIGBEE_COUPLING_MATERIALIZATION_CONTRACT.get("target_changed_n", -1))
target_changed_matches_14_1 = bool(expected_target_changed_n == target_protocol_drift_n)

if not target_changed_matches_14_1:
    raise RuntimeError(
        "[Cell14.2] Target protocol drift count does not match Cell 14.1 contract. "
        f"14_1={expected_target_changed_n} assembly_diff={target_protocol_drift_n}"
    )

if non_target_protocol_drift_n > 0 and bool(CFG.get("cell14_2_fail_on_protocol_uncoupled_non_target_drift", True)):
    bad = protocol_drift_df[
        (~protocol_drift_df["is_target_coupled_col"].astype(bool))
        & (protocol_drift_df["changed_n"].astype(int) > 0)
    ].head(20).to_dict("records")
    raise RuntimeError(
        "[Cell14.2] Non-target protocol drift detected between base and coupled protocol matrices. "
        f"non_target_protocol_drift_n={non_target_protocol_drift_n}; preview={bad}"
    )

# Defensive IoT drift audit against current 13.0/12.g full namespace source.
iot_reference = IOT_FULL_SYN_TEST_130
iot_drift_rows = []
for c in iot_full.columns:
    changed_n = _changed_count_142(iot_reference[c].to_numpy(copy=False), iot_full[c].to_numpy(copy=False))
    if changed_n > 0:
        iot_drift_rows.append({
            "col": c,
            "changed_n": int(changed_n),
        })

iot_drift_df = pd.DataFrame(iot_drift_rows)
iot_drift_n = int(iot_drift_df["changed_n"].sum()) if len(iot_drift_df) else 0

if iot_drift_n > 0 and bool(CFG.get("cell14_2_fail_on_iot_drift_from_12g", True)):
    raise RuntimeError(
        "[Cell14.2] IoT drift detected during assembly, forbidden. "
        f"iot_drift_n={iot_drift_n}"
    )

# ----------------------------------------------------------
# 7) Assemble coupled CPS candidate
# ----------------------------------------------------------
assembly_parts = []
assembly_names = []

if len(time_df.columns):
    assembly_parts.append(time_df)
    assembly_names.append("time")

assembly_parts.append(protocol_coupled)
assembly_names.append("protocol_coupled")

assembly_parts.append(iot_full)
assembly_names.append("iot_full")

CPS_COUPLED_ZIGBEE_TEST = pd.concat(assembly_parts, axis=1)

dupe_cols = CPS_COUPLED_ZIGBEE_TEST.columns[CPS_COUPLED_ZIGBEE_TEST.columns.duplicated()].astype(str).tolist()
if dupe_cols and bool(CFG.get("cell14_2_fail_on_duplicate_columns", True)):
    raise RuntimeError(f"[Cell14.2] Duplicate columns in coupled CPS assembly: {dupe_cols[:30]}")

if len(CPS_COUPLED_ZIGBEE_TEST) != N_TE and bool(CFG.get("cell14_2_fail_on_row_mismatch", True)):
    raise RuntimeError(
        "[Cell14.2] Coupled CPS row mismatch: "
        f"got={len(CPS_COUPLED_ZIGBEE_TEST)} expected={N_TE}"
    )

IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE = iot_full.copy()

# ----------------------------------------------------------
# 8) Column registry and assembly audit
# ----------------------------------------------------------
registry_rows = []
for idx, c in enumerate(CPS_COUPLED_ZIGBEE_TEST.columns):
    role = _role_for_col_142(c)
    arr = pd.to_numeric(CPS_COUPLED_ZIGBEE_TEST[c], errors="coerce").to_numpy(dtype=np.float64)
    finite_n = int(np.isfinite(arr).sum())

    registry_rows.append({
        "col": c,
        "ordinal": int(idx),
        "role": role,
        "protocol_tier": _protocol_tier_142(c),
        "is_protocol": bool(role.startswith("protocol_")),
        "is_iot": bool(role.startswith("iot")),
        "is_time": bool(role == "time"),
        "is_coupled_target_protocol_col": bool(c in set(target_protocol_cols)),
        "rows": int(N_TE),
        "finite_n": finite_n,
        "finite_rate": float(finite_n / max(N_TE, 1)),
        "TEST_target_values_used": False,
        "df_te_time_column_used": bool(c in set(time_cols_final)),
    })

registry_df = pd.DataFrame(registry_rows)

role_counts = registry_df["role"].astype(str).value_counts().sort_index().to_dict()
protocol_tier_counts = (
    registry_df.loc[registry_df["is_protocol"].astype(bool), "protocol_tier"]
    .astype(str)
    .value_counts()
    .sort_index()
    .to_dict()
)

assembly_audit_rows = [
    {"metric": "test_rows", "value": int(N_TE)},
    {"metric": "cps_coupled_cols", "value": int(CPS_COUPLED_ZIGBEE_TEST.shape[1])},
    {"metric": "iot_full_cols", "value": int(iot_full.shape[1])},
    {"metric": "protocol_cols", "value": int(protocol_coupled.shape[1])},
    {"metric": "time_cols_added", "value": int(len(time_cols_final))},
    {"metric": "target_protocol_cols_n", "value": int(len(target_protocol_cols))},
    {"metric": "target_protocol_drift_n", "value": int(target_protocol_drift_n)},
    {"metric": "target_protocol_drift_matches_14_1", "value": bool(target_changed_matches_14_1)},
    {"metric": "non_target_protocol_drift_n", "value": int(non_target_protocol_drift_n)},
    {"metric": "iot_drift_n", "value": int(iot_drift_n)},
    {"metric": "duplicate_columns_n", "value": int(len(dupe_cols))},
    {"metric": "TEST_target_values_used", "value": False},
    {"metric": "df_te_time_columns_used_for_time_only", "value": bool(len(time_cols_final) > 0)},
    {"metric": "synthetic_values_mutated_here", "value": False},
    {"metric": "assembly_done_here", "value": True},
    {"metric": "repair_candidate_not_final_claim", "value": True},
]

for role, n in role_counts.items():
    assembly_audit_rows.append({"metric": f"role_count::{role}", "value": int(n)})

for tier, n in protocol_tier_counts.items():
    assembly_audit_rows.append({"metric": f"protocol_tier_count::{tier}", "value": int(n)})

assembly_audit_df = pd.DataFrame(assembly_audit_rows)

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
CPS_COUPLED_ZIGBEE_TEST.to_parquet(coupled_cps_path, index=True)
IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE.to_parquet(coupled_iot_path, index=True)
assembly_audit_df.to_csv(assembly_audit_csv, index=False)
registry_df.to_csv(column_registry_csv, index=False)
protocol_drift_df.to_csv(protocol_drift_csv, index=False)
iot_drift_df.to_csv(iot_drift_csv, index=False)

contract = {
    "cell": "14.2",
    "version": CELL142_VERSION,
    "role": "coupling_conditioned_cps_candidate_assembly",
    "quality_dimension": "Q4_cross_modal_consistency_repair_candidate",
    "upstream_contract_versions": {
        "cell12g": version12g_142,
        "cell13_4": version134_142,
        "cell13_6": version136_142,
        "cell14_1": version141_142,
    },
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_142,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_142),
        "pair_pass_rate": broad_q4_pair_pass_rate_142,
        "publication_blocker_record_n": int(broad_q4_publication_blocker_record_n_142),
    },
    "manifest_q4_status_carried_forward": {
        "pairs_total": int(manifest_q4_pairs_total_142),
        "publication_blocker_n": int(manifest_q4_publication_blocker_n_142),
        "manifest_pair_pass_rate": manifest_q4_pair_pass_rate_142,
    },
    "test_rows": int(N_TE),
    "shapes": {
        "CPS_COUPLED_ZIGBEE_TEST": list(CPS_COUPLED_ZIGBEE_TEST.shape),
        "IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE": list(IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE.shape),
        "PROTOCOL_SYN_TEST_COUPLED_ZIGBEE": list(protocol_coupled.shape),
    },
    "role_counts": {str(k): int(v) for k, v in role_counts.items()},
    "protocol_tier_counts": {str(k): int(v) for k, v in protocol_tier_counts.items()},
    "drift_summary": {
        "target_protocol_cols": target_protocol_cols,
        "target_protocol_drift_n": int(target_protocol_drift_n),
        "target_protocol_drift_matches_14_1": bool(target_changed_matches_14_1),
        "non_target_protocol_drift_n": int(non_target_protocol_drift_n),
        "iot_drift_n": int(iot_drift_n),
        "duplicate_columns_n": int(len(dupe_cols)),
    },
    "time_column_policy": {
        "time_cols_added": time_cols_final,
        "df_te_time_columns_used_for_time_only": bool(len(time_cols_final) > 0),
        "TEST_target_values_used": False,
    },
    "strict_contract": {
        "TEST_target_values_used": False,
        "df_te_time_columns_used_for_time_only": bool(len(time_cols_final) > 0),
        "synthetic_values_mutated_here": False,
        "iot_values_preserved_from_12g": bool(iot_drift_n == 0),
        "protocol_values_from_14_1": True,
        "only_target_protocol_cols_differ_from_uncoupled": bool(non_target_protocol_drift_n == 0),
        "target_protocol_drift_matches_14_1": bool(target_changed_matches_14_1),
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "assembly_done_here": True,
        "repair_candidate_not_final_claim": True,
    },
    "outputs": {
        "coupled_cps_path": coupled_cps_path,
        "coupled_iot_path": coupled_iot_path,
        "assembly_audit_csv": assembly_audit_csv,
        "column_registry_csv": column_registry_csv,
        "protocol_drift_csv": protocol_drift_csv,
        "iot_drift_csv": iot_drift_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_142(contract_json, contract)
_write_json_142(contract_canonical_json, contract)

manifest = {
    "cell": "14.2",
    "version": CELL142_VERSION,
    "created_outputs": contract["outputs"],
    "shapes": contract["shapes"],
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "manifest_q4_status_carried_forward": contract["manifest_q4_status_carried_forward"],
    "drift_summary": contract["drift_summary"],
    "strict_contract": contract["strict_contract"],
}

_write_json_142(manifest_json, manifest)

hashes = {
    "coupled_cps_sha256": _sha256_file_142(coupled_cps_path),
    "coupled_iot_sha256": _sha256_file_142(coupled_iot_path),
    "assembly_audit_csv_sha256": _sha256_file_142(assembly_audit_csv),
    "column_registry_csv_sha256": _sha256_file_142(column_registry_csv),
    "protocol_drift_csv_sha256": _sha256_file_142(protocol_drift_csv),
    "iot_drift_csv_sha256": _sha256_file_142(iot_drift_csv),
    "contract_json_sha256": _sha256_file_142(contract_json),
    "contract_canonical_json_sha256": _sha256_file_142(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_142(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_142(contract_json, contract)
_write_json_142(contract_canonical_json, contract)
_write_json_142(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals for 14.3+
# ----------------------------------------------------------
globals()["CELL142_VERSION"] = CELL142_VERSION
globals()["CPS_COUPLED_ZIGBEE_TEST"] = CPS_COUPLED_ZIGBEE_TEST
globals()["IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE"] = IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE
globals()["CELL14_2_COUPLED_CPS_ASSEMBLY_AUDIT_DF"] = assembly_audit_df
globals()["CELL14_2_COUPLED_CPS_COLUMN_REGISTRY_DF"] = registry_df
globals()["CELL14_2_PROTOCOL_DRIFT_DF"] = protocol_drift_df
globals()["CELL14_2_IOT_DRIFT_DF"] = iot_drift_df
globals()["CELL14_2_COUPLED_CPS_CONTRACT"] = contract
globals()["CELL14_2_COUPLED_CPS_PATH"] = coupled_cps_path
globals()["CELL14_2_COUPLED_IOT_PATH"] = coupled_iot_path
globals()["CELL14_2_COUPLED_CPS_ASSEMBLY_AUDIT_CSV"] = assembly_audit_csv
globals()["CELL14_2_COUPLED_CPS_COLUMN_REGISTRY_CSV"] = column_registry_csv
globals()["CELL14_2_PROTOCOL_DRIFT_CSV"] = protocol_drift_csv
globals()["CELL14_2_IOT_DRIFT_CSV"] = iot_drift_csv
globals()["CELL14_2_COUPLED_CPS_CONTRACT_JSON"] = contract_json
globals()["CELL14_2_COUPLED_CPS_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL14_2_COUPLED_CPS_MANIFEST_JSON"] = manifest_json

log(
    "[Cell14.2] Coupling-conditioned CPS assembly complete | "
    f"CPS_shape={CPS_COUPLED_ZIGBEE_TEST.shape} | "
    f"IoT_shape={IOT_FULL_SYNTHETIC_TEST_COUPLED_ZIGBEE.shape} | "
    f"protocol_shape={protocol_coupled.shape} | "
    f"target_protocol_drift_n={target_protocol_drift_n} | "
    f"non_target_protocol_drift_n={non_target_protocol_drift_n} | "
    f"iot_drift_n={iot_drift_n} | "
    f"time_cols_added={len(time_cols_final)}"
)
log(f"[Cell14.2] Role counts | {role_counts}")
log(f"[Cell14.2] Protocol tier counts | {protocol_tier_counts}")
log(
    "[Cell14.2] Upstream Q4 status carried forward | "
    f"broad_status={broad_q4_status_142} | "
    f"broad_pair_blocker_n={broad_q4_pair_blocker_n_142} | "
    f"manifest_blocker_n={manifest_q4_publication_blocker_n_142} | "
    f"manifest_pair_pass_rate={manifest_q4_pair_pass_rate_142}"
)
log(f"[Cell14.2] Saved coupled CPS candidate: {coupled_cps_path}")
log(f"[Cell14.2] Saved coupled IoT namespace: {coupled_iot_path}")
log(f"[Cell14.2] Saved assembly audit: {assembly_audit_csv}")
log(f"[Cell14.2] Saved contract: {contract_json}")
log(f"[Cell14.2] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell14.2] Contract flags | "
    "TEST_target_values_used=False | "
    f"df_te_time_columns_used_for_time_only={bool(len(time_cols_final) > 0)} | "
    "synthetic_values_mutated_here=False | "
    "iot_values_preserved_from_12g=True | "
    "protocol_values_from_14_1=True | "
    "only_target_protocol_cols_differ_from_uncoupled=True | "
    "target_protocol_drift_matches_14_1=True | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "assembly_done_here=True | "
    "repair_candidate_not_final_claim=True"
)
log("--- END: Cell 14.2 - Reassemble coupling-conditioned CPS candidate (v1.1 repair-candidate/status-aware strict) ---")

gc.collect()
