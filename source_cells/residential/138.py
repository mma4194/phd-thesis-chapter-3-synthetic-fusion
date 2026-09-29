# %% CELL Q6.ROUTER.RELEASE_POLICY.V1 — predeclared router release rule before Cell 15.6
# Purpose:
#   Decide whether protocol_router may be warning-governed in the Q6 public baseline.
#
# Scientific contract:
#   - No fitting, no generation, no value mutation.
#   - Does not use TEST real values for materialization or selection.
#   - Reads only synthetic public-candidate source columns and existing Q6 role-level audits.
#   - Fails closed: if router columns look identifier-like, nonnumeric, negative, malformed,
#     missing from the source artifact, or blocked by hard privacy/copy evidence, Cell 15.6
#     must not keep protocol_router.
#
# Output:
#   reports/q6_router_release_policy_v1_audit.csv
#   reports/cell15_4_q6_role_release_safety_router_governed.csv
#   artifacts/contracts/q6_router_release_policy_v1_THESIS.json

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "CFG" not in globals() or not isinstance(CFG, dict):
    raise RuntimeError("[Q6.ROUTER] Run Cell 1 first so CFG is defined.")

_required_router = [
    "OUTDIR", "REPORT_DIR", "CONTRACT_DIR",
    "CELL15_0_PRIVACY_ROLE_GROUPS",
    "CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF",
    "CELL15_4_Q6_RELEASE_SAFETY_CONTRACT",
    "CELL15_5_Q6_MITIGATION_PLAN_DF",
    "CELL15_5_Q6_MITIGATION_CONTRACT",
]
_missing_router = [k for k in _required_router if k not in globals()]
if _missing_router:
    raise RuntimeError(f"[Q6.ROUTER] Missing required globals: {_missing_router}")

OUTDIR_P = Path(str(OUTDIR)).expanduser().resolve()
REPORT_DIR_P = Path(str(REPORT_DIR)).expanduser().resolve()
CONTRACT_DIR_P = Path(str(CONTRACT_DIR)).expanduser().resolve()
REPORT_DIR_P.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_P.mkdir(parents=True, exist_ok=True)

def _router_log(msg):
    print(f"[Q6.ROUTER] {msg}")

def _sha256_file_router(path):
    path = Path(str(path))
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _json_sanitize_router(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_router(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize_router(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_router(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_router(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _json_sanitize_router(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_router(path, payload):
    Path(path).write_text(json.dumps(_json_sanitize_router(payload), indent=2, sort_keys=True), encoding="utf-8")

def _row_for_role(df, role):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    role_cols = [c for c in df.columns if c.lower() in {"role", "role_group", "scope", "group"}]
    if not role_cols:
        return {}
    col = role_cols[0]
    m = df[df[col].astype(str).eq(str(role))]
    return m.iloc[0].to_dict() if len(m) else {}

def _norm_status(x):
    return str(x if x is not None else "").strip().lower()

def _truthy_status_is_blocker(x):
    return _norm_status(x) in {"blocker", "release_blocker", "fatal", "fail", "failed"}

def _contains_identifier_name(col):
    # Conservative direct-identifier screen. Do not block generic protocol terms such as "ip_len".
    s = str(col).lower()
    high_risk_patterns = [
        "src_ip", "dst_ip", "ip_src", "ip_dst", "ipv4_addr", "ipv6_addr",
        "mac", "bssid", "ssid", "eui", "hostname", "host_name", "device_id",
        "payload", "raw_payload", "uuid", "serial", "imei", "imsi",
        "cookie", "token", "password", "passwd", "secret", "credential",
        "lat", "latitude", "lon", "longitude", "gps",
    ]
    return any(p in s for p in high_risk_patterns)

role_groups_router = CELL15_0_PRIVACY_ROLE_GROUPS
router_cols = [str(c) for c in role_groups_router.get("protocol_router", [])]
if len(router_cols) != 18:
    raise RuntimeError(
        "[Q6.ROUTER] Expected exactly 18 protocol_router columns for the corrected public baseline; "
        f"got {len(router_cols)}."
    )

source_cps_path = str(globals().get("CELL15_0_FINAL_CPS_NO_Q4_PATH", ""))
if not source_cps_path:
    source_cps_path = str(CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("source_artifacts", {}).get("source_no_q4_cps_path", ""))
if not source_cps_path:
    source_cps_path = str(CELL15_4_Q6_RELEASE_SAFETY_CONTRACT.get("privacy_input_artifacts", {}).get("final_no_q4_cps_path", ""))
if not source_cps_path and isinstance(globals().get("CELL15_0_PRIVACY_INPUT_CONTRACT", None), dict):
    source_cps_path = str(CELL15_0_PRIVACY_INPUT_CONTRACT.get("privacy_input_artifacts", {}).get("final_no_q4_cps_path", ""))

if isinstance(globals().get("CPS_SYNTHETIC_TEST_FINAL_NO_Q4", None), pd.DataFrame):
    _src_df_header = globals()["CPS_SYNTHETIC_TEST_FINAL_NO_Q4"]
    missing_router_cols = [c for c in router_cols if c not in _src_df_header.columns.astype(str)]
    if missing_router_cols:
        raise RuntimeError(f"[Q6.ROUTER] Missing router columns from in-memory no-Q4 source: {missing_router_cols[:10]}")
    router_frame = _src_df_header.loc[:, router_cols].copy()
elif source_cps_path and Path(source_cps_path).exists():
    try:
        router_frame = pd.read_parquet(source_cps_path, columns=router_cols)
    except Exception:
        # fallback for parquet engines that do not support column projection reliably
        tmp = pd.read_parquet(source_cps_path)
        tmp.columns = tmp.columns.astype(str)
        missing_router_cols = [c for c in router_cols if c not in tmp.columns]
        if missing_router_cols:
            raise RuntimeError(f"[Q6.ROUTER] Missing router columns from no-Q4 source: {missing_router_cols[:10]}")
        router_frame = tmp.loc[:, router_cols].copy()
        del tmp
else:
    raise RuntimeError(f"[Q6.ROUTER] Could not resolve source no-Q4 CPS artifact: {source_cps_path}")

router_frame.columns = router_frame.columns.astype(str)

identifier_like_cols = [c for c in router_cols if _contains_identifier_name(c)]
nonnumeric_cols = []
negative_cols = []
nonfinite_cols = []
noninteger_cols = []

for c in router_cols:
    x = pd.to_numeric(router_frame[c], errors="coerce")
    finite = x[np.isfinite(x)]
    if x.notna().mean() == 0:
        nonnumeric_cols.append(c)
        continue
    if int((~np.isfinite(x.to_numpy(dtype=float))).sum()) > 0 and x.notna().any():
        # NaN may represent inactivity; infinities do not.
        arr = x.to_numpy(dtype=float)
        if np.isinf(arr).any():
            nonfinite_cols.append(c)
    if len(finite) and float(finite.min()) < -1e-9:
        negative_cols.append(c)
    # Router public features must be aggregate/count-like; tolerate float storage of integer values.
    if len(finite):
        if float(np.nanmax(np.abs(finite.to_numpy(dtype=float) - np.rint(finite.to_numpy(dtype=float))))) > 1e-6:
            noninteger_cols.append(c)

release_df_router = CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF.copy()
router_release_row = _row_for_role(release_df_router, "protocol_router")
mit_df_router = CELL15_5_Q6_MITIGATION_PLAN_DF.copy()
router_mit_row = _row_for_role(mit_df_router, "protocol_router")

hard_privacy_blocker_flags = {
    "no_copy_status": _truthy_status_is_blocker(router_release_row.get("no_copy_status")),
    "no_copy_forensic_status": _truthy_status_is_blocker(router_release_row.get("no_copy_forensic_status")),
    "dcr_nndr_status": _truthy_status_is_blocker(router_release_row.get("dcr_nndr_status")),
    "dcr_forensic_status": _truthy_status_is_blocker(router_release_row.get("dcr_forensic_status")),
}
synthetic_dist_blocker = _truthy_status_is_blocker(router_release_row.get("synthetic_distinguishability_status"))
temporal_drift_blocker = _truthy_status_is_blocker(router_release_row.get("temporal_drift_status"))
initial_release_status = _norm_status(router_release_row.get("release_status", router_release_row.get("initial_release_status", "")))

mitigation_priority = str(router_mit_row.get("mitigation_priority", router_mit_row.get("priority", ""))).strip().upper()
mitigation_action = str(router_mit_row.get("recommended_action", router_mit_row.get("mitigation_action", ""))).strip()

hard_privacy_blocker_n = int(sum(bool(v) for v in hard_privacy_blocker_flags.values()))
structural_issue_n = int(len(identifier_like_cols) + len(nonnumeric_cols) + len(negative_cols) + len(nonfinite_cols) + len(noninteger_cols))

# Router can be warning-governed only if its blocker is distinguishability/temporal-context type,
# not direct copying, DCR/NNDR, forensic copy, direct identifier naming, or malformed count domain.
router_warning_governable = bool(
    structural_issue_n == 0
    and hard_privacy_blocker_n == 0
    and mitigation_priority not in {"P0", "P1"}
    and (
        initial_release_status in {"release_pass", "pass", "release_warning", "warning"}
        or synthetic_dist_blocker
        or temporal_drift_blocker
    )
)

audit_rows = []
for c in router_cols:
    x = pd.to_numeric(router_frame[c], errors="coerce")
    finite = x[np.isfinite(x)]
    audit_rows.append({
        "col": c,
        "role_group": "protocol_router",
        "direct_identifier_name_hit": bool(c in identifier_like_cols),
        "numeric_rate": float(x.notna().mean()) if len(x) else np.nan,
        "finite_non_nan_n": int(len(finite)),
        "min": float(finite.min()) if len(finite) else np.nan,
        "max": float(finite.max()) if len(finite) else np.nan,
        "nonnegative_ok": bool(c not in negative_cols),
        "integer_countlike_ok": bool(c not in noninteger_cols),
        "eligible_under_router_policy_v1": bool(router_warning_governable),
    })

audit_df_router = pd.DataFrame(audit_rows)
audit_csv_router = REPORT_DIR_P / "q6_router_release_policy_v1_audit.csv"
audit_df_router.to_csv(audit_csv_router, index=False)

release_df_governed = release_df_router.copy()
role_col = next((c for c in release_df_governed.columns if c.lower() in {"role", "role_group", "scope", "group"}), None)
if role_col is not None:
    m = release_df_governed[role_col].astype(str).eq("protocol_router")
    if m.any() and router_warning_governable:
        release_df_governed.loc[m, "release_status"] = "release_warning"
        release_df_governed.loc[m, "release_reasons"] = (
            release_df_governed.loc[m, "release_reasons"].astype(str)
            + "|router_release_policy_v1_warning_governed_nonidentifying_countlike_subset"
        )
        release_df_governed.loc[m, "release_policy_override_applied"] = True
        release_df_governed.loc[m, "release_policy_override_reason"] = (
            "Q6.ROUTER.RELEASE_POLICY.V1: retained only because 18 router columns pass direct-identifier-name, "
            "numeric, nonnegative, integer/count-like, and hard privacy/copy-risk checks; retained with warnings, "
            "not as full-network realism or privacy proof."
        )

governed_release_csv = REPORT_DIR_P / "cell15_4_q6_role_release_safety_router_governed.csv"
release_df_governed.to_csv(governed_release_csv, index=False)

contract_router = {
    "cell": "Q6.ROUTER.RELEASE_POLICY.V1",
    "version": "q6_router_release_policy_v1_THESIS",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "predeclared_warning_governed_router_release_rule_for_corrected_public_baseline",
    "source_cps_path": source_cps_path,
    "source_cps_sha256": _sha256_file_router(source_cps_path) if source_cps_path else "",
    "router_cols_n": int(len(router_cols)),
    "router_cols": list(router_cols),
    "router_release_eligible": bool(router_warning_governable),
    "router_release_status_after_policy": "release_warning" if router_warning_governable else "release_blocker",
    "policy_scope": {
        "kept_as": "public_protocol_benchmarking_scope_with_warning_disclosure",
        "not_claimed_as": [
            "full_network_realism",
            "formal_privacy_proof",
            "all_protocol_coupling_solution",
            "household_behavior_uninferability",
        ],
    },
    "checks": {
        "expected_router_cols_18": bool(len(router_cols) == 18),
        "direct_identifier_name_hits_n": int(len(identifier_like_cols)),
        "direct_identifier_name_hits": identifier_like_cols,
        "nonnumeric_cols": nonnumeric_cols,
        "negative_cols": negative_cols,
        "nonfinite_inf_cols": nonfinite_cols,
        "noninteger_cols": noninteger_cols,
        "structural_issue_n": structural_issue_n,
        "hard_privacy_blocker_flags": hard_privacy_blocker_flags,
        "hard_privacy_blocker_n": hard_privacy_blocker_n,
        "synthetic_distinguishability_blocker_warning_governed": bool(synthetic_dist_blocker),
        "temporal_drift_blocker_warning_governed": bool(temporal_drift_blocker),
        "mitigation_priority": mitigation_priority,
        "mitigation_action": mitigation_action,
    },
    "strict_contract": {
        "predeclared_before_15_6": True,
        "TEST_real_values_used_for_materialization": False,
        "TEST_real_values_used_for_selection": False,
        "synthetic_values_mutated": False,
        "columns_suppressed_or_kept_only": True,
        "router_policy_may_only_downgrade_distinguishability_temporal_context_blockers_to_warning": True,
        "direct_identifier_or_hard_privacy_blocker_remains_fail_closed": True,
    },
    "outputs": {
        "audit_csv": str(audit_csv_router),
        "governed_release_safety_csv": str(governed_release_csv),
        "contract_json": str(CONTRACT_DIR_P / "q6_router_release_policy_v1_THESIS.json"),
    },
}
contract_router["hashes"] = {
    "audit_csv_sha256": _sha256_file_router(audit_csv_router),
    "governed_release_safety_csv_sha256": _sha256_file_router(governed_release_csv),
}
contract_path_router = CONTRACT_DIR_P / "q6_router_release_policy_v1_THESIS.json"
_write_json_router(contract_path_router, contract_router)
contract_router["hashes"]["contract_json_sha256"] = _sha256_file_router(contract_path_router)
_write_json_router(contract_path_router, contract_router)

globals()["CELL_Q6_ROUTER_RELEASE_POLICY_CONTRACT"] = contract_router
globals()["CELL_Q6_ROUTER_RELEASE_POLICY_AUDIT_DF"] = audit_df_router
globals()["CELL15_4_Q6_ROLE_RELEASE_SAFETY_ROUTER_GOVERNED_DF"] = release_df_governed
# Use the governed release-safety table downstream so router is warning-governed, not silently bypassed.
globals()["CELL15_4_Q6_ROLE_RELEASE_SAFETY_DF"] = release_df_governed

if not router_warning_governable:
    print(json.dumps(contract_router["checks"], indent=2, sort_keys=True))
    raise RuntimeError(
        "[Q6.ROUTER] protocol_router is not eligible for warning-governed public retention. "
        "Keep the 30-column public baseline and update the paper unless the hard blocker is legitimately resolved."
    )

# Precommit Cell 15.6 to the corrected public scope. Sparse drivers are no
# longer retained by role name alone: corrected-fatal drivers from Cell 12.e.6R
# are excluded from Q6 release eligibility before the public artifact is built.
if "DRIVER_RELEASE_ELIGIBLE_COLS_CORRECTED" not in globals():
    raise RuntimeError(
        "[Q6.ROUTER] Corrected sparse-driver release eligibility is missing. "
        "Run Cell 12.e.6R before precommitting Q6 public scope."
    )
_driver_release_cols_pre = sorted(map(str, DRIVER_RELEASE_ELIGIBLE_COLS_CORRECTED))
_driver_excluded_cols_pre = sorted(map(str, globals().get("DRIVER_RELEASE_EXCLUDED_FATAL_COLS_CORRECTED", [])))
expected_public_cols_pre = 1 + int(len(router_cols)) + 4 + int(len(_driver_release_cols_pre))
if len(_driver_release_cols_pre) != int(CFG.get("expected_public_sparse_driver_cols", 23)):
    raise RuntimeError(
        "[Q6.ROUTER] Corrected driver release count disagrees with Cell 1 public accounting: "
        f"eligible={len(_driver_release_cols_pre)} expected={CFG.get('expected_public_sparse_driver_cols')}"
    )
CFG["cell15_6_public_policy"] = "router_zigbee_event_driver_corrected_release_v2"
CFG["cell15_6_keep_roles"] = [
    "protocol_router",
    "protocol_zigbee",
    "iot_event_drivers",
]
CFG["cell15_6_drop_roles"] = [
    "protocol_ota",
    "iot_binary_states",
    "iot_continuous_values",
    "iot_observability_masks",
    "iot_placeholders_or_excluded",
]
CFG["cell15_6_include_time_columns"] = True
CFG["cell15_6_time_column_allowlist"] = ["sec_epoch_s__canon"]
CFG["cell15_6_fail_if_dropped_high_risk_column_remains"] = True
CFG["cell15_6_fail_if_kept_values_drift"] = True
CFG["cell15_6_min_public_cols"] = expected_public_cols_pre
CFG["cell15_6_expected_public_cols"] = expected_public_cols_pre
CFG["cell15_6_use_router_release_policy"] = True
CFG["cell15_6_exclude_corrected_fatal_sparse_drivers"] = True

precommit_contract = {
    "cell": "Q6_public_scope_precommit_before_15_6",
    "policy": CFG["cell15_6_public_policy"],
    "keep_roles": CFG["cell15_6_keep_roles"],
    "drop_roles": CFG["cell15_6_drop_roles"],
    "include_time_columns": CFG["cell15_6_include_time_columns"],
    "time_column_allowlist": CFG["cell15_6_time_column_allowlist"],
    "expected_logical_public_cols": int(CFG["cell15_6_expected_public_cols"]),
    "corrected_sparse_driver_release_eligible_n": int(len(_driver_release_cols_pre)),
    "corrected_sparse_driver_release_excluded_fatal_n": int(len(_driver_excluded_cols_pre)),
    "corrected_sparse_driver_release_eligible_cols": _driver_release_cols_pre,
    "corrected_sparse_driver_release_excluded_fatal_cols": _driver_excluded_cols_pre,
    "router_release_policy_contract": str(contract_path_router),
    "router_release_eligible": bool(router_warning_governable),
    "TEST_privacy_outcome_used": False,
    "selection_done_here": True,
    "synthetic_values_mutated": False,
    "public_candidate_must_be_reaudited_once_after_materialization": True,
}
precommit_path = CONTRACT_DIR_P / "q6_public_scope_precommit_before_15_6_THESIS.json"
precommit_path.write_text(json.dumps(_json_sanitize_router(precommit_contract), indent=2, sort_keys=True), encoding="utf-8")

_router_log("PASS: protocol_router is warning-governed under Q6.ROUTER.RELEASE_POLICY.V1.")
_router_log(f"Precommitted Cell 15.6 corrected expected logical public columns = {CFG['cell15_6_expected_public_cols']}")
_router_log(f"Saved router audit: {audit_csv_router}")
_router_log(f"Saved router policy contract: {contract_path_router}")
