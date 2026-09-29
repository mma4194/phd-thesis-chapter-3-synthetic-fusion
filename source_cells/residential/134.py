# ==========================================================
# CELL 15.2a - Q6 DCR/NNDR blocker forensic diagnostic
# v1.0 STUDY-THESIS strict diagnostic-only privacy audit follow-up
#
# Role:
#   - Diagnose Cell 15.2 DCR/NNDR blockers/warnings.
#   - Identify which role(s) failed, why, and whether blockers are:
#       carried no-copy blockers,
#       DCR p01 ratio blockers,
#       NNDR p01 blockers,
#       close-neighbor-rate blockers,
#       active-window calibration issues,
#       or synthetic distinguishability/representation artifacts.
#   - Does NOT mutate data.
#   - Does NOT change Cell 15.2 results.
#   - Does NOT make final Q6 privacy decision.
#
# Outputs:
#   reports/cell15_2a_dcr_nndr_blocker_forensic_diagnostic.csv
#   reports/cell15_2a_dcr_nndr_blocker_top_neighbors.csv
#   reports/cell15_2a_dcr_nndr_status_reason_summary.csv
#   reports/cell15_2a_dcr_nndr_next_action_queue.csv
#   reports/cell15_2a_dcr_nndr_forensic_contract.json
#   artifacts/contracts/cell15_2a_dcr_nndr_forensic_contract_v1_0_THESIS.json
# ==========================================================

log("--- START: Cell 15.2a - Q6 DCR/NNDR blocker forensic diagnostic (v1.0 strict diagnostic-only) ---")

import os
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

_required_152a = [
    "CFG", "log", "OUTDIR", "OUT_SYN", "REPORT_DIR", "SEED",
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "CELL15_2_DCR_NNDR_ROLE_METRICS_DF",
    "CELL15_2_DCR_NNDR_TOP_NEIGHBORS_DF",
    "CELL15_2_DCR_NNDR_CONTRACT",
]
_missing_152a = [k for k in _required_152a if k not in globals()]
if _missing_152a:
    raise RuntimeError(f"[Cell15.2a] Missing required globals: {_missing_152a}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_152a = str(OUT_SYN)
REPORT_DIR_ACTIVE_152a = str(REPORT_DIR)

def _resolve_project_root_152a(outdir, report_dir, out_syn):
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
    raise RuntimeError("[Cell15.2a] Could not resolve canonical project root.")

PROJECT_ROOT_152a = _resolve_project_root_152a(OUTDIR, REPORT_DIR_ACTIVE_152a, OUT_SYN_ACTIVE_152a)
REPORT_DIR_152a = os.path.join(PROJECT_ROOT_152a, "reports")
ARTDIR_152a = os.path.join(PROJECT_ROOT_152a, "artifacts")
CONTRACT_DIR_152a = os.path.join(ARTDIR_152a, "contracts")
os.makedirs(REPORT_DIR_152a, exist_ok=True)
os.makedirs(ARTDIR_152a, exist_ok=True)
os.makedirs(CONTRACT_DIR_152a, exist_ok=True)

CELL152A_VERSION = "cell15_2a_q6_dcr_nndr_blocker_forensic_v1_0"

CFG["cell15_2a_version"] = CELL152A_VERSION
CFG["cell15_2a_diagnostic_only"] = True
CFG["cell15_2a_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_2a_Q4_coupled_artifacts_used"] = False
CFG["cell15_2a_TEST_real_values_used_for_privacy_reference"] = False
CFG["cell15_2a_synthetic_values_mutated"] = False
CFG["cell15_2a_selection_done_here"] = False
CFG["cell15_2a_generator_fit_done_here"] = False
CFG["cell15_2a_materialization_done_here"] = False
CFG["cell15_2a_privacy_decision_done_here"] = False

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _json_sanitize_152a(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_152a(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_152a(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_152a(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_152a(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_152a(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_152a(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_152a(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_152a(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_152a(payload), f, indent=2, sort_keys=True)

def _sha256_file_152a(path):
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_152a(obj, name, expected_substring):
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.2a] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.2a] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

def _safe_float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

# ----------------------------------------------------------
# Validate upstream contracts
# ----------------------------------------------------------
version150_152a = _require_contract_version_152a(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
version151a_152a = _require_contract_version_152a(
    CELL15_1A_NO_COPY_FORENSIC_CONTRACT,
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "cell15_1a_q6_no_copy_blocker_forensic_v1_0",
)
version152_152a = _require_contract_version_152a(
    CELL15_2_DCR_NNDR_CONTRACT,
    "CELL15_2_DCR_NNDR_CONTRACT",
    "cell15_2_q6_dcr_nndr_role_specific_privacy_v1_2",
)

strict150 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
strict152 = CELL15_2_DCR_NNDR_CONTRACT.get("strict_contract", {})

if str(strict150.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.2a] Cell 15.0 does not carry Q4 blocked_no_promotion.")
if bool(strict150.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.2a] Cell 15.0 used Q4 coupled artifacts unexpectedly.")
if bool(strict152.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell15.2a] Cell 15.2 indicates synthetic mutation.")
if not bool(strict152.get("dcr_nndr_audit_done_here", False)):
    raise RuntimeError("[Cell15.2a] Cell 15.2 did not complete DCR/NNDR audit.")

metrics = CELL15_2_DCR_NNDR_ROLE_METRICS_DF.copy()
top = CELL15_2_DCR_NNDR_TOP_NEIGHBORS_DF.copy()

if len(metrics) == 0:
    raise RuntimeError("[Cell15.2a] Empty Cell 15.2 role metrics.")

# ----------------------------------------------------------
# Analyze role-level failures
# ----------------------------------------------------------
reason_counter = Counter()
for reason in metrics.get("reasons", pd.Series(dtype=str)).astype(str):
    for part in reason.split("|"):
        part = part.strip()
        if part:
            reason_counter[part] += 1

diag_rows = []

for _, r in metrics.iterrows():
    role = str(r.get("role_group", ""))
    status = str(r.get("status", ""))
    reasons = str(r.get("reasons", ""))

    carried_no_copy = str(r.get("carried_no_copy_status", ""))
    carried_dcr = str(r.get("carried_dcr_nndr_status", ""))
    representation = str(r.get("representation", ""))

    ratio_active = _safe_float(r.get("syn_train_dcr_p01_ratio_active"), np.nan)
    nndr_active = _safe_float(r.get("syn_to_train_nndr_active_p01"), np.nan)
    close_rate_active = _safe_float(r.get("close_neighbor_rate_active"), np.nan)
    ratio_all = _safe_float(r.get("syn_train_dcr_p01_ratio"), np.nan)
    nndr_all = _safe_float(r.get("syn_to_train_nndr_p01"), np.nan)
    close_rate_all = _safe_float(r.get("close_neighbor_rate"), np.nan)

    calib_degen = bool(r.get("calibration_degenerate_active", False))

    if status == "blocker":
        if "carried_no_copy" in reasons:
            root = "carried_no_copy_blocker"
            action = "inspect Cell15.1a binary-state exact/near-copy matches; DCR/NNDR cannot clear this."
        elif "carried_DCR" in reasons:
            root = "carried_DCR_NNDR_blocker"
            action = "DCR/NNDR role remains blocker; inspect active nearest-neighbor distances."
        elif "DCR_p01_ratio" in reasons:
            root = "active_DCR_p01_ratio_blocker"
            action = "inspect active nearest neighbors and calibration threshold; verify deterministic state patterns."
        elif "NNDR" in reasons:
            root = "active_NNDR_p01_blocker"
            action = "inspect isolated close-neighbor examples; likely memorization-style risk if non-deterministic."
        elif "close_neighbor_rate" in reasons:
            root = "active_close_neighbor_rate_blocker"
            action = "inspect top active neighbors; determine if repeated deterministic states drive the rate."
        else:
            root = "unknown_DCR_NNDR_blocker"
            action = "inspect role metrics and top neighbors."
    elif status == "warning":
        if calib_degen:
            root = "active_calibration_degenerate_warning"
            action = "carry as warning; calibration p01 may be zero/unstable for deterministic roles."
        else:
            root = "DCR_NNDR_warning"
            action = "carry to Q6 summary; inspect if same role also has no-copy blocker."
    elif status == "pass":
        root = "DCR_NNDR_pass"
        action = "no action"
    else:
        root = "not_evaluable"
        action = "decide if role should be excluded or reconfigured."

    diag_rows.append({
        "role_group": role,
        "status": status,
        "reasons": reasons,
        "root_cause_class": root,
        "suggested_next_action": action,
        "representation": representation,
        "selected_cols_n": _safe_int(r.get("selected_cols_n"), 0),
        "syn_samples_n": _safe_int(r.get("syn_samples_n"), 0),
        "train_ref_n": _safe_int(r.get("train_ref_n"), 0),
        "test_ref_n": _safe_int(r.get("test_ref_n"), 0),
        "syn_active_samples_n": _safe_int(r.get("syn_active_samples_n"), 0),
        "real_test_active_samples_n": _safe_int(r.get("real_test_active_samples_n"), 0),
        "syn_idle_samples_n": _safe_int(r.get("syn_idle_samples_n"), 0),
        "real_test_idle_samples_n": _safe_int(r.get("real_test_idle_samples_n"), 0),
        "calibration_degenerate_active": calib_degen,
        "syn_train_dcr_p01_ratio_active": ratio_active,
        "syn_to_train_nndr_active_p01": nndr_active,
        "close_neighbor_rate_active": close_rate_active,
        "syn_train_dcr_p01_ratio_all": ratio_all,
        "syn_to_train_nndr_p01_all": nndr_all,
        "close_neighbor_rate_all": close_rate_all,
        "syn_to_train_dcr_active_p01": _safe_float(r.get("syn_to_train_dcr_active_p01"), np.nan),
        "real_test_to_train_dcr_active_p01": _safe_float(r.get("real_test_to_train_dcr_active_p01"), np.nan),
        "syn_to_train_dcr_p01": _safe_float(r.get("syn_to_train_dcr_p01"), np.nan),
        "real_test_to_train_dcr_p01": _safe_float(r.get("real_test_to_train_dcr_p01"), np.nan),
        "carried_no_copy_status": carried_no_copy,
        "carried_dcr_nndr_status": carried_dcr,
        "previous_v1_dcr_status": str(r.get("previous_v1_dcr_status", "")),
        "previous_v1_dcr_reasons": str(r.get("previous_v1_dcr_reasons", "")),
    })

diag = pd.DataFrame(diag_rows)

blocker_roles = diag[diag["status"].astype(str).eq("blocker")]["role_group"].astype(str).tolist()
warning_roles = diag[diag["status"].astype(str).eq("warning")]["role_group"].astype(str).tolist()

# Top neighbors for blocker/warning roles only.
if len(top) and "role_group" in top.columns:
    focus_top = top[top["role_group"].astype(str).isin(blocker_roles + warning_roles)].copy()
else:
    focus_top = pd.DataFrame()

# Add focus labels.
if len(focus_top):
    if "syn_is_idle_sample" in focus_top.columns:
        focus_top["neighbor_scope"] = np.where(
            focus_top["syn_is_idle_sample"].fillna(False).astype(bool),
            "idle_diagnostic",
            "active_privacy_relevant",
        )
    else:
        focus_top["neighbor_scope"] = "unknown"
    if "nearest_train_is_idle_sample" in focus_top.columns:
        focus_top["nearest_train_scope"] = np.where(
            focus_top["nearest_train_is_idle_sample"].fillna(False).astype(bool),
            "idle_train_neighbor",
            "active_train_neighbor",
        )
    else:
        focus_top["nearest_train_scope"] = "unknown"

# Status/reason summary.
summary_rows = []
for status, sub in diag.groupby("status", dropna=False):
    summary_rows.append({
        "status": str(status),
        "roles_n": int(len(sub)),
        "roles": "|".join(sub["role_group"].astype(str).tolist()),
        "mean_syn_train_dcr_p01_ratio_active": float(pd.to_numeric(sub["syn_train_dcr_p01_ratio_active"], errors="coerce").mean()),
        "mean_syn_to_train_nndr_active_p01": float(pd.to_numeric(sub["syn_to_train_nndr_active_p01"], errors="coerce").mean()),
        "mean_close_neighbor_rate_active": float(pd.to_numeric(sub["close_neighbor_rate_active"], errors="coerce").mean()),
    })

for reason, n in reason_counter.items():
    summary_rows.append({
        "status": f"reason::{reason}",
        "roles_n": int(n),
        "roles": "",
        "mean_syn_train_dcr_p01_ratio_active": np.nan,
        "mean_syn_to_train_nndr_active_p01": np.nan,
        "mean_close_neighbor_rate_active": np.nan,
    })

status_summary = pd.DataFrame(summary_rows)

# Action queue.
severity_order = {"blocker": 0, "warning": 1, "not_evaluable": 2, "pass": 3}
queue = diag.copy()
queue["_order"] = queue["status"].map(severity_order).fillna(99)
queue = queue.sort_values(
    ["_order", "close_neighbor_rate_active", "syn_to_train_nndr_active_p01", "syn_train_dcr_p01_ratio_active"],
    ascending=[True, False, True, True],
).drop(columns=["_order"])

# ----------------------------------------------------------
# Save outputs
# ----------------------------------------------------------
diag_csv = os.path.join(REPORT_DIR_152a, "cell15_2a_dcr_nndr_blocker_forensic_diagnostic.csv")
top_csv = os.path.join(REPORT_DIR_152a, "cell15_2a_dcr_nndr_blocker_top_neighbors.csv")
summary_csv = os.path.join(REPORT_DIR_152a, "cell15_2a_dcr_nndr_status_reason_summary.csv")
queue_csv = os.path.join(REPORT_DIR_152a, "cell15_2a_dcr_nndr_next_action_queue.csv")
contract_json = os.path.join(REPORT_DIR_152a, "cell15_2a_dcr_nndr_forensic_contract.json")
canonical_contract_json = os.path.join(CONTRACT_DIR_152a, "cell15_2a_dcr_nndr_forensic_contract_v1_0_THESIS.json")

diag.to_csv(diag_csv, index=False)
focus_top.to_csv(top_csv, index=False)
status_summary.to_csv(summary_csv, index=False)
queue.to_csv(queue_csv, index=False)

contract = {
    "cell": "15.2a",
    "version": CELL152A_VERSION,
    "role": "q6_dcr_nndr_blocker_forensic_diagnostic",
    "quality_dimension": "Q6_privacy_no_copy_release_safety",
    "upstream_contract_versions": {
        "cell15_0": version150_152a,
        "cell15_1a": version151a_152a,
        "cell15_2": version152_152a,
    },
    "summary": {
        "roles_total": int(len(diag)),
        "blocker_roles_n": int(len(blocker_roles)),
        "warning_roles_n": int(len(warning_roles)),
        "blocker_roles": blocker_roles,
        "warning_roles": warning_roles,
        "reason_counts": dict(reason_counter),
        "ready_for_cell15_3": True,
        "recommendation": (
            "Proceed to MIA as additional evidence, but final Q6 release remains blocked/warning until carried blockers are resolved or explicitly scoped."
            if blocker_roles else
            "No DCR/NNDR blockers; proceed to MIA."
        ),
    },
    "strict_contract": {
        "diagnostic_only": True,
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_for_privacy_reference": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "privacy_decision_done_here": False,
    },
    "outputs": {
        "diag_csv": diag_csv,
        "top_csv": top_csv,
        "summary_csv": summary_csv,
        "queue_csv": queue_csv,
        "contract_json": contract_json,
        "canonical_contract_json": canonical_contract_json,
    },
}

_write_json_152a(contract_json, contract)
_write_json_152a(canonical_contract_json, contract)

globals()["CELL152A_VERSION"] = CELL152A_VERSION
globals()["CELL15_2A_DCR_NNDR_BLOCKER_FORENSIC_DF"] = diag
globals()["CELL15_2A_DCR_NNDR_BLOCKER_TOP_NEIGHBORS_DF"] = focus_top
globals()["CELL15_2A_DCR_NNDR_STATUS_REASON_SUMMARY_DF"] = status_summary
globals()["CELL15_2A_DCR_NNDR_NEXT_ACTION_QUEUE_DF"] = queue
globals()["CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT"] = contract
globals()["CELL15_2A_DCR_NNDR_BLOCKER_FORENSIC_CSV"] = diag_csv
globals()["CELL15_2A_DCR_NNDR_BLOCKER_TOP_NEIGHBORS_CSV"] = top_csv
globals()["CELL15_2A_DCR_NNDR_STATUS_REASON_SUMMARY_CSV"] = summary_csv
globals()["CELL15_2A_DCR_NNDR_NEXT_ACTION_QUEUE_CSV"] = queue_csv
globals()["CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT_JSON"] = contract_json
globals()["CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT_CANONICAL_JSON"] = canonical_contract_json

log(
    "[Cell15.2a] DCR/NNDR blocker forensic diagnostic complete | "
    f"roles={len(diag)} | blocker_roles={blocker_roles} | warning_roles={warning_roles}"
)
log(f"[Cell15.2a] Reason counts | {dict(reason_counter)}")
log(f"[Cell15.2a] Saved diagnostic: {diag_csv} | rows={len(diag)}")
log(f"[Cell15.2a] Saved focused top neighbors: {top_csv} | rows={len(focus_top)}")
log(
    "[Cell15.2a] Contract flags | diagnostic_only=True | Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | TEST_real_values_used_for_privacy_reference=False | "
    "synthetic_values_mutated=False | privacy_decision_done_here=False"
)
log("--- END: Cell 15.2a - Q6 DCR/NNDR blocker forensic diagnostic (v1.0 strict diagnostic-only) ---")