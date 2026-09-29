# ==========================================================
# CELL 15.1a - Q6 no-copy blocker forensic diagnostic
# v1.0 STUDY-THESIS strict diagnostic-only privacy audit follow-up
#
# Role:
#   - Diagnose Cell 15.1 no-copy blockers/warnings.
#   - Identify which role(s) failed, why, and whether the blocker comes from:
#       exact nonzero copied windows,
#       nonzero near-copy windows,
#       p99 nonzero similarity,
#       carried sparse/all-zero behavior,
#       or mixed-role feature overlap.
#   - Does NOT mutate data.
#   - Does NOT change Cell 15.1 results.
#   - Does NOT make final Q6 privacy decision.
#
# Outputs:
#   reports/cell15_1a_no_copy_blocker_forensic_diagnostic.csv
#   reports/cell15_1a_no_copy_blocker_top_matches.csv
#   reports/cell15_1a_no_copy_status_reason_summary.csv
#   reports/cell15_1a_no_copy_next_action_queue.csv
#   reports/cell15_1a_no_copy_forensic_contract.json
#   artifacts/contracts/cell15_1a_no_copy_forensic_contract_v1_0_THESIS.json
# ==========================================================

log("--- START: Cell 15.1a - Q6 no-copy blocker forensic diagnostic (v1.0 strict diagnostic-only) ---")

import os
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

_required_151a = [
    "CFG", "log", "OUTDIR", "OUT_SYN", "REPORT_DIR", "SEED",
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "CELL15_1_NO_COPY_WINDOW_METRICS_DF",
    "CELL15_1_NO_COPY_TOP_MATCHES_DF",
    "CELL15_1_NO_COPY_CONTRACT",
]
_missing_151a = [k for k in _required_151a if k not in globals()]
if _missing_151a:
    raise RuntimeError(f"[Cell15.1a] Missing required globals: {_missing_151a}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_151a = str(OUT_SYN)
REPORT_DIR_ACTIVE_151a = str(REPORT_DIR)

def _resolve_project_root_151a(outdir, report_dir, out_syn):
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
    raise RuntimeError("[Cell15.1a] Could not resolve canonical project root.")

PROJECT_ROOT_151a = _resolve_project_root_151a(OUTDIR, REPORT_DIR_ACTIVE_151a, OUT_SYN_ACTIVE_151a)
REPORT_DIR_151a = os.path.join(PROJECT_ROOT_151a, "reports")
ARTDIR_151a = os.path.join(PROJECT_ROOT_151a, "artifacts")
CONTRACT_DIR_151a = os.path.join(ARTDIR_151a, "contracts")
os.makedirs(REPORT_DIR_151a, exist_ok=True)
os.makedirs(ARTDIR_151a, exist_ok=True)
os.makedirs(CONTRACT_DIR_151a, exist_ok=True)

CELL151A_VERSION = "cell15_1a_q6_no_copy_blocker_forensic_v1_0"

CFG["cell15_1a_version"] = CELL151A_VERSION
CFG["cell15_1a_diagnostic_only"] = True
CFG["cell15_1a_TEST_real_values_used_for_privacy_reference"] = False
CFG["cell15_1a_synthetic_values_mutated"] = False
CFG["cell15_1a_selection_done_here"] = False
CFG["cell15_1a_generator_fit_done_here"] = False
CFG["cell15_1a_materialization_done_here"] = False
CFG["cell15_1a_privacy_decision_done_here"] = False

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _json_sanitize_151a(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_151a(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_151a(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_151a(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_151a(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_151a(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_151a(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_151a(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_151a(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_151a(payload), f, indent=2, sort_keys=True)

def _sha256_file_151a(path):
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_151a(obj, name, expected_substring):
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.1a] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.1a] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
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
version150_151a = _require_contract_version_151a(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
version151_151a = _require_contract_version_151a(
    CELL15_1_NO_COPY_CONTRACT,
    "CELL15_1_NO_COPY_CONTRACT",
    "cell15_1_q6_no_copy_window_audit_v1_1",
)

strict150 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
strict151 = CELL15_1_NO_COPY_CONTRACT.get("strict_contract", {})

if str(strict150.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.1a] Cell 15.0 does not carry Q4 blocked_no_promotion.")
if bool(strict150.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.1a] Cell 15.0 used Q4 coupled artifacts unexpectedly.")
if bool(strict151.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell15.1a] Cell 15.1 indicates synthetic mutation.")
if not bool(strict151.get("no_copy_window_audit_done_here", False)):
    raise RuntimeError("[Cell15.1a] Cell 15.1 did not complete no-copy audit.")

metrics = CELL15_1_NO_COPY_WINDOW_METRICS_DF.copy()
top = CELL15_1_NO_COPY_TOP_MATCHES_DF.copy()

if len(metrics) == 0:
    raise RuntimeError("[Cell15.1a] Empty Cell 15.1 no-copy metrics.")

# ----------------------------------------------------------
# Analyze reasons and status
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

    exact_nonzero_rate = _safe_float(r.get("exact_nonzero_match_rate"), np.nan)
    near_all_rate = _safe_float(r.get("near_copy_rate_all_windows", r.get("near_copy_rate")), np.nan)
    near_nonzero_rate = _safe_float(r.get("near_copy_nonzero_rate"), np.nan)
    gate_near_rate = _safe_float(r.get("near_copy_rate_used_for_gate"), np.nan)
    p99_all = _safe_float(r.get("p99_similarity_all_windows", r.get("p99_similarity")), np.nan)
    p99_nonzero = _safe_float(r.get("p99_nonzero_similarity"), np.nan)
    p99_gate = _safe_float(r.get("p99_similarity_used_for_gate"), np.nan)
    syn_all_zero = _safe_int(r.get("syn_all_zero_windows_n"), 0)
    syn_nonzero = _safe_int(r.get("syn_nonzero_windows_n"), 0)

    if status == "blocker":
        if "exact_nonzero" in reasons:
            root_cause = "exact_nonzero_window_copy"
            action = "inspect_exact_nonzero_matches; likely real copy-risk blocker unless columns/windows are deterministic"
        elif "near_copy_nonzero" in reasons:
            root_cause = "nonzero_near_copy_rate"
            action = "inspect_top_nonzero_matches; decide if deterministic/structural or true copy-risk"
        elif "p99" in reasons:
            root_cause = "p99_nonzero_similarity"
            action = "inspect p99/top matches; likely threshold-sensitive"
        else:
            root_cause = "carried_or_unknown_blocker"
            action = "inspect role metrics and top matches"
    elif status == "warning":
        root_cause = "warning_only"
        action = "carry to Q6 summary as warning unless later DCR/NNDR also blocks"
    elif status == "pass":
        root_cause = "no_copy_pass"
        action = "no action"
    else:
        root_cause = "not_evaluable"
        action = "decide if role should be excluded or reconfigured"

    diag_rows.append({
        "role_group": role,
        "status": status,
        "reasons": reasons,
        "root_cause_class": root_cause,
        "suggested_next_action": action,
        "selected_cols_n": _safe_int(r.get("selected_cols_n"), 0),
        "window_len": _safe_int(r.get("window_len"), 0),
        "syn_windows_n": _safe_int(r.get("syn_windows_n"), 0),
        "syn_all_zero_windows_n": syn_all_zero,
        "syn_nonzero_windows_n": syn_nonzero,
        "exact_match_n": _safe_int(r.get("exact_match_n"), 0),
        "exact_match_rate": _safe_float(r.get("exact_match_rate"), np.nan),
        "exact_nonzero_match_n": _safe_int(r.get("exact_nonzero_match_n"), 0),
        "exact_nonzero_match_rate": exact_nonzero_rate,
        "near_copy_rate_all_windows": near_all_rate,
        "near_copy_nonzero_n": _safe_int(r.get("near_copy_nonzero_n"), 0),
        "near_copy_nonzero_rate": near_nonzero_rate,
        "near_copy_rate_used_for_gate": gate_near_rate,
        "max_similarity": _safe_float(r.get("max_similarity"), np.nan),
        "p99_similarity_all_windows": p99_all,
        "p99_nonzero_similarity": p99_nonzero,
        "p99_similarity_used_for_gate": p99_gate,
        "sparse_all_zero_windows_not_blockers": bool(r.get("sparse_all_zero_windows_not_blockers", True)),
    })

diag = pd.DataFrame(diag_rows)

blocker_roles = diag[diag["status"].astype(str).eq("blocker")]["role_group"].astype(str).tolist()
warning_roles = diag[diag["status"].astype(str).eq("warning")]["role_group"].astype(str).tolist()

# Top matches for blocker/warning roles only.
if len(top) and "role_group" in top.columns:
    focus_top = top[top["role_group"].astype(str).isin(blocker_roles + warning_roles)].copy()
else:
    focus_top = pd.DataFrame()

# Add a nonzero top-match focus if columns exist.
if len(focus_top):
    if "is_syn_all_zero_window" in focus_top.columns:
        focus_top["top_match_scope"] = np.where(
            focus_top["is_syn_all_zero_window"].fillna(False).astype(bool),
            "all_zero_diagnostic",
            "nonzero_privacy_relevant",
        )
    else:
        focus_top["top_match_scope"] = "unknown"

# Status/reason summary.
summary_rows = []
for status, sub in diag.groupby("status", dropna=False):
    summary_rows.append({
        "status": str(status),
        "roles_n": int(len(sub)),
        "roles": "|".join(sub["role_group"].astype(str).tolist()),
        "mean_exact_nonzero_match_rate": float(pd.to_numeric(sub["exact_nonzero_match_rate"], errors="coerce").mean()),
        "mean_near_copy_nonzero_rate": float(pd.to_numeric(sub["near_copy_nonzero_rate"], errors="coerce").mean()),
        "mean_p99_nonzero_similarity": float(pd.to_numeric(sub["p99_nonzero_similarity"], errors="coerce").mean()),
    })

for reason, n in reason_counter.items():
    summary_rows.append({
        "status": f"reason::{reason}",
        "roles_n": int(n),
        "roles": "",
        "mean_exact_nonzero_match_rate": np.nan,
        "mean_near_copy_nonzero_rate": np.nan,
        "mean_p99_nonzero_similarity": np.nan,
    })

status_summary = pd.DataFrame(summary_rows)

# Action queue: blockers first, then warnings.
severity_order = {"blocker": 0, "warning": 1, "not_evaluable": 2, "pass": 3}
queue = diag.copy()
queue["_order"] = queue["status"].map(severity_order).fillna(99)
queue = queue.sort_values(
    ["_order", "near_copy_rate_used_for_gate", "exact_nonzero_match_rate", "p99_similarity_used_for_gate"],
    ascending=[True, False, False, False],
).drop(columns=["_order"])

# ----------------------------------------------------------
# Save outputs
# ----------------------------------------------------------
diag_csv = os.path.join(REPORT_DIR_151a, "cell15_1a_no_copy_blocker_forensic_diagnostic.csv")
top_csv = os.path.join(REPORT_DIR_151a, "cell15_1a_no_copy_blocker_top_matches.csv")
summary_csv = os.path.join(REPORT_DIR_151a, "cell15_1a_no_copy_status_reason_summary.csv")
queue_csv = os.path.join(REPORT_DIR_151a, "cell15_1a_no_copy_next_action_queue.csv")
contract_json = os.path.join(REPORT_DIR_151a, "cell15_1a_no_copy_forensic_contract.json")
canonical_contract_json = os.path.join(CONTRACT_DIR_151a, "cell15_1a_no_copy_forensic_contract_v1_0_THESIS.json")

diag.to_csv(diag_csv, index=False)
focus_top.to_csv(top_csv, index=False)
status_summary.to_csv(summary_csv, index=False)
queue.to_csv(queue_csv, index=False)

contract = {
    "cell": "15.1a",
    "version": CELL151A_VERSION,
    "role": "q6_no_copy_blocker_forensic_diagnostic",
    "quality_dimension": "Q6_privacy_no_copy_release_safety",
    "upstream_contract_versions": {
        "cell15_0": version150_151a,
        "cell15_1": version151_151a,
    },
    "summary": {
        "roles_total": int(len(diag)),
        "blocker_roles_n": int(len(blocker_roles)),
        "warning_roles_n": int(len(warning_roles)),
        "blocker_roles": blocker_roles,
        "warning_roles": warning_roles,
        "reason_counts": dict(reason_counter),
        "ready_for_cell15_2": bool(len(blocker_roles) == 0),
        "recommendation": (
            "Do not finalize Q6; inspect blocker roles before final release summary."
            if blocker_roles else
            "No no-copy blockers; Cell 15.2 can proceed as additional privacy evidence."
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

_write_json_151a(contract_json, contract)
_write_json_151a(canonical_contract_json, contract)

globals()["CELL151A_VERSION"] = CELL151A_VERSION
globals()["CELL15_1A_NO_COPY_BLOCKER_FORENSIC_DF"] = diag
globals()["CELL15_1A_NO_COPY_BLOCKER_TOP_MATCHES_DF"] = focus_top
globals()["CELL15_1A_NO_COPY_STATUS_REASON_SUMMARY_DF"] = status_summary
globals()["CELL15_1A_NO_COPY_NEXT_ACTION_QUEUE_DF"] = queue
globals()["CELL15_1A_NO_COPY_FORENSIC_CONTRACT"] = contract
globals()["CELL15_1A_NO_COPY_BLOCKER_FORENSIC_CSV"] = diag_csv
globals()["CELL15_1A_NO_COPY_BLOCKER_TOP_MATCHES_CSV"] = top_csv
globals()["CELL15_1A_NO_COPY_STATUS_REASON_SUMMARY_CSV"] = summary_csv
globals()["CELL15_1A_NO_COPY_NEXT_ACTION_QUEUE_CSV"] = queue_csv
globals()["CELL15_1A_NO_COPY_FORENSIC_CONTRACT_JSON"] = contract_json
globals()["CELL15_1A_NO_COPY_FORENSIC_CONTRACT_CANONICAL_JSON"] = canonical_contract_json

log(
    "[Cell15.1a] No-copy blocker forensic diagnostic complete | "
    f"roles={len(diag)} | blocker_roles={blocker_roles} | warning_roles={warning_roles}"
)
log(f"[Cell15.1a] Reason counts | {dict(reason_counter)}")
log(f"[Cell15.1a] Saved diagnostic: {diag_csv} | rows={len(diag)}")
log(f"[Cell15.1a] Saved focused top matches: {top_csv} | rows={len(focus_top)}")
log(
    "[Cell15.1a] Contract flags | diagnostic_only=True | Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | TEST_real_values_used_for_privacy_reference=False | "
    "synthetic_values_mutated=False | privacy_decision_done_here=False"
)
log("--- END: Cell 15.1a - Q6 no-copy blocker forensic diagnostic (v1.0 strict diagnostic-only) ---")