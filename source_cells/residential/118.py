# ==========================================================
# CELL 14.6a - A0 fatal blocker forensic diagnostic
# v1.0 STUDY-THESIS strict QA-only diagnostic
#
# Role:
#   - Diagnose why Cell 14.6 A0 candidate failed promotion.
#   - Inspect only saved candidate QA metrics, pre/post deltas, and contracts.
#   - Do NOT mutate synthetic values.
#   - Do NOT promote candidate outputs.
#   - Do NOT fit/select/materialize anything.
#
# Outputs:
#   reports/cell14_6a_A0_fatal_blocker_diagnostic.csv
#   reports/cell14_6a_A0_status_reason_summary.csv
#   reports/cell14_6a_A0_metric_threshold_audit.csv
#   reports/cell14_6a_A0_repair_next_action_queue.csv
#   reports/cell14_6a_A0_fatal_diagnostic_contract.json
# ==========================================================

log("--- START: Cell 14.6a - A0 fatal blocker forensic diagnostic (v1.0 strict QA-only) ---")

import os
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

_required_146a = [
    "CFG", "log", "OUTDIR", "OUT_SYN", "REPORT_DIR",
]
_missing_146a = [k for k in _required_146a if k not in globals()]
if _missing_146a:
    raise RuntimeError(f"[Cell14.6a] Missing required globals: {_missing_146a}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_146a = str(OUT_SYN)
REPORT_DIR_ACTIVE_146a = str(REPORT_DIR)

def _resolve_project_root_146a(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        if "q6_public_reaudit" in parts:
            idx = parts.index("q6_public_reaudit")
            candidates.append(os.sep.join(parts[:idx]))
        else:
            base = os.path.basename(p)
            if base in {"reports", "synthetic", "artifacts"}:
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
    raise RuntimeError("[Cell14.6a] Could not resolve canonical project root.")

PROJECT_ROOT_146a = _resolve_project_root_146a(OUTDIR, REPORT_DIR_ACTIVE_146a, OUT_SYN_ACTIVE_146a)
REPORT_DIR_146a = os.path.join(PROJECT_ROOT_146a, "reports")
ARTDIR_146a = os.path.join(PROJECT_ROOT_146a, "artifacts")
CONTRACT_DIR_146a = os.path.join(ARTDIR_146a, "contracts")

os.makedirs(REPORT_DIR_146a, exist_ok=True)
os.makedirs(ARTDIR_146a, exist_ok=True)
os.makedirs(CONTRACT_DIR_146a, exist_ok=True)

CELL146A_VERSION = "cell14_6a_A0_fatal_blocker_forensic_diagnostic_v1_0"

CFG["cell14_6a_version"] = CELL146A_VERSION
CFG["cell14_6a_QA_diagnostic_only"] = True
CFG["cell14_6a_TEST_real_values_used_for_QA_diagnostic_only"] = False
CFG["cell14_6a_synthetic_values_mutated"] = False
CFG["cell14_6a_selection_done_here"] = False
CFG["cell14_6a_generator_fit_done_here"] = False
CFG["cell14_6a_materialization_done_here"] = False
CFG["cell14_6a_promotion_done_here"] = False

# Thresholds mirror 14.6.
CFG.setdefault("cell14_6_eta_similarity_pass", float(CFG.get("cell13_6_eta_similarity_pass", 0.70)))
CFG.setdefault("cell14_6_eta_similarity_warning", float(CFG.get("cell13_6_eta_similarity_warning", 0.50)))
CFG.setdefault("cell14_6_profile_similarity_pass", float(CFG.get("cell13_6_profile_similarity_pass", 0.70)))
CFG.setdefault("cell14_6_profile_similarity_warning", float(CFG.get("cell13_6_profile_similarity_warning", 0.50)))
CFG.setdefault("cell14_6_lag_peak_error_pass", float(CFG.get("cell13_6_lag_peak_error_pass", 2)))
CFG.setdefault("cell14_6_lag_peak_error_warning", float(CFG.get("cell13_6_lag_peak_error_warning", 5)))
CFG.setdefault("cell14_6_response_window_rate_error_pass", float(CFG.get("cell13_6_response_window_rate_error_pass", 0.10)))
CFG.setdefault("cell14_6_response_window_rate_error_warning", float(CFG.get("cell13_6_response_window_rate_error_warning", 0.25)))

# Input paths.
metrics_csv = globals().get(
    "CELL14_6_A0_CANDIDATE_Q4_METRICS_CSV",
    os.path.join(REPORT_DIR_146a, "cell14_6_A0_candidate_manifest_q4_pair_metrics.csv"),
)
delta_csv = globals().get(
    "CELL14_6_A0_CANDIDATE_Q4_DELTA_CSV",
    os.path.join(REPORT_DIR_146a, "cell14_6_A0_candidate_manifest_q4_prepost_delta.csv"),
)
candidate_contract_json = globals().get(
    "CELL14_6_A0_CANDIDATE_CONTRACT_JSON",
    os.path.join(REPORT_DIR_146a, "cell14_6_A0_candidate_contract.json"),
)
accepted_contract_json = globals().get(
    "CELL14_6_A0_ACCEPTED_CONTRACT_JSON",
    os.path.join(REPORT_DIR_146a, "cell14_6_A0_ACCEPTED_contract.json"),
)

for p, label in [
    (metrics_csv, "candidate metrics"),
    (candidate_contract_json, "candidate contract"),
    (accepted_contract_json, "accepted contract"),
]:
    if not p or not os.path.exists(str(p)):
        raise RuntimeError(f"[Cell14.6a] Missing {label}: {p}")

metrics = pd.read_csv(metrics_csv)
delta = pd.read_csv(delta_csv) if os.path.exists(str(delta_csv)) else pd.DataFrame()

with open(candidate_contract_json, "r", encoding="utf-8") as f:
    candidate_contract = json.load(f)

with open(accepted_contract_json, "r", encoding="utf-8") as f:
    accepted_contract = json.load(f)

if len(metrics) == 0:
    raise RuntimeError("[Cell14.6a] Empty A0 candidate metrics.")

required_cols = [
    "repair_candidate_id", "anchor_col", "protocol_col",
    "A0_q4_status", "A0_q4_reasons", "A0_q4_publication_blocker",
    "ETA_similarity", "lag_peak_error", "response_window_rate_error",
    "manifest_window_profile_similarity",
]
missing_cols = [c for c in required_cols if c not in metrics.columns]
if missing_cols:
    raise RuntimeError(f"[Cell14.6a] Candidate metrics missing columns: {missing_cols}")

def _safe_float_146a(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _sha256_file_146a(path):
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_json_146a(path, payload):
    def san(o):
        if isinstance(o, dict):
            return {str(k): san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [san(v) for v in o]
        if isinstance(o, tuple):
            return [san(v) for v in o]
        if isinstance(o, np.ndarray):
            return san(o.tolist())
        if isinstance(o, pd.DataFrame):
            return san(o.to_dict("records"))
        if isinstance(o, pd.Series):
            return san(o.to_dict())
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            x = float(o)
            return None if not np.isfinite(x) else x
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o
    with open(path, "w", encoding="utf-8") as f:
        json.dump(san(payload), f, indent=2, sort_keys=True)

# Output paths.
fatal_diag_csv = os.path.join(REPORT_DIR_146a, "cell14_6a_A0_fatal_blocker_diagnostic.csv")
status_reason_csv = os.path.join(REPORT_DIR_146a, "cell14_6a_A0_status_reason_summary.csv")
threshold_audit_csv = os.path.join(REPORT_DIR_146a, "cell14_6a_A0_metric_threshold_audit.csv")
action_queue_csv = os.path.join(REPORT_DIR_146a, "cell14_6a_A0_repair_next_action_queue.csv")
contract_json = os.path.join(REPORT_DIR_146a, "cell14_6a_A0_fatal_diagnostic_contract.json")
canonical_contract_json = os.path.join(CONTRACT_DIR_146a, "cell14_6a_A0_fatal_diagnostic_contract_v1_0_THESIS.json")

eta_pass = float(CFG.get("cell14_6_eta_similarity_pass", 0.70))
eta_warn = float(CFG.get("cell14_6_eta_similarity_warning", 0.50))
profile_pass = float(CFG.get("cell14_6_profile_similarity_pass", 0.70))
profile_warn = float(CFG.get("cell14_6_profile_similarity_warning", 0.50))
lag_pass = float(CFG.get("cell14_6_lag_peak_error_pass", 2))
lag_warn = float(CFG.get("cell14_6_lag_peak_error_warning", 5))
resp_pass = float(CFG.get("cell14_6_response_window_rate_error_pass", 0.10))
resp_warn = float(CFG.get("cell14_6_response_window_rate_error_warning", 0.25))

audit_rows = []
for _, r in metrics.iterrows():
    eta = _safe_float_146a(r.get("ETA_similarity"), np.nan)
    prof = _safe_float_146a(r.get("manifest_window_profile_similarity"), np.nan)
    lag = _safe_float_146a(r.get("lag_peak_error"), np.nan)
    resp = _safe_float_146a(r.get("response_window_rate_error"), np.nan)

    metric_flags = []
    fatal_flags = []
    warning_flags = []

    if np.isfinite(eta):
        if eta < eta_warn:
            fatal_flags.append("ETA_similarity_fatal")
        elif eta < eta_pass:
            warning_flags.append("ETA_similarity_warning")
    else:
        warning_flags.append("ETA_similarity_not_finite")

    if np.isfinite(prof):
        if prof < profile_warn:
            fatal_flags.append("manifest_profile_similarity_fatal")
        elif prof < profile_pass:
            warning_flags.append("manifest_profile_similarity_warning")
    else:
        warning_flags.append("manifest_profile_similarity_not_finite")

    if np.isfinite(lag):
        if lag > lag_warn:
            fatal_flags.append("lag_peak_error_fatal")
        elif lag > lag_pass:
            warning_flags.append("lag_peak_error_warning")

    if np.isfinite(resp):
        if resp > resp_warn:
            fatal_flags.append("response_window_rate_error_fatal")
        elif resp > resp_pass:
            warning_flags.append("response_window_rate_error_warning")

    # Suggested action is intentionally conservative.
    if "response_window_rate_error_fatal" in fatal_flags and len(fatal_flags) == 1:
        suggested = "inspect_response_window_threshold_and_profile_amplitude; do_not_promote"
    elif "ETA_similarity_fatal" in fatal_flags or "manifest_profile_similarity_fatal" in fatal_flags:
        suggested = "profile_shape_failure; inspect_anchor_protocol_pair_before_any_new_materializer"
    elif "lag_peak_error_fatal" in fatal_flags:
        suggested = "timing_failure; inspect_lag_window_and_driver_event_alignment"
    elif fatal_flags:
        suggested = "multi_metric_fatal; keep_blocked_and_inspect_pair"
    elif warning_flags:
        suggested = "warning_only; acceptable_only_if_promotion_gate_policy_allows_warning"
    else:
        suggested = "pass"

    audit_rows.append({
        "repair_candidate_id": str(r.get("repair_candidate_id", "")),
        "anchor_col": str(r.get("anchor_col", "")),
        "anchor_name": str(r.get("anchor_name", "")),
        "protocol_col": str(r.get("protocol_col", "")),
        "protocol_tier": str(r.get("protocol_tier", "")),
        "A0_q4_status": str(r.get("A0_q4_status", "")),
        "A0_q4_reasons": str(r.get("A0_q4_reasons", "")),
        "A0_q4_publication_blocker": bool(r.get("A0_q4_publication_blocker", False)),
        "ETA_similarity": eta,
        "ETA_margin_to_pass": eta - eta_pass if np.isfinite(eta) else np.nan,
        "ETA_margin_to_warning": eta - eta_warn if np.isfinite(eta) else np.nan,
        "lag_peak_error": lag,
        "lag_peak_error_margin_to_pass": lag_pass - lag if np.isfinite(lag) else np.nan,
        "lag_peak_error_margin_to_warning": lag_warn - lag if np.isfinite(lag) else np.nan,
        "response_window_rate_error": resp,
        "response_window_error_margin_to_pass": resp_pass - resp if np.isfinite(resp) else np.nan,
        "response_window_error_margin_to_warning": resp_warn - resp if np.isfinite(resp) else np.nan,
        "manifest_window_profile_similarity": prof,
        "profile_margin_to_pass": prof - profile_pass if np.isfinite(prof) else np.nan,
        "profile_margin_to_warning": prof - profile_warn if np.isfinite(prof) else np.nan,
        "fatal_flags_recomputed": "|".join(fatal_flags),
        "warning_flags_recomputed": "|".join(warning_flags),
        "suggested_next_action": suggested,
    })

threshold_audit = pd.DataFrame(audit_rows)
fatal_diag = threshold_audit[threshold_audit["A0_q4_publication_blocker"].astype(bool)].copy()
action_queue = threshold_audit.sort_values(
    ["A0_q4_publication_blocker", "A0_q4_status", "response_window_rate_error", "ETA_similarity"],
    ascending=[False, True, False, True],
).copy()

# Status/reason summary.
reason_counter = Counter()
for reason in metrics["A0_q4_reasons"].astype(str):
    for part in reason.split("|"):
        part = part.strip()
        if part:
            reason_counter[part] += 1

status_rows = []
for status, sub in metrics.groupby("A0_q4_status", dropna=False):
    status_rows.append({
        "A0_q4_status": str(status),
        "rows": int(len(sub)),
        "publication_blocker_n": int(sub["A0_q4_publication_blocker"].fillna(False).astype(bool).sum()),
        "mean_ETA_similarity": float(pd.to_numeric(sub["ETA_similarity"], errors="coerce").mean()),
        "mean_lag_peak_error": float(pd.to_numeric(sub["lag_peak_error"], errors="coerce").mean()),
        "mean_response_window_rate_error": float(pd.to_numeric(sub["response_window_rate_error"], errors="coerce").mean()),
        "mean_manifest_profile_similarity": float(pd.to_numeric(sub["manifest_window_profile_similarity"], errors="coerce").mean()),
    })

for reason, n in reason_counter.items():
    status_rows.append({
        "A0_q4_status": f"reason::{reason}",
        "rows": int(n),
        "publication_blocker_n": np.nan,
        "mean_ETA_similarity": np.nan,
        "mean_lag_peak_error": np.nan,
        "mean_response_window_rate_error": np.nan,
        "mean_manifest_profile_similarity": np.nan,
    })

status_reason = pd.DataFrame(status_rows)

threshold_audit.to_csv(threshold_audit_csv, index=False)
fatal_diag.to_csv(fatal_diag_csv, index=False)
status_reason.to_csv(status_reason_csv, index=False)
action_queue.to_csv(action_queue_csv, index=False)

pairs_total = int(len(metrics))
pass_n = int((metrics["A0_q4_status"].astype(str) == "pass").sum())
warning_n = int((metrics["A0_q4_status"].astype(str) == "warning").sum())
fatal_n = int((metrics["A0_q4_status"].astype(str) == "fatal").sum())
blocker_n = int(metrics["A0_q4_publication_blocker"].fillna(False).astype(bool).sum())

accepted_flag = bool(accepted_contract.get("accepted", False))
candidate_summary = candidate_contract.get("summary", {})
accepted_summary = accepted_contract.get("summary", {})

contract = {
    "cell": "14.6a",
    "version": CELL146A_VERSION,
    "role": "A0_fatal_blocker_forensic_diagnostic",
    "quality_dimension": "Q4_cross_modal_consistency_A0_diagnostic",
    "input_paths": {
        "candidate_metrics_csv": str(metrics_csv),
        "candidate_delta_csv": str(delta_csv),
        "candidate_contract_json": str(candidate_contract_json),
        "accepted_contract_json": str(accepted_contract_json),
    },
    "promotion_state": {
        "cell14_6_accepted": accepted_flag,
        "pairs_total": pairs_total,
        "pass_n": pass_n,
        "warning_n": warning_n,
        "fatal_n": fatal_n,
        "publication_blocker_n": blocker_n,
        "promotion_gate_expected": "8/2/6/0/0",
        "promotion_gate_observed": f"{pairs_total}/{pass_n}/{warning_n}/{fatal_n}/{blocker_n}",
        "promotion_should_remain_blocked": bool(not accepted_flag),
    },
    "thresholds": {
        "eta_pass": eta_pass,
        "eta_warning": eta_warn,
        "profile_pass": profile_pass,
        "profile_warning": profile_warn,
        "lag_pass": lag_pass,
        "lag_warning": lag_warn,
        "response_pass": resp_pass,
        "response_warning": resp_warn,
    },
    "fatal_reason_counts": dict(reason_counter),
    "strict_contract": {
        "QA_diagnostic_only": True,
        "TEST_real_values_used_for_QA_diagnostic_only": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "promotion_done_here": False,
    },
    "outputs": {
        "fatal_diag_csv": fatal_diag_csv,
        "status_reason_csv": status_reason_csv,
        "threshold_audit_csv": threshold_audit_csv,
        "action_queue_csv": action_queue_csv,
        "contract_json": contract_json,
        "canonical_contract_json": canonical_contract_json,
    },
}

_write_json_146a(contract_json, contract)
_write_json_146a(canonical_contract_json, contract)

globals()["CELL146A_VERSION"] = CELL146A_VERSION
globals()["CELL14_6A_A0_FATAL_BLOCKER_DIAGNOSTIC_DF"] = fatal_diag
globals()["CELL14_6A_A0_STATUS_REASON_SUMMARY_DF"] = status_reason
globals()["CELL14_6A_A0_METRIC_THRESHOLD_AUDIT_DF"] = threshold_audit
globals()["CELL14_6A_A0_REPAIR_NEXT_ACTION_QUEUE_DF"] = action_queue
globals()["CELL14_6A_A0_FATAL_DIAGNOSTIC_CONTRACT"] = contract

globals()["CELL14_6A_A0_FATAL_BLOCKER_DIAGNOSTIC_CSV"] = fatal_diag_csv
globals()["CELL14_6A_A0_STATUS_REASON_SUMMARY_CSV"] = status_reason_csv
globals()["CELL14_6A_A0_METRIC_THRESHOLD_AUDIT_CSV"] = threshold_audit_csv
globals()["CELL14_6A_A0_REPAIR_NEXT_ACTION_QUEUE_CSV"] = action_queue_csv
globals()["CELL14_6A_A0_FATAL_DIAGNOSTIC_CONTRACT_JSON"] = contract_json
globals()["CELL14_6A_A0_FATAL_DIAGNOSTIC_CONTRACT_CANONICAL_JSON"] = canonical_contract_json

log(
    "[Cell14.6a] A0 fatal blocker diagnostic complete | "
    f"pairs_total={pairs_total} | pass={pass_n} | warning={warning_n} | "
    f"fatal={fatal_n} | blockers={blocker_n} | accepted={accepted_flag}"
)
log(f"[Cell14.6a] Fatal/blocker rows saved: {fatal_diag_csv} | rows={len(fatal_diag)}")
log(f"[Cell14.6a] Status/reason summary | {status_reason.to_dict('records')}")
log(f"[Cell14.6a] Action queue saved: {action_queue_csv} | rows={len(action_queue)}")
log(
    "[Cell14.6a] Contract flags | "
    "QA_diagnostic_only=True | TEST_real_values_used_for_QA_diagnostic_only=False | "
    "synthetic_values_mutated=False | selection_done_here=False | "
    "generator_fit_done_here=False | materialization_done_here=False | promotion_done_here=False"
)
log("--- END: Cell 14.6a - A0 fatal blocker forensic diagnostic (v1.0 strict QA-only) ---")