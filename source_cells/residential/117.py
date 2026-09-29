# %% CELL 14.6R1 — Frozen A0 terminal TEST evidence gate
# Purpose:
#   Interpret Cell 14.6 terminal TEST QA under the policy frozen by Cell 14.6R0.
#
# Scientific contract:
#   - Does not tune, prune, or repair from TEST.
#   - Does not promote artifacts by itself.
#   - If zero TEST blockers are observed without a fresh untouched holdout, the result is
#     development evidence, not final publication evidence.

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log("--- START: Cell 14.6R1 - Frozen A0 terminal TEST evidence gate ---")

_required_146r1 = ["CFG", "log", "OUTDIR", "REPORT_DIR", "CELL14_6_A0_CONTRACT"]
_missing_146r1 = [k for k in _required_146r1 if k not in globals()]
if _missing_146r1:
    raise RuntimeError(f"[Cell14.6R1] Missing required globals: {_missing_146r1}")

OUTDIR_R1 = Path(str(OUTDIR)).expanduser().resolve()
REPORT_DIR_R1 = Path(str(REPORT_DIR)).expanduser().resolve()
CONTRACT_DIR_R1 = OUTDIR_R1 / "artifacts" / "contracts"
REPORT_DIR_R1.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_R1.mkdir(parents=True, exist_ok=True)

def _sanitize_146r1(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_146r1(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_146r1(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _sanitize_146r1(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _sanitize_146r1(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _sanitize_146r1(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_146r1(path, payload):
    Path(path).write_text(json.dumps(_sanitize_146r1(payload), indent=2, sort_keys=True), encoding="utf-8")

def _load_metrics_146r1():
    if "CELL14_6_A0_Q4_PAIR_METRICS_DF" in globals() and isinstance(CELL14_6_A0_Q4_PAIR_METRICS_DF, pd.DataFrame):
        return CELL14_6_A0_Q4_PAIR_METRICS_DF.copy()
    p = Path(str(CELL14_6_A0_CONTRACT.get("outputs", {}).get("q4_metrics_csv", REPORT_DIR_R1 / "cell14_6_A0_manifest_q4_pair_metrics.csv")))
    if p.exists():
        return pd.read_csv(p)
    p2 = REPORT_DIR_R1 / "cell14_6_A0_candidate_manifest_q4_pair_metrics.csv"
    if p2.exists():
        return pd.read_csv(p2)
    raise RuntimeError("[Cell14.6R1] Could not locate Cell 14.6 A0 pair metrics.")

metrics = _load_metrics_146r1()
metrics.columns = metrics.columns.astype(str)

r0_contract = globals().get("CELL14_6R0_A0_POLICY_CONTRACT", {})
selected_on_val = bool(
    isinstance(r0_contract, dict)
    and r0_contract.get("strict_contract", {}).get("selected_on_VAL", False)
    and not r0_contract.get("strict_contract", {}).get("TEST_real_values_used", True)
)
if not selected_on_val:
    raise RuntimeError("[Cell14.6R1] Missing valid Cell 14.6R0 TRAIN/VAL-only policy contract.")

# ---------------------------------------------------------------------
# Robust Q4 status recovery.
# Cell 14.6 may export pair metrics without an explicit q4_status/status
# column. In that case, derive status from the predeclared Q4 gates:
#   blocker/fatal if:
#       ETA < 0.50 OR manifest/profile similarity < 0.50
#       OR lag_peak_error > 5 s
#       OR response_window_rate_error > 0.25
#   warning if:
#       ETA < 0.70 OR manifest/profile similarity < 0.70
#       OR lag_peak_error > 2 s
#       OR response_window_rate_error > 0.10
#   pass otherwise.
# This is TEST QA interpretation only; it does not tune, select, prune, or repair.
# ---------------------------------------------------------------------

def _find_col_146r1(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if str(cand).lower() in lower_map:
            return lower_map[str(cand).lower()]
    return None

def _as_bool_146r1(s):
    if s is None:
        return None
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .isin(["1", "true", "yes", "y", "fatal", "blocker", "blocked"])
    )

status_col = _find_col_146r1(
    metrics,
    [
        "q4_status",
        "status",
        "qa_status",
        "pair_status",
        "readiness_status",
        "final_status",
        "publication_status",
        "q4_readiness",
    ],
)

if status_col is not None:
    metrics["q4_status_derived_14_6R1"] = metrics[status_col].astype(str).str.lower()
else:
    eta_col = _find_col_146r1(
        metrics,
        [
            "ETA_similarity",
            "eta_similarity",
            "q4_eta_similarity",
            "event_triggered_alignment_similarity",
        ],
    )
    manifest_col = _find_col_146r1(
        metrics,
        [
            "manifest_profile_similarity",
            "profile_similarity",
            "manifest_similarity",
            "window_profile_similarity",
            "manifest_window_profile_similarity",
        ],
    )
    lag_col = _find_col_146r1(
        metrics,
        [
            "lag_peak_error",
            "lag_peak_error_s",
            "lag_error",
            "peak_lag_error",
        ],
    )
    resp_col = _find_col_146r1(
        metrics,
        [
            "response_window_rate_error",
            "response_rate_error",
            "resp_window_rate_error",
            "response_window_error",
        ],
    )

    publication_blocker_col = _find_col_146r1(
        metrics,
        [
            "publication_blocker",
            "q4_publication_blocker",
            "blocker",
            "is_blocker",
            "fatal",
            "q4_blocker",
        ],
    )

    if all(c is None for c in [eta_col, manifest_col, lag_col, resp_col, publication_blocker_col]):
        raise RuntimeError(
            "[Cell14.6R1] Cannot derive A0 Q4 status. "
            f"Available metric columns are: {list(metrics.columns)}"
        )

    eta = pd.to_numeric(metrics[eta_col], errors="coerce") if eta_col else pd.Series(np.nan, index=metrics.index)
    manifest = pd.to_numeric(metrics[manifest_col], errors="coerce") if manifest_col else pd.Series(np.nan, index=metrics.index)
    lag = pd.to_numeric(metrics[lag_col], errors="coerce") if lag_col else pd.Series(np.nan, index=metrics.index)
    resp = pd.to_numeric(metrics[resp_col], errors="coerce") if resp_col else pd.Series(np.nan, index=metrics.index)

    explicit_blocker = (
        _as_bool_146r1(metrics[publication_blocker_col])
        if publication_blocker_col is not None
        else pd.Series(False, index=metrics.index)
    )

    fatal_mask = (
        explicit_blocker
        | eta.lt(0.50).fillna(False)
        | manifest.lt(0.50).fillna(False)
        | lag.gt(5.0).fillna(False)
        | resp.gt(0.25).fillna(False)
    )

    warning_mask = (
        ~fatal_mask
        & (
            eta.lt(0.70).fillna(False)
            | manifest.lt(0.70).fillna(False)
            | lag.gt(2.0).fillna(False)
            | resp.gt(0.10).fillna(False)
        )
    )

    metrics["q4_status_derived_14_6R1"] = np.select(
        [fatal_mask, warning_mask],
        ["fatal", "warning"],
        default="pass",
    )

    metrics["publication_blocker"] = fatal_mask.astype(bool)

status_col = "q4_status_derived_14_6R1"

metrics[status_col] = metrics[status_col].astype(str).str.lower()
fatal_n = int(metrics[status_col].eq("fatal").sum())
warning_n = int(metrics[status_col].eq("warning").sum())
pass_n = int(metrics[status_col].eq("pass").sum())
not_eval_n = int(metrics[status_col].str.contains("not", case=False, na=False).sum())

if "publication_blocker" in metrics.columns:
    blocker_n = int(_as_bool_146r1(metrics["publication_blocker"]).sum())
else:
    blocker_n = fatal_n

summary146 = CELL14_6_A0_CONTRACT.get("summary", {}) if isinstance(CELL14_6_A0_CONTRACT, dict) else {}
outside_drift = int(summary146.get("outside_window_drift_total", 0) or 0)
non_target_drift = int(summary146.get("non_target_protocol_drift_n", 0) or 0)
negative_n = int(summary146.get("negative_n", 0) or 0)
noninteger_n = int(summary146.get("noninteger_n", 0) or 0)
inactive_finite_n = int(summary146.get("inactive_finite_n", 0) or 0)

safety_blocker_n = int(outside_drift + non_target_drift + negative_n + noninteger_n + inactive_finite_n)
zero_test_blocker = bool(blocker_n == 0 and fatal_n == 0 and safety_blocker_n == 0)
fresh_holdout = bool(CFG.get("cell14_6R_fresh_holdout_for_publication", False))

if zero_test_blocker and fresh_holdout:
    terminal_status = "SCOPED_PASS_FRESH_HOLDOUT"
    publication_claim_allowed = True
elif zero_test_blocker:
    terminal_status = "SCOPED_PASS_DEVELOPMENT_EVIDENCE_ONLY"
    publication_claim_allowed = False
else:
    terminal_status = "BLOCKED_NO_PROMOTION"
    publication_claim_allowed = False

scope_status = pd.DataFrame([{
    "scope": "A0_zigbee_safe_frozen_policy",
    "policy_id": str(CFG.get("cell14_6R0_policy_id", "")),
    "selected_on_VAL": True,
    "TEST_used_for_policy": False,
    "fresh_holdout_declared": bool(fresh_holdout),
    "pairs_total": int(len(metrics)),
    "TEST_pass_n": pass_n,
    "TEST_warning_n": warning_n,
    "TEST_fatal_n": fatal_n,
    "TEST_not_evaluable_n": not_eval_n,
    "TEST_publication_blocker_n": blocker_n,
    "safety_blocker_n": safety_blocker_n,
    "mean_ETA_similarity": float(pd.to_numeric(metrics[_find_col_146r1(metrics, ["ETA_similarity", "eta_similarity", "q4_eta_similarity"])], errors="coerce").mean()) if len(metrics) and _find_col_146r1(metrics, ["ETA_similarity", "eta_similarity", "q4_eta_similarity"]) else np.nan,
    "mean_lag_peak_error": float(pd.to_numeric(metrics[_find_col_146r1(metrics, ["lag_peak_error", "lag_peak_error_s", "lag_error", "peak_lag_error"])], errors="coerce").mean()) if len(metrics) and _find_col_146r1(metrics, ["lag_peak_error", "lag_peak_error_s", "lag_error", "peak_lag_error"]) else np.nan,
    "mean_response_window_rate_error": float(pd.to_numeric(metrics[_find_col_146r1(metrics, ["response_window_rate_error", "response_rate_error", "resp_window_rate_error", "response_window_error"])], errors="coerce").mean()) if len(metrics) and _find_col_146r1(metrics, ["response_window_rate_error", "response_rate_error", "resp_window_rate_error", "response_window_error"]) else np.nan,  
    "terminal_status": terminal_status,
    "publication_claim_allowed": publication_claim_allowed,
    "claim_note": (
        "Zero-blocker A0 evidence requires a fresh untouched holdout for final publication because the current TEST split has been development-observed."
        if zero_test_blocker and not fresh_holdout else
        "A0 remains blocked under terminal TEST QA." if not zero_test_blocker else
        "A0 scoped pass on fresh holdout."
    ),
}])

scope_csv = REPORT_DIR_R1 / "cell14_6R1_A0_frozen_policy_terminal_scope_status.csv"
contract_json = CONTRACT_DIR_R1 / "cell14_6R1_A0_terminal_evidence_contract_v1_0_THESIS.json"
scope_status.to_csv(scope_csv, index=False)

contract_r1 = {
    "cell": "14.6R1",
    "version": "cell14_6R1_A0_terminal_evidence_gate_v1_0_THESIS",
    "role": "frozen_A0_terminal_TEST_evidence_gate",
    "summary": scope_status.iloc[0].to_dict(),
    "strict_contract": {
        "policy_selected_on_VAL": True,
        "TEST_used_for_policy_selection": False,
        "TEST_used_for_pair_pruning": False,
        "TEST_used_for_materialization": False,
        "synthetic_values_mutated_here": False,
        "artifact_promotion_done_here": False,
        "publication_claim_requires_fresh_holdout": True,
    },
    "source_contracts": {
        "cell14_6R0": r0_contract.get("outputs", {}).get("contract_json", ""),
        "cell14_6": globals().get("CELL14_6_A0_CONTRACT_JSON", ""),
    },
    "outputs": {
        "scope_status_csv": str(scope_csv),
        "contract_json": str(contract_json),
    },
}
_write_json_146r1(contract_json, contract_r1)

globals()["CELL14_6R1_A0_TERMINAL_SCOPE_STATUS_DF"] = scope_status
globals()["CELL14_6R1_A0_TERMINAL_ACCEPTANCE_CONTRACT"] = contract_r1
globals()["CELL14_6R1_A0_TERMINAL_ACCEPTANCE_CONTRACT_JSON"] = str(contract_json)

print("=== CELL 14.6R1 FROZEN A0 TERMINAL STATUS ===")
print(scope_status.to_string(index=False))
log("--- END: Cell 14.6R1 - Frozen A0 terminal TEST evidence gate ---")
