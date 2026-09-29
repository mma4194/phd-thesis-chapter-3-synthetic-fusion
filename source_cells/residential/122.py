# ==========================================================
# CELL 14.6R2 - VAL-feasible A0 Zigbee subset gate
# v1.0 STUDY-THESIS strict / VAL-only eligibility / no TEST pruning
#
# Purpose:
#   Define a narrower A0 Zigbee-safe coupling subset using only
#   TRAIN/VAL feasibility evidence from Cell 14.6D0.
#
# Rule:
#   A pair is eligible iff VAL_nonfatal_policy_n > 0.
#
# This cell:
#   - does NOT mutate synthetic data,
#   - does NOT repair values,
#   - does NOT select pairs using TEST status,
#   - reports TEST status only after VAL-only eligibility is fixed,
#   - marks the claim as development evidence unless a genuinely
#     fresh holdout is declared.
# ==========================================================

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    log("--- START: Cell 14.6R2 - VAL-feasible A0 Zigbee subset gate ---")
except Exception:
    print("--- START: Cell 14.6R2 - VAL-feasible A0 Zigbee subset gate ---")

REPORT_DIR_R2 = Path(str(REPORT_DIR)).expanduser().resolve()
OUTDIR_R2 = Path(str(OUTDIR)).expanduser().resolve()
CONTRACT_DIR_R2 = OUTDIR_R2 / "artifacts" / "contracts"
REPORT_DIR_R2.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_R2.mkdir(parents=True, exist_ok=True)

CELL146R2_VERSION = "cell14_6R2_VAL_feasible_A0_zigbee_subset_gate_v1_0_THESIS"

root_path = REPORT_DIR_R2 / "cell14_6D0_q4_a0_pair_root_cause.csv"
if not root_path.exists():
    raise RuntimeError(
        f"[Cell14.6R2] Missing diagnostic root-cause file: {root_path}. "
        "Run Cell 14.6D0 first."
    )

root = pd.read_csv(root_path)

required_cols = [
    "repair_candidate_id",
    "TEST_status_norm",
    "TEST_publication_blocker_norm",
    "VAL_nonfatal_policy_n",
]
missing = [c for c in required_cols if c not in root.columns]
if missing:
    raise RuntimeError(f"[Cell14.6R2] Missing required columns in D0 output: {missing}")

def _bool_r2(s):
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(
        ["1", "true", "yes", "y", "fatal", "blocker", "blocked"]
    )

root = root.copy()
root["VAL_nonfatal_policy_n"] = pd.to_numeric(root["VAL_nonfatal_policy_n"], errors="coerce").fillna(0).astype(int)

# ------------------------------------------------------------------
# VAL-only eligibility rule.
# Do not change this based on TEST status.
# ------------------------------------------------------------------
root["A0_VAL_feasible_subset_member"] = root["VAL_nonfatal_policy_n"].gt(0)

root["A0_VAL_feasible_exclusion_reason"] = np.where(
    root["A0_VAL_feasible_subset_member"],
    "",
    "excluded_by_VAL_only_rule:no_nonfatal_policy_in_current_A0_family",
)

selected = root[root["A0_VAL_feasible_subset_member"]].copy()
excluded = root[~root["A0_VAL_feasible_subset_member"]].copy()

# TEST is used only for terminal reporting after membership is fixed.
selected["TEST_status_norm"] = selected["TEST_status_norm"].astype(str).str.lower()
selected["TEST_publication_blocker_bool"] = _bool_r2(selected["TEST_publication_blocker_norm"])

excluded["TEST_status_norm"] = excluded["TEST_status_norm"].astype(str).str.lower()
excluded["TEST_publication_blocker_bool"] = _bool_r2(excluded["TEST_publication_blocker_norm"])

test_pass_n = int(selected["TEST_status_norm"].eq("pass").sum())
test_warning_n = int(selected["TEST_status_norm"].eq("warning").sum())
test_fatal_n = int(selected["TEST_status_norm"].isin(["fatal", "blocker", "blocked"]).sum())
test_blocker_n = int(selected["TEST_publication_blocker_bool"].sum())

fresh_holdout_declared = bool(
    globals().get("CFG", {}).get("q4_fresh_holdout_declared", False)
    if "CFG" in globals()
    else False
)

zero_blocker_current_test = test_blocker_n == 0 and test_fatal_n == 0

if zero_blocker_current_test and fresh_holdout_declared:
    evidence_status = "STRICT_PUBLICATION_CLAIM_ALLOWED_FRESH_HOLDOUT"
    publication_claim_allowed = True
elif zero_blocker_current_test:
    evidence_status = "DEVELOPMENT_EVIDENCE_ZERO_BLOCKER_ON_ALREADY_OBSERVED_TEST"
    publication_claim_allowed = False
else:
    evidence_status = "BLOCKED_CURRENT_TEST"
    publication_claim_allowed = False

metric_cols = {
    "mean_ETA_similarity": "ETA_similarity_num",
    "mean_manifest_similarity": "manifest_similarity_num",
    "mean_lag_peak_error": "lag_peak_error_num",
    "mean_response_window_rate_error": "response_window_rate_error_num",
}

summary = {
    "scope": "A0_VAL_feasible_zigbee_subset",
    "version": CELL146R2_VERSION,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "eligibility_rule": "VAL_nonfatal_policy_n > 0",
    "selected_on_VAL": True,
    "TEST_used_for_membership": False,
    "synthetic_values_mutated_here": False,
    "pair_pruning_by_TEST": False,
    "fresh_holdout_declared": fresh_holdout_declared,
    "all_A0_pair_n": int(len(root)),
    "selected_pair_n": int(len(selected)),
    "excluded_pair_n": int(len(excluded)),
    "TEST_pass_n": test_pass_n,
    "TEST_warning_n": test_warning_n,
    "TEST_fatal_n": test_fatal_n,
    "TEST_publication_blocker_n": test_blocker_n,
    "zero_blocker_current_TEST": bool(zero_blocker_current_test),
    "publication_claim_allowed": bool(publication_claim_allowed),
    "evidence_status": evidence_status,
    "selected_pairs": selected["repair_candidate_id"].astype(str).tolist(),
    "excluded_pairs": excluded["repair_candidate_id"].astype(str).tolist(),
    "excluded_pair_reasons": excluded[
        ["repair_candidate_id", "A0_VAL_feasible_exclusion_reason", "VAL_nonfatal_policy_n", "TEST_status_norm"]
    ].to_dict("records"),
}

for out_name, col in metric_cols.items():
    if col in selected.columns and len(selected):
        summary[out_name] = float(pd.to_numeric(selected[col], errors="coerce").mean())
    else:
        summary[out_name] = np.nan

summary_df = pd.DataFrame([summary])

selected_out = REPORT_DIR_R2 / "cell14_6R2_A0_VAL_feasible_subset_pairs.csv"
excluded_out = REPORT_DIR_R2 / "cell14_6R2_A0_VAL_feasible_excluded_pairs.csv"
summary_out = REPORT_DIR_R2 / "cell14_6R2_A0_VAL_feasible_subset_scope_status.csv"
contract_out = CONTRACT_DIR_R2 / "cell14_6R2_A0_VAL_feasible_subset_contract_v1_0_THESIS.json"

selected.to_csv(selected_out, index=False)
excluded.to_csv(excluded_out, index=False)
summary_df.to_csv(summary_out, index=False)

def _json_safe_r2(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe_r2(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_r2(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.DataFrame):
        return _json_safe_r2(obj.to_dict("records"))
    return obj

contract = {
    "cell": "14.6R2",
    "version": CELL146R2_VERSION,
    "purpose": "VAL-only feasible A0 Zigbee subset gate",
    "strict_contract": {
        "selected_on_VAL": True,
        "TEST_used_for_membership": False,
        "synthetic_values_mutated_here": False,
        "pair_pruning_by_TEST": False,
        "thresholds_changed_here": False,
        "all_8_A0_claim_attempted": False,
        "narrower_subset_claim_created": True,
        "publication_claim_requires_fresh_holdout": True,
    },
    "inputs": {
        "root_cause_csv": str(root_path),
    },
    "outputs": {
        "selected_pairs_csv": str(selected_out),
        "excluded_pairs_csv": str(excluded_out),
        "scope_status_csv": str(summary_out),
        "contract_json": str(contract_out),
    },
    "summary": summary,
}

contract_out.write_text(json.dumps(_json_safe_r2(contract), indent=2, sort_keys=True), encoding="utf-8")

globals()["CELL14_6R2_A0_VAL_FEASIBLE_SELECTED_DF"] = selected
globals()["CELL14_6R2_A0_VAL_FEASIBLE_EXCLUDED_DF"] = excluded
globals()["CELL14_6R2_A0_VAL_FEASIBLE_SUMMARY_DF"] = summary_df
globals()["CELL14_6R2_A0_VAL_FEASIBLE_CONTRACT"] = contract

print("\n=== CELL 14.6R2 A0 VAL-FEASIBLE SUBSET SUMMARY ===")
display(summary_df)

print("\n=== CELL 14.6R2 SELECTED PAIRS ===")
display(
    selected[
        [
            c for c in [
                "repair_candidate_id",
                "anchor_col",
                "protocol_col",
                "TEST_status_norm",
                "TEST_publication_blocker_bool",
                "VAL_nonfatal_policy_n",
                "ETA_similarity_num",
                "manifest_similarity_num",
                "lag_peak_error_num",
                "response_window_rate_error_num",
                "diagnostic_root_cause",
            ] if c in selected.columns
        ]
    ]
)

print("\n=== CELL 14.6R2 EXCLUDED PAIRS ===")
display(
    excluded[
        [
            c for c in [
                "repair_candidate_id",
                "anchor_col",
                "protocol_col",
                "VAL_nonfatal_policy_n",
                "TEST_status_norm",
                "TEST_publication_blocker_bool",
                "diagnostic_root_cause",
                "A0_VAL_feasible_exclusion_reason",
            ] if c in excluded.columns
        ]
    ]
)

print(f"\n[Cell14.6R2] Saved selected pairs: {selected_out}")
print(f"[Cell14.6R2] Saved excluded pairs: {excluded_out}")
print(f"[Cell14.6R2] Saved summary: {summary_out}")
print(f"[Cell14.6R2] Saved contract: {contract_out}")

try:
    log("--- END: Cell 14.6R2 - VAL-feasible A0 Zigbee subset gate ---")
except Exception:
    print("--- END: Cell 14.6R2 - VAL-feasible A0 Zigbee subset gate ---")