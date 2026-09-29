# ==========================================================
# CELL 14.6R3 - Q4 A0 scope arbitration table for paper
# v1.0 / no mutation / paper-ready Q4 ledger
# ==========================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd

REPORT_DIR_R3 = Path(str(REPORT_DIR)).expanduser().resolve()
OUTDIR_R3 = Path(str(OUTDIR)).expanduser().resolve()
CONTRACT_DIR_R3 = OUTDIR_R3 / "artifacts" / "contracts"
CONTRACT_DIR_R3.mkdir(parents=True, exist_ok=True)

r2_path = REPORT_DIR_R3 / "cell14_6R2_A0_VAL_feasible_subset_scope_status.csv"
d0_path = REPORT_DIR_R3 / "cell14_6D0_q4_a0_policy_feasibility.csv"
root_path = REPORT_DIR_R3 / "cell14_6D0_q4_a0_pair_root_cause.csv"
r1_path = REPORT_DIR_R3 / "cell14_6R1_A0_frozen_policy_terminal_scope_status.csv"

if not r2_path.exists():
    raise RuntimeError(f"Missing R2 summary: {r2_path}")
if not d0_path.exists():
    raise RuntimeError(f"Missing D0 policy feasibility: {d0_path}")
if not root_path.exists():
    raise RuntimeError(f"Missing D0 pair root cause: {root_path}")

r2 = pd.read_csv(r2_path)
d0 = pd.read_csv(d0_path)
root = pd.read_csv(root_path)
r1 = pd.read_csv(r1_path) if r1_path.exists() else pd.DataFrame()

root["TEST_status_norm"] = root["TEST_status_norm"].astype(str).str.lower()
root["TEST_publication_blocker_norm"] = root["TEST_publication_blocker_norm"].astype(str).str.lower().isin(
    ["true", "1", "yes", "fatal", "blocker", "blocked"]
)

all8_pair_n = int(len(root))
all8_pass_n = int(root["TEST_status_norm"].eq("pass").sum())
all8_warning_n = int(root["TEST_status_norm"].eq("warning").sum())
all8_fatal_n = int(root["TEST_status_norm"].isin(["fatal", "blocker", "blocked"]).sum())
all8_blocker_n = int(root["TEST_publication_blocker_norm"].sum())

r2_row = r2.iloc[0].to_dict()

rows = [
    {
        "q4_scope": "A0_all_8_zigbee_safe_candidate",
        "pair_n": all8_pair_n,
        "pass_n": all8_pass_n,
        "warning_n": all8_warning_n,
        "fatal_n": all8_fatal_n,
        "publication_blocker_n": all8_blocker_n,
        "mean_ETA_similarity": float(pd.to_numeric(root["ETA_similarity_num"], errors="coerce").mean()),
        "mean_manifest_similarity": float(pd.to_numeric(root["manifest_similarity_num"], errors="coerce").mean()),
        "mean_lag_peak_error": float(pd.to_numeric(root["lag_peak_error_num"], errors="coerce").mean()),
        "mean_response_window_rate_error": float(pd.to_numeric(root["response_window_rate_error_num"], errors="coerce").mean()),
        "selected_on_VAL": True,
        "TEST_used_for_membership": False,
        "fresh_holdout_declared": False,
        "claim_status": "blocked_no_promotion",
        "paper_claim_allowed": False,
        "paper_interpretation": (
            "The original 8-pair A0 Zigbee-safe candidate is blocked because at least one pair "
            "has a terminal Q4 publication blocker; no positive all-8 A0 claim is allowed."
        ),
    },
    {
        "q4_scope": "A0_VAL_feasible_zigbee_subset",
        "pair_n": int(r2_row["selected_pair_n"]),
        "pass_n": int(r2_row["TEST_pass_n"]),
        "warning_n": int(r2_row["TEST_warning_n"]),
        "fatal_n": int(r2_row["TEST_fatal_n"]),
        "publication_blocker_n": int(r2_row["TEST_publication_blocker_n"]),
        "mean_ETA_similarity": float(r2_row["mean_ETA_similarity"]),
        "mean_manifest_similarity": float(r2_row["mean_manifest_similarity"]),
        "mean_lag_peak_error": float(r2_row["mean_lag_peak_error"]),
        "mean_response_window_rate_error": float(r2_row["mean_response_window_rate_error"]),
        "selected_on_VAL": True,
        "TEST_used_for_membership": False,
        "fresh_holdout_declared": bool(r2_row["fresh_holdout_declared"]),
        "claim_status": str(r2_row["evidence_status"]),
        "paper_claim_allowed": bool(r2_row["publication_claim_allowed"]),
        "paper_interpretation": (
            "A narrower VAL-feasible Zigbee A0 subset has zero current-TEST blockers, "
            "but because the subset was introduced after the current TEST audit had been observed, "
            "it is development evidence unless evaluated on a genuinely fresh holdout."
        ),
    },
    {
        "q4_scope": "A0_VAL_ineligible_excluded_pairs",
        "pair_n": int(r2_row["excluded_pair_n"]),
        "pass_n": np.nan,
        "warning_n": np.nan,
        "fatal_n": np.nan,
        "publication_blocker_n": np.nan,
        "mean_ETA_similarity": np.nan,
        "mean_manifest_similarity": np.nan,
        "mean_lag_peak_error": np.nan,
        "mean_response_window_rate_error": np.nan,
        "selected_on_VAL": True,
        "TEST_used_for_membership": False,
        "fresh_holdout_declared": False,
        "claim_status": "excluded_by_VAL_feasibility",
        "paper_claim_allowed": False,
        "paper_interpretation": (
            "Pairs 0005 and 0006 are excluded because no TRAIN/VAL policy in the current A0 family "
            "made them nonfatal; this exclusion is not based on TEST status."
        ),
    },
]

q4_arbitration = pd.DataFrame(rows)

out_csv = REPORT_DIR_R3 / "paper_table_q4_a0_scope_arbitration.csv"
out_json = CONTRACT_DIR_R3 / "cell14_6R3_q4_a0_scope_arbitration_contract_v1_0_THESIS.json"

q4_arbitration.to_csv(out_csv, index=False)

contract = {
    "cell": "14.6R3",
    "version": "cell14_6R3_q4_a0_scope_arbitration_v1_0_THESIS",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Paper-ready arbitration of A0 Q4 claims after VAL-feasible subset diagnostic.",
    "strict_contract": {
        "synthetic_values_mutated": False,
        "thresholds_changed": False,
        "TEST_used_for_membership": False,
        "all_8_A0_claim_allowed": False,
        "VAL_feasible_subset_zero_blocker_current_TEST": bool(r2_row["zero_blocker_current_TEST"]),
        "VAL_feasible_subset_publication_claim_allowed": bool(r2_row["publication_claim_allowed"]),
        "fresh_holdout_required_for_final_positive_claim": True,
    },
    "outputs": {
        "paper_table_q4_a0_scope_arbitration": str(out_csv),
        "contract": str(out_json),
    },
}

out_json.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

globals()["PAPER_TABLE_Q4_A0_SCOPE_ARBITRATION"] = q4_arbitration

print("\n=== PAPER-READY Q4 A0 SCOPE ARBITRATION ===")
display(q4_arbitration)

print(f"\nSaved: {out_csv}")
print(f"Saved contract: {out_json}")