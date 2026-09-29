# %% CELL Q6.0 — Active branch registry and supersession ledger
# Purpose:
#   Create an authoritative active-branch registry for STUDY-THESIS review.
#   This prevents old/superseded artifacts from being misread as active evidence.
#
# Safety:
#   - Reads reports/contracts only.
#   - Does not mutate synthetic values.
#   - Does not select, fit, repair, or promote models.
#   - Does not read raw TEST data.

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def _q6_log(msg):
    print(f"[Q6.0] {msg}")


def _q6_path(x):
    if x is None:
        return None
    return Path(str(x)).expanduser().resolve()


OUTDIR_Q6 = _q6_path(globals().get("OUTDIR", None)) or _q6_path(os.environ.get("CPS_OUTDIR"))
if OUTDIR_Q6 is None:
    raise RuntimeError("[Q6.0] OUTDIR is unresolved.")
REPORT_DIR_Q6 = _q6_path(globals().get("REPORT_DIR", None)) or OUTDIR_Q6 / "reports"
CONTRACT_DIR_Q6 = _q6_path(globals().get("CONTRACT_DIR", None)) or OUTDIR_Q6 / "artifacts" / "contracts"
ARTIFACT_DIR_Q6 = _q6_path(globals().get("ARTIFACT_DIR", None)) or _q6_path(globals().get("ARTDIR", None)) or OUTDIR_Q6 / "artifacts"
OUT_SYN_Q6 = _q6_path(globals().get("OUT_SYN", None)) or OUTDIR_Q6 / "synthetic"

for p in [REPORT_DIR_Q6, CONTRACT_DIR_Q6, ARTIFACT_DIR_Q6, OUT_SYN_Q6]:
    p.mkdir(parents=True, exist_ok=True)

required_active = {
    "continuous_12cR_contract": CONTRACT_DIR_Q6 / "cell12c6R_final_qa_contract.json",
    "continuous_12cR_manifest": REPORT_DIR_Q6 / "cell12c6R_final_publication_manifest.json",
    "binary_v6_contract": CONTRACT_DIR_Q6 / "cell12d6_binary_final_test_qa_contract_v6_0_THESIS.json",
    "binary_v6_summary": REPORT_DIR_Q6 / "cell12d_v6_terminal_test_qa_summary.json",
    "q3_v5_contract": CONTRACT_DIR_Q6 / "q3_post_v5_observability_mask_post_qa_contract.json",
    "q3_v5_terminal_contract": CONTRACT_DIR_Q6 / "cell12f5_mask_final_test_qa_contract_v1_1_THESIS.json",
}
missing = [f"{k}: {v}" for k, v in required_active.items() if not v.exists()]
if missing:
    raise RuntimeError("[Q6.0] Missing active branch contract/report(s):\n" + "\n".join(missing))


def _read_json_or_empty(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}

continuous = _read_json_or_empty(required_active["continuous_12cR_manifest"])
binary = _read_json_or_empty(required_active["binary_v6_summary"])
q3 = _read_json_or_empty(required_active["q3_v5_contract"])
q3_terminal = _read_json_or_empty(required_active["q3_v5_terminal_contract"])

active_rows = [
    {
        "dimension": "Q1_Q2_continuous_values",
        "active_branch": "12.c.R",
        "active_contract": str(required_active["continuous_12cR_contract"]),
        "status": "lineage_safe_value_limited",
        "claim_scope": "partial_limited_continuous_value_claims",
        "key_counts_json": json.dumps({
            "targets": continuous.get("targets"),
            "status_counts": continuous.get("status_counts"),
            "publication_status_counts": continuous.get("publication_status_counts"),
        }, sort_keys=True),
        "blind_final_evidence": False,
        "same_df_te_development_evidence": True,
    },
    {
        "dimension": "Q5_binary_iot_states",
        "active_branch": "12.d.V6",
        "active_contract": str(required_active["binary_v6_contract"]),
        "status": "best_development_binary_branch_partial",
        "claim_scope": "39_included_13_scope_excluded_3_included_blockers",
        "key_counts_json": json.dumps({
            "included_n": binary.get("included_n_trainval_only"),
            "scope_excluded_n": binary.get("scope_excluded_n_trainval_only"),
            "included_pass": binary.get("included_pass"),
            "included_warning": binary.get("included_warning"),
            "included_fatal": binary.get("included_fatal"),
            "included_publication_blocker_n": binary.get("included_publication_blocker_n"),
        }, sort_keys=True),
        "blind_final_evidence": False,
        "same_df_te_development_evidence": True,
    },
    {
        "dimension": "Q3_observability_masks",
        "active_branch": "12.f.V5 / Q3.POST.V5",
        "active_contract": str(required_active["q3_v5_contract"]),
        "status": q3.get("final_status", "unknown"),
        "claim_scope": q3.get("claim_scope", "unknown"),
        "key_counts_json": json.dumps({
            "targets": q3.get("targets"),
            "qa_status_counts": q3.get("qa_status_counts"),
            "publication_status_counts": q3.get("publication_status_counts"),
            "publication_blocker_n": q3.get("publication_blocker_n"),
            "full_scope_ready": q3.get("full_scope_ready"),
        }, sort_keys=True),
        "blind_final_evidence": False,
        "same_df_te_development_evidence": True,
    },
]
active_df = pd.DataFrame(active_rows)

supersession_rows = [
    {"branch_or_cell": "old 12.c.3–12.c.6", "status": "superseded", "active_replacement": "12.c.3R–12.c.6R", "reason": "Old continuous selector lineage contained TEST-policy risk; replacement uses TRAIN/VAL candidate metrics/cache and TEST only for terminal QA."},
    {"branch_or_cell": "12.f.V2 / Q3.POST.V2", "status": "rejected", "active_replacement": "12.f.V5 / Q3.POST.V5", "reason": "V2 collapsed indicator-valued observability targets by applying pd.notna to already-materialized state columns."},
    {"branch_or_cell": "12.f.V3 / Q3.POST.V3", "status": "valid_baseline_superseded", "active_replacement": "12.f.V5 / Q3.POST.V5", "reason": "Correct semantic branch but higher publication blockers than V5."},
    {"branch_or_cell": "12.f.V4 / Q3.POST.V4", "status": "valid_baseline_superseded", "active_replacement": "12.f.V5 / Q3.POST.V5", "reason": "Strict semantic branch; V5 retains scope and improves blockers via TRAIN→VAL materializer backtest."},
    {"branch_or_cell": "old 12.f.3x–12.f.5 and old Q3.POST", "status": "superseded", "active_replacement": "12.f.V5 / Q3.POST.V5", "reason": "Old mask branch is not the active Q3 evidence."},
    {"branch_or_cell": "Binary V2–V5", "status": "superseded", "active_replacement": "12.d.V6", "reason": "V6 is the preferred binary development candidate with fewer included blockers."},
]
supersession_df = pd.DataFrame(supersession_rows)

claim_rows = [
    {"Q": "Q1", "dimension": "support/domain/value validity", "active_source": "12.c.R + downstream final package", "claim_status": "partial", "paper_claim": "Continuous value support/domain is lineage-safe but value fidelity remains limited; report per-column readiness."},
    {"Q": "Q2", "dimension": "temporal realism", "active_source": "12.c.R + Binary V6 + Q3 V5", "claim_status": "partial", "paper_claim": "Temporal dynamics are branch-scoped; binary and masks improved, continuous remains limited."},
    {"Q": "Q3", "dimension": "observability/masks", "active_source": "Q3.POST.V5", "claim_status": q3.get("final_status", "unknown"), "paper_claim": "Partial-admissible only; full-scope observability is not claimed."},
    {"Q": "Q4", "dimension": "cross-layer coupling", "active_source": "Cells 13–14 downstream", "claim_status": "downstream_must_carry_partial_scope", "paper_claim": "Coupling must preserve Q3 partial scope and no-Q4-promotion constraints."},
    {"Q": "Q5", "dimension": "binary IoT", "active_source": "Binary V6", "claim_status": "partial_best_development", "paper_claim": "Binary V6 is the active binary branch; report included/scope-excluded/blocker counts."},
    {"Q": "Q6", "dimension": "governance/leakage/artifact hygiene", "active_source": "Q6.0–Q6.4", "claim_status": "active_governance_layer", "paper_claim": "Active branch ledger and hygiene gates define authoritative evidence."},
]
claim_df = pd.DataFrame(claim_rows)

active_path = REPORT_DIR_Q6 / "q6_active_branch_registry.csv"
super_path = REPORT_DIR_Q6 / "q6_supersession_ledger.csv"
claim_path = REPORT_DIR_Q6 / "q6_q1_q6_claim_matrix.csv"
contract = {
    "cell": "Q6.0",
    "role": "active_branch_registry_and_supersession_ledger",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "active_branches": active_rows,
    "superseded_or_rejected_branches": supersession_rows,
    "q_claim_matrix": claim_rows,
    "policy": {
        "reads_TEST_real_values": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "q3_v2_rejected": True,
        "q3_active_branch": "12.f.V5",
    },
}
active_df.to_csv(active_path, index=False)
supersession_df.to_csv(super_path, index=False)
claim_df.to_csv(claim_path, index=False)
contract_path = CONTRACT_DIR_Q6 / "q6_active_branch_registry_contract_v1_0_THESIS.json"
manifest_path = REPORT_DIR_Q6 / "q6_active_branch_registry_manifest.json"
for p in [contract_path, manifest_path]:
    p.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

CELL_Q6_ACTIVE_BRANCH_REGISTRY = contract
_q6_log("Active branch registry written.")
_q6_log(f"Active registry: {active_path}")
_q6_log(f"Supersession ledger: {super_path}")
_q6_log(f"Contract: {contract_path}")
if display is not None:
    display(active_df)
    display(supersession_df)
    display(claim_df)
