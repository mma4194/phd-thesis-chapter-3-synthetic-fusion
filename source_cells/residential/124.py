# ==========================================================
# CELL 14.7 - Q4 no-promotion governance ledger
# v2.1 STUDY-THESIS strict blocked-branch governance
#
# Role:
#   - Record that A0 was NOT promoted after Cells 14.6/14.6a/14.6b.
#   - Do NOT copy candidate artifacts to final paths.
#   - Do NOT mutate values.
#   - Export explicit downstream status: Q4 remains blocked.
# ==========================================================

log("--- START: Cell 14.7 - Q4 no-promotion governance ledger (v2.1 strict) ---")

import os, json, hashlib
import numpy as np
import pandas as pd

_required_147 = ["CFG", "log", "OUTDIR", "OUT_SYN", "REPORT_DIR"]
_missing_147 = [k for k in _required_147 if k not in globals()]
if _missing_147:
    raise RuntimeError(f"[Cell14.7] Missing required globals: {_missing_147}")

def _resolve_project_root_147(outdir, report_dir, out_syn):
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
        c = os.path.abspath(c)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "synthetic")):
            return c
    raise RuntimeError("[Cell14.7] Could not resolve canonical project root.")

PROJECT_ROOT_147 = _resolve_project_root_147(str(OUTDIR), str(REPORT_DIR), str(OUT_SYN))
REPORT_DIR_147 = os.path.join(PROJECT_ROOT_147, "reports")
ARTDIR_147 = os.path.join(PROJECT_ROOT_147, "artifacts")
CONTRACT_DIR_147 = os.path.join(ARTDIR_147, "contracts")
os.makedirs(REPORT_DIR_147, exist_ok=True)
os.makedirs(CONTRACT_DIR_147, exist_ok=True)

CELL147_VERSION = "cell14_7_q4_no_promotion_governance_v2_1"

CFG["cell14_7_version"] = CELL147_VERSION
CFG["cell14_7_Q4_final_status"] = "blocked_no_promotion"
CFG["cell14_7_A0_promoted"] = False
CFG["cell14_7_artifact_copy_done_here"] = False
CFG["cell14_7_synthetic_values_mutated"] = False
CFG["cell14_7_selection_done_here"] = False
CFG["cell14_7_generator_fit_done_here"] = False
CFG["cell14_7_materialization_done_here"] = False
CFG["cell14_7_promotion_done_here"] = False

def _read_json_147(path):
    if not path or not os.path.exists(str(path)):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _sha256_file_147(path):
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _sanitize_147(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_147(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_147(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_147(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_147(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _sanitize_147(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _sanitize_147(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_147(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_147(payload), f, indent=2, sort_keys=True)

candidate_contract_path = globals().get(
    "CELL14_6_A0_CANDIDATE_CONTRACT_JSON",
    os.path.join(REPORT_DIR_147, "cell14_6_A0_candidate_contract.json"),
)
accepted_contract_path = globals().get(
    "CELL14_6_A0_ACCEPTED_CONTRACT_JSON",
    os.path.join(REPORT_DIR_147, "cell14_6_A0_ACCEPTED_contract.json"),
)
legality_contract_path = globals().get(
    "CELL14_6B_A0_POLICY_LEGALITY_CONTRACT_JSON",
    os.path.join(REPORT_DIR_147, "cell14_6b_A0_policy_legality_contract.json"),
)

candidate_contract = _read_json_147(candidate_contract_path)
accepted_contract = _read_json_147(accepted_contract_path)
legality_contract = _read_json_147(legality_contract_path)

candidate_summary = candidate_contract.get("summary", {})
legality_summary = legality_contract.get("summary", {})
accepted = bool(accepted_contract.get("accepted", False))

pairs_total = int(candidate_summary.get("A0_pairs_total", legality_summary.get("A0_rows", 0)) or 0)
pass_n = int(candidate_summary.get("q4_pass_n", legality_summary.get("pass_n", 0)) or 0)
warning_n = int(candidate_summary.get("q4_warning_n", legality_summary.get("warning_n", 0)) or 0)
fatal_n = int(candidate_summary.get("q4_fatal_n", legality_summary.get("fatal_n", 0)) or 0)
blocker_n = int(candidate_summary.get("q4_publication_blocker_n", fatal_n) or 0)
simple_proxy = bool(legality_summary.get("simple_trainval_proxy_exists_diagnostic", False))
fatal_ids = legality_summary.get("fatal_ids", [])
fatal_reasons = legality_summary.get("fatal_reason_counts", {})

if accepted:
    raise RuntimeError("[Cell14.7] Accepted contract says accepted=True; use strict promoter instead of no-promotion ledger.")
if blocker_n == 0:
    print(
        "[Cell14.7] A0 candidate has zero terminal TEST blockers, but no fresh-holdout "
        "publication promoter is active. Keeping final_q4_status=blocked_no_promotion "
        "unless a separate fresh-holdout acceptance cell is explicitly enabled."
    )

governance_decision = {
    "final_q4_status": "blocked_no_promotion",
    "A0_candidate_status": "blocked",
    "A0_promoted": False,
    "broad_q4_promoted": False,
    "accepted_artifacts_written": False,
    "candidate_artifacts_may_exist": True,
    "candidate_artifacts_are_not_final": True,
    "reason": (
        "A0 candidate improved coupling but failed strict promotion gate; "
        "Cell 14.6b found no defensible simple TRAIN/VAL proxy for pruning fatal rows without TEST-outcome cherry-picking."
    ),
}

if blocker_n == 0:
    governance_decision["A0_candidate_status"] = "zero_blocker_development_evidence_not_promoted"
    governance_decision["reason"] = (
        "A0 has zero terminal TEST blockers under the frozen TRAIN/VAL policy, but this notebook has "
        "already observed the TEST split during development and no fresh untouched holdout is declared. "
        "Therefore final_q4_status remains blocked_no_promotion for publication unless rerun on a fresh holdout."
    )

summary = {
    "cell": "14.7",
    "version": CELL147_VERSION,
    "candidate_name": "A0_zigbee_safe_candidate_blocked",
    "candidate_scope": "A0_zigbee_safe",
    "protocol_scope": "zigbee",
    "pairs_total": pairs_total,
    "q4_pass_n": pass_n,
    "q4_warning_n": warning_n,
    "q4_fatal_n": fatal_n,
    "q4_publication_blocker_n": blocker_n,
    "publication_blocker_n": blocker_n,
    "fatal_ids": fatal_ids,
    "fatal_reason_counts": fatal_reasons,
    "simple_trainval_proxy_exists_diagnostic": simple_proxy,
    "governance_decision": governance_decision,
    "source_candidate_contract_json": str(candidate_contract_path),
    "source_accepted_contract_json": str(accepted_contract_path),
    "source_legality_contract_json": str(legality_contract_path),
}

governance_rows = [
    {"metric": "final_q4_status", "value": governance_decision["final_q4_status"]},
    {"metric": "A0_promoted", "value": False},
    {"metric": "pairs_total", "value": pairs_total},
    {"metric": "q4_pass_n", "value": pass_n},
    {"metric": "q4_warning_n", "value": warning_n},
    {"metric": "q4_fatal_n", "value": fatal_n},
    {"metric": "q4_publication_blocker_n", "value": blocker_n},
    {"metric": "simple_trainval_proxy_exists_diagnostic", "value": simple_proxy},
    {"metric": "fatal_ids", "value": "|".join(map(str, fatal_ids))},
    {"metric": "fatal_reason_counts", "value": json.dumps(fatal_reasons, sort_keys=True)},
]

governance_csv = os.path.join(REPORT_DIR_147, "cell14_7_q4_no_promotion_governance.csv")
governance_json = os.path.join(REPORT_DIR_147, "cell14_7_q4_no_promotion_governance.json")
canonical_contract_json = os.path.join(CONTRACT_DIR_147, "cell14_7_q4_no_promotion_governance_v2_1_THESIS.json")

pd.DataFrame(governance_rows).to_csv(governance_csv, index=False)

contract = {
    "cell": "14.7",
    "version": CELL147_VERSION,
    "role": "Q4_no_promotion_governance_ledger",
    "accepted": False,
    "promoted": False,
    "summary": summary,
    "strict_contract": {
        "selection_done_here": False,
        "selection_source": "none_no_promotion",
        "manual_restore_used": False,
        "frozen_evidence_used": False,
        "q4_recomputed_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated_here": False,
        "artifact_copy_done_here": False,
        "promotion_done_here": False,
        "candidate_outputs_promoted_to_final": False,
    },
    "outputs": {
        "governance_csv": governance_csv,
        "governance_json": governance_json,
        "canonical_contract_json": canonical_contract_json,
    },
    "hashes": {
        "candidate_contract_sha256": _sha256_file_147(candidate_contract_path),
        "accepted_contract_sha256": _sha256_file_147(accepted_contract_path),
        "legality_contract_sha256": _sha256_file_147(legality_contract_path),
    },
}

_write_json_147(governance_json, summary)
_write_json_147(canonical_contract_json, contract)

globals()["CELL147_VERSION"] = CELL147_VERSION
globals()["CELL14_7_Q4_NO_PROMOTION"] = True
globals()["CELL14_7_A0_ACCEPTED"] = False
globals()["CELL14_7_FINAL_Q4_STATUS"] = "blocked_no_promotion"
globals()["CELL14_7_A0_SUMMARY"] = summary
globals()["CELL14_7_Q4_NO_PROMOTION_CONTRACT"] = contract
globals()["CELL14_7_FINAL_PROTOCOL_PATH"] = ""
globals()["CELL14_7_FINAL_CPS_PATH"] = ""
globals()["CELL14_7_Q4_NO_PROMOTION_GOVERNANCE_CSV"] = governance_csv
globals()["CELL14_7_Q4_NO_PROMOTION_GOVERNANCE_JSON"] = governance_json
globals()["CELL14_7_Q4_NO_PROMOTION_CONTRACT_JSON"] = canonical_contract_json

log(
    "[Cell14.7] Q4 no-promotion governance recorded | "
    f"final_status=blocked_no_promotion | pairs={pairs_total} | pass={pass_n} | "
    f"warning={warning_n} | fatal={fatal_n} | blockers={blocker_n} | "
    f"simple_trainval_proxy_exists_diagnostic={simple_proxy}"
)
log(f"[Cell14.7] Governance CSV: {governance_csv}")
log(f"[Cell14.7] Contract: {canonical_contract_json}")
log(
    "[Cell14.7] Contract flags | selection_done_here=False | artifact_copy_done_here=False | "
    "promotion_done_here=False | synthetic_values_mutated_here=False | candidate_outputs_promoted_to_final=False"
)
log("--- END: Cell 14.7 - Q4 no-promotion governance ledger (v2.1 strict) ---")