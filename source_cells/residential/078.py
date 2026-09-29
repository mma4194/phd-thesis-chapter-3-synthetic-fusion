# %% CELL Q3.PRE.R — Readiness guard after continuous 12.c replacement
# Purpose:
#   Confirm Q3 can proceed using the replacement continuous branch plus Binary V6.
#
# This replaces the earlier Q3.PRE scan that flagged historical diagnostic audit files.
# It checks the active replacement contracts/artifacts, not old diagnostic audit reports.

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def _q3prer_log(msg):
    print(f"[Q3.PRE.R] {msg}")


def _q3prer_path(x):
    if x is None:
        return None
    try:
        return Path(str(x)).expanduser().resolve()
    except Exception:
        return None


OUTDIR_P = _q3prer_path(globals().get("OUTDIR", None)) or _q3prer_path(os.environ.get("CPS_OUTDIR", ""))
OUT_SYN_P = _q3prer_path(globals().get("OUT_SYN", None)) or OUTDIR_P / "synthetic"
REPORT_DIR_P = _q3prer_path(globals().get("REPORT_DIR", None)) or OUTDIR_P / "reports"
CONTRACT_DIR_P = _q3prer_path(globals().get("CONTRACT_DIR", None)) or OUTDIR_P / "artifacts" / "contracts"

required = [
    REPORT_DIR_P / "cell12c3R_trainval_only_selector_summary.json",
    REPORT_DIR_P / "cell12c4R_trainval_cache_materialization_summary.json",
    REPORT_DIR_P / "cell12c5R_final_value_enforcement_summary.json",
    REPORT_DIR_P / "cell12c6R_final_publication_manifest.json",
    CONTRACT_DIR_P / "cell12c3R_trainval_only_selector_contract.json",
    CONTRACT_DIR_P / "cell12c4R_trainval_cache_materialization_contract.json",
    CONTRACT_DIR_P / "cell12c5R_final_value_enforcement_contract.json",
    CONTRACT_DIR_P / "cell12c6R_final_qa_contract.json",
    OUT_SYN_P / "IOT_FINAL_CONTINUOUS_TEST.parquet",
    OUT_SYN_P / "IOT_CONT_VALUE_MASK_TEST.parquet",
    REPORT_DIR_P / "cell12d_v6_terminal_test_qa_summary.json",
    OUT_SYN_P / "IOT_FINAL_BINARY_TEST.parquet",
]

missing = [str(p) for p in required if not p.exists()]
if missing:
    raise RuntimeError("[Q3.PRE.R] Missing required active replacement/V6 artifacts:\n" + "\n".join(missing))

# Active replacement scan.
scan_files = required[:8]
suspicious_terms = [
    "after_testqa_regression_audit",
    "final_selection_disabled_after_testqa",
    "testqa_repair",
    "test_qa_repair",
    "repair_queue_from_test",
    "revert_after_test",
    "test_based_repair",
    "test-based repair",
]
hits = []
for p in scan_files:
    txt = p.read_text(encoding="utf-8", errors="ignore").lower()
    for term in suspicious_terms:
        if term in txt:
            hits.append({"path": str(p), "term": term})
if hits:
    print(json.dumps(hits, indent=2))
    raise RuntimeError("[Q3.PRE.R] Active replacement artifacts contain suspicious TEST-policy terms.")

c6 = json.loads((REPORT_DIR_P / "cell12c6R_final_publication_manifest.json").read_text(encoding="utf-8"))
b6 = json.loads((REPORT_DIR_P / "cell12d_v6_terminal_test_qa_summary.json").read_text(encoding="utf-8"))

readiness = {
    "cell": "Q3.PRE.R",
    "role": "q3_readiness_after_continuous_replacement_and_binary_v6",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "policy": {
        "TEST_values_read": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "q3_may_proceed": True,
    },
    "continuous_replacement_summary": {
        "targets": c6.get("targets"),
        "status_counts": c6.get("status_counts"),
        "publication_status_counts": c6.get("publication_status_counts"),
        "mean_metrics": c6.get("mean_metrics"),
        "TEST_values_used_for_QA_only": c6.get("TEST_values_used_for_QA_only"),
        "synthetic_values_mutated": c6.get("synthetic_values_mutated"),
    },
    "binary_v6_summary": {
        "included_n": b6.get("included_n_trainval_only"),
        "scope_excluded_n": b6.get("scope_excluded_n_trainval_only"),
        "included_pass": b6.get("included_pass"),
        "included_warning": b6.get("included_warning"),
        "included_fatal": b6.get("included_fatal"),
        "included_publication_blocker_n": b6.get("included_publication_blocker_n"),
    },
}

report_path = REPORT_DIR_P / "cell12f_pre_q3_after_12c_replacement_readiness_report.json"
contract_path = CONTRACT_DIR_P / "cell12f_pre_q3_after_12c_replacement_readiness_contract.json"
report_path.write_text(json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8")
contract_path.write_text(json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8")

CELL12F_PRE_Q3_AFTER_12C_REPLACEMENT = readiness

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}
CFG["q3_full_iot_observability_ready"] = True
CFG["q3_uses_continuous_replacement_12cR"] = True
CFG["q3_uses_binary_v6"] = True

_q3prer_log("PASS: Q3 may proceed using 12.c.R replacement branch and Binary V6.")
print(json.dumps(readiness, indent=2, sort_keys=True))
