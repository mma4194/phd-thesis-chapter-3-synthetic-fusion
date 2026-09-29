# %% CELL 12.f.V5.3 — Observability-mask V5 final enforcement + Q3 readiness
# Purpose:
#   Enforce binary domain and write readiness precontract before terminal TEST QA.
#
# Safety:
#   - Does not read TEST real values.
#   - Does not change selection.

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "CELL12F_V5_MASK_TEST" not in globals():
    p = OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_TEST.parquet"
    if not p.exists():
        raise RuntimeError("[12.f.V5.3] Run 12.f.V5.2 first.")
    CELL12F_V5_MASK_TEST = pd.read_parquet(p)

mask_df = CELL12F_V5_MASK_TEST.copy()
changed_total = 0
for c in mask_df.columns:
    raw = pd.to_numeric(mask_df[c], errors="coerce").to_numpy(dtype=float)
    enforced = (np.nan_to_num(raw, nan=0.0) > 0.5).astype(np.uint8)
    changed_total += int(np.sum(np.nan_to_num(raw, nan=0.0).astype(float) != enforced.astype(float)))
    mask_df[c] = enforced

out_paths = [
    OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_FINAL_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_OBSERVABILITY_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_MASK_FINAL_TEST.parquet",
    OUT_SYN_P / "IOT_VALUE_MASK_FINAL_TEST.parquet",
]
for p in out_paths:
    mask_df.to_parquet(p, index=False)

pd.DataFrame({
    "col": mask_df.columns,
    "finite_binary_after": [bool(np.isin(mask_df[c].to_numpy(), [0, 1]).all()) for c in mask_df.columns],
    "TEST_real_values_used": False,
}).to_csv(REPORT_DIR_P / "cell12f_v5_mask_final_enforcement_audit.csv", index=False)

c6_path = REPORT_DIR_P / "cell12c6R_final_publication_manifest.json"
continuous_summary = json.loads(c6_path.read_text(encoding="utf-8")) if c6_path.exists() else {}
b6_path = REPORT_DIR_P / "cell12d_v6_terminal_test_qa_summary.json"
binary_summary = json.loads(b6_path.read_text(encoding="utf-8")) if b6_path.exists() else {}

readiness = {
    "cell": "Q3.PRE.V5",
    "role": "q3_readiness_before_mask_v5_terminal_qa",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "policy": {
        "TEST_values_read": False,
        "selection_done_here": False,
        "synthetic_values_mutated": False,
        "q3_may_proceed": True,
        "mask_v5_active": True,
        "semantic_parser": "actual_indicator_values_not_notna",
        "selector": "TRAIN_to_VAL_backtest_only",
        "same_df_te_v5_status": "development_only_unless_fresh_holdout_used",
    },
    "mask_v5": {
        "targets": int(mask_df.shape[1]),
        "shape": list(mask_df.shape),
        "changed_by_enforcement_total": int(changed_total),
        "selection_summary": globals().get("CELL12F_V5_SELECTION_SUMMARY", {}),
    },
    "continuous_replacement_summary": {
        "targets": continuous_summary.get("targets"),
        "status_counts": continuous_summary.get("status_counts"),
        "publication_status_counts": continuous_summary.get("publication_status_counts"),
        "TEST_values_used_for_QA_only": continuous_summary.get("TEST_values_used_for_QA_only"),
        "synthetic_values_mutated": continuous_summary.get("synthetic_values_mutated"),
    },
    "binary_v6_summary": {
        "included_n": binary_summary.get("included_n_trainval_only"),
        "scope_excluded_n": binary_summary.get("scope_excluded_n_trainval_only"),
        "included_pass": binary_summary.get("included_pass"),
        "included_warning": binary_summary.get("included_warning"),
        "included_fatal": binary_summary.get("included_fatal"),
        "included_publication_blocker_n": binary_summary.get("included_publication_blocker_n"),
    },
}
for p in [
    REPORT_DIR_P / "cell12f_pre_q3_v5_readiness_report.json",
    CONTRACT_DIR_P / "cell12f_pre_q3_v5_readiness_contract.json",
    REPORT_DIR_P / "cell12f_pre_q3_after_12c_replacement_readiness_report.json",
    CONTRACT_DIR_P / "cell12f_pre_q3_after_12c_replacement_readiness_contract.json",
]:
    p.write_text(json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8")

summary = {
    "cell": "12.f.V5.3",
    "role": "observability_mask_v5_final_enforcement_and_pre_q3_contract",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "shape": list(mask_df.shape),
    "changed_total": int(changed_total),
    "TEST_real_values_used": False,
    "outputs": [str(p) for p in out_paths],
    "q3_pre_contract": str(CONTRACT_DIR_P / "cell12f_pre_q3_v5_readiness_contract.json"),
}
(REPORT_DIR_P / "cell12f_v5_final_enforcement_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
(CONTRACT_DIR_P / "cell12f_v5_final_enforcement_contract.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12F_V5_MASK_FINAL = mask_df
CELL12F_V5_PRE_Q3 = readiness

_f5_log(f"V5.3 enforcement complete | shape={mask_df.shape} | changed={changed_total}")

