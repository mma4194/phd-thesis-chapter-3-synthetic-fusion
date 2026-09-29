# %% CELL 12.f.V5.5R — Q3 V5 -> Cell 12.g legacy-contract compatibility handoff
# Purpose:
#   Publish the legacy in-memory 12.f.5 contract shape expected by Cell 12.g,
#   while preserving Q3 V5 as the active observability/mask branch.
#
# Why:
#   Cell 12.g expects:
#       CELL12F5_MASK_CONTRACT["version"]
#       CELL12F5_MASK_CONTRACT["metric_summary"]
#       CELL12F5_MASK_CONTRACT["publication_status_counts_after_12f5"]
#       CELL12F5_MASK_CONTRACT["blocker_origin_counts"]
#
#   Q3 V5 wrote scientifically valid V5 artifacts, but the schema is newer and
#   does not contain the old version string expected by 12.g.
#
# Safety:
#   - Does not mutate synthetic data.
#   - Does not rerun selection, fitting, materialization, or TEST QA.
#   - Reads existing Q3 V5 terminal artifacts only.
#   - Does not change pass/warning/fatal/blocker outcomes.
#   - Writes a separate compatibility contract and leaves the active V5 evidence intact.

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

def _v5handoffR_log(msg):
    print(f"[12.f.V5.5R handoff] {msg}")

# ---------------------------------------------------------------------
# 0. Resolve roots
# ---------------------------------------------------------------------
for name in ["OUTDIR", "OUT_SYN", "REPORT_DIR", "CONTRACT_DIR"]:
    if name not in globals():
        raise RuntimeError(f"[12.f.V5.5R] Missing required global: {name}")

OUTDIR_P = Path(str(OUTDIR)).expanduser().resolve()
OUT_SYN_P = Path(str(OUT_SYN)).expanduser().resolve()
REPORT_DIR_P = Path(str(REPORT_DIR)).expanduser().resolve()
CONTRACT_DIR_P = Path(str(CONTRACT_DIR)).expanduser().resolve()

# ---------------------------------------------------------------------
# 1. Locate active Q3 V5 artifacts
# ---------------------------------------------------------------------
mask_contract_candidates = [
    CONTRACT_DIR_P / "cell12f_v5_mask_final_test_qa_contract.json",
    CONTRACT_DIR_P / "cell12f5_mask_final_test_qa_contract_v1_1_THESIS.json",
]

mask_contract_path = None
for p in mask_contract_candidates:
    if p.exists():
        mask_contract_path = p
        break

if mask_contract_path is None:
    raise RuntimeError(
        "[12.f.V5.5R] Missing Q3 V5 / 12.f.5 mask QA contract. Tried:\n"
        + "\n".join(str(p) for p in mask_contract_candidates)
    )

mask_status_path = REPORT_DIR_P / "cell12f5_mask_publication_status.csv"
mask_metrics_path = REPORT_DIR_P / "cell12f5_mask_final_test_qa_metrics.csv"
mask_blocker_path = REPORT_DIR_P / "cell12f5_mask_blocker_origin_audit.csv"
q3_post_v5_contract_path = CONTRACT_DIR_P / "q3_post_v5_observability_mask_post_qa_contract.json"

required = [
    mask_status_path,
    mask_metrics_path,
    mask_blocker_path,
    q3_post_v5_contract_path,
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise RuntimeError("[12.f.V5.5R] Missing required active Q3 V5 artifact(s):\n" + "\n".join(missing))

with open(mask_contract_path, "r", encoding="utf-8") as f:
    raw_mask_contract = json.load(f)

with open(q3_post_v5_contract_path, "r", encoding="utf-8") as f:
    q3_post_v5_contract = json.load(f)

mask_status_df = pd.read_csv(mask_status_path)
mask_metrics_df = pd.read_csv(mask_metrics_path)
mask_blocker_df = pd.read_csv(mask_blocker_path)

# ---------------------------------------------------------------------
# 2. Normalize schemas
# ---------------------------------------------------------------------
for df in [mask_status_df, mask_metrics_df, mask_blocker_df]:
    if "column" not in df.columns and "col" in df.columns:
        df["column"] = df["col"]
    if "col" not in df.columns and "column" in df.columns:
        df["col"] = df["column"]

required_status_cols = [
    "col",
    "column",
    "mask_publication_status_after_12f5",
    "mask_publication_blocker_after_12f5",
]
missing_status = [c for c in required_status_cols if c not in mask_status_df.columns]
if missing_status:
    raise RuntimeError(f"[12.f.V5.5R] Publication status file missing columns: {missing_status}")

def _to_bool_series_12f5R(s):
    if s.dtype == bool:
        return s.astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])

mask_status_df["mask_publication_blocker_after_12f5"] = _to_bool_series_12f5R(
    mask_status_df["mask_publication_blocker_after_12f5"]
)

if "mask_publication_blocker_after_12f5" in mask_blocker_df.columns:
    mask_blocker_df["mask_publication_blocker_after_12f5"] = _to_bool_series_12f5R(
        mask_blocker_df["mask_publication_blocker_after_12f5"]
    )

# ---------------------------------------------------------------------
# 3. Compute carried-forward Q3 status from current final-root artifacts
# ---------------------------------------------------------------------
targets_n = int(len(mask_status_df))
metrics_n = int(len(mask_metrics_df))
blocker_rows_n = int(len(mask_blocker_df))

if metrics_n != targets_n:
    raise RuntimeError(
        f"[12.f.V5.5R] Metrics rows ({metrics_n}) != publication status rows ({targets_n})."
    )

publication_status_counts = (
    mask_status_df["mask_publication_status_after_12f5"]
    .astype(str)
    .value_counts()
    .to_dict()
)

publication_blocker_n = int(mask_status_df["mask_publication_blocker_after_12f5"].sum())

if "test_qa_status" in mask_metrics_df.columns:
    qa_status_counts = (
        mask_metrics_df["test_qa_status"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
else:
    qa_status_counts = {}

if "blocker_origin" in mask_blocker_df.columns:
    blocker_origin_counts = (
        mask_blocker_df["blocker_origin"]
        .astype(str)
        .value_counts()
        .to_dict()
    )
else:
    blocker_origin_counts = {}

# Count active included fatal blockers. V5 names this origin with v5.
new_test_qa_blocker_n = int(
    blocker_origin_counts.get("new_12f5_test_qa_blocker", 0)
    + blocker_origin_counts.get("new_12f_v5_test_qa_blocker", 0)
)

# If the blocker-origin CSV does not carry the exact origin labels, fall back safely.
if new_test_qa_blocker_n == 0 and publication_blocker_n > 0:
    new_test_qa_blocker_n = publication_blocker_n

upstream_publication_blocker_n = int(
    blocker_origin_counts.get("carried_upstream_12f3_blocker", 0)
    + blocker_origin_counts.get("upstream_publication_blocker", 0)
)

split_drift_blocker_n = int(
    blocker_origin_counts.get("trainval_to_test_split_or_regime_drift", 0)
    + blocker_origin_counts.get("split_drift_blocker", 0)
)

# Mean metrics: use current V5 contract if present; otherwise compute from metrics CSV.
existing_mean_metrics = raw_mask_contract.get("mean_metrics", {}) or q3_post_v5_contract.get("mean_metrics", {}) or {}

def _mean_col(df, col):
    if col in df.columns:
        return float(pd.to_numeric(df[col], errors="coerce").mean())
    return None

computed_mean_metrics = {
    "mean_mask_rate_error": _mean_col(mask_metrics_df, "mask_rate_error"),
    "mean_runlength_ks": _mean_col(mask_metrics_df, "runlength_ks"),
    "mean_dwell_wasserstein": _mean_col(mask_metrics_df, "dwell_wasserstein"),
    "mean_mask_c2st_auc": _mean_col(mask_metrics_df, "mask_c2st_auc"),
}
computed_mean_metrics = {k: v for k, v in computed_mean_metrics.items() if v is not None and np.isfinite(v)}

mean_metrics = dict(existing_mean_metrics)
mean_metrics.update({k: v for k, v in computed_mean_metrics.items() if k not in mean_metrics})

metric_summary_legacy = dict(mean_metrics)
metric_summary_legacy.update({
    "targets": targets_n,
    "pass_n": int(qa_status_counts.get("pass", 0)),
    "warning_n": int(qa_status_counts.get("warning", 0)),
    "fatal_n": int(qa_status_counts.get("fatal", 0)),
    "publication_blocker_n": publication_blocker_n,
    "new_12f5_test_qa_blocker_n": new_test_qa_blocker_n,
    "upstream_publication_blocker_n": upstream_publication_blocker_n,
    "split_drift_blocker_n": split_drift_blocker_n,
})

# ---------------------------------------------------------------------
# 4. Validate active branch is not invalid V2 semantic-collapse
# ---------------------------------------------------------------------
selected_text_cols = [
    c for c in [
        "selected_generator",
        "materializer_class_v2",
        "materializer_class_v3",
        "materializer_class_v4",
        "materializer_class_v5",
    ]
    if c in mask_status_df.columns
]
selected_text = " ".join(
    mask_status_df[c].astype(str).str.cat(sep=" ")
    for c in selected_text_cols
)

if "MaskExactObservedConstantV2" in selected_text or "exact_observed_constant" in selected_text:
    raise RuntimeError(
        "[12.f.V5.5R] Refusing to hand off invalid V2 semantic-collapse artifact. "
        "The active branch must be Q3 V5."
    )

if "v5" not in json.dumps(q3_post_v5_contract, sort_keys=True).lower():
    _v5handoffR_log(
        "WARNING: q3_post_v5_contract did not obviously contain 'v5' text, "
        "but path and required files are V5-specific. Continuing with artifact-based checks."
    )

# ---------------------------------------------------------------------
# 5. Build legacy-compatible 12.f.5 contract for Cell 12.g
# ---------------------------------------------------------------------
legacy_version = "cell12f5_observability_mask_final_test_qa_only_strict_v1_1__q3_v5_compat"

legacy_contract = dict(raw_mask_contract)
legacy_contract.update({
    "cell": "12.f.V5.4",
    "version": legacy_version,
    "contract_version": legacy_version,
    "active_mask_branch": "12.f.V5",
    "active_q3_branch": "Q3.POST.V5",
    "role": "observability_mask_v5_terminal_TEST_QA_only_legacy_12g_compatible",
    "targets": targets_n,
    "TEST_real_values_used_for_QA_only": True,
    "synthetic_values_mutated": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "post_TEST_repair_done_here": False,
    "test_qa_done_here": True,
    "metric_summary": metric_summary_legacy,
    "mean_metrics": mean_metrics,
    "qa_status_counts": qa_status_counts,
    "publication_status_counts_after_12f5": publication_status_counts,
    "publication_status_counts": publication_status_counts,
    "blocker_origin_counts": blocker_origin_counts,
    "publication_blocker_n": publication_blocker_n,
    "q3_final_status": q3_post_v5_contract.get("final_status", "BLOCKED_PARTIAL"),
    "q3_claim_scope": q3_post_v5_contract.get("claim_scope", "partial_admissible_only"),
    "q3_full_scope_ready": bool(q3_post_v5_contract.get("full_scope_ready", False)),
    "compatibility_note": (
        "This contract is a 12.g legacy-shape compatibility view over Q3 V5 terminal QA artifacts. "
        "It does not modify synthetic values, selection, materialization, or TEST QA outcomes."
    ),
    "source_contract_path": str(mask_contract_path),
    "q3_post_v5_contract_path": str(q3_post_v5_contract_path),
})

# ---------------------------------------------------------------------
# 6. Publish globals expected by 12.g
# ---------------------------------------------------------------------
CELL12F5_MASK_CONTRACT = legacy_contract
CELL12F5_MASK_PUBLICATION_STATUS_DF = mask_status_df

CELL12F5_MASK_FINAL_TEST_QA_METRICS_DF = mask_metrics_df
CELL12F5_MASK_BLOCKER_ORIGIN_DF = mask_blocker_df
CELL12F5_MASK_FINAL_TEST_QA_CONTRACT_PATH = str(mask_contract_path)
CELL12F5_MASK_PUBLICATION_STATUS_PATH = str(mask_status_path)
CELL12F5_MASK_FINAL_TEST_QA_METRICS_PATH = str(mask_metrics_path)
CELL12F5_MASK_BLOCKER_ORIGIN_PATH = str(mask_blocker_path)
CELL12F5_Q3_POST_CONTRACT = q3_post_v5_contract
CELL12F5_Q3_POST_CONTRACT_PATH = str(q3_post_v5_contract_path)
CELL12F5_ACTIVE_BRANCH = "12.f.V5"

# Aliases for newer/governance cells.
CELL12F_MASK_CONTRACT = CELL12F5_MASK_CONTRACT
CELL12F_MASK_PUBLICATION_STATUS_DF = CELL12F5_MASK_PUBLICATION_STATUS_DF
CELL_Q3_ACTIVE_POST_CONTRACT = q3_post_v5_contract

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}

CFG["active_q3_branch"] = "Q3.POST.V5"
CFG["active_mask_branch"] = "12.f.V5"
CFG["q3_claim_scope"] = legacy_contract["q3_claim_scope"]
CFG["q3_full_scope_ready"] = bool(legacy_contract["q3_full_scope_ready"])
CFG["q3_publication_blocker_n"] = int(publication_blocker_n)
CFG["cell12g_mask_handoff_source"] = "12.f.V5.5R"
CFG["cell12g_q3_legacy_contract_compatibility"] = True

# ---------------------------------------------------------------------
# 7. Write compatibility contract/receipt
# ---------------------------------------------------------------------
compat_contract_path = CONTRACT_DIR_P / "cell12f_v5_to_cell12g_legacy_compat_contract.json"
compat_report_path = REPORT_DIR_P / "cell12f_v5_to_cell12g_legacy_compat_report.json"

compat_payload = {
    "cell": "12.f.V5.5R",
    "role": "q3_v5_to_cell12g_legacy_contract_compatibility_handoff",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "active_q3_branch": "Q3.POST.V5",
    "active_mask_branch": "12.f.V5",
    "targets": targets_n,
    "publication_blocker_n": publication_blocker_n,
    "qa_status_counts": qa_status_counts,
    "publication_status_counts_after_12f5": publication_status_counts,
    "blocker_origin_counts": blocker_origin_counts,
    "metric_summary": metric_summary_legacy,
    "legacy_version_published": legacy_version,
    "globals_published": [
        "CELL12F5_MASK_CONTRACT",
        "CELL12F5_MASK_PUBLICATION_STATUS_DF",
        "CELL12F5_MASK_FINAL_TEST_QA_METRICS_DF",
        "CELL12F5_MASK_BLOCKER_ORIGIN_DF",
        "CELL12F5_Q3_POST_CONTRACT",
    ],
    "policy": {
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "TEST_QA_artifacts_read_only": True,
        "legacy_contract_shape_only": True,
    },
    "sources": {
        "mask_contract": str(mask_contract_path),
        "mask_status": str(mask_status_path),
        "mask_metrics": str(mask_metrics_path),
        "mask_blocker": str(mask_blocker_path),
        "q3_post_v5_contract": str(q3_post_v5_contract_path),
    },
}

for p, obj in [
    (compat_contract_path, legacy_contract),
    (compat_report_path, compat_payload),
]:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

CELL12F5_TO_CELL12G_HANDOFF = compat_payload
CELL12F5_TO_CELL12G_COMPAT_CONTRACT_PATH = str(compat_contract_path)

_v5handoffR_log("Published legacy-compatible CELL12F5_MASK_CONTRACT.")
_v5handoffR_log(f"version = {legacy_contract['version']}")
_v5handoffR_log(f"targets = {targets_n}")
_v5handoffR_log(f"publication_blocker_n = {publication_blocker_n}")
_v5handoffR_log(f"qa_status_counts = {qa_status_counts}")
_v5handoffR_log(f"publication_status_counts_after_12f5 = {publication_status_counts}")
_v5handoffR_log(f"blocker_origin_counts = {blocker_origin_counts}")
_v5handoffR_log(f"Compatibility contract: {compat_contract_path}")
_v5handoffR_log("PASS: Cell 12.g legacy contract requirements are ready.")