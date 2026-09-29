# ==========================================================
# CELL 14.6b - A0 TRAIN/VAL evidence vs TEST-QA outcome audit
# v1.0 STUDY-THESIS strict diagnostic-only policy-legality audit
#
# Role:
#   - Diagnose whether the two Cell 14.6 A0 fatal rows are distinguishable
#     using TRAIN/VAL-only evidence from Cell 14.5.
#   - This helps decide whether a future A0 policy refinement could be
#     justified without TEST-outcome cherry-picking.
#
# Strict contract:
#   - Diagnostic only.
#   - Reads Cell 14.5 TRAIN/VAL manifest evidence and Cell 14.6/14.6a QA summaries.
#   - Does NOT mutate synthetic values.
#   - Does NOT promote outputs.
#   - Does NOT select a final policy.
#   - Any threshold search here is explicitly diagnostic and TEST-outcome-informed;
#     it must not be used as final policy unless rerun as a predeclared VAL-only
#     policy in a clean pipeline.
#
# Outputs:
#   reports/cell14_6b_A0_trainval_evidence_vs_outcome_audit.csv
#   reports/cell14_6b_A0_trainval_threshold_diagnostic.csv
#   reports/cell14_6b_A0_policy_legality_assessment.csv
#   reports/cell14_6b_A0_policy_legality_contract.json
# ==========================================================

log("--- START: Cell 14.6b - A0 TRAIN/VAL evidence vs TEST-QA outcome audit (v1.0 diagnostic-only) ---")

import os
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

_required_146b = [
    "CFG", "log", "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_DF",
]
_missing_146b = [k for k in _required_146b if k not in globals()]
if _missing_146b:
    raise RuntimeError(f"[Cell14.6b] Missing required globals: {_missing_146b}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_146b = str(OUT_SYN)
REPORT_DIR_ACTIVE_146b = str(REPORT_DIR)

def _resolve_project_root_146b(outdir, report_dir, out_syn):
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
    raise RuntimeError("[Cell14.6b] Could not resolve canonical project root.")

PROJECT_ROOT_146b = _resolve_project_root_146b(OUTDIR, REPORT_DIR_ACTIVE_146b, OUT_SYN_ACTIVE_146b)
REPORT_DIR_146b = os.path.join(PROJECT_ROOT_146b, "reports")
ARTDIR_146b = os.path.join(PROJECT_ROOT_146b, "artifacts")
CONTRACT_DIR_146b = os.path.join(ARTDIR_146b, "contracts")
os.makedirs(REPORT_DIR_146b, exist_ok=True)
os.makedirs(ARTDIR_146b, exist_ok=True)
os.makedirs(CONTRACT_DIR_146b, exist_ok=True)

CELL146B_VERSION = "cell14_6b_A0_trainval_evidence_vs_testqa_outcome_audit_v1_0"

CFG["cell14_6b_version"] = CELL146B_VERSION
CFG["cell14_6b_diagnostic_only"] = True
CFG["cell14_6b_synthetic_values_mutated"] = False
CFG["cell14_6b_selection_done_here"] = False
CFG["cell14_6b_generator_fit_done_here"] = False
CFG["cell14_6b_materialization_done_here"] = False
CFG["cell14_6b_promotion_done_here"] = False
CFG["cell14_6b_TEST_QA_outcome_used_for_diagnostic_only"] = True

def _sha256_file_146b(path):
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_146b(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _write_json_146b(path, payload):
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

a0_manifest = CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_DF.copy()
if len(a0_manifest) == 0:
    raise RuntimeError("[Cell14.6b] Empty A0 manifest from Cell 14.5.")

# Prefer in-memory 14.6a audit if present, otherwise read saved CSVs.
if "CELL14_6A_A0_METRIC_THRESHOLD_AUDIT_DF" in globals() and isinstance(CELL14_6A_A0_METRIC_THRESHOLD_AUDIT_DF, pd.DataFrame):
    qa = CELL14_6A_A0_METRIC_THRESHOLD_AUDIT_DF.copy()
    qa_source = "global:CELL14_6A_A0_METRIC_THRESHOLD_AUDIT_DF"
else:
    qa_path = os.path.join(REPORT_DIR_146b, "cell14_6a_A0_metric_threshold_audit.csv")
    if not os.path.exists(qa_path):
        raise RuntimeError(f"[Cell14.6b] Missing 14.6a threshold audit CSV: {qa_path}")
    qa = pd.read_csv(qa_path)
    qa_source = qa_path

required_manifest_cols = ["repair_candidate_id", "anchor_col", "protocol_col", "protocol_tier", "repair_manifest_group"]
missing_manifest_cols = [c for c in required_manifest_cols if c not in a0_manifest.columns]
if missing_manifest_cols:
    raise RuntimeError(f"[Cell14.6b] A0 manifest missing columns: {missing_manifest_cols}")

required_qa_cols = ["repair_candidate_id", "A0_q4_status", "A0_q4_publication_blocker"]
missing_qa_cols = [c for c in required_qa_cols if c not in qa.columns]
if missing_qa_cols:
    raise RuntimeError(f"[Cell14.6b] QA audit missing columns: {missing_qa_cols}")

a0_manifest["repair_candidate_id"] = a0_manifest["repair_candidate_id"].astype(str)
qa["repair_candidate_id"] = qa["repair_candidate_id"].astype(str)

joined = a0_manifest.merge(
    qa,
    on="repair_candidate_id",
    how="left",
    suffixes=("_trainval", "_qa"),
)

if joined["A0_q4_status"].isna().any():
    missing = joined.loc[joined["A0_q4_status"].isna(), "repair_candidate_id"].astype(str).tolist()
    raise RuntimeError(f"[Cell14.6b] Some A0 manifest rows lack QA outcome: {missing}")

# Harmonize important columns.
for c in [
    "replication_score", "z_immediate_core", "z_sharpness_core",
    "coverage_core", "lag_agreement_abs_diff", "delta_immediate_core",
    "sharpness_core", "recommended_weight",
    "ETA_similarity", "lag_peak_error", "response_window_rate_error",
    "manifest_window_profile_similarity",
]:
    if c in joined.columns:
        joined[c] = pd.to_numeric(joined[c], errors="coerce")

joined["qa_is_fatal"] = joined["A0_q4_status"].astype(str).eq("fatal")
joined["qa_is_pass"] = joined["A0_q4_status"].astype(str).eq("pass")
joined["qa_is_warning"] = joined["A0_q4_status"].astype(str).eq("warning")
joined["qa_is_blocker"] = joined["A0_q4_publication_blocker"].fillna(False).astype(bool)

# Rank diagnostics: do fatal rows look weak by TRAIN/VAL evidence?
evidence_cols_high_good = [
    "replication_score", "z_immediate_core", "z_sharpness_core",
    "coverage_core", "delta_immediate_core", "sharpness_core", "recommended_weight",
]
evidence_cols_low_good = ["lag_agreement_abs_diff"]

for c in evidence_cols_high_good:
    if c in joined.columns:
        joined[f"{c}_rank_high_good"] = joined[c].rank(ascending=False, method="min")
        joined[f"{c}_fatal_mean"] = float(joined.loc[joined["qa_is_fatal"], c].mean()) if joined["qa_is_fatal"].any() else np.nan
        joined[f"{c}_nonfatal_mean"] = float(joined.loc[~joined["qa_is_fatal"], c].mean()) if (~joined["qa_is_fatal"]).any() else np.nan

for c in evidence_cols_low_good:
    if c in joined.columns:
        joined[f"{c}_rank_low_good"] = joined[c].rank(ascending=True, method="min")
        joined[f"{c}_fatal_mean"] = float(joined.loc[joined["qa_is_fatal"], c].mean()) if joined["qa_is_fatal"].any() else np.nan
        joined[f"{c}_nonfatal_mean"] = float(joined.loc[~joined["qa_is_fatal"], c].mean()) if (~joined["qa_is_fatal"]).any() else np.nan

# One-dimensional threshold diagnostics. This is TEST-outcome-informed and diagnostic only.
threshold_rows = []

def _eval_keep(mask, rule):
    mask = pd.Series(mask, index=joined.index).fillna(False).astype(bool)
    sub = joined[mask]
    if len(sub) == 0:
        return
    threshold_rows.append({
        "diagnostic_rule": rule,
        "kept_n": int(len(sub)),
        "dropped_n": int((~mask).sum()),
        "kept_pass_n": int(sub["qa_is_pass"].sum()),
        "kept_warning_n": int(sub["qa_is_warning"].sum()),
        "kept_fatal_n": int(sub["qa_is_fatal"].sum()),
        "kept_blocker_n": int(sub["qa_is_blocker"].sum()),
        "dropped_fatal_n": int(joined.loc[~mask, "qa_is_fatal"].sum()),
        "dropped_nonfatal_n": int((~mask & ~joined["qa_is_fatal"]).sum()),
        "TEST_outcome_informed_diagnostic_only": True,
        "policy_legality": "not_final_policy; may only motivate a predeclared VAL-only policy rerun",
    })

for c in evidence_cols_high_good:
    if c not in joined.columns:
        continue
    vals = sorted(joined[c].dropna().unique())
    for thr in vals:
        _eval_keep(joined[c] >= thr, f"{c} >= {thr:.6g}")

for c in evidence_cols_low_good:
    if c not in joined.columns:
        continue
    vals = sorted(joined[c].dropna().unique())
    for thr in vals:
        _eval_keep(joined[c] <= thr, f"{c} <= {thr:.6g}")

threshold_diag = pd.DataFrame(threshold_rows)
if len(threshold_diag):
    threshold_diag = threshold_diag.sort_values(
        ["kept_blocker_n", "kept_fatal_n", "dropped_nonfatal_n", "kept_n"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

# Legality assessment.
fatal_rows = joined[joined["qa_is_fatal"]].copy()
nonfatal_rows = joined[~joined["qa_is_fatal"]].copy()

fatal_ids = fatal_rows["repair_candidate_id"].astype(str).tolist()
fatal_reason_counts = Counter()
for reason in fatal_rows.get("A0_q4_reasons", pd.Series([], dtype=str)).astype(str):
    for p in reason.split("|"):
        p = p.strip()
        if p:
            fatal_reason_counts[p] += 1

best_threshold = threshold_diag.iloc[0].to_dict() if len(threshold_diag) else {}

# Determine if there is any simple train/val proxy that drops all fatal while retaining at least 6 rows and no blocker.
proxy_possible = bool(
    len(threshold_diag)
    and int(best_threshold.get("kept_blocker_n", 999)) == 0
    and int(best_threshold.get("kept_n", 0)) >= 6
)

assessment_rows = [
    {
        "assessment_item": "current_A0_promotion_state",
        "value": "blocked",
        "reason": f"fatal_rows={len(fatal_rows)}; fatal_ids={fatal_ids}",
    },
    {
        "assessment_item": "fatal_reason_counts",
        "value": dict(fatal_reason_counts),
        "reason": "Different fatal mechanisms require different treatment.",
    },
    {
        "assessment_item": "simple_trainval_proxy_exists_diagnostic",
        "value": bool(proxy_possible),
        "reason": (
            "A simple TRAIN/VAL-evidence threshold can separate fatal rows in this TEST-informed diagnostic."
            if proxy_possible else
            "No simple one-dimensional TRAIN/VAL-evidence threshold cleanly removes fatal rows while retaining a sufficiently large subset."
        ),
    },
    {
        "assessment_item": "best_threshold_diagnostic_only",
        "value": best_threshold,
        "reason": "This is diagnostic and TEST-outcome-informed; do not use directly as final policy.",
    },
    {
        "assessment_item": "recommended_next_step",
        "value": (
            "Design a predeclared VAL-only A0 policy/refinement cell before any further TEST promotion attempt."
            if proxy_possible else
            "Keep A0 blocked or inspect pair-level TRAIN/VAL profiles manually; do not promote or prune using TEST outcomes."
        ),
        "reason": "Avoid TEST-outcome cherry-picking."
    }
]
assessment = pd.DataFrame(assessment_rows)

# Outputs.
audit_csv = os.path.join(REPORT_DIR_146b, "cell14_6b_A0_trainval_evidence_vs_outcome_audit.csv")
threshold_csv = os.path.join(REPORT_DIR_146b, "cell14_6b_A0_trainval_threshold_diagnostic.csv")
assessment_csv = os.path.join(REPORT_DIR_146b, "cell14_6b_A0_policy_legality_assessment.csv")
contract_json = os.path.join(REPORT_DIR_146b, "cell14_6b_A0_policy_legality_contract.json")
canonical_contract_json = os.path.join(CONTRACT_DIR_146b, "cell14_6b_A0_policy_legality_contract_v1_0_THESIS.json")

joined.to_csv(audit_csv, index=False)
threshold_diag.to_csv(threshold_csv, index=False)
assessment.to_csv(assessment_csv, index=False)

contract = {
    "cell": "14.6b",
    "version": CELL146B_VERSION,
    "role": "A0_trainval_evidence_vs_TEST_QA_outcome_diagnostic",
    "quality_dimension": "Q4_cross_modal_consistency_A0_policy_legality",
    "input_sources": {
        "A0_manifest_global": "CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_DF",
        "QA_outcome_source": qa_source,
    },
    "summary": {
        "A0_rows": int(len(joined)),
        "fatal_n": int(joined["qa_is_fatal"].sum()),
        "warning_n": int(joined["qa_is_warning"].sum()),
        "pass_n": int(joined["qa_is_pass"].sum()),
        "fatal_ids": fatal_ids,
        "fatal_reason_counts": dict(fatal_reason_counts),
        "simple_trainval_proxy_exists_diagnostic": bool(proxy_possible),
        "best_threshold_diagnostic_only": best_threshold,
    },
    "strict_contract": {
        "diagnostic_only": True,
        "TEST_QA_outcome_used_for_diagnostic_only": True,
        "TEST_QA_outcome_used_for_final_policy": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "promotion_done_here": False,
    },
    "outputs": {
        "audit_csv": audit_csv,
        "threshold_csv": threshold_csv,
        "assessment_csv": assessment_csv,
        "contract_json": contract_json,
        "canonical_contract_json": canonical_contract_json,
    },
}

_write_json_146b(contract_json, contract)
_write_json_146b(canonical_contract_json, contract)

globals()["CELL146B_VERSION"] = CELL146B_VERSION
globals()["CELL14_6B_A0_TRAINVAL_EVIDENCE_VS_OUTCOME_AUDIT_DF"] = joined
globals()["CELL14_6B_A0_TRAINVAL_THRESHOLD_DIAGNOSTIC_DF"] = threshold_diag
globals()["CELL14_6B_A0_POLICY_LEGALITY_ASSESSMENT_DF"] = assessment
globals()["CELL14_6B_A0_POLICY_LEGALITY_CONTRACT"] = contract
globals()["CELL14_6B_A0_TRAINVAL_EVIDENCE_VS_OUTCOME_AUDIT_CSV"] = audit_csv
globals()["CELL14_6B_A0_TRAINVAL_THRESHOLD_DIAGNOSTIC_CSV"] = threshold_csv
globals()["CELL14_6B_A0_POLICY_LEGALITY_ASSESSMENT_CSV"] = assessment_csv
globals()["CELL14_6B_A0_POLICY_LEGALITY_CONTRACT_JSON"] = contract_json
globals()["CELL14_6B_A0_POLICY_LEGALITY_CONTRACT_CANONICAL_JSON"] = canonical_contract_json

log(
    "[Cell14.6b] A0 TRAIN/VAL evidence vs outcome audit complete | "
    f"A0_rows={len(joined)} | pass={int(joined['qa_is_pass'].sum())} | "
    f"warning={int(joined['qa_is_warning'].sum())} | fatal={int(joined['qa_is_fatal'].sum())} | "
    f"simple_trainval_proxy_exists_diagnostic={proxy_possible}"
)
log(f"[Cell14.6b] Fatal ids | {fatal_ids}")
log(f"[Cell14.6b] Fatal reason counts | {dict(fatal_reason_counts)}")
if len(threshold_diag):
    log(f"[Cell14.6b] Best diagnostic threshold | {threshold_diag.head(1).to_dict('records')}")
log(f"[Cell14.6b] Policy legality assessment | {assessment.to_dict('records')}")
log(
    "[Cell14.6b] Contract flags | diagnostic_only=True | "
    "TEST_QA_outcome_used_for_diagnostic_only=True | TEST_QA_outcome_used_for_final_policy=False | "
    "synthetic_values_mutated=False | selection_done_here=False | generator_fit_done_here=False | "
    "materialization_done_here=False | promotion_done_here=False"
)
log("--- END: Cell 14.6b - A0 TRAIN/VAL evidence vs TEST-QA outcome audit (v1.0 diagnostic-only) ---")