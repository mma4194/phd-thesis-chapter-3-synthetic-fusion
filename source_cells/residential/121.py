# ==========================================================
# CELL 14.6D0 - Q4 A0 Zigbee blocker root-cause diagnostic
# v1.0 STUDY-THESIS strict diagnostic-only / no mutation / no promotion
#
# Purpose:
#   Diagnose why A0 Zigbee-safe Q4 still has blockers.
#
# This cell:
#   - reads Cell 14.6 TEST-QA metrics,
#   - reads Cell 14.6R0 TRAIN/VAL policy-grid evidence,
#   - compares TEST blockers against VAL-only feasibility,
#   - identifies whether failures are ETA-shape, manifest-window,
#     lag, response-rate, event-count, or materializer-policy failures,
#   - writes diagnostic ledgers only.
#
# This cell does NOT:
#   - mutate synthetic data,
#   - select a final policy,
#   - promote A0,
#   - remove pairs,
#   - change thresholds,
#   - use TEST outcome as a final policy rule.
# ==========================================================

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    log("--- START: Cell 14.6D0 - Q4 A0 Zigbee blocker root-cause diagnostic ---")
except Exception:
    print("--- START: Cell 14.6D0 - Q4 A0 Zigbee blocker root-cause diagnostic ---")

_required_146d0 = ["REPORT_DIR", "OUTDIR"]
_missing_146d0 = [k for k in _required_146d0 if k not in globals()]
if _missing_146d0:
    raise RuntimeError(f"[Cell14.6D0] Missing required globals: {_missing_146d0}")

REPORT_DIR_D0 = Path(str(REPORT_DIR)).expanduser().resolve()
OUTDIR_D0 = Path(str(OUTDIR)).expanduser().resolve()
ARTDIR_D0 = OUTDIR_D0 / "artifacts"
CONTRACT_DIR_D0 = ARTDIR_D0 / "contracts"
REPORT_DIR_D0.mkdir(parents=True, exist_ok=True)
ARTDIR_D0.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_D0.mkdir(parents=True, exist_ok=True)

CELL146D0_VERSION = "cell14_6D0_q4_a0_zigbee_blocker_diagnostic_v1_0_THESIS"

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def _read_csv_146d0(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise RuntimeError(f"[Cell14.6D0] Required file missing: {path}")
        return pd.DataFrame()
    if path.is_dir():
        if required:
            raise RuntimeError(f"[Cell14.6D0] Expected file but found directory: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

def _json_safe_146d0(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe_146d0(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe_146d0(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _json_safe_146d0(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_safe_146d0(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _json_safe_146d0(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_146d0(path, payload):
    Path(path).write_text(
        json.dumps(_json_safe_146d0(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

def _first_existing_col_146d0(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in lower_map:
            return lower_map[str(c).lower()]
    return None

def _num_146d0(df, col, default=np.nan):
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")

def _status_order_146d0(x):
    x = str(x).strip().lower()
    if x == "pass":
        return 0
    if x == "warning":
        return 1
    if x in {"fatal", "blocker", "blocked"}:
        return 2
    if "not" in x:
        return 3
    return 4

def _bool_series_146d0(df, col):
    if col is None or col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(
        ["1", "true", "yes", "y", "fatal", "blocker", "blocked"]
    )

# ----------------------------------------------------------
# Inputs from current Q4 run
# ----------------------------------------------------------
a0_metrics_path = REPORT_DIR_D0 / "cell14_6_A0_candidate_manifest_q4_pair_metrics.csv"
a0_profile_path = REPORT_DIR_D0 / "cell14_6_A0_profile_audit.csv"
a0_mat_path = REPORT_DIR_D0 / "cell14_6_A0_materialization_audit.csv"
a0_drift_path = REPORT_DIR_D0 / "cell14_6_A0_target_protocol_drift_audit.csv"
a0_prepost_path = REPORT_DIR_D0 / "cell14_6_A0_candidate_manifest_q4_prepost_delta.csv"

r0_policy_grid_path = REPORT_DIR_D0 / "cell14_6R0_A0_trainval_policy_grid.csv"
r0_pair_scores_path = REPORT_DIR_D0 / "cell14_6R0_A0_trainval_pair_scores.csv"

a0 = _read_csv_146d0(a0_metrics_path, required=True)
profile = _read_csv_146d0(a0_profile_path, required=False)
mat = _read_csv_146d0(a0_mat_path, required=False)
drift = _read_csv_146d0(a0_drift_path, required=False)
prepost = _read_csv_146d0(a0_prepost_path, required=False)

grid = _read_csv_146d0(r0_policy_grid_path, required=False)
pair_scores = _read_csv_146d0(r0_pair_scores_path, required=False)

if "repair_candidate_id" not in a0.columns:
    raise RuntimeError("[Cell14.6D0] A0 metrics missing repair_candidate_id.")

# ----------------------------------------------------------
# Normalize TEST-QA status and metric flags
# ----------------------------------------------------------
status_col = _first_existing_col_146d0(
    a0,
    ["A0_q4_status", "q4_status", "status", "pair_status", "readiness_status"],
)
reason_col = _first_existing_col_146d0(
    a0,
    ["A0_q4_reasons", "q4_reasons", "reason", "reasons", "failure_reason"],
)
blocker_col = _first_existing_col_146d0(
    a0,
    ["A0_q4_publication_blocker", "publication_blocker", "q4_publication_blocker", "blocker"],
)

if status_col is None:
    raise RuntimeError(f"[Cell14.6D0] Could not find status column. Available columns: {list(a0.columns)}")

a0["TEST_status_norm"] = a0[status_col].astype(str).str.strip().str.lower()
a0["TEST_reason_norm"] = a0[reason_col].astype(str) if reason_col else ""
a0["TEST_publication_blocker_norm"] = _bool_series_146d0(a0, blocker_col)

eta_col = _first_existing_col_146d0(a0, ["ETA_similarity", "eta_similarity"])
eta_corr_col = _first_existing_col_146d0(a0, ["ETA_corr", "eta_corr"])
eta_norm_mae_col = _first_existing_col_146d0(a0, ["ETA_norm_mae", "eta_norm_mae"])
manifest_col = _first_existing_col_146d0(a0, ["manifest_window_profile_similarity", "manifest_profile_similarity"])
manifest_corr_col = _first_existing_col_146d0(a0, ["manifest_window_profile_corr", "manifest_profile_corr"])
manifest_norm_mae_col = _first_existing_col_146d0(a0, ["manifest_window_profile_norm_mae", "manifest_profile_norm_mae"])
lag_col = _first_existing_col_146d0(a0, ["lag_peak_error", "lag_peak_error_s"])
resp_col = _first_existing_col_146d0(a0, ["response_window_rate_error", "response_rate_error"])
real_resp_col = _first_existing_col_146d0(a0, ["real_response_window_rate"])
syn_resp_col = _first_existing_col_146d0(a0, ["syn_response_window_rate"])

a0["ETA_similarity_num"] = _num_146d0(a0, eta_col)
a0["ETA_corr_num"] = _num_146d0(a0, eta_corr_col)
a0["ETA_norm_mae_num"] = _num_146d0(a0, eta_norm_mae_col)
a0["manifest_similarity_num"] = _num_146d0(a0, manifest_col)
a0["manifest_corr_num"] = _num_146d0(a0, manifest_corr_col)
a0["manifest_norm_mae_num"] = _num_146d0(a0, manifest_norm_mae_col)
a0["lag_peak_error_num"] = _num_146d0(a0, lag_col)
a0["response_window_rate_error_num"] = _num_146d0(a0, resp_col)
a0["real_response_window_rate_num"] = _num_146d0(a0, real_resp_col)
a0["syn_response_window_rate_num"] = _num_146d0(a0, syn_resp_col)
a0["response_rate_delta_syn_minus_real"] = (
    a0["syn_response_window_rate_num"] - a0["real_response_window_rate_num"]
)

# These thresholds mirror the Q4 gate. They are diagnostic labels only.
ETA_FATAL = 0.50
ETA_PASS = 0.70
PROF_FATAL = 0.50
PROF_PASS = 0.70
LAG_FATAL = 5.0
LAG_PASS = 2.0
RESP_FATAL = 0.25
RESP_PASS = 0.10

a0["flag_eta_fatal"] = a0["ETA_similarity_num"].lt(ETA_FATAL).fillna(False)
a0["flag_eta_warning"] = (
    a0["ETA_similarity_num"].ge(ETA_FATAL).fillna(False)
    & a0["ETA_similarity_num"].lt(ETA_PASS).fillna(False)
)
a0["flag_manifest_fatal"] = a0["manifest_similarity_num"].lt(PROF_FATAL).fillna(False)
a0["flag_manifest_warning"] = (
    a0["manifest_similarity_num"].ge(PROF_FATAL).fillna(False)
    & a0["manifest_similarity_num"].lt(PROF_PASS).fillna(False)
)
a0["flag_lag_fatal"] = a0["lag_peak_error_num"].gt(LAG_FATAL).fillna(False)
a0["flag_lag_warning"] = (
    a0["lag_peak_error_num"].gt(LAG_PASS).fillna(False)
    & a0["lag_peak_error_num"].le(LAG_FATAL).fillna(False)
)
a0["flag_response_fatal"] = a0["response_window_rate_error_num"].gt(RESP_FATAL).fillna(False)
a0["flag_response_warning"] = (
    a0["response_window_rate_error_num"].gt(RESP_PASS).fillna(False)
    & a0["response_window_rate_error_num"].le(RESP_FATAL).fillna(False)
)

a0["wide_eta_only_blocker_pattern"] = (
    a0["flag_eta_fatal"]
    & ~a0["flag_manifest_fatal"]
    & ~a0["flag_lag_fatal"]
    & ~a0["flag_response_fatal"]
)

def _root_cause_146d0(row):
    if str(row["TEST_status_norm"]).lower() not in {"fatal", "blocker", "blocked"}:
        if bool(row["flag_eta_warning"]) or bool(row["flag_response_warning"]) or bool(row["flag_manifest_warning"]) or bool(row["flag_lag_warning"]):
            return "warning_only; inspect calibration but not a blocker"
        return "pass_or_nonblocking"
    if bool(row["wide_eta_only_blocker_pattern"]):
        return "ETA_shape_fatal_only; manifest/lag/response are nonfatal; inspect wide-window profile shape"
    if bool(row["flag_manifest_fatal"]):
        return "manifest_window_profile_fatal; claimed response window itself is not preserved"
    if bool(row["flag_lag_fatal"]):
        return "lag_peak_fatal; event-response timing offset too large"
    if bool(row["flag_response_fatal"]):
        return "response_rate_fatal; response-window hit rate mismatch too large"
    if bool(row["flag_eta_fatal"]):
        return "ETA_fatal_mixed_or_unclassified"
    return "fatal_unclassified"

a0["diagnostic_root_cause"] = a0.apply(_root_cause_146d0, axis=1)

# ----------------------------------------------------------
# Join TRAIN/VAL policy evidence
# ----------------------------------------------------------
selected_policy_id = str(
    globals().get("CFG", {}).get("cell14_6R0_policy_id", "")
    if "CFG" in globals()
    else ""
)

if selected_policy_id == "" and not grid.empty and "policy_id" in grid.columns:
    selected_policy_id = str(grid.iloc[0]["policy_id"])

val_selected = pd.DataFrame()
val_best_pair = pd.DataFrame()
val_feas = pd.DataFrame()

if not pair_scores.empty and "repair_candidate_id" in pair_scores.columns:
    pair_scores = pair_scores.copy()
    pair_scores["repair_candidate_id"] = pair_scores["repair_candidate_id"].astype(str)

    if "q4_status" in pair_scores.columns:
        pair_scores["VAL_status_order"] = pair_scores["q4_status"].apply(_status_order_146d0)
    else:
        pair_scores["VAL_status_order"] = 99

    if "publication_blocker" in pair_scores.columns:
        pair_scores["VAL_publication_blocker_bool"] = _bool_series_146d0(pair_scores, "publication_blocker")
    else:
        pair_scores["VAL_publication_blocker_bool"] = pair_scores["VAL_status_order"].eq(2)

    # Pair-level feasibility across all policies in the current grid.
    val_feas = (
        pair_scores.groupby("repair_candidate_id")
        .agg(
            VAL_policy_n=("policy_id", "nunique") if "policy_id" in pair_scores.columns else ("repair_candidate_id", "size"),
            VAL_nonfatal_policy_n=("VAL_publication_blocker_bool", lambda x: int((~x).sum())),
            VAL_blocking_policy_n=("VAL_publication_blocker_bool", lambda x: int(x.sum())),
            VAL_best_status_order=("VAL_status_order", "min"),
            VAL_max_ETA_similarity=("ETA_similarity", "max") if "ETA_similarity" in pair_scores.columns else ("VAL_status_order", "size"),
            VAL_max_manifest_similarity=("manifest_window_profile_similarity", "max") if "manifest_window_profile_similarity" in pair_scores.columns else ("VAL_status_order", "size"),
            VAL_min_lag_peak_error=("lag_peak_error", "min") if "lag_peak_error" in pair_scores.columns else ("VAL_status_order", "size"),
            VAL_min_response_window_rate_error=("response_window_rate_error", "min") if "response_window_rate_error" in pair_scores.columns else ("VAL_status_order", "size"),
        )
        .reset_index()
    )

    # Best VAL policy per pair under the current policy family.
    sort_cols = []
    asc = []
    for c, a in [
        ("VAL_status_order", True),
        ("response_window_rate_error", True),
        ("lag_peak_error", True),
        ("ETA_similarity", False),
        ("manifest_window_profile_similarity", False),
    ]:
        if c in pair_scores.columns:
            sort_cols.append(c)
            asc.append(a)

    if sort_cols:
        val_best_pair = (
            pair_scores.sort_values(sort_cols, ascending=asc)
            .groupby("repair_candidate_id", as_index=False)
            .head(1)
            .copy()
        )
        keep_cols = [
            c for c in [
                "repair_candidate_id", "policy_id", "q4_status", "q4_reasons",
                "ETA_similarity", "manifest_window_profile_similarity",
                "lag_peak_error", "response_window_rate_error",
                "real_response_window_rate", "syn_response_window_rate",
            ] if c in val_best_pair.columns
        ]
        val_best_pair = val_best_pair[keep_cols].rename(columns={
            "policy_id": "VAL_best_pair_policy_id",
            "q4_status": "VAL_best_pair_status",
            "q4_reasons": "VAL_best_pair_reasons",
            "ETA_similarity": "VAL_best_pair_ETA_similarity",
            "manifest_window_profile_similarity": "VAL_best_pair_manifest_similarity",
            "lag_peak_error": "VAL_best_pair_lag_peak_error",
            "response_window_rate_error": "VAL_best_pair_response_window_rate_error",
            "real_response_window_rate": "VAL_best_pair_real_response_rate",
            "syn_response_window_rate": "VAL_best_pair_syn_response_rate",
        })

    # Selected global policy row per pair.
    if selected_policy_id and "policy_id" in pair_scores.columns:
        val_selected = pair_scores[pair_scores["policy_id"].astype(str).eq(selected_policy_id)].copy()
        keep_cols = [
            c for c in [
                "repair_candidate_id", "policy_id", "q4_status", "q4_reasons",
                "ETA_similarity", "manifest_window_profile_similarity",
                "lag_peak_error", "response_window_rate_error",
                "real_response_window_rate", "syn_response_window_rate",
            ] if c in val_selected.columns
        ]
        val_selected = val_selected[keep_cols].rename(columns={
            "policy_id": "VAL_selected_policy_id",
            "q4_status": "VAL_selected_status",
            "q4_reasons": "VAL_selected_reasons",
            "ETA_similarity": "VAL_selected_ETA_similarity",
            "manifest_window_profile_similarity": "VAL_selected_manifest_similarity",
            "lag_peak_error": "VAL_selected_lag_peak_error",
            "response_window_rate_error": "VAL_selected_response_window_rate_error",
            "real_response_window_rate": "VAL_selected_real_response_rate",
            "syn_response_window_rate": "VAL_selected_syn_response_rate",
        })

# ----------------------------------------------------------
# Add profile/materialization diagnostics
# ----------------------------------------------------------
root = a0.copy()
root["repair_candidate_id"] = root["repair_candidate_id"].astype(str)

if not profile.empty and "repair_candidate_id" in profile.columns:
    prof_cols = [
        c for c in [
            "repair_candidate_id", "anchor_col", "protocol_col", "profile_key",
            "lag_lo", "lag_hi", "profile_lag_lo", "profile_lag_hi",
            "train_events", "val_events", "combined_events",
            "train_coverage", "val_coverage",
            "recommended_weight", "profile_valid", "rejection_reason",
        ] if c in profile.columns
    ]
    root = root.merge(
        profile[prof_cols].drop_duplicates("repair_candidate_id"),
        on="repair_candidate_id",
        how="left",
        suffixes=("", "_profile"),
    )

if not val_selected.empty:
    root = root.merge(val_selected, on="repair_candidate_id", how="left")

if not val_best_pair.empty:
    root = root.merge(val_best_pair, on="repair_candidate_id", how="left")

if not val_feas.empty:
    root = root.merge(val_feas, on="repair_candidate_id", how="left")

if not mat.empty and "repair_candidate_id" in mat.columns:
    mat_cols = [
        c for c in [
            "repair_candidate_id", "applied", "events_applied", "target_changed_n",
            "target_changed_inside_legal_window_n", "target_changed_outside_legal_window_n",
            "negative_n", "noninteger_n", "inactive_finite_n",
        ] if c in mat.columns
    ]
    mat_small = mat[mat_cols].copy()
    # Multiple materialization rows can exist; aggregate conservatively.
    agg_spec = {}
    for c in mat_small.columns:
        if c == "repair_candidate_id":
            continue
        if c == "applied":
            agg_spec[c] = lambda x: bool(pd.Series(x).astype(str).str.lower().isin(["true", "1", "yes"]).any())
        else:
            agg_spec[c] = "sum"
    mat_agg = mat_small.groupby("repair_candidate_id").agg(agg_spec).reset_index()
    root = root.merge(mat_agg, on="repair_candidate_id", how="left", suffixes=("", "_mat"))

# ----------------------------------------------------------
# Add diagnostic action recommendation
# ----------------------------------------------------------
def _recommend_146d0(row):
    status = str(row.get("TEST_status_norm", "")).lower()
    nonfatal_policies = row.get("VAL_nonfatal_policy_n", np.nan)

    if status not in {"fatal", "blocker", "blocked"}:
        if row.get("flag_eta_warning", False):
            return "nonblocking; possible ETA tuning, but no publication blocker"
        if row.get("flag_response_warning", False):
            return "nonblocking; response-rate calibration warning"
        return "nonblocking"

    if pd.notna(nonfatal_policies) and float(nonfatal_policies) <= 0:
        return "not recoverable under current TRAIN/VAL policy family; need new predeclared materializer family or predeclared VAL eligibility rule"

    if row.get("wide_eta_only_blocker_pattern", False):
        return "inspect ETA wide-window shape; candidate for new VAL-only profile-shape materializer or justified manifest-window-only claim variant"

    if row.get("flag_response_fatal", False):
        return "response-rate scaling/calibration problem; add VAL-only response-gain policy family"

    if row.get("flag_lag_fatal", False):
        return "lag alignment problem; add VAL-only lag-shift or phase alignment family"

    if row.get("flag_manifest_fatal", False):
        return "manifest-window relationship itself fails; do not promote unless VAL-only materializer fixes it"

    return "fatal but mechanism unclear; inspect curves and raw event counts"

root["recommended_next_diagnostic_step"] = root.apply(_recommend_146d0, axis=1)

# ----------------------------------------------------------
# Global policy-grid feasibility
# ----------------------------------------------------------
policy_summary_rows = []

if not grid.empty:
    grid = grid.copy()

    for c in ["VAL_publication_blocker_n", "VAL_fatal_n", "VAL_warning_n", "VAL_pass_n"]:
        if c in grid.columns:
            grid[c] = pd.to_numeric(grid[c], errors="coerce")

    zero_global = grid[
        grid.get("VAL_publication_blocker_n", pd.Series(999, index=grid.index)).fillna(999).eq(0)
        & grid.get("VAL_fatal_n", pd.Series(999, index=grid.index)).fillna(999).eq(0)
    ].copy()

    top_sort_cols = [
        c for c in [
            "VAL_publication_blocker_n",
            "VAL_fatal_n",
            "VAL_warning_n",
            "VAL_response_window_rate_error_mean",
            "VAL_lag_peak_error_mean",
            "VAL_ETA_similarity_mean",
        ] if c in grid.columns
    ]

    top_20 = grid.sort_values(
        top_sort_cols,
        ascending=[True, True, True, True, True, False][:len(top_sort_cols)],
    ).head(20).copy() if top_sort_cols else grid.head(20).copy()

    policy_summary_rows.append({
        "selected_policy_id": selected_policy_id,
        "grid_rows": int(len(grid)),
        "zero_global_VAL_blocker_policy_n": int(len(zero_global)),
        "best_VAL_publication_blocker_n": int(pd.to_numeric(grid["VAL_publication_blocker_n"], errors="coerce").min()) if "VAL_publication_blocker_n" in grid.columns else None,
        "best_VAL_fatal_n": int(pd.to_numeric(grid["VAL_fatal_n"], errors="coerce").min()) if "VAL_fatal_n" in grid.columns else None,
        "interpretation": (
            "A zero-blocker all-8 A0 policy exists in VAL grid."
            if len(zero_global)
            else "No zero-blocker all-8 A0 policy exists in current VAL grid; all-8 positive claim needs a new predeclared materializer family or must remain blocked."
        ),
    })
else:
    top_20 = pd.DataFrame()
    zero_global = pd.DataFrame()
    policy_summary_rows.append({
        "selected_policy_id": selected_policy_id,
        "grid_rows": 0,
        "zero_global_VAL_blocker_policy_n": 0,
        "best_VAL_publication_blocker_n": None,
        "best_VAL_fatal_n": None,
        "interpretation": "No VAL policy grid found; cannot legally diagnose final policy feasibility.",
    })

policy_summary = pd.DataFrame(policy_summary_rows)

# ----------------------------------------------------------
# Save outputs
# ----------------------------------------------------------
root_out = REPORT_DIR_D0 / "cell14_6D0_q4_a0_pair_root_cause.csv"
policy_out = REPORT_DIR_D0 / "cell14_6D0_q4_a0_policy_feasibility.csv"
top20_out = REPORT_DIR_D0 / "cell14_6D0_q4_a0_top20_val_policies.csv"
zero_out = REPORT_DIR_D0 / "cell14_6D0_q4_a0_zero_val_policies.csv"
contract_out = CONTRACT_DIR_D0 / "cell14_6D0_q4_a0_diagnostic_contract_v1_0_THESIS.json"

root.to_csv(root_out, index=False)
policy_summary.to_csv(policy_out, index=False)
top_20.to_csv(top20_out, index=False)
zero_global.to_csv(zero_out, index=False)

contract = {
    "cell": "14.6D0",
    "version": CELL146D0_VERSION,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "role": "diagnostic_only_q4_a0_root_cause_analysis",
    "strict_contract": {
        "diagnostic_only": True,
        "synthetic_values_mutated": False,
        "materialization_done_here": False,
        "promotion_done_here": False,
        "policy_selected_here": False,
        "TEST_outcome_used_for_final_policy": False,
        "TEST_outcome_used_for_diagnostic_only": True,
        "VAL_grid_used_for_feasibility_audit": bool(not grid.empty),
        "pair_pruning_done_here": False,
        "thresholds_changed_here": False,
    },
    "inputs": {
        "a0_metrics": str(a0_metrics_path),
        "a0_profile_audit": str(a0_profile_path),
        "a0_materialization_audit": str(a0_mat_path),
        "a0_drift_audit": str(a0_drift_path),
        "a0_prepost_delta": str(a0_prepost_path),
        "trainval_policy_grid": str(r0_policy_grid_path),
        "trainval_pair_scores": str(r0_pair_scores_path),
    },
    "outputs": {
        "pair_root_cause_csv": str(root_out),
        "policy_feasibility_csv": str(policy_out),
        "top20_val_policies_csv": str(top20_out),
        "zero_val_policies_csv": str(zero_out),
        "contract_json": str(contract_out),
    },
    "summary": {
        "pairs_total": int(len(root)),
        "TEST_status_counts": root["TEST_status_norm"].value_counts(dropna=False).to_dict(),
        "TEST_blocker_n": int(root["TEST_publication_blocker_norm"].sum()),
        "root_cause_counts": root["diagnostic_root_cause"].value_counts(dropna=False).to_dict(),
        "selected_policy_id": selected_policy_id,
        "zero_global_VAL_blocker_policy_n": int(policy_summary["zero_global_VAL_blocker_policy_n"].iloc[0]),
        "best_VAL_publication_blocker_n": policy_summary["best_VAL_publication_blocker_n"].iloc[0],
        "best_VAL_fatal_n": policy_summary["best_VAL_fatal_n"].iloc[0],
    },
}
_write_json_146d0(contract_out, contract)

globals()["CELL14_6D0_Q4_A0_ROOT_CAUSE_DF"] = root
globals()["CELL14_6D0_Q4_A0_POLICY_FEASIBILITY_DF"] = policy_summary
globals()["CELL14_6D0_Q4_A0_TOP20_VAL_POLICIES_DF"] = top_20
globals()["CELL14_6D0_Q4_A0_ZERO_VAL_POLICIES_DF"] = zero_global
globals()["CELL14_6D0_Q4_A0_DIAGNOSTIC_CONTRACT"] = contract

print("\n=== CELL 14.6D0 POLICY FEASIBILITY ===")
display(policy_summary)

print("\n=== CELL 14.6D0 TEST ROOT-CAUSE SUMMARY ===")
display(
    root[
        [
            c for c in [
                "repair_candidate_id",
                "anchor_col",
                "protocol_col",
                "TEST_status_norm",
                "TEST_reason_norm",
                "TEST_publication_blocker_norm",
                "ETA_similarity_num",
                "ETA_corr_num",
                "ETA_norm_mae_num",
                "manifest_similarity_num",
                "manifest_corr_num",
                "manifest_norm_mae_num",
                "lag_peak_error_num",
                "response_window_rate_error_num",
                "response_rate_delta_syn_minus_real",
                "wide_eta_only_blocker_pattern",
                "diagnostic_root_cause",
                "VAL_selected_status",
                "VAL_selected_reasons",
                "VAL_best_pair_status",
                "VAL_best_pair_policy_id",
                "VAL_nonfatal_policy_n",
                "recommended_next_diagnostic_step",
            ] if c in root.columns
        ]
    ].sort_values(
        ["TEST_publication_blocker_norm", "TEST_status_norm", "ETA_similarity_num"],
        ascending=[False, True, True],
    )
)

print("\n=== CELL 14.6D0 TOP 20 VAL POLICIES ===")
display(top_20)

print(f"\n[Cell14.6D0] Saved pair root cause: {root_out}")
print(f"[Cell14.6D0] Saved policy feasibility: {policy_out}")
print(f"[Cell14.6D0] Saved top-20 VAL policies: {top20_out}")
print(f"[Cell14.6D0] Saved zero-VAL policies: {zero_out}")
print(f"[Cell14.6D0] Saved contract: {contract_out}")

try:
    log("--- END: Cell 14.6D0 - Q4 A0 Zigbee blocker root-cause diagnostic ---")
except Exception:
    print("--- END: Cell 14.6D0 - Q4 A0 Zigbee blocker root-cause diagnostic ---")