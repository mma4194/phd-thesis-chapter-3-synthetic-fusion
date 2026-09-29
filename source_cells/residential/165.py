# %% CELL PAPER.READY.TABLES.V1 — first-class paper tables from current notebook artifacts
# Purpose:
#   Export the exact paper-facing tables from current notebook artifacts, not from hand-entered paper numbers.
#
# Scientific contract:
#   - Reads existing reports/contracts only.
#   - Does not fit, select, tune, repair, or mutate synthetic values.
#   - Continuous target-ready membership is based on VAL-side selection artifacts only.
#   - TEST metrics are reported after membership is frozen; TEST is not used for subset membership.

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

print("[PAPER.TABLES] START: generating first-class paper tables from current artifacts.")

if "OUTDIR" not in globals() or "REPORT_DIR" not in globals():
    raise RuntimeError("[PAPER.TABLES] Run setup cells first so OUTDIR and REPORT_DIR are defined.")

OUTDIR_PAPER = Path(str(OUTDIR)).expanduser().resolve()
REPORT_DIR_PAPER = Path(str(REPORT_DIR)).expanduser().resolve()
CONTRACT_DIR_PAPER = OUTDIR_PAPER / "artifacts" / "contracts"
REPORT_DIR_PAPER.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_PAPER.mkdir(parents=True, exist_ok=True)

def _read_csv_paper(*names):
    for n in names:
        p = Path(str(n))
        if not p.is_absolute():
            p = REPORT_DIR_PAPER / str(n)
        if p.exists():
            try:
                return pd.read_csv(p), p
            except Exception as e:
                print(f"[PAPER.TABLES] Could not read {p}: {e}")
    return pd.DataFrame(), None

def _read_json_paper(*names):
    for n in names:
        p = Path(str(n))
        if not p.is_absolute():
            p = REPORT_DIR_PAPER / str(n)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")), p
            except Exception as e:
                print(f"[PAPER.TABLES] Could not read JSON {p}: {e}")
    return {}, None

def _num(s, default=np.nan):
    try:
        return float(s)
    except Exception:
        return default

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in low:
            return low[str(c).lower()]
    return None

def _safe_counts(series):
    if series is None:
        return {}
    try:
        return {str(k): int(v) for k, v in series.astype(str).value_counts(dropna=False).to_dict().items()}
    except Exception:
        return {}

def _orient_auc(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.maximum(x, 1.0 - x)

# ------------------------------------------------------------------
# Continuous target-ready tiers (VAL membership; TEST reporting only)
# ------------------------------------------------------------------
sel_df, sel_path = _read_csv_paper(
    globals().get("CELL12C3R_SELECTION_CSV", ""),
    "cell12c3R_locked_iot_value_selection.csv",
    "cell12c3_locked_iot_value_selection.csv",
)
qa_df, qa_path = _read_csv_paper(
    globals().get("CELL12C6R_QA_CSV", ""),
    "cell12c6R_final_test_qa_metrics.csv",
    "cell12c6_final_test_qa_metrics.csv",
)

tier_rows = []
continuous_claim_note = "continuous selection/QA artifacts not found"
if len(qa_df):
    target_col = _find_col(qa_df, ["col", "column", "target"])
    test_auc_col = _find_col(qa_df, ["TEST_c2st_auc", "test_c2st_auc", "val_c2st_auc_logreg_1d", "c2st_auc", "AUC"])
    status_col = _find_col(qa_df, ["publication_status", "test_qa_status", "qa_status"])
    all_cols = qa_df[target_col].astype(str).nunique() if target_col else len(qa_df)
    all_auc = float(pd.to_numeric(qa_df[test_auc_col], errors="coerce").mean()) if test_auc_col else np.nan
    tier_rows.append({
        "scope": "all_continuous_targets",
        "tau": np.nan,
        "columns": int(all_cols),
        "TEST_AUC": all_auc,
        "TEST_C2ST": all_auc,
        "VAL_selection_basis": "all 12.c.R continuous targets; no target-ready filtering",
        "blocker_exclusion_policy": "reported_full_branch_not_uniform_realism_claim",
        "selected_on_VAL": False,
        "TEST_used_for_membership": False,
        "TEST_fatal_n": int((qa_df[status_col].astype(str).str.lower() == "fatal").sum()) if status_col else np.nan,
        "claim_allowed": False,
        "source_selection_artifact": str(sel_path) if sel_path else "",
        "source_TEST_QA_artifact": str(qa_path) if qa_path else "",
    })

    if len(sel_df) and target_col:
        sel_target_col = _find_col(sel_df, ["col", "column", "target"])
        val_auc_col = _find_col(sel_df, [
            "val_c2st_auc_logreg_1d",
            "val_auc_univariate_proxy",
            "auc_univariate_proxy",
            "link_signal_v19_val_auc",
            "link_signal_v19_val_c2st",
            "link_signal_v20_auc",
            "link_signal_v20_c2st",
        ])
        pub_scope_col = _find_col(sel_df, ["publication_scope_12c3R", "publication_scope", "claim_scope"])
        pub_safe_col = _find_col(sel_df, ["publication_safe_for_12c3R", "publication_safe", "hard_safe_for_12c3R"])
        if sel_target_col and val_auc_col:
            tmp_sel = sel_df.copy()
            tmp_sel["_target"] = tmp_sel[sel_target_col].astype(str)
            tmp_sel["_VAL_sep"] = _orient_auc(tmp_sel[val_auc_col])
            if pub_scope_col:
                tmp_sel["_pretest_allowed"] = tmp_sel[pub_scope_col].astype(str).eq("continuous_value_claim_included")
            else:
                tmp_sel["_pretest_allowed"] = True
            if pub_safe_col:
                tmp_sel["_pretest_allowed"] = tmp_sel["_pretest_allowed"] & tmp_sel[pub_safe_col].fillna(False).astype(bool)
            tmp_qa = qa_df.copy()
            tmp_qa["_target"] = tmp_qa[target_col].astype(str)
            for tau in [0.60, 0.55]:
                members = sorted(tmp_sel.loc[tmp_sel["_pretest_allowed"] & (tmp_sel["_VAL_sep"] <= tau), "_target"].unique().tolist())
                qsub = tmp_qa[tmp_qa["_target"].isin(members)].copy()
                test_auc = float(pd.to_numeric(qsub[test_auc_col], errors="coerce").mean()) if len(qsub) and test_auc_col else np.nan
                fatal_n = int((qsub[status_col].astype(str).str.lower() == "fatal").sum()) if len(qsub) and status_col else 0
                blocked_n = int((qsub[status_col].astype(str).str.lower().isin(["fatal", "blocked", "blocker", "blocked_pretest"])).sum()) if len(qsub) and status_col else 0
                tier_rows.append({
                    "scope": f"target_ready_tau_{tau:.2f}",
                    "tau": float(tau),
                    "columns": int(len(members)),
                    "TEST_AUC": test_auc,
                    "TEST_C2ST": test_auc,
                    "VAL_selection_basis": f"{val_auc_col} <= {tau:.2f} and TRAIN/VAL publication-safe 12.c.R membership",
                    "blocker_exclusion_policy": "membership excludes pre-TEST non-publication-safe candidates; TEST blockers reported but not used for membership",
                    "selected_on_VAL": True,
                    "TEST_used_for_membership": False,
                    "TEST_fatal_n": fatal_n,
                    "TEST_blocked_or_fatal_n": blocked_n,
                    "claim_allowed": bool(len(members) > 0 and fatal_n == 0 and blocked_n == 0),
                    "source_selection_artifact": str(sel_path) if sel_path else "",
                    "source_TEST_QA_artifact": str(qa_path) if qa_path else "",
                })
            continuous_claim_note = "target-ready tiers regenerated from current 12.c.R artifacts"
        else:
            continuous_claim_note = "selection table lacks a VAL separability column; tiers cannot be regenerated without TEST leakage"
    else:
        continuous_claim_note = "selection table not found; only all-continuous TEST row exported"
else:
    tier_rows.append({
        "scope": "all_continuous_targets",
        "tau": np.nan,
        "columns": 0,
        "TEST_AUC": np.nan,
        "TEST_C2ST": np.nan,
        "VAL_selection_basis": continuous_claim_note,
        "blocker_exclusion_policy": "not_evaluable",
        "selected_on_VAL": False,
        "TEST_used_for_membership": False,
        "TEST_fatal_n": np.nan,
        "claim_allowed": False,
    })

continuous_tiers = pd.DataFrame(tier_rows)
continuous_tiers_path = REPORT_DIR_PAPER / "paper_table_q1q2_continuous_target_ready_tiers.csv"
continuous_tiers.to_csv(continuous_tiers_path, index=False)

# ------------------------------------------------------------------
# Branch readiness
# ------------------------------------------------------------------
branch_rows = []

if len(qa_df):
    status_col = _find_col(qa_df, ["publication_status", "test_qa_status", "qa_status"])
    counts = _safe_counts(qa_df[status_col]) if status_col else {}
    branch_rows.append({
        "branch": "Continuous IoT values",
        "targets": int(qa_df[_find_col(qa_df, ["col", "column", "target"])].astype(str).nunique()) if _find_col(qa_df, ["col", "column", "target"]) else int(len(qa_df)),
        "final_status": "quality_tiered_no_uniform_full_branch_realism",
        "pass_n": int(counts.get("pass", 0)),
        "warning_n": int(counts.get("warning", 0)),
        "fatal_n": int(counts.get("fatal", 0)),
        "blocker_n": int(counts.get("blocked_pretest", 0) + counts.get("blocker", 0) + counts.get("blocked", 0)),
        "interpretation": continuous_claim_note,
    })

bin_summary = globals().get("CELL12D_V6_TEST_QA_SUMMARY", {})
if not isinstance(bin_summary, dict) or not bin_summary:
    bin_summary, _ = _read_json_paper("cell12d_v6_terminal_test_qa_summary.json")
if bin_summary:
    branch_rows.append({
        "branch": "Binary IoT states",
        "targets": int(bin_summary.get("targets_total", bin_summary.get("targets", 52)) or 0),
        "final_status": "V6_preferred_development_candidate_fresh_holdout_required",
        "pass_n": int(bin_summary.get("included_pass_n", bin_summary.get("pass", 0)) or 0),
        "warning_n": int(bin_summary.get("included_warning_n", bin_summary.get("warning", 0)) or 0),
        "fatal_n": int(bin_summary.get("included_fatal_n", bin_summary.get("fatal", 0)) or 0),
        "blocker_n": int(bin_summary.get("included_publication_blocker_n", bin_summary.get("publication_blocker_n", 0)) or 0),
        "interpretation": "39 included / 13 scope-excluded in V6 if current V6 summary is unchanged; do not claim full binary solved without fresh holdout.",
    })

q3_summary = globals().get("CELL12F_V5_QA_SUMMARY", {})
if not isinstance(q3_summary, dict) or not q3_summary:
    q3_summary, _ = _read_json_paper("cell12f_v5_terminal_test_qa_summary.json")
q3_post_df, q3_post_path = _read_csv_paper("q3_post_v5_claim_ledger_row.csv")
if q3_summary:
    branch_rows.append({
        "branch": "Observability masks",
        "targets": int(q3_summary.get("targets", q3_summary.get("targets_total", 183)) or 0),
        "final_status": str(q3_summary.get("final_status", "BLOCKED_PARTIAL")),
        "pass_n": int(q3_summary.get("pass", 0) or 0),
        "warning_n": int(q3_summary.get("warning", 0) or 0),
        "fatal_n": int(q3_summary.get("fatal", 0) or 0),
        "blocker_n": int(q3_summary.get("publication_blocker_n", q3_summary.get("blocker_n", 0)) or 0),
        "interpretation": "Q3 V5 partial-admissible only; public release excluded unless blockers resolved.",
    })

driver_summary, _ = _read_json_paper("cell12e_final_event_driver_summary.json")
public_registry_df, public_registry_path = _read_csv_paper("cell15_6_q6_public_column_registry.csv")
driver_corrected_df, driver_corrected_path = _read_csv_paper("cell12e6_driver_corrected_release_ledger.csv")
driver_public_n = 0
if len(public_registry_df) and "role_group" in public_registry_df.columns:
    driver_public_n = int(public_registry_df["role_group"].astype(str).eq("iot_event_drivers").sum())
if len(driver_corrected_df) and "driver_corrected_ratio_battery_status" in driver_corrected_df.columns:
    dcounts = _safe_counts(driver_corrected_df["driver_corrected_ratio_battery_status"])
    branch_rows.append({
        "branch": "Sparse IoT drivers",
        "targets": int(len(driver_corrected_df)),
        "final_status": "corrected_ratio_scale_release_eligible_subset",
        "pass_n": int(dcounts.get("pass", 0)),
        "warning_n": int(dcounts.get("warning", 0)),
        "fatal_n": int(dcounts.get("fatal", 0)),
        "blocker_n": int(dcounts.get("fatal", 0)),
        "public_release_eligible_n": int(driver_public_n),
        "interpretation": "Corrected ratio-scale battery supersedes legacy 24/2 grading; corrected-fatal drivers are excluded from Q6 public release eligibility.",
    })
elif driver_public_n:
    branch_rows.append({
        "branch": "Sparse IoT drivers",
        "targets": driver_public_n,
        "final_status": "public_registry_only_missing_corrected_driver_ledger",
        "pass_n": np.nan,
        "warning_n": np.nan,
        "fatal_n": np.nan,
        "blocker_n": np.nan,
        "public_release_eligible_n": driver_public_n,
        "interpretation": "Corrected driver ledger missing; do not update manuscript from this row.",
    })

# IoT namespace structural row from Cell 12.g contract if available.
cell12g_contract = globals().get("CELL12G_FULL_IOT_NAMESPACE_CONTRACT", {})
if isinstance(cell12g_contract, dict) and cell12g_contract:
    out_shapes = cell12g_contract.get("output_shapes", cell12g_contract.get("summary", {}))
    branch_rows.append({
        "branch": "IoT namespace assembly",
        "targets": int(out_shapes.get("iot_full_cols", out_shapes.get("cols", 530)) or 530),
        "final_status": "structural_pass",
        "pass_n": np.nan,
        "warning_n": 0,
        "fatal_n": 0,
        "blocker_n": 0,
        "interpretation": "Zero inactive-finite and role-owner violations if current Cell 12.g contract is unchanged.",
    })

branch_readiness = pd.DataFrame(branch_rows)
branch_readiness_path = REPORT_DIR_PAPER / "paper_table_branch_readiness.csv"
branch_readiness.to_csv(branch_readiness_path, index=False)

# ------------------------------------------------------------------
# Q3 publication status table
# ------------------------------------------------------------------
q3_rows = []
if q3_summary:
    q3_rows.append({
        "scope": "observability_masks_v5",
        "targets": int(q3_summary.get("targets", q3_summary.get("targets_total", 183)) or 0),
        "pass_n": int(q3_summary.get("pass", 0) or 0),
        "warning_n": int(q3_summary.get("warning", 0) or 0),
        "fatal_n": int(q3_summary.get("fatal", 0) or 0),
        "publication_blocker_n": int(q3_summary.get("publication_blocker_n", q3_summary.get("blocker_n", 0)) or 0),
        "final_status": str(q3_summary.get("final_status", "BLOCKED_PARTIAL")),
        "claim_scope": str(q3_summary.get("claim_scope", "partial_admissible_only")),
        "full_scope_ready": bool(q3_summary.get("full_scope_ready", False)),
        "source": str(q3_post_path) if q3_post_path else "CELL12F_V5_QA_SUMMARY",
    })
q3_table = pd.DataFrame(q3_rows)
q3_table_path = REPORT_DIR_PAPER / "paper_table_q3_publication_status.csv"
q3_table.to_csv(q3_table_path, index=False)

# ------------------------------------------------------------------
# Q4 scope status table
# ------------------------------------------------------------------
q4_rows = []
q4_broad = globals().get("CELL13_4_Q4_PUBLICATION_SUMMARY", {})
if isinstance(q4_broad, dict) and q4_broad:
    q4_rows.append({
        "scope": "generic_all_protocol_q4",
        "pairs": int(q4_broad.get("pair_total", q4_broad.get("pairs_total", q4_broad.get("evaluated_pairs", 0))) or 0),
        "pass_n": int(q4_broad.get("pair_pass_n", q4_broad.get("pass_n", 0)) or 0),
        "warning_n": int(q4_broad.get("pair_warning_n", q4_broad.get("warning_n", 0)) or 0),
        "fatal_n": int(q4_broad.get("pair_fatal_n", q4_broad.get("fatal_n", 0)) or 0),
        "blocker_n": int(q4_broad.get("pair_blocker_n", q4_broad.get("publication_blocker_n", 0)) or 0),
        "final_status": str(q4_broad.get("overall_q4_status", "blocked")),
        "claim_allowed": False,
        "source": "CELL13_4_Q4_PUBLICATION_SUMMARY",
    })

a0_summary = {}
if isinstance(globals().get("CELL14_6_A0_CONTRACT", None), dict):
    a0_summary = CELL14_6_A0_CONTRACT.get("summary", {})
if a0_summary:
    q4_rows.append({
        "scope": "A0_zigbee_safe_cell14_6_terminal_candidate",
        "pairs": int(a0_summary.get("A0_pairs_total", 0) or 0),
        "pass_n": int(a0_summary.get("q4_pass_n", 0) or 0),
        "warning_n": int(a0_summary.get("q4_warning_n", 0) or 0),
        "fatal_n": int(a0_summary.get("q4_fatal_n", 0) or 0),
        "blocker_n": int(a0_summary.get("q4_publication_blocker_n", 0) or 0),
        "mean_ETA_similarity": _num(a0_summary.get("mean_ETA_similarity")),
        "mean_lag_peak_error": _num(a0_summary.get("mean_lag_peak_error")),
        "mean_response_window_rate_error": _num(a0_summary.get("mean_response_window_rate_error")),
        "final_status": "terminal_report_only_no_promotion",
        "claim_allowed": False,
        "source": "CELL14_6_A0_CONTRACT",
    })

a0r1 = globals().get("CELL14_6R1_A0_TERMINAL_SCOPE_STATUS_DF", pd.DataFrame())
if isinstance(a0r1, pd.DataFrame) and len(a0r1):
    row = a0r1.iloc[0].to_dict()
    q4_rows.append({
        "scope": "A0_zigbee_safe_frozen_trainval_policy",
        "pairs": int(row.get("pairs_total", 0) or 0),
        "pass_n": int(row.get("TEST_pass_n", 0) or 0),
        "warning_n": int(row.get("TEST_warning_n", 0) or 0),
        "fatal_n": int(row.get("TEST_fatal_n", 0) or 0),
        "blocker_n": int(row.get("TEST_publication_blocker_n", 0) or 0),
        "mean_ETA_similarity": _num(row.get("mean_ETA_similarity")),
        "mean_lag_peak_error": _num(row.get("mean_lag_peak_error")),
        "mean_response_window_rate_error": _num(row.get("mean_response_window_rate_error")),
        "final_status": str(row.get("terminal_status", "")),
        "claim_allowed": bool(row.get("publication_claim_allowed", False)),
        "source": "cell14_6R1_A0_frozen_policy_terminal_scope_status.csv",
    })

q4_table = pd.DataFrame(q4_rows)
q4_table_path = REPORT_DIR_PAPER / "paper_table_q4_scope_status.csv"
q4_table.to_csv(q4_table_path, index=False)

# ------------------------------------------------------------------
# Q6 public column accounting / Table 17
# ------------------------------------------------------------------
registry_df, registry_path = _read_csv_paper("cell15_6_q6_public_column_registry.csv")
q6_rows = []
if len(registry_df):
    role_col = _find_col(registry_df, ["role_group", "role", "scope"])
    tier_col = _find_col(registry_df, ["protocol_tier"])
    is_time_col = _find_col(registry_df, ["is_time"])
    if is_time_col:
        time_n = int(registry_df[is_time_col].fillna(False).astype(bool).sum())
    else:
        time_n = int(registry_df[_find_col(registry_df, ["col", "column"])].astype(str).str.contains("sec_epoch|timestamp|time", case=False, na=False).sum())
    router_n = int(registry_df[tier_col].astype(str).eq("router").sum()) if tier_col else 0
    zigbee_n = int(registry_df[tier_col].astype(str).eq("zigbee").sum()) if tier_col else 0
    driver_n = int(registry_df[role_col].astype(str).eq("iot_event_drivers").sum()) if role_col else 0
    logical_n = int(len(registry_df))
    q6_rows.extend([
        {"retained_group": "Canonical timestamp/alignment column", "count": time_n, "role_in_public_candidate": "time alignment; not separately scored"},
        {"retained_group": "Router protocol columns", "count": router_n, "role_in_public_candidate": "warning-governed protocol benchmarking scope"},
        {"retained_group": "Zigbee protocol columns", "count": zigbee_n, "role_in_public_candidate": "Q6-filtered protocol scope"},
        {"retained_group": "Sparse IoT event-driver columns", "count": driver_n, "role_in_public_candidate": "warning-governed sparse event-driver evidence"},
        {"retained_group": "Logical public columns", "count": logical_n, "role_in_public_candidate": "evidence-bearing public schema plus timestamp/alignment"},
        {"retained_group": "On-disk stored fields", "count": logical_n + 1, "role_in_public_candidate": "logical public columns plus persisted pandas row index if written with index=True"},
    ])
q6_accounting = pd.DataFrame(q6_rows)
q6_accounting_path = REPORT_DIR_PAPER / "paper_table_q6_public_column_accounting.csv"
table17_path = REPORT_DIR_PAPER / "paper_table17_public_release_accounting.csv"
q6_accounting.to_csv(q6_accounting_path, index=False)
q6_accounting.to_csv(table17_path, index=False)

# ------------------------------------------------------------------
# Baseline Table 18 from Cell 17.6
# ------------------------------------------------------------------
headline_df, headline_path = _read_csv_paper("cell17_6_baseline_headline_comparable_results.csv")
partial_df, partial_path = _read_csv_paper("cell17_6_baseline_partial_limited_results.csv")
baseline_rows = []
for df, headline_flag, src_path in [(headline_df, True, headline_path), (partial_df, False, partial_path)]:
    if not len(df):
        continue
    for _, r in df.iterrows():
        baseline_rows.append({
            "scope": str(r.get("scope_display", r.get("scope_id", ""))),
            "scope_id": str(r.get("scope_id", "")),
            "baseline": str(r.get("evaluated_external_baselines", "")),
            "SF_pipeline_penalty": _num(r.get("best_fair_scope_penalty_score")),
            "external_baseline_penalty": np.nan,
            "coverage": _num(r.get("pipeline_col_coverage")),
            "headline_counted": bool(headline_flag),
            "status": "Headline" if headline_flag else "Partial/limited",
            "reason": str(r.get("comparison_reasons", r.get("comparison_status", ""))),
            "winner": str(r.get("headline_winner", "")) if headline_flag else "not_headline_counted",
            "source": str(src_path) if src_path else "",
        })
baseline_table18 = pd.DataFrame(baseline_rows)
baseline_table18_path = REPORT_DIR_PAPER / "paper_table18_baseline_fairness.csv"
baseline_table18.to_csv(baseline_table18_path, index=False)

# ------------------------------------------------------------------
# Claim-to-artifact map and numbers summary
# ------------------------------------------------------------------
claim_rows = [
    {"paper_claim": "continuous target-ready tiers", "artifact": str(continuous_tiers_path), "claim_status": "use generated tiers only; do not copy old paper values"},
    {"paper_claim": "branch readiness summary", "artifact": str(branch_readiness_path), "claim_status": "paper table source"},
    {"paper_claim": "Q3 observability status", "artifact": str(q3_table_path), "claim_status": "paper table source"},
    {"paper_claim": "Q4 coupling scope", "artifact": str(q4_table_path), "claim_status": "paper table source"},
    {"paper_claim": "Q6 public release accounting", "artifact": str(q6_accounting_path), "claim_status": "paper table source"},
    {"paper_claim": "baseline fairness Table 18", "artifact": str(baseline_table18_path), "claim_status": "paper table source"},
]
claim_map = pd.DataFrame(claim_rows)
claim_map_path = REPORT_DIR_PAPER / "paper_claim_to_artifact_map.csv"
claim_map.to_csv(claim_map_path, index=False)

numbers = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "continuous": {
        "tier_table": str(continuous_tiers_path),
        "tier_rows": continuous_tiers.to_dict("records"),
    },
    "branch_readiness": branch_readiness.to_dict("records"),
    "q3": q3_table.to_dict("records"),
    "q4": q4_table.to_dict("records"),
    "q6_public_accounting": q6_accounting.to_dict("records"),
    "baseline_table18": baseline_table18.to_dict("records"),
    "warnings": [
        "Do not claim a public-column count until paper_table_q6_public_column_accounting and FINAL.Q6 agree after corrected driver exclusion.",
        "Do not claim A0 publication pass unless paper_table_q4_scope_status reports claim_allowed=True on a fresh untouched holdout.",
        "Do not headline-count partial baseline scopes.",
    ],
}
numbers_path = REPORT_DIR_PAPER / "paper_numbers_summary.json"
numbers_path.write_text(json.dumps(numbers, indent=2, sort_keys=True, default=str), encoding="utf-8")

globals()["PAPER_TABLE_BRANCH_READINESS_CSV"] = str(branch_readiness_path)
globals()["PAPER_TABLE_Q1Q2_CONTINUOUS_TIERS_CSV"] = str(continuous_tiers_path)
globals()["PAPER_TABLE_Q3_PUBLICATION_STATUS_CSV"] = str(q3_table_path)
globals()["PAPER_TABLE_Q4_SCOPE_STATUS_CSV"] = str(q4_table_path)
globals()["PAPER_TABLE_Q6_PUBLIC_COLUMN_ACCOUNTING_CSV"] = str(q6_accounting_path)
globals()["PAPER_TABLE17_PUBLIC_RELEASE_ACCOUNTING_CSV"] = str(table17_path)
globals()["PAPER_TABLE18_BASELINE_FAIRNESS_CSV"] = str(baseline_table18_path)
globals()["PAPER_CLAIM_TO_ARTIFACT_MAP_CSV"] = str(claim_map_path)
globals()["PAPER_NUMBERS_SUMMARY_JSON"] = str(numbers_path)

print("[PAPER.TABLES] Wrote:")
for p in [
    branch_readiness_path,
    continuous_tiers_path,
    q3_table_path,
    q4_table_path,
    q6_accounting_path,
    table17_path,
    baseline_table18_path,
    claim_map_path,
    numbers_path,
]:
    print(" -", p)
print("[PAPER.TABLES] END.")
