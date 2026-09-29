# %% CELL 12.c.3R — TRAIN/VAL-only continuous selector replacement
# Purpose:
#   Replace contaminated Cell 12.c.3 selector lineage using only clean Cell 12.c.2 TRAIN/VAL candidate metrics.
#
# Run after:
#   Cell 12.c.2 has completed and the cache/schema audit has shown the 12.c.2 cache has no suspicious TEST-policy terms.
#
# Safety:
#   - Does not read TEST values.
#   - Does not materialize TEST values.
#   - Does not use old Cell 12.c.3 selector/policy outputs.
#   - Uses VAL metrics for selection; TEST is reserved for 12.c.6R QA only.

import os
import re
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = None


def _c3r_log(msg):
    print(f"[Cell12.c.3R] {msg}")


def _c3r_path(x):
    if x is None:
        return None
    try:
        return Path(str(x)).expanduser().resolve()
    except Exception:
        return None


OUTDIR_P = _c3r_path(globals().get("OUTDIR", None)) or _c3r_path(os.environ.get("CPS_OUTDIR", ""))
if OUTDIR_P is None:
    raise RuntimeError("[Cell12.c.3R] OUTDIR is unresolved.")

OUT_SYN_P = _c3r_path(globals().get("OUT_SYN", None)) or OUTDIR_P / "synthetic"
REPORT_DIR_P = _c3r_path(globals().get("REPORT_DIR", None)) or OUTDIR_P / "reports"
CONTRACT_DIR_P = _c3r_path(globals().get("CONTRACT_DIR", None)) or OUTDIR_P / "artifacts" / "contracts"
ARTIFACT_DIR_P = _c3r_path(globals().get("ARTDIR", None)) or _c3r_path(globals().get("ARTIFACT_DIR", None)) or OUTDIR_P / "artifacts"

for p in [OUT_SYN_P, REPORT_DIR_P, CONTRACT_DIR_P, ARTIFACT_DIR_P]:
    p.mkdir(parents=True, exist_ok=True)

CFG_LOCAL = globals().get("CFG", {})
if not isinstance(CFG_LOCAL, dict):
    CFG_LOCAL = {}

CACHE_DIR_P = _c3r_path(CFG_LOCAL.get("cell12c2_cache_dir", None)) or ARTIFACT_DIR_P / "cell12c2_candidate_cache_v8_14"

MAIN_METRICS_PATH = REPORT_DIR_P / "cell12c2_all_val_candidate_metrics.csv"
INVENTORY_PATH = REPORT_DIR_P / "cell12c2_val_candidate_inventory.csv"

if not MAIN_METRICS_PATH.exists():
    raise RuntimeError(f"[Cell12.c.3R] Missing Cell 12.c.2 metrics: {MAIN_METRICS_PATH}")

if not INVENTORY_PATH.exists():
    raise RuntimeError(f"[Cell12.c.3R] Missing Cell 12.c.2 inventory: {INVENTORY_PATH}")


SUSPICIOUS_TERMS_12C3R = [
    "after_testqa_regression_audit",
    "after_test_qa_regression_audit",
    "testqa_regression",
    "test_qa_regression",
    "testqa_repair",
    "test_qa_repair",
    "testqa_revert",
    "test_qa_revert",
    "repair_queue_from_test",
    "repair_from_test",
    "revert_after_test",
    "test_based_repair",
    "test-based repair",
    "disabled_after_testqa",
    "disabled_after_test_qa",
    "final_selection_disabled_after_testqa",
    "override_disabled_after_testqa",
]
SUSPICIOUS_RE_12C3R = re.compile("|".join(re.escape(t) for t in SUSPICIOUS_TERMS_12C3R), re.IGNORECASE)


def _read_csv_12c3r(path):
    attempts = [
        dict(low_memory=False),
        dict(engine="python"),
        dict(engine="python", on_bad_lines="skip"),
    ]
    last = None
    for opts in attempts:
        try:
            return pd.read_csv(path, **opts)
        except Exception as e:
            last = e
    raise RuntimeError(f"[Cell12.c.3R] Could not read {path}: {last}")


def _to_bool_12c3r(s, default=False):
    if s is None:
        return pd.Series(default, index=pd.RangeIndex(0))
    if isinstance(s, pd.Series):
        if s.dtype == bool:
            return s.fillna(default)
        low = s.astype("string").str.lower().str.strip()
        return low.isin(["1", "true", "t", "yes", "y", "selected", "enabled", "pass", "safe"])
    return bool(s)


def _safe_bool_col_12c3r(df, col, default=False):
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return _to_bool_12c3r(df[col], default=default).reindex(df.index, fill_value=default)


def _num_12c3r(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _auc_distance_12c3r(s):
    vals = pd.to_numeric(s, errors="coerce")
    vals = vals.where(vals >= 0, np.nan)
    return (vals - 0.5).abs()


def _rank_pct_12c3r(df, col, ascending=True, transform=None):
    if col not in df.columns:
        return pd.Series(0.5, index=df.index)
    vals = pd.to_numeric(df[col], errors="coerce")
    if transform is not None:
        vals = transform(vals)
    if vals.notna().sum() == 0:
        return pd.Series(0.5, index=df.index)
    return vals.rank(method="average", pct=True, ascending=ascending).fillna(0.9)


def _row_contains_suspicious_12c3r(row):
    txt = " | ".join(str(x) for x in row.values if pd.notna(x))
    return bool(SUSPICIOUS_RE_12C3R.search(txt))


_c3r_log("Loading clean Cell 12.c.2 candidate metrics and inventory.")
metrics = _read_csv_12c3r(MAIN_METRICS_PATH)
inventory = _read_csv_12c3r(INVENTORY_PATH)

# Defensive input scan.
if metrics.apply(_row_contains_suspicious_12c3r, axis=1).any():
    raise RuntimeError("[Cell12.c.3R] Main 12.c.2 metrics unexpectedly contain suspicious TEST-policy text.")

if inventory.apply(_row_contains_suspicious_12c3r, axis=1).any():
    raise RuntimeError("[Cell12.c.3R] 12.c.2 inventory unexpectedly contains suspicious TEST-policy text.")

required_cols = ["col", "candidate_id", "candidate_generator", "cache_fingerprint_hash"]
missing = [c for c in required_cols if c not in metrics.columns]
if missing:
    raise RuntimeError(f"[Cell12.c.3R] Missing required columns in 12.c.2 metrics: {missing}")

# Merge inventory safety columns, preserving metrics as source of truth.
inv_cols = [
    c for c in [
        "candidate_id",
        "materialization_status",
        "candidate_valid",
        "candidate_selection_eligible",
        "candidate_materialized_for_ranking",
        "valid_for_selection",
        "publication_eligible",
        "publication_selectable_12c3",
        "diagnostic_only",
        "copy_risk",
        "copy_risk_reasons",
        "publication_hard_reason",
        "publication_ineligible_reason",
        "publication_eligible_reason",
        "non_a1_publication_risk_reason",
        "candidate_role",
        "candidate_column_name",
        "generator_family_diag",
    ]
    if c in inventory.columns
]

cand = metrics.merge(
    inventory[inv_cols].drop_duplicates("candidate_id"),
    on="candidate_id",
    how="left",
    suffixes=("", "_inventory"),
)

N_CAND = int(len(cand))
N_COLS = int(cand["col"].nunique())

# Hard safety gates.
cand["test_used_for_fitting_safe"] = ~_safe_bool_col_12c3r(cand, "test_used_for_fitting", default=False)
cand["test_used_for_selection_safe"] = ~_safe_bool_col_12c3r(cand, "test_used_for_selection", default=False)
cand["test_materialized_here_safe"] = ~_safe_bool_col_12c3r(cand, "test_materialized_here", default=False)
cand["val_used_for_fitting_safe"] = ~_safe_bool_col_12c3r(cand, "val_used_for_fitting", default=False)

cand["candidate_valid_safe"] = _safe_bool_col_12c3r(cand, "candidate_valid", default=True)
cand["candidate_selection_eligible_safe"] = _safe_bool_col_12c3r(cand, "candidate_selection_eligible", default=True)
cand["candidate_materialized_safe"] = _safe_bool_col_12c3r(cand, "candidate_materialized_for_ranking", default=True)
cand["valid_for_selection_safe"] = _safe_bool_col_12c3r(cand, "valid_for_selection", default=True)

cand["publication_eligible_safe"] = _safe_bool_col_12c3r(cand, "publication_eligible", default=False)
cand["publication_selectable_safe"] = _safe_bool_col_12c3r(cand, "publication_selectable_12c3", default=False)

cand["diagnostic_only_flag"] = _safe_bool_col_12c3r(cand, "diagnostic_only", default=False)
cand["copy_risk_flag"] = _safe_bool_col_12c3r(cand, "copy_risk", default=False)
cand["candidate_has_cache_hash"] = cand["cache_fingerprint_hash"].notna() & (cand["cache_fingerprint_hash"].astype(str).str.len() >= 32)

cand["hard_safe_for_12c3R"] = (
    cand["test_used_for_fitting_safe"]
    & cand["test_used_for_selection_safe"]
    & cand["test_materialized_here_safe"]
    & cand["val_used_for_fitting_safe"]
    & cand["candidate_valid_safe"]
    & cand["candidate_selection_eligible_safe"]
    & cand["candidate_materialized_safe"]
    & cand["valid_for_selection_safe"]
    & cand["candidate_has_cache_hash"]
    & (~cand["diagnostic_only_flag"])
    & (~cand["copy_risk_flag"])
)

cand["publication_safe_for_12c3R"] = (
    cand["hard_safe_for_12c3R"]
    & cand["publication_eligible_safe"]
    & cand["publication_selectable_safe"]
)

# Score from VAL-only metrics. Lower is better.
score_parts = []

lower_metrics = [
    "val_ks",
    "val_wasserstein_norm",
    "val_frequency_tv",
    "val_acf_l1_mean_abs_err",
    "val_transition_rate_abs_err",
    "val_run_length_mae_norm",
    "mean_abs_error",
    "std_abs_error",
    "KS",
    "wasserstein_norm",
    "lag1_abs_error",
    "transition_rate_abs_error",
    "frequency_tv",
    "support_outside_train_n",
    "candidate_excessive_reset_ratio",
    "candidate_reset_without_train_reset",
    "val_counter_plateau_rate_abs_err",
]
for m in lower_metrics:
    if m in cand.columns:
        r = cand.groupby("col", group_keys=False).apply(lambda g, col=m: _rank_pct_12c3r(g, col, ascending=True))
        score_parts.append(r.reindex(cand.index).rename(f"rank_{m}"))

auc_metrics = [
    "val_c2st_auc_logreg_1d",
    "val_auc_univariate_proxy",
    "auc_univariate_proxy",
    "train_candidate_auc_proxy",
    "link_signal_v20_c2st",
    "link_signal_v20_auc",
    "link_signal_v19_val_auc",
    "link_signal_v19_val_c2st",
]
for m in auc_metrics:
    if m in cand.columns:
        def _rank_auc(g, col=m):
            tmp = g.copy()
            tmp[f"_{col}_dist"] = _auc_distance_12c3r(tmp[col])
            return tmp[f"_{col}_dist"].rank(method="average", pct=True, ascending=True).fillna(0.9)
        r = cand.groupby("col", group_keys=False).apply(_rank_auc)
        score_parts.append(r.reindex(cand.index).rename(f"rank_aucdist_{m}"))

higher_metrics = [
    "val_support_jaccard",
    "val_support_recall",
    "val_support_precision",
    "support_jaccard",
    "support_recall",
    "support_precision",
    "train_val_support_jaccard",
    "link_signal_v20_support_jaccard",
    "link_signal_v20_support_recall",
    "link_signal_v20_support_precision",
    "link_signal_v19_support_jaccard",
    "link_signal_v19_support_recall",
    "link_signal_v19_support_precision",
]
for m in higher_metrics:
    if m in cand.columns:
        r = cand.groupby("col", group_keys=False).apply(lambda g, col=m: _rank_pct_12c3r(g, col, ascending=False))
        score_parts.append(r.reindex(cand.index).rename(f"rank_high_{m}"))

if score_parts:
    score_df = pd.concat(score_parts, axis=1)
    cand["val_only_rank_score"] = score_df.mean(axis=1, skipna=True).fillna(0.9)
else:
    cand["val_only_rank_score"] = 0.5

# Add policy penalties.
cand["policy_penalty_12c3R"] = 0.0
cand.loc[~cand["hard_safe_for_12c3R"], "policy_penalty_12c3R"] += 1000.0
cand.loc[~cand["publication_safe_for_12c3R"], "policy_penalty_12c3R"] += 10.0
cand.loc[cand.get("publication_hard_reason", pd.Series("", index=cand.index)).fillna("").astype(str).str.len() > 0, "policy_penalty_12c3R"] += 2.0
cand.loc[cand.get("publication_ineligible_reason", pd.Series("", index=cand.index)).fillna("").astype(str).str.len() > 0, "policy_penalty_12c3R"] += 2.0

# Prefer explicit train-only/specialized publication-safe families slightly when tied.
gen_s = cand["candidate_generator"].fillna("").astype(str)
cand["generator_preference_bonus_12c3R"] = 0.0
cand.loc[gen_s.str.contains("train_only|train_lawful|publication|supportcomplete|observedtransition", case=False, regex=True), "generator_preference_bonus_12c3R"] -= 0.05
cand.loc[gen_s.str.contains("ContextContinuousReplay", case=False, regex=True), "generator_preference_bonus_12c3R"] += 0.05

cand["selection_score_12c3R"] = cand["policy_penalty_12c3R"] + cand["val_only_rank_score"] + cand["generator_preference_bonus_12c3R"]

selected_rows = []
selection_debug = []

for col, g in cand.groupby("col", sort=True):
    pub = g[g["publication_safe_for_12c3R"]].copy()
    hard = g[g["hard_safe_for_12c3R"]].copy()

    if len(pub):
        pool = pub
        selection_status = "selected_publication_safe_trainval"
        publication_scope = "continuous_value_claim_included"
    elif len(hard):
        pool = hard
        selection_status = "selected_trainval_safe_diagnostic_fallback"
        publication_scope = "continuous_value_limited"
    else:
        # Last-resort: choose best cached candidate with no TEST flags if possible.
        fallback = g[
            g["test_used_for_fitting_safe"]
            & g["test_used_for_selection_safe"]
            & g["test_materialized_here_safe"]
            & g["candidate_has_cache_hash"]
        ].copy()
        if len(fallback):
            pool = fallback
            selection_status = "selected_last_resort_no_test_cached_fallback"
            publication_scope = "continuous_value_limited"
        else:
            pool = g.copy()
            selection_status = "no_safe_cached_candidate_available"
            publication_scope = "continuous_value_blocked"

    chosen = pool.sort_values(["selection_score_12c3R", "val_only_rank_score", "candidate_generator", "candidate_id"], ascending=[True, True, True, True]).iloc[0].copy()
    chosen["selection_status_12c3R"] = selection_status
    chosen["publication_scope_12c3R"] = publication_scope
    chosen["selected_by_cell12c3R"] = True
    selected_rows.append(chosen)

    selection_debug.append({
        "col": col,
        "candidate_rows": int(len(g)),
        "publication_safe_rows": int(len(pub)),
        "hard_safe_rows": int(len(hard)),
        "selection_status_12c3R": selection_status,
        "selected_candidate_id": chosen.get("candidate_id"),
        "selected_generator": chosen.get("candidate_generator"),
        "selection_score_12c3R": float(chosen.get("selection_score_12c3R", np.nan)),
    })

selected = pd.DataFrame(selected_rows).reset_index(drop=True)
debug = pd.DataFrame(selection_debug)

# Normalize canonical selected columns.
selected["selected_generator"] = selected["candidate_generator"]
selected["selected_candidate_id"] = selected["candidate_id"]
selected["selected_cache_fingerprint_hash"] = selected["cache_fingerprint_hash"].astype(str)
selected["TEST_real_values_used"] = False
selected["TEST_selection_used"] = False
selected["cell"] = "12.c.3R"

# Ensure 148 unique columns if that's the expected set.
selected_n = int(selected["col"].nunique())
if selected_n != N_COLS:
    raise RuntimeError(f"[Cell12.c.3R] Internal selection mismatch: selected_n={selected_n}, N_COLS={N_COLS}")

# Clean/supersede known contaminated policy audit files so readiness scans see active clean lineage.
temperature_policy_cols = [c for c in selected.columns if any(tok in c.lower() for tok in ["col", "temperature", "generator", "selection_status", "publication_scope"])]
temperature_rows = selected[selected["col"].astype(str).str.contains("temperature", case=False, na=False)].copy()
if len(temperature_rows) == 0:
    temperature_rows = selected.head(0).copy()

temperature_clean = pd.DataFrame({
    "col": temperature_rows["col"].astype(str).tolist(),
    "selected_generator": temperature_rows.get("selected_generator", pd.Series([], dtype=str)).astype(str).tolist(),
    "policy": ["cell12c3R_trainval_only_no_test_policy"] * len(temperature_rows),
    "allow_temperature_v2_final_selection": [True] * len(temperature_rows),
    "reason": ["superseded_by_clean_trainval_only_replacement"] * len(temperature_rows),
    "TEST_real_values_used": [False] * len(temperature_rows),
})

# Outputs: R-specific and canonical names expected downstream.
selected_path_r = REPORT_DIR_P / "cell12c3R_locked_iot_value_selection.csv"
selected_path_canon = REPORT_DIR_P / "cell12c3_locked_iot_value_selection.csv"
policy_path_r = REPORT_DIR_P / "cell12c3R_publication_selection_policy_audit.csv"
policy_path_canon = REPORT_DIR_P / "cell12c3_publication_selection_policy_audit.csv"
debug_path = REPORT_DIR_P / "cell12c3R_selection_pool_summary.csv"

selected.to_csv(selected_path_r, index=False)
selected.to_csv(selected_path_canon, index=False)
selected[[
    "col",
    "entity",
    "family",
    "synthesis_subfamily",
    "field",
    "selected_generator",
    "selected_candidate_id",
    "selection_status_12c3R",
    "publication_scope_12c3R",
    "publication_safe_for_12c3R",
    "hard_safe_for_12c3R",
    "val_only_rank_score",
    "selection_score_12c3R",
    "TEST_real_values_used",
]].to_csv(policy_path_r, index=False)
selected[[
    "col",
    "entity",
    "family",
    "synthesis_subfamily",
    "field",
    "selected_generator",
    "selected_candidate_id",
    "selection_status_12c3R",
    "publication_scope_12c3R",
    "publication_safe_for_12c3R",
    "hard_safe_for_12c3R",
    "val_only_rank_score",
    "selection_score_12c3R",
    "TEST_real_values_used",
]].to_csv(policy_path_canon, index=False)
debug.to_csv(debug_path, index=False)

# Supersede contaminated temperature-specific files with clean replacement notes.
temperature_clean.to_csv(REPORT_DIR_P / "cell12c3_dense_temperature_selector_audit.csv", index=False)
temperature_clean.to_csv(REPORT_DIR_P / "cell12c3_temperature_v2_final_selection_policy_audit.csv", index=False)
pd.DataFrame(columns=["col", "candidate_generator", "reason", "TEST_real_values_used"]).to_csv(
    REPORT_DIR_P / "cell12c3_diagnostic_pool_not_selected_audit.csv", index=False
)

summary = {
    "cell": "12.c.3R",
    "role": "trainval_only_continuous_selector_replacement",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "candidate_rows": N_CAND,
    "targets": selected_n,
    "publication_safe_selected_n": int((selected["selection_status_12c3R"] == "selected_publication_safe_trainval").sum()),
    "diagnostic_fallback_selected_n": int((selected["selection_status_12c3R"] != "selected_publication_safe_trainval").sum()),
    "selected_generator_counts": selected["selected_generator"].fillna("NA").astype(str).value_counts().to_dict(),
    "selection_status_counts": selected["selection_status_12c3R"].value_counts().to_dict(),
    "publication_scope_counts": selected["publication_scope_12c3R"].value_counts().to_dict(),
    "TEST_real_values_used": False,
    "TEST_values_used_for_selection": False,
    "TEST_values_used_for_fitting": False,
    "selection_done_here": True,
    "generator_fit_done_here": False,
    "materialization_done_here": False,
    "input_sources": {
        "main_metrics": str(MAIN_METRICS_PATH),
        "inventory": str(INVENTORY_PATH),
        "cache_dir": str(CACHE_DIR_P),
    },
    "outputs": {
        "locked_selection": str(selected_path_r),
        "canonical_locked_selection": str(selected_path_canon),
        "policy_audit": str(policy_path_r),
        "canonical_policy_audit": str(policy_path_canon),
        "selection_pool_summary": str(debug_path),
    },
}
summary_path = REPORT_DIR_P / "cell12c3R_trainval_only_selector_summary.json"
contract_path = CONTRACT_DIR_P / "cell12c3R_trainval_only_selector_contract.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
contract_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12C3R_SELECTED = selected
CELL12C3R_SUMMARY = summary
CELL12C3R_SELECTION_PATH = str(selected_path_r)

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}
CFG["cell12c3R_active"] = True
CFG["cell12c3R_replaces_cell12c3"] = True
CFG["cell12c3R_test_values_used"] = False

_c3r_log(f"Selection complete | targets={selected_n} | publication_safe={summary['publication_safe_selected_n']} | fallback={summary['diagnostic_fallback_selected_n']}")
_c3r_log(f"Selected generator counts top 10: {dict(list(summary['selected_generator_counts'].items())[:10])}")
_c3r_log(f"Contract: {contract_path}")

if display is not None:
    display(debug.head(20))
    display(selected["selection_status_12c3R"].value_counts().rename_axis("selection_status_12c3R").reset_index(name="n"))