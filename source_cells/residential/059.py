# %% CELL 12.c.6R — Continuous terminal TEST QA report-only replacement
# Purpose:
#   Terminal TEST QA for 12.c.3R–12.c.5R continuous replacement branch.
#
# Safety:
#   - Reads TEST real values for QA only.
#   - Does not mutate synthetic values.
#   - Does not change selection/materialization.
#   - Does not repair using TEST.
#   - Marks this branch as conservative/limited if failures remain.

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = None


def _c6r_log(msg):
    print(f"[Cell12.c.6R] {msg}")


def _c6r_path(x):
    if x is None:
        return None
    try:
        return Path(str(x)).expanduser().resolve()
    except Exception:
        return None


OUTDIR_P = _c6r_path(globals().get("OUTDIR", None)) or _c6r_path(os.environ.get("CPS_OUTDIR", ""))
OUT_SYN_P = _c6r_path(globals().get("OUT_SYN", None)) or OUTDIR_P / "synthetic"
REPORT_DIR_P = _c6r_path(globals().get("REPORT_DIR", None)) or OUTDIR_P / "reports"
CONTRACT_DIR_P = _c6r_path(globals().get("CONTRACT_DIR", None)) or OUTDIR_P / "artifacts" / "contracts"

for p in [OUT_SYN_P, REPORT_DIR_P, CONTRACT_DIR_P]:
    p.mkdir(parents=True, exist_ok=True)

final_path = OUT_SYN_P / "IOT_FINAL_CONTINUOUS_TEST.parquet"
if not final_path.exists():
    raise RuntimeError("[Cell12.c.6R] Missing final continuous values. Run Cell 12.c.5R first.")

selection_path = REPORT_DIR_P / "cell12c3R_locked_iot_value_selection.csv"
if not selection_path.exists():
    selection_path = REPORT_DIR_P / "cell12c3_locked_iot_value_selection.csv"
selection = pd.read_csv(selection_path)

if "df_te" in globals() and isinstance(globals()["df_te"], pd.DataFrame):
    DF_TE_LOCAL = globals()["df_te"]
elif "DF_TE" in globals() and isinstance(globals()["DF_TE"], pd.DataFrame):
    DF_TE_LOCAL = globals()["DF_TE"]
else:
    raise RuntimeError("[Cell12.c.6R] Need df_te/DF_TE for terminal QA.")

if "df_tr" in globals() and isinstance(globals()["df_tr"], pd.DataFrame):
    DF_TR_LOCAL = globals()["df_tr"]
elif "DF_TR" in globals() and isinstance(globals()["DF_TR"], pd.DataFrame):
    DF_TR_LOCAL = globals()["DF_TR"]
else:
    DF_TR_LOCAL = None

syn = pd.read_parquet(final_path)
sel_by_col = selection.set_index("col", drop=False)


def _ks_12c6r(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 1.0
    a = np.sort(a)
    b = np.sort(b)
    x = np.unique(np.r_[a, b])
    ca = np.searchsorted(a, x, side="right") / len(a)
    cb = np.searchsorted(b, x, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _wasserstein_12c6r(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0:
        return float(np.mean(np.abs(b)))
    if len(b) == 0:
        return float(np.mean(np.abs(a)))
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        qs = np.linspace(0, 1, 101)
        return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _scale_12c6r(col, real_vals):
    vals = None
    if DF_TR_LOCAL is not None and col in DF_TR_LOCAL.columns:
        vals = pd.to_numeric(DF_TR_LOCAL[col], errors="coerce").dropna().to_numpy(dtype="float64")
    if vals is None or len(vals) < 10:
        vals = np.asarray(real_vals, dtype="float64")
        vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    q05, q95 = np.quantile(vals, [0.05, 0.95])
    sc = float(q95 - q05)
    if not np.isfinite(sc) or sc <= 1e-12:
        sc = float(np.nanstd(vals))
    if not np.isfinite(sc) or sc <= 1e-12:
        sc = 1.0
    return sc


def _c2st_1d_auc_12c6r(a, b, max_n=4096):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n = min(len(a), len(b), max_n)
    if n < 32:
        return 0.5
    rng = np.random.default_rng(123)
    if len(a) > n:
        a = rng.choice(a, size=n, replace=False)
    if len(b) > n:
        b = rng.choice(b, size=n, replace=False)
    # Rank-based AUC equivalent without sklearn.
    x = np.r_[a, b]
    y = np.r_[np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)]
    ranks = pd.Series(x).rank(method="average").to_numpy()
    r_pos = ranks[y == 1].sum()
    n_pos = len(b)
    n_neg = len(a)
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / max(1.0, n_pos * n_neg)
    auc = float(auc)
    return max(auc, 1.0 - auc)


def _support_jaccard_12c6r(a, b, max_unique=100):
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[np.isfinite(a.astype("float64"))]
    b = b[np.isfinite(b.astype("float64"))]
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(np.unique(a)) > max_unique or len(np.unique(b)) > max_unique:
        return np.nan
    sa = set(np.unique(a).tolist())
    sb = set(np.unique(b).tolist())
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


rows = []

for i, col in enumerate(syn.columns):
    if (i + 1) == 1 or (i + 1) % 25 == 0 or (i + 1) == len(syn.columns):
        _c6r_log(f"QA {i+1}/{len(syn.columns)} | col={col}")

    real = pd.to_numeric(DF_TE_LOCAL[col], errors="coerce").to_numpy(dtype="float64") if col in DF_TE_LOCAL.columns else np.full(len(syn), np.nan)
    gen = pd.to_numeric(syn[col], errors="coerce").to_numpy(dtype="float64")

    real_f = real[np.isfinite(real)]
    gen_f = gen[np.isfinite(gen)]
    n = min(len(real), len(gen))
    scale = _scale_12c6r(col, real_f)

    mean_abs_error_norm = abs(np.nanmean(real_f) - np.nanmean(gen_f)) / scale if len(real_f) and len(gen_f) else np.nan
    std_abs_error_norm = abs(np.nanstd(real_f) - np.nanstd(gen_f)) / scale if len(real_f) and len(gen_f) else np.nan
    ks = _ks_12c6r(real_f, gen_f)
    wass = _wasserstein_12c6r(real_f, gen_f) / scale
    c2st_auc = _c2st_1d_auc_12c6r(real_f, gen_f)
    support_jaccard = _support_jaccard_12c6r(real_f, gen_f)

    real_finite_rate = float(np.isfinite(real).mean()) if len(real) else 0.0
    syn_finite_rate = float(np.isfinite(gen).mean()) if len(gen) else 0.0
    mask_rate_error = abs(real_finite_rate - syn_finite_rate)

    if col in sel_by_col.index:
        srow = sel_by_col.loc[col]
        selected_generator = srow.get("selected_generator", "")
        publication_scope = srow.get("publication_scope_12c3R", "continuous_value_limited")
        selection_status = srow.get("selection_status_12c3R", "")
        family = srow.get("family", "")
        subfamily = srow.get("synthesis_subfamily", "")
        entity = srow.get("entity", "")
    else:
        selected_generator = ""
        publication_scope = "unknown"
        selection_status = ""
        family = ""
        subfamily = ""
        entity = ""

    reasons = []
    if mask_rate_error > 0.05:
        reasons.append("mask_rate_error_gt_0.05")
    if mean_abs_error_norm > 1.00:
        reasons.append("mean_error_norm_gt_1.00")
    elif mean_abs_error_norm > 0.50:
        reasons.append("mean_error_norm_gt_0.50")
    if ks > 0.60:
        reasons.append("ks_gt_0.60")
    elif ks > 0.35:
        reasons.append("ks_gt_0.35")
    if wass > 1.00:
        reasons.append("wasserstein_norm_gt_1.00")
    elif wass > 0.50:
        reasons.append("wasserstein_norm_gt_0.50")
    if c2st_auc > 0.80:
        reasons.append("c2st_auc_gt_0.80")
    elif c2st_auc > 0.65:
        reasons.append("c2st_auc_gt_0.65")

    if any(r in reasons for r in ["mask_rate_error_gt_0.05", "mean_error_norm_gt_1.00", "ks_gt_0.60", "wasserstein_norm_gt_1.00", "c2st_auc_gt_0.80"]):
        qa_status = "fatal"
    elif reasons:
        qa_status = "warning"
    else:
        qa_status = "pass"
        reasons.append("within_thresholds")

    # Publication status is intentionally conservative.
    if publication_scope == "continuous_value_claim_included":
        publication_status = qa_status
    elif publication_scope == "continuous_value_blocked":
        publication_status = "blocked_pretest"
    else:
        publication_status = "limited_scope"

    rows.append({
        "col": col,
        "entity": entity,
        "family": family,
        "synthesis_subfamily": subfamily,
        "selected_generator": selected_generator,
        "selection_status_12c3R": selection_status,
        "publication_scope_12c3R": publication_scope,
        "test_qa_status": qa_status,
        "publication_status": publication_status,
        "qa_reasons": ";".join(reasons),
        "real_test_finite_rate": real_finite_rate,
        "synthetic_finite_rate": syn_finite_rate,
        "mask_rate_error": mask_rate_error,
        "mean_abs_error_norm": mean_abs_error_norm,
        "std_abs_error_norm": std_abs_error_norm,
        "KS": ks,
        "wasserstein_norm": wass,
        "support_jaccard": support_jaccard,
        "val_c2st_auc_logreg_1d": c2st_auc,
        "TEST_values_used_for_QA_only": True,
        "synthetic_values_mutated": False,
    })

qa = pd.DataFrame(rows)

qa_path_r = REPORT_DIR_P / "cell12c6R_final_test_qa_metrics.csv"
qa_path_canon = REPORT_DIR_P / "cell12c6_final_test_qa_metrics.csv"
selected_table_path = REPORT_DIR_P / "cell12c6_final_selected_generator_table.csv"
failures_path = REPORT_DIR_P / "cell12c6_final_test_qa_failures.csv"
pub_ready_path = REPORT_DIR_P / "cell12c6_publication_ready_test_qa_metrics.csv"
pub_blocked_path = REPORT_DIR_P / "cell12c6_publication_blocked_test_qa_metrics.csv"
summary_csv_path = REPORT_DIR_P / "cell12c6_final_publication_summary.csv"

qa.to_csv(qa_path_r, index=False)
qa.to_csv(qa_path_canon, index=False)
qa.to_csv(selected_table_path, index=False)
qa.loc[qa["test_qa_status"] == "fatal"].to_csv(failures_path, index=False)
qa.loc[qa["publication_status"].isin(["pass", "warning"])].to_csv(pub_ready_path, index=False)
qa.loc[~qa["publication_status"].isin(["pass", "warning"])].to_csv(pub_blocked_path, index=False)

status_counts = qa["test_qa_status"].value_counts().to_dict()
pub_counts = qa["publication_status"].value_counts().to_dict()

summary_table = pd.DataFrame([{
    "targets": int(len(qa)),
    "pass": int(status_counts.get("pass", 0)),
    "warning": int(status_counts.get("warning", 0)),
    "fatal": int(status_counts.get("fatal", 0)),
    "publication_status_counts": json.dumps(pub_counts, sort_keys=True),
    "mean_mask_rate_error": float(qa["mask_rate_error"].mean()),
    "mean_KS": float(qa["KS"].mean()),
    "mean_wasserstein_norm": float(qa["wasserstein_norm"].mean()),
    "mean_c2st_auc": float(qa["val_c2st_auc_logreg_1d"].mean()),
}])
summary_table.to_csv(summary_csv_path, index=False)

manifest = {
    "cell": "12.c.6R",
    "role": "continuous_terminal_test_qa_report_only_replacement",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "targets": int(len(qa)),
    "status_counts": status_counts,
    "publication_status_counts": pub_counts,
    "mean_metrics": {
        "mean_mask_rate_error": float(qa["mask_rate_error"].mean()),
        "mean_KS": float(qa["KS"].mean()),
        "mean_wasserstein_norm": float(qa["wasserstein_norm"].mean()),
        "mean_c2st_auc": float(qa["val_c2st_auc_logreg_1d"].mean()),
    },
    "TEST_values_used_for_QA_only": True,
    "synthetic_values_mutated": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "post_TEST_repair_done_here": False,
    "continuous_branch_interpretation": (
        "Replacement branch built from clean Cell 12.c.2 TRAIN/VAL candidate metrics and cache. "
        "Continuous values are conservative/limited if publication_status is limited_scope or fatal."
    ),
    "outputs": {
        "qa_metrics": str(qa_path_r),
        "canonical_qa_metrics": str(qa_path_canon),
        "selected_generator_table": str(selected_table_path),
        "failures": str(failures_path),
    },
}
manifest_path = REPORT_DIR_P / "cell12c6_final_publication_manifest.json"
manifest_r_path = REPORT_DIR_P / "cell12c6R_final_publication_manifest.json"
contract_path = CONTRACT_DIR_P / "cell12c6_final_qa_contract.json"
contract_v_path = CONTRACT_DIR_P / "cell12c6_final_qa_contract_v1_4_THESIS.json"
contract_r_path = CONTRACT_DIR_P / "cell12c6R_final_qa_contract.json"

for p in [manifest_path, manifest_r_path, contract_path, contract_v_path, contract_r_path]:
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

CELL12C6R_QA = qa
CELL12C6R_MANIFEST = manifest
CELL12C6R_QA_PATH = str(qa_path_r)

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}
CFG["cell12c6R_active"] = True
CFG["cell12c6R_test_values_used_for_QA_only"] = True
CFG["cell12c6R_synthetic_values_mutated"] = False

_c6r_log(f"Terminal QA complete | targets={len(qa)} | status_counts={status_counts} | publication_counts={pub_counts}")
_c6r_log(f"Contract: {contract_path}")

if display is not None:
    display(summary_table)
    display(qa.sort_values(["test_qa_status", "wasserstein_norm", "KS"], ascending=[True, False, False]).head(30))
