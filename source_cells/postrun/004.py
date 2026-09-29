
# %% THESIS.REV.1 — window-vs-pooled reconciliation for EXT.C
# Purpose: explain why Table 13 pooled continuous numbers differ from window-stability numbers.
# Output:
#   reports/review_fix_outputs/window_vs_pooled_reconciliation.csv
#   EXT_OUT/tables/EXT_C_window_vs_pooled_reconciliation.csv
#   reports/review_fix_outputs/window_vs_pooled_explanation.md

window_path = find_one("EXT_C_window_metrics_v3.csv", roots=[EXT_OUT_FIX], label="EXT.C window metrics")
agreement_path = find_one("EXT_C2_instrument_agreement.csv", roots=[EXT_OUT_FIX], label="EXT.C instrument agreement")
cont_ledger_path = find_one([
    "cell12c6R_final_test_qa_metrics.csv",
    "cell12c6_iot_value_final_test_qa_metrics.csv",
    "*12c6*qa*metrics*.csv",
], roots=[REPORT_DIR_FIX, RUN_ROOT], label="continuous TEST QA ledger")

win = read_csv_required(window_path)
agree = read_csv_required(agreement_path)
cont = read_csv_required(cont_ledger_path)

ks_col = pick_col(cont, ["ks"], exclude_any=["risk"], label="continuous KS metric")
auc_col = pick_col(cont, ["auc", "c2st"], label="continuous C2ST/AUC metric")
headline_ks = float(pd.to_numeric(cont[ks_col], errors="coerce").dropna().mean())
headline_auc = float(pd.to_numeric(cont[auc_col], errors="coerce").dropna().mean())
headline_n = int(pd.to_numeric(cont[auc_col], errors="coerce").dropna().shape[0])

win_ks_col = pick_col(win, ["mean_ks"], label="window mean KS")
win_auc_col = pick_col(win, ["mean_percol_c2st", "mean_c2st", "auc"], label="window mean per-column C2ST")
window_ks_mean = float(pd.to_numeric(win[win_ks_col], errors="coerce").dropna().mean())
window_ks_sd = float(pd.to_numeric(win[win_ks_col], errors="coerce").dropna().std(ddof=1))
window_ks_min = float(pd.to_numeric(win[win_ks_col], errors="coerce").dropna().min())
window_ks_max = float(pd.to_numeric(win[win_ks_col], errors="coerce").dropna().max())
window_auc_mean = float(pd.to_numeric(win[win_auc_col], errors="coerce").dropna().mean())
window_auc_sd = float(pd.to_numeric(win[win_auc_col], errors="coerce").dropna().std(ddof=1))
window_auc_min = float(pd.to_numeric(win[win_auc_col], errors="coerce").dropna().min())
window_auc_max = float(pd.to_numeric(win[win_auc_col], errors="coerce").dropna().max())

# EXT.C2 stores the Table-13/pipeline C2ST and the EXT blocked per-column C2ST side by side.
# Usually the EXT column is named ext_auc; the remaining numeric column is the pipeline ledger AUC.
num_cols = []
for c in agree.columns:
    vals = pd.to_numeric(agree[c], errors="coerce")
    if vals.notna().sum() >= max(3, len(agree) // 4):
        num_cols.append(c)
if not num_cols:
    raise RuntimeError(f"No numeric C2ST columns found in {agreement_path}. columns={list(agree.columns)}")
ext_auc_col = "ext_auc" if "ext_auc" in agree.columns else num_cols[-1]
pipe_auc_candidates = [c for c in num_cols if c != ext_auc_col]
if not pipe_auc_candidates:
    raise RuntimeError(f"Could not identify pipeline C2ST column in {agreement_path}. numeric columns={num_cols}")
pipe_auc_col = pipe_auc_candidates[0]
ext_pooled_auc = float(pd.to_numeric(agree[ext_auc_col], errors="coerce").dropna().mean())
pipe_agreement_auc = float(pd.to_numeric(agree[pipe_auc_col], errors="coerce").dropna().mean())

recon = pd.DataFrame([
    {
        "quantity": "Table13_pooled_full_TEST_mean_KS",
        "value": headline_ks,
        "n_units": headline_n,
        "source_file": rel_to_run(cont_ledger_path),
        "metric_instrument": "pipeline per-target KS over full TEST horizon",
        "comparison_note": "Pooled full-TEST branch summary.",
    },
    {
        "quantity": "EXTC_window_mean_KS",
        "value": window_ks_mean,
        "n_units": int(len(win)),
        "source_file": rel_to_run(window_path),
        "metric_instrument": "same KS family recomputed inside contiguous TEST windows",
        "comparison_note": "Higher than pooled KS because it evaluates time-local regimes before temporal mixing.",
    },
    {
        "quantity": "Table13_pooled_full_TEST_mean_C2ST_AUC",
        "value": headline_auc,
        "n_units": headline_n,
        "source_file": rel_to_run(cont_ledger_path),
        "metric_instrument": f"pipeline per-target C2ST/AUC column `{auc_col}`",
        "comparison_note": "Headline Table 13 C2ST instrument.",
    },
    {
        "quantity": "EXTC_same_instrument_pooled_percol_C2ST_AUC",
        "value": ext_pooled_auc,
        "n_units": int(len(agree)),
        "source_file": rel_to_run(agreement_path),
        "metric_instrument": f"EXT blocked per-column C2ST column `{ext_auc_col}` over full TEST",
        "comparison_note": "Same instrument as window C2ST; not numerically identical to Table 13 pipeline C2ST.",
    },
    {
        "quantity": "EXTC_window_mean_percol_C2ST_AUC",
        "value": window_auc_mean,
        "n_units": int(len(win)),
        "source_file": rel_to_run(window_path),
        "metric_instrument": f"EXT blocked per-column C2ST column `{win_auc_col}` inside contiguous TEST windows",
        "comparison_note": "Compare primarily with EXTC_same_instrument_pooled_percol_C2ST_AUC, not directly with Table 13 AUC.",
    },
])

recon_extra = {
    "window_KS_mean_minus_Table13_KS": window_ks_mean - headline_ks,
    "EXT_pooled_C2ST_minus_Table13_C2ST": ext_pooled_auc - headline_auc,
    "window_C2ST_minus_EXT_pooled_C2ST": window_auc_mean - ext_pooled_auc,
    "window_C2ST_minus_Table13_C2ST_raw_difference": window_auc_mean - headline_auc,
    "window_KS_range": [window_ks_min, window_ks_max],
    "window_C2ST_range": [window_auc_min, window_auc_max],
    "window_KS_sd": window_ks_sd,
    "window_C2ST_sd": window_auc_sd,
    "pipeline_auc_mean_from_EXTC2_agreement": pipe_agreement_auc,
    "pipeline_auc_mean_from_continuous_ledger": headline_auc,
    "interpretation": (
        "The KS discrepancy is pooled full-TEST versus contiguous-window local-regime scoring. "
        "The C2ST discrepancy has two components: an instrument offset between the Table 13 pipeline C2ST and EXT.C's blocked per-column C2ST, "
        "plus a smaller local-window increase when the EXT.C instrument is applied inside windows."
    ),
}

out1 = PATCH_DIR / "window_vs_pooled_reconciliation.csv"
out2 = EXT_OUT_FIX / "tables" / "EXT_C_window_vs_pooled_reconciliation.csv"
write_csv(recon, out1)
write_csv(recon, out2)
write_json(recon_extra, PATCH_DIR / "window_vs_pooled_reconciliation_manifest.json")

explanation = f"""# Window-vs-pooled reconciliation for continuous diagnostics

The window-stability values are not numerically the same object as the Table 13 pooled branch means.

* Table 13 KS is the pooled full-TEST branch mean over the continuous QA ledger: `{headline_ks:.6f}`.
* The window KS mean is `{window_ks_mean:.6f}` over {len(win)} contiguous TEST windows. This is larger because each window is scored before temporal regimes are mixed across the full TEST horizon.
* Table 13 C2ST/AUC is the pipeline per-target C2ST instrument: `{headline_auc:.6f}`.
* EXT.C uses a blocked per-column C2ST instrument. On the same full TEST horizon this instrument gives `{ext_pooled_auc:.6f}`; inside contiguous windows it gives `{window_auc_mean:.6f}`.
* Therefore the raw `{window_auc_mean:.6f}` versus `{headline_auc:.6f}` difference should not be presented as one direct pooled-vs-window discrepancy. It is instrument offset plus local-window strictness. The safe paper claim is qualitative: window diagnostics reinforce the denied continuous-realism claim and are not pass/fail gates.
"""
(PATCH_DIR / "window_vs_pooled_explanation.md").write_text(explanation, encoding="utf-8")
print(explanation)
