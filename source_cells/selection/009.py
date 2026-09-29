# %% STRONG.6 — Continuous C2ST primary-instrument reconciliation and Table-13 replacement values
# Preferred interpretation: use the stricter blocked per-column EXT.C instrument as the primary separability diagnostic.

rows = []
legacy_auc = np.nan
legacy_ks = np.nan
legacy_wass = np.nan

if PATHS.get("continuous_qa"):
    cont = pd.read_csv(PATHS["continuous_qa"])
    auc_col = pick_col(cont, ["c2st_auc", "auc", "c2st", "separability"], required=False)
    ks_col = pick_col(cont, ["ks", "ks_stat", "ks_statistic", "mean_ks"], required=False)
    wass_col = pick_col(cont, ["wasserstein", "normalized_wasserstein", "norm_wasserstein"], required=False)
    if auc_col:
        legacy_auc = float(pd.to_numeric(cont[auc_col], errors="coerce").mean())
    if ks_col:
        legacy_ks = float(pd.to_numeric(cont[ks_col], errors="coerce").mean())
    if wass_col:
        legacy_wass = float(pd.to_numeric(cont[wass_col], errors="coerce").mean())
    rows.append({"metric": "legacy_pipeline_table13_c2st_auc_mean", "value": legacy_auc, "source": rel_to_run(PATHS["continuous_qa"]), "interpretation": "legacy pipeline branch-readiness instrument"})
    rows.append({"metric": "legacy_pipeline_table13_ks_mean", "value": legacy_ks, "source": rel_to_run(PATHS["continuous_qa"]), "interpretation": "legacy pooled full-TEST branch mean"})
    rows.append({"metric": "legacy_pipeline_table13_wasserstein_mean", "value": legacy_wass, "source": rel_to_run(PATHS["continuous_qa"]), "interpretation": "legacy pooled full-TEST branch mean"})

ext_pooled_auc = np.nan
instrument_rho = np.nan
if PATHS.get("ext_c_agreement"):
    agree = pd.read_csv(PATHS["ext_c_agreement"])
    # Identify EXT column and pipeline column.
    ext_col = next((c for c in agree.columns if "ext" in str(c).lower() and ("auc" in str(c).lower() or "c2st" in str(c).lower())), None)
    numeric_cols = [c for c in agree.columns if pd.to_numeric(agree[c], errors="coerce").notna().sum() >= max(3, len(agree)//4)]
    pipe_candidates = [c for c in numeric_cols if c != ext_col and ("auc" in str(c).lower() or "c2st" in str(c).lower() or c != agree.columns[0])]
    pipe_col = pipe_candidates[0] if pipe_candidates else None
    if ext_col:
        ext_pooled_auc = float(pd.to_numeric(agree[ext_col], errors="coerce").mean())
    if ext_col and pipe_col and SCIPY_AVAILABLE:
        tmp = agree[[pipe_col, ext_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(tmp) >= 3:
            instrument_rho = float(stats.spearmanr(tmp[pipe_col], tmp[ext_col]).statistic)
    rows.append({"metric": "primary_ext_blocked_per_column_pooled_c2st_auc_mean", "value": ext_pooled_auc, "source": rel_to_run(PATHS["ext_c_agreement"]), "interpretation": "recommended primary continuous separability instrument"})
    rows.append({"metric": "pipeline_vs_ext_spearman_rho", "value": instrument_rho, "source": rel_to_run(PATHS["ext_c_agreement"]), "interpretation": "agreement of column ranking, not level equality"})

window_auc_mean = np.nan
window_auc_sd = np.nan
window_ks_mean = np.nan
window_ks_sd = np.nan
if PATHS.get("ext_c_windows"):
    win = pd.read_csv(PATHS["ext_c_windows"])
    auc_w_col = pick_col(win, ["mean_percol_c2st", "mean_c2st", "c2st", "auc"], required=False)
    ks_w_col = pick_col(win, ["mean_ks", "ks"], required=False)
    if auc_w_col:
        vals = pd.to_numeric(win[auc_w_col], errors="coerce").dropna()
        window_auc_mean = float(vals.mean()) if len(vals) else np.nan
        window_auc_sd = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
    if ks_w_col:
        vals = pd.to_numeric(win[ks_w_col], errors="coerce").dropna()
        window_ks_mean = float(vals.mean()) if len(vals) else np.nan
        window_ks_sd = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
    rows.append({"metric": "primary_ext_window_mean_c2st_auc", "value": window_auc_mean, "source": rel_to_run(PATHS["ext_c_windows"]), "interpretation": "time-local separability under primary EXT.C instrument"})
    rows.append({"metric": "primary_ext_window_sd_c2st_auc", "value": window_auc_sd, "source": rel_to_run(PATHS["ext_c_windows"]), "interpretation": "contiguous-window dispersion, not iid CI"})
    rows.append({"metric": "primary_ext_window_mean_ks", "value": window_ks_mean, "source": rel_to_run(PATHS["ext_c_windows"]), "interpretation": "time-local KS; reinforces denied realism"})

summary = pd.DataFrame(rows)
replacement = pd.DataFrame([{
    "scope": "Continuous IoT values",
    "primary_gate_and_current_evidence_replacement": (
        f"Full branch over 148 targets remains separable. Primary blocked per-column EXT.C C2ST/AUC={ext_pooled_auc:.6f}; "
        f"window mean={window_auc_mean:.6f}; legacy pipeline C2ST/AUC={legacy_auc:.6f} retained as secondary ledger evidence; "
        f"legacy mean KS={legacy_ks:.6f}."
    ),
    "claim_consequence": "No uniform continuous-value realism claim; primary separability instrument is EXT.C blocked per-column C2ST.",
    "primary_instrument": "EXT.C blocked per-column C2ST",
    "secondary_instrument": "pipeline branch-readiness C2ST",
}])

write_csv(summary, STRONG_TABLES / "continuous_c2st_primary_instrument_summary.csv")
write_csv(replacement, STRONG_TABLES / "continuous_c2st_table13_replacement_row.csv")
write_text(
    "# Continuous C2ST primary-instrument note\n\n"
    "Use the blocked per-column EXT.C instrument as the primary continuous separability diagnostic. "
    "The original pipeline C2ST remains a secondary branch-readiness ledger value, not the headline if both are shown.\n",
    STRONG_DIR / "continuous_c2st_primary_instrument_note.md",
)
summary
