# %% CELL EXT.C.1v3 — Window dispersion v3: METRIC-MATCHED per-column KS + per-column C2ST
# Fixes v2's incommensurability: (a) reference recomputed on the SAME column set as the
# windows; (b) C2ST is the MEAN of per-column classifiers (matching the paper's per-target
# construction), not a joint multivariate classifier.
N_WINDOWS = 6
win_blocks = temporal_blocks(len(real_te), N_WINDOWS)
COLSET = sorted([c for c in cont_cols if c in syn_sci.columns and c in real_te.columns])  # ALL 148, no cap
print(f"[EXT.C.1v3] {N_WINDOWS} windows x {len(COLSET)} continuous cols (full branch, no cap)")

def per_col_metrics(rdf, sdf, cols, max_rows=20_000):
    ks_vals, auc_vals = [], []
    for c in cols:
        rr = pd.to_numeric(rdf[c], errors="coerce").dropna().to_numpy()
        ss = pd.to_numeric(sdf[c], errors="coerce").dropna().to_numpy()
        if len(rr) < 100 or len(ss) < 100: continue
        ks_vals.append(stats.ks_2samp(rr, ss).statistic)
        auc_vals.append(c2st_auc_blocked(rr, ss, n_blocks=4, max_rows=max_rows))
    return float(np.mean(ks_vals)), float(np.mean(auc_vals)), len(ks_vals)

# Full-TEST reference on the SAME column set (the number the envelope must bracket)
ref_ks, ref_auc, ref_n = per_col_metrics(real_te, syn_sci, COLSET)
print(f"[EXT.C.1v3] full-TEST reference (same colset): mean_ks={ref_ks:.4f} "
      f"mean_percol_c2st={ref_auc:.4f} over {ref_n} cols")

rows = []
for w, (a, b) in enumerate(win_blocks):
    ks_m, auc_m, n_ok = per_col_metrics(real_te.iloc[a:b], syn_sci.iloc[a:b], COLSET, max_rows=10_000)
    rows.append(dict(window=w, mean_ks=ks_m, mean_percol_c2st=auc_m, n_cols=n_ok))
    print(f"[EXT.C.1v3] window {w}: ks={ks_m:.4f} c2st={auc_m:.4f} ({n_ok} cols)")

resC3 = pd.DataFrame(rows)
env = resC3[["mean_ks","mean_percol_c2st"]].agg(["mean","std","min","max"]).round(4)
resC3.to_csv(EXT_OUT/"tables"/"EXT_C_window_metrics_v3.csv", index=False)
env.to_csv(EXT_OUT/"tables"/"EXT_C_dispersion_summary_v3.csv")
print(env.to_string())
in_ks  = env.loc["min","mean_ks"] <= ref_ks <= env.loc["max","mean_ks"]
in_auc = env.loc["min","mean_percol_c2st"] <= ref_auc <= env.loc["max","mean_percol_c2st"]
print(f"[EXT.C.1v3] bracket check (self-consistent reference): KS {'IN' if in_ks else 'OUT'} | "
      f"per-col C2ST {'IN' if in_auc else 'OUT'} — report whichever it is.")