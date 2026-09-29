# %% CELL EXT.A.T2P.DIAG — label base-rate and cross-period stability check
Xtr_tmp, sp_tr = make_windows(real_tr, ["dummy"] if False else sorted(set(REGIMES["R3_governed"]) & set(real_tr.columns))[:5])
_, sp_te = make_windows(real_te, sorted(set(REGIMES["R3_governed"]) & set(real_te.columns))[:5])
print(f"T2 label base rate — TRAIN windows: {t2_labels(real_tr, sp_tr).mean():.3f} | "
      f"TEST windows: {t2_labels(real_te, sp_te).mean():.3f}")
# If these differ substantially (e.g. 0.5 vs 0.2), TRAIN-median threshold + regime drift explains TRTR<0.5.