# %% CELL EXT.A.T1BASE — T1 label base-rate audit (same standard applied to the surviving task)
feats5 = sorted(set(REGIMES["R3_governed"]) & set(real_tr.columns))[:5]
_, sp_tr = make_windows(real_tr, feats5)
_, sp_te = make_windows(real_te, feats5)
_, sp_ts = make_windows(syn_pub, feats5)
print(f"T1 base rate — TRAIN: {t1_labels(real_tr, sp_tr).mean():.3f} | "
      f"TEST: {t1_labels(real_te, sp_te).mean():.3f} | SYN(pub): {t1_labels(syn_pub, sp_ts).mean():.3f}")
# Reading rule, declared now: 0.15–0.85 on both real splits = healthy; outside that = disclose
# in threats with the same sentence pattern used for T2; drift real-vs-syn > 0.15 = disclose too.