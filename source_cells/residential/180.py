# %% CELL EXT.A.T2P — T2': router-regime label with LABEL-DEFINING CHANNELS REMOVED from features
# ============================================================================
# POST-HOC SUPPLEMENT. Designed AFTER the frozen rule was evaluated and the T2
# confound diagnosed. It does NOT replace T2, does NOT enter the frozen composite
# rule, and is reported in the paper as a labeled post-hoc analysis only.
# Fix implemented: identical T2 label (TRAIN-median threshold on next-window router
# sum) but router columns are EXCLUDED from every regime's feature set, so the task
# measures cross-signal predictability of router regime, not router autocorrelation.
# ============================================================================
rows = []
for regime, feats in REGIMES.items():
    feats_nr = sorted((set(feats) - set(router_cols)) & set(real_tr.columns)
                      & set(real_te.columns) & set(syn_sci.columns))
    if len(feats_nr) < 5:
        print(f"[EXT.A.T2P] {regime}: only {len(feats_nr)} non-router cols — reported but interpret with care")
    syn_src = syn_pub if regime == "R3_governed" else syn_sci
    Xtr_r, sp_tr = make_windows(real_tr, feats_nr)
    Xte_r, sp_te = make_windows(real_te, feats_nr)
    Xtr_s, sp_ts = make_windows(syn_src, feats_nr)
    ytr_r, yte_r = t2_labels(real_tr, sp_tr), t2_labels(real_te, sp_te)   # label unchanged; uses router SIGNAL, not router FEATURES
    ytr_s = t2_labels(syn_src, sp_ts)
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(max_iter=200, random_state=sd)
        trtr = roc_auc_score(yte_r, m.fit(Xtr_r, ytr_r).predict_proba(Xte_r)[:, 1])
        m = HistGradientBoostingClassifier(max_iter=200, random_state=sd)
        tstr = roc_auc_score(yte_r, m.fit(Xtr_s, ytr_s).predict_proba(Xte_r)[:, 1])
        rows.append(dict(regime=regime, task="T2prime_router_no_router_feats", seed=sd,
                         n_feats=len(feats_nr), TRTR=trtr, TSTR=tstr, gap=abs(trtr - tstr)))
    print(f"[EXT.A.T2P] {regime}: {len(feats_nr)} non-router feature cols")

resT2P = pd.DataFrame(rows)
resT2P.to_csv(EXT_OUT/"tables"/"EXT_A_T2prime_posthoc.csv", index=False)
print(resT2P.groupby("regime")[["TRTR","TSTR","gap"]].agg(["mean","std"]).round(4).to_string())
ok = sum(int(resT2P[(resT2P.seed==sd)].set_index("regime")["gap"]["R3_governed"]
             < resT2P[(resT2P.seed==sd)].set_index("regime")["gap"][["R1_aggregate_gate","R2_role_only"]].min())
         for sd in SEEDS)
print(f"[EXT.A.T2P] governed-smallest on T2' in {ok}/{len(SEEDS)} seeds  [POST-HOC — not part of the frozen rule]")