# %% CELL EXT.A.3 — Run TRTR vs TSTR across regimes (seed-pinned, fixed models)
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error

def fit_score_cls(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return np.nan
    m = HistGradientBoostingClassifier(max_iter=200, random_state=RNG_SEED)
    m.fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

def fit_score_reg(Xtr, Ytr, Xte, Yte):
    maes, naive = [], []
    for j in range(Ytr.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=RNG_SEED)
        m.fit(Xtr, Ytr[:, j]); pred = m.predict(Xte)
        maes.append(mean_absolute_error(Yte[:, j], pred))
        naive.append(mean_absolute_error(Yte[:, j], np.full_like(Yte[:, j], Ytr[:, j].mean())))
    return float(np.mean(maes) / max(np.mean(naive), 1e-12))  # <1 beats naive

results = []
for regime, feats in REGIMES.items():
    feats_real = [c for c in feats if c in real_tr.columns and c in real_te.columns]
    feats_syn  = [c for c in feats if c in syn_sci.columns]
    feats_both = sorted(set(feats_real) & set(feats_syn))
    print(f"[EXT.A.3] {regime}: {len(feats_both)} usable columns (real∩syn)")

    Xtr_r, sp_tr = make_windows(real_tr, feats_both)
    Xte_r, sp_te = make_windows(real_te, feats_both)
    Xtr_s, sp_ts = make_windows(syn_sci if regime != "R3_governed" else syn_pub, feats_both)

    for task, builder, scorer, kind in [
        ("T1_driver_next", t1_labels, fit_score_cls, "auroc"),
        ("T2_router_regime", t2_labels, fit_score_cls, "auroc"),
    ]:
        ytr_r, yte_r = builder(real_tr, sp_tr), builder(real_te, sp_te)
        ytr_s = builder(syn_sci if regime != "R3_governed" else syn_pub, sp_ts)
        trtr = scorer(Xtr_r, ytr_r, Xte_r, yte_r)
        tstr = scorer(Xtr_s, ytr_s, Xte_r, yte_r)
        results.append(dict(regime=regime, task=task, metric=kind, TRTR=trtr, TSTR=tstr,
                            gap=abs(trtr - tstr) if np.isfinite(trtr) and np.isfinite(tstr) else np.nan))

    if regime in ("R1_aggregate_gate", "R2_role_only"):  # governance-DENIED scope task
        Ytr_r, tg = t3_targets(real_tr, sp_tr); Yte_r, _ = t3_targets(real_te, sp_te)
        Ytr_s, _  = t3_targets(syn_sci, sp_ts)
        trtr = fit_score_reg(Xtr_r, Ytr_r, Xte_r, Yte_r)
        tstr = fit_score_reg(Xtr_s, Ytr_s, Xte_r, Yte_r)
        results.append(dict(regime=regime, task="T3_cont_forecast(DENIED)", metric="mae_ratio",
                            TRTR=trtr, TSTR=tstr, gap=abs(trtr - tstr)))

resA = pd.DataFrame(results)
resA.to_csv(EXT_OUT / "tables" / "EXT_A_regime_consequence.csv", index=False)
print(resA.to_string(index=False))
print("[EXT.A.3] Interpretation rule (pre-declared): governance demonstrated iff "
      "R3 gap < R1 and R2 gaps on T1/T2, AND T3 TSTR degradation under R1/R2 exceeds "
      "any permitted-scope gap. This rule is frozen BEFORE seeing the numbers.")
