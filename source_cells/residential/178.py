# %% CELL EXT.A.5SEED — Experiment A replication over 5 seeds (rule frozen; variance reported)
SEEDS = [20260702, 20260703, 20260704, 20260705, 20260706]
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error

def fit_score_cls_s(Xtr, ytr, Xte, yte, seed):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2: return np.nan
    m = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

def fit_score_reg_s(Xtr, Ytr, Xte, Yte, seed):
    maes, naive = [], []
    for j in range(Ytr.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=seed).fit(Xtr, Ytr[:, j])
        maes.append(mean_absolute_error(Yte[:, j], m.predict(Xte)))
        naive.append(mean_absolute_error(Yte[:, j], np.full_like(Yte[:, j], Ytr[:, j].mean())))
    return float(np.mean(maes) / max(np.mean(naive), 1e-12))

rows = []
for regime, feats in REGIMES.items():
    feats_both = sorted(set(feats) & set(real_tr.columns) & set(real_te.columns) & set(syn_sci.columns))
    syn_src = syn_pub if regime == "R3_governed" else syn_sci
    Xtr_r, sp_tr = make_windows(real_tr, feats_both)
    Xte_r, sp_te = make_windows(real_te, feats_both)
    Xtr_s, sp_ts = make_windows(syn_src, feats_both)
    tasks = [("T1_driver_next", t1_labels, "cls"), ("T2_router_regime", t2_labels, "cls")]
    for task, builder, kind in tasks:
        ytr_r, yte_r = builder(real_tr, sp_tr), builder(real_te, sp_te)
        ytr_s = builder(syn_src, sp_ts)
        for sd in SEEDS:
            trtr = fit_score_cls_s(Xtr_r, ytr_r, Xte_r, yte_r, sd)
            tstr = fit_score_cls_s(Xtr_s, ytr_s, Xte_r, yte_r, sd)
            rows.append(dict(regime=regime, task=task, seed=sd, TRTR=trtr, TSTR=tstr, gap=abs(trtr - tstr)))
    if regime != "R3_governed":
        Ytr_r, _ = t3_targets(real_tr, sp_tr); Yte_r, _ = t3_targets(real_te, sp_te)
        Ytr_s, _ = t3_targets(syn_sci, sp_ts)
        for sd in SEEDS:
            trtr = fit_score_reg_s(Xtr_r, Ytr_r, Xte_r, Yte_r, sd)
            tstr = fit_score_reg_s(Xtr_s, Ytr_s, Xte_r, Yte_r, sd)
            rows.append(dict(regime=regime, task="T3_cont_forecast(DENIED)", seed=sd,
                             TRTR=trtr, TSTR=tstr, gap=abs(trtr - tstr)))
    print(f"[EXT.A.5SEED] {regime}: done ({len(feats_both)} cols)")

resA5 = pd.DataFrame(rows)
agg = resA5.groupby(["regime","task"])["gap"].agg(["mean","std","min","max"]).round(4)
resA5.to_csv(EXT_OUT/"tables"/"EXT_A_regime_consequence_5seed.csv", index=False)
agg.to_csv(EXT_OUT/"tables"/"EXT_A_gap_summary_5seed.csv")
print(agg.to_string())

# Frozen rule evaluated PER SEED — the claim is 'ordering holds in k/5 seeds', nothing softer
ok = 0
for sd in SEEDS:
    d = resA5[resA5.seed == sd].set_index(["regime","task"])["gap"]
    c1 = d.get(("R3_governed","T1_driver_next"), np.inf) < min(d.get(("R1_aggregate_gate","T1_driver_next"), 0), d.get(("R2_role_only","T1_driver_next"), 0))
    c2 = d.get(("R3_governed","T2_router_regime"), np.inf) < min(d.get(("R1_aggregate_gate","T2_router_regime"), 0), d.get(("R2_role_only","T2_router_regime"), 0))
    denied = min(d.get(("R1_aggregate_gate","T3_cont_forecast(DENIED)"), 0), d.get(("R2_role_only","T3_cont_forecast(DENIED)"), 0))
    permitted_max = max(v for (r, t), v in d.items() if t != "T3_cont_forecast(DENIED)")
    c3 = denied > permitted_max
    ok += int(c1 and c2 and c3)
    print(f"[EXT.A.5SEED] seed {sd}: rule {'HOLDS' if (c1 and c2 and c3) else 'FAILS'} (c1={c1} c2={c2} c3={c3})")
print(f"[EXT.A.5SEED] frozen rule holds in {ok}/{len(SEEDS)} seeds. "
      "Abstract wording is decided by THIS number, not by the single-seed run.")