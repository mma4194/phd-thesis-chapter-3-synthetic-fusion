# %% CELL EXT.A.MF — Model-family sensitivity for the governance consequence study
# Linear family (LogisticRegression / Ridge): DETERMINISTIC -> single run per regime/task.
# RandomForest (optional): stochastic -> 5 seeds. Frozen conditions re-checked per family.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error

RUN_RF = True                      # set False to skip the slower stochastic family
SEEDS = [20260702, 20260703, 20260704, 20260705, 20260706]

def cls_model(family, seed):
    if family == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)

def reg_model(family, seed):
    if family == "linear":
        return make_pipeline(StandardScaler(), Ridge())
    return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)

def score_cls(fam, sd, Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2: return np.nan
    m = cls_model(fam, sd).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

def score_reg(fam, sd, Xtr, Ytr, Xte, Yte):
    maes, naive = [], []
    for j in range(Ytr.shape[1]):
        m = reg_model(fam, sd).fit(Xtr, Ytr[:, j])
        maes.append(mean_absolute_error(Yte[:, j], m.predict(Xte)))
        naive.append(mean_absolute_error(Yte[:, j], np.full_like(Yte[:, j], Ytr[:, j].mean())))
    return float(np.mean(maes) / max(np.mean(naive), 1e-12))

FAMILIES = [("linear", [0])] + ([("rf", SEEDS)] if RUN_RF else [])
rows = []
for regime, feats in REGIMES.items():
    feats_both = sorted(set(feats) & set(real_tr.columns) & set(real_te.columns) & set(syn_sci.columns))
    syn_src = syn_pub if regime == "R3_governed" else syn_sci
    Xtr_r, sp_tr = make_windows(real_tr, feats_both)
    Xte_r, sp_te = make_windows(real_te, feats_both)
    Xtr_s, sp_ts = make_windows(syn_src, feats_both)
    labels = {}
    for task, builder in [("T1_driver_next", t1_labels), ("T2_router_regime", t2_labels)]:
        labels[task] = (builder(real_tr, sp_tr), builder(real_te, sp_te), builder(syn_src, sp_ts))
    for fam, seeds in FAMILIES:
        for task in ("T1_driver_next", "T2_router_regime"):
            ytr_r, yte_r, ytr_s = labels[task]
            for sd in seeds:
                trtr = score_cls(fam, sd, Xtr_r, ytr_r, Xte_r, yte_r)
                tstr = score_cls(fam, sd, Xtr_s, ytr_s, Xte_r, yte_r)
                rows.append(dict(family=fam, regime=regime, task=task, seed=sd,
                                 TRTR=trtr, TSTR=tstr, gap=abs(trtr - tstr)))
        if regime != "R3_governed":
            Ytr_r, _ = t3_targets(real_tr, sp_tr); Yte_r, _ = t3_targets(real_te, sp_te)
            Ytr_s, _ = t3_targets(syn_sci, sp_ts)
            for sd in seeds:
                trtr = score_reg(fam, sd, Xtr_r, Ytr_r, Xte_r, Yte_r)
                tstr = score_reg(fam, sd, Xtr_s, Ytr_s, Xte_r, Yte_r)
                rows.append(dict(family=fam, regime=regime, task="T3_cont_forecast(DENIED)",
                                 seed=sd, TRTR=trtr, TSTR=tstr, gap=abs(trtr - tstr)))
    print(f"[EXT.A.MF] {regime} done ({len(feats_both)} cols)")

resMF = pd.DataFrame(rows)
resMF.to_csv(EXT_OUT/"tables"/"EXT_A_model_family_sensitivity.csv", index=False)
agg = resMF.groupby(["family","regime","task"])["gap"].agg(["mean","std","min","max"]).round(4)
print(agg.to_string())

# Frozen-condition check PER FAMILY (c1: T1 governed smallest; c3: denied > all permitted)
for fam, seeds in FAMILIES:
    for sd in seeds:
        d = resMF[(resMF.family == fam) & (resMF.seed == sd)].set_index(["regime","task"])["gap"]
        c1 = d[("R3_governed","T1_driver_next")] < min(d[("R1_aggregate_gate","T1_driver_next")],
                                                       d[("R2_role_only","T1_driver_next")])
        c2 = d[("R3_governed","T2_router_regime")] < min(d[("R1_aggregate_gate","T2_router_regime")],
                                                         d[("R2_role_only","T2_router_regime")])
        denied = min(d[("R1_aggregate_gate","T3_cont_forecast(DENIED)")],
                     d[("R2_role_only","T3_cont_forecast(DENIED)")])
        c3 = denied > max(v for (r, t), v in d.items() if "DENIED" not in t)
        print(f"[EXT.A.MF] family={fam:6s} seed={sd}: c1={c1} c2={c2} c3={c3}")
print("[EXT.A.MF] Reading rule (declared now): the paper's robustness claim is per-condition — "
      "c1 and c3 must hold for every family/seed to claim family-robustness; c2 is expected to fail (documented confound).")