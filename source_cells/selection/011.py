# %% STRONG.8 — Q5 consequence evidence: downgrade existing 5-seed claim and optional extended-seed rerun
# Default behavior is fast and conservative: create a claim-scope table from existing evidence.
# Set THESIS_RUN_Q5_EXTENDED_SEEDS=1 to rerun the diagnostic over Q5_SEEDS.

q5_scope_rows = []
if PATHS.get("q5_5seed"):
    q5 = pd.read_csv(PATHS["q5_5seed"])
    # Evaluate frozen rule by seed where possible.
    seed_col = pick_col(q5, ["seed"], required=False)
    regime_col = pick_col(q5, ["regime"], required=False)
    task_col = pick_col(q5, ["task"], required=False)
    gap_col = pick_col(q5, ["gap"], required=False)
    ok_n = np.nan
    total_n = np.nan
    if seed_col and regime_col and task_col and gap_col:
        ok = 0
        total = 0
        for sd, g in q5.groupby(seed_col):
            d = g.set_index([regime_col, task_col])[gap_col].to_dict()
            c1 = d.get(("R3_governed", "T1_driver_next"), np.inf) < min(d.get(("R1_aggregate_gate", "T1_driver_next"), 0), d.get(("R2_role_only", "T1_driver_next"), 0))
            c2 = d.get(("R3_governed", "T2_router_regime"), np.inf) < min(d.get(("R1_aggregate_gate", "T2_router_regime"), 0), d.get(("R2_role_only", "T2_router_regime"), 0))
            denied = min(d.get(("R1_aggregate_gate", "T3_cont_forecast(DENIED)"), 0), d.get(("R2_role_only", "T3_cont_forecast(DENIED)"), 0))
            permitted = [v for (r, t), v in d.items() if t != "T3_cont_forecast(DENIED)" and np.isfinite(v)]
            c3 = bool(permitted) and denied > max(permitted)
            ok += int(c1 and c2 and c3)
            total += 1
        ok_n, total_n = ok, total
    q5_scope_rows.append({
        "evidence": "existing_q5_5seed_consequence_study",
        "source": rel_to_run(PATHS["q5_5seed"]),
        "frozen_rule_holds_n": ok_n,
        "seeds_n": total_n,
        "allowed_claim": "diagnostic governance consequence evidence only",
        "disallowed_claim": "leakage-safe anomaly-detection utility or strong downstream validity proof",
        "reason": "underpowered and one frozen-rule failure remains material unless extended seed evidence changes the conclusion",
    })
else:
    q5_scope_rows.append({
        "evidence": "existing_q5_5seed_consequence_study",
        "source": "missing",
        "frozen_rule_holds_n": np.nan,
        "seeds_n": np.nan,
        "allowed_claim": "no Q5 consequence claim from this notebook until evidence ledger is present",
        "disallowed_claim": "downstream utility",
        "reason": "Q5 seed ledger not found",
    })

q5_scope = pd.DataFrame(q5_scope_rows)
write_csv(q5_scope, STRONG_TABLES / "q5_claim_scope_downgrade_from_existing_evidence.csv")

# Optional extended-seed rerun. It reuses the final notebook's task definitions with stricter guards.
extended_status = {"ran": False, "reason": "RUN_Q5_EXTENDED_SEEDS is false"}
if RUN_Q5_EXTENDED_SEEDS:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for extended Q5 seed rerun.")
    needed = [PATHS.get("real_train"), PATHS.get("real_test"), PATHS.get("syn_scientific")]
    if any(p is None for p in needed):
        raise RuntimeError("Extended Q5 requires REAL_TRAIN_SPLIT, REAL_TEST_SPLIT, and scientific synthetic artifact.")
    real_tr = pd.read_parquet(PATHS["real_train"])
    real_te = pd.read_parquet(PATHS["real_test"])
    syn_sci = pd.read_parquet(PATHS["syn_scientific"])
    cons_path = STRONG_SYNTH / "CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet"
    syn_pub_for_q5 = pd.read_parquet(cons_path if (Q5_USE_CONSERVATIVE_PUBLIC and cons_path.exists()) else PATHS["syn_public_existing"])
    # Load role map from role ownership if available.
    if PATHS.get("role_ownership") is None:
        raise RuntimeError("Extended Q5 requires iot_role_ownership.csv for role-scoped regimes.")
    role_df = pd.read_csv(PATHS["role_ownership"])
    col_col = pick_col(role_df, ["col", "column", "feature", "name"], required=True)
    owner_col = pick_col(role_df, ["primary_owner", "owner_role", "role", "owner"], required=True)
    roles = dict(zip(role_df[col_col].astype(str), role_df[owner_col].astype(str)))
    cont_cols = [c for c, r in roles.items() if "continuous" in r.lower() and c in real_tr.columns and c in syn_sci.columns]
    binary_cols = [c for c, r in roles.items() if "binary" in r.lower() and c in real_tr.columns and c in syn_sci.columns]
    driver_cols_all = [c for c, r in roles.items() if "driver" in r.lower() and c in real_tr.columns]
    q6_reg = pd.read_csv(PATHS["q6_registry"])
    reg_col = pick_col(q6_reg, ["col", "column", "feature", "name"], required=True)
    reg_role = pick_col(q6_reg, ["role_group", "owner_role", "role", "branch", "category", "group"], required=False)
    router_cols = [c for c in q6_reg[reg_col].astype(str) if c in syn_pub_for_q5.columns and ("router" in c.lower() or (reg_role and q6_reg.loc[q6_reg[reg_col].astype(str).eq(c), reg_role].astype(str).str.contains("router", case=False, na=False).any()))]
    zigbee_cols = [c for c in q6_reg[reg_col].astype(str) if c in syn_pub_for_q5.columns and ("zigbee" in c.lower() or "zb" in c.lower())]
    driver_public = [c for c in driver_cols_all if c in syn_pub_for_q5.columns]
    excluded = [c for c, r in roles.items() if "excluded" in r.lower()]
    REGIMES = {
        "R1_aggregate_gate": [c for c in syn_sci.columns if c in real_tr.columns and c not in set(excluded)],
        "R2_role_only": sorted(set(driver_public + router_cols + zigbee_cols + cont_cols + binary_cols)),
        "R3_governed": sorted(set(driver_public + router_cols + zigbee_cols)),
    }
    for k in REGIMES:
        REGIMES[k] = [c for c in REGIMES[k] if c in real_tr.columns and c in real_te.columns and (c in syn_sci.columns or k == "R3_governed")]
    WIN_HIST, WIN_FUT, STRIDE = 300, 60, 60
    def make_windows(df, feat_cols, n_hist=WIN_HIST, n_fut=WIN_FUT, stride=STRIDE):
        arr = df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        X, spans = [], []
        for s in range(0, len(df) - n_hist - n_fut, stride):
            h = arr[s:s+n_hist]
            X.append(np.concatenate([h.mean(0), h.std(0), h[-60:].mean(0)]))
            spans.append((s, s+n_hist, s+n_hist+n_fut))
        return np.asarray(X), spans
    def t1_labels(df, spans):
        cols = [c for c in driver_public if c in df.columns]
        if not cols:
            return None
        d = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        return np.array([(d[a:b] > 0).any() for (_, a, b) in spans], dtype=int)
    t2_cols = [c for c in router_cols if c in real_tr.columns]
    if not t2_cols:
        raise RuntimeError("No router columns available for T2.")
    T2_THRESH = float(np.nanmedian(real_tr[t2_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)))
    def t2_labels(df, spans):
        cols = [c for c in router_cols if c in df.columns]
        if not cols:
            return None
        r = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).to_numpy(dtype=np.float32)
        return np.array([float(np.nanmean(r[a:b]) > T2_THRESH) for (_, a, b) in spans], dtype=int)
    def t3_targets(df, spans, k_targets=10):
        tgts = sorted([c for c in cont_cols if c in df.columns])[:k_targets]
        if not tgts:
            return None, []
        arr = df[tgts].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        y = np.array([np.nanmean(arr[a:b], axis=0) for (_, a, b) in spans])
        return np.nan_to_num(y, nan=0.0), tgts
    def fit_score_cls(Xtr, ytr, Xte, yte, seed):
        if ytr is None or yte is None or len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return np.nan
        m = HistGradientBoostingClassifier(max_iter=Q5_MODEL_MAX_ITER, random_state=seed).fit(Xtr, ytr)
        return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
    def fit_score_reg(Xtr, Ytr, Xte, Yte, seed):
        if Ytr is None or Yte is None:
            return np.nan
        maes, naive = [], []
        for j in range(Ytr.shape[1]):
            m = HistGradientBoostingRegressor(max_iter=Q5_MODEL_MAX_ITER, random_state=seed).fit(Xtr, Ytr[:, j])
            pred = m.predict(Xte)
            maes.append(mean_absolute_error(Yte[:, j], pred))
            naive.append(mean_absolute_error(Yte[:, j], np.full_like(Yte[:, j], Ytr[:, j].mean())))
        return float(np.mean(maes) / max(np.mean(naive), 1e-12))
    result_rows = []
    for regime, feats in REGIMES.items():
        feats = sorted([c for c in feats if c in real_tr.columns and c in real_te.columns])
        syn_src = syn_pub_for_q5 if regime == "R3_governed" else syn_sci
        feats = [c for c in feats if c in syn_src.columns]
        if len(feats) < 5:
            warnings.warn(f"Skipping {regime}: only {len(feats)} common features")
            continue
        Xtr_r, sp_tr = make_windows(real_tr, feats)
        Xte_r, sp_te = make_windows(real_te, feats)
        Xtr_s, sp_ts = make_windows(syn_src, feats)
        for task, builder in [("T1_driver_next", t1_labels), ("T2_router_regime", t2_labels)]:
            ytr_r, yte_r, ytr_s = builder(real_tr, sp_tr), builder(real_te, sp_te), builder(syn_src, sp_ts)
            for sd in Q5_SEEDS:
                trtr = fit_score_cls(Xtr_r, ytr_r, Xte_r, yte_r, sd)
                tstr = fit_score_cls(Xtr_s, ytr_s, Xte_r, yte_r, sd)
                result_rows.append({"regime": regime, "task": task, "seed": sd, "TRTR": trtr, "TSTR": tstr, "gap": abs(trtr-tstr) if np.isfinite(trtr) and np.isfinite(tstr) else np.nan})
        if regime != "R3_governed":
            Ytr_r, _ = t3_targets(real_tr, sp_tr); Yte_r, _ = t3_targets(real_te, sp_te); Ytr_s, _ = t3_targets(syn_sci, sp_ts)
            for sd in Q5_SEEDS:
                trtr = fit_score_reg(Xtr_r, Ytr_r, Xte_r, Yte_r, sd)
                tstr = fit_score_reg(Xtr_s, Ytr_s, Xte_r, Yte_r, sd)
                result_rows.append({"regime": regime, "task": "T3_cont_forecast(DENIED)", "seed": sd, "TRTR": trtr, "TSTR": tstr, "gap": abs(trtr-tstr) if np.isfinite(trtr) and np.isfinite(tstr) else np.nan})
    res = pd.DataFrame(result_rows)
    write_csv(res, STRONG_TABLES / "q5_consequence_extended_seed_results.csv")
    if len(res):
        agg = res.groupby(["regime", "task"], dropna=False)["gap"].agg(["mean", "std", "min", "max", "count"]).reset_index()
        write_csv(agg, STRONG_TABLES / "q5_consequence_extended_seed_summary.csv")
    extended_status = {"ran": True, "seeds_n": len(Q5_SEEDS), "result_rows": len(res)}

write_json(extended_status, STRONG_MANIFESTS / "q5_extended_seed_status.json")
write_text(
    "# Q5 claim-scope recommendation\n\n"
    "Unless the optional extended-seed run gives robust frozen-rule support, Q5 should be described as diagnostic consequence evidence, not downstream utility validation or anomaly-detection benchmarking.\n",
    STRONG_DIR / "q5_claim_scope_recommendation.md",
)

print("Q5 existing-evidence scope:")
print(q5_scope.to_string(index=False))
print("Optional extended rerun:", extended_status)
q5_scope
