# %% CELL EXT.Q4H.1 — Untouched-holdout availability check + frozen 6-pair rescore
# The operator must declare, from the split manifest, whether an untouched terminal
# window exists (e.g., rows after the inspected TEST end, or a reserved segment).
UNTOUCHED_HOLDOUT_PARQUET = ""   # <- pin path to untouched real window (same schema as real_test), or leave "" 
UNTOUCHED_SYN_PARQUET     = ""   # <- pin matching-length A0-materialized synthetic window (generated with FROZEN policy)

frozen_policy_p = resolve_one("q4_a0_frozen_policy", PATTERNS["q4_a0_frozen_policy"], required=False)
a0_terminal_p   = resolve_one("q4_a0_terminal", PATTERNS["q4_a0_terminal"], required=False)

if not UNTOUCHED_HOLDOUT_PARQUET:
    print("[EXT.Q4H] NO untouched holdout declared. The 6-pair subset remains DEVELOPMENT EVIDENCE.")
    print("[EXT.Q4H] Action for Mohammed: check the split manifest — 1,277,694 total rows vs "
          "TRAIN+VAL+TEST accounting. If any terminal segment was excluded from all fitting, "
          "selection, and the inspected TEST audit, pin it above and re-run. If not, this "
          "experiment is honestly impossible on this deployment; say so in §6.4.")
else:
    hold = pd.read_parquet(UNTOUCHED_HOLDOUT_PARQUET)
    synh = pd.read_parquet(UNTOUCHED_SYN_PARQUET)
    assert len(hold) == len(synh), "Holdout real/syn length mismatch."
    assert frozen_policy_p is not None, "Frozen A0 policy cache required — refusing to rescore without it."
    policy = json.load(open(frozen_policy_p))
    print(f"[EXT.Q4H] frozen policy loaded from {frozen_policy_p.name}; rescoring 6 pairs on untouched window...")
    # ETA similarity re-implementation matching §3.6 exactly:
    def eta_similarity(real_prof, syn_prof):
        rho = np.corrcoef(real_prof, syn_prof)[0,1] if np.std(real_prof)>0 and np.std(syn_prof)>0 else 0.0
        scale = abs(np.mean(real_prof)) + np.std(real_prof) + 1e-12
        mtil = np.mean(np.abs(real_prof - syn_prof)) / scale
        return 0.5*max(0.0, rho) + 0.5*np.exp(-mtil)
    # Pair extraction requires the pair manifest (driver col, protocol col, window) from the policy cache.
    pairs = policy.get("val_feasible_pairs") or policy.get("pairs")
    assert pairs, "Policy cache does not expose the 6-pair manifest — inspect keys: " + str(list(policy.keys()))
    out = []
    W_PRE, W_POST = 10, 30
    for pr in pairs:
        e_col, z_col = pr["driver"], pr["protocol"]
        ev = np.where(np.nan_to_num(hold[e_col].to_numpy(float)) > 0)[0]
        ev = ev[(ev > W_PRE) & (ev < len(hold) - W_POST)]
        if len(ev) < 5:
            out.append(dict(pair=pr.get("id", f"{e_col}->{z_col}"), status="insufficient_events", eta=np.nan)); continue
        def profile(sig):
            s = np.nan_to_num(sig, nan=0.0)
            mats = np.stack([s[i-W_PRE:i+W_POST] for i in ev])
            base = mats[:, :W_PRE].mean()
            return mats.mean(0) - base
        eta = eta_similarity(profile(hold[z_col].to_numpy(float)), profile(synh[z_col].to_numpy(float)))
        status = "blocker" if eta < 0.50 else ("warning" if eta < 0.70 else "pass")
        out.append(dict(pair=pr.get("id", f"{e_col}->{z_col}"), status=status, eta=float(eta), n_events=int(len(ev))))
    resQ4H = pd.DataFrame(out)
    resQ4H.to_csv(EXT_OUT / "tables" / "EXT_Q4H_fresh_holdout_6pair.csv", index=False)
    print(resQ4H.to_string(index=False))
    print("[EXT.Q4H] If zero blockers here -> the 6-pair claim is promotable in §5.5 with this audit cited.")
