# %% CELL EXT.FIG.5 — Fig.5: Continuous separability ECDF with tier cutoffs
led_p = resolve_one("cont_qa_ledger", PATTERNS["cont_qa_ledger"], required=False)
if led_p is None:
    print("[EXT.FIG.5] Continuous QA ledger not found — pin EXT_PATHS['cont_qa_ledger'].")
else:
    led = pd.read_csv(led_p)
    auc_col = [c for c in led.columns if "auc" in c.lower() or "c2st" in c.lower()][0]
    st_col  = next((c for c in led.columns if "status" in c.lower() or "state" in c.lower()), None)
    aucs = led[auc_col].dropna().sort_values().to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.step(aucs, np.arange(1, len(aucs)+1)/len(aucs), where="post", color="#37474f", lw=1.8)
    if st_col is not None:
        bad = led[led[st_col].astype(str).str.lower().str.contains("fatal|block")]
        ax.plot(bad[auc_col], np.searchsorted(aucs, bad[auc_col])/len(aucs), "x", color="#c62828", ms=6, label=f"fatal/blocked (n={len(bad)})")
    for tau, ls in [(0.55, "--"), (0.60, ":")]:
        ax.axvline(tau, color="#f9a825", ls=ls, lw=1.4); ax.text(tau, 1.02, f"τ={tau}", ha="center", fontsize=8)
    ax.set_xlabel("per-target TEST C2ST/AUC (separability orientation)"); ax.set_ylabel("ECDF")
    ax.legend(fontsize=7); ax.set_title("Fig. 5 — Continuous branch: separable tail, not uniform failure", fontsize=9.5)
    save(fig, "fig5_continuous_separability_ecdf")
