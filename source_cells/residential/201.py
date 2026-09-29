# %% CELL EXT.FIG.4 — Fig.4: Q4 event-triggered alignment profiles (real vs synthetic)
prof_p = resolve_one("q4_pair_profiles", PATTERNS["q4_pair_profiles"], required=False)
if prof_p is None:
    print("[EXT.FIG.4] No persisted ETA profile arrays found. Two options:\n"
          "  (a) re-run CELL 14.6R1 with profile persistence enabled (save real/syn baseline-corrected\n"
          "      window means per pair as npz: keys real_<pairid>, syn_<pairid>), then re-run this cell;\n"
          "  (b) pin the npz path in EXT_PATHS['q4_pair_profiles'].\n"
          "Fail-closed: not fabricating profiles.")
else:
    dat = np.load(prof_p, allow_pickle=True)
    pair_ids = sorted({k.split("_",1)[1] for k in dat.files if k.startswith("real_")})
    # choose exemplars: best pass, one warning, the fatal 0006, worst generic if present
    want = [p for p in pair_ids if "0006" in p][:1] + pair_ids[:3]
    want = list(dict.fromkeys(want))[:4]
    t = np.arange(-10, 30)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for ax, pid in zip(axes.ravel(), want):
        r, s = dat[f"real_{pid}"], dat[f"syn_{pid}"]
        ax.plot(t[:len(r)], r, "-", color="#1565c0", lw=1.6, label="real")
        ax.plot(t[:len(s)], s, "--", color="#e65100", lw=1.6, label="synthetic")
        ax.axvline(0, color="k", lw=0.8, ls=":"); ax.axvspan(0, 10, color="#eeeeee", zorder=0)
        ax.set_title(pid, fontsize=8); ax.set_xlabel("s from event")
    axes[0,0].legend(fontsize=7)
    fig.suptitle("Fig. 4 — Event-triggered baseline-corrected profiles, A0 Zigbee pairs", fontsize=9.5)
    save(fig, "fig4_q4_eta_profiles")
