# %% CELL EXT.FIG.2v3 — corrected sparse-driver evidence-to-claim flow
fig, ax = plt.subplots(figsize=(7.2, 4.8)); ax.axis("off"); ax.set_ylim(-0.15, 1.15)
stages = ["Metric\nevidence", "Per-dimension\nstatus", "Blocker\nattribution", "Claim\npermission", "Release\ndecision"]
xs = np.linspace(0.08, 0.92, len(stages))
for x, stage_label in zip(xs, stages):
    ax.add_patch(Rectangle((x-0.075, 0.38), 0.15, 0.26, fc="#f5f5f5", ec="#616161", lw=1))
    ax.text(x, 0.51, stage_label, ha="center", va="center", fontsize=8.5)
_drv_pass = int((CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF["driver_corrected_ratio_battery_status"].astype(str) == "pass").sum())
_drv_warn = int((CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF["driver_corrected_ratio_battery_status"].astype(str) == "warning").sum())
_drv_fatal = int((CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF["driver_corrected_ratio_battery_status"].astype(str) == "fatal").sum())
_drv_rel = int(len(DRIVER_RELEASE_ELIGIBLE_COLS_CORRECTED))
threads = [
    (f"Sparse drivers ({_drv_pass+_drv_warn+_drv_fatal})", "#2e7d32", [f"{_drv_pass} pass / {_drv_warn} warn / {_drv_fatal} fatal", "corrected\nratio-scale", f"{_drv_fatal} excluded", f"{_drv_rel} permitted\nwith warnings", "release subset"], 0.86),
    ("A0 all-8 Zigbee Q4", "#c62828", ["ETA 0.652 mean", "1 fatal / 6 warn", "1 terminal blocker", "DENIED", "no promotion"], 0.14),
]
for label, color, notes, y in threads:
    ax.plot(xs, [y]*len(xs), color=color, lw=2.2, marker="o", ms=5, zorder=3)
    ax.text(xs[0]-0.05, y, label, ha="right", va="center", fontsize=8.5, color=color, weight="bold")
    for x, note in zip(xs, notes):
        ax.text(x, y - 0.09, note, ha="center", fontsize=7.2, color=color)
ax.plot([xs[2]], [0.14], marker="x", ms=14, mew=3, color="#c62828", zorder=4)
ax.set_title("Evidence-to-claim flow after corrected sparse-driver release gating", fontsize=9.5, pad=14)
save(fig, "fig2_evidence_to_claim_funnel_corrected_driver")
