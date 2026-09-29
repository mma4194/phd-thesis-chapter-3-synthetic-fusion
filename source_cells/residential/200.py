# %% CELL EXT.FIG.3v3 — corrected branch × dimension readiness heatmap
_drv_pass = int((CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF["driver_corrected_ratio_battery_status"].astype(str) == "pass").sum())
_drv_warn = int((CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF["driver_corrected_ratio_battery_status"].astype(str) == "warning").sum())
_drv_fatal = int((CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF["driver_corrected_ratio_battery_status"].astype(str) == "fatal").sum())
_drv_rel = int(len(DRIVER_RELEASE_ELIGIBLE_COLS_CORRECTED))
grid_roles = ["continuous", "binary", "drivers", "masks", "router", "zigbee", "OTA"]
grid_dims = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
CELLS = {
    ("continuous", "Q1"): ("fatal", "71/29/48"), ("continuous", "Q2"): ("fatal", "71/29/48"), ("continuous", "Q3"): ("warning", "joint w/ masks"), ("continuous", "Q5"): ("scope_excluded", "excluded"), ("continuous", "Q6"): ("scope_excluded", "excluded"),
    ("binary", "Q1"): ("warning", "27/9/3"), ("binary", "Q2"): ("warning", "27/9/3"), ("binary", "Q6"): ("blocker", "3 blockers"),
    ("drivers", "Q1"): ("warning", f"{_drv_pass}/{_drv_warn}/{_drv_fatal}"), ("drivers", "Q2"): ("warning", "corrected"), ("drivers", "Q4"): ("warning", "dev subset"), ("drivers", "Q5"): ("pass", "T1 5/5 seeds"), ("drivers", "Q6"): ("warning", f"{_drv_rel} kept"),
    ("masks", "Q2"): ("warning", "run-length"), ("masks", "Q3"): ("fatal", "59/62/62"), ("masks", "Q6"): ("blocker", "6 blockers"),
    ("router", "Q1"): ("warning", "18 cols"), ("router", "Q2"): ("warning", ""), ("router", "Q4"): ("blocker", "generic"), ("router", "Q5"): ("warning", "T2 confounded"), ("router", "Q6"): ("warning", "restricted"),
    ("zigbee", "Q1"): ("warning", "4 cols"), ("zigbee", "Q2"): ("warning", ""), ("zigbee", "Q4"): ("blocker", "all-8 denied"), ("zigbee", "Q6"): ("warning", "released"),
    ("OTA", "Q1"): ("warning", "2 cols"), ("OTA", "Q2"): ("warning", ""), ("OTA", "Q6"): ("scope_excluded", "excluded"),
}
fig, ax = plt.subplots(figsize=(6.8, 4.4))
for i, r in enumerate(grid_roles):
    for j, d in enumerate(grid_dims):
        st, txt = CELLS.get((r, d), ("na", ""))
        ax.add_patch(Rectangle((j, len(grid_roles)-1-i), 1, 1, fc=STATUS_COLORS[st], ec="white", lw=1.5, alpha=0.85))
        if txt:
            ax.text(j+0.5, len(grid_roles)-1-i+0.5, txt, ha="center", va="center", fontsize=6.6, color="white" if st in ("fatal", "blocker", "pass") else "black")
ax.set_xlim(0, len(grid_dims)); ax.set_ylim(0, len(grid_roles))
ax.set_xticks(np.arange(len(grid_dims))+0.5); ax.set_xticklabels(grid_dims)
ax.set_yticks(np.arange(len(grid_roles))+0.5); ax.set_yticklabels(grid_roles[::-1])
ax.set_title("Branch × dimension readiness after corrected driver gating", fontsize=9.5, pad=10)
handles = [Rectangle((0,0),1,1,fc=STATUS_COLORS[k]) for k in ["pass","warning","fatal","blocker","scope_excluded","na"]]
ax.legend(handles, ["pass","warning","fatal","blocker","scope-excluded","n/a"], ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5,-0.06))
save(fig, "fig3_readiness_heatmap_corrected_driver")
