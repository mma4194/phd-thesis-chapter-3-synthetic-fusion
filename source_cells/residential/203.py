# %% CELL EXT.FIG.6v3 — release funnel rebuilt from actual corrected Q6 registry
registry = CELL15_6_Q6_PUBLIC_COLUMN_REGISTRY_DF.copy()
dropped = CELL15_6_Q6_PUBLIC_DROPPED_COLUMNS_DF.copy()
role_col = "role_group" if "role_group" in registry.columns else "role"
def _count_reg(predicate): return int(predicate(registry).sum()) if len(registry) else 0
def _count_drop(predicate): return int(predicate(dropped).sum()) if len(dropped) else 0
kept_time = _count_reg(lambda d: d.get("is_time", pd.Series(False, index=d.index)).fillna(False).astype(bool))
kept_router = _count_reg(lambda d: d.get("protocol_tier", pd.Series("", index=d.index)).astype(str).eq("router"))
kept_zigbee = _count_reg(lambda d: d.get("protocol_tier", pd.Series("", index=d.index)).astype(str).eq("zigbee"))
kept_drivers = _count_reg(lambda d: d[role_col].astype(str).eq("iot_event_drivers"))
drop_cont = _count_drop(lambda d: d.get("role_group", pd.Series("", index=d.index)).astype(str).str.contains("continuous", case=False, na=False))
drop_bin = _count_drop(lambda d: d.get("role_group", pd.Series("", index=d.index)).astype(str).str.contains("binary", case=False, na=False))
drop_masks = _count_drop(lambda d: d.get("role_group", pd.Series("", index=d.index)).astype(str).str.contains("observability|mask", case=False, na=False))
drop_ota = _count_drop(lambda d: d.get("protocol_tier", pd.Series("", index=d.index)).astype(str).eq("ota"))
drop_driver_fatal = _count_drop(lambda d: d.get("drop_reason", pd.Series("", index=d.index)).astype(str).eq("corrected_sparse_driver_fatal_release_excluded"))
known_dropped = drop_cont + drop_bin + drop_masks + drop_ota + drop_driver_fatal
drop_other = max(0, int(len(dropped)) - known_dropped)
groups = [("Continuous values", drop_cont, "#78909c", False), ("Binary/native state columns", drop_bin, "#8d6e63", False), ("Observability masks", drop_masks, "#a1887f", False), ("OTA protocol", drop_ota, "#ce93d8", False), ("Corrected-fatal sparse drivers", drop_driver_fatal, "#ef9a9a", False), ("Other excluded/placeholders", drop_other, "#bdbdbd", False), ("Router protocol", kept_router, "#66bb6a", True), ("Zigbee protocol", kept_zigbee, "#43a047", True), ("Sparse drivers", kept_drivers, "#2e7d32", True), ("Timestamp", kept_time, "#1b5e20", True)]
groups = [(n, int(c), colr, k) for n, c, colr, k in groups if int(c) > 0]
total = int(len(registry) + len(dropped)); kept = int(len(registry))
assert total == int(CELL15_6_Q6_PUBLIC_MITIGATION_CONTRACT["column_counts"]["source_cols"]), f"figure total mismatch: {total}"
assert kept == int(CELL15_6_Q6_PUBLIC_MITIGATION_CONTRACT["column_counts"]["kept_cols"]), f"figure kept mismatch: {kept}"
fig, ax = plt.subplots(figsize=(7.4, 4.8)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(0.02, 0.97, f"Full scientific artifact — {total} logical columns", fontsize=9, weight="bold")
ax.text(0.66, 0.97, f"Q6 public baseline — {kept}", fontsize=9, weight="bold")
y = 0.92; MINH = 0.030
hs = [max(MINH, 0.80*c/max(total,1)) for _, c, _, _ in groups]
hs = [h*0.80/sum(hs) for h in hs]
for (name, count, colr, keep), h in zip(groups, hs):
    ax.add_patch(Rectangle((0.02, y-h), 0.26, h*0.92, fc=colr, ec="white"))
    ax.text(0.29, y-h/2, f"{name} ({count})", va="center", fontsize=7.2)
    if keep:
        ax.annotate("", xy=(0.66, y-h/2), xytext=(0.50, y-h/2), arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.8))
        ax.add_patch(Rectangle((0.66, y-h), 0.20, h*0.92, fc=colr, ec="white"))
        ax.text(0.87, y-h/2, f"kept ({count})", va="center", fontsize=7.2, color=colr)
    else:
        ax.text(0.52, y-h/2, "dropped", va="center", fontsize=6.8, color="#757575", style="italic")
    y -= h + 0.010
ax.set_title(f"Q6 release governance: {total-kept} columns dropped, {kept} released (warning-governed)", fontsize=9.5, pad=6)
save(fig, "fig6_release_funnel_corrected_public_scope")
print(f"[EXT.FIG.6v3] totals reconciled: {total} = {total-kept} dropped + {kept} kept")
