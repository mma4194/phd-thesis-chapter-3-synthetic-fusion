# %% CELL EXT.DRV.UNION — Diagnose the 2.4x window-union inflation: rates vs clustering
dcols = [c for c in driver_cols if c in real_te.columns and c in syn_pub.columns]
R = (np.nan_to_num(real_te[dcols].to_numpy(float)) > 0)
S = (np.nan_to_num(syn_pub[dcols].to_numpy(float)) > 0)
print(f"{len(dcols)} drivers | per-driver MARGINAL event-rate ratio (syn/real), summary:")
r_rate, s_rate = R.mean(0), S.mean(0)
ratio = np.where(r_rate > 0, s_rate / np.maximum(r_rate, 1e-12), np.nan)
print(f"  median {np.nanmedian(ratio):.2f}  IQR [{np.nanpercentile(ratio,25):.2f}, {np.nanpercentile(ratio,75):.2f}]")
W = 60
def union_rate(M):
    n = (len(M)//W)*W
    return M[:n].reshape(-1, W, M.shape[1]).any(axis=(1,2)).mean()
print(f"window-UNION any-driver rate: real {union_rate(R):.3f} | syn {union_rate(S):.3f}")
def cluster_stats(M):
    ev = M.any(axis=1).astype(int)                       # any-driver per second
    n = (len(ev)//W)*W
    per_win = ev[:n].reshape(-1, W).sum(1)
    return ev.mean(), per_win[per_win > 0].mean(), (per_win > 0).mean()
for name, M in [("real", R), ("syn", S)]:
    sec_rate, ev_per_active_win, win_rate = cluster_stats(M)
    print(f"{name}: any-driver sec-rate {sec_rate:.4f} | events per ACTIVE window {ev_per_active_win:.1f} | window rate {win_rate:.3f}")
# Reading rule (declared now): if per-driver ratios ~1 and syn events-per-active-window << real,
# the mechanism is DISPERSION (lost clustering/co-occurrence), not rate inflation.