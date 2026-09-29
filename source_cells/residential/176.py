# %% CELL EXT.A.2 — Task construction (TRAIN-defined labels only; TEST untouched for definitions)
from dataclasses import dataclass

WIN_HIST, WIN_FUT, STRIDE = 300, 60, 60  # seconds; 1 Hz rows

def make_windows(df, feat_cols, n_hist=WIN_HIST, n_fut=WIN_FUT, stride=STRIDE):
    X, spans = [], []
    arr = df[feat_cols].to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)  # inactive->0 for task features only (values remain governed upstream)
    for s in range(0, len(df) - n_hist - n_fut, stride):
        h = arr[s:s+n_hist]
        X.append(np.concatenate([h.mean(0), h.std(0), h[-60:].mean(0)]))
        spans.append((s, s+n_hist, s+n_hist+n_fut))
    return np.asarray(X), spans

def t1_labels(df, spans):
    d = df[[c for c in driver_cols if c in df.columns]].to_numpy(dtype=float)
    d = np.nan_to_num(d, nan=0.0)
    return np.array([(d[a:b] > 0).any() for (_, a, b) in spans], dtype=int)

def t2_threshold_from_train(df_train):
    r = df_train[[c for c in router_cols if c in df_train.columns]].sum(axis=1)
    return float(np.nanmedian(r))

T2_THRESH = t2_threshold_from_train(real_tr)   # TRAIN-only definition. Never recomputed on TEST.
print(f"[EXT.A.2] T2 router-activity median threshold (TRAIN-only): {T2_THRESH:.4f}")

def t2_labels(df, spans):
    r = df[[c for c in router_cols if c in df.columns]].sum(axis=1).to_numpy()
    return np.array([float(np.nanmean(r[a:b]) > T2_THRESH) for (_, a, b) in spans], dtype=int)

def t3_targets(df, spans, k_targets=10):
    # deterministic first-k continuous targets by name for reproducibility
    tgts = sorted([c for c in cont_cols if c in df.columns])[:k_targets]
    arr = df[tgts].to_numpy(dtype=float)
    y = np.array([np.nanmean(arr[a:b], axis=0) for (_, a, b) in spans])
    return np.nan_to_num(y, nan=0.0), tgts

print("[EXT.A.2] task builders ready (labels defined on TRAIN semantics only).")
