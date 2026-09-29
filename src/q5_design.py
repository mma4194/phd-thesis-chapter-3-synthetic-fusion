import numpy as np
import pandas as pd

def corrected_design(real, syn, W=60):
    """Use exact protocol prefixes, full-clock windows and disjoint fit/evaluation support.
    Keep real-prefix-assisted scaling explicit to preserve the earlier access regime.
    """
    if len(real) != len(syn):
        raise ValueError('Different real/synthetic lengths; no silent truncation.')
    t = 'sec_epoch_s__canon'
    if t not in real or t not in syn:
        raise ValueError('Canonical timestamp missing.')
    rt = pd.to_numeric(real[t], errors='raise').to_numpy(float)
    st = pd.to_numeric(syn[t], errors='raise').to_numpy(float)
    if not (np.isfinite(rt).all() and np.isfinite(st).all() and np.array_equal(rt, st) and np.all(np.diff(rt) == 1)):
        raise ValueError('Inputs must share the exact strictly increasing one-second grid.')
    cols = [c for c in syn.columns if c in real and c.startswith(('router__', 'zigbee__')) and pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(syn[c])]
    if not cols:
        raise ValueError('No protocol columns.')
    n = len(real)
    boundary = W * int(0.4 * (n // W))
    if boundary < 10 * W or n - boundary < 10 * W:
        raise ValueError('Too little data.')
    r = real[cols].replace([np.inf, -np.inf], np.nan).astype(float)
    s = syn[cols].replace([np.inf, -np.inf], np.nan).astype(float)
    train = r.iloc[:boundary]
    med = train.median()
    iqr = train.quantile(0.75) - train.quantile(0.25)
    usable = [c for c in cols if np.isfinite(med[c]) and np.isfinite(iqr[c]) and (iqr[c] > 0)]
    kind = 'IQR'
    scale = iqr
    if not usable:
        kind = 'SD'
        scale = train.std(ddof=1)
        usable = [c for c in cols if np.isfinite(med[c]) and np.isfinite(scale[c]) and (scale[c] > 0)]
    if not usable:
        raise ValueError('No finite positive real-prefix scale.')
    scope = pd.DataFrame({'column': cols, 'prefix_median': med, 'prefix_IQR': iqr, 'selected': pd.Series([c in usable for c in cols], index=cols), 'scale_kind': kind}).reset_index(drop=True)
    starts = np.arange(0, n - 2 * W + 1, W)
    fit = np.flatnonzero(starts + 2 * W <= boundary)
    ev = np.flatnonzero(starts >= boundary)
    if len(fit) < 10 or len(ev) < 10:
        raise ValueError('Too few supervised samples.')

    def build(df):
        z = ((df[usable] - med[usable]) / scale[usable]).replace([np.inf, -np.inf], np.nan)
        missing = z.isna().mean()
        arr = z.fillna(0).to_numpy()
        blocks = arr[:n // W * W].reshape(-1, W, len(usable))
        X = np.concatenate([blocks.mean(1), blocks.std(1), blocks.min(1), blocks.max(1), blocks[:, -1, :]], axis=1)[:-1]
        activity = np.abs(blocks).mean(axis=(1, 2))
        return (X, activity[1:], activity[:-1], missing)
    xr, yr, pr, mr = build(r)
    xs, ys, ps, ms = build(s)
    assert np.array_equal(starts[fit] + 2 * W <= boundary, np.ones(len(fit), bool))
    assert starts[fit].max() + 2 * W <= starts[ev].min()
    target_scale = float(np.std(yr[fit], ddof=0))
    if not np.isfinite(target_scale) or target_scale <= 0:
        target_scale = float(np.mean(np.abs(yr[fit])))
    if not np.isfinite(target_scale) or target_scale <= 0:
        target_scale = 1.0
    scope['real_missing_rate'] = scope.column.map(mr)
    scope['synthetic_missing_rate'] = scope.column.map(ms)
    meta = {'rows': n, 'window_rows': W, 'boundary_row_exclusive': boundary, 'fit_samples': len(fit), 'eval_samples': len(ev), 'purged_samples': len(starts) - len(fit) - len(ev), 'n_protocol_inputs': len(usable), 'model_features': xr.shape[1], 'target_scale': target_scale, 'timestamp_in_predictors_or_target': False, 'fresh_holdout': False, 'access_regime': 'real-prefix-assisted preprocessing and target definition', 'normalization_fit_rows': [0, boundary], 'first_evaluation_feature_row': int(starts[ev[0]]), 'last_training_target_end_exclusive': int(starts[fit[-1]] + 2 * W), 'claim_allowed': False}
    return (xr, yr, xs, ys, pr, fit, ev, starts, scope, meta)
