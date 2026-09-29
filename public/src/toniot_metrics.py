import numpy as np
import pandas as pd
import math
Q4_LAG_WINDOWS=5

def robust_z_from_train(x_train: pd.Series, x: pd.Series) -> np.ndarray:
    tr = pd.to_numeric(x_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    vals = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = float(tr.median(skipna=True)) if tr.notna().any() else 0.0
    q25 = float(tr.quantile(0.25)) if tr.notna().any() else 0.0
    q75 = float(tr.quantile(0.75)) if tr.notna().any() else 0.0
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 0:
        scale = float(tr.std(skipna=True)) if tr.notna().sum() > 1 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return ((vals.fillna(med) - med) / scale).to_numpy(dtype=float)

def event_profile(event_vec: np.ndarray, response_vec: np.ndarray, L: int = Q4_LAG_WINDOWS):
    event_vec = np.asarray(event_vec, dtype=float)
    response_vec = np.asarray(response_vec, dtype=float)
    event_idx = np.where(event_vec > 0)[0]
    usable = event_idx[(event_idx - L >= 0) & (event_idx + L < len(response_vec))]
    if len(usable) == 0:
        return None, 0
    mats = [response_vec[idx-L:idx+L+1] for idx in usable]
    return np.nanmean(np.vstack(mats), axis=0), int(len(usable))

def profile_metrics(ref_profile: np.ndarray, cmp_profile: np.ndarray, L: int = Q4_LAG_WINDOWS):
    if ref_profile is None or cmp_profile is None:
        return {"eta_similarity": np.nan, "lag_error_windows": np.nan, "response_window_rel_error": np.nan, "status": "blocker"}
    if np.nanstd(ref_profile) == 0 or np.nanstd(cmp_profile) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(ref_profile, cmp_profile)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    mae = float(np.nanmean(np.abs(ref_profile - cmp_profile)))
    scale = float(np.nanmean(np.abs(ref_profile)))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    nmae = mae / scale
    eta = 0.5 * max(0.0, corr) + 0.5 * math.exp(-nmae)
    post = slice(L, 2 * L + 1)
    try:
        lag_ref = int(np.nanargmax(ref_profile[post]))
        lag_cmp = int(np.nanargmax(cmp_profile[post]))
        lag_error = abs(lag_ref - lag_cmp)
    except Exception:
        lag_error = 999
    resp_ref = float(np.nanmean(ref_profile[post]))
    resp_cmp = float(np.nanmean(cmp_profile[post]))
    response_window_rel_error = abs(resp_ref - resp_cmp) / max(abs(resp_ref), 1e-9)
    if eta >= 0.70 and lag_error <= 2 and response_window_rel_error <= 0.25:
        status = "pass"
    elif eta >= 0.50 and lag_error <= 5 and response_window_rel_error <= 0.50:
        status = "warning"
    else:
        status = "blocker"
    return {
        "eta_similarity": float(eta),
        "lag_error_windows": int(lag_error),
        "response_window_rel_error": float(response_window_rel_error),
        "status": status,
    }
