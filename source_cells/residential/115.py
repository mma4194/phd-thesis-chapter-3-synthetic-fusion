# %% CELL 14.6R0 — TRAIN/VAL-only A0 policy grid search and freeze
# Purpose:
#   Select one global A0 Zigbee-safe materializer policy using TRAIN/VAL only.
#
# Scientific contract:
#   - TRAIN is used to estimate event-response profiles and support bounds.
#   - VAL is used to score/freeze one global policy.
#   - TEST real values are not read, scored, or used here.
#   - No synthetic TEST values are materialized or mutated here.
#   - The selected policy only sets CFG keys consumed by Cell 14.6.
#
# Outputs:
#   reports/cell14_6R0_A0_trainval_policy_grid.csv
#   reports/cell14_6R0_A0_trainval_policy_summary.json
#   artifacts/contracts/cell14_6R0_A0_trainval_policy_contract_v1_0_THESIS.json

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log("--- START: Cell 14.6R0 - TRAIN/VAL-only A0 policy grid search and freeze ---")

_required_146r0 = [
    "CFG", "log", "df_tr", "df_val", "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_DF",
]
_missing_146r0 = [k for k in _required_146r0 if k not in globals()]
if _missing_146r0:
    raise RuntimeError(f"[Cell14.6R0] Missing required globals: {_missing_146r0}")

OUTDIR_R0 = Path(str(OUTDIR)).expanduser().resolve()
OUT_SYN_R0 = Path(str(OUT_SYN)).expanduser().resolve()
REPORT_DIR_R0 = Path(str(REPORT_DIR)).expanduser().resolve()
ARTDIR_R0 = OUTDIR_R0 / "artifacts"
CONTRACT_DIR_R0 = ARTDIR_R0 / "contracts"
REPORT_DIR_R0.mkdir(parents=True, exist_ok=True)
ARTDIR_R0.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_R0.mkdir(parents=True, exist_ok=True)

CELL146R0_VERSION = "cell14_6R0_A0_trainval_policy_freeze_v1_0_THESIS"

def _sanitize_146r0(obj):
    if isinstance(obj, dict):
        return {str(k): _sanitize_146r0(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_146r0(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _sanitize_146r0(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _sanitize_146r0(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _sanitize_146r0(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_146r0(path, payload):
    Path(path).write_text(json.dumps(_sanitize_146r0(payload), indent=2, sort_keys=True), encoding="utf-8")

def _sha256_file_146r0(path):
    path = Path(str(path))
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _to_num_146r0(frame, col, fill=np.nan):
    x = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    if fill is not np.nan:
        x[~np.isfinite(x)] = float(fill)
    return x

def _event_starts_146r0(x):
    x = np.asarray(x, dtype=np.float64)
    active = np.isfinite(x) & (x > 0.5)
    if active.size == 0:
        return np.array([], dtype=np.int64)
    prev = np.r_[False, active[:-1]]
    return np.flatnonzero(active & ~prev).astype(np.int64)

def _edge_safe_events_146r0(events, n, lag_lo, lag_hi):
    events = np.asarray(events, dtype=np.int64)
    if events.size == 0:
        return events
    return events[(events + int(lag_lo) >= 0) & (events + int(lag_hi) < int(n))]

def _window_matrix_146r0(y, events, lag_lo, lag_hi):
    lags = np.arange(int(lag_lo), int(lag_hi) + 1, dtype=np.int64)
    events = _edge_safe_events_146r0(events, len(y), lag_lo, lag_hi)
    if events.size == 0:
        return None
    rows = []
    for e in events:
        seg = y[e + int(lag_lo):e + int(lag_hi) + 1]
        if len(seg) == len(lags):
            rows.append(seg)
    if not rows:
        return None
    return np.vstack(rows).astype(np.float64)

def _baseline_correct_146r0(profile, lags):
    profile = np.asarray(profile, dtype=np.float64)
    lags = np.asarray(lags, dtype=np.int64)
    pre = lags < 0
    if pre.any() and np.isfinite(profile[pre]).any():
        base = float(np.nanmean(profile[pre]))
    else:
        base = float(np.nanmedian(profile[np.isfinite(profile)])) if np.isfinite(profile).any() else 0.0
    return profile - base

def _support_bounds_146r0(y):
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.nan, np.nan
    return float(np.nanpercentile(y, 0.1)), float(np.nanpercentile(y, 99.9))

def _transform_146r0(x):
    arr = np.asarray(x, dtype=np.float64)
    mode = str(CFG.get("cell14_6_protocol_transform", "log1p_abs_signed")).lower()
    if mode == "identity":
        out = arr.copy()
    elif mode == "log1p":
        out = np.log1p(np.maximum(arr, 0.0))
    elif mode == "log1p_abs":
        out = np.log1p(np.abs(arr))
    else:
        out = np.sign(arr) * np.log1p(np.abs(arr))
    out[~np.isfinite(out)] = np.nan
    return out

def _profile_qa_146r0(y, events, lag_lo, lag_hi):
    mat = _window_matrix_146r0(_transform_146r0(y), events, lag_lo, lag_hi)
    lags = np.arange(int(lag_lo), int(lag_hi) + 1, dtype=np.int64)
    if mat is None:
        return lags, np.full(len(lags), np.nan), 0
    with np.errstate(invalid="ignore"):
        prof = np.nanmean(mat, axis=0)
    pre = lags < 0
    if pre.any() and np.isfinite(prof[pre]).any():
        prof = prof - float(np.nanmean(prof[pre]))
    return lags, prof, int(mat.shape[0])

def _eta_similarity_146r0(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.nanstd(aa) < 1e-12 or np.nanstd(bb) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(aa, bb)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    scale = float(abs(np.nanmean(aa)) + np.nanstd(aa) + 1e-9)
    mae_norm = float(np.nanmean(np.abs(aa - bb)) / scale)
    return float(0.5 * max(0.0, corr) + 0.5 * np.exp(-mae_norm))

def _lag_peak_146r0(lags, prof):
    prof = np.asarray(prof, dtype=np.float64)
    if not np.isfinite(prof).any():
        return np.nan
    idx = int(np.nanargmax(np.abs(prof)))
    return int(np.asarray(lags, dtype=np.int64)[idx])

def _response_threshold_146r0(y):
    z = np.abs(_transform_146r0(y))
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.nan
    q = float(np.clip(CFG.get("cell14_6_response_threshold_quantile", 0.90), 0.50, 0.999))
    return float(np.nanquantile(z, q))

def _response_rate_146r0(y, events, lag_lo, lag_hi, threshold):
    events = _edge_safe_events_146r0(events, len(y), lag_lo, lag_hi)
    if events.size == 0 or not np.isfinite(threshold):
        return np.nan
    z = _transform_146r0(y)
    hits = []
    for e in events:
        seg = z[e + int(lag_lo):e + int(lag_hi) + 1]
        seg = seg[np.isfinite(seg)]
        hits.append(bool(seg.size and np.nanmax(np.abs(seg)) >= threshold))
    return float(np.mean(hits)) if hits else np.nan

def _status_146r0(row):
    real_events = int(row.get("real_event_count_window_valid", 0) or 0)
    syn_events = int(row.get("syn_event_count_window_valid", 0) or 0)
    min_events = int(CFG.get("cell14_6_min_events_for_pass", 5))
    if real_events < min_events and syn_events < min_events:
        return "not_evaluable", "insufficient_real_and_synthetic_events"
    if real_events < min_events:
        return "warning", "insufficient_real_events"
    if syn_events < min_events:
        return "warning", "insufficient_synthetic_events"
    fatal = []
    warn = []
    eta = float(row.get("ETA_similarity", np.nan))
    prof = float(row.get("manifest_window_profile_similarity", np.nan))
    lagerr = float(row.get("lag_peak_error", np.nan))
    resperr = float(row.get("response_window_rate_error", np.nan))
    if np.isfinite(eta):
        if eta < float(CFG.get("cell14_6_eta_similarity_warning", 0.50)):
            fatal.append("ETA_similarity_fatal")
        elif eta < float(CFG.get("cell14_6_eta_similarity_pass", 0.70)):
            warn.append("ETA_similarity_warning")
    else:
        warn.append("ETA_similarity_not_finite")
    if np.isfinite(prof):
        if prof < float(CFG.get("cell14_6_profile_similarity_warning", 0.50)):
            fatal.append("manifest_profile_similarity_fatal")
        elif prof < float(CFG.get("cell14_6_profile_similarity_pass", 0.70)):
            warn.append("manifest_profile_similarity_warning")
    else:
        warn.append("manifest_profile_similarity_not_finite")
    if np.isfinite(lagerr):
        if lagerr > float(CFG.get("cell14_6_lag_peak_error_warning", 5)):
            fatal.append("lag_peak_error_fatal")
        elif lagerr > float(CFG.get("cell14_6_lag_peak_error_pass", 2)):
            warn.append("lag_peak_error_warning")
    if np.isfinite(resperr):
        if resperr > float(CFG.get("cell14_6_response_window_rate_error_warning", 0.25)):
            fatal.append("response_window_rate_error_fatal")
        elif resperr > float(CFG.get("cell14_6_response_window_rate_error_pass", 0.10)):
            warn.append("response_window_rate_error_warning")
    if fatal:
        return "fatal", "|".join(fatal)
    if warn:
        return "warning", "|".join(warn)
    return "pass", "A0_VAL_pair_pass"

a0 = CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_DF.copy()
a0.columns = a0.columns.astype(str)
required_cols = ["repair_candidate_id", "anchor_col", "protocol_col", "protocol_tier", "recommended_lag_lo", "recommended_lag_hi"]
missing_cols = [c for c in required_cols if c not in a0.columns]
if missing_cols:
    raise RuntimeError(f"[Cell14.6R0] A0 manifest missing columns: {missing_cols}")
a0["protocol_tier"] = a0["protocol_tier"].astype(str).str.lower()
if "repair_eligible" in a0.columns:
    a0 = a0[a0["repair_eligible"].fillna(False).astype(bool)].copy()
if "direct_repair_allowed_now" in a0.columns:
    a0 = a0[a0["direct_repair_allowed_now"].fillna(False).astype(bool)].copy()
allowed_cols = set(map(str, CFG.get("cell14_6_allowed_protocol_cols", ["zigbee__pkt_total", "zigbee__bytes_total"])))
a0 = a0[a0["protocol_tier"].eq("zigbee") & a0["protocol_col"].astype(str).isin(allowed_cols)].copy()
if len(a0) == 0:
    raise RuntimeError("[Cell14.6R0] No A0 Zigbee-safe rows remain after policy filter.")

a1_val_path = OUTDIR_R0 / "portfolio_candidates" / "A1_VAL_baseline.parquet"
if not a1_val_path.exists():
    raise RuntimeError(f"[Cell14.6R0] Missing VAL protocol baseline: {a1_val_path}")
protocol_base_val = pd.read_parquet(a1_val_path)
protocol_base_val.columns = protocol_base_val.columns.astype(str)
protocol_base_val.index = df_val.index

N_TR_R0 = int(len(df_tr))
N_VAL_R0 = int(len(df_val))

for c in sorted(set(a0["anchor_col"].astype(str))):
    if c not in df_tr.columns or c not in df_val.columns:
        raise RuntimeError(f"[Cell14.6R0] Missing anchor column in TRAIN/VAL: {c}")
for c in sorted(set(a0["protocol_col"].astype(str))):
    if c not in df_tr.columns or c not in df_val.columns or c not in protocol_base_val.columns:
        raise RuntimeError(f"[Cell14.6R0] Missing protocol column in TRAIN/VAL or VAL baseline: {c}")

# Candidate policies are predeclared here and scored only on VAL.
grid = []
for profile_source in list(CFG.get("cell14_6R0_profile_sources", ["mean_profile_bc", "median_profile_bc"])):
    for pre_margin in list(CFG.get("cell14_6R0_profile_pre_margins", [0, 5, 10])):
        for post_margin in list(CFG.get("cell14_6R0_profile_post_margins", [0, 5, 10])):
            for alpha in list(CFG.get("cell14_6R0_base_alpha_scales", [0.40, 0.55, 0.70, 0.85])):
                for clip_margin in list(CFG.get("cell14_6R0_clip_margin_fractions", [0.00, 0.02, 0.05, 0.10])):
                    grid.append({
                        "profile_source": str(profile_source),
                        "profile_pre_margin": int(pre_margin),
                        "profile_post_margin": int(post_margin),
                        "base_alpha_scale": float(alpha),
                        "clip_margin_fraction": float(clip_margin),
                    })

policy_rows = []
pair_rows_all = []

for pidx, pol in enumerate(grid):
    profiles = []
    for _, r in a0.iterrows():
        anchor_col = str(r["anchor_col"])
        protocol_col = str(r["protocol_col"])
        lag_lo = int(r["recommended_lag_lo"])
        lag_hi = int(r["recommended_lag_hi"])
        if lag_lo > lag_hi:
            lag_lo, lag_hi = lag_hi, lag_lo
        profile_lag_lo = lag_lo - int(pol["profile_pre_margin"])
        profile_lag_hi = lag_hi + int(pol["profile_post_margin"])
        lags = np.arange(profile_lag_lo, profile_lag_hi + 1, dtype=np.int64)
        x_tr = _to_num_146r0(df_tr, anchor_col, fill=0.0)
        y_tr = _to_num_146r0(df_tr, protocol_col)
        ev_tr = _edge_safe_events_146r0(_event_starts_146r0(x_tr), N_TR_R0, profile_lag_lo, profile_lag_hi)
        mat_tr = _window_matrix_146r0(y_tr, ev_tr, profile_lag_lo, profile_lag_hi)
        if mat_tr is None:
            continue
        with np.errstate(invalid="ignore"):
            mean_bc = _baseline_correct_146r0(np.nanmean(mat_tr, axis=0), lags)
            median_bc = _baseline_correct_146r0(np.nanmedian(mat_tr, axis=0), lags)
        lo, hi = _support_bounds_146r0(y_tr)
        profiles.append({
            "repair_candidate_id": str(r["repair_candidate_id"]),
            "anchor_col": anchor_col,
            "protocol_col": protocol_col,
            "lag_lo": lag_lo,
            "lag_hi": lag_hi,
            "profile_lag_lo": profile_lag_lo,
            "profile_lag_hi": profile_lag_hi,
            "lags": lags,
            "mean_profile_bc": mean_bc,
            "median_profile_bc": median_bc,
            "support_lo": lo,
            "support_hi": hi,
            "recommended_weight": float(r.get("recommended_weight", 1.0) if pd.notna(r.get("recommended_weight", 1.0)) else 1.0),
            "train_events": int(len(ev_tr)),
        })

    protocol_val = protocol_base_val.copy()
    target_cols = sorted(set([pr["protocol_col"] for pr in profiles]))
    buffers = {
        c: {"sum": np.zeros(N_VAL_R0, dtype=np.float64), "wsum": np.zeros(N_VAL_R0, dtype=np.float64)}
        for c in target_cols
    }

    for pr in profiles:
        anchor_col = pr["anchor_col"]
        protocol_col = pr["protocol_col"]
        prof = np.asarray(pr.get(pol["profile_source"], pr["mean_profile_bc"]), dtype=np.float64)
        x_val = _to_num_146r0(df_val, anchor_col, fill=0.0)
        events = _edge_safe_events_146r0(_event_starts_146r0(x_val), N_VAL_R0, pr["profile_lag_lo"], pr["profile_lag_hi"])
        if events.size == 0:
            continue
        base = _to_num_146r0(protocol_val, protocol_col)
        support_lo = float(pr["support_lo"])
        support_hi = float(pr["support_hi"])
        if np.isfinite(support_lo) and np.isfinite(support_hi):
            margin = float(pol["clip_margin_fraction"]) * max(1.0, support_hi - support_lo)
            clip_lo = max(0.0, support_lo - margin)
            clip_hi = support_hi + margin
        else:
            clip_lo, clip_hi = 0.0, np.inf
        rec_w = float(np.clip(pr.get("recommended_weight", 1.0), 0.0, 1.0))
        alpha = float(np.clip(float(pol["base_alpha_scale"]) * rec_w, float(CFG.get("cell14_6_min_alpha", 0.05)), float(CFG.get("cell14_6_max_alpha", 0.85))))
        for e in events:
            lo = int(e) + pr["profile_lag_lo"]
            hi = int(e) + pr["profile_lag_hi"] + 1
            if lo < 0 or hi > N_VAL_R0:
                continue
            idx = np.arange(lo, hi, dtype=np.int64)
            local_base = base[idx]
            valid = np.isfinite(local_base) & np.isfinite(prof)
            if not valid.any():
                continue
            target = local_base.copy()
            target[valid] = local_base[valid] + prof[valid]
            target[valid] = (1.0 - alpha) * local_base[valid] + alpha * target[valid]
            target[valid] = np.clip(target[valid], clip_lo, clip_hi)
            target[valid] = np.maximum(target[valid], 0.0)
            # Rectangular weights are sufficient for policy search; terminal Cell 14.6 uses the final materializer.
            w = valid.astype(np.float64) * alpha
            buffers[protocol_col]["sum"][idx] += w * target
            buffers[protocol_col]["wsum"][idx] += w
    for c, buf in buffers.items():
        base = _to_num_146r0(protocol_val, c)
        out = base.copy()
        active = (buf["wsum"] > 0) & np.isfinite(base)
        out[active] = buf["sum"][active] / np.maximum(buf["wsum"][active], 1e-12)
        out = np.maximum(out, 0.0)
        out = np.rint(out)
        protocol_val[c] = out.astype(np.float32)

    q4_rows = []
    for pr in profiles:
        anchor_col = pr["anchor_col"]
        protocol_col = pr["protocol_col"]
        lag_lo = int(pr["lag_lo"])
        lag_hi = int(pr["lag_hi"])
        events = _event_starts_146r0(_to_num_146r0(df_val, anchor_col, fill=0.0))
        y_real = _to_num_146r0(df_val, protocol_col)
        y_syn = _to_num_146r0(protocol_val, protocol_col)
        wide_lo = -int(CFG.get("cell14_6_eta_pre_seconds", 10))
        wide_hi = int(CFG.get("cell14_6_eta_post_seconds", 30))
        l_real, prof_real, n_real_wide = _profile_qa_146r0(y_real, events, wide_lo, wide_hi)
        l_syn, prof_syn, n_syn_wide = _profile_qa_146r0(y_syn, events, wide_lo, wide_hi)
        if not np.array_equal(l_real, l_syn):
            eta = np.nan
        else:
            eta = _eta_similarity_146r0(prof_real, prof_syn)
        l_mr, prof_mr, n_real_manifest = _profile_qa_146r0(y_real, events, lag_lo, lag_hi)
        l_ms, prof_ms, n_syn_manifest = _profile_qa_146r0(y_syn, events, lag_lo, lag_hi)
        prof_sim = _eta_similarity_146r0(prof_mr, prof_ms) if np.array_equal(l_mr, l_ms) else np.nan
        lag_real = _lag_peak_146r0(l_mr, prof_mr)
        lag_syn = _lag_peak_146r0(l_ms, prof_ms)
        lag_err = abs(float(lag_real) - float(lag_syn)) if np.isfinite(lag_real) and np.isfinite(lag_syn) else np.nan
        thr = _response_threshold_146r0(y_real)
        rr = _response_rate_146r0(y_real, events, lag_lo, lag_hi, thr)
        sr = _response_rate_146r0(y_syn, events, lag_lo, lag_hi, thr)
        resp_err = abs(rr - sr) if np.isfinite(rr) and np.isfinite(sr) else np.nan
        row = {
            "policy_id": f"A0R0_POLICY_{pidx:04d}",
            **pol,
            "repair_candidate_id": pr["repair_candidate_id"],
            "anchor_col": anchor_col,
            "protocol_col": protocol_col,
            "real_event_count_window_valid": int(n_real_manifest),
            "syn_event_count_window_valid": int(n_syn_manifest),
            "ETA_similarity": eta,
            "manifest_window_profile_similarity": prof_sim,
            "lag_peak_error": lag_err,
            "response_window_rate_error": resp_err,
            "real_response_window_rate": rr,
            "syn_response_window_rate": sr,
        }
        status, reasons = _status_146r0(row)
        row["q4_status"] = status
        row["q4_reasons"] = reasons
        row["publication_blocker"] = bool(status == "fatal")
        q4_rows.append(row)

    qdf = pd.DataFrame(q4_rows)
    counts = qdf["q4_status"].value_counts().to_dict() if len(qdf) else {}
    policy_row = {
        "policy_id": f"A0R0_POLICY_{pidx:04d}",
        **pol,
        "pairs_total": int(len(qdf)),
        "VAL_pass_n": int(counts.get("pass", 0)),
        "VAL_warning_n": int(counts.get("warning", 0)),
        "VAL_fatal_n": int(counts.get("fatal", 0)),
        "VAL_not_evaluable_n": int(counts.get("not_evaluable", 0)),
        "VAL_publication_blocker_n": int(qdf["publication_blocker"].sum()) if len(qdf) else 999,
        "VAL_ETA_similarity_mean": float(pd.to_numeric(qdf.get("ETA_similarity", pd.Series(dtype=float)), errors="coerce").mean()) if len(qdf) else np.nan,
        "VAL_manifest_profile_similarity_mean": float(pd.to_numeric(qdf.get("manifest_window_profile_similarity", pd.Series(dtype=float)), errors="coerce").mean()) if len(qdf) else np.nan,
        "VAL_lag_peak_error_mean": float(pd.to_numeric(qdf.get("lag_peak_error", pd.Series(dtype=float)), errors="coerce").mean()) if len(qdf) else np.nan,
        "VAL_response_window_rate_error_mean": float(pd.to_numeric(qdf.get("response_window_rate_error", pd.Series(dtype=float)), errors="coerce").mean()) if len(qdf) else np.nan,
    }
    policy_rows.append(policy_row)
    pair_rows_all.extend(q4_rows)

grid_df = pd.DataFrame(policy_rows)
pair_df = pd.DataFrame(pair_rows_all)
if grid_df.empty:
    raise RuntimeError("[Cell14.6R0] Policy grid produced no evaluable rows.")

grid_df = grid_df.sort_values(
    [
        "VAL_publication_blocker_n",
        "VAL_fatal_n",
        "VAL_warning_n",
        "VAL_response_window_rate_error_mean",
        "VAL_lag_peak_error_mean",
        "VAL_ETA_similarity_mean",
    ],
    ascending=[True, True, True, True, True, False],
).reset_index(drop=True)

best = grid_df.iloc[0].to_dict()
zero_blocker = bool(int(best.get("VAL_publication_blocker_n", 999)) == 0 and int(best.get("VAL_fatal_n", 999)) == 0)

# Freeze the selected global policy into Cell 14.6 config.
CFG["cell14_6_profile_source"] = str(best["profile_source"])
CFG["cell14_6_profile_pre_margin"] = int(best["profile_pre_margin"])
CFG["cell14_6_profile_post_margin"] = int(best["profile_post_margin"])
CFG["cell14_6_base_alpha_scale"] = float(best["base_alpha_scale"])
CFG["cell14_6_clip_margin_fraction"] = float(best["clip_margin_fraction"])
CFG["cell14_6R0_policy_id"] = str(best["policy_id"])
CFG["cell14_6R0_selected_on_VAL"] = True
CFG["cell14_6R0_val_zero_blocker_policy"] = bool(zero_blocker)
CFG["cell14_6R_fresh_holdout_for_publication"] = bool(CFG.get("cell14_6R_fresh_holdout_for_publication", False))

grid_csv = REPORT_DIR_R0 / "cell14_6R0_A0_trainval_policy_grid.csv"
pair_csv = REPORT_DIR_R0 / "cell14_6R0_A0_trainval_pair_scores.csv"
summary_json = REPORT_DIR_R0 / "cell14_6R0_A0_trainval_policy_summary.json"
contract_json = CONTRACT_DIR_R0 / "cell14_6R0_A0_trainval_policy_contract_v1_0_THESIS.json"
grid_df.to_csv(grid_csv, index=False)
pair_df.to_csv(pair_csv, index=False)

summary_r0 = {
    "cell": "14.6R0",
    "version": CELL146R0_VERSION,
    "selected_policy": best,
    "selected_policy_has_zero_VAL_blockers": bool(zero_blocker),
    "VAL_selection_basis": "global_policy_grid_scored_on_VAL_only",
    "TEST_values_used": False,
    "TEST_used_for_membership_or_policy": False,
    "fresh_holdout_declared_for_publication": bool(CFG.get("cell14_6R_fresh_holdout_for_publication", False)),
}
_write_json_146r0(summary_json, summary_r0)

contract_r0 = {
    "cell": "14.6R0",
    "version": CELL146R0_VERSION,
    "role": "TRAIN_VAL_only_A0_policy_selection_and_freeze",
    "summary": summary_r0,
    "strict_contract": {
        "TRAIN_used_for_profile_estimation": True,
        "VAL_used_for_policy_selection": True,
        "TEST_real_values_used": False,
        "TEST_synthetic_values_used": False,
        "selected_on_VAL": True,
        "per_pair_TEST_pruning_allowed": False,
        "policy_applies_globally_to_all_A0_pairs": True,
        "synthetic_TEST_values_mutated_here": False,
        "materialization_done_here": False,
        "candidate_acceptance_done_here": False,
    },
    "outputs": {
        "grid_csv": str(grid_csv),
        "pair_scores_csv": str(pair_csv),
        "summary_json": str(summary_json),
        "contract_json": str(contract_json),
    },
}
contract_r0["hashes"] = {
    "grid_csv_sha256": _sha256_file_146r0(grid_csv),
    "pair_scores_csv_sha256": _sha256_file_146r0(pair_csv),
    "summary_json_sha256": _sha256_file_146r0(summary_json),
}
_write_json_146r0(contract_json, contract_r0)
contract_r0["hashes"]["contract_json_sha256"] = _sha256_file_146r0(contract_json)
_write_json_146r0(contract_json, contract_r0)

globals()["CELL14_6R0_A0_POLICY_GRID_DF"] = grid_df
globals()["CELL14_6R0_A0_PAIR_SCORES_DF"] = pair_df
globals()["CELL14_6R0_A0_POLICY_CONTRACT"] = contract_r0
globals()["CELL14_6R0_A0_POLICY_CONTRACT_JSON"] = str(contract_json)

print("=== CELL 14.6R0 SELECTED TRAIN/VAL A0 POLICY ===")
print(json.dumps(_sanitize_146r0(summary_r0), indent=2, sort_keys=True))
if not zero_blocker:
    print("[Cell14.6R0] WARNING: no zero-blocker VAL policy found. Cell 14.6 will still run terminal TEST for evidence, but no positive A0 claim should be made.")
log("--- END: Cell 14.6R0 - TRAIN/VAL-only A0 policy grid search and freeze ---")
