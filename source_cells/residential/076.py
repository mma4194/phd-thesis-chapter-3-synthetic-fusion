# ==========================================================
# CELL 12.e.6R - Corrected sparse-driver release ledger
# v2.0 STUDY-THESIS corrected ratio-scale driver gate, pre-Q6
#
# Purpose:
# - Supersede the legacy 12.e.6 absolute-error release grading for Q6.
# - Preserve legacy 12.e.6 metrics as diagnostics only.
# - Recompute independent sparse-driver gates before Q6 public release:
#     1) event-rate ratio
#     2) true gap-thresholded burst-count ratio
#     3) 60-second windowed Fano-factor ratio
#     4) inter-arrival KS
#     5) event-duration KS
# - Export release-eligible and release-excluded driver lists used by Cell 15.6.
#
# Scientific contract:
# - TEST real values are used for QA/release gating only, not generation.
# - No synthetic values are mutated.
# - Q6 must consume this ledger; corrected-fatal drivers must not enter X_pub.
# ==========================================================

log("--- START: Cell 12.e.6R - Corrected sparse-driver release ledger (v2.0 pre-Q6) ---")

import os, json, hashlib, gc
import numpy as np
import pandas as pd

_required_12e6r = [
    "CFG", "log", "df_te", "IOT_DRIVER_COLS", "IOT_FINAL_DRIVER_TEST",
    "CELL12E6_DRIVER_FINAL_TEST_QA_METRICS_DF", "OUTDIR", "REPORT_DIR", "CONTRACT_DIR",
]
_missing_12e6r = [k for k in _required_12e6r if k not in globals()]
if _missing_12e6r:
    raise RuntimeError(f"[Cell12.e.6R] Missing required globals: {_missing_12e6r}")

REPORT_DIR = str(REPORT_DIR)
CONTRACT_DIR = str(CONTRACT_DIR)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

CELL12E6R_VERSION = "cell12e6R_corrected_sparse_driver_ratio_release_v2_0_pre_q6"
CFG["cell12e6R_version"] = CELL12E6R_VERSION
CFG.setdefault("cell12e6R_rate_ratio_warning_hi", 1.50)
CFG.setdefault("cell12e6R_rate_ratio_warning_lo", 2.0 / 3.0)
CFG.setdefault("cell12e6R_rate_ratio_fatal_hi", 3.00)
CFG.setdefault("cell12e6R_rate_ratio_fatal_lo", 1.0 / 3.0)
CFG.setdefault("cell12e6R_burst_gap_seconds", 5)
CFG.setdefault("cell12e6R_fano_window_seconds", 60)
CFG.setdefault("cell12e6R_interarrival_ks_warning", CFG.get("cell12e6_interarrival_ks_warning", 0.50))
CFG.setdefault("cell12e6R_interarrival_ks_fatal", CFG.get("cell12e6_interarrival_ks_fatal", 0.80))
CFG.setdefault("cell12e6R_duration_ks_warning", CFG.get("cell12e6_duration_ks_warning", 0.20))
CFG.setdefault("cell12e6R_duration_ks_fatal", CFG.get("cell12e6_duration_ks_fatal", 0.50))
CFG.setdefault("cell12e6R_expected_release_eligible_drivers", CFG.get("expected_public_sparse_driver_cols", 23))

def _sha256_file_12e6r(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _json_sanitize_12e6r(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12e6r(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize_12e6r(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12e6r(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12e6r(obj.to_dict())
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12e6r(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if np.isposinf(x):
            return "Infinity"
        if np.isneginf(x):
            return "-Infinity"
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_12e6r(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12e6r(payload), f, indent=2, sort_keys=True)

def _as_binary_12e6r(x):
    arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=np.float64)
    return (np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) >= 0.5).astype(np.int8)

def _event_idx_12e6r(x):
    return np.flatnonzero(np.asarray(x, dtype=np.int8) > 0).astype(np.int64)

def _burst_count_12e6r(x, gap=None):
    gap = int(CFG.get("cell12e6R_burst_gap_seconds", 5) if gap is None else gap)
    ev = _event_idx_12e6r(x)
    if ev.size == 0:
        return 0
    return int(1 + np.sum(np.diff(ev) > gap))

def _windowed_fano_12e6r(x, window=None):
    window = int(CFG.get("cell12e6R_fano_window_seconds", 60) if window is None else window)
    arr = np.asarray(x, dtype=np.int8)
    n = int(arr.size // window) * window
    if n <= 0:
        return np.nan
    counts = arr[:n].reshape(-1, window).sum(axis=1).astype(np.float64)
    m = float(counts.mean()) if counts.size else np.nan
    if not np.isfinite(m) or m <= 0:
        return np.nan
    return float(counts.var() / m)

def _ratio_12e6r(syn_value, real_value):
    try:
        s = float(syn_value)
    except Exception:
        s = np.nan
    try:
        r = float(real_value)
    except Exception:
        r = np.nan
    s_pos = np.isfinite(s) and s > 0
    r_pos = np.isfinite(r) and r > 0
    if not r_pos:
        return float("inf") if s_pos else 1.0
    if not np.isfinite(s):
        return 0.0
    return float(s / r)

def _grade_ratio_12e6r(r):
    try:
        r = float(r)
    except Exception:
        return "fatal"
    if not np.isfinite(r):
        return "fatal"
    if r >= float(CFG["cell12e6R_rate_ratio_fatal_hi"]) or r <= float(CFG["cell12e6R_rate_ratio_fatal_lo"]):
        return "fatal"
    if r >= float(CFG["cell12e6R_rate_ratio_warning_hi"]) or r <= float(CFG["cell12e6R_rate_ratio_warning_lo"]):
        return "warning"
    return "pass"

def _grade_ks_12e6r(x, warn, fatal):
    try:
        x = float(x)
    except Exception:
        return "fatal"
    if not np.isfinite(x):
        return "fatal"
    if x >= float(fatal):
        return "fatal"
    if x >= float(warn):
        return "warning"
    return "pass"

def _worst_status_12e6r(gates):
    order = {"pass": 0, "warning": 1, "fatal": 2}
    return max([str(g).lower() for g in gates], key=lambda g: order.get(g, 2))

qa = CELL12E6_DRIVER_FINAL_TEST_QA_METRICS_DF.copy()
qa["col"] = qa["col"].astype(str)
qa_map = {str(r["col"]): r.to_dict() for _, r in qa.iterrows()}

syn_driver = IOT_FINAL_DRIVER_TEST.copy()
syn_driver.columns = syn_driver.columns.astype(str)

rows = []
for col in map(str, IOT_DRIVER_COLS):
    if col not in df_te.columns:
        raise RuntimeError(f"[Cell12.e.6R] Missing real TEST driver column: {col}")
    if col not in syn_driver.columns:
        raise RuntimeError(f"[Cell12.e.6R] Missing synthetic TEST driver column: {col}")
    r = _as_binary_12e6r(df_te[col])
    s = _as_binary_12e6r(syn_driver[col])
    if len(r) != len(s):
        raise RuntimeError(f"[Cell12.e.6R] Length mismatch for {col}: real={len(r)} syn={len(s)}")
    real_rate = float(r.mean()) if len(r) else np.nan
    syn_rate = float(s.mean()) if len(s) else np.nan
    real_event_count = int(r.sum())
    syn_event_count = int(s.sum())
    real_burst = _burst_count_12e6r(r)
    syn_burst = _burst_count_12e6r(s)
    real_fano = _windowed_fano_12e6r(r)
    syn_fano = _windowed_fano_12e6r(s)
    rate_ratio = _ratio_12e6r(syn_rate, real_rate)
    burst_ratio = _ratio_12e6r(syn_burst, real_burst)
    fano_ratio = _ratio_12e6r(syn_fano, real_fano)
    q = qa_map.get(col, {})
    inter_ks = q.get("interarrival_ks", np.nan)
    dur_ks = q.get("duration_ks", np.nan)
    gates = {
        "rate_ratio_gate": _grade_ratio_12e6r(rate_ratio),
        "burst_count_ratio_gate": _grade_ratio_12e6r(burst_ratio),
        "windowed_fano_ratio_gate": _grade_ratio_12e6r(fano_ratio),
        "interarrival_ks_gate": _grade_ks_12e6r(inter_ks, CFG["cell12e6R_interarrival_ks_warning"], CFG["cell12e6R_interarrival_ks_fatal"]),
        "duration_ks_gate": _grade_ks_12e6r(dur_ks, CFG["cell12e6R_duration_ks_warning"], CFG["cell12e6R_duration_ks_fatal"]),
    }
    status = _worst_status_12e6r(gates.values())
    nonpass = [f"{k}={v}" for k, v in gates.items() if v != "pass"]
    rows.append({
        "col": col,
        "legacy_driver_publication_status_after_12e6": q.get("driver_publication_status_after_12e6", ""),
        "legacy_event_rate_error": q.get("event_rate_error", np.nan),
        "legacy_burst_count_error": q.get("burst_count_error", np.nan),
        "legacy_overdispersion_error": q.get("overdispersion_error", np.nan),
        "real_event_rate": real_rate,
        "syn_event_rate": syn_rate,
        "real_event_count": real_event_count,
        "syn_event_count": syn_event_count,
        "event_rate_ratio": rate_ratio,
        "real_burst_count": real_burst,
        "syn_burst_count": syn_burst,
        "burst_count_ratio": burst_ratio,
        "real_windowed_fano": real_fano,
        "syn_windowed_fano": syn_fano,
        "windowed_fano_ratio": fano_ratio,
        "windowed_fano_absdiff": abs(float(real_fano) - float(syn_fano)) if np.isfinite(real_fano) and np.isfinite(syn_fano) else np.nan,
        "interarrival_ks": inter_ks,
        "duration_ks": dur_ks,
        **gates,
        "driver_corrected_ratio_battery_status": status,
        "driver_corrected_ratio_battery_reasons": "corrected_ratio_scale_battery_pass" if not nonpass else "corrected_ratio_scale_battery_nonpass|" + "|".join(nonpass),
        "driver_release_eligible_corrected": bool(status != "fatal"),
        "driver_release_exclusion_reason": "" if status != "fatal" else "corrected_ratio_scale_driver_status_fatal",
        "TEST_real_values_used_for_QA_only": True,
        "synthetic_values_mutated": False,
    })

corrected_driver_ledger = pd.DataFrame(rows)
counts = corrected_driver_ledger["driver_corrected_ratio_battery_status"].value_counts().sort_index().to_dict()
release_eligible_cols = sorted(corrected_driver_ledger.loc[corrected_driver_ledger["driver_release_eligible_corrected"], "col"].astype(str).tolist())
release_excluded_cols = sorted(corrected_driver_ledger.loc[~corrected_driver_ledger["driver_release_eligible_corrected"], "col"].astype(str).tolist())
expected_release_n = int(CFG.get("cell12e6R_expected_release_eligible_drivers", CFG.get("expected_public_sparse_driver_cols", len(release_eligible_cols))))
if len(release_eligible_cols) != expected_release_n:
    raise RuntimeError(
        "[Cell12.e.6R] Corrected release-eligible sparse-driver count changed. "
        f"got={len(release_eligible_cols)} expected={expected_release_n}. "
        "Inspect cell12e6_driver_corrected_release_ledger.csv before proceeding to Q6."
    )

ledger_csv = os.path.join(REPORT_DIR, "cell12e6_driver_corrected_release_ledger.csv")
summary_json = os.path.join(REPORT_DIR, "cell12e6_driver_corrected_release_summary.json")
contract_json = os.path.join(CONTRACT_DIR, "cell12e6R_corrected_sparse_driver_release_contract_v2_0_THESIS.json")
corrected_driver_ledger.to_csv(ledger_csv, index=False)
summary = {
    "cell": "12.e.6R",
    "version": CELL12E6R_VERSION,
    "driver_targets_total": int(len(corrected_driver_ledger)),
    "status_counts": counts,
    "release_eligible_n": int(len(release_eligible_cols)),
    "release_excluded_fatal_n": int(len(release_excluded_cols)),
    "release_eligible_cols": release_eligible_cols,
    "release_excluded_fatal_cols": release_excluded_cols,
    "legacy_grading_superseded_for_q6": True,
    "TEST_real_values_used_for_QA_only": True,
    "synthetic_values_mutated": False,
    "thresholds": {
        "ratio_warning_hi": float(CFG["cell12e6R_rate_ratio_warning_hi"]),
        "ratio_warning_lo": float(CFG["cell12e6R_rate_ratio_warning_lo"]),
        "ratio_fatal_hi": float(CFG["cell12e6R_rate_ratio_fatal_hi"]),
        "ratio_fatal_lo": float(CFG["cell12e6R_rate_ratio_fatal_lo"]),
        "burst_gap_seconds": int(CFG["cell12e6R_burst_gap_seconds"]),
        "fano_window_seconds": int(CFG["cell12e6R_fano_window_seconds"]),
    },
    "outputs": {"ledger_csv": ledger_csv, "summary_json": summary_json, "contract_json": contract_json},
}
_write_json_12e6r(summary_json, summary)
_write_json_12e6r(contract_json, summary)
summary["hashes"] = {
    "ledger_csv_sha256": _sha256_file_12e6r(ledger_csv),
    "summary_json_sha256": _sha256_file_12e6r(summary_json),
    "contract_json_sha256": _sha256_file_12e6r(contract_json),
}
_write_json_12e6r(summary_json, summary)
_write_json_12e6r(contract_json, summary)

globals()["CELL12E6R_VERSION"] = CELL12E6R_VERSION
globals()["CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_DF"] = corrected_driver_ledger
globals()["CELL12E6_DRIVER_CORRECTED_RELEASE_LEDGER_CSV"] = ledger_csv
globals()["CELL12E6_DRIVER_CORRECTED_RELEASE_SUMMARY"] = summary
globals()["CELL12E6R_CORRECTED_RELEASE_CONTRACT"] = summary
globals()["DRIVER_RELEASE_ELIGIBLE_COLS_CORRECTED"] = release_eligible_cols
globals()["DRIVER_RELEASE_EXCLUDED_FATAL_COLS_CORRECTED"] = release_excluded_cols

log(f"[Cell12.e.6R] Corrected driver release status counts | {counts}")
log(f"[Cell12.e.6R] Q6 release eligible drivers={len(release_eligible_cols)} | excluded fatal drivers={len(release_excluded_cols)}")
log(f"[Cell12.e.6R] Saved corrected release ledger: {ledger_csv}")
log("--- END: Cell 12.e.6R - Corrected sparse-driver release ledger ---")

gc.collect()
