# ==========================================================
# CELL 15.2 - Q6 DCR / NNDR role-specific privacy audit
# v1.2 STUDY-THESIS active-window nearest-neighbor privacy risk audit, no-copy-forensic-aware
#
# Role:
#   - Compute role-specific DCR and NNDR privacy metrics.
#   - Compare synthetic TEST samples/windows against real TRAIN / VAL / TEST references.
#   - Use role-aware sampling:
#       continuous / observability / mixed: row-level samples
#       protocol: active windows
#       binary: transition-centered windows
#       event drivers: event-centered windows
#   - Carry forward Cell 15.1 no-copy status, but do not mutate data.
#
# Scientific contract:
#   - No synthetic mutation.
#   - No generator fitting.
#   - TEST real values are used only as privacy/reference audit evidence.
#   - DCR / NNDR are release-risk evidence, not privacy proof.
#
# Outputs:
#   reports/cell15_2_dcr_nndr_role_metrics.csv
#   reports/cell15_2_dcr_nndr_top_neighbors.csv
#   reports/cell15_2_dcr_nndr_role_config.csv
#   reports/cell15_2_dcr_nndr_contract.json
#   artifacts/cell15_2_dcr_nndr_manifest.json
# ==========================================================

log("--- START: Cell 15.2 - Q6 DCR / NNDR role-specific privacy audit (v1.2 no-copy-forensic-aware active-window) ---")

import os
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_NN_AVAILABLE_152 = True
except Exception:
    SKLEARN_NN_AVAILABLE_152 = False

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_152 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "REAL_TRAIN_REF_150",
    "REAL_VAL_REF_150",
    "REAL_TEST_REF_150",
    "SYN_TEST_REF_150",
    "CELL15_0_PRIVACY_ROLE_GROUPS",
    "CELL15_0_PRIVACY_ROLE_GROUP_REGISTRY_DF",
    "CELL15_1_NO_COPY_WINDOW_METRICS_DF",
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "CELL15_1_NO_COPY_CONTRACT",
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "CELL15_1A_NO_COPY_BLOCKER_FORENSIC_DF",
]
_missing_152 = [k for k in _required_152 if k not in globals()]
if _missing_152:
    raise RuntimeError(f"[Cell15.2] Missing required globals: {_missing_152}")

if not SKLEARN_NN_AVAILABLE_152:
    raise RuntimeError("[Cell15.2] scikit-learn NearestNeighbors is required for DCR/NNDR audit.")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)

REAL_TRAIN_152 = REAL_TRAIN_REF_150
REAL_VAL_152 = REAL_VAL_REF_150
REAL_TEST_152 = REAL_TEST_REF_150
SYN_TEST_152 = SYN_TEST_REF_150

N_TR = int(len(REAL_TRAIN_152))
N_VAL = int(len(REAL_VAL_152))
N_TE = int(len(REAL_TEST_152))
N_SYN = int(len(SYN_TEST_152))

CELL152_VERSION = "cell15_2_q6_dcr_nndr_role_specific_privacy_v1_2_no_copy_forensic_aware"

CFG["cell15_2_version"] = CELL152_VERSION
CFG["cell15_2_TEST_real_values_used_for_privacy_reference"] = True
CFG["cell15_2_synthetic_values_mutated"] = False
CFG["cell15_2_selection_done_here"] = False
CFG["cell15_2_generator_fit_done_here"] = False
CFG["cell15_2_materialization_done_here"] = False
CFG["cell15_2_privacy_decision_done_here"] = False
CFG["cell15_2_dcr_nndr_audit_done_here"] = True
CFG["cell15_2_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_2_Q4_coupled_artifacts_used"] = False
CFG["cell15_2_no_copy_forensic_carried_forward"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell15_2_max_cols_per_role", 80)
CFG.setdefault("cell15_2_max_syn_samples", 5000)
CFG.setdefault("cell15_2_max_real_train_samples", 12000)
CFG.setdefault("cell15_2_max_real_val_samples", 5000)
CFG.setdefault("cell15_2_max_real_test_samples", 5000)
CFG.setdefault("cell15_2_event_window_len", 300)
CFG.setdefault("cell15_2_event_window_stride", 30)
CFG.setdefault("cell15_2_event_max_windows", 4000)
CFG.setdefault("cell15_2_protocol_window_len", 60)
CFG.setdefault("cell15_2_protocol_window_stride", 10)
CFG.setdefault("cell15_2_binary_window_len", 120)
CFG.setdefault("cell15_2_binary_window_stride", 10)
CFG.setdefault("cell15_2_min_active_fraction_protocol", 0.01)
CFG.setdefault("cell15_2_min_active_fraction_binary", 0.01)
CFG.setdefault("cell15_2_idle_abs_eps", 1e-12)
CFG.setdefault("cell15_2_low_variance_eps", 1e-12)
CFG.setdefault("cell15_2_fill_value", -999999.0)
CFG.setdefault("cell15_2_clip_z", 20.0)
CFG.setdefault("cell15_2_skip_roles", ["iot_placeholders_or_excluded"])
CFG.setdefault("cell15_2_distance_metric", "euclidean")

# Risk gates. DCR risk is relative: synthetic should not be much closer to real TRAIN
# than real TEST is to real TRAIN.
CFG.setdefault("cell15_2_blocker_syn_to_train_dcr_p01_ratio", 0.25)
CFG.setdefault("cell15_2_warning_syn_to_train_dcr_p01_ratio", 0.50)
CFG.setdefault("cell15_2_blocker_nndr_p01", 0.10)
CFG.setdefault("cell15_2_warning_nndr_p01", 0.20)
CFG.setdefault("cell15_2_blocker_close_neighbor_rate", 0.05)
CFG.setdefault("cell15_2_warning_close_neighbor_rate", 0.01)
CFG.setdefault("cell15_2_blocker_close_neighbor_rate_active", 0.05)
CFG.setdefault("cell15_2_warning_close_neighbor_rate_active", 0.01)

MAX_COLS_152 = int(CFG.get("cell15_2_max_cols_per_role", 80))
MAX_SYN_152 = int(CFG.get("cell15_2_max_syn_samples", 5000))
MAX_TR_152 = int(CFG.get("cell15_2_max_real_train_samples", 12000))
MAX_VAL_152 = int(CFG.get("cell15_2_max_real_val_samples", 5000))
MAX_TE_152 = int(CFG.get("cell15_2_max_real_test_samples", 5000))
LOW_VAR_EPS_152 = float(CFG.get("cell15_2_low_variance_eps", 1e-12))
FILL_VALUE_152 = float(CFG.get("cell15_2_fill_value", -999999.0))
CLIP_Z_152 = float(CFG.get("cell15_2_clip_z", 20.0))
SKIP_ROLES_152 = set(map(str, CFG.get("cell15_2_skip_roles", ["iot_placeholders_or_excluded"])))
DISTANCE_METRIC_152 = str(CFG.get("cell15_2_distance_metric", "euclidean"))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
metrics_csv = os.path.join(REPORT_DIR, "cell15_2_dcr_nndr_role_metrics.csv")
top_neighbors_csv = os.path.join(REPORT_DIR, "cell15_2_dcr_nndr_top_neighbors.csv")
role_config_csv = os.path.join(REPORT_DIR, "cell15_2_dcr_nndr_role_config.csv")
contract_json = os.path.join(REPORT_DIR, "cell15_2_dcr_nndr_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell15_2_dcr_nndr_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell15_2_dcr_nndr_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_152(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_152(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_152(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_152(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_152(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_152(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_152(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_152(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj

def _write_json_152(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_152(payload), f, indent=2, sort_keys=True)

def _sha256_file_152(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_152(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.2] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.2] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

def _dedup_152(seq):
    seen = set()
    out = []
    for x in seq:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _available_common_cols_152(cols):
    cols = _dedup_152(cols)
    return [
        c for c in cols
        if c in REAL_TRAIN_152.columns
        and c in REAL_VAL_152.columns
        and c in REAL_TEST_152.columns
        and c in SYN_TEST_152.columns
    ]

def _choose_cols_152(role: str, cols: list):
    cols = _available_common_cols_152(cols)
    if not cols:
        return []

    rows = []
    for c in cols:
        x_syn = pd.to_numeric(SYN_TEST_152[c], errors="coerce")
        x_tr = pd.to_numeric(REAL_TRAIN_152[c], errors="coerce")
        x_te = pd.to_numeric(REAL_TEST_152[c], errors="coerce")

        syn_finite = float(x_syn.notna().mean()) if len(x_syn) else 0.0
        tr_finite = float(x_tr.notna().mean()) if len(x_tr) else 0.0
        te_finite = float(x_te.notna().mean()) if len(x_te) else 0.0

        syn_std = float(np.nanstd(x_syn.to_numpy(dtype=np.float64))) if x_syn.notna().any() else 0.0
        tr_std = float(np.nanstd(x_tr.to_numpy(dtype=np.float64))) if x_tr.notna().any() else 0.0
        te_std = float(np.nanstd(x_te.to_numpy(dtype=np.float64))) if x_te.notna().any() else 0.0

        rows.append({
            "col": c,
            "score": min(syn_finite, tr_finite, te_finite) * np.log1p(max(syn_std, tr_std, te_std)),
            "syn_finite": syn_finite,
            "tr_finite": tr_finite,
            "te_finite": te_finite,
            "syn_std": syn_std,
            "tr_std": tr_std,
            "te_std": te_std,
        })

    d = pd.DataFrame(rows)
    d["score"] = pd.to_numeric(d["score"], errors="coerce").fillna(0.0)

    if role == "iot_event_drivers":
        chosen = d.sort_values(["score", "col"], ascending=[False, True])["col"].head(MAX_COLS_152).astype(str).tolist()
    else:
        dd = d[
            (pd.to_numeric(d["syn_finite"], errors="coerce").fillna(0.0) > 0.001)
            & (pd.to_numeric(d["tr_finite"], errors="coerce").fillna(0.0) > 0.001)
            & (pd.to_numeric(d["te_finite"], errors="coerce").fillna(0.0) > 0.001)
            & (
                (pd.to_numeric(d["syn_std"], errors="coerce").fillna(0.0) > LOW_VAR_EPS_152)
                | (pd.to_numeric(d["tr_std"], errors="coerce").fillna(0.0) > LOW_VAR_EPS_152)
                | (pd.to_numeric(d["te_std"], errors="coerce").fillna(0.0) > LOW_VAR_EPS_152)
            )
        ].copy()

        if len(dd) == 0:
            dd = d.copy()

        chosen = dd.sort_values(["score", "col"], ascending=[False, True])["col"].head(MAX_COLS_152).astype(str).tolist()

    return chosen

def _sample_rows_152(n: int, max_n: int, seed_offset: int):
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(SEED + seed_offset)
    idx = rng.choice(np.arange(n, dtype=np.int64), size=max_n, replace=False)
    return np.asarray(np.sort(idx), dtype=np.int64)

def _event_starts_152(frame: pd.DataFrame, cols: list):
    if not cols:
        return np.asarray([], dtype=np.int64)

    mat = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    active = np.any(mat > 0.0, axis=1)
    prev = np.r_[False, active[:-1]]
    starts = np.flatnonzero(active & (~prev)).astype(np.int64)
    return starts

def _event_window_starts_152(frame: pd.DataFrame, cols: list, max_n: int, seed_offset: int):
    win = int(CFG.get("cell15_2_event_window_len", 300))
    starts = _event_starts_152(frame, cols)
    if starts.size == 0:
        # Fallback regular sparse windows, but mark through config.
        stride = int(CFG.get("cell15_2_event_window_stride", 30))
        if len(frame) < win:
            return np.asarray([], dtype=np.int64), win
        starts = np.arange(0, len(frame) - win + 1, stride, dtype=np.int64)

    # center event near beginning of window while keeping valid.
    half = win // 2
    starts = np.asarray([max(0, min(len(frame) - win, int(s) - half)) for s in starts], dtype=np.int64)
    starts = np.unique(starts)

    if starts.size > max_n:
        rng = np.random.default_rng(SEED + seed_offset)
        starts = rng.choice(starts, size=max_n, replace=False)
        starts = np.sort(starts)

    return starts.astype(np.int64), win

def _fit_scaler_152(cols):
    med = []
    scale = []
    for c in cols:
        x = pd.to_numeric(REAL_TRAIN_152[c], errors="coerce").to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            med.append(0.0)
            scale.append(1.0)
            continue
        q25, q50, q75 = np.nanquantile(x, [0.25, 0.50, 0.75])
        sc = float(q75 - q25)
        if not np.isfinite(sc) or sc <= LOW_VAR_EPS_152:
            sc = float(np.nanstd(x))
        if not np.isfinite(sc) or sc <= LOW_VAR_EPS_152:
            sc = 1.0
        med.append(float(q50))
        scale.append(float(sc))
    return np.asarray(med, dtype=np.float64), np.asarray(scale, dtype=np.float64)

def _make_row_matrix_152(frame: pd.DataFrame, cols: list, idx: np.ndarray, med: np.ndarray, scale: np.ndarray):
    if len(cols) == 0 or len(idx) == 0:
        return np.empty((0, 0), dtype=np.float32)

    arr = frame.iloc[idx][cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    z = (arr - med.reshape(1, -1)) / scale.reshape(1, -1)
    z = np.where(np.isfinite(z), z, FILL_VALUE_152)
    z = np.clip(z, -CLIP_Z_152, CLIP_Z_152)
    return z.astype(np.float32)

def _make_window_matrix_152(frame: pd.DataFrame, cols: list, starts: np.ndarray, window_len: int, med: np.ndarray, scale: np.ndarray):
    if len(cols) == 0 or len(starts) == 0:
        return np.empty((0, 0), dtype=np.float32)

    arr = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    out = np.empty((len(starts), window_len * len(cols)), dtype=np.float32)

    med2 = med.reshape(1, -1)
    scale2 = scale.reshape(1, -1)

    for i, s in enumerate(starts):
        w = arr[int(s):int(s) + window_len, :]
        z = (w - med2) / scale2
        z = np.where(np.isfinite(z), z, FILL_VALUE_152)
        z = np.clip(z, -CLIP_Z_152, CLIP_Z_152)
        out[i, :] = z.reshape(-1).astype(np.float32)

    return out

def _is_protocol_role_152(role: str) -> bool:
    return str(role) in {"protocol_router", "protocol_ota", "protocol_zigbee"}

def _is_binary_role_152(role: str) -> bool:
    return str(role) == "iot_binary_states"

def _activity_score_152(frame: pd.DataFrame, cols: list) -> np.ndarray:
    if not cols:
        return np.zeros(len(frame), dtype=np.float64)
    mat = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.nanmean(np.abs(mat) > float(CFG.get("cell15_2_idle_abs_eps", 1e-12)), axis=1)

def _transition_score_152(frame: pd.DataFrame, cols: list) -> np.ndarray:
    if not cols:
        return np.zeros(len(frame), dtype=np.float64)
    mat = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if mat.shape[0] <= 1:
        return np.zeros(mat.shape[0], dtype=np.float64)
    diff = np.vstack([np.zeros((1, mat.shape[1]), dtype=np.float64), np.abs(np.diff(mat, axis=0))])
    return np.nanmean(diff > float(CFG.get("cell15_2_idle_abs_eps", 1e-12)), axis=1)

def _active_window_starts_152(frame, cols, window_len, stride, max_n, seed_offset, mode):
    n = len(frame)
    if n < window_len:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=bool)

    starts = np.arange(0, n - window_len + 1, max(1, int(stride)), dtype=np.int64)

    if mode == "protocol_active":
        score = _activity_score_152(frame, cols)
        min_frac = float(CFG.get("cell15_2_min_active_fraction_protocol", 0.01))
    elif mode == "binary_transition":
        score = _transition_score_152(frame, cols)
        min_frac = float(CFG.get("cell15_2_min_active_fraction_binary", 0.01))
    else:
        score = _activity_score_152(frame, cols)
        min_frac = 0.0

    active_flags = []
    for s in starts:
        seg = score[int(s):int(s) + int(window_len)]
        active_flags.append(bool(np.nanmean(seg) >= min_frac) if len(seg) else False)

    active_flags = np.asarray(active_flags, dtype=bool)
    active_starts = starts[active_flags]

    # If strict active filtering finds nothing, fall back to the top-scoring windows.
    if active_starts.size == 0 and starts.size > 0:
        win_scores = []
        for s in starts:
            seg = score[int(s):int(s) + int(window_len)]
            win_scores.append(float(np.nanmean(seg)) if len(seg) else 0.0)
        win_scores = np.asarray(win_scores, dtype=np.float64)
        order = np.argsort(-win_scores)
        keep_n = min(max_n, max(1, min(len(order), max_n)))
        active_starts = starts[order[:keep_n]]
        active_flags = np.zeros(len(starts), dtype=bool)
        active_flags[order[:keep_n]] = True

    if active_starts.size > max_n:
        rng = np.random.default_rng(SEED + seed_offset)
        active_starts = rng.choice(active_starts, size=max_n, replace=False)
        active_starts = np.sort(active_starts)

    return active_starts.astype(np.int64), active_flags

def _make_window_matrix_and_idle_152(frame, cols, starts, window_len, med, scale):
    if len(cols) == 0 or len(starts) == 0:
        return np.empty((0, 0), dtype=np.float32), np.asarray([], dtype=bool)

    arr = frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out = np.empty((len(starts), window_len * len(cols)), dtype=np.float32)
    idle = np.zeros(len(starts), dtype=bool)

    med2 = med.reshape(1, -1)
    scale2 = scale.reshape(1, -1)

    for i, s in enumerate(starts):
        w_raw = arr[int(s):int(s) + int(window_len), :]
        idle[i] = bool(np.nanmean(np.abs(w_raw) > float(CFG.get("cell15_2_idle_abs_eps", 1e-12))) < 1e-12)
        z = (w_raw - med2) / scale2
        z = np.where(np.isfinite(z), z, FILL_VALUE_152)
        z = np.clip(z, -CLIP_Z_152, CLIP_Z_152)
        out[i, :] = z.reshape(-1).astype(np.float32)

    return out, idle

def _build_matrices_152(role: str, cols: list, role_idx: int):
    med, scale = _fit_scaler_152(cols)

    if role == "iot_event_drivers":
        tr_idx, win = _event_window_starts_152(REAL_TRAIN_152, cols, MAX_TR_152, 15210 + role_idx)
        val_idx, _ = _event_window_starts_152(REAL_VAL_152, cols, MAX_VAL_152, 15220 + role_idx)
        te_idx, _ = _event_window_starts_152(REAL_TEST_152, cols, MAX_TE_152, 15230 + role_idx)
        syn_idx, _ = _event_window_starts_152(SYN_TEST_152, cols, MAX_SYN_152, 15240 + role_idx)

        X_tr = _make_window_matrix_152(REAL_TRAIN_152, cols, tr_idx, win, med, scale)
        X_val = _make_window_matrix_152(REAL_VAL_152, cols, val_idx, win, med, scale)
        X_te = _make_window_matrix_152(REAL_TEST_152, cols, te_idx, win, med, scale)
        X_syn = _make_window_matrix_152(SYN_TEST_152, cols, syn_idx, win, med, scale)

        return {
            "representation": "event_centered_window",
            "window_len": int(win),
            "train_idx": tr_idx,
            "val_idx": val_idx,
            "test_idx": te_idx,
            "syn_idx": syn_idx,
            "X_train": X_tr,
            "X_val": X_val,
            "X_test": X_te,
            "X_syn": X_syn,
            "train_idle": np.zeros(int(X_tr.shape[0]), dtype=bool),
            "val_idle": np.zeros(int(X_val.shape[0]), dtype=bool),
            "test_idle": np.zeros(int(X_te.shape[0]), dtype=bool),
            "syn_idle": np.zeros(int(X_syn.shape[0]), dtype=bool),
        }

    if _is_protocol_role_152(role) or _is_binary_role_152(role):
        if _is_protocol_role_152(role):
            win = int(CFG.get("cell15_2_protocol_window_len", 60))
            stride = int(CFG.get("cell15_2_protocol_window_stride", 10))
            mode = "protocol_active"
            representation = "protocol_active_window"
        else:
            win = int(CFG.get("cell15_2_binary_window_len", 120))
            stride = int(CFG.get("cell15_2_binary_window_stride", 10))
            mode = "binary_transition"
            representation = "binary_transition_window"

        tr_idx, _ = _active_window_starts_152(REAL_TRAIN_152, cols, win, stride, MAX_TR_152, 15210 + role_idx, mode)
        val_idx, _ = _active_window_starts_152(REAL_VAL_152, cols, win, stride, MAX_VAL_152, 15220 + role_idx, mode)
        te_idx, _ = _active_window_starts_152(REAL_TEST_152, cols, win, stride, MAX_TE_152, 15230 + role_idx, mode)
        syn_idx, _ = _active_window_starts_152(SYN_TEST_152, cols, win, stride, MAX_SYN_152, 15240 + role_idx, mode)

        X_tr, tr_idle = _make_window_matrix_and_idle_152(REAL_TRAIN_152, cols, tr_idx, win, med, scale)
        X_val, val_idle = _make_window_matrix_and_idle_152(REAL_VAL_152, cols, val_idx, win, med, scale)
        X_te, te_idle = _make_window_matrix_and_idle_152(REAL_TEST_152, cols, te_idx, win, med, scale)
        X_syn, syn_idle = _make_window_matrix_and_idle_152(SYN_TEST_152, cols, syn_idx, win, med, scale)

        return {
            "representation": representation,
            "window_len": int(win),
            "train_idx": tr_idx,
            "val_idx": val_idx,
            "test_idx": te_idx,
            "syn_idx": syn_idx,
            "X_train": X_tr,
            "X_val": X_val,
            "X_test": X_te,
            "X_syn": X_syn,
            "train_idle": tr_idle,
            "val_idle": val_idle,
            "test_idle": te_idle,
            "syn_idle": syn_idle,
        }

    tr_idx = _sample_rows_152(N_TR, MAX_TR_152, 15210 + role_idx)
    val_idx = _sample_rows_152(N_VAL, MAX_VAL_152, 15220 + role_idx)
    te_idx = _sample_rows_152(N_TE, MAX_TE_152, 15230 + role_idx)
    syn_idx = _sample_rows_152(N_SYN, MAX_SYN_152, 15240 + role_idx)

    X_tr = _make_row_matrix_152(REAL_TRAIN_152, cols, tr_idx, med, scale)
    X_val = _make_row_matrix_152(REAL_VAL_152, cols, val_idx, med, scale)
    X_te = _make_row_matrix_152(REAL_TEST_152, cols, te_idx, med, scale)
    X_syn = _make_row_matrix_152(SYN_TEST_152, cols, syn_idx, med, scale)

    return {
        "representation": "row",
        "window_len": 1,
        "train_idx": tr_idx,
        "val_idx": val_idx,
        "test_idx": te_idx,
        "syn_idx": syn_idx,
        "X_train": X_tr,
        "X_val": X_val,
        "X_test": X_te,
        "X_syn": X_syn,
        "train_idle": np.zeros(int(X_tr.shape[0]), dtype=bool),
        "val_idle": np.zeros(int(X_val.shape[0]), dtype=bool),
        "test_idle": np.zeros(int(X_te.shape[0]), dtype=bool),
        "syn_idle": np.zeros(int(X_syn.shape[0]), dtype=bool),
    }

def _nearest_two_152(ref_mat: np.ndarray, query_mat: np.ndarray):
    if ref_mat.shape[0] < 2 or query_mat.shape[0] == 0 or ref_mat.shape[1] == 0:
        n = int(query_mat.shape[0])
        return (
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, np.nan, dtype=np.float64),
            np.full(n, -1, dtype=np.int64),
            np.full(n, -1, dtype=np.int64),
        )

    nn = NearestNeighbors(n_neighbors=2, metric=DISTANCE_METRIC_152, algorithm="auto")
    nn.fit(ref_mat)
    dist, idx = nn.kneighbors(query_mat, return_distance=True)

    d1 = dist[:, 0].astype(np.float64)
    d2 = dist[:, 1].astype(np.float64)
    i1 = idx[:, 0].astype(np.int64)
    i2 = idx[:, 1].astype(np.int64)
    return d1, d2, i1, i2

def _nndr_152(d1, d2):
    d1 = np.asarray(d1, dtype=np.float64)
    d2 = np.asarray(d2, dtype=np.float64)
    return d1 / np.maximum(d2, 1e-12)

def _summarize_dist_152(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": np.nan, "median": np.nan,
            "p01": np.nan, "p05": np.nan, "p10": np.nan,
            "p25": np.nan, "p75": np.nan, "min": np.nan,
        }
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p10": float(np.quantile(x, 0.10)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "min": float(np.min(x)),
    }

def _status_152(
    syn_train_dcr_p01_ratio_active,
    syn_train_nndr_p01_active,
    close_neighbor_rate_active,
    no_copy_status,
    calibration_degenerate=False,
):
    reasons = []

    if (
        not calibration_degenerate
        and np.isfinite(syn_train_dcr_p01_ratio_active)
        and syn_train_dcr_p01_ratio_active < float(CFG.get("cell15_2_blocker_syn_to_train_dcr_p01_ratio", 0.25))
    ):
        reasons.append("synthetic_train_DCR_p01_ratio_blocker")

    if np.isfinite(syn_train_nndr_p01_active) and syn_train_nndr_p01_active < float(CFG.get("cell15_2_blocker_nndr_p01", 0.10)):
        reasons.append("NNDR_p01_blocker")

    if (
        not calibration_degenerate
        and np.isfinite(close_neighbor_rate_active)
        and close_neighbor_rate_active > float(CFG.get("cell15_2_blocker_close_neighbor_rate_active", 0.05))
    ):
        reasons.append("close_neighbor_rate_blocker")

    if str(no_copy_status) == "blocker":
        reasons.append("carried_no_copy_window_blocker")

    if reasons:
        return "blocker", "|".join(reasons)

    warn = []
    if (
        not calibration_degenerate
        and np.isfinite(syn_train_dcr_p01_ratio_active)
        and syn_train_dcr_p01_ratio_active < float(CFG.get("cell15_2_warning_syn_to_train_dcr_p01_ratio", 0.50))
    ):
        warn.append("synthetic_train_DCR_p01_ratio_warning")

    if np.isfinite(syn_train_nndr_p01_active) and syn_train_nndr_p01_active < float(CFG.get("cell15_2_warning_nndr_p01", 0.20)):
        warn.append("NNDR_p01_warning")

    if (
        not calibration_degenerate
        and np.isfinite(close_neighbor_rate_active)
        and close_neighbor_rate_active > float(CFG.get("cell15_2_warning_close_neighbor_rate_active", 0.01))
    ):
        warn.append("close_neighbor_rate_warning")

    if str(no_copy_status) == "warning":
        warn.append("carried_no_copy_window_warning")

    if calibration_degenerate:
        warn.append("real_test_train_active_calibration_degenerate")

    if warn:
        return "warning", "|".join(warn)

    return "pass", "within_DCR_NNDR_privacy_gates"

def _previous_v1_dcr_status_152(syn_train_dcr_p01_ratio, syn_train_nndr_p01, close_neighbor_rate):
    reasons = []
    if np.isfinite(syn_train_dcr_p01_ratio) and syn_train_dcr_p01_ratio < float(CFG.get("cell15_2_blocker_syn_to_train_dcr_p01_ratio", 0.25)):
        reasons.append("synthetic_train_DCR_p01_ratio_blocker")
    if np.isfinite(syn_train_nndr_p01) and syn_train_nndr_p01 < float(CFG.get("cell15_2_blocker_nndr_p01", 0.10)):
        reasons.append("NNDR_p01_blocker")
    if np.isfinite(close_neighbor_rate) and close_neighbor_rate > float(CFG.get("cell15_2_blocker_close_neighbor_rate", 0.05)):
        reasons.append("close_neighbor_rate_blocker")
    if reasons:
        return "blocker", "|".join(reasons)

    warnings = []
    if np.isfinite(syn_train_dcr_p01_ratio) and syn_train_dcr_p01_ratio < float(CFG.get("cell15_2_warning_syn_to_train_dcr_p01_ratio", 0.50)):
        warnings.append("synthetic_train_DCR_p01_ratio_warning")
    if np.isfinite(syn_train_nndr_p01) and syn_train_nndr_p01 < float(CFG.get("cell15_2_warning_nndr_p01", 0.20)):
        warnings.append("NNDR_p01_warning")
    if np.isfinite(close_neighbor_rate) and close_neighbor_rate > float(CFG.get("cell15_2_warning_close_neighbor_rate", 0.01)):
        warnings.append("close_neighbor_rate_warning")
    if warnings:
        return "warning", "|".join(warnings)

    return "pass", "within_v1_DCR_NNDR_gates"

# ----------------------------------------------------------
# 4) Validate upstream no-Q4/no-copy contracts
# ----------------------------------------------------------
version150_152 = _require_contract_version_152(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
version151_152 = _require_contract_version_152(
    CELL15_1_NO_COPY_CONTRACT,
    "CELL15_1_NO_COPY_CONTRACT",
    "cell15_1_q6_no_copy_window_audit_v1_1",
)
version151a_152 = _require_contract_version_152(
    CELL15_1A_NO_COPY_FORENSIC_CONTRACT,
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "cell15_1a_q6_no_copy_blocker_forensic_v1_0",
)

strict150_152 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
if str(strict150_152.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.2] Cell 15.0 does not carry Q4_final_status=blocked_no_promotion.")
if bool(strict150_152.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.2] Cell 15.0 used Q4-coupled artifacts unexpectedly.")

strict151_152 = CELL15_1_NO_COPY_CONTRACT.get("strict_contract", {})
if bool(strict151_152.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell15.2] Cell 15.1 indicates synthetic mutation.")
if not bool(strict151_152.get("no_copy_window_audit_done_here", False)):
    raise RuntimeError("[Cell15.2] Cell 15.1 did not complete no-copy audit.")

strict151a_152 = CELL15_1A_NO_COPY_FORENSIC_CONTRACT.get("strict_contract", {})
if bool(strict151a_152.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell15.2] Cell 15.1a indicates synthetic mutation.")
if not bool(strict151a_152.get("diagnostic_only", False)):
    raise RuntimeError("[Cell15.2] Cell 15.1a did not declare diagnostic_only.")

no_copy_forensic_summary_152 = CELL15_1A_NO_COPY_FORENSIC_CONTRACT.get("summary", {})
no_copy_blocker_roles_152 = list(map(str, no_copy_forensic_summary_152.get("blocker_roles", [])))
no_copy_warning_roles_152 = list(map(str, no_copy_forensic_summary_152.get("warning_roles", [])))
no_copy_reason_counts_152 = no_copy_forensic_summary_152.get("reason_counts", {})

# ----------------------------------------------------------
# 5) Run DCR / NNDR by role
# ----------------------------------------------------------
role_groups = CELL15_0_PRIVACY_ROLE_GROUPS
no_copy_df = CELL15_1_NO_COPY_WINDOW_METRICS_DF.copy()
no_copy_status_map = dict(zip(no_copy_df["role_group"].astype(str), no_copy_df["status"].astype(str))) if len(no_copy_df) else {}

roles_to_run = [r for r in role_groups.keys() if r not in SKIP_ROLES_152]

metrics_rows = []
top_rows = []
role_config_rows = []

log(
    "[Cell15.2] Running DCR/NNDR privacy audit | "
    f"roles={roles_to_run} | metric={DISTANCE_METRIC_152}"
)

for role_idx, role_name in enumerate(roles_to_run, start=1):
    raw_cols = role_groups.get(role_name, [])
    cols = _choose_cols_152(role_name, raw_cols)

    role_config_rows.append({
        "role_group": role_name,
        "raw_cols_n": int(len(raw_cols)),
        "selected_cols_n": int(len(cols)),
        "max_cols": int(MAX_COLS_152),
        "max_syn_samples": int(MAX_SYN_152),
        "max_real_train_samples": int(MAX_TR_152),
        "max_real_val_samples": int(MAX_VAL_152),
        "max_real_test_samples": int(MAX_TE_152),
        "distance_metric": DISTANCE_METRIC_152,
        "protocol_window_len": int(CFG.get("cell15_2_protocol_window_len", 60)),
        "protocol_window_stride": int(CFG.get("cell15_2_protocol_window_stride", 10)),
        "binary_window_len": int(CFG.get("cell15_2_binary_window_len", 120)),
        "binary_window_stride": int(CFG.get("cell15_2_binary_window_stride", 10)),
    })

    if len(cols) == 0:
        metrics_rows.append({
            "role_group": role_name,
            "status": "not_evaluable",
            "reasons": "no_common_informative_columns",
            "selected_cols_n": 0,
            "representation": "",
            "syn_samples_n": 0,
            "train_ref_n": 0,
            "test_ref_n": 0,
            "syn_to_train_dcr_p01": np.nan,
            "real_test_to_train_dcr_p01": np.nan,
            "syn_train_dcr_p01_ratio": np.nan,
            "syn_to_train_nndr_p01": np.nan,
            "close_neighbor_rate": np.nan,
            "syn_active_samples_n": 0,
            "real_test_active_samples_n": 0,
            "syn_idle_samples_n": 0,
            "real_test_idle_samples_n": 0,
            "syn_to_train_dcr_active_p01": np.nan,
            "real_test_to_train_dcr_active_p01": np.nan,
            "syn_train_dcr_p01_ratio_active": np.nan,
            "close_neighbor_threshold_active": np.nan,
            "close_neighbor_rate_active": np.nan,
            "syn_to_train_nndr_active_p01": np.nan,
            "calibration_degenerate_active": True,
            "previous_v1_dcr_status": "",
            "carried_no_copy_status": no_copy_status_map.get(role_name, ""),
            "TEST_real_values_used_for_privacy_reference": True,
            "synthetic_values_mutated": False,
        })
        continue

    log(f"[Cell15.2] role {role_idx}/{len(roles_to_run)} | {role_name} | cols={len(cols)}")

    mats = _build_matrices_152(role_name, cols, role_idx)

    X_tr = mats["X_train"]
    X_val = mats["X_val"]
    X_te = mats["X_test"]
    X_syn = mats["X_syn"]

    if X_tr.shape[0] < 2 or X_syn.shape[0] == 0 or X_te.shape[0] == 0:
        metrics_rows.append({
            "role_group": role_name,
            "status": "not_evaluable",
            "reasons": "insufficient_reference_or_synthetic_samples",
            "selected_cols_n": int(len(cols)),
            "representation": mats["representation"],
            "syn_samples_n": int(X_syn.shape[0]),
            "train_ref_n": int(X_tr.shape[0]),
            "test_ref_n": int(X_te.shape[0]),
            "syn_to_train_dcr_p01": np.nan,
            "real_test_to_train_dcr_p01": np.nan,
            "syn_train_dcr_p01_ratio": np.nan,
            "syn_to_train_nndr_p01": np.nan,
            "close_neighbor_rate": np.nan,
            "syn_active_samples_n": 0,
            "real_test_active_samples_n": 0,
            "syn_idle_samples_n": 0,
            "real_test_idle_samples_n": 0,
            "syn_to_train_dcr_active_p01": np.nan,
            "real_test_to_train_dcr_active_p01": np.nan,
            "syn_train_dcr_p01_ratio_active": np.nan,
            "close_neighbor_threshold_active": np.nan,
            "close_neighbor_rate_active": np.nan,
            "syn_to_train_nndr_active_p01": np.nan,
            "calibration_degenerate_active": True,
            "previous_v1_dcr_status": "",
            "carried_no_copy_status": no_copy_status_map.get(role_name, ""),
            "TEST_real_values_used_for_privacy_reference": True,
            "synthetic_values_mutated": False,
        })
        continue

    # Main privacy reference: synthetic TEST -> real TRAIN.
    syn_d1_tr, syn_d2_tr, syn_i1_tr, syn_i2_tr = _nearest_two_152(X_tr, X_syn)
    syn_nndr_tr = _nndr_152(syn_d1_tr, syn_d2_tr)

    # Calibration references.
    te_d1_tr, te_d2_tr, te_i1_tr, _ = _nearest_two_152(X_tr, X_te)
    te_nndr_tr = _nndr_152(te_d1_tr, te_d2_tr)

    val_d1_tr, val_d2_tr, _, _ = _nearest_two_152(X_tr, X_val)
    val_nndr_tr = _nndr_152(val_d1_tr, val_d2_tr)

    # Extra: synthetic -> real TEST. This is not the main release risk, but helps
    # identify synthetic windows that resemble held-out household behavior.
    syn_d1_te, syn_d2_te, syn_i1_te, _ = _nearest_two_152(X_te, X_syn)
    syn_nndr_te = _nndr_152(syn_d1_te, syn_d2_te)

    syn_dcr_stats = _summarize_dist_152(syn_d1_tr)
    te_dcr_stats = _summarize_dist_152(te_d1_tr)
    val_dcr_stats = _summarize_dist_152(val_d1_tr)
    syn_nndr_stats = _summarize_dist_152(syn_nndr_tr)
    te_nndr_stats = _summarize_dist_152(te_nndr_tr)
    syn_te_dcr_stats = _summarize_dist_152(syn_d1_te)

    syn_train_dcr_p01_ratio = (
        syn_dcr_stats["p01"] / max(te_dcr_stats["p01"], 1e-12)
        if np.isfinite(syn_dcr_stats["p01"]) and np.isfinite(te_dcr_stats["p01"])
        else np.nan
    )

    # Close neighbor threshold derived from the real TEST->TRAIN 1st percentile.
    close_threshold = te_dcr_stats["p01"] if np.isfinite(te_dcr_stats["p01"]) else np.nan
    close_neighbor_rate = (
        float(np.mean(syn_d1_tr <= close_threshold))
        if np.isfinite(close_threshold) and len(syn_d1_tr)
        else np.nan
    )

    syn_idle = np.asarray(mats.get("syn_idle", np.zeros(len(syn_d1_tr), dtype=bool)), dtype=bool)
    te_idle = np.asarray(mats.get("test_idle", np.zeros(len(te_d1_tr), dtype=bool)), dtype=bool)

    syn_active_mask = ~syn_idle
    te_active_mask = ~te_idle

    syn_d1_tr_active = syn_d1_tr[syn_active_mask] if len(syn_d1_tr) == len(syn_active_mask) else syn_d1_tr
    syn_nndr_tr_active = syn_nndr_tr[syn_active_mask] if len(syn_nndr_tr) == len(syn_active_mask) else syn_nndr_tr
    te_d1_tr_active = te_d1_tr[te_active_mask] if len(te_d1_tr) == len(te_active_mask) else te_d1_tr

    syn_dcr_active_stats = _summarize_dist_152(syn_d1_tr_active)
    syn_nndr_active_stats = _summarize_dist_152(syn_nndr_tr_active)
    te_dcr_active_stats = _summarize_dist_152(te_d1_tr_active)

    calibration_degenerate_active = bool(
        (not np.isfinite(te_dcr_active_stats["p01"]))
        or te_dcr_active_stats["p01"] <= 1e-12
    )

    syn_train_dcr_p01_ratio_active = (
        syn_dcr_active_stats["p01"] / max(te_dcr_active_stats["p01"], 1e-12)
        if np.isfinite(syn_dcr_active_stats["p01"])
        and np.isfinite(te_dcr_active_stats["p01"])
        and te_dcr_active_stats["p01"] > 1e-12
        else np.nan
    )

    close_threshold_active = (
        te_dcr_active_stats["p01"]
        if np.isfinite(te_dcr_active_stats["p01"]) and te_dcr_active_stats["p01"] > 1e-12
        else np.nan
    )

    close_neighbor_rate_active = (
        float(np.mean(syn_d1_tr_active <= close_threshold_active))
        if np.isfinite(close_threshold_active) and len(syn_d1_tr_active)
        else np.nan
    )

    previous_v1_dcr_status, previous_v1_dcr_reasons = _previous_v1_dcr_status_152(
        syn_train_dcr_p01_ratio=syn_train_dcr_p01_ratio,
        syn_train_nndr_p01=syn_nndr_stats["p01"],
        close_neighbor_rate=close_neighbor_rate,
    )

    no_copy_status = no_copy_status_map.get(role_name, "")
    status, reasons = _status_152(
        syn_train_dcr_p01_ratio_active=syn_train_dcr_p01_ratio_active,
        syn_train_nndr_p01_active=syn_nndr_active_stats["p01"],
        close_neighbor_rate_active=close_neighbor_rate_active,
        no_copy_status=no_copy_status,
        calibration_degenerate=calibration_degenerate_active,
    )

    metrics_rows.append({
        "role_group": role_name,
        "status": status,
        "reasons": reasons,
        "selected_cols_n": int(len(cols)),
        "representation": mats["representation"],
        "window_len": int(mats["window_len"]),
        "syn_samples_n": int(X_syn.shape[0]),
        "train_ref_n": int(X_tr.shape[0]),
        "val_ref_n": int(X_val.shape[0]),
        "test_ref_n": int(X_te.shape[0]),
        "distance_metric": DISTANCE_METRIC_152,

        "syn_to_train_dcr_mean": syn_dcr_stats["mean"],
        "syn_to_train_dcr_median": syn_dcr_stats["median"],
        "syn_to_train_dcr_p01": syn_dcr_stats["p01"],
        "syn_to_train_dcr_p05": syn_dcr_stats["p05"],
        "syn_to_train_dcr_min": syn_dcr_stats["min"],

        "real_test_to_train_dcr_mean": te_dcr_stats["mean"],
        "real_test_to_train_dcr_median": te_dcr_stats["median"],
        "real_test_to_train_dcr_p01": te_dcr_stats["p01"],
        "real_test_to_train_dcr_p05": te_dcr_stats["p05"],
        "real_test_to_train_dcr_min": te_dcr_stats["min"],

        "real_val_to_train_dcr_p01": val_dcr_stats["p01"],
        "syn_to_real_test_dcr_p01": syn_te_dcr_stats["p01"],

        "syn_train_dcr_p01_ratio": syn_train_dcr_p01_ratio,
        "close_neighbor_threshold_real_test_train_p01": close_threshold,
        "close_neighbor_rate": close_neighbor_rate,
        "syn_active_samples_n": int(np.sum(syn_active_mask)) if len(syn_active_mask) else int(len(syn_d1_tr_active)),
        "real_test_active_samples_n": int(np.sum(te_active_mask)) if len(te_active_mask) else int(len(te_d1_tr_active)),
        "syn_idle_samples_n": int(np.sum(syn_idle)) if len(syn_idle) else 0,
        "real_test_idle_samples_n": int(np.sum(te_idle)) if len(te_idle) else 0,
        "syn_to_train_dcr_active_p01": syn_dcr_active_stats["p01"],
        "real_test_to_train_dcr_active_p01": te_dcr_active_stats["p01"],
        "syn_train_dcr_p01_ratio_active": syn_train_dcr_p01_ratio_active,
        "close_neighbor_threshold_active": close_threshold_active,
        "close_neighbor_rate_active": close_neighbor_rate_active,
        "syn_to_train_nndr_active_p01": syn_nndr_active_stats["p01"],
        "calibration_degenerate_active": bool(calibration_degenerate_active),
        "previous_v1_dcr_status": previous_v1_dcr_status,
        "previous_v1_dcr_reasons": previous_v1_dcr_reasons,

        "syn_to_train_nndr_mean": syn_nndr_stats["mean"],
        "syn_to_train_nndr_median": syn_nndr_stats["median"],
        "syn_to_train_nndr_p01": syn_nndr_stats["p01"],
        "syn_to_train_nndr_p05": syn_nndr_stats["p05"],
        "syn_to_train_nndr_min": syn_nndr_stats["min"],

        "real_test_to_train_nndr_p01": te_nndr_stats["p01"],
        "carried_no_copy_status": no_copy_status,
        "TEST_real_values_used_for_privacy_reference": True,
        "synthetic_values_mutated": False,
    })

    # Top nearest synthetic-to-train neighbors.
    top_k = min(25, len(syn_d1_tr))
    top_idx = np.argsort(syn_d1_tr)[:top_k]
    train_idle = np.asarray(mats.get("train_idle", np.zeros(int(X_tr.shape[0]), dtype=bool)), dtype=bool)
    status_metric_scope = (
        "active_window_for_sparse_roles"
        if _is_protocol_role_152(role_name) or _is_binary_role_152(role_name)
        else "row"
    )

    for rank, si in enumerate(top_idx, start=1):
        train_neighbor_pos = int(syn_i1_tr[si]) if si < len(syn_i1_tr) else -1
        train_row = int(mats["train_idx"][train_neighbor_pos]) if 0 <= train_neighbor_pos < len(mats["train_idx"]) else -1
        syn_row = int(mats["syn_idx"][si]) if si < len(mats["syn_idx"]) else -1

        top_rows.append({
            "role_group": role_name,
            "rank": int(rank),
            "syn_sample_pos": int(si),
            "syn_row_or_window_start": syn_row,
            "nearest_train_sample_pos": train_neighbor_pos,
            "nearest_train_row_or_window_start": train_row,
            "dcr_to_train": float(syn_d1_tr[si]),
            "second_dcr_to_train": float(syn_d2_tr[si]),
            "nndr_to_train": float(syn_nndr_tr[si]),
            "real_test_train_p01_threshold": close_threshold,
            "is_close_neighbor": bool(np.isfinite(close_threshold) and syn_d1_tr[si] <= close_threshold),
            "representation": mats["representation"],
            "syn_is_idle_sample": bool(syn_idle[si]) if si < len(syn_idle) else False,
            "nearest_train_is_idle_sample": bool(train_idle[train_neighbor_pos]) if 0 <= train_neighbor_pos < len(train_idle) else False,
            "status_metric_scope": status_metric_scope,
            "selected_cols_n": int(len(cols)),
        })

# ----------------------------------------------------------
# 6) Save metrics
# ----------------------------------------------------------
metrics_df = pd.DataFrame(metrics_rows)
top_neighbors_df = pd.DataFrame(top_rows)
role_config_df = pd.DataFrame(role_config_rows)

if len(metrics_df) and "role_group" in metrics_df.columns:
    metrics_df["carried_no_copy_forensic_blocker"] = metrics_df["role_group"].astype(str).isin(no_copy_blocker_roles_152)
    metrics_df["carried_no_copy_forensic_warning"] = metrics_df["role_group"].astype(str).isin(no_copy_warning_roles_152)
    metrics_df["no_copy_forensic_blocker_roles"] = "|".join(no_copy_blocker_roles_152)
    metrics_df["no_copy_forensic_reason_counts"] = json.dumps(no_copy_reason_counts_152, sort_keys=True)
else:
    metrics_df["carried_no_copy_forensic_blocker"] = []
    metrics_df["carried_no_copy_forensic_warning"] = []
    metrics_df["no_copy_forensic_blocker_roles"] = []
    metrics_df["no_copy_forensic_reason_counts"] = []

metrics_df.to_csv(metrics_csv, index=False)
top_neighbors_df.to_csv(top_neighbors_csv, index=False)
role_config_df.to_csv(role_config_csv, index=False)

status_counts = metrics_df["status"].astype(str).value_counts().sort_index().to_dict() if len(metrics_df) else {}
blocker_n = int((metrics_df["status"].astype(str) == "blocker").sum()) if len(metrics_df) else 0
warning_n = int((metrics_df["status"].astype(str) == "warning").sum()) if len(metrics_df) else 0
pass_n = int((metrics_df["status"].astype(str) == "pass").sum()) if len(metrics_df) else 0
not_eval_n = int((metrics_df["status"].astype(str) == "not_evaluable").sum()) if len(metrics_df) else 0

# ----------------------------------------------------------
# 7) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "15.2",
    "version": CELL152_VERSION,
    "role": "q6_dcr_nndr_role_specific_privacy_audit",
    "quality_dimension": "Q6_privacy_no_copy_release_safety",
    "upstream_contract_versions": {
        "cell15_0": version150_152,
        "cell15_1": version151_152,
        "cell15_1a": version151a_152
    },
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False
    },
    "carried_no_copy_forensic": {
        "blocker_roles": no_copy_blocker_roles_152,
        "warning_roles": no_copy_warning_roles_152,
        "reason_counts": no_copy_reason_counts_152,
        "note": "DCR/NNDR adds evidence but does not clear carried no-copy blockers."
    },
    "summary": {
        "role_groups_total": int(len(metrics_df)),
        "pass_n": pass_n,
        "warning_n": warning_n,
        "blocker_n": blocker_n,
        "not_evaluable_n": not_eval_n,
        "status_counts": status_counts,
        "distance_metric": DISTANCE_METRIC_152,
        "sklearn_nn_available": bool(SKLEARN_NN_AVAILABLE_152),
        "active_window_roles": [
            "protocol_router",
            "protocol_ota",
            "protocol_zigbee",
            "iot_binary_states",
        ],
    },
    "method": {
        "DCR": (
            "Distance to closest real record/window. Main release-risk metric is synthetic TEST "
            "to real TRAIN, calibrated against real TEST to real TRAIN. Protocol and binary-state "
            "roles use active-window DCR so repeated idle/all-zero states are diagnostic rather "
            "than direct blocker evidence."
        ),
        "NNDR": (
            "Nearest-neighbor distance ratio d1/d2. Very low values indicate an isolated close "
            "neighbor and therefore higher copy risk. Protocol and binary-state roles use active "
            "windows for blocker-facing NNDR."
        ),
        "role_specificity": (
            "Protocol, IoT continuous, binary, driver, observability, and mixed roles are audited separately. "
            "Protocol roles use active windows; iot_binary_states uses transition-centered windows; "
            "idle exact matches are retained as diagnostics and not direct blockers."
        ),
        "idle_match_policy": "Exact/near-zero idle nearest-neighbor matches are diagnostic; blocker gates use non-idle active-window metrics for sparse protocol/binary roles.",
        "active_window_roles": [
            "protocol_router",
            "protocol_ota",
            "protocol_zigbee",
            "iot_binary_states",
        ],
        "privacy_scope": "Release-risk evidence, not a formal privacy proof.",
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "no_copy_forensic_carried_forward": True,
        "DCR_NNDR_does_not_clear_no_copy_blockers": True,
        "TEST_real_values_used_for_privacy_reference": True,
        "TEST_real_values_used_for_model_fitting": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "privacy_decision_done_here": False,
        "dcr_nndr_audit_done_here": True,
    },
    "outputs": {
        "metrics_csv": metrics_csv,
        "top_neighbors_csv": top_neighbors_csv,
        "role_config_csv": role_config_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

def _write_json_152(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_152(payload), f, indent=2, sort_keys=True)

_write_json_152(contract_json, contract)
_write_json_152(contract_canonical_json, contract)

manifest = {
    "cell": "15.2",
    "version": CELL152_VERSION,
    "created_outputs": contract["outputs"],
    "summary": contract["summary"],
    "q4_governance": contract["q4_governance"],
    "carried_no_copy_forensic": contract["carried_no_copy_forensic"],
    "strict_contract": contract["strict_contract"],
}

_write_json_152(manifest_json, manifest)

hashes = {
    "metrics_csv_sha256": _sha256_file_152(metrics_csv),
    "top_neighbors_csv_sha256": _sha256_file_152(top_neighbors_csv),
    "role_config_csv_sha256": _sha256_file_152(role_config_csv),
    "contract_json_sha256": _sha256_file_152(contract_json),
    "contract_canonical_json_sha256": _sha256_file_152(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_152(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_152(contract_json, contract)
_write_json_152(contract_canonical_json, contract)
_write_json_152(manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals
# ----------------------------------------------------------
globals()["CELL152_VERSION"] = CELL152_VERSION
globals()["CELL15_2_DCR_NNDR_ROLE_METRICS_DF"] = metrics_df
globals()["CELL15_2_DCR_NNDR_TOP_NEIGHBORS_DF"] = top_neighbors_df
globals()["CELL15_2_DCR_NNDR_ROLE_CONFIG_DF"] = role_config_df
globals()["CELL15_2_DCR_NNDR_CONTRACT"] = contract

globals()["CELL15_2_DCR_NNDR_ROLE_METRICS_CSV"] = metrics_csv
globals()["CELL15_2_DCR_NNDR_TOP_NEIGHBORS_CSV"] = top_neighbors_csv
globals()["CELL15_2_DCR_NNDR_ROLE_CONFIG_CSV"] = role_config_csv
globals()["CELL15_2_DCR_NNDR_CONTRACT_JSON"] = contract_json
globals()["CELL15_2_DCR_NNDR_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL15_2_DCR_NNDR_MANIFEST_JSON"] = manifest_json

log(
    "[Cell15.2] Q6 DCR/NNDR privacy audit complete | "
    f"roles={len(metrics_df)} | pass={pass_n} | warning={warning_n} | "
    f"blocker={blocker_n} | not_evaluable={not_eval_n}"
)
log(f"[Cell15.2] Status counts | {status_counts}")
log(f"[Cell15.2] Saved metrics: {metrics_csv} | rows={len(metrics_df)}")
log(f"[Cell15.2] Saved top neighbors: {top_neighbors_csv} | rows={len(top_neighbors_df)}")
log(f"[Cell15.2] Saved canonical contract: {contract_canonical_json}")
log(f"[Cell15.2] Carried no-copy forensic blocker roles | {no_copy_blocker_roles_152}")
log(
    "[Cell15.2] Contract flags | "
    "TEST_real_values_used_for_privacy_reference=True | "
    "TEST_real_values_used_for_model_fitting=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "privacy_decision_done_here=False | "
    "dcr_nndr_audit_done_here=True | Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False | no_copy_forensic_carried_forward=True"
)
log("--- END: Cell 15.2 - Q6 DCR / NNDR role-specific privacy audit (v1.2 no-copy-forensic-aware active-window) ---")

gc.collect()
