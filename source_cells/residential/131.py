# ==========================================================
# CELL 15.1 - Q6 no-copy window audit
# v1.1 STUDY-THESIS strict role-specific no-copy/privacy audit, no-Q4-aware and sparse-zero-safe
#
# Role:
#   - Audit whether final synthetic TEST windows appear to copy real windows.
#   - Uses role groups prepared by Cell 15.0.
#   - Compares synthetic TEST windows against real TRAIN / VAL / TEST windows.
#   - Reports exact-hash copy risk and near-copy similarity risk.
#
# Scientific contract:
#   - No fitting of generators.
#   - No synthetic mutation.
#   - TEST real values are used only as privacy/reference audit evidence.
#   - Exact all-zero sparse windows are audited separately and are not treated
#     as copy blockers by themselves.
#
# Outputs:
#   reports/cell15_1_no_copy_window_metrics.csv
#   reports/cell15_1_no_copy_top_matches.csv
#   reports/cell15_1_no_copy_role_config.csv
#   reports/cell15_1_no_copy_contract.json
#   artifacts/cell15_1_no_copy_manifest.json
# ==========================================================

log("--- START: Cell 15.1 - Q6 no-copy window audit (v1.1 no-Q4-aware sparse-zero-safe strict) ---")

import os
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_NN_AVAILABLE_151 = True
except Exception:
    SKLEARN_NN_AVAILABLE_151 = False

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_151 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "REAL_TRAIN_REF_150",
    "REAL_VAL_REF_150",
    "REAL_TEST_REF_150",
    "SYN_TEST_REF_150",
    "CELL15_0_PRIVACY_ROLE_GROUPS",
    "CELL15_0_PRIVACY_ROLE_GROUP_REGISTRY_DF",
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
]
_missing_151 = [k for k in _required_151 if k not in globals()]
if _missing_151:
    raise RuntimeError(f"[Cell15.1] Missing required globals: {_missing_151}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)

REAL_TRAIN_151 = REAL_TRAIN_REF_150
REAL_VAL_151 = REAL_VAL_REF_150
REAL_TEST_151 = REAL_TEST_REF_150
SYN_TEST_151 = SYN_TEST_REF_150

N_TR = int(len(REAL_TRAIN_151))
N_VAL = int(len(REAL_VAL_151))
N_TE = int(len(REAL_TEST_151))
N_SYN = int(len(SYN_TEST_151))

CELL151_VERSION = "cell15_1_q6_no_copy_window_audit_v1_1_no_q4_sparse_zero_safe"

CFG["cell15_1_version"] = CELL151_VERSION
CFG["cell15_1_TEST_real_values_used_for_privacy_reference"] = True
CFG["cell15_1_synthetic_values_mutated"] = False
CFG["cell15_1_selection_done_here"] = False
CFG["cell15_1_generator_fit_done_here"] = False
CFG["cell15_1_materialization_done_here"] = False
CFG["cell15_1_privacy_decision_done_here"] = False
CFG["cell15_1_no_copy_audit_done_here"] = True
CFG["cell15_1_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_1_Q4_coupled_artifacts_used"] = False
CFG["cell15_1_sparse_all_zero_windows_not_blockers"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell15_1_window_len_default", 60)
CFG.setdefault("cell15_1_window_len_sparse_driver", 300)
CFG.setdefault("cell15_1_window_len_observability", 120)
CFG.setdefault("cell15_1_window_len_protocol", 60)
CFG.setdefault("cell15_1_window_stride_default", 10)
CFG.setdefault("cell15_1_window_stride_sparse_driver", 30)
CFG.setdefault("cell15_1_max_cols_per_role", 80)
CFG.setdefault("cell15_1_max_real_windows_per_ref", 2500)
CFG.setdefault("cell15_1_max_syn_windows", 2500)
CFG.setdefault("cell15_1_exact_round_decimals", 6)
CFG.setdefault("cell15_1_near_copy_similarity_threshold", 0.995)
CFG.setdefault("cell15_1_high_similarity_threshold", 0.990)
CFG.setdefault("cell15_1_blocker_exact_nonzero_match_rate", 0.005)
CFG.setdefault("cell15_1_blocker_near_copy_rate", 0.05)
CFG.setdefault("cell15_1_warning_near_copy_rate", 0.01)
CFG.setdefault("cell15_1_warning_p99_similarity", 0.990)
CFG.setdefault("cell15_1_sparse_near_copy_gate_uses_nonzero_windows", True)
CFG.setdefault("cell15_1_min_common_cols", 1)
CFG.setdefault("cell15_1_low_variance_eps", 1e-12)
CFG.setdefault("cell15_1_fill_value", -999999.0)
CFG.setdefault("cell15_1_clip_z", 20.0)
CFG.setdefault("cell15_1_skip_roles", ["iot_placeholders_or_excluded"])

WINDOW_DEFAULT_151 = int(CFG.get("cell15_1_window_len_default", 60))
STRIDE_DEFAULT_151 = int(CFG.get("cell15_1_window_stride_default", 10))
MAX_COLS_151 = int(CFG.get("cell15_1_max_cols_per_role", 80))
MAX_REAL_WINDOWS_151 = int(CFG.get("cell15_1_max_real_windows_per_ref", 2500))
MAX_SYN_WINDOWS_151 = int(CFG.get("cell15_1_max_syn_windows", 2500))
ROUND_DECIMALS_151 = int(CFG.get("cell15_1_exact_round_decimals", 6))
NEAR_COPY_THRESHOLD_151 = float(CFG.get("cell15_1_near_copy_similarity_threshold", 0.995))
HIGH_SIM_THRESHOLD_151 = float(CFG.get("cell15_1_high_similarity_threshold", 0.990))
FILL_VALUE_151 = float(CFG.get("cell15_1_fill_value", -999999.0))
CLIP_Z_151 = float(CFG.get("cell15_1_clip_z", 20.0))
SKIP_ROLES_151 = set(map(str, CFG.get("cell15_1_skip_roles", ["iot_placeholders_or_excluded"])))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
metrics_csv = os.path.join(REPORT_DIR, "cell15_1_no_copy_window_metrics.csv")
top_matches_csv = os.path.join(REPORT_DIR, "cell15_1_no_copy_top_matches.csv")
role_config_csv = os.path.join(REPORT_DIR, "cell15_1_no_copy_role_config.csv")
contract_json = os.path.join(REPORT_DIR, "cell15_1_no_copy_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell15_1_no_copy_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell15_1_no_copy_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_151(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_151(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_151(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_151(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_151(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_151(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_151(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_151(obj.to_dict())
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

def _write_json_151(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_151(payload), f, indent=2, sort_keys=True)

def _sha256_file_151(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_151(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.1] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.1] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

def _dedup_151(seq):
    seen = set()
    out = []
    for x in seq:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _window_len_for_role_151(role: str) -> int:
    role = str(role)
    if role == "iot_event_drivers":
        return int(CFG.get("cell15_1_window_len_sparse_driver", 300))
    if role == "iot_observability_masks":
        return int(CFG.get("cell15_1_window_len_observability", 120))
    if role.startswith("protocol_"):
        return int(CFG.get("cell15_1_window_len_protocol", 60))
    return WINDOW_DEFAULT_151

def _stride_for_role_151(role: str) -> int:
    role = str(role)
    if role == "iot_event_drivers":
        return int(CFG.get("cell15_1_window_stride_sparse_driver", 30))
    return STRIDE_DEFAULT_151

def _available_common_cols_151(cols):
    cols = _dedup_151(cols)
    return [
        c for c in cols
        if c in REAL_TRAIN_151.columns
        and c in REAL_VAL_151.columns
        and c in REAL_TEST_151.columns
        and c in SYN_TEST_151.columns
    ]

def _choose_cols_151(role: str, cols: list):
    cols = _available_common_cols_151(cols)
    if not cols:
        return []

    # Prefer columns with useful finite rate and nonzero variance in synthetic
    # and real train. This keeps no-copy vectors informative.
    rows = []
    for c in cols:
        x_syn = pd.to_numeric(SYN_TEST_151[c], errors="coerce")
        x_tr = pd.to_numeric(REAL_TRAIN_151[c], errors="coerce")

        syn_finite = float(x_syn.notna().mean()) if len(x_syn) else 0.0
        tr_finite = float(x_tr.notna().mean()) if len(x_tr) else 0.0

        syn_std = float(np.nanstd(x_syn.to_numpy(dtype=np.float64))) if x_syn.notna().any() else 0.0
        tr_std = float(np.nanstd(x_tr.to_numpy(dtype=np.float64))) if x_tr.notna().any() else 0.0

        rows.append({
            "col": c,
            "score": min(syn_finite, tr_finite) * np.log1p(max(syn_std, tr_std)),
            "syn_finite": syn_finite,
            "tr_finite": tr_finite,
            "syn_std": syn_std,
            "tr_std": tr_std,
        })

    d = pd.DataFrame(rows)
    d["score"] = pd.to_numeric(d["score"], errors="coerce").fillna(0.0)

    # For sparse drivers, keep all if below cap, even if low variance.
    if role == "iot_event_drivers":
        chosen = d.sort_values(["score", "col"], ascending=[False, True])["col"].head(MAX_COLS_151).astype(str).tolist()
    else:
        eps = float(CFG.get("cell15_1_low_variance_eps", 1e-12))
        dd = d[
            (pd.to_numeric(d["syn_finite"], errors="coerce").fillna(0.0) > 0.001)
            & (pd.to_numeric(d["tr_finite"], errors="coerce").fillna(0.0) > 0.001)
            & (
                (pd.to_numeric(d["syn_std"], errors="coerce").fillna(0.0) > eps)
                | (pd.to_numeric(d["tr_std"], errors="coerce").fillna(0.0) > eps)
            )
        ].copy()

        if len(dd) == 0:
            dd = d.copy()

        chosen = dd.sort_values(["score", "col"], ascending=[False, True])["col"].head(MAX_COLS_151).astype(str).tolist()

    return chosen

def _sample_window_starts_151(n_rows: int, window_len: int, stride: int, max_windows: int, seed_offset: int):
    if n_rows < window_len:
        return np.asarray([], dtype=np.int64)

    starts = np.arange(0, n_rows - window_len + 1, max(1, stride), dtype=np.int64)

    if starts.size <= max_windows:
        return starts

    rng = np.random.default_rng(SEED + seed_offset)
    pick = rng.choice(starts, size=max_windows, replace=False)
    return np.asarray(np.sort(pick), dtype=np.int64)

def _fit_scaler_151(cols):
    # Robust scaler from real TRAIN only.
    med = []
    iqr = []
    for c in cols:
        x = pd.to_numeric(REAL_TRAIN_151[c], errors="coerce").to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            med.append(0.0)
            iqr.append(1.0)
            continue
        q25, q50, q75 = np.nanquantile(x, [0.25, 0.50, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.nanstd(x))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        med.append(float(q50))
        iqr.append(float(scale))
    return np.asarray(med, dtype=np.float64), np.asarray(iqr, dtype=np.float64)

def _make_window_matrix_151(frame: pd.DataFrame, cols: list, starts: np.ndarray, window_len: int, med: np.ndarray, scale: np.ndarray):
    if len(cols) == 0 or len(starts) == 0:
        return np.empty((0, 0), dtype=np.float32), np.asarray([], dtype=bool)

    arr = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    out = np.empty((len(starts), window_len * len(cols)), dtype=np.float32)
    all_zero_like = np.zeros(len(starts), dtype=bool)

    med = med.reshape(1, -1)
    scale = scale.reshape(1, -1)

    for i, s in enumerate(starts):
        w = arr[int(s):int(s) + window_len, :]
        finite = np.isfinite(w)
        raw_nonzero = np.nan_to_num(w, nan=0.0)
        all_zero_like[i] = bool(np.all(raw_nonzero == 0.0))

        z = (w - med) / scale
        z = np.where(np.isfinite(z), z, FILL_VALUE_151)
        z = np.clip(z, -CLIP_Z_151, CLIP_Z_151)
        out[i, :] = z.reshape(-1).astype(np.float32)

    return out, all_zero_like

def _hash_windows_151(mat: np.ndarray):
    if mat.size == 0 or mat.shape[0] == 0:
        return []
    q = np.round(mat.astype(np.float64), ROUND_DECIMALS_151).astype(np.float32)
    hashes = []
    for i in range(q.shape[0]):
        h = hashlib.sha256(q[i].tobytes()).hexdigest()
        hashes.append(h)
    return hashes

def _nearest_similarity_151(real_mat: np.ndarray, syn_mat: np.ndarray):
    if real_mat.shape[0] == 0 or syn_mat.shape[0] == 0 or real_mat.shape[1] == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64)

    # Cosine similarity on robust-scaled windows.
    if SKLEARN_NN_AVAILABLE_151:
        nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="auto")
        nn.fit(real_mat)
        dist, idx = nn.kneighbors(syn_mat, return_distance=True)
        sim = 1.0 - dist.reshape(-1)
        return sim.astype(np.float64), idx.reshape(-1).astype(np.int64)

    # Fallback: chunked cosine.
    r = real_mat.astype(np.float64)
    s = syn_mat.astype(np.float64)

    r_norm = np.linalg.norm(r, axis=1)
    s_norm = np.linalg.norm(s, axis=1)
    r = r / np.maximum(r_norm[:, None], 1e-12)
    s = s / np.maximum(s_norm[:, None], 1e-12)

    best_sim = np.full(s.shape[0], -np.inf, dtype=np.float64)
    best_idx = np.full(s.shape[0], -1, dtype=np.int64)

    chunk = 256
    for start in range(0, s.shape[0], chunk):
        end = min(s.shape[0], start + chunk)
        dots = s[start:end] @ r.T
        idx = np.argmax(dots, axis=1)
        val = dots[np.arange(end - start), idx]
        best_sim[start:end] = val
        best_idx[start:end] = idx

    return best_sim, best_idx

def _status_from_metrics_151(exact_nonzero_rate, near_rate_for_gate, p99_sim_for_gate):
    """
    Gates intentionally use nonzero synthetic windows when configured. This avoids
    flagging legitimate repeated all-zero sparse windows as privacy copy blockers.
    All-window rates are still reported diagnostically.
    """
    reasons = []

    if np.isfinite(exact_nonzero_rate) and exact_nonzero_rate > float(CFG.get("cell15_1_blocker_exact_nonzero_match_rate", 0.005)):
        reasons.append("exact_nonzero_window_copy_rate_blocker")

    if np.isfinite(near_rate_for_gate) and near_rate_for_gate > float(CFG.get("cell15_1_blocker_near_copy_rate", 0.05)):
        reasons.append("near_copy_nonzero_window_rate_blocker")

    if reasons:
        return "blocker", "|".join(reasons)

    warn = []
    if np.isfinite(near_rate_for_gate) and near_rate_for_gate > float(CFG.get("cell15_1_warning_near_copy_rate", 0.01)):
        warn.append("near_copy_nonzero_window_rate_warning")

    if np.isfinite(p99_sim_for_gate) and p99_sim_for_gate > float(CFG.get("cell15_1_warning_p99_similarity", 0.990)):
        warn.append("p99_nonzero_similarity_warning")

    if warn:
        return "warning", "|".join(warn)

    return "pass", "within_no_copy_window_gates"

# ----------------------------------------------------------
# 4) Validate Cell 15.0 no-Q4 privacy input contract
# ----------------------------------------------------------
version150_151 = _require_contract_version_151(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
strict150_151 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
q4gov150_151 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("q4_governance", {})

if str(strict150_151.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.1] Cell 15.0 does not declare Q4_final_status=blocked_no_promotion.")
if bool(strict150_151.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.1] Cell 15.0 used Q4-coupled artifacts; refusing Q6 no-copy audit.")
if bool(strict150_151.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell15.1] Cell 15.0 contract indicates synthetic mutation.")
if bool(strict150_151.get("privacy_decision_done_here", True)):
    raise RuntimeError("[Cell15.1] Cell 15.0 should only prepare inputs, not make privacy decisions.")

# ----------------------------------------------------------
# 5) Run no-copy audit by role group
# ----------------------------------------------------------
role_groups = CELL15_0_PRIVACY_ROLE_GROUPS

metrics_rows = []
top_match_rows = []
role_config_rows = []

roles_to_run = [r for r in role_groups.keys() if r not in SKIP_ROLES_151]

log(
    "[Cell15.1] Running no-copy window audit | "
    f"roles={roles_to_run} | sklearn_nn={SKLEARN_NN_AVAILABLE_151}"
)

for role_idx, role_name in enumerate(roles_to_run, start=1):
    raw_cols = role_groups.get(role_name, [])
    cols = _choose_cols_151(role_name, raw_cols)

    window_len = _window_len_for_role_151(role_name)
    stride = _stride_for_role_151(role_name)

    role_config_rows.append({
        "role_group": role_name,
        "raw_cols_n": int(len(raw_cols)),
        "common_selected_cols_n": int(len(cols)),
        "window_len": int(window_len),
        "stride": int(stride),
        "max_real_windows_per_ref": int(MAX_REAL_WINDOWS_151),
        "max_syn_windows": int(MAX_SYN_WINDOWS_151),
        "near_copy_similarity_threshold": float(NEAR_COPY_THRESHOLD_151),
        "round_decimals_for_hash": int(ROUND_DECIMALS_151),
    })

    if len(cols) < int(CFG.get("cell15_1_min_common_cols", 1)):
        metrics_rows.append({
            "role_group": role_name,
            "status": "not_evaluable",
            "reasons": "insufficient_common_columns",
            "selected_cols_n": int(len(cols)),
            "window_len": int(window_len),
            "syn_windows_n": 0,
            "real_windows_total_n": 0,
            "exact_match_n": 0,
            "exact_match_rate": np.nan,
            "exact_nonzero_match_n": 0,
            "exact_nonzero_match_rate": np.nan,
            "near_copy_rate": np.nan,
            "max_similarity": np.nan,
            "p99_similarity": np.nan,
            "mean_similarity": np.nan,
            "TEST_real_values_used_for_privacy_reference": True,
            "synthetic_values_mutated": False,
        })
        continue

    log(
        f"[Cell15.1] role {role_idx}/{len(roles_to_run)} | "
        f"{role_name} | cols={len(cols)} | window_len={window_len}"
    )

    med, scale = _fit_scaler_151(cols)

    syn_starts = _sample_window_starts_151(
        N_SYN,
        window_len,
        stride,
        MAX_SYN_WINDOWS_151,
        seed_offset=15100 + role_idx,
    )

    real_refs = [
        ("real_train", REAL_TRAIN_151, N_TR, 15110 + role_idx),
        ("real_val", REAL_VAL_151, N_VAL, 15120 + role_idx),
        ("real_test", REAL_TEST_151, N_TE, 15130 + role_idx),
    ]

    syn_mat, syn_all_zero = _make_window_matrix_151(
        SYN_TEST_151,
        cols,
        syn_starts,
        window_len,
        med,
        scale,
    )
    syn_hashes = _hash_windows_151(syn_mat)

    real_mats = []
    real_hash_sets = {}
    real_hash_source = {}
    real_start_lookup = []

    for ref_name, ref_frame, ref_n, seed_offset in real_refs:
        starts = _sample_window_starts_151(
            ref_n,
            window_len,
            stride,
            MAX_REAL_WINDOWS_151,
            seed_offset=seed_offset,
        )

        mat, all_zero = _make_window_matrix_151(
            ref_frame,
            cols,
            starts,
            window_len,
            med,
            scale,
        )

        hashes = _hash_windows_151(mat)
        hset = set(hashes)
        real_hash_sets[ref_name] = hset

        for j, h in enumerate(hashes):
            if h not in real_hash_source:
                real_hash_source[h] = {
                    "real_ref": ref_name,
                    "real_window_idx": int(j),
                    "real_start": int(starts[j]) if j < len(starts) else -1,
                    "real_all_zero": bool(all_zero[j]) if j < len(all_zero) else False,
                }

        if mat.shape[0]:
            real_mats.append(mat)
            for j, s in enumerate(starts):
                real_start_lookup.append({
                    "real_ref": ref_name,
                    "real_window_idx": int(j),
                    "real_start": int(s),
                    "real_all_zero": bool(all_zero[j]) if j < len(all_zero) else False,
                })

    if real_mats:
        real_mat_all = np.vstack(real_mats)
    else:
        real_mat_all = np.empty((0, syn_mat.shape[1] if syn_mat.ndim == 2 else 0), dtype=np.float32)

    real_hash_all = set()
    for hset in real_hash_sets.values():
        real_hash_all |= hset

    exact_match_flags = np.asarray([h in real_hash_all for h in syn_hashes], dtype=bool)
    exact_match_n = int(exact_match_flags.sum())
    exact_match_rate = float(exact_match_n / max(1, len(syn_hashes)))

    exact_nonzero_match_flags = exact_match_flags & (~syn_all_zero)
    exact_nonzero_match_n = int(exact_nonzero_match_flags.sum())
    nonzero_syn_n = int((~syn_all_zero).sum())
    exact_nonzero_match_rate = float(exact_nonzero_match_n / max(1, nonzero_syn_n))

    sim, nn_idx = _nearest_similarity_151(real_mat_all, syn_mat)

    near_copy_flags = sim >= NEAR_COPY_THRESHOLD_151 if sim.size else np.asarray([], dtype=bool)
    near_copy_rate = float(np.mean(near_copy_flags)) if near_copy_flags.size else np.nan
    max_similarity = float(np.max(sim)) if sim.size else np.nan
    p99_similarity = float(np.quantile(sim, 0.99)) if sim.size else np.nan
    mean_similarity = float(np.mean(sim)) if sim.size else np.nan

    if sim.size and len(syn_all_zero) == len(sim):
        nonzero_mask_for_gate = ~syn_all_zero
        sim_nonzero = sim[nonzero_mask_for_gate]
        near_copy_nonzero_flags = near_copy_flags[nonzero_mask_for_gate]
    else:
        nonzero_mask_for_gate = np.asarray([], dtype=bool)
        sim_nonzero = np.asarray([], dtype=np.float64)
        near_copy_nonzero_flags = np.asarray([], dtype=bool)

    near_copy_nonzero_n = int(near_copy_nonzero_flags.sum()) if near_copy_nonzero_flags.size else 0
    near_copy_nonzero_rate = (
        float(near_copy_nonzero_n / max(1, nonzero_syn_n))
        if nonzero_syn_n > 0 else np.nan
    )
    p99_nonzero_similarity = float(np.quantile(sim_nonzero, 0.99)) if sim_nonzero.size else np.nan
    max_nonzero_similarity = float(np.max(sim_nonzero)) if sim_nonzero.size else np.nan
    mean_nonzero_similarity = float(np.mean(sim_nonzero)) if sim_nonzero.size else np.nan

    if bool(CFG.get("cell15_1_sparse_near_copy_gate_uses_nonzero_windows", True)):
        near_rate_for_gate = near_copy_nonzero_rate
        p99_for_gate = p99_nonzero_similarity
    else:
        near_rate_for_gate = near_copy_rate
        p99_for_gate = p99_similarity

    status, reasons = _status_from_metrics_151(
        exact_nonzero_match_rate,
        near_rate_for_gate,
        p99_for_gate,
    )

    metrics_rows.append({
        "role_group": role_name,
        "status": status,
        "reasons": reasons,
        "selected_cols_n": int(len(cols)),
        "raw_cols_n": int(len(raw_cols)),
        "window_len": int(window_len),
        "stride": int(stride),
        "syn_windows_n": int(len(syn_starts)),
        "real_windows_total_n": int(real_mat_all.shape[0]),
        "syn_all_zero_windows_n": int(syn_all_zero.sum()),
        "syn_nonzero_windows_n": int(nonzero_syn_n),
        "exact_match_n": int(exact_match_n),
        "exact_match_rate": exact_match_rate,
        "exact_nonzero_match_n": int(exact_nonzero_match_n),
        "exact_nonzero_match_rate": exact_nonzero_match_rate,
        "near_copy_threshold": float(NEAR_COPY_THRESHOLD_151),
        "near_copy_n": int(near_copy_flags.sum()) if near_copy_flags.size else 0,
        "near_copy_rate_all_windows": near_copy_rate,
        "near_copy_nonzero_n": int(near_copy_nonzero_n),
        "near_copy_nonzero_rate": near_copy_nonzero_rate,
        "near_copy_rate_used_for_gate": near_rate_for_gate,
        "sparse_all_zero_windows_not_blockers": True,
        "high_similarity_threshold": float(HIGH_SIM_THRESHOLD_151),
        "high_similarity_n": int((sim >= HIGH_SIM_THRESHOLD_151).sum()) if sim.size else 0,
        "max_similarity": max_similarity,
        "p99_similarity_all_windows": p99_similarity,
        "mean_similarity_all_windows": mean_similarity,
        "max_nonzero_similarity": max_nonzero_similarity,
        "p99_nonzero_similarity": p99_nonzero_similarity,
        "mean_nonzero_similarity": mean_nonzero_similarity,
        "p99_similarity_used_for_gate": p99_for_gate,
        "sklearn_nn_available": bool(SKLEARN_NN_AVAILABLE_151),
        "TEST_real_values_used_for_privacy_reference": True,
        "synthetic_values_mutated": False,
    })

    # Top matches for audit.
    if sim.size:
        top_k = min(25, len(sim))
        top_idx = np.argsort(-sim)[:top_k]
        for rank, si in enumerate(top_idx, start=1):
            ni = int(nn_idx[si]) if nn_idx.size else -1
            real_info = real_start_lookup[ni] if 0 <= ni < len(real_start_lookup) else {}
            syn_hash = syn_hashes[si] if si < len(syn_hashes) else ""
            exact_info = real_hash_source.get(syn_hash, {})

            top_match_rows.append({
                "role_group": role_name,
                "rank": int(rank),
                "similarity": float(sim[si]),
                "near_copy_threshold": float(NEAR_COPY_THRESHOLD_151),
                "is_near_copy": bool(sim[si] >= NEAR_COPY_THRESHOLD_151),
                "is_exact_hash_match": bool(exact_match_flags[si]) if si < len(exact_match_flags) else False,
                "is_syn_all_zero_window": bool(syn_all_zero[si]) if si < len(syn_all_zero) else False,
                "syn_window_idx": int(si),
                "syn_start": int(syn_starts[si]) if si < len(syn_starts) else -1,
                "nearest_real_global_idx": int(ni),
                "nearest_real_ref": str(real_info.get("real_ref", "")),
                "nearest_real_start": int(real_info.get("real_start", -1)),
                "nearest_real_all_zero": bool(real_info.get("real_all_zero", False)),
                "exact_hash_real_ref": str(exact_info.get("real_ref", "")),
                "exact_hash_real_start": int(exact_info.get("real_start", -1)) if exact_info else -1,
                "selected_cols_n": int(len(cols)),
                "window_len": int(window_len),
            })

# ----------------------------------------------------------
# 6) Save metrics
# ----------------------------------------------------------
metrics_df = pd.DataFrame(metrics_rows)
top_matches_df = pd.DataFrame(top_match_rows)
role_config_df = pd.DataFrame(role_config_rows)

metrics_df.to_csv(metrics_csv, index=False)
top_matches_df.to_csv(top_matches_csv, index=False)
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
    "cell": "15.1",
    "version": CELL151_VERSION,
    "role": "q6_no_copy_window_audit",
    "quality_dimension": "Q6_privacy_no_copy_release_safety",
    "upstream_contract_versions": {
        "cell15_0": version150_151
    },
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "source_cell15_0_q4_governance": q4gov150_151
    },
    "summary": {
        "role_groups_total": int(len(metrics_df)),
        "pass_n": pass_n,
        "warning_n": warning_n,
        "blocker_n": blocker_n,
        "not_evaluable_n": not_eval_n,
        "status_counts": status_counts,
        "near_copy_similarity_threshold": float(NEAR_COPY_THRESHOLD_151),
        "exact_round_decimals": int(ROUND_DECIMALS_151),
        "sklearn_nn_available": bool(SKLEARN_NN_AVAILABLE_151),
        "sparse_all_zero_windows_not_blockers": True,
        "near_copy_gate_uses_nonzero_windows": bool(CFG.get("cell15_1_sparse_near_copy_gate_uses_nonzero_windows", True)),
    },
    "method": {
        "exact_hash": (
            "Robust-scaled windows are rounded and SHA256-hashed. "
            "Exact all-zero sparse windows are tracked separately and are not blockers by themselves."
        ),
        "near_copy": (
            "Nearest-neighbor cosine similarity is computed between synthetic TEST windows "
            "and real TRAIN/VAL/TEST windows within each role group. Copy gates use nonzero "
            "synthetic windows when configured so repeated all-zero sparse windows are not "
            "treated as copy blockers."
        ),
        "privacy_scope": (
            "This is a release-risk audit, not a formal privacy proof."
        ),
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_for_privacy_reference": True,
        "TEST_real_values_used_for_model_fitting": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "privacy_decision_done_here": False,
        "no_copy_window_audit_done_here": True,
    },
    "outputs": {
        "metrics_csv": metrics_csv,
        "top_matches_csv": top_matches_csv,
        "role_config_csv": role_config_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_151(contract_json, contract)
_write_json_151(contract_canonical_json, contract)

manifest = {
    "cell": "15.1",
    "version": CELL151_VERSION,
    "created_outputs": contract["outputs"],
    "summary": contract["summary"],
    "q4_governance": contract["q4_governance"],
    "strict_contract": contract["strict_contract"],
}

_write_json_151(manifest_json, manifest)

hashes = {
    "metrics_csv_sha256": _sha256_file_151(metrics_csv),
    "top_matches_csv_sha256": _sha256_file_151(top_matches_csv),
    "role_config_csv_sha256": _sha256_file_151(role_config_csv),
    "contract_json_sha256": _sha256_file_151(contract_json),
    "contract_canonical_json_sha256": _sha256_file_151(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_151(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_151(contract_json, contract)
_write_json_151(contract_canonical_json, contract)
_write_json_151(manifest_json, manifest)

# ----------------------------------------------------------
# 8) Export globals
# ----------------------------------------------------------
globals()["CELL151_VERSION"] = CELL151_VERSION
globals()["CELL15_1_NO_COPY_WINDOW_METRICS_DF"] = metrics_df
globals()["CELL15_1_NO_COPY_TOP_MATCHES_DF"] = top_matches_df
globals()["CELL15_1_NO_COPY_ROLE_CONFIG_DF"] = role_config_df
globals()["CELL15_1_NO_COPY_CONTRACT"] = contract

globals()["CELL15_1_NO_COPY_WINDOW_METRICS_CSV"] = metrics_csv
globals()["CELL15_1_NO_COPY_TOP_MATCHES_CSV"] = top_matches_csv
globals()["CELL15_1_NO_COPY_ROLE_CONFIG_CSV"] = role_config_csv
globals()["CELL15_1_NO_COPY_CONTRACT_JSON"] = contract_json
globals()["CELL15_1_NO_COPY_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL15_1_NO_COPY_MANIFEST_JSON"] = manifest_json

log(
    "[Cell15.1] Q6 no-copy window audit complete | "
    f"roles={len(metrics_df)} | pass={pass_n} | warning={warning_n} | "
    f"blocker={blocker_n} | not_evaluable={not_eval_n}"
)
log(f"[Cell15.1] Status counts | {status_counts}")
log(f"[Cell15.1] Saved metrics: {metrics_csv} | rows={len(metrics_df)}")
log(f"[Cell15.1] Saved top matches: {top_matches_csv} | rows={len(top_matches_df)}")
log(f"[Cell15.1] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell15.1] Contract flags | "
    "TEST_real_values_used_for_privacy_reference=True | "
    "TEST_real_values_used_for_model_fitting=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "privacy_decision_done_here=False | "
    "no_copy_window_audit_done_here=True | Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False | sparse_all_zero_windows_not_blockers=True"
)
log("--- END: Cell 15.1 - Q6 no-copy window audit (v1.1 no-Q4-aware sparse-zero-safe strict) ---")

gc.collect()