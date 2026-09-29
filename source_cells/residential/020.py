# ============================================================
# CELL 8 — DDPM candidate generator training — v18-THESIS
# STUDY-THESIS / HYBRID-PORTFOLIO / ADVISORY-ONLY DDPM CANDIDATES
#
# Portfolio role:
# - A1 temporal block-bootstrap is the default backbone for all protocol columns.
# - DDPM is only one predefined candidate family for dense/semi-dense protocol
#   values and conditional refinement.
# - This cell trains DDPM and exports candidate-generation utilities/artifacts.
# - This cell does NOT perform final candidate selection.
# - A downstream VAL-only selector is the authoritative selector against true A1
#   and all eligible generator families.
#
# Purpose:
# - Train SeqDenoiser DDPM on legitimate Cell 6 DDPM_VALUE_COLS only.
# - Use Cell 7 shared CPS conditioning exactly.
# - Use strict value-finite x_obs:
#       x_obs = Cell 6 logical DDPM tier obs mask AND isfinite(raw target value)
# - Use absolute protocol-aware transformed targets.
# - Generate DDPM-derived VAL candidates for advisory diagnostics only.
# - Export a DDPM candidate registry for the downstream VAL-only selector.
#
# Candidate families exported:
#   - ddpm_absolute
#   - ddpm_train_marginal_anchored
#   - ddpm_tail_repaired
#   - ddpm_sparse_rate_aligned
#   - ddpm_empirical_blend_light
#
# Scientific policy:
# - TEST is never used in training, fitting, calibration, candidate generation,
#   diagnostics, or model selection.
# - VAL diagnostics in this cell are advisory only.
# - This cell does not claim DDPM beats A1.
# - Final keep/drop criterion belongs to the downstream VAL-only selector:
#       candidate must beat true A1 on VAL by predefined thresholds.
# - If DDPM produces zero advisory columns, that is a valid diagnostic outcome.
# ============================================================

log("--- START: Cell 8 — DDPM candidate generator training (v18-THESIS advisory-only) ---")

import os
import re
import json
import math
import time
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from torch.utils.data import DataLoader


# ------------------------------------------------------------
# 0) Required globals
# ------------------------------------------------------------
need = [
    "CFG", "log",
    "df_tr", "df_va",
    "DDPM_VALUE_COLS", "DDPM_TIERS_USED",
    "DDPM_OBS_TR", "DDPM_OBS_VA",
    "cond_tr", "cond_va",
    "reg_tr", "reg_va",
    "tod_tr", "tod_va",
    "SequenceDataset", "DDPM",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell8] Missing prerequisites: {missing}. Run Cells 1–7 first.")

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
REP_DIR = os.path.join(OUTDIR, "reports")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Clean-run leakage guard inherited from Cell 1
# ------------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(
            f"[Cell8] Clean DDPM candidate training forbids CFG['{_flag}']=True."
        )

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell8] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

# Cell 8 is TRAIN/VAL only. Stale TEST-side DDPM globals indicate a dirty kernel.
if any(k in globals() for k in ["Xte_ddpm_candidate", "DDPM_TEST_DIAGNOSTICS", "DDPM_TEST_CANDIDATES"]):
    raise RuntimeError(
        "[Cell8] Found stale TEST-side DDPM candidate globals. Restart kernel or delete stale TEST globals."
    )

seed = int(CFG.get("seed", 1337))
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# 0b) Reproducibility and CUDA policy
# ------------------------------------------------------------
deterministic_warn_only = False
try:
    torch.use_deterministic_algorithms(True, warn_only=deterministic_warn_only)
except Exception as e:
    log(f"[Cell8][WARN] Could not enable deterministic algorithms: {type(e).__name__}: {e}")

allow_flash_attention = bool(CFG.get("torch_allow_flash_attention", False))
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(bool(allow_flash_attention))
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception as e:
        log(f"[Cell8][WARN] Could not configure CUDA SDP kernels: {type(e).__name__}: {e}")


# ------------------------------------------------------------
# 0c) Device resolution
# ------------------------------------------------------------
def _resolve_torch_device_from_cfg() -> str:
    requested = str(CFG.get("device", "auto")).strip().lower()

    if requested in {"cpu", "none"}:
        return "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            log(f"[Cell8] Requested device={requested}, but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        return requested

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


device = _resolve_torch_device_from_cfg()
device_obj = torch.device(device)

if device_obj.type == "cuda":
    torch.backends.cudnn.benchmark = bool(CFG.get("torch_cudnn_benchmark", False))

    # Keep default False to remain consistent with Cell 1 reproducibility policy.
    torch_allow_tf32_effective = bool(CFG.get("torch_allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = torch_allow_tf32_effective
    torch.backends.cudnn.allow_tf32 = torch_allow_tf32_effective

    try:
        gpu_name = torch.cuda.get_device_name(device_obj)
    except Exception:
        gpu_name = "cuda"

    log(
        f"[Cell8] Using GPU device={device} | gpu={gpu_name} | "
        f"cuda_mem_allocated={torch.cuda.memory_allocated(device_obj) / (1024**2):.1f} MiB | "
        f"tf32_allowed={torch_allow_tf32_effective}"
    )
else:
    torch_allow_tf32_effective = False
    log("[Cell8] Using CPU. Set CFG['device']='cuda' or CFG['device']='auto' with CUDA available to train on GPU.")



DDPM_COLS = list(globals()["DDPM_VALUE_COLS"])

globals()["ddpm_legacy"] = globals().get("ddpm", None)
globals()["ddpm_model_legacy"] = globals().get("ddpm_model", None)


# ------------------------------------------------------------
# 1) Helpers
# ------------------------------------------------------------
def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
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


def _write_json(path: str, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2)
    os.replace(tmp, path)


def _check_unique(name: str, cols: list):
    if len(cols) != len(set(cols)):
        vals, cnts = np.unique(np.asarray(cols, dtype=object), return_counts=True)
        dupes = [str(v) for v, c in zip(vals, cnts) if c > 1]
        raise RuntimeError(f"[Cell8] Duplicate entries in {name}: {dupes[:20]}")


def _sha256_pipejoin(xs) -> str:
    return hashlib.sha256(("||".join(map(str, xs))).encode("utf-8")).hexdigest()


def _state_dict_all_finite(sd: dict, label: str):
    bad = []
    for k, v in sd.items():
        if not torch.is_tensor(v):
            bad.append((k, "non_tensor"))
            continue
        if torch.is_floating_point(v) or torch.is_complex(v):
            if not torch.isfinite(v).all().item():
                bad.append((k, "non_finite"))
    if bad:
        raise RuntimeError(f"[Cell8] {label} contains invalid entries: {bad[:5]}")


def _infer_legacy_cond_dim(model):
    if model is None:
        return None

    if hasattr(model, "cond_dim"):
        try:
            return int(model.cond_dim)
        except Exception:
            pass

    cp = getattr(model, "cond_proj", None)
    if cp is not None:
        if isinstance(cp, nn.Linear):
            return int(cp.in_features)
        if isinstance(cp, nn.Sequential):
            for m in cp.modules():
                if isinstance(m, nn.Linear):
                    return int(m.in_features)

    try:
        for name, m in model.named_modules():
            if "cond" in str(name).lower() and isinstance(m, nn.Linear):
                return int(m.in_features)
    except Exception:
        pass

    return None


def _coerce_matrix(df_part: pd.DataFrame, cols: list, *, label: str) -> np.ndarray:
    if not cols:
        raise RuntimeError(f"[Cell8] {label} received empty column list.")

    miss = [c for c in cols if c not in df_part.columns]
    if miss:
        raise RuntimeError(f"[Cell8] Missing columns in {label}: {miss[:20]}")

    M = df_part[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    if np.isinf(M).any():
        bad = int(np.isinf(M).sum())
        raise RuntimeError(f"[Cell8] {label} contains +/-inf values: {bad}")

    return M


def _validate_binary_matrix(name: str, arr: np.ndarray, expected_shape=None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise RuntimeError(f"[Cell8] {name} must be 2D, got {arr.shape}")

    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise RuntimeError(f"[Cell8] {name} shape mismatch: got={arr.shape}, expected={expected_shape}")

    if not np.isfinite(arr).all():
        raise RuntimeError(f"[Cell8] {name} contains non-finite values.")

    vals = set(np.unique(np.round(arr, 6)).tolist())
    if not vals.issubset({0.0, 1.0}):
        raise RuntimeError(f"[Cell8] {name} must be binary 0/1. values={sorted(vals)[:20]}")

    return arr.astype(np.float32, copy=False)


def _validate_ddpm_obs_for_split(name: str, arr: np.ndarray, expected_n: int, expected_d: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.shape != (expected_n, expected_d):
        raise RuntimeError(
            f"[Cell8] {name} shape mismatch: got={arr.shape}, expected={(expected_n, expected_d)}"
        )

    if not np.isfinite(arr).all():
        raise RuntimeError(f"[Cell8] {name} contains non-finite values.")

    vals = set(np.unique(np.round(arr, 6)).tolist())
    if not vals.issubset({0.0, 1.0}):
        raise RuntimeError(f"[Cell8] {name} must be binary 0/1. values={sorted(vals)[:20]}")

    return arr.astype(np.float32, copy=False)


def _ensure_tod_contract(tod_arr, expected_dim: int, name: str):
    arr = np.asarray(tod_arr, dtype=np.float32)

    if arr.ndim == 1:
        arr2 = arr.reshape(-1, 1)
    elif arr.ndim == 2:
        arr2 = arr
    else:
        raise RuntimeError(f"[Cell8] {name} must be 1D or 2D, got shape={arr.shape}")

    if arr2.shape[1] == expected_dim:
        if not np.isfinite(arr2).all():
            raise RuntimeError(f"[Cell8] {name} contains non-finite values.")
        return arr2.astype(np.float32, copy=False), "already_contract_dim"

    if arr2.shape[1] == 1 and expected_dim == 2:
        scalar = arr2[:, 0].astype(np.float32)
        finite = scalar[np.isfinite(scalar)]

        if finite.size == 0:
            raise RuntimeError(f"[Cell8] {name} contains no finite values.")

        mn = float(np.nanmin(finite))
        mx = float(np.nanmax(finite))

        if mn >= -1e-6 and mx <= 1.0 + 1e-6:
            angle = 2.0 * np.pi * scalar
            mode = "scalar_normalized_0_1_to_sin_cos"
        elif mn >= -1e-6 and mx <= 24.0 + 1e-6:
            angle = 2.0 * np.pi * (scalar / 24.0)
            mode = "scalar_hour_0_24_to_sin_cos"
        elif mn >= -1e-6 and mx <= (2.0 * np.pi + 1e-6):
            angle = scalar
            mode = "scalar_radian_0_2pi_to_sin_cos"
        else:
            raise RuntimeError(f"[Cell8] {name} scalar TOD range cannot be converted to sin/cos: min={mn}, max={mx}")

        out = np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)

        if not np.isfinite(out).all():
            raise RuntimeError(f"[Cell8] reconstructed {name} contains non-finite values.")

        return out, mode

    raise RuntimeError(f"[Cell8] {name} TOD dim mismatch: got {arr2.shape[1]}, expected {expected_dim}.")


def _finite_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def _safe_mean(x):
    x = _finite_1d(x)
    return float(np.mean(x)) if x.size else np.nan


def _safe_std(x):
    x = _finite_1d(x)
    return float(np.std(x)) if x.size > 1 else np.nan


def _safe_quantile(x, q):
    x = _finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.quantile(x, q))


def _nonzero_rate(x):
    x = _finite_1d(x)
    if x.size == 0:
        return np.nan
    return float(np.mean(x > 0))


def _transition_rate(x):
    x = _finite_1d(x)
    if x.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x)) > 1e-9))


def _lag1_autocorr(x):
    x = _finite_1d(x)
    if x.size < 3:
        return np.nan
    a = x[:-1]
    b = x[1:]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _ks_stat_fast(a, b):
    a = np.sort(_finite_1d(a))
    b = np.sort(_finite_1d(b))

    if a.size == 0 or b.size == 0:
        return np.nan

    vals = np.sort(np.unique(np.concatenate([a, b])))
    if vals.size == 0:
        return np.nan

    ca = np.searchsorted(a, vals, side="right") / float(a.size)
    cb = np.searchsorted(b, vals, side="right") / float(b.size)
    return float(np.max(np.abs(ca - cb)))


def _wasserstein_1d_fast(a, b, max_points: int = 5000):
    a = _finite_1d(a)
    b = _finite_1d(b)

    if a.size == 0 or b.size == 0:
        return np.nan

    qs = np.linspace(0.0, 1.0, min(max_points, max(2, min(a.size, b.size))))
    aq = np.quantile(a, qs)
    bq = np.quantile(b, qs)
    return float(np.mean(np.abs(aq - bq)))


def _ratio(num, den, eps=1e-9):
    if not np.isfinite(num) or not np.isfinite(den):
        return np.nan
    return float((num + eps) / (den + eps))


# ------------------------------------------------------------
# 2) DDPM value-column contract from Cell 6
# ------------------------------------------------------------
if not DDPM_COLS:
    raise RuntimeError("[Cell8] DDPM_VALUE_COLS is empty.")

_check_unique("DDPM_VALUE_COLS", DDPM_COLS)

ddpm_sig_local = _sha256_pipejoin(DDPM_COLS)
ddpm_sig_upstream = str(globals().get("DDPM_VALUE_SIG", "") or "").strip()

if ddpm_sig_upstream:
    if ddpm_sig_local != ddpm_sig_upstream:
        raise RuntimeError(
            "[Cell8] DDPM signature drift detected vs Cell 6. "
            f"local={ddpm_sig_local[:12]} upstream={ddpm_sig_upstream[:12]}."
        )
    ddpm_sig = ddpm_sig_upstream
else:
    ddpm_sig = ddpm_sig_local

ddpm_sig12 = ddpm_sig[:12]
globals()["ddpm_sig12"] = ddpm_sig12

log(f"[Cell8] DDPM target contract aligned: sig12={ddpm_sig12} | D={len(DDPM_COLS)}")


# ------------------------------------------------------------
# 3) Load and validate Cell 7 shared conditioning contract
# ------------------------------------------------------------
cond_spec_path = os.path.join(ARTDIR, "cond_spec.json")
if not os.path.exists(cond_spec_path):
    raise RuntimeError(f"[Cell8] Missing conditioning contract artifact: {cond_spec_path} (run Cell 7).")

with open(cond_spec_path, "r", encoding="utf-8") as f:
    cond_spec = json.load(f)

if not isinstance(cond_spec, dict):
    raise RuntimeError("[Cell8] cond_spec.json is not a dict.")

required_cond_keys = [
    "version",
    "obs_cols",
    "bin_cols",
    "cont_state_cols",
    "count_ctx_cols",
    "obs_dim",
    "bin_dim",
    "cont_state_dim",
    "count_ctx_dim",
    "cond_dim",
]
missing_cond_keys = [k for k in required_cond_keys if k not in cond_spec]
if missing_cond_keys:
    raise RuntimeError(f"[Cell8] cond_spec.json missing required keys: {missing_cond_keys}")

COND_OBS_COLS = list(cond_spec["obs_cols"])
COND_BIN_COLS = list(cond_spec["bin_cols"])
COND_CONT_STATE_COLS = list(cond_spec["cont_state_cols"])
COND_COUNT_CTX_COLS = list(cond_spec["count_ctx_cols"])

for name, cols in [
    ("COND_OBS_COLS", COND_OBS_COLS),
    ("COND_BIN_COLS", COND_BIN_COLS),
    ("COND_CONT_STATE_COLS", COND_CONT_STATE_COLS),
    ("COND_COUNT_CTX_COLS", COND_COUNT_CTX_COLS),
]:
    _check_unique(name, cols)

expected_cond_dim = int(cond_spec["cond_dim"])
expected_obs_dim = int(cond_spec["obs_dim"])
expected_bin_dim = int(cond_spec["bin_dim"])
expected_cont_state_dim = int(cond_spec["cont_state_dim"])
expected_count_ctx_dim = int(cond_spec["count_ctx_dim"])

if expected_obs_dim != len(COND_OBS_COLS):
    raise RuntimeError("[Cell8] cond_spec obs_dim mismatch.")
if expected_bin_dim != len(COND_BIN_COLS):
    raise RuntimeError("[Cell8] cond_spec bin_dim mismatch.")
if expected_cont_state_dim != len(COND_CONT_STATE_COLS):
    raise RuntimeError("[Cell8] cond_spec cont_state_dim mismatch.")
if expected_count_ctx_dim != len(COND_COUNT_CTX_COLS):
    raise RuntimeError("[Cell8] cond_spec count_ctx_dim mismatch.")

recomputed_cond_dim = (
    len(COND_OBS_COLS)
    + len(COND_BIN_COLS)
    + len(COND_CONT_STATE_COLS)
    + len(COND_COUNT_CTX_COLS)
)

if recomputed_cond_dim != expected_cond_dim:
    raise RuntimeError(
        f"[Cell8] cond_spec internal cond_dim mismatch: recomputed={recomputed_cond_dim} stored={expected_cond_dim}"
    )

cond_tr = np.asarray(cond_tr, dtype=np.float32)
cond_va = np.asarray(cond_va, dtype=np.float32)

if cond_tr.ndim != 2 or cond_va.ndim != 2:
    raise RuntimeError(f"[Cell8] cond_tr/cond_va must be 2D: {cond_tr.shape}, {cond_va.shape}")

if cond_tr.shape[1] != expected_cond_dim or cond_va.shape[1] != expected_cond_dim:
    raise RuntimeError(
        f"[Cell8] Conditioning dimension mismatch vs cond_spec: "
        f"cond_tr={cond_tr.shape[1]} cond_va={cond_va.shape[1]} expected={expected_cond_dim}"
    )

if len(df_tr) != cond_tr.shape[0] or len(df_va) != cond_va.shape[0]:
    raise RuntimeError(
        f"[Cell8] Conditioning row mismatch: "
        f"df_tr={len(df_tr)} cond_tr={cond_tr.shape[0]} | df_va={len(df_va)} cond_va={cond_va.shape[0]}"
    )

if not np.isfinite(cond_tr).all():
    raise RuntimeError("[Cell8] cond_tr contains non-finite values.")
if not np.isfinite(cond_va).all():
    raise RuntimeError("[Cell8] cond_va contains non-finite values.")

globals()["COND_SPEC"] = cond_spec
globals()["COND_DIM"] = int(expected_cond_dim)
globals()["COND_OBS_COLS"] = COND_OBS_COLS
globals()["COND_BIN_COLS"] = COND_BIN_COLS
globals()["COND_CONT_STATE_COLS"] = COND_CONT_STATE_COLS
globals()["COND_COUNT_CTX_COLS"] = COND_COUNT_CTX_COLS

log(
    f"[Cell8] Shared conditioning contract loaded: "
    f"obs={expected_obs_dim} | bin={expected_bin_dim} | "
    f"cont_state={expected_cont_state_dim} | count_ctx={expected_count_ctx_dim} | "
    f"cond_dim={expected_cond_dim} | version={cond_spec.get('version')}"
)


# ------------------------------------------------------------
# 4) TOD contract
# ------------------------------------------------------------
SEQ_TOD_DIM = int(CFG.get("ddpm_tod_dim", 2))
if SEQ_TOD_DIM != 2:
    raise RuntimeError(f"[Cell8] This canonical SeqDenoiser expects ddpm_tod_dim=2, got {SEQ_TOD_DIM}.")

tod_tr, tod_tr_mode = _ensure_tod_contract(tod_tr, expected_dim=SEQ_TOD_DIM, name="tod_tr")
tod_va, tod_va_mode = _ensure_tod_contract(tod_va, expected_dim=SEQ_TOD_DIM, name="tod_va")

globals()["tod_tr"] = tod_tr
globals()["tod_va"] = tod_va

log(
    f"[Cell8] TOD contract: tod_dim={SEQ_TOD_DIM} | "
    f"tod_tr={tod_tr.shape} ({tod_tr_mode}) | tod_va={tod_va.shape} ({tod_va_mode})"
)


# ------------------------------------------------------------
# 5) Raw DDPM target matrices
# ------------------------------------------------------------
Xtr_raw = _coerce_matrix(df_tr, DDPM_COLS, label="Xtr_raw/DDPM_VALUE_COLS")
Xva_raw = _coerce_matrix(df_va, DDPM_COLS, label="Xva_raw/DDPM_VALUE_COLS")

D = int(Xtr_raw.shape[1])
if D != len(DDPM_COLS):
    raise RuntimeError(f"[Cell8] DDPM dimension mismatch: D={D}, len(DDPM_COLS)={len(DDPM_COLS)}")

globals()["ddpm_D"] = int(D)

raw_nan_tr = int(np.isnan(Xtr_raw).sum())
raw_nan_va = int(np.isnan(Xva_raw).sum())

log(
    f"[Cell8] Raw targets: Xtr_raw={Xtr_raw.shape} | Xva_raw={Xva_raw.shape} | "
    f"nan_train={raw_nan_tr:,} | nan_val={raw_nan_va:,} | D={D}"
)


# ------------------------------------------------------------
# 6) x_obs = Cell 6 logical DDPM obs mask AND finite value
# ------------------------------------------------------------
TIERS_SUPERSET = ["router", "ota", "zigbee", "zwave"]

def _tier_for_col(c: str) -> str:
    for t in TIERS_SUPERSET:
        if str(c).startswith(f"{t}__"):
            return t
    raise RuntimeError(f"[Cell8] Could not infer tier prefix for DDPM column: {c}")


DDPM_COL_TIERS = [_tier_for_col(c) for c in DDPM_COLS]
DDPM_TIERS_IN_COLS = sorted(set(DDPM_COL_TIERS), key=lambda x: TIERS_SUPERSET.index(x))

DDPM_TIERS_USED_CELL8 = list(globals()["DDPM_TIERS_USED"])
if not DDPM_TIERS_USED_CELL8:
    raise RuntimeError("[Cell8] DDPM_TIERS_USED is empty.")

_ddpm_tier_to_obs_idx = {t: i for i, t in enumerate(DDPM_TIERS_USED_CELL8)}

for t in DDPM_TIERS_IN_COLS:
    if t not in _ddpm_tier_to_obs_idx:
        raise RuntimeError(
            f"[Cell8] DDPM column tier {t!r} missing from DDPM_TIERS_USED={DDPM_TIERS_USED_CELL8}"
        )

DDPM_OBS_TR_CELL8 = _validate_ddpm_obs_for_split(
    "DDPM_OBS_TR",
    globals()["DDPM_OBS_TR"],
    len(df_tr),
    len(DDPM_TIERS_USED_CELL8),
)

DDPM_OBS_VA_CELL8 = _validate_ddpm_obs_for_split(
    "DDPM_OBS_VA",
    globals()["DDPM_OBS_VA"],
    len(df_va),
    len(DDPM_TIERS_USED_CELL8),
)


def _tier_obs_matrix_for_cols_from_cell6(ddpm_obs_split: np.ndarray, cols: list, split_name: str) -> np.ndarray:
    Np = int(ddpm_obs_split.shape[0])
    col_tiers = np.array([_tier_for_col(c) for c in cols], dtype=object)
    XOBS_TIER = np.empty((Np, len(cols)), dtype=np.float32)

    for t in sorted(set(col_tiers), key=lambda x: TIERS_SUPERSET.index(x)):
        if t not in _ddpm_tier_to_obs_idx:
            raise RuntimeError(
                f"[Cell8] {split_name}: tier {t!r} missing from DDPM_TIERS_USED={DDPM_TIERS_USED_CELL8}"
            )

        tier_mask = ddpm_obs_split[:, _ddpm_tier_to_obs_idx[t]].astype(np.float32, copy=False)
        where_t = np.where(col_tiers == t)[0]
        XOBS_TIER[:, where_t] = tier_mask[:, None]

    return XOBS_TIER.astype(np.float32, copy=False)


def _build_value_finite_xobs(X_raw: np.ndarray, cols: list, split_name: str) -> tuple:
    if split_name == "train":
        tier_obs = _tier_obs_matrix_for_cols_from_cell6(DDPM_OBS_TR_CELL8, cols, split_name)
    elif split_name == "val":
        tier_obs = _tier_obs_matrix_for_cols_from_cell6(DDPM_OBS_VA_CELL8, cols, split_name)
    else:
        raise RuntimeError(f"[Cell8] Unsupported x_obs split_name={split_name!r}")

    finite_obs = np.isfinite(X_raw).astype(np.float32, copy=False)
    xobs = (tier_obs * finite_obs).astype(np.float32, copy=False)

    audit = {
        "split": split_name,
        "shape": [int(xobs.shape[0]), int(xobs.shape[1])],
        "tier_only_obs_rate": float(tier_obs.mean()) if tier_obs.size else 0.0,
        "value_finite_rate": float(finite_obs.mean()) if finite_obs.size else 0.0,
        "final_x_obs_rate": float(xobs.mean()) if xobs.size else 0.0,
        "cells_tier_obs_but_value_nan": int(np.sum((tier_obs > 0.5) & (finite_obs <= 0.5))),
        "policy": "x_obs = Cell6 logical DDPM tier obs mask AND isfinite(raw_target_value)",
        "logical_obs_source": "Cell6 DDPM_OBS_* matrices",
        "ddpm_tiers_used": list(DDPM_TIERS_USED_CELL8),
    }

    return xobs, tier_obs, finite_obs, audit


xobs_tr, xobs_tier_tr, xfinite_tr, xobs_audit_tr = _build_value_finite_xobs(Xtr_raw, DDPM_COLS, "train")
xobs_va, xobs_tier_va, xfinite_va, xobs_audit_va = _build_value_finite_xobs(Xva_raw, DDPM_COLS, "val")

xobs_tr = _validate_binary_matrix("xobs_tr", xobs_tr, expected_shape=Xtr_raw.shape)
xobs_va = _validate_binary_matrix("xobs_va", xobs_va, expected_shape=Xva_raw.shape)

obs_rate_tr = float(xobs_tr.mean())
obs_rate_va = float(xobs_va.mean())

log(
    f"[Cell8] x_obs logical value-finite policy: "
    f"train={obs_rate_tr:.6f} | val={obs_rate_va:.6f} | "
    f"train_tier_obs_but_value_nan={xobs_audit_tr['cells_tier_obs_but_value_nan']:,} | "
    f"val_tier_obs_but_value_nan={xobs_audit_va['cells_tier_obs_but_value_nan']:,}"
)

if obs_rate_tr <= 0.0:
    raise RuntimeError("[Cell8] TRAIN x_obs is all zeros.")
if obs_rate_va <= 0.0:
    log("[Cell8][WARN] VAL x_obs is all zeros. VAL diagnostics will be degenerate.")


# ------------------------------------------------------------
# 7) Absolute protocol-aware target transform
# ------------------------------------------------------------
COUNTLIKE_RX = re.compile(
    r"(_total$|_pkt$|_pkts$|_packets$|_bytes$|_syn$|_ack$|_rst$|_fin$|"
    r"_query$|_queries$|_response$|_responses$|_req$|_reply$|_cmd$|"
    r"_beacon$|_data$|_mgmt$|_pps$|_other_type$|_rcode\d+_.*$|_count$|_events$)",
    re.IGNORECASE,
)

VOLUME_RX = re.compile(r"(bytes|pkt_total|packets|_pkt$|_pps$)", re.IGNORECASE)


def _is_heavy_nonnegative_col(c: str, vals: np.ndarray) -> bool:
    vals = _finite_1d(vals)
    if vals.size == 0:
        return bool(COUNTLIKE_RX.search(str(c)))

    vals = np.maximum(vals, 0.0)

    q50 = _safe_quantile(vals, 0.50)
    q95 = _safe_quantile(vals, 0.95)
    q99 = _safe_quantile(vals, 0.99)
    mean = _safe_mean(vals)

    dynamic_tail = (
        np.isfinite(q99)
        and np.isfinite(q50)
        and q99 > max(10.0, 3.0 * max(q50, 1.0))
    )

    high_scale = bool(np.isfinite(q95) and q95 >= 20.0)
    name_hint = bool(VOLUME_RX.search(str(c)))

    return bool(name_hint or high_scale or dynamic_tail or mean > 10.0)


def _fit_target_transform_stats(X_raw: np.ndarray, X_obs: np.ndarray, cols: list):
    Dloc = X_raw.shape[1]

    center = np.zeros(Dloc, dtype=np.float32)
    scale = np.ones(Dloc, dtype=np.float32)
    cap_hi = np.zeros(Dloc, dtype=np.float32)
    transform_kind = []
    obs_support = np.zeros(Dloc, dtype=np.int64)

    rows = []

    cap_q = float(np.clip(float(CFG.get("ddpm_target_cap_q", 0.9995)), 0.95, 0.99999))
    min_scale = float(CFG.get("ddpm_target_min_scale", 1e-6))

    for j, c in enumerate(cols):
        obs = X_obs[:, j] > 0.5
        x = X_raw[:, j]
        good = obs & np.isfinite(x)
        obs_support[j] = int(np.sum(good))

        if np.any(good):
            vals_raw = x[good].astype(np.float64, copy=False)
            source = "observed_train"
        else:
            finite = np.isfinite(x)
            if np.any(finite):
                vals_raw = x[finite].astype(np.float64, copy=False)
                source = "finite_fallback_no_observed_support"
            else:
                vals_raw = np.asarray([0.0], dtype=np.float64)
                source = "zero_fallback_no_finite_values"

        vals_nonneg = np.maximum(vals_raw, 0.0)
        use_log = _is_heavy_nonnegative_col(c, vals_nonneg)
        kind = "log1p_robust" if use_log else "identity_robust"

        if use_log:
            cap = float(np.quantile(vals_nonneg, cap_q)) if vals_nonneg.size else 0.0
            cap = max(cap, 1.0) if np.isfinite(cap) and cap > 0 else 0.0
            z = np.log1p(np.minimum(vals_nonneg, cap)).astype(np.float64, copy=False)
        else:
            cap = np.nan
            z = vals_raw.astype(np.float64, copy=False)

        med = float(np.median(z))
        q25 = float(np.quantile(z, 0.25))
        q75 = float(np.quantile(z, 0.75))
        iqr = float(q75 - q25)
        sc = max(iqr, min_scale)

        center[j] = np.float32(med)
        scale[j] = np.float32(sc)
        cap_hi[j] = np.float32(cap if np.isfinite(cap) else np.nan)
        transform_kind.append(kind)

        rows.append({
            "col": str(c),
            "tier": str(DDPM_COL_TIERS[j]),
            "transform": kind,
            "source": source,
            "observed_train_n": int(obs_support[j]),
            "raw_nonzero_rate": _nonzero_rate(vals_raw),
            "raw_mean": _safe_mean(vals_raw),
            "raw_std": _safe_std(vals_raw),
            "raw_q50": _safe_quantile(vals_raw, 0.50),
            "raw_q95": _safe_quantile(vals_raw, 0.95),
            "raw_q99": _safe_quantile(vals_raw, 0.99),
            "raw_transition_rate": _transition_rate(vals_raw),
            "cap_q": cap_q if use_log else None,
            "cap_hi": cap if use_log else None,
            "center": med,
            "scale": sc,
        })

    if not np.isfinite(center).all():
        raise RuntimeError("[Cell8] target transform center contains non-finite values.")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("[Cell8] target transform scale invalid.")

    return center, scale, cap_hi, transform_kind, obs_support, rows


def _apply_target_transform(X_raw, center, scale, cap_hi, transform_kind, clip):
    X = np.asarray(X_raw, dtype=np.float32).copy()
    out = np.zeros_like(X, dtype=np.float32)

    for j, kind in enumerate(transform_kind):
        x = X[:, j].astype(np.float64, copy=False)
        bad = ~np.isfinite(x)

        if kind == "log1p_robust":
            fill_raw = max(np.expm1(float(center[j])), 0.0)
            x = np.where(bad, fill_raw, x)
            x = np.maximum(x, 0.0)

            cap = float(cap_hi[j])
            if np.isfinite(cap) and cap > 0:
                x = np.minimum(x, cap)

            z = np.log1p(x)
        else:
            fill_raw = float(center[j])
            x = np.where(bad, fill_raw, x)
            z = x

        z = (z - float(center[j])) / float(scale[j])

        if clip is not None:
            z = np.clip(z, -float(clip), float(clip))

        out[:, j] = z.astype(np.float32, copy=False)

    if not np.isfinite(out).all():
        raise RuntimeError("[Cell8] transformed DDPM matrix contains non-finite values.")

    return out.astype(np.float32, copy=False)


def _inverse_target_transform(X_scaled, center, scale, cap_hi, transform_kind):
    Z = np.asarray(X_scaled, dtype=np.float32)
    out = np.zeros_like(Z, dtype=np.float32)

    if Z.ndim != 2 or Z.shape[1] != len(transform_kind):
        raise RuntimeError(f"[Cell8] inverse target transform shape mismatch: X={Z.shape}")

    for j, kind in enumerate(transform_kind):
        z = Z[:, j].astype(np.float64, copy=False)
        x = z * float(scale[j]) + float(center[j])

        if kind == "log1p_robust":
            x = np.expm1(x)
            x = np.maximum(x, 0.0)
        else:
            x = np.maximum(x, 0.0)

        out[:, j] = x.astype(np.float32, copy=False)

    if not np.isfinite(out).all():
        bad = int((~np.isfinite(out)).sum())
        raise RuntimeError(f"[Cell8] inverse transformed DDPM output contains non-finite values: bad={bad}")

    return out.astype(np.float32, copy=False)


class ProtocolAwareTargetScaler:
    """
    Minimal sklearn-like scaler for downstream compatibility.
    """

    def __init__(self, center, scale, cap_hi, transform_kind, cols):
        self.center_ = np.asarray(center, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.cap_hi_ = np.asarray(cap_hi, dtype=np.float64)
        self.transform_kind_ = list(transform_kind)
        self.cols_ = list(cols)
        self.n_features_in_ = int(len(self.cols_))
        self.scaler_type = "ProtocolAwareTargetScaler"

    def transform(self, X):
        return _apply_target_transform(
            X,
            self.center_.astype(np.float32),
            self.scale_.astype(np.float32),
            self.cap_hi_.astype(np.float32),
            self.transform_kind_,
            clip=None,
        )

    def inverse_transform(self, X):
        return _inverse_target_transform(
            X,
            self.center_.astype(np.float32),
            self.scale_.astype(np.float32),
            self.cap_hi_.astype(np.float32),
            self.transform_kind_,
        )


scale_clip = float(CFG.get("ddpm_scale_clip", 8.0))

ddpm_center, ddpm_scale, ddpm_cap_hi, ddpm_transform_kind, ddpm_obs_support, transform_rows = _fit_target_transform_stats(
    Xtr_raw,
    xobs_tr,
    DDPM_COLS,
)

Xtr_ddpm = _apply_target_transform(
    Xtr_raw,
    ddpm_center,
    ddpm_scale,
    ddpm_cap_hi,
    ddpm_transform_kind,
    clip=scale_clip,
)

Xva_ddpm = _apply_target_transform(
    Xva_raw,
    ddpm_center,
    ddpm_scale,
    ddpm_cap_hi,
    ddpm_transform_kind,
    clip=scale_clip,
)

ddpm_scaler = ProtocolAwareTargetScaler(
    center=ddpm_center,
    scale=ddpm_scale,
    cap_hi=ddpm_cap_hi,
    transform_kind=ddpm_transform_kind,
    cols=DDPM_COLS,
)

globals()["Xtr_ddpm"] = Xtr_ddpm
globals()["Xva_ddpm"] = Xva_ddpm
globals()["xobs_tr_ddpm"] = xobs_tr
globals()["xobs_va_ddpm"] = xobs_va
globals()["ddpm_cols_use"] = list(DDPM_COLS)
globals()["ddpm_scaler"] = ddpm_scaler
globals()["ddpm_median"] = ddpm_center
globals()["ddpm_scale"] = ddpm_scale
globals()["ddpm_target_center"] = ddpm_center
globals()["ddpm_target_scale"] = ddpm_scale
globals()["ddpm_target_cap_hi"] = ddpm_cap_hi
globals()["ddpm_target_transform_kind"] = list(ddpm_transform_kind)
globals()["ddpm_obs_support"] = ddpm_obs_support

transform_counts = dict(pd.Series(ddpm_transform_kind).value_counts().sort_index())

log(
    f"[Cell8] Absolute protocol-aware target transform complete: "
    f"Xtr_ddpm={Xtr_ddpm.shape} | Xva_ddpm={Xva_ddpm.shape} | "
    f"transforms={transform_counts} | mode=absolute_protocol_aware"
)


# ------------------------------------------------------------
# 7b) Target support advisory pre-screen
# ------------------------------------------------------------
STRUCTURAL_RX = re.compile(
    r"(__obs_present$|_obs_present$|__channel$|_channel$|_unique$|uniq_|"
    r"_src16_unique$|_dst16_unique$|_ip_src_unique$|_ip_dst_unique$|_port_unique$)",
    re.IGNORECASE,
)

RARE_EVENT_RX = re.compile(
    r"(assoc_req|assoc_resp|auth|deauth|disassoc|dns_rcode3|aps_present|zcl_present|app_obs_present|"
    r"aps_obs_present|zcl_obs_present)",
    re.IGNORECASE,
)


def _column_stats_for_support(col: str, x_raw: np.ndarray, x_obs: np.ndarray) -> dict:
    obs = x_obs > 0.5
    vals = x_raw[obs & np.isfinite(x_raw)].astype(np.float64, copy=False)

    if vals.size == 0:
        return {
            "col": col,
            "n_obs": 0,
            "nonzero_rate": None,
            "unique_n": 0,
            "std": None,
            "q50": None,
            "q95": None,
            "q99": None,
            "reason": "no_observed_train_values",
        }

    rounded = np.round(vals, 6)

    return {
        "col": col,
        "n_obs": int(vals.size),
        "nonzero_rate": float(np.mean(vals > 0)),
        "unique_n": int(np.unique(rounded).size),
        "std": float(np.std(vals)),
        "q50": float(np.quantile(vals, 0.50)),
        "q95": float(np.quantile(vals, 0.95)),
        "q99": float(np.quantile(vals, 0.99)),
        "transition_rate": _transition_rate(vals),
        "reason": "ok",
    }


min_train_support_advisory = int(CFG.get("ddpm_candidate_min_train_support", 5000))
min_nonzero_rate_advisory = float(CFG.get("ddpm_candidate_min_nonzero_rate", 0.005))
min_unique_advisory = int(CFG.get("ddpm_candidate_min_unique_values", 3))

target_support_rows = []
advisory_cols_pre = []
excluded_advisory_cols_pre = {}

for j, c in enumerate(DDPM_COLS):
    st = _column_stats_for_support(c, Xtr_raw[:, j], xobs_tr[:, j])
    tier = DDPM_COL_TIERS[j]
    reasons = []

    if STRUCTURAL_RX.search(c):
        reasons.append("structural_or_identifier_like")
    if RARE_EVENT_RX.search(c):
        reasons.append("rare_event_like")
    if int(st["n_obs"]) < min_train_support_advisory:
        reasons.append("too_few_observed_train_values")

    nz = st["nonzero_rate"]
    if nz is None or float(nz) < min_nonzero_rate_advisory:
        reasons.append("too_sparse_for_ddpm_candidate")

    if int(st["unique_n"]) < min_unique_advisory:
        reasons.append("near_constant_or_low_support")

    if st["std"] is None or float(st["std"]) <= 1e-12:
        reasons.append("near_zero_variance")

    allow = len(reasons) == 0

    row = dict(st)
    row["tier"] = tier
    row["target_transform"] = ddpm_transform_kind[j]
    row["advisory_pre_val_eligible"] = bool(allow)
    row["exclusion_reasons"] = list(reasons)
    target_support_rows.append(row)

    if allow:
        advisory_cols_pre.append(c)
    else:
        excluded_advisory_cols_pre[c] = reasons

DDPM_TARGET_SUPPORT_AUDIT = {
    "version": "ddpm_target_support_audit_v18_THESIS_advisory",
    "policy": (
        "This audit does not remove columns from DDPM training and does not authorize final selection. "
        "It only identifies columns that are plausible DDPM diagnostic candidates before VAL diagnostics. "
        "The downstream VAL-only selector must compare candidates against true A1 on VAL."
    ),
    "min_train_support_advisory": int(min_train_support_advisory),
    "min_nonzero_rate_advisory": float(min_nonzero_rate_advisory),
    "min_unique_advisory": int(min_unique_advisory),
    "rows": target_support_rows,
    "advisory_candidate_cols_pre_val": list(advisory_cols_pre),
    "excluded_advisory_candidate_cols_pre_val": excluded_advisory_cols_pre,
}

target_support_path = os.path.join(ARTDIR, f"ddpm_target_support_audit_sig{ddpm_sig12}_D{D}.json")
target_support_csv_path = os.path.join(REP_DIR, f"ddpm_target_support_audit_sig{ddpm_sig12}_D{D}.csv")

_write_json(target_support_path, DDPM_TARGET_SUPPORT_AUDIT)
pd.DataFrame(target_support_rows).to_csv(target_support_csv_path, index=False)

globals()["DDPM_TARGET_SUPPORT_AUDIT"] = DDPM_TARGET_SUPPORT_AUDIT
globals()["DDPM_ADVISORY_COLS_PRE_VAL"] = list(advisory_cols_pre)
globals()["DDPM_EXCLUDED_ADVISORY_COLS_PRE_VAL"] = dict(excluded_advisory_cols_pre)

log(
    "[Cell8] DDPM advisory candidate pre-screen | "
    f"eligible_pre_val={len(advisory_cols_pre)} | "
    f"excluded={len(excluded_advisory_cols_pre)} | audit={target_support_path}"
)


# ------------------------------------------------------------
# 8) Persist pre-training artifacts
# ------------------------------------------------------------
cond_sig = str(cond_spec.get("sig_cond", "") or "")
cond_sig12 = cond_sig[:12] if cond_sig else ""

seq_len = int(CFG.get("ddpm_seq_len", 60))
stride = int(CFG.get("ddpm_stride", 5))
batch_size = int(CFG.get("ddpm_batch", 128))
timesteps = int(CFG.get("ddpm_timesteps", 200))
epochs = int(CFG.get("ddpm_epochs", 12))
lr = float(CFG.get("ddpm_lr", 2e-4))

if seq_len <= 0:
    raise RuntimeError(f"[Cell8] Invalid ddpm_seq_len={seq_len}")
if stride <= 0:
    raise RuntimeError(f"[Cell8] Invalid ddpm_stride={stride}")
if stride > seq_len:
    raise RuntimeError(f"[Cell8] Invalid stride/seq_len: stride={stride}, seq_len={seq_len}")
if batch_size <= 0:
    raise RuntimeError(f"[Cell8] Invalid ddpm_batch={batch_size}")
if timesteps < 10:
    raise RuntimeError(f"[Cell8] Invalid ddpm_timesteps={timesteps}; must be >=10.")
if epochs <= 0:
    raise RuntimeError(f"[Cell8] Invalid ddpm_epochs={epochs}")
if lr <= 0:
    raise RuntimeError(f"[Cell8] Invalid ddpm_lr={lr}")

xtr_path = os.path.join(ARTDIR, f"Xtr_ddpm_sig{ddpm_sig12}_D{D}.npy")
xva_path = os.path.join(ARTDIR, f"Xva_ddpm_sig{ddpm_sig12}_D{D}.npy")
xobs_tr_path = os.path.join(ARTDIR, f"xobs_tr_ddpm_sig{ddpm_sig12}_D{D}.npy")
xobs_va_path = os.path.join(ARTDIR, f"xobs_va_ddpm_sig{ddpm_sig12}_D{D}.npy")
center_path = os.path.join(ARTDIR, f"ddpm_target_center_sig{ddpm_sig12}_D{D}.npy")
scale_path = os.path.join(ARTDIR, f"ddpm_target_scale_sig{ddpm_sig12}_D{D}.npy")
cap_path = os.path.join(ARTDIR, f"ddpm_target_cap_hi_sig{ddpm_sig12}_D{D}.npy")
support_path = os.path.join(ARTDIR, f"ddpm_obs_support_sig{ddpm_sig12}_D{D}.npy")
scaler_joblib_path = os.path.join(ARTDIR, f"ddpm_scaler_sig{ddpm_sig12}_D{D}.joblib")
spec_path = os.path.join(ARTDIR, f"ddpm_train_spec_sig{ddpm_sig12}_D{D}.json")
xobs_audit_path = os.path.join(ARTDIR, f"ddpm_xobs_value_finite_audit_sig{ddpm_sig12}_D{D}.json")
target_transform_path = os.path.join(ARTDIR, f"ddpm_target_transform_audit_sig{ddpm_sig12}_D{D}.json")
target_transform_csv_path = os.path.join(REP_DIR, f"ddpm_target_transform_audit_sig{ddpm_sig12}_D{D}.csv")

np.save(xtr_path, Xtr_ddpm)
np.save(xva_path, Xva_ddpm)
np.save(xobs_tr_path, xobs_tr.astype(np.float32))
np.save(xobs_va_path, xobs_va.astype(np.float32))
np.save(center_path, ddpm_center.astype(np.float32))
np.save(scale_path, ddpm_scale.astype(np.float32))
np.save(cap_path, ddpm_cap_hi.astype(np.float32))
np.save(support_path, ddpm_obs_support.astype(np.int64))

joblib.dump(ddpm_scaler, scaler_joblib_path)

xobs_audit = {
    "version": "ddpm_xobs_value_finite_policy_v5_cell6_logical_obs",
    "policy": "x_obs = Cell6 logical DDPM tier obs mask AND isfinite(raw_target_value)",
    "logical_obs_source": "DDPM_OBS_TR/DDPM_OBS_VA from Cell6",
    "tiers_in_ddpm_cols": DDPM_TIERS_IN_COLS,
    "ddpm_tiers_used": list(DDPM_TIERS_USED_CELL8),
    "train": xobs_audit_tr,
    "val": xobs_audit_va,
    "zero_observed_train_support_cols": [
        DDPM_COLS[i] for i, n in enumerate(ddpm_obs_support) if int(n) <= 0
    ],
}
_write_json(xobs_audit_path, xobs_audit)

target_transform_audit = {
    "version": "ddpm_target_transform_audit_v18_THESIS_absolute_protocol_aware",
    "policy": (
        "DDPM targets are absolute protocol values transformed into denoising space. "
        "Heavy nonnegative count/volume columns use log1p + robust scaling. "
        "Other protocol columns use robust scaling. TEST is not used."
    ),
    "target_mode": "absolute_protocol_aware",
    "scale_clip": float(scale_clip),
    "transform_counts": transform_counts,
    "rows": transform_rows,
}
_write_json(target_transform_path, target_transform_audit)
pd.DataFrame(transform_rows).to_csv(target_transform_csv_path, index=False)

log(f"[Cell8] Saved scaler artifact: {scaler_joblib_path}")
log(f"[Cell8] Saved x_obs value-finite audit: {xobs_audit_path}")
log(f"[Cell8] Saved target transform audit: {target_transform_path}")


# ------------------------------------------------------------
# 9) Dataset / loaders
# ------------------------------------------------------------
if len(reg_tr) != len(df_tr) or len(reg_va) != len(df_va):
    raise RuntimeError(
        f"[Cell8] Regime length mismatch: "
        f"len(reg_tr)={len(reg_tr)} len(df_tr)={len(df_tr)} | "
        f"len(reg_va)={len(reg_va)} len(df_va)={len(df_va)}"
    )

if len(tod_tr) != len(df_tr) or len(tod_va) != len(df_va):
    raise RuntimeError(
        f"[Cell8] Time-of-day length mismatch: "
        f"len(tod_tr)={len(tod_tr)} len(df_tr)={len(df_tr)} | "
        f"len(tod_va)={len(tod_va)} len(df_va)={len(df_va)}"
    )

reg_tr = np.asarray(reg_tr, dtype=np.int64).reshape(-1)
reg_va = np.asarray(reg_va, dtype=np.int64).reshape(-1)

globals()["reg_tr"] = reg_tr
globals()["reg_va"] = reg_va

ds_tr = SequenceDataset(
    X=Xtr_ddpm,
    x_obs=xobs_tr,
    cond_seq=cond_tr,
    regime_seq=reg_tr,
    tod_seq=tod_tr,
    seq_len=seq_len,
    stride=stride,
    return_t_idx=False,
    debug_checks=True,
    x_finite_ref=xobs_tr,
    x_finite_ref_mode="mask",
    seed=seed,
)

ds_va = SequenceDataset(
    X=Xva_ddpm,
    x_obs=xobs_va,
    cond_seq=cond_va,
    regime_seq=reg_va,
    tod_seq=tod_va,
    seq_len=seq_len,
    stride=stride,
    return_t_idx=False,
    debug_checks=True,
    x_finite_ref=xobs_va,
    x_finite_ref_mode="mask",
    seed=seed + 1,
)

_use_pin_memory = (device_obj.type == "cuda")
loader_workers = int(CFG.get("ddpm_dataloader_workers", 0))

loader_gen = torch.Generator()
loader_gen.manual_seed(seed)

loader_tr = DataLoader(
    ds_tr,
    batch_size=batch_size,
    shuffle=True,
    num_workers=loader_workers,
    pin_memory=_use_pin_memory,
    drop_last=False,
    generator=loader_gen,
)

loader_va = DataLoader(
    ds_va,
    batch_size=batch_size,
    shuffle=False,
    num_workers=loader_workers,
    pin_memory=_use_pin_memory,
    drop_last=False,
)

globals()["loader_tr"] = loader_tr
globals()["loader_va"] = loader_va
globals()["ds_tr"] = ds_tr
globals()["ds_va"] = ds_va

log(f"[Cell8] Datasets: train_windows={len(ds_tr)} | val_windows={len(ds_va)}")


# ------------------------------------------------------------
# 10) SeqDenoiser
# ------------------------------------------------------------
class SeqResidualBlock(nn.Module):
    def __init__(self, width: int, k: int = 3, p: float = 0.0):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=k, padding=pad),
            nn.SiLU(),
            nn.Dropout(float(p)),
            nn.Conv1d(width, width, kernel_size=k, padding=pad),
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=width)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(x + self.net(x)))


class SeqSelfAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width * 2, width),
        )

    def forward(self, x):
        z = x.transpose(1, 2)
        z_norm = self.norm(z)
        a, _ = self.attn(z_norm, z_norm, z_norm, need_weights=False)
        z = z + a
        z = z + self.ff(z)
        return z.transpose(1, 2)


class SeqDenoiser(nn.Module):
    def __init__(
        self,
        D: int,
        cond_dim: int,
        K_reg: int,
        width: int = 384,
        depth: int = 6,
        t_emb_dim: int = 96,
        tod_dim: int = 2,
        timesteps: int = 200,
        dropout: float = 0.02,
        use_attention: bool = True,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.D = int(D)
        self.cond_dim = int(cond_dim)
        self.K_reg = int(max(1, K_reg))
        self.width = int(width)
        self.depth = int(depth)
        self.t_emb_dim = int(t_emb_dim)
        self.tod_dim = int(tod_dim)
        self.timesteps = int(timesteps)
        self.dropout = float(dropout)
        self.use_attention = bool(use_attention)
        self.attention_heads = int(attention_heads)

        if self.D <= 0:
            raise RuntimeError("[SeqDenoiser] D must be positive.")
        if self.cond_dim <= 0:
            raise RuntimeError("[SeqDenoiser] cond_dim must be positive.")
        if self.tod_dim <= 0:
            raise RuntimeError("[SeqDenoiser] tod_dim must be positive.")
        if self.timesteps < 10:
            raise RuntimeError("[SeqDenoiser] timesteps must be >=10.")
        if self.width <= 0:
            raise RuntimeError("[SeqDenoiser] width must be positive.")
        if self.depth <= 0:
            raise RuntimeError("[SeqDenoiser] depth must be positive.")
        if self.width % max(1, self.attention_heads) != 0:
            raise RuntimeError(
                f"[SeqDenoiser] width={self.width} must be divisible by attention_heads={self.attention_heads}"
            )

        self.t_embed = nn.Embedding(self.timesteps, self.t_emb_dim)
        self.reg_embed = nn.Embedding(self.K_reg, 16)

        in_dim = self.D + self.cond_dim + self.tod_dim + self.t_emb_dim + 16
        self.in_proj = nn.Linear(in_dim, self.width)

        blocks = []
        attn_every = max(1, int(CFG.get("ddpm_attention_every", 2)))

        for i in range(self.depth):
            blocks.append(SeqResidualBlock(self.width, k=3, p=self.dropout))
            if self.use_attention and ((i + 1) % attn_every == 0):
                blocks.append(
                    SeqSelfAttentionBlock(
                        self.width,
                        heads=self.attention_heads,
                        dropout=self.dropout,
                    )
                )

        self.blocks = nn.ModuleList(blocks)

        self.mid = nn.Sequential(
            nn.Conv1d(self.width, self.width, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.width, self.width, kernel_size=1),
            nn.SiLU(),
        )

        self.out_proj = nn.Conv1d(self.width, self.D, kernel_size=1)

    def forward(self, x_t, t, cond_seq, reg_seq, tod_seq):
        B, L, Dloc = x_t.shape

        if Dloc != self.D:
            raise RuntimeError(f"[SeqDenoiser] D mismatch: got {Dloc}, expected {self.D}")
        if t.ndim != 1 or t.shape[0] != B:
            raise RuntimeError(f"[SeqDenoiser] t must be (B,), got {tuple(t.shape)}")
        if int(t.min().detach().cpu().item()) < 0 or int(t.max().detach().cpu().item()) >= self.timesteps:
            raise RuntimeError("[SeqDenoiser] timestep outside embedding range.")
        if cond_seq.shape != (B, L, self.cond_dim):
            raise RuntimeError(
                f"[SeqDenoiser] cond_seq shape mismatch: got {tuple(cond_seq.shape)}, expected {(B, L, self.cond_dim)}"
            )
        if tod_seq.ndim == 2:
            tod_seq = tod_seq.unsqueeze(-1)
        if tod_seq.shape != (B, L, self.tod_dim):
            raise RuntimeError(
                f"[SeqDenoiser] tod_seq shape mismatch: got {tuple(tod_seq.shape)}, expected {(B, L, self.tod_dim)}"
            )
        if reg_seq.ndim != 2 or reg_seq.shape != (B, L):
            raise RuntimeError(f"[SeqDenoiser] reg_seq must be (B,L), got {tuple(reg_seq.shape)}")
        if int(reg_seq.min().detach().cpu().item()) < 0:
            raise RuntimeError("[SeqDenoiser] reg_seq contains negative regime id.")
        if int(reg_seq.max().detach().cpu().item()) >= self.K_reg:
            raise RuntimeError("[SeqDenoiser] reg_seq exceeds K_reg.")

        t_emb = self.t_embed(t).unsqueeze(1).expand(B, L, -1)
        reg_emb = self.reg_embed(reg_seq)

        z = torch.cat([x_t, cond_seq, tod_seq, t_emb, reg_emb], dim=-1)
        z = self.in_proj(z)
        z = z.transpose(1, 2)

        for blk in self.blocks:
            z = blk(z)

        z = self.mid(z)
        out = self.out_proj(z).transpose(1, 2)

        return out


K_reg_effective = int(globals().get("K_reg", max(1, int(np.max(reg_tr)) + 1)))

net_width = int(CFG.get("ddpm_width", 384))
net_depth = int(CFG.get("ddpm_depth", 6))
t_emb_dim = int(CFG.get("ddpm_t_emb_dim", 96))
ddpm_dropout = float(CFG.get("ddpm_dropout", 0.02))
use_attention = bool(CFG.get("ddpm_use_attention", True))
attention_heads = int(CFG.get("ddpm_attention_heads", 4))

net = SeqDenoiser(
    D=D,
    cond_dim=int(expected_cond_dim),
    K_reg=K_reg_effective,
    width=net_width,
    depth=net_depth,
    t_emb_dim=t_emb_dim,
    tod_dim=SEQ_TOD_DIM,
    timesteps=timesteps,
    dropout=ddpm_dropout,
    use_attention=use_attention,
    attention_heads=attention_heads,
)

expected_in_width = int(D + expected_cond_dim + SEQ_TOD_DIM + t_emb_dim + 16)
actual_in_width = int(net.in_proj.weight.shape[1])

if actual_in_width != expected_in_width:
    raise RuntimeError(
        f"[Cell8] SeqDenoiser input width mismatch: actual={actual_in_width}, expected={expected_in_width}"
    )


# ------------------------------------------------------------
# 11) Build active DDPM wrapper
# ------------------------------------------------------------
ddpm_seq = DDPM(
    net=net,
    T=int(timesteps),
    device=device,
    ema_decay=float(CFG.get("ddpm_ema_decay", 0.999)),
)

if ddpm_seq.device_obj.type != device_obj.type:
    raise RuntimeError(f"[Cell8] DDPM wrapper device mismatch: wrapper={ddpm_seq.device_obj}, requested={device_obj}")

ddpm_seq.fit_feature_weights(loader_tr)

globals()["ddpm"] = ddpm_seq
globals()["ddpm_seq"] = ddpm_seq
globals()["ddpm_net"] = net
globals()["ddpm_model"] = net

globals()["DDPM_ACTIVE_MODEL_FAMILY"] = "SeqDenoiser"
globals()["DDPM_ACTIVE_WRAPPER_GLOBAL"] = "ddpm_seq"
globals()["DDPM_ACTIVE_NET_GLOBAL"] = "ddpm_net"

if type(globals()["ddpm_net"]).__name__ != "SeqDenoiser":
    raise RuntimeError(f"[Cell8] Active ddpm_net must be SeqDenoiser, got {type(globals()['ddpm_net']).__name__}")

if globals()["ddpm_model"] is not globals()["ddpm_net"]:
    raise RuntimeError("[Cell8] ddpm_model compatibility alias must point to ddpm_net.")

if int(getattr(globals()["ddpm_net"], "cond_dim", -1)) != int(expected_cond_dim):
    raise RuntimeError("[Cell8] ddpm_net cond_dim mismatch.")

log(
    "[Cell8] Active DDPM runtime exported: "
    f"ddpm_seq={type(ddpm_seq).__name__} | ddpm_net={type(net).__name__} | "
    f"ddpm_model_alias={type(globals()['ddpm_model']).__name__} | device={ddpm_seq.device}"
)

legacy_model = globals().get("ddpm_model_legacy", None)
legacy_cond_dim = _infer_legacy_cond_dim(legacy_model)
if legacy_cond_dim is not None and int(legacy_cond_dim) != int(expected_cond_dim):
    log(
        "[Cell8][LEGACY] Previous ddpm_model preserved as ddpm_model_legacy: "
        f"type={type(legacy_model).__name__}, cond_dim={legacy_cond_dim}, active_cond_dim={expected_cond_dim}"
    )


# ------------------------------------------------------------
# 12) Train
# ------------------------------------------------------------
use_amp = bool(CFG.get("ddpm_use_amp", True))

log(
    f"[Cell8] Training DDPM candidate generator: "
    f"T={timesteps} | epochs={epochs} | batch={batch_size} | lr={lr} | "
    f"seq_len={seq_len} | stride={stride} | D={D} | cond_dim={int(expected_cond_dim)} | "
    f"tod_dim={SEQ_TOD_DIM} | K_reg={K_reg_effective} | width={net_width} | depth={net_depth} | "
    f"attention={use_attention} | heads={attention_heads} | dropout={ddpm_dropout} | "
    f"device={device} | amp={use_amp} | target_mode=absolute_protocol_aware"
)

ddpm_seq.train_model(
    loader=loader_tr,
    epochs=epochs,
    lr=lr,
    use_amp=use_amp,
    log_every=1,
)


# ------------------------------------------------------------
# 13) Validation loss
# ------------------------------------------------------------
@torch.no_grad()
def _eval_ddpm_loss(ddpm_obj: DDPM, loader) -> float:
    prev_mode = ddpm_obj.net.training
    ema_stored = False

    losses = []
    eps_denom = 1e-6

    try:
        if hasattr(ddpm_obj, "ema") and ddpm_obj.ema is not None:
            ddpm_obj.ema.store(ddpm_obj.net)
            ema_stored = True
            ddpm_obj.ema.copy_to(ddpm_obj.net)

        ddpm_obj.net.eval()

        for batch in loader:
            x0 = batch["x"].to(ddpm_obj.device_obj, dtype=torch.float32)
            xobs = batch["x_obs"].to(ddpm_obj.device_obj, dtype=torch.float32)
            cond_seq_b = batch["cond_seq"].to(ddpm_obj.device_obj, dtype=torch.float32)
            reg_seq_b = batch["regime_seq"].to(ddpm_obj.device_obj, dtype=torch.long)
            tod_seq_b = batch["tod_seq"].to(ddpm_obj.device_obj, dtype=torch.float32)

            ddpm_obj._validate_batch_contract(
                x=x0,
                x_obs=xobs,
                cond_seq=cond_seq_b,
                reg_seq=reg_seq_b,
                tod_seq=tod_seq_b,
                context="Cell8.eval",
            )

            B = int(x0.shape[0])
            t = torch.randint(0, ddpm_obj.T, (B,), device=ddpm_obj.device_obj, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = ddpm_obj.q_sample(x0, t, noise)
            pred = ddpm_obj.net(x_t, t, cond_seq_b, reg_seq_b, tod_seq_b)

            diff2 = (pred - noise) ** 2
            w = xobs * ddpm_obj.feat_w
            denom = torch.sum(w)

            if float(denom.detach().cpu().item()) <= 0.0:
                raise RuntimeError("[Cell8] Validation batch has zero observed weighted entries.")

            loss = torch.sum(diff2 * w) / (denom + eps_denom)

            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu().item()))
            else:
                raise RuntimeError("[Cell8] Non-finite validation loss encountered.")

    finally:
        if ema_stored:
            ddpm_obj.ema.restore(ddpm_obj.net)
        ddpm_obj.net.train(prev_mode)

    if not losses:
        raise RuntimeError("[Cell8] Validation loader produced no losses.")

    return float(np.mean(losses))


val_loss = _eval_ddpm_loss(ddpm_seq, loader_va)
log(f"[Cell8] Validation loss (EMA, absolute protocol-aware target): {val_loss:.6f}")


# ------------------------------------------------------------
# 14) DDPM candidate generation + advisory VAL diagnostics
# ------------------------------------------------------------
def _sample_train_marginal_proxy(col_idx: int, n: int, rng: np.random.Generator):
    mask = xobs_tr[:, col_idx] > 0.5
    vals = Xtr_raw[mask, col_idx]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros(n, dtype=np.float32)
    idx = rng.integers(0, vals.size, size=n)
    return vals[idx].astype(np.float32, copy=False)


def _train_observed_values(col_idx: int):
    mask = xobs_tr[:, col_idx] > 0.5
    vals = Xtr_raw[mask, col_idx]
    vals = vals[np.isfinite(vals)]
    return np.maximum(vals.astype(np.float64, copy=False), 0.0)


def _rank_to_train_marginal(cand: np.ndarray, train_vals: np.ndarray):
    cand = np.asarray(cand, dtype=np.float64).reshape(-1)
    train_vals = _finite_1d(train_vals)
    train_vals = np.maximum(train_vals, 0.0)

    if cand.size == 0:
        return cand.astype(np.float32)

    if train_vals.size < 4:
        return np.maximum(cand, 0.0).astype(np.float32)

    order = np.argsort(cand, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(cand), endpoint=True)

    out = np.quantile(train_vals, ranks)
    return np.maximum(out, 0.0).astype(np.float32)


def _tail_repair_candidate(cand: np.ndarray, train_vals: np.ndarray, tail_q: float = 0.95):
    cand = np.maximum(_finite_1d(cand), 0.0)
    train_vals = np.maximum(_finite_1d(train_vals), 0.0)

    if cand.size == 0:
        return cand.astype(np.float32)

    if train_vals.size < 8:
        return cand.astype(np.float32)

    q = float(np.clip(tail_q, 0.80, 0.995))
    c_q = _safe_quantile(cand, q)
    t_q = _safe_quantile(train_vals, q)

    out = cand.copy()

    if np.isfinite(c_q) and np.isfinite(t_q) and c_q > 0 and t_q > 0:
        high = out >= c_q
        out[high] = out[high] * (t_q / c_q)

    return np.maximum(out, 0.0).astype(np.float32)


def _sparse_rate_align_candidate(cand: np.ndarray, train_vals: np.ndarray, rng: np.random.Generator):
    cand = np.maximum(np.asarray(cand, dtype=np.float64).reshape(-1), 0.0)
    train_vals = np.maximum(_finite_1d(train_vals), 0.0)

    if cand.size == 0:
        return cand.astype(np.float32)

    if train_vals.size < 8:
        return cand.astype(np.float32)

    target_nz = float(np.mean(train_vals > 0))
    curr_nz = float(np.mean(cand > 0))

    out = cand.copy()

    if target_nz <= 0:
        out[:] = 0.0
        return out.astype(np.float32)

    if curr_nz <= 1e-12:
        k = int(round(target_nz * cand.size))
        if k > 0:
            nz_train = train_vals[train_vals > 0]
            if nz_train.size:
                idx = rng.choice(cand.size, size=min(k, cand.size), replace=False)
                out[idx] = rng.choice(nz_train, size=len(idx), replace=True)
        return np.maximum(out, 0.0).astype(np.float32)

    if curr_nz > target_nz:
        nz_idx = np.where(out > 0)[0]
        drop_n = int(round((curr_nz - target_nz) * cand.size))
        if drop_n > 0 and nz_idx.size:
            drop_idx = rng.choice(nz_idx, size=min(drop_n, nz_idx.size), replace=False)
            out[drop_idx] = 0.0

    elif curr_nz < target_nz:
        zero_idx = np.where(out <= 0)[0]
        add_n = int(round((target_nz - curr_nz) * cand.size))
        nz_train = train_vals[train_vals > 0]
        if add_n > 0 and zero_idx.size and nz_train.size:
            add_idx = rng.choice(zero_idx, size=min(add_n, zero_idx.size), replace=False)
            out[add_idx] = rng.choice(nz_train, size=len(add_idx), replace=True)

    return np.maximum(out, 0.0).astype(np.float32)


def _quality_for_candidate(ref, cand):
    ref = np.maximum(_finite_1d(ref), 0.0)
    cand = np.maximum(_finite_1d(cand), 0.0)

    ref_mean = _safe_mean(ref)
    cand_mean = _safe_mean(cand)
    ref_std = _safe_std(ref)
    cand_std = _safe_std(cand)
    ref_nz = _nonzero_rate(ref)
    cand_nz = _nonzero_rate(cand)

    out = {
        "n_ref": int(ref.size),
        "n_cand": int(cand.size),
        "ks": _ks_stat_fast(ref, cand),
        "wasserstein": _wasserstein_1d_fast(ref, cand),
        "ref_mean": ref_mean,
        "cand_mean": cand_mean,
        "mean_abs_error": abs(cand_mean - ref_mean) if np.isfinite(cand_mean) and np.isfinite(ref_mean) else np.nan,
        "ref_std": ref_std,
        "cand_std": cand_std,
        "std_ratio": _ratio(cand_std, ref_std),
        "ref_nonzero_rate": ref_nz,
        "cand_nonzero_rate": cand_nz,
        "nonzero_rate_abs_error": abs(cand_nz - ref_nz) if np.isfinite(cand_nz) and np.isfinite(ref_nz) else np.nan,
        "ref_q50": _safe_quantile(ref, 0.50),
        "cand_q50": _safe_quantile(cand, 0.50),
        "ref_q95": _safe_quantile(ref, 0.95),
        "cand_q95": _safe_quantile(cand, 0.95),
        "ref_q99": _safe_quantile(ref, 0.99),
        "cand_q99": _safe_quantile(cand, 0.99),
        "ref_transition_rate": _transition_rate(ref),
        "cand_transition_rate": _transition_rate(cand),
        "ref_lag1_autocorr": _lag1_autocorr(ref),
        "cand_lag1_autocorr": _lag1_autocorr(cand),
    }

    for q in ["q50", "q95", "q99"]:
        a = out[f"ref_{q}"]
        b = out[f"cand_{q}"]
        out[f"{q}_abs_error"] = abs(b - a) if np.isfinite(a) and np.isfinite(b) else np.nan

    out["transition_rate_abs_error"] = (
        abs(out["cand_transition_rate"] - out["ref_transition_rate"])
        if np.isfinite(out["cand_transition_rate"]) and np.isfinite(out["ref_transition_rate"])
        else np.nan
    )

    out["lag1_autocorr_abs_error"] = (
        abs(out["cand_lag1_autocorr"] - out["ref_lag1_autocorr"])
        if np.isfinite(out["cand_lag1_autocorr"]) and np.isfinite(out["ref_lag1_autocorr"])
        else np.nan
    )

    return out


def _composite_screen_score(metrics: dict, ref_vals):
    ref_vals = _finite_1d(ref_vals)
    if ref_vals.size == 0:
        return float("inf")

    q95 = _safe_quantile(ref_vals, 0.95)
    q25 = _safe_quantile(ref_vals, 0.25)
    q75 = _safe_quantile(ref_vals, 0.75)
    iqr = q75 - q25 if np.isfinite(q75) and np.isfinite(q25) else 0.0
    scale = max(abs(q95) if np.isfinite(q95) else 0.0, abs(iqr), 1.0)

    vals = [
        float(metrics.get("ks", np.nan)),
        float(metrics.get("wasserstein", np.nan)) / scale,
        float(metrics.get("nonzero_rate_abs_error", np.nan)),
        float(metrics.get("q95_abs_error", np.nan)) / scale,
        float(metrics.get("q99_abs_error", np.nan)) / scale,
        abs(float(metrics.get("std_ratio", np.nan)) - 1.0) if np.isfinite(float(metrics.get("std_ratio", np.nan))) else np.nan,
        float(metrics.get("transition_rate_abs_error", np.nan)) if np.isfinite(float(metrics.get("transition_rate_abs_error", np.nan))) else 0.0,
    ]

    if not all(np.isfinite(v) for v in vals):
        return float("inf")

    return float(
        1.00 * vals[0]
        + 0.50 * vals[1]
        + 1.50 * vals[2]
        + 0.30 * vals[3]
        + 0.20 * vals[4]
        + 0.20 * vals[5]
        + 0.10 * vals[6]
    )


def _diagnose_candidate_failure(proxy_metrics, cand_metrics):
    reasons = []

    ks_margin = float(CFG.get("ddpm_val_diag_ks_improve_margin", CFG.get("ddpm_val_screen_ks_improve_margin", 0.005)))
    wass_margin = float(CFG.get("ddpm_val_diag_wasserstein_improve_margin", CFG.get("ddpm_val_screen_wasserstein_improve_margin", 0.0)))
    nz_slack = float(CFG.get("ddpm_val_diag_nonzero_error_slack", CFG.get("ddpm_val_screen_nonzero_error_slack", 0.01)))

    bks = float(proxy_metrics.get("ks", np.nan))
    cks = float(cand_metrics.get("ks", np.nan))
    bw = float(proxy_metrics.get("wasserstein", np.nan))
    cw = float(cand_metrics.get("wasserstein", np.nan))
    bnz = float(proxy_metrics.get("nonzero_rate_abs_error", np.nan))
    cnz = float(cand_metrics.get("nonzero_rate_abs_error", np.nan))

    if not (np.isfinite(cks) and np.isfinite(bks) and cks <= bks - ks_margin):
        reasons.append("ks_worse_or_not_improved_vs_train_marginal_proxy")

    if not (np.isfinite(cw) and np.isfinite(bw) and cw <= bw + wass_margin):
        reasons.append("wasserstein_worse_vs_train_marginal_proxy")

    if not (np.isfinite(cnz) and np.isfinite(bnz) and cnz <= bnz + nz_slack):
        reasons.append("nonzero_rate_worse_vs_train_marginal_proxy")

    ref_q95 = float(cand_metrics.get("ref_q95", np.nan))
    cand_q95 = float(cand_metrics.get("cand_q95", np.nan))
    ref_q99 = float(cand_metrics.get("ref_q99", np.nan))
    cand_q99 = float(cand_metrics.get("cand_q99", np.nan))

    if np.isfinite(ref_q95) and np.isfinite(cand_q95):
        if abs(cand_q95 - ref_q95) > max(2.0, 0.25 * max(abs(ref_q95), 1.0)):
            reasons.append("q95_mismatch")

    if np.isfinite(ref_q99) and np.isfinite(cand_q99):
        if abs(cand_q99 - ref_q99) > max(3.0, 0.35 * max(abs(ref_q99), 1.0)):
            reasons.append("q99_mismatch")

    std_ratio = float(cand_metrics.get("std_ratio", np.nan))
    if np.isfinite(std_ratio) and std_ratio < float(CFG.get("ddpm_val_diag_variance_collapse_ratio", CFG.get("ddpm_val_screen_variance_collapse_ratio", 0.50))):
        reasons.append("variance_collapse")

    ref_tr = float(cand_metrics.get("ref_transition_rate", np.nan))
    cand_tr = float(cand_metrics.get("cand_transition_rate", np.nan))
    if np.isfinite(ref_tr) and np.isfinite(cand_tr) and cand_tr < 0.5 * ref_tr:
        reasons.append("over_smoothing")

    if "variance_collapse" in reasons or "over_smoothing" in reasons:
        reasons.append("architecture_underfit_or_sampler_smoothing")

    if "q95_mismatch" in reasons or "q99_mismatch" in reasons:
        reasons.append("transform_or_tail_mismatch")

    return sorted(set(reasons))


def _parse_float_list_cfg(key: str, default_list: list):
    v = CFG.get(key, default_list)
    if isinstance(v, str):
        return [float(p.strip()) for p in v.split(",") if p.strip()]
    return [float(x) for x in list(v)]


def _parse_clip_list_cfg(key: str, default_list: list):
    v = CFG.get(key, default_list)
    out = []
    if isinstance(v, str):
        parts = [p.strip() for p in v.split(",") if p.strip()]
        for p in parts:
            out.append(None if p.lower() in {"none", "null"} else float(p))
    else:
        for x in list(v):
            out.append(None if x is None else float(x))
    return out


def _parse_str_list_cfg(key: str, default_list: list):
    v = CFG.get(key, default_list)
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    return [str(x) for x in list(v)]


@torch.no_grad()
def _cell8_generate_ddpm_raw_candidate(
    N: int,
    cond_seq,
    reg_seq,
    tod_seq,
    *,
    temperature: float,
    clip_x0,
    overlap_weight: str,
    seed_offset: int,
):
    N = int(N)
    if N <= 0:
        raise RuntimeError("[Cell8] Cannot generate DDPM candidate with N <= 0.")

    screen_seq_len = min(seq_len, N)
    screen_stride = min(stride, max(1, screen_seq_len))

    X_scaled, audit = ddpm_seq.generate(
        N=N,
        D=D,
        cond_seq=np.asarray(cond_seq, dtype=np.float32)[:N],
        reg_seq=np.asarray(reg_seq, dtype=np.int64)[:N],
        tod_seq=np.asarray(tod_seq, dtype=np.float32)[:N],
        seq_len=screen_seq_len,
        stride=screen_stride,
        seed=seed + int(seed_offset),
        use_ema=True,
        clip_x0=clip_x0,
        sequence_batch=int(CFG.get("ddpm_val_screen_sequence_batch", CFG.get("ddpm_downstream_val_selector_sequence_batch", 64))),
        overlap_weight=str(overlap_weight),
        log_every_batches=int(CFG.get("ddpm_val_screen_log_every_batches", 0)),
        temperature=float(temperature),
        return_audit=True,
    )

    if X_scaled.shape != (N, D):
        raise RuntimeError(f"[Cell8] DDPM candidate shape mismatch: got={X_scaled.shape}, expected={(N, D)}")

    X_raw = ddpm_scaler.inverse_transform(X_scaled).astype(np.float32, copy=False)

    if X_raw.shape != (N, D):
        raise RuntimeError(f"[Cell8] inverse DDPM candidate shape mismatch: got={X_raw.shape}, expected={(N, D)}")

    if not np.isfinite(X_raw).all():
        bad = int((~np.isfinite(X_raw)).sum())
        raise RuntimeError(f"[Cell8] DDPM raw candidate contains non-finite values: bad={bad}")

    audit = dict(audit)
    audit["temperature"] = float(temperature)
    audit["clip_x0"] = None if clip_x0 is None else float(clip_x0)
    audit["overlap_weight"] = str(overlap_weight)

    return X_raw, audit


def _cell8_apply_candidate_family(X_abs_raw: np.ndarray, family: str, rng_seed: int = 0) -> np.ndarray:
    """
    Applies TRAIN-only candidate refinements to a raw DDPM absolute candidate.

    This function uses only TRAIN-fitted/observed information.
    The downstream VAL-only selector may call this to generate DDPM candidate families, but the downstream VAL-only selector
    must perform final selection against true A1 on VAL.
    """
    X_abs_raw = np.asarray(X_abs_raw, dtype=np.float32)
    if X_abs_raw.ndim != 2 or X_abs_raw.shape[1] != D:
        raise RuntimeError(f"[Cell8] Candidate family input shape mismatch: {X_abs_raw.shape}")

    rng = np.random.default_rng(int(rng_seed))
    family = str(family)

    out = np.maximum(X_abs_raw.astype(np.float64, copy=True), 0.0)

    for j in range(D):
        train_vals = _train_observed_values(j)

        if family == "ddpm_absolute":
            continue

        if family == "ddpm_train_marginal_anchored":
            out[:, j] = _rank_to_train_marginal(out[:, j], train_vals)

        elif family == "ddpm_tail_repaired":
            out[:, j] = _tail_repair_candidate(out[:, j], train_vals, tail_q=0.95)

        elif family == "ddpm_sparse_rate_aligned":
            out[:, j] = _sparse_rate_align_candidate(out[:, j], train_vals, rng)

        elif family == "ddpm_empirical_blend_light":
            anchored = _rank_to_train_marginal(out[:, j], train_vals)
            out[:, j] = 0.75 * out[:, j] + 0.25 * anchored

        else:
            raise RuntimeError(f"[Cell8] Unknown DDPM candidate family: {family}")

    return np.maximum(out, 0.0).astype(np.float32)


globals()["_cell8_generate_ddpm_raw_candidate"] = _cell8_generate_ddpm_raw_candidate
globals()["_cell8_apply_candidate_family"] = _cell8_apply_candidate_family


DDPM_CANDIDATE_FAMILIES = [
    "ddpm_absolute",
    "ddpm_train_marginal_anchored",
    "ddpm_tail_repaired",
    "ddpm_sparse_rate_aligned",
    "ddpm_empirical_blend_light",
]

globals()["DDPM_CANDIDATE_FAMILIES"] = list(DDPM_CANDIDATE_FAMILIES)

run_val_diag = bool(CFG.get("ddpm_run_val_screen_after_training", True))

val_diag_rows_all_candidates = []
candidate_generate_audits = []
DDPM_ADVISORY_ACCEPT_COLS = []
DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL = {}
DDPM_ADVISORY_RECOMMENDATION_BY_COL = {}
val_diag_n = 0
raw_candidate_grid = []

if run_val_diag:
    max_n = int(CFG.get("ddpm_val_screen_n", min(len(df_va), 50000)))
    val_diag_n = min(max_n, len(df_va))

    temperatures = _parse_float_list_cfg("ddpm_val_screen_temperatures", [0.70, 0.90, 1.10])
    clips = _parse_clip_list_cfg("ddpm_val_screen_clip_x0_values", [4.0, 6.0])
    overlap_weights = _parse_str_list_cfg("ddpm_val_screen_overlap_weights", ["hann", "triangular"])

    max_raw_candidates = int(CFG.get("ddpm_val_screen_max_raw_candidates", 8))

    for temp in temperatures:
        for clip in clips:
            for ow in overlap_weights:
                raw_candidate_grid.append((float(temp), clip, str(ow)))

    if len(raw_candidate_grid) > max_raw_candidates:
        log(
            f"[Cell8][VAL-DIAG] Raw candidate grid has {len(raw_candidate_grid)} combinations; "
            f"truncating to max_raw_candidates={max_raw_candidates}."
        )
        raw_candidate_grid = raw_candidate_grid[:max_raw_candidates]

    total_candidate_count = len(raw_candidate_grid) * len(DDPM_CANDIDATE_FAMILIES)

    log(
        f"[Cell8] Running VAL-only DDPM advisory diagnostics | "
        f"n={val_diag_n} | raw_candidates={len(raw_candidate_grid)} | "
        f"families={len(DDPM_CANDIDATE_FAMILIES)} | total_candidates={total_candidate_count}"
    )

    marginal_proxy_by_col = {}
    marginal_proxy_metrics_by_col = {}

    base_rng = np.random.default_rng(seed + 7301)
    for j, col in enumerate(DDPM_COLS):
        val_mask = xobs_va[:val_diag_n, j] > 0.5
        ref = Xva_raw[:val_diag_n, j][val_mask]
        ref = np.maximum(ref[np.isfinite(ref)], 0.0)

        proxy = _sample_train_marginal_proxy(j, val_diag_n, base_rng)[val_mask]
        proxy = np.maximum(proxy[np.isfinite(proxy)], 0.0)

        marginal_proxy_by_col[col] = proxy
        marginal_proxy_metrics_by_col[col] = _quality_for_candidate(ref, proxy)

    for raw_id, (temp, clip, ow) in enumerate(raw_candidate_grid):
        raw_candidate_id = f"raw{raw_id:02d}_temp{temp}_clip{clip}_ow{ow}"

        Xdd_abs_raw, gen_audit = _cell8_generate_ddpm_raw_candidate(
            N=val_diag_n,
            cond_seq=cond_va,
            reg_seq=reg_va,
            tod_seq=tod_va,
            temperature=temp,
            clip_x0=clip,
            overlap_weight=ow,
            seed_offset=8808 + raw_id * 101,
        )

        gen_audit["raw_candidate_id"] = raw_candidate_id
        candidate_generate_audits.append(gen_audit)

        for family in DDPM_CANDIDATE_FAMILIES:
            candidate_id = f"{raw_candidate_id}_{family}"
            Xcand = _cell8_apply_candidate_family(
                Xdd_abs_raw,
                family=family,
                rng_seed=seed + 9100 + raw_id,
            )

            for j, col in enumerate(DDPM_COLS):
                val_mask = xobs_va[:val_diag_n, j] > 0.5

                ref = Xva_raw[:val_diag_n, j][val_mask]
                ref = np.maximum(ref[np.isfinite(ref)], 0.0)

                cand = Xcand[:val_diag_n, j][val_mask]
                cand = np.maximum(cand[np.isfinite(cand)], 0.0)

                proxy = marginal_proxy_by_col[col]
                proxy_metrics = marginal_proxy_metrics_by_col[col]
                cand_metrics = _quality_for_candidate(ref, cand)

                min_n = int(CFG.get("ddpm_val_screen_min_n", 500))

                row = {
                    "col": str(col),
                    "col_idx": int(j),
                    "tier": str(DDPM_COL_TIERS[j]),
                    "target_transform": str(ddpm_transform_kind[j]),
                    "pre_val_advisory_eligible": bool(col in advisory_cols_pre),
                    "raw_candidate_id": raw_candidate_id,
                    "candidate_family": family,
                    "candidate_id": candidate_id,
                    "temperature": float(temp),
                    "clip_x0": None if clip is None else float(clip),
                    "overlap_weight": str(ow),
                    "n_val_ref": int(ref.size),
                    "n_marginal_proxy": int(proxy.size),
                    "n_candidate": int(cand.size),
                    "diagnostic_decision": "reject",
                    "diagnostic_reason": None,
                    "failure_reasons": [],
                    "pre_val_exclusion_reasons": list(excluded_advisory_cols_pre.get(col, [])),
                    "authoritative_for_downstream_val_selection": False,
                    "selection_note": "The downstream VAL-only selector must compare this candidate against true A1 on VAL.",
                }

                if col not in advisory_cols_pre:
                    row["diagnostic_reason"] = "pre_val_advisory_excluded"
                    row["failure_reasons"] = ["pre_val_advisory_excluded"] + list(excluded_advisory_cols_pre.get(col, []))
                    val_diag_rows_all_candidates.append(row)
                    continue

                if ref.size < min_n or cand.size < min_n or proxy.size < min_n:
                    row["diagnostic_reason"] = "too_few_val_diagnostic_values"
                    row["min_n"] = int(min_n)
                    row["failure_reasons"] = ["too_few_val_diagnostic_values"]
                    val_diag_rows_all_candidates.append(row)
                    continue

                proxy_score = _composite_screen_score(proxy_metrics, ref)
                cand_score = _composite_screen_score(cand_metrics, ref)

                failure_reasons = _diagnose_candidate_failure(proxy_metrics, cand_metrics)

                ks_margin = float(CFG.get("ddpm_val_diag_ks_improve_margin", CFG.get("ddpm_val_screen_ks_improve_margin", 0.005)))
                wass_margin = float(CFG.get("ddpm_val_diag_wasserstein_improve_margin", CFG.get("ddpm_val_screen_wasserstein_improve_margin", 0.0)))
                nz_slack = float(CFG.get("ddpm_val_diag_nonzero_error_slack", CFG.get("ddpm_val_screen_nonzero_error_slack", 0.01)))
                score_ratio = float(CFG.get("ddpm_val_diag_score_ratio", CFG.get("ddpm_val_screen_score_ratio", 0.98)))

                cond_ks = bool(
                    np.isfinite(cand_metrics["ks"])
                    and np.isfinite(proxy_metrics["ks"])
                    and cand_metrics["ks"] <= proxy_metrics["ks"] - ks_margin
                )
                cond_wass = bool(
                    np.isfinite(cand_metrics["wasserstein"])
                    and np.isfinite(proxy_metrics["wasserstein"])
                    and cand_metrics["wasserstein"] <= proxy_metrics["wasserstein"] + wass_margin
                )
                cond_nz = bool(
                    np.isfinite(cand_metrics["nonzero_rate_abs_error"])
                    and np.isfinite(proxy_metrics["nonzero_rate_abs_error"])
                    and cand_metrics["nonzero_rate_abs_error"] <= proxy_metrics["nonzero_rate_abs_error"] + nz_slack
                )
                cond_score = bool(
                    np.isfinite(cand_score)
                    and np.isfinite(proxy_score)
                    and cand_score <= proxy_score * score_ratio
                )
                cond_no_collapse = "variance_collapse" not in failure_reasons

                row.update({
                    "marginal_proxy_score": float(proxy_score) if np.isfinite(proxy_score) else None,
                    "candidate_score": float(cand_score) if np.isfinite(cand_score) else None,
                    "candidate_minus_marginal_proxy_score": (
                        float(cand_score - proxy_score) if np.isfinite(cand_score) and np.isfinite(proxy_score) else None
                    ),

                    "marginal_proxy_ks": proxy_metrics["ks"],
                    "candidate_ks": cand_metrics["ks"],
                    "candidate_minus_marginal_proxy_ks": (
                        cand_metrics["ks"] - proxy_metrics["ks"]
                        if np.isfinite(cand_metrics["ks"]) and np.isfinite(proxy_metrics["ks"])
                        else np.nan
                    ),

                    "marginal_proxy_wasserstein": proxy_metrics["wasserstein"],
                    "candidate_wasserstein": cand_metrics["wasserstein"],
                    "candidate_minus_marginal_proxy_wasserstein": (
                        cand_metrics["wasserstein"] - proxy_metrics["wasserstein"]
                        if np.isfinite(cand_metrics["wasserstein"]) and np.isfinite(proxy_metrics["wasserstein"])
                        else np.nan
                    ),

                    "ref_nonzero_rate": cand_metrics["ref_nonzero_rate"],
                    "marginal_proxy_nonzero_rate": proxy_metrics["cand_nonzero_rate"],
                    "candidate_nonzero_rate": cand_metrics["cand_nonzero_rate"],
                    "marginal_proxy_nonzero_error": proxy_metrics["nonzero_rate_abs_error"],
                    "candidate_nonzero_error": cand_metrics["nonzero_rate_abs_error"],
                    "candidate_minus_marginal_proxy_nonzero_error": (
                        cand_metrics["nonzero_rate_abs_error"] - proxy_metrics["nonzero_rate_abs_error"]
                        if np.isfinite(cand_metrics["nonzero_rate_abs_error"]) and np.isfinite(proxy_metrics["nonzero_rate_abs_error"])
                        else np.nan
                    ),

                    "ref_mean": cand_metrics["ref_mean"],
                    "marginal_proxy_mean": proxy_metrics["cand_mean"],
                    "candidate_mean": cand_metrics["cand_mean"],

                    "ref_std": cand_metrics["ref_std"],
                    "marginal_proxy_std": proxy_metrics["cand_std"],
                    "candidate_std": cand_metrics["cand_std"],
                    "candidate_std_ratio": cand_metrics["std_ratio"],

                    "ref_q50": cand_metrics["ref_q50"],
                    "marginal_proxy_q50": proxy_metrics["cand_q50"],
                    "candidate_q50": cand_metrics["cand_q50"],

                    "ref_q95": cand_metrics["ref_q95"],
                    "marginal_proxy_q95": proxy_metrics["cand_q95"],
                    "candidate_q95": cand_metrics["cand_q95"],

                    "ref_q99": cand_metrics["ref_q99"],
                    "marginal_proxy_q99": proxy_metrics["cand_q99"],
                    "candidate_q99": cand_metrics["cand_q99"],

                    "ref_transition_rate": cand_metrics["ref_transition_rate"],
                    "marginal_proxy_transition_rate": proxy_metrics["cand_transition_rate"],
                    "candidate_transition_rate": cand_metrics["cand_transition_rate"],

                    "ref_lag1_autocorr": cand_metrics["ref_lag1_autocorr"],
                    "marginal_proxy_lag1_autocorr": proxy_metrics["cand_lag1_autocorr"],
                    "candidate_lag1_autocorr": cand_metrics["cand_lag1_autocorr"],

                    "diag_cond_score": bool(cond_score),
                    "diag_cond_ks_improved_vs_proxy": bool(cond_ks),
                    "diag_cond_wasserstein_not_worse_vs_proxy": bool(cond_wass),
                    "diag_cond_nonzero_not_worse_vs_proxy": bool(cond_nz),
                    "diag_cond_no_variance_collapse": bool(cond_no_collapse),
                    "ks_improve_margin": float(ks_margin),
                    "wasserstein_margin": float(wass_margin),
                    "nonzero_error_slack": float(nz_slack),
                    "score_ratio": float(score_ratio),
                })

                if cond_score and cond_ks and cond_wass and cond_nz and cond_no_collapse:
                    row["diagnostic_decision"] = "advisory_pass"
                    row["diagnostic_reason"] = "candidate_beats_train_marginal_proxy"
                    row["failure_reasons"] = []
                else:
                    row["diagnostic_decision"] = "reject"
                    row["diagnostic_reason"] = "+".join(failure_reasons) if failure_reasons else "candidate_not_better_than_marginal_proxy"
                    row["failure_reasons"] = failure_reasons if failure_reasons else ["candidate_not_better_than_marginal_proxy"]

                val_diag_rows_all_candidates.append(row)

            del Xcand

        del Xdd_abs_raw

        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

else:
    log("[Cell8] VAL DDPM advisory diagnostics disabled by CFG.ddpm_run_val_screen_after_training=False.")
    val_diag_n = 0


# ------------------------------------------------------------
# 14b) Best advisory candidate per DDPM column
# ------------------------------------------------------------
if run_val_diag:
    expected_rows = len(DDPM_COLS) * len(DDPM_CANDIDATE_FAMILIES) * len(raw_candidate_grid)
    if len(val_diag_rows_all_candidates) != expected_rows:
        raise RuntimeError(
            f"[Cell8] VAL diagnostic audit incomplete: rows={len(val_diag_rows_all_candidates)} expected={expected_rows}"
        )

    screen_df_all = pd.DataFrame(val_diag_rows_all_candidates)
    best_rows = []

    for col, g in screen_df_all.groupby("col", sort=False):
        gg = g.copy()

        gg["_score_sort"] = pd.to_numeric(gg["candidate_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        gg["_ks_sort"] = pd.to_numeric(gg["candidate_ks"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        gg["_wass_sort"] = pd.to_numeric(gg["candidate_wasserstein"], errors="coerce").replace([np.inf, -np.inf], np.nan)

        gg["_score_sort"] = gg["_score_sort"].fillna(np.inf)
        gg["_ks_sort"] = gg["_ks_sort"].fillna(np.inf)
        gg["_wass_sort"] = gg["_wass_sort"].fillna(np.inf)

        gg["_advisory_sort"] = (gg["diagnostic_decision"].astype(str) != "advisory_pass").astype(int)

        gg = gg.sort_values(
            ["_advisory_sort", "_score_sort", "_ks_sort", "_wass_sort"],
            ascending=True,
        )

        best = gg.iloc[0].drop(labels=["_score_sort", "_ks_sort", "_wass_sort", "_advisory_sort"]).to_dict()
        best_rows.append(best)

    val_diag_rows_best = best_rows

else:
    val_diag_rows_best = []
    for j, c in enumerate(DDPM_COLS):
        reason = "val_diagnostics_disabled_no_val_evidence"

        row = {
            "col": str(c),
            "col_idx": int(j),
            "tier": str(DDPM_COL_TIERS[j]),
            "target_transform": str(ddpm_transform_kind[j]),
            "pre_val_advisory_eligible": bool(c in advisory_cols_pre),
            "diagnostic_decision": "reject",
            "diagnostic_reason": reason,
            "failure_reasons": [reason],
            "candidate_family": "ddpm_absolute",
            "candidate_id": "val_diagnostics_disabled",
            "temperature": 1.0,
            "clip_x0": float(CFG.get("ddpm_clip_x0_eval", 6.0)),
            "overlap_weight": str(CFG.get("ddpm_downstream_val_selector_overlap_weight", "hann")),
            "authoritative_for_downstream_val_selection": False,
            "methodological_guard": (
                "Pre-VAL eligibility is not sufficient evidence for downstream VAL-only selection. "
                "VAL diagnostics disabled, so DDPM candidates are exported for generation only."
            ),
        }
        val_diag_rows_best.append(row)


if len(val_diag_rows_best) != len(DDPM_COLS):
    raise RuntimeError(
        f"[Cell8] Best VAL diagnostic audit incomplete: rows={len(val_diag_rows_best)} expected={len(DDPM_COLS)}"
    )

best_cols = [r["col"] for r in val_diag_rows_best]
if best_cols != list(DDPM_COLS):
    raise RuntimeError(
        "[Cell8] Best VAL diagnostic column order mismatch vs DDPM_COLS. "
        f"first_best={best_cols[:5]} first_ddpm={DDPM_COLS[:5]}"
    )

decision_counts = dict(pd.Series([r.get("diagnostic_decision", "missing") for r in val_diag_rows_best]).value_counts().sort_index())
reason_counts = dict(pd.Series([str(r.get("diagnostic_reason", "missing")) for r in val_diag_rows_best]).value_counts().sort_values(ascending=False))

failure_counter = Counter()
for r in val_diag_rows_best:
    fr = r.get("failure_reasons", [])
    if isinstance(fr, str):
        fr = [fr]
    for x in fr:
        failure_counter[str(x)] += 1

DDPM_ADVISORY_ACCEPT_COLS = []
DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL = {}
DDPM_ADVISORY_EXCLUDED_COLS = {}

for r in val_diag_rows_best:
    col = str(r["col"])

    selected = {
        "col": col,
        "diagnostic_decision": str(r.get("diagnostic_decision", "reject")),
        "diagnostic_reason": str(r.get("diagnostic_reason", "")),
        "candidate_family": str(r.get("candidate_family", "ddpm_absolute")),
        "candidate_id": str(r.get("candidate_id", "")),
        "temperature": r.get("temperature", 1.0),
        "clip_x0": r.get("clip_x0", None),
        "overlap_weight": r.get("overlap_weight", "hann"),
        "candidate_score": r.get("candidate_score", None),
        "marginal_proxy_score": r.get("marginal_proxy_score", None),
        "candidate_ks": r.get("candidate_ks", None),
        "marginal_proxy_ks": r.get("marginal_proxy_ks", None),
        "candidate_wasserstein": r.get("candidate_wasserstein", None),
        "marginal_proxy_wasserstein": r.get("marginal_proxy_wasserstein", None),
        "failure_reasons": r.get("failure_reasons", []),
        "authoritative_for_downstream_val_selection": False,
    }

    DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL[col] = selected

    if str(r.get("diagnostic_decision", "")) == "advisory_pass":
        DDPM_ADVISORY_ACCEPT_COLS.append(col)
    else:
        DDPM_ADVISORY_EXCLUDED_COLS[col] = selected

    DDPM_ADVISORY_RECOMMENDATION_BY_COL[col] = {
        **selected,
        "recommendation": (
            "downstream_VAL_selector_may_evaluate_this_DDPM_candidate_against_true_A1"
            if str(r.get("diagnostic_decision", "")) == "advisory_pass"
            else "downstream_VAL_selector_may_still_evaluate_candidate_but_no_Cell8_advisory_pass"
        ),
        "downstream_val_selector_requirement": "Final selection requires VAL improvement over true A1 temporal block-bootstrap.",
    }

if len(DDPM_ADVISORY_ACCEPT_COLS) + len(DDPM_ADVISORY_EXCLUDED_COLS) != len(DDPM_COLS):
    raise RuntimeError(
        "[Cell8] Advisory diagnostic partition invalid: "
        f"advisory_pass={len(DDPM_ADVISORY_ACCEPT_COLS)} "
        f"excluded={len(DDPM_ADVISORY_EXCLUDED_COLS)} D={len(DDPM_COLS)}"
    )

# Legacy names are intentionally fail-closed to prevent old the downstream VAL-only selector from treating
# Cell 8 diagnostics as final overwrite authorization.
DDPM_CELL11_SAFE_OVERWRITE_COLS = []
DDPM_CELL11_EXCLUDED_OVERWRITE_COLS = {
    c: {
        "reason": "legacy_direct_overwrite_disabled_in_cell8_v18_THESIS",
        "downstream_val_selector_requirement": "Use DDPM candidate registry and compare against true A1 on VAL.",
    }
    for c in DDPM_COLS
}
DDPM_CELL11_SELECTED_CANDIDATE_BY_COL = {}


# ------------------------------------------------------------
# 14c) Persist VAL advisory diagnostics + candidate registry
# ------------------------------------------------------------
DDPM_VAL_DIAGNOSTIC_AUDIT = {
    "version": "ddpm_val_diagnostic_audit_v18_THESIS_advisory_only",
    "policy": (
        "VAL-only advisory diagnostics comparing DDPM-derived candidate families against a TRAIN-only "
        "marginal empirical proxy. This is not final model selection. "
        "Final selection against true A1 temporal block-bootstrap belongs to the downstream VAL-only selector. TEST is not used."
    ),
    "ddpm_role": "conditional_refinement_candidate_generator",
    "target_mode": "absolute_protocol_aware",
    "run": bool(run_val_diag),
    "D": int(len(DDPM_COLS)),
    "val_diag_n": int(val_diag_n) if run_val_diag else None,
    "candidate_families": list(DDPM_CANDIDATE_FAMILIES),
    "raw_candidate_count": int(len(raw_candidate_grid)) if run_val_diag else 0,
    "candidate_generate_audits": candidate_generate_audits,
    "advisory_pass_cols": list(DDPM_ADVISORY_ACCEPT_COLS),
    "advisory_pass_count": int(len(DDPM_ADVISORY_ACCEPT_COLS)),
    "advisory_excluded_cols": dict(DDPM_ADVISORY_EXCLUDED_COLS),
    "advisory_excluded_count": int(len(DDPM_ADVISORY_EXCLUDED_COLS)),
    "decision_counts": decision_counts,
    "reason_counts": reason_counts,
    "failure_reason_counts": dict(failure_counter.most_common()),
    "best_advisory_candidate_by_col": DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL,
    "recommendation_by_col": DDPM_ADVISORY_RECOMMENDATION_BY_COL,
    "rows_best": val_diag_rows_best,
    "rows_all_candidates": val_diag_rows_all_candidates,
    "interpretation": {
        "advisory_pass_zero_means": (
            "DDPM runtime and training can be valid while no DDPM candidate beats a simple TRAIN-marginal proxy."
        ),
        "downstream_val_selector_policy": (
            "Cell 8 exports DDPM candidates and advisory diagnostics only. "
            "the downstream VAL-only selector must perform final VAL-only selection against true A1 temporal block-bootstrap "
            "and all other eligible candidate generators."
        ),
        "why_not_final_selection": [
            "The diagnostic baseline here is a TRAIN-marginal proxy, not true A1.",
            "The true A1 temporal block-bootstrap is constructed in the downstream VAL-only selector.",
            "Final hybrid selection must compare all eligible generator families consistently.",
        ],
    },
}

val_diag_path = os.path.join(ARTDIR, f"ddpm_val_diagnostic_audit_sig{ddpm_sig12}_D{D}.json")
val_diag_csv_path = os.path.join(REP_DIR, f"ddpm_val_diagnostic_audit_sig{ddpm_sig12}_D{D}.csv")
val_diag_all_csv_path = os.path.join(REP_DIR, f"ddpm_val_diagnostic_all_candidates_sig{ddpm_sig12}_D{D}.csv")
recommendation_path = os.path.join(ARTDIR, f"ddpm_refinement_recommendation_sig{ddpm_sig12}_D{D}.json")
recommendation_csv_path = os.path.join(REP_DIR, f"ddpm_refinement_recommendation_sig{ddpm_sig12}_D{D}.csv")
candidate_registry_path = os.path.join(ARTDIR, f"ddpm_candidate_registry_sig{ddpm_sig12}_D{D}.json")

_write_json(val_diag_path, DDPM_VAL_DIAGNOSTIC_AUDIT)
pd.DataFrame(val_diag_rows_best).to_csv(val_diag_csv_path, index=False)

if val_diag_rows_all_candidates:
    pd.DataFrame(val_diag_rows_all_candidates).to_csv(val_diag_all_csv_path, index=False)
else:
    pd.DataFrame(val_diag_rows_best).to_csv(val_diag_all_csv_path, index=False)

_write_json(recommendation_path, DDPM_ADVISORY_RECOMMENDATION_BY_COL)
pd.DataFrame(list(DDPM_ADVISORY_RECOMMENDATION_BY_COL.values())).to_csv(recommendation_csv_path, index=False)

DDPM_CANDIDATE_REGISTRY = {
    "version": "ddpm_candidate_registry_v17",
    "generator_family": "DDPM",
    "generator_role": "conditional_refinement_candidate",
    "selection_authority": "downstream_VAL_selector_only",
    "final_selection_rule": "The downstream VAL-only selector must compare against true A1 on VAL and select only if predefined thresholds are met.",
    "test_usage": "TEST forbidden here; TEST final QA only.",
    "candidate_cols": list(DDPM_COLS),
    "candidate_families": list(DDPM_CANDIDATE_FAMILIES),
    "candidate_generation_functions": {
        "raw_generator": "_cell8_generate_ddpm_raw_candidate",
        "family_transform": "_cell8_apply_candidate_family",
    },
    "required_inputs_for_generation": {
        "cond_seq": "shared conditioning array matching cond_spec",
        "reg_seq": "regime ids",
        "tod_seq": "TOD sin/cos array",
        "N": "target horizon length",
    },
    "artifacts": {
        "checkpoint_path": None,
        "scaler_path": scaler_joblib_path,
        "target_transform_audit_path": target_transform_path,
        "xobs_audit_path": xobs_audit_path,
        "val_diagnostic_audit_path": val_diag_path,
        "val_diagnostic_csv_path": val_diag_csv_path,
        "val_diagnostic_all_candidates_csv_path": val_diag_all_csv_path,
        "recommendation_path": recommendation_path,
    },
    "advisory": {
        "advisory_pass_cols": list(DDPM_ADVISORY_ACCEPT_COLS),
        "advisory_selected_candidate_by_col": dict(DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL),
        "advisory_recommendation_by_col": dict(DDPM_ADVISORY_RECOMMENDATION_BY_COL),
    },
}

_write_json(candidate_registry_path, DDPM_CANDIDATE_REGISTRY)

globals()["DDPM_ADVISORY_ACCEPT_COLS"] = list(DDPM_ADVISORY_ACCEPT_COLS)
globals()["DDPM_ADVISORY_EXCLUDED_COLS"] = dict(DDPM_ADVISORY_EXCLUDED_COLS)
globals()["DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL"] = dict(DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL)
globals()["DDPM_ADVISORY_RECOMMENDATION_BY_COL"] = dict(DDPM_ADVISORY_RECOMMENDATION_BY_COL)

globals()["DDPM_CELL11_SAFE_OVERWRITE_COLS"] = list(DDPM_CELL11_SAFE_OVERWRITE_COLS)
globals()["DDPM_CELL11_EXCLUDED_OVERWRITE_COLS"] = dict(DDPM_CELL11_EXCLUDED_OVERWRITE_COLS)
globals()["DDPM_CELL11_SELECTED_CANDIDATE_BY_COL"] = dict(DDPM_CELL11_SELECTED_CANDIDATE_BY_COL)
globals()["DDPM_CELL11_BEST_CANDIDATE_BY_COL"] = dict(DDPM_CELL11_SELECTED_CANDIDATE_BY_COL)

globals()["DDPM_VAL_DIAGNOSTIC_AUDIT"] = DDPM_VAL_DIAGNOSTIC_AUDIT
globals()["DDPM_VAL_DIAGNOSTIC_AUDIT_PATH"] = val_diag_path
globals()["DDPM_VAL_DIAGNOSTIC_AUDIT_CSV_PATH"] = val_diag_csv_path
globals()["DDPM_VAL_DIAGNOSTIC_ALL_CANDIDATES_CSV_PATH"] = val_diag_all_csv_path
globals()["DDPM_REFINEMENT_RECOMMENDATION_PATH"] = recommendation_path
globals()["DDPM_CANDIDATE_REGISTRY"] = DDPM_CANDIDATE_REGISTRY
globals()["DDPM_CANDIDATE_REGISTRY_PATH"] = candidate_registry_path

log(
    "[Cell8] DDPM advisory diagnostic contract | "
    f"D={len(DDPM_COLS)} | advisory_pass={len(DDPM_ADVISORY_ACCEPT_COLS)} | "
    f"excluded={len(DDPM_ADVISORY_EXCLUDED_COLS)} | "
    f"decision_counts={decision_counts} | "
    f"top_failures={dict(failure_counter.most_common(5))} | "
    f"audit={val_diag_path}"
)


# ------------------------------------------------------------
# 15) Save train spec and checkpoint
# ------------------------------------------------------------
ddpm_train_spec = {
    "version": "ddpm_train_spec_v18_THESIS_advisory_only_candidate_generator",
    "model_family": "SeqDenoiser",
    "ddpm_role": "conditional_refinement_candidate_generator",
    "selection_authority": "downstream_VAL_selector_only",
    "active_model_global": "ddpm_net",
    "active_wrapper_global": "ddpm_seq",
    "compat_model_global": "ddpm_model",

    "sig": ddpm_sig,
    "sig12": ddpm_sig12,
    "D": int(D),
    "ddpm_cols_use": list(DDPM_COLS),
    "ddpm_col_tiers": list(DDPM_COL_TIERS),

    "cond_spec_path": os.path.basename(cond_spec_path),
    "cond_spec_version": str(cond_spec.get("version", "")),
    "cond_sig": cond_sig,
    "cond_sig12": cond_sig12,
    "cond_dim": int(expected_cond_dim),

    "tod_dim": int(SEQ_TOD_DIM),
    "tod_tr_mode": str(tod_tr_mode),
    "tod_va_mode": str(tod_va_mode),

    "x_obs_basis": "Cell6_logical_tier_obs_AND_value_isfinite",
    "x_obs_tiers_superset": list(TIERS_SUPERSET),
    "x_obs_audit_path": os.path.basename(xobs_audit_path),

    "target_mode": "absolute_protocol_aware",
    "target_transform_audit_path": os.path.basename(target_transform_path),
    "target_transform_counts": transform_counts,

    "scaler_type": "ProtocolAwareTargetScaler",
    "scaler_joblib_path": os.path.basename(scaler_joblib_path),
    "center_path": os.path.basename(center_path),
    "scale_path": os.path.basename(scale_path),
    "cap_hi_path": os.path.basename(cap_path),
    "obs_support_path": os.path.basename(support_path),

    "target_support_audit_path": os.path.basename(target_support_path),
    "advisory_candidate_cols_pre_val": list(advisory_cols_pre),
    "excluded_advisory_candidate_cols_pre_val": dict(excluded_advisory_cols_pre),

    "candidate_families": list(DDPM_CANDIDATE_FAMILIES),
    "candidate_registry_path": os.path.basename(candidate_registry_path),
    "val_diagnostic_audit_path": os.path.basename(val_diag_path),
    "val_diagnostic_best_csv_path": os.path.basename(val_diag_csv_path),
    "val_diagnostic_all_candidates_csv_path": os.path.basename(val_diag_all_csv_path),
    "refinement_recommendation_path": os.path.basename(recommendation_path),

    "advisory_pass_cols": list(DDPM_ADVISORY_ACCEPT_COLS),
    "advisory_pass_count": int(len(DDPM_ADVISORY_ACCEPT_COLS)),
    "advisory_selected_candidate_by_col": dict(DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL),

    "legacy_downstream_val_selector_safe_overwrite_cols": [],
    "legacy_direct_overwrite_disabled": True,

    "seq_len": int(seq_len),
    "stride": int(stride),
    "epochs": int(epochs),
    "batch": int(batch_size),
    "timesteps": int(timesteps),
    "lr": float(lr),
    "train_rows": int(Xtr_ddpm.shape[0]),
    "val_rows": int(Xva_ddpm.shape[0]),
    "scale_clip": float(scale_clip),
    "seed": int(seed),
    "device": str(device),

    "net_width": int(net_width),
    "net_depth": int(net_depth),
    "t_emb_dim": int(t_emb_dim),
    "dropout": float(ddpm_dropout),
    "use_attention": bool(use_attention),
    "attention_heads": int(attention_heads),
    "input_width": int(actual_in_width),

    "determinism": {
        "torch_deterministic_algorithms": True,
        "warn_only": bool(deterministic_warn_only),
        "allow_flash_attention": bool(allow_flash_attention),
        "allow_tf32": bool(torch_allow_tf32_effective),
    },

    "val_loss": float(val_loss),
}
_write_json(spec_path, ddpm_train_spec)
globals()["ddpm_train_spec"] = ddpm_train_spec

ckpt_path = os.path.join(ARTDIR, f"ddpm_core_ckpt_sig{ddpm_sig12}_D{D}.pt")

ckpt_spec = {
    "version": "ddpm_ckpt_spec_v18_THESIS_advisory_only_candidate_generator",
    "model_family": "SeqDenoiser",
    "ddpm_role": "conditional_refinement_candidate_generator",
    "selection_authority": "downstream_VAL_selector_only",
    "active_model_global": "ddpm_net",
    "active_wrapper_global": "ddpm_seq",
    "compat_model_global": "ddpm_model",

    "sig": ddpm_sig,
    "sig12": ddpm_sig12,
    "D": int(D),
    "ddpm_cols_use": list(DDPM_COLS),
    "ddpm_col_tiers": list(DDPM_COL_TIERS),

    "cond_spec_path": os.path.basename(cond_spec_path),
    "cond_spec_version": str(cond_spec.get("version", "")),
    "cond_sig": cond_sig,
    "cond_sig12": cond_sig12,
    "cond_dim": int(expected_cond_dim),

    "tod_dim": int(SEQ_TOD_DIM),

    "x_obs_basis": "Cell6_logical_tier_obs_AND_value_isfinite",
    "x_obs_audit_path": os.path.basename(xobs_audit_path),

    "target_mode": "absolute_protocol_aware",
    "target_transform_audit_path": os.path.basename(target_transform_path),

    "scaler_type": "ProtocolAwareTargetScaler",
    "scaler_joblib_path": os.path.basename(scaler_joblib_path),

    "K_reg": int(K_reg_effective),
    "seq_len": int(seq_len),
    "stride": int(stride),
    "timesteps": int(timesteps),
    "scale_clip": float(scale_clip),
    "seed": int(seed),
    "device": str(device),

    "net_width": int(net_width),
    "net_depth": int(net_depth),
    "t_emb_dim": int(t_emb_dim),
    "dropout": float(ddpm_dropout),
    "use_attention": bool(use_attention),
    "attention_heads": int(attention_heads),
    "input_width": int(actual_in_width),

    "candidate_families": list(DDPM_CANDIDATE_FAMILIES),
    "candidate_registry_path": os.path.basename(candidate_registry_path),
    "target_support_audit_path": os.path.basename(target_support_path),
    "val_diagnostic_audit_path": os.path.basename(val_diag_path),
    "advisory_pass_cols": list(DDPM_ADVISORY_ACCEPT_COLS),
    "advisory_selected_candidate_by_col": dict(DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL),
    "legacy_direct_overwrite_disabled": True,
}

state_dict = ddpm_seq.net.state_dict()
ema_shadow = ddpm_seq.ema.shadow

_state_dict_all_finite(state_dict, "model_state_dict")
_state_dict_all_finite(ema_shadow, "ema_shadow")

torch.save(
    {
        "spec": ckpt_spec,
        "model_state_dict": state_dict,
        "ema_shadow": ema_shadow,

        "model_family": "SeqDenoiser",
        "ddpm_role": "conditional_refinement_candidate_generator",
        "selection_authority": "downstream_VAL_selector_only",
        "active_model_global": "ddpm_net",
        "active_wrapper_global": "ddpm_seq",
        "compat_model_global": "ddpm_model",

        "ddpm_sig": ddpm_sig,
        "ddpm_sig12": ddpm_sig12,
        "D": int(D),
        "ddpm_cols_use": list(DDPM_COLS),
        "ddpm_col_tiers": list(DDPM_COL_TIERS),
        "cond_dim": int(expected_cond_dim),
        "cond_sig": cond_sig,
        "tod_dim": int(SEQ_TOD_DIM),
        "K_reg": int(K_reg_effective),
        "val_loss": float(val_loss),
        "x_obs_basis": "Cell6_logical_tier_obs_AND_value_isfinite",
        "target_mode": "absolute_protocol_aware",
        "candidate_families": list(DDPM_CANDIDATE_FAMILIES),
        "candidate_registry_path": os.path.basename(candidate_registry_path),
        "advisory_pass_cols": list(DDPM_ADVISORY_ACCEPT_COLS),
        "advisory_selected_candidate_by_col": dict(DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL),
        "legacy_direct_overwrite_disabled": True,
        "cfg_excerpt": {
            "ddpm_timesteps": int(timesteps),
            "ddpm_epochs": int(epochs),
            "ddpm_batch": int(batch_size),
            "ddpm_lr": float(lr),
            "ddpm_seq_len": int(seq_len),
            "ddpm_stride": int(stride),
            "ddpm_width": int(net_width),
            "ddpm_depth": int(net_depth),
            "ddpm_t_emb_dim": int(t_emb_dim),
            "ddpm_dropout": float(ddpm_dropout),
            "ddpm_use_attention": bool(use_attention),
            "ddpm_attention_heads": int(attention_heads),
            "ddpm_tod_dim": int(SEQ_TOD_DIM),
            "device": str(device),
        },
    },
    ckpt_path,
)

globals()["ddpm_ckpt_path"] = ckpt_path

# Update registry with checkpoint path after checkpoint exists.
DDPM_CANDIDATE_REGISTRY["artifacts"]["checkpoint_path"] = ckpt_path
_write_json(candidate_registry_path, DDPM_CANDIDATE_REGISTRY)

log(f"[Cell8] Saved DDPM training spec: {spec_path}")
log(f"[Cell8] Saved checkpoint: {ckpt_path}")
log(f"[Cell8] Saved DDPM candidate registry: {candidate_registry_path}")


# ------------------------------------------------------------
# 16) Active runtime contract for downstream candidate consumers
# ------------------------------------------------------------
DDPM_ACTIVE_CONTRACT = {
    "version": "ddpm_active_contract_from_cell8_v18_THESIS_advisory_only_candidate_generator",
    "sig12": ddpm_sig12,
    "D": int(D),
    "ddpm_cols_use": list(DDPM_COLS),
    "ddpm_col_tiers": list(DDPM_COL_TIERS),

    "model_family": "SeqDenoiser",
    "ddpm_role": "conditional_refinement_candidate_generator",
    "selection_authority": "downstream_VAL_selector_only",
    "active_model_global": "ddpm_net",
    "active_wrapper_global": "ddpm_seq",
    "compat_model_global": "ddpm_model",

    "cond_dim": int(expected_cond_dim),
    "cond_sig": str(cond_sig),
    "tod_dim": int(SEQ_TOD_DIM),
    "K_reg": int(K_reg_effective),

    "seq_len": int(seq_len),
    "stride": int(stride),
    "timesteps": int(timesteps),
    "input_width": int(actual_in_width),

    "net_width": int(net_width),
    "net_depth": int(net_depth),
    "t_emb_dim": int(t_emb_dim),
    "dropout": float(ddpm_dropout),
    "use_attention": bool(use_attention),
    "attention_heads": int(attention_heads),

    "x_obs_basis": "Cell6_logical_tier_obs_AND_value_isfinite",
    "target_mode": "absolute_protocol_aware",
    "scaler_type": "ProtocolAwareTargetScaler",
    "target_transform_counts": transform_counts,

    "candidate_families": list(DDPM_CANDIDATE_FAMILIES),
    "candidate_registry_path": candidate_registry_path,
    "downstream_val_selector_candidate_policy": (
        "A1 temporal block-bootstrap is the protocol backbone. "
        "DDPM candidates are optional refinements. "
        "Cell 8 diagnostics are advisory only. "
        "The downstream VAL-only selector must compare DDPM candidates against true A1 on VAL before selecting them."
    ),

    "x_obs_audit_path": xobs_audit_path,
    "checkpoint_path": ckpt_path,
    "train_spec_path": spec_path,
    "scaler_path": scaler_joblib_path,
    "target_support_audit_path": target_support_path,
    "target_transform_audit_path": target_transform_path,
    "val_diagnostic_audit_path": val_diag_path,
    "val_diagnostic_csv_path": val_diag_csv_path,
    "val_diagnostic_all_candidates_csv_path": val_diag_all_csv_path,
    "refinement_recommendation_path": recommendation_path,

    "advisory_pass_cols": list(DDPM_ADVISORY_ACCEPT_COLS),
    "advisory_pass_count": int(len(DDPM_ADVISORY_ACCEPT_COLS)),
    "advisory_excluded_cols": dict(DDPM_ADVISORY_EXCLUDED_COLS),
    "advisory_selected_candidate_by_col": dict(DDPM_ADVISORY_SELECTED_CANDIDATE_BY_COL),
    "advisory_recommendation_by_col": dict(DDPM_ADVISORY_RECOMMENDATION_BY_COL),

    "legacy_downstream_val_selector_safe_overwrite_cols": [],
    "legacy_direct_overwrite_disabled": True,

    "val_loss": float(val_loss),
    "device": str(device),

    "methodological_note": (
        "This DDPM is trained on Cell 6 legitimate DDPM value columns only. "
        "It is not an unconditional replacement for A1. "
        "It is a conditional refinement/candidate generator. "
        "Cell 8 does not authorize final overwrite columns. "
        "the downstream VAL-only selector is the sole authoritative selector and must use VAL-only comparison against true A1 "
        "and other eligible generator families. TEST remains final QA only."
    ),
}

globals()["DDPM_ACTIVE_CONTRACT"] = DDPM_ACTIVE_CONTRACT

active_contract_path = os.path.join(ARTDIR, "ddpm_active_contract_from_cell8.json")
_write_json(active_contract_path, DDPM_ACTIVE_CONTRACT)

# ------------------------------------------------------------
# 17) Generator manifest contract for paper/result ledger
# ------------------------------------------------------------
DDPM_CELL8_GENERATOR_CONTRACT = {
    "component": "Cell8_DDPM_candidate_training",
    "component_type": "candidate_generator_training_and_val_advisory_diagnostics",
    "version": "cell8_ddpm_candidate_training_v18_THESIS",
    "generator_family": "DDPM",
    "model_family": "SeqDenoiser",
    "branch_scope": [
        "protocol_candidate_generation",
        "conditional_refinement_candidate_generation",
    ],
    "parameters": {
        "timesteps": int(timesteps),
        "epochs": int(epochs),
        "batch": int(batch_size),
        "lr": float(lr),
        "seq_len": int(seq_len),
        "stride": int(stride),
        "net_width": int(net_width),
        "net_depth": int(net_depth),
        "t_emb_dim": int(t_emb_dim),
        "dropout": float(ddpm_dropout),
        "use_attention": bool(use_attention),
        "attention_heads": int(attention_heads),
        "target_mode": "absolute_protocol_aware",
        "x_obs_policy": "Cell6 logical tier obs AND finite raw target value",
        "conditioning_contract": "Cell7 shared conditioning contract",
        "torch_allow_tf32": bool(torch_allow_tf32_effective),
    },
    "seed": int(seed),
    "train_fit_scope": (
        "TRAIN only: target transform statistics, feature weights, DDPM model "
        "parameters, and checkpoint are fitted/trained using TRAIN data only."
    ),
    "val_selection_metric": (
        "Cell8 computes VAL advisory diagnostics only against a TRAIN-marginal proxy; "
        "authoritative final selection must be performed downstream against true A1 "
        "and all eligible candidate families."
    ),
    "test_only_qa": (
        "No TEST values, TEST features, TEST candidates, or TEST diagnostics are used in Cell8. "
        "TEST is reserved for downstream final QA after TRAIN/VAL decisions are frozen."
    ),
    "contributes_to_final_artifact": False,
    "final_artifact_contribution_rule": (
        "Cell8 exports DDPM candidates and advisory diagnostics only. No DDPM output "
        "enters the final scientific or public artifact unless selected by the downstream "
        "VAL-only selector and then audited on TEST without repair or overwrite."
    ),
    "candidate_or_rejected_label_rule": (
        "DDPM candidates remain candidate/diagnostic artifacts here. Advisory-pass columns "
        "are recommendations for downstream evaluation, not accepted final columns."
    ),
    "leakage_status": {
        "reads_train_values": True,
        "reads_val_values": True,
        "reads_test_values": False,
        "uses_test_for_fitting": False,
        "uses_test_for_threshold_selection": False,
        "uses_test_for_candidate_selection": False,
        "uses_test_for_repair": False,
        "mutates_synthetic_artifact": False,
        "overwrites_final_artifact": False,
    },
    "artifacts": {
        "train_spec_path": spec_path,
        "checkpoint_path": ckpt_path,
        "active_contract_path": active_contract_path,
        "candidate_registry_path": candidate_registry_path,
        "val_diagnostic_audit_path": val_diag_path,
        "target_support_audit_path": target_support_path,
        "x_obs_audit_path": xobs_audit_path,
        "target_transform_audit_path": target_transform_path,
    },
    "advisory_summary": {
        "D": int(D),
        "advisory_pass_count": int(len(DDPM_ADVISORY_ACCEPT_COLS)),
        "advisory_excluded_count": int(len(DDPM_ADVISORY_EXCLUDED_COLS)),
        "val_loss": float(val_loss),
    },
    "paper_claim_status": (
        "Not citable as final quality evidence by itself. Cite only downstream selection "
        "and TEST-QA ledgers if a DDPM candidate is selected under the frozen policy."
    ),
}

ddpm_cell8_contract_path = os.path.join(CONTRACT_DIR, "ddpm_cell8_generator_contract.json")
_write_json(ddpm_cell8_contract_path, DDPM_CELL8_GENERATOR_CONTRACT)

if "RUN_META" in globals():
    RUN_META.setdefault("generator_contracts", {})
    RUN_META["generator_contracts"]["DDPM_Cell8_candidate_training"] = DDPM_CELL8_GENERATOR_CONTRACT

globals()["DDPM_CELL8_GENERATOR_CONTRACT"] = DDPM_CELL8_GENERATOR_CONTRACT
globals()["DDPM_CELL8_GENERATOR_CONTRACT_PATH"] = ddpm_cell8_contract_path

log(f"[Cell8] Saved active DDPM contract: {active_contract_path}")
log(f"[Cell8] Saved DDPM Cell8 generator contract: {ddpm_cell8_contract_path}")

log(
    "[Cell8] Active DDPM contract finalized: "
    f"sig12={ddpm_sig12} | D={D} | cond_dim={expected_cond_dim} | "
    f"model=SeqDenoiser | role=conditional_refinement_candidate_generator | "
    f"selection_authority=downstream_VAL_selector_only | "
    f"seq_len={seq_len} | stride={stride} | timesteps={timesteps} | "
    f"tod_dim={SEQ_TOD_DIM} | input_width={actual_in_width} | "
    f"x_obs=Cell6_logical_value_finite | target_mode=absolute_protocol_aware | "
    f"scaler=protocol_aware | width={net_width} | depth={net_depth} | "
    f"attention={use_attention} | device={device} | "
    f"advisory_pass_cols={len(DDPM_ADVISORY_ACCEPT_COLS)} | "
    f"legacy_direct_overwrite_disabled=True"
)

if len(DDPM_ADVISORY_ACCEPT_COLS) == 0:
    log(
        "[Cell8][DIAGNOSTIC] DDPM produced zero advisory-pass columns versus train-marginal proxy. "
        "This does NOT mean DDPM is invalid. It means the downstream VAL-only selector should only select DDPM if "
        "a DDPM candidate beats true A1 under the final VAL selector."
    )
else:
    log(
        "[Cell8][DIAGNOSTIC] DDPM advisory-pass columns for downstream_VAL_selector evaluation: "
        f"{DDPM_ADVISORY_ACCEPT_COLS}"
    )

if device_obj.type == "cuda":
    log(
        f"[Cell8] GPU memory after training | "
        f"allocated={torch.cuda.memory_allocated(device_obj) / (1024**2):.1f} MiB | "
        f"reserved={torch.cuda.memory_reserved(device_obj) / (1024**2):.1f} MiB"
    )

log("--- END:   Cell 8 — DDPM candidate generator training (v18-THESIS advisory-only) ---")