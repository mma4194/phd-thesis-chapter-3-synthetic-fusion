# ==========================================================
# CELL 17.3 - TimeGAN external baseline
# v1.1 STUDY-THESIS strict comparable-scope temporal baseline, no-Q4-promotion aware
#
# Role:
#   - Train TimeGAN-style sequence baselines on fair temporal scopes from Cell 17.0.
#   - Generate TEST-length synthetic sequences for each eligible sequence scope.
#   - Save baseline artifacts for Cell 17.4 metric computation.
#
# Scientific contract:
#   - TRAIN real values are used for fitting.
#   - VAL real values are not used for model selection here.
#   - TEST real values are NOT used for fitting, selection, or materialization.
#   - Baseline scope is limited to compact sequence scopes.
#   - This is a sequence baseline, not a full CPS / Q3 / Q4 / Q6 generator.
#
# Outputs:
#   synthetic/baselines/timegan_<scope_id>_synthetic_test.parquet
#   reports/cell17_3_timegan_baseline_run_audit.csv
#   reports/cell17_3_timegan_baseline_artifact_registry.csv
#   reports/cell17_3_timegan_baseline_contract.json
#   artifacts/cell17_3_timegan_baseline_manifest.json
# ==========================================================

log("--- START: Cell 17.3 - TimeGAN external baseline (v1.1 no-Q4-promotion strict) ---")

import os
import gc
import json
import math
import hashlib
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_173 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL17_0_BASELINE_SCOPE_REGISTRY_DF",
    "CELL17_0_BASELINE_COLUMN_REGISTRY_DF",
    "CELL17_0_BASELINE_FAIRNESS_CONTRACT",
    "CELL17_0_BASELINE_PLAN",
]
_missing_173 = [k for k in _required_173 if k not in globals()]
if _missing_173:
    raise RuntimeError(f"[Cell17.3] Missing required globals: {_missing_173}")

ORIGINAL_OUTDIR_173 = str(OUTDIR)
ORIGINAL_OUT_SYN_173 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_173 = str(REPORT_DIR)

def _resolve_project_root_173(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        for marker in ["q6_public_reaudit_v1", "q6_public_reaudit"]:
            if marker in parts:
                candidates.append(os.sep.join(parts[:parts.index(marker)]))
        if os.path.basename(p) in {"reports", "synthetic", "artifacts"}:
            candidates.append(os.path.dirname(p))
        else:
            candidates.append(p)

    _env_project_root = os.environ.get("CPS_CANONICAL_PROJECT_ROOT", "").strip()
    if _env_project_root:
        candidates.append(_env_project_root)

    seen = set()
    for c in candidates:
        c = os.path.abspath(str(c))
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "synthetic")):
            return c

    raise RuntimeError("[Cell17.3] Could not resolve canonical project root.")

PROJECT_ROOT_173 = _resolve_project_root_173(ORIGINAL_OUTDIR_173, ORIGINAL_REPORT_DIR_173, ORIGINAL_OUT_SYN_173)
OUT_SYN_BASE_173 = os.path.join(PROJECT_ROOT_173, "synthetic")
REPORT_DIR_BASE_173 = os.path.join(PROJECT_ROOT_173, "reports")
ARTDIR_BASE_173 = os.path.join(PROJECT_ROOT_173, "artifacts")
CONTRACT_DIR_BASE_173 = os.path.join(ARTDIR_BASE_173, "contracts")
BASELINE_DIR = os.path.join(OUT_SYN_BASE_173, "baselines")
BASELINE_META_DIR = os.path.join(ARTDIR_BASE_173, "baselines")

os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(BASELINE_META_DIR, exist_ok=True)
os.makedirs(REPORT_DIR_BASE_173, exist_ok=True)
os.makedirs(ARTDIR_BASE_173, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_173, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

CELL173_VERSION = "cell17_3_timegan_external_baseline_v1_1_no_q4_promotion"

CFG["cell17_3_version"] = CELL173_VERSION
CFG["cell17_3_Q4_final_status"] = "blocked_no_promotion"
CFG["cell17_3_Q4_coupled_artifacts_used"] = False
CFG["cell17_3_train_values_used_for_fitting"] = True
CFG["cell17_3_val_values_used_for_selection"] = False
CFG["cell17_3_TEST_real_values_used"] = False
CFG["cell17_3_TEST_real_values_used_for_materialization"] = False
CFG["cell17_3_pipeline_synthetic_values_mutated"] = False
CFG["cell17_3_external_baseline_values_generated"] = True
CFG["cell17_3_selection_done_here"] = False
CFG["cell17_3_generator_fit_done_here"] = True
CFG["cell17_3_materialization_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell17_3_train_row_cap", 160000)
CFG.setdefault("cell17_3_seq_len", 64)
CFG.setdefault("cell17_3_seq_stride", 32)
CFG.setdefault("cell17_3_epochs_autoencoder", 8)
CFG.setdefault("cell17_3_epochs_adversarial", 8)
CFG.setdefault("cell17_3_batch_size", 128)
CFG.setdefault("cell17_3_hidden_dim", 96)
CFG.setdefault("cell17_3_latent_dim", 32)
CFG.setdefault("cell17_3_noise_dim", 32)
CFG.setdefault("cell17_3_lr", 1e-3)
CFG.setdefault("cell17_3_weight_decay", 1e-6)
CFG.setdefault("cell17_3_grad_clip_norm", 1.0)
CFG.setdefault("cell17_3_supervised_weight", 10.0)
CFG.setdefault("cell17_3_reconstruction_weight", 10.0)
CFG.setdefault("cell17_3_generator_moment_weight", 2.0)
CFG.setdefault("cell17_3_device", "auto")
CFG.setdefault("cell17_3_num_workers", 0)
CFG.setdefault("cell17_3_log_every_epochs", 4)
CFG.setdefault("cell17_3_integer_unique_threshold", 64)
CFG.setdefault("cell17_3_clip_to_train_quantiles", True)
CFG.setdefault("cell17_3_clip_q_low", 0.001)
CFG.setdefault("cell17_3_clip_q_high", 0.999)
CFG.setdefault("cell17_3_fail_if_torch_unavailable", False)
CFG.setdefault("cell17_3_fail_on_scope_failure", False)

TRAIN_ROW_CAP_173 = int(CFG.get("cell17_3_train_row_cap", 160000))
SEQ_LEN_173 = int(CFG.get("cell17_3_seq_len", 64))
SEQ_STRIDE_173 = int(CFG.get("cell17_3_seq_stride", 32))
EPOCHS_AE_173 = int(CFG.get("cell17_3_epochs_autoencoder", 8))
EPOCHS_ADV_173 = int(CFG.get("cell17_3_epochs_adversarial", 8))
BATCH_SIZE_173 = int(CFG.get("cell17_3_batch_size", 128))
HIDDEN_DIM_173 = int(CFG.get("cell17_3_hidden_dim", 96))
LATENT_DIM_173 = int(CFG.get("cell17_3_latent_dim", 32))
NOISE_DIM_173 = int(CFG.get("cell17_3_noise_dim", 32))
LR_173 = float(CFG.get("cell17_3_lr", 1e-3))
WEIGHT_DECAY_173 = float(CFG.get("cell17_3_weight_decay", 1e-6))
GRAD_CLIP_173 = float(CFG.get("cell17_3_grad_clip_norm", 1.0))
INTEGER_UNIQUE_THRESHOLD_173 = int(CFG.get("cell17_3_integer_unique_threshold", 64))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
run_audit_csv = os.path.join(REPORT_DIR_BASE_173, "cell17_3_timegan_baseline_run_audit.csv")
artifact_registry_csv = os.path.join(REPORT_DIR_BASE_173, "cell17_3_timegan_baseline_artifact_registry.csv")
contract_json = os.path.join(REPORT_DIR_BASE_173, "cell17_3_timegan_baseline_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_173, "cell17_3_timegan_baseline_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_173, "cell17_3_timegan_baseline_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_173(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_173(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_173(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_173(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_173(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_173(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_173(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_173(obj.to_dict())
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

def _write_json_173(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_173(payload), f, indent=2, sort_keys=True)

def _sha256_file_173(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_173(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _slug_173(x: str) -> str:
    s = str(x)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")

def _require_17_0_governance_173(contract: dict):
    if not isinstance(contract, dict):
        raise RuntimeError("[Cell17.3] CELL17_0_BASELINE_FAIRNESS_CONTRACT is not a dict.")
    version = str(contract.get("version", ""))
    if "cell17_0_external_baseline_scope_contract_v1_1" not in version:
        raise RuntimeError(f"[Cell17.3] Unexpected Cell17.0 contract version: {version}")
    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})
    if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
        raise RuntimeError("[Cell17.3] Cell17.0 does not carry Q4 blocked_no_promotion governance.")
    if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
        raise RuntimeError("[Cell17.3] Cell17.0 indicates Q4-coupled artifacts were used.")
    return version

def _scope_cols_173(scope_id: str):
    reg = CELL17_0_BASELINE_COLUMN_REGISTRY_DF.copy()
    reg["scope_id"] = reg["scope_id"].astype(str)
    reg["col"] = reg["col"].astype(str)
    cols = reg.loc[reg["scope_id"].eq(str(scope_id)), "col"].tolist()
    return list(dict.fromkeys(cols))

def _numeric_train_frame_173(cols):
    cols = [str(c) for c in cols if str(c) in df_tr.columns]
    if not cols:
        return pd.DataFrame()
    d = df_tr[cols].copy(deep=True)
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    return d.copy(deep=True)

def _fit_preprocess_173(d: pd.DataFrame):
    meta = {}
    out = d.copy(deep=True)

    for c in out.columns.astype(str):
        x = pd.to_numeric(out[c], errors="coerce").astype("float64")
        x_arr = np.array(x.to_numpy(dtype=np.float64, copy=True), dtype=np.float64, copy=True)

        finite_mask = np.isfinite(x_arr)
        finite = x_arr[finite_mask]

        if len(finite) == 0:
            fill = 0.0
            mean = 0.0
            std = 1.0
            qlo = 0.0
            qhi = 0.0
            is_integer_like = True
            is_nonnegative = True
            unique_values = []
        else:
            fill = float(np.nanmedian(finite))
            mean = float(np.nanmean(finite))
            std = float(np.nanstd(finite))
            if not np.isfinite(std) or std <= 1e-9:
                std = 1.0

            qlo = float(np.nanquantile(finite, float(CFG.get("cell17_3_clip_q_low", 0.001))))
            qhi = float(np.nanquantile(finite, float(CFG.get("cell17_3_clip_q_high", 0.999))))

            rounded = np.isclose(finite, np.rint(finite), atol=1e-8)
            is_integer_like = bool(np.mean(rounded) >= 0.999)
            is_nonnegative = bool(np.nanmin(finite) >= 0.0)
            unique = np.unique(finite)
            unique_values = unique.tolist() if len(unique) <= INTEGER_UNIQUE_THRESHOLD_173 else []

        arr = np.array(x_arr, dtype=np.float64, copy=True)
        arr[~np.isfinite(arr)] = fill
        z = (arr - mean) / std

        out[c] = np.array(z, dtype=np.float32, copy=True)

        meta[c] = {
            "fill": fill,
            "mean": mean,
            "std": std,
            "clip_lo": qlo,
            "clip_hi": qhi,
            "is_integer_like": is_integer_like,
            "is_nonnegative": is_nonnegative,
            "unique_values": unique_values,
            "train_min": float(np.nanmin(finite)) if len(finite) else np.nan,
            "train_max": float(np.nanmax(finite)) if len(finite) else np.nan,
            "train_mean": float(np.nanmean(finite)) if len(finite) else np.nan,
            "train_std": float(np.nanstd(finite)) if len(finite) else np.nan,
        }

    return out.copy(deep=True), meta

def _inverse_postprocess_173(z_df: pd.DataFrame, meta: dict, ordered_cols: list):
    out = pd.DataFrame(index=np.arange(len(z_df)))

    for c in ordered_cols:
        if c not in z_df.columns:
            z = np.zeros(len(z_df), dtype=np.float64)
        else:
            z = np.array(
                pd.to_numeric(z_df[c], errors="coerce").to_numpy(dtype=np.float64, copy=True),
                dtype=np.float64,
                copy=True,
            )

        m = meta.get(c, {})
        mean = _safe_float_173(m.get("mean"), 0.0)
        std = _safe_float_173(m.get("std"), 1.0)
        fill = _safe_float_173(m.get("fill"), 0.0)

        x = np.array(z * std + mean, dtype=np.float64, copy=True)
        x[~np.isfinite(x)] = fill

        if bool(CFG.get("cell17_3_clip_to_train_quantiles", True)):
            lo = _safe_float_173(m.get("clip_lo"), np.nan)
            hi = _safe_float_173(m.get("clip_hi"), np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
                x = np.clip(x, lo, hi)

        x = np.array(x, dtype=np.float64, copy=True)

        if bool(m.get("is_nonnegative", False)):
            x = np.maximum(x, 0.0)

        if bool(m.get("is_integer_like", False)):
            x = np.rint(x)

        out[c] = x.astype(np.float32, copy=True)

    return out

def _deterministic_train_prefix_173(d: pd.DataFrame, row_cap: int):
    if len(d) <= int(row_cap):
        return d.reset_index(drop=True).copy()
    # Preserve chronology for sequence baseline.
    return d.iloc[:int(row_cap)].reset_index(drop=True).copy()

def _make_windows_173(x_np: np.ndarray, seq_len: int, stride: int):
    n = x_np.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, x_np.shape[1]), dtype=np.float32)

    starts = np.arange(0, n - seq_len + 1, stride, dtype=np.int64)
    windows = np.stack([x_np[s:s + seq_len, :] for s in starts], axis=0).astype(np.float32)
    return windows

def _stitch_windows_to_length_173(windows: np.ndarray, n_rows: int):
    """
    Convert generated windows to a continuous TEST-length sequence by overlap averaging.
    """
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[0] == 0:
        raise RuntimeError("no_generated_windows_to_stitch")

    seq_len = windows.shape[1]
    dim = windows.shape[2]
    stride = max(1, SEQ_STRIDE_173)

    out = np.zeros((int(n_rows), dim), dtype=np.float64)
    wsum = np.zeros((int(n_rows), 1), dtype=np.float64)

    pos = 0
    wi = 0
    while pos < n_rows:
        win = windows[wi % len(windows)].astype(np.float64)
        end = min(n_rows, pos + seq_len)
        take = end - pos
        out[pos:end, :] += win[:take, :]
        wsum[pos:end, :] += 1.0
        pos += stride
        wi += 1

    wsum = np.maximum(wsum, 1.0)
    out = out / wsum

    # In case the last stride pattern leaves zeros, forward fill from valid rows.
    valid = (wsum[:, 0] > 0)
    if not np.all(valid):
        last = np.zeros(dim, dtype=np.float64)
        for i in range(n_rows):
            if valid[i]:
                last = out[i]
            else:
                out[i] = last

    return out.astype(np.float32)

def _load_torch_173():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, F, DataLoader, TensorDataset, True, ""
    except Exception as e:
        return None, None, None, None, None, False, str(e)

torch, nn, F, DataLoader, TensorDataset, torch_available, torch_error = _load_torch_173()

if not torch_available:
    msg = (
        "[Cell17.3] PyTorch unavailable. No TimeGAN baseline will be generated; "
        "this is recorded as unavailable."
    )
    log(msg)
    if bool(CFG.get("cell17_3_fail_if_torch_unavailable", False)):
        raise RuntimeError(msg)

if torch_available:
    class _Encoder173(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.proj = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x):
            h, _ = self.gru(x)
            return self.proj(h)

    class _Recovery173(nn.Module):
        def __init__(self, latent_dim, hidden_dim, output_dim):
            super().__init__()
            self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
            self.proj = nn.Linear(hidden_dim, output_dim)

        def forward(self, z):
            h, _ = self.gru(z)
            return self.proj(h)

    class _Generator173(nn.Module):
        def __init__(self, noise_dim, hidden_dim, latent_dim):
            super().__init__()
            self.gru = nn.GRU(noise_dim, hidden_dim, batch_first=True)
            self.proj = nn.Linear(hidden_dim, latent_dim)

        def forward(self, e):
            h, _ = self.gru(e)
            return self.proj(h)

    class _Supervisor173(nn.Module):
        def __init__(self, latent_dim, hidden_dim):
            super().__init__()
            self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
            self.proj = nn.Linear(hidden_dim, latent_dim)

        def forward(self, z):
            h, _ = self.gru(z)
            return self.proj(h)

    class _Discriminator173(nn.Module):
        def __init__(self, latent_dim, hidden_dim):
            super().__init__()
            self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
            self.proj = nn.Linear(hidden_dim, 1)

        def forward(self, z):
            h, _ = self.gru(z)
            return self.proj(h).squeeze(-1)

def _resolve_device_173():
    if not torch_available:
        return None
    req = str(CFG.get("cell17_3_device", "auto")).lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)

def _train_timegan_173(train_windows: np.ndarray, scope_seed: int):
    device = _resolve_device_173()

    torch.manual_seed(scope_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(scope_seed)

    x_tensor = torch.tensor(train_windows, dtype=torch.float32)
    ds = TensorDataset(x_tensor)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE_173,
        shuffle=True,
        drop_last=False,
        num_workers=int(CFG.get("cell17_3_num_workers", 0)),
    )

    input_dim = int(train_windows.shape[2])

    E = _Encoder173(input_dim, HIDDEN_DIM_173, LATENT_DIM_173).to(device)
    R = _Recovery173(LATENT_DIM_173, HIDDEN_DIM_173, input_dim).to(device)
    G = _Generator173(NOISE_DIM_173, HIDDEN_DIM_173, LATENT_DIM_173).to(device)
    S = _Supervisor173(LATENT_DIM_173, HIDDEN_DIM_173).to(device)
    D = _Discriminator173(LATENT_DIM_173, HIDDEN_DIM_173).to(device)

    opt_er = torch.optim.AdamW(list(E.parameters()) + list(R.parameters()), lr=LR_173, weight_decay=WEIGHT_DECAY_173)
    opt_gs = torch.optim.AdamW(list(G.parameters()) + list(S.parameters()), lr=LR_173, weight_decay=WEIGHT_DECAY_173)
    opt_d = torch.optim.AdamW(D.parameters(), lr=LR_173, weight_decay=WEIGHT_DECAY_173)

    bce = nn.BCEWithLogitsLoss()

    losses = {
        "autoencoder": [],
        "supervised": [],
        "generator": [],
        "discriminator": [],
    }

    # ------------------------------------------------------
    # Phase 1: autoencoder embedding/recovery
    # ------------------------------------------------------
    E.train(); R.train(); G.train(); S.train(); D.train()

    for epoch in range(1, EPOCHS_AE_173 + 1):
        batch_losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            z = E(xb)
            xr = R(z)
            loss = F.mse_loss(xr, xb)

            opt_er.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_173 and GRAD_CLIP_173 > 0:
                torch.nn.utils.clip_grad_norm_(list(E.parameters()) + list(R.parameters()), GRAD_CLIP_173)
            opt_er.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
        losses["autoencoder"].append(epoch_loss)

        if int(CFG.get("cell17_3_log_every_epochs", 4)) > 0 and (
            epoch == 1 or epoch == EPOCHS_AE_173 or epoch % int(CFG.get("cell17_3_log_every_epochs", 4)) == 0
        ):
            log(f"[Cell17.3]   AE epoch {epoch}/{EPOCHS_AE_173} | loss={epoch_loss:.6f}")

    # ------------------------------------------------------
    # Phase 2: supervised latent dynamics pretraining
    # ------------------------------------------------------
    for epoch in range(1, max(1, EPOCHS_AE_173 // 2) + 1):
        batch_losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            with torch.no_grad():
                z = E(xb)
            z_pred = S(z[:, :-1, :])
            loss = F.mse_loss(z_pred, z[:, 1:, :])

            opt_gs.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_173 and GRAD_CLIP_173 > 0:
                torch.nn.utils.clip_grad_norm_(list(S.parameters()), GRAD_CLIP_173)
            opt_gs.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        losses["supervised"].append(float(np.mean(batch_losses)) if batch_losses else np.nan)

    # ------------------------------------------------------
    # Phase 3: adversarial training
    # ------------------------------------------------------
    sup_w = float(CFG.get("cell17_3_supervised_weight", 10.0))
    rec_w = float(CFG.get("cell17_3_reconstruction_weight", 10.0))
    mom_w = float(CFG.get("cell17_3_generator_moment_weight", 2.0))

    for epoch in range(1, EPOCHS_ADV_173 + 1):
        g_losses = []
        d_losses = []

        for (xb,) in loader:
            xb = xb.to(device)
            bsz, seq_len, _ = xb.shape

            # ------------------------------
            # Train discriminator
            # ------------------------------
            with torch.no_grad():
                z_real = E(xb)
                noise = torch.randn(bsz, seq_len, NOISE_DIM_173, device=device)
                z_fake0 = G(noise)
                z_fake = S(z_fake0)

            d_real = D(z_real)
            d_fake = D(z_fake)
            y_real = torch.ones_like(d_real)
            y_fake = torch.zeros_like(d_fake)

            d_loss = bce(d_real, y_real) + bce(d_fake, y_fake)

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            if GRAD_CLIP_173 and GRAD_CLIP_173 > 0:
                torch.nn.utils.clip_grad_norm_(D.parameters(), GRAD_CLIP_173)
            opt_d.step()

            # ------------------------------
            # Train generator + supervisor
            # ------------------------------
            noise = torch.randn(bsz, seq_len, NOISE_DIM_173, device=device)
            z_fake0 = G(noise)
            z_fake = S(z_fake0)
            x_fake = R(z_fake)

            d_fake_for_g = D(z_fake)
            g_adv = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))

            with torch.no_grad():
                z_real = E(xb)

            sup_loss = F.mse_loss(S(z_real[:, :-1, :]), z_real[:, 1:, :])

            # Moment matching in observed data space to reduce collapse.
            real_mean = xb.mean(dim=(0, 1))
            fake_mean = x_fake.mean(dim=(0, 1))
            real_std = xb.std(dim=(0, 1))
            fake_std = x_fake.std(dim=(0, 1))
            mom_loss = torch.mean(torch.abs(real_mean - fake_mean)) + torch.mean(torch.abs(real_std - fake_std))

            # Mild reconstruction consistency through E/R.
            z_re = E(x_fake)
            x_re = R(z_re)
            rec_loss = F.mse_loss(x_re, x_fake)

            g_loss = g_adv + sup_w * sup_loss + mom_w * mom_loss + 0.1 * rec_w * rec_loss

            opt_gs.zero_grad(set_to_none=True)
            g_loss.backward()
            if GRAD_CLIP_173 and GRAD_CLIP_173 > 0:
                torch.nn.utils.clip_grad_norm_(list(G.parameters()) + list(S.parameters()), GRAD_CLIP_173)
            opt_gs.step()

            g_losses.append(float(g_loss.detach().cpu().item()))
            d_losses.append(float(d_loss.detach().cpu().item()))

        g_epoch = float(np.mean(g_losses)) if g_losses else np.nan
        d_epoch = float(np.mean(d_losses)) if d_losses else np.nan
        losses["generator"].append(g_epoch)
        losses["discriminator"].append(d_epoch)

        if int(CFG.get("cell17_3_log_every_epochs", 4)) > 0 and (
            epoch == 1 or epoch == EPOCHS_ADV_173 or epoch % int(CFG.get("cell17_3_log_every_epochs", 4)) == 0
        ):
            log(f"[Cell17.3]   ADV epoch {epoch}/{EPOCHS_ADV_173} | g_loss={g_epoch:.6f} | d_loss={d_epoch:.6f}")

    return {
        "E": E,
        "R": R,
        "G": G,
        "S": S,
        "D": D,
        "losses": losses,
        "device": str(device),
        "input_dim": input_dim,
    }

def _sample_timegan_windows_173(model_pack, n_windows: int):
    device = torch.device(model_pack["device"])
    G = model_pack["G"]
    S = model_pack["S"]
    R = model_pack["R"]

    G.eval(); S.eval(); R.eval()

    all_windows = []
    remaining = int(n_windows)

    with torch.no_grad():
        while remaining > 0:
            bsz = min(BATCH_SIZE_173, remaining)
            noise = torch.randn(bsz, SEQ_LEN_173, NOISE_DIM_173, device=device)
            z = S(G(noise))
            x = R(z)
            all_windows.append(x.detach().cpu().numpy().astype(np.float32))
            remaining -= bsz

    return np.concatenate(all_windows, axis=0).astype(np.float32)

# ----------------------------------------------------------
# 4) Validate Cell 17.0 fairness governance
# ----------------------------------------------------------
CELL17_0_VERSION_173 = _require_17_0_governance_173(CELL17_0_BASELINE_FAIRNESS_CONTRACT)

# ----------------------------------------------------------
# 5) Determine eligible TimeGAN scopes
# ----------------------------------------------------------
plan = CELL17_0_BASELINE_PLAN.get("TimeGAN", {})
allowed_scopes = list(plan.get("allowed_scopes", []))

scope_registry = CELL17_0_BASELINE_SCOPE_REGISTRY_DF.copy()
scope_registry["scope_id"] = scope_registry["scope_id"].astype(str)

eligible_scopes = []
for sid in allowed_scopes:
    d = scope_registry[scope_registry["scope_id"].eq(str(sid))]
    if len(d) == 0:
        continue
    cols = _scope_cols_173(str(sid))
    if len(cols) == 0:
        continue
    eligible_scopes.append(str(sid))

if not eligible_scopes:
    raise RuntimeError("[Cell17.3] No eligible TimeGAN scopes found from Cell 17.0.")

# ----------------------------------------------------------
# 6) Train/generate per scope
# ----------------------------------------------------------
run_rows = []
artifact_rows = []

log(
    "[Cell17.3] Running TimeGAN baseline | "
    f"torch_available={torch_available} | scopes={eligible_scopes} | "
    f"seq_len={SEQ_LEN_173} | stride={SEQ_STRIDE_173}"
)

for scope_i, scope_id in enumerate(eligible_scopes, start=1):
    cols = _scope_cols_173(scope_id)
    cols = [c for c in cols if c in df_tr.columns]

    out_path = os.path.join(BASELINE_DIR, f"timegan_{_slug_173(scope_id)}_synthetic_test.parquet")
    meta_path = os.path.join(BASELINE_META_DIR, f"cell17_3_timegan_{_slug_173(scope_id)}_metadata.json")

    log(f"[Cell17.3] scope {scope_i}/{len(eligible_scopes)} | {scope_id} | cols={len(cols)}")

    row_base = {
        "baseline": "TimeGAN",
        "scope_id": scope_id,
        "torch_available": bool(torch_available),
        "cols_n": int(len(cols)),
        "train_rows_available": int(len(df_tr)),
        "train_row_cap": int(TRAIN_ROW_CAP_173),
        "generated_rows_target": int(N_TE),
        "output_path": out_path,
        "metadata_path": meta_path,
        "TRAIN_real_values_used_for_fitting": True,
        "VAL_real_values_used_for_selection": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "pipeline_synthetic_values_mutated": False,
        "external_baseline_values_generated": bool(torch_available),
    }

    if not torch_available:
        run_rows.append({
            **row_base,
            "status": "not_run",
            "reason": "torch_unavailable",
            "train_rows_used": 0,
            "train_windows_used": 0,
            "generated_rows": 0,
            "fit_succeeded": False,
            "sample_succeeded": False,
            "device": "",
            "final_autoencoder_loss": np.nan,
            "final_generator_loss": np.nan,
            "final_discriminator_loss": np.nan,
            "error": torch_error,
        })
        artifact_rows.append({
            "baseline": "TimeGAN",
            "scope_id": scope_id,
            "path": out_path,
            "exists": False,
            "rows": 0,
            "cols": int(len(cols)),
            "sha256": "",
            "status": "not_run",
            "reason": "torch_unavailable",
        })
        continue

    try:
        train_raw = _numeric_train_frame_173(cols)
        if train_raw.empty or train_raw.shape[1] == 0:
            raise RuntimeError("empty_training_frame_after_scope_column_resolution")

        train_z, meta = _fit_preprocess_173(train_raw)
        train_fit = _deterministic_train_prefix_173(train_z, row_cap=TRAIN_ROW_CAP_173)

        x_np = train_fit.to_numpy(dtype=np.float32, copy=True)
        windows = _make_windows_173(x_np, seq_len=SEQ_LEN_173, stride=SEQ_STRIDE_173)

        if windows.shape[0] == 0:
            raise RuntimeError(
                f"no_training_windows: train_rows={len(train_fit)} seq_len={SEQ_LEN_173} stride={SEQ_STRIDE_173}"
            )

        model_pack = _train_timegan_173(
            train_windows=windows,
            scope_seed=SEED + scope_i * 4073,
        )

        n_windows_needed = int(math.ceil(N_TE / max(1, SEQ_STRIDE_173))) + 2
        gen_windows = _sample_timegan_windows_173(model_pack, n_windows=n_windows_needed)
        z_seq = _stitch_windows_to_length_173(gen_windows, n_rows=N_TE)

        z_df = pd.DataFrame(z_seq, columns=list(train_fit.columns))
        sample_pp = _inverse_postprocess_173(
            z_df=z_df,
            meta=meta,
            ordered_cols=list(train_fit.columns),
        )

        if len(sample_pp) != N_TE:
            raise RuntimeError(f"generated_row_mismatch: got={len(sample_pp)} expected={N_TE}")

        sample_pp.to_parquet(out_path, index=False)

        losses = model_pack["losses"]
        final_ae = float(losses["autoencoder"][-1]) if losses.get("autoencoder") else np.nan
        final_g = float(losses["generator"][-1]) if losses.get("generator") else np.nan
        final_d = float(losses["discriminator"][-1]) if losses.get("discriminator") else np.nan

        meta_payload = {
            "cell": "17.3",
            "version": CELL173_VERSION,
            "baseline": "TimeGAN",
            "scope_id": scope_id,
            "columns": list(train_fit.columns),
            "train_rows_used": int(len(train_fit)),
            "train_windows_used": int(len(windows)),
            "generated_rows": int(len(sample_pp)),
            "generated_windows": int(len(gen_windows)),
            "device": model_pack["device"],
            "losses": losses,
            "final_autoencoder_loss": final_ae,
            "final_generator_loss": final_g,
            "final_discriminator_loss": final_d,
            "config": {
                "train_row_cap": int(TRAIN_ROW_CAP_173),
                "seq_len": int(SEQ_LEN_173),
                "seq_stride": int(SEQ_STRIDE_173),
                "epochs_autoencoder": int(EPOCHS_AE_173),
                "epochs_adversarial": int(EPOCHS_ADV_173),
                "batch_size": int(BATCH_SIZE_173),
                "hidden_dim": int(HIDDEN_DIM_173),
                "latent_dim": int(LATENT_DIM_173),
                "noise_dim": int(NOISE_DIM_173),
                "lr": float(LR_173),
                "weight_decay": float(WEIGHT_DECAY_173),
            },
            "column_metadata": meta,
            "fairness_scope": scope_registry.loc[
                scope_registry["scope_id"].eq(scope_id)
            ].to_dict("records"),
            "q4_governance": {
                "final_q4_status": "blocked_no_promotion",
                "q4_coupled_artifacts_used": False,
            },
            "strict_contract": {
                "TRAIN_real_values_used_for_fitting": True,
                "VAL_real_values_used_for_selection": False,
                "TEST_real_values_used": False,
                "TEST_real_values_used_for_materialization": False,
                "pipeline_synthetic_values_mutated": False,
        "external_baseline_values_generated": bool(torch_available),
                "selection_done_here": False,
            },
        }
        _write_json_173(meta_path, meta_payload)

        run_rows.append({
            **row_base,
            "status": "success",
            "reason": "timegan_fit_sample_completed",
            "train_rows_used": int(len(train_fit)),
            "train_windows_used": int(len(windows)),
            "generated_rows": int(len(sample_pp)),
            "fit_succeeded": True,
            "sample_succeeded": True,
            "device": model_pack["device"],
            "final_autoencoder_loss": final_ae,
            "final_generator_loss": final_g,
            "final_discriminator_loss": final_d,
            "error": "",
        })

        artifact_rows.append({
            "baseline": "TimeGAN",
            "scope_id": scope_id,
            "path": out_path,
            "exists": True,
            "rows": int(len(sample_pp)),
            "cols": int(sample_pp.shape[1]),
            "sha256": _sha256_file_173(out_path),
            "metadata_path": meta_path,
            "metadata_sha256": _sha256_file_173(meta_path),
            "status": "success",
            "reason": "timegan_fit_sample_completed",
        })

        del train_raw, train_z, train_fit, x_np, windows, model_pack, gen_windows, z_seq, z_df, sample_pp
        if torch_available and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        log(f"[Cell17.3] WARNING: TimeGAN scope failed | scope={scope_id} | error={err}")

        run_rows.append({
            **row_base,
            "status": "failed",
            "reason": "timegan_scope_exception",
            "train_rows_used": 0,
            "train_windows_used": 0,
            "generated_rows": 0,
            "fit_succeeded": False,
            "sample_succeeded": False,
            "device": str(_resolve_device_173()) if torch_available else "",
            "final_autoencoder_loss": np.nan,
            "final_generator_loss": np.nan,
            "final_discriminator_loss": np.nan,
            "error": err,
        })

        artifact_rows.append({
            "baseline": "TimeGAN",
            "scope_id": scope_id,
            "path": out_path,
            "exists": False,
            "rows": 0,
            "cols": int(len(cols)),
            "sha256": "",
            "metadata_path": meta_path,
            "metadata_sha256": "",
            "status": "failed",
            "reason": err,
        })

        if torch_available and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if bool(CFG.get("cell17_3_fail_on_scope_failure", False)):
            raise

run_audit_df = pd.DataFrame(run_rows)
artifact_registry_df = pd.DataFrame(artifact_rows)

# ----------------------------------------------------------
# 7) Save registries
# ----------------------------------------------------------
run_audit_df.to_csv(run_audit_csv, index=False)
artifact_registry_df.to_csv(artifact_registry_csv, index=False)

success_n = int((run_audit_df["status"].astype(str) == "success").sum()) if len(run_audit_df) else 0
failed_n = int((run_audit_df["status"].astype(str) == "failed").sum()) if len(run_audit_df) else 0
not_run_n = int((run_audit_df["status"].astype(str) == "not_run").sum()) if len(run_audit_df) else 0

# ----------------------------------------------------------
# 8) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "17.3",
    "version": CELL173_VERSION,
    "role": "timegan_external_baseline_no_q4_promotion",
    "baseline": "TimeGAN",
    "torch_available": bool(torch_available),
    "eligible_scopes": eligible_scopes,
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "baseline_not_compared_to_Q4_coupled_artifact": True,
    },
    "q6_public_context": CELL17_0_BASELINE_FAIRNESS_CONTRACT.get("q6_public_context", {}),
    "upstream_contract_versions": {
        "cell17_0": CELL17_0_VERSION_173,
    },
    "summary": {
        "scopes_total": int(len(eligible_scopes)),
        "success_n": int(success_n),
        "failed_n": int(failed_n),
        "not_run_n": int(not_run_n),
        "generated_test_rows_per_successful_scope": int(N_TE),
    },
    "fairness_contract": {
        "allowed_scopes": allowed_scopes,
        "allowed_metrics": plan.get("allowed_metrics", []),
        "not_allowed_claims": plan.get("not_allowed_claims", []),
        "do_not_compare_against_full_pipeline": True,
        "do_not_claim_q3_q4_or_q6_capability": True,
        "do_not_claim_public_release_privacy_pass": True,
    },
    "timegan_config": {
        "train_row_cap": int(TRAIN_ROW_CAP_173),
        "seq_len": int(SEQ_LEN_173),
        "seq_stride": int(SEQ_STRIDE_173),
        "epochs_autoencoder": int(EPOCHS_AE_173),
        "epochs_adversarial": int(EPOCHS_ADV_173),
        "batch_size": int(BATCH_SIZE_173),
        "hidden_dim": int(HIDDEN_DIM_173),
        "latent_dim": int(LATENT_DIM_173),
        "noise_dim": int(NOISE_DIM_173),
        "lr": float(LR_173),
        "weight_decay": float(WEIGHT_DECAY_173),
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TRAIN_real_values_used_for_fitting": True,
        "VAL_real_values_used_for_selection": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "pipeline_synthetic_values_mutated": False,
        "external_baseline_values_generated": bool(torch_available),
        "selection_done_here": False,
        "generator_fit_done_here": bool(torch_available and success_n > 0),
        "materialization_done_here": bool(success_n > 0),
        "external_baseline_only": True,
    },
    "outputs": {
        "run_audit_csv": run_audit_csv,
        "artifact_registry_csv": artifact_registry_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_173(contract_json, contract)
_write_json_173(contract_canonical_json, contract)

manifest = {
    "cell": "17.3",
    "version": CELL173_VERSION,
    "baseline": "TimeGAN",
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "artifact_registry": artifact_registry_df.to_dict("records"),
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
    "fairness_contract": contract["fairness_contract"],
}

_write_json_173(manifest_json, manifest)

hashes = {
    "run_audit_csv_sha256": _sha256_file_173(run_audit_csv),
    "artifact_registry_csv_sha256": _sha256_file_173(artifact_registry_csv),
    "contract_json_sha256": _sha256_file_173(contract_json),
    "contract_canonical_json_sha256": _sha256_file_173(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_173(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_173(contract_json, contract)
_write_json_173(contract_canonical_json, contract)
_write_json_173(manifest_json, manifest)

# ----------------------------------------------------------
# 9) Export globals
# ----------------------------------------------------------
globals()["CELL173_VERSION"] = CELL173_VERSION
globals()["CELL17_3_TIMEGAN_RUN_AUDIT_DF"] = run_audit_df
globals()["CELL17_3_TIMEGAN_ARTIFACT_REGISTRY_DF"] = artifact_registry_df
globals()["CELL17_3_TIMEGAN_BASELINE_CONTRACT"] = contract

globals()["CELL17_3_TIMEGAN_RUN_AUDIT_CSV"] = run_audit_csv
globals()["CELL17_3_TIMEGAN_ARTIFACT_REGISTRY_CSV"] = artifact_registry_csv
globals()["CELL17_3_TIMEGAN_CONTRACT_JSON"] = contract_json
globals()["CELL17_3_TIMEGAN_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL17_3_TIMEGAN_MANIFEST_JSON"] = manifest_json

log(
    "[Cell17.3] TimeGAN baseline complete | "
    f"torch_available={torch_available} | "
    f"success={success_n} | failed={failed_n} | not_run={not_run_n} | q4_status=blocked_no_promotion"
)
log(f"[Cell17.3] Run audit saved: {run_audit_csv}")
log(f"[Cell17.3] Artifact registry saved: {artifact_registry_csv}")
log(f"[Cell17.3] Contract saved: {contract_json}")
log(f"[Cell17.3] Canonical contract saved: {contract_canonical_json}")
log(
    "[Cell17.3] Contract flags | "
    "TRAIN_real_values_used_for_fitting=True | "
    "VAL_real_values_used_for_selection=False | "
    "TEST_real_values_used=False | "
    "TEST_real_values_used_for_materialization=False | "
    "pipeline_synthetic_values_mutated=False | "
    "selection_done_here=False | "
    f"generator_fit_done_here={bool(torch_available and success_n > 0)} | "
    f"materialization_done_here={bool(success_n > 0)} | "
    "Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False | "
    "external_baseline_only=True"
)
log("--- END: Cell 17.3 - TimeGAN external baseline (v1.1 no-Q4-promotion strict) ---")

gc.collect()