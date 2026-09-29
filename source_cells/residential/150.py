# ==========================================================
# CELL 17.2 - TabDDPM external baseline
# v1.1 STUDY-THESIS strict comparable-scope tabular diffusion baseline, no-Q4-promotion aware
#
# Role:
#   - Train TabDDPM-style tabular diffusion baselines on fair tabular scopes
#     defined in Cell 17.0.
#   - Generate TEST-length synthetic samples for each eligible tabular scope.
#   - Save baseline artifacts for Cell 17.4 metric computation.
#
# Final governance inherited from Cell 17.0:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative or used as a baseline target.
#   - Public candidate is direct-privacy warning but strict-release blocked by
#     synthetic-real distinguishability.
#
# Scientific contract:
#   - TRAIN real values are used for TabDDPM fitting.
#   - VAL real values are not used for model selection here.
#   - TEST real values are NOT used for fitting, selection, or materialization.
#   - Baseline scope is limited to comparable tabular scopes.
#   - This is a numeric TabDDPM-style DDPM baseline, not a full CPS generator.
#   - It is not compared against full role-aware/mask-aware/Q4-coupled pipeline.
#   - If PyTorch is unavailable, this cell records an unavailable baseline;
#     it does NOT silently substitute another model and call it TabDDPM.
#
# Outputs:
#   synthetic/baselines/tabddpm_<scope_id>_synthetic_test.parquet
#   reports/cell17_2_tabddpm_baseline_run_audit.csv
#   reports/cell17_2_tabddpm_baseline_artifact_registry.csv
#   reports/cell17_2_tabddpm_baseline_contract.json
#   artifacts/contracts/cell17_2_tabddpm_baseline_contract_v1_1_THESIS.json
#   artifacts/cell17_2_tabddpm_baseline_manifest.json
# ==========================================================

log("--- START: Cell 17.2 - TabDDPM external baseline (v1.1 no-Q4-promotion strict) ---")

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
_required_172 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL17_0_BASELINE_SCOPE_REGISTRY_DF",
    "CELL17_0_BASELINE_COLUMN_REGISTRY_DF",
    "CELL17_0_BASELINE_FAIRNESS_CONTRACT",
    "CELL17_0_BASELINE_PLAN",
]
_missing_172 = [k for k in _required_172 if k not in globals()]
if _missing_172:
    raise RuntimeError(f"[Cell17.2] Missing required globals: {_missing_172}")

ORIGINAL_OUTDIR_172 = str(OUTDIR)
ORIGINAL_OUT_SYN_172 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_172 = str(REPORT_DIR)

def _resolve_project_root_172(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell17.2] Could not resolve canonical project root.")

PROJECT_ROOT_172 = _resolve_project_root_172(ORIGINAL_OUTDIR_172, ORIGINAL_REPORT_DIR_172, ORIGINAL_OUT_SYN_172)
OUT_SYN_BASE_172 = os.path.join(PROJECT_ROOT_172, "synthetic")
REPORT_DIR_BASE_172 = os.path.join(PROJECT_ROOT_172, "reports")
ARTDIR_BASE_172 = os.path.join(PROJECT_ROOT_172, "artifacts")
CONTRACT_DIR_BASE_172 = os.path.join(ARTDIR_BASE_172, "contracts")
BASELINE_DIR = os.path.join(OUT_SYN_BASE_172, "baselines")
BASELINE_META_DIR = os.path.join(ARTDIR_BASE_172, "baselines")

os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(BASELINE_META_DIR, exist_ok=True)
os.makedirs(REPORT_DIR_BASE_172, exist_ok=True)
os.makedirs(ARTDIR_BASE_172, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_172, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

CELL172_VERSION = "cell17_2_tabddpm_external_baseline_v1_1_no_q4_promotion"

CFG["cell17_2_version"] = CELL172_VERSION
CFG["cell17_2_Q4_final_status"] = "blocked_no_promotion"
CFG["cell17_2_Q4_coupled_artifacts_used"] = False
CFG["cell17_2_train_values_used_for_fitting"] = True
CFG["cell17_2_val_values_used_for_selection"] = False
CFG["cell17_2_TEST_real_values_used"] = False
CFG["cell17_2_TEST_real_values_used_for_materialization"] = False
CFG["cell17_2_pipeline_synthetic_values_mutated"] = False
CFG["cell17_2_external_baseline_values_generated"] = True
CFG["cell17_2_selection_done_here"] = False
CFG["cell17_2_generator_fit_done_here"] = True
CFG["cell17_2_materialization_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell17_2_train_row_cap", 80000)
CFG.setdefault("cell17_2_epochs", 12)
CFG.setdefault("cell17_2_batch_size", 1024)
CFG.setdefault("cell17_2_hidden_dim", 256)
CFG.setdefault("cell17_2_time_embed_dim", 64)
CFG.setdefault("cell17_2_lr", 1e-3)
CFG.setdefault("cell17_2_weight_decay", 1e-5)
CFG.setdefault("cell17_2_diffusion_steps", 100)
CFG.setdefault("cell17_2_sampling_steps", 100)
CFG.setdefault("cell17_2_beta_start", 1e-4)
CFG.setdefault("cell17_2_beta_end", 0.02)
CFG.setdefault("cell17_2_grad_clip_norm", 1.0)
CFG.setdefault("cell17_2_device", "auto")
CFG.setdefault("cell17_2_integer_unique_threshold", 64)
CFG.setdefault("cell17_2_clip_to_train_quantiles", True)
CFG.setdefault("cell17_2_clip_q_low", 0.001)
CFG.setdefault("cell17_2_clip_q_high", 0.999)
CFG.setdefault("cell17_2_fail_if_torch_unavailable", False)
CFG.setdefault("cell17_2_fail_on_scope_failure", False)
CFG.setdefault("cell17_2_num_workers", 0)
CFG.setdefault("cell17_2_log_every_epochs", 4)

TRAIN_ROW_CAP_172 = int(CFG.get("cell17_2_train_row_cap", 80000))
EPOCHS_172 = int(CFG.get("cell17_2_epochs", 12))
BATCH_SIZE_172 = int(CFG.get("cell17_2_batch_size", 1024))
HIDDEN_DIM_172 = int(CFG.get("cell17_2_hidden_dim", 256))
TIME_EMBED_DIM_172 = int(CFG.get("cell17_2_time_embed_dim", 64))
LR_172 = float(CFG.get("cell17_2_lr", 1e-3))
WEIGHT_DECAY_172 = float(CFG.get("cell17_2_weight_decay", 1e-5))
T_STEPS_172 = int(CFG.get("cell17_2_diffusion_steps", 100))
SAMPLING_STEPS_172 = int(CFG.get("cell17_2_sampling_steps", T_STEPS_172))
BETA_START_172 = float(CFG.get("cell17_2_beta_start", 1e-4))
BETA_END_172 = float(CFG.get("cell17_2_beta_end", 0.02))
GRAD_CLIP_172 = float(CFG.get("cell17_2_grad_clip_norm", 1.0))
INTEGER_UNIQUE_THRESHOLD_172 = int(CFG.get("cell17_2_integer_unique_threshold", 64))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
run_audit_csv = os.path.join(REPORT_DIR_BASE_172, "cell17_2_tabddpm_baseline_run_audit.csv")
artifact_registry_csv = os.path.join(REPORT_DIR_BASE_172, "cell17_2_tabddpm_baseline_artifact_registry.csv")
contract_json = os.path.join(REPORT_DIR_BASE_172, "cell17_2_tabddpm_baseline_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_172, "cell17_2_tabddpm_baseline_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_172, "cell17_2_tabddpm_baseline_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_172(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_172(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_172(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_172(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_172(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_172(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_172(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_172(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_172(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_172(payload), f, indent=2, sort_keys=True)

def _sha256_file_172(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_172(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _slug_172(x: str) -> str:
    s = str(x)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")

def _require_17_0_governance_172(contract: dict):
    if not isinstance(contract, dict):
        raise RuntimeError("[Cell17.2] CELL17_0_BASELINE_FAIRNESS_CONTRACT is not a dict.")
    version = str(contract.get("version", ""))
    if "cell17_0_external_baseline_scope_contract_v1_1" not in version:
        raise RuntimeError(f"[Cell17.2] Unexpected Cell17.0 contract version: {version}")
    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})
    if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
        raise RuntimeError("[Cell17.2] Cell17.0 does not carry Q4 blocked_no_promotion governance.")
    if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
        raise RuntimeError("[Cell17.2] Cell17.0 indicates Q4-coupled artifacts were used.")
    return version

def _scope_cols_172(scope_id: str):
    reg = CELL17_0_BASELINE_COLUMN_REGISTRY_DF.copy()
    reg["scope_id"] = reg["scope_id"].astype(str)
    reg["col"] = reg["col"].astype(str)
    cols = reg.loc[reg["scope_id"].eq(str(scope_id)), "col"].tolist()
    return list(dict.fromkeys(cols))

def _numeric_train_frame_172(cols):
    cols = [str(c) for c in cols if str(c) in df_tr.columns]
    if not cols:
        return pd.DataFrame()

    d = df_tr[cols].copy(deep=True)
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float64")
    return d.copy(deep=True)

def _fit_preprocess_172(d: pd.DataFrame):
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

            qlo = float(np.nanquantile(finite, float(CFG.get("cell17_2_clip_q_low", 0.001))))
            qhi = float(np.nanquantile(finite, float(CFG.get("cell17_2_clip_q_high", 0.999))))

            rounded = np.isclose(finite, np.rint(finite), atol=1e-8)
            is_integer_like = bool(np.mean(rounded) >= 0.999)
            is_nonnegative = bool(np.nanmin(finite) >= 0.0)
            unique = np.unique(finite)
            unique_values = unique.tolist() if len(unique) <= INTEGER_UNIQUE_THRESHOLD_172 else []

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

def _inverse_postprocess_172(z_df: pd.DataFrame, meta: dict, ordered_cols: list):
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
        mean = _safe_float_172(m.get("mean"), 0.0)
        std = _safe_float_172(m.get("std"), 1.0)
        fill = _safe_float_172(m.get("fill"), 0.0)

        x = np.array(z * std + mean, dtype=np.float64, copy=True)
        x[~np.isfinite(x)] = fill

        if bool(CFG.get("cell17_2_clip_to_train_quantiles", True)):
            lo = _safe_float_172(m.get("clip_lo"), np.nan)
            hi = _safe_float_172(m.get("clip_hi"), np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
                x = np.clip(x, lo, hi)

        x = np.array(x, dtype=np.float64, copy=True)

        if bool(m.get("is_nonnegative", False)):
            x = np.maximum(x, 0.0)

        if bool(m.get("is_integer_like", False)):
            x = np.rint(x)

        out[c] = x.astype(np.float32, copy=True)

    return out

def _deterministic_train_sample_172(d: pd.DataFrame, row_cap: int, seed: int):
    if len(d) <= int(row_cap):
        return d.reset_index(drop=True).copy()
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(np.arange(len(d)), size=int(row_cap), replace=False)
    idx = np.sort(idx)
    return d.iloc[idx].reset_index(drop=True).copy()

def _load_torch_172():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, F, DataLoader, TensorDataset, True, ""
    except Exception as e:
        return None, None, None, None, None, False, str(e)

# ----------------------------------------------------------
# 4) Torch model helpers
# ----------------------------------------------------------
torch, nn, F, DataLoader, TensorDataset, torch_available, torch_error = _load_torch_172()

if not torch_available:
    msg = (
        "[Cell17.2] PyTorch unavailable. No TabDDPM baseline will be generated; "
        "this is recorded as unavailable."
    )
    log(msg)
    if bool(CFG.get("cell17_2_fail_if_torch_unavailable", False)):
        raise RuntimeError(msg)

if torch_available:
    class _SinusoidalTimeEmbedding172(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = int(dim)

        def forward(self, t):
            device = t.device
            half = self.dim // 2
            freqs = torch.exp(
                -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
            )
            args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            if self.dim % 2 == 1:
                emb = torch.cat([emb, torch.zeros(len(t), 1, device=device)], dim=1)
            return emb

    class _TabDDPMDenoiser172(nn.Module):
        def __init__(self, input_dim, hidden_dim, time_dim):
            super().__init__()
            self.time_emb = _SinusoidalTimeEmbedding172(time_dim)
            self.net = nn.Sequential(
                nn.Linear(input_dim + time_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x, t):
            te = self.time_emb(t)
            return self.net(torch.cat([x, te], dim=1))

def _make_schedule_172(device):
    betas = torch.linspace(BETA_START_172, BETA_END_172, T_STEPS_172, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }

def _resolve_device_172():
    if not torch_available:
        return None
    req = str(CFG.get("cell17_2_device", "auto")).lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)

def _train_tabddpm_172(train_z: pd.DataFrame, scope_seed: int):
    device = _resolve_device_172()

    torch.manual_seed(scope_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(scope_seed)

    x_np = train_z.to_numpy(dtype=np.float32, copy=True)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    ds = TensorDataset(x_tensor)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE_172,
        shuffle=True,
        drop_last=False,
        num_workers=int(CFG.get("cell17_2_num_workers", 0)),
    )

    model = _TabDDPMDenoiser172(
        input_dim=train_z.shape[1],
        hidden_dim=HIDDEN_DIM_172,
        time_dim=TIME_EMBED_DIM_172,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR_172, weight_decay=WEIGHT_DECAY_172)
    schedule = _make_schedule_172(device)

    losses = []

    model.train()
    for epoch in range(1, EPOCHS_172 + 1):
        batch_losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            bsz = xb.shape[0]
            t = torch.randint(0, T_STEPS_172, (bsz,), device=device).long()
            noise = torch.randn_like(xb)

            sqrt_ac = schedule["sqrt_alphas_cumprod"][t].unsqueeze(1)
            sqrt_om = schedule["sqrt_one_minus_alphas_cumprod"][t].unsqueeze(1)

            x_t = sqrt_ac * xb + sqrt_om * noise
            pred = model(x_t, t)

            loss = F.mse_loss(pred, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if GRAD_CLIP_172 and GRAD_CLIP_172 > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_172)
            opt.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
        losses.append(epoch_loss)

        if int(CFG.get("cell17_2_log_every_epochs", 4)) > 0 and (
            epoch == 1 or epoch == EPOCHS_172 or epoch % int(CFG.get("cell17_2_log_every_epochs", 4)) == 0
        ):
            log(f"[Cell17.2]   epoch {epoch}/{EPOCHS_172} | loss={epoch_loss:.6f}")

    return model, schedule, losses, str(device)

def _sample_tabddpm_172(model, schedule, n_rows: int, input_dim: int, device_str: str):
    device = torch.device(device_str)
    model.eval()

    with torch.no_grad():
        x = torch.randn(int(n_rows), int(input_dim), device=device)

        if SAMPLING_STEPS_172 >= T_STEPS_172:
            steps = list(range(T_STEPS_172 - 1, -1, -1))
        else:
            steps = np.linspace(T_STEPS_172 - 1, 0, SAMPLING_STEPS_172).round().astype(int).tolist()
            steps = sorted(set(steps), reverse=True)

        betas = schedule["betas"]
        alphas = schedule["alphas"]
        alphas_cumprod = schedule["alphas_cumprod"]

        for t_int in steps:
            t = torch.full((x.shape[0],), int(t_int), device=device, dtype=torch.long)
            eps_theta = model(x, t)

            beta_t = betas[t_int]
            alpha_t = alphas[t_int]
            alpha_bar_t = alphas_cumprod[t_int]

            coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (x - coef * eps_theta)

            if t_int > 0:
                z = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * z
            else:
                x = mean

        return x.detach().cpu().numpy().astype(np.float32)

# ----------------------------------------------------------
# 5) Validate Cell 17.0 fairness governance
# ----------------------------------------------------------
CELL17_0_VERSION_172 = _require_17_0_governance_172(CELL17_0_BASELINE_FAIRNESS_CONTRACT)

# ----------------------------------------------------------
# 6) Determine eligible TabDDPM scopes
# ----------------------------------------------------------
plan = CELL17_0_BASELINE_PLAN.get("TabDDPM", {})
allowed_scopes = list(plan.get("allowed_scopes", []))

scope_registry = CELL17_0_BASELINE_SCOPE_REGISTRY_DF.copy()
scope_registry["scope_id"] = scope_registry["scope_id"].astype(str)

eligible_scopes = []
for sid in allowed_scopes:
    d = scope_registry[scope_registry["scope_id"].eq(str(sid))]
    if len(d) == 0:
        continue
    cols = _scope_cols_172(str(sid))
    if len(cols) == 0:
        continue
    eligible_scopes.append(str(sid))

if not eligible_scopes:
    raise RuntimeError("[Cell17.2] No eligible TabDDPM scopes found from Cell 17.0.")

# ----------------------------------------------------------
# 7) Train/generate per scope
# ----------------------------------------------------------
run_rows = []
artifact_rows = []

log(
    "[Cell17.2] Running TabDDPM baseline | "
    f"torch_available={torch_available} | scopes={eligible_scopes} | "
    f"epochs={EPOCHS_172} | T={T_STEPS_172} | q4_status=blocked_no_promotion"
)

for scope_i, scope_id in enumerate(eligible_scopes, start=1):
    cols = _scope_cols_172(scope_id)
    cols = [c for c in cols if c in df_tr.columns]

    out_path = os.path.join(BASELINE_DIR, f"tabddpm_{_slug_172(scope_id)}_synthetic_test.parquet")
    meta_path = os.path.join(BASELINE_META_DIR, f"cell17_2_tabddpm_{_slug_172(scope_id)}_metadata.json")

    log(f"[Cell17.2] scope {scope_i}/{len(eligible_scopes)} | {scope_id} | cols={len(cols)}")

    row_base = {
        "baseline": "TabDDPM",
        "scope_id": scope_id,
        "torch_available": bool(torch_available),
        "cols_n": int(len(cols)),
        "train_rows_available": int(len(df_tr)),
        "train_row_cap": int(TRAIN_ROW_CAP_172),
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
            "generated_rows": 0,
            "fit_succeeded": False,
            "sample_succeeded": False,
            "device": "",
            "final_loss": np.nan,
            "error": torch_error,
        })
        artifact_rows.append({
            "baseline": "TabDDPM",
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
        train_raw = _numeric_train_frame_172(cols)
        if train_raw.empty or train_raw.shape[1] == 0:
            raise RuntimeError("empty_training_frame_after_scope_column_resolution")

        train_z, meta = _fit_preprocess_172(train_raw)
        train_fit = _deterministic_train_sample_172(
            train_z,
            row_cap=TRAIN_ROW_CAP_172,
            seed=SEED + scope_i * 2039,
        )

        model, schedule, losses, device_str = _train_tabddpm_172(
            train_z=train_fit,
            scope_seed=SEED + scope_i * 3001,
        )

        z_sample = _sample_tabddpm_172(
            model=model,
            schedule=schedule,
            n_rows=N_TE,
            input_dim=train_fit.shape[1],
            device_str=device_str,
        )

        z_df = pd.DataFrame(z_sample, columns=list(train_fit.columns))
        sample_pp = _inverse_postprocess_172(
            z_df=z_df,
            meta=meta,
            ordered_cols=list(train_fit.columns),
        )

        if len(sample_pp) != N_TE:
            raise RuntimeError(f"generated_row_mismatch: got={len(sample_pp)} expected={N_TE}")

        sample_pp.to_parquet(out_path, index=False)

        meta_payload = {
            "cell": "17.2",
            "version": CELL172_VERSION,
            "baseline": "TabDDPM",
            "scope_id": scope_id,
            "columns": list(train_fit.columns),
            "train_rows_used": int(len(train_fit)),
            "generated_rows": int(len(sample_pp)),
            "device": device_str,
            "losses": losses,
            "final_loss": float(losses[-1]) if losses else np.nan,
            "config": {
                "epochs": int(EPOCHS_172),
                "batch_size": int(BATCH_SIZE_172),
                "hidden_dim": int(HIDDEN_DIM_172),
                "time_embed_dim": int(TIME_EMBED_DIM_172),
                "lr": float(LR_172),
                "weight_decay": float(WEIGHT_DECAY_172),
                "diffusion_steps": int(T_STEPS_172),
                "sampling_steps": int(SAMPLING_STEPS_172),
                "beta_start": float(BETA_START_172),
                "beta_end": float(BETA_END_172),
            },
            "column_metadata": meta,
            "fairness_scope": scope_registry.loc[scope_registry["scope_id"].eq(scope_id)].to_dict("records"),
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
                "external_baseline_values_generated": True,
                "selection_done_here": False,
            },
        }
        _write_json_172(meta_path, meta_payload)

        run_rows.append({
            **row_base,
            "status": "success",
            "reason": "tabddpm_fit_sample_completed",
            "train_rows_used": int(len(train_fit)),
            "generated_rows": int(len(sample_pp)),
            "fit_succeeded": True,
            "sample_succeeded": True,
            "device": device_str,
            "final_loss": float(losses[-1]) if losses else np.nan,
            "error": "",
        })

        artifact_rows.append({
            "baseline": "TabDDPM",
            "scope_id": scope_id,
            "path": out_path,
            "exists": True,
            "rows": int(len(sample_pp)),
            "cols": int(sample_pp.shape[1]),
            "sha256": _sha256_file_172(out_path),
            "metadata_path": meta_path,
            "metadata_sha256": _sha256_file_172(meta_path),
            "status": "success",
            "reason": "tabddpm_fit_sample_completed",
        })

        del train_raw, train_z, train_fit, model, schedule, z_sample, z_df, sample_pp
        if torch_available and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        log(f"[Cell17.2] WARNING: TabDDPM scope failed | scope={scope_id} | error={err}")

        device_str = ""
        try:
            device_str = str(_resolve_device_172()) if torch_available else ""
        except Exception:
            device_str = ""

        run_rows.append({
            **row_base,
            "status": "failed",
            "reason": "tabddpm_scope_exception",
            "train_rows_used": 0,
            "generated_rows": 0,
            "fit_succeeded": False,
            "sample_succeeded": False,
            "device": device_str,
            "final_loss": np.nan,
            "error": err,
        })

        artifact_rows.append({
            "baseline": "TabDDPM",
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

        if bool(CFG.get("cell17_2_fail_on_scope_failure", False)):
            raise

run_audit_df = pd.DataFrame(run_rows)
artifact_registry_df = pd.DataFrame(artifact_rows)

# ----------------------------------------------------------
# 8) Save registries
# ----------------------------------------------------------
run_audit_df.to_csv(run_audit_csv, index=False)
artifact_registry_df.to_csv(artifact_registry_csv, index=False)

success_n = int((run_audit_df["status"].astype(str) == "success").sum()) if len(run_audit_df) else 0
failed_n = int((run_audit_df["status"].astype(str) == "failed").sum()) if len(run_audit_df) else 0
not_run_n = int((run_audit_df["status"].astype(str) == "not_run").sum()) if len(run_audit_df) else 0

# ----------------------------------------------------------
# 9) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "17.2",
    "version": CELL172_VERSION,
    "role": "tabddpm_external_baseline_no_q4_promotion",
    "baseline": "TabDDPM",
    "torch_available": bool(torch_available),
    "eligible_scopes": eligible_scopes,
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "baseline_not_compared_to_Q4_coupled_artifact": True,
    },
    "q6_public_context": CELL17_0_BASELINE_FAIRNESS_CONTRACT.get("q6_public_context", {}),
    "upstream_contract_versions": {
        "cell17_0": CELL17_0_VERSION_172,
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
        "do_not_claim_q4_or_q6_capability": True,
        "do_not_claim_public_release_privacy_pass": True,
    },
    "tabddpm_config": {
        "train_row_cap": int(TRAIN_ROW_CAP_172),
        "epochs": int(EPOCHS_172),
        "batch_size": int(BATCH_SIZE_172),
        "hidden_dim": int(HIDDEN_DIM_172),
        "time_embed_dim": int(TIME_EMBED_DIM_172),
        "lr": float(LR_172),
        "weight_decay": float(WEIGHT_DECAY_172),
        "diffusion_steps": int(T_STEPS_172),
        "sampling_steps": int(SAMPLING_STEPS_172),
        "beta_start": float(BETA_START_172),
        "beta_end": float(BETA_END_172),
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TRAIN_real_values_used_for_fitting": True,
        "VAL_real_values_used_for_selection": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "pipeline_synthetic_values_mutated": False,
        "external_baseline_values_generated": bool(success_n > 0),
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

_write_json_172(contract_json, contract)
_write_json_172(contract_canonical_json, contract)

manifest = {
    "cell": "17.2",
    "version": CELL172_VERSION,
    "baseline": "TabDDPM",
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "artifact_registry": artifact_registry_df.to_dict("records"),
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
    "fairness_contract": contract["fairness_contract"],
}
_write_json_172(manifest_json, manifest)

hashes = {
    "run_audit_csv_sha256": _sha256_file_172(run_audit_csv),
    "artifact_registry_csv_sha256": _sha256_file_172(artifact_registry_csv),
    "contract_json_sha256": _sha256_file_172(contract_json),
    "contract_canonical_json_sha256": _sha256_file_172(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_172(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_172(contract_json, contract)
_write_json_172(contract_canonical_json, contract)
_write_json_172(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals
# ----------------------------------------------------------
globals()["CELL172_VERSION"] = CELL172_VERSION
globals()["CELL17_2_TABDDPM_RUN_AUDIT_DF"] = run_audit_df
globals()["CELL17_2_TABDDPM_ARTIFACT_REGISTRY_DF"] = artifact_registry_df
globals()["CELL17_2_TABDDPM_BASELINE_CONTRACT"] = contract

globals()["CELL17_2_TABDDPM_RUN_AUDIT_CSV"] = run_audit_csv
globals()["CELL17_2_TABDDPM_ARTIFACT_REGISTRY_CSV"] = artifact_registry_csv
globals()["CELL17_2_TABDDPM_CONTRACT_JSON"] = contract_json
globals()["CELL17_2_TABDDPM_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL17_2_TABDDPM_MANIFEST_JSON"] = manifest_json

log(
    "[Cell17.2] TabDDPM baseline complete | "
    f"torch_available={torch_available} | "
    f"success={success_n} | failed={failed_n} | not_run={not_run_n} | "
    "q4_status=blocked_no_promotion"
)
log(f"[Cell17.2] Run audit saved: {run_audit_csv}")
log(f"[Cell17.2] Artifact registry saved: {artifact_registry_csv}")
log(f"[Cell17.2] Contract saved: {contract_json}")
log(f"[Cell17.2] Canonical contract saved: {contract_canonical_json}")
log(
    "[Cell17.2] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TRAIN_real_values_used_for_fitting=True | "
    "VAL_real_values_used_for_selection=False | "
    "TEST_real_values_used=False | "
    "TEST_real_values_used_for_materialization=False | "
    "pipeline_synthetic_values_mutated=False | "
    "selection_done_here=False | "
    f"generator_fit_done_here={bool(torch_available and success_n > 0)} | "
    f"materialization_done_here={bool(success_n > 0)} | "
    "external_baseline_only=True"
)
log("--- END: Cell 17.2 - TabDDPM external baseline (v1.1 no-Q4-promotion strict) ---")

gc.collect()