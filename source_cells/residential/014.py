# ============================================================
# CORE DDPM HELPER — Cosine beta schedule
# Decision: UPDATE + KEEP
#
# Purpose:
# - Provide the canonical DDPM cosine beta schedule used by candidate
#   diffusion generators in the protocol/continuous branches.
#
# Claim-scope rule:
# - This helper defines a noise schedule only.
# - It does not fit a model.
# - It does not read TRAIN/VAL/TEST data.
# - It does not select candidates.
# - It does not mutate artifacts.
# - It cannot by itself support any paper claim.
#
# Placement:
# - Keep in the protocol/DDPM helper section.
# - Do not place between Cell 1 and the real Cell 2.
#
# Based on:
# - Nichol and Dhariwal (2021), Improved Denoising Diffusion
#   Probabilistic Models.
# ============================================================

import math
import torch


def cosine_beta_schedule(
    T: int,
    s: float = 0.008,
    device=None,
    dtype: torch.dtype = torch.float32,
    beta_min: float = 1e-8,
    beta_max: float = 0.999,
) -> torch.Tensor:
    """
    Construct a numerically stable cosine beta schedule for DDPM.

    Parameters
    ----------
    T:
        Number of diffusion steps. Must be >= 10.
    s:
        Cosine schedule offset from Nichol & Dhariwal. Must satisfy 0 <= s < 1.
    device:
        Optional torch device for the returned tensor.
    dtype:
        Output dtype. Must be torch.float32 or torch.float64.
    beta_min:
        Minimum beta clamp. Must satisfy 0 < beta_min < beta_max.
    beta_max:
        Maximum beta clamp. Must satisfy beta_min < beta_max < 1.

    Returns
    -------
    torch.Tensor
        Tensor of shape (T,), with beta values strictly inside (0, 1).

    Scientific contract
    -------------------
    This function is deterministic for fixed inputs and torch version. It is
    a schedule helper only; it performs no model fitting, no data access, no
    candidate selection, and no TEST-informed operation.
    """
    T = int(T)

    if T < 10:
        raise ValueError(f"DDPM diffusion steps T must be >= 10. Got T={T}.")

    s = float(s)
    if not (0.0 <= s < 1.0):
        raise ValueError(f"s must be in [0, 1). Got s={s}.")

    beta_min = float(beta_min)
    beta_max = float(beta_max)

    if not (0.0 < beta_min < beta_max < 1.0):
        raise ValueError(
            "Require 0 < beta_min < beta_max < 1. "
            f"Got beta_min={beta_min}, beta_max={beta_max}."
        )

    if dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            "dtype must be torch.float32 or torch.float64 for stable DDPM schedules. "
            f"Got dtype={dtype}."
        )

    # Work internally in float64 for stable schedule construction, then cast.
    work_dtype = torch.float64
    x = torch.linspace(0, T, T + 1, dtype=work_dtype, device=device)

    alphas_cumprod = torch.cos(
        ((x / float(T)) + s) / (1.0 + s) * math.pi * 0.5
    ).pow(2)

    first = torch.clamp(alphas_cumprod[0], min=1e-20)
    alphas_cumprod = alphas_cumprod / first

    denom = torch.clamp(alphas_cumprod[:-1], min=1e-20)
    betas = 1.0 - (alphas_cumprod[1:] / denom)
    betas = torch.clamp(betas, min=beta_min, max=beta_max)

    if tuple(betas.shape) != (T,):
        raise RuntimeError(
            f"Beta schedule shape mismatch: got {tuple(betas.shape)}, expected {(T,)}."
        )

    if not torch.isfinite(betas).all().item():
        raise RuntimeError("Beta schedule contains non-finite values.")

    beta_lo = float(betas.min().detach().cpu())
    beta_hi = float(betas.max().detach().cpu())

    if beta_lo <= 0.0 or beta_hi >= 1.0:
        raise RuntimeError(
            f"Beta schedule outside (0, 1): min={beta_lo}, max={beta_hi}."
        )

    return betas.to(dtype=dtype)


# Lightweight contract smoke test. This is deterministic and reads no data.
_beta_smoke = cosine_beta_schedule(
    T=int(CFG["ddpm_timesteps"]) if "CFG" in globals() else 200,
    device="cpu",
    dtype=torch.float32,
)

assert _beta_smoke.ndim == 1
assert int(_beta_smoke.shape[0]) == (int(CFG["ddpm_timesteps"]) if "CFG" in globals() else 200)
assert torch.isfinite(_beta_smoke).all().item()
assert float(_beta_smoke.min()) > 0.0
assert float(_beta_smoke.max()) < 1.0

del _beta_smoke

log("[DDPM helper] cosine_beta_schedule loaded and smoke-tested; no data accessed.")