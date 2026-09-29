# ==========================================================
# CORE DDPM HELPER — Exponential Moving Average (EMA)
# Decision: UPDATE + KEEP
#
# Purpose:
# - Provide a robust EMA helper for DDPM candidate training/sampling.
# - Compatible with DDPM wrapper code that expects EMA.update(),
#   EMA.store(), EMA.copy_to(), and EMA.restore().
#
# Claim-scope rule:
# - This helper does not read TRAIN, VAL, or TEST data.
# - It does not fit a generator by itself.
# - It does not select candidates.
# - It does not perform TEST QA.
# - It does not repair, overwrite, or release artifacts.
# - It may support DDPM/SeqDenoiser candidate generators, but any
#   final claim must come from downstream VAL-selected and TEST-audited
#   branch ledgers.
#
# Placement:
# - Keep in the DDPM / sequence-model helper section.
# - Do not place between Cell 1 and the real Cell 2.
# ==========================================================

import torch
import numpy as np


class EMA:
    """
    Exponential Moving Average over model.state_dict().

    Floating tensors:
        shadow = decay * shadow + (1 - decay) * current_model_value

    Non-floating tensors:
        shadow = current_model_value

    Rationale
    ---------
    Integer and boolean buffers should not be averaged. They represent
    indices, counters, masks, or bookkeeping states and must be copied exactly.

    Scientific contract
    -------------------
    EMA is a training helper only. It performs no data access, no split
    selection, no candidate promotion, no TEST-informed operation, and no
    artifact mutation outside the model object passed by the caller.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"[EMA] model must be torch.nn.Module, got {type(model)}")

        decay = float(decay)
        if not np.isfinite(decay) or not (0.0 <= decay < 1.0):
            raise ValueError(f"[EMA] decay must be finite and in [0, 1). Got {decay}")

        self.decay = decay
        self.shadow = {
            k: v.detach().clone().contiguous()
            for k, v in model.state_dict().items()
        }
        self._backup = None

    def _check_keys(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"[EMA] model must be torch.nn.Module, got {type(model)}")

        msd = model.state_dict()

        if set(msd.keys()) != set(self.shadow.keys()):
            missing_in_shadow = sorted(set(msd.keys()) - set(self.shadow.keys()))
            missing_in_model = sorted(set(self.shadow.keys()) - set(msd.keys()))
            raise RuntimeError(
                "[EMA] Model state_dict keys changed since EMA initialization. "
                f"missing_in_shadow={missing_in_shadow[:20]} | "
                f"missing_in_model={missing_in_model[:20]}"
            )

        for k in self.shadow.keys():
            if tuple(msd[k].shape) != tuple(self.shadow[k].shape):
                raise RuntimeError(
                    f"[EMA] Shape mismatch for key={k}: "
                    f"model={tuple(msd[k].shape)} shadow={tuple(self.shadow[k].shape)}"
                )

        return msd

    @staticmethod
    def _align_like(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        out = src
        if out.device != tgt.device:
            out = out.to(tgt.device)
        if out.dtype != tgt.dtype:
            out = out.to(dtype=tgt.dtype)
        if not out.is_contiguous():
            out = out.contiguous()
        return out

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        Update EMA shadow weights from the current model state.
        """
        msd = self._check_keys(model)

        for k in list(self.shadow.keys()):
            v_model = msd[k].detach()

            if tuple(self.shadow[k].shape) != tuple(v_model.shape):
                raise RuntimeError(
                    f"[EMA.update] Shape mismatch for key={k}: "
                    f"shadow={tuple(self.shadow[k].shape)} model={tuple(v_model.shape)}"
                )

            self.shadow[k] = self._align_like(self.shadow[k], v_model)

            if torch.is_floating_point(v_model):
                self.shadow[k].mul_(self.decay).add_(v_model, alpha=(1.0 - self.decay))
            else:
                self.shadow[k].copy_(v_model)

    @torch.no_grad()
    def store(self, model: torch.nn.Module):
        """
        Save current model state before applying EMA weights.
        """
        msd = self._check_keys(model)
        self._backup = {
            k: v.detach().clone().contiguous()
            for k, v in msd.items()
        }

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        """
        Restore model state saved by store().
        Device and dtype are aligned to the current model state before loading.
        """
        if self._backup is None:
            return

        msd = self._check_keys(model)

        if set(msd.keys()) != set(self._backup.keys()):
            missing_in_backup = sorted(set(msd.keys()) - set(self._backup.keys()))
            missing_in_model = sorted(set(self._backup.keys()) - set(msd.keys()))
            self._backup = None
            raise RuntimeError(
                "[EMA.restore] Backup keys differ from current model keys. "
                f"missing_in_backup={missing_in_backup[:20]} | "
                f"missing_in_model={missing_in_model[:20]}"
            )

        load_sd = {}
        for k, v in self._backup.items():
            tgt = msd[k]

            if tuple(v.shape) != tuple(tgt.shape):
                self._backup = None
                raise RuntimeError(
                    f"[EMA.restore] Shape mismatch for key={k}: "
                    f"backup={tuple(v.shape)} model={tuple(tgt.shape)}"
                )

            load_sd[k] = self._align_like(v, tgt)

        model.load_state_dict(load_sd, strict=True)
        self._backup = None

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        """
        Copy EMA shadow state into the model, moving tensors to the target
        device and dtype if needed.
        """
        msd = self._check_keys(model)

        load_sd = {}
        for k, v in self.shadow.items():
            tgt = msd[k]

            if tuple(v.shape) != tuple(tgt.shape):
                raise RuntimeError(
                    f"[EMA.copy_to] Shape mismatch for key={k}: "
                    f"shadow={tuple(v.shape)} model={tuple(tgt.shape)}"
                )

            load_sd[k] = self._align_like(v, tgt)

        model.load_state_dict(load_sd, strict=True)


# ==========================================================
# MANIFEST — EMA helper contract
# ==========================================================
EMA_HELPER_CONTRACT = {
    "component": "EMA",
    "component_type": "core_ddpm_training_helper",
    "generator_family": "none_helper_only",
    "supports_generator_families": [
        "DDPM_candidate_generator",
        "SeqDenoiser_candidate_generator",
    ],
    "parameters": {
        "decay_default": 0.999,
        "floating_tensor_update": "shadow = decay * shadow + (1 - decay) * model",
        "nonfloating_tensor_update": "copy_current_model_value_exactly",
        "state_dict_key_check": True,
        "state_dict_shape_check": True,
        "device_dtype_alignment": True,
    },
    "seed": "none_in_this_helper",
    "train_fit_scope": "none_in_this_helper",
    "val_selection_metric": "none_in_this_helper",
    "test_only_qa": "none_in_this_helper",
    "contributes_to_final_artifact": False,
    "final_artifact_contribution_rule": (
        "EMA may improve/stabilize candidate model training, but final artifact "
        "contribution is determined only by downstream branch selection, VAL scoring, "
        "materialization, enforcement, and TEST-only QA ledgers."
    ),
    "leakage_status": {
        "reads_train_values": False,
        "reads_val_values": False,
        "reads_test_values": False,
        "uses_test_for_selection": False,
        "uses_test_for_repair": False,
        "mutates_synthetic_artifact": False,
        "overwrites_final_artifact": False,
    },
    "paper_claim_status": (
        "Not citable as quality evidence. Cite only downstream selected branch results "
        "whose TRAIN/VAL/TEST contracts and final-artifact contribution are recorded separately."
    ),
}

if "RUN_META" in globals():
    RUN_META.setdefault("helper_contracts", {})
    RUN_META["helper_contracts"]["EMA"] = EMA_HELPER_CONTRACT

# Lightweight smoke test: no data access.
_ema_smoke_model = torch.nn.Sequential(
    torch.nn.Linear(3, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 2),
)
_ema_smoke = EMA(_ema_smoke_model, decay=0.999)
_ema_smoke.update(_ema_smoke_model)
_ema_smoke.store(_ema_smoke_model)
_ema_smoke.copy_to(_ema_smoke_model)
_ema_smoke.restore(_ema_smoke_model)

del _ema_smoke, _ema_smoke_model

log("[DDPM helper] EMA loaded, smoke-tested, and registered as non-claim-supporting infrastructure.")