# ==========================================================
# CORE DDPM WRAPPER — Candidate generator only
# Decision: UPDATE + KEEP
# Version: DDPM wrapper v5.3-THESIS
#
# Purpose:
# - Provide a SeqDenoiser-compatible DDPM training wrapper.
# - Provide a batched stride-aware sampler for TEST-length candidate
#   materialization.
#
# Claim-scope rule:
# - DDPM is a conditional refinement / candidate generator only.
# - DDPM is NOT the universal protocol generator.
# - DDPM does NOT select final candidates.
# - DDPM does NOT perform publication acceptance.
# - DDPM does NOT make TEST-informed repair decisions.
# - Final contribution to the scientific artifact is determined only by
#   downstream VAL-only selection, materialization, enforcement, and
#   TEST-only QA ledgers.
#
# Placement:
# - Keep in the DDPM / sequence-model candidate-generator section.
# - Do not place between Cell 1 and the real Cell 2.
# ==========================================================
#
# Key fixes vs v5.1:
# - EMA store/copy/restore is exception-safe.
# - D is validated before zero-length generation return.
# - training hyperparameters weight_decay and grad_clip are fail-closed.
# - sampler audit version updated to v5.2.
# - generation audit explicitly declares DDPM candidate-only role.
# - stricter finite checks on diffusion schedule buffers.
# - stricter temperature/sequence_batch/log_every validation.
# - clear handling of previous net train/eval mode after generation.
#
# Preserved behavior:
# - validates EMA interface, not only existence.
# - validates reg_seq before int64 casting in flat generation.
# - strict binary x_obs validation inside training batches.
# - validates regime IDs against K_reg in batch and flat paths.
# - fails on zero-observed training batches.
# - explicit t-index validation in q_sample / reverse step.
# - correct denoiser call accounting: total_batches * T.
# - preserves registered buffers via copy_ only.
# - generation returns scaled DDPM-space values shaped (N, D).
# -----------------------------

from typing import Optional, Tuple
import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------
if "EMA" not in globals():
    raise RuntimeError("[DDPM wrapper] Missing EMA class/function in notebook scope.")

if "cosine_beta_schedule" not in globals():
    raise RuntimeError("[DDPM wrapper] Missing cosine_beta_schedule(...) in notebook scope.")


class DDPM(nn.Module):
    """
    DDPM wrapper for sequence denoising models.

    Expected net signature:
        net(x_t, t, cond_seq, reg_seq, tod_seq) -> predicted_noise

    Required net attributes:
        net.D
        net.cond_dim
        net.tod_dim
        net.K_reg

    Important:
        This wrapper only trains/samples a DDPM candidate generator.
        It does not perform portfolio selection, VAL gating, final A2 construction,
        or TEST evaluation.
    """
    WRAPPER_VERSION = "DDPM_wrapper_v5_3_THESIS"
    SAMPLER_VERSION = "DDPM.generate_batched_stride_aware_v5_3_THESIS"
    GENERATOR_FAMILY = "DDPM"
    ROLE = "conditional_refinement_candidate_generator"
    ARCHITECTURE_POLICY = "hybrid_portfolio_candidate_only"
    FINAL_SELECTION_POLICY = "downstream_VAL_only_selector"
    TEST_POLICY = "TEST_audit_only_no_selection_no_repair"
    CONTRIBUTES_TO_FINAL_ARTIFACT_BY_ITSELF = False

    def __init__(
        self,
        net: nn.Module,
        T: int,
        device: str,
        ema_decay: float = 0.999,
    ):
        super().__init__()

        if not isinstance(net, nn.Module):
            raise TypeError("[DDPM.__init__] net must be a torch.nn.Module.")

        if not hasattr(net, "D"):
            raise AttributeError("[DDPM.__init__] DDPM expects net.D to exist.")

        self.T = int(T)
        if self.T < 10:
            raise RuntimeError(f"[DDPM.__init__] T must be >= 10, got T={T}")

        ema_decay = float(ema_decay)
        if not (0.0 < ema_decay < 1.0):
            raise RuntimeError(
                f"[DDPM.__init__] ema_decay must be in (0,1). Got {ema_decay}"
            )

        requested_device = str(device)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            self._log_msg(
                f"[DDPM.__init__] Requested device={requested_device}, but CUDA is unavailable. "
                "Falling back to CPU."
            )
            requested_device = "cpu"

        self.device_obj = torch.device(requested_device)
        self.device = str(self.device_obj)

        self.net = net.to(self.device_obj)
        self.D = int(net.D)

        self.cond_dim = int(getattr(net, "cond_dim", -1))
        self.tod_dim = int(getattr(net, "tod_dim", 2))
        self.K_reg = int(getattr(net, "K_reg", -1))

        if self.D <= 0:
            raise RuntimeError(f"[DDPM.__init__] Invalid net.D={self.D}")

        if self.tod_dim <= 0:
            raise RuntimeError(f"[DDPM.__init__] Invalid net.tod_dim={self.tod_dim}")

        if self.cond_dim == 0:
            raise RuntimeError("[DDPM.__init__] net.cond_dim must not be zero.")

        if self.K_reg == 0:
            raise RuntimeError("[DDPM.__init__] net.K_reg must not be zero.")

        # --- diffusion schedule ---
        betas = cosine_beta_schedule(
            self.T,
            device=self.device_obj,
            dtype=torch.float32,
        )
        betas = torch.clamp(betas.float(), min=1e-8, max=0.999)

        if betas.shape != (self.T,):
            raise RuntimeError(
                f"[DDPM.__init__] betas shape mismatch: got={tuple(betas.shape)} expected={(self.T,)}"
            )

        if not torch.isfinite(betas).all().item():
            raise RuntimeError("[DDPM.__init__] betas contains non-finite values.")

        if torch.any(betas <= 0).item() or torch.any(betas >= 1).item():
            raise RuntimeError(
                f"[DDPM.__init__] betas outside valid range: "
                f"min={float(betas.min())}, max={float(betas.max())}"
            )

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        if not torch.isfinite(alphas).all().item():
            raise RuntimeError("[DDPM.__init__] alphas contains non-finite values.")

        if not torch.isfinite(alpha_bars).all().item():
            raise RuntimeError("[DDPM.__init__] alpha_bars contains non-finite values.")

        if torch.any(alphas <= 0).item() or torch.any(alphas >= 1).item():
            raise RuntimeError(
                f"[DDPM.__init__] alphas outside valid range: "
                f"min={float(alphas.min())}, max={float(alphas.max())}"
            )

        if torch.any(alpha_bars <= 0).item() or torch.any(alpha_bars > 1).item():
            raise RuntimeError(
                f"[DDPM.__init__] alpha_bars outside valid range: "
                f"min={float(alpha_bars.min())}, max={float(alpha_bars.max())}"
            )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        alpha_bars_prev = torch.cat(
            [
                torch.ones(1, device=self.device_obj, dtype=alpha_bars.dtype),
                alpha_bars[:-1],
            ],
            dim=0,
        )

        if alpha_bars_prev.shape != alpha_bars.shape:
            raise RuntimeError(
                f"[DDPM.__init__] alpha_bars_prev shape mismatch: "
                f"{tuple(alpha_bars_prev.shape)} vs {tuple(alpha_bars.shape)}"
            )

        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

        denom = torch.clamp(1.0 - alpha_bars, min=1e-12)
        beta_tilde = betas * (1.0 - alpha_bars_prev) / denom
        beta_tilde = torch.clamp(beta_tilde, min=1e-20)

        if not torch.isfinite(beta_tilde).all().item():
            raise RuntimeError("[DDPM.__init__] beta_tilde contains non-finite values.")

        self.register_buffer("beta_tilde", beta_tilde)

        self.register_buffer(
            "feat_w",
            torch.ones(1, 1, self.D, device=self.device_obj, dtype=torch.float32),
        )

        self.ema = EMA(self.net, decay=ema_decay)
        self._validate_ema_interface()

        self.last_generate_audit = None
        self.last_train_audit = None

    # -----------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------
    @staticmethod
    def _log_msg(msg: str):
        if "log" in globals() and callable(globals()["log"]):
            globals()["log"](msg)
        else:
            print(msg)

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------
    def _validate_ema_interface(self):
        required = ["update", "store", "copy_to", "restore"]
        missing = [
            m
            for m in required
            if not hasattr(self.ema, m) or not callable(getattr(self.ema, m))
        ]
        if missing:
            raise RuntimeError(f"[DDPM.__init__] EMA object missing required methods: {missing}")

    def _expected_tod_dim(self) -> int:
        return int(getattr(self.net, "tod_dim", self.tod_dim))

    def _expected_cond_dim(self) -> Optional[int]:
        if hasattr(self.net, "cond_dim"):
            return int(self.net.cond_dim)
        return None

    def _expected_K_reg(self) -> Optional[int]:
        if hasattr(self.net, "K_reg"):
            k = int(self.net.K_reg)
            return k if k > 0 else None
        return self.K_reg if self.K_reg > 0 else None

    @staticmethod
    def _validate_reg_tensor(
        reg_seq: torch.Tensor,
        *,
        context: str,
        K_reg: Optional[int],
    ):
        if reg_seq.dtype not in {torch.int64, torch.long, torch.int32, torch.int16, torch.int8}:
            raise RuntimeError(f"[{context}] reg_seq must be integer dtype, got {reg_seq.dtype}")

        if reg_seq.numel() == 0:
            raise RuntimeError(f"[{context}] reg_seq is empty.")

        min_reg = int(reg_seq.min().detach().cpu().item())
        max_reg = int(reg_seq.max().detach().cpu().item())

        if min_reg < 0:
            raise RuntimeError(f"[{context}] reg_seq contains negative ids: min={min_reg}")

        if K_reg is not None and max_reg >= int(K_reg):
            raise RuntimeError(f"[{context}] reg_seq max id {max_reg} exceeds K_reg={K_reg}")

    @staticmethod
    def _validate_binary_tensor(x_obs: torch.Tensor, *, context: str):
        if x_obs.numel() == 0:
            raise RuntimeError(f"[{context}] x_obs is empty.")

        if not torch.isfinite(x_obs).all().item():
            raise RuntimeError(f"[{context}] x_obs contains non-finite values.")

        rounded = torch.round(x_obs)
        max_abs = (
            torch.max(torch.abs(x_obs - rounded)).detach().cpu().item()
            if x_obs.numel()
            else 0.0
        )

        if max_abs > 1e-6:
            raise RuntimeError(
                f"[{context}] x_obs must be binary 0/1-like before thresholding. "
                f"max_abs_err={max_abs}"
            )

        vals = torch.unique(rounded.detach())
        bad = vals[(vals != 0) & (vals != 1)]
        if bad.numel() > 0:
            raise RuntimeError(
                f"[{context}] x_obs contains non-binary values: {bad.detach().cpu().tolist()}"
            )

    def _validate_batch_contract(
        self,
        x: torch.Tensor,
        x_obs: Optional[torch.Tensor],
        cond_seq: torch.Tensor,
        reg_seq: torch.Tensor,
        tod_seq: torch.Tensor,
        context: str,
    ):
        if x.ndim != 3:
            raise RuntimeError(f"[{context}] x must be 3D (B,L,D), got {tuple(x.shape)}")

        if x.shape[-1] != self.D:
            raise RuntimeError(
                f"[{context}] x feature dim mismatch: got {x.shape[-1]} expected {self.D}"
            )

        if x.shape[0] <= 0 or x.shape[1] <= 0:
            raise RuntimeError(f"[{context}] x has invalid batch/time shape: {tuple(x.shape)}")

        if not torch.isfinite(x).all().item():
            raise RuntimeError(f"[{context}] x contains non-finite values.")

        if x_obs is not None:
            if x_obs.ndim != 3:
                raise RuntimeError(f"[{context}] x_obs must be 3D (B,L,D), got {tuple(x_obs.shape)}")
            if x_obs.shape != x.shape:
                raise RuntimeError(
                    f"[{context}] x_obs shape mismatch: x_obs={tuple(x_obs.shape)}, x={tuple(x.shape)}"
                )
            self._validate_binary_tensor(x_obs, context=context)

        if cond_seq.ndim != 3:
            raise RuntimeError(
                f"[{context}] cond_seq must be 3D (B,L,C), got {tuple(cond_seq.shape)}"
            )

        if cond_seq.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[{context}] cond_seq time shape mismatch: "
                f"cond_seq={tuple(cond_seq.shape)}, x={tuple(x.shape)}"
            )

        expected_cond = self._expected_cond_dim()
        if expected_cond is not None and expected_cond > 0 and cond_seq.shape[-1] != expected_cond:
            raise RuntimeError(
                f"[{context}] cond_seq dim mismatch: got {cond_seq.shape[-1]} expected {expected_cond}"
            )

        if reg_seq.ndim != 2:
            raise RuntimeError(f"[{context}] reg_seq must be 2D (B,L), got {tuple(reg_seq.shape)}")

        if reg_seq.shape != x.shape[:2]:
            raise RuntimeError(
                f"[{context}] reg_seq shape mismatch: reg_seq={tuple(reg_seq.shape)}, x={tuple(x.shape)}"
            )

        self._validate_reg_tensor(reg_seq, context=context, K_reg=self._expected_K_reg())

        expected_tod = self._expected_tod_dim()
        if tod_seq.ndim != 3 or tod_seq.shape[-1] != expected_tod:
            raise RuntimeError(
                f"[{context}] tod_seq must be 3D (B,L,{expected_tod}), got {tuple(tod_seq.shape)}"
            )

        if tod_seq.shape[:2] != x.shape[:2]:
            raise RuntimeError(
                f"[{context}] tod_seq time shape mismatch: tod_seq={tuple(tod_seq.shape)}, "
                f"x={tuple(x.shape)}"
            )

        for name, tensor in [
            ("cond_seq", cond_seq),
            ("tod_seq", tod_seq),
        ]:
            if not torch.isfinite(tensor).all().item():
                raise RuntimeError(f"[{context}] {name} contains non-finite values.")

    def _validate_flat_generation_inputs(
        self,
        N: int,
        D: int,
        cond_seq: np.ndarray,
        reg_seq: np.ndarray,
        tod_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = int(N)
        D = int(D)

        if N < 0:
            raise RuntimeError(f"[DDPM.generate] N must be >= 0, got {N}")

        if D <= 0:
            raise RuntimeError(f"[DDPM.generate] D must be positive, got {D}")

        if D != self.D:
            raise RuntimeError(f"[DDPM.generate] Requested D={D}, but net.D={self.D}")

        cond_seq = np.asarray(cond_seq, dtype=np.float32, order="C")

        reg_raw = np.asarray(reg_seq)
        if reg_raw.ndim != 1:
            reg_raw = reg_raw.reshape(-1)

        try:
            reg_float = reg_raw.astype(np.float64, copy=False)
        except Exception as e:
            raise RuntimeError("[DDPM.generate] reg_seq must be numeric/integer-like.") from e

        if not np.isfinite(reg_float).all():
            bad = int((~np.isfinite(reg_float)).sum())
            raise RuntimeError(f"[DDPM.generate] reg_seq contains non-finite values: bad={bad}")

        rounded = np.rint(reg_float)
        max_abs_err = float(np.max(np.abs(reg_float - rounded))) if rounded.size else 0.0
        if max_abs_err > 1e-6:
            bad_idx = int(np.where(np.abs(reg_float - rounded) > 1e-6)[0][0])
            raise RuntimeError(
                f"[DDPM.generate] reg_seq must be integer-like. "
                f"first_bad_idx={bad_idx}, value={reg_float[bad_idx]}, max_abs_err={max_abs_err}"
            )

        reg_seq = rounded.astype(np.int64, copy=False)

        tod_seq = np.asarray(tod_seq, dtype=np.float32, order="C")
        if tod_seq.ndim == 1:
            tod_seq = tod_seq.reshape(-1, 1)

        if cond_seq.ndim != 2:
            raise RuntimeError(f"[DDPM.generate] cond_seq must be 2D (N,C), got {cond_seq.shape}")

        expected_cond = self._expected_cond_dim()
        if expected_cond is not None and expected_cond > 0 and cond_seq.shape[1] != expected_cond:
            raise RuntimeError(
                f"[DDPM.generate] cond_seq dim mismatch: got {cond_seq.shape[1]} expected {expected_cond}"
            )

        expected_tod = self._expected_tod_dim()
        if tod_seq.ndim != 2 or tod_seq.shape[1] != expected_tod:
            raise RuntimeError(
                f"[DDPM.generate] tod_seq must be (N,{expected_tod}), got {tod_seq.shape}"
            )

        if cond_seq.shape[0] != N or reg_seq.shape[0] != N or tod_seq.shape[0] != N:
            raise RuntimeError(
                f"[DDPM.generate] Length mismatch: N={N}, cond={cond_seq.shape}, "
                f"reg={reg_seq.shape}, tod={tod_seq.shape}"
            )

        if not np.isfinite(cond_seq).all():
            raise RuntimeError("[DDPM.generate] cond_seq contains non-finite values.")

        if not np.isfinite(tod_seq).all():
            raise RuntimeError("[DDPM.generate] tod_seq contains non-finite values.")

        if reg_seq.size > 0:
            if int(np.min(reg_seq)) < 0:
                raise RuntimeError(
                    f"[DDPM.generate] reg_seq contains negative ids: min={int(np.min(reg_seq))}"
                )

            K = self._expected_K_reg()
            if K is not None and int(np.max(reg_seq)) >= K:
                raise RuntimeError(
                    f"[DDPM.generate] reg_seq max id {int(np.max(reg_seq))} exceeds K_reg={K}"
                )

        return cond_seq, reg_seq, tod_seq

    @staticmethod
    def _make_torch_generator(device_obj: torch.device, seed: int) -> torch.Generator:
        seed = int(np.uint32(int(seed)).item())
        try:
            g = torch.Generator(device=device_obj)
        except Exception:
            g = torch.Generator()
        g.manual_seed(seed)
        return g

    @staticmethod
    def _overlap_weights(L: int, mode: str) -> np.ndarray:
        mode = str(mode).lower()
        L = int(L)

        if L <= 0:
            raise RuntimeError(f"[DDPM.generate] Invalid window length L={L}")

        if mode == "hann":
            w = np.hanning(L).astype(np.float32)
            return np.maximum(w, 1e-3)

        if mode == "triangular":
            if L == 1:
                return np.ones(1, dtype=np.float32)
            mid = (L - 1) / 2.0
            idx = np.arange(L, dtype=np.float32)
            w = 1.0 - np.abs((idx - mid) / max(mid, 1.0))
            return np.maximum(w.astype(np.float32), 1e-3)

        if mode == "flat":
            return np.ones(L, dtype=np.float32)

        raise RuntimeError(
            f"[DDPM.generate] Unknown overlap_weight={mode}. "
            "Use 'hann', 'triangular', or 'flat'."
        )

    # -----------------------------------------------------------------
    # Feature weights
    # -----------------------------------------------------------------
    @torch.no_grad()
    def fit_feature_weights(self, loader, eps: float = 1e-6, clip_hi: float = 50.0):
        eps = float(eps)
        clip_hi = float(clip_hi)

        if eps <= 0:
            raise RuntimeError(f"[DDPM.fit_feature_weights] eps must be positive, got {eps}")

        if clip_hi <= 0:
            raise RuntimeError(f"[DDPM.fit_feature_weights] clip_hi must be positive, got {clip_hi}")

        obs_sum = None
        n_sum = 0.0
        D_expected = self.D

        for batch in loader:
            if "x_obs" not in batch:
                raise RuntimeError("[DDPM.fit_feature_weights] Batch missing key 'x_obs'.")

            xobs = batch["x_obs"]
            if xobs.dtype != torch.float32:
                xobs = xobs.float()

            if xobs.ndim != 3 or xobs.shape[-1] != D_expected:
                raise RuntimeError(
                    f"[DDPM.fit_feature_weights] x_obs must be (B,L,{D_expected}), "
                    f"got {tuple(xobs.shape)}"
                )

            self._validate_binary_tensor(xobs, context="DDPM.fit_feature_weights")

            xobs = xobs.reshape(-1, xobs.shape[-1]).to(self.device_obj, dtype=torch.float32)

            s = xobs.sum(dim=0)
            obs_sum = s if obs_sum is None else (obs_sum + s)
            n_sum += float(xobs.shape[0])

        if obs_sum is None or n_sum <= 0:
            self._log_msg("[DDPM.fit_feature_weights] No observations seen; keeping uniform feat_w.")
            return

        p = (obs_sum / max(eps, float(n_sum))).clamp(min=eps, max=1.0)
        w = (1.0 / p).clamp(max=clip_hi)
        w = w / w.mean().clamp(min=eps)
        w = w.view(1, 1, -1).to(self.device_obj, dtype=torch.float32)

        if self.feat_w.shape != w.shape:
            raise RuntimeError(
                f"[DDPM.fit_feature_weights] shape mismatch: "
                f"feat_w={tuple(self.feat_w.shape)} vs w={tuple(w.shape)}"
            )

        if not torch.isfinite(w).all().item():
            raise RuntimeError("[DDPM.fit_feature_weights] computed feature weights are non-finite.")

        self.feat_w.copy_(w)

    # -----------------------------------------------------------------
    # Forward diffusion
    # -----------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if x0.shape != noise.shape:
            raise RuntimeError(
                f"[DDPM.q_sample] x0/noise shape mismatch: "
                f"x0={tuple(x0.shape)}, noise={tuple(noise.shape)}"
            )

        if x0.ndim != 3:
            raise RuntimeError(f"[DDPM.q_sample] x0 must be 3D (B,L,D), got {tuple(x0.shape)}")

        if x0.shape[-1] != self.D:
            raise RuntimeError(
                f"[DDPM.q_sample] x0 feature dim mismatch: got {x0.shape[-1]} expected {self.D}"
            )

        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise RuntimeError(
                f"[DDPM.q_sample] t must be (B,), got t={tuple(t.shape)}, x0={tuple(x0.shape)}"
            )

        if t.dtype != torch.long:
            raise RuntimeError(f"[DDPM.q_sample] t must be torch.long, got {t.dtype}")

        if int(t.min().detach().cpu().item()) < 0 or int(t.max().detach().cpu().item()) >= self.T:
            raise RuntimeError("[DDPM.q_sample] t contains timestep outside [0, T-1].")

        a_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(torch.clamp(1.0 - a_bar, min=0.0)) * noise

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def train_model(
        self,
        loader,
        epochs: int,
        lr: float,
        use_amp: bool = True,
        log_every: int = 1,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
    ):
        epochs = int(epochs)
        lr = float(lr)
        log_every = max(1, int(log_every))
        weight_decay = float(weight_decay)
        grad_clip = float(grad_clip)

        if epochs <= 0:
            raise RuntimeError(f"[DDPM.train_model] epochs must be positive, got {epochs}")

        if lr <= 0:
            raise RuntimeError(f"[DDPM.train_model] lr must be positive, got {lr}")

        if weight_decay < 0:
            raise RuntimeError(
                f"[DDPM.train_model] weight_decay must be non-negative, got {weight_decay}"
            )

        if grad_clip <= 0:
            raise RuntimeError(f"[DDPM.train_model] grad_clip must be positive, got {grad_clip}")

        opt = torch.optim.AdamW(
            self.net.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=weight_decay,
        )

        self.net.train()

        amp_enabled = bool(use_amp) and self.device.startswith("cuda") and torch.cuda.is_available()

        try:
            from torch.amp import autocast, GradScaler
            scaler = GradScaler("cuda", enabled=amp_enabled)
            amp_ctx = lambda: autocast("cuda", enabled=amp_enabled)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            amp_ctx = lambda: torch.cuda.amp.autocast(enabled=amp_enabled)

        eps_denom = 1e-6

        for ep in range(1, epochs + 1):
            losses = []
            frac_obs_list = []

            for batch in loader:
                for k in ["x", "x_obs", "cond_seq", "regime_seq", "tod_seq"]:
                    if k not in batch:
                        raise RuntimeError(f"[DDPM.train_model] Batch missing key '{k}'.")

                x0 = batch["x"].to(self.device_obj, dtype=torch.float32)
                xobs = batch["x_obs"].to(self.device_obj, dtype=torch.float32)
                cond_seq = batch["cond_seq"].to(self.device_obj, dtype=torch.float32)
                reg_seq = batch["regime_seq"].to(self.device_obj, dtype=torch.long)
                tod_seq = batch["tod_seq"].to(self.device_obj, dtype=torch.float32)

                self._validate_batch_contract(
                    x=x0,
                    x_obs=xobs,
                    cond_seq=cond_seq,
                    reg_seq=reg_seq,
                    tod_seq=tod_seq,
                    context="DDPM.train_model",
                )

                B = int(x0.shape[0])
                t = torch.randint(0, self.T, (B,), device=self.device_obj, dtype=torch.long)

                noise = torch.randn_like(x0)
                x_t = self.q_sample(x0, t, noise)

                opt.zero_grad(set_to_none=True)

                with amp_ctx():
                    pred = self.net(x_t, t, cond_seq, reg_seq, tod_seq)

                    if pred.shape != noise.shape:
                        raise RuntimeError(
                            f"[DDPM.train_model] net output shape mismatch: "
                            f"got {tuple(pred.shape)} expected {tuple(noise.shape)}"
                        )

                    if not torch.isfinite(pred).all().item():
                        raise RuntimeError("[DDPM.train_model] net prediction contains non-finite values.")

                    diff2 = (pred - noise) ** 2
                    w = xobs * self.feat_w
                    denom = torch.sum(w)

                    if float(denom.detach().cpu().item()) <= 0.0:
                        raise RuntimeError(
                            "[DDPM.train_model] Batch has zero observed weighted entries. "
                            "This indicates a broken mask/window contract."
                        )

                    loss = torch.sum(diff2 * w) / (denom + eps_denom)

                if not torch.isfinite(loss):
                    raise RuntimeError("[DDPM.train_model] Non-finite loss encountered.")

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()

                self.ema.update(self.net)

                losses.append(float(loss.detach().cpu().item()))
                frac_obs_list.append(float(xobs.mean().detach().cpu().item()))

            if not losses:
                raise RuntimeError("[DDPM.train_model] No batches were processed.")

            if (ep % log_every) == 0:
                self._log_msg(
                    f"[DDPM] ep {ep}/{epochs} loss={float(np.mean(losses)):.4f} | "
                    f"mean_obs_frac={float(np.mean(frac_obs_list)):.3f}"
                )
        self.last_train_audit = {
            "wrapper_version": self.WRAPPER_VERSION,
            "generator_family": self.GENERATOR_FAMILY,
            "role": self.ROLE,
            "architecture_policy": self.ARCHITECTURE_POLICY,
            "final_selection_policy": self.FINAL_SELECTION_POLICY,
            "test_policy": self.TEST_POLICY,
            "contributes_to_final_artifact_by_itself": bool(self.CONTRIBUTES_TO_FINAL_ARTIFACT_BY_ITSELF),
            "train_fit_scope": "caller_loader_scope_expected_TRAIN_only",
            "val_selection_metric": "none_inside_DDPM_wrapper",
            "test_only_qa": "none_inside_DDPM_wrapper",
            "epochs": int(epochs),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "grad_clip": float(grad_clip),
            "timesteps": int(self.T),
            "D": int(self.D),
            "cond_dim": int(self.cond_dim),
            "tod_dim": int(self.tod_dim),
            "K_reg": int(self.K_reg),
            "ema_decay": float(self.ema.decay),
            "use_amp": bool(use_amp),
            "amp_enabled_effective": bool(amp_enabled),
            "device": str(self.device),
            "uses_test_for_selection": False,
            "uses_test_for_repair": False,
            "mutates_final_artifact": False,
        }

    # -----------------------------------------------------------------
    # Reverse step
    # -----------------------------------------------------------------
    def _reverse_step(
        self,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        cond_seq: torch.Tensor,
        reg_seq: torch.Tensor,
        tod_seq: torch.Tensor,
        generator: Optional[torch.Generator],
        clip_x0: Optional[float],
    ) -> torch.Tensor:
        if t_tensor.ndim != 1 or t_tensor.shape[0] != x_t.shape[0]:
            raise RuntimeError(
                f"[DDPM.generate] t_tensor must be (B,), got {tuple(t_tensor.shape)} "
                f"for x_t={tuple(x_t.shape)}"
            )

        if t_tensor.dtype != torch.long:
            raise RuntimeError(f"[DDPM.generate] t_tensor must be torch.long, got {t_tensor.dtype}")

        tt_min = int(t_tensor.min().detach().cpu().item())
        tt_max = int(t_tensor.max().detach().cpu().item())

        if tt_min != tt_max:
            raise RuntimeError("[DDPM.generate] _reverse_step expects a constant timestep across batch.")

        if tt_min < 0 or tt_max >= self.T:
            raise RuntimeError("[DDPM.generate] t_tensor outside [0, T-1].")

        beta_t = self.betas[t_tensor].view(-1, 1, 1)
        alpha_t = self.alphas[t_tensor].view(-1, 1, 1)
        a_bar_t = self.alpha_bars[t_tensor].view(-1, 1, 1)
        a_bar_pm = self.alpha_bars_prev[t_tensor].view(-1, 1, 1)
        beta_td = self.beta_tilde[t_tensor].view(-1, 1, 1)

        eps = self.net(x_t, t_tensor, cond_seq, reg_seq, tod_seq)

        if eps.shape != x_t.shape:
            raise RuntimeError(
                f"[DDPM.generate] net output shape mismatch: got {tuple(eps.shape)} "
                f"expected {tuple(x_t.shape)}"
            )

        if not torch.isfinite(eps).all().item():
            raise RuntimeError("[DDPM.generate] net output contains non-finite values.")

        denom = torch.sqrt(torch.clamp(a_bar_t, min=1e-12))
        x0_pred = (x_t - torch.sqrt(torch.clamp(1.0 - a_bar_t, min=0.0)) * eps) / denom

        if clip_x0 is not None:
            clip_val = float(clip_x0)
            if clip_val <= 0:
                raise RuntimeError(f"[DDPM.generate] clip_x0 must be positive or None, got {clip_x0}")
            x0_pred = torch.clamp(x0_pred, -clip_val, clip_val)

        denom2 = torch.clamp(1.0 - a_bar_t, min=1e-12)
        coef1 = torch.sqrt(torch.clamp(a_bar_pm, min=0.0)) * beta_t / denom2
        coef2 = torch.sqrt(torch.clamp(alpha_t, min=0.0)) * (1.0 - a_bar_pm) / denom2
        mean = coef1 * x0_pred + coef2 * x_t

        if not torch.isfinite(mean).all().item():
            raise RuntimeError("[DDPM.generate] reverse mean contains non-finite values.")

        if tt_min == 0:
            return mean

        try:
            z = torch.randn(
                x_t.shape,
                device=self.device_obj,
                dtype=x_t.dtype,
                generator=generator,
            )
        except TypeError:
            z = torch.randn(x_t.shape, device=self.device_obj, dtype=x_t.dtype)

        out = mean + torch.sqrt(beta_td) * z

        if not torch.isfinite(out).all().item():
            raise RuntimeError("[DDPM.generate] reverse step output contains non-finite values.")

        return out

    # -----------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        N: int,
        D: int,
        cond_seq: np.ndarray,
        reg_seq: np.ndarray,
        tod_seq: np.ndarray,
        seq_len: int,
        stride: int,
        seed: int,
        use_ema: bool = True,
        clip_x0: Optional[float] = 6.0,
        sequence_batch: Optional[int] = None,
        overlap_weight: str = "hann",
        log_every_batches: int = 25,
        temperature: float = 1.0,
        return_audit: bool = False,
    ):
        N = int(N)
        D = int(D)

        if D <= 0:
            raise RuntimeError(f"[DDPM.generate] D must be positive, got {D}")

        if D != self.D:
            raise RuntimeError(f"[DDPM.generate] Requested D={D}, but net.D={self.D}")

        if N < 0:
            raise RuntimeError(f"[DDPM.generate] N must be >= 0, got {N}")

        if N == 0:
            out = np.zeros((0, self.D), dtype=np.float32)
            audit = {
                "sampler": self.SAMPLER_VERSION,
                "role": self.ROLE,
                "architecture_policy": self.ARCHITECTURE_POLICY,
                "N": 0,
                "D": int(self.D),
                "windows": 0,
                "wrapper_version": self.WRAPPER_VERSION,
                "generator_family": self.GENERATOR_FAMILY,
                "final_selection_policy": self.FINAL_SELECTION_POLICY,
                "test_policy": self.TEST_POLICY,
                "contributes_to_final_artifact_by_itself": bool(self.CONTRIBUTES_TO_FINAL_ARTIFACT_BY_ITSELF),
                "policy": "DDPM candidate generator only; not universal protocol generator",
                "uses_test_for_selection": False,
                "uses_test_for_repair": False,
                "mutates_final_artifact": False,
                "finite_rate": 1.0,
                "device": str(self.device),
            }
            self.last_generate_audit = audit
            return (out, audit) if return_audit else out

        cond_seq, reg_seq, tod_seq = self._validate_flat_generation_inputs(
            N=N,
            D=D,
            cond_seq=cond_seq,
            reg_seq=reg_seq,
            tod_seq=tod_seq,
        )

        L = int(seq_len)
        S = int(stride)

        if L <= 0:
            raise RuntimeError(f"[DDPM.generate] seq_len must be positive, got {L}")

        if S <= 0:
            raise RuntimeError(f"[DDPM.generate] stride must be positive, got {S}")

        if S > L:
            raise RuntimeError(f"[DDPM.generate] stride={S} cannot exceed seq_len={L}")

        L = min(L, N)

        seq_batch = int(sequence_batch if sequence_batch is not None else 64)
        if seq_batch <= 0:
            raise RuntimeError(f"[DDPM.generate] sequence_batch must be positive, got {seq_batch}")

        temp = float(temperature)
        if temp <= 0:
            raise RuntimeError(f"[DDPM.generate] temperature must be positive, got {temp}")

        log_every_batches = int(log_every_batches)
        if log_every_batches < 0:
            raise RuntimeError(
                f"[DDPM.generate] log_every_batches must be >= 0, got {log_every_batches}"
            )

        starts = list(range(0, max(N - L + 1, 1), S))
        final_start = max(0, N - L)
        if len(starts) == 0 or starts[-1] != final_start:
            starts.append(final_start)
        starts = np.asarray(sorted(set(starts)), dtype=np.int64)

        w = self._overlap_weights(L, overlap_weight).astype(np.float64)

        acc = np.zeros((N, D), dtype=np.float64)
        wacc = np.zeros((N, 1), dtype=np.float64)

        generator = self._make_torch_generator(self.device_obj, int(seed))

        prev_mode = self.net.training
        ema_stored = False

        total_windows = int(len(starts))
        total_batches = int(math.ceil(total_windows / seq_batch))

        try:
            if use_ema:
                self.ema.store(self.net)
                ema_stored = True
                self.ema.copy_to(self.net)

            self.net.eval()

            for batch_id, s0 in enumerate(range(0, total_windows, seq_batch)):
                s1 = min(total_windows, s0 + seq_batch)
                batch_starts = starts[s0:s1]
                B = int(len(batch_starts))

                cond_win = np.empty((B, L, cond_seq.shape[1]), dtype=np.float32)
                reg_win = np.empty((B, L), dtype=np.int64)
                tod_win = np.empty((B, L, tod_seq.shape[1]), dtype=np.float32)

                for bi, st in enumerate(batch_starts):
                    st = int(st)
                    en = int(st + L)

                    if en <= N:
                        cond_win[bi] = cond_seq[st:en]
                        reg_win[bi] = reg_seq[st:en]
                        tod_win[bi] = tod_seq[st:en]
                    else:
                        valid = N - st
                        if valid <= 0:
                            raise RuntimeError(
                                f"[DDPM.generate] Invalid valid length while padding: "
                                f"st={st}, N={N}, valid={valid}"
                            )

                        cond_win[bi, :valid] = cond_seq[st:N]
                        reg_win[bi, :valid] = reg_seq[st:N]
                        tod_win[bi, :valid] = tod_seq[st:N]

                        cond_win[bi, valid:] = cond_seq[-1]
                        reg_win[bi, valid:] = reg_seq[-1]
                        tod_win[bi, valid:] = tod_seq[-1]

                cseq = torch.as_tensor(cond_win, device=self.device_obj, dtype=torch.float32)
                rseq = torch.as_tensor(reg_win, device=self.device_obj, dtype=torch.long)
                tdseq = torch.as_tensor(tod_win, device=self.device_obj, dtype=torch.float32)

                try:
                    x_t = (
                        torch.randn(
                            (B, L, D),
                            device=self.device_obj,
                            dtype=torch.float32,
                            generator=generator,
                        )
                        * temp
                    )
                except TypeError:
                    x_t = torch.randn((B, L, D), device=self.device_obj, dtype=torch.float32) * temp

                self._validate_batch_contract(
                    x=x_t,
                    x_obs=None,
                    cond_seq=cseq,
                    reg_seq=rseq,
                    tod_seq=tdseq,
                    context="DDPM.generate",
                )

                for tt in reversed(range(self.T)):
                    t_tensor = torch.full((B,), int(tt), device=self.device_obj, dtype=torch.long)
                    x_t = self._reverse_step(
                        x_t=x_t,
                        t_tensor=t_tensor,
                        cond_seq=cseq,
                        reg_seq=rseq,
                        tod_seq=tdseq,
                        generator=generator,
                        clip_x0=clip_x0,
                    )

                if not torch.isfinite(x_t).all().item():
                    raise RuntimeError("[DDPM.generate] Non-finite generated tensor.")

                Xb = x_t.detach().cpu().numpy().astype(np.float32, copy=False)

                if Xb.shape != (B, L, D):
                    raise RuntimeError(
                        f"[DDPM.generate] batch output shape mismatch: "
                        f"got {Xb.shape}, expected {(B, L, D)}"
                    )

                for bi, st in enumerate(batch_starts):
                    st = int(st)
                    en = min(st + L, N)
                    valid = en - st

                    if valid <= 0:
                        raise RuntimeError(
                            f"[DDPM.generate] Invalid aggregation span: st={st}, en={en}, valid={valid}"
                        )

                    ww = w[:valid].reshape(-1, 1)
                    acc[st:en, :] += Xb[bi, :valid, :].astype(np.float64) * ww
                    wacc[st:en, :] += ww

                del x_t, cseq, rseq, tdseq, Xb

                if torch.cuda.is_available() and self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

                if log_every_batches and (
                    ((batch_id + 1) % log_every_batches == 0)
                    or (s1 == total_windows)
                ):
                    self._log_msg(
                        "[DDPM.generate] stride-aware sampling progress | "
                        f"windows={s1}/{total_windows} | "
                        f"batches={batch_id + 1}/{total_batches}"
                    )

            uncovered = int(np.sum(wacc[:, 0] <= 0))
            if uncovered > 0:
                raise RuntimeError(
                    f"[DDPM.generate] overlap aggregation left uncovered rows: {uncovered}"
                )

            out = (acc / np.maximum(wacc, 1e-12)).astype(np.float32, copy=False)

            if out.shape != (N, D):
                raise RuntimeError(
                    f"[DDPM.generate] final output shape mismatch: "
                    f"got {out.shape}, expected {(N, D)}"
                )

            finite_rate = float(np.isfinite(out).mean())
            if finite_rate < 1.0:
                raise RuntimeError(
                    f"[DDPM.generate] output contains non-finite values: "
                    f"finite_rate={finite_rate:.6f}"
                )

            audit = {
                "sampler": self.SAMPLER_VERSION,
                "role": self.ROLE,
                "architecture_policy": self.ARCHITECTURE_POLICY,
                "wrapper_version": self.WRAPPER_VERSION,
                "generator_family": self.GENERATOR_FAMILY,
                "final_selection_policy": self.FINAL_SELECTION_POLICY,
                "test_policy": self.TEST_POLICY,
                "contributes_to_final_artifact_by_itself": bool(self.CONTRIBUTES_TO_FINAL_ARTIFACT_BY_ITSELF),
                "policy": "DDPM candidate generator only; final selection must be performed by the downstream VAL-only selector",
                "train_fit_scope": "none_inside_generate",
                "val_selection_metric": "none_inside_generate",
                "test_only_qa": "none_inside_generate",
                "uses_test_for_selection": False,
                "uses_test_for_repair": False,
                "mutates_final_artifact": False,
                "N": int(N),
                "D": int(D),
                "seq_len": int(L),
                "stride": int(S),
                "windows": int(total_windows),
                "timesteps": int(self.T),
                "sequence_batch": int(seq_batch),
                "batches": int(total_batches),
                "denoiser_calls": int(total_batches * self.T),
                "window_denoising_items": int(total_windows * self.T),
                "overlap_weight": str(overlap_weight),
                "temperature": float(temp),
                "use_ema": bool(use_ema),
                "ema_stored": bool(ema_stored),
                "clip_x0": None if clip_x0 is None else float(clip_x0),
                "finite_rate": float(finite_rate),
                "cond_dim": int(cond_seq.shape[1]),
                "tod_dim": int(tod_seq.shape[1]),
                "K_reg": None if self._expected_K_reg() is None else int(self._expected_K_reg()),
                "device": str(self.device),
            }

            self.last_generate_audit = audit

            return (out, audit) if return_audit else out

        finally:
            if ema_stored:
                self.ema.restore(self.net)
            self.net.train(prev_mode)
# ==========================================================
# MANIFEST — DDPM wrapper candidate-generator contract
# ==========================================================
DDPM_WRAPPER_CONTRACT = {
    "component": "DDPM",
    "component_type": "candidate_generator_wrapper",
    "wrapper_version": "DDPM_wrapper_v5_3_THESIS",
    "generator_family": "DDPM",
    "supports_generator_families": [
        "DDPM_candidate_generator",
        "SeqDenoiser_candidate_generator",
    ],
    "branch_scope": [
        "protocol_candidate_generation",
        "continuous_candidate_generation",
        "sequence_refinement_candidate_generation",
    ],
    "parameters": {
        "timesteps_source": "CFG['ddpm_timesteps']",
        "ema_decay_source": "CFG['ddpm_ema_decay']",
        "epochs_source": "CFG['ddpm_epochs']",
        "batch_source": "CFG['ddpm_batch']",
        "learning_rate_source": "CFG['ddpm_lr']",
        "sequence_length_source": "CFG['ddpm_seq_len']",
        "stride_source": "CFG['ddpm_stride']",
        "amp_source": "CFG['ddpm_use_amp'] and CUDA availability",
        "feature_weighting": "observed-mask inverse-frequency weighting via fit_feature_weights",
        "overlap_sampler": "stride-aware overlapping sequence sampler",
        "overlap_weight_options": ["hann", "triangular", "flat"],
        "default_overlap_weight": "hann",
        "default_clip_x0": 6.0,
    },
    "seed": "caller-provided; must be recorded by downstream branch",
    "train_fit_scope": (
        "DDPM.train_model consumes the caller-provided loader. In the canonical "
        "pipeline this loader must be TRAIN-only."
    ),
    "val_selection_metric": (
        "None inside the DDPM wrapper. VAL scoring and candidate selection must "
        "be performed downstream by branch-specific selection ledgers."
    ),
    "test_only_qa": (
        "None inside the DDPM wrapper. TEST is allowed only for final QA after "
        "TRAIN/VAL decisions are frozen."
    ),
    "contributes_to_final_artifact": False,
    "final_artifact_contribution_rule": (
        "This wrapper may generate candidate values, but it does not itself "
        "authorize any generated column to enter the final scientific or public "
        "artifact. Contribution requires downstream VAL-only selection, "
        "materialization, enforcement, and TEST-only QA."
    ),
    "candidate_or_rejected_label_rule": (
        "Any DDPM/SeqDenoiser output must be labeled selected, rejected, warning, "
        "or diagnostic by the downstream candidate ledger. The wrapper output "
        "alone is never paper evidence."
    ),
    "leakage_status": {
        "reads_train_values": "caller_loader_dependent",
        "reads_val_values": False,
        "reads_test_values": False,
        "uses_test_for_selection": False,
        "uses_test_for_repair": False,
        "mutates_synthetic_artifact": False,
        "overwrites_final_artifact": False,
    },
    "paper_claim_status": (
        "Not citable as final quality evidence by itself. Cite only downstream "
        "branch results where DDPM was selected by VAL-only policy and audited "
        "on TEST without repair or overwrite."
    ),
}

if "RUN_META" in globals():
    RUN_META.setdefault("generator_contracts", {})
    RUN_META["generator_contracts"]["DDPM"] = DDPM_WRAPPER_CONTRACT

if "log" in globals() and callable(globals()["log"]):
    log("[DDPM wrapper] DDPM class loaded and registered as candidate-generator infrastructure only.")
else:
    print("[DDPM wrapper] DDPM class loaded and registered as candidate-generator infrastructure only.")