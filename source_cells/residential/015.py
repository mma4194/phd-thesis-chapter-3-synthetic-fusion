# ==========================================================
# CORE CLASS — SequenceDataset
# v3.1 — contract-strict DDPM sequence dataset
#
# Purpose:
# - Build sliding windows for DDPM training/evaluation.
# - Enforce exact dimensional contracts for:
#       X:          (N, D), finite by default
#       x_obs:      (N, D), binary observed mask
#       cond_seq:   (N, C), finite
#       regime_seq: (N,), integer-like non-negative regime ids
#       tod_seq:    (N, tod_dim), canonical tod_dim=2
#
# Key behavior:
# - X is expected to be already mask-aware preprocessed/imputed.
# - x_obs is validated as binary by default.
# - optional thresholding of x_obs must be explicitly enabled.
# - x_finite_ref may be:
#       * explicit bool/0-1 mask
#       * raw-X matrix before imputation, using isfinite(rawX)
# - auto mode compares both interpretations and chooses lower mismatch.
# - exact TOD dimension is enforced.
# - exposes D, C, tod_dim, N, windows for audits.
# ==========================================================

from torch.utils.data import Dataset
import numpy as np
import torch


class SequenceDataset(Dataset):
    """
    Sliding-window dataset for DDPM training.

    Returns:
        {
            "x":          torch.FloatTensor, shape (L, D)
            "x_obs":      torch.FloatTensor, shape (L, D), values {0,1}
            "cond_seq":   torch.FloatTensor, shape (L, C)
            "regime_seq": torch.LongTensor,  shape (L,)
            "tod_seq":    torch.FloatTensor, shape (L, tod_dim)
            "t_idx":      torch.LongTensor,  shape (L,)     # optional
        }
    """

    def __init__(
        self,
        X,
        x_obs,
        cond_seq,
        regime_seq,
        tod_seq,
        seq_len: int,
        stride: int,
        return_t_idx: bool = False,
        debug_checks: bool = False,
        x_finite_ref=None,
        x_finite_ref_mode: str = "auto",  # {"auto", "mask", "raw"}
        debug_sample: int = 20000,
        seed: int = 1337,
        obs_threshold: float = 0.5,
        allow_threshold_x_obs: bool = False,
        expected_tod_dim: int = 2,
        max_ref_mismatch: float = 0.001,
        warn_all_observed_threshold: float = 0.999999,
        require_finite_X: bool = True,
        allow_empty_windows: bool = False,
    ):
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.return_t_idx = bool(return_t_idx)
        self.debug_checks = bool(debug_checks)
        self.obs_threshold = float(obs_threshold)
        self.allow_threshold_x_obs = bool(allow_threshold_x_obs)
        self.x_finite_ref_mode = str(x_finite_ref_mode).strip().lower()
        self.expected_tod_dim = int(expected_tod_dim)
        self.max_ref_mismatch = float(max_ref_mismatch)
        self.warn_all_observed_threshold = float(warn_all_observed_threshold)
        self.require_finite_X = bool(require_finite_X)

        rng = np.random.default_rng(int(seed))

        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive. Got seq_len={self.seq_len}")

        if self.stride <= 0:
            raise ValueError(f"stride must be positive. Got stride={self.stride}")

        if self.expected_tod_dim <= 0:
            raise ValueError(f"expected_tod_dim must be positive. Got {self.expected_tod_dim}")

        if self.x_finite_ref_mode not in {"auto", "mask", "raw"}:
            raise ValueError(
                f"x_finite_ref_mode must be one of {{'auto','mask','raw'}}. "
                f"Got {x_finite_ref_mode!r}"
            )

        if not (0.0 <= self.obs_threshold < 1.0):
            raise ValueError(f"obs_threshold must be in [0,1). Got {self.obs_threshold}")

        if self.max_ref_mismatch < 0.0:
            raise ValueError(f"max_ref_mismatch must be non-negative. Got {self.max_ref_mismatch}")

        if not (0.0 < self.warn_all_observed_threshold <= 1.0):
            raise ValueError(
                f"warn_all_observed_threshold must be in (0,1]. Got {self.warn_all_observed_threshold}"
            )

        # --------------------------------------------------
        # Local conversion helpers
        # --------------------------------------------------
        def _to_np(a, name: str, dtype=None, ndim=None, make_contig: bool = True):
            if a is None:
                raise ValueError(f"{name} must not be None.")

            if torch.is_tensor(a):
                a = a.detach().cpu().numpy()

            a = np.asarray(a)

            if ndim is not None and a.ndim != ndim:
                raise ValueError(
                    f"Expected {name}.ndim={ndim}. Got ndim={a.ndim}, shape={a.shape}"
                )

            if dtype is not None:
                try:
                    a = a.astype(dtype, copy=False)
                except Exception as e:
                    raise ValueError(f"Could not cast {name} to {dtype}.") from e

            if make_contig and not a.flags["C_CONTIGUOUS"]:
                a = np.ascontiguousarray(a)

            return a

        def _as_float32_2d(a, name: str):
            return _to_np(a, name=name, dtype=np.float32, ndim=2, make_contig=True)

        def _as_regime_int64(a, name: str):
            raw = _to_np(a, name=name, dtype=None, ndim=1, make_contig=True)

            try:
                f = raw.astype(np.float64, copy=False)
            except Exception as e:
                raise ValueError(f"{name} must be numeric/integer-like.") from e

            if not np.isfinite(f).all():
                bad = int((~np.isfinite(f)).sum())
                raise ValueError(f"{name} contains non-finite values: bad={bad}")

            rounded = np.rint(f)
            max_abs_err = float(np.max(np.abs(f - rounded))) if f.size else 0.0
            if max_abs_err > 1e-6:
                bad_idx = int(np.where(np.abs(f - rounded) > 1e-6)[0][0])
                raise ValueError(
                    f"{name} must be integer-like. first_bad_idx={bad_idx}, "
                    f"value={f[bad_idx]}, max_abs_err={max_abs_err}"
                )

            if rounded.size and np.min(rounded) < 0:
                raise ValueError(
                    f"{name} contains negative regime ids. min={int(np.min(rounded))}"
                )

            return rounded.astype(np.int64, copy=False)

        def _validate_binary_obs_mask(a, name: str):
            arr = _to_np(a, name=name, dtype=None, ndim=2, make_contig=True)

            if arr.dtype == np.bool_:
                return arr.astype(bool, copy=False)

            try:
                f = arr.astype(np.float32, copy=False)
            except Exception as e:
                raise ValueError(f"{name} must be bool or numeric 0/1 mask.") from e

            finite = np.isfinite(f)
            if not finite.all():
                bad = int((~finite).sum())
                if self.allow_threshold_x_obs:
                    f = f.copy()
                    f[~finite] = 0.0
                else:
                    raise ValueError(
                        f"{name} contains non-finite values: bad={bad}. "
                        "Fix upstream mask or set allow_threshold_x_obs=True explicitly."
                    )

            if self.allow_threshold_x_obs:
                return (f > self.obs_threshold)

            rounded = np.rint(f)
            max_abs_err = float(np.max(np.abs(f - rounded))) if f.size else 0.0
            if max_abs_err > 1e-6:
                bad_idx = np.argwhere(np.abs(f - rounded) > 1e-6)[0]
                raise ValueError(
                    f"{name} must be binary 0/1-like when allow_threshold_x_obs=False. "
                    f"first_bad_idx={tuple(map(int, bad_idx))}, value={float(f[tuple(bad_idx)])}, "
                    f"max_abs_err={max_abs_err}"
                )

            vals = set(np.unique(rounded.astype(np.int64)).tolist())
            if not vals.issubset({0, 1}):
                raise ValueError(
                    f"{name} contains non-binary values: {sorted(vals)}. "
                    "Fix upstream mask or set allow_threshold_x_obs=True explicitly."
                )

            return rounded.astype(bool, copy=False)

        # --------------------------------------------------
        # Convert and validate base arrays
        # --------------------------------------------------
        X = _as_float32_2d(X, "X")
        cond_seq = _as_float32_2d(cond_seq, "cond_seq")
        tod_seq = _as_float32_2d(tod_seq, "tod_seq")
        regime_seq = _as_regime_int64(regime_seq, "regime_seq")
        x_obs_bool = _validate_binary_obs_mask(x_obs, "x_obs")

        if X.shape[0] <= 0:
            raise ValueError("X has zero rows.")

        if X.shape[1] <= 0:
            raise ValueError("X has zero columns.")

        if x_obs_bool.shape != X.shape:
            raise ValueError(f"Shape mismatch: X.shape={X.shape} vs x_obs.shape={x_obs_bool.shape}")

        N = int(X.shape[0])
        D = int(X.shape[1])
        C = int(cond_seq.shape[1])
        tod_dim = int(tod_seq.shape[1])

        if C <= 0:
            raise ValueError(f"cond_seq has zero conditioning columns: shape={cond_seq.shape}")

        if tod_dim != self.expected_tod_dim:
            raise ValueError(
                f"Expected tod_seq shape (N,{self.expected_tod_dim}). Got {tod_seq.shape}"
            )

        if not (
            x_obs_bool.shape[0]
            == cond_seq.shape[0]
            == regime_seq.shape[0]
            == tod_seq.shape[0]
            == N
        ):
            raise ValueError(
                f"Length mismatch: "
                f"N={N}, x_obs={x_obs_bool.shape[0]}, cond={cond_seq.shape[0]}, "
                f"reg={regime_seq.shape[0]}, tod={tod_seq.shape[0]}"
            )

        if self.require_finite_X and not np.isfinite(X).all():
            bad = np.argwhere(~np.isfinite(X))
            r, c = bad[0]
            raise ValueError(
                f"X contains non-finite values, but require_finite_X=True. "
                f"Example row={int(r)}, col={int(c)}. "
                "DDPM dataset X should be mask-aware preprocessed/imputed; missingness belongs in x_obs."
            )

        if not self.require_finite_X:
            bad_observed = x_obs_bool & (~np.isfinite(X))
            if bad_observed.any():
                r, c = np.argwhere(bad_observed)[0]
                raise ValueError(
                    f"X contains non-finite value where x_obs==1. "
                    f"Example row={int(r)}, col={int(c)}."
                )

        if not np.isfinite(cond_seq).all():
            bad = np.argwhere(~np.isfinite(cond_seq))
            r, c = bad[0]
            raise ValueError(f"cond_seq contains non-finite values. Example row={int(r)}, col={int(c)}")

        if not np.isfinite(tod_seq).all():
            bad = np.argwhere(~np.isfinite(tod_seq))
            r, c = bad[0]
            raise ValueError(f"tod_seq contains non-finite values. Example row={int(r)}, col={int(c)}")

        # --------------------------------------------------
        # Reference-mask helpers
        # --------------------------------------------------
        def _mask_from_explicit(ref_arr: np.ndarray) -> np.ndarray:
            if ref_arr.dtype == np.bool_:
                return ref_arr.astype(bool, copy=False)

            r = np.asarray(ref_arr, dtype=np.float32)

            if not np.isfinite(r).all():
                # For an explicit mask, non-finite means unobserved only if
                # the caller explicitly chose mask semantics. Preserve old
                # conservative behavior here by mapping non-finite to 0.
                r = r.copy()
                r[~np.isfinite(r)] = 0.0

            rounded = np.rint(r)
            vals = set(np.unique(rounded[np.isfinite(rounded)].astype(np.int64)).tolist())
            if vals.issubset({0, 1}) and np.max(np.abs(r - rounded)) <= 1e-6:
                return rounded.astype(bool, copy=False)

            return r > 0.5

        def _mask_from_raw(ref_arr: np.ndarray) -> np.ndarray:
            return np.isfinite(np.asarray(ref_arr))

        def _sample_mismatch(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> float:
            if idx.size == 0:
                return 0.0
            return float(np.mean(a[idx] != b[idx]))

        # --------------------------------------------------
        # Debug / contract checks
        # --------------------------------------------------
        self.reference_mode_chosen = None
        self.reference_mismatch_rate = None
        self.obs_rate = float(x_obs_bool.mean())

        if self.debug_checks:
            if self.obs_rate <= 0.0:
                raise ValueError("x_obs is all zeros; DDPM would learn only missingness.")

            if self.obs_rate >= self.warn_all_observed_threshold:
                print(
                    "WARNING(SequenceDataset): x_obs is ~all ones. "
                    "This may be valid for dense tiers, but suspicious for sparse/capture-aware targets."
                )

            take = min(int(debug_sample), N)
            if take <= 0:
                raise ValueError("debug_sample produced zero sampled rows.")

            idx = rng.choice(N, size=take, replace=False)

            bad = x_obs_bool[idx] & (~np.isfinite(X[idx]))
            if np.any(bad):
                frac = float(np.mean(bad))
                r0, c0 = np.argwhere(bad)[0]
                rr = int(idx[r0])
                raise ValueError(
                    f"Found non-finite X where x_obs==1 in sampled rows. "
                    f"bad_rate={frac:.6f}. Example row={rr}, col={int(c0)}. "
                    "Likely cause: x_obs was computed from a different mask than this DDPM target X."
                )

            if x_finite_ref is not None:
                ref = _to_np(
                    x_finite_ref,
                    name="x_finite_ref",
                    dtype=None,
                    ndim=2,
                    make_contig=True,
                )

                if ref.shape != X.shape:
                    raise ValueError(f"x_finite_ref shape {ref.shape} != X shape {X.shape}")

                if self.x_finite_ref_mode == "mask":
                    ref_mask = _mask_from_explicit(ref)
                    mismatch = _sample_mismatch(ref_mask, x_obs_bool, idx)
                    chosen_mode = "mask"

                elif self.x_finite_ref_mode == "raw":
                    ref_mask = _mask_from_raw(ref)
                    mismatch = _sample_mismatch(ref_mask, x_obs_bool, idx)
                    chosen_mode = "raw"

                else:
                    ref_mask_mask = _mask_from_explicit(ref)
                    ref_mask_raw = _mask_from_raw(ref)

                    mismatch_mask = _sample_mismatch(ref_mask_mask, x_obs_bool, idx)
                    mismatch_raw = _sample_mismatch(ref_mask_raw, x_obs_bool, idx)

                    if abs(mismatch_mask - mismatch_raw) < 1e-4:
                        if min(mismatch_mask, mismatch_raw) <= self.max_ref_mismatch:
                            ref_mask = ref_mask_mask
                            mismatch = mismatch_mask
                            chosen_mode = "auto->mask_tie_ok"
                        else:
                            raise ValueError(
                                f"x_finite_ref auto-mode is ambiguous and both interpretations mismatch too much: "
                                f"mismatch_mask={mismatch_mask:.6f}, mismatch_raw={mismatch_raw:.6f}. "
                                "Set x_finite_ref_mode explicitly to 'mask' or 'raw', or fix upstream masks."
                            )

                    elif mismatch_mask < mismatch_raw:
                        ref_mask = ref_mask_mask
                        mismatch = mismatch_mask
                        chosen_mode = "auto->mask"

                    else:
                        ref_mask = ref_mask_raw
                        mismatch = mismatch_raw
                        chosen_mode = "auto->raw"

                self.reference_mode_chosen = chosen_mode
                self.reference_mismatch_rate = float(mismatch)

                if mismatch > self.max_ref_mismatch:
                    raise ValueError(
                        f"x_obs mismatch vs reference mask: mismatch_rate={mismatch:.6f} "
                        f"(mode={chosen_mode}, allowed={self.max_ref_mismatch:.6f}). "
                        "Ensure x_finite_ref is either the explicit observed mask or the raw pre-impute X "
                        "corresponding to this exact DDPM target matrix."
                    )

        # --------------------------------------------------
        # Store arrays
        # --------------------------------------------------
        self.X = np.ascontiguousarray(X.astype(np.float32, copy=False))
        self.x_obs = np.ascontiguousarray(x_obs_bool.astype(np.float32, copy=False))
        self.cond = np.ascontiguousarray(cond_seq.astype(np.float32, copy=False))
        self.reg = np.ascontiguousarray(regime_seq.astype(np.int64, copy=False))
        self.tod = np.ascontiguousarray(tod_seq.astype(np.float32, copy=False))

        self.N = int(N)
        self.D = int(D)
        self.C = int(C)
        self.tod_dim = int(tod_dim)

        max_start = self.N - self.seq_len

        if max_start < 0:
            if allow_empty_windows:
                self.indices = np.zeros(0, dtype=np.int64)
            else:
                raise ValueError(
                    f"seq_len={self.seq_len} is larger than dataset length N={self.N}"
                )
        else:
            self.indices = np.arange(0, max_start + 1, self.stride, dtype=np.int64)

        if self.indices.size == 0 and not allow_empty_windows:
            raise ValueError(
                f"No windows produced. N={self.N}, seq_len={self.seq_len}, stride={self.stride}"
            )

        self.windows = int(self.indices.size)
        self._base_idx = np.arange(self.N, dtype=np.int64) if self.return_t_idx else None

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, idx: int):
        idx = int(idx)

        if idx < 0:
            idx += len(self)

        if idx < 0 or idx >= len(self):
            raise IndexError(f"SequenceDataset index out of range: {idx}")

        s = int(self.indices[idx])
        e = s + self.seq_len

        x = np.ascontiguousarray(self.X[s:e])
        x_obs = np.ascontiguousarray(self.x_obs[s:e])
        cond = np.ascontiguousarray(self.cond[s:e])
        reg = np.ascontiguousarray(self.reg[s:e])
        tod = np.ascontiguousarray(self.tod[s:e])

        out = {
            "x": torch.from_numpy(x),
            "x_obs": torch.from_numpy(x_obs),
            "cond_seq": torch.from_numpy(cond),
            "regime_seq": torch.from_numpy(reg),
            "tod_seq": torch.from_numpy(tod),
        }

        if self.return_t_idx:
            out["t_idx"] = torch.from_numpy(self._base_idx[s:e])

        return out

    def audit_dict(self):
        return {
            "class": "SequenceDataset",
            "version": "v3.1",
            "N": int(self.N),
            "D": int(self.D),
            "C": int(self.C),
            "tod_dim": int(self.tod_dim),
            "seq_len": int(self.seq_len),
            "stride": int(self.stride),
            "windows": int(self.windows),
            "return_t_idx": bool(self.return_t_idx),
            "obs_threshold": float(self.obs_threshold),
            "allow_threshold_x_obs": bool(self.allow_threshold_x_obs),
            "require_finite_X": bool(self.require_finite_X),
            "obs_rate": float(self.obs_rate),
            "x_finite_ref_mode": str(self.x_finite_ref_mode),
            "reference_mode_chosen": self.reference_mode_chosen,
            "reference_mismatch_rate": self.reference_mismatch_rate,
        }

    def __repr__(self):
        return (
            f"SequenceDataset(N={self.N}, D={self.D}, C={self.C}, tod_dim={self.tod_dim}, "
            f"seq_len={self.seq_len}, stride={self.stride}, windows={self.windows}, "
            f"return_t_idx={self.return_t_idx}, obs_threshold={self.obs_threshold}, "
            f"allow_threshold_x_obs={self.allow_threshold_x_obs}, "
            f"require_finite_X={self.require_finite_X}, "
            f"x_finite_ref_mode='{self.x_finite_ref_mode}')"
        )