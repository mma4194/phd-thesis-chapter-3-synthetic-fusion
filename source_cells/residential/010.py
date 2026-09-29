# ============================================================
# HOTFIX CELL 4.6 — IoT generator helper functions — v4
#
# Purpose:
# - Provide numerically stable helper generators for downstream IoT synthesis
# - Keep behavior conservative and explicit when model objects are malformed
# - Fail closed on malformed runtime conditioning inputs
# - Avoid silent optimistic fallbacks that can inflate realism artificially
#
# Policy:
# - These are helper functions only
# - They do NOT decide final generation strategy
# - Unknown / malformed model objects degrade to low-information behavior
#   without inventing rich structure
# - Runtime alignment errors do NOT degrade silently; they raise
#
# Key hardening in v4:
# - no silent padding/truncation of conditioning arrays
# - strict TOD shape validation when TOD is provided
# - strict event_intensity length validation when provided
# - safer array conversion with explicit copies
# - bounded regime validation
# - Markov fallback preserves approximate stationary base rate
# ============================================================

log("--- START: Cell 4.6 — IoT generator helper functions ---")

import numpy as np

# ------------------------------------------------------------
# Small numeric helpers
# ------------------------------------------------------------
def _stable_seed(seed=0) -> int:
    try:
        s = int(seed)
    except Exception:
        s = 0
    return int(np.uint32(s).item())


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)

    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)

    return out


def _to_1d_float(x, fill=0.0, *, name="array") -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=np.float32)

    arr = np.asarray(x)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    try:
        arr = arr.reshape(-1).astype(np.float32, copy=True)
    except Exception as e:
        raise ValueError(f"{name} could not be converted to 1D float32.") from e

    arr[~np.isfinite(arr)] = float(fill)
    return arr.astype(np.float32, copy=False)


def _as_1d_float(
    x,
    N,
    fill=0.0,
    *,
    name="array",
    allow_none=True,
    strict_len=True,
) -> np.ndarray:
    """
    Convert x to 1D float32 length N.

    Policy:
    - None may become fill only if allow_none=True.
    - scalar may broadcast.
    - wrong non-scalar length is an error when strict_len=True.
    """
    N = int(N)
    if N < 0:
        raise ValueError("N must be non-negative.")

    if x is None:
        if not allow_none:
            raise ValueError(f"{name} must not be None.")
        return np.full(N, float(fill), dtype=np.float32)

    arr = _to_1d_float(x, fill=fill, name=name)

    if arr.size == N:
        out = arr.copy()
    elif arr.size == 1:
        out = np.full(N, float(arr[0]), dtype=np.float32)
    else:
        if strict_len:
            raise ValueError(
                f"{name} length mismatch: got {arr.size}, expected {N}. "
                "Refusing to pad/truncate because this would misalign time."
            )
        out = np.full(N, float(fill), dtype=np.float32)
        m = min(N, int(arr.size))
        out[:m] = arr[:m]

    out[~np.isfinite(out)] = float(fill)
    return out.astype(np.float32, copy=False)


def _as_tod_cols(tod, N, *, strict_len=True):
    """
    Return TOD sin/cos arrays.

    Policy:
    - tod=None is a conservative low-information fallback.
    - malformed provided TOD raises, because TOD misalignment corrupts
      temporal conditioning.
    """
    N = int(N)
    if N < 0:
        raise ValueError("N must be non-negative.")

    if tod is None:
        return np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

    try:
        tod = np.asarray(tod, dtype=np.float32)
    except Exception as e:
        raise ValueError("tod could not be converted to float32 array.") from e

    if tod.ndim != 2:
        raise ValueError(f"tod must be 2D with shape (N, >=2). Got ndim={tod.ndim}.")

    if tod.shape[1] < 2:
        raise ValueError(f"tod must have at least two columns [sin, cos]. Got shape={tod.shape}.")

    if tod.shape[0] != N:
        if strict_len:
            raise ValueError(
                f"tod length mismatch: got {tod.shape[0]}, expected {N}. "
                "Refusing to pad/truncate because this would misalign time."
            )

    sin_t = _as_1d_float(tod[:, 0], N, fill=0.0, name="tod[:,0]", strict_len=strict_len)
    cos_t = _as_1d_float(tod[:, 1], N, fill=0.0, name="tod[:,1]", strict_len=strict_len)

    return sin_t, cos_t


def _regime_K(reg_ids, default=1, *, max_k=1024) -> int:
    reg_ids = np.asarray(reg_ids).reshape(-1)

    if reg_ids.size == 0:
        return int(default)

    try:
        reg_num = np.asarray(reg_ids, dtype=np.float64).reshape(-1)
    except Exception:
        return int(default)

    finite = np.isfinite(reg_num)
    if not finite.any():
        return int(default)

    max_id = int(np.nanmax(reg_num[finite]))
    K = max(int(default), max_id + 1)

    if K > int(max_k):
        raise ValueError(
            f"Inferred regime count K={K} exceeds max_k={max_k}. "
            "Regime IDs may be sparse/corrupted; pass a fitted K_reg."
        )

    return int(K)


def _clip_reg(reg_ids, K):
    arr = np.asarray(reg_ids).reshape(-1)

    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)

    try:
        arr = np.asarray(arr, dtype=np.float64)
    except Exception as e:
        raise ValueError("reg_ids could not be converted to numeric array.") from e

    arr[~np.isfinite(arr)] = 0.0
    arr = np.round(arr).astype(np.int64, copy=False)

    K = max(1, int(K))
    return np.clip(arr, 0, K - 1).astype(np.int64, copy=False)


def _safe_event_log1p(event_intensity, N, cap=1e6, *, strict_len=True):
    ev = _as_1d_float(
        event_intensity,
        N,
        fill=0.0,
        name="event_intensity",
        allow_none=True,
        strict_len=strict_len,
    )

    ev[~np.isfinite(ev)] = 0.0
    ev = np.maximum(ev, 0.0)
    ev = np.clip(ev, 0.0, float(cap)).astype(np.float32, copy=False)

    return np.log1p(ev).astype(np.float32, copy=False)


def _parse_prob_or_nan(p):
    try:
        p = float(p)
    except Exception:
        return float("nan")

    if not np.isfinite(p):
        return float("nan")

    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def _safe_prob(p, default=0.5):
    p2 = _parse_prob_or_nan(p)

    if not np.isfinite(p2):
        try:
            p2 = float(default)
        except Exception:
            p2 = 0.5

    if not np.isfinite(p2):
        p2 = 0.5

    return float(np.clip(p2, 1e-6, 1.0 - 1e-6))


def _safe_float(x, default=0.0, *, lo=None, hi=None):
    try:
        y = float(x)
    except Exception:
        y = float(default)

    if not np.isfinite(y):
        y = float(default)

    if lo is not None:
        y = max(float(lo), y)
    if hi is not None:
        y = min(float(hi), y)

    return float(y)


def _student_t_noise(rng, df, loc, scale):
    df = _safe_float(df, 8.0, lo=2.1)
    loc = _safe_float(loc, 0.0)
    scale = _safe_float(scale, 1e-6, lo=1e-6)

    return float(rng.standard_t(df)) * scale + loc


def _validate_N_from_reg_ids(reg_ids, *, name="reg_ids") -> int:
    arr = np.asarray(reg_ids).reshape(-1)

    if arr.size == 0:
        return 0

    try:
        tmp = np.asarray(arr, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"{name} must be numeric-like.") from e

    if not np.isfinite(tmp).any():
        raise ValueError(f"{name} contains no finite values.")

    return int(arr.size)


# ------------------------------------------------------------
# IoT observability generator
# ------------------------------------------------------------
def gen_iot_obs_sequence(model_obj, reg_ids, tod, event_intensity, seed=0):
    """
    Generate an IoT observability mask sequence.

    Supported model types:
      - constant: Bernoulli(p)
      - logreg: autoregressive logistic model over
                [obs_tm1, reg_id, tod_sin, tod_cos, log1p(event_intensity)]

    Conservative degradation:
      - malformed / missing model => Bernoulli(base_rate)
      - default base_rate is 0.5, never fully observed
      - malformed runtime conditioning lengths raise
    """
    rng = np.random.default_rng(_stable_seed(seed))

    reg_ids = np.asarray(reg_ids).reshape(-1)
    N = _validate_N_from_reg_ids(reg_ids)

    if N <= 0:
        return np.zeros(0, dtype=np.int8)

    model_obj = model_obj if isinstance(model_obj, dict) else {}
    typ = str(model_obj.get("type", "constant")).lower().strip()

    K = int(model_obj.get("K_reg", _regime_K(reg_ids, default=1)))
    K = max(1, K)

    reg_ids = _clip_reg(reg_ids, K)
    ev_log = _safe_event_log1p(event_intensity, N, cap=1e6, strict_len=True)
    tod_sin, tod_cos = _as_tod_cols(tod, N, strict_len=True)

    base_rate = _safe_prob(model_obj.get("p", model_obj.get("base_rate", 0.5)), default=0.5)
    reg_encoding = str(model_obj.get("reg_encoding", "scalar")).lower().strip()

    if typ == "constant":
        return (rng.random(N) < base_rate).astype(np.int8)

    if typ == "logreg":
        mdl = model_obj.get("model", None)

        if reg_encoding != "scalar" or mdl is None:
            return (rng.random(N) < base_rate).astype(np.int8)

        out = np.empty(N, dtype=np.int8)
        obs_tm1 = float(np.clip(model_obj.get("init_obs", round(base_rate)), 0.0, 1.0))

        for t in range(N):
            X = np.array(
                [[
                    float(obs_tm1),
                    float(reg_ids[t]),
                    float(tod_sin[t]),
                    float(tod_cos[t]),
                    float(ev_log[t]),
                ]],
                dtype=np.float32,
            )

            try:
                p1 = float(mdl.predict_proba(X)[0, 1])
            except Exception:
                try:
                    s = float(np.asarray(mdl.decision_function(X)).reshape(-1)[0])
                    p1 = float(_sigmoid(s).reshape(-1)[0])
                except Exception:
                    p1 = base_rate

            p1 = _safe_prob(p1, default=base_rate)
            out[t] = 1 if (rng.random() < p1) else 0
            obs_tm1 = float(out[t])

        return out.astype(np.int8, copy=False)

    return (rng.random(N) < base_rate).astype(np.int8)


# ------------------------------------------------------------
# Continuous IoT generator: AR(1) + exogenous inputs
# ------------------------------------------------------------
def gen_continuous_from_ar1_exog(model_obj, reg_ids, tod, event_intensity, y0=0.0, seed=0):
    """
    Generate a continuous IoT sequence from an AR(1)+exogenous model.

    Expected design:
      x = [y_{t-1}, intercept, one-hot regime..., tod_sin, tod_cos, log1p(event_intensity)]
      y_t = x dot beta + Student-t noise

    model_obj fields:
      beta, K_reg, resid_loc, resid_scale, resid_df, lo, hi

    Conservative degradation:
      - malformed beta => low-information path around y0 with residual noise
      - malformed runtime conditioning lengths raise
      - no rich structure is invented from malformed models
    """
    rng = np.random.default_rng(_stable_seed(seed))

    reg_ids = np.asarray(reg_ids).reshape(-1)
    N = _validate_N_from_reg_ids(reg_ids)

    if N <= 0:
        return np.zeros(0, dtype=np.float32)

    model_obj = model_obj if isinstance(model_obj, dict) else {}

    K = int(model_obj.get("K_reg", _regime_K(reg_ids, default=1)))
    K = max(1, K)
    ridx = _clip_reg(reg_ids, K)

    resid_loc = _safe_float(model_obj.get("resid_loc", 0.0), 0.0)
    resid_scale = _safe_float(model_obj.get("resid_scale", 0.05), 0.05, lo=1e-6)
    resid_df = _safe_float(model_obj.get("resid_df", 8.0), 8.0, lo=2.1)

    lo = _safe_float(model_obj.get("lo", -np.inf), -np.inf)
    hi = _safe_float(model_obj.get("hi", np.inf), np.inf)

    if np.isfinite(lo) and np.isfinite(hi) and lo > hi:
        lo, hi = hi, lo

    ev_log = _safe_event_log1p(event_intensity, N, cap=1e6, strict_len=True).astype(np.float64, copy=False)

    tod_sin, tod_cos = _as_tod_cols(tod, N, strict_len=True)
    tod_sin = tod_sin.astype(np.float64, copy=False)
    tod_cos = tod_cos.astype(np.float64, copy=False)

    beta_raw = model_obj.get("beta", None)
    beta = None

    if beta_raw is not None:
        try:
            beta = np.asarray(beta_raw, dtype=np.float64).reshape(-1)
            if not np.isfinite(beta).all():
                beta = None
        except Exception:
            beta = None

    expected = 2 + K + 3

    y0 = _safe_float(y0, 0.0)

    if np.isfinite(lo) or np.isfinite(hi):
        y0 = float(np.clip(y0, lo, hi))

    # Conservative fallback for malformed beta.
    if beta is None or beta.size != expected:
        y = np.empty(N, dtype=np.float32)
        anchor = float(y0)

        for t in range(N):
            yt = anchor + _student_t_noise(rng, resid_df, resid_loc, resid_scale)

            if np.isfinite(lo) or np.isfinite(hi):
                yt = float(np.clip(yt, lo, hi))

            if not np.isfinite(yt):
                yt = float(anchor)

            y[t] = np.float32(yt)

        return y.astype(np.float32, copy=False)

    y = np.empty(N, dtype=np.float32)
    y_prev = float(y0)

    for t in range(N):
        k = int(ridx[t])

        x = np.zeros(expected, dtype=np.float64)
        x[0] = float(y_prev)
        x[1] = 1.0
        x[2 + k] = 1.0
        x[2 + K + 0] = float(tod_sin[t])
        x[2 + K + 1] = float(tod_cos[t])
        x[2 + K + 2] = float(ev_log[t])

        mu = float(np.dot(x, beta))
        eps = _student_t_noise(rng, resid_df, resid_loc, resid_scale)
        yt = mu + eps

        if np.isfinite(lo) or np.isfinite(hi):
            yt = float(np.clip(yt, lo, hi))

        if not np.isfinite(yt):
            yt = float(y_prev)

        y[t] = np.float32(yt)
        y_prev = yt

    return y.astype(np.float32, copy=False)


# ------------------------------------------------------------
# Binary IoT generator: regime-conditioned Markov chain
# ------------------------------------------------------------
def gen_binary_markov_regime(model_obj, reg_ids, x0=None, seed=0):
    """
    Generate a binary IoT state sequence using a regime-conditioned two-state Markov chain.

    model_obj fields:
      K_reg, base_rate, switch_rate, and optional per-regime dicts:
        model_obj[str(k)] = {"p01": ..., "p10": ...}

    Conservative degradation:
      - malformed / underspecified models revert to weak-switching chain
        around base_rate
      - default base_rate is 0.5, not 0.0
      - fallback transition probabilities approximately preserve stationary
        base rate
    """
    rng = np.random.default_rng(_stable_seed(seed))

    reg_ids = np.asarray(reg_ids).reshape(-1)
    N = _validate_N_from_reg_ids(reg_ids)

    if N <= 0:
        return np.zeros(0, dtype=np.int8)

    model_obj = model_obj if isinstance(model_obj, dict) else {}

    K = int(model_obj.get("K_reg", _regime_K(reg_ids, default=1)))
    K = max(1, K)
    ridx = _clip_reg(reg_ids, K)

    base = _safe_prob(model_obj.get("base_rate", 0.5), default=0.5)
    switch_rate = _safe_float(model_obj.get("switch_rate", 0.02), 0.02, lo=1e-5, hi=0.5)

    def _fallback_p01_p10(base_rate):
        """
        For a two-state Markov chain, stationary P(1) is:
            p01 / (p01 + p10)

        Choose p01=s*base and p10=s*(1-base), clipped away from zero.
        """
        b = _safe_prob(base_rate, default=0.5)
        s = float(np.clip(switch_rate, 1e-5, 0.5))

        p01 = s * b
        p10 = s * (1.0 - b)

        p01 = float(np.clip(p01, 1e-6, 0.5))
        p10 = float(np.clip(p10, 1e-6, 0.5))
        return p01, p10

    if x0 is None:
        state = 1 if (rng.random() < base) else 0
    else:
        try:
            state = 1 if int(x0) else 0
        except Exception:
            state = 1 if (rng.random() < base) else 0

    x = np.empty(N, dtype=np.int8)

    for t in range(N):
        k = int(ridx[t])
        kk = str(k)

        if kk in model_obj and isinstance(model_obj[kk], dict):
            p01 = _parse_prob_or_nan(model_obj[kk].get("p01", np.nan))
            p10 = _parse_prob_or_nan(model_obj[kk].get("p10", np.nan))

            if not np.isfinite(p01) or not np.isfinite(p10):
                p01, p10 = _fallback_p01_p10(base)
        else:
            p01, p10 = _fallback_p01_p10(base)

        if state == 0:
            if rng.random() < p01:
                state = 1
        else:
            if rng.random() < p10:
                state = 0

        x[t] = state

    return x.astype(np.int8, copy=False)


log("--- END:   Cell 4.6 — IoT generator helper functions ---")