# ==========================================================
# CELL 10.4e — OTA packet-rate regime candidate portfolio — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Generate packet-count-specific OTA candidates for ota__pkt_total only.
# - Do not touch ota__bytes_total.
# - Build predefined packet-rate regime variants:
#     1) short regime block replay
#     2) medium regime block replay
#     3) long regime block replay
#     4) quantile-state packet-rate candidate
#     5) hybrid regime + quantile candidate
#
# Scientific contract:
# - Fit TRAIN only.
# - Emit VAL and TEST-horizon candidate artifacts.
# - Do not read real TEST values.
# - Do not use VAL for fitting.
# - Do not tune on TEST.
# - Do not select anything in this cell.
# - Not final evidence unless selected downstream on VAL against A1.
#
# Public-release caveat:
# - Block/hybrid variants may replay TRAIN packet-rate trajectories. If selected
#   for any public-release scope, Q6/copy-risk/no-copy audits are mandatory.
# ==========================================================

log("--- START: Cell 10.4e — OTA packet-rate regime candidate portfolio (v2-THESIS) ---")

import os
import json
import time
import hashlib
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
need = [
    "CFG", "log",
    "df_tr", "df_va", "df_te",
    "CELL10_PROTO_COLS",
    "CELL10_VAL_MASKS", "CELL10_TEST_MASKS",
    "_cell10_register_portfolio_candidate",
    "_cell10_register_unavailable_backend",
    "_mask_for_tier", "_postprocess_protocol_values",
    "_safe_mean", "_safe_var", "_nonzero_rate", "_transition_rate",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.4e] Missing prerequisites: {missing}. Run Cells 10.1–10.4d first.")

# ----------------------------------------------------------
# 0b) Clean-run leakage guard
# ----------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[Cell10.4e] Clean OTA packet-rate candidate forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.4e] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

OUTDIR = str(CFG["outdir"])
OUT_PORT = os.path.join(OUTDIR, "portfolio_candidates")
OUT_ART = os.path.join(OUTDIR, "artifacts")
OUT_REP = os.path.join(OUTDIR, "reports")
CONTRACT_DIR = os.path.join(OUT_ART, "contracts")

for d in [OUT_PORT, OUT_ART, OUT_REP, CONTRACT_DIR]:
    os.makedirs(d, exist_ok=True)

seed = int(CFG.get("seed", 1337))
TARGET_COL = "ota__pkt_total"

# ----------------------------------------------------------
# 1) Resolve target schema without using VAL/TEST values
# ----------------------------------------------------------
target_available = (
    TARGET_COL in set(map(str, CELL10_PROTO_COLS))
    and TARGET_COL in df_tr.columns
    and TARGET_COL in df_va.columns
    and TARGET_COL in df_te.columns
)

eligibility_path = os.path.join(
    OUT_REP,
    "cell10_4e_ota_pkt_rate_regime_eligibility_v2_THESIS.csv",
)

pd.DataFrame([{
    "col": TARGET_COL,
    "expected": True,
    "in_CELL10_PROTO_COLS": TARGET_COL in set(map(str, CELL10_PROTO_COLS)),
    "in_df_tr_schema": TARGET_COL in df_tr.columns,
    "in_df_va_schema": TARGET_COL in df_va.columns,
    "in_df_te_schema": TARGET_COL in df_te.columns,
    "selected_target": bool(target_available),
}]).to_csv(eligibility_path, index=False)

# ----------------------------------------------------------
# 2) JSON helpers and unavailable finalizer
# ----------------------------------------------------------
def _write_atomic_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    os.replace(tmp, path)


def _stable_seed_from_parts(*parts, base_seed: int = 1337) -> int:
    payload = "::".join(str(p) for p in parts)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int((int(h[:12], 16) + int(base_seed)) % (2**32 - 1))


def _finalize_unavailable(reason: str, extra_meta: dict = None):
    _cell10_register_unavailable_backend(
        "ota_pkt_rate_regime",
        reason=reason,
        meta={
            "source": "Cell10.4e",
            "version": "cell10_4e_ota_pkt_rate_regime_v2_THESIS",
            "target_col": TARGET_COL,
            "eligibility_audit_path": eligibility_path,
            **dict(extra_meta or {}),
        },
    )

    audit = {
        "version": "cell10_4e_ota_pkt_rate_regime_candidate_v2_THESIS",
        "candidate_available": False,
        "reason": reason,
        "target_col": TARGET_COL,
        "registered_candidate_ids": [],
        "variants_n": 0,
        "eligibility_audit_path": eligibility_path,
        "leakage_status": {
            "uses_train_values_for_fitting": True,
            "uses_val_values_for_fitting": False,
            "uses_test_values_for_fitting": False,
            "uses_val_masks_for_val_candidate": False,
            "uses_test_masks_for_test_candidate": False,
            "uses_test_real_values_for_generation": False,
            "uses_test_values_for_selection": False,
            "uses_test_values_for_repair": False,
            "uses_test_values_for_candidate_promotion": False,
            "materializes_test_candidate": False,
            "overwrites_final_artifact": False,
        },
        "paper_claim_status": (
            "OTA packet-rate regime backend unavailable because the target schema "
            "or TRAIN-only support conditions were not satisfied."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_4e_ota_pkt_rate_regime_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_4e_ota_pkt_rate_regime_contract_v2_THESIS.json")

    _write_atomic_json(audit_path, audit)
    _write_atomic_json(contract_path, audit)

    globals()["CELL10_4E_OTA_PKT_AUDIT"] = audit
    globals()["CELL10_4E_OTA_PKT_AUDIT_PATH"] = audit_path
    globals()["CELL10_4E_OTA_PKT_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["ota_pkt_rate_regime"] = audit

    log(f"[Cell10.4e] OTA packet-rate regime unavailable: {reason}")
    log(f"[Cell10.4e] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.4e] Saved audit: {audit_path}")
    log(f"[Cell10.4e] Saved contract: {contract_path}")


if not target_available:
    _finalize_unavailable(
        reason=f"Target column {TARGET_COL!r} not found in required protocol schemas.",
        extra_meta={"target_col": TARGET_COL},
    )

else:
    # ------------------------------------------------------
    # 3) Helper functions
    # ------------------------------------------------------
    def _postprocess_pkt(x) -> np.ndarray:
        return _postprocess_protocol_values(TARGET_COL, x)


    def _train_ota_active_mask() -> np.ndarray:
        if "ota__obs_present" not in df_tr.columns:
            raise RuntimeError("[Cell10.4e] TRAIN missing ota__obs_present.")
        v = pd.to_numeric(df_tr["ota__obs_present"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        return v > 0.5


    def _contiguous_segments_from_mask(mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []

        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks] + 1, idx[-1] + 1]

        return [
            (int(s), int(e))
            for s, e in zip(starts, ends)
            if int(e) > int(s)
        ]


    def _lag_autocorr(x, lag: int):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        lag = int(lag)

        if lag <= 0 or x.size <= lag + 2:
            return np.nan

        a = x[:-lag]
        b = x[lag:]

        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            return np.nan

        return float(np.corrcoef(a, b)[0, 1])


    def _basic_profile(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]

        if x.size == 0:
            return {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "q05": np.nan,
                "q25": np.nan,
                "q50": np.nan,
                "q75": np.nan,
                "q95": np.nan,
                "q99": np.nan,
                "nonzero_rate": np.nan,
                "transition_rate": np.nan,
                "lag1": np.nan,
                "lag5": np.nan,
                "lag30": np.nan,
                "lag60": np.nan,
                "lag300": np.nan,
            }

        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "q05": float(np.quantile(x, 0.05)),
            "q25": float(np.quantile(x, 0.25)),
            "q50": float(np.quantile(x, 0.50)),
            "q75": float(np.quantile(x, 0.75)),
            "q95": float(np.quantile(x, 0.95)),
            "q99": float(np.quantile(x, 0.99)),
            "nonzero_rate": float(np.mean(x > 0)),
            "transition_rate": float(np.mean(np.abs(np.diff(x)) > 1e-9)) if x.size > 1 else np.nan,
            "lag1": _lag_autocorr(x, 1),
            "lag5": _lag_autocorr(x, 5),
            "lag30": _lag_autocorr(x, 30),
            "lag60": _lag_autocorr(x, 60),
            "lag300": _lag_autocorr(x, 300),
        }


    def _make_quantile_bins(x, n_bins: int):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]

        if x.size == 0:
            return np.asarray([0.0, 1.0], dtype=np.float64)

        n_bins = int(max(2, n_bins))
        edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)

        if edges.size < 2:
            val = float(edges[0]) if edges.size else 0.0
            edges = np.asarray([val, val + 1.0], dtype=np.float64)

        return edges.astype(np.float64)


    def _states_from_edges(x, edges):
        x = np.asarray(x, dtype=np.float64)
        states = np.searchsorted(edges, x, side="right") - 1
        states = np.clip(states, 0, len(edges) - 2)
        return states.astype(np.int64)


    def _state_pools(x, states):
        pools = {}
        x = np.asarray(x, dtype=np.float32)
        states = np.asarray(states, dtype=np.int64)

        for s in np.unique(states):
            vals = x[states == int(s)]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                pools[int(s)] = vals.astype(np.float32, copy=False)

        return pools


    def _state_transition_matrix(states, n_states: int, alpha: float = 0.5):
        states = np.asarray(states, dtype=np.int64)
        n_states = int(n_states)
        M = np.full((n_states, n_states), float(alpha), dtype=np.float64)

        if states.size >= 2:
            for a, b in zip(states[:-1], states[1:]):
                if 0 <= a < n_states and 0 <= b < n_states:
                    M[int(a), int(b)] += 1.0

        M = M / M.sum(axis=1, keepdims=True)
        return M


    def _sample_markov_states(rng_local: np.random.Generator, M: np.ndarray, init_probs: np.ndarray, n: int):
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.int64)

        n_states = int(M.shape[0])
        out = np.zeros(n, dtype=np.int64)
        out[0] = int(rng_local.choice(np.arange(n_states), p=init_probs))

        for i in range(1, n):
            out[i] = int(rng_local.choice(np.arange(n_states), p=M[out[i - 1]]))

        return out


    def _sample_values_from_states(
        rng_local: np.random.Generator,
        states: np.ndarray,
        pools: dict,
        global_pool: np.ndarray,
    ):
        states = np.asarray(states, dtype=np.int64)
        out = np.zeros(states.size, dtype=np.float32)

        for s in np.unique(states):
            idx = np.flatnonzero(states == int(s))
            pool = pools.get(int(s), global_pool)

            if pool is None or len(pool) == 0:
                pool = global_pool

            draw = rng_local.choice(pool, size=len(idx), replace=True)
            out[idx] = draw.astype(np.float32)

        return out


    def _sample_block_length(
        rng_local: np.random.Generator,
        block_min: int,
        block_max: int,
        remaining: int,
        mode: str,
    ):
        block_min = int(max(1, block_min))
        block_max = int(max(block_min, block_max))
        remaining = int(max(1, remaining))

        hi = min(block_max, remaining)
        lo = min(block_min, hi)

        if hi <= lo:
            return int(hi)

        mode = str(mode)

        if mode == "short":
            u = rng_local.beta(1.4, 4.0)
            return int(round(lo + u * (hi - lo)))

        if mode == "medium":
            lo_log = np.log(max(1, lo))
            hi_log = np.log(max(1, hi))
            return int(round(np.exp(rng_local.uniform(lo_log, hi_log))))

        if mode == "long":
            u = rng_local.beta(4.0, 1.4)
            return int(round(lo + u * (hi - lo)))

        return int(rng_local.integers(lo, hi + 1))


    # ------------------------------------------------------
    # 4) TRAIN-only fitting
    # ------------------------------------------------------
    try:
        train_active = _train_ota_active_mask()
        x_full = pd.to_numeric(df_tr[TARGET_COL], errors="coerce").to_numpy(dtype=np.float32)

        finite = np.isfinite(x_full)
        valid_active = train_active & finite

        min_train_active = int(CFG.get("cell10_4e_ota_pkt_min_train_active_rows", 1000))
        if int(valid_active.sum()) < min_train_active:
            raise RuntimeError(
                f"Insufficient valid TRAIN OTA pkt rows: {int(valid_active.sum())} < {min_train_active}"
            )

        x_full = np.maximum(x_full, 0.0)
        x_full = np.rint(x_full).astype(np.float32)

        train_vals = _postprocess_pkt(x_full[valid_active])
        train_vals = train_vals[np.isfinite(train_vals)]

        if train_vals.size < min_train_active:
            raise RuntimeError(
                f"Insufficient postprocessed TRAIN OTA pkt values: {train_vals.size} < {min_train_active}"
            )

        raw_segments = _contiguous_segments_from_mask(valid_active)
        min_seg_len = int(CFG.get("cell10_4e_ota_pkt_min_segment_len", 30))
        segments = [(s, e) for s, e in raw_segments if (e - s) >= min_seg_len]

        if not segments:
            idx = np.flatnonzero(valid_active)
            if idx.size <= 0:
                raise RuntimeError("No valid TRAIN OTA pkt active rows.")
            segments = [(int(idx[0]), int(idx[-1]) + 1)]

        segment_infos = []
        for si, (s, e) in enumerate(segments):
            xs = x_full[s:e]
            xs = xs[np.isfinite(xs)]

            if xs.size == 0:
                continue

            segment_infos.append({
                "segment_id": int(si),
                "start": int(s),
                "end": int(e),
                "length": int(e - s),
                "profile": _basic_profile(xs),
            })

        if not segment_infos:
            raise RuntimeError("No usable TRAIN OTA packet segments.")

        seg_lengths = np.asarray([x["length"] for x in segment_infos], dtype=np.float64)
        if seg_lengths.size <= 0 or float(seg_lengths.sum()) <= 0:
            raise RuntimeError("Invalid OTA packet segment lengths.")

        seg_probs = seg_lengths / float(seg_lengths.sum())

        n_state_bins = int(CFG.get("cell10_4e_ota_pkt_state_bins", 8))
        edges = _make_quantile_bins(train_vals, n_state_bins)
        train_states = _states_from_edges(train_vals, edges)
        n_states = int(len(edges) - 1)

        pools = _state_pools(train_vals, train_states)
        global_pool = train_vals.astype(np.float32)

        init_window = min(len(train_states), int(CFG.get("cell10_4e_init_window", 10000)))
        init_counts = np.bincount(train_states[:init_window], minlength=n_states).astype(np.float64)
        init_probs = init_counts + 0.5
        init_probs = init_probs / init_probs.sum()

        M = _state_transition_matrix(
            train_states,
            n_states=n_states,
            alpha=float(CFG.get("cell10_4e_ota_pkt_markov_alpha", 0.5)),
        )

        train_profile = _basic_profile(train_vals)

    except Exception as e:
        _finalize_unavailable(
            reason=f"{type(e).__name__}: {e}",
            extra_meta={"target_col": TARGET_COL},
        )

    else:
        # --------------------------------------------------
        # 5) Generation functions
        # --------------------------------------------------
        def _choose_segment(rng_local: np.random.Generator):
            return segment_infos[int(rng_local.choice(np.arange(len(segment_infos)), p=seg_probs))]


        def _sample_train_block(
            rng_local: np.random.Generator,
            block_min: int,
            block_max: int,
            remaining: int,
            length_mode: str,
        ):
            remaining = int(remaining)
            if remaining <= 0:
                return np.zeros(0, dtype=np.float32)

            for _ in range(200):
                seg = _choose_segment(rng_local)
                s = int(seg["start"])
                e = int(seg["end"])
                seg_len = int(e - s)

                if seg_len <= 0:
                    continue

                L = _sample_block_length(
                    rng_local,
                    block_min=block_min,
                    block_max=min(block_max, seg_len),
                    remaining=remaining,
                    mode=length_mode,
                )
                L = int(min(max(1, L), seg_len, remaining))

                start_hi = e - L
                if start_hi < s:
                    continue

                start = int(rng_local.integers(s, start_hi + 1))
                block = x_full[start:start + L]
                block = block[np.isfinite(block)]

                if block.size:
                    return _postprocess_pkt(block)

            take = min(remaining, max(1, int(block_min)))
            draw = rng_local.choice(global_pool, size=take, replace=True)
            return _postprocess_pkt(draw)


        def _generate_block_variant(mask_dict: dict, N: int, split_name: str, spec: dict):
            active = _mask_for_tier(mask_dict, "ota", N, missing_policy="error")
            n_active = int(active.sum())

            out = np.full(N, np.nan, dtype=np.float32)
            block_lengths = []

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_4e_ota_pkt_rate_regime",
                    split_name,
                    spec["candidate_id"],
                    seed,
                    base_seed=seed + 10049,
                )
            )

            if n_active > 0:
                chunks = []
                produced = 0
                guard = 0

                while produced < n_active:
                    guard += 1
                    if guard > n_active + 10000:
                        raise RuntimeError(
                            f"[Cell10.4e] Block generation guard exceeded for {split_name}/{spec['candidate_id']}."
                        )

                    block = _sample_train_block(
                        rng_local,
                        block_min=int(spec["block_min"]),
                        block_max=int(spec["block_max"]),
                        remaining=n_active - produced,
                        length_mode=str(spec["length_mode"]),
                    )

                    if block.size == 0:
                        continue

                    chunks.append(block)
                    block_lengths.append(int(block.size))
                    produced += int(block.size)

                vals = np.concatenate(chunks)[:n_active].astype(np.float32)
                out[active] = _postprocess_pkt(vals)

            return pd.DataFrame({TARGET_COL: _postprocess_pkt(out)}), block_lengths


        def _generate_quantile_state(mask_dict: dict, N: int, split_name: str, spec: dict):
            active = _mask_for_tier(mask_dict, "ota", N, missing_policy="error")
            n_active = int(active.sum())

            out = np.full(N, np.nan, dtype=np.float32)

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_4e_ota_pkt_rate_quantile_state",
                    split_name,
                    spec["candidate_id"],
                    seed,
                    base_seed=seed + 10050,
                )
            )

            block_lengths = []

            if n_active > 0:
                states = _sample_markov_states(rng_local, M, init_probs, n_active)
                vals = _sample_values_from_states(rng_local, states, pools, global_pool)

                if bool(spec.get("smooth_state_runs", False)):
                    patch_frac = float(spec.get("patch_frac", 0.10))
                    patch_budget = int(max(0, min(n_active, round(n_active * patch_frac))))
                    pos = 0

                    while pos < patch_budget:
                        L = int(rng_local.integers(10, min(120, n_active - pos) + 1))
                        if L <= 0:
                            break

                        block = _sample_train_block(rng_local, 10, 120, L, "short")
                        if block.size == 0:
                            break

                        start = int(rng_local.integers(0, max(1, n_active - block.size + 1)))
                        take = max(0, min(block.size, n_active - start))
                        vals[start:start + take] = block[:take]
                        block_lengths.append(int(take))
                        pos += int(take)

                out[active] = _postprocess_pkt(vals)

            return pd.DataFrame({TARGET_COL: _postprocess_pkt(out)}), block_lengths


        def _generate_hybrid(mask_dict: dict, N: int, split_name: str, spec: dict):
            active = _mask_for_tier(mask_dict, "ota", N, missing_policy="error")
            n_active = int(active.sum())

            out = np.full(N, np.nan, dtype=np.float32)
            block_lengths = []

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_4e_ota_pkt_rate_hybrid",
                    split_name,
                    spec["candidate_id"],
                    seed,
                    base_seed=seed + 10051,
                )
            )

            if n_active > 0:
                vals = np.zeros(n_active, dtype=np.float32)
                produced = 0
                guard = 0

                while produced < n_active:
                    guard += 1
                    if guard > n_active + 10000:
                        raise RuntimeError(
                            f"[Cell10.4e] Hybrid generation guard exceeded for {split_name}/{spec['candidate_id']}."
                        )

                    use_block = bool(rng_local.random() < float(spec.get("block_prob", 0.65)))

                    if use_block:
                        block = _sample_train_block(
                            rng_local,
                            block_min=int(spec["block_min"]),
                            block_max=int(spec["block_max"]),
                            remaining=n_active - produced,
                            length_mode=str(spec["length_mode"]),
                        )
                    else:
                        L = _sample_block_length(
                            rng_local,
                            block_min=int(spec["block_min"]),
                            block_max=int(spec["block_max"]),
                            remaining=n_active - produced,
                            mode=str(spec["length_mode"]),
                        )
                        states = _sample_markov_states(rng_local, M, init_probs, L)
                        block = _sample_values_from_states(rng_local, states, pools, global_pool)

                    if block.size == 0:
                        continue

                    take = min(block.size, n_active - produced)
                    vals[produced:produced + take] = block[:take]
                    block_lengths.append(int(take))
                    produced += int(take)

                out[active] = _postprocess_pkt(vals)

            return pd.DataFrame({TARGET_COL: _postprocess_pkt(out)}), block_lengths


        def _validate_candidate_frame(df_out: pd.DataFrame, mask_dict: dict, N: int, split_name: str, candidate_id: str, block_lengths):
            if list(df_out.columns) != [TARGET_COL]:
                raise RuntimeError(
                    f"[Cell10.4e] {split_name}/{candidate_id}: unexpected columns {list(df_out.columns)}"
                )

            if int(len(df_out)) != int(N):
                raise RuntimeError(
                    f"[Cell10.4e] {split_name}/{candidate_id}: length mismatch {len(df_out)} vs {N}"
                )

            active = _mask_for_tier(mask_dict, "ota", N, missing_policy="error")
            x = pd.to_numeric(df_out[TARGET_COL], errors="coerce").to_numpy(dtype=np.float64)

            finite_active = int(np.isfinite(x[active]).sum())
            finite_inactive = int(np.isfinite(x[~active]).sum())

            if int(active.sum()) > 0 and finite_active != int(active.sum()):
                raise RuntimeError(
                    f"[Cell10.4e] {split_name}/{candidate_id}: active finite mismatch "
                    f"{finite_active} vs {int(active.sum())}"
                )

            if finite_inactive != 0:
                raise RuntimeError(
                    f"[Cell10.4e] {split_name}/{candidate_id}: inactive finite values {finite_inactive}"
                )

            xa = x[active]

            return {
                "split": split_name,
                "candidate_id": candidate_id,
                "col": TARGET_COL,
                "tier": "ota",
                "N": int(N),
                "active_n": int(active.sum()),
                "active_rate": float(np.mean(active)) if active.size else np.nan,
                "finite_active": int(finite_active),
                "finite_inactive": int(finite_inactive),
                "generated_mean_active": _safe_mean(xa),
                "generated_var_active": _safe_var(xa),
                "generated_nonzero_rate_active": _nonzero_rate(xa),
                "generated_transition_rate_active": _transition_rate(xa),
                "block_draws": int(len(block_lengths)),
                "block_len_median": float(np.median(block_lengths)) if block_lengths else np.nan,
                "block_len_p95": float(np.quantile(block_lengths, 0.95)) if block_lengths else np.nan,
            }


        # --------------------------------------------------
        # 6) Predefined variants
        # --------------------------------------------------
        variant_specs = [
            {
                "generator": "ota_pkt_rate_regime",
                "candidate_id": "ota_pkt_rate_regime_short_v2_THESIS",
                "variant_name": "short_block_pkt_rate",
                "kind": "block",
                "block_min": int(CFG.get("cell10_4e_pkt_short_block_min", 5)),
                "block_max": int(CFG.get("cell10_4e_pkt_short_block_max", 90)),
                "length_mode": "short",
                "role": "local_packet_transition_candidate",
            },
            {
                "generator": "ota_pkt_rate_regime",
                "candidate_id": "ota_pkt_rate_regime_medium_v2_THESIS",
                "variant_name": "medium_block_pkt_rate",
                "kind": "block",
                "block_min": int(CFG.get("cell10_4e_pkt_medium_block_min", 30)),
                "block_max": int(CFG.get("cell10_4e_pkt_medium_block_max", 600)),
                "length_mode": "medium",
                "role": "balanced_packet_autocorrelation_candidate",
            },
            {
                "generator": "ota_pkt_rate_regime",
                "candidate_id": "ota_pkt_rate_regime_long_v2_THESIS",
                "variant_name": "long_block_pkt_rate",
                "kind": "block",
                "block_min": int(CFG.get("cell10_4e_pkt_long_block_min", 180)),
                "block_max": int(CFG.get("cell10_4e_pkt_long_block_max", 2400)),
                "length_mode": "long",
                "role": "long_range_packet_autocorrelation_candidate",
            },
            {
                "generator": "ota_pkt_rate_regime",
                "candidate_id": "ota_pkt_rate_quantile_state_v2_THESIS",
                "variant_name": "quantile_state_pkt_rate",
                "kind": "quantile_state",
                "smooth_state_runs": False,
                "role": "packet_support_distribution_state_candidate",
            },
            {
                "generator": "ota_pkt_rate_regime",
                "candidate_id": "ota_pkt_rate_hybrid_v2_THESIS",
                "variant_name": "hybrid_block_quantile_pkt_rate",
                "kind": "hybrid",
                "block_min": int(CFG.get("cell10_4e_pkt_hybrid_block_min", 10)),
                "block_max": int(CFG.get("cell10_4e_pkt_hybrid_block_max", 900)),
                "length_mode": "medium",
                "block_prob": float(CFG.get("cell10_4e_pkt_hybrid_block_prob", 0.65)),
                "role": "hybrid_packet_temporal_support_candidate",
            },
        ]

        for spec in variant_specs:
            spec["block_min"] = int(max(1, spec.get("block_min", 1)))
            spec["block_max"] = int(max(spec["block_min"], spec.get("block_max", spec["block_min"])))

        registered = []
        meta_rows = []
        generation_audit_rows = []

        # --------------------------------------------------
        # 7) Materialize/register variants
        # --------------------------------------------------
        for spec in variant_specs:
            cid = str(spec["candidate_id"])
            gen = str(spec["generator"])

            if spec["kind"] == "block":
                val_df, val_block_lengths = _generate_block_variant(CELL10_VAL_MASKS, len(df_va), "VAL", spec)
                test_df, test_block_lengths = _generate_block_variant(CELL10_TEST_MASKS, len(df_te), "TEST", spec)
            elif spec["kind"] == "quantile_state":
                val_df, val_block_lengths = _generate_quantile_state(CELL10_VAL_MASKS, len(df_va), "VAL", spec)
                test_df, test_block_lengths = _generate_quantile_state(CELL10_TEST_MASKS, len(df_te), "TEST", spec)
            elif spec["kind"] == "hybrid":
                val_df, val_block_lengths = _generate_hybrid(CELL10_VAL_MASKS, len(df_va), "VAL", spec)
                test_df, test_block_lengths = _generate_hybrid(CELL10_TEST_MASKS, len(df_te), "TEST", spec)
            else:
                raise RuntimeError(f"[Cell10.4e] Unknown variant kind: {spec['kind']}")

            generation_audit_rows.append(
                _validate_candidate_frame(val_df, CELL10_VAL_MASKS, len(df_va), "VAL", cid, val_block_lengths)
            )
            generation_audit_rows.append(
                _validate_candidate_frame(test_df, CELL10_TEST_MASKS, len(df_te), "TEST", cid, test_block_lengths)
            )

            val_path = os.path.join(OUT_PORT, f"candidate_{cid}_VAL.parquet")
            test_path = os.path.join(OUT_PORT, f"candidate_{cid}_TEST.parquet")
            meta_path = os.path.join(OUT_PORT, f"candidate_{cid}_fit_meta_v2_THESIS.json")

            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)

            meta = {
                "version": "cell10_4e_ota_pkt_rate_regime_variant_v2_THESIS",
                "generator": gen,
                "candidate_id": cid,
                "variant_name": spec.get("variant_name"),
                "kind": spec.get("kind"),
                "columns": [TARGET_COL],
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "test_used_for_fitting": False,
                "test_used_for_repair": False,
                "target_col": TARGET_COL,
                "train_active_n": int(train_active.sum()),
                "train_valid_active_n": int(valid_active.sum()),
                "train_profile": train_profile,
                "segments_n": int(len(segment_infos)),
                "segment_length_min": int(np.min([s["length"] for s in segment_infos])),
                "segment_length_max": int(np.max([s["length"] for s in segment_infos])),
                "segment_length_mean": float(np.mean([s["length"] for s in segment_infos])),
                "state_bins_n": int(n_states),
                "state_edges": [float(x) for x in edges.tolist()],
                "variant_spec": spec,
                "role": spec.get("role"),
                "copy_risk_note": (
                    "This candidate uses TRAIN-only packet-rate trajectory/state sampling. "
                    "If promoted to public artifacts, Q6/no-copy audits must be rerun."
                ),
            }

            meta["record_sha256"] = hashlib.sha256(
                json.dumps(meta, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

            _write_atomic_json(meta_path, meta)

            rec = _cell10_register_portfolio_candidate({
                "generator": gen,
                "candidate_id": cid,
                "columns": [TARGET_COL],
                "val_path": val_path,
                "test_path": test_path,
                "fit_meta_path": meta_path,
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "test_used_for_fitting": False,
                "test_used_for_repair": False,
                "role": spec.get("role", "ota_pkt_rate_regime_candidate"),
                "eligibility": {
                    "target_col": TARGET_COL,
                    "train_active_n": int(train_active.sum()),
                    "train_valid_active_n": int(valid_active.sum()),
                    "kind": spec.get("kind"),
                    "variant_name": spec.get("variant_name"),
                },
            })

            registered.append(cid)
            meta_rows.append(meta)

            log(
                "[Cell10.4e] Registered OTA pkt-rate candidate | "
                f"candidate_id={cid} | kind={spec['kind']} | "
                f"VAL={val_df.shape} | TEST={test_df.shape} | role={spec.get('role')}"
            )

        # --------------------------------------------------
        # 8) Persist portfolio audit/contract
        # --------------------------------------------------
        fit_audit_path = os.path.join(OUT_REP, "cell10_4e_ota_pkt_rate_regime_fit_audit_v2_THESIS.csv")
        gen_audit_path = os.path.join(OUT_REP, "cell10_4e_ota_pkt_rate_regime_generation_audit_v2_THESIS.csv")
        audit_path = os.path.join(OUT_REP, "cell10_4e_ota_pkt_rate_regime_audit_v2_THESIS.json")
        contract_path = os.path.join(CONTRACT_DIR, "cell10_4e_ota_pkt_rate_regime_contract_v2_THESIS.json")
        portfolio_meta_path = os.path.join(OUT_PORT, "candidate_ota_pkt_rate_regime_portfolio_meta_v2_THESIS.json")

        fit_audit_df = pd.DataFrame([{
            "generator": "ota_pkt_rate_regime",
            "target_col": TARGET_COL,
            "train_active_n": int(train_active.sum()),
            "train_valid_active_n": int(valid_active.sum()),
            "segments_n": int(len(segment_infos)),
            "segment_length_min": int(np.min([s["length"] for s in segment_infos])),
            "segment_length_max": int(np.max([s["length"] for s in segment_infos])),
            "segment_length_mean": float(np.mean([s["length"] for s in segment_infos])),
            "state_bins_n": int(n_states),
            "train_mean": train_profile["mean"],
            "train_std": train_profile["std"],
            "train_nonzero_rate": train_profile["nonzero_rate"],
            "train_transition_rate": train_profile["transition_rate"],
            "train_lag1": train_profile["lag1"],
            "variant_ids": "|".join(registered),
            "variants_n": int(len(registered)),
        }])

        gen_audit_df = pd.DataFrame(generation_audit_rows)

        fit_audit_df.to_csv(fit_audit_path, index=False)
        gen_audit_df.to_csv(gen_audit_path, index=False)

        portfolio_meta = {
            "version": "cell10_4e_ota_pkt_rate_regime_portfolio_v2_THESIS",
            "generator": "ota_pkt_rate_regime",
            "portfolio_role": "predefined_ota_packet_rate_regime_candidate_portfolio",
            "target_col": TARGET_COL,
            "registered_candidate_ids": list(registered),
            "variants_n": int(len(registered)),
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "test_used_for_selection": False,
            "train_active_n": int(train_active.sum()),
            "train_valid_active_n": int(valid_active.sum()),
            "train_profile": train_profile,
            "segments_n": int(len(segment_infos)),
            "variant_meta": meta_rows,
        }
        _write_atomic_json(portfolio_meta_path, portfolio_meta)

        audit = {
            "version": "cell10_4e_ota_pkt_rate_regime_candidate_v2_THESIS",
            "candidate_available": True,
            "target_col": TARGET_COL,
            "registered_candidate_ids": list(registered),
            "variants_n": int(len(registered)),
            "fit_audit_path": fit_audit_path,
            "generation_audit_path": gen_audit_path,
            "eligibility_audit_path": eligibility_path,
            "portfolio_meta_path": portfolio_meta_path,
            "fit_summary": {
                "train_active_n": int(train_active.sum()),
                "train_valid_active_n": int(valid_active.sum()),
                "segments_n": int(len(segment_infos)),
                "state_bins_n": int(n_states),
                "train_profile": train_profile,
            },
            "copy_risk_status": {
                "uses_train_block_replay_or_train_support_sampling": True,
                "public_release_safe_without_Q6_audit": False,
                "q6_no_copy_audit_required_if_selected": True,
            },
            "leakage_status": {
                "uses_train_values_for_fitting": True,
                "uses_val_values_for_fitting": False,
                "uses_test_values_for_fitting": False,
                "uses_val_masks_for_val_candidate": True,
                "uses_test_masks_for_test_candidate": True,
                "uses_test_real_values_for_generation": False,
                "uses_test_values_for_selection": False,
                "uses_test_values_for_repair": False,
                "uses_test_values_for_candidate_promotion": False,
                "materializes_test_candidate": True,
                "overwrites_final_artifact": False,
            },
            "paper_claim_status": (
                "OTA packet-rate regime candidates are predefined protocol-specific "
                "portfolio candidates for ota__pkt_total only. They are not final evidence "
                "unless selected later by the downstream VAL-only selector against A1, and "
                "remain subject to Q6/copy-risk checks before any public-release claim."
            ),
            "ts_unix": float(time.time()),
        }

        _write_atomic_json(audit_path, audit)
        _write_atomic_json(contract_path, audit)

        globals()["CELL10_4E_OTA_PKT_AUDIT"] = audit
        globals()["CELL10_4E_OTA_PKT_AUDIT_PATH"] = audit_path
        globals()["CELL10_4E_OTA_PKT_CONTRACT_PATH"] = contract_path

        if "RUN_META" in globals():
            RUN_META.setdefault("portfolio_candidate_audits", {})
            RUN_META["portfolio_candidate_audits"]["ota_pkt_rate_regime"] = audit

        log(f"[Cell10.4e] Saved eligibility audit: {eligibility_path}")
        log(f"[Cell10.4e] Saved fit audit: {fit_audit_path}")
        log(f"[Cell10.4e] Saved generation audit: {gen_audit_path}")
        log(f"[Cell10.4e] Saved portfolio meta: {portfolio_meta_path}")
        log(f"[Cell10.4e] Saved audit: {audit_path}")
        log(f"[Cell10.4e] Saved contract: {contract_path}")
        log(
            "[Cell10.4e] OTA packet-rate regime candidate portfolio complete | "
            f"variants={registered}"
        )

log("--- END: Cell 10.4e — OTA packet-rate regime candidate portfolio (v2-THESIS) ---")
