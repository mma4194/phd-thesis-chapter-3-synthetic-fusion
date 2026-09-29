# ==========================================================
# CELL 10.4d — OTA block-intensity replay variant portfolio — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Generate a predefined OTA paired block-replay variant portfolio for:
#     ota__pkt_total
#     ota__bytes_total
# - Preserve paired packet/byte temporal trajectories using contiguous TRAIN blocks.
# - Emit VAL and TEST-horizon candidate artifacts for downstream VAL-only selection.
#
# Scientific contract:
# - Fit TRAIN only.
# - Generate VAL and TEST synthetic candidates from TRAIN block library and masks.
# - Do not read real TEST values.
# - Do not tune on TEST.
# - Do not select anything here.
# - Not final evidence unless selected downstream on VAL against A1.
#
# Public-release caveat:
# - These variants replay contiguous TRAIN blocks. If any variant is selected for
#   public release, Q6/copy-risk/no-copy audits are mandatory.
# ==========================================================

log("--- START: Cell 10.4d — OTA block-intensity replay variant portfolio (v2-THESIS) ---")

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
    raise RuntimeError(f"[Cell10.4d] Missing prerequisites: {missing}. Run Cells 10.1–10.4c first.")

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
        raise RuntimeError(f"[Cell10.4d] Clean OTA variant generation forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.4d] Unexpected selection_split_policy. "
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

# ----------------------------------------------------------
# 1) Resolve targets without using VAL/TEST values
# ----------------------------------------------------------
EXPECTED_TARGET_COLS = ["ota__pkt_total", "ota__bytes_total"]

TARGET_COLS = [
    c for c in EXPECTED_TARGET_COLS
    if c in set(map(str, CELL10_PROTO_COLS))
    and c in df_tr.columns
    and c in df_va.columns
    and c in df_te.columns
]

eligibility_path = os.path.join(
    OUT_REP,
    "cell10_4d_ota_block_intensity_replay_variants_eligibility_v2_THESIS.csv",
)

eligibility_df = pd.DataFrame([
    {
        "col": c,
        "expected": True,
        "in_CELL10_PROTO_COLS": c in set(map(str, CELL10_PROTO_COLS)),
        "in_df_tr_schema": c in df_tr.columns,
        "in_df_va_schema": c in df_va.columns,
        "in_df_te_schema": c in df_te.columns,
        "selected_target": c in TARGET_COLS,
    }
    for c in EXPECTED_TARGET_COLS
])
eligibility_df.to_csv(eligibility_path, index=False)

# ----------------------------------------------------------
# 2) JSON helpers and unavailable path
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
        "ota_block_intensity_replay_variants",
        reason=reason,
        meta={
            "source": "Cell10.4d",
            "version": "cell10_4d_ota_block_intensity_replay_variants_v2_THESIS",
            "expected_cols": EXPECTED_TARGET_COLS,
            "found_cols": TARGET_COLS,
            "eligibility_audit_path": eligibility_path,
            **dict(extra_meta or {}),
        },
    )

    audit = {
        "version": "cell10_4d_ota_block_intensity_replay_variants_v2_THESIS",
        "candidate_available": False,
        "reason": reason,
        "target_cols": list(TARGET_COLS),
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
            "OTA block-replay variant portfolio unavailable because required schema "
            "or TRAIN-only support conditions were not satisfied."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(
        OUT_REP,
        "cell10_4d_ota_block_intensity_replay_variants_audit_v2_THESIS.json",
    )
    contract_path = os.path.join(
        CONTRACT_DIR,
        "cell10_4d_ota_block_intensity_replay_variants_contract_v2_THESIS.json",
    )

    _write_atomic_json(audit_path, audit)
    _write_atomic_json(contract_path, audit)

    globals()["CELL10_4D_OTA_VARIANTS_AUDIT"] = audit
    globals()["CELL10_4D_OTA_VARIANTS_AUDIT_PATH"] = audit_path
    globals()["CELL10_4D_OTA_VARIANTS_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["ota_block_intensity_replay_variants"] = audit

    log(f"[Cell10.4d] OTA block-replay variants unavailable: {reason}")
    log(f"[Cell10.4d] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.4d] Saved audit: {audit_path}")
    log(f"[Cell10.4d] Saved contract: {contract_path}")


if len(TARGET_COLS) < 2:
    _finalize_unavailable(
        reason="Required paired OTA aggregate target columns not found.",
        extra_meta={"expected_cols": EXPECTED_TARGET_COLS, "found_cols": TARGET_COLS},
    )

else:
    # ------------------------------------------------------
    # 3) TRAIN-only fit: contiguous OTA active segment library
    # ------------------------------------------------------
    def _train_ota_active_mask() -> np.ndarray:
        if "ota__obs_present" not in df_tr.columns:
            raise RuntimeError("[Cell10.4d] TRAIN missing ota__obs_present.")
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


    def _segment_stats(X: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            return {
                "mean_pkt": 0.0,
                "mean_bytes": 0.0,
                "std_pkt": 0.0,
                "std_bytes": 0.0,
                "p95_pkt": 0.0,
                "p95_bytes": 0.0,
            }

        return {
            "mean_pkt": float(np.mean(X[:, 0])),
            "mean_bytes": float(np.mean(X[:, 1])),
            "std_pkt": float(np.std(X[:, 0])),
            "std_bytes": float(np.std(X[:, 1])),
            "p95_pkt": float(np.quantile(X[:, 0], 0.95)),
            "p95_bytes": float(np.quantile(X[:, 1], 0.95)),
        }


    def _lag1_corr_2d(X: np.ndarray, j: int):
        x = np.asarray(X[:, j], dtype=np.float64)
        if x.size < 3:
            return np.nan
        a = x[:-1]
        b = x[1:]
        if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])


    def _make_intensity_bins(segment_infos: list, n_bins: int = 4) -> dict:
        if not segment_infos:
            return {}

        n_bins = int(max(1, n_bins))
        means = np.asarray(
            [
                float(b["stats"]["mean_pkt"]) + float(b["stats"]["mean_bytes"]) / 1000.0
                for b in segment_infos
            ],
            dtype=np.float64,
        )

        if len(np.unique(means)) <= 1:
            return {0: list(range(len(segment_infos)))}

        qs = np.quantile(means, np.linspace(0.0, 1.0, n_bins + 1))
        bins = {i: [] for i in range(n_bins)}

        for i, m in enumerate(means):
            b = int(np.searchsorted(qs, m, side="right") - 1)
            b = max(0, min(n_bins - 1, b))
            bins[b].append(i)

        return {int(k): list(v) for k, v in bins.items() if len(v) > 0}


    def _sample_block_length(
        rng_local: np.random.Generator,
        block_min: int,
        block_max: int,
        max_len_remaining: int,
        mode: str,
    ) -> int:
        L_hi = int(min(block_max, max_len_remaining))
        L_lo = int(min(block_min, L_hi))

        if L_hi <= 0:
            return 0
        if L_hi <= L_lo:
            return int(L_hi)

        mode = str(mode)

        if mode == "fixed_mid":
            return int((L_lo + L_hi) // 2)

        if mode == "uniform":
            return int(rng_local.integers(L_lo, L_hi + 1))

        if mode == "loguniform":
            lo = np.log(max(1, L_lo))
            hi = np.log(max(1, L_hi))
            L = int(round(np.exp(rng_local.uniform(lo, hi))))
            return int(min(max(L, L_lo), L_hi))

        if mode == "long_biased":
            u = rng_local.beta(4.0, 1.5)
            return int(round(L_lo + u * (L_hi - L_lo)))

        if mode == "short_biased":
            u = rng_local.beta(1.5, 4.0)
            return int(round(L_lo + u * (L_hi - L_lo)))

        return int(rng_local.integers(L_lo, L_hi + 1))


    try:
        train_active = _train_ota_active_mask()
        train_X_full = df_tr[TARGET_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        finite_rows = np.isfinite(train_X_full).all(axis=1)
        train_valid_active = train_active & finite_rows

        min_train_active = int(CFG.get("cell10_4d_ota_min_train_active_rows", 1000))
        if int(train_valid_active.sum()) < min_train_active:
            raise RuntimeError(
                f"Insufficient valid TRAIN OTA active rows: "
                f"{int(train_valid_active.sum())} < {min_train_active}"
            )

        train_X_full = np.maximum(train_X_full, 0.0)
        train_X_full = np.rint(train_X_full).astype(np.float32)

        raw_segments = _contiguous_segments_from_mask(train_valid_active)
        if not raw_segments:
            idx = np.flatnonzero(train_valid_active)
            if idx.size <= 0:
                raise RuntimeError("No valid TRAIN OTA active rows.")
            raw_segments = [(int(idx[0]), int(idx[-1]) + 1)]

        min_seg_len = int(CFG.get("cell10_4d_ota_min_segment_len", 30))
        segments = [(s, e) for s, e in raw_segments if (e - s) >= min_seg_len]
        if not segments:
            segments = raw_segments

        segment_infos = []
        for si, (s, e) in enumerate(segments):
            Xseg = train_X_full[s:e, :]
            segment_infos.append({
                "segment_id": int(si),
                "start": int(s),
                "end": int(e),
                "length": int(e - s),
                "stats": _segment_stats(Xseg),
            })

        seg_lengths = np.asarray([x["length"] for x in segment_infos], dtype=np.float64)
        if seg_lengths.size <= 0 or float(seg_lengths.sum()) <= 0:
            raise RuntimeError("No usable TRAIN OTA segments for variant block replay.")

        seg_probs_len = seg_lengths / float(seg_lengths.sum())
        intensity_bins = _make_intensity_bins(
            segment_infos,
            n_bins=int(CFG.get("cell10_4d_ota_intensity_bins", 4)),
        )

        train_valid_X = train_X_full[train_valid_active, :]
        train_positive_rate_any = float(np.mean((train_valid_X > 0).any(axis=1)))
        train_pair_corr = (
            float(np.corrcoef(train_valid_X[:, 0], train_valid_X[:, 1])[0, 1])
            if train_valid_X.shape[0] > 2
            and np.std(train_valid_X[:, 0]) > 1e-12
            and np.std(train_valid_X[:, 1]) > 1e-12
            else np.nan
        )

        train_lag1 = {
            TARGET_COLS[j]: _lag1_corr_2d(train_valid_X, j)
            for j in range(len(TARGET_COLS))
        }

    except Exception as e:
        _finalize_unavailable(
            reason=f"{type(e).__name__}: {e}",
            extra_meta={"targets": TARGET_COLS},
        )

    else:
        # --------------------------------------------------
        # 4) Predefined variant specs
        # --------------------------------------------------
        variant_specs = [
            {
                "name": "short",
                "candidate_id": "ota_block_intensity_replay_short_v2_THESIS",
                "block_min": int(CFG.get("cell10_4d_ota_short_block_min", 10)),
                "block_max": int(CFG.get("cell10_4d_ota_short_block_max", 120)),
                "length_mode": "short_biased",
                "segment_policy": "length_weighted",
                "description": "Short blocks to improve local transitions and reduce block instability.",
            },
            {
                "name": "medium",
                "candidate_id": "ota_block_intensity_replay_medium_v2_THESIS",
                "block_min": int(CFG.get("cell10_4d_ota_medium_block_min", 60)),
                "block_max": int(CFG.get("cell10_4d_ota_medium_block_max", 900)),
                "length_mode": "loguniform",
                "segment_policy": "length_weighted",
                "description": "Medium blocks balancing local morphology and distributional coverage.",
            },
            {
                "name": "long",
                "candidate_id": "ota_block_intensity_replay_long_v2_THESIS",
                "block_min": int(CFG.get("cell10_4d_ota_long_block_min", 300)),
                "block_max": int(CFG.get("cell10_4d_ota_long_block_max", 3600)),
                "length_mode": "long_biased",
                "segment_policy": "length_weighted",
                "description": "Long blocks to preserve long-range autocorrelation.",
            },
            {
                "name": "mixed",
                "candidate_id": "ota_block_intensity_replay_mixed_v2_THESIS",
                "block_min": int(CFG.get("cell10_4d_ota_mixed_block_min", 20)),
                "block_max": int(CFG.get("cell10_4d_ota_mixed_block_max", 2400)),
                "length_mode": "loguniform",
                "segment_policy": "mixed_length_intensity",
                "description": "Mixed segment/intensity replay for robust VAL block stability.",
            },
            {
                "name": "segment_conditioned",
                "candidate_id": "ota_block_intensity_replay_segment_conditioned_v2_THESIS",
                "block_min": int(CFG.get("cell10_4d_ota_segment_block_min", 30)),
                "block_max": int(CFG.get("cell10_4d_ota_segment_block_max", 1800)),
                "length_mode": "fixed_mid",
                "segment_policy": "round_robin_segments",
                "description": "Round-robin segment-conditioned replay to reduce overuse of one TRAIN segment.",
            },
        ]

        for spec in variant_specs:
            spec["block_min"] = max(1, int(spec["block_min"]))
            spec["block_max"] = max(spec["block_min"], int(spec["block_max"]))

        # --------------------------------------------------
        # 5) Generation helpers
        # --------------------------------------------------
        def _choose_segment_index(rng_local: np.random.Generator, spec: dict, produced_blocks: int) -> int:
            policy = str(spec["segment_policy"])

            if policy == "length_weighted":
                return int(rng_local.choice(np.arange(len(segment_infos)), p=seg_probs_len))

            if policy == "round_robin_segments":
                return int(produced_blocks % len(segment_infos))

            if policy == "mixed_length_intensity":
                if rng_local.random() < 0.5 or not intensity_bins:
                    return int(rng_local.choice(np.arange(len(segment_infos)), p=seg_probs_len))

                nonempty_bins = sorted([b for b, idxs in intensity_bins.items() if len(idxs) > 0])
                b = int(nonempty_bins[int(rng_local.integers(0, len(nonempty_bins)))])
                idxs = intensity_bins[b]
                return int(idxs[int(rng_local.integers(0, len(idxs)))])

            return int(rng_local.choice(np.arange(len(segment_infos)), p=seg_probs_len))


        def _sample_train_block(
            rng_local: np.random.Generator,
            spec: dict,
            max_len_remaining: int,
            produced_blocks: int,
        ) -> np.ndarray:
            max_len_remaining = int(max_len_remaining)
            if max_len_remaining <= 0:
                return np.zeros((0, len(TARGET_COLS)), dtype=np.float32)

            for _ in range(200):
                si = _choose_segment_index(rng_local, spec, produced_blocks)
                info = segment_infos[si]

                s = int(info["start"])
                e = int(info["end"])
                seg_len = int(e - s)
                if seg_len <= 0:
                    continue

                L_hi = min(int(spec["block_max"]), seg_len, max_len_remaining)
                L_lo = min(int(spec["block_min"]), L_hi)
                if L_hi < 1:
                    continue

                L = _sample_block_length(
                    rng_local,
                    block_min=max(1, L_lo),
                    block_max=max(1, L_hi),
                    max_len_remaining=max_len_remaining,
                    mode=str(spec["length_mode"]),
                )
                L = int(min(max(1, L), seg_len, max_len_remaining))

                start_hi = int(e - L)
                if start_hi < s:
                    continue

                start = int(rng_local.integers(s, start_hi + 1))
                block = train_X_full[start:start + L, :]

                if block.shape[0] > 0 and np.isfinite(block).all():
                    return block.astype(np.float32, copy=False)

            idx = np.flatnonzero(train_valid_active)
            take = min(max_len_remaining, max(1, int(spec["block_min"])))
            draw = rng_local.choice(idx, size=take, replace=True)
            return train_X_full[draw, :].astype(np.float32, copy=False)


        def _generate_split_for_variant(mask_dict: dict, N: int, split_name: str, spec: dict):
            N = int(N)
            if N <= 0:
                raise RuntimeError(f"[Cell10.4d] {split_name}/{spec['name']}: N must be positive.")

            active = _mask_for_tier(mask_dict, "ota", N, missing_policy="error")
            n_active = int(active.sum())

            out = {c: np.full(N, np.nan, dtype=np.float32) for c in TARGET_COLS}
            block_lengths_drawn = []

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_4d_ota_block_variant",
                    split_name,
                    spec["candidate_id"],
                    "|".join(TARGET_COLS),
                    seed,
                    base_seed=seed + 10047,
                )
            )

            if n_active > 0:
                chunks = []
                produced = 0
                produced_blocks = 0
                guard = 0

                while produced < n_active:
                    guard += 1
                    if guard > n_active + 10000:
                        raise RuntimeError(
                            f"[Cell10.4d] Generation guard exceeded for {split_name}/{spec['name']}."
                        )

                    block = _sample_train_block(rng_local, spec, n_active - produced, produced_blocks)
                    if block.shape[0] == 0:
                        continue

                    chunks.append(block)
                    block_lengths_drawn.append(int(block.shape[0]))
                    produced += int(block.shape[0])
                    produced_blocks += 1

                X = np.vstack(chunks)[:n_active, :].astype(np.float32, copy=False)

                if bool(CFG.get("cell10_4d_ota_enable_tiny_jitter", False)):
                    jitter_std = float(CFG.get("cell10_4d_ota_jitter_std", 0.003))
                    jitter = rng_local.normal(1.0, jitter_std, size=X.shape)
                    X = np.maximum(0.0, np.rint(X * jitter)).astype(np.float32)

                for j, c in enumerate(TARGET_COLS):
                    arr = np.full(N, np.nan, dtype=np.float32)
                    arr[active] = X[:, j]
                    out[c] = _postprocess_protocol_values(c, arr)

            df_out = pd.DataFrame(out)

            gen_rows = []
            for c in TARGET_COLS:
                x = pd.to_numeric(df_out[c], errors="coerce").to_numpy(dtype=np.float64)

                finite_active = int(np.isfinite(x[active]).sum())
                finite_inactive = int(np.isfinite(x[~active]).sum())

                if n_active > 0 and finite_active != n_active:
                    raise RuntimeError(
                        f"[Cell10.4d] {split_name}/{spec['name']}.{c}: active finite mismatch: "
                        f"finite_active={finite_active}, active_n={n_active}"
                    )

                if finite_inactive != 0:
                    raise RuntimeError(
                        f"[Cell10.4d] {split_name}/{spec['name']}.{c}: inactive rows contain finite values: "
                        f"{finite_inactive}"
                    )

                xa = x[active]

                gen_rows.append({
                    "split": split_name,
                    "variant_name": spec["name"],
                    "candidate_id": spec["candidate_id"],
                    "col": c,
                    "tier": "ota",
                    "N": int(N),
                    "active_n": int(n_active),
                    "active_rate": float(np.mean(active)) if active.size else np.nan,
                    "finite_active": int(finite_active),
                    "finite_inactive": int(finite_inactive),
                    "generated_mean_active": _safe_mean(xa),
                    "generated_var_active": _safe_var(xa),
                    "generated_nonzero_rate_active": _nonzero_rate(xa),
                    "generated_transition_rate_active": _transition_rate(xa),
                    "block_draws": int(len(block_lengths_drawn)),
                    "block_len_median": float(np.median(block_lengths_drawn)) if block_lengths_drawn else np.nan,
                    "block_len_p95": float(np.quantile(block_lengths_drawn, 0.95)) if block_lengths_drawn else np.nan,
                    "train_valid_active_n": int(train_valid_active.sum()),
                    "train_positive_rate_any": float(train_positive_rate_any),
                })

            return df_out, pd.DataFrame(gen_rows)

        # --------------------------------------------------
        # 6) Materialize and register all predefined variants
        # --------------------------------------------------
        registered = []
        variant_meta_rows = []
        all_generation_audits = []

        for spec in variant_specs:
            cid = str(spec["candidate_id"])
            gen = "ota_block_intensity_replay_variants"

            val_df, val_gen_audit = _generate_split_for_variant(
                CELL10_VAL_MASKS,
                len(df_va),
                "VAL",
                spec,
            )
            test_df, test_gen_audit = _generate_split_for_variant(
                CELL10_TEST_MASKS,
                len(df_te),
                "TEST",
                spec,
            )

            val_path = os.path.join(OUT_PORT, f"candidate_{cid}_VAL.parquet")
            test_path = os.path.join(OUT_PORT, f"candidate_{cid}_TEST.parquet")
            meta_path = os.path.join(OUT_PORT, f"candidate_{cid}_fit_meta_v2_THESIS.json")

            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)

            meta = {
                "generator": gen,
                "candidate_id": cid,
                "variant_name": str(spec["name"]),
                "columns": list(TARGET_COLS),
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "test_used_for_fitting": False,
                "test_used_for_repair": False,
                "train_active_n": int(train_active.sum()),
                "train_valid_active_n": int(train_valid_active.sum()),
                "usable_segments_n": int(len(segment_infos)),
                "segment_length_min": int(np.min(seg_lengths)) if len(seg_lengths) else 0,
                "segment_length_max": int(np.max(seg_lengths)) if len(seg_lengths) else 0,
                "segment_length_mean": float(np.mean(seg_lengths)) if len(seg_lengths) else None,
                "block_min": int(spec["block_min"]),
                "block_max": int(spec["block_max"]),
                "length_mode": str(spec["length_mode"]),
                "segment_policy": str(spec["segment_policy"]),
                "description": str(spec["description"]),
                "train_positive_rate_any": float(train_positive_rate_any),
                "train_pair_corr_pkt_bytes": train_pair_corr,
                "train_lag1": train_lag1,
                "val_active_n": int(np.asarray(CELL10_VAL_MASKS["ota"], dtype=bool).sum()),
                "test_active_n": int(np.asarray(CELL10_TEST_MASKS["ota"], dtype=bool).sum()),
                "role": "ota_temporal_autocorrelation_block_replay_candidate_variant",
                "copy_risk_note": (
                    "This candidate replays contiguous TRAIN blocks and must remain "
                    "subject to Q6/no-copy audits before public-release use."
                ),
            }

            meta["record_sha256"] = hashlib.sha256(
                json.dumps(meta, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

            _write_atomic_json(meta_path, meta)

            rec = _cell10_register_portfolio_candidate({
                "generator": gen,
                "candidate_id": cid,
                "columns": list(TARGET_COLS),
                "val_path": val_path,
                "test_path": test_path,
                "fit_meta_path": meta_path,
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "test_used_for_fitting": False,
                "test_used_for_repair": False,
                "role": "ota_temporal_autocorrelation_block_replay_candidate_variant",
                "eligibility": {
                    "targets": list(TARGET_COLS),
                    "train_active_n": int(train_active.sum()),
                    "train_valid_active_n": int(train_valid_active.sum()),
                    "usable_segments_n": int(len(segment_infos)),
                    "block_min": int(spec["block_min"]),
                    "block_max": int(spec["block_max"]),
                    "length_mode": str(spec["length_mode"]),
                    "segment_policy": str(spec["segment_policy"]),
                    "variant_name": str(spec["name"]),
                },
            })

            registered.append(cid)
            variant_meta_rows.append(meta)
            all_generation_audits.append(val_gen_audit)
            all_generation_audits.append(test_gen_audit)

            log(
                "[Cell10.4d] Registered OTA block variant | "
                f"candidate_id={cid} | variant={spec['name']} | "
                f"VAL={val_df.shape} | TEST={test_df.shape} | "
                f"block={spec['block_min']}-{spec['block_max']} | "
                f"length_mode={spec['length_mode']} | segment_policy={spec['segment_policy']}"
            )

        # --------------------------------------------------
        # 7) Persist portfolio audit/contract
        # --------------------------------------------------
        fit_audit_path = os.path.join(
            OUT_REP,
            "cell10_4d_ota_block_intensity_replay_variants_fit_audit_v2_THESIS.csv",
        )
        gen_audit_path = os.path.join(
            OUT_REP,
            "cell10_4d_ota_block_intensity_replay_variants_generation_audit_v2_THESIS.csv",
        )
        audit_path = os.path.join(
            OUT_REP,
            "cell10_4d_ota_block_intensity_replay_variants_audit_v2_THESIS.json",
        )
        contract_path = os.path.join(
            CONTRACT_DIR,
            "cell10_4d_ota_block_intensity_replay_variants_contract_v2_THESIS.json",
        )
        portfolio_meta_path = os.path.join(
            OUT_PORT,
            "candidate_ota_block_intensity_replay_variant_portfolio_meta_v2_THESIS.json",
        )

        fit_audit_df = pd.DataFrame([
            {
                "generator": "ota_block_intensity_replay_variants",
                "target_cols": "|".join(TARGET_COLS),
                "train_active_n": int(train_active.sum()),
                "train_valid_active_n": int(train_valid_active.sum()),
                "usable_segments_n": int(len(segment_infos)),
                "segment_length_min": int(np.min(seg_lengths)) if len(seg_lengths) else 0,
                "segment_length_max": int(np.max(seg_lengths)) if len(seg_lengths) else 0,
                "segment_length_mean": float(np.mean(seg_lengths)) if len(seg_lengths) else None,
                "train_positive_rate_any": float(train_positive_rate_any),
                "train_pair_corr_pkt_bytes": train_pair_corr,
                "variant_ids": "|".join(registered),
                "variants_n": int(len(registered)),
            }
        ])

        gen_audit_df = pd.concat(all_generation_audits, axis=0, ignore_index=True)

        fit_audit_df.to_csv(fit_audit_path, index=False)
        gen_audit_df.to_csv(gen_audit_path, index=False)

        portfolio_meta = {
            "version": "cell10_4d_ota_block_intensity_replay_variants_v2_THESIS",
            "generator": "ota_block_intensity_replay_variants",
            "portfolio_role": "predefined_ota_temporal_block_replay_variant_portfolio",
            "columns": list(TARGET_COLS),
            "registered_candidate_ids": list(registered),
            "variants_n": int(len(registered)),
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "test_used_for_selection": False,
            "train_active_n": int(train_active.sum()),
            "train_valid_active_n": int(train_valid_active.sum()),
            "segments_n": int(len(segment_infos)),
            "variant_meta": variant_meta_rows,
        }
        _write_atomic_json(portfolio_meta_path, portfolio_meta)

        audit = {
            "version": "cell10_4d_ota_block_intensity_replay_variants_v2_THESIS",
            "candidate_available": True,
            "registered_candidate_ids": list(registered),
            "variants_n": int(len(registered)),
            "target_cols": list(TARGET_COLS),
            "fit_audit_path": fit_audit_path,
            "generation_audit_path": gen_audit_path,
            "eligibility_audit_path": eligibility_path,
            "portfolio_meta_path": portfolio_meta_path,
            "fit_summary": {
                "train_active_n": int(train_active.sum()),
                "train_valid_active_n": int(train_valid_active.sum()),
                "usable_segments_n": int(len(segment_infos)),
                "segment_length_mean": float(np.mean(seg_lengths)) if len(seg_lengths) else None,
                "train_positive_rate_any": float(train_positive_rate_any),
                "train_pair_corr_pkt_bytes": train_pair_corr,
            },
            "copy_risk_status": {
                "uses_contiguous_train_block_replay": True,
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
                "OTA block-intensity replay variants are predefined protocol-specific "
                "portfolio candidates. They are not final evidence unless selected later "
                "by the downstream VAL-only selector against A1, and remain subject to "
                "Q6/copy-risk checks before any public-release claim."
            ),
            "ts_unix": float(time.time()),
        }

        _write_atomic_json(audit_path, audit)
        _write_atomic_json(contract_path, audit)

        globals()["CELL10_4D_OTA_VARIANTS_AUDIT"] = audit
        globals()["CELL10_4D_OTA_VARIANTS_AUDIT_PATH"] = audit_path
        globals()["CELL10_4D_OTA_VARIANTS_CONTRACT_PATH"] = contract_path

        if "RUN_META" in globals():
            RUN_META.setdefault("portfolio_candidate_audits", {})
            RUN_META["portfolio_candidate_audits"]["ota_block_intensity_replay_variants"] = audit

        log(f"[Cell10.4d] Saved eligibility audit: {eligibility_path}")
        log(f"[Cell10.4d] Saved fit audit: {fit_audit_path}")
        log(f"[Cell10.4d] Saved generation audit: {gen_audit_path}")
        log(f"[Cell10.4d] Saved portfolio meta: {portfolio_meta_path}")
        log(f"[Cell10.4d] Saved audit: {audit_path}")
        log(f"[Cell10.4d] Saved contract: {contract_path}")
        log(
            "[Cell10.4d] OTA block-intensity replay variant portfolio complete | "
            f"variants={registered}"
        )

log("--- END: Cell 10.4d — OTA block-intensity replay variant portfolio (v2-THESIS) ---")