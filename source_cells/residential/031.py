# ==========================================================
# CELL 10.4c — OTA paired block-intensity replay candidate — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Add a TRAIN-fitted OTA temporal candidate for:
#     ota__pkt_total
#     ota__bytes_total
# - Preserve paired packet/byte temporal trajectories by replaying contiguous
#   TRAIN active blocks.
# - Emit VAL and TEST-horizon candidate artifacts for downstream VAL-only selection.
#
# Scientific contract:
# - Fit TRAIN only.
# - Generate VAL and TEST synthetic candidates from TRAIN block library and masks.
# - Do not read real TEST values.
# - Do not tune on TEST.
# - Do not select anything in this cell.
# - Not final evidence unless selected downstream on VAL against A1.
#
# Caveat:
# - This candidate replays TRAIN contiguous blocks, so it is suitable for internal
#   scientific candidate evaluation but must remain subject to Q6/copy-risk audits
#   before any public-release claim.
# ==========================================================

log("--- START: Cell 10.4c — OTA paired block-intensity replay candidate (v2-THESIS) ---")

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
    raise RuntimeError(f"[Cell10.4c] Missing prerequisites: {missing}. Run Cells 10.1–10.4b first.")

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
        raise RuntimeError(f"[Cell10.4c] Clean OTA block replay forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.4c] Unexpected selection_split_policy. "
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
# 1) Resolve target columns without using VAL/TEST values
# ----------------------------------------------------------
EXPECTED_TARGET_COLS = ["ota__pkt_total", "ota__bytes_total"]

TARGET_COLS = [
    c for c in EXPECTED_TARGET_COLS
    if c in set(map(str, CELL10_PROTO_COLS))
    and c in df_tr.columns
    and c in df_va.columns
    and c in df_te.columns
]

eligibility_path = os.path.join(OUT_REP, "cell10_4c_ota_block_intensity_replay_eligibility_v2_THESIS.csv")

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
# 2) Unavailable helper
# ----------------------------------------------------------
def _write_atomic_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    os.replace(tmp, path)


def _finalize_unavailable(reason: str, extra_meta: dict = None):
    _cell10_register_unavailable_backend(
        "ota_block_intensity_replay",
        reason=reason,
        meta={
            "source": "Cell10.4c",
            "version": "cell10_4c_ota_block_intensity_replay_v2_THESIS",
            "expected_cols": EXPECTED_TARGET_COLS,
            "found_cols": TARGET_COLS,
            "eligibility_audit_path": eligibility_path,
            **dict(extra_meta or {}),
        },
    )

    audit = {
        "version": "cell10_4c_ota_block_intensity_replay_candidate_v2_THESIS",
        "candidate_available": False,
        "reason": reason,
        "target_cols": list(TARGET_COLS),
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
            "OTA block-intensity replay candidate unavailable because required schema "
            "or TRAIN-only support conditions were not satisfied."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_4c_ota_block_intensity_replay_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_4c_ota_block_intensity_replay_contract_v2_THESIS.json")

    _write_atomic_json(audit_path, audit)
    _write_atomic_json(contract_path, audit)

    globals()["CELL10_4C_OTA_BLOCK_AUDIT"] = audit
    globals()["CELL10_4C_OTA_BLOCK_AUDIT_PATH"] = audit_path
    globals()["CELL10_4C_OTA_BLOCK_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["ota_block_intensity_replay"] = audit

    log(f"[Cell10.4c] OTA block-intensity replay unavailable: {reason}")
    log(f"[Cell10.4c] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.4c] Saved audit: {audit_path}")
    log(f"[Cell10.4c] Saved contract: {contract_path}")


if len(TARGET_COLS) < 2:
    _finalize_unavailable(
        reason="Required paired OTA aggregate target columns not found.",
        extra_meta={"expected_cols": EXPECTED_TARGET_COLS, "found_cols": TARGET_COLS},
    )

else:
    # ------------------------------------------------------
    # 3) Helpers
    # ------------------------------------------------------
    def _stable_seed_from_parts(*parts, base_seed: int = 1337) -> int:
        payload = "::".join(str(p) for p in parts)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int((int(h[:12], 16) + int(base_seed)) % (2**32 - 1))


    def _train_ota_active_mask() -> np.ndarray:
        if "ota__obs_present" not in df_tr.columns:
            raise RuntimeError("[Cell10.4c] TRAIN missing ota__obs_present.")
        v = pd.to_numeric(df_tr["ota__obs_present"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        return v > 0.5


    def _contiguous_segments_from_mask(mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size == 0:
            return []

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


    def _lag1_corr_2d(X: np.ndarray, j: int):
        x = np.asarray(X[:, j], dtype=np.float64)
        if x.size < 3:
            return np.nan
        a = x[:-1]
        b = x[1:]
        if np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])


    # ------------------------------------------------------
    # 4) TRAIN-only fit: contiguous active block library
    # ------------------------------------------------------
    try:
        train_active = _train_ota_active_mask()
        train_X_full = df_tr[TARGET_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        finite_rows = np.isfinite(train_X_full).all(axis=1)
        train_valid_active = train_active & finite_rows

        min_train_active = int(CFG.get("cell10_4c_ota_min_train_active_rows", 1000))
        if int(train_valid_active.sum()) < min_train_active:
            raise RuntimeError(
                f"Insufficient valid TRAIN OTA active rows: "
                f"{int(train_valid_active.sum())} < {min_train_active}"
            )

        train_X_full = np.maximum(train_X_full, 0.0)
        train_X_full = np.rint(train_X_full).astype(np.float32)

        active_segments = _contiguous_segments_from_mask(train_valid_active)

        block_min = int(CFG.get("cell10_4c_ota_block_min", 30))
        block_max = int(CFG.get("cell10_4c_ota_block_max", 1800))

        if block_min <= 0:
            raise RuntimeError(f"Invalid block_min={block_min}.")
        if block_max < block_min:
            block_max = block_min

        usable_segments = [
            (s, e)
            for s, e in active_segments
            if int(e - s) >= block_min
        ]

        if not usable_segments:
            valid_idx = np.flatnonzero(train_valid_active)
            if valid_idx.size <= 0:
                raise RuntimeError("No TRAIN valid OTA active rows after filtering.")
            usable_segments = [(int(valid_idx[0]), int(valid_idx[-1]) + 1)]

        seg_lengths = np.asarray([e - s for s, e in usable_segments], dtype=np.float64)
        if seg_lengths.size <= 0 or float(seg_lengths.sum()) <= 0:
            raise RuntimeError("No usable TRAIN OTA segments for block replay.")

        seg_probs = seg_lengths / float(seg_lengths.sum())

        train_valid_X = train_X_full[train_valid_active, :]
        train_positive_rate_any = float(np.mean((train_valid_X > 0).any(axis=1)))
        train_pair_corr = (
            float(np.corrcoef(train_valid_X[:, 0], train_valid_X[:, 1])[0, 1])
            if train_valid_X.shape[0] > 2
            and np.std(train_valid_X[:, 0]) > 1e-12
            and np.std(train_valid_X[:, 1]) > 1e-12
            else np.nan
        )

        fit_model_meta = {
            "target_cols": list(TARGET_COLS),
            "train_active_n": int(train_active.sum()),
            "train_valid_active_n": int(train_valid_active.sum()),
            "usable_segments_n": int(len(usable_segments)),
            "usable_segment_length_min": int(np.min(seg_lengths)),
            "usable_segment_length_max": int(np.max(seg_lengths)),
            "usable_segment_length_mean": float(np.mean(seg_lengths)),
            "block_min": int(block_min),
            "block_max": int(block_max),
            "loguniform_blocks": bool(CFG.get("cell10_4c_ota_loguniform_blocks", True)),
            "tiny_jitter_enabled": bool(CFG.get("cell10_4c_ota_enable_tiny_jitter", False)),
            "train_positive_rate_any": float(train_positive_rate_any),
            "train_pair_corr_pkt_bytes": train_pair_corr,
            "train_lag1": {
                TARGET_COLS[j]: _lag1_corr_2d(train_valid_X, j)
                for j in range(len(TARGET_COLS))
            },
            "copy_risk_note": (
                "This candidate replays contiguous TRAIN blocks. It is valid as an "
                "internal scientific candidate but must remain subject to Q6/no-copy "
                "audits before public release."
            ),
        }

    except Exception as e:
        _finalize_unavailable(
            reason=f"{type(e).__name__}: {e}",
            extra_meta={"targets": TARGET_COLS},
        )

    else:
        # --------------------------------------------------
        # 5) Generate VAL/TEST candidates from block library and masks
        # --------------------------------------------------
        def _sample_train_block(max_len_remaining: int, rng_local: np.random.Generator) -> np.ndarray:
            max_len_remaining = int(max_len_remaining)
            if max_len_remaining <= 0:
                return np.zeros((0, len(TARGET_COLS)), dtype=np.float32)

            for _ in range(100):
                si = int(rng_local.choice(np.arange(len(usable_segments)), p=seg_probs))
                s, e = usable_segments[si]
                seg_len = int(e - s)
                if seg_len <= 0:
                    continue

                L_hi = min(block_max, seg_len, max_len_remaining)
                L_lo = min(block_min, L_hi)

                if L_hi < 1:
                    continue

                if L_hi <= L_lo:
                    L = int(L_hi)
                else:
                    if bool(CFG.get("cell10_4c_ota_loguniform_blocks", True)):
                        lo = np.log(max(1, L_lo))
                        hi = np.log(max(1, L_hi))
                        L = int(round(np.exp(rng_local.uniform(lo, hi))))
                        L = min(max(L, L_lo), L_hi)
                    else:
                        L = int(rng_local.integers(L_lo, L_hi + 1))

                start_hi = int(e - L)
                if start_hi < s:
                    continue

                start = int(rng_local.integers(s, start_hi + 1))
                block = train_X_full[start:start + L, :]

                if block.shape[0] > 0 and np.isfinite(block).all():
                    return block.astype(np.float32, copy=False)

            # Fallback: sample valid active rows if repeated segment sampling fails.
            idx = np.flatnonzero(train_valid_active)
            take = min(max_len_remaining, max(1, block_min))
            draw = rng_local.choice(idx, size=take, replace=True)
            return train_X_full[draw, :].astype(np.float32, copy=False)


        def _generate_split(mask_dict: dict, N: int, split_name: str):
            N = int(N)
            if N <= 0:
                raise RuntimeError(f"[Cell10.4c] {split_name}: N must be positive.")

            active = _mask_for_tier(mask_dict, "ota", N, missing_policy="error")
            n_active = int(active.sum())

            out = {c: np.full(N, np.nan, dtype=np.float32) for c in TARGET_COLS}
            block_lengths_drawn = []

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_4c_ota_block_intensity_replay",
                    split_name,
                    "|".join(TARGET_COLS),
                    seed,
                    base_seed=seed + 10046,
                )
            )

            if n_active > 0:
                chunks = []
                produced = 0
                guard = 0

                while produced < n_active:
                    guard += 1
                    if guard > n_active + 10000:
                        raise RuntimeError(f"[Cell10.4c] Generation guard exceeded for {split_name}.")

                    block = _sample_train_block(n_active - produced, rng_local)
                    if block.shape[0] == 0:
                        continue

                    chunks.append(block)
                    block_lengths_drawn.append(int(block.shape[0]))
                    produced += int(block.shape[0])

                X = np.vstack(chunks)[:n_active, :].astype(np.float32, copy=False)

                if bool(CFG.get("cell10_4c_ota_enable_tiny_jitter", False)):
                    jitter_std = float(CFG.get("cell10_4c_ota_jitter_std", 0.005))
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
                        f"[Cell10.4c] {split_name}.{c}: active finite mismatch: "
                        f"finite_active={finite_active}, active_n={n_active}"
                    )

                if finite_inactive != 0:
                    raise RuntimeError(
                        f"[Cell10.4c] {split_name}.{c}: inactive rows contain finite values: {finite_inactive}"
                    )

                xa = x[active]

                gen_rows.append({
                    "split": split_name,
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
                    "train_valid_active_n": int(fit_model_meta["train_valid_active_n"]),
                    "train_positive_rate_any": float(fit_model_meta["train_positive_rate_any"]),
                })

            return df_out, pd.DataFrame(gen_rows)


        val_df, val_gen_audit = _generate_split(CELL10_VAL_MASKS, len(df_va), "VAL")
        test_df, test_gen_audit = _generate_split(CELL10_TEST_MASKS, len(df_te), "TEST")

        # --------------------------------------------------
        # 6) Persist candidate artifacts
        # --------------------------------------------------
        val_path = os.path.join(OUT_PORT, "candidate_ota_block_intensity_replay_VAL.parquet")
        test_path = os.path.join(OUT_PORT, "candidate_ota_block_intensity_replay_TEST.parquet")
        meta_path = os.path.join(OUT_PORT, "candidate_ota_block_intensity_replay_fit_meta_v2_THESIS.json")

        fit_audit_path = os.path.join(OUT_REP, "cell10_4c_ota_block_intensity_replay_fit_audit_v2_THESIS.csv")
        gen_audit_path = os.path.join(OUT_REP, "cell10_4c_ota_block_intensity_replay_generation_audit_v2_THESIS.csv")
        audit_path = os.path.join(OUT_REP, "cell10_4c_ota_block_intensity_replay_audit_v2_THESIS.json")
        contract_path = os.path.join(CONTRACT_DIR, "cell10_4c_ota_block_intensity_replay_contract_v2_THESIS.json")

        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        _write_atomic_json(meta_path, fit_model_meta)

        fit_audit_df = pd.DataFrame([{
            "generator": "ota_block_intensity_replay",
            "candidate_id": "ota_block_intensity_replay_v2_THESIS",
            "target_cols": "|".join(TARGET_COLS),
            "train_active_n": int(fit_model_meta["train_active_n"]),
            "train_valid_active_n": int(fit_model_meta["train_valid_active_n"]),
            "usable_segments_n": int(fit_model_meta["usable_segments_n"]),
            "usable_segment_length_min": int(fit_model_meta["usable_segment_length_min"]),
            "usable_segment_length_max": int(fit_model_meta["usable_segment_length_max"]),
            "usable_segment_length_mean": float(fit_model_meta["usable_segment_length_mean"]),
            "block_min": int(fit_model_meta["block_min"]),
            "block_max": int(fit_model_meta["block_max"]),
            "train_positive_rate_any": float(fit_model_meta["train_positive_rate_any"]),
            "train_pair_corr_pkt_bytes": fit_model_meta["train_pair_corr_pkt_bytes"],
        }])

        gen_audit_df = pd.concat([val_gen_audit, test_gen_audit], axis=0, ignore_index=True)

        fit_audit_df.to_csv(fit_audit_path, index=False)
        gen_audit_df.to_csv(gen_audit_path, index=False)

        rec = _cell10_register_portfolio_candidate({
            "generator": "ota_block_intensity_replay",
            "candidate_id": "ota_block_intensity_replay_v2_THESIS",
            "columns": list(TARGET_COLS),
            "val_path": val_path,
            "test_path": test_path,
            "fit_meta_path": meta_path,
            "fit_audit_path": fit_audit_path,
            "generation_audit_path": gen_audit_path,
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "test_used_for_selection": False,
            "test_used_for_fitting": False,
            "test_used_for_repair": False,
            "role": "ota_temporal_autocorrelation_block_replay_candidate",
            "eligibility": {
                "targets": list(TARGET_COLS),
                "train_active_n": int(fit_model_meta["train_active_n"]),
                "train_valid_active_n": int(fit_model_meta["train_valid_active_n"]),
                "usable_segments_n": int(fit_model_meta["usable_segments_n"]),
                "block_min": int(block_min),
                "block_max": int(block_max),
                "eligibility_audit_path": eligibility_path,
            },
        })

        audit = {
            "version": "cell10_4c_ota_block_intensity_replay_candidate_v2_THESIS",
            "candidate_available": True,
            "registered_candidate": rec,
            "target_cols": list(TARGET_COLS),
            "eligible_cols_n": int(len(TARGET_COLS)),
            "val_path": val_path,
            "test_path": test_path,
            "fit_meta_path": meta_path,
            "fit_audit_path": fit_audit_path,
            "generation_audit_path": gen_audit_path,
            "eligibility_audit_path": eligibility_path,
            "val_shape": [int(val_df.shape[0]), int(val_df.shape[1])],
            "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
            "fit_summary": {
                "train_active_n": int(fit_model_meta["train_active_n"]),
                "train_valid_active_n": int(fit_model_meta["train_valid_active_n"]),
                "usable_segments_n": int(fit_model_meta["usable_segments_n"]),
                "usable_segment_length_mean": float(fit_model_meta["usable_segment_length_mean"]),
                "block_min": int(block_min),
                "block_max": int(block_max),
                "train_positive_rate_any": float(fit_model_meta["train_positive_rate_any"]),
                "train_pair_corr_pkt_bytes": fit_model_meta["train_pair_corr_pkt_bytes"],
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
                "OTA block-intensity replay is an active protocol-specific portfolio "
                "candidate for paired OTA aggregate temporal trajectories. It is not "
                "final evidence unless selected later by the downstream VAL-only selector "
                "against A1, and it remains subject to Q6/copy-risk checks before any "
                "public-release claim."
            ),
            "ts_unix": float(time.time()),
        }

        _write_atomic_json(audit_path, audit)
        _write_atomic_json(contract_path, audit)

        globals()["CELL10_4C_OTA_BLOCK_AUDIT"] = audit
        globals()["CELL10_4C_OTA_BLOCK_AUDIT_PATH"] = audit_path
        globals()["CELL10_4C_OTA_BLOCK_CONTRACT_PATH"] = contract_path

        if "RUN_META" in globals():
            RUN_META.setdefault("portfolio_candidate_audits", {})
            RUN_META["portfolio_candidate_audits"]["ota_block_intensity_replay"] = audit

        log(
            "[Cell10.4c] Registered OTA block-intensity replay candidate | "
            f"cols={TARGET_COLS} | VAL={val_df.shape} | TEST={test_df.shape} | "
            f"train_valid_active_n={int(fit_model_meta['train_valid_active_n'])} | "
            f"segments={int(fit_model_meta['usable_segments_n'])} | "
            f"block_min={block_min} | block_max={block_max}"
        )
        log(f"[Cell10.4c] Saved eligibility audit: {eligibility_path}")
        log(f"[Cell10.4c] Saved fit audit: {fit_audit_path}")
        log(f"[Cell10.4c] Saved generation audit: {gen_audit_path}")
        log(f"[Cell10.4c] Saved audit: {audit_path}")
        log(f"[Cell10.4c] Saved contract: {contract_path}")

log("--- END: Cell 10.4c — OTA paired block-intensity replay candidate (v2-THESIS) ---")