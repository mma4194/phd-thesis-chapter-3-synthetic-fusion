# ==========================================================
# CELL 10.4b — OTA burst/run-length active-window candidate — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Add a protocol-specific candidate for OTA aggregate totals:
#     ota__pkt_total, ota__bytes_total
# - Preserve TRAIN active-window event/no-event run lengths.
# - Preserve joint pkt/bytes positive magnitudes by sampling TRAIN positive pairs.
# - Emit VAL and TEST-horizon candidate artifacts for downstream VAL-only selection.
#
# Scientific contract:
# - Fit TRAIN only.
# - Generate VAL and TEST synthetic candidates from fitted TRAIN process and masks.
# - Do not read real TEST values.
# - Do not select here.
# - Not final evidence unless selected downstream on VAL against A1.
# ==========================================================

log("--- START: Cell 10.4b — OTA burst/run-length active-window candidate (v2-THESIS) ---")

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
    "_safe_mean", "_nonzero_rate", "_transition_rate",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.4b] Missing prerequisites: {missing}. Run Cells 10.1–10.4 first.")

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
        raise RuntimeError(f"[Cell10.4b] Clean OTA candidate generation forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.4b] Unexpected selection_split_policy. "
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
# 1) Resolve target columns without reading VAL/TEST values
# ----------------------------------------------------------
EXPECTED_TARGET_COLS = ["ota__pkt_total", "ota__bytes_total"]

TARGET_COLS = [
    c for c in EXPECTED_TARGET_COLS
    if c in set(map(str, CELL10_PROTO_COLS))
    and c in df_tr.columns
    and c in df_va.columns
    and c in df_te.columns
]

eligibility_path = os.path.join(OUT_REP, "cell10_4b_ota_burst_runlength_eligibility_v2_THESIS.csv")

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
# 2) Register unavailable if no target columns
# ----------------------------------------------------------
if len(TARGET_COLS) == 0:
    reason = "No OTA aggregate target columns found in protocol schema."

    _cell10_register_unavailable_backend(
        "ota_burst_runlength_active_window",
        reason=reason,
        meta={
            "source": "Cell10.4b",
            "version": "cell10_4b_ota_burst_runlength_v2_THESIS",
            "expected_cols": EXPECTED_TARGET_COLS,
            "eligibility_audit_path": eligibility_path,
        },
    )

    audit = {
        "version": "cell10_4b_ota_burst_runlength_candidate_v2_THESIS",
        "candidate_available": False,
        "reason": reason,
        "target_cols": [],
        "eligibility_audit_path": eligibility_path,
        "leakage_status": {
            "uses_train_values_for_fitting": True,
            "uses_val_values_for_fitting": False,
            "uses_test_values_for_fitting": False,
            "uses_test_values_for_selection": False,
            "uses_test_values_for_repair": False,
            "uses_test_values_for_candidate_promotion": False,
            "materializes_test_candidate": False,
            "overwrites_final_artifact": False,
        },
        "paper_claim_status": (
            "OTA burst/run-length candidate unavailable because required OTA aggregate "
            "target columns were absent from the current protocol schema."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_4b_ota_burst_runlength_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_4b_ota_burst_runlength_contract_v2_THESIS.json")

    for path, obj in [(audit_path, audit), (contract_path, audit)]:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        os.replace(tmp, path)

    globals()["CELL10_4B_OTA_AUDIT"] = audit
    globals()["CELL10_4B_OTA_AUDIT_PATH"] = audit_path
    globals()["CELL10_4B_OTA_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["ota_burst_runlength_active_window"] = audit

    log("[Cell10.4b] No OTA aggregate target columns found; registered backend as unavailable.")
    log(f"[Cell10.4b] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.4b] Saved audit: {audit_path}")
    log(f"[Cell10.4b] Saved contract: {contract_path}")

else:
    # ------------------------------------------------------
    # 3) Helpers
    # ------------------------------------------------------
    def _stable_seed_from_parts(*parts, base_seed: int = 1337) -> int:
        payload = "::".join(str(p) for p in parts)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int((int(h[:12], 16) + int(base_seed)) % (2**32 - 1))


    def _active_mask_train_ota() -> np.ndarray:
        if "ota__obs_present" not in df_tr.columns:
            raise RuntimeError("[Cell10.4b] TRAIN missing ota__obs_present.")
        v = pd.to_numeric(df_tr["ota__obs_present"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        return v > 0.5


    def _runs_binary(b):
        b = np.asarray(b, dtype=np.int8).reshape(-1)
        if b.size == 0:
            return []

        out = []
        start = 0
        cur = int(b[0])

        for i in range(1, b.size + 1):
            if i == b.size or int(b[i]) != cur:
                out.append((int(cur), int(i - start)))
                if i < b.size:
                    start = i
                    cur = int(b[i])

        return out


    def _sample_binary_runs(model: dict, n: int, rng_local: np.random.Generator) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.int8)

        lens_by_state = {
            0: np.asarray(model.get("lens_0", [1]), dtype=np.int64),
            1: np.asarray(model.get("lens_1", [1]), dtype=np.int64),
        }

        if lens_by_state[0].size == 0:
            lens_by_state[0] = np.asarray([1], dtype=np.int64)
        if lens_by_state[1].size == 0:
            lens_by_state[1] = np.asarray([1], dtype=np.int64)

        p_start_one = float(np.clip(float(model.get("p_start_one", 0.0)), 0.0, 1.0))
        state = int(rng_local.random() < p_start_one)

        out = np.zeros(n, dtype=np.int8)
        pos = 0

        while pos < n:
            pool = lens_by_state[state]
            L = int(pool[int(rng_local.integers(0, pool.size))])
            L = max(1, L)

            take = min(L, n - pos)
            out[pos:pos + take] = state
            pos += take
            state = 1 - state

        return out


    # ------------------------------------------------------
    # 4) TRAIN-only fit
    # ------------------------------------------------------
    def _fit_ota_pair_model():
        active = _active_mask_train_ota()

        min_active = int(CFG.get("cell10_4b_ota_min_active_rows", 1000))
        if int(active.sum()) < min_active:
            raise RuntimeError(
                f"Insufficient TRAIN OTA active rows: {int(active.sum())} < {min_active}"
            )

        train_arrays = {}
        for c in TARGET_COLS:
            x = pd.to_numeric(df_tr.loc[active, c], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            x = _postprocess_protocol_values(c, x)
            train_arrays[c] = np.where(np.isfinite(x), x, 0.0).astype(np.float32)

        stacked = np.column_stack([train_arrays[c] for c in TARGET_COLS]).astype(np.float32, copy=False)

        # Event if any target aggregate is positive.
        event = (np.nan_to_num(stacked, nan=0.0) > 0).any(axis=1).astype(np.int8)

        runs = _runs_binary(event)
        if not runs:
            raise RuntimeError("[Cell10.4b] TRAIN OTA event sequence has no runs.")

        lens_0 = np.asarray([L for s, L in runs if int(s) == 0], dtype=np.int64)
        lens_1 = np.asarray([L for s, L in runs if int(s) == 1], dtype=np.int64)

        if lens_0.size == 0:
            lens_0 = np.asarray([1], dtype=np.int64)
        if lens_1.size == 0:
            lens_1 = np.asarray([1], dtype=np.int64)

        start_states = np.asarray([s for s, _ in runs], dtype=np.int8)
        p_start_one = float(np.mean(start_states == 1)) if start_states.size else float(np.mean(event))

        pos_idx = np.flatnonzero(event > 0)
        if pos_idx.size == 0:
            raise RuntimeError("[Cell10.4b] TRAIN OTA aggregates have no positive events.")

        positive_pairs = stacked[pos_idx, :].astype(np.float32)

        keep = (np.nan_to_num(positive_pairs, nan=0.0) > 0).any(axis=1)
        positive_pairs = positive_pairs[keep]

        if positive_pairs.shape[0] == 0:
            raise RuntimeError("[Cell10.4b] No usable positive OTA aggregate rows after filtering.")

        return {
            "target_cols": list(TARGET_COLS),
            "train_active_n": int(active.sum()),
            "train_event_rate": float(np.mean(event > 0)),
            "train_positive_rows_n": int(positive_pairs.shape[0]),
            "train_event_runs_n": int(len(runs)),
            "lens_0": lens_0.astype(int).tolist(),
            "lens_1": lens_1.astype(int).tolist(),
            "p_start_one": float(p_start_one),
            "zero_run_median": float(np.median(lens_0)),
            "one_run_median": float(np.median(lens_1)),
            "zero_run_p95": float(np.quantile(lens_0, 0.95)),
            "one_run_p95": float(np.quantile(lens_1, 0.95)),
            "positive_pairs": positive_pairs,
            "positive_pair_mean": {
                c: float(np.mean(positive_pairs[:, j]))
                for j, c in enumerate(TARGET_COLS)
            },
            "positive_pair_q95": {
                c: float(np.quantile(positive_pairs[:, j], 0.95))
                for j, c in enumerate(TARGET_COLS)
            },
        }


    try:
        model = _fit_ota_pair_model()
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"

        _cell10_register_unavailable_backend(
            "ota_burst_runlength_active_window",
            reason=reason,
            meta={
                "source": "Cell10.4b",
                "version": "cell10_4b_ota_burst_runlength_v2_THESIS",
                "targets": list(TARGET_COLS),
                "eligibility_audit_path": eligibility_path,
            },
        )

        audit = {
            "version": "cell10_4b_ota_burst_runlength_candidate_v2_THESIS",
            "candidate_available": False,
            "reason": reason,
            "target_cols": list(TARGET_COLS),
            "eligibility_audit_path": eligibility_path,
            "leakage_status": {
                "uses_train_values_for_fitting": True,
                "uses_val_values_for_fitting": False,
                "uses_test_values_for_fitting": False,
                "uses_test_values_for_selection": False,
                "uses_test_values_for_repair": False,
                "uses_test_values_for_candidate_promotion": False,
                "materializes_test_candidate": False,
                "overwrites_final_artifact": False,
            },
            "paper_claim_status": (
                "OTA burst/run-length candidate unavailable because TRAIN-only fit failed "
                "or eligibility/support was insufficient."
            ),
            "ts_unix": float(time.time()),
        }

        audit_path = os.path.join(OUT_REP, "cell10_4b_ota_burst_runlength_audit_v2_THESIS.json")
        contract_path = os.path.join(CONTRACT_DIR, "cell10_4b_ota_burst_runlength_contract_v2_THESIS.json")

        for path, obj in [(audit_path, audit), (contract_path, audit)]:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
            os.replace(tmp, path)

        globals()["CELL10_4B_OTA_AUDIT"] = audit
        globals()["CELL10_4B_OTA_AUDIT_PATH"] = audit_path
        globals()["CELL10_4B_OTA_CONTRACT_PATH"] = contract_path

        if "RUN_META" in globals():
            RUN_META.setdefault("portfolio_candidate_audits", {})
            RUN_META["portfolio_candidate_audits"]["ota_burst_runlength_active_window"] = audit

        log(f"[Cell10.4b] OTA burst/run-length unavailable: {reason}")
        log(f"[Cell10.4b] Saved audit: {audit_path}")
        log(f"[Cell10.4b] Saved contract: {contract_path}")

    else:
        # --------------------------------------------------
        # 5) Generate VAL/TEST candidates from fitted model and masks
        # --------------------------------------------------
        def _generate_split(mask_dict: dict, N: int, split_name: str):
            N = int(N)
            if N <= 0:
                raise RuntimeError(f"[Cell10.4b] {split_name}: N must be positive.")

            active = _mask_for_tier(mask_dict, "ota", N, missing_policy="error")
            active_n = int(active.sum())

            out = {c: np.full(N, np.nan, dtype=np.float32) for c in TARGET_COLS}

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_4b_ota_burst_runlength",
                    split_name,
                    "|".join(TARGET_COLS),
                    seed,
                    base_seed=seed + 10045,
                )
            )

            event_syn = np.zeros(active_n, dtype=np.int8)
            active_values = np.zeros((active_n, len(TARGET_COLS)), dtype=np.float32)

            if active_n > 0:
                event_syn = _sample_binary_runs(model, active_n, rng_local)

                pos = np.flatnonzero(event_syn > 0)
                if pos.size:
                    draw_idx = rng_local.integers(0, model["positive_pairs"].shape[0], size=pos.size)
                    active_values[pos, :] = model["positive_pairs"][draw_idx, :]

            for j, c in enumerate(TARGET_COLS):
                arr = np.full(N, np.nan, dtype=np.float32)
                if active_n > 0:
                    arr[active] = active_values[:, j]
                out[c] = _postprocess_protocol_values(c, arr)

            df_out = pd.DataFrame(out)

            gen_rows = []
            for c in TARGET_COLS:
                x = pd.to_numeric(df_out[c], errors="coerce").to_numpy(dtype=np.float64)
                finite_active = int(np.isfinite(x[active]).sum())
                finite_inactive = int(np.isfinite(x[~active]).sum())

                if active_n > 0 and finite_active != active_n:
                    raise RuntimeError(
                        f"[Cell10.4b] {split_name}.{c}: active finite mismatch: "
                        f"finite_active={finite_active}, active_n={active_n}"
                    )

                if finite_inactive != 0:
                    raise RuntimeError(
                        f"[Cell10.4b] {split_name}.{c}: inactive rows contain finite values: {finite_inactive}"
                    )

                gen_rows.append({
                    "split": split_name,
                    "col": c,
                    "tier": "ota",
                    "N": int(N),
                    "active_n": int(active_n),
                    "active_rate": float(np.mean(active)) if active.size else np.nan,
                    "finite_active": finite_active,
                    "finite_inactive": finite_inactive,
                    "generated_mean_active": _safe_mean(x[active]),
                    "generated_nonzero_rate_active": _nonzero_rate(x[active]),
                    "generated_transition_rate_event_active": _transition_rate(
                        np.nan_to_num(x[active], nan=0.0) > 0
                    ),
                    "model_train_event_rate": float(model["train_event_rate"]),
                    "model_train_positive_rows_n": int(model["train_positive_rows_n"]),
                })

            return df_out, pd.DataFrame(gen_rows)


        val_df, val_gen_audit = _generate_split(CELL10_VAL_MASKS, len(df_va), "VAL")
        test_df, test_gen_audit = _generate_split(CELL10_TEST_MASKS, len(df_te), "TEST")

        # --------------------------------------------------
        # 6) Persist candidate artifacts
        # --------------------------------------------------
        val_path = os.path.join(OUT_PORT, "candidate_ota_burst_runlength_active_window_VAL.parquet")
        test_path = os.path.join(OUT_PORT, "candidate_ota_burst_runlength_active_window_TEST.parquet")
        meta_path = os.path.join(OUT_PORT, "candidate_ota_burst_runlength_active_window_fit_meta_v2_THESIS.json")

        fit_audit_path = os.path.join(OUT_REP, "cell10_4b_ota_burst_runlength_fit_audit_v2_THESIS.csv")
        gen_audit_path = os.path.join(OUT_REP, "cell10_4b_ota_burst_runlength_generation_audit_v2_THESIS.csv")
        audit_path = os.path.join(OUT_REP, "cell10_4b_ota_burst_runlength_audit_v2_THESIS.json")
        contract_path = os.path.join(CONTRACT_DIR, "cell10_4b_ota_burst_runlength_contract_v2_THESIS.json")

        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)

        model_to_save = dict(model)
        model_to_save["positive_pairs_shape"] = [
            int(model["positive_pairs"].shape[0]),
            int(model["positive_pairs"].shape[1]),
        ]
        model_to_save.pop("positive_pairs", None)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(model_to_save, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)

        fit_audit_df = pd.DataFrame([
            {
                "generator": "ota_burst_runlength_active_window",
                "candidate_id": "ota_burst_runlength_active_window_v2_THESIS",
                "target_cols": "|".join(TARGET_COLS),
                "train_active_n": int(model["train_active_n"]),
                "train_event_rate": float(model["train_event_rate"]),
                "train_positive_rows_n": int(model["train_positive_rows_n"]),
                "train_event_runs_n": int(model["train_event_runs_n"]),
                "p_start_one": float(model["p_start_one"]),
                "zero_run_median": float(model["zero_run_median"]),
                "one_run_median": float(model["one_run_median"]),
                "zero_run_p95": float(model["zero_run_p95"]),
                "one_run_p95": float(model["one_run_p95"]),
            }
        ])
        gen_audit_df = pd.concat([val_gen_audit, test_gen_audit], axis=0, ignore_index=True)

        fit_audit_df.to_csv(fit_audit_path, index=False)
        gen_audit_df.to_csv(gen_audit_path, index=False)

        rec = _cell10_register_portfolio_candidate({
            "generator": "ota_burst_runlength_active_window",
            "candidate_id": "ota_burst_runlength_active_window_v2_THESIS",
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
            "role": "ota_temporal_runlength_autocorrelation_candidate",
            "eligibility": {
                "targets": list(TARGET_COLS),
                "train_active_n": int(model["train_active_n"]),
                "train_positive_rows_n": int(model["train_positive_rows_n"]),
                "eligibility_audit_path": eligibility_path,
            },
        })

        audit = {
            "version": "cell10_4b_ota_burst_runlength_candidate_v2_THESIS",
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
                "train_active_n": int(model["train_active_n"]),
                "train_event_rate": float(model["train_event_rate"]),
                "train_positive_rows_n": int(model["train_positive_rows_n"]),
                "train_event_runs_n": int(model["train_event_runs_n"]),
                "zero_run_median": float(model["zero_run_median"]),
                "one_run_median": float(model["one_run_median"]),
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
                "OTA burst/run-length is an active protocol-specific portfolio candidate "
                "for OTA aggregate totals. It is not final evidence unless selected later "
                "by the downstream VAL-only selector against A1."
            ),
            "ts_unix": float(time.time()),
        }

        for path, obj in [(audit_path, audit), (contract_path, audit)]:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
            os.replace(tmp, path)

        globals()["CELL10_4B_OTA_AUDIT"] = audit
        globals()["CELL10_4B_OTA_AUDIT_PATH"] = audit_path
        globals()["CELL10_4B_OTA_CONTRACT_PATH"] = contract_path

        if "RUN_META" in globals():
            RUN_META.setdefault("portfolio_candidate_audits", {})
            RUN_META["portfolio_candidate_audits"]["ota_burst_runlength_active_window"] = audit

        log(
            "[Cell10.4b] Registered OTA burst/run-length candidate | "
            f"cols={TARGET_COLS} | VAL={val_df.shape} | TEST={test_df.shape} | "
            f"train_active_n={model['train_active_n']} | "
            f"train_event_rate={model['train_event_rate']:.6f} | "
            f"positive_rows={model['train_positive_rows_n']}"
        )
        log(f"[Cell10.4b] Saved eligibility audit: {eligibility_path}")
        log(f"[Cell10.4b] Saved fit audit: {fit_audit_path}")
        log(f"[Cell10.4b] Saved generation audit: {gen_audit_path}")
        log(f"[Cell10.4b] Saved audit: {audit_path}")
        log(f"[Cell10.4b] Saved contract: {contract_path}")

log("--- END: Cell 10.4b — OTA burst/run-length active-window candidate (v2-THESIS) ---")