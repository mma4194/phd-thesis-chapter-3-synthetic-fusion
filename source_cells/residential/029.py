# ==========================================================
# CELL 10.4 — Markov / semi-Markov sparse event candidate — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Generate candidates for sparse event/count protocol columns.
# - Fit binary activity dwell/run structure from TRAIN active values only.
# - Sample positive magnitudes from TRAIN positive support only.
# - Materialize VAL and TEST-horizon candidate artifacts.
#
# Scientific contract:
# - TRAIN values are used for fitting.
# - VAL candidate is used later for selection.
# - TEST candidate is generated for final application/QA only.
# - TEST real values are never used for fitting, selection, repair, or promotion.
# - TEST horizon length and synthetic TEST masks may be used to materialize the
#   synthetic TEST candidate.
# ==========================================================

log("--- START: Cell 10.4 — Markov / semi-Markov sparse event candidate (v2-THESIS) ---")

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
    "CELL10_PROTO_COLS", "CELL10_FAMILY_DF",
    "CELL10_VAL_MASKS", "CELL10_TEST_MASKS",
    "_cell10_register_portfolio_candidate",
    "_cell10_register_unavailable_backend",
    "_tier_of", "_mask_for_tier", "_column_train_values",
    "_postprocess_protocol_values", "_finite_1d",
    "_safe_mean", "_safe_var", "_nonzero_rate", "_transition_rate",
    "_write_json",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.4] Missing prerequisites: {missing}. Run Cell 10.1 first.")

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
        raise RuntimeError(f"[Cell10.4] Clean Markov candidate generation forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.4] Unexpected selection_split_policy. "
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
# 1) Eligibility
# ----------------------------------------------------------
nz_thr = float(CFG.get("cell10_markov_sparse_nz_thr", 0.20))
min_train_n = int(CFG.get("cell10_markov_min_train_n", 200))
min_pos_n = int(CFG.get("cell10_markov_min_positive_n", 5))

if not (0.0 < nz_thr <= 1.0):
    raise RuntimeError(f"[Cell10.4] nz_thr must be in (0,1], got {nz_thr}.")
if min_train_n <= 0:
    raise RuntimeError(f"[Cell10.4] min_train_n must be positive, got {min_train_n}.")
if min_pos_n <= 0:
    raise RuntimeError(f"[Cell10.4] min_pos_n must be positive, got {min_pos_n}.")

candidate_cols = []
eligibility_rows = []

family_df = CELL10_FAMILY_DF.copy()
if "col" not in family_df.columns:
    raise RuntimeError("[Cell10.4] CELL10_FAMILY_DF missing required column 'col'.")

for _, row in family_df.iterrows():
    c = str(row["col"])

    if c not in set(map(str, CELL10_PROTO_COLS)):
        continue

    countlike = bool(row.get("countlike", False))
    train_n = int(row["train_n"]) if pd.notna(row.get("train_n", np.nan)) else 0
    nz = float(row["train_nonzero_rate"]) if pd.notna(row.get("train_nonzero_rate", np.nan)) else np.nan

    pos_n = 0
    if countlike and train_n >= min_train_n and np.isfinite(nz) and nz <= nz_thr:
        x = _column_train_values(c, active_only=True)
        xf = np.maximum(np.rint(_finite_1d(x)), 0)
        pos_n = int(np.sum(xf > 0))

    eligible = (
        countlike
        and train_n >= min_train_n
        and np.isfinite(nz)
        and nz <= nz_thr
        and pos_n >= min_pos_n
    )

    eligibility_rows.append({
        "col": c,
        "tier": _tier_of(c),
        "countlike": bool(countlike),
        "train_n": int(train_n),
        "train_nonzero_rate": nz if np.isfinite(nz) else np.nan,
        "train_positive_n": int(pos_n),
        "eligible": bool(eligible),
        "reason": (
            "eligible"
            if eligible else
            "not_countlike_or_insufficient_support_or_not_sparse_or_too_few_positive_events"
        ),
    })

    if eligible:
        candidate_cols.append(c)

candidate_cols = list(dict.fromkeys(candidate_cols))

eligibility_df = pd.DataFrame(eligibility_rows)
eligibility_path = os.path.join(OUT_REP, "cell10_4_markov_semimarkov_eligibility_v2_THESIS.csv")
eligibility_df.to_csv(eligibility_path, index=False)

# ----------------------------------------------------------
# 2) Register unavailable if no eligible columns
# ----------------------------------------------------------
if not candidate_cols:
    reason = "No sparse count/event columns met TRAIN-only Markov/semi-Markov eligibility."

    _cell10_register_unavailable_backend(
        "markov_semimarkov",
        reason=reason,
        meta={
            "source": "Cell10.4",
            "version": "cell10_4_markov_semimarkov_v2_THESIS",
            "nz_thr": float(nz_thr),
            "min_train_n": int(min_train_n),
            "min_pos_n": int(min_pos_n),
            "eligibility_audit_path": eligibility_path,
        },
    )

    audit = {
        "version": "cell10_4_markov_semimarkov_candidate_v2_THESIS",
        "candidate_available": False,
        "reason": reason,
        "eligible_cols_n": 0,
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
            "Markov/semi-Markov backend unavailable for this run because no protocol "
            "sparse event/count column satisfied TRAIN-only support and sparsity eligibility."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_4_markov_semimarkov_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_4_markov_semimarkov_contract_v2_THESIS.json")

    for path, obj in [(audit_path, audit), (contract_path, audit)]:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        os.replace(tmp, path)

    globals()["CELL10_4_MARKOV_AUDIT"] = audit
    globals()["CELL10_4_MARKOV_AUDIT_PATH"] = audit_path
    globals()["CELL10_4_MARKOV_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["markov_semimarkov"] = audit

    log("[Cell10.4] No eligible Markov/semi-Markov columns; registered backend as unavailable.")
    log(f"[Cell10.4] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.4] Saved audit: {audit_path}")
    log(f"[Cell10.4] Saved contract: {contract_path}")

else:
    # ------------------------------------------------------
    # 3) TRAIN-only semi-Markov fitting
    # ------------------------------------------------------
    def _stable_seed_from_parts(*parts, base_seed: int = 1337) -> int:
        payload = "::".join(str(p) for p in parts)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int((int(h[:12], 16) + int(base_seed)) % (2**32 - 1))


    def _runs_binary(b):
        b = np.asarray(b, dtype=np.int8).reshape(-1)
        if b.size == 0:
            return []

        out = []
        s = 0
        cur = int(b[0])

        for i in range(1, b.size + 1):
            if i == b.size or int(b[i]) != cur:
                out.append((int(cur), int(i - s)))
                if i < b.size:
                    s = i
                    cur = int(b[i])

        return out


    def _fit_semimarkov_sparse_model(x_train):
        x = np.maximum(np.rint(_finite_1d(x_train)), 0).astype(np.float32, copy=False)

        if x.size < min_train_n:
            raise RuntimeError(
                f"[Cell10.4] Cannot fit semi-Markov model: train_n={x.size} < {min_train_n}"
            )

        b = (x > 0).astype(np.int8)
        pos_vals = x[x > 0]

        if pos_vals.size < min_pos_n:
            raise RuntimeError(
                f"[Cell10.4] Cannot fit semi-Markov model: pos_n={pos_vals.size} < {min_pos_n}"
            )

        runs = _runs_binary(b)
        if not runs:
            raise RuntimeError("[Cell10.4] Cannot fit semi-Markov model: no binary runs.")

        lens_0 = np.asarray([L for v, L in runs if int(v) == 0], dtype=np.int64)
        lens_1 = np.asarray([L for v, L in runs if int(v) == 1], dtype=np.int64)

        if lens_0.size == 0:
            lens_0 = np.asarray([1], dtype=np.int64)
        if lens_1.size == 0:
            lens_1 = np.asarray([1], dtype=np.int64)

        starts = np.asarray([v for v, _ in runs], dtype=np.int8)
        p_start_one = float(np.mean(starts == 1))

        return {
            "train_n": int(x.size),
            "train_positive_n": int(pos_vals.size),
            "train_nonzero_rate": float(np.mean(b > 0)),
            "train_transition_rate": _transition_rate(b),
            "p_start_one": float(p_start_one),
            "lens_0": lens_0.astype(int).tolist(),
            "lens_1": lens_1.astype(int).tolist(),
            "pos_values": pos_vals.astype(float).tolist(),
            "pos_value_unique_n": int(len(np.unique(pos_vals))),
            "zero_run_median": float(np.median(lens_0)) if lens_0.size else np.nan,
            "one_run_median": float(np.median(lens_1)) if lens_1.size else np.nan,
            "zero_run_p95": float(np.quantile(lens_0, 0.95)) if lens_0.size else np.nan,
            "one_run_p95": float(np.quantile(lens_1, 0.95)) if lens_1.size else np.nan,
        }


    def _generate_binary_from_model(model: dict, n: int, rng_local: np.random.Generator) -> np.ndarray:
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


    fit_meta = {}
    fit_rows = []

    for c in candidate_cols:
        tier = _tier_of(c)
        if tier is None:
            raise RuntimeError(f"[Cell10.4] Cannot resolve protocol tier for column: {c}")

        xtr = _column_train_values(c, active_only=True)
        model = _fit_semimarkov_sparse_model(xtr)

        fit_meta[c] = {
            "tier": tier,
            "model": model,
        }

        fit_rows.append({
            "col": c,
            "tier": tier,
            "train_n": int(model["train_n"]),
            "train_positive_n": int(model["train_positive_n"]),
            "train_nonzero_rate": float(model["train_nonzero_rate"]),
            "train_transition_rate": model["train_transition_rate"],
            "p_start_one": float(model["p_start_one"]),
            "zero_run_median": model["zero_run_median"],
            "one_run_median": model["one_run_median"],
            "zero_run_p95": model["zero_run_p95"],
            "one_run_p95": model["one_run_p95"],
            "pos_value_unique_n": int(model["pos_value_unique_n"]),
        })

    # ------------------------------------------------------
    # 4) Generate VAL/TEST candidates from fitted models and masks
    # ------------------------------------------------------
    def _generate_col(c: str, N: int, mask_dict: dict, split_name: str) -> tuple:
        tier = fit_meta[c]["tier"]
        model = fit_meta[c]["model"]

        active = _mask_for_tier(
            mask_dict,
            tier,
            N,
            missing_policy="zero" if tier == "zwave" else "error",
        )

        out = np.full(N, np.nan, dtype=np.float32)

        rng_local = np.random.default_rng(
            _stable_seed_from_parts(
                "cell10_4_markov_semimarkov",
                split_name,
                c,
                seed,
                base_seed=seed + 10040,
            )
        )

        active_n = int(active.sum())
        if active_n > 0:
            b = _generate_binary_from_model(model, active_n, rng_local)
            vals = np.zeros(active_n, dtype=np.float32)

            pos_idx = np.flatnonzero(b > 0)
            pos_values = np.asarray(model.get("pos_values", [1.0]), dtype=np.float32)
            pos_values = pos_values[np.isfinite(pos_values)]
            pos_values = pos_values[pos_values > 0]

            if pos_values.size == 0:
                pos_values = np.asarray([1.0], dtype=np.float32)

            if pos_idx.size > 0:
                draws = rng_local.choice(pos_values, size=pos_idx.size, replace=True)
                vals[pos_idx] = draws.astype(np.float32, copy=False)

            out[active] = vals

        out = _postprocess_protocol_values(c, out)

        finite_active = int(np.isfinite(out[active]).sum())
        finite_inactive = int(np.isfinite(out[~active]).sum())

        if active_n > 0 and finite_active != active_n:
            raise RuntimeError(
                f"[Cell10.4] {split_name}.{c}: active finite mismatch: "
                f"finite_active={finite_active}, active_n={active_n}"
            )

        if finite_inactive != 0:
            raise RuntimeError(
                f"[Cell10.4] {split_name}.{c}: inactive rows contain finite values: {finite_inactive}"
            )

        gen_row = {
            "split": split_name,
            "col": c,
            "tier": tier,
            "N": int(N),
            "active_n": int(active_n),
            "active_rate": float(np.mean(active)) if active.size else np.nan,
            "finite_active": finite_active,
            "finite_inactive": finite_inactive,
            "generated_nonzero_rate_active": _nonzero_rate(out[active]),
            "generated_transition_rate_active": _transition_rate(np.nan_to_num(out[active], nan=0.0) > 0),
            "generated_mean_active": _safe_mean(out[active]),
            "model_train_nonzero_rate": float(model["train_nonzero_rate"]),
            "model_train_positive_n": int(model["train_positive_n"]),
        }

        return out, gen_row


    def _generate_split(N: int, mask_dict: dict, cols: list, split_name: str):
        N = int(N)
        if N <= 0:
            raise RuntimeError(f"[Cell10.4] {split_name}: N must be positive.")

        out = {}
        gen_rows = []

        for c in cols:
            vals, row = _generate_col(c, N, mask_dict, split_name)
            out[c] = vals
            gen_rows.append(row)

        return pd.DataFrame(out), pd.DataFrame(gen_rows)


    val_df, val_gen_audit = _generate_split(
        N=len(df_va),
        mask_dict=CELL10_VAL_MASKS,
        cols=candidate_cols,
        split_name="VAL",
    )

    test_df, test_gen_audit = _generate_split(
        N=len(df_te),
        mask_dict=CELL10_TEST_MASKS,
        cols=candidate_cols,
        split_name="TEST",
    )

    # ------------------------------------------------------
    # 5) Persist candidate artifacts
    # ------------------------------------------------------
    val_path = os.path.join(OUT_PORT, "candidate_markov_semimarkov_VAL.parquet")
    test_path = os.path.join(OUT_PORT, "candidate_markov_semimarkov_TEST.parquet")
    meta_path = os.path.join(OUT_PORT, "candidate_markov_semimarkov_fit_meta_v2_THESIS.json")

    fit_audit_path = os.path.join(OUT_REP, "cell10_4_markov_semimarkov_fit_audit_v2_THESIS.csv")
    gen_audit_path = os.path.join(OUT_REP, "cell10_4_markov_semimarkov_generation_audit_v2_THESIS.csv")
    audit_path = os.path.join(OUT_REP, "cell10_4_markov_semimarkov_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_4_markov_semimarkov_contract_v2_THESIS.json")

    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    _write_json(meta_path, fit_meta)

    fit_audit_df = pd.DataFrame(fit_rows)
    gen_audit_df = pd.concat([val_gen_audit, test_gen_audit], axis=0, ignore_index=True)

    fit_audit_df.to_csv(fit_audit_path, index=False)
    gen_audit_df.to_csv(gen_audit_path, index=False)

    rec = _cell10_register_portfolio_candidate({
        "generator": "markov_semimarkov",
        "candidate_id": "semimarkov_sparse_event_dwell_v2_THESIS",
        "columns": candidate_cols,
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
        "role": "rare_discrete_event_dwell_transition_candidate",
        "eligibility": {
            "train_nonzero_rate_max": float(nz_thr),
            "min_train_n": int(min_train_n),
            "min_positive_n": int(min_pos_n),
            "eligibility_audit_path": eligibility_path,
        },
    })

    audit = {
        "version": "cell10_4_markov_semimarkov_candidate_v2_THESIS",
        "candidate_available": True,
        "registered_candidate": rec,
        "eligible_cols_n": int(len(candidate_cols)),
        "eligible_cols": list(candidate_cols),
        "val_path": val_path,
        "test_path": test_path,
        "fit_meta_path": meta_path,
        "fit_audit_path": fit_audit_path,
        "generation_audit_path": gen_audit_path,
        "eligibility_audit_path": eligibility_path,
        "val_shape": [int(val_df.shape[0]), int(val_df.shape[1])],
        "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
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
            "Markov/semi-Markov is an active predefined portfolio candidate for eligible "
            "sparse event/count protocol columns. It is not final evidence unless selected "
            "later by the downstream VAL-only selector against A1."
        ),
        "ts_unix": float(time.time()),
    }

    for path, obj in [(audit_path, audit), (contract_path, audit)]:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        os.replace(tmp, path)

    globals()["CELL10_4_MARKOV_AUDIT"] = audit
    globals()["CELL10_4_MARKOV_AUDIT_PATH"] = audit_path
    globals()["CELL10_4_MARKOV_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["markov_semimarkov"] = audit

    log(
        "[Cell10.4] Registered Markov/semi-Markov candidate | "
        f"cols={len(candidate_cols)} | VAL={val_df.shape} | TEST={test_df.shape}"
    )
    log(f"[Cell10.4] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.4] Saved fit audit: {fit_audit_path}")
    log(f"[Cell10.4] Saved generation audit: {gen_audit_path}")
    log(f"[Cell10.4] Saved audit: {audit_path}")
    log(f"[Cell10.4] Saved contract: {contract_path}")

log("--- END: Cell 10.4 — Markov / semi-Markov sparse event candidate (v2-THESIS) ---")