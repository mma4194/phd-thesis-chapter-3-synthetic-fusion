# ==========================================================
# CELL 10.3 — Negative Binomial / Poisson-Gamma count candidate — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Generate candidates for overdispersed count/burst protocol columns.
# - Fit all distribution parameters from TRAIN active values only.
# - Materialize VAL and TEST-horizon candidate artifacts.
# - Register the candidate into the predefined Cell 10 portfolio registry.
#
# Scientific contract:
# - TRAIN values are used for fitting.
# - VAL candidate is used later for selection.
# - TEST candidate is generated for final application/QA only.
# - TEST real values are never used for fitting, selection, repair, or promotion.
# - TEST horizon length and synthetic TEST masks may be used to materialize the
#   synthetic TEST candidate.
# ==========================================================

log("--- START: Cell 10.3 — Negative Binomial / Poisson-Gamma count candidate (v2-THESIS) ---")

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
    "_safe_mean", "_safe_var", "_nonzero_rate",
    "_write_json",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.3] Missing prerequisites: {missing}. Run Cell 10.1 first.")

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
        raise RuntimeError(f"[Cell10.3] Clean NB candidate generation forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.3] Unexpected selection_split_policy. "
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
vtm_thr = float(CFG.get("cell10_nb_vtm_threshold", 1.25))
min_train_n = int(CFG.get("cell10_nb_min_train_n", 200))
min_mean = float(CFG.get("cell10_nb_min_mean", 1e-6))

if vtm_thr <= 1.0:
    raise RuntimeError(f"[Cell10.3] vtm_thr should be >1 for overdispersion, got {vtm_thr}.")
if min_train_n <= 0:
    raise RuntimeError(f"[Cell10.3] min_train_n must be positive, got {min_train_n}.")
if min_mean < 0:
    raise RuntimeError(f"[Cell10.3] min_mean must be nonnegative, got {min_mean}.")

candidate_cols = []
eligibility_rows = []

family_df = CELL10_FAMILY_DF.copy()
if "col" not in family_df.columns:
    raise RuntimeError("[Cell10.3] CELL10_FAMILY_DF missing required column 'col'.")

for _, row in family_df.iterrows():
    c = str(row["col"])

    if c not in set(map(str, CELL10_PROTO_COLS)):
        continue

    countlike = bool(row.get("countlike", False))
    vtm = float(row["variance_to_mean"]) if pd.notna(row.get("variance_to_mean", np.nan)) else np.nan
    train_n = int(row["train_n"]) if pd.notna(row.get("train_n", np.nan)) else 0
    mean = float(row["train_mean"]) if pd.notna(row.get("train_mean", np.nan)) else np.nan

    eligible = (
        countlike
        and train_n >= min_train_n
        and np.isfinite(vtm)
        and vtm >= vtm_thr
        and np.isfinite(mean)
        and mean > min_mean
    )

    eligibility_rows.append({
        "col": c,
        "tier": _tier_of(c),
        "countlike": bool(countlike),
        "train_n": int(train_n),
        "train_mean": mean if np.isfinite(mean) else np.nan,
        "variance_to_mean": vtm if np.isfinite(vtm) else np.nan,
        "eligible": bool(eligible),
        "reason": (
            "eligible"
            if eligible else
            "not_countlike_or_insufficient_support_or_not_overdispersed"
        ),
    })

    if eligible:
        candidate_cols.append(c)

candidate_cols = list(dict.fromkeys(candidate_cols))

eligibility_df = pd.DataFrame(eligibility_rows)
eligibility_path = os.path.join(OUT_REP, "cell10_3_nb_poisson_gamma_eligibility_v2_THESIS.csv")
eligibility_df.to_csv(eligibility_path, index=False)

# ----------------------------------------------------------
# 2) Register unavailable if no eligible columns
# ----------------------------------------------------------
if not candidate_cols:
    reason = "No count columns met overdispersion/min-support eligibility."

    _cell10_register_unavailable_backend(
        "negative_binomial_poisson_gamma",
        reason=reason,
        meta={
            "source": "Cell10.3",
            "version": "cell10_3_nb_poisson_gamma_v2_THESIS",
            "vtm_threshold": float(vtm_thr),
            "min_train_n": int(min_train_n),
            "min_mean": float(min_mean),
            "eligibility_audit_path": eligibility_path,
        },
    )

    audit = {
        "version": "cell10_3_nb_poisson_gamma_candidate_v2_THESIS",
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
            "NB/Poisson-Gamma backend unavailable for this run because no protocol "
            "count column satisfied TRAIN-only overdispersion/support eligibility."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_3_nb_poisson_gamma_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_3_nb_poisson_gamma_contract_v2_THESIS.json")

    for path, obj in [(audit_path, audit), (contract_path, audit)]:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        os.replace(tmp, path)

    globals()["CELL10_3_NB_AUDIT"] = audit
    globals()["CELL10_3_NB_AUDIT_PATH"] = audit_path
    globals()["CELL10_3_NB_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["negative_binomial_poisson_gamma"] = audit

    log("[Cell10.3] No eligible NB/Poisson-Gamma columns; registered backend as unavailable.")
    log(f"[Cell10.3] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.3] Saved audit: {audit_path}")
    log(f"[Cell10.3] Saved contract: {contract_path}")

else:
    # ------------------------------------------------------
    # 3) Fit TRAIN-only NB/Poisson models once
    # ------------------------------------------------------
    def _stable_seed_from_parts(*parts, base_seed: int = 1337) -> int:
        payload = "::".join(str(p) for p in parts)
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int((int(h[:12], 16) + int(base_seed)) % (2**32 - 1))


    def _fit_nb_moments(x):
        x = _finite_1d(x)
        x = np.maximum(np.rint(x), 0)

        if x.size <= 0:
            raise RuntimeError("[Cell10.3] Cannot fit NB/Poisson model on empty data.")

        mu = float(np.mean(x))
        var = float(np.var(x)) if x.size > 1 else mu

        if not np.isfinite(mu) or mu < 0:
            raise RuntimeError(f"[Cell10.3] Invalid fitted mean: {mu}")
        if not np.isfinite(var) or var < 0:
            raise RuntimeError(f"[Cell10.3] Invalid fitted variance: {var}")

        if var <= mu + 1e-9:
            return {
                "kind": "poisson",
                "mu": float(max(mu, 0.0)),
                "var": float(var),
            }

        r = float(mu * mu / max(var - mu, 1e-9))
        p = float(r / max(r + mu, 1e-9))

        return {
            "kind": "negative_binomial",
            "mu": float(mu),
            "var": float(var),
            "r": float(max(r, 1e-6)),
            "p": float(min(max(p, 1e-9), 1.0 - 1e-9)),
        }


    def _sample_model(model: dict, n: int, rng_local: np.random.Generator) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        kind = str(model.get("kind", ""))
        if kind == "poisson":
            return rng_local.poisson(float(model["mu"]), size=n).astype(np.float32)

        if kind == "negative_binomial":
            return rng_local.negative_binomial(
                float(model["r"]),
                float(model["p"]),
                size=n,
            ).astype(np.float32)

        raise RuntimeError(f"[Cell10.3] Unknown count model kind: {kind!r}")


    fit_meta = {}
    fit_rows = []

    for c in candidate_cols:
        tier = _tier_of(c)
        if tier is None:
            raise RuntimeError(f"[Cell10.3] Cannot resolve protocol tier for column: {c}")

        xtr = _column_train_values(c, active_only=True)
        xtr_f = _finite_1d(xtr)
        xtr_f = np.maximum(np.rint(xtr_f), 0)

        if xtr_f.size < min_train_n:
            raise RuntimeError(
                f"[Cell10.3] Column {c} passed eligibility but has insufficient TRAIN data at fit time: "
                f"{xtr_f.size} < {min_train_n}"
            )

        model = _fit_nb_moments(xtr_f)

        train_mean = _safe_mean(xtr_f)
        train_var = _safe_var(xtr_f)
        train_vtm = (
            float(train_var / max(train_mean, 1e-9))
            if np.isfinite(train_var) and np.isfinite(train_mean) and train_mean > 0
            else None
        )

        fit_meta[c] = {
            "tier": tier,
            "train_n": int(xtr_f.size),
            "train_mean": train_mean,
            "train_var": train_var,
            "train_vtm": train_vtm,
            "train_nonzero_rate": _nonzero_rate(xtr_f),
            "model": model,
        }

        fit_rows.append({
            "col": c,
            "tier": tier,
            "train_n": int(xtr_f.size),
            "train_mean": train_mean,
            "train_var": train_var,
            "train_vtm": train_vtm,
            "train_nonzero_rate": _nonzero_rate(xtr_f),
            "model_kind": model["kind"],
            "model_mu": float(model.get("mu", np.nan)),
            "model_var": float(model.get("var", np.nan)),
            "model_r": float(model.get("r", np.nan)) if model["kind"] == "negative_binomial" else np.nan,
            "model_p": float(model.get("p", np.nan)) if model["kind"] == "negative_binomial" else np.nan,
        })

    # ------------------------------------------------------
    # 4) Generate VAL/TEST candidates from fitted models and masks
    # ------------------------------------------------------
    def _generate_split(N: int, mask_dict: dict, cols: list, split_name: str):
        N = int(N)
        if N <= 0:
            raise RuntimeError(f"[Cell10.3] {split_name}: N must be positive.")

        out = {}
        gen_rows = []

        for c in cols:
            tier = fit_meta[c]["tier"]
            missing_policy = "zero" if tier == "zwave" else "error"
            active = _mask_for_tier(mask_dict, tier, N, missing_policy=missing_policy)

            vals = np.full(N, np.nan, dtype=np.float32)

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_3_nb_poisson_gamma",
                    split_name,
                    c,
                    seed,
                    base_seed=seed + 10030,
                )
            )

            if int(active.sum()) > 0:
                vals[active] = _sample_model(fit_meta[c]["model"], int(active.sum()), rng_local)

            vals = _postprocess_protocol_values(c, vals)

            finite_active = int(np.isfinite(vals[active]).sum())
            finite_inactive = int(np.isfinite(vals[~active]).sum())

            if int(active.sum()) > 0 and finite_active != int(active.sum()):
                raise RuntimeError(
                    f"[Cell10.3] {split_name}.{c}: active finite mismatch: "
                    f"finite_active={finite_active}, active_n={int(active.sum())}"
                )

            if finite_inactive != 0:
                raise RuntimeError(
                    f"[Cell10.3] {split_name}.{c}: inactive rows contain finite values: {finite_inactive}"
                )

            out[c] = vals

            gen_rows.append({
                "split": split_name,
                "col": c,
                "tier": tier,
                "N": int(N),
                "active_n": int(active.sum()),
                "active_rate": float(np.mean(active)) if active.size else np.nan,
                "finite_active": finite_active,
                "finite_inactive": finite_inactive,
                "generated_mean_active": _safe_mean(vals[active]),
                "generated_nonzero_rate_active": _nonzero_rate(vals[active]),
                "model_kind": fit_meta[c]["model"]["kind"],
            })

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
    val_path = os.path.join(OUT_PORT, "candidate_negative_binomial_VAL.parquet")
    test_path = os.path.join(OUT_PORT, "candidate_negative_binomial_TEST.parquet")
    meta_path = os.path.join(OUT_PORT, "candidate_negative_binomial_fit_meta_v2_THESIS.json")

    fit_audit_path = os.path.join(OUT_REP, "cell10_3_nb_poisson_gamma_fit_audit_v2_THESIS.csv")
    gen_audit_path = os.path.join(OUT_REP, "cell10_3_nb_poisson_gamma_generation_audit_v2_THESIS.csv")
    audit_path = os.path.join(OUT_REP, "cell10_3_nb_poisson_gamma_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_3_nb_poisson_gamma_contract_v2_THESIS.json")

    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    _write_json(meta_path, fit_meta)

    fit_audit_df = pd.DataFrame(fit_rows)
    gen_audit_df = pd.concat([val_gen_audit, test_gen_audit], axis=0, ignore_index=True)

    fit_audit_df.to_csv(fit_audit_path, index=False)
    gen_audit_df.to_csv(gen_audit_path, index=False)

    rec = _cell10_register_portfolio_candidate({
        "generator": "negative_binomial_poisson_gamma",
        "candidate_id": "nb_poisson_gamma_overdispersed_counts_v2_THESIS",
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
        "role": "overdispersed_count_burst_candidate",
        "eligibility": {
            "variance_to_mean_threshold": float(vtm_thr),
            "min_train_n": int(min_train_n),
            "min_mean": float(min_mean),
            "eligibility_audit_path": eligibility_path,
        },
    })

    audit = {
        "version": "cell10_3_nb_poisson_gamma_candidate_v2_THESIS",
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
            "NB/Poisson-Gamma is an active predefined portfolio candidate for eligible "
            "overdispersed count columns. It is not final evidence unless selected later "
            "by the downstream VAL-only selector against A1."
        ),
        "ts_unix": float(time.time()),
    }

    for path, obj in [(audit_path, audit), (contract_path, audit)]:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        os.replace(tmp, path)

    globals()["CELL10_3_NB_AUDIT"] = audit
    globals()["CELL10_3_NB_AUDIT_PATH"] = audit_path
    globals()["CELL10_3_NB_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["negative_binomial_poisson_gamma"] = audit

    log(
        "[Cell10.3] Registered NB/Poisson-Gamma candidate | "
        f"cols={len(candidate_cols)} | VAL={val_df.shape} | TEST={test_df.shape}"
    )
    log(f"[Cell10.3] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.3] Saved fit audit: {fit_audit_path}")
    log(f"[Cell10.3] Saved generation audit: {gen_audit_path}")
    log(f"[Cell10.3] Saved audit: {audit_path}")
    log(f"[Cell10.3] Saved contract: {contract_path}")

log("--- END: Cell 10.3 — Negative Binomial / Poisson-Gamma count candidate (v2-THESIS) ---")