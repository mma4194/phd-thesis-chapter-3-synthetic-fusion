# ==========================================================
# CELL 10.5 — Gaussian Copula continuous-correlation candidate — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Generate correlated continuous/semi-continuous protocol candidates when
#   eligible non-count protocol columns exist.
# - Fit Gaussian copula per protocol tier using TRAIN active values only.
# - Emit VAL and TEST-horizon candidate artifacts for downstream VAL-only selection.
#
# Scientific contract:
# - Fit TRAIN only.
# - Generate VAL and TEST candidates from TRAIN-fitted copula and masks.
# - Do not read real TEST values.
# - Do not use VAL for fitting.
# - Do not select anything in this cell.
# - Not final evidence unless selected downstream on VAL against A1.
#
# Expected outcome:
# - If no eligible continuous protocol groups exist, register the backend as
#   unavailable. This is valid and should not fail the pipeline.
# ==========================================================

log("--- START: Cell 10.5 — Gaussian Copula continuous-correlation candidate (v2-THESIS) ---")

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
    "_tier_of", "_mask_for_tier",
    "_postprocess_protocol_values",
    "_safe_mean", "_safe_var", "_nonzero_rate", "_transition_rate",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.5] Missing prerequisites: {missing}. Run Cell 10.1 first.")

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
        raise RuntimeError(f"[Cell10.5] Clean Gaussian Copula generation forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.5] Unexpected selection_split_policy. "
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
# 1) JSON helpers and unavailable finalizer
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
        "gaussian_copula",
        reason=reason,
        meta={
            "source": "Cell10.5",
            "version": "cell10_5_gaussian_copula_v2_THESIS",
            **dict(extra_meta or {}),
        },
    )

    audit = {
        "version": "cell10_5_gaussian_copula_candidate_v2_THESIS",
        "candidate_available": False,
        "reason": reason,
        "registered_candidate_id": None,
        "eligible_tiers": [],
        "eligible_cols": [],
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
            "Gaussian Copula backend unavailable for this run. This is valid if no "
            "tier has enough TRAIN-supported continuous/semi-continuous protocol "
            "columns for correlation-preserving copula fitting."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_5_gaussian_copula_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_5_gaussian_copula_contract_v2_THESIS.json")

    _write_atomic_json(audit_path, audit)
    _write_atomic_json(contract_path, audit)

    globals()["CELL10_5_COPULA_AUDIT"] = audit
    globals()["CELL10_5_COPULA_AUDIT_PATH"] = audit_path
    globals()["CELL10_5_COPULA_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["gaussian_copula"] = audit

    log(f"[Cell10.5] Gaussian Copula unavailable: {reason}")
    log(f"[Cell10.5] Saved audit: {audit_path}")
    log(f"[Cell10.5] Saved contract: {contract_path}")


# ----------------------------------------------------------
# 2) Optional scipy dependency
# ----------------------------------------------------------
try:
    from scipy.stats import norm
except Exception as e:
    _finalize_unavailable(
        reason=f"scipy.stats.norm unavailable: {type(e).__name__}: {e}",
        extra_meta={"dependency": "scipy.stats.norm"},
    )
else:
    # ------------------------------------------------------
    # 3) Eligibility: continuous/semi-continuous non-count columns per tier
    # ------------------------------------------------------
    min_cols = int(CFG.get("cell10_copula_min_cols_per_tier", 2))
    min_rows = int(CFG.get("cell10_copula_min_train_rows", 500))
    min_unique = int(CFG.get("cell10_copula_min_unique", 20))

    if min_cols <= 1:
        raise RuntimeError(f"[Cell10.5] min_cols must be >=2 for correlation modeling, got {min_cols}.")
    if min_rows <= 0:
        raise RuntimeError(f"[Cell10.5] min_rows must be positive, got {min_rows}.")
    if min_unique <= 1:
        raise RuntimeError(f"[Cell10.5] min_unique must be >1, got {min_unique}.")

    proto_col_set = set(map(str, CELL10_PROTO_COLS))
    family_df = CELL10_FAMILY_DF.copy()

    required_cols = {"col", "tier", "countlike", "train_unique_n"}
    missing_family_cols = sorted(required_cols - set(family_df.columns))
    if missing_family_cols:
        raise RuntimeError(f"[Cell10.5] CELL10_FAMILY_DF missing required columns: {missing_family_cols}")

    tier_cols = {t: [] for t in ["router", "ota", "zigbee"]}

    eligibility_rows = []
    for _, row in family_df.iterrows():
        c = str(row["col"])
        tier = str(row["tier"])
        countlike = bool(row["countlike"])
        train_unique_n = int(row["train_unique_n"]) if pd.notna(row["train_unique_n"]) else 0

        schema_ok = (
            c in proto_col_set
            and c in df_tr.columns
            and c in df_va.columns
            and c in df_te.columns
            and tier in tier_cols
        )

        eligible = (
            schema_ok
            and (not countlike)
            and train_unique_n >= min_unique
        )

        eligibility_rows.append({
            "col": c,
            "tier": tier,
            "schema_ok": bool(schema_ok),
            "countlike": bool(countlike),
            "train_unique_n": int(train_unique_n),
            "eligible_pre_group": bool(eligible),
            "reason": (
                "eligible_pre_group"
                if eligible else
                "not_schema_ok_or_countlike_or_low_unique_support"
            ),
        })

        if eligible:
            tier_cols[tier].append(c)

    tier_cols = {
        t: list(dict.fromkeys(cols))
        for t, cols in tier_cols.items()
        if len(cols) >= min_cols
    }

    eligibility_path = os.path.join(OUT_REP, "cell10_5_gaussian_copula_eligibility_v2_THESIS.csv")
    pd.DataFrame(eligibility_rows).to_csv(eligibility_path, index=False)

    if not tier_cols:
        _finalize_unavailable(
            reason="No tier had enough continuous non-count columns for Gaussian Copula fitting.",
            extra_meta={
                "min_cols": min_cols,
                "min_unique": min_unique,
                "eligibility_audit_path": eligibility_path,
            },
        )

    else:
        # --------------------------------------------------
        # 4) Helpers
        # --------------------------------------------------
        def _active_mask_from_train_df(tier: str) -> np.ndarray:
            c = f"{tier}__obs_present"
            if c not in df_tr.columns:
                raise RuntimeError(f"[Cell10.5] TRAIN missing observability column: {c}")
            v = pd.to_numeric(df_tr[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            return v > 0.5


        def _fit_tier_copula(tier: str, cols: list):
            active = _active_mask_from_train_df(tier)
            X = df_tr.loc[active, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

            row_ok = np.isfinite(X).all(axis=1)
            X = X[row_ok]

            if X.shape[0] < min_rows:
                return None, {
                    "tier": tier,
                    "cols": list(cols),
                    "fit_available": False,
                    "train_rows": int(X.shape[0]),
                    "reason": f"train_rows < {min_rows}",
                }

            X = np.maximum(X, 0.0)

            Z_cols = []
            supports = {}
            support_summaries = {}

            for j, c in enumerate(cols):
                x = X[:, j].astype(np.float64)
                xs = np.sort(x)

                if xs.size <= 1:
                    return None, {
                        "tier": tier,
                        "cols": list(cols),
                        "fit_available": False,
                        "train_rows": int(X.shape[0]),
                        "reason": f"column {c} has insufficient support after finite filtering",
                    }

                supports[c] = xs.astype(np.float32)

                ranks = np.searchsorted(xs, x, side="right")
                u = (ranks - 0.5) / max(len(xs), 1)
                u = np.clip(u, 1e-4, 1.0 - 1e-4)
                Z_cols.append(norm.ppf(u))

                support_summaries[c] = {
                    "train_n": int(xs.size),
                    "support_unique_n": int(np.unique(np.round(xs, 6)).size),
                    "mean": float(np.mean(xs)),
                    "std": float(np.std(xs)),
                    "q50": float(np.quantile(xs, 0.50)),
                    "q95": float(np.quantile(xs, 0.95)),
                    "q99": float(np.quantile(xs, 0.99)),
                }

            Z = np.column_stack(Z_cols)
            R = np.corrcoef(Z, rowvar=False)
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            R = 0.5 * (R + R.T)
            np.fill_diagonal(R, 1.0)

            shrink = float(CFG.get("cell10_copula_corr_shrink", 0.03))
            shrink = float(np.clip(shrink, 0.0, 1.0))
            R = (1.0 - shrink) * R + shrink * np.eye(R.shape[0])

            # Numerical positive-definiteness guard.
            eig = np.linalg.eigvalsh(R)
            min_eig = float(np.min(eig))
            if min_eig <= 1e-8:
                bump = abs(min_eig) + 1e-6
                R = R + bump * np.eye(R.shape[0])
                R = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))
                np.fill_diagonal(R, 1.0)

            model = {
                "tier": tier,
                "cols": list(cols),
                "corr": R,
                "supports": supports,
                "support_summaries": support_summaries,
                "train_rows": int(X.shape[0]),
                "corr_shrink": float(shrink),
                "min_eig_before_guard": min_eig,
            }

            meta = {
                "tier": tier,
                "cols": list(cols),
                "fit_available": True,
                "train_rows": int(X.shape[0]),
                "corr_shape": [int(R.shape[0]), int(R.shape[1])],
                "corr_abs_mean_offdiag": (
                    float(np.mean(np.abs(R[np.triu_indices_from(R, k=1)])))
                    if R.shape[0] > 1 else 0.0
                ),
                "corr_shrink": float(shrink),
                "min_eig_before_guard": min_eig,
                "reason": "available",
            }

            return model, meta


        def _sample_tier(model: dict, mask_dict: dict, N: int, split_name: str):
            tier = str(model["tier"])
            cols = list(model["cols"])

            active = _mask_for_tier(mask_dict, tier, N, missing_policy="error")
            n = int(active.sum())

            out = {c: np.full(N, np.nan, dtype=np.float32) for c in cols}
            gen_rows = []

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_5_gaussian_copula",
                    split_name,
                    tier,
                    "|".join(cols),
                    seed,
                    base_seed=seed + 10050,
                )
            )

            if n > 0:
                R = np.asarray(model["corr"], dtype=np.float64)
                Z = rng_local.multivariate_normal(mean=np.zeros(len(cols)), cov=R, size=n)
                U = np.clip(norm.cdf(Z), 1e-6, 1.0 - 1e-6)

                for j, c in enumerate(cols):
                    support = np.asarray(model["supports"][c], dtype=np.float64)
                    vals = np.quantile(support, U[:, j])

                    arr = np.full(N, np.nan, dtype=np.float32)
                    arr[active] = vals.astype(np.float32)
                    out[c] = _postprocess_protocol_values(c, arr)

            df_out = pd.DataFrame(out)

            for c in cols:
                x = pd.to_numeric(df_out[c], errors="coerce").to_numpy(dtype=np.float64)
                finite_active = int(np.isfinite(x[active]).sum())
                finite_inactive = int(np.isfinite(x[~active]).sum())

                if n > 0 and finite_active != n:
                    raise RuntimeError(
                        f"[Cell10.5] {split_name}/{tier}.{c}: active finite mismatch {finite_active} vs {n}"
                    )

                if finite_inactive != 0:
                    raise RuntimeError(
                        f"[Cell10.5] {split_name}/{tier}.{c}: inactive finite values {finite_inactive}"
                    )

                xa = x[active]

                gen_rows.append({
                    "split": split_name,
                    "tier": tier,
                    "col": c,
                    "N": int(N),
                    "active_n": int(n),
                    "active_rate": float(np.mean(active)) if active.size else np.nan,
                    "finite_active": int(finite_active),
                    "finite_inactive": int(finite_inactive),
                    "generated_mean_active": _safe_mean(xa),
                    "generated_var_active": _safe_var(xa),
                    "generated_nonzero_rate_active": _nonzero_rate(xa),
                    "generated_transition_rate_active": _transition_rate(xa),
                    "train_rows": int(model["train_rows"]),
                })

            return df_out, pd.DataFrame(gen_rows)

        # --------------------------------------------------
        # 5) Fit per-tier models
        # --------------------------------------------------
        models = {}
        fit_meta_rows = []

        for tier, cols in sorted(tier_cols.items()):
            model, meta = _fit_tier_copula(tier, cols)
            fit_meta_rows.append(meta)

            if model is not None:
                models[tier] = model

        fit_audit_path = os.path.join(OUT_REP, "cell10_5_gaussian_copula_fit_audit_v2_THESIS.csv")
        pd.DataFrame(fit_meta_rows).to_csv(fit_audit_path, index=False)

        if not models:
            _finalize_unavailable(
                reason="Eligible copula groups existed but failed TRAIN min-row/support fitting.",
                extra_meta={
                    "min_rows": min_rows,
                    "tier_cols": tier_cols,
                    "eligibility_audit_path": eligibility_path,
                    "fit_audit_path": fit_audit_path,
                },
            )

        else:
            # --------------------------------------------------
            # 6) Materialize VAL/TEST candidates
            # --------------------------------------------------
            val_parts = []
            test_parts = []
            generation_audits = []

            for tier, model in models.items():
                val_tier, val_audit = _sample_tier(model, CELL10_VAL_MASKS, len(df_va), "VAL")
                test_tier, test_audit = _sample_tier(model, CELL10_TEST_MASKS, len(df_te), "TEST")

                val_parts.append(val_tier)
                test_parts.append(test_tier)
                generation_audits.append(val_audit)
                generation_audits.append(test_audit)

            val_df = pd.concat(val_parts, axis=1)
            test_df = pd.concat(test_parts, axis=1)

            ordered_cols = []
            for tier in ["router", "ota", "zigbee"]:
                if tier in models:
                    ordered_cols.extend(models[tier]["cols"])
            ordered_cols = [c for c in ordered_cols if c in val_df.columns]

            val_df = val_df.loc[:, ordered_cols]
            test_df = test_df.loc[:, ordered_cols]

            val_path = os.path.join(OUT_PORT, "candidate_gaussian_copula_VAL.parquet")
            test_path = os.path.join(OUT_PORT, "candidate_gaussian_copula_TEST.parquet")
            meta_path = os.path.join(OUT_PORT, "candidate_gaussian_copula_fit_meta_v2_THESIS.json")
            gen_audit_path = os.path.join(OUT_REP, "cell10_5_gaussian_copula_generation_audit_v2_THESIS.csv")
            audit_path = os.path.join(OUT_REP, "cell10_5_gaussian_copula_audit_v2_THESIS.json")
            contract_path = os.path.join(CONTRACT_DIR, "cell10_5_gaussian_copula_contract_v2_THESIS.json")

            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)

            gen_audit_df = pd.concat(generation_audits, axis=0, ignore_index=True)
            gen_audit_df.to_csv(gen_audit_path, index=False)

            meta = {
                "version": "cell10_5_gaussian_copula_fit_meta_v2_THESIS",
                "generator": "gaussian_copula",
                "candidate_id": "gaussian_copula_continuous_tier_groups_v2_THESIS",
                "columns": list(ordered_cols),
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "tiers": {
                    t: {
                        "cols": list(m["cols"]),
                        "train_rows": int(m["train_rows"]),
                        "corr_shape": list(np.asarray(m["corr"]).shape),
                        "corr_shrink": float(m["corr_shrink"]),
                        "support_summaries": m["support_summaries"],
                    }
                    for t, m in models.items()
                },
            }
            _write_atomic_json(meta_path, meta)

            rec = _cell10_register_portfolio_candidate({
                "generator": "gaussian_copula",
                "candidate_id": "gaussian_copula_continuous_tier_groups_v2_THESIS",
                "columns": list(ordered_cols),
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
                "role": "continuous_correlation_preservation_candidate",
                "eligibility": {
                    "min_cols": int(min_cols),
                    "min_rows": int(min_rows),
                    "min_unique": int(min_unique),
                    "eligible_tiers": list(models.keys()),
                    "eligibility_audit_path": eligibility_path,
                },
            })

            audit = {
                "version": "cell10_5_gaussian_copula_candidate_v2_THESIS",
                "candidate_available": True,
                "registered_candidate": rec,
                "candidate_id": "gaussian_copula_continuous_tier_groups_v2_THESIS",
                "eligible_tiers": list(models.keys()),
                "eligible_cols": list(ordered_cols),
                "eligible_cols_n": int(len(ordered_cols)),
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
                    "Gaussian Copula is an active predefined portfolio candidate for "
                    "eligible continuous/semi-continuous protocol columns. It is not final "
                    "evidence unless selected later by the downstream VAL-only selector "
                    "against A1."
                ),
                "ts_unix": float(time.time()),
            }

            _write_atomic_json(audit_path, audit)
            _write_atomic_json(contract_path, audit)

            globals()["CELL10_5_COPULA_AUDIT"] = audit
            globals()["CELL10_5_COPULA_AUDIT_PATH"] = audit_path
            globals()["CELL10_5_COPULA_CONTRACT_PATH"] = contract_path

            if "RUN_META" in globals():
                RUN_META.setdefault("portfolio_candidate_audits", {})
                RUN_META["portfolio_candidate_audits"]["gaussian_copula"] = audit

            log(
                "[Cell10.5] Registered Gaussian Copula candidate | "
                f"cols={len(ordered_cols)} | tiers={list(models.keys())} | "
                f"VAL={val_df.shape} | TEST={test_df.shape}"
            )
            log(f"[Cell10.5] Saved eligibility audit: {eligibility_path}")
            log(f"[Cell10.5] Saved fit audit: {fit_audit_path}")
            log(f"[Cell10.5] Saved generation audit: {gen_audit_path}")
            log(f"[Cell10.5] Saved audit: {audit_path}")
            log(f"[Cell10.5] Saved contract: {contract_path}")

log("--- END: Cell 10.5 — Gaussian Copula continuous-correlation candidate (v2-THESIS) ---")
