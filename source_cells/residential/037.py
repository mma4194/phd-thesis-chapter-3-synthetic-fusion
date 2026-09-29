# ==========================================================
# CELL 10.6 — CTGAN / TVAE optional tabular candidate — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Optionally generate mixed tabular protocol candidates using SDV CTGAN/TVAE
#   if explicitly enabled and installed.
# - Fit on TRAIN active rows only.
# - Emit VAL and TEST-horizon candidate artifacts for downstream VAL-only selection.
#
# Scientific contract:
# - This backend is optional and disabled by default.
# - If disabled or unavailable, it registers an explicit unavailable-backend audit.
# - If enabled, fitting uses TRAIN values only.
# - VAL candidate is for downstream selection.
# - TEST candidate is for final application/QA only.
# - TEST real values are never used for fitting, selection, repair, or promotion.
# - This cell does not select anything and does not modify A0/A1/A2 artifacts.
# ==========================================================

log("--- START: Cell 10.6 — CTGAN / TVAE optional tabular candidate (v2-THESIS) ---")

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
    "_mask_for_tier", "_postprocess_protocol_values",
    "_is_countlike", "_safe_mean", "_safe_var",
    "_nonzero_rate", "_transition_rate",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.6] Missing prerequisites: {missing}. Run Cell 10.1 first.")

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
        raise RuntimeError(f"[Cell10.6] Clean CTGAN/TVAE candidate generation forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.6] Unexpected selection_split_policy. "
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


def _finalize_unavailable(reason: str, extra_meta: dict = None):
    _cell10_register_unavailable_backend(
        "ctgan_tvae",
        reason=reason,
        meta={
            "source": "Cell10.6",
            "version": "cell10_6_ctgan_tvae_optional_v2_THESIS",
            **dict(extra_meta or {}),
        },
    )

    audit = {
        "version": "cell10_6_ctgan_tvae_optional_candidate_v2_THESIS",
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
            "CTGAN/TVAE is an optional predefined backend. It contributes no active "
            "candidate in this run when disabled, unavailable, or unsupported by the "
            "TRAIN-only eligibility checks."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_6_ctgan_tvae_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_6_ctgan_tvae_contract_v2_THESIS.json")

    _write_atomic_json(audit_path, audit)
    _write_atomic_json(contract_path, audit)

    globals()["CELL10_6_CTGAN_TVAE_AUDIT"] = audit
    globals()["CELL10_6_CTGAN_TVAE_AUDIT_PATH"] = audit_path
    globals()["CELL10_6_CTGAN_TVAE_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["ctgan_tvae"] = audit

    log(f"[Cell10.6] CTGAN/TVAE unavailable: {reason}")
    log(f"[Cell10.6] Saved audit: {audit_path}")
    log(f"[Cell10.6] Saved contract: {contract_path}")


# ----------------------------------------------------------
# 2) Disabled-by-default gate
# ----------------------------------------------------------
enable = bool(CFG.get("cell10_enable_ctgan_tvae", False))
if not enable:
    _finalize_unavailable(
        reason="Disabled by CFG['cell10_enable_ctgan_tvae']=False.",
        extra_meta={
            "how_to_enable": "Set CFG['cell10_enable_ctgan_tvae']=True before running Cell 10.6.",
            "default_policy": "disabled_optional_backend",
        },
    )

else:
    # ------------------------------------------------------
    # 3) Optional SDV import
    # ------------------------------------------------------
    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    except Exception as e:
        _finalize_unavailable(
            reason=f"SDV backend unavailable: {type(e).__name__}: {e}",
            extra_meta={"required_package": "sdv"},
        )

    else:
        # --------------------------------------------------
        # 4) Eligibility and configuration
        # --------------------------------------------------
        max_cols_per_tier = int(CFG.get("cell10_ctgan_max_cols_per_tier", 18))
        min_rows = int(CFG.get("cell10_ctgan_min_train_rows", 1000))
        epochs = int(CFG.get("cell10_ctgan_epochs", 50))
        model_kind = str(CFG.get("cell10_ctgan_model", "tvae")).lower()

        if max_cols_per_tier < 2:
            raise RuntimeError(f"[Cell10.6] max_cols_per_tier must be >=2, got {max_cols_per_tier}.")
        if min_rows <= 0:
            raise RuntimeError(f"[Cell10.6] min_rows must be positive, got {min_rows}.")
        if epochs <= 0:
            raise RuntimeError(f"[Cell10.6] epochs must be positive, got {epochs}.")
        if model_kind not in {"ctgan", "tvae"}:
            raise RuntimeError(f"[Cell10.6] Unsupported model kind: {model_kind!r}.")

        proto_col_set = set(map(str, CELL10_PROTO_COLS))
        selected = {t: [] for t in ["router", "ota", "zigbee"]}
        eligibility_rows = []

        for _, row in CELL10_FAMILY_DF.iterrows():
            c = str(row["col"])
            tier = str(row["tier"])

            schema_ok = (
                c in proto_col_set
                and c in df_tr.columns
                and c in df_va.columns
                and c in df_te.columns
                and tier in selected
            )
            unique_like = bool(row.get("unique_like", False))

            eligible_pre_cap = bool(schema_ok and not unique_like)

            eligibility_rows.append({
                "col": c,
                "tier": tier,
                "schema_ok": bool(schema_ok),
                "unique_like": bool(unique_like),
                "eligible_pre_cap": bool(eligible_pre_cap),
                "reason": "eligible_pre_cap" if eligible_pre_cap else "schema_missing_or_unique_like_or_unsupported_tier",
            })

            if eligible_pre_cap:
                selected[tier].append(c)

        # Deterministic bounded columns per tier.
        selected = {
            t: list(dict.fromkeys(cols))[:max_cols_per_tier]
            for t, cols in selected.items()
            if len(cols) >= 2
        }

        eligibility_path = os.path.join(OUT_REP, "cell10_6_ctgan_tvae_eligibility_v2_THESIS.csv")
        pd.DataFrame(eligibility_rows).to_csv(eligibility_path, index=False)

        if not selected:
            _finalize_unavailable(
                reason="No tier had enough non-unique protocol columns for CTGAN/TVAE fitting.",
                extra_meta={
                    "min_rows": min_rows,
                    "max_cols_per_tier": max_cols_per_tier,
                    "eligibility_audit_path": eligibility_path,
                },
            )

        else:
            # ----------------------------------------------
            # 5) Fit and sample tier-local SDV models
            # ----------------------------------------------
            def _train_active_mask(tier: str) -> np.ndarray:
                c = f"{tier}__obs_present"
                if c not in df_tr.columns:
                    raise RuntimeError(f"[Cell10.6] TRAIN missing observability column: {c}")
                v = pd.to_numeric(df_tr[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                return v > 0.5


            def _prepare_train_frame(tier: str, cols: list):
                active = _train_active_mask(tier)
                train = df_tr.loc[active, cols].apply(pd.to_numeric, errors="coerce")
                train = train.replace([np.inf, -np.inf], np.nan)

                usable_cols = []
                col_meta = {}

                for c in cols:
                    x = pd.to_numeric(train[c], errors="coerce")
                    finite_n = int(x.notna().sum())
                    unique_n = int(x.nunique(dropna=True))

                    if finite_n >= min_rows and unique_n >= 2:
                        usable_cols.append(c)
                        col_meta[c] = {
                            "finite_n": finite_n,
                            "unique_n": unique_n,
                            "countlike": bool(_is_countlike(c)),
                        }

                if len(usable_cols) < 2:
                    return None, {
                        "tier": tier,
                        "fit_available": False,
                        "cols_requested": list(cols),
                        "usable_cols": list(usable_cols),
                        "reason": "fewer_than_two_usable_columns",
                    }

                train = train[usable_cols].copy()

                for c in usable_cols:
                    x = pd.to_numeric(train[c], errors="coerce")
                    fill = 0.0 if int(x.notna().sum()) == 0 else float(x.median())
                    x = x.fillna(fill).clip(lower=0.0)

                    if _is_countlike(c):
                        x = np.rint(x).astype(int)

                    train[c] = x

                if len(train) < min_rows:
                    return None, {
                        "tier": tier,
                        "fit_available": False,
                        "cols_requested": list(cols),
                        "usable_cols": list(usable_cols),
                        "train_rows": int(len(train)),
                        "reason": f"train_rows < {min_rows}",
                    }

                return train, {
                    "tier": tier,
                    "fit_available": True,
                    "cols_requested": list(cols),
                    "usable_cols": list(usable_cols),
                    "train_rows": int(len(train)),
                    "column_meta": col_meta,
                    "reason": "available",
                }


            def _fit_sdv_model(train: pd.DataFrame):
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(train)

                if model_kind == "ctgan":
                    synth = CTGANSynthesizer(metadata, epochs=epochs)
                else:
                    synth = TVAESynthesizer(metadata, epochs=epochs)

                synth.fit(train)
                return synth


            def _sample_for_tier(synth, usable_cols: list, tier: str, mask_dict: dict, N: int, split_name: str):
                active_h = _mask_for_tier(mask_dict, tier, N, missing_policy="error")
                n = int(active_h.sum())

                out = {c: np.full(N, np.nan, dtype=np.float32) for c in usable_cols}

                if n > 0:
                    samp = synth.sample(num_rows=n)

                    for c in usable_cols:
                        vals = pd.to_numeric(samp[c], errors="coerce").to_numpy(dtype=np.float32)
                        arr = np.full(N, np.nan, dtype=np.float32)
                        arr[active_h] = vals
                        out[c] = _postprocess_protocol_values(c, arr)

                df_out = pd.DataFrame(out)
                gen_rows = []

                for c in usable_cols:
                    x = pd.to_numeric(df_out[c], errors="coerce").to_numpy(dtype=np.float64)

                    finite_active = int(np.isfinite(x[active_h]).sum())
                    finite_inactive = int(np.isfinite(x[~active_h]).sum())

                    if n > 0 and finite_active != n:
                        raise RuntimeError(
                            f"[Cell10.6] {split_name}/{tier}.{c}: active finite mismatch {finite_active} vs {n}"
                        )

                    if finite_inactive != 0:
                        raise RuntimeError(
                            f"[Cell10.6] {split_name}/{tier}.{c}: inactive finite values {finite_inactive}"
                        )

                    xa = x[active_h]

                    gen_rows.append({
                        "split": split_name,
                        "tier": tier,
                        "col": c,
                        "N": int(N),
                        "active_n": int(n),
                        "active_rate": float(np.mean(active_h)) if active_h.size else np.nan,
                        "finite_active": int(finite_active),
                        "finite_inactive": int(finite_inactive),
                        "generated_mean_active": _safe_mean(xa),
                        "generated_var_active": _safe_var(xa),
                        "generated_nonzero_rate_active": _nonzero_rate(xa),
                        "generated_transition_rate_active": _transition_rate(xa),
                    })

                return df_out, pd.DataFrame(gen_rows)


            all_val = []
            all_test = []
            fit_meta = {}
            fit_rows = []
            generation_audits = []

            for tier, cols in sorted(selected.items()):
                train, prep_meta = _prepare_train_frame(tier, cols)
                fit_rows.append(prep_meta)

                if train is None:
                    log(f"[Cell10.6][WARN] Skipping tier={tier}: {prep_meta.get('reason')}")
                    continue

                try:
                    synth = _fit_sdv_model(train)
                except Exception as e:
                    fit_rows[-1]["fit_available"] = False
                    fit_rows[-1]["reason"] = f"fit_failed_{type(e).__name__}: {e}"
                    log(f"[Cell10.6][WARN] SDV fit failed for tier={tier}: {type(e).__name__}: {e}")
                    continue

                usable_cols = list(train.columns)

                val_tier, val_audit = _sample_for_tier(
                    synth,
                    usable_cols,
                    tier,
                    CELL10_VAL_MASKS,
                    len(df_va),
                    "VAL",
                )
                test_tier, test_audit = _sample_for_tier(
                    synth,
                    usable_cols,
                    tier,
                    CELL10_TEST_MASKS,
                    len(df_te),
                    "TEST",
                )

                all_val.append(val_tier)
                all_test.append(test_tier)
                generation_audits.append(val_audit)
                generation_audits.append(test_audit)

                fit_meta[tier] = {
                    "model": model_kind,
                    "cols": usable_cols,
                    "train_rows": int(len(train)),
                    "epochs": int(epochs),
                }

            fit_audit_path = os.path.join(OUT_REP, "cell10_6_ctgan_tvae_fit_audit_v2_THESIS.csv")
            pd.DataFrame(fit_rows).to_csv(fit_audit_path, index=False)

            if not all_val:
                _finalize_unavailable(
                    reason="Backend available but no tier/model met TRAIN fitting requirements.",
                    extra_meta={
                        "min_rows": min_rows,
                        "max_cols_per_tier": max_cols_per_tier,
                        "model_kind": model_kind,
                        "epochs": epochs,
                        "eligibility_audit_path": eligibility_path,
                        "fit_audit_path": fit_audit_path,
                    },
                )

            else:
                # ------------------------------------------
                # 6) Persist candidate artifacts
                # ------------------------------------------
                val_df = pd.concat(all_val, axis=1)
                test_df = pd.concat(all_test, axis=1)

                ordered_cols = []
                for tier in ["router", "ota", "zigbee"]:
                    if tier in fit_meta:
                        ordered_cols.extend(fit_meta[tier]["cols"])
                ordered_cols = [c for c in ordered_cols if c in val_df.columns]

                val_df = val_df.loc[:, ordered_cols]
                test_df = test_df.loc[:, ordered_cols]

                val_path = os.path.join(OUT_PORT, f"candidate_{model_kind}_VAL.parquet")
                test_path = os.path.join(OUT_PORT, f"candidate_{model_kind}_TEST.parquet")
                meta_path = os.path.join(OUT_PORT, f"candidate_{model_kind}_fit_meta_v2_THESIS.json")
                gen_audit_path = os.path.join(OUT_REP, "cell10_6_ctgan_tvae_generation_audit_v2_THESIS.csv")
                audit_path = os.path.join(OUT_REP, "cell10_6_ctgan_tvae_audit_v2_THESIS.json")
                contract_path = os.path.join(CONTRACT_DIR, "cell10_6_ctgan_tvae_contract_v2_THESIS.json")

                val_df.to_parquet(val_path, index=False)
                test_df.to_parquet(test_path, index=False)

                gen_audit_df = pd.concat(generation_audits, axis=0, ignore_index=True)
                gen_audit_df.to_csv(gen_audit_path, index=False)

                fit_meta_out = {
                    "version": "cell10_6_ctgan_tvae_fit_meta_v2_THESIS",
                    "generator": "ctgan_tvae",
                    "candidate_id": f"{model_kind}_mixed_tabular_v2_THESIS",
                    "backend": "sdv",
                    "model_kind": model_kind,
                    "epochs": int(epochs),
                    "fit_split": "TRAIN",
                    "selection_split": "VAL",
                    "test_used_for_selection": False,
                    "tiers": fit_meta,
                    "columns": list(ordered_cols),
                }
                _write_atomic_json(meta_path, fit_meta_out)

                rec = _cell10_register_portfolio_candidate({
                    "generator": "ctgan_tvae",
                    "candidate_id": f"{model_kind}_mixed_tabular_v2_THESIS",
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
                    "role": "mixed_tabular_marginal_correlation_candidate",
                    "backend": "sdv",
                    "model_kind": model_kind,
                    "eligibility": {
                        "max_cols_per_tier": int(max_cols_per_tier),
                        "min_rows": int(min_rows),
                        "epochs": int(epochs),
                        "eligibility_audit_path": eligibility_path,
                    },
                })

                audit = {
                    "version": "cell10_6_ctgan_tvae_optional_candidate_v2_THESIS",
                    "candidate_available": True,
                    "registered_candidate": rec,
                    "candidate_id": f"{model_kind}_mixed_tabular_v2_THESIS",
                    "backend": "sdv",
                    "model_kind": model_kind,
                    "epochs": int(epochs),
                    "eligible_tiers": list(fit_meta.keys()),
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
                        "CTGAN/TVAE is an optional predefined mixed-tabular portfolio "
                        "candidate. It is not final evidence unless selected later by the "
                        "downstream VAL-only selector against A1."
                    ),
                    "ts_unix": float(time.time()),
                }

                _write_atomic_json(audit_path, audit)
                _write_atomic_json(contract_path, audit)

                globals()["CELL10_6_CTGAN_TVAE_AUDIT"] = audit
                globals()["CELL10_6_CTGAN_TVAE_AUDIT_PATH"] = audit_path
                globals()["CELL10_6_CTGAN_TVAE_CONTRACT_PATH"] = contract_path

                if "RUN_META" in globals():
                    RUN_META.setdefault("portfolio_candidate_audits", {})
                    RUN_META["portfolio_candidate_audits"]["ctgan_tvae"] = audit

                log(
                    f"[Cell10.6] Registered {model_kind.upper()} candidate | "
                    f"cols={len(ordered_cols)} | tiers={list(fit_meta.keys())} | "
                    f"VAL={val_df.shape} | TEST={test_df.shape}"
                )
                log(f"[Cell10.6] Saved eligibility audit: {eligibility_path}")
                log(f"[Cell10.6] Saved fit audit: {fit_audit_path}")
                log(f"[Cell10.6] Saved generation audit: {gen_audit_path}")
                log(f"[Cell10.6] Saved audit: {audit_path}")
                log(f"[Cell10.6] Saved contract: {contract_path}")

log("--- END: Cell 10.6 — CTGAN / TVAE optional tabular candidate (v2-THESIS) ---")
