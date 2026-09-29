# ==========================================================
# CELL 10.4s — Router support/quantile candidate portfolio — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Generate TRAIN-only support/quantile candidates for router support/magnitude
#   columns:
#     primary:   router__bytes_total
#     secondary: router__dhcp_pkt, router__dns_rcode3_nx,
#                router__icmp_pkt, router__tcp_rst
# - Preserve empirical TRAIN support and quantile structure.
# - Emit VAL and TEST-horizon candidate artifacts for downstream VAL-only selection.
#
# Scientific contract:
# - Fit TRAIN only.
# - Generate VAL and TEST candidates from TRAIN support/quantile caches and masks.
# - Do not read real TEST values.
# - Do not use VAL for fitting.
# - Do not select anything in this cell.
# - Do not modify A0/A1/A2 artifacts.
# - Do not retry router__pkt_total or router__udp_pkt here.
# - Not final evidence unless selected downstream on VAL against A1.
#
# Public-release caveat:
# - These candidates sample TRAIN empirical/quantile support. If selected for any
#   public-release scope, Q6/copy-risk/no-copy audits are mandatory.
# ==========================================================

log("--- START: Cell 10.4s — Router support/quantile candidate portfolio (v2-THESIS) ---")

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
    raise RuntimeError(f"[Cell10.4s] Missing prerequisites: {missing}. Run Cells 10.1–10.4r first.")

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
        raise RuntimeError(f"[Cell10.4s] Clean router support candidate forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.4s] Unexpected selection_split_policy. "
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
# 1) Targets from predefined router support/magnitude plan
# ----------------------------------------------------------
PRIMARY_TARGETS_10_4S = [
    "router__bytes_total",
]

SECONDARY_TARGETS_10_4S = [
    "router__dhcp_pkt",
    "router__dns_rcode3_nx",
    "router__icmp_pkt",
    "router__tcp_rst",
]

# Explicitly excluded because 10.4r router service-block replay already targets/diagnoses them.
EXCLUDED_FAILED_BLOCK_REPLAY_COLS_10_4S = [
    "router__pkt_total",
    "router__udp_pkt",
]

ALL_TARGETS_10_4S = PRIMARY_TARGETS_10_4S + SECONDARY_TARGETS_10_4S
proto_col_set = set(map(str, CELL10_PROTO_COLS))

TARGET_COLS = [
    c for c in ALL_TARGETS_10_4S
    if c in proto_col_set
    and c in df_tr.columns
    and c in df_va.columns
    and c in df_te.columns
]

PRIMARY_PRESENT = [c for c in PRIMARY_TARGETS_10_4S if c in TARGET_COLS]
SECONDARY_PRESENT = [c for c in SECONDARY_TARGETS_10_4S if c in TARGET_COLS]

eligibility_rows = []
for c in ALL_TARGETS_10_4S:
    eligibility_rows.append({
        "col": c,
        "target_role": "primary" if c in PRIMARY_TARGETS_10_4S else "secondary",
        "expected": True,
        "in_CELL10_PROTO_COLS": c in proto_col_set,
        "in_df_tr_schema": c in df_tr.columns,
        "in_df_va_schema": c in df_va.columns,
        "in_df_te_schema": c in df_te.columns,
        "selected_target": c in TARGET_COLS,
    })

for c in EXCLUDED_FAILED_BLOCK_REPLAY_COLS_10_4S:
    eligibility_rows.append({
        "col": c,
        "target_role": "explicitly_excluded_failed_block_replay",
        "expected": False,
        "in_CELL10_PROTO_COLS": c in proto_col_set,
        "in_df_tr_schema": c in df_tr.columns,
        "in_df_va_schema": c in df_va.columns,
        "in_df_te_schema": c in df_te.columns,
        "selected_target": False,
    })

eligibility_path = os.path.join(
    OUT_REP,
    "cell10_4s_router_support_quantile_eligibility_v2_THESIS.csv",
)
pd.DataFrame(eligibility_rows).to_csv(eligibility_path, index=False)

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
        "router_support_quantile",
        reason=reason,
        meta={
            "source": "Cell10.4s",
            "version": "cell10_4s_router_support_quantile_v2_THESIS",
            "primary_targets": PRIMARY_TARGETS_10_4S,
            "secondary_targets": SECONDARY_TARGETS_10_4S,
            "target_cols": TARGET_COLS,
            "excluded_failed_block_replay_cols": EXCLUDED_FAILED_BLOCK_REPLAY_COLS_10_4S,
            "eligibility_audit_path": eligibility_path,
            **dict(extra_meta or {}),
        },
    )

    audit = {
        "version": "cell10_4s_router_support_quantile_candidate_v2_THESIS",
        "candidate_available": False,
        "reason": reason,
        "target_cols": list(TARGET_COLS),
        "primary_targets": PRIMARY_PRESENT,
        "secondary_targets": SECONDARY_PRESENT,
        "registered_candidate_ids": [],
        "variants_n": 0,
        "excluded_failed_block_replay_cols": EXCLUDED_FAILED_BLOCK_REPLAY_COLS_10_4S,
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
            "Router support/quantile candidate unavailable because required schema "
            "or TRAIN-only support conditions were not satisfied."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_4s_router_support_quantile_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_4s_router_support_quantile_contract_v2_THESIS.json")

    _write_atomic_json(audit_path, audit)
    _write_atomic_json(contract_path, audit)

    globals()["CELL10_4S_ROUTER_SUPPORT_AUDIT"] = audit
    globals()["CELL10_4S_ROUTER_SUPPORT_AUDIT_PATH"] = audit_path
    globals()["CELL10_4S_ROUTER_SUPPORT_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["router_support_quantile"] = audit

    log(f"[Cell10.4s] Router support/quantile unavailable: {reason}")
    log(f"[Cell10.4s] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.4s] Saved audit: {audit_path}")
    log(f"[Cell10.4s] Saved contract: {contract_path}")


if not TARGET_COLS:
    _finalize_unavailable(
        reason="No router support/quantile target columns found in required schemas.",
        extra_meta={
            "primary_targets": PRIMARY_TARGETS_10_4S,
            "secondary_targets": SECONDARY_TARGETS_10_4S,
        },
    )

else:
    # ------------------------------------------------------
    # 3) Helper functions
    # ------------------------------------------------------
    def _router_train_active_mask() -> np.ndarray:
        if "router__obs_present" not in df_tr.columns:
            raise RuntimeError("[Cell10.4s] TRAIN missing router__obs_present.")
        v = pd.to_numeric(df_tr["router__obs_present"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        return v > 0.5


    def _finite_train_values(col: str) -> np.ndarray:
        active = _router_train_active_mask()
        x = pd.to_numeric(df_tr.loc[active, col], errors="coerce").to_numpy(dtype=np.float32)
        x = x[np.isfinite(x)]
        x = _postprocess_protocol_values(col, x)
        x = x[np.isfinite(x)]
        return x.astype(np.float32, copy=False)


    def _basic_profile(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]

        if x.size == 0:
            return {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "q01": np.nan,
                "q05": np.nan,
                "q25": np.nan,
                "q50": np.nan,
                "q75": np.nan,
                "q95": np.nan,
                "q99": np.nan,
                "max": np.nan,
                "nonzero_rate": np.nan,
                "unique_n": 0,
                "transition_rate": np.nan,
            }

        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "q01": float(np.quantile(x, 0.01)),
            "q05": float(np.quantile(x, 0.05)),
            "q25": float(np.quantile(x, 0.25)),
            "q50": float(np.quantile(x, 0.50)),
            "q75": float(np.quantile(x, 0.75)),
            "q95": float(np.quantile(x, 0.95)),
            "q99": float(np.quantile(x, 0.99)),
            "max": float(np.max(x)),
            "nonzero_rate": float(np.mean(x > 0)),
            "unique_n": int(np.unique(np.rint(x)).size),
            "transition_rate": float(np.mean(np.abs(np.diff(x)) > 1e-9)) if x.size > 1 else np.nan,
        }


    def _support_values(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]

        if x.size == 0:
            return np.asarray([0.0], dtype=np.float32)

        vals = np.unique(np.maximum(np.rint(x), 0.0)).astype(np.float32)
        return vals


    def _empirical_probs(x):
        vals = _support_values(x)
        xx = np.asarray(x, dtype=np.float64)
        xx = xx[np.isfinite(xx)]
        xx = np.maximum(np.rint(xx), 0.0)

        if vals.size == 0 or xx.size == 0:
            return np.asarray([0.0], dtype=np.float32), np.asarray([1.0], dtype=np.float64)

        counts = np.asarray([np.sum(xx == float(v)) for v in vals], dtype=np.float64)
        counts = counts + 1e-9
        probs = counts / counts.sum()

        return vals.astype(np.float32), probs.astype(np.float64)


    def _make_quantile_grid(x, n_q: int = 4096):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]

        if x.size == 0:
            return np.asarray([0.0], dtype=np.float32)

        qs = np.linspace(0.0, 1.0, int(max(32, n_q)))
        vals = np.quantile(x, qs)
        vals = np.maximum(np.rint(vals), 0.0)

        return vals.astype(np.float32)


    def _sample_empirical_support(rng_local: np.random.Generator, x, n: int):
        vals, probs = _empirical_probs(x)
        if vals.size == 0:
            return np.zeros(int(n), dtype=np.float32)
        return rng_local.choice(vals, size=int(n), replace=True, p=probs).astype(np.float32)


    def _sample_quantile_smoothed(rng_local: np.random.Generator, x, n: int):
        grid = _make_quantile_grid(x, n_q=int(CFG.get("cell10_4s_quantile_grid_n", 4096)))
        if grid.size == 0:
            return np.zeros(int(n), dtype=np.float32)

        idx = rng_local.integers(0, grid.size, size=int(n))
        vals = grid[idx].astype(np.float32)

        # Deliberately snaps to the quantile grid, which itself is rounded to
        # observed TRAIN support/magnitude scale. It does not use VAL/TEST values.
        return vals


    def _sample_tail_aware(rng_local: np.random.Generator, x, n: int, tail_prob: float = 0.20):
        x = np.asarray(x, dtype=np.float32)
        x = x[np.isfinite(x)]

        n = int(n)
        if x.size == 0:
            return np.zeros(n, dtype=np.float32)

        q95 = float(np.quantile(x, 0.95))
        body = x[x <= q95]
        tail = x[x > q95]

        if body.size == 0:
            body = x
        if tail.size == 0:
            tail = x

        choose_tail = rng_local.random(n) < float(tail_prob)
        out = np.zeros(n, dtype=np.float32)

        if int((~choose_tail).sum()) > 0:
            out[~choose_tail] = rng_local.choice(body, size=int((~choose_tail).sum()), replace=True)
        if int(choose_tail.sum()) > 0:
            out[choose_tail] = rng_local.choice(tail, size=int(choose_tail.sum()), replace=True)

        return np.maximum(np.rint(out), 0).astype(np.float32)


    def _make_regime_labels_from_train_index(n: int):
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.int64)

        edges = np.linspace(0, n, 5)
        edges = np.round(edges).astype(int)
        labels = np.zeros(n, dtype=np.int64)

        for i in range(len(edges) - 1):
            labels[edges[i]:edges[i + 1]] = i

        return labels


    def _sample_regime_quantile(rng_local: np.random.Generator, x, n: int):
        x = np.asarray(x, dtype=np.float32)
        x = x[np.isfinite(x)]

        n = int(n)
        if x.size == 0:
            return np.zeros(n, dtype=np.float32)

        labels = _make_regime_labels_from_train_index(len(x))
        regimes = sorted(set(labels.tolist()))

        pools = {}
        for r in regimes:
            pool = x[labels == int(r)]
            if pool.size == 0:
                pool = x
            pools[int(r)] = pool

        out = np.zeros(n, dtype=np.float32)
        edges = np.linspace(0, n, len(regimes) + 1)
        edges = np.round(edges).astype(int)

        for i, r in enumerate(regimes):
            a = int(edges[i])
            b = int(edges[i + 1])

            if b <= a:
                continue

            out[a:b] = _sample_quantile_smoothed(rng_local, pools[int(r)], b - a)

        return np.maximum(np.rint(out), 0).astype(np.float32)


    def _generate_split_for_cols(
        mask_dict: dict,
        N: int,
        split_name: str,
        cols: list,
        fit_values: dict,
        variant_kind: str,
        variant_meta: dict,
    ):
        N = int(N)
        if N <= 0:
            raise RuntimeError(f"[Cell10.4s] {split_name}/{variant_meta.get('candidate_id')}: N must be positive.")

        active = _mask_for_tier(mask_dict, "router", N, missing_policy="error")
        n_active = int(active.sum())

        rng_local = np.random.default_rng(
            _stable_seed_from_parts(
                "cell10_4s_router_support_quantile",
                split_name,
                variant_meta.get("candidate_id"),
                "|".join(cols),
                seed,
                base_seed=seed + 10060,
            )
        )

        out = {}
        gen_rows = []

        for c in cols:
            arr = np.full(N, np.nan, dtype=np.float32)

            if n_active > 0:
                x = fit_values[c]

                if variant_kind == "empirical_support":
                    vals = _sample_empirical_support(rng_local, x, n_active)
                elif variant_kind == "quantile_smoothed":
                    vals = _sample_quantile_smoothed(rng_local, x, n_active)
                elif variant_kind == "tail_aware":
                    vals = _sample_tail_aware(
                        rng_local,
                        x,
                        n_active,
                        tail_prob=float(variant_meta.get("tail_prob", 0.20)),
                    )
                elif variant_kind == "regime_quantile":
                    vals = _sample_regime_quantile(rng_local, x, n_active)
                else:
                    raise RuntimeError(f"[Cell10.4s] Unknown variant kind: {variant_kind}")

                arr[active] = vals

            arr = _postprocess_protocol_values(c, arr)
            out[c] = arr

            finite_active = int(np.isfinite(arr[active]).sum())
            finite_inactive = int(np.isfinite(arr[~active]).sum())

            if n_active > 0 and finite_active != n_active:
                raise RuntimeError(
                    f"[Cell10.4s] {split_name}/{variant_meta.get('candidate_id')}.{c}: "
                    f"active finite mismatch {finite_active} vs {n_active}"
                )

            if finite_inactive != 0:
                raise RuntimeError(
                    f"[Cell10.4s] {split_name}/{variant_meta.get('candidate_id')}.{c}: "
                    f"inactive finite values {finite_inactive}"
                )

            xa = arr[active]

            gen_rows.append({
                "split": split_name,
                "candidate_id": str(variant_meta.get("candidate_id")),
                "variant_name": str(variant_meta.get("variant_name")),
                "kind": str(variant_kind),
                "col": c,
                "tier": "router",
                "N": int(N),
                "active_n": int(n_active),
                "active_rate": float(np.mean(active)) if active.size else np.nan,
                "finite_active": int(finite_active),
                "finite_inactive": int(finite_inactive),
                "generated_mean_active": _safe_mean(xa),
                "generated_var_active": _safe_var(xa),
                "generated_nonzero_rate_active": _nonzero_rate(xa),
                "generated_transition_rate_active": _transition_rate(xa),
                "train_n": int(fit_meta[c]["train_n"]),
                "train_support_n": int(fit_meta[c]["support_n"]),
                "target_role": str(fit_meta[c]["target_role"]),
            })

        return pd.DataFrame(out), pd.DataFrame(gen_rows)

    # ------------------------------------------------------
    # 4) Fit TRAIN support/quantile caches
    # ------------------------------------------------------
    min_train_n = int(CFG.get("cell10_4s_min_train_n", 500))
    fit_values = {}
    fit_meta = {}
    support_rows = []

    for c in TARGET_COLS:
        x = _finite_train_values(c)

        if x.size < min_train_n:
            support_rows.append({
                "col": c,
                "target_role": "primary" if c in PRIMARY_PRESENT else "secondary",
                "fit_available": False,
                "train_n": int(x.size),
                "support_n": int(_support_values(x).size) if x.size else 0,
                "reason": f"train_n < {min_train_n}",
            })
            log(f"[Cell10.4s][WARN] Skipping {c}: train_n={x.size} < {min_train_n}")
            continue

        fit_values[c] = x
        fit_meta[c] = {
            "profile": _basic_profile(x),
            "support_n": int(_support_values(x).size),
            "train_n": int(x.size),
            "target_role": "primary" if c in PRIMARY_PRESENT else "secondary",
        }

        support_rows.append({
            "col": c,
            "target_role": "primary" if c in PRIMARY_PRESENT else "secondary",
            "fit_available": True,
            "train_n": int(x.size),
            "support_n": int(fit_meta[c]["support_n"]),
            "reason": "available",
        })

    support_path = os.path.join(
        OUT_REP,
        "cell10_4s_router_support_quantile_support_audit_v2_THESIS.csv",
    )
    pd.DataFrame(support_rows).to_csv(support_path, index=False)

    FIT_COLS = list(fit_values.keys())
    PRIMARY_FIT_COLS = [c for c in PRIMARY_PRESENT if c in fit_values]
    SECONDARY_FIT_COLS = [c for c in SECONDARY_PRESENT if c in fit_values]

    if not FIT_COLS:
        _finalize_unavailable(
            reason="No router support/quantile columns met TRAIN support requirements.",
            extra_meta={
                "target_cols": TARGET_COLS,
                "min_train_n": min_train_n,
                "support_path": support_path,
            },
        )

    else:
        log(
            "[Cell10.4s] Fitted TRAIN support/quantile caches | "
            f"fit_cols={FIT_COLS} | primary={PRIMARY_FIT_COLS} | secondary={SECONDARY_FIT_COLS}"
        )

        # --------------------------------------------------
        # 5) Predefined variants
        # --------------------------------------------------
        variant_specs = [
            {
                "generator": "router_support_quantile",
                "candidate_id": "router_support_quantile_bytes_v2_THESIS",
                "variant_name": "bytes_quantile_primary",
                "kind": "quantile_smoothed",
                "cols": PRIMARY_FIT_COLS,
                "role": "primary_router_bytes_quantile_candidate",
            },
            {
                "generator": "router_support_quantile",
                "candidate_id": "router_support_tail_aware_bytes_v2_THESIS",
                "variant_name": "bytes_tail_aware_primary",
                "kind": "tail_aware",
                "cols": PRIMARY_FIT_COLS,
                "tail_prob": float(CFG.get("cell10_4s_tail_prob_bytes", 0.20)),
                "role": "primary_router_bytes_tail_candidate",
            },
            {
                "generator": "router_support_quantile",
                "candidate_id": "router_support_regime_quantile_bytes_v2_THESIS",
                "variant_name": "bytes_regime_quantile_primary",
                "kind": "regime_quantile",
                "cols": PRIMARY_FIT_COLS,
                "role": "primary_router_bytes_regime_quantile_candidate",
            },
            {
                "generator": "router_service_support_quantile",
                "candidate_id": "router_service_support_empirical_low_volume_v2_THESIS",
                "variant_name": "low_volume_empirical_support",
                "kind": "empirical_support",
                "cols": SECONDARY_FIT_COLS,
                "role": "secondary_low_volume_router_support_candidate",
            },
            {
                "generator": "router_service_support_quantile",
                "candidate_id": "router_service_support_quantile_low_volume_v2_THESIS",
                "variant_name": "low_volume_quantile_support",
                "kind": "quantile_smoothed",
                "cols": SECONDARY_FIT_COLS,
                "role": "secondary_low_volume_router_quantile_candidate",
            },
            {
                "generator": "router_support_quantile",
                "candidate_id": "router_support_quantile_all_v2_THESIS",
                "variant_name": "all_support_quantile",
                "kind": "quantile_smoothed",
                "cols": FIT_COLS,
                "role": "all_router_support_quantile_candidate",
            },
            {
                "generator": "router_support_quantile",
                "candidate_id": "router_support_empirical_all_v2_THESIS",
                "variant_name": "all_empirical_support",
                "kind": "empirical_support",
                "cols": FIT_COLS,
                "role": "all_router_empirical_support_candidate",
            },
        ]

        registered = []
        portfolio_meta_rows = []
        all_generation_audits = []

        # --------------------------------------------------
        # 6) Materialize/register variants
        # --------------------------------------------------
        for spec0 in variant_specs:
            spec = dict(spec0)
            cols = [c for c in spec.get("cols", []) if c in fit_values]

            if not cols:
                log(f"[Cell10.4s][SKIP] No columns for candidate_id={spec['candidate_id']}")
                continue

            val_df, val_audit = _generate_split_for_cols(
                CELL10_VAL_MASKS,
                len(df_va),
                "VAL",
                cols,
                fit_values,
                variant_kind=str(spec["kind"]),
                variant_meta=spec,
            )

            test_df, test_audit = _generate_split_for_cols(
                CELL10_TEST_MASKS,
                len(df_te),
                "TEST",
                cols,
                fit_values,
                variant_kind=str(spec["kind"]),
                variant_meta=spec,
            )

            val_df = val_df.loc[:, cols]
            test_df = test_df.loc[:, cols]

            gen = str(spec["generator"])
            cid = str(spec["candidate_id"])

            val_path = os.path.join(OUT_PORT, f"candidate_{cid}_VAL.parquet")
            test_path = os.path.join(OUT_PORT, f"candidate_{cid}_TEST.parquet")
            meta_path = os.path.join(OUT_PORT, f"candidate_{cid}_fit_meta_v2_THESIS.json")

            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)

            meta = {
                "version": "cell10_4s_router_support_quantile_variant_v2_THESIS",
                "cell": "10.4s",
                "generator": gen,
                "candidate_id": cid,
                "variant_name": spec["variant_name"],
                "kind": spec["kind"],
                "columns": list(cols),
                "primary_targets": PRIMARY_FIT_COLS,
                "secondary_targets": SECONDARY_FIT_COLS,
                "excluded_failed_block_replay_cols": EXCLUDED_FAILED_BLOCK_REPLAY_COLS_10_4S,
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "test_used_for_fitting": False,
                "test_used_for_repair": False,
                "variant_spec": spec,
                "fit_meta": {c: fit_meta[c] for c in cols},
                "role": spec.get("role"),
                "copy_risk_note": (
                    "This candidate samples TRAIN empirical/quantile support. "
                    "If promoted into a public artifact, Q6/no-copy audits must be rerun."
                ),
            }

            meta["record_sha256"] = hashlib.sha256(
                json.dumps(meta, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

            _write_atomic_json(meta_path, meta)

            rec = _cell10_register_portfolio_candidate({
                "generator": gen,
                "candidate_id": cid,
                "columns": list(cols),
                "val_path": val_path,
                "test_path": test_path,
                "fit_meta_path": meta_path,
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "test_used_for_fitting": False,
                "test_used_for_repair": False,
                "role": spec.get("role", "router_support_quantile_candidate"),
                "eligibility": {
                    "columns": list(cols),
                    "primary_targets": PRIMARY_FIT_COLS,
                    "secondary_targets": SECONDARY_FIT_COLS,
                    "excluded_failed_block_replay_cols": EXCLUDED_FAILED_BLOCK_REPLAY_COLS_10_4S,
                    "kind": spec["kind"],
                    "variant_name": spec["variant_name"],
                },
            })

            registered.append(cid)
            portfolio_meta_rows.append(meta)
            all_generation_audits.append(val_audit)
            all_generation_audits.append(test_audit)

            log(
                "[Cell10.4s] Registered router support/quantile candidate | "
                f"candidate_id={cid} | kind={spec['kind']} | "
                f"VAL={val_df.shape} | TEST={test_df.shape} | cols={cols}"
            )

        # --------------------------------------------------
        # 7) Persist portfolio audit/contract
        # --------------------------------------------------
        fit_audit_path = os.path.join(
            OUT_REP,
            "cell10_4s_router_support_quantile_fit_audit_v2_THESIS.csv",
        )
        gen_audit_path = os.path.join(
            OUT_REP,
            "cell10_4s_router_support_quantile_generation_audit_v2_THESIS.csv",
        )
        audit_path = os.path.join(
            OUT_REP,
            "cell10_4s_router_support_quantile_audit_v2_THESIS.json",
        )
        contract_path = os.path.join(
            CONTRACT_DIR,
            "cell10_4s_router_support_quantile_contract_v2_THESIS.json",
        )
        portfolio_meta_path = os.path.join(
            OUT_PORT,
            "candidate_router_support_quantile_portfolio_meta_v2_THESIS.json",
        )

        fit_audit_df = pd.DataFrame([
            {
                "col": c,
                "target_role": fit_meta[c]["target_role"],
                "train_n": int(fit_meta[c]["train_n"]),
                "support_n": int(fit_meta[c]["support_n"]),
                "profile_json": json.dumps(fit_meta[c]["profile"], sort_keys=True, default=str),
            }
            for c in FIT_COLS
        ])

        gen_audit_df = (
            pd.concat(all_generation_audits, axis=0, ignore_index=True)
            if all_generation_audits
            else pd.DataFrame()
        )

        fit_audit_df.to_csv(fit_audit_path, index=False)
        gen_audit_df.to_csv(gen_audit_path, index=False)

        portfolio_meta = {
            "version": "cell10_4s_router_support_quantile_portfolio_v2_THESIS",
            "cell": "10.4s",
            "generator_family": "router_support_quantile",
            "portfolio_role": "router_support_magnitude_quantile_candidate_portfolio",
            "registered_candidate_ids": list(registered),
            "variants_n": int(len(registered)),
            "fit_cols": FIT_COLS,
            "primary_fit_cols": PRIMARY_FIT_COLS,
            "secondary_fit_cols": SECONDARY_FIT_COLS,
            "excluded_failed_block_replay_cols": EXCLUDED_FAILED_BLOCK_REPLAY_COLS_10_4S,
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "test_used_for_selection": False,
            "fit_meta": fit_meta,
            "variant_meta": portfolio_meta_rows,
        }
        _write_atomic_json(portfolio_meta_path, portfolio_meta)

        audit = {
            "version": "cell10_4s_router_support_quantile_candidate_v2_THESIS",
            "candidate_available": bool(len(registered) > 0),
            "registered_candidate_ids": list(registered),
            "variants_n": int(len(registered)),
            "target_cols": list(TARGET_COLS),
            "fit_cols": list(FIT_COLS),
            "primary_fit_cols": list(PRIMARY_FIT_COLS),
            "secondary_fit_cols": list(SECONDARY_FIT_COLS),
            "excluded_failed_block_replay_cols": EXCLUDED_FAILED_BLOCK_REPLAY_COLS_10_4S,
            "support_path": support_path,
            "fit_audit_path": fit_audit_path,
            "generation_audit_path": gen_audit_path,
            "eligibility_audit_path": eligibility_path,
            "portfolio_meta_path": portfolio_meta_path,
            "fit_summary": {
                "fit_cols_n": int(len(FIT_COLS)),
                "primary_fit_cols_n": int(len(PRIMARY_FIT_COLS)),
                "secondary_fit_cols_n": int(len(SECONDARY_FIT_COLS)),
                "min_train_n": int(min_train_n),
            },
            "copy_risk_status": {
                "uses_train_empirical_or_quantile_support_sampling": True,
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
                "Router support/quantile candidates are predefined protocol-specific "
                "portfolio candidates for router support/magnitude columns. They are not "
                "final evidence unless selected later by the downstream VAL-only selector "
                "against A1, and remain subject to Q6/copy-risk checks before any "
                "public-release claim."
            ),
            "ts_unix": float(time.time()),
        }

        _write_atomic_json(audit_path, audit)
        _write_atomic_json(contract_path, audit)

        globals()["CELL10_4S_ROUTER_SUPPORT_AUDIT"] = audit
        globals()["CELL10_4S_ROUTER_SUPPORT_AUDIT_PATH"] = audit_path
        globals()["CELL10_4S_ROUTER_SUPPORT_CONTRACT_PATH"] = contract_path

        if "RUN_META" in globals():
            RUN_META.setdefault("portfolio_candidate_audits", {})
            RUN_META["portfolio_candidate_audits"]["router_support_quantile"] = audit

        log(f"[Cell10.4s] Saved eligibility audit: {eligibility_path}")
        log(f"[Cell10.4s] Saved support audit: {support_path}")
        log(f"[Cell10.4s] Saved fit audit: {fit_audit_path}")
        log(f"[Cell10.4s] Saved generation audit: {gen_audit_path}")
        log(f"[Cell10.4s] Saved portfolio meta: {portfolio_meta_path}")
        log(f"[Cell10.4s] Saved audit: {audit_path}")
        log(f"[Cell10.4s] Saved contract: {contract_path}")
        log(
            "[Cell10.4s] Router support/quantile candidate portfolio complete | "
            f"variants={registered}"
        )

log("--- END: Cell 10.4s — Router support/quantile candidate portfolio (v2-THESIS) ---")
