# ==========================================================
# CELL 10.4r — Router service-aware multivariate block replay candidate — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Add a TRAIN-fitted router service-aware multivariate block-replay portfolio.
# - Preserve synchronized router service bursts and co-movement using TRAIN-only
#   contiguous multivariate blocks.
# - Emit VAL and TEST-horizon candidate artifacts for downstream VAL-only selection.
#
# Target strategy:
# - Target temporal/service columns only:
#     aggregate: router__pkt_total
#     arp:       router__arp_pkt, router__arp_reply, router__arp_req
#     dns:       router__dns_pkt, router__dns_query, router__dns_rcode0_ok, router__dns_response
#     tcp:       router__tcp_pkt, router__tcp_ack, router__tcp_syn, router__tcp_fin
#     udp:       router__udp_pkt
#     unique:    router__dst_port_unique, router__ip_dst_unique, router__ip_src_unique
#
# - Intentionally exclude support/marginal-only columns:
#     router__bytes_total
#     router__dhcp_pkt
#     router__dns_rcode3_nx
#     router__icmp_pkt
#     router__tcp_rst
#
# Scientific contract:
# - Fit TRAIN only.
# - Generate VAL and TEST candidate files from TRAIN block library and masks.
# - Do not read real TEST values.
# - Do not use VAL for fitting.
# - Do not select anything in this cell.
# - Do not modify A0/A1/A2 artifacts.
# - Not final evidence unless selected downstream on VAL against A1.
#
# Public-release caveat:
# - Variants replay contiguous TRAIN blocks. If selected for any public-release
#   scope, Q6/copy-risk/no-copy audits are mandatory.
# ==========================================================

log("--- START: Cell 10.4r — Router service-aware multivariate block replay candidate (v2-THESIS) ---")

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
    raise RuntimeError(f"[Cell10.4r] Missing prerequisites: {missing}. Run Cells 10.1–10.4e first.")

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
        raise RuntimeError(f"[Cell10.4r] Clean router candidate generation forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.4r] Unexpected selection_split_policy. "
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
# 1) Target groups from the predefined router diagnostic plan
# ----------------------------------------------------------
ROUTER_GROUPS_10_4R = {
    "aggregate_pkt": [
        "router__pkt_total",
    ],
    "arp": [
        "router__arp_pkt",
        "router__arp_reply",
        "router__arp_req",
    ],
    "dns_core": [
        "router__dns_pkt",
        "router__dns_query",
        "router__dns_rcode0_ok",
        "router__dns_response",
    ],
    "tcp_core": [
        "router__tcp_pkt",
        "router__tcp_ack",
        "router__tcp_syn",
        "router__tcp_fin",
    ],
    "udp": [
        "router__udp_pkt",
    ],
    "unique_counts": [
        "router__dst_port_unique",
        "router__ip_dst_unique",
        "router__ip_src_unique",
    ],
}

ROUTER_EXCLUDED_SUPPORT_COLS_10_4R = [
    "router__bytes_total",
    "router__dhcp_pkt",
    "router__dns_rcode3_nx",
    "router__icmp_pkt",
    "router__tcp_rst",
]

proto_col_set = set(map(str, CELL10_PROTO_COLS))

TARGET_GROUPS = {}
eligibility_rows = []

for group, cols in ROUTER_GROUPS_10_4R.items():
    present = []
    for c in cols:
        ok = (
            c in proto_col_set
            and c in df_tr.columns
            and c in df_va.columns
            and c in df_te.columns
        )
        eligibility_rows.append({
            "group": group,
            "col": c,
            "expected": True,
            "in_CELL10_PROTO_COLS": c in proto_col_set,
            "in_df_tr_schema": c in df_tr.columns,
            "in_df_va_schema": c in df_va.columns,
            "in_df_te_schema": c in df_te.columns,
            "selected_target": bool(ok),
        })
        if ok:
            present.append(c)

    if present:
        TARGET_GROUPS[group] = list(present)

TARGET_COLS = []
for group in ROUTER_GROUPS_10_4R:
    TARGET_COLS.extend(TARGET_GROUPS.get(group, []))
TARGET_COLS = list(dict.fromkeys(TARGET_COLS))

for c in ROUTER_EXCLUDED_SUPPORT_COLS_10_4R:
    eligibility_rows.append({
        "group": "excluded_support_or_marginal_only",
        "col": c,
        "expected": False,
        "in_CELL10_PROTO_COLS": c in proto_col_set,
        "in_df_tr_schema": c in df_tr.columns,
        "in_df_va_schema": c in df_va.columns,
        "in_df_te_schema": c in df_te.columns,
        "selected_target": False,
    })

eligibility_path = os.path.join(
    OUT_REP,
    "cell10_4r_router_service_block_replay_eligibility_v2_THESIS.csv",
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
        "router_service_block_replay",
        reason=reason,
        meta={
            "source": "Cell10.4r",
            "version": "cell10_4r_router_service_block_replay_v2_THESIS",
            "target_groups": TARGET_GROUPS,
            "target_cols": TARGET_COLS,
            "excluded_support_cols": ROUTER_EXCLUDED_SUPPORT_COLS_10_4R,
            "eligibility_audit_path": eligibility_path,
            **dict(extra_meta or {}),
        },
    )

    audit = {
        "version": "cell10_4r_router_service_block_replay_candidate_v2_THESIS",
        "candidate_available": False,
        "reason": reason,
        "target_groups": TARGET_GROUPS,
        "target_cols": TARGET_COLS,
        "registered_candidate_ids": [],
        "variants_n": 0,
        "excluded_support_cols": ROUTER_EXCLUDED_SUPPORT_COLS_10_4R,
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
            "Router service block-replay candidate unavailable because required schema "
            "or TRAIN-only support conditions were not satisfied."
        ),
        "ts_unix": float(time.time()),
    }

    audit_path = os.path.join(OUT_REP, "cell10_4r_router_service_block_replay_audit_v2_THESIS.json")
    contract_path = os.path.join(CONTRACT_DIR, "cell10_4r_router_service_block_replay_contract_v2_THESIS.json")

    _write_atomic_json(audit_path, audit)
    _write_atomic_json(contract_path, audit)

    globals()["CELL10_4R_ROUTER_AUDIT"] = audit
    globals()["CELL10_4R_ROUTER_AUDIT_PATH"] = audit_path
    globals()["CELL10_4R_ROUTER_CONTRACT_PATH"] = contract_path

    if "RUN_META" in globals():
        RUN_META.setdefault("portfolio_candidate_audits", {})
        RUN_META["portfolio_candidate_audits"]["router_service_block_replay"] = audit

    log(f"[Cell10.4r] Router service block replay unavailable: {reason}")
    log(f"[Cell10.4r] Saved eligibility audit: {eligibility_path}")
    log(f"[Cell10.4r] Saved audit: {audit_path}")
    log(f"[Cell10.4r] Saved contract: {contract_path}")


if not TARGET_COLS:
    _finalize_unavailable(
        reason="No router temporal target columns found in all required schemas.",
        extra_meta={"expected_groups": ROUTER_GROUPS_10_4R},
    )

else:
    # ------------------------------------------------------
    # 3) Helper functions
    # ------------------------------------------------------
    def _router_train_active_mask() -> np.ndarray:
        if "router__obs_present" not in df_tr.columns:
            raise RuntimeError("[Cell10.4r] TRAIN missing router__obs_present.")
        v = pd.to_numeric(df_tr["router__obs_present"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
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


    def _basic_profile_1d(x) -> dict:
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]

        if x.size == 0:
            return {
                "n": 0,
                "mean": np.nan,
                "std": np.nan,
                "q50": np.nan,
                "q95": np.nan,
                "q99": np.nan,
                "nonzero_rate": np.nan,
                "unique_n": 0,
                "transition_rate": np.nan,
            }

        return {
            "n": int(x.size),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "q50": float(np.quantile(x, 0.50)),
            "q95": float(np.quantile(x, 0.95)),
            "q99": float(np.quantile(x, 0.99)),
            "nonzero_rate": float(np.mean(x > 0)),
            "unique_n": int(np.unique(np.rint(x)).size),
            "transition_rate": float(np.mean(np.abs(np.diff(x)) > 1e-9)) if x.size > 1 else np.nan,
        }


    def _group_profile(X: np.ndarray, cols: list) -> dict:
        X = np.asarray(X, dtype=np.float64)
        prof = {}

        for j, c in enumerate(cols):
            prof[c] = _basic_profile_1d(X[:, j])

        if X.shape[0] > 2 and X.shape[1] > 1:
            with np.errstate(all="ignore"):
                corr = np.corrcoef(X, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            tri = corr[np.triu_indices_from(corr, k=1)]
            prof["_corr_shape"] = list(corr.shape)
            prof["_corr_abs_mean_offdiag"] = float(np.mean(np.abs(tri))) if tri.size else 0.0
        else:
            prof["_corr_shape"] = [int(X.shape[1]), int(X.shape[1])] if X.ndim == 2 else [0, 0]
            prof["_corr_abs_mean_offdiag"] = 0.0

        return prof


    def _sample_block_length(
        rng_local: np.random.Generator,
        block_min: int,
        block_max: int,
        remaining: int,
        mode: str,
    ) -> int:
        block_min = int(max(1, block_min))
        block_max = int(max(block_min, block_max))
        remaining = int(max(1, remaining))

        hi = min(block_max, remaining)
        lo = min(block_min, hi)

        if hi <= lo:
            return int(hi)

        mode = str(mode)

        if mode == "short":
            u = rng_local.beta(1.5, 4.0)
            return int(round(lo + u * (hi - lo)))

        if mode == "medium":
            lo_log = np.log(max(1, lo))
            hi_log = np.log(max(1, hi))
            return int(round(np.exp(rng_local.uniform(lo_log, hi_log))))

        if mode == "long":
            u = rng_local.beta(4.0, 1.5)
            return int(round(lo + u * (hi - lo)))

        if mode == "fixed_mid":
            return int((lo + hi) // 2)

        return int(rng_local.integers(lo, hi + 1))


    def _make_train_group_matrix(cols: list) -> dict:
        active = _router_train_active_mask()
        X_full = df_tr[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        finite_rows = np.isfinite(X_full).all(axis=1)
        valid_active = active & finite_rows

        if int(valid_active.sum()) <= 0:
            raise RuntimeError(f"No TRAIN valid active rows for columns={cols}")

        X_full = np.maximum(X_full, 0.0)
        X_full = np.rint(X_full).astype(np.float32)

        return {
            "X_full": X_full,
            "valid_active": valid_active,
            "active": active,
            "segments": _contiguous_segments_from_mask(valid_active),
            "train_valid_active_n": int(valid_active.sum()),
            "train_active_n": int(active.sum()),
        }


    def _usable_segments(group_fit: dict, min_seg_len: int):
        segs = [
            (s, e)
            for s, e in group_fit["segments"]
            if int(e - s) >= int(min_seg_len)
        ]

        if not segs:
            idx = np.flatnonzero(group_fit["valid_active"])
            if idx.size:
                segs = [(int(idx[0]), int(idx[-1]) + 1)]

        return segs


    def _choose_segment(
        rng_local: np.random.Generator,
        segment_infos: list,
        probs: np.ndarray,
        policy: str,
        produced_blocks: int = 0,
    ) -> dict:
        if not segment_infos:
            raise RuntimeError("[Cell10.4r] No segment infos available.")

        policy = str(policy)

        if policy == "round_robin":
            return segment_infos[int(produced_blocks % len(segment_infos))]

        if policy == "uniform_segment":
            return segment_infos[int(rng_local.integers(0, len(segment_infos)))]

        return segment_infos[int(rng_local.choice(np.arange(len(segment_infos)), p=probs))]


    def _sample_multivariate_block(
        rng_local: np.random.Generator,
        fit: dict,
        segment_infos: list,
        probs: np.ndarray,
        spec: dict,
        remaining: int,
        produced_blocks: int,
    ) -> np.ndarray:
        X_full = fit["X_full"]
        remaining = int(remaining)

        if remaining <= 0:
            return np.zeros((0, X_full.shape[1]), dtype=np.float32)

        for _ in range(200):
            seg = _choose_segment(
                rng_local,
                segment_infos,
                probs,
                policy=str(spec.get("segment_policy", "length_weighted")),
                produced_blocks=produced_blocks,
            )

            s = int(seg["start"])
            e = int(seg["end"])
            seg_len = int(e - s)

            if seg_len <= 0:
                continue

            L = _sample_block_length(
                rng_local,
                int(spec["block_min"]),
                min(int(spec["block_max"]), seg_len),
                remaining,
                str(spec["length_mode"]),
            )
            L = int(min(max(1, L), seg_len, remaining))

            start_hi = e - L
            if start_hi < s:
                continue

            start = int(rng_local.integers(s, start_hi + 1))
            block = X_full[start:start + L, :]

            if block.shape[0] > 0 and np.isfinite(block).all():
                return np.maximum(np.rint(block), 0).astype(np.float32, copy=False)

        idx = np.flatnonzero(fit["valid_active"])
        take = min(remaining, max(1, int(spec["block_min"])))
        draw = rng_local.choice(idx, size=take, replace=True)
        block = X_full[draw, :]
        return np.maximum(np.rint(block), 0).astype(np.float32, copy=False)


    # ------------------------------------------------------
    # 4) TRAIN-only group fitting
    # ------------------------------------------------------
    min_train_active = int(CFG.get("cell10_4r_router_min_train_active_rows", 1000))
    group_fits = {}
    group_fit_meta = {}
    group_support_rows = []

    for group, cols in TARGET_GROUPS.items():
        try:
            fit = _make_train_group_matrix(cols)
        except Exception as e:
            group_support_rows.append({
                "group": group,
                "cols": "|".join(cols),
                "fit_available": False,
                "train_active_n": 0,
                "train_valid_active_n": 0,
                "segments_n": 0,
                "reason": f"{type(e).__name__}: {e}",
            })
            log(f"[Cell10.4r][WARN] Skipping group={group}: {type(e).__name__}: {e}")
            continue

        if int(fit["train_valid_active_n"]) < min_train_active:
            group_support_rows.append({
                "group": group,
                "cols": "|".join(cols),
                "fit_available": False,
                "train_active_n": int(fit["train_active_n"]),
                "train_valid_active_n": int(fit["train_valid_active_n"]),
                "segments_n": int(len(fit["segments"])),
                "reason": f"train_valid_active_n < {min_train_active}",
            })
            log(
                f"[Cell10.4r][WARN] Skipping group={group}: "
                f"train_valid_active_n={fit['train_valid_active_n']} < {min_train_active}"
            )
            continue

        group_fits[group] = fit

        Xtrain = fit["X_full"][fit["valid_active"], :]
        group_fit_meta[group] = {
            "cols": list(cols),
            "train_active_n": int(fit["train_active_n"]),
            "train_valid_active_n": int(fit["train_valid_active_n"]),
            "segments_n": int(len(fit["segments"])),
            "profile": _group_profile(Xtrain, cols),
        }

        group_support_rows.append({
            "group": group,
            "cols": "|".join(cols),
            "fit_available": True,
            "train_active_n": int(fit["train_active_n"]),
            "train_valid_active_n": int(fit["train_valid_active_n"]),
            "segments_n": int(len(fit["segments"])),
            "reason": "available",
        })

    group_support_path = os.path.join(
        OUT_REP,
        "cell10_4r_router_service_block_replay_group_support_v2_THESIS.csv",
    )
    pd.DataFrame(group_support_rows).to_csv(group_support_path, index=False)

    if not group_fits:
        _finalize_unavailable(
            reason="No router service groups met TRAIN support requirements.",
            extra_meta={
                "target_groups": TARGET_GROUPS,
                "min_train_active": min_train_active,
                "group_support_path": group_support_path,
            },
        )

    else:
        log(
            "[Cell10.4r] Fitted TRAIN router group caches | "
            f"groups={list(group_fits.keys())} | "
            f"cols={sum(len(TARGET_GROUPS[g]) for g in group_fits.keys())}"
        )

        # --------------------------------------------------
        # 5) Predefined variants
        # --------------------------------------------------
        variant_specs = [
            {
                "generator": "router_service_block_replay",
                "candidate_id": "router_service_block_replay_short_v2_THESIS",
                "variant_name": "short_service_blocks",
                "block_min": int(CFG.get("cell10_4r_short_block_min", 5)),
                "block_max": int(CFG.get("cell10_4r_short_block_max", 120)),
                "length_mode": "short",
                "segment_policy": "length_weighted",
                "min_segment_len": int(CFG.get("cell10_4r_short_min_segment_len", 30)),
                "role": "router_local_transition_candidate",
            },
            {
                "generator": "router_service_block_replay",
                "candidate_id": "router_service_block_replay_medium_v2_THESIS",
                "variant_name": "medium_service_blocks",
                "block_min": int(CFG.get("cell10_4r_medium_block_min", 30)),
                "block_max": int(CFG.get("cell10_4r_medium_block_max", 900)),
                "length_mode": "medium",
                "segment_policy": "length_weighted",
                "min_segment_len": int(CFG.get("cell10_4r_medium_min_segment_len", 60)),
                "role": "router_balanced_temporal_candidate",
            },
            {
                "generator": "router_service_block_replay",
                "candidate_id": "router_service_block_replay_long_v2_THESIS",
                "variant_name": "long_service_blocks",
                "block_min": int(CFG.get("cell10_4r_long_block_min", 180)),
                "block_max": int(CFG.get("cell10_4r_long_block_max", 2400)),
                "length_mode": "long",
                "segment_policy": "length_weighted",
                "min_segment_len": int(CFG.get("cell10_4r_long_min_segment_len", 180)),
                "role": "router_long_autocorrelation_candidate",
            },
            {
                "generator": "router_service_block_replay",
                "candidate_id": "router_service_block_replay_roundrobin_v2_THESIS",
                "variant_name": "roundrobin_service_blocks",
                "block_min": int(CFG.get("cell10_4r_roundrobin_block_min", 20)),
                "block_max": int(CFG.get("cell10_4r_roundrobin_block_max", 600)),
                "length_mode": "medium",
                "segment_policy": "round_robin",
                "min_segment_len": int(CFG.get("cell10_4r_roundrobin_min_segment_len", 30)),
                "role": "router_segment_balance_candidate",
            },
        ]

        for spec in variant_specs:
            spec["block_min"] = int(max(1, spec["block_min"]))
            spec["block_max"] = int(max(spec["block_min"], spec["block_max"]))
            spec["min_segment_len"] = int(max(1, spec["min_segment_len"]))

        # --------------------------------------------------
        # 6) Generate candidate for each group/spec
        # --------------------------------------------------
        def _generate_group_candidate(
            mask_dict: dict,
            N: int,
            split_name: str,
            group: str,
            cols: list,
            fit: dict,
            spec: dict,
        ):
            N = int(N)
            if N <= 0:
                raise RuntimeError(f"[Cell10.4r] {split_name}/{group}: N must be positive.")

            active = _mask_for_tier(mask_dict, "router", N, missing_policy="error")
            n_active = int(active.sum())

            out = {c: np.full(N, np.nan, dtype=np.float32) for c in cols}
            block_lengths = []

            min_seg_len = int(spec.get("min_segment_len", 30))
            segs = _usable_segments(fit, min_seg_len=min_seg_len)

            if not segs:
                raise RuntimeError(f"[Cell10.4r] No usable segments for group={group}")

            segment_infos = []
            for si, (s, e) in enumerate(segs):
                Xseg = fit["X_full"][s:e, :]
                segment_infos.append({
                    "segment_id": int(si),
                    "start": int(s),
                    "end": int(e),
                    "length": int(e - s),
                    "profile": _group_profile(Xseg, cols),
                })

            lengths = np.asarray([x["length"] for x in segment_infos], dtype=np.float64)
            probs = lengths / max(float(lengths.sum()), 1.0)

            rng_local = np.random.default_rng(
                _stable_seed_from_parts(
                    "cell10_4r_router_service_block_replay",
                    split_name,
                    spec["candidate_id"],
                    group,
                    "|".join(cols),
                    seed,
                    base_seed=seed + 10054,
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
                            f"[Cell10.4r] Generation guard exceeded for "
                            f"group={group}, split={split_name}, spec={spec['candidate_id']}"
                        )

                    block = _sample_multivariate_block(
                        rng_local=rng_local,
                        fit=fit,
                        segment_infos=segment_infos,
                        probs=probs,
                        spec=spec,
                        remaining=n_active - produced,
                        produced_blocks=produced_blocks,
                    )

                    if block.shape[0] == 0:
                        continue

                    chunks.append(block)
                    block_lengths.append(int(block.shape[0]))
                    produced += int(block.shape[0])
                    produced_blocks += 1

                X = np.vstack(chunks)[:n_active, :].astype(np.float32, copy=False)

                for j, c in enumerate(cols):
                    arr = np.full(N, np.nan, dtype=np.float32)
                    arr[active] = X[:, j]
                    out[c] = _postprocess_protocol_values(c, arr)

            df_out = pd.DataFrame(out)

            gen_rows = []
            for c in cols:
                x = pd.to_numeric(df_out[c], errors="coerce").to_numpy(dtype=np.float64)

                finite_active = int(np.isfinite(x[active]).sum())
                finite_inactive = int(np.isfinite(x[~active]).sum())

                if n_active > 0 and finite_active != n_active:
                    raise RuntimeError(
                        f"[Cell10.4r] {split_name}/{spec['candidate_id']}/{group}.{c}: "
                        f"active finite mismatch {finite_active} vs {n_active}"
                    )

                if finite_inactive != 0:
                    raise RuntimeError(
                        f"[Cell10.4r] {split_name}/{spec['candidate_id']}/{group}.{c}: "
                        f"inactive finite values {finite_inactive}"
                    )

                xa = x[active]

                gen_rows.append({
                    "split": split_name,
                    "candidate_id": spec["candidate_id"],
                    "variant_name": spec["variant_name"],
                    "group": group,
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
                    "block_draws": int(len(block_lengths)),
                    "block_len_median": float(np.median(block_lengths)) if block_lengths else np.nan,
                    "block_len_p95": float(np.quantile(block_lengths, 0.95)) if block_lengths else np.nan,
                    "train_valid_active_n": int(fit["train_valid_active_n"]),
                    "segments_used_n": int(len(segment_infos)),
                })

            return df_out, pd.DataFrame(gen_rows)

        # --------------------------------------------------
        # 7) Materialize/register variants
        # --------------------------------------------------
        registered = []
        portfolio_meta_rows = []
        all_generation_audits = []

        fit_audit_rows = []
        for group, meta in group_fit_meta.items():
            fit_audit_rows.append({
                "group": group,
                "cols": "|".join(meta["cols"]),
                "train_active_n": int(meta["train_active_n"]),
                "train_valid_active_n": int(meta["train_valid_active_n"]),
                "segments_n": int(meta["segments_n"]),
                "profile_json": json.dumps(meta["profile"], sort_keys=True, default=str),
            })

        for spec in variant_specs:
            val_parts = []
            test_parts = []
            groups_generated = []

            for group in group_fits:
                fit = group_fits[group]
                cols = TARGET_GROUPS[group]

                val_g, val_audit = _generate_group_candidate(
                    CELL10_VAL_MASKS,
                    len(df_va),
                    "VAL",
                    group=group,
                    cols=cols,
                    fit=fit,
                    spec=spec,
                )

                test_g, test_audit = _generate_group_candidate(
                    CELL10_TEST_MASKS,
                    len(df_te),
                    "TEST",
                    group=group,
                    cols=cols,
                    fit=fit,
                    spec=spec,
                )

                val_parts.append(val_g)
                test_parts.append(test_g)
                all_generation_audits.append(val_audit)
                all_generation_audits.append(test_audit)
                groups_generated.append(group)

            val_df = pd.concat(val_parts, axis=1)
            test_df = pd.concat(test_parts, axis=1)

            # Deterministic column order: group order then column order.
            ordered_cols = []
            for group in ROUTER_GROUPS_10_4R:
                for c in TARGET_GROUPS.get(group, []):
                    if c in val_df.columns and c not in ordered_cols:
                        ordered_cols.append(c)

            val_df = val_df.loc[:, ordered_cols]
            test_df = test_df.loc[:, ordered_cols]

            cid = str(spec["candidate_id"])
            gen = str(spec["generator"])

            val_path = os.path.join(OUT_PORT, f"candidate_{cid}_VAL.parquet")
            test_path = os.path.join(OUT_PORT, f"candidate_{cid}_TEST.parquet")
            meta_path = os.path.join(OUT_PORT, f"candidate_{cid}_fit_meta_v2_THESIS.json")

            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)

            meta = {
                "version": "cell10_4r_router_service_block_replay_variant_v2_THESIS",
                "cell": "10.4r",
                "generator": gen,
                "candidate_id": cid,
                "variant_name": spec["variant_name"],
                "columns": list(ordered_cols),
                "target_groups": {g: TARGET_GROUPS[g] for g in groups_generated},
                "excluded_support_cols": ROUTER_EXCLUDED_SUPPORT_COLS_10_4R,
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "test_used_for_fitting": False,
                "test_used_for_repair": False,
                "variant_spec": spec,
                "group_fit_meta": group_fit_meta,
                "role": spec.get("role"),
                "copy_risk_note": (
                    "This candidate replays contiguous TRAIN multivariate router service blocks. "
                    "If promoted into public artifact, Q6/no-copy audits must be rerun."
                ),
            }

            meta["record_sha256"] = hashlib.sha256(
                json.dumps(meta, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()

            _write_atomic_json(meta_path, meta)

            rec = _cell10_register_portfolio_candidate({
                "generator": gen,
                "candidate_id": cid,
                "columns": list(ordered_cols),
                "val_path": val_path,
                "test_path": test_path,
                "fit_meta_path": meta_path,
                "fit_split": "TRAIN",
                "selection_split": "VAL",
                "test_used_for_selection": False,
                "test_used_for_fitting": False,
                "test_used_for_repair": False,
                "role": spec.get("role", "router_service_block_replay_candidate"),
                "eligibility": {
                    "target_groups": {g: TARGET_GROUPS[g] for g in groups_generated},
                    "excluded_support_cols": ROUTER_EXCLUDED_SUPPORT_COLS_10_4R,
                    "kind": "multivariate_service_block_replay",
                    "variant_name": spec["variant_name"],
                    "block_min": int(spec["block_min"]),
                    "block_max": int(spec["block_max"]),
                    "length_mode": str(spec["length_mode"]),
                    "segment_policy": str(spec["segment_policy"]),
                },
            })

            registered.append(cid)
            portfolio_meta_rows.append(meta)

            log(
                "[Cell10.4r] Registered router service block candidate | "
                f"candidate_id={cid} | VAL={val_df.shape} | TEST={test_df.shape} | "
                f"cols={len(ordered_cols)} | groups={groups_generated} | "
                f"block={spec['block_min']}-{spec['block_max']} | mode={spec['length_mode']}"
            )

        # --------------------------------------------------
        # 8) Persist portfolio audit/contract
        # --------------------------------------------------
        fit_audit_path = os.path.join(
            OUT_REP,
            "cell10_4r_router_service_block_replay_fit_audit_v2_THESIS.csv",
        )
        gen_audit_path = os.path.join(
            OUT_REP,
            "cell10_4r_router_service_block_replay_generation_audit_v2_THESIS.csv",
        )
        audit_path = os.path.join(
            OUT_REP,
            "cell10_4r_router_service_block_replay_audit_v2_THESIS.json",
        )
        contract_path = os.path.join(
            CONTRACT_DIR,
            "cell10_4r_router_service_block_replay_contract_v2_THESIS.json",
        )
        portfolio_meta_path = os.path.join(
            OUT_PORT,
            "candidate_router_service_block_replay_portfolio_meta_v2_THESIS.json",
        )

        fit_audit_df = pd.DataFrame(fit_audit_rows)
        gen_audit_df = pd.concat(all_generation_audits, axis=0, ignore_index=True)

        fit_audit_df.to_csv(fit_audit_path, index=False)
        gen_audit_df.to_csv(gen_audit_path, index=False)

        portfolio_meta = {
            "version": "cell10_4r_router_service_block_replay_portfolio_v2_THESIS",
            "cell": "10.4r",
            "generator": "router_service_block_replay",
            "portfolio_role": "predefined_router_service_multivariate_block_replay_candidate_portfolio",
            "registered_candidate_ids": list(registered),
            "variants_n": int(len(registered)),
            "target_groups": TARGET_GROUPS,
            "target_cols": TARGET_COLS,
            "excluded_support_cols": ROUTER_EXCLUDED_SUPPORT_COLS_10_4R,
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "test_used_for_selection": False,
            "group_fit_meta": group_fit_meta,
            "variant_meta": portfolio_meta_rows,
        }

        _write_atomic_json(portfolio_meta_path, portfolio_meta)

        audit = {
            "version": "cell10_4r_router_service_block_replay_candidate_v2_THESIS",
            "candidate_available": True,
            "registered_candidate_ids": list(registered),
            "variants_n": int(len(registered)),
            "target_groups": TARGET_GROUPS,
            "target_cols": TARGET_COLS,
            "excluded_support_cols": ROUTER_EXCLUDED_SUPPORT_COLS_10_4R,
            "group_support_path": group_support_path,
            "fit_audit_path": fit_audit_path,
            "generation_audit_path": gen_audit_path,
            "eligibility_audit_path": eligibility_path,
            "portfolio_meta_path": portfolio_meta_path,
            "fit_summary": {
                "groups_available": list(group_fits.keys()),
                "groups_available_n": int(len(group_fits)),
                "target_cols_generated_n": int(sum(len(TARGET_GROUPS[g]) for g in group_fits.keys())),
                "min_train_active": int(min_train_active),
            },
            "copy_risk_status": {
                "uses_contiguous_train_multivariate_block_replay": True,
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
                "Router service block-replay variants are predefined protocol-specific "
                "portfolio candidates for temporal/service router columns. They are not "
                "final evidence unless selected later by the downstream VAL-only selector "
                "against A1, and remain subject to Q6/copy-risk checks before any "
                "public-release claim."
            ),
            "ts_unix": float(time.time()),
        }

        _write_atomic_json(audit_path, audit)
        _write_atomic_json(contract_path, audit)

        globals()["CELL10_4R_ROUTER_AUDIT"] = audit
        globals()["CELL10_4R_ROUTER_AUDIT_PATH"] = audit_path
        globals()["CELL10_4R_ROUTER_CONTRACT_PATH"] = contract_path

        if "RUN_META" in globals():
            RUN_META.setdefault("portfolio_candidate_audits", {})
            RUN_META["portfolio_candidate_audits"]["router_service_block_replay"] = audit

        log(f"[Cell10.4r] Saved eligibility audit: {eligibility_path}")
        log(f"[Cell10.4r] Saved group support audit: {group_support_path}")
        log(f"[Cell10.4r] Saved fit audit: {fit_audit_path}")
        log(f"[Cell10.4r] Saved generation audit: {gen_audit_path}")
        log(f"[Cell10.4r] Saved portfolio meta: {portfolio_meta_path}")
        log(f"[Cell10.4r] Saved audit: {audit_path}")
        log(f"[Cell10.4r] Saved contract: {contract_path}")
        log(
            "[Cell10.4r] Router service-aware block replay candidate portfolio complete | "
            f"variants={registered}"
        )

log("--- END: Cell 10.4r — Router service-aware multivariate block replay candidate (v2-THESIS) ---")
