# ==========================================================
# CELL 10.1 — Generator portfolio contract + A0/A1 baselines
# v3-THESIS
#
# Purpose:
# - Define the predefined generator portfolio.
# - Build TRAIN-only protocol baselines for controlled variants:
#
#     A0_VAL / A0_TEST:
#       TRAIN-only non-copying baseline values under real VAL/TEST masks.
#
#     A1_VAL / A1_TEST:
#       Same TRAIN-only non-copying baseline values under synthetic VAL/TEST masks.
#
# - Provide shared registration/evaluation utilities for downstream portfolio candidate cells.
#
# Scientific contract:
# - TRAIN values are used to fit/generate all baseline values.
# - A0 uses real VAL/TEST masks only as mask processes, never real values.
# - A1 uses upstream synthetic VAL/TEST masks produced by Cell 9.
# - This cell fails closed if synthetic masks are missing.
# - Missing upstream synthetic masks fail closed. No local fallback masks are allowed
#   in the canonical STUDY-THESIS run.
# - VAL baselines are used for downstream VAL-only candidate selection.
# - TEST baselines are generated for final application/QA only.
# - TEST values are never used for candidate fitting or selection.
# ==========================================================

log("--- START: Cell 10.1 — Generator portfolio contract + A0/A1 VAL/TEST baselines (v3-THESIS) ---")

import os
import re
import json
import hashlib
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 0) Required globals
# ---------------------------------------------------------------------
need = [
    "CFG", "log",
    "df_tr", "df_va", "df_te",
    "PROTO_VALUE_COLS",
]

missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.1] Missing required globals: {missing}. Run Cells 1–9 first.")

# ---------------------------------------------------------------------
# 0b) Clean-run leakage guard inherited from Cell 1
# ---------------------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(
            f"[Cell10.1] Clean portfolio baseline cell forbids CFG['{_flag}']=True."
        )

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.1] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

if bool(CFG.get("cell10_allow_val_mask_bootstrap_fallback", False)):
    raise RuntimeError(
        "[Cell10.1] Canonical STUDY-THESIS run forbids cell10_allow_val_mask_bootstrap_fallback=True. "
        "Run Cell 9 v11-THESIS and use its SYN_MASKS_VAL_FULL/SYN_MASKS_TEST_FULL outputs."
    )

OUTDIR = str(CFG["outdir"])
OUT_ART = os.path.join(OUTDIR, "artifacts")
OUT_REP = os.path.join(OUTDIR, "reports")
OUT_SYN = os.path.join(OUTDIR, "synthetic")
OUT_PORT = os.path.join(OUTDIR, "portfolio_candidates")
OUT_CONTRACTS = os.path.join(OUT_ART, "contracts")

for d in [OUT_ART, OUT_REP, OUT_SYN, OUT_PORT, OUT_CONTRACTS]:
    os.makedirs(d, exist_ok=True)

seed = int(CFG.get("seed", 1337))
rng = np.random.default_rng(seed + 10010)

PORTFOLIO_REGISTRY_PATH = os.path.join(OUT_ART, "cell10_portfolio_candidate_registry.json")
PORTFOLIO_UNAVAILABLE_PATH = os.path.join(OUT_ART, "cell10_portfolio_unavailable_backends.json")
PORTFOLIO_CONTRACT_PATH = os.path.join(OUT_ART, "cell10_portfolio_contract.json")
PORTFOLIO_CONTRACT_CANONICAL_PATH = os.path.join(OUT_CONTRACTS, "cell10_portfolio_contract_v3_THESIS.json")
PORTFOLIO_FAMILY_PATH = os.path.join(OUT_ART, "cell10_protocol_family_contract.csv")
BASELINE_AUDIT_PATH = os.path.join(OUT_REP, "cell10_a0_a1_baseline_audit.csv")
MASK_AUDIT_PATH = os.path.join(OUT_REP, "cell10_a0_a1_mask_audit.csv")
BASELINE_MANIFEST_PATH = os.path.join(OUT_ART, "cell10_a0_a1_baseline_manifest.json")
BASELINE_CONTRACT_PATH = os.path.join(OUT_CONTRACTS, "cell10_a0_a1_baseline_contract_v3_THESIS.json")

# Reset registry for this run.
with open(PORTFOLIO_REGISTRY_PATH, "w", encoding="utf-8") as f:
    json.dump([], f, indent=2)

with open(PORTFOLIO_UNAVAILABLE_PATH, "w", encoding="utf-8") as f:
    json.dump([], f, indent=2)


# ---------------------------------------------------------------------
# 1) Portfolio policy
# ---------------------------------------------------------------------
GENERATOR_PORTFOLIO_SPEC = {
    "version": "cell10_generator_portfolio_v3_THESIS_a0_a1_controls",
    "selection_authority": "downstream_VAL_only_selector",
    "test_usage": "final_QA_only_no_selection_no_repair",
    "controlled_variants": {
        "A0": {
            "name": "A0_real_mask_baseline",
            "mask_source": "real_VAL_or_TEST_masks",
            "value_generator": "TRAIN_only_temporal_block_bootstrap",
            "role": "value_realism_under_real_capture_masks",
            "test_values_used": False,
        },
        "A1": {
            "name": "A1_synthetic_mask_baseline",
            "mask_source": "upstream_synthetic_masks_from_Cell9_v11_THESIS",
            "value_generator": "TRAIN_only_temporal_block_bootstrap",
            "role": "capture_realism_stress_test_baseline",
            "test_values_used": False,
            "fallback_policy": "fail_closed_no_local_fallback_in_canonical_run",
        },
        "A2": {
            "name": "A2_selected_generator_portfolio",
            "mask_source": "synthetic_masks",
            "value_generator": "VAL_selected_predefined_generator_portfolio",
            "role": "full_selected_protocol_generator_stage",
            "materialized_by": "downstream_VAL_selected_materializer",
            "test_values_used_for_selection": False,
        },
    },
    "baseline": {
        "name": "A1_temporal_block_bootstrap",
        "always_include": True,
        "role": "default_backbone_for_all_protocol_columns",
        "fit_split": "TRAIN",
        "selection_baseline_for_A2": "A1_VAL_baseline",
    },
    "generators": {
        "ddpm": {
            "role": "dense_semidense_conditional_refinement",
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "keep_drop": "keep_only_if_VAL_beats_A1",
        },
        "negative_binomial_poisson_gamma": {
            "role": "overdispersed_count_burst_candidate",
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "keep_drop": "keep_if_count_burst_distribution_improves_on_VAL",
        },
        "markov_semimarkov": {
            "role": "rare_discrete_event_dwell_transition_candidate",
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "keep_drop": "keep_if_dwell_transition_metrics_improve_on_VAL",
        },
        "gaussian_copula": {
            "role": "continuous_correlation_preservation_candidate",
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "keep_drop": "keep_if_correlation_covariance_improves_on_VAL",
        },
        "ctgan_tvae": {
            "role": "mixed_tabular_marginal_correlation_candidate",
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "keep_drop": "keep_if_non_temporal_marginal_or_correlation_metrics_improve_on_VAL",
        },
        "timegan_sequence_gan": {
            "role": "temporal_burst_autocorrelation_candidate",
            "fit_split": "TRAIN",
            "selection_split": "VAL",
            "keep_drop": "keep_if_transition_burst_autocorr_metrics_improve_on_VAL",
        },
    },
    "leakage_contract": {
        "train_used_for_value_fitting": True,
        "val_used_for_selection": True,
        "train_or_val_masks_may_fit_capture_model": True,
        "test_values_used_for_selection": False,
        "test_values_used_for_fitting": False,
        "test_masks_used_by_A0_control": True,
        "test_real_values_used_by_A0_control": False,
        "test_horizon_masks_used_for_control_only": True,
        "test_used_for_final_QA_only": True,
    },
}

globals()["GENERATOR_PORTFOLIO_SPEC"] = GENERATOR_PORTFOLIO_SPEC

with open(PORTFOLIO_CONTRACT_PATH, "w", encoding="utf-8") as f:
    json.dump(GENERATOR_PORTFOLIO_SPEC, f, indent=2)
with open(PORTFOLIO_CONTRACT_CANONICAL_PATH, "w", encoding="utf-8") as f:
    json.dump(GENERATOR_PORTFOLIO_SPEC, f, indent=2)


# ---------------------------------------------------------------------
# 2) Generic helpers
# ---------------------------------------------------------------------
TIER_MASK_COL = {
    "router": "router__obs_present",
    "ota": "ota__obs_present",
    "zigbee": "zigbee__obs_present",
    "zwave": "zwave__obs_present",
}

COUNTLIKE_RX = re.compile(
    r"(_total$|_pkt$|_pkts$|_packets$|_bytes$|_syn$|_ack$|_rst$|_fin$|"
    r"_query$|_queries$|_response$|_responses$|_req$|_reply$|_unique$|"
    r"_rcode\d+_.*$|_count$|_events$)",
    re.IGNORECASE,
)

UNIQUE_RX = re.compile(r"(uniq_|_unique$|_unique\b)", re.IGNORECASE)


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def _sha256_jsonable(obj) -> str:
    payload = json.dumps(_json_sanitize(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2)


def _read_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise RuntimeError(f"[Cell10.1] Expected list JSON at {path}, got {type(obj)}")
    return obj


def _append_json_list(path: str, row: dict) -> None:
    rows = _read_json_list(path)
    rows.append(_json_sanitize(row))
    _write_json(path, rows)


def _tier_of(col: str):
    col = str(col)
    for t in ["router", "ota", "zigbee", "zwave"]:
        if col.startswith(f"{t}__"):
            return t
    return None


def _is_countlike(col: str) -> bool:
    return bool(COUNTLIKE_RX.search(str(col)))


def _is_unique_like(col: str) -> bool:
    return bool(UNIQUE_RX.search(str(col)))


def _finite_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def _safe_quantile(x, q: float):
    x = _finite_1d(x)
    return np.nan if x.size == 0 else float(np.quantile(x, q))


def _safe_mean(x):
    x = _finite_1d(x)
    return np.nan if x.size == 0 else float(np.mean(x))


def _safe_var(x):
    x = _finite_1d(x)
    return np.nan if x.size <= 1 else float(np.var(x))


def _nonzero_rate(x):
    x = _finite_1d(x)
    return np.nan if x.size == 0 else float(np.mean(x > 0))


def _transition_rate(x):
    x = _finite_1d(x)
    if x.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x)) > 1e-9))


def _burst_rate(x):
    x = _finite_1d(x)
    if x.size < 2:
        return np.nan
    nz = x > 0
    return float(np.mean((~nz[:-1]) & (nz[1:])))


def _lag1_autocorr(x):
    x = _finite_1d(x)
    if x.size < 3:
        return np.nan
    a = x[:-1]
    b = x[1:]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _ks_stat_fast(a, b):
    a = np.sort(_finite_1d(a))
    b = np.sort(_finite_1d(b))
    if a.size == 0 or b.size == 0:
        return np.nan
    vals = np.sort(np.unique(np.concatenate([a, b])))
    if vals.size == 0:
        return np.nan
    ca = np.searchsorted(a, vals, side="right") / float(a.size)
    cb = np.searchsorted(b, vals, side="right") / float(b.size)
    return float(np.max(np.abs(ca - cb)))


def _wasserstein_1d_fast(a, b, max_points: int = 5000):
    a = _finite_1d(a)
    b = _finite_1d(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    m = int(min(max_points, max(10, min(a.size, b.size))))
    qs = np.linspace(0.0, 1.0, m)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _postprocess_protocol_values(col: str, x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    finite = np.isfinite(x)
    if finite.any():
        x[finite] = np.maximum(x[finite], 0.0)
        if _is_countlike(col):
            x[finite] = np.rint(x[finite])
    x[~np.isfinite(x)] = np.nan
    return x.astype(np.float32, copy=False)


def _active_mask_from_df(df_part: pd.DataFrame, tier: str, *, default_if_missing: str = "error") -> np.ndarray:
    c = TIER_MASK_COL.get(tier)
    if c is not None and c in df_part.columns:
        v = pd.to_numeric(df_part[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        return (v > 0.5)

    if default_if_missing == "one":
        return np.ones(len(df_part), dtype=bool)
    if default_if_missing == "zero":
        return np.zeros(len(df_part), dtype=bool)

    raise RuntimeError(f"[Cell10.1] Missing mask column for tier={tier}: {c}")


def _real_mask_dict(df_part: pd.DataFrame, *, label: str) -> dict:
    out = {}
    for tier in ["router", "ota", "zigbee", "zwave"]:
        c = TIER_MASK_COL[tier]
        if c in df_part.columns:
            out[tier] = _active_mask_from_df(df_part, tier, default_if_missing="error")
        elif tier == "zwave":
            out[tier] = np.zeros(len(df_part), dtype=bool)
        else:
            raise RuntimeError(f"[Cell10.1] Missing real {label} mask column for tier={tier}: {c}")
    return out


def _mask_for_tier(mask_dict: dict, tier: str, N: int, *, missing_policy: str = "error") -> np.ndarray:
    if tier in mask_dict:
        m = np.asarray(mask_dict[tier], dtype=bool)
        if m.shape[0] != N:
            raise RuntimeError(f"[Cell10.1] Mask length mismatch tier={tier}: {m.shape[0]} vs {N}")
        return m

    if missing_policy == "zero":
        return np.zeros(N, dtype=bool)
    if missing_policy == "one":
        return np.ones(N, dtype=bool)

    raise RuntimeError(f"[Cell10.1] Missing mask for tier={tier}")


def _values_on_active(df_part: pd.DataFrame, col: str, active: np.ndarray) -> np.ndarray:
    if col not in df_part.columns:
        return np.array([], dtype=np.float32)

    active = np.asarray(active, dtype=bool)
    if active.shape[0] != len(df_part):
        raise RuntimeError(
            f"[Cell10.1] Active mask length mismatch for {col}: "
            f"mask={active.shape[0]} df={len(df_part)}"
        )

    x = pd.to_numeric(df_part.loc[active, col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    x = x[np.isfinite(x)]
    if x.size:
        x = np.maximum(x, 0.0)
    return x.astype(np.float32, copy=False)


def _column_train_values(col: str, *, active_only: bool = True) -> np.ndarray:
    tier = _tier_of(col)
    if col not in df_tr.columns:
        return np.array([], dtype=np.float32)

    if active_only and tier is not None and TIER_MASK_COL.get(tier) in df_tr.columns:
        active = _active_mask_from_df(df_tr, tier, default_if_missing="one")
        return _values_on_active(df_tr, col, active)

    x = pd.to_numeric(df_tr[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    x = x[np.isfinite(x)]
    if x.size:
        x = np.maximum(x, 0.0)
    return x.astype(np.float32, copy=False)


def _register_portfolio_candidate(record: dict):
    required = ["generator", "candidate_id", "columns", "val_path", "test_path", "fit_split", "selection_split"]
    missing = [k for k in required if k not in record]
    if missing:
        raise RuntimeError(f"[Cell10.1] Candidate registry record missing keys: {missing}")

    rec = dict(record)
    rec.setdefault("version", "cell10_candidate_record_v2")
    rec.setdefault("test_used_for_selection", False)
    rec.setdefault("train_used_for_fitting", True)
    rec.setdefault("created_by", "Cell10.x")
    rec["columns"] = [str(c) for c in rec["columns"]]
    rec["columns_n"] = int(len(rec["columns"]))
    rec["record_sha256"] = _sha256_jsonable({k: v for k, v in rec.items() if k != "record_sha256"})
    _append_json_list(PORTFOLIO_REGISTRY_PATH, rec)
    return rec


def _register_unavailable_backend(generator: str, reason: str, meta: dict = None):
    row = {
        "generator": generator,
        "available": False,
        "reason": str(reason),
        "meta": dict(meta or {}),
    }
    _append_json_list(PORTFOLIO_UNAVAILABLE_PATH, row)
    log(f"[Cell10.1][UNAVAILABLE] {generator}: {reason}")


globals()["_cell10_register_portfolio_candidate"] = _register_portfolio_candidate
globals()["_cell10_register_unavailable_backend"] = _register_unavailable_backend


# ---------------------------------------------------------------------
# 3) Synthetic mask helpers
# ---------------------------------------------------------------------
def _tiers_from_matrix_width(width: int) -> list:
    width = int(width)
    base = ["router", "ota", "zigbee"]
    if width >= 4:
        base.append("zwave")
    return base[:width]


def _matrix_to_mask_dict(M, N: int, tiers: list, *, label: str) -> dict:
    M = np.asarray(M)
    if M.ndim != 2 or M.shape[0] != N:
        raise RuntimeError(f"[Cell10.1] {label} synthetic mask shape mismatch: got={M.shape}, N={N}")

    if len(tiers) != M.shape[1]:
        raise RuntimeError(
            f"[Cell10.1] {label} tier/matrix width mismatch: tiers={len(tiers)} width={M.shape[1]}"
        )

    out = {}
    for j, t in enumerate(tiers):
        out[str(t)] = (M[:, j].astype(np.int8, copy=False) > 0)

    if "zwave" not in out:
        out["zwave"] = np.zeros(N, dtype=bool)

    for t in ["router", "ota", "zigbee"]:
        if t not in out:
            raise RuntimeError(f"[Cell10.1] {label} synthetic masks missing required tier={t}")

    return out


def _get_upstream_syn_mask_dict_for_horizon(N: int, horizon: str):
    """
    Try to find upstream synthetic masks for VAL/TEST.

    Supported common global names:
      VAL:
        SYN_MASKS_VAL_FULL, SYN_MASKS_FULL_VAL, SYN_MASKS_VAL,
        SYN_MASKS_DDPM_VAL, CELL9_SYN_MASKS_VAL
      TEST:
        SYN_MASKS_FULL, SYN_MASKS_TEST_FULL, SYN_MASKS_FULL_TEST,
        SYN_MASKS_TEST, SYN_MASKS_DDPM, SYN_MASKS_DDPM_TEST,
        CELL9_SYN_MASKS_TEST
    """
    horizon = str(horizon).upper()

    if horizon == "VAL":
        candidates = [
            ("SYN_MASKS_VAL_FULL", None),
            ("SYN_MASKS_FULL_VAL", None),
            ("SYN_MASKS_VAL", None),
            ("SYN_MASKS_DDPM_VAL", "DDPM_TIERS_USED"),
            ("CELL9_SYN_MASKS_VAL", None),
        ]
    elif horizon == "TEST":
        candidates = [
            ("SYN_MASKS_TEST_FULL", None),
            ("SYN_MASKS_FULL_TEST", None),
            ("SYN_MASKS_FULL", None),  # Cell 9 backward-compatible alias for TEST
            ("SYN_MASKS_TEST", None),
            ("SYN_MASKS_DDPM_TEST", "DDPM_TIERS_USED"),
            ("SYN_MASKS_DDPM", "DDPM_TIERS_USED"),
            ("CELL9_SYN_MASKS_TEST", None),
        ]
    else:
        raise RuntimeError(f"[Cell10.1] Unknown horizon={horizon}")

    for var_name, tier_var in candidates:
        if var_name not in globals():
            continue

        M = np.asarray(globals()[var_name])
        if M.ndim != 2 or M.shape[0] != N:
            continue

        if tier_var is not None and tier_var in globals():
            tiers = [str(x) for x in list(globals()[tier_var])]
        else:
            tiers = _tiers_from_matrix_width(M.shape[1])

        mask_dict = _matrix_to_mask_dict(M, N, tiers, label=f"{horizon}:{var_name}")
        return mask_dict, {
            "source": var_name,
            "method": "upstream_synthetic_mask_matrix",
            "tiers": sorted(mask_dict.keys()),
        }

    return None, {
        "source": None,
        "method": "not_found",
        "tiers": [],
    }



def _synthetic_mask_dict_for_horizon(N: int, horizon: str) -> tuple:
    """
    Strict synthetic-mask resolver.

    STUDY-THESIS contract:
    - Cell 9 is the authority for synthetic VAL/TEST capture masks.
    - Cell 10.1 must not silently invent fallback masks, because that would make
      A1 semantics depend on an undocumented local fallback.
    - No local fallback is allowed in the canonical STUDY-THESIS run.
    """
    upstream, meta = _get_upstream_syn_mask_dict_for_horizon(N, horizon)
    if upstream is not None:
        return upstream, meta

    raise RuntimeError(
        f"[Cell10.1] Missing upstream synthetic masks for horizon={horizon}. "
        "Run Cell 9 v11-THESIS first and ensure it publishes SYN_MASKS_VAL_FULL and "
        "SYN_MASKS_TEST_FULL/SYN_MASKS_FULL. Canonical STUDY-THESIS runs do not allow "
        "local VAL-fitted fallback masks because that would change A1 semantics."
    )


# ---------------------------------------------------------------------
# 4) Protocol family contract
# ---------------------------------------------------------------------
PROTO_COLS = [
    str(c)
    for c in list(PROTO_VALUE_COLS)
    if str(c) in df_tr.columns or str(c) in df_va.columns or str(c) in df_te.columns
]

if not PROTO_COLS:
    raise RuntimeError("[Cell10.1] No protocol columns found in current splits.")

family_rows = []

for c in PROTO_COLS:
    tier = _tier_of(c)
    x = _column_train_values(c, active_only=True)

    mean = _safe_mean(x)
    var = _safe_var(x)
    nz = _nonzero_rate(x)
    trans = _transition_rate(x)
    ac1 = _lag1_autocorr(x)
    countlike = _is_countlike(c)
    unique_like = _is_unique_like(c)
    finite_x = _finite_1d(x)
    unique_n = int(len(np.unique(np.round(finite_x, 6)))) if finite_x.size else 0
    vtm = float(var / max(mean, 1e-9)) if np.isfinite(var) and np.isfinite(mean) and mean > 1e-9 else np.nan

    family = "continuous"
    eligible = []

    if countlike:
        family = "count"
        eligible.append("negative_binomial_poisson_gamma")
        if np.isfinite(vtm) and vtm >= float(CFG.get("cell10_nb_vtm_threshold", 1.25)):
            eligible.append("negative_binomial_poisson_gamma_overdispersed")

    if countlike and np.isfinite(nz) and nz <= float(CFG.get("cell10_sparse_event_nz_thr", 0.20)):
        family = "sparse_event"
        eligible.append("markov_semimarkov")

    if (not countlike) and unique_n >= int(CFG.get("cell10_copula_min_unique", 20)):
        family = "continuous"
        eligible.append("gaussian_copula")

    if tier in {"router", "ota", "zigbee"}:
        eligible.append("ctgan_tvae")

    if np.isfinite(ac1) or np.isfinite(trans):
        eligible.append("timegan_sequence_gan")

    if c in set(map(str, globals().get("ddpm_cols_use", []))):
        eligible.append("ddpm")

    family_rows.append({
        "col": c,
        "tier": tier,
        "family": family,
        "countlike": bool(countlike),
        "unique_like": bool(unique_like),
        "train_n": int(x.size),
        "train_mean": mean,
        "train_var": var,
        "variance_to_mean": vtm,
        "train_nonzero_rate": nz,
        "train_transition_rate": trans,
        "train_lag1_autocorr": ac1,
        "train_unique_n": unique_n,
        "eligible_generators": "|".join(sorted(set(eligible))),
    })

family_df = pd.DataFrame(family_rows)
family_df.to_csv(PORTFOLIO_FAMILY_PATH, index=False)

globals()["CELL10_PROTO_COLS"] = list(PROTO_COLS)
globals()["CELL10_FAMILY_DF"] = family_df


# ---------------------------------------------------------------------
# 5) Real and synthetic masks for A0/A1
# ---------------------------------------------------------------------
A0_VAL_MASKS = _real_mask_dict(df_va, label="VAL")
A0_TEST_MASKS = _real_mask_dict(df_te, label="TEST")

A1_VAL_MASKS, A1_VAL_MASK_META = _synthetic_mask_dict_for_horizon(len(df_va), "VAL")
A1_TEST_MASKS, A1_TEST_MASK_META = _synthetic_mask_dict_for_horizon(len(df_te), "TEST")

globals()["CELL10_A0_VAL_MASKS"] = A0_VAL_MASKS
globals()["CELL10_A0_TEST_MASKS"] = A0_TEST_MASKS
globals()["CELL10_A1_VAL_MASKS"] = A1_VAL_MASKS
globals()["CELL10_A1_TEST_MASKS"] = A1_TEST_MASKS

# Backward-compatible aliases expected by later cells.
globals()["CELL10_VAL_MASKS"] = A1_VAL_MASKS
globals()["CELL10_TEST_MASKS"] = A1_TEST_MASKS


# ---------------------------------------------------------------------
# 6) Baseline builder
# ---------------------------------------------------------------------
def _stable_seed_from_parts(*parts, base_seed: int = 1337) -> int:
    payload = "::".join(str(p) for p in parts)
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int((int(h[:12], 16) + int(base_seed)) % (2**32 - 1))

def _build_protocol_baseline_for_horizon(
    df_horizon: pd.DataFrame,
    mask_dict: dict,
    horizon_name: str,
    variant_name: str,
) -> tuple:
    """
    TRAIN-only non-copying temporal block-bootstrap baseline.

    Values are sampled only from TRAIN active rows. The horizon dataframe supplies
    only row count and, for A0, real masks. It never supplies values.
    """
    N_h = int(len(df_horizon))

    tier_to_cols = defaultdict(list)
    for c in PROTO_COLS:
        tier = _tier_of(c)
        if tier is not None:
            tier_to_cols[tier].append(c)

    out = {}
    audit_rows = []

    block_len = int(CFG.get("cell10_a1_block_len", CFG.get("cell11_tier_block_len", 120)))
    min_train_rows = int(CFG.get("cell10_a1_min_train_rows", 200))

    for tier, cols in sorted(tier_to_cols.items()):
        missing_policy = "zero" if tier == "zwave" else "error"
        active_h = _mask_for_tier(mask_dict, tier, N_h, missing_policy=missing_policy)

        for c in cols:
            out[c] = np.full(N_h, np.nan, dtype=np.float32)

        if active_h.sum() == 0:
            for c in cols:
                audit_rows.append({
                    "variant": variant_name,
                    "horizon": horizon_name,
                    "tier": tier,
                    "col": c,
                    "active_horizon_n": int(active_h.sum()),
                    "train_active_rows": 0,
                    "method": "zero_active_mask",
                    "finite_active_values": 0,
                    "inactive_finite_values": 0,
                })
            continue

        train_active = _active_mask_from_df(
            df_tr,
            tier,
            default_if_missing="zero" if tier == "zwave" else "one",
        )

        cols_present = [c for c in cols if c in df_tr.columns]
        if not cols_present:
            raise RuntimeError(f"[Cell10.1] No TRAIN columns for tier={tier}")

        X = df_tr.loc[train_active, cols_present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        row_ok = np.isfinite(X).any(axis=1)
        X = X[row_ok]

        if X.shape[0] < min_train_rows:
            raise RuntimeError(
                f"[Cell10.1] Too few TRAIN active rows for {variant_name} {horizon_name} tier={tier}: "
                f"{X.shape[0]} < {min_train_rows}"
            )

        for j, c in enumerate(cols_present):
            col = X[:, j]
            finite = np.isfinite(col)
            fill = np.float32(0.0 if not finite.any() else np.nanmedian(col[finite]))
            col[~finite] = fill
            X[:, j] = _postprocess_protocol_values(c, col)

        target_idx = np.flatnonzero(active_h)
        pos = 0
        block_draws = 0

        local_rng = np.random.default_rng(
            _stable_seed_from_parts(
                "cell10_1_baseline",
                variant_name,
                horizon_name,
                tier,
                seed,
                base_seed=seed + 10010,
            )
        )

        while pos < target_idx.size:
            if X.shape[0] <= block_len:
                block = X
            else:
                s = int(local_rng.integers(0, X.shape[0] - block_len + 1))
                block = X[s:s + block_len]

            take = int(min(block.shape[0], target_idx.size - pos))
            idx = target_idx[pos:pos + take]

            for j, c in enumerate(cols_present):
                out[c][idx] = block[:take, j]

            pos += take
            block_draws += 1

        for c in cols:
            out[c] = _postprocess_protocol_values(c, out[c])

            x = np.asarray(out[c], dtype=np.float64)
            audit_rows.append({
                "variant": variant_name,
                "horizon": horizon_name,
                "tier": tier,
                "col": c,
                "active_horizon_n": int(active_h.sum()),
                "train_active_rows": int(X.shape[0]),
                "method": "TRAIN_temporal_block_bootstrap",
                "block_len": int(block_len),
                "block_draws": int(block_draws),
                "finite_active_values": int(np.isfinite(x[active_h]).sum()),
                "inactive_finite_values": int(np.isfinite(x[~active_h]).sum()),
                "active_nonzero_rate": _nonzero_rate(x[active_h]),
                "active_mean": _safe_mean(x[active_h]),
                "active_q50": _safe_quantile(x[active_h], 0.50),
                "active_q95": _safe_quantile(x[active_h], 0.95),
            })

    df_out = pd.DataFrame(out)

    # Strict mask contract validation.
    for c in PROTO_COLS:
        tier = _tier_of(c)
        if tier is None:
            continue

        active = _mask_for_tier(mask_dict, tier, N_h, missing_policy="zero" if tier == "zwave" else "error")
        x = pd.to_numeric(df_out[c], errors="coerce").to_numpy(dtype=np.float64)

        inactive_finite = int(np.isfinite(x[~active]).sum())
        if inactive_finite > 0:
            raise RuntimeError(
                f"[Cell10.1] {variant_name}_{horizon_name}.{c} has finite values on inactive {tier} rows: "
                f"{inactive_finite}"
            )

        if active.sum() > 0 and int(np.isfinite(x[active]).sum()) == 0:
            raise RuntimeError(
                f"[Cell10.1] {variant_name}_{horizon_name}.{c} has no finite values on active rows."
            )

    return df_out, pd.DataFrame(audit_rows)


# ---------------------------------------------------------------------
# 7) Build A0 and A1 baselines
# ---------------------------------------------------------------------
A0_VAL, A0_VAL_AUDIT = _build_protocol_baseline_for_horizon(
    df_horizon=df_va,
    mask_dict=A0_VAL_MASKS,
    horizon_name="VAL",
    variant_name="A0_REAL_MASK_BASELINE",
)

A0_TEST, A0_TEST_AUDIT = _build_protocol_baseline_for_horizon(
    df_horizon=df_te,
    mask_dict=A0_TEST_MASKS,
    horizon_name="TEST",
    variant_name="A0_REAL_MASK_BASELINE",
)

A1_VAL, A1_VAL_AUDIT = _build_protocol_baseline_for_horizon(
    df_horizon=df_va,
    mask_dict=A1_VAL_MASKS,
    horizon_name="VAL",
    variant_name="A1_SYN_MASK_BASELINE",
)

A1_TEST, A1_TEST_AUDIT = _build_protocol_baseline_for_horizon(
    df_horizon=df_te,
    mask_dict=A1_TEST_MASKS,
    horizon_name="TEST",
    variant_name="A1_SYN_MASK_BASELINE",
)


# ---------------------------------------------------------------------
# 8) Save baseline artifacts
# ---------------------------------------------------------------------
A0_VAL_PATH = os.path.join(OUT_PORT, "A0_VAL_baseline.parquet")
A0_TEST_PATH = os.path.join(OUT_PORT, "A0_TEST_baseline.parquet")
A1_VAL_PATH = os.path.join(OUT_PORT, "A1_VAL_baseline.parquet")
A1_TEST_PATH = os.path.join(OUT_PORT, "A1_TEST_baseline.parquet")

A0_VAL.to_parquet(A0_VAL_PATH, index=False)
A0_TEST.to_parquet(A0_TEST_PATH, index=False)
A1_VAL.to_parquet(A1_VAL_PATH, index=False)
A1_TEST.to_parquet(A1_TEST_PATH, index=False)

baseline_audit = pd.concat(
    [A0_VAL_AUDIT, A0_TEST_AUDIT, A1_VAL_AUDIT, A1_TEST_AUDIT],
    axis=0,
    ignore_index=True,
)
baseline_audit.to_csv(BASELINE_AUDIT_PATH, index=False)


# ---------------------------------------------------------------------
# 9) Mask audit
# ---------------------------------------------------------------------
mask_rows = []

for variant, horizon, masks, N_h, meta in [
    ("A0_REAL_MASK_BASELINE", "VAL", A0_VAL_MASKS, len(df_va), {"source": "df_va_real_masks", "method": "real_masks"}),
    ("A0_REAL_MASK_BASELINE", "TEST", A0_TEST_MASKS, len(df_te), {"source": "df_te_real_masks", "method": "real_masks"}),
    ("A1_SYN_MASK_BASELINE", "VAL", A1_VAL_MASKS, len(df_va), A1_VAL_MASK_META),
    ("A1_SYN_MASK_BASELINE", "TEST", A1_TEST_MASKS, len(df_te), A1_TEST_MASK_META),
]:
    for tier in ["router", "ota", "zigbee", "zwave"]:
        m = _mask_for_tier(masks, tier, N_h, missing_policy="zero" if tier == "zwave" else "error")
        mask_rows.append({
            "variant": variant,
            "horizon": horizon,
            "tier": tier,
            "N": int(N_h),
            "active_n": int(m.sum()),
            "active_rate": float(m.mean()) if m.size else np.nan,
            "mask_source": meta.get("source"),
            "mask_method": meta.get("method"),
            "mask_meta": json.dumps(_json_sanitize(meta), sort_keys=True),
        })

mask_audit = pd.DataFrame(mask_rows)
mask_audit.to_csv(MASK_AUDIT_PATH, index=False)


# ---------------------------------------------------------------------
# 10) Manifest, globals, compatibility aliases
# ---------------------------------------------------------------------
def _mask_rate_summary(mask_dict: dict, N: int) -> dict:
    out = {}
    for tier in ["router", "ota", "zigbee", "zwave"]:
        m = _mask_for_tier(mask_dict, tier, N, missing_policy="zero" if tier == "zwave" else "error")
        out[tier] = float(np.mean(m)) if m.size else np.nan
    return out


def _mask_rate_drift(a: dict, b: dict) -> dict:
    out = {}
    for tier in ["router", "ota", "zigbee", "zwave"]:
        av = a.get(tier, np.nan)
        bv = b.get(tier, np.nan)
        out[tier] = float(av - bv) if np.isfinite(av) and np.isfinite(bv) else np.nan
    return out


A0_VAL_MASK_RATES = _mask_rate_summary(A0_VAL_MASKS, len(df_va))
A0_TEST_MASK_RATES = _mask_rate_summary(A0_TEST_MASKS, len(df_te))
A1_VAL_MASK_RATES = _mask_rate_summary(A1_VAL_MASKS, len(df_va))
A1_TEST_MASK_RATES = _mask_rate_summary(A1_TEST_MASKS, len(df_te))

A1_VAL_MINUS_REAL_VAL_MASK_DRIFT = _mask_rate_drift(A1_VAL_MASK_RATES, A0_VAL_MASK_RATES)
A1_TEST_MINUS_REAL_TEST_MASK_DRIFT = _mask_rate_drift(A1_TEST_MASK_RATES, A0_TEST_MASK_RATES)

manifest = {
    "version": "cell10_1_v3_THESIS_a0_a1_baselines",
    "seed": int(seed),
    "proto_cols_n": int(len(PROTO_COLS)),
    "proto_cols": list(PROTO_COLS),
    "outputs": {
        "A0_VAL_baseline": A0_VAL_PATH,
        "A0_TEST_baseline": A0_TEST_PATH,
        "A1_VAL_baseline": A1_VAL_PATH,
        "A1_TEST_baseline": A1_TEST_PATH,
    },
    "reports": {
        "baseline_audit": BASELINE_AUDIT_PATH,
        "mask_audit": MASK_AUDIT_PATH,
        "family_contract": PORTFOLIO_FAMILY_PATH,
        "portfolio_contract": PORTFOLIO_CONTRACT_PATH,
        "portfolio_contract_canonical": PORTFOLIO_CONTRACT_CANONICAL_PATH,
        "baseline_contract": BASELINE_CONTRACT_PATH,
        "candidate_registry": PORTFOLIO_REGISTRY_PATH,
        "unavailable_backends": PORTFOLIO_UNAVAILABLE_PATH,
    },
    "mask_sources": {
        "A0_VAL": {"source": "df_va_real_masks", "method": "real_masks"},
        "A0_TEST": {"source": "df_te_real_masks", "method": "real_masks"},
        "A1_VAL": A1_VAL_MASK_META,
        "A1_TEST": A1_TEST_MASK_META,
    },
    "mask_rate_summary": {
        "A0_VAL_real_mask_rates": A0_VAL_MASK_RATES,
        "A0_TEST_real_mask_rates": A0_TEST_MASK_RATES,
        "A1_VAL_synthetic_mask_rates": A1_VAL_MASK_RATES,
        "A1_TEST_synthetic_mask_rates": A1_TEST_MASK_RATES,
        "A1_VAL_minus_REAL_VAL_mask_drift": A1_VAL_MINUS_REAL_VAL_MASK_DRIFT,
        "A1_TEST_minus_REAL_TEST_mask_drift_eval_only": A1_TEST_MINUS_REAL_TEST_MASK_DRIFT,
    },
    "leakage_contract": {
        "train_values_used_for_baselines": True,
        "val_values_used_for_baseline_generation": False,
        "test_values_used_for_baseline_generation": False,
        "val_masks_used_by_A0_control_only": True,
        "test_masks_used_by_A0_control_only": True,
        "a0_uses_real_masks_only_as_capture_process": True,
        "a1_uses_synthetic_masks_from_cell9": True,
        "test_values_used_for_selection": False,
        "test_values_used_for_fitting": False,
        "test_values_used_for_repair": False,
        "test_real_masks_used_for_model_selection": False,
    },
    "artifact_hashes": {
        "A0_VAL_sha256": _sha256_file(A0_VAL_PATH),
        "A0_TEST_sha256": _sha256_file(A0_TEST_PATH),
        "A1_VAL_sha256": _sha256_file(A1_VAL_PATH),
        "A1_TEST_sha256": _sha256_file(A1_TEST_PATH),
        "baseline_audit_sha256": _sha256_file(BASELINE_AUDIT_PATH),
        "mask_audit_sha256": _sha256_file(MASK_AUDIT_PATH),
        "family_contract_sha256": _sha256_file(PORTFOLIO_FAMILY_PATH),
        "portfolio_contract_sha256": _sha256_file(PORTFOLIO_CONTRACT_PATH),
        "portfolio_contract_canonical_sha256": _sha256_file(PORTFOLIO_CONTRACT_CANONICAL_PATH),
    },
}

CELL10_A0_A1_BASELINE_CONTRACT = dict(manifest)
CELL10_A0_A1_BASELINE_CONTRACT["component"] = "cell10_1_a0_a1_baseline_contract"
CELL10_A0_A1_BASELINE_CONTRACT["component_type"] = "portfolio_baseline_generator_no_selection"
CELL10_A0_A1_BASELINE_CONTRACT["selection_authority"] = "downstream_VAL_only_selector"
CELL10_A0_A1_BASELINE_CONTRACT["claim_status"] = (
    "Provides A0/A1 baseline artifacts and portfolio registration utilities only; "
    "does not select final A2 generators and does not support protocol realism claims by itself."
)

_write_json(BASELINE_MANIFEST_PATH, manifest)
_write_json(BASELINE_CONTRACT_PATH, CELL10_A0_A1_BASELINE_CONTRACT)

globals()["CELL10_PROTO_COLS"] = list(PROTO_COLS)
globals()["CELL10_A0_VAL_PATH"] = A0_VAL_PATH
globals()["CELL10_A0_TEST_PATH"] = A0_TEST_PATH
globals()["CELL10_A1_VAL_PATH"] = A1_VAL_PATH
globals()["CELL10_A1_TEST_PATH"] = A1_TEST_PATH

globals()["CELL10_A0_VAL"] = A0_VAL
globals()["CELL10_A0_TEST"] = A0_TEST
globals()["CELL10_A1_VAL"] = A1_VAL
globals()["CELL10_A1_TEST"] = A1_TEST

globals()["CELL10_A0_VAL_MASKS"] = A0_VAL_MASKS
globals()["CELL10_A0_TEST_MASKS"] = A0_TEST_MASKS
globals()["CELL10_A1_VAL_MASKS"] = A1_VAL_MASKS
globals()["CELL10_A1_TEST_MASKS"] = A1_TEST_MASKS

# Backward-compatible aliases expected by downstream portfolio candidate cells.
globals()["CELL10_VAL_MASKS"] = A1_VAL_MASKS
globals()["CELL10_TEST_MASKS"] = A1_TEST_MASKS
globals()["CELL10_PORTFOLIO_BASELINE_MANIFEST"] = manifest
globals()["CELL10_A0_A1_BASELINE_CONTRACT"] = CELL10_A0_A1_BASELINE_CONTRACT
globals()["CELL10_A0_A1_BASELINE_CONTRACT_PATH"] = BASELINE_CONTRACT_PATH
globals()["CELL10_BASELINE_AUDIT"] = baseline_audit
globals()["CELL10_MASK_AUDIT"] = mask_audit

if "RUN_META" in globals():
    RUN_META.setdefault("generator_contracts", {})
    RUN_META["generator_contracts"]["Cell10_1_A0_A1_baselines"] = CELL10_A0_A1_BASELINE_CONTRACT
    RUN_META.setdefault("portfolio_contracts", {})
    RUN_META["portfolio_contracts"]["Cell10_generator_portfolio"] = GENERATOR_PORTFOLIO_SPEC

log(f"[Cell10.1] Saved A0_VAL baseline: {A0_VAL_PATH} | shape={A0_VAL.shape}")
log(f"[Cell10.1] Saved A0_TEST baseline: {A0_TEST_PATH} | shape={A0_TEST.shape}")
log(f"[Cell10.1] Saved A1_VAL baseline: {A1_VAL_PATH} | shape={A1_VAL.shape}")
log(f"[Cell10.1] Saved A1_TEST baseline: {A1_TEST_PATH} | shape={A1_TEST.shape}")
log(f"[Cell10.1] Saved baseline audit: {BASELINE_AUDIT_PATH}")
log(f"[Cell10.1] Saved mask audit: {MASK_AUDIT_PATH}")
log(f"[Cell10.1] Saved family contract: {PORTFOLIO_FAMILY_PATH} | rows={len(family_df)}")
log(f"[Cell10.1] Saved portfolio contract: {PORTFOLIO_CONTRACT_PATH}")
log(f"[Cell10.1] Saved canonical portfolio contract: {PORTFOLIO_CONTRACT_CANONICAL_PATH}")
log(f"[Cell10.1] Saved baseline manifest: {BASELINE_MANIFEST_PATH}")
log(f"[Cell10.1] Saved baseline contract: {BASELINE_CONTRACT_PATH}")

log(
    "[Cell10.1] Mask source summary | "
    f"A1_VAL={A1_VAL_MASK_META.get('method')}:{A1_VAL_MASK_META.get('source')} | "
    f"A1_TEST={A1_TEST_MASK_META.get('method')}:{A1_TEST_MASK_META.get('source')}"
)

log("--- END: Cell 10.1 — Generator portfolio contract + A0/A1 VAL/TEST baselines (v3-THESIS) ---")