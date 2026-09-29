# ==========================================================
# CELL 11 — Final A0/A1/A2 protocol variant QA + publication
# v13.1-THESIS STUDY-THESIS-GRADE
#
# Purpose:
# - Load the locked protocol variants materialized by Cell 10.9:
#     A0_PROTOCOL_TEST
#     A1_PROTOCOL_TEST
#     A2_PROTOCOL_TEST
#
# - Validate the controlled experimental semantics:
#     A0 = TRAIN-only value baseline under real TEST masks.
#     A1 = TRAIN-only value baseline under synthetic TEST masks.
#     A2 = VAL-selected predefined generator-portfolio output under synthetic TEST masks.
#
# - Evaluate A0/A1/A2 against real TEST protocol values for final QA only.
#
# Scientific contract:
# - This cell does NOT generate values.
# - This cell does NOT select generators.
# - This cell does NOT fit models.
# - This cell does NOT use TEST for candidate acceptance.
# - TEST is used only as final reference for QA/reporting.
#
# Required upstream cells:
# - Cell 9: synthetic VAL/TEST masks.
# - Cell 10.1: A0/A1 baselines.
# - Cells 10.2–10.7: optional predefined candidate materialization.
# - Cell 10.8: VAL-only portfolio selection.
# - Cell 10.9: A0/A1/A2 protocol variant materialization.
# - PRE-CELL11: runtime contract guard.
# ==========================================================

log("--- START: Cell 11 — Final A0/A1/A2 protocol variant QA + publication (v13.1-THESIS contract-locked) ---")

import os
import re
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 0) Required globals
# ---------------------------------------------------------------------
need = [
    "CFG",
    "log",
    "df_te",
    "PROTO_VALUE_COLS",
]

missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell11] Missing required globals: {missing}. Run Cells 1–10.9 first.")

OUTDIR = str(CFG["outdir"])
OUT_SYN = os.path.join(OUTDIR, "synthetic")
OUT_ART = os.path.join(OUTDIR, "artifacts")
OUT_REP = os.path.join(OUTDIR, "reports")
OUT_CONTRACTS = os.path.join(OUT_ART, "contracts")

for d in [OUT_SYN, OUT_ART, OUT_REP, OUT_CONTRACTS]:
    os.makedirs(d, exist_ok=True)

N_TEST = int(len(df_te))
if N_TEST <= 0:
    raise RuntimeError("[Cell11] df_te is empty.")

seed = int(CFG.get("seed", 1337))

# Fail closed if any TEST-informed fitting/selection/repair flag is enabled.
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[Cell11] Clean final QA forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell11] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

CELL10_9_MANIFEST_PATH = os.path.join(OUT_ART, "cell10_9_protocol_variant_manifest.json")
CELL10_SELECTION_JSON = os.path.join(OUT_ART, "cell10_selected_generator_by_col.json")
CELL10_SELECTION_RICH_JSON = os.path.join(OUT_ART, "cell10_selected_generator_by_col_rich.json")
CELL10_SELECTION_SUMMARY_JSON = os.path.join(OUT_ART, "cell10_portfolio_selection_summary.json")

# Canonical contract-locked upstream guards.
CELL10_8_SELECTOR_CONTRACT_PATH = os.path.join(
    OUT_CONTRACTS, "cell10_8_val_only_selector_contract_v3_6_THESIS.json"
)
CELL10_9_MATERIALIZER_CONTRACT_PATH = os.path.join(
    OUT_CONTRACTS, "cell10_9_protocol_variant_materialization_contract_v2_1_THESIS.json"
)
PRECELL11_CONTRACT_PATH = os.path.join(
    OUT_CONTRACTS, "pre_cell11_a0_a1_a2_variant_runtime_contract_v4_1_THESIS.json"
)
# Legacy path retained only for backward-compatibility metadata.
PRECELL11_LEGACY_CONTRACT_PATH = os.path.join(OUT_ART, "pre_cell11_a0_a1_a2_variant_runtime_contract.json")

FINAL_MANIFEST_PATH = os.path.join(OUT_ART, "cell11_final_protocol_variant_manifest.json")
FINAL_AUDIT_PATH = os.path.join(OUT_REP, "cell11_final_protocol_variant_audit.json")

QUALITY_REPORT_PATH = os.path.join(OUT_REP, "cell11_a0_a1_a2_protocol_quality_report.csv")
MASK_CONTRACT_PATH = os.path.join(OUT_REP, "cell11_a0_a1_a2_mask_contract_audit.csv")
DELTA_AUDIT_PATH = os.path.join(OUT_REP, "cell11_a0_a1_a2_delta_audit.csv")
SELECTION_CONSISTENCY_PATH = os.path.join(OUT_REP, "cell11_selection_consistency_audit.csv")
SUMMARY_CSV_PATH = os.path.join(OUT_REP, "cell11_variant_quality_summary.csv")
HASHES_PATH = os.path.join(OUT_REP, "cell11_final_protocol_artifact_hashes.json")

# ---------------------------------------------------------------------
# Phase 1 quality-decomposition scaffold artifacts
# ---------------------------------------------------------------------
QUALITY_DIMENSION_REGISTRY_PATH = os.path.join(
    OUT_ART, "quality_dimension_registry.json"
)
A0_A1_A2_VARIANT_MANIFEST_PATH = os.path.join(
    OUT_ART, "a0_a1_a2_variant_manifest.json"
)
QUALITY_ATTRIBUTION_MATRIX_PATH = os.path.join(
    OUT_REP, "quality_attribution_matrix.csv"
)

FINAL_A0_PATH = os.path.join(OUT_SYN, "A0_PROTOCOL_FINAL.parquet")
FINAL_A1_PATH = os.path.join(OUT_SYN, "A1_PROTOCOL_FINAL.parquet")
FINAL_A2_PATH = os.path.join(OUT_SYN, "A2_PROTOCOL_FINAL.parquet")


# ---------------------------------------------------------------------
# 1) Helpers
# ---------------------------------------------------------------------
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


def _write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2)


def _read_json_if_exists(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_json_dict_required(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"[Cell11] Missing required {label}: {path}")
    obj = _read_json_if_exists(path, default=None)
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell11] Required {label} must be a JSON dict. Got {type(obj).__name__}: {path}")
    return obj


def _realpath(path: str) -> str:
    return os.path.realpath(os.path.abspath(str(path)))


def _sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_jsonable(obj) -> str:
    payload = json.dumps(_json_sanitize(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tier_of(col: str):
    col = str(col)
    if col.startswith("router__"):
        return "router"
    if col.startswith("ota__"):
        return "ota"
    if col.startswith("zigbee__"):
        return "zigbee"
    if col.startswith("zwave__"):
        return "zwave"
    return None


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


def _is_countlike(col: str) -> bool:
    return bool(COUNTLIKE_RX.search(str(col)))


def _is_unique_like(col: str) -> bool:
    return bool(UNIQUE_RX.search(str(col)))


def _finite_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def _bounded_finite_1d(x, max_n: int = 100_000) -> np.ndarray:
    x = _finite_1d(x)
    if x.size <= max_n:
        return x
    idx = np.linspace(0, x.size - 1, num=max_n)
    idx = np.unique(np.round(idx).astype(np.int64))
    return x[idx]


def _safe_mean(x):
    x = _bounded_finite_1d(x)
    return np.nan if x.size == 0 else float(np.mean(x))


def _safe_std(x):
    x = _bounded_finite_1d(x)
    return np.nan if x.size <= 1 else float(np.std(x))


def _safe_quantile(x, q: float):
    x = _bounded_finite_1d(x)
    return np.nan if x.size == 0 else float(np.quantile(x, q))


def _nonzero_rate(x):
    x = _bounded_finite_1d(x)
    return np.nan if x.size == 0 else float(np.mean(x > 0))


def _transition_rate(x):
    x = _bounded_finite_1d(x)
    if x.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x)) > 1e-9))


def _burst_rate(x):
    x = _bounded_finite_1d(x)
    if x.size < 2:
        return np.nan
    nz = x > 0
    return float(np.mean((~nz[:-1]) & (nz[1:])))


def _lag1_autocorr(x):
    x = _bounded_finite_1d(x)
    if x.size < 3:
        return np.nan
    a = x[:-1]
    b = x[1:]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _ks_stat_fast(a, b):
    a = np.sort(_bounded_finite_1d(a))
    b = np.sort(_bounded_finite_1d(b))
    if a.size == 0 or b.size == 0:
        return np.nan

    max_grid = int(CFG.get("cell11_ks_grid", 4096))
    max_grid = max(256, max_grid)

    if a.size + b.size <= max_grid:
        vals = np.sort(np.unique(np.concatenate([a, b])))
    else:
        qs = np.linspace(0.0, 1.0, max_grid)
        vals = np.sort(np.unique(np.concatenate([np.quantile(a, qs), np.quantile(b, qs)])))

    if vals.size == 0:
        return np.nan

    ca = np.searchsorted(a, vals, side="right") / float(a.size)
    cb = np.searchsorted(b, vals, side="right") / float(b.size)
    return float(np.max(np.abs(ca - cb)))


def _wasserstein_1d_fast(a, b):
    a = _bounded_finite_1d(a)
    b = _bounded_finite_1d(b)
    if a.size == 0 or b.size == 0:
        return np.nan

    max_points = int(CFG.get("cell11_wasserstein_points", 4096))
    max_points = max(256, max_points)
    m = int(min(max_points, max(256, min(a.size, b.size))))
    qs = np.linspace(0.0, 1.0, m)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _postprocess_protocol_values(col: str, x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).copy()
    x[~np.isfinite(x)] = np.nan
    finite = np.isfinite(x)

    if finite.any():
        x[finite] = np.maximum(x[finite], 0.0)
        if _is_countlike(col):
            x[finite] = np.rint(x[finite])

    return x.astype(np.float32, copy=False)


def _metric_row(real_vals, syn_vals) -> dict:
    real = _finite_1d(real_vals)
    syn = _finite_1d(syn_vals)

    if real.size == 0 or syn.size == 0:
        return {
            "real_n": int(real.size),
            "syn_n": int(syn.size),
            "ks": np.nan,
            "wasserstein": np.nan,
            "mean_abs_error": np.nan,
            "std_ratio": np.nan,
            "nonzero_rate_abs_error": np.nan,
            "q50_abs_error": np.nan,
            "q95_abs_error": np.nan,
            "q99_abs_error": np.nan,
            "transition_rate_abs_error": np.nan,
            "burst_rate_abs_error": np.nan,
            "lag1_abs_error": np.nan,
        }

    real_mean = _safe_mean(real)
    syn_mean = _safe_mean(syn)
    real_std = _safe_std(real)
    syn_std = _safe_std(syn)
    real_nz = _nonzero_rate(real)
    syn_nz = _nonzero_rate(syn)
    real_q50 = _safe_quantile(real, 0.50)
    syn_q50 = _safe_quantile(syn, 0.50)
    real_q95 = _safe_quantile(real, 0.95)
    syn_q95 = _safe_quantile(syn, 0.95)
    real_q99 = _safe_quantile(real, 0.99)
    syn_q99 = _safe_quantile(syn, 0.99)
    real_tr = _transition_rate(real)
    syn_tr = _transition_rate(syn)
    real_br = _burst_rate(real)
    syn_br = _burst_rate(syn)
    real_ac = _lag1_autocorr(real)
    syn_ac = _lag1_autocorr(syn)

    return {
        "real_n": int(real.size),
        "syn_n": int(syn.size),

        "real_mean": real_mean,
        "syn_mean": syn_mean,
        "mean_abs_error": (
            float(abs(real_mean - syn_mean))
            if np.isfinite(real_mean) and np.isfinite(syn_mean)
            else np.nan
        ),

        "real_std": real_std,
        "syn_std": syn_std,
        "std_ratio": (
            float((syn_std + 1e-9) / (real_std + 1e-9))
            if np.isfinite(real_std) and np.isfinite(syn_std)
            else np.nan
        ),

        "real_nonzero_rate": real_nz,
        "syn_nonzero_rate": syn_nz,
        "nonzero_rate_abs_error": (
            float(abs(real_nz - syn_nz))
            if np.isfinite(real_nz) and np.isfinite(syn_nz)
            else np.nan
        ),

        "real_q50": real_q50,
        "syn_q50": syn_q50,
        "q50_abs_error": (
            float(abs(real_q50 - syn_q50))
            if np.isfinite(real_q50) and np.isfinite(syn_q50)
            else np.nan
        ),

        "real_q95": real_q95,
        "syn_q95": syn_q95,
        "q95_abs_error": (
            float(abs(real_q95 - syn_q95))
            if np.isfinite(real_q95) and np.isfinite(syn_q95)
            else np.nan
        ),

        "real_q99": real_q99,
        "syn_q99": syn_q99,
        "q99_abs_error": (
            float(abs(real_q99 - syn_q99))
            if np.isfinite(real_q99) and np.isfinite(syn_q99)
            else np.nan
        ),

        "real_transition_rate": real_tr,
        "syn_transition_rate": syn_tr,
        "transition_rate_abs_error": (
            float(abs(real_tr - syn_tr))
            if np.isfinite(real_tr) and np.isfinite(syn_tr)
            else np.nan
        ),

        "real_burst_rate": real_br,
        "syn_burst_rate": syn_br,
        "burst_rate_abs_error": (
            float(abs(real_br - syn_br))
            if np.isfinite(real_br) and np.isfinite(syn_br)
            else np.nan
        ),

        "real_lag1": real_ac,
        "syn_lag1": syn_ac,
        "lag1_abs_error": (
            float(abs(real_ac - syn_ac))
            if np.isfinite(real_ac) and np.isfinite(syn_ac)
            else np.nan
        ),

        "ks": float(_ks_stat_fast(real, syn)),
        "wasserstein": float(_wasserstein_1d_fast(real, syn)),
    }


def _variant_score(metrics: dict, real_vals) -> float:
    real = _finite_1d(real_vals)
    if real.size == 0:
        return float("inf")

    q25 = _safe_quantile(real, 0.25)
    q75 = _safe_quantile(real, 0.75)
    q95 = _safe_quantile(real, 0.95)
    iqr = float(q75 - q25) if np.isfinite(q25) and np.isfinite(q75) else 0.0
    scale = max(abs(q95) if np.isfinite(q95) else 0.0, abs(iqr), 1.0)

    vals = [
        float(metrics.get("ks", np.nan)),
        float(metrics.get("wasserstein", np.nan)) / scale,
        float(metrics.get("mean_abs_error", np.nan)) / scale,
        float(metrics.get("nonzero_rate_abs_error", np.nan)),
        float(metrics.get("q50_abs_error", np.nan)) / scale,
        float(metrics.get("q95_abs_error", np.nan)) / scale,
        float(metrics.get("transition_rate_abs_error", np.nan)),
        float(metrics.get("burst_rate_abs_error", np.nan)),
        float(metrics.get("lag1_abs_error", 0.0)) if np.isfinite(metrics.get("lag1_abs_error", np.nan)) else 0.0,
    ]

    if not all(np.isfinite(v) for v in vals):
        return float("inf")

    return float(
        1.00 * vals[0]
        + 0.50 * vals[1]
        + 0.10 * vals[2]
        + 1.50 * vals[3]
        + 0.25 * vals[4]
        + 0.25 * vals[5]
        + 0.50 * vals[6]
        + 0.50 * vals[7]
        + 0.25 * vals[8]
    )


# ---------------------------------------------------------------------
# 2) Resolve Cell 10.9 materialized variant paths
# ---------------------------------------------------------------------
cell10_9_manifest = _read_json_dict_required(CELL10_9_MANIFEST_PATH, "Cell 10.9 manifest")
cell10_8_selector_contract = _read_json_dict_required(CELL10_8_SELECTOR_CONTRACT_PATH, "Cell 10.8 v3.6 selector contract")
cell10_9_materializer_contract = _read_json_dict_required(CELL10_9_MATERIALIZER_CONTRACT_PATH, "Cell 10.9 v2.1 materializer contract")
precell11_contract = _read_json_dict_required(PRECELL11_CONTRACT_PATH, "PRE-CELL11 v4.1 runtime contract")

# Enforce upstream purity contracts before final TEST QA.
for _key in [
    "test_values_read",
    "test_used_for_fitting",
    "test_used_for_thresholding",
    "test_used_for_selection",
    "test_rescue_materialization",
    "in_selector_candidate_generation",
    "a0_used_for_selection",
    "unregistered_file_scan",
]:
    if bool(cell10_8_selector_contract.get(_key, False)):
        raise RuntimeError(f"[Cell11] Cell 10.8 selector contract violation: {_key}=True")

for _key in [
    "test_values_read",
    "test_metrics_computed",
    "test_used_for_fitting",
    "test_used_for_thresholding",
    "test_used_for_selection",
    "test_used_for_repair",
    "downstream_test_rescue_materialization",
    "a0_used_for_selection",
]:
    if bool(cell10_9_materializer_contract.get(_key, False)):
        raise RuntimeError(f"[Cell11] Cell 10.9 materializer contract violation: {_key}=True")

for _key in [
    "cell11_must_not_select_generators",
    "cell11_must_not_register_ddpm_for_protocol_generation",
    "cell11_must_not_materialize_protocol_candidates",
    "cell11_must_not_apply_test_informed_rescue_or_repair",
    "cell11_must_not_read_test_for_selection",
]:
    if not bool(precell11_contract.get(_key, False)):
        raise RuntimeError(f"[Cell11] PRE-CELL11 runtime contract missing required guard: {_key}=True")

outputs = cell10_9_manifest.get("variant_outputs", None)
if not isinstance(outputs, dict):
    raise RuntimeError(
        "[Cell11] Cell 10.9 v2.1 manifest must expose 'variant_outputs'. "
        "Do not fall back to legacy 'outputs' for the canonical run."
    )

A0_PROTOCOL_PATH = outputs.get("A0_PROTOCOL_TEST")
A1_PROTOCOL_PATH = outputs.get("A1_PROTOCOL_TEST")
A2_PROTOCOL_PATH = outputs.get("A2_PROTOCOL_TEST")

for _label, _path in {
    "A0_PROTOCOL_TEST": A0_PROTOCOL_PATH,
    "A1_PROTOCOL_TEST": A1_PROTOCOL_PATH,
    "A2_PROTOCOL_TEST": A2_PROTOCOL_PATH,
}.items():
    if not _path:
        raise RuntimeError(f"[Cell11] Cell 10.9 manifest missing variant_outputs[{_label!r}].")
    expected = os.path.join(OUT_SYN, f"{_label}.parquet")
    if _realpath(_path) != _realpath(expected):
        raise RuntimeError(
            f"[Cell11] Cell 10.9 variant output path mismatch for {_label}: "
            f"manifest={_path} expected={expected}"
        )

for p, label in [
    (A0_PROTOCOL_PATH, "A0 protocol variant"),
    (A1_PROTOCOL_PATH, "A1 protocol variant"),
    (A2_PROTOCOL_PATH, "A2 protocol variant"),
]:
    if not os.path.exists(str(p)):
        raise RuntimeError(f"[Cell11] Missing {label}: {p}. Run Cell 10.9 first.")

A0_PROTOCOL = pd.read_parquet(A0_PROTOCOL_PATH)
A1_PROTOCOL = pd.read_parquet(A1_PROTOCOL_PATH)
A2_PROTOCOL = pd.read_parquet(A2_PROTOCOL_PATH)

log(
    "[Cell11] Loaded Cell 10.9 protocol variants | "
    f"A0={A0_PROTOCOL.shape} | A1={A1_PROTOCOL.shape} | A2={A2_PROTOCOL.shape}"
)


# ---------------------------------------------------------------------
# 3) Protocol schema checks
# ---------------------------------------------------------------------
PROTO_COLS = [
    str(c)
    for c in list(PROTO_VALUE_COLS)
    if str(c) in df_te.columns
]

if not PROTO_COLS:
    raise RuntimeError("[Cell11] No protocol columns from PROTO_VALUE_COLS are present in df_te.")

missing_variant_cols = {
    "A0": sorted(set(PROTO_COLS) - set(A0_PROTOCOL.columns)),
    "A1": sorted(set(PROTO_COLS) - set(A1_PROTOCOL.columns)),
    "A2": sorted(set(PROTO_COLS) - set(A2_PROTOCOL.columns)),
}

bad_missing = {k: v for k, v in missing_variant_cols.items() if v}
if bad_missing:
    raise RuntimeError(f"[Cell11] Protocol variants missing required columns: {bad_missing}")

if len(A0_PROTOCOL) != N_TEST:
    raise RuntimeError(f"[Cell11] A0 row mismatch: {len(A0_PROTOCOL)} vs df_te={N_TEST}")
if len(A1_PROTOCOL) != N_TEST:
    raise RuntimeError(f"[Cell11] A1 row mismatch: {len(A1_PROTOCOL)} vs df_te={N_TEST}")
if len(A2_PROTOCOL) != N_TEST:
    raise RuntimeError(f"[Cell11] A2 row mismatch: {len(A2_PROTOCOL)} vs df_te={N_TEST}")

A0_PROTOCOL = A0_PROTOCOL.loc[:, PROTO_COLS].copy()
A1_PROTOCOL = A1_PROTOCOL.loc[:, PROTO_COLS].copy()
A2_PROTOCOL = A2_PROTOCOL.loc[:, PROTO_COLS].copy()

REAL_TEST_PROTOCOL = df_te.loc[:, PROTO_COLS].copy()

for c in PROTO_COLS:
    A0_PROTOCOL[c] = _postprocess_protocol_values(c, pd.to_numeric(A0_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float32))
    A1_PROTOCOL[c] = _postprocess_protocol_values(c, pd.to_numeric(A1_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float32))
    A2_PROTOCOL[c] = _postprocess_protocol_values(c, pd.to_numeric(A2_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float32))
    REAL_TEST_PROTOCOL[c] = _postprocess_protocol_values(c, pd.to_numeric(REAL_TEST_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float32))

log(f"[Cell11] Protocol schema locked | proto_cols={len(PROTO_COLS)}")


# ---------------------------------------------------------------------
# 4) Resolve real and synthetic TEST masks
# ---------------------------------------------------------------------
def _matrix_to_mask_dict(M, N: int, label: str):
    M = np.asarray(M)
    if M.ndim != 2:
        raise RuntimeError(f"[Cell11] {label} must be 2D; got shape={M.shape}")
    if M.shape[0] != N:
        raise RuntimeError(f"[Cell11] {label} row mismatch: {M.shape[0]} vs {N}")

    tiers = ["router", "ota", "zigbee"]
    if M.shape[1] >= 4:
        tiers.append("zwave")

    if M.shape[1] < 3:
        raise RuntimeError(f"[Cell11] {label} must contain at least router/ota/zigbee columns.")

    out = {}
    for j, t in enumerate(tiers[:M.shape[1]]):
        out[t] = (M[:, j].astype(np.int8, copy=False) > 0)

    if "zwave" not in out:
        out["zwave"] = np.zeros(N, dtype=bool)

    return out


def _mask_dict_from_df_obs(df_part: pd.DataFrame, N: int, label: str):
    out = {}
    for tier, c in TIER_MASK_COL.items():
        if c in df_part.columns:
            v = pd.to_numeric(df_part[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if v.shape[0] != N:
                raise RuntimeError(f"[Cell11] {label}.{c} length mismatch.")
            out[tier] = (v > 0.5)
        elif tier == "zwave":
            out[tier] = np.zeros(N, dtype=bool)
        else:
            raise RuntimeError(f"[Cell11] Missing required {label} mask column: {c}")
    return out


def _normalize_mask_dict(mask_dict: dict, N: int, label: str):
    if not isinstance(mask_dict, dict):
        raise RuntimeError(f"[Cell11] {label} mask object must be dict.")

    out = {}
    for tier in ["router", "ota", "zigbee", "zwave"]:
        if tier in mask_dict:
            m = np.asarray(mask_dict[tier], dtype=bool).reshape(-1)
            if m.shape[0] != N:
                raise RuntimeError(f"[Cell11] {label} mask length mismatch for {tier}: {m.shape[0]} vs {N}")
            out[tier] = m
        elif tier == "zwave":
            out[tier] = np.zeros(N, dtype=bool)
        else:
            raise RuntimeError(f"[Cell11] {label} missing required mask tier: {tier}")
    return out


if "CELL10_A0_TEST_MASKS" in globals():
    REAL_TEST_MASKS = _normalize_mask_dict(globals()["CELL10_A0_TEST_MASKS"], N_TEST, "CELL10_A0_TEST_MASKS")
elif "REAL_TEST_MASKS_FULL" in globals():
    REAL_TEST_MASKS = _matrix_to_mask_dict(globals()["REAL_TEST_MASKS_FULL"], N_TEST, "REAL_TEST_MASKS_FULL")
else:
    REAL_TEST_MASKS = _mask_dict_from_df_obs(df_te, N_TEST, "df_te")

if "CELL10_A1_TEST_MASKS" in globals():
    SYN_TEST_MASKS = _normalize_mask_dict(globals()["CELL10_A1_TEST_MASKS"], N_TEST, "CELL10_A1_TEST_MASKS")
elif "SYN_MASKS_TEST_FULL" in globals():
    SYN_TEST_MASKS = _matrix_to_mask_dict(globals()["SYN_MASKS_TEST_FULL"], N_TEST, "SYN_MASKS_TEST_FULL")
else:
    raise RuntimeError(
        "[Cell11] Missing synthetic TEST masks. Run Cell 9 and expose CELL10_A1_TEST_MASKS "
        "or SYN_MASKS_TEST_FULL. The ambiguous SYN_MASKS_FULL fallback is intentionally disabled."
    )

real_mask_rates = {t: float(np.mean(m)) for t, m in REAL_TEST_MASKS.items()}
syn_mask_rates = {t: float(np.mean(m)) for t, m in SYN_TEST_MASKS.items()}
mask_rate_drift = {
    t: float(syn_mask_rates[t] - real_mask_rates[t])
    for t in ["router", "ota", "zigbee", "zwave"]
}

log(f"[Cell11] Real TEST mask rates: {real_mask_rates}")
log(f"[Cell11] Synthetic TEST mask rates: {syn_mask_rates}")
log(f"[Cell11] Synthetic-minus-real TEST mask drift: {mask_rate_drift}")


# ---------------------------------------------------------------------
# 5) Variant mask-contract validation
# ---------------------------------------------------------------------
def _validate_variant_mask_contract(df_variant: pd.DataFrame, masks: dict, variant_name: str):
    rows = []

    for c in PROTO_COLS:
        tier = _tier_of(c)
        if tier is None:
            continue

        active = np.asarray(masks[tier], dtype=bool)
        x = pd.to_numeric(df_variant[c], errors="coerce").to_numpy(dtype=np.float64)

        if x.shape[0] != N_TEST:
            raise RuntimeError(f"[Cell11] {variant_name}.{c} length mismatch.")

        inactive_finite_n = int(np.isfinite(x[~active]).sum())
        active_finite_n = int(np.isfinite(x[active]).sum())
        active_n = int(active.sum())

        negative_active_n = 0
        noninteger_active_frac = np.nan

        if active_finite_n > 0:
            xa = x[active]
            xa = xa[np.isfinite(xa)]
            negative_active_n = int(np.sum(xa < -1e-6))

            if _is_countlike(c):
                noninteger_active_frac = float(np.mean(np.abs(xa - np.rint(xa)) > 1e-5))

        row = {
            "variant": variant_name,
            "col": c,
            "tier": tier,
            "active_n": active_n,
            "active_rate": float(np.mean(active)),
            "active_finite_n": active_finite_n,
            "inactive_finite_n": inactive_finite_n,
            "negative_active_n": negative_active_n,
            "countlike": bool(_is_countlike(c)),
            "noninteger_active_frac": noninteger_active_frac,
            "inactive_nan_contract_ok": bool(inactive_finite_n == 0),
            "active_all_finite_ok": bool(active_finite_n == active_n),
            "nonnegative_active_ok": bool(negative_active_n == 0),
            "count_integer_ok": bool(
                (not _is_countlike(c))
                or (not np.isfinite(noninteger_active_frac))
                or noninteger_active_frac <= float(CFG.get("cell11_countlike_noninteger_frac_max", 0.001))
            ),
        }

        row["valid"] = bool(
            row["inactive_nan_contract_ok"]
            and row["active_all_finite_ok"]
            and row["nonnegative_active_ok"]
            and row["count_integer_ok"]
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    bad = out[out["valid"] == False].copy()
    if len(bad) > 0:
        preview = bad[
            [
                "variant",
                "col",
                "tier",
                "active_n",
                "active_finite_n",
                "inactive_finite_n",
                "negative_active_n",
                "noninteger_active_frac",
            ]
        ].head(20).to_dict("records")
        raise RuntimeError(f"[Cell11] Mask/value contract failed for {variant_name}. Preview={preview}")

    return out


mask_contract_audit = pd.concat(
    [
        _validate_variant_mask_contract(A0_PROTOCOL, REAL_TEST_MASKS, "A0_REAL_MASK_BASELINE"),
        _validate_variant_mask_contract(A1_PROTOCOL, SYN_TEST_MASKS, "A1_SYN_MASK_BASELINE"),
        _validate_variant_mask_contract(A2_PROTOCOL, SYN_TEST_MASKS, "A2_VAL_SELECTED_PORTFOLIO"),
    ],
    axis=0,
    ignore_index=True,
)

mask_contract_audit.to_csv(MASK_CONTRACT_PATH, index=False)
log(f"[Cell11] Mask/value contract passed for A0/A1/A2 | audit={MASK_CONTRACT_PATH}")


# ---------------------------------------------------------------------
# 6) Selection consistency and A1/A2 delta audit
# ---------------------------------------------------------------------
selected_by_col = _read_json_if_exists(CELL10_SELECTION_JSON, default={})
selection_rich = _read_json_if_exists(CELL10_SELECTION_RICH_JSON, default={})
selection_summary = _read_json_if_exists(CELL10_SELECTION_SUMMARY_JSON, default={})
precell11_contract = _read_json_if_exists(PRECELL11_CONTRACT_PATH, default={})

if not isinstance(selected_by_col, dict):
    raise RuntimeError(f"[Cell11] Selection registry must be dict: {CELL10_SELECTION_JSON}")

missing_sel = sorted(set(PROTO_COLS) - set(selected_by_col.keys()))
extra_sel = sorted(set(selected_by_col.keys()) - set(PROTO_COLS))
if missing_sel or extra_sel:
    raise RuntimeError(
        "[Cell11] Selected-generator registry is not a complete protocol-column partition. "
        f"missing={missing_sel[:20]} extra={extra_sel[:20]}"
    )

delta_rows = []
selection_rows = []

for c in PROTO_COLS:
    tier = _tier_of(c)
    active_syn = SYN_TEST_MASKS[tier]

    x1 = pd.to_numeric(A1_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float64)
    x2 = pd.to_numeric(A2_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float64)

    active_both = active_syn & np.isfinite(x1) & np.isfinite(x2)
    inactive_any_finite = int((np.isfinite(x1[~active_syn]) | np.isfinite(x2[~active_syn])).sum())

    if active_both.any():
        abs_delta = np.abs(x2[active_both] - x1[active_both])
        changed_active_n = int(np.sum(abs_delta > 1e-8))
        max_abs_delta = float(np.max(abs_delta)) if abs_delta.size else 0.0
        mean_abs_delta = float(np.mean(abs_delta)) if abs_delta.size else 0.0
    else:
        changed_active_n = 0
        max_abs_delta = 0.0
        mean_abs_delta = 0.0

    changed_active_rate = float(changed_active_n / max(int(active_both.sum()), 1))

    sel = selected_by_col.get(c, {})
    selected_generator = str(sel.get("selected_generator", "UNKNOWN"))
    selected_candidate_uid = str(sel.get("selected_candidate_uid", "UNKNOWN"))
    selected_is_a1 = selected_generator == "A1_temporal_block_bootstrap"

    expected_equal_a1 = bool(selected_is_a1)
    observed_equal_a1 = bool(changed_active_n == 0)

    consistency_ok = bool(
        inactive_any_finite == 0
        and (
            (expected_equal_a1 and observed_equal_a1)
            or (not expected_equal_a1)
        )
    )

    delta_rows.append({
        "col": c,
        "tier": tier,
        "selected_generator": selected_generator,
        "selected_candidate_uid": selected_candidate_uid,
        "expected_equal_a1": expected_equal_a1,
        "observed_equal_a1": observed_equal_a1,
        "active_syn_n": int(active_syn.sum()),
        "active_comparable_n": int(active_both.sum()),
        "changed_active_n": changed_active_n,
        "changed_active_rate": changed_active_rate,
        "max_abs_delta": max_abs_delta,
        "mean_abs_delta": mean_abs_delta,
        "inactive_any_finite_n": inactive_any_finite,
        "consistency_ok": consistency_ok,
    })

    selection_rows.append({
        "col": c,
        "tier": tier,
        "selected_generator": selected_generator,
        "selected_candidate_id": sel.get("selected_candidate_id", None),
        "selected_candidate_uid": selected_candidate_uid,
        "reason": sel.get("reason", None),
        "requires_downstream_test_materialization": bool(sel.get("requires_downstream_test_materialization", False)),
        "test_path": sel.get("test_path", None),
        "a2_equals_a1_for_col": observed_equal_a1,
        "selection_delta_consistent": consistency_ok,
    })

delta_audit = pd.DataFrame(delta_rows)
selection_consistency = pd.DataFrame(selection_rows)

bad_delta = delta_audit[delta_audit["consistency_ok"] == False].copy()
if len(bad_delta) > 0:
    preview = bad_delta.head(20).to_dict("records")
    raise RuntimeError(f"[Cell11] A1/A2 delta consistency failed. Preview={preview}")

delta_audit.to_csv(DELTA_AUDIT_PATH, index=False)
selection_consistency.to_csv(SELECTION_CONSISTENCY_PATH, index=False)

selected_counts = Counter(selection_consistency["selected_generator"].astype(str).tolist())
all_a1_selected = bool(len(selected_counts) == 1 and selected_counts.get("A1_temporal_block_bootstrap", 0) == len(PROTO_COLS))
a2_equals_a1 = bool(delta_audit["observed_equal_a1"].all())
a2_equals_a0 = bool(
    all(
        np.array_equal(
            np.nan_to_num(A2_PROTOCOL[c].to_numpy(dtype=np.float64), nan=-999999999.0),
            np.nan_to_num(A0_PROTOCOL[c].to_numpy(dtype=np.float64), nan=-999999999.0),
        )
        for c in PROTO_COLS
    )
)

log(
    "[Cell11] Selection/delta consistency passed | "
    f"selected_counts={dict(selected_counts)} | "
    f"all_a1_selected={all_a1_selected} | "
    f"A2_equals_A1={a2_equals_a1} | "
    f"A2_equals_A0={a2_equals_a0}"
)


# ---------------------------------------------------------------------
# 7) Final TEST QA quality report
# ---------------------------------------------------------------------
quality_rows = []

for c in PROTO_COLS:
    tier = _tier_of(c)
    if tier is None:
        continue

    real_mask = REAL_TEST_MASKS[tier]
    syn_mask = SYN_TEST_MASKS[tier]

    real_values = pd.to_numeric(REAL_TEST_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float64)
    a0_values = pd.to_numeric(A0_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float64)
    a1_values = pd.to_numeric(A1_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float64)
    a2_values = pd.to_numeric(A2_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float64)

    # A0 uses real TEST masks, so evaluate A0 against real TEST values on real-mask active rows.
    eval_a0 = real_mask & np.isfinite(real_values) & np.isfinite(a0_values)

    # A1/A2 use synthetic TEST masks. Use observed-only intersection for fair value comparison.
    eval_syn = real_mask & syn_mask & np.isfinite(real_values)

    eval_a1 = eval_syn & np.isfinite(a1_values)
    eval_a2 = eval_syn & np.isfinite(a2_values)

    m0 = _metric_row(real_values[eval_a0], a0_values[eval_a0])
    m1 = _metric_row(real_values[eval_a1], a1_values[eval_a1])
    m2 = _metric_row(real_values[eval_a2], a2_values[eval_a2])

    s0 = _variant_score(m0, real_values[eval_a0])
    s1 = _variant_score(m1, real_values[eval_a1])
    s2 = _variant_score(m2, real_values[eval_a2])

    row = {
        "col": c,
        "tier": tier,

        "real_test_active_rate": float(np.mean(real_mask)),
        "synthetic_test_active_rate": float(np.mean(syn_mask)),
        "synthetic_minus_real_active_rate": float(np.mean(syn_mask) - np.mean(real_mask)),

        "A0_eval_n": int(eval_a0.sum()),
        "A1_eval_n": int(eval_a1.sum()),
        "A2_eval_n": int(eval_a2.sum()),

        "selected_generator": selected_by_col[c].get("selected_generator", None),
        "selected_candidate_uid": selected_by_col[c].get("selected_candidate_uid", None),

        "A0_score": float(s0) if np.isfinite(s0) else np.nan,
        "A1_score": float(s1) if np.isfinite(s1) else np.nan,
        "A2_score": float(s2) if np.isfinite(s2) else np.nan,
    }

    for prefix, metrics in [("A0", m0), ("A1", m1), ("A2", m2)]:
        for k, v in metrics.items():
            row[f"{prefix}_{k}"] = v

    row["A1_minus_A0_score"] = (
        float(row["A1_score"] - row["A0_score"])
        if np.isfinite(row["A1_score"]) and np.isfinite(row["A0_score"])
        else np.nan
    )
    row["A2_minus_A1_score"] = (
        float(row["A2_score"] - row["A1_score"])
        if np.isfinite(row["A2_score"]) and np.isfinite(row["A1_score"])
        else np.nan
    )
    row["A2_minus_A0_score"] = (
        float(row["A2_score"] - row["A0_score"])
        if np.isfinite(row["A2_score"]) and np.isfinite(row["A0_score"])
        else np.nan
    )

    row["A2_minus_A1_ks"] = (
        float(row["A2_ks"] - row["A1_ks"])
        if np.isfinite(row.get("A2_ks", np.nan)) and np.isfinite(row.get("A1_ks", np.nan))
        else np.nan
    )
    row["A2_minus_A1_wasserstein"] = (
        float(row["A2_wasserstein"] - row["A1_wasserstein"])
        if np.isfinite(row.get("A2_wasserstein", np.nan)) and np.isfinite(row.get("A1_wasserstein", np.nan))
        else np.nan
    )

    row["A0_best_score"] = bool(
        np.isfinite(row["A0_score"])
        and np.isfinite(row["A1_score"])
        and np.isfinite(row["A2_score"])
        and row["A0_score"] <= min(row["A1_score"], row["A2_score"])
    )
    row["A1_best_score"] = bool(
        np.isfinite(row["A1_score"])
        and np.isfinite(row["A0_score"])
        and np.isfinite(row["A2_score"])
        and row["A1_score"] <= min(row["A0_score"], row["A2_score"])
    )
    row["A2_best_score"] = bool(
        np.isfinite(row["A2_score"])
        and np.isfinite(row["A0_score"])
        and np.isfinite(row["A1_score"])
        and row["A2_score"] <= min(row["A0_score"], row["A1_score"])
    )

    quality_rows.append(row)

quality_report = pd.DataFrame(quality_rows)
quality_report.to_csv(QUALITY_REPORT_PATH, index=False)

if quality_report.empty:
    raise RuntimeError("[Cell11] Quality report is empty.")

summary_rows = []

for variant in ["A0", "A1", "A2"]:
    score_col = f"{variant}_score"
    ks_col = f"{variant}_ks"
    wass_col = f"{variant}_wasserstein"
    nz_col = f"{variant}_nonzero_rate_abs_error"

    summary_rows.append({
        "variant": variant,
        "cols_n": int(len(quality_report)),
        "median_score": float(pd.to_numeric(quality_report[score_col], errors="coerce").median(skipna=True)),
        "mean_score": float(pd.to_numeric(quality_report[score_col], errors="coerce").mean(skipna=True)),
        "median_ks": float(pd.to_numeric(quality_report[ks_col], errors="coerce").median(skipna=True)),
        "mean_ks": float(pd.to_numeric(quality_report[ks_col], errors="coerce").mean(skipna=True)),
        "median_wasserstein": float(pd.to_numeric(quality_report[wass_col], errors="coerce").median(skipna=True)),
        "mean_wasserstein": float(pd.to_numeric(quality_report[wass_col], errors="coerce").mean(skipna=True)),
        "median_nonzero_rate_abs_error": float(pd.to_numeric(quality_report[nz_col], errors="coerce").median(skipna=True)),
        "mean_nonzero_rate_abs_error": float(pd.to_numeric(quality_report[nz_col], errors="coerce").mean(skipna=True)),
        "best_score_cols": int(quality_report[f"{variant}_best_score"].sum()),
    })

variant_summary = pd.DataFrame(summary_rows)
variant_summary.to_csv(SUMMARY_CSV_PATH, index=False)

log(f"[Cell11] Saved quality report: {QUALITY_REPORT_PATH}")
log(f"[Cell11] Saved quality summary: {SUMMARY_CSV_PATH}")


# ---------------------------------------------------------------------
# 8) Final fail-closed QA guards
# ---------------------------------------------------------------------
fail_on_contract = bool(CFG.get("cell11_fail_on_contract_violation", True))
fail_on_material_a2_regression = bool(CFG.get("cell11_fail_on_material_a2_regression", False))

material_score_regress = float(CFG.get("cell11_material_a2_score_regression", 0.05))
material_ks_regress = float(CFG.get("cell11_material_a2_ks_regression", 0.005))
material_wass_regress_frac = float(CFG.get("cell11_material_a2_wass_regression_frac", 0.05))
material_wass_abs_floor = float(CFG.get("cell11_material_a2_wass_abs_floor", 0.01))

qa = quality_report.copy()

qa["A2_score_regression_vs_A1"] = pd.to_numeric(qa["A2_minus_A1_score"], errors="coerce") > material_score_regress
qa["A2_ks_regression_vs_A1"] = pd.to_numeric(qa["A2_minus_A1_ks"], errors="coerce") > material_ks_regress

a1_wass = pd.to_numeric(qa["A1_wasserstein"], errors="coerce").fillna(0.0)
a2_minus_a1_wass = pd.to_numeric(qa["A2_minus_A1_wasserstein"], errors="coerce")
qa["A2_wasserstein_regression_threshold"] = np.maximum(
    material_wass_abs_floor,
    material_wass_regress_frac * np.maximum(a1_wass, 0.0),
)
qa["A2_wasserstein_regression_vs_A1"] = (
    a2_minus_a1_wass > qa["A2_wasserstein_regression_threshold"]
)

qa["A2_material_regression_vs_A1"] = (
    qa["A2_score_regression_vs_A1"]
    | qa["A2_ks_regression_vs_A1"]
    | qa["A2_wasserstein_regression_vs_A1"]
)

material_regression_rows = qa[qa["A2_material_regression_vs_A1"]].copy()

qa_summary = {
    "fail_on_contract": bool(fail_on_contract),
    "fail_on_material_a2_regression": bool(fail_on_material_a2_regression),
    "material_a2_regression_cols_n": int(len(material_regression_rows)),
    "material_score_regression_threshold": float(material_score_regress),
    "material_ks_regression_threshold": float(material_ks_regress),
    "material_wasserstein_regression_frac": float(material_wass_regress_frac),
    "material_wasserstein_abs_floor": float(material_wass_abs_floor),
    "all_a1_selected": bool(all_a1_selected),
    "a2_equals_a1": bool(a2_equals_a1),
    "a2_equals_a0": bool(a2_equals_a0),
}

if fail_on_material_a2_regression and len(material_regression_rows) > 0:
    preview = material_regression_rows[
        [
            "col",
            "tier",
            "selected_generator",
            "A1_score",
            "A2_score",
            "A2_minus_A1_score",
            "A1_ks",
            "A2_ks",
            "A2_minus_A1_ks",
            "A1_wasserstein",
            "A2_wasserstein",
            "A2_minus_A1_wasserstein",
        ]
    ].head(20).to_dict("records")

    raise RuntimeError(
        "[Cell11] Final TEST QA found material A2 regressions versus A1. "
        "This does not imply leakage, but it means the locked A2 output should be reviewed. "
        f"Preview={preview}"
    )

if len(material_regression_rows) > 0:
    log(
        "[Cell11][QA-WARN] Material A2 regressions versus A1 detected, but not fail-closed by CFG | "
        f"cols={len(material_regression_rows)}"
    )
else:
    log("[Cell11] Final A2-vs-A1 QA passed: no material A2 regressions.")


# ---------------------------------------------------------------------
# 8.5) Phase 1 pipeline-level quality-decomposition scaffold
# ---------------------------------------------------------------------
# Purpose:
# - Make Cell 11 the canonical controller for A0/A1/A2 quality semantics.
# - Force downstream QA to map to explicit Q1–Q6 quality dimensions.
# - Separate diagnostic artifacts from publication-safe artifacts.
# - Provide an attribution matrix showing which cells own which variant/dimension.
#
# This block does not generate, select, or fit anything.
# It only writes scaffold artifacts for later IoT/protocol QA integration.

QUALITY_DIMENSION_REGISTRY = {
    "version": "quality_dimension_registry_v1_phase1_study_thesis",
    "declared_in_cell": "Cell11",
    "scope": "whole_pipeline_protocol_plus_iot_cps_quality_decomposition",
    "test_usage_policy": {
        "selection_or_fitting": "FORBIDDEN",
        "final_QA_reporting": "ALLOWED",
        "publication_claims_must_distinguish_VAL_selection_from_TEST_QA": True,
    },
    "dimensions": {
        "Q1_capture_observability_fidelity": {
            "short_name": "capture_observability",
            "question": (
                "Does the synthetic dataset reproduce capture/availability structure, "
                "including protocol masks, IoT observability masks, staleness, and missingness?"
            ),
            "primary_metrics": [
                "mask_active_rate_abs_error",
                "mask_transition_rate_abs_error",
                "mask_run_length_distribution_error",
                "entity_observability_rate_error",
                "staleness_distribution_error",
            ],
            "current_cell11_outputs": [
                "cell11_a0_a1_a2_mask_contract_audit.csv",
                "cell11_variant_quality_summary.csv",
            ],
            "downstream_required_outputs": [
                "iot_value_mask_audit",
                "iot_binary_mask_audit",
                "iot_driver_mask_or_event_presence_audit",
                "cross_modal_mask_alignment_audit",
            ],
            "publication_role": "required_for_release",
        },
        "Q2_value_distribution_fidelity": {
            "short_name": "value_distribution",
            "question": (
                "Do synthetic values match real distributions under the correct observed-only "
                "evaluation intersections?"
            ),
            "primary_metrics": [
                "KS",
                "Wasserstein",
                "normalized_Wasserstein",
                "mean_abs_error",
                "quantile_abs_error",
                "support_jaccard",
                "C2ST_AUC",
            ],
            "current_cell11_outputs": [
                "cell11_a0_a1_a2_protocol_quality_report.csv",
                "cell11_variant_quality_summary.csv",
            ],
            "downstream_required_outputs": [
                "cell12c6_final_test_qa_metrics.csv",
                "binary_value_distribution_audit",
                "driver_event_count_distribution_audit",
            ],
            "publication_role": "required_for_release",
        },
        "Q3_temporal_dynamics_fidelity": {
            "short_name": "temporal_dynamics",
            "question": (
                "Do synthetic sequences preserve temporal behavior: transitions, bursts, "
                "dwell times, autocorrelation, event spacing, and run lengths?"
            ),
            "primary_metrics": [
                "transition_rate_abs_error",
                "burst_rate_abs_error",
                "lag1_autocorr_abs_error",
                "run_length_distribution_error",
                "inter_event_time_distribution_error",
                "dwell_time_distribution_error",
            ],
            "current_cell11_outputs": [
                "cell11_a0_a1_a2_protocol_quality_report.csv",
            ],
            "downstream_required_outputs": [
                "iot_temporal_audit",
                "binary_transition_audit",
                "driver_inter_event_time_audit",
                "cross_modal_lag_audit",
            ],
            "publication_role": "required_for_release",
        },
        "Q4_cross_modal_coupling_fidelity": {
            "short_name": "cross_modal_coupling",
            "question": (
                "Does the synthetic CPS preserve IoT↔protocol coupling, lagged responses, "
                "event-driven protocol bursts, and regime/TOD-conditioned dependencies?"
            ),
            "primary_metrics": [
                "lagged_cross_correlation_error",
                "event_to_protocol_response_delay_error",
                "protocol_burst_to_iot_event_cooccurrence_error",
                "regime_conditioned_distribution_error",
                "TOD_conditioned_distribution_error",
            ],
            "current_cell11_outputs": [
                "cell11_iot_downstream_protocol_context.json",
                "cell11_iot_downstream_protocol_activity_summary.csv",
            ],
            "downstream_required_outputs": [
                "iot_protocol_coupling_audit",
                "driver_to_protocol_lag_audit",
                "binary_state_to_protocol_activity_audit",
                "value_event_to_protocol_response_audit",
            ],
            "publication_role": "required_for_release",
        },
        "Q5_schema_domain_contract_fidelity": {
            "short_name": "schema_domain_contract",
            "question": (
                "Does every synthetic artifact satisfy schema, mask, domain, support, "
                "integer/count, nonnegative, and role-ownership contracts?"
            ),
            "primary_metrics": [
                "schema_match",
                "inactive_finite_violations",
                "active_nonfinite_violations",
                "domain_violation_count",
                "noninteger_countlike_fraction",
                "role_owner_partition_coverage",
            ],
            "current_cell11_outputs": [
                "cell11_a0_a1_a2_mask_contract_audit.csv",
                "cell11_final_protocol_variant_audit.json",
            ],
            "downstream_required_outputs": [
                "iot_partition_contract.json",
                "iot_role_ownership.csv",
                "cell12c5_final_value_enforcement_audit.csv",
                "binary_domain_contract_audit",
                "driver_domain_contract_audit",
            ],
            "publication_role": "hard_gate",
        },
        "Q6_selection_utility_publication_readiness": {
            "short_name": "selection_utility_publication",
            "question": (
                "Are generator choices VAL-locked, attributable, publication-safe, and "
                "useful compared with baselines without TEST leakage?"
            ),
            "primary_metrics": [
                "selected_generator_counts",
                "VAL_selection_reason",
                "A2_minus_A1_score",
                "material_regression_count",
                "publication_blocker_count",
                "utility_TSTR_or_downstream_task_delta",
            ],
            "current_cell11_outputs": [
                "cell11_selection_consistency_audit.csv",
                "cell11_a0_a1_a2_delta_audit.csv",
                "cell11_final_protocol_variant_manifest.json",
            ],
            "downstream_required_outputs": [
                "cell12c3_locked_iot_value_selection.csv",
                "cell12c6_final_selected_generator_table.csv",
                "iot_binary_locked_selection_audit",
                "iot_driver_locked_selection_audit",
                "whole_cps_publication_readiness_report",
            ],
            "publication_role": "hard_gate",
        },
    },
    "mandatory_mapping_policy": {
        "every_downstream_QA_report_must_include_quality_dimension": True,
        "allowed_quality_dimension_ids": [
            "Q1_capture_observability_fidelity",
            "Q2_value_distribution_fidelity",
            "Q3_temporal_dynamics_fidelity",
            "Q4_cross_modal_coupling_fidelity",
            "Q5_schema_domain_contract_fidelity",
            "Q6_selection_utility_publication_readiness",
        ],
        "unmapped_QA_metrics_are_publication_blockers": True,
    },
}

# Variant publication semantics.
# Important: A0 is intentionally diagnostic. A1 is the attribution baseline.
# A2 is the final candidate only when it is not merely degenerate to A1 for
# the relevant release scope.
A0_A1_A2_VARIANT_MANIFEST = {
    "version": "a0_a1_a2_variant_manifest_v1_phase1_study_thesis",
    "declared_in_cell": "Cell11",
    "N_TEST": int(N_TEST),
    "protocol_value_cols_n": int(len(PROTO_COLS)),
    "variants": {
        "A0": {
            "name": "A0_real_mask_train_only_value_probe",
            "meaning": "TRAIN-only block/bootstrap value realism probe under real TEST masks.",
            "mask_source": "real_TEST_masks",
            "value_source": "TRAIN_only_baseline",
            "publication_role": "diagnostic_only",
            "publication_safe_as_final_release": False,
            "allowed_claims": [
                "diagnostic upper/lower-bound probe for value realism under real capture support",
                "not a releasable synthetic CPS dataset because it uses real TEST masks",
            ],
            "forbidden_claims": [
                "final synthetic CPS output",
                "publication release candidate",
            ],
        },
        "A1": {
            "name": "A1_synthetic_mask_safe_value_baseline",
            "meaning": "Synthetic/captured masks with safe TRAIN-only value baseline.",
            "mask_source": "synthetic_TEST_masks_from_Cell9",
            "value_source": "TRAIN_only_baseline",
            "publication_role": "attribution_baseline",
            "publication_safe_as_final_release": False,
            "allowed_claims": [
                "baseline for attributing value-generator improvement",
                "capture-aware baseline under synthetic masks",
            ],
            "forbidden_claims": [
                "full proposed coupled CPS generator unless A2 is identical by justified VAL selection and documented as degenerate",
            ],
        },
        "A2": {
            "name": "A2_val_selected_coupled_synthetic_cps_candidate",
            "meaning": "VAL-selected predefined generator-portfolio output under synthetic TEST masks.",
            "mask_source": "synthetic_TEST_masks_from_Cell9",
            "value_source": "VAL_selected_generator_portfolio",
            "publication_role": "final_candidate",
            "publication_safe_as_final_release": False,
            "protocol_value_status": (
                "degenerate_to_A1_for_protocol_values"
                if bool(a2_equals_a1)
                else "distinct_from_A1_for_protocol_values"
            ),
            "allowed_claims": [
                "protocol-side final candidate only; whole-artifact release remains gated by IoT, binary, driver, observability, coupling, and Q6 audits",
                "VAL-selected portfolio output; TEST used only for final QA",
            ],
            "forbidden_claims": [
                "TEST-selected generator",
                "full coupled CPS realism if downstream IoT/coupling QA remains failing",
            ],
        },
    },
    "observed_protocol_selection_state": {
        "selected_counts": dict(selected_counts),
        "all_a1_selected": bool(all_a1_selected),
        "a2_equals_a1": bool(a2_equals_a1),
        "a2_equals_a0": bool(a2_equals_a0),
        "material_A2_regressions": int(qa_summary["material_a2_regression_cols_n"]),
    },
    "release_gate_interpretation": {
        "A0_must_not_be_used_as_final_release": True,
        "A1_is_baseline_not_final_release": True,
        "A2_requires_non_degenerate_or_justified_degenerate_status": True,
        "current_protocol_A2_is_degenerate_to_A1": bool(a2_equals_a1),
        "whole_pipeline_release_requires_downstream_iot_binary_driver_coupling_QA": True,
    },
}

quality_attribution_rows = [
    {
        "pipeline_stage": "taxonomy",
        "cell": "Cell 4.5",
        "artifact_or_output": "iot_feature_groups.json; iot_role_ownership.csv; iot_partition_contract.json",
        "variant": "A0/A1/A2",
        "variant_role": "shared_contract",
        "quality_dimensions": "Q5_schema_domain_contract_fidelity",
        "owns_generation": False,
        "owns_selection": False,
        "owns_final_QA": False,
        "publication_note": "Defines exhaustive IoT role ownership; required before any IoT target generation.",
    },
    {
        "pipeline_stage": "contract_loader",
        "cell": "Cell 6.5",
        "artifact_or_output": "iot_group_loader_contract.json; iot_cps_coupling_policy.json",
        "variant": "A0/A1/A2",
        "variant_role": "shared_contract",
        "quality_dimensions": "Q4_cross_modal_coupling_fidelity|Q5_schema_domain_contract_fidelity",
        "owns_generation": False,
        "owns_selection": False,
        "owns_final_QA": False,
        "publication_note": "Forbids standalone IoT synthesis and binds IoT to CPS/protocol context.",
    },
    {
        "pipeline_stage": "protocol_mask_generation",
        "cell": "Cell 9",
        "artifact_or_output": "synthetic TEST protocol masks",
        "variant": "A1/A2",
        "variant_role": "synthetic_capture_context",
        "quality_dimensions": "Q1_capture_observability_fidelity|Q5_schema_domain_contract_fidelity",
        "owns_generation": True,
        "owns_selection": False,
        "owns_final_QA": False,
        "publication_note": "Owns synthetic protocol capture masks used by A1 and A2.",
    },
    {
        "pipeline_stage": "protocol_value_baseline",
        "cell": "Cell 10.1",
        "artifact_or_output": "A0/A1 TRAIN-only protocol value baselines",
        "variant": "A0/A1",
        "variant_role": "diagnostic_and_attribution_baseline",
        "quality_dimensions": "Q2_value_distribution_fidelity|Q3_temporal_dynamics_fidelity",
        "owns_generation": True,
        "owns_selection": False,
        "owns_final_QA": False,
        "publication_note": "A0 is diagnostic only; A1 is attribution baseline.",
    },
    {
        "pipeline_stage": "protocol_candidate_generation",
        "cell": "Cells 10.2–10.7",
        "artifact_or_output": "predefined protocol generator candidates",
        "variant": "A2",
        "variant_role": "candidate_pool",
        "quality_dimensions": "Q2_value_distribution_fidelity|Q3_temporal_dynamics_fidelity|Q6_selection_utility_publication_readiness",
        "owns_generation": True,
        "owns_selection": False,
        "owns_final_QA": False,
        "publication_note": "Predefined role-based candidate portfolio; not arbitrary model shopping.",
    },
    {
        "pipeline_stage": "protocol_val_selection",
        "cell": "Cell 10.8",
        "artifact_or_output": "cell10_selected_generator_by_col.json",
        "variant": "A2",
        "variant_role": "VAL_locked_selection",
        "quality_dimensions": "Q6_selection_utility_publication_readiness",
        "owns_generation": False,
        "owns_selection": True,
        "owns_final_QA": False,
        "publication_note": "Selection must be VAL-only. TEST must never influence candidate acceptance.",
    },
    {
        "pipeline_stage": "protocol_test_materialization",
        "cell": "Cell 10.9",
        "artifact_or_output": "A0_PROTOCOL_TEST; A1_PROTOCOL_TEST; A2_PROTOCOL_TEST",
        "variant": "A0/A1/A2",
        "variant_role": "TEST_length_materialization",
        "quality_dimensions": "Q1_capture_observability_fidelity|Q2_value_distribution_fidelity|Q3_temporal_dynamics_fidelity|Q5_schema_domain_contract_fidelity",
        "owns_generation": True,
        "owns_selection": False,
        "owns_final_QA": False,
        "publication_note": "Materializes locked variants only; must not select or fit.",
    },
    {
        "pipeline_stage": "protocol_final_QA_publication",
        "cell": "Cell 11",
        "artifact_or_output": "A0_PROTOCOL_FINAL; A1_PROTOCOL_FINAL; A2_PROTOCOL_FINAL; quality reports",
        "variant": "A0/A1/A2",
        "variant_role": "final_protocol_QA_and_manifest",
        "quality_dimensions": "Q1_capture_observability_fidelity|Q2_value_distribution_fidelity|Q3_temporal_dynamics_fidelity|Q5_schema_domain_contract_fidelity|Q6_selection_utility_publication_readiness",
        "owns_generation": False,
        "owns_selection": False,
        "owns_final_QA": True,
        "publication_note": "TEST is final QA only. A0 diagnostic; A1 baseline; A2 candidate.",
    },
    {
        "pipeline_stage": "iot_midstage_generation",
        "cell": "Cell 12.a/12.b",
        "artifact_or_output": "IoT drivers, binary targets, masks, conditioning state",
        "variant": "A2",
        "variant_role": "future_or_downstream_full_CPS_generation",
        "quality_dimensions": "Q1_capture_observability_fidelity|Q3_temporal_dynamics_fidelity|Q4_cross_modal_coupling_fidelity|Q5_schema_domain_contract_fidelity",
        "owns_generation": True,
        "owns_selection": False,
        "owns_final_QA": False,
        "publication_note": "Must generate IoT state in the same CPS timeline and protocol context, not standalone.",
    },
    {
        "pipeline_stage": "iot_value_portfolio_selection",
        "cell": "Cell 12.c.3",
        "artifact_or_output": "cell12c3_locked_iot_value_selection.csv",
        "variant": "A2",
        "variant_role": "VAL_locked_iot_value_selection",
        "quality_dimensions": "Q2_value_distribution_fidelity|Q3_temporal_dynamics_fidelity|Q6_selection_utility_publication_readiness",
        "owns_generation": False,
        "owns_selection": True,
        "owns_final_QA": False,
        "publication_note": "IoT value generator selection must remain VAL-only.",
    },
    {
        "pipeline_stage": "iot_value_materialization",
        "cell": "Cell 12.c.4/12.c.5",
        "artifact_or_output": "IOT_SELECTED_VALUES_TEST; IOT_FINAL_VALUES_TEST",
        "variant": "A2",
        "variant_role": "TEST_length_iot_value_materialization_and_enforcement",
        "quality_dimensions": "Q2_value_distribution_fidelity|Q3_temporal_dynamics_fidelity|Q5_schema_domain_contract_fidelity",
        "owns_generation": True,
        "owns_selection": False,
        "owns_final_QA": False,
        "publication_note": "Must obey locked selection and final mask/domain/schema enforcement.",
    },
    {
        "pipeline_stage": "iot_final_QA",
        "cell": "Cell 12.c.6",
        "artifact_or_output": "cell12c6_final_test_qa_metrics.csv; publication manifest",
        "variant": "A2",
        "variant_role": "final_iot_value_QA",
        "quality_dimensions": "Q2_value_distribution_fidelity|Q3_temporal_dynamics_fidelity|Q5_schema_domain_contract_fidelity|Q6_selection_utility_publication_readiness",
        "owns_generation": False,
        "owns_selection": False,
        "owns_final_QA": True,
        "publication_note": "TEST is final QA only. Failures are publication blockers until resolved or explicitly scoped out.",
    },
    {
        "pipeline_stage": "whole_cps_coupling_QA",
        "cell": "Future Cell 12.d or Cell 13",
        "artifact_or_output": "whole_cps_coupling_audit",
        "variant": "A2",
        "variant_role": "final_coupled_CPS_release_gate",
        "quality_dimensions": "Q4_cross_modal_coupling_fidelity|Q6_selection_utility_publication_readiness",
        "owns_generation": False,
        "owns_selection": False,
        "owns_final_QA": True,
        "publication_note": "Required before claiming full coupled CPS realism.",
    },
]

QUALITY_ATTRIBUTION_MATRIX = pd.DataFrame(quality_attribution_rows)

# Save scaffold artifacts.
_write_json(QUALITY_DIMENSION_REGISTRY_PATH, QUALITY_DIMENSION_REGISTRY)
_write_json(A0_A1_A2_VARIANT_MANIFEST_PATH, A0_A1_A2_VARIANT_MANIFEST)
QUALITY_ATTRIBUTION_MATRIX.to_csv(QUALITY_ATTRIBUTION_MATRIX_PATH, index=False)

# Hard validation: the scaffold must exist and must define Q1–Q6.
_required_qids = {
    "Q1_capture_observability_fidelity",
    "Q2_value_distribution_fidelity",
    "Q3_temporal_dynamics_fidelity",
    "Q4_cross_modal_coupling_fidelity",
    "Q5_schema_domain_contract_fidelity",
    "Q6_selection_utility_publication_readiness",
}
_observed_qids = set(QUALITY_DIMENSION_REGISTRY.get("dimensions", {}).keys())
if _observed_qids != _required_qids:
    raise RuntimeError(
        "[Cell11] Quality dimension registry is incomplete. "
        f"observed={sorted(_observed_qids)} expected={sorted(_required_qids)}"
    )

if not os.path.exists(QUALITY_DIMENSION_REGISTRY_PATH):
    raise RuntimeError(f"[Cell11] Failed to save {QUALITY_DIMENSION_REGISTRY_PATH}")
if not os.path.exists(A0_A1_A2_VARIANT_MANIFEST_PATH):
    raise RuntimeError(f"[Cell11] Failed to save {A0_A1_A2_VARIANT_MANIFEST_PATH}")
if not os.path.exists(QUALITY_ATTRIBUTION_MATRIX_PATH):
    raise RuntimeError(f"[Cell11] Failed to save {QUALITY_ATTRIBUTION_MATRIX_PATH}")

globals()["QUALITY_DIMENSION_REGISTRY"] = QUALITY_DIMENSION_REGISTRY
globals()["QUALITY_DIMENSION_REGISTRY_PATH"] = QUALITY_DIMENSION_REGISTRY_PATH
globals()["A0_A1_A2_VARIANT_MANIFEST"] = A0_A1_A2_VARIANT_MANIFEST
globals()["A0_A1_A2_VARIANT_MANIFEST_PATH"] = A0_A1_A2_VARIANT_MANIFEST_PATH
globals()["QUALITY_ATTRIBUTION_MATRIX"] = QUALITY_ATTRIBUTION_MATRIX
globals()["QUALITY_ATTRIBUTION_MATRIX_PATH"] = QUALITY_ATTRIBUTION_MATRIX_PATH

log(f"[Cell11] Saved quality dimension registry: {QUALITY_DIMENSION_REGISTRY_PATH}")
log(f"[Cell11] Saved A0/A1/A2 variant manifest: {A0_A1_A2_VARIANT_MANIFEST_PATH}")
log(
    f"[Cell11] Saved quality attribution matrix: {QUALITY_ATTRIBUTION_MATRIX_PATH} | "
    f"rows={len(QUALITY_ATTRIBUTION_MATRIX)}"
)
log(
    "[Cell11] Phase 1 quality-decomposition scaffold ready | "
    f"Q_dimensions={len(_observed_qids)} | "
    f"A0_publication_role={A0_A1_A2_VARIANT_MANIFEST['variants']['A0']['publication_role']} | "
    f"A1_publication_role={A0_A1_A2_VARIANT_MANIFEST['variants']['A1']['publication_role']} | "
    f"A2_publication_role={A0_A1_A2_VARIANT_MANIFEST['variants']['A2']['publication_role']} | "
    f"A2_protocol_status={A0_A1_A2_VARIANT_MANIFEST['variants']['A2']['protocol_value_status']}"
)
# ---------------------------------------------------------------------
# 9) Publish final protocol artifacts
# ---------------------------------------------------------------------
A0_PROTOCOL.to_parquet(FINAL_A0_PATH, index=False)
A1_PROTOCOL.to_parquet(FINAL_A1_PATH, index=False)
A2_PROTOCOL.to_parquet(FINAL_A2_PATH, index=False)

globals()["CELL11_A0_PROTOCOL"] = A0_PROTOCOL
globals()["CELL11_A1_PROTOCOL"] = A1_PROTOCOL
globals()["CELL11_A2_PROTOCOL"] = A2_PROTOCOL
globals()["CELL11_REAL_TEST_PROTOCOL"] = REAL_TEST_PROTOCOL
globals()["CELL11_PROTOCOL_QUALITY_REPORT"] = quality_report
globals()["CELL11_VARIANT_QUALITY_SUMMARY"] = variant_summary
globals()["CELL11_SELECTION_CONSISTENCY_AUDIT"] = selection_consistency
globals()["CELL11_DELTA_AUDIT"] = delta_audit
globals()["CELL11_MASK_CONTRACT_AUDIT"] = mask_contract_audit
# ---------------------------------------------------------------------
# 9.5) Downstream IoT protocol-context handoff
# ---------------------------------------------------------------------
# Purpose:
# - Publish the protocol-side context that downstream IoT synthesis is allowed
#   to use.
# - IoT synthesis must be CPS-coupled to the finalized synthetic protocol
#   variant A2, not generated as a standalone tabular IoT block.
# - Real TEST protocol values remain QA-only and are explicitly forbidden
#   for downstream IoT fitting, selection, materialization, repair, or calibration.

IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH = os.path.join(
    OUT_ART,
    "cell11_iot_downstream_protocol_context.json",
)
IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH = os.path.join(
    OUT_REP,
    "cell11_iot_downstream_protocol_activity_summary.csv",
)

if A2_PROTOCOL.shape[0] != N_TEST:
    raise RuntimeError(
        f"[Cell11] A2_PROTOCOL row mismatch before IoT handoff: "
        f"{A2_PROTOCOL.shape[0]} vs N_TEST={N_TEST}"
    )

if not isinstance(SYN_TEST_MASKS, dict) or not SYN_TEST_MASKS:
    raise RuntimeError("[Cell11] SYN_TEST_MASKS unavailable for IoT downstream handoff.")

if not PROTO_COLS:
    raise RuntimeError("[Cell11] PROTO_COLS is empty before IoT downstream handoff.")

for _tier in ["router", "ota", "zigbee", "zwave"]:
    if _tier not in SYN_TEST_MASKS:
        raise RuntimeError(f"[Cell11] SYN_TEST_MASKS missing tier for IoT handoff: {_tier}")
    _m = np.asarray(SYN_TEST_MASKS[_tier], dtype=bool).reshape(-1)
    if _m.shape[0] != N_TEST:
        raise RuntimeError(
            f"[Cell11] SYN_TEST_MASKS[{_tier}] length mismatch for IoT handoff: "
            f"{_m.shape[0]} vs {N_TEST}"
        )

# Downstream IoT conditioning summary.
# Important: computed ONLY from A2_PROTOCOL and SYN_TEST_MASKS.
# Do not use REAL_TEST_PROTOCOL, REAL_TEST_MASKS, or df_te values here.
iot_activity_rows = []

for c in PROTO_COLS:
    tier = _tier_of(c)
    if tier is None:
        continue

    active = np.asarray(SYN_TEST_MASKS[tier], dtype=bool).reshape(-1)
    x = pd.to_numeric(A2_PROTOCOL[c], errors="coerce").to_numpy(dtype=np.float64)

    if x.shape[0] != N_TEST:
        raise RuntimeError(
            f"[Cell11] A2_PROTOCOL[{c}] length mismatch in IoT activity summary: "
            f"{x.shape[0]} vs {N_TEST}"
        )

    xa = x[active]
    finite = np.isfinite(xa)
    xf = xa[finite]

    if xf.size == 0:
        mean_v = np.nan
        std_v = np.nan
        nz_v = np.nan
        q50_v = np.nan
        q95_v = np.nan
        q99_v = np.nan
        transition_v = np.nan
        burst_v = np.nan
    else:
        mean_v = float(np.mean(xf))
        std_v = float(np.std(xf)) if xf.size > 1 else 0.0
        nz_v = float(np.mean(xf > 0))
        q50_v = float(np.quantile(xf, 0.50))
        q95_v = float(np.quantile(xf, 0.95))
        q99_v = float(np.quantile(xf, 0.99))
        transition_v = _transition_rate(xf)
        burst_v = _burst_rate(xf)

    iot_activity_rows.append({
        "col": c,
        "tier": tier,
        "synthetic_active_rate": float(np.mean(active)),
        "active_n": int(active.sum()),
        "finite_n_under_synthetic_mask": int(finite.sum()),
        "finite_rate_under_synthetic_mask": (
            float(finite.mean()) if finite.size else np.nan
        ),
        "mean_under_synthetic_mask": mean_v,
        "std_under_synthetic_mask": std_v,
        "nonzero_rate_under_synthetic_mask": nz_v,
        "q50_under_synthetic_mask": q50_v,
        "q95_under_synthetic_mask": q95_v,
        "q99_under_synthetic_mask": q99_v,
        "transition_rate_under_synthetic_mask": transition_v,
        "burst_rate_under_synthetic_mask": burst_v,
        "is_countlike": bool(_is_countlike(c)),
        "is_unique_like": bool(_is_unique_like(c)),
    })

CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY = pd.DataFrame(iot_activity_rows)

if CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY.empty:
    raise RuntimeError("[Cell11] IoT downstream protocol activity summary is empty.")

CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY.to_csv(
    IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH,
    index=False,
)

# Optional upstream IoT contract pointers created by Cell 4.5 / Cell 6.5.
iot_feature_groups_path = os.path.join(OUT_ART, "iot_feature_groups.json")
iot_semantic_contract_path = os.path.join(OUT_ART, "iot_semantic_contract.json")
iot_partition_contract_path = os.path.join(OUT_ART, "iot_partition_contract.json")
iot_cell_routing_manifest_path = os.path.join(OUT_ART, "iot_cell_routing_manifest.json")

iot_cps_coupling_policy_path = globals().get(
    "IOT_CPS_COUPLING_POLICY_JSON",
    os.path.join(OUT_ART, "iot_cps_coupling_policy.json"),
)
iot_group_loader_contract_path = globals().get(
    "IOT_GROUP_LOADER_CONTRACT_JSON",
    os.path.join(OUT_ART, "iot_group_loader_contract.json"),
)

def _existing_or_none(path):
    return str(path) if path and os.path.exists(str(path)) else None

CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT = {
    "version": "cell11_iot_downstream_protocol_context_v1",
    "declared_in_cell": "Cell11",
    "N_TEST": int(N_TEST),
    "seed": int(seed),

    "canonical_protocol_context_for_iot": {
        "variant": "A2",
        "description": (
            "A2 is the finalized VAL-selected synthetic protocol variant under "
            "synthetic TEST masks. Downstream IoT synthesis must condition on this "
            "synthetic protocol context, not on real TEST protocol values."
        ),
        "A2_PROTOCOL_FINAL": FINAL_A2_PATH,
        "A1_PROTOCOL_FINAL": FINAL_A1_PATH,
        "A0_PROTOCOL_FINAL": FINAL_A0_PATH,
        "A2_PROTOCOL_FINAL_sha256": _sha256_file(FINAL_A2_PATH),
        "A1_PROTOCOL_FINAL_sha256": _sha256_file(FINAL_A1_PATH),
        "A0_PROTOCOL_FINAL_sha256": _sha256_file(FINAL_A0_PATH),
    },

    "protocol_columns": {
        "proto_cols_n": int(len(PROTO_COLS)),
        "proto_cols": list(PROTO_COLS),
        "proto_cols_sha256": _sha256_jsonable(PROTO_COLS),
        "protocol_tiers": sorted(list(set([_tier_of(c) for c in PROTO_COLS if _tier_of(c) is not None]))),
    },

    "synthetic_protocol_masks_for_iot": {
        "source": "SYN_TEST_MASKS",
        "usage": "allowed_for_downstream_iot_generation",
        "mask_rates": syn_mask_rates,
        "mask_tiers": {
            t: {
                "active_n": int(np.asarray(SYN_TEST_MASKS[t], dtype=bool).sum()),
                "active_rate": float(np.asarray(SYN_TEST_MASKS[t], dtype=bool).mean()),
            }
            for t in ["router", "ota", "zigbee", "zwave"]
        },
    },

    "real_test_protocol_reference": {
        "source": "REAL_TEST_PROTOCOL",
        "usage": "forbidden_for_downstream_iot_generation__allowed_only_for_final_QA_inside_Cell11",
        "real_mask_rates_QA_only": real_mask_rates,
        "synthetic_minus_real_mask_drift_QA_only": mask_rate_drift,
    },

    "downstream_iot_allowed_inputs": [
        "CELL11_A2_PROTOCOL",
        "A2_PROTOCOL_FINAL.parquet",
        "SYN_TEST_MASKS",
        "TOD arrays or equivalent synthetic TOD context",
        "regime arrays or equivalent synthetic regime context",
        "IOT_DRIVER_COLS / IOT_DRIVER_TARGETS from Cell6.5",
        "IOT_META_COLS and derived IoT meta/conditioning groups from Cell6.5",
        "IOT_CPS_COUPLING_POLICY",
        "cell11_iot_downstream_protocol_activity_summary.csv",
    ],

    "downstream_iot_forbidden_inputs": [
        "CELL11_REAL_TEST_PROTOCOL",
        "REAL_TEST_PROTOCOL",
        "REAL_TEST_MASKS",
        "df_te protocol values",
        "any TEST protocol statistics for IoT fitting",
        "any TEST protocol statistics for IoT generator selection",
        "any TEST protocol statistics for IoT materialization",
        "any TEST protocol statistics for IoT repair or calibration",
    ],

    "leakage_contract": {
        "test_protocol_values_used_for_iot_generation": False,
        "test_protocol_values_used_for_iot_selection": False,
        "test_protocol_values_used_for_iot_fitting": False,
        "test_protocol_values_used_for_iot_materialization": False,
        "test_protocol_values_used_for_iot_repair": False,
        "test_protocol_values_used_for_iot_calibration": False,
        "test_protocol_values_allowed_for_final_QA_only": True,
        "downstream_iot_generation_must_use_A2_synthetic_protocol_context": True,
        "downstream_iot_generation_must_use_synthetic_masks": True,
        "standalone_iot_generation_allowed": False,
    },

    "required_downstream_iot_behavior": {
        "protocol_context_ready_for_downstream_iot_generation": True,
        "condition_on_A2_protocol_context": True,
        "condition_on_synthetic_protocol_masks": True,
        "condition_on_shared_timeline": True,
        "condition_on_TOD_and_regime_context": True,
        "condition_on_iot_driver_and_meta_contracts": True,
        "selection_split": "VAL_only",
        "final_TEST_usage": "QA_only",
    },

    "activity_summary": {
        "path": IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH,
        "rows": int(len(CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY)),
        "computed_from": "A2_PROTOCOL_plus_SYN_TEST_MASKS_only",
        "uses_REAL_TEST_PROTOCOL": False,
        "sha256": _sha256_file(IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH),
    },

    "iot_contract_pointers": {
        "iot_cps_coupling_policy_json": _existing_or_none(iot_cps_coupling_policy_path),
        "iot_group_loader_contract_json": _existing_or_none(iot_group_loader_contract_path),
        "iot_feature_groups_json": _existing_or_none(iot_feature_groups_path),
        "iot_semantic_contract_json": _existing_or_none(iot_semantic_contract_path),
        "iot_partition_contract_json": _existing_or_none(iot_partition_contract_path),
        "iot_cell_routing_manifest_json": _existing_or_none(iot_cell_routing_manifest_path),
    },

    "cell11_audit_reports": {
        "quality_report": QUALITY_REPORT_PATH,
        "variant_quality_summary": SUMMARY_CSV_PATH,
        "mask_contract_audit": MASK_CONTRACT_PATH,
        "delta_audit": DELTA_AUDIT_PATH,
        "selection_consistency_audit": SELECTION_CONSISTENCY_PATH,
        "final_protocol_audit": FINAL_AUDIT_PATH,
    },
}

_write_json(
    IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH,
    CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT,
)

# Explicit globals for Cell 12.a+.
globals()["CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT"] = CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT
globals()["CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH"] = IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH
globals()["CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY"] = CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY
globals()["CELL11_A2_PROTOCOL_CONTEXT_FOR_IOT"] = A2_PROTOCOL
globals()["CELL11_SYN_TEST_MASKS_FOR_IOT"] = SYN_TEST_MASKS

log(
    "[Cell11] Saved IoT downstream protocol context: "
    f"{IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH}"
)
log(
    "[Cell11] Saved IoT downstream protocol activity summary: "
    f"{IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH} | "
    f"rows={len(CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY)}"
)
log(
    "[Cell11] CPS-coupled IoT downstream handoff ready | "
    "uses=A2_PROTOCOL_FINAL | synthetic_masks=True | real_TEST_values_forbidden=True"
)

# ---------------------------------------------------------------------
# 10) Manifest and artifact hashes
# ---------------------------------------------------------------------
artifact_hashes = {
    "A0_PROTOCOL_FINAL_sha256": _sha256_file(FINAL_A0_PATH),
    "A1_PROTOCOL_FINAL_sha256": _sha256_file(FINAL_A1_PATH),
    "A2_PROTOCOL_FINAL_sha256": _sha256_file(FINAL_A2_PATH),

    "A0_PROTOCOL_TEST_input_sha256": _sha256_file(A0_PROTOCOL_PATH),
    "A1_PROTOCOL_TEST_input_sha256": _sha256_file(A1_PROTOCOL_PATH),
    "A2_PROTOCOL_TEST_input_sha256": _sha256_file(A2_PROTOCOL_PATH),

    "quality_report_sha256": _sha256_file(QUALITY_REPORT_PATH),
    "mask_contract_audit_sha256": _sha256_file(MASK_CONTRACT_PATH),
    "delta_audit_sha256": _sha256_file(DELTA_AUDIT_PATH),
    "selection_consistency_audit_sha256": _sha256_file(SELECTION_CONSISTENCY_PATH),
    "variant_summary_sha256": _sha256_file(SUMMARY_CSV_PATH),

    "quality_dimension_registry_sha256": _sha256_file(QUALITY_DIMENSION_REGISTRY_PATH),
    "a0_a1_a2_variant_manifest_sha256": _sha256_file(A0_A1_A2_VARIANT_MANIFEST_PATH),
    "quality_attribution_matrix_sha256": _sha256_file(QUALITY_ATTRIBUTION_MATRIX_PATH),

    "selected_by_col_sha256": _sha256_jsonable(selected_by_col),
    "proto_cols_sha256": _sha256_jsonable(PROTO_COLS),
    "quality_dimension_registry_sha256": _sha256_file(QUALITY_DIMENSION_REGISTRY_PATH),
    "a0_a1_a2_variant_manifest_sha256": _sha256_file(A0_A1_A2_VARIANT_MANIFEST_PATH),
    "quality_attribution_matrix_sha256": _sha256_file(QUALITY_ATTRIBUTION_MATRIX_PATH),
}

_write_json(HASHES_PATH, artifact_hashes)

audit = {
    "version": "cell11_v13_0_final_a0_a1_a2_protocol_variant_qa",
    "N_TEST": int(N_TEST),
    "proto_cols_n": int(len(PROTO_COLS)),
    "proto_cols": list(PROTO_COLS),

    "loaded_inputs": {
        "A0_PROTOCOL_TEST": A0_PROTOCOL_PATH,
        "A1_PROTOCOL_TEST": A1_PROTOCOL_PATH,
        "A2_PROTOCOL_TEST": A2_PROTOCOL_PATH,
        "cell10_9_manifest": CELL10_9_MANIFEST_PATH if os.path.exists(CELL10_9_MANIFEST_PATH) else None,
        "cell10_selection_json": CELL10_SELECTION_JSON if os.path.exists(CELL10_SELECTION_JSON) else None,
        "precell11_contract": PRECELL11_CONTRACT_PATH if os.path.exists(PRECELL11_CONTRACT_PATH) else None,
    },

    "published_outputs": {
        "A0_PROTOCOL_FINAL": FINAL_A0_PATH,
        "A1_PROTOCOL_FINAL": FINAL_A1_PATH,
        "A2_PROTOCOL_FINAL": FINAL_A2_PATH,
    },

    "reports": {
        "quality_report": QUALITY_REPORT_PATH,
        "mask_contract_audit": MASK_CONTRACT_PATH,
        "delta_audit": DELTA_AUDIT_PATH,
        "selection_consistency_audit": SELECTION_CONSISTENCY_PATH,
        "variant_quality_summary": SUMMARY_CSV_PATH,
    
        # Phase 1 quality-decomposition scaffold
        "quality_dimension_registry": QUALITY_DIMENSION_REGISTRY_PATH,
        "a0_a1_a2_variant_manifest": A0_A1_A2_VARIANT_MANIFEST_PATH,
        "quality_attribution_matrix": QUALITY_ATTRIBUTION_MATRIX_PATH,
    
        "artifact_hashes": HASHES_PATH,
    
        # IoT downstream CPS-coupling handoff
        "iot_downstream_protocol_context": IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH,
        "iot_downstream_protocol_activity_summary": IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH,
    },

    "variant_semantics": A0_A1_A2_VARIANT_MANIFEST["variants"],

    "quality_decomposition_scaffold": {
        "quality_dimension_registry": QUALITY_DIMENSION_REGISTRY_PATH,
        "a0_a1_a2_variant_manifest": A0_A1_A2_VARIANT_MANIFEST_PATH,
        "quality_attribution_matrix": QUALITY_ATTRIBUTION_MATRIX_PATH,
        "quality_dimensions": list(QUALITY_DIMENSION_REGISTRY["dimensions"].keys()),
        "mandatory_mapping_policy": QUALITY_DIMENSION_REGISTRY["mandatory_mapping_policy"],
        "a0_publication_role": A0_A1_A2_VARIANT_MANIFEST["variants"]["A0"]["publication_role"],
        "a1_publication_role": A0_A1_A2_VARIANT_MANIFEST["variants"]["A1"]["publication_role"],
        "a2_publication_role": A0_A1_A2_VARIANT_MANIFEST["variants"]["A2"]["publication_role"],
        "a2_protocol_value_status": A0_A1_A2_VARIANT_MANIFEST["variants"]["A2"]["protocol_value_status"],
    },

    "mask_rates": {
        "real_TEST": real_mask_rates,
        "synthetic_TEST": syn_mask_rates,
        "synthetic_minus_real_TEST": mask_rate_drift,
    },

    "selection": {
        "selected_counts": dict(selected_counts),
        "all_a1_selected": bool(all_a1_selected),
        "a2_equals_a1": bool(a2_equals_a1),
        "a2_equals_a0": bool(a2_equals_a0),
        "selection_summary": selection_summary,
    },

    "qa_summary": qa_summary,

    "quality_summary": variant_summary.to_dict("records"),

    "leakage_contract": {
        "this_cell_generates_values": False,
        "this_cell_fits_models": False,
        "this_cell_selects_generators": False,
        "test_used_for_selection": False,
        "test_used_for_fitting": False,
        "test_used_for_final_QA_only": True,
        "a0_uses_real_test_masks_only_not_test_values_for_generation": True,
        "a1_a2_use_synthetic_test_masks": True,

        "protocol_context_ready_for_downstream_iot_generation": True,
        "downstream_iot_generation_uses_A2_protocol_context": True,
        "downstream_iot_generation_uses_synthetic_protocol_masks": True,
        "downstream_iot_generation_uses_real_TEST_protocol_values": False,
        "downstream_iot_generation_uses_real_TEST_masks": False,
        "standalone_iot_generation_allowed": False,
    },

    "methodological_note": (
        "Cell 11 is a final QA/publication cell for the already materialized protocol variants. "
        "Generator selection is locked upstream by Cell 10.8 using VAL only. Cell 10.9 materializes "
        "A0/A1/A2. This cell validates mask/value contracts, checks A1/A2 selection consistency, "
        "computes final TEST-reference QA metrics, and publishes canonical final protocol artifacts. "
        "No candidate is accepted, rejected, fitted, or generated in this cell."
    ),
}

_write_json(FINAL_AUDIT_PATH, audit)

manifest = {
    "version": "cell11_v13_1_final_protocol_variant_manifest",
    "seed": int(seed),
    "N_TEST": int(N_TEST),
    "proto_cols_n": int(len(PROTO_COLS)),
    "outputs": {
        "A0_PROTOCOL_FINAL": FINAL_A0_PATH,
        "A1_PROTOCOL_FINAL": FINAL_A1_PATH,
        "A2_PROTOCOL_FINAL": FINAL_A2_PATH,
    },
    "source_inputs": {
        "A0_PROTOCOL_TEST": A0_PROTOCOL_PATH,
        "A1_PROTOCOL_TEST": A1_PROTOCOL_PATH,
        "A2_PROTOCOL_TEST": A2_PROTOCOL_PATH,
        "cell10_9_manifest": CELL10_9_MANIFEST_PATH if os.path.exists(CELL10_9_MANIFEST_PATH) else None,
    },
    "reports": {
        "quality_report": QUALITY_REPORT_PATH,
        "mask_contract_audit": MASK_CONTRACT_PATH,
        "delta_audit": DELTA_AUDIT_PATH,
        "selection_consistency_audit": SELECTION_CONSISTENCY_PATH,
        "variant_quality_summary": SUMMARY_CSV_PATH,
    
        # Phase 1 quality-decomposition scaffold
        "quality_dimension_registry": QUALITY_DIMENSION_REGISTRY_PATH,
        "a0_a1_a2_variant_manifest": A0_A1_A2_VARIANT_MANIFEST_PATH,
        "quality_attribution_matrix": QUALITY_ATTRIBUTION_MATRIX_PATH,
    
        "final_audit": FINAL_AUDIT_PATH,
        "artifact_hashes": HASHES_PATH,
    
        # IoT downstream CPS-coupling handoff
        "iot_downstream_protocol_context": IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH,
        "iot_downstream_protocol_activity_summary": IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH,
    },
    "selected_counts": dict(selected_counts),
    "all_a1_selected": bool(all_a1_selected),
    "a2_equals_a1": bool(a2_equals_a1),
    "a2_equals_a0": bool(a2_equals_a0),
    "qa_summary": qa_summary,
    "quality_decomposition_scaffold": {
        "quality_dimension_registry": QUALITY_DIMENSION_REGISTRY_PATH,
        "a0_a1_a2_variant_manifest": A0_A1_A2_VARIANT_MANIFEST_PATH,
        "quality_attribution_matrix": QUALITY_ATTRIBUTION_MATRIX_PATH,
        "quality_dimensions": list(QUALITY_DIMENSION_REGISTRY["dimensions"].keys()),
        "a0_publication_role": A0_A1_A2_VARIANT_MANIFEST["variants"]["A0"]["publication_role"],
        "a1_publication_role": A0_A1_A2_VARIANT_MANIFEST["variants"]["A1"]["publication_role"],
        "a2_publication_role": A0_A1_A2_VARIANT_MANIFEST["variants"]["A2"]["publication_role"],
        "a2_protocol_value_status": A0_A1_A2_VARIANT_MANIFEST["variants"]["A2"]["protocol_value_status"],
    },
    "iot_downstream_handoff": {
        "protocol_context_ready_for_downstream_iot_generation": True,
        "downstream_iot_generation_uses_A2_protocol_context": True,
        "downstream_iot_generation_uses_synthetic_protocol_masks": True,
        "downstream_iot_generation_uses_real_TEST_protocol_values": False,
        "standalone_iot_generation_allowed": False,
        "context_json": IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH,
        "activity_summary_csv": IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH,
    },
    "artifact_hashes": artifact_hashes,
}

_write_json(FINAL_MANIFEST_PATH, manifest)

globals()["CELL11_FINAL_PROTOCOL_VARIANT_MANIFEST"] = manifest
globals()["CELL11_FINAL_PROTOCOL_VARIANT_AUDIT"] = audit
globals()["CELL11_FINAL_PROTOCOL_VARIANT_MANIFEST_PATH"] = FINAL_MANIFEST_PATH
globals()["CELL11_FINAL_PROTOCOL_VARIANT_AUDIT_PATH"] = FINAL_AUDIT_PATH


# ---------------------------------------------------------------------
# 11) Final console summary
# ---------------------------------------------------------------------
summary_print = variant_summary.set_index("variant").to_dict("index")

log(f"[Cell11] Saved final A0 protocol artifact: {FINAL_A0_PATH} | shape={A0_PROTOCOL.shape}")
log(f"[Cell11] Saved final A1 protocol artifact: {FINAL_A1_PATH} | shape={A1_PROTOCOL.shape}")
log(f"[Cell11] Saved final A2 protocol artifact: {FINAL_A2_PATH} | shape={A2_PROTOCOL.shape}")
log(f"[Cell11] Saved final manifest: {FINAL_MANIFEST_PATH}")
log(f"[Cell11] Saved final audit: {FINAL_AUDIT_PATH}")

log(
    "[Cell11] Final QA summary | "
    f"selected_counts={dict(selected_counts)} | "
    f"all_a1_selected={all_a1_selected} | "
    f"A2_equals_A1={a2_equals_a1} | "
    f"A2_equals_A0={a2_equals_a0} | "
    f"material_A2_regressions={qa_summary['material_a2_regression_cols_n']} | "
    f"A0_median_score={summary_print.get('A0', {}).get('median_score')} | "
    f"A1_median_score={summary_print.get('A1', {}).get('median_score')} | "
    f"A2_median_score={summary_print.get('A2', {}).get('median_score')} | "
    f"A0_best_cols={summary_print.get('A0', {}).get('best_score_cols')} | "
    f"A1_best_cols={summary_print.get('A1', {}).get('best_score_cols')} | "
    f"A2_best_cols={summary_print.get('A2', {}).get('best_score_cols')} | "
    f"A0_role={A0_A1_A2_VARIANT_MANIFEST['variants']['A0']['publication_role']} | "
    f"A1_role={A0_A1_A2_VARIANT_MANIFEST['variants']['A1']['publication_role']} | "
    f"A2_role={A0_A1_A2_VARIANT_MANIFEST['variants']['A2']['publication_role']} | "
    f"A2_protocol_status={A0_A1_A2_VARIANT_MANIFEST['variants']['A2']['protocol_value_status']} | "
    f"Q_dimensions={len(QUALITY_DIMENSION_REGISTRY['dimensions'])}"
)

log("--- END: Cell 11 — Final A0/A1/A2 protocol variant QA + publication (v13.1-THESIS contract-locked) ---")