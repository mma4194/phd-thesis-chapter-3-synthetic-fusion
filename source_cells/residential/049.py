# ==========================================================
# CELL 12.c.0 — IoT value synthesis foundation/contract
# v1.2-THESIS STUDY-THESIS modular stack
#
# Role:
#   Foundation cell for modular Cell 12.c.
#
# This cell DOES:
#   1) Validate Cell 12.b v27 scaffold outputs.
#   2) Validate IoT value/binary/mask/entity-regime artifacts.
#   3) Validate routing policy and portfolio eligibility.
#   4) Register Cell 11 protocol A2 context path without loading TEST-length protocol values.
#   5) Define shared metric helpers for later 12.c generator cells.
#   6) Define strict no-TEST-selection policy.
#   7) Export a 12.c.0 contract manifest and audit files.
#
# This cell DOES NOT:
#   - generate any IoT continuous values
#   - train DDPM, TimeGAN, CTGAN, TVAE, Copula, Markov, NB, or any model
#   - select generators using TEST
#   - mutate VALUE_AVAIL_SYN, IOT_SYN_BINARY, IOT_SYN_MASKS, or ENTITY_REGIME_SYN
#   - consume stale DDPM outputs from old Cell 12.b versions
#
# Scientific boundary:
#   TRAIN/VAL may be used later for fitting/selection.
#   TEST is only the target-length/schema/final-QA domain.
#   Generator selection must be VAL-locked before final TEST QA.
# ==========================================================

log("--- START: Cell 12.c.0 — IoT value synthesis foundation/contract (v1.2-THESIS contract-locked modular STUDY-THESIS) ---")

# ----------------------------------------------------------
# Imports
# ----------------------------------------------------------
import os
import re
import gc
import json
import math
import hashlib
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except Exception as _sk_exc:
    raise RuntimeError(f"[Cell12.c.0] scikit-learn is required for later shared metrics: {_sk_exc}")

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_base = [
    "CFG", "log",
    "df_tr", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "DRV_COLS", "BIN_COLS", "CONT_COLS", "CONT_MASK_COLS", "CONT_VALUE_COLS",
    "FAMILY_MAP", "FAMILY_GROUPS",
    "DDPM_COLS", "NON_DDPM_COLS",
    "IOT_SYN_MASKS", "IOT_SYN_BINARY", "VALUE_AVAIL_SYN",
    "VALUE_AVAIL_MODE", "VALUE_AVAIL_RATE_MAP",
    "ENTITY_REGIME_SYN", "ENTITY_DRIVER_ACTIVITY_SYN",
    "ENTITY_LIST",
    "CANON_ENTITY_MAP",
    "CONT_ROUTING_POLICY_DF",

    # Phase-1 role-family contracts from Cells 12.a/12.b
    "IOT_CONTINUOUS_VALUE_TARGETS",
    "IOT_BINARY_TARGETS",
    "IOT_DRIVER_TARGETS",
    "IOT_TEST_VALUE_ACCESS_POLICY",
]

_missing_base = [k for k in _required_base if k not in globals()]
if _missing_base:
    raise RuntimeError(f"[Cell12.c.0] Missing required globals from Cells 12.a/12.b: {_missing_base}")

if "df_val" in globals() and isinstance(globals()["df_val"], pd.DataFrame):
    df_val = globals()["df_val"]
    log("[Cell12.c.0] Using VAL dataframe from global: df_val")
elif "df_va" in globals() and isinstance(globals()["df_va"], pd.DataFrame):
    df_val = globals()["df_va"]
    log("[Cell12.c.0] Using VAL dataframe from global: df_va")
else:
    raise RuntimeError("[Cell12.c.0] Missing canonical VAL dataframe: expected df_val or df_va.")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")


os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

# ----------------------------------------------------------
# 0B) Upstream contract locks
# ----------------------------------------------------------
def _read_json_dict_required_12c0(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"[Cell12.c.0] Missing required {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell12.c.0] Required {label} must be a JSON object: {path}")
    return obj

CELL12B_SCAFFOLD_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "cell12b_iot_scaffold_contract_v27_1_1_THESIS.json",
)
CELL12A_FOUNDATION_MANIFEST_PATH = os.path.join(
    ARTDIR,
    "cell12a_foundation_manifest_v24_0.json",
)
CELL10_8_SELECTOR_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "cell10_8_val_only_selector_contract_v3_6_THESIS.json",
)
CELL10_9_MATERIALIZER_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "cell10_9_protocol_variant_materialization_contract_v2_1_THESIS.json",
)
PRECELL11_RUNTIME_CONTRACT_PATH = os.path.join(
    CONTRACT_DIR,
    "pre_cell11_a0_a1_a2_variant_runtime_contract_v4_1_THESIS.json",
)
CELL11_FINAL_MANIFEST_PATH_REQUIRED = os.path.join(
    ARTDIR,
    "cell11_final_protocol_variant_manifest.json",
)
CELL11B_QUARANTINE_MANIFEST_PATH = os.path.join(
    ARTDIR,
    "cell11b_protocol_test_regression_quarantine_manifest.json",
)

CELL12B_SCAFFOLD_CONTRACT = _read_json_dict_required_12c0(
    CELL12B_SCAFFOLD_CONTRACT_PATH,
    "Cell 12.b v27.1.1 scaffold contract",
)
CELL12A_FOUNDATION_MANIFEST = _read_json_dict_required_12c0(
    CELL12A_FOUNDATION_MANIFEST_PATH,
    "Cell 12.a foundation manifest",
)
CELL10_8_SELECTOR_CONTRACT = _read_json_dict_required_12c0(
    CELL10_8_SELECTOR_CONTRACT_PATH,
    "Cell 10.8 v3.6 selector contract",
)
CELL10_9_MATERIALIZER_CONTRACT = _read_json_dict_required_12c0(
    CELL10_9_MATERIALIZER_CONTRACT_PATH,
    "Cell 10.9 v2.1 materializer contract",
)
PRECELL11_RUNTIME_CONTRACT = _read_json_dict_required_12c0(
    PRECELL11_RUNTIME_CONTRACT_PATH,
    "PRE-CELL11 v4.1 runtime contract",
)
CELL11_FINAL_MANIFEST_REQUIRED = _read_json_dict_required_12c0(
    CELL11_FINAL_MANIFEST_PATH_REQUIRED,
    "Cell 11 final protocol manifest",
)

# Refuse to proceed if noncanonical Cell 11b TEST-quarantine artifacts remain active.
if os.path.exists(CELL11B_QUARANTINE_MANIFEST_PATH):
    _cell11b_manifest = _read_json_dict_required_12c0(
        CELL11B_QUARANTINE_MANIFEST_PATH,
        "noncanonical Cell 11b quarantine manifest",
    )
    _cell11b_outputs = _cell11b_manifest.get("outputs", {}) if isinstance(_cell11b_manifest, dict) else {}
    _cell11b_policy = _cell11b_manifest.get("policy", {}) if isinstance(_cell11b_manifest, dict) else {}
    _cell11b_overwrote = bool(
        _cell11b_policy.get("canonical_A2_overwritten", False)
        or _cell11b_outputs.get("A2_PROTOCOL_FINAL_overwritten")
        or _cell11b_outputs.get("A2_PROTOCOL_TEST_overwritten")
    )
    if _cell11b_overwrote:
        raise RuntimeError(
            "[Cell12.c.0] Refusing to proceed because noncanonical Cell 11b TEST-quarantine "
            "artifacts indicate canonical A2 was overwritten."
        )

# Enforce upstream no-TEST-selection / no-repair lineage.
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
    if bool(CELL10_8_SELECTOR_CONTRACT.get(_key, False)):
        raise RuntimeError(f"[Cell12.c.0] Cell 10.8 selector contract violation: {_key}=True")

for _key in [
    "test_values_read",
    "test_metrics_computed",
    "test_used_for_fitting",
    "test_used_for_thresholding",
    "test_used_for_selection",
    "test_used_for_repair",
    "a0_used_for_selection",
    "downstream_test_rescue_materialization",
]:
    if bool(CELL10_9_MATERIALIZER_CONTRACT.get(_key, False)):
        raise RuntimeError(f"[Cell12.c.0] Cell 10.9 materializer contract violation: {_key}=True")

for _key in [
    "cell11_must_not_select_generators",
    "cell11_must_not_register_ddpm_for_protocol_generation",
    "cell11_must_not_materialize_protocol_candidates",
    "cell11_must_not_apply_test_informed_rescue_or_repair",
]:
    if not bool(PRECELL11_RUNTIME_CONTRACT.get(_key, False)):
        raise RuntimeError(f"[Cell12.c.0] PRE-CELL11 runtime contract missing required flag: {_key}")

_s12b_test_usage = CELL12B_SCAFFOLD_CONTRACT.get("test_usage", {})
if not bool(_s12b_test_usage.get("test_length_index_schema_only", False)):
    raise RuntimeError("[Cell12.c.0] Cell 12.b contract does not limit TEST to length/index/schema use.")

for _key in [
    "test_values_used_for_fitting",
    "test_values_used_for_selection",
    "test_values_used_for_calibration",
    "test_values_used_for_repair",
    "test_values_used_for_continuous_value_generation",
]:
    if bool(_s12b_test_usage.get(_key, False)):
        raise RuntimeError(f"[Cell12.c.0] Cell 12.b scaffold contract violation: {_key}=True")

log(
    "[Cell12.c.0] Upstream contracts locked | "
    "cell12a=True | cell12b_v27_1_1=True | cell10_8_v3_6=True | "
    "cell10_9_v2_1=True | precell11_v4_1=True | cell11=True"
)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

if N_TR <= 0 or N_VAL <= 0 or N_TE <= 0:
    raise RuntimeError(f"[Cell12.c.0] Invalid split lengths: N_TR={N_TR}, N_VAL={N_VAL}, N_TE={N_TE}")

# ----------------------------------------------------------
# 1) Policy lock
# ----------------------------------------------------------
CELL12C0_VERSION = "v1_2_THESIS_contract_locked_modular_foundation_phase1_shared_utilities"

CELL12C_POLICY = {
    "version": CELL12C0_VERSION,
    "role": "foundation_only_no_generation",
    "test_usage": "target_length_schema_and_final_QA_only",
    "train_usage": "generator_fit_allowed_in_later_cells",
    "val_usage": "generator_selection_and_calibration_allowed_in_later_cells",
    "selection_split": "VAL_only",
    "test_used_for_generator_selection": False,
    "test_used_for_generator_training": False,
    "test_used_for_candidate_acceptance": False,
    "a1_backbone_required": True,
    "portfolio_neutral": True,
    "ddpm_is_optional": True,
    "model_shopping_policy": "forbidden",
    "allowed_later_generators": [
        "A1_temporal_block_bootstrap",
        "A1_support_preserving_replay",
        "GaussianCopula_continuous",
        "CTGAN_tabular_optional",
        "TVAE_tabular_optional",
        "DDPM_conditional_sequence",
        "TimeGAN_sequence",
        "Markov_SemiMarkov_ordinal_dwell",
        "Markov_SemiMarkov_counter_state",
        "NegativeBinomial_PoissonGamma",
        "ContextContinuousReplay",
        "ConstantOrQuasiStaticReplay",
        "PowerEpisodeReplayGenerator",
        "LightPatternGenerator",
        "EnvironmentalGenerator",
        "PrecipitationGenerator",
        "S5Specialized",
        "WithingsSummaryGenerator",
        "SnoringGenerator",
        "Semantic_step_progress_generator",
    ],
    "hard_contracts": {
        "no_value_generation_in_cell12c0": True,
        "do_not_mutate_scaffold_outputs": True,
        "each_cont_value_col_must_have_portfolio_eligibility": True,
        "each_cont_value_col_must_have_availability_mask": True,
        "each_cont_value_col_must_have_routing_policy": True,
        "inactive_value_rows_must_remain_nan_in_later_cells": True,
        "generator_selection_must_be_VAL_locked": True,
    },
}

globals()["CELL12C_POLICY"] = CELL12C_POLICY

CFG["cell12c_policy_version"] = CELL12C0_VERSION
CFG["cell12c_selection_split"] = "VAL_only"
CFG["cell12c_test_usage"] = "target_length_schema_and_final_QA_only"
CFG["cell12c_portfolio_neutral"] = True
CFG["cell12c_ddpm_optional"] = True
CFG["cell12c_model_shopping_policy"] = "forbidden"

# ----------------------------------------------------------
# 2) Helper utilities
# ----------------------------------------------------------
def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize(v) for v in obj])
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

def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2)

def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _sha256_file(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _sha256_jsonable(obj):
    payload = json.dumps(_json_sanitize(obj), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _safe_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")

def _finite_1d(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]

def _safe_mean(x):
    x = _finite_1d(x)
    return float(np.mean(x)) if x.size else np.nan

def _safe_std(x):
    x = _finite_1d(x)
    return float(np.std(x)) if x.size > 1 else np.nan

def _safe_quantile(x, q):
    x = _finite_1d(x)
    return float(np.quantile(x, q)) if x.size else np.nan

def _nonzero_rate(x):
    x = _finite_1d(x)
    return float(np.mean(x > 0)) if x.size else np.nan

def _transition_rate(x):
    x = _finite_1d(x)
    if x.size < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(x)) > 1e-9))

def _lag1_autocorr(x):
    x = _finite_1d(x)
    if x.size < 3:
        return np.nan
    a = x[:-1]
    b = x[1:]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def _run_lengths_from_bool(mask_bool):
    x = np.asarray(mask_bool, dtype=np.int8).reshape(-1)
    runs0, runs1 = [], []
    if x.size == 0:
        return runs0, runs1
    cur = int(x[0])
    L = 1
    for v in x[1:]:
        v = int(v)
        if v == cur:
            L += 1
        else:
            (runs1 if cur == 1 else runs0).append(int(L))
            cur = v
            L = 1
    (runs1 if cur == 1 else runs0).append(int(L))
    return runs0, runs1

def _tier_of_protocol_col(col):
    c = str(col)
    if c.startswith("router__"):
        return "router"
    if c.startswith("ota__"):
        return "ota"
    if c.startswith("zigbee__"):
        return "zigbee"
    if c.startswith("zwave__"):
        return "zwave"
    return None

def _infer_iot_entity(col):
    if str(col) in CANON_ENTITY_MAP:
        return str(CANON_ENTITY_MAP[str(col)])
    s = str(col).lower()
    if s.startswith("iot__"):
        s = s[len("iot__"):]
    parts = s.split("__")
    return parts[0] if parts else "unknown"

def _col_family(col):
    return str(FAMILY_MAP.get(col, "unknown"))

def _countlike_iot_col(col):
    s = str(col).lower()
    return any(k in s for k in [
        "count", "coffees", "cups", "total_cleaning",
        "cleaning_time", "cleaning_area", "consumption",
        "events_total", "events_entity_unique",
    ])

def _ordinal_like_iot_col(col):
    s = str(col).lower()
    return any(k in s for k in [
        "linkquality", "rssi", "lqi", "signal_strength",
        "signal_level", "battery", "voltage", "programme_progress",
        "brightness", "color_temp", "volume_level",
    ])

def _strict_value_mask_for_col(col):
    if col not in VALUE_AVAIL_SYN.columns:
        raise RuntimeError(f"[Cell12.c.0] VALUE_AVAIL_SYN missing value column: {col}")
    m = pd.to_numeric(VALUE_AVAIL_SYN[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if m.shape[0] != N_TE:
        raise RuntimeError(f"[Cell12.c.0] VALUE_AVAIL_SYN[{col}] length mismatch: {m.shape[0]} vs N_TE={N_TE}")
    return (m > 0.5)

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

def _wasserstein_1d_fast(a, b, max_points=5000):
    a = _finite_1d(a)
    b = _finite_1d(b)
    if a.size == 0 or b.size == 0:
        return np.nan
    m = int(min(max_points, max(32, min(a.size, b.size))))
    qs = np.linspace(0.0, 1.0, m)
    aq = np.quantile(a, qs)
    bq = np.quantile(b, qs)
    return float(np.mean(np.abs(aq - bq)))

def _auc_univariate_proxy(real_vals, syn_vals):
    real = _finite_1d(real_vals)
    syn = _finite_1d(syn_vals)
    if real.size < 8 or syn.size < 8:
        return np.nan
    y = np.r_[np.ones(real.size, dtype=np.int8), np.zeros(syn.size, dtype=np.int8)]
    score = np.r_[real, syn].astype(np.float64)
    try:
        auc = float(roc_auc_score(y, score))
        return float(max(auc, 1.0 - auc))
    except Exception:
        return np.nan

def _c2st_auc_lr(real_df, syn_df, cols, max_rows=50000, seed=1337):
    cols = [c for c in cols if c in real_df.columns and c in syn_df.columns]
    if len(cols) == 0:
        return np.nan, {"reason": "no_common_cols"}

    R = real_df[cols].apply(pd.to_numeric, errors="coerce")
    S = syn_df[cols].apply(pd.to_numeric, errors="coerce")

    R = R.replace([np.inf, -np.inf], np.nan)
    S = S.replace([np.inf, -np.inf], np.nan)

    med = R.median(axis=0, skipna=True).fillna(0.0)
    R = R.fillna(med)
    S = S.fillna(med)

    n = min(len(R), len(S), int(max_rows))
    if n < 50:
        return np.nan, {"reason": "too_few_rows", "n": int(n)}

    rrng = np.random.default_rng(int(seed))
    ri = rrng.choice(len(R), size=n, replace=False) if len(R) > n else np.arange(len(R))
    si = rrng.choice(len(S), size=n, replace=False) if len(S) > n else np.arange(len(S))

    X = np.vstack([
        R.iloc[ri].to_numpy(dtype=np.float32),
        S.iloc[si].to_numpy(dtype=np.float32),
    ])
    y = np.r_[np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)]

    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="saga",
                penalty="l2",
                C=1.0,
                max_iter=5000,
                n_jobs=1,
                random_state=int(seed),
            ),
        )
        clf.fit(X, y)
        prob = clf.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, prob))
        auc = float(max(auc, 1.0 - auc))
        return auc, {"reason": "ok", "n": int(n), "cols": int(len(cols))}
    except Exception as e:
        return np.nan, {"reason": f"exception:{type(e).__name__}:{e}", "n": int(n), "cols": int(len(cols))}

def _distribution_metric_bundle(real_vals, syn_vals):
    real = _finite_1d(real_vals)
    syn = _finite_1d(syn_vals)

    return {
        "real_n": int(real.size),
        "syn_n": int(syn.size),
        "real_mean": _safe_mean(real),
        "syn_mean": _safe_mean(syn),
        "real_std": _safe_std(real),
        "syn_std": _safe_std(syn),
        "real_q01": _safe_quantile(real, 0.01),
        "syn_q01": _safe_quantile(syn, 0.01),
        "real_q10": _safe_quantile(real, 0.10),
        "syn_q10": _safe_quantile(syn, 0.10),
        "real_q50": _safe_quantile(real, 0.50),
        "syn_q50": _safe_quantile(syn, 0.50),
        "real_q90": _safe_quantile(real, 0.90),
        "syn_q90": _safe_quantile(syn, 0.90),
        "real_q99": _safe_quantile(real, 0.99),
        "syn_q99": _safe_quantile(syn, 0.99),
        "real_nonzero_rate": _nonzero_rate(real),
        "syn_nonzero_rate": _nonzero_rate(syn),
        "ks": _ks_stat_fast(real, syn),
        "wasserstein": _wasserstein_1d_fast(real, syn),
        "auc_univariate_proxy": _auc_univariate_proxy(real, syn),
        "real_lag1": _lag1_autocorr(real),
        "syn_lag1": _lag1_autocorr(syn),
        "real_transition_rate": _transition_rate(real),
        "syn_transition_rate": _transition_rate(syn),
    }

globals()["_cell12c0_finite_1d"] = _finite_1d
globals()["_cell12c0_ks_stat_fast"] = _ks_stat_fast
globals()["_cell12c0_wasserstein_1d_fast"] = _wasserstein_1d_fast
globals()["_cell12c0_auc_univariate_proxy"] = _auc_univariate_proxy
globals()["_cell12c0_c2st_auc_lr"] = _c2st_auc_lr
globals()["_cell12c0_distribution_metric_bundle"] = _distribution_metric_bundle
globals()["_cell12c0_strict_value_mask_for_col"] = _strict_value_mask_for_col
# ----------------------------------------------------------
# 2B) Phase-1 shared continuous-value utilities
# ----------------------------------------------------------

def _cell12c0_role_aware_continuous_targets() -> list:
    """
    Resolve the only columns allowed to enter modular Cell 12.c.

    Contract:
      - exactly the continuous value target family from Cell 12.a/12.b
      - excludes drivers, binary targets, masks, meta, and excluded columns
      - must match CONT_VALUE_COLS after role normalization
      - must be present in TRAIN, VAL, TEST schema and VALUE_AVAIL_SYN
    """
    if "IOT_CONTINUOUS_VALUE_TARGETS" in globals():
        targets = list(globals()["IOT_CONTINUOUS_VALUE_TARGETS"])
    else:
        targets = list(CONT_VALUE_COLS)

    targets = [str(c) for c in targets]

    if len(targets) == 0:
        raise RuntimeError("[Cell12.c.0] Role-aware continuous target resolver returned zero columns.")

    dupes = sorted([c for c, n in Counter(targets).items() if n > 1])
    if dupes:
        raise RuntimeError(f"[Cell12.c.0] Duplicate continuous targets detected: {dupes[:20]}")

    forbidden_sets = {
        "driver": set(map(str, globals().get("IOT_DRIVER_TARGETS", DRV_COLS))),
        "binary": set(map(str, globals().get("IOT_BINARY_TARGETS", BIN_COLS))),
        "mask": set(map(str, CONT_MASK_COLS)),
        "meta": set(map(str, globals().get("IOT_NONMODELED_META_COLS", []))),
        "excluded": set(map(str, globals().get("IOT_NONMODELED_EXCLUDED_COLS", []))),
    }

    violations = {}
    target_set = set(targets)
    for role, cols in forbidden_sets.items():
        overlap = sorted(target_set & cols)
        if overlap:
            violations[role] = overlap[:30]

    if violations:
        raise RuntimeError(
            "[Cell12.c.0] Role-aware continuous targets overlap forbidden role families: "
            f"{violations}"
        )

    expected = set(map(str, CONT_VALUE_COLS))
    if target_set != expected:
        raise RuntimeError(
            "[Cell12.c.0] Role-aware continuous targets do not match CONT_VALUE_COLS. "
            f"missing_from_role={sorted(expected - target_set)[:30]} "
            f"extra_in_role={sorted(target_set - expected)[:30]}"
        )

    schema_missing = {
        "TRAIN": [c for c in targets if c not in df_tr.columns],
        "VAL": [c for c in targets if c not in df_val.columns],
        "TEST": [c for c in targets if c not in df_te.columns],
        "VALUE_AVAIL_SYN": [c for c in targets if c not in VALUE_AVAIL_SYN.columns],
        "routing_policy": [
            c for c in targets
            if c not in set(CONT_ROUTING_POLICY_DF["col"].astype(str).tolist())
        ],
    }
    schema_missing = {k: v for k, v in schema_missing.items() if v}
    if schema_missing:
        raise RuntimeError(
            "[Cell12.c.0] Role-aware continuous target schema validation failed: "
            f"{schema_missing}"
        )

    if len(targets) != 148:
        raise RuntimeError(
            "[Cell12.c.0] Expected exactly 148 continuous value targets for this Phase-1 contract, "
            f"got {len(targets)}."
        )

    return targets


def _cell12c0_observed_only_arrays(
    real_values,
    syn_values,
    *,
    real_observed_mask=None,
    syn_observed_mask=None,
    segment_mask=None,
    require_finite=True,
):
    """
    Standard observed-only evaluator primitive for 12.c.3 and 12.c.6.

    It compares only rows where:
      - real is observed,
      - synthetic is observed,
      - optional segment/capture mask is active,
      - and both values are finite if require_finite=True.

    This function is split-agnostic. Later cells decide whether real_values
    come from VAL or TEST. Cell 12.c.0 only defines the utility.
    """
    r = np.asarray(real_values, dtype=np.float64).reshape(-1)
    s = np.asarray(syn_values, dtype=np.float64).reshape(-1)

    if r.shape[0] != s.shape[0]:
        raise RuntimeError(
            "[Cell12.c.0] observed-only evaluator length mismatch: "
            f"real={r.shape[0]} syn={s.shape[0]}"
        )

    keep = np.ones(r.shape[0], dtype=bool)

    if real_observed_mask is not None:
        m = np.asarray(real_observed_mask).reshape(-1)
        if m.shape[0] != r.shape[0]:
            raise RuntimeError("[Cell12.c.0] real_observed_mask length mismatch.")
        keep &= m.astype(bool)

    if syn_observed_mask is not None:
        m = np.asarray(syn_observed_mask).reshape(-1)
        if m.shape[0] != r.shape[0]:
            raise RuntimeError("[Cell12.c.0] syn_observed_mask length mismatch.")
        keep &= m.astype(bool)

    if segment_mask is not None:
        m = np.asarray(segment_mask).reshape(-1)
        if m.shape[0] != r.shape[0]:
            raise RuntimeError("[Cell12.c.0] segment_mask length mismatch.")
        keep &= m.astype(bool)

    if require_finite:
        keep &= np.isfinite(r) & np.isfinite(s)

    return r[keep], s[keep], keep


def _cell12c0_observed_only_metric_bundle(
    real_values,
    syn_values,
    *,
    real_observed_mask=None,
    syn_observed_mask=None,
    segment_mask=None,
    col=None,
):
    """
    Standard observed-only metric bundle.

    Used by:
      - 12.c.3 VAL-only selector
      - 12.c.6 final TEST QA

    It intentionally returns both distribution metrics and support-aware metrics.
    """
    r_obs, s_obs, keep = _cell12c0_observed_only_arrays(
        real_values,
        syn_values,
        real_observed_mask=real_observed_mask,
        syn_observed_mask=syn_observed_mask,
        segment_mask=segment_mask,
        require_finite=True,
    )

    dist = _distribution_metric_bundle(r_obs, s_obs)
    support = _cell12c0_support_metric_bundle(
        r_obs,
        s_obs,
        integer_like=bool(_countlike_iot_col(col) or _ordinal_like_iot_col(col)) if col is not None else None,
    )

    out = {}
    out.update(dist)
    out.update(support)
    out["observed_intersection_n"] = int(len(r_obs))
    out["observed_intersection_rate"] = float(np.mean(keep)) if len(keep) else np.nan
    out["col"] = str(col) if col is not None else ""

    return out


def _cell12c0_contiguous_segments_from_mask(mask, *, min_len=1):
    """
    Return contiguous active segments from a binary capture/availability mask.

    Output segments are half-open intervals:
      [(start, end), ...]
    where end is exclusive.

    This prevents later replay utilities from crossing capture gaps.
    """
    x = np.asarray(mask).reshape(-1).astype(bool)
    segments = []

    if x.size == 0:
        return segments

    start = None
    for i, v in enumerate(x):
        if v and start is None:
            start = int(i)
        elif (not v) and start is not None:
            if int(i) - start >= int(min_len):
                segments.append((start, int(i)))
            start = None

    if start is not None and x.size - start >= int(min_len):
        segments.append((start, int(x.size)))

    return segments


def _cell12c0_segment_ids_from_mask(mask):
    """
    Assign segment IDs to active runs. Inactive rows receive -1.
    """
    x = np.asarray(mask).reshape(-1).astype(bool)
    seg_id = np.full(x.shape[0], -1, dtype=np.int32)
    for sid, (a, b) in enumerate(_cell12c0_contiguous_segments_from_mask(x)):
        seg_id[a:b] = int(sid)
    return seg_id


def _cell12c0_segment_aware_bool_runs(mask, *, segment_mask=None):
    """
    Compute run lengths without allowing runs to cross inactive capture gaps.

    If segment_mask is supplied, runs are computed independently inside each
    active segment of segment_mask.
    """
    x = np.asarray(mask).reshape(-1).astype(bool)

    if segment_mask is None:
        return _run_lengths_from_bool(x)

    segs = _cell12c0_contiguous_segments_from_mask(segment_mask)
    runs0_all, runs1_all = [], []

    for a, b in segs:
        r0, r1 = _run_lengths_from_bool(x[a:b])
        runs0_all.extend(r0)
        runs1_all.extend(r1)

    return runs0_all, runs1_all


def _cell12c0_segment_aware_sample_starts(segment_mask, block_len, *, rng_obj=None, max_draws=1000):
    """
    Sample a valid block start such that [start, start + block_len) stays
    inside one active segment. Returns None if impossible.
    """
    if rng_obj is None:
        rng_obj = np.random.default_rng(SEED)

    L = int(block_len)
    if L <= 0:
        raise ValueError("[Cell12.c.0] block_len must be positive.")

    candidates = []
    for a, b in _cell12c0_contiguous_segments_from_mask(segment_mask, min_len=L):
        max_start = int(b - L)
        if max_start >= int(a):
            candidates.append((int(a), max_start))

    if not candidates:
        return None

    lengths = np.asarray([(b - a + 1) for a, b in candidates], dtype=np.float64)
    probs = lengths / max(float(lengths.sum()), 1e-12)

    k = int(rng_obj.choice(np.arange(len(candidates)), p=probs))
    a, b = candidates[k]
    return int(rng_obj.integers(a, b + 1))


def _cell12c0_support_metric_bundle(real_values, syn_values, *, integer_like=None, round_decimals=6):
    """
    Support-aware metrics for discrete, ordinal, count-like, and low-cardinality
    continuous columns.

    Metrics:
      - support_jaccard
      - real_unique_n / syn_unique_n
      - unique_ratio
      - support_missing_n / support_extra_n
      - exact_support_match
      - integer_support_match
    """
    r = _finite_1d(real_values)
    s = _finite_1d(syn_values)

    if r.size == 0 or s.size == 0:
        return {
            "support_jaccard": np.nan,
            "real_unique_n": int(np.unique(r).size),
            "syn_unique_n": int(np.unique(s).size),
            "unique_ratio": np.nan,
            "support_missing_n": np.nan,
            "support_extra_n": np.nan,
            "exact_support_match": False,
            "integer_support_match": False,
        }

    if integer_like is None:
        integer_like = bool(
            np.all(np.isclose(r, np.round(r), atol=1e-6))
            and np.all(np.isclose(s, np.round(s), atol=1e-6))
        )

    if bool(integer_like):
        r_sup = set(np.round(r).astype(np.int64).tolist())
        s_sup = set(np.round(s).astype(np.int64).tolist())
    else:
        r_sup = set(np.round(r.astype(np.float64), int(round_decimals)).tolist())
        s_sup = set(np.round(s.astype(np.float64), int(round_decimals)).tolist())

    union = r_sup | s_sup
    inter = r_sup & s_sup

    real_unique_n = int(len(r_sup))
    syn_unique_n = int(len(s_sup))
    unique_ratio = float(syn_unique_n / max(real_unique_n, 1))

    exact_support_match = bool(r_sup == s_sup)

    integer_support_match = False
    if bool(integer_like):
        integer_support_match = exact_support_match

    return {
        "support_jaccard": float(len(inter) / max(len(union), 1)),
        "real_unique_n": real_unique_n,
        "syn_unique_n": syn_unique_n,
        "unique_ratio": unique_ratio,
        "support_missing_n": int(len(r_sup - s_sup)),
        "support_extra_n": int(len(s_sup - r_sup)),
        "exact_support_match": exact_support_match,
        "integer_support_match": bool(integer_support_match),
    }


CELL12C_PUBLICATION_SAFETY_THRESHOLDS = {
    "fatal": {
        "role_contract_violations": 0,
        "test_used_for_selection": 0,
        "missing_value_contract_rows": 0,
        "missing_a1_baseline_cols": 0,
        "invalid_domain_cols": 0,
    },
    "warning": {
        "support_jaccard_min": 0.60,
        "unique_ratio_min": 0.50,
        "unique_ratio_max": 2.00,
        "auc_univariate_proxy_max": 0.70,
        "ks_max": 0.50,
        "availability_rate_abs_err_max": 0.05,
    },
}


def _cell12c0_publication_safety_gate(
    rows,
    *,
    context,
    thresholds=None,
    raise_on_fatal=True,
):
    """
    Central publication-safety gate.

    rows: list of dicts with:
      - check
      - value
      - threshold
      - severity: fatal/warning/pass
      - passed: bool
      - reason

    Returns a DataFrame and optionally raises if any fatal check fails.
    """
    if thresholds is None:
        thresholds = CELL12C_PUBLICATION_SAFETY_THRESHOLDS

    df = pd.DataFrame(rows)

    if df.empty:
        df = pd.DataFrame(
            columns=["context", "check", "value", "threshold", "severity", "passed", "reason"]
        )

    if "context" not in df.columns:
        df["context"] = str(context)

    if "passed" not in df.columns:
        df["passed"] = True

    if "severity" not in df.columns:
        df["severity"] = "pass"

    df["context"] = df["context"].fillna(str(context)).astype(str)
    df["severity"] = df["severity"].astype(str)
    df["passed"] = df["passed"].astype(bool)

    fatal_bad = df.loc[
        df["severity"].eq("fatal") & (~df["passed"])
    ].copy()

    if raise_on_fatal and len(fatal_bad):
        raise RuntimeError(
            f"[Cell12.c.0] Publication-safety gate failed for {context}. "
            f"Fatal checks={fatal_bad[['check', 'value', 'threshold', 'reason']].to_dict('records')[:10]}"
        )

    return df


CELL12C_CONT_VALUE_TARGETS = _cell12c0_role_aware_continuous_targets()
CONT_VALUE_COLS_ROLE_AWARE = list(CELL12C_CONT_VALUE_TARGETS)

globals()["CELL12C_CONT_VALUE_TARGETS"] = CELL12C_CONT_VALUE_TARGETS
globals()["CONT_VALUE_COLS_ROLE_AWARE"] = CONT_VALUE_COLS_ROLE_AWARE

globals()["_cell12c0_role_aware_continuous_targets"] = _cell12c0_role_aware_continuous_targets
globals()["_cell12c0_observed_only_arrays"] = _cell12c0_observed_only_arrays
globals()["_cell12c0_observed_only_metric_bundle"] = _cell12c0_observed_only_metric_bundle
globals()["_cell12c0_contiguous_segments_from_mask"] = _cell12c0_contiguous_segments_from_mask
globals()["_cell12c0_segment_ids_from_mask"] = _cell12c0_segment_ids_from_mask
globals()["_cell12c0_segment_aware_bool_runs"] = _cell12c0_segment_aware_bool_runs
globals()["_cell12c0_segment_aware_sample_starts"] = _cell12c0_segment_aware_sample_starts
globals()["_cell12c0_support_metric_bundle"] = _cell12c0_support_metric_bundle
globals()["CELL12C_PUBLICATION_SAFETY_THRESHOLDS"] = CELL12C_PUBLICATION_SAFETY_THRESHOLDS
globals()["_cell12c0_publication_safety_gate"] = _cell12c0_publication_safety_gate

log(
    "[Cell12.c.0] Phase-1 shared utilities registered | "
    f"role_aware_continuous_targets={len(CELL12C_CONT_VALUE_TARGETS)} | "
    "observed_only_evaluator=True | "
    "segment_aware_runs=True | "
    "support_metrics=True | "
    "publication_safety_gate=True"
)
# ----------------------------------------------------------
# 3) Stale-global contamination guard
# ----------------------------------------------------------
# These are old monolithic/DDPM-centric 12.b/12.c artifacts. Their presence
# after v27.1.1 scaffold can silently contaminate modular 12.c.
_STALE_DDPM_GLOBALS = [
    "syn_ddpm_df",
    "syn_ddpm_val_df",
    "ddpm_model",
    "ddpm",
    "scaler",
    "ddpm_train_df",
    "ddpm_val_df",
    "ddpm_test_real_df",
    "M_tr",
    "M_val",
    "M_te",
    "train_medians",
    "ddpm_train_imp",
    "ddpm_val_imp",
    "Xdd_tr",
    "Xdd_val",
    "Xdd_te_seed",
    "HIST_TR",
    "HIST_VAL",
    "DRV_TR_ARR",
    "DRV_VAL_ARR",
    "DRV_TE_ARR",
    "BIN_TR_ARR",
    "BIN_VAL_ARR",
    "BIN_TE_ARR",
    "REGIME_TR_ARR",
    "REGIME_VAL_ARR",
    "REGIME_TE_ARR",
    "SEG_LEN",
    "STRIDE",
    "TR_SEG_STARTS",
    "train_ds",
    "train_loader",
    "opt",
    "DDPM_COLUMN_CALIBRATION",
    "DDPM_COLUMN_CALIBRATION_DF",
    "DDPM_COLUMN_CALIBRATION_EXPORT",
]

_PRESENT_STALE_GLOBALS = [k for k in _STALE_DDPM_GLOBALS if k in globals()]

if _PRESENT_STALE_GLOBALS:
    raise RuntimeError(
        "[Cell12.c.0] Stale monolithic/DDPM-centric globals are still present. "
        "Run the POST-12b cleanup cell before Cell 12.c.0. "
        f"Present stale globals: {_PRESENT_STALE_GLOBALS}"
    )

log("[Cell12.c.0] Stale DDPM/global contamination guard passed.")

# ----------------------------------------------------------
# 4) Validate scaffold shapes and schemas
# ----------------------------------------------------------
CONT_VALUE_COLS = list(CELL12C_CONT_VALUE_TARGETS)
BIN_COLS = list(BIN_COLS)
DRV_COLS = list(DRV_COLS)
CONT_MASK_COLS = list(CONT_MASK_COLS)
DDPM_COLS = list(DDPM_COLS)
NON_DDPM_COLS = list(NON_DDPM_COLS)
ENTITY_LIST = list(ENTITY_LIST)

if len(CONT_VALUE_COLS) == 0:
    raise RuntimeError("[Cell12.c.0] CONT_VALUE_COLS is empty.")

if set(DDPM_COLS) & set(NON_DDPM_COLS):
    raise RuntimeError("[Cell12.c.0] DDPM_COLS and NON_DDPM_COLS overlap.")

if set(DDPM_COLS) | set(NON_DDPM_COLS) != set(CONT_VALUE_COLS):
    missing_from_partition = sorted(set(CONT_VALUE_COLS) - (set(DDPM_COLS) | set(NON_DDPM_COLS)))
    extra_in_partition = sorted((set(DDPM_COLS) | set(NON_DDPM_COLS)) - set(CONT_VALUE_COLS))
    raise RuntimeError(
        "[Cell12.c.0] DDPM/NON_DDPM partition does not match CONT_VALUE_COLS. "
        f"missing={missing_from_partition[:20]} extra={extra_in_partition[:20]}"
    )

for _name, _df, _cols, _n in [
    ("IOT_SYN_MASKS", IOT_SYN_MASKS, CONT_MASK_COLS, N_TE),
    ("IOT_SYN_BINARY", IOT_SYN_BINARY, BIN_COLS, N_TE),
    ("VALUE_AVAIL_SYN", VALUE_AVAIL_SYN, CONT_VALUE_COLS, N_TE),
]:
    if not isinstance(_df, pd.DataFrame):
        raise RuntimeError(f"[Cell12.c.0] {_name} must be a DataFrame.")
    if len(_df) != _n:
        raise RuntimeError(f"[Cell12.c.0] {_name} row mismatch: len={len(_df)} expected={_n}")
    miss = [c for c in _cols if c not in _df.columns]
    if miss:
        raise RuntimeError(f"[Cell12.c.0] {_name} missing required columns: {miss[:30]}")

# ----------------------------------------------------------
# Normalize entity-regime and entity-driver scaffold artifacts
# ----------------------------------------------------------
def _normalize_entity_frame(obj, *, name: str, artifact_path: str, entity_list: list, n_rows: int) -> pd.DataFrame:
    """
    Accepts:
      - DataFrame with entity columns
      - dict/entity -> 1D array or Series
      - saved parquet artifact from Cell 12.b

    Returns:
      DataFrame indexed like df_te, columns=entity_list.
    """
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()

    elif isinstance(obj, dict):
        out = pd.DataFrame(index=df_te.index)
        for e in entity_list:
            if e not in obj:
                continue
            arr = np.asarray(obj[e]).reshape(-1)
            if arr.shape[0] != n_rows:
                raise RuntimeError(
                    f"[Cell12.c.0] {name}[{e}] length mismatch in dict: "
                    f"{arr.shape[0]} vs expected {n_rows}"
                )
            out[e] = arr

    elif os.path.exists(artifact_path):
        out = pd.read_parquet(artifact_path)

    else:
        raise RuntimeError(
            f"[Cell12.c.0] {name} must be a DataFrame or dict, or artifact must exist: "
            f"{artifact_path}. Got type={type(obj).__name__}"
        )

    if len(out) != n_rows:
        raise RuntimeError(f"[Cell12.c.0] {name} row mismatch: {len(out)} vs {n_rows}")

    missing_entities = [e for e in entity_list if e not in out.columns]
    if missing_entities:
        raise RuntimeError(
            f"[Cell12.c.0] {name} missing entities: {missing_entities[:20]} "
            f"(n_missing={len(missing_entities)})"
        )

    out = out.loc[:, entity_list].copy()
    out.index = df_te.index

    return out


_ENTITY_REGIME_ARTIFACT_PATH = os.path.join(OUT_SYN, "IOT_ENTITY_REGIME_SYN.parquet")
_ENTITY_DRIVER_ARTIFACT_PATH = os.path.join(OUT_SYN, "IOT_ENTITY_DRIVER_ACTIVITY_SYN.parquet")

ENTITY_REGIME_SYN = _normalize_entity_frame(
    ENTITY_REGIME_SYN,
    name="ENTITY_REGIME_SYN",
    artifact_path=_ENTITY_REGIME_ARTIFACT_PATH,
    entity_list=ENTITY_LIST,
    n_rows=N_TE,
)

ENTITY_DRIVER_ACTIVITY_SYN = _normalize_entity_frame(
    ENTITY_DRIVER_ACTIVITY_SYN,
    name="ENTITY_DRIVER_ACTIVITY_SYN",
    artifact_path=_ENTITY_DRIVER_ARTIFACT_PATH,
    entity_list=ENTITY_LIST,
    n_rows=N_TE,
)

globals()["ENTITY_REGIME_SYN"] = ENTITY_REGIME_SYN
globals()["ENTITY_DRIVER_ACTIVITY_SYN"] = ENTITY_DRIVER_ACTIVITY_SYN

missing_entities_regime = [e for e in ENTITY_LIST if e not in ENTITY_REGIME_SYN.columns]
if missing_entities_regime:
    raise RuntimeError(f"[Cell12.c.0] ENTITY_REGIME_SYN missing entities: {missing_entities_regime[:20]}")

missing_entities_drv = [e for e in ENTITY_LIST if e not in ENTITY_DRIVER_ACTIVITY_SYN.columns]
if missing_entities_drv:
    raise RuntimeError(f"[Cell12.c.0] ENTITY_DRIVER_ACTIVITY_SYN missing entities: {missing_entities_drv[:20]}")

log(
    "[Cell12.c.0] Normalized entity scaffold frames | "
    f"ENTITY_REGIME_SYN={ENTITY_REGIME_SYN.shape} | "
    f"ENTITY_DRIVER_ACTIVITY_SYN={ENTITY_DRIVER_ACTIVITY_SYN.shape}"
)


# Binary/mask/routing sanity
_bad_mask_cols = []
for c in CONT_MASK_COLS:
    x = pd.to_numeric(IOT_SYN_MASKS[c], errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
    if not np.isin(x, [0.0, 1.0]).all():
        _bad_mask_cols.append(c)

_bad_avail_cols = []
for c in CONT_VALUE_COLS:
    x = pd.to_numeric(VALUE_AVAIL_SYN[c], errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
    if not np.isin(x, [0.0, 1.0]).all():
        _bad_avail_cols.append(c)

_bad_binary_cols = []
for c in BIN_COLS:
    x = pd.to_numeric(IOT_SYN_BINARY[c], errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
    if not np.isin(x, [0.0, 1.0]).all():
        _bad_binary_cols.append(c)

_bad_regime_cols = []
for e in ENTITY_LIST:
    x = pd.to_numeric(ENTITY_REGIME_SYN[e], errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
    if not np.isin(x, [0.0, 1.0, 2.0, 3.0]).all():
        _bad_regime_cols.append(e)

if _bad_mask_cols or _bad_avail_cols or _bad_binary_cols or _bad_regime_cols:
    raise RuntimeError(
        "[Cell12.c.0] Scaffold binary-domain validation failed: "
        f"bad_masks={_bad_mask_cols[:20]} "
        f"bad_availability={_bad_avail_cols[:20]} "
        f"bad_binary={_bad_binary_cols[:20]} "
        f"bad_regime={_bad_regime_cols[:20]}"
    )

log(
    "[Cell12.c.0] Scaffold schema/domain validation passed | "
    f"value_cols={len(CONT_VALUE_COLS)} | ddpm_optional={len(DDPM_COLS)} | "
    f"non_ddpm={len(NON_DDPM_COLS)} | binary={len(BIN_COLS)} | entities={len(ENTITY_LIST)}"
)

# ----------------------------------------------------------
# 5) Validate / load Cell 12.b manifest and portfolio eligibility
# ----------------------------------------------------------
cell12b_manifest_path = os.path.join(ARTDIR, "cell12b_iot_scaffold_manifest_v27_1.json")
cell12b_summary_path = os.path.join(REPORT_DIR, "cell12b_v27_1_scaffold_summary.json")
cell12b_portfolio_csv = os.path.join(REPORT_DIR, "cell12b_v27_1_iot_value_portfolio_eligibility.csv")

CELL12B_MANIFEST = {}
CELL12B_SUMMARY = {}

if os.path.exists(cell12b_manifest_path):
    CELL12B_MANIFEST = _read_json(cell12b_manifest_path)
else:
    log(f"[Cell12.c.0][WARN] Cell 12.b manifest not found at canonical path: {cell12b_manifest_path}")

if os.path.exists(cell12b_summary_path):
    CELL12B_SUMMARY = _read_json(cell12b_summary_path)
else:
    log(f"[Cell12.c.0][WARN] Cell 12.b summary not found at canonical path: {cell12b_summary_path}")

if (
    "IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF" in globals()
    and isinstance(globals()["IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF"], pd.DataFrame)
):
    _port_df_raw = globals()["IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF"].copy()
elif os.path.exists(cell12b_portfolio_csv):
    _port_df_raw = pd.read_csv(cell12b_portfolio_csv)
else:
    raise RuntimeError(
        "[Cell12.c.0] Missing IoT value portfolio eligibility. Expected global "
        "IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF or CSV: "
        f"{cell12b_portfolio_csv}"
    )

if not isinstance(_port_df_raw, pd.DataFrame) or len(_port_df_raw) == 0:
    raise RuntimeError("[Cell12.c.0] IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF is empty or invalid.")

# ----------------------------------------------------------
# Normalize Cell 12.b portfolio eligibility to long format.
#
# Cell 12.b v27.1.1 writes one row per value column with:
#   eligible_generators = list-like / stringified-list of candidate names
#
# Older downstream code expected one row per (col, generator).
# This block converts both formats into the canonical long schema:
#   col | generator | eligible | plus metadata columns
# ----------------------------------------------------------
_port_df_raw = _port_df_raw.copy()

if "col" not in _port_df_raw.columns:
    for cand in ["column", "value_col", "iot_col"]:
        if cand in _port_df_raw.columns:
            _port_df_raw = _port_df_raw.rename(columns={cand: "col"})
            break

if "col" not in _port_df_raw.columns:
    raise RuntimeError(
        "[Cell12.c.0] Portfolio eligibility is missing a column identifier. "
        f"Available columns={list(_port_df_raw.columns)}"
    )

_port_df_raw["col"] = _port_df_raw["col"].astype(str)

def _parse_eligible_generators_cell12c0(x):
    """
    Robustly parse Cell 12.b v27.1.1 eligible_generators values.

    Accepts:
      - Python list/tuple/set
      - JSON list string
      - Python repr list string
      - pipe/comma/semicolon-separated string
      - scalar generator name
    """
    if x is None:
        return []

    if isinstance(x, (list, tuple, set)):
        vals = list(x)
        return [str(v).strip() for v in vals if str(v).strip()]

    if isinstance(x, float) and np.isnan(x):
        return []

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    # JSON/Python-list style, e.g. "['A1', 'DDPM']" or '["A1","DDPM"]'
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                return [str(v).strip() for v in obj if str(v).strip()]
        except Exception:
            pass

    # Delimited fallback.
    for sep in ["|", ";", ","]:
        if sep in s:
            return [p.strip().strip("'\"") for p in s.split(sep) if p.strip().strip("'\"")]

    return [s.strip().strip("'\"")]

# Case A: already long format with generator column.
if "generator" in _port_df_raw.columns:
    _port_df = _port_df_raw.copy()

elif any(c in _port_df_raw.columns for c in ["candidate_generator", "generator_name", "eligible_generator"]):
    _port_df = _port_df_raw.copy()
    for cand in ["candidate_generator", "generator_name", "eligible_generator"]:
        if cand in _port_df.columns:
            _port_df = _port_df.rename(columns={cand: "generator"})
            break

# Case B: Cell 12.b v27.1.1 wide format with eligible_generators.
elif "eligible_generators" in _port_df_raw.columns:
    long_rows = []

    metadata_cols = [
        c for c in _port_df_raw.columns
        if c not in {"eligible_generators", "generator", "candidate_generator", "generator_name", "eligible_generator"}
    ]

    for _, r in _port_df_raw.iterrows():
        col = str(r["col"])
        gens = _parse_eligible_generators_cell12c0(r.get("eligible_generators", []))

        # Remove empty/duplicate generator names while preserving order.
        seen = set()
        gens_clean = []
        for g in gens:
            g = str(g).strip()
            if not g or g in seen:
                continue
            seen.add(g)
            gens_clean.append(g)

        for g in gens_clean:
            row = {mc: r[mc] for mc in metadata_cols if mc in r.index}
            row["col"] = col
            row["generator"] = g
            row["eligible"] = True
            long_rows.append(row)

    if not long_rows:
        raise RuntimeError(
            "[Cell12.c.0] Portfolio eligibility has eligible_generators column, "
            "but no generator entries could be parsed."
        )

    _port_df = pd.DataFrame(long_rows)

else:
    raise RuntimeError(
        "[Cell12.c.0] Portfolio eligibility has neither a generator column nor eligible_generators. "
        f"Available columns={list(_port_df_raw.columns)}"
    )

if "eligible" not in _port_df.columns:
    _port_df["eligible"] = True

_required_port_cols = {"col", "generator", "eligible"}
_missing_port_cols = sorted(_required_port_cols - set(_port_df.columns))
if _missing_port_cols:
    raise RuntimeError(
        f"[Cell12.c.0] Portfolio eligibility missing required columns after normalization: {_missing_port_cols}. "
        f"Available columns={list(_port_df.columns)}"
    )

_port_df["col"] = _port_df["col"].astype(str)
_port_df["generator"] = _port_df["generator"].astype(str).str.strip()
_port_df["eligible"] = _port_df["eligible"].astype(bool)

# Drop empty generator names and duplicate (col, generator) rows.
_port_df = _port_df.loc[_port_df["generator"].astype(str).str.len() > 0].copy()
_port_df = _port_df.drop_duplicates(subset=["col", "generator"], keep="last").reset_index(drop=True)

missing_port_value_cols = sorted(set(CONT_VALUE_COLS) - set(_port_df["col"].tolist()))
extra_port_value_cols = sorted(set(_port_df["col"].tolist()) - set(CONT_VALUE_COLS))

if missing_port_value_cols:
    raise RuntimeError(
        "[Cell12.c.0] Portfolio eligibility does not cover all continuous value columns. "
        f"missing={missing_port_value_cols[:30]} n_missing={len(missing_port_value_cols)}"
    )

if extra_port_value_cols:
    log(
        "[Cell12.c.0][WARN] Portfolio eligibility contains extra columns not in CONT_VALUE_COLS; "
        f"extras_preview={extra_port_value_cols[:20]} n_extra={len(extra_port_value_cols)}"
    )

# A1 must be eligible for every value column.
_a1_names = {"A1_temporal_block_bootstrap", "A1_support_preserving_replay"}
_a1_by_col = (
    _port_df.loc[_port_df["eligible"] & _port_df["generator"].isin(_a1_names)]
    .groupby("col")["generator"]
    .apply(lambda x: sorted(set(x.astype(str))))
    .to_dict()
)

missing_a1_cols = [c for c in CONT_VALUE_COLS if c not in _a1_by_col]
if missing_a1_cols:
    raise RuntimeError(
        "[Cell12.c.0] A1 baseline/support replay is not eligible for all value columns. "
        f"missing={missing_a1_cols[:30]} n_missing={len(missing_a1_cols)}"
    )

portfolio_generator_counts = (
    _port_df.loc[_port_df["eligible"], "generator"]
    .value_counts()
    .sort_index()
    .to_dict()
)

# Preserve both:
#   1) wide raw Cell 12.b table for traceability
#   2) long normalized table for modular Cell 12.c generator selection
IOT_VALUE_PORTFOLIO_ELIGIBILITY_RAW_DF = _port_df_raw.copy()
IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF = _port_df.copy()

globals()["IOT_VALUE_PORTFOLIO_ELIGIBILITY_RAW_DF"] = IOT_VALUE_PORTFOLIO_ELIGIBILITY_RAW_DF
globals()["IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF"] = IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF

portfolio_long_csv = os.path.join(REPORT_DIR, "cell12c0_iot_value_portfolio_eligibility_long.csv")
IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF.to_csv(portfolio_long_csv, index=False)
globals()["IOT_VALUE_PORTFOLIO_ELIGIBILITY_LONG_CSV"] = portfolio_long_csv

log(
    "[Cell12.c.0] Portfolio eligibility validated | "
    f"raw_rows={len(IOT_VALUE_PORTFOLIO_ELIGIBILITY_RAW_DF)} | "
    f"long_rows={len(IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF)} | "
    f"value_cols={len(CONT_VALUE_COLS)} | "
    f"eligible_generator_counts={portfolio_generator_counts} | "
    f"long_csv={portfolio_long_csv}"
)

# ----------------------------------------------------------
# 6) Validate routing policy
# ----------------------------------------------------------
_policy_df = CONT_ROUTING_POLICY_DF.copy()

if "col" not in _policy_df.columns:
    raise RuntimeError("[Cell12.c.0] CONT_ROUTING_POLICY_DF missing 'col'.")

_required_policy_cols = {
    "col",
    "family",
    "policy_bucket",
    "policy_generator_family",
    "expected_stage12b_route",
    "expected_stage12c_kind",
}
_missing_policy_cols = sorted(_required_policy_cols - set(_policy_df.columns))
if _missing_policy_cols:
    raise RuntimeError(f"[Cell12.c.0] CONT_ROUTING_POLICY_DF missing required columns: {_missing_policy_cols}")

_policy_df["col"] = _policy_df["col"].astype(str)
_policy_df = _policy_df.drop_duplicates(subset=["col"], keep="last").reset_index(drop=True)

missing_policy_value_cols = sorted(set(CONT_VALUE_COLS) - set(_policy_df["col"].tolist()))
if missing_policy_value_cols:
    raise RuntimeError(
        "[Cell12.c.0] Routing policy missing continuous value columns: "
        f"{missing_policy_value_cols[:30]} n_missing={len(missing_policy_value_cols)}"
    )

_policy_ddpm = sorted(
    _policy_df.loc[
        _policy_df["expected_stage12b_route"].astype(str).str.strip().eq("ddpm"),
        "col"
    ].tolist()
)

if set(_policy_ddpm) != set(DDPM_COLS):
    raise RuntimeError(
        "[Cell12.c.0] Routing policy DDPM route does not match current DDPM_COLS. "
        f"policy_only={sorted(set(_policy_ddpm) - set(DDPM_COLS))[:20]} "
        f"ddpm_only={sorted(set(DDPM_COLS) - set(_policy_ddpm))[:20]}"
    )

routing_bucket_counts = _policy_df["policy_bucket"].astype(str).value_counts().sort_index().to_dict()
routing_kind_counts = _policy_df["expected_stage12c_kind"].astype(str).value_counts().sort_index().to_dict()

log(
    "[Cell12.c.0] Routing policy validated | "
    f"bucket_counts={routing_bucket_counts} | "
    f"stage12c_kind_counts={routing_kind_counts}"
)

# ----------------------------------------------------------
# ----------------------------------------------------------
# 7) Register Cell 11 protocol context path only
# ----------------------------------------------------------
# Revised architecture:
#   - 12.c.0 must not consume TEST values or TEST-length protocol matrices.
#   - This cell only records where the synthetic protocol context can be loaded later.
#   - 12.c.4 may load A2_PROTOCOL_FINAL after VAL selection is locked, strictly for
#     TEST-length materialization from locked generator choices.
#   - 12.c.6 performs final TEST QA.

_protocol_candidate_paths = [
    os.path.join(OUT_SYN, "A2_PROTOCOL_FINAL.parquet"),
    os.path.join(OUT_SYN, "A2_PROTOCOL_TEST.parquet"),
    os.path.join(OUT_SYN, "A2_HYBRID_PROTOCOL_ONLY.parquet"),
    os.path.join(OUT_SYN, "A2.parquet"),
]

A2_PROTOCOL_PATH = None
for _p in _protocol_candidate_paths:
    if os.path.exists(_p):
        A2_PROTOCOL_PATH = _p
        break

if A2_PROTOCOL_PATH is None:
    raise RuntimeError(
        "[Cell12.c.0] Could not find A2 protocol artifact path. Checked: "
        f"{_protocol_candidate_paths}"
    )

# Do NOT read the protocol parquet here. Reading/materializing TEST-length
# context belongs in 12.c.4 after VAL-locked choices exist.
PROTO_CONTEXT_COLS = []
IOT_PROTOCOL_CONTEXT_TE = None

globals()["A2_PROTOCOL_PATH"] = A2_PROTOCOL_PATH
globals()["PROTO_CONTEXT_COLS"] = PROTO_CONTEXT_COLS
globals()["IOT_PROTOCOL_CONTEXT_TE"] = IOT_PROTOCOL_CONTEXT_TE

log(
    "[Cell12.c.0] Protocol context path registered without loading TEST-length matrix | "
    f"path={A2_PROTOCOL_PATH}"
)

# ----------------------------------------------------------
# 8) Build column-level 12.c contract table
# ----------------------------------------------------------
_contract_rows = []

_port_by_col = defaultdict(list)
for _, r in IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF.loc[IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF["eligible"]].iterrows():
    _port_by_col[str(r["col"])].append(str(r["generator"]))

_policy_by_col = _policy_df.set_index("col").to_dict(orient="index")

for c in CONT_VALUE_COLS:
    fam = _col_family(c)
    entity = _infer_iot_entity(c)
    avail = _strict_value_mask_for_col(c)
    avail_rate = float(avail.mean())

    tr_present = float(pd.to_numeric(df_tr[c], errors="coerce").notna().mean()) if c in df_tr.columns else np.nan
    val_present = float(pd.to_numeric(df_val[c], errors="coerce").notna().mean()) if c in df_val.columns else np.nan

    p = _policy_by_col.get(c, {})
    eligible_gens = sorted(set(_port_by_col.get(c, [])))

    if not eligible_gens:
        raise RuntimeError(f"[Cell12.c.0] No eligible generators for value column: {c}")

    _contract_rows.append({
        "col": c,
        "entity": entity,
        "family": fam,
        "policy_bucket": str(p.get("policy_bucket", "")),
        "policy_generator_family": str(p.get("policy_generator_family", "")),
        "expected_stage12b_route": str(p.get("expected_stage12b_route", "")),
        "expected_stage12c_kind": str(p.get("expected_stage12c_kind", "")),
        "is_ddpm_optional_candidate": bool(c in DDPM_COLS),
        "is_non_ddpm": bool(c in NON_DDPM_COLS),
        "is_countlike": bool(_countlike_iot_col(c)),
        "is_ordinal_like": bool(_ordinal_like_iot_col(c)),
        "availability_mode": str(VALUE_AVAIL_MODE.get(c, "")),
        "availability_rate_syn": avail_rate,
        "availability_rate_train": tr_present,
        "availability_rate_val": val_present,
        "availability_rate_test_observed_for_QA_only": np.nan,
        "test_observed_availability_deferred_to": "Cell 12.c.6 final QA only",
        "eligible_generators": "|".join(eligible_gens),
        "eligible_generator_count": int(len(eligible_gens)),
        "has_a1_baseline": bool(any(g in _a1_names for g in eligible_gens)),
        "test_used_for_selection": False,
    })

CELL12C_VALUE_CONTRACT_DF = pd.DataFrame(_contract_rows)

if len(CELL12C_VALUE_CONTRACT_DF) != len(CONT_VALUE_COLS):
    raise RuntimeError("[Cell12.c.0] Value contract row count mismatch.")

if not bool(CELL12C_VALUE_CONTRACT_DF["has_a1_baseline"].all()):
    bad = CELL12C_VALUE_CONTRACT_DF.loc[~CELL12C_VALUE_CONTRACT_DF["has_a1_baseline"], "col"].head(20).tolist()
    raise RuntimeError(f"[Cell12.c.0] Some columns lack A1 baseline eligibility: {bad}")

CELL12C_VALUE_CONTRACT_CSV = os.path.join(REPORT_DIR, "cell12c0_value_contract.csv")
CELL12C_VALUE_CONTRACT_DF.to_csv(CELL12C_VALUE_CONTRACT_CSV, index=False)

globals()["CELL12C_VALUE_CONTRACT_DF"] = CELL12C_VALUE_CONTRACT_DF
globals()["CELL12C_VALUE_CONTRACT_CSV"] = CELL12C_VALUE_CONTRACT_CSV

# Backward-compatible aliases expected by earlier modular cells.
VALUE_CONTRACT_DF = CELL12C_VALUE_CONTRACT_DF.copy()
VALUE_CONTRACT_CSV = CELL12C_VALUE_CONTRACT_CSV
globals()["VALUE_CONTRACT_DF"] = VALUE_CONTRACT_DF
globals()["VALUE_CONTRACT_CSV"] = VALUE_CONTRACT_CSV

log(
    "[Cell12.c.0] Value contract table built | "
    f"rows={len(CELL12C_VALUE_CONTRACT_DF)} | "
    f"csv={CELL12C_VALUE_CONTRACT_CSV}"
)
# ----------------------------------------------------------
# 8B) Publication-safety gate for 12.c.0 contract readiness
# ----------------------------------------------------------
_safety_rows = []

_safety_rows.append({
    "check": "role_contract_violations",
    "value": 0,
    "threshold": CELL12C_PUBLICATION_SAFETY_THRESHOLDS["fatal"]["role_contract_violations"],
    "severity": "fatal",
    "passed": True,
    "reason": "role-aware continuous resolver passed",
})

_safety_rows.append({
    "check": "continuous_value_target_count",
    "value": int(len(CONT_VALUE_COLS)),
    "threshold": 148,
    "severity": "fatal",
    "passed": bool(len(CONT_VALUE_COLS) == 148),
    "reason": "Cell 12.c must receive exactly 148 continuous IoT value targets",
})

_safety_rows.append({
    "check": "missing_value_contract_rows",
    "value": int(len(CONT_VALUE_COLS) - len(CELL12C_VALUE_CONTRACT_DF)),
    "threshold": CELL12C_PUBLICATION_SAFETY_THRESHOLDS["fatal"]["missing_value_contract_rows"],
    "severity": "fatal",
    "passed": bool(len(CELL12C_VALUE_CONTRACT_DF) == len(CONT_VALUE_COLS)),
    "reason": "one value-contract row required per continuous target",
})

_missing_a1_n = int((~CELL12C_VALUE_CONTRACT_DF["has_a1_baseline"].astype(bool)).sum())
_safety_rows.append({
    "check": "missing_a1_baseline_cols",
    "value": _missing_a1_n,
    "threshold": CELL12C_PUBLICATION_SAFETY_THRESHOLDS["fatal"]["missing_a1_baseline_cols"],
    "severity": "fatal",
    "passed": bool(_missing_a1_n == 0),
    "reason": "A1 baseline must be eligible for every continuous target",
})

_safety_rows.append({
    "check": "test_used_for_selection",
    "value": 0,
    "threshold": CELL12C_PUBLICATION_SAFETY_THRESHOLDS["fatal"]["test_used_for_selection"],
    "severity": "fatal",
    "passed": True,
    "reason": "Cell 12.c.0 defines utilities/contracts only and does not select generators",
})

CELL12C0_PUBLICATION_SAFETY_DF = _cell12c0_publication_safety_gate(
    _safety_rows,
    context="Cell12.c.0",
    raise_on_fatal=True,
)

CELL12C0_PUBLICATION_SAFETY_CSV = os.path.join(
    REPORT_DIR,
    "cell12c0_publication_safety_gate.csv",
)
CELL12C0_PUBLICATION_SAFETY_DF.to_csv(CELL12C0_PUBLICATION_SAFETY_CSV, index=False)

globals()["CELL12C0_PUBLICATION_SAFETY_DF"] = CELL12C0_PUBLICATION_SAFETY_DF
globals()["CELL12C0_PUBLICATION_SAFETY_CSV"] = CELL12C0_PUBLICATION_SAFETY_CSV

log(
    "[Cell12.c.0] Publication-safety gate passed | "
    f"csv={CELL12C0_PUBLICATION_SAFETY_CSV}"
)
# ----------------------------------------------------------
# 9) Scaffold audit tables
# ----------------------------------------------------------
_mask_rows = []
for c in CONT_MASK_COLS:
    syn = pd.to_numeric(IOT_SYN_MASKS[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    _mask_rows.append({
        "col": c,
        "syn_rate": float(np.mean(syn > 0.5)),
        "test_rate_QA_only": np.nan,
        "syn_minus_test_QA_only": np.nan,
        "test_QA_deferred_to": "Cell 12.c.6",
        "test_used_for_selection": False,
    })

CELL12C0_MASK_AUDIT_DF = pd.DataFrame(_mask_rows)
CELL12C0_MASK_AUDIT_CSV = os.path.join(REPORT_DIR, "cell12c0_mask_audit.csv")
CELL12C0_MASK_AUDIT_DF.to_csv(CELL12C0_MASK_AUDIT_CSV, index=False)

_binary_rows = []
for c in BIN_COLS:
    syn = pd.to_numeric(IOT_SYN_BINARY[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    tr = pd.to_numeric(df_tr[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if c in df_tr.columns else np.zeros(N_TR, dtype=np.float32)
    va = pd.to_numeric(df_val[c], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if c in df_val.columns else np.zeros(N_VAL, dtype=np.float32)

    target_trainval = float((tr.sum() + va.sum()) / max(1, len(tr) + len(va)))
    syn_rate = float(np.mean(syn > 0.5))

    _binary_rows.append({
        "col": c,
        "entity": _infer_iot_entity(c),
        "syn_rate": syn_rate,
        "trainval_rate": target_trainval,
        "abs_err_vs_trainval": float(abs(syn_rate - target_trainval)),
        "test_rate_QA_only": np.nan,
        "test_QA_deferred_to": "Cell 12.c.6",
        "test_used_for_selection": False,
    })

CELL12C0_BINARY_AUDIT_DF = pd.DataFrame(_binary_rows)
CELL12C0_BINARY_AUDIT_CSV = os.path.join(REPORT_DIR, "cell12c0_binary_scaffold_audit.csv")
CELL12C0_BINARY_AUDIT_DF.to_csv(CELL12C0_BINARY_AUDIT_CSV, index=False)

_regime_rows = []
for e in ENTITY_LIST:
    x = pd.to_numeric(ENTITY_REGIME_SYN[e], errors="coerce").fillna(-1).to_numpy(dtype=np.int16)
    occ = {f"state_{s}_frac": float(np.mean(x == s)) for s in range(4)}
    dom_state = int(max(range(4), key=lambda s: occ[f"state_{s}_frac"]))
    dom_frac = float(occ[f"state_{dom_state}_frac"])
    switches = int(np.sum(x[1:] != x[:-1])) if len(x) > 1 else 0
    _regime_rows.append({
        "entity": e,
        "dominant_state": dom_state,
        "dominant_frac": dom_frac,
        "switches": switches,
        "switch_rate_per_10k": float(switches * 10000.0 / max(1, len(x))),
        **occ,
    })

CELL12C0_REGIME_AUDIT_DF = pd.DataFrame(_regime_rows)
CELL12C0_REGIME_AUDIT_CSV = os.path.join(REPORT_DIR, "cell12c0_entity_regime_scaffold_audit.csv")
CELL12C0_REGIME_AUDIT_DF.to_csv(CELL12C0_REGIME_AUDIT_CSV, index=False)

globals()["CELL12C0_MASK_AUDIT_DF"] = CELL12C0_MASK_AUDIT_DF
globals()["CELL12C0_BINARY_AUDIT_DF"] = CELL12C0_BINARY_AUDIT_DF
globals()["CELL12C0_REGIME_AUDIT_DF"] = CELL12C0_REGIME_AUDIT_DF

log(
    "[Cell12.c.0] Scaffold audits saved | "
    f"mask={CELL12C0_MASK_AUDIT_CSV} | "
    f"binary={CELL12C0_BINARY_AUDIT_CSV} | "
    f"regime={CELL12C0_REGIME_AUDIT_CSV}"
)

# ----------------------------------------------------------
# 10) Define modular 12.c downstream cell plan
# ----------------------------------------------------------
CELL12C_MODULE_PLAN = {
    "12.c.0": {
        "name": "shared_portfolio_utilities_and_contract",
        "role": "validate scaffold, normalize portfolio, define shared scoring/domain utilities",
        "uses_test_values": False,
        "generates_values": False,
        "selects_generators": False,
    },
    "12.c.1": {
        "name": "A1_iot_value_baseline_from_train",
        "role": "build mandatory A1 IoT value baseline from TRAIN only",
        "uses_test_values": False,
        "generates_values": True,
        "selection": "A1 is default baseline for every value column",
    },
    "12.c.2": {
        "name": "val_candidate_generation_all_eligible_generators",
        "role": "generate VAL-sized candidates for all eligible generators",
        "uses_test_values": False,
        "generates_values": True,
        "selection": "none; candidate materialization only",
    },
    "12.c.3": {
        "name": "val_only_portfolio_selection",
        "role": "score A1 and all VAL candidates, then lock one generator per column/family",
        "uses_test_values": False,
        "generates_values": False,
        "selection": "VAL only; A1 wins unless a candidate materially beats it",
    },
    "12.c.4": {
        "name": "test_length_materialization_from_locked_choices",
        "role": "materialize TEST-length synthetic IoT values using only locked choices from 12.c.3",
        "uses_test_values": False,
        "generates_values": True,
        "selection": "forbidden; choices already locked",
    },
    "12.c.5": {
        "name": "final_contract_enforcement_and_mask_application",
        "role": "apply synthetic masks, enforce domains, align final schema",
        "uses_test_values": False,
        "generates_values": False,
        "selection": "forbidden",
    },
    "12.c.6": {
        "name": "final_test_QA_publication_artifacts",
        "role": "compare final synthetic IoT values against real TEST for QA only",
        "uses_test_values": True,
        "generates_values": False,
        "selection": "forbidden; QA only",
    },
}

globals()["CELL12C_MODULE_PLAN"] = CELL12C_MODULE_PLAN

# ----------------------------------------------------------
# 11) Final manifest
# ----------------------------------------------------------
CELL12C0_CONTRACT = {
    "version": CELL12C0_VERSION,
    "policy": CELL12C_POLICY,
    "split_lengths": {
        "N_TR": int(N_TR),
        "N_VAL": int(N_VAL),
        "N_TE": int(N_TE),
    },
    "scaffold_sources": {
        "cell12b_manifest_path": cell12b_manifest_path if os.path.exists(cell12b_manifest_path) else None,
        "cell12b_summary_path": cell12b_summary_path if os.path.exists(cell12b_summary_path) else None,
        "cell12b_portfolio_csv": cell12b_portfolio_csv if os.path.exists(cell12b_portfolio_csv) else None,
        "cell12b_scaffold_contract_path": CELL12B_SCAFFOLD_CONTRACT_PATH,
        "cell12a_foundation_manifest_path": CELL12A_FOUNDATION_MANIFEST_PATH,
        "cell10_8_selector_contract_path": CELL10_8_SELECTOR_CONTRACT_PATH,
        "cell10_9_materializer_contract_path": CELL10_9_MATERIALIZER_CONTRACT_PATH,
        "precell11_runtime_contract_path": PRECELL11_RUNTIME_CONTRACT_PATH,
        "cell11_final_manifest_path": CELL11_FINAL_MANIFEST_PATH_REQUIRED,
    },
    "protocol_context": {
        "A2_PROTOCOL_PATH": A2_PROTOCOL_PATH,
        "protocol_matrix_loaded_in_12c0": False,
        "protocol_context_materialization_deferred_to": "Cell 12.c.4",
        "proto_cols_n": None,
        "protocol_context_shape": None,
    },
    "iot_contract": {
        "drivers_n": int(len(DRV_COLS)),
        "binary_n": int(len(BIN_COLS)),
        "cont_mask_n": int(len(CONT_MASK_COLS)),
        "cont_value_n": int(len(CONT_VALUE_COLS)),
        "role_aware_cont_value_n": int(len(CELL12C_CONT_VALUE_TARGETS)),
        "ddpm_optional_n": int(len(DDPM_COLS)),
        "non_ddpm_n": int(len(NON_DDPM_COLS)),
        "entities_n": int(len(ENTITY_LIST)),
        "only_role_aware_continuous_targets_enter_12c": True,
    },
    "shared_utilities": {
        "role_aware_column_resolver": "_cell12c0_role_aware_continuous_targets",
        "observed_only_arrays": "_cell12c0_observed_only_arrays",
        "observed_only_metric_bundle": "_cell12c0_observed_only_metric_bundle",
        "segment_contiguous_segments": "_cell12c0_contiguous_segments_from_mask",
        "segment_ids_from_mask": "_cell12c0_segment_ids_from_mask",
        "segment_aware_bool_runs": "_cell12c0_segment_aware_bool_runs",
        "segment_aware_sample_starts": "_cell12c0_segment_aware_sample_starts",
        "support_metric_bundle": "_cell12c0_support_metric_bundle",
        "publication_safety_gate": "_cell12c0_publication_safety_gate",
    },
    "routing": {
        "bucket_counts": routing_bucket_counts,
        "stage12c_kind_counts": routing_kind_counts,
    },
    "portfolio": {
        "eligible_rows": int(len(IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF)),
        "eligible_generator_counts": portfolio_generator_counts,
        "a1_available_for_all_value_cols": True,
    },
    "scaffold_audits": {
        "value_contract_csv": CELL12C_VALUE_CONTRACT_CSV,
        "mask_audit_csv": CELL12C0_MASK_AUDIT_CSV,
        "binary_audit_csv": CELL12C0_BINARY_AUDIT_CSV,
        "regime_audit_csv": CELL12C0_REGIME_AUDIT_CSV,
        "publication_safety_gate_csv": CELL12C0_PUBLICATION_SAFETY_CSV,
    },
    "module_plan": CELL12C_MODULE_PLAN,
    "hashes": {
        "value_contract_sha256": _sha256_file(CELL12C_VALUE_CONTRACT_CSV),
        "mask_audit_sha256": _sha256_file(CELL12C0_MASK_AUDIT_CSV),
        "binary_audit_sha256": _sha256_file(CELL12C0_BINARY_AUDIT_CSV),
        "regime_audit_sha256": _sha256_file(CELL12C0_REGIME_AUDIT_CSV),
        "publication_safety_gate_sha256": _sha256_file(CELL12C0_PUBLICATION_SAFETY_CSV),
        "policy_sha256": _sha256_jsonable(CELL12C_POLICY),
        "module_plan_sha256": _sha256_jsonable(CELL12C_MODULE_PLAN),
    },
    "no_generation_assertion": {
        "continuous_values_generated": False,
        "models_trained": False,
        "generators_selected": False,
        "test_used_for_selection": False,
    },
}

CELL12C0_CONTRACT_JSON = os.path.join(ARTDIR, "cell12c0_foundation_contract_manifest.json")
CELL12C0_CONTRACT_PATH = os.path.join(CONTRACT_DIR, "cell12c0_foundation_contract_v1_2_THESIS.json")
CELL12C0_AUDIT_JSON = os.path.join(REPORT_DIR, "cell12c0_foundation_audit.json")

_write_json(CELL12C0_CONTRACT_JSON, CELL12C0_CONTRACT)
_write_json(CELL12C0_CONTRACT_PATH, CELL12C0_CONTRACT)
_write_json(CELL12C0_AUDIT_JSON, CELL12C0_CONTRACT)

CELL12C0_CONTRACT["hashes"]["contract_manifest_sha256"] = _sha256_file(CELL12C0_CONTRACT_JSON)
CELL12C0_CONTRACT["hashes"]["canonical_contract_sha256"] = _sha256_file(CELL12C0_CONTRACT_PATH)
CELL12C0_CONTRACT["hashes"]["foundation_audit_sha256"] = _sha256_file(CELL12C0_AUDIT_JSON)

_write_json(CELL12C0_CONTRACT_JSON, CELL12C0_CONTRACT)
_write_json(CELL12C0_CONTRACT_PATH, CELL12C0_CONTRACT)
_write_json(CELL12C0_AUDIT_JSON, CELL12C0_CONTRACT)

globals()["CELL12C0_CONTRACT"] = CELL12C0_CONTRACT
globals()["CELL12C0_CONTRACT_JSON"] = CELL12C0_CONTRACT_JSON
globals()["CELL12C0_CONTRACT_PATH"] = CELL12C0_CONTRACT_PATH
globals()["CELL12C0_AUDIT_JSON"] = CELL12C0_AUDIT_JSON

# ----------------------------------------------------------
# 12) Final sanity assertions
# ----------------------------------------------------------
_required_exports = [
    "CELL12C_POLICY",
    "CELL12C_MODULE_PLAN",
    "CELL12C_VALUE_CONTRACT_DF",
    "CELL12C_VALUE_CONTRACT_CSV",
    "CELL12C0_MASK_AUDIT_DF",
    "CELL12C0_BINARY_AUDIT_DF",
    "CELL12C0_REGIME_AUDIT_DF",
    "CELL12C0_PUBLICATION_SAFETY_DF",
    "CELL12C0_PUBLICATION_SAFETY_CSV",
    "A2_PROTOCOL_PATH",
    "IOT_PROTOCOL_CONTEXT_TE",
    "VALUE_CONTRACT_DF",
    "VALUE_CONTRACT_CSV",
    "IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF",
    "CELL12C_CONT_VALUE_TARGETS",
    "CONT_VALUE_COLS_ROLE_AWARE",
    "CELL12C0_CONTRACT",
    "CELL12C0_CONTRACT_JSON",
    "CELL12C0_CONTRACT_PATH",
    "CELL12C0_AUDIT_JSON",
    "_cell12c0_distribution_metric_bundle",
    "_cell12c0_c2st_auc_lr",
    "_cell12c0_strict_value_mask_for_col",
    "_cell12c0_role_aware_continuous_targets",
    "_cell12c0_observed_only_arrays",
    "_cell12c0_observed_only_metric_bundle",
    "_cell12c0_contiguous_segments_from_mask",
    "_cell12c0_segment_ids_from_mask",
    "_cell12c0_segment_aware_bool_runs",
    "_cell12c0_segment_aware_sample_starts",
    "_cell12c0_support_metric_bundle",
    "_cell12c0_publication_safety_gate",
]

_missing_exports = [k for k in _required_exports if k not in globals()]
if _missing_exports:
    raise RuntimeError(f"[Cell12.c.0] Missing expected exports: {_missing_exports}")

if bool(CELL12C0_CONTRACT["no_generation_assertion"]["continuous_values_generated"]):
    raise RuntimeError("[Cell12.c.0] Internal contradiction: foundation cell claims values were generated.")

gc.collect()

log(
    "[Cell12.c.0] Contract locked | "
    f"role_aware_value_cols={len(CELL12C_CONT_VALUE_TARGETS)} | "
    f"value_cols={len(CONT_VALUE_COLS)} | "
    f"A1_eligible_all=True | "
    f"portfolio_generators={len(portfolio_generator_counts)} | "
    f"observed_only_evaluator=True | "
    f"segment_aware_run_utils=True | "
    f"support_aware_metrics=True | "
    f"publication_safety_gate=True | "
    f"protocol_context_loaded=False | "
    f"protocol_context_path={A2_PROTOCOL_PATH} | "
    f"manifest={CELL12C0_CONTRACT_JSON} | contract={CELL12C0_CONTRACT_PATH}"
)

log(
    "[Cell12.c.0] Next cells | "
    "12.c.1=A1 TRAIN baseline, "
    "12.c.2=VAL candidates for all eligible generators, "
    "12.c.3=VAL-only selection, "
    "12.c.4=TEST-length materialization from locked choices, "
    "12.c.5=contract/mask enforcement, "
    "12.c.6=final TEST QA only"
)

log("--- END: Cell 12.c.0 — IoT value synthesis foundation/contract (v1.2-THESIS contract-locked modular STUDY-THESIS) ---")