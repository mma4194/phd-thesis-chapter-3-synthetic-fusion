# ==========================================================
# CELL 12.a — IoT synthesis foundation
# v24.1-THESIS STUDY-THESIS-GRADE
#       + PHASE-1 QUALITY-DECOMPOSITION AWARENESS
#       + CELL 4.5/6.5 ROLE-OWNERSHIP CONTRACT CONSUMPTION
#       + CELL 11 A0/A1/A2 QUALITY SCAFFOLD CONSUMPTION
#       + BINARY / DRIVER / CONTINUOUS TARGET PLAN EXPORT
#       + A0/A1/A2 PROTOCOL-AWARE IoT FOUNDATION
#       + STRICT TRAIN/VAL FITTING, TEST-LENGTH ONLY
#       + ROUTING-FIRST DDPM POLICY
#       + PYTHON 3.9 COMPATIBLE
#       + CPS COUPLING CONTEXT FROM SELECTED PROTOCOL VARIANT
#
# Purpose:
# - Build the IoT foundation used by Cells 12.b/12.c.
# - Preserve the canonical IoT column contracts.
# - Load Cell 11 protocol A0/A1/A2 final variants.
# - Expose synthetic protocol context from A2 so IoT generation is coupled
#   with the selected protocol capture/value process.
# - Fit IoT grouping, taxonomy, regimes, and driver process from TRAIN/VAL.
# - Generate TEST-length synthetic IoT drivers without using TEST IoT values.
#
# Scientific boundary:
# - TRAIN: fitting taxonomy, entity maps, state clusters, driver episodes.
# - VAL: calibration/targeting of driver process and routing diagnostics.
# - TEST: length, schema, optional evaluation-only audits.
# - Synthetic protocol A2: allowed conditioning context for IoT generation.
# - Real TEST IoT values are never used to fit/generate IoT synthetic values.
# ==========================================================

log("--- START: Cell 12.a — IoT synthesis foundation (v24.1-THESIS phase1-quality-aware, contract-locked) ---")

import os
import re
import gc
import json
import math
import copy
import hashlib
import warnings
from collections import defaultdict, Counter
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

try:
    import torch
except Exception as e:
    raise RuntimeError(f"[Cell12.a] PyTorch is required. Error: {e}")

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
need = [
    "CFG", "log",
    "df_tr", "df_te",
    "time_col",

    # Canonical IoT groups from Cell 4.5 / 6.5
    "IOT_FULL_NAMESPACE_COLS",
    "IOT_DRIVER_COLS",
    "IOT_BIN_COLS",
    "IOT_CONT_COLS",
    "IOT_STATE_COLS",
    "IOT_META_COLS",
    "IOT_EXCLUDED_COLS",

    # Continuous semantic subtypes
    "IOT_COUNTLIKE_COLS",
    "IOT_TELEMETRY_COLS",
    "IOT_STEP_PROGRESS_COLS",
    "IOT_DISCRETE_STATE_COLS",
    "IOT_SETTINGLIKE_COLS",

    # Phase-1 CPS coupling from Cell 6.5
    "IOT_CPS_COUPLING_POLICY",
    "IOT_GROUP_LOADER_CONTRACT",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell12.a] Missing required global(s): {missing}")

if "df_val" in globals() and isinstance(globals()["df_val"], pd.DataFrame):
    df_val = globals()["df_val"].copy()
    log("[Cell12.a] Using VAL dataframe from global: df_val")
elif "df_va" in globals() and isinstance(globals()["df_va"], pd.DataFrame):
    df_val = globals()["df_va"].copy()
    log("[Cell12.a] Using VAL dataframe from global: df_va")
else:
    raise RuntimeError(
        "[Cell12.a] No canonical VAL dataframe found. "
        "Run the canonical split cells first; do not reconstruct VAL from TRAIN."
    )

OUTDIR = str(CFG["outdir"])
OUT_SYN = os.path.join(OUTDIR, "synthetic")
OUT_ART = os.path.join(OUTDIR, "artifacts")
REPORT_DIR = os.path.join(OUTDIR, "reports")

for _d in [OUT_SYN, OUT_ART, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

SEED = int(CFG.get("seed", 42))
rng = np.random.default_rng(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"[Cell12.a] Torch device: {device}")

TIME_COL = str(globals()["time_col"])
for _nm, _df in [("TRAIN", df_tr), ("VAL", df_val), ("TEST", df_te)]:
    if TIME_COL not in _df.columns:
        raise RuntimeError(f"[Cell12.a] Canonical time column missing from {_nm}: {TIME_COL}")

N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

if N_TR <= 0 or N_VAL <= 0 or N_TE <= 0:
    raise RuntimeError(f"[Cell12.a] Empty split detected: N_TR={N_TR}, N_VAL={N_VAL}, N_TE={N_TE}")

# ----------------------------------------------------------
# 1) Configuration
# ----------------------------------------------------------
CFG12 = {
    "window_len": int(CFG.get("cell12a_window_len", 512)),
    "window_jitter": int(CFG.get("cell12a_window_jitter", 1800)),
    "state_cluster_k": int(CFG.get("cell12a_state_cluster_k", 14)),
    "driver_burst_gap_thr": int(CFG.get("cell12a_driver_burst_gap_thr", 12)),
    "driver_base_event_boost_clip": tuple(CFG.get("cell12a_driver_base_event_boost_clip", (0.75, 1.35))),

    "binary_near_constant_low": float(CFG.get("cell12a_binary_near_constant_low", 0.005)),
    "binary_near_constant_high": float(CFG.get("cell12a_binary_near_constant_high", 0.995)),
    "binary_long_run_thr": int(CFG.get("cell12a_binary_long_run_thr", 512)),

    "calib_clip_q": tuple(CFG.get("cell12a_calib_clip_q", (0.001, 0.999))),
    "std_floor": float(CFG.get("cell12a_std_floor", 1e-6)),
    "sampling_temperature": float(CFG.get("cell12a_sampling_temperature", 0.95)),

    "value_mask_piece_min": int(CFG.get("cell12a_value_mask_piece_min", 256)),
    "value_mask_piece_max": int(CFG.get("cell12a_value_mask_piece_max", 4096)),
    "binary_piece_min": int(CFG.get("cell12a_binary_piece_min", 256)),
    "binary_piece_max": int(CFG.get("cell12a_binary_piece_max", 4096)),
    "counter_piece_min": int(CFG.get("cell12a_counter_piece_min", 256)),
    "counter_piece_max": int(CFG.get("cell12a_counter_piece_max", 4096)),

    "telemetry_fallback_low_ratio": float(CFG.get("cell12a_telemetry_fallback_low_ratio", 0.35)),
    "telemetry_fallback_high_ratio": float(CFG.get("cell12a_telemetry_fallback_high_ratio", 1.85)),
    "telemetry_fallback_lag_thr": float(CFG.get("cell12a_telemetry_fallback_lag_thr", 0.20)),

    "step_progress_min_episode_len": int(CFG.get("cell12a_step_progress_min_episode_len", 4)),
    "step_progress_idle_quantile": float(CFG.get("cell12a_step_progress_idle_quantile", 0.10)),

    "suspicious_print_topk": int(CFG.get("cell12a_suspicious_print_topk", 25)),

    # Semantic fail-fast guards
    "max_binary_non01_cols": int(CFG.get("cell12a_max_binary_non01_cols", 0)),
    "max_multi_block_overlap_cols": int(CFG.get("cell12a_max_multi_block_overlap_cols", 0)),
    "require_step_progress_min": int(CFG.get("cell12a_require_step_progress_min", 1)),
    "max_telemetry_misroutes": int(CFG.get("cell12a_max_telemetry_misroutes", 0)),

    # State-cluster governance
    "warn_largest_cluster_frac": float(CFG.get("cell12a_warn_largest_cluster_frac", 0.60)),
    "max_driver_named_frac": float(CFG.get("cell12a_max_driver_named_frac", 0.85)),
    "min_effective_cluster_count": float(CFG.get("cell12a_min_effective_cluster_count", 3.0)),
    "min_non_background_frac": float(CFG.get("cell12a_min_non_background_frac", 0.08)),
    "min_transition_or_driver_frac": float(CFG.get("cell12a_min_transition_or_driver_frac", 0.02)),

    # Protocol coupling
    "require_cell11_protocol_variants": bool(CFG.get("cell12a_require_cell11_protocol_variants", True)),
    "protocol_context_roll_windows": list(CFG.get("cell12a_protocol_context_roll_windows", [30, 300, 1800])),
    "driver_target_train_weight": float(CFG.get("cell12a_driver_target_train_weight", 0.35)),
    "driver_target_val_weight": float(CFG.get("cell12a_driver_target_val_weight", 0.65)),
    "driver_protocol_context_strength": float(CFG.get("cell12a_driver_protocol_context_strength", 0.35)),
}

if CFG12["window_len"] <= 0:
    raise RuntimeError("[Cell12.a] CFG12['window_len'] must be positive.")

window_fits = bool(N_TR >= CFG12["window_len"])

# ----------------------------------------------------------
# 2) General helpers
# ----------------------------------------------------------
def _stable_int(s: str, mod: int = 1000000) -> int:
    h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
    return int(h[:10], 16) % mod


def _rng_for(name: str, offset: int = 0):
    return np.random.default_rng(SEED + offset + _stable_int(name))


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

def _read_json_required(path: str, label: str):
    if not os.path.exists(path):
        raise RuntimeError(f"[Cell12.a] Missing required {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell12.a] {label} must be a JSON object: {path}")
    return obj


def _read_csv_required(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(f"[Cell12.a] Missing required {label}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"[Cell12.a] Required {label} is empty: {path}")
    return df


def _sha_list(xs) -> str:
    return hashlib.sha256(("||".join(map(str, xs))).encode("utf-8")).hexdigest()
        
def _sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_numeric_frame(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    return out.astype(np.float32)


def _finite_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _safe_mean(x):
    x = _finite_1d(x)
    return np.nan if x.size == 0 else float(np.mean(x))


def _safe_std(x):
    x = _finite_1d(x)
    return np.nan if x.size <= 1 else float(np.std(x))


def _safe_quantile(x, q):
    x = _finite_1d(x)
    return np.nan if x.size == 0 else float(np.quantile(x, q))


def _nonzero_rate(x):
    x = _finite_1d(x)
    return np.nan if x.size == 0 else float(np.mean(x > 0))


def _series_is_binary_01(s: pd.Series) -> bool:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return False
    u = np.unique(x.to_numpy(dtype=np.float32))
    return len(u) <= 2 and set(u.tolist()).issubset({0.0, 1.0})


def _to_binary01_array(s: pd.Series) -> np.ndarray:
    """
    Robust binary-state parser.

    Handles:
    - numeric 0/1
    - bool
    - common HA/string states: on/off, true/false, open/closed,
      detected/clear, connected/disconnected, home/not_home
    - unavailable/unknown/none/nan as 0 for synthesis foundation

    This is intentionally conservative: anything not confidently active
    becomes 0 rather than being treated as a continuous numeric signal.
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool).to_numpy(dtype=np.int8)

    x_num = pd.to_numeric(s, errors="coerce")
    finite_rate = float(x_num.notna().mean()) if len(x_num) else 0.0

    if finite_rate >= 0.95:
        x = x_num.fillna(0.0).to_numpy(dtype=np.float32)
        return (x > 0.5).astype(np.int8)

    ss = s.astype(str).str.strip().str.lower()

    active = {
        "1", "1.0", "true", "t", "yes", "y",
        "on", "open", "opened", "active", "detected",
        "connected", "online", "home", "present",
        "motion", "occupied", "wet", "unlocked",
    }

    inactive = {
        "0", "0.0", "false", "f", "no", "n",
        "off", "closed", "close", "inactive", "clear",
        "not_detected", "not detected", "disconnected", "offline",
        "not_home", "away", "absent", "idle", "dry", "locked",
        "unknown", "unavailable", "none", "nan", "", "null",
    }

    out = np.zeros(len(ss), dtype=np.int8)
    out[ss.isin(active).to_numpy()] = 1

    # Numeric strings that were sparse/dirty still get parsed if possible.
    x_num2 = pd.to_numeric(ss, errors="coerce")
    num_ok = x_num2.notna().to_numpy()
    out[num_ok] = (x_num2[num_ok].to_numpy(dtype=np.float32) > 0.5).astype(np.int8)

    # Explicit inactive stays 0.
    out[ss.isin(inactive).to_numpy()] = 0

    return out.astype(np.int8, copy=False)


def _safe_binary_frame(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Canonicalize binary IoT columns to strict 0/1 float32.
    """
    data = {}
    for c in cols:
        data[c] = _to_binary01_array(df[c]).astype(np.float32)
    return pd.DataFrame(data, index=df.index).astype(np.float32)


def _time_of_day_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    try:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        ok = float(ts.notna().mean())
    except Exception:
        ts = None
        ok = 0.0

    if ts is not None and ok > 0.95:
        sec = (
            ts.dt.hour.astype(np.int32) * 3600
            + ts.dt.minute.astype(np.int32) * 60
            + ts.dt.second.astype(np.int32)
        ).to_numpy(dtype=np.float32)
    else:
        sec = (np.arange(len(df), dtype=np.int64) % 86400).astype(np.float32)

    theta = 2.0 * np.pi * sec / 86400.0
    return pd.DataFrame(
        {
            "tod_sin": np.sin(theta).astype(np.float32),
            "tod_cos": np.cos(theta).astype(np.float32),
        },
        index=df.index,
    )


# ----------------------------------------------------------
# 3) Load Cell 11 protocol variants for CPS coupling
# ----------------------------------------------------------
def _load_protocol_variant(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise RuntimeError(f"[Cell12.a] Missing {label} protocol artifact: {path}")
    df = pd.read_parquet(path)
    if len(df) != N_TE:
        raise RuntimeError(f"[Cell12.a] {label} row mismatch: {len(df)} vs N_TE={N_TE}")
    return df.reset_index(drop=True)


CELL11_FINAL_MANIFEST_PATH = os.path.join(OUT_ART, "cell11_final_protocol_variant_manifest.json")

if os.path.exists(CELL11_FINAL_MANIFEST_PATH):
    with open(CELL11_FINAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        CELL11_FINAL_MANIFEST = json.load(f)
else:
    CELL11_FINAL_MANIFEST = {}

A0_PROTOCOL_FINAL_PATH = os.path.join(OUT_SYN, "A0_PROTOCOL_FINAL.parquet")
A1_PROTOCOL_FINAL_PATH = os.path.join(OUT_SYN, "A1_PROTOCOL_FINAL.parquet")
A2_PROTOCOL_FINAL_PATH = os.path.join(OUT_SYN, "A2_PROTOCOL_FINAL.parquet")

# ----------------------------------------------------------
# 3.0) Phase-1 quality-decomposition artifacts from Cell 11
# ----------------------------------------------------------
QUALITY_DIMENSION_REGISTRY_PATH = os.path.join(
    OUT_ART, "quality_dimension_registry.json"
)

A0_A1_A2_VARIANT_MANIFEST_PATH = os.path.join(
    OUT_ART, "a0_a1_a2_variant_manifest.json"
)

QUALITY_ATTRIBUTION_MATRIX_PATH = os.path.join(
    REPORT_DIR, "quality_attribution_matrix.csv"
)

CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH = os.path.join(
    OUT_ART, "cell11_iot_downstream_protocol_context.json"
)

CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH = os.path.join(
    REPORT_DIR, "cell11_iot_downstream_protocol_activity_summary.csv"
)

IOT_ROLE_OWNERSHIP_PATH = os.path.join(
    OUT_ART, "iot_role_ownership.csv"
)

IOT_PARTITION_CONTRACT_PATH = os.path.join(
    OUT_ART, "iot_partition_contract.json"
)

IOT_CELL_ROUTING_MANIFEST_PATH = os.path.join(
    OUT_ART, "iot_cell_routing_manifest.json"
)

# ----------------------------------------------------------
# 3.0b) Contract-lock guards from Cell 11 / PRE-CELL11
# ----------------------------------------------------------
CELL10_8_SELECTOR_CONTRACT_PATH = os.path.join(
    OUT_ART, "contracts", "cell10_8_val_only_selector_contract_v3_6_THESIS.json"
)
CELL10_9_MATERIALIZER_CONTRACT_PATH = os.path.join(
    OUT_ART, "contracts", "cell10_9_protocol_variant_materialization_contract_v2_1_THESIS.json"
)
PRECELL11_RUNTIME_CONTRACT_PATH = os.path.join(
    OUT_ART, "contracts", "pre_cell11_a0_a1_a2_variant_runtime_contract_v4_1_THESIS.json"
)
CELL11_FINAL_AUDIT_PATH = os.path.join(
    REPORT_DIR, "cell11_final_protocol_variant_audit.json"
)
CELL11B_QUARANTINE_MANIFEST_PATH = os.path.join(
    OUT_ART, "cell11b_protocol_test_regression_quarantine_manifest.json"
)

CELL10_8_SELECTOR_CONTRACT = _read_json_required(
    CELL10_8_SELECTOR_CONTRACT_PATH,
    "Cell 10.8 v3.6 selector contract",
)
CELL10_9_MATERIALIZER_CONTRACT = _read_json_required(
    CELL10_9_MATERIALIZER_CONTRACT_PATH,
    "Cell 10.9 v2.1 materializer contract",
)
PRECELL11_RUNTIME_CONTRACT = _read_json_required(
    PRECELL11_RUNTIME_CONTRACT_PATH,
    "PRE-CELL11 v4.1 runtime contract",
)

for _name, _contract, _forbidden in [
    (
        "Cell10.8 selector",
        CELL10_8_SELECTOR_CONTRACT,
        [
            "test_values_read",
            "test_used_for_fitting",
            "test_used_for_thresholding",
            "test_used_for_selection",
            "test_rescue_materialization",
            "in_selector_candidate_generation",
            "a0_used_for_selection",
            "unregistered_file_scan",
        ],
    ),
    (
        "Cell10.9 materializer",
        CELL10_9_MATERIALIZER_CONTRACT,
        [
            "test_values_read",
            "test_metrics_computed",
            "test_used_for_fitting",
            "test_used_for_thresholding",
            "test_used_for_selection",
            "test_used_for_repair",
            "downstream_test_rescue_materialization",
            "a0_used_for_selection",
        ],
    ),
]:
    for _key in _forbidden:
        if bool(_contract.get(_key, False)):
            raise RuntimeError(f"[Cell12.a] {_name} contract violation: {_key}=True")

if bool(PRECELL11_RUNTIME_CONTRACT.get("cell11_must_not_apply_test_informed_rescue_or_repair", True)) is not True:
    raise RuntimeError("[Cell12.a] PRE-CELL11 contract must forbid TEST-informed rescue/repair.")

# Cell 11b is a diagnostic-only idea in the cleaned pipeline. If a prior run
# wrote a quarantine manifest that overwrote canonical A2 artifacts, fail closed.
if os.path.exists(CELL11B_QUARANTINE_MANIFEST_PATH):
    _cell11b_manifest = _read_json_required(
        CELL11B_QUARANTINE_MANIFEST_PATH,
        "Cell 11b quarantine manifest",
    )
    _cell11b_outputs = _cell11b_manifest.get("outputs", {}) if isinstance(_cell11b_manifest, dict) else {}
    _cell11b_policy = _cell11b_manifest.get("policy", {}) if isinstance(_cell11b_manifest, dict) else {}
    _overwritten = bool(
        _cell11b_policy.get("canonical_A2_overwritten", False)
        or _cell11b_outputs.get("A2_PROTOCOL_FINAL_overwritten")
        or _cell11b_outputs.get("A2_PROTOCOL_TEST_overwritten")
    )
    if _overwritten:
        raise RuntimeError(
            "[Cell12.a] Refusing to use TEST-quarantined protocol A2 from Cell 11b. "
            "Delete/revert Cell 11b outputs and rerun Cells 10.8–11 canonical path."
        )

# TEST IoT values may be loaded below for schema/final-QA compatibility only.
# They must not be used to fit, select, calibrate, materialize, repair, or
# fallback-generate IoT synthetic targets in Cell 12.b/12.c.

# ----------------------------------------------------------
# 3.1) Load Phase-1 quality-decomposition scaffold
# ----------------------------------------------------------
QUALITY_DIMENSION_REGISTRY = _read_json_required(
    QUALITY_DIMENSION_REGISTRY_PATH,
    "quality_dimension_registry.json from Cell 11",
)

A0_A1_A2_VARIANT_MANIFEST = _read_json_required(
    A0_A1_A2_VARIANT_MANIFEST_PATH,
    "a0_a1_a2_variant_manifest.json from Cell 11",
)

QUALITY_ATTRIBUTION_MATRIX = _read_csv_required(
    QUALITY_ATTRIBUTION_MATRIX_PATH,
    "quality_attribution_matrix.csv from Cell 11",
)

CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT = _read_json_required(
    CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT_PATH,
    "cell11_iot_downstream_protocol_context.json from Cell 11",
)

CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY = _read_csv_required(
    CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY_PATH,
    "cell11_iot_downstream_protocol_activity_summary.csv from Cell 11",
)

IOT_ROLE_OWNERSHIP_DF = _read_csv_required(
    IOT_ROLE_OWNERSHIP_PATH,
    "iot_role_ownership.csv from Cell 4.5",
)

IOT_PARTITION_CONTRACT = _read_json_required(
    IOT_PARTITION_CONTRACT_PATH,
    "iot_partition_contract.json from Cell 4.5",
)

IOT_CELL_ROUTING_MANIFEST = _read_json_required(
    IOT_CELL_ROUTING_MANIFEST_PATH,
    "iot_cell_routing_manifest.json from Cell 4.5",
)

# ----------------------------------------------------------
# 3.2) Normalize and validate Q1–Q6 quality-dimension registry
# ----------------------------------------------------------
def _normalize_quality_dimension_registry(obj: dict) -> Dict[str, Dict[str, object]]:
    """
    Normalize Cell 11 quality-dimension registry to canonical Q1–Q6.

    Accepted layouts:
    1. {"quality_dimensions": {"Q1": {...}, ..., "Q6": {...}}}
    2. {"dimensions": {"Q1": {...}, ..., "Q6": {...}}}
    3. {"dimensions": [{"id": "Q1", ...}, ...]}
    4. {"dimensions": {"distributional_fidelity": {...}, ...}}
       In this case, preserve Cell 11 insertion order and map first six to Q1–Q6.

    The scientific contract is that six quality dimensions exist and are
    propagated consistently. Cell 12.a should not depend on fragile JSON key names.
    """
    if not isinstance(obj, dict):
        raise RuntimeError("[Cell12.a] QUALITY_DIMENSION_REGISTRY must be a JSON object.")

    candidate_keys = [
        "quality_dimensions",
        "dimensions",
        "quality_dimension_registry",
        "registry",
        "Q_dimensions",
    ]

    candidates = []
    for key in candidate_keys:
        if key in obj:
            candidates.append((key, obj[key]))

    # Also allow top-level {"Q1": ..., ..., "Q6": ...}
    candidates.append(("top_level", obj))

    def _extract_qid(k, v, fallback_qid=None):
        if isinstance(v, dict):
            for kk in [
                "id",
                "qid",
                "q_id",
                "dimension_id",
                "quality_dimension_id",
                "quality_id",
                "code",
                "label",
            ]:
                if kk in v:
                    vv = str(v.get(kk))
                    m = re.search(r"\bQ([1-6])\b", vv, flags=re.IGNORECASE)
                    if m:
                        return f"Q{m.group(1)}"

        ks = str(k)
        m = re.search(r"\bQ([1-6])\b", ks, flags=re.IGNORECASE)
        if m:
            return f"Q{m.group(1)}"

        return fallback_qid

    for source_key, cand in candidates:
        # --------------------------------------------------
        # Case A: dict-based dimensions
        # --------------------------------------------------
        if isinstance(cand, dict):
            out = {}

            # First pass: explicit Q ids from keys or values.
            for k, v in cand.items():
                qid = _extract_qid(k, v, fallback_qid=None)
                if qid is not None and re.fullmatch(r"Q[1-6]", qid):
                    vv = dict(v) if isinstance(v, dict) else {"value": v}
                    vv.setdefault("original_dimension_key", str(k))
                    vv.setdefault("normalization_source", source_key)
                    out[qid] = vv

            if all(f"Q{i}" in out for i in range(1, 7)):
                return {f"Q{i}": out[f"Q{i}"] for i in range(1, 7)}

            # Second pass: semantic-key dict with >= 6 dimensions.
            # Preserve insertion order from JSON.
            if len(cand) >= 6:
                ordered_items = list(cand.items())[:6]
                out = {}
                for i, (k, v) in enumerate(ordered_items, start=1):
                    qid = f"Q{i}"
                    vv = dict(v) if isinstance(v, dict) else {"value": v}
                    vv.setdefault("original_dimension_key", str(k))
                    vv.setdefault("normalization_source", source_key)
                    vv.setdefault("assigned_qid_by_order", True)
                    out[qid] = vv

                return out

        # --------------------------------------------------
        # Case B: list-based dimensions
        # --------------------------------------------------
        if isinstance(cand, list):
            out = {}

            # First pass: explicit Q ids.
            for i, item in enumerate(cand, start=1):
                fallback_qid = f"Q{i}" if i <= 6 else None
                qid = _extract_qid(i, item, fallback_qid=fallback_qid)

                if qid is not None and re.fullmatch(r"Q[1-6]", qid):
                    vv = dict(item) if isinstance(item, dict) else {"value": item}
                    vv.setdefault("original_dimension_index", int(i - 1))
                    vv.setdefault("normalization_source", source_key)
                    out[qid] = vv

            if all(f"Q{i}" in out for i in range(1, 7)):
                return {f"Q{i}": out[f"Q{i}"] for i in range(1, 7)}

            # Second pass: order-based fallback.
            if len(cand) >= 6:
                out = {}
                for i, item in enumerate(cand[:6], start=1):
                    qid = f"Q{i}"
                    vv = dict(item) if isinstance(item, dict) else {"value": item}
                    vv.setdefault("original_dimension_index", int(i - 1))
                    vv.setdefault("normalization_source", source_key)
                    vv.setdefault("assigned_qid_by_order", True)
                    out[qid] = vv

                return out

    dim_obj = obj.get("dimensions", None)
    if isinstance(dim_obj, dict):
        dim_preview = list(dim_obj.keys())[:20]
    elif isinstance(dim_obj, list):
        dim_preview = [type(x).__name__ for x in dim_obj[:10]]
    else:
        dim_preview = str(type(dim_obj))

    raise RuntimeError(
        "[Cell12.a] Could not normalize quality_dimension_registry.json into Q1–Q6. "
        f"Available top-level keys={list(obj.keys())[:30]} | "
        f"dimensions_preview={dim_preview}"
    )


QDIM_REGISTRY = _normalize_quality_dimension_registry(QUALITY_DIMENSION_REGISTRY)
_qdims = QDIM_REGISTRY

_missing_qdims = [f"Q{i}" for i in range(1, 7) if f"Q{i}" not in QDIM_REGISTRY]
if _missing_qdims:
    raise RuntimeError(
        f"[Cell12.a] quality_dimension_registry.json is missing required dimensions: {_missing_qdims}"
    )

globals()["QDIM_REGISTRY"] = QDIM_REGISTRY

_variant_defs = A0_A1_A2_VARIANT_MANIFEST.get("variants", {})
if not isinstance(_variant_defs, dict):
    raise RuntimeError("[Cell12.a] A0/A1/A2 variant manifest has invalid 'variants' object.")

for _v in ["A0", "A1", "A2"]:
    if _v not in _variant_defs:
        raise RuntimeError(f"[Cell12.a] Variant manifest missing {_v} definition.")

_a0_role = str(_variant_defs["A0"].get("publication_role", ""))
_a1_role = str(_variant_defs["A1"].get("publication_role", ""))
_a2_role = str(_variant_defs["A2"].get("publication_role", ""))

if _a0_role != "diagnostic_only":
    raise RuntimeError(f"[Cell12.a] A0 must be diagnostic_only, got: {_a0_role}")
if _a1_role != "attribution_baseline":
    raise RuntimeError(f"[Cell12.a] A1 must be attribution_baseline, got: {_a1_role}")
if _a2_role != "final_candidate":
    raise RuntimeError(f"[Cell12.a] A2 must be final_candidate, got: {_a2_role}")

log(
    "[Cell12.a] Loaded Phase-1 quality scaffold | "
    f"Q_dimensions={len(_qdims)} | "
    f"A0_role={_a0_role} | A1_role={_a1_role} | A2_role={_a2_role} | "
    f"attribution_rows={len(QUALITY_ATTRIBUTION_MATRIX)}"
)

if CFG12["require_cell11_protocol_variants"]:
    A0_PROTOCOL_FINAL = _load_protocol_variant(A0_PROTOCOL_FINAL_PATH, "A0_PROTOCOL_FINAL")
    A1_PROTOCOL_FINAL = _load_protocol_variant(A1_PROTOCOL_FINAL_PATH, "A1_PROTOCOL_FINAL")
    A2_PROTOCOL_FINAL = _load_protocol_variant(A2_PROTOCOL_FINAL_PATH, "A2_PROTOCOL_FINAL")
else:
    A0_PROTOCOL_FINAL = None
    A1_PROTOCOL_FINAL = None
    A2_PROTOCOL_FINAL = None

TIER_MASK_COL = {
    "router": "router__obs_present",
    "ota": "ota__obs_present",
    "zigbee": "zigbee__obs_present",
    "zwave": "zwave__obs_present",
}

def _infer_protocol_tier_mask_from_frame(
    protocol_df: Optional[pd.DataFrame],
    tier: str,
    *,
    reference_df: Optional[pd.DataFrame] = None,
    label: str = "protocol",
) -> np.ndarray:
    """
    Infer active tier mask for protocol-only artifacts.

    Priority:
    1. Explicit <tier>__obs_present column if present.
    2. Any finite value among protocol value columns for that tier.
    3. Reference dataframe explicit obs mask if supplied.
    4. Zero mask only for absent zwave; otherwise fail.

    This is required because Cell 11 final protocol artifacts are value-only
    with 54 protocol columns and may not include obs_present columns.
    """
    tier = str(tier)
    obs_col = TIER_MASK_COL.get(tier)

    if protocol_df is not None and obs_col in protocol_df.columns:
        return _to_binary01_array(protocol_df[obs_col]).astype(bool)

    if protocol_df is not None:
        tier_value_cols = [
            c for c in protocol_df.columns
            if str(c).startswith(f"{tier}__")
            and str(c) != obs_col
            and not str(c).endswith("__obs_present")
        ]

        if tier_value_cols:
            block = protocol_df[tier_value_cols].apply(pd.to_numeric, errors="coerce")
            mask = np.isfinite(block.to_numpy(dtype=np.float64, copy=False)).any(axis=1)

            if mask.shape[0] != N_TE:
                raise RuntimeError(
                    f"[Cell12.a] {label}:{tier} inferred mask length mismatch: "
                    f"{mask.shape[0]} vs N_TE={N_TE}"
                )

            return mask.astype(bool, copy=False)

    if reference_df is not None and obs_col in reference_df.columns:
        return _to_binary01_array(reference_df[obs_col]).astype(bool)

    if tier == "zwave":
        return np.zeros(N_TE, dtype=bool)

    raise RuntimeError(
        f"[Cell12.a] Cannot infer protocol active mask for tier={tier!r} from {label}. "
        "Expected explicit obs_present column or finite value columns."
    )


PROTOCOL_SYN_MASKS_TEST = {}
PROTOCOL_REAL_MASKS_TEST = {}

if A2_PROTOCOL_FINAL is not None:
    for tier in ["router", "ota", "zigbee", "zwave"]:
        PROTOCOL_SYN_MASKS_TEST[tier] = _infer_protocol_tier_mask_from_frame(
            A2_PROTOCOL_FINAL,
            tier,
            reference_df=None,
            label="A2_PROTOCOL_FINAL",
        )

if A0_PROTOCOL_FINAL is not None:
    for tier in ["router", "ota", "zigbee", "zwave"]:
        PROTOCOL_REAL_MASKS_TEST[tier] = _infer_protocol_tier_mask_from_frame(
            A0_PROTOCOL_FINAL,
            tier,
            reference_df=df_te,
            label="A0_PROTOCOL_FINAL",
        )

PROTOCOL_SYN_MASK_RATES_TEST = {
    t: float(np.mean(m)) for t, m in PROTOCOL_SYN_MASKS_TEST.items()
}
PROTOCOL_REAL_MASK_RATES_TEST = {
    t: float(np.mean(m)) for t, m in PROTOCOL_REAL_MASKS_TEST.items()
}
PROTOCOL_SYN_MINUS_REAL_MASK_DRIFT_TEST = {
    t: float(PROTOCOL_SYN_MASK_RATES_TEST.get(t, np.nan) - PROTOCOL_REAL_MASK_RATES_TEST.get(t, np.nan))
    for t in sorted(set(PROTOCOL_SYN_MASK_RATES_TEST) | set(PROTOCOL_REAL_MASK_RATES_TEST))
}

log(
    "[Cell12.a] Loaded Cell 11 protocol context | "
    f"A2={None if A2_PROTOCOL_FINAL is None else A2_PROTOCOL_FINAL.shape} | "
    f"syn_mask_rates={PROTOCOL_SYN_MASK_RATES_TEST} | "
    f"syn_minus_real_mask_drift={PROTOCOL_SYN_MINUS_REAL_MASK_DRIFT_TEST}"
)


def _build_protocol_context_features(protocol_df: Optional[pd.DataFrame], label: str) -> pd.DataFrame:
    if protocol_df is None:
        return pd.DataFrame(index=df_te.index)

    dfp = protocol_df.reset_index(drop=True)
    ctx = pd.DataFrame(index=np.arange(len(dfp)))

    for tier, mask_col in TIER_MASK_COL.items():
        if label == "A2_PROTOCOL_FINAL" and tier in PROTOCOL_SYN_MASKS_TEST:
            m = PROTOCOL_SYN_MASKS_TEST[tier].astype(np.float32)
        elif label == "A0_PROTOCOL_FINAL" and tier in PROTOCOL_REAL_MASKS_TEST:
            m = PROTOCOL_REAL_MASKS_TEST[tier].astype(np.float32)
        elif mask_col in dfp.columns:
            m = _to_binary01_array(dfp[mask_col]).astype(np.float32)
        else:
            m = _infer_protocol_tier_mask_from_frame(
                dfp,
                tier,
                reference_df=None,
                label=label,
            ).astype(np.float32)
    
        ctx[f"protoctx__{tier}__obs_present"] = m
        for w in CFG12["protocol_context_roll_windows"]:
            w = int(w)
            ctx[f"protoctx__{tier}__obs_rate_{w}"] = (
                pd.Series(m).rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32)
            )

    protocol_value_cols = [
        c for c in dfp.columns
        if (
            c.startswith("router__")
            or c.startswith("ota__")
            or c.startswith("zigbee__")
            or c.startswith("zwave__")
        )
        and not c.endswith("__obs_present")
    ]

    for tier in ["router", "ota", "zigbee", "zwave"]:
        cols = [c for c in protocol_value_cols if c.startswith(f"{tier}__")]
        if not cols:
            ctx[f"protoctx__{tier}__value_activity"] = np.zeros(len(dfp), dtype=np.float32)
            continue

        block = dfp[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
        vals = block.to_numpy(dtype=np.float32, copy=False)
        value_activity = np.log1p(np.maximum(vals, 0.0).sum(axis=1)).astype(np.float32)
        ctx[f"protoctx__{tier}__value_activity"] = value_activity

        for w in CFG12["protocol_context_roll_windows"]:
            w = int(w)
            ctx[f"protoctx__{tier}__value_activity_{w}"] = (
                pd.Series(value_activity).rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32)
            )

    if TIME_COL in df_te.columns:
        tod = _time_of_day_features(df_te, TIME_COL).reset_index(drop=True)
        ctx["protoctx__tod_sin"] = tod["tod_sin"].to_numpy(dtype=np.float32)
        ctx["protoctx__tod_cos"] = tod["tod_cos"].to_numpy(dtype=np.float32)

    ctx = ctx.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    ctx.index = df_te.index

    log(f"[Cell12.a] Built protocol context features for {label}: shape={ctx.shape}")
    return ctx


PROTOCOL_CONTEXT_TEST = _build_protocol_context_features(A2_PROTOCOL_FINAL, "A2_PROTOCOL_FINAL")
globals()["PROTOCOL_CONTEXT_TEST"] = PROTOCOL_CONTEXT_TEST
globals()["PROTOCOL_SYN_MASKS_TEST"] = PROTOCOL_SYN_MASKS_TEST
globals()["PROTOCOL_REAL_MASKS_TEST"] = PROTOCOL_REAL_MASKS_TEST
globals()["PROTOCOL_SYN_MASK_RATES_TEST"] = PROTOCOL_SYN_MASK_RATES_TEST
globals()["PROTOCOL_REAL_MASK_RATES_TEST"] = PROTOCOL_REAL_MASK_RATES_TEST
globals()["PROTOCOL_SYN_MINUS_REAL_MASK_DRIFT_TEST"] = PROTOCOL_SYN_MINUS_REAL_MASK_DRIFT_TEST

# ----------------------------------------------------------
# 4) Canonical IoT contracts from Cell 4.5 / 6.5
# ----------------------------------------------------------
FULL_IOT_NAMESPACE_COLS = list(globals()["IOT_FULL_NAMESPACE_COLS"])

DRV_COLS = list(globals()["IOT_DRIVER_COLS"])
BIN_COLS = list(globals()["IOT_BIN_COLS"])
CONT_VALUE_COLS = list(globals()["IOT_CONT_COLS"])
STATE_TARGET_COLS = list(globals()["IOT_STATE_COLS"])
IOT_META = list(globals()["IOT_META_COLS"])
IOT_EXCLUDED = list(globals()["IOT_EXCLUDED_COLS"])

IOT_COUNTLIKE_COLS_12A = list(globals()["IOT_COUNTLIKE_COLS"])
IOT_TELEMETRY_COLS_12A = list(globals()["IOT_TELEMETRY_COLS"])
IOT_STEP_PROGRESS_COLS_12A = list(globals()["IOT_STEP_PROGRESS_COLS"])
IOT_DISCRETE_STATE_COLS_12A = list(globals()["IOT_DISCRETE_STATE_COLS"])
IOT_SETTINGLIKE_COLS_12A = list(globals()["IOT_SETTINGLIKE_COLS"])

# Derived target aliases from the v13 role-ownership contract.
IOT_DRIVER_TARGETS = list(DRV_COLS)
IOT_BINARY_TARGETS = list(BIN_COLS)
IOT_CONTINUOUS_VALUE_TARGETS = list(CONT_VALUE_COLS)
IOT_STATE_TARGETS = list(STATE_TARGET_COLS)
IOT_NONMODELED_META_COLS = list(IOT_META)
IOT_NONMODELED_EXCLUDED_COLS = list(IOT_EXCLUDED)

if set(IOT_STATE_TARGETS) != (set(IOT_BINARY_TARGETS) | set(IOT_CONTINUOUS_VALUE_TARGETS)):
    raise RuntimeError(
        "[Cell12.a] IOT_STATE_TARGETS must equal binary targets ∪ continuous value targets."
    )

if set(FULL_IOT_NAMESPACE_COLS) != (
    set(IOT_DRIVER_TARGETS)
    | set(IOT_BINARY_TARGETS)
    | set(IOT_CONTINUOUS_VALUE_TARGETS)
    | set(IOT_NONMODELED_META_COLS)
    | set(IOT_NONMODELED_EXCLUDED_COLS)
):
    raise RuntimeError(
        "[Cell12.a] Full IoT namespace is not exhausted by driver/binary/continuous/meta/excluded."
    )

_top_level_overlap = {
    "driver_binary": sorted(set(IOT_DRIVER_TARGETS) & set(IOT_BINARY_TARGETS)),
    "driver_continuous": sorted(set(IOT_DRIVER_TARGETS) & set(IOT_CONTINUOUS_VALUE_TARGETS)),
    "driver_meta": sorted(set(IOT_DRIVER_TARGETS) & set(IOT_NONMODELED_META_COLS)),
    "driver_excluded": sorted(set(IOT_DRIVER_TARGETS) & set(IOT_NONMODELED_EXCLUDED_COLS)),
    "binary_continuous": sorted(set(IOT_BINARY_TARGETS) & set(IOT_CONTINUOUS_VALUE_TARGETS)),
    "binary_meta": sorted(set(IOT_BINARY_TARGETS) & set(IOT_NONMODELED_META_COLS)),
    "binary_excluded": sorted(set(IOT_BINARY_TARGETS) & set(IOT_NONMODELED_EXCLUDED_COLS)),
    "continuous_meta": sorted(set(IOT_CONTINUOUS_VALUE_TARGETS) & set(IOT_NONMODELED_META_COLS)),
    "continuous_excluded": sorted(set(IOT_CONTINUOUS_VALUE_TARGETS) & set(IOT_NONMODELED_EXCLUDED_COLS)),
    "meta_excluded": sorted(set(IOT_NONMODELED_META_COLS) & set(IOT_NONMODELED_EXCLUDED_COLS)),
}

_bad_overlap = {k: v for k, v in _top_level_overlap.items() if len(v) > 0}
if _bad_overlap:
    raise RuntimeError(f"[Cell12.a] Top-level IoT ownership overlap detected: {_bad_overlap}")

IOT_PHASE2_TARGET_PLAN = {
    "version": "cell12a_v24_phase2_target_plan",
    "full_iot_namespace_n": int(len(FULL_IOT_NAMESPACE_COLS)),
    "driver_targets_n": int(len(IOT_DRIVER_TARGETS)),
    "binary_targets_n": int(len(IOT_BINARY_TARGETS)),
    "continuous_value_targets_n": int(len(IOT_CONTINUOUS_VALUE_TARGETS)),
    "state_targets_n": int(len(IOT_STATE_TARGETS)),
    "meta_cols_n": int(len(IOT_NONMODELED_META_COLS)),
    "excluded_cols_n": int(len(IOT_NONMODELED_EXCLUDED_COLS)),
    "driver_targets": list(IOT_DRIVER_TARGETS),
    "binary_targets": list(IOT_BINARY_TARGETS),
    "continuous_value_targets": list(IOT_CONTINUOUS_VALUE_TARGETS),
    "state_targets": list(IOT_STATE_TARGETS),
    "meta_cols": list(IOT_NONMODELED_META_COLS),
    "excluded_cols": list(IOT_NONMODELED_EXCLUDED_COLS),
    "quality_dimensions_available": sorted(map(str, _qdims.keys())),
    "a0_role": _a0_role,
    "a1_role": _a1_role,
    "a2_role": _a2_role,
    "iot_synthesis_scope": {
        "drivers": "generate_or_materialize_as event-driver target streams",
        "binary": "generate as strict 0/1 IoT state targets",
        "continuous": "generate as typed continuous/count/discrete value targets",
        "meta": "do not model as target; use only as masks/conditioning/bookkeeping where allowed",
        "excluded": "do not model; no TRAIN/VAL support or non-modelable",
    },
}

for nm, _df in [("TRAIN", df_tr), ("VAL", df_val), ("TEST", df_te)]:
    miss_drv = [c for c in DRV_COLS if c not in _df.columns]
    miss_bin = [c for c in BIN_COLS if c not in _df.columns]
    miss_cont = [c for c in CONT_VALUE_COLS if c not in _df.columns]
    miss_meta = [c for c in IOT_META if c not in _df.columns]
    if miss_drv or miss_bin or miss_cont or miss_meta:
        raise RuntimeError(
            f"[Cell12.a] Split schema drift in {nm}: "
            f"missing_drv={miss_drv[:10]} "
            f"missing_bin={miss_bin[:10]} "
            f"missing_cont={miss_cont[:10]} "
            f"missing_meta={miss_meta[:10]}"
        )

CONT_MASK_COLS = [c for c in ["iot__obs_present"] if c in IOT_META]
CONT_COLS = list(CONT_MASK_COLS) + list(CONT_VALUE_COLS)

log(
    f"[Cell12.a] Canonical IoT role-ownership contract | "
    f"full_namespace={len(FULL_IOT_NAMESPACE_COLS)} | "
    f"drivers={len(IOT_DRIVER_TARGETS)} | "
    f"binary_targets={len(IOT_BINARY_TARGETS)} | "
    f"continuous_value_targets={len(IOT_CONTINUOUS_VALUE_TARGETS)} | "
    f"state_targets={len(IOT_STATE_TARGETS)} | "
    f"meta={len(IOT_NONMODELED_META_COLS)} | "
    f"excluded={len(IOT_NONMODELED_EXCLUDED_COLS)} | "
    f"cont_masks={len(CONT_MASK_COLS)}"
)

# ----------------------------------------------------------
# 5) Load IoT matrices
# ----------------------------------------------------------
log("[Cell12.a] Loading TRAIN/VAL/TEST IoT matrices...")

X_drv_tr = _safe_numeric_frame(df_tr, DRV_COLS)
X_bin_tr = _safe_binary_frame(df_tr, BIN_COLS)
X_cont_tr = _safe_numeric_frame(df_tr, CONT_COLS)

X_drv_val = _safe_numeric_frame(df_val, DRV_COLS)
X_bin_val = _safe_binary_frame(df_val, BIN_COLS)
X_cont_val = _safe_numeric_frame(df_val, CONT_COLS)

# TEST matrices are evaluation/schema-only in this foundation cell.
X_drv_te = _safe_numeric_frame(df_te, DRV_COLS)
X_bin_te = _safe_binary_frame(df_te, BIN_COLS)
X_cont_te = _safe_numeric_frame(df_te, CONT_COLS)

log(f"[Cell12.a] TRAIN matrices: drv={X_drv_tr.shape} | bin={X_bin_tr.shape} | cont={X_cont_tr.shape}")
log(f"[Cell12.a] VAL matrices:   drv={X_drv_val.shape} | bin={X_bin_val.shape} | cont={X_cont_val.shape}")
log(f"[Cell12.a] TEST matrices:  drv={X_drv_te.shape} | bin={X_bin_te.shape} | cont={X_cont_te.shape}")
log(f"[Cell12.a] N_TR={N_TR} | N_VAL={N_VAL} | N_TE={N_TE} | window_fits={window_fits}")

# ----------------------------------------------------------
# 6) Entity helpers
# ----------------------------------------------------------
def _infer_entity_from_col(col: str) -> str:
    s = str(col).lower()

    if s.startswith("iot__"):
        s2 = s[len("iot__"):]
    else:
        s2 = s

    if s2.startswith("events_in_sec__entity__"):
        return s2.split("events_in_sec__entity__", 1)[1]

    if s2.startswith("events_in_sec__feat__"):
        parts = s2.split("__")
        if len(parts) >= 3:
            return parts[2]

    parts = s2.split("__")
    ent = parts[0] if parts else "misc"

    if ent.startswith("coffee_maker_") and ent not in {
        "coffee_maker_milk",
        "coffee_maker_hot",
        "coffee_maker_local_control",
    }:
        ent = "coffee_maker"

    suffixes = [
        "connectivity", "remote_start", "local_control", "bean_container_empty",
        "water_tank_empty", "drip_tray_full", "cloud_connection",
        "auto_update_enabled", "led",
    ]
    for h in suffixes:
        suf = "_" + h
        if ent.endswith(suf):
            base = ent[: -len(suf)]
            if base:
                ent = base
                break

    return ent if ent else "misc"


ALL_IOT_COLS = sorted(set(DRV_COLS) | set(BIN_COLS) | set(CONT_VALUE_COLS))
ENTITY_MAP = {c: _infer_entity_from_col(c) for c in ALL_IOT_COLS}

ENTITY_TO_DRV = defaultdict(list)
ENTITY_TO_BIN = defaultdict(list)
ENTITY_TO_CONT = defaultdict(list)

for c in DRV_COLS:
    ENTITY_TO_DRV[ENTITY_MAP[c]].append(c)
for c in BIN_COLS:
    ENTITY_TO_BIN[ENTITY_MAP[c]].append(c)
for c in CONT_VALUE_COLS:
    ENTITY_TO_CONT[ENTITY_MAP[c]].append(c)


def _related_driver_cols_for_entity(entity: str):
    entity = str(entity)
    cols = list(ENTITY_TO_DRV.get(entity, []))
    if cols:
        return sorted(set(cols))

    out = []
    for c in DRV_COLS:
        try:
            ent = _infer_entity_from_col(c)
        except Exception:
            ent = None
        if ent == entity:
            out.append(c)
    return sorted(set(out))


def _entity_activity_from_driver_df(df_drv: pd.DataFrame, entity: str) -> np.ndarray:
    if not isinstance(df_drv, pd.DataFrame):
        raise TypeError("[Cell12.a] df_drv must be a pandas DataFrame.")
    rel_cols = [c for c in _related_driver_cols_for_entity(entity) if c in df_drv.columns]
    if len(rel_cols) == 0:
        return np.zeros(len(df_drv), dtype=np.int8)
    block = df_drv[rel_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (block.to_numpy(dtype=np.float32, copy=False).sum(axis=1) > 0.0).astype(np.int8)

# ----------------------------------------------------------
# 7) Family routing
# ----------------------------------------------------------
def _infer_cont_family_strict(col: str):
    s = str(col).lower()

    if re.search(r"__sensor__programme_progress__value$", s):
        return "step_progress"

    if s in {"iot__events_total", "iot__events_entity_unique"}:
        return "aux_global_counter"

    if "s5_max" in s:
        return "s5"

    if any(k in s for k in [
        "today_s_consumption",
        "this_month_s_consumption",
        "total_consumption",
        "today_s__sensor__consumption",
        "this_month_s__sensor__consumption",
        "total__sensor__consumption",
        "total_cleaning_area",
        "total_cleaning_time",
        "total_cleaning_count",
        "cleaning_area",
        "cleaning_time",
        "power_outage_count",
        "trigger_count",
        "open_count",
        "close_count",
        "motion_count",
        "press_count",
        "coffees",
        "cups",
        "coffee_and_milk_cups",
        "hot_water_cups",
        "count__value",
    ]):
        return "cumulative_count"

    if re.search(r"__sensor__(battery|voltage|linkquality|rssi|lqi|signal_strength|signal_level)__value$", s):
        return "device_telemetry"
    if re.search(r"__sensor__device_temperature__value$", s):
        return "device_telemetry"
    if "battery" in s and s.endswith("__value"):
        return "device_telemetry"
    if "rain_gauge_battery" in s:
        return "device_telemetry"
    if "strength" in s and s.endswith("__value"):
        return "device_telemetry"

    if re.search(r"__sensor__temperature__value$", s):
        return "temperature"
    if re.search(r"__sensor__humidity__value$", s):
        return "humidity"
    if re.search(r"__sensor__pressure__value$", s):
        return "pressure"
    if re.search(r"__sensor__(illuminance|luminance|brightness|lux)__value$", s):
        return "light"
    if re.search(r"__sensor__(carbon_dioxide|co2|pm25|pm10|voc|purity|air_quality)__value$", s):
        return "environmental"

    if re.search(r"__sensor__(current|power|energy|watt|ampere)__value$", s):
        return "power_energy"
    if re.search(r"__sensor__current_consumption__value$", s):
        return "power_energy"
    if "current__sensor__consumption" in s:
        return "power_energy"

    if any(k in s for k in ["steps", "distance", "calories", "heart_rate", "sleep", "activity"]):
        return "activity_tracker"
    if any(k in s for k in [
        "respiratory_rate",
        "snoring",
        "wakeup_count",
        "wakeup_time",
        "time_to_wakeup",
        "breathing_disturbances",
    ]):
        return "activity_tracker"

    if "__light__state__brightness" in s:
        return "light"
    if any(k in s for k in ["__light__state__color_temp", "__light__state__color_temp_kelvin"]):
        return "light"

    if any(k in s for k in [
        "anemometer_gust_angle",
        "anemometer_wind_angle",
        "anemometer_gust_strength",
        "anemometer_wind_speed",
        "rain_gauge_precipitation",
        "rain_gauge_precipitation_last_hour",
        "rain_gauge_precipitation_today",
        "__sensor__noise__value",
    ]):
        return "other"

    if any(k in s for k in [
        "x_axis", "y_axis", "z_axis",
        "angle_x", "angle_y", "angle_z",
        "direction", "angle",
        "rgb_color", "volume_level",
        "time_left", "fill_quantity", "number__state__value",
        "sensor__water__value",
        "power_protection",
        "occupancy_timeout",
        "sensitivity",
    ]):
        return "other"

    if "temperature" in s:
        return "temperature"
    if "humidity" in s:
        return "humidity"
    if "pressure" in s:
        return "pressure"
    if any(k in s for k in ["illumin", "lux", "luminance", "brightness"]):
        return "light"
    if any(k in s for k in ["carbon_dioxide", "co2", "purity", "air_quality", "voc", "pm25", "pm10"]):
        return "environmental"
    if any(k in s for k in ["current", "power", "energy"]):
        return "power_energy"
    if any(k in s for k in ["voltage", "linkquality", "rssi", "lqi", "signal_strength", "signal_level"]):
        return "device_telemetry"

    return "other"


def _infer_other_subtype(col: str):
    s = str(col).lower()

    if any(k in s for k in ["x_axis", "y_axis", "z_axis"]):
        return "axis"
    if any(k in s for k in ["angle_x", "angle_y", "angle_z", "wind_angle", "gust_angle", "direction", "angle"]):
        return "angle"
    if any(k in s for k in ["anemometer_wind_speed", "anemometer_gust_strength", "wind_speed", "gust_strength"]):
        return "weather_flow"
    if any(k in s for k in ["rain_gauge_precipitation", "precipitation", "rain"]):
        return "precipitation"
    if "noise" in s:
        return "acoustic"
    if "rgb_color" in s:
        return "color_state"
    if "volume_level" in s:
        return "media_state"
    if "time_left" in s:
        return "remaining_life"
    if "fill_quantity" in s or "number__state__value" in s or "power_protection" in s:
        return "numeric_setting"
    if "occupancy_timeout" in s:
        return "timeout_setting"
    if "sensitivity" in s:
        return "device_setting"
    if "sensor__water__value" in s:
        return "resource_state"

    return "misc_other"


FAMILY_MAP = {c: _infer_cont_family_strict(c) for c in CONT_VALUE_COLS}

for c in CONT_VALUE_COLS:
    if "s5_max" in str(c).lower():
        FAMILY_MAP[c] = "s5"

OTHER_SUBTYPE_MAP = {
    c: _infer_other_subtype(c)
    for c in CONT_VALUE_COLS
    if FAMILY_MAP[c] == "other"
}

FAMILY_GROUPS = {}
for c, fam in FAMILY_MAP.items():
    FAMILY_GROUPS.setdefault(fam, []).append(c)

DEVICE_TELEMETRY_COLS = sorted([c for c in CONT_VALUE_COLS if FAMILY_MAP[c] == "device_telemetry"])
CUMULATIVE_COUNT_COLS = sorted([c for c in CONT_VALUE_COLS if FAMILY_MAP[c] == "cumulative_count"])
STEP_PROGRESS_COLS = sorted([c for c in CONT_VALUE_COLS if FAMILY_MAP[c] == "step_progress"])
AUX_GLOBAL_COUNTER_COLS = sorted([c for c in CONT_VALUE_COLS if FAMILY_MAP[c] == "aux_global_counter"])


def _count_subtype(col: str) -> str:
    s = str(col).lower()

    if any(k in s for k in [
        "today_s_consumption",
        "this_month_s_consumption",
        "total_consumption",
        "today_s__sensor__consumption",
        "this_month_s__sensor__consumption",
        "total__sensor__consumption",
        "total_cleaning_area",
        "total_cleaning_time",
        "total_cleaning_count",
    ]):
        return "accumulator"

    if any(k in s for k in [
        "cleaning_area",
        "cleaning_time",
        "coffee_maker__sensor__coffees",
        "coffee_maker__sensor__hot_water_cups",
        "coffee_maker__sensor__coffee_and_milk_cups",
        "coffee_maker_milk__sensor__cups",
    ]):
        return "session_counter"

    if any(k in s for k in [
        "trigger_count",
        "power_outage_count",
        "open_count",
        "close_count",
        "motion_count",
        "press_count",
        "count__value",
    ]):
        return "event_counter"

    if "total_" in s or s.startswith("iot__events_"):
        return "accumulator"

    if any(k in s for k in ["cups", "coffees", "cleaning_"]):
        return "session_counter"

    return "event_counter"


COUNT_SUBTYPE_MAP = {c: _count_subtype(c) for c in CUMULATIVE_COUNT_COLS}
SESSION_COUNT_COLS = sorted([c for c in CUMULATIVE_COUNT_COLS if COUNT_SUBTYPE_MAP[c] == "session_counter"])
ACCUMULATOR_COUNT_COLS = sorted([c for c in CUMULATIVE_COUNT_COLS if COUNT_SUBTYPE_MAP[c] == "accumulator"])
EVENT_COUNT_COLS = sorted([c for c in CUMULATIVE_COUNT_COLS if COUNT_SUBTYPE_MAP[c] == "event_counter"])


def _infer_synthesis_subfamily(col: str, family: str, other_subtype: Optional[str] = None):
    s = str(col).lower()
    fam = str(family)

    if fam == "s5":
        if any(k in s for k in ["filter_time_left", "main_brush_time_left", "side_brush_time_left", "sensor_time_left"]):
            return "s5_remaining_life"
        if any(k in s for k in ["total_cleaning_time", "total_cleaning_area", "total_cleaning_count"]):
            return "s5_lifetime_total"
        if any(k in s for k in ["cleaning_time", "cleaning_area"]):
            return "s5_periodic_counter"
        if "battery" in s:
            return "s5_battery_like"
        if "state" in s:
            return "s5_state_like"
        return "s5_other"

    if fam == "device_telemetry":
        if any(k in s for k in ["linkquality", "rssi", "lqi", "signal_strength", "signal_level"]):
            return "telemetry_linkquality_like"
        if "device_temperature" in s:
            return "telemetry_device_temperature_like"
        if "mains_voltage" in s:
            return "telemetry_mains_voltage_like"
        if any(k in s for k in ["battery", "voltage"]):
            return "telemetry_battery_voltage_like"
        return "telemetry_other"

    if fam == "environmental":
        if "purity" in s:
            return "environmental_purity"
        if any(k in s for k in ["co2", "carbon_dioxide"]):
            return "environmental_co2"
        if any(k in s for k in ["voc", "pm25", "pm10", "air_quality"]):
            return "environmental_air_quality"
        return "environmental_other"

    if fam == "light":
        if "illuminance" in s or "lux" in s or "luminance" in s:
            return "light_illuminance"
        if "brightness" in s:
            return "light_brightness"
        if "color_temp" in s:
            return "light_color_temp"
        return "light_other"

    if fam == "activity_tracker":
        if "snoring" in s:
            return "activity_snoring_state"
        if any(k in s for k in ["sleep", "wakeup", "breathing_disturbances", "respiratory_rate"]):
            return "activity_sleep_like"
        return "activity_other"

    if fam == "humidity":
        return "humidity_scalar"
    if fam == "temperature":
        return "temperature_scalar"
    if fam == "pressure":
        return "pressure_scalar"

    if fam == "power_energy":
        if "current" in s:
            return "power_current_like"
        if "power" in s:
            return "power_power_like"
        if "energy" in s or "consumption" in s:
            return "power_energy_like"
        return "power_other"

    if fam == "cumulative_count":
        csub = COUNT_SUBTYPE_MAP.get(col, None)
        if csub == "accumulator":
            return "count_accumulator"
        if csub == "session_counter":
            return "count_session_like"
        return "count_event_like"

    if fam == "step_progress":
        return "step_progress"
    if fam == "aux_global_counter":
        return "aux_global_counter"

    ost = str(other_subtype) if other_subtype is not None else _infer_other_subtype(col)
    if ost == "axis":
        return "other_axis_inertial"
    if ost == "angle":
        return "other_angle_direction"
    if ost == "weather_flow":
        return "other_weather_flow"
    if ost == "precipitation":
        return "other_precipitation"
    if ost == "acoustic":
        return "other_acoustic"
    if ost == "media_state":
        return "other_media_state"
    if ost == "remaining_life":
        return "other_remaining_life"
    if ost == "numeric_setting":
        return "other_numeric_setting"
    if ost == "timeout_setting":
        return "other_timeout_setting"
    if ost == "device_setting":
        return "other_device_setting"
    if ost == "resource_state":
        return "other_resource_state"
    if ost == "color_state":
        return "other_color_state"
    return "other_misc"


SYNTHESIS_SUBFAMILY_MAP = {
    c: _infer_synthesis_subfamily(c, FAMILY_MAP[c], OTHER_SUBTYPE_MAP.get(c, None))
    for c in CONT_VALUE_COLS
}

OTHER_SUBTYPE_COUNTS = pd.Series(list(OTHER_SUBTYPE_MAP.values())).value_counts().to_dict()
SYNTHESIS_SUBFAMILY_COUNTS = pd.Series(list(SYNTHESIS_SUBFAMILY_MAP.values())).value_counts().to_dict()

log(f"[Cell12.a] Other-subtype breakdown: {OTHER_SUBTYPE_COUNTS}")
log(f"[Cell12.a] Synthesis-subfamily breakdown: {SYNTHESIS_SUBFAMILY_COUNTS}")

# ----------------------------------------------------------
# 8) Routing-first continuous policy
# ----------------------------------------------------------
IOT_DISCRETE_STATE_COLS = list(globals().get("IOT_DISCRETE_STATE_COLS", []))
IOT_SETTINGLIKE_COLS = list(globals().get("IOT_SETTINGLIKE_COLS", []))

SET_DISCRETE_STATE = set(IOT_DISCRETE_STATE_COLS)
SET_SETTINGLIKE = set(IOT_SETTINGLIKE_COLS)
SET_TELEMETRY = set(DEVICE_TELEMETRY_COLS)
SET_COUNTLIKE = set(CUMULATIVE_COUNT_COLS)
SET_STEP_PROGRESS = set(STEP_PROGRESS_COLS)


def _const_status_trva(col: str) -> str:
    vals = []
    if col in X_cont_tr.columns:
        vals.append(pd.to_numeric(X_cont_tr[col], errors="coerce").dropna().to_numpy(dtype=np.float32))
    if col in X_cont_val.columns:
        vals.append(pd.to_numeric(X_cont_val[col], errors="coerce").dropna().to_numpy(dtype=np.float32))
    vals = [v for v in vals if len(v)]
    if not vals:
        return "all_nan"
    x = np.concatenate(vals).astype(np.float32)
    u = np.unique(np.round(x, 6))
    if len(u) == 1:
        return "constant"
    if len(u) <= 2:
        return "binary_or_two_point"
    return "variable"


def _ddpm_route_allowed(col: str, fam: str, ssf: str, const_status: str) -> bool:
    """
    Conservative DDPM routing.
    Specialized physical/semantic families are intentionally not sent to DDPM
    here. They are handled by Cell 12.c family-aware generators.
    """
    s = str(col).lower()

    if const_status in {"all_nan", "constant", "binary_or_two_point"}:
        return False

    if fam in {
        "s5",
        "device_telemetry",
        "cumulative_count",
        "step_progress",
        "aux_global_counter",
        "power_energy",
        "activity_tracker",
        "light",
        "environmental",
        "other",
    }:
        return False

    if fam in {"temperature", "humidity", "pressure"}:
        return True

    return False


CONT_ROUTING_POLICY_ROWS = []

for c in CONT_VALUE_COLS:
    fam = FAMILY_MAP.get(c, "other")
    ssf = SYNTHESIS_SUBFAMILY_MAP.get(c, "")
    other_sub = OTHER_SUBTYPE_MAP.get(c, "")
    const_status = _const_status_trva(c)
    is_constant_trva = const_status == "constant"
    lc = str(c).lower()

    policy_bucket = None
    policy_generator_family = None
    expected_stage12b_route = "non_ddpm"
    expected_stage12c_kind = None
    policy_reason = None

    if fam == "activity_tracker":
        if "__withings_snoring__sensor__state__value" in lc:
            policy_bucket = "family_specialized"
            policy_generator_family = "SnoringGenerator"
            expected_stage12c_kind = "snoring_specialized"
            policy_reason = "withings snoring specialized override"
        elif any(k in lc for k in [
            "__withings_deep_sleep__sensor__state__value",
            "__withings_light_sleep__sensor__state__value",
            "__withings_rem_sleep__sensor__state__value",
        ]):
            policy_bucket = "discrete_state"
            policy_generator_family = "DiscreteStateGenerator"
            expected_stage12c_kind = "state_dwell_discrete"
            policy_reason = "withings sleep-stage discrete override"
        elif "__withings_" in lc:
            policy_bucket = "family_specialized"
            policy_generator_family = "WithingsSummaryGenerator"
            expected_stage12c_kind = "withings_summary_specialized"
            policy_reason = "withings summary specialized override"
        else:
            policy_bucket = "family_specialized"
            policy_generator_family = "CircadianActivityGenerator"
            expected_stage12c_kind = "circadian_activity"
            policy_reason = "activity tracker specialized override"

    elif fam == "power_energy":
        policy_bucket = "family_specialized"
        policy_generator_family = "PowerEpisodeReplayGenerator"
        expected_stage12c_kind = "power_episode_replay"
        policy_reason = "power-energy specialized override"

    elif fam == "s5":
        policy_bucket = "family_specialized"
        policy_generator_family = "S5Specialized"
        expected_stage12c_kind = "s5_specialized"
        policy_reason = "S5 family override"

    elif ssf == "other_precipitation":
        policy_bucket = "family_specialized"
        policy_generator_family = "PrecipitationGenerator"
        expected_stage12c_kind = "sparse_event_rollup"
        policy_reason = "precipitation specialized override"

    elif fam == "environmental":
        policy_bucket = "family_specialized"
        policy_generator_family = "EnvironmentalGenerator"
        expected_stage12c_kind = "environmental_bounded"
        policy_reason = "environmental specialized override"

    elif fam == "light":
        policy_bucket = "family_specialized"
        policy_generator_family = "LightPatternGenerator"
        expected_stage12c_kind = "light_pattern"
        policy_reason = "light specialized override"

    elif _ddpm_route_allowed(c, fam, ssf, const_status):
        policy_bucket = "ddpm"
        policy_generator_family = "DDPM"
        expected_stage12b_route = "ddpm"
        expected_stage12c_kind = "ddpm_passthrough"
        policy_reason = "smooth scalar family eligible for DDPM"

    elif c in SET_TELEMETRY:
        policy_bucket = "telemetry"
        policy_generator_family = "TelemetryGenerator"
        expected_stage12c_kind = "telemetry_subfamily_specific"
        policy_reason = "telemetry subtype"

    elif c in SET_COUNTLIKE:
        policy_bucket = "countlike"
        policy_generator_family = "CountProcessGenerator"
        expected_stage12c_kind = "counter"
        policy_reason = "countlike subtype"

    elif c in SET_STEP_PROGRESS:
        policy_bucket = "step_progress"
        policy_generator_family = "ProgressGenerator"
        expected_stage12c_kind = "coffee_episode_progress"
        policy_reason = "step_progress subtype"

    elif c in SET_SETTINGLIKE and is_constant_trva:
        policy_bucket = "setting_like_constant"
        policy_generator_family = "ConstantOrQuasiStaticReplay"
        expected_stage12c_kind = "quasi_static"
        policy_reason = "setting_like + constant"

    elif c in SET_DISCRETE_STATE:
        policy_bucket = "discrete_state"
        policy_generator_family = "DiscreteStateGenerator"
        expected_stage12c_kind = "state_dwell_discrete"
        policy_reason = "discrete_state subtype"

    else:
        policy_bucket = "dynamic_residual_continuous"
        policy_generator_family = "ContinuousReplayOrFamilyFallback"
        expected_stage12c_kind = "context_continuous"
        policy_reason = "dynamic residual continuous fallback"

    CONT_ROUTING_POLICY_ROWS.append({
        "col": c,
        "family": fam,
        "other_subtype": other_sub,
        "synthesis_subfamily": ssf,
        "const_status_trva": const_status,
        "is_telemetry": bool(c in SET_TELEMETRY),
        "is_countlike": bool(c in SET_COUNTLIKE),
        "is_step_progress": bool(c in SET_STEP_PROGRESS),
        "is_discrete_state": bool(c in SET_DISCRETE_STATE),
        "is_setting_like": bool(c in SET_SETTINGLIKE),
        "policy_bucket": policy_bucket,
        "policy_generator_family": policy_generator_family,
        "expected_stage12b_route": expected_stage12b_route,
        "expected_stage12c_kind": expected_stage12c_kind,
        "policy_reason": policy_reason,
    })

CONT_ROUTING_POLICY_DF = pd.DataFrame(CONT_ROUTING_POLICY_ROWS).sort_values(
    ["policy_bucket", "family", "col"]
).reset_index(drop=True)

DDPM_COLS = sorted(
    CONT_ROUTING_POLICY_DF.loc[
        CONT_ROUTING_POLICY_DF["expected_stage12b_route"] == "ddpm",
        "col",
    ].astype(str).tolist()
)
NON_DDPM_COLS = sorted([c for c in CONT_VALUE_COLS if c not in set(DDPM_COLS)])

cont_routing_policy_csv = os.path.join(REPORT_DIR, "cell12a_v24_1_cont_routing_policy.csv")
CONT_ROUTING_POLICY_DF.to_csv(cont_routing_policy_csv, index=False)

CONT_ROUTING_POLICY_MANIFEST = {
    "version": "cell12a_v24_1_routing_first_policy",
    "n_cont_value_cols": int(len(CONT_VALUE_COLS)),
    "n_policy_rows": int(len(CONT_ROUTING_POLICY_DF)),
    "ddpm_cols_n": int(len(DDPM_COLS)),
    "non_ddpm_cols_n": int(len(NON_DDPM_COLS)),
    "bucket_counts": CONT_ROUTING_POLICY_DF["policy_bucket"].value_counts().sort_index().to_dict(),
    "expected_stage12b_route_counts": CONT_ROUTING_POLICY_DF["expected_stage12b_route"].value_counts().sort_index().to_dict(),
    "expected_stage12c_kind_counts": CONT_ROUTING_POLICY_DF["expected_stage12c_kind"].value_counts().sort_index().to_dict(),
    "policy_note": (
        "DDPM_COLS are derived from the routing policy, not from broad family membership. "
        "Specialized families such as power, light, environmental, Withings/activity, S5, "
        "telemetry, counters, and precipitation are kept out of DDPM and handled downstream "
        "by family-aware IoT generators."
    ),
}

cont_routing_policy_manifest_json = os.path.join(REPORT_DIR, "cell12a_v24_1_cont_routing_policy_manifest.json")
_write_json(cont_routing_policy_manifest_json, CONT_ROUTING_POLICY_MANIFEST)

globals()["CONT_ROUTING_POLICY_DF"] = CONT_ROUTING_POLICY_DF
globals()["CONT_ROUTING_POLICY_CSV"] = cont_routing_policy_csv
globals()["CONT_ROUTING_POLICY_MANIFEST"] = CONT_ROUTING_POLICY_MANIFEST
globals()["CONT_ROUTING_POLICY_MANIFEST_JSON"] = cont_routing_policy_manifest_json

log(f"[Cell12.a] Continuous routing policy saved: {cont_routing_policy_csv}")
log(
    "[Cell12.a] Routing buckets | "
    + " | ".join(
        f"{k}={v}" for k, v in CONT_ROUTING_POLICY_DF["policy_bucket"].value_counts().sort_index().to_dict().items()
    )
)
log(f"[Cell12.a] DDPM continuous cols after routing-first policy: {len(DDPM_COLS)} | non-DDPM cols: {len(NON_DDPM_COLS)}")

# ----------------------------------------------------------
# 8B) Binary-target routing policy
# ----------------------------------------------------------
def _binary_rate_trva(col: str) -> float:
    xs = []
    if col in X_bin_tr.columns:
        xs.append(pd.to_numeric(X_bin_tr[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    if col in X_bin_val.columns:
        xs.append(pd.to_numeric(X_bin_val[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    if not xs:
        return 0.0
    x = np.concatenate(xs)
    return float(np.mean(x > 0.5))


def _binary_transition_rate_trva(col: str) -> float:
    xs = []
    if col in X_bin_tr.columns:
        xs.append(pd.to_numeric(X_bin_tr[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    if col in X_bin_val.columns:
        xs.append(pd.to_numeric(X_bin_val[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32))
    if not xs:
        return 0.0
    x = np.concatenate(xs)
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(x)) > 0.5))


def _infer_binary_policy_bucket(col: str, rate: float, tr_rate: float) -> Tuple[str, str, str]:
    s = str(col).lower()

    if rate <= CFG12["binary_near_constant_low"]:
        return (
            "binary_near_constant_zero",
            "BinaryQuasiStaticGenerator",
            "near-zero binary state; preserve low activation rate",
        )

    if rate >= CFG12["binary_near_constant_high"]:
        return (
            "binary_near_constant_one",
            "BinaryQuasiStaticGenerator",
            "near-one binary state; preserve high activation rate",
        )

    if tr_rate <= 1.0 / max(float(CFG12["binary_long_run_thr"]), 1.0):
        return (
            "binary_long_dwell",
            "BinarySemiMarkovDwellGenerator",
            "low transition rate; use dwell/episode process",
        )

    if any(k in s for k in ["motion", "occupancy", "presence", "detected", "open", "contact"]):
        return (
            "binary_event_state",
            "BinaryEventConditionedGenerator",
            "event-like binary state; condition on drivers/protocol/TOD",
        )

    if any(k in s for k in ["connect", "online", "available", "problem", "tamper", "battery_low"]):
        return (
            "binary_device_status",
            "BinaryStatusGenerator",
            "device status binary; preserve availability/status dynamics",
        )

    return (
        "binary_markov_state",
        "BinaryMarkovGenerator",
        "general binary state; use regime/TOD/protocol-conditioned Markov",
    )


BINARY_ROUTING_POLICY_ROWS = []

for c in BIN_COLS:
    rate = _binary_rate_trva(c)
    tr_rate = _binary_transition_rate_trva(c)
    bucket, gen_family, reason = _infer_binary_policy_bucket(c, rate, tr_rate)

    BINARY_ROUTING_POLICY_ROWS.append({
        "col": c,
        "entity": ENTITY_MAP.get(c, _infer_entity_from_col(c)),
        "binary_rate_trva": float(rate),
        "binary_transition_rate_trva": float(tr_rate),
        "policy_bucket": bucket,
        "policy_generator_family": gen_family,
        "expected_stage12b_route": "binary_state_generation",
        "expected_stage12c_kind": "binary_state_target",
        "policy_reason": reason,
    })

BINARY_ROUTING_POLICY_DF = pd.DataFrame(BINARY_ROUTING_POLICY_ROWS).sort_values(
    ["policy_bucket", "entity", "col"]
).reset_index(drop=True)

BINARY_ROUTING_POLICY_CSV = os.path.join(
    REPORT_DIR, "cell12a_v24_1_binary_routing_policy.csv"
)
BINARY_ROUTING_POLICY_DF.to_csv(BINARY_ROUTING_POLICY_CSV, index=False)

BINARY_ROUTING_POLICY_MANIFEST = {
    "version": "cell12a_v24_1_binary_routing_policy",
    "binary_targets_n": int(len(BIN_COLS)),
    "policy_rows_n": int(len(BINARY_ROUTING_POLICY_DF)),
    "bucket_counts": BINARY_ROUTING_POLICY_DF["policy_bucket"].value_counts().sort_index().to_dict(),
    "generator_family_counts": BINARY_ROUTING_POLICY_DF["policy_generator_family"].value_counts().sort_index().to_dict(),
    "policy_note": (
        "Binary IoT columns are first-class synthetic state targets. They must be generated "
        "as strict 0/1 streams in Cell 12.b using TRAIN/VAL-fitted rate, dwell, TOD, driver, "
        "entity availability, and protocol context. They are not continuous-value targets."
    ),
}

BINARY_ROUTING_POLICY_MANIFEST_JSON = os.path.join(
    REPORT_DIR, "cell12a_v24_1_binary_routing_policy_manifest.json"
)
_write_json(BINARY_ROUTING_POLICY_MANIFEST_JSON, BINARY_ROUTING_POLICY_MANIFEST)

globals()["BINARY_ROUTING_POLICY_DF"] = BINARY_ROUTING_POLICY_DF
globals()["BINARY_ROUTING_POLICY_CSV"] = BINARY_ROUTING_POLICY_CSV
globals()["BINARY_ROUTING_POLICY_MANIFEST"] = BINARY_ROUTING_POLICY_MANIFEST
globals()["BINARY_ROUTING_POLICY_MANIFEST_JSON"] = BINARY_ROUTING_POLICY_MANIFEST_JSON

log(f"[Cell12.a] Binary routing policy saved: {BINARY_ROUTING_POLICY_CSV}")
log(
    "[Cell12.a] Binary routing buckets | "
    + " | ".join(
        f"{k}={v}"
        for k, v in BINARY_ROUTING_POLICY_DF["policy_bucket"].value_counts().sort_index().to_dict().items()
    )
)

# ----------------------------------------------------------
# 9) Operative contract audits
# ----------------------------------------------------------
BIN_NON01_COLS = []
for c in BIN_COLS:
    # Validate the canonicalized binary matrix used by synthesis, not raw df_tr.
    # Raw Home Assistant state columns may be strings such as on/off or connected/disconnected.
    if c not in X_bin_tr.columns:
        BIN_NON01_COLS.append(c)
        continue

    x = pd.to_numeric(X_bin_tr[c], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if x.size == 0:
        BIN_NON01_COLS.append(c)
        continue

    u = np.unique(x)
    if len(u) > 2 or not set(u.tolist()).issubset({0.0, 1.0}):
        BIN_NON01_COLS.append(c)

BLOCK_MEMBERSHIP = defaultdict(list)
for c in DRV_COLS:
    BLOCK_MEMBERSHIP[c].append("driver")
for c in BIN_COLS:
    BLOCK_MEMBERSHIP[c].append("binary")
for c in CONT_VALUE_COLS:
    BLOCK_MEMBERSHIP[c].append("cont_value")
for c in CONT_MASK_COLS:
    BLOCK_MEMBERSHIP[c].append("cont_mask")

MULTI_BLOCK_OVERLAP = {
    c: blocks
    for c, blocks in BLOCK_MEMBERSHIP.items()
    if len(set(blocks)) > 1
}

audit_rows = []

for c in DRV_COLS:
    flags = []
    if c in MULTI_BLOCK_OVERLAP:
        flags.append("multi_block_overlap")
    audit_rows.append({
        "col": c,
        "block": "driver",
        "entity": ENTITY_MAP.get(c, "misc"),
        "family": "driver",
        "other_subtype": "",
        "synthesis_subfamily": "",
        "routing_bucket": "",
        "flags": "|".join(flags),
    })

for c in BIN_COLS:
    flags = []
    if c in BIN_NON01_COLS:
        flags.append("binary_block_but_not_binary01_after_canonicalization")
    if c in MULTI_BLOCK_OVERLAP:
        flags.append("multi_block_overlap")
    audit_rows.append({
        "col": c,
        "block": "binary",
        "entity": ENTITY_MAP.get(c, "misc"),
        "family": "binary",
        "other_subtype": "",
        "synthesis_subfamily": "",
        "routing_bucket": "",
        "flags": "|".join(flags),
    })

routing_bucket_by_col = dict(zip(CONT_ROUTING_POLICY_DF["col"], CONT_ROUTING_POLICY_DF["policy_bucket"]))

for c in CONT_VALUE_COLS:
    fam = FAMILY_MAP.get(c, "other")
    ost = OTHER_SUBTYPE_MAP.get(c, "") if fam == "other" else ""
    ssf = SYNTHESIS_SUBFAMILY_MAP.get(c, "")
    flags = []
    s = str(c).lower()

    if c in MULTI_BLOCK_OVERLAP:
        flags.append("multi_block_overlap")

    if fam != "device_telemetry":
        if ("s5_max" not in s) and any(k in s for k in [
            "linkquality", "rssi", "lqi", "signal_strength", "signal_level", "device_temperature"
        ]):
            flags.append("telemetry_like_not_routed_to_device_telemetry")
        if ("s5_max" not in s) and re.search(r"__sensor__(battery|voltage)__value$", s):
            flags.append("telemetry_like_not_routed_to_device_telemetry")

    if re.search(r"__sensor__programme_progress__value$", s) and fam != "step_progress":
        flags.append("programme_progress_not_routed_to_step_progress")

    if "s5_max" in s and fam != "s5":
        flags.append("s5_not_routed_to_s5")

    audit_rows.append({
        "col": c,
        "block": "cont_value",
        "entity": ENTITY_MAP.get(c, "misc"),
        "family": fam,
        "other_subtype": ost,
        "synthesis_subfamily": ssf,
        "routing_bucket": routing_bucket_by_col.get(c, ""),
        "flags": "|".join(flags),
    })

CLASS_AUDIT_DF = pd.DataFrame(audit_rows)
CLASS_SUSPICIOUS_DF = CLASS_AUDIT_DF[
    CLASS_AUDIT_DF["flags"].astype(str).str.len() > 0
].copy()

CLASS_AUDIT_CSV = os.path.join(REPORT_DIR, "iot_classification_audit_v24_0.csv")
CLASS_AUDIT_DF.to_csv(CLASS_AUDIT_CSV, index=False)

log(
    f"[Cell12.a] Classification audit | total_iot_cols={len(CLASS_AUDIT_DF)} | "
    f"suspicious={len(CLASS_SUSPICIOUS_DF)} | audit={CLASS_AUDIT_CSV}"
)

for _, r in CLASS_SUSPICIOUS_DF.head(int(CFG12["suspicious_print_topk"])).iterrows():
    log(
        f"[Cell12.a]   suspicious | {r['col']} | block={r['block']} | "
        f"family={r['family']} | flags={r['flags']}"
    )

log(f"[Cell12.a] Continuous families: {{ {', '.join(f'{k!r}: {len(v)}' for k, v in FAMILY_GROUPS.items())} }}")
log(
    f"[Cell12.a] Family routing summary | "
    f"device_telemetry={len(DEVICE_TELEMETRY_COLS)} | "
    f"cumulative_count={len(CUMULATIVE_COUNT_COLS)} | "
    f"aux_global_counter={len(AUX_GLOBAL_COUNTER_COLS)} | "
    f"step_progress={len(STEP_PROGRESS_COLS)} | "
    f"temperature={len(FAMILY_GROUPS.get('temperature', []))} | "
    f"humidity={len(FAMILY_GROUPS.get('humidity', []))} | "
    f"pressure={len(FAMILY_GROUPS.get('pressure', []))} | "
    f"light={len(FAMILY_GROUPS.get('light', []))} | "
    f"environmental={len(FAMILY_GROUPS.get('environmental', []))} | "
    f"power_energy={len(FAMILY_GROUPS.get('power_energy', []))} | "
    f"other={len(FAMILY_GROUPS.get('other', []))}"
)
log(
    f"[Cell12.a] Count subtype split | "
    f"session_counter={len(SESSION_COUNT_COLS)} | "
    f"accumulator={len(ACCUMULATOR_COUNT_COLS)} | "
    f"event_counter={len(EVENT_COUNT_COLS)}"
)

# ----------------------------------------------------------
# 10) State clusters
# ----------------------------------------------------------
def _build_state_feature_frame(df_bin: pd.DataFrame, df_drv: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    bin_f = df_bin.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    drv_f = df_drv.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)

    reg = pd.DataFrame(index=bin_f.index)

    bin_sum = bin_f.sum(axis=1).astype(np.float32)
    bin_any = (bin_sum > 0).astype(np.float32)
    bin_change = bin_f.diff().abs().fillna(0.0).astype(np.float32)
    bin_change_sum = bin_change.sum(axis=1).astype(np.float32)
    bin_change_any = (bin_change_sum > 0).astype(np.float32)

    reg["bin_sum"] = bin_sum
    reg["bin_any"] = bin_any
    reg["bin_change_sum"] = bin_change_sum
    reg["bin_change_any"] = bin_change_any
    reg["bin_change_rate_30"] = bin_change_any.rolling(30, min_periods=1).mean().astype(np.float32)
    reg["bin_change_rate_300"] = bin_change_any.rolling(300, min_periods=1).mean().astype(np.float32)
    reg["bin_change_mass_30"] = bin_change_sum.rolling(30, min_periods=1).mean().astype(np.float32)
    reg["bin_change_mass_300"] = bin_change_sum.rolling(300, min_periods=1).mean().astype(np.float32)
    reg["bin_mean_30"] = bin_sum.rolling(30, min_periods=1).mean().astype(np.float32)
    reg["bin_mean_300"] = bin_sum.rolling(300, min_periods=1).mean().astype(np.float32)
    reg["bin_var_30"] = bin_f.rolling(30, min_periods=1).std().fillna(0.0).sum(axis=1).astype(np.float32)

    drv_sum = drv_f.sum(axis=1).astype(np.float32)
    drv_any = (drv_sum > 0).astype(np.float32)
    reg["drv_sum"] = drv_sum
    reg["drv_any"] = drv_any
    reg["drv_rate_30"] = drv_any.rolling(30, min_periods=1).mean().astype(np.float32)
    reg["drv_rate_300"] = drv_any.rolling(300, min_periods=1).mean().astype(np.float32)

    entity_activity_cols = []
    for ent, cols in ENTITY_TO_BIN.items():
        cols = [c for c in cols if c in bin_f.columns]
        if not cols:
            continue
        ent_change = bin_f[cols].diff().abs().fillna(0.0).sum(axis=1).astype(np.float32)
        ent_change_30 = ent_change.rolling(30, min_periods=1).mean().astype(np.float32)
        if float(ent_change_30.mean()) > 1e-5:
            cname = f"ent_act__{ent}"
            reg[cname] = ent_change_30
            entity_activity_cols.append(cname)

    tod = _time_of_day_features(source_df, TIME_COL)
    reg["tod_sin"] = tod["tod_sin"].to_numpy(dtype=np.float32)
    reg["tod_cos"] = tod["tod_cos"].to_numpy(dtype=np.float32)

    reg["active_score"] = (
        0.20 * reg["bin_mean_30"]
        + 0.10 * reg["bin_mean_300"]
        + 0.20 * reg["bin_change_rate_30"]
        + 0.10 * reg["bin_change_rate_300"]
        + 0.20 * reg["bin_change_mass_30"]
        + 0.10 * reg["bin_var_30"]
        + 0.05 * reg["drv_rate_30"]
        + 0.05 * reg["drv_rate_300"]
    ).astype(np.float32)

    return reg.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _robust_fit_transform(X_raw: np.ndarray):
    med = np.nanmedian(X_raw, axis=0).astype(np.float32)
    q25 = np.nanquantile(X_raw, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(X_raw, 0.75, axis=0).astype(np.float32)
    iqr = np.maximum(q75 - q25, 1e-6).astype(np.float32)
    X = ((X_raw - med) / iqr).astype(np.float32)
    return X, {"median": med, "iqr": iqr}


def _robust_transform(X_raw: np.ndarray, scaler: dict):
    med = np.asarray(scaler["median"], dtype=np.float32)
    iqr = np.asarray(scaler["iqr"], dtype=np.float32)
    return ((X_raw - med) / np.maximum(iqr, 1e-6)).astype(np.float32)


def _build_state_clusters(df_bin, df_drv, source_df, k=12):
    reg = _build_state_feature_frame(df_bin, df_drv, source_df)

    feat_cols = list(reg.columns)
    X_raw = reg[feat_cols].to_numpy(dtype=np.float32)

    X, scaler = _robust_fit_transform(X_raw)

    k_eff = int(max(6, min(int(k), len(X) // 5000 if len(X) > 20000 else int(k))))
    k_eff = max(2, min(k_eff, max(2, len(X) - 1)))

    km = KMeans(n_clusters=k_eff, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)

    cluster_rows = []
    for cid in range(k_eff):
        mask = labels == cid
        if not mask.any():
            cluster_rows.append({
                "cluster_id": cid,
                "frac": 0.0,
                "bin_mean_30_mean": 0.0,
                "bin_change_rate_30_mean": 0.0,
                "bin_change_mass_30_mean": 0.0,
                "bin_var_30_mean": 0.0,
                "drv_rate_30_mean": 0.0,
                "active_score_mean": 0.0,
            })
            continue

        cluster_rows.append({
            "cluster_id": cid,
            "frac": float(mask.mean()),
            "bin_mean_30_mean": float(reg.loc[mask, "bin_mean_30"].mean()),
            "bin_change_rate_30_mean": float(reg.loc[mask, "bin_change_rate_30"].mean()),
            "bin_change_mass_30_mean": float(reg.loc[mask, "bin_change_mass_30"].mean()),
            "bin_var_30_mean": float(reg.loc[mask, "bin_var_30"].mean()),
            "drv_rate_30_mean": float(reg.loc[mask, "drv_rate_30"].mean()),
            "active_score_mean": float(reg.loc[mask, "active_score"].mean()),
        })

    cluster_df = pd.DataFrame(cluster_rows).sort_values("cluster_id").reset_index(drop=True)

    active_q75 = float(cluster_df["active_score_mean"].quantile(0.75))
    active_q40 = float(cluster_df["active_score_mean"].quantile(0.40))
    trans_q75 = float(cluster_df["bin_change_rate_30_mean"].quantile(0.75))
    trans_mass_q75 = float(cluster_df["bin_change_mass_30_mean"].quantile(0.75))
    drv_q75 = float(cluster_df["drv_rate_30_mean"].quantile(0.75))

    names = []
    for _, r in cluster_df.iterrows():
        cid = int(r["cluster_id"])
        active_score_mean = float(r["active_score_mean"])
        trans_rate = float(r["bin_change_rate_30_mean"])
        trans_mass = float(r["bin_change_mass_30_mean"])
        drv_rate = float(r["drv_rate_30_mean"])

        if drv_rate >= max(0.01, drv_q75) and active_score_mean >= max(0.15, active_q75):
            name = f"driver_active_{cid:03d}"
        elif trans_rate >= max(0.02, trans_q75) or trans_mass >= max(0.05, trans_mass_q75):
            name = f"transition_state_{cid:03d}"
        elif active_score_mean <= active_q40:
            name = f"background_{cid:03d}"
        else:
            name = f"mixed_state_{cid:03d}"

        names.append(name)

    return labels.astype(np.int16), km, names, feat_cols, scaler, reg


def _predict_state_clusters(df_bin, df_drv, source_df, feat_cols, scaler, km):
    reg = _build_state_feature_frame(df_bin, df_drv, source_df)

    for c in feat_cols:
        if c not in reg.columns:
            reg[c] = 0.0

    X_raw = reg[feat_cols].to_numpy(dtype=np.float32)
    X = _robust_transform(X_raw, scaler)
    labels = km.predict(X).astype(np.int16)
    return labels, reg


state_labels_tr, state_km, state_names, STATE_FEATURE_COLS, STATE_FEATURE_SCALER, STATE_FEATURES_TR = _build_state_clusters(
    X_bin_tr, X_drv_tr, df_tr, k=CFG12["state_cluster_k"]
)

state_labels_val, STATE_FEATURES_VAL = _predict_state_clusters(
    X_bin_val, X_drv_val, df_val, STATE_FEATURE_COLS, STATE_FEATURE_SCALER, state_km
)

state_labels_te_real_eval, STATE_FEATURES_TE_REAL_EVAL = _predict_state_clusters(
    X_bin_te, X_drv_te, df_te, STATE_FEATURE_COLS, STATE_FEATURE_SCALER, state_km
)

_state_df = pd.DataFrame({
    "cluster_id": state_labels_tr.astype(np.int32),
    "bin_sum": STATE_FEATURES_TR["bin_sum"].to_numpy(dtype=np.float32),
    "drv_sum": STATE_FEATURES_TR["drv_sum"].to_numpy(dtype=np.float32),
    "bin_any": STATE_FEATURES_TR["bin_any"].to_numpy(dtype=np.float32),
    "drv_any": STATE_FEATURES_TR["drv_any"].to_numpy(dtype=np.float32),
    "bin_diff_30": STATE_FEATURES_TR["bin_change_mass_30"].to_numpy(dtype=np.float32),
    "drv_diff_30": STATE_FEATURES_TR["drv_rate_30"].to_numpy(dtype=np.float32),
})

STATE_CLUSTER_AUDIT_DF = (
    _state_df.groupby("cluster_id", as_index=False)
    .agg(
        n=("cluster_id", "size"),
        frac=("cluster_id", lambda x: float(len(x) / len(_state_df))),
        bin_sum_mean=("bin_sum", "mean"),
        drv_sum_mean=("drv_sum", "mean"),
        bin_any_rate=("bin_any", "mean"),
        drv_any_rate=("drv_any", "mean"),
        bin_diff_30_mean=("bin_diff_30", "mean"),
        drv_diff_30_mean=("drv_diff_30", "mean"),
    )
    .sort_values(["frac", "drv_any_rate", "bin_any_rate"], ascending=False)
    .reset_index(drop=True)
)

STATE_CLUSTER_AUDIT_DF["cluster_name"] = STATE_CLUSTER_AUDIT_DF["cluster_id"].map(
    {i: n for i, n in enumerate(state_names)}
)

STATE_CLUSTER_AUDIT_CSV = os.path.join(REPORT_DIR, "iot_state_cluster_audit_v24_0.csv")
STATE_CLUSTER_AUDIT_DF.to_csv(STATE_CLUSTER_AUDIT_CSV, index=False)

cluster_name_counts = pd.Series(state_names).value_counts().to_dict()
largest_cluster_frac = float(STATE_CLUSTER_AUDIT_DF["frac"].max()) if len(STATE_CLUSTER_AUDIT_DF) else 1.0

driver_named_frac = float(
    STATE_CLUSTER_AUDIT_DF.loc[
        STATE_CLUSTER_AUDIT_DF["cluster_name"].astype(str).str.startswith("driver_active_"),
        "frac"
    ].sum()
) if len(STATE_CLUSTER_AUDIT_DF) else 0.0

transition_named_frac = float(
    STATE_CLUSTER_AUDIT_DF.loc[
        STATE_CLUSTER_AUDIT_DF["cluster_name"].astype(str).str.startswith("transition_state_"),
        "frac"
    ].sum()
) if len(STATE_CLUSTER_AUDIT_DF) else 0.0

background_named_frac = float(
    STATE_CLUSTER_AUDIT_DF.loc[
        STATE_CLUSTER_AUDIT_DF["cluster_name"].astype(str).str.startswith("background_"),
        "frac"
    ].sum()
) if len(STATE_CLUSTER_AUDIT_DF) else 1.0

non_background_frac = float(max(0.0, 1.0 - background_named_frac))

p = STATE_CLUSTER_AUDIT_DF["frac"].to_numpy(dtype=np.float64) if len(STATE_CLUSTER_AUDIT_DF) else np.array([1.0])
p = p[p > 0]
state_cluster_entropy = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
effective_cluster_count = float(np.exp(state_cluster_entropy))

log(
    f"[Cell12.a] Built state clusters: n_clusters={len(np.unique(state_labels_tr))} | "
    f"name_counts={cluster_name_counts} | largest_cluster_frac={largest_cluster_frac:.4f} | "
    f"background_named_frac={background_named_frac:.4f} | non_background_frac={non_background_frac:.4f} | "
    f"transition_named_frac={transition_named_frac:.4f} | driver_named_frac={driver_named_frac:.4f} | "
    f"effective_cluster_count={effective_cluster_count:.4f}"
)

# ----------------------------------------------------------
# 11) Fail-fast validation
# ----------------------------------------------------------
binary_non01_df = CLASS_SUSPICIOUS_DF[
    CLASS_SUSPICIOUS_DF["flags"].str.contains("binary_block_but_not_binary01_after_canonicalization", na=False)
].copy()

multi_overlap_df = CLASS_SUSPICIOUS_DF[
    CLASS_SUSPICIOUS_DF["flags"].str.contains("multi_block_overlap", na=False)
].copy()

telemetry_misroute_df = CLASS_AUDIT_DF[
    CLASS_AUDIT_DF["flags"].astype(str).str.contains("telemetry_like_not_routed_to_device_telemetry", na=False)
].copy()

s5_misroute_df = CLASS_AUDIT_DF[
    CLASS_AUDIT_DF["flags"].astype(str).str.contains("s5_not_routed_to_s5", na=False)
].copy()

step_n = len(STEP_PROGRESS_COLS)

fail_msgs = []

if len(binary_non01_df) > int(CFG12["max_binary_non01_cols"]):
    fail_msgs.append(
        f"binary contract invalid: found {len(binary_non01_df)} BIN_COLS that are not binary 0/1 on TRAIN"
    )

if len(multi_overlap_df) > int(CFG12["max_multi_block_overlap_cols"]):
    fail_msgs.append(
        f"contract overlap invalid: found {len(multi_overlap_df)} columns assigned to multiple blocks"
    )

if step_n < int(CFG12["require_step_progress_min"]):
    fail_msgs.append(
        f"step_progress routing invalid: found {step_n}, expected at least {int(CFG12['require_step_progress_min'])}"
    )

if len(telemetry_misroute_df) > int(CFG12["max_telemetry_misroutes"]):
    fail_msgs.append(
        f"telemetry routing invalid: found {len(telemetry_misroute_df)} non-S5 telemetry-like columns not routed to device_telemetry"
    )

if len(s5_misroute_df) > 0:
    fail_msgs.append(
        f"s5 routing invalid: found {len(s5_misroute_df)} s5_max columns not routed to s5"
    )

if driver_named_frac > float(CFG12["max_driver_named_frac"]):
    fail_msgs.append(
        f"state regime dominance invalid: driver_active clusters cover {driver_named_frac:.4f} of TRAIN rows"
    )

# Sparse smart-home CPS data can legitimately be background-dominant.
# These state-cluster diagnostics are warning-only in Cell 12.a.
# Downstream Cells 12.b/12.c should use protocol context, TOD, availability,
# and driver processes rather than requiring balanced state clusters here.
STATE_CLUSTER_WARNING_REASONS = []

if (
    largest_cluster_frac > float(CFG12["warn_largest_cluster_frac"])
    and effective_cluster_count < float(CFG12["min_effective_cluster_count"])
    and non_background_frac < float(CFG12["min_non_background_frac"])
):
    STATE_CLUSTER_WARNING_REASONS.append(
        "dominant_background_low_effective_cluster_diversity"
    )

if (transition_named_frac + driver_named_frac) < float(CFG12["min_transition_or_driver_frac"]):
    STATE_CLUSTER_WARNING_REASONS.append(
        "low_transition_or_driver_cluster_mass"
    )

globals()["STATE_CLUSTER_WARNING_REASONS"] = STATE_CLUSTER_WARNING_REASONS

if fail_msgs:
    detail_path = os.path.join(REPORT_DIR, "iot_contract_failfast_summary_v24_0.json")
    _write_json(
        detail_path,
        {
            "fail_msgs": fail_msgs,
            "binary_non01_cols_preview": binary_non01_df["col"].head(50).tolist(),
            "multi_overlap_cols_preview": multi_overlap_df["col"].head(50).tolist(),
            "telemetry_misroute_cols_preview": telemetry_misroute_df["col"].head(50).tolist(),
            "s5_misroute_cols_preview": s5_misroute_df["col"].head(50).tolist(),
            "family_counts": {k: len(v) for k, v in FAMILY_GROUPS.items()},
            "count_subtype_counts": {
                "session_counter": len(SESSION_COUNT_COLS),
                "accumulator": len(ACCUMULATOR_COUNT_COLS),
                "event_counter": len(EVENT_COUNT_COLS),
            },
            "largest_cluster_frac": largest_cluster_frac,
            "driver_named_frac": driver_named_frac,
            "transition_named_frac": transition_named_frac,
            "background_named_frac": background_named_frac,
            "non_background_frac": non_background_frac,
            "state_cluster_entropy": state_cluster_entropy,
            "effective_cluster_count": effective_cluster_count,
            "cluster_name_counts": cluster_name_counts,
            "classification_audit_csv": CLASS_AUDIT_CSV,
            "state_cluster_audit_csv": STATE_CLUSTER_AUDIT_CSV,
            "routing_policy_csv": cont_routing_policy_csv,
        },
    )
    raise RuntimeError(
        "[Cell12.a] Upstream IoT grouping/taxonomy contract is invalid for generation.\n"
        + "\n".join(f"- {m}" for m in fail_msgs)
        + f"\nDetails: {detail_path}"
    )

STATE_CLUSTER_GOV_ROWS = []

def _gov_warn(metric, value, threshold, reason):
    STATE_CLUSTER_GOV_ROWS.append({
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "severity": "warning",
        "reason": reason,
    })

if largest_cluster_frac > float(CFG12["warn_largest_cluster_frac"]):
    _gov_warn(
        "largest_cluster_frac",
        largest_cluster_frac,
        CFG12["warn_largest_cluster_frac"],
        "dominant background regime can be legitimate in sparse smart-home CPS",
    )

if effective_cluster_count < float(CFG12["min_effective_cluster_count"]):
    _gov_warn(
        "effective_cluster_count",
        effective_cluster_count,
        CFG12["min_effective_cluster_count"],
        "low effective cluster diversity warning",
    )

if non_background_frac < float(CFG12["min_non_background_frac"]):
    _gov_warn(
        "non_background_frac",
        non_background_frac,
        CFG12["min_non_background_frac"],
        "low non-background mass warning",
    )

transition_plus_driver_frac = float(transition_named_frac + driver_named_frac)
if transition_plus_driver_frac < float(CFG12["min_transition_or_driver_frac"]):
    _gov_warn(
        "transition_plus_driver_frac",
        transition_plus_driver_frac,
        CFG12["min_transition_or_driver_frac"],
        "low transition/driver mass warning",
    )

combined_collapse = (
    largest_cluster_frac > float(CFG12["warn_largest_cluster_frac"])
    and effective_cluster_count < float(CFG12["min_effective_cluster_count"])
    and non_background_frac < float(CFG12["min_non_background_frac"])
)

if combined_collapse:
    STATE_CLUSTER_GOV_ROWS.append({
        "metric": "combined_state_cluster_collapse",
        "value": 1.0,
        "threshold": 0.0,
        "severity": "warning",
        "reason": (
            "dominant sparse background regime detected; warning only because "
            "1 Hz smart-home CPS data can legitimately be background-heavy"
        ),
    })

STATE_CLUSTER_GOV_DF = pd.DataFrame(STATE_CLUSTER_GOV_ROWS)
STATE_CLUSTER_GOV_CSV = os.path.join(REPORT_DIR, "cell12a_state_cluster_governance_v24_0.csv")
STATE_CLUSTER_GOV_DF.to_csv(STATE_CLUSTER_GOV_CSV, index=False)

if len(STATE_CLUSTER_GOV_DF):
    for _, r in STATE_CLUSTER_GOV_DF.iterrows():
        log(
            f"[Cell12.a] state-cluster-governance | metric={r['metric']} | "
            f"value={float(r['value']):.6f} | threshold={float(r['threshold']):.6f} | "
            f"severity={r['severity']} | reason={r['reason']}"
        )

hard_fail_df = STATE_CLUSTER_GOV_DF.loc[STATE_CLUSTER_GOV_DF["severity"] == "hard_fail"].copy()
if len(hard_fail_df):
    raise RuntimeError(f"[Cell12.a] State-cluster governance failed. See: {STATE_CLUSTER_GOV_CSV}")

log(
    f"[Cell12.a] State-cluster governance passed with warning-only sparsity diagnostics. "
    f"Audit: {STATE_CLUSTER_GOV_CSV}"
)

# ----------------------------------------------------------
# 12) Driver synthesis with protocol-context coupling
# ----------------------------------------------------------
def _detect_driver_episodes(X_drv: pd.DataFrame):
    arr = X_drv.fillna(0).to_numpy(dtype=np.float32)
    any_evt = (arr.sum(axis=1) > 0).astype(np.int8)

    if len(any_evt) == 0:
        return arr, []

    starts = np.flatnonzero((any_evt[1:] == 1) & (any_evt[:-1] == 0)) + 1
    if any_evt[0] == 1:
        starts = np.r_[0, starts]

    ends = np.flatnonzero((any_evt[:-1] == 1) & (any_evt[1:] == 0))
    if any_evt[-1] == 1:
        ends = np.r_[ends, len(any_evt) - 1]

    episodes = [(int(s), int(e)) for s, e in zip(starts, ends) if int(e) >= int(s)]
    return arr, episodes


def _cluster_driver_bursts(episodes, gap_thr=12):
    if not episodes:
        return []
    clusters = []
    cur = [episodes[0]]
    for ep in episodes[1:]:
        gap = int(ep[0] - cur[-1][1] - 1)
        if gap <= int(gap_thr):
            cur.append(ep)
        else:
            clusters.append(cur)
            cur = [ep]
    clusters.append(cur)
    return clusters


def _protocol_context_intensity() -> np.ndarray:
    if PROTOCOL_CONTEXT_TEST is None or PROTOCOL_CONTEXT_TEST.empty:
        return np.ones(N_TE, dtype=np.float64)

    preferred = [
        c for c in PROTOCOL_CONTEXT_TEST.columns
        if c.endswith("__value_activity_300")
        or c.endswith("__value_activity_1800")
        or c.endswith("__obs_rate_300")
    ]
    if not preferred:
        preferred = list(PROTOCOL_CONTEXT_TEST.columns)

    block = PROTOCOL_CONTEXT_TEST[preferred].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = block.to_numpy(dtype=np.float64, copy=False)
    score = np.nanmean(x, axis=1)

    if not np.isfinite(score).any():
        return np.ones(N_TE, dtype=np.float64)

    score = np.nan_to_num(score, nan=np.nanmedian(score[np.isfinite(score)]), posinf=0.0, neginf=0.0)
    lo = np.quantile(score, 0.05)
    hi = np.quantile(score, 0.95)
    if hi <= lo + 1e-12:
        return np.ones(N_TE, dtype=np.float64)

    z = np.clip((score - lo) / (hi - lo), 0.0, 1.0)
    strength = float(CFG12["driver_protocol_context_strength"])
    intensity = (1.0 - strength) + strength * (0.5 + z)
    intensity = np.clip(intensity, 0.10, 3.0)
    return intensity.astype(np.float64)


log("[Cell12.a] Detecting sparse driver episodes on TRAIN...")
drv_arr_tr, drv_episodes = _detect_driver_episodes(X_drv_tr)

if len(drv_episodes) > 1:
    gaps = np.diff([s for s, _ in drv_episodes])
else:
    gaps = np.array([N_TR], dtype=np.int64)

mean_len = float(np.mean([e - s + 1 for s, e in drv_episodes])) if drv_episodes else 0.0
mean_gap = float(np.mean(np.maximum(1, gaps))) if len(gaps) else 0.0

log(f"[Cell12.a] TRAIN driver episodes: n={len(drv_episodes)} | mean_len={mean_len:.2f} | mean_gap={mean_gap:.2f}")

drv_clusters = _cluster_driver_bursts(drv_episodes, gap_thr=CFG12["driver_burst_gap_thr"])
mean_cluster_len = float(np.mean([sum(e - s + 1 for s, e in cl) for cl in drv_clusters])) if drv_clusters else 0.0
log(f"[Cell12.a] TRAIN burst-clusters: n={len(drv_clusters)} | mean_len={mean_cluster_len:.2f}")


def _synthesize_drivers_protocol_aware():
    syn = np.zeros((N_TE, len(DRV_COLS)), dtype=np.float32)

    tr_any = (X_drv_tr.to_numpy(dtype=np.float32).sum(axis=1) > 0)
    val_any = (X_drv_val.to_numpy(dtype=np.float32).sum(axis=1) > 0)

    tr_rate = float(tr_any.mean())
    val_rate = float(val_any.mean())

    tw = float(CFG12["driver_target_train_weight"])
    vw = float(CFG12["driver_target_val_weight"])
    denom = max(tw + vw, 1e-9)
    target_rate = (tw * tr_rate + vw * val_rate) / denom

    if not np.isfinite(target_rate) or target_rate < 0:
        target_rate = tr_rate

    expected_active = int(round(target_rate * N_TE))

    if not drv_clusters or expected_active <= 0:
        return pd.DataFrame(syn, columns=DRV_COLS, index=df_te.index)

    intensity = _protocol_context_intensity()
    p_start = intensity / max(float(intensity.sum()), 1e-12)

    max_attempts = int(max(1000, expected_active * 20))
    attempts = 0
    placed_active = 0

    while placed_active < expected_active and attempts < max_attempts:
        attempts += 1

        cl = drv_clusters[int(rng.integers(0, len(drv_clusters)))]

        if rng.random() < 0.72:
            pieces = cl
        else:
            pieces = [cl[int(rng.integers(0, len(cl)))]]

        total_len = int(sum(e - s + 1 for s, e in pieces))
        if total_len <= 0 or total_len >= N_TE:
            continue

        start = int(rng.choice(np.arange(N_TE), p=p_start))
        start = min(max(0, start), max(0, N_TE - total_len - 1))

        t = start
        local_placed = 0

        for s, e in pieces:
            ep = drv_arr_tr[s:e + 1]
            L = int(len(ep))
            if L <= 0 or t + L > N_TE:
                break

            before = (syn[t:t + L].sum(axis=1) > 0).sum()
            syn[t:t + L] = np.maximum(syn[t:t + L], ep)
            after = (syn[t:t + L].sum(axis=1) > 0).sum()
            local_placed += int(after - before)

            gap = int(max(1, rng.poisson(max(2.0, mean_gap * 0.20))))
            t += L + gap

        placed_active += max(0, local_placed)

    syn_any = (syn.sum(axis=1) > 0).astype(np.int8)
    cur_rate = float(syn_any.mean())

    if cur_rate > 0:
        mult = np.clip(target_rate / max(cur_rate, 1e-9), *CFG12["driver_base_event_boost_clip"])
    else:
        mult = 1.0

    if cur_rate < target_rate and mult > 1.02:
        need_more = int(round((target_rate - cur_rate) * N_TE))
        candidate = np.where(syn_any == 0)[0]

        if need_more > 0 and len(candidate) > 0:
            p_cols = np.asarray(X_drv_val.mean(axis=0), dtype=np.float64)
            if not np.isfinite(p_cols).all() or p_cols.sum() <= 0:
                p_cols = np.asarray(X_drv_tr.mean(axis=0), dtype=np.float64)
            p_cols = np.clip(p_cols, 1e-8, None)
            p_cols = p_cols / p_cols.sum()

            local_intensity = intensity[candidate].astype(np.float64)
            local_p = local_intensity / max(float(local_intensity.sum()), 1e-12)
            choose = rng.choice(candidate, size=min(len(candidate), need_more), replace=False, p=local_p)

            for idx in choose:
                cidx = int(rng.choice(len(DRV_COLS), p=p_cols))
                syn[int(idx), cidx] = 1.0

    return pd.DataFrame(syn, columns=DRV_COLS, index=df_te.index)


log("[Cell12.a] Synthesizing sparse driver stream with A2 protocol-context coupling...")
IOT_SYN_DRIVERS = _synthesize_drivers_protocol_aware()

drv_real_rate = float((X_drv_te.to_numpy(dtype=np.float32).sum(axis=1) > 0).mean())
drv_syn_rate = float((IOT_SYN_DRIVERS.to_numpy(dtype=np.float32).sum(axis=1) > 0).mean())
drv_real_mean = float(X_drv_te.to_numpy(dtype=np.float32).mean())
drv_syn_mean = float(IOT_SYN_DRIVERS.to_numpy(dtype=np.float32).mean())

driver_audit = {
    "version": "cell12a_v24_1_protocol_aware_driver_synthesis",
    "train_event_rate": float((X_drv_tr.to_numpy(dtype=np.float32).sum(axis=1) > 0).mean()),
    "val_event_rate": float((X_drv_val.to_numpy(dtype=np.float32).sum(axis=1) > 0).mean()),
    "test_real_event_rate_eval_only": drv_real_rate,
    "synthetic_event_rate": drv_syn_rate,
    "test_real_mean_eval_only": drv_real_mean,
    "synthetic_mean": drv_syn_mean,
    "episodes_train_n": int(len(drv_episodes)),
    "burst_clusters_train_n": int(len(drv_clusters)),
    "mean_episode_len_train": mean_len,
    "mean_episode_gap_train": mean_gap,
    "protocol_context_used": bool(PROTOCOL_CONTEXT_TEST is not None and not PROTOCOL_CONTEXT_TEST.empty),
    "protocol_context_strength": float(CFG12["driver_protocol_context_strength"]),
    "test_values_used_for_generation": False,
}

DRIVER_AUDIT_JSON = os.path.join(REPORT_DIR, "cell12a_v24_1_driver_synthesis_audit.json")
_write_json(DRIVER_AUDIT_JSON, driver_audit)

log(
    f"[Cell12.a] Driver done | event_rate={drv_syn_rate:.6f} | "
    f"mean={drv_syn_mean:.6f} | active_rows={(IOT_SYN_DRIVERS.to_numpy(dtype=np.float32).sum(axis=1) > 0).sum()} | "
    f"audit={DRIVER_AUDIT_JSON}"
)

# ----------------------------------------------------------
# 13) Aligned base window
# ----------------------------------------------------------
def _select_base_window():
    L = N_TE
    if L <= CFG12["window_len"]:
        start = 0
    else:
        max_start = max(1, N_TR - L)
        start = int(rng.integers(0, max_start))
    start = max(0, min(start, max(0, N_TR - L)))
    jitter = rng.integers(-CFG12["window_jitter"], CFG12["window_jitter"] + 1, size=8)
    return start, jitter


log("[Cell12.a] Selecting aligned household base window...")
base_offset, jitter_sample = _select_base_window()
log(
    f"[Cell12.a] Base window offset={base_offset} | "
    f"jitter_min={int(jitter_sample.min())} | "
    f"jitter_med={int(np.median(jitter_sample))} | "
    f"jitter_max={int(jitter_sample.max())}"
)

# ----------------------------------------------------------
# 14) Foundation manifest
# ----------------------------------------------------------
CELL12A_FOUNDATION_MANIFEST = {
    "version": "cell12a_v24_1_study_thesis_phase1_quality_aware_contract_locked",
    "N_TR": int(N_TR),
    "N_VAL": int(N_VAL),
    "N_TE": int(N_TE),
    "seed": int(SEED),
    "torch_device": str(device),
    "columns": {
        "full_iot_namespace": int(len(FULL_IOT_NAMESPACE_COLS)),
        "drivers": int(len(DRV_COLS)),
        "binary_targets": int(len(BIN_COLS)),
        "continuous_value_targets": int(len(CONT_VALUE_COLS)),
        "state_targets": int(len(STATE_TARGET_COLS)),
        "meta": int(len(IOT_META)),
        "excluded": int(len(IOT_EXCLUDED)),
        "cont_mask": int(len(CONT_MASK_COLS)),
        "ddpm": int(len(DDPM_COLS)),
        "non_ddpm": int(len(NON_DDPM_COLS)),
    },
    "protocol_coupling": {
        "required": bool(CFG12["require_cell11_protocol_variants"]),
        "a0_protocol_final_path": A0_PROTOCOL_FINAL_PATH,
        "a1_protocol_final_path": A1_PROTOCOL_FINAL_PATH,
        "a2_protocol_final_path": A2_PROTOCOL_FINAL_PATH,
        "cell11_manifest_path": CELL11_FINAL_MANIFEST_PATH if os.path.exists(CELL11_FINAL_MANIFEST_PATH) else None,
        "protocol_context_shape": list(PROTOCOL_CONTEXT_TEST.shape) if isinstance(PROTOCOL_CONTEXT_TEST, pd.DataFrame) else None,
        "synthetic_mask_rates_test": PROTOCOL_SYN_MASK_RATES_TEST,
        "real_mask_rates_test_eval_only": PROTOCOL_REAL_MASK_RATES_TEST,
        "synthetic_minus_real_mask_drift_test_eval_only": PROTOCOL_SYN_MINUS_REAL_MASK_DRIFT_TEST,
    },
    "upstream_contracts": {
        "cell10_8_selector_contract_path": CELL10_8_SELECTOR_CONTRACT_PATH,
        "cell10_9_materializer_contract_path": CELL10_9_MATERIALIZER_CONTRACT_PATH,
        "precell11_runtime_contract_path": PRECELL11_RUNTIME_CONTRACT_PATH,
        "cell11_final_audit_path": CELL11_FINAL_AUDIT_PATH if os.path.exists(CELL11_FINAL_AUDIT_PATH) else None,
        "cell11b_quarantine_manifest_present": bool(os.path.exists(CELL11B_QUARANTINE_MANIFEST_PATH)),
        "cell11b_quarantine_allowed_for_canonical_run": False,
    },
    "phase1_quality_decomposition": {
        "quality_dimension_registry_path": QUALITY_DIMENSION_REGISTRY_PATH,
        "a0_a1_a2_variant_manifest_path": A0_A1_A2_VARIANT_MANIFEST_PATH,
        "quality_attribution_matrix_path": QUALITY_ATTRIBUTION_MATRIX_PATH,
        "quality_dimensions_n": int(len(_qdims)),
        "a0_publication_role": _a0_role,
        "a1_publication_role": _a1_role,
        "a2_publication_role": _a2_role,
        "quality_attribution_rows": int(len(QUALITY_ATTRIBUTION_MATRIX)),
        "quality_dimensions": QDIM_REGISTRY,
        },
    "iot_role_ownership": {
        "iot_role_ownership_csv": IOT_ROLE_OWNERSHIP_PATH,
        "iot_partition_contract_json": IOT_PARTITION_CONTRACT_PATH,
        "iot_cell_routing_manifest_json": IOT_CELL_ROUTING_MANIFEST_PATH,
        "full_namespace_sha256": _sha_list(FULL_IOT_NAMESPACE_COLS),
        "driver_targets_sha256": _sha_list(IOT_DRIVER_TARGETS),
        "binary_targets_sha256": _sha_list(IOT_BINARY_TARGETS),
        "continuous_value_targets_sha256": _sha_list(IOT_CONTINUOUS_VALUE_TARGETS),
        "state_targets_sha256": _sha_list(IOT_STATE_TARGETS),
    },
    "phase2_target_plan": IOT_PHASE2_TARGET_PLAN,
    "binary_routing": BINARY_ROUTING_POLICY_MANIFEST,
    "routing": CONT_ROUTING_POLICY_MANIFEST,
    "families": {k: int(len(v)) for k, v in FAMILY_GROUPS.items()},
    "count_subtypes": {
        "session_counter": int(len(SESSION_COUNT_COLS)),
        "accumulator": int(len(ACCUMULATOR_COUNT_COLS)),
        "event_counter": int(len(EVENT_COUNT_COLS)),
    },
    "state_clusters": {
        "n_clusters": int(len(np.unique(state_labels_tr))),
        "cluster_name_counts": cluster_name_counts,
        "largest_cluster_frac": largest_cluster_frac,
        "background_named_frac": background_named_frac,
        "non_background_frac": non_background_frac,
        "transition_named_frac": transition_named_frac,
        "driver_named_frac": driver_named_frac,
        "effective_cluster_count": effective_cluster_count,
        "state_cluster_audit_csv": STATE_CLUSTER_AUDIT_CSV,
        "state_cluster_governance_csv": STATE_CLUSTER_GOV_CSV,
    },
    "driver_synthesis": driver_audit,
    "reports": {
        "classification_audit_csv": CLASS_AUDIT_CSV,
        "routing_policy_csv": cont_routing_policy_csv,
        "routing_policy_manifest_json": cont_routing_policy_manifest_json,
        "driver_audit_json": DRIVER_AUDIT_JSON,
        "binary_routing_policy_csv": BINARY_ROUTING_POLICY_CSV,
        "binary_routing_policy_manifest_json": BINARY_ROUTING_POLICY_MANIFEST_JSON,
        "quality_dimension_registry_json": QUALITY_DIMENSION_REGISTRY_PATH,
        "a0_a1_a2_variant_manifest_json": A0_A1_A2_VARIANT_MANIFEST_PATH,
        "quality_attribution_matrix_csv": QUALITY_ATTRIBUTION_MATRIX_PATH,
        "iot_role_ownership_csv": IOT_ROLE_OWNERSHIP_PATH,
        "iot_partition_contract_json": IOT_PARTITION_CONTRACT_PATH,
        "iot_cell_routing_manifest_json": IOT_CELL_ROUTING_MANIFEST_PATH,
    },
    "leakage_contract": {
        "train_used_for_fitting": True,
        "val_used_for_calibration": True,
        "test_iot_values_used_for_generation": False,
        "test_iot_values_used_for_eval_audit_only": True,
        "synthetic_protocol_A2_used_as_generation_context": True,
        "real_test_protocol_values_used_for_iot_generation": False,
        "test_iot_matrices_exported_for_downstream_QA_only": True,
        "downstream_cells_must_not_use_X_drv_te_X_bin_te_X_cont_te_for_generation": True,
    },
}

CELL12A_FOUNDATION_MANIFEST_JSON = os.path.join(OUT_ART, "cell12a_foundation_manifest_v24_0.json")
_write_json(CELL12A_FOUNDATION_MANIFEST_JSON, CELL12A_FOUNDATION_MANIFEST)

globals()["CELL12A_FOUNDATION_MANIFEST"] = CELL12A_FOUNDATION_MANIFEST
globals()["CELL12A_FOUNDATION_MANIFEST_JSON"] = CELL12A_FOUNDATION_MANIFEST_JSON

# ----------------------------------------------------------
# 15) Publish globals expected by 12.b / 12.c
# ----------------------------------------------------------
# TEST IoT matrices are exported only for downstream QA/audit compatibility.
# They must not be used by Cells 12.b/12.c for fitting, selection, routing,
# calibration, materialization, or fallback generation.
IOT_TEST_VALUE_ACCESS_POLICY = {
    "version": "cell12a_v24_test_iot_value_access_policy",
    "X_drv_te_exported": True,
    "X_bin_te_exported": True,
    "X_cont_te_exported": True,
    "allowed_use": "final_QA_or_schema_audit_only",
    "forbidden_use": [
        "generator_fitting",
        "candidate_selection",
        "threshold_calibration",
        "fallback_materialization",
        "support_replay",
        "domain_repair_using_real_test_values",
        "availability_generation",
        "binary_generation",
        "continuous_value_generation",
    ],
    "test_iot_values_used_in_cell12a_generation": False,
}

globals()["IOT_TEST_VALUE_ACCESS_POLICY"] = IOT_TEST_VALUE_ACCESS_POLICY
globals()["df_val"] = df_val

globals()["OUTDIR"] = OUTDIR
globals()["OUT_SYN"] = OUT_SYN
globals()["REPORT_DIR"] = REPORT_DIR
globals()["SEED"] = SEED
globals()["rng"] = rng
globals()["device"] = device
globals()["CFG12"] = CFG12
globals()["TIME_COL"] = TIME_COL

globals()["DRV_COLS"] = DRV_COLS
globals()["BIN_COLS"] = BIN_COLS
globals()["CONT_COLS"] = CONT_COLS
globals()["CONT_MASK_COLS"] = CONT_MASK_COLS
globals()["CONT_VALUE_COLS"] = CONT_VALUE_COLS

globals()["X_drv_tr"] = X_drv_tr
globals()["X_bin_tr"] = X_bin_tr
globals()["X_cont_tr"] = X_cont_tr
globals()["X_drv_val"] = X_drv_val
globals()["X_bin_val"] = X_bin_val
globals()["X_cont_val"] = X_cont_val
globals()["X_drv_te"] = X_drv_te
globals()["X_bin_te"] = X_bin_te
globals()["X_cont_te"] = X_cont_te

globals()["N_TR"] = N_TR
globals()["N_VAL"] = N_VAL
globals()["N_TE"] = N_TE
globals()["window_fits"] = window_fits

globals()["ENTITY_MAP"] = ENTITY_MAP
globals()["ENTITY_TO_DRV"] = ENTITY_TO_DRV
globals()["ENTITY_TO_BIN"] = ENTITY_TO_BIN
globals()["ENTITY_TO_CONT"] = ENTITY_TO_CONT

globals()["FAMILY_MAP"] = FAMILY_MAP
globals()["OTHER_SUBTYPE_MAP"] = OTHER_SUBTYPE_MAP
globals()["FAMILY_GROUPS"] = FAMILY_GROUPS
globals()["DDPM_COLS"] = DDPM_COLS
globals()["NON_DDPM_COLS"] = NON_DDPM_COLS

globals()["DEVICE_TELEMETRY_COLS"] = DEVICE_TELEMETRY_COLS
globals()["CUMULATIVE_COUNT_COLS"] = CUMULATIVE_COUNT_COLS
globals()["STEP_PROGRESS_COLS"] = STEP_PROGRESS_COLS
globals()["AUX_GLOBAL_COUNTER_COLS"] = AUX_GLOBAL_COUNTER_COLS

globals()["COUNT_SUBTYPE_MAP"] = COUNT_SUBTYPE_MAP
globals()["SESSION_COUNT_COLS"] = SESSION_COUNT_COLS
globals()["ACCUMULATOR_COUNT_COLS"] = ACCUMULATOR_COUNT_COLS
globals()["EVENT_COUNT_COLS"] = EVENT_COUNT_COLS
globals()["SYNTHESIS_SUBFAMILY_MAP"] = SYNTHESIS_SUBFAMILY_MAP

globals()["CLASS_AUDIT_DF"] = CLASS_AUDIT_DF
globals()["CLASS_SUSPICIOUS_DF"] = CLASS_SUSPICIOUS_DF
globals()["CLASS_AUDIT_CSV"] = CLASS_AUDIT_CSV

globals()["BIN_NON01_COLS"] = BIN_NON01_COLS
globals()["MULTI_BLOCK_OVERLAP"] = MULTI_BLOCK_OVERLAP
globals()["OTHER_SUBTYPE_COUNTS"] = OTHER_SUBTYPE_COUNTS
globals()["SYNTHESIS_SUBFAMILY_COUNTS"] = SYNTHESIS_SUBFAMILY_COUNTS

globals()["STATE_CLUSTER_AUDIT_DF"] = STATE_CLUSTER_AUDIT_DF
globals()["STATE_CLUSTER_AUDIT_CSV"] = STATE_CLUSTER_AUDIT_CSV
globals()["STATE_CLUSTER_GOV_DF"] = STATE_CLUSTER_GOV_DF
globals()["STATE_CLUSTER_GOV_CSV"] = STATE_CLUSTER_GOV_CSV

globals()["state_labels_tr"] = state_labels_tr
globals()["state_labels_val"] = state_labels_val
globals()["state_labels_te_real_eval"] = state_labels_te_real_eval
globals()["state_km"] = state_km
globals()["state_names"] = state_names
globals()["STATE_FEATURE_COLS"] = STATE_FEATURE_COLS
globals()["STATE_FEATURE_SCALER"] = STATE_FEATURE_SCALER
globals()["STATE_FEATURES_TR"] = STATE_FEATURES_TR
globals()["STATE_FEATURES_VAL"] = STATE_FEATURES_VAL
globals()["STATE_FEATURES_TE_REAL_EVAL"] = STATE_FEATURES_TE_REAL_EVAL

globals()["IOT_SYN_DRIVERS"] = IOT_SYN_DRIVERS
globals()["drv_real_rate"] = drv_real_rate
globals()["drv_syn_rate"] = drv_syn_rate
globals()["drv_real_mean"] = drv_real_mean
globals()["drv_syn_mean"] = drv_syn_mean
globals()["DRIVER_AUDIT_JSON"] = DRIVER_AUDIT_JSON

globals()["base_offset"] = base_offset
globals()["jitter_sample"] = jitter_sample

globals()["_related_driver_cols_for_entity"] = _related_driver_cols_for_entity
globals()["_entity_activity_from_driver_df"] = _entity_activity_from_driver_df

globals()["A0_PROTOCOL_FINAL"] = A0_PROTOCOL_FINAL
globals()["A1_PROTOCOL_FINAL"] = A1_PROTOCOL_FINAL
globals()["A2_PROTOCOL_FINAL"] = A2_PROTOCOL_FINAL
globals()["CELL11_FINAL_MANIFEST"] = CELL11_FINAL_MANIFEST
globals()["CELL11_FINAL_MANIFEST_PATH"] = CELL11_FINAL_MANIFEST_PATH

globals()["PROTOCOL_CONTEXT_TEST"] = PROTOCOL_CONTEXT_TEST
globals()["PROTOCOL_SYN_MASKS_TEST"] = PROTOCOL_SYN_MASKS_TEST
globals()["PROTOCOL_REAL_MASKS_TEST"] = PROTOCOL_REAL_MASKS_TEST
globals()["PROTOCOL_SYN_MASK_RATES_TEST"] = PROTOCOL_SYN_MASK_RATES_TEST
globals()["PROTOCOL_REAL_MASK_RATES_TEST"] = PROTOCOL_REAL_MASK_RATES_TEST
globals()["PROTOCOL_SYN_MINUS_REAL_MASK_DRIFT_TEST"] = PROTOCOL_SYN_MINUS_REAL_MASK_DRIFT_TEST

# Backward-compatible routing globals
globals()["CONT_ROUTING_POLICY_DF"] = CONT_ROUTING_POLICY_DF
globals()["CONT_ROUTING_POLICY_CSV"] = cont_routing_policy_csv
globals()["CONT_ROUTING_POLICY_MANIFEST"] = CONT_ROUTING_POLICY_MANIFEST
globals()["CONT_ROUTING_POLICY_MANIFEST_JSON"] = cont_routing_policy_manifest_json
globals()["FULL_IOT_NAMESPACE_COLS"] = FULL_IOT_NAMESPACE_COLS
globals()["STATE_TARGET_COLS"] = STATE_TARGET_COLS
globals()["IOT_EXCLUDED"] = IOT_EXCLUDED

globals()["IOT_DRIVER_TARGETS"] = IOT_DRIVER_TARGETS
globals()["IOT_BINARY_TARGETS"] = IOT_BINARY_TARGETS
globals()["IOT_CONTINUOUS_VALUE_TARGETS"] = IOT_CONTINUOUS_VALUE_TARGETS
globals()["IOT_STATE_TARGETS"] = IOT_STATE_TARGETS
globals()["IOT_NONMODELED_META_COLS"] = IOT_NONMODELED_META_COLS
globals()["IOT_NONMODELED_EXCLUDED_COLS"] = IOT_NONMODELED_EXCLUDED_COLS

globals()["IOT_PHASE2_TARGET_PLAN"] = IOT_PHASE2_TARGET_PLAN

globals()["QUALITY_DIMENSION_REGISTRY"] = QUALITY_DIMENSION_REGISTRY
globals()["QDIM_REGISTRY"] = QDIM_REGISTRY
globals()["A0_A1_A2_VARIANT_MANIFEST"] = A0_A1_A2_VARIANT_MANIFEST
globals()["QUALITY_ATTRIBUTION_MATRIX"] = QUALITY_ATTRIBUTION_MATRIX
globals()["CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT"] = CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT
globals()["CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY"] = CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY

globals()["CELL10_8_SELECTOR_CONTRACT"] = CELL10_8_SELECTOR_CONTRACT
globals()["CELL10_8_SELECTOR_CONTRACT_PATH"] = CELL10_8_SELECTOR_CONTRACT_PATH
globals()["CELL10_9_MATERIALIZER_CONTRACT"] = CELL10_9_MATERIALIZER_CONTRACT
globals()["CELL10_9_MATERIALIZER_CONTRACT_PATH"] = CELL10_9_MATERIALIZER_CONTRACT_PATH
globals()["PRECELL11_RUNTIME_CONTRACT"] = PRECELL11_RUNTIME_CONTRACT
globals()["PRECELL11_RUNTIME_CONTRACT_PATH"] = PRECELL11_RUNTIME_CONTRACT_PATH

globals()["IOT_ROLE_OWNERSHIP_DF"] = IOT_ROLE_OWNERSHIP_DF
globals()["IOT_PARTITION_CONTRACT"] = IOT_PARTITION_CONTRACT
globals()["IOT_CELL_ROUTING_MANIFEST"] = IOT_CELL_ROUTING_MANIFEST

globals()["BINARY_ROUTING_POLICY_DF"] = BINARY_ROUTING_POLICY_DF
globals()["BINARY_ROUTING_POLICY_CSV"] = BINARY_ROUTING_POLICY_CSV
globals()["BINARY_ROUTING_POLICY_MANIFEST"] = BINARY_ROUTING_POLICY_MANIFEST
globals()["BINARY_ROUTING_POLICY_MANIFEST_JSON"] = BINARY_ROUTING_POLICY_MANIFEST_JSON

# ----------------------------------------------------------
# 16) Completion checks
# ----------------------------------------------------------
required_12a_outputs = [
    "df_val",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED", "rng", "device", "CFG12", "TIME_COL",
    "DRV_COLS", "BIN_COLS", "CONT_COLS", "CONT_MASK_COLS", "CONT_VALUE_COLS",
    "X_drv_tr", "X_bin_tr", "X_cont_tr",
    "X_drv_val", "X_bin_val", "X_cont_val",
    "X_drv_te", "X_bin_te", "X_cont_te",
    "N_TR", "N_VAL", "N_TE", "window_fits",
    "ENTITY_MAP", "ENTITY_TO_DRV", "ENTITY_TO_BIN", "ENTITY_TO_CONT",
    "FAMILY_MAP", "OTHER_SUBTYPE_MAP", "FAMILY_GROUPS",
    "DDPM_COLS", "NON_DDPM_COLS",
    "DEVICE_TELEMETRY_COLS", "CUMULATIVE_COUNT_COLS", "STEP_PROGRESS_COLS",
    "AUX_GLOBAL_COUNTER_COLS",
    "COUNT_SUBTYPE_MAP", "SESSION_COUNT_COLS", "ACCUMULATOR_COUNT_COLS", "EVENT_COUNT_COLS",
    "SYNTHESIS_SUBFAMILY_MAP",
    "CLASS_AUDIT_DF", "CLASS_SUSPICIOUS_DF", "CLASS_AUDIT_CSV",
    "BIN_NON01_COLS", "MULTI_BLOCK_OVERLAP",
    "OTHER_SUBTYPE_COUNTS", "SYNTHESIS_SUBFAMILY_COUNTS",
    "STATE_CLUSTER_AUDIT_DF", "STATE_CLUSTER_AUDIT_CSV",
    "state_labels_tr", "state_km", "state_names",
    "IOT_SYN_DRIVERS",
    "drv_real_rate", "drv_syn_rate", "drv_real_mean", "drv_syn_mean",
    "base_offset", "jitter_sample",
    "_related_driver_cols_for_entity",
    "_entity_activity_from_driver_df",
    "CONT_ROUTING_POLICY_DF",
    "CONT_ROUTING_POLICY_CSV",
    "PROTOCOL_CONTEXT_TEST",
    "PROTOCOL_SYN_MASKS_TEST",
    "A2_PROTOCOL_FINAL",
    "CELL12A_FOUNDATION_MANIFEST",
    "CELL12A_FOUNDATION_MANIFEST_JSON",
    "FULL_IOT_NAMESPACE_COLS",
    "STATE_TARGET_COLS",
    "IOT_DRIVER_TARGETS",
    "IOT_BINARY_TARGETS",
    "IOT_CONTINUOUS_VALUE_TARGETS",
    "IOT_STATE_TARGETS",
    "IOT_NONMODELED_META_COLS",
    "IOT_NONMODELED_EXCLUDED_COLS",
    "IOT_PHASE2_TARGET_PLAN",
    "QUALITY_DIMENSION_REGISTRY",
    "A0_A1_A2_VARIANT_MANIFEST",
    "QUALITY_ATTRIBUTION_MATRIX",
    "CELL11_IOT_DOWNSTREAM_PROTOCOL_CONTEXT",
    "CELL11_IOT_DOWNSTREAM_PROTOCOL_ACTIVITY_SUMMARY",
    "IOT_ROLE_OWNERSHIP_DF",
    "IOT_PARTITION_CONTRACT",
    "IOT_CELL_ROUTING_MANIFEST",
    "BINARY_ROUTING_POLICY_DF",
    "BINARY_ROUTING_POLICY_CSV",
    "BINARY_ROUTING_POLICY_MANIFEST",
    "BINARY_ROUTING_POLICY_MANIFEST_JSON",
    "IOT_TEST_VALUE_ACCESS_POLICY",
]

missing_12a_outputs = [k for k in required_12a_outputs if k not in globals()]
if missing_12a_outputs:
    raise RuntimeError(f"[Cell12.a] Missing expected output globals: {missing_12a_outputs}")

log(
    f"[Cell12.a] Ready for Cell 12.b | "
    f"full_iot_namespace={len(FULL_IOT_NAMESPACE_COLS)} | "
    f"drivers={len(DRV_COLS)} | "
    f"binary_targets={len(BIN_COLS)} | "
    f"continuous_value_targets={len(CONT_VALUE_COLS)} | "
    f"state_targets={len(STATE_TARGET_COLS)} | "
    f"meta={len(IOT_META)} | excluded={len(IOT_EXCLUDED)} | "
    f"cont_mask={len(CONT_MASK_COLS)} | "
    f"ddpm={len(DDPM_COLS)} | non_ddpm={len(NON_DDPM_COLS)} | "
    f"binary_policy_buckets={BINARY_ROUTING_POLICY_DF['policy_bucket'].nunique()} | "
    f"Q_dimensions={len(_qdims)} | "
    f"protocol_context={PROTOCOL_CONTEXT_TEST.shape} | "
    f"manifest={CELL12A_FOUNDATION_MANIFEST_JSON}"
)

gc.collect()

log("--- END: Cell 12.a — IoT synthesis foundation (v24.1-THESIS phase1-quality-aware, contract-locked) ---")