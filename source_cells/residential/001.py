# ============================================================
# Cell 1: imports + config + determinism + canonical contract
# v4.1-THESIS-managed-Jupyter-safe
#
# Replacement for the pasted Cell 1. This version fixes:
# - incorrect hard failure when PYTHONHASHSEED cannot be set in managed Jupyter
# - undefined _HARD_FAIL_MISSING_PYTHONHASHSEED
# - public-scope accounting after corrected sparse-driver audit: 46 logical cols
# - explicit managed_jupyter_best_effort reproducibility contract
# - artifact-defining stable sorting helpers
# - leakage, C2ST, artifact, and release contracts written to disk
# ============================================================

import os, sys, json, time, math, random, hashlib, platform, warnings, gc, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence, Any

# -------------------------
# Environment controls
# -------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

EXPECTED_PYTHONHASHSEED = "0"
PYTHONHASHSEED_VALUE = str(os.environ.get("PYTHONHASHSEED", "")).strip()
PYTHONHASHSEED_LOCKED_AT_START = bool(PYTHONHASHSEED_VALUE == EXPECTED_PYTHONHASHSEED)

# Optional strict archival mode only. Default must remain 0 for managed OnDemand/Jupyter.
_HARD_FAIL_MISSING_PYTHONHASHSEED = (
    os.environ.get("CPS_REQUIRE_STRICT_PYTHONHASHSEED", "0").strip().lower()
    in {"1", "true", "yes", "y"}
)

if _HARD_FAIL_MISSING_PYTHONHASHSEED and not PYTHONHASHSEED_LOCKED_AT_START:
    raise RuntimeError(
        "[Cell1] CPS_REQUIRE_STRICT_PYTHONHASHSEED=1 but PYTHONHASHSEED=0 was not "
        "present before kernel startup. This cannot be fixed with %%bash inside a running notebook. "
        "Restart the Jupyter kernel from a launcher that sets PYTHONHASHSEED=0, or unset "
        "CPS_REQUIRE_STRICT_PYTHONHASHSEED for managed-Jupyter best-effort mode."
    )

FINAL_REPRO_MODE = "strict_pythonhashseed_zero" if PYTHONHASHSEED_LOCKED_AT_START else "managed_jupyter_best_effort"
PYTHONHASHSEED_POLICY = (
    "locked_at_interpreter_start_strict_zero"
    if PYTHONHASHSEED_LOCKED_AT_START
    else "not_locked_at_interpreter_start_recorded_best_effort"
)

if not PYTHONHASHSEED_LOCKED_AT_START:
    warnings.warn(
        "[Cell1] PYTHONHASHSEED was not set to 0 before kernel startup. Continuing in "
        "managed_jupyter_best_effort mode. Do not claim bitwise reproducibility; report "
        "runtime seeds, stable artifact ordering, and checksums.",
        RuntimeWarning,
    )

warnings.filterwarnings("default", category=FutureWarning)

# -------------------------
# Third-party imports
# -------------------------
import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise ImportError("[Cell1] pyarrow is required for metadata-only Parquet checks.") from e

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("default", category=ConvergenceWarning)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import sklearn
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Helpers
# -------------------------
def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), "|", msg, flush=True)


def _normpath(p: Optional[str]) -> str:
    if p is None:
        return ""
    p = str(p).strip().strip('"').strip("'")
    return os.path.abspath(os.path.expanduser(p)) if p else ""


def _ensure_file(path: str, what: str) -> None:
    if not path:
        raise ValueError(f"{what} resolved to an empty path.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} does not exist or is not a regular file: {path}")


def _ensure_dir(path: str, what: str) -> None:
    if not path:
        raise ValueError(f"{what} resolved to an empty path.")
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{what} is not a directory after creation attempt: {path}")


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, default=str)


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def stable_sorted_str_list(values) -> List[str]:
    """Canonical ordering helper for artifact-defining column lists."""
    if values is None:
        return []
    return sorted(map(str, list(values)))


def stable_unique_sorted_str_list(values) -> List[str]:
    """Canonical sorted unique string list."""
    if values is None:
        return []
    return sorted(set(map(str, list(values))))


def _sha256_file(path: str, block_size: int = 16 * 1024 * 1024) -> str:
    _ensure_file(path, "file for sha256")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _cfg_from_pairs(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    seen, dupes = set(), []
    for k, v in pairs:
        if k in seen:
            dupes.append(k)
        seen.add(k)
        out[k] = v
    if dupes:
        raise ValueError(f"[Cell1] Duplicate CFG keys are not allowed: {sorted(set(dupes))}")
    return out


def _jsonable_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    clean = {}
    for k, v in cfg.items():
        if isinstance(v, Path):
            clean[k] = str(v)
        elif isinstance(v, tuple):
            clean[k] = list(v)
        elif isinstance(v, np.integer):
            clean[k] = int(v)
        elif isinstance(v, np.floating):
            clean[k] = float(v)
        elif isinstance(v, np.bool_):
            clean[k] = bool(v)
        else:
            clean[k] = v
    return clean


def _validate_positive_int_cfg(cfg: Dict[str, Any], key: str) -> None:
    val = int(cfg[key])
    if val <= 0:
        raise ValueError(f"[Cell1] CFG['{key}'] must be positive. Got {val}.")


def _validate_nonnegative_int_cfg(cfg: Dict[str, Any], key: str) -> None:
    val = int(cfg[key])
    if val < 0:
        raise ValueError(f"[Cell1] CFG['{key}'] must be non-negative. Got {val}.")


def _validate_positive_float_cfg(cfg: Dict[str, Any], key: str) -> None:
    val = float(cfg[key])
    if not np.isfinite(val) or val <= 0:
        raise ValueError(f"[Cell1] CFG['{key}'] must be finite and > 0. Got {val}.")


def _validate_probability_cfg(cfg: Dict[str, Any], key: str) -> None:
    val = float(cfg[key])
    if not np.isfinite(val) or not (0.0 <= val <= 1.0):
        raise ValueError(f"[Cell1] CFG['{key}'] must be in [0, 1]. Got {val}.")


def _read_parquet_contract_metadata(path: str) -> Dict[str, Any]:
    _ensure_file(path, "CFG['input_parquet']")
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    meta = pf.metadata
    return {
        "num_rows": int(meta.num_rows),
        "num_columns": int(len(names)),
        "num_row_groups": int(meta.num_row_groups),
        "schema_names": names,
        "created_by": str(getattr(meta, "created_by", "")),
        "serialized_size": int(getattr(meta, "serialized_size", 0) or 0),
    }


def _inspect_pandas_parquet_index_metadata(path: str) -> List[str]:
    try:
        pf = pq.ParquetFile(path)
        meta = pf.metadata.metadata or {}
        if b"pandas" not in meta:
            return []
        pandas_meta = json.loads(meta[b"pandas"].decode("utf-8"))
        return list(map(str, pandas_meta.get("index_columns", []) or []))
    except Exception as e:
        log(f"[WARN] Could not inspect pandas Parquet index metadata: {e}")
        return []


def _read_epoch_seconds_from_parquet_time_col(parquet_path: str, raw_time_col: str) -> Tuple[np.ndarray, str, str]:
    try:
        tdf = pd.read_parquet(parquet_path, columns=[raw_time_col])
    except Exception as e:
        raise RuntimeError(f"[Cell1] Failed to read raw time column '{raw_time_col}'.") from e
    s = tdf[raw_time_col]
    if s.isna().any():
        raise ValueError(f"[Cell1] Raw time column contains NaN/NaT: {raw_time_col}")
    if pd.api.types.is_datetime64_any_dtype(s):
        sec = (s.astype("int64") // 1_000_000_000).to_numpy(dtype=np.int64)
        return sec, str(s.dtype), "datetime64"
    sec_float = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(sec_float).all():
        raise ValueError(f"[Cell1] Raw time column contains non-finite values: {raw_time_col}")
    rounded = np.rint(sec_float)
    max_abs_frac = float(np.max(np.abs(sec_float - rounded))) if len(sec_float) else 0.0
    if max_abs_frac > 1e-6:
        raise ValueError(f"[Cell1] Raw time column is not integer-like epoch seconds; max_abs_frac={max_abs_frac}")
    return rounded.astype(np.int64), str(s.dtype), str(s.dtype)


def _build_split_duration_audit(sec: np.ndarray, n_train: int, n_val: int, n_test: int) -> pd.DataFrame:
    if int(n_train) + int(n_val) + int(n_test) != len(sec):
        raise ValueError("[Cell1] Split sizes do not sum to observed time length.")
    intervals = [
        ("raw_reference_trace", 0, len(sec) - 1),
        ("TRAIN", 0, int(n_train) - 1),
        ("VAL", int(n_train), int(n_train) + int(n_val) - 1),
        ("TEST", int(n_train) + int(n_val), len(sec) - 1),
    ]
    rows = []
    for name, start_idx, end_idx in intervals:
        first_sec, last_sec = int(sec[start_idx]), int(sec[end_idx])
        duration_seconds_inclusive = int(last_sec - first_sec + 1)
        n_rows = int(end_idx - start_idx + 1)
        rows.append({
            "scope": name,
            "start_row_index": int(start_idx),
            "end_row_index_inclusive": int(end_idx),
            "n_rows": n_rows,
            "first_epoch_sec": first_sec,
            "last_epoch_sec": last_sec,
            "duration_seconds_inclusive": duration_seconds_inclusive,
            "duration_days_inclusive": float(duration_seconds_inclusive / 86400.0),
            "rows_equal_duration_seconds": bool(n_rows == duration_seconds_inclusive),
        })
    return pd.DataFrame(rows)


def _validate_epoch_second_time_contract(
    parquet_path: str,
    raw_time_col: str,
    expected_rows: int,
    expected_hz: int,
    require_monotonic: bool,
    require_unique: bool,
    n_train: int,
    n_val: int,
    n_test: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    sec, raw_dtype, raw_kind = _read_epoch_seconds_from_parquet_time_col(parquet_path, raw_time_col)
    if len(sec) != int(expected_rows):
        raise ValueError(f"[Cell1] Time row-count mismatch: observed={len(sec)} expected={expected_rows}")
    diffs = np.diff(sec)
    if require_monotonic and np.any(diffs <= 0):
        bad_idx = int(np.where(diffs <= 0)[0][0])
        raise ValueError(f"[Cell1] Time is not strictly increasing at index {bad_idx}.")
    if require_unique and len(np.unique(sec)) != len(sec):
        raise ValueError("[Cell1] Raw time column contains duplicate timestamps.")
    expected_step = int(round(1.0 / float(expected_hz)))
    if len(diffs):
        unique_steps, step_counts = np.unique(diffs, return_counts=True)
        step_count_map = {int(k): int(v) for k, v in zip(unique_steps, step_counts)}
        bad_steps = {int(k): int(v) for k, v in step_count_map.items() if int(k) != expected_step}
    else:
        step_count_map, bad_steps = {}, {}
    if bad_steps:
        raise ValueError(f"[Cell1] Raw time column is not a strict 1 Hz grid: {bad_steps}")
    split_df = _build_split_duration_audit(sec, n_train, n_val, n_test)
    contract = {
        "raw_time_col": str(raw_time_col),
        "raw_time_dtype": str(raw_dtype),
        "raw_time_kind": str(raw_kind),
        "n": int(len(sec)),
        "first_sec": int(sec[0]),
        "last_sec": int(sec[-1]),
        "duration_seconds_inclusive": int(sec[-1] - sec[0] + 1),
        "duration_days_inclusive": float((int(sec[-1] - sec[0] + 1)) / 86400.0),
        "min_step": int(diffs.min()) if len(diffs) else None,
        "max_step": int(diffs.max()) if len(diffs) else None,
        "median_step": float(np.median(diffs)) if len(diffs) else None,
        "step_counts": step_count_map,
        "is_strictly_increasing": bool(np.all(diffs > 0)) if len(diffs) else True,
        "is_unique": bool(len(np.unique(sec)) == len(sec)),
        "is_epoch_second_plausible": bool(sec[0] > 1_000_000_000 and sec[-1] > sec[0]),
        "split_duration_audit": split_df.to_dict(orient="records"),
    }
    return contract, split_df

# -------------------------
# Canonical configuration
# -------------------------
CFG: Dict[str, Any] = _cfg_from_pairs([
    ("project_line", "v4.4.7.32-FULL-COUPLED"),
    ("notebook_line", "STUDY-THESIS-final-governed-public-scope-fixed"),
    ("input_parquet", _normpath(os.environ.get("CPS_INPUT_PARQUET", "cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet"))),
    ("outdir", _normpath(os.environ.get("CPS_OUTDIR", "cps_synth_v4_4_7_32_BINARY_V4_DEV"))),

    ("raw_time_col", "sec"),
    ("canonical_time_col", "sec_epoch_s__canon"),
    ("expected_hz", 1),
    ("warmup_trim_seconds", 300),
    ("time_col_candidates", ["sec_epoch_s__canon", "sec", "sec_epoch_s", "ts", "timestamp", "time_s"]),
    ("require_monotonic_time", True),
    ("require_unique_time", True),
    ("materialize_canonical_time_in_cell2", True),

    ("expected_total_rows", 1277694),
    ("expected_total_columns", 740),
    ("expected_total_columns_hint", 740),
    ("n_train", 766616),
    ("n_val", 255538),
    ("n_test", 255540),

    ("expected_iot_namespace_cols", 530),
    ("expected_iot_continuous_source_cols", 148),
    ("expected_iot_continuous_final_owned_cols", 147),
    ("expected_iot_binary_cols", 52),
    ("expected_iot_sparse_driver_cols", 26),
    ("expected_iot_observability_mask_cols", 183),
    ("expected_iot_excluded_cols", 72),
    ("expected_iot_placeholder_cols", 50),
    ("expected_full_scientific_logical_cols", 555),
    ("expected_router_scientific_cols", 18),
    ("expected_ota_scientific_cols", 2),
    ("expected_zigbee_scientific_cols", 4),
    ("expected_zwave_scientific_value_cols", 0),
    ("expected_protocol_scientific_cols", 24),

    # Corrected Q6 public-release contract after driver battery regrade.
    ("expected_public_logical_cols", 46),
    ("expected_public_time_cols", 1),
    ("expected_public_router_cols", 18),
    ("expected_public_zigbee_cols", 4),
    ("expected_public_sparse_driver_cols", 23),
    ("expected_sparse_driver_corrected_pass_cols", 13),
    ("expected_sparse_driver_corrected_warning_cols", 10),
    ("expected_sparse_driver_corrected_fatal_cols", 3),
    ("expected_sparse_driver_release_eligible_cols", 23),
    ("allow_pandas_index_physical_extra_col", True),

    ("expected_proto_value_cols_core", 54),
    ("expected_events_entity_cols", 12),
    ("expected_events_feat_cols", 14),
    ("expected_zb_driver_cols", 2),
    ("expected_iot_meta_cols", 181),
    ("expected_iot_entity_obs_cols", 90),
    ("expected_iot_entity_stale_cols", 90),
    ("expected_ddpm_tiers_used", ["router", "ota", "zigbee"]),
    ("expected_protocol_value_tiers", ["router", "ota", "zigbee"]),

    ("drop_all_zero_zwave_obs_from_modeling", True),
    ("allow_schema_placeholder_columns", True),
    ("strict_schema_checks", True),
    ("strict_contract_checks", True),
    ("forbid_duplicate_columns", True),
    ("forbid_object_dtype_features", True),

    ("selection_split_policy", "TRAIN_fit_VAL_select_TEST_audit_only"),
    ("allow_test_values_for_fitting", False),
    ("allow_test_values_for_threshold_selection", False),
    ("allow_test_values_for_candidate_selection", False),
    ("allow_test_values_for_repair", False),
    ("allow_test_values_for_artifact_overwrite", False),
    ("cell11b_apply_test_regression_quarantine", False),
    ("cell12c4b_apply_test_derived_repairs", False),
    ("cell12c4c_keep_or_revert_using_test_metrics", False),
    ("cell12f4b_apply_test_derived_observability_repairs", False),
    ("q4_allow_precomputed_or_restored_promotion", False),

    ("seed", int(os.environ.get("CPS_SEED", "1337"))),
    ("default_float", os.environ.get("CPS_DEFAULT_FLOAT", "float32").lower()),
    ("determinism_mode", os.environ.get("CPS_DETERMINISM_MODE", "best_effort").lower()),
    ("final_repro_mode", FINAL_REPRO_MODE),
    ("require_pythonhashseed", bool(_HARD_FAIL_MISSING_PYTHONHASHSEED)),
    ("pythonhashseed_locked_at_start", bool(PYTHONHASHSEED_LOCKED_AT_START)),
    ("pythonhashseed_value", PYTHONHASHSEED_VALUE),
    ("pythonhashseed_policy", PYTHONHASHSEED_POLICY),
    ("stable_sort_artifact_lists", True),
    ("input_sha256_enabled", os.environ.get("CPS_INPUT_SHA256", "1").strip().lower() in {"1", "true", "yes", "y"}),
    ("torch_num_threads", int(os.environ.get("CPS_TORCH_THREADS", "1"))),
    ("torch_num_interop_threads", int(os.environ.get("CPS_TORCH_INTEROP_THREADS", "1"))),

    ("ddpm_timesteps", 200),
    ("ddpm_epochs", 12),
    ("ddpm_batch", 128),
    ("ddpm_lr", 2e-4),
    ("ddpm_seq_len", 60),
    ("ddpm_stride", 5),
    ("block_len", 120),
    ("ddpm_scale_clip", 8.0),
    ("ddpm_width", 256),
    ("ddpm_depth", 4),
    ("ddpm_ema_decay", 0.999),
    ("ddpm_use_amp", True),
    ("ddpm_cond_hist_k", 10),
    ("iot_cond_hist_k", 10),
    ("ddpm_strict_countlike_match", True),

    ("mask_source", "val"),
    ("mask_repair_policy", "same_split_only"),
    ("strict_mask_writeback", True),
    ("never_impute_unobserved_values", True),
    ("strict_presence_value_semantics", True),

    ("other_rest_enable", True),
    ("runvalid_max_tries", 200),
    ("roll_windows", (5, 30)),
    ("coupling_enable", True),
    ("coupling_window_pre", 2),
    ("coupling_window_post", 3),
    ("coupling_min_prob", 0.70),
    ("q4_eta_blocker_min", 0.50),
    ("q4_manifest_profile_similarity_blocker_min", 0.50),
    ("q4_lag_peak_error_blocker_sec", 5.0),
    ("q4_response_rate_warning_abs", 0.10),
    ("q4_response_rate_blocker_abs", 0.30),
    ("mixed_anchor_bucket_calibration_enable", True),

    ("continuous_target_ready_taus", (0.55, 0.60)),
    ("continuous_ks_max", 0.50),
    ("continuous_wasserstein_norm_max", 2.0),
    ("c2st_max_rows", 120000),
    ("c2st_solver", "saga"),
    ("c2st_max_iter", 5000),
    ("c2st_model_family", "logistic_regression"),
    ("c2st_auc_orientation", "separability_max_auc_1_minus_auc"),
    ("c2st_split_policy", "temporal_blocked"),
    ("c2st_block_len_seconds", 3600),
    ("c2st_repetitions", 5),
    ("c2st_test_frac", 0.35),
    ("c2st_scale_features", True),
    ("bootstrap_policy", "temporal_block"),
    ("bootstrap_block_len_seconds", 3600),
])

# -------------------------
# Config validation
# -------------------------
if CFG["default_float"] not in {"float32", "float64"}:
    raise ValueError(f"Unsupported default_float: {CFG['default_float']}")
if CFG["determinism_mode"] not in {"best_effort", "strict"}:
    raise ValueError("determinism_mode must be 'best_effort' or 'strict'.")
if int(CFG["expected_hz"]) != 1:
    raise ValueError("expected_hz must remain 1.")
if int(CFG["expected_total_columns"]) != int(CFG["expected_total_columns_hint"]):
    raise ValueError("expected_total_columns and hint disagree.")

_positive_int_keys = [
    "expected_total_rows", "expected_total_columns", "n_train", "n_val", "n_test",
    "expected_iot_namespace_cols", "expected_iot_continuous_source_cols", "expected_iot_continuous_final_owned_cols",
    "expected_iot_binary_cols", "expected_iot_sparse_driver_cols", "expected_iot_observability_mask_cols",
    "expected_iot_excluded_cols", "expected_iot_placeholder_cols", "expected_full_scientific_logical_cols",
    "expected_protocol_scientific_cols", "expected_public_logical_cols", "expected_public_time_cols",
    "expected_public_router_cols", "expected_public_zigbee_cols", "expected_public_sparse_driver_cols",
    "expected_sparse_driver_corrected_pass_cols", "expected_sparse_driver_corrected_warning_cols",
    "expected_sparse_driver_corrected_fatal_cols", "expected_sparse_driver_release_eligible_cols",
    "ddpm_timesteps", "ddpm_epochs", "ddpm_batch", "ddpm_seq_len", "ddpm_stride", "block_len",
    "ddpm_width", "ddpm_depth", "ddpm_cond_hist_k", "iot_cond_hist_k", "runvalid_max_tries",
    "c2st_max_rows", "c2st_max_iter", "c2st_block_len_seconds", "c2st_repetitions", "bootstrap_block_len_seconds",
]
for _key in _positive_int_keys:
    _validate_positive_int_cfg(CFG, _key)
for _key in ("coupling_window_pre", "coupling_window_post", "expected_zwave_scientific_value_cols"):
    _validate_nonnegative_int_cfg(CFG, _key)
for _key in ("ddpm_lr", "ddpm_scale_clip", "q4_lag_peak_error_blocker_sec"):
    _validate_positive_float_cfg(CFG, _key)
for _key in ("ddpm_ema_decay", "coupling_min_prob", "c2st_test_frac", "q4_eta_blocker_min", "q4_manifest_profile_similarity_blocker_min", "q4_response_rate_warning_abs", "q4_response_rate_blocker_abs"):
    _validate_probability_cfg(CFG, _key)

if float(CFG["q4_response_rate_warning_abs"]) >= float(CFG["q4_response_rate_blocker_abs"]):
    raise ValueError("q4 response warning threshold must be lower than blocker threshold.")
if int(CFG["block_len"]) < int(CFG["ddpm_seq_len"]):
    raise ValueError("block_len must be >= ddpm_seq_len.")

_taus = tuple(float(x) for x in CFG["continuous_target_ready_taus"])
if tuple(sorted(_taus)) != _taus or any((not np.isfinite(x)) or x < 0.5 or x > 1.0 for x in _taus):
    raise ValueError(f"continuous_target_ready_taus must be sorted finite values in [0.5,1.0]. Got {_taus}")
CFG["continuous_target_ready_taus"] = _taus

if str(CFG["mask_source"]).lower() != "val":
    raise ValueError("mask_source must remain 'val'.")
if str(CFG["c2st_split_policy"]) != "temporal_blocked":
    raise ValueError("c2st_split_policy must be temporal_blocked.")
if str(CFG["c2st_auc_orientation"]) != "separability_max_auc_1_minus_auc":
    raise ValueError("C2ST AUC orientation must be separability_max_auc_1_minus_auc.")
CFG["c2st_requires_scaled_features"] = bool(CFG["c2st_scale_features"] or str(CFG["c2st_solver"]).lower() in {"sag", "saga"})

if int(CFG["n_train"]) + int(CFG["n_val"]) + int(CFG["n_test"]) != int(CFG["expected_total_rows"]):
    raise ValueError("split sizes do not sum to expected_total_rows.")

_iot_sum = sum(int(CFG[k]) for k in [
    "expected_iot_continuous_final_owned_cols", "expected_iot_binary_cols", "expected_iot_sparse_driver_cols",
    "expected_iot_observability_mask_cols", "expected_iot_excluded_cols", "expected_iot_placeholder_cols",
])
if _iot_sum != int(CFG["expected_iot_namespace_cols"]):
    raise ValueError(f"IoT namespace accounting mismatch: {_iot_sum}")

_protocol_sum = int(CFG["expected_router_scientific_cols"]) + int(CFG["expected_ota_scientific_cols"]) + int(CFG["expected_zigbee_scientific_cols"]) + int(CFG["expected_zwave_scientific_value_cols"])
if _protocol_sum != int(CFG["expected_protocol_scientific_cols"]):
    raise ValueError(f"Protocol scientific accounting mismatch: {_protocol_sum}")

_public_sum = int(CFG["expected_public_time_cols"]) + int(CFG["expected_public_router_cols"]) + int(CFG["expected_public_zigbee_cols"]) + int(CFG["expected_public_sparse_driver_cols"])
if _public_sum != int(CFG["expected_public_logical_cols"]):
    raise ValueError(f"Public artifact accounting mismatch: {_public_sum}")

_driver_status_sum = int(CFG["expected_sparse_driver_corrected_pass_cols"]) + int(CFG["expected_sparse_driver_corrected_warning_cols"]) + int(CFG["expected_sparse_driver_corrected_fatal_cols"])
if _driver_status_sum != int(CFG["expected_iot_sparse_driver_cols"]):
    raise ValueError("Corrected sparse-driver status counts do not sum to 26.")
if int(CFG["expected_sparse_driver_corrected_pass_cols"]) + int(CFG["expected_sparse_driver_corrected_warning_cols"]) != int(CFG["expected_sparse_driver_release_eligible_cols"]):
    raise ValueError("Sparse-driver release-eligible count must equal corrected pass+warning count.")
if int(CFG["expected_sparse_driver_release_eligible_cols"]) != int(CFG["expected_public_sparse_driver_cols"]):
    raise ValueError("Public sparse-driver count must equal corrected release-eligible count.")

_full_scientific_sum = int(CFG["expected_iot_namespace_cols"]) + int(CFG["expected_protocol_scientific_cols"]) + 1
if _full_scientific_sum != int(CFG["expected_full_scientific_logical_cols"]):
    raise ValueError("Full scientific logical accounting mismatch.")

for _flag in [
    "allow_test_values_for_fitting", "allow_test_values_for_threshold_selection", "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair", "allow_test_values_for_artifact_overwrite", "cell11b_apply_test_regression_quarantine",
    "cell12c4b_apply_test_derived_repairs", "cell12c4c_keep_or_revert_using_test_metrics",
    "cell12f4b_apply_test_derived_observability_repairs", "q4_allow_precomputed_or_restored_promotion",
]:
    if bool(CFG[_flag]):
        raise ValueError(f"Final governed notebook forbids CFG['{_flag}']=True.")

# -------------------------
# Filesystem and dataset contract validation
# -------------------------
_ensure_file(str(CFG["input_parquet"]), "CFG['input_parquet']")
_ensure_dir(str(CFG["outdir"]), "CFG['outdir']")
for sub in ("reports", "synthetic", "artifacts", "artifacts/contracts"):
    _ensure_dir(os.path.join(str(CFG["outdir"]), sub), f"outdir/{sub}")

PARQUET_META = _read_parquet_contract_metadata(str(CFG["input_parquet"]))
_schema_names = list(PARQUET_META["schema_names"])
_schema_name_set = set(_schema_names)

if bool(CFG["strict_contract_checks"]):
    if int(PARQUET_META["num_rows"]) != int(CFG["expected_total_rows"]):
        raise ValueError(f"Input Parquet row-count drift: {PARQUET_META['num_rows']} != {CFG['expected_total_rows']}")
    if int(PARQUET_META["num_columns"]) != int(CFG["expected_total_columns"]):
        raise ValueError(f"Input Parquet column-count drift: {PARQUET_META['num_columns']} != {CFG['expected_total_columns']}")
if bool(CFG["forbid_duplicate_columns"]) and len(_schema_names) != len(set(_schema_names)):
    dupes = sorted({x for x in _schema_names if _schema_names.count(x) > 1})
    raise ValueError(f"Duplicate columns detected in input schema: {dupes[:20]}")

_pandas_index_columns = _inspect_pandas_parquet_index_metadata(str(CFG["input_parquet"]))
_present_time_candidates = [c for c in CFG["time_col_candidates"] if c in _schema_name_set]
_missing_time_candidates = [c for c in CFG["time_col_candidates"] if c not in _schema_name_set]
_index_time_candidates = [c for c in CFG["time_col_candidates"] if c in set(map(str, _pandas_index_columns))]

if str(CFG["raw_time_col"]) in _schema_name_set:
    CFG["detected_time_source"] = str(CFG["raw_time_col"])
    CFG["detected_time_source_kind"] = "configured_raw_schema_column"
elif str(CFG["canonical_time_col"]) in _schema_name_set:
    CFG["detected_time_source"] = str(CFG["canonical_time_col"])
    CFG["detected_time_source_kind"] = "canonical_schema_column"
elif _present_time_candidates:
    CFG["detected_time_source"] = str(_present_time_candidates[0])
    CFG["detected_time_source_kind"] = "candidate_schema_column"
elif _index_time_candidates:
    CFG["detected_time_source"] = str(_index_time_candidates[0])
    CFG["detected_time_source_kind"] = "candidate_pandas_index"
else:
    raise ValueError(
        "[Cell1] No usable time source found. "
        f"schema_sample={_schema_names[:30]} pandas_index_columns={_pandas_index_columns}"
    )

if str(CFG["canonical_time_col"]) not in _schema_name_set:
    log(f"[INFO] canonical_time_col absent; Cell 2 must materialize {CFG['canonical_time_col']} from {CFG['detected_time_source']}.")

TIME_CONTRACT, SPLIT_DURATION_AUDIT = _validate_epoch_second_time_contract(
    parquet_path=str(CFG["input_parquet"]),
    raw_time_col=str(CFG["detected_time_source"]),
    expected_rows=int(CFG["expected_total_rows"]),
    expected_hz=int(CFG["expected_hz"]),
    require_monotonic=bool(CFG["require_monotonic_time"]),
    require_unique=bool(CFG["require_unique_time"]),
    n_train=int(CFG["n_train"]),
    n_val=int(CFG["n_val"]),
    n_test=int(CFG["n_test"]),
)
if not bool(TIME_CONTRACT["is_epoch_second_plausible"]):
    raise ValueError(f"Detected time source is not plausible Unix epoch seconds: {TIME_CONTRACT}")
CFG["detected_time_first_sec"] = int(TIME_CONTRACT["first_sec"])
CFG["detected_time_last_sec"] = int(TIME_CONTRACT["last_sec"])
CFG["detected_time_duration_seconds_inclusive"] = int(TIME_CONTRACT["duration_seconds_inclusive"])

INPUT_PARQUET_SHA256 = _sha256_file(str(CFG["input_parquet"])) if bool(CFG["input_sha256_enabled"]) else "DISABLED_BY_CPS_INPUT_SHA256"

# -------------------------
# Device and determinism setup
# -------------------------
def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

CFG["device"] = _pick_device()
CFG["ddpm_amp_effective"] = bool(CFG["ddpm_use_amp"] and CFG["device"] == "cuda")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            pass
    try:
        if torch.get_num_threads() != int(CFG["torch_num_threads"]):
            torch.set_num_threads(int(CFG["torch_num_threads"]))
    except Exception as e:
        log(f"[WARN] torch.set_num_threads failed: {e}")
    try:
        if torch.get_num_interop_threads() != int(CFG["torch_num_interop_threads"]):
            torch.set_num_interop_threads(int(CFG["torch_num_interop_threads"]))
    except Exception as e:
        log(f"[NOTE] torch.set_num_interop_threads not applied: {e}")
    try:
        if CFG["determinism_mode"] == "strict":
            torch.use_deterministic_algorithms(True)
        else:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            log(f"[WARN] Deterministic algorithms not fully enforced: {e}")
    except Exception as e:
        log(f"[WARN] Deterministic algorithms not fully enforced: {e}")
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.set_default_dtype(torch.float32 if CFG["default_float"] == "float32" else torch.float64)

set_seed(int(CFG["seed"]))
NP_RNG = np.random.default_rng(int(CFG["seed"]))
TORCH_CPU_GENERATOR = torch.Generator(device="cpu").manual_seed(int(CFG["seed"]))
if CFG["device"] == "cuda":
    try:
        TORCH_DEVICE_GENERATOR = torch.Generator(device="cuda").manual_seed(int(CFG["seed"]))
    except Exception:
        TORCH_DEVICE_GENERATOR = None
else:
    TORCH_DEVICE_GENERATOR = None

# -------------------------
# Contracts and metadata
# -------------------------
ARTIFACT_ACCOUNTING_CONTRACT = {
    "iot_namespace_cols": int(CFG["expected_iot_namespace_cols"]),
    "iot_namespace_components": {
        "continuous_source_cols": int(CFG["expected_iot_continuous_source_cols"]),
        "continuous_final_owned_cols": int(CFG["expected_iot_continuous_final_owned_cols"]),
        "binary_cols": int(CFG["expected_iot_binary_cols"]),
        "sparse_driver_cols": int(CFG["expected_iot_sparse_driver_cols"]),
        "observability_mask_cols": int(CFG["expected_iot_observability_mask_cols"]),
        "excluded_cols": int(CFG["expected_iot_excluded_cols"]),
        "placeholder_cols": int(CFG["expected_iot_placeholder_cols"]),
    },
    "protocol_scientific_cols": int(CFG["expected_protocol_scientific_cols"]),
    "protocol_scientific_components": {
        "router": int(CFG["expected_router_scientific_cols"]),
        "ota": int(CFG["expected_ota_scientific_cols"]),
        "zigbee": int(CFG["expected_zigbee_scientific_cols"]),
        "zwave_value_cols": int(CFG["expected_zwave_scientific_value_cols"]),
    },
    "full_scientific_logical_cols": int(CFG["expected_full_scientific_logical_cols"]),
    "public_logical_cols": int(CFG["expected_public_logical_cols"]),
    "public_components": {
        "time": int(CFG["expected_public_time_cols"]),
        "router": int(CFG["expected_public_router_cols"]),
        "zigbee": int(CFG["expected_public_zigbee_cols"]),
        "sparse_driver": int(CFG["expected_public_sparse_driver_cols"]),
    },
    "sparse_driver_corrected_release_contract": {
        "total_driver_cols": int(CFG["expected_iot_sparse_driver_cols"]),
        "corrected_pass_cols": int(CFG["expected_sparse_driver_corrected_pass_cols"]),
        "corrected_warning_cols": int(CFG["expected_sparse_driver_corrected_warning_cols"]),
        "corrected_fatal_cols": int(CFG["expected_sparse_driver_corrected_fatal_cols"]),
        "release_eligible_cols": int(CFG["expected_sparse_driver_release_eligible_cols"]),
        "policy": "corrected_fatal_sparse_drivers_are_excluded_from_Q6_public_baseline",
    },
    "allow_pandas_index_physical_extra_col": bool(CFG["allow_pandas_index_physical_extra_col"]),
}

LEAKAGE_POLICY_CONTRACT = {
    "selection_split_policy": str(CFG["selection_split_policy"]),
    "allow_test_values_for_fitting": bool(CFG["allow_test_values_for_fitting"]),
    "allow_test_values_for_threshold_selection": bool(CFG["allow_test_values_for_threshold_selection"]),
    "allow_test_values_for_candidate_selection": bool(CFG["allow_test_values_for_candidate_selection"]),
    "allow_test_values_for_repair": bool(CFG["allow_test_values_for_repair"]),
    "allow_test_values_for_artifact_overwrite": bool(CFG["allow_test_values_for_artifact_overwrite"]),
    "disabled_cells_or_paths": {
        "cell11b_apply_test_regression_quarantine": bool(CFG["cell11b_apply_test_regression_quarantine"]),
        "cell12c4b_apply_test_derived_repairs": bool(CFG["cell12c4b_apply_test_derived_repairs"]),
        "cell12c4c_keep_or_revert_using_test_metrics": bool(CFG["cell12c4c_keep_or_revert_using_test_metrics"]),
        "cell12f4b_apply_test_derived_observability_repairs": bool(CFG["cell12f4b_apply_test_derived_observability_repairs"]),
        "q4_allow_precomputed_or_restored_promotion": bool(CFG["q4_allow_precomputed_or_restored_promotion"]),
    },
}

C2ST_CONTRACT = {
    "model_family": str(CFG["c2st_model_family"]),
    "solver": str(CFG["c2st_solver"]),
    "max_iter": int(CFG["c2st_max_iter"]),
    "max_rows": int(CFG["c2st_max_rows"]),
    "split_policy": str(CFG["c2st_split_policy"]),
    "block_len_seconds": int(CFG["c2st_block_len_seconds"]),
    "repetitions": int(CFG["c2st_repetitions"]),
    "test_frac": float(CFG["c2st_test_frac"]),
    "scale_features": bool(CFG["c2st_requires_scaled_features"]),
    "auc_orientation": str(CFG["c2st_auc_orientation"]),
    "bootstrap_policy": str(CFG["bootstrap_policy"]),
    "bootstrap_block_len_seconds": int(CFG["bootstrap_block_len_seconds"]),
}

REPRODUCIBILITY_CONTRACT = {
    "final_repro_mode": FINAL_REPRO_MODE,
    "pythonhashseed_expected": EXPECTED_PYTHONHASHSEED,
    "pythonhashseed_actual": PYTHONHASHSEED_VALUE,
    "pythonhashseed_locked_at_interpreter_start": bool(PYTHONHASHSEED_LOCKED_AT_START),
    "pythonhashseed_policy": PYTHONHASHSEED_POLICY,
    "strict_pythonhashseed_required_by_env": bool(_HARD_FAIL_MISSING_PYTHONHASHSEED),
    "stable_sort_artifact_lists_required": True,
    "paper_language": (
        "If final_repro_mode is managed_jupyter_best_effort, report fixed runtime seeds, "
        "sorted artifact-defining lists, and checksums; do not claim bitwise reproducibility."
    ),
}

CFG_JSONABLE = _jsonable_cfg(CFG)
CFG_SHA1 = _sha1_text(_stable_json_dumps(CFG_JSONABLE))

RUN_META: Dict[str, Any] = {
    "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    "platform": platform.platform(),
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "pyarrow": pa.__version__,
    "sklearn": sklearn.__version__,
    "torch": torch.__version__,
    "device": CFG["device"],
    "seed": int(CFG["seed"]),
    "determinism_mode": str(CFG["determinism_mode"]),
    "final_repro_mode": FINAL_REPRO_MODE,
    "reproducibility_contract": REPRODUCIBILITY_CONTRACT,
    "determinism_note": (
        "Best-effort numerical reproducibility is used unless PYTHONHASHSEED=0 was present before interpreter startup. "
        "All final claims must rely on fixed seeds, stable artifact ordering, split contracts, and artifact checksums."
    ),
    "input_parquet": str(CFG["input_parquet"]),
    "input_parquet_sha256": str(INPUT_PARQUET_SHA256),
    "outdir": str(CFG["outdir"]),
    "raw_time_col": str(CFG["raw_time_col"]),
    "canonical_time_col": str(CFG["canonical_time_col"]),
    "detected_time_source": str(CFG["detected_time_source"]),
    "detected_time_source_kind": str(CFG["detected_time_source_kind"]),
    "time_contract": TIME_CONTRACT,
    "split_sizes": {"train": int(CFG["n_train"]), "val": int(CFG["n_val"]), "test": int(CFG["n_test"]), "total": int(CFG["expected_total_rows"])},
    "split_duration_audit_csv": os.path.join(str(CFG["outdir"]), "reports", "source_trace_duration_audit.csv"),
    "parquet_contract": {
        "num_rows": int(PARQUET_META["num_rows"]),
        "num_columns": int(PARQUET_META["num_columns"]),
        "num_row_groups": int(PARQUET_META["num_row_groups"]),
        "created_by": str(PARQUET_META.get("created_by", "")),
        "serialized_size": int(PARQUET_META.get("serialized_size", 0)),
        "raw_time_col_present": bool(str(CFG["raw_time_col"]) in _schema_name_set),
        "canonical_time_col_present": bool(str(CFG["canonical_time_col"]) in _schema_name_set),
        "present_time_candidates": list(_present_time_candidates),
        "missing_time_candidates": list(_missing_time_candidates),
        "pandas_index_columns": list(map(str, _pandas_index_columns)),
    },
    "artifact_accounting_contract": ARTIFACT_ACCOUNTING_CONTRACT,
    "leakage_policy_contract": LEAKAGE_POLICY_CONTRACT,
    "c2st_contract": C2ST_CONTRACT,
    "thread_env": {k: os.environ.get(k, "") for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED"]},
    "torch_determinism": {
        "use_deterministic_algorithms": True,
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
    },
    "torch_threads": {
        "requested_num_threads": int(CFG["torch_num_threads"]),
        "requested_num_interop_threads": int(CFG["torch_num_interop_threads"]),
        "actual_num_threads": int(torch.get_num_threads()),
        "actual_num_interop_threads": int(torch.get_num_interop_threads()),
    },
    "cfg_sha1": CFG_SHA1,
}

# -------------------------
# Persist metadata artifacts
# -------------------------
_report_dir = os.path.join(str(CFG["outdir"]), "reports")
_artifact_dir = os.path.join(str(CFG["outdir"]), "artifacts")
_contract_dir = os.path.join(str(CFG["outdir"]), "artifacts", "contracts")

SPLIT_DURATION_AUDIT.to_csv(os.path.join(_report_dir, "source_trace_duration_audit.csv"), index=False)

for fname, obj in [
    ("run_meta_cell1.json", RUN_META),
    ("effective_cfg_cell1.json", CFG_JSONABLE),
]:
    with open(os.path.join(_artifact_dir, fname), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

with open(os.path.join(_artifact_dir, "input_schema_columns_cell1.json"), "w", encoding="utf-8") as f:
    json.dump({"input_parquet": str(CFG["input_parquet"]), "input_parquet_sha256": str(INPUT_PARQUET_SHA256), "num_columns": int(PARQUET_META["num_columns"]), "columns": _schema_names}, f, indent=2, ensure_ascii=False)

for fname, obj in [
    ("artifact_accounting_contract_cell1.json", ARTIFACT_ACCOUNTING_CONTRACT),
    ("leakage_policy_contract_cell1.json", LEAKAGE_POLICY_CONTRACT),
    ("c2st_contract_cell1.json", C2ST_CONTRACT),
    ("reproducibility_contract_cell1.json", REPRODUCIBILITY_CONTRACT),
]:
    with open(os.path.join(_contract_dir, fname), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

# -------------------------
# Banner
# -------------------------
log("--- START: Cell 1 (config + determinism + canonical contract) ---")
log(f"Project line: {CFG['project_line']}")
log(f"Notebook line: {CFG['notebook_line']}")
log(f"Platform:     {RUN_META['platform']}")
log(f"Python:       {RUN_META['python']}")
log(f"pandas:       {RUN_META['pandas']}")
log(f"numpy:        {RUN_META['numpy']}")
log(f"pyarrow:      {RUN_META['pyarrow']}")
log(f"sklearn:      {RUN_META['sklearn']}")
log(f"torch:        {RUN_META['torch']}")
log(f"Device:       {RUN_META['device']}")
log(f"Seed:         {RUN_META['seed']}")
log(f"Input:        {RUN_META['input_parquet']}")
log(f"Input SHA256: {RUN_META['input_parquet_sha256']}")
log(f"Outdir:       {RUN_META['outdir']}")
log(f"Raw time:     {RUN_META['raw_time_col']}")
log(f"Canonical time target: {RUN_META['canonical_time_col']}")
log(f"Time contract: source={RUN_META['detected_time_source']} | kind={RUN_META['detected_time_source_kind']} | first={TIME_CONTRACT['first_sec']} | last={TIME_CONTRACT['last_sec']} | step={TIME_CONTRACT['median_step']} | duration_days={TIME_CONTRACT['duration_days_inclusive']:.6f}")
log(f"Split sizes: train={RUN_META['split_sizes']['train']} | val={RUN_META['split_sizes']['val']} | test={RUN_META['split_sizes']['test']} | total={RUN_META['split_sizes']['total']}")
log(f"Parquet contract: rows={RUN_META['parquet_contract']['num_rows']} | cols={RUN_META['parquet_contract']['num_columns']} | row_groups={RUN_META['parquet_contract']['num_row_groups']}")
log(f"Artifact accounting: IoT={ARTIFACT_ACCOUNTING_CONTRACT['iot_namespace_cols']} | protocol={ARTIFACT_ACCOUNTING_CONTRACT['protocol_scientific_cols']} | full_logical={ARTIFACT_ACCOUNTING_CONTRACT['full_scientific_logical_cols']} | public_logical={ARTIFACT_ACCOUNTING_CONTRACT['public_logical_cols']}")
log(f"Sparse-driver release contract: total={ARTIFACT_ACCOUNTING_CONTRACT['sparse_driver_corrected_release_contract']['total_driver_cols']} | pass={ARTIFACT_ACCOUNTING_CONTRACT['sparse_driver_corrected_release_contract']['corrected_pass_cols']} | warning={ARTIFACT_ACCOUNTING_CONTRACT['sparse_driver_corrected_release_contract']['corrected_warning_cols']} | fatal_excluded={ARTIFACT_ACCOUNTING_CONTRACT['sparse_driver_corrected_release_contract']['corrected_fatal_cols']} | release_eligible={ARTIFACT_ACCOUNTING_CONTRACT['sparse_driver_corrected_release_contract']['release_eligible_cols']}")
log(f"Leakage policy: {LEAKAGE_POLICY_CONTRACT['selection_split_policy']} | TEST_repair={LEAKAGE_POLICY_CONTRACT['allow_test_values_for_repair']} | TEST_selection={LEAKAGE_POLICY_CONTRACT['allow_test_values_for_candidate_selection']}")
log(f"C2ST policy: split={C2ST_CONTRACT['split_policy']} | block={C2ST_CONTRACT['block_len_seconds']}s | reps={C2ST_CONTRACT['repetitions']} | orientation={C2ST_CONTRACT['auc_orientation']}")
log(f"Determinism mode: {RUN_META['determinism_mode']}")
log(f"Final reproducibility mode: {RUN_META['final_repro_mode']}")
log(f"PYTHONHASHSEED policy: {PYTHONHASHSEED_POLICY}")
log(f"PYTHONHASHSEED locked at startup: {PYTHONHASHSEED_LOCKED_AT_START}")
log(f"PYTHONHASHSEED value: {PYTHONHASHSEED_VALUE!r}")
log(f"CFG SHA1:     {CFG_SHA1}")
log(f"CUBLAS_WORKSPACE_CONFIG: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}")
log(f"OMP_NUM_THREADS:         {os.environ.get('OMP_NUM_THREADS', '')}")
log(f"MKL_NUM_THREADS:         {os.environ.get('MKL_NUM_THREADS', '')}")
log(f"OPENBLAS_NUM_THREADS:    {os.environ.get('OPENBLAS_NUM_THREADS', '')}")
log(f"torch_num_threads(req):  {RUN_META['torch_threads']['requested_num_threads']}")
log(f"torch_num_threads(act):  {RUN_META['torch_threads']['actual_num_threads']}")
log(f"torch_interop(req):      {RUN_META['torch_threads']['requested_num_interop_threads']}")
log(f"torch_interop(act):      {RUN_META['torch_threads']['actual_num_interop_threads']}")
log(f"AMP requested:           {bool(CFG['ddpm_use_amp'])}")
log(f"AMP effective:           {bool(CFG['ddpm_amp_effective'])}")
log(f"Wrote split audit:       {RUN_META['split_duration_audit_csv']}")
log("NOTE: Cell 1 validates only metadata plus the raw epoch-second time column.")
log("NOTE: Cell 2 must materialize sec_epoch_s__canon from the validated raw time source.")
log("NOTE: Later cells must obey leakage_policy_contract_cell1.json and c2st_contract_cell1.json.")
log("NOTE: Later cells must use stable_sorted_str_list() for artifact-defining column lists.")
if FINAL_REPRO_MODE != "strict_pythonhashseed_zero":
    log("REPRO WARNING: managed_jupyter_best_effort mode is active. Report checksums and fixed seeds, not bitwise reproducibility.")
log("[INFO] Cell 1 loaded and verified canonical run config only; no modeling logic is present.")
