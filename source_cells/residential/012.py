# ============================================================
# CELL 6 — Protocol generator role contract
# v6 STUDY-THESIS-GRADE HYBRID PORTFOLIO ROLE CONTRACT
#
# Purpose:
# - Freeze the full protocol value universe from Cell 5.
# - Classify every protocol value column into scientifically appropriate
#   generator-role eligibility groups for the downstream hybrid portfolio.
#
# This cell does NOT generate synthetic data.
# This cell does NOT select final winners.
# This cell defines which generator families are allowed to propose candidates.
#
# Required downstream framing:
# - A1 is the TRAIN-only temporal block-bootstrap baseline for all protocol columns.
# - DDPM is one optional candidate family, not the universal protocol generator.
# - Other generators may propose candidates according to role eligibility.
# - Cell 11 must select candidates using VAL only.
# - TEST is final QA/reporting only.
#
# Critical invariants:
# - PROTO_COLS_CONTRACT is the full protocol value universe.
# - DDPM_VALUE_COLS is a subset of PROTO_COLS_CONTRACT.
# - DDPM_VALUE_COLS must NOT be forced to equal PROTO_COLS_CONTRACT.
# - This cell must not mutate CFG['expected_proto_value_cols_core'].
# - TEST statistics are audit-only and must not decide eligibility.
#
# Artifacts:
# - protocol_generator_role_contract.json
# - protocol_generator_role_contract.csv
# - ddpm_value_cols.json
# - ddpm_target_role_contract.json
# - ddpm_value_cols_manifest.json
# - protocol_generator_role_manifest.json
# ============================================================

log("--- START: Cell 6 — Protocol generator role contract (v6 STUDY-THESIS hybrid portfolio) ---")

import os
import re
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 0) Required globals
# ------------------------------------------------------------
need = [
    "CFG", "log",
    "df_tr", "df_va", "df_te",
    "PROTO_COLS_CONTRACT",
    "PROTOCOL_TIERS_USED",
    "PROTO_OBS_TR", "PROTO_OBS_VA", "PROTO_OBS_TE",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell6] Missing prerequisites: {missing}. Run Cells 1–5 first.")

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
REP_DIR = os.path.join(OUTDIR, "reports")
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

TIERS_SUPERSET = ["router", "ota", "zigbee", "zwave"]

# ------------------------------------------------------------
# 1) Helpers
# ------------------------------------------------------------
def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(v) for v in obj]
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
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(obj), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _sha_cols(cols) -> str:
    return hashlib.sha256(("||".join(map(str, cols))).encode("utf-8")).hexdigest()

def _assert_unique(name: str, cols: list) -> None:
    if len(cols) != len(set(cols)):
        vals, cnts = np.unique(np.asarray(cols, dtype=object), return_counts=True)
        dupes = [str(v) for v, c in zip(vals, cnts) if c > 1]
        raise RuntimeError(f"[Cell6] Duplicate columns in {name}: {dupes[:30]}")

def _assert_ordered_subset(xs, universe, *, label: str) -> None:
    xs = list(xs)
    universe = list(universe)

    if len(xs) != len(set(xs)):
        raise RuntimeError(f"[Cell6] Duplicate entries in {label}: {xs}")

    unknown = [x for x in xs if x not in universe]
    if unknown:
        raise RuntimeError(f"[Cell6] Unknown entries in {label}: {unknown}. Allowed={universe}")

    idx = [universe.index(x) for x in xs]
    if idx != sorted(idx):
        raise RuntimeError(
            f"[Cell6] Order mismatch in {label}. Expected subset order of {universe}, got {xs}"
        )

def _tier_of(col: str):
    c = str(col)
    for t in TIERS_SUPERSET:
        if c.startswith(f"{t}__"):
            return t
    return None

def _tier_has_values(contract_cols, tier: str) -> bool:
    pref = f"{tier}__"
    return any(str(c).startswith(pref) for c in contract_cols)

def _is_numeric_col_all_splits(col: str) -> bool:
    return (
        col in df_tr.columns and pd.api.types.is_numeric_dtype(df_tr[col])
        and col in df_va.columns and pd.api.types.is_numeric_dtype(df_va[col])
        and col in df_te.columns and pd.api.types.is_numeric_dtype(df_te[col])
    )

def _numeric_1d(df_part: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df_part.columns:
        raise RuntimeError(f"[Cell6] Missing column: {col}")
    x = pd.to_numeric(df_part[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    if np.isinf(x).any():
        raise RuntimeError(f"[Cell6] Column {col} contains +/-inf.")
    return x

def _finite_1d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]

def _validate_obs_matrix(arr, *, name: str, expected_n: int, expected_d: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.shape != (int(expected_n), int(expected_d)):
        raise RuntimeError(
            f"[Cell6] {name} shape mismatch: got={arr.shape}, "
            f"expected={(int(expected_n), int(expected_d))}"
        )

    if not np.isfinite(arr).all():
        raise RuntimeError(f"[Cell6] {name} contains non-finite values.")

    vals = set(np.unique(np.round(arr, 6)).tolist())
    if not vals.issubset({0.0, 1.0}):
        raise RuntimeError(f"[Cell6] {name} is not binary. values={sorted(vals)[:20]}")

    return arr.astype(np.float32, copy=False)

def _value_nan_inf_audit(df_part: pd.DataFrame, cols: list, split_name: str) -> dict:
    cols = list(cols)
    if len(cols) == 0:
        return {
            "shape": [int(len(df_part)), 0],
            "nan_count": 0,
            "total_values": 0,
            "nan_rate": 0.0,
            "n_cols_with_nan": 0,
            "n_all_nan_cols": 0,
            "cols_with_nan_first_50": [],
            "all_nan_cols": [],
        }

    arr = df_part.loc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    inf_count = int(np.isinf(arr).sum())
    if inf_count > 0:
        raise RuntimeError(f"[Cell6] {split_name} value matrix contains +/-inf values: {inf_count}")

    nan_mask = np.isnan(arr)
    col_nan_counts = nan_mask.sum(axis=0).astype(np.int64)

    cols_with_nan = [
        {
            "col": str(c),
            "nan_count": int(col_nan_counts[j]),
            "nan_rate": float(col_nan_counts[j] / max(arr.shape[0], 1)),
        }
        for j, c in enumerate(cols)
        if int(col_nan_counts[j]) > 0
    ]

    all_nan_cols = [
        str(cols[j])
        for j, cnt in enumerate(col_nan_counts)
        if int(cnt) == arr.shape[0]
    ]

    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "nan_count": int(nan_mask.sum()),
        "total_values": int(arr.size),
        "nan_rate": float(nan_mask.sum() / max(arr.size, 1)),
        "n_cols_with_nan": int(len(cols_with_nan)),
        "n_all_nan_cols": int(len(all_nan_cols)),
        "cols_with_nan_first_50": cols_with_nan[:50],
        "all_nan_cols": all_nan_cols,
    }

def _safe_float(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x

def _split_active_mask_from_cell5(split_key: str, tier: str) -> np.ndarray:
    """
    Use Cell 5 logical masks, not raw physical mask columns.

    This matters for OTA: Cell 5 defines logical OTA as
    rowwise_or(ota__obs_present, ota24__obs_present, ota5__obs_present).
    """
    if "OBS_MASKS_LOGICAL" in globals():
        obs = globals()["OBS_MASKS_LOGICAL"]
        if split_key in obs and tier in obs[split_key]:
            return np.asarray(obs[split_key][tier], dtype=bool)

    protocol_tiers_used = list(globals()["PROTOCOL_TIERS_USED"])
    if tier not in protocol_tiers_used:
        raise RuntimeError(f"[Cell6] Tier {tier} is not present in PROTOCOL_TIERS_USED={protocol_tiers_used}")

    j = protocol_tiers_used.index(tier)
    if split_key == "train":
        arr = globals()["PROTO_OBS_TR"]
    elif split_key == "val":
        arr = globals()["PROTO_OBS_VA"]
    elif split_key == "test":
        arr = globals()["PROTO_OBS_TE"]
    else:
        raise RuntimeError(f"[Cell6] Unknown split_key={split_key}")

    arr = np.asarray(arr, dtype=np.float32)
    return arr[:, j] > 0.5

def _values_active(split_key: str, df_part: pd.DataFrame, col: str, tier: str) -> np.ndarray:
    x = _numeric_1d(df_part, col)
    active = _split_active_mask_from_cell5(split_key, tier)
    if x.shape[0] != active.shape[0]:
        raise RuntimeError(
            f"[Cell6] Length mismatch for active values: split={split_key} col={col} "
            f"x={x.shape[0]} active={active.shape[0]}"
        )
    return x[active]

def _run_transition_rate_finite(x) -> float:
    x = _finite_1d(x)
    if x.size <= 1:
        return 0.0
    return float(np.mean(np.diff(x) != 0))

def _approx_integer_like_rate(x) -> float:
    x = _finite_1d(x)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x - np.rint(x)) <= 1e-6))

def _column_train_support_stats(col: str, tier: str) -> dict:
    x_tr_active = _values_active("train", df_tr, col, tier)
    x_va_active = _values_active("val", df_va, col, tier)
    x_te_active = _values_active("test", df_te, col, tier)

    tr = _finite_1d(x_tr_active)
    va = _finite_1d(x_va_active)
    te = _finite_1d(x_te_active)

    tr_active_den = int(_split_active_mask_from_cell5("train", tier).sum())
    va_active_den = int(_split_active_mask_from_cell5("val", tier).sum())
    te_active_den = int(_split_active_mask_from_cell5("test", tier).sum())

    if tr.size > 0:
        rounded = np.round(tr, 6)
        unique_n = int(np.unique(rounded).size)
        nonzero_rate = float(np.mean(tr > 0))
        std = float(np.std(tr)) if tr.size > 1 else 0.0
        mean = float(np.mean(tr))
        var = float(np.var(tr)) if tr.size > 1 else 0.0
        q50 = float(np.quantile(tr, 0.50))
        q95 = float(np.quantile(tr, 0.95))
        q99 = float(np.quantile(tr, 0.99))
        min_v = float(np.min(tr))
        max_v = float(np.max(tr))
        integer_like_rate = _approx_integer_like_rate(tr)
        transition_rate = _run_transition_rate_finite(tr)
    else:
        unique_n = 0
        nonzero_rate = None
        std = None
        mean = None
        var = None
        q50 = None
        q95 = None
        q99 = None
        min_v = None
        max_v = None
        integer_like_rate = None
        transition_rate = None

    overdispersion_ratio = None
    if mean is not None and mean > 1e-12 and var is not None:
        overdispersion_ratio = float(var / max(mean, 1e-12))

    return {
        "train_active_finite_n": int(tr.size),
        "val_active_finite_n": int(va.size),
        "test_active_finite_n": int(te.size),

        "train_active_finite_rate": float(tr.size / max(tr_active_den, 1)),
        "val_active_finite_rate": float(va.size / max(va_active_den, 1)),
        "test_active_finite_rate": float(te.size / max(te_active_den, 1)),

        "train_nonzero_rate": _safe_float(nonzero_rate),
        "train_unique_n": int(unique_n),
        "train_std": _safe_float(std),
        "train_mean": _safe_float(mean),
        "train_var": _safe_float(var),
        "train_var_to_mean": _safe_float(overdispersion_ratio),
        "train_q50": _safe_float(q50),
        "train_q95": _safe_float(q95),
        "train_q99": _safe_float(q99),
        "train_min": _safe_float(min_v),
        "train_max": _safe_float(max_v),
        "train_integer_like_rate": _safe_float(integer_like_rate),
        "train_transition_rate": _safe_float(transition_rate),
    }

# ------------------------------------------------------------
# 2) Name-pattern policy
# ------------------------------------------------------------
STRUCTURAL_RX = re.compile(
    r"("
    r"__obs_present$|_obs_present$|"
    r"__present$|_present$|"
    r"__mask$|_mask$|"
    r"__staleness_s$|_staleness_s$|"
    r"__stale_flag$|_stale_flag$|"
    r"__channel$|_channel$|"
    r"_unique$|uniq_|_src16_unique$|_dst16_unique$|"
    r"_ip_src_unique$|_ip_dst_unique$|"
    r"_src_port_unique$|_dst_port_unique$|_port_unique$|"
    r"traffic_present|traffic_obs_present"
    r")",
    re.IGNORECASE,
)

RARE_EVENT_RX = re.compile(
    r"("
    r"deauth|disassoc|assoc_req|assoc_resp|auth|"
    r"dns_rcode3|dns_nx|nxdomain|"
    r"aps_present|zcl_present|nwk_present|app_obs_present|"
    r"aps_obs_present|zcl_obs_present|"
    r"malformed|error|fail|reset_event|attack|anomaly"
    r")",
    re.IGNORECASE,
)

EMPIRICAL_BLOCK_RX = re.compile(
    r"("
    r"bytes_total|pkt_total|packet_total|packets_total|"
    r"_bytes$|_pkts$|_packets$|"
    r"_total$|"
    r"_pps$"
    r")",
    re.IGNORECASE,
)

COUNTLIKE_RX = re.compile(
    r"("
    r"_pkt$|_pkts$|_packets$|_bytes$|"
    r"_syn$|_ack$|_rst$|_fin$|"
    r"_query$|_queries$|_response$|_responses$|"
    r"_req$|_reply$|_cmd$|_data$|_mgmt$|_beacon$|"
    r"_count$|_events$|_rcode\d+_.*$"
    r")",
    re.IGNORECASE,
)

DISCRETE_STATE_RX = re.compile(
    r"("
    r"_cmd$|__cmd$|"
    r"_type$|__type$|"
    r"_state$|__state$|"
    r"_protocol$|__protocol$|"
    r"_rcode\d*|rcode"
    r")",
    re.IGNORECASE,
)

def _looks_structural(c: str) -> bool:
    return bool(STRUCTURAL_RX.search(str(c)))

def _looks_rare_event(c: str) -> bool:
    return bool(RARE_EVENT_RX.search(str(c)))

def _looks_empirical_block(c: str) -> bool:
    return bool(EMPIRICAL_BLOCK_RX.search(str(c)))

def _looks_countlike(c: str) -> bool:
    return bool(COUNTLIKE_RX.search(str(c)))

def _looks_discrete_state(c: str) -> bool:
    return bool(DISCRETE_STATE_RX.search(str(c)))

def _base_role_from_train_only(col: str, stats: dict) -> tuple:
    """
    Return (primary_role, reasons).

    TEST is never used here.
    VAL/TEST support statistics are audit-only.
    """
    c = str(col)
    cl = c.lower()
    reasons = []

    train_n = int(stats.get("train_active_finite_n", 0))
    unique_n = int(stats.get("train_unique_n", 0))
    nz = stats.get("train_nonzero_rate", None)
    std = stats.get("train_std", None)
    q95 = stats.get("train_q95", None)

    min_train_n = int(CFG.get("cell6_ddpm_min_train_active_finite_n", 5000))
    min_unique_n = int(CFG.get("cell6_ddpm_min_unique_n", 4))
    min_nonzero = float(CFG.get("cell6_ddpm_min_nonzero_rate", 0.005))
    max_nonzero_for_rare = float(CFG.get("cell6_rare_event_nonzero_rate_max", 0.01))

    if _looks_structural(c):
        reasons.append("structural_or_observability_or_identifier_like")
        return "structural_rule", reasons

    if c.startswith("iot__"):
        reasons.append("iot_leakage_guard")
        return "structural_rule", reasons

    if c.startswith("events_in_sec__") or c.startswith("telemetry_in_sec__"):
        reasons.append("event_or_telemetry_helper_leakage_guard")
        return "structural_rule", reasons

    if c.startswith(("ota24__", "ota5__")):
        reasons.append("physical_ota_namespace_not_logical_protocol")
        return "structural_rule", reasons

    if train_n <= 0:
        reasons.append("no_finite_train_support")
        return "excluded_or_no_support", reasons

    if _looks_rare_event(c):
        reasons.append("rare_event_name_pattern")
        return "rare_event", reasons

    if nz is not None and float(nz) <= max_nonzero_for_rare:
        reasons.append("rare_event_extreme_sparsity")
        return "rare_event", reasons

    if unique_n <= 1 or (std is not None and float(std) <= 1e-12):
        reasons.append("near_constant_train_support")
        return "structural_rule", reasons

    if _looks_empirical_block(c):
        reasons.append("aggregate_total_or_volume_or_rate_better_for_block_bootstrap")
        return "empirical_block", reasons

    if train_n < min_train_n:
        reasons.append("too_few_finite_train_values_for_neural_target")
        return "empirical_block", reasons

    if unique_n < min_unique_n:
        reasons.append("too_few_unique_train_values_for_neural_target")
        return "empirical_block", reasons

    if nz is None or float(nz) < min_nonzero:
        reasons.append("too_sparse_for_ddpm")
        return "rare_event", reasons

    if q95 is not None and float(q95) <= 0.0:
        reasons.append("nonpositive_q95")
        return "rare_event", reasons

    reasons.append("train_supported_stochastic_protocol_value")
    return "ddpm_value", reasons

def _eligible_generator_families(col: str, primary_role: str, stats: dict) -> tuple:
    """
    Return (families, reasons).

    Families are candidate-generator eligibility labels, not final winners.
    A1 temporal block baseline remains available for every protocol column.
    """
    c = str(col)
    families = ["temporal_block_baseline"]
    reasons = ["A1_available_for_all_protocol_columns"]

    train_n = int(stats.get("train_active_finite_n", 0))
    unique_n = int(stats.get("train_unique_n", 0))
    nz = stats.get("train_nonzero_rate", None)
    std = stats.get("train_std", None)
    integer_like_rate = stats.get("train_integer_like_rate", None)
    var_to_mean = stats.get("train_var_to_mean", None)

    dense_enough = bool(train_n >= int(CFG.get("cell6_ddpm_min_train_active_finite_n", 5000)))
    variable = bool(std is not None and float(std) > 1e-12)
    moderately_supported = bool(train_n >= int(CFG.get("cell6_min_candidate_train_n", 1000)))

    is_countlike = _looks_countlike(c)
    is_discrete_state = _looks_discrete_state(c)
    integer_like = bool(integer_like_rate is not None and float(integer_like_rate) >= 0.98)
    overdispersed = bool(var_to_mean is not None and float(var_to_mean) > float(CFG.get("cell6_negbin_min_var_to_mean", 1.25)))

    if primary_role == "ddpm_value":
        families.append("ddpm")
        reasons.append("eligible_ddpm_train_supported_stochastic_value")

    if primary_role in {"ddpm_value", "empirical_block"} and dense_enough and variable and unique_n >= 4:
        families.append("copula")
        reasons.append("eligible_copula_dense_variable_numeric")

    if primary_role in {"ddpm_value", "empirical_block"} and dense_enough and variable and unique_n >= 8:
        families.append("ctgan")
        families.append("tvae")
        reasons.append("eligible_tabular_deep_generator_candidate")

    if primary_role in {"rare_event", "empirical_block"} or is_discrete_state:
        if moderately_supported:
            families.append("markov_semimarkov")
            reasons.append("eligible_markov_semimarkov_sparse_or_discrete_temporal")

    if primary_role in {"rare_event", "empirical_block", "ddpm_value"} and is_countlike and integer_like:
        if overdispersed or (nz is not None and float(nz) < 0.25):
            families.append("negative_binomial")
            reasons.append("eligible_negative_binomial_count_burst_or_overdispersed")

    if primary_role == "structural_rule":
        families.append("structural_rule")
        reasons.append("eligible_structural_rule_preservation")

    if primary_role == "excluded_or_no_support":
        reasons.append("no_generator_beyond_schema_baseline_due_to_no_train_support")

    # Preserve order and uniqueness.
    seen = set()
    families_unique = []
    for f in families:
        if f not in seen:
            seen.add(f)
            families_unique.append(f)

    return families_unique, reasons

# ------------------------------------------------------------
# 3) Resolve and validate protocol contract
# ------------------------------------------------------------
proto_contract = list(globals().get("PROTO_COLS_CONTRACT", []))
if not proto_contract:
    raise RuntimeError("[Cell6] Missing/empty PROTO_COLS_CONTRACT. Cell 5 must run successfully first.")

_assert_unique("PROTO_COLS_CONTRACT", proto_contract)

proto_contract_sig = _sha_cols(proto_contract)

protocol_tiers_used = list(globals().get("PROTOCOL_TIERS_USED", []))
if not protocol_tiers_used:
    raise RuntimeError("[Cell6] PROTOCOL_TIERS_USED missing. Cell 5 must run successfully first.")

_assert_ordered_subset(protocol_tiers_used, TIERS_SUPERSET, label="PROTOCOL_TIERS_USED")

tiers_with_values = [t for t in TIERS_SUPERSET if _tier_has_values(proto_contract, t)]
if not tiers_with_values:
    raise RuntimeError("[Cell6] No tiers have value columns in PROTO_COLS_CONTRACT.")

missing_runtime_value_tiers = [t for t in protocol_tiers_used if t not in tiers_with_values]
if missing_runtime_value_tiers:
    raise RuntimeError(
        "[Cell6] PROTOCOL_TIERS_USED contains tier(s) with no protocol value columns: "
        f"{missing_runtime_value_tiers}"
    )

bad_tier_cols = [c for c in proto_contract if _tier_of(c) is None]
if bad_tier_cols:
    raise RuntimeError(
        "[Cell6] PROTO_COLS_CONTRACT contains columns outside known protocol tier namespaces: "
        f"{bad_tier_cols[:30]}"
    )

missing_schema = [
    c for c in proto_contract
    if not (c in df_tr.columns and c in df_va.columns and c in df_te.columns)
]
if missing_schema:
    raise RuntimeError(
        "[Cell6] PROTO_COLS_CONTRACT columns missing from one or more splits: "
        f"{missing_schema[:30]}"
    )

non_numeric = [c for c in proto_contract if not _is_numeric_col_all_splits(c)]
if non_numeric:
    raise RuntimeError(f"[Cell6] PROTO_COLS_CONTRACT contains non-numeric columns: {non_numeric[:30]}")

log(f"[Cell6] PROTO_COLS_CONTRACT n={len(proto_contract)} | sha256={proto_contract_sig[:12]}...")
log(f"[Cell6] tiers_with_values(from PROTO_COLS_CONTRACT)={tiers_with_values}")
log(f"[Cell6] PROTOCOL_TIERS_USED={protocol_tiers_used}")

# ------------------------------------------------------------
# 4) Validate Cell 5 logical obs matrices
# ------------------------------------------------------------
PROTO_OBS_TR_ARR = _validate_obs_matrix(
    globals()["PROTO_OBS_TR"],
    name="PROTO_OBS_TR",
    expected_n=len(df_tr),
    expected_d=len(protocol_tiers_used),
)
PROTO_OBS_VA_ARR = _validate_obs_matrix(
    globals()["PROTO_OBS_VA"],
    name="PROTO_OBS_VA",
    expected_n=len(df_va),
    expected_d=len(protocol_tiers_used),
)
PROTO_OBS_TE_ARR = _validate_obs_matrix(
    globals()["PROTO_OBS_TE"],
    name="PROTO_OBS_TE",
    expected_n=len(df_te),
    expected_d=len(protocol_tiers_used),
)

protocol_tier_to_idx = {t: i for i, t in enumerate(protocol_tiers_used)}

# ------------------------------------------------------------
# 5) Decide DDPM tiers
# ------------------------------------------------------------
cfg_tiers = CFG.get("ddpm_tiers_used", None)

if cfg_tiers is None:
    ddpm_tiers = list(protocol_tiers_used)
else:
    ddpm_tiers = [str(x).strip().lower() for x in list(cfg_tiers)]
    _assert_ordered_subset(ddpm_tiers, TIERS_SUPERSET, label="CFG['ddpm_tiers_used']")

allow_zwave = bool(CFG.get("ddpm_allow_zwave", False))
if not allow_zwave:
    ddpm_tiers = [t for t in ddpm_tiers if t != "zwave"]

ddpm_tiers = [t for t in ddpm_tiers if t in protocol_tiers_used]

if not ddpm_tiers and not bool(CFG.get("cell6_allow_empty_ddpm_tiers", False)):
    raise RuntimeError("[Cell6] DDPM tier selection is empty after policy filtering.")

_assert_ordered_subset(ddpm_tiers, TIERS_SUPERSET, label="DDPM_TIERS_USED")

DDPM_TIERS_USED = list(ddpm_tiers)
globals()["DDPM_TIERS_USED"] = list(DDPM_TIERS_USED)

log(f"[Cell6] DDPM_TIERS_USED={DDPM_TIERS_USED} (allow_zwave={allow_zwave})")

# ------------------------------------------------------------
# 6) Classify every protocol column using TRAIN-only eligibility
# ------------------------------------------------------------
role_rows = []
primary_role_to_cols = defaultdict(list)
family_to_cols = defaultdict(list)

for c in proto_contract:
    tier = _tier_of(c)
    if tier is None:
        raise RuntimeError(f"[Cell6] Could not infer tier for protocol column: {c}")

    if tier not in protocol_tiers_used:
        # This can happen only if a non-runtime tier survived into the value contract.
        # Treat as structural/excluded rather than silently training on it.
        stats = {
            "train_active_finite_n": 0,
            "val_active_finite_n": 0,
            "test_active_finite_n": 0,
            "train_active_finite_rate": 0.0,
            "val_active_finite_rate": 0.0,
            "test_active_finite_rate": 0.0,
            "train_nonzero_rate": None,
            "train_unique_n": 0,
            "train_std": None,
            "train_mean": None,
            "train_var": None,
            "train_var_to_mean": None,
            "train_q50": None,
            "train_q95": None,
            "train_q99": None,
            "train_min": None,
            "train_max": None,
            "train_integer_like_rate": None,
            "train_transition_rate": None,
        }
        primary_role = "excluded_or_no_support"
        primary_reasons = [f"tier_not_in_PROTOCOL_TIERS_USED:{tier}"]
    else:
        stats = _column_train_support_stats(c, tier)
        primary_role, primary_reasons = _base_role_from_train_only(c, stats)

    families, family_reasons = _eligible_generator_families(c, primary_role, stats)

    # DDPM eligibility must also obey DDPM_TIERS_USED.
    if "ddpm" in families and tier not in DDPM_TIERS_USED:
        families = [f for f in families if f != "ddpm"]
        family_reasons.append(f"ddpm_removed_tier_not_in_DDPM_TIERS_USED:{tier}")

    row = {
        "col": str(c),
        "tier": str(tier),
        "primary_role": str(primary_role),
        "primary_reasons": list(primary_reasons),
        "eligible_generators": list(families),
        "eligibility_reasons": list(family_reasons),

        "train_active_finite_n": int(stats["train_active_finite_n"]),
        "val_active_finite_n": int(stats["val_active_finite_n"]),
        "test_active_finite_n": int(stats["test_active_finite_n"]),

        "train_active_finite_rate": float(stats["train_active_finite_rate"]),
        "val_active_finite_rate": float(stats["val_active_finite_rate"]),
        "test_active_finite_rate": float(stats["test_active_finite_rate"]),

        "train_nonzero_rate": stats["train_nonzero_rate"],
        "train_unique_n": int(stats["train_unique_n"]),
        "train_std": stats["train_std"],
        "train_mean": stats["train_mean"],
        "train_var": stats["train_var"],
        "train_var_to_mean": stats["train_var_to_mean"],
        "train_q50": stats["train_q50"],
        "train_q95": stats["train_q95"],
        "train_q99": stats["train_q99"],
        "train_min": stats["train_min"],
        "train_max": stats["train_max"],
        "train_integer_like_rate": stats["train_integer_like_rate"],
        "train_transition_rate": stats["train_transition_rate"],
    }

    role_rows.append(row)
    primary_role_to_cols[primary_role].append(c)
    for fam in families:
        family_to_cols[fam].append(c)

# ------------------------------------------------------------
# 7) Manual overrides: DDPM include/exclude only
# ------------------------------------------------------------
DDPM_VALUE_COLS = list(family_to_cols.get("ddpm", []))

manual_include = list(CFG.get("cell6_ddpm_force_include_cols", []))
manual_exclude = list(CFG.get("cell6_ddpm_force_exclude_cols", []))

if manual_include:
    unknown = [c for c in manual_include if c not in proto_contract]
    if unknown:
        raise RuntimeError(f"[Cell6] CFG.cell6_ddpm_force_include_cols contains unknown cols: {unknown[:20]}")

    bad_tier = [c for c in manual_include if _tier_of(c) not in DDPM_TIERS_USED]
    if bad_tier:
        raise RuntimeError(
            "[Cell6] CFG.cell6_ddpm_force_include_cols requested columns outside DDPM_TIERS_USED: "
            f"{bad_tier[:20]}"
        )

    for c in manual_include:
        if c not in DDPM_VALUE_COLS:
            DDPM_VALUE_COLS.append(c)

if manual_exclude:
    unknown = [c for c in manual_exclude if c not in proto_contract]
    if unknown:
        raise RuntimeError(f"[Cell6] CFG.cell6_ddpm_force_exclude_cols contains unknown cols: {unknown[:20]}")

    exclude_set = set(manual_exclude)
    DDPM_VALUE_COLS = [c for c in DDPM_VALUE_COLS if c not in exclude_set]

# Preserve protocol ordering.
ddpm_set = set(DDPM_VALUE_COLS)
DDPM_VALUE_COLS = [c for c in proto_contract if c in ddpm_set]

if len(DDPM_VALUE_COLS) == 0 and not bool(CFG.get("cell6_allow_empty_ddpm_value_cols", False)):
    preview = pd.DataFrame(role_rows)[
        [
            "col", "tier", "primary_role", "eligible_generators",
            "train_active_finite_n", "train_nonzero_rate", "train_unique_n",
        ]
    ].head(30).to_dict("records")

    raise RuntimeError(
        "[Cell6] DDPM target selection produced zero DDPM_VALUE_COLS. "
        "This is allowed only if you intentionally disable DDPM as a protocol candidate. "
        f"Preview={preview}"
    )

_assert_unique("DDPM_VALUE_COLS", DDPM_VALUE_COLS)

DDPM_VALUE_SIG = _sha_cols(DDPM_VALUE_COLS)
PROTO_CONTRACT_SIG = proto_contract_sig

# ------------------------------------------------------------
# 8) Construct role/family column groups
# ------------------------------------------------------------
EMPIRICAL_BLOCK_COLS = [c for c in proto_contract if c in set(primary_role_to_cols.get("empirical_block", []))]
RARE_EVENT_COLS = [c for c in proto_contract if c in set(primary_role_to_cols.get("rare_event", []))]
STRUCTURAL_RULE_COLS = [c for c in proto_contract if c in set(primary_role_to_cols.get("structural_rule", []))]
EXCLUDED_OR_NO_SUPPORT_COLS = [c for c in proto_contract if c in set(primary_role_to_cols.get("excluded_or_no_support", []))]

TEMPORAL_BLOCK_BASELINE_COLS = list(proto_contract)
COPULA_CANDIDATE_COLS = [c for c in proto_contract if c in set(family_to_cols.get("copula", []))]
MARKOV_CANDIDATE_COLS = [c for c in proto_contract if c in set(family_to_cols.get("markov_semimarkov", []))]
NEGATIVE_BINOMIAL_CANDIDATE_COLS = [c for c in proto_contract if c in set(family_to_cols.get("negative_binomial", []))]
CTGAN_CANDIDATE_COLS = [c for c in proto_contract if c in set(family_to_cols.get("ctgan", []))]
TVAE_CANDIDATE_COLS = [c for c in proto_contract if c in set(family_to_cols.get("tvae", []))]
STRUCTURAL_RULE_CANDIDATE_COLS = [c for c in proto_contract if c in set(family_to_cols.get("structural_rule", []))]

# Compatibility names from v5.
DDPM_EMPIRICAL_BLOCK_COLS = list(EMPIRICAL_BLOCK_COLS)
DDPM_RARE_EVENT_COLS = list(RARE_EVENT_COLS)
DDPM_STRUCTURAL_COLS = list(STRUCTURAL_RULE_COLS)
DDPM_NO_SUPPORT_COLS = list(EXCLUDED_OR_NO_SUPPORT_COLS)

primary_partition = (
    list(DDPM_VALUE_COLS)
    + list(EMPIRICAL_BLOCK_COLS)
    + list(RARE_EVENT_COLS)
    + list(STRUCTURAL_RULE_COLS)
    + list(EXCLUDED_OR_NO_SUPPORT_COLS)
)

# Important: DDPM_VALUE_COLS is a candidate family, while primary roles contain
# ddpm_value before manual exclusion. Rebuild primary partition using row roles
# for exact coverage.
primary_role_partition = []
for role_name in ["ddpm_value", "empirical_block", "rare_event", "structural_rule", "excluded_or_no_support"]:
    primary_role_partition.extend(primary_role_to_cols.get(role_name, []))

if set(primary_role_partition) != set(proto_contract):
    missing = sorted(set(proto_contract) - set(primary_role_partition))
    extra = sorted(set(primary_role_partition) - set(proto_contract))
    raise RuntimeError(
        "[Cell6] Primary role partition does not cover PROTO_COLS_CONTRACT exactly. "
        f"missing={missing[:20]} extra={extra[:20]}"
    )

if len(primary_role_partition) != len(set(primary_role_partition)):
    vals, cnts = np.unique(np.asarray(primary_role_partition, dtype=object), return_counts=True)
    dupes = [str(v) for v, c in zip(vals, cnts) if c > 1]
    raise RuntimeError(f"[Cell6] Primary role partition has duplicates: {dupes[:30]}")

if not set(DDPM_VALUE_COLS).issubset(set(proto_contract)):
    raise RuntimeError("[Cell6] DDPM_VALUE_COLS is not a subset of PROTO_COLS_CONTRACT.")

if set(DDPM_VALUE_COLS) == set(proto_contract) and not bool(CFG.get("cell6_allow_ddpm_equal_proto_contract", False)):
    raise RuntimeError(
        "[Cell6] Invalid old invariant revived: DDPM_VALUE_COLS equals PROTO_COLS_CONTRACT. "
        "Under the hybrid portfolio policy, DDPM must be a role-selected candidate subset unless explicitly allowed."
    )

# Hard anti-leakage validation on final DDPM targets.
bad_mask_meta = [c for c in DDPM_VALUE_COLS if _looks_structural(c)]
if bad_mask_meta:
    raise RuntimeError(f"[Cell6] DDPM_VALUE_COLS contains structural/mask/meta cols: {bad_mask_meta[:30]}")

bad_iot = [c for c in DDPM_VALUE_COLS if str(c).startswith("iot__")]
if bad_iot:
    raise RuntimeError(f"[Cell6] IoT columns leaked into DDPM_VALUE_COLS: {bad_iot[:30]}")

bad_events = [c for c in DDPM_VALUE_COLS if str(c).startswith("events_in_sec__")]
if bad_events:
    raise RuntimeError(f"[Cell6] Event-driver columns leaked into DDPM_VALUE_COLS: {bad_events[:30]}")

bad_telemetry_helpers = [c for c in DDPM_VALUE_COLS if str(c).startswith("telemetry_in_sec__")]
if bad_telemetry_helpers:
    raise RuntimeError(f"[Cell6] Telemetry helper columns leaked into DDPM_VALUE_COLS: {bad_telemetry_helpers[:30]}")

bad_physical_only = [c for c in DDPM_VALUE_COLS if str(c).startswith(("ota24__", "ota5__"))]
if bad_physical_only:
    raise RuntimeError(f"[Cell6] Physical OTA namespace leaked into DDPM_VALUE_COLS: {bad_physical_only[:30]}")

expected_ddpm_D = CFG.get("expected_ddpm_value_cols_v6", CFG.get("expected_ddpm_value_cols_v5", None))
if expected_ddpm_D is not None:
    expected_ddpm_D = int(expected_ddpm_D)
    if expected_ddpm_D >= 0 and len(DDPM_VALUE_COLS) != expected_ddpm_D:
        raise RuntimeError(
            f"[Cell6] DDPM target count mismatch: got={len(DDPM_VALUE_COLS)} "
            f"expected={expected_ddpm_D}"
        )

# ------------------------------------------------------------
# 9) DDPM-specific logical observability matrices
# ------------------------------------------------------------
missing_obs_tiers = [t for t in DDPM_TIERS_USED if t not in protocol_tier_to_idx]
if missing_obs_tiers:
    raise RuntimeError(
        f"[Cell6] DDPM tier(s) missing from Cell 5 logical obs matrices: {missing_obs_tiers}"
    )

ddpm_obs_idx = [protocol_tier_to_idx[t] for t in DDPM_TIERS_USED]

DDPM_OBS_TR = PROTO_OBS_TR_ARR[:, ddpm_obs_idx].astype(np.float32, copy=True)
DDPM_OBS_VA = PROTO_OBS_VA_ARR[:, ddpm_obs_idx].astype(np.float32, copy=True)
DDPM_OBS_TE = PROTO_OBS_TE_ARR[:, ddpm_obs_idx].astype(np.float32, copy=True)

for nm, arr, expected_n in [
    ("DDPM_OBS_TR", DDPM_OBS_TR, len(df_tr)),
    ("DDPM_OBS_VA", DDPM_OBS_VA, len(df_va)),
    ("DDPM_OBS_TE", DDPM_OBS_TE, len(df_te)),
]:
    _validate_obs_matrix(arr, name=nm, expected_n=expected_n, expected_d=len(DDPM_TIERS_USED))

ddpm_obs_rates = {
    "train": {t: float(DDPM_OBS_TR[:, j].mean()) for j, t in enumerate(DDPM_TIERS_USED)},
    "val": {t: float(DDPM_OBS_VA[:, j].mean()) for j, t in enumerate(DDPM_TIERS_USED)},
    "test": {t: float(DDPM_OBS_TE[:, j].mean()) for j, t in enumerate(DDPM_TIERS_USED)},
}

# ------------------------------------------------------------
# 10) NaN/inf audits
# ------------------------------------------------------------
ddpm_value_nan_audit = {
    "train": _value_nan_inf_audit(df_tr, DDPM_VALUE_COLS, "TRAIN/DDPM_VALUE_COLS"),
    "val": _value_nan_inf_audit(df_va, DDPM_VALUE_COLS, "VAL/DDPM_VALUE_COLS"),
    "test": _value_nan_inf_audit(df_te, DDPM_VALUE_COLS, "TEST/DDPM_VALUE_COLS"),
}

protocol_contract_nan_audit = {
    "train": _value_nan_inf_audit(df_tr, proto_contract, "TRAIN/PROTO_COLS_CONTRACT"),
    "val": _value_nan_inf_audit(df_va, proto_contract, "VAL/PROTO_COLS_CONTRACT"),
    "test": _value_nan_inf_audit(df_te, proto_contract, "TEST/PROTO_COLS_CONTRACT"),
}

if ddpm_value_nan_audit["train"]["n_all_nan_cols"] > 0:
    raise RuntimeError(
        "[Cell6] TRAIN has DDPM_VALUE_COLS that are entirely NaN. "
        f"These cannot be fitted: {ddpm_value_nan_audit['train']['all_nan_cols'][:30]}"
    )

# ------------------------------------------------------------
# 11) Build contracts
# ------------------------------------------------------------
role_counts = dict(Counter([r["primary_role"] for r in role_rows]))
family_counts = {
    "temporal_block_baseline": int(len(TEMPORAL_BLOCK_BASELINE_COLS)),
    "ddpm": int(len(DDPM_VALUE_COLS)),
    "copula": int(len(COPULA_CANDIDATE_COLS)),
    "markov_semimarkov": int(len(MARKOV_CANDIDATE_COLS)),
    "negative_binomial": int(len(NEGATIVE_BINOMIAL_CANDIDATE_COLS)),
    "ctgan": int(len(CTGAN_CANDIDATE_COLS)),
    "tvae": int(len(TVAE_CANDIDATE_COLS)),
    "structural_rule": int(len(STRUCTURAL_RULE_CANDIDATE_COLS)),
}

primary_role_counts_ordered = {
    "ddpm_value": int(role_counts.get("ddpm_value", 0)),
    "empirical_block": int(role_counts.get("empirical_block", 0)),
    "rare_event": int(role_counts.get("rare_event", 0)),
    "structural_rule": int(role_counts.get("structural_rule", 0)),
    "excluded_or_no_support": int(role_counts.get("excluded_or_no_support", 0)),
}

tier_value_counts_ddpm = {
    t: int(sum(str(c).startswith(f"{t}__") for c in DDPM_VALUE_COLS))
    for t in DDPM_TIERS_USED
}

protocol_generator_role_contract = {
    "version": "protocol_generator_role_contract_v6_study_thesis_hybrid_portfolio",
    "policy": {
        "architecture": "validation_selected_hybrid_portfolio",
        "baseline": "TRAIN_only_temporal_block_bootstrap_A1",
        "A2": "VAL_selected_hybrid_protocol_output",
        "ddpm_role": "candidate_generator_only",
        "candidate_selection_split": "VAL_only",
        "test_usage": "final_QA_only",
        "test_used_for_eligibility": False,
        "cell6_decides": "generator_family_eligibility_only",
        "cell6_does_not_decide": "final_candidate_selection",
        "ddpm_equal_proto_contract_allowed": bool(CFG.get("cell6_allow_ddpm_equal_proto_contract", False)),
    },
    "proto_cols_contract": list(proto_contract),
    "proto_cols_contract_n": int(len(proto_contract)),
    "proto_cols_contract_sha256": str(proto_contract_sig),

    "protocol_tiers_used": list(protocol_tiers_used),
    "tiers_with_values": list(tiers_with_values),
    "ddpm_tiers_used": list(DDPM_TIERS_USED),

    "primary_role_counts": primary_role_counts_ordered,
    "candidate_family_counts": family_counts,

    "primary_roles": {
        "ddpm_value": list(primary_role_to_cols.get("ddpm_value", [])),
        "empirical_block": list(EMPIRICAL_BLOCK_COLS),
        "rare_event": list(RARE_EVENT_COLS),
        "structural_rule": list(STRUCTURAL_RULE_COLS),
        "excluded_or_no_support": list(EXCLUDED_OR_NO_SUPPORT_COLS),
    },

    "candidate_families": {
        "temporal_block_baseline_cols": list(TEMPORAL_BLOCK_BASELINE_COLS),
        "ddpm_candidate_cols": list(DDPM_VALUE_COLS),
        "empirical_block_cols": list(EMPIRICAL_BLOCK_COLS),
        "rare_event_cols": list(RARE_EVENT_COLS),
        "structural_rule_cols": list(STRUCTURAL_RULE_COLS),
        "copula_candidate_cols": list(COPULA_CANDIDATE_COLS),
        "markov_candidate_cols": list(MARKOV_CANDIDATE_COLS),
        "negative_binomial_candidate_cols": list(NEGATIVE_BINOMIAL_CANDIDATE_COLS),
        "ctgan_candidate_cols": list(CTGAN_CANDIDATE_COLS),
        "tvae_candidate_cols": list(TVAE_CANDIDATE_COLS),
        "excluded_or_no_support_cols": list(EXCLUDED_OR_NO_SUPPORT_COLS),
    },

    "thresholds": {
        "cell6_ddpm_min_train_active_finite_n": int(CFG.get("cell6_ddpm_min_train_active_finite_n", 5000)),
        "cell6_ddpm_min_unique_n": int(CFG.get("cell6_ddpm_min_unique_n", 4)),
        "cell6_ddpm_min_nonzero_rate": float(CFG.get("cell6_ddpm_min_nonzero_rate", 0.005)),
        "cell6_rare_event_nonzero_rate_max": float(CFG.get("cell6_rare_event_nonzero_rate_max", 0.01)),
        "cell6_min_candidate_train_n": int(CFG.get("cell6_min_candidate_train_n", 1000)),
        "cell6_negbin_min_var_to_mean": float(CFG.get("cell6_negbin_min_var_to_mean", 1.25)),
    },

    "manual_overrides": {
        "ddpm_force_include_cols": list(manual_include),
        "ddpm_force_exclude_cols": list(manual_exclude),
    },

    "required_downstream_invariants": [
        "set(DDPM_VALUE_COLS).issubset(set(PROTO_COLS_CONTRACT))",
        "DDPM_VALUE_COLS == PROTOCOL_GENERATOR_ROLE_CONTRACT['candidate_families']['ddpm_candidate_cols']",
        "Cell8 trains DDPM only on DDPM_VALUE_COLS",
        "Cell11 builds A1 for all PROTO_COLS_CONTRACT",
        "Cell11 selects final A2 candidates using VAL only",
        "TEST is final QA only",
    ],

    "rows": role_rows,
}

target_role_contract = {
    "version": "ddpm_target_role_contract_v6_compatibility",
    "policy": (
        "Compatibility artifact for DDPM-specific downstream cells. "
        "The authoritative full generator-role contract is "
        "protocol_generator_role_contract.json. DDPM_VALUE_COLS is a subset "
        "of PROTO_COLS_CONTRACT and is only one candidate family."
    ),
    "role_counts": primary_role_counts_ordered,
    "roles": {
        "ddpm_value": list(DDPM_VALUE_COLS),
        "empirical_block": list(EMPIRICAL_BLOCK_COLS),
        "rare_event": list(RARE_EVENT_COLS),
        "structural": list(STRUCTURAL_RULE_COLS),
        "excluded_no_support": list(EXCLUDED_OR_NO_SUPPORT_COLS),
    },
    "rows": role_rows,
}

ddpm_obs_contract = {
    "version": "ddpm_obs_contract_v6",
    "source": "Cell5 PROTO_OBS_* logical matrices",
    "protocol_tiers_used": list(protocol_tiers_used),
    "ddpm_tiers_used": list(DDPM_TIERS_USED),
    "ddpm_obs_idx_from_protocol_obs": [int(i) for i in ddpm_obs_idx],
    "ddpm_obs_shapes": {
        "train": list(DDPM_OBS_TR.shape),
        "val": list(DDPM_OBS_VA.shape),
        "test": list(DDPM_OBS_TE.shape),
    },
    "ddpm_obs_rates": ddpm_obs_rates,
}

ddpm_value_manifest = {
    "version": "ddpm_value_cols_manifest_v6_study_thesis_hybrid_portfolio",
    "source_contract": "Cell5: PROTO_COLS_CONTRACT",
    "source_contract_n": int(len(proto_contract)),
    "source_contract_sha256": str(PROTO_CONTRACT_SIG),

    "authoritative_role_contract": "protocol_generator_role_contract.json",
    "compatibility_role_contract": "ddpm_target_role_contract.json",

    "DDPM_TIERS_USED": list(DDPM_TIERS_USED),
    "n_ddpm_value_cols": int(len(DDPM_VALUE_COLS)),
    "ddpm_value_sha256": str(DDPM_VALUE_SIG),
    "tier_value_counts": tier_value_counts_ddpm,

    "canonical_equal_proto_contract": False,
    "ddpm_value_cols_are_subset_of_proto_contract": True,
    "why_not_equal_proto_contract": (
        "The full protocol contract includes empirical, rare-event, structural, "
        "and no-support columns. DDPM is only a conditional candidate generator."
    ),

    "downstream_contract": {
        "Cell8_should_train_on": "DDPM_VALUE_COLS only",
        "Cell8_should_not_train_on": "full PROTO_COLS_CONTRACT",
        "Cell11_should_build_A1_for": "full PROTO_COLS_CONTRACT",
        "Cell11_should_select_A2_using": "VAL-only candidate portfolio",
        "Cell11_test_usage": "final QA/reporting only",
    },
}

protocol_generator_role_manifest = {
    "version": "protocol_generator_role_manifest_v6",
    "proto_cols_contract_n": int(len(proto_contract)),
    "proto_cols_contract_sha256": str(PROTO_CONTRACT_SIG),
    "primary_role_counts": primary_role_counts_ordered,
    "candidate_family_counts": family_counts,
    "ddpm_value_cols_n": int(len(DDPM_VALUE_COLS)),
    "ddpm_value_sha256": str(DDPM_VALUE_SIG),
    "policy": "hybrid_validation_selected_portfolio",
}

# ------------------------------------------------------------
# 12) Persist artifacts
# ------------------------------------------------------------
protocol_role_json_path = os.path.join(ARTDIR, "protocol_generator_role_contract.json")
protocol_role_csv_path = os.path.join(REP_DIR, "protocol_generator_role_contract.csv")
protocol_role_manifest_path = os.path.join(ARTDIR, "protocol_generator_role_manifest.json")

ddpm_tiers_path = os.path.join(ARTDIR, "ddpm_tiers_used.json")
ddpm_value_cols_path = os.path.join(ARTDIR, "ddpm_value_cols.json")
ddpm_roles_json_path = os.path.join(ARTDIR, "ddpm_target_role_contract.json")
ddpm_roles_csv_path = os.path.join(REP_DIR, "ddpm_target_role_contract.csv")

ddpm_nan_audit_path = os.path.join(ARTDIR, "ddpm_value_nan_audit.json")
protocol_nan_audit_path = os.path.join(ARTDIR, "protocol_contract_nan_audit.json")
ddpm_obs_contract_path = os.path.join(ARTDIR, "ddpm_obs_contract.json")
ddpm_value_manifest_path = os.path.join(ARTDIR, "ddpm_value_cols_manifest.json")

_write_json(protocol_role_json_path, protocol_generator_role_contract)
_write_json(protocol_role_manifest_path, protocol_generator_role_manifest)

roles_df = pd.DataFrame(role_rows)
roles_df_csv = roles_df.copy()
roles_df_csv["primary_reasons"] = roles_df_csv["primary_reasons"].apply(lambda x: "|".join(map(str, x)))
roles_df_csv["eligible_generators"] = roles_df_csv["eligible_generators"].apply(lambda x: "|".join(map(str, x)))
roles_df_csv["eligibility_reasons"] = roles_df_csv["eligibility_reasons"].apply(lambda x: "|".join(map(str, x)))
roles_df_csv.to_csv(protocol_role_csv_path, index=False)

_write_json(ddpm_tiers_path, DDPM_TIERS_USED)
_write_json(ddpm_value_cols_path, DDPM_VALUE_COLS)
_write_json(ddpm_roles_json_path, target_role_contract)
roles_df_csv.to_csv(ddpm_roles_csv_path, index=False)

_write_json(ddpm_nan_audit_path, ddpm_value_nan_audit)
_write_json(protocol_nan_audit_path, protocol_contract_nan_audit)
_write_json(ddpm_obs_contract_path, ddpm_obs_contract)
_write_json(ddpm_value_manifest_path, ddpm_value_manifest)

# Backward-compatible individual role artifacts.
_write_json(os.path.join(ARTDIR, "ddpm_empirical_block_cols.json"), EMPIRICAL_BLOCK_COLS)
_write_json(os.path.join(ARTDIR, "ddpm_rare_event_cols.json"), RARE_EVENT_COLS)
_write_json(os.path.join(ARTDIR, "ddpm_structural_cols.json"), STRUCTURAL_RULE_COLS)
_write_json(os.path.join(ARTDIR, "ddpm_no_support_cols.json"), EXCLUDED_OR_NO_SUPPORT_COLS)

_write_json(os.path.join(ARTDIR, "copula_candidate_cols.json"), COPULA_CANDIDATE_COLS)
_write_json(os.path.join(ARTDIR, "markov_candidate_cols.json"), MARKOV_CANDIDATE_COLS)
_write_json(os.path.join(ARTDIR, "negative_binomial_candidate_cols.json"), NEGATIVE_BINOMIAL_CANDIDATE_COLS)
_write_json(os.path.join(ARTDIR, "ctgan_candidate_cols.json"), CTGAN_CANDIDATE_COLS)
_write_json(os.path.join(ARTDIR, "tvae_candidate_cols.json"), TVAE_CANDIDATE_COLS)

# ------------------------------------------------------------
# 13) Export globals
# ------------------------------------------------------------
globals()["PROTOCOL_GENERATOR_ROLE_CONTRACT"] = protocol_generator_role_contract
globals()["PROTOCOL_GENERATOR_ROLE_ROWS"] = role_rows

globals()["DDPM_VALUE_COLS"] = list(DDPM_VALUE_COLS)
globals()["DDPM_VALUE_SIG"] = str(DDPM_VALUE_SIG)
globals()["DDPM_TIERS_USED"] = list(DDPM_TIERS_USED)

globals()["DDPM_OBS_TR"] = DDPM_OBS_TR
globals()["DDPM_OBS_VA"] = DDPM_OBS_VA
globals()["DDPM_OBS_TE"] = DDPM_OBS_TE

globals()["ddpm_obs_tr"] = DDPM_OBS_TR
globals()["ddpm_obs_va"] = DDPM_OBS_VA
globals()["ddpm_obs_te"] = DDPM_OBS_TE

globals()["DDPM_TARGET_ROLE_CONTRACT"] = target_role_contract
globals()["DDPM_VALUE_MANIFEST"] = ddpm_value_manifest
globals()["DDPM_VALUE_NAN_AUDIT"] = ddpm_value_nan_audit
globals()["PROTOCOL_CONTRACT_NAN_AUDIT"] = protocol_contract_nan_audit
globals()["DDPM_OBS_CONTRACT"] = ddpm_obs_contract

globals()["PROTO_TEMPORAL_BLOCK_BASELINE_COLS"] = list(TEMPORAL_BLOCK_BASELINE_COLS)
globals()["PROTO_EMPIRICAL_BLOCK_COLS"] = list(EMPIRICAL_BLOCK_COLS)
globals()["PROTO_RARE_EVENT_COLS"] = list(RARE_EVENT_COLS)
globals()["PROTO_STRUCTURAL_RULE_COLS"] = list(STRUCTURAL_RULE_COLS)
globals()["PROTO_EXCLUDED_OR_NO_SUPPORT_COLS"] = list(EXCLUDED_OR_NO_SUPPORT_COLS)

globals()["PROTO_COPULA_CANDIDATE_COLS"] = list(COPULA_CANDIDATE_COLS)
globals()["PROTO_MARKOV_CANDIDATE_COLS"] = list(MARKOV_CANDIDATE_COLS)
globals()["PROTO_NEGATIVE_BINOMIAL_CANDIDATE_COLS"] = list(NEGATIVE_BINOMIAL_CANDIDATE_COLS)
globals()["PROTO_CTGAN_CANDIDATE_COLS"] = list(CTGAN_CANDIDATE_COLS)
globals()["PROTO_TVAE_CANDIDATE_COLS"] = list(TVAE_CANDIDATE_COLS)

# Backward-compatible aliases from v5.
globals()["PROTO_DDPM_VALUE_COLS"] = list(DDPM_VALUE_COLS)
globals()["DDPM_EMPIRICAL_BLOCK_COLS"] = list(EMPIRICAL_BLOCK_COLS)
globals()["DDPM_RARE_EVENT_COLS"] = list(RARE_EVENT_COLS)
globals()["DDPM_STRUCTURAL_COLS"] = list(STRUCTURAL_RULE_COLS)
globals()["DDPM_NO_SUPPORT_COLS"] = list(EXCLUDED_OR_NO_SUPPORT_COLS)

# ------------------------------------------------------------
# 14) Final consistency checks
# ------------------------------------------------------------
if set(DDPM_VALUE_COLS) != set(
    protocol_generator_role_contract["candidate_families"]["ddpm_candidate_cols"]
):
    raise RuntimeError("[Cell6] DDPM_VALUE_COLS does not match protocol generator role contract.")

if not set(DDPM_VALUE_COLS).issubset(set(proto_contract)):
    raise RuntimeError("[Cell6] DDPM_VALUE_COLS is not a subset of PROTO_COLS_CONTRACT.")

if len(TEMPORAL_BLOCK_BASELINE_COLS) != len(proto_contract):
    raise RuntimeError("[Cell6] Temporal block baseline must cover all protocol columns.")

for fam_name, cols in [
    ("copula", COPULA_CANDIDATE_COLS),
    ("markov", MARKOV_CANDIDATE_COLS),
    ("negative_binomial", NEGATIVE_BINOMIAL_CANDIDATE_COLS),
    ("ctgan", CTGAN_CANDIDATE_COLS),
    ("tvae", TVAE_CANDIDATE_COLS),
]:
    bad = [c for c in cols if c not in proto_contract]
    if bad:
        raise RuntimeError(f"[Cell6] {fam_name} candidate cols outside protocol contract: {bad[:20]}")

# Do not mutate CFG['expected_proto_value_cols_core'] here.
if int(CFG.get("expected_proto_value_cols_core", len(proto_contract))) != len(proto_contract):
    log(
        "[WARN] CFG['expected_proto_value_cols_core'] does not match PROTO_COLS_CONTRACT length. "
        f"cfg={CFG.get('expected_proto_value_cols_core')} proto={len(proto_contract)}. "
        "Cell6 will not mutate it; fix CFG upstream if this is unintended."
    )

log(
    "[Cell6] Protocol generator roles | "
    f"proto_contract={len(proto_contract)} | "
    f"ddpm_candidate={len(DDPM_VALUE_COLS)} | "
    f"empirical_block={len(EMPIRICAL_BLOCK_COLS)} | "
    f"rare_event={len(RARE_EVENT_COLS)} | "
    f"structural_rule={len(STRUCTURAL_RULE_COLS)} | "
    f"excluded_or_no_support={len(EXCLUDED_OR_NO_SUPPORT_COLS)}"
)

log(
    "[Cell6] Candidate family eligibility | "
    f"A1_block={len(TEMPORAL_BLOCK_BASELINE_COLS)} | "
    f"ddpm={len(DDPM_VALUE_COLS)} | "
    f"copula={len(COPULA_CANDIDATE_COLS)} | "
    f"markov={len(MARKOV_CANDIDATE_COLS)} | "
    f"negbin={len(NEGATIVE_BINOMIAL_CANDIDATE_COLS)} | "
    f"ctgan={len(CTGAN_CANDIDATE_COLS)} | "
    f"tvae={len(TVAE_CANDIDATE_COLS)}"
)

log(f"[Cell6] DDPM_VALUE_COLS={len(DDPM_VALUE_COLS)} | sha256={DDPM_VALUE_SIG[:12]}...")
log(f"[Cell6] DDPM tier value counts={tier_value_counts_ddpm}")
log(f"[Cell6] DDPM_OBS shapes: tr={DDPM_OBS_TR.shape} va={DDPM_OBS_VA.shape} te={DDPM_OBS_TE.shape}")
log(
    "[Cell6] DDPM target NaN rates | "
    f"train={ddpm_value_nan_audit['train']['nan_rate']:.6f} | "
    f"val={ddpm_value_nan_audit['val']['nan_rate']:.6f} | "
    f"test={ddpm_value_nan_audit['test']['nan_rate']:.6f}"
)

log(f"[Cell6] Saved protocol generator role contract: {protocol_role_json_path}")
log(f"[Cell6] Saved protocol generator role CSV: {protocol_role_csv_path}")
log(f"[Cell6] Saved DDPM value cols: {ddpm_value_cols_path}")
log(f"[Cell6] Saved DDPM target compatibility contract: {ddpm_roles_json_path}")
log(f"[Cell6] Saved DDPM value manifest: {ddpm_value_manifest_path}")

log(
    "[Cell6] STUDY-THESIS generator-role contract summary | "
    f"proto_contract={len(proto_contract)} | "
    f"ddpm_value={len(DDPM_VALUE_COLS)} | "
    f"empirical_block={len(EMPIRICAL_BLOCK_COLS)} | "
    f"rare_event={len(RARE_EVENT_COLS)} | "
    f"structural_rule={len(STRUCTURAL_RULE_COLS)} | "
    f"excluded_or_no_support={len(EXCLUDED_OR_NO_SUPPORT_COLS)} | "
    f"ddpm_sig={DDPM_VALUE_SIG[:12]}"
)

log("--- END:   Cell 6 — Protocol generator role contract (v6 STUDY-THESIS hybrid portfolio) ---")