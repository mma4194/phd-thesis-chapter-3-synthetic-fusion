# ==========================================================
# CELL 14.5 - Coupling manifest consolidation and repair eligibility policy
# v1.1 STUDY-THESIS strict repair-manifest policy layer, repair-status-aware
#
# Role:
#   - Convert Cell 14.4 TRAIN/VAL coupling evidence into controlled
#     repair-eligibility manifests.
#   - Do NOT mutate data.
#   - Do NOT fit models.
#   - Do NOT select generators.
#   - Do NOT use TEST values.
#
# Inputs:
#   CELL14_4_DEEP_COUPLING_REPLICATION_DF
#   CELL14_4_DEEP_COUPLING_TIERA_DF
#   CELL14_4_DEEP_COUPLING_TIERB_DF
#   CELL14_4_DEEP_COUPLING_TIERC_DF
#   CELL14_4_DEEP_COUPLING_CONTRACT
#
# Outputs:
#   reports/cell14_5_repair_manifest_A0_zigbee_safe.csv
#   reports/cell14_5_repair_manifest_A1_zigbee_extended.csv
#   reports/cell14_5_repair_manifest_B_router_review.csv
#   reports/cell14_5_repair_manifest_diagnostic_only.csv
#   reports/cell14_5_repair_manifest_all_policy.csv
#   reports/cell14_5_repair_manifest_policy_summary.csv
#   reports/cell14_5_repair_manifest_contract.json
#   artifacts/cell14_5_repair_manifest_policy_manifest.json
# ==========================================================

log("--- START: Cell 14.5 - Coupling manifest consolidation and repair eligibility policy (v1.1 repair-status-aware strict) ---")

import os
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_145 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL14_4_DEEP_COUPLING_REPLICATION_DF",
    "CELL14_4_DEEP_COUPLING_CONTRACT",
    "CELL14_3_Q4_REPAIR_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
]
_missing_145 = [k for k in _required_145 if k not in globals()]
if _missing_145:
    raise RuntimeError(f"[Cell14.5] Missing required globals from Cell 14.4: {_missing_145}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

CELL145_VERSION = "cell14_5_coupling_repair_manifest_policy_v1_1_repair_status_aware"

CFG["cell14_5_version"] = CELL145_VERSION
CFG["cell14_5_trainval_evidence_used"] = True
CFG["cell14_5_TEST_real_values_used"] = False
CFG["cell14_5_synthetic_values_mutated"] = False
CFG["cell14_5_selection_done_here"] = False
CFG["cell14_5_generator_fit_done_here"] = False
CFG["cell14_5_materialization_done_here"] = False
CFG["cell14_5_policy_layer_only"] = True
CFG["cell14_5_repair_manifest_policy_assignment_done_here"] = True
CFG["cell14_5_repair_eligibility_gating_done_here"] = True
CFG["cell14_5_broad_q4_status_carried_forward"] = True
CFG["cell14_5_repair_status_carried_forward"] = True

# -----------------------------
# Policy knobs
# -----------------------------
CFG.setdefault("cell14_5_max_A0_zigbee_safe_pairs", 40)
CFG.setdefault("cell14_5_max_A1_zigbee_extended_pairs", 80)
CFG.setdefault("cell14_5_max_B_router_review_pairs", 60)

# A0: immediately safe targets for current 14.1-style repair.
CFG.setdefault(
    "cell14_5_A0_zigbee_safe_protocol_cols",
    [
        "zigbee__pkt_total",
        "zigbee__bytes_total",
    ],
)

# A1: extended Zigbee candidates. These need additional domain constraints.
CFG.setdefault(
    "cell14_5_A1_zigbee_extended_protocol_col_allowlist",
    [
        "zigbee__pkt_total",
        "zigbee__bytes_total",
        "zigbee__cmd",
        "zigbee__ack",
        "zigbee__data",
        "zigbee__cmd_rate",
        "zigbee__pps",
        "zigbee__src16_unique",
        "zigbee__dst16_unique",
        "zigbee__panid_unique",
    ],
)

# Router review: primitive/raw-ish router metrics only.
CFG.setdefault(
    "cell14_5_B_router_review_protocol_col_allowlist",
    [
        "router__pkt_total",
        "router__bytes_total",
        "router__udp_pkt",
        "router__tcp_pkt",
        "router__dns_pkt",
        "router__dns_query",
        "router__dns_response",
        "router__dns_rcode0_ok",
        "router__arp_pkt",
        "router__arp_req",
        "router__arp_reply",
    ],
)

# Block direct repair for derived protocol columns unless explicitly reviewed.
CFG.setdefault(
    "cell14_5_derived_protocol_block_tokens",
    [
        "__roll",
        "__absdiff",
        "_rate",
        "__rate",
        "entropy",
        "share",
        "ratio",
        "unique",
        "std",
        "mean",
        "median",
        "q95",
        "q99",
    ],
)

CFG.setdefault("cell14_5_A0_min_replication_score", 0.0)
CFG.setdefault("cell14_5_A1_min_replication_score", 0.0)
CFG.setdefault("cell14_5_router_min_replication_score", 0.0)

# A0 should use event-driver anchors first; binary state anchors can be evidence but are riskier.
CFG.setdefault("cell14_5_A0_require_event_driver_anchor", True)
CFG.setdefault("cell14_5_A1_require_event_driver_anchor", True)
CFG.setdefault("cell14_5_router_require_event_driver_anchor", True)

# Conservative lag windows for repair.
CFG.setdefault("cell14_5_A0_allowed_lag_lo", 0)
CFG.setdefault("cell14_5_A0_allowed_lag_hi", 5)
CFG.setdefault("cell14_5_A1_allowed_lag_lo", 0)
CFG.setdefault("cell14_5_A1_allowed_lag_hi", 10)
CFG.setdefault("cell14_5_router_allowed_lag_lo", 0)
CFG.setdefault("cell14_5_router_allowed_lag_hi", 10)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
A0_zigbee_safe_csv = os.path.join(REPORT_DIR, "cell14_5_repair_manifest_A0_zigbee_safe.csv")
A1_zigbee_extended_csv = os.path.join(REPORT_DIR, "cell14_5_repair_manifest_A1_zigbee_extended.csv")
B_router_review_csv = os.path.join(REPORT_DIR, "cell14_5_repair_manifest_B_router_review.csv")
diagnostic_only_csv = os.path.join(REPORT_DIR, "cell14_5_repair_manifest_diagnostic_only.csv")
all_policy_csv = os.path.join(REPORT_DIR, "cell14_5_repair_manifest_all_policy.csv")
policy_summary_csv = os.path.join(REPORT_DIR, "cell14_5_repair_manifest_policy_summary.csv")
contract_json = os.path.join(REPORT_DIR, "cell14_5_repair_manifest_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_5_repair_manifest_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell14_5_repair_manifest_policy_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_145(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_145(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_145(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_145(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_145(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_145(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_145(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_145(obj.to_dict())
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

def _write_json_145(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_145(payload), f, indent=2, sort_keys=True)

def _sha256_file_145(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_145(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _protocol_tier_145(col: str) -> str:
    s = str(col).lower()
    if s.startswith("zigbee__") or s.startswith("zb__") or "zigbee" in s:
        return "zigbee"
    if s.startswith("router__") or s.startswith("dns__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__") or s.startswith("wifi__"):
        return "ota"
    if s.startswith("zwave__") or "zwave" in s or "z_wave" in s:
        return "zwave"
    return "other"

def _anchor_type_145(col: str) -> str:
    s = str(col)
    if s.startswith("events_in_sec__entity__"):
        return "event_driver_entity"
    if s.startswith("events_in_sec__feat__"):
        return "event_driver_feature"
    if s.startswith("iot__") and "__binary_sensor__" in s:
        return "iot_binary_sensor_state"
    if s.startswith("iot__") and ("__switch__" in s or "__light__" in s or "__lock__" in s):
        return "iot_binary_state"
    return "other"

def _is_event_driver_anchor_145(col: str) -> bool:
    s = str(col)
    return bool(s.startswith("events_in_sec__entity__") or s.startswith("events_in_sec__feat__"))

def _is_derived_protocol_col_145(col: str) -> bool:
    s = str(col).lower()
    tokens = [str(t).lower() for t in CFG.get("cell14_5_derived_protocol_block_tokens", [])]
    return any(tok in s for tok in tokens)

def _numeric_col_145(df: pd.DataFrame, col: str, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def _lag_ok_145(row, lo_key: str, hi_key: str) -> bool:
    lag_tr = _safe_float_145(row.get("peak_lag_sec_train"), np.nan)
    lag_va = _safe_float_145(row.get("peak_lag_sec_val"), np.nan)
    lo = int(CFG.get(lo_key, 0))
    hi = int(CFG.get(hi_key, 10))
    return bool(
        np.isfinite(lag_tr)
        and np.isfinite(lag_va)
        and lo <= lag_tr <= hi
        and lo <= lag_va <= hi
    )

def _recommended_lag_window_145(row, mode: str):
    lag_tr = _safe_float_145(row.get("peak_lag_sec_train"), np.nan)
    lag_va = _safe_float_145(row.get("peak_lag_sec_val"), np.nan)

    if np.isfinite(lag_tr) and np.isfinite(lag_va):
        center = int(round(float(np.mean([lag_tr, lag_va]))))
    elif np.isfinite(lag_tr):
        center = int(round(lag_tr))
    elif np.isfinite(lag_va):
        center = int(round(lag_va))
    else:
        center = 0

    if mode == "A0":
        lo_allowed = int(CFG.get("cell14_5_A0_allowed_lag_lo", 0))
        hi_allowed = int(CFG.get("cell14_5_A0_allowed_lag_hi", 5))
        half = 1
    elif mode == "A1":
        lo_allowed = int(CFG.get("cell14_5_A1_allowed_lag_lo", 0))
        hi_allowed = int(CFG.get("cell14_5_A1_allowed_lag_hi", 10))
        half = 2
    elif mode == "router":
        lo_allowed = int(CFG.get("cell14_5_router_allowed_lag_lo", 0))
        hi_allowed = int(CFG.get("cell14_5_router_allowed_lag_hi", 10))
        half = 2
    else:
        lo_allowed, hi_allowed, half = 0, 10, 2

    lag_lo = max(lo_allowed, center - half)
    lag_hi = min(hi_allowed, center + half)

    if lag_lo > lag_hi:
        lag_lo, lag_hi = lo_allowed, hi_allowed

    return int(lag_lo), int(lag_hi)

def _repair_weight_145(row, group: str) -> float:
    score = _safe_float_145(row.get("replication_score"), 0.0)
    zimm = _safe_float_145(row.get("z_immediate_core"), 0.0)
    zsharp = _safe_float_145(row.get("z_sharpness_core"), 0.0)
    cov = _safe_float_145(row.get("coverage_core"), 0.0)
    lagdiff = _safe_float_145(row.get("lag_agreement_abs_diff"), 10.0)

    raw = (
        0.40 * max(0.0, zimm) / 6.0
        + 0.30 * max(0.0, zsharp) / 6.0
        + 0.20 * max(0.0, cov)
        + 0.10 * max(0.0, score) / 10.0
        - 0.05 * max(0.0, lagdiff) / 10.0
    )

    if group == "A0":
        lo, hi = 0.25, 1.00
    elif group == "A1":
        lo, hi = 0.15, 0.70
    elif group == "router":
        lo, hi = 0.05, 0.45
    else:
        lo, hi = 0.05, 0.25

    return float(np.clip(raw, lo, hi))

def _profile_family_145(protocol_col: str) -> str:
    s = str(protocol_col).lower()

    if "pkt" in s or "packet" in s:
        return "count_packet"
    if "byte" in s:
        return "count_bytes"
    if "dns" in s:
        return "count_dns"
    if "arp" in s:
        return "count_arp"
    if "tcp" in s:
        return "count_tcp"
    if "udp" in s:
        return "count_udp"
    if "cmd" in s or "ack" in s or "data" in s:
        return "zigbee_semantic_count"
    if "unique" in s:
        return "cardinality"
    if "rate" in s or "pps" in s:
        return "rate_or_intensity"
    return "generic_protocol"

def _repair_safety_class_145(protocol_col: str, manifest_group: str) -> str:
    if manifest_group == "A0_zigbee_safe":
        return "safe_immediate"
    if manifest_group == "A1_zigbee_extended":
        if _is_derived_protocol_col_145(protocol_col):
            return "needs_derived_feature_policy"
        return "needs_domain_constraints"
    if manifest_group == "B_router_review":
        if _is_derived_protocol_col_145(protocol_col):
            return "diagnostic_only_derived_router_feature"
        return "review_required_router_raw_feature"
    return "diagnostic_only"

def _sort_manifest_145(df: pd.DataFrame):
    if len(df) == 0:
        return df.copy()

    sort_cols = []
    asc = []

    for c, a in [
        ("repair_priority", True),
        ("repair_manifest_group", True),
        ("replication_score", False),
        ("z_immediate_core", False),
        ("z_sharpness_core", False),
        ("coverage_core", False),
        ("lag_agreement_abs_diff", True),
    ]:
        if c in df.columns:
            sort_cols.append(c)
            asc.append(a)

    if sort_cols:
        return df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    return df.reset_index(drop=True)

# ----------------------------------------------------------
# 3) Validate upstream discovery / repair status contracts
# ----------------------------------------------------------
def _require_contract_version_145(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.5] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.5] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

version144_145 = _require_contract_version_145(
    CELL14_4_DEEP_COUPLING_CONTRACT,
    "CELL14_4_DEEP_COUPLING_CONTRACT",
    "cell14_4_two_stage_trainval_iot_protocol_coupling_tiering_v2_1",
)
version143_145 = _require_contract_version_145(
    CELL14_3_Q4_REPAIR_CONTRACT,
    "CELL14_3_Q4_REPAIR_CONTRACT",
    "cell14_3_coupling_conditioned_q4_re_evaluation_v1_1",
)
version134_145 = _require_contract_version_145(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)

strict144_145 = CELL14_4_DEEP_COUPLING_CONTRACT.get("strict_contract", {})
if bool(strict144_145.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.5] Cell 14.4 contract indicates TEST real values were used.")
if bool(strict144_145.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.5] Cell 14.4 contract indicates synthetic values were mutated.")
if bool(strict144_145.get("generator_fit_done_here", True)):
    raise RuntimeError("[Cell14.5] Cell 14.4 contract indicates generator fitting was done.")
if bool(strict144_145.get("materialization_done_here", True)):
    raise RuntimeError("[Cell14.5] Cell 14.4 contract indicates materialization was done.")

strict143_145 = CELL14_3_Q4_REPAIR_CONTRACT.get("strict_contract", {})
if bool(strict143_145.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.5] Cell 14.3 contract indicates synthetic values were mutated.")
if bool(strict143_145.get("generator_fit_done_here", True)):
    raise RuntimeError("[Cell14.5] Cell 14.3 contract indicates generator fitting was done.")
if bool(strict143_145.get("materialization_done_here", True)):
    raise RuntimeError("[Cell14.5] Cell 14.3 contract indicates materialization was done.")
if not bool(strict143_145.get("repair_candidate_not_final_claim", False)):
    raise RuntimeError("[Cell14.5] Cell 14.3 did not declare repair_candidate_not_final_claim.")

broad_q4_status_145 = str(CELL13_4_Q4_PUBLICATION_SUMMARY.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_145 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("pair_blocker_n", 0) or 0
)
broad_q4_pair_pass_rate_145 = _safe_float_145(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)

repair_status_145 = str(CELL14_3_Q4_REPAIR_CONTRACT.get("repair_candidate_publication_status", ""))
repair_reasons_145 = CELL14_3_Q4_REPAIR_CONTRACT.get("repair_candidate_publication_reasons", [])
repair_post_summary_145 = CELL14_3_Q4_REPAIR_CONTRACT.get("post_coupling_summary_14_3", {})
repair_delta_summary_145 = CELL14_3_Q4_REPAIR_CONTRACT.get("delta_summary", {})
repair_post_blocker_n_145 = int(repair_post_summary_145.get("publication_blocker_n", 0) or 0)
repair_post_pass_rate_145 = _safe_float_145(repair_post_summary_145.get("pass_rate", np.nan), np.nan)

# ----------------------------------------------------------
# 4) Load evidence
# ----------------------------------------------------------
rep = CELL14_4_DEEP_COUPLING_REPLICATION_DF.copy()

if len(rep) == 0:
    raise RuntimeError("[Cell14.5] Empty Cell 14.4 replication dataframe.")

required_cols = [
    "anchor_col",
    "anchor_name",
    "anchor_type",
    "protocol_col",
    "protocol_tier",
    "coupling_tier",
    "replication_score",
]
missing_cols = [c for c in required_cols if c not in rep.columns]
if missing_cols:
    raise RuntimeError(f"[Cell14.5] Cell 14.4 replication dataframe missing columns: {missing_cols}")

rep["anchor_col"] = rep["anchor_col"].astype(str)
rep["anchor_name"] = rep["anchor_name"].astype(str)
rep["anchor_type"] = rep["anchor_type"].astype(str)
rep["protocol_col"] = rep["protocol_col"].astype(str)
rep["protocol_tier"] = rep["protocol_tier"].astype(str).str.lower()
rep["coupling_tier"] = rep["coupling_tier"].astype(str)

# Ensure numeric helper columns.
for c in [
    "replication_score",
    "z_immediate_core",
    "z_sharpness_core",
    "coverage_core",
    "lag_agreement_abs_diff",
    "delta_immediate_core",
    "sharpness_core",
    "peak_lag_sec_train",
    "peak_lag_sec_val",
    "delta_immediate_med_train",
    "delta_immediate_med_val",
    "sharpness_med_train",
    "sharpness_med_val",
]:
    if c in rep.columns:
        rep[c] = pd.to_numeric(rep[c], errors="coerce")

# ----------------------------------------------------------
# 5) Policy classification
# ----------------------------------------------------------
A0_cols = set(map(str, CFG.get("cell14_5_A0_zigbee_safe_protocol_cols", [])))
A1_zigbee_allow = set(map(str, CFG.get("cell14_5_A1_zigbee_extended_protocol_col_allowlist", [])))
router_allow = set(map(str, CFG.get("cell14_5_B_router_review_protocol_col_allowlist", [])))

policy_rows = []

for _, r in rep.iterrows():
    row = r.to_dict()

    coupling_tier = str(row.get("coupling_tier", ""))
    protocol_col = str(row.get("protocol_col", ""))
    protocol_tier = str(row.get("protocol_tier", "")).lower()
    anchor_col = str(row.get("anchor_col", ""))

    is_event_driver = _is_event_driver_anchor_145(anchor_col)
    derived_protocol = _is_derived_protocol_col_145(protocol_col)
    replication_score = _safe_float_145(row.get("replication_score"), np.nan)

    repair_group = "diagnostic_only"
    repair_eligible = False
    repair_priority = 999
    repair_reason = []
    mode = "diagnostic"

    # A0: safest immediate Zigbee expansion.
    if (
        coupling_tier == "A"
        and protocol_tier == "zigbee"
        and protocol_col in A0_cols
        and replication_score >= float(CFG.get("cell14_5_A0_min_replication_score", 0.0))
        and _lag_ok_145(row, "cell14_5_A0_allowed_lag_lo", "cell14_5_A0_allowed_lag_hi")
        and (is_event_driver or not bool(CFG.get("cell14_5_A0_require_event_driver_anchor", True)))
    ):
        repair_group = "A0_zigbee_safe"
        repair_eligible = True
        repair_priority = 0
        mode = "A0"
        repair_reason.append("tierA_zigbee_safe_raw_count_target")

    # A1: extended Zigbee features.
    elif (
        coupling_tier == "A"
        and protocol_tier == "zigbee"
        and protocol_col in A1_zigbee_allow
        and protocol_col not in A0_cols
        and replication_score >= float(CFG.get("cell14_5_A1_min_replication_score", 0.0))
        and _lag_ok_145(row, "cell14_5_A1_allowed_lag_lo", "cell14_5_A1_allowed_lag_hi")
        and (is_event_driver or not bool(CFG.get("cell14_5_A1_require_event_driver_anchor", True)))
    ):
        repair_group = "A1_zigbee_extended"
        repair_eligible = True
        repair_priority = 1
        mode = "A1"
        repair_reason.append("tierA_zigbee_extended_domain_constrained_target")

    # Router evidence, review only.
    elif (
        coupling_tier in {"A", "B"}
        and protocol_tier == "router"
        and protocol_col in router_allow
        and replication_score >= float(CFG.get("cell14_5_router_min_replication_score", 0.0))
        and _lag_ok_145(row, "cell14_5_router_allowed_lag_lo", "cell14_5_router_allowed_lag_hi")
        and (is_event_driver or not bool(CFG.get("cell14_5_router_require_event_driver_anchor", True)))
    ):
        repair_group = "B_router_review"
        repair_eligible = False  # Intentionally not directly repairable yet.
        repair_priority = 2
        mode = "router"
        repair_reason.append("router_coupling_evidence_review_before_repair")

    else:
        repair_group = "diagnostic_only"
        repair_eligible = False
        repair_priority = 9

        if coupling_tier not in {"A", "B", "C"}:
            repair_reason.append("weak_or_rejected_coupling_tier")
        if protocol_tier not in {"zigbee", "router"}:
            repair_reason.append("unsupported_protocol_tier_for_repair")
        if not is_event_driver:
            repair_reason.append("non_event_driver_anchor")
        if derived_protocol:
            repair_reason.append("derived_protocol_feature_not_directly_repairable")
        if protocol_tier == "zigbee" and protocol_col not in A1_zigbee_allow:
            repair_reason.append("zigbee_protocol_col_not_in_repair_allowlist")
        if protocol_tier == "router" and protocol_col not in router_allow:
            repair_reason.append("router_protocol_col_not_in_review_allowlist")

    lag_lo, lag_hi = _recommended_lag_window_145(row, mode)
    weight = _repair_weight_145(row, mode)

    safety_class = _repair_safety_class_145(protocol_col, repair_group)

    direct_repair_allowed_now = bool(repair_group == "A0_zigbee_safe" and repair_eligible)
    requires_additional_domain_constraints = bool(repair_group == "A1_zigbee_extended")
    requires_manual_review = bool(repair_group == "B_router_review")
    diagnostic_only_flag = bool(repair_group == "diagnostic_only")

    out = dict(row)
    out.update({
        "repair_manifest_group": repair_group,
        "repair_eligible": bool(repair_eligible),
        "direct_repair_allowed_now": bool(direct_repair_allowed_now),
        "requires_additional_domain_constraints": bool(requires_additional_domain_constraints),
        "requires_manual_review": bool(requires_manual_review),
        "diagnostic_only": bool(diagnostic_only_flag),
        "repair_priority": int(repair_priority),
        "repair_reason": "|".join(repair_reason) if repair_reason else "policy_pass",
        "repair_safety_class": safety_class,
        "is_event_driver_anchor": bool(is_event_driver),
        "is_derived_protocol_col": bool(derived_protocol),
        "profile_family": _profile_family_145(protocol_col),
        "recommended_lag_lo": int(lag_lo),
        "recommended_lag_hi": int(lag_hi),
        "recommended_weight": float(weight),
        "TEST_real_values_used": False,
        "synthetic_values_mutated": False,
    })

    policy_rows.append(out)

policy_df = pd.DataFrame(policy_rows)
policy_df = _sort_manifest_145(policy_df)

# ----------------------------------------------------------
# 6) Split manifests
# ----------------------------------------------------------
A0_zigbee_safe = policy_df[policy_df["repair_manifest_group"].eq("A0_zigbee_safe")].copy()
A1_zigbee_extended = policy_df[policy_df["repair_manifest_group"].eq("A1_zigbee_extended")].copy()
B_router_review = policy_df[policy_df["repair_manifest_group"].eq("B_router_review")].copy()
diagnostic_only = policy_df[policy_df["repair_manifest_group"].eq("diagnostic_only")].copy()

# Apply max caps.
A0_zigbee_safe = A0_zigbee_safe.head(int(CFG.get("cell14_5_max_A0_zigbee_safe_pairs", 40))).copy()
A1_zigbee_extended = A1_zigbee_extended.head(int(CFG.get("cell14_5_max_A1_zigbee_extended_pairs", 80))).copy()
B_router_review = B_router_review.head(int(CFG.get("cell14_5_max_B_router_review_pairs", 60))).copy()

# Important: anything excluded by cap should remain visible as diagnostic.
kept_keys = set()
for d in [A0_zigbee_safe, A1_zigbee_extended, B_router_review]:
    if len(d):
        kept_keys.update(zip(d["anchor_col"].astype(str), d["protocol_col"].astype(str), d["repair_manifest_group"].astype(str)))

def _is_kept_policy_row(row):
    return (
        str(row["anchor_col"]),
        str(row["protocol_col"]),
        str(row["repair_manifest_group"]),
    ) in kept_keys

policy_df["kept_in_capped_manifest"] = policy_df.apply(_is_kept_policy_row, axis=1)

# Recompute diagnostic_only to include capped-out non-kept rows.
diagnostic_only = policy_df[
    policy_df["repair_manifest_group"].eq("diagnostic_only")
    | (~policy_df["kept_in_capped_manifest"].astype(bool))
].copy()

# Add candidate IDs.
def _assign_candidate_ids_145(df: pd.DataFrame, prefix: str):
    out = df.copy()
    if len(out):
        out.insert(0, "repair_candidate_id", [f"{prefix}_{i:04d}" for i in range(len(out))])
    else:
        out["repair_candidate_id"] = []
    return out

A0_zigbee_safe = _assign_candidate_ids_145(A0_zigbee_safe, "A0_ZIGBEE_SAFE")
A1_zigbee_extended = _assign_candidate_ids_145(A1_zigbee_extended, "A1_ZIGBEE_EXT")
B_router_review = _assign_candidate_ids_145(B_router_review, "B_ROUTER_REVIEW")
diagnostic_only = _assign_candidate_ids_145(diagnostic_only, "DIAGNOSTIC_ONLY")

# ----------------------------------------------------------
# 7) Save outputs
# ----------------------------------------------------------
policy_df.to_csv(all_policy_csv, index=False)
A0_zigbee_safe.to_csv(A0_zigbee_safe_csv, index=False)
A1_zigbee_extended.to_csv(A1_zigbee_extended_csv, index=False)
B_router_review.to_csv(B_router_review_csv, index=False)
diagnostic_only.to_csv(diagnostic_only_csv, index=False)

summary_rows = []

def _summary_for_145(name, df):
    if len(df) == 0:
        return {
            "manifest_group": name,
            "rows": 0,
            "unique_anchors": 0,
            "unique_protocol_cols": 0,
            "protocol_tiers": {},
            "mean_replication_score": np.nan,
            "mean_z_immediate_core": np.nan,
            "mean_z_sharpness_core": np.nan,
            "mean_coverage_core": np.nan,
            "mean_lag_abs_diff": np.nan,
            "mean_recommended_weight": np.nan,
        }

    return {
        "manifest_group": name,
        "rows": int(len(df)),
        "unique_anchors": int(df["anchor_col"].astype(str).nunique()),
        "unique_protocol_cols": int(df["protocol_col"].astype(str).nunique()),
        "protocol_tiers": df["protocol_tier"].astype(str).value_counts().sort_index().to_dict(),
        "mean_replication_score": float(pd.to_numeric(df["replication_score"], errors="coerce").mean()),
        "mean_z_immediate_core": float(pd.to_numeric(df["z_immediate_core"], errors="coerce").mean()) if "z_immediate_core" in df.columns else np.nan,
        "mean_z_sharpness_core": float(pd.to_numeric(df["z_sharpness_core"], errors="coerce").mean()) if "z_sharpness_core" in df.columns else np.nan,
        "mean_coverage_core": float(pd.to_numeric(df["coverage_core"], errors="coerce").mean()) if "coverage_core" in df.columns else np.nan,
        "mean_lag_abs_diff": float(pd.to_numeric(df["lag_agreement_abs_diff"], errors="coerce").mean()) if "lag_agreement_abs_diff" in df.columns else np.nan,
        "mean_recommended_weight": float(pd.to_numeric(df["recommended_weight"], errors="coerce").mean()),
    }

summary_rows.append(_summary_for_145("A0_zigbee_safe", A0_zigbee_safe))
summary_rows.append(_summary_for_145("A1_zigbee_extended", A1_zigbee_extended))
summary_rows.append(_summary_for_145("B_router_review", B_router_review))
summary_rows.append(_summary_for_145("diagnostic_only", diagnostic_only))

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(policy_summary_csv, index=False)

policy_counts = policy_df["repair_manifest_group"].astype(str).value_counts().sort_index().to_dict()
kept_counts = {
    "A0_zigbee_safe": int(len(A0_zigbee_safe)),
    "A1_zigbee_extended": int(len(A1_zigbee_extended)),
    "B_router_review": int(len(B_router_review)),
    "diagnostic_only": int(len(diagnostic_only)),
}

direct_repair_allowed_now_n = int(
    policy_df.get("direct_repair_allowed_now", pd.Series(False, index=policy_df.index))
    .fillna(False).astype(bool).sum()
)
requires_additional_domain_constraints_n = int(
    policy_df.get("requires_additional_domain_constraints", pd.Series(False, index=policy_df.index))
    .fillna(False).astype(bool).sum()
)
requires_manual_review_n = int(
    policy_df.get("requires_manual_review", pd.Series(False, index=policy_df.index))
    .fillna(False).astype(bool).sum()
)

# ----------------------------------------------------------
# 8) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "14.5",
    "version": CELL145_VERSION,
    "role": "coupling_repair_manifest_policy_layer",
    "quality_dimension": "Q4_cross_modal_consistency_repair_manifest_expansion",
    "upstream_contract_versions": {
        "cell13_4": version134_145,
        "cell14_3": version143_145,
        "cell14_4": version144_145
    },
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_145,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_145),
        "pair_pass_rate": broad_q4_pair_pass_rate_145
    },
    "repair_candidate_status_carried_forward": {
        "repair_candidate_status": repair_status_145,
        "repair_candidate_reasons": repair_reasons_145,
        "post_repair_publication_blocker_n": int(repair_post_blocker_n_145),
        "post_repair_pass_rate": repair_post_pass_rate_145,
        "repair_delta_summary": repair_delta_summary_145
    },
    "input_evidence": {
        "source_cell": "14.4",
        "input_replication_rows": int(len(rep)),
        "cell14_4_version": str(CELL14_4_DEEP_COUPLING_CONTRACT.get("version", "")),
        "cell14_4_tier_counts": CELL14_4_DEEP_COUPLING_CONTRACT.get("tier_counts", {}),
        "cell14_4_tier_protocol_counts": CELL14_4_DEEP_COUPLING_CONTRACT.get("tier_protocol_counts", []),
    },
    "policy_counts_before_caps": {str(k): int(v) for k, v in policy_counts.items()},
    "kept_manifest_counts": kept_counts,
    "repair_safety_counts": {
        "direct_repair_allowed_now_n": int(direct_repair_allowed_now_n),
        "requires_additional_domain_constraints_n": int(requires_additional_domain_constraints_n),
        "requires_manual_review_n": int(requires_manual_review_n)
    },
    "policy": {
        "A0_zigbee_safe": {
            "purpose": "immediate safe extension of current 14.1-style Zigbee repair",
            "protocol_cols": sorted(list(A0_cols)),
            "max_pairs": int(CFG.get("cell14_5_max_A0_zigbee_safe_pairs", 40)),
            "repair_eligible": True,
            "direct_repair_allowed_now": True,
        },
        "A1_zigbee_extended": {
            "purpose": "extended Zigbee repair candidates requiring domain constraints",
            "protocol_col_allowlist": sorted(list(A1_zigbee_allow)),
            "max_pairs": int(CFG.get("cell14_5_max_A1_zigbee_extended_pairs", 80)),
            "repair_eligible": True,
            "direct_repair_allowed_now": False,
            "requires_additional_domain_constraints": True,
            "warning": "Do not apply with the same operator blindly unless protocol-specific constraints are implemented.",
        },
        "B_router_review": {
            "purpose": "router coupling evidence requiring manual review and separate repair branch",
            "protocol_col_allowlist": sorted(list(router_allow)),
            "max_pairs": int(CFG.get("cell14_5_max_B_router_review_pairs", 60)),
            "repair_eligible": False,
            "direct_repair_allowed_now": False,
            "requires_manual_review": True,
            "warning": "Router repair is intentionally disabled here because router coupling is more confounded.",
        },
        "diagnostic_only": {
            "purpose": "evidence not safe for direct repair or capped out of manifests",
            "repair_eligible": False,
        },
    },
    "strict_contract": {
        "trainval_evidence_used": True,
        "broad_q4_status_carried_forward": True,
        "repair_status_carried_forward": True,
        "repair_manifest_policy_assignment_done_here": True,
        "repair_eligibility_gating_done_here": True,
        "TEST_real_values_used": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "policy_layer_only": True,
    },
    "outputs": {
        "A0_zigbee_safe_csv": A0_zigbee_safe_csv,
        "A1_zigbee_extended_csv": A1_zigbee_extended_csv,
        "B_router_review_csv": B_router_review_csv,
        "diagnostic_only_csv": diagnostic_only_csv,
        "all_policy_csv": all_policy_csv,
        "policy_summary_csv": policy_summary_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_145(contract_json, contract)
_write_json_145(contract_canonical_json, contract)

manifest = {
    "cell": "14.5",
    "version": CELL145_VERSION,
    "created_outputs": contract["outputs"],
    "kept_manifest_counts": kept_counts,
    "repair_safety_counts": contract["repair_safety_counts"],
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "repair_candidate_status_carried_forward": contract["repair_candidate_status_carried_forward"],
    "policy_counts_before_caps": {str(k): int(v) for k, v in policy_counts.items()},
    "strict_contract": contract["strict_contract"],
}

_write_json_145(manifest_json, manifest)

hashes = {
    "A0_zigbee_safe_csv_sha256": _sha256_file_145(A0_zigbee_safe_csv),
    "A1_zigbee_extended_csv_sha256": _sha256_file_145(A1_zigbee_extended_csv),
    "B_router_review_csv_sha256": _sha256_file_145(B_router_review_csv),
    "diagnostic_only_csv_sha256": _sha256_file_145(diagnostic_only_csv),
    "all_policy_csv_sha256": _sha256_file_145(all_policy_csv),
    "policy_summary_csv_sha256": _sha256_file_145(policy_summary_csv),
    "contract_json_sha256": _sha256_file_145(contract_json),
    "contract_canonical_json_sha256": _sha256_file_145(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_145(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_145(contract_json, contract)
_write_json_145(contract_canonical_json, contract)
_write_json_145(manifest_json, manifest)

# ----------------------------------------------------------
# 9) Export globals
# ----------------------------------------------------------
globals()["CELL145_VERSION"] = CELL145_VERSION
globals()["CELL14_5_REPAIR_MANIFEST_ALL_POLICY_DF"] = policy_df
globals()["CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_DF"] = A0_zigbee_safe
globals()["CELL14_5_REPAIR_MANIFEST_A1_ZIGBEE_EXTENDED_DF"] = A1_zigbee_extended
globals()["CELL14_5_REPAIR_MANIFEST_B_ROUTER_REVIEW_DF"] = B_router_review
globals()["CELL14_5_REPAIR_MANIFEST_DIAGNOSTIC_ONLY_DF"] = diagnostic_only
globals()["CELL14_5_REPAIR_MANIFEST_POLICY_SUMMARY_DF"] = summary_df
globals()["CELL14_5_REPAIR_MANIFEST_CONTRACT"] = contract

globals()["CELL14_5_REPAIR_MANIFEST_A0_ZIGBEE_SAFE_CSV"] = A0_zigbee_safe_csv
globals()["CELL14_5_REPAIR_MANIFEST_A1_ZIGBEE_EXTENDED_CSV"] = A1_zigbee_extended_csv
globals()["CELL14_5_REPAIR_MANIFEST_B_ROUTER_REVIEW_CSV"] = B_router_review_csv
globals()["CELL14_5_REPAIR_MANIFEST_DIAGNOSTIC_ONLY_CSV"] = diagnostic_only_csv
globals()["CELL14_5_REPAIR_MANIFEST_ALL_POLICY_CSV"] = all_policy_csv
globals()["CELL14_5_REPAIR_MANIFEST_POLICY_SUMMARY_CSV"] = policy_summary_csv
globals()["CELL14_5_REPAIR_MANIFEST_CONTRACT_JSON"] = contract_json
globals()["CELL14_5_REPAIR_MANIFEST_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL14_5_REPAIR_MANIFEST_POLICY_MANIFEST_JSON"] = manifest_json

log(
    "[Cell14.5] Repair manifest policy complete | "
    f"input_rows={len(rep)} | "
    f"A0_zigbee_safe={len(A0_zigbee_safe)} | "
    f"A1_zigbee_extended={len(A1_zigbee_extended)} | "
    f"B_router_review={len(B_router_review)} | "
    f"diagnostic_only={len(diagnostic_only)}"
)
log(f"[Cell14.5] Policy counts before caps | {policy_counts}")
log(f"[Cell14.5] Kept manifest counts | {kept_counts}")
log(f"[Cell14.5] Saved A0 Zigbee safe manifest: {A0_zigbee_safe_csv} | rows={len(A0_zigbee_safe)}")
log(f"[Cell14.5] Saved A1 Zigbee extended manifest: {A1_zigbee_extended_csv} | rows={len(A1_zigbee_extended)}")
log(f"[Cell14.5] Saved router review manifest: {B_router_review_csv} | rows={len(B_router_review)}")
log(f"[Cell14.5] Saved diagnostic-only manifest: {diagnostic_only_csv} | rows={len(diagnostic_only)}")
log(f"[Cell14.5] Saved contract: {contract_json}")
log(f"[Cell14.5] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell14.5] Safety-policy counts | "
    f"direct_repair_allowed_now_n={direct_repair_allowed_now_n} | "
    f"requires_additional_domain_constraints_n={requires_additional_domain_constraints_n} | "
    f"requires_manual_review_n={requires_manual_review_n}"
)
log(
    "[Cell14.5] Upstream Q4/repair status carried forward | "
    f"broad_status={broad_q4_status_145} | "
    f"broad_pair_blocker_n={broad_q4_pair_blocker_n_145} | "
    f"repair_status={repair_status_145} | "
    f"repair_post_blocker_n={repair_post_blocker_n_145} | "
    f"repair_post_pass_rate={repair_post_pass_rate_145}"
)
log(
    "[Cell14.5] Contract flags | "
    "trainval_evidence_used=True | "
    "TEST_real_values_used=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "policy_layer_only=True | repair_manifest_policy_assignment_done_here=True | direct_repair_allowed_now_is_A0_only=True"
)
log("--- END: Cell 14.5 - Coupling manifest consolidation and repair eligibility policy (v1.1 repair-status-aware strict) ---")

gc.collect()