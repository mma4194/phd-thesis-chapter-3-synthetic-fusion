# ==========================================================
# CELL 14.8 - A1 and router evidence audit before expansion
# v1.1 STUDY-THESIS strict audit-only protocol-consistency review, Q4-governance-aware
#
# Role:
#   - Audit A1 Zigbee extended evidence and router evidence before any expansion.
#   - Split A1 Zigbee into:
#       A1a_zigbee_semantic_counts
#       A1b_zigbee_rates
#       A1c_zigbee_cardinality
#       A1d_zigbee_other
#   - Split router evidence into:
#       Router_R0_raw_review
#       Router_R1_derived_or_confounded
#   - Identify domain consistency constraints required before direct repair.
#
# Scientific contract:
#   - Uses TRAIN/VAL evidence from Cells 14.4 and 14.5.
#   - Does NOT use TEST real values.
#   - Does NOT mutate synthetic values.
#   - Does NOT fit models.
#   - Does NOT materialize.
#   - Does NOT promote candidates.
#
# Outputs:
#   reports/cell14_8_A1_zigbee_expansion_audit.csv
#   reports/cell14_8_router_evidence_audit.csv
#   reports/cell14_8_protocol_consistency_constraints.csv
#   reports/cell14_8_expansion_policy_summary.csv
#   reports/cell14_8_A1_router_audit_contract.json
#   artifacts/cell14_8_A1_router_audit_manifest.json
# ==========================================================

log("--- START: Cell 14.8 - A1 and router evidence audit before expansion (v1.1 Q4-governance-aware strict) ---")

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
_required_148 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL14_4_DEEP_COUPLING_REPLICATION_DF",
    "CELL14_5_REPAIR_MANIFEST_ALL_POLICY_DF",
    "CELL14_5_REPAIR_MANIFEST_A1_ZIGBEE_EXTENDED_DF",
    "CELL14_5_REPAIR_MANIFEST_B_ROUTER_REVIEW_DF",
    "CELL14_5_REPAIR_MANIFEST_DIAGNOSTIC_ONLY_DF",
    "CELL14_5_REPAIR_MANIFEST_CONTRACT",
    "CELL14_7_Q4_NO_PROMOTION_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
]
_missing_148 = [k for k in _required_148 if k not in globals()]
if _missing_148:
    raise RuntimeError(f"[Cell14.8] Missing required globals from Cells 14.4/14.5: {_missing_148}")

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

CELL148_VERSION = "cell14_8_A1_router_protocol_consistency_audit_v1_1_q4_governance_aware"

CFG["cell14_8_version"] = CELL148_VERSION
CFG["cell14_8_trainval_evidence_used"] = True
CFG["cell14_8_TEST_real_values_used"] = False
CFG["cell14_8_synthetic_values_mutated"] = False
CFG["cell14_8_selection_done_here"] = False
CFG["cell14_8_generator_fit_done_here"] = False
CFG["cell14_8_materialization_done_here"] = False
CFG["cell14_8_audit_only"] = True
CFG["cell14_8_q4_no_promotion_status_carried_forward"] = True
CFG["cell14_8_expansion_not_authorized_here"] = True
CFG["cell14_8_direct_repair_authorized_here"] = False

# ----------------------------------------------------------
# 1) Policy knobs
# ----------------------------------------------------------
CFG.setdefault(
    "cell14_8_zigbee_semantic_count_cols",
    ["zigbee__cmd", "zigbee__ack", "zigbee__data"],
)
CFG.setdefault(
    "cell14_8_zigbee_rate_cols",
    ["zigbee__cmd_rate", "zigbee__pps"],
)
CFG.setdefault(
    "cell14_8_zigbee_cardinality_cols",
    ["zigbee__src16_unique", "zigbee__dst16_unique", "zigbee__panid_unique"],
)
CFG.setdefault(
    "cell14_8_zigbee_base_count_cols",
    ["zigbee__pkt_total", "zigbee__bytes_total"],
)
CFG.setdefault(
    "cell14_8_router_raw_review_cols",
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
CFG.setdefault(
    "cell14_8_derived_feature_tokens",
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
        "zscore",
        "diff",
        "delta",
    ],
)
CFG.setdefault("cell14_8_high_confidence_min_replication_score", 5.0)
CFG.setdefault("cell14_8_high_confidence_min_z_immediate_core", 5.0)
CFG.setdefault("cell14_8_high_confidence_min_z_sharpness_core", 5.0)
CFG.setdefault("cell14_8_high_confidence_max_lag_abs_diff", 2.0)
CFG.setdefault("cell14_8_min_coverage_core_for_expansion", 0.80)

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
a1_audit_csv = os.path.join(REPORT_DIR, "cell14_8_A1_zigbee_expansion_audit.csv")
router_audit_csv = os.path.join(REPORT_DIR, "cell14_8_router_evidence_audit.csv")
constraints_csv = os.path.join(REPORT_DIR, "cell14_8_protocol_consistency_constraints.csv")
summary_csv = os.path.join(REPORT_DIR, "cell14_8_expansion_policy_summary.csv")
contract_json = os.path.join(REPORT_DIR, "cell14_8_A1_router_audit_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell14_8_A1_router_audit_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell14_8_A1_router_audit_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_148(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_148(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_148(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_148(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_148(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_148(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_148(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_148(obj.to_dict())
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

def _write_json_148(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_148(payload), f, indent=2, sort_keys=True)

def _sha256_file_148(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_148(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_int_148(x, default=0) -> int:
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _to_num_148(frame: pd.DataFrame, col: str) -> np.ndarray:
    if col not in frame.columns:
        return np.asarray([], dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _is_derived_col_148(col: str) -> bool:
    s = str(col).lower()
    return any(tok in s for tok in [str(t).lower() for t in CFG.get("cell14_8_derived_feature_tokens", [])])

def _is_event_driver_anchor_148(col: str) -> bool:
    s = str(col)
    return bool(s.startswith("events_in_sec__entity__") or s.startswith("events_in_sec__feat__"))

def _protocol_tier_148(col: str) -> str:
    s = str(col).lower()
    if s.startswith("zigbee__") or s.startswith("zb__") or "zigbee" in s:
        return "zigbee"
    if s.startswith("router__") or s.startswith("dns__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__") or s.startswith("wifi__"):
        return "ota"
    if s.startswith("zwave__") or "zwave" in s:
        return "zwave"
    return "other"

def _zigbee_subtier_148(protocol_col: str) -> str:
    col = str(protocol_col)
    semantic = set(map(str, CFG.get("cell14_8_zigbee_semantic_count_cols", [])))
    rates = set(map(str, CFG.get("cell14_8_zigbee_rate_cols", [])))
    cardinality = set(map(str, CFG.get("cell14_8_zigbee_cardinality_cols", [])))
    base = set(map(str, CFG.get("cell14_8_zigbee_base_count_cols", [])))

    if col in base:
        return "A0_base_count_already_accepted"
    if col in semantic:
        return "A1a_zigbee_semantic_counts"
    if col in rates:
        return "A1b_zigbee_rates"
    if col in cardinality:
        return "A1c_zigbee_cardinality"
    return "A1d_zigbee_other"

def _router_subtier_148(protocol_col: str) -> str:
    col = str(protocol_col)
    raw_review = set(map(str, CFG.get("cell14_8_router_raw_review_cols", [])))

    if col in raw_review and not _is_derived_col_148(col):
        return "Router_R0_raw_review"
    if _is_derived_col_148(col):
        return "Router_R1_derived_or_confounded"
    return "Router_R2_other_review"

def _evidence_strength_148(row: dict) -> str:
    score = _safe_float_148(row.get("replication_score"), np.nan)
    zimm = _safe_float_148(row.get("z_immediate_core"), np.nan)
    zsharp = _safe_float_148(row.get("z_sharpness_core"), np.nan)
    lagdiff = _safe_float_148(row.get("lag_agreement_abs_diff"), np.nan)
    cov = _safe_float_148(row.get("coverage_core"), np.nan)

    strong = bool(
        np.isfinite(score) and score >= float(CFG.get("cell14_8_high_confidence_min_replication_score", 5.0))
        and np.isfinite(zimm) and zimm >= float(CFG.get("cell14_8_high_confidence_min_z_immediate_core", 5.0))
        and np.isfinite(zsharp) and zsharp >= float(CFG.get("cell14_8_high_confidence_min_z_sharpness_core", 5.0))
        and np.isfinite(lagdiff) and lagdiff <= float(CFG.get("cell14_8_high_confidence_max_lag_abs_diff", 2.0))
        and np.isfinite(cov) and cov >= float(CFG.get("cell14_8_min_coverage_core_for_expansion", 0.80))
    )
    if strong:
        return "high_confidence"

    moderate = bool(
        np.isfinite(score) and score >= 2.5
        and np.isfinite(zimm) and zimm >= 2.5
        and np.isfinite(cov) and cov >= 0.50
    )
    if moderate:
        return "moderate_confidence"

    return "weak_or_review"

def _domain_summary_148(col: str) -> dict:
    """
    TRAIN/VAL-only domain summary.
    This does not touch TEST values.
    """
    tr = _to_num_148(df_tr, col)
    va = _to_num_148(df_val, col)

    x = np.concatenate([tr[np.isfinite(tr)], va[np.isfinite(va)]]) if tr.size or va.size else np.asarray([], dtype=np.float64)

    if x.size == 0:
        return {
            "trainval_finite_n": 0,
            "trainval_min": np.nan,
            "trainval_max": np.nan,
            "trainval_mean": np.nan,
            "trainval_nonzero_rate": np.nan,
            "trainval_integer_like": False,
            "trainval_nonnegative": False,
            "trainval_unique_n": 0,
        }

    integer_like = bool(np.nanmax(np.abs(x - np.rint(x))) <= 1e-6)
    nonnegative = bool(np.nanmin(x) >= 0.0)

    return {
        "trainval_finite_n": int(x.size),
        "trainval_min": float(np.nanmin(x)),
        "trainval_max": float(np.nanmax(x)),
        "trainval_mean": float(np.nanmean(x)),
        "trainval_nonzero_rate": float(np.mean(x > 0)),
        "trainval_integer_like": integer_like,
        "trainval_nonnegative": nonnegative,
        "trainval_unique_n": int(len(np.unique(x))),
    }

def _constraint_rows_for_zigbee_148(subtier: str, protocol_col: str) -> list:
    rows = []

    base_constraints = {
        "nonnegative": "values must be >= 0",
        "integer_like": "count/cardinality columns must remain integer-like",
        "respect_observability": "values must be NaN or inactive-safe under zigbee observability mask",
        "legal_window_only": "repair may only occur inside event-conditioned legal windows",
        "no_iot_mutation": "IoT driver/value/state columns must not be mutated",
        "no_test_materialization": "TEST real values must not be used for materialization",
    }

    for k, desc in base_constraints.items():
        rows.append({
            "protocol_col": protocol_col,
            "subtier": subtier,
            "constraint_name": k,
            "constraint_description": desc,
            "required_before_direct_repair": True,
        })

    if subtier == "A1a_zigbee_semantic_counts":
        rows.extend([
            {
                "protocol_col": protocol_col,
                "subtier": subtier,
                "constraint_name": "semantic_sum_bound",
                "constraint_description": "semantic counts such as cmd/ack/data should not jointly exceed zigbee__pkt_total within the same window unless packet semantics permit overlap",
                "required_before_direct_repair": True,
            },
            {
                "protocol_col": protocol_col,
                "subtier": subtier,
                "constraint_name": "pkt_total_consistency",
                "constraint_description": "semantic count uplift should be coordinated with zigbee__pkt_total, not independently over-amplified",
                "required_before_direct_repair": True,
            },
        ])

    elif subtier == "A1b_zigbee_rates":
        rows.extend([
            {
                "protocol_col": protocol_col,
                "subtier": subtier,
                "constraint_name": "rate_recompute_preferred",
                "constraint_description": "rate/intensity features should preferably be recomputed from base count features and elapsed time rather than directly nudged",
                "required_before_direct_repair": True,
            },
            {
                "protocol_col": protocol_col,
                "subtier": subtier,
                "constraint_name": "rate_count_consistency",
                "constraint_description": "rate features must remain consistent with corresponding count features in the response window",
                "required_before_direct_repair": True,
            },
        ])

    elif subtier == "A1c_zigbee_cardinality":
        rows.extend([
            {
                "protocol_col": protocol_col,
                "subtier": subtier,
                "constraint_name": "cardinality_upper_bound",
                "constraint_description": "unique/cardinality features must not exceed packet or event counts in the same window",
                "required_before_direct_repair": True,
            },
            {
                "protocol_col": protocol_col,
                "subtier": subtier,
                "constraint_name": "support_projection",
                "constraint_description": "cardinality values should be projected onto TRAIN/VAL support or plausible integer support",
                "required_before_direct_repair": True,
            },
        ])

    return rows

def _constraint_rows_for_router_148(subtier: str, protocol_col: str) -> list:
    rows = [
        {
            "protocol_col": protocol_col,
            "subtier": subtier,
            "constraint_name": "router_repair_separate_branch_required",
            "constraint_description": "router coupling is broader and more confounded; repair should be implemented in a separate router-specific branch",
            "required_before_direct_repair": True,
        },
        {
            "protocol_col": protocol_col,
            "subtier": subtier,
            "constraint_name": "primitive_feature_only",
            "constraint_description": "direct repair should target primitive count/byte/packet columns only; derived features should be recomputed",
            "required_before_direct_repair": True,
        },
        {
            "protocol_col": protocol_col,
            "subtier": subtier,
            "constraint_name": "non_target_protocol_drift_forbidden",
            "constraint_description": "router repair must prove no unintended drift in non-target router/OTA/Zigbee columns",
            "required_before_direct_repair": True,
        },
        {
            "protocol_col": protocol_col,
            "subtier": subtier,
            "constraint_name": "regime_confounding_audit_required",
            "constraint_description": "router coupling must be checked for broad regime/diurnal confounding before promotion",
            "required_before_direct_repair": True,
        },
    ]

    if subtier == "Router_R1_derived_or_confounded":
        rows.append({
            "protocol_col": protocol_col,
            "subtier": subtier,
            "constraint_name": "derived_feature_recompute_required",
            "constraint_description": "rolling/diff/rate/entropy/share/unique features should be recomputed from primitive columns, not directly repaired",
            "required_before_direct_repair": True,
        })

    return rows

def _expansion_recommendation_148(row: dict, family: str) -> tuple:
    """
    Return recommendation and rationale.
    """
    strength = str(row.get("evidence_strength", "weak_or_review"))
    subtier = str(row.get("expansion_subtier", ""))
    derived = bool(row.get("is_derived_protocol_col", False))
    event_driver = bool(row.get("is_event_driver_anchor", False))

    if family == "zigbee":
        if subtier == "A1a_zigbee_semantic_counts" and strength in {"high_confidence", "moderate_confidence"} and event_driver and not derived:
            return (
                "eligible_after_semantic_count_constraints",
                "May be repaired after enforcing semantic-count and pkt_total consistency constraints.",
            )
        if subtier == "A1b_zigbee_rates":
            return (
                "recompute_or_derive_not_direct_repair",
                "Rate/intensity features should be recomputed from repaired base count columns where possible.",
            )
        if subtier == "A1c_zigbee_cardinality":
            return (
                "eligible_only_with_cardinality_constraints",
                "Cardinality features require support projection and upper-bound constraints.",
            )
        return (
            "diagnostic_or_defer",
            "Zigbee evidence exists but is not yet safe for direct repair under current constraints.",
        )

    if family == "router":
        if subtier == "Router_R0_raw_review" and strength == "high_confidence" and event_driver and not derived:
            return (
                "router_raw_candidate_for_future_R0_branch",
                "Potential future router repair candidate, but requires router-specific confounding and drift audits.",
            )
        if subtier == "Router_R1_derived_or_confounded":
            return (
                "diagnostic_only_recompute_if_needed",
                "Derived/confounded router feature; do not directly repair.",
            )
        return (
            "router_review_defer",
            "Router evidence should remain review-only until a separate router branch is designed.",
        )

    return "diagnostic_only", "Unsupported family for expansion."

# ----------------------------------------------------------
# 4) Validate upstream policy and Q4 no-promotion governance
# ----------------------------------------------------------
def _require_contract_version_148(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.8] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.8] Unexpected {name} version. "
            f"Expected substring={expected_substring}, got={version}"
        )
    return version

version145_148 = _require_contract_version_148(
    CELL14_5_REPAIR_MANIFEST_CONTRACT,
    "CELL14_5_REPAIR_MANIFEST_CONTRACT",
    "cell14_5_coupling_repair_manifest_policy_v1_1",
)
version147_148 = _require_contract_version_148(
    CELL14_7_Q4_NO_PROMOTION_CONTRACT,
    "CELL14_7_Q4_NO_PROMOTION_CONTRACT",
    "cell14_7_q4_no_promotion_governance_v2_1",
)
version134_148 = _require_contract_version_148(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)

strict145_148 = CELL14_5_REPAIR_MANIFEST_CONTRACT.get("strict_contract", {})
if bool(strict145_148.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell14.8] Cell 14.5 contract indicates TEST real values were used.")
if bool(strict145_148.get("synthetic_values_mutated", True)):
    raise RuntimeError("[Cell14.8] Cell 14.5 contract indicates synthetic values were mutated.")
if bool(strict145_148.get("generator_fit_done_here", True)):
    raise RuntimeError("[Cell14.8] Cell 14.5 contract indicates generator fitting was done.")
if bool(strict145_148.get("materialization_done_here", True)):
    raise RuntimeError("[Cell14.8] Cell 14.5 contract indicates materialization was done.")

strict147_148 = CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("strict_contract", {})
if bool(CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("accepted", True)):
    raise RuntimeError("[Cell14.8] Cell 14.7 governance contract is not a no-promotion contract.")
if bool(CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("promoted", True)):
    raise RuntimeError("[Cell14.8] Cell 14.7 governance contract indicates promotion happened.")
if bool(strict147_148.get("candidate_outputs_promoted_to_final", True)):
    raise RuntimeError("[Cell14.8] Cell 14.7 governance contract indicates candidate outputs were promoted.")
if bool(strict147_148.get("synthetic_values_mutated_here", True)):
    raise RuntimeError("[Cell14.8] Cell 14.7 governance contract indicates synthetic mutation.")

q4_no_promotion_summary_148 = CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("summary", {})
q4_final_status_148 = str(
    q4_no_promotion_summary_148.get("governance_decision", {}).get("final_q4_status", "")
)
if q4_final_status_148 != "blocked_no_promotion":
    raise RuntimeError(
        "[Cell14.8] Cell 14.7 final Q4 status is not blocked_no_promotion: "
        f"{q4_final_status_148}"
    )

broad_q4_status_148 = str(CELL13_4_Q4_PUBLICATION_SUMMARY.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_148 = int(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("pair_blocker_n", 0) or 0
)
broad_q4_pair_pass_rate_148 = _safe_float_148(
    CELL13_4_Q4_PUBLICATION_SUMMARY.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)

# ----------------------------------------------------------
# 5) Load policy/evidence data
# ----------------------------------------------------------
all_policy = CELL14_5_REPAIR_MANIFEST_ALL_POLICY_DF.copy()
a1 = CELL14_5_REPAIR_MANIFEST_A1_ZIGBEE_EXTENDED_DF.copy()
router_review = CELL14_5_REPAIR_MANIFEST_B_ROUTER_REVIEW_DF.copy()
diagnostic = CELL14_5_REPAIR_MANIFEST_DIAGNOSTIC_ONLY_DF.copy()
rep = CELL14_4_DEEP_COUPLING_REPLICATION_DF.copy()

if len(all_policy) == 0:
    raise RuntimeError("[Cell14.8] Empty Cell 14.5 policy dataframe.")

# Normalize key columns.
for df in [all_policy, a1, router_review, diagnostic, rep]:
    if isinstance(df, pd.DataFrame) and len(df):
        for c in ["anchor_col", "protocol_col", "protocol_tier", "coupling_tier", "repair_manifest_group"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
        for c in [
            "replication_score", "z_immediate_core", "z_sharpness_core",
            "coverage_core", "lag_agreement_abs_diff",
            "delta_immediate_core", "sharpness_core",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

# ----------------------------------------------------------
# 6) A1 Zigbee audit
# ----------------------------------------------------------
a1_rows = []
constraint_rows = []

for _, r in a1.iterrows():
    row = r.to_dict()
    protocol_col = str(row.get("protocol_col", ""))
    anchor_col = str(row.get("anchor_col", ""))

    subtier = _zigbee_subtier_148(protocol_col)
    dom = _domain_summary_148(protocol_col)
    strength = _evidence_strength_148(row)
    derived = _is_derived_col_148(protocol_col)
    event_driver = _is_event_driver_anchor_148(anchor_col)

    out = dict(row)
    out.update({
        "expansion_family": "zigbee",
        "expansion_subtier": subtier,
        "evidence_strength": strength,
        "is_event_driver_anchor": event_driver,
        "is_derived_protocol_col_recomputed": derived,
        **dom,
    })

    rec, rationale = _expansion_recommendation_148(out, "zigbee")
    out["expansion_recommendation"] = rec
    out["expansion_rationale"] = rationale

    a1_rows.append(out)
    constraint_rows.extend(_constraint_rows_for_zigbee_148(subtier, protocol_col))

a1_audit_df = pd.DataFrame(a1_rows)

# ----------------------------------------------------------
# 7) Router evidence audit
# ----------------------------------------------------------
# Include B_router_review if any, plus router rows from all_policy/diagnostic
# that were diagnostic_only but came from A/B/C evidence.
router_sources = []

if len(router_review):
    router_sources.append(router_review)

router_diag = all_policy[
    all_policy["protocol_tier"].astype(str).str.lower().eq("router")
].copy()

if len(router_diag):
    router_sources.append(router_diag)

if router_sources:
    router_all = pd.concat(router_sources, axis=0, ignore_index=True).drop_duplicates(
        subset=["anchor_col", "protocol_col", "coupling_tier", "replication_score"],
        keep="first",
    )
else:
    router_all = pd.DataFrame()

router_rows = []

for _, r in router_all.iterrows():
    row = r.to_dict()
    protocol_col = str(row.get("protocol_col", ""))
    anchor_col = str(row.get("anchor_col", ""))

    subtier = _router_subtier_148(protocol_col)
    dom = _domain_summary_148(protocol_col)
    strength = _evidence_strength_148(row)
    derived = _is_derived_col_148(protocol_col)
    event_driver = _is_event_driver_anchor_148(anchor_col)

    out = dict(row)
    out.update({
        "expansion_family": "router",
        "expansion_subtier": subtier,
        "evidence_strength": strength,
        "is_event_driver_anchor": event_driver,
        "is_derived_protocol_col_recomputed": derived,
        **dom,
    })

    rec, rationale = _expansion_recommendation_148(out, "router")
    out["expansion_recommendation"] = rec
    out["expansion_rationale"] = rationale

    router_rows.append(out)
    constraint_rows.extend(_constraint_rows_for_router_148(subtier, protocol_col))

router_audit_df = pd.DataFrame(router_rows)

# ----------------------------------------------------------
# 8) Constraints table
# ----------------------------------------------------------
constraints_df = pd.DataFrame(constraint_rows)
if len(constraints_df):
    constraints_df = constraints_df.drop_duplicates(
        subset=["protocol_col", "subtier", "constraint_name"],
        keep="first",
    ).sort_values(["subtier", "protocol_col", "constraint_name"]).reset_index(drop=True)

# ----------------------------------------------------------
# 9) Summary
# ----------------------------------------------------------
summary_rows = []

def _summarize_group_148(name: str, df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return {
            "group": name,
            "rows": 0,
            "unique_protocol_cols": 0,
            "unique_anchors": 0,
            "subtier_counts": {},
            "recommendation_counts": {},
            "evidence_strength_counts": {},
            "mean_replication_score": np.nan,
            "mean_z_immediate_core": np.nan,
            "mean_z_sharpness_core": np.nan,
            "mean_coverage_core": np.nan,
        }

    return {
        "group": name,
        "rows": int(len(df)),
        "unique_protocol_cols": int(df["protocol_col"].astype(str).nunique()) if "protocol_col" in df.columns else 0,
        "unique_anchors": int(df["anchor_col"].astype(str).nunique()) if "anchor_col" in df.columns else 0,
        "subtier_counts": df["expansion_subtier"].astype(str).value_counts().sort_index().to_dict() if "expansion_subtier" in df.columns else {},
        "recommendation_counts": df["expansion_recommendation"].astype(str).value_counts().sort_index().to_dict() if "expansion_recommendation" in df.columns else {},
        "evidence_strength_counts": df["evidence_strength"].astype(str).value_counts().sort_index().to_dict() if "evidence_strength" in df.columns else {},
        "mean_replication_score": float(pd.to_numeric(df["replication_score"], errors="coerce").mean()) if "replication_score" in df.columns else np.nan,
        "mean_z_immediate_core": float(pd.to_numeric(df["z_immediate_core"], errors="coerce").mean()) if "z_immediate_core" in df.columns else np.nan,
        "mean_z_sharpness_core": float(pd.to_numeric(df["z_sharpness_core"], errors="coerce").mean()) if "z_sharpness_core" in df.columns else np.nan,
        "mean_coverage_core": float(pd.to_numeric(df["coverage_core"], errors="coerce").mean()) if "coverage_core" in df.columns else np.nan,
    }

summary_rows.append(_summarize_group_148("A1_zigbee_extended", a1_audit_df))
summary_rows.append(_summarize_group_148("router_evidence", router_audit_df))

# Add specific next-step counts.
a1_semantic_ready_n = int(
    (a1_audit_df.get("expansion_recommendation", pd.Series(dtype=str)).astype(str)
     == "eligible_after_semantic_count_constraints").sum()
) if len(a1_audit_df) else 0

a1_rate_recompute_n = int(
    (a1_audit_df.get("expansion_recommendation", pd.Series(dtype=str)).astype(str)
     == "recompute_or_derive_not_direct_repair").sum()
) if len(a1_audit_df) else 0

a1_cardinality_ready_n = int(
    (a1_audit_df.get("expansion_recommendation", pd.Series(dtype=str)).astype(str)
     == "eligible_only_with_cardinality_constraints").sum()
) if len(a1_audit_df) else 0

router_r0_future_n = int(
    (router_audit_df.get("expansion_recommendation", pd.Series(dtype=str)).astype(str)
     == "router_raw_candidate_for_future_R0_branch").sum()
) if len(router_audit_df) else 0

summary_rows.append({
    "group": "next_step_counts",
    "rows": int(
        a1_semantic_ready_n
        + a1_rate_recompute_n
        + a1_cardinality_ready_n
        + router_r0_future_n
    ),
    "unique_protocol_cols": np.nan,
    "unique_anchors": np.nan,
    "subtier_counts": {
        "A1_semantic_count_candidates": a1_semantic_ready_n,
        "A1_rate_recompute_candidates": a1_rate_recompute_n,
        "A1_cardinality_candidates": a1_cardinality_ready_n,
        "Router_R0_future_candidates": router_r0_future_n,
    },
    "recommendation_counts": {},
    "evidence_strength_counts": {},
    "mean_replication_score": np.nan,
    "mean_z_immediate_core": np.nan,
    "mean_z_sharpness_core": np.nan,
    "mean_coverage_core": np.nan,
})

summary_df = pd.DataFrame(summary_rows)

# ----------------------------------------------------------
# 10) Save outputs
# ----------------------------------------------------------
a1_audit_df.to_csv(a1_audit_csv, index=False)
router_audit_df.to_csv(router_audit_csv, index=False)
constraints_df.to_csv(constraints_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

contract = {
    "cell": "14.8",
    "version": CELL148_VERSION,
    "role": "A1_zigbee_and_router_evidence_audit_before_expansion",
    "quality_dimension": "Q4_cross_modal_consistency_expansion_safety_audit",
    "upstream_contract_versions": {
        "cell13_4": version134_148,
        "cell14_5": version145_148,
        "cell14_7_no_promotion": version147_148
    },
    "q4_no_promotion_status_carried_forward": {
        "final_q4_status": q4_final_status_148,
        "A0_candidate_status": q4_no_promotion_summary_148.get("governance_decision", {}).get("A0_candidate_status", ""),
        "A0_promoted": bool(q4_no_promotion_summary_148.get("governance_decision", {}).get("A0_promoted", False)),
        "fatal_ids": q4_no_promotion_summary_148.get("fatal_ids", []),
        "publication_blocker_n": int(q4_no_promotion_summary_148.get("publication_blocker_n", 0) or 0)
    },
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_148,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_148),
        "pair_pass_rate": broad_q4_pair_pass_rate_148
    },
    "expansion_authorization": {
        "direct_repair_authorized_here": False,
        "A1_authorized_for_materialization": False,
        "router_authorized_for_materialization": False,
        "purpose": "audit-only evidence review before any predeclared future branch"
    },
    "input_counts": {
        "cell14_4_replication_rows": int(len(rep)),
        "cell14_5_all_policy_rows": int(len(all_policy)),
        "A1_zigbee_extended_rows": int(len(a1)),
        "B_router_review_rows": int(len(router_review)),
        "diagnostic_only_rows": int(len(diagnostic)),
    },
    "audit_counts": {
        "A1_zigbee_audit_rows": int(len(a1_audit_df)),
        "router_audit_rows": int(len(router_audit_df)),
        "constraints_rows": int(len(constraints_df)),
        "A1_semantic_count_candidates": int(a1_semantic_ready_n),
        "A1_rate_recompute_candidates": int(a1_rate_recompute_n),
        "A1_cardinality_candidates": int(a1_cardinality_ready_n),
        "Router_R0_future_candidates": int(router_r0_future_n),
    },
    "summary": summary_df.to_dict("records"),
    "recommended_next_steps": [
        {
            "step": "A1a_zigbee_semantic_counts",
            "recommendation": (
                "Consider a separate semantic-count branch for zigbee__cmd, zigbee__ack, and zigbee__data, "
                "with count-domain and pkt_total consistency constraints."
            ),
            "eligible_now": bool(a1_semantic_ready_n > 0),
        },
        {
            "step": "A1b_zigbee_rates",
            "recommendation": (
                "Do not directly repair rate columns such as zigbee__cmd_rate or zigbee__pps. "
                "Recompute or derive them from repaired count features where possible."
            ),
            "eligible_now": False,
        },
        {
            "step": "A1c_zigbee_cardinality",
            "recommendation": (
                "Only repair cardinality features after implementing support projection and upper-bound constraints "
                "against packet/event counts."
            ),
            "eligible_now": False,
        },
        {
            "step": "Router_R0_raw_review",
            "recommendation": (
                "Router evidence should remain review-only until a separate router-specific branch handles "
                "confounding, primitive-feature selection, and non-target drift audits."
            ),
            "eligible_now": False,
        },
    ],
    "strict_contract": {
        "trainval_evidence_used": True,
        "q4_no_promotion_status_carried_forward": True,
        "expansion_not_authorized_here": True,
        "direct_repair_authorized_here": False,
        "TEST_real_values_used": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "audit_only": True,
    },
    "outputs": {
        "a1_audit_csv": a1_audit_csv,
        "router_audit_csv": router_audit_csv,
        "constraints_csv": constraints_csv,
        "summary_csv": summary_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_148(contract_json, contract)
_write_json_148(contract_canonical_json, contract)

manifest = {
    "cell": "14.8",
    "version": CELL148_VERSION,
    "created_outputs": contract["outputs"],
    "audit_counts": contract["audit_counts"],
    "q4_no_promotion_status_carried_forward": contract["q4_no_promotion_status_carried_forward"],
    "expansion_authorization": contract["expansion_authorization"],
    "recommended_next_steps": contract["recommended_next_steps"],
    "strict_contract": contract["strict_contract"],
}

_write_json_148(manifest_json, manifest)

hashes = {
    "a1_audit_csv_sha256": _sha256_file_148(a1_audit_csv),
    "router_audit_csv_sha256": _sha256_file_148(router_audit_csv),
    "constraints_csv_sha256": _sha256_file_148(constraints_csv),
    "summary_csv_sha256": _sha256_file_148(summary_csv),
    "contract_json_sha256": _sha256_file_148(contract_json),
    "contract_canonical_json_sha256": _sha256_file_148(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_148(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_148(contract_json, contract)
_write_json_148(contract_canonical_json, contract)
_write_json_148(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL148_VERSION"] = CELL148_VERSION
globals()["CELL14_8_A1_ZIGBEE_EXPANSION_AUDIT_DF"] = a1_audit_df
globals()["CELL14_8_ROUTER_EVIDENCE_AUDIT_DF"] = router_audit_df
globals()["CELL14_8_PROTOCOL_CONSISTENCY_CONSTRAINTS_DF"] = constraints_df
globals()["CELL14_8_EXPANSION_POLICY_SUMMARY_DF"] = summary_df
globals()["CELL14_8_A1_ROUTER_AUDIT_CONTRACT"] = contract

globals()["CELL14_8_A1_ZIGBEE_EXPANSION_AUDIT_CSV"] = a1_audit_csv
globals()["CELL14_8_ROUTER_EVIDENCE_AUDIT_CSV"] = router_audit_csv
globals()["CELL14_8_PROTOCOL_CONSISTENCY_CONSTRAINTS_CSV"] = constraints_csv
globals()["CELL14_8_EXPANSION_POLICY_SUMMARY_CSV"] = summary_csv
globals()["CELL14_8_A1_ROUTER_AUDIT_CONTRACT_JSON"] = contract_json
globals()["CELL14_8_A1_ROUTER_AUDIT_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL14_8_A1_ROUTER_AUDIT_MANIFEST_JSON"] = manifest_json

log(
    "[Cell14.8] A1/router expansion audit complete | "
    f"A1_rows={len(a1_audit_df)} | "
    f"router_rows={len(router_audit_df)} | "
    f"constraints={len(constraints_df)} | "
    f"A1_semantic_count_candidates={a1_semantic_ready_n} | "
    f"A1_rate_recompute_candidates={a1_rate_recompute_n} | "
    f"A1_cardinality_candidates={a1_cardinality_ready_n} | "
    f"Router_R0_future_candidates={router_r0_future_n}"
)
log(
    "[Cell14.8] A1 subtier counts | "
    f"{a1_audit_df['expansion_subtier'].astype(str).value_counts().sort_index().to_dict() if len(a1_audit_df) else {}}"
)
log(
    "[Cell14.8] Router subtier counts | "
    f"{router_audit_df['expansion_subtier'].astype(str).value_counts().sort_index().to_dict() if len(router_audit_df) else {}}"
)
log(f"[Cell14.8] Saved A1 audit: {a1_audit_csv} | rows={len(a1_audit_df)}")
log(f"[Cell14.8] Saved router audit: {router_audit_csv} | rows={len(router_audit_df)}")
log(f"[Cell14.8] Saved constraints: {constraints_csv} | rows={len(constraints_df)}")
log(f"[Cell14.8] Saved contract: {contract_json}")
log(f"[Cell14.8] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell14.8] Q4 governance carried forward | "
    f"final_q4_status={q4_final_status_148} | "
    f"A0_promoted={bool(q4_no_promotion_summary_148.get('governance_decision', {}).get('A0_promoted', False))} | "
    f"broad_q4_status={broad_q4_status_148} | "
    f"broad_pair_blocker_n={broad_q4_pair_blocker_n_148}"
)
log(
    "[Cell14.8] Contract flags | "
    "trainval_evidence_used=True | "
    "TEST_real_values_used=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "audit_only=True | expansion_not_authorized_here=True | direct_repair_authorized_here=False"
)
log("--- END: Cell 14.8 - A1 and router evidence audit before expansion (v1.1 Q4-governance-aware strict) ---")

gc.collect()