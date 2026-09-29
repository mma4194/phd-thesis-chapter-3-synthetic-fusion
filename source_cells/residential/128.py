# ==========================================================
# CELL 14.11 - Final Q4 evidence table and paper-ready no-promotion scope audit
# v1.1 STUDY-THESIS strict final Q4 blocked-scope evidence consolidation
#
# Role:
#   - Consolidate the full Q4/coupling evidence trail into paper-ready tables.
#   - Compare:
#       1) broad generic all-pair Q4 baseline from Cell 13.4
#       2) manifest-informed Zigbee pre-repair QA from Cell 13.6 if available
#       3) 5-pair/initial Zigbee repair from Cell 14.3 if available
#       4) A0 Zigbee-safe candidate from Cell 14.6/14.7
#       5) A1/router audit from Cell 14.8
#       6) A1a deferral from Cell 14.9
#       7) final no-promotion ledger from Cell 14.10
#   - Verify final Q4 state is blocked_no_promotion.
#   - Produce a paper-ready Q4 scope statement.
#
# Scientific contract:
#   - No fitting.
#   - No generator selection.
#   - No materialization.
#   - No synthetic mutation.
#   - No TEST real values used here.
#   - No artifact promotion/copy.
# ==========================================================

log("--- START: Cell 14.11 - Final Q4 evidence table and paper-ready no-promotion scope audit (v1.1 strict) ---")

import os
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_1411 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "CELL14_7_Q4_NO_PROMOTION_CONTRACT",
    "CELL14_8_A1_ROUTER_AUDIT_CONTRACT",
    "CELL14_9_A1A_DEFERRAL_CONTRACT",
    "CELL14_10_FINAL_Q4_DECISION_CONTRACT",
]
_missing_1411 = [k for k in _required_1411 if k not in globals()]
if _missing_1411:
    raise RuntimeError(f"[Cell14.11] Missing required globals from prior cells: {_missing_1411}")

OUTDIR = str(OUTDIR)
OUT_SYN_ACTIVE_1411 = str(OUT_SYN)
REPORT_DIR_ACTIVE_1411 = str(REPORT_DIR)

def _resolve_project_root_1411(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        if "q6_public_reaudit" in parts:
            candidates.append(os.sep.join(parts[:parts.index("q6_public_reaudit")]))
        elif os.path.basename(p) in {"reports", "synthetic", "artifacts"}:
            candidates.append(os.path.dirname(p))
        else:
            candidates.append(p)

    _env_project_root = os.environ.get("CPS_CANONICAL_PROJECT_ROOT", "").strip()
    if _env_project_root:
        candidates.append(_env_project_root)

    seen = set()
    for c in candidates:
        c = os.path.abspath(str(c))
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "synthetic")):
            return c

    raise RuntimeError("[Cell14.11] Could not resolve canonical project root.")

PROJECT_ROOT_1411 = _resolve_project_root_1411(OUTDIR, REPORT_DIR_ACTIVE_1411, OUT_SYN_ACTIVE_1411)
REPORT_DIR_1411 = os.path.join(PROJECT_ROOT_1411, "reports")
ARTDIR_1411 = os.path.join(PROJECT_ROOT_1411, "artifacts")
CONTRACT_DIR_1411 = os.path.join(ARTDIR_1411, "contracts")

os.makedirs(REPORT_DIR_1411, exist_ok=True)
os.makedirs(ARTDIR_1411, exist_ok=True)
os.makedirs(CONTRACT_DIR_1411, exist_ok=True)

SEED = int(SEED)
CELL1411_VERSION = "cell14_11_final_q4_no_promotion_scope_evidence_audit_v1_1"

CFG["cell14_11_version"] = CELL1411_VERSION
CFG["cell14_11_final_q4_status"] = "blocked_no_promotion"
CFG["cell14_11_TEST_real_values_used"] = False
CFG["cell14_11_synthetic_values_mutated"] = False
CFG["cell14_11_selection_done_here"] = False
CFG["cell14_11_generator_fit_done_here"] = False
CFG["cell14_11_materialization_done_here"] = False
CFG["cell14_11_artifact_copy_done_here"] = False
CFG["cell14_11_decision_audit_done_here"] = True

# ----------------------------------------------------------
# 1) Paths
# ----------------------------------------------------------
evidence_table_csv = os.path.join(REPORT_DIR_1411, "cell14_11_final_q4_evidence_table.csv")
scope_decision_csv = os.path.join(REPORT_DIR_1411, "cell14_11_final_q4_scope_decision_table.csv")
deferred_branch_csv = os.path.join(REPORT_DIR_1411, "cell14_11_final_q4_deferred_branch_table.csv")
paper_statement_txt = os.path.join(REPORT_DIR_1411, "cell14_11_final_q4_paper_statement.txt")
contract_json = os.path.join(REPORT_DIR_1411, "cell14_11_final_q4_scope_audit_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_1411, "cell14_11_final_q4_scope_audit_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_1411, "cell14_11_final_q4_scope_audit_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_1411(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_1411(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_1411(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_1411(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_1411(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_1411(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_1411(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_1411(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_1411(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_1411(payload), f, indent=2, sort_keys=True)

def _sha256_file_1411(path, block=1 << 20):
    if not path or not os.path.exists(str(path)):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _exists(path):
    return bool(path and os.path.exists(str(path)))

def _safe_float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return default

def _read_csv(path):
    try:
        if _exists(path):
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def _read_json(path):
    try:
        if _exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return {}
    return {}

def _require_contract_version_1411(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell14.11] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell14.11] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

def _first_col(df, candidates):
    if df is None or df.empty:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if str(name).lower() in lower:
            return lower[str(name).lower()]
    for name in candidates:
        n = str(name).lower()
        for c in df.columns:
            if n in str(c).lower():
                return c
    return None

def _metric_mean(df, col):
    if df is None or df.empty or col not in df.columns:
        return np.nan
    return _safe_float(pd.to_numeric(df[col], errors="coerce").mean(), np.nan)

def _summarize_q4_pair_df(df):
    if df is None or df.empty:
        return {
            "pairs_total": 0, "pass_n": np.nan, "warning_n": np.nan,
            "fatal_n": np.nan, "blocker_n": np.nan,
            "mean_ETA_similarity": np.nan,
            "mean_lag_peak_error": np.nan,
            "mean_response_window_rate_error": np.nan,
            "mean_manifest_profile_similarity": np.nan,
            "mean_cross_corr_similarity": np.nan,
            "mean_cross_corr_mae": np.nan,
        }

    status_col = _first_col(df, [
        "A0_q4_status", "post_manifest_q4_status", "manifest_q4_status",
        "q4_status", "publication_status", "qa_status", "status", "pair_status"
    ])
    blocker_col = _first_col(df, [
        "A0_q4_publication_blocker", "post_manifest_q4_publication_blocker",
        "manifest_q4_publication_blocker", "q4_blocker", "publication_blocker", "is_blocker", "blocker"
    ])

    if status_col:
        status_s = df[status_col].astype(str).str.lower()
        pass_n = int(status_s.eq("pass").sum())
        warning_n = int(status_s.eq("warning").sum())
        fatal_n = int(status_s.isin(["fatal", "blocker"]).sum())
    else:
        pass_n = warning_n = fatal_n = np.nan

    if blocker_col:
        blocker_n = int(df[blocker_col].fillna(False).astype(bool).sum())
    elif status_col:
        blocker_n = int(df[status_col].astype(str).str.lower().isin(["fatal", "blocker"]).sum())
    else:
        blocker_n = np.nan

    eta_col = _first_col(df, ["ETA_similarity", "eta_similarity"])
    lag_col = _first_col(df, ["lag_peak_error", "mean_lag_peak_error"])
    resp_col = _first_col(df, ["response_window_rate_error"])
    prof_col = _first_col(df, ["manifest_window_profile_similarity", "manifest_profile_similarity", "profile_similarity"])
    cc_sim_col = _first_col(df, ["cross_corr_similarity"])
    cc_mae_col = _first_col(df, ["cross_corr_mae"])

    return {
        "pairs_total": int(len(df)),
        "pass_n": pass_n,
        "warning_n": warning_n,
        "fatal_n": fatal_n,
        "blocker_n": blocker_n,
        "mean_ETA_similarity": _metric_mean(df, eta_col) if eta_col else np.nan,
        "mean_lag_peak_error": _metric_mean(df, lag_col) if lag_col else np.nan,
        "mean_response_window_rate_error": _metric_mean(df, resp_col) if resp_col else np.nan,
        "mean_manifest_profile_similarity": _metric_mean(df, prof_col) if prof_col else np.nan,
        "mean_cross_corr_similarity": _metric_mean(df, cc_sim_col) if cc_sim_col else np.nan,
        "mean_cross_corr_mae": _metric_mean(df, cc_mae_col) if cc_mae_col else np.nan,
    }

# ----------------------------------------------------------
# 3) Validate final upstream governance
# ----------------------------------------------------------
version134_1411 = _require_contract_version_1411(
    CELL13_4_Q4_PUBLICATION_SUMMARY,
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
    "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1",
)
version147_1411 = _require_contract_version_1411(
    CELL14_7_Q4_NO_PROMOTION_CONTRACT,
    "CELL14_7_Q4_NO_PROMOTION_CONTRACT",
    "cell14_7_q4_no_promotion_governance_v2_1",
)
version148_1411 = _require_contract_version_1411(
    CELL14_8_A1_ROUTER_AUDIT_CONTRACT,
    "CELL14_8_A1_ROUTER_AUDIT_CONTRACT",
    "cell14_8_A1_router_protocol_consistency_audit_v1_1",
)
version149_1411 = _require_contract_version_1411(
    CELL14_9_A1A_DEFERRAL_CONTRACT,
    "CELL14_9_A1A_DEFERRAL_CONTRACT",
    "cell14_9_A1a_expansion_deferral_v1_1",
)
version1410_1411 = _require_contract_version_1411(
    CELL14_10_FINAL_Q4_DECISION_CONTRACT,
    "CELL14_10_FINAL_Q4_DECISION_CONTRACT",
    "cell14_10_final_q4_no_promotion_decision_ledger_v1_1",
)

strict147 = CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("strict_contract", {})
strict148 = CELL14_8_A1_ROUTER_AUDIT_CONTRACT.get("strict_contract", {})
strict149 = CELL14_9_A1A_DEFERRAL_CONTRACT.get("strict_contract", {})
strict1410 = CELL14_10_FINAL_Q4_DECISION_CONTRACT.get("strict_contract", {})

if bool(CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("accepted", True)):
    raise RuntimeError("[Cell14.11] Cell 14.7 unexpectedly has accepted=True.")
if bool(CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("promoted", True)):
    raise RuntimeError("[Cell14.11] Cell 14.7 unexpectedly has promoted=True.")
if bool(strict147.get("candidate_outputs_promoted_to_final", True)):
    raise RuntimeError("[Cell14.11] Cell 14.7 indicates candidate outputs were promoted.")
if bool(strict148.get("direct_repair_authorized_here", True)):
    raise RuntimeError("[Cell14.11] Cell 14.8 unexpectedly authorized direct repair.")
if not bool(strict149.get("A1a_deferred", False)):
    raise RuntimeError("[Cell14.11] Cell 14.9 did not defer A1a.")
if str(CELL14_10_FINAL_Q4_DECISION_CONTRACT.get("final_decision", {}).get("final_q4_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell14.11] Cell 14.10 final Q4 status is not blocked_no_promotion.")
if bool(strict1410.get("promotion_done_here", True)):
    raise RuntimeError("[Cell14.11] Cell 14.10 indicates promotion occurred.")

q4_134_summary = CELL13_4_Q4_PUBLICATION_SUMMARY
q4_147_summary = CELL14_7_Q4_NO_PROMOTION_CONTRACT.get("summary", {})
q4_governance = q4_147_summary.get("governance_decision", {})
audit_148_counts = CELL14_8_A1_ROUTER_AUDIT_CONTRACT.get("audit_counts", {})
a1a_149_summary = CELL14_9_A1A_DEFERRAL_CONTRACT.get("summary", {})
q4_1410_decision = CELL14_10_FINAL_Q4_DECISION_CONTRACT.get("final_decision", {})

# ----------------------------------------------------------
# 4) Load optional prior metric tables
# ----------------------------------------------------------
pre_manifest_pair_df = globals().get("CELL13_6_MANIFEST_Q4_PAIR_METRICS_DF", None)
if not isinstance(pre_manifest_pair_df, pd.DataFrame):
    pre_manifest_pair_df = _read_csv(os.path.join(REPORT_DIR_1411, "cell13_6_manifest_q4_pair_metrics.csv"))

repair_143_pair_df = globals().get("CELL14_3_MANIFEST_Q4_POST_PAIR_METRICS_DF", None)
if not isinstance(repair_143_pair_df, pd.DataFrame):
    repair_143_pair_df = _read_csv(os.path.join(REPORT_DIR_1411, "cell14_3_manifest_q4_post_pair_metrics.csv"))

a0_pair_df = globals().get("CELL14_6_A0_Q4_PAIR_METRICS_DF", None)
if not isinstance(a0_pair_df, pd.DataFrame):
    a0_pair_df = _read_csv(os.path.join(REPORT_DIR_1411, "cell14_6_A0_candidate_manifest_q4_pair_metrics.csv"))

# ----------------------------------------------------------
# 5) Evidence table
# ----------------------------------------------------------
rows = []

# Broad Q4.
pair_summary_134 = q4_134_summary.get("pair_summary", {})
rows.append({
    "candidate_scope": "generic_all_pair_Q4",
    "cell_sources": "13.0-13.4",
    "protocol_scope": "router/ota/zigbee broad registry",
    "pair_scope": "schema-driven driver-protocol registry",
    "status": "blocked",
    "final_publication_candidate": False,
    "pairs_total": _safe_int(pair_summary_134.get("pairs_total", 624), 624),
    "pairs_pass": _safe_int(pair_summary_134.get("pair_pass_n", 0), 0),
    "pairs_warning": _safe_int(pair_summary_134.get("pair_warning_n", np.nan), -1),
    "pairs_fatal": _safe_int(pair_summary_134.get("pair_blocker_n", 623), 623),
    "publication_blocker_n": _safe_int(pair_summary_134.get("pair_blocker_n", 623), 623),
    "mean_ETA_similarity": _safe_float(q4_134_summary.get("metric_summary", {}).get("mean_ETA_similarity"), np.nan),
    "mean_lag_peak_error": _safe_float(q4_134_summary.get("metric_summary", {}).get("mean_lag_peak_error"), np.nan),
    "mean_response_window_rate_error": _safe_float(q4_134_summary.get("metric_summary", {}).get("mean_response_window_rate_error"), np.nan),
    "mean_manifest_profile_similarity": np.nan,
    "decision": "blocked_as_broad_publication_claim",
    "paper_claim": "Use as diagnostic evidence that broad generic all-pair Q4 is not publication-ready.",
})

# Manifest pre-repair optional.
pre = _summarize_q4_pair_df(pre_manifest_pair_df)
if pre["pairs_total"] > 0:
    rows.append({
        "candidate_scope": "manifest_zigbee_pre_repair",
        "cell_sources": "13.5-13.6",
        "protocol_scope": "manifest-informed Zigbee pairs",
        "pair_scope": "TRAIN/VAL manifest subset",
        "status": "pre_repair_reference",
        "final_publication_candidate": False,
        "pairs_total": pre["pairs_total"],
        "pairs_pass": pre["pass_n"],
        "pairs_warning": pre["warning_n"],
        "pairs_fatal": pre["fatal_n"],
        "publication_blocker_n": pre["blocker_n"],
        "mean_ETA_similarity": pre["mean_ETA_similarity"],
        "mean_lag_peak_error": pre["mean_lag_peak_error"],
        "mean_response_window_rate_error": pre["mean_response_window_rate_error"],
        "mean_manifest_profile_similarity": pre["mean_manifest_profile_similarity"],
        "decision": "reference_only",
        "paper_claim": "Reference subset before repair; not promoted.",
    })

# Initial 14.3 optional.
r143 = _summarize_q4_pair_df(repair_143_pair_df)
if r143["pairs_total"] > 0:
    rows.append({
        "candidate_scope": "initial_manifest_zigbee_repair",
        "cell_sources": "14.0-14.3",
        "protocol_scope": "Zigbee manifest subset",
        "pair_scope": "initial manifest repair",
        "status": "candidate_not_final",
        "final_publication_candidate": False,
        "pairs_total": r143["pairs_total"],
        "pairs_pass": r143["pass_n"],
        "pairs_warning": r143["warning_n"],
        "pairs_fatal": r143["fatal_n"],
        "publication_blocker_n": r143["blocker_n"],
        "mean_ETA_similarity": r143["mean_ETA_similarity"],
        "mean_lag_peak_error": r143["mean_lag_peak_error"],
        "mean_response_window_rate_error": r143["mean_response_window_rate_error"],
        "mean_manifest_profile_similarity": r143["mean_manifest_profile_similarity"],
        "decision": "diagnostic_candidate_not_promoted",
        "paper_claim": "Demonstrates improvement but remains not final.",
    })

# A0 blocked candidate.
a0 = _summarize_q4_pair_df(a0_pair_df)
rows.append({
    "candidate_scope": "A0_zigbee_safe_candidate",
    "cell_sources": "14.4-14.7",
    "protocol_scope": "Zigbee A0 safe subset",
    "pair_scope": "8 A0 manifest-approved pairs",
    "status": "blocked_no_promotion",
    "final_publication_candidate": False,
    "pairs_total": _safe_int(q4_147_summary.get("pairs_total", a0["pairs_total"]), 8),
    "pairs_pass": _safe_int(q4_147_summary.get("q4_pass_n", a0["pass_n"]), 2),
    "pairs_warning": _safe_int(q4_147_summary.get("q4_warning_n", a0["warning_n"]), 4),
    "pairs_fatal": _safe_int(q4_147_summary.get("q4_fatal_n", a0["fatal_n"]), 2),
    "publication_blocker_n": _safe_int(q4_147_summary.get("publication_blocker_n", a0["blocker_n"]), 2),
    "mean_ETA_similarity": _safe_float(q4_147_summary.get("mean_ETA_similarity", a0["mean_ETA_similarity"]), a0["mean_ETA_similarity"]),
    "mean_lag_peak_error": _safe_float(q4_147_summary.get("mean_lag_peak_error", a0["mean_lag_peak_error"]), a0["mean_lag_peak_error"]),
    "mean_response_window_rate_error": _safe_float(q4_147_summary.get("mean_response_window_rate_error", a0["mean_response_window_rate_error"]), a0["mean_response_window_rate_error"]),
    "mean_manifest_profile_similarity": _safe_float(q4_147_summary.get("mean_manifest_profile_similarity", a0["mean_manifest_profile_similarity"]), a0["mean_manifest_profile_similarity"]),
    "decision": "blocked_no_promotion",
    "paper_claim": "Improved but retained blockers; no TEST-outcome pruning accepted.",
})

# A1 and router rows.
rows.append({
    "candidate_scope": "A1a_zigbee_semantic_counts",
    "cell_sources": "14.8-14.9",
    "protocol_scope": "Zigbee semantic-count candidates",
    "pair_scope": "A1a candidates from audit",
    "status": "deferred_no_materialization",
    "final_publication_candidate": False,
    "pairs_total": _safe_int(a1a_149_summary.get("cell14_8_A1_semantic_count_candidates", audit_148_counts.get("A1_semantic_count_candidates", 0)), 0),
    "pairs_pass": np.nan,
    "pairs_warning": np.nan,
    "pairs_fatal": np.nan,
    "publication_blocker_n": np.nan,
    "mean_ETA_similarity": np.nan,
    "mean_lag_peak_error": np.nan,
    "mean_response_window_rate_error": np.nan,
    "mean_manifest_profile_similarity": np.nan,
    "decision": "deferred_no_materialization",
    "paper_claim": "Deferred because no accepted A0 final base exists.",
})
rows.append({
    "candidate_scope": "A1b_zigbee_rates",
    "cell_sources": "14.8-14.10",
    "protocol_scope": "Zigbee rate/recompute candidates",
    "pair_scope": "deferred branch",
    "status": "deferred",
    "final_publication_candidate": False,
    "pairs_total": _safe_int(audit_148_counts.get("A1_rate_recompute_candidates", 0), 0),
    "pairs_pass": np.nan, "pairs_warning": np.nan, "pairs_fatal": np.nan,
    "publication_blocker_n": np.nan,
    "mean_ETA_similarity": np.nan, "mean_lag_peak_error": np.nan,
    "mean_response_window_rate_error": np.nan, "mean_manifest_profile_similarity": np.nan,
    "decision": "deferred",
    "paper_claim": "Deferred; rates should be derived/recomputed, not directly repaired.",
})
rows.append({
    "candidate_scope": "A1c_zigbee_cardinality",
    "cell_sources": "14.8-14.10",
    "protocol_scope": "Zigbee cardinality candidates",
    "pair_scope": "deferred branch",
    "status": "deferred",
    "final_publication_candidate": False,
    "pairs_total": _safe_int(audit_148_counts.get("A1_cardinality_candidates", 0), 0),
    "pairs_pass": np.nan, "pairs_warning": np.nan, "pairs_fatal": np.nan,
    "publication_blocker_n": np.nan,
    "mean_ETA_similarity": np.nan, "mean_lag_peak_error": np.nan,
    "mean_response_window_rate_error": np.nan, "mean_manifest_profile_similarity": np.nan,
    "decision": "deferred",
    "paper_claim": "Deferred; cardinality constraints are required before repair.",
})
rows.append({
    "candidate_scope": "router_coupling",
    "cell_sources": "14.4-14.10",
    "protocol_scope": "router",
    "pair_scope": "audit-only router evidence",
    "status": "deferred",
    "final_publication_candidate": False,
    "pairs_total": _safe_int(audit_148_counts.get("Router_R0_future_candidates", 0), 0),
    "pairs_pass": np.nan, "pairs_warning": np.nan, "pairs_fatal": np.nan,
    "publication_blocker_n": np.nan,
    "mean_ETA_similarity": np.nan, "mean_lag_peak_error": np.nan,
    "mean_response_window_rate_error": np.nan, "mean_manifest_profile_similarity": np.nan,
    "decision": "deferred",
    "paper_claim": "Deferred; no router repair branch authorized.",
})

evidence_df = pd.DataFrame(rows)

# ----------------------------------------------------------
# 6) Scope decision and deferred tables
# ----------------------------------------------------------
scope_df = pd.DataFrame([
    {
        "scope": "final_publication_Q4_status",
        "decision": "blocked_no_promotion",
        "accepted": False,
        "artifact_protocol_path": "",
        "artifact_cps_path": "",
        "claim_allowed": "Q4 is reported as a blocked quality dimension with diagnostic improvement evidence.",
        "claim_not_allowed": "Do not claim final Q4-coupled artifact or broad cross-modal realism.",
    },
    {
        "scope": "A0_zigbee_safe_candidate",
        "decision": "improved_but_blocked",
        "accepted": False,
        "artifact_protocol_path": "",
        "artifact_cps_path": "",
        "claim_allowed": "Report as diagnostic evidence that protocol-specific repair improved some coupling metrics.",
        "claim_not_allowed": "Do not promote or release as final Q4 artifact.",
    },
    {
        "scope": "A1_router_expansion",
        "decision": "deferred",
        "accepted": False,
        "artifact_protocol_path": "",
        "artifact_cps_path": "",
        "claim_allowed": "Report as future work/deferred constrained branches.",
        "claim_not_allowed": "Do not claim A1/router repair was materialized or validated.",
    },
])

deferred_df = pd.DataFrame([
    {
        "branch": "A1a_zigbee_semantic_counts",
        "decision": "deferred_no_materialization",
        "reason": "No accepted A0 final base exists.",
        "needed_before_future_promotion": "Predeclared A0-accepted base plus semantic-to-packet consistency constraints.",
    },
    {
        "branch": "A1b_zigbee_rates",
        "decision": "deferred",
        "reason": "Rate/intensity features should be derived from count features.",
        "needed_before_future_promotion": "Derived-rate recomputation policy tied to accepted count repairs.",
    },
    {
        "branch": "A1c_zigbee_cardinality",
        "decision": "deferred",
        "reason": "Cardinality repair needs support and upper-bound constraints.",
        "needed_before_future_promotion": "TRAIN/VAL support-envelope and upper-bound constraints.",
    },
    {
        "branch": "router_coupling",
        "decision": "deferred",
        "reason": "Router evidence remains broad/confounded and audit-only.",
        "needed_before_future_promotion": "Router-specific branch with confound control and non-target drift audits.",
    },
])

# ----------------------------------------------------------
# 7) Final assertions
# ----------------------------------------------------------
failures = []

if str(q4_1410_decision.get("final_q4_status", "")) != "blocked_no_promotion":
    failures.append("Cell 14.10 final decision is not blocked_no_promotion.")
if str(q4_governance.get("final_q4_status", "")) != "blocked_no_promotion":
    failures.append("Cell 14.7 governance is not blocked_no_promotion.")
if bool(q4_governance.get("A0_promoted", True)):
    failures.append("Cell 14.7 says A0 was promoted.")
if bool(q4_1410_decision.get("A0_promoted", True)):
    failures.append("Cell 14.10 says A0 was promoted.")
if bool(evidence_df["final_publication_candidate"].fillna(False).astype(bool).any()):
    failures.append("Evidence table contains a final publication Q4 candidate even though Q4 is blocked.")

if failures:
    raise RuntimeError("[Cell14.11] Final Q4 no-promotion scope audit failed: " + " | ".join(failures))

# ----------------------------------------------------------
# 8) Paper-ready statement
# ----------------------------------------------------------
def fmt(x):
    return f"{x:.6f}" if np.isfinite(_safe_float(x, np.nan)) else "nan"

a0_row = evidence_df[evidence_df["candidate_scope"].eq("A0_zigbee_safe_candidate")].iloc[0].to_dict()

statement = f"""Final Q4 coupling scope statement

Final Q4 status:
- blocked_no_promotion

No final Q4-coupled protocol or CPS artifact is promoted.

Evidence trail:
The broad schema-driven Q4 registry remained blocked, with {pair_summary_134.get("pair_blocker_n", 623)} blocked driver-protocol pairs. A constrained A0 Zigbee-safe repair branch improved the coupling evidence but retained {a0_row.get("publication_blocker_n", 2)} publication blockers. The A0 candidate evaluated {a0_row.get("pairs_total", 8)} pairs with pass/warning/fatal counts of {a0_row.get("pairs_pass", 2)}/{a0_row.get("pairs_warning", 4)}/{a0_row.get("pairs_fatal", 2)}. Its mean metrics were ETA similarity {fmt(a0_row.get("mean_ETA_similarity"))}, lag peak error {fmt(a0_row.get("mean_lag_peak_error"))}, response-window error {fmt(a0_row.get("mean_response_window_rate_error"))}, and manifest profile similarity {fmt(a0_row.get("mean_manifest_profile_similarity"))}.

Governance decision:
Because the A0 candidate retained fatal blockers, and the follow-up legality diagnostic found no defensible simple TRAIN/VAL-only pruning rule that could remove the fatal rows while retaining a sufficiently large clean subset, A0 was not promoted. A1a semantic-count repair was deferred because no accepted A0 final base exists. A1b rate features, A1c cardinality features, and router coupling remain deferred/audit-only.

Paper claim:
The paper should not claim broad cross-modal coupling realism or a final Q4-coupled release artifact. The defensible claim is that Q4 exposed a hard coupling boundary: protocol-specific Zigbee repair improved selected metrics, but final release governance blocked promotion to avoid TEST-outcome cherry-picking and unsupported protocol repair.
"""

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
evidence_df.to_csv(evidence_table_csv, index=False)
scope_df.to_csv(scope_decision_csv, index=False)
deferred_df.to_csv(deferred_branch_csv, index=False)

with open(paper_statement_txt, "w", encoding="utf-8") as f:
    f.write(statement)

contract = {
    "cell": "14.11",
    "version": CELL1411_VERSION,
    "role": "final_Q4_no_promotion_evidence_table_and_scope_audit",
    "final_q4_status": "blocked_no_promotion",
    "accepted_candidate": "",
    "final_protocol_path": "",
    "final_cps_path": "",
    "generic_all_pair_claim_allowed": False,
    "A0_promoted": False,
    "A1a_promoted": False,
    "A1b_A1c_router_deferred": True,
    "evidence_rows_n": int(len(evidence_df)),
    "scope_decision_rows_n": int(len(scope_df)),
    "deferred_branch_rows_n": int(len(deferred_df)),
    "A0_summary": {
        "pairs_total": _json_sanitize_1411(a0_row.get("pairs_total")),
        "pairs_pass": _json_sanitize_1411(a0_row.get("pairs_pass")),
        "pairs_warning": _json_sanitize_1411(a0_row.get("pairs_warning")),
        "pairs_fatal": _json_sanitize_1411(a0_row.get("pairs_fatal")),
        "publication_blocker_n": _json_sanitize_1411(a0_row.get("publication_blocker_n")),
        "mean_ETA_similarity": _json_sanitize_1411(a0_row.get("mean_ETA_similarity")),
        "mean_lag_peak_error": _json_sanitize_1411(a0_row.get("mean_lag_peak_error")),
        "mean_response_window_rate_error": _json_sanitize_1411(a0_row.get("mean_response_window_rate_error")),
        "mean_manifest_profile_similarity": _json_sanitize_1411(a0_row.get("mean_manifest_profile_similarity")),
    },
    "upstream_contract_versions": {
        "cell13_4": version134_1411,
        "cell14_7_no_promotion": version147_1411,
        "cell14_8_audit": version148_1411,
        "cell14_9_deferral": version149_1411,
        "cell14_10_no_promotion": version1410_1411,
    },
    "scientific_contract": {
        "TEST_real_values_used": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "artifact_copy_done_here": False,
        "decision_audit_only": True,
    },
    "outputs": {
        "evidence_table_csv": evidence_table_csv,
        "scope_decision_csv": scope_decision_csv,
        "deferred_branch_csv": deferred_branch_csv,
        "paper_statement_txt": paper_statement_txt,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_1411(contract_json, contract)
_write_json_1411(contract_canonical_json, contract)

manifest = {
    "cell": "14.11",
    "version": CELL1411_VERSION,
    "contract": contract,
    "outputs": contract["outputs"],
}
_write_json_1411(manifest_json, manifest)

hashes = {
    "evidence_table_csv_sha256": _sha256_file_1411(evidence_table_csv),
    "scope_decision_csv_sha256": _sha256_file_1411(scope_decision_csv),
    "deferred_branch_csv_sha256": _sha256_file_1411(deferred_branch_csv),
    "paper_statement_txt_sha256": _sha256_file_1411(paper_statement_txt),
    "contract_json_sha256": _sha256_file_1411(contract_json),
    "contract_canonical_json_sha256": _sha256_file_1411(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_1411(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_1411(contract_json, contract)
_write_json_1411(contract_canonical_json, contract)
_write_json_1411(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals
# ----------------------------------------------------------
globals()["CELL14_11_VERSION"] = CELL1411_VERSION
globals()["CELL14_11_FINAL_Q4_EVIDENCE_DF"] = evidence_df
globals()["CELL14_11_FINAL_Q4_SCOPE_DECISION_DF"] = scope_df
globals()["CELL14_11_FINAL_Q4_DEFERRED_BRANCH_DF"] = deferred_df
globals()["CELL14_11_FINAL_Q4_PAPER_STATEMENT"] = statement
globals()["CELL14_11_FINAL_Q4_SCOPE_AUDIT_CONTRACT"] = contract
globals()["CELL14_11_FINAL_Q4_STATUS"] = "blocked_no_promotion"
globals()["CELL14_11_FINAL_Q4_EVIDENCE_TABLE_CSV"] = evidence_table_csv
globals()["CELL14_11_FINAL_Q4_SCOPE_DECISION_CSV"] = scope_decision_csv
globals()["CELL14_11_FINAL_Q4_DEFERRED_BRANCH_CSV"] = deferred_branch_csv
globals()["CELL14_11_FINAL_Q4_PAPER_STATEMENT_TXT"] = paper_statement_txt
globals()["CELL14_11_FINAL_Q4_SCOPE_AUDIT_CONTRACT_JSON"] = contract_json
globals()["CELL14_11_FINAL_Q4_SCOPE_AUDIT_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL14_11_FINAL_Q4_SCOPE_AUDIT_MANIFEST_JSON"] = manifest_json

# ----------------------------------------------------------
# 11) Logs / compact output
# ----------------------------------------------------------
log(
    "[Cell14.11] Final Q4 no-promotion evidence audit complete | "
    "final_q4_status=blocked_no_promotion | accepted_candidate=<none> | "
    f"evidence_rows={len(evidence_df)} | A0_blockers={a0_row.get('publication_blocker_n', 'NA')}"
)
log(
    "[Cell14.11] Final Q4 claim | "
    "generic_all_pair_Q4=blocked/diagnostic | A0=improved_but_blocked | "
    "A1a=deferred | A1b/A1c/router=deferred"
)
log(f"[Cell14.11] Saved evidence table: {evidence_table_csv}")
log(f"[Cell14.11] Saved scope decision table: {scope_decision_csv}")
log(f"[Cell14.11] Saved paper statement: {paper_statement_txt}")
log(f"[Cell14.11] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell14.11] Contract flags | TEST_real_values_used=False | synthetic_values_mutated=False | "
    "selection_done_here=False | generator_fit_done_here=False | materialization_done_here=False | "
    "artifact_copy_done_here=False | decision_audit_only=True"
)
log("--- END: Cell 14.11 - Final Q4 evidence table and paper-ready no-promotion scope audit (v1.1 strict) ---")

print("\\n=== CELL 14.11 FINAL Q4 EVIDENCE TABLE ===")
display_cols = [
    "candidate_scope", "status", "final_publication_candidate", "protocol_scope",
    "pairs_total", "pairs_pass", "pairs_warning", "pairs_fatal",
    "publication_blocker_n", "mean_ETA_similarity", "mean_lag_peak_error",
    "mean_response_window_rate_error", "mean_manifest_profile_similarity", "decision"
]
display_cols = [c for c in display_cols if c in evidence_df.columns]
print(evidence_df[display_cols].to_string(index=False, max_colwidth=120))

print("\\n=== CELL 14.11 FINAL Q4 SCOPE DECISION ===")
print(scope_df.to_string(index=False, max_colwidth=120))

print("\\n=== CELL 14.11 PAPER STATEMENT ===")
print(statement)

print("\\nSaved:")
print(evidence_table_csv)
print(scope_decision_csv)
print(deferred_branch_csv)
print(paper_statement_txt)
print(contract_json)
print(contract_canonical_json)
print(manifest_json)