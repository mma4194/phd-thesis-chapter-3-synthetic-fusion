# ==========================================================
# CELL 16.5 - A0/A1/A2 final decomposition report
# v1.2 STUDY-THESIS strict no-Q4-promotion final decomposition synthesis
#
# Role:
#   - Consolidate Cells 16.1–16.4 into a final decomposition report.
#   - Explain quality contribution/failure across:
#       Q1 marginal quality
#       Q2 temporal quality
#       Q3 observability quality
#       Q4 cross-modal coupling governance/failure boundary
#       Q6 privacy/release context
#
# Final governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled protocol/CPS artifact is authoritative or promoted.
#   - A0 Zigbee-safe is an improved diagnostic/candidate stage, not a final
#     promoted coupled artifact.
#   - The authoritative scientific artifact is the no-Q4 CPS artifact.
#   - The public artifact is the Q6 role-restricted public candidate.
#
# Critical fixes relative to older 16.5:
#   - Does NOT call the scientific artifact "Q4_final_cps".
#   - Does NOT claim A0 was accepted/promoted.
#   - Separates full no-Q4 scientific artifact, public Q6 direct privacy status,
#     public Q6 distinguishability status, and strict combined public status.
#   - Reads public Q6 policy-split evidence from q6_public_reaudit_v1/reports.
#
# Scientific contract:
#   - No generation.
#   - No selection.
#   - No repair.
#   - No synthetic mutation.
#   - Uses only existing decomposition outputs and already-computed QA evidence.
#
# Outputs:
#   reports/cell16_5_a0_a1_a2_final_decomposition_summary.csv
#   reports/cell16_5_a0_a1_a2_quality_dimension_matrix.csv
#   reports/cell16_5_a0_a1_a2_key_findings.csv
#   reports/cell16_5_a0_a1_a2_publication_report.md
#   reports/cell16_5_a0_a1_a2_final_report_contract.json
#   artifacts/contracts/cell16_5_a0_a1_a2_final_report_contract_v1_2_THESIS.json
#   artifacts/cell16_5_a0_a1_a2_final_report_manifest.json
# ==========================================================

log("--- START: Cell 16.5 - A0/A1/A2 final decomposition report (v1.2 no-Q4-promotion strict) ---")

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
_required_165 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL16_0_DECOMPOSITION_INPUT_CONTRACT",
    "CELL16_1_Q1_VARIANT_SUMMARY_DF",
    "CELL16_1_Q1_ROLE_SUMMARY_DF",
    "CELL16_1_Q1_GAIN_LOSS_LEDGER_DF",
    "CELL16_1_Q1_DECOMPOSITION_CONTRACT",
    "CELL16_2_Q2_VARIANT_SUMMARY_DF",
    "CELL16_2_Q2_ROLE_SUMMARY_DF",
    "CELL16_2_Q2_GAIN_LOSS_LEDGER_DF",
    "CELL16_2_Q2_DECOMPOSITION_CONTRACT",
    "CELL16_3_Q3_VARIANT_SUMMARY_DF",
    "CELL16_3_Q3_ROLE_SUMMARY_DF",
    "CELL16_3_Q3_GAIN_LOSS_LEDGER_DF",
    "CELL16_3_Q3_DECOMPOSITION_CONTRACT",
    "CELL16_4_Q4_STAGE_SUMMARY_DF",
    "CELL16_4_Q4_TIER_SUMMARY_DF",
    "CELL16_4_Q4_GAIN_LOSS_LEDGER_DF",
    "CELL16_4_Q4_DECOMPOSITION_CONTRACT",
]
_missing_165 = [k for k in _required_165 if k not in globals()]
if _missing_165:
    raise RuntimeError(f"[Cell16.5] Missing required globals: {_missing_165}")

ORIGINAL_OUTDIR_165 = str(OUTDIR)
ORIGINAL_OUT_SYN_165 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_165 = str(REPORT_DIR)
SEED = int(SEED)

def _resolve_project_root_165(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        for marker in ["q6_public_reaudit_v1", "q6_public_reaudit"]:
            if marker in parts:
                candidates.append(os.sep.join(parts[:parts.index(marker)]))
        if os.path.basename(p) in {"reports", "synthetic", "artifacts"}:
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

    raise RuntimeError("[Cell16.5] Could not resolve canonical project root.")

PROJECT_ROOT_165 = _resolve_project_root_165(ORIGINAL_OUTDIR_165, ORIGINAL_REPORT_DIR_165, ORIGINAL_OUT_SYN_165)
OUTDIR_BASE_165 = PROJECT_ROOT_165
OUT_SYN_BASE_165 = os.path.join(PROJECT_ROOT_165, "synthetic")
REPORT_DIR_BASE_165 = os.path.join(PROJECT_ROOT_165, "reports")
ARTDIR_BASE_165 = os.path.join(PROJECT_ROOT_165, "artifacts")
CONTRACT_DIR_BASE_165 = os.path.join(ARTDIR_BASE_165, "contracts")

PUBLIC_Q6_ROOT_165 = os.path.join(PROJECT_ROOT_165, "q6_public_reaudit_v1")
PUBLIC_Q6_REPORT_DIR_165 = os.path.join(PUBLIC_Q6_ROOT_165, "reports")
PUBLIC_Q6_CONTRACT_DIR_165 = os.path.join(PUBLIC_Q6_ROOT_165, "artifacts", "contracts")

os.makedirs(REPORT_DIR_BASE_165, exist_ok=True)
os.makedirs(ARTDIR_BASE_165, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_165, exist_ok=True)

CELL165_VERSION = "cell16_5_a0_a1_a2_final_decomposition_report_v1_2_no_q4_promotion"

CFG["cell16_5_version"] = CELL165_VERSION
CFG["cell16_5_Q4_final_status"] = "blocked_no_promotion"
CFG["cell16_5_Q4_coupled_artifacts_used"] = False
CFG["cell16_5_TEST_real_values_used_here"] = False
CFG["cell16_5_TEST_real_values_used_for_materialization"] = False
CFG["cell16_5_synthetic_values_mutated"] = False
CFG["cell16_5_selection_done_here"] = False
CFG["cell16_5_generator_fit_done_here"] = False
CFG["cell16_5_materialization_done_here"] = False
CFG["cell16_5_final_decomposition_report_done_here"] = True

log(
    "[Cell16.5] Report resolver configured | "
    f"PROJECT_ROOT={PROJECT_ROOT_165} | "
    f"BASE_REPORT_DIR={REPORT_DIR_BASE_165} | "
    f"PUBLIC_Q6_REPORT_DIR={PUBLIC_Q6_REPORT_DIR_165}"
)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
final_summary_csv = os.path.join(REPORT_DIR_BASE_165, "cell16_5_a0_a1_a2_final_decomposition_summary.csv")
dimension_matrix_csv = os.path.join(REPORT_DIR_BASE_165, "cell16_5_a0_a1_a2_quality_dimension_matrix.csv")
key_findings_csv = os.path.join(REPORT_DIR_BASE_165, "cell16_5_a0_a1_a2_key_findings.csv")
publication_report_md = os.path.join(REPORT_DIR_BASE_165, "cell16_5_a0_a1_a2_publication_report.md")
contract_json = os.path.join(REPORT_DIR_BASE_165, "cell16_5_a0_a1_a2_final_report_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_165, "cell16_5_a0_a1_a2_final_report_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_165, "cell16_5_a0_a1_a2_final_report_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_165(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_165(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_165(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_165(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_165(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_165(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_165(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_165(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_165(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_165(payload), f, indent=2, sort_keys=True)

def _sha256_file_165(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_165(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _read_csv_optional_165(path: str):
    if not _exists_165(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _read_json_optional_165(path: str):
    if not _exists_165(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _safe_float_165(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_int_165(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _fmt_165(x, digits=4):
    x = _safe_float_165(x, np.nan)
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"

def _row_for_variant_165(df, variant):
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or "variant" not in df.columns:
        return {}
    d = df[df["variant"].astype(str).eq(str(variant))]
    if len(d) == 0:
        return {}
    return d.iloc[0].to_dict()

def _row_for_stage_165(df, stage_id):
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or "stage_id" not in df.columns:
        return {}
    d = df[df["stage_id"].astype(str).eq(str(stage_id))]
    if len(d) == 0:
        return {}
    return d.iloc[0].to_dict()

def _best_row_165(df, score_col):
    if not isinstance(df, pd.DataFrame) or len(df) == 0 or score_col not in df.columns:
        return {}
    d = df.copy()
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
    d = d[np.isfinite(d[score_col])]
    if len(d) == 0:
        return {}
    return d.sort_values(score_col, ascending=True).iloc[0].to_dict()

def _status_from_counts_165(pass_n, warning_n, fatal_n):
    if _safe_int_165(fatal_n) > 0:
        return "blocker"
    if _safe_int_165(warning_n) > 0:
        return "warning"
    if _safe_int_165(pass_n) > 0:
        return "pass"
    return "not_evaluable"

def _append_finding(rows, dimension, severity, finding, evidence, implication, recommended_wording):
    rows.append({
        "dimension": dimension,
        "severity": severity,
        "finding": finding,
        "evidence": evidence,
        "implication": implication,
        "recommended_paper_wording": recommended_wording,
        "TEST_real_values_used_here": False,
        "synthetic_values_mutated": False,
    })

def _contract_governance_165(contract, name, expected_substring):
    if not isinstance(contract, dict):
        raise RuntimeError(f"[Cell16.5] {name} is not a dict.")
    version = str(contract.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(f"[Cell16.5] Unexpected {name} version: {version}")
    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})
    status = str(q4.get("final_q4_status", strict.get("Q4_final_status", "")))
    used = bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True)))
    if status != "blocked_no_promotion":
        raise RuntimeError(f"[Cell16.5] {name} does not carry Q4 blocked_no_promotion governance.")
    if used:
        raise RuntimeError(f"[Cell16.5] {name} indicates Q4-coupled artifacts were used.")
    return version

def _resolve_report_165(filename, context="base"):
    candidates = []
    if context in {"base", "both"}:
        candidates.append(os.path.join(REPORT_DIR_BASE_165, filename))
    if context in {"public", "both"}:
        candidates.append(os.path.join(PUBLIC_Q6_REPORT_DIR_165, filename))
    candidates.extend([
        os.path.join(REPORT_DIR_BASE_165, filename),
        os.path.join(PUBLIC_Q6_REPORT_DIR_165, filename),
        os.path.join(ORIGINAL_REPORT_DIR_165, filename),
    ])
    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]
    for p in candidates:
        if _exists_165(p):
            return p, candidates
    return candidates[0], candidates

def _release_status_from_role_df_165(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return {
            "overall_status": "not_evaluable",
            "release_pass": 0,
            "release_warning": 0,
            "release_blocker": 0,
            "role_rows": 0,
            "status_counts": {},
            "warning_roles": [],
            "blocker_roles": [],
        }

    status_col = None
    for c in ["release_status", "q6_release_status", "status", "publication_status"]:
        if c in df.columns:
            status_col = c
            break

    role_col = "role_group" if "role_group" in df.columns else None

    if status_col is None:
        return {
            "overall_status": "not_evaluable",
            "release_pass": 0,
            "release_warning": 0,
            "release_blocker": 0,
            "role_rows": int(len(df)),
            "status_counts": {},
            "warning_roles": [],
            "blocker_roles": [],
        }

    counts = df[status_col].astype(str).value_counts().to_dict()
    blockers = _safe_int_165(counts.get("release_blocker", counts.get("blocker", 0)), 0)
    warnings = _safe_int_165(counts.get("release_warning", counts.get("warning", 0)), 0)
    passes = _safe_int_165(counts.get("release_pass", counts.get("pass", 0)), 0)

    if blockers > 0:
        overall = "release_blocker"
    elif warnings > 0:
        overall = "release_warning"
    elif passes > 0:
        overall = "release_pass"
    else:
        overall = "not_evaluable"

    warning_roles = []
    blocker_roles = []
    if role_col is not None:
        warning_roles = sorted(df.loc[df[status_col].astype(str).isin(["release_warning", "warning"]), role_col].astype(str).tolist())
        blocker_roles = sorted(df.loc[df[status_col].astype(str).isin(["release_blocker", "blocker"]), role_col].astype(str).tolist())

    return {
        "overall_status": overall,
        "release_pass": int(passes),
        "release_warning": int(warnings),
        "release_blocker": int(blockers),
        "role_rows": int(len(df)),
        "status_counts": counts,
        "warning_roles": warning_roles,
        "blocker_roles": blocker_roles,
    }

def _read_public_policy_split_165():
    role_path, role_searched = _resolve_report_165("cell15_4b_q6_policy_split_role_summary.csv", context="public")
    summary_path, summary_searched = _resolve_report_165("cell15_4b_q6_policy_split_summary.csv", context="public")
    role_df = _read_csv_optional_165(role_path)
    summary_df = _read_csv_optional_165(summary_path)

    metrics = {}
    if isinstance(summary_df, pd.DataFrame) and len(summary_df) and {"metric", "value"}.issubset(set(summary_df.columns)):
        metrics = dict(zip(summary_df["metric"].astype(str), summary_df["value"]))

    return {
        "role_path": role_path,
        "role_searched": role_searched,
        "summary_path": summary_path,
        "summary_searched": summary_searched,
        "role_df": role_df,
        "summary_df": summary_df,
        "metrics": metrics,
        "direct_privacy_overall_status": str(metrics.get("direct_privacy_overall_status", "")),
        "direct_privacy_blocker_roles": str(metrics.get("direct_privacy_blocker_roles", "")),
        "direct_privacy_warning_roles": str(metrics.get("direct_privacy_warning_roles", "")),
        "distinguishability_overall_status": str(metrics.get("distinguishability_overall_status", "")),
        "distinguishability_blocker_roles": str(metrics.get("distinguishability_blocker_roles", "")),
        "distinguishability_warning_roles": str(metrics.get("distinguishability_warning_roles", "")),
        "strict_combined_overall_status": str(metrics.get("strict_combined_overall_status", "")),
    }

# ----------------------------------------------------------
# 3) Validate upstream governance
# ----------------------------------------------------------
CELL16_0_VERSION_165 = _contract_governance_165(CELL16_0_DECOMPOSITION_INPUT_CONTRACT, "CELL16_0_DECOMPOSITION_INPUT_CONTRACT", "cell16_0_a0_a1_a2_decomposition_input_contract_v1_1")
CELL16_1_VERSION_165 = _contract_governance_165(CELL16_1_Q1_DECOMPOSITION_CONTRACT, "CELL16_1_Q1_DECOMPOSITION_CONTRACT", "cell16_1_q1_marginal_decomposition_v1_1")
CELL16_2_VERSION_165 = _contract_governance_165(CELL16_2_Q2_DECOMPOSITION_CONTRACT, "CELL16_2_Q2_DECOMPOSITION_CONTRACT", "cell16_2_q2_temporal_decomposition_v1_1")
CELL16_3_VERSION_165 = _contract_governance_165(CELL16_3_Q3_DECOMPOSITION_CONTRACT, "CELL16_3_Q3_DECOMPOSITION_CONTRACT", "cell16_3_q3_observability_decomposition_v1_1")
CELL16_4_VERSION_165 = _contract_governance_165(CELL16_4_Q4_DECOMPOSITION_CONTRACT, "CELL16_4_Q4_DECOMPOSITION_CONTRACT", "cell16_4_q4_coupling_decomposition_v1_1")

# ----------------------------------------------------------
# 4) Extract key summaries
# ----------------------------------------------------------
q1 = CELL16_1_Q1_VARIANT_SUMMARY_DF.copy()
q2 = CELL16_2_Q2_VARIANT_SUMMARY_DF.copy()
q3 = CELL16_3_Q3_VARIANT_SUMMARY_DF.copy()
q4 = CELL16_4_Q4_STAGE_SUMMARY_DF.copy()

q1_best = _best_row_165(q1, "q1_penalty_score")
q2_best = _best_row_165(q2, "q2_temporal_penalty_score")
q3_best = _best_row_165(q3, "q3_observability_penalty_score")
q4_best = _best_row_165(q4, "q4_coupling_penalty_score")

q1_sci = _row_for_variant_165(q1, "scientific_no_q4_cps")
q1_public = _row_for_variant_165(q1, "Q6_public_cps")
q2_sci = _row_for_variant_165(q2, "scientific_no_q4_cps")
q2_public = _row_for_variant_165(q2, "Q6_public_cps")
q3_sci = _row_for_variant_165(q3, "scientific_no_q4_cps")
q3_public = _row_for_variant_165(q3, "Q6_public_cps")

q4_broad = _row_for_stage_165(q4, "S0_broad_generic_Q4")
q4_initial = _row_for_stage_165(q4, "S2_initial_manifest_zigbee_repair")
q4_a0 = _row_for_stage_165(q4, "S3_A0_zigbee_safe_candidate")
q4_a0_diag = _row_for_stage_165(q4, "S3a_A0_fatal_diagnostic")

# ----------------------------------------------------------
# 5) Q6 evidence: full vs public policy split
# ----------------------------------------------------------
full_q6_path, full_q6_searched = _resolve_report_165("cell15_4_q6_role_release_safety.csv", context="base")
full_q6_df = _read_csv_optional_165(full_q6_path)
full_q6 = _release_status_from_role_df_165(full_q6_df)

public_q6_path, public_q6_searched = _resolve_report_165("cell15_4_q6_role_release_safety.csv", context="public")
public_q6_df = _read_csv_optional_165(public_q6_path)
public_q6 = _release_status_from_role_df_165(public_q6_df)

public_policy = _read_public_policy_split_165()

full_q6_overall_status = full_q6["overall_status"]
public_strict_q6_status = public_policy["strict_combined_overall_status"] or public_q6["overall_status"]
public_direct_privacy_status = public_policy["direct_privacy_overall_status"] or "not_evaluable"
public_distinguishability_status = public_policy["distinguishability_overall_status"] or "not_evaluable"

log(
    "[Cell16.5] Q6 evidence resolved | "
    f"full_q6_status={full_q6_overall_status} | "
    f"public_direct={public_direct_privacy_status} | "
    f"public_distinguishability={public_distinguishability_status} | "
    f"public_strict={public_strict_q6_status}"
)

# ----------------------------------------------------------
# 6) Build quality dimension matrix
# ----------------------------------------------------------
matrix_rows = []

matrix_rows.append({
    "quality_dimension": "Q1_marginal",
    "primary_question": "Do synthetic columns match real TEST marginal distributions?",
    "best_variant_or_stage": str(q1_best.get("variant", "")),
    "best_artifact_role": str(q1_best.get("artifact_role", "")),
    "best_penalty_score": _safe_float_165(q1_best.get("q1_penalty_score"), np.nan),
    "scientific_no_q4_score": _safe_float_165(q1_sci.get("q1_penalty_score"), np.nan),
    "public_candidate_score": _safe_float_165(q1_public.get("q1_penalty_score"), np.nan),
    "scientific_no_q4_status": _status_from_counts_165(q1_sci.get("pass_n"), q1_sci.get("warning_n"), q1_sci.get("fatal_n")),
    "public_candidate_status": _status_from_counts_165(q1_public.get("pass_n"), q1_public.get("warning_n"), q1_public.get("fatal_n")),
    "main_interpretation": "Public role restriction improves Q1 because high-risk/high-mismatch IoT roles are excluded; broad no-Q4 CPS Q1 remains dominated by IoT marginal failures.",
})

matrix_rows.append({
    "quality_dimension": "Q2_temporal",
    "primary_question": "Do synthetic columns preserve temporal dynamics?",
    "best_variant_or_stage": str(q2_best.get("variant", "")),
    "best_artifact_role": str(q2_best.get("artifact_role", "")),
    "best_penalty_score": _safe_float_165(q2_best.get("q2_temporal_penalty_score"), np.nan),
    "scientific_no_q4_score": _safe_float_165(q2_sci.get("q2_temporal_penalty_score"), np.nan),
    "public_candidate_score": _safe_float_165(q2_public.get("q2_temporal_penalty_score"), np.nan),
    "scientific_no_q4_status": _status_from_counts_165(q2_sci.get("pass_n"), q2_sci.get("warning_n"), q2_sci.get("fatal_n")),
    "public_candidate_status": _status_from_counts_165(q2_public.get("pass_n"), q2_public.get("warning_n"), q2_public.get("fatal_n")),
    "main_interpretation": "Public candidate is temporally strongest over its restricted scope; the broad no-Q4 CPS artifact still contains temporal failures across wider IoT roles.",
})

matrix_rows.append({
    "quality_dimension": "Q3_observability",
    "primary_question": "Do synthetic observability/missingness/staleness masks match real behavior?",
    "best_variant_or_stage": str(q3_best.get("variant", "")),
    "best_artifact_role": str(q3_best.get("artifact_role", "")),
    "best_penalty_score": _safe_float_165(q3_best.get("q3_observability_penalty_score"), np.nan),
    "scientific_no_q4_score": _safe_float_165(q3_sci.get("q3_observability_penalty_score"), np.nan),
    "public_candidate_score": _safe_float_165(q3_public.get("q3_observability_penalty_score"), np.nan),
    "scientific_no_q4_status": _status_from_counts_165(q3_sci.get("pass_n"), q3_sci.get("warning_n"), q3_sci.get("fatal_n")),
    "public_candidate_status": "role_restricted_not_evaluated" if _safe_int_165(q3_public.get("cols_evaluable"), 0) == 0 else _status_from_counts_165(q3_public.get("pass_n"), q3_public.get("warning_n"), q3_public.get("fatal_n")),
    "main_interpretation": "Full/no-Q4 scientific artifact retains severe Q3 failures; public candidate intentionally excludes fine-grained observability columns.",
})

matrix_rows.append({
    "quality_dimension": "Q4_coupling",
    "primary_question": "Does IoT-driver timing align with protocol response timing?",
    "best_variant_or_stage": str(q4_best.get("stage_id", "")),
    "best_artifact_role": str(q4_best.get("stage_label", "")),
    "best_penalty_score": _safe_float_165(q4_best.get("q4_coupling_penalty_score"), np.nan),
    "scientific_no_q4_score": _safe_float_165(q4_a0.get("q4_coupling_penalty_score"), np.nan),
    "public_candidate_score": np.nan,
    "scientific_no_q4_status": "blocked_no_promotion",
    "public_candidate_status": "no_Q4_coupled_public_artifact",
    "main_interpretation": "A0 improves coupling evidence but remains blocked/no-promotion; no Q4-coupled artifact is authoritative.",
})

matrix_rows.append({
    "quality_dimension": "Q6_privacy_release",
    "primary_question": "Is the artifact safe for public release under no-copy/privacy audits?",
    "best_variant_or_stage": "Q6_public_cps_policy_split",
    "best_artifact_role": "role_restricted_public_candidate",
    "best_penalty_score": np.nan,
    "scientific_no_q4_score": np.nan,
    "public_candidate_score": np.nan,
    "scientific_no_q4_status": full_q6_overall_status,
    "public_candidate_status": (
        f"direct={public_direct_privacy_status}; "
        f"distinguishability={public_distinguishability_status}; "
        f"strict={public_strict_q6_status}"
    ),
    "main_interpretation": "Public candidate has no direct privacy/no-copy blocker but remains strict-release blocked due to synthetic-real distinguishability.",
})

dimension_matrix_df = pd.DataFrame(matrix_rows)

# ----------------------------------------------------------
# 7) Final summary rows
# ----------------------------------------------------------
summary_rows = [
    {
        "report_item": "scientific_no_q4_artifact",
        "artifact": "CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet",
        "status": "authoritative_scientific_internal_artifact",
        "q1": _status_from_counts_165(q1_sci.get("pass_n"), q1_sci.get("warning_n"), q1_sci.get("fatal_n")),
        "q2": _status_from_counts_165(q2_sci.get("pass_n"), q2_sci.get("warning_n"), q2_sci.get("fatal_n")),
        "q3": _status_from_counts_165(q3_sci.get("pass_n"), q3_sci.get("warning_n"), q3_sci.get("fatal_n")),
        "q4": "blocked_no_promotion",
        "q6": full_q6_overall_status,
        "interpretation": "Authoritative scientific artifact for decomposition; Q4 coupling was not promoted and unrestricted public release is not supported.",
    },
    {
        "report_item": "public_q6_mitigated_artifact",
        "artifact": "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet",
        "status": "role_restricted_public_candidate_with_strict_release_blocker",
        "q1": _status_from_counts_165(q1_public.get("pass_n"), q1_public.get("warning_n"), q1_public.get("fatal_n")),
        "q2": _status_from_counts_165(q2_public.get("pass_n"), q2_public.get("warning_n"), q2_public.get("fatal_n")),
        "q3": "role_restricted_excluded",
        "q4": "no_Q4_coupled_public_artifact",
        "q6": (
            f"direct={public_direct_privacy_status}; "
            f"distinguishability={public_distinguishability_status}; "
            f"strict={public_strict_q6_status}"
        ),
        "interpretation": "Public candidate removes direct no-copy/DCR blockers but remains strict-release blocked by synthetic-real distinguishability.",
    },
    {
        "report_item": "A0_A1_A2_protocol_decomposition",
        "artifact": "A0/A1/A2 protocol artifacts",
        "status": "decomposed",
        "q1": f"best={q1_best.get('variant', 'NA')}",
        "q2": f"best={q2_best.get('variant', 'NA')}",
        "q3": "not_protocol_primary",
        "q4": "A0_improved_but_blocked_no_promotion",
        "q6": "contextual",
        "interpretation": "Protocol decomposition shows metric-specific behavior; no single variant should be claimed universally superior across Q1–Q4.",
    },
]

final_summary_df = pd.DataFrame(summary_rows)

# ----------------------------------------------------------
# 8) Key findings
# ----------------------------------------------------------
findings = []

_append_finding(
    findings,
    "Q1",
    "major",
    "Broad no-Q4 scientific CPS Q1 is dominated by IoT marginal mismatches.",
    f"scientific_no_q4_cps fatal_n={_safe_int_165(q1_sci.get('fatal_n'))}, q1_penalty={_fmt_165(q1_sci.get('q1_penalty_score'))}; Q6_public_cps q1_penalty={_fmt_165(q1_public.get('q1_penalty_score'))}.",
    "The full scientific artifact should not be described as uniformly high marginal fidelity across all IoT roles.",
    "Marginal quality is role-dependent; the public artifact improves Q1 by restricting release to lower-risk roles.",
)

_append_finding(
    findings,
    "Q2",
    "major",
    "Public restricted artifact has the best aggregate temporal score over its restricted role scope.",
    f"Q6_public_cps q2_penalty={_fmt_165(q2_public.get('q2_temporal_penalty_score'))}; scientific_no_q4_cps q2_penalty={_fmt_165(q2_sci.get('q2_temporal_penalty_score'))}.",
    "Temporal quality is strongest after excluding weak OTA and fine-grained IoT state/observability roles.",
    "Temporal realism should be reported separately for full scientific and public restricted artifacts.",
)

_append_finding(
    findings,
    "Q3",
    "major",
    "Full/no-Q4 artifact preserves Q3 observability scope but with substantial failures.",
    f"scientific_no_q4_cps Q3 fatal_n={_safe_int_165(q3_sci.get('fatal_n'))}, penalty={_fmt_165(q3_sci.get('q3_observability_penalty_score'))}.",
    "Fine-grained entity observability/staleness remains a weak point and should not be overclaimed.",
    "Q3 observability is retained for scientific analysis but excluded from the public candidate due to fidelity/privacy risk.",
)

_append_finding(
    findings,
    "Q4",
    "major",
    "A0 Zigbee-safe candidate improved Q4 evidence but was not promoted.",
    f"S0 broad/generic blockers={_safe_int_165(q4_broad.get('publication_blocker_n'))}; A0 candidate blockers={_safe_int_165(q4_a0.get('publication_blocker_n'))}; A0 ETA={_fmt_165(q4_a0.get('mean_ETA_similarity'))}; A0 lag={_fmt_165(q4_a0.get('mean_lag_peak_error'))}.",
    "The Q4 section should report staged improvement and final no-promotion, not an accepted coupled artifact.",
    "Cross-modal consistency is a limitation/boundary: no final Q4-coupled artifact is authoritative.",
)

_append_finding(
    findings,
    "Q6",
    "major",
    "Public Q6 candidate clears direct no-copy/DCR blockers but remains strict-release blocked by distinguishability.",
    (
        f"full_q6_status={full_q6_overall_status}; "
        f"public_direct_privacy={public_direct_privacy_status}; "
        f"public_distinguishability={public_distinguishability_status}; "
        f"public_strict={public_strict_q6_status}."
    ),
    "The public artifact can be described as direct-privacy warning/no-copy blocker-free, but not as fully public-release approved under the strict combined policy.",
    "Report direct privacy/no-copy status separately from synthetic-real distinguishability and do not claim a formal privacy guarantee.",
)

key_findings_df = pd.DataFrame(findings)

# ----------------------------------------------------------
# 9) Publication report markdown
# ----------------------------------------------------------
report_lines = []
report_lines.append("# A0/A1/A2 Decomposition Report")
report_lines.append("")
report_lines.append(f"Version: `{CELL165_VERSION}`")
report_lines.append("")
report_lines.append("## Executive Summary")
report_lines.append("")
report_lines.append(
    "The decomposition shows that quality is dimension-specific and artifact-specific. "
    "The authoritative scientific artifact is the no-Q4 CPS artifact, because final Q4 governance is `blocked_no_promotion`. "
    "The public candidate is a role-restricted Q6 artifact. It clears direct no-copy/DCR blocker evidence but remains strict-release blocked by synthetic-real distinguishability."
)
report_lines.append("")
report_lines.append("### Artifact Tracks")
report_lines.append("")
report_lines.append("| Track | Artifact | Status | Interpretation |")
report_lines.append("|---|---|---|---|")
for _, r in final_summary_df.iterrows():
    report_lines.append(f"| {r['report_item']} | `{r['artifact']}` | {r['status']} | {r['interpretation']} |")

report_lines.append("")
report_lines.append("## Quality Dimension Matrix")
report_lines.append("")
report_lines.append("| Dimension | Best Variant/Stage | Scientific/no-Q4 Status | Public Status | Interpretation |")
report_lines.append("|---|---|---|---|---|")
for _, r in dimension_matrix_df.iterrows():
    report_lines.append(
        f"| {r['quality_dimension']} | {r['best_variant_or_stage']} | "
        f"{r['scientific_no_q4_status']} | {r['public_candidate_status']} | {r['main_interpretation']} |"
    )

report_lines.append("")
report_lines.append("## Q1 Marginal Decomposition")
report_lines.append("")
report_lines.append(
    f"Best Q1 aggregate variant: `{q1_best.get('variant', 'NA')}` with penalty `{_fmt_165(q1_best.get('q1_penalty_score'))}`. "
    f"The no-Q4 scientific CPS artifact has Q1 penalty `{_fmt_165(q1_sci.get('q1_penalty_score'))}`, while the public candidate has Q1 penalty `{_fmt_165(q1_public.get('q1_penalty_score'))}`."
)
report_lines.append("")
report_lines.append("Interpretation: Q1 is role-dependent. The public candidate improves aggregate marginal quality because it excludes roles that were both fidelity-weak and privacy-sensitive.")

report_lines.append("")
report_lines.append("## Q2 Temporal Decomposition")
report_lines.append("")
report_lines.append(
    f"Best Q2 aggregate variant: `{q2_best.get('variant', 'NA')}` with temporal penalty `{_fmt_165(q2_best.get('q2_temporal_penalty_score'))}`. "
    f"The no-Q4 scientific CPS artifact has Q2 penalty `{_fmt_165(q2_sci.get('q2_temporal_penalty_score'))}`, while the public candidate has Q2 penalty `{_fmt_165(q2_public.get('q2_temporal_penalty_score'))}`."
)
report_lines.append("")
report_lines.append("Interpretation: the public candidate is temporally stronger over its restricted scope because it keeps router/Zigbee/event-driver roles and removes weak OTA and fine-grained IoT roles.")

report_lines.append("")
report_lines.append("## Q3 Observability Decomposition")
report_lines.append("")
report_lines.append(
    f"The no-Q4 scientific artifact keeps Q3 observability columns and has `{_safe_int_165(q3_sci.get('fatal_n'))}` Q3 fatal columns. "
    "The public candidate intentionally excludes fine-grained observability columns and should therefore be described as role-restricted, not Q3-complete."
)

report_lines.append("")
report_lines.append("## Q4 Coupling Decomposition")
report_lines.append("")
report_lines.append(
    f"Q4 governance is `blocked_no_promotion`. A0 improved the candidate evidence but was not promoted: "
    f"A0 ETA `{_fmt_165(q4_a0.get('mean_ETA_similarity'))}`, lag `{_fmt_165(q4_a0.get('mean_lag_peak_error'))}`, "
    f"response-window error `{_fmt_165(q4_a0.get('mean_response_window_rate_error'))}`, blockers `{_safe_int_165(q4_a0.get('publication_blocker_n'))}`. "
    "No Q4-coupled protocol/CPS artifact is authoritative."
)
report_lines.append("")
report_lines.append("Interpretation: Q4 should be framed as a boundary and governance result, not as a fully accepted broad coupling repair.")

report_lines.append("")
report_lines.append("## Q6 Privacy and Release Safety")
report_lines.append("")
report_lines.append(
    f"The full/no-Q4 scientific artifact has Q6 status `{full_q6_overall_status}`. "
    f"The public Q6-mitigated candidate has direct privacy/no-copy status `{public_direct_privacy_status}`, "
    f"synthetic-real distinguishability status `{public_distinguishability_status}`, and strict combined status `{public_strict_q6_status}`. "
    "This means the public candidate is not blocked by direct copy/DCR evidence, but it remains strict-release blocked by distinguishability."
)

report_lines.append("")
report_lines.append("## Key Findings")
report_lines.append("")
for _, r in key_findings_df.iterrows():
    report_lines.append(f"- **{r['dimension']} / {r['severity']}**: {r['finding']} {r['evidence']}")

report_lines.append("")
report_lines.append("## Recommended Paper Wording")
report_lines.append("")
report_lines.append(
    "We report two non-equivalent artifacts: an internal no-Q4 scientific artifact for full-namespace analysis and a role-restricted public candidate for release-governed evaluation. "
    "Q4 coupling is not promoted to a final artifact; instead, it is reported as a staged governance boundary with no authoritative Q4-coupled release. "
    "Q6 is also split: the public candidate clears direct no-copy/DCR blocker evidence but remains strict-release blocked under synthetic-real distinguishability. "
    "Consequently, all claims are dimension-scoped and release-scope-specific."
)

publication_report = "\n".join(report_lines)

# ----------------------------------------------------------
# 10) Save outputs
# ----------------------------------------------------------
final_summary_df.to_csv(final_summary_csv, index=False)
dimension_matrix_df.to_csv(dimension_matrix_csv, index=False)
key_findings_df.to_csv(key_findings_csv, index=False)

with open(publication_report_md, "w", encoding="utf-8") as f:
    f.write(publication_report)

# ----------------------------------------------------------
# 11) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "16.5",
    "version": CELL165_VERSION,
    "role": "a0_a1_a2_final_decomposition_report_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "A0_promoted": False,
        "legacy_q4_artifacts_non_authoritative": True,
    },
    "upstream_contract_versions": {
        "cell16_0": CELL16_0_VERSION_165,
        "cell16_1": CELL16_1_VERSION_165,
        "cell16_2": CELL16_2_VERSION_165,
        "cell16_3": CELL16_3_VERSION_165,
        "cell16_4": CELL16_4_VERSION_165,
    },
    "summary": {
        "final_summary_rows": int(len(final_summary_df)),
        "dimension_matrix_rows": int(len(dimension_matrix_df)),
        "key_findings_rows": int(len(key_findings_df)),
        "full_q6_overall_status": full_q6_overall_status,
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_q6_status,
        "full_q6_release_counts": {
            "release_pass": int(full_q6["release_pass"]),
            "release_warning": int(full_q6["release_warning"]),
            "release_blocker": int(full_q6["release_blocker"]),
        },
        "public_q6_role_counts": {
            "release_pass": int(public_q6["release_pass"]),
            "release_warning": int(public_q6["release_warning"]),
            "release_blocker": int(public_q6["release_blocker"]),
        },
        "q6_evidence_paths": {
            "full_q6_role_path": full_q6_path,
            "full_q6_role_searched": full_q6_searched,
            "public_q6_role_path": public_q6_path,
            "public_q6_role_searched": public_q6_searched,
            "public_policy_role_path": public_policy["role_path"],
            "public_policy_role_searched": public_policy["role_searched"],
            "public_policy_summary_path": public_policy["summary_path"],
            "public_policy_summary_searched": public_policy["summary_searched"],
        },
        "best_q1_variant": q1_best,
        "best_q2_variant": q2_best,
        "best_q3_variant": q3_best,
        "best_q4_stage": q4_best,
    },
    "core_conclusion": {
        "scientific_artifact": "The authoritative scientific artifact is no-Q4; Q4 remains blocked/no-promotion.",
        "public_artifact": "The public Q6 candidate clears direct no-copy/DCR blockers but remains strict-release blocked by distinguishability.",
        "decomposition_value": "Quality contribution is dimension-specific; global aggregate claims would be misleading.",
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "final_decomposition_report_done_here": True,
    },
    "outputs": {
        "final_summary_csv": final_summary_csv,
        "dimension_matrix_csv": dimension_matrix_csv,
        "key_findings_csv": key_findings_csv,
        "publication_report_md": publication_report_md,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_165(contract_json, contract)
_write_json_165(contract_canonical_json, contract)

manifest = {
    "cell": "16.5",
    "version": CELL165_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "summary": contract["summary"],
    "core_conclusion": contract["core_conclusion"],
    "strict_contract": contract["strict_contract"],
}
_write_json_165(manifest_json, manifest)

hashes = {
    "final_summary_csv_sha256": _sha256_file_165(final_summary_csv),
    "dimension_matrix_csv_sha256": _sha256_file_165(dimension_matrix_csv),
    "key_findings_csv_sha256": _sha256_file_165(key_findings_csv),
    "publication_report_md_sha256": _sha256_file_165(publication_report_md),
    "contract_json_sha256": _sha256_file_165(contract_json),
    "contract_canonical_json_sha256": _sha256_file_165(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_165(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_165(contract_json, contract)
_write_json_165(contract_canonical_json, contract)
_write_json_165(manifest_json, manifest)

# ----------------------------------------------------------
# 12) Export globals
# ----------------------------------------------------------
globals()["CELL165_VERSION"] = CELL165_VERSION
globals()["CELL16_5_FINAL_SUMMARY_DF"] = final_summary_df
globals()["CELL16_5_QUALITY_DIMENSION_MATRIX_DF"] = dimension_matrix_df
globals()["CELL16_5_KEY_FINDINGS_DF"] = key_findings_df
globals()["CELL16_5_PUBLICATION_REPORT_MD_TEXT"] = publication_report
globals()["CELL16_5_FINAL_DECOMPOSITION_CONTRACT"] = contract

globals()["CELL16_5_FINAL_SUMMARY_CSV"] = final_summary_csv
globals()["CELL16_5_QUALITY_DIMENSION_MATRIX_CSV"] = dimension_matrix_csv
globals()["CELL16_5_KEY_FINDINGS_CSV"] = key_findings_csv
globals()["CELL16_5_PUBLICATION_REPORT_MD"] = publication_report_md
globals()["CELL16_5_CONTRACT_JSON"] = contract_json
globals()["CELL16_5_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL16_5_MANIFEST_JSON"] = manifest_json

log(
    "[Cell16.5] Final A0/A1/A2 decomposition report complete | "
    f"summary_rows={len(final_summary_df)} | "
    f"dimension_rows={len(dimension_matrix_df)} | "
    f"findings={len(key_findings_df)} | "
    f"q4_status=blocked_no_promotion | "
    f"full_q6_status={full_q6_overall_status} | "
    f"public_direct={public_direct_privacy_status} | "
    f"public_distinguishability={public_distinguishability_status} | "
    f"public_strict={public_strict_q6_status}"
)
log(
    "[Cell16.5] Core conclusion | "
    "scientific_artifact=no-Q4 internal scientific artifact | "
    "public_artifact=direct-privacy-warning but strict-distinguishability-blocked | "
    "Q4=no-promotion | "
    "quality_claims=dimension-specific"
)
log(f"[Cell16.5] Saved final summary: {final_summary_csv}")
log(f"[Cell16.5] Saved dimension matrix: {dimension_matrix_csv}")
log(f"[Cell16.5] Saved key findings: {key_findings_csv}")
log(f"[Cell16.5] Saved publication report: {publication_report_md}")
log(f"[Cell16.5] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell16.5] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "final_decomposition_report_done_here=True"
)
log("--- END: Cell 16.5 - A0/A1/A2 final decomposition report (v1.2 no-Q4-promotion strict) ---")

gc.collect()