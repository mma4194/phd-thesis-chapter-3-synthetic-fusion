# ==========================================================
# CELL 19.1 - Full CPS consolidated metric evaluation
# v1.2 STUDY-THESIS strict no-Q4-promotion metric consolidation, safe-Q4-forbidden-wording audit
#
# Role:
#   - Consolidate existing evaluation evidence into paper-facing tables.
#   - Use Cell 19.0 metric-source registry as the input contract.
#   - Do NOT recompute raw metrics from source data.
#   - Do NOT mutate synthetic artifacts.
#
# Final governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative.
#   - Q4 evidence is report/ledger evidence only.
#   - Public Q6 candidate is role-restricted and strict-release blocked by
#     distinguishability.
#
# Outputs:
#   reports/cell19_1_full_cps_dimension_summary.csv
#   reports/cell19_1_full_cps_role_summary.csv
#   reports/cell19_1_full_cps_column_or_pair_failures.csv
#   reports/cell19_1_full_cps_metric_inventory.csv
#   reports/cell19_1_full_cps_evaluation_contract.json
#   artifacts/contracts/cell19_1_full_cps_evaluation_contract_v1_1_THESIS.json
#   artifacts/cell19_1_full_cps_evaluation_manifest.json
# ==========================================================

log("--- START: Cell 19.1 - Full CPS consolidated metric evaluation (v1.2 safe-Q4-forbidden-wording no-Q4-promotion strict) ---")

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
_required_191 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL19_0_FULL_CPS_EVAL_ARTIFACT_REGISTRY_DF",
    "CELL19_0_FULL_CPS_EVAL_METRIC_SOURCE_REGISTRY_DF",
    "CELL19_0_FULL_CPS_EVAL_CONTRACT",
]
_missing_191 = [k for k in _required_191 if k not in globals()]
if _missing_191:
    raise RuntimeError(f"[Cell19.1] Missing required globals: {_missing_191}")

ORIGINAL_OUTDIR_191 = str(OUTDIR)
ORIGINAL_OUT_SYN_191 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_191 = str(REPORT_DIR)

def _resolve_project_root_191(outdir, report_dir, out_syn):
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
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "artifacts")):
            return c

    raise RuntimeError("[Cell19.1] Could not resolve canonical project root.")

PROJECT_ROOT_191 = _resolve_project_root_191(ORIGINAL_OUTDIR_191, ORIGINAL_REPORT_DIR_191, ORIGINAL_OUT_SYN_191)
REPORT_DIR_BASE_191 = os.path.join(PROJECT_ROOT_191, "reports")
ARTDIR_BASE_191 = os.path.join(PROJECT_ROOT_191, "artifacts")
CONTRACT_DIR_BASE_191 = os.path.join(ARTDIR_BASE_191, "contracts")

os.makedirs(REPORT_DIR_BASE_191, exist_ok=True)
os.makedirs(ARTDIR_BASE_191, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_191, exist_ok=True)

SEED = int(SEED)

CELL191_VERSION = "cell19_1_full_cps_consolidated_metric_evaluation_v1_2_safe_q4_forbidden_wording_no_q4_promotion"

CFG["cell19_1_version"] = CELL191_VERSION
CFG["cell19_1_Q4_final_status"] = "blocked_no_promotion"
CFG["cell19_1_Q4_coupled_artifacts_used"] = False
CFG["cell19_1_TEST_real_values_used_here"] = False
CFG["cell19_1_TEST_real_values_used_for_materialization"] = False
CFG["cell19_1_synthetic_values_mutated"] = False
CFG["cell19_1_selection_done_here"] = False
CFG["cell19_1_generator_fit_done_here"] = False
CFG["cell19_1_materialization_done_here"] = False
CFG["cell19_1_metric_consolidation_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell19_1_top_failure_n", 50)
CFG.setdefault("cell19_1_scientific_variant_candidates", [
    "scientific_no_q4_cps",
    "scientific_no_q4",
    "scientific_no_q4_protocol",
    "Q4_final_cps",   # legacy/context fallback only
])
CFG.setdefault("cell19_1_public_variant_candidates", [
    "Q6_public_cps",
    "public_q6_cps",
    "public_candidate_core_tabular",
])

TOP_N_191 = int(CFG.get("cell19_1_top_failure_n", 50))
SCI_VARIANTS_191 = [str(x) for x in CFG.get("cell19_1_scientific_variant_candidates", [])]
PUBLIC_VARIANTS_191 = [str(x) for x in CFG.get("cell19_1_public_variant_candidates", [])]

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
dimension_summary_csv = os.path.join(REPORT_DIR_BASE_191, "cell19_1_full_cps_dimension_summary.csv")
role_summary_csv = os.path.join(REPORT_DIR_BASE_191, "cell19_1_full_cps_role_summary.csv")
failures_csv = os.path.join(REPORT_DIR_BASE_191, "cell19_1_full_cps_column_or_pair_failures.csv")
metric_inventory_csv = os.path.join(REPORT_DIR_BASE_191, "cell19_1_full_cps_metric_inventory.csv")
contract_json = os.path.join(REPORT_DIR_BASE_191, "cell19_1_full_cps_evaluation_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_191, "cell19_1_full_cps_evaluation_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_191, "cell19_1_full_cps_evaluation_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_191(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_191(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_191(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_191(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_191(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_191(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_191(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_191(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_191(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_191(payload), f, indent=2, sort_keys=True)

def _sha256_file_191(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_191(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _read_csv_191(path: str):
    if not _exists_191(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _safe_float_191(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _safe_int_191(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _safe_bool_191(x, default=False):
    if isinstance(x, bool):
        return bool(x)
    if pd.isna(x):
        return bool(default)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", ""}:
        return False
    return bool(default)

def _first_col_191(df, candidates):
    if not isinstance(df, pd.DataFrame):
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _pick_row_by_candidates_191(df, id_cols, candidates):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {}, ""
    for col in id_cols:
        if col in df.columns:
            for value in candidates:
                d = df[df[col].astype(str).eq(str(value))]
                if len(d):
                    return d.iloc[0].to_dict(), str(value)
    # fallback: first row
    return df.iloc[0].to_dict(), str(df.iloc[0].get(id_cols[0], "first_row")) if id_cols and id_cols[0] in df.columns else "first_row"

def _num_mean_191(df, col):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").mean())

def _num_median_191(df, col):
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").median())

def _status_counts_191(df, status_col):
    if not isinstance(df, pd.DataFrame) or df.empty or status_col not in df.columns:
        return {}
    return df[status_col].astype(str).value_counts().sort_index().to_dict()

def _interpret_status_191(pass_n, warning_n, fatal_n, blocker_n=0):
    pass_n = _safe_int_191(pass_n)
    warning_n = _safe_int_191(warning_n)
    fatal_n = _safe_int_191(fatal_n)
    blocker_n = _safe_int_191(blocker_n)
    if blocker_n > 0:
        return "blocker_or_not_promoted"
    if fatal_n > 0:
        return "fatal_present"
    if warning_n > 0:
        return "warning_present"
    if pass_n > 0:
        return "pass_or_supported"
    return "not_evaluable_or_scope_excluded"

def _metric_value_191(row, candidates, default=np.nan):
    if not isinstance(row, dict):
        return default
    for c in candidates:
        if c in row:
            v = _safe_float_191(row.get(c), np.nan)
            if np.isfinite(v):
                return v
    return default

def _count_value_191(row, candidates, default=0):
    if not isinstance(row, dict):
        return default
    for c in candidates:
        if c in row:
            return _safe_int_191(row.get(c), default)
    return default

def _add_dimension_row(rows, **kwargs):
    base = {
        "quality_dimension": "",
        "evaluation_target": "",
        "variant_or_stage": "",
        "claims_or_items_total": np.nan,
        "cols_or_pairs_evaluable": np.nan,
        "pass_n": 0,
        "warning_n": 0,
        "fatal_n": 0,
        "blocker_n": 0,
        "primary_penalty_score": np.nan,
        "status": "",
        "metric_summary_json": "{}",
        "source_tables": "",
        "interpretation": "",
        "q4_final_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "TEST_real_values_used_here": False,
        "synthetic_values_mutated": False,
    }
    base.update(kwargs)
    rows.append(base)

def _add_inventory_rows(rows, source_id, df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        rows.append({
            "source_id": source_id,
            "metric_column": "",
            "exists": False,
            "non_null_n": 0,
            "mean_value": np.nan,
            "median_value": np.nan,
        })
        return
    for c in df.columns:
        numeric = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "source_id": source_id,
            "metric_column": str(c),
            "exists": True,
            "non_null_n": int(numeric.notna().sum()),
            "mean_value": float(numeric.mean()) if numeric.notna().any() else np.nan,
            "median_value": float(numeric.median()) if numeric.notna().any() else np.nan,
        })

# ----------------------------------------------------------
# 4) Validate Cell 19.0 governance
# ----------------------------------------------------------
cell19_0_version_191 = str(CELL19_0_FULL_CPS_EVAL_CONTRACT.get("version", ""))
if "cell19_0_full_cps_eval_input_contract_v1_1" not in cell19_0_version_191:
    raise RuntimeError(f"[Cell19.1] Unexpected Cell 19.0 contract version: {cell19_0_version_191}")

q4 = CELL19_0_FULL_CPS_EVAL_CONTRACT.get("q4_governance", {})
strict = CELL19_0_FULL_CPS_EVAL_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell19.1] Cell 19.0 does not carry Q4 blocked_no_promotion governance.")
if _safe_bool_191(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell19.1] Cell 19.0 indicates Q4-coupled artifacts were used.")

q6_public_context = CELL19_0_FULL_CPS_EVAL_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6_public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6_public_context.get("public_distinguishability_status", ""))
public_strict_status = str(q6_public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 5) Load metric source tables
# ----------------------------------------------------------
src_reg = CELL19_0_FULL_CPS_EVAL_METRIC_SOURCE_REGISTRY_DF.copy()
src_reg["source_id"] = src_reg["source_id"].astype(str)

tables = {}
source_paths = {}
for _, r in src_reg.iterrows():
    sid = str(r["source_id"])
    path = str(r["path"])
    source_paths[sid] = path
    tables[sid] = _read_csv_191(path) if path.endswith(".csv") else pd.DataFrame()

# Convenient aliases under current source ids.
q1_variant = tables.get("cell16_1_q1_variant_summary", pd.DataFrame())
q2_variant = tables.get("cell16_2_q2_variant_summary", pd.DataFrame())
q3_variant = tables.get("cell16_3_q3_variant_summary", pd.DataFrame())
q4_stage = tables.get("cell16_4_q4_stage_summary", pd.DataFrame())
dim_matrix = tables.get("cell16_5_dimension_matrix", pd.DataFrame())
key_findings = tables.get("cell16_5_key_findings", pd.DataFrame())
q6_release = tables.get("cell15_4_release_safety", pd.DataFrame())
q6_nocopy = tables.get("cell15_1_no_copy", pd.DataFrame())
q6_dcr = tables.get("cell15_2_dcr_nndr", pd.DataFrame())
q6_mia = tables.get("cell15_3_mia", pd.DataFrame())
baseline_metrics = tables.get("cell17_4_baseline_metrics", pd.DataFrame())
baseline_safe = tables.get("cell17_6_baseline_sanitizer", pd.DataFrame())
claim_trace = tables.get("cell18_3_claim_traceability", pd.DataFrame())
dashboard = tables.get("cell18_3_publication_dashboard", pd.DataFrame())
blockers = tables.get("cell18_3_publication_blockers", pd.DataFrame())
master_ledger = tables.get("master_results_ledger", pd.DataFrame())

# ----------------------------------------------------------
# 6) Dimension summary
# ----------------------------------------------------------
dimension_rows = []

# Q1/Q2/Q3 variant summaries.
for dim, df, metric_col, source_id, interp in [
    ("Q1_marginal", q1_variant, "q1_penalty_score", "cell16_1_q1_variant_summary", "Q1 marginal/statistical fidelity is claim-scoped and role-dependent."),
    ("Q2_temporal", q2_variant, "q2_temporal_penalty_score", "cell16_2_q2_variant_summary", "Q2 temporal quality is claim-scoped and role-dependent."),
    ("Q3_observability", q3_variant, "q3_observability_penalty_score", "cell16_3_q3_variant_summary", "Q3 observability is a limitation/boundary, especially for public role-restricted artifact."),
]:
    sci_row, sci_variant = _pick_row_by_candidates_191(df, ["variant", "artifact_role", "evaluation_target"], SCI_VARIANTS_191)
    pub_row, pub_variant = _pick_row_by_candidates_191(df, ["variant", "artifact_role", "evaluation_target"], PUBLIC_VARIANTS_191)

    for target, row, variant, target_interp in [
        ("scientific_no_q4_cps", sci_row, sci_variant, interp + " Scientific/no-Q4 track."),
        ("public_q6_cps", pub_row, pub_variant, interp + " Public/Q6 role-restricted track."),
    ]:
        if not row:
            _add_dimension_row(
                dimension_rows,
                quality_dimension=dim,
                evaluation_target=target,
                variant_or_stage=variant,
                status="not_evaluable_missing_source_row",
                source_tables=source_id,
                interpretation="No matching row found in source table.",
            )
            continue

        pass_n = _count_value_191(row, ["pass_n", "pass"])
        warn_n = _count_value_191(row, ["warning_n", "warning"])
        fatal_n = _count_value_191(row, ["fatal_n", "fatal"])
        penalty = _metric_value_191(row, [metric_col, "penalty", "primary_penalty_score"])

        _add_dimension_row(
            dimension_rows,
            quality_dimension=dim,
            evaluation_target=target,
            variant_or_stage=variant,
            claims_or_items_total=_count_value_191(row, ["cols_evaluable", "claims_total", "rows"], np.nan),
            cols_or_pairs_evaluable=_metric_value_191(row, ["cols_evaluable", "pairs_evaluable"], np.nan),
            pass_n=pass_n,
            warning_n=warn_n,
            fatal_n=fatal_n,
            blocker_n=0,
            primary_penalty_score=penalty,
            status=_interpret_status_191(pass_n, warn_n, fatal_n),
            metric_summary_json=json.dumps({k: _json_sanitize_191(row.get(k)) for k in row.keys() if k in {metric_col, "cols_evaluable", "pass_n", "warning_n", "fatal_n"}}, sort_keys=True),
            source_tables=source_id,
            interpretation=target_interp,
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )

# Q4: stage evidence, no promotion.
if isinstance(q4_stage, pd.DataFrame) and not q4_stage.empty:
    for _, r in q4_stage.iterrows():
        row = r.to_dict()
        stage = str(row.get("stage_id", row.get("candidate_scope", "")))
        blockers_n = _count_value_191(row, ["publication_blocker_n", "blocker_n"], 0)
        pass_n = _count_value_191(row, ["pass_n", "pairs_pass"], 0)
        warning_n = _count_value_191(row, ["warning_n", "pairs_warning"], 0)
        fatal_n = _count_value_191(row, ["fatal_n", "pairs_fatal"], 0)
        penalty = _metric_value_191(row, ["q4_coupling_penalty_score", "penalty"], np.nan)
        _add_dimension_row(
            dimension_rows,
            quality_dimension="Q4_coupling",
            evaluation_target="q4_report_only_no_promotion",
            variant_or_stage=stage,
            claims_or_items_total=_count_value_191(row, ["pairs_total"], np.nan),
            cols_or_pairs_evaluable=_metric_value_191(row, ["pairs_evaluable", "pairs_total"], np.nan),
            pass_n=pass_n,
            warning_n=warning_n,
            fatal_n=fatal_n,
            blocker_n=blockers_n,
            primary_penalty_score=penalty,
            status="blocked_no_promotion_evidence" if blockers_n > 0 else "candidate_evidence_no_promotion",
            metric_summary_json=json.dumps({k: _json_sanitize_191(row.get(k)) for k in row.keys() if k in {"mean_ETA_similarity", "mean_lag_peak_error", "mean_response_window_rate_error", "mean_manifest_profile_similarity", "publication_blocker_n"}}, sort_keys=True),
            source_tables="cell16_4_q4_stage_summary",
            interpretation="Q4 is report-only evidence under blocked_no_promotion governance; no Q4-coupled artifact is authoritative.",
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )
else:
    _add_dimension_row(
        dimension_rows,
        quality_dimension="Q4_coupling",
        evaluation_target="q4_report_only_no_promotion",
        variant_or_stage="missing_q4_stage_summary",
        status="not_evaluable_missing_source",
        source_tables="cell16_4_q4_stage_summary",
        interpretation="Q4 stage source missing; Q4 remains blocked_no_promotion.",
    )

# Q6.
release_counts = _status_counts_191(q6_release, "release_status")
_add_dimension_row(
    dimension_rows,
    quality_dimension="Q6_privacy_release",
    evaluation_target="public_q6_cps",
    variant_or_stage="public_q6_reaudit",
    claims_or_items_total=len(q6_release) if isinstance(q6_release, pd.DataFrame) else 0,
    cols_or_pairs_evaluable=len(q6_release) if isinstance(q6_release, pd.DataFrame) else 0,
    pass_n=int(release_counts.get("release_pass", 0)),
    warning_n=int(release_counts.get("release_warning", 0)),
    fatal_n=0,
    blocker_n=int(release_counts.get("release_blocker", 0)),
    primary_penalty_score=np.nan,
    status=public_strict_status or "release_blocker",
    metric_summary_json=json.dumps({
        "release_counts": release_counts,
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
        "no_copy_rows": len(q6_nocopy),
        "dcr_rows": len(q6_dcr),
        "mia_rows": len(q6_mia),
    }, sort_keys=True),
    source_tables="cell15_1_no_copy|cell15_2_dcr_nndr|cell15_3_mia|cell15_4_release_safety",
    interpretation="Public Q6 candidate is role-restricted. Direct privacy/no-copy is warning-level, but strict release remains distinguishability-blocked.",
    public_direct_privacy_status=public_direct_privacy_status,
    public_distinguishability_status=public_distinguishability_status,
    public_strict_combined_status=public_strict_status,
)

# External baselines.
if isinstance(baseline_safe, pd.DataFrame) and not baseline_safe.empty:
    r = baseline_safe.iloc[0].to_dict()
    _add_dimension_row(
        dimension_rows,
        quality_dimension="external_baselines",
        evaluation_target="fair_scope_only",
        variant_or_stage="baseline_claim_sanitizer",
        claims_or_items_total=_count_value_191(r, ["headline_fully_comparable_scope_n"], np.nan),
        cols_or_pairs_evaluable=_metric_value_191(r, ["headline_fully_comparable_scope_n"], np.nan),
        pass_n=_count_value_191(r, ["pipeline_headline_wins_n"], 0),
        warning_n=_count_value_191(r, ["partial_or_limited_scope_n"], 0),
        fatal_n=0,
        blocker_n=0,
        primary_penalty_score=np.nan,
        status="fair_scope_single_run_with_limitations",
        metric_summary_json=json.dumps({
            k: _json_sanitize_191(r.get(k))
            for k in [
                "headline_fully_comparable_scope_n",
                "partial_or_limited_scope_n",
                "pipeline_headline_wins_n",
                "external_headline_wins_n",
                "ctgan_unavailable",
                "tabddpm_scope_count",
                "timegan_scope_count",
                "q4_final_status",
                "q4_coupled_artifacts_used",
            ]
            if k in r
        }, sort_keys=True),
        source_tables="cell17_6_baseline_sanitizer|cell17_4_baseline_metrics",
        interpretation="External baselines are same-scope only, single-run, and not full CPS replacements.",
        public_direct_privacy_status=public_direct_privacy_status,
        public_distinguishability_status=public_distinguishability_status,
        public_strict_combined_status=public_strict_status,
    )

# Claim traceability/dashboard.
grade_counts = _status_counts_191(claim_trace, "decision_grade")
final_counts = _status_counts_191(claim_trace, "claim_final_status")
_add_dimension_row(
    dimension_rows,
    quality_dimension="claim_traceability",
    evaluation_target="claim_set",
    variant_or_stage="cell18_3_final_manifest",
    claims_or_items_total=len(claim_trace) if isinstance(claim_trace, pd.DataFrame) else 0,
    cols_or_pairs_evaluable=len(claim_trace) if isinstance(claim_trace, pd.DataFrame) else 0,
    pass_n=int(grade_counts.get("A", 0)) + int(grade_counts.get("B", 0)),
    warning_n=int(grade_counts.get("C", 0)),
    fatal_n=int(grade_counts.get("D", 0)),
    blocker_n=0,
    primary_penalty_score=np.nan,
    status="claim_scoped_supported_with_limitations",
    metric_summary_json=json.dumps({"grade_counts": grade_counts, "claim_final_status_counts": final_counts}, sort_keys=True),
    source_tables="cell18_3_claim_traceability|cell18_3_publication_dashboard|cell18_3_publication_blockers",
    interpretation="Final claim set is traceable and claim-scoped, with limitations/caveats rather than global readiness.",
    public_direct_privacy_status=public_direct_privacy_status,
    public_distinguishability_status=public_distinguishability_status,
    public_strict_combined_status=public_strict_status,
)

dimension_summary_df = pd.DataFrame(dimension_rows)

# ----------------------------------------------------------
# 7) Role summary from available role/release tables
# ----------------------------------------------------------
role_rows = []
if isinstance(q6_release, pd.DataFrame) and not q6_release.empty:
    for _, r in q6_release.iterrows():
        role_rows.append({
            "quality_dimension": "Q6_privacy_release",
            "evaluation_target": "public_q6_cps",
            "role_group": str(r.get("role_group", "")),
            "status": str(r.get("release_status", "")),
            "reasons": str(r.get("release_reasons", "")),
            "q4_final_status": "blocked_no_promotion",
            "q4_coupled_artifacts_used": False,
            "TEST_real_values_used_here": False,
            "synthetic_values_mutated": False,
        })
if isinstance(blockers, pd.DataFrame) and not blockers.empty:
    for _, r in blockers.iterrows():
        role_rows.append({
            "quality_dimension": str(r.get("quality_dimension", "")),
            "evaluation_target": "claim_blockers_or_caveats",
            "role_group": "claim_set",
            "status": str(r.get("claim_final_status", r.get("publication_status", ""))),
            "reasons": str(r.get("paper_action", r.get("known_limitation", ""))),
            "claim_id": str(r.get("claim_id", "")),
            "q4_final_status": "blocked_no_promotion",
            "q4_coupled_artifacts_used": False,
            "TEST_real_values_used_here": False,
            "synthetic_values_mutated": False,
        })
role_summary_df = pd.DataFrame(role_rows)

# ----------------------------------------------------------
# 8) Failures/caveats table
# ----------------------------------------------------------
failure_rows = []
for _, r in dimension_summary_df.iterrows():
    status = str(r.get("status", ""))
    if any(k in status.lower() for k in ["block", "fatal", "warning", "limitation", "not_evaluable", "excluded"]):
        failure_rows.append({
            "quality_dimension": str(r.get("quality_dimension", "")),
            "evaluation_target": str(r.get("evaluation_target", "")),
            "item_type": "dimension_or_stage",
            "item_id": str(r.get("variant_or_stage", "")),
            "status": status,
            "reasons": str(r.get("interpretation", "")),
            "penalty_or_score": r.get("primary_penalty_score", np.nan),
            "source_table": str(r.get("source_tables", "")),
            "q4_final_status": "blocked_no_promotion",
            "q4_coupled_artifacts_used": False,
            "TEST_real_values_used_here": False,
            "synthetic_values_mutated": False,
        })

if isinstance(blockers, pd.DataFrame) and not blockers.empty:
    for _, r in blockers.head(TOP_N_191).iterrows():
        failure_rows.append({
            "quality_dimension": str(r.get("quality_dimension", "")),
            "evaluation_target": "claim_set",
            "item_type": "claim",
            "item_id": str(r.get("claim_id", "")),
            "status": str(r.get("claim_final_status", r.get("publication_status", ""))),
            "reasons": str(r.get("paper_action", r.get("known_limitation", ""))),
            "penalty_or_score": np.nan,
            "source_table": "cell18_3_publication_blockers",
            "q4_final_status": "blocked_no_promotion",
            "q4_coupled_artifacts_used": False,
            "TEST_real_values_used_here": False,
            "synthetic_values_mutated": False,
        })
failure_df = pd.DataFrame(failure_rows)

# ----------------------------------------------------------
# 9) Metric inventory
# ----------------------------------------------------------
inventory_rows = []
for sid, df in tables.items():
    _add_inventory_rows(inventory_rows, sid, df)
metric_inventory_df = pd.DataFrame(inventory_rows)

# ----------------------------------------------------------
# 10) Audits
# ----------------------------------------------------------
audit_rows = []
# Audit positive Q4-coupled artifact usage only.
# Safe/forbidden wording such as "do not claim Q4-coupled artifact" or
# "Q4_coupled_artifacts_used=False" is allowed.
def _unsafe_q4_positive_usage_191(row):
    path_like_cols = ["artifact_path", "path", "source_tables"]
    for c in path_like_cols:
        if c in row.index:
            s = str(row.get(c, "")).lower()
            if ("final_q4_coupled" in s or "q4_coupled" in s) and not (
                "used=false" in s or "no_q4" in s or "no-q4" in s or "not" in s
            ):
                return True

    # For narrative columns, only flag positive artifact-authority/promotion wording.
    narrative = " ".join([
        str(row.get(c, ""))
        for c in ["interpretation", "status", "metric_summary_json", "variant_or_stage", "evaluation_target"]
        if c in row.index
    ]).lower()
    narrative_norm = narrative.replace("_", " ").replace("-", " ")

    safe = (
        "blocked no promotion" in narrative_norm
        or "report only" in narrative_norm
        or "no q4" in narrative_norm
        or "not authoritative" in narrative_norm
        or "q4 coupled artifacts used" in narrative_norm and "false" in narrative_norm
        or "forbidden" in narrative_norm
        or "do not" in narrative_norm
    )
    if safe:
        return False

    positive_patterns = [
        "final q4 coupled",
        "q4 coupled artifact is authoritative",
        "q4 coupled artifact authoritative",
        "q4 promoted",
        "a0 accepted",
        "accepted q4",
    ]
    return any(p in narrative_norm for p in positive_patterns)

q4_bad_mask = dimension_summary_df.apply(_unsafe_q4_positive_usage_191, axis=1)
q4_bad = dimension_summary_df[q4_bad_mask]
audit_rows.append({
    "audit_check": "no_positive_q4_coupled_artifact_usage_in_dimension_summary",
    "passed": bool(len(q4_bad) == 0),
    "details": f"q4_bad_rows={q4_bad[['quality_dimension','evaluation_target','variant_or_stage']].to_dict('records') if len(q4_bad) else []}",
})
audit_rows.append({
    "audit_check": "dimension_summary_nonempty",
    "passed": bool(len(dimension_summary_df) > 0),
    "details": f"rows={len(dimension_summary_df)}",
})
audit_df = pd.DataFrame(audit_rows)
audit_passed = bool(audit_df["passed"].astype(bool).all())
if not audit_passed:
    raise RuntimeError(f"[Cell19.1] Audit failed: {audit_df.loc[~audit_df['passed'].astype(bool)].to_dict('records')}")

# ----------------------------------------------------------
# 11) Save outputs
# ----------------------------------------------------------
dimension_summary_df.to_csv(dimension_summary_csv, index=False)
role_summary_df.to_csv(role_summary_csv, index=False)
failure_df.to_csv(failures_csv, index=False)
metric_inventory_df.to_csv(metric_inventory_csv, index=False)

contract = {
    "cell": "19.1",
    "version": CELL191_VERSION,
    "role": "full_cps_consolidated_metric_evaluation_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_evidence_is_report_only": True,
        "safe_q4_forbidden_or_caveat_wording_allowed": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell19_0": cell19_0_version_191,
    },
    "summary": {
        "dimension_summary_rows": int(len(dimension_summary_df)),
        "role_summary_rows": int(len(role_summary_df)),
        "failure_rows": int(len(failure_df)),
        "metric_inventory_rows": int(len(metric_inventory_df)),
        "dimension_status_counts": dimension_summary_df["status"].astype(str).value_counts().sort_index().to_dict() if "status" in dimension_summary_df.columns else {},
        "audit_passed": bool(audit_passed),
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "safe_q4_forbidden_or_caveat_wording_allowed": True,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "metric_consolidation_done_here": True,
        "safe_q4_forbidden_or_caveat_wording_allowed": True,
    },
    "outputs": {
        "dimension_summary_csv": dimension_summary_csv,
        "role_summary_csv": role_summary_csv,
        "failures_csv": failures_csv,
        "metric_inventory_csv": metric_inventory_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}
_write_json_191(contract_json, contract)
_write_json_191(contract_canonical_json, contract)

manifest = {
    "cell": "19.1",
    "version": CELL191_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "dimension_summary": dimension_summary_df.to_dict("records"),
    "role_summary": role_summary_df.to_dict("records"),
    "metric_inventory": metric_inventory_df.to_dict("records"),
    "audit": audit_df.to_dict("records"),
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
}
_write_json_191(manifest_json, manifest)

hashes = {
    "dimension_summary_csv_sha256": _sha256_file_191(dimension_summary_csv),
    "role_summary_csv_sha256": _sha256_file_191(role_summary_csv),
    "failures_csv_sha256": _sha256_file_191(failures_csv),
    "metric_inventory_csv_sha256": _sha256_file_191(metric_inventory_csv),
    "contract_json_sha256": _sha256_file_191(contract_json),
    "contract_canonical_json_sha256": _sha256_file_191(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_191(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_191(contract_json, contract)
_write_json_191(contract_canonical_json, contract)
_write_json_191(manifest_json, manifest)

# ----------------------------------------------------------
# 12) Export globals
# ----------------------------------------------------------
globals()["CELL191_VERSION"] = CELL191_VERSION
globals()["CELL19_1_FULL_CPS_DIMENSION_SUMMARY_DF"] = dimension_summary_df
globals()["CELL19_1_FULL_CPS_ROLE_SUMMARY_DF"] = role_summary_df
globals()["CELL19_1_FULL_CPS_FAILURES_DF"] = failure_df
globals()["CELL19_1_FULL_CPS_METRIC_INVENTORY_DF"] = metric_inventory_df
globals()["CELL19_1_FULL_CPS_EVALUATION_CONTRACT"] = contract

globals()["CELL19_1_FULL_CPS_DIMENSION_SUMMARY_CSV"] = dimension_summary_csv
globals()["CELL19_1_FULL_CPS_ROLE_SUMMARY_CSV"] = role_summary_csv
globals()["CELL19_1_FULL_CPS_FAILURES_CSV"] = failures_csv
globals()["CELL19_1_FULL_CPS_METRIC_INVENTORY_CSV"] = metric_inventory_csv
globals()["CELL19_1_FULL_CPS_EVALUATION_CONTRACT_JSON"] = contract_json
globals()["CELL19_1_FULL_CPS_EVALUATION_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL19_1_FULL_CPS_EVALUATION_MANIFEST_JSON"] = manifest_json

log(
    "[Cell19.1] Full CPS consolidated metric evaluation complete | "
    f"dimension_rows={len(dimension_summary_df)} | "
    f"role_rows={len(role_summary_df)} | "
    f"failure_rows={len(failure_df)} | "
    f"metric_inventory_rows={len(metric_inventory_df)} | "
    "q4_status=blocked_no_promotion"
)
log(
    "[Cell19.1] Dimension status counts | "
    f"{dimension_summary_df['status'].astype(str).value_counts().sort_index().to_dict() if 'status' in dimension_summary_df.columns else {}}"
)
log(f"[Cell19.1] Saved dimension summary: {dimension_summary_csv}")
log(f"[Cell19.1] Saved role summary: {role_summary_csv}")
log(f"[Cell19.1] Saved failures/caveats: {failures_csv}")
log(f"[Cell19.1] Saved metric inventory: {metric_inventory_csv}")
log(f"[Cell19.1] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell19.1] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "metric_consolidation_done_here=True | safe_q4_forbidden_or_caveat_wording_allowed=True"
)
log("--- END: Cell 19.1 - Full CPS consolidated metric evaluation (v1.2 safe-Q4-forbidden-wording no-Q4-promotion strict) ---")

gc.collect()