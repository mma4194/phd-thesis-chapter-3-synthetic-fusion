# ==========================================================
# CELL 18.0a - Master results ledger for paper
# v1.2 STUDY-THESIS strict no-Q4-promotion master-ledger builder, safe-negation-aware helper fix
#
# Purpose:
#   Build reports/MASTER_RESULTS_LEDGER_FOR_PAPER.csv from validated
#   pipeline/decomposition/baseline reports so Cell 18.0 can generate a
#   ledger-derived claim registry.
#
# Why this exists:
#   Cell 18.0 intentionally refuses to run without the master ledger because
#   the checklist requires claim registry/readiness/dashboard to be output-only
#   and derived from MASTER_RESULTS_LEDGER_FOR_PAPER.csv, not hand-authored
#   optimism.
#
# Governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative.
#   - Public direct privacy/no-copy = release_warning.
#   - Public distinguishability/strict release = release_blocker.
#
# Scientific contract:
#   - No generation.
#   - No fitting.
#   - No materialization.
#   - No synthetic mutation.
#   - TEST real values are not used here.
#
# Outputs:
#   reports/MASTER_RESULTS_LEDGER_FOR_PAPER.csv
#   reports/cell18_0a_master_results_ledger_audit.csv
#   reports/cell18_0a_master_results_ledger_contract.json
#   artifacts/contracts/cell18_0a_master_results_ledger_contract_v1_0_THESIS.json
#   artifacts/cell18_0a_master_results_ledger_manifest.json
# ==========================================================

log("--- START: Cell 18.0a - Master results ledger for paper (v1.2 safe-negation-helper-fix no-Q4-promotion strict) ---")

import os
import gc
import json
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_180a = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT",
]
_missing_180a = [k for k in _required_180a if k not in globals()]
if _missing_180a:
    raise RuntimeError(f"[Cell18.0a] Missing required globals: {_missing_180a}")

ORIGINAL_OUTDIR_180a = str(OUTDIR)
ORIGINAL_OUT_SYN_180a = str(OUT_SYN)
ORIGINAL_REPORT_DIR_180a = str(REPORT_DIR)

def _resolve_project_root_180a(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell18.0a] Could not resolve canonical project root.")

PROJECT_ROOT_180a = _resolve_project_root_180a(ORIGINAL_OUTDIR_180a, ORIGINAL_REPORT_DIR_180a, ORIGINAL_OUT_SYN_180a)
REPORT_DIR_BASE_180a = os.path.join(PROJECT_ROOT_180a, "reports")
ARTDIR_BASE_180a = os.path.join(PROJECT_ROOT_180a, "artifacts")
CONTRACT_DIR_BASE_180a = os.path.join(ARTDIR_BASE_180a, "contracts")

os.makedirs(REPORT_DIR_BASE_180a, exist_ok=True)
os.makedirs(ARTDIR_BASE_180a, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_180a, exist_ok=True)

SEED = int(SEED)

CELL180A_VERSION = "cell18_0a_master_results_ledger_v1_2_safe_negation_helper_fix_no_q4_promotion"

CFG["cell18_0a_version"] = CELL180A_VERSION
CFG["cell18_0a_Q4_final_status"] = "blocked_no_promotion"
CFG["cell18_0a_Q4_coupled_artifacts_used"] = False
CFG["cell18_0a_TEST_real_values_used_here"] = False
CFG["cell18_0a_TEST_real_values_used_for_materialization"] = False
CFG["cell18_0a_synthetic_values_mutated"] = False
CFG["cell18_0a_selection_done_here"] = False
CFG["cell18_0a_generator_fit_done_here"] = False
CFG["cell18_0a_materialization_done_here"] = False
CFG["cell18_0a_master_ledger_done_here"] = True

# ----------------------------------------------------------
# 1) Paths
# ----------------------------------------------------------
master_ledger_csv = os.path.join(REPORT_DIR_BASE_180a, "MASTER_RESULTS_LEDGER_FOR_PAPER.csv")
audit_csv = os.path.join(REPORT_DIR_BASE_180a, "cell18_0a_master_results_ledger_audit.csv")
contract_json = os.path.join(REPORT_DIR_BASE_180a, "cell18_0a_master_results_ledger_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_180a, "cell18_0a_master_results_ledger_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_180a, "cell18_0a_master_results_ledger_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_180a(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_180a(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_180a(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_180a(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_180a(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_180a(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_180a(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_180a(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_180a(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_180a(payload), f, indent=2, sort_keys=True)

def _sha256_file_180a(path):
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_csv_180a(filename):
    path = os.path.join(REPORT_DIR_BASE_180a, filename)
    if not os.path.exists(path):
        return pd.DataFrame(), path, False
    try:
        return pd.read_csv(path), path, True
    except Exception:
        return pd.DataFrame(), path, False

def _safe_float_180a(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _safe_int_180a(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _add_row(rows, **kw):
    base = {
        "ledger_row_id": f"ML-{len(rows)+1:04d}",
        "source_cell": "",
        "source_file": "",
        "source_row_key": "",
        "quality_dimension": "",
        "claim_scope": "",
        "claim_type": "",
        "claim_text": "",
        "permitted_claim": "",
        "forbidden_claim": "",
        "metric_name": "",
        "metric_value": np.nan,
        "metric_unit": "",
        "artifact_type": "",
        "artifact_path": "",
        "selection_split": "TRAIN_fit_VAL_select_only;TEST_not_used_for_selection",
        "evaluation_split": "TEST_QA_reference_only",
        "TEST_used_for_selection": False,
        "TEST_used_for_repair": False,
        "seed_policy": "single_seed_current_run;multi_seed_CI_not_claimed",
        "known_limitation": "",
        "overclaim_risk": "medium",
        "claim_scoped_status": "pending_traceability",
        "q4_final_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "public_direct_privacy_status": "",
        "public_distinguishability_status": "",
        "public_strict_combined_status": "",
    }
    base.update(kw)
    rows.append(base)

def _fmt(x, digits=4):
    x = _safe_float_180a(x, np.nan)
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"

# ----------------------------------------------------------
# 3) Validate upstream 17.6 governance
# ----------------------------------------------------------
cell17_6_version = str(CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT.get("version", ""))
if "cell17_6_baseline_claim_sanitizer_v1_3" not in cell17_6_version:
    raise RuntimeError(f"[Cell18.0a] Unexpected Cell 17.6 contract version: {cell17_6_version}")

q4 = CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT.get("q4_governance", {})
strict = CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell18.0a] Cell 17.6 does not carry Q4 blocked_no_promotion governance.")
if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell18.0a] Cell 17.6 indicates Q4-coupled artifacts were used.")

q6_public_context = CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6_public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6_public_context.get("public_distinguishability_status", ""))
public_strict_status = str(q6_public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 4) Load source reports
# ----------------------------------------------------------
source_files = {
    "16.1": "cell16_1_q1_marginal_decomposition_by_variant.csv",
    "16.2": "cell16_2_q2_temporal_decomposition_by_variant.csv",
    "16.3": "cell16_3_q3_observability_decomposition_by_variant.csv",
    "16.4": "cell16_4_q4_coupling_decomposition_by_stage.csv",
    "16.5_matrix": "cell16_5_a0_a1_a2_quality_dimension_matrix.csv",
    "16.5_findings": "cell16_5_a0_a1_a2_key_findings.csv",
    "17.6_safe": "cell17_6_baseline_publication_safe_summary.csv",
    "17.6_headline": "cell17_6_baseline_headline_comparable_results.csv",
    "17.6_partial": "cell17_6_baseline_partial_limited_results.csv",
}

loaded = {}
audit_rows = []
for key, filename in source_files.items():
    df, path, ok = _read_csv_180a(filename)
    loaded[key] = (df, path, ok)
    audit_rows.append({
        "source_key": key,
        "filename": filename,
        "path": path,
        "loaded": bool(ok),
        "rows": int(len(df)),
        "required": bool(key in {"16.5_matrix", "17.6_safe"}),
    })

missing_required = [r for r in audit_rows if r["required"] and not r["loaded"]]
if missing_required:
    raise RuntimeError(f"[Cell18.0a] Missing required ledger source reports: {missing_required}")

rows = []

# ----------------------------------------------------------
# 5) Add Q1/Q2/Q3 decomposition rows
# ----------------------------------------------------------
q1, q1_path, q1_ok = loaded["16.1"]
if q1_ok and len(q1):
    for _, r in q1.iterrows():
        variant = str(r.get("variant", ""))
        artifact_role = str(r.get("artifact_role", ""))
        penalty = _safe_float_180a(r.get("q1_penalty_score"), np.nan)
        fatal_n = _safe_int_180a(r.get("fatal_n"), 0)
        _add_row(
            rows,
            source_cell="16.1",
            source_file=q1_path,
            source_row_key=variant,
            quality_dimension="Q1_marginal",
            claim_scope=variant,
            claim_type="empirical_decomposition",
            claim_text=f"Q1 marginal decomposition reports `{variant}` with penalty {_fmt(penalty)} and fatal_n={fatal_n}.",
            permitted_claim=(
                f"For Q1 marginal fidelity, `{variant}` has penalty {_fmt(penalty)} and fatal_n={fatal_n}; "
                "interpretation is scoped to this variant and evaluated columns only."
            ),
            forbidden_claim="Do not claim global/full-namespace marginal realism from this row alone.",
            metric_name="q1_penalty_score",
            metric_value=penalty,
            artifact_type=artifact_role,
            known_limitation="Q1 is variant/role scoped; full CPS Q1 remains role-dependent.",
            overclaim_risk="medium" if "public" in variant else "low",
            claim_scoped_status="claim_supported_with_caveat" if fatal_n > 0 else "claim_supported",
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )

q2, q2_path, q2_ok = loaded["16.2"]
if q2_ok and len(q2):
    for _, r in q2.iterrows():
        variant = str(r.get("variant", ""))
        artifact_role = str(r.get("artifact_role", ""))
        penalty = _safe_float_180a(r.get("q2_temporal_penalty_score"), np.nan)
        fatal_n = _safe_int_180a(r.get("fatal_n"), 0)
        _add_row(
            rows,
            source_cell="16.2",
            source_file=q2_path,
            source_row_key=variant,
            quality_dimension="Q2_temporal",
            claim_scope=variant,
            claim_type="empirical_decomposition",
            claim_text=f"Q2 temporal decomposition reports `{variant}` with penalty {_fmt(penalty)} and fatal_n={fatal_n}.",
            permitted_claim=(
                f"For Q2 temporal quality, `{variant}` has penalty {_fmt(penalty)} and fatal_n={fatal_n}; "
                "interpretation is scoped to this variant and evaluated columns only."
            ),
            forbidden_claim="Do not claim full temporal realism across all CPS roles from this row alone.",
            metric_name="q2_temporal_penalty_score",
            metric_value=penalty,
            artifact_type=artifact_role,
            known_limitation="Q2 is role/variant scoped; public candidate strength partly reflects role restriction.",
            overclaim_risk="medium",
            claim_scoped_status="claim_supported_with_caveat" if fatal_n > 0 else "claim_supported",
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )

q3, q3_path, q3_ok = loaded["16.3"]
if q3_ok and len(q3):
    for _, r in q3.iterrows():
        variant = str(r.get("variant", ""))
        artifact_role = str(r.get("artifact_role", ""))
        penalty = _safe_float_180a(r.get("q3_observability_penalty_score"), np.nan)
        fatal_n = _safe_int_180a(r.get("fatal_n"), 0)
        cols_eval = _safe_int_180a(r.get("cols_evaluable"), 0)
        _add_row(
            rows,
            source_cell="16.3",
            source_file=q3_path,
            source_row_key=variant,
            quality_dimension="Q3_observability",
            claim_scope=variant,
            claim_type="empirical_decomposition" if cols_eval else "scope_boundary",
            claim_text=f"Q3 observability decomposition reports `{variant}` with evaluable_cols={cols_eval}, penalty {_fmt(penalty)}, fatal_n={fatal_n}.",
            permitted_claim=(
                f"For Q3 observability, `{variant}` has evaluable_cols={cols_eval}, penalty {_fmt(penalty)}, fatal_n={fatal_n}. "
                "If evaluable_cols=0 for the public candidate, this is a role-restriction boundary, not a Q3 pass."
            ),
            forbidden_claim="Do not claim public candidate preserves full fine-grained observability/missingness/staleness realism.",
            metric_name="q3_observability_penalty_score",
            metric_value=penalty,
            artifact_type=artifact_role,
            known_limitation="Full/no-Q4 artifact has severe Q3 failures; public candidate excludes Q3 scope.",
            overclaim_risk="high",
            claim_scoped_status="claim_supported_with_caveat",
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )

# ----------------------------------------------------------
# 6) Add Q4 governance/decomposition rows
# ----------------------------------------------------------
q4df, q4_path, q4_ok = loaded["16.4"]
if q4_ok and len(q4df):
    for _, r in q4df.iterrows():
        stage_id = str(r.get("stage_id", ""))
        penalty = _safe_float_180a(r.get("q4_coupling_penalty_score"), np.nan)
        blockers = _safe_int_180a(r.get("publication_blocker_n"), 0)
        _add_row(
            rows,
            source_cell="16.4",
            source_file=q4_path,
            source_row_key=stage_id,
            quality_dimension="Q4_coupling",
            claim_scope=stage_id,
            claim_type="q4_governance_decomposition",
            claim_text=f"Q4 stage `{stage_id}` has penalty {_fmt(penalty)} and publication_blocker_n={blockers}.",
            permitted_claim=(
                f"Q4 stage `{stage_id}` is evidence in the no-promotion governance ledger; penalty={_fmt(penalty)}, blockers={blockers}. "
                "No Q4-coupled artifact is authoritative or promoted."
            ),
            forbidden_claim="Do not claim A0/Q4 was accepted, promoted, publication-ready, or that a Q4-coupled artifact is authoritative.",
            metric_name="q4_coupling_penalty_score",
            metric_value=penalty,
            artifact_type=str(r.get("stage_governance_status", "")),
            known_limitation="Q4 remains blocked_no_promotion.",
            overclaim_risk="high",
            claim_scoped_status="claim_supported_with_blocker_or_negative_result",
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )

# ----------------------------------------------------------
# 7) Add dimension matrix / baseline sanitizer rows
# ----------------------------------------------------------
matrix, matrix_path, matrix_ok = loaded["16.5_matrix"]
if matrix_ok and len(matrix):
    for _, r in matrix.iterrows():
        dim = str(r.get("quality_dimension", ""))
        sci_status = str(r.get("scientific_no_q4_status", ""))
        pub_status = str(r.get("public_candidate_status", ""))
        best = str(r.get("best_variant_or_stage", ""))
        _add_row(
            rows,
            source_cell="16.5",
            source_file=matrix_path,
            source_row_key=dim,
            quality_dimension=dim,
            claim_scope="dimension_matrix",
            claim_type="dimension_scoped_summary",
            claim_text=f"Dimension `{dim}` has best variant/stage `{best}`, scientific status `{sci_status}`, and public status `{pub_status}`.",
            permitted_claim=(
                f"Dimension `{dim}` must be reported with scoped statuses: scientific/no-Q4=`{sci_status}`, public=`{pub_status}`, best=`{best}`."
            ),
            forbidden_claim="Do not collapse dimension-scoped statuses into a single global publication_ready result.",
            metric_name="dimension_status",
            metric_value=np.nan,
            artifact_type="cell16_5_quality_dimension_matrix",
            known_limitation=str(r.get("main_interpretation", "")),
            overclaim_risk="medium",
            claim_scoped_status="claim_supported_with_caveat",
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )

safe, safe_path, safe_ok = loaded["17.6_safe"]
if safe_ok and len(safe):
    r = safe.iloc[0]
    pipeline_wins = _safe_int_180a(r.get("pipeline_headline_wins_n"), 0)
    partial_n = _safe_int_180a(r.get("partial_or_limited_scope_n"), 0)
    external_wins = _safe_int_180a(r.get("external_headline_wins_n"), 0)
    _add_row(
        rows,
        source_cell="17.6",
        source_file=safe_path,
        source_row_key="baseline_publication_safe_summary",
        quality_dimension="external_baselines",
        claim_scope="fully_comparable_same_scope_only",
        claim_type="baseline_fairness_summary",
        claim_text=f"Baseline sanitizer reports pipeline_headline_wins={pipeline_wins}, external_headline_wins={external_wins}, partial_or_limited_scope_n={partial_n}.",
        permitted_claim=(
            f"The role-aware pipeline outperformed external baselines in {pipeline_wins} fully comparable fair scopes; "
            f"{partial_n} scope(s) were partial/limited or missing a comparator and are not counted as headline wins."
        ),
        forbidden_claim=str(r.get("forbidden_claim", "Do not claim the pipeline outperforms all external baselines across the full CPS artifact.")),
        metric_name="pipeline_headline_wins_n",
        metric_value=pipeline_wins,
        artifact_type="baseline_claim_sanitizer",
        known_limitation="Single-run fair-scope baseline evidence only; no multi-seed superiority CI.",
        overclaim_risk="high",
        claim_scoped_status="claim_supported_with_caveat",
        public_direct_privacy_status=public_direct_privacy_status,
        public_distinguishability_status=public_distinguishability_status,
        public_strict_combined_status=public_strict_status,
    )

headline, headline_path, headline_ok = loaded["17.6_headline"]
if headline_ok and len(headline):
    for _, r in headline.iterrows():
        scope = str(r.get("scope_id", ""))
        winner = str(r.get("headline_winner", ""))
        score = _safe_float_180a(r.get("best_fair_scope_penalty_score"), np.nan)
        _add_row(
            rows,
            source_cell="17.6",
            source_file=headline_path,
            source_row_key=scope,
            quality_dimension="external_baselines",
            claim_scope=scope,
            claim_type="headline_comparable_scope",
            claim_text=f"Fully comparable baseline scope `{scope}` has headline winner `{winner}` and best penalty {_fmt(score)}.",
            permitted_claim=f"On fully comparable same-scope baseline scope `{scope}`, headline winner is `{winner}`; lower penalty is better.",
            forbidden_claim="Do not generalize this same-scope result to full CPS, Q3, Q4, Q6, or non-comparable scopes.",
            metric_name="best_fair_scope_penalty_score",
            metric_value=score,
            artifact_type="baseline_headline_scope",
            known_limitation="Single-run fair-scope comparison only.",
            overclaim_risk="medium",
            claim_scoped_status="claim_supported",
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )

partial, partial_path, partial_ok = loaded["17.6_partial"]
if partial_ok and len(partial):
    for _, r in partial.iterrows():
        scope = str(r.get("scope_id", ""))
        reason = str(r.get("comparison_reasons", ""))
        _add_row(
            rows,
            source_cell="17.6",
            source_file=partial_path,
            source_row_key=scope,
            quality_dimension="external_baselines",
            claim_scope=scope,
            claim_type="partial_or_limited_baseline_scope",
            claim_text=f"Baseline scope `{scope}` is partial/limited: {reason}.",
            permitted_claim=f"Baseline scope `{scope}` is reported as partial/limited or baseline-only and is not counted as a headline win.",
            forbidden_claim="Do not count this scope as a pipeline headline win or as evidence of full baseline superiority.",
            metric_name="partial_or_limited_scope",
            metric_value=np.nan,
            artifact_type="baseline_partial_scope",
            known_limitation=reason,
            overclaim_risk="high",
            claim_scoped_status="claim_supported_with_caveat",
            public_direct_privacy_status=public_direct_privacy_status,
            public_distinguishability_status=public_distinguishability_status,
            public_strict_combined_status=public_strict_status,
        )

master_ledger_df = pd.DataFrame(rows)
if len(master_ledger_df) == 0:
    raise RuntimeError("[Cell18.0a] No ledger rows were generated from available source reports.")


def _q4_stale_promoted_180(text):
    """
    Positive stale-Q4 claim detector.

    Allows safe negations such as:
      - no_Q4_coupled_public_artifact
      - no Q4-coupled artifact is authoritative
      - Q4 remains blocked_no_promotion
      - no-promotion governance
    """
    s = str(text).lower()
    s_norm = re.sub(r"[_\-]+", " ", s)

    safe_negation_patterns = [
        "no q4 coupled",
        "no q4 final",
        "no q4 artifact",
        "no q4 coupled public artifact",
        "not q4 coupled",
        "not promoted",
        "no promotion",
        "blocked no promotion",
        "blocked/no promotion",
        "blocked no promotion",
        "q4 remains blocked",
        "q4 final status = blocked",
        "q4 final status=blocked",
        "no q4 coupled artifact is authoritative",
        "q4 coupled artifacts used=false",
        "q4 coupled artifact is not authoritative",
    ]
    if any(p in s_norm for p in safe_negation_patterns):
        return False

    stale_patterns = [
        r"\bfinal\s+q4\s+coupled\b",
        r"\bq4\s+final\s+coupled\b",
        r"\baccepted\s+a0\b",
        r"\ba0\s+accepted\b",
        r"\ba0\s+promoted\b",
        r"\bq4\s+promoted\b",
        r"\baccepted\s+q4\b",
        r"\bzero\s+q4\s+blockers\b",
        r"\bq4\s+publication\s+ready\b",
        r"\bpublication\s+ready\s+q4\b",
        r"\bfinal\s+q4\s+artifact\b",
        r"\bq4\s+artifact\s+authoritative\b",
        r"\bq4\s+coupled\s+artifact\s+authoritative\b",
    ]
    return any(re.search(p, s_norm) for p in stale_patterns)


# ----------------------------------------------------------
# 8) Audit and save
# ----------------------------------------------------------
audit_rows.append({
    "source_key": "MASTER_LEDGER",
    "filename": "MASTER_RESULTS_LEDGER_FOR_PAPER.csv",
    "path": master_ledger_csv,
    "loaded": True,
    "rows": int(len(master_ledger_df)),
    "required": True,
})

audit_df = pd.DataFrame(audit_rows)

required_ledger_cols = [
    "ledger_row_id", "source_cell", "quality_dimension", "claim_scope",
    "claim_text", "permitted_claim", "forbidden_claim", "metric_name",
    "metric_value", "selection_split", "evaluation_split",
    "TEST_used_for_selection", "TEST_used_for_repair"
]
missing = [c for c in required_ledger_cols if c not in master_ledger_df.columns]
if missing:
    raise RuntimeError(f"[Cell18.0a] Internal error: missing ledger columns {missing}")

# Self-test the Q4 stale-promotion detector before using it.
_q4_detector_selftest_180a = {
    "no_Q4_coupled_public_artifact": False,
    "no Q4-coupled artifact is authoritative": False,
    "Q4 remains blocked_no_promotion": False,
    "A0 accepted": True,
    "Q4 promoted": True,
    "final Q4 artifact": True,
}
for _txt, _expected in _q4_detector_selftest_180a.items():
    _got = bool(_q4_stale_promoted_180(_txt))
    if _got != _expected:
        raise RuntimeError(
            f"[Cell18.0a] Q4 detector self-test failed: text={_txt!r} got={_got} expected={_expected}"
        )

# Protect against forbidden stale positive-Q4 language in permitted claims.
# Safe negations such as "no_Q4_coupled_public_artifact" are allowed.
unsafe_q4_mask = master_ledger_df["permitted_claim"].astype(str).apply(_q4_stale_promoted_180)
unsafe_q4 = master_ledger_df[unsafe_q4_mask]
if len(unsafe_q4):
    raise RuntimeError(
        "[Cell18.0a] Unsafe stale positive-Q4 permitted claims detected: "
        f"{unsafe_q4[['ledger_row_id','permitted_claim']].to_dict('records')[:5]}"
    )

unsafe_public = master_ledger_df[
    master_ledger_df["permitted_claim"].astype(str).str.lower().str.contains(
        "public release ready|release-ready|formal privacy guarantee|q6 pass|unrestricted public", regex=True, na=False
    )
]
if len(unsafe_public):
    raise RuntimeError(f"[Cell18.0a] Unsafe public-release permitted claims detected: {unsafe_public[['ledger_row_id','permitted_claim']].to_dict('records')[:5]}")

master_ledger_df.to_csv(master_ledger_csv, index=False)
audit_df.to_csv(audit_csv, index=False)

contract = {
    "cell": "18.0a",
    "version": CELL180A_VERSION,
    "role": "master_results_ledger_builder_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_promotion_claims_blocked": True,
        "safe_no_q4_negations_allowed": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell17_6": cell17_6_version,
    },
    "summary": {
        "ledger_rows": int(len(master_ledger_df)),
        "quality_dimension_counts": master_ledger_df["quality_dimension"].astype(str).value_counts().sort_index().to_dict(),
        "claim_type_counts": master_ledger_df["claim_type"].astype(str).value_counts().sort_index().to_dict(),
        "source_reports_loaded": audit_df[audit_df["loaded"].astype(bool)]["source_key"].astype(str).tolist(),
        "source_reports_missing": audit_df[~audit_df["loaded"].astype(bool)]["source_key"].astype(str).tolist(),
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "safe_no_q4_negations_allowed": True,
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "master_ledger_done_here": True,
    },
    "outputs": {
        "master_ledger_csv": master_ledger_csv,
        "audit_csv": audit_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}
_write_json_180a(contract_json, contract)
_write_json_180a(contract_canonical_json, contract)

manifest = {
    "cell": "18.0a",
    "version": CELL180A_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
}
_write_json_180a(manifest_json, manifest)

hashes = {
    "master_ledger_csv_sha256": _sha256_file_180a(master_ledger_csv),
    "audit_csv_sha256": _sha256_file_180a(audit_csv),
    "contract_json_sha256": _sha256_file_180a(contract_json),
    "contract_canonical_json_sha256": _sha256_file_180a(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_180a(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_180a(contract_json, contract)
_write_json_180a(contract_canonical_json, contract)
_write_json_180a(manifest_json, manifest)

# ----------------------------------------------------------
# 9) Export globals
# ----------------------------------------------------------
globals()["CELL180A_VERSION"] = CELL180A_VERSION
globals()["CELL18_0A_MASTER_RESULTS_LEDGER_DF"] = master_ledger_df
globals()["CELL18_0A_MASTER_RESULTS_LEDGER_AUDIT_DF"] = audit_df
globals()["CELL18_0A_MASTER_RESULTS_LEDGER_CONTRACT"] = contract

globals()["CELL18_0A_MASTER_RESULTS_LEDGER_CSV"] = master_ledger_csv
globals()["CELL18_0A_MASTER_RESULTS_LEDGER_AUDIT_CSV"] = audit_csv
globals()["CELL18_0A_MASTER_RESULTS_LEDGER_CONTRACT_JSON"] = contract_json
globals()["CELL18_0A_MASTER_RESULTS_LEDGER_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL18_0A_MASTER_RESULTS_LEDGER_MANIFEST_JSON"] = manifest_json

log(
    "[Cell18.0a] Master results ledger complete | "
    f"rows={len(master_ledger_df)} | "
    f"dimensions={master_ledger_df['quality_dimension'].nunique()} | "
    f"path={master_ledger_csv}"
)
log(
    "[Cell18.0a] Dimension counts | "
    f"{master_ledger_df['quality_dimension'].astype(str).value_counts().sort_index().to_dict()}"
)
log(f"[Cell18.0a] Saved master ledger: {master_ledger_csv}")
log(f"[Cell18.0a] Saved audit: {audit_csv}")
log(f"[Cell18.0a] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell18.0a] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "master_ledger_done_here=True | safe_no_q4_negations_allowed=True"
)
log("--- END: Cell 18.0a - Master results ledger for paper (v1.2 safe-negation-helper-fix no-Q4-promotion strict) ---")

gc.collect()