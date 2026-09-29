# ==========================================================
# CELL 17.5 - External baseline fairness and limitation report
# v1.2 STUDY-THESIS strict no-Q4-promotion baseline interpretation report, temporal-blocked C2ST aware
#
# Role:
#   - Summarize external baseline results from Cells 17.1–17.4.
#   - Explain fair comparison boundaries.
#   - Prevent unfair/overbroad claims against CTGAN, TabDDPM, and TimeGAN.
#   - Flag incomplete, unavailable, partial, or comparator-missing baseline comparisons.
#
# Final governance inherited from Cell 17.4:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative or used as a comparator.
#   - Public candidate direct privacy is warning-level, but strict public release
#     remains blocked by synthetic-real distinguishability.
#
# Scientific contract:
#   - No generation.
#   - No fitting.
#   - No materialization.
#   - No synthetic mutation.
#   - Uses existing Cell 17.0–17.4 outputs only.
#
# Outputs:
#   reports/cell17_5_baseline_fairness_summary.csv
#   reports/cell17_5_baseline_limitation_findings.csv
#   reports/cell17_5_baseline_publication_report.md
#   reports/cell17_5_baseline_fairness_contract.json
#   artifacts/contracts/cell17_5_baseline_fairness_contract_v1_1_THESIS.json
#   artifacts/cell17_5_baseline_fairness_manifest.json
# ==========================================================

log("--- START: Cell 17.5 - External baseline fairness and limitation report (v1.2 temporal-blocked-C2ST no-Q4-promotion strict) ---")

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
_required_175 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL17_0_BASELINE_SCOPE_REGISTRY_DF",
    "CELL17_0_BASELINE_COLUMN_REGISTRY_DF",
    "CELL17_0_BASELINE_FAIRNESS_CONTRACT",
    "CELL17_0_BASELINE_PLAN",
    "CELL17_1_CTGAN_RUN_AUDIT_DF",
    "CELL17_1_CTGAN_ARTIFACT_REGISTRY_DF",
    "CELL17_1_CTGAN_BASELINE_CONTRACT",
    "CELL17_2_TABDDPM_RUN_AUDIT_DF",
    "CELL17_2_TABDDPM_ARTIFACT_REGISTRY_DF",
    "CELL17_2_TABDDPM_BASELINE_CONTRACT",
    "CELL17_3_TIMEGAN_RUN_AUDIT_DF",
    "CELL17_3_TIMEGAN_ARTIFACT_REGISTRY_DF",
    "CELL17_3_TIMEGAN_BASELINE_CONTRACT",
    "CELL17_4_BASELINE_ARTIFACT_METRICS_DF",
    "CELL17_4_BASELINE_SCOPE_SUMMARY_DF",
    "CELL17_4_BASELINE_COMPARISON_LEDGER_DF",
    "CELL17_4_BASELINE_METRIC_CONTRACT",
]
_missing_175 = [k for k in _required_175 if k not in globals()]
if _missing_175:
    raise RuntimeError(f"[Cell17.5] Missing required globals: {_missing_175}")

ORIGINAL_OUTDIR_175 = str(OUTDIR)
ORIGINAL_OUT_SYN_175 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_175 = str(REPORT_DIR)

def _resolve_project_root_175(outdir, report_dir, out_syn):
    """
    Final STUDY-safe resolver.

    Cell 17.5 must read/write from the active clean final root only.
    It must not fall back to old FULL_COUPLED development roots.
    """
    candidates = []

    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue

        pp = os.path.abspath(str(p))
        base = os.path.basename(pp)

        if base in {"reports", "synthetic", "artifacts"}:
            candidates.append(os.path.dirname(pp))
        elif base == "contracts":
            candidates.append(os.path.dirname(os.path.dirname(pp)))
        else:
            candidates.append(pp)

    seen = set()

    for c in candidates:
        c = os.path.abspath(str(c))

        if c in seen:
            continue

        seen.add(c)

        if (
            os.path.isdir(os.path.join(c, "reports"))
            and os.path.isdir(os.path.join(c, "synthetic"))
        ):
            if (
                "FULL_COUPLED" in c
                and "STUDY_FINAL" not in c
                and "BINARY_V6_Q3V5_STUDY_FINAL" not in c
            ):
                raise RuntimeError(
                    "[Cell17.5] Refusing old FULL_COUPLED root for final baseline report: "
                    + c
                )

            return c

    raise RuntimeError(
        "[Cell17.5] Could not resolve active clean project root from current OUTDIR/REPORT_DIR/OUT_SYN. "
        "Do not fall back to old FULL_COUPLED roots."
    )

PROJECT_ROOT_175 = _resolve_project_root_175(ORIGINAL_OUTDIR_175, ORIGINAL_REPORT_DIR_175, ORIGINAL_OUT_SYN_175)
REPORT_DIR_BASE_175 = os.path.join(PROJECT_ROOT_175, "reports")
ARTDIR_BASE_175 = os.path.join(PROJECT_ROOT_175, "artifacts")
CONTRACT_DIR_BASE_175 = os.path.join(ARTDIR_BASE_175, "contracts")

os.makedirs(REPORT_DIR_BASE_175, exist_ok=True)
os.makedirs(ARTDIR_BASE_175, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_175, exist_ok=True)

SEED = int(SEED)

CELL175_VERSION = "cell17_5_external_baseline_fairness_report_v1_2_temporal_blocked_c2st_no_q4_promotion"

CFG["cell17_5_version"] = CELL175_VERSION
CFG["cell17_5_Q4_final_status"] = "blocked_no_promotion"
CFG["cell17_5_Q4_coupled_artifacts_used"] = False
CFG["cell17_5_TEST_real_values_used_here"] = False
CFG["cell17_5_TEST_real_values_used_for_materialization"] = False
CFG["cell17_5_synthetic_values_mutated"] = False
CFG["cell17_5_selection_done_here"] = False
CFG["cell17_5_generator_fit_done_here"] = False
CFG["cell17_5_materialization_done_here"] = False
CFG["cell17_5_baseline_report_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell17_5_min_comparator_col_coverage", 0.80)
CFG.setdefault("cell17_5_material_score_delta_threshold", 0.02)

MIN_COVERAGE_175 = float(CFG.get("cell17_5_min_comparator_col_coverage", 0.80))
MATERIAL_DELTA_175 = float(CFG.get("cell17_5_material_score_delta_threshold", 0.02))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
fairness_summary_csv = os.path.join(REPORT_DIR_BASE_175, "cell17_5_baseline_fairness_summary.csv")
limitation_findings_csv = os.path.join(REPORT_DIR_BASE_175, "cell17_5_baseline_limitation_findings.csv")
publication_report_md = os.path.join(REPORT_DIR_BASE_175, "cell17_5_baseline_publication_report.md")
contract_json = os.path.join(REPORT_DIR_BASE_175, "cell17_5_baseline_fairness_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_175, "cell17_5_baseline_fairness_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_175, "cell17_5_baseline_fairness_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_175(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_175(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_175(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_175(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_175(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_175(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_175(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_175(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_175(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_175(payload), f, indent=2, sort_keys=True)

def _sha256_file_175(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_175(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_int_175(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _fmt_175(x, digits=4):
    x = _safe_float_175(x, np.nan)
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"

def _boolish_175(x, default=False):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if x is None:
        return bool(default)

    try:
        if pd.isna(x):
            return bool(default)
    except Exception:
        pass

    s = str(x).strip().lower()

    if s in {"true", "1", "yes", "y", "t"}:
        return True

    if s in {"false", "0", "no", "n", "f", ""}:
        return False

    return bool(default)


def _require_contract_175(contract: dict, name: str, expected_version_substrings):
    """
    Validate no-Q4-promotion baseline contracts.

    expected_version_substrings may be:
      - a single string
      - a list/tuple of accepted version substrings

    This accepts both the original Cell 17.1 CTGAN contract and the newer
    kernel-safe Cell 17.1R subprocess contract.
    """
    if not isinstance(contract, dict):
        raise RuntimeError(f"[Cell17.5] {name} is not a dict.")

    version = str(contract.get("version", ""))

    if isinstance(expected_version_substrings, str):
        expected_version_substrings = [expected_version_substrings]

    ok_version = any(str(s) in version for s in expected_version_substrings)

    if not ok_version:
        raise RuntimeError(
            f"[Cell17.5] Unexpected {name} version: {version}. "
            f"Accepted substrings: {expected_version_substrings}"
        )

    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})

    final_q4_status = str(
        q4.get("final_q4_status", strict.get("Q4_final_status", ""))
    )

    if final_q4_status != "blocked_no_promotion":
        raise RuntimeError(
            f"[Cell17.5] {name} does not carry Q4 blocked_no_promotion governance. "
            f"Observed: {final_q4_status}"
        )

    q4_used = _boolish_175(
        q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True)),
        default=True,
    )

    if q4_used:
        raise RuntimeError(f"[Cell17.5] {name} indicates Q4-coupled artifacts were used.")

    return version

def _scope_cols_n_175(scope_id: str) -> int:
    d = CELL17_0_BASELINE_COLUMN_REGISTRY_DF.copy()
    d["scope_id"] = d["scope_id"].astype(str)
    return int(d[d["scope_id"].eq(str(scope_id))]["col"].nunique())

def _scope_display_175(scope_id: str) -> str:
    d = CELL17_0_BASELINE_SCOPE_REGISTRY_DF.copy()
    d["scope_id"] = d["scope_id"].astype(str)
    row = d[d["scope_id"].eq(str(scope_id))]
    if len(row):
        return str(row.iloc[0].get("display_name", scope_id))
    return str(scope_id)

def _intended_baselines_175(scope_id: str) -> str:
    d = CELL17_0_BASELINE_SCOPE_REGISTRY_DF.copy()
    d["scope_id"] = d["scope_id"].astype(str)
    row = d[d["scope_id"].eq(str(scope_id))]
    if len(row):
        return str(row.iloc[0].get("intended_baselines", ""))
    return ""

def _append_finding_175(rows, severity, scope_id, baseline, finding_type, finding, evidence, implication, paper_wording):
    rows.append({
        "severity": severity,
        "scope_id": scope_id,
        "scope_display": _scope_display_175(scope_id) if scope_id else "",
        "baseline": baseline,
        "finding_type": finding_type,
        "finding": finding,
        "evidence": evidence,
        "implication": implication,
        "recommended_paper_wording": paper_wording,
        "TEST_real_values_used_here": False,
        "synthetic_values_mutated": False,
    })

# ----------------------------------------------------------
# 4) Validate upstream contracts
# ----------------------------------------------------------
CELL17_0_VERSION_175 = _require_contract_175(
    CELL17_0_BASELINE_FAIRNESS_CONTRACT,
    "CELL17_0_BASELINE_FAIRNESS_CONTRACT",
    "cell17_0_external_baseline_scope_contract_v1_1",
)

CELL17_1_VERSION_175 = _require_contract_175(
    CELL17_1_CTGAN_BASELINE_CONTRACT,
    "CELL17_1_CTGAN_BASELINE_CONTRACT",
    [
        "cell17_1_ctgan_external_baseline_v1_1",
        "cell17_1R_ctgan_external_baseline_subprocess_v1_2_THESIS",
    ],
)

CELL17_2_VERSION_175 = _require_contract_175(
    CELL17_2_TABDDPM_BASELINE_CONTRACT,
    "CELL17_2_TABDDPM_BASELINE_CONTRACT",
    [
        "cell17_2_tabddpm_external_baseline_v1_1",
        "cell17_2R_tabddpm_external_baseline",
    ],
)

CELL17_3_VERSION_175 = _require_contract_175(
    CELL17_3_TIMEGAN_BASELINE_CONTRACT,
    "CELL17_3_TIMEGAN_BASELINE_CONTRACT",
    [
        "cell17_3_timegan_external_baseline_v1_1",
        "cell17_3R_timegan_external_baseline",
    ],
)

CELL17_4_VERSION_175 = _require_contract_175(
    CELL17_4_BASELINE_METRIC_CONTRACT,
    "CELL17_4_BASELINE_METRIC_CONTRACT",
    [
        "cell17_4_external_baseline_metric_computation_v1_2",
        "cell17_4_external_baseline_metric_computation_v1_2_temporal_blocked_c2st_no_q4_promotion",
    ],
)
# ----------------------------------------------------------
# 5) Collect source data
# ----------------------------------------------------------
scope_registry = CELL17_0_BASELINE_SCOPE_REGISTRY_DF.copy()
scope_registry["scope_id"] = scope_registry["scope_id"].astype(str)

artifact_metrics = CELL17_4_BASELINE_ARTIFACT_METRICS_DF.copy()
scope_summary = CELL17_4_BASELINE_SCOPE_SUMMARY_DF.copy()
comparison_ledger = CELL17_4_BASELINE_COMPARISON_LEDGER_DF.copy()

ctgan_run = CELL17_1_CTGAN_RUN_AUDIT_DF.copy()
tabddpm_run = CELL17_2_TABDDPM_RUN_AUDIT_DF.copy()
timegan_run = CELL17_3_TIMEGAN_RUN_AUDIT_DF.copy()

public_context = CELL17_0_BASELINE_FAIRNESS_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(public_context.get("public_distinguishability_status", ""))
public_strict_status = str(public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 6) Build fairness summary
# ----------------------------------------------------------
summary_rows = []

for _, r in scope_summary.iterrows():
    scope_id = str(r.get("scope_id", ""))
    scope_cols_n = _scope_cols_n_175(scope_id)
    metric_family = str(r.get("metric_family", ""))
    best_artifact = str(r.get("best_artifact", ""))
    best_score = _safe_float_175(r.get("best_fair_scope_penalty_score"), np.nan)

    ext_rows = artifact_metrics[
        artifact_metrics["scope_id"].astype(str).eq(scope_id)
        & artifact_metrics["artifact_type"].astype(str).eq("external_baseline")
    ].copy()

    pipe_rows = artifact_metrics[
        artifact_metrics["scope_id"].astype(str).eq(scope_id)
        & artifact_metrics["artifact_type"].astype(str).eq("pipeline_same_scope_comparator")
    ].copy()

    evaluated_external = ext_rows[ext_rows["metric_status"].astype(str).eq("evaluated")].copy()
    not_eval_external = ext_rows[~ext_rows["metric_status"].astype(str).eq("evaluated")].copy()

    pipeline_eval_cols = np.nan
    pipeline_col_coverage = np.nan
    if len(pipe_rows) and "eval_cols_n" in pipe_rows.columns and str(pipe_rows.iloc[0].get("metric_status", "")) == "evaluated":
        pipeline_eval_cols = _safe_float_175(pipe_rows.iloc[0].get("eval_cols_n"), np.nan)
        pipeline_col_coverage = pipeline_eval_cols / max(1, scope_cols_n) if np.isfinite(pipeline_eval_cols) else np.nan

    status = "fair_comparison_available"
    reasons = []

    if int(r.get("external_baselines_evaluated", 0)) == 0:
        status = "limited_no_external_baseline_evaluated"
        reasons.append("no_external_baseline_evaluated")

    if not bool(r.get("pipeline_comparator_evaluated", False)):
        status = "limited_no_pipeline_comparator"
        reasons.append("pipeline_comparator_not_evaluated")

    if np.isfinite(pipeline_col_coverage) and pipeline_col_coverage < MIN_COVERAGE_175:
        status = "limited_partial_pipeline_comparator"
        reasons.append(f"pipeline_comparator_column_coverage_below_{MIN_COVERAGE_175}")

    winners = []
    led = pd.DataFrame()
    if len(comparison_ledger):
        led = comparison_ledger[comparison_ledger["scope_id"].astype(str).eq(scope_id)].copy()
        if len(led) and "winner" in led.columns:
            winners = led["winner"].astype(str).tolist()

    summary_rows.append({
        "scope_id": scope_id,
        "scope_display": _scope_display_175(scope_id),
        "metric_family": metric_family,
        "scope_cols_n": int(scope_cols_n),
        "intended_baselines": _intended_baselines_175(scope_id),
        "artifacts_total": _safe_int_175(r.get("artifacts_total"), 0),
        "artifacts_evaluated": _safe_int_175(r.get("artifacts_evaluated"), 0),
        "external_baselines_evaluated": _safe_int_175(r.get("external_baselines_evaluated"), 0),
        "pipeline_comparator_evaluated": bool(r.get("pipeline_comparator_evaluated", False)),
        "pipeline_eval_cols_n": pipeline_eval_cols,
        "pipeline_col_coverage": pipeline_col_coverage,
        "best_artifact": best_artifact,
        "best_fair_scope_penalty_score": best_score,
        "external_winner_n": int(sum(1 for w in winners if w == "external_baseline")),
        "pipeline_winner_n": int(sum(1 for w in winners if w == "pipeline_same_scope")),
        "not_comparable_n": int(sum(1 for w in winners if w == "not_comparable")),
        "comparison_status": status,
        "comparison_reasons": "|".join(reasons) if reasons else "fair_scope_comparison_completed",
        "not_evaluable_external_baselines": "|".join(sorted(not_eval_external["baseline"].astype(str).drop_duplicates().tolist())) if len(not_eval_external) else "",
        "evaluated_external_baselines": "|".join(sorted(evaluated_external["baseline"].astype(str).drop_duplicates().tolist())) if len(evaluated_external) else "",
        "TEST_real_values_used_here": False,
        "synthetic_values_mutated": False,
    })

fairness_summary_df = pd.DataFrame(summary_rows)

# ----------------------------------------------------------
# 7) Findings
# ----------------------------------------------------------
findings = []

ctgan_success = int((ctgan_run["status"].astype(str) == "success").sum()) if len(ctgan_run) else 0
ctgan_not_run = int((ctgan_run["status"].astype(str) == "not_run").sum()) if len(ctgan_run) else 0
ctgan_failed = int((ctgan_run["status"].astype(str) == "failed").sum()) if len(ctgan_run) else 0
ctgan_skipped = int((ctgan_run["status"].astype(str) == "skipped").sum()) if len(ctgan_run) else 0

if ctgan_success == 0:
    if ctgan_not_run > 0 and ctgan_failed == 0 and ctgan_skipped == 0:
        ctgan_finding = "CTGAN was not run because the backend was unavailable or disabled."
        ctgan_implication = "CTGAN cannot be used as an empirical comparator unless the backend is available and rerun."
        ctgan_wording = (
            "CTGAN was specified as a planned external baseline, but it was unavailable or disabled in the execution environment; "
            "therefore, no CTGAN empirical result is reported."
        )
    elif ctgan_failed > 0:
        ctgan_finding = "CTGAN was attempted but failed under the declared resource/runtime constraints."
        ctgan_implication = (
            "CTGAN cannot be used as an empirical comparator for failed scopes; failures should be reported as execution/resource limitations."
        )
        ctgan_wording = (
            "CTGAN was attempted under the declared baseline protocol, but it did not complete successfully under the available execution constraints; "
            "therefore, no successful CTGAN empirical result is reported for those scopes."
        )
    elif ctgan_skipped > 0:
        ctgan_finding = "CTGAN scopes were skipped by the declared kernel-safe resource policy."
        ctgan_implication = (
            "Skipped CTGAN scopes are not empirical baseline results and must not be treated as losses or wins."
        )
        ctgan_wording = (
            "CTGAN was included in the baseline plan, but one or more scopes were skipped by the declared resource-safety policy; "
            "therefore, CTGAN is reported as not evaluable for those scopes."
        )
    else:
        ctgan_finding = "No successful CTGAN baseline artifact was available."
        ctgan_implication = "CTGAN cannot be used as an empirical comparator without successful artifacts."
        ctgan_wording = (
            "No successful CTGAN artifact was available for fair-scope evaluation; therefore, no CTGAN empirical result is reported."
        )

    _append_finding_175(
        findings,
        severity="major",
        scope_id="",
        baseline="CTGAN",
        finding_type="baseline_unavailable_or_failed",
        finding=ctgan_finding,
        evidence=(
            f"CTGAN success_n={ctgan_success}, not_run_n={ctgan_not_run}, "
            f"failed_n={ctgan_failed}, skipped_n={ctgan_skipped}."
        ),
        implication=ctgan_implication,
        paper_wording=ctgan_wording,
    )

tab_success = int((tabddpm_run["status"].astype(str) == "success").sum()) if len(tabddpm_run) else 0
tab_failed = int((tabddpm_run["status"].astype(str) == "failed").sum()) if len(tabddpm_run) else 0
_append_finding_175(
    findings,
    severity="informative" if tab_success else "major",
    scope_id="",
    baseline="TabDDPM",
    finding_type="baseline_execution",
    finding="TabDDPM completed all configured tabular fair scopes." if tab_success == 3 else "TabDDPM did not complete all configured fair scopes.",
    evidence=f"TabDDPM success_n={tab_success}, failed_n={tab_failed}.",
    implication="TabDDPM comparisons are usable only for successful tabular scopes.",
    paper_wording="TabDDPM is evaluated only on compact tabular scopes where its modeling assumptions are appropriate.",
)

tg_success = int((timegan_run["status"].astype(str) == "success").sum()) if len(timegan_run) else 0
tg_failed = int((timegan_run["status"].astype(str) == "failed").sum()) if len(timegan_run) else 0
_append_finding_175(
    findings,
    severity="informative" if tg_success == 2 else "major",
    scope_id="",
    baseline="TimeGAN",
    finding_type="baseline_execution",
    finding="TimeGAN completed both configured temporal fair scopes." if tg_success == 2 else "TimeGAN did not complete all configured temporal scopes.",
    evidence=f"TimeGAN success_n={tg_success}, failed_n={tg_failed}.",
    implication="TimeGAN comparisons are usable only for successful temporal scopes.",
    paper_wording="TimeGAN is evaluated only on compact temporal scopes rather than the full CPS namespace.",
)

# Same-scope comparison findings.
same_scope_pipeline_wins = 0
same_scope_external_wins = 0
same_scope_not_comparable = 0

if len(comparison_ledger):
    for _, r in comparison_ledger.iterrows():
        scope_id = str(r.get("scope_id", ""))
        baseline = str(r.get("baseline", ""))
        winner = str(r.get("winner", ""))
        delta = _safe_float_175(r.get("delta_external_minus_pipeline"), np.nan)
        ext_score = _safe_float_175(r.get("external_baseline_score"), np.nan)
        pipe_score = _safe_float_175(r.get("pipeline_same_scope_score"), np.nan)

        if winner == "pipeline_same_scope":
            same_scope_pipeline_wins += 1
        elif winner == "external_baseline":
            same_scope_external_wins += 1
        else:
            same_scope_not_comparable += 1

        severity = "informative"
        if np.isfinite(delta) and abs(delta) > MATERIAL_DELTA_175:
            severity = "major"

        _append_finding_175(
            findings,
            severity=severity,
            scope_id=scope_id,
            baseline=baseline,
            finding_type="same_scope_result",
            finding=(
                "Role-aware pipeline outperformed the external baseline on this same-scope comparison."
                if winner == "pipeline_same_scope"
                else "External baseline outperformed the role-aware pipeline on this same-scope comparison."
                if winner == "external_baseline"
                else "Scope comparison was not fully comparable."
            ),
            evidence=(
                f"external_score={_fmt_175(ext_score)}, pipeline_score={_fmt_175(pipe_score)}, "
                f"delta_external_minus_pipeline={_fmt_175(delta)}."
            ),
            implication="This is a same-scope result only and must not be generalized to Q3/Q4/Q6 or full-CPS claims.",
            paper_wording=(
                f"On the `{scope_id}` fair scope, {baseline} obtained penalty {_fmt_175(ext_score)}, "
                f"whereas the same-scope pipeline comparator obtained {_fmt_175(pipe_score)}; lower is better."
            ),
        )

# Missing comparator caveats.
no_comparator_rows = fairness_summary_df[
    fairness_summary_df["comparison_status"].astype(str).eq("limited_no_pipeline_comparator")
].copy()
for _, r in no_comparator_rows.iterrows():
    _append_finding_175(
        findings,
        severity="major",
        scope_id=str(r["scope_id"]),
        baseline="RoleAwarePipeline",
        finding_type="missing_pipeline_comparator",
        finding="No same-scope pipeline comparator was available for this scope.",
        evidence=f"scope={r['scope_id']}; external_baselines_evaluated={r['external_baselines_evaluated']}; best_artifact={r['best_artifact']}.",
        implication="This scope may report baseline quality, but it cannot support a pipeline-vs-baseline win/loss claim.",
        paper_wording=(
            f"The `{r['scope_id']}` scope is reported as baseline-only because no same-scope pipeline comparator was available."
        ),
    )

# Partial comparator caveats.
partial_rows = fairness_summary_df[
    pd.to_numeric(fairness_summary_df["pipeline_col_coverage"], errors="coerce") < MIN_COVERAGE_175
].copy()
for _, r in partial_rows.iterrows():
    _append_finding_175(
        findings,
        severity="major",
        scope_id=str(r["scope_id"]),
        baseline="RoleAwarePipeline",
        finding_type="partial_comparator_coverage",
        finding="Pipeline same-scope comparator did not cover enough columns for a strong comparison.",
        evidence=(
            f"pipeline_eval_cols_n={_fmt_175(r.get('pipeline_eval_cols_n'), 0)}, "
            f"scope_cols_n={r.get('scope_cols_n')}, coverage={_fmt_175(_safe_float_175(r.get('pipeline_col_coverage')))}."
        ),
        implication="This scope should be reported as limited/partial, not as a decisive pipeline-vs-baseline win.",
        paper_wording=(
            f"The `{r['scope_id']}` comparison is reported as partial because same-scope pipeline comparator coverage was below the configured threshold."
        ),
    )

_append_finding_175(
    findings,
    severity="major",
    scope_id="",
    baseline="ALL",
    finding_type="fairness_boundary",
    finding="External baselines are not full CPS replacements.",
    evidence="Cell 17.0 excludes Q3 observability, Q4 coupling repair, Q6 release safety, and full namespace assembly from baseline claims.",
    implication="The paper should compare baselines only on fair scopes and separately report pipeline-only capabilities and governance boundaries.",
    paper_wording=(
        "External baselines are evaluated only on comparable tabular or temporal scopes. "
        "They are not evaluated as replacements for the full role-aware, mask-aware, coupling-aware CPS pipeline."
    ),
)

_append_finding_175(
    findings,
    severity="major",
    scope_id="",
    baseline="ALL",
    finding_type="q4_q6_governance_boundary",
    finding="Baseline comparisons inherit no-Q4-promotion and strict public-Q6 governance.",
    evidence=(
        f"Q4_final_status=blocked_no_promotion; public_direct_privacy={public_direct_privacy_status}; "
        f"public_distinguishability={public_distinguishability_status}; public_strict={public_strict_status}."
    ),
    implication="Baseline comparisons cannot be used to claim Q4-coupled release success or public-release privacy success.",
    paper_wording=(
        "Baseline comparisons are reported separately from Q4 and Q6 governance. "
        "The public candidate clears direct no-copy/DCR blocker evidence only at warning level and remains strict-release blocked by distinguishability."
    ),
)

findings_df = pd.DataFrame(findings)

# ----------------------------------------------------------
# 8) Markdown report
# ----------------------------------------------------------
report_lines = []
report_lines.append("# External Baseline Fairness and Limitation Report")
report_lines.append("")
report_lines.append(f"Version: `{CELL175_VERSION}`")
report_lines.append("")
report_lines.append("## Executive Summary")
report_lines.append("")
report_lines.append(
    "External baselines were evaluated only on comparable target scopes. "
    "This avoids unfair comparison of generic tabular or sequence models against the full role-aware, mask-aware CPS pipeline. "
    "The baseline section also inherits the final governance boundary: Q4 remains `blocked_no_promotion`, and public Q6 remains strict-release blocked by distinguishability."
)
report_lines.append("")
report_lines.append("## Baseline Availability")
report_lines.append("")
report_lines.append("| Baseline | Status | Evidence |")
report_lines.append("|---|---|---|")
ctgan_report_status = (
    "evaluated" if ctgan_success > 0
    else "not_run" if ctgan_not_run > 0 and ctgan_failed == 0 and ctgan_skipped == 0
    else "failed" if ctgan_failed > 0
    else "skipped" if ctgan_skipped > 0
    else "not_evaluable"
)

report_lines.append(
    f"| CTGAN | {ctgan_report_status} | "
    f"success={ctgan_success}, not_run={ctgan_not_run}, failed={ctgan_failed}, skipped={ctgan_skipped} |"
)
report_lines.append(f"| TabDDPM | evaluated | success={tab_success}, failed={tab_failed} |")
report_lines.append(f"| TimeGAN | evaluated | success={tg_success}, failed={tg_failed} |")
report_lines.append("")
report_lines.append("## Fair Scope Summary")
report_lines.append("")
report_lines.append("| Scope | Metric Family | Evaluated External Baselines | Best Artifact | Status | Notes |")
report_lines.append("|---|---|---|---|---|---|")

for _, r in fairness_summary_df.iterrows():
    report_lines.append(
        f"| `{r['scope_id']}` | {r['metric_family']} | {r['evaluated_external_baselines'] or 'none'} | "
        f"{r['best_artifact']} | {r['comparison_status']} | {r['comparison_reasons']} |"
    )

report_lines.append("")
report_lines.append("## Same-Scope Comparison Results")
report_lines.append("")
if len(comparison_ledger):
    report_lines.append("| Scope | Baseline | External Score | Pipeline Score | Winner |")
    report_lines.append("|---|---|---:|---:|---|")
    for _, r in comparison_ledger.iterrows():
        report_lines.append(
            f"| `{r.get('scope_id', '')}` | {r.get('baseline', '')} | "
            f"{_fmt_175(r.get('external_baseline_score'))} | "
            f"{_fmt_175(r.get('pipeline_same_scope_score'))} | {r.get('winner', '')} |"
        )
else:
    report_lines.append("No same-scope comparison ledger was available.")

report_lines.append("")
report_lines.append("## Interpretation")
report_lines.append("")
report_lines.append(
    f"The same-scope comparison ledger contains {same_scope_pipeline_wins} pipeline wins, "
    f"{same_scope_external_wins} external-baseline wins, and {same_scope_not_comparable} non-comparable rows. "
    "Scopes without a same-scope pipeline comparator are baseline-only and must not be counted as pipeline wins or losses."
)
report_lines.append("")
report_lines.append("## Limitations")
report_lines.append("")
report_lines.append(
    "- CTGAN is reported according to the declared execution outcome: successful scopes are evaluated, "
    "while unavailable, failed, or skipped scopes are reported as not evaluable rather than silently replaced."
)
report_lines.append("- TabDDPM is evaluated as a numeric tabular diffusion baseline on compact tabular scopes only.")
report_lines.append("- TimeGAN is evaluated as a compact sequence baseline on temporal scopes only.")
report_lines.append("- No external baseline is evaluated on Q3 observability, Q4 coupling repair, Q6 release safety, or full-namespace generation.")
report_lines.append("- Scope-level conclusions are valid only where same-scope comparators exist.")
report_lines.append("- Q4 remains `blocked_no_promotion`; no Q4-coupled artifact is used as a comparator.")
report_lines.append("- Public Q6 direct privacy is warning-level, but strict public release remains blocked by distinguishability.")

if len(no_comparator_rows):
    report_lines.append("")
    report_lines.append("### Baseline-only / Missing-comparator Scopes")
    report_lines.append("")
    for _, r in no_comparator_rows.iterrows():
        report_lines.append(
            f"- `{r['scope_id']}`: no same-scope pipeline comparator was evaluated; report as baseline-only."
        )

if len(partial_rows):
    report_lines.append("")
    report_lines.append("### Partial Comparisons")
    report_lines.append("")
    for _, r in partial_rows.iterrows():
        coverage = _safe_float_175(r.get("pipeline_col_coverage"), np.nan)
        report_lines.append(
            f"- `{r['scope_id']}`: pipeline comparator coverage={_fmt_175(coverage * 100, 1)}%; interpret as limited/partial."
        )

report_lines.append("")
report_lines.append("## Recommended Paper Wording")
report_lines.append("")
report_lines.append(
    "External baselines are evaluated only on scoped tasks that match their modeling assumptions. "
    "TabDDPM is evaluated on compact row-wise tabular scopes, while TimeGAN is evaluated on compact temporal scopes. "
    "CTGAN is reported according to the declared execution outcome: successful scopes are evaluated, and unavailable, failed, or skipped scopes are reported as not evaluable. "
    "These baselines are not treated as full replacements for the role-aware CPS generator because they do not implement strict observability handling, IoT role contracts, Q4 coupling repair, or Q6 release-safety controls. "
    "The comparison is same-scope only; scopes without a same-scope pipeline comparator are reported as baseline-only rather than as pipeline wins."
)

publication_report = "\n".join(report_lines)

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
fairness_summary_df.to_csv(fairness_summary_csv, index=False)
findings_df.to_csv(limitation_findings_csv, index=False)

with open(publication_report_md, "w", encoding="utf-8") as f:
    f.write(publication_report)

# ----------------------------------------------------------
# 10) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "17.5",
    "version": CELL175_VERSION,
    "role": "external_baseline_fairness_and_limitation_report_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "baseline_comparators_use_no_q4_or_public_q6_only": True,
        "baseline_c2st_uses_temporal_blocked_split": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell17_0": CELL17_0_VERSION_175,
        "cell17_1": CELL17_1_VERSION_175,
        "cell17_2": CELL17_2_VERSION_175,
        "cell17_3": CELL17_3_VERSION_175,
        "cell17_4": CELL17_4_VERSION_175,
    },
    "summary": {
        "fairness_summary_rows": int(len(fairness_summary_df)),
        "limitation_findings_rows": int(len(findings_df)),
        "ctgan_success_n": int(ctgan_success),
        "ctgan_not_run_n": int(ctgan_not_run),
        "ctgan_failed_n": int(ctgan_failed),
        "ctgan_skipped_n": int(ctgan_skipped),
        "tabddpm_success_n": int(tab_success),
        "tabddpm_failed_n": int(tab_failed),
        "timegan_success_n": int(tg_success),
        "timegan_failed_n": int(tg_failed),
        "same_scope_pipeline_wins": int(same_scope_pipeline_wins),
        "same_scope_external_wins": int(same_scope_external_wins),
        "same_scope_not_comparable": int(same_scope_not_comparable),
        "missing_comparator_scopes": no_comparator_rows["scope_id"].astype(str).tolist() if len(no_comparator_rows) else [],
        "partial_comparison_scopes": partial_rows["scope_id"].astype(str).tolist() if len(partial_rows) else [],
    },
    "core_conclusion": {
        "baseline_result": (
            "Same-scope pipeline comparator wins only where a comparator exists; "
            "scopes without comparator are baseline-only and must not be counted as pipeline wins."
        ),
        "ctgan": (
            "Reported according to declared execution outcome: successful scopes are evaluated; "
            "unavailable, failed, or skipped scopes are not treated as empirical results."
        ),
        "tabddpm": "Evaluated on tabular fair scopes.",
        "timegan": "Evaluated on temporal fair scopes.",
        "fairness_boundary": "Do not compare external baselines against Q3/Q4/Q6/full-namespace capabilities.",
        "q4_boundary": "Q4 remains blocked_no_promotion.",
        "q6_boundary": "Public candidate is direct-privacy warning but strict distinguishability blocked.",
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "baseline_c2st_uses_temporal_blocked_split": True,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "baseline_report_done_here": True,
    },
    "outputs": {
        "fairness_summary_csv": fairness_summary_csv,
        "limitation_findings_csv": limitation_findings_csv,
        "publication_report_md": publication_report_md,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_175(contract_json, contract)
_write_json_175(contract_canonical_json, contract)

manifest = {
    "cell": "17.5",
    "version": CELL175_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "summary": contract["summary"],
    "core_conclusion": contract["core_conclusion"],
    "strict_contract": contract["strict_contract"],
}
_write_json_175(manifest_json, manifest)

hashes = {
    "fairness_summary_csv_sha256": _sha256_file_175(fairness_summary_csv),
    "limitation_findings_csv_sha256": _sha256_file_175(limitation_findings_csv),
    "publication_report_md_sha256": _sha256_file_175(publication_report_md),
    "contract_json_sha256": _sha256_file_175(contract_json),
    "contract_canonical_json_sha256": _sha256_file_175(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_175(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_175(contract_json, contract)
_write_json_175(contract_canonical_json, contract)
_write_json_175(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL175_VERSION"] = CELL175_VERSION
globals()["CELL17_5_BASELINE_FAIRNESS_SUMMARY_DF"] = fairness_summary_df
globals()["CELL17_5_BASELINE_LIMITATION_FINDINGS_DF"] = findings_df
globals()["CELL17_5_BASELINE_PUBLICATION_REPORT_MD_TEXT"] = publication_report
globals()["CELL17_5_BASELINE_FAIRNESS_CONTRACT"] = contract

globals()["CELL17_5_BASELINE_FAIRNESS_SUMMARY_CSV"] = fairness_summary_csv
globals()["CELL17_5_BASELINE_LIMITATION_FINDINGS_CSV"] = limitation_findings_csv
globals()["CELL17_5_BASELINE_PUBLICATION_REPORT_MD"] = publication_report_md
globals()["CELL17_5_BASELINE_CONTRACT_JSON"] = contract_json
globals()["CELL17_5_BASELINE_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL17_5_BASELINE_MANIFEST_JSON"] = manifest_json

log(
    "[Cell17.5] External baseline fairness report complete | "
    f"summary_rows={len(fairness_summary_df)} | "
    f"findings={len(findings_df)} | "
    f"ctgan_success={ctgan_success} | "
    f"tabddpm_success={tab_success} | "
    f"timegan_success={tg_success} | "
    f"pipeline_wins={same_scope_pipeline_wins} | "
    f"external_wins={same_scope_external_wins} | "
    f"missing_comparator_scopes={no_comparator_rows['scope_id'].astype(str).tolist() if len(no_comparator_rows) else []} | "
    "q4_status=blocked_no_promotion | c2st_split_policy=temporal_blocked_same_positions"
)
if len(partial_rows):
    log(
        "[Cell17.5] Partial comparison scopes flagged | "
        f"{partial_rows[['scope_id', 'pipeline_eval_cols_n', 'scope_cols_n', 'pipeline_col_coverage']].to_dict('records')}"
    )
log(f"[Cell17.5] Saved fairness summary: {fairness_summary_csv}")
log(f"[Cell17.5] Saved limitation findings: {limitation_findings_csv}")
log(f"[Cell17.5] Saved publication report: {publication_report_md}")
log(f"[Cell17.5] Saved contract: {contract_json}")
log(f"[Cell17.5] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell17.5] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_here=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "baseline_report_done_here=True"
)
log("--- END: Cell 17.5 - External baseline fairness and limitation report (v1.2 temporal-blocked-C2ST no-Q4-promotion strict) ---")

gc.collect()
