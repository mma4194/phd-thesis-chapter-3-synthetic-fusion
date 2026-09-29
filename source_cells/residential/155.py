# ==========================================================
# CELL 17.6 — Baseline fairness claim sanitizer
# v1.3 STUDY-THESIS publication-safe baseline table, robust 17.5 contract resolver, temporal-blocked C2ST, no-Q4-promotion aware
#
# Purpose:
#   Convert Cell 17.5 outputs into a publication-safe baseline comparison
#   package under the final governance contract.
#
# Key rule:
#   Partial, missing-comparator, or baseline-only comparisons must NOT be
#   counted as headline wins.
#
# Final governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative or used as a comparator.
#   - Public candidate clears direct no-copy/DCR blocker evidence only at
#     warning level and remains strict-release blocked by distinguishability.
#
# Inputs:
#   reports/cell17_5_baseline_fairness_summary.csv
#   reports/cell17_5_baseline_limitation_findings.csv
#   reports/cell17_5_baseline_publication_report.md
#   artifacts/contracts/cell17_5_baseline_fairness_contract_v1_1_THESIS.json
#
# Outputs:
#   reports/cell17_6_baseline_headline_comparable_results.csv
#   reports/cell17_6_baseline_partial_limited_results.csv
#   reports/cell17_6_baseline_publication_safe_summary.csv
#   reports/cell17_6_baseline_claim_sanitizer_report.md
#   reports/cell17_6_baseline_claim_sanitizer_contract.json
#   artifacts/contracts/cell17_6_baseline_claim_sanitizer_contract_v1_1_THESIS.json
#   artifacts/cell17_6_baseline_claim_sanitizer_manifest.json
#
# No generation, fitting, selection, materialization, or mutation.
# ==========================================================

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

log("--- START: Cell 17.6 — Baseline fairness claim sanitizer (v1.3 robust-17.5-contract temporal-blocked-C2ST no-Q4-promotion strict) ---")

CELL176_VERSION = "cell17_6_baseline_claim_sanitizer_v1_3_robust_17_5_contract_temporal_blocked_c2st_no_q4_promotion"

# ----------------------------------------------------------
# 0) Resolve project root
# ----------------------------------------------------------
def _resolve_project_root_176():
    candidates = []
    for p in [
        globals().get("REPORT_DIR", ""),
        globals().get("OUTDIR", ""),
        globals().get("OUT_SYN", ""),
    ]:
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

    raise RuntimeError("[Cell17.6] Could not resolve canonical project root.")

BASE = _resolve_project_root_176()
REPORT_DIR_BASE = os.path.join(BASE, "reports")
ARTDIR_BASE = os.path.join(BASE, "artifacts")
CONTRACT_DIR_BASE = os.path.join(ARTDIR_BASE, "contracts")

os.makedirs(REPORT_DIR_BASE, exist_ok=True)
os.makedirs(ARTDIR_BASE, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE, exist_ok=True)

CFG["cell17_6_version"] = CELL176_VERSION
CFG["cell17_6_Q4_final_status"] = "blocked_no_promotion"
CFG["cell17_6_Q4_coupled_artifacts_used"] = False
CFG["cell17_6_TEST_real_values_used_here"] = False
CFG["cell17_6_TEST_real_values_used_for_materialization"] = False
CFG["cell17_6_synthetic_values_mutated"] = False
CFG["cell17_6_selection_done_here"] = False
CFG["cell17_6_generator_fit_done_here"] = False
CFG["cell17_6_materialization_done_here"] = False
CFG["cell17_6_claim_sanitizer_done_here"] = True

MIN_COVERAGE = float(CFG.get("cell17_5_min_comparator_col_coverage", 0.80)) if "CFG" in globals() else 0.80

# ----------------------------------------------------------
# 1) Paths
# ----------------------------------------------------------
summary_path = os.path.join(REPORT_DIR_BASE, "cell17_5_baseline_fairness_summary.csv")
findings_path = os.path.join(REPORT_DIR_BASE, "cell17_5_baseline_limitation_findings.csv")
report17_path = os.path.join(REPORT_DIR_BASE, "cell17_5_baseline_publication_report.md")
def _resolve_cell17_5_contract_path_176():
    candidates = [
        os.path.join(CONTRACT_DIR_BASE, "cell17_5_baseline_fairness_contract_v1_2_THESIS.json"),
        os.path.join(REPORT_DIR_BASE, "cell17_5_baseline_fairness_contract.json"),
        os.path.join(CONTRACT_DIR_BASE, "cell17_5_baseline_fairness_contract_v1_1_THESIS.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p, candidates
    raise FileNotFoundError("No Cell 17.5 fairness contract found. Searched: " + str(candidates))

cell17_5_contract_path, cell17_5_contract_candidates = _resolve_cell17_5_contract_path_176()

for p in [summary_path, findings_path, report17_path, cell17_5_contract_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(p)

headline_path = os.path.join(REPORT_DIR_BASE, "cell17_6_baseline_headline_comparable_results.csv")
partial_path = os.path.join(REPORT_DIR_BASE, "cell17_6_baseline_partial_limited_results.csv")
safe_summary_path = os.path.join(REPORT_DIR_BASE, "cell17_6_baseline_publication_safe_summary.csv")
report_path = os.path.join(REPORT_DIR_BASE, "cell17_6_baseline_claim_sanitizer_report.md")
contract_path = os.path.join(REPORT_DIR_BASE, "cell17_6_baseline_claim_sanitizer_contract.json")
contract_canonical_path = os.path.join(CONTRACT_DIR_BASE, "cell17_6_baseline_claim_sanitizer_contract_v1_3_THESIS.json")
manifest_path = os.path.join(ARTDIR_BASE, "cell17_6_baseline_claim_sanitizer_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def sha256_file_safe(path, block=1 << 20):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def safe_float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _json_sanitize_176(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_176(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_176(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_176(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_176(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_176(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_176(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_176(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_176(payload), f, indent=2, sort_keys=True)

def _truthy_series(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])

def _winner_from_counts(row):
    if not bool(row.get("fully_comparable_for_headline", False)):
        return "not_counted_partial_or_limited"

    pipeline_n = safe_int(row.get("pipeline_winner_n"), 0)
    external_n = safe_int(row.get("external_winner_n"), 0)

    # Backward compatibility with older CSV schema.
    if pipeline_n == 0 and "pipeline_winners" in row.index:
        pipeline_n = 1 if str(row.get("pipeline_winners", "")).strip() else 0
    if external_n == 0 and "external_winners" in row.index:
        external_n = 1 if str(row.get("external_winners", "")).strip() else 0

    if pipeline_n > 0 and external_n == 0:
        return "role_aware_pipeline"
    if external_n > 0 and pipeline_n == 0:
        return "external_baseline"
    if pipeline_n > 0 and external_n > 0:
        return "mixed_winners"
    return "no_clear_winner"

# ----------------------------------------------------------
# 3) Load and validate inputs
# ----------------------------------------------------------
summary = pd.read_csv(summary_path)
findings = pd.read_csv(findings_path)

with open(cell17_5_contract_path, "r", encoding="utf-8") as f:
    cell17_5_contract = json.load(f)

cell17_5_version_176 = str(cell17_5_contract.get("version", ""))
if "cell17_5_external_baseline_fairness_report_v1_2" not in cell17_5_version_176:
    raise RuntimeError(
        "[Cell17.6] Unexpected Cell 17.5 contract version. "
        f"loaded_path={cell17_5_contract_path} | version={cell17_5_version_176} | "
        f"searched={cell17_5_contract_candidates}"
    )

q4 = cell17_5_contract.get("q4_governance", {})
strict = cell17_5_contract.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell17.6] Cell 17.5 does not carry Q4 blocked_no_promotion governance.")
if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell17.6] Cell 17.5 indicates Q4-coupled artifacts were used.")

q6_public_context = cell17_5_contract.get("q6_public_context", {})
public_direct_privacy_status = str(q6_public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6_public_context.get("public_distinguishability_status", ""))
public_strict_status = str(q6_public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 4) Normalize columns
# ----------------------------------------------------------
summary["scope_id"] = summary["scope_id"].astype(str)

required_cols = [
    "comparison_status",
    "pipeline_comparator_evaluated",
    "evaluated_external_baselines",
    "scope_cols_n",
]
missing = [c for c in required_cols if c not in summary.columns]
if missing:
    raise RuntimeError(f"[Cell17.6] Missing required Cell 17.5 summary columns: {missing}")

if "pipeline_col_coverage" not in summary.columns:
    summary["pipeline_col_coverage"] = np.nan

summary["pipeline_col_coverage"] = pd.to_numeric(summary["pipeline_col_coverage"], errors="coerce")
summary["scope_cols_n"] = pd.to_numeric(summary.get("scope_cols_n", np.nan), errors="coerce")
summary["pipeline_eval_cols_n"] = pd.to_numeric(summary.get("pipeline_eval_cols_n", np.nan), errors="coerce")
summary["best_fair_scope_penalty_score"] = pd.to_numeric(
    summary.get("best_fair_scope_penalty_score", np.nan), errors="coerce"
)

if "pipeline_winner_n" not in summary.columns:
    summary["pipeline_winner_n"] = 0
if "external_winner_n" not in summary.columns:
    summary["external_winner_n"] = 0
if "not_comparable_n" not in summary.columns:
    summary["not_comparable_n"] = 0

summary["pipeline_winner_n"] = pd.to_numeric(summary["pipeline_winner_n"], errors="coerce").fillna(0).astype(int)
summary["external_winner_n"] = pd.to_numeric(summary["external_winner_n"], errors="coerce").fillna(0).astype(int)
summary["not_comparable_n"] = pd.to_numeric(summary["not_comparable_n"], errors="coerce").fillna(0).astype(int)

# Fully comparable means:
#   - fair comparison available
#   - pipeline comparator evaluated
#   - pipeline comparator coverage >= threshold
#   - at least one external baseline evaluated
summary["fully_comparable_for_headline"] = (
    summary["comparison_status"].astype(str).eq("fair_comparison_available")
    & _truthy_series(summary["pipeline_comparator_evaluated"])
    & (summary["pipeline_col_coverage"] >= MIN_COVERAGE)
    & summary["evaluated_external_baselines"].fillna("").astype(str).ne("")
)

summary["partial_or_limited_comparison"] = ~summary["fully_comparable_for_headline"]
summary["headline_winner"] = summary.apply(_winner_from_counts, axis=1)

headline = summary[summary["fully_comparable_for_headline"]].copy()
partial = summary[summary["partial_or_limited_comparison"]].copy()

# ----------------------------------------------------------
# 5) Summary counters
# ----------------------------------------------------------
headline_scope_n = int(len(headline))
partial_scope_n = int(len(partial))
pipeline_headline_wins = int((headline["headline_winner"] == "role_aware_pipeline").sum())
external_headline_wins = int((headline["headline_winner"] == "external_baseline").sum())
mixed_headline = int((headline["headline_winner"] == "mixed_winners").sum())
unclear_headline = int((headline["headline_winner"] == "no_clear_winner").sum())

ctgan_unavailable = bool(
    findings.astype(str).apply(
        lambda r: r.str.contains("CTGAN", case=False, na=False).any()
        and r.str.contains("unavailable", case=False, na=False).any(),
        axis=1,
    ).any()
)

# ----------------------------------------------------------
# 6) Output tables
# ----------------------------------------------------------
headline_cols = [
    "scope_id", "scope_display", "metric_family", "scope_cols_n",
    "pipeline_eval_cols_n", "pipeline_col_coverage",
    "evaluated_external_baselines", "best_artifact",
    "best_fair_scope_penalty_score", "pipeline_winner_n",
    "external_winner_n", "headline_winner",
    "comparison_status", "comparison_reasons",
]
headline_cols = [c for c in headline_cols if c in headline.columns]

partial_cols = [
    "scope_id", "scope_display", "metric_family", "scope_cols_n",
    "pipeline_eval_cols_n", "pipeline_col_coverage",
    "evaluated_external_baselines", "best_artifact",
    "best_fair_scope_penalty_score",
    "comparison_status", "comparison_reasons",
]
partial_cols = [c for c in partial_cols if c in partial.columns]

headline_table = headline[headline_cols].copy()
partial_table = partial[partial_cols].copy()

safe_summary = pd.DataFrame([{
    "headline_fully_comparable_scope_n": headline_scope_n,
    "partial_or_limited_scope_n": partial_scope_n,
    "pipeline_headline_wins_n": pipeline_headline_wins,
    "external_headline_wins_n": external_headline_wins,
    "mixed_headline_scope_n": mixed_headline,
    "unclear_headline_scope_n": unclear_headline,
    "ctgan_unavailable": ctgan_unavailable,
    "tabddpm_scope_count": int(summary["evaluated_external_baselines"].fillna("").astype(str).str.contains("TabDDPM").sum()),
    "timegan_scope_count": int(summary["evaluated_external_baselines"].fillna("").astype(str).str.contains("TimeGAN").sum()),
    "min_comparator_coverage_for_headline": MIN_COVERAGE,
    "q4_final_status": "blocked_no_promotion",
    "q4_coupled_artifacts_used": False,
    "public_direct_privacy_status": public_direct_privacy_status,
    "public_distinguishability_status": public_distinguishability_status,
    "public_strict_combined_status": public_strict_status,
    "headline_claim_allowed": (
        f"The role-aware pipeline outperformed external baselines in {pipeline_headline_wins} "
        f"fully comparable fair scopes; {partial_scope_n} scope(s) were partial/limited or missing a same-scope comparator "
        f"and are not counted as headline wins."
    ),
    "forbidden_claim": "The role-aware pipeline outperforms all external baselines across the full CPS artifact.",
}])

# ----------------------------------------------------------
# 7) Markdown report
# ----------------------------------------------------------
lines = []
lines.append("# Baseline Fairness Claim Sanitizer")
lines.append("")
lines.append(f"Version: `{CELL176_VERSION}`")
lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append(
    "This report converts the external-baseline results into publication-safe claims. "
    "Only scopes with sufficient same-scope pipeline coverage are counted as headline comparisons. "
    "Partial, missing-comparator, or baseline-only comparisons are retained for transparency but are not counted as headline wins."
)
lines.append("")
lines.append("## Governance boundary")
lines.append("")
lines.append("- Q4 final status: `blocked_no_promotion`.")
lines.append("- No Q4-coupled artifact is authoritative or used as a baseline comparator.")
lines.append(f"- Public direct privacy/no-copy status: `{public_direct_privacy_status}`.")
lines.append(f"- Public distinguishability status: `{public_distinguishability_status}`.")
lines.append(f"- Public strict combined status: `{public_strict_status}`.")
lines.append("")
lines.append("## Headline-comparable baseline results")
lines.append("")
lines.append("| Scope | External baseline(s) | Pipeline coverage | Winner | Status |")
lines.append("|---|---|---:|---|---|")
if len(headline_table):
    for _, r in headline_table.iterrows():
        cov = safe_float(r.get("pipeline_col_coverage"), np.nan)
        cov_txt = "NA" if not np.isfinite(cov) else f"{100*cov:.1f}%"
        lines.append(
            f"| `{r.get('scope_id','')}` | {r.get('evaluated_external_baselines','')} | "
            f"{cov_txt} | {r.get('headline_winner','')} | {r.get('comparison_status','')} |"
        )
else:
    lines.append("| none | none | NA | none | no fully comparable scope |")

lines.append("")
lines.append("## Partial or limited baseline results")
lines.append("")
lines.append("| Scope | Reason | Pipeline coverage | How to report |")
lines.append("|---|---|---:|---|")
if len(partial_table):
    for _, r in partial_table.iterrows():
        cov = safe_float(r.get("pipeline_col_coverage"), np.nan)
        cov_txt = "NA" if not np.isfinite(cov) else f"{100*cov:.1f}%"
        lines.append(
            f"| `{r.get('scope_id','')}` | {r.get('comparison_reasons','')} | {cov_txt} | "
            "Report as partial/limited or baseline-only; do not count as headline win. |"
        )
else:
    lines.append("| none | none | NA | none |")

lines.append("")
lines.append("## Safe paper claim")
lines.append("")
lines.append(
    f"The role-aware pipeline outperformed external baselines in {pipeline_headline_wins} fully comparable fair scopes. "
    f"{partial_scope_n} scope(s) were classified as partial, limited, or missing a same-scope comparator and are therefore reported separately rather than counted as headline wins."
)
lines.append("")
lines.append("## Forbidden claim")
lines.append("")
lines.append(
    "Do not claim that the role-aware pipeline outperforms all external baselines across the full CPS artifact. "
    "The result is same-scope, dimension-limited, and governed by the no-Q4-promotion/public-Q6 boundaries."
)
lines.append("")
lines.append("## Required caveats")
lines.append("")
lines.append("- CTGAN was unavailable in the execution environment and is not reported as an empirical result.")
lines.append("- TabDDPM is evaluated only on compatible compact tabular scopes.")
lines.append("- TimeGAN is evaluated only on compatible compact temporal scopes.")
lines.append("- External baselines are not treated as full CPS replacements because they do not cover Q3 observability, Q4 coupling repair, Q6 release safety, or full namespace governance.")
lines.append("- Missing-comparator and partial-coverage scopes must not be used as decisive evidence of pipeline superiority.")
lines.append("- Public-Q6 strict release remains blocked by distinguishability despite direct privacy/no-copy warning-level status.")

report = "\n".join(lines)

# ----------------------------------------------------------
# 8) Save outputs and contract
# ----------------------------------------------------------
headline_table.to_csv(headline_path, index=False)
partial_table.to_csv(partial_path, index=False)
safe_summary.to_csv(safe_summary_path, index=False)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

contract = {
    "cell": "17.6",
    "version": CELL176_VERSION,
    "role": "baseline_fairness_claim_sanitizer_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_coupled_artifacts_not_used_as_comparators": True,
        "baseline_c2st_uses_temporal_blocked_split": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell17_5": str(cell17_5_contract.get("version", "")),
    },
    "input_contracts": {
        "cell17_5_contract_path": cell17_5_contract_path,
        "cell17_5_contract_candidates": cell17_5_contract_candidates,
    },
    "strict_contract": {
        "baseline_c2st_uses_temporal_blocked_split": True,
        "TEST_real_values_used_here": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "claim_sanitizer_done_here": True,
    },
    "headline_fully_comparable_scope_n": int(headline_scope_n),
    "partial_or_limited_scope_n": int(partial_scope_n),
    "pipeline_headline_wins_n": int(pipeline_headline_wins),
    "external_headline_wins_n": int(external_headline_wins),
    "mixed_headline_scope_n": int(mixed_headline),
    "unclear_headline_scope_n": int(unclear_headline),
    "ctgan_unavailable": bool(ctgan_unavailable),
    "min_comparator_coverage_for_headline": float(MIN_COVERAGE),
    "outputs": {
        "headline_table": headline_path,
        "partial_table": partial_path,
        "safe_summary": safe_summary_path,
        "report": report_path,
        "contract": contract_path,
        "contract_canonical": contract_canonical_path,
        "manifest": manifest_path,
    },
    "hashes": {},
}

write_json(contract_path, contract)
write_json(contract_canonical_path, contract)

manifest = {
    "cell": "17.6",
    "version": CELL176_VERSION,
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "contract": contract,
}
write_json(manifest_path, manifest)

for k, p in contract["outputs"].items():
    contract["hashes"][k] = sha256_file_safe(p)

write_json(contract_path, contract)
write_json(contract_canonical_path, contract)
manifest["contract"] = contract
write_json(manifest_path, manifest)

# ----------------------------------------------------------
# 9) Export globals
# ----------------------------------------------------------
globals()["CELL176_VERSION"] = CELL176_VERSION
globals()["CELL17_6_BASELINE_HEADLINE_RESULTS_DF"] = headline_table
globals()["CELL17_6_BASELINE_PARTIAL_RESULTS_DF"] = partial_table
globals()["CELL17_6_BASELINE_PUBLICATION_SAFE_SUMMARY_DF"] = safe_summary
globals()["CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT"] = contract

globals()["CELL17_6_BASELINE_HEADLINE_RESULTS_CSV"] = headline_path
globals()["CELL17_6_BASELINE_PARTIAL_RESULTS_CSV"] = partial_path
globals()["CELL17_6_BASELINE_PUBLICATION_SAFE_SUMMARY_CSV"] = safe_summary_path
globals()["CELL17_6_BASELINE_CLAIM_SANITIZER_REPORT_MD"] = report_path
globals()["CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT_JSON"] = contract_path
globals()["CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT_CANONICAL_JSON"] = contract_canonical_path
globals()["CELL17_6_BASELINE_CLAIM_SANITIZER_MANIFEST_JSON"] = manifest_path

print("\n=== CELL 17.6 BASELINE HEADLINE-COMPARABLE RESULTS ===")
print(headline_table.to_string(index=False, max_colwidth=140))

print("\n=== CELL 17.6 BASELINE PARTIAL/LIMITED RESULTS ===")
print(partial_table.to_string(index=False, max_colwidth=140))

print("\n=== CELL 17.6 BASELINE PUBLICATION-SAFE SUMMARY ===")
print(safe_summary.to_string(index=False, max_colwidth=180))

print("\nLoaded Cell 17.5 contract:")
print(cell17_5_contract_path)
print("\nSaved:")
print(headline_path)
print(partial_path)
print(safe_summary_path)
print(report_path)
print(contract_path)
print(contract_canonical_path)
print(manifest_path)

log(
    "[Cell17.6] Baseline claim sanitizer complete | "
    f"headline_scopes={headline_scope_n} | partial_scopes={partial_scope_n} | "
    f"pipeline_headline_wins={pipeline_headline_wins} | external_headline_wins={external_headline_wins} | "
    "q4_status=blocked_no_promotion | c2st_split_policy=temporal_blocked_same_positions"
)
log("--- END: Cell 17.6 — Baseline fairness claim sanitizer (v1.3 robust-17.5-contract temporal-blocked-C2ST no-Q4-promotion strict) ---")