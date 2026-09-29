import json
from pathlib import Path
import pandas as pd

outdir = Path(str(globals().get("OUTDIR", CFG["outdir"]))).expanduser().resolve()
report_dir = Path(str(globals().get("REPORT_DIR", outdir / "reports"))).expanduser().resolve()
contract_dir = outdir / "artifacts" / "contracts"
contract_dir.mkdir(parents=True, exist_ok=True)
summary_path = report_dir / "cell17_4_baseline_scope_summary.csv"
if not summary_path.exists():
    raise RuntimeError("Run Cell17.4 first; missing baseline scope summary.")
summary = pd.read_csv(summary_path)
if "pipeline_comparator_evaluated" not in summary.columns or "scope_id" not in summary.columns:
    raise RuntimeError(f"Baseline summary schema not recognized: {list(summary.columns)}")
summary["pipeline_comparator_evaluated"] = summary["pipeline_comparator_evaluated"].fillna(False).astype(bool)
full_scopes = summary[summary["pipeline_comparator_evaluated"]].copy()
limited_scopes = summary[~summary["pipeline_comparator_evaluated"]].copy()
claim_policy = {
    "cell": "baseline_claim_gate_after_17_4",
    "fully_comparable_scopes": full_scopes["scope_id"].astype(str).tolist(),
    "limited_scopes_no_headline_claim": limited_scopes["scope_id"].astype(str).tolist(),
    "full_CPS_outperforms_all_external_baselines_claim_allowed": False,
    "scope_limited_baseline_claim_allowed": bool(len(full_scopes) > 0),
    "TEST_values_used_for_claim_selection": False,
}
(contract_dir / "baseline_claim_gate_after_17_4_THESIS.json").write_text(
    json.dumps(claim_policy, indent=2, sort_keys=True), encoding="utf-8"
)
print("[Q5] fully comparable scopes:", claim_policy["fully_comparable_scopes"])
print("[Q5] limited scopes excluded from headline claim:", claim_policy["limited_scopes_no_headline_claim"])