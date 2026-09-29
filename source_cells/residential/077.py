# %% CHECK Binary V6 outputs before moving to mask branch

from pathlib import Path
import json

root = Path(OUTDIR).expanduser().resolve()
report_dir = Path(REPORT_DIR).expanduser().resolve()
syn_dir = Path(OUT_SYN).expanduser().resolve()

required = [
    syn_dir / "IOT_FINAL_BINARY_TEST.parquet",
    report_dir / "cell12d_v6_terminal_test_qa_summary.json",
    report_dir / "cell12d_v6_included_claim_terminal_test_qa_metrics.csv",
    report_dir / "cell12d_v6_scope_excluded_terminal_test_diagnostics.csv",
]

for p in required:
    print("FOUND" if p.exists() else "MISSING", p)

summary_path = report_dir / "cell12d_v6_terminal_test_qa_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "included_n": summary.get("included_n_trainval_only"),
        "scope_excluded_n": summary.get("scope_excluded_n_trainval_only"),
        "included_pass": summary.get("included_pass"),
        "included_warning": summary.get("included_warning"),
        "included_fatal": summary.get("included_fatal"),
        "included_publication_blocker_n": summary.get("included_publication_blocker_n"),
        "mean_rate_error_included": summary.get("mean_rate_error_included"),
        "mean_runlength_ks_included": summary.get("mean_runlength_ks_included"),
        "mean_binary_c2st_auc_included": summary.get("mean_binary_c2st_auc_included"),
    }, indent=2))