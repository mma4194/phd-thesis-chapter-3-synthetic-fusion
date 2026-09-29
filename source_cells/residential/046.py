from pathlib import Path
import json

if "CFG" not in globals() or not isinstance(CFG, dict):
    raise RuntimeError("Run Cell 1 first so CFG is defined.")

OUTDIR_PREFLIGHT = Path(str(globals().get("OUTDIR", CFG["outdir"]))).expanduser().resolve()
ARTDIR_PREFLIGHT = OUTDIR_PREFLIGHT / "artifacts"
REPDIR_PREFLIGHT = OUTDIR_PREFLIGHT / "reports"
SYNDIR_PREFLIGHT = OUTDIR_PREFLIGHT / "synthetic"
CONTRACT_PREFLIGHT = ARTDIR_PREFLIGHT / "contracts"
CONTRACT_PREFLIGHT.mkdir(parents=True, exist_ok=True)

forbidden_noncanonical = [
    ARTDIR_PREFLIGHT / "cell11b_protocol_test_regression_quarantine_manifest.json",
    REPDIR_PREFLIGHT / "cell11b_protocol_test_regression_quarantine_audit.csv",
    REPDIR_PREFLIGHT / "cell11b_protocol_test_regression_quarantine_summary.csv",
    REPDIR_PREFLIGHT / "cell11b_a2_materialization_by_col_post_quarantine.csv",
    SYNDIR_PREFLIGHT / "A2_PROTOCOL_FINAL_BEFORE_11B_QUARANTINE.parquet",
    SYNDIR_PREFLIGHT / "A2_PROTOCOL_FINAL_QA_QUARANTINED.parquet",
    SYNDIR_PREFLIGHT / "PROTOCOL_SYN_TEST_ACCEPTED_ZIGBEE_A0.parquet",
    SYNDIR_PREFLIGHT / "CPS_ACCEPTED_ZIGBEE_A0_TEST.parquet",
    SYNDIR_PREFLIGHT / "PROTOCOL_SYN_TEST_COUPLED_ZIGBEE_FINAL.parquet",
    SYNDIR_PREFLIGHT / "CPS_SYNTHETIC_TEST_Q4_COUPLED_FINAL.parquet",
]
present = [str(p) for p in forbidden_noncanonical if p.exists()]
if present:
    raise RuntimeError(
        "Stale TEST-derived/quarantine/promoted artifacts exist in this OUTDIR. "
        "Do not archive them inside the notebook. Start from a clean CPS_OUTDIR or manually move them outside the submitted artifact.\n"
        + "\n".join(present)
    )

contract = {
    "cell": "stale_artifact_preflight_replacement",
    "status": "pass_clean_outdir",
    "forbidden_checked_n": len(forbidden_noncanonical),
    "forbidden_present_n": len(present),
    "test_derived_artifacts_archived_by_notebook": False,
}
(CONTRACT_PREFLIGHT / "stale_artifact_preflight_contract_THESIS.json").write_text(
    json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8"
)
print("[preflight] clean canonical OUTDIR: no stale TEST-derived promoted/quarantine artifacts found")