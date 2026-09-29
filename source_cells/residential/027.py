# ==========================================================
# CELL 10.2 — DDPM portfolio availability registration — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Register DDPM into the Cell 10 portfolio audit trail.
# - Do not materialize DDPM candidates when the locked Cell 8/8.4
#   advisory diagnostic reports zero advisory-pass columns.
#
# Scientific contract:
# - DDPM was trained/evaluated upstream using TRAIN/VAL only.
# - Current run has zero DDPM advisory-pass columns.
# - Therefore DDPM is negative evidence and is not materialized as an
#   active A2 portfolio candidate.
# - TEST is not used for DDPM selection, repair, or candidate promotion.
# ==========================================================

log("--- START: Cell 10.2 — DDPM portfolio availability registration (v2-THESIS) ---")

import os
import json
import time

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
need = [
    "CFG", "log",
    "_cell10_register_unavailable_backend",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.2] Missing prerequisites: {missing}. Run Cells 8.4 and 10.1 first.")

# ----------------------------------------------------------
# 0b) Clean-run leakage guard
# ----------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[Cell10.2] Clean DDPM portfolio registration forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.2] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

OUTDIR = str(CFG["outdir"])
OUT_ART = os.path.join(OUTDIR, "artifacts")
OUT_REP = os.path.join(OUTDIR, "reports")
CONTRACT_DIR = os.path.join(OUT_ART, "contracts")

os.makedirs(OUT_ART, exist_ok=True)
os.makedirs(OUT_REP, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1) Resolve DDPM advisory status from active globals/artifacts
# ----------------------------------------------------------
ddpm_runtime_audit_path = os.path.join(OUT_ART, "ddpm_runtime_audit_cell8_4.json")
ddpm_active_contract_path = os.path.join(OUT_ART, "ddpm_active_contract_from_cell8.json")

runtime_audit = {}
active_contract = {}

if "DDPM_RUNTIME_AUDIT" in globals() and isinstance(globals()["DDPM_RUNTIME_AUDIT"], dict):
    runtime_audit = dict(globals()["DDPM_RUNTIME_AUDIT"])
elif os.path.exists(ddpm_runtime_audit_path):
    with open(ddpm_runtime_audit_path, "r", encoding="utf-8") as f:
        runtime_audit = json.load(f)

if "DDPM_ACTIVE_CONTRACT" in globals() and isinstance(globals()["DDPM_ACTIVE_CONTRACT"], dict):
    active_contract = dict(globals()["DDPM_ACTIVE_CONTRACT"])
elif os.path.exists(ddpm_active_contract_path):
    with open(ddpm_active_contract_path, "r", encoding="utf-8") as f:
        active_contract = json.load(f)

if not runtime_audit and not active_contract:
    raise RuntimeError(
        "[Cell10.2] Cannot find DDPM runtime/active contract. "
        "Run Cell 8.4 first."
    )

source = runtime_audit if runtime_audit else active_contract

ddpm_role = str(
    source.get("ddpm_role", source.get("role", "conditional_refinement_candidate_generator"))
)
selection_authority = str(source.get("selection_authority", "downstream_VAL_selector_only"))

if ddpm_role != "conditional_refinement_candidate_generator":
    raise RuntimeError(f"[Cell10.2] Unexpected DDPM role: {ddpm_role!r}")

if selection_authority != "downstream_VAL_selector_only":
    raise RuntimeError(f"[Cell10.2] Unexpected DDPM selection authority: {selection_authority!r}")

# Accept several possible field names from Cell 8/8.4 contracts.
advisory_pass_cols = list(
    source.get(
        "advisory_pass_cols",
        globals().get("DDPM_ADVISORY_PASS_COLS", []),
    )
    or []
)

advisory_pass_count = int(
    source.get(
        "advisory_pass_cols_n",
        source.get(
            "advisory_pass_count",
            len(advisory_pass_cols),
        ),
    )
    or 0
)

advisory_excluded_count = int(
    source.get(
        "advisory_excluded_cols_n",
        source.get("advisory_excluded_count", 0),
    )
    or 0
)

candidate_families = list(source.get("candidate_families", []))

ddpm_cols = list(globals().get("ddpm_cols_use", source.get("ddpm_cols_use", [])) or [])
ddpm_sig12 = str(source.get("sig12", globals().get("ddpm_sig12", "")) or "")
ddpm_D = int(source.get("D", len(ddpm_cols)) or len(ddpm_cols))

if ddpm_D <= 0:
    raise RuntimeError("[Cell10.2] DDPM D is invalid or unavailable.")

# ----------------------------------------------------------
# 2) Canonical current-run behavior: skip materialization if zero pass
# ----------------------------------------------------------
CELL10_2_DDPM_AUDIT = {
    "version": "cell10_2_ddpm_portfolio_registration_v2_THESIS",
    "component": "DDPM_portfolio_availability_registration",
    "ddpm_role": ddpm_role,
    "selection_authority": selection_authority,
    "sig12": ddpm_sig12,
    "D": int(ddpm_D),
    "ddpm_cols_n": int(len(ddpm_cols)),
    "candidate_families": candidate_families,
    "advisory_pass_cols": advisory_pass_cols,
    "advisory_pass_count": int(advisory_pass_count),
    "advisory_excluded_count": int(advisory_excluded_count),
    "materialized_VAL_candidate": False,
    "materialized_TEST_candidate": False,
    "registered_as_active_candidate": False,
    "registered_as_unavailable_backend": False,
    "reason": None,
    "leakage_status": {
        "uses_train_fitted_model_from_upstream": True,
        "uses_val_diagnostic_from_upstream": True,
        "uses_test_for_selection": False,
        "uses_test_for_fitting": False,
        "uses_test_for_repair": False,
        "uses_test_for_candidate_promotion": False,
        "materializes_test_candidate": False,
        "overwrites_final_artifact": False,
    },
    "paper_claim_status": (
        "DDPM is retained as evaluated negative evidence for this run. "
        "It contributes no active A2 candidate because the upstream VAL advisory "
        "diagnostic found zero pass columns."
    ),
    "ts_unix": float(time.time()),
}

if advisory_pass_count <= 0:
    reason = (
        "DDPM skipped: upstream Cell 8/8.4 advisory diagnostic found zero "
        "advisory-pass columns; all DDPM columns were rejected before portfolio "
        "materialization."
    )

    _cell10_register_unavailable_backend(
        "ddpm",
        reason=reason,
        meta={
            "source": "Cell10.2",
            "sig12": ddpm_sig12,
            "D": int(ddpm_D),
            "advisory_pass_count": int(advisory_pass_count),
            "advisory_excluded_count": int(advisory_excluded_count),
            "candidate_families": candidate_families,
            "note": (
                "This is not a runtime failure. It is the expected canonical "
                "outcome for the current DDPM diagnostic: DDPM remains negative "
                "evidence and A1 remains the mandatory baseline."
            ),
        },
    )

    CELL10_2_DDPM_AUDIT["registered_as_unavailable_backend"] = True
    CELL10_2_DDPM_AUDIT["reason"] = reason

    log(
        "[Cell10.2] DDPM not materialized | "
        f"advisory_pass={advisory_pass_count} | "
        f"advisory_excluded={advisory_excluded_count} | "
        "registered as unavailable/skipped negative-evidence backend."
    )

else:
    raise RuntimeError(
        "[Cell10.2] DDPM has advisory-pass columns in this run, but canonical "
        "materialization is not enabled in this cleaned v2 cell. If this happens, "
        "add a separate reviewed materializer that only materializes advisory-pass "
        "columns and validates conditioning leakage before writing VAL/TEST candidates."
    )

# ----------------------------------------------------------
# 3) Persist audit/contract
# ----------------------------------------------------------
audit_path = os.path.join(OUT_REP, "cell10_2_ddpm_portfolio_registration_audit_v2_THESIS.json")
contract_path = os.path.join(CONTRACT_DIR, "cell10_2_ddpm_portfolio_registration_contract_v2_THESIS.json")

for path, obj in [
    (audit_path, CELL10_2_DDPM_AUDIT),
    (contract_path, CELL10_2_DDPM_AUDIT),
]:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    os.replace(tmp, path)

globals()["CELL10_2_DDPM_AUDIT"] = CELL10_2_DDPM_AUDIT
globals()["CELL10_2_DDPM_AUDIT_PATH"] = audit_path
globals()["CELL10_2_DDPM_CONTRACT_PATH"] = contract_path

if "RUN_META" in globals():
    RUN_META.setdefault("portfolio_candidate_audits", {})
    RUN_META["portfolio_candidate_audits"]["ddpm"] = CELL10_2_DDPM_AUDIT

log(f"[Cell10.2] Saved audit: {audit_path}")
log(f"[Cell10.2] Saved contract: {contract_path}")
log("--- END: Cell 10.2 — DDPM portfolio availability registration (v2-THESIS) ---")