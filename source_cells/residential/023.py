# ==========================================================
# DDPM CONTRACT BRIDGE — post Cell 8.4 compatibility alias
# Decision: KEEP ONLY IF DOWNSTREAM CELLS REQUIRE DDPM_ACTIVE_CONTRACT
#
# Purpose:
# - Expose the already validated DDPM Cell 8.4 runtime audit as
#   DDPM_ACTIVE_CONTRACT for downstream clean selector cells.
# - Does not train, generate, select, repair, or overwrite anything.
# ==========================================================

log("--- START: DDPM CONTRACT BRIDGE — post Cell 8.4 compatibility alias ---")

import os
import json

need = ["CFG", "log", "DDPM_RUNTIME_AUDIT", "DDPM_CANDIDATE_REGISTRY"]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[DDPM_CONTRACT_BRIDGE] Missing prerequisites: {missing}. Run Cell 8.4 first.")

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")
os.makedirs(CONTRACT_DIR, exist_ok=True)

for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[DDPM_CONTRACT_BRIDGE] Clean run forbids CFG['{_flag}']=True.")

audit = dict(DDPM_RUNTIME_AUDIT)

if audit.get("selection_authority") != "downstream_VAL_selector_only":
    raise RuntimeError(
        "[DDPM_CONTRACT_BRIDGE] Unexpected DDPM selection authority: "
        f"{audit.get('selection_authority')!r}"
    )

if int(audit.get("advisory_pass_cols_n", audit.get("advisory_pass_count", 0))) != 0:
    log(
        "[DDPM_CONTRACT_BRIDGE][NOTE] DDPM has advisory-pass columns; downstream selector must still compare against A1 on VAL."
    )

DDPM_ACTIVE_CONTRACT = {
    "version": "ddpm_active_contract_bridge_post_cell8_4_v1",
    "source_runtime_audit": "DDPM_RUNTIME_AUDIT",
    "ddpm_role": audit.get("ddpm_role", "conditional_refinement_candidate_generator"),
    "selection_authority": "downstream_VAL_selector_only",
    "target_mode": audit.get("target_mode", "absolute_protocol_aware"),
    "D": int(audit.get("D")),
    "cond_dim": int(audit.get("cond_dim")),
    "sig12": str(audit.get("sig12")),
    "candidate_families": list(audit.get("candidate_families", [])),
    "advisory_pass_cols": list(audit.get("advisory_pass_cols", [])),
    "advisory_pass_count": int(audit.get("advisory_pass_cols_n", 0)),
    "advisory_excluded_cols": list(audit.get("advisory_excluded_cols", [])),
    "advisory_excluded_count": int(audit.get("advisory_excluded_cols_n", 0)),
    "direct_safe_overwrite_cols": [],
    "direct_safe_overwrite_cols_n": 0,
    "contributes_to_final_artifact": False,
    "leakage_status": {
        "uses_test_for_selection": False,
        "uses_test_for_repair": False,
        "mutates_synthetic_artifact": False,
        "overwrites_final_artifact": False,
    },
    "paper_claim_status": (
        "DDPM is validated as a candidate-only generator. It is not direct paper evidence "
        "and is not final-contributing unless a downstream VAL-only selector selects it."
    ),
}

globals()["DDPM_ACTIVE_CONTRACT"] = DDPM_ACTIVE_CONTRACT

bridge_path = os.path.join(CONTRACT_DIR, "ddpm_active_contract_bridge_post_cell8_4.json")
with open(bridge_path, "w", encoding="utf-8") as f:
    json.dump(DDPM_ACTIVE_CONTRACT, f, indent=2, ensure_ascii=False)

globals()["DDPM_ACTIVE_CONTRACT_PATH"] = bridge_path

if "RUN_META" in globals():
    RUN_META.setdefault("contract_bridges", {})
    RUN_META["contract_bridges"]["DDPM_ACTIVE_CONTRACT"] = DDPM_ACTIVE_CONTRACT

log(f"[DDPM_CONTRACT_BRIDGE] Saved: {bridge_path}")
log("[DDPM_CONTRACT_BRIDGE] PASS: exposed DDPM_ACTIVE_CONTRACT from Cell 8.4 audit; no modeling or selection performed.")
log("--- END: DDPM CONTRACT BRIDGE ---")