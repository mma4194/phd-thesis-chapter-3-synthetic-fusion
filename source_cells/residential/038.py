# ==========================================================
# CELL 10.7 — TimeGAN / sequence-GAN optional backend registration — v2-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Register TimeGAN / sequence-GAN availability status in the predefined
#   generator portfolio.
# - Keep disabled unless a tested project-specific backend adapter exists.
#
# Scientific contract:
# - This cell does not fit, generate, select, repair, or overwrite artifacts.
# - No pseudo-TimeGAN fallback is allowed.
# - If disabled or no adapter is configured, it registers a clean unavailable
#   backend with an audit/contract.
# - Any future implementation must fit TRAIN only, generate VAL/TEST candidates,
#   and be selected only downstream on VAL against A1.
# ==========================================================

log("--- START: Cell 10.7 — TimeGAN / sequence-GAN optional backend registration (v2-THESIS) ---")

import os
import json
import time

need = [
    "CFG", "log",
    "_cell10_register_unavailable_backend",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell10.7] Missing prerequisites: {missing}. Run Cell 10.1 first.")

# ----------------------------------------------------------
# 0) Clean-run leakage guard
# ----------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(f"[Cell10.7] Clean TimeGAN registration forbids CFG['{_flag}']=True.")

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell10.7] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

OUTDIR = str(CFG["outdir"])
OUT_ART = os.path.join(OUTDIR, "artifacts")
OUT_REP = os.path.join(OUTDIR, "reports")
CONTRACT_DIR = os.path.join(OUT_ART, "contracts")

for d in [OUT_ART, OUT_REP, CONTRACT_DIR]:
    os.makedirs(d, exist_ok=True)

def _write_atomic_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    os.replace(tmp, path)

enable = bool(CFG.get("cell10_enable_timegan", False))
adapter_name = str(CFG.get("cell10_timegan_adapter", "") or "").strip()

if not enable:
    reason = "Disabled by CFG['cell10_enable_timegan']=False."
    meta = {
        "scientific_note": (
            "TimeGAN should only be enabled after a backend is validated on TRAIN/VAL. "
            "Do not replace it with a pseudo-TimeGAN or block-bootstrap fallback."
        ),
        "how_to_enable": (
            "Set CFG['cell10_enable_timegan']=True and provide a tested "
            "CFG['cell10_timegan_adapter'] implementation."
        ),
    }

elif not adapter_name:
    reason = (
        "TimeGAN requested, but no project-specific backend adapter is configured."
    )
    meta = {
        "required_contract": {
            "fit_split": "TRAIN",
            "generate_VAL_candidate": True,
            "generate_TEST_candidate": True,
            "selection_split": "VAL",
            "test_used_for_selection": False,
            "required_metrics": [
                "transition_rate",
                "burst_rate",
                "lag1_autocorr",
                "KS",
                "Wasserstein",
            ],
        },
        "scientific_note": (
            "No candidate is materialized because TimeGAN APIs differ across libraries. "
            "A reviewed adapter is required before this backend can be active."
        ),
    }

else:
    reason = (
        f"TimeGAN adapter {adapter_name!r} was requested, but Cell 10.7 v2-THESIS "
        "does not include an implementation hook. Add a reviewed adapter cell before enabling."
    )
    meta = {
        "requested_adapter": adapter_name,
        "required_contract": {
            "fit_split": "TRAIN",
            "generate_VAL_candidate": True,
            "generate_TEST_candidate": True,
            "selection_split": "VAL",
            "test_used_for_selection": False,
            "no_pseudo_timegan_fallback": True,
        },
    }

_cell10_register_unavailable_backend(
    "timegan_sequence_gan",
    reason=reason,
    meta={
        "source": "Cell10.7",
        "version": "cell10_7_timegan_sequence_gan_optional_v2_THESIS",
        **meta,
    },
)

audit = {
    "version": "cell10_7_timegan_sequence_gan_optional_v2_THESIS",
    "candidate_available": False,
    "reason": reason,
    "enabled_requested": bool(enable),
    "adapter_name": adapter_name or None,
    "registered_candidate_id": None,
    "materialized_VAL_candidate": False,
    "materialized_TEST_candidate": False,
    "registered_as_unavailable_backend": True,
    "leakage_status": {
        "uses_train_values_for_fitting": False,
        "uses_val_values_for_fitting": False,
        "uses_test_values_for_fitting": False,
        "uses_val_values_for_selection": False,
        "uses_test_values_for_selection": False,
        "uses_test_values_for_repair": False,
        "uses_test_values_for_candidate_promotion": False,
        "materializes_test_candidate": False,
        "overwrites_final_artifact": False,
    },
    "required_future_backend_contract": {
        "fit_split": "TRAIN",
        "selection_split": "VAL",
        "test_usage": "final_QA_only",
        "must_emit_VAL_candidate": True,
        "must_emit_TEST_candidate": True,
        "must_not_use_TEST_for_selection": True,
        "must_not_use_pseudo_timegan_fallback": True,
    },
    "paper_claim_status": (
        "TimeGAN/sequence-GAN is retained as a predefined optional backend but "
        "contributes no active candidate in this run because no validated adapter "
        "is enabled."
    ),
    "ts_unix": float(time.time()),
}

audit_path = os.path.join(OUT_REP, "cell10_7_timegan_sequence_gan_audit_v2_THESIS.json")
contract_path = os.path.join(CONTRACT_DIR, "cell10_7_timegan_sequence_gan_contract_v2_THESIS.json")

_write_atomic_json(audit_path, audit)
_write_atomic_json(contract_path, audit)

globals()["CELL10_7_TIMEGAN_AUDIT"] = audit
globals()["CELL10_7_TIMEGAN_AUDIT_PATH"] = audit_path
globals()["CELL10_7_TIMEGAN_CONTRACT_PATH"] = contract_path

if "RUN_META" in globals():
    RUN_META.setdefault("portfolio_candidate_audits", {})
    RUN_META["portfolio_candidate_audits"]["timegan_sequence_gan"] = audit

log(f"[Cell10.7] TimeGAN/sequence-GAN registered unavailable: {reason}")
log(f"[Cell10.7] Saved audit: {audit_path}")
log(f"[Cell10.7] Saved contract: {contract_path}")
log("--- END: Cell 10.7 — TimeGAN / sequence-GAN optional backend registration (v2-THESIS) ---")