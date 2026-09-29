# ==========================================================
# PRE-CELL-9 MASK POLICY CONTRACT — tier-specific protocol mask targets — v5-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Configure protocol mask-target calibration before Cell 9.
# - Use one single auditable policy artifact consumed by Cell 9.
# - Keep TEST forbidden for mask fitting, target calibration, candidate selection,
#   threshold selection, and repair.
# - Avoid stale mask-policy knobs silently overriding the canonical policy.
#
# Scientific contract:
# - This cell configures mask-target policy only.
# - It does not generate masks.
# - It does not fit protocol-value generators.
# - It does not select candidates.
# - It does not read TEST values.
# - It writes a machine-readable policy contract for Cell 9.
#
# Motivation:
# - P-MASK-01: remove conflicting/stale mask policy settings.
# - P-MASK-02: prevent artificial Zigbee undercoverage.
# ==========================================================

log("--- START: PRE-CELL-9 MASK POLICY CONTRACT v5-THESIS ---")

import os
import json
import time
import copy

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
need = ["CFG", "log"]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[PRE_CELL9_MASK_POLICY] Missing prerequisites: {missing}")

# ----------------------------------------------------------
# 0b) Clean-run leakage guard inherited from Cell 1
# ----------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(
            f"[PRE_CELL9_MASK_POLICY] Clean mask policy forbids CFG['{_flag}']=True."
        )

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[PRE_CELL9_MASK_POLICY] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
REP_DIR = os.path.join(OUTDIR, "reports")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

# ----------------------------------------------------------
# 1) Resolve active protocol-mask tiers
# ----------------------------------------------------------
if "DDPM_TIERS_USED" in globals():
    ACTIVE_MASK_TIERS = list(globals()["DDPM_TIERS_USED"])
elif "PROTOCOL_TIERS_USED" in globals():
    ACTIVE_MASK_TIERS = list(globals()["PROTOCOL_TIERS_USED"])
else:
    raise RuntimeError(
        "[PRE_CELL9_MASK_POLICY] Cannot resolve active mask tiers. "
        "Run the protocol-tier contract cells before this policy cell."
    )

ACTIVE_MASK_TIERS = [str(t).strip().lower() for t in ACTIVE_MASK_TIERS]

CANONICAL_MASK_TIERS = ["router", "ota", "zigbee"]
ALLOWED_TIERS = {"router", "ota", "zigbee"}
ALLOWED_POLICIES = {"quantile_tail", "force_ones_if_high", "conservative_cap"}

if ACTIVE_MASK_TIERS != CANONICAL_MASK_TIERS:
    raise RuntimeError(
        "[PRE_CELL9_MASK_POLICY] Active mask tiers differ from canonical expectation. "
        f"active={ACTIVE_MASK_TIERS} expected={CANONICAL_MASK_TIERS}. "
        "Update the policy explicitly if the tier contract changes."
    )

bad_tiers = [t for t in ACTIVE_MASK_TIERS if t not in ALLOWED_TIERS]
if bad_tiers:
    raise RuntimeError(f"[PRE_CELL9_MASK_POLICY] Unsupported active tiers: {bad_tiers}")

# ----------------------------------------------------------
# 2) Canonical v5 policy
# ----------------------------------------------------------
# Design:
# - TRAIN-only target calibration and TRAIN-only mask-model fitting.
# - VAL remains for candidate/model selection downstream.
# - TEST remains final QA only.
# - Router/OTA are high-coverage tiers.
# - Zigbee is allowed lower but guarded coverage to avoid artificial undercoverage.
MASK_TARGET_POLICY_SPEC = {
    "version": "pre_cell9_mask_policy_v5_THESIS_protocol_realism_guarded",
    "component": "pre_cell9_protocol_mask_policy_contract",
    "component_type": "policy_contract_no_generation",
    "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    "active_mask_tiers": list(ACTIVE_MASK_TIERS),

    "mask_target_policy": "by_tier",

    # Canonical strict setting for STUDY-THESIS clean run.
    "mask_target_base": "train",
    "mask_target_window": 1800,
    "mask_target_tail_frac": 0.25,

    "mask_target_policy_by_tier": {
        "router": "conservative_cap",
        "ota": "force_ones_if_high",
        "zigbee": "conservative_cap",
    },

    "mask_target_q": 0.50,
    "mask_target_q_by_tier": {
        "router": 0.25,
        "ota": 0.50,
        "zigbee": 0.50,
    },

    "mask_target_cap_at_base_mean": True,
    "mask_target_margin_default": 0.0,
    "mask_target_margin_by_tier": {
        "router": 0.0,
        "ota": 0.0,
        "zigbee": 0.0,
    },

    "mask_target_floor_by_tier": {
        "router": 0.9975,
        "ota": 0.9990,
        "zigbee": 0.9000,
    },

    "mask_target_ceiling_by_tier": {
        "router": 0.9995,
        "ota": 1.0,
        "zigbee": 0.9300,
    },

    "mask_force_mean_match": True,
    "mask_force_mean_mode": "boundary",
    "fixA_max_flip_frac": 0.002,

    "ota_force_ones_thr": float(CFG.get("ota_force_ones_thr", 0.999)),

    "mask_model_fit_split": "train",
    "mask_fit_tail_base": "train",
    "mask_model_tail_frac": 0.25,
    "mask_model_min_reg_n": 500,

    "mask_pi_mult": 1.0,
    "mask_pi_mult_by_tier": {},

    "mask_target_tolerance_warn": 0.015,

    "leakage_status": {
        "uses_train_for_target_calibration": True,
        "uses_train_for_mask_model_fit": True,
        "uses_val_for_target_calibration": False,
        "uses_val_for_mask_model_fit": False,
        "uses_test_for_target_calibration": False,
        "uses_test_for_mask_model_fit": False,
        "uses_test_for_threshold_selection": False,
        "uses_test_for_candidate_selection": False,
        "uses_test_for_repair": False,
        "mutates_synthetic_artifact": False,
        "overwrites_final_artifact": False,
    },

    "selection_split_policy": str(CFG.get("selection_split_policy", "")),

    "paper_claim_status": (
        "This cell supports protocol-mask policy traceability only. It does not "
        "support protocol value realism, coupling, utility, or release claims by itself."
    ),

    "rationale": (
        "Mask target calibration and mask-model fitting use TRAIN only in this v5 policy. "
        "VAL remains reserved for downstream protocol candidate selection, and TEST remains "
        "final QA only. Tier-specific floors and ceilings prevent artificially pessimistic "
        "Zigbee coverage while keeping router and OTA near their observed high-coverage behavior."
    ),
}

# ----------------------------------------------------------
# 3) Validation helpers
# ----------------------------------------------------------
def _require_bool(name, value):
    if not isinstance(value, bool):
        raise RuntimeError(
            f"[PRE_CELL9_MASK_POLICY] {name} must be bool, got {type(value).__name__}"
        )


def _require_float_range(name, value, lo, hi, lo_inc=True, hi_inc=True):
    try:
        v = float(value)
    except Exception as e:
        raise RuntimeError(f"[PRE_CELL9_MASK_POLICY] {name} must be numeric. Got {value!r}") from e

    if not (v == v):
        raise RuntimeError(f"[PRE_CELL9_MASK_POLICY] {name} is NaN.")

    ok_lo = v >= lo if lo_inc else v > lo
    ok_hi = v <= hi if hi_inc else v < hi

    if not (ok_lo and ok_hi):
        raise RuntimeError(
            f"[PRE_CELL9_MASK_POLICY] {name} out of range: {v}; expected "
            f"{'[' if lo_inc else '('}{lo}, {hi}{']' if hi_inc else ')'}"
        )


def _validate_policy_spec(spec):
    if spec["mask_target_policy"] != "by_tier":
        raise RuntimeError("[PRE_CELL9_MASK_POLICY] mask_target_policy must be 'by_tier'.")

    if spec["mask_target_base"] != "train":
        raise RuntimeError(
            "[PRE_CELL9_MASK_POLICY] Canonical THESIS policy requires mask_target_base='train'. "
            f"Got {spec['mask_target_base']!r}."
        )

    if spec["mask_model_fit_split"] != "train":
        raise RuntimeError(
            "[PRE_CELL9_MASK_POLICY] Canonical THESIS policy requires mask_model_fit_split='train'. "
            f"Got {spec['mask_model_fit_split']!r}."
        )

    if spec["mask_fit_tail_base"] != "train":
        raise RuntimeError(
            "[PRE_CELL9_MASK_POLICY] Canonical THESIS policy requires mask_fit_tail_base='train'. "
            f"Got {spec['mask_fit_tail_base']!r}."
        )

    for tier in ACTIVE_MASK_TIERS:
        pol = spec["mask_target_policy_by_tier"].get(tier)
        if pol not in ALLOWED_POLICIES:
            raise RuntimeError(f"[PRE_CELL9_MASK_POLICY] Bad policy for {tier}: {pol}")

        floor = spec["mask_target_floor_by_tier"].get(tier)
        ceiling = spec["mask_target_ceiling_by_tier"].get(tier)

        _require_float_range(f"{tier}.floor", floor, 0.0, 1.0)
        _require_float_range(f"{tier}.ceiling", ceiling, 0.0, 1.0)

        if float(floor) > float(ceiling):
            raise RuntimeError(
                f"[PRE_CELL9_MASK_POLICY] floor > ceiling for {tier}: {floor}>{ceiling}"
            )

    _require_bool("mask_force_mean_match", spec["mask_force_mean_match"])
    _require_bool("mask_target_cap_at_base_mean", spec["mask_target_cap_at_base_mean"])
    _require_float_range("fixA_max_flip_frac", spec["fixA_max_flip_frac"], 0.0, 0.01)

    leak = spec.get("leakage_status", {})
    if not isinstance(leak, dict):
        raise RuntimeError("[PRE_CELL9_MASK_POLICY] leakage_status must be a dict.")

    for key in [
        "uses_test_for_target_calibration",
        "uses_test_for_mask_model_fit",
        "uses_test_for_threshold_selection",
        "uses_test_for_candidate_selection",
        "uses_test_for_repair",
        "mutates_synthetic_artifact",
        "overwrites_final_artifact",
    ]:
        if bool(leak.get(key, False)):
            raise RuntimeError(f"[PRE_CELL9_MASK_POLICY] Leakage violation: {key}=True.")

_validate_policy_spec(MASK_TARGET_POLICY_SPEC)

# ----------------------------------------------------------
# 4) Apply policy to CFG and remove stale knobs
# ----------------------------------------------------------
CFG_BEFORE_MASK_POLICY = {
    k: copy.deepcopy(CFG[k])
    for k in list(CFG.keys())
    if str(k).startswith("mask_") or str(k).startswith("fixA_") or str(k).startswith("ota_force_")
}

for k, v in MASK_TARGET_POLICY_SPEC.items():
    if k in {
        "version",
        "component",
        "component_type",
        "created_at_local",
        "active_mask_tiers",
        "leakage_status",
        "selection_split_policy",
        "paper_claim_status",
        "rationale",
    }:
        continue
    CFG[k] = copy.deepcopy(v)

# Explicitly remove older knobs that can silently override this policy.
STALE_MASK_KEYS_REMOVED = []
for stale_key in [
    "mask_target_margin",
    "mask_target_floor",
    "mask_target_ceiling",
    "mask_pi_mult_by_tier_old",
    "mask_target_base_old",
    "mask_model_fit_split_old",
    "mask_fit_tail_base_old",
]:
    if stale_key in CFG:
        STALE_MASK_KEYS_REMOVED.append(stale_key)
        del CFG[stale_key]

CFG_AFTER_MASK_POLICY = {
    k: copy.deepcopy(CFG[k])
    for k in list(CFG.keys())
    if str(k).startswith("mask_") or str(k).startswith("fixA_") or str(k).startswith("ota_force_")
}

MASK_POLICY_APPLICATION_AUDIT = {
    "version": "pre_cell9_mask_policy_application_audit_v5_THESIS",
    "policy_version": MASK_TARGET_POLICY_SPEC["version"],
    "cfg_before_mask_policy": CFG_BEFORE_MASK_POLICY,
    "cfg_after_mask_policy": CFG_AFTER_MASK_POLICY,
    "stale_mask_keys_removed": list(STALE_MASK_KEYS_REMOVED),
    "changed_or_added_keys": sorted(
        set(CFG_AFTER_MASK_POLICY.keys()).union(CFG_BEFORE_MASK_POLICY.keys())
    ),
    "leakage_status": MASK_TARGET_POLICY_SPEC["leakage_status"],
    "ts_unix": float(time.time()),
}

# ----------------------------------------------------------
# 5) Persist artifacts
# ----------------------------------------------------------
policy_path = os.path.join(ARTDIR, "pre_cell9_mask_policy.json")
contract_path = os.path.join(CONTRACT_DIR, "pre_cell9_mask_policy_contract_v5_THESIS.json")
audit_path = os.path.join(REP_DIR, "pre_cell9_mask_policy_application_audit_v5_THESIS.json")

for path, obj in [
    (policy_path, MASK_TARGET_POLICY_SPEC),
    (contract_path, MASK_TARGET_POLICY_SPEC),
    (audit_path, MASK_POLICY_APPLICATION_AUDIT),
]:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    os.replace(tmp, path)

globals()["MASK_TARGET_POLICY_SPEC"] = MASK_TARGET_POLICY_SPEC
globals()["PRE_CELL9_MASK_POLICY_PATH"] = policy_path
globals()["PRE_CELL9_MASK_POLICY_CONTRACT_PATH"] = contract_path
globals()["PRE_CELL9_MASK_POLICY_APPLICATION_AUDIT"] = MASK_POLICY_APPLICATION_AUDIT
globals()["PRE_CELL9_MASK_POLICY_APPLICATION_AUDIT_PATH"] = audit_path

if "RUN_META" in globals():
    RUN_META.setdefault("policy_contracts", {})
    RUN_META["policy_contracts"]["pre_cell9_mask_policy"] = MASK_TARGET_POLICY_SPEC
    RUN_META.setdefault("policy_application_audits", {})
    RUN_META["policy_application_audits"]["pre_cell9_mask_policy"] = MASK_POLICY_APPLICATION_AUDIT

log(
    "[PRE_CELL9_MASK_POLICY] Applied mask policy | "
    f"target_base={CFG['mask_target_base']} | fit_split={CFG['mask_model_fit_split']} | "
    f"policy_by_tier={CFG['mask_target_policy_by_tier']} | "
    f"floors={CFG['mask_target_floor_by_tier']} | ceilings={CFG['mask_target_ceiling_by_tier']}"
)
log(f"[PRE_CELL9_MASK_POLICY] Wrote Cell 9 policy artifact: {policy_path}")
log(f"[PRE_CELL9_MASK_POLICY] Wrote contract: {contract_path}")
log(f"[PRE_CELL9_MASK_POLICY] Wrote application audit: {audit_path}")
log("[PRE_CELL9_MASK_POLICY] PASS: policy configured only; no masks generated and no TEST used.")
log("--- END: PRE-CELL-9 MASK POLICY CONTRACT v5-THESIS ---")