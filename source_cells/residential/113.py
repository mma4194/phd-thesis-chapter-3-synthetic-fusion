# %% CELL Q4.1R — VAL-only Q4 acceptance gate, empty-registry safe
# Purpose:
#   Decide whether Q4 can be accepted before terminal TEST.
#
# Safety:
#   - Uses only the VAL-frozen pair registry.
#   - Does not use TEST.
#   - If registry is empty or missing metrics, Q4 remains blocked/no-promotion.

import json
import re
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = None


def _q4a_log(msg):
    print(f"[Q4.1R] {msg}")


# ---------------------------------------------------------------------
# 0. Resolve paths
# ---------------------------------------------------------------------
if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}

outdir = Path(str(globals().get("OUTDIR", CFG.get("outdir", ".")))).expanduser().resolve()
report_dir = Path(str(globals().get("REPORT_DIR", outdir / "reports"))).expanduser().resolve()
contract_dir = Path(str(globals().get("CONTRACT_DIR", outdir / "artifacts" / "contracts"))).expanduser().resolve()

report_dir.mkdir(parents=True, exist_ok=True)
contract_dir.mkdir(parents=True, exist_ok=True)

registry_path = report_dir / "q4_study_trainval_frozen_publication_pair_registry.csv"
contract_path = contract_dir / "q4_val_only_acceptance_gate_contract_THESIS.json"


# ---------------------------------------------------------------------
# 1. Load registry, or create empty fail-closed registry
# ---------------------------------------------------------------------
registry = globals().get("Q4_CLAIM_PAIR_REGISTRY_VAL_FROZEN_DF")

if not isinstance(registry, pd.DataFrame):
    if registry_path.exists():
        try:
            registry = pd.read_csv(registry_path)
        except pd.errors.EmptyDataError:
            registry = pd.DataFrame(columns=[
                "anchor_col",
                "protocol_col",
                "tier",
                "q4_publication_registry_source",
                "broad_all_pair_scan_role",
                "q4_no_claim_reason",
            ])
    else:
        registry = pd.DataFrame(columns=[
            "anchor_col",
            "protocol_col",
            "tier",
            "q4_publication_registry_source",
            "broad_all_pair_scan_role",
            "q4_no_claim_reason",
        ])
        registry.to_csv(registry_path, index=False)

registry = registry.copy()


# ---------------------------------------------------------------------
# 2. Guard against TEST-like columns
# ---------------------------------------------------------------------
test_cols = [
    c for c in registry.columns
    if re.search(r"(^|_|-)(test|final_qa|post_test|after_test|testqa)($|_|-)", str(c), re.I)
]

if test_cols:
    raise RuntimeError("Q4 acceptance registry contains TEST-like columns: " + str(test_cols))


# ---------------------------------------------------------------------
# 3. VAL-only gate metrics
# ---------------------------------------------------------------------
def _num_series(df, candidates):
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce"), c
    return None, None


eta, eta_col = _num_series(
    registry,
    ["eta_similarity_val", "ETA_similarity_val", "mean_ETA_similarity_val", "eta_similarity"],
)

profile, profile_col = _num_series(
    registry,
    ["profile_similarity_val", "manifest_profile_similarity_val", "profile_similarity"],
)

lag, lag_col = _num_series(
    registry,
    ["lag_peak_error_val", "lag_error_val", "lag_peak_error"],
)

rate, rate_col = _num_series(
    registry,
    ["response_window_rate_error_val", "window_rate_error_val", "response_window_rate_error"],
)

required_available = {
    "eta": eta is not None,
    "profile": profile is not None,
    "lag": lag is not None,
    "rate": rate is not None,
}

checks = []

if eta is not None:
    checks.append(eta >= float(CFG.get("Q4_ACCEPT_MIN_VAL_ETA", 0.50)))

if profile is not None:
    checks.append(profile >= float(CFG.get("Q4_ACCEPT_MIN_VAL_PROFILE", 0.70)))

if lag is not None:
    checks.append(lag <= float(CFG.get("Q4_ACCEPT_MAX_VAL_LAG", 2.0)))

if rate is not None:
    checks.append(rate <= float(CFG.get("Q4_ACCEPT_MAX_VAL_WINDOW_RATE", 0.15)))


# ---------------------------------------------------------------------
# 4. Fail-closed acceptance decision
# ---------------------------------------------------------------------
if len(registry) == 0:
    accept = False
    reason = "empty_VAL_frozen_registry_no_Q4_publication_claim"

elif len(checks) < 4:
    accept = False
    reason = "missing_required_VAL_metrics_or_empty_registry"

else:
    pair_pass = np.logical_and.reduce([
        c.fillna(False).to_numpy(dtype=bool)
        for c in checks
    ])

    accept = bool(pair_pass.all())
    reason = "all_registered_pairs_clear_VAL_gates" if accept else "one_or_more_registered_pairs_failed_VAL_gates"


# ---------------------------------------------------------------------
# 5. Contract
# ---------------------------------------------------------------------
Q4_VAL_ACCEPTANCE_CONTRACT = {
    "cell": "Q4_VAL_only_acceptance_gate",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "registered_pairs": int(len(registry)),
    "required_metrics_available": required_available,
    "metric_columns_used": {
        "eta": eta_col,
        "profile": profile_col,
        "lag": lag_col,
        "rate": rate_col,
    },
    "accepted_for_final_materialization_before_TEST": bool(accept),
    "reason": reason,
    "TEST_QA_may_promote": False,
    "TEST_real_values_used": False,
    "policy": {
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "terminal_TEST_may_report_only": True,
        "empty_registry_means_no_q4_promotion": True,
    },
}

contract_path.write_text(
    json.dumps(Q4_VAL_ACCEPTANCE_CONTRACT, indent=2, sort_keys=True),
    encoding="utf-8",
)

globals()["Q4_ACCEPTED_FOR_FINAL_BEFORE_TEST"] = bool(accept)
globals()["Q4_VAL_ACCEPTANCE_CONTRACT"] = Q4_VAL_ACCEPTANCE_CONTRACT
globals()["Q4_CLAIM_PAIR_REGISTRY_VAL_FROZEN_DF"] = registry
globals()["Q4_CLAIM_PAIR_REGISTRY_VAL_FROZEN_PATH"] = str(registry_path)

_q4a_log(f"accepted_for_final_materialization_before_TEST = {accept}")
_q4a_log(f"reason = {reason}")
_q4a_log(f"registered_pairs = {len(registry)}")
_q4a_log(f"contract = {contract_path}")

if not accept:
    _q4a_log("Q4 remains blocked/no-promotion. Terminal TEST QA may report evidence but cannot promote.")

if display is not None:
    display(pd.DataFrame([Q4_VAL_ACCEPTANCE_CONTRACT]))