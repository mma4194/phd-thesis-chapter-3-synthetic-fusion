# ==========================================================
# CELL 12.e.3 - Driver VAL selection
# v1.1 STUDY-THESIS strict sparse binary driver VAL-only selection, contract-hardened
#
# Role:
#   - Select one locked driver generator per driver target.
#   - Use Cell 12.e.2 VAL candidate metrics only.
#   - Compare non-A1 candidates against A1 reference.
#   - Select non-A1 only if it materially improves driver timing
#     without damaging event-rate/burst-count fidelity.
#
# Driver scope:
#   - 26 sparse binary events_in_sec__... driver targets.
#
# Selection metrics:
#   event_rate_error
#   burst_count_error
#   interarrival_ks
#   duration_ks
#   overdispersion_error
#   regime_event_rate_error
#   copy-risk diagnostics
#
# Strict rules:
#   - Do NOT read real TEST values.
#   - Do NOT generate TEST-length values.
#   - Do NOT fit generators.
#   - Do NOT mutate candidate values.
#
# Outputs:
#   reports/cell12e3_locked_driver_selection.csv
#   reports/cell12e3_driver_selection_scores.csv
#   reports/cell12e3_driver_selection_audit.csv
#   reports/cell12e3_driver_publication_risk_audit.csv
#   reports/cell12e3_driver_contract.json
#   artifacts/cell12e3_driver_selection_manifest.json
# ==========================================================

log("--- START: Cell 12.e.3 - Driver VAL selection (v1.1 strict sparse binary, contract-hardened) ---")

import os
import re
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_12e3 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_DRIVER_COLS",
    "IOT_DRIVER_TARGET_CONTRACT_DF",
    "CELL12E_DRIVER_CONTRACT",
    "CELL12E1_A1_DRIVER_METRICS_DF",
    "CELL12E2_DRIVER_CANDIDATE_METRICS_DF",
    "CELL12E2_DRIVER_CANDIDATE_INVENTORY_DF",
    "CELL12E2_DRIVER_CANDIDATE_COVERAGE_DF",
    "CELL12E2_DRIVER_CONTRACT",
]
_missing_12e3 = [k for k in _required_12e3 if k not in globals()]
if _missing_12e3:
    raise RuntimeError(f"[Cell12.e.3] Missing required globals from prior cells: {_missing_12e3}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

IOT_DRIVER_COLS = list(map(str, IOT_DRIVER_COLS))
EXPECTED_DRIVER_TARGET_COUNT_12E = int(CFG.get("cell12e_expected_driver_target_count", 26))

if len(IOT_DRIVER_COLS) != EXPECTED_DRIVER_TARGET_COUNT_12E:
    raise RuntimeError(
        "[Cell12.e.3] Driver target count mismatch: "
        f"got={len(IOT_DRIVER_COLS)} expected={EXPECTED_DRIVER_TARGET_COUNT_12E}"
    )

CELL12E3_VERSION = "cell12e3_sparse_binary_driver_val_selection_strict_v1_1_contract_hardened"

CFG["cell12e3_version"] = CELL12E3_VERSION
CFG["cell12e3_TEST_real_values_used"] = False
CFG["cell12e3_selection_done_here"] = True
CFG["cell12e3_generator_fit_done_here"] = False
CFG["cell12e3_test_length_materialization_done_here"] = False
CFG["cell12e3_val_values_used_for_selection"] = True

# Conservative selection thresholds.
CFG.setdefault("cell12e3_min_score_rel_improvement", 0.03)
CFG.setdefault("cell12e3_min_score_abs_improvement", 0.005)
CFG.setdefault("cell12e3_min_interarrival_ks_improvement", 0.05)
CFG.setdefault("cell12e3_min_regime_rate_improvement", 0.00005)
CFG.setdefault("cell12e3_max_event_rate_error_multiplier", 1.10)
CFG.setdefault("cell12e3_max_burst_count_error_multiplier", 1.10)
CFG.setdefault("cell12e3_max_duration_ks_multiplier", 1.25)
CFG.setdefault("cell12e3_max_copy_overlap_allowed", 0.20)

# Publication/audit thresholds.
CFG.setdefault("cell12e3_event_rate_error_warning", 0.001)
CFG.setdefault("cell12e3_event_rate_error_blocker", 0.005)
CFG.setdefault("cell12e3_burst_count_error_warning", 0.001)
CFG.setdefault("cell12e3_burst_count_error_blocker", 0.005)
CFG.setdefault("cell12e3_interarrival_ks_warning", 0.50)
CFG.setdefault("cell12e3_interarrival_ks_blocker", 0.80)
CFG.setdefault("cell12e3_duration_ks_warning", 0.20)
CFG.setdefault("cell12e3_duration_ks_blocker", 0.50)
CFG.setdefault("cell12e3_regime_event_rate_error_warning", 0.002)
CFG.setdefault("cell12e3_regime_event_rate_error_blocker", 0.010)
CFG.setdefault("cell12e3_copy_overlap_warning", 0.20)
CFG.setdefault("cell12e3_copy_overlap_blocker", 0.50)

A1_DRIVER_FAMILY_12E3 = "A1_driver_event_block_bootstrap_reference"

NON_A1_DRIVER_FAMILIES_12E3 = [
    "RegimeConditionedDriverMarkov",
    "RareEventBernoulliMarkovDriver",
    "SparseEventGapRenewalDriver",
]

VALID_DRIVER_FAMILIES_12E3 = [A1_DRIVER_FAMILY_12E3] + NON_A1_DRIVER_FAMILIES_12E3

# ----------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------
def _json_sanitize_12e3(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_12e3(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_12e3(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_12e3(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_12e3(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_12e3(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_12e3(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_12e3(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj

def _write_json_12e3(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_12e3(payload), f, indent=2, sort_keys=True)

def _sha256_file_12e3(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_12e3(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_bool_12e3(x, default=False) -> bool:
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
    if s in {"false", "0", "no", "n", "f", "", "nan", "none", "null"}:
        return False
    return bool(default)

def _entity_from_driver_col_12e3(col: str) -> str:
    """Parse entity names robustly for events_in_sec driver namespaces."""
    parts = [p for p in str(col).split("__") if p != ""]
    if not parts:
        return str(col)

    if parts[0] == "events_in_sec":
        if len(parts) >= 3 and parts[1] in {"entity", "device", "iot"}:
            return parts[2]
        if len(parts) >= 3 and parts[1] == "feat":
            return parts[2]
        return parts[1] if len(parts) > 1 else str(col)

    if parts[0] == "iot":
        return parts[1] if len(parts) > 1 else str(col)

    if "__feat__" in str(col):
        tail = str(col).split("__feat__", 1)[1]
        return tail.split("__", 1)[0]

    if "__entity__" in str(col):
        tail = str(col).split("__entity__", 1)[1]
        return tail.split("__", 1)[0]

    return str(col)

def _driver_measurement_12e3(col: str) -> str:
    """Parse measurement/driver semantics without collapsing the entity into the measurement."""
    parts = [p for p in str(col).split("__") if p != ""]
    if not parts:
        return str(col)

    if parts[0] == "events_in_sec":
        if len(parts) >= 2 and parts[1] == "entity":
            return "entity_event"
        if len(parts) >= 4 and parts[1] == "feat":
            return "__".join(parts[3:])
        if len(parts) >= 3:
            return "__".join(parts[2:])
        return str(col)

    if parts[0] == "iot":
        if len(parts) >= 4:
            return "__".join(parts[2:])
        return "__".join(parts[1:])

    if "__feat__" in str(col):
        tail = str(col).split("__feat__", 1)[1]
        bits = tail.split("__")
        return "__".join(bits[1:]) if len(bits) > 1 else tail

    if "__entity__" in str(col):
        return "entity_event"

    return str(col)

def _improvement_12e3(a1_val, cand_val) -> float:
    a1 = _safe_float_12e3(a1_val, np.nan)
    cand = _safe_float_12e3(cand_val, np.nan)
    if not np.isfinite(a1) or not np.isfinite(cand):
        return np.nan
    return float(a1 - cand)

def _rel_improvement_12e3(a1_val, cand_val) -> float:
    a1 = _safe_float_12e3(a1_val, np.nan)
    cand = _safe_float_12e3(cand_val, np.nan)
    if not np.isfinite(a1) or not np.isfinite(cand):
        return np.nan
    denom = max(abs(a1), 1e-9)
    return float((a1 - cand) / denom)

def _not_worse_multiplier_12e3(cand_val, a1_val, multiplier: float, eps: float = 1e-12) -> bool:
    cand = _safe_float_12e3(cand_val, np.nan)
    a1 = _safe_float_12e3(a1_val, np.nan)
    if not np.isfinite(cand) or not np.isfinite(a1):
        return False
    return bool(cand <= max(a1 * float(multiplier), a1 + eps))

def _risk_status_12e3(row: dict) -> tuple:
    reasons = []
    status = "pass"

    event_rate_error = _safe_float_12e3(row.get("event_rate_error"), np.nan)
    burst_count_error = _safe_float_12e3(row.get("burst_count_error"), np.nan)
    interarrival_ks = _safe_float_12e3(row.get("interarrival_ks"), np.nan)
    duration_ks = _safe_float_12e3(row.get("duration_ks"), np.nan)
    regime_event_rate_error = _safe_float_12e3(row.get("regime_event_rate_error"), np.nan)
    copy_overlap = _safe_float_12e3(row.get("exact_event_index_overlap_rate"), np.nan)

    if np.isfinite(event_rate_error):
        if event_rate_error >= float(CFG["cell12e3_event_rate_error_blocker"]):
            reasons.append("event_rate_error_blocker")
            status = "blocker"
        elif event_rate_error >= float(CFG["cell12e3_event_rate_error_warning"]) and status != "blocker":
            reasons.append("event_rate_error_warning")
            status = "warning"

    if np.isfinite(burst_count_error):
        if burst_count_error >= float(CFG["cell12e3_burst_count_error_blocker"]):
            reasons.append("burst_count_error_blocker")
            status = "blocker"
        elif burst_count_error >= float(CFG["cell12e3_burst_count_error_warning"]) and status != "blocker":
            reasons.append("burst_count_error_warning")
            status = "warning"

    if np.isfinite(interarrival_ks):
        if interarrival_ks >= float(CFG["cell12e3_interarrival_ks_blocker"]):
            reasons.append("interarrival_ks_blocker")
            status = "blocker"
        elif interarrival_ks >= float(CFG["cell12e3_interarrival_ks_warning"]) and status != "blocker":
            reasons.append("interarrival_ks_warning")
            status = "warning"

    if np.isfinite(duration_ks):
        if duration_ks >= float(CFG["cell12e3_duration_ks_blocker"]):
            reasons.append("duration_ks_blocker")
            status = "blocker"
        elif duration_ks >= float(CFG["cell12e3_duration_ks_warning"]) and status != "blocker":
            reasons.append("duration_ks_warning")
            status = "warning"

    if np.isfinite(regime_event_rate_error):
        if regime_event_rate_error >= float(CFG["cell12e3_regime_event_rate_error_blocker"]):
            reasons.append("regime_event_rate_error_blocker")
            status = "blocker"
        elif regime_event_rate_error >= float(CFG["cell12e3_regime_event_rate_error_warning"]) and status != "blocker":
            reasons.append("regime_event_rate_error_warning")
            status = "warning"

    if np.isfinite(copy_overlap):
        if copy_overlap >= float(CFG["cell12e3_copy_overlap_blocker"]):
            reasons.append("copy_overlap_blocker")
            status = "blocker"
        elif copy_overlap >= float(CFG["cell12e3_copy_overlap_warning"]) and status != "blocker":
            reasons.append("copy_overlap_warning")
            status = "warning"

    if not reasons:
        reasons.append("within_driver_selection_gates")

    return status, "|".join(reasons)

# ----------------------------------------------------------
# 2) Load and validate inputs
# ----------------------------------------------------------
metrics_df = CELL12E2_DRIVER_CANDIDATE_METRICS_DF.copy()
inventory_df = CELL12E2_DRIVER_CANDIDATE_INVENTORY_DF.copy()
coverage_df = CELL12E2_DRIVER_CANDIDATE_COVERAGE_DF.copy()
contract_df = IOT_DRIVER_TARGET_CONTRACT_DF.copy()

for df in [metrics_df, inventory_df, coverage_df, contract_df]:
    if "col" in df.columns:
        df["col"] = df["col"].astype(str)

required_metric_cols = [
    "col",
    "candidate_id",
    "candidate_family",
    "is_a1",
    "candidate_valid",
    "valid_for_selection",
    "score",
    "event_rate_error",
    "burst_count_error",
    "interarrival_ks",
    "duration_ks",
    "overdispersion_error",
    "regime_event_rate_error",
]
missing_metric_cols = [c for c in required_metric_cols if c not in metrics_df.columns]
if missing_metric_cols:
    raise RuntimeError(f"[Cell12.e.3] Missing metric columns from 12.e.2: {missing_metric_cols}")

unknown_cols = sorted(set(metrics_df["col"].astype(str)) - set(IOT_DRIVER_COLS))
missing_cols = sorted(set(IOT_DRIVER_COLS) - set(metrics_df["col"].astype(str)))
if unknown_cols or missing_cols:
    raise RuntimeError(
        "[Cell12.e.3] Candidate metrics column-set mismatch. "
        f"unknown_cols={unknown_cols[:10]} missing_cols={missing_cols[:10]}"
    )

expected_rows = int(len(IOT_DRIVER_COLS) * len(VALID_DRIVER_FAMILIES_12E3))
if len(metrics_df) != expected_rows:
    raise RuntimeError(
        f"[Cell12.e.3] Candidate metrics row count mismatch: got={len(metrics_df)} expected={expected_rows}"
    )

numeric_cols = [
    "score",
    "event_rate_error",
    "burst_count_error",
    "interarrival_ks",
    "duration_ks",
    "overdispersion_error",
    "regime_event_rate_error",
    "real_event_rate",
    "syn_event_rate",
    "real_event_count",
    "syn_event_count",
    "train_event_rate",
    "val_event_rate",
    "abs_train_val_event_rate_delta",
    "exact_event_index_overlap_rate",
    "copy_risk_high",
]
for c in numeric_cols:
    if c in metrics_df.columns and c != "copy_risk_high":
        metrics_df[c] = pd.to_numeric(metrics_df[c], errors="coerce")

for c in ["is_a1", "candidate_valid", "valid_for_selection", "copy_risk_high"]:
    if c in metrics_df.columns:
        metrics_df[c] = metrics_df[c].map(lambda x: _safe_bool_12e3(x, False))

# Contract metadata.
driver_type_map = {}
recommended_map = {}
if "driver_type" in contract_df.columns:
    driver_type_map = dict(zip(contract_df["col"], contract_df["driver_type"].astype(str)))
if "recommended_12e2_candidate" in contract_df.columns:
    recommended_map = dict(zip(contract_df["col"], contract_df["recommended_12e2_candidate"].astype(str)))

# ----------------------------------------------------------
# 3) Selection loop
# ----------------------------------------------------------
selection_rows = []
score_rows = []
audit_rows = []
risk_rows = []

log(
    "[Cell12.e.3] Selecting driver generators from VAL metrics | "
    f"targets={len(IOT_DRIVER_COLS)} | candidate_rows={len(metrics_df)}"
)

for j, col in enumerate(IOT_DRIVER_COLS, start=1):
    if j == 1 or j % 5 == 0 or j == len(IOT_DRIVER_COLS):
        log(f"[Cell12.e.3] progress {j}/{len(IOT_DRIVER_COLS)} | col={col}")

    sub = metrics_df[metrics_df["col"].astype(str) == col].copy()
    if len(sub) != len(VALID_DRIVER_FAMILIES_12E3):
        raise RuntimeError(
            f"[Cell12.e.3] Expected {len(VALID_DRIVER_FAMILIES_12E3)} candidate rows for {col}, got {len(sub)}"
        )

    a1_sub = sub[sub["candidate_family"].astype(str).eq(A1_DRIVER_FAMILY_12E3)]
    if len(a1_sub) != 1:
        raise RuntimeError(
            f"[Cell12.e.3] Missing or duplicated A1 driver reference for {col}: rows={len(a1_sub)}"
        )

    a1 = a1_sub.iloc[0].to_dict()
    a1_score = _safe_float_12e3(a1.get("score"), np.inf)

    if not np.isfinite(a1_score):
        raise RuntimeError(f"[Cell12.e.3] A1 score is not finite for {col}")

    best_row = dict(a1)
    selected_reason = "a1_default_reference"
    selected_is_non_a1 = False
    non_a1_candidates_considered_n = 0
    non_a1_valid_n = 0
    non_a1_materially_better_n = 0
    non_a1_rejected_reasons = []

    non_a1 = sub[~sub["candidate_family"].astype(str).eq(A1_DRIVER_FAMILY_12E3)].copy()

    for _, cand_ser in non_a1.iterrows():
        cand = cand_ser.to_dict()
        fam = str(cand.get("candidate_family", ""))
        cid = str(cand.get("candidate_id", ""))

        candidate_valid = _safe_bool_12e3(cand.get("candidate_valid"), False)
        valid_for_selection = _safe_bool_12e3(cand.get("valid_for_selection"), False)
        non_a1_candidates_considered_n += 1

        score_abs_improvement = _improvement_12e3(a1.get("score"), cand.get("score"))
        score_rel_improvement = _rel_improvement_12e3(a1.get("score"), cand.get("score"))
        event_rate_improvement = _improvement_12e3(a1.get("event_rate_error"), cand.get("event_rate_error"))
        burst_count_improvement = _improvement_12e3(a1.get("burst_count_error"), cand.get("burst_count_error"))
        interarrival_improvement = _improvement_12e3(a1.get("interarrival_ks"), cand.get("interarrival_ks"))
        duration_improvement = _improvement_12e3(a1.get("duration_ks"), cand.get("duration_ks"))
        regime_improvement = _improvement_12e3(a1.get("regime_event_rate_error"), cand.get("regime_event_rate_error"))

        score_gate = bool(
            np.isfinite(score_abs_improvement)
            and np.isfinite(score_rel_improvement)
            and score_abs_improvement >= float(CFG["cell12e3_min_score_abs_improvement"])
            and score_rel_improvement >= float(CFG["cell12e3_min_score_rel_improvement"])
        )

        metric_gate = bool(
            (
                np.isfinite(interarrival_improvement)
                and interarrival_improvement >= float(CFG["cell12e3_min_interarrival_ks_improvement"])
            )
            or (
                np.isfinite(regime_improvement)
                and regime_improvement >= float(CFG["cell12e3_min_regime_rate_improvement"])
            )
            or (
                np.isfinite(event_rate_improvement)
                and event_rate_improvement > 0.0
                and np.isfinite(burst_count_improvement)
                and burst_count_improvement > 0.0
            )
        )

        not_worse_event_rate = _not_worse_multiplier_12e3(
            cand.get("event_rate_error"),
            a1.get("event_rate_error"),
            float(CFG["cell12e3_max_event_rate_error_multiplier"]),
        )
        not_worse_burst = _not_worse_multiplier_12e3(
            cand.get("burst_count_error"),
            a1.get("burst_count_error"),
            float(CFG["cell12e3_max_burst_count_error_multiplier"]),
        )
        not_worse_duration = _not_worse_multiplier_12e3(
            cand.get("duration_ks"),
            a1.get("duration_ks"),
            float(CFG["cell12e3_max_duration_ks_multiplier"]),
        )

        copy_overlap = _safe_float_12e3(cand.get("exact_event_index_overlap_rate"), np.nan)
        copy_gate = bool(
            not np.isfinite(copy_overlap)
            or copy_overlap <= float(CFG["cell12e3_max_copy_overlap_allowed"])
        )

        no_major_regression_gate = bool(
            not_worse_event_rate
            and not_worse_burst
            and not_worse_duration
            and copy_gate
        )

        materially_better = bool(
            candidate_valid
            and valid_for_selection
            and score_gate
            and metric_gate
            and no_major_regression_gate
        )

        if candidate_valid and valid_for_selection:
            non_a1_valid_n += 1
        if materially_better:
            non_a1_materially_better_n += 1

        reject_reasons = []
        if not candidate_valid:
            reject_reasons.append("candidate_invalid")
        if not valid_for_selection:
            reject_reasons.append("not_valid_for_selection")
        if not score_gate:
            reject_reasons.append("insufficient_score_improvement")
        if not metric_gate:
            reject_reasons.append("insufficient_driver_metric_improvement")
        if not no_major_regression_gate:
            reject_reasons.append("major_regression_or_copy_risk_vs_a1")
        if not reject_reasons:
            reject_reasons.append("accepted_candidate")

        score_rows.append({
            "col": col,
            "candidate_id": cid,
            "candidate_family": fam,
            "is_a1": False,
            "candidate_valid": bool(candidate_valid),
            "valid_for_selection": bool(valid_for_selection),
            "a1_score": a1_score,
            "candidate_score": _safe_float_12e3(cand.get("score"), np.inf),
            "score_abs_improvement_vs_a1": score_abs_improvement,
            "score_rel_improvement_vs_a1": score_rel_improvement,
            "event_rate_error_improvement_vs_a1": event_rate_improvement,
            "burst_count_error_improvement_vs_a1": burst_count_improvement,
            "interarrival_ks_improvement_vs_a1": interarrival_improvement,
            "duration_ks_improvement_vs_a1": duration_improvement,
            "regime_event_rate_error_improvement_vs_a1": regime_improvement,
            "score_gate": bool(score_gate),
            "metric_gate": bool(metric_gate),
            "no_major_regression_gate": bool(no_major_regression_gate),
            "copy_gate": bool(copy_gate),
            "materially_better_than_a1": bool(materially_better),
            "selection_reject_reasons": "|".join(reject_reasons),
            "TEST_real_values_used": False,
        })

        if materially_better:
            cand_score = _safe_float_12e3(cand.get("score"), np.inf)
            best_score = _safe_float_12e3(best_row.get("score"), np.inf)

            if cand_score < best_score:
                best_row = dict(cand)
                selected_reason = (
                    "non_a1_materially_beats_a1"
                    f"|score_abs_improvement={score_abs_improvement:.6g}"
                    f"|score_rel_improvement={score_rel_improvement:.6g}"
                    f"|driver_metric_gate=True"
                )
                selected_is_non_a1 = True
        else:
            non_a1_rejected_reasons.append(f"{fam}:{'|'.join(reject_reasons)}")

    # A1 score row for completeness.
    score_rows.append({
        "col": col,
        "candidate_id": str(a1.get("candidate_id", "")),
        "candidate_family": str(a1.get("candidate_family", "")),
        "is_a1": True,
        "candidate_valid": True,
        "valid_for_selection": True,
        "a1_score": a1_score,
        "candidate_score": a1_score,
        "score_abs_improvement_vs_a1": 0.0,
        "score_rel_improvement_vs_a1": 0.0,
        "event_rate_error_improvement_vs_a1": 0.0,
        "burst_count_error_improvement_vs_a1": 0.0,
        "interarrival_ks_improvement_vs_a1": 0.0,
        "duration_ks_improvement_vs_a1": 0.0,
        "regime_event_rate_error_improvement_vs_a1": 0.0,
        "score_gate": False,
        "metric_gate": False,
        "no_major_regression_gate": True,
        "copy_gate": True,
        "materially_better_than_a1": False,
        "selection_reject_reasons": "a1_reference",
        "TEST_real_values_used": False,
    })

    selected_family = str(best_row.get("candidate_family", ""))
    selected_candidate_id = str(best_row.get("candidate_id", ""))
    selected_score = _safe_float_12e3(best_row.get("score"), np.inf)

    if selected_family not in VALID_DRIVER_FAMILIES_12E3:
        raise RuntimeError(
            f"[Cell12.e.3] Invalid selected driver family for {col}: {selected_family}"
        )

    selection_base = {
        "col": col,
        "entity": _entity_from_driver_col_12e3(col),
        "measurement_name": _driver_measurement_12e3(col),
        "driver_type": driver_type_map.get(col, "binary_driver"),
        "selected_driver_generator": selected_family,
        "selected_candidate_id": selected_candidate_id,
        "selected_is_a1": bool(selected_family == A1_DRIVER_FAMILY_12E3),
        "selected_is_non_a1": bool(selected_is_non_a1),
        "selected_score": selected_score,
        "selected_reason": selected_reason,
        "a1_candidate_id": str(a1.get("candidate_id", "")),
        "a1_score": a1_score,
        "a1_event_rate_error": _safe_float_12e3(a1.get("event_rate_error"), np.nan),
        "a1_burst_count_error": _safe_float_12e3(a1.get("burst_count_error"), np.nan),
        "a1_interarrival_ks": _safe_float_12e3(a1.get("interarrival_ks"), np.nan),
        "a1_duration_ks": _safe_float_12e3(a1.get("duration_ks"), np.nan),
        "a1_overdispersion_error": _safe_float_12e3(a1.get("overdispersion_error"), np.nan),
        "a1_regime_event_rate_error": _safe_float_12e3(a1.get("regime_event_rate_error"), np.nan),
        "selected_event_rate_error": _safe_float_12e3(best_row.get("event_rate_error"), np.nan),
        "selected_burst_count_error": _safe_float_12e3(best_row.get("burst_count_error"), np.nan),
        "selected_interarrival_ks": _safe_float_12e3(best_row.get("interarrival_ks"), np.nan),
        "selected_duration_ks": _safe_float_12e3(best_row.get("duration_ks"), np.nan),
        "selected_overdispersion_error": _safe_float_12e3(best_row.get("overdispersion_error"), np.nan),
        "selected_regime_event_rate_error": _safe_float_12e3(best_row.get("regime_event_rate_error"), np.nan),
        "selected_exact_event_index_overlap_rate": _safe_float_12e3(best_row.get("exact_event_index_overlap_rate"), np.nan),
        "selected_copy_risk_high": _safe_bool_12e3(best_row.get("copy_risk_high"), False),
        "train_observed_n": int(_safe_float_12e3(best_row.get("train_observed_n"), 0)),
        "val_observed_n": int(_safe_float_12e3(best_row.get("val_observed_n"), 0)),
        "train_event_rate": _safe_float_12e3(best_row.get("train_event_rate"), np.nan),
        "val_event_rate": _safe_float_12e3(best_row.get("val_event_rate"), np.nan),
        "abs_train_val_event_rate_delta": _safe_float_12e3(best_row.get("abs_train_val_event_rate_delta"), np.nan),
        "train_event_count": int(_safe_float_12e3(best_row.get("train_event_count"), 0)),
        "val_event_count": int(_safe_float_12e3(best_row.get("val_event_count"), 0)),
        "non_a1_candidates_considered_n": int(non_a1_candidates_considered_n),
        "non_a1_valid_n": int(non_a1_valid_n),
        "non_a1_materially_better_n": int(non_a1_materially_better_n),
        "non_a1_rejected_reasons_preview": " || ".join(non_a1_rejected_reasons[:8]),
        "selection_scope": "VAL_only",
        "TEST_real_values_used": False,
        "test_length_materialization_done_here": False,
        "generator_fit_done_here": False,
        "feeds_cell13_coupling_qa": True,
    }

    pub_status, pub_reasons = _risk_status_12e3({
        "event_rate_error": selection_base["selected_event_rate_error"],
        "burst_count_error": selection_base["selected_burst_count_error"],
        "interarrival_ks": selection_base["selected_interarrival_ks"],
        "duration_ks": selection_base["selected_duration_ks"],
        "regime_event_rate_error": selection_base["selected_regime_event_rate_error"],
        "exact_event_index_overlap_rate": selection_base["selected_exact_event_index_overlap_rate"],
    })

    selection_base["driver_publication_status_after_12e3"] = pub_status
    selection_base["driver_publication_reasons_after_12e3"] = pub_reasons
    selection_base["driver_publication_blocker_after_12e3"] = bool(pub_status == "blocker")

    selection_rows.append(selection_base)

    audit_rows.append({
        "col": col,
        "selected_driver_generator": selected_family,
        "selected_candidate_id": selected_candidate_id,
        "selected_is_a1": bool(selected_family == A1_DRIVER_FAMILY_12E3),
        "selected_is_non_a1": bool(selected_is_non_a1),
        "selected_reason": selected_reason,
        "a1_score": a1_score,
        "selected_score": selected_score,
        "score_delta_a1_minus_selected": float(a1_score - selected_score),
        "non_a1_candidates_considered_n": int(non_a1_candidates_considered_n),
        "non_a1_valid_n": int(non_a1_valid_n),
        "non_a1_materially_better_n": int(non_a1_materially_better_n),
        "publication_status": pub_status,
        "publication_reasons": pub_reasons,
        "TEST_real_values_used": False,
    })

    risk_rows.append({
        "col": col,
        "selected_driver_generator": selected_family,
        "selected_candidate_id": selected_candidate_id,
        "selected_is_a1": bool(selected_family == A1_DRIVER_FAMILY_12E3),
        "selected_event_rate_error": selection_base["selected_event_rate_error"],
        "selected_burst_count_error": selection_base["selected_burst_count_error"],
        "selected_interarrival_ks": selection_base["selected_interarrival_ks"],
        "selected_duration_ks": selection_base["selected_duration_ks"],
        "selected_overdispersion_error": selection_base["selected_overdispersion_error"],
        "selected_regime_event_rate_error": selection_base["selected_regime_event_rate_error"],
        "selected_exact_event_index_overlap_rate": selection_base["selected_exact_event_index_overlap_rate"],
        "selected_copy_risk_high": selection_base["selected_copy_risk_high"],
        "publication_status": pub_status,
        "publication_reasons": pub_reasons,
        "publication_blocker": bool(pub_status == "blocker"),
        "TEST_real_values_used": False,
    })

# ----------------------------------------------------------
# 4) Build outputs
# ----------------------------------------------------------
selection_df = pd.DataFrame(selection_rows)
scores_df = pd.DataFrame(score_rows)
audit_df = pd.DataFrame(audit_rows)
risk_df = pd.DataFrame(risk_rows)

if len(selection_df) != len(IOT_DRIVER_COLS):
    raise RuntimeError(
        f"[Cell12.e.3] Selection row count mismatch: got={len(selection_df)} expected={len(IOT_DRIVER_COLS)}"
    )

if selection_df["col"].duplicated().any():
    dupes = selection_df.loc[selection_df["col"].duplicated(), "col"].astype(str).tolist()
    raise RuntimeError(f"[Cell12.e.3] Duplicate selected driver columns: {dupes[:20]}")

missing_selected = sorted(set(IOT_DRIVER_COLS) - set(selection_df["col"].astype(str)))
if missing_selected:
    raise RuntimeError(f"[Cell12.e.3] Missing selected driver columns: {missing_selected[:20]}")

bad_selected_family = sorted(
    set(selection_df["selected_driver_generator"].astype(str))
    - set(VALID_DRIVER_FAMILIES_12E3)
)
if bad_selected_family:
    raise RuntimeError(f"[Cell12.e.3] Invalid selected driver families: {bad_selected_family}")

# Stable ordering.
_order_map = {c: i for i, c in enumerate(IOT_DRIVER_COLS)}

for df_name, df in [
    ("selection_df", selection_df),
    ("audit_df", audit_df),
    ("risk_df", risk_df),
]:
    df["_order"] = df["col"].map(_order_map)
    df.sort_values("_order", inplace=True)
    df.drop(columns=["_order"], inplace=True)
    df.reset_index(drop=True, inplace=True)

selected_counts = (
    selection_df["selected_driver_generator"].astype(str).value_counts().sort_index().to_dict()
)

publication_counts = (
    selection_df["driver_publication_status_after_12e3"].astype(str).value_counts().sort_index().to_dict()
)

non_a1_selected_n = int(selection_df["selected_is_non_a1"].fillna(False).astype(bool).sum())
a1_selected_n = int(selection_df["selected_is_a1"].fillna(False).astype(bool).sum())
blocker_n = int(selection_df["driver_publication_blocker_after_12e3"].fillna(False).astype(bool).sum())
warning_n = int((selection_df["driver_publication_status_after_12e3"].astype(str) == "warning").sum())
pass_n = int((selection_df["driver_publication_status_after_12e3"].astype(str) == "pass").sum())

metric_summary = {
    "selected_score_mean": float(pd.to_numeric(selection_df["selected_score"], errors="coerce").mean()),
    "selected_event_rate_error_mean": float(pd.to_numeric(selection_df["selected_event_rate_error"], errors="coerce").mean()),
    "selected_burst_count_error_mean": float(pd.to_numeric(selection_df["selected_burst_count_error"], errors="coerce").mean()),
    "selected_interarrival_ks_mean": float(pd.to_numeric(selection_df["selected_interarrival_ks"], errors="coerce").mean()),
    "selected_duration_ks_mean": float(pd.to_numeric(selection_df["selected_duration_ks"], errors="coerce").mean()),
    "selected_overdispersion_error_mean": float(pd.to_numeric(selection_df["selected_overdispersion_error"], errors="coerce").mean()),
    "selected_regime_event_rate_error_mean": float(pd.to_numeric(selection_df["selected_regime_event_rate_error"], errors="coerce").mean()),
    "selected_exact_event_index_overlap_rate_mean": float(pd.to_numeric(selection_df["selected_exact_event_index_overlap_rate"], errors="coerce").mean()),
}

# ----------------------------------------------------------
# 5) Save reports
# ----------------------------------------------------------
locked_selection_csv = os.path.join(REPORT_DIR, "cell12e3_locked_driver_selection.csv")
selection_scores_csv = os.path.join(REPORT_DIR, "cell12e3_driver_selection_scores.csv")
selection_audit_csv = os.path.join(REPORT_DIR, "cell12e3_driver_selection_audit.csv")
risk_audit_csv = os.path.join(REPORT_DIR, "cell12e3_driver_publication_risk_audit.csv")
contract_json = os.path.join(REPORT_DIR, "cell12e3_driver_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell12e3_driver_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell12e3_driver_selection_manifest.json")

selection_df.to_csv(locked_selection_csv, index=False)
scores_df.to_csv(selection_scores_csv, index=False)
audit_df.to_csv(selection_audit_csv, index=False)
risk_df.to_csv(risk_audit_csv, index=False)

contract = {
    "cell": "12.e.3",
    "version": CELL12E3_VERSION,
    "role": "sparse_binary_driver_VAL_only_portfolio_selection",
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "expected_driver_targets_total": int(EXPECTED_DRIVER_TARGET_COUNT_12E),
    "driver_type_scope": "sparse_binary_event_drivers",
    "train_rows_schema_only": int(N_TR),
    "val_rows": int(N_VAL),
    "test_rows_schema_only": int(N_TE),
    "selected_counts": selected_counts,
    "publication_counts_after_12e3": publication_counts,
    "a1_selected_n": int(a1_selected_n),
    "non_a1_selected_n": int(non_a1_selected_n),
    "pass_n": int(pass_n),
    "warning_n": int(warning_n),
    "blocker_n": int(blocker_n),
    "metric_summary": metric_summary,
    "selection_policy": {
        "min_score_rel_improvement": float(CFG["cell12e3_min_score_rel_improvement"]),
        "min_score_abs_improvement": float(CFG["cell12e3_min_score_abs_improvement"]),
        "min_interarrival_ks_improvement": float(CFG["cell12e3_min_interarrival_ks_improvement"]),
        "min_regime_rate_improvement": float(CFG["cell12e3_min_regime_rate_improvement"]),
        "max_event_rate_error_multiplier": float(CFG["cell12e3_max_event_rate_error_multiplier"]),
        "max_burst_count_error_multiplier": float(CFG["cell12e3_max_burst_count_error_multiplier"]),
        "max_duration_ks_multiplier": float(CFG["cell12e3_max_duration_ks_multiplier"]),
        "max_copy_overlap_allowed": float(CFG["cell12e3_max_copy_overlap_allowed"]),
    },
    "feeds_cell13_coupling_qa": True,
    "TEST_real_values_used": False,
    "test_length_materialization_done_here": False,
    "selection_done_here": True,
    "generator_fit_done_here": False,
    "val_values_used_for_selection": True,
    "df_te_used_for_index_length_schema_only": True,
    "copy_risk_interpretation": "copy overlap is an event-coordinate diagnostic only; sparse duplicate zeros are not treated as copying evidence by themselves.",
    "outputs": {
        "locked_selection_csv": locked_selection_csv,
        "selection_scores_csv": selection_scores_csv,
        "selection_audit_csv": selection_audit_csv,
        "risk_audit_csv": risk_audit_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_12e3(contract_json, contract)
_write_json_12e3(contract_canonical_json, contract)

manifest = {
    "cell": "12.e.3",
    "version": CELL12E3_VERSION,
    "created_outputs": contract["outputs"],
    "driver_targets_total": int(len(IOT_DRIVER_COLS)),
    "selected_counts": selected_counts,
    "publication_counts_after_12e3": publication_counts,
    "metric_summary": metric_summary,
    "feeds_cell13_coupling_qa": True,
    "no_TEST_leakage_contract": {
        "TEST_real_values_used": False,
        "test_length_materialization_done_here": False,
        "df_te_used_for_index_length_schema_only": True,
        "selection_done_here": True,
        "generator_fit_done_here": False,
    },
}

_write_json_12e3(manifest_json, manifest)

hashes = {
    "locked_selection_csv_sha256": _sha256_file_12e3(locked_selection_csv),
    "selection_scores_csv_sha256": _sha256_file_12e3(selection_scores_csv),
    "selection_audit_csv_sha256": _sha256_file_12e3(selection_audit_csv),
    "risk_audit_csv_sha256": _sha256_file_12e3(risk_audit_csv),
    "contract_json_sha256": _sha256_file_12e3(contract_json),
    "contract_canonical_json_sha256": _sha256_file_12e3(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_12e3(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_12e3(contract_json, contract)
_write_json_12e3(contract_canonical_json, contract)
_write_json_12e3(manifest_json, manifest)

# ----------------------------------------------------------
# 6) Export globals for 12.e.4+
# ----------------------------------------------------------
globals()["CELL12E3_VERSION"] = CELL12E3_VERSION
globals()["CELL12E3_LOCKED_DRIVER_SELECTION_DF"] = selection_df
globals()["CELL12E3_DRIVER_SELECTION_SCORES_DF"] = scores_df
globals()["CELL12E3_DRIVER_SELECTION_AUDIT_DF"] = audit_df
globals()["CELL12E3_DRIVER_PUBLICATION_RISK_AUDIT_DF"] = risk_df
globals()["CELL12E3_DRIVER_CONTRACT"] = contract
globals()["CELL12E3_LOCKED_DRIVER_SELECTION_CSV"] = locked_selection_csv
globals()["CELL12E3_DRIVER_SELECTION_SCORES_CSV"] = selection_scores_csv
globals()["CELL12E3_DRIVER_SELECTION_AUDIT_CSV"] = selection_audit_csv
globals()["CELL12E3_DRIVER_PUBLICATION_RISK_AUDIT_CSV"] = risk_audit_csv
globals()["CELL12E3_DRIVER_CONTRACT_JSON"] = contract_json
globals()["CELL12E3_DRIVER_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL12E3_DRIVER_MANIFEST_JSON"] = manifest_json

log(
    "[Cell12.e.3] Driver VAL selection complete | "
    f"targets={len(selection_df)} | "
    f"A1_selected={a1_selected_n} | "
    f"non_A1_selected={non_a1_selected_n} | "
    f"pass={pass_n} | warning={warning_n} | blocker={blocker_n}"
)
log(f"[Cell12.e.3] Selected generator counts | {selected_counts}")
log(f"[Cell12.e.3] Publication counts after 12.e.3 | {publication_counts}")
log(
    "[Cell12.e.3] Metric summary | "
    f"mean_event_rate_error={metric_summary['selected_event_rate_error_mean']:.8f} | "
    f"mean_burst_count_error={metric_summary['selected_burst_count_error_mean']:.8f} | "
    f"mean_interarrival_ks={metric_summary['selected_interarrival_ks_mean']:.6f} | "
    f"mean_duration_ks={metric_summary['selected_duration_ks_mean']:.6f} | "
    f"mean_overdispersion_error={metric_summary['selected_overdispersion_error_mean']:.6f} | "
    f"mean_regime_event_rate_error={metric_summary['selected_regime_event_rate_error_mean']:.8f}"
)
log(f"[Cell12.e.3] Saved locked selection: {locked_selection_csv} | rows={len(selection_df)}")
log(f"[Cell12.e.3] Saved selection scores: {selection_scores_csv} | rows={len(scores_df)}")
log(f"[Cell12.e.3] Saved selection audit: {selection_audit_csv} | rows={len(audit_df)}")
log(f"[Cell12.e.3] Saved publication risk audit: {risk_audit_csv} | rows={len(risk_df)}")
log(f"[Cell12.e.3] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell12.e.3] Contract flags | "
    "TEST_real_values_used=False | "
    "test_length_materialization_done_here=False | "
    "selection_done_here=True | "
    "generator_fit_done_here=False | "
    "df_te_used_for_index_length_schema_only=True"
)
log("--- END: Cell 12.e.3 - Driver VAL selection (v1.1 strict sparse binary, contract-hardened) ---")

gc.collect()