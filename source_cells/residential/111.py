# %% CELL Q4.0R — VAL-frozen Q4 publication pair registry, fail-closed/no-claim safe
# Purpose:
#   Build the Q4 publication pair registry from TRAIN/VAL-only Cell 14.4 discovery outputs.
#
# Fixes:
#   - Skips empty CSV files instead of crashing with EmptyDataError.
#   - Refuses TEST-like columns.
#   - If no valid TRAIN/VAL discovery table exists, writes an empty no-claim registry
#     instead of failing downstream cells.
#
# Safety:
#   - Does not read TEST values.
#   - Does not use terminal QA for pair selection.
#   - Does not fabricate coupling pairs.
#   - Empty registry means Q4 remains blocked/no-promotion.

import json
import re
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = None


def _q4r_log(msg):
    print(f"[Q4.0R] {msg}")


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
contract_path = contract_dir / "q4_val_frozen_pair_registry_contract_THESIS.json"


# ---------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------
def _first(df, opts):
    return next((c for c in opts if c in df.columns), None)


def _is_test_like_col(c):
    return bool(re.search(r"(^|_|-)(test|final_qa|post_test|after_test|testqa)($|_|-)", str(c), re.I))


def _read_csv_nonempty(path):
    path = Path(path)

    if not path.exists():
        raise ValueError("missing_file")

    if path.stat().st_size == 0:
        raise ValueError("empty_file_size_0")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError("empty_data_error_no_columns_to_parse")

    if df.shape[1] == 0:
        raise ValueError("no_columns")

    if len(df) == 0:
        raise ValueError("zero_rows")

    return df


def _write_empty_no_claim_registry(reason, source_attempts=None):
    cols = [
        "anchor_col",
        "protocol_col",
        "tier",
        "q4_publication_registry_source",
        "broad_all_pair_scan_role",
        "q4_no_claim_reason",
    ]

    empty = pd.DataFrame(columns=cols)
    empty.to_csv(registry_path, index=False)

    contract = {
        "cell": "Q4_VAL_frozen_pair_registry",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": 0,
        "publication_pair_rows": 0,
        "publication_tiers": sorted(map(str, CFG.get("Q4_PUBLICATION_TIERS", ["A", "B"]))),
        "metric_gates_applied": [],
        "TEST_real_values_used": False,
        "broad_all_pair_scan_publication_gate": False,
        "q4_publication_claim_available": False,
        "accepted_for_final_materialization_before_TEST": False,
        "reason": reason,
        "source_attempts": source_attempts or [],
        "policy": {
            "synthetic_values_mutated": False,
            "selection_done_here": False,
            "generator_fit_done_here": False,
            "post_TEST_repair_done_here": False,
            "coupling_pairs_fabricated_here": False,
            "empty_registry_means_no_q4_promotion": True,
        },
    }

    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

    globals()["Q4_CLAIM_PAIR_REGISTRY_VAL_FROZEN_DF"] = empty
    globals()["Q4_CLAIM_PAIR_REGISTRY_VAL_FROZEN_PATH"] = str(registry_path)
    globals()["Q4_VAL_FROZEN_PAIR_REGISTRY_CONTRACT"] = contract

    _q4r_log(f"No-claim empty registry written: {registry_path}")
    _q4r_log(f"Reason: {reason}")

    return empty, contract


# ---------------------------------------------------------------------
# 2. Resolve TRAIN/VAL discovery table
# ---------------------------------------------------------------------
source_attempts = []
stage = None
stage_source = None

# Prefer in-memory Cell 14.4 outputs when present.
for global_name in ["CELL14_4_STAGE2_REPLICATION_DF", "CELL14_4_COUPLING_DISCOVERY_DF"]:
    obj = globals().get(global_name)

    if not isinstance(obj, pd.DataFrame):
        continue

    if obj.shape[1] == 0:
        source_attempts.append({
            "source": f"global::{global_name}",
            "status": "skipped",
            "reason": "zero_columns",
        })
        continue

    test_like_cols = [c for c in obj.columns if _is_test_like_col(c)]
    if test_like_cols:
        source_attempts.append({
            "source": f"global::{global_name}",
            "status": "rejected",
            "reason": f"TEST_like_columns_present:{test_like_cols}",
            "rows": int(len(obj)),
            "cols": int(obj.shape[1]),
        })
        continue

    if len(obj) == 0:
        source_attempts.append({
            "source": f"global::{global_name}",
            "status": "skipped",
            "reason": "zero_rows",
        })
        continue

    stage = obj.copy()
    stage_source = f"global::{global_name}"
    source_attempts.append({
        "source": stage_source,
        "status": "selected",
        "rows": int(len(stage)),
        "cols": int(stage.shape[1]),
    })
    break


# Fall back to files.
if stage is None:
    candidate_paths = []
    candidate_paths += list(report_dir.glob("cell14_4*replication*.csv"))
    candidate_paths += list(report_dir.glob("cell14_4*stage*.csv"))
    candidate_paths += list(report_dir.glob("*trainval*coupling*.csv"))
    candidate_paths += list(report_dir.glob("cell14_4*coupling*.csv"))

    candidate_paths = sorted(
        set(candidate_paths),
        key=lambda p: ("stage2" not in p.name.lower(), len(p.name), p.name)
    )

    for path in candidate_paths:
        try:
            df = _read_csv_nonempty(path)
        except Exception as e:
            source_attempts.append({
                "source": str(path),
                "status": "skipped",
                "reason": str(e),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            })
            continue

        test_like_cols = [c for c in df.columns if _is_test_like_col(c)]
        if test_like_cols:
            source_attempts.append({
                "source": str(path),
                "status": "rejected",
                "reason": f"TEST_like_columns_present:{test_like_cols}",
                "rows": int(len(df)),
                "cols": int(df.shape[1]),
            })
            continue

        stage = df.copy()
        stage_source = str(path)
        source_attempts.append({
            "source": str(path),
            "status": "selected",
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
        })
        break


if stage is None:
    registry, contract = _write_empty_no_claim_registry(
        reason="no_valid_nonempty_TRAIN_VAL_Cell14_4_coupling_discovery_table_found",
        source_attempts=source_attempts,
    )

    if display is not None:
        display(registry)

else:
    stage = stage.copy()

    # -----------------------------------------------------------------
    # 3. Validate schema
    # -----------------------------------------------------------------
    anchor_col = _first(stage, ["anchor_col", "driver_col", "iot_driver_col", "event_driver", "source_col"])
    proto_col = _first(stage, ["protocol_col", "target_col", "response_col", "dependent_col"])
    tier_col = _first(stage, ["tier", "replication_tier", "evidence_tier", "stage2_tier"])

    if anchor_col is None or proto_col is None or tier_col is None:
        registry, contract = _write_empty_no_claim_registry(
            reason=f"Cell14_4_registry_schema_not_recognized_columns={list(stage.columns)}",
            source_attempts=source_attempts,
        )

        if display is not None:
            display(registry)

    else:
        # -------------------------------------------------------------
        # 4. Apply TRAIN/VAL gates only
        # -------------------------------------------------------------
        keep_tiers = set(map(str, CFG.get("Q4_PUBLICATION_TIERS", ["A", "B"])))
        registry = stage[stage[tier_col].astype(str).isin(keep_tiers)].copy()

        metric_gates = [
            (
                ["eta_similarity_val", "ETA_similarity_val", "mean_ETA_similarity_val", "eta_similarity"],
                ">=",
                float(CFG.get("Q4_MIN_VAL_ETA_SIMILARITY", 0.50)),
            ),
            (
                ["profile_similarity_val", "manifest_profile_similarity_val", "profile_similarity"],
                ">=",
                float(CFG.get("Q4_MIN_VAL_PROFILE_SIMILARITY", 0.70)),
            ),
            (
                ["lag_peak_error_val", "lag_error_val", "lag_peak_error"],
                "<=",
                float(CFG.get("Q4_MAX_VAL_LAG_ERROR", 2.0)),
            ),
            (
                ["response_window_rate_error_val", "window_rate_error_val", "response_window_rate_error"],
                "<=",
                float(CFG.get("Q4_MAX_VAL_WINDOW_RATE_ERROR", 0.15)),
            ),
        ]

        applied = []

        for opts, op, thresh in metric_gates:
            c = _first(registry, opts)

            if c is None:
                continue

            x = pd.to_numeric(registry[c], errors="coerce")

            if op == ">=":
                registry = registry[x >= thresh].copy()
            else:
                registry = registry[x <= thresh].copy()

            applied.append({"column": c, "op": op, "threshold": thresh})

        registry = registry.drop_duplicates([anchor_col, proto_col]).reset_index(drop=True)

        registry["q4_publication_registry_source"] = "TRAIN_VAL_frozen_before_TEST"
        registry["broad_all_pair_scan_role"] = "diagnostic_only_not_publication_gate"
        registry["q4_no_claim_reason"] = ""

        if len(registry) == 0:
            registry["q4_no_claim_reason"] = "all_TRAIN_VAL_candidate_pairs_filtered_by_publication_gates"

        registry.to_csv(registry_path, index=False)

        contract = {
            "cell": "Q4_VAL_frozen_pair_registry",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": stage_source,
            "input_rows": int(len(stage)),
            "publication_pair_rows": int(len(registry)),
            "publication_tiers": sorted(keep_tiers),
            "metric_gates_applied": applied,
            "anchor_col": anchor_col,
            "protocol_col": proto_col,
            "tier_col": tier_col,
            "TEST_real_values_used": False,
            "broad_all_pair_scan_publication_gate": False,
            "q4_publication_claim_available": bool(len(registry) > 0),
            "accepted_for_final_materialization_before_TEST": False,
            "source_attempts": source_attempts,
            "policy": {
                "synthetic_values_mutated": False,
                "selection_done_here": True,
                "selection_basis": "TRAIN_VAL_only",
                "generator_fit_done_here": False,
                "post_TEST_repair_done_here": False,
                "TEST_QA_may_promote": False,
            },
        }

        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

        globals()["Q4_CLAIM_PAIR_REGISTRY_VAL_FROZEN_DF"] = registry
        globals()["Q4_CLAIM_PAIR_REGISTRY_VAL_FROZEN_PATH"] = str(registry_path)
        globals()["Q4_VAL_FROZEN_PAIR_REGISTRY_CONTRACT"] = contract

        _q4r_log(f"Frozen publication pairs: {len(registry)}")
        _q4r_log(f"Registry: {registry_path}")
        _q4r_log(f"Contract: {contract_path}")

        if display is not None:
            display(registry.head(20))