# %% CELL 12.f.V5.2 — Observability-mask V5 TEST-length materialization
# Purpose:
#   Refit selected materializers on TRAIN+VAL and materialize TEST-length states.
#
# Safety:
#   - Uses TEST length/schema only.
#   - Does not read TEST real values.

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "CELL12F_V5_SELECTION" not in globals():
    p = REPORT_DIR_P / "cell12f_v5_locked_mask_selection.csv"
    if not p.exists():
        raise RuntimeError("[12.f.V5.2] Run 12.f.V5.1 first.")
    CELL12F_V5_SELECTION = pd.read_csv(p)

mat, audit_rows = {}, []

for i, row in CELL12F_V5_SELECTION.iterrows():
    col = str(row["col"])
    materializer = str(row["materializer_class_v5"])
    tv = CELL12F_V5_MASK_CACHE[col]["tv"]

    if (i + 1) == 1 or (i + 1) % 25 == 0 or (i + 1) == len(CELL12F_V5_SELECTION):
        _f5_log(f"V5.2 materializing {i+1}/{len(CELL12F_V5_SELECTION)} | {col} | {materializer}")

    if materializer == "exact_one_constant":
        arr = np.ones(N_TEST_12F5, dtype=np.uint8)
    elif materializer == "exact_zero_constant":
        arr = np.zeros(N_TEST_12F5, dtype=np.uint8)
    elif materializer in {"scope_excluded_tail_replay", "tail_replay"}:
        arr = _tail_replay_12f5(tv, N_TEST_12F5)
    elif materializer == "head_replay":
        arr = _head_replay_12f5(tv, N_TEST_12F5)
    elif materializer == "circular_replay":
        arr = _circular_replay_12f5(tv, N_TEST_12F5, _seed_12f5(col + "::mask_v5_final"))
    elif materializer == "block_bootstrap_8192":
        arr = _block_bootstrap_12f5(tv, N_TEST_12F5, _seed_12f5(col + "::mask_v5_final"), block=8192)
    elif materializer == "block_bootstrap_32768":
        arr = _block_bootstrap_12f5(tv, N_TEST_12F5, _seed_12f5(col + "::mask_v5_final"), block=32768)
    elif materializer == "run_bootstrap":
        arr = _run_bootstrap_12f5(tv, N_TEST_12F5, _seed_12f5(col + "::mask_v5_final"))
    elif materializer == "rare_episode_bootstrap":
        arr = _rare_episode_bootstrap_12f5(tv, N_TEST_12F5, _seed_12f5(col + "::mask_v5_final"))
    elif materializer == "exact_rate_spread":
        arr = _exact_rate_spread_12f5(tv, N_TEST_12F5, _seed_12f5(col + "::mask_v5_final"))
    else:
        raise RuntimeError(f"[12.f.V5.2] Unknown materializer {materializer!r} for {col}")

    arr = arr.astype(np.uint8)
    mat[col] = arr
    s = _stats_12f5(arr)
    tv_s = CELL12F_V5_MASK_CACHE[col]["tv_stats"]

    audit_rows.append({
        "col": col,
        "semantic_mode_v5": row["semantic_mode_v5"],
        "selected_generator": row["selected_generator"],
        "materializer_class_v5": materializer,
        "v5_claim_status_trainval_only": row["v5_claim_status_trainval_only"],
        "claim_included": _boolish_12f5(row["claim_included"]),
        "scope_excluded": _boolish_12f5(row["scope_excluded"]),
        "trainval_state_one_rate": tv_s["state_one_rate"],
        "synthetic_state_one_rate": s["state_one_rate"],
        "abs_state_one_rate_delta_vs_trainval": abs(s["state_one_rate"] - tv_s["state_one_rate"]),
        "synthetic_transition_rate": s["transition_rate"],
        "synthetic_all_max": s["all_max"],
        "synthetic_one_max": s["one_max"],
        "synthetic_zero_max": s["zero_max"],
        "TEST_real_values_used": False,
    })

mask_df = pd.DataFrame(mat)[CELL12F_V5_SELECTION["col"].astype(str).tolist()]

out_paths = [
    OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_TEST.parquet",
    OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_FINAL_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_OBSERVABILITY_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_MASK_FINAL_TEST.parquet",
    OUT_SYN_P / "IOT_VALUE_MASK_FINAL_TEST.parquet",
]
for p in out_paths:
    mask_df.to_parquet(p, index=False)

audit = pd.DataFrame(audit_rows)
audit.to_csv(REPORT_DIR_P / "cell12f_v5_mask_materialization_audit.csv", index=False)

summary = {
    "cell": "12.f.V5.2",
    "role": "observability_mask_v5_test_length_materialization",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "targets": int(mask_df.shape[1]),
    "shape": list(mask_df.shape),
    "mean_abs_state_one_rate_delta_vs_trainval": float(audit["abs_state_one_rate_delta_vs_trainval"].mean()),
    "max_abs_state_one_rate_delta_vs_trainval": float(audit["abs_state_one_rate_delta_vs_trainval"].max()),
    "materializer_class_counts": audit["materializer_class_v5"].value_counts().to_dict(),
    "TEST_real_values_used": False,
    "TEST_length_schema_only": True,
    "selection_done_here": False,
    "synthetic_values_mutated": True,
    "post_TEST_repair_done_here": False,
    "same_df_te_v5_status": "development_only_unless_fresh_holdout_used",
    "outputs": [str(p) for p in out_paths],
}
(REPORT_DIR_P / "cell12f_v5_materialization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
(CONTRACT_DIR_P / "cell12f_v5_materialization_contract.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12F_V5_MASK_TEST = mask_df
CELL12F_V5_MATERIALIZATION_SUMMARY = summary

_f5_log(f"V5.2 materialization complete | shape={mask_df.shape} | contract={CONTRACT_DIR_P / 'cell12f_v5_materialization_contract.json'}")

if display is not None:
    display(audit.head(20))

