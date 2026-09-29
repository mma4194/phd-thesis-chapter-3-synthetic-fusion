import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "CELL12D_V6_SELECTION_DF" not in globals():
    selection_path = REPORT_DIR_P / "cell12d6_locked_binary_v6_claim_scoped_selection.csv"
    if not selection_path.exists():
        raise RuntimeError("Run CELL 12.d.V6.1 first.")
    CELL12D_V6_SELECTION_DF = pd.read_csv(selection_path)

mat = {}
audit_rows = []

for i, row in CELL12D_V6_SELECTION_DF.iterrows():
    col = str(row["column"])
    family = str(row["selected_generator"])
    materializer_class = str(row["materializer_class_v6"])
    claim_status = str(row["v6_claim_status_trainval_only"])
    claim_included = bool(row["claim_included"])

    if (i + 1) == 1 or (i + 1) % 10 == 0 or (i + 1) == len(CELL12D_V6_SELECTION_DF):
        _bv6_log(f"V6.2 progress {i+1}/{len(CELL12D_V6_SELECTION_DF)} | col={col} | class={materializer_class}")

    cache = CELL12D_V6_TRAINVAL_CACHE[col]
    tr = cache["tr_stats"]
    va = cache["va_stats"]
    tv = cache["tv_stats"]
    tv_vals = cache["tv_vals"]

    N = N_TEST_BV6
    seed = _bv6_seed_for_col(col, "binary-v6-materialization")
    target_rate = float(tv["rate"]) if np.isfinite(tv["rate"]) else 0.0

    matrix_completion_only = not claim_included

    if materializer_class == "exact_constant":
        if tr["support"] == "constant_1" and va["support"] == "constant_1":
            arr = np.ones(N, dtype=np.uint8)
        else:
            arr = np.zeros(N, dtype=np.uint8)

    elif materializer_class == "exact_zero_constant":
        arr = np.zeros(N, dtype=np.uint8)

    elif materializer_class == "exact_one_constant":
        arr = np.ones(N, dtype=np.uint8)

    elif materializer_class == "trainval_tail_replay":
        arr = _bv6_tail_replay(tv_vals, N)

    elif materializer_class == "trainval_circular_replay":
        arr = _bv6_circular_replay(tv_vals, N, seed)

    elif materializer_class == "trainval_block_bootstrap_32768":
        arr = _bv6_block_bootstrap(tv_vals, N, seed, block=32768)

    elif materializer_class == "scope_excluded_matrix_completion":
        # Matrix-completion only. These columns are not part of the V6 included binary claim.
        # Use TRAIN/VAL central tendency and conservative replay where dynamic.
        if target_rate <= 0.001:
            arr = np.zeros(N, dtype=np.uint8)
        elif target_rate >= 0.999:
            arr = np.ones(N, dtype=np.uint8)
        else:
            arr = _bv6_tail_replay(tv_vals, N)

    else:
        raise RuntimeError(f"Unknown V6 materializer_class_v6={materializer_class!r} for {col}")

    arr = arr.astype(np.uint8)
    mat[col] = arr

    s = _bv6_stats(arr)
    audit_rows.append({
        "column": col,
        "selected_generator": family,
        "materializer_class_v6": materializer_class,
        "v6_claim_status_trainval_only": claim_status,
        "claim_included": claim_included,
        "scope_excluded": bool(not claim_included),
        "matrix_completion_only": bool(matrix_completion_only),
        "target_rate_trainval": target_rate,
        "synthetic_rate": s["rate"],
        "abs_rate_delta_vs_trainval": abs(s["rate"] - target_rate),
        "synthetic_transition_rate": s["transition_rate"],
        "synthetic_all_max": s["all_max"],
        "synthetic_one_max": s["one_max"],
        "synthetic_zero_max": s["zero_max"],
        "TEST_real_values_used": False,
    })

binary_df = pd.DataFrame(mat)
binary_df = binary_df[CELL12D_V6_SELECTION_DF["column"].tolist()]

selected_path = OUT_SYN_P / "IOT_SELECTED_BINARY_TEST.parquet"
binary_df.to_parquet(selected_path, index=False)

audit_df = pd.DataFrame(audit_rows)
audit_path = REPORT_DIR_P / "cell12d6_binary_v6_materialization_audit.csv"
legacy_audit_path = REPORT_DIR_P / "cell12d4_binary_test_materialization_audit.csv"
audit_df.to_csv(audit_path, index=False)
audit_df.to_csv(legacy_audit_path, index=False)

summary = {
    "cell": "12.d.V6.2",
    "role": "binary_v6_test_length_materialization_from_locked_trainval_class_rules",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "shape": list(binary_df.shape),
    "selected_generator_counts": CELL12D_V6_SELECTION_DF["selected_generator"].value_counts().to_dict(),
    "materializer_class_counts": CELL12D_V6_SELECTION_DF["materializer_class_v6"].value_counts().to_dict(),
    "included_n_trainval_only": int(audit_df["claim_included"].sum()),
    "scope_excluded_n_trainval_only": int((~audit_df["claim_included"]).sum()),
    "mean_abs_rate_delta_vs_trainval_all": float(audit_df["abs_rate_delta_vs_trainval"].mean()),
    "max_abs_rate_delta_vs_trainval_all": float(audit_df["abs_rate_delta_vs_trainval"].max()),
    "mean_abs_rate_delta_vs_trainval_included": float(audit_df.loc[audit_df["claim_included"], "abs_rate_delta_vs_trainval"].mean()),
    "max_abs_rate_delta_vs_trainval_included": float(audit_df.loc[audit_df["claim_included"], "abs_rate_delta_vs_trainval"].max()),
    "TEST_real_values_used": False,
    "TEST_length_schema_only": True,
    "selection_done_here": False,
    "generator_fit_done_here": True,
    "per_column_TEST_winners_used": False,
    "development_only_on_current_df_te": True,
}
summary_path = REPORT_DIR_P / "cell12d_v6_materialization_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

contract = {
    **summary,
    "contract_version": "v6_0_THESIS",
    "synthetic_path": str(selected_path),
}
contract_path = CONTRACT_DIR_P / "cell12d4_binary_test_materialization_contract_v6_0_THESIS.json"
contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")

IOT_SELECTED_BINARY_TEST = binary_df
CELL12D_V6_MATERIALIZATION_AUDIT = audit_df
CELL12D_V6_SELECTED_BINARY_TEST_PATH = str(selected_path)

_bv6_log(
    f"V6.2 materialization complete | shape={binary_df.shape} | "
    f"included={summary['included_n_trainval_only']} | scope_excluded={summary['scope_excluded_n_trainval_only']}"
)
_bv6_log(
    f"Rate delta vs TRAIN/VAL | included mean={summary['mean_abs_rate_delta_vs_trainval_included']:.6f} | "
    f"included max={summary['max_abs_rate_delta_vs_trainval_included']:.6f}"
)
_bv6_log(f"Saved: {selected_path}")
_bv6_log(f"Contract: {contract_path}")