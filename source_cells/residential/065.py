import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "IOT_FINAL_BINARY_TEST" not in globals():
    p = OUT_SYN_P / "IOT_FINAL_BINARY_TEST.parquet"
    if not p.exists():
        raise RuntimeError("Run CELL 12.d.V6.3 first.")
    IOT_FINAL_BINARY_TEST = pd.read_parquet(p)

pre_rows = []

for _, row in CELL12D_V6_SELECTION_DF.iterrows():
    col = str(row["column"])
    family = str(row["selected_generator"])
    materializer_class = str(row["materializer_class_v6"])
    claim_status = str(row["v6_claim_status_trainval_only"])
    claim_included = bool(row["claim_included"])

    cache = CELL12D_V6_TRAINVAL_CACHE[col]
    tr = cache["tr_stats"]
    va = cache["va_stats"]
    tv = cache["tv_stats"]

    vals, _ = _bv6_bin_values(IOT_FINAL_BINARY_TEST[col])
    s = _bv6_stats(vals)
    tv_rate = float(tv["rate"]) if np.isfinite(tv["rate"]) else 0.0

    pre_rows.append({
        "column": col,
        "selected_generator": family,
        "materializer_class_v6": materializer_class,
        "v6_claim_status_trainval_only": claim_status,
        "claim_included": claim_included,
        "scope_excluded": bool(not claim_included),
        "train_rate": tr["rate"],
        "val_rate": va["rate"],
        "trainval_rate": tv_rate,
        "synthetic_rate": s["rate"],
        "abs_rate_delta_vs_trainval": abs(s["rate"] - tv_rate),
        "synthetic_transition_rate": s["transition_rate"],
        "synthetic_all_max": s["all_max"],
        "synthetic_one_max": s["one_max"],
        "synthetic_zero_max": s["zero_max"],
        "TEST_real_values_used": False,
    })

pre = pd.DataFrame(pre_rows)
pre_path = REPORT_DIR_P / "cell12d_v6_pretest_trainval_materialization_audit.csv"
pre.to_csv(pre_path, index=False)

included = pre["claim_included"]

summary = {
    "cell": "12.d.V6.4",
    "role": "binary_v6_pretest_trainval_materialization_audit",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "n_targets": int(len(pre)),
    "included_n_trainval_only": int(included.sum()),
    "scope_excluded_n_trainval_only": int((~included).sum()),
    "mean_abs_rate_delta_vs_trainval_all": float(pre["abs_rate_delta_vs_trainval"].mean()),
    "max_abs_rate_delta_vs_trainval_all": float(pre["abs_rate_delta_vs_trainval"].max()),
    "mean_abs_rate_delta_vs_trainval_included": float(pre.loc[included, "abs_rate_delta_vs_trainval"].mean()),
    "max_abs_rate_delta_vs_trainval_included": float(pre.loc[included, "abs_rate_delta_vs_trainval"].max()),
    "claim_status_counts_trainval_only": pre["v6_claim_status_trainval_only"].value_counts().to_dict(),
    "materializer_class_counts": pre["materializer_class_v6"].value_counts().to_dict(),
    "TEST_real_values_used": False,
}
summary_path = REPORT_DIR_P / "cell12d_v6_pretest_trainval_materialization_audit_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12D_V6_PRETEST_AUDIT = pre
CELL12D_V6_PRETEST_SUMMARY = summary

_bv6_log(
    f"V6.4 complete | included={summary['included_n_trainval_only']} | "
    f"scope_excluded={summary['scope_excluded_n_trainval_only']} | "
    f"included mean_rate_delta={summary['mean_abs_rate_delta_vs_trainval_included']:.6f} | "
    f"included max_rate_delta={summary['max_abs_rate_delta_vs_trainval_included']:.6f}"
)
_bv6_log(f"Audit: {pre_path}")