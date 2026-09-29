# %% CELL 12.c.5R — Continuous final value enforcement replacement
# Purpose:
#   Enforce finite numeric/domain constraints on 12.c.4R materialized continuous values using TRAIN-only ranges.
#
# Safety:
#   - Does not read TEST values.
#   - Does not repair based on TEST QA.
#   - Does not change selector decisions.
#   - Writes canonical continuous final artifacts for downstream Q3/12.g compatibility.

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = None


def _c5r_log(msg):
    print(f"[Cell12.c.5R] {msg}")


def _c5r_path(x):
    if x is None:
        return None
    try:
        return Path(str(x)).expanduser().resolve()
    except Exception:
        return None


OUTDIR_P = _c5r_path(globals().get("OUTDIR", None)) or _c5r_path(os.environ.get("CPS_OUTDIR", ""))
OUT_SYN_P = _c5r_path(globals().get("OUT_SYN", None)) or OUTDIR_P / "synthetic"
REPORT_DIR_P = _c5r_path(globals().get("REPORT_DIR", None)) or OUTDIR_P / "reports"
CONTRACT_DIR_P = _c5r_path(globals().get("CONTRACT_DIR", None)) or OUTDIR_P / "artifacts" / "contracts"

for p in [OUT_SYN_P, REPORT_DIR_P, CONTRACT_DIR_P]:
    p.mkdir(parents=True, exist_ok=True)

values_path = OUT_SYN_P / "IOT_SELECTED_CONTINUOUS_TEST.parquet"
if not values_path.exists():
    raise RuntimeError("[Cell12.c.5R] Missing 12.c.4R selected values. Run Cell 12.c.4R first.")

selection_path = REPORT_DIR_P / "cell12c3R_locked_iot_value_selection.csv"
if not selection_path.exists():
    selection_path = REPORT_DIR_P / "cell12c3_locked_iot_value_selection.csv"
selection = pd.read_csv(selection_path)

values = pd.read_parquet(values_path)

if "df_tr" in globals() and isinstance(globals()["df_tr"], pd.DataFrame):
    DF_TR_LOCAL = globals()["df_tr"]
elif "DF_TR" in globals() and isinstance(globals()["DF_TR"], pd.DataFrame):
    DF_TR_LOCAL = globals()["DF_TR"]
else:
    DF_TR_LOCAL = None

final = values.copy()
audit_rows = []

for col in final.columns:
    arr = pd.to_numeric(final[col], errors="coerce").to_numpy(dtype="float64")
    before = arr.copy()

    if DF_TR_LOCAL is not None and col in DF_TR_LOCAL.columns:
        tr = pd.to_numeric(DF_TR_LOCAL[col], errors="coerce").dropna().to_numpy(dtype="float64")
    else:
        tr = np.asarray([], dtype="float64")

    if len(tr):
        med = float(np.median(tr))
        q001 = float(np.quantile(tr, 0.001))
        q999 = float(np.quantile(tr, 0.999))
        tmin = float(np.min(tr))
        tmax = float(np.max(tr))
        # Loose TRAIN-derived clip range, no TEST information.
        lo = min(tmin, q001)
        hi = max(tmax, q999)
        if np.isfinite(lo) and np.isfinite(hi) and lo <= hi:
            arr = np.clip(arr, lo, hi)
    else:
        med = 0.0
        lo = None
        hi = None

    nonfinite_before = int((~np.isfinite(arr)).sum())
    arr[~np.isfinite(arr)] = med

    final[col] = arr

    changed_total = int(np.sum(~np.isclose(np.nan_to_num(before, nan=med), arr, equal_nan=True)))
    audit_rows.append({
        "col": col,
        "selected_generator": selection.set_index("col").loc[col, "selected_generator"] if col in set(selection["col"].astype(str)) else "",
        "train_clip_lo": lo,
        "train_clip_hi": hi,
        "train_median_fill": med,
        "nonfinite_filled_n": nonfinite_before,
        "changed_by_enforcement_n": changed_total,
        "finite_after_rate": float(np.isfinite(arr).mean()) if len(arr) else 0.0,
        "TEST_real_values_used": False,
    })

mask = final.notna().astype("uint8")

final_path = OUT_SYN_P / "IOT_FINAL_CONTINUOUS_TEST.parquet"
final_alias_1 = OUT_SYN_P / "IOT_CONTINUOUS_FINAL_TEST.parquet"
final_alias_2 = OUT_SYN_P / "IOT_CONT_VALUE_FINAL_TEST.parquet"
mask_path = OUT_SYN_P / "IOT_CONT_VALUE_MASK_TEST.parquet"

final.to_parquet(final_path, index=False)
final.to_parquet(final_alias_1, index=False)
final.to_parquet(final_alias_2, index=False)
mask.to_parquet(mask_path, index=False)

audit = pd.DataFrame(audit_rows)
audit_path_r = REPORT_DIR_P / "cell12c5R_final_value_enforcement_audit.csv"
audit_path_canon = REPORT_DIR_P / "cell12c5_final_value_enforcement_audit.csv"
registry_path = REPORT_DIR_P / "cell12c5_publication_column_registry.csv"
ready_cols_path = REPORT_DIR_P / "cell12c5_publication_ready_columns.csv"

audit.to_csv(audit_path_r, index=False)
audit.to_csv(audit_path_canon, index=False)

registry = selection.copy()
registry["cell12c5R_final_value_artifact"] = str(final_path)
registry["cell12c5R_mask_artifact"] = str(mask_path)
registry["TEST_real_values_used_for_enforcement"] = False
registry.to_csv(registry_path, index=False)

ready = registry[registry.get("publication_scope_12c3R", pd.Series("", index=registry.index)).astype(str).eq("continuous_value_claim_included")].copy()
ready.to_csv(ready_cols_path, index=False)

summary = {
    "cell": "12.c.5R",
    "role": "continuous_final_value_enforcement_train_only",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "shape": list(final.shape),
    "targets": int(final.shape[1]),
    "changed_total": int(audit["changed_by_enforcement_n"].sum()),
    "nonfinite_filled_total": int(audit["nonfinite_filled_n"].sum()),
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "final_enforcement_done_here": True,
    "outputs": {
        "final_values": str(final_path),
        "final_alias_1": str(final_alias_1),
        "final_alias_2": str(final_alias_2),
        "mask": str(mask_path),
        "registry": str(registry_path),
    },
}
summary_path = REPORT_DIR_P / "cell12c5R_final_value_enforcement_summary.json"
contract_path = CONTRACT_DIR_P / "cell12c5_final_value_enforcement_contract_v1_4_THESIS.json"
contract_alias_path = CONTRACT_DIR_P / "cell12c5R_final_value_enforcement_contract.json"

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
contract_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
contract_alias_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12C5R_FINAL_VALUES = final
CELL12C5R_FINAL_MASK = mask
CELL12C5R_SUMMARY = summary
CELL12C5R_FINAL_VALUES_PATH = str(final_path)

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}
CFG["cell12c5R_active"] = True
CFG["cell12c5R_test_values_used"] = False

_c5r_log(f"Final enforcement complete | shape={final.shape} | changed_total={summary['changed_total']} | filled={summary['nonfinite_filled_total']}")
_c5r_log(f"Contract: {contract_path}")

if display is not None:
    display(audit.head(20))