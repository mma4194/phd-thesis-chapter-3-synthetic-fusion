import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

if "IOT_SELECTED_BINARY_TEST" not in globals():
    p = OUT_SYN_P / "IOT_SELECTED_BINARY_TEST.parquet"
    if not p.exists():
        raise RuntimeError("Run CELL 12.d.V6.2 first.")
    IOT_SELECTED_BINARY_TEST = pd.read_parquet(p)

df = IOT_SELECTED_BINARY_TEST.copy()
changed_total = 0
failure_rows = []

for col in df.columns:
    raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    before = raw.copy()
    raw[~np.isfinite(raw)] = 0.0
    enforced = (raw > 0.5).astype(np.uint8)

    changed = int(np.sum((np.nan_to_num(before, nan=0.0) > 0.5).astype(np.uint8) != enforced))
    changed_total += changed

    nonbinary_after = int(np.sum(~np.isin(enforced, [0, 1])))
    if nonbinary_after:
        failure_rows.append({"column": col, "nonbinary_after": nonbinary_after})

    df[col] = enforced

failure_count = len(failure_rows)
final_path = OUT_SYN_P / "IOT_FINAL_BINARY_TEST.parquet"
df.to_parquet(final_path, index=False)

audit = pd.DataFrame(failure_rows)
audit_path = REPORT_DIR_P / "cell12d5_binary_final_enforcement_audit.csv"
audit.to_csv(audit_path, index=False)

summary = {
    "cell": "12.d.V6.3",
    "role": "binary_v6_final_enforcement",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "shape": list(df.shape),
    "enforcement_failure_count": int(failure_count),
    "changed_total": int(changed_total),
    "TEST_real_values_used": False,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "final_enforcement_done_here": True,
}
contract_path = CONTRACT_DIR_P / "cell12d5_binary_final_enforcement_contract_v6_0_THESIS.json"
contract_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

IOT_FINAL_BINARY_TEST = df
CELL12D_V6_FINAL_BINARY_TEST_PATH = str(final_path)

_bv6_log(f"V6.3 final enforcement complete | shape={df.shape} | enforcement_failure_count={failure_count} | changed_total={changed_total}")
_bv6_log(f"Saved: {final_path}")