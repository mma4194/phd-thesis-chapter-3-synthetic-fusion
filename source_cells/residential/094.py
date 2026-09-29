# %% CELL 12.c.R.8 — Apply continuous-value observability mask before Cell 12.g assembly
# Purpose:
#   Cell 12.g fails if continuous values are finite when their continuous-value
#   observability mask is inactive.
#
#   This cell applies the already-produced continuous value mask to the active
#   12.c.R continuous value matrix:
#
#       if mask[col] == 0:
#           value[col] = NaN
#
# Safety:
#   - Does not rerun generation, selection, fitting, materialization, or QA.
#   - Does not read real TEST values.
#   - Does not change active branch decisions.
#   - Does not improve QA metrics.
#   - Performs deterministic namespace-observability enforcement only.
#
# Important:
#   This intentionally introduces NaNs into inactive cells. That is expected.
#   Do not use the older 12.c.R.7 "finite only" check after this cell.

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

def _cR8_log(msg):
    print(f"[12.c.R.8 mask-enforce] {msg}")

# ---------------------------------------------------------------------
# 0. Resolve roots
# ---------------------------------------------------------------------
for name in ["OUTDIR", "OUT_SYN", "REPORT_DIR", "CONTRACT_DIR"]:
    if name not in globals():
        raise RuntimeError(f"[12.c.R.8] Missing required global: {name}")

OUTDIR_P = Path(str(OUTDIR)).expanduser().resolve()
OUT_SYN_P = Path(str(OUT_SYN)).expanduser().resolve()
REPORT_DIR_P = Path(str(REPORT_DIR)).expanduser().resolve()
CONTRACT_DIR_P = Path(str(CONTRACT_DIR)).expanduser().resolve()

OUT_SYN_P.mkdir(parents=True, exist_ok=True)
REPORT_DIR_P.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_P.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load active continuous values
# ---------------------------------------------------------------------
value_candidates = [
    OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_CONTINUOUS_VALUES_TEST.parquet",
    OUT_SYN_P / "IOT_CONTINUOUS_FINAL_VALUES_TEST.parquet",
    OUT_SYN_P / "IOT_CONTINUOUS_VALUES_FINAL_TEST.parquet",
]

value_path = None
for p in value_candidates:
    if p.exists():
        value_path = p
        break

if value_path is None:
    raise RuntimeError(
        "[12.c.R.8] Could not find active continuous value matrix. "
        "Run 12.c.R.7 first."
    )

values = pd.read_parquet(value_path)
value_cols = list(map(str, values.columns))
N = int(values.shape[0])

_cR8_log(f"Loaded values: {value_path} | shape={values.shape}")

# ---------------------------------------------------------------------
# 2. Load continuous-value mask
# ---------------------------------------------------------------------
mask_candidates = [
    OUT_SYN_P / "IOT_CONT_VALUE_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_CONTINUOUS_VALUE_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_VALUE_MASK_CONTINUOUS_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_VALUE_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_VALUE_MASK_FINAL_TEST.parquet",
]

valid_masks = []

for p in mask_candidates:
    if not p.exists():
        continue
    try:
        m = pd.read_parquet(p)
    except Exception as e:
        _cR8_log(f"Could not read mask candidate {p.name}: {e}")
        continue

    m_cols = list(map(str, m.columns))
    present = [c for c in value_cols if c in m_cols]

    if m.shape[0] != N:
        _cR8_log(f"Reject mask {p.name}: row count {m.shape[0]} != {N}")
        continue

    if len(present) < max(1, int(0.90 * len(value_cols))):
        _cR8_log(f"Reject mask {p.name}: value-column coverage {len(present)}/{len(value_cols)}")
        continue

    valid_masks.append((p, m[present].copy(), len(present)))

if not valid_masks:
    raise RuntimeError(
        "[12.c.R.8] Could not find a continuous-value observability mask "
        "covering the active 12.c.R value columns."
    )

# Prefer the highest coverage; if tied, prefer explicit continuous mask filename.
valid_masks = sorted(
    valid_masks,
    key=lambda t: (
        t[2],
        "CONT" in t[0].name.upper(),
        "FINAL" in t[0].name.upper(),
        len(t[0].name),
    ),
    reverse=True,
)

mask_path, mask_df, covered_n = valid_masks[0]
mask_cols = list(map(str, mask_df.columns))

_cR8_log(f"Loaded mask: {mask_path} | shape={mask_df.shape} | coverage={covered_n}/{len(value_cols)}")

# ---------------------------------------------------------------------
# 3. Validate mask binary semantics
# ---------------------------------------------------------------------
bad_mask_cols = {}
for c in mask_cols:
    arr = pd.to_numeric(mask_df[c], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if not finite.all():
        bad_mask_cols[c] = "nonfinite"
        continue
    unique = set(np.unique(arr).tolist())
    if not unique.issubset({0.0, 1.0}):
        bad_mask_cols[c] = sorted(list(unique))[:10]

if bad_mask_cols:
    sample = dict(list(bad_mask_cols.items())[:10])
    raise RuntimeError(f"[12.c.R.8] Mask is not binary for sample columns: {sample}")

# ---------------------------------------------------------------------
# 4. Apply mask
# ---------------------------------------------------------------------
masked = values.copy()

before_finite_total = int(np.isfinite(masked.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)).sum())
changed_to_nan_total = 0
inactive_cells_total = 0
per_col_rows = []

for c in value_cols:
    if c not in mask_cols:
        # Leave uncovered columns unchanged, but record them.
        per_col_rows.append({
            "col": c,
            "mask_available": False,
            "inactive_n": 0,
            "finite_before_inactive_n": 0,
            "changed_to_nan_n": 0,
        })
        continue

    v = pd.to_numeric(masked[c], errors="coerce").to_numpy(dtype=float)
    m = pd.to_numeric(mask_df[c], errors="coerce").to_numpy(dtype=float)

    inactive = (m <= 0.5)
    finite_before_inactive = inactive & np.isfinite(v)

    n_inactive = int(inactive.sum())
    n_finite_inactive = int(finite_before_inactive.sum())

    v[finite_before_inactive] = np.nan
    masked[c] = v

    inactive_cells_total += n_inactive
    changed_to_nan_total += n_finite_inactive

    per_col_rows.append({
        "col": c,
        "mask_available": True,
        "inactive_n": n_inactive,
        "finite_before_inactive_n": n_finite_inactive,
        "changed_to_nan_n": n_finite_inactive,
    })

after_arr = masked.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
after_finite_total = int(np.isfinite(after_arr).sum())

# Verify no finite values remain under inactive mask for covered columns.
remaining_violations = 0
for c in mask_cols:
    v = pd.to_numeric(masked[c], errors="coerce").to_numpy(dtype=float)
    m = pd.to_numeric(mask_df[c], errors="coerce").to_numpy(dtype=float)
    remaining_violations += int(((m <= 0.5) & np.isfinite(v)).sum())

if remaining_violations != 0:
    raise RuntimeError(
        f"[12.c.R.8] Mask enforcement failed; remaining inactive finite violations={remaining_violations}"
    )

# ---------------------------------------------------------------------
# 5. Write masked values to the legacy paths consumed by 12.g
# ---------------------------------------------------------------------
output_paths = [
    OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_CONTINUOUS_VALUES_TEST.parquet",
    OUT_SYN_P / "IOT_CONTINUOUS_FINAL_VALUES_TEST.parquet",
    OUT_SYN_P / "IOT_CONTINUOUS_VALUES_FINAL_TEST.parquet",
]

for p in output_paths:
    masked.to_parquet(p, index=False)

# Also keep an explicit audit-named output.
explicit_masked_path = OUT_SYN_P / "IOT_FINAL_VALUES_TEST_MASKED_BY_CONTINUOUS_OBSERVABILITY.parquet"
masked.to_parquet(explicit_masked_path, index=False)

# ---------------------------------------------------------------------
# 6. Publish globals
# ---------------------------------------------------------------------
CELL12C_FINAL_VALUES_TEST_DF = masked
IOT_FINAL_VALUES_TEST_DF = masked
CELL12C_FINAL_VALUES_TEST_PATH = str(OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet")
IOT_FINAL_VALUES_TEST_PATH = str(OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet")
CELL12C_ACTIVE_BRANCH = "12.c.R"

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}

CFG["active_continuous_branch"] = "12.c.R"
CFG["cell12g_continuous_mask_enforcement_source"] = "12.c.R.8"
CFG["cell12g_continuous_values_masked_for_inactive_intervals"] = True

# ---------------------------------------------------------------------
# 7. Write audit/contract
# ---------------------------------------------------------------------
def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

per_col_df = pd.DataFrame(per_col_rows)
per_col_path = REPORT_DIR_P / "cell12cR8_continuous_mask_enforcement_by_column.csv"
per_col_df.to_csv(per_col_path, index=False)

summary = {
    "cell": "12.c.R.8",
    "role": "continuous_value_observability_mask_enforcement_before_cell12g",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "active_continuous_branch": "12.c.R",
    "value_source_path": str(value_path),
    "mask_source_path": str(mask_path),
    "shape": list(masked.shape),
    "value_cols_n": int(len(value_cols)),
    "mask_covered_cols_n": int(covered_n),
    "inactive_cells_total_for_covered_cols": int(inactive_cells_total),
    "finite_before_total": int(before_finite_total),
    "finite_after_total": int(after_finite_total),
    "changed_to_nan_total": int(changed_to_nan_total),
    "remaining_inactive_finite_violations_after": int(remaining_violations),
    "uncovered_cols": [c for c in value_cols if c not in mask_cols],
    "policy": {
        "synthetic_values_mutated": True,
        "mutation_type": "deterministic_namespace_observability_enforcement",
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "TEST_real_values_read": False,
        "qa_metrics_improved_here": False,
        "branch_decisions_changed": False,
    },
    "outputs": {
        "legacy_values_path": str(OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet"),
        "explicit_masked_values_path": str(explicit_masked_path),
        "per_column_audit": str(per_col_path),
    },
    "sha256": {
        "legacy_values": _sha256_file(OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet"),
        "explicit_masked_values": _sha256_file(explicit_masked_path),
    },
}

report_path = REPORT_DIR_P / "cell12cR8_continuous_mask_enforcement_report.json"
contract_path = CONTRACT_DIR_P / "cell12cR8_continuous_mask_enforcement_contract.json"

for p in [report_path, contract_path]:
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12CR8_MASK_ENFORCEMENT = summary



# ---------------------------------------------------------------------
# 8. Compatibility globals for downstream 12.g
# ---------------------------------------------------------------------
# These aliases ensure legacy cells do not fall back to stale unmasked continuous
# artifacts or older in-memory dataframes.
IOT_FINAL_VALUES_TEST = masked
IOT_FINAL_VALUES_TEST_DF = masked
CELL12C_FINAL_VALUES_TEST_DF = masked
CELL12C5_FINAL_VALUES_TEST_DF = masked
CELL12C5_FINAL_VALUES_TEST_PATH = str(OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet")
CELL12C_FINAL_VALUES_TEST_PATH = str(OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet")
IOT_FINAL_VALUES_TEST_PATH = str(OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet")

# Publish the active continuous availability mask under the exact legacy names
# inspected by Cell 12.g. This is the active TEST-length synthetic mask, not a
# real TEST value source.
IOT_VALUE_AVAIL_SYN_DF = mask_df[value_cols].copy()
IOT_VALUE_AVAIL_SYN = IOT_VALUE_AVAIL_SYN_DF
VALUE_AVAIL_SYN = IOT_VALUE_AVAIL_SYN_DF

CFG["cell12g_continuous_values_path"] = str(OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet")
CFG["cell12g_continuous_availability_mask_source"] = str(mask_path)

_cR8_log(f"Applied mask source: {mask_path}")
_cR8_log(f"changed_to_nan_total = {changed_to_nan_total}")
_cR8_log(f"remaining_inactive_finite_violations_after = {remaining_violations}")
_cR8_log(f"Wrote masked legacy values: {OUT_SYN_P / 'IOT_FINAL_VALUES_TEST.parquet'}")
_cR8_log(f"Contract: {contract_path}")
_cR8_log("PASS: continuous values are masked for inactive intervals before 12.g.")
