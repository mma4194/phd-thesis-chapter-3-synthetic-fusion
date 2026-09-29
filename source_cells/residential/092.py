# %% CELL 12.c.R.7 — 12.c.R -> Cell 12.g continuous-values compatibility handoff
# Purpose:
#   Cell 12.g expects the final continuous IoT value matrix at:
#       OUT_SYN/IOT_FINAL_VALUES_TEST.parquet
#
#   The active replacement branch 12.c.R may write the same matrix under a
#   newer replacement-branch filename. This cell locates the active 12.c.R
#   final continuous TEST matrix, verifies it against 12.c.6R QA targets,
#   and writes the legacy alias expected by 12.g.
#
# Safety:
#   - Does not read real TEST values.
#   - Does not rerun selection, fitting, materialization, or QA.
#   - Does not alter values; it writes a compatibility alias from the active
#     synthetic continuous artifact.
#   - Refuses to use df_te / real TEST dataframes as a source.

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

def _cR7_log(msg):
    print(f"[12.c.R.7 handoff] {msg}")

# ---------------------------------------------------------------------
# 0. Resolve roots
# ---------------------------------------------------------------------
for name in ["OUTDIR", "OUT_SYN", "REPORT_DIR", "CONTRACT_DIR"]:
    if name not in globals():
        raise RuntimeError(f"[12.c.R.7] Missing required global: {name}")

OUTDIR_P = Path(str(OUTDIR)).expanduser().resolve()
OUT_SYN_P = Path(str(OUT_SYN)).expanduser().resolve()
REPORT_DIR_P = Path(str(REPORT_DIR)).expanduser().resolve()
CONTRACT_DIR_P = Path(str(CONTRACT_DIR)).expanduser().resolve()
ARTIFACT_DIR_P = Path(str(globals().get("ARTIFACT_DIR", globals().get("ARTDIR", OUTDIR_P / "artifacts")))).expanduser().resolve()

OUT_SYN_P.mkdir(parents=True, exist_ok=True)
REPORT_DIR_P.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_P.mkdir(parents=True, exist_ok=True)

target_legacy_path = OUT_SYN_P / "IOT_FINAL_VALUES_TEST.parquet"

# ---------------------------------------------------------------------
# 1. Resolve expected continuous columns from active 12.c.R QA outputs
# ---------------------------------------------------------------------
expected_cols = []

candidate_col_sources = [
    REPORT_DIR_P / "cell12c6_final_test_qa_metrics.csv",
    REPORT_DIR_P / "cell12c6_final_selected_generator_table.csv",
    REPORT_DIR_P / "cell12c5_final_value_enforcement_audit.csv",
    REPORT_DIR_P / "cell12c4R_materialization_audit.csv",
    REPORT_DIR_P / "cell12c3R_locked_selection.csv",
    REPORT_DIR_P / "cell12c3R_trainval_only_selection.csv",
]

for p in candidate_col_sources:
    if not p.exists():
        continue
    try:
        df_tmp = pd.read_csv(p, nrows=10000)
    except Exception:
        continue
    col_name = None
    for c in ["col", "column", "target", "target_col"]:
        if c in df_tmp.columns:
            col_name = c
            break
    if col_name:
        vals = [str(x) for x in df_tmp[col_name].dropna().astype(str).tolist()]
        vals = [x for x in dict.fromkeys(vals) if x.startswith("iot__")]
        if len(vals) >= 100:
            expected_cols = vals
            _cR7_log(f"Expected continuous columns resolved from {p.name}: {len(expected_cols)}")
            break

if not expected_cols:
    raise RuntimeError(
        "[12.c.R.7] Could not resolve expected 12.c.R continuous target columns. "
        "Expected cell12c6_final_test_qa_metrics.csv or equivalent to exist."
    )

# Resolve TEST length from df_te if available, otherwise from Q3/continuous files.
N_TE = None
if "df_te" in globals() and isinstance(globals()["df_te"], pd.DataFrame):
    N_TE = int(len(globals()["df_te"]))
elif "DF_TE" in globals() and isinstance(globals()["DF_TE"], pd.DataFrame):
    N_TE = int(len(globals()["DF_TE"]))

if N_TE is None:
    # Last resort: use active Q3 mask file length if present.
    for p in [
        OUT_SYN_P / "IOT_FINAL_OBSERVABILITY_MASK_TEST.parquet",
        OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_FINAL_TEST.parquet",
        OUT_SYN_P / "IOT_FINAL_BINARY_TEST.parquet",
    ]:
        if p.exists():
            try:
                N_TE = int(len(pd.read_parquet(p, columns=[])))
            except Exception:
                try:
                    N_TE = int(pd.read_parquet(p).shape[0])
                except Exception:
                    pass
            if N_TE is not None:
                break

if N_TE is None:
    raise RuntimeError("[12.c.R.7] Could not resolve TEST length.")

_cR7_log(f"Expected continuous cols = {len(expected_cols)} | N_TE = {N_TE}")

# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _json_paths(obj):
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_json_paths(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_json_paths(v))
    elif isinstance(obj, str):
        s = obj.strip()
        if s.endswith(".parquet") or ".parquet" in s:
            out.append(s)
    return out

def _is_bad_continuous_source_name(path):
    name = path.name.lower()
    bad_tokens = [
        "mask",
        "binary",
        "protocol",
        "driver",
        "observability",
        "a0_",
        "a1_",
        "a2_",
    ]
    if any(tok in name for tok in bad_tokens):
        return True
    return False

def _score_candidate_path(path):
    name = path.name.lower()
    score = 0

    # Positive indicators.
    if "iot" in name:
        score += 10
    if "value" in name or "values" in name:
        score += 15
    if "continuous" in name or "cont" in name:
        score += 15
    if "final" in name:
        score += 25
    if "test" in name:
        score += 10
    if "selected" in name:
        score += 3

    # Negative indicators.
    if "mask" in name:
        score -= 200
    if "binary" in name:
        score -= 200
    if "protocol" in name:
        score -= 200
    if "driver" in name:
        score -= 200
    if "observability" in name:
        score -= 200
    if "val" in name:
        score -= 20

    return score

def _try_load_and_validate(path):
    path = Path(path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return None
    if path.suffix.lower() != ".parquet":
        return None
    if _is_bad_continuous_source_name(path):
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {"path": str(path), "valid": False, "reason": f"read_failed:{e}"}

    rows, cols = df.shape
    col_list = list(map(str, df.columns))
    expected_set = set(expected_cols)
    present_expected = [c for c in expected_cols if c in col_list]
    coverage = len(present_expected) / max(1, len(expected_cols))

    if rows != N_TE:
        return {
            "path": str(path),
            "valid": False,
            "reason": f"row_count_mismatch:{rows}!={N_TE}",
            "rows": rows,
            "cols": cols,
            "expected_col_coverage": coverage,
        }

    if coverage < 0.95:
        return {
            "path": str(path),
            "valid": False,
            "reason": f"expected_col_coverage_too_low:{coverage:.3f}",
            "rows": rows,
            "cols": cols,
            "expected_col_coverage": coverage,
        }

    # Normalize to expected continuous columns only, preserving expected order.
    df_norm = df[present_expected].copy()
    if len(present_expected) != len(expected_cols):
        missing = sorted(expected_set - set(present_expected))[:20]
        return {
            "path": str(path),
            "valid": False,
            "reason": f"missing_expected_columns:{len(expected_cols)-len(present_expected)}",
            "rows": rows,
            "cols": cols,
            "expected_col_coverage": coverage,
            "missing_sample": missing,
        }

    return {
        "path": str(path),
        "valid": True,
        "reason": "valid",
        "rows": rows,
        "cols": cols,
        "normalized_cols": len(df_norm.columns),
        "expected_col_coverage": coverage,
        "score": _score_candidate_path(path),
        "df": df_norm,
    }

# ---------------------------------------------------------------------
# 3. Collect candidate parquet paths
# ---------------------------------------------------------------------
candidate_paths = set()

# 3a. Known likely active 12.c.R output names.
known_names = [
    "IOT_FINAL_CONTINUOUS_VALUES_TEST.parquet",
    "IOT_CONTINUOUS_FINAL_VALUES_TEST.parquet",
    "IOT_CONTINUOUS_VALUES_FINAL_TEST.parquet",
    "IOT_CONTINUOUS_VALUES_TEST.parquet",
    "IOT_FINAL_CONTINUOUS_TEST.parquet",
    "IOT_CONTINUOUS_FINAL_TEST.parquet",
    "IOT_VALUES_FINAL_TEST.parquet",
    "IOT_SELECTED_CONTINUOUS_VALUES_TEST.parquet",
    "IOT_SELECTED_CONTINUOUS_TEST.parquet",
    "IOT_SELECTED_VALUES_TEST.parquet",
    "IOT_SYN_CONTINUOUS_VALUES_TEST.parquet",
]
for name in known_names:
    candidate_paths.add(OUT_SYN_P / name)

# 3b. All non-mask/non-binary parquets under OUT_SYN.
for p in OUT_SYN_P.rglob("*.parquet"):
    candidate_paths.add(p)

# 3c. Parquet paths embedded in JSON contracts/manifests.
json_roots = [REPORT_DIR_P, CONTRACT_DIR_P, ARTIFACT_DIR_P]
for root in json_roots:
    if not root.exists():
        continue
    for jp in root.rglob("*.json"):
        name = jp.name.lower()
        if not any(tok in name for tok in ["12c", "continuous", "value", "manifest", "contract"]):
            continue
        try:
            obj = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        for s in _json_paths(obj):
            pp = Path(s)
            if not pp.is_absolute():
                pp = (OUTDIR_P / pp).resolve()
            candidate_paths.add(pp)

# ---------------------------------------------------------------------
# 4. Validate candidates
# ---------------------------------------------------------------------
valid_candidates = []
invalid_records = []

for p in sorted(candidate_paths, key=lambda x: str(x)):
    rec = _try_load_and_validate(p)
    if rec is None:
        continue
    if rec.get("valid"):
        valid_candidates.append(rec)
    else:
        invalid_records.append({k: v for k, v in rec.items() if k != "df"})

# ---------------------------------------------------------------------
# 5. If no file candidate found, try safe in-memory synthetic globals
# ---------------------------------------------------------------------
if not valid_candidates:
    _cR7_log("No valid parquet source found; trying safe in-memory synthetic globals.")
    forbidden_names = {"df_te", "DF_TE", "test_df", "DF_TEST", "real_test", "df_real_te"}
    for name, obj in list(globals().items()):
        if name in forbidden_names:
            continue
        if not isinstance(obj, pd.DataFrame):
            continue
        lname = name.lower()
        if "te" in lname and "synthetic" not in lname and "syn" not in lname and "12c" not in lname:
            continue
        if not any(tok in lname for tok in ["12c", "continuous", "cont", "iot", "final", "synthetic", "syn"]):
            continue
        if any(tok in lname for tok in ["mask", "binary", "protocol", "driver"]):
            continue

        rows, cols = obj.shape
        if rows != N_TE:
            continue
        obj_cols = list(map(str, obj.columns))
        present = [c for c in expected_cols if c in obj_cols]
        if len(present) == len(expected_cols):
            df_norm = obj[present].copy()
            valid_candidates.append({
                "path": f"global::{name}",
                "valid": True,
                "reason": "valid_global_dataframe",
                "rows": rows,
                "cols": cols,
                "normalized_cols": len(df_norm.columns),
                "expected_col_coverage": 1.0,
                "score": 1,
                "df": df_norm,
            })

if not valid_candidates:
    report_path = REPORT_DIR_P / "cell12cR_to_cell12g_continuous_handoff_failed_candidates.json"
    report_path.write_text(
        json.dumps({
            "cell": "12.c.R.7",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "expected_cols_n": len(expected_cols),
            "N_TE": N_TE,
            "invalid_records_sample": invalid_records[:50],
            "searched_paths_n": len(candidate_paths),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    raise RuntimeError(
        "[12.c.R.7] Could not locate a valid active 12.c.R continuous TEST matrix. "
        f"Diagnostic report written to: {report_path}"
    )

# Pick the best candidate.
valid_candidates_sorted = sorted(valid_candidates, key=lambda r: (r["score"], r["expected_col_coverage"]), reverse=True)
best = valid_candidates_sorted[0]
source_desc = best["path"]
continuous_df = best["df"]

# ---------------------------------------------------------------------
# 6. Write legacy alias expected by 12.g
# ---------------------------------------------------------------------
if continuous_df.shape != (N_TE, len(expected_cols)):
    raise RuntimeError(
        f"[12.c.R.7] Internal shape check failed: {continuous_df.shape} "
        f"!= ({N_TE}, {len(expected_cols)})"
    )

# Ensure numeric/finiteness sanity without changing values.
finite_rate_by_col = {}
nonfinite_total = 0
for c in continuous_df.columns:
    arr = pd.to_numeric(continuous_df[c], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(arr)
    finite_rate_by_col[c] = float(finite.mean())
    nonfinite_total += int((~finite).sum())

if nonfinite_total > 0:
    raise RuntimeError(
        f"[12.c.R.7] Refusing to hand off continuous matrix with nonfinite values: {nonfinite_total}"
    )

# Write alias. This is a compatibility write, not value mutation.
continuous_df.to_parquet(target_legacy_path, index=False)

# Optional common alias for downstream reviewers.
alias_extra = OUT_SYN_P / "IOT_FINAL_CONTINUOUS_VALUES_TEST.parquet"
if alias_extra != target_legacy_path:
    continuous_df.to_parquet(alias_extra, index=False)

# ---------------------------------------------------------------------
# 7. Publish compatibility globals
# ---------------------------------------------------------------------
CELL12C_FINAL_VALUES_TEST_DF = continuous_df
CELL12C_FINAL_VALUES_TEST_PATH = str(target_legacy_path)
CELL12C_ACTIVE_BRANCH = "12.c.R"

# Optional aliases for downstream cells.
IOT_FINAL_VALUES_TEST_PATH = str(target_legacy_path)
IOT_FINAL_VALUES_TEST_DF = continuous_df

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}

CFG["active_continuous_branch"] = "12.c.R"
CFG["cell12g_continuous_handoff_source"] = "12.c.R.7"
CFG["cell12g_continuous_values_path"] = str(target_legacy_path)

# ---------------------------------------------------------------------
# 8. Write report/contract
# ---------------------------------------------------------------------
handoff = {
    "cell": "12.c.R.7",
    "role": "continuous_12cR_to_cell12g_legacy_values_handoff",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "active_continuous_branch": "12.c.R",
    "source": source_desc,
    "target_legacy_path": str(target_legacy_path),
    "extra_alias_path": str(alias_extra),
    "shape": list(continuous_df.shape),
    "expected_cols_n": int(len(expected_cols)),
    "N_TE": int(N_TE),
    "nonfinite_total": int(nonfinite_total),
    "source_score": best["score"],
    "valid_candidate_count": len(valid_candidates),
    "policy": {
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "TEST_real_values_read": False,
        "legacy_alias_write_only": True,
        "refuses_df_te_as_source": True,
    },
    "files": {
        "target_sha256": _sha256_file(target_legacy_path),
        "extra_alias_sha256": _sha256_file(alias_extra),
    },
}

report_path = REPORT_DIR_P / "cell12cR_to_cell12g_continuous_handoff_report.json"
contract_path = CONTRACT_DIR_P / "cell12cR_to_cell12g_continuous_handoff_contract.json"

for p in [report_path, contract_path]:
    p.write_text(json.dumps(handoff, indent=2, sort_keys=True), encoding="utf-8")

CELL12C_TO_CELL12G_HANDOFF = handoff

_cR7_log(f"Source: {source_desc}")
_cR7_log(f"Wrote legacy path: {target_legacy_path}")
_cR7_log(f"Shape: {continuous_df.shape}")
_cR7_log(f"Contract: {contract_path}")
_cR7_log("PASS: Cell 12.g continuous values handoff is ready.")