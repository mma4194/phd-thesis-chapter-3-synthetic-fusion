# %% CELL 12.c.4R — TRAIN/VAL-cache continuous TEST-length materialization
# Purpose:
#   Materialize TEST-length continuous values from the locked 12.c.3R selector.
#
# Safety:
#   - Uses TEST length/schema only.
#   - Does not read TEST target values.
#   - Does not use old contaminated 12.c.4 outputs.
#   - Uses clean 12.c.2 cached candidate arrays via cache_fingerprint_hash.

import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = None


def _c4r_log(msg):
    print(f"[Cell12.c.4R] {msg}")


def _c4r_path(x):
    if x is None:
        return None
    try:
        return Path(str(x)).expanduser().resolve()
    except Exception:
        return None


OUTDIR_P = _c4r_path(globals().get("OUTDIR", None)) or _c4r_path(os.environ.get("CPS_OUTDIR", ""))
OUT_SYN_P = _c4r_path(globals().get("OUT_SYN", None)) or OUTDIR_P / "synthetic"
REPORT_DIR_P = _c4r_path(globals().get("REPORT_DIR", None)) or OUTDIR_P / "reports"
CONTRACT_DIR_P = _c4r_path(globals().get("CONTRACT_DIR", None)) or OUTDIR_P / "artifacts" / "contracts"
ARTIFACT_DIR_P = _c4r_path(globals().get("ARTDIR", None)) or _c4r_path(globals().get("ARTIFACT_DIR", None)) or OUTDIR_P / "artifacts"

for p in [OUT_SYN_P, REPORT_DIR_P, CONTRACT_DIR_P, ARTIFACT_DIR_P]:
    p.mkdir(parents=True, exist_ok=True)

CFG_LOCAL = globals().get("CFG", {})
if not isinstance(CFG_LOCAL, dict):
    CFG_LOCAL = {}
CACHE_DIR_P = _c4r_path(CFG_LOCAL.get("cell12c2_cache_dir", None)) or ARTIFACT_DIR_P / "cell12c2_candidate_cache_v8_14"

selection_path = REPORT_DIR_P / "cell12c3R_locked_iot_value_selection.csv"
if not selection_path.exists():
    selection_path = REPORT_DIR_P / "cell12c3_locked_iot_value_selection.csv"
if not selection_path.exists():
    raise RuntimeError("[Cell12.c.4R] Missing locked 12.c.3R selection. Run Cell 12.c.3R first.")

selection = pd.read_csv(selection_path)

for req in ["col", "selected_generator", "selected_cache_fingerprint_hash"]:
    if req not in selection.columns:
        raise RuntimeError(f"[Cell12.c.4R] Selection missing required column {req}.")

# Resolve df_te length without reading TEST target values.
if "df_te" in globals() and isinstance(globals()["df_te"], pd.DataFrame):
    N_TE = int(len(globals()["df_te"]))
elif "DF_TE" in globals() and isinstance(globals()["DF_TE"], pd.DataFrame):
    N_TE = int(len(globals()["DF_TE"]))
else:
    raise RuntimeError("[Cell12.c.4R] Need df_te/DF_TE for TEST length only.")

if "df_tr" in globals() and isinstance(globals()["df_tr"], pd.DataFrame):
    DF_TR_LOCAL = globals()["df_tr"]
elif "DF_TR" in globals() and isinstance(globals()["DF_TR"], pd.DataFrame):
    DF_TR_LOCAL = globals()["DF_TR"]
else:
    DF_TR_LOCAL = None

if "df_va" in globals() and isinstance(globals()["df_va"], pd.DataFrame):
    DF_VA_LOCAL = globals()["df_va"]
elif "DF_VA" in globals() and isinstance(globals()["DF_VA"], pd.DataFrame):
    DF_VA_LOCAL = globals()["DF_VA"]
else:
    DF_VA_LOCAL = None


def _cache_npy_path_12c4r(hash_value):
    h = str(hash_value)
    if not h or h.lower() == "nan":
        return None
    p = CACHE_DIR_P / h[:2] / f"{h}.npy"
    if p.exists():
        return p
    hits = list(CACHE_DIR_P.rglob(f"{h}.npy")) if CACHE_DIR_P.exists() else []
    return hits[0] if hits else None


def _to_numeric_array_12c4r(x):
    arr = np.asarray(x)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype="float64")


def _fit_to_length_12c4r(arr, n, seed=0):
    arr = _to_numeric_array_12c4r(arr)
    if len(arr) == 0:
        return np.full(n, np.nan, dtype="float64")
    if len(arr) >= n:
        return arr[-n:].astype("float64")
    reps = int(math.ceil(n / len(arr)))
    tiled = np.tile(arr, reps)[:n].astype("float64")
    if n > 0 and len(arr) > 1:
        shift = int(seed % n)
        tiled = np.roll(tiled, shift)
    return tiled


def _seed_12c4r(s):
    import hashlib
    h = hashlib.sha256(str(s).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _train_fallback_12c4r(col, n):
    raise RuntimeError(f"Fresh continuous candidate missing or unreadable for {col}. No observed-data fallback is permitted.")


mat = {}
audit_rows = []

for i, row in selection.iterrows():
    col = str(row["col"])
    h = str(row["selected_cache_fingerprint_hash"])
    npy_path = _cache_npy_path_12c4r(h)
    seed = _seed_12c4r(h + "::" + col)

    if (i + 1) == 1 or (i + 1) % 25 == 0 or (i + 1) == len(selection):
        _c4r_log(f"Materializing {i+1}/{len(selection)} | col={col}")

    cache_loaded = False
    fallback_used = False
    fallback_reason = ""

    if npy_path is not None:
        try:
            arr0 = np.load(npy_path, allow_pickle=False)
            arr = _fit_to_length_12c4r(arr0, N_TE, seed=seed)
            cache_loaded = True
            materializer_used = "cell12c2_cached_candidate_tail_replay_to_TEST_length"
        except Exception as e:
            arr, materializer_used = _train_fallback_12c4r(col, N_TE)
            fallback_used = True
            fallback_reason = f"cache_load_failed:{e}"
    else:
        arr, materializer_used = _train_fallback_12c4r(col, N_TE)
        fallback_used = True
        fallback_reason = "cache_npy_missing"

    mat[col] = arr.astype("float64")

    finite = np.isfinite(arr)
    audit_rows.append({
        "col": col,
        "selected_generator": row.get("selected_generator"),
        "selected_candidate_id": row.get("selected_candidate_id"),
        "selected_cache_fingerprint_hash": h,
        "cache_npy_path": str(npy_path) if npy_path is not None else "",
        "cache_loaded": bool(cache_loaded),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "materializer_used_12c4R": materializer_used,
        "N_TE_length_only": int(N_TE),
        "synthetic_finite_rate": float(finite.mean()) if len(finite) else 0.0,
        "synthetic_mean": float(np.nanmean(arr)) if np.isfinite(arr).any() else None,
        "synthetic_std": float(np.nanstd(arr)) if np.isfinite(arr).any() else None,
        "TEST_real_values_used": False,
    })

values = pd.DataFrame(mat)
values = values[selection["col"].astype(str).tolist()]
mask = values.notna().astype("uint8")

selected_values_path = OUT_SYN_P / "IOT_SELECTED_CONTINUOUS_TEST.parquet"
selected_alias_path = OUT_SYN_P / "IOT_CONT_VALUE_TEST.parquet"
mask_path = OUT_SYN_P / "IOT_CONT_VALUE_MASK_TEST.parquet"

values.to_parquet(selected_values_path, index=False)
values.to_parquet(selected_alias_path, index=False)
mask.to_parquet(mask_path, index=False)

audit = pd.DataFrame(audit_rows)
audit_path_r = REPORT_DIR_P / "cell12c4R_trainval_cache_materialization_audit.csv"
audit_path_canon = REPORT_DIR_P / "cell12c4_locked_materializer_trainval_capability_audit.csv"
fidelity_path_canon = REPORT_DIR_P / "cell12c4_trainval_materializer_fidelity_audit.csv"

audit.to_csv(audit_path_r, index=False)
audit.to_csv(audit_path_canon, index=False)
audit.to_csv(fidelity_path_canon, index=False)

summary = {
    "cell": "12.c.4R",
    "role": "trainval_cache_continuous_test_length_materialization",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "targets": int(values.shape[1]),
    "N_TE_length_only": int(N_TE),
    "shape": list(values.shape),
    "cache_loaded_n": int(audit["cache_loaded"].sum()),
    "fallback_used_n": int(audit["fallback_used"].sum()),
    "mean_synthetic_finite_rate": float(audit["synthetic_finite_rate"].mean()),
    "TEST_real_values_used": False,
    "TEST_length_schema_only": True,
    "selection_done_here": False,
    "generator_fit_done_here": False,
    "materialization_done_here": True,
    "outputs": {
        "selected_values": str(selected_values_path),
        "selected_alias": str(selected_alias_path),
        "mask": str(mask_path),
        "audit": str(audit_path_r),
    },
}
summary_path = REPORT_DIR_P / "cell12c4R_trainval_cache_materialization_summary.json"
contract_path = CONTRACT_DIR_P / "cell12c4R_trainval_cache_materialization_contract.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
contract_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12C4R_VALUES = values
CELL12C4R_MASK = mask
CELL12C4R_SUMMARY = summary
CELL12C4R_VALUES_PATH = str(selected_values_path)

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}
CFG["cell12c4R_active"] = True
CFG["cell12c4R_test_values_used"] = False

_c4r_log(f"Materialization complete | shape={values.shape} | cache_loaded={summary['cache_loaded_n']} | fallback={summary['fallback_used_n']}")
_c4r_log(f"Contract: {contract_path}")

if display is not None:
    display(audit.head(20))