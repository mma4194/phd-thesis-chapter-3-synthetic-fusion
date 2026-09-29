# ==========================================================
# ALIGNMENT GUARD — split/regime/TOD/conditioning alignment — v3-THESIS
# Decision: UPDATE + KEEP
#
# Purpose:
# - Verify that df_* rows, regime arrays, TOD arrays, and conditioning arrays
#   are semantically aligned after Cells 5–8.4.
# - Avoid relying on pandas indexes, because reg/tod are usually NumPy arrays.
# - Recompute TOD and regimes from canonical time/spec to prove semantic
#   alignment, not just length equality.
#
# Scientific contract:
# - This is a QA guard only.
# - It does not fit, select, calibrate, repair, generate, or overwrite artifacts.
# - TEST split is read only for alignment QA.
# - No TEST statistics are used for model fitting, threshold selection,
#   candidate selection, or repair.
# ==========================================================

log("--- START: ALIGNMENT GUARD v3-THESIS — split/regime/TOD/conditioning alignment ---")

import os
import json
import time
import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
need = [
    "CFG", "log",
    "df_tr", "df_va", "df_te",
    "reg_tr", "reg_va", "reg_te",
    "tod_tr", "tod_va", "tod_te",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[ALIGNMENT_GUARD] Missing required globals: {missing}")

# ----------------------------------------------------------
# 0b) Clean-run leakage guard inherited from Cell 1
# ----------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(
            f"[ALIGNMENT_GUARD] Clean alignment guard forbids CFG['{_flag}']=True."
        )

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[ALIGNMENT_GUARD] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
REP_DIR = os.path.join(OUTDIR, "reports")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

REGIME_SPEC_PATH = os.path.join(ARTDIR, "regime_spec.json")

if not os.path.exists(REGIME_SPEC_PATH):
    raise RuntimeError(f"[ALIGNMENT_GUARD] Missing regime_spec.json: {REGIME_SPEC_PATH}")

with open(REGIME_SPEC_PATH, "r", encoding="utf-8") as f:
    regime_spec = json.load(f)

if not isinstance(regime_spec, dict):
    raise RuntimeError(f"[ALIGNMENT_GUARD] regime_spec.json must contain a dict: {REGIME_SPEC_PATH}")

# Prefer the canonical time column materialized by Cell 2.
CANON_TIME_COL = str(CFG.get("canonical_time_col", "sec_epoch_s__canon"))
RAW_TIME_COL = str(CFG.get("raw_time_col", "sec"))

if all(CANON_TIME_COL in df.columns for df in [df_tr, df_va, df_te]):
    SEC_COL_USED = CANON_TIME_COL
    SEC_COL_SOURCE = "CFG['canonical_time_col']"
elif "SEC_COL" in globals() and all(str(globals()["SEC_COL"]) in df.columns for df in [df_tr, df_va, df_te]):
    SEC_COL_USED = str(globals()["SEC_COL"])
    SEC_COL_SOURCE = "global_SEC_COL"
elif "time_col" in globals() and all(str(globals()["time_col"]) in df.columns for df in [df_tr, df_va, df_te]):
    SEC_COL_USED = str(globals()["time_col"])
    SEC_COL_SOURCE = "global_time_col"
elif all(RAW_TIME_COL in df.columns for df in [df_tr, df_va, df_te]):
    # This fallback is allowed for debugging, but the canonical notebook should
    # normally have sec_epoch_s__canon by this point.
    SEC_COL_USED = RAW_TIME_COL
    SEC_COL_SOURCE = "CFG['raw_time_col']_fallback"
    log(
        "[ALIGNMENT_GUARD][WARN] canonical_time_col is absent; "
        f"falling back to raw_time_col={RAW_TIME_COL!r}. "
        "For the final STUDY-THESIS run, Cell 2 should materialize the canonical time column."
    )
else:
    raise RuntimeError(
        "[ALIGNMENT_GUARD] No usable time column found in all splits. "
        f"Tried canonical={CANON_TIME_COL!r}, global SEC_COL/time_col, raw={RAW_TIME_COL!r}."
    )

K_REG = int(globals().get("K_reg", regime_spec.get("K_reg_effective", -1)))

if K_REG <= 0:
    raise RuntimeError(f"[ALIGNMENT_GUARD] Invalid K_reg: {K_REG}")

edges_sod = np.asarray(regime_spec.get("edges_sec_of_day", []), dtype=np.float64)
if edges_sod.ndim != 1 or edges_sod.size < 2:
    raise RuntimeError(f"[ALIGNMENT_GUARD] Invalid regime edges in {REGIME_SPEC_PATH}: {edges_sod}")

if not np.isfinite(edges_sod).all():
    raise RuntimeError("[ALIGNMENT_GUARD] Regime edges contain non-finite values.")

if np.any(np.diff(edges_sod) <= 0):
    raise RuntimeError(f"[ALIGNMENT_GUARD] Regime edges must be strictly increasing: {edges_sod}")

if int(edges_sod.size - 1) != int(K_REG):
    raise RuntimeError(
        f"[ALIGNMENT_GUARD] Regime edge count mismatch: edges imply K={edges_sod.size - 1}, "
        f"K_reg={K_REG}"
    )

# ----------------------------------------------------------
# 1) Helpers
# ----------------------------------------------------------
def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _as_1d_reg(x, name: str) -> np.ndarray:
    arr = np.asarray(x)

    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if arr.size == 0:
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} is empty.")

    try:
        arr_f = arr.astype(np.float64)
    except Exception as e:
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} cannot be converted to numeric regime ids.") from e

    if not np.isfinite(arr_f).all():
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} contains non-finite values.")

    arr_i = np.round(arr_f).astype(np.int64)

    if not np.allclose(arr_f, arr_i.astype(np.float64), atol=1e-6, rtol=0.0):
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} contains non-integer-like regime ids.")

    if arr_i.min() < 0:
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} contains negative regime id: min={arr_i.min()}")

    if arr_i.max() >= K_REG:
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} exceeds K_reg={K_REG}: max={arr_i.max()}")

    return arr_i


def _as_tod(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} must have shape (N,2), got {arr.shape}")

    if arr.shape[0] == 0:
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} is empty.")

    if not np.isfinite(arr).all():
        raise RuntimeError(f"[ALIGNMENT_GUARD] {name} contains non-finite values.")

    # sin^2 + cos^2 should be very close to 1, allowing float32 tolerance.
    norm = np.sum(arr.astype(np.float64) ** 2, axis=1)
    max_dev = float(np.max(np.abs(norm - 1.0)))
    if max_dev > 5e-4:
        raise RuntimeError(
            f"[ALIGNMENT_GUARD] {name} does not look like cyclic sin/cos TOD. "
            f"max |norm-1|={max_dev:.6g}"
        )

    return arr


def _sec_array(df_part: pd.DataFrame, split: str) -> np.ndarray:
    if SEC_COL_USED not in df_part.columns:
        raise RuntimeError(f"[ALIGNMENT_GUARD] {split}: missing SEC_COL_USED={SEC_COL_USED}")

    sec = pd.to_numeric(df_part[SEC_COL_USED], errors="coerce").to_numpy(dtype=np.float64, copy=False)

    if not np.isfinite(sec).all():
        bad = int(np.sum(~np.isfinite(sec)))
        raise RuntimeError(f"[ALIGNMENT_GUARD] {split}: non-finite canonical seconds: {bad}")

    sec_i = np.round(sec).astype(np.int64)
    if not np.allclose(sec, sec_i.astype(np.float64), atol=1e-6, rtol=0.0):
        raise RuntimeError(f"[ALIGNMENT_GUARD] {split}: canonical seconds are not integer-like.")

    if len(sec_i) > 1:
        diffs = np.diff(sec_i)
        if np.any(diffs <= 0):
            first_bad = int(np.where(diffs <= 0)[0][0])
            raise RuntimeError(
                f"[ALIGNMENT_GUARD] {split}: canonical seconds are not strictly increasing. "
                f"first_bad_idx={first_bad}"
            )

        step_vals, step_counts = np.unique(diffs, return_counts=True)
        step_map = {int(k): int(v) for k, v in zip(step_vals, step_counts)}
        if set(step_map.keys()) != {1}:
            raise RuntimeError(
                f"[ALIGNMENT_GUARD] {split}: expected strict 1 Hz steps; got step_counts={step_map}"
            )

    return sec_i


def _tod_from_sec(sec: np.ndarray) -> np.ndarray:
    day = 86400.0
    phase = (sec.astype(np.float64) % day) / day * (2.0 * np.pi)
    return np.column_stack([np.sin(phase), np.cos(phase)]).astype(np.float32)


def _reg_from_sec(sec: np.ndarray) -> np.ndarray:
    day = 86400.0
    sod = np.mod(sec.astype(np.float64), day)
    sod[sod < 0.0] += day

    r = np.searchsorted(edges_sod[1:], sod, side="right").astype(np.int64)
    return np.clip(r, 0, K_REG - 1)


def _check_cond(split: str, n_expected: int) -> dict:
    gname = {
        "TRAIN": "cond_tr",
        "VAL": "cond_va",
        "TEST": "cond_te",
    }[split]

    if gname not in globals():
        log(f"[ALIGNMENT_GUARD] {split}: {gname} not present; skipping conditioning check.")
        return {
            "conditioning_present": False,
            "conditioning_global": gname,
            "conditioning_shape": None,
            "conditioning_finite": None,
        }

    arr = np.asarray(globals()[gname])

    if arr.ndim != 2:
        raise RuntimeError(f"[ALIGNMENT_GUARD] {split}: {gname} must be 2D, got {arr.shape}")

    if arr.shape[0] != n_expected:
        raise RuntimeError(
            f"[ALIGNMENT_GUARD] {split}: {gname} row mismatch: "
            f"got={arr.shape[0]}, expected={n_expected}"
        )

    if not np.isfinite(arr).all():
        raise RuntimeError(f"[ALIGNMENT_GUARD] {split}: {gname} contains non-finite values.")

    return {
        "conditioning_present": True,
        "conditioning_global": gname,
        "conditioning_shape": [int(arr.shape[0]), int(arr.shape[1])],
        "conditioning_finite": True,
    }


# ----------------------------------------------------------
# 2) Split alignment checks
# ----------------------------------------------------------
splits = [
    ("TRAIN", df_tr, reg_tr, tod_tr),
    ("VAL", df_va, reg_va, tod_va),
    ("TEST", df_te, reg_te, tod_te),
]

split_rows = []
split_ranges = {}

for split, df_part, reg_part, tod_part in splits:
    n_df = int(len(df_part))
    reg = _as_1d_reg(reg_part, f"reg_{split.lower()}")
    tod = _as_tod(tod_part, f"tod_{split.lower()}")
    sec = _sec_array(df_part, split)

    if not (n_df == reg.shape[0] == tod.shape[0] == sec.shape[0]):
        raise RuntimeError(
            f"[ALIGNMENT_GUARD] {split} length mismatch: "
            f"df={n_df}, reg={reg.shape[0]}, tod={tod.shape[0]}, "
            f"sec={sec.shape[0]}, tod_shape={tod.shape}"
        )

    tod_expected = _tod_from_sec(sec)
    tod_max_abs = float(np.max(np.abs(tod.astype(np.float32) - tod_expected.astype(np.float32))))

    if tod_max_abs > 2e-5:
        raise RuntimeError(
            f"[ALIGNMENT_GUARD] {split}: stored TOD does not match recomputed TOD from {SEC_COL_USED}. "
            f"max_abs_diff={tod_max_abs:.8f}"
        )

    reg_expected = _reg_from_sec(sec)
    reg_mismatch = int(np.sum(reg != reg_expected))

    if reg_mismatch != 0:
        mismatch_rate = float(reg_mismatch / max(1, len(reg)))
        first_idx = int(np.where(reg != reg_expected)[0][0])
        raise RuntimeError(
            f"[ALIGNMENT_GUARD] {split}: stored regimes do not match regime_spec recomputation. "
            f"mismatch={reg_mismatch} rate={mismatch_rate:.8f} first_idx={first_idx} "
            f"stored={int(reg[first_idx])} expected={int(reg_expected[first_idx])}"
        )

    cond_audit = _check_cond(split, n_df)

    reg_counts = np.bincount(reg, minlength=K_REG).astype(int).tolist()

    row = {
        "split": split,
        "n": n_df,
        "sec_first": int(sec[0]),
        "sec_last": int(sec[-1]),
        "duration_seconds_inclusive": int(sec[-1] - sec[0] + 1),
        "duration_days_inclusive": float((sec[-1] - sec[0] + 1) / 86400.0),
        "reg_counts": reg_counts,
        "tod_max_abs_diff": float(tod_max_abs),
        **cond_audit,
    }

    split_rows.append(row)

    split_ranges[split] = {
        "n": n_df,
        "sec_first": int(sec[0]),
        "sec_last": int(sec[-1]),
        "duration_seconds_inclusive": int(sec[-1] - sec[0] + 1),
        "duration_days_inclusive": float((sec[-1] - sec[0] + 1) / 86400.0),
        "reg_counts": reg_counts,
        "tod_max_abs_diff": float(tod_max_abs),
        "conditioning": cond_audit,
    }

    log(
        f"[ALIGNMENT_GUARD] {split}: n={n_df:,} | sec=[{int(sec[0])}, {int(sec[-1])}] | "
        f"days={(sec[-1] - sec[0] + 1) / 86400.0:.6f} | "
        f"reg_counts={reg_counts} | tod_max_abs_diff={tod_max_abs:.3e}"
    )

# Split boundary checks.
if not (
    split_ranges["TRAIN"]["sec_last"] < split_ranges["VAL"]["sec_first"]
    and split_ranges["VAL"]["sec_last"] < split_ranges["TEST"]["sec_first"]
):
    raise RuntimeError(
        "[ALIGNMENT_GUARD] Split time ranges overlap or are not strictly ordered: "
        f"{split_ranges}"
    )

if (
    split_ranges["TRAIN"]["sec_last"] + 1 != split_ranges["VAL"]["sec_first"]
    or split_ranges["VAL"]["sec_last"] + 1 != split_ranges["TEST"]["sec_first"]
):
    raise RuntimeError(
        "[ALIGNMENT_GUARD] Split time ranges are strictly ordered but not contiguous 1 Hz boundaries: "
        f"{split_ranges}"
    )

# ----------------------------------------------------------
# 3) Persist audit
# ----------------------------------------------------------
ALIGNMENT_GUARD_AUDIT = {
    "version": "alignment_guard_v3_THESIS",
    "component": "split_regime_tod_conditioning_alignment_guard",
    "component_type": "QA_guard_no_modeling",
    "SEC_COL_USED": SEC_COL_USED,
    "SEC_COL_SOURCE": SEC_COL_SOURCE,
    "canonical_time_col": CANON_TIME_COL,
    "raw_time_col": RAW_TIME_COL,
    "K_reg": int(K_REG),
    "regime_spec_path": REGIME_SPEC_PATH,
    "regime_edges_sec_of_day": [float(x) for x in edges_sod.tolist()],
    "split_ranges": split_ranges,
    "selection_split_policy": str(CFG.get("selection_split_policy", "")),
    "leakage_status": {
        "reads_train_split": True,
        "reads_val_split": True,
        "reads_test_split": True,
        "uses_test_for_alignment_QA_only": True,
        "uses_test_for_fitting": False,
        "uses_test_for_threshold_selection": False,
        "uses_test_for_candidate_selection": False,
        "uses_test_for_repair": False,
        "mutates_synthetic_artifact": False,
        "overwrites_final_artifact": False,
    },
    "paper_claim_status": (
        "Supports structural alignment validity only. Does not support distributional, "
        "temporal, coupling, utility, or release-quality claims by itself."
    ),
    "ts_unix": float(time.time()),
}

alignment_audit_json = os.path.join(CONTRACT_DIR, "alignment_guard_v3_THESIS.json")
alignment_audit_csv = os.path.join(REP_DIR, "alignment_guard_v3_split_summary.csv")

_write_json(alignment_audit_json, ALIGNMENT_GUARD_AUDIT)
pd.DataFrame(split_rows).to_csv(alignment_audit_csv, index=False)

globals()["ALIGNMENT_GUARD_AUDIT"] = ALIGNMENT_GUARD_AUDIT
globals()["ALIGNMENT_GUARD_AUDIT_PATH"] = alignment_audit_json

if "RUN_META" in globals():
    RUN_META.setdefault("qa_guards", {})
    RUN_META["qa_guards"]["alignment_guard_v3_THESIS"] = ALIGNMENT_GUARD_AUDIT

log(f"[ALIGNMENT_GUARD] Saved audit JSON: {alignment_audit_json}")
log(f"[ALIGNMENT_GUARD] Saved split summary CSV: {alignment_audit_csv}")
log("[ALIGNMENT_GUARD] PASS: split/regime/TOD/conditioning are semantically aligned with canonical time and regime_spec.")
log("--- END: ALIGNMENT GUARD v3-THESIS ---")