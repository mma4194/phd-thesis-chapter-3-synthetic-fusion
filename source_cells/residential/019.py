# ============================================================
# CELL 7 — Shared conditioning contract for protocol candidate generators
# v7 STUDY-THESIS / CPS-COUPLED / HYBRID-PORTFOLIO COMPATIBLE
#
# Purpose:
# - Build a shared conditioning contract from validated protocol + IoT contracts.
# - Freeze conditioning dimensions and column order.
# - Use TRAIN-only fitting for all transforms / imputers.
# - Preserve missingness semantics by avoiding blind zero-fill for IoT state.
# - Exclude IoT state columns from conditioning when TRAIN has no finite support,
#   while keeping them valid in the broader IoT taxonomy.
#
# Architectural role:
# - This cell is no longer DDPM-only conceptually.
# - It builds shared conditioning arrays usable by DDPM and other conditional
#   candidate generators.
# - DDPM still consumes cond_tr/cond_va/cond_te for compatibility.
#
# Critical policies:
# - TEST is audit/final-QA only.
# - No TEST statistics are used to fit imputers, caps, or scalers.
# - IoT event drivers may be zero-filled because event absence is semantically zero.
# - IoT state values must not be blindly zero-filled.
# - Countlike IoT state context uses TRAIN median imputation.
# - Event-driver context uses zero-fill.
# - The final count-context scaler is fitted directly in the main path; no stale
#   post-hoc hotfix is allowed.
#
# Conditioning families:
#   protocol_obs_context -> logical protocol obs pass-through
#   binary_state         -> TRAIN-supported binary columns only; TRAIN-mode imputation
#   continuous_state     -> TRAIN-supported residual continuous columns only;
#                           TRAIN-median imputation + StandardScaler
#   countlike_context    -> TRAIN-supported countlike state columns only;
#                           TRAIN-median imputation + cap + log1p
#   event_driver_context -> zero-fill + cap + log1p
#   count_context        -> concat(countlike_context, event_driver_context)
#                           + one StandardScaler fit on TRAIN only
#
# Contract note:
# - IOT_COUNTLIKE_COLS are subsets of IOT_CONT_COLS.
# - CONT_STATE_COLS_ALL = IOT_CONT_COLS \ IOT_COUNTLIKE_COLS, preserving Cell 6.5 order.
# ============================================================

log("--- START: Cell 7 — Shared conditioning contract (v7 CPS-coupled hybrid portfolio) ---")

import os
import json
import time
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

need = [
    "df_tr", "df_va", "df_te", "CFG", "log",
    "IOT_BIN_COLS", "IOT_CONT_COLS", "IOT_DRIVER_COLS", "IOT_COUNTLIKE_COLS",
    "DDPM_OBS_TR", "DDPM_OBS_VA", "DDPM_OBS_TE",
    "DDPM_TIERS_USED",
    "PROTO_COLS_CONTRACT",
    "PROTOCOL_GENERATOR_ROLE_CONTRACT",
    "IOT_CPS_COUPLING_POLICY",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(
        f"[Cell7] Missing prerequisites: {missing}. "
        "Run Cells 4.5, 5, 6, and 6.5 first."
    )

# ------------------------------------------------------------
# Clean-run leakage guard inherited from Cell 1
# ------------------------------------------------------------
for _flag in [
    "allow_test_values_for_fitting",
    "allow_test_values_for_threshold_selection",
    "allow_test_values_for_candidate_selection",
    "allow_test_values_for_repair",
    "allow_test_values_for_artifact_overwrite",
]:
    if bool(CFG.get(_flag, False)):
        raise RuntimeError(
            f"[Cell7] Clean conditioning contract forbids CFG['{_flag}']=True."
        )

if str(CFG.get("selection_split_policy", "")) != "TRAIN_fit_VAL_select_TEST_audit_only":
    raise RuntimeError(
        "[Cell7] Unexpected selection_split_policy. "
        f"Got {CFG.get('selection_split_policy')!r}."
    )

OUTDIR = str(CFG["outdir"])
ARTDIR = os.path.join(OUTDIR, "artifacts")
REP_DIR = os.path.join(OUTDIR, "reports")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

seed = int(CFG.get("seed", 1337))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _write_json(path, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _sha256_list(xs) -> str:
    h = hashlib.sha256()
    for x in xs:
        h.update((str(x) + "\n").encode("utf-8"))
    return h.hexdigest()


def _check_unique(name: str, cols: list):
    if len(cols) != len(set(cols)):
        seen, dupes = set(), []
        for c in cols:
            if c in seen:
                dupes.append(c)
            seen.add(c)
        raise RuntimeError(
            f"[Cell7] Duplicate columns in {name}: {sorted(set(dupes))[:20]}"
        )


def _check_disjoint(name_a: str, a: list, name_b: str, b: list):
    ov = sorted(set(a).intersection(set(b)))
    if ov:
        raise RuntimeError(f"[Cell7] {name_a} ∩ {name_b} not empty: {ov[:20]}")


def _check_subset(name_a: str, a: list, name_b: str, b: list):
    miss = sorted(set(a) - set(b))
    if miss:
        raise RuntimeError(f"[Cell7] {name_a} is not a subset of {name_b}: {miss[:20]}")


def _numeric_matrix_preserve_nan(df_part: pd.DataFrame, cols: list, *, block_name: str) -> np.ndarray:
    cols = list(cols)
    M = np.empty((len(df_part), len(cols)), dtype=np.float32)

    for j, c in enumerate(cols):
        if c not in df_part.columns:
            raise RuntimeError(f"[Cell7] Missing column in {block_name}: {c}")

        M[:, j] = pd.to_numeric(
            df_part[c],
            errors="coerce",
        ).to_numpy(dtype=np.float32, copy=False)

    if np.isinf(M).any():
        bad = int(np.isinf(M).sum())
        raise RuntimeError(f"[Cell7] {block_name} contains +/-inf values: {bad}")

    return M


def _finite_stats(M: np.ndarray) -> dict:
    M = np.asarray(M)
    finite = np.isfinite(M)

    if M.ndim != 2:
        raise RuntimeError(f"[Cell7] _finite_stats expected 2D matrix, got {M.shape}")

    return {
        "shape": [int(M.shape[0]), int(M.shape[1])],
        "finite_rate": float(finite.mean()) if M.size else 1.0,
        "nan_count": int(np.isnan(M).sum()),
        "inf_count": int(np.isinf(M).sum()),
        "cols_all_nan": int(np.sum(~finite.any(axis=0))) if M.shape[1] else 0,
    }


def _train_supported_cols(df_train: pd.DataFrame, cols: list, *, block_name: str) -> tuple:
    """
    Return (supported, unsupported) based only on TRAIN finite support.

    A column can be a valid IoT taxonomy column but invalid as a TRAIN-fitted
    conditioning feature if TRAIN has no finite support.
    """
    supported = []
    unsupported = []

    for c in cols:
        if c not in df_train.columns:
            raise RuntimeError(f"[Cell7] Missing TRAIN column in {block_name}: {c}")

        x = pd.to_numeric(
            df_train[c],
            errors="coerce",
        ).to_numpy(dtype=np.float32, copy=False)

        if np.isfinite(x).any():
            supported.append(c)
        else:
            unsupported.append(c)

    if unsupported:
        log(
            f"[Cell7] Excluding {len(unsupported)} {block_name} conditioning columns "
            f"with no finite TRAIN support. first20={unsupported[:20]}"
        )

    return supported, unsupported


def _support_rates_by_split(cols: list) -> dict:
    out = {}

    for c in cols:
        row = {}

        for split_name, df_part in [
            ("train", df_tr),
            ("val", df_va),
            ("test", df_te),
        ]:
            if c not in df_part.columns:
                row[f"{split_name}_finite_rate"] = None
                continue

            x = pd.to_numeric(
                df_part[c],
                errors="coerce",
            ).to_numpy(dtype=np.float32, copy=False)

            row[f"{split_name}_finite_rate"] = float(np.isfinite(x).mean()) if len(x) else 0.0

        out[c] = row

    return out


def _fit_train_median(
    M_tr: np.ndarray,
    cols: list,
    *,
    block_name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    if M_tr.shape[1] != len(cols):
        raise RuntimeError(f"[Cell7] {block_name} column count mismatch.")

    med = np.empty(M_tr.shape[1], dtype=np.float32)

    for j, c in enumerate(cols):
        x = M_tr[:, j]
        finite = np.isfinite(x)

        if not finite.any():
            raise RuntimeError(
                f"[Cell7] {block_name}:{c} has no finite TRAIN support. "
                "It should have been filtered before fitting."
            )

        vals = x[finite].astype(np.float64, copy=False)

        if nonnegative:
            vals = np.maximum(vals, 0.0)

        med[j] = np.float32(np.nanmedian(vals))

    med[~np.isfinite(med)] = 0.0

    if nonnegative:
        med = np.maximum(med, 0.0).astype(np.float32, copy=False)

    return med.astype(np.float32, copy=False)


def _impute_with_values(M: np.ndarray, fill_values: np.ndarray, *, block_name: str) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32).copy()
    fill_values = np.asarray(fill_values, dtype=np.float32).reshape(1, -1)

    if M.shape[1] != fill_values.shape[1]:
        raise RuntimeError(
            f"[Cell7] {block_name} imputation shape mismatch: "
            f"M={M.shape}, fill={fill_values.shape}"
        )

    bad = ~np.isfinite(M)
    if bad.any():
        M[bad] = np.take(fill_values.reshape(-1), np.where(bad)[1])

    if not np.isfinite(M).all():
        raise RuntimeError(f"[Cell7] {block_name} still contains non-finite values after imputation.")

    return M.astype(np.float32, copy=False)


def _validate_binary_finite_values(M: np.ndarray, cols: list, *, block_name: str) -> None:
    for j, c in enumerate(cols):
        x = M[:, j]
        finite = np.isfinite(x)

        if not finite.any():
            raise RuntimeError(
                f"[Cell7] {block_name}:{c} has no finite TRAIN support. "
                "It should have been filtered before binary validation."
            )

        u = np.unique(np.round(x[finite], 6))
        if not set(u.tolist()).issubset({0.0, 1.0}):
            raise RuntimeError(
                f"[Cell7] {block_name}:{c} has non-binary finite values. "
                f"unique_sample={u[:20].tolist()}"
            )


def _fit_binary_mode(M_tr: np.ndarray, cols: list, *, block_name: str) -> np.ndarray:
    _validate_binary_finite_values(M_tr, cols, block_name=block_name)

    fill = np.empty(M_tr.shape[1], dtype=np.float32)

    for j, c in enumerate(cols):
        x = M_tr[:, j]
        finite = np.isfinite(x)

        p = float(np.mean(x[finite] > 0.5))
        fill[j] = np.float32(1.0 if p >= 0.5 else 0.0)

    return fill.astype(np.float32, copy=False)


def _impute_binary(M: np.ndarray, fill_values: np.ndarray, cols: list, *, block_name: str) -> np.ndarray:
    for j, c in enumerate(cols):
        x = M[:, j]
        finite = np.isfinite(x)

        if finite.any():
            u = np.unique(np.round(x[finite], 6))
            if not set(u.tolist()).issubset({0.0, 1.0}):
                raise RuntimeError(
                    f"[Cell7] {block_name}:{c} has non-binary finite values. "
                    f"unique_sample={u[:20].tolist()}"
                )

    out = _impute_with_values(M, fill_values, block_name=block_name)
    out = (out > 0.5).astype(np.float32, copy=False)
    return out


def _zero_fill_nonnegative(M: np.ndarray, *, block_name: str) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32).copy()
    M[~np.isfinite(M)] = 0.0
    M = np.maximum(M, 0.0).astype(np.float32, copy=False)

    if not np.isfinite(M).all():
        raise RuntimeError(f"[Cell7] {block_name} still contains non-finite values after zero-fill.")

    return M


def _cap_log1p_nonneg(M: np.ndarray, caps_: np.ndarray, *, block_name: str) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    caps_ = np.asarray(caps_, dtype=np.float32).reshape(1, -1)

    if M.shape[1] != caps_.shape[1]:
        raise RuntimeError(
            f"[Cell7] {block_name} cap shape mismatch: M={M.shape}, caps={caps_.shape}"
        )

    M2 = np.minimum(np.maximum(M, 0.0), caps_).astype(np.float32, copy=False)
    out = np.log1p(M2).astype(np.float32, copy=False)

    if not np.isfinite(out).all():
        raise RuntimeError(f"[Cell7] {block_name} log1p output contains non-finite values.")

    return out


def _fit_caps_train(M_tr_nonneg: np.ndarray, cols: list, *, block_name: str) -> np.ndarray:
    if M_tr_nonneg.shape[1] != len(cols):
        raise RuntimeError(f"[Cell7] {block_name} cap input column mismatch.")

    if len(cols) == 0:
        return np.zeros(0, dtype=np.float32)

    cap_q = float(np.clip(float(CFG.get("cond_cap_q", 0.999)), 0.95, 0.99999))

    caps = np.quantile(M_tr_nonneg, cap_q, axis=0).astype(np.float32)
    caps = np.where(np.isfinite(caps), caps, 0.0)
    caps = np.maximum(caps, 0.0)

    active = (np.max(M_tr_nonneg, axis=0) > 0)
    caps = np.where(active & (caps < 1.0), 1.0, caps).astype(np.float32)

    if not np.isfinite(caps).all():
        raise RuntimeError(f"[Cell7] {block_name} caps contain non-finite values.")

    return caps.astype(np.float32, copy=False)


def _fit_standard_scaler_train(M_tr: np.ndarray, *, block_name: str) -> StandardScaler:
    if not np.isfinite(M_tr).all():
        raise RuntimeError(
            f"[Cell7] Cannot fit scaler for {block_name}: TRAIN matrix has non-finite values."
        )

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(M_tr)

    if not np.isfinite(scaler.mean_).all():
        raise RuntimeError(f"[Cell7] {block_name} scaler mean_ contains non-finite values.")

    if not np.isfinite(scaler.scale_).all():
        raise RuntimeError(f"[Cell7] {block_name} scaler scale_ contains non-finite values.")

    if np.any(np.asarray(scaler.scale_) <= 0):
        raise RuntimeError(f"[Cell7] {block_name} scaler scale_ contains non-positive values.")

    return scaler


def _transform_scaler(scaler: StandardScaler, M: np.ndarray, *, block_name: str) -> np.ndarray:
    out = scaler.transform(M).astype(np.float32, copy=False)

    if not np.isfinite(out).all():
        raise RuntimeError(f"[Cell7] {block_name} scaled output contains non-finite values.")

    return out


def _validate_obs_matrix(name: str, arr: np.ndarray, expected_n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise RuntimeError(f"[Cell7] {name} must be 2D, got {arr.shape}")

    if arr.shape[0] != expected_n:
        raise RuntimeError(f"[Cell7] {name} row mismatch: got={arr.shape[0]} expected={expected_n}")

    if not np.isfinite(arr).all():
        raise RuntimeError(f"[Cell7] {name} contains non-finite values.")

    vals = set(np.unique(np.round(arr, 6)).tolist())
    if not vals.issubset({0.0, 1.0}):
        raise RuntimeError(f"[Cell7] {name} must be binary 0/1. values={sorted(vals)[:20]}")

    return arr.astype(np.float32, copy=False)


def _artifact_scaler_dim_check(path: str, expected_dim: int, *, name: str):
    sc = joblib.load(path)
    mean = np.asarray(getattr(sc, "mean_", []), dtype=np.float64).reshape(-1)
    scale = np.asarray(getattr(sc, "scale_", []), dtype=np.float64).reshape(-1)

    if mean.size != expected_dim or scale.size != expected_dim:
        raise RuntimeError(
            f"[Cell7] Saved {name} dimension mismatch: "
            f"mean={mean.size} scale={scale.size} expected={expected_dim}"
        )

    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError(f"[Cell7] Saved {name} has invalid mean/scale values.")


# ------------------------------------------------------------
# 0) Conditioning obs block from protocol logical obs contract
# ------------------------------------------------------------
obs_tr = _validate_obs_matrix("DDPM_OBS_TR", globals()["DDPM_OBS_TR"], len(df_tr))
obs_va = _validate_obs_matrix("DDPM_OBS_VA", globals()["DDPM_OBS_VA"], len(df_va))
obs_te = _validate_obs_matrix("DDPM_OBS_TE", globals()["DDPM_OBS_TE"], len(df_te))

OBS_TIERS = list(globals().get("DDPM_TIERS_USED", []))
if not OBS_TIERS:
    raise RuntimeError("[Cell7] DDPM_TIERS_USED is missing or empty.")

if len(OBS_TIERS) != obs_tr.shape[1]:
    raise RuntimeError(
        f"[Cell7] DDPM_TIERS_USED length mismatch with DDPM_OBS: "
        f"tiers={OBS_TIERS}, obs_dim={obs_tr.shape[1]}"
    )

OBS_COLS = [f"protocol_obs__{t}" for t in OBS_TIERS]
obs_dim = int(obs_tr.shape[1])


# ------------------------------------------------------------
# 1) Build current conditioning contracts
# ------------------------------------------------------------
BIN_COLS_ALL = list(IOT_BIN_COLS)
RAW_CONT_COLS_ALL = list(IOT_CONT_COLS)
COUNTLIKE_COLS_ALL = list(IOT_COUNTLIKE_COLS)
IOT_DRV_COLS = list(IOT_DRIVER_COLS)

# Auxiliary Zigbee event/intensity drivers, if materialized by earlier cells.
ZB_DRIVER_COLS = [
    c for c in ["zb__app_event_intensity", "zb__app_event_flag"]
    if c in df_tr.columns
]

expected_zb_drv_cfg = int(CFG.get("expected_zb_driver_cols", len(ZB_DRIVER_COLS)))
if len(ZB_DRIVER_COLS) != expected_zb_drv_cfg:
    raise RuntimeError(
        f"[Cell7] ZB_DRIVER_COLS count mismatch: got={len(ZB_DRIVER_COLS)} "
        f"expected={expected_zb_drv_cfg}. cols={ZB_DRIVER_COLS}"
    )

for name, cols in [
    ("BIN_COLS_ALL", BIN_COLS_ALL),
    ("RAW_CONT_COLS_ALL", RAW_CONT_COLS_ALL),
    ("COUNTLIKE_COLS_ALL", COUNTLIKE_COLS_ALL),
    ("IOT_DRV_COLS", IOT_DRV_COLS),
    ("ZB_DRIVER_COLS", ZB_DRIVER_COLS),
]:
    _check_unique(name, cols)

for nm, _df in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    for block_name, cols in [
        ("BIN_COLS_ALL", BIN_COLS_ALL),
        ("RAW_CONT_COLS_ALL", RAW_CONT_COLS_ALL),
        ("COUNTLIKE_COLS_ALL", COUNTLIKE_COLS_ALL),
        ("IOT_DRV_COLS", IOT_DRV_COLS),
        ("ZB_DRIVER_COLS", ZB_DRIVER_COLS),
    ]:
        miss = [c for c in cols if c not in _df.columns]
        if miss:
            raise RuntimeError(f"[Cell7] Conditioning schema drift in {nm}:{block_name}: {miss[:20]}")

if len(BIN_COLS_ALL) == 0:
    raise RuntimeError("[Cell7] BIN_COLS_ALL is empty.")

if len(RAW_CONT_COLS_ALL) == 0:
    raise RuntimeError("[Cell7] IOT_CONT_COLS is empty.")

if len(COUNTLIKE_COLS_ALL) == 0:
    raise RuntimeError("[Cell7] COUNTLIKE_COLS_ALL is empty.")

if len(IOT_DRV_COLS) == 0:
    raise RuntimeError("[Cell7] IOT_DRV_COLS is empty.")

_check_subset("COUNTLIKE_COLS_ALL", COUNTLIKE_COLS_ALL, "RAW_CONT_COLS_ALL", RAW_CONT_COLS_ALL)

countlike_set_all = set(COUNTLIKE_COLS_ALL)

# Preserve Cell 6.5 / Cell 4.5 continuous order. Do not sort.
CONT_STATE_COLS_ALL = [c for c in RAW_CONT_COLS_ALL if c not in countlike_set_all]

# TRAIN-only conditioning eligibility filter.
BIN_COLS, BIN_NO_TRAIN_SUPPORT_COLS = _train_supported_cols(
    df_tr,
    BIN_COLS_ALL,
    block_name="binary",
)

CONT_STATE_COLS, CONT_STATE_NO_TRAIN_SUPPORT_COLS = _train_supported_cols(
    df_tr,
    CONT_STATE_COLS_ALL,
    block_name="continuous_state",
)

COUNTLIKE_COLS, COUNTLIKE_NO_TRAIN_SUPPORT_COLS = _train_supported_cols(
    df_tr,
    COUNTLIKE_COLS_ALL,
    block_name="countlike",
)

if len(BIN_COLS) == 0:
    raise RuntimeError("[Cell7] All binary conditioning columns have no finite TRAIN support.")

if len(CONT_STATE_COLS) == 0:
    raise RuntimeError("[Cell7] All continuous-state conditioning columns have no finite TRAIN support.")

if len(COUNTLIKE_COLS) == 0:
    raise RuntimeError("[Cell7] All countlike conditioning columns have no finite TRAIN support.")

COND_NO_TRAIN_SUPPORT_EXCLUDED_COLS = (
    list(BIN_NO_TRAIN_SUPPORT_COLS)
    + list(CONT_STATE_NO_TRAIN_SUPPORT_COLS)
    + list(COUNTLIKE_NO_TRAIN_SUPPORT_COLS)
)

COND_NO_TRAIN_SUPPORT_AUDIT = {
    "policy": (
        "Columns are valid IoT taxonomy columns but excluded from conditioning "
        "because TRAIN has no finite support. VAL/TEST support is audit-only and "
        "must not be used to fit conditioning transforms."
    ),
    "binary_no_train_support": list(BIN_NO_TRAIN_SUPPORT_COLS),
    "continuous_state_no_train_support": list(CONT_STATE_NO_TRAIN_SUPPORT_COLS),
    "countlike_no_train_support": list(COUNTLIKE_NO_TRAIN_SUPPORT_COLS),
    "n_total_no_train_support": int(len(COND_NO_TRAIN_SUPPORT_EXCLUDED_COLS)),
    "support_rates": _support_rates_by_split(COND_NO_TRAIN_SUPPORT_EXCLUDED_COLS),
}

# Strict disjointness after residualization + TRAIN-support filtering.
_check_disjoint("BIN_COLS", BIN_COLS, "CONT_STATE_COLS", CONT_STATE_COLS)
_check_disjoint("BIN_COLS", BIN_COLS, "COUNTLIKE_COLS", COUNTLIKE_COLS)
_check_disjoint("BIN_COLS", BIN_COLS, "IOT_DRV_COLS", IOT_DRV_COLS)
_check_disjoint("BIN_COLS", BIN_COLS, "ZB_DRIVER_COLS", ZB_DRIVER_COLS)
_check_disjoint("CONT_STATE_COLS", CONT_STATE_COLS, "COUNTLIKE_COLS", COUNTLIKE_COLS)
_check_disjoint("CONT_STATE_COLS", CONT_STATE_COLS, "IOT_DRV_COLS", IOT_DRV_COLS)
_check_disjoint("CONT_STATE_COLS", CONT_STATE_COLS, "ZB_DRIVER_COLS", ZB_DRIVER_COLS)
_check_disjoint("COUNTLIKE_COLS", COUNTLIKE_COLS, "IOT_DRV_COLS", IOT_DRV_COLS)
_check_disjoint("COUNTLIKE_COLS", COUNTLIKE_COLS, "ZB_DRIVER_COLS", ZB_DRIVER_COLS)
_check_disjoint("IOT_DRV_COLS", IOT_DRV_COLS, "ZB_DRIVER_COLS", ZB_DRIVER_COLS)

bin_dim = len(BIN_COLS)
cont_state_dim = len(CONT_STATE_COLS)
countlike_dim = len(COUNTLIKE_COLS)
iot_drv_dim = len(IOT_DRV_COLS)
zb_drv_dim = len(ZB_DRIVER_COLS)

COUNTLIKE_CTX_COLS = list(COUNTLIKE_COLS)
EVENT_CTX_COLS = list(IOT_DRV_COLS) + list(ZB_DRIVER_COLS)
COUNT_CTX_COLS = list(COUNTLIKE_CTX_COLS) + list(EVENT_CTX_COLS)

_check_unique("COUNTLIKE_CTX_COLS", COUNTLIKE_CTX_COLS)
_check_unique("EVENT_CTX_COLS", EVENT_CTX_COLS)
_check_unique("COUNT_CTX_COLS", COUNT_CTX_COLS)

countlike_ctx_dim = len(COUNTLIKE_CTX_COLS)
event_ctx_dim = len(EVENT_CTX_COLS)
count_ctx_dim = len(COUNT_CTX_COLS)

feat_total = bin_dim + cont_state_dim + count_ctx_dim
cond_dim = obs_dim + feat_total


# ------------------------------------------------------------
# 1b) Optional exact-count pins from CFG
# ------------------------------------------------------------
expected_bin = int(CFG.get("expected_cond_iot_bin_cols", bin_dim))
expected_cont = int(CFG.get("expected_cond_iot_cont_state_cols", cont_state_dim))
expected_countlike = int(CFG.get("expected_cond_iot_countlike_cols", countlike_dim))
expected_drv = int(
    CFG.get(
        "expected_iot_driver_cols",
        int(CFG.get("expected_events_entity_cols", 12))
        + int(CFG.get("expected_events_feat_cols", 14)),
    )
)
expected_zb_drv = int(CFG.get("expected_zb_driver_cols", zb_drv_dim))
expected_obs_dim = int(CFG.get("expected_ddpm_obs_dim", obs_dim))
expected_cond_dim = int(
    CFG.get(
        "expected_cond_dim_v7",
        CFG.get(
            "expected_cond_dim_v6",
            obs_dim + bin_dim + cont_state_dim + countlike_dim + iot_drv_dim + zb_drv_dim,
        ),
    )
)

if obs_dim != expected_obs_dim:
    raise RuntimeError(f"[Cell7] obs_dim mismatch: got={obs_dim} expected={expected_obs_dim}")

if bin_dim != expected_bin:
    raise RuntimeError(f"[Cell7] BIN_COLS count mismatch: got={bin_dim} expected={expected_bin}")

if cont_state_dim != expected_cont:
    raise RuntimeError(
        f"[Cell7] CONT_STATE_COLS count mismatch: got={cont_state_dim} expected={expected_cont}"
    )

if countlike_dim != expected_countlike:
    raise RuntimeError(
        f"[Cell7] COUNTLIKE_COLS count mismatch: got={countlike_dim} expected={expected_countlike}"
    )

if iot_drv_dim != expected_drv:
    raise RuntimeError(f"[Cell7] IOT_DRV_COLS count mismatch: got={iot_drv_dim} expected={expected_drv}")

if zb_drv_dim != expected_zb_drv:
    raise RuntimeError(f"[Cell7] ZB_DRIVER_COLS count mismatch: got={zb_drv_dim} expected={expected_zb_drv}")

if cond_dim != expected_cond_dim:
    raise RuntimeError(f"[Cell7] cond_dim mismatch: got={cond_dim} expected={expected_cond_dim}")

log(
    f"[Cell7] Contracts: obs={obs_dim} bin={bin_dim} cont_state={cont_state_dim} "
    f"countlike={countlike_dim} iot_drv={iot_drv_dim} zb_drv={zb_drv_dim} "
    f"=> count_ctx={count_ctx_dim} cond_dim={cond_dim}"
)

if COND_NO_TRAIN_SUPPORT_EXCLUDED_COLS:
    log(
        f"[Cell7] TRAIN-support conditioning exclusions: "
        f"binary={len(BIN_NO_TRAIN_SUPPORT_COLS)} | "
        f"cont_state={len(CONT_STATE_NO_TRAIN_SUPPORT_COLS)} | "
        f"countlike={len(COUNTLIKE_NO_TRAIN_SUPPORT_COLS)} | "
        f"total={len(COND_NO_TRAIN_SUPPORT_EXCLUDED_COLS)}"
    )


# ------------------------------------------------------------
# 2) Build raw matrices, preserving NaNs where semantically meaningful
# ------------------------------------------------------------
B_tr_raw = _numeric_matrix_preserve_nan(df_tr, BIN_COLS, block_name="binary/TRAIN")
B_va_raw = _numeric_matrix_preserve_nan(df_va, BIN_COLS, block_name="binary/VAL")
B_te_raw = _numeric_matrix_preserve_nan(df_te, BIN_COLS, block_name="binary/TEST")

S_tr_raw = _numeric_matrix_preserve_nan(df_tr, CONT_STATE_COLS, block_name="continuous_state/TRAIN")
S_va_raw = _numeric_matrix_preserve_nan(df_va, CONT_STATE_COLS, block_name="continuous_state/VAL")
S_te_raw = _numeric_matrix_preserve_nan(df_te, CONT_STATE_COLS, block_name="continuous_state/TEST")

Kc_tr_raw = _numeric_matrix_preserve_nan(df_tr, COUNTLIKE_CTX_COLS, block_name="countlike_context/TRAIN")
Kc_va_raw = _numeric_matrix_preserve_nan(df_va, COUNTLIKE_CTX_COLS, block_name="countlike_context/VAL")
Kc_te_raw = _numeric_matrix_preserve_nan(df_te, COUNTLIKE_CTX_COLS, block_name="countlike_context/TEST")

Ke_tr_raw = _numeric_matrix_preserve_nan(df_tr, EVENT_CTX_COLS, block_name="event_context/TRAIN")
Ke_va_raw = _numeric_matrix_preserve_nan(df_va, EVENT_CTX_COLS, block_name="event_context/VAL")
Ke_te_raw = _numeric_matrix_preserve_nan(df_te, EVENT_CTX_COLS, block_name="event_context/TEST")

raw_support_audit = {
    "binary_train": _finite_stats(B_tr_raw),
    "binary_val": _finite_stats(B_va_raw),
    "binary_test": _finite_stats(B_te_raw),
    "cont_state_train": _finite_stats(S_tr_raw),
    "cont_state_val": _finite_stats(S_va_raw),
    "cont_state_test": _finite_stats(S_te_raw),
    "countlike_train": _finite_stats(Kc_tr_raw),
    "countlike_val": _finite_stats(Kc_va_raw),
    "countlike_test": _finite_stats(Kc_te_raw),
    "event_context_train": _finite_stats(Ke_tr_raw),
    "event_context_val": _finite_stats(Ke_va_raw),
    "event_context_test": _finite_stats(Ke_te_raw),
}


# ------------------------------------------------------------
# 3) TRAIN-only transforms per family
# ------------------------------------------------------------
# 3a) Binary state: strict finite values + TRAIN-mode imputation.
bin_fill = _fit_binary_mode(B_tr_raw, BIN_COLS, block_name="binary/TRAIN")

B_tr = _impute_binary(B_tr_raw, bin_fill, BIN_COLS, block_name="binary/TRAIN")
B_va = _impute_binary(B_va_raw, bin_fill, BIN_COLS, block_name="binary/VAL")
B_te = _impute_binary(B_te_raw, bin_fill, BIN_COLS, block_name="binary/TEST")

# 3b) Continuous state: TRAIN-median imputation + StandardScaler.
state_fill = _fit_train_median(
    S_tr_raw,
    CONT_STATE_COLS,
    block_name="continuous_state/TRAIN",
    nonnegative=False,
)

S_tr_imp = _impute_with_values(S_tr_raw, state_fill, block_name="continuous_state/TRAIN")
S_va_imp = _impute_with_values(S_va_raw, state_fill, block_name="continuous_state/VAL")
S_te_imp = _impute_with_values(S_te_raw, state_fill, block_name="continuous_state/TEST")

state_scaler = _fit_standard_scaler_train(S_tr_imp, block_name="continuous_state")

S_tr_s = _transform_scaler(state_scaler, S_tr_imp, block_name="continuous_state/TRAIN")
S_va_s = _transform_scaler(state_scaler, S_va_imp, block_name="continuous_state/VAL")
S_te_s = _transform_scaler(state_scaler, S_te_imp, block_name="continuous_state/TEST")

# 3c) Countlike state context: TRAIN-median imputation, nonnegative, cap + log1p.
countlike_fill = _fit_train_median(
    Kc_tr_raw,
    COUNTLIKE_CTX_COLS,
    block_name="countlike_context/TRAIN",
    nonnegative=True,
)

Kc_tr_imp = _impute_with_values(Kc_tr_raw, countlike_fill, block_name="countlike_context/TRAIN")
Kc_va_imp = _impute_with_values(Kc_va_raw, countlike_fill, block_name="countlike_context/VAL")
Kc_te_imp = _impute_with_values(Kc_te_raw, countlike_fill, block_name="countlike_context/TEST")

Kc_tr_imp = np.maximum(Kc_tr_imp, 0.0).astype(np.float32, copy=False)
Kc_va_imp = np.maximum(Kc_va_imp, 0.0).astype(np.float32, copy=False)
Kc_te_imp = np.maximum(Kc_te_imp, 0.0).astype(np.float32, copy=False)

countlike_caps = _fit_caps_train(
    Kc_tr_imp,
    COUNTLIKE_CTX_COLS,
    block_name="countlike_context",
)

Kc_tr_log = _cap_log1p_nonneg(Kc_tr_imp, countlike_caps, block_name="countlike_context/TRAIN")
Kc_va_log = _cap_log1p_nonneg(Kc_va_imp, countlike_caps, block_name="countlike_context/VAL")
Kc_te_log = _cap_log1p_nonneg(Kc_te_imp, countlike_caps, block_name="countlike_context/TEST")

# 3d) Event/ZB context: zero-fill is semantically defensible for event intensity absence.
Ke_tr_imp = _zero_fill_nonnegative(Ke_tr_raw, block_name="event_context/TRAIN")
Ke_va_imp = _zero_fill_nonnegative(Ke_va_raw, block_name="event_context/VAL")
Ke_te_imp = _zero_fill_nonnegative(Ke_te_raw, block_name="event_context/TEST")

event_caps = _fit_caps_train(
    Ke_tr_imp,
    EVENT_CTX_COLS,
    block_name="event_context",
)

Ke_tr_log = _cap_log1p_nonneg(Ke_tr_imp, event_caps, block_name="event_context/TRAIN")
Ke_va_log = _cap_log1p_nonneg(Ke_va_imp, event_caps, block_name="event_context/VAL")
Ke_te_log = _cap_log1p_nonneg(Ke_te_imp, event_caps, block_name="event_context/TEST")

# 3e) Final count-context block:
# Fit ONE scaler on the exact final count-context TRAIN design matrix.
# This replaces the old stale HOTFIX and guarantees artifact lineage consistency.
K_tr_log = np.concatenate([Kc_tr_log, Ke_tr_log], axis=1).astype(np.float32, copy=False)
K_va_log = np.concatenate([Kc_va_log, Ke_va_log], axis=1).astype(np.float32, copy=False)
K_te_log = np.concatenate([Kc_te_log, Ke_te_log], axis=1).astype(np.float32, copy=False)

if K_tr_log.shape[1] != count_ctx_dim:
    raise RuntimeError(
        f"[Cell7] K_tr_log count-context dim mismatch: got={K_tr_log.shape[1]} expected={count_ctx_dim}"
    )

count_ctx_scaler = _fit_standard_scaler_train(K_tr_log, block_name="count_context")

K_tr_s = _transform_scaler(count_ctx_scaler, K_tr_log, block_name="count_context/TRAIN")
K_va_s = _transform_scaler(count_ctx_scaler, K_va_log, block_name="count_context/VAL")
K_te_s = _transform_scaler(count_ctx_scaler, K_te_log, block_name="count_context/TEST")

# Optional compatibility scalers for legacy readers. These are not used to build cond_*.
countlike_scaler = _fit_standard_scaler_train(Kc_tr_log, block_name="countlike_context_compat")
event_scaler = _fit_standard_scaler_train(Ke_tr_log, block_name="event_context_compat")


# ------------------------------------------------------------
# 4) Concatenate final conditioning arrays
# ------------------------------------------------------------
cond_tr = np.concatenate([obs_tr, B_tr, S_tr_s, K_tr_s], axis=1).astype(np.float32, copy=False)
cond_va = np.concatenate([obs_va, B_va, S_va_s, K_va_s], axis=1).astype(np.float32, copy=False)
cond_te = np.concatenate([obs_te, B_te, S_te_s, K_te_s], axis=1).astype(np.float32, copy=False)

for name, arr, expected_n in [
    ("cond_tr", cond_tr, len(df_tr)),
    ("cond_va", cond_va, len(df_va)),
    ("cond_te", cond_te, len(df_te)),
]:
    if arr.shape != (expected_n, cond_dim):
        raise RuntimeError(
            f"[Cell7] {name} shape mismatch: got={arr.shape}, expected={(expected_n, cond_dim)}"
        )

    if not np.isfinite(arr).all():
        raise RuntimeError(f"[Cell7] {name} contains non-finite values.")

globals()["cond_tr"] = cond_tr
globals()["cond_va"] = cond_va
globals()["cond_te"] = cond_te

log(f"[Cell7] Built cond: cond_tr={cond_tr.shape} cond_va={cond_va.shape} cond_te={cond_te.shape}")
log(
    f"[Cell7] cond order: protocol_obs({obs_dim}) + bin({bin_dim}) + "
    f"cont_state({cont_state_dim}) + countlike_ctx({countlike_ctx_dim}) + "
    f"event_ctx({event_ctx_dim})"
)


# ------------------------------------------------------------
# 5) Persist contracts + artifacts
# ------------------------------------------------------------
paths = {
    "obs_cols": os.path.join(ARTDIR, "cond_obs_cols.json"),
    "bin_cols": os.path.join(ARTDIR, "cond_bin_cols.json"),
    "bin_cols_all": os.path.join(ARTDIR, "cond_bin_cols_all_taxonomy.json"),
    "raw_cont_cols": os.path.join(ARTDIR, "cond_raw_cont_cols.json"),
    "raw_cont_cols_all": os.path.join(ARTDIR, "cond_raw_cont_cols_all_taxonomy.json"),
    "cont_state_cols": os.path.join(ARTDIR, "cond_cont_state_cols.json"),
    "cont_state_cols_all": os.path.join(ARTDIR, "cond_cont_state_cols_all_taxonomy.json"),
    "countlike_cols": os.path.join(ARTDIR, "cond_countlike_cols.json"),
    "countlike_cols_all": os.path.join(ARTDIR, "cond_countlike_cols_all_taxonomy.json"),
    "iot_driver_cols": os.path.join(ARTDIR, "cond_iot_driver_cols.json"),
    "zb_driver_cols": os.path.join(ARTDIR, "cond_zb_driver_cols.json"),
    "countlike_ctx_cols": os.path.join(ARTDIR, "cond_countlike_ctx_cols.json"),
    "event_ctx_cols": os.path.join(ARTDIR, "cond_event_ctx_cols.json"),
    "count_ctx_cols": os.path.join(ARTDIR, "cond_count_ctx_cols.json"),
    "state_fill": os.path.join(ARTDIR, "cond_state_fill_values.json"),
    "binary_fill": os.path.join(ARTDIR, "cond_binary_fill_values.json"),
    "countlike_fill": os.path.join(ARTDIR, "cond_countlike_fill_values.json"),
    "countlike_caps": os.path.join(ARTDIR, "cond_countlike_caps.json"),
    "event_caps": os.path.join(ARTDIR, "cond_event_caps.json"),
    "count_caps": os.path.join(ARTDIR, "cond_count_caps.json"),
    "state_scaler": os.path.join(ARTDIR, "cond_state_scaler.joblib"),
    "count_scaler": os.path.join(ARTDIR, "cond_count_scaler.joblib"),
    "countlike_scaler_compat": os.path.join(ARTDIR, "cond_countlike_scaler.joblib"),
    "event_scaler_compat": os.path.join(ARTDIR, "cond_event_scaler.joblib"),
    "support_audit": os.path.join(ARTDIR, "cond_support_audit.json"),
    "no_train_support_audit": os.path.join(ARTDIR, "cond_no_train_support_exclusions.json"),
    "spec": os.path.join(ARTDIR, "cond_spec.json"),
}

_write_json(paths["obs_cols"], OBS_COLS)

_write_json(paths["bin_cols"], BIN_COLS)
_write_json(paths["bin_cols_all"], BIN_COLS_ALL)

_write_json(paths["raw_cont_cols"], RAW_CONT_COLS_ALL)
_write_json(paths["raw_cont_cols_all"], RAW_CONT_COLS_ALL)

_write_json(paths["cont_state_cols"], CONT_STATE_COLS)
_write_json(paths["cont_state_cols_all"], CONT_STATE_COLS_ALL)

_write_json(paths["countlike_cols"], COUNTLIKE_COLS)
_write_json(paths["countlike_cols_all"], COUNTLIKE_COLS_ALL)

_write_json(paths["iot_driver_cols"], IOT_DRV_COLS)
_write_json(paths["zb_driver_cols"], ZB_DRIVER_COLS)
_write_json(paths["countlike_ctx_cols"], COUNTLIKE_CTX_COLS)
_write_json(paths["event_ctx_cols"], EVENT_CTX_COLS)
_write_json(paths["count_ctx_cols"], COUNT_CTX_COLS)

_write_json(
    paths["binary_fill"],
    {BIN_COLS[i]: float(bin_fill[i]) for i in range(len(BIN_COLS))},
)

_write_json(
    paths["state_fill"],
    {CONT_STATE_COLS[i]: float(state_fill[i]) for i in range(len(CONT_STATE_COLS))},
)

_write_json(
    paths["countlike_fill"],
    {COUNTLIKE_CTX_COLS[i]: float(countlike_fill[i]) for i in range(len(COUNTLIKE_CTX_COLS))},
)

_write_json(
    paths["countlike_caps"],
    {COUNTLIKE_CTX_COLS[i]: float(countlike_caps[i]) for i in range(len(COUNTLIKE_CTX_COLS))},
)

_write_json(
    paths["event_caps"],
    {EVENT_CTX_COLS[i]: float(event_caps[i]) for i in range(len(EVENT_CTX_COLS))},
)

combined_caps = {}
for i, c in enumerate(COUNTLIKE_CTX_COLS):
    combined_caps[c] = float(countlike_caps[i])
for i, c in enumerate(EVENT_CTX_COLS):
    combined_caps[c] = float(event_caps[i])
_write_json(paths["count_caps"], combined_caps)

_write_json(paths["support_audit"], raw_support_audit)
_write_json(paths["no_train_support_audit"], COND_NO_TRAIN_SUPPORT_AUDIT)

joblib.dump(state_scaler, paths["state_scaler"])
joblib.dump(count_ctx_scaler, paths["count_scaler"])
joblib.dump(countlike_scaler, paths["countlike_scaler_compat"])
joblib.dump(event_scaler, paths["event_scaler_compat"])

_artifact_scaler_dim_check(paths["state_scaler"], cont_state_dim, name="cond_state_scaler.joblib")
_artifact_scaler_dim_check(paths["count_scaler"], count_ctx_dim, name="cond_count_scaler.joblib")
_artifact_scaler_dim_check(paths["countlike_scaler_compat"], countlike_ctx_dim, name="cond_countlike_scaler.joblib")
_artifact_scaler_dim_check(paths["event_scaler_compat"], event_ctx_dim, name="cond_event_scaler.joblib")

spec = {
    "version": "v13_shared_cps_conditioning_contract",
    "ts_unix": float(time.time()),
    "seed": int(seed),

    "architecture_policy": {
        "conditioning_role": "shared_conditioning_for_protocol_candidate_generators",
        "ddpm_role": "candidate_generator_only_not_final_evidence_by_itself",
        "hybrid_selector": "downstream_VAL_only_candidate_selector",
        "candidate_selection_split": "VAL_only",
        "test_usage": "final_QA_only",
        "iot_cps_coupling_policy_required": True,
        "standalone_iot_synthesis_allowed": False,
    },

    "contract_note": (
        "COUNTLIKE_COLS are a subset of IOT_CONT_COLS; "
        "CONT_STATE_COLS_ALL is IOT_CONT_COLS minus COUNTLIKE_COLS, preserving upstream order. "
        "Protocol obs block comes from Cell 6 DDPM_OBS_* logical matrices. "
        "IoT state missingness is imputed from TRAIN only, not blindly zero-filled. "
        "Valid taxonomy columns with no finite TRAIN support are excluded from conditioning. "
        "The final cond_count_scaler.joblib is fitted directly on the final count-context "
        "TRAIN design matrix used in cond_tr."
    ),

    "protocol_context": {
        "proto_cols_contract_n": int(len(globals()["PROTO_COLS_CONTRACT"])),
        "proto_cols_contract_sha256": _sha256_list(globals()["PROTO_COLS_CONTRACT"]),
        "protocol_generator_role_contract_version": globals()["PROTOCOL_GENERATOR_ROLE_CONTRACT"].get("version"),
        "ddpm_candidate_cols_n": int(len(globals().get("DDPM_VALUE_COLS", []))),
        "ddpm_candidate_cols_sha256": _sha256_list(globals().get("DDPM_VALUE_COLS", [])),
    },

    "iot_cps_coupling_policy_version": globals()["IOT_CPS_COUPLING_POLICY"].get("version"),

    "obs_tiers": OBS_TIERS,
    "obs_cols": OBS_COLS,

    "bin_cols": BIN_COLS,
    "bin_cols_all": BIN_COLS_ALL,

    "raw_cont_cols_all": RAW_CONT_COLS_ALL,
    "cont_state_cols": CONT_STATE_COLS,
    "cont_state_cols_all": CONT_STATE_COLS_ALL,

    "countlike_cols": COUNTLIKE_COLS,
    "countlike_cols_all": COUNTLIKE_COLS_ALL,

    "iot_driver_cols": IOT_DRV_COLS,
    "zb_driver_cols": ZB_DRIVER_COLS,

    "countlike_ctx_cols": COUNTLIKE_CTX_COLS,
    "event_ctx_cols": EVENT_CTX_COLS,
    "count_ctx_cols": COUNT_CTX_COLS,

    "obs_dim": int(obs_dim),
    "bin_dim": int(bin_dim),
    "bin_dim_all_taxonomy": int(len(BIN_COLS_ALL)),
    "raw_cont_dim_all_taxonomy": int(len(RAW_CONT_COLS_ALL)),
    "cont_state_dim": int(cont_state_dim),
    "cont_state_dim_all_taxonomy": int(len(CONT_STATE_COLS_ALL)),
    "countlike_dim": int(countlike_dim),
    "countlike_dim_all_taxonomy": int(len(COUNTLIKE_COLS_ALL)),
    "iot_driver_dim": int(iot_drv_dim),
    "zb_driver_dim": int(zb_drv_dim),
    "countlike_ctx_dim": int(countlike_ctx_dim),
    "event_ctx_dim": int(event_ctx_dim),
    "count_ctx_dim": int(count_ctx_dim),
    "feat_total": int(feat_total),
    "cond_dim": int(cond_dim),

    "full_taxonomy_counts": {
        "iot_bin_cols_all": int(len(BIN_COLS_ALL)),
        "iot_cont_cols_all": int(len(RAW_CONT_COLS_ALL)),
        "iot_cont_state_cols_all": int(len(CONT_STATE_COLS_ALL)),
        "iot_countlike_cols_all": int(len(COUNTLIKE_COLS_ALL)),
    },

    "conditioning_exclusions": {
        "policy": "exclude_from_conditioning_if_no_finite_TRAIN_support",
        "binary_no_train_support": list(BIN_NO_TRAIN_SUPPORT_COLS),
        "continuous_state_no_train_support": list(CONT_STATE_NO_TRAIN_SUPPORT_COLS),
        "countlike_no_train_support": list(COUNTLIKE_NO_TRAIN_SUPPORT_COLS),
        "n_total_no_train_support": int(len(COND_NO_TRAIN_SUPPORT_EXCLUDED_COLS)),
    },

    "imputation_policy": {
        "binary": "TRAIN_mode_per_column",
        "continuous_state": "TRAIN_median_per_column",
        "countlike_context": "TRAIN_median_per_column_then_nonnegative_cap_log1p",
        "event_context": "missing_to_zero_then_nonnegative_cap_log1p",
    },

    "scaling_policy": {
        "continuous_state": "StandardScaler_fit_on_TRAIN_after_TRAIN_median_imputation",
        "count_context": (
            "ONE StandardScaler fit on concat(countlike_log, event_log) TRAIN matrix; "
            "this is the authoritative cond_count_scaler.joblib"
        ),
        "compatibility_scalers": [
            "cond_countlike_scaler.joblib",
            "cond_event_scaler.joblib",
        ],
    },

    "sig_obs": _sha256_list(OBS_COLS),
    "sig_bin": _sha256_list(BIN_COLS),
    "sig_bin_all": _sha256_list(BIN_COLS_ALL),
    "sig_raw_cont_all": _sha256_list(RAW_CONT_COLS_ALL),
    "sig_cont_state": _sha256_list(CONT_STATE_COLS),
    "sig_cont_state_all": _sha256_list(CONT_STATE_COLS_ALL),
    "sig_countlike": _sha256_list(COUNTLIKE_COLS),
    "sig_countlike_all": _sha256_list(COUNTLIKE_COLS_ALL),
    "sig_iot_driver": _sha256_list(IOT_DRV_COLS),
    "sig_zb_driver": _sha256_list(ZB_DRIVER_COLS),
    "sig_countlike_ctx": _sha256_list(COUNTLIKE_CTX_COLS),
    "sig_event_ctx": _sha256_list(EVENT_CTX_COLS),
    "sig_count_ctx": _sha256_list(COUNT_CTX_COLS),

    "cfg_expectations_used": {
        "expected_ddpm_obs_dim": int(expected_obs_dim),
        "expected_cond_iot_bin_cols": int(expected_bin),
        "expected_cond_iot_cont_state_cols": int(expected_cont),
        "expected_cond_iot_countlike_cols": int(expected_countlike),
        "expected_iot_driver_cols": int(expected_drv),
        "expected_zb_driver_cols": int(expected_zb_drv),
        "expected_cond_dim": int(expected_cond_dim),
    },

    "artifact_paths": {k: os.path.basename(v) for k, v in paths.items()},
}

spec["sig_cond"] = _sha256_list([
    spec["sig_obs"],
    spec["sig_bin"],
    spec["sig_cont_state"],
    spec["sig_countlike_ctx"],
    spec["sig_event_ctx"],
])

SHARED_CONDITIONING_CONTRACT = {
    "component": "shared_conditioning_contract",
    "component_type": "conditioning_matrix_builder",
    "version": "v7_STUDY_THESIS_shared_conditioning",
    "generator_family": "none_conditioning_helper_only",
    "supports_generator_families": [
        "DDPM_candidate_generator",
        "SeqDenoiser_candidate_generator",
        "other_conditional_protocol_candidate_generators",
    ],
    "branch_scope": [
        "protocol_candidate_generation",
        "continuous_candidate_generation",
        "cross_modal_candidate_generation",
    ],
    "parameters": {
        "obs_context": "protocol logical observation pass-through",
        "binary_state": "TRAIN-supported binary columns with TRAIN-mode imputation",
        "continuous_state": "TRAIN-supported continuous residual columns with TRAIN-median imputation and TRAIN-fitted StandardScaler",
        "countlike_context": "TRAIN-supported countlike state columns with TRAIN-median imputation, nonnegative cap, log1p",
        "event_driver_context": "event absence zero-fill, nonnegative cap, log1p",
        "count_context_scaler": "single StandardScaler fitted on TRAIN final count-context matrix",
        "seed": int(seed),
        "cond_dim": int(cond_dim),
        "obs_dim": int(obs_dim),
        "bin_dim": int(bin_dim),
        "cont_state_dim": int(cont_state_dim),
        "countlike_ctx_dim": int(countlike_ctx_dim),
        "event_ctx_dim": int(event_ctx_dim),
        "count_ctx_dim": int(count_ctx_dim),
    },
    "train_fit_scope": (
        "TRAIN only: binary fill, continuous median, countlike median, caps, "
        "and StandardScaler parameters are fitted from TRAIN."
    ),
    "val_selection_metric": "none_inside_conditioning_builder",
    "test_only_qa": (
        "TEST conditioning matrix is transformed using TRAIN-fitted parameters only; "
        "TEST statistics are written only to audit fields and are not used for fitting, "
        "selection, thresholding, or repair."
    ),
    "contributes_to_final_artifact": False,
    "final_artifact_contribution_rule": (
        "This cell creates conditioning arrays and contracts only. It does not generate, "
        "select, repair, overwrite, or release synthetic values. Downstream branch ledgers "
        "must determine whether a candidate contributes to the final artifact."
    ),
    "leakage_status": {
        "fits_on_train": True,
        "uses_val_for_selection_inside_cell": False,
        "uses_test_for_fitting": False,
        "uses_test_for_threshold_selection": False,
        "uses_test_for_candidate_selection": False,
        "uses_test_for_repair": False,
        "mutates_synthetic_artifact": False,
        "overwrites_final_artifact": False,
    },
    "artifact_paths": {k: os.path.basename(v) for k, v in paths.items()},
    "sig_cond": spec["sig_cond"],
}

_write_json(paths["spec"], spec)

conditioning_contract_path = os.path.join(
    CONTRACT_DIR,
    "shared_conditioning_contract_cell7.json",
)
_write_json(conditioning_contract_path, SHARED_CONDITIONING_CONTRACT)

if "RUN_META" in globals():
    RUN_META.setdefault("helper_contracts", {})
    RUN_META["helper_contracts"]["shared_conditioning_contract_cell7"] = SHARED_CONDITIONING_CONTRACT




# ------------------------------------------------------------
# 6) Export globals
# ------------------------------------------------------------
globals()["COND_OBS_COLS"] = list(OBS_COLS)
globals()["COND_OBS_TIERS"] = list(OBS_TIERS)

globals()["COND_BIN_COLS"] = list(BIN_COLS)
globals()["COND_BIN_COLS_ALL"] = list(BIN_COLS_ALL)

globals()["COND_RAW_CONT_COLS"] = list(RAW_CONT_COLS_ALL)
globals()["COND_RAW_CONT_COLS_ALL"] = list(RAW_CONT_COLS_ALL)

globals()["COND_CONT_STATE_COLS"] = list(CONT_STATE_COLS)
globals()["COND_CONT_STATE_COLS_ALL"] = list(CONT_STATE_COLS_ALL)

globals()["COND_COUNTLIKE_COLS"] = list(COUNTLIKE_COLS)
globals()["COND_COUNTLIKE_COLS_ALL"] = list(COUNTLIKE_COLS_ALL)

globals()["COND_IOT_DRIVER_COLS"] = list(IOT_DRV_COLS)
globals()["COND_ZB_DRIVER_COLS"] = list(ZB_DRIVER_COLS)

globals()["COND_COUNTLIKE_CTX_COLS"] = list(COUNTLIKE_CTX_COLS)
globals()["COND_EVENT_CTX_COLS"] = list(EVENT_CTX_COLS)
globals()["COND_COUNT_CTX_COLS"] = list(COUNT_CTX_COLS)

globals()["COND_DIM"] = int(cond_dim)

globals()["COND_BINARY_FILL"] = bin_fill
globals()["COND_STATE_FILL"] = state_fill
globals()["COND_COUNTLIKE_FILL"] = countlike_fill
globals()["COND_COUNTLIKE_CAPS"] = countlike_caps
globals()["COND_EVENT_CAPS"] = event_caps
globals()["COND_COUNT_CAPS"] = combined_caps

globals()["COND_STATE_SCALER"] = state_scaler
globals()["COND_COUNT_SCALER"] = count_ctx_scaler
globals()["COND_COUNTLIKE_SCALER"] = countlike_scaler
globals()["COND_EVENT_SCALER"] = event_scaler

globals()["COND_NO_TRAIN_SUPPORT_EXCLUDED_COLS"] = list(COND_NO_TRAIN_SUPPORT_EXCLUDED_COLS)
globals()["COND_BIN_NO_TRAIN_SUPPORT_COLS"] = list(BIN_NO_TRAIN_SUPPORT_COLS)
globals()["COND_CONT_STATE_NO_TRAIN_SUPPORT_COLS"] = list(CONT_STATE_NO_TRAIN_SUPPORT_COLS)
globals()["COND_COUNTLIKE_NO_TRAIN_SUPPORT_COLS"] = list(COUNTLIKE_NO_TRAIN_SUPPORT_COLS)
globals()["COND_NO_TRAIN_SUPPORT_AUDIT"] = COND_NO_TRAIN_SUPPORT_AUDIT

globals()["COND_SPEC"] = spec
globals()["SHARED_CONDITIONING_CONTRACT"] = SHARED_CONDITIONING_CONTRACT
globals()["COND_CONTRACT_PATH"] = conditioning_contract_path

log(f"[Cell7] Saved cond spec: {paths['spec']} | sig_cond={spec['sig_cond'][:12]}...")
log(f"[Cell7] Saved support audit: {paths['support_audit']}")
log(f"[Cell7] Saved no-TRAIN-support audit: {paths['no_train_support_audit']}")
log(
    "[Cell7] Saved authoritative count-context scaler: "
    f"{paths['count_scaler']} | count_ctx_dim={count_ctx_dim}"
)
log(f"[Cell7] Saved shared conditioning contract: {conditioning_contract_path}")
log("[Cell7] Conditioning builder registered as helper only; no artifact generation or candidate selection performed.")
log("--- END:   Cell 7 ---")