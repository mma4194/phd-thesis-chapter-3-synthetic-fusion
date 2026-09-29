# ============================================================
# CELL 2 — Core helpers (CPS unified dataset hardened) — v5
#
# Purpose:
# - provide strict reusable helpers for schema, time, masks, drivers,
#   leakage prevention, and train-only constant pruning
# - materialize the canonical time column required by later cells
# - keep raw epoch time out of modeling features
#
# Alignment with Cell 1 v3.1:
# - raw time source is CFG["raw_time_col"] == "sec"
# - canonical target is CFG["canonical_time_col"] == "sec_epoch_s__canon"
# - Cell 1 validates that "sec" is strict 1 Hz Unix epoch seconds
# - Cell 2 materializes and validates "sec_epoch_s__canon"
#
# Design principles:
# - no silent zero-fill for required conditioning columns
# - no silent acceptance of duplicate columns
# - no hidden re-entry of raw/provenance time into modeling features
# - preserve events_in_sec* bases while dropping absolute-time leakage
# - protect observability semantics from accidental pruning
# - fail closed on schema ambiguity
#
# Hardening in v5:
# - adds materialize_canonical_time_column()
# - detect_time_column now handles raw+canonical coexistence safely
# - absolute-time detection understands raw time, canonical time, and provenance
# - provenance columns are explicitly modeling-forbidden
# - protocol_obs_matrix supports ordered tier subsets
# - object/category rejection remains explicit
# - train-only constant dropping rejects provenance leakage by default
# ============================================================

import re
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_bool_dtype,
    is_integer_dtype,
    is_object_dtype,
)
from pandas import CategoricalDtype

from typing import List, Tuple, Optional, Sequence, Dict, Any

# ------------------------------------------------------------
# Small logging helper
# ------------------------------------------------------------
def _log_msg(msg: str) -> None:
    if "log" in globals() and callable(log):
        log(msg)
    else:
        print(msg)


# ------------------------------------------------------------
# Config access helpers
# ------------------------------------------------------------
def _cfg_get(key: str, default: Any = None) -> Any:
    if "CFG" in globals() and isinstance(CFG, dict):
        return CFG.get(key, default)
    return default


def get_raw_time_col(default: str = "sec") -> str:
    return str(_cfg_get("raw_time_col", default))


def get_canonical_time_col(default: str = "sec_epoch_s__canon") -> str:
    return str(_cfg_get("canonical_time_col", default))


def get_time_candidates(default: Optional[Sequence[str]] = None) -> List[str]:
    if default is None:
        default = [
            get_canonical_time_col(),
            get_raw_time_col(),
            "sec_epoch_s",
            "ts",
            "timestamp",
            "time_s",
        ]
    x = _cfg_get("time_col_candidates", list(default))
    return list(x)


# ------------------------------------------------------------
# Canonical tiers
# ------------------------------------------------------------
TIERS = ["router", "ota", "zigbee", "zwave"]
PROTOCOL_TIERS = list(TIERS)


def assert_tiers_equal(tiers: Sequence[str]) -> None:
    if list(tiers) != TIERS:
        raise RuntimeError(f"TIER ORDER MISMATCH. Expected {TIERS} but got {list(tiers)}")


def assert_tiers_ordered_subset(
    tiers: Sequence[str],
    full: Optional[Sequence[str]] = None,
) -> None:
    full = list(TIERS if full is None else full)
    x = list(tiers)

    if len(x) != len(set(x)):
        raise RuntimeError(f"Duplicate tier entries detected: {x}")

    for item in x:
        if item not in full:
            raise RuntimeError(f"Unknown tier '{item}'. Allowed={full}")

    idx = [full.index(item) for item in x]
    if idx != sorted(idx):
        raise RuntimeError(f"Tier order mismatch. Expected subset order from {full}, got {x}")


# ------------------------------------------------------------
# Column integrity helpers
# ------------------------------------------------------------
def assert_no_duplicate_columns(df: pd.DataFrame, where: str = "DataFrame") -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{where}] Expected pandas DataFrame, got {type(df)}")

    dup_mask = df.columns.duplicated(keep=False)
    if dup_mask.any():
        dupes = df.columns[dup_mask].tolist()
        uniq = []
        seen = set()
        for c in dupes:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        raise RuntimeError(
            f"[{where}] Duplicate columns detected ({len(uniq)} unique duplicates): {uniq[:20]}"
        )


def assert_expected_columns_present(
    df: pd.DataFrame,
    required_cols: Sequence[str],
    *,
    where: str,
) -> None:
    assert_no_duplicate_columns(df, where)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[{where}] Missing required columns ({len(missing)}): {missing[:20]}"
            + (" ..." if len(missing) > 20 else "")
        )


def assert_no_unexpected_object_columns(
    df: pd.DataFrame,
    *,
    allow_cols: Optional[Sequence[str]] = None,
    where: str = "DataFrame",
) -> None:
    assert_no_duplicate_columns(df, where)

    allow = set(allow_cols or [])
    bad = []

    for c in df.columns:
        if c in allow:
            continue

        dt = df[c].dtype
        if is_object_dtype(dt) or isinstance(dt, CategoricalDtype):
            bad.append((c, str(dt)))

    if bad:
        preview = [f"{c}:{dt}" for c, dt in bad[:20]]
        raise RuntimeError(
            f"[{where}] Unexpected object/category columns detected ({len(bad)}): {preview}"
        )


def assert_columns_equal_ordered(
    frames: Sequence[pd.DataFrame],
    names: Sequence[str],
    *,
    where: str,
) -> None:
    if len(frames) != len(names):
        raise ValueError("frames and names must have the same length.")
    if len(frames) == 0:
        raise ValueError("No frames supplied.")

    base_cols = frames[0].columns
    base_name = names[0]

    for df, nm in zip(frames, names):
        assert_no_duplicate_columns(df, f"{where}/{nm}")

    for df, nm in zip(frames[1:], names[1:]):
        if not base_cols.equals(df.columns):
            missing = [c for c in base_cols if c not in df.columns]
            extra = [c for c in df.columns if c not in base_cols]
            raise RuntimeError(
                f"[{where}] Column mismatch between {base_name} and {nm}. "
                f"missing_in_{nm}={missing[:20]} | extra_in_{nm}={extra[:20]}"
            )


# ------------------------------------------------------------
# Time canonicalization
# ------------------------------------------------------------
def _coerce_time_to_epoch_seconds_series(
    s: pd.Series,
    *,
    source_name: str,
    require_integer_like: bool = True,
) -> pd.Series:
    """
    Convert a numeric/datetime time column into epoch-second float64 Series.

    Policy:
    - datetime values are converted to UTC epoch seconds
    - numeric values must be finite
    - for this CPS full-grid dataset, numeric time should be integer-like
    """
    if is_datetime64_any_dtype(s.dtype):
        ts = pd.to_datetime(s, utc=True, errors="coerce")
        if ts.isna().any():
            n_bad = int(ts.isna().sum())
            raise RuntimeError(
                f"[time canonicalization] '{source_name}' contains {n_bad} non-parsable datetime rows."
            )

        sec = ts.astype("int64").astype("float64") / 1e9
        out = pd.Series(sec.to_numpy(dtype=np.float64, copy=False), index=s.index, name=source_name)
    elif is_numeric_dtype(s.dtype):
        out = pd.to_numeric(s, errors="coerce").astype("float64")
        out = pd.Series(out.to_numpy(dtype=np.float64, copy=False), index=s.index, name=source_name)
    else:
        raise RuntimeError(
            f"[time canonicalization] Time column '{source_name}' is not numeric/datetime. dtype={s.dtype}"
        )

    arr = out.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"[time canonicalization] Time column '{source_name}' contains {n_bad} non-finite rows."
        )

    if require_integer_like:
        rounded = np.rint(arr)
        max_abs_frac = float(np.max(np.abs(arr - rounded))) if len(arr) else 0.0
        if max_abs_frac > 1e-6:
            raise RuntimeError(
                f"[time canonicalization] Time column '{source_name}' is not integer-like epoch seconds. "
                f"max_abs_fractional_error={max_abs_frac}"
            )
        out = pd.Series(rounded.astype(np.float64), index=s.index, name=source_name)

    return out


def validate_epoch_second_series(
    sec_epoch: pd.Series,
    *,
    where: str,
    expected_hz: int = 1,
    require_monotonic: bool = True,
    require_unique: bool = True,
) -> Dict[str, Any]:
    """
    Validate strict full-grid epoch-second semantics.
    """
    sec = pd.to_numeric(sec_epoch, errors="coerce").to_numpy(dtype=np.float64, copy=False)

    if len(sec) == 0:
        raise RuntimeError(f"[{where}] Empty time series.")

    if not np.isfinite(sec).all():
        n_bad = int((~np.isfinite(sec)).sum())
        raise RuntimeError(f"[{where}] Non-finite epoch seconds detected: {n_bad}")

    rounded = np.rint(sec)
    max_abs_frac = float(np.max(np.abs(sec - rounded)))
    if max_abs_frac > 1e-6:
        raise RuntimeError(
            f"[{where}] Epoch seconds are not integer-like. max_abs_fractional_error={max_abs_frac}"
        )

    sec_i = rounded.astype(np.int64)
    diffs = np.diff(sec_i)

    if require_monotonic and len(diffs) and np.any(diffs <= 0):
        bad_idx = int(np.where(diffs <= 0)[0][0])
        raise RuntimeError(
            f"[{where}] Time is not strictly increasing. "
            f"first_bad_idx={bad_idx} | t[i]={sec_i[bad_idx]} | t[i+1]={sec_i[bad_idx + 1]}"
        )

    if require_unique and len(np.unique(sec_i)) != len(sec_i):
        raise RuntimeError(f"[{where}] Duplicate epoch-second timestamps detected.")

    expected_step = int(round(1.0 / float(expected_hz)))
    if expected_step <= 0:
        raise RuntimeError(f"[{where}] Unsupported expected_hz={expected_hz}")

    if len(diffs):
        unique_steps, step_counts = np.unique(diffs, return_counts=True)
        step_counts_map = {int(k): int(v) for k, v in zip(unique_steps, step_counts)}
        bad_steps = {k: v for k, v in step_counts_map.items() if k != expected_step}
    else:
        step_counts_map = {}
        bad_steps = {}

    if bad_steps:
        raise RuntimeError(
            f"[{where}] Time is not a strict {expected_hz} Hz full grid. "
            f"expected_step={expected_step} | bad_steps={bad_steps}"
        )

    return {
        "n": int(len(sec_i)),
        "first_sec": int(sec_i[0]),
        "last_sec": int(sec_i[-1]),
        "duration_seconds_inclusive": int(sec_i[-1] - sec_i[0] + 1),
        "min_step": int(diffs.min()) if len(diffs) else None,
        "max_step": int(diffs.max()) if len(diffs) else None,
        "median_step": float(np.median(diffs)) if len(diffs) else None,
        "step_counts": step_counts_map,
        "is_strictly_increasing": bool(np.all(diffs > 0)) if len(diffs) else True,
        "is_unique": bool(len(np.unique(sec_i)) == len(sec_i)),
        "is_epoch_second_plausible": bool(sec_i[0] > 1_000_000_000 and sec_i[-1] > sec_i[0]),
    }


def materialize_canonical_time_column(
    df: pd.DataFrame,
    *,
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
    expected_hz: Optional[int] = None,
    require_monotonic: Optional[bool] = None,
    require_unique: Optional[bool] = None,
    overwrite: bool = False,
    where: str = "materialize_canonical_time_column",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create/validate the canonical time column expected by later cells.

    For the current dataset:
      raw_time_col        = "sec"
      canonical_time_col  = "sec_epoch_s__canon"

    Policy:
    - raw time must exist unless canonical already exists and overwrite=False
    - canonical time is int64 epoch seconds
    - if canonical already exists, it must match raw time exactly unless overwrite=True
    - no feature engineering beyond canonical time materialization occurs here
    """
    assert_no_duplicate_columns(df, where)

    raw_time_col = str(raw_time_col or get_raw_time_col())
    canonical_time_col = str(canonical_time_col or get_canonical_time_col())
    expected_hz = int(expected_hz if expected_hz is not None else _cfg_get("expected_hz", 1))
    require_monotonic = bool(
        require_monotonic if require_monotonic is not None else _cfg_get("require_monotonic_time", True)
    )
    require_unique = bool(
        require_unique if require_unique is not None else _cfg_get("require_unique_time", True)
    )

    out = df.copy()

    raw_exists = raw_time_col in out.columns
    canon_exists = canonical_time_col in out.columns

    if not raw_exists and not canon_exists:
        raise RuntimeError(
            f"[{where}] Neither raw_time_col nor canonical_time_col exists. "
            f"raw={raw_time_col} | canonical={canonical_time_col}"
        )

    if raw_exists:
        raw_sec_f = _coerce_time_to_epoch_seconds_series(
            out[raw_time_col],
            source_name=raw_time_col,
            require_integer_like=True,
        )
        raw_sec_i = np.rint(raw_sec_f.to_numpy(dtype=np.float64, copy=False)).astype(np.int64)
    else:
        raw_sec_i = None

    if canon_exists:
        canon_sec_f = _coerce_time_to_epoch_seconds_series(
            out[canonical_time_col],
            source_name=canonical_time_col,
            require_integer_like=True,
        )
        canon_sec_i = np.rint(canon_sec_f.to_numpy(dtype=np.float64, copy=False)).astype(np.int64)

        if raw_sec_i is not None and not np.array_equal(raw_sec_i, canon_sec_i):
            max_abs = int(np.max(np.abs(raw_sec_i - canon_sec_i))) if len(raw_sec_i) else 0
            if not overwrite:
                raise RuntimeError(
                    f"[{where}] Existing canonical time does not match raw time. "
                    f"raw={raw_time_col} | canonical={canonical_time_col} | max_abs_diff={max_abs}"
                )
            _log_msg(
                f"[{where}] Overwriting mismatched canonical time '{canonical_time_col}' "
                f"from raw source '{raw_time_col}'. max_abs_diff={max_abs}"
            )
            out[canonical_time_col] = raw_sec_i.astype(np.int64)
    else:
        if raw_sec_i is None:
            raise RuntimeError(
                f"[{where}] Cannot materialize canonical time without raw source '{raw_time_col}'."
            )
        out[canonical_time_col] = raw_sec_i.astype(np.int64)
        _log_msg(
            f"[{where}] Materialized canonical time '{canonical_time_col}' from raw source '{raw_time_col}'."
        )

    contract = validate_epoch_second_series(
        out[canonical_time_col],
        where=where,
        expected_hz=expected_hz,
        require_monotonic=require_monotonic,
        require_unique=require_unique,
    )

    if not bool(contract["is_epoch_second_plausible"]):
        raise RuntimeError(
            f"[{where}] Canonical time is not plausible Unix epoch seconds: {contract}"
        )

    return out, contract


def detect_time_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
    *,
    preferred: Optional[str] = None,
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
    allow_raw_and_canonical_pair: bool = True,
) -> Tuple[str, pd.Series]:
    """
    Strict but pipeline-aware detection.

    Policy:
    - at least one candidate must exist
    - if canonical exists, prefer it
    - raw+canonical coexistence is valid only when explicitly allowed
    - multiple unrelated time candidates remain an error
    - returns source column name and epoch-seconds Series as float64
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("detect_time_column expects a pandas DataFrame.")
    if not candidates:
        raise ValueError("detect_time_column received an empty candidates list.")

    assert_no_duplicate_columns(df, "detect_time_column")

    raw_time_col = str(raw_time_col or get_raw_time_col())
    canonical_time_col = str(canonical_time_col or get_canonical_time_col())
    preferred = str(preferred or canonical_time_col)

    found = [c for c in candidates if c in df.columns]

    if not found:
        raise RuntimeError(f"No valid time column detected. Expected one of: {list(candidates)}")

    if preferred in found:
        chosen = preferred
    elif allow_raw_and_canonical_pair and set(found).issubset({raw_time_col, canonical_time_col}):
        chosen = canonical_time_col if canonical_time_col in found else raw_time_col
    elif len(found) == 1:
        chosen = found[0]
    else:
        raise RuntimeError(
            f"[detect_time_column] Ambiguous time columns detected: {found}. "
            "Expected only canonical/raw controlled pair or provide a stricter candidate list."
        )

    sec = _coerce_time_to_epoch_seconds_series(
        df[chosen],
        source_name=chosen,
        require_integer_like=True,
    )
    return chosen, sec


# ------------------------------------------------------------
# Provenance and leakage detection
# ------------------------------------------------------------
def is_provenance_column(col: str) -> bool:
    return isinstance(col, str) and col.startswith("PROVENANCE__")


def is_modeling_forbidden_column(
    col: str,
    *,
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
) -> bool:
    """
    Columns that must never be used as modeling features.

    This includes:
    - raw absolute time
    - canonical absolute time
    - provenance copies of time
    """
    if not isinstance(col, str):
        return False

    raw_time_col = str(raw_time_col or get_raw_time_col())
    canonical_time_col = str(canonical_time_col or get_canonical_time_col())

    if col in {raw_time_col, canonical_time_col}:
        return True

    if is_provenance_column(col):
        return True

    return False


# ------------------------------------------------------------
# Absolute-time leakage detection
# ------------------------------------------------------------
# Drop sec* / *__sec* absolute-time features
# but NEVER drop true events_in_sec* driver bases.
# ------------------------------------------------------------
EVENT_BASE_RX = re.compile(r"(^|__)events_in_sec(__|$)", re.IGNORECASE)

ABS_TIME_RX = re.compile(
    r"""
    (^sec($|__)) |
    (^sec_epoch_s($|__)) |
    (^timestamp($|__)) |
    (^time_s($|__)) |
    (^epoch($|__)) |
    (__sec$) |
    (__sec_epoch_s$) |
    (__timestamp$) |
    (__time_s$) |
    (__epoch$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_observability_or_staleness_col(col: str) -> bool:
    if not isinstance(col, str):
        return False

    cl = col.lower()

    return bool(
        cl.endswith("__obs_present")
        or cl.endswith("__traffic_present")
        or cl.endswith("__traffic_obs_present")
        or cl.endswith("__stale_flag")
        or cl.endswith("__staleness_s")
        or cl.startswith("iot__entity_obs__")
        or cl.startswith("iot__entity_stale__")
    )


def is_abs_time_feature(
    col: str,
    time_col: Optional[str] = None,
    *,
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
    include_canonical: bool = True,
) -> bool:
    """
    Detect absolute-time features that leak wall-clock position or absolute chronology.

    Protected:
    - events_in_sec* driver bases
    - observability / stale metadata

    Forbidden:
    - raw time source
    - canonical absolute time, unless include_canonical=False
    - provenance copies
    - sec*/timestamp*/epoch-like derived absolute-time features
    """
    if not isinstance(col, str):
        return False

    raw_time_col = str(raw_time_col or get_raw_time_col())
    canonical_time_col = str(canonical_time_col or get_canonical_time_col())

    if time_col is not None and col == str(time_col):
        # Kept for backward compatibility with old callers. Prefer explicit
        # raw/canonical handling below.
        return bool(include_canonical)

    if col == raw_time_col:
        return True

    if col == canonical_time_col:
        return bool(include_canonical)

    if is_provenance_column(col):
        return True

    if EVENT_BASE_RX.search(col):
        return False

    if is_observability_or_staleness_col(col):
        return False

    return bool(ABS_TIME_RX.search(col.lower()))


def collect_abs_time_features(
    cols: Sequence[str],
    time_col: Optional[str] = None,
    *,
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
    include_canonical: bool = True,
) -> List[str]:
    return [
        c for c in cols
        if is_abs_time_feature(
            c,
            time_col=time_col,
            raw_time_col=raw_time_col,
            canonical_time_col=canonical_time_col,
            include_canonical=include_canonical,
        )
    ]


def drop_modeling_forbidden_columns(
    df: pd.DataFrame,
    *,
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
    extra_forbidden: Optional[Sequence[str]] = None,
    where: str = "drop_modeling_forbidden_columns",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove columns that must not enter modeling.

    This should be called before building modeling matrices.
    """
    assert_no_duplicate_columns(df, where)

    raw_time_col = str(raw_time_col or get_raw_time_col())
    canonical_time_col = str(canonical_time_col or get_canonical_time_col())
    extra = set(extra_forbidden or [])

    drop_cols = []
    for c in df.columns:
        if (
            is_modeling_forbidden_column(
                c,
                raw_time_col=raw_time_col,
                canonical_time_col=canonical_time_col,
            )
            or c in extra
        ):
            drop_cols.append(c)

    if not drop_cols:
        return df.copy(), []

    out = df.drop(columns=drop_cols)
    _log_msg(f"[{where}] dropped modeling-forbidden columns ({len(drop_cols)}): {drop_cols[:20]}")
    return out, drop_cols


# ------------------------------------------------------------
# Constant feature removal (TRAIN only)
# ------------------------------------------------------------
# Protect:
# - protocol observability masks
# - IoT observability/stale columns
# - IoT driver/event/telemetry helper columns
# - Zigbee channel
# - Z-Wave placeholder obs column
#
# Do NOT protect:
# - raw time
# - canonical time
# - PROVENANCE__* columns
# ------------------------------------------------------------
PROTECT_RX = re.compile(
    r"""
    (__obs_present$)|
    (__traffic_present$)|
    (__traffic_obs_present$)|
    (__stale_flag$)|
    (__staleness_s$)|
    (^iot__entity_obs__)|
    (^iot__entity_stale__)|
    (^events_in_sec__)|
    (^telemetry_in_sec__)|
    (^zigbee__channel$)|
    (^zwave__obs_present$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_protected_constant_drop_column(col: str) -> bool:
    if not isinstance(col, str):
        return False

    if is_provenance_column(col):
        return False

    if col in {get_raw_time_col(), get_canonical_time_col()}:
        return False

    return bool(PROTECT_RX.search(col))


def drop_constants_train_only(
    Xtr: pd.DataFrame,
    Xva: pd.DataFrame,
    Xte: pd.DataFrame,
    *,
    rel_eps: float = 1e-12,
    fail_on_object_cols: bool = True,
    fail_on_forbidden_cols: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    assert_columns_equal_ordered(
        [Xtr, Xva, Xte],
        ["Xtr", "Xva", "Xte"],
        where="drop_constants_train_only",
    )

    if fail_on_forbidden_cols:
        forbidden = [
            c for c in Xtr.columns
            if is_modeling_forbidden_column(c)
        ]
        if forbidden:
            raise RuntimeError(
                "[drop_constants_train_only] Modeling-forbidden columns are still present: "
                f"{forbidden[:20]}"
            )

    if fail_on_object_cols:
        assert_no_unexpected_object_columns(Xtr, where="drop_constants_train_only/Xtr")
        assert_no_unexpected_object_columns(Xva, where="drop_constants_train_only/Xva")
        assert_no_unexpected_object_columns(Xte, where="drop_constants_train_only/Xte")

    if not np.isfinite(float(rel_eps)) or float(rel_eps) < 0:
        raise ValueError(f"[drop_constants_train_only] rel_eps must be finite and >= 0. Got {rel_eps}")

    cols = Xtr.columns.tolist()
    keep_mask = np.ones(len(cols), dtype=bool)
    dropped: List[str] = []

    for i, c in enumerate(cols):
        if is_protected_constant_drop_column(c):
            continue

        s = Xtr[c]

        if is_bool_dtype(s.dtype) or is_integer_dtype(s.dtype):
            nunq = int(s.nunique(dropna=True))
            if nunq <= 1:
                keep_mask[i] = False
                dropped.append(c)
            continue

        if is_numeric_dtype(s.dtype):
            arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
            finite = np.isfinite(arr)

            if not finite.any():
                keep_mask[i] = False
                dropped.append(c)
                continue

            x = arr[finite]

            if np.unique(x).size <= 1:
                keep_mask[i] = False
                dropped.append(c)
                continue

            std = float(np.std(x))
            mean_abs = float(np.mean(np.abs(x)))
            thr = float(rel_eps) * max(1.0, mean_abs)

            if std <= thr:
                keep_mask[i] = False
                dropped.append(c)
            continue

        # Object/category columns have already failed if fail_on_object_cols=True.
        # Other extension dtypes are kept rather than silently coerced.
        continue

    kept_cols = Xtr.columns[keep_mask].tolist()
    _log_msg(f"[drop_constants_train_only] kept={len(kept_cols)} dropped={len(dropped)}")

    return (
        Xtr.loc[:, kept_cols].copy(),
        Xva.loc[:, kept_cols].copy(),
        Xte.loc[:, kept_cols].copy(),
        dropped,
    )


# ------------------------------------------------------------
# Raw time column quarantine
# ------------------------------------------------------------
def quarantine_raw_time_column(
    df: pd.DataFrame,
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
    *,
    keep_provenance: bool = True,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    True quarantine:
    - removes raw source time column from dataframe namespace
    - optionally preserves a provenance copy under PROVENANCE__*
    - provenance columns are explicitly modeling-forbidden
    """
    assert_no_duplicate_columns(df, "quarantine_raw_time_column")

    raw_time_col = str(raw_time_col or get_raw_time_col())
    canonical_time_col = str(canonical_time_col or get_canonical_time_col())

    if raw_time_col not in df.columns:
        raise RuntimeError(f"[time quarantine] raw_time_col not found: {raw_time_col}")

    if canonical_time_col not in df.columns:
        raise RuntimeError(f"[time quarantine] canonical_time_col not found: {canonical_time_col}")

    if raw_time_col == canonical_time_col:
        return df.copy(), None

    out = df.copy()
    prov_col = None

    if keep_provenance:
        prov_col = f"PROVENANCE__{raw_time_col}__raw_time"
        if prov_col in out.columns:
            raise RuntimeError(
                f"[time quarantine] Refusing to overwrite existing provenance column: {prov_col}"
            )
        out[prov_col] = out[raw_time_col].to_numpy(copy=True)

    out = out.drop(columns=[raw_time_col])

    if keep_provenance:
        _log_msg(
            f"[time quarantine] removed modeling-visible raw time '{raw_time_col}', "
            f"preserved as modeling-forbidden '{prov_col}'"
        )
    else:
        _log_msg(f"[time quarantine] removed modeling-visible raw time '{raw_time_col}'")

    return out, prov_col


# ------------------------------------------------------------
# Time-of-day features
# ------------------------------------------------------------
def make_tod_features(sec_epoch: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cyclic time-of-day features only.

    This preserves daily periodicity while discarding absolute date.
    """
    sec = pd.to_numeric(sec_epoch, errors="coerce").to_numpy(dtype=np.float64, copy=False)

    if not np.isfinite(sec).all():
        n_bad = int((~np.isfinite(sec)).sum())
        raise RuntimeError(f"[make_tod_features] Non-finite sec_epoch values: {n_bad}")

    day = 86400.0
    phase = 2.0 * np.pi * ((sec % day) / day)

    sin_t = np.sin(phase).astype(np.float32, copy=False)
    cos_t = np.cos(phase).astype(np.float32, copy=False)
    return sin_t, cos_t


# ------------------------------------------------------------
# Protocol observation matrix
# ------------------------------------------------------------
def protocol_obs_matrix(
    df: pd.DataFrame,
    tiers: Sequence[str],
    *,
    require_all_cols: bool = True,
    coerce_binary: bool = True,
) -> np.ndarray:
    assert_tiers_ordered_subset(tiers)
    assert_no_duplicate_columns(df, "protocol_obs_matrix")

    N = len(df)
    mats = []
    missing = []

    for t in tiers:
        col = f"{t}__obs_present"

        if col not in df.columns:
            missing.append(col)
            if require_all_cols:
                continue
            mats.append(np.zeros(N, dtype=np.float32))
            continue

        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        x = x.copy()
        x[~np.isfinite(x)] = 0.0

        if coerce_binary:
            x = (x > 0.0).astype(np.float32, copy=False)
        else:
            x = x.astype(np.float32, copy=False)

        mats.append(x)

    if require_all_cols and missing:
        raise RuntimeError(
            f"[protocol_obs_matrix] Missing required obs columns: {missing}"
        )

    if len(mats) != len(list(tiers)):
        raise RuntimeError(
            f"[protocol_obs_matrix] Internal matrix count mismatch. mats={len(mats)} tiers={len(list(tiers))}"
        )

    out = np.stack(mats, axis=1).astype(np.float32, copy=False)

    if out.shape != (N, len(list(tiers))):
        raise RuntimeError(
            f"[protocol_obs_matrix] Unexpected shape {out.shape}; expected {(N, len(list(tiers)))}"
        )

    return out


# ------------------------------------------------------------
# IoT driver matrix
# ------------------------------------------------------------
def build_iot_driver_matrix_from_df(
    df: pd.DataFrame,
    driver_cols: Sequence[str],
    *,
    require_all_driver_cols: bool = True,
    clip_negative_to_zero: bool = True,
    nan_policy: str = "zero",
) -> np.ndarray:
    """
    Build the explicit IoT driver matrix from named driver columns.

    Critical policy:
    - missing driver columns are an error by default
    - no silent zero-filling for broken contracts unless explicitly requested
    - NaN handling is explicit
    """
    assert_no_duplicate_columns(df, "build_iot_driver_matrix_from_df")

    if driver_cols is None:
        raise ValueError("driver_cols must not be None.")

    driver_cols = list(driver_cols)
    N = len(df)

    if len(driver_cols) == 0:
        return np.zeros((N, 0), dtype=np.float32)

    if nan_policy not in {"zero", "error"}:
        raise ValueError(f"Unsupported nan_policy={nan_policy}. Expected 'zero' or 'error'.")

    if len(set(driver_cols)) != len(driver_cols):
        seen = set()
        dupes = []
        for c in driver_cols:
            if c in seen:
                dupes.append(c)
            seen.add(c)
        raise RuntimeError(
            f"[build_iot_driver_matrix_from_df] Duplicate driver columns detected: {sorted(set(dupes))[:20]}"
        )

    missing = [c for c in driver_cols if c not in df.columns]

    if missing and require_all_driver_cols:
        raise RuntimeError(
            f"[build_iot_driver_matrix_from_df] Missing required driver columns ({len(missing)}): "
            f"{missing[:20]}" + (" ..." if len(missing) > 20 else "")
        )

    mats = []

    for c in driver_cols:
        if c not in df.columns:
            # Only reachable when require_all_driver_cols=False.
            v = np.zeros(N, dtype=np.float32)
            mats.append(v)
            continue

        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        v = v.copy()
        finite = np.isfinite(v)

        if not finite.all():
            n_bad = int((~finite).sum())
            if nan_policy == "error":
                raise RuntimeError(
                    f"[build_iot_driver_matrix_from_df] Driver column '{c}' contains {n_bad} NaN/non-finite rows."
                )
            v[~finite] = 0.0

        if clip_negative_to_zero:
            v[v < 0.0] = 0.0

        mats.append(v.astype(np.float32, copy=False))

    out = np.stack(mats, axis=1).astype(np.float32, copy=False)

    if out.shape != (N, len(driver_cols)):
        raise RuntimeError(
            f"[build_iot_driver_matrix_from_df] Unexpected shape {out.shape}; expected {(N, len(driver_cols))}"
        )

    return out


# ------------------------------------------------------------
# Helper: list protected columns
# ------------------------------------------------------------
def list_protected_constant_drop_columns(cols: Sequence[str]) -> List[str]:
    return [c for c in cols if is_protected_constant_drop_column(c)]


# ------------------------------------------------------------
# Helper: strict feature-frame validation before modeling
# ------------------------------------------------------------
def assert_no_modeling_forbidden_columns(
    df: pd.DataFrame,
    *,
    where: str = "modeling_frame",
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
) -> None:
    assert_no_duplicate_columns(df, where)

    bad = [
        c for c in df.columns
        if is_modeling_forbidden_column(
            c,
            raw_time_col=raw_time_col,
            canonical_time_col=canonical_time_col,
        )
    ]

    if bad:
        raise RuntimeError(
            f"[{where}] Modeling-forbidden columns detected ({len(bad)}): {bad[:20]}"
        )


def assert_no_abs_time_leakage_columns(
    df: pd.DataFrame,
    *,
    where: str = "modeling_frame",
    raw_time_col: Optional[str] = None,
    canonical_time_col: Optional[str] = None,
    include_canonical: bool = True,
) -> None:
    assert_no_duplicate_columns(df, where)

    bad = collect_abs_time_features(
        list(df.columns),
        raw_time_col=raw_time_col,
        canonical_time_col=canonical_time_col,
        include_canonical=include_canonical,
    )

    if bad:
        raise RuntimeError(
            f"[{where}] Absolute-time leakage columns detected ({len(bad)}): {bad[:20]}"
        )


_log_msg("[Cell2] Core helpers loaded — v5 | time canonicalization + leakage guards active.")