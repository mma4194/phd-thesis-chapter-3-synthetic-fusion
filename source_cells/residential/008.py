# ============================================================
# CELL 4.4 — Clean taxonomy split (DRIVER vs MASK vs PROTO_VALUE) — v6
#
# Purpose:
# - freeze a clean separation between:
#     1) IoT driver columns
#     2) protocol observability/mask/meta columns
#     3) protocol core value targets
#     4) protocol derived-value columns
#     5) physical-only protocol metadata columns
#
# Critical correction:
# - protocol DDPM value targets must come from LOGICAL tiers only:
#       router__, ota__, zigbee__
# - physical OTA split namespaces (ota24__, ota5__) are kept as protocol
#   meta / conditioning / diagnostics, NOT as DDPM value targets
# - derived protocol values are explicitly tracked:
#       if MODEL_DERIVED_VALUES=False:
#           they are excluded from DDPM value targets and recorded as
#           PROTO_DERIVED_EXCLUDED_COLS
#       if MODEL_DERIVED_VALUES=True:
#           they are added to PROTO_VALUE_COLS
#
# Strict policy:
# - DRIVER_COLS must come from Cell 4 contract only
# - required protocol obs masks must come from Cell 4 contract only
# - no recovery-by-regex path
# - no non-protocol columns in logical protocol value space
# - no physical-subband columns in logical DDPM target space
# - no unaccounted protocol numeric columns
#
# Hardening in v6:
# - validates PROTOCOL_TIERS_USED as ordered subset of canonical tier order
# - validates protocol namespace against Cell 3 registry
# - treats derived columns as an explicit excluded/modelled category
# - validates full protocol taxonomy closure
# - writes taxonomy_split_manifest.json with full registry counts
# ============================================================

log("--- START: Cell 4.4 — taxonomy split (drivers / masks / values) ---")

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Sequence

need = [
    "df_tr", "df_va", "df_te",
    "CFG", "log",
    "NUMERIC_COLS_PROTO", "time_col",
    "IOT_DRIVER_COLS",
    "PROTOCOL_TIERS_USED",
    "PROTOCOL_OBS_COLS_REQUIRED",
    "PROTO_NAMESPACE_COLS",
    "AUX_META_COLS",
]
missing = [k for k in need if k not in globals()]
if missing:
    raise RuntimeError(f"[Cell4.4] Missing prerequisites. Run Cells 1–4 first. Missing={missing}")

CANON_TIME_COL = str(globals()["time_col"])
ARTDIR = os.path.join(str(CFG["outdir"]), "artifacts")
os.makedirs(ARTDIR, exist_ok=True)

# ------------------------------------------------------------
# 0) Canonical tier definitions / validators
# ------------------------------------------------------------
TIERS_CANONICAL = list(globals().get("TIERS", ["router", "ota", "zigbee", "zwave"]))
LOGICAL_VALUE_TIERS_DEFAULT = ["router", "ota", "zigbee"]

# Physical namespaces are useful as masks/meta/diagnostics but are not logical
# protocol DDPM value targets in this notebook line.
PHYSICAL_ONLY_META_PREFIXES = ("ota24__", "ota5__", "zwave__")

PROTOCOL_NAMESPACE_PREFIXES = (
    "router__",
    "ota__",
    "ota24__",
    "ota5__",
    "zigbee__",
    "zwave__",
)

def _assert_tiers_ordered_subset_local(tiers: Sequence[str], *, label: str) -> None:
    t = list(tiers)

    if len(t) != len(set(t)):
        raise RuntimeError(f"[Cell4.4] Duplicate tiers not allowed in {label}: {t}")

    for x in t:
        if x not in TIERS_CANONICAL:
            raise RuntimeError(
                f"[Cell4.4] Unknown tier '{x}' in {label}. Allowed={TIERS_CANONICAL}"
            )

    idx = [TIERS_CANONICAL.index(x) for x in t]
    if idx != sorted(idx):
        raise RuntimeError(
            f"[Cell4.4] Tier order mismatch in {label}. "
            f"Expected subset order of {TIERS_CANONICAL}, got {t}"
        )

def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

def _ordered_unique(xs: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(list(xs)))

def _require_cols_in_splits(cols: Sequence[str], *, label: str) -> None:
    cols = list(cols)
    for nm, _df in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
        miss = [c for c in cols if c not in _df.columns]
        if miss:
            raise RuntimeError(
                f"[Cell4.4] Split schema drift for {label} in {nm}: missing={miss[:30]}"
            )

def _assert_disjoint(a: Sequence[str], b: Sequence[str], *, name_a: str, name_b: str) -> None:
    inter = sorted(set(a).intersection(set(b)))
    if inter:
        raise RuntimeError(
            f"[Cell4.4] {name_a} ∩ {name_b} is not empty ({len(inter)}): {inter[:30]}"
        )

def _is_numeric_in_train(c: str) -> bool:
    return c in df_tr.columns and pd.api.types.is_numeric_dtype(df_tr[c])

# ------------------------------------------------------------
# 1) Resolve DRIVER_COLS from strict source of truth
# ------------------------------------------------------------
DRIVER_COLS = list(globals().get("IOT_DRIVER_COLS") or [])

if not DRIVER_COLS:
    raise RuntimeError("[Cell4.4] IOT_DRIVER_COLS is empty. Do not recover silently; fix Cell 4.")

if len(DRIVER_COLS) != len(set(DRIVER_COLS)):
    raise RuntimeError("[Cell4.4] DRIVER_COLS contains duplicates.")

bad_driver_names = [
    c for c in DRIVER_COLS
    if not (isinstance(c, str) and c.startswith("events_in_sec__"))
]
if bad_driver_names:
    raise RuntimeError(
        f"[Cell4.4] DRIVER_COLS contains non-events_in_sec entries: {bad_driver_names[:20]}"
    )

_require_cols_in_splits(DRIVER_COLS, label="DRIVER_COLS")

globals()["DRIVER_COLS"] = list(DRIVER_COLS)
log(f"[Cell4.4] DRIVER_COLS count={len(DRIVER_COLS)} | first 20: {DRIVER_COLS[:20]}")

# ------------------------------------------------------------
# 2) Validate runtime tier contract and freeze MASK / META registries
# ------------------------------------------------------------
PROTOCOL_TIERS_USED_LOCAL = list(globals().get("PROTOCOL_TIERS_USED") or [])

if not PROTOCOL_TIERS_USED_LOCAL:
    raise RuntimeError("[Cell4.4] PROTOCOL_TIERS_USED is empty. Do not infer runtime tiers here.")

_assert_tiers_ordered_subset_local(PROTOCOL_TIERS_USED_LOCAL, label="PROTOCOL_TIERS_USED")

MASK_COLS_TIER = list(globals().get("PROTOCOL_OBS_COLS_REQUIRED") or [])

if not MASK_COLS_TIER:
    raise RuntimeError("[Cell4.4] PROTOCOL_OBS_COLS_REQUIRED is empty. Do not infer masks here.")

bad_mask_names = [
    c for c in MASK_COLS_TIER
    if not (isinstance(c, str) and c.lower().endswith("__obs_present"))
]
if bad_mask_names:
    raise RuntimeError(
        f"[Cell4.4] PROTOCOL_OBS_COLS_REQUIRED contains non-obs_present entries: {bad_mask_names[:20]}"
    )

_require_cols_in_splits(MASK_COLS_TIER, label="MASK_COLS_TIER")

CANONICAL_PROTOCOL_META_RX = re.compile(
    r"(__traffic_present$|__traffic_obs_present$|__staleness_s$|__stale_flag$|__present$)",
    re.IGNORECASE,
)

PROTO_NAMESPACE_SET = set(globals().get("PROTO_NAMESPACE_COLS") or [])

if not PROTO_NAMESPACE_SET:
    raise RuntimeError("[Cell4.4] PROTO_NAMESPACE_COLS is empty. Cell 3 registry is missing.")

def _is_protocol_namespace_col(c: str) -> bool:
    return (
        isinstance(c, str)
        and c in PROTO_NAMESPACE_SET
        and c.startswith(PROTOCOL_NAMESPACE_PREFIXES)
    )

def _is_protocol_meta_non_tier(c: str) -> bool:
    if c in MASK_COLS_TIER:
        return False
    if not _is_protocol_namespace_col(c):
        return False
    return bool(CANONICAL_PROTOCOL_META_RX.search(c))

MASK_COLS_SUBLAYER = sorted([
    c for c in df_tr.columns
    if _is_protocol_meta_non_tier(c)
])

MASK_COLS_ALL = sorted(set(MASK_COLS_TIER + MASK_COLS_SUBLAYER))

# Explicit cross-modal meta from Cell 3.
PROTO_AUX_META_COLS = sorted(list(globals().get("AUX_META_COLS") or []))

_require_cols_in_splits(MASK_COLS_ALL, label="MASK_COLS_ALL")
_require_cols_in_splits(PROTO_AUX_META_COLS, label="PROTO_AUX_META_COLS")

globals()["MASK_COLS_TIER"] = list(MASK_COLS_TIER)
globals()["MASK_COLS_SUBLAYER"] = list(MASK_COLS_SUBLAYER)
globals()["MASK_COLS_ALL"] = list(MASK_COLS_ALL)
globals()["AUX_MASK_COLS"] = []   # legacy alias kept empty intentionally
globals()["PROTO_AUX_META_COLS"] = list(PROTO_AUX_META_COLS)

log(f"[Cell4.4] PROTOCOL_TIERS_USED={PROTOCOL_TIERS_USED_LOCAL}")
log(f"[Cell4.4] MASK_COLS_TIER count={len(MASK_COLS_TIER)} | first 20: {MASK_COLS_TIER[:20]}")
log(f"[Cell4.4] MASK_COLS_SUBLAYER count={len(MASK_COLS_SUBLAYER)} | first 20: {MASK_COLS_SUBLAYER[:20]}")
log(f"[Cell4.4] MASK_COLS_ALL count={len(MASK_COLS_ALL)}")
log(f"[Cell4.4] PROTO_AUX_META_COLS count={len(PROTO_AUX_META_COLS)} | first 20: {PROTO_AUX_META_COLS[:20]}")

# ------------------------------------------------------------
# 3) Logical vs physical protocol namespaces
# ------------------------------------------------------------
expected_logical_value_tiers = CFG.get(
    "expected_protocol_value_tiers",
    LOGICAL_VALUE_TIERS_DEFAULT,
)
expected_logical_value_tiers = [
    str(x).strip().lower()
    for x in list(expected_logical_value_tiers)
]

_assert_tiers_ordered_subset_local(expected_logical_value_tiers, label="expected_protocol_value_tiers")

logical_value_tiers = [
    t for t in PROTOCOL_TIERS_USED_LOCAL
    if t in LOGICAL_VALUE_TIERS_DEFAULT
]

if not logical_value_tiers:
    raise RuntimeError(
        f"[Cell4.4] No logical value tiers remain after restriction to {LOGICAL_VALUE_TIERS_DEFAULT}. "
        f"PROTOCOL_TIERS_USED={PROTOCOL_TIERS_USED_LOCAL}"
    )

if logical_value_tiers != expected_logical_value_tiers:
    raise RuntimeError(
        f"[Cell4.4] logical_value_tiers mismatch: got {logical_value_tiers} "
        f"expected {expected_logical_value_tiers}"
    )

LOGICAL_PROTOCOL_PREFIXES = tuple(f"{t}__" for t in logical_value_tiers)

def _is_logical_protocol_candidate(c: str) -> bool:
    return (
        isinstance(c, str)
        and c in PROTO_NAMESPACE_SET
        and c.startswith(LOGICAL_PROTOCOL_PREFIXES)
    )

def _is_physical_only_protocol_meta(c: str) -> bool:
    return (
        isinstance(c, str)
        and c in PROTO_NAMESPACE_SET
        and c.startswith(PHYSICAL_ONLY_META_PREFIXES)
    )

# Broad protocol pool from Cell 3 after TRAIN-only protocol constant dropping.
PROTO_CANDIDATES_BROAD = [
    c for c in list(NUMERIC_COLS_PROTO)
    if c in df_tr.columns
]

if not PROTO_CANDIDATES_BROAD:
    raise RuntimeError("[Cell4.4] NUMERIC_COLS_PROTO is empty or inconsistent with TRAIN dataframe.")

bad_proto_candidates = [
    c for c in PROTO_CANDIDATES_BROAD
    if not _is_protocol_namespace_col(c)
]
if bad_proto_candidates:
    raise RuntimeError(
        f"[Cell4.4] NUMERIC_COLS_PROTO contains non-protocol namespace columns: {bad_proto_candidates[:30]}"
    )

PROTO_CANDIDATES_LOGICAL = [
    c for c in PROTO_CANDIDATES_BROAD
    if _is_logical_protocol_candidate(c)
]

PROTO_PHYSICAL_META_ONLY = sorted([
    c for c in PROTO_CANDIDATES_BROAD
    if _is_physical_only_protocol_meta(c)
])

if not PROTO_CANDIDATES_LOGICAL:
    raise RuntimeError("[Cell4.4] No logical protocol candidates remain after namespace restriction.")

log(f"[Cell4.4] logical_value_tiers={logical_value_tiers}")
log(f"[Cell4.4] PROTO_CANDIDATES_BROAD count={len(PROTO_CANDIDATES_BROAD)}")
log(f"[Cell4.4] PROTO_CANDIDATES_LOGICAL count={len(PROTO_CANDIDATES_LOGICAL)}")
log(
    f"[Cell4.4] PROTO_PHYSICAL_META_ONLY count={len(PROTO_PHYSICAL_META_ONLY)} | "
    f"first 30: {PROTO_PHYSICAL_META_ONLY[:30]}"
)

# ------------------------------------------------------------
# 4) Split logical protocol columns into META vs CORE_VALUE vs DERIVED
# ------------------------------------------------------------
DERIVED_RX = re.compile(
    r"("
    r"__roll\d+_(mean|std|z)$|"
    r"__diff1$|__absdiff1$|"
    r"_rate$|"
    r"_share$|"
    r"_ratio$|"
    r"_per_[a-z0-9_]+$|"
    r"entropy(_norm)?$|"
    r"avg_pkt_bytes$|"
    r"(?:^|_)(minus|delta|imbalance|burst)(?:$|_)"
    r")",
    re.IGNORECASE,
)

PROTO_VALUE_COLS_CORE: List[str] = []
PROTO_VALUE_COLS_DERIVED: List[str] = []
PROTO_META_COLS: List[str] = []

# Physical-only protocol columns are meta/conditioning/diagnostics only.
PROTO_META_COLS.extend(PROTO_PHYSICAL_META_ONLY)

for c in PROTO_CANDIDATES_LOGICAL:
    if c in DRIVER_COLS:
        PROTO_META_COLS.append(c)
        continue

    if c in MASK_COLS_ALL:
        PROTO_META_COLS.append(c)
        continue

    if c in PROTO_AUX_META_COLS:
        PROTO_META_COLS.append(c)
        continue

    if DERIVED_RX.search(c):
        PROTO_VALUE_COLS_DERIVED.append(c)
    else:
        PROTO_VALUE_COLS_CORE.append(c)

# Keep masks and explicit aux meta inside the protocol meta registry.
PROTO_META_COLS.extend(MASK_COLS_ALL)
PROTO_META_COLS.extend(PROTO_AUX_META_COLS)

PROTO_VALUE_COLS_CORE = _ordered_unique(PROTO_VALUE_COLS_CORE)
PROTO_VALUE_COLS_DERIVED = _ordered_unique(PROTO_VALUE_COLS_DERIVED)
PROTO_META_COLS = _ordered_unique(PROTO_META_COLS)

MODEL_DERIVED_VALUES = bool(CFG.get("model_derived_protocol_values", False))

if MODEL_DERIVED_VALUES:
    PROTO_VALUE_COLS = _ordered_unique(PROTO_VALUE_COLS_CORE + PROTO_VALUE_COLS_DERIVED)
    PROTO_DERIVED_EXCLUDED_COLS: List[str] = []
else:
    PROTO_VALUE_COLS = list(PROTO_VALUE_COLS_CORE)
    PROTO_DERIVED_EXCLUDED_COLS = list(PROTO_VALUE_COLS_DERIVED)

# These are all logical protocol columns that are not modeled as value targets
# and should not silently enter DDPM value space.
PROTO_NONVALUE_LOGICAL_COLS = _ordered_unique([
    c for c in PROTO_CANDIDATES_LOGICAL
    if c not in set(PROTO_VALUE_COLS)
])

PROTO_ALL_TARGETABLE_COLS = _ordered_unique(PROTO_VALUE_COLS_CORE + PROTO_VALUE_COLS_DERIVED)

globals()["PROTO_VALUE_COLS_CORE"] = list(PROTO_VALUE_COLS_CORE)
globals()["PROTO_VALUE_COLS_DERIVED"] = list(PROTO_VALUE_COLS_DERIVED)
globals()["PROTO_VALUE_COLS"] = list(PROTO_VALUE_COLS)
globals()["PROTO_META_COLS"] = list(PROTO_META_COLS)
globals()["PROTO_DERIVED_EXCLUDED_COLS"] = list(PROTO_DERIVED_EXCLUDED_COLS)
globals()["PROTO_NONVALUE_LOGICAL_COLS"] = list(PROTO_NONVALUE_LOGICAL_COLS)
globals()["PROTO_ALL_TARGETABLE_COLS"] = list(PROTO_ALL_TARGETABLE_COLS)
globals()["PROTO_PHYSICAL_META_ONLY"] = list(PROTO_PHYSICAL_META_ONLY)

log(f"[Cell4.4] MODEL_DERIVED_VALUES={MODEL_DERIVED_VALUES}")
log(f"[Cell4.4] PROTO_VALUE_COLS_CORE count={len(PROTO_VALUE_COLS_CORE)} | first 30: {PROTO_VALUE_COLS_CORE[:30]}")
log(f"[Cell4.4] PROTO_VALUE_COLS_DERIVED count={len(PROTO_VALUE_COLS_DERIVED)} | first 30: {PROTO_VALUE_COLS_DERIVED[:30]}")
log(f"[Cell4.4] PROTO_DERIVED_EXCLUDED_COLS count={len(PROTO_DERIVED_EXCLUDED_COLS)}")
log(f"[Cell4.4] PROTO_VALUE_COLS count={len(PROTO_VALUE_COLS)}")
log(f"[Cell4.4] PROTO_META_COLS count={len(PROTO_META_COLS)} | first 30: {PROTO_META_COLS[:30]}")

if not PROTO_VALUE_COLS_CORE:
    raise RuntimeError("[Cell4.4] No core protocol value columns remain after taxonomy split.")

# ------------------------------------------------------------
# 5) Sanity checks: disjointness, leakage, schema stability
# ------------------------------------------------------------
_assert_disjoint(DRIVER_COLS, MASK_COLS_ALL, name_a="DRIVER", name_b="MASK")
_assert_disjoint(DRIVER_COLS, PROTO_VALUE_COLS, name_a="DRIVER", name_b="PROTO_VALUE")
_assert_disjoint(MASK_COLS_ALL, PROTO_VALUE_COLS, name_a="MASK", name_b="PROTO_VALUE")
_assert_disjoint(PROTO_META_COLS, PROTO_VALUE_COLS, name_a="PROTO_META", name_b="PROTO_VALUE")
_assert_disjoint(PROTO_AUX_META_COLS, PROTO_VALUE_COLS, name_a="AUX_META", name_b="PROTO_VALUE")
_assert_disjoint(PROTO_PHYSICAL_META_ONLY, PROTO_VALUE_COLS, name_a="PHYSICAL_META_ONLY", name_b="PROTO_VALUE")
_assert_disjoint(PROTO_DERIVED_EXCLUDED_COLS, PROTO_VALUE_COLS, name_a="DERIVED_EXCLUDED", name_b="PROTO_VALUE")

for nm, _df in [("TRAIN", df_tr), ("VAL", df_va), ("TEST", df_te)]:
    miss_vals = [c for c in PROTO_VALUE_COLS if c not in _df.columns]
    miss_core = [c for c in PROTO_VALUE_COLS_CORE if c not in _df.columns]
    miss_derived = [c for c in PROTO_VALUE_COLS_DERIVED if c not in _df.columns]
    miss_meta = [c for c in PROTO_META_COLS if c not in _df.columns]

    if miss_vals or miss_core or miss_derived or miss_meta:
        raise RuntimeError(
            f"[Cell4.4] Split schema drift in {nm}. "
            f"missing_values={miss_vals[:20]} "
            f"missing_core={miss_core[:20]} "
            f"missing_derived={miss_derived[:20]} "
            f"missing_meta={miss_meta[:20]}"
        )

# Hard leak checks.
BAD_OBS_IN_VALUES = [
    "router__obs_present",
    "ota__obs_present",
    "ota24__obs_present",
    "ota5__obs_present",
    "zigbee__obs_present",
    "zwave__obs_present",
]
for bad in BAD_OBS_IN_VALUES:
    if bad in PROTO_VALUE_COLS:
        raise RuntimeError(f"[Cell4.4] Required observability column leaked into value targets: {bad}")

if any(c.startswith("iot__") for c in PROTO_VALUE_COLS):
    raise RuntimeError("[Cell4.4] IoT columns leaked into protocol value targets.")

if any(c.startswith("events_in_sec__") for c in PROTO_VALUE_COLS):
    raise RuntimeError("[Cell4.4] Driver columns leaked into protocol value targets.")

if any(c.startswith(PHYSICAL_ONLY_META_PREFIXES) for c in PROTO_VALUE_COLS):
    raise RuntimeError("[Cell4.4] Physical-only protocol namespaces leaked into logical DDPM value targets.")

if any(c == CANON_TIME_COL for c in PROTO_VALUE_COLS):
    raise RuntimeError("[Cell4.4] Canonical time leaked into protocol value targets.")

if any(c.startswith("PROVENANCE__") for c in PROTO_VALUE_COLS):
    raise RuntimeError("[Cell4.4] Provenance column leaked into protocol value targets.")

# Contract expectation for current validated project state.
expected_core = int(CFG.get("expected_proto_value_cols_core", 54))
if expected_core > 0 and len(PROTO_VALUE_COLS_CORE) != expected_core:
    raise RuntimeError(
        f"[Cell4.4] PROTO_VALUE_COLS_CORE count mismatch: "
        f"got {len(PROTO_VALUE_COLS_CORE)} expected {expected_core}."
    )

# Full protocol taxonomy closure.
# Every numeric protocol column from Cell 3 must be accounted for as:
# - modeled core value
# - modeled/excluded derived value
# - meta/mask/physical-only meta
accounted_protocol = set(PROTO_VALUE_COLS_CORE) | set(PROTO_VALUE_COLS_DERIVED) | set(PROTO_META_COLS)
unaccounted_protocol = sorted(set(PROTO_CANDIDATES_BROAD) - accounted_protocol)
unknown_accounted = sorted(accounted_protocol - set(PROTO_CANDIDATES_BROAD) - set(PROTO_AUX_META_COLS))

if unaccounted_protocol:
    raise RuntimeError(
        f"[Cell4.4] Unaccounted protocol numeric columns detected ({len(unaccounted_protocol)}): "
        f"{unaccounted_protocol[:50]}"
    )

# PROTO_AUX_META_COLS may be cross-modal auxiliary metadata and not part of NUMERIC_COLS_PROTO.
# Therefore unknown_accounted only errors if non-aux unknowns exist.
if unknown_accounted:
    raise RuntimeError(
        f"[Cell4.4] Taxonomy includes columns outside protocol pool unexpectedly: "
        f"{unknown_accounted[:50]}"
    )

# Numeric check for value targets.
non_numeric_values = [c for c in PROTO_VALUE_COLS if not _is_numeric_in_train(c)]
if non_numeric_values:
    raise RuntimeError(
        f"[Cell4.4] Non-numeric protocol value targets detected: {non_numeric_values[:30]}"
    )

globals()["PROTO_VALUE_TIER_PREFIXES"] = list(logical_value_tiers)

taxonomy_manifest = {
    "version": "taxonomy_split_v6_closed_protocol_partition",
    "tiers_superset": list(TIERS_CANONICAL),
    "protocol_tiers_used_runtime": list(PROTOCOL_TIERS_USED_LOCAL),
    "logical_value_tiers": list(logical_value_tiers),
    "protocol_obs_cols_required": list(MASK_COLS_TIER),
    "n_driver_cols": int(len(DRIVER_COLS)),
    "n_mask_cols_tier": int(len(MASK_COLS_TIER)),
    "n_mask_cols_sublayer": int(len(MASK_COLS_SUBLAYER)),
    "n_mask_cols_all": int(len(MASK_COLS_ALL)),
    "n_proto_aux_meta_cols": int(len(PROTO_AUX_META_COLS)),
    "n_proto_candidates_broad": int(len(PROTO_CANDIDATES_BROAD)),
    "n_proto_candidates_logical": int(len(PROTO_CANDIDATES_LOGICAL)),
    "n_proto_physical_meta_only": int(len(PROTO_PHYSICAL_META_ONLY)),
    "n_proto_value_cols_core": int(len(PROTO_VALUE_COLS_CORE)),
    "n_proto_value_cols_derived": int(len(PROTO_VALUE_COLS_DERIVED)),
    "n_proto_derived_excluded_cols": int(len(PROTO_DERIVED_EXCLUDED_COLS)),
    "n_proto_value_cols": int(len(PROTO_VALUE_COLS)),
    "n_proto_meta_cols": int(len(PROTO_META_COLS)),
    "n_proto_nonvalue_logical_cols": int(len(PROTO_NONVALUE_LOGICAL_COLS)),
    "model_derived_values": bool(MODEL_DERIVED_VALUES),
    "expected_proto_value_cols_core": int(expected_core),
    "expected_protocol_value_tiers": list(expected_logical_value_tiers),
    "physical_only_meta_prefixes": list(PHYSICAL_ONLY_META_PREFIXES),
    "proto_value_cols_core": list(PROTO_VALUE_COLS_CORE),
    "proto_value_cols_derived": list(PROTO_VALUE_COLS_DERIVED),
    "proto_derived_excluded_cols": list(PROTO_DERIVED_EXCLUDED_COLS),
    "proto_value_cols": list(PROTO_VALUE_COLS),
    "proto_meta_cols": list(PROTO_META_COLS),
    "proto_physical_meta_only": list(PROTO_PHYSICAL_META_ONLY),
    "mask_cols_all": list(MASK_COLS_ALL),
    "driver_cols": list(DRIVER_COLS),
}

taxonomy_manifest_path = os.path.join(ARTDIR, "taxonomy_split_manifest.json")
_write_json(taxonomy_manifest_path, taxonomy_manifest)

globals()["TAXONOMY_MANIFEST"] = taxonomy_manifest

log(f"[Cell4.4] Wrote taxonomy manifest: {taxonomy_manifest_path}")
log(
    "[Cell4.4] Sanity OK: closed protocol taxonomy; "
    "DRIVER, MASK, META, PHYSICAL_META_ONLY, DERIVED_EXCLUDED, and PROTO_VALUE are controlled."
)
log("--- END:   Cell 4.4 — taxonomy split (drivers / masks / values) ---")