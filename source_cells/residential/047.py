# ==========================================================
# CELL 12.b — IoT synthesis scaffold stage — v27.1-THESIS
# (STUDY-THESIS PORTFOLIO-NEUTRAL / CONTRACT-LOCKED / NO DDPM TRAINING / NO TEST FITTING)
#
# Purpose:
#   Build the shared IoT scaffold consumed by the modular Cell 12.c
#   portfolio:
#     1) global / explicit IoT observation masks
#     2) canonical entity maps and entity activity
#     3) entity regimes
#     4) binary IoT states
#     5) continuous-value availability masks
#     6) portfolio eligibility metadata
#
# Scientific boundary:
#   - TRAIN+VAL are used to fit scaffold models.
#   - TEST is used only for:
#       * output length N_TE
#       * output index alignment
#       * schema checks
#   - TEST values are never used to fit rates, distributions, regimes,
#     binary states, availability masks, or generator choices.
#
# Important architectural change:
#   - This cell does NOT train DDPM.
#   - This cell does NOT sample DDPM.
#   - This cell does NOT generate continuous values.
#   - Continuous-value generation is delegated to modular Cell 12.c:
#       12.c.0 shared utilities/metrics
#       12.c.1 A1 value baseline
#       12.c.2 VAL candidates
#       12.c.3 VAL selector
#       12.c.4 TEST-length materialization
#       12.c.5 final contract enforcement
#       12.c.6 final QA/publication
# ==========================================================

log("--- START: Cell 12.b — IoT synthesis scaffold stage (v27.1.1-THESIS contract-locked, portfolio-neutral, no DDPM training) ---")

# ----------------------------------------------------------
# Imports
# ----------------------------------------------------------
import os
import re
import gc
import json
import math
import copy
import hashlib
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

try:
    import torch
except Exception as _torch_exc:
    raise RuntimeError(f"[Cell12.b] PyTorch is required by the notebook runtime but unavailable: {_torch_exc}")

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------------------------------------
# Required globals from Cell 12.a
# ----------------------------------------------------------
required_12b = [
    "CFG", "log",
    "df_tr", "df_te",
    "SEED", "rng", "device", "CFG12",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "DRV_COLS", "BIN_COLS", "CONT_COLS", "CONT_MASK_COLS", "CONT_VALUE_COLS",
    "X_drv_tr", "X_bin_tr", "X_cont_tr",
    "X_drv_val", "X_bin_val", "X_cont_val",
    "X_drv_te", "X_bin_te", "X_cont_te",
    "N_TR", "N_VAL", "N_TE",
    "ENTITY_MAP", "ENTITY_TO_DRV", "ENTITY_TO_BIN", "ENTITY_TO_CONT",
    "FAMILY_MAP", "FAMILY_GROUPS",
    "DDPM_COLS", "NON_DDPM_COLS",
    "DEVICE_TELEMETRY_COLS", "CUMULATIVE_COUNT_COLS", "STEP_PROGRESS_COLS",
    "IOT_SYN_DRIVERS", "base_offset",
    "_infer_entity_from_col",
]

if "df_val" in globals() and isinstance(globals()["df_val"], pd.DataFrame):
    df_val = globals()["df_val"]
elif "df_va" in globals() and isinstance(globals()["df_va"], pd.DataFrame):
    df_val = globals()["df_va"]
else:
    raise RuntimeError("[Cell12.b] Missing canonical VAL dataframe: expected df_val or df_va.")

missing_12b = [k for k in required_12b if k not in globals()]
if missing_12b:
    raise RuntimeError(f"[Cell12.b] Missing required globals from Cell 12.a: {missing_12b}")

if int(len(df_tr)) != int(N_TR):
    raise RuntimeError(f"[Cell12.b] N_TR mismatch: len(df_tr)={len(df_tr)} vs N_TR={N_TR}")
if int(len(df_val)) != int(N_VAL):
    raise RuntimeError(f"[Cell12.b] N_VAL mismatch: len(df_val)={len(df_val)} vs N_VAL={N_VAL}")
if int(len(df_te)) != int(N_TE):
    raise RuntimeError(f"[Cell12.b] N_TE mismatch: len(df_te)={len(df_te)} vs N_TE={N_TE}")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
ART_DIR = os.path.join(str(OUTDIR), "artifacts")
os.makedirs(ART_DIR, exist_ok=True)
CONTRACT_DIR = os.path.join(ART_DIR, "contracts")
os.makedirs(CONTRACT_DIR, exist_ok=True)

# ----------------------------------------------------------
# v27.1 upstream contract locks
# ----------------------------------------------------------
def _read_json_dict_required_12b(path: str, label: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"[Cell12.b] Missing required {label}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell12.b] Required {label} must be a JSON object: {path}")
    return obj

CELL12A_FOUNDATION_MANIFEST_PATH = os.path.join(ART_DIR, "cell12a_foundation_manifest_v24_0.json")
CELL10_8_SELECTOR_CONTRACT_PATH = os.path.join(CONTRACT_DIR, "cell10_8_val_only_selector_contract_v3_6_THESIS.json")
CELL10_9_MATERIALIZER_CONTRACT_PATH = os.path.join(CONTRACT_DIR, "cell10_9_protocol_variant_materialization_contract_v2_1_THESIS.json")
PRECELL11_RUNTIME_CONTRACT_PATH = os.path.join(CONTRACT_DIR, "pre_cell11_a0_a1_a2_variant_runtime_contract_v4_1_THESIS.json")
CELL11_FINAL_MANIFEST_PATH_REQUIRED = os.path.join(ART_DIR, "cell11_final_protocol_variant_manifest.json")
CELL11B_QUARANTINE_MANIFEST_PATH = os.path.join(ART_DIR, "cell11b_protocol_test_regression_quarantine_manifest.json")

CELL12A_FOUNDATION_MANIFEST = _read_json_dict_required_12b(
    CELL12A_FOUNDATION_MANIFEST_PATH,
    "Cell 12.a foundation manifest",
)
CELL10_8_SELECTOR_CONTRACT = _read_json_dict_required_12b(
    CELL10_8_SELECTOR_CONTRACT_PATH,
    "Cell 10.8 v3.6 selector contract",
)
CELL10_9_MATERIALIZER_CONTRACT = _read_json_dict_required_12b(
    CELL10_9_MATERIALIZER_CONTRACT_PATH,
    "Cell 10.9 v2.1 materializer contract",
)
PRECELL11_RUNTIME_CONTRACT = _read_json_dict_required_12b(
    PRECELL11_RUNTIME_CONTRACT_PATH,
    "PRE-CELL11 v4.1 runtime contract",
)
CELL11_FINAL_MANIFEST_REQUIRED = _read_json_dict_required_12b(
    CELL11_FINAL_MANIFEST_PATH_REQUIRED,
    "Cell 11 final protocol manifest",
)

# Fail closed if noncanonical Cell 11b quarantine artifacts remain active.
if os.path.exists(CELL11B_QUARANTINE_MANIFEST_PATH):
    _cell11b_manifest = _read_json_dict_required_12b(
        CELL11B_QUARANTINE_MANIFEST_PATH,
        "noncanonical Cell 11b quarantine manifest",
    )
    _cell11b_outputs = _cell11b_manifest.get("outputs", {}) if isinstance(_cell11b_manifest, dict) else {}
    _cell11b_policy = _cell11b_manifest.get("policy", {}) if isinstance(_cell11b_manifest, dict) else {}
    _cell11b_overwrote = bool(
        _cell11b_policy.get("canonical_A2_overwritten", False)
        or _cell11b_outputs.get("A2_PROTOCOL_FINAL_overwritten")
        or _cell11b_outputs.get("A2_PROTOCOL_TEST_overwritten")
    )
    if _cell11b_overwrote:
        raise RuntimeError(
            "[Cell12.b] Refusing to proceed because noncanonical Cell 11b TEST-quarantine "
            "artifacts indicate canonical A2 was overwritten. Archive Cell 11b artifacts and "
            "rerun the canonical Cell 10.8 -> 10.9 -> PRE-CELL11 -> Cell 11 path."
        )

# Enforce pure VAL-locked protocol lineage.
for _key in [
    "test_values_read",
    "test_used_for_fitting",
    "test_used_for_thresholding",
    "test_used_for_selection",
    "test_rescue_materialization",
    "in_selector_candidate_generation",
    "a0_used_for_selection",
    "unregistered_file_scan",
]:
    if bool(CELL10_8_SELECTOR_CONTRACT.get(_key, False)):
        raise RuntimeError(f"[Cell12.b] Cell 10.8 selector contract violation: {_key}=True")

for _key in [
    "test_values_read",
    "test_metrics_computed",
    "test_used_for_fitting",
    "test_used_for_thresholding",
    "test_used_for_selection",
    "test_used_for_repair",
    "a0_used_for_selection",
    "downstream_test_rescue_materialization",
]:
    if bool(CELL10_9_MATERIALIZER_CONTRACT.get(_key, False)):
        raise RuntimeError(f"[Cell12.b] Cell 10.9 materializer contract violation: {_key}=True")

# PRE-CELL11 v4.1 hard prohibitions must be explicit in the JSON contract.
for _key in [
    "cell11_must_not_select_generators",
    "cell11_must_not_register_ddpm_for_protocol_generation",
    "cell11_must_not_materialize_protocol_candidates",
    "cell11_must_not_apply_test_informed_rescue_or_repair",
]:
    if not bool(PRECELL11_RUNTIME_CONTRACT.get(_key, False)):
        raise RuntimeError(f"[Cell12.b] PRE-CELL11 runtime contract missing required flag: {_key}")

# Some PRE-CELL11 runs set cell11_use_materialized_protocol_variants as a runtime CFG
# flag rather than serializing it in the JSON contract. Accept either explicit
# contract flag, matching CFG flag, or the canonical materialization authority/path
# evidence. This remains fail-closed if the materialized A0/A1/A2 paths are absent.
_materialized_variant_flag_ok = bool(
    PRECELL11_RUNTIME_CONTRACT.get("cell11_use_materialized_protocol_variants", False)
    or CFG.get("cell11_use_materialized_protocol_variants", False)
    or (
        str(PRECELL11_RUNTIME_CONTRACT.get("materialization_authority", "")).startswith("Cell10.9")
        and isinstance(PRECELL11_RUNTIME_CONTRACT.get("paths", {}), dict)
        and all(
            k in PRECELL11_RUNTIME_CONTRACT.get("paths", {})
            for k in ["A0_PROTOCOL_TEST", "A1_PROTOCOL_TEST", "A2_PROTOCOL_TEST"]
        )
    )
)
if not _materialized_variant_flag_ok:
    raise RuntimeError(
        "[Cell12.b] PRE-CELL11 runtime contract does not prove Cell 11 used "
        "materialized A0/A1/A2 protocol variants."
    )

log(
    "[Cell12.b] Upstream protocol/IoT contracts locked | "
    "cell12a_manifest=True | cell10_8_v3_6=True | cell10_9_v2_1=True | "
    "precell11_v4_1=True | cell11_manifest=True"
)

N_CPU = int(globals().get("N_CPU", max(1, os.cpu_count() or 4)))

# ----------------------------------------------------------
# v27.1 configuration
# ----------------------------------------------------------
_SCAFFOLD_DEFAULTS = {
    # Global / explicit mask synthesis
    "iot_mask_piece_min": 256,
    "iot_mask_piece_max": 8192,
    "iot_mask_piece_draws": 768,
    "iot_mask_exact_rate_correct": True,

    # Entity regime synthesis
    "entity_regime_piece_min": 256,
    "entity_regime_piece_max": 4096,
    "entity_regime_piece_draws": 768,
    "entity_regime_driver_radius": 10,
    "entity_regime_active_floor_dynamic": 0.15,
    "entity_regime_target_occ_blend_trainval": 1.0,

    # Binary synthesis
    "binary_piece_min": 256,
    "binary_piece_max": 4096,
    "binary_piece_draws": 768,
    "binary_exact_rate_correct": True,
    "binary_rare_event_run_cap": 8,
    "binary_reporting_indicator_apply_global_mask": False,

    # Value availability synthesis
    "value_avail_piece_min": 256,
    "value_avail_piece_max": 8192,
    "value_avail_piece_draws": 768,
    "value_avail_exact_rate_correct": True,

    # Semantics
    "dense_rate_threshold": 0.985,
    "near_dense_rate_threshold": 0.950,
    "sparse_rate_threshold": 0.100,
    "very_sparse_rate_threshold": 0.020,

    # Audits/gates
    "max_value_avail_rate_mae_trainval_target": 0.035,
    "max_binary_rate_mae_trainval_target": 0.035,
    "max_regime_occ_mae": 0.150,
    "publication_strict_scaffold_gate": False,
}

for _k, _v in _SCAFFOLD_DEFAULTS.items():
    CFG12.setdefault(_k, _v)

# hard-disable old DDPM-in-12b behavior
CFG12["cell12b_trains_ddpm"] = False
CFG12["cell12b_samples_ddpm"] = False
CFG12["cell12b_generates_continuous_values"] = False
CFG12["cell12b_role"] = "iot_scaffold_only"
CFG12["cell12b_test_usage"] = "length_index_schema_only"
CFG12["cell12b_generator_selection"] = "none_deferred_to_modular_cell12c"

# ----------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------
def _write_json(path, obj):
    def _clean(x):
        if isinstance(x, dict):
            return {str(k): _clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_clean(v) for v in x]
        if isinstance(x, tuple):
            return [_clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return _clean(x.tolist())
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            y = float(x)
            return None if not np.isfinite(y) else y
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, float):
            return None if not np.isfinite(x) else x
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=2)


def _sha256_file(path, block_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _stable_hash_int(name, offset=0):
    h = hashlib.sha256(f"{SEED}|{name}|{offset}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32 - 1)


def _rng_for(name, offset=0):
    return np.random.default_rng(_stable_hash_int(name, offset))


def _safe_float_series(s):
    return pd.to_numeric(s, errors="coerce").astype(np.float32)


def _safe_float_frame(df_like):
    out = pd.DataFrame(index=df_like.index)
    for c in df_like.columns:
        out[c] = pd.to_numeric(df_like[c], errors="coerce").astype(np.float32)
    return out


def _finite_np(x, dtype=np.float32):
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    return arr[np.isfinite(arr)]


def _clip01_np(x):
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _as_binary01(x):
    return (_clip01_np(x) > 0.5).astype(np.float32)


def _normalize_prob_vector(p, n=None):
    arr = np.asarray(p, dtype=np.float64).reshape(-1)
    if n is not None and int(n) != len(arr):
        raise ValueError(f"Probability vector length mismatch: got {len(arr)}, expected {n}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0:
        n_eff = int(n) if n is not None else int(len(arr))
        return np.full(n_eff, 1.0 / max(1, n_eff), dtype=np.float64)
    arr = arr / s
    return arr.astype(np.float64)


def _run_lengths_bool(mask_bool):
    x = np.asarray(mask_bool, dtype=np.int8).reshape(-1)
    runs0, runs1 = [], []
    if len(x) == 0:
        return runs0, runs1

    cur = int(x[0])
    L = 1
    for v in x[1:]:
        v = int(v)
        if v == cur:
            L += 1
        else:
            if cur == 1:
                runs1.append(int(L))
            else:
                runs0.append(int(L))
            cur = v
            L = 1

    if cur == 1:
        runs1.append(int(L))
    else:
        runs0.append(int(L))

    return runs0, runs1


def _make_segments_from_state(arr):
    x = np.asarray(arr, dtype=np.int16).reshape(-1)
    if len(x) == 0:
        return []
    out = []
    s = 0
    cur = int(x[0])
    for i in range(1, len(x)):
        if int(x[i]) != cur:
            out.append((int(s), int(i), int(cur)))
            s = i
            cur = int(x[i])
    out.append((int(s), int(len(x)), int(cur)))
    return out


def _sample_run_length(bank, rrng, fallback=16):
    if bank is None or len(bank) == 0:
        return int(max(1, fallback))
    return int(max(1, bank[int(rrng.integers(0, len(bank)))]))


def _exact_rate_correct_binary(arr, target, rrng):
    x = _as_binary01(arr)
    n = int(len(x))
    if n == 0:
        return x

    target = float(np.clip(target, 0.0, 1.0))
    desired = int(round(target * n))
    cur = int(x.sum())

    if cur < desired:
        zeros = np.where(x < 0.5)[0]
        add_n = min(len(zeros), desired - cur)
        if add_n > 0:
            choose = rrng.choice(zeros, size=add_n, replace=False)
            x[choose] = 1.0

    elif cur > desired:
        ones = np.where(x > 0.5)[0]
        drop_n = min(len(ones), cur - desired)
        if drop_n > 0:
            choose = rrng.choice(ones, size=drop_n, replace=False)
            x[choose] = 0.0

    return x.astype(np.float32)


def _build_piece_bank(mask_arr, name, piece_min, piece_max, n_draws):
    x = np.asarray(mask_arr, dtype=np.int8).reshape(-1)
    n = int(len(x))
    if n == 0:
        return []

    rrng = _rng_for(f"piece_bank::{name}", 1010)
    out = []
    draws = int(max(16, min(int(n_draws), max(16, n // max(1, int(piece_min))))))

    for _ in range(draws):
        L = int(rrng.integers(int(piece_min), int(piece_max) + 1))
        L = int(max(1, min(L, n)))
        if n <= L:
            seg = x.copy()
        else:
            s = int(rrng.integers(0, n - L + 1))
            seg = x[s:s + L].copy()
        out.append(seg.astype(np.int8))

    return out


def _assemble_from_piece_bank(piece_bank, n_out, rrng, fallback_rate=1.0):
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    if piece_bank is None or len(piece_bank) == 0:
        return (rrng.random(n_out) < float(fallback_rate)).astype(np.float32)

    chunks = []
    total = 0
    while total < n_out:
        seg = np.asarray(piece_bank[int(rrng.integers(0, len(piece_bank)))], dtype=np.int8)
        if len(seg) == 0:
            continue
        chunks.append(seg)
        total += int(len(seg))

    return np.concatenate(chunks, axis=0)[:n_out].astype(np.float32)


def _binary_rate(x):
    arr = _as_binary01(x)
    return float(arr.mean()) if len(arr) else np.nan


def _mean_run_length_binary(x, state=1):
    arr = _as_binary01(x).astype(np.int8)
    runs = []
    cur = 0
    for v in arr:
        if int(v) == int(state):
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)
    return float(np.mean(runs)) if len(runs) else 0.0


def _transition_rate(x):
    arr = _as_binary01(x).astype(np.int8)
    if len(arr) < 2:
        return 0.0
    return float(np.mean(arr[1:] != arr[:-1]))


def _multi_state_run_lengths(arr, n_states):
    x = np.asarray(arr, dtype=np.int16).reshape(-1)
    out = {int(s): [] for s in range(int(n_states))}
    if len(x) == 0:
        return out

    cur = int(x[0])
    L = 1
    for v in x[1:]:
        v = int(v)
        if v == cur:
            L += 1
        else:
            if cur in out:
                out[cur].append(int(L))
            cur = v
            L = 1

    if cur in out:
        out[cur].append(int(L))

    return out


def _state_occupancy(arr, n_states=4):
    x = np.asarray(arr, dtype=np.int16).reshape(-1)
    occ = np.zeros(int(n_states), dtype=np.float64)
    if len(x) == 0:
        return occ
    vals, cnts = np.unique(x, return_counts=True)
    for v, c in zip(vals, cnts):
        vi = int(v)
        if 0 <= vi < int(n_states):
            occ[vi] = float(c) / float(len(x))
    return occ.astype(np.float32)


def _fit_transition_table(arr, n_states=4, alpha=1.0):
    x = np.asarray(arr, dtype=np.int16).reshape(-1)
    T = np.full((int(n_states), int(n_states)), float(alpha), dtype=np.float64)

    if len(x) >= 2:
        for a, b in zip(x[:-1], x[1:]):
            ai = int(a)
            bi = int(b)
            if 0 <= ai < int(n_states) and 0 <= bi < int(n_states):
                T[ai, bi] += 1.0

    T = T / np.maximum(T.sum(axis=1, keepdims=True), 1e-12)
    return T.astype(np.float32)


def _active_near_driver_fraction(reg_arr, drv_arr, states=(2, 3), radius=10):
    reg = np.asarray(reg_arr, dtype=np.int8)
    drv = np.asarray(drv_arr, dtype=np.int8)
    n = int(len(reg))
    if n == 0:
        return 0.0

    idx = np.where(drv > 0)[0]
    if len(idx) == 0:
        return 0.0

    near = np.zeros(n, dtype=bool)
    r = int(radius)
    for t in idx:
        s = max(0, int(t) - r)
        e = min(n, int(t) + r + 1)
        near[s:e] = True

    st_mask = np.isin(reg, list(states))
    denom = int(st_mask.sum())
    if denom == 0:
        return 0.0

    return float((st_mask & near).sum() / max(1, denom))


# ----------------------------------------------------------
# Canonical entity mapping
# ----------------------------------------------------------
def _canonical_entity(col):
    s = str(col).lower()

    m = re.match(r"^(telemetry_in_sec|events_in_sec)__entity__(.+)$", s)
    if m:
        return m.group(2)

    replacements = [
        ("smart_plug_ps5_this_month_s", "smart_plug_ps5"),
        ("smart_plug_ps5_today_s", "smart_plug_ps5"),
        ("smart_plug_this_month_s", "smart_plug"),
        ("smart_plug_today_s", "smart_plug"),
    ]
    for src, dst in replacements:
        if s.startswith(f"iot__{src}__") or s == src:
            return dst

    if s.startswith("iot__coffee_maker_milk__sensor__cups__value"):
        return "coffee_maker"
    if s.startswith("iot__coffee_maker__sensor__"):
        return "coffee_maker"
    if s.startswith("iot__coffee_maker_connectivity__"):
        return "coffee_maker"

    ent = ENTITY_MAP.get(col, None)
    if ent is not None:
        if ent in {"smart_plug_today_s", "smart_plug_this_month_s"}:
            return "smart_plug"
        if ent in {"smart_plug_ps5_today_s", "smart_plug_ps5_this_month_s"}:
            return "smart_plug_ps5"
        if ent == "coffee_maker_milk" and "cups__value" in s:
            return "coffee_maker"
        return str(ent)

    try:
        ent = _infer_entity_from_col(col)
    except Exception:
        ent = "misc"

    if ent in {"smart_plug_today_s", "smart_plug_this_month_s"}:
        return "smart_plug"
    if ent in {"smart_plug_ps5_today_s", "smart_plug_ps5_this_month_s"}:
        return "smart_plug_ps5"
    if ent == "coffee_maker_milk" and "cups__value" in s:
        return "coffee_maker"

    return str(ent) if str(ent) else "misc"


ALL_IOT_COLS = sorted(set(DRV_COLS) | set(BIN_COLS) | set(CONT_VALUE_COLS))
CANON_ENTITY_MAP = {c: _canonical_entity(c) for c in ALL_IOT_COLS}
ENTITY_LIST = sorted(set(CANON_ENTITY_MAP.values()))

ENTITY_TO_DRV_CANON = {e: [c for c in DRV_COLS if CANON_ENTITY_MAP.get(c) == e] for e in ENTITY_LIST}
ENTITY_TO_BIN_CANON = {e: [c for c in BIN_COLS if CANON_ENTITY_MAP.get(c) == e] for e in ENTITY_LIST}
ENTITY_TO_CONT_CANON = {e: [c for c in CONT_VALUE_COLS if CANON_ENTITY_MAP.get(c) == e] for e in ENTITY_LIST}

def _resolve_entity_for_any_col(col):
    if col in CANON_ENTITY_MAP:
        return CANON_ENTITY_MAP[col]

    s = str(col).lower()
    base = re.sub(r"__(obs_present|stale_flag|value|state)$", "", s)
    cands = [s, base, f"{base}__value", f"{base}__state", f"{base}__obs_present", f"{base}__stale_flag"]

    for cand in cands:
        if cand in CANON_ENTITY_MAP:
            return CANON_ENTITY_MAP[cand]

    return _canonical_entity(col)


# ----------------------------------------------------------
# TRAIN+VAL matrices
# ----------------------------------------------------------
X_drv_tr = _safe_float_frame(X_drv_tr)
X_drv_val = _safe_float_frame(X_drv_val)
X_drv_te = _safe_float_frame(X_drv_te)

X_bin_tr = _safe_float_frame(X_bin_tr)
X_bin_val = _safe_float_frame(X_bin_val)
X_bin_te = _safe_float_frame(X_bin_te)

X_cont_tr = _safe_float_frame(X_cont_tr)
X_cont_val = _safe_float_frame(X_cont_val)
X_cont_te = _safe_float_frame(X_cont_te)

TRAINVAL_DRV = pd.concat([X_drv_tr[DRV_COLS], X_drv_val[DRV_COLS]], axis=0).reset_index(drop=True)
TRAINVAL_BIN = pd.concat([X_bin_tr[BIN_COLS], X_bin_val[BIN_COLS]], axis=0).reset_index(drop=True)
TRAINVAL_CONT = pd.concat([X_cont_tr[CONT_COLS], X_cont_val[CONT_COLS]], axis=0).reset_index(drop=True)

N_TRVA = int(len(TRAINVAL_DRV))
if N_TRVA != int(N_TR + N_VAL):
    raise RuntimeError(f"[Cell12.b] TRAINVAL length mismatch: {N_TRVA} vs {N_TR + N_VAL}")

log(
    f"[Cell12.b] Loaded TRAIN+VAL scaffold matrices | "
    f"DRV={TRAINVAL_DRV.shape} | BIN={TRAINVAL_BIN.shape} | CONT={TRAINVAL_CONT.shape} | "
    f"N_TE={N_TE}"
)

# ----------------------------------------------------------
# Entity driver activity
# ----------------------------------------------------------
def _entity_activity_from_driver_df_canon(df_drv, entity):
    rel_cols = [c for c in ENTITY_TO_DRV_CANON.get(entity, []) if c in df_drv.columns]
    if not rel_cols:
        return np.zeros(len(df_drv), dtype=np.int8)

    block = df_drv[rel_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return (block.to_numpy(dtype=np.float32, copy=False).sum(axis=1) > 0.0).astype(np.int8)


ENTITY_DRIVER_ACTIVITY_TRVA = {
    e: _entity_activity_from_driver_df_canon(TRAINVAL_DRV, e).astype(np.int8)
    for e in ENTITY_LIST
}

ENTITY_DRIVER_ACTIVITY_TR = {
    e: _entity_activity_from_driver_df_canon(X_drv_tr, e).astype(np.int8)
    for e in ENTITY_LIST
}

ENTITY_DRIVER_ACTIVITY_VAL = {
    e: _entity_activity_from_driver_df_canon(X_drv_val, e).astype(np.int8)
    for e in ENTITY_LIST
}

ENTITY_DRIVER_ACTIVITY_SYN = {
    e: _entity_activity_from_driver_df_canon(IOT_SYN_DRIVERS, e).astype(np.int8)
    for e in ENTITY_LIST
}

# ----------------------------------------------------------
# Explicit/global IoT mask synthesis
# ----------------------------------------------------------
def _fit_explicit_mask_generator(mask_col):
    tr = pd.to_numeric(X_cont_tr[mask_col], errors="coerce").fillna(0.0).clip(0, 1).to_numpy(dtype=np.float32)
    va = pd.to_numeric(X_cont_val[mask_col], errors="coerce").fillna(0.0).clip(0, 1).to_numpy(dtype=np.float32)

    full = np.concatenate([tr, va], axis=0)
    full01 = _as_binary01(full).astype(np.int8)

    piece_bank = _build_piece_bank(
        full01,
        name=f"explicit_mask::{mask_col}",
        piece_min=int(CFG12["iot_mask_piece_min"]),
        piece_max=int(CFG12["iot_mask_piece_max"]),
        n_draws=int(CFG12["iot_mask_piece_draws"]),
    )

    return {
        "mask_col": mask_col,
        "train_rate": float(_binary_rate(tr)),
        "val_rate": float(_binary_rate(va)),
        "trainval_rate": float(_binary_rate(full01)),
        "runs_trainval": _run_lengths_bool(full01),
        "piece_bank_trainval": piece_bank,
    }


def _generate_explicit_mask(mask_col, gen):
    rrng = _rng_for(f"explicit_mask_generate::{mask_col}", 2100)
    target = float(gen["trainval_rate"])

    if target >= 0.999999:
        return np.ones(N_TE, dtype=np.float32)
    if target <= 1e-9:
        return np.zeros(N_TE, dtype=np.float32)

    arr = _assemble_from_piece_bank(
        gen.get("piece_bank_trainval", []),
        N_TE,
        rrng,
        fallback_rate=target,
    )

    if bool(CFG12.get("iot_mask_exact_rate_correct", True)):
        arr = _exact_rate_correct_binary(arr, target, rrng)

    return arr.astype(np.float32)


log(f"[Cell12.b] Explicit mask cols: n={len(CONT_MASK_COLS)} | cols={list(CONT_MASK_COLS)}")

EXPL_MASK_GENS = {}
IOT_SYN_MASKS = pd.DataFrame(index=df_te.index)

for mcol in CONT_MASK_COLS:
    if mcol not in X_cont_tr.columns or mcol not in X_cont_val.columns:
        raise RuntimeError(f"[Cell12.b] Explicit mask column missing from TRAIN/VAL matrices: {mcol}")

    EXPL_MASK_GENS[mcol] = _fit_explicit_mask_generator(mcol)
    IOT_SYN_MASKS[mcol] = _generate_explicit_mask(mcol, EXPL_MASK_GENS[mcol]).astype(np.float32)

GLOBAL_IOT_MASK_SYN = (
    IOT_SYN_MASKS["iot__obs_present"].to_numpy(dtype=np.float32)
    if "iot__obs_present" in IOT_SYN_MASKS.columns
    else np.ones(N_TE, dtype=np.float32)
)

mask_rates = {c: float(IOT_SYN_MASKS[c].mean()) for c in IOT_SYN_MASKS.columns}
mask_fit_rates = {
    c: {
        "train_rate": EXPL_MASK_GENS[c]["train_rate"],
        "val_rate": EXPL_MASK_GENS[c]["val_rate"],
        "trainval_rate": EXPL_MASK_GENS[c]["trainval_rate"],
        "syn_rate": float(IOT_SYN_MASKS[c].mean()),
    }
    for c in IOT_SYN_MASKS.columns
}

log(f"[Cell12.b] Explicit mask synthesis complete | rates={mask_rates}")

# ----------------------------------------------------------
# Entity profile fitting
# ----------------------------------------------------------
def _fit_entity_profiles():
    profiles = {}

    for e in ENTITY_LIST:
        dcols = [c for c in ENTITY_TO_DRV_CANON.get(e, []) if c in TRAINVAL_DRV.columns]
        bcols = [c for c in ENTITY_TO_BIN_CANON.get(e, []) if c in TRAINVAL_BIN.columns]
        ccols = [c for c in ENTITY_TO_CONT_CANON.get(e, []) if c in TRAINVAL_CONT.columns]

        drv = ENTITY_DRIVER_ACTIVITY_TRVA[e]

        if bcols:
            B = TRAINVAL_BIN[bcols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1).to_numpy(dtype=np.float32)
            bin_any = (B.sum(axis=1) > 0).astype(np.int8)
            bin_rate_row = B.mean(axis=1).astype(np.float32)
        else:
            bin_any = np.zeros(N_TRVA, dtype=np.int8)
            bin_rate_row = np.zeros(N_TRVA, dtype=np.float32)

        if ccols:
            Cmask = TRAINVAL_CONT[ccols].notna().to_numpy(dtype=np.float32)
            row_avail = Cmask.mean(axis=1).astype(np.float32)
            col_avail = Cmask.mean(axis=0).astype(np.float32)
        else:
            row_avail = np.zeros(N_TRVA, dtype=np.float32)
            col_avail = np.zeros(0, dtype=np.float32)

        drv_rate = float(drv.mean()) if len(drv) else 0.0
        bin_rate = float(bin_any.mean()) if len(bin_any) else 0.0
        row_avail_q10 = float(np.quantile(row_avail, 0.10)) if len(row_avail) else 0.0
        row_avail_q50 = float(np.quantile(row_avail, 0.50)) if len(row_avail) else 0.0
        dense_cols_n = int(np.sum(col_avail >= 0.95)) if len(col_avail) else 0

        if dense_cols_n >= 2 and row_avail_q10 >= 0.35:
            entity_class = "structural_online"
        elif dense_cols_n >= 1 or row_avail_q50 >= 0.50:
            entity_class = "mostly_online"
        elif drv_rate >= 0.005 or bin_rate >= 0.020:
            entity_class = "activity_online"
        else:
            entity_class = "sparse_entity"

        profiles[e] = {
            "entity": e,
            "entity_class": entity_class,
            "driver_cols": dcols,
            "binary_cols": bcols,
            "cont_cols": ccols,
            "drv_full": drv.astype(np.int8),
            "bin_any_full": bin_any.astype(np.int8),
            "bin_rate_row_full": bin_rate_row.astype(np.float32),
            "row_avail_full": row_avail.astype(np.float32),
            "drv_rate_trainval": float(drv_rate),
            "bin_any_rate_trainval": float(bin_rate),
            "row_avail_q10": float(row_avail_q10),
            "row_avail_q50": float(row_avail_q50),
            "dense_cont_cols_n": int(dense_cols_n),
        }

    return profiles


ENTITY_PROFILES = _fit_entity_profiles()
entity_class_counts = pd.Series([ENTITY_PROFILES[e]["entity_class"] for e in ENTITY_LIST]).value_counts().to_dict()

log(f"[Cell12.b] Entity classes: {entity_class_counts}")

# ----------------------------------------------------------
# Entity regime fitting/generation
# ----------------------------------------------------------
ENTITY_REGIME_STATES = {0: "offline", 1: "online_idle", 2: "online_active", 3: "recovery"}

def _derive_trainval_regime_for_entity(entity):
    prof = ENTITY_PROFILES[entity]
    drv = np.asarray(prof["drv_full"], dtype=np.int8)
    bin_any = np.asarray(prof["bin_any_full"], dtype=np.int8)
    bin_rate = np.asarray(prof["bin_rate_row_full"], dtype=np.float32)
    row_avail = np.asarray(prof["row_avail_full"], dtype=np.float32)
    entity_class = prof["entity_class"]

    score = (
        2.50 * drv.astype(np.float32)
        + 1.25 * bin_any.astype(np.float32)
        + 0.75 * np.clip(bin_rate, 0.0, 1.0)
        + 0.35 * np.clip(row_avail, 0.0, 1.0)
    ).astype(np.float32)

    reg = np.ones(len(score), dtype=np.int8)

    if entity_class in {"structural_online", "mostly_online"}:
        offline = (drv == 0) & (bin_any == 0) & (row_avail <= 0.01)
    else:
        offline = (drv == 0) & (bin_any == 0) & (row_avail <= 0.03)

    reg[offline] = 0
    reg[score >= 1.50] = 2

    prev_active = np.r_[0, (reg[:-1] == 2).astype(np.int8)]
    next_active = np.r_[(reg[1:] == 2).astype(np.int8), 0]
    reg[(reg == 1) & ((prev_active == 1) | (next_active == 1))] = 3

    if entity_class == "structural_online":
        # Structural devices should not become mostly offline unless TRAIN+VAL says so.
        off_rate = float(np.mean(reg == 0))
        if off_rate < 0.001:
            reg[reg == 0] = 1

    return reg.astype(np.int8)


def _build_regime_donor_bank(regime, drv, entity, piece_min, piece_max, n_draws):
    reg = np.asarray(regime, dtype=np.int8)
    drv = np.asarray(drv, dtype=np.int8)
    n = int(len(reg))
    rrng = _rng_for(f"regime_donor_bank::{entity}", 3000)

    if n == 0:
        return {"all": [], "low": [], "mid": [], "high": []}

    pieces = []
    draws = int(max(32, min(int(n_draws), max(32, n // max(1, int(piece_min))))))

    for _ in range(draws):
        L = int(rrng.integers(int(piece_min), int(piece_max) + 1))
        L = int(max(8, min(L, n)))
        if n <= L:
            s = 0
            e = n
        else:
            s = int(rrng.integers(0, n - L + 1))
            e = s + L

        seg_reg = reg[s:e].copy()
        seg_drv = drv[s:e].copy()
        drv_rate = float(seg_drv.mean()) if len(seg_drv) else 0.0

        pieces.append({
            "reg": seg_reg.astype(np.int8),
            "drv": seg_drv.astype(np.int8),
            "len": int(len(seg_reg)),
            "drv_rate": float(drv_rate),
            "occ": _state_occupancy(seg_reg, 4),
        })

    by_drv = {"low": [], "mid": [], "high": [], "all": pieces}
    for p in pieces:
        r = float(p["drv_rate"])
        if r <= 0.005:
            by_drv["low"].append(p)
        elif r <= 0.050:
            by_drv["mid"].append(p)
        else:
            by_drv["high"].append(p)

    return by_drv


def _extract_regime_event_motifs(regime, drv, entity_class, radius=10):
    reg = np.asarray(regime, dtype=np.int8)
    drv = np.asarray(drv, dtype=np.int8)
    n = int(len(reg))
    motifs = []

    for s, e, st in _make_segments_from_state((drv > 0).astype(np.int8)):
        if int(st) != 1:
            continue

        seg_len = int(e - s)
        pre = int(np.clip(max(4, radius), 4, 96))
        post = int(np.clip(max(8, 2 * radius), 8, 192))
        ws = max(0, int(s) - pre)
        we = min(n, int(e) + post)

        reg_win = reg[ws:we].copy()
        drv_win = drv[ws:we].copy()

        if len(reg_win) < 4:
            continue

        nonidle_frac = float(np.mean(np.isin(reg_win, [2, 3])))
        active_frac = float(np.mean(reg_win == 2))

        if entity_class in {"mostly_online", "activity_online"} and nonidle_frac <= 0.005 and active_frac <= 0.002:
            continue

        motifs.append({
            "reg": reg_win.astype(np.int8),
            "drv": drv_win.astype(np.int8),
            "len": int(len(reg_win)),
            "seg_len": int(seg_len),
            "anchor_lo": int(s - ws),
            "anchor_hi": int(e - ws),
            "nonidle_frac": float(nonidle_frac),
            "active_frac": float(active_frac),
        })

    return motifs


def _fit_entity_regime_models():
    models = {}

    for e in ENTITY_LIST:
        reg = _derive_trainval_regime_for_entity(e)
        drv = np.asarray(ENTITY_PROFILES[e]["drv_full"], dtype=np.int8)
        cls = ENTITY_PROFILES[e]["entity_class"]

        target_occ = _state_occupancy(reg, 4)
        trans = _fit_transition_table(reg, n_states=4, alpha=2.0)
        run_banks = _multi_state_run_lengths(reg, n_states=4)

        donor_bank_by_drv = _build_regime_donor_bank(
            regime=reg,
            drv=drv,
            entity=e,
            piece_min=int(CFG12["entity_regime_piece_min"]),
            piece_max=int(CFG12["entity_regime_piece_max"]),
            n_draws=int(CFG12["entity_regime_piece_draws"]),
        )

        motif_bank = _extract_regime_event_motifs(
            regime=reg,
            drv=drv,
            entity_class=cls,
            radius=int(CFG12["entity_regime_driver_radius"]),
        )

        models[e] = {
            "entity": e,
            "entity_class": cls,
            "regime_trainval": reg.astype(np.int8),
            "drv_full": drv.astype(np.int8),
            "target_occ": target_occ.astype(np.float32),
            "trans": trans.astype(np.float32),
            "run_banks": run_banks,
            "donor_bank_by_drv": donor_bank_by_drv,
            "event_motif_bank": motif_bank,
            "real_active_near_driver_frac": _active_near_driver_fraction(reg, drv, states=(2,), radius=int(CFG12["entity_regime_driver_radius"])),
            "real_nonidle_near_driver_frac": _active_near_driver_fraction(reg, drv, states=(2, 3), radius=int(CFG12["entity_regime_driver_radius"])),
        }

    return models


ENTITY_REGIME_MODELS = _fit_entity_regime_models()


def _choose_regime_piece(bank_by_drv, local_drv_rate, rrng):
    if local_drv_rate <= 0.005 and len(bank_by_drv.get("low", [])) > 0:
        bank = bank_by_drv["low"]
    elif local_drv_rate <= 0.050 and len(bank_by_drv.get("mid", [])) > 0:
        bank = bank_by_drv["mid"]
    elif len(bank_by_drv.get("high", [])) > 0:
        bank = bank_by_drv["high"]
    else:
        bank = bank_by_drv.get("all", [])

    if len(bank) == 0:
        return None

    return bank[int(rrng.integers(0, len(bank)))]


def _overlay_regime_event_motifs(out, drv_syn, motif_bank, entity_class, rrng):
    out = np.asarray(out, dtype=np.int8).copy()
    drv_syn = np.asarray(drv_syn, dtype=np.int8)
    n = int(len(out))

    if len(motif_bank) == 0:
        out[drv_syn > 0] = np.maximum(out[drv_syn > 0], 2).astype(np.int8)
        return out.astype(np.int8)

    for s, e, st in _make_segments_from_state((drv_syn > 0).astype(np.int8)):
        if int(st) != 1:
            continue

        seg_len = int(e - s)
        if seg_len <= 0:
            continue

        lens = np.array([abs(int(m["seg_len"]) - seg_len) for m in motif_bank], dtype=np.int32)
        k = int(min(8, len(motif_bank)))
        top_idx = np.argsort(lens)[:k]
        chosen = motif_bank[int(top_idx[int(rrng.integers(0, len(top_idx)))])]

        reg_win = np.asarray(chosen["reg"], dtype=np.int8)
        anchor_lo = int(chosen["anchor_lo"])

        ws = max(0, int(s) - anchor_lo)
        we = min(n, ws + int(len(reg_win)))

        take = int(max(0, we - ws))
        if take <= 0:
            continue

        reg_take = reg_win[:take]
        cur = out[ws:we].copy()

        if entity_class in {"mostly_online", "activity_online"}:
            cur = reg_take.copy()
        else:
            dyn = np.isin(reg_take, [2, 3])
            cur[dyn] = reg_take[dyn]

        out[ws:we] = cur

        core_s = max(0, int(s))
        core_e = min(n, int(e))
        out[core_s:core_e] = 2

        halo_e = min(n, core_e + max(2, min(8, seg_len)))
        if halo_e > core_e:
            halo_idx = np.arange(core_e, halo_e, dtype=np.int64)
            idle = halo_idx[out[halo_idx] == 1]
            out[idle] = 3

    return out.astype(np.int8)


def _mild_regime_occupancy_rebalance(arr, drv_syn, target_occ, entity_class, rrng):
    arr = np.asarray(arr, dtype=np.int8).copy()
    drv_syn = np.asarray(drv_syn, dtype=np.int8)
    n = int(len(arr))
    if n == 0:
        return arr

    cur_occ = _state_occupancy(arr, 4).astype(np.float64)
    tgt = np.asarray(target_occ, dtype=np.float64)

    # Restore active mass near driver-positive regions first.
    active_deficit = float(tgt[2] - cur_occ[2])
    if active_deficit > 0.010:
        need = int(round(active_deficit * n))
        cand = np.where((drv_syn > 0) & np.isin(arr, [1, 3]))[0]
        if len(cand) > 0 and need > 0:
            take = rrng.choice(cand, size=min(len(cand), need), replace=False)
            arr[take] = 2

    # Prevent excessive active inflation.
    active_excess = float(cur_occ[2] - tgt[2])
    if active_excess > 0.020:
        need = int(round(active_excess * n))
        cand = np.where((arr == 2) & (drv_syn == 0))[0]
        if len(cand) > 0 and need > 0:
            take = rrng.choice(cand, size=min(len(cand), need), replace=False)
            arr[take] = 1

    # Structural devices should not become offline-heavy unless real TRAIN+VAL says so.
    if entity_class in {"structural_online", "mostly_online"}:
        cur_off = float(np.mean(arr == 0))
        max_off = float(tgt[0] + (0.010 if entity_class == "structural_online" else 0.030))
        if cur_off > max_off:
            need = int(round((cur_off - max_off) * n))
            cand = np.where(arr == 0)[0]
            if len(cand) > 0 and need > 0:
                take = rrng.choice(cand, size=min(len(cand), need), replace=False)
                arr[take] = 1

    prev_active = np.r_[0, (arr[:-1] == 2).astype(np.int8)]
    next_active = np.r_[(arr[1:] == 2).astype(np.int8), 0]
    arr[(arr == 1) & ((prev_active == 1) | (next_active == 1))] = 3

    return arr.astype(np.int8)


def _generate_entity_regimes():
    syn = {}

    for e in ENTITY_LIST:
        rrng = _rng_for(f"entity_regime_generate::{e}", 3200)
        model = ENTITY_REGIME_MODELS[e]
        cls = model["entity_class"]
        drv_syn = ENTITY_DRIVER_ACTIVITY_SYN[e].astype(np.int8)
        bank_by_drv = model["donor_bank_by_drv"]
        all_bank = bank_by_drv.get("all", [])

        if len(all_bank) == 0:
            arr = np.ones(N_TE, dtype=np.int8)
            arr[drv_syn > 0] = 2
            syn[e] = arr
            continue

        out = np.ones(N_TE, dtype=np.int8)
        pos = 0

        while pos < N_TE:
            look_e = min(N_TE, pos + 128)
            local_drv_rate = float(np.mean(drv_syn[pos:look_e])) if look_e > pos else 0.0
            piece = _choose_regime_piece(bank_by_drv, local_drv_rate, rrng)
            if piece is None:
                piece = all_bank[int(rrng.integers(0, len(all_bank)))]

            seg = np.asarray(piece["reg"], dtype=np.int8)
            if len(seg) == 0:
                break

            L = int(len(seg))
            if L >= 32:
                Lj = int(round(L * (1.0 + rrng.normal(0.0, 0.08))))
                Lj = int(np.clip(Lj, 16, max(16, 2 * L)))
            else:
                Lj = L

            take = min(N_TE - pos, max(1, Lj))

            if len(seg) == take:
                out[pos:pos + take] = seg
            elif len(seg) > take:
                s0 = int(rrng.integers(0, len(seg) - take + 1))
                out[pos:pos + take] = seg[s0:s0 + take]
            else:
                reps = int(np.ceil(take / max(1, len(seg))))
                out[pos:pos + take] = np.tile(seg, reps)[:take]

            pos += take

        out = _overlay_regime_event_motifs(
            out=out,
            drv_syn=drv_syn,
            motif_bank=model["event_motif_bank"],
            entity_class=cls,
            rrng=rrng,
        )

        out = _mild_regime_occupancy_rebalance(
            arr=out,
            drv_syn=drv_syn,
            target_occ=model["target_occ"],
            entity_class=cls,
            rrng=rrng,
        )

        if cls == "structural_online" and float(model["target_occ"][0]) < 1e-6:
            out[out == 0] = 1

        syn[e] = out.astype(np.int8)

    return syn


ENTITY_REGIME_SYN = _generate_entity_regimes()

# Keep backward-compatible alias used by some older 12.c variants.
ENTITY_REGIME_STATES = ENTITY_REGIME_STATES

# ----------------------------------------------------------
# Regime audits
# ----------------------------------------------------------
_regime_rows = []
for e in ENTITY_LIST:
    real_reg = np.asarray(ENTITY_REGIME_MODELS[e]["regime_trainval"], dtype=np.int8)
    syn_reg = np.asarray(ENTITY_REGIME_SYN[e], dtype=np.int8)
    real_drv = np.asarray(ENTITY_REGIME_MODELS[e]["drv_full"], dtype=np.int8)
    syn_drv = np.asarray(ENTITY_DRIVER_ACTIVITY_SYN[e], dtype=np.int8)

    real_occ = _state_occupancy(real_reg, 4)
    syn_occ = _state_occupancy(syn_reg, 4)

    real_switch = _transition_rate(real_reg)
    syn_switch = _transition_rate(syn_reg)

    _regime_rows.append({
        "entity": e,
        "entity_class": ENTITY_REGIME_MODELS[e]["entity_class"],
        "trva_regime_0_frac": float(real_occ[0]),
        "trva_regime_1_frac": float(real_occ[1]),
        "trva_regime_2_frac": float(real_occ[2]),
        "trva_regime_3_frac": float(real_occ[3]),
        "syn_regime_0_frac": float(syn_occ[0]),
        "syn_regime_1_frac": float(syn_occ[1]),
        "syn_regime_2_frac": float(syn_occ[2]),
        "syn_regime_3_frac": float(syn_occ[3]),
        "abs_err_sum": float(np.abs(syn_occ - real_occ).sum()),
        "max_occ_abs_err": float(np.max(np.abs(syn_occ - real_occ))),
        "real_dom_frac": float(np.max(real_occ)),
        "syn_dom_frac": float(np.max(syn_occ)),
        "raw_dom_collapsed_ge_0_98": bool(np.max(syn_occ) >= 0.98),
        "real_switch_rate": float(real_switch),
        "syn_switch_rate": float(syn_switch),
        "switch_rate_abs_err": float(abs(syn_switch - real_switch)),
        "real_active_near_driver_frac": float(_active_near_driver_fraction(real_reg, real_drv, states=(2,), radius=int(CFG12["entity_regime_driver_radius"]))),
        "syn_active_near_driver_frac": float(_active_near_driver_fraction(syn_reg, syn_drv, states=(2,), radius=int(CFG12["entity_regime_driver_radius"]))),
        "real_nonidle_near_driver_frac": float(_active_near_driver_fraction(real_reg, real_drv, states=(2, 3), radius=int(CFG12["entity_regime_driver_radius"]))),
        "syn_nonidle_near_driver_frac": float(_active_near_driver_fraction(syn_reg, syn_drv, states=(2, 3), radius=int(CFG12["entity_regime_driver_radius"]))),
    })

REGIME_PATCH_AUDIT_DF = pd.DataFrame(_regime_rows)
REGIME_PATCH_AUDIT_CSV = os.path.join(REPORT_DIR, "cell12b_v27_1_entity_regime_audit.csv")
REGIME_PATCH_AUDIT_DF.to_csv(REGIME_PATCH_AUDIT_CSV, index=False)

regime_good_n = int((pd.to_numeric(REGIME_PATCH_AUDIT_DF["abs_err_sum"], errors="coerce") < 0.25).sum())
regime_raw_collapsed_n = int(REGIME_PATCH_AUDIT_DF["raw_dom_collapsed_ge_0_98"].sum())
regime_mean_abs_err = float(pd.to_numeric(REGIME_PATCH_AUDIT_DF["abs_err_sum"], errors="coerce").mean())
regime_max_occ_mae = float(pd.to_numeric(REGIME_PATCH_AUDIT_DF["max_occ_abs_err"], errors="coerce").mean())

log(
    f"[Cell12.b] Entity regimes complete | "
    f"good(abs_err_sum<0.25)={regime_good_n}/{len(ENTITY_LIST)} | "
    f"raw_dom_collapsed={regime_raw_collapsed_n}/{len(ENTITY_LIST)} | "
    f"mean_abs_err_sum={regime_mean_abs_err:.6f} | audit={REGIME_PATCH_AUDIT_CSV}"
)

# ----------------------------------------------------------
# Value availability modelling
# ----------------------------------------------------------
VALUE_TO_EXPLICIT_MASK = {}
for v in CONT_VALUE_COLS:
    base = str(v).replace("__value", "")
    mcol = None
    for cand in [f"{base}__obs_present", f"{base}__stale_flag"]:
        if cand in CONT_MASK_COLS:
            mcol = cand
            break
    VALUE_TO_EXPLICIT_MASK[v] = mcol


def _value_rate_trainval(col):
    tr = pd.to_numeric(df_tr[col], errors="coerce").notna().to_numpy(dtype=np.float32)
    va = pd.to_numeric(df_val[col], errors="coerce").notna().to_numpy(dtype=np.float32)
    return float(np.concatenate([tr, va], axis=0).mean())


def _value_rate_train(col):
    return float(pd.to_numeric(df_tr[col], errors="coerce").notna().mean())


def _value_rate_val(col):
    return float(pd.to_numeric(df_val[col], errors="coerce").notna().mean())


def _is_stateful_carried_value(col):
    s = str(col).lower()
    return any(k in s for k in [
        "__light__state__brightness",
        "__light__state__color_temp",
        "__light__state__color_temp_kelvin",
        "__light__state__rgb_color",
        "__media_player__state__volume_level",
        "__number__state__value",
        "power_protection",
        "time_left",
        "fill_quantity",
        "__sensor__water__value",
        "__sensor__programme_progress__value",
    ]) or (s.startswith("iot__withings_") and "__sensor__state__value" in s)


def _is_persisted_accumulator_value(col):
    s = str(col).lower()
    return any(k in s for k in [
        "today_s_consumption",
        "this_month_s_consumption",
        "total_consumption",
        "total__sensor__consumption",
        "today_s__sensor__consumption",
        "this_month_s__sensor__consumption",
        "coffee_and_milk_cups",
        "hot_water_cups",
        "__sensor__cups__value",
        "coffees",
        "total_cleaning_area",
        "total_cleaning_time",
        "total_cleaning_count",
        "cleaning_area",
        "cleaning_time",
    ])


def _is_weather_dense_scalar(col):
    s = str(col).lower()
    return any(k in s for k in [
        "rain_gauge_precipitation__value",
        "rain_gauge_precipitation_last_hour__value",
        "rain_gauge_precipitation_today__value",
        "anemometer_gust_angle__value",
        "anemometer_wind_angle__value",
        "anemometer_gust_strength__value",
        "anemometer_wind_speed__value",
        "__sensor__noise__value",
    ])


def _is_protected_telemetry_value(col, fam, rate):
    s = str(col).lower()
    return bool(
        fam == "device_telemetry"
        and rate >= 0.40
        and any(k in s for k in [
            "linkquality",
            "rssi",
            "lqi",
            "signal_strength",
            "signal_level",
            "voltage",
            "battery",
            "device_temperature",
        ])
    )


def _value_availability_class(col, rate_trainval, fam):
    s = str(col).lower()
    rate = float(rate_trainval)

    if VALUE_TO_EXPLICIT_MASK.get(col) is not None:
        return "explicit_mask"

    if rate >= 0.999:
        return "deterministic_dense"

    if _is_persisted_accumulator_value(col):
        return "persisted_accumulator"

    if _is_stateful_carried_value(col):
        return "stateful_carried_value"

    if "s5_max" in s:
        if fam == "cumulative_count":
            return "persisted_accumulator"
        return "stateful_carried_value"

    if _is_weather_dense_scalar(col) and rate >= 0.90:
        return "weather_dense_scalar"

    if _is_protected_telemetry_value(col, fam, rate):
        return "protected_telemetry_value"

    if rate >= float(CFG12["dense_rate_threshold"]):
        return "near_dense"

    if fam == "step_progress":
        return "episode_sparse"

    if fam == "cumulative_count":
        return "event_counter_state"

    if rate <= float(CFG12["very_sparse_rate_threshold"]):
        return "very_sparse"

    if rate <= float(CFG12["sparse_rate_threshold"]):
        return "sparse"

    return "mixed"


def _fit_value_availability_generators():
    meta = {}
    entity_reporting = {}

    for c in CONT_VALUE_COLS:
        entity = _resolve_entity_for_any_col(c)
        fam = FAMILY_MAP.get(c, "other")

        tr_mask = pd.to_numeric(df_tr[c], errors="coerce").notna().to_numpy(dtype=np.int8)
        va_mask = pd.to_numeric(df_val[c], errors="coerce").notna().to_numpy(dtype=np.int8)
        full_mask = np.concatenate([tr_mask, va_mask], axis=0).astype(np.int8)

        rate_tr = float(tr_mask.mean())
        rate_val = float(va_mask.mean())
        rate_trainval = float(full_mask.mean())

        mode = _value_availability_class(c, rate_trainval, fam)

        piece_bank = _build_piece_bank(
            full_mask,
            name=f"value_avail::{c}",
            piece_min=int(CFG12["value_avail_piece_min"]),
            piece_max=int(CFG12["value_avail_piece_max"]),
            n_draws=int(CFG12["value_avail_piece_draws"]),
        )

        meta[c] = {
            "col": c,
            "entity": entity,
            "entity_class": ENTITY_PROFILES.get(entity, {}).get("entity_class", "unknown"),
            "family": fam,
            "mode": mode,
            "rate_tr": rate_tr,
            "rate_val": rate_val,
            "rate_trainval": rate_trainval,
            "explicit_mask_col": VALUE_TO_EXPLICIT_MASK.get(c),
            "full_mask_trainval": full_mask.astype(np.int8),
            "piece_bank_trainval": piece_bank,
        }

        entity_reporting[c] = {
            "entity": entity,
            "entity_class": ENTITY_PROFILES.get(entity, {}).get("entity_class", "unknown"),
            "value_family": fam,
            "availability_class": mode,
            "rate_tr": rate_tr,
            "rate_val": rate_val,
            "rate_trainval": rate_trainval,
        }

    return meta, entity_reporting


VALUE_AVAIL_META, ENTITY_REPORTING_MODELS = _fit_value_availability_generators()
avail_mode_counts = pd.Series([VALUE_AVAIL_META[c]["mode"] for c in CONT_VALUE_COLS]).value_counts().to_dict()

log(f"[Cell12.b] Value availability classes: {avail_mode_counts}")


def _generate_value_availability():
    out = pd.DataFrame(index=df_te.index)
    mode_map = {}
    rate_map = {}
    rows = []

    for c in CONT_VALUE_COLS:
        meta = VALUE_AVAIL_META[c]
        mode = str(meta["mode"])
        target = float(meta["rate_trainval"])
        rrng = _rng_for(f"value_avail_generate::{c}", 4300)
        mode_map[c] = mode

        if mode == "explicit_mask":
            mcol = meta["explicit_mask_col"]
            if mcol in IOT_SYN_MASKS.columns:
                arr = _as_binary01(IOT_SYN_MASKS[mcol].to_numpy(dtype=np.float32))
            else:
                arr = np.ones(N_TE, dtype=np.float32)

        elif target >= 0.999:
            arr = np.ones(N_TE, dtype=np.float32)

        elif target <= 1e-9:
            arr = np.zeros(N_TE, dtype=np.float32)

        else:
            arr = _assemble_from_piece_bank(
                meta.get("piece_bank_trainval", []),
                N_TE,
                rrng,
                fallback_rate=target,
            )

            if bool(CFG12.get("value_avail_exact_rate_correct", True)):
                arr = _exact_rate_correct_binary(arr, target, rrng)

        out[c] = arr.astype(np.float32)
        rate_map[c] = float(out[c].mean())

        rows.append({
            "col": c,
            "entity": meta["entity"],
            "entity_class": meta["entity_class"],
            "family": meta["family"],
            "mode": mode,
            "rate_tr": float(meta["rate_tr"]),
            "rate_val": float(meta["rate_val"]),
            "rate_trainval_target": float(target),
            "rate_syn": float(rate_map[c]),
            "abs_err_vs_trainval_target": float(abs(rate_map[c] - target)),
            "explicit_mask_col": meta.get("explicit_mask_col", None),
        })

    audit = pd.DataFrame(rows)
    return out.astype(np.float32), mode_map, rate_map, audit


VALUE_AVAIL_SYN, VALUE_AVAIL_MODE, VALUE_AVAIL_RATE_MAP, VALUE_AVAIL_AUDIT_DF = _generate_value_availability()

AVAILABILITY_PATCH_AUDIT_DF = VALUE_AVAIL_AUDIT_DF.copy()
AVAILABILITY_PATCH_AUDIT_CSV = os.path.join(REPORT_DIR, "cell12b_v27_1_value_availability_audit.csv")
VALUE_AVAIL_AUDIT_DF.to_csv(AVAILABILITY_PATCH_AUDIT_CSV, index=False)

availability_rate_mae = float(pd.to_numeric(VALUE_AVAIL_AUDIT_DF["abs_err_vs_trainval_target"], errors="coerce").mean())

log(
    f"[Cell12.b] Value availability complete | "
    f"rate_MAE_vs_TRAINVAL_target={availability_rate_mae:.6f} | "
    f"audit={AVAILABILITY_PATCH_AUDIT_CSV}"
)

# ----------------------------------------------------------
# Binary synthesis
# ----------------------------------------------------------
def _is_reporting_indicator_binary(col):
    s = str(col).lower()
    return s.startswith("telemetry_in_sec__entity__") or s.startswith("events_in_sec__entity__")


def _is_persistent_switch_state(col):
    s = str(col).lower()
    return "__switch__state__state" in s or "__switch__state__value" in s


def _is_mi_box_remote_state(col):
    s = str(col).lower()
    return "mi_box" in s and ("__remote__state__state" in s or "__remote__state__value" in s)


def _binary_semantic_subtype(col, mean_rate, run1_mean, run0_mean, transition_rate):
    s = str(col).lower()

    if mean_rate >= 0.999:
        return "constant_one"
    if mean_rate <= 0.001:
        return "constant_zero"

    if _is_reporting_indicator_binary(col):
        return "reporting_indicator_binary"

    if _is_mi_box_remote_state(col):
        return "mi_box_remote_state"

    if _is_persistent_switch_state(col):
        return "persistent_switch_state"

    if "cloud_connection" in s:
        return "intermittent_config_state"

    event_kw = [
        "motion", "occupancy", "presence", "opening", "door", "window",
        "tamper", "vibration", "button", "press", "trigger", "update"
    ]
    config_kw = [
        "led", "auto_update", "connectivity", "enabled", "available", "cloud_connection"
    ]

    if any(k in s for k in config_kw):
        if mean_rate >= 0.85:
            return "persistent_high_config"
        return "intermittent_config_state"

    if any(k in s for k in event_kw):
        if mean_rate <= 0.08:
            return "rare_event_binary"
        return "moderate_event_binary"

    if transition_rate <= 1e-5 and mean_rate >= 0.50:
        return "persistent_high"
    if transition_rate <= 1e-5 and mean_rate < 0.50:
        return "persistent_low"

    if mean_rate >= 0.90 and run1_mean >= 256:
        return "persistent_high"
    if mean_rate <= 0.10 and run0_mean >= 256:
        return "persistent_low"
    if max(run1_mean, run0_mean) >= 256:
        return "persistent_state"

    return "operational_state"


def _fit_binary_models():
    models = {}

    for c in BIN_COLS:
        entity = _resolve_entity_for_any_col(c)

        tr = pd.to_numeric(X_bin_tr[c], errors="coerce").fillna(0.0).clip(0, 1).to_numpy(dtype=np.float32)
        va = pd.to_numeric(X_bin_val[c], errors="coerce").fillna(0.0).clip(0, 1).to_numpy(dtype=np.float32)
        full = _as_binary01(np.concatenate([tr, va], axis=0)).astype(np.int8)

        rate_tr = float(_binary_rate(tr))
        rate_val = float(_binary_rate(va))
        rate_trainval = float(_binary_rate(full))
        run1_mean = _mean_run_length_binary(full, 1)
        run0_mean = _mean_run_length_binary(full, 0)
        tr_rate = _transition_rate(full)

        subtype = _binary_semantic_subtype(
            c,
            mean_rate=rate_trainval,
            run1_mean=run1_mean,
            run0_mean=run0_mean,
            transition_rate=tr_rate,
        )

        reg_full = ENTITY_REGIME_MODELS[entity]["regime_trainval"]
        drv_full = ENTITY_REGIME_MODELS[entity]["drv_full"]

        piece_bank = _build_piece_bank(
            full,
            name=f"binary::{c}",
            piece_min=int(CFG12["binary_piece_min"]),
            piece_max=int(CFG12["binary_piece_max"]),
            n_draws=int(CFG12["binary_piece_draws"]),
        )

        regime_banks = {}
        for st in range(4):
            idx = np.where(reg_full == st)[0]
            if len(idx) >= 8:
                regime_banks[st] = full[idx].astype(np.int8)
            else:
                regime_banks[st] = full.astype(np.int8)

        models[c] = {
            "col": c,
            "entity": entity,
            "entity_class": ENTITY_PROFILES.get(entity, {}).get("entity_class", "unknown"),
            "rate_tr": rate_tr,
            "rate_val": rate_val,
            "rate_trainval": rate_trainval,
            "run1_mean": float(run1_mean),
            "run0_mean": float(run0_mean),
            "transition_rate_trainval": float(tr_rate),
            "series_trainval": full.astype(np.int8),
            "regime_trainval": reg_full.astype(np.int8),
            "drv_trainval": drv_full.astype(np.int8),
            "piece_bank_trainval": piece_bank,
            "regime_banks": regime_banks,
            "subtype": subtype,
        }

    return models


BIN_MODELS = _fit_binary_models()


def _generate_binary_column_from_regime(col, meta, rrng):
    subtype = meta["subtype"]
    target = float(meta["rate_trainval"])
    entity = meta["entity"]
    regime_syn = np.asarray(ENTITY_REGIME_SYN[entity], dtype=np.int8)
    drv_syn = np.asarray(ENTITY_DRIVER_ACTIVITY_SYN[entity], dtype=np.int8)

    if subtype == "constant_one":
        return np.ones(N_TE, dtype=np.float32)

    if subtype == "constant_zero":
        return np.zeros(N_TE, dtype=np.float32)

    if subtype in {
        "persistent_high_config",
        "intermittent_config_state",
        "persistent_switch_state",
        "mi_box_remote_state",
        "persistent_high",
        "persistent_low",
        "persistent_state",
        "operational_state",
    }:
        arr = _assemble_from_piece_bank(
            meta.get("piece_bank_trainval", []),
            N_TE,
            rrng,
            fallback_rate=target,
        )
        arr = _as_binary01(arr)

        if subtype == "persistent_switch_state":
            arr = np.maximum(arr, (rrng.random(N_TE) < max(target, 0.95)).astype(np.float32))
        elif subtype == "persistent_high_config":
            arr = np.maximum(arr, (rrng.random(N_TE) < max(target, 0.85)).astype(np.float32))
        elif subtype == "persistent_low":
            arr = np.minimum(arr, (rrng.random(N_TE) < min(max(target, 0.01), 0.25)).astype(np.float32))

        if bool(CFG12.get("binary_exact_rate_correct", True)):
            arr = _exact_rate_correct_binary(arr, target, rrng)

        return arr.astype(np.float32)

    if subtype == "reporting_indicator_binary":
        arr = _assemble_from_piece_bank(
            meta.get("piece_bank_trainval", []),
            N_TE,
            rrng,
            fallback_rate=target,
        )
        arr = _as_binary01(arr)

        if bool(CFG12.get("binary_reporting_indicator_apply_global_mask", False)):
            arr = np.minimum(arr, GLOBAL_IOT_MASK_SYN).astype(np.float32)

        if bool(CFG12.get("binary_exact_rate_correct", True)):
            arr = _exact_rate_correct_binary(arr, target, rrng)

        if bool(CFG12.get("binary_reporting_indicator_apply_global_mask", False)):
            arr = np.minimum(arr, GLOBAL_IOT_MASK_SYN).astype(np.float32)

        return arr.astype(np.float32)

    if subtype in {"rare_event_binary", "moderate_event_binary"}:
        arr = np.zeros(N_TE, dtype=np.float32)

        base_p = max(target, 1e-8)
        t = 0
        while t < N_TE:
            st = int(regime_syn[t])
            drv_on = bool(drv_syn[t] > 0)

            if subtype == "rare_event_binary":
                if drv_on or st == 2:
                    p = min(0.15, max(4.0 * base_p, 0.002))
                elif st == 3:
                    p = min(0.05, max(2.0 * base_p, 0.001))
                else:
                    p = min(0.01, max(0.35 * base_p, 0.0001))
            else:
                if drv_on or st == 2:
                    p = min(0.35, max(2.0 * base_p, 0.005))
                elif st == 3:
                    p = min(0.20, max(1.5 * base_p, 0.002))
                else:
                    p = min(0.10, max(0.75 * base_p, 0.001))

            if rrng.random() < p:
                L = int(max(1, min(int(CFG12["binary_rare_event_run_cap"]), rrng.poisson(2) + 1)))
                arr[t:min(N_TE, t + L)] = 1.0
                t += L
            else:
                t += 1

        if bool(CFG12.get("binary_exact_rate_correct", True)):
            arr = _exact_rate_correct_binary(arr, target, rrng)

        return arr.astype(np.float32)

    arr = _assemble_from_piece_bank(
        meta.get("piece_bank_trainval", []),
        N_TE,
        rrng,
        fallback_rate=target,
    )

    if bool(CFG12.get("binary_exact_rate_correct", True)):
        arr = _exact_rate_correct_binary(arr, target, rrng)

    return arr.astype(np.float32)


def _generate_binary_states():
    out = pd.DataFrame(index=df_te.index)
    rows = []

    for c in BIN_COLS:
        meta = BIN_MODELS[c]
        rrng = _rng_for(f"binary_generate::{c}", 5200)
        arr = _generate_binary_column_from_regime(c, meta, rrng)

        out[c] = arr.astype(np.float32)

        syn_rate = float(out[c].mean())
        target = float(meta["rate_trainval"])

        rows.append({
            "col": c,
            "entity": meta["entity"],
            "entity_class": meta["entity_class"],
            "subtype": meta["subtype"],
            "rate_tr": float(meta["rate_tr"]),
            "rate_val": float(meta["rate_val"]),
            "rate_trainval_target": float(target),
            "rate_syn": float(syn_rate),
            "abs_err_vs_trainval_target": float(abs(syn_rate - target)),
            "run1_mean_trainval": float(meta["run1_mean"]),
            "run0_mean_trainval": float(meta["run0_mean"]),
            "transition_rate_trainval": float(meta["transition_rate_trainval"]),
            "transition_rate_syn": float(_transition_rate(arr)),
        })

    audit = pd.DataFrame(rows)
    return out[BIN_COLS].astype(np.float32), audit


IOT_SYN_BINARY, BINARY_PATCH_AUDIT_DF = _generate_binary_states()

BINARY_PATCH_AUDIT_CSV = os.path.join(REPORT_DIR, "cell12b_v27_1_binary_audit.csv")
BINARY_PATCH_AUDIT_DF.to_csv(BINARY_PATCH_AUDIT_CSV, index=False)

bin_rate_mae = float(pd.to_numeric(BINARY_PATCH_AUDIT_DF["abs_err_vs_trainval_target"], errors="coerce").mean())
binary_flagged_bad_cols = int((pd.to_numeric(BINARY_PATCH_AUDIT_DF["abs_err_vs_trainval_target"], errors="coerce") >= 0.05).sum())

log(
    f"[Cell12.b] Binary synthesis complete | "
    f"rate_MAE_vs_TRAINVAL_target={bin_rate_mae:.6f} | "
    f"flagged_bad_cols(abs_err>=0.05)={binary_flagged_bad_cols} | "
    f"audit={BINARY_PATCH_AUDIT_CSV}"
)

# ----------------------------------------------------------
# Entity binary group metadata for modular 12.c
# ----------------------------------------------------------
def _infer_binary_groups():
    groups = {}

    for e in ENTITY_LIST:
        cols = [c for c in ENTITY_TO_BIN_CANON.get(e, []) if c in TRAINVAL_BIN.columns]

        if not cols:
            groups[e] = {
                "cols": [],
                "exclusive_pairs": [],
                "coactive_pairs": [],
                "group_templates_n": 0,
            }
            continue

        B = TRAINVAL_BIN[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1).to_numpy(dtype=np.float32)

        if B.shape[1] == 1:
            corr = np.ones((1, 1), dtype=np.float32)
        else:
            corr = np.corrcoef(B.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        exclusive_pairs = []
        coactive_pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                cij = float(corr[i, j])
                if cij <= -0.35:
                    exclusive_pairs.append((cols[i], cols[j], cij))
                elif cij >= 0.55:
                    coactive_pairs.append((cols[i], cols[j], cij))

        groups[e] = {
            "cols": cols,
            "exclusive_pairs": exclusive_pairs,
            "coactive_pairs": coactive_pairs,
            "group_templates_n": int(len(_make_segments_from_state(ENTITY_REGIME_MODELS[e]["regime_trainval"]))),
        }

    return groups


ENTITY_BINARY_GROUPS = _infer_binary_groups()

# ----------------------------------------------------------
# Portfolio eligibility metadata for modular Cell 12.c
# ----------------------------------------------------------
if "CONT_ROUTING_POLICY_DF" not in globals() or not isinstance(CONT_ROUTING_POLICY_DF, pd.DataFrame):
    raise RuntimeError("[Cell12.b] CONT_ROUTING_POLICY_DF is required from Cell 12.a.")

_policy_required = {
    "col",
    "policy_bucket",
    "policy_generator_family",
    "expected_stage12b_route",
    "expected_stage12c_kind",
}
missing_policy_cols = sorted(_policy_required - set(CONT_ROUTING_POLICY_DF.columns))
if missing_policy_cols:
    raise RuntimeError(f"[Cell12.b] CONT_ROUTING_POLICY_DF missing required columns: {missing_policy_cols}")

_POLICY_DF = (
    CONT_ROUTING_POLICY_DF.copy()
    .drop_duplicates(subset=["col"], keep="last")
    .reset_index(drop=True)
)

missing_policy_rows = sorted(set(CONT_VALUE_COLS) - set(_POLICY_DF["col"].astype(str).tolist()))
if missing_policy_rows:
    raise RuntimeError(
        f"[Cell12.b] Missing routing-policy rows for continuous columns: "
        f"{missing_policy_rows[:20]} (n={len(missing_policy_rows)})"
    )

def _eligible_generators_for_col(col):
    row = _POLICY_DF.loc[_POLICY_DF["col"].astype(str) == str(col)].iloc[0]
    bucket = str(row["policy_bucket"])
    kind = str(row["expected_stage12c_kind"])
    fam = FAMILY_MAP.get(col, "other")
    s = str(col).lower()

    eligible = ["A1_temporal_block_bootstrap"]

    # DDPM is optional, not privileged.
    if str(row["expected_stage12b_route"]).strip() == "ddpm":
        eligible.append("DDPM_conditional_sequence")

    if bucket == "ddpm":
        if fam in {"temperature", "humidity", "pressure", "environmental", "light"}:
            eligible.extend(["GaussianCopula_continuous", "TimeGAN_sequence"])
        elif fam in {"power_energy"}:
            eligible.extend(["CTGAN_TVAE_mixed_tabular", "GaussianCopula_continuous"])
        else:
            eligible.append("GaussianCopula_continuous")

    elif bucket == "telemetry":
        if any(k in s for k in ["linkquality", "rssi", "lqi", "signal_strength", "signal_level"]):
            eligible.extend(["Markov_SemiMarkov_ordinal_dwell", "A1_support_preserving_replay"])
        elif any(k in s for k in ["battery", "voltage", "device_temperature"]):
            eligible.extend(["A1_support_preserving_replay", "GaussianCopula_continuous"])
        else:
            eligible.append("A1_support_preserving_replay")

    elif bucket == "countlike":
        eligible.extend(["NegativeBinomial_PoissonGamma", "Markov_SemiMarkov_counter_state"])

    elif bucket == "discrete_state":
        eligible.extend(["Markov_SemiMarkov_ordinal_dwell", "A1_support_preserving_replay"])

    elif bucket == "step_progress":
        eligible.append("Semantic_step_progress_generator")

    elif bucket == "family_specialized":
        pg = str(row.get("policy_generator_family", ""))
        if pg:
            eligible.append(pg)
        else:
            eligible.append("Semantic_family_specialized_generator")

    elif bucket == "setting_like_constant":
        eligible.append("ConstantOrQuasiStaticReplay")

    else:
        eligible.extend(["ContextContinuousReplay", "GaussianCopula_continuous"])

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for g in eligible:
        if g not in seen:
            out.append(g)
            seen.add(g)
    return out


portfolio_rows = []
routing_rows = []

old_ddpm_set = set(globals().get("DDPM_COLS", []))

for c in CONT_VALUE_COLS:
    prow = _POLICY_DF.loc[_POLICY_DF["col"].astype(str) == str(c)].iloc[0]
    eligible = _eligible_generators_for_col(c)
    is_ddpm_candidate = "DDPM_conditional_sequence" in eligible

    portfolio_rows.append({
        "col": c,
        "entity": _resolve_entity_for_any_col(c),
        "family": FAMILY_MAP.get(c, "other"),
        "policy_bucket": str(prow["policy_bucket"]),
        "policy_generator_family": str(prow["policy_generator_family"]),
        "expected_stage12b_route": str(prow["expected_stage12b_route"]),
        "expected_stage12c_kind": str(prow["expected_stage12c_kind"]),
        "availability_mode": VALUE_AVAIL_MODE.get(c, ""),
        "availability_rate_target_trainval": float(VALUE_AVAIL_META[c]["rate_trainval"]),
        "eligible_generators": "|".join(eligible),
        "eligible_generators_n": int(len(eligible)),
        "ddpm_is_optional_candidate": bool(is_ddpm_candidate),
        "test_used_for_eligibility": False,
    })

    routing_rows.append({
        "col": c,
        "family": FAMILY_MAP.get(c, "other"),
        "policy_bucket": str(prow["policy_bucket"]),
        "expected_stage12c_kind": str(prow["expected_stage12c_kind"]),
        "was_ddpm": bool(c in old_ddpm_set),
        "is_ddpm": bool(is_ddpm_candidate),
        "dropped_from_ddpm": bool((c in old_ddpm_set) and not is_ddpm_candidate),
        "promoted_to_ddpm": bool((c not in old_ddpm_set) and is_ddpm_candidate),
        "routing_reason": "portfolio_eligibility_from_cell12a_policy_not_selection",
    })

IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF = pd.DataFrame(portfolio_rows)
IOT_VALUE_PORTFOLIO_ELIGIBILITY_CSV = os.path.join(REPORT_DIR, "cell12b_v27_1_iot_value_portfolio_eligibility.csv")
IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF.to_csv(IOT_VALUE_PORTFOLIO_ELIGIBILITY_CSV, index=False)

PATCH12B_DDPM_ROUTING_DF = pd.DataFrame(routing_rows)
DDPM_ROUTING_PATCH_DF = PATCH12B_DDPM_ROUTING_DF.copy()
DDPM_ROUTING_PATCH_CSV = os.path.join(REPORT_DIR, "cell12b_v27_1_portfolio_ddpm_candidate_routing.csv")
PATCH12B_DDPM_ROUTING_DF.to_csv(DDPM_ROUTING_PATCH_CSV, index=False)

# Update DDPM_COLS/NON_DDPM_COLS as candidate-eligibility labels only.
# These are NOT selected generators.
old_DDPM_COLS = list(globals().get("DDPM_COLS", []))
DDPM_COLS = sorted(PATCH12B_DDPM_ROUTING_DF.loc[PATCH12B_DDPM_ROUTING_DF["is_ddpm"], "col"].astype(str).tolist())
NON_DDPM_COLS = sorted(set(CONT_VALUE_COLS) - set(DDPM_COLS))

portfolio_generator_counts = Counter()
for s in IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF["eligible_generators"].astype(str):
    for g in s.split("|"):
        if g:
            portfolio_generator_counts[g] += 1

log(
    f"[Cell12.b] Portfolio eligibility exported | "
    f"ddpm_optional_candidates={len(DDPM_COLS)} | non_ddpm={len(NON_DDPM_COLS)} | "
    f"generator_counts={dict(portfolio_generator_counts)} | "
    f"audit={IOT_VALUE_PORTFOLIO_ELIGIBILITY_CSV}"
)

# ----------------------------------------------------------
# Scaffolding gates
# ----------------------------------------------------------
gate_rows = []

if availability_rate_mae > float(CFG12["max_value_avail_rate_mae_trainval_target"]):
    gate_rows.append({
        "metric": "availability_rate_mae_trainval_target",
        "value": float(availability_rate_mae),
        "threshold": float(CFG12["max_value_avail_rate_mae_trainval_target"]),
        "severity": "hard_fail" if bool(CFG12.get("publication_strict_scaffold_gate", False)) else "warning",
        "reason": "synthetic value availability rates diverge from TRAIN+VAL scaffold target",
    })

if bin_rate_mae > float(CFG12["max_binary_rate_mae_trainval_target"]):
    gate_rows.append({
        "metric": "binary_rate_mae_trainval_target",
        "value": float(bin_rate_mae),
        "threshold": float(CFG12["max_binary_rate_mae_trainval_target"]),
        "severity": "hard_fail" if bool(CFG12.get("publication_strict_scaffold_gate", False)) else "warning",
        "reason": "synthetic binary rates diverge from TRAIN+VAL scaffold target",
    })

if regime_max_occ_mae > float(CFG12["max_regime_occ_mae"]):
    gate_rows.append({
        "metric": "regime_max_occ_mae",
        "value": float(regime_max_occ_mae),
        "threshold": float(CFG12["max_regime_occ_mae"]),
        "severity": "hard_fail" if bool(CFG12.get("publication_strict_scaffold_gate", False)) else "warning",
        "reason": "entity regime occupancy differs from TRAIN+VAL scaffold target",
    })

SCAFFOLD_GATE_DF = pd.DataFrame(gate_rows) if gate_rows else pd.DataFrame(
    columns=["metric", "value", "threshold", "severity", "reason"]
)
SCAFFOLD_GATE_CSV = os.path.join(REPORT_DIR, "cell12b_v27_1_scaffold_gate.csv")
SCAFFOLD_GATE_DF.to_csv(SCAFFOLD_GATE_CSV, index=False)

if len(SCAFFOLD_GATE_DF):
    for _, r in SCAFFOLD_GATE_DF.iterrows():
        log(
            f"[Cell12.b] scaffold-gate | metric={r['metric']} | "
            f"value={float(r['value']):.6f} | threshold={float(r['threshold']):.6f} | "
            f"severity={r['severity']} | reason={r['reason']}"
        )

hard_fail_df = SCAFFOLD_GATE_DF.loc[SCAFFOLD_GATE_DF["severity"].astype(str).eq("hard_fail")].copy()
if len(hard_fail_df):
    raise RuntimeError(f"[Cell12.b] Scaffold gate failed in publication-strict mode. See: {SCAFFOLD_GATE_CSV}")

# ----------------------------------------------------------
# Save scaffold artifacts
# ----------------------------------------------------------
IOT_SYN_MASKS_PATH = os.path.join(OUT_SYN, "IOT_SYN_MASKS.parquet")
IOT_SYN_BINARY_PATH = os.path.join(OUT_SYN, "IOT_SYN_BINARY.parquet")
VALUE_AVAIL_SYN_PATH = os.path.join(OUT_SYN, "IOT_VALUE_AVAIL_SYN.parquet")

# Phase-1 role-family handoff artifacts
IOT_CONT_VALUE_MASK_TEST_PATH = os.path.join(OUT_SYN, "IOT_CONT_VALUE_MASK_TEST.parquet")
IOT_BINARY_TARGET_CONTRACT_PATH = os.path.join(ART_DIR, "IOT_BINARY_TARGET_CONTRACT.json")
IOT_DRIVER_TARGET_CONTRACT_PATH = os.path.join(ART_DIR, "IOT_DRIVER_TARGET_CONTRACT.json")
IOT_OBSERVABILITY_CONTRACT_PATH = os.path.join(ART_DIR, "IOT_OBSERVABILITY_CONTRACT.json")
IOT_ROLE_READINESS_AUDIT_CSV = os.path.join(REPORT_DIR, "cell12b_iot_role_readiness_audit.csv")

IOT_SYN_MASKS.to_parquet(IOT_SYN_MASKS_PATH, index=False)
IOT_SYN_BINARY.to_parquet(IOT_SYN_BINARY_PATH, index=False)
VALUE_AVAIL_SYN.to_parquet(VALUE_AVAIL_SYN_PATH, index=False)

# Canonical Phase-1 continuous-value mask handoff.
# This is intentionally an alias of VALUE_AVAIL_SYN with a clearer downstream contract name.
# It contains TEST-length synthetic availability/mask indicators for the 148 continuous IoT value targets.
IOT_CONT_VALUE_MASK_TEST = VALUE_AVAIL_SYN.loc[:, CONT_VALUE_COLS].copy().astype(np.float32)
IOT_CONT_VALUE_MASK_TEST.to_parquet(IOT_CONT_VALUE_MASK_TEST_PATH, index=False)

# Regime matrix export
ENTITY_REGIME_SYN_DF = pd.DataFrame(
    {e: np.asarray(ENTITY_REGIME_SYN[e], dtype=np.int8) for e in ENTITY_LIST},
    index=df_te.index,
)
ENTITY_REGIME_SYN_PATH = os.path.join(OUT_SYN, "IOT_ENTITY_REGIME_SYN.parquet")
ENTITY_REGIME_SYN_DF.to_parquet(ENTITY_REGIME_SYN_PATH, index=False)

# Driver activity export
ENTITY_DRIVER_ACTIVITY_SYN_DF = pd.DataFrame(
    {e: np.asarray(ENTITY_DRIVER_ACTIVITY_SYN[e], dtype=np.int8) for e in ENTITY_LIST},
    index=df_te.index,
)
ENTITY_DRIVER_ACTIVITY_SYN_PATH = os.path.join(OUT_SYN, "IOT_ENTITY_DRIVER_ACTIVITY_SYN.parquet")
ENTITY_DRIVER_ACTIVITY_SYN_DF.to_parquet(ENTITY_DRIVER_ACTIVITY_SYN_PATH, index=False)

log(f"[Cell12.b] Saved scaffold mask artifact: {IOT_SYN_MASKS_PATH}")
log(f"[Cell12.b] Saved scaffold binary artifact: {IOT_SYN_BINARY_PATH}")
log(f"[Cell12.b] Saved scaffold value-availability artifact: {VALUE_AVAIL_SYN_PATH}")
log(f"[Cell12.b] Saved scaffold entity-regime artifact: {ENTITY_REGIME_SYN_PATH}")
log(f"[Cell12.b] Saved scaffold entity-driver activity artifact: {ENTITY_DRIVER_ACTIVITY_SYN_PATH}")
# ----------------------------------------------------------
# Phase-1 role-family handoff contracts
# ----------------------------------------------------------
MASK_OBSERVABILITY_META_COLS = list(globals().get("IOT_MASK_OBSERVABILITY_META_COLS", []))
STALENESS_META_COLS = list(globals().get("IOT_STALENESS_META_COLS", []))
CONDITIONING_META_COLS = list(globals().get("IOT_CONDITIONING_META_COLS", []))
DERIVED_META_COLS = list(globals().get("IOT_DERIVED_META_COLS", []))

if not MASK_OBSERVABILITY_META_COLS:
    MASK_OBSERVABILITY_META_COLS = [c for c in globals().get("IOT_NONMODELED_META_COLS", []) if "obs_present" in str(c).lower()]

if not STALENESS_META_COLS:
    STALENESS_META_COLS = [c for c in globals().get("IOT_NONMODELED_META_COLS", []) if "stale" in str(c).lower()]

mask_meta_ready_n = int(len(MASK_OBSERVABILITY_META_COLS))
staleness_meta_ready_n = int(len(STALENESS_META_COLS))
conditioning_meta_ready_n = int(len(CONDITIONING_META_COLS))
derived_meta_ready_n = int(len(DERIVED_META_COLS))

continuous_value_targets_ready = int(
    len(CONT_VALUE_COLS)
    if isinstance(IOT_CONT_VALUE_MASK_TEST, pd.DataFrame)
    and len(IOT_CONT_VALUE_MASK_TEST) == int(N_TE)
    and list(IOT_CONT_VALUE_MASK_TEST.columns) == list(CONT_VALUE_COLS)
    else 0
)

binary_targets_ready = int(
    len(BIN_COLS)
    if isinstance(IOT_SYN_BINARY, pd.DataFrame)
    and len(IOT_SYN_BINARY) == int(N_TE)
    and list(IOT_SYN_BINARY.columns) == list(BIN_COLS)
    else 0
)

driver_targets_ready = int(
    len(DRV_COLS)
    if isinstance(IOT_SYN_DRIVERS, pd.DataFrame)
    and len(IOT_SYN_DRIVERS) == int(N_TE)
    and list(IOT_SYN_DRIVERS.columns) == list(DRV_COLS)
    else 0
)

_readiness_rows = [
    {
        "role_family": "continuous_value_targets",
        "expected_n": int(len(CONT_VALUE_COLS)),
        "ready_n": int(continuous_value_targets_ready),
        "artifact": IOT_CONT_VALUE_MASK_TEST_PATH,
        "contract": "TEST-length synthetic availability masks for continuous IoT value targets",
        "uses_test_values": False,
        "violation": bool(continuous_value_targets_ready != int(len(CONT_VALUE_COLS))),
    },
    {
        "role_family": "binary_targets",
        "expected_n": int(len(BIN_COLS)),
        "ready_n": int(binary_targets_ready),
        "artifact": IOT_SYN_BINARY_PATH,
        "contract": "TEST-length synthetic strict 0/1 binary IoT target streams",
        "uses_test_values": False,
        "violation": bool(binary_targets_ready != int(len(BIN_COLS))),
    },
    {
        "role_family": "driver_targets",
        "expected_n": int(len(DRV_COLS)),
        "ready_n": int(driver_targets_ready),
        "artifact": "IOT_SYN_DRIVERS global",
        "contract": "TEST-length synthetic IoT event-driver target streams from Cell 12.a",
        "uses_test_values": False,
        "violation": bool(driver_targets_ready != int(len(DRV_COLS))),
    },
    {
        "role_family": "mask_observability_meta",
        "expected_n": int(mask_meta_ready_n),
        "ready_n": int(mask_meta_ready_n),
        "artifact": IOT_OBSERVABILITY_CONTRACT_PATH,
        "contract": "non-modeled observability metadata registered for downstream mask/schema enforcement",
        "uses_test_values": False,
        "violation": False,
    },
    {
        "role_family": "staleness_meta",
        "expected_n": int(staleness_meta_ready_n),
        "ready_n": int(staleness_meta_ready_n),
        "artifact": IOT_OBSERVABILITY_CONTRACT_PATH,
        "contract": "non-modeled staleness metadata registered for downstream staleness/schema enforcement",
        "uses_test_values": False,
        "violation": False,
    },
]

IOT_ROLE_READINESS_AUDIT_DF = pd.DataFrame(_readiness_rows)
role_contract_violations = int(IOT_ROLE_READINESS_AUDIT_DF["violation"].astype(bool).sum())
IOT_ROLE_READINESS_AUDIT_DF.to_csv(IOT_ROLE_READINESS_AUDIT_CSV, index=False)

IOT_BINARY_TARGET_CONTRACT = {
    "version": "cell12b_v27_1_binary_target_contract",
    "role_family": "binary_targets",
    "target_cols_n": int(len(BIN_COLS)),
    "target_cols": list(BIN_COLS),
    "artifact": IOT_SYN_BINARY_PATH,
    "row_count": int(len(IOT_SYN_BINARY)),
    "expected_rows": int(N_TE),
    "dtype_contract": "strict_binary_0_1_float32",
    "generation_boundary": {
        "train_used_for_fitting": True,
        "val_used_for_calibration": True,
        "test_values_used": False,
        "test_used_for_length_index_schema_only": True,
    },
    "audit_csv": BINARY_PATCH_AUDIT_CSV,
    "rate_mae_vs_trainval_target": float(bin_rate_mae),
    "flagged_bad_cols_abs_err_ge_0_05": int(binary_flagged_bad_cols),
}

IOT_DRIVER_TARGET_CONTRACT = {
    "version": "cell12b_v27_1_driver_target_contract",
    "role_family": "driver_targets",
    "target_cols_n": int(len(DRV_COLS)),
    "target_cols": list(DRV_COLS),
    "artifact": "IOT_SYN_DRIVERS global from Cell 12.a",
    "row_count": int(len(IOT_SYN_DRIVERS)),
    "expected_rows": int(N_TE),
    "dtype_contract": "event_driver_numeric_binary_or_countlike_streams",
    "generation_boundary": {
        "generated_in_cell12a": True,
        "consumed_in_cell12b": True,
        "train_used_for_fitting": True,
        "val_used_for_calibration": True,
        "test_values_used": False,
        "test_used_for_length_index_schema_only": True,
    },
    "entity_driver_activity_artifact": ENTITY_DRIVER_ACTIVITY_SYN_PATH,
}

IOT_OBSERVABILITY_CONTRACT = {
    "version": "cell12b_v27_1_observability_staleness_contract",
    "role_family": "observability_and_staleness_meta",
    "explicit_cont_mask_cols_n": int(len(CONT_MASK_COLS)),
    "explicit_cont_mask_cols": list(CONT_MASK_COLS),
    "mask_observability_meta_cols_n": int(mask_meta_ready_n),
    "mask_observability_meta_cols": list(MASK_OBSERVABILITY_META_COLS),
    "staleness_meta_cols_n": int(staleness_meta_ready_n),
    "staleness_meta_cols": list(STALENESS_META_COLS),
    "conditioning_meta_cols_n": int(conditioning_meta_ready_n),
    "conditioning_meta_cols": list(CONDITIONING_META_COLS),
    "derived_meta_cols_n": int(derived_meta_ready_n),
    "derived_meta_cols": list(DERIVED_META_COLS),
    "global_iot_mask_artifact": IOT_SYN_MASKS_PATH,
    "continuous_value_mask_artifact": IOT_CONT_VALUE_MASK_TEST_PATH,
    "value_availability_audit_csv": AVAILABILITY_PATCH_AUDIT_CSV,
    "contract": {
        "meta_columns_are_not_generated_as_targets": True,
        "continuous_value_masks_are_synthetic_test_length": True,
        "test_values_used": False,
        "downstream_cells_must_use_masks_for_schema_enforcement": True,
    },
}

_write_json(IOT_BINARY_TARGET_CONTRACT_PATH, IOT_BINARY_TARGET_CONTRACT)
_write_json(IOT_DRIVER_TARGET_CONTRACT_PATH, IOT_DRIVER_TARGET_CONTRACT)
_write_json(IOT_OBSERVABILITY_CONTRACT_PATH, IOT_OBSERVABILITY_CONTRACT)

if role_contract_violations != 0:
    raise RuntimeError(
        "[Cell12.b] IoT role readiness contract violations detected. "
        f"violations={role_contract_violations} | audit={IOT_ROLE_READINESS_AUDIT_CSV}"
    )

log(f"[Cell12.b] Saved continuous-value mask handoff: {IOT_CONT_VALUE_MASK_TEST_PATH}")
log(f"[Cell12.b] Saved binary target contract: {IOT_BINARY_TARGET_CONTRACT_PATH}")
log(f"[Cell12.b] Saved driver target contract: {IOT_DRIVER_TARGET_CONTRACT_PATH}")
log(f"[Cell12.b] Saved observability/staleness contract: {IOT_OBSERVABILITY_CONTRACT_PATH}")
log(f"[Cell12.b] Saved role readiness audit: {IOT_ROLE_READINESS_AUDIT_CSV}")

log(
    "[Cell12.b] Phase-1 IoT role-family handoff ready | "
    f"continuous_value_targets_ready={continuous_value_targets_ready} | "
    f"binary_targets_ready={binary_targets_ready} | "
    f"driver_targets_ready={driver_targets_ready} | "
    f"mask_meta_ready={mask_meta_ready_n} | "
    f"staleness_meta_ready={staleness_meta_ready_n} | "
    f"role_contract_violations={role_contract_violations} | "
    "TEST_values_used=False"
)
# ----------------------------------------------------------
# Patch summary / manifest
# ----------------------------------------------------------
PATCH12B_SUMMARY = {
    "version": "v27_1_portfolio_neutral_scaffold_only",
    "scientific_boundary": {
        "train_used": True,
        "val_used": True,
        "test_used_for_length_index_schema_only": True,
        "test_values_used_for_fitting": False,
        "test_values_used_for_rates": False,
        "test_values_used_for_generator_selection": False,
        "ddpm_trained_in_cell12b": False,
        "continuous_values_generated_in_cell12b": False,
    },
    "n_rows": {
        "N_TR": int(N_TR),
        "N_VAL": int(N_VAL),
        "N_TRVA": int(N_TRVA),
        "N_TE": int(N_TE),
    },
    "columns": {
        "drivers": int(len(DRV_COLS)),
        "binary": int(len(BIN_COLS)),
        "cont_value": int(len(CONT_VALUE_COLS)),
        "cont_mask": int(len(CONT_MASK_COLS)),
        "mask_observability_meta": int(mask_meta_ready_n),
        "staleness_meta": int(staleness_meta_ready_n),
        "conditioning_meta": int(conditioning_meta_ready_n),
        "derived_meta": int(derived_meta_ready_n),
        "ddpm_optional_candidate_cols": int(len(DDPM_COLS)),
        "non_ddpm_cols": int(len(NON_DDPM_COLS)),
    },
    "phase1_role_handoff": {
        "continuous_value_targets_ready": int(continuous_value_targets_ready),
        "binary_targets_ready": int(binary_targets_ready),
        "driver_targets_ready": int(driver_targets_ready),
        "mask_meta_ready": int(mask_meta_ready_n),
        "staleness_meta_ready": int(staleness_meta_ready_n),
        "role_contract_violations": int(role_contract_violations),
        "test_values_used": False,
        "continuous_value_mask_test_artifact": IOT_CONT_VALUE_MASK_TEST_PATH,
        "binary_target_contract": IOT_BINARY_TARGET_CONTRACT_PATH,
        "driver_target_contract": IOT_DRIVER_TARGET_CONTRACT_PATH,
        "observability_contract": IOT_OBSERVABILITY_CONTRACT_PATH,
        "role_readiness_audit_csv": IOT_ROLE_READINESS_AUDIT_CSV,
    },
    "entity_profiles": {
        "entities": int(len(ENTITY_LIST)),
        "entity_class_counts": entity_class_counts,
    },
    "masks": {
        "explicit_mask_cols": list(CONT_MASK_COLS),
        "mask_rates": mask_rates,
        "mask_fit_rates": mask_fit_rates,
        "artifact": IOT_SYN_MASKS_PATH,
    },
    "regime": {
        "good_entities_abs_err_lt_0_25": int(regime_good_n),
        "raw_dom_collapsed_ge_0_98": int(regime_raw_collapsed_n),
        "mean_abs_err_sum": float(regime_mean_abs_err),
        "mean_max_occ_abs_err": float(regime_max_occ_mae),
        "audit_csv": REGIME_PATCH_AUDIT_CSV,
        "artifact": ENTITY_REGIME_SYN_PATH,
    },
    "availability": {
        "mode_counts": avail_mode_counts,
        "rate_mae_vs_trainval_target": float(availability_rate_mae),
        "audit_csv": AVAILABILITY_PATCH_AUDIT_CSV,
        "artifact": VALUE_AVAIL_SYN_PATH,
    },
    "binary": {
        "rate_mae_vs_trainval_target": float(bin_rate_mae),
        "flagged_bad_cols_abs_err_ge_0_05": int(binary_flagged_bad_cols),
        "audit_csv": BINARY_PATCH_AUDIT_CSV,
        "artifact": IOT_SYN_BINARY_PATH,
    },
    "portfolio_eligibility": {
        "generator_counts": dict(portfolio_generator_counts),
        "eligibility_csv": IOT_VALUE_PORTFOLIO_ELIGIBILITY_CSV,
        "ddpm_candidate_routing_csv": DDPM_ROUTING_PATCH_CSV,
        "selection_deferred_to": "modular Cell 12.c VAL-only portfolio selector",
    },
    "scaffold_gate": {
        "audit_csv": SCAFFOLD_GATE_CSV,
        "rows": SCAFFOLD_GATE_DF.to_dict("records"),
    },
}

PATCH12B_SUMMARY_JSON = os.path.join(REPORT_DIR, "cell12b_v27_1_scaffold_summary.json")
_write_json(PATCH12B_SUMMARY_JSON, PATCH12B_SUMMARY)

IOT_SCAFFOLD_MANIFEST = {
    "version": "cell12b_v27_1_portfolio_neutral_scaffold_only",
    "summary_json": PATCH12B_SUMMARY_JSON,
    "outputs": {
        "IOT_SYN_MASKS": IOT_SYN_MASKS_PATH,
        "IOT_SYN_BINARY": IOT_SYN_BINARY_PATH,
        "VALUE_AVAIL_SYN": VALUE_AVAIL_SYN_PATH,
        "IOT_CONT_VALUE_MASK_TEST": IOT_CONT_VALUE_MASK_TEST_PATH,
        "ENTITY_REGIME_SYN": ENTITY_REGIME_SYN_PATH,
        "ENTITY_DRIVER_ACTIVITY_SYN": ENTITY_DRIVER_ACTIVITY_SYN_PATH,
        "IOT_BINARY_TARGET_CONTRACT": IOT_BINARY_TARGET_CONTRACT_PATH,
        "IOT_DRIVER_TARGET_CONTRACT": IOT_DRIVER_TARGET_CONTRACT_PATH,
        "IOT_OBSERVABILITY_CONTRACT": IOT_OBSERVABILITY_CONTRACT_PATH,
    },
    "reports": {
        "regime_audit": REGIME_PATCH_AUDIT_CSV,
        "value_availability_audit": AVAILABILITY_PATCH_AUDIT_CSV,
        "binary_audit": BINARY_PATCH_AUDIT_CSV,
        "portfolio_eligibility": IOT_VALUE_PORTFOLIO_ELIGIBILITY_CSV,
        "ddpm_candidate_routing": DDPM_ROUTING_PATCH_CSV,
        "scaffold_gate": SCAFFOLD_GATE_CSV,
        "iot_role_readiness_audit": IOT_ROLE_READINESS_AUDIT_CSV,
    },
    "artifact_hashes": {
        "IOT_SYN_MASKS_sha256": _sha256_file(IOT_SYN_MASKS_PATH),
        "IOT_SYN_BINARY_sha256": _sha256_file(IOT_SYN_BINARY_PATH),
        "VALUE_AVAIL_SYN_sha256": _sha256_file(VALUE_AVAIL_SYN_PATH),
        "IOT_CONT_VALUE_MASK_TEST_sha256": _sha256_file(IOT_CONT_VALUE_MASK_TEST_PATH),
        "ENTITY_REGIME_SYN_sha256": _sha256_file(ENTITY_REGIME_SYN_PATH),
        "ENTITY_DRIVER_ACTIVITY_SYN_sha256": _sha256_file(ENTITY_DRIVER_ACTIVITY_SYN_PATH),
        "IOT_BINARY_TARGET_CONTRACT_sha256": _sha256_file(IOT_BINARY_TARGET_CONTRACT_PATH),
        "IOT_DRIVER_TARGET_CONTRACT_sha256": _sha256_file(IOT_DRIVER_TARGET_CONTRACT_PATH),
        "IOT_OBSERVABILITY_CONTRACT_sha256": _sha256_file(IOT_OBSERVABILITY_CONTRACT_PATH),
        "IOT_ROLE_READINESS_AUDIT_sha256": _sha256_file(IOT_ROLE_READINESS_AUDIT_CSV),
        "summary_sha256": _sha256_file(PATCH12B_SUMMARY_JSON),
    },
    "contract": {
        "cell12b_role": "iot_mid_stage_role_family_handoff",
        "continuous_value_generation": "deferred_to_modular_cell12c",
        "continuous_value_mask_generation": "completed_in_cell12b",
        "binary_target_generation": "completed_in_cell12b",
        "driver_target_generation": "completed_in_cell12a_consumed_and_contracted_in_cell12b",
        "observability_staleness_contract": "completed_in_cell12b",
        "generator_selection": "deferred_to_modular_cell12c_VAL_only",
        "test_values_used_for_fitting": False,
        "test_values_used_for_selection": False,
        "test_values_used": False,
        "role_contract_violations": int(role_contract_violations),
    },
}

IOT_SCAFFOLD_MANIFEST_JSON = os.path.join(ART_DIR, "cell12b_iot_scaffold_manifest_v27_1.json")
_write_json(IOT_SCAFFOLD_MANIFEST_JSON, IOT_SCAFFOLD_MANIFEST)

IOT_SCAFFOLD_CONTRACT_PATH = os.path.join(CONTRACT_DIR, "cell12b_iot_scaffold_contract_v27_1_1_THESIS.json")
IOT_SCAFFOLD_CONTRACT = {
    "version": "cell12b_iot_scaffold_contract_v27_1_1_THESIS",
    "component": "iot_scaffold_only",
    "upstream_contracts": {
        "cell12a_foundation_manifest": CELL12A_FOUNDATION_MANIFEST_PATH,
        "cell10_8_selector_contract": CELL10_8_SELECTOR_CONTRACT_PATH,
        "cell10_9_materializer_contract": CELL10_9_MATERIALIZER_CONTRACT_PATH,
        "precell11_runtime_contract": PRECELL11_RUNTIME_CONTRACT_PATH,
        "cell11_final_manifest": CELL11_FINAL_MANIFEST_PATH_REQUIRED,
    },
    "outputs": IOT_SCAFFOLD_MANIFEST["outputs"],
    "reports": IOT_SCAFFOLD_MANIFEST["reports"],
    "role": {
        "continuous_value_generation": "deferred_to_modular_cell12c",
        "binary_target_scaffold": "completed_in_cell12b",
        "continuous_value_mask_generation": "completed_in_cell12b",
        "driver_target_generation": "completed_in_cell12a_consumed_by_cell12b",
        "observability_staleness_contract": "completed_in_cell12b",
        "generator_selection": "none_deferred_to_cell12c_VAL_only",
    },
    "test_usage": {
        "test_length_index_schema_only": True,
        "test_values_used_for_fitting": False,
        "test_values_used_for_selection": False,
        "test_values_used_for_calibration": False,
        "test_values_used_for_repair": False,
        "test_values_used_for_continuous_value_generation": False,
    },
    "target_counts": {
        "drivers": int(len(DRV_COLS)),
        "binary": int(len(BIN_COLS)),
        "continuous_value": int(len(CONT_VALUE_COLS)),
        "continuous_availability_masks": int(len(CONT_VALUE_COLS)),
        "entities": int(len(ENTITY_LIST)),
    },
    "quality_summary": {
        "availability_rate_mae_vs_trainval_target": float(availability_rate_mae),
        "binary_rate_mae_vs_trainval_target": float(bin_rate_mae),
        "binary_flagged_bad_cols_abs_err_ge_0_05": int(binary_flagged_bad_cols),
        "regime_mean_abs_err_sum": float(regime_mean_abs_err),
        "regime_raw_dom_collapsed_n": int(regime_raw_collapsed_n),
        "role_contract_violations": int(role_contract_violations),
    },
    "paper_claim_status": (
        "Cell 12.b is an IoT scaffold/handoff cell. It does not generate continuous IoT "
        "values and does not provide final IoT realism evidence."
    ),
}
_write_json(IOT_SCAFFOLD_CONTRACT_PATH, IOT_SCAFFOLD_CONTRACT)

log(f"[Cell12.b] Patch summary saved: {PATCH12B_SUMMARY_JSON}")
log(f"[Cell12.b] Scaffold manifest saved: {IOT_SCAFFOLD_MANIFEST_JSON}")
log(f"[Cell12.b] Scaffold contract saved: {IOT_SCAFFOLD_CONTRACT_PATH}")

# ----------------------------------------------------------
# Compatibility globals
# ----------------------------------------------------------
# These names are intentionally retained for downstream compatibility,
# but they now represent candidate eligibility / scaffold outputs, not
# selected generator outputs.
REGIME_PATCH_AUDIT_CSV = REGIME_PATCH_AUDIT_CSV
AVAILABILITY_PATCH_AUDIT_DF = VALUE_AVAIL_AUDIT_DF.copy()
AVAILABILITY_PATCH_AUDIT_CSV = AVAILABILITY_PATCH_AUDIT_CSV
BINARY_PATCH_AUDIT_DF = BINARY_PATCH_AUDIT_DF
BINARY_PATCH_AUDIT_CSV = BINARY_PATCH_AUDIT_CSV
globals()["IOT_CONT_VALUE_MASK_TEST"] = IOT_CONT_VALUE_MASK_TEST
globals()["IOT_CONT_VALUE_MASK_TEST_PATH"] = IOT_CONT_VALUE_MASK_TEST_PATH

globals()["IOT_BINARY_TARGET_CONTRACT"] = IOT_BINARY_TARGET_CONTRACT
globals()["IOT_BINARY_TARGET_CONTRACT_PATH"] = IOT_BINARY_TARGET_CONTRACT_PATH

globals()["IOT_DRIVER_TARGET_CONTRACT"] = IOT_DRIVER_TARGET_CONTRACT
globals()["IOT_DRIVER_TARGET_CONTRACT_PATH"] = IOT_DRIVER_TARGET_CONTRACT_PATH

globals()["IOT_OBSERVABILITY_CONTRACT"] = IOT_OBSERVABILITY_CONTRACT
globals()["IOT_OBSERVABILITY_CONTRACT_PATH"] = IOT_OBSERVABILITY_CONTRACT_PATH

globals()["IOT_SCAFFOLD_MANIFEST"] = IOT_SCAFFOLD_MANIFEST
globals()["IOT_SCAFFOLD_MANIFEST_JSON"] = IOT_SCAFFOLD_MANIFEST_JSON
globals()["IOT_SCAFFOLD_CONTRACT"] = IOT_SCAFFOLD_CONTRACT
globals()["IOT_SCAFFOLD_CONTRACT_PATH"] = IOT_SCAFFOLD_CONTRACT_PATH

globals()["IOT_ROLE_READINESS_AUDIT_DF"] = IOT_ROLE_READINESS_AUDIT_DF
globals()["IOT_ROLE_READINESS_AUDIT_CSV"] = IOT_ROLE_READINESS_AUDIT_CSV

globals()["continuous_value_targets_ready"] = continuous_value_targets_ready
globals()["binary_targets_ready"] = binary_targets_ready
globals()["driver_targets_ready"] = driver_targets_ready
globals()["mask_meta_ready_n"] = mask_meta_ready_n
globals()["staleness_meta_ready_n"] = staleness_meta_ready_n
globals()["role_contract_violations"] = role_contract_violations

# Do NOT define syn_ddpm_df / syn_ddpm_val_df here.
# They belong in modular Cell 12.c candidate-generation cells.
for _legacy_ddpm_output in ["syn_ddpm_df", "syn_ddpm_val_df", "ddpm_model", "ddpm", "scaler"]:
    if _legacy_ddpm_output in globals():
        log(
            f"[Cell12.b][WARN] Legacy global {_legacy_ddpm_output!r} already exists from a previous run. "
            "It is not produced by v27.1. Restart kernel or delete stale globals before running modular 12.c."
        )

# ----------------------------------------------------------
# Required output sanity check
# ----------------------------------------------------------
required_12b_outputs = [
    "CANON_ENTITY_MAP",
    "ENTITY_LIST",
    "ENTITY_TO_DRV_CANON",
    "ENTITY_TO_BIN_CANON",
    "ENTITY_TO_CONT_CANON",
    "ENTITY_PROFILES",
    "ENTITY_DRIVER_ACTIVITY_TRVA",
    "ENTITY_DRIVER_ACTIVITY_SYN",
    "VALUE_TO_EXPLICIT_MASK",
    "EXPL_MASK_GENS",
    "IOT_SYN_MASKS",
    "GLOBAL_IOT_MASK_SYN",
    "VALUE_AVAIL_META",
    "VALUE_AVAIL_SYN",
    "VALUE_AVAIL_MODE",
    "VALUE_AVAIL_RATE_MAP",
    "VALUE_AVAIL_AUDIT_DF",
    "ENTITY_REPORTING_MODELS",
    "BIN_MODELS",
    "IOT_SYN_BINARY",
    "ENTITY_REGIME_STATES",
    "ENTITY_REGIME_MODELS",
    "ENTITY_REGIME_SYN",
    "ENTITY_BINARY_GROUPS",
    "REGIME_PATCH_AUDIT_DF",
    "AVAILABILITY_PATCH_AUDIT_DF",
    "BINARY_PATCH_AUDIT_DF",
    "PATCH12B_DDPM_ROUTING_DF",
    "DDPM_ROUTING_PATCH_DF",
    "IOT_VALUE_PORTFOLIO_ELIGIBILITY_DF",
    "IOT_SCAFFOLD_MANIFEST",
    "PATCH12B_SUMMARY",
    "bin_rate_mae",
    "availability_rate_mae",
    "IOT_CONT_VALUE_MASK_TEST",
    "IOT_BINARY_TARGET_CONTRACT",
    "IOT_DRIVER_TARGET_CONTRACT",
    "IOT_OBSERVABILITY_CONTRACT",
    "IOT_ROLE_READINESS_AUDIT_DF",
]

missing_outputs = [k for k in required_12b_outputs if k not in globals()]
if missing_outputs:
    raise RuntimeError(f"[Cell12.b] Missing expected output globals: {missing_outputs}")

# Row/shape contract checks
if len(IOT_SYN_MASKS) != N_TE:
    raise RuntimeError(f"[Cell12.b] IOT_SYN_MASKS row mismatch: {len(IOT_SYN_MASKS)} vs {N_TE}")
if len(IOT_SYN_BINARY) != N_TE:
    raise RuntimeError(f"[Cell12.b] IOT_SYN_BINARY row mismatch: {len(IOT_SYN_BINARY)} vs {N_TE}")
if len(VALUE_AVAIL_SYN) != N_TE:
    raise RuntimeError(f"[Cell12.b] VALUE_AVAIL_SYN row mismatch: {len(VALUE_AVAIL_SYN)} vs {N_TE}")
if len(IOT_CONT_VALUE_MASK_TEST) != N_TE:
    raise RuntimeError(
        f"[Cell12.b] IOT_CONT_VALUE_MASK_TEST row mismatch: "
        f"{len(IOT_CONT_VALUE_MASK_TEST)} vs {N_TE}"
    )
if list(IOT_SYN_BINARY.columns) != list(BIN_COLS):
    raise RuntimeError("[Cell12.b] IOT_SYN_BINARY column order mismatch vs BIN_COLS.")
if list(VALUE_AVAIL_SYN.columns) != list(CONT_VALUE_COLS):
    raise RuntimeError("[Cell12.b] VALUE_AVAIL_SYN column order mismatch vs CONT_VALUE_COLS.")
if list(IOT_CONT_VALUE_MASK_TEST.columns) != list(CONT_VALUE_COLS):
    raise RuntimeError("[Cell12.b] IOT_CONT_VALUE_MASK_TEST column order mismatch vs CONT_VALUE_COLS.")
# Binary value checks
for c in BIN_COLS:
    vals = pd.to_numeric(IOT_SYN_BINARY[c], errors="coerce").dropna().unique()
    bad = [v for v in vals if float(v) not in {0.0, 1.0}]
    if bad:
        raise RuntimeError(f"[Cell12.b] IOT_SYN_BINARY column is not binary 0/1: {c} | bad_preview={bad[:10]}")

for c in CONT_VALUE_COLS:
    vals = pd.to_numeric(VALUE_AVAIL_SYN[c], errors="coerce").dropna().unique()
    bad = [v for v in vals if float(v) not in {0.0, 1.0}]
    if bad:
        raise RuntimeError(f"[Cell12.b] VALUE_AVAIL_SYN column is not binary 0/1: {c} | bad_preview={bad[:10]}")

for c in CONT_VALUE_COLS:
    vals = pd.to_numeric(IOT_CONT_VALUE_MASK_TEST[c], errors="coerce").dropna().unique()
    bad = [v for v in vals if float(v) not in {0.0, 1.0}]
    if bad:
        raise RuntimeError(
            f"[Cell12.b] IOT_CONT_VALUE_MASK_TEST column is not binary 0/1: "
            f"{c} | bad_preview={bad[:10]}"
        )

log("[Cell12.b] Final summary:")
log(f"[Cell12.b]   role=scaffold_only | continuous_values_generated=False | DDPM_trained=False")
log(f"[Cell12.b]   entities={len(ENTITY_LIST)} | entity_classes={entity_class_counts}")
log(f"[Cell12.b]   masks={len(CONT_MASK_COLS)} | mask_rates={mask_rates}")
log(
    f"[Cell12.b]   regimes good={regime_good_n}/{len(ENTITY_LIST)} | "
    f"raw_dom_collapsed={regime_raw_collapsed_n}/{len(ENTITY_LIST)} | "
    f"mean_abs_err_sum={regime_mean_abs_err:.6f}"
)
log(
    f"[Cell12.b]   availability rate_MAE_vs_TRAINVAL={availability_rate_mae:.6f} | "
    f"modes={avail_mode_counts}"
)
log(
    f"[Cell12.b]   binary rate_MAE_vs_TRAINVAL={bin_rate_mae:.6f} | "
    f"flagged_bad_cols={binary_flagged_bad_cols}"
)
log(
    f"[Cell12.b]   portfolio eligibility | "
    f"A1 baseline mandatory for all {len(CONT_VALUE_COLS)} value columns | "
    f"DDPM optional candidates={len(DDPM_COLS)} | "
    f"generator_counts={dict(portfolio_generator_counts)}"
)
log(
    f"[Cell12.b] Ready for modular Cell 12.c | "
    f"value_avail={len(CONT_VALUE_COLS)} | binary={len(BIN_COLS)} | "
    f"entities={len(ENTITY_LIST)} | manifest={IOT_SCAFFOLD_MANIFEST_JSON}"
)

log(
    "[Cell12.b] Phase-1 definition-of-done | "
    f"continuous_value_targets_ready={continuous_value_targets_ready} | "
    f"binary_targets_ready={binary_targets_ready} | "
    f"driver_targets_ready={driver_targets_ready} | "
    f"mask_meta_ready={mask_meta_ready_n} | "
    f"role_contract_violations={role_contract_violations} | "
    "TEST_values_used=False"
)

log("--- END: Cell 12.b — IoT synthesis scaffold stage (v27.1.1-THESIS contract-locked, portfolio-neutral, no DDPM training) ---")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()