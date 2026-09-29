# ==========================================================
# CELL 13.0 - Q4 cross-modal coupling input contract
# v1.1 STUDY-THESIS strict coupling loader, schema-driven pair registry
#
# Role:
#   - Load/validate:
#       1) full synthetic IoT TEST namespace from Cell 12.g
#       2) synthetic protocol/network TEST matrix from Phase 1 / A0-A1-A2
#       3) real TEST references for IoT + protocol
#       4) driver columns from 12.e / 12.g
#   - Discover protocol tiers:
#       router, ota, zigbee, zwave
#   - Build candidate IoT-driver -> protocol metric pairs.
#   - Export strict Cell 13 Q4 input contract.
#
# Strict rules:
#   - This cell may read real TEST references because Cell 13 is QA only.
#   - Do NOT mutate synthetic values.
#   - Do NOT fit generators.
#   - Do NOT select generators.
#   - Do NOT materialize new synthetic values.
#
# Outputs:
#   reports/cell13_0_q4_coupling_input_contract.json
#   reports/cell13_0_q4_coupling_input_audit.csv
#   reports/cell13_0_driver_protocol_pair_registry.csv
#   artifacts/cell13_0_q4_coupling_manifest.json
# ==========================================================

log("--- START: Cell 13.0 - Q4 cross-modal coupling input contract (v1.1 schema-driven strict) ---")

import os
import re
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_130 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL12G_IOT_FULL_NAMESPACE_CONTRACT",
]
_missing_130 = [k for k in _required_130 if k not in globals()]
if _missing_130:
    raise RuntimeError(f"[Cell13.0] Missing required globals: {_missing_130}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(OUT_SYN, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TE = int(len(df_te))

if N_TE <= 0:
    raise RuntimeError(f"[Cell13.0] Invalid TEST length: N_TE={N_TE}")

CELL130_VERSION = "cell13_0_q4_cross_modal_coupling_input_contract_v1_1_schema_driven_pair_registry"

CFG["cell13_0_version"] = CELL130_VERSION
CFG["cell13_0_QA_only"] = True
CFG["cell13_0_TEST_real_values_used_for_QA_reference"] = True
CFG["cell13_0_synthetic_values_mutated"] = False
CFG["cell13_0_selection_done_here"] = False
CFG["cell13_0_generator_fit_done_here"] = False
CFG["cell13_0_materialization_done_here"] = False
CFG["cell13_0_pair_registry_selection_policy"] = "schema_driven_no_TEST_activity_ranking"
CFG["cell13_0_TEST_activity_used_for_pair_selection"] = False

CFG.setdefault("cell13_expected_iot_full_cols", 530)
CFG.setdefault("cell13_expected_test_rows", N_TE)
CFG.setdefault("cell13_lag_window_seconds", 10)
CFG.setdefault("cell13_max_lag_seconds", 60)
CFG.setdefault("cell13_eta_pre_window_seconds", 10)
CFG.setdefault("cell13_eta_post_window_seconds", 30)
CFG.setdefault("cell13_min_driver_events_for_pair", 5)
CFG.setdefault("cell13_min_protocol_nonzero_rate", 1e-7)
CFG.setdefault("cell13_pair_registry_top_protocol_metrics_per_tier", None)  # ignored in v1.1; pair registry is schema-driven
CFG.setdefault("cell13_include_observability_drivers", False)

EXPECTED_IOT_FULL_COLS_130 = int(CFG.get("cell13_expected_iot_full_cols", 530))
EXPECTED_TEST_ROWS_130 = int(CFG.get("cell13_expected_test_rows", N_TE))

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
contract_json = os.path.join(REPORT_DIR, "cell13_0_q4_coupling_input_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell13_0_q4_coupling_input_contract_v1_1_THESIS.json")
input_audit_csv = os.path.join(REPORT_DIR, "cell13_0_q4_coupling_input_audit.csv")
pair_registry_csv = os.path.join(REPORT_DIR, "cell13_0_driver_protocol_pair_registry.csv")
manifest_json = os.path.join(ARTDIR, "cell13_0_q4_coupling_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_130(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_130(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_130(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_130(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_130(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_130(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_130(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_130(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj

def _write_json_130(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_130(payload), f, indent=2, sort_keys=True)

def _sha256_file_130(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_csv_if_exists_130(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _read_parquet_if_exists_130(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        return None
    return None

def _normalise_test_frame_130(obj, name: str, required_cols=None) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        out = obj.copy()
    else:
        out = pd.DataFrame(obj)

    if len(out) != N_TE:
        raise RuntimeError(
            f"[Cell13.0] {name} row mismatch: got={len(out)} expected={N_TE}"
        )

    out.index = df_te.index
    out.columns = out.columns.astype(str)

    if required_cols is not None:
        required_cols = list(map(str, required_cols))
        missing = sorted(set(required_cols) - set(out.columns))
        if missing:
            raise RuntimeError(
                f"[Cell13.0] {name} missing required columns. Preview={missing[:30]}"
            )
        out = out[required_cols].copy()

    return out

def _safe_float_130(x, default=np.nan) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _to_numeric_array_130(frame: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

def _finite_rate_130(frame: pd.DataFrame, col: str) -> float:
    x = _to_numeric_array_130(frame, col)
    return float(np.isfinite(x).mean()) if x.size else np.nan

def _nonzero_rate_130(frame: pd.DataFrame, col: str) -> float:
    x = _to_numeric_array_130(frame, col)
    finite = np.isfinite(x)
    if not finite.any():
        return np.nan
    return float(np.mean(np.abs(x[finite]) > 1e-12))

def _mean_abs_130(frame: pd.DataFrame, col: str) -> float:
    x = _to_numeric_array_130(frame, col)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.mean(np.abs(x)))

def _is_iot_col_130(col: str) -> bool:
    s = str(col)
    return s.startswith("iot__") or s.startswith("events_in_sec__")

def _driver_entity_130(col: str) -> str:
    s = str(col)
    if s.startswith("events_in_sec__entity__"):
        return s.split("events_in_sec__entity__", 1)[1]
    if s.startswith("events_in_sec__feat__"):
        tail = s.split("events_in_sec__feat__", 1)[1]
        return tail.split("__", 1)[0]
    return s

def _protocol_tier_130(col: str) -> str:
    s = str(col).lower()

    if s.startswith("router__") or s.startswith("dns__") or s.startswith("wan__") or s.startswith("lan__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__") or s.startswith("wifi__") or s.startswith("wlan__"):
        return "ota"
    if s.startswith("zigbee__") or "zigbee" in s:
        return "zigbee"
    if s.startswith("zwave__") or "zwave" in s or "z_wave" in s:
        return "zwave"
    return "other"

def _looks_like_protocol_metric_130(col: str) -> bool:
    s = str(col).lower()

    if _protocol_tier_130(col) not in {"router", "ota", "zigbee", "zwave"}:
        return False

    # Avoid absolute time/canonical index features.
    if s in {"sec", "timestamp", "time", "datetime", "date_time", "sec_epoch_s__canon"}:
        return False

    if re.search(r"(^sec$|__sec$|^sec_|epoch|timestamp|datetime)", s):
        if not any(tok in s for tok in ["events_in_sec", "packets_per_sec", "bytes_per_sec"]):
            return False

    # Prefer count/volume/state/protocol metrics likely to respond to IoT events.
    response_tokens = [
        "pkt", "packet", "bytes", "frame", "count", "event", "cmd", "command",
        "tx", "rx", "retry", "rssi", "lqi", "linkquality", "signal",
        "broadcast", "multicast", "unicast", "ack", "dns", "query", "response",
        "tcp", "udp", "icmp", "arp", "flow", "conn", "len", "size", "rate",
        "total", "sum", "mean", "max",
    ]
    return any(tok in s for tok in response_tokens)

def _load_frame_by_candidates_130(global_names, path_candidates, label: str):
    for g in global_names:
        obj = globals().get(g, None)
        if isinstance(obj, pd.DataFrame):
            log(f"[Cell13.0] Loaded {label} from global {g}")
            return obj.copy(), f"global:{g}"

    for p in path_candidates:
        df = _read_parquet_if_exists_130(p)
        if isinstance(df, pd.DataFrame):
            log(f"[Cell13.0] Loaded {label} from parquet {p}")
            return df, f"path:{p}"

    return None, ""

# ----------------------------------------------------------
# 2b) Validate upstream full IoT namespace assembly contract
# ----------------------------------------------------------
contract12g_130 = CELL12G_IOT_FULL_NAMESPACE_CONTRACT
if not isinstance(contract12g_130, dict):
    raise RuntimeError("[Cell13.0] CELL12G_IOT_FULL_NAMESPACE_CONTRACT is not a dict.")

version12g_130 = str(contract12g_130.get("version", ""))
if "cell12g_full_iot_namespace_assembly_strict_v1_2" not in version12g_130:
    raise RuntimeError(
        "[Cell13.0] Unexpected Cell 12.g contract version. "
        f"Expected v1.2 quality-status-aware contract, got: {version12g_130}"
    )

if bool(contract12g_130.get("TEST_real_values_used", True)):
    raise RuntimeError("[Cell13.0] Cell 12.g contract indicates TEST target values were used before Q4 QA.")

if bool(contract12g_130.get("synthetic_values_mutated", False)):
    raise RuntimeError("[Cell13.0] Cell 12.g contract indicates synthetic values were mutated.")

quality_status_12g_130 = contract12g_130.get("quality_status_carried_forward", {})
q3_status_130 = quality_status_12g_130.get("q3_observability", {}) if isinstance(quality_status_12g_130, dict) else {}
q3_publication_blocker_n_130 = int(q3_status_130.get("publication_blocker_n", 0) or 0)

# ----------------------------------------------------------
# 3) Load full synthetic IoT TEST namespace
# ----------------------------------------------------------
iot_syn_path_candidates = [
    globals().get("CELL12G_FULL_IOT_SYNTHETIC_TEST_PATH", ""),
    os.path.join(OUT_SYN, "IOT_FULL_SYNTHETIC_TEST.parquet"),
]

iot_syn_raw, iot_syn_source = _load_frame_by_candidates_130(
    global_names=["IOT_FULL_SYNTHETIC_TEST", "CELL13_IOT_FULL_SYN_TEST"],
    path_candidates=iot_syn_path_candidates,
    label="full synthetic IoT TEST namespace",
)

if not isinstance(iot_syn_raw, pd.DataFrame):
    raise RuntimeError(
        "[Cell13.0] Could not locate IOT_FULL_SYNTHETIC_TEST from Cell 12.g."
    )

IOT_FULL_SYN_TEST_130 = _normalise_test_frame_130(
    iot_syn_raw,
    "IOT_FULL_SYNTHETIC_TEST",
)

if IOT_FULL_SYN_TEST_130.shape[1] != EXPECTED_IOT_FULL_COLS_130:
    raise RuntimeError(
        "[Cell13.0] Full IoT synthetic namespace column count mismatch: "
        f"got={IOT_FULL_SYN_TEST_130.shape[1]} expected={EXPECTED_IOT_FULL_COLS_130}"
    )

# ----------------------------------------------------------
# 4) Load synthetic protocol/network TEST matrix
# ----------------------------------------------------------
protocol_syn_path_candidates = [
    globals().get("CELL11_A2_PROTOCOL_TEST_PATH", ""),
    globals().get("A2_PROTOCOL_TEST_PATH", ""),
    globals().get("CELL11_PROTOCOL_SYN_TEST_PATH", ""),
    globals().get("PROTOCOL_SYN_TEST_PATH", ""),
    os.path.join(OUT_SYN, "A2_PROTOCOL_TEST.parquet"),
    os.path.join(OUT_SYN, "A2_PROTOCOL_VALUES_TEST.parquet"),
    os.path.join(OUT_SYN, "PROTOCOL_SYNTHETIC_TEST.parquet"),
    os.path.join(OUT_SYN, "PROTOCOL_FINAL_TEST.parquet"),
]

protocol_syn_raw, protocol_syn_source = _load_frame_by_candidates_130(
    global_names=[
        "A2_PROTOCOL_TEST",
        "A2_PROTOCOL_VALUES_TEST",
        "PROTOCOL_SYNTHETIC_TEST",
        "PROTOCOL_FINAL_TEST",
        "CELL11_A2_PROTOCOL_TEST",
        "CELL13_PROTOCOL_SYN_TEST",
    ],
    path_candidates=protocol_syn_path_candidates,
    label="synthetic protocol TEST matrix",
)

if not isinstance(protocol_syn_raw, pd.DataFrame):
    # Fallback: if protocol synthetic matrix is embedded in a larger generated frame,
    # use known protocol-looking columns from globals or files if available.
    raise RuntimeError(
        "[Cell13.0] Could not locate synthetic protocol/network TEST matrix. "
        "Expected a global such as A2_PROTOCOL_TEST / PROTOCOL_SYNTHETIC_TEST "
        "or a parquet such as synthetic/A2_PROTOCOL_TEST.parquet."
    )

protocol_syn_raw.columns = protocol_syn_raw.columns.astype(str)
protocol_candidate_cols = [
    c for c in protocol_syn_raw.columns
    if _looks_like_protocol_metric_130(c)
]

if not protocol_candidate_cols:
    raise RuntimeError(
        "[Cell13.0] Synthetic protocol matrix loaded, but no protocol metric columns were detected."
    )

PROTOCOL_SYN_TEST_130 = _normalise_test_frame_130(
    protocol_syn_raw,
    "synthetic protocol TEST matrix",
    required_cols=protocol_candidate_cols,
)

# ----------------------------------------------------------
# 5) Load real TEST references
# ----------------------------------------------------------
# Real IoT reference is df_te restricted to generated IoT full columns that exist.
iot_real_cols = [c for c in IOT_FULL_SYN_TEST_130.columns if c in df_te.columns]
IOT_REAL_TEST_REF_130 = _normalise_test_frame_130(
    df_te,
    "real TEST IoT reference",
    required_cols=iot_real_cols,
)

# Real protocol reference is df_te restricted to protocol synthetic columns that exist.
protocol_real_cols = [c for c in PROTOCOL_SYN_TEST_130.columns if c in df_te.columns]

if not protocol_real_cols:
    raise RuntimeError(
        "[Cell13.0] No overlap between synthetic protocol columns and df_te real TEST protocol columns."
    )

PROTOCOL_REAL_TEST_REF_130 = _normalise_test_frame_130(
    df_te,
    "real TEST protocol reference",
    required_cols=protocol_real_cols,
)

# Align synthetic protocol to real-overlap protocol columns.
PROTOCOL_SYN_TEST_130 = PROTOCOL_SYN_TEST_130[protocol_real_cols].copy()

# ----------------------------------------------------------
# 6) Discover driver columns
# ----------------------------------------------------------
driver_cols = []

if "IOT_DRIVER_COLS" in globals():
    try:
        driver_cols.extend(list(map(str, list(IOT_DRIVER_COLS))))
    except Exception:
        pass

driver_contract_path = os.path.join(REPORT_DIR, "cell12e0_driver_target_contract.csv")
driver_contract_df = _read_csv_if_exists_130(driver_contract_path)
if isinstance(driver_contract_df, pd.DataFrame) and "col" in driver_contract_df.columns:
    driver_cols.extend(driver_contract_df["col"].dropna().astype(str).tolist())

driver_registry_path = os.path.join(REPORT_DIR, "cell12e5_driver_publication_column_registry.csv")
driver_registry_df = _read_csv_if_exists_130(driver_registry_path)
if isinstance(driver_registry_df, pd.DataFrame):
    for cand in ["col", "driver_col", "target_col"]:
        if cand in driver_registry_df.columns:
            driver_cols.extend(driver_registry_df[cand].dropna().astype(str).tolist())
            break

driver_cols = sorted(list(dict.fromkeys(driver_cols)))

# Keep only drivers present in synthetic IoT and real TEST.
driver_cols = [
    c for c in driver_cols
    if c in IOT_FULL_SYN_TEST_130.columns and c in df_te.columns
]

if not driver_cols:
    # Fallback strict pattern.
    driver_cols = sorted([
        c for c in IOT_FULL_SYN_TEST_130.columns
        if str(c).startswith("events_in_sec__") and c in df_te.columns
    ])

if not driver_cols:
    raise RuntimeError("[Cell13.0] No driver columns found for Q4 coupling QA.")

# ----------------------------------------------------------
# 7) Protocol tier registry
# ----------------------------------------------------------
tier_cols = defaultdict(list)
for c in PROTOCOL_SYN_TEST_130.columns:
    tier = _protocol_tier_130(c)
    if tier in {"router", "ota", "zigbee", "zwave"}:
        tier_cols[tier].append(c)

tier_cols = {k: sorted(v) for k, v in tier_cols.items()}

if not any(len(v) for v in tier_cols.values()):
    raise RuntimeError("[Cell13.0] No protocol tier columns detected after alignment.")

tier_counts = {k: int(len(v)) for k, v in sorted(tier_cols.items())}

# ----------------------------------------------------------
# 8) Build protocol metric registry using schema-driven scope
# ----------------------------------------------------------
# v1.1 policy:
#   - The pair registry must not be selected/ranked using real TEST activity.
#   - Use protocol-tier/name detection and column alignment only.
#   - Real/synthetic activity statistics are recorded as QA audit metadata only.
protocol_metric_rows = []
for tier, cols in tier_cols.items():
    for c in cols:
        real_nonzero = _nonzero_rate_130(PROTOCOL_REAL_TEST_REF_130, c)
        syn_nonzero = _nonzero_rate_130(PROTOCOL_SYN_TEST_130, c)
        real_mean_abs = _mean_abs_130(PROTOCOL_REAL_TEST_REF_130, c)
        syn_mean_abs = _mean_abs_130(PROTOCOL_SYN_TEST_130, c)
        real_finite = _finite_rate_130(PROTOCOL_REAL_TEST_REF_130, c)
        syn_finite = _finite_rate_130(PROTOCOL_SYN_TEST_130, c)

        protocol_metric_rows.append({
            "protocol_col": c,
            "protocol_tier": tier,
            "real_nonzero_rate": real_nonzero,
            "syn_nonzero_rate": syn_nonzero,
            "real_mean_abs": real_mean_abs,
            "syn_mean_abs": syn_mean_abs,
            "real_finite_rate": real_finite,
            "syn_finite_rate": syn_finite,
            "activity_stats_used_for_pair_selection": False,
            "selection_policy": "schema_driven_protocol_tier_name_detection",
        })

protocol_metric_df = pd.DataFrame(protocol_metric_rows)

selected_protocol_cols = sorted(list(dict.fromkeys(map(str, PROTOCOL_SYN_TEST_130.columns))))

if not selected_protocol_cols:
    raise RuntimeError("[Cell13.0] No selected protocol metrics for coupling pair registry.")

# Sanity: selected protocol columns must be exactly schema/name-driven aligned protocol columns.
_activity_selected_n_130 = 0

# ----------------------------------------------------------
# 9) Driver registry
# ----------------------------------------------------------
driver_rows = []
for c in driver_cols:
    real_x = _to_numeric_array_130(df_te, c)
    syn_x = _to_numeric_array_130(IOT_FULL_SYN_TEST_130, c)

    real_finite = np.isfinite(real_x)
    syn_finite = np.isfinite(syn_x)

    real_events = int(np.sum(real_finite & (real_x > 0.5)))
    syn_events = int(np.sum(syn_finite & (syn_x > 0.5)))

    driver_rows.append({
        "driver_col": c,
        "entity": _driver_entity_130(c),
        "real_event_count": real_events,
        "syn_event_count": syn_events,
        "real_event_rate": float(real_events / max(N_TE, 1)),
        "syn_event_rate": float(syn_events / max(N_TE, 1)),
        "real_finite_rate": float(real_finite.mean()) if real_finite.size else np.nan,
        "syn_finite_rate": float(syn_finite.mean()) if syn_finite.size else np.nan,
        "eligible_for_q4_pairing": bool(
            real_events >= int(CFG.get("cell13_min_driver_events_for_pair", 5))
            or syn_events >= int(CFG.get("cell13_min_driver_events_for_pair", 5))
        ),
    })

driver_df = pd.DataFrame(driver_rows)

eligible_driver_cols = driver_df.loc[
    driver_df["eligible_for_q4_pairing"].astype(bool),
    "driver_col",
].astype(str).tolist()

if not eligible_driver_cols:
    raise RuntimeError(
        "[Cell13.0] Driver columns were found, but none have enough events for Q4 pairing."
    )

# ----------------------------------------------------------
# 10) Driver-protocol pair registry
# ----------------------------------------------------------
pair_rows = []
pair_id = 0

for dcol in eligible_driver_cols:
    entity = _driver_entity_130(dcol)

    for pcol in selected_protocol_cols:
        tier = _protocol_tier_130(pcol)

        # Lightweight semantic prior: all drivers may pair with router/ota;
        # Zigbee/Z-Wave pairs are more relevant to IoT sensor entities but still allowed.
        dlow = dcol.lower()
        plow = pcol.lower()

        semantic_hint = "generic_driver_protocol_pair"
        if "zigbee" in plow or tier == "zigbee":
            semantic_hint = "driver_to_zigbee_pair"
        elif "zwave" in plow or tier == "zwave":
            semantic_hint = "driver_to_zwave_pair"
        elif tier == "ota":
            semantic_hint = "driver_to_ota_wifi_pair"
        elif tier == "router":
            semantic_hint = "driver_to_router_pair"

        pair_rows.append({
            "pair_id": f"q4_pair_{pair_id:05d}",
            "driver_col": dcol,
            "driver_entity": entity,
            "protocol_col": pcol,
            "protocol_tier": tier,
            "semantic_hint": semantic_hint,
            "eta_pre_window_seconds": int(CFG.get("cell13_eta_pre_window_seconds", 10)),
            "eta_post_window_seconds": int(CFG.get("cell13_eta_post_window_seconds", 30)),
            "lag_window_seconds": int(CFG.get("cell13_lag_window_seconds", 10)),
            "max_lag_seconds": int(CFG.get("cell13_max_lag_seconds", 60)),
            "eligible_for_13_1_eta": True,
            "eligible_for_13_2_lag": True,
            "eligible_for_13_3_cross_corr": True,
            "TEST_real_values_used_for_QA_reference": True,
        })
        pair_id += 1

PAIR_REGISTRY_130 = pd.DataFrame(pair_rows)

if len(PAIR_REGISTRY_130) == 0:
    raise RuntimeError("[Cell13.0] Empty driver-protocol pair registry.")

# ----------------------------------------------------------
# 11) Input audits
# ----------------------------------------------------------
input_audit_rows = [
    {"metric": "test_rows", "value": int(N_TE)},
    {"metric": "iot_full_syn_cols", "value": int(IOT_FULL_SYN_TEST_130.shape[1])},
    {"metric": "iot_real_reference_cols_overlap", "value": int(len(iot_real_cols))},
    {"metric": "protocol_syn_cols_aligned", "value": int(PROTOCOL_SYN_TEST_130.shape[1])},
    {"metric": "protocol_real_reference_cols_aligned", "value": int(PROTOCOL_REAL_TEST_REF_130.shape[1])},
    {"metric": "driver_cols_total", "value": int(len(driver_cols))},
    {"metric": "eligible_driver_cols", "value": int(len(eligible_driver_cols))},
    {"metric": "selected_protocol_cols", "value": int(len(selected_protocol_cols))},
    {"metric": "protocol_pair_selection_policy", "value": "schema_driven_no_TEST_activity_ranking"},
    {"metric": "TEST_activity_used_for_pair_selection", "value": False},
    {"metric": "q3_observability_publication_blocker_n_carried_forward", "value": int(q3_publication_blocker_n_130)},
    {"metric": "driver_protocol_pairs", "value": int(len(PAIR_REGISTRY_130))},
    {"metric": "TEST_real_values_used_for_QA_reference", "value": True},
    {"metric": "synthetic_values_mutated", "value": False},
    {"metric": "selection_done_here", "value": False},
    {"metric": "generator_fit_done_here", "value": False},
]

for tier, n in tier_counts.items():
    input_audit_rows.append({"metric": f"protocol_tier_cols::{tier}", "value": int(n)})

for tier, sub in protocol_metric_df[protocol_metric_df["protocol_col"].isin(selected_protocol_cols)].groupby("protocol_tier"):
    input_audit_rows.append({"metric": f"selected_protocol_cols::{tier}", "value": int(len(sub))})

input_audit_df = pd.DataFrame(input_audit_rows)

# ----------------------------------------------------------
# 12) Save outputs
# ----------------------------------------------------------
input_audit_df.to_csv(input_audit_csv, index=False)
PAIR_REGISTRY_130.to_csv(pair_registry_csv, index=False)

contract = {
    "cell": "13.0",
    "version": CELL130_VERSION,
    "role": "Q4_cross_modal_coupling_input_contract",
    "quality_dimension": "Q4_cross_modal_consistency",
    "test_rows": int(N_TE),
    "inputs": {
        "iot_synthetic_source": iot_syn_source,
        "protocol_synthetic_source": protocol_syn_source,
        "iot_real_reference": "df_te restricted to IoT full namespace overlap",
        "protocol_real_reference": "df_te restricted to protocol synthetic overlap",
    },
    "shapes": {
        "IOT_FULL_SYN_TEST_130": list(IOT_FULL_SYN_TEST_130.shape),
        "IOT_REAL_TEST_REF_130": list(IOT_REAL_TEST_REF_130.shape),
        "PROTOCOL_SYN_TEST_130": list(PROTOCOL_SYN_TEST_130.shape),
        "PROTOCOL_REAL_TEST_REF_130": list(PROTOCOL_REAL_TEST_REF_130.shape),
    },
    "driver_summary": {
        "driver_cols_total": int(len(driver_cols)),
        "eligible_driver_cols": int(len(eligible_driver_cols)),
        "min_driver_events_for_pair": int(CFG.get("cell13_min_driver_events_for_pair", 5)),
    },
    "protocol_summary": {
        "protocol_tier_counts": tier_counts,
        "selected_protocol_cols_total": int(len(selected_protocol_cols)),
        "pair_registry_selection_policy": "schema_driven_no_TEST_activity_ranking",
        "TEST_activity_used_for_pair_selection": False,
        "selected_protocol_cols_by_tier": {
            tier: int(sum(_protocol_tier_130(c) == tier for c in selected_protocol_cols))
            for tier in sorted(set(_protocol_tier_130(c) for c in selected_protocol_cols))
        },
    },
    "pair_registry": {
        "driver_protocol_pairs": int(len(PAIR_REGISTRY_130)),
        "pair_registry_csv": pair_registry_csv,
        "selection_policy": "all eligible drivers crossed with schema-detected aligned protocol metric columns",
        "activity_stats_used_for_pair_selection": False,
    },
    "quality_status_carried_forward": {
        "cell12g_version": version12g_130,
        "q3_observability": q3_status_130,
        "q3_publication_blocker_n": int(q3_publication_blocker_n_130),
        "note": "Q3 status is carried for claim scoping; Q4 driver-protocol QA uses event-driver columns, not observability masks as drivers by default.",
    },
    "cell13_plan": {
        "13.1": {
            "purpose": "IoT-driver-to-network event-triggered average QA",
            "metrics": ["ETA_similarity"],
        },
        "13.2": {
            "purpose": "Lag distribution similarity and response-window metrics",
            "metrics": ["lag_peak_error", "response_window_rate_error"],
        },
        "13.3": {
            "purpose": "Cross-correlation preservation by protocol tier",
            "metrics": ["cross_corr_mae"],
        },
        "13.4": {
            "purpose": "Q4 coupling summary and publication blockers",
            "metrics": ["driver_protocol_pair_pass_rate"],
        },
    },
    "strict_contract": {
        "TEST_real_values_used_for_QA_reference": True,
        "TEST_activity_used_for_pair_selection": False,
        "pair_registry_schema_driven": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
    },
    "outputs": {
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "input_audit_csv": input_audit_csv,
        "pair_registry_csv": pair_registry_csv,
        "manifest_json": manifest_json,
    },
}

_write_json_130(contract_json, contract)
_write_json_130(contract_canonical_json, contract)

manifest = {
    "cell": "13.0",
    "version": CELL130_VERSION,
    "created_outputs": contract["outputs"],
    "quality_dimension": "Q4_cross_modal_consistency",
    "test_rows": int(N_TE),
    "driver_protocol_pairs": int(len(PAIR_REGISTRY_130)),
    "protocol_tier_counts": tier_counts,
    "pair_registry_selection_policy": "schema_driven_no_TEST_activity_ranking",
    "q3_publication_blocker_n_carried_forward": int(q3_publication_blocker_n_130),
    "strict_contract": contract["strict_contract"],
}

_write_json_130(manifest_json, manifest)

hashes = {
    "contract_json_sha256": _sha256_file_130(contract_json),
    "contract_canonical_json_sha256": _sha256_file_130(contract_canonical_json),
    "input_audit_csv_sha256": _sha256_file_130(input_audit_csv),
    "pair_registry_csv_sha256": _sha256_file_130(pair_registry_csv),
    "manifest_json_sha256": _sha256_file_130(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_130(contract_json, contract)
_write_json_130(contract_canonical_json, contract)
_write_json_130(manifest_json, manifest)

# ----------------------------------------------------------
# 13) Export globals for 13.1-13.4
# ----------------------------------------------------------
globals()["CELL130_VERSION"] = CELL130_VERSION
globals()["IOT_FULL_SYN_TEST_130"] = IOT_FULL_SYN_TEST_130
globals()["IOT_REAL_TEST_REF_130"] = IOT_REAL_TEST_REF_130
globals()["PROTOCOL_SYN_TEST_130"] = PROTOCOL_SYN_TEST_130
globals()["PROTOCOL_REAL_TEST_REF_130"] = PROTOCOL_REAL_TEST_REF_130
globals()["CELL13_DRIVER_COLS"] = driver_cols
globals()["CELL13_ELIGIBLE_DRIVER_COLS"] = eligible_driver_cols
globals()["CELL13_PROTOCOL_TIER_COLS"] = tier_cols
globals()["CELL13_SELECTED_PROTOCOL_COLS"] = selected_protocol_cols
globals()["CELL13_PROTOCOL_METRIC_REGISTRY_DF"] = protocol_metric_df
globals()["CELL13_DRIVER_REGISTRY_DF"] = driver_df
globals()["CELL13_DRIVER_PROTOCOL_PAIR_REGISTRY_DF"] = PAIR_REGISTRY_130
globals()["CELL130_Q4_COUPLING_INPUT_AUDIT_DF"] = input_audit_df
globals()["CELL130_Q4_COUPLING_INPUT_CONTRACT"] = contract
globals()["CELL130_Q4_COUPLING_INPUT_CONTRACT_JSON"] = contract_json
globals()["CELL130_Q4_COUPLING_INPUT_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL130_Q4_COUPLING_INPUT_AUDIT_CSV"] = input_audit_csv
globals()["CELL130_DRIVER_PROTOCOL_PAIR_REGISTRY_CSV"] = pair_registry_csv
globals()["CELL130_Q4_COUPLING_MANIFEST_JSON"] = manifest_json

log(
    "[Cell13.0] Q4 coupling inputs loaded | "
    f"IoT_syn_shape={IOT_FULL_SYN_TEST_130.shape} | "
    f"IoT_real_ref_cols={len(iot_real_cols)} | "
    f"protocol_syn_shape={PROTOCOL_SYN_TEST_130.shape} | "
    f"protocol_real_shape={PROTOCOL_REAL_TEST_REF_130.shape}"
)
log(
    "[Cell13.0] Driver/protocol registry | "
    f"drivers_total={len(driver_cols)} | "
    f"eligible_drivers={len(eligible_driver_cols)} | "
    f"selected_protocol_cols={len(selected_protocol_cols)} | "
    f"pairs={len(PAIR_REGISTRY_130)}"
)
log(f"[Cell13.0] Protocol tier counts | {tier_counts}")
log(f"[Cell13.0] Saved input audit: {input_audit_csv}")
log(f"[Cell13.0] Saved pair registry: {pair_registry_csv}")
log(f"[Cell13.0] Saved Q4 contract: {contract_json}")
log(f"[Cell13.0] Saved canonical Q4 contract: {contract_canonical_json}")
log(
    "[Cell13.0] Pair registry policy | "
    "schema_driven_no_TEST_activity_ranking=True | "
    f"q3_publication_blocker_n_carried_forward={q3_publication_blocker_n_130}"
)
log(
    "[Cell13.0] Contract flags | "
    "TEST_real_values_used_for_QA_reference=True | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | TEST_activity_used_for_pair_selection=False"
)
log("--- END: Cell 13.0 - Q4 cross-modal coupling input contract (v1.1 schema-driven strict) ---")

gc.collect()