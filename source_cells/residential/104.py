# ==========================================================
# CELL 13.5 - Manifest-informed Zigbee Q4 coupling registry
# v1.1 STUDY-THESIS strict manifest-informed Q4 registry, broad-Q4-aware
#
# Role:
#   - Load replicated coupling manifest artifacts from the earlier
#     spike-sensitive TRAIN+VAL validator.
#   - Build a narrow Q4 pair registry from manifest Tier A / Tier B pairs.
#   - Map:
#       anchor_name -> events_in_sec__entity__{anchor_name}
#       signal      -> protocol signal, usually zigbee__pkt_total / zigbee__bytes_total
#   - Validate every pair against current Cell 13.0 frames:
#       IOT_FULL_SYN_TEST_130
#       IOT_REAL_TEST_REF_130
#       PROTOCOL_SYN_TEST_130
#       PROTOCOL_REAL_TEST_REF_130
#
# Scientific stance:
#   - Broad all-pair Q4 in 13.1-13.4 remains valid and currently blocked.
#   - This cell evaluates the replicated manifest-defined coupling subset.
#   - No synthetic values are mutated.
#
# Outputs:
#   reports/cell13_5_manifest_q4_pair_registry.csv
#   reports/cell13_5_manifest_q4_registry_audit.csv
#   reports/cell13_5_manifest_q4_contract.json
#   artifacts/cell13_5_manifest_q4_manifest.json
# ==========================================================

log("--- START: Cell 13.5 - Manifest-informed Zigbee Q4 registry (v1.1 broad-Q4-aware strict) ---")

import os
import gc
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_135 = [
    "CFG", "log",
    "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "IOT_FULL_SYN_TEST_130",
    "IOT_REAL_TEST_REF_130",
    "PROTOCOL_SYN_TEST_130",
    "PROTOCOL_REAL_TEST_REF_130",
    "CELL130_Q4_COUPLING_INPUT_CONTRACT",
    "CELL13_4_Q4_PUBLICATION_SUMMARY",
]
_missing_135 = [k for k in _required_135 if k not in globals()]
if _missing_135:
    raise RuntimeError(f"[Cell13.5] Missing required globals from Cell 13.0: {_missing_135}")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")
MANIFEST_DIR = os.path.normpath(OUTDIR + "_manifest")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)
N_TE = int(len(df_te))

CELL135_VERSION = "cell13_5_manifest_informed_zigbee_q4_registry_v1_1_broad_q4_aware"

CFG["cell13_5_version"] = CELL135_VERSION
CFG["cell13_5_QA_registry_only"] = True
CFG["cell13_5_TEST_real_values_used_for_QA_reference"] = False
CFG["cell13_5_synthetic_values_mutated"] = False
CFG["cell13_5_selection_done_here"] = False
CFG["cell13_5_generator_fit_done_here"] = False
CFG["cell13_5_materialization_done_here"] = False
CFG["cell13_5_broad_q4_status_carried_forward"] = True
CFG["cell13_5_registry_scope_selection_policy"] = "preexisting_trainval_manifest_only"
CFG["cell13_5_does_not_repair_or_override_broad_q4"] = True

CFG.setdefault("cell13_5_include_tierA", True)
CFG.setdefault("cell13_5_include_tierB", True)
CFG.setdefault("cell13_5_allowed_manifest_tiers", ["A", "B"])
CFG.setdefault("cell13_5_allowed_protocol_tiers", ["zigbee"])
CFG.setdefault("cell13_5_min_manifest_weight", 0.0)
CFG.setdefault("cell13_5_fail_if_no_manifest_pairs", True)

# ----------------------------------------------------------
# 1) Output paths
# ----------------------------------------------------------
manifest_pair_registry_csv = os.path.join(REPORT_DIR, "cell13_5_manifest_q4_pair_registry.csv")
manifest_registry_audit_csv = os.path.join(REPORT_DIR, "cell13_5_manifest_q4_registry_audit.csv")
manifest_contract_json = os.path.join(REPORT_DIR, "cell13_5_manifest_q4_contract.json")
manifest_contract_canonical_json = os.path.join(CONTRACT_DIR, "cell13_5_manifest_q4_contract_v1_1_THESIS.json")
manifest_manifest_json = os.path.join(ARTDIR, "cell13_5_manifest_q4_manifest.json")

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_135(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_135(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_135(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_135(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_135(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_135(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_135(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_135(obj.to_dict())
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

def _write_json_135(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_135(payload), f, indent=2, sort_keys=True)

def _sha256_file_135(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_parquet_if_exists_135(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_parquet(path)
    except Exception:
        return None
    return None

def _read_csv_if_exists_135(path: str):
    try:
        if path and os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _safe_float_135(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _protocol_tier_135(col: str) -> str:
    s = str(col).lower()
    if s.startswith("zigbee__") or s.startswith("zb__") or "zigbee" in s:
        return "zigbee"
    if s.startswith("router__") or s.startswith("dns__"):
        return "router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__") or s.startswith("wifi__"):
        return "ota"
    if s.startswith("zwave__") or "zwave" in s or "z_wave" in s:
        return "zwave"
    return "other"

def _driver_col_from_anchor_135(anchor_name: str) -> str:
    return f"events_in_sec__entity__{str(anchor_name)}"

def _normalise_manifest_tier_135(x) -> str:
    s = str(x).strip().upper()
    if s in {"TIERA", "TIER_A"}:
        return "A"
    if s in {"TIERB", "TIER_B"}:
        return "B"
    return s

# ----------------------------------------------------------
# 2b) Validate upstream broad Q4 summary / input contract
# ----------------------------------------------------------
contract130_135 = CELL130_Q4_COUPLING_INPUT_CONTRACT
if not isinstance(contract130_135, dict):
    raise RuntimeError("[Cell13.5] CELL130_Q4_COUPLING_INPUT_CONTRACT is not a dict.")

version130_135 = str(contract130_135.get("version", ""))
if "cell13_0_q4_cross_modal_coupling_input_contract_v1_1" not in version130_135:
    raise RuntimeError(
        "[Cell13.5] Unexpected Cell 13.0 contract version. "
        f"Expected v1.1 schema-driven pair registry, got: {version130_135}"
    )

strict130_135 = contract130_135.get("strict_contract", {})
if bool(strict130_135.get("TEST_activity_used_for_pair_selection", True)):
    raise RuntimeError("[Cell13.5] Cell 13.0 contract indicates TEST activity was used for pair selection.")
if not bool(strict130_135.get("pair_registry_schema_driven", False)):
    raise RuntimeError("[Cell13.5] Cell 13.0 contract does not declare schema-driven pair registry.")

q4_summary_135 = CELL13_4_Q4_PUBLICATION_SUMMARY
if not isinstance(q4_summary_135, dict):
    raise RuntimeError("[Cell13.5] CELL13_4_Q4_PUBLICATION_SUMMARY is not a dict.")

version134_135 = str(q4_summary_135.get("version", ""))
if "cell13_4_q4_cross_modal_coupling_publication_summary_v1_1" not in version134_135:
    raise RuntimeError(
        "[Cell13.5] Unexpected Cell 13.4 summary version. "
        f"Expected v1.1 contract-hardened Q4 summary, got: {version134_135}"
    )

broad_q4_status_135 = str(q4_summary_135.get("overall_q4_status", ""))
broad_q4_pair_blocker_n_135 = int(q4_summary_135.get("pair_summary", {}).get("pair_blocker_n", 0) or 0)
broad_q4_pair_pass_rate_135 = _safe_float_135(
    q4_summary_135.get("pair_summary", {}).get("driver_protocol_pair_pass_rate", np.nan),
    np.nan,
)
broad_q4_publication_blocker_record_n_135 = int(
    q4_summary_135.get("publication_blocker_record_n", q4_summary_135.get("publication_blocker_n", 0)) or 0
)

if broad_q4_status_135 != "blocker":
    log(
        "[Cell13.5] WARNING: broad Q4 summary is not marked blocker. "
        f"status={broad_q4_status_135}"
    )

def _load_manifest_table_135():
    frames = []
    sources = []

    # Prefer in-memory globals if present.
    for gname, tier_name in [
        ("COUPLING_MANIFEST_TIERA", "A"),
        ("COUPLING_MANIFEST_TIERB", "B"),
    ]:
        obj = globals().get(gname, None)
        if isinstance(obj, pd.DataFrame) and len(obj):
            cur = obj.copy()
            if "manifest_tier" not in cur.columns:
                cur["manifest_tier"] = tier_name
            frames.append(cur)
            sources.append(f"global:{gname}")

    # Path candidates from old manifest builder.
    candidates = [
        (os.path.join(MANIFEST_DIR, "coupling_manifest_tierA.parquet"), "A"),
        (os.path.join(MANIFEST_DIR, "coupling_manifest_tierB.parquet"), "B"),
        (os.path.join(MANIFEST_DIR, "coupling_manifest_all.parquet"), ""),
        (os.path.join(OUTDIR, "coupling_manifest_tierA.parquet"), "A"),
        (os.path.join(OUTDIR, "coupling_manifest_tierB.parquet"), "B"),
        (os.path.join(OUTDIR, "coupling_manifest_all.parquet"), ""),
        (os.path.join(REPORT_DIR, "coupling_manifest_tierA.parquet"), "A"),
        (os.path.join(REPORT_DIR, "coupling_manifest_tierB.parquet"), "B"),
        (os.path.join(REPORT_DIR, "coupling_manifest_all.parquet"), ""),
    ]

    for path, tier_name in candidates:
        df = _read_parquet_if_exists_135(path)
        if isinstance(df, pd.DataFrame) and len(df):
            cur = df.copy()
            if "manifest_tier" not in cur.columns and tier_name:
                cur["manifest_tier"] = tier_name
            frames.append(cur)
            sources.append(path)

    if not frames:
        # CSV fallback.
        candidates_csv = [
            (os.path.join(MANIFEST_DIR, "coupling_manifest_tierA.csv"), "A"),
            (os.path.join(MANIFEST_DIR, "coupling_manifest_tierB.csv"), "B"),
            (os.path.join(MANIFEST_DIR, "coupling_manifest_all.csv"), ""),
            (os.path.join(OUTDIR, "coupling_manifest_tierA.csv"), "A"),
            (os.path.join(OUTDIR, "coupling_manifest_tierB.csv"), "B"),
            (os.path.join(OUTDIR, "coupling_manifest_all.csv"), ""),
        ]
        for path, tier_name in candidates_csv:
            df = _read_csv_if_exists_135(path)
            if isinstance(df, pd.DataFrame) and len(df):
                cur = df.copy()
                if "manifest_tier" not in cur.columns and tier_name:
                    cur["manifest_tier"] = tier_name
                frames.append(cur)
                sources.append(path)

    if not frames:
        raise RuntimeError(
            "[Cell13.5] Could not locate coupling manifest artifacts. Expected files such as "
            f"{os.path.join(MANIFEST_DIR, 'coupling_manifest_tierA.parquet')}"
        )

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates()

    return all_df, sources

def _required_manifest_columns_135(df: pd.DataFrame):
    required = [
        "anchor_name",
        "signal",
        "manifest_tier",
        "lag_lo",
        "lag_hi",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"[Cell13.5] Coupling manifest missing required columns: {missing}")

# ----------------------------------------------------------
# 3) Load manifest and validate current frames
# ----------------------------------------------------------
manifest_raw, manifest_sources = _load_manifest_table_135()
_required_manifest_columns_135(manifest_raw)

for frame_name, frame in [
    ("IOT_FULL_SYN_TEST_130", IOT_FULL_SYN_TEST_130),
    ("IOT_REAL_TEST_REF_130", IOT_REAL_TEST_REF_130),
    ("PROTOCOL_SYN_TEST_130", PROTOCOL_SYN_TEST_130),
    ("PROTOCOL_REAL_TEST_REF_130", PROTOCOL_REAL_TEST_REF_130),
]:
    if len(frame) != N_TE:
        raise RuntimeError(f"[Cell13.5] {frame_name} row mismatch: got={len(frame)} expected={N_TE}")
    frame.columns = frame.columns.astype(str)

allowed_manifest_tiers = set(str(x).upper() for x in CFG.get("cell13_5_allowed_manifest_tiers", ["A", "B"]))
allowed_protocol_tiers = set(str(x).lower() for x in CFG.get("cell13_5_allowed_protocol_tiers", ["zigbee"]))

# ----------------------------------------------------------
# 4) Build manifest-informed pair registry
# ----------------------------------------------------------
audit_rows = []
pair_rows = []
seen = set()
pair_i = 0

for _, r in manifest_raw.iterrows():
    manifest_tier = _normalise_manifest_tier_135(r.get("manifest_tier", ""))
    anchor_name = str(r.get("anchor_name", "")).strip()
    signal = str(r.get("signal", "")).strip()

    if not anchor_name or not signal:
        audit_rows.append({
            "anchor_name": anchor_name,
            "signal": signal,
            "manifest_tier": manifest_tier,
            "registry_selected": False,
            "rejection_reason": "missing_anchor_or_signal",
        })
        continue

    if manifest_tier not in allowed_manifest_tiers:
        audit_rows.append({
            "anchor_name": anchor_name,
            "signal": signal,
            "manifest_tier": manifest_tier,
            "registry_selected": False,
            "rejection_reason": "manifest_tier_not_allowed",
        })
        continue

    protocol_tier = str(r.get("tier_consensus", "")).strip().lower()
    if not protocol_tier or protocol_tier == "nan":
        protocol_tier = _protocol_tier_135(signal)

    if protocol_tier not in allowed_protocol_tiers:
        audit_rows.append({
            "anchor_name": anchor_name,
            "signal": signal,
            "manifest_tier": manifest_tier,
            "protocol_tier": protocol_tier,
            "registry_selected": False,
            "rejection_reason": "protocol_tier_not_allowed",
        })
        continue

    driver_col = _driver_col_from_anchor_135(anchor_name)
    protocol_col = signal

    lag_lo = int(round(_safe_float_135(r.get("lag_lo"), 0)))
    lag_hi = int(round(_safe_float_135(r.get("lag_hi"), 0)))
    if lag_lo > lag_hi:
        lag_lo, lag_hi = lag_hi, lag_lo

    recommended_weight = _safe_float_135(r.get("recommended_weight"), 1.0)
    if not np.isfinite(recommended_weight):
        recommended_weight = 1.0

    if recommended_weight < float(CFG.get("cell13_5_min_manifest_weight", 0.0)):
        audit_rows.append({
            "anchor_name": anchor_name,
            "signal": signal,
            "manifest_tier": manifest_tier,
            "protocol_tier": protocol_tier,
            "registry_selected": False,
            "rejection_reason": "recommended_weight_below_threshold",
        })
        continue

    present_driver_syn = driver_col in IOT_FULL_SYN_TEST_130.columns
    present_driver_real = driver_col in IOT_REAL_TEST_REF_130.columns
    present_protocol_syn = protocol_col in PROTOCOL_SYN_TEST_130.columns
    present_protocol_real = protocol_col in PROTOCOL_REAL_TEST_REF_130.columns

    missing = []
    if not present_driver_syn:
        missing.append("driver_missing_synthetic_iot")
    if not present_driver_real:
        missing.append("driver_missing_real_iot")
    if not present_protocol_syn:
        missing.append("protocol_missing_synthetic")
    if not present_protocol_real:
        missing.append("protocol_missing_real")

    if missing:
        audit_rows.append({
            "anchor_name": anchor_name,
            "driver_col": driver_col,
            "signal": signal,
            "protocol_col": protocol_col,
            "manifest_tier": manifest_tier,
            "protocol_tier": protocol_tier,
            "registry_selected": False,
            "rejection_reason": "|".join(missing),
        })
        continue

    key = (driver_col, protocol_col, manifest_tier, lag_lo, lag_hi)
    if key in seen:
        audit_rows.append({
            "anchor_name": anchor_name,
            "driver_col": driver_col,
            "signal": signal,
            "protocol_col": protocol_col,
            "manifest_tier": manifest_tier,
            "protocol_tier": protocol_tier,
            "registry_selected": False,
            "rejection_reason": "duplicate_manifest_pair",
        })
        continue
    seen.add(key)

    pair_id = f"manifest_q4_pair_{pair_i:05d}"
    pair_i += 1

    pair_rows.append({
        "pair_id": pair_id,
        "manifest_tier": manifest_tier,
        "anchor_name": anchor_name,
        "driver_col": driver_col,
        "protocol_col": protocol_col,
        "protocol_tier": protocol_tier,
        "lag_lo": int(lag_lo),
        "lag_hi": int(lag_hi),
        "recommended_weight": float(recommended_weight),
        "replication_score": _safe_float_135(r.get("replication_score"), np.nan),
        "consensus_lag": _safe_float_135(r.get("consensus_lag"), np.nan),
        "peak_lag_sec_train": _safe_float_135(r.get("peak_lag_sec__train"), np.nan),
        "peak_lag_sec_val": _safe_float_135(r.get("peak_lag_sec__val"), np.nan),
        "z_immediate_train": _safe_float_135(r.get("z_immediate_vs_null__train"), np.nan),
        "z_immediate_val": _safe_float_135(r.get("z_immediate_vs_null__val"), np.nan),
        "z_sharpness_train": _safe_float_135(r.get("z_sharpness_vs_null__train"), np.nan),
        "z_sharpness_val": _safe_float_135(r.get("z_sharpness_vs_null__val"), np.nan),
        "coverage_train": _safe_float_135(r.get("coverage_modality_near__train"), np.nan),
        "coverage_val": _safe_float_135(r.get("coverage_modality_near__val"), np.nan),
        "alignment_label_train": str(r.get("alignment_label__train", "")),
        "alignment_label_val": str(r.get("alignment_label__val", "")),
        "source_manifest_rows_merged": 1,
        "eligible_for_13_6_manifest_q4": True,
    })

    audit_rows.append({
        "anchor_name": anchor_name,
        "driver_col": driver_col,
        "signal": signal,
        "protocol_col": protocol_col,
        "manifest_tier": manifest_tier,
        "protocol_tier": protocol_tier,
        "lag_lo": int(lag_lo),
        "lag_hi": int(lag_hi),
        "registry_selected": True,
        "rejection_reason": "",
    })

manifest_pair_registry_df = pd.DataFrame(pair_rows)
manifest_registry_audit_df = pd.DataFrame(audit_rows)

manifest_pair_registry_df.to_csv(manifest_pair_registry_csv, index=False)
manifest_registry_audit_df.to_csv(manifest_registry_audit_csv, index=False)

if len(manifest_pair_registry_df) == 0 and bool(CFG.get("cell13_5_fail_if_no_manifest_pairs", True)):
    raise RuntimeError(
        "[Cell13.5] No manifest-informed Q4 pairs survived validation against current 13.0 frames. "
        f"Audit saved to: {manifest_registry_audit_csv}"
    )

tier_counts = (
    manifest_pair_registry_df["protocol_tier"].astype(str).value_counts().sort_index().to_dict()
    if len(manifest_pair_registry_df)
    else {}
)
manifest_tier_counts = (
    manifest_pair_registry_df["manifest_tier"].astype(str).value_counts().sort_index().to_dict()
    if len(manifest_pair_registry_df)
    else {}
)

contract = {
    "cell": "13.5",
    "version": CELL135_VERSION,
    "role": "manifest_informed_zigbee_q4_pair_registry",
    "quality_dimension": "Q4_cross_modal_consistency_manifest_subset",
    "cell13_0_contract_version_seen": version130_135,
    "cell13_4_summary_version_seen": version134_135,
    "broad_q4_status_carried_forward": {
        "overall_q4_status": broad_q4_status_135,
        "pair_blocker_n": int(broad_q4_pair_blocker_n_135),
        "pair_pass_rate": broad_q4_pair_pass_rate_135,
        "publication_blocker_record_n": int(broad_q4_publication_blocker_record_n_135),
        "does_not_repair_or_override_broad_q4": True,
        "interpretation": "This manifest registry defines a narrow follow-up evaluation subset; broad all-pair Q4 remains separately valid and blocked."
    },
    "manifest_sources": manifest_sources,
    "raw_manifest_rows": int(len(manifest_raw)),
    "registry_rows": int(len(manifest_pair_registry_df)),
    "audit_rows": int(len(manifest_registry_audit_df)),
    "manifest_tier_counts": manifest_tier_counts,
    "protocol_tier_counts": tier_counts,
    "allowed_manifest_tiers": sorted(list(allowed_manifest_tiers)),
    "allowed_protocol_tiers": sorted(list(allowed_protocol_tiers)),
    "strict_contract": {
        "TEST_real_values_used_for_QA_reference": False,
        "TEST_activity_used_for_pair_selection": False,
        "pair_registry_scope": "preexisting_trainval_manifest_only",
        "broad_q4_status_carried_forward": True,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
    },
    "outputs": {
        "manifest_pair_registry_csv": manifest_pair_registry_csv,
        "manifest_registry_audit_csv": manifest_registry_audit_csv,
        "manifest_contract_json": manifest_contract_json,
        "manifest_contract_canonical_json": manifest_contract_canonical_json,
        "manifest_manifest_json": manifest_manifest_json,
    },
}

_write_json_135(manifest_contract_json, contract)
_write_json_135(manifest_contract_canonical_json, contract)

manifest = {
    "cell": "13.5",
    "version": CELL135_VERSION,
    "created_outputs": contract["outputs"],
    "registry_rows": int(len(manifest_pair_registry_df)),
    "broad_q4_status_carried_forward": contract["broad_q4_status_carried_forward"],
    "manifest_tier_counts": manifest_tier_counts,
    "protocol_tier_counts": tier_counts,
    "strict_contract": contract["strict_contract"],
}

_write_json_135(manifest_manifest_json, manifest)

hashes = {
    "manifest_pair_registry_csv_sha256": _sha256_file_135(manifest_pair_registry_csv),
    "manifest_registry_audit_csv_sha256": _sha256_file_135(manifest_registry_audit_csv),
    "manifest_contract_json_sha256": _sha256_file_135(manifest_contract_json),
    "manifest_contract_canonical_json_sha256": _sha256_file_135(manifest_contract_canonical_json),
    "manifest_manifest_json_sha256": _sha256_file_135(manifest_manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_135(manifest_contract_json, contract)
_write_json_135(manifest_contract_canonical_json, contract)
_write_json_135(manifest_manifest_json, manifest)

# ----------------------------------------------------------
# 5) Export globals
# ----------------------------------------------------------
globals()["CELL135_VERSION"] = CELL135_VERSION
globals()["CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_DF"] = manifest_pair_registry_df
globals()["CELL13_5_MANIFEST_Q4_REGISTRY_AUDIT_DF"] = manifest_registry_audit_df
globals()["CELL13_5_MANIFEST_Q4_CONTRACT"] = contract
globals()["CELL13_5_MANIFEST_Q4_PAIR_REGISTRY_CSV"] = manifest_pair_registry_csv
globals()["CELL13_5_MANIFEST_Q4_REGISTRY_AUDIT_CSV"] = manifest_registry_audit_csv
globals()["CELL13_5_MANIFEST_Q4_CONTRACT_JSON"] = manifest_contract_json
globals()["CELL13_5_MANIFEST_Q4_CONTRACT_CANONICAL_JSON"] = manifest_contract_canonical_json
globals()["CELL13_5_MANIFEST_Q4_MANIFEST_JSON"] = manifest_manifest_json

log(
    "[Cell13.5] Manifest-informed Q4 registry built | "
    f"raw_manifest_rows={len(manifest_raw)} | "
    f"registry_pairs={len(manifest_pair_registry_df)} | "
    f"manifest_tiers={manifest_tier_counts} | "
    f"protocol_tiers={tier_counts}"
)
log(f"[Cell13.5] Saved manifest pair registry: {manifest_pair_registry_csv}")
log(f"[Cell13.5] Saved registry audit: {manifest_registry_audit_csv}")
log(f"[Cell13.5] Saved contract: {manifest_contract_json}")
log(f"[Cell13.5] Saved canonical contract: {manifest_contract_canonical_json}")
log(
    "[Cell13.5] Broad Q4 status carried forward | "
    f"overall_status={broad_q4_status_135} | "
    f"pair_blocker_n={broad_q4_pair_blocker_n_135} | "
    f"pair_pass_rate={broad_q4_pair_pass_rate_135}"
)
log(
    "[Cell13.5] Contract flags | "
    "TEST_real_values_used_for_QA_reference=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | broad_q4_status_carried_forward=True | registry_scope=preexisting_trainval_manifest_only"
)
log("--- END: Cell 13.5 - Manifest-informed Zigbee Q4 registry (v1.1 broad-Q4-aware strict) ---")

gc.collect()