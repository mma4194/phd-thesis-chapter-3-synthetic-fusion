# ==========================================================
# CELL 17.0 - External baseline comparable target scopes
# v1.1 STUDY-THESIS strict no-Q4-promotion baseline fairness contract
#
# Role:
#   - Define fair, comparable target scopes for external baselines.
#   - Prevent unfair comparison of CTGAN / TabDDPM / TimeGAN against the
#     full role-aware, mask-aware CPS pipeline.
#
# Final governance inherited from Cell 16.5:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative or used as a baseline target.
#   - Public Q6 candidate clears direct no-copy/DCR blockers only at warning
#     level, but remains strict-release blocked by synthetic-real
#     distinguishability.
#
# Baseline philosophy:
#   CTGAN / TabDDPM:
#     tabular, row-wise baselines; compare marginal/correlation/tabular metrics.
#
#   TimeGAN:
#     sequence baseline; compare temporal metrics on compact sequence scopes.
#
# Explicit non-goals:
#   - Do not expect external baselines to reproduce full namespace assembly.
#   - Do not expect them to preserve strict observability masks.
#   - Do not expect them to perform Q4 IoT↔protocol coupling repair.
#   - Do not compare them directly to the full no-Q4 556-column scientific artifact.
#   - Do not use Q4-coupled legacy artifacts as baseline targets.
#
# Outputs:
#   reports/cell17_0_baseline_scope_registry.csv
#   reports/cell17_0_baseline_column_registry.csv
#   reports/cell17_0_baseline_fairness_contract.json
#   artifacts/contracts/cell17_0_baseline_fairness_contract_v1_1_THESIS.json
#   artifacts/cell17_0_baseline_scope_manifest.json
# ==========================================================

log("--- START: Cell 17.0 - External baseline comparable target scopes (v1.1 no-Q4-promotion strict) ---")

import os
import gc
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_170 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL16_5_FINAL_DECOMPOSITION_CONTRACT",
]
_missing_170 = [k for k in _required_170 if k not in globals()]
if _missing_170:
    raise RuntimeError(f"[Cell17.0] Missing required globals: {_missing_170}")

ORIGINAL_OUTDIR_170 = str(OUTDIR)
ORIGINAL_OUT_SYN_170 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_170 = str(REPORT_DIR)

def _resolve_project_root_170(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        for marker in ["q6_public_reaudit_v1", "q6_public_reaudit"]:
            if marker in parts:
                candidates.append(os.sep.join(parts[:parts.index(marker)]))
        if os.path.basename(p) in {"reports", "synthetic", "artifacts"}:
            candidates.append(os.path.dirname(p))
        else:
            candidates.append(p)

    _env_project_root = os.environ.get("CPS_CANONICAL_PROJECT_ROOT", "").strip()
    if _env_project_root:
        candidates.append(_env_project_root)

    seen = set()
    for c in candidates:
        c = os.path.abspath(str(c))
        if c in seen:
            continue
        seen.add(c)
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "synthetic")):
            return c

    raise RuntimeError("[Cell17.0] Could not resolve canonical project root.")

PROJECT_ROOT_170 = _resolve_project_root_170(ORIGINAL_OUTDIR_170, ORIGINAL_REPORT_DIR_170, ORIGINAL_OUT_SYN_170)
OUT_SYN_BASE_170 = os.path.join(PROJECT_ROOT_170, "synthetic")
REPORT_DIR_BASE_170 = os.path.join(PROJECT_ROOT_170, "reports")
ARTDIR_BASE_170 = os.path.join(PROJECT_ROOT_170, "artifacts")
CONTRACT_DIR_BASE_170 = os.path.join(ARTDIR_BASE_170, "contracts")
PUBLIC_Q6_REPORT_DIR_170 = os.path.join(PROJECT_ROOT_170, "q6_public_reaudit_v1", "reports")

os.makedirs(OUT_SYN_BASE_170, exist_ok=True)
os.makedirs(REPORT_DIR_BASE_170, exist_ok=True)
os.makedirs(ARTDIR_BASE_170, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_170, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

CELL170_VERSION = "cell17_0_external_baseline_scope_contract_v1_1_no_q4_promotion"

CFG["cell17_0_version"] = CELL170_VERSION
CFG["cell17_0_Q4_final_status"] = "blocked_no_promotion"
CFG["cell17_0_Q4_coupled_artifacts_used"] = False
CFG["cell17_0_TEST_real_values_used"] = False
CFG["cell17_0_TEST_real_values_used_for_materialization"] = False
CFG["cell17_0_synthetic_values_mutated"] = False
CFG["cell17_0_selection_done_here"] = False
CFG["cell17_0_generator_fit_done_here"] = False
CFG["cell17_0_materialization_done_here"] = False
CFG["cell17_0_baseline_scope_definition_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell17_0_max_tabular_cols", 64)
CFG.setdefault("cell17_0_max_sequence_cols", 32)
CFG.setdefault("cell17_0_min_train_finite_rate", 0.05)
CFG.setdefault("cell17_0_min_train_variance", 1e-12)

CFG.setdefault("cell17_0_protocol_core_cols", [
    "router__pkt_total",
    "router__bytes_total",
    "router__tcp_pkt",
    "router__udp_pkt",
    "router__dns_pkt",
    "router__dns_query",
    "router__dns_response",
    "router__arp_pkt",
    "zigbee__pkt_total",
    "zigbee__bytes_total",
    "zigbee__ack",
    "zigbee__cmd",
])

CFG.setdefault("cell17_0_public_release_core_cols", [
    "router__pkt_total",
    "router__bytes_total",
    "router__tcp_pkt",
    "router__udp_pkt",
    "router__dns_pkt",
    "router__dns_query",
    "router__dns_response",
    "router__arp_pkt",
    "zigbee__pkt_total",
    "zigbee__bytes_total",
    "zigbee__ack",
    "zigbee__cmd",
])

CFG.setdefault("cell17_0_event_driver_core_prefixes", [
    "events_in_sec__entity__",
    "events_in_sec__feat__",
])

CFG.setdefault("cell17_0_iot_value_core_keywords", [
    "__sensor__temperature__value",
    "__sensor__humidity__value",
    "__sensor__pressure__value",
    "__sensor__linkquality__value",
    "__sensor__voltage__value",
    "__sensor__battery__value",
    "__sensor__current__value",
    "__sensor__power__value",
    "__sensor__energy__value",
])

CFG.setdefault("cell17_0_exclude_cols_regex_fragments", [
    "sec_epoch",
    "timestamp",
    "datetime",
    "__obs_present",
    "__stale",
    "entity_obs",
    "entity_stale",
])

MAX_TABULAR_COLS_170 = int(CFG.get("cell17_0_max_tabular_cols", 64))
MAX_SEQUENCE_COLS_170 = int(CFG.get("cell17_0_max_sequence_cols", 32))
MIN_FINITE_RATE_170 = float(CFG.get("cell17_0_min_train_finite_rate", 0.05))
MIN_VAR_170 = float(CFG.get("cell17_0_min_train_variance", 1e-12))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
scope_registry_csv = os.path.join(REPORT_DIR_BASE_170, "cell17_0_baseline_scope_registry.csv")
column_registry_csv = os.path.join(REPORT_DIR_BASE_170, "cell17_0_baseline_column_registry.csv")
contract_json = os.path.join(REPORT_DIR_BASE_170, "cell17_0_baseline_fairness_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_170, "cell17_0_baseline_fairness_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_170, "cell17_0_baseline_scope_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_170(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_170(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_170(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_170(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_170(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_170(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_170(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_170(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_170(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_170(payload), f, indent=2, sort_keys=True)

def _sha256_file_170(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_csv_optional_170(path: str):
    if not isinstance(path, str) or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _safe_float_170(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _require_16_5_governance_170(contract: dict):
    if not isinstance(contract, dict):
        raise RuntimeError("[Cell17.0] CELL16_5_FINAL_DECOMPOSITION_CONTRACT is not a dict.")
    version = str(contract.get("version", ""))
    if "cell16_5_a0_a1_a2_final_decomposition_report_v1_2" not in version:
        raise RuntimeError(f"[Cell17.0] Unexpected Cell 16.5 contract version: {version}")

    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})
    if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
        raise RuntimeError("[Cell17.0] Cell 16.5 does not carry Q4 blocked_no_promotion governance.")
    if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
        raise RuntimeError("[Cell17.0] Cell 16.5 indicates Q4-coupled artifacts were used.")

    return version

def _is_time_col_170(c: str) -> bool:
    s = str(c).lower()
    return (
        s in {"sec", "sec_epoch_s__canon", "timestamp", "time", "datetime"}
        or s.endswith("__time")
        or "time_iso" in s
    )

def _excluded_for_baseline_170(c: str) -> bool:
    s = str(c).lower()
    if _is_time_col_170(s):
        return True
    for frag in CFG.get("cell17_0_exclude_cols_regex_fragments", []):
        if str(frag).lower() in s:
            return True
    return False

def _role_group_170(c: str) -> str:
    s = str(c)
    if s.startswith("router__"):
        return "protocol_router"
    if s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__"):
        return "protocol_ota"
    if s.startswith("zigbee__") or s.startswith("zb__"):
        return "protocol_zigbee"
    if s.startswith("zwave__"):
        return "protocol_zwave"
    if s.startswith("events_in_sec__"):
        return "iot_event_drivers"
    if s.startswith("iot__") and (s.endswith("__state") or "__binary_sensor__" in s or "__switch__" in s):
        return "iot_binary_states"
    if s.startswith("iot__") and s.endswith("__value"):
        return "iot_continuous_values"
    if s.startswith("iot__"):
        return "iot_other"
    return "other"

def _numeric_profile_170(df: pd.DataFrame, c: str):
    if c not in df.columns:
        return {"exists": False, "finite_rate": 0.0, "variance": np.nan, "unique_n": 0, "nonzero_rate": np.nan}
    x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(x)
    xf = x[finite]
    return {
        "exists": True,
        "finite_rate": float(finite.mean()) if len(x) else 0.0,
        "variance": float(np.nanvar(xf)) if len(xf) else np.nan,
        "unique_n": int(len(np.unique(xf))) if len(xf) else 0,
        "nonzero_rate": float(np.mean(np.abs(xf) > 1e-12)) if len(xf) else np.nan,
    }

def _eligible_numeric_col_170(df: pd.DataFrame, c: str) -> bool:
    if c not in df.columns or _excluded_for_baseline_170(c):
        return False
    prof = _numeric_profile_170(df, c)
    return bool(
        prof["exists"]
        and prof["finite_rate"] >= MIN_FINITE_RATE_170
        and np.isfinite(prof["variance"])
        and prof["variance"] >= MIN_VAR_170
    )

def _score_col_for_scope_170(df: pd.DataFrame, c: str):
    prof = _numeric_profile_170(df, c)
    finite_rate = _safe_float_170(prof.get("finite_rate"), 0.0)
    variance = _safe_float_170(prof.get("variance"), 0.0)
    nonzero = _safe_float_170(prof.get("nonzero_rate"), 0.0)
    unique_n = _safe_float_170(prof.get("unique_n"), 0.0)
    return float(finite_rate + np.log1p(max(variance, 0.0)) + 0.25 * nonzero + 0.001 * unique_n)

def _dedup_170(seq):
    seen = set()
    out = []
    for x in seq:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _select_existing_eligible_170(df: pd.DataFrame, cols, max_cols=None):
    out = [str(c) for c in cols if str(c) in df.columns and _eligible_numeric_col_170(df, str(c))]
    out = _dedup_170(out)
    if max_cols is not None and len(out) > int(max_cols):
        out = sorted(out, key=lambda c: (-_score_col_for_scope_170(df, c), c))[:int(max_cols)]
    return out

def _select_by_role_170(df: pd.DataFrame, roles, max_cols):
    roles = set(roles)
    cols = [str(c) for c in df.columns.astype(str) if _role_group_170(str(c)) in roles and _eligible_numeric_col_170(df, str(c))]
    cols = _dedup_170(cols)
    if len(cols) > int(max_cols):
        cols = sorted(cols, key=lambda c: (-_score_col_for_scope_170(df, c), c))[:int(max_cols)]
    return cols

def _select_iot_values_by_keywords_170(df: pd.DataFrame, max_cols):
    keys = [str(k).lower() for k in CFG.get("cell17_0_iot_value_core_keywords", [])]
    cols = []
    for c in df.columns.astype(str):
        cl = c.lower()
        if _role_group_170(c) != "iot_continuous_values":
            continue
        if not any(k in cl for k in keys):
            continue
        if _eligible_numeric_col_170(df, c):
            cols.append(c)
    cols = _dedup_170(cols)
    if len(cols) > int(max_cols):
        cols = sorted(cols, key=lambda c: (-_score_col_for_scope_170(df, c), c))[:int(max_cols)]
    return cols

def _select_event_driver_cols_170(df: pd.DataFrame, max_cols):
    prefixes = tuple(str(p) for p in CFG.get("cell17_0_event_driver_core_prefixes", []))
    cols = [str(c) for c in df.columns.astype(str) if str(c).startswith(prefixes) and _eligible_numeric_col_170(df, str(c))]
    cols = _dedup_170(cols)
    if len(cols) > int(max_cols):
        cols = sorted(cols, key=lambda c: (-_score_col_for_scope_170(df, c), c))[:int(max_cols)]
    return cols

def _public_candidate_cols_170():
    rg = _read_csv_optional_170(os.path.join(PUBLIC_Q6_REPORT_DIR_170, "cell15_7a_public_q6_role_group_registry.csv"))
    cols = []
    if len(rg) and "columns_json" in rg.columns:
        for _, r in rg.iterrows():
            role = str(r.get("role_group", ""))
            if role in {"protocol_router", "protocol_zigbee", "iot_event_drivers"}:
                try:
                    parsed = json.loads(str(r.get("columns_json", "[]")))
                    cols.extend([str(c) for c in parsed])
                except Exception:
                    pass

    if not cols:
        for p in [
            os.path.join(OUT_SYN_BASE_170, "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet"),
            globals().get("CELL16_0_PUBLIC_Q6_CPS_PATH", ""),
        ]:
            if isinstance(p, str) and os.path.exists(p):
                try:
                    import pyarrow.parquet as pq
                    cols = [str(c) for c in pq.ParquetFile(p).schema.names]
                    break
                except Exception:
                    try:
                        cols = [str(c) for c in pd.read_parquet(p).columns]
                        break
                    except Exception:
                        pass

    return _dedup_170([c for c in cols if c in df_tr.columns])

def _make_scope_row_170(scope_id, display_name, intended_baselines, cols, fairness_rationale, allowed_metrics, excluded_claims):
    cols = _dedup_170(cols)
    roles = Counter(_role_group_170(c) for c in cols)
    return {
        "scope_id": str(scope_id),
        "display_name": str(display_name),
        "cols_n": int(len(cols)),
        "role_counts": json.dumps(dict(sorted(roles.items())), sort_keys=True),
        "intended_baselines": "|".join(map(str, intended_baselines)),
        "allowed_metrics": "|".join(map(str, allowed_metrics)),
        "excluded_claims": "|".join(map(str, excluded_claims)),
        "fairness_rationale": str(fairness_rationale),
        "TEST_real_values_used": False,
        "synthetic_values_mutated": False,
    }

# ----------------------------------------------------------
# 4) Validate upstream final decomposition governance
# ----------------------------------------------------------
CELL16_5_VERSION_170 = _require_16_5_governance_170(CELL16_5_FINAL_DECOMPOSITION_CONTRACT)
PUBLIC_Q6_DIRECT_STATUS_170 = str(CELL16_5_FINAL_DECOMPOSITION_CONTRACT.get("summary", {}).get("public_direct_privacy_status", ""))
PUBLIC_Q6_DISTINGUISHABILITY_STATUS_170 = str(CELL16_5_FINAL_DECOMPOSITION_CONTRACT.get("summary", {}).get("public_distinguishability_status", ""))
PUBLIC_Q6_STRICT_STATUS_170 = str(CELL16_5_FINAL_DECOMPOSITION_CONTRACT.get("summary", {}).get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 5) Build comparable scopes from TRAIN only
# ----------------------------------------------------------
train_cols = list(map(str, df_tr.columns))
test_cols = set(map(str, df_te.columns))

# Scope 1: protocol core tabular — fair for CTGAN / TabDDPM.
protocol_core_cols = _select_existing_eligible_170(
    df_tr,
    CFG.get("cell17_0_protocol_core_cols", []),
    max_cols=MAX_TABULAR_COLS_170,
)
if len(protocol_core_cols) < 8:
    fallback_protocol_cols = _select_by_role_170(
        df_tr,
        roles=["protocol_router", "protocol_zigbee"],
        max_cols=MAX_TABULAR_COLS_170,
    )
    protocol_core_cols = _dedup_170(protocol_core_cols + fallback_protocol_cols)[:MAX_TABULAR_COLS_170]

# Scope 2: public candidate core — actual public candidate columns when available.
public_candidate_cols_all = _public_candidate_cols_170()
public_candidate_role_cols = [
    c for c in public_candidate_cols_all
    if _role_group_170(c) in {"protocol_router", "protocol_zigbee", "iot_event_drivers"}
    and _eligible_numeric_col_170(df_tr, c)
]
if public_candidate_role_cols:
    public_release_core_cols = sorted(
        _dedup_170(public_candidate_role_cols),
        key=lambda c: (-_score_col_for_scope_170(df_tr, c), c)
    )[:MAX_TABULAR_COLS_170]
else:
    public_protocol_cols = _select_existing_eligible_170(
        df_tr,
        CFG.get("cell17_0_public_release_core_cols", []),
        max_cols=MAX_TABULAR_COLS_170,
    )
    public_event_cols_fallback = _select_event_driver_cols_170(
        df_tr,
        max_cols=max(1, MAX_TABULAR_COLS_170 - len(public_protocol_cols)),
    )
    public_release_core_cols = _dedup_170(public_protocol_cols + public_event_cols_fallback)[:MAX_TABULAR_COLS_170]

public_event_cols = [c for c in public_release_core_cols if _role_group_170(c) == "iot_event_drivers"]

# Scope 3: IoT continuous compact tabular — fair only for row-wise IoT marginal baseline.
iot_continuous_compact_cols = _select_iot_values_by_keywords_170(df_tr, max_cols=MAX_TABULAR_COLS_170)
if len(iot_continuous_compact_cols) < 16:
    fallback_iot_values = _select_by_role_170(df_tr, roles=["iot_continuous_values"], max_cols=MAX_TABULAR_COLS_170)
    iot_continuous_compact_cols = _dedup_170(iot_continuous_compact_cols + fallback_iot_values)[:MAX_TABULAR_COLS_170]

# Scope 4: compact CPS sequence scope — fair for TimeGAN.
sequence_core_cols = _dedup_170(protocol_core_cols[:16] + public_event_cols[:16])[:MAX_SEQUENCE_COLS_170]

# Scope 5: protocol-only temporal compact — fair for TimeGAN protocol comparison.
protocol_sequence_cols = _select_by_role_170(df_tr, roles=["protocol_router", "protocol_zigbee"], max_cols=MAX_SEQUENCE_COLS_170)

scope_defs = {
    "protocol_core_tabular": protocol_core_cols,
    "public_candidate_core_tabular": public_release_core_cols,
    "iot_continuous_compact_tabular": iot_continuous_compact_cols,
    "sequence_core_temporal": sequence_core_cols,
    "protocol_sequence_temporal": protocol_sequence_cols,
}

missing_in_test = {sid: sorted([c for c in cols if c not in test_cols]) for sid, cols in scope_defs.items()}
bad_missing = {k: v for k, v in missing_in_test.items() if v}
if bad_missing:
    raise RuntimeError(
        "[Cell17.0] Some baseline scope columns are missing from TEST reference: "
        f"{ {k: v[:10] for k, v in bad_missing.items()} }"
    )

empty_scopes = [sid for sid, cols in scope_defs.items() if len(cols) == 0]
if empty_scopes:
    raise RuntimeError(f"[Cell17.0] Empty baseline scopes: {empty_scopes}")

# ----------------------------------------------------------
# 6) Scope registry
# ----------------------------------------------------------
scope_rows = [
    _make_scope_row_170(
        scope_id="protocol_core_tabular",
        display_name="Protocol core tabular scope",
        intended_baselines=["CTGAN", "TabDDPM"],
        cols=protocol_core_cols,
        fairness_rationale=(
            "Compares row-wise tabular baselines on protocol columns they can model. "
            "Does not require observability enforcement or Q4 coupling."
        ),
        allowed_metrics=["Q1_marginal", "correlation", "row_level_C2ST"],
        excluded_claims=["Q2_sequence_dynamics", "Q3_observability", "Q4_coupling", "Q6_release_safety"],
    ),
    _make_scope_row_170(
        scope_id="public_candidate_core_tabular",
        display_name="Public candidate core tabular scope",
        intended_baselines=["CTGAN", "TabDDPM"],
        cols=public_release_core_cols,
        fairness_rationale=(
            "Uses the public-candidate role scope where available: router, Zigbee, and event drivers. "
            "Fair for row-wise tabular baselines, but this scope inherits the Cell 16.5 warning that "
            "public strict release remains blocked by distinguishability."
        ),
        allowed_metrics=["Q1_marginal", "correlation", "row_level_C2ST"],
        excluded_claims=["Q3_observability", "Q4_coupling_repair", "full_namespace_assembly", "strict_Q6_release_pass"],
    ),
    _make_scope_row_170(
        scope_id="iot_continuous_compact_tabular",
        display_name="IoT continuous compact tabular scope",
        intended_baselines=["CTGAN", "TabDDPM"],
        cols=iot_continuous_compact_cols,
        fairness_rationale=(
            "Compares tabular baselines on a compact subset of IoT continuous values. "
            "Does not require mask-aware generation, entity-state contracts, or full IoT namespace assembly."
        ),
        allowed_metrics=["Q1_marginal", "correlation", "row_level_C2ST"],
        excluded_claims=["Q2_temporal_sequences", "Q3_observability", "Q4_coupling", "full_IoT_realism"],
    ),
    _make_scope_row_170(
        scope_id="sequence_core_temporal",
        display_name="Compact CPS sequence scope",
        intended_baselines=["TimeGAN"],
        cols=sequence_core_cols,
        fairness_rationale=(
            "Compares sequence baselines on compact protocol/event-driver columns only. "
            "Avoids asking TimeGAN to synthesize the full no-Q4 scientific artifact."
        ),
        allowed_metrics=["Q2_temporal", "sequence_C2ST", "autocorrelation", "transition_rate"],
        excluded_claims=["Q1_full_namespace", "Q3_observability", "Q4_coupling_repair", "Q6_release_safety"],
    ),
    _make_scope_row_170(
        scope_id="protocol_sequence_temporal",
        display_name="Protocol-only sequence scope",
        intended_baselines=["TimeGAN"],
        cols=protocol_sequence_cols,
        fairness_rationale=(
            "Protocol-only sequence comparison for TimeGAN. "
            "This isolates temporal modeling from IoT role/mask complexity."
        ),
        allowed_metrics=["Q2_temporal", "autocorrelation", "transition_rate", "run_length"],
        excluded_claims=["IoT_state_realism", "Q3_observability", "Q4_coupling_repair"],
    ),
]
scope_registry_df = pd.DataFrame(scope_rows)

# ----------------------------------------------------------
# 7) Column registry
# ----------------------------------------------------------
column_rows = []
for scope_id, cols in scope_defs.items():
    for ordinal, c in enumerate(cols):
        prof = _numeric_profile_170(df_tr, c)
        column_rows.append({
            "scope_id": scope_id,
            "ordinal": int(ordinal),
            "col": c,
            "role_group": _role_group_170(c),
            "train_finite_rate": prof["finite_rate"],
            "train_variance": prof["variance"],
            "train_unique_n": prof["unique_n"],
            "train_nonzero_rate": prof["nonzero_rate"],
            "eligible_numeric": bool(_eligible_numeric_col_170(df_tr, c)),
            "in_train": bool(c in df_tr.columns),
            "in_val": bool(c in df_val.columns),
            "in_test": bool(c in df_te.columns),
            "TEST_real_values_used": False,
            "synthetic_values_mutated": False,
        })
column_registry_df = pd.DataFrame(column_rows)

# ----------------------------------------------------------
# 8) Baseline run plan
# ----------------------------------------------------------
baseline_plan = {
    "CTGAN": {
        "cells": ["17.1"],
        "allowed_scopes": [
            "protocol_core_tabular",
            "public_candidate_core_tabular",
            "iot_continuous_compact_tabular",
        ],
        "allowed_metrics": ["Q1_marginal", "correlation", "row_level_C2ST"],
        "not_allowed_claims": [
            "full_CPS_generation",
            "strict_mask_enforcement",
            "Q4_coupling",
            "public_release_privacy_pass",
        ],
    },
    "TabDDPM": {
        "cells": ["17.2"],
        "allowed_scopes": [
            "protocol_core_tabular",
            "public_candidate_core_tabular",
            "iot_continuous_compact_tabular",
        ],
        "allowed_metrics": ["Q1_marginal", "correlation", "row_level_C2ST"],
        "not_allowed_claims": [
            "full_CPS_generation",
            "strict_mask_enforcement",
            "Q4_coupling",
            "public_release_privacy_pass",
        ],
    },
    "TimeGAN": {
        "cells": ["17.3"],
        "allowed_scopes": [
            "sequence_core_temporal",
            "protocol_sequence_temporal",
        ],
        "allowed_metrics": ["Q2_temporal", "autocorrelation", "transition_rate", "run_length"],
        "not_allowed_claims": [
            "full_CPS_generation",
            "Q3_observability",
            "Q4_coupling",
            "public_release_privacy_pass",
        ],
    },
}

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
scope_registry_df.to_csv(scope_registry_csv, index=False)
column_registry_df.to_csv(column_registry_csv, index=False)

contract = {
    "cell": "17.0",
    "version": CELL170_VERSION,
    "role": "external_baseline_fairness_scope_contract_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "baselines_not_compared_to_Q4_coupled_artifact": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": PUBLIC_Q6_DIRECT_STATUS_170,
        "public_distinguishability_status": PUBLIC_Q6_DISTINGUISHABILITY_STATUS_170,
        "public_strict_combined_status": PUBLIC_Q6_STRICT_STATUS_170,
        "public_candidate_columns_available_n": int(len(public_candidate_cols_all)),
    },
    "upstream_contract_versions": {
        "cell16_5": CELL16_5_VERSION_170,
    },
    "scope_summary": scope_registry_df.to_dict("records"),
    "column_registry_rows": int(len(column_registry_df)),
    "baseline_plan": baseline_plan,
    "fairness_principles": [
        "External baselines are compared only on target scopes they can reasonably model.",
        "Tabular baselines are not evaluated as full role-aware temporal CPS generators.",
        "Sequence baselines are not required to synthesize the full no-Q4 scientific artifact.",
        "Q4 coupling remains blocked/no-promotion and is not used as a baseline target.",
        "Q3 strict observability and Q6 release safety are pipeline governance dimensions, not baseline failure points.",
        "Baseline results should be reported as external baselines on comparable scopes, not as direct replacements for the full pipeline.",
    ],
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "baseline_scope_definition_done_here": True,
    },
    "outputs": {
        "scope_registry_csv": scope_registry_csv,
        "column_registry_csv": column_registry_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_170(contract_json, contract)
_write_json_170(contract_canonical_json, contract)

manifest = {
    "cell": "17.0",
    "version": CELL170_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "scope_ids": scope_registry_df["scope_id"].astype(str).tolist(),
    "baseline_plan": baseline_plan,
    "strict_contract": contract["strict_contract"],
}
_write_json_170(manifest_json, manifest)

hashes = {
    "scope_registry_csv_sha256": _sha256_file_170(scope_registry_csv),
    "column_registry_csv_sha256": _sha256_file_170(column_registry_csv),
    "contract_json_sha256": _sha256_file_170(contract_json),
    "contract_canonical_json_sha256": _sha256_file_170(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_170(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_170(contract_json, contract)
_write_json_170(contract_canonical_json, contract)
_write_json_170(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals
# ----------------------------------------------------------
globals()["CELL170_VERSION"] = CELL170_VERSION
globals()["CELL17_0_BASELINE_SCOPE_REGISTRY_DF"] = scope_registry_df
globals()["CELL17_0_BASELINE_COLUMN_REGISTRY_DF"] = column_registry_df
globals()["CELL17_0_BASELINE_FAIRNESS_CONTRACT"] = contract
globals()["CELL17_0_BASELINE_PLAN"] = baseline_plan

globals()["CELL17_0_BASELINE_SCOPE_REGISTRY_CSV"] = scope_registry_csv
globals()["CELL17_0_BASELINE_COLUMN_REGISTRY_CSV"] = column_registry_csv
globals()["CELL17_0_BASELINE_CONTRACT_JSON"] = contract_json
globals()["CELL17_0_BASELINE_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL17_0_BASELINE_MANIFEST_JSON"] = manifest_json

for scope_id, cols in scope_defs.items():
    globals()[f"CELL17_0_SCOPE_{scope_id.upper()}_COLS"] = list(cols)

log(
    "[Cell17.0] External baseline comparable scopes complete | "
    f"scopes={len(scope_registry_df)} | "
    f"column_rows={len(column_registry_df)} | "
    f"q4_status=blocked_no_promotion | "
    f"public_strict_q6={PUBLIC_Q6_STRICT_STATUS_170}"
)
log(f"[Cell17.0] Scope summary | {scope_registry_df[['scope_id', 'cols_n', 'intended_baselines']].to_dict('records')}")
log(f"[Cell17.0] Saved scope registry: {scope_registry_csv}")
log(f"[Cell17.0] Saved column registry: {column_registry_csv}")
log(f"[Cell17.0] Saved fairness contract: {contract_json}")
log(f"[Cell17.0] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell17.0] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "baseline_scope_definition_done_here=True"
)
log("--- END: Cell 17.0 - External baseline comparable target scopes (v1.1 no-Q4-promotion strict) ---")

gc.collect()