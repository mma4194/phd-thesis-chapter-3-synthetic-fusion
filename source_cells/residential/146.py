# ==========================================================
# CELL 16.4 - Q4 coupling decomposition
# v1.1 STUDY-THESIS strict no-Q4-promotion cross-modal coupling decomposition
#
# Role:
#   - Decompose cross-modal coupling quality across:
#       broad/generic Q4 failure,
#       manifest-informed Zigbee pre-repair,
#       initial Zigbee repair candidate,
#       A0 Zigbee-safe candidate,
#       A1/A1a/A1b/A1c/router expansion deferrals,
#       final Q4 no-promotion decision,
#       public Q6-mitigated candidate context.
#
# Final governance:
#   - Q4 final status = blocked_no_promotion.
#   - No Q4-coupled protocol/CPS artifact is authoritative.
#   - A0 Zigbee-safe is diagnostic/improved but not promoted.
#   - A1a/A1b/A1c/router expansion is deferred/no-materialization.
#
# Scientific contract:
#   - No generation.
#   - No selection.
#   - No repair.
#   - No synthetic mutation.
#   - TEST real values are used only through existing Q4 QA reports/contracts.
#   - This cell reports Q4 decomposition; it does not change the pipeline.
#
# Outputs:
#   reports/cell16_4_q4_coupling_decomposition_by_stage.csv
#   reports/cell16_4_q4_coupling_decomposition_by_tier.csv
#   reports/cell16_4_q4_coupling_decomposition_by_pair.csv
#   reports/cell16_4_q4_coupling_gain_loss_ledger.csv
#   reports/cell16_4_q4_coupling_decomposition_contract.json
#   artifacts/contracts/cell16_4_q4_coupling_decomposition_contract_v1_1_THESIS.json
#   artifacts/cell16_4_q4_coupling_decomposition_manifest.json
# ==========================================================

log("--- START: Cell 16.4 - Q4 coupling decomposition (v1.1 no-Q4-promotion strict) ---")

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
_required_164 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL16_0_ARTIFACT_REGISTRY_DF",
    "CELL16_0_METRIC_SOURCE_REGISTRY_DF",
    "CELL16_0_DECOMPOSITION_INPUT_CONTRACT",
    "CELL16_1_Q1_VARIANT_SUMMARY_DF",
    "CELL16_1_Q1_DECOMPOSITION_CONTRACT",
    "CELL16_2_Q2_VARIANT_SUMMARY_DF",
    "CELL16_2_Q2_DECOMPOSITION_CONTRACT",
    "CELL16_3_Q3_VARIANT_SUMMARY_DF",
    "CELL16_3_Q3_DECOMPOSITION_CONTRACT",
]
_missing_164 = [k for k in _required_164 if k not in globals()]
if _missing_164:
    raise RuntimeError(f"[Cell16.4] Missing required globals: {_missing_164}")

ORIGINAL_OUTDIR_164 = str(OUTDIR)
ORIGINAL_OUT_SYN_164 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_164 = str(REPORT_DIR)

def _resolve_project_root_164(outdir, report_dir, out_syn):
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

    raise RuntimeError("[Cell16.4] Could not resolve canonical project root.")

PROJECT_ROOT_164 = _resolve_project_root_164(ORIGINAL_OUTDIR_164, ORIGINAL_REPORT_DIR_164, ORIGINAL_OUT_SYN_164)
OUTDIR_BASE_164 = PROJECT_ROOT_164
OUT_SYN_BASE_164 = os.path.join(PROJECT_ROOT_164, "synthetic")
REPORT_DIR_BASE_164 = os.path.join(PROJECT_ROOT_164, "reports")
ARTDIR_BASE_164 = os.path.join(PROJECT_ROOT_164, "artifacts")
CONTRACT_DIR_BASE_164 = os.path.join(ARTDIR_BASE_164, "contracts")

PUBLIC_Q6_ROOT_164 = os.path.join(PROJECT_ROOT_164, "q6_public_reaudit_v1")
PUBLIC_Q6_REPORT_DIR_164 = os.path.join(PUBLIC_Q6_ROOT_164, "reports")
PUBLIC_Q6_CONTRACT_DIR_164 = os.path.join(PUBLIC_Q6_ROOT_164, "artifacts", "contracts")

os.makedirs(REPORT_DIR_BASE_164, exist_ok=True)
os.makedirs(ARTDIR_BASE_164, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_164, exist_ok=True)

SEED = int(SEED)

CELL164_VERSION = "cell16_4_q4_coupling_decomposition_v1_1_no_q4_promotion"

CFG["cell16_4_version"] = CELL164_VERSION
CFG["cell16_4_Q4_final_status"] = "blocked_no_promotion"
CFG["cell16_4_Q4_coupled_artifacts_used"] = False
CFG["cell16_4_TEST_real_values_used_through_existing_QA_only"] = True
CFG["cell16_4_TEST_real_values_used_for_materialization"] = False
CFG["cell16_4_synthetic_values_mutated"] = False
CFG["cell16_4_selection_done_here"] = False
CFG["cell16_4_generator_fit_done_here"] = False
CFG["cell16_4_materialization_done_here"] = False
CFG["cell16_4_decomposition_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell16_4_eta_similarity_pass", 0.70)
CFG.setdefault("cell16_4_eta_similarity_warning", 0.50)
CFG.setdefault("cell16_4_profile_similarity_pass", 0.70)
CFG.setdefault("cell16_4_profile_similarity_warning", 0.50)
CFG.setdefault("cell16_4_lag_peak_error_pass", 2.0)
CFG.setdefault("cell16_4_lag_peak_error_warning", 5.0)
CFG.setdefault("cell16_4_response_window_rate_error_pass", 0.10)
CFG.setdefault("cell16_4_response_window_rate_error_warning", 0.25)

ETA_PASS_164 = float(CFG.get("cell16_4_eta_similarity_pass", 0.70))
ETA_WARN_164 = float(CFG.get("cell16_4_eta_similarity_warning", 0.50))
PROF_PASS_164 = float(CFG.get("cell16_4_profile_similarity_pass", 0.70))
PROF_WARN_164 = float(CFG.get("cell16_4_profile_similarity_warning", 0.50))
LAG_PASS_164 = float(CFG.get("cell16_4_lag_peak_error_pass", 2.0))
LAG_WARN_164 = float(CFG.get("cell16_4_lag_peak_error_warning", 5.0))
RESP_PASS_164 = float(CFG.get("cell16_4_response_window_rate_error_pass", 0.10))
RESP_WARN_164 = float(CFG.get("cell16_4_response_window_rate_error_warning", 0.25))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
stage_summary_csv = os.path.join(REPORT_DIR_BASE_164, "cell16_4_q4_coupling_decomposition_by_stage.csv")
tier_summary_csv = os.path.join(REPORT_DIR_BASE_164, "cell16_4_q4_coupling_decomposition_by_tier.csv")
pair_summary_csv = os.path.join(REPORT_DIR_BASE_164, "cell16_4_q4_coupling_decomposition_by_pair.csv")
gain_loss_ledger_csv = os.path.join(REPORT_DIR_BASE_164, "cell16_4_q4_coupling_gain_loss_ledger.csv")
source_registry_csv = os.path.join(REPORT_DIR_BASE_164, "cell16_4_q4_coupling_source_registry.csv")
contract_json = os.path.join(REPORT_DIR_BASE_164, "cell16_4_q4_coupling_decomposition_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_164, "cell16_4_q4_coupling_decomposition_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_164, "cell16_4_q4_coupling_decomposition_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_164(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_164(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_164(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_164(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_164(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_164(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_164(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_164(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_164(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_164(payload), f, indent=2, sort_keys=True)

def _sha256_file_164(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_164(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _read_csv_optional_164(path: str):
    if not _exists_164(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _read_json_optional_164(path: str):
    if not _exists_164(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _safe_float_164(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _safe_int_164(x, default=0):
    try:
        if pd.isna(x):
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _first_col_164(df: pd.DataFrame, candidates):
    if not isinstance(df, pd.DataFrame):
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _contract_field_164(path, *keys, default=None):
    d = _read_json_optional_164(path)
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _require_contract_governance_164(contract: dict, name: str, expected_version_substring: str):
    if not isinstance(contract, dict):
        raise RuntimeError(f"[Cell16.4] {name} is not a dict.")
    version = str(contract.get("version", ""))
    if expected_version_substring not in version:
        raise RuntimeError(
            f"[Cell16.4] Unexpected {name} version. Expected substring={expected_version_substring}, got={version}"
        )
    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})
    status = str(q4.get("final_q4_status", strict.get("Q4_final_status", "")))
    used = bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True)))
    if status != "blocked_no_promotion":
        raise RuntimeError(f"[Cell16.4] {name} does not carry Q4 blocked_no_promotion governance.")
    if used:
        raise RuntimeError(f"[Cell16.4] {name} indicates Q4-coupled artifacts were used.")
    return version

def _resolve_input_164(filename, context="base", required=False):
    filename = str(filename)
    candidates = []
    if context in {"base", "both"}:
        candidates.append(os.path.join(REPORT_DIR_BASE_164, filename))
    if context in {"public", "both"}:
        candidates.append(os.path.join(PUBLIC_Q6_REPORT_DIR_164, filename))
    # fallback
    candidates.extend([
        os.path.join(REPORT_DIR_BASE_164, filename),
        os.path.join(PUBLIC_Q6_REPORT_DIR_164, filename),
        os.path.join(ORIGINAL_REPORT_DIR_164, filename),
    ])
    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]
    for p in candidates:
        if _exists_164(p):
            return p, candidates
    if required:
        raise RuntimeError(f"[Cell16.4] Missing required input report: {filename}; searched={candidates}")
    return candidates[0], candidates

def _resolve_contract_164(filename, context="base", required=False):
    candidates = []
    if context in {"base", "both"}:
        candidates.append(os.path.join(CONTRACT_DIR_BASE_164, filename))
    if context in {"public", "both"}:
        candidates.append(os.path.join(PUBLIC_Q6_CONTRACT_DIR_164, filename))
    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]
    for p in candidates:
        if _exists_164(p):
            return p, candidates
    if required:
        raise RuntimeError(f"[Cell16.4] Missing required input contract: {filename}; searched={candidates}")
    return candidates[0], candidates

def _stage_status_164(eta, lag, resp, prof, source_status=""):
    source_status = str(source_status or "").strip().lower()
    if source_status in {"pass", "warning", "fatal", "blocker"}:
        if source_status == "blocker":
            return "fatal", "source_blocker"
        return source_status, f"source_status_{source_status}"

    fatal = []
    warn = []

    if np.isfinite(eta):
        if eta < ETA_WARN_164:
            fatal.append("ETA_similarity_fatal")
        elif eta < ETA_PASS_164:
            warn.append("ETA_similarity_warning")

    if np.isfinite(prof):
        if prof < PROF_WARN_164:
            fatal.append("profile_similarity_fatal")
        elif prof < PROF_PASS_164:
            warn.append("profile_similarity_warning")

    if np.isfinite(lag):
        if lag > LAG_WARN_164:
            fatal.append("lag_peak_error_fatal")
        elif lag > LAG_PASS_164:
            warn.append("lag_peak_error_warning")

    if np.isfinite(resp):
        if resp > RESP_WARN_164:
            fatal.append("response_window_rate_error_fatal")
        elif resp > RESP_PASS_164:
            warn.append("response_window_rate_error_warning")

    if fatal:
        return "fatal", "|".join(fatal)
    if warn:
        return "warning", "|".join(warn)
    return "pass", "within_q4_coupling_gates"

def _norm_stage_metric_columns_164(df: pd.DataFrame, stage_id: str, stage_order: int, stage_label: str, stage_governance_status: str):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame()

    d = df.copy()
    d.columns = d.columns.astype(str)

    pair_col = _first_col_164(d, ["pair_id", "repair_candidate_id", "candidate_id", "manifest_pair_id"])
    anchor_col = _first_col_164(d, ["driver_col", "anchor_col", "anchor_driver_col", "iot_col"])
    protocol_col = _first_col_164(d, ["protocol_col", "signal", "target_protocol_col"])
    protocol_tier_col = _first_col_164(d, ["protocol_tier", "tier", "tier_consensus"])
    manifest_tier_col = _first_col_164(d, ["manifest_tier", "repair_manifest_group", "coupling_tier"])

    eta_col = _first_col_164(d, ["ETA_similarity", "eta_similarity", "post_ETA_similarity", "mean_ETA_similarity"])
    lag_col = _first_col_164(d, ["lag_peak_error", "eta_peak_error", "post_lag_peak_error", "mean_lag_peak_error"])
    resp_col = _first_col_164(d, ["response_window_rate_error", "post_response_window_rate_error", "mean_response_window_rate_error"])
    profile_col = _first_col_164(d, [
        "manifest_window_profile_similarity",
        "manifest_profile_similarity",
        "post_manifest_profile_similarity",
        "mean_manifest_profile_similarity",
    ])

    status_col = _first_col_164(d, [
        "q4_status",
        "manifest_q4_status",
        "A1a_q4_status",
        "post_manifest_q4_status",
        "eta_status",
        "lag_response_status",
        "cross_corr_status",
        "publication_status",
        "status",
    ])

    blocker_col = _first_col_164(d, [
        "q4_publication_blocker",
        "manifest_q4_publication_blocker",
        "A1a_q4_publication_blocker",
        "post_publication_blocker",
        "publication_blocker",
        "publication_blocker_flag",
    ])

    rows = []
    for i, r in d.iterrows():
        eta = _safe_float_164(r.get(eta_col), np.nan) if eta_col else np.nan
        lag = _safe_float_164(r.get(lag_col), np.nan) if lag_col else np.nan
        resp = _safe_float_164(r.get(resp_col), np.nan) if resp_col else np.nan
        prof = _safe_float_164(r.get(profile_col), np.nan) if profile_col else np.nan

        source_status = str(r.get(status_col, "")) if status_col else ""
        status, reasons = _stage_status_164(eta, lag, resp, prof, source_status=source_status)

        if blocker_col:
            blocker_raw = r.get(blocker_col, False)
            if isinstance(blocker_raw, str):
                blocker = blocker_raw.strip().lower() in {"1", "true", "yes", "blocker", "fatal"}
            else:
                blocker = bool(blocker_raw)
        else:
            blocker = bool(status == "fatal")

        if status == "fatal":
            blocker = True

        rows.append({
            "stage_id": stage_id,
            "stage_order": int(stage_order),
            "stage_label": stage_label,
            "stage_governance_status": stage_governance_status,
            "pair_id": str(r.get(pair_col, f"{stage_id}_row_{i:05d}")) if pair_col else f"{stage_id}_row_{i:05d}",
            "anchor_col": str(r.get(anchor_col, "")) if anchor_col else "",
            "protocol_col": str(r.get(protocol_col, "")) if protocol_col else "",
            "protocol_tier": str(r.get(protocol_tier_col, "unknown")) if protocol_tier_col else "unknown",
            "manifest_tier": str(r.get(manifest_tier_col, "")) if manifest_tier_col else "",
            "ETA_similarity": eta,
            "lag_peak_error": lag,
            "response_window_rate_error": resp,
            "manifest_profile_similarity": prof,
            "q4_status": status,
            "q4_status_reasons": reasons,
            "q4_publication_blocker": bool(blocker),
            "source_status": source_status,
            "source_rows": int(len(d)),
            "TEST_real_values_used_through_existing_QA_only": True,
            "synthetic_values_mutated": False,
        })

    return pd.DataFrame(rows)

def _stage_score_164(g: pd.DataFrame):
    if not isinstance(g, pd.DataFrame) or len(g) == 0:
        return np.nan

    parts = []
    if g["ETA_similarity"].notna().any():
        parts.append(1.0 - pd.to_numeric(g["ETA_similarity"], errors="coerce").clip(lower=0, upper=1).fillna(0.0))
    if g["manifest_profile_similarity"].notna().any():
        parts.append(1.0 - pd.to_numeric(g["manifest_profile_similarity"], errors="coerce").clip(lower=0, upper=1).fillna(0.0))
    if g["lag_peak_error"].notna().any():
        parts.append((pd.to_numeric(g["lag_peak_error"], errors="coerce").fillna(LAG_WARN_164) / max(LAG_WARN_164, 1e-9)).clip(lower=0, upper=2))
    if g["response_window_rate_error"].notna().any():
        parts.append((pd.to_numeric(g["response_window_rate_error"], errors="coerce").fillna(RESP_WARN_164) / max(RESP_WARN_164, 1e-9)).clip(lower=0, upper=2))

    if not parts:
        return np.nan
    return float(pd.concat(parts, axis=1).mean(axis=1).mean())

def _summary_group_164(g: pd.DataFrame):
    status_counts = g["q4_status"].astype(str).value_counts().to_dict()
    blocker_n = int(g["q4_publication_blocker"].fillna(False).astype(bool).sum())

    return pd.Series({
        "pairs_total": int(len(g)),
        "pairs_evaluable": int(len(g)),
        "pass_n": int(status_counts.get("pass", 0)),
        "warning_n": int(status_counts.get("warning", 0)),
        "fatal_n": int(status_counts.get("fatal", 0)),
        "publication_blocker_n": int(blocker_n),
        "pair_pass_rate": float((g["q4_status"].astype(str).eq("pass")).mean()) if len(g) else np.nan,
        "pair_warning_or_pass_rate": float((~g["q4_status"].astype(str).eq("fatal")).mean()) if len(g) else np.nan,
        "mean_ETA_similarity": float(pd.to_numeric(g["ETA_similarity"], errors="coerce").mean()) if len(g) else np.nan,
        "median_ETA_similarity": float(pd.to_numeric(g["ETA_similarity"], errors="coerce").median()) if len(g) else np.nan,
        "mean_lag_peak_error": float(pd.to_numeric(g["lag_peak_error"], errors="coerce").mean()) if len(g) else np.nan,
        "mean_response_window_rate_error": float(pd.to_numeric(g["response_window_rate_error"], errors="coerce").mean()) if len(g) else np.nan,
        "mean_manifest_profile_similarity": float(pd.to_numeric(g["manifest_profile_similarity"], errors="coerce").mean()) if len(g) else np.nan,
        "q4_coupling_penalty_score": _stage_score_164(g),
    })

def _load_stage_164(stage_id, stage_order, stage_label, filename, governance_status, required=False):
    path, searched = _resolve_input_164(filename, context="base", required=required)
    df = _read_csv_optional_164(path)

    source_row = {
        "stage_id": stage_id,
        "stage_order": int(stage_order),
        "stage_label": stage_label,
        "filename": filename,
        "path": path,
        "searched_paths": "|".join(map(str, searched)),
        "exists": isinstance(df, pd.DataFrame),
        "rows": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "cols": int(len(df.columns)) if isinstance(df, pd.DataFrame) else 0,
        "required": bool(required),
        "governance_status": governance_status,
    }

    if isinstance(df, pd.DataFrame):
        norm = _norm_stage_metric_columns_164(df, stage_id, stage_order, stage_label, governance_status)
    else:
        norm = pd.DataFrame()

    return source_row, norm

# ----------------------------------------------------------
# 4) Validate upstream no-Q4 governance contracts
# ----------------------------------------------------------
CELL16_0_CONTRACT_VERSION_164 = _require_contract_governance_164(
    CELL16_0_DECOMPOSITION_INPUT_CONTRACT,
    "CELL16_0_DECOMPOSITION_INPUT_CONTRACT",
    "cell16_0_a0_a1_a2_decomposition_input_contract_v1_1",
)
CELL16_1_CONTRACT_VERSION_164 = _require_contract_governance_164(
    CELL16_1_Q1_DECOMPOSITION_CONTRACT,
    "CELL16_1_Q1_DECOMPOSITION_CONTRACT",
    "cell16_1_q1_marginal_decomposition_v1_1",
)
CELL16_2_CONTRACT_VERSION_164 = _require_contract_governance_164(
    CELL16_2_Q2_DECOMPOSITION_CONTRACT,
    "CELL16_2_Q2_DECOMPOSITION_CONTRACT",
    "cell16_2_q2_temporal_decomposition_v1_1",
)
CELL16_3_CONTRACT_VERSION_164 = _require_contract_governance_164(
    CELL16_3_Q3_DECOMPOSITION_CONTRACT,
    "CELL16_3_Q3_DECOMPOSITION_CONTRACT",
    "cell16_3_q3_observability_decomposition_v1_1",
)

# ----------------------------------------------------------
# 5) Load Q4 report sources
# ----------------------------------------------------------
stage_specs = [
    ("S0_broad_generic_Q4", 0, "Broad all-pair generic Q4", "cell14_11_final_q4_evidence_table.csv", "blocked_broad_generic_q4", True),
    ("S1_manifest_zigbee_pre_repair", 1, "Manifest-informed Zigbee pre-repair", "cell13_6_manifest_zigbee_pair_metrics.csv", "pre_repair_reference", False),
    ("S2_initial_manifest_zigbee_repair", 2, "Initial manifest Zigbee repair", "cell14_3_manifest_q4_post_pair_metrics.csv", "candidate_not_final", False),
    ("S3_A0_zigbee_safe_candidate", 3, "A0 Zigbee-safe candidate", "cell14_6_A0_candidate_manifest_q4_pair_metrics.csv", "blocked_no_promotion", False),
    ("S3a_A0_fatal_diagnostic", 4, "A0 fatal blocker diagnostic", "cell14_6a_A0_fatal_blocker_diagnostic.csv", "diagnostic_only", False),
    ("S4_A1_router_expansion_audit", 5, "A1/router expansion audit", "cell14_8_A1_router_audit.csv", "audit_only_deferred", False),
]

source_rows = []
pair_tables = []

for spec in stage_specs:
    src, norm = _load_stage_164(*spec)
    source_rows.append(src)
    if len(norm):
        # Special handling for 14.11 evidence table: keep only Q4 stage/candidate rows that have pair metrics.
        if src["stage_id"] == "S0_broad_generic_Q4":
            # The final evidence table includes candidate-scope rows, not all pair rows.
            # Keep it as candidate-level/pseudo-pair evidence if no explicit pair table exists.
            pass
        pair_tables.append(norm)

source_load_df = pd.DataFrame(source_rows)

if not pair_tables:
    raise RuntimeError(
        "[Cell16.4] No Q4 pair/stage metric sources could be loaded. "
        f"source_loads={source_load_df.to_dict('records')}"
    )

pair_summary_df = pd.concat(pair_tables, axis=0, ignore_index=True)

# ----------------------------------------------------------
# 6) Load final decision, freshness, and Q6 public context
# ----------------------------------------------------------
decision_ledger_path, decision_ledger_searched_164 = _resolve_input_164("cell14_10_final_q4_decision_ledger.csv", context="base", required=True)
candidate_comparison_path, candidate_comparison_searched_164 = _resolve_input_164("cell14_10_final_q4_candidate_comparison.csv", context="base", required=True)
q4_statement_path, q4_statement_searched_164 = _resolve_input_164("cell14_11_final_q4_paper_statement.txt", context="base", required=False)
q4_freshness_contract_path, q4_freshness_contract_searched_164 = _resolve_contract_164("cell14_12_q4_no_promotion_freshness_audit_v2_1_THESIS.json", context="base", required=False)

public_15_4_path, public_15_4_searched_164 = _resolve_input_164("cell15_4_q6_role_release_safety.csv", context="public", required=False)
public_15_4b_path, public_15_4b_searched_164 = _resolve_input_164("cell15_4b_q6_policy_split_role_summary.csv", context="public", required=False)
public_15_4b_summary_path, public_15_4b_summary_searched_164 = _resolve_input_164("cell15_4b_q6_policy_split_summary.csv", context="public", required=False)

decision_df = _read_csv_optional_164(decision_ledger_path)
candidate_df = _read_csv_optional_164(candidate_comparison_path)
public_15_4_df = _read_csv_optional_164(public_15_4_path)
public_15_4b_df = _read_csv_optional_164(public_15_4b_path)
public_15_4b_summary_df = _read_csv_optional_164(public_15_4b_summary_path)

q4_freshness_status = _contract_field_164(q4_freshness_contract_path, "final_q4_status", default="")
q4_freshness_decision = _contract_field_164(q4_freshness_contract_path, "decision", default="")
q4_freshness_severity = _contract_field_164(q4_freshness_contract_path, "decision_severity", default="")

# ----------------------------------------------------------
# 7) Stage and tier summaries
# ----------------------------------------------------------
stage_summary_df = (
    pair_summary_df
    .groupby(["stage_order", "stage_id", "stage_label", "stage_governance_status"], dropna=False)
    .apply(_summary_group_164)
    .reset_index()
    .sort_values("stage_order")
    .reset_index(drop=True)
)

tier_summary_df = (
    pair_summary_df
    .groupby(["stage_order", "stage_id", "stage_label", "stage_governance_status", "protocol_tier", "manifest_tier"], dropna=False)
    .apply(_summary_group_164)
    .reset_index()
    .sort_values(["stage_order", "protocol_tier", "manifest_tier"])
    .reset_index(drop=True)
)

# ----------------------------------------------------------
# 8) Gain/loss ledger
# ----------------------------------------------------------
ledger_rows = []

stage_by_id = {str(r["stage_id"]): r for _, r in stage_summary_df.iterrows()}

for a, b, label in [
    ("S1_manifest_zigbee_pre_repair", "S2_initial_manifest_zigbee_repair", "manifest_pre_to_initial_repair"),
    ("S1_manifest_zigbee_pre_repair", "S3_A0_zigbee_safe_candidate", "manifest_pre_to_A0_candidate"),
    ("S2_initial_manifest_zigbee_repair", "S3_A0_zigbee_safe_candidate", "initial_repair_to_A0_candidate"),
    ("S3_A0_zigbee_safe_candidate", "S4_A1_router_expansion_audit", "A0_candidate_to_A1_router_audit"),
]:
    if a in stage_by_id and b in stage_by_id:
        ra = stage_by_id[a]
        rb = stage_by_id[b]
        ledger_rows.append({
            "ledger_item": f"q4_stage_delta::{label}",
            "from_stage": a,
            "to_stage": b,
            "delta_mean_ETA_similarity": _safe_float_164(rb.get("mean_ETA_similarity"), np.nan) - _safe_float_164(ra.get("mean_ETA_similarity"), np.nan),
            "delta_mean_lag_peak_error": _safe_float_164(rb.get("mean_lag_peak_error"), np.nan) - _safe_float_164(ra.get("mean_lag_peak_error"), np.nan),
            "delta_mean_response_window_rate_error": _safe_float_164(rb.get("mean_response_window_rate_error"), np.nan) - _safe_float_164(ra.get("mean_response_window_rate_error"), np.nan),
            "delta_mean_manifest_profile_similarity": _safe_float_164(rb.get("mean_manifest_profile_similarity"), np.nan) - _safe_float_164(ra.get("mean_manifest_profile_similarity"), np.nan),
            "delta_publication_blocker_n": _safe_int_164(rb.get("publication_blocker_n"), 0) - _safe_int_164(ra.get("publication_blocker_n"), 0),
            "delta_q4_penalty_score": _safe_float_164(rb.get("q4_coupling_penalty_score"), np.nan) - _safe_float_164(ra.get("q4_coupling_penalty_score"), np.nan),
            "interpretation": "Positive ETA/profile deltas improve; negative lag/response/blocker/penalty deltas improve.",
        })

# Final Q4 governance.
ledger_rows.append({
    "ledger_item": "final_q4_governance",
    "final_q4_status": "blocked_no_promotion",
    "q4_coupled_artifacts_used": False,
    "q4_freshness_status": q4_freshness_status,
    "q4_freshness_decision": q4_freshness_decision,
    "q4_freshness_severity": q4_freshness_severity,
    "interpretation": "Final Q4 governance blocks promotion; no Q4-coupled artifact is authoritative.",
})

if isinstance(decision_df, pd.DataFrame) and len(decision_df):
    for _, r in decision_df.iterrows():
        ledger_rows.append({
            "ledger_item": "cell14_10_final_q4_decision",
            "final_q4_status": str(r.get("final_q4_status", r.get("q4_final_status", ""))),
            "accepted_candidate": str(r.get("accepted_candidate", "")),
            "promotion_done": bool(r.get("promotion_done", False)) if "promotion_done" in decision_df.columns else False,
            "artifact_copy_done": bool(r.get("artifact_copy_done_here", False)) if "artifact_copy_done_here" in decision_df.columns else False,
            "interpretation": "Final Q4 no-promotion decision ledger row from Cell 14.10.",
        })

if isinstance(candidate_df, pd.DataFrame) and len(candidate_df):
    for _, r in candidate_df.iterrows():
        candidate_name = str(r.get("candidate_scope", r.get("candidate", r.get("candidate_name", ""))))
        ledger_rows.append({
            "ledger_item": "cell14_10_q4_candidate_comparison",
            "candidate": candidate_name,
            "status": str(r.get("status", "")),
            "final_publication_candidate": bool(r.get("final_publication_candidate", False)) if "final_publication_candidate" in candidate_df.columns else False,
            "publication_blocker_n": _safe_int_164(r.get("publication_blocker_n", r.get("q4_publication_blocker_n", np.nan)), 0),
            "pairs_total": _safe_int_164(r.get("pairs_total", np.nan), 0),
            "mean_ETA_similarity": _safe_float_164(r.get("mean_ETA_similarity"), np.nan),
            "mean_lag_peak_error": _safe_float_164(r.get("mean_lag_peak_error"), np.nan),
            "mean_response_window_rate_error": _safe_float_164(r.get("mean_response_window_rate_error"), np.nan),
            "mean_manifest_profile_similarity": _safe_float_164(r.get("mean_manifest_profile_similarity"), np.nan),
            "interpretation": "Candidate comparison from final no-promotion Q4 decision.",
        })

# Public Q6 context.
if isinstance(public_15_4_df, pd.DataFrame) and len(public_15_4_df):
    status_counts = public_15_4_df["release_status"].astype(str).value_counts().to_dict() if "release_status" in public_15_4_df.columns else {}
    ledger_rows.append({
        "ledger_item": "public_q6_strict_release_context",
        "release_pass_n": int(status_counts.get("release_pass", 0)),
        "release_warning_n": int(status_counts.get("release_warning", 0)),
        "release_blocker_n": int(status_counts.get("release_blocker", 0)),
        "warning_roles": "|".join(sorted(public_15_4_df.loc[public_15_4_df["release_status"].astype(str).eq("release_warning"), "role_group"].astype(str).tolist())) if "release_status" in public_15_4_df.columns else "",
        "blocker_roles": "|".join(sorted(public_15_4_df.loc[public_15_4_df["release_status"].astype(str).eq("release_blocker"), "role_group"].astype(str).tolist())) if "release_status" in public_15_4_df.columns else "",
        "interpretation": "Public Q6 strict combined status after role-restricted mitigation.",
    })

if isinstance(public_15_4b_summary_df, pd.DataFrame) and len(public_15_4b_summary_df):
    metric_map = dict(zip(public_15_4b_summary_df["metric"].astype(str), public_15_4b_summary_df["value"]))
    ledger_rows.append({
        "ledger_item": "public_q6_policy_split_context",
        "direct_privacy_overall_status": str(metric_map.get("direct_privacy_overall_status", "")),
        "direct_privacy_blocker_roles": str(metric_map.get("direct_privacy_blocker_roles", "")),
        "direct_privacy_warning_roles": str(metric_map.get("direct_privacy_warning_roles", "")),
        "distinguishability_overall_status": str(metric_map.get("distinguishability_overall_status", "")),
        "distinguishability_blocker_roles": str(metric_map.get("distinguishability_blocker_roles", "")),
        "strict_combined_overall_status": str(metric_map.get("strict_combined_overall_status", "")),
        "interpretation": "Public candidate separates direct no-copy/DCR privacy from synthetic-real distinguishability.",
    })

# Worst remaining A0 diagnostic/candidate pairs.
a0_pairs = pair_summary_df[pair_summary_df["stage_id"].astype(str).eq("S3_A0_zigbee_safe_candidate")].copy()
if len(a0_pairs):
    a0_pairs["pair_penalty"] = (
        (1.0 - pd.to_numeric(a0_pairs["ETA_similarity"], errors="coerce").clip(0, 1).fillna(0.0))
        + (1.0 - pd.to_numeric(a0_pairs["manifest_profile_similarity"], errors="coerce").clip(0, 1).fillna(0.0))
        + (pd.to_numeric(a0_pairs["lag_peak_error"], errors="coerce").fillna(LAG_WARN_164) / max(LAG_WARN_164, 1e-9)).clip(0, 2)
        + (pd.to_numeric(a0_pairs["response_window_rate_error"], errors="coerce").fillna(RESP_WARN_164) / max(RESP_WARN_164, 1e-9)).clip(0, 2)
    )
    for _, r in a0_pairs.sort_values("pair_penalty", ascending=False).head(20).iterrows():
        ledger_rows.append({
            "ledger_item": "worst_remaining_A0_candidate_q4_pair",
            "stage_id": str(r["stage_id"]),
            "pair_id": str(r["pair_id"]),
            "anchor_col": str(r["anchor_col"]),
            "protocol_col": str(r["protocol_col"]),
            "protocol_tier": str(r["protocol_tier"]),
            "q4_status": str(r["q4_status"]),
            "q4_publication_blocker": bool(r["q4_publication_blocker"]),
            "pair_penalty": _safe_float_164(r["pair_penalty"], np.nan),
            "ETA_similarity": _safe_float_164(r["ETA_similarity"], np.nan),
            "lag_peak_error": _safe_float_164(r["lag_peak_error"], np.nan),
            "response_window_rate_error": _safe_float_164(r["response_window_rate_error"], np.nan),
            "manifest_profile_similarity": _safe_float_164(r["manifest_profile_similarity"], np.nan),
            "interpretation": "Worst residual Q4 pair in A0 candidate; A0 remains blocked/no-promotion.",
        })

gain_loss_ledger_df = pd.DataFrame(ledger_rows)

# ----------------------------------------------------------
# 9) Save outputs
# ----------------------------------------------------------
pair_summary_df.to_csv(pair_summary_csv, index=False)
stage_summary_df.to_csv(stage_summary_csv, index=False)
tier_summary_df.to_csv(tier_summary_csv, index=False)
gain_loss_ledger_df.to_csv(gain_loss_ledger_csv, index=False)
source_load_df.to_csv(source_registry_csv, index=False)

# ----------------------------------------------------------
# 10) Contract / manifest
# ----------------------------------------------------------
best_stage = (
    stage_summary_df.sort_values("q4_coupling_penalty_score", na_position="last").iloc[0].to_dict()
    if len(stage_summary_df)
    else {}
)

contract = {
    "cell": "16.4",
    "version": CELL164_VERSION,
    "role": "q4_cross_modal_coupling_decomposition_no_q4_promotion",
    "quality_dimension": "Q4_cross_modal_coupling_quality",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "A0_promoted": False,
        "A1a_deferred": True,
        "legacy_q4_artifacts_non_authoritative": True,
    },
    "upstream_contract_versions": {
        "cell16_0": CELL16_0_CONTRACT_VERSION_164,
        "cell16_1": CELL16_1_CONTRACT_VERSION_164,
        "cell16_2": CELL16_2_CONTRACT_VERSION_164,
        "cell16_3": CELL16_3_CONTRACT_VERSION_164,
    },
    "summary": {
        "source_loads": source_load_df.to_dict("records"),
        "pair_metric_rows": int(len(pair_summary_df)),
        "stage_summary_rows": int(len(stage_summary_df)),
        "tier_summary_rows": int(len(tier_summary_df)),
        "gain_loss_ledger_rows": int(len(gain_loss_ledger_df)),
        "best_stage_by_q4_penalty": best_stage,
        "project_root": PROJECT_ROOT_164,
        "base_report_dir": REPORT_DIR_BASE_164,
        "public_q6_report_dir": PUBLIC_Q6_REPORT_DIR_164,
        "q4_freshness_contract": q4_freshness_contract_path,
        "q4_freshness_status": q4_freshness_status,
        "q4_freshness_decision": q4_freshness_decision,
        "q4_freshness_severity": q4_freshness_severity,
    },
    "method": {
        "metrics": [
            "ETA similarity",
            "lag peak error",
            "response-window rate error",
            "manifest/profile similarity",
            "publication blocker count",
        ],
        "stage_order": [
            "S0 broad/generic Q4 failure",
            "S1 manifest-informed Zigbee pre-repair",
            "S2 initial Zigbee repair candidate",
            "S3 A0 Zigbee-safe candidate blocked/no-promotion",
            "S4 A1/router expansion audit deferred",
        ],
        "interpretation": (
            "Q4 quality is decomposed as staged evidence. The final conclusion is governance-level "
            "blocked_no_promotion, not an accepted Q4-coupled artifact."
        ),
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_through_existing_QA_only": True,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "decomposition_done_here": True,
    },
    "outputs": {
        "stage_summary_csv": stage_summary_csv,
        "tier_summary_csv": tier_summary_csv,
        "pair_summary_csv": pair_summary_csv,
        "gain_loss_ledger_csv": gain_loss_ledger_csv,
        "source_registry_csv": source_registry_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_164(contract_json, contract)
_write_json_164(contract_canonical_json, contract)

manifest = {
    "cell": "16.4",
    "version": CELL164_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
}
_write_json_164(manifest_json, manifest)

hashes = {
    "stage_summary_csv_sha256": _sha256_file_164(stage_summary_csv),
    "tier_summary_csv_sha256": _sha256_file_164(tier_summary_csv),
    "pair_summary_csv_sha256": _sha256_file_164(pair_summary_csv),
    "gain_loss_ledger_csv_sha256": _sha256_file_164(gain_loss_ledger_csv),
    "source_registry_csv_sha256": _sha256_file_164(source_registry_csv),
    "contract_json_sha256": _sha256_file_164(contract_json),
    "contract_canonical_json_sha256": _sha256_file_164(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_164(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_164(contract_json, contract)
_write_json_164(contract_canonical_json, contract)
_write_json_164(manifest_json, manifest)

# ----------------------------------------------------------
# 11) Export globals
# ----------------------------------------------------------
globals()["CELL164_VERSION"] = CELL164_VERSION
globals()["CELL16_4_Q4_PAIR_SUMMARY_DF"] = pair_summary_df
globals()["CELL16_4_Q4_STAGE_SUMMARY_DF"] = stage_summary_df
globals()["CELL16_4_Q4_TIER_SUMMARY_DF"] = tier_summary_df
globals()["CELL16_4_Q4_GAIN_LOSS_LEDGER_DF"] = gain_loss_ledger_df
globals()["CELL16_4_Q4_SOURCE_REGISTRY_DF"] = source_load_df
globals()["CELL16_4_Q4_DECOMPOSITION_CONTRACT"] = contract

globals()["CELL16_4_Q4_PAIR_SUMMARY_CSV"] = pair_summary_csv
globals()["CELL16_4_Q4_STAGE_SUMMARY_CSV"] = stage_summary_csv
globals()["CELL16_4_Q4_TIER_SUMMARY_CSV"] = tier_summary_csv
globals()["CELL16_4_Q4_GAIN_LOSS_LEDGER_CSV"] = gain_loss_ledger_csv
globals()["CELL16_4_Q4_SOURCE_REGISTRY_CSV"] = source_registry_csv
globals()["CELL16_4_Q4_CONTRACT_JSON"] = contract_json
globals()["CELL16_4_Q4_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL16_4_Q4_MANIFEST_JSON"] = manifest_json

log(
    "[Cell16.4] Q4 coupling decomposition complete | "
    f"q4_status=blocked_no_promotion | "
    f"pair_rows={len(pair_summary_df)} | "
    f"stages={stage_summary_df['stage_id'].nunique() if len(stage_summary_df) else 0} | "
    f"ledger_rows={len(gain_loss_ledger_df)}"
)
if len(stage_summary_df):
    show_cols = [
        "stage_id", "stage_governance_status", "pairs_total", "pass_n", "warning_n",
        "fatal_n", "publication_blocker_n", "mean_ETA_similarity",
        "mean_lag_peak_error", "mean_response_window_rate_error",
        "mean_manifest_profile_similarity", "q4_coupling_penalty_score"
    ]
    show_cols = [c for c in show_cols if c in stage_summary_df.columns]
    log(f"[Cell16.4] Stage summary | {stage_summary_df[show_cols].to_dict('records')}")
log(f"[Cell16.4] Saved stage summary: {stage_summary_csv}")
log(f"[Cell16.4] Saved tier summary: {tier_summary_csv}")
log(f"[Cell16.4] Saved pair summary: {pair_summary_csv}")
log(f"[Cell16.4] Saved gain/loss ledger: {gain_loss_ledger_csv}")
log(f"[Cell16.4] Saved source registry: {source_registry_csv}")
log(f"[Cell16.4] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell16.4] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_through_existing_QA_only=True | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "decomposition_done_here=True"
)
log("--- END: Cell 16.4 - Q4 coupling decomposition (v1.1 no-Q4-promotion strict) ---")

gc.collect()