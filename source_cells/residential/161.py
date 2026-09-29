# ==========================================================
# CELL 19.0 - Full CPS comprehensive evaluation input contract
# v1.1 STUDY-THESIS strict no-Q4-promotion evaluation registry
#
# Role:
#   - Gather all artifacts and metric/report sources required for a
#     comprehensive CPS evidence/evaluation registry.
#   - Centralize evidence from Q1/Q2/Q3/Q4/Q6/baselines/traceability.
#   - Validate artifact existence and metric-source availability.
#   - Do NOT recompute metrics here.
#   - Do NOT mutate synthetic data.
#
# Important final governance:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative.
#   - The primary scientific artifact is the no-Q4 scientific CPS artifact.
#   - Q4 evidence is report/ledger evidence only, not a promoted artifact.
#   - Public Q6 candidate is role-restricted and strict-release blocked by
#     distinguishability.
#
# Outputs:
#   reports/cell19_0_full_cps_eval_artifact_registry.csv
#   reports/cell19_0_full_cps_eval_metric_source_registry.csv
#   reports/cell19_0_full_cps_eval_contract.json
#   artifacts/contracts/cell19_0_full_cps_eval_contract_v1_1_THESIS.json
#   artifacts/cell19_0_full_cps_eval_manifest.json
# ==========================================================

log("--- START: Cell 19.0 - Full CPS comprehensive evaluation input contract (v1.1 no-Q4-promotion strict) ---")

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
_required_190 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL18_3_PUBLICATION_DASHBOARD_CONTRACT",
]
_missing_190 = [k for k in _required_190 if k not in globals()]
if _missing_190:
    raise RuntimeError(f"[Cell19.0] Missing required globals: {_missing_190}")

ORIGINAL_OUTDIR_190 = str(OUTDIR)
ORIGINAL_OUT_SYN_190 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_190 = str(REPORT_DIR)

# ----------------------------------------------------------
# 0.1) Canonical project/report/synthetic resolver
# ----------------------------------------------------------
def _resolve_project_root_190(outdir, report_dir, out_syn):
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
        if os.path.isdir(os.path.join(c, "synthetic")) and os.path.isdir(os.path.join(c, "reports")):
            return c

    raise RuntimeError("[Cell19.0] Could not resolve canonical project root.")

PROJECT_ROOT_190 = _resolve_project_root_190(ORIGINAL_OUTDIR_190, ORIGINAL_REPORT_DIR_190, ORIGINAL_OUT_SYN_190)
MAIN_SYN_DIR_190 = os.path.join(PROJECT_ROOT_190, "synthetic")
MAIN_REPORT_DIR_190 = os.path.join(PROJECT_ROOT_190, "reports")
MAIN_ART_DIR_190 = os.path.join(PROJECT_ROOT_190, "artifacts")
CONTRACT_DIR_190 = os.path.join(MAIN_ART_DIR_190, "contracts")

PUBLIC_REAUDIT_ROOTS_190 = [
    os.path.join(PROJECT_ROOT_190, "q6_public_reaudit_v1"),
    os.path.join(PROJECT_ROOT_190, "q6_public_reaudit"),
]
PUBLIC_REAUDIT_REPORT_DIRS_190 = [os.path.join(p, "reports") for p in PUBLIC_REAUDIT_ROOTS_190]
PUBLIC_REAUDIT_SYN_DIRS_190 = [os.path.join(p, "synthetic") for p in PUBLIC_REAUDIT_ROOTS_190]

os.makedirs(MAIN_REPORT_DIR_190, exist_ok=True)
os.makedirs(MAIN_ART_DIR_190, exist_ok=True)
os.makedirs(CONTRACT_DIR_190, exist_ok=True)

SEED = int(SEED)

CELL190_VERSION = "cell19_0_full_cps_eval_input_contract_v1_1_no_q4_promotion"

CFG["cell19_0_version"] = CELL190_VERSION
CFG["cell19_0_Q4_final_status"] = "blocked_no_promotion"
CFG["cell19_0_Q4_coupled_artifacts_used"] = False
CFG["cell19_0_TEST_real_values_used"] = False
CFG["cell19_0_TEST_real_values_used_for_materialization"] = False
CFG["cell19_0_synthetic_values_mutated"] = False
CFG["cell19_0_selection_done_here"] = False
CFG["cell19_0_generator_fit_done_here"] = False
CFG["cell19_0_materialization_done_here"] = False
CFG["cell19_0_full_cps_eval_input_contract_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell19_0_fail_on_missing_required_artifacts", True)
CFG.setdefault("cell19_0_fail_on_missing_required_metric_sources", False)
CFG.setdefault("cell19_0_fail_on_artifact_shape_mismatch", True)

# ----------------------------------------------------------
# 2) Helpers
# ----------------------------------------------------------
def _json_sanitize_190(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_190(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_190(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_190(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_190(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_190(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_190(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_190(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_190(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_190(payload), f, indent=2, sort_keys=True)

def _sha256_file_190(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_190(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _safe_bool_190(x, default=False):
    if isinstance(x, bool):
        return bool(x)
    if pd.isna(x):
        return bool(default)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", ""}:
        return False
    return bool(default)

def _parquet_shape_190(path: str):
    if not _exists_190(path):
        return None, None, ""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return int(pf.metadata.num_rows), int(len(pf.schema_arrow.names)), "pyarrow_metadata"
    except Exception:
        try:
            d = pd.read_parquet(path)
            return int(d.shape[0]), int(d.shape[1]), "pandas_read"
        except Exception as e:
            return None, None, f"shape_read_failed:{type(e).__name__}:{e}"

def _csv_shape_190(path: str):
    if not _exists_190(path):
        return None, None, ""
    try:
        d = pd.read_csv(path, nrows=5)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            n_lines = sum(1 for _ in f)
        rows = max(0, n_lines - 1)
        return int(rows), int(d.shape[1]), "csv_header_line_count"
    except Exception as e:
        return None, None, f"csv_shape_failed:{type(e).__name__}:{e}"

def _json_exists_summary_190(path: str):
    if not _exists_190(path):
        return None, None, ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return 1, len(obj.keys()), "json_dict"
        if isinstance(obj, list):
            return len(obj), None, "json_list"
        return 1, None, type(obj).__name__
    except Exception as e:
        return None, None, f"json_read_failed:{type(e).__name__}:{e}"

def _parquet_cols_190(path: str):
    if not _exists_190(path):
        return []
    try:
        import pyarrow.parquet as pq
        return list(map(str, pq.ParquetFile(path).schema_arrow.names))
    except Exception:
        try:
            return list(map(str, pd.read_parquet(path).columns))
        except Exception:
            return []

def _role_counts_from_cols_190(cols):
    counts = Counter()
    for c in cols:
        s = str(c)
        if s in {"sec", "sec_epoch_s__canon", "timestamp", "time", "datetime"}:
            counts["time"] += 1
        elif s.startswith("router__"):
            counts["protocol_router"] += 1
        elif s.startswith("ota__") or s.startswith("ota24__") or s.startswith("ota5__"):
            counts["protocol_ota"] += 1
        elif s.startswith("zigbee__") or s.startswith("zb__"):
            counts["protocol_zigbee"] += 1
        elif s.startswith("zwave__"):
            counts["protocol_zwave"] += 1
        elif s.startswith("events_in_sec__"):
            counts["iot_event_drivers"] += 1
        elif s.startswith("iot__") and ("obs" in s or "stale" in s or "present" in s):
            counts["iot_observability_masks"] += 1
        elif s.startswith("iot__") and (s.endswith("__state") or "__binary_sensor__" in s or "__switch__" in s):
            counts["iot_binary_states"] += 1
        elif s.startswith("iot__") and s.endswith("__value"):
            counts["iot_continuous_values"] += 1
        elif s.startswith("iot__"):
            counts["iot_other"] += 1
        else:
            counts["other"] += 1
    return dict(sorted(counts.items()))

def _first_existing_190(paths):
    for p in paths:
        if _exists_190(str(p)):
            return str(p)
    return str(paths[0]) if paths else ""

def _resolve_report_file_190(filename, prefer="main"):
    filename = str(filename)
    if prefer == "public":
        dirs = PUBLIC_REAUDIT_REPORT_DIRS_190 + [MAIN_REPORT_DIR_190]
    else:
        dirs = [MAIN_REPORT_DIR_190] + PUBLIC_REAUDIT_REPORT_DIRS_190

    candidates = []
    for d in dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            candidates.append(p)

    if not candidates:
        return os.path.join(MAIN_REPORT_DIR_190, filename)

    return sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)[0]

def _artifact_row_190(artifact_id, artifact_role, path, required=True, expected_rows=None, expected_cols_min=None, description=""):
    path = str(path or "")
    exists = _exists_190(path)
    rows = None
    cols = None
    read_status = ""
    role_counts = {}

    if exists and path.endswith(".parquet"):
        rows, cols, read_status = _parquet_shape_190(path)
        role_counts = _role_counts_from_cols_190(_parquet_cols_190(path))
    elif exists and path.endswith(".csv"):
        rows, cols, read_status = _csv_shape_190(path)
    elif exists and path.endswith(".json"):
        rows, cols, read_status = _json_exists_summary_190(path)
    else:
        read_status = "missing" if not exists else "unknown_extension"

    row_ok = True
    col_ok = True
    if expected_rows is not None and rows is not None:
        row_ok = int(rows) == int(expected_rows)
    if expected_cols_min is not None and cols is not None:
        col_ok = int(cols) >= int(expected_cols_min)

    return {
        "artifact_id": str(artifact_id),
        "artifact_role": str(artifact_role),
        "path": path,
        "required": bool(required),
        "exists": bool(exists),
        "rows": rows,
        "cols": cols,
        "expected_rows": expected_rows,
        "expected_cols_min": expected_cols_min,
        "row_count_ok": bool(row_ok),
        "col_count_ok": bool(col_ok),
        "read_status": read_status,
        "role_counts": json.dumps(role_counts, sort_keys=True),
        "sha256": _sha256_file_190(path),
        "description": str(description),
    }

def _metric_source_row_190(source_id, quality_dimension, path, required=True, metric_families=None, expected_columns=None, description=""):
    path = str(path or "")
    exists = _exists_190(path)
    rows = None
    cols = None
    read_status = ""
    missing_expected_columns = []

    if exists and path.endswith(".csv"):
        rows, cols, read_status = _csv_shape_190(path)
        try:
            header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
        except Exception:
            header = []
        if expected_columns:
            missing_expected_columns = [c for c in expected_columns if c not in header]
    elif exists and path.endswith(".json"):
        rows, cols, read_status = _json_exists_summary_190(path)
    elif exists and path.endswith(".txt"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            rows = len(text.splitlines())
            read_status = "txt_read"
        except Exception as e:
            read_status = f"txt_read_failed:{type(e).__name__}:{e}"
    else:
        read_status = "missing" if not exists else "unknown_extension"

    return {
        "source_id": str(source_id),
        "quality_dimension": str(quality_dimension),
        "path": path,
        "required": bool(required),
        "exists": bool(exists),
        "rows": rows,
        "cols": cols,
        "read_status": read_status,
        "metric_families": "|".join(metric_families or []),
        "expected_columns": "|".join(expected_columns or []),
        "missing_expected_columns": "|".join(missing_expected_columns),
        "schema_ok": bool(len(missing_expected_columns) == 0),
        "sha256": _sha256_file_190(path),
        "description": str(description),
    }

# ----------------------------------------------------------
# 3) Validate upstream Cell 18.3 governance
# ----------------------------------------------------------
cell18_3_version_190 = str(CELL18_3_PUBLICATION_DASHBOARD_CONTRACT.get("version", ""))
if "cell18_3_publication_readiness_dashboard_v1_2" not in cell18_3_version_190:
    raise RuntimeError(f"[Cell19.0] Unexpected Cell 18.3 contract version: {cell18_3_version_190}")

q4 = CELL18_3_PUBLICATION_DASHBOARD_CONTRACT.get("q4_governance", {})
strict = CELL18_3_PUBLICATION_DASHBOARD_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell19.0] Cell 18.3 does not carry Q4 blocked_no_promotion governance.")
if _safe_bool_190(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell19.0] Cell 18.3 indicates Q4-coupled artifacts were used.")
if _safe_bool_190(strict.get("global_publication_ready_used", False)):
    raise RuntimeError("[Cell19.0] Cell 18.3 used global publication_ready status, which is forbidden.")

q6_public_context = CELL18_3_PUBLICATION_DASHBOARD_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6_public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6_public_context.get("public_distinguishability_status", ""))
public_strict_status = str(q6_public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 4) Resolve artifact paths, no-Q4 first
# ----------------------------------------------------------
scientific_no_q4_cps_path = _first_existing_190([
    globals().get("CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH", ""),
    os.path.join(MAIN_SYN_DIR_190, "CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet"),
    os.path.join(MAIN_SYN_DIR_190, "CPS_SYNTHETIC_TEST_FINAL_NO_Q4_COUPLED_REMOVED.parquet"),
])
scientific_no_q4_protocol_path = _first_existing_190([
    globals().get("CELL16_0_SCIENTIFIC_NO_Q4_PROTOCOL_PATH", ""),
    os.path.join(MAIN_SYN_DIR_190, "PROTOCOL_SYN_TEST_FINAL_NO_Q4.parquet"),
])
public_q6_cps_path = _first_existing_190([
    globals().get("CELL16_0_PUBLIC_Q6_CPS_PATH", ""),
    os.path.join(MAIN_SYN_DIR_190, "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet"),
])
public_q6_protocol_path = _first_existing_190([
    globals().get("CELL16_0_PUBLIC_Q6_PROTOCOL_PATH", ""),
    os.path.join(MAIN_SYN_DIR_190, "PROTOCOL_SYN_TEST_PUBLIC_Q6_MITIGATED.parquet"),
])
public_q6_iot_path = _first_existing_190([
    globals().get("CELL16_0_PUBLIC_Q6_IOT_PATH", ""),
    os.path.join(MAIN_SYN_DIR_190, "IOT_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet"),
])

real_test_ref_path = str(CFG.get("cell19_0_real_test_reference_path", ""))

# Infer expected rows from no-Q4 scientific artifact first; fallback public.
rows1, cols1, _ = _parquet_shape_190(scientific_no_q4_cps_path)
rows2, cols2, _ = _parquet_shape_190(public_q6_cps_path)
expected_test_rows = rows1 if rows1 is not None else rows2

# ----------------------------------------------------------
# 5) Output paths
# ----------------------------------------------------------
artifact_registry_csv = os.path.join(MAIN_REPORT_DIR_190, "cell19_0_full_cps_eval_artifact_registry.csv")
metric_source_registry_csv = os.path.join(MAIN_REPORT_DIR_190, "cell19_0_full_cps_eval_metric_source_registry.csv")
contract_json = os.path.join(MAIN_REPORT_DIR_190, "cell19_0_full_cps_eval_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_190, "cell19_0_full_cps_eval_contract_v1_1_THESIS.json")
manifest_json = os.path.join(MAIN_ART_DIR_190, "cell19_0_full_cps_eval_manifest.json")

# ----------------------------------------------------------
# 6) Artifact registry
# ----------------------------------------------------------
artifact_rows = [
    _artifact_row_190(
        artifact_id="scientific_no_q4_cps",
        artifact_role="primary_scientific_internal_cps_no_q4",
        path=scientific_no_q4_cps_path,
        required=True,
        expected_rows=expected_test_rows,
        expected_cols_min=100,
        description="Primary scientific/internal no-Q4 CPS synthetic TEST artifact. Q4-coupled artifact is not authoritative.",
    ),
    _artifact_row_190(
        artifact_id="scientific_no_q4_protocol",
        artifact_role="scientific_protocol_component_no_q4",
        path=scientific_no_q4_protocol_path,
        required=True,
        expected_rows=expected_test_rows,
        expected_cols_min=1,
        description="No-Q4 protocol component for scientific/internal artifact track.",
    ),
    _artifact_row_190(
        artifact_id="public_q6_cps",
        artifact_role="role_restricted_public_candidate",
        path=public_q6_cps_path,
        required=True,
        expected_rows=expected_test_rows,
        expected_cols_min=20,
        description="Role-restricted public Q6 candidate; strict release remains distinguishability-blocked.",
    ),
    _artifact_row_190(
        artifact_id="public_q6_protocol",
        artifact_role="public_protocol_component",
        path=public_q6_protocol_path,
        required=True,
        expected_rows=expected_test_rows,
        expected_cols_min=1,
        description="Public Q6 protocol component.",
    ),
    _artifact_row_190(
        artifact_id="public_q6_iot",
        artifact_role="public_iot_component",
        path=public_q6_iot_path,
        required=True,
        expected_rows=expected_test_rows,
        expected_cols_min=1,
        description="Public Q6 IoT/event-driver component.",
    ),
]

optional_artifacts = [
    ("A0_protocol_test", "protocol_baseline_A0_context_only", os.path.join(MAIN_SYN_DIR_190, "A0_SYN_TEST.parquet")),
    ("A1_protocol_test", "protocol_backbone_A1_context_only", os.path.join(MAIN_SYN_DIR_190, "A1_SYN_TEST.parquet")),
    ("A2_protocol_test", "protocol_hybrid_A2_context_only", os.path.join(MAIN_SYN_DIR_190, "A2_SYN_TEST.parquet")),
    ("iot_full_test", "full_iot_namespace_context_only", os.path.join(MAIN_SYN_DIR_190, "IOT_FULL_SYNTHETIC_TEST.parquet")),
    ("real_test_reference", "optional_real_test_reference", real_test_ref_path),
]
for aid, role, path in optional_artifacts:
    artifact_rows.append(_artifact_row_190(
        artifact_id=aid,
        artifact_role=role,
        path=path,
        required=False,
        expected_rows=expected_test_rows if path else None,
        expected_cols_min=None,
        description="Optional upstream/context artifact if available; not a Q4-promoted artifact.",
    ))

artifact_registry_df = pd.DataFrame(artifact_rows)

# ----------------------------------------------------------
# 7) Metric source registry
# ----------------------------------------------------------
metric_sources = [
    _metric_source_row_190(
        source_id="master_results_ledger",
        quality_dimension="claim_traceability",
        path=_resolve_report_file_190("MASTER_RESULTS_LEDGER_FOR_PAPER.csv"),
        required=True,
        metric_families=["permitted_claim", "forbidden_claim", "selection_split", "evaluation_split"],
        expected_columns=["permitted_claim", "forbidden_claim", "selection_split", "evaluation_split", "TEST_used_for_selection", "TEST_used_for_repair"],
        description="Master results ledger used to derive claim registry.",
    ),
    _metric_source_row_190(
        source_id="cell18_3_claim_traceability",
        quality_dimension="claim_traceability",
        path=_resolve_report_file_190("claim_traceability_manifest.csv"),
        required=True,
        metric_families=["claim_metric", "decision_grade", "claim_scoped_status"],
        expected_columns=["claim_id", "claim_text", "quality_dimension", "metric_name", "metric_value", "decision_grade", "claim_final_status"],
        description="Final claim-to-artifact traceability manifest.",
    ),
    _metric_source_row_190(
        source_id="cell18_3_publication_dashboard",
        quality_dimension="publication_readiness",
        path=_resolve_report_file_190("publication_readiness_dashboard.csv"),
        required=True,
        metric_families=["claim_scoped_dashboard"],
        expected_columns=["quality_dimension", "claims_n", "dashboard_status"],
        description="Final claim-scoped publication dashboard.",
    ),
    _metric_source_row_190(
        source_id="cell18_3_publication_blockers",
        quality_dimension="publication_readiness",
        path=_resolve_report_file_190("publication_blockers.csv"),
        required=True,
        metric_families=["claim_warnings", "claim_caveats", "required_resolution"],
        expected_columns=["claim_id", "quality_dimension", "readiness_severity", "publication_status", "required_resolution"],
        description="Final blockers/caveats ledger.",
    ),

    # Q1/Q2/Q3/Q4 decomposition and evidence sources.
    _metric_source_row_190("cell16_1_q1_variant_summary", "Q1_marginal", _resolve_report_file_190("cell16_1_q1_marginal_decomposition_by_variant.csv"), True, ["Q1_penalty", "KS", "WassNorm", "C2ST"], ["variant", "q1_penalty_score"], "Q1 variant summary."),
    _metric_source_row_190("cell16_2_q2_variant_summary", "Q2_temporal", _resolve_report_file_190("cell16_2_q2_temporal_decomposition_by_variant.csv"), True, ["Q2_penalty", "autocorrelation", "run_length"], ["variant", "q2_temporal_penalty_score"], "Q2 variant summary."),
    _metric_source_row_190("cell16_3_q3_variant_summary", "Q3_observability", _resolve_report_file_190("cell16_3_q3_observability_decomposition_by_variant.csv"), True, ["Q3_penalty", "observability"], ["variant", "q3_observability_penalty_score"], "Q3 variant summary."),
    _metric_source_row_190("cell16_4_q4_stage_summary", "Q4_coupling", _resolve_report_file_190("cell16_4_q4_coupling_decomposition_by_stage.csv"), True, ["Q4_stage", "blocker_count", "ETA", "lag"], ["stage_id", "publication_blocker_n"], "Q4 no-promotion stage evidence."),
    _metric_source_row_190("cell16_5_dimension_matrix", "A0_A1_A2_decomposition", _resolve_report_file_190("cell16_5_a0_a1_a2_quality_dimension_matrix.csv"), True, ["dimension_status"], ["quality_dimension"], "Final dimension matrix."),
    _metric_source_row_190("cell16_5_key_findings", "A0_A1_A2_decomposition", _resolve_report_file_190("cell16_5_a0_a1_a2_key_findings.csv"), True, ["findings", "paper_wording"], ["finding"], "Key findings and wording."),

    # Public Q6 re-audit / release evidence.
    _metric_source_row_190("cell15_1_no_copy", "Q6_privacy_no_copy", _resolve_report_file_190("cell15_1_no_copy_window_metrics.csv", prefer="public"), False, ["no_copy"], [], "Q6 no-copy evidence if available."),
    _metric_source_row_190("cell15_2_dcr_nndr", "Q6_privacy_DCR_NNDR", _resolve_report_file_190("cell15_2_dcr_nndr_role_metrics.csv", prefer="public"), False, ["DCR", "NNDR"], [], "Q6 DCR/NNDR evidence if available."),
    _metric_source_row_190("cell15_3_mia", "Q6_privacy_MIA", _resolve_report_file_190("cell15_3_mia_role_metrics.csv", prefer="public"), False, ["MIA", "distinguishability"], [], "Q6 MIA/distinguishability evidence if available."),
    _metric_source_row_190("cell15_4_release_safety", "Q6_release_safety", _resolve_report_file_190("cell15_4_q6_role_release_safety.csv", prefer="public"), False, ["release_status"], [], "Q6 release status evidence if available."),

    # Baselines.
    _metric_source_row_190("cell17_4_baseline_metrics", "external_baselines", _resolve_report_file_190("cell17_4_baseline_metric_by_artifact.csv"), True, ["same_scope_penalty", "temporal_blocked_C2ST"], ["baseline", "scope_id", "fair_scope_penalty_score"], "External baseline metrics."),
    _metric_source_row_190("cell17_6_baseline_sanitizer", "external_baselines", _resolve_report_file_190("cell17_6_baseline_publication_safe_summary.csv"), True, ["headline_comparable", "partial_limited"], ["headline_fully_comparable_scope_n", "pipeline_headline_wins_n"], "Baseline claim sanitizer."),
]

metric_source_registry_df = pd.DataFrame(metric_sources)

# ----------------------------------------------------------
# 8) Validations
# ----------------------------------------------------------
missing_required_artifacts = artifact_registry_df[
    artifact_registry_df["required"].astype(bool) & (~artifact_registry_df["exists"].astype(bool))
].copy()
shape_bad_required_artifacts = artifact_registry_df[
    artifact_registry_df["required"].astype(bool)
    & artifact_registry_df["exists"].astype(bool)
    & ((~artifact_registry_df["row_count_ok"].astype(bool)) | (~artifact_registry_df["col_count_ok"].astype(bool)))
].copy()
missing_required_metric_sources = metric_source_registry_df[
    metric_source_registry_df["required"].astype(bool) & (~metric_source_registry_df["exists"].astype(bool))
].copy()
schema_bad_metric_sources = metric_source_registry_df[
    metric_source_registry_df["required"].astype(bool)
    & metric_source_registry_df["exists"].astype(bool)
    & (~metric_source_registry_df["schema_ok"].astype(bool))
].copy()

# No Q4-coupled artifacts as required or primary.
q4_coupled_required_artifacts = artifact_registry_df[
    artifact_registry_df["required"].astype(bool)
    & artifact_registry_df["path"].astype(str).str.contains("Q4_COUPLED|FINAL_Q4|q4_coupled", case=False, regex=True, na=False)
].copy()
if len(q4_coupled_required_artifacts):
    raise RuntimeError(
        "[Cell19.0] Unsafe required Q4-coupled artifact detected: "
        f"{q4_coupled_required_artifacts[['artifact_id','path']].to_dict('records')}"
    )

if len(missing_required_artifacts) and bool(CFG.get("cell19_0_fail_on_missing_required_artifacts", True)):
    raise RuntimeError(
        "[Cell19.0] Missing required no-Q4/public artifacts: "
        f"{missing_required_artifacts[['artifact_id', 'path']].to_dict('records')}"
    )

if len(shape_bad_required_artifacts) and bool(CFG.get("cell19_0_fail_on_artifact_shape_mismatch", True)):
    raise RuntimeError(
        "[Cell19.0] Required artifact shape mismatch: "
        f"{shape_bad_required_artifacts[['artifact_id', 'rows', 'cols', 'expected_rows', 'expected_cols_min', 'row_count_ok', 'col_count_ok']].to_dict('records')}"
    )

if len(missing_required_metric_sources) and bool(CFG.get("cell19_0_fail_on_missing_required_metric_sources", False)):
    raise RuntimeError(
        "[Cell19.0] Missing required metric sources: "
        f"{missing_required_metric_sources[['source_id', 'path']].to_dict('records')}"
    )

# ----------------------------------------------------------
# 9) Save registries
# ----------------------------------------------------------
artifact_registry_df.to_csv(artifact_registry_csv, index=False)
metric_source_registry_df.to_csv(metric_source_registry_csv, index=False)

metric_sources_by_dimension = (
    metric_source_registry_df.groupby("quality_dimension")["source_id"]
    .apply(lambda s: sorted(map(str, s.tolist())))
    .to_dict()
)

artifact_summary = {
    "artifacts_total": int(len(artifact_registry_df)),
    "required_artifacts_total": int(artifact_registry_df["required"].astype(bool).sum()),
    "required_artifacts_existing": int((artifact_registry_df["required"].astype(bool) & artifact_registry_df["exists"].astype(bool)).sum()),
    "optional_artifacts_existing": int((~artifact_registry_df["required"].astype(bool) & artifact_registry_df["exists"].astype(bool)).sum()),
    "missing_required_artifacts_n": int(len(missing_required_artifacts)),
    "shape_bad_required_artifacts_n": int(len(shape_bad_required_artifacts)),
}

metric_source_summary = {
    "metric_sources_total": int(len(metric_source_registry_df)),
    "required_metric_sources_total": int(metric_source_registry_df["required"].astype(bool).sum()),
    "required_metric_sources_existing": int((metric_source_registry_df["required"].astype(bool) & metric_source_registry_df["exists"].astype(bool)).sum()),
    "missing_required_metric_sources_n": int(len(missing_required_metric_sources)),
    "schema_bad_required_metric_sources_n": int(len(schema_bad_metric_sources)),
    "metric_sources_by_dimension": metric_sources_by_dimension,
}

contract = {
    "cell": "19.0",
    "version": CELL190_VERSION,
    "role": "full_cps_comprehensive_evaluation_input_contract_no_q4_promotion",
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_evidence_is_report_only": True,
        "no_q4_coupled_artifact_authoritative": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
    },
    "upstream_contract_versions": {
        "cell18_3": cell18_3_version_190,
    },
    "primary_scientific_no_q4_artifact": scientific_no_q4_cps_path,
    "public_q6_artifact": public_q6_cps_path,
    "expected_test_rows": expected_test_rows,
    "artifact_summary": artifact_summary,
    "metric_source_summary": metric_source_summary,
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "full_cps_eval_input_contract_done_here": True,
    },
    "outputs": {
        "artifact_registry_csv": artifact_registry_csv,
        "metric_source_registry_csv": metric_source_registry_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_190(contract_json, contract)
_write_json_190(contract_canonical_json, contract)

manifest = {
    "cell": "19.0",
    "version": CELL190_VERSION,
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "artifact_registry": artifact_registry_df.to_dict("records"),
    "metric_source_registry": metric_source_registry_df.to_dict("records"),
    "artifact_summary": artifact_summary,
    "metric_source_summary": metric_source_summary,
    "strict_contract": contract["strict_contract"],
}
_write_json_190(manifest_json, manifest)

hashes = {
    "artifact_registry_csv_sha256": _sha256_file_190(artifact_registry_csv),
    "metric_source_registry_csv_sha256": _sha256_file_190(metric_source_registry_csv),
    "contract_json_sha256": _sha256_file_190(contract_json),
    "contract_canonical_json_sha256": _sha256_file_190(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_190(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes
_write_json_190(contract_json, contract)
_write_json_190(contract_canonical_json, contract)
_write_json_190(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals
# ----------------------------------------------------------
globals()["CELL190_VERSION"] = CELL190_VERSION
globals()["CELL19_0_FULL_CPS_EVAL_ARTIFACT_REGISTRY_DF"] = artifact_registry_df
globals()["CELL19_0_FULL_CPS_EVAL_METRIC_SOURCE_REGISTRY_DF"] = metric_source_registry_df
globals()["CELL19_0_FULL_CPS_EVAL_CONTRACT"] = contract

globals()["CELL19_0_FULL_CPS_EVAL_ARTIFACT_REGISTRY_CSV"] = artifact_registry_csv
globals()["CELL19_0_FULL_CPS_EVAL_METRIC_SOURCE_REGISTRY_CSV"] = metric_source_registry_csv
globals()["CELL19_0_FULL_CPS_EVAL_CONTRACT_JSON"] = contract_json
globals()["CELL19_0_FULL_CPS_EVAL_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL19_0_FULL_CPS_EVAL_MANIFEST_JSON"] = manifest_json

globals()["CELL19_0_SCIENTIFIC_NO_Q4_CPS_PATH"] = scientific_no_q4_cps_path
globals()["CELL19_0_PUBLIC_Q6_CPS_PATH"] = public_q6_cps_path
globals()["CELL19_0_SCIENTIFIC_NO_Q4_PROTOCOL_PATH"] = scientific_no_q4_protocol_path
globals()["CELL19_0_PUBLIC_Q6_PROTOCOL_PATH"] = public_q6_protocol_path
globals()["CELL19_0_PUBLIC_Q6_IOT_PATH"] = public_q6_iot_path

log(
    "[Cell19.0] Full CPS comprehensive evaluation input contract complete | "
    f"artifacts={len(artifact_registry_df)} | "
    f"required_artifacts_existing={artifact_summary['required_artifacts_existing']}/"
    f"{artifact_summary['required_artifacts_total']} | "
    f"metric_sources={len(metric_source_registry_df)} | "
    f"required_metric_sources_existing={metric_source_summary['required_metric_sources_existing']}/"
    f"{metric_source_summary['required_metric_sources_total']} | "
    "q4_status=blocked_no_promotion"
)
log(
    "[Cell19.0] Primary artifacts | "
    f"scientific_no_q4_cps={scientific_no_q4_cps_path} | "
    f"public_q6_cps={public_q6_cps_path}"
)
if len(missing_required_metric_sources):
    log(
        "[Cell19.0] WARNING: Missing required metric sources, downstream cells should degrade gracefully | "
        f"{missing_required_metric_sources[['source_id', 'path']].to_dict('records')}"
    )
if len(schema_bad_metric_sources):
    log(
        "[Cell19.0] WARNING: Metric source schema gaps detected | "
        f"{schema_bad_metric_sources[['source_id', 'missing_expected_columns']].to_dict('records')}"
    )
log(f"[Cell19.0] Saved artifact registry: {artifact_registry_csv}")
log(f"[Cell19.0] Saved metric source registry: {metric_source_registry_csv}")
log(f"[Cell19.0] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell19.0] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "full_cps_eval_input_contract_done_here=True"
)
log("--- END: Cell 19.0 - Full CPS comprehensive evaluation input contract (v1.1 no-Q4-promotion strict) ---")

gc.collect()