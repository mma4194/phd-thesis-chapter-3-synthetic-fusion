# ==========================================================
# CELL 16.0 - A0/A1/A2 decomposition input contract
# v1.1 STUDY-THESIS strict no-Q4-promotion / public-Q6-aware setup
#
# Role:
#   - Load/register decomposition inputs after final governance:
#       Q4 final status = blocked_no_promotion
#       no final Q4-coupled artifact is authoritative
#       Q6 public candidate exists and has policy-split status
#   - Register A0/A1/A2 protocol artifacts when available.
#   - Register authoritative scientific no-Q4 artifact:
#       synthetic/CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet
#       synthetic/PROTOCOL_SYN_TEST_FINAL_NO_Q4.parquet
#   - Register Q6 role-restricted public candidate:
#       synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet
#       synthetic/PROTOCOL_SYN_TEST_PUBLIC_Q6_MITIGATED.parquet
#       synthetic/IOT_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet
#   - Register Q1/Q2/Q3/Q4/Q6 metric/report sources for decomposition.
#
# Decomposition scope:
#   Q1 marginal quality
#   Q2 temporal quality
#   Q3 observability quality
#   Q4 cross-modal coupling governance/failure boundary
#   Q6 privacy/release status as contextual release-safety metadata
#
# Scientific contract:
#   - No generation.
#   - No selection.
#   - No repair.
#   - No synthetic mutation.
#   - No Q4 promotion.
#   - TEST real values are not newly used for fitting/materialization.
#   - This cell only loads artifacts/reports and validates decomposition inputs.
#
# Outputs:
#   reports/cell16_0_decomposition_input_audit.csv
#   reports/cell16_0_decomposition_artifact_registry.csv
#   reports/cell16_0_decomposition_metric_source_registry.csv
#   reports/cell16_0_decomposition_input_contract.json
#   artifacts/contracts/cell16_0_decomposition_input_contract_v1_1_THESIS.json
#   artifacts/cell16_0_decomposition_manifest.json
# ==========================================================

log("--- START: Cell 16.0 - A0/A1/A2 decomposition input contract (v1.1 no-Q4-promotion strict) ---")

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
_required_160 = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
]
_missing_160 = [k for k in _required_160 if k not in globals()]
if _missing_160:
    raise RuntimeError(f"[Cell16.0] Missing required globals: {_missing_160}")

ORIGINAL_OUTDIR_160 = str(OUTDIR)
ORIGINAL_OUT_SYN_160 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_160 = str(REPORT_DIR)

def _resolve_project_root_160(outdir, report_dir, out_syn):
    candidates = []
    for p in [outdir, report_dir, out_syn]:
        if not p:
            continue
        p = os.path.abspath(str(p))
        parts = p.split(os.sep)
        # Public re-audit contexts should resolve back to the base project.
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

    raise RuntimeError("[Cell16.0] Could not resolve canonical project root.")

PROJECT_ROOT_160 = _resolve_project_root_160(ORIGINAL_OUTDIR_160, ORIGINAL_REPORT_DIR_160, ORIGINAL_OUT_SYN_160)
OUTDIR_BASE_160 = PROJECT_ROOT_160
OUT_SYN_BASE_160 = os.path.join(PROJECT_ROOT_160, "synthetic")
REPORT_DIR_BASE_160 = os.path.join(PROJECT_ROOT_160, "reports")
ARTDIR_BASE_160 = os.path.join(PROJECT_ROOT_160, "artifacts")
CONTRACT_DIR_BASE_160 = os.path.join(ARTDIR_BASE_160, "contracts")

PUBLIC_Q6_ROOT_160 = os.path.join(PROJECT_ROOT_160, "q6_public_reaudit_v1")
PUBLIC_Q6_REPORT_DIR_160 = os.path.join(PUBLIC_Q6_ROOT_160, "reports")
PUBLIC_Q6_ARTDIR_160 = os.path.join(PUBLIC_Q6_ROOT_160, "artifacts")
PUBLIC_Q6_CONTRACT_DIR_160 = os.path.join(PUBLIC_Q6_ARTDIR_160, "contracts")

os.makedirs(OUT_SYN_BASE_160, exist_ok=True)
os.makedirs(REPORT_DIR_BASE_160, exist_ok=True)
os.makedirs(ARTDIR_BASE_160, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_160, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

CELL160_VERSION = "cell16_0_a0_a1_a2_decomposition_input_contract_v1_1_no_q4_promotion"

CFG["cell16_0_version"] = CELL160_VERSION
CFG["cell16_0_Q4_final_status"] = "blocked_no_promotion"
CFG["cell16_0_Q4_coupled_artifacts_used"] = False
CFG["cell16_0_TEST_real_values_used_for_reference_only"] = True
CFG["cell16_0_TEST_real_values_used_for_materialization"] = False
CFG["cell16_0_synthetic_values_mutated"] = False
CFG["cell16_0_selection_done_here"] = False
CFG["cell16_0_generator_fit_done_here"] = False
CFG["cell16_0_materialization_done_here"] = False
CFG["cell16_0_decomposition_input_contract_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell16_0_fail_on_missing_required_artifact", True)
CFG.setdefault("cell16_0_fail_on_required_report_missing", False)
CFG.setdefault("cell16_0_load_large_parquets", False)
CFG.setdefault("cell16_0_required_test_rows", N_TE)
CFG.setdefault("cell16_0_include_public_q6_candidate", True)
CFG.setdefault("cell16_0_required_q4_status", "blocked_no_promotion")

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
input_audit_csv = os.path.join(REPORT_DIR_BASE_160, "cell16_0_decomposition_input_audit.csv")
artifact_registry_csv = os.path.join(REPORT_DIR_BASE_160, "cell16_0_decomposition_artifact_registry.csv")
metric_source_registry_csv = os.path.join(REPORT_DIR_BASE_160, "cell16_0_decomposition_metric_source_registry.csv")
loaded_reports_csv = os.path.join(REPORT_DIR_BASE_160, "cell16_0_decomposition_loaded_reports.csv")
contract_json = os.path.join(REPORT_DIR_BASE_160, "cell16_0_decomposition_input_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_160, "cell16_0_decomposition_input_contract_v1_1_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_160, "cell16_0_decomposition_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_160(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_160(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_160(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_160(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_160(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_160(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_160(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_160(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_160(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_160(payload), f, indent=2, sort_keys=True)

def _sha256_file_160(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _exists_160(path: str) -> bool:
    return isinstance(path, str) and bool(path.strip()) and os.path.exists(path)

def _read_csv_optional_160(path: str):
    if not _exists_160(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def _read_json_optional_160(path: str):
    if not _exists_160(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _parquet_shape_160(path: str):
    if not _exists_160(path):
        return (None, None, None)
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return int(pf.metadata.num_rows), int(pf.metadata.num_columns), int(pf.metadata.num_row_groups)
    except Exception:
        try:
            d = pd.read_parquet(path)
            return int(d.shape[0]), int(d.shape[1]), None
        except Exception:
            return (None, None, None)

def _resolve_path_160(global_name: str, candidates: list):
    g = globals().get(global_name, None)
    if isinstance(g, str) and _exists_160(g):
        return g, f"global:{global_name}"
    for p in candidates:
        if _exists_160(p):
            return p, "candidate_path"
    return "", "missing"

def _register_artifact_160(
    artifact_id,
    role,
    variant,
    path,
    required=False,
    expected_rows=None,
    authoritative=True,
    governance_status="",
    notes="",
):
    exists = _exists_160(path)
    n_rows, n_cols, n_row_groups = _parquet_shape_160(path) if exists and str(path).endswith(".parquet") else (None, None, None)

    row_match = (
        bool(n_rows == int(expected_rows))
        if expected_rows is not None and n_rows is not None
        else None
    )

    return {
        "artifact_id": str(artifact_id),
        "role": str(role),
        "variant": str(variant),
        "path": str(path),
        "exists": bool(exists),
        "required": bool(required),
        "authoritative": bool(authoritative),
        "governance_status": str(governance_status),
        "rows": n_rows,
        "cols": n_cols,
        "row_groups": n_row_groups,
        "expected_rows": int(expected_rows) if expected_rows is not None else None,
        "row_match": row_match,
        "sha256": _sha256_file_160(path) if exists else "",
        "notes": str(notes),
    }

def _register_report_160(
    source_id,
    quality_dimension,
    path,
    required=False,
    authoritative=True,
    report_context="base",
    notes="",
):
    exists = _exists_160(path)
    rows = None
    cols = None
    if exists and str(path).endswith(".csv"):
        try:
            d = pd.read_csv(path, nrows=5)
            cols = int(len(d.columns))
            d_full = pd.read_csv(path, usecols=[0])
            rows = int(len(d_full))
        except Exception:
            rows = None
            cols = None

    return {
        "source_id": str(source_id),
        "quality_dimension": str(quality_dimension),
        "path": str(path),
        "exists": bool(exists),
        "required": bool(required),
        "authoritative": bool(authoritative),
        "report_context": str(report_context),
        "rows": rows,
        "cols": cols,
        "sha256": _sha256_file_160(path) if exists else "",
        "notes": str(notes),
    }

def _load_report_global_160(global_name, path):
    d = _read_csv_optional_160(path)
    if isinstance(d, pd.DataFrame):
        globals()[global_name] = d
        return True, int(len(d)), int(len(d.columns))
    return False, 0, 0

def _load_json_global_160(global_name, path):
    d = _read_json_optional_160(path)
    if isinstance(d, dict):
        globals()[global_name] = d
        return True, 1, int(len(d.keys()))
    return False, 0, 0

def _contract_version_160(path):
    d = _read_json_optional_160(path)
    if isinstance(d, dict):
        return str(d.get("version", ""))
    return ""

def _contract_field_160(path, *keys, default=None):
    d = _read_json_optional_160(path)
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# ----------------------------------------------------------
# 4) Resolve artifact paths
# ----------------------------------------------------------
A0_PROTOCOL_PATH_160, A0_PROTOCOL_SOURCE_160 = _resolve_path_160(
    "CELL11_A0_PROTOCOL_TEST_PATH",
    [
        os.path.join(OUT_SYN_BASE_160, "A0_PROTOCOL_TEST.parquet"),
        os.path.join(OUT_SYN_BASE_160, "A0.parquet"),
        os.path.join(OUT_SYN_BASE_160, "A0_BASELINE_TEST.parquet"),
    ],
)

A1_PROTOCOL_PATH_160, A1_PROTOCOL_SOURCE_160 = _resolve_path_160(
    "CELL11_A1_PROTOCOL_TEST_PATH",
    [
        os.path.join(OUT_SYN_BASE_160, "A1_PROTOCOL_TEST.parquet"),
        os.path.join(OUT_SYN_BASE_160, "A1_PROTOCOL_ONLY.parquet"),
        os.path.join(OUT_SYN_BASE_160, "A1.parquet"),
    ],
)

A2_PROTOCOL_PATH_160, A2_PROTOCOL_SOURCE_160 = _resolve_path_160(
    "CELL11_A2_PROTOCOL_TEST_PATH",
    [
        os.path.join(OUT_SYN_BASE_160, "A2_PROTOCOL_TEST.parquet"),
        os.path.join(OUT_SYN_BASE_160, "A2_HYBRID_PROTOCOL_ONLY.parquet"),
        os.path.join(OUT_SYN_BASE_160, "A2_PROTOCOL_ONLY.parquet"),
        os.path.join(OUT_SYN_BASE_160, "A2.parquet"),
    ],
)

IOT_FULL_PATH_160, IOT_FULL_SOURCE_160 = _resolve_path_160(
    "CELL12G_FULL_IOT_SYNTHETIC_TEST_PATH",
    [
        os.path.join(OUT_SYN_BASE_160, "IOT_FULL_SYNTHETIC_TEST.parquet"),
        os.path.join(OUT_SYN_BASE_160, "IOT_FULL_SYN_TEST.parquet"),
    ],
)

NO_Q4_PROTOCOL_PATH_160, NO_Q4_PROTOCOL_SOURCE_160 = _resolve_path_160(
    "CELL15_0_FINAL_PROTOCOL_NO_Q4_PATH",
    [os.path.join(OUT_SYN_BASE_160, "PROTOCOL_SYN_TEST_FINAL_NO_Q4.parquet")],
)

NO_Q4_CPS_PATH_160, NO_Q4_CPS_SOURCE_160 = _resolve_path_160(
    "CELL15_0_FINAL_CPS_NO_Q4_PATH",
    [os.path.join(OUT_SYN_BASE_160, "CPS_SYNTHETIC_TEST_FINAL_NO_Q4.parquet")],
)

# Legacy/context Q4-coupled paths may exist but are not authoritative.
LEGACY_Q4_PROTOCOL_PATH_160, LEGACY_Q4_PROTOCOL_SOURCE_160 = _resolve_path_160(
    "CELL14_10_FINAL_PROTOCOL_PATH",
    [os.path.join(OUT_SYN_BASE_160, "PROTOCOL_SYN_TEST_FINAL_Q4_COUPLED.parquet")],
)

LEGACY_Q4_CPS_PATH_160, LEGACY_Q4_CPS_SOURCE_160 = _resolve_path_160(
    "CELL14_10_FINAL_CPS_PATH",
    [os.path.join(OUT_SYN_BASE_160, "CPS_SYNTHETIC_TEST_FINAL_Q4_COUPLED.parquet")],
)

PUBLIC_Q6_CPS_PATH_160, PUBLIC_Q6_CPS_SOURCE_160 = _resolve_path_160(
    "CELL15_6_PUBLIC_CPS_PATH",
    [os.path.join(OUT_SYN_BASE_160, "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet")],
)

PUBLIC_Q6_PROTOCOL_PATH_160, PUBLIC_Q6_PROTOCOL_SOURCE_160 = _resolve_path_160(
    "CELL15_6_PUBLIC_PROTOCOL_PATH",
    [os.path.join(OUT_SYN_BASE_160, "PROTOCOL_SYN_TEST_PUBLIC_Q6_MITIGATED.parquet")],
)

PUBLIC_Q6_IOT_PATH_160, PUBLIC_Q6_IOT_SOURCE_160 = _resolve_path_160(
    "CELL15_6_PUBLIC_IOT_PATH",
    [os.path.join(OUT_SYN_BASE_160, "IOT_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet")],
)

artifact_rows = [
    _register_artifact_160(
        "A0_protocol_test",
        "protocol_baseline",
        "A0",
        A0_PROTOCOL_PATH_160,
        required=False,
        expected_rows=N_TE,
        authoritative=False,
        governance_status="optional_baseline",
        notes=f"source={A0_PROTOCOL_SOURCE_160}; optional if A0 exists only as report-level baseline",
    ),
    _register_artifact_160(
        "A1_protocol_test",
        "protocol_backbone",
        "A1",
        A1_PROTOCOL_PATH_160,
        required=False,
        expected_rows=N_TE,
        authoritative=False,
        governance_status="optional_baseline",
        notes=f"source={A1_PROTOCOL_SOURCE_160}",
    ),
    _register_artifact_160(
        "A2_protocol_test",
        "protocol_hybrid",
        "A2",
        A2_PROTOCOL_PATH_160,
        required=False,
        expected_rows=N_TE,
        authoritative=False,
        governance_status="optional_baseline",
        notes=f"source={A2_PROTOCOL_SOURCE_160}",
    ),
    _register_artifact_160(
        "iot_full_test",
        "iot_full_namespace",
        "12g",
        IOT_FULL_PATH_160,
        required=False,
        expected_rows=N_TE,
        authoritative=True,
        governance_status="scientific_component",
        notes=f"source={IOT_FULL_SOURCE_160}; optional if already included in no-Q4 CPS artifact",
    ),
    _register_artifact_160(
        "scientific_no_q4_protocol",
        "scientific_protocol_no_q4",
        "no_Q4",
        NO_Q4_PROTOCOL_PATH_160,
        required=True,
        expected_rows=N_TE,
        authoritative=True,
        governance_status="authoritative_scientific_protocol_Q4_blocked",
        notes=f"source={NO_Q4_PROTOCOL_SOURCE_160}; authoritative protocol artifact because Q4 was not promoted",
    ),
    _register_artifact_160(
        "scientific_no_q4_cps",
        "scientific_cps_no_q4",
        "no_Q4",
        NO_Q4_CPS_PATH_160,
        required=True,
        expected_rows=N_TE,
        authoritative=True,
        governance_status="authoritative_scientific_CPS_Q4_blocked",
        notes=f"source={NO_Q4_CPS_SOURCE_160}; authoritative scientific/internal artifact",
    ),
    _register_artifact_160(
        "legacy_q4_protocol_not_authoritative",
        "legacy_context_protocol",
        "Q4_legacy",
        LEGACY_Q4_PROTOCOL_PATH_160,
        required=False,
        expected_rows=N_TE,
        authoritative=False,
        governance_status="not_authoritative_Q4_blocked_no_promotion",
        notes=f"source={LEGACY_Q4_PROTOCOL_SOURCE_160}; not required and must not be treated as accepted final Q4 artifact",
    ),
    _register_artifact_160(
        "legacy_q4_cps_not_authoritative",
        "legacy_context_cps",
        "Q4_legacy",
        LEGACY_Q4_CPS_PATH_160,
        required=False,
        expected_rows=N_TE,
        authoritative=False,
        governance_status="not_authoritative_Q4_blocked_no_promotion",
        notes=f"source={LEGACY_Q4_CPS_SOURCE_160}; not required and must not be treated as accepted final Q4 artifact",
    ),
    _register_artifact_160(
        "public_q6_cps",
        "public_q6_mitigated_cps",
        "Q6_public",
        PUBLIC_Q6_CPS_PATH_160,
        required=bool(CFG.get("cell16_0_include_public_q6_candidate", True)),
        expected_rows=N_TE,
        authoritative=True,
        governance_status="public_candidate_policy_split",
        notes=f"source={PUBLIC_Q6_CPS_SOURCE_160}; role-restricted public candidate",
    ),
    _register_artifact_160(
        "public_q6_protocol",
        "public_q6_mitigated_protocol",
        "Q6_public",
        PUBLIC_Q6_PROTOCOL_PATH_160,
        required=bool(CFG.get("cell16_0_include_public_q6_candidate", True)),
        expected_rows=N_TE,
        authoritative=True,
        governance_status="public_candidate_policy_split",
        notes=f"source={PUBLIC_Q6_PROTOCOL_SOURCE_160}",
    ),
    _register_artifact_160(
        "public_q6_iot",
        "public_q6_mitigated_iot",
        "Q6_public",
        PUBLIC_Q6_IOT_PATH_160,
        required=bool(CFG.get("cell16_0_include_public_q6_candidate", True)),
        expected_rows=N_TE,
        authoritative=True,
        governance_status="public_candidate_policy_split",
        notes=f"source={PUBLIC_Q6_IOT_SOURCE_160}",
    ),
]

artifact_registry_df = pd.DataFrame(artifact_rows)

missing_required_artifacts = artifact_registry_df[
    artifact_registry_df["required"].astype(bool) & (~artifact_registry_df["exists"].astype(bool))
].copy()

bad_required_rows = artifact_registry_df[
    artifact_registry_df["required"].astype(bool)
    & artifact_registry_df["exists"].astype(bool)
    & artifact_registry_df["row_match"].eq(False)
].copy()

if len(missing_required_artifacts) and bool(CFG.get("cell16_0_fail_on_missing_required_artifact", True)):
    raise RuntimeError(
        "[Cell16.0] Missing required decomposition artifacts: "
        f"{missing_required_artifacts[['artifact_id', 'path']].to_dict('records')}"
    )

if len(bad_required_rows):
    raise RuntimeError(
        "[Cell16.0] Required decomposition artifact row-count mismatch: "
        f"{bad_required_rows[['artifact_id', 'rows', 'expected_rows']].to_dict('records')}"
    )

# ----------------------------------------------------------
# 5) Register metric/report sources
# ----------------------------------------------------------
metric_sources = [
    # Cell 11 / protocol A0-A1-A2
    _register_report_160(
        "cell11_a0_a1_a2_protocol_quality_report",
        "A0_A1_A2_protocol_registry",
        os.path.join(REPORT_DIR_BASE_160, "cell11_a0_a1_a2_protocol_quality_report.csv"),
        required=False,
        authoritative=True,
        notes="Protocol A0/A1/A2 quality report when available.",
    ),
    _register_report_160(
        "cell11_final_protocol_variant_manifest",
        "A0_A1_A2_protocol_manifest",
        os.path.join(ARTDIR_BASE_160, "cell11_final_protocol_variant_manifest.json"),
        required=False,
        authoritative=True,
        notes="Cell 11 final protocol variant manifest.",
    ),

    # Q1 / Q2 IoT continuous
    _register_report_160(
        "cell12c6_iot_value_final_test_qa",
        "Q1_Q2_iot_continuous_values",
        os.path.join(REPORT_DIR_BASE_160, "cell12c6_iot_value_final_test_qa_metrics.csv"),
        required=False,
        authoritative=True,
        notes="Final TEST QA metrics for continuous/value IoT branch if available.",
    ),
    _register_report_160(
        "cell12d6_binary_final_test_qa",
        "Q1_Q2_iot_binary_states",
        os.path.join(REPORT_DIR_BASE_160, "cell12d6_binary_final_test_qa_metrics.csv"),
        required=False,
        authoritative=True,
        notes="Final TEST QA metrics for binary branch if available.",
    ),
    _register_report_160(
        "cell12e_driver_final_test_qa",
        "Q1_Q2_iot_event_drivers",
        os.path.join(REPORT_DIR_BASE_160, "cell12e_final_driver_test_qa_metrics.csv"),
        required=False,
        authoritative=True,
        notes="Driver-branch final QA if available.",
    ),

    # Q3 observability
    _register_report_160(
        "cell12f5_observability_final_test_qa",
        "Q3_observability",
        os.path.join(REPORT_DIR_BASE_160, "cell12f5_mask_final_test_qa_metrics.csv"),
        required=False,
        authoritative=True,
        notes="Final Q3 observability/mask TEST QA if available.",
    ),

    # Full IoT namespace
    _register_report_160(
        "cell12g_iot_full_assembly",
        "full_iot_namespace",
        os.path.join(REPORT_DIR_BASE_160, "iot_full_assembly_audit.csv"),
        required=False,
        authoritative=True,
        notes="Full IoT namespace assembly audit if available.",
    ),
    _register_report_160(
        "cell12g_iot_full_registry",
        "full_iot_namespace",
        os.path.join(REPORT_DIR_BASE_160, "iot_full_namespace_column_registry.csv"),
        required=False,
        authoritative=True,
        notes="Full IoT namespace registry if available.",
    ),

    # Q4 governance / failure boundary
    _register_report_160(
        "cell14_7_q4_no_promotion_governance",
        "Q4_no_promotion_governance",
        os.path.join(REPORT_DIR_BASE_160, "cell14_7_q4_no_promotion_governance.csv"),
        required=True,
        authoritative=True,
        notes="Q4 no-promotion governance ledger.",
    ),
    _register_report_160(
        "cell14_10_final_q4_decision_ledger",
        "Q4_final_decision_no_promotion",
        os.path.join(REPORT_DIR_BASE_160, "cell14_10_final_q4_decision_ledger.csv"),
        required=True,
        authoritative=True,
        notes="Final Q4 decision ledger; should report blocked_no_promotion.",
    ),
    _register_report_160(
        "cell14_10_final_q4_candidate_comparison",
        "Q4_final_decision_no_promotion",
        os.path.join(REPORT_DIR_BASE_160, "cell14_10_final_q4_candidate_comparison.csv"),
        required=True,
        authoritative=True,
        notes="Q4 candidate comparison; no final publication candidate.",
    ),
    _register_report_160(
        "cell14_11_final_q4_evidence_table",
        "Q4_final_evidence_no_promotion",
        os.path.join(REPORT_DIR_BASE_160, "cell14_11_final_q4_evidence_table.csv"),
        required=True,
        authoritative=True,
        notes="Paper-ready Q4 no-promotion evidence table.",
    ),
    _register_report_160(
        "cell14_12_q4_no_promotion_freshness_audit",
        "Q4_final_freshness_no_promotion",
        os.path.join(REPORT_DIR_BASE_160, "cell14_12_q4_no_promotion_freshness_audit.csv"),
        required=True,
        authoritative=True,
        notes="Final Q4 no-promotion freshness audit.",
    ),

    # Q6 full no-Q4 scientific artifact context
    _register_report_160(
        "cell15_4_full_q6_role_release_safety",
        "Q6_full_artifact_privacy_release_safety",
        os.path.join(REPORT_DIR_BASE_160, "cell15_4_q6_role_release_safety.csv"),
        required=True,
        authoritative=True,
        report_context="base_full_artifact",
        notes="Full no-Q4 artifact Q6 role-level release-safety summary.",
    ),
    _register_report_160(
        "cell15_5_q6_mitigation_plan",
        "Q6_mitigation_plan",
        os.path.join(REPORT_DIR_BASE_160, "cell15_5_q6_mitigation_plan.csv"),
        required=True,
        authoritative=True,
        report_context="base_full_artifact",
        notes="Q6 mitigation plan for full artifact.",
    ),
    _register_report_160(
        "cell15_6_public_mitigation_audit",
        "Q6_public_candidate_builder",
        os.path.join(REPORT_DIR_BASE_160, "cell15_6_q6_public_mitigation_audit.csv"),
        required=True,
        authoritative=True,
        report_context="base_full_artifact",
        notes="Public role-restricted mitigation audit.",
    ),

    # Q6 public candidate re-audit and policy split
    _register_report_160(
        "public_cell15_4_q6_role_release_safety",
        "Q6_public_candidate_strict_combined_status",
        os.path.join(PUBLIC_Q6_REPORT_DIR_160, "cell15_4_q6_role_release_safety.csv"),
        required=True,
        authoritative=True,
        report_context="public_candidate_reaudit",
        notes="Public candidate strict combined Q6 release-safety summary.",
    ),
    _register_report_160(
        "public_cell15_4b_policy_split_role_summary",
        "Q6_public_candidate_policy_split",
        os.path.join(PUBLIC_Q6_REPORT_DIR_160, "cell15_4b_q6_policy_split_role_summary.csv"),
        required=True,
        authoritative=True,
        report_context="public_candidate_reaudit",
        notes="Public candidate policy-split direct privacy vs distinguishability summary.",
    ),
    _register_report_160(
        "public_cell15_4b_policy_split_summary",
        "Q6_public_candidate_policy_split",
        os.path.join(PUBLIC_Q6_REPORT_DIR_160, "cell15_4b_q6_policy_split_summary.csv"),
        required=True,
        authoritative=True,
        report_context="public_candidate_reaudit",
        notes="Public candidate policy-split summary table.",
    ),
]

metric_source_registry_df = pd.DataFrame(metric_sources)

missing_required_reports = metric_source_registry_df[
    metric_source_registry_df["required"].astype(bool) & (~metric_source_registry_df["exists"].astype(bool))
].copy()

if len(missing_required_reports) and bool(CFG.get("cell16_0_fail_on_required_report_missing", False)):
    raise RuntimeError(
        "[Cell16.0] Missing required decomposition reports: "
        f"{missing_required_reports[['source_id', 'path']].to_dict('records')}"
    )

# ----------------------------------------------------------
# 6) Validate key governance status from available contracts/reports
# ----------------------------------------------------------
q4_decision_contract_path = os.path.join(CONTRACT_DIR_BASE_160, "cell14_10_final_q4_decision_contract_v1_1_THESIS.json")
q4_scope_contract_path = os.path.join(CONTRACT_DIR_BASE_160, "cell14_11_final_q4_scope_audit_contract_v1_1_THESIS.json")
q4_freshness_contract_path = os.path.join(CONTRACT_DIR_BASE_160, "cell14_12_q4_no_promotion_freshness_audit_v2_1_THESIS.json")
q6_public_split_contract_path = os.path.join(PUBLIC_Q6_CONTRACT_DIR_160, "cell15_4b_q6_policy_split_contract_v1_0_THESIS.json")

q4_status_from_1410 = _contract_field_160(q4_decision_contract_path, "final_decision", "final_q4_status", default="")
q4_status_from_1411 = _contract_field_160(q4_scope_contract_path, "final_q4_status", default="")
q4_status_from_1412 = _contract_field_160(q4_freshness_contract_path, "final_q4_status", default="")
public_direct_status = _contract_field_160(q6_public_split_contract_path, "policy_split", "direct_privacy_overall_status", default="")
public_dist_status = _contract_field_160(q6_public_split_contract_path, "policy_split", "distinguishability_overall_status", default="")
public_strict_status = _contract_field_160(q6_public_split_contract_path, "policy_split", "strict_combined_overall_status", default="")

q4_status_values = [s for s in [q4_status_from_1410, q4_status_from_1411, q4_status_from_1412] if s]
q4_status_consistent = bool(q4_status_values and all(s == "blocked_no_promotion" for s in q4_status_values))

if q4_status_values and not q4_status_consistent:
    raise RuntimeError(f"[Cell16.0] Inconsistent Q4 governance status: {q4_status_values}")

# ----------------------------------------------------------
# 7) Load lightweight key tables into globals when available
# ----------------------------------------------------------
loaded_report_rows = []

csv_load_map = [
    ("CELL16_0_CELL14_10_Q4_DECISION_LEDGER_DF", "cell14_10_final_q4_decision_ledger"),
    ("CELL16_0_CELL14_10_Q4_CANDIDATE_COMPARISON_DF", "cell14_10_final_q4_candidate_comparison"),
    ("CELL16_0_CELL14_11_Q4_EVIDENCE_DF", "cell14_11_final_q4_evidence_table"),
    ("CELL16_0_CELL15_4_FULL_Q6_RELEASE_SAFETY_DF", "cell15_4_full_q6_role_release_safety"),
    ("CELL16_0_CELL15_5_Q6_MITIGATION_PLAN_DF", "cell15_5_q6_mitigation_plan"),
    ("CELL16_0_CELL15_6_PUBLIC_MITIGATION_AUDIT_DF", "cell15_6_public_mitigation_audit"),
    ("CELL16_0_PUBLIC_CELL15_4_Q6_RELEASE_SAFETY_DF", "public_cell15_4_q6_role_release_safety"),
    ("CELL16_0_PUBLIC_CELL15_4B_POLICY_SPLIT_DF", "public_cell15_4b_policy_split_role_summary"),
    ("CELL16_0_PUBLIC_CELL15_4B_POLICY_SPLIT_SUMMARY_DF", "public_cell15_4b_policy_split_summary"),
]

for global_name, source_id in csv_load_map:
    src = metric_source_registry_df[metric_source_registry_df["source_id"].eq(source_id)]
    if len(src):
        path = str(src.iloc[0]["path"])
        ok, rows, cols = _load_report_global_160(global_name, path)
        loaded_report_rows.append({
            "global_name": global_name,
            "source_id": source_id,
            "loaded": bool(ok),
            "rows": int(rows),
            "cols": int(cols),
            "path": path,
        })

json_load_map = [
    ("CELL16_0_CELL14_10_Q4_DECISION_CONTRACT", q4_decision_contract_path),
    ("CELL16_0_CELL14_11_Q4_SCOPE_CONTRACT", q4_scope_contract_path),
    ("CELL16_0_CELL14_12_Q4_FRESHNESS_CONTRACT", q4_freshness_contract_path),
    ("CELL16_0_PUBLIC_CELL15_4B_POLICY_SPLIT_CONTRACT", q6_public_split_contract_path),
]

for global_name, path in json_load_map:
    ok, rows, cols = _load_json_global_160(global_name, path)
    loaded_report_rows.append({
        "global_name": global_name,
        "source_id": os.path.basename(path),
        "loaded": bool(ok),
        "rows": int(rows),
        "cols": int(cols),
        "path": path,
    })

loaded_report_df = pd.DataFrame(loaded_report_rows)

# ----------------------------------------------------------
# 8) Optional large parquet loading
# ----------------------------------------------------------
if bool(CFG.get("cell16_0_load_large_parquets", False)):
    if _exists_160(NO_Q4_CPS_PATH_160):
        globals()["CELL16_0_SCIENTIFIC_NO_Q4_CPS_DF"] = pd.read_parquet(NO_Q4_CPS_PATH_160)
    if _exists_160(NO_Q4_PROTOCOL_PATH_160):
        globals()["CELL16_0_SCIENTIFIC_NO_Q4_PROTOCOL_DF"] = pd.read_parquet(NO_Q4_PROTOCOL_PATH_160)
    if _exists_160(PUBLIC_Q6_CPS_PATH_160):
        globals()["CELL16_0_PUBLIC_Q6_CPS_DF"] = pd.read_parquet(PUBLIC_Q6_CPS_PATH_160)

# ----------------------------------------------------------
# 9) Input audit
# ----------------------------------------------------------
input_audit_rows = [
    {"metric": "cell_version", "value": CELL160_VERSION},
    {"metric": "project_root", "value": PROJECT_ROOT_160},
    {"metric": "original_OUTDIR", "value": ORIGINAL_OUTDIR_160},
    {"metric": "base_REPORT_DIR", "value": REPORT_DIR_BASE_160},
    {"metric": "public_Q6_REPORT_DIR", "value": PUBLIC_Q6_REPORT_DIR_160},
    {"metric": "train_rows", "value": int(N_TR)},
    {"metric": "val_rows", "value": int(N_VAL)},
    {"metric": "test_rows", "value": int(N_TE)},
    {"metric": "q4_status_from_1410", "value": q4_status_from_1410},
    {"metric": "q4_status_from_1411", "value": q4_status_from_1411},
    {"metric": "q4_status_from_1412", "value": q4_status_from_1412},
    {"metric": "q4_status_consistent_blocked_no_promotion", "value": bool(q4_status_consistent)},
    {"metric": "public_direct_privacy_status", "value": public_direct_status},
    {"metric": "public_distinguishability_status", "value": public_dist_status},
    {"metric": "public_strict_combined_status", "value": public_strict_status},
    {"metric": "artifact_rows", "value": int(len(artifact_registry_df))},
    {"metric": "artifacts_existing", "value": int(artifact_registry_df["exists"].astype(bool).sum())},
    {"metric": "artifacts_missing", "value": int((~artifact_registry_df["exists"].astype(bool)).sum())},
    {"metric": "required_artifacts_missing", "value": int(len(missing_required_artifacts))},
    {"metric": "required_artifact_row_mismatches", "value": int(len(bad_required_rows))},
    {"metric": "metric_sources_total", "value": int(len(metric_source_registry_df))},
    {"metric": "metric_sources_existing", "value": int(metric_source_registry_df["exists"].astype(bool).sum())},
    {"metric": "metric_sources_missing", "value": int((~metric_source_registry_df["exists"].astype(bool)).sum())},
    {"metric": "required_metric_sources_missing", "value": int(len(missing_required_reports))},
    {"metric": "loaded_lightweight_reports", "value": int(loaded_report_df["loaded"].astype(bool).sum()) if len(loaded_report_df) else 0},
    {"metric": "large_parquets_loaded", "value": bool(CFG.get("cell16_0_load_large_parquets", False))},
    {"metric": "TEST_real_values_used_for_reference_only", "value": True},
    {"metric": "TEST_real_values_used_for_materialization", "value": False},
    {"metric": "synthetic_values_mutated", "value": False},
    {"metric": "selection_done_here", "value": False},
    {"metric": "generator_fit_done_here", "value": False},
    {"metric": "materialization_done_here", "value": False},
]

input_audit_df = pd.DataFrame(input_audit_rows)

# ----------------------------------------------------------
# 10) Save registries
# ----------------------------------------------------------
artifact_registry_df.to_csv(artifact_registry_csv, index=False)
metric_source_registry_df.to_csv(metric_source_registry_csv, index=False)
loaded_report_df.to_csv(loaded_reports_csv, index=False)
input_audit_df.to_csv(input_audit_csv, index=False)

# ----------------------------------------------------------
# 11) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "16.0",
    "version": CELL160_VERSION,
    "role": "a0_a1_a2_decomposition_input_contract_no_q4_promotion",
    "decomposition_scope": {
        "Q1": "marginal_distribution_decomposition",
        "Q2": "temporal_dynamics_decomposition",
        "Q3": "observability_mask_decomposition",
        "Q4": "cross_modal_coupling_governance_and_failure_boundary",
        "Q6": "privacy_release_safety_context_and_public_candidate_policy_split",
    },
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "q4_status_values": q4_status_values,
        "q4_status_consistent": bool(q4_status_consistent),
        "legacy_q4_artifacts_registered_as_non_authoritative": True,
    },
    "q6_public_candidate_context": {
        "public_direct_privacy_status": public_direct_status,
        "public_distinguishability_status": public_dist_status,
        "public_strict_combined_status": public_strict_status,
        "public_candidate_path": PUBLIC_Q6_CPS_PATH_160,
        "policy_split_contract": q6_public_split_contract_path,
    },
    "rows": {
        "train": int(N_TR),
        "val": int(N_VAL),
        "test": int(N_TE),
    },
    "artifact_registry": artifact_registry_df.to_dict("records"),
    "metric_source_registry": metric_source_registry_df.to_dict("records"),
    "loaded_lightweight_reports": loaded_report_df.to_dict("records"),
    "missing_required_artifacts": missing_required_artifacts.to_dict("records"),
    "missing_required_reports": missing_required_reports.to_dict("records"),
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used_for_reference_only": True,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "decomposition_input_contract_done_here": True,
    },
    "outputs": {
        "input_audit_csv": input_audit_csv,
        "artifact_registry_csv": artifact_registry_csv,
        "metric_source_registry_csv": metric_source_registry_csv,
        "loaded_reports_csv": loaded_reports_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_160(contract_json, contract)
_write_json_160(contract_canonical_json, contract)

manifest = {
    "cell": "16.0",
    "version": CELL160_VERSION,
    "created_outputs": contract["outputs"],
    "decomposition_scope": contract["decomposition_scope"],
    "q4_governance": contract["q4_governance"],
    "q6_public_candidate_context": contract["q6_public_candidate_context"],
    "rows": contract["rows"],
    "artifact_ids": artifact_registry_df["artifact_id"].astype(str).tolist(),
    "metric_source_ids": metric_source_registry_df["source_id"].astype(str).tolist(),
    "strict_contract": contract["strict_contract"],
}
_write_json_160(manifest_json, manifest)

hashes = {
    "input_audit_csv_sha256": _sha256_file_160(input_audit_csv),
    "artifact_registry_csv_sha256": _sha256_file_160(artifact_registry_csv),
    "metric_source_registry_csv_sha256": _sha256_file_160(metric_source_registry_csv),
    "loaded_reports_csv_sha256": _sha256_file_160(loaded_reports_csv),
    "contract_json_sha256": _sha256_file_160(contract_json),
    "contract_canonical_json_sha256": _sha256_file_160(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_160(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_160(contract_json, contract)
_write_json_160(contract_canonical_json, contract)
_write_json_160(manifest_json, manifest)

# ----------------------------------------------------------
# 12) Export globals
# ----------------------------------------------------------
globals()["CELL160_VERSION"] = CELL160_VERSION
globals()["CELL16_0_ARTIFACT_REGISTRY_DF"] = artifact_registry_df
globals()["CELL16_0_METRIC_SOURCE_REGISTRY_DF"] = metric_source_registry_df
globals()["CELL16_0_INPUT_AUDIT_DF"] = input_audit_df
globals()["CELL16_0_LOADED_REPORTS_DF"] = loaded_report_df
globals()["CELL16_0_DECOMPOSITION_INPUT_CONTRACT"] = contract

globals()["CELL16_0_A0_PROTOCOL_PATH"] = A0_PROTOCOL_PATH_160
globals()["CELL16_0_A1_PROTOCOL_PATH"] = A1_PROTOCOL_PATH_160
globals()["CELL16_0_A2_PROTOCOL_PATH"] = A2_PROTOCOL_PATH_160
globals()["CELL16_0_IOT_FULL_PATH"] = IOT_FULL_PATH_160
globals()["CELL16_0_SCIENTIFIC_NO_Q4_PROTOCOL_PATH"] = NO_Q4_PROTOCOL_PATH_160
globals()["CELL16_0_SCIENTIFIC_NO_Q4_CPS_PATH"] = NO_Q4_CPS_PATH_160
globals()["CELL16_0_LEGACY_Q4_PROTOCOL_PATH"] = LEGACY_Q4_PROTOCOL_PATH_160
globals()["CELL16_0_LEGACY_Q4_CPS_PATH"] = LEGACY_Q4_CPS_PATH_160
globals()["CELL16_0_PUBLIC_Q6_CPS_PATH"] = PUBLIC_Q6_CPS_PATH_160
globals()["CELL16_0_PUBLIC_Q6_PROTOCOL_PATH"] = PUBLIC_Q6_PROTOCOL_PATH_160
globals()["CELL16_0_PUBLIC_Q6_IOT_PATH"] = PUBLIC_Q6_IOT_PATH_160

globals()["CELL16_0_INPUT_AUDIT_CSV"] = input_audit_csv
globals()["CELL16_0_ARTIFACT_REGISTRY_CSV"] = artifact_registry_csv
globals()["CELL16_0_METRIC_SOURCE_REGISTRY_CSV"] = metric_source_registry_csv
globals()["CELL16_0_LOADED_REPORTS_CSV"] = loaded_reports_csv
globals()["CELL16_0_CONTRACT_JSON"] = contract_json
globals()["CELL16_0_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL16_0_MANIFEST_JSON"] = manifest_json

log(
    "[Cell16.0] Decomposition input contract complete | "
    f"q4_status=blocked_no_promotion | "
    f"public_direct_privacy={public_direct_status} | "
    f"public_distinguishability={public_dist_status} | "
    f"public_strict_combined={public_strict_status} | "
    f"artifacts={len(artifact_registry_df)} | "
    f"existing_artifacts={int(artifact_registry_df['exists'].astype(bool).sum())} | "
    f"metric_sources={len(metric_source_registry_df)} | "
    f"existing_metric_sources={int(metric_source_registry_df['exists'].astype(bool).sum())} | "
    f"missing_required_artifacts={len(missing_required_artifacts)} | "
    f"missing_required_reports={len(missing_required_reports)}"
)
log(
    "[Cell16.0] Artifact registry summary | "
    f"{artifact_registry_df[['artifact_id', 'exists', 'required', 'authoritative', 'rows', 'cols']].to_dict('records')}"
)
log(f"[Cell16.0] Saved artifact registry: {artifact_registry_csv}")
log(f"[Cell16.0] Saved metric source registry: {metric_source_registry_csv}")
log(f"[Cell16.0] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell16.0] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used_for_reference_only=True | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "decomposition_input_contract_done_here=True"
)
log("--- END: Cell 16.0 - A0/A1/A2 decomposition input contract (v1.1 no-Q4-promotion strict) ---")

gc.collect()