# ==========================================================
# CELL 18.0 - Claim registry
# v1.3 STUDY-THESIS strict ledger-derived claim-governance registry, claim-scoped-status fix
#
# Purpose:
#   Generate claim registry from MASTER_RESULTS_LEDGER_FOR_PAPER.csv.
#   This cell is output-only / registry-only; it does not create optimistic,
#   hand-authored publication claims.
#
# Checklist action implemented:
#   - Generate from MASTER_RESULTS_LEDGER_FOR_PAPER.csv.
#   - Add fields:
#       selection_split
#       evaluation_split
#       TEST_used_for_selection
#       TEST_used_for_repair
#       permitted_claim
#       forbidden_claim
#   - Do not use global "publication_ready".
#   - Use claim-scoped status only.
#   - Remove/block stale Q4-promoted / Q4-final-coupled claims.
#
# Final governance inherited from Cell 17.6:
#   - Q4_final_status = blocked_no_promotion.
#   - No Q4-coupled artifact is authoritative.
#   - Public direct privacy/no-copy status = release_warning.
#   - Public strict combined status = release_blocker.
#
# Outputs:
#   reports/cell18_0_claim_registry.csv
#   reports/cell18_0_claim_registry_audit.csv
#   reports/cell18_0_claim_registry_contract.json
#   artifacts/contracts/cell18_0_claim_registry_contract_v1_1_THESIS.json
#   artifacts/cell18_0_claim_registry_manifest.json
#
# Later expected final outputs from Cell 18.3:
#   claim_traceability_manifest.csv
#   claim_traceability_manifest.json
#   publication_readiness_dashboard.csv
#   publication_blockers.csv
# ==========================================================

log("--- START: Cell 18.0 - Claim registry (v1.3 claim-scoped-status ledger-derived no-Q4-promotion strict) ---")

import os
import gc
import json
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_180 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT",
]
_missing_180 = [k for k in _required_180 if k not in globals()]
if _missing_180:
    raise RuntimeError(f"[Cell18.0] Missing required globals: {_missing_180}")

ORIGINAL_OUTDIR_180 = str(OUTDIR)
ORIGINAL_OUT_SYN_180 = str(OUT_SYN)
ORIGINAL_REPORT_DIR_180 = str(REPORT_DIR)

def _resolve_project_root_180(outdir, report_dir, out_syn):
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
        if os.path.isdir(os.path.join(c, "reports")) and os.path.isdir(os.path.join(c, "artifacts")):
            return c

    raise RuntimeError("[Cell18.0] Could not resolve canonical project root.")

PROJECT_ROOT_180 = _resolve_project_root_180(ORIGINAL_OUTDIR_180, ORIGINAL_REPORT_DIR_180, ORIGINAL_OUT_SYN_180)
REPORT_DIR_BASE_180 = os.path.join(PROJECT_ROOT_180, "reports")
ARTDIR_BASE_180 = os.path.join(PROJECT_ROOT_180, "artifacts")
CONTRACT_DIR_BASE_180 = os.path.join(ARTDIR_BASE_180, "contracts")

os.makedirs(REPORT_DIR_BASE_180, exist_ok=True)
os.makedirs(ARTDIR_BASE_180, exist_ok=True)
os.makedirs(CONTRACT_DIR_BASE_180, exist_ok=True)

SEED = int(SEED)

CELL180_VERSION = "cell18_0_claim_registry_v1_3_claim_scoped_status_ledger_derived_no_q4_promotion"

CFG["cell18_0_version"] = CELL180_VERSION
CFG["cell18_0_Q4_final_status"] = "blocked_no_promotion"
CFG["cell18_0_Q4_coupled_artifacts_used"] = False
CFG["cell18_0_TEST_real_values_used"] = False
CFG["cell18_0_TEST_real_values_used_for_materialization"] = False
CFG["cell18_0_synthetic_values_mutated"] = False
CFG["cell18_0_selection_done_here"] = False
CFG["cell18_0_generator_fit_done_here"] = False
CFG["cell18_0_materialization_done_here"] = False
CFG["cell18_0_claim_registry_done_here"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell18_0_require_master_ledger", True)
CFG.setdefault("cell18_0_master_ledger_filename", "MASTER_RESULTS_LEDGER_FOR_PAPER.csv")
CFG.setdefault("cell18_0_require_unique_claim_ids", True)
CFG.setdefault("cell18_0_default_seed_policy", "single_seed_current_run;multi_seed_CI_not_claimed")
CFG.setdefault("cell18_0_default_selection_split", "TRAIN_fit_VAL_select_only;TEST_not_used_for_selection")
CFG.setdefault("cell18_0_default_evaluation_split", "TEST_QA_reference_only")
CFG.setdefault("cell18_0_default_claim_status", "pending_traceability")
CFG.setdefault("cell18_0_strict_no_global_publication_ready", True)
CFG.setdefault("cell18_0_strict_no_q4_promotion_claims", True)

DEFAULT_SEED_POLICY_180 = str(CFG.get("cell18_0_default_seed_policy", "single_seed_current_run;multi_seed_CI_not_claimed"))
DEFAULT_SELECTION_SPLIT_180 = str(CFG.get("cell18_0_default_selection_split", "TRAIN_fit_VAL_select_only;TEST_not_used_for_selection"))
DEFAULT_EVALUATION_SPLIT_180 = str(CFG.get("cell18_0_default_evaluation_split", "TEST_QA_reference_only"))
DEFAULT_CLAIM_STATUS_180 = str(CFG.get("cell18_0_default_claim_status", "pending_traceability"))

# ----------------------------------------------------------
# 2) Paths
# ----------------------------------------------------------
def _resolve_master_ledger_180():
    filename = str(CFG.get("cell18_0_master_ledger_filename", "MASTER_RESULTS_LEDGER_FOR_PAPER.csv"))
    candidates = [
        os.path.join(REPORT_DIR_BASE_180, filename),
        os.path.join(ARTDIR_BASE_180, filename),
        os.path.join(PROJECT_ROOT_180, filename),
        os.path.join(REPORT_DIR_BASE_180, filename.lower()),
        os.path.join(REPORT_DIR_BASE_180, filename.upper()),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p, candidates
    if bool(CFG.get("cell18_0_require_master_ledger", True)):
        raise FileNotFoundError(
            "[Cell18.0] Required MASTER_RESULTS_LEDGER_FOR_PAPER.csv not found. "
            f"Searched={candidates}. Generate or copy the master results ledger before running Cell 18.0."
        )
    return "", candidates

master_ledger_csv, master_ledger_candidates = _resolve_master_ledger_180()

claim_registry_csv = os.path.join(REPORT_DIR_BASE_180, "cell18_0_claim_registry.csv")
claim_registry_audit_csv = os.path.join(REPORT_DIR_BASE_180, "cell18_0_claim_registry_audit.csv")
contract_json = os.path.join(REPORT_DIR_BASE_180, "cell18_0_claim_registry_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR_BASE_180, "cell18_0_claim_registry_contract_v1_3_THESIS.json")
manifest_json = os.path.join(ARTDIR_BASE_180, "cell18_0_claim_registry_manifest.json")

final_claim_trace_csv = os.path.join(REPORT_DIR_BASE_180, "claim_traceability_manifest.csv")
final_claim_trace_json = os.path.join(REPORT_DIR_BASE_180, "claim_traceability_manifest.json")
final_readiness_csv = os.path.join(REPORT_DIR_BASE_180, "publication_readiness_dashboard.csv")
final_blockers_csv = os.path.join(REPORT_DIR_BASE_180, "publication_blockers.csv")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_180(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_180(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_180(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_180(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_180(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_180(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_180(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_180(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_180(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_180(payload), f, indent=2, sort_keys=True)

def _sha256_file_180(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_str_180(x, default=""):
    if pd.isna(x):
        return default
    return str(x)

def _safe_float_180(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default

def _parse_bool_180(x, default=False):
    if isinstance(x, bool):
        return bool(x)
    if pd.isna(x):
        return bool(default)
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "used"}:
        return True
    if s in {"false", "0", "no", "n", "not_used", "none", ""}:
        return False
    return bool(default)

def _first_present_180(row, candidates, default=""):
    for c in candidates:
        if c in row.index:
            val = row.get(c)
            if not pd.isna(val) and str(val).strip() != "":
                return val
    return default

def _infer_dimension_180(row):
    text = " ".join([str(v) for v in row.values if not pd.isna(v)]).lower()
    direct = _first_present_180(row, ["quality_dimension", "dimension", "q_dimension", "metric_family", "claim_dimension"], "")
    if str(direct).strip():
        return str(direct).strip()

    if "q1" in text or "marginal" in text or "ks" in text or "wasserstein" in text:
        return "Q1_marginal"
    if "q2" in text or "temporal" in text or "lag" in text or "autocorr" in text or "run_length" in text:
        return "Q2_temporal"
    if "q3" in text or "observability" in text or "stale" in text or "mask" in text:
        return "Q3_observability"
    if "q4" in text or "coupling" in text or "eta" in text or "manifest" in text:
        return "Q4_coupling"
    if "q6" in text or "privacy" in text or "release" in text or "copy" in text or "dcr" in text or "nndr" in text:
        return "Q6_privacy_release"
    if "baseline" in text or "tabddpm" in text or "timegan" in text or "ctgan" in text:
        return "external_baselines"
    return "artifact_or_method"

def _infer_source_cell_180(row):
    val = _first_present_180(row, ["source_cell", "cell", "cell_id", "source", "producer_cell", "notebook_cell"], "")
    return str(val).strip()

def _infer_metric_name_180(row):
    val = _first_present_180(row, ["metric_name", "metric", "result_metric", "measure", "expected_metric_name"], "")
    if str(val).strip():
        return str(val).strip()
    dim = _infer_dimension_180(row)
    return {
        "Q1_marginal": "q1_metric_from_master_ledger",
        "Q2_temporal": "q2_metric_from_master_ledger",
        "Q3_observability": "q3_metric_from_master_ledger",
        "Q4_coupling": "q4_metric_from_master_ledger",
        "Q6_privacy_release": "q6_metric_from_master_ledger",
        "external_baselines": "baseline_metric_from_master_ledger",
    }.get(dim, "metric_from_master_ledger")

def _infer_metric_value_180(row):
    val = _first_present_180(row, ["metric_value", "value", "result_value", "score", "penalty", "count"], np.nan)
    return _safe_float_180(val, np.nan)

def _infer_artifact_type_180(row):
    val = _first_present_180(row, ["artifact_type", "artifact", "expected_artifact_type", "artifact_role", "scope", "claim_scope"], "")
    if str(val).strip():
        return str(val).strip()
    return _infer_dimension_180(row)

def _infer_artifact_path_180(row):
    val = _first_present_180(row, ["artifact_path", "path", "output_path", "file", "file_path"], "")
    return str(val).strip()

def _infer_claim_text_180(row):
    val = _first_present_180(row, ["claim_text", "paper_claim", "claim", "permitted_claim", "interpretation", "finding", "headline_claim_allowed"], "")
    if str(val).strip():
        return str(val).strip()

    dim = _infer_dimension_180(row)
    metric = _infer_metric_name_180(row)
    value = _infer_metric_value_180(row)
    src = _infer_source_cell_180(row)
    if np.isfinite(value):
        return f"{dim} result `{metric}` from {src or 'master ledger'} has value {value:.6g}."
    return f"{dim} result `{metric}` is recorded in the master results ledger."

def _infer_selection_split_180(row):
    val = _first_present_180(row, ["selection_split", "selection_split_used", "select_split"], "")
    if str(val).strip():
        return str(val).strip()

    text = " ".join([str(v) for v in row.values if not pd.isna(v)]).lower()
    if "test_used_for_selection" in text and "true" in text:
        return "INVALID_TEST_used_for_selection"
    if "val" in text or "validation" in text:
        return "VAL_selection_or_thresholding"
    if "train" in text:
        return "TRAIN_fit_or_estimation"
    return DEFAULT_SELECTION_SPLIT_180

def _infer_evaluation_split_180(row):
    val = _first_present_180(row, ["evaluation_split", "eval_split", "qa_split", "split_used"], "")
    if str(val).strip():
        return str(val).strip()

    text = " ".join([str(v) for v in row.values if not pd.isna(v)]).lower()
    if "test" in text:
        return "TEST_QA_reference_only"
    if "val" in text:
        return "VAL_metric_or_selection_evidence"
    if "train" in text:
        return "TRAIN_fit_evidence_only"
    return DEFAULT_EVALUATION_SPLIT_180

def _infer_test_used_for_selection_180(row):
    val = _first_present_180(row, ["TEST_used_for_selection", "test_used_for_selection", "used_TEST_for_selection"], None)
    if val is not None:
        return _parse_bool_180(val, False)
    text = " ".join([str(v) for v in row.values if not pd.isna(v)]).lower()
    return bool("test_used_for_selection=true" in text or "test used for selection" in text)

def _infer_test_used_for_repair_180(row):
    val = _first_present_180(row, ["TEST_used_for_repair", "test_used_for_repair", "used_TEST_for_repair"], None)
    if val is not None:
        return _parse_bool_180(val, False)
    text = " ".join([str(v) for v in row.values if not pd.isna(v)]).lower()
    return bool("test_used_for_repair=true" in text or "test used for repair" in text)

def _q4_stale_promoted_180(text):
    """
    Positive stale-Q4 claim detector.
    Safe negations are allowed: no_Q4_coupled_public_artifact,
    no Q4-coupled artifact is authoritative, Q4 remains blocked_no_promotion.
    """
    s = str(text).lower()
    s_norm = re.sub(r"[_\-]+", " ", s)

    safe_negation_patterns = [
        "no q4 coupled",
        "no q4 final",
        "no q4 artifact",
        "no q4 coupled public artifact",
        "not q4 coupled",
        "not promoted",
        "no promotion",
        "blocked no promotion",
        "blocked/no promotion",
        "q4 remains blocked",
        "q4 final status = blocked",
        "q4 final status=blocked",
        "no q4 coupled artifact is authoritative",
        "q4 coupled artifacts used=false",
        "q4 coupled artifact is not authoritative",
    ]
    if any(p in s_norm for p in safe_negation_patterns):
        return False

    stale_patterns = [
        r"\bfinal\s+q4\s+coupled\b",
        r"\bq4\s+final\s+coupled\b",
        r"\baccepted\s+a0\b",
        r"\ba0\s+accepted\b",
        r"\ba0\s+promoted\b",
        r"\bq4\s+promoted\b",
        r"\baccepted\s+q4\b",
        r"\bzero\s+q4\s+blockers\b",
        r"\bq4\s+publication\s+ready\b",
        r"\bpublication\s+ready\s+q4\b",
        r"\bfinal\s+q4\s+artifact\b",
        r"\bq4\s+artifact\s+authoritative\b",
        r"\bq4\s+coupled\s+artifact\s+authoritative\b",
    ]
    return any(re.search(p, s_norm) for p in stale_patterns)

def _public_release_overclaim_180(text):
    """
    Positive public-release overclaim detector.

    Allows safe blocker/caveat wording such as release_blocker,
    release_warning, strict-release blocked, not release ready, and
    no formal privacy guarantee.
    """
    s = str(text).lower()
    s_norm = re.sub(r"[_\-]+", " ", s)

    safe_caveat_patterns = [
        "release blocker",
        "release warning",
        "strict release blocked",
        "strict release remains blocked",
        "remains strict release blocked",
        "not release ready",
        "not public release ready",
        "not suitable for unrestricted public",
        "no unrestricted public",
        "no formal privacy guarantee",
        "formal privacy guarantee is not claimed",
        "no global privacy pass",
        "direct no copy",
        "warning level status",
        "blocked by distinguishability",
        "distinguishability blocked",
        "public strict combined status release blocker",
        "public direct privacy status release warning",
    ]
    if any(p in s_norm for p in safe_caveat_patterns):
        return False

    positive_patterns = [
        r"\bpublic\s+release\s+ready\b",
        r"\brelease\s+ready\b",
        r"\bprivacy\s+pass\b",
        r"\bfull\s+privacy\s+pass\b",
        r"\bq6\s+pass\b",
        r"\bunrestricted\s+public\b",
        r"\bsafe\s+for\s+public\s+release\b",
        r"\bno\s+privacy\s+risk\b",
        r"\bformal\s+privacy\s+guarantee\b",
    ]
    return any(re.search(p, s_norm) for p in positive_patterns)

def _sanitize_permitted_for_governance_180(claim_text, dim):
    original = str(claim_text)

    if dim == "Q4_coupling" or _q4_stale_promoted_180(original):
        return (
            "Q4 is reported as a blocked/no-promotion governance boundary. "
            "A0/Zigbee evidence may be discussed only as diagnostic or candidate-stage evidence; "
            "no Q4-coupled artifact is authoritative or promoted."
        )

    if dim == "Q6_privacy_release" or _public_release_overclaim_180(original):
        return (
            "Q6 is claim-scoped: the public candidate has direct no-copy/DCR warning-level status, "
            "but strict combined public release remains blocked by distinguishability. "
            "No formal privacy guarantee or unrestricted public-release pass is claimed."
        )

    return original

def _forbidden_claim_180(row, permitted_claim, dim):
    existing = _first_present_180(row, ["forbidden_claim", "forbidden", "not_allowed_claim", "forbidden_wording"], "")
    if str(existing).strip():
        return str(existing).strip()

    if dim == "Q4_coupling":
        return "Do not claim that A0/Q4 was accepted, promoted, publication-ready, or that a Q4-coupled artifact is authoritative."
    if dim == "Q6_privacy_release":
        return "Do not claim unrestricted public release, formal privacy guarantee, Q6 pass, or full public-release readiness."
    if dim == "external_baselines":
        return "Do not claim external baselines are full CPS replacements or that pipeline superiority holds beyond fully comparable same-scope rows."
    if dim == "Q3_observability":
        return "Do not claim the public candidate preserves full fine-grained observability/missingness/staleness realism."
    return "Do not generalize beyond the claim scope, artifact scope, and evaluation split recorded for this claim."

def _claim_status_180(row, permitted_claim, forbidden_claim):
    """
    Assign claim-scoped status from claim-specific evidence only.

    Do NOT use global public_* context fields for every row; otherwise the
    global Q6 release_blocker incorrectly makes all claims look blocker-level.
    """
    claim_text = _infer_claim_text_180(row)
    dim = _infer_dimension_180(row)
    claim_type = str(_first_present_180(row, ["claim_type", "type"], "")).lower()
    metric_name = str(_infer_metric_name_180(row)).lower()
    permitted = str(permitted_claim)
    forbidden = str(forbidden_claim)

    claim_specific_text = " ".join([
        str(claim_text),
        str(permitted),
        str(forbidden),
        str(dim),
        str(claim_type),
        str(metric_name),
        str(_first_present_180(row, ["known_limitation", "limitation", "caveat"], "")),
    ]).lower()

    if _infer_test_used_for_selection_180(row) or _infer_test_used_for_repair_180(row):
        return "claim_blocked_test_leakage"

    # Positive stale Q4/public overclaims in the raw claim remain blocked,
    # but safe blocker/negation wording is allowed by the detectors.
    if _q4_stale_promoted_180(claim_text):
        return "claim_blocked_stale_q4_promotion"

    if _public_release_overclaim_180(claim_text):
        return "claim_blocked_public_release_overclaim"

    # Dimension-specific conservative statuses.
    if dim == "Q4_coupling":
        return "claim_supported_with_blocker_or_negative_result"

    if dim == "Q6_privacy_release":
        return "claim_supported_with_blocker_or_negative_result"

    if dim == "external_baselines":
        if "partial" in claim_specific_text or "limited" in claim_specific_text or "ctgan" in claim_specific_text or "unavailable" in claim_specific_text:
            return "claim_supported_with_caveat"
        return "claim_supported"

    if dim == "Q3_observability":
        return "claim_supported_with_caveat"

    if "fatal_n=0" in claim_specific_text or "success" in claim_specific_text or "done" in claim_specific_text:
        return "claim_supported"

    if "warning" in claim_specific_text or "partial" in claim_specific_text or "limited" in claim_specific_text or "caveat" in claim_specific_text:
        return "claim_supported_with_caveat"

    if "blocker" in claim_specific_text or "fatal" in claim_specific_text or "negative" in claim_specific_text:
        return "claim_supported_with_blocker_or_negative_result"

    # Most ledger claims are evidence-backed but still need 18.1/18.2 traceability/stability.
    return "claim_supported_with_caveat"

def _claim_scope_180(row, dim):
    val = _first_present_180(row, ["claim_scope", "scope_id", "scope", "artifact_scope", "role_scope"], "")
    if str(val).strip():
        return str(val).strip()
    if dim == "external_baselines":
        return "same_scope_only"
    if dim == "Q6_privacy_release":
        return "claim_scoped_release_governance"
    if dim == "Q4_coupling":
        return "q4_no_promotion_governance"
    return "dimension_scoped"

# ----------------------------------------------------------
# 4) Validate upstream governance
# ----------------------------------------------------------
cell17_6_version_180 = str(CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT.get("version", ""))
if "cell17_6_baseline_claim_sanitizer_v1_3" not in cell17_6_version_180:
    raise RuntimeError(f"[Cell18.0] Unexpected Cell 17.6 contract version: {cell17_6_version_180}")

q4 = CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT.get("q4_governance", {})
strict = CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT.get("strict_contract", {})
if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
    raise RuntimeError("[Cell18.0] Cell 17.6 does not carry Q4 blocked_no_promotion governance.")
if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
    raise RuntimeError("[Cell18.0] Cell 17.6 indicates Q4-coupled artifacts were used.")

q6_public_context = CELL17_6_BASELINE_CLAIM_SANITIZER_CONTRACT.get("q6_public_context", {})
public_direct_privacy_status = str(q6_public_context.get("public_direct_privacy_status", ""))
public_distinguishability_status = str(q6_public_context.get("public_distinguishability_status", ""))
public_strict_status = str(q6_public_context.get("public_strict_combined_status", ""))

# ----------------------------------------------------------
# 5) Load master ledger and build registry
# ----------------------------------------------------------
master_ledger_df = pd.read_csv(master_ledger_csv)
if len(master_ledger_df) == 0:
    raise RuntimeError(f"[Cell18.0] Master ledger is empty: {master_ledger_csv}")

claim_rows = []
for i, row in master_ledger_df.iterrows():
    dim = _infer_dimension_180(row)
    claim_text_raw = _infer_claim_text_180(row)
    permitted_claim = _first_present_180(row, ["permitted_claim", "allowed_claim", "safe_claim"], "")
    if not str(permitted_claim).strip():
        permitted_claim = _sanitize_permitted_for_governance_180(claim_text_raw, dim)
    else:
        permitted_claim = _sanitize_permitted_for_governance_180(permitted_claim, dim)

    forbidden_claim = _forbidden_claim_180(row, permitted_claim, dim)

    claim_id = _first_present_180(row, ["claim_id", "id", "claim_key"], "")
    if not str(claim_id).strip():
        claim_id = f"LEDGER-CLAIM-{i+1:04d}"

    source_cell = _infer_source_cell_180(row)
    metric_name = _infer_metric_name_180(row)
    metric_value = _infer_metric_value_180(row)
    artifact_type = _infer_artifact_type_180(row)
    artifact_path = _infer_artifact_path_180(row)
    selection_split = _infer_selection_split_180(row)
    evaluation_split = _infer_evaluation_split_180(row)
    test_used_for_selection = _infer_test_used_for_selection_180(row)
    test_used_for_repair = _infer_test_used_for_repair_180(row)
    publication_status = _claim_status_180(row, permitted_claim, forbidden_claim)

    claim_rows.append({
        "claim_id": str(claim_id),
        "claim_text_raw_from_ledger": str(claim_text_raw),
        "permitted_claim": str(permitted_claim),
        "forbidden_claim": str(forbidden_claim),
        "quality_dimension": str(dim),
        "claim_scope": _claim_scope_180(row, dim),
        "claim_type": str(_first_present_180(row, ["claim_type", "type"], "ledger_derived")),
        "expected_metric_name": str(metric_name),
        "metric_name": str(metric_name),
        "metric_value": metric_value,
        "expected_artifact_type": str(artifact_type),
        "artifact_path": str(artifact_path),
        "source_cell_candidates": str(_first_present_180(row, ["source_cell_candidates", "source_cells"], source_cell)),
        "source_cell": str(source_cell),
        "selection_split": str(selection_split),
        "evaluation_split": str(evaluation_split),
        "split_used": f"selection={selection_split};evaluation={evaluation_split}",
        "TEST_used_for_selection": bool(test_used_for_selection),
        "TEST_used_for_repair": bool(test_used_for_repair),
        "seed_policy": str(_first_present_180(row, ["seed_policy", "seed"], DEFAULT_SEED_POLICY_180)),
        "decision_grade_policy": str(_first_present_180(row, ["decision_grade_policy"], "assigned_in_cell18_2_from_metric_and_claim_scope")),
        "decision_grade": str(_first_present_180(row, ["decision_grade"], "pending_cell18_2")),
        "known_limitation": str(_first_present_180(row, ["known_limitation", "limitation", "caveat"], "")),
        "overclaim_risk": str(_first_present_180(row, ["overclaim_risk", "risk"], "medium")),
        "publication_status": str(publication_status),
        "claim_scoped_status": str(publication_status),
        "global_publication_ready": False,
        "q4_final_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
        "master_ledger_row_index": int(i),
        "master_ledger_path": master_ledger_csv,
        "TEST_real_values_used_here": False,
        "synthetic_values_mutated": False,
    })

claim_registry_df = pd.DataFrame(claim_rows)

# ----------------------------------------------------------
# 6) Detector self-tests
# ----------------------------------------------------------
_q4_detector_selftest_180 = {
    "no_Q4_coupled_public_artifact": False,
    "no Q4-coupled artifact is authoritative": False,
    "Q4 remains blocked_no_promotion": False,
    "A0 accepted": True,
    "Q4 promoted": True,
    "final Q4 artifact": True,
}
for _txt, _expected in _q4_detector_selftest_180.items():
    _got = bool(_q4_stale_promoted_180(_txt))
    if _got != _expected:
        raise RuntimeError(
            f"[Cell18.0] Q4 detector self-test failed: text={_txt!r} got={_got} expected={_expected}"
        )

_public_detector_selftest_180 = {
    "public_direct_privacy_status=release_warning": False,
    "public_strict_combined_status=release_blocker": False,
    "strict release remains blocked by distinguishability": False,
    "no formal privacy guarantee is claimed": False,
    "public release ready": True,
    "formal privacy guarantee": True,
    "Q6 pass": True,
}
for _txt, _expected in _public_detector_selftest_180.items():
    _got = bool(_public_release_overclaim_180(_txt))
    if _got != _expected:
        raise RuntimeError(
            f"[Cell18.0] Public-release detector self-test failed: text={_txt!r} got={_got} expected={_expected}"
        )

# ----------------------------------------------------------
# 7) Audit
# ----------------------------------------------------------
audit_rows = []

required_final_cols = [
    "claim_id",
    "permitted_claim",
    "forbidden_claim",
    "quality_dimension",
    "metric_name",
    "metric_value",
    "artifact_path",
    "source_cell",
    "selection_split",
    "evaluation_split",
    "TEST_used_for_selection",
    "TEST_used_for_repair",
    "split_used",
    "seed_policy",
    "decision_grade",
    "known_limitation",
    "publication_status",
    "claim_scoped_status",
]

missing_cols = [c for c in required_final_cols if c not in claim_registry_df.columns]
audit_rows.append({
    "audit_check": "required_claim_registry_columns_present",
    "passed": bool(len(missing_cols) == 0),
    "details": f"missing={missing_cols}",
})

dupes = claim_registry_df.loc[claim_registry_df["claim_id"].astype(str).duplicated(keep=False), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "claim_id_unique",
    "passed": bool(len(dupes) == 0),
    "details": f"duplicates={sorted(set(dupes))}",
})

empty_permitted = claim_registry_df.loc[claim_registry_df["permitted_claim"].astype(str).str.strip().eq(""), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "permitted_claim_nonempty",
    "passed": bool(len(empty_permitted) == 0),
    "details": f"empty_permitted_claim_ids={empty_permitted}",
})

empty_forbidden = claim_registry_df.loc[claim_registry_df["forbidden_claim"].astype(str).str.strip().eq(""), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "forbidden_claim_nonempty",
    "passed": bool(len(empty_forbidden) == 0),
    "details": f"empty_forbidden_claim_ids={empty_forbidden}",
})

test_sel = claim_registry_df.loc[claim_registry_df["TEST_used_for_selection"].astype(bool), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "no_test_used_for_selection_claims",
    "passed": bool(len(test_sel) == 0),
    "details": f"test_selection_claim_ids={test_sel}",
})

test_rep = claim_registry_df.loc[claim_registry_df["TEST_used_for_repair"].astype(bool), "claim_id"].astype(str).tolist()
audit_rows.append({
    "audit_check": "no_test_used_for_repair_claims",
    "passed": bool(len(test_rep) == 0),
    "details": f"test_repair_claim_ids={test_rep}",
})

if bool(CFG.get("cell18_0_strict_no_global_publication_ready", True)):
    text_cols = ["claim_text_raw_from_ledger", "permitted_claim", "publication_status", "claim_scoped_status"]
    global_ready = []
    for _, r in claim_registry_df.iterrows():
        s = " ".join([str(r.get(c, "")) for c in text_cols]).lower()
        if "publication_ready" in s or "global ready" in s or "all claims ready" in s:
            global_ready.append(str(r["claim_id"]))
    audit_rows.append({
        "audit_check": "no_global_publication_ready_language",
        "passed": bool(len(global_ready) == 0),
        "details": f"global_ready_claim_ids={global_ready}",
    })

if bool(CFG.get("cell18_0_strict_no_q4_promotion_claims", True)):
    q4_unsafe = []
    for _, r in claim_registry_df.iterrows():
        # Raw may be unsafe, but permitted claim must not be unsafe.
        if _q4_stale_promoted_180(str(r.get("permitted_claim", ""))):
            q4_unsafe.append(str(r["claim_id"]))
    audit_rows.append({
        "audit_check": "no_stale_q4_promotion_in_permitted_claims",
        "passed": bool(len(q4_unsafe) == 0),
        "details": f"q4_unsafe_permitted_claim_ids={q4_unsafe}",
    })

# Public-release overclaim blocked from permitted claims.
public_unsafe = []
for _, r in claim_registry_df.iterrows():
    if _public_release_overclaim_180(str(r.get("permitted_claim", ""))):
        public_unsafe.append(str(r["claim_id"]))
audit_rows.append({
    "audit_check": "no_public_release_overclaim_in_permitted_claims",
    "passed": bool(len(public_unsafe) == 0),
    "details": f"public_unsafe_permitted_claim_ids={public_unsafe}",
})

status_counts_pre_audit = claim_registry_df["claim_scoped_status"].astype(str).value_counts().to_dict()
all_blocker_status = (
    len(claim_registry_df) > 0
    and len(status_counts_pre_audit) == 1
    and "claim_supported_with_blocker_or_negative_result" in status_counts_pre_audit
)
audit_rows.append({
    "audit_check": "claim_scoped_status_not_globally_collapsed_to_blocker",
    "passed": bool(not all_blocker_status),
    "details": f"claim_scoped_status_counts={status_counts_pre_audit}",
})

claim_audit_df = pd.DataFrame(audit_rows)
audit_passed = bool(claim_audit_df["passed"].astype(bool).all())

if not audit_passed:
    raise RuntimeError(
        "[Cell18.0] Ledger-derived claim registry audit failed: "
        f"{claim_audit_df.loc[~claim_audit_df['passed'].astype(bool)].to_dict('records')}"
    )

if bool(CFG.get("cell18_0_require_unique_claim_ids", True)) and len(dupes):
    raise RuntimeError(f"[Cell18.0] Duplicate claim IDs: {sorted(set(dupes))}")

# ----------------------------------------------------------
# 8) Save outputs
# ----------------------------------------------------------
claim_registry_df.to_csv(claim_registry_csv, index=False)
claim_audit_df.to_csv(claim_registry_audit_csv, index=False)

contract = {
    "cell": "18.0",
    "version": CELL180_VERSION,
    "role": "ledger_derived_claim_registry_no_q4_promotion",
    "master_ledger": {
        "path": master_ledger_csv,
        "searched": master_ledger_candidates,
        "rows": int(len(master_ledger_df)),
        "sha256": _sha256_file_180(master_ledger_csv),
    },
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "stale_q4_promotion_claims_blocked_or_sanitized": True,
        "safe_no_q4_negations_allowed": True,
    },
    "q6_public_context": {
        "public_direct_privacy_status": public_direct_privacy_status,
        "public_distinguishability_status": public_distinguishability_status,
        "public_strict_combined_status": public_strict_status,
        "public_release_overclaims_blocked_or_sanitized": True,
        "safe_release_blocker_warning_wording_allowed": True,
        "claim_scoped_status_ignores_global_q6_context_for_non_q6_claims": True,
    },
    "upstream_contract_versions": {
        "cell17_6": cell17_6_version_180,
    },
    "summary": {
        "claims_total": int(len(claim_registry_df)),
        "quality_dimension_counts": claim_registry_df["quality_dimension"].astype(str).value_counts().sort_index().to_dict(),
        "claim_scoped_status_counts": claim_registry_df["claim_scoped_status"].astype(str).value_counts().sort_index().to_dict(),
        "audit_passed": bool(audit_passed),
        "audit_rows": int(len(claim_audit_df)),
        "test_used_for_selection_n": int(claim_registry_df["TEST_used_for_selection"].astype(bool).sum()),
        "test_used_for_repair_n": int(claim_registry_df["TEST_used_for_repair"].astype(bool).sum()),
    },
    "required_final_traceability_columns": required_final_cols,
    "reserved_final_outputs": {
        "claim_traceability_manifest_csv": final_claim_trace_csv,
        "claim_traceability_manifest_json": final_claim_trace_json,
        "publication_readiness_dashboard_csv": final_readiness_csv,
        "publication_blockers_csv": final_blockers_csv,
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "claim_registry_done_here": True,
        "generated_from_master_results_ledger": True,
        "global_publication_ready_used": False,
        "claim_scoped_status_ignores_global_q6_context_for_non_q6_claims": True,
        "safe_no_q4_negations_allowed": True,
        "safe_release_blocker_warning_wording_allowed": True,
    },
    "outputs": {
        "claim_registry_csv": claim_registry_csv,
        "claim_registry_audit_csv": claim_registry_audit_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_180(contract_json, contract)
_write_json_180(contract_canonical_json, contract)

manifest = {
    "cell": "18.0",
    "version": CELL180_VERSION,
    "created_outputs": contract["outputs"],
    "master_ledger": contract["master_ledger"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "summary": contract["summary"],
    "reserved_final_outputs": contract["reserved_final_outputs"],
    "strict_contract": contract["strict_contract"],
}

_write_json_180(manifest_json, manifest)

hashes = {
    "claim_registry_csv_sha256": _sha256_file_180(claim_registry_csv),
    "claim_registry_audit_csv_sha256": _sha256_file_180(claim_registry_audit_csv),
    "contract_json_sha256": _sha256_file_180(contract_json),
    "contract_canonical_json_sha256": _sha256_file_180(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_180(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_180(contract_json, contract)
_write_json_180(contract_canonical_json, contract)
_write_json_180(manifest_json, manifest)

# ----------------------------------------------------------
# 9) Export globals
# ----------------------------------------------------------
globals()["CELL180_VERSION"] = CELL180_VERSION
globals()["CELL18_0_CLAIM_REGISTRY_DF"] = claim_registry_df
globals()["CELL18_0_CLAIM_REGISTRY_AUDIT_DF"] = claim_audit_df
globals()["CELL18_0_CLAIM_REGISTRY_CONTRACT"] = contract

globals()["CELL18_0_CLAIM_REGISTRY_CSV"] = claim_registry_csv
globals()["CELL18_0_CLAIM_REGISTRY_AUDIT_CSV"] = claim_registry_audit_csv
globals()["CELL18_0_CLAIM_REGISTRY_CONTRACT_JSON"] = contract_json
globals()["CELL18_0_CLAIM_REGISTRY_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL18_0_CLAIM_REGISTRY_MANIFEST_JSON"] = manifest_json

globals()["CELL18_FINAL_CLAIM_TRACEABILITY_MANIFEST_CSV"] = final_claim_trace_csv
globals()["CELL18_FINAL_CLAIM_TRACEABILITY_MANIFEST_JSON"] = final_claim_trace_json
globals()["CELL18_FINAL_PUBLICATION_READINESS_DASHBOARD_CSV"] = final_readiness_csv
globals()["CELL18_FINAL_PUBLICATION_BLOCKERS_CSV"] = final_blockers_csv

log(
    "[Cell18.0] Ledger-derived claim registry complete | "
    f"claims={len(claim_registry_df)} | "
    f"dimensions={claim_registry_df['quality_dimension'].nunique()} | "
    f"audit_passed={audit_passed} | "
    f"master_ledger={master_ledger_csv}"
)
log(
    "[Cell18.0] Claim counts by dimension | "
    f"{claim_registry_df['quality_dimension'].astype(str).value_counts().sort_index().to_dict()}"
)
log(f"[Cell18.0] Saved claim registry: {claim_registry_csv}")
log(f"[Cell18.0] Saved audit: {claim_registry_audit_csv}")
log(f"[Cell18.0] Saved canonical contract: {contract_canonical_json}")
log(
    "[Cell18.0] Reserved final outputs | "
    f"claim_traceability_manifest_csv={final_claim_trace_csv} | "
    f"claim_traceability_manifest_json={final_claim_trace_json} | "
    f"publication_readiness_dashboard_csv={final_readiness_csv} | "
    f"publication_blockers_csv={final_blockers_csv}"
)
log(
    "[Cell18.0] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TEST_real_values_used=False | "
    "TEST_real_values_used_for_materialization=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "claim_registry_done_here=True | "
    "generated_from_master_results_ledger=True | "
    "global_publication_ready_used=False | safe_release_blocker_warning_wording_allowed=True | safe_no_q4_negations_allowed=True | claim_scoped_status_ignores_global_q6_context_for_non_q6_claims=True"
)
log("--- END: Cell 18.0 - Claim registry (v1.3 claim-scoped-status ledger-derived no-Q4-promotion strict) ---")

gc.collect()