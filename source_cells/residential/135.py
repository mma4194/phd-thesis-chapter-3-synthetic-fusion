# ==========================================================
# CELL 15.3 - Q6 MIA audit: random split + temporal split
# v1.2 STUDY-THESIS separated-risk role-specific MIA audit, DCR/NNDR-forensic-aware
#
# Role:
#   - Audit membership-inference exposure by role group.
#   - Runs two complementary MIA protocols:
#       1) random split MIA
#       2) temporal split MIA
#   - Uses real TRAIN-like members vs synthetic/non-member references.
#   - Reports classifier separability as release-risk evidence.
#
# Scientific contract:
#   - No synthetic mutation.
#   - No generator fitting.
#   - TEST real values are used only as privacy/reference audit evidence.
#   - MIA is release-risk evidence, not a formal privacy proof.
#
# Outputs:
#   reports/cell15_3_mia_role_metrics.csv
#   reports/cell15_3_mia_protocol_metrics.csv
#   reports/cell15_3_mia_role_config.csv
#   reports/cell15_3_mia_contract.json
#   artifacts/cell15_3_mia_manifest.json
# ==========================================================

log("--- START: Cell 15.3 - Q6 MIA audit random + temporal split (v1.2 DCR/NNDR-forensic-aware separated-risk) ---")

import os
import gc
import json
import hashlib
from collections import Counter

import numpy as np
import pandas as pd

try:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
    SKLEARN_MIA_AVAILABLE_153 = True
except Exception:
    SKLEARN_MIA_AVAILABLE_153 = False

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_153 = [
    "CFG", "log",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "REAL_TRAIN_REF_150",
    "REAL_VAL_REF_150",
    "REAL_TEST_REF_150",
    "SYN_TEST_REF_150",
    "CELL15_0_PRIVACY_ROLE_GROUPS",
    "CELL15_1_NO_COPY_WINDOW_METRICS_DF",
    "CELL15_2_DCR_NNDR_ROLE_METRICS_DF",
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "CELL15_1_NO_COPY_CONTRACT",
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "CELL15_2_DCR_NNDR_CONTRACT",
    "CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT",
    "CELL15_2A_DCR_NNDR_BLOCKER_FORENSIC_DF",
]
_missing_153 = [k for k in _required_153 if k not in globals()]
if _missing_153:
    raise RuntimeError(f"[Cell15.3] Missing required globals: {_missing_153}")

if not SKLEARN_MIA_AVAILABLE_153:
    raise RuntimeError("[Cell15.3] scikit-learn is required for MIA audit.")

OUTDIR = str(OUTDIR)
OUT_SYN = str(OUT_SYN)
REPORT_DIR = str(REPORT_DIR)
ARTDIR = os.path.join(OUTDIR, "artifacts")
CONTRACT_DIR = os.path.join(ARTDIR, "contracts")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(ARTDIR, exist_ok=True)
os.makedirs(CONTRACT_DIR, exist_ok=True)

SEED = int(SEED)

REAL_TRAIN_153 = REAL_TRAIN_REF_150
REAL_VAL_153 = REAL_VAL_REF_150
REAL_TEST_153 = REAL_TEST_REF_150
SYN_TEST_153 = SYN_TEST_REF_150

N_TR = int(len(REAL_TRAIN_153))
N_VAL = int(len(REAL_VAL_153))
N_TE = int(len(REAL_TEST_153))
N_SYN = int(len(SYN_TEST_153))

CELL153_VERSION = "cell15_3_q6_mia_random_temporal_privacy_v1_2_dcr_forensic_aware"

CFG["cell15_3_version"] = CELL153_VERSION
CFG["cell15_3_TEST_real_values_used_for_privacy_reference"] = True
CFG["cell15_3_synthetic_values_mutated"] = False
CFG["cell15_3_selection_done_here"] = False
CFG["cell15_3_generator_fit_done_here"] = False
CFG["cell15_3_materialization_done_here"] = False
CFG["cell15_3_privacy_decision_done_here"] = False
CFG["cell15_3_mia_audit_done_here"] = True
CFG["cell15_3_Q4_final_status"] = "blocked_no_promotion"
CFG["cell15_3_Q4_coupled_artifacts_used"] = False
CFG["cell15_3_DCR_NNDR_forensic_carried_forward"] = True
CFG["cell15_3_MIA_does_not_clear_prior_privacy_blockers"] = True

# ----------------------------------------------------------
# 1) Config
# ----------------------------------------------------------
CFG.setdefault("cell15_3_max_cols_per_role", 80)
CFG.setdefault("cell15_3_max_samples_per_class", 8000)
CFG.setdefault("cell15_3_max_train_rows_for_classifier", 12000)
CFG.setdefault("cell15_3_low_variance_eps", 1e-12)
CFG.setdefault("cell15_3_fill_value", -999999.0)
CFG.setdefault("cell15_3_clip_z", 20.0)
CFG.setdefault("cell15_3_skip_roles", ["iot_placeholders_or_excluded"])
CFG.setdefault("cell15_3_models", ["logistic", "random_forest"])
CFG.setdefault("cell15_3_random_member_source", "real_train")
CFG.setdefault("cell15_3_random_nonmember_source", "synthetic_test")
CFG.setdefault("cell15_3_temporal_member_source", "real_train")
CFG.setdefault("cell15_3_temporal_nonmember_source", "real_test")
CFG.setdefault("cell15_3_classifier_max_iter", 3000)
CFG.setdefault("cell15_3_random_forest_estimators", 120)
CFG.setdefault("cell15_3_blocker_auc", 0.70)
CFG.setdefault("cell15_3_warning_auc", 0.60)
CFG.setdefault("cell15_3_blocker_balanced_accuracy", 0.65)
CFG.setdefault("cell15_3_warning_balanced_accuracy", 0.58)
CFG.setdefault("cell15_3_synth_distinguishability_blocker_auc", 0.85)
CFG.setdefault("cell15_3_synth_distinguishability_warning_auc", 0.70)
CFG.setdefault("cell15_3_synth_distinguishability_blocker_bal_acc", 0.80)
CFG.setdefault("cell15_3_synth_distinguishability_warning_bal_acc", 0.65)
CFG.setdefault("cell15_3_temporal_drift_high_auc", 0.85)
CFG.setdefault("cell15_3_temporal_drift_moderate_auc", 0.70)
CFG.setdefault("cell15_3_temporal_drift_high_bal_acc", 0.80)
CFG.setdefault("cell15_3_temporal_drift_moderate_bal_acc", 0.65)
CFG.setdefault("cell15_3_treat_synthetic_distinguishability_as_release_blocker", True)
CFG.setdefault("cell15_3_treat_temporal_drift_as_release_blocker", False)

MAX_COLS_153 = int(CFG.get("cell15_3_max_cols_per_role", 80))
MAX_PER_CLASS_153 = int(CFG.get("cell15_3_max_samples_per_class", 8000))
MAX_CLF_ROWS_153 = int(CFG.get("cell15_3_max_train_rows_for_classifier", 12000))
LOW_VAR_EPS_153 = float(CFG.get("cell15_3_low_variance_eps", 1e-12))
FILL_VALUE_153 = float(CFG.get("cell15_3_fill_value", -999999.0))
CLIP_Z_153 = float(CFG.get("cell15_3_clip_z", 20.0))
SKIP_ROLES_153 = set(map(str, CFG.get("cell15_3_skip_roles", ["iot_placeholders_or_excluded"])))
MODELS_153 = list(map(str, CFG.get("cell15_3_models", ["logistic", "random_forest"])))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
role_metrics_csv = os.path.join(REPORT_DIR, "cell15_3_mia_role_metrics.csv")
protocol_metrics_csv = os.path.join(REPORT_DIR, "cell15_3_mia_protocol_metrics.csv")
role_config_csv = os.path.join(REPORT_DIR, "cell15_3_mia_role_config.csv")
contract_json = os.path.join(REPORT_DIR, "cell15_3_mia_contract.json")
contract_canonical_json = os.path.join(CONTRACT_DIR, "cell15_3_mia_contract_v1_2_THESIS.json")
manifest_json = os.path.join(ARTDIR, "cell15_3_mia_manifest.json")

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_153(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_153(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_153(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_153(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_153(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_153(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_153(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_153(obj.to_dict())
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

def _write_json_153(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_153(payload), f, indent=2, sort_keys=True)

def _sha256_file_153(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _require_contract_version_153(obj, name: str, expected_substring: str) -> str:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[Cell15.3] {name} is not a dict.")
    version = str(obj.get("version", ""))
    if expected_substring not in version:
        raise RuntimeError(
            f"[Cell15.3] Unexpected {name} version. Expected substring={expected_substring}, got={version}"
        )
    return version

def _dedup_153(seq):
    seen = set()
    out = []
    for x in seq:
        x = str(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _source_frame_153(source_name: str):
    s = str(source_name)
    if s == "real_train":
        return REAL_TRAIN_153
    if s == "real_val":
        return REAL_VAL_153
    if s == "real_test":
        return REAL_TEST_153
    if s == "synthetic_test":
        return SYN_TEST_153
    raise ValueError(f"[Cell15.3] Unknown source: {source_name}")

def _available_common_cols_153(cols, sources):
    cols = _dedup_153(cols)
    frames = [_source_frame_153(s) for s in sources]
    return [c for c in cols if all(c in f.columns for f in frames)]

def _choose_cols_153(role: str, cols: list, sources: list):
    cols = _available_common_cols_153(cols, sources)
    if not cols:
        return []

    rows = []
    frames = [_source_frame_153(s) for s in sources]

    for c in cols:
        finite_rates = []
        stds = []
        for f in frames:
            x = pd.to_numeric(f[c], errors="coerce")
            finite_rates.append(float(x.notna().mean()) if len(x) else 0.0)
            stds.append(float(np.nanstd(x.to_numpy(dtype=np.float64))) if x.notna().any() else 0.0)

        rows.append({
            "col": c,
            "score": min(finite_rates) * np.log1p(max(stds)),
            "min_finite_rate": min(finite_rates),
            "max_std": max(stds),
        })

    d = pd.DataFrame(rows)
    d["score"] = pd.to_numeric(d["score"], errors="coerce").fillna(0.0)
    d["min_finite_rate"] = pd.to_numeric(d["min_finite_rate"], errors="coerce").fillna(0.0)
    d["max_std"] = pd.to_numeric(d["max_std"], errors="coerce").fillna(0.0)

    if role == "iot_event_drivers":
        dd = d.copy()
    else:
        dd = d[
            (d["min_finite_rate"] > 0.001)
            & (d["max_std"] > LOW_VAR_EPS_153)
        ].copy()
        if len(dd) == 0:
            dd = d.copy()

    return dd.sort_values(["score", "col"], ascending=[False, True])["col"].head(MAX_COLS_153).astype(str).tolist()

def _sample_indices_153(n, max_n, seed_offset, temporal_mode=False, side="head"):
    n = int(n)
    max_n = int(max_n)
    if n <= 0:
        return np.asarray([], dtype=np.int64)

    take = min(n, max_n)
    if temporal_mode:
        if side == "head":
            return np.arange(0, take, dtype=np.int64)
        if side == "tail":
            return np.arange(n - take, n, dtype=np.int64)
        if side == "middle":
            start = max(0, (n - take) // 2)
            return np.arange(start, start + take, dtype=np.int64)

    rng = np.random.default_rng(SEED + seed_offset)
    idx = rng.choice(np.arange(n, dtype=np.int64), size=take, replace=False)
    return np.asarray(np.sort(idx), dtype=np.int64)

def _fit_scaler_153(cols, member_source):
    f = _source_frame_153(member_source)
    med = []
    scale = []

    for c in cols:
        x = pd.to_numeric(f[c], errors="coerce").to_numpy(dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            med.append(0.0)
            scale.append(1.0)
            continue

        q25, q50, q75 = np.nanquantile(x, [0.25, 0.50, 0.75])
        sc = float(q75 - q25)
        if not np.isfinite(sc) or sc <= LOW_VAR_EPS_153:
            sc = float(np.nanstd(x))
        if not np.isfinite(sc) or sc <= LOW_VAR_EPS_153:
            sc = 1.0

        med.append(float(q50))
        scale.append(float(sc))

    return np.asarray(med, dtype=np.float64), np.asarray(scale, dtype=np.float64)

def _matrix_from_frame_153(frame, cols, idx, med, scale):
    if len(cols) == 0 or len(idx) == 0:
        return np.empty((0, 0), dtype=np.float32)

    arr = frame.iloc[idx][cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    z = (arr - med.reshape(1, -1)) / scale.reshape(1, -1)
    z = np.where(np.isfinite(z), z, FILL_VALUE_153)
    z = np.clip(z, -CLIP_Z_153, CLIP_Z_153)
    return z.astype(np.float32)

def _balanced_train_test_split_153(X, y, seed):
    # Manual balanced split, no dependency on train_test_split.
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = max(1, int(0.35 * len(idx0)))
    n1_test = max(1, int(0.35 * len(idx1)))

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])

    if len(train_idx) > MAX_CLF_ROWS_153:
        train_idx = rng.choice(train_idx, size=MAX_CLF_ROWS_153, replace=False)

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx

def _fit_predict_153(model_name, X_train, y_train, X_test):
    model_name = str(model_name)

    if model_name == "logistic":
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                max_iter=int(CFG.get("cell15_3_classifier_max_iter", 3000)),
                solver="lbfgs",
                class_weight="balanced",
                random_state=SEED + 153,
            ),
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(CFG.get("cell15_3_random_forest_estimators", 120)),
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=SEED + 153,
        )
    else:
        raise ValueError(f"[Cell15.3] Unknown MIA model: {model_name}")

    clf.fit(X_train, y_train)

    if hasattr(clf, "predict_proba"):
        score = clf.predict_proba(X_test)[:, 1]
    else:
        score = clf.decision_function(X_test)

    pred = (score >= np.nanmedian(score)).astype(int)
    return score, pred

def _safe_metric_153(fn, *args, default=np.nan):
    try:
        return float(fn(*args))
    except Exception:
        return float(default)

def _mia_status_153(max_auc, max_bal_acc, carried_no_copy, carried_dcr):
    reasons = []

    if np.isfinite(max_auc) and max_auc >= float(CFG.get("cell15_3_blocker_auc", 0.70)):
        reasons.append("MIA_AUC_blocker")

    if np.isfinite(max_bal_acc) and max_bal_acc >= float(CFG.get("cell15_3_blocker_balanced_accuracy", 0.65)):
        reasons.append("MIA_balanced_accuracy_blocker")

    if str(carried_no_copy) == "blocker":
        reasons.append("carried_no_copy_blocker")

    if str(carried_dcr) == "blocker":
        reasons.append("carried_DCR_NNDR_blocker")

    if reasons:
        return "blocker", "|".join(reasons)

    warn = []
    if np.isfinite(max_auc) and max_auc >= float(CFG.get("cell15_3_warning_auc", 0.60)):
        warn.append("MIA_AUC_warning")

    if np.isfinite(max_bal_acc) and max_bal_acc >= float(CFG.get("cell15_3_warning_balanced_accuracy", 0.58)):
        warn.append("MIA_balanced_accuracy_warning")

    if str(carried_no_copy) == "warning":
        warn.append("carried_no_copy_warning")

    if str(carried_dcr) == "warning":
        warn.append("carried_DCR_NNDR_warning")

    if warn:
        return "warning", "|".join(warn)

    return "pass", "within_MIA_privacy_gates"

def _synthetic_distinguishability_status_153(max_auc, max_bal_acc):
    reasons = []
    if np.isfinite(max_auc) and max_auc >= float(CFG.get("cell15_3_synth_distinguishability_blocker_auc", 0.85)):
        reasons.append("synthetic_distinguishability_AUC_blocker")
    if np.isfinite(max_bal_acc) and max_bal_acc >= float(CFG.get("cell15_3_synth_distinguishability_blocker_bal_acc", 0.80)):
        reasons.append("synthetic_distinguishability_balanced_accuracy_blocker")
    if reasons:
        return "blocker", "|".join(reasons)

    warn = []
    if np.isfinite(max_auc) and max_auc >= float(CFG.get("cell15_3_synth_distinguishability_warning_auc", 0.70)):
        warn.append("synthetic_distinguishability_AUC_warning")
    if np.isfinite(max_bal_acc) and max_bal_acc >= float(CFG.get("cell15_3_synth_distinguishability_warning_bal_acc", 0.65)):
        warn.append("synthetic_distinguishability_balanced_accuracy_warning")
    if warn:
        return "warning", "|".join(warn)

    return "pass", "synthetic_distinguishability_within_gates"

def _temporal_drift_status_153(max_auc, max_bal_acc):
    reasons = []
    if np.isfinite(max_auc) and max_auc >= float(CFG.get("cell15_3_temporal_drift_high_auc", 0.85)):
        reasons.append("high_temporal_real_train_test_drift_AUC")
    if np.isfinite(max_bal_acc) and max_bal_acc >= float(CFG.get("cell15_3_temporal_drift_high_bal_acc", 0.80)):
        reasons.append("high_temporal_real_train_test_drift_balanced_accuracy")
    if reasons:
        return "warning", "|".join(reasons)

    moderate = []
    if np.isfinite(max_auc) and max_auc >= float(CFG.get("cell15_3_temporal_drift_moderate_auc", 0.70)):
        moderate.append("moderate_temporal_real_train_test_drift_AUC")
    if np.isfinite(max_bal_acc) and max_bal_acc >= float(CFG.get("cell15_3_temporal_drift_moderate_bal_acc", 0.65)):
        moderate.append("moderate_temporal_real_train_test_drift_balanced_accuracy")
    if moderate:
        return "warning", "|".join(moderate)

    return "pass", "temporal_drift_within_calibration_gates"

def _carried_privacy_status_153(carried_no_copy, carried_dcr):
    reasons = []
    if str(carried_no_copy) == "blocker":
        reasons.append("carried_no_copy_blocker")
    if str(carried_dcr) == "blocker":
        reasons.append("carried_DCR_NNDR_blocker")
    if reasons:
        return "blocker", "|".join(reasons)

    warn = []
    if str(carried_no_copy) == "warning":
        warn.append("carried_no_copy_warning")
    if str(carried_dcr) == "warning":
        warn.append("carried_DCR_NNDR_warning")
    if warn:
        return "warning", "|".join(warn)

    return "pass", "no_carried_privacy_blockers"

def _combine_release_status_153(synth_status, temporal_status, carried_status):
    reasons = []

    # Hard release blockers.
    if str(carried_status) == "blocker":
        reasons.append("carried_privacy_blocker")

    if bool(CFG.get("cell15_3_treat_synthetic_distinguishability_as_release_blocker", True)):
        if str(synth_status) == "blocker":
            reasons.append("synthetic_distinguishability_blocker")

    if bool(CFG.get("cell15_3_treat_temporal_drift_as_release_blocker", False)):
        if str(temporal_status) == "blocker":
            reasons.append("temporal_drift_blocker")

    if reasons:
        return "blocker", "|".join(reasons)

    warn = []
    if str(carried_status) == "warning":
        warn.append("carried_privacy_warning")
    if str(synth_status) == "warning":
        warn.append("synthetic_distinguishability_warning")
    if str(temporal_status) == "warning":
        warn.append("temporal_drift_warning_context_only")
    if str(synth_status) == "blocker" and not bool(CFG.get("cell15_3_treat_synthetic_distinguishability_as_release_blocker", True)):
        warn.append("synthetic_distinguishability_blocker_downgraded_by_policy")
    if str(temporal_status) == "blocker" and not bool(CFG.get("cell15_3_treat_temporal_drift_as_release_blocker", False)):
        warn.append("temporal_drift_blocker_downgraded_by_policy")

    if warn:
        return "warning", "|".join(warn)

    return "pass", "within_separated_MIA_release_gates"

# ----------------------------------------------------------
# 4) Validate upstream no-Q4 and prior Q6 privacy contracts
# ----------------------------------------------------------
version150_153 = _require_contract_version_153(
    CELL15_0_PRIVACY_INPUT_CONTRACT,
    "CELL15_0_PRIVACY_INPUT_CONTRACT",
    "cell15_0_q6_privacy_input_preparation_v1_1_no_q4_promotion",
)
version151_153 = _require_contract_version_153(
    CELL15_1_NO_COPY_CONTRACT,
    "CELL15_1_NO_COPY_CONTRACT",
    "cell15_1_q6_no_copy_window_audit_v1_1",
)
version151a_153 = _require_contract_version_153(
    CELL15_1A_NO_COPY_FORENSIC_CONTRACT,
    "CELL15_1A_NO_COPY_FORENSIC_CONTRACT",
    "cell15_1a_q6_no_copy_blocker_forensic_v1_0",
)
version152_153 = _require_contract_version_153(
    CELL15_2_DCR_NNDR_CONTRACT,
    "CELL15_2_DCR_NNDR_CONTRACT",
    "cell15_2_q6_dcr_nndr_role_specific_privacy_v1_2",
)
version152a_153 = _require_contract_version_153(
    CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT,
    "CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT",
    "cell15_2a_q6_dcr_nndr_blocker_forensic_v1_0",
)

strict150_153 = CELL15_0_PRIVACY_INPUT_CONTRACT.get("strict_contract", {})
if str(strict150_153.get("Q4_final_status", "")) != "blocked_no_promotion":
    raise RuntimeError("[Cell15.3] Cell 15.0 does not carry Q4_final_status=blocked_no_promotion.")
if bool(strict150_153.get("Q4_coupled_artifacts_used", True)):
    raise RuntimeError("[Cell15.3] Cell 15.0 used Q4-coupled artifacts unexpectedly.")

for name, contract, key in [
    ("Cell15.1", CELL15_1_NO_COPY_CONTRACT, "no_copy_window_audit_done_here"),
    ("Cell15.2", CELL15_2_DCR_NNDR_CONTRACT, "dcr_nndr_audit_done_here"),
]:
    strict = contract.get("strict_contract", {})
    if bool(strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell15.3] {name} indicates synthetic mutation.")
    if not bool(strict.get(key, False)):
        raise RuntimeError(f"[Cell15.3] {name} did not complete required audit.")

for name, contract in [
    ("Cell15.1a", CELL15_1A_NO_COPY_FORENSIC_CONTRACT),
    ("Cell15.2a", CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT),
]:
    strict = contract.get("strict_contract", {})
    if bool(strict.get("synthetic_values_mutated", True)):
        raise RuntimeError(f"[Cell15.3] {name} indicates synthetic mutation.")
    if not bool(strict.get("diagnostic_only", False)):
        raise RuntimeError(f"[Cell15.3] {name} did not declare diagnostic_only.")

no_copy_forensic_summary_153 = CELL15_1A_NO_COPY_FORENSIC_CONTRACT.get("summary", {})
dcr_forensic_summary_153 = CELL15_2A_DCR_NNDR_FORENSIC_CONTRACT.get("summary", {})
no_copy_blocker_roles_153 = list(map(str, no_copy_forensic_summary_153.get("blocker_roles", [])))
dcr_blocker_roles_153 = list(map(str, dcr_forensic_summary_153.get("blocker_roles", [])))
dcr_warning_roles_153 = list(map(str, dcr_forensic_summary_153.get("warning_roles", [])))
combined_carried_blocker_roles_153 = sorted(set(no_copy_blocker_roles_153) | set(dcr_blocker_roles_153))
combined_carried_warning_roles_153 = sorted((set(dcr_warning_roles_153) | set(map(str, no_copy_forensic_summary_153.get("warning_roles", [])))) - set(combined_carried_blocker_roles_153))

# ----------------------------------------------------------
# 5) Run MIA protocols
# ----------------------------------------------------------
role_groups = CELL15_0_PRIVACY_ROLE_GROUPS

no_copy_df = CELL15_1_NO_COPY_WINDOW_METRICS_DF.copy()
dcr_df = CELL15_2_DCR_NNDR_ROLE_METRICS_DF.copy()

no_copy_status_map = dict(zip(no_copy_df["role_group"].astype(str), no_copy_df["status"].astype(str))) if len(no_copy_df) else {}
dcr_status_map = dict(zip(dcr_df["role_group"].astype(str), dcr_df["status"].astype(str))) if len(dcr_df) else {}

protocols_153 = [
    {
        "mia_protocol": "random_split_synthetic_nonmember",
        "member_source": str(CFG.get("cell15_3_random_member_source", "real_train")),
        "nonmember_source": str(CFG.get("cell15_3_random_nonmember_source", "synthetic_test")),
        "temporal_mode": False,
    },
    {
        "mia_protocol": "temporal_split_real_holdout",
        "member_source": str(CFG.get("cell15_3_temporal_member_source", "real_train")),
        "nonmember_source": str(CFG.get("cell15_3_temporal_nonmember_source", "real_test")),
        "temporal_mode": True,
    },
]

roles_to_run = [r for r in role_groups.keys() if r not in SKIP_ROLES_153]

protocol_metric_rows = []
role_config_rows = []

log(
    "[Cell15.3] Running MIA audit | "
    f"roles={roles_to_run} | protocols={[p['mia_protocol'] for p in protocols_153]} | models={MODELS_153}"
)

for role_idx, role_name in enumerate(roles_to_run, start=1):
    raw_cols = role_groups.get(role_name, [])

    for proto_idx, proto in enumerate(protocols_153, start=1):
        member_source = proto["member_source"]
        nonmember_source = proto["nonmember_source"]
        temporal_mode = bool(proto["temporal_mode"])
        mia_protocol = proto["mia_protocol"]

        cols = _choose_cols_153(role_name, raw_cols, [member_source, nonmember_source])

        role_config_rows.append({
            "role_group": role_name,
            "mia_protocol": mia_protocol,
            "raw_cols_n": int(len(raw_cols)),
            "selected_cols_n": int(len(cols)),
            "member_source": member_source,
            "nonmember_source": nonmember_source,
            "temporal_mode": temporal_mode,
            "max_samples_per_class": int(MAX_PER_CLASS_153),
            "models": "|".join(MODELS_153),
        })

        if len(cols) == 0:
            for model_name in MODELS_153:
                protocol_metric_rows.append({
                    "role_group": role_name,
                    "mia_protocol": mia_protocol,
                    "model": model_name,
                    "risk_interpretation": (
                        "synthetic_distinguishability"
                        if mia_protocol == "random_split_synthetic_nonmember"
                        else "temporal_drift_calibration"
                    ),
                    "direct_privacy_blocker_candidate": bool(mia_protocol == "random_split_synthetic_nonmember"),
                    "status": "not_evaluable",
                    "reasons": "no_common_informative_columns",
                    "selected_cols_n": 0,
                    "member_source": member_source,
                    "nonmember_source": nonmember_source,
                    "member_n": 0,
                    "nonmember_n": 0,
                    "test_auc": np.nan,
                    "test_average_precision": np.nan,
                    "test_accuracy": np.nan,
                    "test_balanced_accuracy": np.nan,
                    "carried_no_copy_status": no_copy_status_map.get(role_name, ""),
                    "carried_dcr_nndr_status": dcr_status_map.get(role_name, ""),
                    "TEST_real_values_used_for_privacy_reference": True,
                    "synthetic_values_mutated": False,
                })
            continue

        log(
            f"[Cell15.3] role {role_idx}/{len(roles_to_run)} | {role_name} | "
            f"protocol={mia_protocol} | cols={len(cols)}"
        )

        member_frame = _source_frame_153(member_source)
        nonmember_frame = _source_frame_153(nonmember_source)

        if temporal_mode:
            member_idx = _sample_indices_153(len(member_frame), MAX_PER_CLASS_153, 15310 + role_idx + proto_idx, temporal_mode=True, side="tail")
            nonmember_idx = _sample_indices_153(len(nonmember_frame), MAX_PER_CLASS_153, 15320 + role_idx + proto_idx, temporal_mode=True, side="tail")
        else:
            member_idx = _sample_indices_153(len(member_frame), MAX_PER_CLASS_153, 15310 + role_idx + proto_idx, temporal_mode=False)
            nonmember_idx = _sample_indices_153(len(nonmember_frame), MAX_PER_CLASS_153, 15320 + role_idx + proto_idx, temporal_mode=False)

        n = min(len(member_idx), len(nonmember_idx))
        if n < 100:
            for model_name in MODELS_153:
                protocol_metric_rows.append({
                    "role_group": role_name,
                    "mia_protocol": mia_protocol,
                    "model": model_name,
                    "risk_interpretation": (
                        "synthetic_distinguishability"
                        if mia_protocol == "random_split_synthetic_nonmember"
                        else "temporal_drift_calibration"
                    ),
                    "direct_privacy_blocker_candidate": bool(mia_protocol == "random_split_synthetic_nonmember"),
                    "status": "not_evaluable",
                    "reasons": "insufficient_balanced_samples",
                    "selected_cols_n": int(len(cols)),
                    "member_source": member_source,
                    "nonmember_source": nonmember_source,
                    "member_n": int(len(member_idx)),
                    "nonmember_n": int(len(nonmember_idx)),
                    "test_auc": np.nan,
                    "test_average_precision": np.nan,
                    "test_accuracy": np.nan,
                    "test_balanced_accuracy": np.nan,
                    "carried_no_copy_status": no_copy_status_map.get(role_name, ""),
                    "carried_dcr_nndr_status": dcr_status_map.get(role_name, ""),
                    "TEST_real_values_used_for_privacy_reference": True,
                    "synthetic_values_mutated": False,
                })
            continue

        member_idx = member_idx[:n]
        nonmember_idx = nonmember_idx[:n]

        med, scale = _fit_scaler_153(cols, member_source=member_source)
        X_member = _matrix_from_frame_153(member_frame, cols, member_idx, med, scale)
        X_nonmember = _matrix_from_frame_153(nonmember_frame, cols, nonmember_idx, med, scale)

        X = np.vstack([X_member, X_nonmember]).astype(np.float32)
        y = np.concatenate([
            np.ones(X_member.shape[0], dtype=np.int8),
            np.zeros(X_nonmember.shape[0], dtype=np.int8),
        ])

        X_train, X_test, y_train, y_test, train_idx, test_idx = _balanced_train_test_split_153(
            X,
            y,
            seed=SEED + 15300 + role_idx * 10 + proto_idx,
        )

        for model_name in MODELS_153:
            try:
                score, pred = _fit_predict_153(model_name, X_train, y_train, X_test)

                auc = _safe_metric_153(roc_auc_score, y_test, score)
                ap = _safe_metric_153(average_precision_score, y_test, score)
                acc = _safe_metric_153(accuracy_score, y_test, pred)
                bal_acc = _safe_metric_153(balanced_accuracy_score, y_test, pred)

                # Per-protocol local status, without carried upstream statuses.
                if mia_protocol == "random_split_synthetic_nonmember":
                    local_status, local_reasons = _synthetic_distinguishability_status_153(auc, bal_acc)
                elif mia_protocol == "temporal_split_real_holdout":
                    local_status, local_reasons = _temporal_drift_status_153(auc, bal_acc)
                else:
                    local_status, local_reasons = _mia_status_153(
                        max_auc=auc,
                        max_bal_acc=bal_acc,
                        carried_no_copy="",
                        carried_dcr="",
                    )

                protocol_metric_rows.append({
                    "role_group": role_name,
                    "mia_protocol": mia_protocol,
                    "model": model_name,
                    "risk_interpretation": (
                        "synthetic_distinguishability"
                        if mia_protocol == "random_split_synthetic_nonmember"
                        else "temporal_drift_calibration"
                    ),
                    "direct_privacy_blocker_candidate": bool(mia_protocol == "random_split_synthetic_nonmember"),
                    "local_mia_status": local_status,
                    "local_mia_reasons": local_reasons,
                    "selected_cols_n": int(len(cols)),
                    "member_source": member_source,
                    "nonmember_source": nonmember_source,
                    "member_n": int(X_member.shape[0]),
                    "nonmember_n": int(X_nonmember.shape[0]),
                    "classifier_train_n": int(len(y_train)),
                    "classifier_test_n": int(len(y_test)),
                    "test_auc": auc,
                    "test_average_precision": ap,
                    "test_accuracy": acc,
                    "test_balanced_accuracy": bal_acc,
                    "carried_no_copy_status": no_copy_status_map.get(role_name, ""),
                    "carried_dcr_nndr_status": dcr_status_map.get(role_name, ""),
                    "status": local_status,
                    "reasons": local_reasons,
                    "TEST_real_values_used_for_privacy_reference": True,
                    "synthetic_values_mutated": False,
                })
            except Exception as e:
                protocol_metric_rows.append({
                    "role_group": role_name,
                    "mia_protocol": mia_protocol,
                    "model": model_name,
                    "risk_interpretation": (
                        "synthetic_distinguishability"
                        if mia_protocol == "random_split_synthetic_nonmember"
                        else "temporal_drift_calibration"
                    ),
                    "direct_privacy_blocker_candidate": bool(mia_protocol == "random_split_synthetic_nonmember"),
                    "local_mia_status": "not_evaluable",
                    "local_mia_reasons": f"classifier_error:{type(e).__name__}:{str(e)[:200]}",
                    "selected_cols_n": int(len(cols)),
                    "member_source": member_source,
                    "nonmember_source": nonmember_source,
                    "member_n": int(X_member.shape[0]),
                    "nonmember_n": int(X_nonmember.shape[0]),
                    "classifier_train_n": 0,
                    "classifier_test_n": 0,
                    "test_auc": np.nan,
                    "test_average_precision": np.nan,
                    "test_accuracy": np.nan,
                    "test_balanced_accuracy": np.nan,
                    "carried_no_copy_status": no_copy_status_map.get(role_name, ""),
                    "carried_dcr_nndr_status": dcr_status_map.get(role_name, ""),
                    "status": "not_evaluable",
                    "reasons": f"classifier_error:{type(e).__name__}",
                    "TEST_real_values_used_for_privacy_reference": True,
                    "synthetic_values_mutated": False,
                })

protocol_metrics_df = pd.DataFrame(protocol_metric_rows)
role_config_df = pd.DataFrame(role_config_rows)

# ----------------------------------------------------------
# 6) Aggregate role status
# ----------------------------------------------------------
role_rows = []

for role_name, g in protocol_metrics_df.groupby("role_group", dropna=False):
    gg = g[g["status"].astype(str) != "not_evaluable"].copy()
    synth_g = gg[gg["mia_protocol"].astype(str) == "random_split_synthetic_nonmember"].copy()
    temporal_g = gg[gg["mia_protocol"].astype(str) == "temporal_split_real_holdout"].copy()

    if len(synth_g):
        synth_max_auc = float(pd.to_numeric(synth_g["test_auc"], errors="coerce").max())
        synth_max_bal_acc = float(pd.to_numeric(synth_g["test_balanced_accuracy"], errors="coerce").max())
        synth_max_ap = float(pd.to_numeric(synth_g["test_average_precision"], errors="coerce").max())
        synth_best_model = str(
            synth_g.sort_values(["test_auc", "test_balanced_accuracy"], ascending=[False, False])
            .iloc[0]["model"]
        )
        synth_status, synth_reasons = _synthetic_distinguishability_status_153(
            synth_max_auc,
            synth_max_bal_acc,
        )
        best_protocol = "random_split_synthetic_nonmember"
    else:
        synth_max_auc = np.nan
        synth_max_bal_acc = np.nan
        synth_max_ap = np.nan
        synth_best_model = ""
        synth_status = "not_evaluable"
        synth_reasons = "no_evaluable_synthetic_distinguishability_protocol"
        best_protocol = ""

    if len(temporal_g):
        temporal_max_auc = float(pd.to_numeric(temporal_g["test_auc"], errors="coerce").max())
        temporal_max_bal_acc = float(pd.to_numeric(temporal_g["test_balanced_accuracy"], errors="coerce").max())
        temporal_max_ap = float(pd.to_numeric(temporal_g["test_average_precision"], errors="coerce").max())
        temporal_best_model = str(
            temporal_g.sort_values(["test_auc", "test_balanced_accuracy"], ascending=[False, False])
            .iloc[0]["model"]
        )
        temporal_status, temporal_reasons = _temporal_drift_status_153(
            temporal_max_auc,
            temporal_max_bal_acc,
        )
    else:
        temporal_max_auc = np.nan
        temporal_max_bal_acc = np.nan
        temporal_max_ap = np.nan
        temporal_best_model = ""
        temporal_status = "not_evaluable"
        temporal_reasons = "no_evaluable_temporal_drift_protocol"

    carried_no_copy = no_copy_status_map.get(str(role_name), "")
    carried_dcr = dcr_status_map.get(str(role_name), "")
    carried_status, carried_reasons = _carried_privacy_status_153(carried_no_copy, carried_dcr)

    final_status, final_reasons = _combine_release_status_153(
        synth_status,
        temporal_status,
        carried_status,
    )

    carried_available = bool(str(carried_no_copy) or str(carried_dcr))
    if synth_status == "not_evaluable" and not carried_available:
        final_status = "not_evaluable"
        final_reasons = "synthetic_distinguishability_and_carried_privacy_not_evaluable"

    role_rows.append({
        "role_group": str(role_name),
        "synthetic_distinguishability_status": synth_status,
        "synthetic_distinguishability_reasons": synth_reasons,
        "synthetic_distinguishability_max_auc": synth_max_auc,
        "synthetic_distinguishability_max_balanced_accuracy": synth_max_bal_acc,
        "synthetic_distinguishability_max_average_precision": synth_max_ap,
        "synthetic_distinguishability_best_model": synth_best_model,
        "temporal_drift_status": temporal_status,
        "temporal_drift_reasons": temporal_reasons,
        "temporal_drift_max_auc": temporal_max_auc,
        "temporal_drift_max_balanced_accuracy": temporal_max_bal_acc,
        "temporal_drift_max_average_precision": temporal_max_ap,
        "temporal_drift_best_model": temporal_best_model,
        "carried_privacy_status": carried_status,
        "carried_privacy_reasons": carried_reasons,
        "final_mia_release_status": final_status,
        "final_mia_release_reasons": final_reasons,
        "status": final_status,
        "reasons": final_reasons,
        "max_mia_auc": synth_max_auc,
        "max_mia_balanced_accuracy": synth_max_bal_acc,
        "max_mia_average_precision": synth_max_ap,
        "best_mia_protocol": best_protocol,
        "best_model": synth_best_model,
        "carried_no_copy_status": carried_no_copy,
        "carried_dcr_nndr_status": carried_dcr,
        "local_mia_blocker_n": int((g["local_mia_status"].astype(str) == "blocker").sum()) if "local_mia_status" in g.columns else 0,
        "local_mia_warning_n": int((g["local_mia_status"].astype(str) == "warning").sum()) if "local_mia_status" in g.columns else 0,
        "protocol_rows_n": int(len(g)),
        "TEST_real_values_used_for_privacy_reference": True,
        "synthetic_values_mutated": False,
    })

role_metrics_df = pd.DataFrame(role_rows)

# ----------------------------------------------------------
# 7) Save outputs
# ----------------------------------------------------------
role_metrics_df.to_csv(role_metrics_csv, index=False)
protocol_metrics_df.to_csv(protocol_metrics_csv, index=False)
role_config_df.to_csv(role_config_csv, index=False)

status_counts = role_metrics_df["status"].astype(str).value_counts().sort_index().to_dict() if len(role_metrics_df) else {}
blocker_n = int((role_metrics_df["status"].astype(str) == "blocker").sum()) if len(role_metrics_df) else 0
warning_n = int((role_metrics_df["status"].astype(str) == "warning").sum()) if len(role_metrics_df) else 0
pass_n = int((role_metrics_df["status"].astype(str) == "pass").sum()) if len(role_metrics_df) else 0
not_eval_n = int((role_metrics_df["status"].astype(str) == "not_evaluable").sum()) if len(role_metrics_df) else 0
synthetic_distinguishability_status_counts = (
    role_metrics_df["synthetic_distinguishability_status"].astype(str).value_counts().sort_index().to_dict()
    if len(role_metrics_df) and "synthetic_distinguishability_status" in role_metrics_df.columns
    else {}
)
temporal_drift_status_counts = (
    role_metrics_df["temporal_drift_status"].astype(str).value_counts().sort_index().to_dict()
    if len(role_metrics_df) and "temporal_drift_status" in role_metrics_df.columns
    else {}
)
carried_privacy_status_counts = (
    role_metrics_df["carried_privacy_status"].astype(str).value_counts().sort_index().to_dict()
    if len(role_metrics_df) and "carried_privacy_status" in role_metrics_df.columns
    else {}
)
synthetic_distinguishability_blocker_n = (
    int((role_metrics_df["synthetic_distinguishability_status"].astype(str) == "blocker").sum())
    if len(role_metrics_df) and "synthetic_distinguishability_status" in role_metrics_df.columns
    else 0
)
temporal_drift_warning_n = (
    int((role_metrics_df["temporal_drift_status"].astype(str) == "warning").sum())
    if len(role_metrics_df) and "temporal_drift_status" in role_metrics_df.columns
    else 0
)
carried_privacy_blocker_n = (
    int((role_metrics_df["carried_privacy_status"].astype(str) == "blocker").sum())
    if len(role_metrics_df) and "carried_privacy_status" in role_metrics_df.columns
    else 0
)

# ----------------------------------------------------------
# 8) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "15.3",
    "version": CELL153_VERSION,
    "role": "q6_mia_random_temporal_privacy_audit",
    "quality_dimension": "Q6_privacy_no_copy_release_safety",
    "upstream_contract_versions": {
        "cell15_0": version150_153,
        "cell15_1": version151_153,
        "cell15_1a": version151a_153,
        "cell15_2": version152_153,
        "cell15_2a": version152a_153
    },
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False
    },
    "carried_privacy_forensics": {
        "no_copy_blocker_roles": no_copy_blocker_roles_153,
        "dcr_nndr_blocker_roles": dcr_blocker_roles_153,
        "dcr_nndr_warning_roles": dcr_warning_roles_153,
        "combined_carried_blocker_roles": combined_carried_blocker_roles_153,
        "combined_carried_warning_roles": combined_carried_warning_roles_153,
        "note": "MIA adds evidence but does not clear carried no-copy or DCR/NNDR blockers."
    },
    "summary": {
        "role_groups_total": int(len(role_metrics_df)),
        "pass_n": pass_n,
        "warning_n": warning_n,
        "blocker_n": blocker_n,
        "not_evaluable_n": not_eval_n,
        "status_counts": status_counts,
        "synthetic_distinguishability_status_counts": synthetic_distinguishability_status_counts,
        "temporal_drift_status_counts": temporal_drift_status_counts,
        "carried_privacy_status_counts": carried_privacy_status_counts,
        "synthetic_distinguishability_blocker_n": int(synthetic_distinguishability_blocker_n),
        "temporal_drift_warning_n": int(temporal_drift_warning_n),
        "carried_privacy_blocker_n": int(carried_privacy_blocker_n),
        "temporal_drift_treated_as_release_blocker": bool(
            CFG.get("cell15_3_treat_temporal_drift_as_release_blocker", False)
        ),
        "synthetic_distinguishability_treated_as_release_blocker": bool(
            CFG.get("cell15_3_treat_synthetic_distinguishability_as_release_blocker", True)
        ),
        "protocol_rows_n": int(len(protocol_metrics_df)),
        "models": MODELS_153,
        "protocols": [p["mia_protocol"] for p in protocols_153],
    },
    "method": {
        "random_split_synthetic_nonmember": (
            "Synthetic-vs-real distinguishability audit: classifies real member-like rows "
            "from synthetic non-member rows. This is a distributional and release-risk smoke test."
        ),
        "temporal_split_real_holdout": (
            "Temporal drift calibration: classifies earlier real member source against later "
            "real holdout source to estimate temporal separability and household routine drift. "
            "It is not treated as a release blocker unless "
            "cell15_3_treat_temporal_drift_as_release_blocker=True."
        ),
        "carried_privacy_risk": "Carries no-copy and DCR/NNDR privacy statuses from Cells 15.1 and 15.2.",
        "release_status_policy": (
            "Role release status combines synthetic distinguishability, carried privacy status, "
            "and temporal drift context without treating temporal split separability as direct "
            "privacy leakage by default."
        ),
        "privacy_scope": "Release-risk evidence, not a formal privacy proof.",
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "prior_privacy_forensics_carried_forward": True,
        "MIA_does_not_clear_no_copy_or_DCR_NNDR_blockers": True,
        "TEST_real_values_used_for_privacy_reference": True,
        "TEST_real_values_used_for_model_fitting": False,
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "materialization_done_here": False,
        "privacy_decision_done_here": False,
        "mia_audit_done_here": True,
    },
    "outputs": {
        "role_metrics_csv": role_metrics_csv,
        "protocol_metrics_csv": protocol_metrics_csv,
        "role_config_csv": role_config_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_153(contract_json, contract)
_write_json_153(contract_canonical_json, contract)

manifest = {
    "cell": "15.3",
    "version": CELL153_VERSION,
    "created_outputs": contract["outputs"],
    "summary": contract["summary"],
    "q4_governance": contract["q4_governance"],
    "carried_privacy_forensics": contract["carried_privacy_forensics"],
    "strict_contract": contract["strict_contract"],
}

_write_json_153(manifest_json, manifest)

hashes = {
    "role_metrics_csv_sha256": _sha256_file_153(role_metrics_csv),
    "protocol_metrics_csv_sha256": _sha256_file_153(protocol_metrics_csv),
    "role_config_csv_sha256": _sha256_file_153(role_config_csv),
    "contract_json_sha256": _sha256_file_153(contract_json),
    "contract_canonical_json_sha256": _sha256_file_153(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_153(manifest_json),
}

contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_153(contract_json, contract)
_write_json_153(contract_canonical_json, contract)
_write_json_153(manifest_json, manifest)

# ----------------------------------------------------------
# 9) Export globals
# ----------------------------------------------------------
globals()["CELL153_VERSION"] = CELL153_VERSION
globals()["CELL15_3_MIA_ROLE_METRICS_DF"] = role_metrics_df
globals()["CELL15_3_MIA_PROTOCOL_METRICS_DF"] = protocol_metrics_df
globals()["CELL15_3_MIA_ROLE_CONFIG_DF"] = role_config_df
globals()["CELL15_3_MIA_CONTRACT"] = contract

globals()["CELL15_3_MIA_ROLE_METRICS_CSV"] = role_metrics_csv
globals()["CELL15_3_MIA_PROTOCOL_METRICS_CSV"] = protocol_metrics_csv
globals()["CELL15_3_MIA_ROLE_CONFIG_CSV"] = role_config_csv
globals()["CELL15_3_MIA_CONTRACT_JSON"] = contract_json
globals()["CELL15_3_MIA_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL15_3_MIA_MANIFEST_JSON"] = manifest_json

log(
    "[Cell15.3] Q6 MIA privacy audit complete | "
    f"roles={len(role_metrics_df)} | pass={pass_n} | warning={warning_n} | "
    f"blocker={blocker_n} | not_evaluable={not_eval_n}"
)
log(f"[Cell15.3] Status counts | {status_counts}")
log(f"[Cell15.3] Synthetic distinguishability status counts | {synthetic_distinguishability_status_counts}")
log(f"[Cell15.3] Temporal drift status counts | {temporal_drift_status_counts}")
log(f"[Cell15.3] Carried privacy status counts | {carried_privacy_status_counts}")
log(f"[Cell15.3] Saved role metrics: {role_metrics_csv} | rows={len(role_metrics_df)}")
log(f"[Cell15.3] Saved protocol metrics: {protocol_metrics_csv} | rows={len(protocol_metrics_df)}")
log(f"[Cell15.3] Saved canonical contract: {contract_canonical_json}")
log(f"[Cell15.3] Carried privacy forensic blocker roles | {combined_carried_blocker_roles_153}")
log(
    "[Cell15.3] Contract flags | "
    "TEST_real_values_used_for_privacy_reference=True | "
    "TEST_real_values_used_for_model_fitting=False | "
    "synthetic_values_mutated=False | "
    "selection_done_here=False | "
    "generator_fit_done_here=False | "
    "materialization_done_here=False | "
    "privacy_decision_done_here=False | "
    "mia_audit_done_here=True | Q4_final_status=blocked_no_promotion | Q4_coupled_artifacts_used=False | prior_privacy_forensics_carried_forward=True"
)
log("--- END: Cell 15.3 - Q6 MIA audit random + temporal split (v1.2 DCR/NNDR-forensic-aware separated-risk) ---")

gc.collect()
