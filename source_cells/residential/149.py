# %% CELL 17.1R — CTGAN external baseline, kernel-safe subprocess version
# Purpose:
#   Replace the original in-process CTGAN baseline cell.
#
# Why this replacement exists:
#   CTGAN/PyTorch can kill the Jupyter kernel via CUDA/OOM/low-level runtime errors.
#   This cell runs each CTGAN scope in a separate subprocess. If CTGAN crashes,
#   times out, or is killed by the OS, the notebook kernel survives and records
#   the baseline as failed/unavailable.
#
# Scientific contract:
#   - TRAIN real values are used for CTGAN fitting.
#   - VAL real values are NOT used for model selection here.
#   - TEST real values are NOT used for fitting, selection, or materialization.
#   - TEST length is used only to request a comparable number of synthetic rows.
#   - No Q4-coupled artifact is used.
#   - No fallback model is silently substituted and called CTGAN.
#   - Failed/unavailable CTGAN scopes are recorded honestly.

log("--- START: Cell 17.1R - CTGAN external baseline, kernel-safe subprocess ---")

import os
import sys
import gc
import json
import time
import shutil
import hashlib
import importlib.util
import subprocess
import textwrap
import traceback
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# 0) Required globals
# ----------------------------------------------------------
_required_171r = [
    "CFG", "log",
    "df_tr", "df_val", "df_te",
    "OUTDIR", "OUT_SYN", "REPORT_DIR",
    "SEED",
    "CELL17_0_BASELINE_SCOPE_REGISTRY_DF",
    "CELL17_0_BASELINE_COLUMN_REGISTRY_DF",
    "CELL17_0_BASELINE_FAIRNESS_CONTRACT",
    "CELL17_0_BASELINE_PLAN",
]
_missing_171r = [k for k in _required_171r if k not in globals()]
if _missing_171r:
    raise RuntimeError(f"[Cell17.1R] Missing required globals: {_missing_171r}")

# IMPORTANT: never resolve/fallback to old FULL_COUPLED roots in a final STUDY notebook.
OUTDIR_171R = Path(str(OUTDIR)).expanduser().resolve()
OUT_SYN_171R = Path(str(OUT_SYN)).expanduser().resolve()
REPORT_DIR_171R = Path(str(REPORT_DIR)).expanduser().resolve()
ARTDIR_171R = Path(str(globals().get("ARTIFACT_DIR", globals().get("ARTDIR", OUTDIR_171R / "artifacts")))).expanduser().resolve()
CONTRACT_DIR_171R = Path(str(globals().get("CONTRACT_DIR", ARTDIR_171R / "contracts"))).expanduser().resolve()

BASELINE_DIR_171R = OUT_SYN_171R / "baselines"
BASELINE_META_DIR_171R = ARTDIR_171R / "baselines"
BASELINE_WORK_DIR_171R = ARTDIR_171R / "cell17_1_ctgan_subprocess_work"

for p in [BASELINE_DIR_171R, BASELINE_META_DIR_171R, BASELINE_WORK_DIR_171R, REPORT_DIR_171R, ARTDIR_171R, CONTRACT_DIR_171R]:
    p.mkdir(parents=True, exist_ok=True)

SEED = int(SEED)
N_TR = int(len(df_tr))
N_VAL = int(len(df_val))
N_TE = int(len(df_te))

CELL171_VERSION = "cell17_1R_ctgan_external_baseline_subprocess_v1_2_THESIS"

# ----------------------------------------------------------
# 1) Config — conservative defaults for notebook stability
# ----------------------------------------------------------
CFG["cell17_1_version"] = CELL171_VERSION
CFG["cell17_1_Q4_final_status"] = "blocked_no_promotion"
CFG["cell17_1_Q4_coupled_artifacts_used"] = False
CFG["cell17_1_train_values_used_for_fitting"] = True
CFG["cell17_1_val_values_used_for_selection"] = False
CFG["cell17_1_TEST_real_values_used"] = False
CFG["cell17_1_TEST_real_values_used_for_materialization"] = False
CFG["cell17_1_pipeline_synthetic_values_mutated"] = False
CFG["cell17_1_external_baseline_values_generated"] = True
CFG["cell17_1_selection_done_here"] = False
CFG["cell17_1_generator_fit_done_here"] = True
CFG["cell17_1_materialization_done_here"] = True
CFG["cell17_1_kernel_safe_subprocess"] = True

# The original cell used large in-process CTGAN settings that can kill a kernel.
# For a final artifact, keep these declared. For a stronger CTGAN-only experiment,
# run a separate batch job and point downstream to its artifacts.
CFG.setdefault("cell17_1_ctgan_enabled", True)
CFG.setdefault("cell17_1_ctgan_train_row_cap", 20000)
CFG.setdefault("cell17_1_ctgan_epochs", 20)
CFG.setdefault("cell17_1_ctgan_batch_size", 500)
CFG.setdefault("cell17_1_ctgan_embedding_dim", 128)
CFG.setdefault("cell17_1_ctgan_generator_dim", (256, 256))
CFG.setdefault("cell17_1_ctgan_discriminator_dim", (256, 256))
CFG.setdefault("cell17_1_ctgan_verbose", False)
CFG.setdefault("cell17_1_ctgan_cuda", False)  # CPU default prevents CUDA OOM from killing the notebook.
CFG.setdefault("cell17_1_integer_unique_threshold", 64)
CFG.setdefault("cell17_1_discrete_unique_threshold", 20)
CFG.setdefault("cell17_1_clip_to_train_quantiles", True)
CFG.setdefault("cell17_1_clip_q_low", 0.001)
CFG.setdefault("cell17_1_clip_q_high", 0.999)
CFG.setdefault("cell17_1_fail_if_ctgan_unavailable", False)
CFG.setdefault("cell17_1_fail_on_scope_failure", False)
CFG.setdefault("cell17_1_subprocess_timeout_sec", 7200)
CFG.setdefault("cell17_1_subprocess_threads", 2)
CFG.setdefault("cell17_1_max_scope_cols", 160)
CFG.setdefault("cell17_1_cleanup_scope_workdir", True)

TRAIN_ROW_CAP_171 = int(CFG.get("cell17_1_ctgan_train_row_cap", 20000))
CTGAN_EPOCHS_171 = int(CFG.get("cell17_1_ctgan_epochs", 20))
BATCH_SIZE_171 = int(CFG.get("cell17_1_ctgan_batch_size", 500))
INTEGER_UNIQUE_THRESHOLD_171 = int(CFG.get("cell17_1_integer_unique_threshold", 64))
DISCRETE_UNIQUE_THRESHOLD_171 = int(CFG.get("cell17_1_discrete_unique_threshold", 20))
SUBPROCESS_TIMEOUT_171 = int(CFG.get("cell17_1_subprocess_timeout_sec", 7200))
MAX_SCOPE_COLS_171 = int(CFG.get("cell17_1_max_scope_cols", 160))

# ----------------------------------------------------------
# 2) Output paths
# ----------------------------------------------------------
run_audit_csv = str(REPORT_DIR_171R / "cell17_1_ctgan_baseline_run_audit.csv")
artifact_registry_csv = str(REPORT_DIR_171R / "cell17_1_ctgan_baseline_artifact_registry.csv")
contract_json = str(REPORT_DIR_171R / "cell17_1_ctgan_baseline_contract.json")
contract_canonical_json = str(CONTRACT_DIR_171R / "cell17_1_ctgan_baseline_contract_v1_2_THESIS.json")
manifest_json = str(ARTDIR_171R / "cell17_1_ctgan_baseline_manifest.json")
worker_py = BASELINE_WORK_DIR_171R / "cell17_1_ctgan_worker.py"

# ----------------------------------------------------------
# 3) Helpers
# ----------------------------------------------------------
def _json_sanitize_171(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize_171(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize_171(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize_171(v) for v in obj]
    if isinstance(obj, set):
        return sorted([_json_sanitize_171(v) for v in obj])
    if isinstance(obj, np.ndarray):
        return _json_sanitize_171(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _json_sanitize_171(obj.to_dict("records"))
    if isinstance(obj, pd.Series):
        return _json_sanitize_171(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

def _write_json_171(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize_171(payload), f, indent=2, sort_keys=True)

def _sha256_file_171(path: str) -> str:
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_float_171(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else float(default)
    except Exception:
        return float(default)

def _slug_171(x: str) -> str:
    s = str(x)
    keep = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")[:120]

def _require_17_0_governance_171(contract: dict):
    if not isinstance(contract, dict):
        raise RuntimeError("[Cell17.1R] CELL17_0_BASELINE_FAIRNESS_CONTRACT is not a dict.")
    version = str(contract.get("version", ""))
    if "cell17_0_external_baseline_scope_contract_v1_1" not in version:
        raise RuntimeError(f"[Cell17.1R] Unexpected Cell17.0 contract version: {version}")
    q4 = contract.get("q4_governance", {})
    strict = contract.get("strict_contract", {})
    if str(q4.get("final_q4_status", strict.get("Q4_final_status", ""))) != "blocked_no_promotion":
        raise RuntimeError("[Cell17.1R] Cell17.0 does not carry Q4 blocked_no_promotion governance.")
    if bool(q4.get("q4_coupled_artifacts_used", strict.get("Q4_coupled_artifacts_used", True))):
        raise RuntimeError("[Cell17.1R] Cell17.0 indicates Q4-coupled artifacts were used.")
    return version

def _scope_cols_171(scope_id: str):
    reg = CELL17_0_BASELINE_COLUMN_REGISTRY_DF.copy()
    reg["scope_id"] = reg["scope_id"].astype(str)
    reg["col"] = reg["col"].astype(str)
    cols = reg.loc[reg["scope_id"].eq(str(scope_id)), "col"].tolist()
    return list(dict.fromkeys(cols))

def _numeric_train_frame_171(cols):
    cols = [str(c) for c in cols if str(c) in df_tr.columns]
    if not cols:
        return pd.DataFrame()
    d = df_tr[cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def _fit_impute_metadata_171(d: pd.DataFrame):
    meta = {}
    out = d.copy()

    for c in out.columns.astype(str):
        x = pd.to_numeric(out[c], errors="coerce")
        finite = x[np.isfinite(x)]

        if len(finite) == 0:
            fill = 0.0
            lo = 0.0
            hi = 0.0
            is_integer_like = True
            is_nonnegative = True
            discrete = True
            unique_values = []
        else:
            fill = float(np.nanmedian(finite))
            qlo = float(np.nanquantile(finite, float(CFG.get("cell17_1_clip_q_low", 0.001))))
            qhi = float(np.nanquantile(finite, float(CFG.get("cell17_1_clip_q_high", 0.999))))
            lo, hi = qlo, qhi

            rounded = np.isclose(finite, np.rint(finite), atol=1e-8)
            is_integer_like = bool(np.mean(rounded) >= 0.999)
            is_nonnegative = bool(np.nanmin(finite) >= 0.0)
            unique = np.unique(finite)
            discrete = bool(len(unique) <= DISCRETE_UNIQUE_THRESHOLD_171)
            unique_values = unique.tolist() if len(unique) <= INTEGER_UNIQUE_THRESHOLD_171 else []

        arr = x.fillna(fill).to_numpy(dtype=np.float64)
        arr[~np.isfinite(arr)] = fill
        out[c] = arr

        meta[c] = {
            "fill": fill,
            "clip_lo": lo,
            "clip_hi": hi,
            "is_integer_like": is_integer_like,
            "is_nonnegative": is_nonnegative,
            "discrete": discrete,
            "unique_values": unique_values,
            "train_min": float(np.nanmin(finite)) if len(finite) else np.nan,
            "train_max": float(np.nanmax(finite)) if len(finite) else np.nan,
            "train_mean": float(np.nanmean(finite)) if len(finite) else np.nan,
            "train_std": float(np.nanstd(finite)) if len(finite) else np.nan,
        }

    return out, meta

def _postprocess_sample_171(sample: pd.DataFrame, meta: dict, ordered_cols: list):
    out = sample.copy()

    for c in ordered_cols:
        if c not in out.columns:
            out[c] = meta.get(c, {}).get("fill", 0.0)

    out = out[ordered_cols].copy()

    for c in ordered_cols:
        m = meta.get(c, {})
        x = pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=np.float64)
        fill = _safe_float_171(m.get("fill"), 0.0)
        x = np.where(np.isfinite(x), x, fill)

        if bool(CFG.get("cell17_1_clip_to_train_quantiles", True)):
            lo = _safe_float_171(m.get("clip_lo"), np.nan)
            hi = _safe_float_171(m.get("clip_hi"), np.nan)
            if np.isfinite(lo) and np.isfinite(hi) and hi >= lo:
                x = np.clip(x, lo, hi)

        if bool(m.get("is_nonnegative", False)):
            x = np.maximum(x, 0.0)

        if bool(m.get("is_integer_like", False)):
            x = np.rint(x)

        out[c] = x.astype(np.float32)

    return out

def _deterministic_train_sample_171(d: pd.DataFrame, row_cap: int, seed: int):
    if len(d) <= int(row_cap):
        return d.reset_index(drop=True).copy()
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(np.arange(len(d)), size=int(row_cap), replace=False)
    idx = np.sort(idx)
    return d.iloc[idx].reset_index(drop=True).copy()

def _ctgan_standalone_available_171():
    return importlib.util.find_spec("ctgan") is not None

# ----------------------------------------------------------
# 4) Worker script
# ----------------------------------------------------------
worker_code = r"""
import os
import sys
import json
import traceback
from pathlib import Path

import pandas as pd

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    import random
    import numpy as np
    import torch
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result_path = cfg["result_json"]
    try:
        from ctgan import CTGAN
    except Exception as e:
        write_json(result_path, {
            "status": "failed",
            "reason": "ctgan_import_failed",
            "error": repr(e),
        })
        return 20

    try:
        train = pd.read_parquet(cfg["train_parquet"])
        discrete_columns = cfg.get("discrete_columns", [])
        n_rows = int(cfg["n_rows"])
        output_parquet = cfg["output_parquet"]

        kwargs = {
            "epochs": int(cfg.get("epochs", 20)),
            "batch_size": int(cfg.get("batch_size", 500)),
            "embedding_dim": int(cfg.get("embedding_dim", 128)),
            "generator_dim": tuple(cfg.get("generator_dim", [256, 256])),
            "discriminator_dim": tuple(cfg.get("discriminator_dim", [256, 256])),
            "verbose": bool(cfg.get("verbose", False)),
            "cuda": bool(cfg.get("cuda", False)),
        }

        model = CTGAN(**kwargs)
        used_kwargs = kwargs
        model.set_random_state(seed)
        model.fit(train, discrete_columns=discrete_columns)

        if cfg.get("encoding_policy") == "ctgan_numeric_counts_v1":
            model.save(str(Path(output_parquet).parent / "fitted_ctgan.pkl"))

        sample = model.sample(n_rows)

        if not isinstance(sample, pd.DataFrame):
            sample = pd.DataFrame(sample, columns=list(train.columns))

        Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
        sample.to_parquet(output_parquet, index=False)

        write_json(result_path, {
            "status": "success",
            "reason": "ctgan_fit_sample_completed",
            "rows": int(len(sample)),
            "cols": int(sample.shape[1]),
            "used_kwargs": used_kwargs,
            "seed": seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        })
        return 0

    except BaseException as e:
        write_json(result_path, {
            "status": "failed",
            "reason": "ctgan_worker_exception",
            "error": "".join(traceback.format_exception_only(type(e), e)).strip(),
            "traceback_tail": traceback.format_exc()[-4000:],
        })
        return 30

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
"""
worker_py.write_text(worker_code, encoding="utf-8")

# ----------------------------------------------------------
# 5) Validate Cell 17.0 fairness governance
# ----------------------------------------------------------
CELL17_0_VERSION_171 = _require_17_0_governance_171(CELL17_0_BASELINE_FAIRNESS_CONTRACT)

# ----------------------------------------------------------
# 6) Determine eligible CTGAN scopes
# ----------------------------------------------------------
plan = CELL17_0_BASELINE_PLAN.get("CTGAN", {})
allowed_scopes = list(plan.get("allowed_scopes", []))

scope_registry = CELL17_0_BASELINE_SCOPE_REGISTRY_DF.copy()
scope_registry["scope_id"] = scope_registry["scope_id"].astype(str)

eligible_scopes = []
for sid in allowed_scopes:
    d = scope_registry[scope_registry["scope_id"].eq(str(sid))]
    if len(d) == 0:
        continue
    cols = _scope_cols_171(str(sid))
    if len(cols) == 0:
        continue
    eligible_scopes.append(str(sid))

if not eligible_scopes:
    raise RuntimeError("[Cell17.1R] No eligible CTGAN scopes found from Cell 17.0.")

ctgan_enabled = bool(CFG.get("cell17_1_ctgan_enabled", True))
ctgan_available = _ctgan_standalone_available_171() if ctgan_enabled else False
ctgan_backend = "ctgan.CTGAN" if ctgan_available else ("disabled_by_CFG" if not ctgan_enabled else "unavailable")

if not ctgan_available:
    msg = (
        "[Cell17.1R] Standalone ctgan package is unavailable or disabled. "
        "No CTGAN baseline will be generated; this is recorded honestly."
    )
    log(msg)
    if bool(CFG.get("cell17_1_fail_if_ctgan_unavailable", False)):
        raise RuntimeError(msg)

# ----------------------------------------------------------
# 7) Train/generate per scope via subprocess
# ----------------------------------------------------------
run_rows = []
artifact_rows = []

log(
    "[Cell17.1R] Running CTGAN baseline | "
    f"available={ctgan_available} | backend={ctgan_backend} | scopes={eligible_scopes} | "
    f"subprocess_timeout_sec={SUBPROCESS_TIMEOUT_171} | cuda={bool(CFG.get('cell17_1_ctgan_cuda', False))}"
)

for scope_i, scope_id in enumerate(eligible_scopes, start=1):
    cols = _scope_cols_171(scope_id)
    cols = [c for c in cols if c in df_tr.columns]

    scope_slug = _slug_171(scope_id)
    out_path = str(BASELINE_DIR_171R / f"ctgan_{scope_slug}_synthetic_test.parquet")
    meta_path = str(BASELINE_META_DIR_171R / f"cell17_1_ctgan_{scope_slug}_metadata.json")
    work_dir = BASELINE_WORK_DIR_171R / scope_slug
    work_dir.mkdir(parents=True, exist_ok=True)

    train_parquet = work_dir / "train_fit.parquet"
    raw_sample_parquet = work_dir / "raw_sample.parquet"
    worker_config_json = work_dir / "worker_config.json"
    worker_result_json = work_dir / "worker_result.json"
    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"

    log(f"[Cell17.1R] scope {scope_i}/{len(eligible_scopes)} | {scope_id} | cols={len(cols)}")

    row_base = {
        "baseline": "CTGAN",
        "scope_id": scope_id,
        "backend": ctgan_backend,
        "ctgan_available": bool(ctgan_available),
        "cols_n": int(len(cols)),
        "train_rows_available": int(len(df_tr)),
        "train_row_cap": int(TRAIN_ROW_CAP_171),
        "generated_rows_target": int(N_TE),
        "output_path": out_path,
        "metadata_path": meta_path,
        "TRAIN_real_values_used_for_fitting": True,
        "VAL_real_values_used_for_selection": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "pipeline_synthetic_values_mutated": False,
        "external_baseline_values_generated": bool(ctgan_available),
        "kernel_safe_subprocess": True,
    }

    if not ctgan_available:
        run_rows.append({
            **row_base,
            "status": "not_run",
            "reason": "ctgan_backend_unavailable_or_disabled",
            "train_rows_used": 0,
            "generated_rows": 0,
            "fit_succeeded": False,
            "sample_succeeded": False,
            "discrete_cols_n": 0,
            "subprocess_returncode": None,
            "error": "",
        })
        artifact_rows.append({
            "baseline": "CTGAN",
            "scope_id": scope_id,
            "path": out_path,
            "exists": False,
            "rows": 0,
            "cols": int(len(cols)),
            "sha256": "",
            "metadata_path": meta_path,
            "metadata_sha256": "",
            "status": "not_run",
            "reason": "ctgan_backend_unavailable_or_disabled",
        })
        continue

    if len(cols) > MAX_SCOPE_COLS_171:
        reason = f"scope_too_wide_for_kernel_safe_ctgan: cols={len(cols)} max={MAX_SCOPE_COLS_171}"
        log(f"[Cell17.1R] WARNING: {reason}")
        run_rows.append({
            **row_base,
            "status": "skipped",
            "reason": reason,
            "train_rows_used": 0,
            "generated_rows": 0,
            "fit_succeeded": False,
            "sample_succeeded": False,
            "discrete_cols_n": 0,
            "subprocess_returncode": None,
            "error": reason,
        })
        artifact_rows.append({
            "baseline": "CTGAN",
            "scope_id": scope_id,
            "path": out_path,
            "exists": False,
            "rows": 0,
            "cols": int(len(cols)),
            "sha256": "",
            "metadata_path": meta_path,
            "metadata_sha256": "",
            "status": "skipped",
            "reason": reason,
        })
        continue

    try:
        train_raw = _numeric_train_frame_171(cols)
        if train_raw.empty or train_raw.shape[1] == 0:
            raise RuntimeError("empty_training_frame_after_scope_column_resolution")

        train_imp, meta = _fit_impute_metadata_171(train_raw)
        train_fit = _deterministic_train_sample_171(
            train_imp,
            row_cap=TRAIN_ROW_CAP_171,
            seed=SEED + scope_i * 1009,
        )

        discrete_columns = [
            c for c, m in meta.items()
            if bool(m.get("discrete", False)) or bool(m.get("is_integer_like", False))
        ]

        if scope_id in {"protocol_core_tabular", "public_candidate_core_tabular"}:
            from ctgan_encoding import classify
            from workflow import ROOT as STUDY_ROOT
            policy = json.loads((STUDY_ROOT / "design_inputs/ctgan_encoding_policy.json").read_text())
            discrete_columns, encoding_rows = classify(train_fit, train_raw, policy)
            pd.DataFrame(encoding_rows).to_csv(work_dir / "encoding_manifest.csv", index=False)

        train_fit.to_parquet(train_parquet, index=False)

        worker_cfg = {
            "encoding_policy": "ctgan_numeric_counts_v1" if scope_id in {"protocol_core_tabular", "public_candidate_core_tabular"} else "original_compact_iot",
            "train_parquet": str(train_parquet),
            "output_parquet": str(raw_sample_parquet),
            "result_json": str(worker_result_json),
            "discrete_columns": discrete_columns,
            "n_rows": int(N_TE),
            "seed": int(SEED + scope_i * 1009),
            "epochs": int(CTGAN_EPOCHS_171),
            "batch_size": int(BATCH_SIZE_171),
            "embedding_dim": int(CFG.get("cell17_1_ctgan_embedding_dim", 128)),
            "generator_dim": list(CFG.get("cell17_1_ctgan_generator_dim", (256, 256))),
            "discriminator_dim": list(CFG.get("cell17_1_ctgan_discriminator_dim", (256, 256))),
            "verbose": bool(CFG.get("cell17_1_ctgan_verbose", False)),
            "cuda": bool(CFG.get("cell17_1_ctgan_cuda", False)),
        }
        _write_json_171(str(worker_config_json), worker_cfg)

        env = os.environ.copy()
        threads = str(int(CFG.get("cell17_1_subprocess_threads", 2)))
        env["OMP_NUM_THREADS"] = threads
        env["MKL_NUM_THREADS"] = threads
        env["OPENBLAS_NUM_THREADS"] = threads
        env["NUMEXPR_NUM_THREADS"] = threads
        if not bool(CFG.get("cell17_1_ctgan_cuda", False)):
            env["CUDA_VISIBLE_DEVICES"] = ""

        start = time.time()
        with open(stdout_path, "w", encoding="utf-8") as out_f, open(stderr_path, "w", encoding="utf-8") as err_f:
            proc = subprocess.run(
                [sys.executable, str(worker_py), str(worker_config_json)],
                stdout=out_f,
                stderr=err_f,
                timeout=SUBPROCESS_TIMEOUT_171,
                env=env,
                cwd=str(work_dir),
            )
        elapsed = float(time.time() - start)

        worker_result = {}
        if worker_result_json.exists():
            try:
                worker_result = json.loads(worker_result_json.read_text(encoding="utf-8"))
            except Exception:
                worker_result = {}

        if proc.returncode != 0:
            reason = worker_result.get("reason", f"subprocess_returncode_{proc.returncode}")
            err = worker_result.get("error", "")
            if proc.returncode < 0:
                reason = f"subprocess_killed_signal_{abs(proc.returncode)}"
            raise RuntimeError(f"{reason}; returncode={proc.returncode}; error={err}")

        if not raw_sample_parquet.exists():
            raise RuntimeError("subprocess_success_but_raw_sample_missing")

        sample_raw = pd.read_parquet(raw_sample_parquet)
        if not isinstance(sample_raw, pd.DataFrame):
            sample_raw = pd.DataFrame(sample_raw, columns=list(train_fit.columns))

        sample_pp = _postprocess_sample_171(
            sample=sample_raw,
            meta=meta,
            ordered_cols=list(train_fit.columns),
        )

        if len(sample_pp) != N_TE:
            raise RuntimeError(f"generated_row_mismatch: got={len(sample_pp)} expected={N_TE}")

        sample_pp.to_parquet(out_path, index=False)

        meta_payload = {
            "cell": "17.1R",
            "version": CELL171_VERSION,
            "baseline": "CTGAN",
            "scope_id": scope_id,
            "backend": ctgan_backend,
            "worker_result": worker_result,
            "worker_elapsed_sec": elapsed,
            "columns": list(train_fit.columns),
            "discrete_columns": discrete_columns,
            "train_rows_used": int(len(train_fit)),
            "generated_rows": int(len(sample_pp)),
            "column_metadata": meta,
            "fairness_scope": scope_registry.loc[scope_registry["scope_id"].eq(scope_id)].to_dict("records"),
            "q4_governance": {
                "final_q4_status": "blocked_no_promotion",
                "q4_coupled_artifacts_used": False,
            },
            "strict_contract": {
                "TRAIN_real_values_used_for_fitting": True,
                "VAL_real_values_used_for_selection": False,
                "TEST_real_values_used": False,
                "TEST_real_values_used_for_materialization": False,
                "pipeline_synthetic_values_mutated": False,
                "external_baseline_values_generated": True,
                "selection_done_here": False,
                "kernel_safe_subprocess": True,
            },
        }
        _write_json_171(meta_path, meta_payload)

        run_rows.append({
            **row_base,
            "status": "success",
            "reason": "ctgan_subprocess_fit_sample_completed",
            "train_rows_used": int(len(train_fit)),
            "generated_rows": int(len(sample_pp)),
            "fit_succeeded": True,
            "sample_succeeded": True,
            "discrete_cols_n": int(len(discrete_columns)),
            "subprocess_returncode": int(proc.returncode),
            "elapsed_sec": elapsed,
            "error": "",
        })

        artifact_rows.append({
            "baseline": "CTGAN",
            "scope_id": scope_id,
            "path": out_path,
            "exists": True,
            "rows": int(len(sample_pp)),
            "cols": int(sample_pp.shape[1]),
            "sha256": _sha256_file_171(out_path),
            "metadata_path": meta_path,
            "metadata_sha256": _sha256_file_171(meta_path),
            "status": "success",
            "reason": "ctgan_subprocess_fit_sample_completed",
        })

        del train_raw, train_imp, train_fit, sample_raw, sample_pp
        gc.collect()

        if bool(CFG.get("cell17_1_cleanup_scope_workdir", True)):
            shutil.rmtree(work_dir, ignore_errors=True)

    except subprocess.TimeoutExpired:
        err = f"ctgan_subprocess_timeout_after_{SUBPROCESS_TIMEOUT_171}_sec"
        log(f"[Cell17.1R] WARNING: CTGAN scope timeout | scope={scope_id} | {err}")

        run_rows.append({
            **row_base,
            "status": "failed",
            "reason": "ctgan_subprocess_timeout",
            "train_rows_used": 0,
            "generated_rows": 0,
            "fit_succeeded": False,
            "sample_succeeded": False,
            "discrete_cols_n": 0,
            "subprocess_returncode": None,
            "elapsed_sec": float(SUBPROCESS_TIMEOUT_171),
            "error": err,
        })
        artifact_rows.append({
            "baseline": "CTGAN",
            "scope_id": scope_id,
            "path": out_path,
            "exists": False,
            "rows": 0,
            "cols": int(len(cols)),
            "sha256": "",
            "metadata_path": meta_path,
            "metadata_sha256": "",
            "status": "failed",
            "reason": err,
        })
        if bool(CFG.get("cell17_1_fail_on_scope_failure", False)):
            raise

    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        log(f"[Cell17.1R] WARNING: CTGAN scope failed | scope={scope_id} | error={err}")

        run_rows.append({
            **row_base,
            "status": "failed",
            "reason": "ctgan_scope_exception_or_subprocess_failure",
            "train_rows_used": 0,
            "generated_rows": 0,
            "fit_succeeded": False,
            "sample_succeeded": False,
            "discrete_cols_n": 0,
            "subprocess_returncode": None,
            "elapsed_sec": np.nan,
            "error": err,
        })
        artifact_rows.append({
            "baseline": "CTGAN",
            "scope_id": scope_id,
            "path": out_path,
            "exists": False,
            "rows": 0,
            "cols": int(len(cols)),
            "sha256": "",
            "metadata_path": meta_path,
            "metadata_sha256": "",
            "status": "failed",
            "reason": err,
        })
        if bool(CFG.get("cell17_1_fail_on_scope_failure", False)):
            raise

# ----------------------------------------------------------
# 8) Save registries
# ----------------------------------------------------------
run_audit_df = pd.DataFrame(run_rows)
artifact_registry_df = pd.DataFrame(artifact_rows)

run_audit_df.to_csv(run_audit_csv, index=False)
artifact_registry_df.to_csv(artifact_registry_csv, index=False)

success_n = int((run_audit_df["status"].astype(str) == "success").sum()) if len(run_audit_df) else 0
failed_n = int((run_audit_df["status"].astype(str) == "failed").sum()) if len(run_audit_df) else 0
not_run_n = int((run_audit_df["status"].astype(str) == "not_run").sum()) if len(run_audit_df) else 0
skipped_n = int((run_audit_df["status"].astype(str) == "skipped").sum()) if len(run_audit_df) else 0

# ----------------------------------------------------------
# 9) Contract / manifest
# ----------------------------------------------------------
contract = {
    "cell": "17.1R",
    "version": CELL171_VERSION,
    "role": "ctgan_external_baseline_no_q4_promotion_kernel_safe_subprocess",
    "baseline": "CTGAN",
    "ctgan_backend": ctgan_backend,
    "ctgan_available": bool(ctgan_available),
    "eligible_scopes": eligible_scopes,
    "q4_governance": {
        "final_q4_status": "blocked_no_promotion",
        "q4_coupled_artifacts_used": False,
        "baseline_not_compared_to_Q4_coupled_artifact": True,
    },
    "q6_public_context": CELL17_0_BASELINE_FAIRNESS_CONTRACT.get("q6_public_context", {}),
    "upstream_contract_versions": {
        "cell17_0": CELL17_0_VERSION_171,
    },
    "summary": {
        "scopes_total": int(len(eligible_scopes)),
        "success_n": int(success_n),
        "failed_n": int(failed_n),
        "not_run_n": int(not_run_n),
        "skipped_n": int(skipped_n),
        "generated_test_rows_per_successful_scope": int(N_TE),
        "kernel_safe_subprocess": True,
    },
    "fairness_contract": {
        "allowed_scopes": allowed_scopes,
        "allowed_metrics": plan.get("allowed_metrics", []),
        "not_allowed_claims": plan.get("not_allowed_claims", []),
        "do_not_compare_against_full_pipeline": True,
        "do_not_claim_q4_or_q6_capability": True,
        "do_not_claim_public_release_privacy_pass": True,
    },
    "ctgan_config": {
        "encoding_policy_by_scope": {"protocol_core_tabular": "ctgan_numeric_counts_v1", "public_candidate_core_tabular": "ctgan_numeric_counts_v1", "iot_continuous_compact_tabular": "original_compact_iot"},
        "train_row_cap": int(TRAIN_ROW_CAP_171),
        "epochs": int(CTGAN_EPOCHS_171),
        "batch_size": int(BATCH_SIZE_171),
        "integer_unique_threshold": int(INTEGER_UNIQUE_THRESHOLD_171),
        "discrete_unique_threshold": int(DISCRETE_UNIQUE_THRESHOLD_171),
        "cuda": bool(CFG.get("cell17_1_ctgan_cuda", False)),
        "subprocess_timeout_sec": int(SUBPROCESS_TIMEOUT_171),
        "subprocess_threads": int(CFG.get("cell17_1_subprocess_threads", 2)),
        "max_scope_cols": int(MAX_SCOPE_COLS_171),
    },
    "strict_contract": {
        "Q4_final_status": "blocked_no_promotion",
        "Q4_coupled_artifacts_used": False,
        "TRAIN_real_values_used_for_fitting": True,
        "VAL_real_values_used_for_selection": False,
        "TEST_real_values_used": False,
        "TEST_real_values_used_for_materialization": False,
        "pipeline_synthetic_values_mutated": False,
        "external_baseline_values_generated": bool(success_n > 0),
        "selection_done_here": False,
        "generator_fit_done_here": bool(ctgan_available and success_n > 0),
        "materialization_done_here": bool(success_n > 0),
        "external_baseline_only": True,
        "kernel_safe_subprocess": True,
        "fallback_model_used": False,
    },
    "outputs": {
        "run_audit_csv": run_audit_csv,
        "artifact_registry_csv": artifact_registry_csv,
        "contract_json": contract_json,
        "contract_canonical_json": contract_canonical_json,
        "manifest_json": manifest_json,
    },
}

_write_json_171(contract_json, contract)
_write_json_171(contract_canonical_json, contract)

manifest = {
    "cell": "17.1R",
    "version": CELL171_VERSION,
    "baseline": "CTGAN",
    "created_outputs": contract["outputs"],
    "q4_governance": contract["q4_governance"],
    "q6_public_context": contract["q6_public_context"],
    "artifact_registry": artifact_registry_df.to_dict("records"),
    "summary": contract["summary"],
    "strict_contract": contract["strict_contract"],
    "fairness_contract": contract["fairness_contract"],
}
_write_json_171(manifest_json, manifest)

hashes = {
    "run_audit_csv_sha256": _sha256_file_171(run_audit_csv),
    "artifact_registry_csv_sha256": _sha256_file_171(artifact_registry_csv),
    "contract_json_sha256": _sha256_file_171(contract_json),
    "contract_canonical_json_sha256": _sha256_file_171(contract_canonical_json),
    "manifest_json_sha256": _sha256_file_171(manifest_json),
}
contract["hashes"] = hashes
manifest["hashes"] = hashes

_write_json_171(contract_json, contract)
_write_json_171(contract_canonical_json, contract)
_write_json_171(manifest_json, manifest)

# ----------------------------------------------------------
# 10) Export globals
# ----------------------------------------------------------
globals()["CELL171_VERSION"] = CELL171_VERSION
globals()["CELL17_1_CTGAN_RUN_AUDIT_DF"] = run_audit_df
globals()["CELL17_1_CTGAN_ARTIFACT_REGISTRY_DF"] = artifact_registry_df
globals()["CELL17_1_CTGAN_BASELINE_CONTRACT"] = contract

globals()["CELL17_1_CTGAN_RUN_AUDIT_CSV"] = run_audit_csv
globals()["CELL17_1_CTGAN_ARTIFACT_REGISTRY_CSV"] = artifact_registry_csv
globals()["CELL17_1_CTGAN_CONTRACT_JSON"] = contract_json
globals()["CELL17_1_CTGAN_CONTRACT_CANONICAL_JSON"] = contract_canonical_json
globals()["CELL17_1_CTGAN_MANIFEST_JSON"] = manifest_json

log(
    "[Cell17.1R] CTGAN baseline complete | "
    f"backend={ctgan_backend} | available={ctgan_available} | "
    f"success={success_n} | failed={failed_n} | skipped={skipped_n} | not_run={not_run_n} | "
    "q4_status=blocked_no_promotion"
)
log(f"[Cell17.1R] Run audit saved: {run_audit_csv}")
log(f"[Cell17.1R] Artifact registry saved: {artifact_registry_csv}")
log(f"[Cell17.1R] Contract saved: {contract_json}")
log(f"[Cell17.1R] Canonical contract saved: {contract_canonical_json}")
log(
    "[Cell17.1R] Contract flags | "
    "Q4_final_status=blocked_no_promotion | "
    "Q4_coupled_artifacts_used=False | "
    "TRAIN_real_values_used_for_fitting=True | "
    "VAL_real_values_used_for_selection=False | "
    "TEST_real_values_used=False | "
    "TEST_real_values_used_for_materialization=False | "
    "pipeline_synthetic_values_mutated=False | "
    "selection_done_here=False | "
    f"generator_fit_done_here={bool(ctgan_available and success_n > 0)} | "
    f"materialization_done_here={bool(success_n > 0)} | "
    "external_baseline_only=True | "
    "kernel_safe_subprocess=True"
)
log("--- END: Cell 17.1R - CTGAN external baseline, kernel-safe subprocess ---")

gc.collect()
