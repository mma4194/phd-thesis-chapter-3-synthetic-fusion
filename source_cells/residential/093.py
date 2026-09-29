# %% CELL 12.f.V5.6 — Q3 V5 observability mask parquet alias for Cell 12.g
# Purpose:
#   Cell 12.g expects the selected observability-mask matrix at:
#       OUT_SYN/IOT_SELECTED_OBSERVABILITY_MASKS_TEST.parquet
#
#   The active Q3 V5 branch writes semantically correct indicator-mask files
#   under V5/final names. This cell locates the active Q3 V5 mask matrix,
#   validates shape/columns against the Q3 V5 status file, and writes the
#   legacy filename expected by 12.g.
#
# Safety:
#   - Does not read real TEST values.
#   - Does not rerun selection, fitting, materialization, or QA.
#   - Does not change mask values; it writes compatibility aliases only.
#   - Refuses invalid V2 semantic-collapse artifacts.

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

def _v5mask_alias_log(msg):
    print(f"[12.f.V5.6 mask-alias] {msg}")

# ---------------------------------------------------------------------
# 0. Resolve roots
# ---------------------------------------------------------------------
for name in ["OUTDIR", "OUT_SYN", "REPORT_DIR", "CONTRACT_DIR"]:
    if name not in globals():
        raise RuntimeError(f"[12.f.V5.6] Missing required global: {name}")

OUTDIR_P = Path(str(OUTDIR)).expanduser().resolve()
OUT_SYN_P = Path(str(OUT_SYN)).expanduser().resolve()
REPORT_DIR_P = Path(str(REPORT_DIR)).expanduser().resolve()
CONTRACT_DIR_P = Path(str(CONTRACT_DIR)).expanduser().resolve()

OUT_SYN_P.mkdir(parents=True, exist_ok=True)
REPORT_DIR_P.mkdir(parents=True, exist_ok=True)
CONTRACT_DIR_P.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Required Q3 V5 metadata files
# ---------------------------------------------------------------------
status_path = REPORT_DIR_P / "cell12f5_mask_publication_status.csv"
metrics_path = REPORT_DIR_P / "cell12f5_mask_final_test_qa_metrics.csv"
q3_post_v5_path = CONTRACT_DIR_P / "q3_post_v5_observability_mask_post_qa_contract.json"

required = [status_path, metrics_path, q3_post_v5_path]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise RuntimeError("[12.f.V5.6] Missing required Q3 V5 artifacts:\n" + "\n".join(missing))

status_df = pd.read_csv(status_path)
metrics_df = pd.read_csv(metrics_path)

with open(q3_post_v5_path, "r", encoding="utf-8") as f:
    q3_post = json.load(f)

for df in [status_df, metrics_df]:
    if "column" not in df.columns and "col" in df.columns:
        df["column"] = df["col"]
    if "col" not in df.columns and "column" in df.columns:
        df["col"] = df["column"]

if "col" not in status_df.columns:
    raise RuntimeError("[12.f.V5.6] Status file has no col/column field.")

expected_cols = [str(x) for x in status_df["col"].dropna().astype(str).tolist()]
expected_cols = list(dict.fromkeys(expected_cols))

if len(expected_cols) != 183:
    raise RuntimeError(f"[12.f.V5.6] Expected 183 Q3 mask columns, found {len(expected_cols)}.")

# Guard against V2 semantic collapse.
bad_v2_text = ""
for c in ["selected_generator", "materializer_class_v2", "materializer_class_v5"]:
    if c in status_df.columns:
        bad_v2_text += " " + status_df[c].astype(str).str.cat(sep=" ")

if "MaskExactObservedConstantV2" in bad_v2_text or "exact_observed_constant" in bad_v2_text:
    raise RuntimeError(
        "[12.f.V5.6] Refusing to alias invalid V2 semantic-collapse masks. "
        "Active branch must be Q3 V5."
    )

# ---------------------------------------------------------------------
# 2. Resolve TEST length
# ---------------------------------------------------------------------
N_TE = None
if "df_te" in globals() and isinstance(globals()["df_te"], pd.DataFrame):
    N_TE = int(len(globals()["df_te"]))
elif "DF_TE" in globals() and isinstance(globals()["DF_TE"], pd.DataFrame):
    N_TE = int(len(globals()["DF_TE"]))
else:
    # infer from any existing active branch matrix
    for p in [
        OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_FINAL_TEST.parquet",
        OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_TEST.parquet",
        OUT_SYN_P / "IOT_FINAL_OBSERVABILITY_MASK_TEST.parquet",
        OUT_SYN_P / "IOT_FINAL_MASK_TEST.parquet",
        OUT_SYN_P / "IOT_MASK_FINAL_TEST.parquet",
        OUT_SYN_P / "IOT_VALUE_MASK_FINAL_TEST.parquet",
    ]:
        if p.exists():
            tmp = pd.read_parquet(p)
            N_TE = int(tmp.shape[0])
            break

if N_TE is None:
    raise RuntimeError("[12.f.V5.6] Could not resolve TEST length.")

# ---------------------------------------------------------------------
# 3. Locate active V5 mask matrix
# ---------------------------------------------------------------------
candidate_paths = [
    OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_FINAL_TEST.parquet",
    OUT_SYN_P / "IOT_OBSERVABILITY_MASK_V5_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_OBSERVABILITY_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_FINAL_MASK_TEST.parquet",
    OUT_SYN_P / "IOT_MASK_FINAL_TEST.parquet",
    OUT_SYN_P / "IOT_VALUE_MASK_FINAL_TEST.parquet",
]

valid_candidates = []

for p in candidate_paths:
    if not p.exists():
        continue
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        _v5mask_alias_log(f"Could not read {p.name}: {e}")
        continue

    cols = list(map(str, df.columns))
    present = [c for c in expected_cols if c in cols]

    if df.shape[0] != N_TE:
        _v5mask_alias_log(f"Reject {p.name}: row count {df.shape[0]} != {N_TE}")
        continue

    if len(present) != len(expected_cols):
        _v5mask_alias_log(f"Reject {p.name}: expected col coverage {len(present)}/{len(expected_cols)}")
        continue

    # Normalize order and validate binary values.
    df_norm = df[expected_cols].copy()
    non_binary = {}
    for c in expected_cols:
        vals = pd.to_numeric(df_norm[c], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if not finite.all():
            non_binary[c] = "nonfinite"
            continue
        unique = set(np.unique(vals).tolist())
        if not unique.issubset({0.0, 1.0}):
            non_binary[c] = sorted(list(unique))[:10]

    if non_binary:
        sample = dict(list(non_binary.items())[:10])
        _v5mask_alias_log(f"Reject {p.name}: non-binary sample={sample}")
        continue

    valid_candidates.append((p, df_norm))

if not valid_candidates:
    raise RuntimeError(
        "[12.f.V5.6] Could not locate a valid active Q3 V5 observability mask matrix."
    )

# Prefer explicitly V5-named file if present.
valid_candidates = sorted(
    valid_candidates,
    key=lambda t: (("V5" in t[0].name), ("FINAL" in t[0].name), len(t[0].name)),
    reverse=True,
)

source_path, mask_df = valid_candidates[0]

# ---------------------------------------------------------------------
# 4. Write legacy aliases expected by 12.g
# ---------------------------------------------------------------------
legacy_selected_path = OUT_SYN_P / "IOT_SELECTED_OBSERVABILITY_MASKS_TEST.parquet"
legacy_final_path = OUT_SYN_P / "IOT_FINAL_OBSERVABILITY_MASKS_TEST.parquet"

mask_df.to_parquet(legacy_selected_path, index=False)
mask_df.to_parquet(legacy_final_path, index=False)

# Keep singular aliases too, for downstream cells that may use them.
singular_selected_path = OUT_SYN_P / "IOT_SELECTED_OBSERVABILITY_MASK_TEST.parquet"
singular_final_path = OUT_SYN_P / "IOT_FINAL_OBSERVABILITY_MASK_TEST.parquet"
mask_df.to_parquet(singular_selected_path, index=False)
mask_df.to_parquet(singular_final_path, index=False)

def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# ---------------------------------------------------------------------
# 5. Publish in-memory globals likely useful downstream
# ---------------------------------------------------------------------
CELL12F_SELECTED_OBSERVABILITY_MASKS_TEST_DF = mask_df
CELL12F_SELECTED_OBSERVABILITY_MASKS_TEST_PATH = str(legacy_selected_path)
CELL12F_FINAL_OBSERVABILITY_MASKS_TEST_PATH = str(legacy_final_path)

CELL12F5_SELECTED_OBSERVABILITY_MASKS_TEST_DF = mask_df
CELL12F5_SELECTED_OBSERVABILITY_MASKS_TEST_PATH = str(legacy_selected_path)
CELL12F5_ACTIVE_BRANCH = "12.f.V5"

if "CFG" not in globals() or not isinstance(CFG, dict):
    CFG = {}

CFG["active_q3_branch"] = "Q3.POST.V5"
CFG["active_mask_branch"] = "12.f.V5"
CFG["cell12g_observability_mask_handoff_source"] = "12.f.V5.6"
CFG["cell12g_observability_mask_path"] = str(legacy_selected_path)

# ---------------------------------------------------------------------
# 6. Write contract/report
# ---------------------------------------------------------------------
summary = {
    "cell": "12.f.V5.6",
    "role": "q3_v5_observability_mask_matrix_alias_for_cell12g",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "active_q3_branch": "Q3.POST.V5",
    "active_mask_branch": "12.f.V5",
    "source_path": str(source_path),
    "legacy_selected_path": str(legacy_selected_path),
    "legacy_final_path": str(legacy_final_path),
    "singular_selected_path": str(singular_selected_path),
    "singular_final_path": str(singular_final_path),
    "shape": list(mask_df.shape),
    "targets": int(mask_df.shape[1]),
    "N_TE": int(mask_df.shape[0]),
    "q3_final_status": q3_post.get("final_status", "BLOCKED_PARTIAL"),
    "q3_claim_scope": q3_post.get("claim_scope", "partial_admissible_only"),
    "q3_full_scope_ready": bool(q3_post.get("full_scope_ready", False)),
    "q3_publication_blocker_n": int(q3_post.get("publication_blocker_n", 0)),
    "policy": {
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "TEST_real_values_read": False,
        "legacy_alias_write_only": True,
        "invalid_v2_semantic_collapse_rejected": True,
    },
    "files": {
        "legacy_selected_sha256": _sha256_file(legacy_selected_path),
        "legacy_final_sha256": _sha256_file(legacy_final_path),
        "singular_selected_sha256": _sha256_file(singular_selected_path),
        "singular_final_sha256": _sha256_file(singular_final_path),
    },
}

report_path = REPORT_DIR_P / "cell12f_v5_to_cell12g_mask_alias_report.json"
contract_path = CONTRACT_DIR_P / "cell12f_v5_to_cell12g_mask_alias_contract.json"

for p in [report_path, contract_path]:
    p.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

CELL12F5_TO_CELL12G_MASK_ALIAS = summary

_v5mask_alias_log(f"Source: {source_path}")
_v5mask_alias_log(f"Wrote legacy selected mask path: {legacy_selected_path}")
_v5mask_alias_log(f"Shape: {mask_df.shape}")
_v5mask_alias_log(f"Contract: {contract_path}")
_v5mask_alias_log("PASS: Cell 12.g observability mask alias is ready.")