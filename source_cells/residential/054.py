# %% CELL Q6.0x — External quarantine for superseded continuous selector-policy artifacts
# Purpose:
#   Move known old/superseded continuous selector-policy diagnostic files
#   outside the final STUDY submission root.
#
# Why:
#   Cell 12.c.2 may emit legacy diagnostic artifacts with old
#   "final_selection_disabled_after_testqa" wording. These files are not
#   active in the final branch, but their presence inside OUTDIR is enough
#   for Q6.1 to fail correctly.
#
# Safety:
#   - Does not modify synthetic values.
#   - Does not modify active 12.c.R, Binary V6, or Q3 V5 artifacts.
#   - Does not delete evidence; it moves superseded files outside the
#     submitted artifact root.
#   - The detailed manifest is written outside OUTDIR.
#   - The in-root manifest avoids recording the risky old filenames.

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone

def _q6x_log(msg):
    print(f"[Q6.0x] {msg}")

# ---------------------------------------------------------------------
# Resolve roots
# ---------------------------------------------------------------------
required_globals = ["OUTDIR", "REPORT_DIR", "CONTRACT_DIR"]
for name in required_globals:
    if name not in globals():
        raise RuntimeError(f"[Q6.0x] Missing required global: {name}")

OUTDIR_P = Path(str(OUTDIR)).expanduser().resolve()
REPORT_DIR_P = Path(str(REPORT_DIR)).expanduser().resolve()
CONTRACT_DIR_P = Path(str(CONTRACT_DIR)).expanduser().resolve()

ARTIFACT_DIR_P = Path(str(globals().get("ARTIFACT_DIR", globals().get("ARTDIR", OUTDIR_P / "artifacts")))).expanduser().resolve()
OUT_SYN_P = Path(str(globals().get("OUT_SYN", OUTDIR_P / "synthetic"))).expanduser().resolve()

if not OUTDIR_P.exists():
    raise RuntimeError(f"[Q6.0x] OUTDIR does not exist: {OUTDIR_P}")

# ---------------------------------------------------------------------
# Exact known superseded files.
# Keep this exact and conservative. Do not broad-delete cell12c3 files.
# ---------------------------------------------------------------------
KNOWN_SUPERSEDED_NAMES = {
    "cell12c3_temperature_v2_final_selection_policy_audit.csv",
    "cell12c3_dense_temperature_selector_audit.csv",
    "cell12c3_diagnostic_pool_not_selected_audit.csv",
    # These may exist in some development roots; include as exact-name guards.
    "cell12c_continuous_selector_test_policy_active_hits.csv",
    "cell12c_continuous_selector_test_policy_file_hits.csv",
    "cell12c_continuous_selector_test_policy_row_hits.csv",
    "cell12c_lineage_final_artifact_candidates.csv",
}

# ---------------------------------------------------------------------
# Destination outside the final submission root.
# ---------------------------------------------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
external_quarantine_root = (
    OUTDIR_P.parent
    / f"{OUTDIR_P.name}__NON_SUBMISSION_QUARANTINE"
    / timestamp
    / "continuous_selector_policy_superseded"
).resolve()

# Hard assertion: quarantine must not be inside OUTDIR.
try:
    external_quarantine_root.relative_to(OUTDIR_P)
    raise RuntimeError(
        "[Q6.0x] Refusing to quarantine inside OUTDIR. "
        f"Destination is inside final root: {external_quarantine_root}"
    )
except ValueError:
    pass

external_quarantine_root.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Locate exact-name hits under the canonical active root.
# ---------------------------------------------------------------------
search_roots = []
for p in [REPORT_DIR_P, CONTRACT_DIR_P, ARTIFACT_DIR_P, OUT_SYN_P]:
    if p.exists():
        search_roots.append(p)

# Fallback: report dir may be enough, but include OUTDIR as a last conservative pass.
# This still only moves exact known filenames.
if OUTDIR_P.exists():
    search_roots.append(OUTDIR_P)

seen = set()
hits = []

for root in search_roots:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        if p.name in KNOWN_SUPERSEDED_NAMES:
            hits.append(p.resolve())

hits = sorted(hits, key=lambda x: str(x))

_q6x_log(f"OUTDIR = {OUTDIR_P}")
_q6x_log(f"External quarantine root = {external_quarantine_root}")
_q6x_log(f"Superseded exact-name hits found = {len(hits)}")

# ---------------------------------------------------------------------
# Move hits outside OUTDIR.
# ---------------------------------------------------------------------
moved_records = []

for src in hits:
    rel = src.relative_to(OUTDIR_P) if src.is_relative_to(OUTDIR_P) else Path(src.name)
    dst = external_quarantine_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        dst = dst.with_name(f"{stem}__duplicate_{hashlib.sha256(str(src).encode()).hexdigest()[:8]}{suffix}")

    before_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.move(str(src), str(dst))
    after_hash = hashlib.sha256(dst.read_bytes()).hexdigest()

    moved_records.append({
        "original_relative_path": str(rel),
        "original_abs_path": str(src),
        "quarantine_abs_path": str(dst),
        "sha256_before": before_hash,
        "sha256_after": after_hash,
        "hash_match_after_move": before_hash == after_hash,
        "moved_utc": datetime.now(timezone.utc).isoformat(),
    })

    _q6x_log(f"Moved outside OUTDIR: {rel}")

# ---------------------------------------------------------------------
# Write detailed manifest outside OUTDIR.
# ---------------------------------------------------------------------
external_manifest = external_quarantine_root / "detailed_external_quarantine_manifest.json"
external_manifest.write_text(
    json.dumps({
        "cell": "Q6.0x",
        "role": "external_quarantine_for_superseded_continuous_selector_policy_artifacts",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": str(OUTDIR_P),
        "external_quarantine_root": str(external_quarantine_root),
        "moved_n": len(moved_records),
        "records": moved_records,
        "policy": {
            "synthetic_values_mutated": False,
            "selection_done_here": False,
            "generator_fit_done_here": False,
            "post_TEST_repair_done_here": False,
            "quarantine_inside_submission_root": False,
            "detailed_manifest_inside_submission_root": False,
        },
    }, indent=2, sort_keys=True),
    encoding="utf-8",
)

# ---------------------------------------------------------------------
# Write safe in-root receipt.
# Important: avoid writing old risky filenames/content into OUTDIR.
# ---------------------------------------------------------------------
safe_receipt = {
    "cell": "Q6.0x",
    "role": "safe_receipt_for_external_quarantine",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "moved_n": len(moved_records),
    "external_quarantine_root": str(external_quarantine_root),
    "external_manifest_sha256": hashlib.sha256(external_manifest.read_bytes()).hexdigest(),
    "policy": {
        "synthetic_values_mutated": False,
        "selection_done_here": False,
        "generator_fit_done_here": False,
        "post_TEST_repair_done_here": False,
        "quarantine_inside_submission_root": False,
        "safe_receipt_omits_superseded_filenames": True,
    },
}

safe_receipt_path = REPORT_DIR_P / "q6_external_quarantine_safe_receipt.json"
safe_receipt_path.write_text(
    json.dumps(safe_receipt, indent=2, sort_keys=True),
    encoding="utf-8",
)

# ---------------------------------------------------------------------
# Final verification: exact known files should no longer exist under OUTDIR.
# ---------------------------------------------------------------------
remaining = []
for p in OUTDIR_P.rglob("*"):
    if p.is_file() and p.name in KNOWN_SUPERSEDED_NAMES:
        remaining.append(str(p.relative_to(OUTDIR_P)))

if remaining:
    print(json.dumps(remaining, indent=2))
    raise RuntimeError(
        "[Q6.0x] Some superseded exact-name files remain inside OUTDIR. "
        "Do not continue to Q6.1 until these are removed."
    )

_q6x_log(f"External detailed manifest: {external_manifest}")
_q6x_log(f"Safe in-root receipt: {safe_receipt_path}")
_q6x_log("PASS: superseded continuous selector-policy artifacts are outside the final submission root.")

CELL_Q6_EXTERNAL_QUARANTINE_RECEIPT = safe_receipt