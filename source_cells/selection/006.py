# ============================================================
# STRONG.3N — Q6 sparse-driver sequence-risk null calibration
# Purpose:
#   Calibrate exact-window and nearest-Hamming sparse-driver
#   sequence-risk screens against a real-vs-real null.
#
# Run after:
#   STRONG.3
#
# This cell DOES NOT change the public artifact. It writes
# null-calibrated diagnostics and a recommended interpretation.
# ============================================================

from pathlib import Path
import os
import json
import math
import hashlib
import datetime
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 0. Resolve run paths
# ------------------------------------------------------------------

RUN_ROOT = Path(os.environ.get(
    "CPS_CANONICAL_RUN",
    "/path/to/input/cps_synth_v4_4_7_32_BINARY_V6_Q3V5_STUDY_FINAL_20260708_035653"
)).expanduser().resolve()

if not RUN_ROOT.exists():
    raise RuntimeError(f"RUN_ROOT does not exist: {RUN_ROOT}")

REAL_TEST_PATH = RUN_ROOT / "synthetic/REAL_TEST_SPLIT.parquet"
SYN_PRIOR_PUBLIC_PATH = RUN_ROOT / "synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet"
SYN_CONSERVATIVE_PUBLIC_PATH = RUN_ROOT / "synthetic/CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet"

for p in [REAL_TEST_PATH, SYN_PRIOR_PUBLIC_PATH]:
    if not p.exists():
        raise FileNotFoundError(p)

OUT_DIR = RUN_ROOT / "reports/thesis_strong_revision_fixes"
TABLE_DIR = OUT_DIR / "tables"
LEDGER_DIR = OUT_DIR / "ledgers"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = int(os.environ.get("Q6_SEQUENCE_NULL_WINDOW_SIZE", "60"))

print("RUN_ROOT:", RUN_ROOT)
print("REAL_TEST:", REAL_TEST_PATH)
print("SYN_PRIOR_PUBLIC:", SYN_PRIOR_PUBLIC_PATH)
print("WINDOW_SIZE:", WINDOW_SIZE)

# ------------------------------------------------------------------
# 1. Helpers
# ------------------------------------------------------------------

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()

def pick_col(df: pd.DataFrame, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def is_time_like(c) -> bool:
    s = str(c).lower()
    return (
        s in {"time", "timestamp", "datetime", "date", "index", "__index_level_0__"}
        or "timestamp" in s
        or s.endswith("_time")
        or s.endswith("_date")
    )

def is_protocol_like(c) -> bool:
    s = str(c).lower()
    tokens = [
        "router", "zigbee", "z-wave", "zwave", "wifi", "wlan", "ethernet",
        "packet", "packets", "byte", "bytes", "proto", "protocol", "tcp",
        "udp", "ip", "dns", "http", "ssl", "conn", "port", "ota"
    ]
    return any(t in s for t in tokens)

def display_df(title, df, max_rows=30):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    try:
        display(df.head(max_rows) if len(df) > max_rows else df)
    except Exception:
        print((df.head(max_rows) if len(df) > max_rows else df).to_string(index=False))

def safe_ratio(num, den):
    if den is None or not np.isfinite(den) or den == 0:
        return np.nan
    return float(num / den)

def numeric_binary_matrix(df: pd.DataFrame, cols):
    tmp = pd.DataFrame(index=df.index)
    for c in cols:
        tmp[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Sparse drivers should be event indicators. Any positive value means active.
    return (tmp.to_numpy(dtype=float) > 0).astype(np.uint8)

def window_active_vectors(binary_arr: np.ndarray, window_size: int):
    n_rows, n_cols = binary_arr.shape
    n_windows = n_rows // window_size
    if n_windows <= 0:
        raise RuntimeError(f"Not enough rows for window_size={window_size}: {n_rows}")
    trimmed = binary_arr[:n_windows * window_size, :]
    win = trimmed.reshape(n_windows, window_size, n_cols).max(axis=1).astype(np.uint8)
    return win

def encode_windows(win: np.ndarray):
    n_windows, n_bits = win.shape
    if n_bits > 62:
        raise RuntimeError(f"This encoder supports <=62 sparse-driver columns; got {n_bits}.")
    weights = (np.uint64(1) << np.arange(n_bits, dtype=np.uint64))
    codes = (win.astype(np.uint64) * weights).sum(axis=1)
    return codes

def nearest_hamming_normalized(query_codes, ref_codes, n_bits):
    query_codes = np.asarray(query_codes, dtype=np.uint64)
    ref_codes = np.unique(np.asarray(ref_codes, dtype=np.uint64))

    if len(query_codes) == 0 or len(ref_codes) == 0:
        return np.array([], dtype=float)

    # Fast path for small bit vectors.
    if n_bits <= 24:
        table_size = 1 << n_bits
        bitcounts = np.fromiter((i.bit_count() for i in range(table_size)), dtype=np.uint8, count=table_size)
        ref32 = ref_codes.astype(np.uint32)
        out = np.empty(len(query_codes), dtype=float)
        for i, q in enumerate(query_codes.astype(np.uint32)):
            xor = np.bitwise_xor(ref32, q)
            out[i] = bitcounts[xor].min() / float(n_bits)
        return out

    # Fallback.
    out = []
    ref_int = [int(x) for x in ref_codes]
    for q in query_codes:
        qi = int(q)
        out.append(min((qi ^ r).bit_count() for r in ref_int) / float(n_bits))
    return np.asarray(out, dtype=float)

def repeated_rates(codes_active):
    """
    Returns:
      duplicate_member_rate: fraction of active windows whose signature occurs >=2 times.
      repeat_after_first_rate: fraction after removing one copy of each unique signature.
    """
    codes_active = np.asarray(codes_active, dtype=np.uint64)
    if len(codes_active) == 0:
        return np.nan, np.nan, 0
    vals, counts = np.unique(codes_active, return_counts=True)
    duplicate_member_count = int(counts[counts >= 2].sum())
    duplicate_member_rate = duplicate_member_count / len(codes_active)
    repeat_after_first_rate = (len(codes_active) - len(vals)) / len(codes_active)
    return float(duplicate_member_rate), float(repeat_after_first_rate), int(len(vals))

def compute_sequence_metrics(query_win, ref_win, label, n_bits):
    query_codes = encode_windows(query_win)
    ref_codes = encode_windows(ref_win)

    query_active = query_win.any(axis=1)
    ref_active = ref_win.any(axis=1)

    q_active_codes = query_codes[query_active]
    r_active_codes = ref_codes[ref_active]

    active_rate = float(query_active.mean())
    ref_active_rate = float(ref_active.mean())

    ref_active_set = set(int(x) for x in r_active_codes)
    if len(q_active_codes) == 0:
        exact_seen_rate_active = np.nan
        exact_seen_count = 0
    else:
        exact_seen_count = int(sum(int(x) in ref_active_set for x in q_active_codes))
        exact_seen_rate_active = exact_seen_count / len(q_active_codes)

    duplicate_member_rate, repeat_after_first_rate, n_unique_active_signatures = repeated_rates(q_active_codes)

    d = nearest_hamming_normalized(q_active_codes, r_active_codes, n_bits)
    if len(d) == 0:
        h_min = h_p01 = h_p05 = h_median = h_mean = np.nan
    else:
        h_min = float(np.min(d))
        h_p01 = float(np.quantile(d, 0.01))
        h_p05 = float(np.quantile(d, 0.05))
        h_median = float(np.median(d))
        h_mean = float(np.mean(d))

    return {
        "comparison": label,
        "query_windows": int(len(query_win)),
        "reference_windows": int(len(ref_win)),
        "query_active_windows": int(query_active.sum()),
        "reference_active_windows": int(ref_active.sum()),
        "query_active_union_rate": active_rate,
        "reference_active_union_rate": ref_active_rate,
        "query_over_reference_union_rate_ratio": safe_ratio(active_rate, ref_active_rate),
        "exact_active_windows_seen_in_reference_count": exact_seen_count,
        "exact_active_windows_seen_in_reference_rate": exact_seen_rate_active,
        "duplicate_member_rate_among_query_active_windows": duplicate_member_rate,
        "repeat_after_first_rate_among_query_active_windows": repeat_after_first_rate,
        "query_unique_active_signatures": n_unique_active_signatures,
        "nearest_reference_hamming_min": h_min,
        "nearest_reference_hamming_p01": h_p01,
        "nearest_reference_hamming_p05": h_p05,
        "nearest_reference_hamming_median": h_median,
        "nearest_reference_hamming_mean": h_mean,
    }

# ------------------------------------------------------------------
# 2. Load artifacts and identify the 23 prior sparse-driver columns
# ------------------------------------------------------------------

real = pd.read_parquet(REAL_TEST_PATH)
syn_prior_public = pd.read_parquet(SYN_PRIOR_PUBLIC_PATH)

# Remove persisted pandas index if present.
for df in [real, syn_prior_public]:
    if "__index_level_0__" in df.columns:
        df.drop(columns=["__index_level_0__"], inplace=True)

driver_cols = []

# Preferred source: STRONG.3 driver-contribution output.
contrib_candidates = [
    TABLE_DIR / "q6_sparse_driver_sequence_driver_contributions.csv",
    LEDGER_DIR / "q6_sparse_driver_sequence_driver_contributions.csv",
]
for p in contrib_candidates:
    if p.exists() and not driver_cols:
        d = pd.read_csv(p)
        col = pick_col(d, [
            "driver_column", "sparse_driver_column", "column", "col",
            "feature", "feature_name", "target", "variable"
        ])
        if col is not None:
            vals = [str(x) for x in d[col].dropna().unique().tolist()]
            driver_cols = [c for c in vals if c in real.columns and c in syn_prior_public.columns]

# Secondary source: Q6 registry.
registry_candidates = [
    RUN_ROOT / "reports/cell15_6_q6_public_column_registry.csv",
    RUN_ROOT / "reports/q6_public_column_registry.csv",
    RUN_ROOT / "reports/release_scope_table.csv",
]
for p in registry_candidates:
    if p.exists() and not driver_cols:
        r = pd.read_csv(p)
        name_col = pick_col(r, ["column", "col", "feature", "feature_name", "target"])
        role_cols = [
            c for c in r.columns
            if any(tok in c.lower() for tok in ["role", "owner", "group", "branch", "class"])
        ]
        if name_col is not None and role_cols:
            mask = pd.Series(False, index=r.index)
            for rc in role_cols:
                mask = mask | r[rc].astype(str).str.lower().str.contains("driver|sparse", regex=True, na=False)
            vals = [str(x) for x in r.loc[mask, name_col].dropna().unique().tolist()]
            driver_cols = [c for c in vals if c in real.columns and c in syn_prior_public.columns]

# Last-resort fallback: prior public columns minus timestamp/protocol-like columns.
# This is intentionally printed as fallback if used.
used_fallback = False
if not driver_cols:
    used_fallback = True
    candidate_cols = []
    for c in syn_prior_public.columns:
        if c not in real.columns:
            continue
        if is_time_like(c) or is_protocol_like(c):
            continue
        if pd.api.types.is_numeric_dtype(syn_prior_public[c]) and pd.api.types.is_numeric_dtype(real[c]):
            candidate_cols.append(c)
    driver_cols = candidate_cols

driver_cols = sorted(dict.fromkeys(driver_cols))

if len(driver_cols) == 0:
    raise RuntimeError(
        "Could not identify prior sparse-driver columns. "
        "Check q6_sparse_driver_sequence_driver_contributions.csv or Q6 registry files."
    )

print(f"Identified sparse-driver columns: {len(driver_cols)}")
print("Used fallback detection:", used_fallback)
for c in driver_cols:
    print(" -", c)

driver_set_path = TABLE_DIR / "q6_sparse_driver_sequence_risk_null_driver_set.csv"
pd.DataFrame({"sparse_driver_column": driver_cols}).to_csv(driver_set_path, index=False)

# ------------------------------------------------------------------
# 3. Build aligned 60-row active-window vectors
# ------------------------------------------------------------------

n = min(len(real), len(syn_prior_public))
real_aligned = real.iloc[:n].reset_index(drop=True)
syn_aligned = syn_prior_public.iloc[:n].reset_index(drop=True)

real_bin = numeric_binary_matrix(real_aligned, driver_cols)
syn_bin = numeric_binary_matrix(syn_aligned, driver_cols)

real_win = window_active_vectors(real_bin, WINDOW_SIZE)
syn_win = window_active_vectors(syn_bin, WINDOW_SIZE)

n_windows = min(len(real_win), len(syn_win))
real_win = real_win[:n_windows]
syn_win = syn_win[:n_windows]

n_bits = len(driver_cols)
split = n_windows // 2
real_ref_win = real_win[:split]
real_holdout_win = real_win[split:]

if len(real_ref_win) == 0 or len(real_holdout_win) == 0:
    raise RuntimeError("Real-vs-real split produced an empty side.")

print("\nWindow summary:")
print("  rows aligned:", n)
print("  windows:", n_windows)
print("  real_ref_windows:", len(real_ref_win))
print("  real_holdout_windows:", len(real_holdout_win))
print("  sparse_driver_bits:", n_bits)

# ------------------------------------------------------------------
# 4. Compute null-calibrated diagnostics
# ------------------------------------------------------------------

records = []

# Real-vs-real null: future real holdout compared to earlier real reference.
records.append(compute_sequence_metrics(
    real_holdout_win,
    real_ref_win,
    "real_holdout_vs_real_reference_null",
    n_bits
))

# Synthetic compared to same real reference.
records.append(compute_sequence_metrics(
    syn_win,
    real_ref_win,
    "synthetic_vs_real_reference_calibrated",
    n_bits
))

# Synthetic compared to all real windows, to reproduce/compare old legacy style.
records.append(compute_sequence_metrics(
    syn_win,
    real_win,
    "synthetic_vs_real_full_legacy",
    n_bits
))

# Real full self-description.
records.append(compute_sequence_metrics(
    real_win,
    real_win,
    "real_full_vs_real_full_self_description",
    n_bits
))

diag = pd.DataFrame(records)

diag_path = TABLE_DIR / "q6_sparse_driver_sequence_risk_null_calibration.csv"
diag.to_csv(diag_path, index=False)

# ------------------------------------------------------------------
# 5. Calibrated ratios and decision interpretation
# ------------------------------------------------------------------

null = diag.set_index("comparison").loc["real_holdout_vs_real_reference_null"]
syn_cal = diag.set_index("comparison").loc["synthetic_vs_real_reference_calibrated"]
syn_legacy = diag.set_index("comparison").loc["synthetic_vs_real_full_legacy"]

ratio_records = []

def add_ratio(metric, syn_value, null_value, note):
    ratio_records.append({
        "metric": metric,
        "synthetic_calibrated_value": float(syn_value) if pd.notna(syn_value) else np.nan,
        "real_vs_real_null_value": float(null_value) if pd.notna(null_value) else np.nan,
        "synthetic_minus_null": float(syn_value - null_value) if pd.notna(syn_value) and pd.notna(null_value) else np.nan,
        "synthetic_over_null": safe_ratio(syn_value, null_value),
        "interpretation_note": note,
    })

add_ratio(
    "active_union_rate",
    syn_cal["query_active_union_rate"],
    null["query_active_union_rate"],
    "Compares synthetic public sparse-driver activity prevalence to real holdout sparse-driver activity prevalence."
)

add_ratio(
    "exact_active_windows_seen_in_reference_rate",
    syn_cal["exact_active_windows_seen_in_reference_rate"],
    null["exact_active_windows_seen_in_reference_rate"],
    "Exact-window matches are interpreted only relative to the real-vs-real null; nonzero exact matches are not standalone replay proof."
)

add_ratio(
    "duplicate_member_rate_among_active_windows",
    syn_cal["duplicate_member_rate_among_query_active_windows"],
    null["duplicate_member_rate_among_query_active_windows"],
    "Repeated active-window signatures are interpreted relative to real-vs-real repetition."
)

add_ratio(
    "repeat_after_first_rate_among_active_windows",
    syn_cal["repeat_after_first_rate_among_query_active_windows"],
    null["repeat_after_first_rate_among_query_active_windows"],
    "Alternative repetition statistic: repeated windows after keeping one copy of each active signature."
)

add_ratio(
    "nearest_reference_hamming_p01",
    syn_cal["nearest_reference_hamming_p01"],
    null["nearest_reference_hamming_p01"],
    "Low Hamming distance is interpreted relative to real-vs-real sparse-window self-similarity."
)

add_ratio(
    "nearest_reference_hamming_median",
    syn_cal["nearest_reference_hamming_median"],
    null["nearest_reference_hamming_median"],
    "Median nearest-reference proximity, calibrated against real-vs-real null."
)

ratios = pd.DataFrame(ratio_records)
ratios_path = TABLE_DIR / "q6_sparse_driver_sequence_risk_null_calibrated_ratios.csv"
ratios.to_csv(ratios_path, index=False)

# Conservative decision rules.
# These rules intentionally do NOT block merely because exact_seen_rate > 0
# or Hamming p01 == 0. They treat those as contextual evidence unless they
# exceed the real-vs-real null.
real_full_union = float(diag.set_index("comparison").loc["real_full_vs_real_full_self_description", "query_active_union_rate"])
syn_full_union = float(syn_legacy["query_active_union_rate"])
legacy_union_ratio = safe_ratio(syn_full_union, real_full_union)

null_exact = float(null["exact_active_windows_seen_in_reference_rate"])
syn_exact = float(syn_cal["exact_active_windows_seen_in_reference_rate"])

null_dup = float(null["duplicate_member_rate_among_query_active_windows"])
syn_dup = float(syn_cal["duplicate_member_rate_among_query_active_windows"])

null_repeat_after_first = float(null["repeat_after_first_rate_among_query_active_windows"])
syn_repeat_after_first = float(syn_cal["repeat_after_first_rate_among_query_active_windows"])

# Thresholds are deliberately written out for auditability.
UNION_RATIO_BLOCK_THRESHOLD = 2.0
UNION_ABS_DIFF_BLOCK_THRESHOLD = 0.25
DUPLICATE_EXCESS_BLOCK_THRESHOLD = 0.10
DUPLICATE_RATIO_BLOCK_THRESHOLD = 1.25
EXACT_EXCESS_WARNING_THRESHOLD = 0.05

union_ratio_block = bool(
    np.isfinite(legacy_union_ratio)
    and legacy_union_ratio >= UNION_RATIO_BLOCK_THRESHOLD
)

union_absdiff_block = bool(
    np.isfinite(syn_full_union)
    and np.isfinite(real_full_union)
    and (syn_full_union - real_full_union) >= UNION_ABS_DIFF_BLOCK_THRESHOLD
)

duplicate_excess = syn_dup - null_dup if np.isfinite(syn_dup) and np.isfinite(null_dup) else np.nan
duplicate_ratio = safe_ratio(syn_dup, null_dup)
duplicate_block = bool(
    np.isfinite(duplicate_excess)
    and np.isfinite(duplicate_ratio)
    and duplicate_excess >= DUPLICATE_EXCESS_BLOCK_THRESHOLD
    and duplicate_ratio >= DUPLICATE_RATIO_BLOCK_THRESHOLD
)

exact_excess = syn_exact - null_exact if np.isfinite(syn_exact) and np.isfinite(null_exact) else np.nan
exact_context_warning = bool(
    np.isfinite(exact_excess)
    and exact_excess >= EXACT_EXCESS_WARNING_THRESHOLD
)

# Main recommended decision: block if union drift or calibrated repetition remains unsafe.
recommended_sparse_public_release_block = bool(union_ratio_block or union_absdiff_block or duplicate_block)

decision = pd.DataFrame([{
    "decision_id": "q6_sparse_driver_null_calibrated_sequence_gate_v1",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "window_size_rows": WINDOW_SIZE,
    "sparse_driver_columns": n_bits,
    "real_test_sha256": sha256_file(REAL_TEST_PATH),
    "synthetic_prior_public_sha256": sha256_file(SYN_PRIOR_PUBLIC_PATH),
    "real_windows_total": int(len(real_win)),
    "real_reference_windows": int(len(real_ref_win)),
    "real_holdout_windows": int(len(real_holdout_win)),
    "synthetic_windows_total": int(len(syn_win)),

    "real_full_active_union_rate": real_full_union,
    "synthetic_active_union_rate": syn_full_union,
    "synthetic_over_real_full_union_ratio": legacy_union_ratio,
    "union_ratio_block_threshold": UNION_RATIO_BLOCK_THRESHOLD,
    "union_abs_diff_block_threshold": UNION_ABS_DIFF_BLOCK_THRESHOLD,
    "union_ratio_block": union_ratio_block,
    "union_absdiff_block": union_absdiff_block,

    "real_vs_real_exact_seen_rate_null": null_exact,
    "synthetic_exact_seen_rate_calibrated": syn_exact,
    "synthetic_exact_seen_minus_null": exact_excess,
    "exact_seen_context_warning": exact_context_warning,
    "exact_seen_is_standalone_blocker": False,

    "real_vs_real_duplicate_member_rate_null": null_dup,
    "synthetic_duplicate_member_rate_calibrated": syn_dup,
    "synthetic_duplicate_member_minus_null": duplicate_excess,
    "synthetic_duplicate_member_over_null": duplicate_ratio,
    "duplicate_block": duplicate_block,

    "real_vs_real_repeat_after_first_rate_null": null_repeat_after_first,
    "synthetic_repeat_after_first_rate_calibrated": syn_repeat_after_first,

    "real_vs_real_hamming_p01_null": float(null["nearest_reference_hamming_p01"]),
    "synthetic_hamming_p01_calibrated": float(syn_cal["nearest_reference_hamming_p01"]),
    "hamming_zero_is_standalone_blocker": False,

    "recommended_sparse_public_release_block": recommended_sparse_public_release_block,
    "recommended_claim_interpretation": (
        "Block sparse-driver public release if union drift or calibrated repetition remains unsafe. "
        "Exact-match and zero-Hamming screens are reported relative to the real-vs-real null and are not "
        "standalone blockers for sparse binary signatures."
        if recommended_sparse_public_release_block else
        "Do not block solely on exact-match or zero-Hamming evidence; sparse-driver release would require "
        "additional policy review because calibrated sequence-risk blockers did not fire."
    ),
}])

decision_path = TABLE_DIR / "q6_sparse_driver_sequence_risk_null_calibrated_decision.csv"
decision.to_csv(decision_path, index=False)

# ------------------------------------------------------------------
# 6. Package-index addendum
# ------------------------------------------------------------------

files = [driver_set_path, diag_path, ratios_path, decision_path]
index_records = []
now = datetime.datetime.now(datetime.timezone.utc).isoformat()

for p in files:
    index_records.append({
        "artifact_group": "q6_sparse_driver_null_calibrated_sequence_risk",
        "artifact_role": "diagnostic_table",
        "path": str(p),
        "relative_path": str(p.relative_to(RUN_ROOT)),
        "filename": p.name,
        "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "created_utc": now,
        "claim_status": (
            "Null-calibrated Q6 sequence-risk evidence; exact-match and Hamming screens are contextual, "
            "not standalone blockers."
        ),
        "review_note": (
            "Adds real-vs-real null calibration for sparse-driver exact-window and Hamming proximity screens. "
            "Recommended release decision should rest on union drift and calibrated repetition if those remain unsafe."
        ),
    })

index_addendum = pd.DataFrame(index_records)
index_addendum_path = RUN_ROOT / "reports/artifact_package_index_Q6_SEQUENCE_NULL_CALIBRATION_ADDENDUM.csv"
index_addendum.to_csv(index_addendum_path, index=False)

# ------------------------------------------------------------------
# 7. Display outputs
# ------------------------------------------------------------------

display_df("Sparse-driver columns used", pd.DataFrame({"sparse_driver_column": driver_cols}), max_rows=40)
display_df("Null-calibrated Q6 sequence-risk diagnostics", diag, max_rows=10)
display_df("Synthetic-vs-null ratios", ratios, max_rows=20)
display_df("Null-calibrated recommended decision", decision, max_rows=5)
display_df("Package-index addendum", index_addendum, max_rows=10)

print("\nWrote:")
print(" ", driver_set_path)
print(" ", diag_path)
print(" ", ratios_path)
print(" ", decision_path)
print(" ", index_addendum_path)

print("\nSend me these displayed tables, especially:")
print("  1. Null-calibrated Q6 sequence-risk diagnostics")
print("  2. Synthetic-vs-null ratios")
print("  3. Null-calibrated recommended decision")