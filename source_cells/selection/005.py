# %% STRONG.3 — Q6 sparse-driver sequence-risk gate and driver-contribution audit
# This cell upgrades the earlier diagnostic from "disclose" to a conservative Q6 release gate.

real_test = pd.read_parquet(PATHS["real_test"])
syn_public_existing = pd.read_parquet(PATHS["syn_public_existing"])
q6_registry = pd.read_csv(PATHS["q6_registry"])
driver_ledger = pd.read_csv(PATHS["driver_ledger"])

# -------- detect sparse-driver public columns --------
reg_col = pick_col(q6_registry, ["col", "column", "feature", "name", "target"], required=True, label="Q6 registry feature column")
reg_role = pick_col(q6_registry, ["role_group", "owner_role", "role", "branch", "category", "group"], required=False, label="Q6 registry role column")

ledger_col = pick_col(driver_ledger, ["col", "column", "feature", "name", "target"], required=True, label="driver ledger feature column")
ledger_elig = pick_col(driver_ledger, ["driver_release_eligible_corrected", "release_eligible", "q6_release_eligible"], required=False, label="driver ledger release eligibility")

ledger_release_cols = set()
if ledger_elig:
    ledger_release_cols = set(driver_ledger.loc[bool_series(driver_ledger[ledger_elig]), ledger_col].astype(str))
else:
    status_col = pick_col(driver_ledger, ["driver_corrected_ratio_battery_status", "status"], required=False)
    if status_col:
        ledger_release_cols = set(driver_ledger.loc[~driver_ledger[status_col].astype(str).str.lower().eq("fatal"), ledger_col].astype(str))

if reg_role:
    role_mask = q6_registry[reg_role].astype(str).str.contains(r"driver|event|sparse", case=False, regex=True, na=False)
else:
    role_mask = pd.Series(False, index=q6_registry.index)
ledger_mask = q6_registry[reg_col].astype(str).isin(ledger_release_cols)
driver_cols = q6_registry.loc[role_mask | ledger_mask, reg_col].astype(str).tolist()
# Keep only overlapping, non-index columns.
driver_cols = [c for c in dict.fromkeys(driver_cols) if c in syn_public_existing.columns and c in real_test.columns and not str(c).lower().startswith(("index", "__index", "unnamed"))]

if not driver_cols:
    raise RuntimeError("No public sparse-driver columns detected. Check q6_registry and driver ledger column names.")

n = min(len(real_test), len(syn_public_existing))
n_blocks = n // WINDOW_ROWS
if n_blocks < 2:
    raise RuntimeError(f"Not enough rows for {WINDOW_ROWS}-row sequence windows: n={n}")
n_use = n_blocks * WINDOW_ROWS


def window_any_binary(df: pd.DataFrame, cols: Sequence[str], n_use: int, n_blocks: int, window_rows: int) -> np.ndarray:
    arr = df.iloc[:n_use][list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    arr = (arr > 0).astype(np.uint8)
    return arr.reshape(n_blocks, window_rows, len(cols)).max(axis=1).astype(np.uint8)


def row_key(row: np.ndarray) -> bytes:
    return bytes(np.asarray(row, dtype=np.uint8).tolist())


def min_hamming_to_real(syn_mat: np.ndarray, real_mat: np.ndarray, chunk_syn: int = 512, chunk_real: int = 2048) -> np.ndarray:
    if len(syn_mat) == 0 or len(real_mat) == 0:
        return np.array([], dtype=float)
    syn_mat = syn_mat.astype(np.uint8)
    real_mat = real_mat.astype(np.uint8)
    mins = []
    for i in range(0, len(syn_mat), chunk_syn):
        s = syn_mat[i:i + chunk_syn]
        dmin = np.full(len(s), np.inf, dtype=float)
        for j in range(0, len(real_mat), chunk_real):
            r = real_mat[j:j + chunk_real]
            dist = np.not_equal(s[:, None, :], r[None, :, :]).mean(axis=2)
            dmin = np.minimum(dmin, dist.min(axis=1))
        mins.append(dmin)
    return np.concatenate(mins) if mins else np.array([], dtype=float)

real_w = window_any_binary(real_test, driver_cols, n_use, n_blocks, WINDOW_ROWS)
syn_w = window_any_binary(syn_public_existing, driver_cols, n_use, n_blocks, WINDOW_ROWS)
real_union = real_w.max(axis=1)
syn_union = syn_w.max(axis=1)
real_active = real_w[real_union > 0]
syn_active = syn_w[syn_union > 0]

real_counter = Counter(row_key(r) for r in real_active)
syn_counter = Counter(row_key(r) for r in syn_active)
real_set = set(real_counter)
syn_active_keys = [row_key(r) for r in syn_active]
exact_mask = np.array([k in real_set for k in syn_active_keys], dtype=bool)
exact_seen_n = int(exact_mask.sum())
exact_seen_rate = float(exact_seen_n / max(1, len(syn_active_keys)))
repeated_rate = float(1.0 - len(set(syn_active_keys)) / max(1, len(syn_active_keys)))

if len(syn_active) > MAX_NN_SYN_WINDOWS:
    rng = np.random.default_rng(20260708)
    nn_idx = np.sort(rng.choice(len(syn_active), size=MAX_NN_SYN_WINDOWS, replace=False))
    syn_for_nn = syn_active[nn_idx]
    nn_sampled = True
else:
    syn_for_nn = syn_active
    nn_sampled = False
nn = min_hamming_to_real(syn_for_nn, real_active)

union_ratio = float(syn_union.mean() / max(float(real_union.mean()), 1e-12))

block_reasons = []
warning_reasons = []
if exact_seen_rate > 0:
    block_reasons.append("exact_synthetic_active_window_seen_in_real")
if len(nn) and float(np.min(nn)) <= 0.0:
    block_reasons.append("nearest_real_active_window_hamming_min_is_zero")
if len(nn) and float(np.quantile(nn, 0.01)) <= 0.0:
    block_reasons.append("nearest_real_active_window_hamming_p01_is_zero")
if union_ratio > UNION_RATIO_WARN_HI or union_ratio < UNION_RATIO_WARN_LO:
    warning_reasons.append("active_union_rate_ratio_outside_warning_band")
    # In a conservative release gate, major union drift also blocks group-level routine release.
    block_reasons.append("active_union_rate_ratio_outside_conservative_band")
if repeated_rate > REPEATED_ACTIVE_RATE_WARN_HI:
    warning_reasons.append("repeated_synthetic_active_window_rate_high")
    block_reasons.append("repeated_synthetic_active_window_rate_high")

sparse_group_decision = "block_sparse_driver_public_release" if (SEQUENCE_GATE_BLOCKS_SPARSE_GROUP and block_reasons) else "allow_sparse_driver_public_release_with_warnings"

# Driver contribution ledger for exact/zero-distance windows.
exact_syn_vecs = syn_active[exact_mask] if len(syn_active) else np.empty((0, len(driver_cols)), dtype=np.uint8)
contrib_rows = []
for j, c in enumerate(driver_cols):
    contrib_rows.append({
        "col": c,
        "active_in_exact_replay_windows_n": int(exact_syn_vecs[:, j].sum()) if len(exact_syn_vecs) else 0,
        "share_of_exact_replay_windows_with_driver_active": float(exact_syn_vecs[:, j].mean()) if len(exact_syn_vecs) else 0.0,
        "syn_active_window_driver_rate": float(syn_active[:, j].mean()) if len(syn_active) else 0.0,
        "real_active_window_driver_rate": float(real_active[:, j].mean()) if len(real_active) else 0.0,
        "syn_all_window_driver_rate": float(syn_w[:, j].mean()) if len(syn_w) else 0.0,
        "real_all_window_driver_rate": float(real_w[:, j].mean()) if len(real_w) else 0.0,
        "driver_active_rate_delta_all_windows": float(syn_w[:, j].mean() - real_w[:, j].mean()) if len(real_w) and len(syn_w) else np.nan,
    })
contrib = pd.DataFrame(contrib_rows).sort_values(
    ["active_in_exact_replay_windows_n", "syn_all_window_driver_rate"], ascending=[False, False]
).reset_index(drop=True)

# Exact pattern table: store vector hash and active columns, capped for package size.
pattern_rows = []
for k in sorted(set(real_counter) & set(syn_counter), key=lambda x: syn_counter[x], reverse=True)[:200]:
    bits = np.frombuffer(k, dtype=np.uint8)
    active_cols = [driver_cols[i] for i, v in enumerate(bits) if int(v) > 0]
    pattern_rows.append({
        "pattern_sha256": hashlib.sha256(k).hexdigest(),
        "synthetic_active_window_count": int(syn_counter[k]),
        "real_active_window_count": int(real_counter[k]),
        "active_driver_count": int(bits.sum()),
        "active_driver_cols": "|".join(active_cols),
    })
patterns = pd.DataFrame(pattern_rows)

diag = pd.DataFrame([
    {"diagnostic": "release_sparse_driver_columns", "value": len(driver_cols), "interpretation": "Retained public sparse-driver columns screened."},
    {"diagnostic": "window_rows", "value": WINDOW_ROWS, "interpretation": "Rows per contiguous active-window vector."},
    {"diagnostic": "windows_total", "value": n_blocks, "interpretation": "Aligned windows screened."},
    {"diagnostic": "real_active_union_window_rate", "value": float(real_union.mean()), "interpretation": "Fraction of real windows with at least one retained sparse-driver event."},
    {"diagnostic": "synthetic_active_union_window_rate", "value": float(syn_union.mean()), "interpretation": "Fraction of synthetic windows with at least one retained sparse-driver event."},
    {"diagnostic": "synthetic_to_real_union_rate_ratio", "value": union_ratio, "interpretation": "Group-level routine/union drift screen."},
    {"diagnostic": "exact_synthetic_active_windows_seen_in_real_n", "value": exact_seen_n, "interpretation": "Synthetic active windows exactly equal to at least one real active-window vector."},
    {"diagnostic": "exact_synthetic_active_windows_seen_in_real_rate", "value": exact_seen_rate, "interpretation": "Exact active-window replay rate among synthetic active windows."},
    {"diagnostic": "repeated_synthetic_active_window_rate", "value": repeated_rate, "interpretation": "Routine repetition among synthetic active windows."},
    {"diagnostic": "nearest_real_active_window_hamming_min", "value": float(np.min(nn)) if len(nn) else np.nan, "interpretation": "Minimum nearest-real Hamming distance over active windows."},
    {"diagnostic": "nearest_real_active_window_hamming_p01", "value": float(np.quantile(nn, 0.01)) if len(nn) else np.nan, "interpretation": "One-percentile nearest-real Hamming distance."},
    {"diagnostic": "nearest_real_active_window_hamming_median", "value": float(np.median(nn)) if len(nn) else np.nan, "interpretation": "Median nearest-real Hamming distance."},
    {"diagnostic": "nn_distance_sampled", "value": bool(nn_sampled), "interpretation": "Whether nearest-neighbor distances were computed on a deterministic sample."},
])

decision = {
    "created_utc": utc_now(),
    "policy": "conservative_q6_sparse_driver_sequence_risk_gate_v1",
    "sparse_group_decision": sparse_group_decision,
    "block_reasons": block_reasons,
    "warning_reasons": warning_reasons,
    "driver_columns_n": len(driver_cols),
    "driver_columns": driver_cols,
    "thresholds": {
        "exact_active_window_replay_blocks_if_rate_gt": 0.0,
        "nearest_real_hamming_min_blocks_if_eq": 0.0,
        "nearest_real_hamming_p01_blocks_if_eq": 0.0,
        "union_ratio_warning_hi": UNION_RATIO_WARN_HI,
        "union_ratio_warning_lo": UNION_RATIO_WARN_LO,
        "repeated_active_rate_warning_hi": REPEATED_ACTIVE_RATE_WARN_HI,
    },
    "diagnostics": {r["diagnostic"]: r["value"] for r in diag.to_dict("records")},
}

write_csv(diag, STRONG_TABLES / "q6_sparse_driver_sequence_risk_diagnostics.csv")
write_csv(contrib, STRONG_TABLES / "q6_sparse_driver_sequence_driver_contributions.csv")
write_csv(patterns, STRONG_TABLES / "q6_sparse_driver_exact_replay_patterns_top200.csv")
write_json(decision, STRONG_MANIFESTS / "q6_conservative_sequence_gate_decision.json")
write_text(
    "# Q6 conservative sparse-driver sequence-risk gate\n\n"
    f"Decision: `{sparse_group_decision}`\n\n"
    f"Block reasons: {', '.join(block_reasons) if block_reasons else 'none'}\n\n"
    f"Screened {len(driver_cols)} sparse-driver columns over {n_blocks} windows of {WINDOW_ROWS} rows.\n",
    STRONG_DIR / "q6_conservative_sequence_gate_decision.md",
)

print(diag.to_string(index=False))
print("\nQ6 sequence decision:", sparse_group_decision)
print("Block reasons:", block_reasons)
print("Top driver contributions to exact replay windows:")
contrib.head(10)
