
# %% THESIS.REV.4 — optional Q6 sparse-driver copy/replay/routine-risk screening
# This is NOT a privacy proof. It produces a small diagnostic table for the 23 release-eligible sparse drivers.
# Output:
#   reports/review_fix_outputs/q6_sparse_driver_sequence_safety_diagnostics.csv
#   reports/review_fix_outputs/q6_sparse_driver_sequence_safety_manifest.json

RUN_Q6_SEQUENCE_SAFETY = True
WINDOW_ROWS = 60          # 60 one-second rows per window
MAX_NN_SYN_WINDOWS = 5000 # full TEST has ~4259 minute windows, so this usually covers all.

if RUN_Q6_SEQUENCE_SAFETY:
    real_split_dir_env = _first_existing_path([os.environ.get("CPS_REAL_SPLIT_DIR", "")]) if os.environ.get("CPS_REAL_SPLIT_DIR", "") else None
    real_test_candidates = [
        os.environ.get("CPS_REAL_TEST_PARQUET", ""),
        SYN_DIR_FIX / "REAL_TEST_SPLIT.parquet",
    ]
    if real_split_dir_env is not None:
        real_test_candidates.append(real_split_dir_env / "REAL_TEST_SPLIT.parquet")
    real_test_path = _first_existing_path(real_test_candidates)
    syn_pub_path = _first_existing_path([
        os.environ.get("CPS_PUBLIC_PARQUET", ""),
        globals().get("CELL15_6_PUBLIC_CPS_PATH", ""),
        SYN_DIR_FIX / "CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet",
    ])
    if real_test_path is None or syn_pub_path is None:
        raise RuntimeError("Need REAL_TEST_SPLIT.parquet and CPS_SYNTHETIC_TEST_PUBLIC_Q6_MITIGATED.parquet for Q6 sequence-safety diagnostics.")

    real_df = pd.read_parquet(real_test_path)
    syn_df = pd.read_parquet(syn_pub_path)

    # Determine public sparse-driver columns. Prefer Q6 public registry; otherwise infer by excluding time/protocol columns.
    registry_path = find_one("cell15_6_q6_public_column_registry.csv", roots=[REPORT_DIR_FIX], required=False, label="Q6 public column registry")
    driver_cols = []
    if registry_path is not None:
        reg = pd.read_csv(registry_path)
        col_col = pick_col(reg, ["column", "feature", "name"], required=False, label="public registry column name")
        role_col = pick_col(reg, ["role", "owner", "group", "category"], required=False, label="public registry role/group")
        if col_col:
            if role_col:
                mask = reg[role_col].astype(str).str.contains("driver|sparse|event", case=False, na=False)
                driver_cols = reg.loc[mask, col_col].astype(str).tolist()
            if not driver_cols:
                # Fall through to inference from public columns.
                pass
    if not driver_cols:
        exclude_re = re.compile(r"(^sec|timestamp|time|router|zigbee|z_?wave|ota|wlan|wifi|ethernet|protocol)", re.I)
        driver_cols = [c for c in syn_df.columns if not exclude_re.search(str(c))]

    driver_cols = [c for c in driver_cols if c in syn_df.columns and c in real_df.columns]
    if len(driver_cols) == 0:
        raise RuntimeError("No overlapping sparse-driver columns found between real TEST and public synthetic candidate.")

    n = min(len(real_df), len(syn_df))
    n_blocks = n // WINDOW_ROWS
    if n_blocks < 2:
        raise RuntimeError(f"Not enough rows for {WINDOW_ROWS}-row windows: n={n}")
    n_use = n_blocks * WINDOW_ROWS

    def window_any_binary(df, cols):
        arr = df.loc[:n_use-1, cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        arr = (arr > 0).astype(np.uint8)
        return arr.reshape(n_blocks, WINDOW_ROWS, len(cols)).max(axis=1).astype(np.uint8)

    real_w = window_any_binary(real_df, driver_cols)
    syn_w = window_any_binary(syn_df, driver_cols)
    real_union = real_w.max(axis=1)
    syn_union = syn_w.max(axis=1)
    real_active = real_w[real_union > 0]
    syn_active = syn_w[syn_union > 0]

    def rows_as_set(a):
        return {bytes(row.tolist()) for row in a.astype(np.uint8)}

    real_set = rows_as_set(real_active)
    syn_active_keys = [bytes(row.tolist()) for row in syn_active.astype(np.uint8)]
    exact_seen = sum(1 for k in syn_active_keys if k in real_set)
    exact_seen_rate = exact_seen / max(1, len(syn_active_keys))
    repeated_rate = 1.0 - (len(set(syn_active_keys)) / max(1, len(syn_active_keys)))

    def min_hamming_to_real(syn_mat, real_mat, chunk=512):
        if len(syn_mat) == 0 or len(real_mat) == 0:
            return np.array([], dtype=float)
        syn_mat = syn_mat.astype(np.uint8)
        real_mat = real_mat.astype(np.uint8)
        mins = []
        for i in range(0, len(syn_mat), chunk):
            s = syn_mat[i:i+chunk]
            # Hamming count via XOR, chunked to avoid memory surprises.
            dmin = np.full(len(s), np.inf)
            for j in range(0, len(real_mat), 2048):
                r = real_mat[j:j+2048]
                dist = np.not_equal(s[:, None, :], r[None, :, :]).mean(axis=2)
                dmin = np.minimum(dmin, dist.min(axis=1))
            mins.append(dmin)
        return np.concatenate(mins)

    # This is usually small (minute windows), but keep a deterministic cap.
    if len(syn_active) > MAX_NN_SYN_WINDOWS:
        rng = np.random.default_rng(20260708)
        idx = np.sort(rng.choice(len(syn_active), size=MAX_NN_SYN_WINDOWS, replace=False))
        syn_for_nn = syn_active[idx]
    else:
        syn_for_nn = syn_active
    nn = min_hamming_to_real(syn_for_nn, real_active)

    diagnostics = pd.DataFrame([
        {"diagnostic": "release_sparse_driver_columns", "value": len(driver_cols), "interpretation": "Number of retained sparse-driver columns included in the Q6 sequence-risk screen."},
        {"diagnostic": "window_rows", "value": WINDOW_ROWS, "interpretation": "Rows per contiguous window; one-second TEST rows imply 60-second windows."},
        {"diagnostic": "windows_total", "value": n_blocks, "interpretation": "Total aligned windows screened."},
        {"diagnostic": "real_active_union_window_rate", "value": float(real_union.mean()), "interpretation": "Fraction of real windows with at least one retained sparse-driver event."},
        {"diagnostic": "synthetic_active_union_window_rate", "value": float(syn_union.mean()), "interpretation": "Fraction of synthetic windows with at least one retained sparse-driver event."},
        {"diagnostic": "synthetic_to_real_union_rate_ratio", "value": float(syn_union.mean() / max(real_union.mean(), 1e-12)), "interpretation": "Union-rate drift screen; not a privacy metric."},
        {"diagnostic": "exact_synthetic_active_windows_seen_in_real_rate", "value": float(exact_seen_rate), "interpretation": "Copy/replay screen over binary active-window vectors; not a formal privacy proof."},
        {"diagnostic": "repeated_synthetic_active_window_rate", "value": float(repeated_rate), "interpretation": "Routine repetition screen among synthetic active windows."},
        {"diagnostic": "nearest_real_active_window_hamming_min", "value": float(np.min(nn)) if len(nn) else np.nan, "interpretation": "Sequence proximity screen; lower means closer to some real active-window vector."},
        {"diagnostic": "nearest_real_active_window_hamming_p01", "value": float(np.quantile(nn, 0.01)) if len(nn) else np.nan, "interpretation": "Low-tail sequence proximity screen."},
        {"diagnostic": "nearest_real_active_window_hamming_median", "value": float(np.median(nn)) if len(nn) else np.nan, "interpretation": "Typical nearest-neighbor proximity over active windows."},
    ])
    out_diag = PATCH_DIR / "q6_sparse_driver_sequence_safety_diagnostics.csv"
    write_csv(diagnostics, out_diag)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Q6 sparse-driver copy/replay/routine-risk screen; not a formal privacy proof.",
        "real_test_path": rel_to_run(real_test_path),
        "synthetic_public_path": rel_to_run(syn_pub_path),
        "driver_columns_n": len(driver_cols),
        "window_rows": WINDOW_ROWS,
        "windows_total": int(n_blocks),
        "diagnostics_csv": rel_to_run(out_diag),
        "driver_columns": driver_cols,
    }
    write_json(manifest, PATCH_DIR / "q6_sparse_driver_sequence_safety_manifest.json")
    print(diagnostics.to_string(index=False))
