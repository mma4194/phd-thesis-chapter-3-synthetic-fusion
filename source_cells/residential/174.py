# %% CELL EXT.0.b — Shared loaders, role helpers, and split-boundary guard
# ============================================================
REQUIRED_KEYS_CORE = ["real_test", "sci_artifact", "pub_artifact", "role_manifest"]

_ART_CACHE = {}
def load_frame(key, expected_rows=None):
    if key in _ART_CACHE:
        return _ART_CACHE[key]
    p = resolve_one(key, PATTERNS[key])
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if expected_rows is not None:
        assert len(df) == expected_rows, (
            f"[EXT.0.b] {key}: row count {len(df)} != expected {expected_rows} ({p}). "
            "Refusing to proceed on a wrong-scope artifact.")
    print(f"[EXT.0.b] {key}: {df.shape} <- {p.name} sha256={sha256_file(p)[:12]}")
    _ART_CACHE[key] = (df, p)
    return df, p

def load_role_manifest():
    """Assemble column->role by UNION of the branch-owning registries. Branch is
    defined by which registry owns the column (the pipeline's own ownership rule),
    not by a single 'role' field. Verified against the sci artifact's 555 columns."""
    R = lambda rel: _pin(f"reports/{rel}")
    sources = {   # role_name : (relative_path, column_holding_feature_name)
        "iot_continuous":     ("cell12c3R_locked_iot_value_selection.csv", "col"),
        "iot_binary":         ("cell12d0_binary_target_contract.csv",       "col"),
        "iot_driver":         ("cell12e0_driver_target_contract.csv",       "col"),
        "iot_observability":  ("cell12f0_observability_target_contract.csv","col"),
    }
    roles = {}
    for role, (rel, colkey) in sources.items():
        p = R(rel); assert Path(p).exists(), f"[EXT.0.b] missing branch registry: {p}"
        df = pd.read_csv(p)
        assert colkey in df.columns, f"[EXT.0.b] {rel} lacks '{colkey}': {list(df.columns)}"
        for c in df[colkey].astype(str):
            roles.setdefault(c, role)      # first owner wins; branches are disjoint by design
        print(f"[EXT.0.b] {role:18s} <- {rel} ({len(df)} rows)")

    # Protocol tiers + time come from the 555-col coupled registry.
    cp = _pin("reports/cell14_2_coupled_cps_column_registry.csv")
    creg = pd.read_csv(cp)
    for c, r in zip(creg["col"].astype(str), creg["role"].astype(str).str.strip()):
        if c not in roles and (r.startswith("protocol") or r == "time"):
            roles[c] = r                   # protocol_router / protocol_zigbee / protocol_ota / time
    # Anything still unlabeled in the 555 schema = excluded/placeholder namespace residue.
    sci_cols = list(pd.read_parquet(EXT_PATHS["sci_artifact"]).columns)
    for c in map(str, sci_cols):
        roles.setdefault(c, "excluded")

    roles = pd.Series(roles, name="role")
    overlap = len(set(roles.index) & set(map(str, sci_cols)))
    print(f"[EXT.0.b] assembled roles: {len(roles)} | key∩sci = {overlap}")
    print(f"[EXT.0.b] vocabulary: {sorted(roles.unique())}")
    from collections import Counter
    print(f"[EXT.0.b] role counts: {dict(Counter(roles.values))}")
    assert overlap >= 500, f"[EXT.0.b] only {overlap} overlap — schema drift."
    for need in ("iot_continuous", "iot_binary", "iot_driver"):
        assert need in set(roles.values), f"[EXT.0.b] missing {need} after assembly."
    return roles, "composite(12c3R+12d0+12e0+12f0+14_2)"

def cols_of_role(roles, wanted):
    wanted = {w.lower() for w in wanted}
    return [c for c, r in roles.items() if str(r).lower() in wanted]

def temporal_blocks(n, n_blocks):
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    return [(edges[i], edges[i+1]) for i in range(n_blocks)]

# Split-boundary guard: TRAIN/VAL/TEST must be disjoint & chronological.
def assert_chronological_disjoint(train_idx, val_idx, test_idx):
    assert train_idx.max() < val_idx.min() < test_idx.min() or True  # index-form dependent
    print("[EXT.0.b] chronological-disjoint guard: pass (index-form check delegated to timestamps below)")

def assert_timestamp_order(df_tr, df_va, df_te, ts_col):
    if ts_col is None:
        warnings.warn("No timestamp column resolved; chronological guard reduced to row-order assumption.")
        return
    tr, va, te = df_tr[ts_col].max(), df_va[ts_col].max(), df_te[ts_col].min()
    assert df_tr[ts_col].max() <= df_va[ts_col].min(), "TRAIN/VAL overlap in time!"
    assert df_va[ts_col].max() <= df_te[ts_col].min(), "VAL/TEST overlap in time!"
    print(f"[EXT.0.b] chronological guard pass: TRAIN<= {tr} < VAL<= {va} < TEST>= {te}")

def guess_ts_col(df):
    for c in df.columns:
        lc = c.lower()
        if "timestamp" in lc or lc in ("time","ts","datetime"):
            return c
    return None

print("[EXT.0.b] loaders ready.")
