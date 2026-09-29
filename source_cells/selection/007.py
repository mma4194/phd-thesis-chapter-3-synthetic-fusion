# %% STRONG.4 — Materialize conservative public candidate and release composition by policy
# Recommended strong-THESIS behavior: if the sparse-driver group is blocked by STRONG.3,
# public release becomes timestamp + router + Zigbee only.

syn_public_existing = pd.read_parquet(PATHS["syn_public_existing"])
q6_registry = pd.read_csv(PATHS["q6_registry"])
q6_dropped = pd.read_csv(PATHS["q6_dropped"]) if PATHS.get("q6_dropped") else pd.DataFrame()
driver_ledger = pd.read_csv(PATHS["driver_ledger"])
syn_scientific_cols_n = len(pd.read_parquet(PATHS["syn_scientific"]).columns) if PATHS.get("syn_scientific") else np.nan

reg_col = pick_col(q6_registry, ["col", "column", "feature", "name", "target"], required=True, label="Q6 registry feature column")
reg_role = pick_col(q6_registry, ["role_group", "owner_role", "role", "branch", "category", "group"], required=False, label="Q6 registry role column")

sequence_decision = read_json_any(STRONG_MANIFESTS / "q6_conservative_sequence_gate_decision.json")
driver_cols = [c for c in sequence_decision.get("driver_columns", []) if c in syn_public_existing.columns]
block_sparse = sequence_decision.get("sparse_group_decision") == "block_sparse_driver_public_release"

index_like = [c for c in syn_public_existing.columns if str(c).lower().startswith(("index", "__index", "unnamed"))]
if block_sparse:
    conservative_keep_cols = [c for c in syn_public_existing.columns if c not in set(driver_cols) and c not in set(index_like)]
else:
    conservative_keep_cols = [c for c in syn_public_existing.columns if c not in set(index_like)]

syn_public_conservative = syn_public_existing[conservative_keep_cols].copy()
conservative_parquet = STRONG_SYNTH / "CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet"
syn_public_conservative.to_parquet(conservative_parquet, index=False)

# Build conservative registry and drop ledger.
if reg_role:
    reg_driver_mask = q6_registry[reg_col].astype(str).isin(driver_cols) | q6_registry[reg_role].astype(str).str.contains(r"driver|event|sparse", case=False, regex=True, na=False)
else:
    reg_driver_mask = q6_registry[reg_col].astype(str).isin(driver_cols)

conservative_registry = q6_registry.loc[q6_registry[reg_col].astype(str).isin(conservative_keep_cols)].copy()
conservative_registry["conservative_q6_sequence_gate"] = sequence_decision.get("sparse_group_decision")
conservative_registry["conservative_public_release_candidate"] = True
conservative_registry["sequence_gate_value_mutated"] = False

new_drop_rows = []
for c in driver_cols if block_sparse else []:
    old = q6_registry.loc[q6_registry[reg_col].astype(str).eq(c)].head(1).to_dict("records")
    base = old[0] if old else {reg_col: c, (reg_role or "role_group"): "iot_event_drivers"}
    base.update({
        "col": c,
        "role_group": base.get("role_group", base.get(reg_role, "iot_event_drivers")),
        "drop_reason": "blocked_by_conservative_q6_sparse_driver_sequence_risk_gate",
        "public_release_candidate": False,
        "sequence_gate_value_mutated": False,
        "sequence_gate_block_reasons": "|".join(sequence_decision.get("block_reasons", [])),
    })
    new_drop_rows.append(base)
conservative_new_drops = pd.DataFrame(new_drop_rows)
if len(q6_dropped):
    conservative_dropped = pd.concat([q6_dropped, conservative_new_drops], ignore_index=True, sort=False)
else:
    conservative_dropped = conservative_new_drops

# Role count helper.
def _role_of_col(c: str) -> str:
    if reg_role and c in set(q6_registry[reg_col].astype(str)):
        vals = q6_registry.loc[q6_registry[reg_col].astype(str).eq(c), reg_role].astype(str).tolist()
        if vals:
            return vals[0]
    s = str(c).lower()
    if "router" in s:
        return "protocol_router"
    if "zigbee" in s or s.startswith("zb"):
        return "protocol_zigbee"
    if c in set(driver_cols):
        return "iot_event_drivers"
    if "time" in s or "timestamp" in s or s in {"sec", "seconds"}:
        return "time"
    return "other"

orig_cols = [c for c in syn_public_existing.columns if c not in set(index_like)]
cons_cols = list(syn_public_conservative.columns)

def count_role(cols: Sequence[str], pattern: str) -> int:
    return sum(1 for c in cols if re.search(pattern, _role_of_col(c), flags=re.I) or re.search(pattern, str(c), flags=re.I))

# Absolute-Fano sensitivity from corrected driver ledger.
status_col = pick_col(driver_ledger, ["driver_corrected_ratio_battery_status", "status"], required=False)
ledger_col = pick_col(driver_ledger, ["col", "column", "feature", "name"], required=True)

def grade_ratio(x: Any) -> str:
    try:
        r = float(x)
    except Exception:
        return "fatal"
    if not np.isfinite(r):
        return "fatal"
    if r >= 3.0 or r <= 1.0 / 3.0:
        return "fatal"
    if r >= 1.5 or r <= 2.0 / 3.0:
        return "warning"
    return "pass"

def grade_abs_fano(x: Any, warn: float, fatal: float) -> str:
    try:
        v = abs(float(x))
    except Exception:
        return "fatal"
    if not np.isfinite(v):
        return "fatal"
    if v >= fatal:
        return "fatal"
    if v >= warn:
        return "warning"
    return "pass"

def grade_ks(x: Any, warn: float, fatal: float) -> str:
    try:
        v = float(x)
    except Exception:
        return "fatal"
    if not np.isfinite(v):
        return "fatal"
    if v >= fatal:
        return "fatal"
    if v >= warn:
        return "warning"
    return "pass"

order = {"pass": 0, "warning": 1, "fatal": 2}
def worst_status(vals: Iterable[str]) -> str:
    return max([str(v).lower() for v in vals], key=lambda z: order.get(z, 2))

# Use existing gate columns if present; otherwise recompute where possible.
abs_rows = []
for _, r in driver_ledger.iterrows():
    gates = []
    for gate_col, source_col, mode in [
        ("rate_ratio_gate", "event_rate_ratio", "ratio"),
        ("burst_count_ratio_gate", "burst_count_ratio", "ratio"),
        ("interarrival_ks_gate", "interarrival_ks", "ks_inter"),
        ("duration_ks_gate", "duration_ks", "ks_dur"),
    ]:
        if gate_col in driver_ledger.columns and str(r.get(gate_col, "")).lower() in order:
            gates.append(str(r.get(gate_col)).lower())
        elif source_col in driver_ledger.columns:
            if mode == "ratio":
                gates.append(grade_ratio(r.get(source_col)))
            elif mode == "ks_inter":
                gates.append(grade_ks(r.get(source_col), 0.50, 0.80))
            else:
                gates.append(grade_ks(r.get(source_col), 0.20, 0.50))
    fano_abs_col = "windowed_fano_absdiff" if "windowed_fano_absdiff" in driver_ledger.columns else None
    if fano_abs_col:
        gates.append(grade_abs_fano(r.get(fano_abs_col), ABS_FANO_WARN_THRESHOLD, ABS_FANO_FATAL_THRESHOLD))
    else:
        # Fall back to ratio gate if absolute Fano is unavailable; row will be marked incomplete below.
        if "windowed_fano_ratio_gate" in driver_ledger.columns:
            gates.append(str(r.get("windowed_fano_ratio_gate", "fatal")).lower())
        elif "windowed_fano_ratio" in driver_ledger.columns:
            gates.append(grade_ratio(r.get("windowed_fano_ratio")))
        else:
            gates.append("fatal")
    abs_rows.append({"col": str(r[ledger_col]), "absolute_fano_policy_status": worst_status(gates), "absolute_fano_available": bool(fano_abs_col)})
abs_policy = pd.DataFrame(abs_rows)
abs_counts = abs_policy["absolute_fano_policy_status"].value_counts().to_dict()

# Sensitivity grid.
grid_rows = []
if "windowed_fano_absdiff" in driver_ledger.columns:
    warn_values = [0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
    fatal_values = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00]
    for warn_t in warn_values:
        for fatal_t in fatal_values:
            if fatal_t <= warn_t:
                continue
            rows = []
            for _, r in driver_ledger.iterrows():
                gates = []
                for gate_col, source_col, mode in [
                    ("rate_ratio_gate", "event_rate_ratio", "ratio"),
                    ("burst_count_ratio_gate", "burst_count_ratio", "ratio"),
                    ("interarrival_ks_gate", "interarrival_ks", "ks_inter"),
                    ("duration_ks_gate", "duration_ks", "ks_dur"),
                ]:
                    if gate_col in driver_ledger.columns and str(r.get(gate_col, "")).lower() in order:
                        gates.append(str(r.get(gate_col)).lower())
                    elif source_col in driver_ledger.columns:
                        gates.append(grade_ratio(r.get(source_col)) if mode == "ratio" else grade_ks(r.get(source_col), 0.50, 0.80) if mode == "ks_inter" else grade_ks(r.get(source_col), 0.20, 0.50))
                gates.append(grade_abs_fano(r.get("windowed_fano_absdiff"), warn_t, fatal_t))
                rows.append(worst_status(gates))
            vc = pd.Series(rows).value_counts().to_dict()
            grid_rows.append({
                "abs_fano_warn_threshold": warn_t,
                "abs_fano_fatal_threshold": fatal_t,
                "pass": int(vc.get("pass", 0)),
                "warning": int(vc.get("warning", 0)),
                "fatal": int(vc.get("fatal", 0)),
                "release_eligible_drivers": int(vc.get("pass", 0) + vc.get("warning", 0)),
                "public_total_if_only_fatal_excluded": int(len(orig_cols) - len(driver_cols) + vc.get("pass", 0) + vc.get("warning", 0)),
                "matches_supplement_6_14_6": bool(vc.get("pass", 0) == 6 and vc.get("warning", 0) == 14 and vc.get("fatal", 0) == 6),
            })
grid = pd.DataFrame(grid_rows)

ratio_counts = driver_ledger[status_col].astype(str).str.lower().value_counts().to_dict() if status_col else {}
composition = pd.DataFrame([
    {
        "policy": "existing_ratio_scale_policy",
        "sparse_driver_pass": int(ratio_counts.get("pass", 0)),
        "sparse_driver_warning": int(ratio_counts.get("warning", 0)),
        "sparse_driver_fatal": int(ratio_counts.get("fatal", 0)),
        "sparse_driver_public_kept": len(driver_cols),
        "timestamp_public_kept": count_role(orig_cols, r"time|timestamp"),
        "router_public_kept": count_role(orig_cols, r"router"),
        "zigbee_public_kept": count_role(orig_cols, r"zigbee|zb"),
        "public_logical_columns": len(orig_cols),
        "excluded_logical_columns": int(syn_scientific_cols_n - len(orig_cols)) if np.isfinite(syn_scientific_cols_n) else np.nan,
        "claim_interpretation": "prior warning-governed candidate; not recommended after sequence-risk gate if blocked",
    },
    {
        "policy": f"absolute_fano_sensitivity_warn_{ABS_FANO_WARN_THRESHOLD}_fatal_{ABS_FANO_FATAL_THRESHOLD}",
        "sparse_driver_pass": int(abs_counts.get("pass", 0)),
        "sparse_driver_warning": int(abs_counts.get("warning", 0)),
        "sparse_driver_fatal": int(abs_counts.get("fatal", 0)),
        "sparse_driver_public_kept": int(abs_counts.get("pass", 0) + abs_counts.get("warning", 0)),
        "timestamp_public_kept": count_role(orig_cols, r"time|timestamp"),
        "router_public_kept": count_role(orig_cols, r"router"),
        "zigbee_public_kept": count_role(orig_cols, r"zigbee|zb"),
        "public_logical_columns": int(len(orig_cols) - len(driver_cols) + abs_counts.get("pass", 0) + abs_counts.get("warning", 0)),
        "excluded_logical_columns": int(syn_scientific_cols_n - (len(orig_cols) - len(driver_cols) + abs_counts.get("pass", 0) + abs_counts.get("warning", 0))) if np.isfinite(syn_scientific_cols_n) else np.nan,
        "claim_interpretation": "threshold-sensitivity composition; use grid to show fragility, not as final release claim",
    },
    {
        "policy": "conservative_sequence_risk_gate_recommended",
        "sparse_driver_pass": int(ratio_counts.get("pass", 0)),
        "sparse_driver_warning": int(ratio_counts.get("warning", 0)),
        "sparse_driver_fatal": int(ratio_counts.get("fatal", 0)),
        "sparse_driver_public_kept": 0 if block_sparse else len(driver_cols),
        "timestamp_public_kept": count_role(cons_cols, r"time|timestamp"),
        "router_public_kept": count_role(cons_cols, r"router"),
        "zigbee_public_kept": count_role(cons_cols, r"zigbee|zb"),
        "public_logical_columns": len(cons_cols),
        "excluded_logical_columns": int(syn_scientific_cols_n - len(cons_cols)) if np.isfinite(syn_scientific_cols_n) else np.nan,
        "claim_interpretation": "recommended strong-submission public candidate if sparse sequence risk blocks",
    },
])

write_csv(conservative_registry, STRONG_TABLES / "q6_conservative_public_column_registry.csv")
write_csv(conservative_dropped, STRONG_TABLES / "q6_conservative_public_dropped_columns.csv")
write_csv(composition, STRONG_TABLES / "q6_release_composition_by_policy.csv")
write_csv(abs_policy, STRONG_TABLES / "q6_sparse_driver_absolute_fano_policy_status.csv")
write_csv(grid, STRONG_TABLES / "q6_sparse_driver_absolute_fano_sensitivity_grid.csv")

manifest = {
    "created_utc": utc_now(),
    "conservative_public_parquet": rel_to_run(conservative_parquet),
    "conservative_public_sha256": sha256_file(conservative_parquet),
    "existing_public_parquet": rel_to_run(PATHS["syn_public_existing"]),
    "sequence_gate_decision": sequence_decision.get("sparse_group_decision"),
    "driver_columns_dropped_by_sequence_gate": driver_cols if block_sparse else [],
    "conservative_public_logical_columns": len(cons_cols),
    "existing_public_logical_columns": len(orig_cols),
    "values_mutated": False,
}
write_json(manifest, STRONG_MANIFESTS / "q6_conservative_public_candidate_manifest.json")

print("Wrote conservative public artifact:", conservative_parquet)
print("Existing public logical columns:", len(orig_cols))
print("Conservative public logical columns:", len(cons_cols))
print("Sparse drivers dropped by conservative gate:", len(driver_cols) if block_sparse else 0)
composition
