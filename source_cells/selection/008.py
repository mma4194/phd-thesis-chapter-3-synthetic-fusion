# %% STRONG.5 — RQ1 role-assignment protocol, ambiguity queue, and ownership audit

role_path = PATHS.get("role_ownership")
if role_path is None:
    warnings.warn("No iot_role_ownership.csv found; role audit will be limited to Q6 registry and driver ledger.")
    role_df = pd.DataFrame()
else:
    role_df = pd.read_csv(role_path)

rules = pd.DataFrame([
    {"precedence": 1, "rule": "explicit sparse-driver contract", "primary_owner": "driver_event_target", "evidence_source": "Cell 12.e / driver contract", "test_usage": "TEST audit only", "rationale": "Rare event drivers must not be reclassified as ordinary binary states."},
    {"precedence": 2, "rule": "canonical time and telemetry_in_sec helpers", "primary_owner": "conditioning_meta", "evidence_source": "Cell 4.5 taxonomy", "test_usage": "no TEST selection", "rationale": "Conditioning axes are not generated CPS feature claims."},
    {"precedence": 3, "rule": "observability/staleness name patterns: obs, present, active, stale, staleness, mask", "primary_owner": "mask_observability_meta or staleness_meta", "evidence_source": "Cell 4.5 meta-owner resolver", "test_usage": "TEST audit only", "rationale": "Reporting and staleness are target processes, not missing-value nuisances."},
    {"precedence": 4, "rule": "no TRAIN+VAL numeric support", "primary_owner": "excluded_no_dev_support / excluded_low_dev_support / excluded_non_numeric", "evidence_source": "TRAIN+VAL finite-rate checks", "test_usage": "TEST not used for inclusion", "rationale": "Prevents TEST-only feature resurrection."},
    {"precedence": 5, "rule": "state-value-like and binary on TRAIN+VAL", "primary_owner": "binary_state_target", "evidence_source": "TRAIN+VAL support", "test_usage": "TEST audit only", "rationale": "Binary operational state gets binary branch ownership."},
    {"precedence": 6, "rule": "state-value-like but nonbinary on TRAIN+VAL", "primary_owner": "continuous_value_target", "evidence_source": "TRAIN+VAL support", "test_usage": "TEST audit only", "rationale": "Ordinal or numerical state values use continuous/ordinal treatment."},
    {"precedence": 7, "rule": "suffix __value", "primary_owner": "continuous_value_target", "evidence_source": "semantic name rule + TRAIN+VAL numeric support", "test_usage": "TEST audit only", "rationale": "Sensor-like values use observed-only Q1/Q2 scoring."},
    {"precedence": 8, "rule": "suffix __state and binary on TRAIN+VAL", "primary_owner": "binary_state_target", "evidence_source": "semantic name rule + TRAIN+VAL support", "test_usage": "TEST audit only", "rationale": "State booleans use binary branch metrics."},
    {"precedence": 9, "rule": "residual numeric binary", "primary_owner": "binary_state_target", "evidence_source": "TRAIN+VAL support", "test_usage": "TEST audit only", "rationale": "Fallback only after driver/meta/exclusion rules."},
    {"precedence": 10, "rule": "residual numeric nonbinary", "primary_owner": "continuous_value_target", "evidence_source": "TRAIN+VAL support", "test_usage": "TEST audit only", "rationale": "Fallback continuous owner after higher-precedence exclusions."},
])
write_csv(rules, STRONG_TABLES / "role_assignment_protocol_rules.csv")

summary_rows = []
ambiguity_rows = []
if len(role_df):
    col_col = pick_col(role_df, ["col", "column", "feature", "name"], required=True)
    owner_col = pick_col(role_df, ["primary_owner", "owner_role", "role", "owner"], required=True)
    role_df[col_col] = role_df[col_col].astype(str)
    role_df[owner_col] = role_df[owner_col].astype(str)
    dup = role_df.groupby(col_col)[owner_col].nunique().reset_index(name="n_unique_primary_owners")
    dup_bad = dup[dup["n_unique_primary_owners"] != 1]
    counts = role_df[owner_col].value_counts(dropna=False).reset_index()
    counts.columns = ["primary_owner", "count"]
    counts["audit"] = "owner count from iot_role_ownership.csv"
    summary_rows.extend(counts.to_dict("records"))
    summary_rows.append({"primary_owner": "__duplicate_owner_columns__", "count": int(len(dup_bad)), "audit": "must be zero"})
    # Ambiguity heuristics for reviewer queue.
    reason_col = pick_col(role_df, ["reason", "assignment_reason", "owner_reason"], required=False)
    top_role_col = pick_col(role_df, ["top_role", "assigned_top_role"], required=False)
    for _, r in role_df.iterrows():
        c = str(r[col_col])
        lc = c.lower()
        owner = str(r[owner_col])
        reason = str(r.get(reason_col, "")) if reason_col else ""
        flags = []
        if "battery" in lc or "voltage" in lc or "linkquality" in lc or "rssi" in lc or "lqi" in lc:
            flags.append("telemetry_name")
        if any(k in lc for k in ["obs", "present", "active", "stale", "staleness", "mask"]):
            flags.append("observability_or_mask_name")
        if "event" in lc or "events_in_sec" in lc or "trigger" in lc:
            flags.append("event_or_driver_name")
        if lc.endswith("__state") or "__state__" in lc:
            flags.append("state_name")
        if lc.endswith("__value"):
            flags.append("value_name")
        if "residual" in reason or "but_nonbinary" in reason or "low_dev" in reason or len(flags) >= 2:
            ambiguity_rows.append({
                "col": c,
                "primary_owner": owner,
                "top_role": str(r.get(top_role_col, "")) if top_role_col else "",
                "reason": reason,
                "ambiguity_flags": "|".join(flags),
                "review_action": "confirm precedence rule and document if manuscript examples need contested-case evidence",
            })
    # Continuous source/final reconciliation.
    cont_selection_path = PATHS.get("continuous_selection")
    if cont_selection_path is not None:
        cont_sel = pd.read_csv(cont_selection_path)
        cont_col = pick_col(cont_sel, ["col", "column", "target", "feature", "name"], required=False)
        if cont_col:
            selected = set(cont_sel[cont_col].astype(str))
            owned = set(role_df.loc[role_df[owner_col].astype(str).str.contains("continuous", case=False, na=False), col_col].astype(str))
            recon = pd.DataFrame([{
                "quantity": "continuous_selection_rows", "count": len(selected),
                "interpretation": "source/selection-side continuous targets"
            }, {
                "quantity": "continuous_role_owned_rows", "count": len(owned),
                "interpretation": "role-ownership continuous targets"
            }, {
                "quantity": "selected_minus_owned", "count": len(selected - owned),
                "interpretation": "explain if nonzero; likely source/final-owned discrepancy"
            }, {
                "quantity": "owned_minus_selected", "count": len(owned - selected),
                "interpretation": "explain if nonzero; likely schema/selection discrepancy"
            }])
            write_csv(recon, STRONG_TABLES / "role_assignment_continuous_count_reconciliation.csv")
            write_csv(pd.DataFrame({"selected_minus_owned": sorted(selected-owned)}), STRONG_TABLES / "role_assignment_continuous_selected_minus_owned.csv")
            write_csv(pd.DataFrame({"owned_minus_selected": sorted(owned-selected)}), STRONG_TABLES / "role_assignment_continuous_owned_minus_selected.csv")
else:
    q6_registry = pd.read_csv(PATHS["q6_registry"])
    reg_col = pick_col(q6_registry, ["col", "column", "feature", "name"], required=True)
    reg_role = pick_col(q6_registry, ["role_group", "owner_role", "role", "branch", "category", "group"], required=False)
    if reg_role:
        counts = q6_registry[reg_role].astype(str).value_counts().reset_index()
        counts.columns = ["primary_owner", "count"]
        counts["audit"] = "limited public-registry role count"
        summary_rows.extend(counts.to_dict("records"))

summary = pd.DataFrame(summary_rows)
ambiguity = pd.DataFrame(ambiguity_rows).sort_values(["ambiguity_flags", "col"]) if ambiguity_rows else pd.DataFrame(columns=["col", "primary_owner", "top_role", "reason", "ambiguity_flags", "review_action"])
write_csv(summary, STRONG_TABLES / "role_assignment_owner_summary.csv")
write_csv(ambiguity, STRONG_TABLES / "role_assignment_ambiguity_review_queue.csv")

note = """# Role-assignment reproducibility note

The role assignment is rule-based and TRAIN+VAL-only for eligibility. TEST values are audit-only. The priority order is recorded in `role_assignment_protocol_rules.csv`.

Use `role_assignment_ambiguity_review_queue.csv` to choose examples for the manuscript: battery/link-quality telemetry, observability masks, event-driver columns, and state/value ambiguities.

A positive RQ1 claim should be framed as: reproducible owner assignment plus zero ownership violations, not merely zero violations after assignment.
"""
write_text(note, STRONG_DIR / "role_assignment_reproducibility_note.md")

print("Role protocol rules:", len(rules))
print("Ambiguity queue rows:", len(ambiguity))
summary.head(20)
