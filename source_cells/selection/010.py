# %% STRONG.7 — Baseline-fairness claim downgrade / fixed-budget diagnostic framing
# This addresses the reviewer concern that fixed external budgets + no VAL selection are not comparable to a VAL-selected portfolio.

baseline_rows = []
if PATHS.get("postrun_baseline_facts"):
    facts = pd.read_csv(PATHS["postrun_baseline_facts"])
else:
    facts = pd.DataFrame()

if PATHS.get("baseline_summary"):
    base = pd.read_csv(PATHS["baseline_summary"])
else:
    base = pd.DataFrame()

# Extract comparator facts if postrun facts exist; otherwise use contracts with best effort.
if len(facts):
    for _, r in facts.iterrows():
        comp = str(r.get("comparator", r.get("baseline", "unknown")))
        val_use = str(r.get("val_use", "unknown"))
        train_data = str(r.get("training_data", "unknown"))
        test_use = str(r.get("test_use", "unknown"))
        budget = str(r.get("budget_fields_from_contract", ""))
        no_val = "no val" in val_use.lower() or "fixed train" in val_use.lower()
        baseline_rows.append({
            "comparator": comp,
            "training_data": train_data,
            "val_use": val_use,
            "test_use": test_use,
            "budget_fields": budget,
            "selection_comparability_with_pipeline": "not comparable" if no_val else "partially comparable",
            "allowed_claim": "fixed-budget compatible-scope diagnostic probe; not method-superiority evidence" if no_val else "VAL-aware comparator evidence if scope is compatible",
            "headline_superiority_claim_allowed": False if no_val else True,
            "requires_val_tuned_reaudit_before_superiority_claim": True if no_val else False,
        })
else:
    # Scan available Cell 17 contracts/audits.
    contract_hits = find_files(["*ctgan*contract*.json", "*tabddpm*contract*.json", "*timegan*contract*.json", "*baseline*contract*.json", "*baseline*run_audit*.csv"], [REPORT_DIR, ARTIFACT_DIR])
    for p in contract_hits:
        comp = "CTGAN" if "ctgan" in p.name.lower() else "TabDDPM" if "tabddpm" in p.name.lower() else "TimeGAN" if "timegan" in p.name.lower() else "unknown"
        baseline_rows.append({
            "comparator": comp,
            "training_data": "see contract",
            "val_use": "unknown_from_scan",
            "test_use": "unknown_from_scan",
            "budget_fields": p.name,
            "selection_comparability_with_pipeline": "unknown; inspect contract before claiming fairness",
            "allowed_claim": "do not claim method superiority until VAL-selection parity is established",
            "headline_superiority_claim_allowed": False,
            "requires_val_tuned_reaudit_before_superiority_claim": True,
        })

claim_table = pd.DataFrame(baseline_rows)

# Reframe existing baseline comparison rows if available.
reframed_rows = []
if len(base):
    for _, r in base.iterrows():
        row = r.to_dict()
        row.update({
            "selection_comparability_caveat": "external baselines are fixed-budget/no-VAL unless a comparator-specific VAL ledger says otherwise",
            "allowed_claim_after_fix": "lower penalty in a fixed-budget compatible-scope diagnostic audit",
            "disallowed_claim_after_fix": "global superiority or controlled model-selection superiority",
            "headline_count_for_superiority": False,
            "headline_count_for_diagnostic_scope": True,
        })
        reframed_rows.append(row)
reframed = pd.DataFrame(reframed_rows)

write_csv(claim_table, STRONG_TABLES / "baseline_budget_asymmetry_claim_downgrade.csv")
write_csv(reframed, STRONG_TABLES / "baseline_fairness_reframed_rows.csv")
write_text(
    "# Baseline fairness interpretation after strong-review fix\n\n"
    "The baseline rows should be cited as fixed-budget compatible-scope diagnostics. "
    "Do not use the ratios as method-superiority evidence unless CTGAN/TabDDPM/TimeGAN receive a declared VAL-selection protocol comparable to the pipeline portfolio selection.\n",
    STRONG_DIR / "baseline_fairness_claim_downgrade_note.md",
)

claim_table
