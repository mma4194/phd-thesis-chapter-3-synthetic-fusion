
# %% THESIS.REV.2 — baseline split/budget facts: remove the "if recorded in ledger" hedge
# Output:
#   reports/review_fix_outputs/baseline_budget_split_facts_from_contracts.csv
#   reports/review_fix_outputs/baseline_table23_replacement.csv
#   reports/review_fix_outputs/baseline_raw_audit_field_inventory.csv

FAIL_ON_UNKNOWN_VAL_USE = True


def flatten_dict(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            kk = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(v, kk))
    elif isinstance(d, list):
        # keep lists compact; do not explode long rows
        out[prefix] = json.dumps(d, sort_keys=True)[:1000]
    else:
        out[prefix] = d
    return out


def read_json_any(patterns, label):
    hits = find_files(patterns, roots=[REPORT_DIR_FIX, ARTIFACT_DIR_FIX])
    hits = [p for p in hits if "QUARANTINE" not in str(p).upper()]
    if not hits:
        raise FileNotFoundError(f"No {label} JSON found for patterns={patterns}")
    # Prefer canonical THESIS contract under artifacts/contracts, then reports.
    hits = sorted(hits, key=lambda p: (0 if "artifacts/contracts" in str(p) and "THESIS" in p.name else 1, len(str(p))))
    p = hits[0]
    return p, json.loads(p.read_text(encoding="utf-8"))


def read_csv_optional(patterns):
    hits = find_files(patterns, roots=[REPORT_DIR_FIX, ARTIFACT_DIR_FIX])
    hits = [p for p in hits if "QUARANTINE" not in str(p).upper()]
    if not hits:
        return None, pd.DataFrame()
    p = sorted(hits, key=lambda x: (0 if "/reports/" in str(x) else 1, len(str(x))))[0]
    try:
        return p, pd.read_csv(p)
    except Exception:
        return p, pd.DataFrame()


def first_flat(flat, keys, default=None):
    keys_l = [k.lower() for k in keys]
    for k, v in flat.items():
        kl = k.lower()
        if any(target in kl for target in keys_l):
            return v
    return default


def bool_or_unknown(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1"}: return True
        if s in {"false", "no", "0"}: return False
    return None

baseline_specs = [
    {
        "comparator": "CTGAN",
        "patterns_json": ["cell17_1*ctgan*contract*.json", "*ctgan*baseline*contract*.json"],
        "patterns_audit": ["cell17_1_ctgan_baseline_run_audit.csv", "*ctgan*run_audit*.csv"],
        "counted_scope": "IoT continuous compact tabular",
        "status_text": "Disclosed; not the best external row in the counted table.",
    },
    {
        "comparator": "TabDDPM",
        "patterns_json": ["cell17_2*tabddpm*contract*.json", "*tabddpm*baseline*contract*.json"],
        "patterns_audit": ["cell17_2_tabddpm_baseline_run_audit.csv", "*tabddpm*run_audit*.csv"],
        "counted_scope": "IoT continuous compact, protocol core, and public-release core tabular",
        "status_text": "Best external comparator in three fully compatible tabular rows.",
    },
    {
        "comparator": "TimeGAN",
        "patterns_json": ["cell17_3*timegan*contract*.json", "*timegan*baseline*contract*.json"],
        "patterns_audit": ["cell17_3_timegan_baseline_run_audit.csv", "*timegan*run_audit*.csv"],
        "counted_scope": "Compact CPS sequence; protocol-sequence temporal",
        "status_text": "Compact sequence counted; protocol-sequence row partial and excluded from headline counts.",
    },
]

rows = []
inventory_rows = []
unknown_val = []

for spec in baseline_specs:
    contract_path, contract = read_json_any(spec["patterns_json"], spec["comparator"])
    audit_path, audit_df = read_csv_optional(spec["patterns_audit"])
    flat = flatten_dict(contract)

    train_fit = bool_or_unknown(first_flat(flat, ["TRAIN_real_values_used_for_fitting", "train_values_used_for_fitting", "train_real_values_used"]))
    val_selection = bool_or_unknown(first_flat(flat, ["VAL_real_values_used_for_selection", "val_values_used_for_selection", "val_real_values_used_for_selection"]))
    test_used = bool_or_unknown(first_flat(flat, ["TEST_real_values_used", "test_values_used"], default=None))
    test_materialization = bool_or_unknown(first_flat(flat, ["TEST_real_values_used_for_materialization", "test_values_used_for_materialization"], default=None))
    selection_done_here = bool_or_unknown(first_flat(flat, ["selection_done_here"], default=None))

    if val_selection is None:
        unknown_val.append(spec["comparator"])
        val_text = "VAL role not recorded in baseline contract; disclose this rather than using a conditional hedge."
    elif val_selection is False and (selection_done_here is False or selection_done_here is None):
        val_text = "No VAL selection/calibration; fixed TRAIN-only budget."
    elif val_selection is True:
        val_text = "VAL-only selection/calibration recorded; no TEST tuning."
    else:
        val_text = "No VAL real-value selection recorded; verify if any non-real VAL calibration is used."

    if test_used is False and test_materialization is False:
        test_text = "Final compatible-scope audit only; no TEST fitting, selection, or materialization."
    elif test_used is False:
        test_text = "No TEST real values used by generator; TEST length/scope may be used for audit alignment only."
    else:
        test_text = "CHECK: contract indicates possible TEST use; do not claim no TEST tuning until resolved."

    budget_keys = []
    for k, v in flat.items():
        kl = k.lower()
        if any(tok in kl for tok in ["epoch", "iteration", "batch_size", "hidden_dim", "embedding_dim", "diffusion_steps", "sampling_steps", "seq_len", "seq_stride", "timeout", "seed", "runtime"]):
            if isinstance(v, (str, int, float, bool)) and str(v) not in {"", "None", "nan"}:
                budget_keys.append(f"{k.split('.')[-1]}={v}")
    budget_text = "; ".join(dict.fromkeys(budget_keys))[:800]

    if audit_df is not None and len(audit_df):
        for col in audit_df.columns:
            if any(tok in col.lower() for tok in ["epoch", "iteration", "batch", "runtime", "elapsed", "scope", "status", "success", "fail", "seed", "coverage"]):
                vals = sorted(audit_df[col].dropna().astype(str).unique())[:12]
                inventory_rows.append({
                    "comparator": spec["comparator"],
                    "audit_file": rel_to_run(audit_path),
                    "field": col,
                    "unique_values_preview": " | ".join(vals),
                    "n_unique": int(audit_df[col].dropna().astype(str).nunique()),
                })

    rows.append({
        "comparator": spec["comparator"],
        "counted_compatible_scope_s": spec["counted_scope"],
        "training_data": "TRAIN only" if train_fit is True else f"CHECK train_fit={train_fit}",
        "val_use": val_text,
        "test_use": test_text,
        "counted_status": spec["status_text"],
        "budget_fields_from_contract": budget_text,
        "contract_file": rel_to_run(contract_path),
        "audit_file": rel_to_run(audit_path) if audit_path else "",
        "train_fit_flag": train_fit,
        "val_real_values_used_for_selection_flag": val_selection,
        "test_real_values_used_flag": test_used,
        "test_real_values_used_for_materialization_flag": test_materialization,
        "selection_done_here_flag": selection_done_here,
    })

facts = pd.DataFrame(rows)
write_csv(facts, PATCH_DIR / "baseline_budget_split_facts_from_contracts.csv")
write_csv(pd.DataFrame(inventory_rows), PATCH_DIR / "baseline_raw_audit_field_inventory.csv")

replacement = facts[["comparator", "counted_compatible_scope_s", "training_data", "val_use", "test_use", "counted_status", "budget_fields_from_contract"]].copy()
write_csv(replacement, PATCH_DIR / "baseline_table23_replacement.csv")

print(replacement.to_string(index=False))

if unknown_val and FAIL_ON_UNKNOWN_VAL_USE:
    raise RuntimeError(f"VAL-use facts unknown for {unknown_val}; inspect contracts/ledgers before manuscript update.")
if (facts["test_real_values_used_flag"] == True).any() or (facts["test_real_values_used_for_materialization_flag"] == True).any():
    raise RuntimeError("At least one baseline contract indicates TEST use. Do not claim no TEST tuning until resolved.")
print("Baseline split/budget facts are ledger-backed; Table 23 can remove `if recorded in ledger`.")
