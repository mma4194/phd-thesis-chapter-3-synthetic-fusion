
# %% THESIS.REV.3 — Smart* accounting closure and package-index rows
# Requires SMARTSTAR_ARTIFACT_ROOT or SMARTSTAR_OUT to point to the completed Smart* smoke-test artifact directory.
# Output:
#   Smart*/tables/smartstar_feature_accounting_reconciled.csv
#   Smart*/manuscript/smartstar_table_s26_accounting_note.md
#   reports/review_fix_outputs/smartstar_feature_accounting_reconciled.csv
#   reports/smartstar_package_index_rows.csv
#   reports/smartstar_package_checksums.csv

if SMARTSTAR_ROOT_FIX is None or not Path(SMARTSTAR_ROOT_FIX).exists():
    raise RuntimeError("Set SMARTSTAR_ARTIFACT_ROOT=/path/to/smartstar_homeA_study_thesis_transfer_artifact, then rerun THESIS.REV.0 and this cell.")
SMART = Path(SMARTSTAR_ROOT_FIX).expanduser().resolve()

elig_path = find_one("feature_eligibility_train_val_only.csv", roots=[SMART], label="Smart* feature eligibility ledger")
test_path = find_one("test_evidence_q1_q2_q3_ledger.csv", roots=[SMART], label="Smart* TEST evidence ledger")

elig = read_csv_required(elig_path)
test = read_csv_required(test_path)

role_col_e = pick_col(elig, ["owner_role", "role"], label="eligibility role column")
elig_col = pick_col(elig, ["eligible"], label="eligibility boolean column")
feature_col_e = pick_col(elig, ["feature", "column", "name"], required=False, label="eligibility feature column") or elig.columns[0]
role_col_t = pick_col(test, ["owner_role", "role"], label="TEST role column")
status_col_t = pick_col(test, ["test_status", "readiness", "status"], label="TEST status column")
feature_col_t = pick_col(test, ["feature", "column", "name"], required=False, label="TEST feature column") or test.columns[0]

def norm_role(x):
    s = str(x).strip()
    return s

def norm_status(x):
    s = str(x).strip().lower().replace("-", "_")
    if "warn" in s: return "warning"
    if "fatal" in s or "fail" in s: return "fatal"
    if "pass" in s: return "pass"
    if "exclude" in s: return "excluded"
    return s or "unknown"

def to_bool_series(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "eligible"])

elig2 = elig.copy()
elig2["_role"] = elig2[role_col_e].map(norm_role)
elig2["_eligible"] = to_bool_series(elig2[elig_col])
test2 = test.copy()
test2["_role"] = test2[role_col_t].map(norm_role)
test2["_status"] = test2[status_col_t].map(norm_status)

assigned = elig2.groupby("_role").size().rename("assigned_role_owned_features")
eligible = elig2[elig2["_eligible"]].groupby("_role").size().rename("eligible_audited_features")
status = test2.pivot_table(index="_role", columns="_status", values=feature_col_t, aggfunc="count", fill_value=0)
for col in ["pass", "warning", "fatal", "excluded", "unknown"]:
    if col not in status.columns:
        status[col] = 0

acct = pd.concat([assigned, eligible, status[["pass", "warning", "fatal", "excluded", "unknown"]]], axis=1).fillna(0).astype(int)
acct["excluded_before_test_by_train_val_eligibility"] = acct["assigned_role_owned_features"] - acct["eligible_audited_features"]
acct["audited_status_total"] = acct["pass"] + acct["warning"] + acct["fatal"] + acct["excluded"] + acct["unknown"]
acct["accounting_closes"] = acct["eligible_audited_features"].eq(acct["audited_status_total"])
acct = acct.reset_index().rename(columns={"_role": "owner_role"}).sort_values("owner_role")

# Write results into both Smart* artifact and main patch directory.
smart_table_dir = SMART / "tables"; smart_table_dir.mkdir(parents=True, exist_ok=True)
smart_man_dir = SMART / "manuscript"; smart_man_dir.mkdir(parents=True, exist_ok=True)
write_csv(acct, smart_table_dir / "smartstar_feature_accounting_reconciled.csv")
write_csv(acct, PATCH_DIR / "smartstar_feature_accounting_reconciled.csv")

binary_row = acct[acct["owner_role"].astype(str).str.contains("binary", case=False, na=False)]
if not binary_row.empty:
    br = binary_row.iloc[0]
    note = (
        f"Role-inventory counts are pre-eligibility role-owner counts; pass/warning/fatal triples are post-eligibility TEST-audit counts. "
        f"In the Smart* smoke test, {int(br['excluded_before_test_by_train_val_eligibility'])} of {int(br['assigned_role_owned_features'])} binary role-owned features "
        f"are excluded by TRAIN/VAL eligibility before TEST scoring, leaving {int(br['eligible_audited_features'])} audited binary features "
        f"with {int(br['pass'])}/{int(br['warning'])}/{int(br['fatal'])} pass/warning/fatal."
    )
else:
    note = "No binary role row was found; inspect smartstar_feature_accounting_reconciled.csv before updating Table S26."

(smart_man_dir / "smartstar_table_s26_accounting_note.md").write_text(note + "\n", encoding="utf-8")
(PATCH_DIR / "smartstar_table_s26_accounting_note.md").write_text(note + "\n", encoding="utf-8")
print(note)

if not acct["accounting_closes"].all():
    bad = acct[~acct["accounting_closes"]]
    raise RuntimeError("Smart* accounting does not close for some roles:\n" + bad.to_string(index=False))

# Build package checksum/index rows for all Smart* smoke-test files.
checksum_rows = []
for p in sorted(SMART.rglob("*")):
    if p.is_file():
        checksum_rows.append({
            "relative_path_within_smartstar_artifact": str(p.relative_to(SMART)).replace(os.sep, "/"),
            "review_package_relative_path": "smartstar_homeA_study_thesis_transfer_artifact/" + str(p.relative_to(SMART)).replace(os.sep, "/"),
            "bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
            "claim_supported": "Smart* procedural governance-transfer smoke test only; no numerical transfer or Q4/Q5 claim.",
        })
smart_checksums = pd.DataFrame(checksum_rows)
write_csv(smart_checksums, REPORT_DIR_FIX / "smartstar_package_checksums.csv")
write_csv(smart_checksums, SMART / "manifests" / "smartstar_package_checksums.csv")

index_rows = smart_checksums.rename(columns={"review_package_relative_path": "path"}).copy()
index_rows.insert(0, "artifact_id", ["smartstar_transfer_" + re.sub(r"[^A-Za-z0-9]+", "_", x).strip("_")[:80] for x in index_rows["relative_path_within_smartstar_artifact"]])
index_rows.insert(1, "scope", "smartstar_governance_transfer_smoke_test")
index_rows["description"] = "Public Smart* Home A smoke-test artifact file. Supports procedural re-instantiation only."
index_rows["release_status"] = "public_source_derived_smoke_test_artifact"
index_rows["reviewer_use"] = "Audit Smart* role ownership, split discipline, evidence ledgers, unsupported Q4/Q5 declarations, and checksums."
index_rows["required"] = index_rows["relative_path_within_smartstar_artifact"].isin([
    "ARTIFACT_README_STUDY_THESIS_SMARTSTAR.md",
    "manifests/transfer_policy_manifest_v0.json",
    "manifests/raw_file_manifest.csv",
    "manifests/role_owner_manifest.csv",
    "manifests/split_manifest.csv",
    "ledgers/feature_eligibility_train_val_only.csv",
    "ledgers/candidate_selection_val_only_ledger.csv",
    "ledgers/test_evidence_q1_q2_q3_ledger.csv",
    "tables/branch_readiness_summary.csv",
    "tables/q4_scope_status.csv",
    "tables/q5_scope_status.csv",
    "tables/smartstar_feature_accounting_reconciled.csv",
])
write_csv(index_rows, REPORT_DIR_FIX / "smartstar_package_index_rows.csv")

print(acct.to_string(index=False))
print("Smart* package rows:", len(index_rows), "required:", int(index_rows["required"].sum()))
