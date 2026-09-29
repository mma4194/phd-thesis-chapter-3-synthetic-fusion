# %% STRONG.10 — Strong-fix manifest, package-index merge, and runbook

# Collect outputs from this notebook.
strong_files = []
for p in sorted(STRONG_DIR.rglob("*")):
    if p.is_file():
        strong_files.append({
            "relative_path": rel_to_run(p),
            "artifact_group": "thesis_strong_revision_fix",
            "description": p.name,
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        })
# Add conservative parquet explicitly if outside STRONG_DIR.
cons_parquet = STRONG_SYNTH / "CPS_SYNTHETIC_TEST_PUBLIC_Q6_CONSERVATIVE.parquet"
if cons_parquet.exists():
    strong_files.append({
        "relative_path": rel_to_run(cons_parquet),
        "artifact_group": "thesis_strong_revision_fix",
        "description": "conservative public candidate after Q6 sequence-risk gate",
        "sha256": sha256_file(cons_parquet),
        "bytes": cons_parquet.stat().st_size,
    })

strong_manifest = pd.DataFrame(strong_files).drop_duplicates(subset=["relative_path"]).sort_values("relative_path")
write_csv(strong_manifest, STRONG_MANIFESTS / "thesis_strong_fix_outputs_manifest.csv")

# Merge with existing review-fix package index if available; otherwise this becomes the package index.
base_index_path = PATHS.get("postrun_patched_index") or (REPORT_DIR / "artifact_package_index_REVIEW_FIXES_PATCHED.csv")
if base_index_path and Path(base_index_path).exists():
    base = pd.read_csv(base_index_path)
else:
    base = pd.DataFrame()

# Normalize to a broad, packaging-friendly schema.
idx_rows = []
if len(base):
    idx_rows.extend(base.to_dict("records"))
for _, r in strong_manifest.iterrows():
    idx_rows.append({
        "relative_path": r["relative_path"],
        "path": r["relative_path"],
        "artifact_group": r["artifact_group"],
        "description": r["description"],
        "sha256": r["sha256"],
        "bytes": int(r["bytes"]),
        "claim_supported": "strong revision / reviewer-response evidence",
    })
merged = pd.DataFrame(idx_rows)
# Remove exact duplicate rows by relative path/path.
key_col = "relative_path" if "relative_path" in merged.columns else "path"
if key_col in merged.columns:
    merged = merged.drop_duplicates(subset=[key_col], keep="last")

strong_index_path = REPORT_DIR / "artifact_package_index_STRONG_FIXES_PATCHED.csv"
write_csv(merged, strong_index_path)

runbook = f"""# How to run the strong STUDY-THESIS fixes

## Recommended order

1. Run `THESIS_FINAL_NOTEBOOK (1)(1).ipynb` through the final canonical run cell.
2. Run `SmartStar_HomeA_STUDY_THESIS_Governance_Smoke_Test_v3_REVIEW_SAFE(1).ipynb` through its checksum/package cells.
3. Run `THESIS_Postrun_Audit_Fix_Cells(1).ipynb` through `THESIS.REV.5`.
4. Run this notebook from top to bottom.
5. In the postrun notebook packaging section, set:

```python
PATCHED_INDEX = RUN_ROOT / "reports/artifact_package_index_STRONG_FIXES_PATCHED.csv"
```

then run the package materialization/sanitization/checksum/zip cells.

## Paste-in placement if you do not want a separate notebook

Paste cells `STRONG.0` through `STRONG.10` into `THESIS_Postrun_Audit_Fix_Cells(1).ipynb` immediately after `THESIS.REV.5 — patched package indexes/checksums + anonymity grep` and before the review-safe package materialization cell. Then set `PATCHED_INDEX` to `artifact_package_index_STRONG_FIXES_PATCHED.csv`.

## Principal output to use for the manuscript later

- Conservative Q6 artifact: `{rel_to_run(cons_parquet) if cons_parquet.exists() else '<not written>'}`
- Release composition table: `reports/thesis_strong_revision_fixes/tables/q6_release_composition_by_policy.csv`
- Sequence-risk diagnostics: `reports/thesis_strong_revision_fixes/tables/q6_sparse_driver_sequence_risk_diagnostics.csv`
- Driver contribution audit: `reports/thesis_strong_revision_fixes/tables/q6_sparse_driver_sequence_driver_contributions.csv`
- Role-assignment protocol: `reports/thesis_strong_revision_fixes/tables/role_assignment_protocol_rules.csv`
- C2ST primary-instrument row: `reports/thesis_strong_revision_fixes/tables/continuous_c2st_table13_replacement_row.csv`
- Baseline claim downgrade: `reports/thesis_strong_revision_fixes/tables/baseline_budget_asymmetry_claim_downgrade.csv`
- Q5 claim-scope recommendation: `reports/thesis_strong_revision_fixes/q5_claim_scope_recommendation.md`

## Decision rule

If `q6_conservative_sequence_gate_decision.json` says `block_sparse_driver_public_release`, the strong-submission public artifact is the conservative artifact, not the old 46-column candidate.
"""
write_text(runbook, STRONG_DIR / "THESIS_STRONG_REVISION_RUNBOOK.md")

print("Strong outputs manifest:", STRONG_MANIFESTS / "thesis_strong_fix_outputs_manifest.csv")
print("Strong package index:", strong_index_path)
print("Runbook:", STRONG_DIR / "THESIS_STRONG_REVISION_RUNBOOK.md")
print("Strong output files:", len(strong_manifest))

# Optional materialization into a package root. Usually leave this off and use the postrun packaging cells.
if MATERIALIZE_STRONG_PACKAGE:
    pkg = _as_path(STRONG_PACKAGE_ROOT)
    if pkg is None:
        raise RuntimeError("Set THESIS_STRONG_PACKAGE_ROOT when THESIS_MATERIALIZE_STRONG_PACKAGE=1")
    pkg.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in strong_manifest["relative_path"].astype(str):
        src = RUN_ROOT / rel
        if src.exists():
            dst = pkg / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    print("Copied strong outputs into package root:", pkg, "files:", len(copied))

strong_manifest.head(50)
