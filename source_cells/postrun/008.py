
# %% THESIS.REV.5 — patched package indexes/checksums + anonymity grep
# Replaces the broken generic Cell 209 behavior. This does not rely on generic filenames at Path(".").
# Output:
#   reports/artifact_package_index_REVIEW_FIXES_PATCHED.csv
#   reports/artifact_package_checksums_REVIEW_FIXES_PATCHED.csv
#   reports/ARTIFACT_README_STUDY_THESIS_REVIEW_FIXES_ADDENDUM.md
#   reports/review_fix_outputs/anonymity_grep_findings.csv

# Collect existing artifact index if available.
base_index_path = REPORT_DIR_FIX / "artifact_package_index.csv"
base_index_raw = pd.read_csv(base_index_path) if base_index_path.exists() else pd.DataFrame()

# Keep resolved paths internally only; do NOT write absolute build paths into reviewer-facing indexes.
def resolve_for_build(path_value):
    raw = str(path_value or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    if raw.startswith("smartstar_homeA_study_thesis_transfer_artifact/") and SMARTSTAR_ROOT_FIX:
        return Path(SMARTSTAR_ROOT_FIX) / raw.split("/", 1)[1]
    return RUN_ROOT / raw

def sanitize_for_review(path_value):
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        try:
            return str(p.resolve().relative_to(RUN_ROOT.resolve())).replace(os.sep, "/")
        except Exception:
            pass
        if SMARTSTAR_ROOT_FIX:
            try:
                return "smartstar_homeA_study_thesis_transfer_artifact/" + str(p.resolve().relative_to(Path(SMARTSTAR_ROOT_FIX).resolve())).replace(os.sep, "/")
            except Exception:
                pass
        return "external_path_redacted/" + p.name
    return raw.replace(os.sep, "/")

patch_files = []
for p in sorted(PATCH_DIR.rglob("*")):
    if p.is_file():
        rel_path = str(p.relative_to(RUN_ROOT)).replace(os.sep, "/") if str(p.resolve()).startswith(str(RUN_ROOT.resolve())) else "reports/review_fix_outputs/" + p.name
        patch_files.append({
            "artifact_id": "review_fix_" + re.sub(r"[^A-Za-z0-9]+", "_", p.name).strip("_"),
            "scope": "review_fix_evidence",
            "path": rel_path,
            "_resolved_path": str(p),
            "description": "Reviewer-requested audit/fix output generated after final notebook run.",
            "release_status": "documentation",
            "reviewer_use": "Audit revised claims and manuscript tables.",
            "required": True,
            "exists": True,
            "size_bytes": int(p.stat().st_size),
            "sha256": sha256_file(p),
        })
patch_index = pd.DataFrame(patch_files)

# Include Smart* index rows if generated.
smart_index_path = REPORT_DIR_FIX / "smartstar_package_index_rows.csv"
if smart_index_path.exists():
    smart_index = pd.read_csv(smart_index_path)
    if "exists" not in smart_index.columns:
        smart_index["exists"] = True
    if "size_bytes" not in smart_index.columns and "bytes" in smart_index.columns:
        smart_index["size_bytes"] = smart_index["bytes"]
    smart_index["_resolved_path"] = smart_index["path"].map(lambda x: str(resolve_for_build(x)) if resolve_for_build(x) is not None else "")
else:
    smart_index = pd.DataFrame()

# Add internal resolved paths to base index without exposing them.
base_index = base_index_raw.copy()
if len(base_index):
    if "path" in base_index.columns:
        base_index["_resolved_path"] = base_index["path"].map(lambda x: str(resolve_for_build(x)) if resolve_for_build(x) is not None else "")
        base_index["path"] = base_index["path"].map(sanitize_for_review)
    else:
        base_index["path"] = ""
        base_index["_resolved_path"] = ""

# Harmonize columns.
all_cols = []
for df in [base_index, patch_index, smart_index]:
    for col in df.columns:
        if col not in all_cols:
            all_cols.append(col)
for name in ["base_index", "patch_index", "smart_index"]:
    df = locals()[name]
    for col in all_cols:
        if col not in df.columns:
            df[col] = ""
    locals()[name] = df[all_cols]

patched_raw = pd.concat([base_index, patch_index, smart_index], ignore_index=True) if all_cols else patch_index
# Reviewer-facing index must not include absolute local paths.
patched_out = patched_raw.drop(columns=["_resolved_path"], errors="ignore").copy()
if "path" in patched_out.columns:
    patched_out["path"] = patched_out["path"].map(sanitize_for_review)
patched_path = REPORT_DIR_FIX / "artifact_package_index_REVIEW_FIXES_PATCHED.csv"
write_csv(patched_out, patched_path)

checksum_rows = []
for _, r in patched_raw.iterrows():
    resolved = str(r.get("_resolved_path", "")).strip()
    if not resolved:
        resolved_p = resolve_for_build(r.get("path", ""))
    else:
        resolved_p = Path(resolved)
    if resolved_p is None or not resolved_p.exists() or not resolved_p.is_file():
        continue
    checksum_rows.append({
        "file_path": sanitize_for_review(r.get("path", "")),
        "sha256": sha256_file(resolved_p),
        "bytes": int(resolved_p.stat().st_size),
    })
checksums = pd.DataFrame(checksum_rows).drop_duplicates(subset=["file_path", "sha256"])
checksums_path = REPORT_DIR_FIX / "artifact_package_checksums_REVIEW_FIXES_PATCHED.csv"
write_csv(checksums, checksums_path)

readme_addendum = f"""# Review-fix artifact addendum

Generated: {datetime.now(timezone.utc).isoformat()}

This addendum indexes the post-run reviewer-fix outputs:

1. Window-vs-pooled continuous diagnostic reconciliation.
2. Baseline TRAIN/VAL/TEST and budget facts copied from submitted contracts/ledgers.
3. Smart* Home A smoke-test accounting closure and package-index rows.
4. Optional Q6 sparse-driver copy/replay/routine-risk screen.
5. Anonymity grep findings.

Primary patched index: `{patched_path.name}`
Primary patched checksum file: `{checksums_path.name}`

These outputs support manuscript/supplement edits only; they do not create new positive realism, Q4, Q5 utility, or privacy claims.
"""
readme_path = REPORT_DIR_FIX / "ARTIFACT_README_STUDY_THESIS_REVIEW_FIXES_ADDENDUM.md"
readme_path.write_text(readme_addendum, encoding="utf-8")
print(readme_addendum)

# Optional materialized package copy. Use only when you are ready to build the review artifact directory.
MATERIALIZE_PACKAGE = os.environ.get("CPS_MATERIALIZE_PACKAGE", "0").strip() == "1"
PACKAGE_ROOT_FIX = Path(os.environ.get("CPS_REVIEW_PACKAGE_ROOT", str(RUN_ROOT / "synthetic_fusion_study_thesis_submission_package_PATCHED"))).expanduser().resolve()
if MATERIALIZE_PACKAGE:
    PACKAGE_ROOT_FIX.mkdir(parents=True, exist_ok=True)
    for _, row in patched_raw.iterrows():
        resolved = str(row.get("_resolved_path", "")).strip()
        src = Path(resolved) if resolved else resolve_for_build(row.get("path", ""))
        if src is None or not src.exists() or not src.is_file():
            continue
        rel = Path(sanitize_for_review(row.get("path", src.name)))
        if str(rel).startswith("external_path_redacted"):
            rel = Path("external_path_redacted") / src.name
        dst = PACKAGE_ROOT_FIX / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print("Materialized patched package at <CPS_REVIEW_PACKAGE_ROOT>")
else:
    print("Package not materialized. Set CPS_MATERIALIZE_PACKAGE=1 and CPS_REVIEW_PACKAGE_ROOT=/path/to/package to copy files.")

# Anonymity grep over generated review-facing text/metadata files. Binary/parquet files are skipped.
patterns = [
    r"c\.c21126547", r"/shared/home1", r"/tmp/c\.c", r"Mohammed", r"Cardiff", r"Rana", r"Perera", r"Documents/Smart_Star_Dataset",
]
regex = re.compile("|".join(patterns), re.IGNORECASE)
scan_roots = [REPORT_DIR_FIX, ARTIFACT_DIR_FIX]
if SMARTSTAR_ROOT_FIX:
    scan_roots.append(Path(SMARTSTAR_ROOT_FIX))
text_exts = {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".tex", ".bib", ".log"}
findings = []
for root in scan_roots:
    root = Path(root)
    if not root.exists():
        continue
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in text_exts:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(txt.splitlines(), start=1):
            if regex.search(line):
                findings.append({
                    "file": sanitize_for_review(str(p)),
                    "line": line_no,
                    "matched_text_preview": regex.sub("<IDENTITY_OR_PATH_TOKEN>", line)[:300],
                })
findings_df = pd.DataFrame(findings)
write_csv(findings_df, PATCH_DIR / "anonymity_grep_findings.csv")
if len(findings_df):
    print("WARNING: anonymity/path findings exist. Inspect review_fix_outputs/anonymity_grep_findings.csv before submission.")
else:
    print("No anonymity/path findings in scanned review-facing text files.")
