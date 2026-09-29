FOCUSED_CHECKS = focused_joint_review(CFG_INSPECT, OUTPUT_DIR)
print(FOCUSED_CHECKS.to_string(index=False) if not FOCUSED_CHECKS.empty else "No focused columns available.")
manifest_path = OUTPUT_DIR / "inspection_manifest.json"
manifest = json.loads(manifest_path.read_text())
for name in ["original_notebook_source_review.json", "original_notebook_cell_inventory.csv"]:
    p = OUTPUT_DIR / name
    manifest["outputs"].append({"file": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
manifest["source_notebook_sha256"] = SOURCE_REVIEW["sha256"]
manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
print("\nOpen the report:", OUTPUT_DIR / "feature_review.html")
try:
    from IPython.display import display, FileLink
    display(FileLink(str((OUTPUT_DIR / "feature_review.html").relative_to(Path.cwd()))))
except (ImportError, ValueError):
    pass
