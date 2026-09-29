OUTPUT_DIR, FEATURE_INVENTORY = inspect_parquet(CFG_INSPECT, EMBEDDED_M7_REFERENCE)
# Save the documentary source trace beside the sampled evidence.
(OUTPUT_DIR / "original_notebook_source_review.json").write_text(json.dumps(SOURCE_REVIEW, indent=2) + "\n")
pd.DataFrame(SOURCE_REVIEW["cell_inventory"]).to_csv(OUTPUT_DIR / "original_notebook_cell_inventory.csv", index=False)
print("\nCounts by observed representation:")
print(FEATURE_INVENTORY["observed_representation"].value_counts().to_string())
