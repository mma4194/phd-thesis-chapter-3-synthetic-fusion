from pathlib import Path

DATA_PATH = Path("/path/to/input/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet")
RUN_ROOT = Path("/path/to/input/cps_synth_v4_4_7_32_BINARY_V6_Q3V5_STUDY_FINAL_20260708_035653")
OUTPUT_PARENT = Path.cwd() / "THESIS_feature_inspection_outputs"

SAMPLE_COUNT = 10                    # 8 or 10 recommended; samples use distinct row positions.
FEATURE_SCOPE = "all"                # "all", "iot", "m7", or "custom"
CUSTOM_FEATURES = ["iot__events_total", "iot__events_entity_unique", "iot__events_update_any"]
PRINT_EVERY_FEATURE = True           # HTML/TXT exports always include every selected feature.
SEED = 20260907
COLUMN_GROUP_SIZE = 8                # Lower this if RAM is constrained.
SOURCE_MODE = "canonical_fullgrid"  # Alternative: "persisted_train_split", with DATA_PATH changed explicitly.
TIME_COLUMN = None                   # auto: sec for canonical; sec_epoch_s__canon for persisted TRAIN.
STRICT_SOURCE_CONTRACT = True        # Check original row count, source schema width and TRAIN time origin.
COMPUTE_SOURCE_SHA256 = False         # Optional full-file hash; can be expensive for a large Parquet.

CFG_INSPECT = {
    "data_path": DATA_PATH, "run_root": RUN_ROOT, "output_parent": OUTPUT_PARENT,
    "sample_count": SAMPLE_COUNT, "scope": FEATURE_SCOPE, "custom_features": CUSTOM_FEATURES,
    "print_every_feature": PRINT_EVERY_FEATURE, "seed": SEED, "column_group_size": COLUMN_GROUP_SIZE,
    "source_mode": SOURCE_MODE, "time_column": TIME_COLUMN, "strict_source_contract": STRICT_SOURCE_CONTRACT,
    "compute_source_sha256": COMPUTE_SOURCE_SHA256,
    "expected_total_rows": 1277694, "expected_columns": 740, "n_train": 766616,
    "expected_first_epoch": 1750446300,
    "expected_input_sha256": "a4d7446daba0e46dab2dc21607a99c5a734a5366a6c24018a82ef2ef35fc4bda",
    "arrow_batch_rows": 65536, "top_values": 10, "support_limit": 20, "context_radius": 2,
}
assert isinstance(SAMPLE_COUNT, int) and 1 <= SAMPLE_COUNT <= 100
assert isinstance(COLUMN_GROUP_SIZE, int) and COLUMN_GROUP_SIZE >= 1
print("Input:", DATA_PATH)
print("Scope:", FEATURE_SCOPE, "| Samples:", SAMPLE_COUNT, "| Split: TRAIN only")
