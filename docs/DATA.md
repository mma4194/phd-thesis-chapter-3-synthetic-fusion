# Datasets and prepared research inputs

## Smart* Home A, 2013 release

Official source: https://traces.cs.umass.edu/docs/traces/smartstar/

Use **UMass Smart* Dataset – 2013 release / Home A**, not the 2017 Home A electrical dataset. Download the following six archives from the official page and extract them so the `--input` directory directly contains their six folders:

| Archive/folder | Files in the recorded input inventory | Total extracted bytes in that inventory |
|---|---:|---:|
| homeA-circuit.tar.gz / homeA-circuit | 92 | 675758807 |
| homeA-environmental.tar.gz / homeA-environmental | 93 | 3839767 |
| homeA-switch.tar.gz / homeA-switch | 92 | 1142981 |
| homeA-furnace.tar.gz / homeA-furnace | 92 | 3862 |
| homeA-door.tar.gz / homeA-door | 93 | 609797 |
| homeA-motion.tar.gz / homeA-motion | 92 | 6543313 |

Counts include FORMAT/summary files present in the original directory. They are diagnostics, not substitutes for file hashes. The parser excludes metadata files. The supplied archives did not include the full original per-file raw hash inventory, so byte-identical raw-source provenance cannot be asserted in advance. Each new run records the actual relative filenames, sizes and SHA-256 hashes. The original parsed table spans 2012-04-30 09:16 UTC to 2012-07-31 09:15 UTC, with 132480 one-minute rows and 293 feature columns before eligibility. The TEST audit covers 291 eligible features.

Do not include `homeA-meter` or `homeA-phase` in the evaluated scope. If downloading `homeA-all.tar.gz`, retain the six required directories and leave the optional ones unused. Do not flatten the daily files into one directory or rename them: relative file names can affect feature names and deterministic keys.

The canonical settings are 60-second windows; chronological 60/20/20 split; seed 20260717; binary forward fill limited to 180 windows; minimum TRAIN+VAL support 30 and VAL support 5; no file or feature truncation. Structural zero handling for event/mask windows is part of the source implementation. Original split sizes: TRAIN 79488, VAL 26496, TEST 26496.

## TON-IoT processed streams

Official source: https://research.unsw.edu.au/projects/toniot-datasets

Follow its dataset download link. Use the public **Processed_datasets** hierarchy. Your `--input` directory must contain:

- `Processed_datasets/Processed_Network_dataset/Network_dataset_1.csv` through `Network_dataset_23.csv` (all 23 files).
- `Processed_datasets/Processed_IoT_dataset/IoT_Fridge.csv`
- `Processed_datasets/Processed_IoT_dataset/IoT_GPS_Tracker.csv`
- `Processed_datasets/Processed_IoT_dataset/IoT_Garage_Door.csv`
- `Processed_datasets/Processed_IoT_dataset/IoT_Modbus.csv`
- `Processed_datasets/Processed_IoT_dataset/IoT_Motion_Light.csv`
- `Processed_datasets/Processed_IoT_dataset/IoT_Thermostat.csv`
- `Processed_datasets/Processed_IoT_dataset/IoT_Weather.csv`

The raw-source route streams these files in chunks (500000 rows by default in the preserved source). They can require several GB of disk space including extracted inputs and outputs. All source CSVs are required by the full workflow. Credentials or temporary signed download links are not embedded in this artifact.

Do not substitute `Train_Test_datasets`, raw PCAPs or another paper's TON-IoT benchmark. The documented run has 7782 network windows, 9906 IoT windows and 7024 overlapping one-minute windows. Aligned split sizes: TRAIN 4214, VAL 1404, TEST 1406. Rows are not a fully contiguous clock grid. The physical Q4 analysis constructs 40 drivers from 18 source columns; 30 drivers are eligible, and 50 pairs are selected from 1260 TRAIN+VAL candidate scores.

## Residential source

The full parent workflow also requires the residential Parquet described in the root README and notebook. Keep all source datasets outside the package. The public-data routines do not substitute for residential training or evaluation.

The residential input must have 1,277,694 rows and 740 physical columns, with SHA-256:

`a4d7446daba0e46dab2dc21607a99c5a734a5366a6c24018a82ef2ef35fc4bda`

Its filename in the author workflow is `cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet`.
The supplied feature-role annotations and coupling files are inputs separate from that Parquet.
The residential asset is publicly available in the [residential-input-v1 release](https://github.com/mma4194/phd-thesis-chapter-3-synthetic-fusion/releases/tag/residential-input-v1). [Download the exact Parquet](https://github.com/mma4194/phd-thesis-chapter-3-synthetic-fusion/releases/download/residential-input-v1/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet) (238,609,923 bytes). It is a separate asset from the source-code ZIP. GitHub reports the registered SHA-256 for the uploaded asset; the local helper independently verifies downloaded bytes. Its machine-readable descriptor is `datasets/residential/dataset.json`.

If supplied as a GitHub release asset, download the asset separately from the source-code ZIP. Keep the exact Parquet bytes. A raw capture download or a newly exported, similarly named Parquet is not a substitute for the registered input.

Verify an existing file:

```bash
python tools/residential_data.py --file /absolute/path/to/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet
```

From the repository root, download and verify it with:

```bash
python tools/residential_data.py --url https://github.com/mma4194/phd-thesis-chapter-3-synthetic-fusion/releases/download/residential-input-v1/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet --output datasets/residential/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet
```

The helper streams to a partial file, checks the SHA-256 and Parquet dimensions, then installs the verified output. It will not overwrite an existing file.

The 16 files under `design_inputs/` and the author-role annotations are already included. See [INPUT_INVENTORY.md](INPUT_INVENTORY.md) for their roles. Old synthetic datasets, checkpoint files and baseline outputs are regenerated, not downloaded as source inputs.

## Configuration example

Copy `config/example.json` to `config/local.json` and set the four paths. For Smart*, the root is
`.../Smart_Star_Dataset/homeA`, immediately above the six `homeA-*` folders. For TON-IoT, it is
`.../TON_IoT`, immediately above `Processed_datasets`. Scratch is used only for output; inputs
and the notebook can remain in Home. No dataset files are downloaded or copied by the runner.

The recorded public TON-IoT aligned matrix in `public/reference/toniot/` is a comparison target only. The pipeline first rebuilds alignment from all 30 source CSVs and then compares it. It never uses that reference matrix for generation, fitting, event construction or pair selection.
