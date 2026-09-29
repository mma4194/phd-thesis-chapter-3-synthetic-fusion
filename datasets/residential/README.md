# Residential prepared input

The source archive contains the dataset descriptor, not the residential Parquet itself. Download or obtain the exact asset identified in `dataset.json`, verify it with `tools/residential_data.py`, and point `residential_parquet` in `config/local.json` to it.

The optional local location is this directory with the canonical filename. Parquet data remains excluded from ordinary Git tracking. A GitHub release data asset is downloaded separately from its source-code ZIP.

Expected dimensions: 1,277,694 rows and 740 physical columns. Required SHA-256:

`a4d7446daba0e46dab2dc21607a99c5a734a5366a6c24018a82ef2ef35fc4bda`

The reproduction begins at this prepared table. Raw packet capture and upstream feature extraction are outside this executable scope.

Published asset: [download the Parquet](https://github.com/mma4194/phd-thesis-chapter-3-synthetic-fusion/releases/download/residential-input-v1/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet).

GitHub asset ID: 598007521. The reported size and SHA-256 match the Falcon input audit. Run the local verification helper after download.
