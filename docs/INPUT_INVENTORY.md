# Required inputs and their provenance

A fresh run reads the three source datasets, the supplied research definitions and the numerical verification targets. These have different roles.

| Input | Included? | Used for |
|---|---|---|
| Canonical residential prepared Parquet | [Separate published data asset](https://github.com/mma4194/phd-thesis-chapter-3-synthetic-fusion/releases/tag/residential-input-v1) | Residential TRAIN/VAL/TEST and generation |
| Six Smart* Home A folders | Download from provider | Public study reconstruction |
| 23 TON-IoT network CSVs and seven IoT CSVs | Download from provider | Alignment and physical/network checks |
| Eight `design_inputs/coupling_manifest*` files | Yes | Fixed TRAIN/VAL Q4 pair definitions |
| Six `design_inputs/supporting_records/*` files | Yes | Manifest consistency and eligibility checks |
| `design_inputs/INPUT_LOCK.json` | Yes | Hashes for the 14 files above |
| `design_inputs/ctgan_encoding_policy.json` | Yes | Numeric-count and discrete-field rules |
| `reference/author_feature_roles.csv` | Yes | 115 supplied human role annotations |
| `reference/recovered/q6_conservative_public_column_registry.csv` | Yes | Expected final 23-column schema check |
| `reference/current/*` | Yes | Reported-result, regression and window comparisons |
| `public/reference/*` | Yes | Controlled and public-study comparisons after computation |

The complete file-level inventory, sizes and SHA-256 values are in [reference/input_catalog.csv](../reference/input_catalog.csv). The package-wide manifest additionally covers source code, notebook and documentation.

## Q4 manifest contents

`coupling_manifest_all.csv`, `coupling_manifest_tierA.csv` and `coupling_manifest_tierB.csv` have Parquet companions. `coupling_manifest_meta.json` and `coupling_manifest_preview.json` provide metadata. The files are copied explicitly into a new run; no search through unrelated old run directories selects them.

Supporting records include TRAIN and VAL alignment results, the TRAIN/VAL replication table and three recorded builder-input descriptions. All eight consumed manifest files match the SHA-256 values recorded by the completed author run. [reference/provenance/q4_manifest_identity.json](../reference/provenance/q4_manifest_identity.json) records that comparison without account-specific paths.

The validator performs identity, schema, cross-format consistency and eligibility checks. The upstream pair-discovery program was not recovered; these archived data-derived definitions are supplied inputs. Do not describe their validation as rediscovering the pairs from raw data.

## Annotations and roles

The author-role CSV has 115 unique feature names: 94 continuous, 19 binary, one observability and one event-driver annotation. These human labels support inspection and are distinct from the automated generator-ownership rules. Ownership code and its explicit exceptions are included in the residential source cells.

## What a fresh run rebuilds

Real-data splits, continuous candidates, baseline fits, synthetic tables, branch profiles, selection reports, regression predictions, scientific output hashes and numerical figures are produced in the current run. Previously saved models, array caches and generated candidate tables are not required external inputs.

The public TON-IoT reference matrix is not fed to fitting or alignment. The new alignment is built from all 30 provider CSVs before comparison. Expected results never replace computed values.
