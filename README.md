# Synthetic Fusion
### Chapter 3 · PhD examination and viva

Synthetic Fusion generates and assesses heterogeneous smart-home data. This repository supports examination and viva discussion of Chapter 3 of Mohammed Alosaimi's PhD thesis. It provides the executable study, fixed research inputs, numerical reference results and supplementary methods needed to inspect and reproduce the experiments.

**Start here:** [Reproduce the chapter](docs/REPRODUCIBILITY.md) · [Required data](docs/DATA.md) · [Understand the results](docs/RESULTS.md) · [Supplementary PDF](supplementary/Synthetic_Fusion_Supplement.pdf)

## Study workflow

The workflow starts from the prepared residential Parquet, Smart* Home A files and TON-IoT processed streams. It trains the residential generators and baselines, evaluates quality, selects a candidate, runs prediction diagnostics and reconstructs the registered numerical figures. It then compares the new outputs with 228 fixed reported-result checks.

The notebook is the main guided entry point. The terminal runner executes the same implementation. Models and synthetic candidates are generated within the new run; earlier fitted models are not required.

| Component | Included material and recorded checks |
|---|---|
| Source, required Q4 manifests, annotations, encoding policy and numerical references | Included |
| Residential prepared Parquet | [Download verified input](https://github.com/mma4194/phd-thesis-chapter-3-synthetic-fusion/releases/download/residential-input-v1/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet) (238,609,923 bytes); distributed separately from source code. |
| Smart* and TON-IoT raw input files | Obtain from their providers using the data guide |
| Falcon result validation | **228/228 registered comparisons passed**; detailed record included |
| Software checks | 37 automated tests passed; package-integrity checks included |

The validation record covers the completed Falcon run and the separately evaluated numeric-count CTGAN baselines. The notebook brings generation, baseline evaluation and result comparison together in one workflow.

## Run the study

Use Linux, Python 3.11.11 and the recorded environment. The strict reproduction preflight requires an NVIDIA L40S, PyTorch CUDA build 12.8 and cuDNN 91002. See [environment setup](docs/ENVIRONMENT.md) before installing or selecting a kernel. The historical full execution exceeded 24 hours; the current edition has no measured end-to-end runtime or peak-memory guarantee.

1. Obtain the three datasets and verify the residential file.
2. Copy `config/example.json` to `config/local.json` and enter your actual input/output paths.
3. From the repository root, run:

```bash
python verify_package.py
python preflight.py --config config/local.json
```

4. After preflight passes, open [Synthetic_Fusion_Chapter3.ipynb](notebooks/Synthetic_Fusion_Chapter3.ipynb), select the same Python environment, restart its kernel and run all cells.

For terminal execution instead:

```bash
python run.py --config config/local.json
```

Run only one entry point at a time. Use an allocated compute session, keep this repository and its environment in Home on Falcon, and put generated output on scratch. Every fresh run receives a new directory; keep the printed `RUN_DIRECTORY`.

## Check the outcome

Read `FINAL_REPORT.json` in the new run. Reproduction of the registered results requires:

```json
{
  "all_stages_completed": true,
  "paper_reference_status": "PASS_FOR_VERSIONED_REFERENCE"
}
```

Inspect `paper_comparison/comparison.csv` for each expected value and its computed result. The [results guide](docs/RESULTS.md) explains tolerances, missing evidence, baseline completeness and independent repeatability. Successful execution alone does not establish numerical agreement.

## Documentation

| Guide | Purpose |
|---|---|
| [Reproduction walkthrough](docs/REPRODUCIBILITY.md) | From a clean checkout to a checked result |
| [Environment](docs/ENVIRONMENT.md) | Python, dependencies, GPU requirements and software checks |
| [Datasets](docs/DATA.md) | Exact files, folder layouts, residential hash and acquisition |
| [Included input inventory](docs/INPUT_INVENTORY.md) | Why each manifest, annotation and reference is supplied |
| [Implementation](docs/IMPLEMENTATION.md) | Stage order, generator/evaluation rules and source locations |
| [Results and expected values](docs/RESULTS.md) | Output locations and the complete 228-check result map |
| [Recovery](docs/RECOVERY.md) | Resume completed stages without silently repeating training |
| [Repository map](docs/FILE_MAP.md) | Where to find code, fixed inputs, references and supplementary material |

## Scientific scope

The starting point is the canonical prepared residential table, not raw packet capture. Q4 TRAIN/VAL manifests and feature-role annotations are supplied research inputs. Their integrity and consistency are checked; the original upstream Q4 manifest-builder program is not included. Six branch-window rows use the documented reconstruction in the implementation. These conditions define the reproduction supplied here.

The expected values are comparison targets, not training inputs. The small TON-IoT aligned reference is used only after rebuilding alignment from the provider files. Smart* and TON-IoT raw datasets are not redistributed.

## Third-party inputs

Smart* and TON-IoT source data are obtained from their providers under the applicable terms. Their attribution is retained in [third-party notices](public/THIRD_PARTY_NOTICES.md). Python dependencies retain their own licences.
