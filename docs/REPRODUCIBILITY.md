# Reproduce Chapter 3

This walkthrough uses one configuration file and one execution entry point. It begins with the prepared study inputs and ends with a comparison against the registered results.

## 1. Check the source package

Extract or clone into a new directory. Do not overlay source files under an active run. From the repository root:

```bash
python verify_package.py
```

Expected: `status: PASS` and an empty failures list. This verifies distributed files, including notebook source. Notebook execution outputs are not part of its source identity. Do not edit source or expected values to remove an integrity failure.

## 2. Obtain and check all inputs

Follow [DATA.md](DATA.md). The residential file must match the registered hash, not merely have the same filename. Smart* needs six Home A folders and TON-IoT needs all 30 named CSV files. The Q4 manifest, support records and author annotations are already included: [input inventory](INPUT_INVENTORY.md).

To check the residential file without model training:

```bash
python tools/residential_data.py --file /absolute/path/to/cps_unified_1s_FULLGRID_FINAL_WITH_IOT_STATEFUL_STRICTMASKED_TRIMMED.parquet
```

The raw packet captures, old trained models, continuous candidate caches and old run directory are not required for a fresh run.

## 3. Configure four locations

Copy `config/example.json` to `config/local.json` and set:

| Key | Location |
|---|---|
| `residential_parquet` | Exact prepared `.parquet` file |
| `smartstar_root` | Folder directly containing the six `homeA-*` directories |
| `toniot_root` | Folder directly containing `Processed_datasets` |
| `output_parent` | Writable output folder outside the repository, normally scratch |

Keep `n_jobs: 1` and `public_timeout_seconds: 0`. Use absolute paths. Input directories are read; they are not copied or modified by configuration. Both notebook and CLI read `config/local.json` automatically when present; an explicit `--config` path selects a different file. This file is ignored by Git and is not included in package identity checks.

The author's defaults still work on Falcon. An optional verified residential copy at `datasets/residential/` is also detected. Outside Falcon, the default output is beside the repository; explicit configuration is recommended.

## 4. Run preflight in the compute allocation

Use the environment described in [ENVIRONMENT.md](ENVIRONMENT.md):

```bash
python preflight.py --config config/local.json
```

Preflight creates a small run record, checks dependencies, hardware, data layout, residential identity, package identity and Q4 manifest consistency. It performs no model training. Read its `preflight.csv`: every row must be `PASS`. Its final report will not claim all stages completed because no training stages have run.

A later full run creates a new directory. The preflight-only directory is not a training checkpoint.

## 5. Execute the study

Choose ONE route.

**Notebook:** open `notebooks/Synthetic_Fusion_Chapter3.ipynb` in the matching kernel, restart it, then Run All. Its sections explain each stage, show result tables and render the numerical figures.

**Terminal:**

```bash
python run.py --config config/local.json
```

Keep the printed run directory. Logs are under `logs/` and stage outcomes under `status/`. Each stage runs in a fresh process; residential research cells share one stage namespace where fitted objects are needed.

Order: controlled experiment → Smart* → TON-IoT → residential generation/baselines → post-generation summaries → candidate selection → feature inspection → protocol regression → window analysis → registered-result comparison → figures → output inventory.

Do not start a second controller or edit the notebook while the run is active. Use [RECOVERY.md](RECOVERY.md) if interrupted.

## 6. Check the results

Read [RESULTS.md](RESULTS.md). Require `all_stages_completed: true` and `paper_reference_status: PASS_FOR_VERSIONED_REFERENCE` in `FINAL_REPORT.json`. Inspect individual rows in `paper_comparison/comparison.csv`, not only console messages. The baseline check also requires three successful CTGAN scopes and no classifier fallback events.

Expected values and the mapping to generated files are listed in `reference/result_map.csv`. They are fixed comparison targets. A difference is retained, not fitted away or replaced with a reference number.

## 7. Preserve a reproducibility record

Retain the exact source version, input hashes, configuration, environment/hardware records, status files, comparison tables and diagnostic ZIP. Keep the full generated run until its output inventory and comparisons are reviewed. Do not substitute the diagnostic ZIP for the source datasets or complete run.

## 8. Test independent repeatability when required

For this stronger claim, perform a second independent fresh run with unchanged source, inputs, configuration values and environment, then:

```bash
python compare_runs.py /absolute/path/to/first_run /absolute/path/to/second_run --output /absolute/path/to/new_comparison
```

Read the declared fingerprint scope and exclusions. Matching the reported results in one run and matching two independent runs are different tests. Neither automatically covers every literal number in the thesis.
