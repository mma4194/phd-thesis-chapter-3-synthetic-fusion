# Read and verify the results

## Three separate questions

1. **Did execution finish?** All stage status files and `FINAL_REPORT.json` report completion.
2. **Did the registered results match?** The 228-check comparison and baseline-completeness conditions pass.
3. **Did independent runs agree?** `compare_runs.py` checks the declared scientific-output inventories from two fresh runs.

The recorded Falcon results, including the separately evaluated numeric-count CTGAN configurations, passed all 228 registered comparisons on 29 September 2026. The complete comparison is supplied in `validation/author_reference/`. Reviewers can use the same fixed reference to assess their own execution.

## Files to inspect first

| Generated file, relative to the new run | Purpose |
|---|---|
| `FINAL_REPORT.json` | Execution, registered agreement and repeatability status |
| `paper_comparison/SUMMARY.json` | Check counts, differences, CTGAN completion and C2ST fallback audit |
| `paper_comparison/comparison.csv` | Expected/computed value for every registered identifier |
| `paper_comparison/issues.csv` | Non-passing comparisons |
| `candidate_schema_check.json` | Final candidate's 23-column schema |
| `fresh_continuous_training.json` | Current-run continuous candidates, with zero prior-candidate hits |
| `protocol_regression/q5_corrected_scores.csv` | Paired prediction results and simple controls |
| `window_reconstruction/window_metrics.csv` | Six branch/window rows |
| `residential_BINARY_V6/reports/cell17_4_baseline_metric_by_artifact.csv` | Baseline and pipeline metrics by scope |
| `figures/` | Numerical Figures 2, 3 and S1 plus plotting data |
| `repeatability_inventory.json` | Scientific fingerprints and explicit metadata exclusions |

The PDF supplement documents the definitions and reference findings. It is not automatically rewritten to make a different run appear to agree. Figure 1 is an authored conceptual diagram, not a generated measurement.

## Complete result map

[`reference/result_map.csv`](../reference/result_map.csv) expands all 228 registered checks into a readable table with identifier, expected value, reported precision, generated evidence path, filter and aggregation. [`reference/current/paper_reference_v1.json`](../reference/current/paper_reference_v1.json) is the immutable machine-readable authority. Its SHA-256 is:

`40b25bb5bd94c7a357349693a9bf54d6ed9bca0d59218d9aca835e2c843762f4`

The source-document hashes in that reference identify the documents at reference registration, not every later typography or repository edit.

## Selected expected values

| Quantity | Expected reported value |
|---|---:|
| Residential source rows / physical columns | 1,277,694 / 740 |
| TRAIN / VAL / TEST rows | 766,616 / 255,538 / 255,540 |
| Full synthetic artifact logical columns | 555 |
| Intermediate / final candidate columns | 46 / 23 |
| Protocol-core penalty: pipeline / CTGAN / TabDDPM | 0.074814 / 0.186232 / 0.150078 |
| Intermediate-candidate core penalty: pipeline / CTGAN / TabDDPM | 0.062348 / 0.216353 / 0.201483 |
| Regression median / p95 normalized MAE gap | 0.092861 / 0.102925 |

The intermediate-candidate baseline scope has 45 eligible fields; it is not the final 23-column candidate. Scope names and filters are retained in the result map. Lower aggregate penalty does not mean superiority on every metric or model family. The regression diagnostic is reference-assisted and exploratory; persistence has lower MAE than the learned controls.

## Acceptance rules

A registered comparison must agree at displayed precision AND satisfy the unchanged numerical comparison tolerance (`rtol=1e-7`, `atol=1e-9`). Missing, nonfinite and ambiguous evidence cannot pass. These tolerances are not the quality-grade thresholds.

The full comparison also checks that all three CTGAN scopes succeeded, no C2ST fallback occurred and input evidence did not change during comparison. `DIFFERENT`, `BLOCKED` and `ERROR` are retained distinctly in the comparison rows.

The integrated CTGAN uses numeric packet/byte counts and genuinely discrete indicators. The reference's two `CTGAN_numeric_counts_v1` labels resolve to this integrated CTGAN in the same scopes. File/label routing changes no expected value or tolerance.

## Exact repeatability scope

Output comparison uses the declared ordered-value fingerprints. Paths, execution times, compression and serialized model bytes are not scientific equality targets. Read all exclusions. One passing run is not an independent-run repeatability demonstration, and 228 mapped checks are not an exhaustive audit of all thesis prose.
