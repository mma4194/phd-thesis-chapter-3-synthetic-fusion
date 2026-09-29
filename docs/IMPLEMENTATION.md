# Implementation guide

[Study workflow diagram](figures/study_workflow.pdf) · [Thesis supplementary methods](../supplementary/Synthetic_Fusion_Supplement.pdf)

## Execution model

The master notebook calls `src/workflow.py`. Each stage runs `src/worker.py` in a separate Python process, with fixed hash seed and numerical thread limits. Residential source cells share one stage namespace because downstream steps depend on fitted objects and contracts. The executor assigns each cell its own `__file__`, preserving source guards and pickle/dataclass context.

The standalone selection diagnostic uses an isolated namespace, preventing its `datetime` import from replacing the timestamp class used by other cells. `write_json` uses a temporary file and replacement to avoid incomplete status JSON after interruption. A file lock prevents concurrent controllers; the worker inherits the lock descriptor.

## Stage map

| Stage | Main implementation | Inputs and principal outputs |
|---|---|---|
| Controlled | `public/src/controlled.py`, `public/src/verify.py` | Binary-process trials → decision counts and intervals. |
| Smart* | `public/src/smartstar.py` | Six Home A folders → aligned windows, generators, feature grades and selection manifest. |
| TON-IoT | `public/run.py`, `public/src/toniot*.py` | Processed streams → alignment, physical events, Q4 comparisons, clock sensitivity and selection. |
| Residential | `source_cells/residential/` through `worker.residential` | Canonical Parquet → fresh candidates, protocol/IoT outputs, Q1–Q4 and external baseline metrics. |
| Postrun | `source_cells/postrun/003.py`–`008.py` | Generated outputs → detailed consistency and window summaries. |
| Selection | `source_cells/selection/002.py`–`013.py`, `016_fixed_policy.py` | Current generation → Q6 policies, sparse activity-window checks, final 23-column candidate. |
| Inspection | `source_cells/inspection/`, `reference/author_feature_roles.csv` | Source TRAIN values and human role labels → feature inspection records. |
| Regression | `src/q5_design.py`, `worker.regression` | Real TEST and selected synthetic table → paired forest scores and simple controls. |
| Windows | `src/window_reconstruction.py` | Full synthetic and real TEST tables → six reconstructed window rows. |
| Paper comparison | `src/current_reference.py`, `src/reference_reader.py` | Saved current-run evidence → 228 registered comparisons and baseline-completeness checks. |
| Figures | `src/figures.py` | Current outputs → Figures 2, 3, S1 and figure data. |
| Inventory | `src/repeatability.py` | Current outputs → ordered-value fingerprints and declared exclusions. |

`source_cell_index.csv` lists every distributed extracted source cell and whether the master worker executes it. Numbering retains the original experiment's identifiers for traceability. It is not a second notebook or a sequence of patches the reader must apply.

## Residential scientific sequence

1. Load the canonical 1 Hz table and preserve the fixed chronological split: 766,616 TRAIN, 255,538 VAL and 255,540 TEST rows. Raw input has 740 physical columns; the full generated scientific artifact has 555 logical columns.
2. Establish schema, observation, role and branch-ownership contracts. Eligibility and model selection use the declared TRAIN/VAL rules. Role exceptions remain explicit, including the cups-value field.
3. Fit protocol and IoT branches. Continuous source targets number 148, with 147 final continuous owners after assembly arbitration. Binary and observation-mask target lists are separate scientific scopes.
4. Build all continuous model candidates in the current run. Files with `cache` in inherited artifact identifiers are current-run candidate intermediates, not imported earlier results. A reuse audit requires zero previous-candidate hits. Missing selected candidates cause failure rather than real-data substitution.
5. Assess marginal, temporal and observation properties. Q4 pair evaluation consumes the checked supplied TRAIN/VAL manifest. Its selection definitions are not tuned against TEST results.
6. Assemble the full table, preserve quality/blocker ledgers and select the final candidate. Individually nonfatal drivers are not automatically safe at group level; the activity-window policy can remove the group.
7. Fit and evaluate the external baseline configurations in their recorded scopes.

The complete mathematical definitions, thresholds, observation policies and limitations are in the [supplement](../supplementary/Synthetic_Fusion_Supplement.pdf). This guide locates implementation rather than introducing alternate definitions.

## CTGAN and baseline comparisons

`design_inputs/ctgan_encoding_policy.json` explicitly classifies protocol packet/byte counts as numeric and named event indicators as binary. Unknown fields or unexpected TRAIN support fail. Integer-valued support alone does not make a field categorical. Compact-IoT encoding is unchanged.

For the two amended scopes, source cell 149 retains TRAIN-only imputation, deterministic 20,000-row sampling, per-scope seed `SEED + scope_i * 1009`, architecture, 20 epochs, batch size 500, CPU execution and two threads. The same fit and save-before-sample sequence used by the evaluated amendment is integrated. TRAIN-derived clipping, nonnegativity and integer rounding are unchanged. Categorical counts are no longer attempted first.

Source cell 152 retains the original per-field metrics and five-component penalty formulas. C2ST fallback events are additionally recorded. The current comparison requires no fallback and three successful CTGAN scopes. Seven complete pairwise comparisons span four scopes; the protocol-sequence pipeline comparator covers one of 32 fields and is excluded from complete-scope conclusions.

The benchmark families use fixed unequal budgets and different selection opportunities. Lower aggregate penalty is not a universal statement of model-family superiority or a guarantee that every component improves.

## Regression

The final-candidate task is a reference-assisted, exploratory next-minute regression diagnostic. Five paired learner seeds fit random forests with 300 trees, maximum depth 6, minimum leaf size 5 and one worker. The design, target scaling, fitting/evaluation positions, persisted predictions and baselines are recorded. `reference/current/q5_scores.csv` is used only after scoring. It does not enter model fitting.

## Record identities

Source package hashes, notebook code identity, environment, hardware, raw residential hash and public file inventories are recorded. Completion receipts cover declared primary outputs; the final inventory has the wider output scope. Inventory hashing excludes explicit execution metadata and model serialization. A metadata-only table records row count and exclusions without pretending to have a scientific value digest.
