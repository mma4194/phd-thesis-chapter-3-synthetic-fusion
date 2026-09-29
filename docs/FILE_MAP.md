# Repository map

| Location | Contents | Needed by |
|---|---|---|
| `notebooks/` | Guided Chapter 3 notebook | Notebook users |
| `run.py`, `preflight.py` | Full execution and prerequisite checks | Notebook/CLI workflow |
| `src/` | Controller, evaluation adapters, figures and result checks | Runtime |
| `source_cells/` | Ordered residential and downstream research implementation | Runtime |
| `public/src/`, `public/run.py` | Controlled, Smart* and TON-IoT processing | Runtime |
| `design_inputs/` | Q4 manifests, alignment support, input lock and CTGAN policy | Fixed research inputs |
| `reference/` | Annotations, candidate schema, immutable results and readable maps | Fixed inputs / verification |
| `public/reference/` | Public-study comparison targets | Verification after recomputation |
| `datasets/residential/` | Dataset descriptor and verification instructions; optional local data file | Residential input acquisition |
| `environment/`, `requirements*.txt` | Recorded software/hardware and dependency specifications | Setup and preflight |
| `config/example.json` | Four-location configuration template | Setup |
| `docs/` | Reviewer-facing methods, data, execution and interpretation | Readers |
| `supplementary/` | Thesis supplementary PDF only | Scientific methods/results |
| `validation/author_reference/` | Recorded 228-check author comparison | Provenance |
| `tests/`, `public/tests/` | Fast checks of implementation behaviour | Software validation |
| `tools/residential_data.py` | Download or verify the exact residential input | Data preparation |
| `MANIFEST.json`, `verify_package.py` | Distributed-source integrity | Preflight / source verification |

There is one study notebook and one shared implementation. Extracted source numbering preserves research-cell traceability; it is not a list of manual patches. `docs/source_cell_index.csv` maps the individual cells.

Do not run files in `source_cells/` independently. They are executed by the worker in the documented order. Supporting CSV/JSON/Parquet files are retained when they are consumed as research inputs or numerical comparison targets. Generated arrays, fitted models and previous synthetic outputs are not inputs to a fresh run.
