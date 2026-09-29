# Public dataset implementation

These modules are called by the root master notebook and `run.py`. Use those root entry points
for the full study. Smart* and TON-IoT are rebuilt from source files, not bundled prepared tables.
The numerical reference tables here support the unchanged public-stage comparisons.
Read `../docs/DATA.md` for acquisition and `THIRD_PARTY_NOTICES.md` for attribution.

The recorded public TON-IoT aligned matrix in `public/reference/toniot/` is a comparison target only. The pipeline first rebuilds alignment from all 30 source CSVs and then compares it. It never uses that reference matrix for generation, fitting, event construction or pair selection.
