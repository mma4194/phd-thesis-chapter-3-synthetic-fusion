# Progress, failure and recovery

## Locate the current state

Use the run directory printed at startup. Read `status/<stage>.json` and `logs/<stage>.log`. A file existing on disk does not by itself mean its stage completed.

| Situation | Action |
|---|---|
| Preflight is blocked | Fix the reported path, dependency or hardware issue before training. |
| A stage is still running | Monitor its log; do not launch a second controller. |
| Later reporting/evaluation stage failed | Resume the same run after diagnosing the failure. |
| Residential generation stage failed internally | Preserve outputs and logs. Automatic cell-suffix recovery is not supported. |
| All stages completed but values differ | Inspect comparison details. Do not overwrite reference values or rerun blindly. |
| Home quota exhausted | Keep source/environment in Home; place future outputs in configured scratch. Preserve current required outputs. |

## Resume a run created by this edition

```bash
python run.py --resume /absolute/path/to/existing_run
```

Omit `--config`: resume reads the recorded run configuration. It verifies source package, notebook source and core environment identity. Completed stages are skipped only when their recorded primary-output hashes match. Failed attempt status/logs are preserved under `attempt_history/`.

Inventory, figure or later comparison failure does not require generator training again. Some post-generation stages may rewrite their own derived reports when retried; expensive completed residential outputs remain.

For notebook-based resume, set `SF_RESUME_RUN` to the existing run before launching its kernel. Changing notebook source changes its identity. Terminal resume is usually simpler.

## Limits

This is stage-level recovery. Residential cells share trained objects in memory, so a missing middle cell cannot safely be replayed from arbitrary files. The controller intentionally stops rather than silently constructing a different run. This edition cannot automatically resume earlier differently packaged author runs.

The diagnostic archive is printed on failure and completion. It is for inspection; it does not contain all model weights, source datasets or generated records. Preserve the full run while investigating.
