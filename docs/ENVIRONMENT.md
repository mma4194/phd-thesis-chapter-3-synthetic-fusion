# Computational environment

The full study targets **Linux and Python 3.11.11**. The process controller uses Linux file locking. Windows can be used to inspect the notebook and repository; native Windows execution is not the validated full-run route.

## Recorded configuration

| Item | Recorded value |
|---|---|
| GPU class | NVIDIA L40S; two devices were visible in the author run |
| PyTorch | 2.10.0, CUDA build 12.8 |
| cuDNN | 91002 |
| Driver | 580.65.06 in the recorded environment |
| CPU | Intel Xeon Gold 6430 |
| Core scientific packages | Exact versions in `requirements.txt` |
| Complete observed environment | `environment/reference_environment_full.json` |

Preflight checks the visible GPU class, CUDA build and cuDNN version, as well as Python and pinned packages. It does not enforce a two-device count or the recorded driver version. A different device can change numerical results; bypassing a preflight failure is not a verified reproduction route.

Use the established environment if it already passes. On the author's Falcon account its executable is:

```text
/shared/home1/c.c21126547/.trinity/venvs/ood-jupyter/bin/python3
```

That private account path is an example, not a dependency. Other reviewers should use the Python executable in their own matching environment.

## Build a separate environment

With Python 3.11.11 available and a compatible GPU allocation:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install ipykernel
.venv/bin/python -m ipykernel install --user --name synthetic-fusion-ch3 --display-name "Synthetic Fusion Chapter 3"
```

This is an installation recipe, not a claim that a newly installed environment has already passed the experiment. Check it with preflight. `reference_environment_full.json` is an observation of the author's environment, not a portable lockfile for every operating system.

Use `.venv/bin/python` in the commands below if the virtual environment is not activated. In Jupyter, select **Synthetic Fusion Chapter 3** and check the Python path printed in the notebook.

## Fast checks without training

```bash
python verify_package.py
python -m unittest discover -s tests -v
python -m unittest discover -s public/tests -v
python check_manifest.py
```

The unit checks can run without the datasets or GPU, using `requirements-checks.txt` in a separate test environment. They check software behaviour, not generation results. The full preflight requires the full dependencies and data. Do not replace the full environment with the lightweight test requirements.

## Resources and determinism

The author full run exceeded 24 hours. The semantic CTGAN count encoding completed its two amended fits in approximately two and four minutes, but that is not a timing estimate for the full study. Continuous candidates and other stages remain expensive. A peak RAM or disk requirement has not been measured for this consolidated edition; no fixed minimum is asserted.

On Falcon keep the environment and notebook in Home, with output in scratch. Use an allocation long enough for the complete run and retain its outputs until comparisons are finished. Model serialization and compression bytes are excluded from scientific exact-equality targets.

The controller fixes hash and numerical-thread settings. CPU CTGAN uses its recorded two threads; regression uses one worker. Random seeds do not guarantee equality across environments. Official reference: https://docs.pytorch.org/docs/2.10/notes/randomness.html
