# AGENTS.md

## Cursor Cloud specific instructions

### Overview

`mpoints` is a pure-Python scientific library implementing state-dependent Hawkes processes (simulation, estimation, plotting). There are no external services — only Python 3.12+, a C compiler, and pip are needed.

### Environment

- **Python 3.12** (or newer) is the target runtime. A virtualenv lives at `/workspace/.venv`.
- Activate with: `source /workspace/.venv/bin/activate`

### Build

The Cython extension must be compiled before running tests or the package:

```
cd mpoints/ && python setup.py build_ext --inplace && cd ..
```

To build a wheel: `python -m build` (both isolated and non-isolated builds work).

Then install: `pip install dist/mpoints-*-cp*.whl`

### Lint

```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,build,dist
flake8 . --count --ignore=W503,W605 --max-complexity=32 --max-line-length=127 --statistics --exclude=.venv,build,dist
```

### Test

```
pytest tests/*_test.py
```

### Gotchas

- The `.venv/` directory must be excluded from flake8 runs (add `--exclude=.venv,build,dist`).
- One test (`test_simulate_and_estimate`) is intentionally `@unittest.skip`-ped in the repo.
- If you modify the `.pyx` file, regenerate the `.c` by running `build_ext --inplace` from the `mpoints/` directory before building a wheel from root.
