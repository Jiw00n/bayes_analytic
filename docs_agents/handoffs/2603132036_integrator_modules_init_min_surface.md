## Task

Reduce the `gallery/constrained_gen/modules` package export surface to the active workflow minimum.

## Files Checked

- `gallery/constrained_gen/modules/__init__.py`
- `gallery/constrained_gen/generate.py`
- `gallery/constrained_gen/order_test.ipynb`

## Changes

- Replaced the broad re-export list in `gallery/constrained_gen/modules/__init__.py` with a minimal package surface:
  - `build_symbolic_state`
  - `ScheduleGenerator`
- Added `__all__` so the remaining package-level API is explicit.

## Verification

- Ran syntax check:
  - `python -m py_compile gallery/constrained_gen/modules/__init__.py gallery/constrained_gen/modules/record_loader.py gallery/constrained_gen/modules/projected_gpu_validation.py`

## Outcome

- The package no longer presents most internal modules as a pseudo-public API.
- Direct-module imports remain the intended path for internal code.

## Remaining Uncertainty

- Documentation files still describe the older, broader package surface.

## Recommended Next Owner

- `integrator` for follow-up cleanup of thin wrappers in `schedule_generator.py` and legacy record/sketch helpers.
