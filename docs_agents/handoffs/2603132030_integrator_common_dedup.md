## Task

Remove duplicated helper implementations between `common.py` and `record_loader.py`.

## Files Checked

- `gallery/constrained_gen/modules/common.py`
- `gallery/constrained_gen/modules/record_loader.py`

## Changes

- Removed the duplicate `clean_name` implementation from `record_loader.py`.
- Removed the duplicate `load_and_register_network` implementation from `record_loader.py`.
- Imported `clean_name` and `load_and_register_network` from `common.py` instead, so `record_loader.py` keeps one source of truth for those helpers.

## Verification

- Ran syntax check:
  - `python -m py_compile gallery/constrained_gen/modules/record_loader.py gallery/constrained_gen/modules/projected_gpu_validation.py gallery/constrained_gen/modules/common.py`

## Outcome

- `common.py` is now the only implementation owner for the duplicated dataset/helper functions.
- `record_loader.py` still exposes the same names through module imports, but no longer carries a second copy of the logic.

## Remaining Uncertainty

- Documentation still describes some legacy helper surfaces and has not been refreshed in this step.

## Recommended Next Owner

- `integrator` for the next cleanup pass on `modules/__init__.py` exports and remaining unused record/sketch grouping helpers.
