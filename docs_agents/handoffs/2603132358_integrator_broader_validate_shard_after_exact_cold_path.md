# 2603132358 Broader Validate Shard After Exact Cold Path Changes

- Agent: `integrator`
- Date: `2026-03-13 23:58`
- Status: `completed`
- Note: `single-session validation only`

## What Was Run

- `python -m py_compile gallery/constrained_gen/modules/exact_gpu_constraints.py gallery/constrained_gen/modules/constraint_set.py gallery/constrained_gen/validate.py`
- Broader representative validate shard:
  - `python gallery/constrained_gen/validate.py --task-index 0`
  - `python gallery/constrained_gen/validate.py --task-index 8`
  - `python gallery/constrained_gen/validate.py --task-index 16`
  - `python gallery/constrained_gen/validate.py --task-index 29`
  - `python gallery/constrained_gen/validate.py --task-index 45`
  - `python gallery/constrained_gen/validate.py --task-index 68`
  - `python gallery/constrained_gen/validate.py --task-index 84`
  - `python gallery/constrained_gen/validate.py --task-index 101`
  - `python gallery/constrained_gen/validate.py --task-index 126`
  - `python gallery/constrained_gen/validate.py --task-index 155`
  - `python gallery/constrained_gen/validate.py --task-index 210`
  - `python gallery/constrained_gen/validate.py --task-index 260`
  - `python gallery/constrained_gen/validate.py --task-index 320`
  - `python gallery/constrained_gen/validate.py --task-index 383`

## Files Checked

- `src/auto_scheduler/exact_gpu_constraints.cc`
- `gallery/constrained_gen/modules/exact_gpu_constraints.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/validate.py`

## Artifact Path

- `/tmp/projected_gpu_full_validation/validator/exact_broader_shard/summary.json`

## Outcome

- `pass_count = 14`
- `fail_count = 0`
- `stopped_on_failure = false`

### Task Timings

- task `0`: `2.388s`
- task `8`: `2.479s`
- task `16`: `2.415s`
- task `29`: `2.435s`
- task `45`: `4.430s`
- task `68`: `2.413s`
- task `84`: `4.987s`
- task `101`: `2.384s`
- task `126`: `2.414s`
- task `155`: `4.290s`
- task `210`: `4.931s`
- task `260`: `4.303s`
- task `320`: `4.306s`
- task `383`: `3.321s`

## Remaining Uncertainty

- This was a broader representative shard, not a full sweep.
- A role-pure validator note for the same shard was not produced in this session because execution was completed in the main thread after agent orchestration became unreliable.

## Next Recommended Owner

- `reviewer` if a stricter evidence sign-off is still desired
