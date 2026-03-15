## Summary

Confirmed a current false-reject reproducer in the active constrained generation path.

## What Was Run

Environment:

- `source /root/work/venv/bin/activate`
- `export TVM_HOME=/root/work/tvm-ansor`
- `export PYTHONPATH=$TVM_HOME/python`
- `export TVM_LIBRARY_PATH=$TVM_HOME/build-release`

Search command:

```bash
python - <<'PY'
import json
from tvm.auto_scheduler import SketchPolicy
from gallery.constrained_gen.modules.task_paths import load_and_register_tasks
from gallery.constrained_gen.modules.schedule_generator import ScheduleGenerator

tasks = load_and_register_tasks()
for task_index, task in enumerate(tasks[:150]):
    sketches = list(SketchPolicy(task, params={"sample_init_no_invalid": 1}, verbose=False).generate_concrete_sketches())
    for sketch_index, state in enumerate(sketches[:3]):
        gen = ScheduleGenerator.from_task_state(task, state)
        params = gen.randomize_params()
        pruning = gen.check_all_pruning(params)
        exact = gen.check_all_exact(params)
        final = gen.check_all_final(params)
        if (pruning or exact) and not final:
            print(json.dumps({
                "task_index": task_index,
                "task_desc": task.desc,
                "workload_key": task.workload_key,
                "sketch_index": sketch_index,
                "pruning": pruning,
                "exact": exact,
                "final": final,
                "params": params,
            }, default=str))
            raise SystemExit(0)
print(json.dumps({"found": 0, "checked_tasks": 150}))
PY
```

Targeted artifact command:

```bash
python - <<'PY'
import json
from tvm.auto_scheduler import SketchPolicy
from gallery.constrained_gen.modules.task_paths import load_and_register_tasks
from gallery.constrained_gen.modules.schedule_generator import ScheduleGenerator
from gallery.constrained_gen.modules.gpu_projection_diagnostics import collect_false_reject_diagnostics

task_index = 135
sketch_index = 0
params = {
    "sp_0_0": 1,
    "sp_2_0": 1,
    "sp_0_1": 1,
    "sp_1_0": 1,
    "sp_0_2": 1,
    "sp_2_1": 1,
    "sp_1_1": 1,
    "sp_0_3": 1,
    "sp_1_2": 1,
    "sp_1_3": 1,
    "sp_24_0": 4,
    "sp_19_0": 2,
    "ur_28": 16,
}

task = load_and_register_tasks()[task_index]
state = list(SketchPolicy(task, params={"sample_init_no_invalid": 1}, verbose=False).generate_concrete_sketches())[sketch_index]
gen = ScheduleGenerator.from_task_state(task, state)
report = {
    "task_index": task_index,
    "task_desc": task.desc,
    "workload_key": task.workload_key,
    "sketch_index": sketch_index,
    "params": params,
    "pruning_violations": gen.check_all_pruning(params),
    "exact_violations": gen.check_all_exact(params),
    "final_violations": gen.check_all_final(params),
    "concrete_final_result": gen._get_concrete_final_result(params),
    "false_reject_diagnostics": collect_false_reject_diagnostics(gen, params, gen.check_all_exact(params)),
}
with open("/tmp/projected_gpu_full_validation/validator/false_reject_task135/task135_sketch0_false_reject.json", "w") as f:
    json.dump(report, f, indent=2, sort_keys=True, default=str)
PY
```

## Files / Functions Checked

- `gallery/constrained_gen/generate_programs.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/gpu_projection_diagnostics.py`
- `ScheduleGenerator.check_all_exact()`
- `ScheduleGenerator.check_all_final()`
- `ScheduleGenerator.check_all_hybrid()`
- `ParamSampler._validate_sample()`
- `collect_false_reject_diagnostics(...)`

## Raw Artifacts

- `/tmp/projected_gpu_full_validation/validator/false_reject_task135/task135_sketch0_false_reject.json`

## Concrete Outcome

Found a current reproducer:

- `task_index = 135`
- `task_desc = vm_mod_fused_nn_dense_nn_relu`
- `sketch_index = 0`

Reproduced behavior for the saved `params`:

- `pruning_violations = []`
- `exact_violations = ["shared_memory: exact shared bytes upper bound ≤ limit: actual=65540"]`
- `final_violations = []`
- `concrete_final_result.ok = true`

Diagnostics classified the issue as:

- `root_causes = ["runtime_projection_upper_bound_insufficient"]`

This is a current false reject under the active path because exact rejects while concrete final validation accepts.

## Remaining Uncertainty

- This note establishes one current reproducer, not a full-count estimate.
- The search shard covered the first 150 tasks and up to 3 sketches per task, stopping at the first reproducer.
- The precise fix direction still needs review / root-cause ownership classification.

## Next Recommended Owner

- `reviewer` to confirm evidence sufficiency and classify ownership.
