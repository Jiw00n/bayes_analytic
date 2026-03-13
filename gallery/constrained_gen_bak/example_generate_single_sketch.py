"""Minimal example: load one sketch, generate params once, and save the result.

Run from repo root:

    source /root/work/venv/bin/activate
    export TVM_HOME=/root/work/tvm-ansor
    export PYTHONPATH=$TVM_HOME/python
    export TVM_LIBRARY_PATH=$TVM_HOME/build-release
    python gallery/constrained_gen/example_generate_single_sketch.py
"""

import json
import os
import random

from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import save_records

from modules.common import TO_MEASURE_PROGRAM_FOLDER
from modules.param_manager import build_symbolic_state
from modules.projected_gpu_validation import load_sketch_lines, load_sketch_record, load_tasks_by_workload
from modules.schedule_generator import ScheduleGenerator
from modules.tvm_verify import lower_with_gpu_passes, params_to_state, verify_gpu_module_errors


# Change only this index if you want to try another sketch.
SKETCH_INDEX = 2

NETWORK_INFO_DIR = "/root/work/tvm-ansor/gallery/dataset/network_info"
SKETCHES_PATH = f"{TO_MEASURE_PROGRAM_FOLDER}/all_sketches.json"
OUTPUT_DIR = "/tmp/projected_gpu_full_validation/example_generate_single_sketch"
RNG_SEED = 0
MAX_RETRIES = 16


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[1] load tasks from {NETWORK_INFO_DIR}")
    _, tasks_by_wkey = load_tasks_by_workload(NETWORK_INFO_DIR)

    print(f"[2] load sketch lines from {SKETCHES_PATH}")
    sketch_lines = load_sketch_lines(SKETCHES_PATH)
    line = sketch_lines[SKETCH_INDEX]
    if not line:
        raise RuntimeError(f"Sketch index {SKETCH_INDEX} is empty")

    print(f"[3] decode sketch record at index {SKETCH_INDEX}")
    task, base_inp, base_res, base_state = load_sketch_record(line, tasks_by_wkey)
    print(f"    task_desc={task.desc}")
    print(f"    workload_key={task.workload_key}")

    print("[4] build symbolic state from the sketch state")
    sym_state = build_symbolic_state(task, base_state)

    print("[5] build schedule generator from the symbolic state")
    gen = ScheduleGenerator(
        sym_state,
        task=task,
        base_input=base_inp,
        base_result=base_res,
    )

    print("[6] sample one concrete param assignment")
    rng = random.Random(RNG_SEED)
    params = gen.randomize_params(rng=rng, max_retries=MAX_RETRIES)
    print(f"    params={json.dumps(params, sort_keys=True)}")

    print("[7] turn params back into a concrete auto_scheduler.State")
    new_state = params_to_state(task, base_inp, base_res, params)

    print("[8] lower with GPU passes and verify the generated schedule")
    mod = lower_with_gpu_passes(task, new_state)
    verify_errors = verify_gpu_module_errors(mod)
    if verify_errors:
        raise RuntimeError(
            "Generated schedule did not pass GPU verification: "
            + "; ".join(verify_errors)
        )
    print("    GPU verification passed")

    print("[9] save one generated MeasureInput/MeasureResult record")
    output_record_path = os.path.join(OUTPUT_DIR, f"sketch_{SKETCH_INDEX}_generated.json")
    save_records(
        output_record_path,
        [auto_scheduler.MeasureInput(task, new_state)],
        [base_res],
    )
    print(f"    saved record -> {output_record_path}")

    print("[10] save a small human-readable summary next to the record")
    output_summary_path = os.path.join(OUTPUT_DIR, f"sketch_{SKETCH_INDEX}_summary.json")
    summary = {
        "sketch_index": SKETCH_INDEX,
        "task_desc": task.desc,
        "workload_key": task.workload_key,
        "record_path": output_record_path,
        "params": params,
        "verify_errors": verify_errors,
    }
    with open(output_summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"    saved summary -> {output_summary_path}")

    print("done")


if __name__ == "__main__":
    main()
