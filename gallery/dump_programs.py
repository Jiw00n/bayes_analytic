"""Dump programs for all tasks"""

import argparse
import pickle
import gc
import glob
import time
import os

from tqdm import tqdm

from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import load_records, dump_record_to_string

from common import load_and_register_tasks, get_to_measure_filename, load_and_register_network, TO_MEASURE_PROGRAM_FOLDER

# ─── sketch fingerprint (record_loader와 동일 로직) ───
def _step_structural_fingerprint(step):
    tk = step.type_key.split(".")[-1]
    if tk == "AnnotationStep":
        return (tk, int(step.stage_id), int(step.iter_id), int(step.annotation))
    elif tk == "FuseStep":
        return (tk, int(step.stage_id), tuple(int(x) for x in step.fused_ids))
    elif tk == "PragmaStep":
        ptype = str(step.pragma_type).split("$")[0]
        return (tk, int(step.stage_id), int(step.iter_id), ptype)
    elif tk == "ReorderStep":
        return (tk, int(step.stage_id), tuple(int(x) for x in step.after_ids))
    elif tk == "SplitStep":
        return (tk, int(step.stage_id), int(step.iter_id),
                len(step.lengths), bool(step.inner_to_outer))
    elif tk == "FollowSplitStep":
        return (tk, int(step.stage_id), int(step.iter_id),
                int(step.src_step_id), int(step.n_split))
    elif tk == "FollowFusedSplitStep":
        return (tk, int(step.stage_id), int(step.iter_id),
                tuple(int(x) for x in step.src_step_ids),
                int(step.level), bool(step.factor_or_nparts))
    elif tk == "StorageAlignStep":
        return (tk, int(step.stage_id), int(step.iter_id),
                int(step.factor), int(step.offset))
    elif tk == "ComputeAtStep":
        return (tk, int(step.stage_id), int(step.target_stage_id), int(step.target_iter_id))
    elif tk == "ComputeInlineStep":
        return (tk, int(step.stage_id))
    elif tk == "ComputeRootStep":
        return (tk, int(step.stage_id))
    elif tk == "CacheReadStep":
        return (tk, int(step.stage_id), str(step.scope_name),
                tuple(int(x) for x in step.reader_stage_ids))
    elif tk == "CacheWriteStep":
        return (tk, int(step.stage_id), str(step.scope_name))
    else:
        return (tk, int(step.stage_id))


def state_sketch_fingerprint(state):
    return tuple(_step_structural_fingerprint(s) for s in state.transform_steps)


# ─── all_sketches.json 관리 ───
ALL_SKETCHES_PATH = os.path.join(TO_MEASURE_PROGRAM_FOLDER, "all_sketches.json")
_sketch_seen = None  # lazy-loaded set of (workload_key, sketch_fp)


def _load_known_sketches():
    """all_sketches.json에서 기존 (wkey, sketch_fp) 집합을 로드."""
    global _sketch_seen
    _sketch_seen = set()
    if os.path.exists(ALL_SKETCHES_PATH):
        for inp, res in load_records(ALL_SKETCHES_PATH):
            wkey = inp.task.workload_key
            fp = state_sketch_fingerprint(inp.state)
            _sketch_seen.add((wkey, fp))


def _append_new_sketches(task, states):
    """생성된 states에서 새로운 sketch를 all_sketches.json에 추가."""
    global _sketch_seen
    if _sketch_seen is None:
        _load_known_sketches()

    new_lines = []
    wkey = task.workload_key
    for state in states:
        fp = state_sketch_fingerprint(state)
        key = (wkey, fp)
        if key not in _sketch_seen:
            _sketch_seen.add(key)
            inp = auto_scheduler.MeasureInput(task, state)
            res = auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time())
            new_lines.append(dump_record_to_string(inp, res))

    if new_lines:
        os.makedirs(os.path.dirname(ALL_SKETCHES_PATH), exist_ok=True)
        with open(ALL_SKETCHES_PATH, "a") as f:
            for line in new_lines:
                f.write(line)


def dump_program(task, size, network_name=None, max_retry_iter=1):
    filename = get_to_measure_filename(task, network_name)
    if os.path.exists(filename):
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    policy = auto_scheduler.SketchPolicy(task, auto_scheduler.XGBModel(), verbose=0)


    # Generate unique states
    init_states = policy.sample_initial_population()
    states = policy.evolutionary_search(init_states, size)


    # Append new sketches to all_sketches.json (before saving records)
    _append_new_sketches(task, states)

    # Make measure inputs and results
    measure_inputs = []
    measure_results = []
    for state in states:
        measure_inputs.append(auto_scheduler.MeasureInput(task, state))
        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

    # Dump to file
    auto_scheduler.save_records(filename, measure_inputs, measure_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--size", type=int, default=2000)
    args = parser.parse_args()


    tasks = load_and_register_tasks()

    # Dump programs for all tasks
    for task in tqdm(tasks):
        dump_program(task, size=args.size)
        gc.collect()

    # all_tasks = {}
    # tasks_paths = glob.glob("/root/work/tvm-ansor/gallery/dataset/network_info/*.task.pkl")

    # for idx, network_task_path in enumerate(tasks_paths):
        
    #     network_name = os.path.basename(network_task_path).replace(".task.pkl", "")
    #     tasks = load_and_register_network(network_task_path)
    #     all_tasks[network_name] = tasks

    # total_tasks = sum(len(tasks) for tasks in all_tasks.values())

    # with tqdm(total=total_tasks) as pbar:
    #     for network_name, tasks in all_tasks.items():
    #         for task in tasks:
    #             dump_program(network_name, task, size=args.size)
    #             gc.collect()
    #             pbar.update(1)