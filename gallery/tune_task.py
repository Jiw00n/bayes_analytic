import os
import tvm
from tvm import auto_scheduler
from common import load_and_register_tasks, clean_name

TARGET = tvm.target.Target("cuda")

GALLERY_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_INFO_FOLDER = os.path.join(GALLERY_DIR, "dataset", "network_info_all")
MEASURE_EXP_DIR = os.path.join(GALLERY_DIR, "data_ansor_tune")


def resolve_paths(task, task_idx):
    task_key = (task.workload_key, str(task.target.kind))
    basename = f"{task_idx}_{clean_name(task_key)}"
    os.makedirs(MEASURE_EXP_DIR, exist_ok=True)
    return {
        "json": os.path.join(MEASURE_EXP_DIR, f"{basename}.json"),
        "tsv": os.path.join(MEASURE_EXP_DIR, f"{basename}.tsv"),
    }


def run_tuning(tasks, task_weights, paths):
    print("="*80)
    print("Begin tuning...")
    print(f"json log : {paths['json']}")
    print(f"tsv log  : {paths['tsv']}")

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3, min_repeat_ms=300, timeout=10)

    load_log_file = paths["json"] if os.path.isfile(paths["json"]) else None

    tuner = auto_scheduler.TaskScheduler(
        tasks,
        task_weights,
        load_log_file=load_log_file,
        tsv_log_path=paths["tsv"],
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=4000,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(paths["json"])],
    )

    tuner.tune(tune_option)


def main_():
    task_idx = 1490

    all_tasks = load_and_register_tasks(NETWORK_INFO_FOLDER)
    assert 0 <= task_idx < len(all_tasks), (
        f"task_idx={task_idx} out of range [0, {len(all_tasks)})"
    )

    task = all_tasks[task_idx]
    print(f"Selected task [{task_idx}] : {task.desc}")
    print(f"workload_key : {task.workload_key}")

    tasks = [task]
    task_weights = [1]

    paths = resolve_paths(task, task_idx)

    run_tuning(tasks, task_weights, paths)


if __name__ == "__main__":
    print("="*80)
    main_()
