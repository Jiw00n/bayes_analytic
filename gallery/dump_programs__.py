from util_manager import PathManager, get_network
import tvm
from tvm import auto_scheduler
import os
from types import SimpleNamespace


TARGET = tvm.target.Target("cuda")

def get_tasks(mod, params, path_manager, verbose=True, get_pkl=True):
    if get_pkl:
        tasks, task_weights = path_manager.tasks_pkl_use()
    
    if get_pkl is False or tasks is None:
        print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, TARGET)
        if path_manager.tasks_pkl_check() is False:
            path_manager.tasks_pkl_save(tasks, task_weights)

    if verbose:
        for idx, task in enumerate(tasks):
            print("========== Task %d : %s  (workload key: %s) ==========" % (idx, task.desc, task.workload_key))
            print(task.compute_dag)
    
    print(f"Total tasks length : {len(tasks)}")
    return tasks, task_weights


def dump_programs(search_policies, tasks, dum_dir):
    import time
    from tvm.auto_scheduler.measure_record import save_records
    def clean_name(x):
        x = str(x)
        x = x.replace(" ", "")
        x = x.replace("\"", "")
        x = x.replace("'", "")
        return x
    def get_dump_filename(dump_dir, task):
        task_key = (task.workload_key, str(task.target.kind))
        return f"{dump_dir}/{clean_name(task_key)}.json"


    for task_idx, (search_policy, task) in enumerate(zip(search_policies, tasks)):
        dump_json = get_dump_filename(dump_dir, task)
        if os.path.exists(dump_json):
            print(f"Task {task_idx} already dumped at {dump_json}, skipping...")
            continue
        init_states = search_policy.sample_initial_population()
        states = search_policy.evolutionary_search(init_states, 2000)
        measure_inputs = []
        measure_results = []
        for state in states:
            measure_inputs.append(auto_scheduler.MeasureInput(task, state))
            measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))
        save_records(dump_json, measure_inputs, measure_results)
        print(f"Dumped task {task_idx} to {dump_json}")
        print("=" * 60)
    print(f"All tasks dumped to {dump_dir}")


def main(args, dump_dir):
    mod, params, input_shape, output_shape = get_network(args.network, args.batch_size, args.layout, dtype=args.dtype)
    path_manager = PathManager(args.network, input_shape, args, None)
    tasks, task_weights = get_tasks(None, params, path_manager, verbose=False, get_pkl=True)
    tasks, task_weights = zip(*sorted(zip(tasks, task_weights), key=lambda x: x[0].desc))

    search_policies = []
    for idx, (task, weight) in enumerate(zip(tasks, task_weights)):
        print(f"T{idx} : {task.desc} ({weight})")
        search_policies.append(
            auto_scheduler.SketchPolicy(task, auto_scheduler.XGBModel())
        )
    
    dump_programs(search_policies, tasks, dump_dir)




if __name__ == "__main__":
    
    args = SimpleNamespace(
        network="resnet_50",
        batch_size=1,
        dtype="float32",
        layout="NHWC",
        timenow=None,
        json=None
    )
    dump_dir = f"/root/work/tvm-ansor/gallery/dump_programs/{args.network}"
    main(args, dump_dir)
