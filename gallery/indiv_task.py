import os
ppath = os.environ.get("PYTHONPATH")
buildpath = os.environ.get("TVM_LIBRARY_PATH")
gdb_mode = os.environ.get("TVM_GDB_MODE")

# breakpoint()
import util_manager
if gdb_mode == "1":
    gdb_manager = util_manager.GDBManager()
    gdb_manager.gdb_log_set()

print("="*80)
print("PYTHONPATH :", ppath)
print("TVM_LIBRARY_PATH :", buildpath)

if buildpath.endswith("build"):
    print("DEBUG MODE")
elif buildpath.endswith("release"):
    print("RELEASE MODE")
else:
    AssertionError("Set Environment release/debug")

import numpy as np
from util_manager import PathManager, get_network, get_arg
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor
import argparse


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
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)
            # breakpoint()
    
    print(f"Total tasks length : {len(tasks)}")
    # breakpoint()
    return tasks, task_weights




def run_tuning(tasks, task_weights, paths):
    print("="*80)
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3, min_repeat_ms=300, timeout=10000)
    

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, tsv_log_path=paths["tsv"])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(paths["json"])],
    )
    # breakpoint()

    tuner.tune(tune_option)




def build_moudle_compile(paths, mod, params, input_shape, dtype):
    print("="*80)
    # Compile with the history best
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(paths['json']):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=TARGET, params=params)
            # breakpoint()


    # 어떤 스케줄에 어떤 소스 코드가 매핑되는지 확인해야하는 함수 만들 것
    # cuda_source = lib.lib.imported_modules[0].get_source()
    # llvm_source = lib.lib.get_source()
    # with open("resnet_18-cuda.cu", "w") as f:
    #     f.write(cuda_source)
    # with open("resnet_18-llvm.ll", "w") as f:
    #     f.write(llvm_source)


    # Create graph executor
    dev = tvm.device(str(TARGET), 0)
    module = graph_executor.GraphModule(lib["default"](dev))

    # breakpoint()
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)

    # Evaluate
    breakpoint()
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, repeat=3, min_repeat_ms=500))


def task_compile_run(tasks, dev, target):
    dev = tvm.device(str(target), 0)
    task = tasks[0]
    searchpolicy = auto_scheduler.SketchPolicy(task, program_cost_model=auto_scheduler.XGBModel(), verbose=1,)
    init_states = searchpolicy.sample_initial_population()
    # for i, task in enumerate(tasks):
    for i in range(min(10, len(init_states))):
        while True:
            try:
                sch, tensors = task.compute_dag.apply_steps_from_state(init_states[i])
                tensors_nd = []
                for tensor in tensors:
                    shape = tuple(int(s) for s in tensor.shape)
                    tensors_nd.append(tvm.nd.array((np.random.uniform(size=shape)).astype("float32"), dev))
                
                lo = tvm.lower(sch, tensors, name=task.desc, simple_mode=True)
                rt_mod = tvm.build(lo, target=target)
                # print(f"Task {i} : Build successfully")
                # breakpoint()
                break
            except Exception as e:
                print("Error during building the task.")

        # breakpoint()
        evaluator = rt_mod.time_evaluator(task.desc, dev, number=1, repeat=1)
        res = evaluator(*tensors_nd).mean * 1000
        print(f"{task.desc} latency (s) : {res:.4f}")


def main_(args):

    network = args.network
    batch_size = args.batch_size
    layout = args.layout
    dtype = "float32"

    # resnet_18, resnet_50
    network = "resnet_18"
    # batch_size = 1
    # layout = "NHWC"
    


    # 네트워크 불러오기
    mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)


    # 경로 설정
    # path_manager = PathManager(network, input_shape, args, gdb_mode)
    path_manager = PathManager(network, input_shape, args, gdb_mode, json="/root/work/tvm-ansor/gallery/logs_json/resnet_18/resnet_18-B1.json")


    # task 추출
    tasks, task_weights = get_tasks(mod, params, path_manager, verbose=False, get_pkl=True)

    
    # 튜닝
    # run_tuning(tasks, task_weights, path_manager.paths)


    # 스케줄 선택
    # schedule_sel(path_manager.paths)


    # build_module 컴파일
    # build_moudle_compile(path_manager.paths, mod, params, input_shape, dtype)

    # task 컴파일
    task_compile_run(tasks, dev=tvm.device(str(TARGET), 0), target=TARGET)




if __name__ == "__main__":
    print("="*80)
    parser = argparse.ArgumentParser(description="Ansor CUDA")
    args = get_arg(parser)

    main_(args)
