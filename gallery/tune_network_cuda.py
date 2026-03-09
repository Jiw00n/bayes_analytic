import os

from tvm.auto_scheduler.search_policy import CustomPrintState



ppath = os.environ.get("PYTHONPATH")
buildpath = os.environ.get("TVM_LIBRARY_PATH")
gdb_mode = os.environ.get("TVM_GDB_MODE")
use_ncu = os.environ.get("USE_NCU")

# breakpoint()
import util_manager
if gdb_mode == "1":
    gdb_manager = util_manager.GDBManager()
    gdb_manager.gdb_log_set()

print("="*80)
print("PYTHONPATH :", ppath)
print("TVM_LIBRARY_PATH :", buildpath)
print("USE_NCU :", use_ncu)

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
from tvm.auto_scheduler.feature import get_per_store_features_from_file
from tvm.auto_scheduler.search_task import HardwareParams

TARGET = tvm.target.Target("cuda")

# HW_PARAM = HardwareParams(num_cores=2147483647, vector_unit_bytes=2147483647, cache_line_bytes=2147483647, max_shared_memory_per_block=2147483647, max_local_memory_per_block=2147483647, max_threads_per_block=2147483647, max_vthread_extent=2147483647, warp_size=32)

def get_tasks(mod, params, path_manager, verbose=True, get_pkl=True):
    if get_pkl:
        tasks, task_weights = path_manager.tasks_pkl_use()
        # tasks, task_weights = path_manager.tasks_pkl_use("resnet_18-(1,224,224,3)-hw_param.pkl")
    
    if get_pkl is False or tasks is None:
        print("Extract tasks...")
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, TARGET)
        if path_manager.tasks_pkl_check() is False:
            path_manager.tasks_pkl_save(tasks, task_weights)

    if verbose:
        for idx, task in enumerate(tasks):
            print("========== Task %d : %s  (workload key: %s) ==========" % (idx, task.desc, task.workload_key))
            print(task.compute_dag)
            # breakpoint()
    
    print(f"Total tasks length : {len(tasks)}")
    breakpoint()
    return tasks, task_weights




def run_tuning(tasks, task_weights, paths):
    print("="*80)
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
    

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

    breakpoint()


    raw_features, raw_normalized_throughputs, task_ids = get_per_store_features_from_file(paths['json'], 10000)
    
    # 어떤 스케줄에 어떤 소스 코드가 매핑되는지 확인해야하는 함수 만들 것
    cuda_source = lib.lib.imported_modules[0].get_source()
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
    # print(module.benchmark(dev, repeat=1, min_repeat_ms=500))
    print(module.benchmark(dev, repeat=1))



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
    path_manager = PathManager(network, input_shape, args, gdb_mode)
    # path_manager = PathManager(network, input_shape, args, gdb_mode, json="/root/work/tvm-ansor/gallery/logs_json/resnet_18/resnet_18-B1.json")
    # path_manager = PathManager(network, input_shape, args, gdb_mode, json="/root/work/tvm-ansor/gallery/logs_json/tmp.json")


    # task 추출
    tasks, task_weights = get_tasks(mod, params, path_manager, verbose=False, get_pkl=True)

    # breakpoint()
    
    # 튜닝
    run_tuning(tasks, task_weights, path_manager.paths)

    inputs_ = auto_scheduler.RecordReader(path_manager.paths['json']).read_lines()
    inputs = []
    wk_desc_states = ""
    for i in inputs_[0]:
        for t in tasks:
            if i.task.workload_key == t.workload_key:
                desc = t.desc
        new_i = auto_scheduler.measure.recover_measure_input(i, True, desc=desc)
        init_pop_str = CustomPrintState(new_i.state, delete_trivial_loop=True, dag_show=False)
        # breakpoint()
        
        wk_desc_states += desc + "\n"
        wk_desc_states += new_i.task.workload_key + "\n"
        wk_desc_states += init_pop_str + "\n\n\n"
    with open("tmp_states.txt", "w") as f:
        f.write(wk_desc_states)
    breakpoint()
    


    # build_module 컴파일
    build_moudle_compile(path_manager.paths, mod, params, input_shape, dtype)



if __name__ == "__main__":
    print("="*80)
    parser = argparse.ArgumentParser(description="Ansor CUDA")
    args = get_arg(parser)

    main_(args)
