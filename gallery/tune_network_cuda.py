import os
ppath = os.environ.get("PYTHONPATH")
buildpath = os.environ.get("TVM_LIBRARY_PATH")
gdb_mode = os.environ.get("TVM_GDB_MODE")

# breakpoint()
import util_manager
if gdb_mode == "1":
    gdb_manager = util_manager.GDBManager()
    gdb_manager.gdb_log_set()

print("PYTHONPATH=", ppath)
print("TVM_LIBRARY_PATH=", buildpath)

if buildpath.endswith("build"):
    print("Debug mode")
elif buildpath.endswith("release"):
    print("Release mode")
else:
    AssertionError("Set Environment release/debug")

import numpy as np
from util_manager import PathManager, get_network, get_arg
import tvm
from tvm import relay, auto_scheduler
from tvm.contrib import graph_executor


import pickle


#################################################################
# Define a Network
# ----------------
# First, we need to define the network with relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX, PyTorch, and TensorFlow
# (see :ref:`front end tutorials<tutorial-frontend>`).
#
# For convolutional neural networks, although auto-scheduler can work correctly
# with any layout, we found the best performance is typically achieved with NHWC layout.
# We also implemented more optimizations for NHWC layout with the auto-scheduler.
# So it is recommended to convert your models to NHWC layout to use the auto-scheduler.
# You can use :ref:`ConvertLayout <convert-layout-usage>` pass to do the layout conversion in TVM.






import argparse
parser = argparse.ArgumentParser(description="Ansor CUDA")
args = get_arg(parser)


network = args.network
batch_size = args.batch_size
layout = args.layout
dtype = "float32"

# resnet_18, resnet_50
network = "resnet_18"
batch_size = 1
# layout = "NHWC"

target = tvm.target.Target("cuda")


get_pkl = True




print("Extract tasks...")

mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
# print(mod)

path_manager = PathManager(get_pkl=get_pkl, 
                           network=network, input_shape=input_shape, args=args, gdb_mode=gdb_mode)
paths = path_manager.paths

# 있는 json 파일 사용
path_manager.use_json("/root/work/tvm-ansor/gallery/logs_json/resnet_18/resnet_18-B1.json")



if get_pkl:
    task_pack = path_manager.use_tasks_pkl()
    if task_pack is None:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        path_manager.save_tasks_pkl(tasks, task_weights)
    else:
        tasks, task_weights = task_pack
else:
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)



# breakpoint()


print("Using json path:", paths["json"])

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3, min_repeat_ms=300, timeout=10)
    

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, tsv_log_path=paths["tsv"])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(paths["json"])],
    )
    # breakpoint()

    tuner.tune(tune_option)


# run_tuning()




# Compile with the history best
print("Compile...")
with auto_scheduler.ApplyHistoryBest(paths['json']):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)
        # breakpoint()


# 어떤 스케줄에 어떤 소스 코드가 매핑되는지 확인해야하는 함수 만들 것
# cuda_source = lib.lib.imported_modules[0].get_source()
# llvm_source = lib.lib.get_source()
# with open("resnet_18-cuda.cu", "w") as f:
#     f.write(cuda_source)
# with open("resnet_18-llvm.ll", "w") as f:
#     f.write(llvm_source)


# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))

# breakpoint()
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))