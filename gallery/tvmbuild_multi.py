# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Auto-scheduling a Neural Network for NVIDIA GPU
===============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole neural
network for NVIDIA GPU with the auto-scheduler.

To auto-tune a neural network, we partition the network into small subgraphs and 
tune them independently. Each subgraph is treated as one search task.
A task scheduler slices the time and dynamically allocates time resources to
these tasks. The task scheduler predicts the impact of each task on the end-to-end
execution time and prioritizes the one that can reduce the execution time the most.

For each subgraph, we use the compute declaration in :code:`tvm/python/topi` to
get the computational DAG in the tensor expression form.
We then use the auto-scheduler to construct a search space of this DAG and search
for good schedules (low-level optimizations).

Different from the template-based :ref:`autotvm <tutorials-autotvm-sec>` which relies on
manual templates to define the search space, the auto-scheduler does not require any
schedule templates. In other words, the auto-scheduler only uses the compute declarations
in :code:`tvm/python/topi` and does not use existing schedule templates.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""
import os
ppath = os.environ.get("PYTHONPATH")
buildpath = os.environ.get("TVM_LIBRARY_PATH")
print("PYTHONPATH=", ppath)
print("TVM_LIBRARY_PATH=", buildpath)
if "release" in buildpath:
    print("Release mode")
else:
    print("Debug mode")
import numpy as np
from datetime import datetime
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm.relay.testing.init import create_workload
from time import time
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




def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""
    
    print("Getting network : ", name)

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet_"):
        n_layer = int(name.split("_")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )

    elif name.startswith("tiny_conv"):
        layer_n = int(name.split("_")[-1])
        net = relay.testing.resnet.tiny_convnet_nhwc(layer_n=layer_n)        # input shape까지 relay에 이미 정의됨
        mod, params = create_workload(net)
        input_shape = (batch_size, 32, 32, 3)
        output_shape = (batch_size, 10)
        # from tvm.relay import transform
        # mod = transform.InferType()(mod)
        # mod = transform.SimplifyInference()(mod)

    elif name.startswith("tiny_res"):
        net = relay.testing.resnet.tiny_resnet_nhwc()
        mod, params = create_workload(net)
        input_shape = (batch_size, 32, 32, 3)
        output_shape = (batch_size, 10)
        from tvm.relay import transform
        mod = transform.InferType()(mod)
        mod = transform.SimplifyInference()(mod)

    elif name.startswith("resnet3d_"):
        n_layer = int(name.split("_")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape

import argparse

parser = argparse.ArgumentParser(description="Ansor CUDA - Multiple Task Tuning")
parser.add_argument("--network", type=str, help="Name of the network", default="tiny_conv_1")
parser.add_argument("--batch_size", type=int, help="Batch size", default=1)
parser.add_argument("--layout", type=str, help="Layout of the input data", default="NHWC")
parser.add_argument("--measure", action="store_true", help="Whether to measure or not")
args = parser.parse_args()


network = args.network
batch_size = args.batch_size
layout = args.layout
measure = args.measure

# tiny_conv_1, tiny_res, resnet_18, resnet_50
network = "resnet_18"
# batch_size = 1
# layout = "NHWC"

tvm_gallery = "/root/work/tvm-ansor/gallery"

target = tvm.target.Target("cuda")
dtype = "float32"
# os.makedirs(f"{tvm_gallery}/logs_json/multi-{network}", exist_ok=True)
if measure:
    time = datetime.now().strftime("%m%d-%H%M")
    log_file = f"{tvm_gallery}/logs_json/multi-{network}/{network}-B{batch_size}-{time}.json"
else:
    log_file = f"{tvm_gallery}/logs_json/multi-{network}/{network}-B{batch_size}.json"
log_file = f"/root/work/tvm-ansor/gallery/logs_json/resnet_18/resnet_18-B1.json"



# Extract tasks from the network
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
# print(mod)

time_start = time()

import pickle

no_pkl = False

if os.path.exists(f"{tvm_gallery}/ansor_tasks_pkl/{network}-{str(input_shape)}.pkl") and not no_pkl:
    with open(f"{tvm_gallery}/ansor_tasks_pkl/{network}-{str(input_shape)}.pkl", "rb") as f:
        tasks, task_weights = pickle.load(f)
elif not no_pkl:
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    os.makedirs("{tvm_gallery}/ansor_tasks_pkl", exist_ok=True)
    with open(f"{tvm_gallery}/ansor_tasks_pkl/{network}-{str(input_shape)}.pkl", "wb") as f:
        pickle.dump((tasks, task_weights), f)
else:
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

# breakpoint()




def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)
    
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=100,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    # breakpoint()

    tuner.tune(tune_option)

# run_tuning()



# time_mid = time() - time_start



# ncu --set full --target-processes all python indiv_task.py 2>&1 | tee resnet18_indiv_task.log


results = []
dev = tvm.device(str(target), 0)
# for i, task in enumerate(tasks):
for i in range(10):
    task = tasks[0]
    searchpolicy = auto_scheduler.SketchPolicy(task, program_cost_model=auto_scheduler.XGBModel(), verbose=1,)
    while True:
        try:
            init_states = searchpolicy.sample_initial_population()
            sch, tensors = searchpolicy.search_task.compute_dag.apply_steps_from_state(init_states[0])
            # sch, tensors = task.apply_best(log_file)
            tensors_nd = []
            # breakpoint()
            for tensor in tensors:
                shape = tuple(int(s) for s in tensor.shape)
                tensors_nd.append(tvm.nd.array((np.random.uniform(size=shape)).astype("float32"), dev))
            
            lo = tvm.lower(sch, tensors, name=f"default{i}", simple_mode=True)
            rt_mod = tvm.build(lo, target=target)
            # print(f"Task {i} : Build successfully")
            # breakpoint()
            break
        except Exception as e:
            print("Error during building the task.")

    breakpoint()
    evaluator = rt_mod.time_evaluator(f'default{i}', dev, number=10, repeat=3)
    res = evaluator(*tensors_nd).mean
    total = res * np.float64(task_weights[i])
    # print(f"{i+1}/{len(tasks)} latency (s) : {res:.8f} * {task_weights[i]} = {total:.8f}")
    results.append(total)
    
    
# print("all results:", results)
print(f"Sum (s) : {sum(results):.8f}")


# breakpoint()

# if not os.path.exists(log_file):
#     print("No log file. Tuning...")
#     run_tuning()

# print("Compile...")
# with auto_scheduler.ApplyHistoryBest(log_file):
#     with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
#         lib = relay.build(mod, target=target, params=params)
#         # breakpoint()



# # Create graph executor
# dev = tvm.device(str(target), 0)
# module = graph_executor.GraphModule(lib["default"](dev))

# # breakpoint()
# data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
# module.set_input("data", data_tvm)

# # Evaluate
# print("Evaluate inference time cost...")
# print(module.benchmark(dev, repeat=3, min_repeat_ms=500))