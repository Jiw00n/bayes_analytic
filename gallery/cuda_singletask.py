import os
ppath = os.environ.get("PYTHONPATH")
buildpath = os.environ.get("TVM_LIBRARY_PATH")
print("PYTHONPATH=", ppath)
print("TVM_LIBRARY_PATH=", buildpath)
if "release" in buildpath:
    print("Release mode")
elif "debug" in buildpath:
    from util_manager import DebugManager
    debug_manager = DebugManager()
    debug_manager.debug_log_set(ppath, buildpath)
    print("Debug mode")
else:
    AssertionError("Set Environment release/debug")
# os.environ["USE_DAG_MOD"] = "1"
import numpy as np
from tvm import auto_scheduler
import tvm
from tvm import relay, tir
from tvm.auto_scheduler import SearchTask
from tvm.auto_scheduler import SketchPolicy
from tvm.contrib import graph_executor
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform
from tvm.relay.backend import te_compiler
from time import time

from util_manager import PathManager, get_network, get_arg

# os.environ["TVM_LOG_DEBUG"] = "relay/backend/te_compiler.cc=1"
# os.environ["TVM_LOG_DEBUG"] = "DEFAULT=0,relay/backend/te_compiler.cc=1"
# os.environ["GLOG_v"] = "1"
# export TVM_LOG_DEBUG="DEFAULT=0"



def get_mod(network_name, batch_size, layout, inner=False, dtype="float32"):

    print("Getting network : ", network_name)

    num_classes = 1000

    mod, params, input_shape, output_shape = get_network(
        network_name,
        batch_size,
        layout,
        dtype=dtype,
        num_classes=num_classes,
    )


    def basic_convert(mod, params):
        func = mod["main"]
        # func_bound = bind_params_by_name(func, params)
        # mod = tvm.IRModule.from_expr(func_bound)
        mod = relay.transform.SimplifyInference()(mod)
        mod = relay.transform.FoldScaleAxis()(mod)
        mod = relay.transform.FoldConstant()(mod)
        mod = relay.transform.InferType()(mod)
        return mod, params

    def experiment(mod, params, inner):
        # cuda = tvm.target.Target("cuda")
        # llvm = tvm.target.Target("llvm")
        # get_cfg = tvm.get_global_func("relay.backend.GetCompilationConfig", allow_missing=True)
        # if get_cfg is not None:
        #     # 보통 (mod, targets or multi-target) 형태를 받는다
        #     cfg = get_cfg(mod, [cuda, llvm])
        #     mod = relay.transform.PlanDevices(cfg)(mod)
        
        # mod = tvm.IRModule.from_expr(mod)
        
        passes = [
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.InferType(),
            relay.transform.Inline(),
            relay.transform.InferType(),
            relay.transform.LabelOps(),
            relay.transform.InferType(),
            relay.transform.FoldExplicitPadding(),
            relay.transform.InferType(),
        ]
        with tvm.transform.PassContext(opt_level=3, 
                                    config={"relay.backend.use_auto_scheduler": True,},
                                    disabled_pass={"AutoSchedulerLayoutRewrite"}):
                                    #    ):
            mod = tvm.transform.Sequential(passes)(mod)
            if inner:
                f = mod["inner"].with_attr("Primitive", tir.IntImm("int32", 1))
                mod.update_func(mod.get_global_var("inner"), f)
                mod = transform.InferType()(mod)
        return mod, params


    # mod, params = experiment(mod, params, inner)    
    mod, params = basic_convert(mod, params)

    return mod, params, input_shape, output_shape




def make_cached_func(mod, target, func_name="main"):
    print("Lowering..")
    tecompiler = te_compiler.get()
    cached_func = tecompiler.lower(mod[func_name], target)
    return cached_func

def make_dag_key(cached_func):
    print("Making DAG, key ..")
    dag = auto_scheduler.ComputeDAG(list(cached_func.inputs) + list(cached_func.outputs))
    key = dag.workload_key()
    auto_scheduler.workload_registry.register_workload_tensors(key, dag.tensors)
    breakpoint()
    return dag, key

def make_task_policy(dag, key, target):
    tasks = [SearchTask(
        workload_key = key,
        compute_dag=dag,
        target=target
    )]

    searchpolicy = SketchPolicy(
            tasks[0],
            auto_scheduler.XGBModel(),
            verbose=1,
    )
    return tasks, searchpolicy


inner = False


import argparse
parser = argparse.ArgumentParser(description="Ansor CUDA - Multiple Task Tuning")
args = get_arg(parser)

network = args.network
batch_size = args.batch_size
layout = args.layout
dtype = "float32"
target = tvm.target.Target("cuda")



# tiny_conv_1, tiny_res, resnet_18, resnet_50
network = "resnet_18"
batch_size = 1
# layout = "NHWC"
task_type = "single"


mod, params, input_shape, output_shape = get_mod(network, batch_size, layout, inner)
print(mod)

path_manager = PathManager(get_pkl=False, task_type=task_type, 
                           network=network, input_shape=input_shape, args=args)

paths = path_manager.paths

# other_json = "/root/work/tvm-ansor/gallery/logs_json/single-tiny_conv_1/(1,224,224,3)-0922_1117.json"
# path_manager.use_json(other_json)


print("Using json path:", paths["json"])



cached_func = make_cached_func(mod, target)
# breakpoint()

dag, key = make_dag_key(cached_func)

tasks, searchpolicy = make_task_policy(dag, key, target)

# init_states = searchpolicy.sample_initial_population()


# breakpoint()
# tune_option = auto_scheduler.TuningOptions(
#     num_measure_trials=1000,  # change this to 20000 to achieve the best performance
#     runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True)
# )





# breakpoint()

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3, min_repeat_ms=300, timeout=10000000)
    tuner = auto_scheduler.TaskScheduler(tasks, [1], tsv_log_path=paths["tsv"])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(paths["json"]),],
    )

    tuner.tune(tune_option)

run_tuning()




dev = tvm.device(str(target), 0)
input_nd = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype), dev)
output_nd = tvm.nd.array(np.zeros(output_shape, dtype=dtype), dev)
params_gpu = {k: tvm.nd.array(v, device=dev) for k, v in params.items()}
tensors_nd = [input_nd] + list(params_gpu.values()) + [output_nd]


# seed
np.random.seed(0)


# print("Compile...")
# with auto_scheduler.ApplyHistoryBest(paths["json"]):
#     with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
#         # breakpoint()
        # lib = relay.build(mod, target=target, params=params)


# module = graph_executor.GraphModule(lib["default"](dev))
# module.set_input("data", data_tvm)

# print(build_run(mod, params, target, input_nd))

# print("Evaluate inference time cost...")
# print(module.benchmark(dev, repeat=3, min_repeat_ms=500))




# from tvm.auto_scheduler import measure_record

# best_ctx = auto_scheduler.ApplyHistoryBest(paths["json"])
# mr = measure_record.load_best_record(paths["json"], key, target)

sch, tensors = tasks[0].apply_best(paths["json"])

# init_states = searchpolicy.sample_initial_population()
# sch, tensors = searchpolicy.search_task.compute_dag.apply_steps_from_state(init_states[0])


# breakpoint()
lo = tvm.lower(sch, tensors, name="default", simple_mode=True)
rt_mod = tvm.build(lo, target=target)

print("Build done")


# breakpoint()


func = rt_mod["default"]
func(*tensors_nd)
print(tensors_nd[-1])


# breakpoint()

evaluator = rt_mod.time_evaluator('default', dev, number=10, repeat=3)

res = evaluator(*tensors_nd).mean * 1000
# res = evaluator(input_nd, output_nd).mean
print(f"Latency (ms) : {res:.4f}")

breakpoint()

# print("Total time : ", time() - time_start)