"""Dataset path, task-loading, and legacy utility helpers for constrained-gen.

The active ownership here is dataset/path helpers and TVM task registration.
The remaining formatting or framework helpers are legacy shared utilities used
by adjacent scripts.
"""

import argparse
from collections import namedtuple
import pickle

import tvm
from tvm import auto_scheduler, relay
from tvm.auto_scheduler.utils import to_str_round

# -----------------------------------------------------------------------------
# Path and task-loading helpers
# -----------------------------------------------------------------------------

NETWORK_INFO_FOLDER = '/root/work/tvm-ansor/gallery/dataset/network_info'
TO_MEASURE_PROGRAM_FOLDER = '/root/work/tvm-ansor/gallery/dataset/to_measure_programs'
TO_MEASURE_GEN_PROGRAM_FOLDER = '/root/work/tvm-ansor/gallery/dataset/to_measure_gen_programs'
TO_MEASURE_NETWORK_FOLDER = '/root/work/tvm-ansor/gallery/dataset/to_measure_networks'
MEASURED_FOLDER = '/root/work/tvm-ansor/gallery/dataset/measured_gen_programs'


def clean_name(x):
    """파일명 등에 쓸 수 있도록 문자열에서 공백·따옴표를 제거한다."""
    x = str(x)
    x = x.replace(" ", "")
    x = x.replace('"', '')
    x = x.replace("'", '')
    return x

def get_to_measure_gen_filename(task, output_dir=TO_MEASURE_GEN_PROGRAM_FOLDER):
    """생성된 측정 대상 프로그램 JSON 파일 경로를 반환한다."""
    task_key = (task.workload_key, str(task.target.kind))
    return f"{output_dir}/{clean_name(task_key)}.json"


def _register_task_workloads(tasks):
    """task 목록의 workload를 TVM 워크로드 레지스트리에 등록한다."""
    for task in tasks:
        auto_scheduler.workload_registry.register_workload_tensors(
            task.workload_key, task.compute_dag.tensors)


def load_and_register_tasks(network_info_folder=NETWORK_INFO_FOLDER):
    """all_tasks.pkl에서 task 목록을 로드하고 워크로드를 등록한 뒤 반환한다."""
    tasks = pickle.load(open(f"{network_info_folder}/all_tasks.pkl", "rb"))
    _register_task_workloads(tasks)
    return tasks


# -----------------------------------------------------------------------------
# Deprecated
# -----------------------------------------------------------------------------


# def convert_to_nhwc(mod):
#     """Relay 모듈을 NHWC 레이아웃으로 변환한다."""
#     desired_layouts = {
#         "nn.conv2d": ["NHWC", "default"],
#         "nn.conv3d": ["NDHWC", "default"],
#     }
#     seq = tvm.transform.Sequential(
#         [
#             relay.transform.RemoveUnusedFunctions(),
#             relay.transform.ConvertLayout(desired_layouts),
#         ]
#     )
#     with tvm.transform.PassContext(opt_level=3):
#         mod = seq(mod)
#     return mod


# # The format for a line in the results file.
# BenchmarkRecord = namedtuple(
#     "BenchmarkRecord",
#     ['device', 'backend', 'workload_type', 'workload_name',
#      'library', 'algorithm', 'value', 'time_stamp'],
# )


# def log_line(record, out_file):
#     """벤치마크 레코드 한 줄을 TSV 형식으로 파일에 추가한다."""
#     with open(out_file, 'a') as fout:
#         fout.write("\t".join([to_str_round(x) for x in record]) + '\n')


# def get_relay_ir_filename(network_key):
#     """network_key에 대응하는 Relay IR 피클 파일 경로를 반환한다."""
#     return f"{NETWORK_INFO_FOLDER}/{clean_name(network_key)}.relay.pkl"


# def get_task_info_filename(network_key, target):
#     """(network_key, target)에 대응하는 task 정보 피클 파일 경로를 반환한다."""
#     network_task_key = (network_key,) + (str(target.kind),)
#     return f"{NETWORK_INFO_FOLDER}/{clean_name(network_task_key)}.task.pkl"


# def get_to_measure_filename(task, network_name=None):
#     """측정 대상 프로그램 JSON 파일 경로를 반환한다 (선택적으로 network_name 서브디렉터리)."""
#     task_key = (task.workload_key, str(task.target.kind))
#     if network_name is not None:
#         return f"{TO_MEASURE_PROGRAM_FOLDER}/{network_name}/{clean_name(task_key)}.json"
#     return f"{TO_MEASURE_PROGRAM_FOLDER}/{clean_name(task_key)}.json"


# def get_measure_record_filename(task, target=None):
#     """측정 완료된 레코드 JSON 파일 경로를 반환한다."""
#     target = target or task.target
#     task_key = (task.workload_key, str(target.kind))
#     return f"{MEASURED_FOLDER}/{target.model}/{clean_name(task_key)}.json"


# def load_and_register_network(network_task_path=TO_MEASURE_NETWORK_FOLDER):
#     """네트워크(task+가중치) 피클을 로드하고 워크로드를 등록한 뒤 (tasks, weights)를 반환한다."""
#     tasks, task_weights = pickle.load(open(network_task_path, "rb"))
#     _register_task_workloads(tasks)
#     return tasks


# def dtype2torch(x):
#     """TVM dtype 문자열을 PyTorch dtype으로 변환한다."""
#     import torch

#     return {
#         'float32': torch.float32
#     }[x]


# def str2bool(v):
#     """문자열/불리언을 bool로 파싱한다 (argparse 등에서 사용)."""
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     if v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     raise argparse.ArgumentTypeError('Boolean value expected.')
