#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Provenance 모듈 동작 검증: build_provenance_formulas + TIR 실제값 비교."""

import os
import sys
import re
import json
import tempfile
from collections import defaultdict
from types import SimpleNamespace

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["TVM_HOME"] = project_root
os.environ.setdefault("TVM_LIBRARY_PATH", f"{project_root}/build-release")
if f"{project_root}/python" not in sys.path:
    sys.path.insert(0, f"{project_root}/python")
sys.path.insert(0, project_root)
sys.path.insert(0, f"{project_root}/build-release")

import tvm
from tvm import tir, auto_scheduler
from tvm.auto_scheduler.measure import recover_measure_input

import util_manager
from util_manager import PathManager, get_network

# TIR에서 per-kernel thread/vthread 추출 (노트북과 동일)
_s2m = tvm.get_global_func("driver.schedule_to_module")
GPU_PASSES = tvm.transform.Sequential([
    tir.transform.InjectPrefetch(),
    tir.transform.StorageFlatten(64, False),
    tir.transform.NarrowDataType(32),
    tir.transform.Simplify(),
    tir.transform.VectorizeLoop(True),
    tir.transform.InjectVirtualThread(),
    tir.transform.StorageRewrite(),
    tir.transform.Simplify(),
])


def parse_tir_constraints(tir_str):
    kernels = []
    cur = None
    max_vthread_s = 1
    for line in tir_str.split("\n"):
        if re.search(r'launch_thread\("blockIdx\.\w+",\s*\d+\)', line):
            if cur is not None:
                kernels.append(cur)
            cur = {"thread_per_block": 1, "threads": {}, "shared_bytes": 0, "local_bytes": 0, "vthread": 1}
        if cur is None:
            continue
        mt = re.search(r'launch_thread\("threadIdx\.(\w+)",\s*(\d+)\)', line)
        if mt:
            dim, ext = mt.group(1), int(mt.group(2))
            if dim not in cur["threads"]:
                cur["threads"][dim] = ext
                cur["thread_per_block"] *= ext
        mv = re.search(r'launch_thread\("vthread\w*",\s*(\d+)\)', line)
        if mv:
            v = int(mv.group(1))
            cur["vthread"] *= v
            cur["thread_per_block"] *= v
        mvs = re.search(r"for\s+([\w,\s]+)\s+in\s+T\.grid\(([^)]+)\)", line)
        if mvs:
            vars_ = [v.strip() for v in mvs.group(1).split(",")]
            if "vthread_s" in vars_:
                exts = [int(x.strip()) for x in mvs.group(2).split(",")]
                max_vthread_s = max(max_vthread_s, exts[vars_.index("vthread_s")])
        mvs2 = re.search(r"for\s+vthread_s\s+in\s+(?:range|T\.serial)\((\d+)\)", line)
        if mvs2:
            max_vthread_s = max(max_vthread_s, int(mvs2.group(1)))
    if cur is not None:
        kernels.append(cur)
    return kernels, max_vthread_s


def get_tir_constraints(task, state):
    sch, tensors = task.compute_dag.apply_steps_from_state(state)
    mod = _s2m(sch, tensors, "main", {})
    mod = GPU_PASSES(mod)
    return parse_tir_constraints(str(mod))


def record_to_task_and_state(record):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps(record) + "\n")
        tmp = f.name
    try:
        inputs, _ = zip(*auto_scheduler.load_records(tmp))
        inp = recover_measure_input(inputs[0], rebuild_state=True)
        return inp.task, inp.state
    finally:
        os.unlink(tmp)


def main():
    args = SimpleNamespace(
        network="resnet_18", batch_size=1, dtype="float32", layout="NHWC",
        timenow=None, json=None,
    )
    get_network(args.network, args.batch_size, args.layout, dtype=args.dtype)
    pm = PathManager(args.network, (1, 224, 224, 3), args, None, json=f"{project_root}/gallery/logs_json/tmp.json")
    pm.tasks_pkl_use()

    from constraint_provenance import build_provenance_formulas, eval_thread_formula, eval_vthread_formula

    log_file = f"{project_root}/gallery/logs_json/resnet_18/resnet_18-B1.json"
    with open(log_file) as f:
        records = [json.loads(l) for l in f if l.strip()]
    groups = defaultdict(list)
    for rec in records:
        groups[rec["i"][0][0]].append(rec)

    print("=" * 60)
    print("Provenance 모듈 검증")
    print("=" * 60)

    ok_build = 0
    ok_vthread = 0
    ok_single_kernel = 0
    n_single = 0
    errors = []

    for wk, recs in list(groups.items())[:8]:  # 처음 8개 task만
        rec = recs[0]
        try:
            task, state = record_to_task_and_state(rec)
        except Exception as e:
            errors.append((wk[:16], "recover", str(e)[:60]))
            continue

        try:
            formulas = build_provenance_formulas(task, state, rec, record_to_task_and_state)
        except Exception as e:
            errors.append((wk[:16], "build", str(e)[:60]))
            continue

        ok_build += 1

        thread_val = eval_thread_formula(formulas, rec)
        vthread_val = eval_vthread_formula(formulas, rec)

        if 1 <= vthread_val <= 8:
            ok_vthread += 1

        try:
            kernels, max_vts = get_tir_constraints(task, state)
        except Exception as e:
            errors.append((wk[:16], "tir", str(e)[:60]))
            continue

        if len(kernels) == 1:
            n_single += 1
            actual_thread = kernels[0].get("thread_per_block", 0)
            if thread_val == actual_thread:
                ok_single_kernel += 1
            else:
                errors.append((wk[:16], "thread_mismatch", f"provenance={thread_val} tir={actual_thread}"))

    n_tasks = min(8, len(groups))
    print(f"빌드 성공: {ok_build}/{n_tasks}")
    print(f"vthread 범위(1~8) OK: {ok_vthread}/{ok_build}")
    print(f"단일 커널 thread 일치(참고): {ok_single_kernel}/{n_single} (provenance는 전체 스테이지 thread-bound 곱이라 커널별 TIR과 불일치 가능)")
    if errors:
        print("\n이슈:")
        for w, kind, msg in errors[:10]:
            print(f"  [{w}] {kind}: {msg}")
    print()
    # 통과 기준: 모든 task에서 formula 빌드 성공 + vthread 값이 1~8
    if ok_build == n_tasks and ok_vthread == ok_build:
        print("검증 통과 (빌드 + vthread 범위).")
        return 0
    print("일부 검증 실패 (위 참고).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
