#!/usr/bin/env python3
"""Debug FFSP failures in T12, T14, T20"""
import os, sys
project_root = "/root/work/tvm-ansor"
os.environ["TVM_HOME"] = project_root
os.environ["TVM_LIBRARY_PATH"] = f"{project_root}/build-release"
if f"{project_root}/python" not in sys.path:
    sys.path.insert(0, f"{project_root}/python")
sys.path = [p for p in sys.path if not p.startswith(f"{project_root}/build")]
sys.path.append(f"{project_root}/build-release")

import numpy as np
from util_manager import PathManager, get_network
import tvm
from tvm import auto_scheduler
from types import SimpleNamespace

TARGET = tvm.target.Target("cuda")
args = SimpleNamespace(network="resnet_18", batch_size=1, dtype="float32", layout="NHWC", timenow=None, json=None)
mod, params, input_shape, output_shape = get_network(args.network, args.batch_size, args.layout, dtype=args.dtype)
path_manager = PathManager(args.network, input_shape, args, None, json="/root/work/tvm-ansor/gallery/logs_json/tmp.json")

def get_tasks(mod, params, path_manager, verbose=False, get_pkl=True):
    if get_pkl:
        tasks, task_weights = path_manager.tasks_pkl_use()
    if get_pkl is False or tasks is None:
        tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, TARGET)
    return tasks, task_weights

tasks, task_weights = get_tasks(None, params, path_manager, verbose=False, get_pkl=True)
tasks, task_weights = zip(*sorted(zip(tasks, task_weights), key=lambda x: x[0].desc))

# Import SymbolicState from notebook (exec the class definition)
# Quick approach: just test T20 
task_idx = 20
task = tasks[task_idx]
dag = task.compute_dag
print(f"T{task_idx}: {task.desc}")
print(dag)

policy = auto_scheduler.SketchPolicy(task, auto_scheduler.XGBModel())
states_init = policy.sample_initial_population()
evo_states = policy.evolutionary_search(states_init, 1000)

print(f"\n{len(evo_states)} states from evolutionary search")

# Find states where FFSP inner doesn't match
replay_func = tvm._ffi.get_global_func("auto_scheduler.ReplayStepsPartial")

for si, st in enumerate(evo_states):
    steps = st.transform_steps
    bounded = dag.infer_bound_from_state(st)
    
    # Look for FFSP steps
    for i, step in enumerate(steps):
        tk = step.type_key.split(".")[-1]
        if tk != "FollowFusedSplitStep":
            continue
        
        sid = step.stage_id
        iid = step.iter_id
        src_step_ids = [int(x) for x in step.src_step_ids]
        level = int(step.level)
        factor_or_nparts = bool(step.factor_or_nparts)
        
        # Calculate what our code would produce
        # Get the fused factor: product of sp_{src_id}_{level} values
        fused_factor = 1
        for src_id in src_step_ids:
            src_step = steps[src_id]
            src_lens = list(src_step.lengths)
            if level < len(src_lens) and src_lens[level] is not None:
                fused_factor *= int(src_lens[level])
        
        # Get the extent at step time
        ps = replay_func(dag, st, i)
        ps_bounded = dag.infer_bound_from_state(ps)
        if sid < len(ps_bounded.stages):
            ps_stage = ps_bounded.stages[sid]
            if iid < len(ps_stage.iters) and ps_stage.iters[iid].range is not None:
                tosplit_ext = int(ps_stage.iters[iid].range.extent)
            else:
                tosplit_ext = None
        else:
            tosplit_ext = None
        
        if tosplit_ext is None:
            continue
        
        # Our calculation
        import math
        if factor_or_nparts:
            our_inner = fused_factor
            our_outer = math.ceil(tosplit_ext / fused_factor)
        else:
            our_outer = fused_factor
            our_inner = math.ceil(tosplit_ext / fused_factor)
        
        # Get after-step state to see actual extents  
        ps_after = replay_func(dag, st, i + 1)
        ps_after_bounded = dag.infer_bound_from_state(ps_after)
        if sid < len(ps_after_bounded.stages):
            after_stage = ps_after_bounded.stages[sid]
            # After FFSP, iid is replaced by two iters: iid (outer) and iid+1 (inner)
            if iid + 1 < len(after_stage.iters):
                real_outer = int(after_stage.iters[iid].range.extent) if after_stage.iters[iid].range else None
                real_inner = int(after_stage.iters[iid+1].range.extent) if after_stage.iters[iid+1].range else None
            else:
                real_outer = None
                real_inner = None
        else:
            real_outer = None
            real_inner = None
        
        if real_outer != our_outer or real_inner != our_inner:
            print(f"\n=== S{si} Step[{i}] FFSP MISMATCH ===")
            print(f"  sid={sid} iid={iid} src_ids={src_step_ids} level={level} f_or_n={factor_or_nparts}")
            print(f"  fused_factor={fused_factor}")
            print(f"  tosplit_ext={tosplit_ext}")
            print(f"  OUR:  outer={our_outer}, inner={our_inner}")
            print(f"  REAL: outer={real_outer}, inner={real_inner}")
            
            # Also print the final bounded state for this stage
            final_bounded = dag.infer_bound_from_state(st)
            print(f"  Final bounded stage {sid}:")
            for ii, it in enumerate(final_bounded.stages[sid].iters):
                ext = int(it.range.extent) if it.range else None
                print(f"    i{ii} '{it.name}': {ext}")
            
            # Print all steps for context
            print(f"\n  All steps ({len(steps)}):")
            for j, s in enumerate(steps):
                stk = s.type_key.split(".")[-1]
                if stk == "SplitStep":
                    lens = [int(l) if l is not None else None for l in s.lengths]
                    print(f"    [{j}] {stk} sid={s.stage_id} iid={s.iter_id} lens={lens}")
                elif stk == "FuseStep":
                    fids = [int(x) for x in s.fused_ids]
                    print(f"    [{j}] {stk} sid={s.stage_id} fids={fids}")
                elif stk == "FollowFusedSplitStep":
                    srcs = [int(x) for x in s.src_step_ids]
                    print(f"    [{j}] {stk} sid={s.stage_id} iid={s.iter_id} srcs={srcs} lvl={s.level} fon={s.factor_or_nparts}")
                elif stk == "ComputeAtStep":
                    print(f"    [{j}] {stk} sid={s.stage_id} target=s{s.target_stage_id}.i{s.target_iter_id}")
                elif stk == "CacheWriteStep":
                    print(f"    [{j}] {stk} sid={s.stage_id}")
                elif stk == "CacheReadStep":
                    rids = [int(x) for x in s.reader_stage_ids]
                    print(f"    [{j}] {stk} sid={s.stage_id} rids={rids}")
                else:
                    print(f"    [{j}] {stk} sid={s.stage_id}")
            break  # 1 mismatch per state is enough
    
    # Only report first 3 mismatches
    if sum(1 for _ in []) >= 3:  # placeholder
        break

print("\nDone.")
