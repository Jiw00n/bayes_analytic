import json
import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import load_records

json_file = '/root/work/tvm-ansor/gallery/constrained_gen/data/measured_family_ansor/415_([e7c984cba151d5c7c1e081f0b1910087,[1,112,112,32],[3,3,32,1],[1,1,1,32],[1,112,112,32]],cuda).json'

vthread_vals = set()
threadx_vals = set()
records = load_records(json_file)

for inp, res in records:
    task = inp.task
    state = inp.state
    # Replay
    try:
        dag = task.compute_dag
        state = dag.infer_bound_from_state(state)
        for stage in state.stages:
            for it in stage.iters:
                if int(it.annotation) == 4:
                    ext = int(it.range.extent)
                    vthread_vals.add(ext)
                elif int(it.annotation) == 1:
                    threadx_vals.add(int(it.range.extent))
    except Exception as e:
        pass

print(f"Observed vthread extents: {vthread_vals}")
print(f"Observed threadIdx.x extents: {threadx_vals}")
