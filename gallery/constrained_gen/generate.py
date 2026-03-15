import numpy as np
import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler import SketchPolicy

TARGET = tvm.target.Target("cuda")

from modules.task_paths import load_and_register_tasks
from modules.schedule_generator import ScheduleGenerator



tasks = load_and_register_tasks()
concrete_states = {}
sketches_by_idx = {}
for idx, task in enumerate(tasks):
    concrete_state = SketchPolicy(task, params={'sample_init_no_invalid': 1 }, verbose=False).generate_concrete_sketches()
    for i, state in enumerate(concrete_state):
        sketches_by_idx[idx] = (task.desc, f"{task.workload_key}_{i}")
        concrete_states[f"{task.workload_key}_{i}"] = (task, state)


sketch_idx = 178
print(sketches_by_idx[sketch_idx])
sample = list(concrete_states.values())[sketch_idx]
sg = ScheduleGenerator.from_task_state(sample[0], sample[1])
