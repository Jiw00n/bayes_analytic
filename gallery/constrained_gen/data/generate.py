import numpy as np
import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler import SketchPolicy

TARGET = tvm.target.Target("cuda")

from ..modules.task_paths import load_and_register_tasks
from ..modules.schedule_generator import ScheduleGenerator
from ..modules.symbolic_state_bridge import build_symbolic_state



tasks = load_and_register_tasks()

task_idx = 584
task = tasks[task_idx]
print(task.desc)
sample = SketchPolicy(task, params={'sample_init_no_invalid': 1 }, verbose=False).generate_concrete_sketches()[0]
sym_state = build_symbolic_state(task, sample)
sg = ScheduleGenerator(sym_state, task=task, base_state=sample)
breakpoint()