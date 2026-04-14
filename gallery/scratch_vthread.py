import json
from tvm.auto_scheduler.measure_record import load_record_from_string
from tvm import auto_scheduler

json_file = '/root/work/tvm-ansor/gallery/constrained_gen/data/measured_family_ansor/415_([e7c984cba151d5c7c1e081f0b1910087,[1,112,112,32],[3,3,32,1],[1,1,1,32],[1,112,112,32]],cuda).json'

vthread_exceed = []
with open(json_file, 'r') as f:
    for line in f:
        try:
            inp, res = load_record_from_string(line)
        except Exception:
            continue
        state = inp.state
        vthread_extents = {}
        for step in state.transform_steps:
             if hasattr(step, 'annotation') and step.annotation == 4: # vthread
                 pass # Wait, state.transform_steps has AnnotationStep
                 # Actually, you can't easily get the *extent* from the AnnotationStep without replaying it or 
                 # parsing the state's `stages`.
        
        # Another way: use the sym_map of our GeneratorRegistry!
        pass
