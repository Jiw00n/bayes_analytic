import os
import tvm
from tvm import auto_scheduler
import time
import random



def make_states(filename, task, size, max_retry_iter=10):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    policy = auto_scheduler.SketchPolicy(task, auto_scheduler.XGBModel(),
            params={'evolutionary_search_num_iters': 1,
                    'evolutionary_search_population': min(size, 2560)}, verbose=0)

    states = policy.sample_initial_population()

    # Generate unique states
    all_state_str_set = set()
    all_state_list = []

    retry_ct = 0
    niter = 0

    while len(all_state_list) < size and retry_ct < max_retry_iter:
        ct_before = len(all_state_list)

        states = policy.evolutionary_search(states, len(states))
        for s in states:
            str_s = str(s)
            if str_s not in all_state_str_set:
                all_state_str_set.add(str_s)
                all_state_list.append(s)

            if len(all_state_list) >= size:
                break

        ct_after = len(all_state_list)

        if ct_before == ct_after:
            states = policy.sample_initial_population()
            
            retry_ct += 1
        else:
            retry_ct = 0

        print(niter, len(all_state_list))
        niter += 1
    all_state_list = all_state_list[:size]

    # Make measure inputs and results
    measure_inputs = []
    measure_results = []
    for state in all_state_list:
        measure_inputs.append(auto_scheduler.MeasureInput(task, state))
        measure_results.append(auto_scheduler.MeasureResult([0.0], 0, "", 0, time.time()))

    
    # # Dump to file
    # auto_scheduler.save_records(filename, measure_inputs, measure_results)
    return measure_inputs, measure_results


def select_features(model, features, used_indices, sample_size=64):
    """사용하지 않은 feature 중에서 sample_size개 선택"""
    available = [i for i in range(len(features)) if i not in used_indices]
    return random.sample(available, min(sample_size, len(available)))