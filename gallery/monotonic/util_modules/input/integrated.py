import json
import re
import numpy as np

# from common import load_and_register_tasks
from util_modules.input.extent import state_to_records
from tvm import auto_scheduler
from tvm.auto_scheduler.measure import recover_measure_input
from tvm.auto_scheduler.feature import get_per_store_features_from_measure_pairs, get_per_store_features_from_states

def filter_InpRes_by_cost(inputs, results, cost_threshold=1e+10):
    filtered_inputs = []
    filtered_results = []
    for inp, res in zip(inputs, results):
        cost = np.mean([c.value for c in res.costs])
        if cost < cost_threshold:
            filtered_inputs.append(inp)
            filtered_results.append(res)

    # 정렬하기 전 인덱스를 저장. 추후에 다시 원래 순서로 복원할 때 사용 가능
    original_indices = list(range(len(filtered_inputs)))

    # order by cost ascending
    cost_result_pairs = list(zip(filtered_inputs, filtered_results, original_indices))
    cost_result_pairs.sort(key=lambda pair: np.mean([c.value for c in pair[1].costs]))
    filtered_inputs, filtered_results, original_indices = zip(*cost_result_pairs)

    return filtered_inputs, filtered_results, original_indices


def measureInput_to_Sch(inputs):
    def extract_numbers(x):
        _int_re = re.compile(r"-?\d+")
        nums = []
        if isinstance(x, int):
            nums.append(x)
        elif isinstance(x, str):
            # 문자열 안의 모든 정수 토큰을 추출
            nums.extend(int(m.group()) for m in _int_re.finditer(x))
        elif isinstance(x, list):
            for v in x:
                nums.extend(extract_numbers(v))
        return nums

    def differing_indices(arrays):
        # arrays: List[List[int]]
        if not arrays:
            return []

        length = len(arrays[0])
        for a in arrays:
            assert len(a) == length

        result = []
        for i in range(length):
            vals = {a[i] for a in arrays}
            if len(vals) > 1:
                result.append(i)
        return result
    
    
    # breakpoint()
    all_sch_params_li = []
    for inp in inputs:
        obj = json.loads(inp.serialize()[0])
        numbers = extract_numbers(obj[1][1])
        all_sch_params_li.append(numbers)
    # split, unroll 파라미터 추출
    diff_idxs = differing_indices(all_sch_params_li)
    # breakpoint()
    all_sch_params_array = np.array(all_sch_params_li)
    sch_params_arr = all_sch_params_array[:, diff_idxs]
    return sch_params_arr

def divide_by_sketches(inputs, results):
    tasks = [inp.task for inp in inputs]
    task_set = set(tasks)

    sketches = {task: [] for task in task_set}
    sketch_results = {task: [] for task in task_set}

    for inp, res in zip(inputs, results):
        task = inp.task
        sketches[task].append(inp)
        sketch_results[task].append(res)

    return sketches, sketch_results


def json_to_VecCosts(json, type="schedules", return_raw_cost=False):
    inputs, results = auto_scheduler.RecordReader(json).read_lines()
    inputs, results, original_indices = filter_InpRes_by_cost(inputs, results)

    costs = np.array([-np.log(np.mean([c.value for c in res.costs])) for res in results])
    if return_raw_cost:
        costs = np.array([np.mean([c.value for c in res.costs]) for res in results])

    if type == "schedules":
        output = measureInput_to_Sch(inputs)
    elif type == "features":
        output, _, _, _ = get_per_store_features_from_measure_pairs(inputs, results)
    elif type == "extents":
        # extent, unroll 모두 포함
        states = [recover_measure_input(inp, True).state for inp in inputs]
        for state in states:
            
            if "floor" in str(state):
                print("\"floor\" 감지됨")
                raise ValueError("State contains 'floor' directive, which is not supported for extent extraction.")
        output = state_to_records(states, ret_type="all")
        output = np.stack(output, axis=0)
        
    else:
        raise ValueError(f"Unknown type: {type}")

    # 원래 순서로 복원
    output = output[np.array(original_indices)]
    costs = costs[np.array(original_indices)]

    return output, costs


# if __name__ == "__main__":
#     json_file = "/root/work/tenset/dataset/measure_records_tenset/k80/([0bcc0b358b2b1d00bc591087e839592d,1,35,35,64,4,4,96,64,1,1,1,96,1,35,35,96],cuda).json"
#     load_and_register_tasks()
#     inputs, costs = json_to_VecCosts(json_file, type="schedules")