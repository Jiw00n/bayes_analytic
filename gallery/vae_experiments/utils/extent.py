import os
import sys
import re
import time
import numpy as np
from tvm import auto_scheduler

def find_common_for_loops(schedules):
    """
    모든 스케줄에서 공통으로 나타나는 (0,1) for문 변수명을 찾음
    """
    common_vars = None
    
    for schedule in schedules:
        lines = schedule.split('\n')
        vars_in_schedule = set()
        
        for line in lines:
            stripped = line.lstrip()
            match = re.match(r'for\s+(\S+)\s+\(0,\s*1\)', stripped)
            if match:
                vars_in_schedule.add(match.group(1))
        
        if common_vars is None:
            common_vars = vars_in_schedule
        else:
            common_vars &= vars_in_schedule  # 교집합
    
    return common_vars if common_vars is not None else set()


def remove_common_for_loops(schedule, common_vars):
    """
    스케줄 코드에서 공통으로 나타나는 (0,1) for문을 제거하고 들여쓰기를 정리
    """
    lines = schedule.split('\n')
    result_lines = []
    
    # 제거할 for문의 인덱스들을 먼저 찾기
    remove_indices = set()
    for_loop_indents = {}  # 제거될 for문의 인덱스 -> 들여쓰기 레벨
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        indent_level = len(line) - len(stripped)
        
        # (0,1) for문인지 확인
        match = re.match(r'for\s+(\S+)\s+\(0,\s*1\)', stripped)
        if match and match.group(1) in common_vars:
            remove_indices.add(i)
            for_loop_indents[i] = indent_level
    
    # 각 줄에 대해 들여쓰기를 얼마나 줄여야 하는지 계산
    indent_reduction = [0] * len(lines)
    
    for idx in sorted(remove_indices):
        base_indent = for_loop_indents[idx]
        # 이 for문 다음부터 같거나 작은 들여쓰기가 나올 때까지 2칸씩 줄이기
        for j in range(idx + 1, len(lines)):
            if j in remove_indices:
                continue
            line = lines[j]
            stripped = line.lstrip()
            if not stripped:  # 빈 줄
                continue
            current_indent = len(line) - len(stripped)
            
            # 이 for문의 body인 경우 (들여쓰기가 더 큰 경우)
            if current_indent > base_indent:
                indent_reduction[j] += 2
            else:
                # 같거나 작은 들여쓰기 레벨이 나오면 이 for문 블록 종료
                break
    
    # 제거하지 않는 줄들에 대해 들여쓰기를 조정하여 결과 생성
    for i, line in enumerate(lines):
        if i in remove_indices:
            continue
        
        if not line.strip():  # 빈 줄
            result_lines.append(line)
            continue
        
        stripped = line.lstrip()
        original_indent = len(line) - len(stripped)
        new_indent = max(0, original_indent - indent_reduction[i])
        result_lines.append(' ' * new_indent + stripped)
    
    return '\n'.join(result_lines)




def state_to_records(state_strs):
    """
    state_strs: state 문자열 set
    """

    state_strs = list(map(lambda x: x.strip(), list(state_strs)))

    records = {"schedules": [], "extents": [], "unroll" : [], "all": []}

    for state in state_strs:
        schedule = state.split("Placeholder")[-1][2:]
        records["schedules"].append(schedule)

    common_for_loops = find_common_for_loops(records["schedules"])
    print(f"발견된 공통 (0,1) for문 변수: {common_for_loops}")


    # 모든 스케줄에 적용
    cleaned_schedules = []
    for i, schedule in enumerate(records["schedules"]):
        extents = [float(x) for x in re.findall(r'\(0,\s*(\d+)\)', schedule)]

    for i, schedule in enumerate(records["schedules"]):
        extents = [float(x) for x in re.findall(r'\(0,\s*(\d+)\)', schedule)]
        unrolls = [float(x) for x in re.findall(r'auto_unroll:\s*(\d+)', schedule)]
        records["extents"].append(extents)
        if unrolls == []:
            unrolls = [0.0]
        records["unroll"].append(unrolls)
        feature = extents+unrolls
        records["all"].append(np.array(feature, dtype=np.float32))
        
        cleaned = remove_common_for_loops(schedule, common_for_loops)
        cleaned_schedules.append(cleaned)
    records["cleaned_schedules"] = cleaned_schedules


    total_removed = sum(len(orig.split('\n')) - len(clean.split('\n')) 
                        for orig, clean in zip(records['schedules'], cleaned_schedules))
    avg_removed = total_removed / len(cleaned_schedules)
    print(f"제거된 줄 수: {avg_removed:.1f}")

    return records



def gen_program_pool(task, size, evo_population, min_population, seed=2023):
    policy = auto_scheduler.SketchPolicy(
        task, 
        auto_scheduler.XGBModel(),
        params={
            'evolutionary_search_num_iters': 4,  # 충분한 정제
            'evolutionary_search_population': evo_population,
            'sample_init_min_population': min_population,
        }, 
        seed=seed,
        verbose=0
    )

    # 단일 패스로 빠르게 생성
    print(f"Fast generating {size} states...")
    states = policy.sample_initial_population()

    # Evolutionary search로 한 번에 많이 생성
    states = policy.evolutionary_search(states, size * 2)

    # 중복 제거만 (검증은 Skip)
    state_strs = set()
    unique_states = []
    for s in states:
        str_s = str(s)
        if str_s not in state_strs:
            state_strs.add(str_s)
            unique_states.append(s)
            if len(unique_states) >= size:
                break

    print(f"Generated {len(unique_states)} unique states (no validation)")

    records = state_to_records(state_strs)

    return records