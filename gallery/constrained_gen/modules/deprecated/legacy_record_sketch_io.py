"""Legacy record/sketch I/O and sketch identity helpers.

This module remains for measurement and diagnostic workflows that still load
AutoScheduler records or saved sketch dumps. New code should construct
``ScheduleGenerator`` instances directly via ``ScheduleGenerator.from_task_state``
after the task/state pair is already available.
"""
import hashlib
import os
from collections import defaultdict

from tvm import auto_scheduler
from tvm.auto_scheduler.measure_record import load_records
from tvm.auto_scheduler.measure_record import load_record_from_string

from ..task_paths import TO_MEASURE_PROGRAM_FOLDER, clean_name


STEP_RECORD_CODE = {
    "AnnotationStep": "AN",
    "FuseStep": "FU",
    "PragmaStep": "PR",
    "ReorderStep": "RE",
    "SplitStep": "SP",
    "FollowSplitStep": "FSP",
    "FollowFusedSplitStep": "FFSP",
    "StorageAlignStep": "SA",
    "ComputeAtStep": "CA",
    "ComputeInlineStep": "CI",
    "ComputeRootStep": "CR",
    "CacheReadStep": "CHR",
    "CacheWriteStep": "CHW",
    "RfactorStep": "RF",
}

# ------------------------------------------------------------------
# Deprecated
# ------------------------------------------------------------------

# ─────────────────────────────────────────────
# sketch fingerprint: 구조적 속성으로 sketch 식별
# ─────────────────────────────────────────────
def _step_structural_fingerprint(step):
    """step의 구조적 속성만 추출하여 hashable tuple 반환.
    SplitStep의 lengths 값, PragmaStep의 unroll 값은 제외 (이것들이 파라미터)."""
    tk = step.type_key.split(".")[-1]
    if tk == "AnnotationStep":
        return (tk, int(step.stage_id), int(step.iter_id), int(step.annotation))
    elif tk == "FuseStep":
        return (tk, int(step.stage_id), tuple(int(x) for x in step.fused_ids))
    elif tk == "PragmaStep":
        ptype = str(step.pragma_type).split("$")[0]
        return (tk, int(step.stage_id), int(step.iter_id), ptype)
    elif tk == "ReorderStep":
        return (tk, int(step.stage_id), tuple(int(x) for x in step.after_ids))
    elif tk == "SplitStep":
        return (tk, int(step.stage_id), int(step.iter_id),
                len(step.lengths), bool(step.inner_to_outer))
    elif tk == "FollowSplitStep":
        return (tk, int(step.stage_id), int(step.iter_id),
                int(step.src_step_id), int(step.n_split))
    elif tk == "FollowFusedSplitStep":
        return (tk, int(step.stage_id), int(step.iter_id),
                tuple(int(x) for x in step.src_step_ids),
                int(step.level), bool(step.factor_or_nparts))
    elif tk == "StorageAlignStep":
        return (tk, int(step.stage_id), int(step.iter_id),
                int(step.factor), int(step.offset))
    elif tk == "ComputeAtStep":
        return (tk, int(step.stage_id), int(step.target_stage_id), int(step.target_iter_id))
    elif tk == "ComputeInlineStep":
        return (tk, int(step.stage_id))
    elif tk == "ComputeRootStep":
        return (tk, int(step.stage_id))
    elif tk == "CacheReadStep":
        return (tk, int(step.stage_id), str(step.scope_name),
                tuple(int(x) for x in step.reader_stage_ids))
    elif tk == "CacheWriteStep":
        return (tk, int(step.stage_id), str(step.scope_name))
    else:
        return (tk, int(step.stage_id))


def step_record_code(step):
    """TVM Step 객체를 measure-record의 short code로 변환."""
    tk = step.type_key.split(".")[-1]
    return STEP_RECORD_CODE.get(tk, tk)


def state_step_codes(state):
    """state.transform_steps를 raw measure record와 동일한 short code 시퀀스로 변환."""
    return tuple(step_record_code(step) for step in state.transform_steps)


def state_step_signature(state):
    """state.transform_steps short code 시퀀스를 문자열로 반환."""
    return "-".join(state_step_codes(state))


def state_sketch_fingerprint(state):
    """state의 전체 step 시퀀스에서 구조적 fingerprint를 추출.
    split factor 값과 pragma 값을 제외한 모든 구조 속성이 같아야 같은 sketch."""
    return tuple(_step_structural_fingerprint(s) for s in state.transform_steps)


def sketch_fingerprint_repr(fp):
    """tuple 기반 fingerprint를 재현 가능한 문자열로 직렬화."""
    return repr(fp)


def sketch_fingerprint_hash(fp):
    """긴 fingerprint를 compact하게 식별할 수 있는 안정 해시."""
    return hashlib.sha1(sketch_fingerprint_repr(fp).encode("utf-8")).hexdigest()[:16]


def raw_record_steps(record):
    """raw measure-record JSON dict에서 transform step 배열을 반환."""
    return record["i"][1][1]


def raw_record_step_codes(record):
    """raw measure-record JSON dict의 short code 시퀀스를 반환."""
    return tuple(step[0] for step in raw_record_steps(record))


def raw_record_step_signature(record):
    """raw measure-record JSON dict의 short code 시퀀스를 문자열로 반환."""
    return "-".join(raw_record_step_codes(record))


# ─────────────────────────────────────────────
# 레코드 로드
# ─────────────────────────────────────────────


def get_task_json_name(records_dir, task):
    """task에 대응하는 측정 레코드 JSON 파일 경로를 반환한다."""
    task_key = (task.workload_key, str(task.target.kind))
    return f"{records_dir}/{clean_name(task_key)}.json"


def load_records_from_dir(tasks, records_dir):
    """각 task에 대응하는 JSON 파일에서 레코드를 로드해 workload_key별로 반환한다."""
    records = {}
    for task in tasks:
        json_path = get_task_json_name(records_dir, task)
        if os.path.exists(json_path):
            recs = load_records(json_path)
            records[task.workload_key] = [(mi, mr) for mi, mr in recs]
    return records


# ─────────────────────────────────────────────
# wkey → sketch 2단계 그룹핑
# ─────────────────────────────────────────────
def group_records_by_wkey_and_sketch(records):
    """레코드를 workload_key → sketch(step type 시퀀스) 기준으로 2단계 그룹핑.

    Returns:
        grouped: dict {wkey: {sketch_fp: [(MeasureInput, MeasureResult), ...]}}
    """
    grouped = {}
    for wkey, recs in records.items():
        sketch_groups = defaultdict(list)
        for inp, res in recs:
            fp = state_sketch_fingerprint(inp.state)
            sketch_groups[fp].append((inp, res))
        grouped[wkey] = dict(sketch_groups)
    return grouped


def group_by_sketches_from_json(tasks, records_dir, verbose=False):
    """json 파일에서 레코드를 로드하여 sketch 기준으로 그룹핑한 결과 반환."""
    records = load_records_from_dir(tasks, records_dir)
    grouped = group_records_by_wkey_and_sketch(records)

    if verbose:
        wkey_to_task = {t.workload_key: t for t in tasks}
        for wkey, sketch_dict in grouped.items():
            task = wkey_to_task.get(wkey)
            desc = task.desc if task else wkey
            total_recs = sum(len(v) for v in sketch_dict.values())
            print(f"\n[{desc}] total records: {total_recs}, sketches: {len(sketch_dict)}")

    return grouped


def load_sketch_lines(sketches_path=None):
    """저장된 all_sketches.json에서 한 줄씩 읽어 문자열 목록으로 반환한다."""
    if sketches_path is None:
        sketches_path = f"{TO_MEASURE_PROGRAM_FOLDER}/all_sketches.json"
    with open(sketches_path) as f:
        return [line.strip() for line in f]


def load_sketch_record(line, tasks_by_wkey):
    """한 줄 레코드 문자열을 디코딩해 (task, base_inp, base_res, state)를 반환한다."""
    base_inp, base_res = load_record_from_string(line)
    recovered = auto_scheduler.measure.recover_measure_input(base_inp)
    task = tasks_by_wkey[recovered.task.workload_key]
    return task, base_inp, base_res, recovered.state
