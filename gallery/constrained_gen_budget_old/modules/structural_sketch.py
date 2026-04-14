"""Structural-sketch helpers for canonical symbolic-state construction."""

from .concrete_gpu_verify import params_to_state_from_state


def build_canonical_param_values(state, split_value=1, unroll_value=0):
    """구조 스케치용 정규화된 구체 파라미터 할당(sp_*, ur_*)을 반환한다."""
    params = {}
    for step_idx, step in enumerate(state.transform_steps):
        tk = step.type_key.split(".")[-1]
        if tk == "SplitStep":
            for length_idx in range(len(step.lengths)):
                params[f"sp_{step_idx}_{length_idx}"] = int(split_value)
        elif tk == "PragmaStep":
            pragma_type = str(step.pragma_type)
            if pragma_type.startswith("auto_unroll_max_step"):
                params[f"ur_{step_idx}"] = int(unroll_value)
    return params


def build_canonical_state(task, state, split_value=1, unroll_value=0):
    """구조 스케치의 대표 State를 정규 파라미터로 만들어 반환한다."""
    params = build_canonical_param_values(
        state,
        split_value=split_value,
        unroll_value=unroll_value,
    )
    return params_to_state_from_state(task, state, params)
