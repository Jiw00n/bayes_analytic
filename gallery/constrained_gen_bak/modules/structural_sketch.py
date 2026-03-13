"""Structural-sketch helpers for canonical symbolic-state construction."""

from .tvm_verify import params_to_state_from_state


def build_canonical_param_values(state, split_value=1, unroll_value=0):
    """Return a canonical concrete parameter assignment for a structural sketch."""
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
    """Build a deterministic representative State for the structural sketch."""
    params = build_canonical_param_values(
        state,
        split_value=split_value,
        unroll_value=unroll_value,
    )
    return params_to_state_from_state(task, state, params)
