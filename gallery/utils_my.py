from tvm.auto_scheduler.measure import MeasureInput
from tvm.target import Target
import tvm.auto_scheduler as auto_scheduler
from tvm.auto_scheduler.search_task import SearchTask

def full_recover_measure_input(inp, rebuild_state=False):
    """
    Recover a deserialized MeasureInput by rebuilding the missing fields.
    1. Rebuid the compute_dag in inp.task
    2. (Optional) Rebuild the stages in inp.state

    Parameters
    ----------
    inp: MeasureInput
        The deserialized MeasureInput
    rebuild_state: bool = False
        Whether rebuild the stages in MeasureInput.State

    Returns
    -------
    new_input: MeasureInput
        The fully recovered MeasureInput with all fields rebuilt.
    """
    # pylint: disable=import-outside-toplevel

    task = inp.task
    task.target, task.target_host = Target.canon_target_and_host(task.target, task.target_host)
    new_task = SearchTask(
        workload_key=task.workload_key,
        target=task.target,
        hardware_params=task.hardware_params,
        layout_rewrite_option=task.layout_rewrite_option,
        task_inputs=list(task.task_input_names),
        # desc=desc,
    )
    # breakpoint()

    if rebuild_state:
        new_state = new_task.compute_dag.infer_bound_from_state(inp.state)
    else:
        new_state = inp.state

    return MeasureInput(new_task, new_state)
