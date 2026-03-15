# 2603131951 Module Workflow Reference

- Agent: `integrator`
- Date: `2026-03-13`
- Status: `completed`
- Decision Topic: `current constrained_gen module and workflow reference`

## Inputs Considered

- Reviewer note:
  - none
- Specialist note:
  - none
- Relevant artifacts:
  - [MODULE_AND_WORKFLOW_REFERENCE.md](/root/work/tvm-ansor/gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md)

## Files Checked

- [gallery/constrained_gen/modules/__init__.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/__init__.py): package re-exports
- [gallery/constrained_gen/modules/common.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/common.py): dataset/task/path helpers
- [gallery/constrained_gen/modules/constraint_set.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/constraint_set.py): symbolic constraint building and checking
- [gallery/constrained_gen/modules/domain_propagator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/domain_propagator.py): domain narrowing and prefix analysis
- [gallery/constrained_gen/modules/exact_gpu_constraints.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/exact_gpu_constraints.py): exact/projected GPU node builders
- [gallery/constrained_gen/modules/expr_nodes.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/expr_nodes.py): symbolic expression nodes and parser
- [gallery/constrained_gen/modules/param_manager.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/param_manager.py): symbolic-state construction bridge
- [gallery/constrained_gen/modules/param_sampler.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/param_sampler.py): sampling implementation
- [gallery/constrained_gen/modules/projected_gpu_validation.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/projected_gpu_validation.py): diagnostic helpers
- [gallery/constrained_gen/modules/record_loader.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/record_loader.py): record/sketch identity helpers
- [gallery/constrained_gen/modules/schedule_generator.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/schedule_generator.py): central facade
- [gallery/constrained_gen/modules/structural_sketch.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/structural_sketch.py): canonical sketch helpers
- [gallery/constrained_gen/modules/sym_types.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/sym_types.py): primitive symbolic types
- [gallery/constrained_gen/modules/symbolic_state.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/symbolic_state.py): symbolic state object
- [gallery/constrained_gen/modules/transform_applier.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/transform_applier.py): transform replay engine
- [gallery/constrained_gen/modules/tvm_verify.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/tvm_verify.py): concrete TVM verification bridge
- [gallery/constrained_gen/modules/var_order_planner.py](/root/work/tvm-ansor/gallery/constrained_gen/modules/var_order_planner.py): variable ordering phases
- [gallery/constrained_gen/generate.py](/root/work/tvm-ansor/gallery/constrained_gen/generate.py): current surviving top-level script

## Decision

- Chosen direction:
  - Add one current-state reference document that explains module roles, classes/functions, helper groups, and end-to-end workflows in a single place.
- Rejected alternatives:
  - split the documentation across several smaller files before the current module boundaries are settled
  - describe deleted or historical validation entrypoints as if they were still active
- Why:
  - the user asked for one detailed document first, and the current tree has already changed enough that a code-backed single reference is more useful than a speculative reorganization note

## Impact

- Files likely to change:
  - [gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md](/root/work/tvm-ansor/gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md)
- Validation needed after change:
  - documentation-only change; no runtime validation required

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - use the new reference document to decide whether to keep a flat modules directory or start a small number of larger subgroups
