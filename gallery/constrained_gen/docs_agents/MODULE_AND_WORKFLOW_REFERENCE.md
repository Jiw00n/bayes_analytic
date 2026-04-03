# Constrained Gen Module And Workflow Reference

This document summarizes the current `gallery/constrained_gen` code that still exists in the repository.

It is intentionally code-backed rather than aspirational. The descriptions below are based on the current Python modules under `gallery/constrained_gen/modules/` plus the current top-level entrypoints.

The organizing question for this document is:

- what each module currently owns
- what classes/functions it defines
- what internal helper groups matter
- how data moves across modules
- what the current end-to-end workflows are

## Current Module Inventory

Current modules in `gallery/constrained_gen/modules/`:

- `__init__.py`
- `task_paths.py`
- `constraint_set.py`
- `domain_propagator.py`
- `gpu_projection_constraints.py`
- `gpu_case_constraints.py`
- `expr_nodes.py`
- `param_sampler.py`
- `gpu_projection_diagnostics.py`
- `legacy_record_sketch_io.py`
- `schedule_generator.py`
- `structural_sketch.py`
- `symbolic_state_bridge.py`
- `sym_types.py`
- `symbolic_state.py`
- `transform_applier.py`
- `concrete_gpu_verify.py`
- `var_order_planner.py`

Current top-level entrypoints:

- `validate.py`
- `generate_programs.py`
- `measure_programs.py`

Current exploratory script:

- `generate.py`

## High-Level Architecture

The current codebase is easier to read if it is viewed as six layers.

### 1. Symbolic data model

These modules define the internal symbolic representation:

- `sym_types.py`
- `symbolic_state.py`
- `expr_nodes.py`

They do not run sampling or validation by themselves. They provide the data structures that other modules consume.

### 2. Concrete-state to symbolic-state reconstruction

These modules build a symbolic schedule state from a TVM task/state:

- `structural_sketch.py`
- `transform_applier.py`
- `symbolic_state_bridge.py`

This is the bridge from a concrete TVM `auto_scheduler.State` into the symbolic world.

### 3. Constraint construction and checking

These modules derive and evaluate symbolic constraints:

- `constraint_set.py`
- `gpu_projection_constraints.py`
- `gpu_case_constraints.py`
- `concrete_gpu_verify.py`

`constraint_set.py` is the central symbolic checker. `gpu_projection_constraints.py` owns the projected GPU-bound pruning helpers used by the active generation path. `gpu_case_constraints.py` owns exact case-table materialization and post-vectorize exact lowering. `concrete_gpu_verify.py` is the concrete TVM/GPU verification bridge.

### 4. Ordering, propagation, and sampling

These modules make partial and full parameter assignment feasible:

- `var_order_planner.py`
- `domain_propagator.py`
- `param_sampler.py`

They decide assignment order, tighten domains, filter candidates, and produce sampled assignments.

### 5. Facade/orchestration

The main orchestrator is:

- `schedule_generator.py`

This file owns the facade that wires the symbolic state, constraints, planner, propagator, sampler, and optional concrete final validation context together.

### 6. Data/record/diagnostic utilities

These modules support dataset path handling, record loading, and diagnostic triage:

- `task_paths.py`
- `legacy_record_sketch_io.py`
- `gpu_projection_diagnostics.py`

These are not the semantic core, but they are essential to reconstructing inputs and analyzing mismatches.

## End-To-End Workflows

## Workflow A: Build A Symbolic Generator From A Concrete TVM State

This is the main setup flow.

1. Load or obtain a TVM `task` and a concrete `auto_scheduler.State`.
2. Canonicalize the concrete state into a structural representative.
3. Replay the structural steps into a `SymbolicState`.
4. Wrap that symbolic state in `ScheduleGenerator`.
5. Let `ConstraintSet.preprocess()` compute all derived metadata and indexes.

Concrete module path:

- `structural_sketch.build_canonical_state(task, state)`
- `symbolic_state_bridge.build_symbolic_state(task, state)`
- `TransformApplier.apply_steps(structural_state)`
- `ScheduleGenerator.__init__(sym_state, ...)`
- `ConstraintSet.preprocess()`

Important consequence:

- the symbolic pipeline is structural-first
- original split/unroll values are not preserved as the source of truth
- instead, the symbolic state is rebuilt from a canonical sketch representative

## Workflow B: Build Constraints And Precompute Sampling Metadata

This happens during `ScheduleGenerator` construction.

`ConstraintSet.preprocess()` does most of the setup work:

- derives split-parameter groups from `sp_*` names
- derives original split extents from the replayed TVM state
- identifies innermost split variables
- identifies unroll variables
- builds symbolic constraints across enabled constraint families
- builds per-variable reverse indexes into those constraints
- computes preferred thread variables
- computes variable-order phase entries and final variable order

Key outputs written into the generator object:

- `_constraints`
- `_var_constraints`
- `_sp_groups`
- `_sp_extents`
- `_all_sp_names`
- `_ur_names`
- `_innermost_names`
- `_preferred_thread_vars`
- `_var_order_phase_entries`
- `_var_order`

This preprocessing step is the real setup hub for the entire sampler/checker pipeline.

## Workflow C: Random Sampling With Incremental Domain Propagation

The full random sampling flow currently lives in `ParamSampler`.

1. Initialize domains for all split parameters from original step extents.
2. Greedily fix already-singleton variables.
3. Visit remaining variables in the planner-computed order.
4. Compute legal divisors for the step-local remaining extent.
5. Clip by domain bounds.
6. Filter candidates through constraint-driven propagation.
7. Randomly choose one candidate.
8. Propagate the assignment to tighten other domains.
9. Optionally assign unroll variables.
10. Run final symbolic or hybrid validation.

Main functions involved:

- `ParamSampler._assign_initial_fixed_vars(...)`
- `ParamSampler._randomize_params_with_order(...)`
- `ParamSampler._sample_split_var(...)`
- `ParamSampler._validate_sample(...)`
- `DomainPropagator.filter_by_constraints(...)`
- `DomainPropagator.propagate_domain(...)`
- `ScheduleGenerator.check_all_hybrid(...)`
- `ScheduleGenerator.check_all_exact(...)`

Current acceptance semantics worth remembering:

- full `randomize_params()` uses hybrid validation when unroll is assigned
- in the active `from_task_state(...)` path, `check_all_hybrid()` uses concrete-final validation first
- exact is only the fallback when concrete-final context is unavailable
- prefix sampling intentionally skips full validation
- `_enumerate_all_params()` remains an internal exhaustive helper; it exact-checks split assignments before appending unroll combinations

## Workflow D: Prefix Analysis And Constraint Explainability

The current code has a useful debug flow for partial assignments.

`ScheduleGenerator.randomize_params_prefix(stop_after_phase, ...)`:

- resolves a phase prefix from the planner
- samples only that prefix
- returns a structured report with:
  - `query`
  - `phase_selection`
  - `param_order`
  - `phases`
  - `assignment`
  - `domains`
  - `constraints`

This is the main explainability path for understanding how much a variable-order prefix already determines.

## Workflow E: Symbolic Checking Versus Concrete Final Validation

There are three important symbolic/concrete checking levels.

### Pruning

`ScheduleGenerator.check_all_pruning(...)`

- fast symbolic pruning
- based on projected/syntactic symbolic constraints

### Exact

`ScheduleGenerator.check_all_exact(...)`

- stronger symbolic checking
- may use exact GPU case tables from `gpu_case_constraints.py`

### Hybrid / Final

`ScheduleGenerator.check_all_hybrid(...)`
- if concrete final context exists, uses concrete lowering/GPU verification plus structural checks
- otherwise falls back to exact symbolic checks

`ScheduleGenerator.check_all_final(...)`
- always tries concrete final validation when possible

Important current behavior:

- `check_all_hybrid()` does not always call `check_all_exact()` when concrete final context exists
- the generator's concrete-final helper is the true bridge to lowered GPU verification

## Workflow F: Exact And Projected GPU Constraint Construction

`gpu_projection_constraints.py` and `gpu_case_constraints.py` together own the projected/exact GPU-bound construction split.

The workflow is:

1. Lower symbolic pre-vectorize TIR using registered TVM globals.
2. Collect runtime extent expressions and vector-loop metadata.
3. Build runtime domains.
4. Build projected vectorize/shared-memory pruning nodes for the active generation path.
5. Optionally enumerate feasible selector tuples for vectorized loop extents.
6. Optionally extract per-case GPU stats with TVM global functions.
7. Build exact `CaseSplitNode` trees when validation/diagnostics explicitly request them.
8. Project those exact trees back into pruning-friendly upper-bound nodes only for exact-aware diagnostics or compatibility paths.

This exact/projected split is one of the most important ideas in the repository:

- exact nodes are more faithful but heavier
- projected nodes are cheaper and friendlier to pruning
- active generation uses projected pruning plus concrete final validation
- diagnostics compare exact/projected/concrete outcomes to classify mismatches

## Workflow G: Concrete TVM Reconstruction And GPU Verification

`concrete_gpu_verify.py` owns the concrete side.

The param-to-state path is:

1. Serialize a measure record to JSON.
2. Patch `SP` and `PR` step payloads with the requested `sp_*` / `ur_*` params.
3. Reload the patched record back into a new TVM `State`.
4. Apply TVM compute-dag steps.
5. Lower to an `IRModule`.
6. Run the GPU verification pass pipeline.
7. Run `tir.analysis.verify_gpu_code` or `verify_gpu_code_errors`.

This reconstruction path is used both by:

- the generator's concrete-final helper
- structural-sketch canonicalization helpers

## Workflow H: Diagnostic And Triage Helpers

`gpu_projection_diagnostics.py` is the diagnostic layer.

Its job is not to define core semantics. Its job is to help answer questions like:

- why did projected pruning disagree with concrete validation?
- was this a projection upper-bound issue?
- was this an exact symbolic case-stat issue?
- was this a post-vectorize lowering mismatch?
- did concrete verification fail even though symbolic checks looked safe?

Its main outputs are diagnostic dictionaries with:

- exact violations
- projected violations
- per-kind snapshots
- likely root causes
- concrete final result
- exact-lowering differential data

## Current Surviving Top-Level Script

`gallery/constrained_gen/generate.py` is currently an ad hoc exploratory script.

It:

- loads all tasks with `load_and_register_tasks()`
- generates concrete sketches with `SketchPolicy.generate_concrete_sketches()`
- stores them in in-memory maps
- hardcodes `sketch_idx = 178`
- builds one symbolic state
- constructs one `ScheduleGenerator`

It currently does not define functions or a reusable API surface.

## Module-By-Module Reference

## `modules/__init__.py`

Role:

- minimal package surface for the constrained-gen module package

Exports:

- from `schedule_generator.py`
  - `ScheduleGenerator`

Functionality:

- explicit minimal package API only
- internal helpers are expected to be imported from their defining modules

## `modules/task_paths.py`

Role:

- shared dataset-path and TVM task-loading helpers, plus a small legacy utility tail

Top-level objects:

- `BenchmarkRecord`: namedtuple for benchmark logging
- path constants:
  - `NETWORK_INFO_FOLDER`
  - `TO_MEASURE_PROGRAM_FOLDER`
  - `TO_MEASURE_GEN_PROGRAM_FOLDER`
  - `TO_MEASURE_NETWORK_FOLDER`
  - `MEASURED_FOLDER`

Functions:

- `convert_to_nhwc(mod)`
  - Relay layout conversion helper for conv2d/conv3d
- `log_line(record, out_file)`
  - append one benchmark record line to a TSV-like file
- `clean_name(x)`
  - string sanitizer for filenames
- `get_relay_ir_filename(network_key)`
  - path helper for relay IR pickle
- `get_task_info_filename(network_key, target)`
  - path helper for task-info pickle
- `get_to_measure_filename(task, network_name=None)`
  - path helper for per-task to-measure JSON
- `get_to_measure_gen_filename(task, output_dir=...)`
  - path helper for generated program JSON
- `get_measure_record_filename(task, target=None)`
  - path helper for measured program JSON
- `load_and_register_tasks(network_info_folder=...)`
  - load `all_tasks.pkl` and register workloads with TVM
- `load_and_register_network(network_task_path=...)`
  - load a `(tasks, task_weights)` bundle and register workloads
- `_register_task_workloads(tasks)`
  - internal deduplicated workload-registration helper
- `dtype2torch(x)`
  - currently only maps `float32`
- `str2bool(v)`
  - parse boolean-like strings

Workflow position:

- early data-loading and path support
- the active ownership is path helpers plus task registration
- the remaining Relay/torch/string utilities are legacy shared helpers, not generator-core logic

## `modules/sym_types.py`

Role:

- primitive symbolic types and small symbolic-expression construction helpers

Top-level constants:

- `ANNOTATION_STR`
  - iterator annotation code to human-readable string
- `CA_ROOT`
- `CA_INLINED`
- `CA_ITER`

Classes:

- `SymExpr`
  - wrapper around either a concrete int or a symbolic expression string
  - properties:
    - `is_concrete`
  - methods:
    - `__repr__`, `__str__`, `__int__`
    - `ceildiv(a, b)`
    - `_needs_parens_for_mul(s)`
    - `mul(a, b)`
    - `product(items)`
    - `min(a, b)`
    - `max(items)`
  - purpose:
    - build symbolic loop extents and split products without immediately lowering to TIR

- `SymIter`
  - symbolic iterator record
  - fields:
    - `name`
    - `extent`
    - `annotation`
    - `iter_kind`
  - methods:
    - `clone()`
    - `__repr__()`

- `SymStage`
  - symbolic stage record
  - fields:
    - `op_name`
    - `op_type`
    - `iters`
    - `compute_at`
    - `auto_unroll_max_step`
    - `storage_offset`
    - `attach_stage_id`
    - `attach_iter_id`
    - `dtype`
  - methods:
    - `clone()`
    - `dtype_bytes`
  - purpose:
    - symbolic stage tree manipulated by `TransformApplier`

Functions:

- `eval_sym_extent(expr, sym_map)`
  - evaluate a `SymExpr` string by substituting values from `sym_map`
  - supports `ceil(...)` through `math.ceil`

Workflow position:

- foundational representation layer
- consumed heavily by `symbolic_state.py`, `transform_applier.py`, and `schedule_generator.py`

## `modules/symbolic_state.py`

Role:

- pure Python symbolic mirror of an `auto_scheduler.State`

Class:

- `SymbolicState`
  - fields initialized in `__init__`:
    - `stages`
    - `sym_map`
    - `compute_dag`
    - `_state`
    - `_ca_saved_extents`
    - `_split_sym_products`
    - `_cache_read_consumer`
    - `_cache_read_stencil_info`
    - `_shared_fused_extents`
  - static helpers:
    - `_safe_int_extent(extent_expr)`
    - `_clone_symexpr(expr)`
  - public helpers:
    - `canonicalize_param_values()`
    - `clone()`
    - `to_str(delete_trivial_loop=True)`
    - `get_vectorize_extents()`
    - `get_thread_extents()`
    - `get_vthread_extents()`
    - `get_shared_memory_extents()`
  - internal stage/tree maintenance:
    - `_shift_ca_saved_extents(...)`
    - `_print_stage(...)`
    - `_collect_extents_by_annotation(...)`

Functionality:

- clones stage/iter structure from `compute_dag.ops`
- tracks symbolic parameters in `sym_map`
- stores metadata needed to recover extents through compute-at, cache-read, and split transforms
- provides display helpers for debugging the symbolic stage tree

Workflow position:

- central mutable symbolic state that is replayed by `TransformApplier` and later consumed by `ScheduleGenerator`

## `modules/transform_applier.py`

Role:

- replay TVM transform steps into `SymbolicState`

Class:

- `TransformApplier`
  - constructor:
    - `__init__(sym_state)`
  - public entry:
    - `apply_steps(state)`
  - general helper:
    - `_clamp_positive_split_extent(sym_ext, tosplit_extent)`
  - stage/extent recovery helpers:
    - `_clone_real_stage(real_stage)`
    - `_product_of_defined_iters(stage)`
    - `_get_cache_read_restore_ctx(stage_id)`
    - `_get_safe_saved_extent(stage_id, iter_id, real_ext)`
    - `_recover_iter_extent(...)`
    - `_restore_stage_extents_if_needed(stage_id, step_idx)`
    - `_get_consumer_split_sym_products(cache_read_stage_id)`
    - `_match_cr_extent(real_ext, candidates)`
    - `_iter_base_name(name)`
    - `_match_compute_at_inner_extent(sid, iid, real_ext)`
    - `_infer_bound_final(state)`
  - step handlers:
    - `_apply_annotation(step)`
    - `_apply_fuse(step, step_idx)`
    - `_apply_pragma(step, step_idx)`
    - `_apply_reorder(step)`
    - `_apply_split(step, step_idx)`
    - `_apply_follow_split(step, all_steps, step_idx)`
    - `_apply_follow_fused_split(step, all_steps, step_idx)`
    - `_apply_storage_align(step)`
    - `_apply_compute_at(step)`
    - `_apply_compute_inline(step)`
    - `_apply_compute_root(step)`
    - `_apply_cache_read(step, state, step_idx)`
    - `_analyze_cache_read_stencil(cr_stage_id, orig_tensor_sid, consumer_sid)`
    - `_find_producer_loads(expr, tensor_name)`
    - `_analyze_index_expr(expr, spatial_axes, reduce_axes)`
    - `_apply_cache_write(step, state, step_idx)`

Functionality:

- dispatches by TVM step type
- mutates `SymbolicState.stages`, symbolic extents, annotations, and attach relations
- records metadata to later recover loop extents after compute-at and cache transforms
- runs final `infer_bound`-based reconciliation at the end of replay

Workflow position:

- the core replay engine behind `build_symbolic_state()`

## `modules/structural_sketch.py`

Role:

- deterministic canonicalization helpers for structural sketches

Functions:

- `build_canonical_param_values(state, split_value=1, unroll_value=0)`
  - create a deterministic `sp_*` / `ur_*` assignment for a concrete state
- `build_canonical_state(task, state, split_value=1, unroll_value=0)`
  - rebuild a concrete TVM state using the canonical param assignment

Functionality:

- strips away original split/unroll choices while preserving transform structure
- used before symbolic replay so the symbolic pipeline starts from a structural representative

Workflow position:

- preprocessing step before `TransformApplier`

## `modules/symbolic_state_bridge.py`

Role:

- symbolic-parameter bookkeeping and symbolic-state construction

Class:

- `SymParamManager`
  - field:
    - `self.s`
  - class constant:
    - `UNROLL_CANDIDATES = [0, 16, 64, 512, 1024]`
  - helper methods:
    - `_build_sp_groups()`
    - `_build_sp_extents(sp_groups)`
    - `_divisors(n)`

Top-level functions:

- `build_symbolic_state(task, state)`
  - canonicalize the concrete state
  - create a `SymbolicState`
  - replay transform steps into it
- `verify_symbolic_state(task, state, sym_state, verbose=False)`
  - compare symbolic reconstruction with TVM `infer_bound_from_state`

Functionality:

- `SymParamManager` is smaller than its name suggests in the current tree
- it mainly provides split-group/extents helpers and divisor enumeration
- the top-level functions are more important than the class itself
- this is the current owner after the naming-normalization pass
- `symbolic_state_bridge.py` remains as a compatibility shim only

Workflow position:

- bridge into symbolic reconstruction
- utility supplier for the generator/sampler

## `modules/legacy_record_sketch_io.py`

Role:

- legacy AutoScheduler record/sketch I/O and stable sketch identity helpers

Top-level constants:

- `STEP_RECORD_CODE`

Functions:

- structural fingerprint helpers:
  - `_step_structural_fingerprint(step)`
  - `step_record_code(step)`
  - `state_step_codes(state)`
  - `state_step_signature(state)`
  - `state_sketch_fingerprint(state)`
  - `sketch_fingerprint_repr(fp)`
  - `sketch_fingerprint_hash(fp)`
  - `raw_record_steps(record)`
  - `raw_record_step_codes(record)`
- `raw_record_step_signature(record)`
- filename and loading helpers:
  - `get_task_json_name(records_dir, task)`
  - `load_records_from_dir(tasks, records_dir)`
- grouping helpers:
  - `group_records_by_wkey_and_sketch(records)`
  - `group_by_sketches_from_json(tasks, records_dir, verbose=False)`
- legacy sketch-dump helpers:
  - `load_sketch_lines(sketches_path=None)`
  - `load_sketch_record(line, tasks_by_wkey)`

Functionality:

- fingerprints a sketch structurally rather than by exact split/unroll values
- groups records first by workload and then by structural sketch
- supports compact sketch IDs and human-readable signatures
- remains the legacy owner for record/sketch ingestion paths
- the only current non-I/O consumers in the repo are measurement-side sketch fingerprint imports

Workflow position:

- legacy ingestion and grouping layer for record-based workflows

## `modules/expr_nodes.py`

Role:

- symbolic expression tree representation used by constraints, propagation, and exact/projected GPU nodes

Base class:

- `ExprNode`

Node classes:

- `ConstNode`
- `VarNode`
- `MulNode`
- `AddNode`
- `SubNode`
- `MinNode`
- `CeilDivNode`
- `ScaleMulNode`
- `SumNode`
- `MaxNode`
- `PrimExprNode`
- `ProjectedExprNode`
- `BoundedMaxNode`
- `CaseSplitNode`

Per-node common operations:

- `interval(domains)`
- `evaluate(assignment)`
- `variables()`
- `__repr__()`

Important node-specific roles:

- `PrimExprNode`
  - wraps a TVM PrimExpr and evaluates intervals/eager values against TVM arithmetic
- `ProjectedExprNode`
  - holds display and exact nodes together
- `BoundedMaxNode`
  - max-like helper that tracks bound variables
- `CaseSplitNode`
  - exact case-table node with selectors, cases, optional defaults, and extra domains

Top-level helpers:

- `_safe_int_expr(expr)`
- `_maybe_int_expr(expr)`
- `parse_expr_tree(sym_expr_str)`
- parser internals:
  - `_skip_spaces(s, pos)`
  - `_parse_add_sub(s, pos)`
  - `_parse_mul(s, pos)`
  - `_parse_atom(s, pos)`

Functionality:

- represents symbolic left-hand sides of constraints
- supports both full evaluation and interval evaluation under partial assignments
- is shared by constraint building, propagation, exact/projected GPU construction, and diagnostics

Workflow position:

- foundational symbolic expression layer

## `modules/constraint_set.py`

Role:

- main symbolic constraint builder, checker, and preprocessing hub

Class:

- `ConstraintSet`

Main readable entry points:

- `preprocess()`
- `check_all_pruning(sym_map=None)`
- `check_all_exact(sym_map=None)`

Internal per-family builders/checkers:

- `_build_vectorize_constraints()`
- `_build_shared_memory_constraints()`
- `_build_max_threads_constraints()`
- `_build_max_vthread_constraints()`
- `_build_innermost_split_constraints()`
- `_build_split_structure_constraints()`
- `_check_vectorize(sym_map=None)`
- `_check_vectorize_exact(sym_map=None)`
- `_check_shared_memory(sym_map=None)`
- `_check_shared_memory_exact(sym_map=None)`
- `_check_max_threads(sym_map=None)`
- `_check_max_threads_exact(sym_map=None)`
- `_check_max_vthread(sym_map=None)`
- `_check_max_vthread_exact(sym_map=None)`
- `_check_innermost_split(sym_map=None)`
- `_check_split_structure(sym_map=None)`

Exact/projected GPU helpers:

- `_exact_upper_bound(node, sym_map)`
- `_exact_upper_bound_from_interval(interval)`
- `_evaluate_exact_upper_bounds(sym_map)`
- `_can_evaluate_exact_cases_concretely(...)`
- `_evaluate_exact_upper_bounds_concretely(...)`
- `_ensure_exact_gpu_constraints()`
- `_ensure_projected_gpu_context()`
- `_ensure_projected_gpu_constraints(kinds=None)`

Thread/vthread and structural helpers:

- `_collect_thread_binding_axes()`
- `_collect_vthread_binding_axes()`
- `_resolve_thread_axis_anchor(...)`
- `_resolve_block_scope(...)`
- `_binding_item_order_key(item)`
- `_group_binding_items_by_block_scope(items)`
- `_canonicalize_block_scope_binding_items(items)`
- `_format_block_scope(block_scope)`
- `_build_thread_per_block_constraint_item(block_scope, scoped_items)`
- `_build_thread_alias_entries(group, chosen)`
- `_thread_extent_preference_key(item)`
- `_coerce_product_form_tree(node)`
- `_normalize_legal_product_tree(node)`
- `_extract_product_form_vars(node)`
- `_extract_product_form_meta(node)`
- `_ordered_tree_variables(node)`
- `_ordered_unique_tree_variables(node)`
- `_append_unique_vars(target, names)`
- `_collect_preferred_thread_vars(items)`
- `_thread_axis_limit(ann)`
- `_flatten_max_terms(node)`
- `_dedupe_constraint_items(items, key_fn)`
- `_vectorize_item_key(item)`
- `_thread_extent_item_key(item)`
- `_vthread_item_key(item)`
- `_split_structure_item_key(item)`
- `_build_split_bound_denominator(names, extent)`
- `_build_split_structure_fast_path(sym_name, dependency_names, extent)`
- `_build_vthread_split_display_rhs(...)`
- `_flatten_mul_nodes(node)`
- `_collect_vthread_clamped_sp_names()`
- `_has_nonlinear(node)`

Functionality:

- builds all symbolic constraint bundles
- lazily builds exact/projected GPU nodes when needed
- collects metadata that later drives planner and propagator
- owns the core symbolic checking logic

Workflow position:

- semantic core of the generator/checker stack

## `modules/var_order_planner.py`

Role:

- compute variable order and phase decomposition for sampling and prefix debugging

Class:

- `VarOrderPlanner`

Methods:

- public-facing workflow methods:
  - `compute_var_order()`
  - `get_var_order_phase_entries()`
  - `_resolve_var_order_stop_index(stop_after_phase)`
- phase-first ordering:
  - `_build_phase_first_order(...)`
  - `_append_legacy_fallback_vars(...)`
  - `_build_var_order_phase_entries()`
  - `_make_phase_entry(...)`
- scope and ownership:
  - `_build_grid_scope_infos(...)`
  - `_build_scope_owned_vars(...)`
- phase variable collection:
  - `_collect_scoped_product_phase_vars(...)`
  - `_build_initial_domains()`
  - `_quick_interval(...)`
  - `_clip_interval(lo, hi, cap)`
  - `_classify_non_product_item_vars(...)`
  - `_collect_non_product_phase_vars(...)`
  - `_collect_step_indices_for_vars(var_names)`
  - `_collect_split_phase_vars_for_steps(step_indices, inner_first=False)`
- legacy fallback heuristics:
  - `_compute_legacy_var_order()`

Functionality:

- creates a phase-structured order rather than a flat arbitrary sequence
- groups vars into execution, memory, and instruction phases
- prefers pure-product thread/vthread factors early
- keeps prefix debugging aligned with semantically meaningful phases

Workflow position:

- between constraint preprocessing and sampling

## `modules/domain_propagator.py`

Role:

- incremental domain tightening, candidate filtering, and partial-domain analysis

Class:

- `DomainPropagator`

Methods:

- domain snapshots and state views:
  - `_snapshot_domains(domains)`
  - `_fixed_and_remaining_from_domains(domains)`
- constraint analysis:
  - `analyze_constraints_under_domains(domains)`
  - `_analyze_constraint_record(...)`
  - `_render_constraint_under_fixed_values(...)`
  - `_remaining_domain_subset(...)`
  - `_classify_constraint_bounds(...)`
  - `_build_constraint_analysis_item(...)`
  - `_analyze_constraint_bounds(...)`
  - `_enumerate_constraint_bounds(...)`
  - `_enumeration_budget(...)`
  - `_interval_bounds_from_expr_text(...)`
- upper-bound propagation:
  - `_apply_upper_bound_to_domain(dom, hi_allowed)`
  - `_get_sym_value(sym_map, name)`
  - `_propagate_product_form_upper(...)`
  - `_propagate_split_structure_upper(...)`
  - `_propagate_upper_domain_fast(...)`
  - `propagate_domain(assigned_name, domains)`
  - `_propagate_constraint_to_var(...)`
  - `_propagate_upper_constraint_to_var(...)`
  - `_tighten_upper_domain_from_candidates(...)`
  - `_tighten_upper_domain_by_interval(...)`
  - `_propagate_lower_constraint_to_var(...)`
- candidate filtering / bisection:
  - `_candidate_values_for_domain(var_name, dom)`
  - `filter_by_constraints(var_name, candidates, constraint_indices, domains)`
  - `_partition_constraints(...)`
  - `_bisect_upper(...)`
  - `_bisect_lower(...)`

Functionality:

- updates domains after assignments
- filters candidate values before random choice
- provides explainability reports for partial prefixes
- now reads as a staged workflow: domain views, constraint analysis, upper-bound propagation, and candidate filtering

Workflow position:

- inside the inner loop of sampling

## `modules/gpu_projection_constraints.py`

Role:

- build projected GPU pruning nodes from lowered symbolic TIR

Public builders:

- `build_projected_gpu_context(sym_state)`
- `build_projected_vectorize_constraint_node(projection_context, hw, allowed_var_names=None)`
- `build_projected_shared_memory_constraint_node(projection_context, hw, allowed_var_names=None)`
- `build_projected_constraint_nodes(exact_nodes, hw, allowed_var_names=None)`

Important internal helpers:

- runtime metadata extraction:
  - `_collect_projected_gpu_metadata(pre_func)`
- node conversion/projection:
  - `_expr_node_to_primexpr(node)`
  - `_has_noninteger_var(expr)`
  - `_project_runtime_upper(expr_node, runtime_domains)`
  - `_wrap_primexpr(expr)`
  - `_validate_projected_free_vars(...)`
  - `_collapse_max(children)`
  - `_dedupe_nodes(nodes)`
  - `_to_pruning_expr_node(node, constraint_name)`

Functionality:

- active projected vectorize/shared-memory pruning construction
- projection of exact nodes back into pruning nodes when diagnostics still request it
- runtime-domain-aware approximation

Workflow position:

- used by `ConstraintSet` for active projected pruning

## `modules/gpu_case_constraints.py`

Role:

- build exact GPU case tables from lowered symbolic TIR
- expose post-vectorize exact lowering helpers used by diagnostics

Public builders:

- `build_exact_constraint_nodes(...)`
- `lower_symbolic_post_vectorize_case(pre_func, selector_values)`

Important internal helpers:

- split-domain enumeration:
  - `_divisors(n)`
  - `_enumerate_group_assignments(...)`
  - `_build_sp_domains(...)`
- selector and case-table handling:
  - `_enumerate_selector_value_tuples(...)`
  - `_make_case_node(...)`
  - `_normalize_vector_cases(vector_cases)`
  - `_extract_all_gpu_case_stats_cached(pre_func, vector_cases)`

Functionality:

- exact symbolic GPU case-table construction
- exact case-stat extraction via TVM global functions

Workflow position:

- used lazily by `ConstraintSet` for explicit exact checking
- used by diagnostics that need post-vectorize exact lowering

## `modules/concrete_gpu_verify.py`

Role:

- concrete TVM lowering and GPU verification utilities

Top-level objects:

- `_s2m`
- `_verify_gpu_code`
- `_verify_gpu_code_errors`
- `GPU_PASSES`
- `GPU_VERIFY_CONSTRAINTS`

Functions:

- lowering and verification:
  - `lower_with_gpu_passes(task, state)`
  - `verify_gpu_module(mod, constraints=None)`
  - `verify_gpu_func_errors(func, constraints=None)`
  - `verify_gpu_module_errors(mod, constraints=None)`
- record patching and reconstruction:
  - `_patch_record_steps(record, params)`
  - `_params_to_state_from_measure_record(base_inp, base_res, params)`
  - `params_to_state_from_record(task, base_inp, base_res, params)`
  - `params_to_state_from_state(task, base_state, params)`
  - `params_to_state(task, base_inp, base_res, params)`
  - `params_to_lowered_gpu_module(task, base_inp, base_res, params)`

Functionality:

- rebuilds concrete states from sampled params
- lowers them through the GPU pass pipeline
- returns boolean or detailed error views from GPU verification

Workflow position:

- concrete final validation layer

## `modules/schedule_generator.py`

Role:

- central facade over the full symbolic-checking and sampling stack

Class:

- `ScheduleGenerator`
- `_ScheduleGeneratorInspector` (internal same-module helper owner)

Key class constants:

- `DEFAULT_HW_PARAM`
- `ALL_CONSTRAINT_KINDS`
- `VAR_ORDER_PHASE_FAMILIES`
- `_FORMAT_WRAP_LIMIT`

Constructor responsibilities:

- store symbolic state and hardware config
- store optional concrete-validation context
- create:
  - `SymParamManager`
  - `ConstraintSet`
  - `VarOrderPlanner`
  - `DomainPropagator`
  - `ParamSampler`
- initialize caches and bundle placeholders
- trigger preprocessing

Public workflow APIs:

- `from_task_state(...)`
- `get_full_var_order_entries()`
- `get_param_candidates(...)`
- `propagate_param_assignment(...)`
- `get_constraints_under_assignment(...)`
- `check_all_pruning(...)`
- `check_all_exact(...)`
- `check_all_hybrid(...)`
- `check_all_final(...)`
- `randomize_params(...)`
- `randomize_params_prefix(...)`
- `params_to_state(...)`

Research/debug observability contract:

- `get_full_var_order_entries()`
  - returns `{phase_count, param_order, phases}`
- `get_param_candidates(...)`
  - returns `{query, candidates, assignment, domains, constraints}`
- `propagate_param_assignment(...)`
  - returns `{query, assignment, domains, constraints}`
- `get_constraints_under_assignment(...)`
  - returns `{query, assignment, domains, constraints}`
  - `constraints.text` holds the formatted string view
- `randomize_params_prefix(...)`
  - returns `{query, phase_selection, param_order, phases, assignment, domains, constraints}`

Shared nested report fields:

- `assignment`
  - `{params}`
- `domains`
  - `{all, fixed, remaining}`
- `constraints`
  - `{text, leftover, resolved_false, resolved_true_count}`
- phase entries
  - `{phase_name, phase_family, phase_label, phase_index, grid_scope, grid_scope_label, param_names, param_entries, param_count, param_start, param_stop, prefix_param_names}`
- constraint-analysis items
  - `{constraint_kind, constraint_text, param_names?, domains?}`

Current in-repo helper surface:

- `_check_all_final_with_concrete_result(...)`
- `_normalize_concrete_params(...)`
- `_has_concrete_final_context()`
- `_get_concrete_final_result(sym_map=None)`
- `_get_constraint_records()`
- `_get_constraints_str(...)`
- `_get_raw_exact_constraints_str(...)`
- `_get_var_order_phase_entries()`
- `_materialize_assignment_state(...)`
- `_build_observability_report(...)`

Formatting and simplification bridge methods left on `ScheduleGenerator`:

- `_get_constraints_with_assignment_str(...)`
- `_simplify_constraint_expr_text(...)`
- `_simplify_constraint_rhs_text(...)`

Formatting and simplification ownership:

- `_ScheduleGeneratorInspector` now owns the formatting, expression rendering, simplification, and assignment-inspection machinery.
- The bridge methods above delegate to that helper rather than carrying the full implementation in the main facade body.

Lazy exact/projected-GPU helpers:

- `_ensure_exact_gpu_constraints()`
- `_ensure_projected_gpu_constraints(kinds=None)`

Functionality:

- workflow-oriented facade over checking, propagation, sampling, and concrete reconstruction
- delegates per-family symbolic checking to `ConstraintSet`
- delegates formatting/simplification internals to `_ScheduleGeneratorInspector`
- still exposes a small helper surface for current internal diagnostics callers

Workflow position:

- central facade and main user-facing object inside the module layer

## `modules/param_sampler.py`

Role:

- full and partial parameter sampling logic

Class:

- `ParamSampler`

Methods:

- `_assign_initial_fixed_vars(...)`
- `_build_split_domains()`
- `_initialize_split_sampling_state(...)`
- `_sample_split_var(...)`
- `_assign_unroll_vars(...)`
- `_validate_sample(...)`
- `_build_prefix_report(...)`
- `_randomize_params_with_order(...)`
- `randomize_params(rng=None, max_retries=1)`
- `randomize_params_prefix(stop_after_phase, rng=None, max_retries=1)`
- `_enumerate_all_params(max_results=100_000)`

Functionality:

- separates initialization, per-variable sampling, final validation, and prefix-report assembly
- samples split variables in planned order
- uses propagator hooks before and after assignments
- optionally assigns unroll values
- returns full samples, prefix-debug payloads, or exhaustive enumeration through an internal helper

Workflow position:

- concrete sampling engine behind `ScheduleGenerator`

## `modules/gpu_projection_diagnostics.py`

Role:

- diagnostic and triage helpers for projected/exact/concrete GPU mismatch analysis

Top-level constants:

- `ROOT_CAUSE_RUNTIME_PROJECTION`
- `ROOT_CAUSE_VECTORIZE_PROJECTION`
- `ROOT_CAUSE_VALIDATOR_DRIVER`
- `ROOT_CAUSE_SYMBOLIC_THREAD_BINDING`
- `ROOT_CAUSE_EXACT_INTERVAL_UNKNOWN`
- `ROOT_CAUSE_EXACT_SYMBOLIC_CASE_STAT`
- `ROOT_CAUSE_CUSTOM_POST_VECTORIZE_LOWERING`
- `_SENTINEL_UPPER_BOUND`
- `_LOWER_SYMBOLIC_POST_VECTORIZE`

Public helpers:

- `ensure_parent_dir(path)`
- `collect_gpu_projection_diagnostics(gen, params)`
- `collect_false_reject_diagnostics(gen, params, violations)`

Important internal helpers:

- `_kind_snapshot(...)`
- `_gpu_error_kind(message)`
- `_simplify_primfunc(func)`
- `_substitute_params_in_primfunc(func, params)`
- `_finite_interval_upper(expr, domains)`
- `_collect_exact_lowering_differential(gen, params, max_cases=8)`
- `_classify_exact_false_reject(kind, exact_lowering_differential)`

Functionality:

- load and rebuild `ScheduleGenerator` instances with concrete context
- compare exact and projected symbolic values
- compare symbolic cases against concrete GPU verify errors
- classify likely false reject / false accept causes

Workflow position:

- downstream diagnostic layer

## `generate.py`

Role:

- current ad hoc top-level exploration script

Top-level behavior:

- import TVM / NumPy
- set `TARGET = cuda`
- load tasks with `load_and_register_tasks()`
- call `SketchPolicy(...).generate_concrete_sketches()` for each task
- store sketches in in-memory maps
- hardcode `sketch_idx = 178`
- print the chosen sketch label
- build one symbolic state from the chosen sketch
- create one `ScheduleGenerator`

Functionality:

- manual experimentation only
- not a reusable module
- not a stable entrypoint contract

## Practical Read Order

If the goal is to understand the current code quickly, the most useful read order is:

1. `schedule_generator.py`
2. `constraint_set.py`
3. `var_order_planner.py`
4. `domain_propagator.py`
5. `param_sampler.py`
6. `symbolic_state_bridge.py`
7. `transform_applier.py`
8. `gpu_case_constraints.py`
9. `concrete_gpu_verify.py`
10. `gpu_projection_diagnostics.py`
11. `expr_nodes.py`
12. `symbolic_state.py`
13. `sym_types.py`
14. `legacy_record_sketch_io.py`
15. `task_paths.py`
16. `generate.py`

That order follows the actual semantic center of the repository:

- ordering
- constraints
- propagation
- sampling
- symbolic-state construction
- exact/projected/concrete verification
- diagnostic utilities

## Summary

The current repository is centered on four tightly connected modules:

- `constraint_set.py`
- `var_order_planner.py`
- `domain_propagator.py`
- `schedule_generator.py`

`param_sampler.py` is the execution engine for assignment, `symbolic_state_bridge.py` and `transform_applier.py` reconstruct the symbolic world, `gpu_case_constraints.py` and `concrete_gpu_verify.py` connect symbolic checks to concrete GPU semantics, and `gpu_projection_diagnostics.py` is the mismatch-triage layer.

If this codebase is reorganized later, the most stable semantic spine to preserve is:

- symbolic state reconstruction
- constraint construction
- variable-order planning
- domain propagation
- sampling
- exact/projected/concrete checking
- diagnostics
