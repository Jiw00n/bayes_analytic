# Codex Working Context for `gallery/constrained_gen`

This document is the stable starting point for Codex sessions working in `gallery/constrained_gen`.

Use this document as the primary working context. It summarizes the current code structure, the main validation entry points, and the role boundaries.

## Orchestration Modes

There are two supported ways to work in this area.

- Preferred: Codex CLI multi-agent orchestration using `.codex/config.toml` and `.codex/agents/*.toml`
- Fallback: manual multi-session operation using `docs_agents/VSCODE_MULTI_AGENT_SETUP.md` and `docs_agents/prompts/`

In both modes:

- the main thread or main session should act as the orchestrator
- validator runs raw execution
- reviewer checks validator evidence
- specialist handles deeper root-cause analysis
- integrator owns cross-module decisions and final implementation ownership
- optimizer is optional and should be used only for explicit profiling or performance tasks
- validator and reviewer should be used after non-trivial patches touching correctness-sensitive generator or exact-check code
- no single session should both land a non-trivial semantics patch and self-sign it off when multi-agent orchestration is available

## Current phase

The current phase is correctness-first work on the constrained schedule generator.

- Finish schedule generation semantics.
- Tighten symbolic constraints and hybrid acceptance behavior.
- Validate symbolic results against concrete lowering and GPU verification.
- Build reproducible diagnostics and narrow validation shards.

The following work is intentionally deferred unless it directly unblocks correctness:

- performance tuning
- broad refactors
- cleanup-only code motion

## Scope

This area contains the constrained schedule generation and validation workflow around GPU-related schedule checks.

The main split in the current code is:

- symbolic pruning and exact checks
- concrete final validation through lowering and GPU verification

## Current Structure Verified from Code

### `ScheduleGenerator` facade

`modules/schedule_generator.py` is the main facade.

It owns:

- the workflow-level public API:
  - `from_task_state()`
  - `get_full_var_order_entries()`
  - `get_param_candidates()`
  - `propagate_param_assignment()`
  - `get_constraints_under_assignment()`
  - `randomize_params()`
  - `randomize_params_prefix()`
  - `params_to_state()`
  - `check_all_pruning()`
  - `check_all_exact()`
  - `check_all_hybrid()`
  - `check_all_final()`
- concrete-final helpers still used by in-repo diagnostics
- coordination across helper components
- an internal same-module inspector helper that owns formatting/simplification utilities

It delegates most implementation work to:

- `modules/constraint_set.py`
- `modules/var_order_planner.py`
- `modules/domain_propagator.py`
- `modules/gpu_projection_constraints.py`
- `modules/param_sampler.py`
- `modules/symbolic_state_bridge.py`
- the internal `_ScheduleGeneratorInspector` helper inside `modules/schedule_generator.py`

Naming note:

- `modules/symbolic_state_bridge.py` is the owning name for symbolic-state construction and parameter bookkeeping (no shim; param_manager was removed)

Current internal boundaries worth preserving:

- `ConstraintSet` reads through `preprocess()`, `check_all_pruning()`, and `check_all_exact()`; per-family build/check helpers are internal.
- `VarOrderPlanner` builds phase-first order first and only then appends legacy fallback vars.
- `ParamSampler` separates sampling-state initialization, per-variable sampling, final validation, and prefix-report assembly behind its internal helpers.
- `DomainPropagator` should read in staged sections: domain views, constraint analysis, upper-bound propagation, and candidate filtering.

Current observability contract:

- assignment-style reports use nested `assignment`, `domains`, and `constraints` objects
- `domains` is shaped as `{all, fixed, remaining}`
- `constraints` is shaped as `{text, leftover, resolved_false, resolved_true_count}`
- phase reports use `phase_*` field names plus `param_names` / `param_entries`

### Sampling acceptance semantics

Current code path:

- `ParamSampler._randomize_params_with_order()`
- if full validation is required and unroll is assigned, it calls `g.check_all_hybrid(result)`

`check_all_hybrid()` currently:

- first tries concrete final validation through the generator's concrete-final helper when concrete context exists
- falls back to `check_all_exact()` only when concrete final context is unavailable

Full sampling acceptance semantics must be checked from code, not inferred from notes or memory.

### Variable ordering

`modules/var_order_planner.py` builds explicit phase entries and then produces the final variable order.

Current phase families:

- `pure_product_max_threads`
- `pure_product_max_vthread`
- `split_structure_max_threads`
- `split_structure_max_vthread`
- `scaled_product_upper_bound`
- `non_product_direct_arm`
- `non_product_gate_vars`

The planner prefers a main compute anchor scope when ordering grid scopes.

## Validation Entry Points

### Exact-vs-concrete validation

Use `validate.py` as the main narrow validation harness for the active concrete-sketch -> symbolic generator workflow.

It currently:

- builds a `ScheduleGenerator` from a concrete sketch
- inspects var-order output through `get_full_var_order_entries()`
- runs `gen.randomize_params_prefix(...)`
- runs `gen.randomize_params(...)`
- checks `gen.check_all_pruning(params)`, `gen.check_all_exact(params)`, and `gen.check_all_final(params)`

This is the main script for finding:

- narrow workflow regressions
- checker mismatches
- observability/report-shape breakage

### Generation validation

Use `generate_programs.py` when checking the current constrained record-generation path.

It currently:

- iterates selected tasks from `load_and_register_tasks()`
- generates concrete sketches with `SketchPolicy(...).generate_concrete_sketches()`
- constructs `ScheduleGenerator` from each sketch
- samples params, checks final acceptance, converts params to concrete state, and saves records

This is the main script for confirming whether the active generation path can produce valid saved records.

Important current behavior:

- the active `generate_programs.py` runtime path does not gate acceptance on `check_all_exact()`
- active generation depends on projected pruning plus concrete final validation
- exact symbolic checks remain available for validation and diagnostics, not for the normal generation accept/reject path

### Measurement validation

Use `measure_programs.py` when checking whether generated constrained-gen records can be measured and saved through the repo's standard measurement path.

It currently:

- reads generated record files from one file or a directory
- rebuilds `MeasureInput` objects from saved states
- runs the standard TVM measurement path
- writes measured records and reports usable-vs-error outcomes

### Shared validation helpers

`modules/legacy_record_sketch_io.py` is the legacy helper path for:

- loading sketch records
- reading saved sketch dumps
- decoding saved sketch lines with a caller-supplied workload-key map

This is a legacy helper path, not the intended public API location.

`modules/gpu_projection_diagnostics.py` is the diagnostics-side module for:

- collecting projection diagnostics
- collecting exact-lowering differential data

`modules/gpu_projection_constraints.py` is the projected-pruning owner for:

- pre-vectorize symbolic lowering context
- projected vectorize/shared-memory pruning nodes
- projection of exact case nodes back into pruning nodes when diagnostics still request it

## Role Boundaries For The Current Phase

- `integrator`
  - owns cross-module decisions
  - owns final edits in `modules/schedule_generator.py` and `modules/param_sampler.py`
  - may edit `modules/var_order_planner.py` when generator logic requires it
  - is not the default first investigator for reviewer-confirmed issues that remain inside specialist-owned symbolic or lowering paths
- `validator`
  - owns `validate.py`
  - owns `modules/gpu_projection_diagnostics.py`
  - uses `generate_programs.py` and `measure_programs.py` as execution harnesses when a reproducer needs the current generation or measurement path
  - produces raw validation artifacts and repro outputs
  - should not change generator semantics directly
- `reviewer`
  - owns `audit_non_pruning_correctness.py`
  - owns `refresh_all_sketches_non_pruning_validation.py`
  - owns `analyze_exact_case_dedupe_generalization.py`
  - owns `select_representative_projected_sketches.py`
  - reviews validator artifacts before escalation
  - classifies likely root-cause ownership as integrator-owned, specialist-owned, or inconclusive
- `specialist`
  - owns deeper root-cause analysis after validator reproduces an issue and reviewer confirms it
  - covers both projected/pruning false-reject investigation and exact-vs-concrete lowering mismatch analysis
  - works in `modules/constraint_set.py`, `modules/domain_propagator.py`, `modules/concrete_gpu_verify.py`, `modules/gpu_projection_constraints.py`, `modules/gpu_case_constraints.py`, `src/auto_scheduler/projected_gpu_constraints.cc`, and `src/auto_scheduler/exact_gpu_constraints.cc`
  - is the default next investigation owner for reviewer-confirmed issues in those paths
- `optimizer`
  - optional role for explicit profiling and performance work
  - owns `profile_schedule_generator_timing.py`, `generate.py`, and `measure_programs.py` for performance investigation
  - may inspect `modules/param_sampler.py` or related hot paths after profiling confirms them as bottlenecks

## Coordination Rules

- verify claims against current code before planning
- do not let two active sessions edit the same file at the same time
- only activate the optimizer role when the task explicitly focuses on profiling or performance
- if reviewer confirms a reproduced issue and the suspected root-cause path stays inside `constraint_set.py`, `domain_propagator.py`, `concrete_gpu_verify.py`, `gpu_projection_constraints.py`, `gpu_case_constraints.py`, `src/auto_scheduler/projected_gpu_constraints.cc`, or `src/auto_scheduler/exact_gpu_constraints.cc`, spawn `specialist` before routing implementation planning to `integrator` unless the required fix is obviously isolated to `schedule_generator.py` or `param_sampler.py`
- if a change affects both symbolic pruning semantics and concrete lowering semantics, route the decision through the `integrator` session
- prefer the smallest reproducible validation shard before any broad rerun
- require validator artifacts before reviewer sign-off
- require reviewer sign-off before specialist escalation when the issue is not already obvious from a minimal reproducer
- if a session cites prior notes or memory, require it to cite the current file and function it checked
- require validator follow-up after meaningful edits in `schedule_generator.py`, `param_sampler.py`, `constraint_set.py`, `domain_propagator.py`, `concrete_gpu_verify.py`, `gpu_projection_constraints.py`, `gpu_case_constraints.py`, `src/auto_scheduler/projected_gpu_constraints.cc`, `src/auto_scheduler/exact_gpu_constraints.cc`, or the current execution entry points (`validate.py`, `generate_programs.py`, `measure_programs.py`)
- for explicit performance work, default to:
  - optimizer measures
  - integrator owns semantics-sensitive implementation
  - validator checks regressions
  - reviewer decides whether evidence is sufficient

## Durable Context Rules

Do not rely on chat history for cross-session continuity.

- Raw execution output belongs under `/tmp/projected_gpu_full_validation/<agent-name>/...`.
- Durable handoff notes belong under `docs_agents/handoffs/`.
- Reusable formats live under `docs_agents/templates/`.
- Manual role prompts for the VS Code fallback flow live under `docs_agents/prompts/`.

Every meaningful task should leave:

- a named artifact directory or output file
- a markdown handoff note with outcome and next owner

Expected note types by role:

- validator: run note
- reviewer: review note
- specialist: root-cause note
- integrator: decision note
- optimizer: optimization note

## Artifact Location

Write validation outputs under:

- `/tmp/projected_gpu_full_validation/validator/...`
- `/tmp/projected_gpu_full_validation/reviewer/...`
- `/tmp/projected_gpu_full_validation/optimizer/...`

Write durable handoff notes under:

- `docs_agents/handoffs/`

## Environment

Before running repo code, set:

- `source /root/work/venv/bin/activate`
- `export TVM_HOME=/root/work/tvm-ansor`
- `export PYTHONPATH=$TVM_HOME/python`
- `export TVM_LIBRARY_PATH=$TVM_HOME/build-release`
