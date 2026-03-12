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

- public checker API such as `check_all_pruning()`, `check_all_exact()`, `check_all_hybrid()`, and `check_all_final()`
- concrete final validation through `get_concrete_final_result()`
- coordination across helper components

It delegates most implementation work to:

- `modules/constraint_set.py`
- `modules/var_order_planner.py`
- `modules/domain_propagator.py`
- `modules/param_sampler.py`

### Sampling acceptance semantics

Current code path:

- `ParamSampler._randomize_params_with_order()`
- if full validation is required and unroll is assigned, it calls `g.check_all_hybrid(result)`

`check_all_hybrid()` currently:

- runs `check_all_exact()`
- then, when concrete final context exists, runs concrete final validation through `get_concrete_final_result()`

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

Use `validate_exact_gpu_constraints.py` when checking whether symbolic pruning or exact checks disagree with concrete final validation.

It compares:

- `gen.check_all_pruning(params)`
- `gen.check_all_final(params)`
- `gen.get_concrete_final_result(params)`

This is the main script for finding:

- false accepts
- false rejects
- final checker mismatches

### Generation validation

Use `validate_projected_gpu_generation.py` when checking the output of random generation.

It currently:

- builds a `ScheduleGenerator`
- calls `gen.randomize_params(...)`
- lowers sampled params through `params_to_state(...)` and `lower_with_gpu_passes(...)`
- verifies the lowered module

This is the main script for confirming whether generated schedules survive concrete GPU verification.

### Shared validation helpers

`modules/projected_gpu_validation.py` contains common helpers for:

- loading sketch records
- building a `ScheduleGenerator`
- collecting projection diagnostics
- collecting exact-lowering differential data

## Role Boundaries For The Current Phase

- `integrator`
  - owns cross-module decisions
  - owns final edits in `modules/schedule_generator.py` and `modules/param_sampler.py`
  - may edit `modules/var_order_planner.py` when generator logic requires it
- `validator`
  - owns `validate_exact_gpu_constraints.py`
  - owns `validate_projected_gpu_generation.py`
  - owns `modules/projected_gpu_validation.py`
  - produces raw validation artifacts and repro outputs
  - should not change generator semantics directly
- `reviewer`
  - owns `audit_non_pruning_correctness.py`
  - owns `refresh_all_sketches_non_pruning_validation.py`
  - owns `analyze_exact_case_dedupe_generalization.py`
  - owns `select_representative_projected_sketches.py`
  - reviews validator artifacts before escalation
- `specialist`
  - owns deeper root-cause analysis after validator reproduces an issue and reviewer confirms it
  - covers both projected/pruning false-reject investigation and exact-vs-concrete lowering mismatch analysis
  - works in `modules/constraint_set.py`, `modules/domain_propagator.py`, `modules/tvm_verify.py`, `modules/exact_gpu_constraints.py`, and `src/auto_scheduler/exact_gpu_constraints.cc`
- `optimizer`
  - optional role for explicit profiling and performance work
  - owns `profile_schedule_generator_timing.py`, `generate.py`, and `measure_programs.py` for performance investigation
  - may inspect `modules/param_sampler.py` or related hot paths after profiling confirms them as bottlenecks

## Coordination Rules

- verify claims against current code before planning
- do not let two active sessions edit the same file at the same time
- only activate the optimizer role when the task explicitly focuses on profiling or performance
- if a change affects both symbolic pruning semantics and concrete lowering semantics, route the decision through the `integrator` session
- prefer the smallest reproducible validation shard before any broad rerun
- require validator artifacts before reviewer sign-off
- require reviewer sign-off before specialist escalation when the issue is not already obvious from a minimal reproducer
- if a session cites prior notes or memory, require it to cite the current file and function it checked
- require validator follow-up after meaningful edits in `schedule_generator.py`, `param_sampler.py`, `constraint_set.py`, `domain_propagator.py`, `tvm_verify.py`, `exact_gpu_constraints.py`, `src/auto_scheduler/exact_gpu_constraints.cc`, or validation entry points
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
