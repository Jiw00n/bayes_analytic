# constrained_gen agent rules

These instructions apply when working in the `gallery/constrained_gen` tree.

## First read

- Read `docs_agents/CODEX_WORKING_CONTEXT.md` before planning.
- If using Codex CLI multi-agent orchestration, read `.codex/config.toml`.
- If using Codex CLI multi-agent orchestration, read `docs_agents/CLI_MULTI_AGENT_SETUP.md`.
- If working across multiple sessions, read `docs_agents/VSCODE_MULTI_AGENT_SETUP.md`.
- Verify every important claim against the current code before making a plan.

## Current phase

The current phase is generator completion and correctness hardening.

- Prioritize:
  - finishing schedule generation semantics
  - tightening symbolic constraints
  - validating symbolic vs concrete behavior
  - building reproducible diagnostics
- Do not prioritize yet:
  - performance tuning
  - broad refactors
  - code cleanup without a correctness payoff
- Do not activate the optional `optimizer` role unless the task explicitly focuses on profiling or performance optimization.
- If optimization opportunities are found, record them and defer them unless they directly unblock correctness work.

## Source of truth

Use concrete code paths before proposing changes.

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/modules/projected_gpu_validation.py`
- `gallery/constrained_gen/validate_exact_gpu_constraints.py`
- `gallery/constrained_gen/validate_projected_gpu_generation.py`
- `gallery/constrained_gen/modules/tvm_verify.py`
- `src/auto_scheduler/exact_gpu_constraints.cc`

## Current codebase caveat

- Check current code first.
- In particular, verify the current implementations of `ParamSampler._randomize_params_with_order()` and `ScheduleGenerator.check_all_hybrid()` before assuming anything about full sampling acceptance semantics.

## Agent split

When using multiple Codex sessions, keep responsibilities separate for the current correctness phase.

For Codex CLI multi-agent mode, the preferred setup is:

- main thread as orchestrator
- `validator` for raw execution
- `reviewer` for validation review
- `specialist` for root-cause analysis
- `integrator` for cross-module decisions and final edits when needed
- optional `optimizer` only for explicit perf or bottleneck tasks

The project-local Codex role config lives under `.codex/`.

- `integrator`
  - Owns generator behavior and cross-module API decisions.
  - Sole owner of `gallery/constrained_gen/modules/schedule_generator.py`.
  - Sole owner of `gallery/constrained_gen/modules/param_sampler.py`.
  - May also edit `gallery/constrained_gen/modules/var_order_planner.py` when generator logic requires it.
  - Accepts or rejects cross-cutting proposals from other sessions.
  - Is not the default first investigator for reviewer-confirmed issues that stay inside specialist-owned symbolic or lowering paths.

- `validator`
  - Owns validation execution and raw artifact production.
  - Works in `gallery/constrained_gen/validate_exact_gpu_constraints.py`, `gallery/constrained_gen/validate_projected_gpu_generation.py`, and `gallery/constrained_gen/modules/projected_gpu_validation.py`.
  - Reproduces issues before proposing fixes.
  - Writes raw validation artifacts under `/tmp/projected_gpu_full_validation/validator/...`.
  - Should not modify generator semantics directly.

- `reviewer`
  - Owns validation review, artifact sanity checks, and issue summarization.
  - Works in `gallery/constrained_gen/audit_non_pruning_correctness.py`, `gallery/constrained_gen/refresh_all_sketches_non_pruning_validation.py`, `gallery/constrained_gen/analyze_exact_case_dedupe_generalization.py`, and `gallery/constrained_gen/select_representative_projected_sketches.py`.
  - Reviews validator-produced artifacts before an issue is treated as established.
  - Produces concise issue summaries, mismatch counts, and escalation notes for the integrator or specialist.
  - Must classify whether the likely root-cause is integrator-owned, specialist-owned, or still inconclusive.

- `specialist`
  - Owns deeper root-cause analysis after validator reproduces an issue and reviewer confirms the evidence is sufficient.
  - Covers both projected/pruning false rejects and exact-vs-concrete lowering mismatches.
  - Works in `gallery/constrained_gen/modules/constraint_set.py`, `gallery/constrained_gen/modules/domain_propagator.py`, `gallery/constrained_gen/modules/tvm_verify.py`, `gallery/constrained_gen/modules/exact_gpu_constraints.py`, and `src/auto_scheduler/exact_gpu_constraints.cc`.
  - Becomes the default next investigation owner once reviewer confirms a reproduced issue in those paths.
  - Escalates cross-cutting API decisions back to the integrator.

- `optimizer`
  - Optional role for explicit profiling and performance optimization tasks.
  - Works in `gallery/constrained_gen/profile_schedule_generator_timing.py`, `gallery/constrained_gen/generate.py`, `gallery/constrained_gen/measure_programs.py`, and confirmed hot paths such as `gallery/constrained_gen/modules/param_sampler.py`.
  - Must measure first, optimize second, and preserve correctness semantics.

## Coordination rules

- Do not edit files owned by another active agent unless the integrator session explicitly takes over.
  - An agent is active until it leaves a handoff note or the integrator explicitly reassigns ownership.
- Do not edit `gallery/constrained_gen/modules/schedule_generator.py` or `gallery/constrained_gen/modules/param_sampler.py` outside the `integrator` session.
- Do not implement perf-only or cleanup-only changes in the current phase.
- If a change touches both symbolic pruning and concrete lowering semantics, stop and hand the decision to the integrator.
- If a validator run does not include a reproducer, treat the report as incomplete.
- If reviewer confirms a reproduced issue and the suspected root-cause path is specialist-owned, spawn `specialist` before asking `integrator` for the implementation plan unless the required change is clearly isolated to `schedule_generator.py` or `param_sampler.py`.
- Validator records raw validation outputs under `/tmp/projected_gpu_full_validation/validator/...`.
- Reviewer records reviewed summaries under `/tmp/projected_gpu_full_validation/reviewer/...`.
- Optimizer records profiling artifacts under `/tmp/projected_gpu_full_validation/optimizer/...`.
- Record deferred optimization ideas separately instead of mixing them into correctness patches.
- Prefer small reproducible shards before running full validation.
- When reporting a bug, include:
  - sketch index
  - task description
  - exact checker result
  - final checker result
  - concrete verifier result
- A reviewer-approved issue report should also state whether the current evidence is sufficient for specialist investigation.

## Role discipline

- Keep role boundaries strict when multi-agent orchestration is available.
- Do not let one session both:
  - author a non-trivial generator, exact-check, or lowering patch
  - and declare that patch sufficiently validated
- Use this default flow for non-trivial work:
  - `optimizer` measures bottlenecks and leaves artifacts
  - `integrator` accepts the implementation direction and owns cross-module edits
  - `validator` runs the smallest meaningful regression or correctness shard after the patch
  - `reviewer` decides whether the validator evidence is sufficient
- Treat the following files and areas as validation-required after meaningful edits:
  - `gallery/constrained_gen/modules/schedule_generator.py`
  - `gallery/constrained_gen/modules/param_sampler.py`
  - `gallery/constrained_gen/modules/constraint_set.py`
  - `gallery/constrained_gen/modules/domain_propagator.py`
  - `gallery/constrained_gen/modules/tvm_verify.py`
  - `gallery/constrained_gen/modules/exact_gpu_constraints.py`
  - `src/auto_scheduler/exact_gpu_constraints.cc`
  - validation entry points under `gallery/constrained_gen/validate_*.py`
- `optimizer` may patch only narrowly scoped, measured hotspots.
- If an optimization touches cross-module APIs, acceptance semantics, or exact/concrete correctness behavior, hand implementation ownership back to `integrator`.
- `validator` validates; it does not provide final approval.
- `reviewer` approves evidence; it does not silently replace validator execution except when validator artifacts are missing or unusable.
- `integrator` should not absorb the first-pass investigation of reviewer-confirmed issues in specialist-owned paths just because it can edit cross-module code.
- If multi-agent orchestration is unavailable and one session must do multiple roles, record that explicitly in the handoff note as `single-session validation only`.

## Documentation contract

Do not treat chat history as durable state. Each agent must leave file-based context before handing work off or ending a session.

- Temporary execution artifacts belong under `/tmp/projected_gpu_full_validation/<agent-name>/...`.
- Durable session notes belong under `docs_agents/handoffs/`.
- Use the templates in `docs_agents/templates/`.
- Name durable handoff notes with a `YYMMDDHHmm_` filename prefix so they sort in execution order.
- Prefer one short markdown note per meaningful task or investigation, instead of one giant running log.
- If a session ends without creating either a durable handoff note or a clearly named artifact set, treat that work as undocumented.

Minimum required handoff content:

- what was run or changed
- exact files/functions checked
- artifact paths
- concrete outcome
- remaining uncertainty
- next recommended owner

Role-specific documentation rules:

- `validator`
  - Must write raw artifacts under `/tmp/projected_gpu_full_validation/validator/...`.
  - Must leave a markdown handoff in `docs_agents/handoffs/` for any non-trivial run, failed reproducer, or issue escalation.
- `reviewer`
  - Must leave a markdown review note in `docs_agents/handoffs/` whenever it approves or rejects escalation.
- `specialist`
  - Must leave a markdown root-cause note in `docs_agents/handoffs/` whenever it recommends a fix or rejects a suspected root cause.
- `integrator`
  - Must leave a markdown decision note in `docs_agents/handoffs/` whenever it accepts a cross-module change or closes an investigation.
- `optimizer`
  - Must leave a markdown optimization note in `docs_agents/handoffs/` whenever it proposes or validates a performance change.

## Validation workflow

- For checker mismatches, start from `gallery/constrained_gen/validate_exact_gpu_constraints.py`.
- For generation failures, start from `gallery/constrained_gen/validate_projected_gpu_generation.py`.
- For prefix-domain triage, use `gallery/constrained_gen/refresh_all_sketches_prefix_through_split_structure.py`.
- For a proposed fix, validate the narrow reproducer first, then run the smallest meaningful shard, then decide whether a full sweep is justified.

## Environment

Before running repo code, set:

- `source /root/work/venv/bin/activate`
- `export TVM_HOME=/root/work/tvm-ansor`
- `export PYTHONPATH=$TVM_HOME/python`
- `export TVM_LIBRARY_PATH=$TVM_HOME/build-release`
