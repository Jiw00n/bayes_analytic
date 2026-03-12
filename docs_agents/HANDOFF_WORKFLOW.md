# Handoff Workflow for `gallery/constrained_gen`

This file defines how Codex sessions leave durable context for later sessions.

## Rule

Chat history is not a durable handoff channel.

If a task matters after the current session ends, record it in files.

## Storage Split

Use two storage classes:

- temporary execution artifacts
  - location: `/tmp/projected_gpu_full_validation/<agent-name>/...`
  - examples: json summaries, jsonl mismatch logs, shard logs, raw stdout captures
- durable handoff notes
  - location: `docs_agents/handoffs/`
  - format: short markdown notes using `docs_agents/templates/`

## Required When Ending A Meaningful Task

Before ending a meaningful task, the active agent must leave:

1. at least one named artifact path or a clear statement that no artifact was produced
2. one markdown note in `docs_agents/handoffs/`

A meaningful task includes:

- a validation run
- a reproducer attempt
- a review pass
- a root-cause investigation
- a performance investigation
- a cross-module decision

## Naming Convention

Use this filename pattern for durable notes:

`docs_agents/handoffs/YYMMDDHHmm_<agent-name>_<topic>.md`

Use the same `YYMMDDHHmm` prefix at the start of the markdown title so recent notes are easy to scan in both directory listings and file contents.

Examples:

- `docs_agents/handoffs/2603122113_validator_generation_shard_00.md`
- `docs_agents/handoffs/2603122114_reviewer_exact_mismatch_triage.md`
- `docs_agents/handoffs/2603122116_integrator_hybrid_acceptance_decision.md`

## Minimum Note Contents

Every durable note must include:

- task or investigation name
- agent name
- date
- files and functions checked
- commands or scripts run
- artifact paths
- result or decision
- open questions or uncertainties
- recommended next owner

## Role Expectations

### Validator

- leave raw execution artifacts under `/tmp/projected_gpu_full_validation/validator/...`
- write a run note for any non-trivial run
- if reproduction failed, say what was tried and what is still missing

### Reviewer

- read validator artifacts before signing off
- write a review note that explicitly says one of:
  - sufficient for specialist escalation
  - insufficient, needs another validator pass
  - no issue confirmed

### Specialist

- write a root-cause note before proposing a broad fix
- distinguish confirmed root cause from plausible hypothesis

### Integrator

- write a decision note when accepting a cross-module change, rejecting a proposed direction, or closing an investigation
- if a non-trivial patch was applied, say whether validator/reviewer follow-up has already happened or is still required

### Optimizer

- write an optimization note whenever profiling leads to a proposed or validated performance change
- record both before and after measurements, or clearly state that no improvement was confirmed
- if the optimization changed correctness-sensitive code, say which validator follow-up is required

## Validation Expectation After Patches

Use validator and reviewer after non-trivial edits whenever multi-agent orchestration is available.

This is required in particular for edits touching:

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_sampler.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/domain_propagator.py`
- `gallery/constrained_gen/modules/tvm_verify.py`
- `gallery/constrained_gen/modules/exact_gpu_constraints.py`
- `src/auto_scheduler/exact_gpu_constraints.cc`
- validation entry points under `gallery/constrained_gen/validate_*.py`

If a single session had to both implement and validate because multi-agent orchestration was unavailable, state that explicitly in the note as `single-session validation only`.

## Templates

Use these files:

- `docs_agents/templates/HANDOFF_NOTE_TEMPLATE.md`
- `docs_agents/templates/VALIDATOR_RUN_NOTE_TEMPLATE.md`
- `docs_agents/templates/REVIEWER_NOTE_TEMPLATE.md`
- `docs_agents/templates/INTEGRATOR_DECISION_TEMPLATE.md`
- `docs_agents/templates/OPTIMIZER_NOTE_TEMPLATE.md`

## Reading Order For A Fresh Session

1. `AGENTS.md`
2. `docs_agents/CODEX_WORKING_CONTEXT.md`
3. `docs_agents/HANDOFF_WORKFLOW.md`
4. the most relevant recent file under `docs_agents/handoffs/`
5. the referenced artifact paths under `/tmp/projected_gpu_full_validation/...`
