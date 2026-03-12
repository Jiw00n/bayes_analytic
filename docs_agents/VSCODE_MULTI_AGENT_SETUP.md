# VS Code Multi-Agent Setup for `gallery/constrained_gen`

This guide is the fallback operating mode when you are using the Codex VS Code extension or otherwise want explicit manual session control.

If Codex CLI multi-agent orchestration is available for your workflow, prefer `docs_agents/CLI_MULTI_AGENT_SETUP.md` first.

## What "multi-agent" Means Here

In the VS Code extension, the practical model is multiple parallel Codex sessions, each with a narrow role.

- You normally drive each session directly.
- There is no project-local automatic sub-agent orchestration to rely on here.
- Use one session as the decision-maker and keep the others scoped to evidence gathering or isolated fixes.
- The current phase is correctness-first work. Do not spin up a separate perf or refactor session yet.
- Treat files, not chat, as the shared memory between sessions.
- When multiple sessions are available, do not let one session both land a non-trivial semantics patch and self-sign it off.

## Recommended Session Layout

Use four sessions for the current `constrained_gen` workflow when you want validation execution and validation review to be separated.

If you have an explicit profiling or performance task, you may open a fifth session as `optimizer`.

### `integrator`

Use this as the main session in your editor.

- reads current working docs and current code
- decides whether a proposed fix belongs in sampler, facade, projected constraints, or lowering
- owns final edits when multiple modules are involved
- hands post-patch regression or correctness checks to `validator`
- expects `reviewer` to say whether the validator evidence is sufficient

Suggested opening prompt:

```text
You are the integrator for gallery/constrained_gen.
Read AGENTS.md first, then read docs_agents/CODEX_WORKING_CONTEXT.md.
Your job is to make final decisions and own edits in modules/schedule_generator.py and modules/param_sampler.py when coordination is needed.
Verify code paths first and treat current code as the only source of truth.
```

### `validator`

Use this for validation execution and raw artifact generation.

- re-runs exact validation and generation validation
- produces raw shard outputs and mismatch examples
- avoids broad semantic changes unless the reproducer proves them necessary

Suggested opening prompt:

```text
You are the validator for gallery/constrained_gen.
Read AGENTS.md first.
Then read docs_agents/CODEX_WORKING_CONTEXT.md.
Focus on validate_exact_gpu_constraints.py, validate_projected_gpu_generation.py, and modules/projected_gpu_validation.py.
Reproduce issues with the smallest meaningful shard, write artifacts under /tmp/projected_gpu_full_validation/validator, and report only code-verified findings.
```

### `reviewer`

Use this to inspect validation artifacts and decide whether the evidence is strong enough to escalate.

- reviews validator outputs instead of re-owning the same execution loop
- runs audit or summary scripts over representative or full-shard artifacts
- produces issue summaries and escalation notes for the integrator or specialist

Suggested opening prompt:

```text
You are the reviewer for gallery/constrained_gen.
Read AGENTS.md first.
Then read docs_agents/CODEX_WORKING_CONTEXT.md.
Focus on audit_non_pruning_correctness.py, refresh_all_sketches_non_pruning_validation.py, analyze_exact_case_dedupe_generalization.py, and select_representative_projected_sketches.py.
Review validator artifacts under /tmp/projected_gpu_full_validation/validator, write reviewed summaries under /tmp/projected_gpu_full_validation/reviewer, and say explicitly whether the evidence is sufficient for specialist escalation.
```

### `specialist`

Use this when the issue needs deeper technical analysis beyond straightforward validation.

- covers projected upper bounds that are too conservative
- covers projected/pruning false rejects
- covers exact symbolic lowering vs concrete lowering mismatch

Suggested opening prompt:

```text
You are the specialist agent for gallery/constrained_gen.
Read AGENTS.md first.
Then read docs_agents/CODEX_WORKING_CONTEXT.md.
Focus on the relevant root-cause path in modules/constraint_set.py, modules/domain_propagator.py, modules/tvm_verify.py, modules/exact_gpu_constraints.py, and src/auto_scheduler/exact_gpu_constraints.cc.
Produce a minimal repro before suggesting edits, and hand cross-module API decisions back to the integrator. Treat reviewer sign-off as the default gate before deep investigation.
```

### `optimizer` (optional)

Use this only for explicit profiling or bottleneck reduction tasks.

- profiles before changing code
- focuses on performance entry points and confirmed hot paths
- leaves before and after measurements instead of intuition-only claims
- hands semantics-sensitive implementation ownership back to `integrator`
- does not self-sign off correctness-sensitive performance patches

Suggested opening prompt:

```text
You are the optimizer for gallery/constrained_gen.
Read AGENTS.md first.
Then read docs_agents/CODEX_WORKING_CONTEXT.md and docs_agents/HANDOFF_WORKFLOW.md.
Focus on profile_schedule_generator_timing.py, generate.py, measure_programs.py, and any confirmed hotspot such as modules/param_sampler.py.
Measure first, optimize second, leave profiling artifacts under /tmp/projected_gpu_full_validation/optimizer, and write an optimization note under docs_agents/handoffs/ using docs_agents/templates/OPTIMIZER_NOTE_TEMPLATE.md.
```

Ready-to-paste copies of these prompts live under `docs_agents/prompts/`.
Durable handoff rules and templates live under `docs_agents/HANDOFF_WORKFLOW.md` and `docs_agents/templates/`.

## Recommended Session Flow

Use these default chains unless the task is trivial:

- Performance task:
  - `optimizer` measures
  - `integrator` accepts implementation ownership when the change is cross-module or semantics-sensitive
  - `validator` runs the narrowest meaningful regression shard
  - `reviewer` decides whether the evidence is sufficient
- Correctness bug:
  - `validator` reproduces
  - `reviewer` confirms evidence
  - `specialist` analyzes root cause when needed
  - `integrator` owns final implementation
  - `validator` re-runs the narrow shard
  - `reviewer` signs off on evidence quality

## Practical Use

Lightweight mode:

- open one Codex chat as `integrator`
- open a second Codex chat as `validator`
- open a third Codex chat as `reviewer`
- open a fourth Codex chat as `specialist`
- open an optional fifth Codex chat as `optimizer` only for explicit perf work
- paste the role prompt into each session
- keep each session inside its assigned files

Safer mode with worktrees:

```bash
git worktree add ../tvm-ansor-validator -b codex/validator
git worktree add ../tvm-ansor-reviewer -b codex/reviewer
git worktree add ../tvm-ansor-specialist -b codex/specialist
git worktree add ../tvm-ansor-optimizer -b codex/optimizer
```

Then open each worktree in a separate VS Code window and attach one Codex session per window.

## Refreshing Slow Sessions

Long Codex sessions often slow down. Treat sessions as disposable and keep state in files, not chat history.

- keep `integrator` relatively long-lived
- keep `validator`, `reviewer`, `specialist`, and `optimizer` short-lived
- restart a specialist session once it becomes slow
- use docs plus artifact paths as the session refresh context

Recommended pattern:

1. finish a narrow task
2. write the result into an artifact path and a durable handoff note
3. close the slow session
4. open a fresh session
5. read `AGENTS.md`, `docs_agents/CODEX_WORKING_CONTEXT.md`, `docs_agents/HANDOFF_WORKFLOW.md`, and this file again
6. continue from files, not chat memory
