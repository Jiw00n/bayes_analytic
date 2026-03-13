# Codex CLI Multi-Agent Setup for `gallery/constrained_gen`

This is the preferred setup when you want one main Codex thread to orchestrate sub-agents automatically.

## Where The Role Config Lives

Project-local Codex multi-agent settings live under:

- `.codex/config.toml`
- `.codex/agents/integrator.toml`
- `.codex/agents/validator.toml`
- `.codex/agents/reviewer.toml`
- `.codex/agents/specialist.toml`
- `.codex/agents/optimizer.toml`

The role file paths are wired through `.codex/config.toml`.

## What The Main Thread Should Do

Use the main thread as the orchestrator.

The main thread should:

- keep requirements and final decisions in the main context
- delegate validation execution to `validator`
- delegate validation review to `reviewer`
- delegate root-cause analysis in specialist-owned paths to `specialist`
- avoid using `integrator` as the default first investigator for reviewer-confirmed issues in `constraint_set.py`, `domain_propagator.py`, `tvm_verify.py`, `exact_gpu_constraints.py`, or `src/auto_scheduler/exact_gpu_constraints.cc`
- use `integrator` when a cross-module implementation or final edit ownership decision is needed after the root-cause path is clear
- use `optimizer` only when the task explicitly focuses on profiling or bottleneck reduction
- avoid doing implementation, validation, and review in the same session when multi-agent orchestration is available

This follows the Codex multi-agent guidance that parallel agents are best for read-heavy exploration, tests, triage, and summarization, while write-heavy work should stay tightly coordinated. Source: OpenAI Codex multi-agent docs.

## Suggested Main-Thread Prompt Pattern

Use prompts like:

```text
Work on gallery/constrained_gen.
Keep the main thread focused on decisions and summaries.
Delegate raw validation execution to validator, validation review to reviewer, and deep root-cause analysis to specialist.
Do not send reviewer-confirmed specialist-owned failures straight to integrator for first-pass investigation.
Use integrator only if a final cross-module edit plan or implementation is needed.
Wait for all delegated work, then summarize the result and next action.
```

For a concrete issue:

```text
Investigate whether hybrid acceptance is letting invalid schedules through in gallery/constrained_gen.
Spawn validator to reproduce on the smallest meaningful shard.
Then spawn reviewer to inspect the validator artifacts and decide whether the evidence is sufficient.
If reviewer confirms a real issue and the likely root-cause path is specialist-owned, spawn specialist before asking integrator for the implementation plan.
Use integrator only for the final implementation decision or cross-module edits after the root cause is narrowed.
Keep final decisions in the main thread.
```

For a performance task:

```text
Investigate slow paths in gallery/constrained_gen.
Use optimizer to profile before making any code changes.
Require optimizer to leave profiling artifacts and an optimization note.
If a performance patch is proposed, route implementation ownership through integrator when the change is semantics-sensitive or cross-module.
After the patch, require validator to run a narrow correctness or regression shard and require reviewer to decide whether the evidence is sufficient.
Keep final acceptance in the main thread.
```

## Operational Rules

- CLI multi-agent is experimental.
- Multi-agent activity is currently surfaced in the CLI first.
- Sub-agents inherit the parent sandbox and approval state unless the role config overrides it.
- Each meaningful delegated task should still leave artifacts plus a handoff note following `docs_agents/HANDOFF_WORKFLOW.md`.
- Do not use `optimizer` for cleanup-only or speculative tuning tasks.
- Treat `optimizer -> integrator -> validator -> reviewer` as the default chain for non-trivial performance work.
- Treat `validator -> reviewer -> specialist -> integrator -> validator -> reviewer` as the default chain for correctness bugs that need a fix.
- If `reviewer` marks a reproduced issue as specialist-owned and escalation-sufficient, `specialist` should be the default next worker instead of `integrator`.

## Fallback Mode

If you are not working in the CLI surface or want explicit manual control, use the manual role prompts in:

- `docs_agents/prompts/`
- `docs_agents/VSCODE_MULTI_AGENT_SETUP.md`
