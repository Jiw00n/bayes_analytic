# Role Discipline Generalization

- Agent: `integrator`
- Date: `2026-03-12`
- Status: `completed`

## What Changed

- Generalized multi-agent role-discipline rules across:
  - [AGENTS.md](/root/work/tvm-ansor/AGENTS.md)
  - [.codex/config.toml](/root/work/tvm-ansor/.codex/config.toml)
  - [.codex/agents/integrator.toml](/root/work/tvm-ansor/.codex/agents/integrator.toml)
  - [.codex/agents/optimizer.toml](/root/work/tvm-ansor/.codex/agents/optimizer.toml)
  - [.codex/agents/reviewer.toml](/root/work/tvm-ansor/.codex/agents/reviewer.toml)
  - [.codex/agents/specialist.toml](/root/work/tvm-ansor/.codex/agents/specialist.toml)
  - [.codex/agents/validator.toml](/root/work/tvm-ansor/.codex/agents/validator.toml)
  - [docs_agents/CLI_MULTI_AGENT_SETUP.md](/root/work/tvm-ansor/docs_agents/CLI_MULTI_AGENT_SETUP.md)
  - [docs_agents/CODEX_WORKING_CONTEXT.md](/root/work/tvm-ansor/docs_agents/CODEX_WORKING_CONTEXT.md)
  - [docs_agents/HANDOFF_WORKFLOW.md](/root/work/tvm-ansor/docs_agents/HANDOFF_WORKFLOW.md)
  - [docs_agents/README.md](/root/work/tvm-ansor/docs_agents/README.md)
  - [docs_agents/VSCODE_MULTI_AGENT_SETUP.md](/root/work/tvm-ansor/docs_agents/VSCODE_MULTI_AGENT_SETUP.md)
  - role prompts under [docs_agents/prompts/](/root/work/tvm-ansor/docs_agents/prompts/)

## Decision

- Make the default workflow explicit:
  - `optimizer` measures
  - `integrator` owns semantics-sensitive implementation
  - `validator` runs post-patch validation
  - `reviewer` decides whether the evidence is sufficient
- Require validator/reviewer follow-up after non-trivial edits to correctness-sensitive generator, exact-check, lowering, or validation-entry files.
- Explicitly forbid one multi-agent session from both landing a non-trivial semantics patch and self-signing it off.
- Require `single-session validation only` wording in handoff notes when multi-agent separation is unavailable.

## Files And Functions Checked

- Read current role/config docs under `.codex/` and `docs_agents/`
- Read current repo-local operating rules in [AGENTS.md](/root/work/tvm-ansor/AGENTS.md)

## Artifact Paths

- No raw execution artifact was required for this documentation-only update.

## Remaining Uncertainty

- These updates strengthen the operating contract, but they do not enforce role separation mechanically.
- Future work could add lint-like checks for required handoff metadata or validator/reviewer follow-up markers.

## Next Recommended Owner

- Recommended owner: `integrator`
- Recommended next step: follow these updated rules on the next non-trivial optimization or correctness patch and confirm the workflow is practical.
