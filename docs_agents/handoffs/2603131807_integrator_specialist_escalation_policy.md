# 2603131807 Specialist Escalation Policy

- Agent: `integrator`
- Date: `2026-03-13`
- Status: `completed`

## What Changed

- Tightened the multi-agent operating rules so reviewer-confirmed issues in specialist-owned symbolic or lowering paths route to `specialist` before `integrator` implementation planning.
- Updated role prompts and role configs so `reviewer` must classify likely ownership and `integrator` is no longer treated as the default first investigator for those cases.

## Files And Functions Checked

- [AGENTS.md](/root/work/tvm-ansor/AGENTS.md)
- [.codex/config.toml](/root/work/tvm-ansor/.codex/config.toml)
- [.codex/agents/integrator.toml](/root/work/tvm-ansor/.codex/agents/integrator.toml)
- [.codex/agents/reviewer.toml](/root/work/tvm-ansor/.codex/agents/reviewer.toml)
- [.codex/agents/specialist.toml](/root/work/tvm-ansor/.codex/agents/specialist.toml)
- [docs_agents/CODEX_WORKING_CONTEXT.md](/root/work/tvm-ansor/docs_agents/CODEX_WORKING_CONTEXT.md)
- [docs_agents/CLI_MULTI_AGENT_SETUP.md](/root/work/tvm-ansor/docs_agents/CLI_MULTI_AGENT_SETUP.md)
- [docs_agents/VSCODE_MULTI_AGENT_SETUP.md](/root/work/tvm-ansor/docs_agents/VSCODE_MULTI_AGENT_SETUP.md)
- [docs_agents/HANDOFF_WORKFLOW.md](/root/work/tvm-ansor/docs_agents/HANDOFF_WORKFLOW.md)
- [docs_agents/prompts/integrator.md](/root/work/tvm-ansor/docs_agents/prompts/integrator.md)
- [docs_agents/prompts/reviewer.md](/root/work/tvm-ansor/docs_agents/prompts/reviewer.md)
- [docs_agents/prompts/specialist.md](/root/work/tvm-ansor/docs_agents/prompts/specialist.md)

## Commands Or Scripts Run

- `sed -n` reads over the active role docs and prompts
- `date +%y%m%d%H%M`

## Artifact Paths

- No raw execution artifact was produced for this documentation-only policy update.

## Result Or Decision

- The default correctness workflow is now documented more explicitly as `validator -> reviewer -> specialist -> integrator -> validator -> reviewer` when reviewer confirms a reproduced issue in specialist-owned paths.
- `reviewer` is now responsible for saying whether the likely root-cause is integrator-owned, specialist-owned, or inconclusive.

## Open Questions Or Uncertainties

- These changes improve the written contract but do not mechanically enforce orchestration order.
- A future improvement could add template fields or automation checks that fail a handoff when reviewer ownership classification is missing.

## Recommended Next Owner

- Recommended owner: `integrator`
- Recommended next step: use the updated prompts on the next reproduced correctness issue and confirm that `specialist` is spawned before `integrator` for specialist-owned paths.
