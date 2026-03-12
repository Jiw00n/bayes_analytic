# `constrained_gen` Docs

This directory contains the canonical human-facing docs for Codex work in `gallery/constrained_gen`.

`AGENTS.md` remains at the repository root so its instructions apply when working in `gallery/constrained_gen`. Supporting docs and reusable prompts live under `docs_agents/`.

## Start Here

When opening a new Codex session:

1. Read `AGENTS.md`
2. Read `docs_agents/CODEX_WORKING_CONTEXT.md`
3. If you are using multiple VS Code sessions, read `docs_agents/VSCODE_MULTI_AGENT_SETUP.md`
4. If you are using Codex CLI multi-agent orchestration, read `docs_agents/CLI_MULTI_AGENT_SETUP.md`
5. For the actual role wiring, inspect `.codex/config.toml`
6. Read `docs_agents/HANDOFF_WORKFLOW.md` before assuming one session can own implementation and sign-off

## Docs

- `CODEX_WORKING_CONTEXT.md`
  - current code structure, source-of-truth files, validation entry points
- `VSCODE_MULTI_AGENT_SETUP.md`
  - multi-session roles, startup prompts, slow-session refresh workflow
- `CLI_MULTI_AGENT_SETUP.md`
  - project-local Codex multi-agent roles and orchestrator workflow
- `HANDOFF_WORKFLOW.md`
  - durable documentation rules, naming scheme, and required note contents
  - patch follow-up expectations for validator and reviewer
- `prompts/`
  - ready-to-paste startup prompts for each agent role
- `templates/`
  - markdown templates for run notes, review notes, and decision handoffs
- `HISTORICAL_DOCUMENTS.md`
  - historical records and how to treat them

## Scope Note

- `AGENTS.md` is the only top-level operational doc that stays outside `docs_agents/`.
- Active durable notes live under `docs_agents/handoffs/`.
- Active durable notes should start with `YYMMDDHHmm_` so lexicographic order matches creation order.
- Historical handoff records live under `docs_agents/historical/`.
