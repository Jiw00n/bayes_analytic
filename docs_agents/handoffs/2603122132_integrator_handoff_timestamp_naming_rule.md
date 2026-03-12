# 2603122132 Handoff Timestamp Naming Rule

- Agent: `integrator`
- Date: `2026-03-12 21:32`
- Status: `accepted`
- Decision Topic: `handoff filename and title ordering`

## Inputs Considered

- Relevant docs:
  - [AGENTS.md](/root/work/tvm-ansor/AGENTS.md)
  - [HANDOFF_WORKFLOW.md](/root/work/tvm-ansor/docs_agents/HANDOFF_WORKFLOW.md)
  - [README.md](/root/work/tvm-ansor/docs_agents/README.md)
  - [HANDOFF_NOTE_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/HANDOFF_NOTE_TEMPLATE.md)
  - [INTEGRATOR_DECISION_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/INTEGRATOR_DECISION_TEMPLATE.md)
  - [OPTIMIZER_NOTE_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/OPTIMIZER_NOTE_TEMPLATE.md)
  - [REVIEWER_NOTE_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/REVIEWER_NOTE_TEMPLATE.md)
  - [VALIDATOR_RUN_NOTE_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/VALIDATOR_RUN_NOTE_TEMPLATE.md)

## Files Checked

- [AGENTS.md](/root/work/tvm-ansor/AGENTS.md): documentation contract
- [HANDOFF_WORKFLOW.md](/root/work/tvm-ansor/docs_agents/HANDOFF_WORKFLOW.md): naming convention
- [README.md](/root/work/tvm-ansor/docs_agents/README.md): active note guidance
- [HANDOFF_NOTE_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/HANDOFF_NOTE_TEMPLATE.md): generic note title/date fields
- [INTEGRATOR_DECISION_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/INTEGRATOR_DECISION_TEMPLATE.md): decision note title/date fields
- [OPTIMIZER_NOTE_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/OPTIMIZER_NOTE_TEMPLATE.md): optimizer note title/date fields
- [REVIEWER_NOTE_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/REVIEWER_NOTE_TEMPLATE.md): reviewer note title/date fields
- [VALIDATOR_RUN_NOTE_TEMPLATE.md](/root/work/tvm-ansor/docs_agents/templates/VALIDATOR_RUN_NOTE_TEMPLATE.md): validator note title/date fields

## Decision

- Chosen direction:
  - Require durable handoff filenames to start with `YYMMDDHHmm_`
  - Require markdown titles to start with the same `YYMMDDHHmm` prefix
  - Record note metadata dates with minute precision as `YYYY-MM-DD HH:MM`
- Why:
  - lexicographic ordering now reflects creation order within the same day
  - file listings and note bodies show the same timestamp prefix

## Impact

- Files likely to change:
  - none beyond the documentation and templates already updated
- Validation needed after change:
  - none; this is a docs/template rule change

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - use the new `YYMMDDHHmm_` prefix for all future notes under `docs_agents/handoffs/`
