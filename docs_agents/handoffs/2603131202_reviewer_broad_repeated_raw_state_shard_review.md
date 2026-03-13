# Broad Repeated Raw-State Shard Review

- Agent: `reviewer`
- Date: `2026-03-13 12:02`
- Status: `approved`
- Review Target: `2603131200_validator_broad_repeated_raw_state_shard`

## Reviewed Inputs

- Validator note:
  - [2603131200_validator_broad_repeated_raw_state_shard.md](/root/work/tvm-ansor/docs_agents/handoffs/2603131200_validator_broad_repeated_raw_state_shard.md)
- Validator artifacts:
  - [/tmp/projected_gpu_full_validation/validator/raw_state_repeated_broad_2603131159/summary.json](/tmp/projected_gpu_full_validation/validator/raw_state_repeated_broad_2603131159/summary.json)
  - [/tmp/projected_gpu_full_validation/validator/raw_state_repeated_broad_2603131159/details.jsonl](/tmp/projected_gpu_full_validation/validator/raw_state_repeated_broad_2603131159/details.jsonl)

## Review Checks

- Sanity checks performed:
  - aggregate counts in `summary.json`
  - per-task coverage and family spread in `details.jsonl`
  - presence of diversity signals via sketch fingerprint and step signature counts
  - absence of concrete verify failures or exceptions in detailed rows
- Scripts or summaries used:
  - local JSON aggregation over `details.jsonl`

## Findings

- Review outcome phrase: `insufficient evidence`
  - There is insufficient evidence of a remaining issue after the structural representative state refactor.
- Aggregate validator evidence is internally consistent:
  - `selected_task_count=26`
  - `repeat_calls=156`
  - `raw_states_seen=222`
  - `randomize_success=444`
  - `concrete_invalid=0`
  - `exceptions=0`
- Coverage is meaningfully broad, not just a narrow reproducer:
  - `conv2d=144` attempts
  - `dense=60` attempts
  - `batch_matmul=48` attempts
  - `pool=48` attempts
  - `misc=144` attempts
- Diversity is present in the shard, so the run is not trivially repeating one sketch form everywhere.
  - e.g. several conv/dense tasks show `2` unique sketch fingerprints / step signatures.
- Missing or weak evidence:
  - no full-task repeated raw-state sweep
  - no explicit cold-process repetition matrix across multiple fresh processes

## Escalation Decision

- Evidence sufficient for specialist escalation: `no`
- If no, what validator should add:
  - only if we want still higher confidence, run a wider repeated raw-state shard or a full-task sweep
- If yes, what specialist should investigate first:
  - not applicable

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - accept the current structural representative state change for now
  - treat the remaining risk as validation breadth, not an established correctness issue
