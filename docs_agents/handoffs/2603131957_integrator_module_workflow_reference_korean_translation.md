# 2603131957 Module Workflow Reference Korean Translation

- Agent: `integrator`
- Date: `2026-03-13`
- Status: `completed`
- Decision Topic: `Korean translation for constrained_gen module/workflow reference`

## Inputs Considered

- Relevant artifacts:
  - [MODULE_AND_WORKFLOW_REFERENCE.md](/root/work/tvm-ansor/gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md)

## Files Checked

- [gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md](/root/work/tvm-ansor/gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.md): source English reference document

## Decision

- Chosen direction:
  - Add a separate Korean translation file next to the English reference.
- Rejected alternatives:
  - replacing the English original
  - mixing Korean text inline into the English document
- Why:
  - the user asked for an additional Korean version, and a parallel file preserves the original while keeping the translated copy easy to find

## Impact

- Files changed:
  - [gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.ko.md](/root/work/tvm-ansor/gallery/constrained_gen/MODULE_AND_WORKFLOW_REFERENCE.ko.md)
- Validation needed after change:
  - documentation-only change; no runtime validation required

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - update the Korean translation when the English reference document changes materially
