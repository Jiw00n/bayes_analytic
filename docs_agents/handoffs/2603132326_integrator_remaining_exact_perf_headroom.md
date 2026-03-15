# 2603132326 Remaining Exact Perf Headroom

- Agent: `integrator`
- Date: `2026-03-13 23:26`
- Status: `measured`
- Topic: `remaining optimization headroom after exact parallel extract + in-memory cache`

## Files Checked

- `gallery/constrained_gen/modules/exact_gpu_constraints.py`
- `gallery/constrained_gen/modules/constraint_set.py`
- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/param_manager.py`
- `src/auto_scheduler/exact_gpu_constraints.cc`

## What Was Measured

- `task 383` on `gallery/constrained_gen/validate.py` path
- cold vs warm stage timing inside one Python process
- exact build decomposition:
  - projected GPU context build
  - selector-case enumeration
  - cached case-stats extraction
- exact evaluation after init
- construction split between symbolic-state build and `ScheduleGenerator` init

## Measured Results

- `task 383` stage breakdown, cold process:
  - `construct`: `~0.451s`
  - `var_order`: `~0.000s`
  - `prefix`: `~0.018s`
  - `sample`: `~0.031s`
  - `pruning`: `~0.001s`
  - `exact`: `~6.372s`
  - `final`: `~0.000s`
- `task 383` stage breakdown, warm same-process repeat:
  - `construct`: `~0.266s`
  - `exact`: `~0.341s`

- exact build decomposition:
  - cold:
    - `build_projected_gpu_context`: `~0.219s`
    - `_enumerate_selector_value_tuples`: `~0.006s`
    - `_extract_all_gpu_case_stats_cached`: `~5.992s`
  - warm same process:
    - `build_projected_gpu_context`: `~0.218s`
    - `_enumerate_selector_value_tuples`: `~0.006s`
    - `_extract_all_gpu_case_stats_cached`: `~0.017s`

- repeated exact evaluation after init:
  - `gen.check_all_exact(params)`: `~0.196s` per call
  - feasible exact cases: `256`
  - `feasible_case_values(...)`: `~0.091s`
  - `_evaluate_exact_upper_bounds_concretely(...)`: `~0.100s`

- construction split:
  - `build_symbolic_state(task, state)`: `~0.203s`
  - `ScheduleGenerator(...)` init: `~0.247s`

## Ranked Remaining Optimization Candidates

1. `ExtractAllGpuCaseStats(...)` cold-path lowering remains the dominant hotspot.
   - Evidence:
     - still `~5.99s` of the `~6.37s` cold exact time
     - selector enumeration and projected-context build are much smaller
   - Likely payoff:
     - largest remaining cold-start win by far
   - Direction:
     - reduce per-case lowering work or redesign post-vectorize analysis

2. Exact evaluation after init (`check_all_exact`) is now the top warm-path hotspot.
   - Evidence:
     - `~0.196s` per call after init
     - split almost evenly between feasible-case enumeration and concrete max reduction over `256` cases
   - Likely payoff:
     - moderate warm-path win
   - Direction:
     - reduce feasible-case work or aggregate exact maxima more cheaply

3. `ScheduleGenerator.from_task_state(...)` construction cost is the next visible warm-path cost.
   - Evidence:
     - `~0.266s` warm repeat, with `~0.203s` in symbolic-state build and `~0.247s` in generator init on cold split
   - Likely payoff:
     - secondary compared with exact cold path

## Low-Payoff Areas

- `get_full_var_order_entries()`: effectively negligible
- prefix sampling: `~0.018s`
- full sampling: `~0.031s`
- pruning/final checks: negligible
- selector-case enumeration itself: `~0.006s`

## Remaining Uncertainty

- no broader task sweep was run here
- no role-pure optimizer/reviewer sign-off for this specific headroom ranking

## Recommended Next Owner

- `specialist` for further cold exact-lowering reduction in `exact_gpu_constraints.py` / `exact_gpu_constraints.cc`
- `integrator` only after the specialist picks a narrow semantics-safe direction
