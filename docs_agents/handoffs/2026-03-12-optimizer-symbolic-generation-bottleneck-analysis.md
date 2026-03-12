# Symbolic Generation Bottleneck Analysis

- Agent: `optimizer`
- Date: `2026-03-12`
- Status: `completed`
- Optimization Topic: `sketch -> symbolic state -> ScheduleGenerator init -> randomize_params bottleneck analysis`

## Scope

- Goal: Measure where time is spent from sketch recovery through symbolic-state creation and parameter generation.
- Target metric: Wall-clock time per stage on a small but meaningful reproducer.
- Why this hotspot was selected: The user reported that symbolic-state creation through parameter generation felt too slow.

## Files Checked

- `gallery/constrained_gen/profile_schedule_generator_timing.py`: `_profile_single_init_serial`, `_profile_randomize_internal`, `run`
- `gallery/constrained_gen/modules/schedule_generator.py`: `check_all_hybrid`, `get_concrete_final_result`
- `gallery/constrained_gen/modules/constraint_set.py`: `preprocess`, `check_all_exact`, `_ensure_exact_gpu_constraints`
- `gallery/constrained_gen/modules/exact_gpu_constraints.py`: `build_exact_constraint_nodes`, `build_projected_constraint_nodes`
- `gallery/constrained_gen/modules/param_sampler.py`: `_randomize_params_with_order`
- `gallery/constrained_gen/modules/domain_propagator.py`: `filter_by_constraints`, `propagate_domain`, `_bisect_upper`, `_bisect_lower`
- `gallery/constrained_gen/modules/expr_nodes.py`: `CaseSplitNode.interval`
- `gallery/constrained_gen/generate.py`: `generate_for_sketch`

## Measurement Setup

- Commands or scripts run:
  - `python gallery/constrained_gen/profile_schedule_generator_timing.py --sketch-index 0 --build-repeats 2 --phase-repeats 2 --randomize-repeats 2 --max-retries-values 1 --internal-repeats 1 --internal-max-retries 1`
  - multiple ad hoc Python timing snippets with the required TVM environment set
- Input shard or workload:
  - quick script sanity check on `sketch_index=0` (`vm_mod_fused_variance`)
  - detailed profiling on `sketch_index=2` (`vm_mod_fused_nn_batch_matmul_3`)
- Baseline artifact path:
  - `/tmp/projected_gpu_full_validation/optimizer/symbolic_generation_bottleneck_20260312/summary.json`

## Bottleneck Finding

- Confirmed hotspot:
  - `ScheduleGenerator` initialization for `sketch_index=2` is dominated by exact GPU case-table construction.
  - per-sample `randomize_params()` time on the same sketch is dominated by `check_all_exact()`, not by divisor enumeration or domain propagation.
- Evidence:
  - `build_symbolic_state`: about `208 ms`
  - `ScheduleGenerator(sym)`: about `20.85 s`
  - serial init breakdown: `build_vectorize_constraints` alone took about `20.89 s`
  - inside `build_exact_constraint_nodes`, `_EXTRACT_GPU_CASE_STATS` over `256` selector cases took about `20.36 s`
  - `build_projected_constraint_nodes` took about `218 ms`
  - `randomize_params()` on the same sketch averaged about `526 ms`
  - `check_all_exact()` alone averaged about `490 ms`
  - `check_all_pruning()` on the same params averaged about `0.594 ms`
  - domain propagation plus filtering during sampling was only about `33 ms` total per run
  - easy sketches with `1` vector case (`sketch_index=0`, `7`) randomized in about `2 ms`

## Change Summary

- What changed:
  - no code changes
  - created one raw artifact JSON and this handoff note
- What was intentionally not changed:
  - generator semantics
  - profiling script logic

## Results

- Before:
  - no current, working end-to-end profiler for this path; script is stale against current APIs
- After:
  - measured the main end-to-end stages and isolated the dominant hot sections
  - identified the current profiler breakage points
- Artifact paths:
  - `/tmp/projected_gpu_full_validation/optimizer/symbolic_generation_bottleneck_20260312/summary.json`

## Correctness Risk

- What might regress:
  - any optimization that changes exact-case construction, case selection sharing, or sampling acceptance flow could change correctness
- Required follow-up validation:
  - validate narrow repro on `sketch_index=2`
  - if an optimization is proposed, re-run exact-vs-concrete and projected-generation validation shards

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - decide whether to optimize init-time exact case-table construction first or exact-check reuse during `randomize_params`, then route validation through `validator`
