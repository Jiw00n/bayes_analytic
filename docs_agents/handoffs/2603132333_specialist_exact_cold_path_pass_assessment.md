# 2603132333 Exact Cold Path Pass Assessment

- Agent: `specialist`
- Date: `2026-03-13 23:33`
- Status: `completed`
- Topic: `remaining cold exact-lowering hotspot pass assessment`

## Scope

- Goal:
  - determine which passes in `MakePostVectorizePipeline()` are required for `BuildGpuCaseStats()` correctness
  - assess whether `Simplify` and/or `InjectVirtualThread` can be removed or bypassed for `ExtractAllGpuCaseStats(...)`
- Why this mattered:
  - after parallel extract + process-local cache, the dominant remaining cold-path cost still sits in `src/auto_scheduler/exact_gpu_constraints.cc`

## Files Checked

- `src/auto_scheduler/exact_gpu_constraints.cc`: `MakePostVectorizePipeline`, `BuildGpuCaseStats`, `ExtractAllGpuCaseStats`, `GpuCaseStatsCollector`, `RuntimeDomainCollector`
- `gallery/constrained_gen/modules/exact_gpu_constraints.py`: `_extract_all_gpu_case_stats_cached`, `build_exact_constraint_nodes`
- `src/tir/transforms/inject_virtual_thread.cc`: `VirtualThreadInjector`, `VTInjector`
- `src/tir/transforms/storage_rewrite.cc`: `VisitStmt_(const AttrStmtNode*)`, scope handling around `attr::virtual_thread`

## Work Performed

- Commands or scripts run:
  - read current code in the files above
  - locally tested two one-line pipeline variants in this specialist workspace:
    - remove `InjectVirtualThread()`
    - restore `InjectVirtualThread()` and remove `Simplify()`
  - rebuilt:
    - `ninja -C build-release tvm`
  - narrow validation:
    - `python gallery/constrained_gen/validate.py --task-index 29`
    - `python gallery/constrained_gen/validate.py --task-index 383`
  - cold exact-init measurement:
    - a fresh Python process building task `383` and timing `_ensure_exact_gpu_constraints()`
- Code paths verified:
  - `BuildGpuCaseStats()` only consumes:
    - vectorized dtypes from `Allocate/Cast/BufferLoad/BufferStore`
    - thread/vthread extents from `AttrStmt` / `For`
    - runtime domains from `AttrStmt` / `For` / `Let`
  - `UpperBoundOverRuntimeVars()` and `BuildGpuCaseStats()` already call `arith::Analyzer::Simplify(...)` on the final expressions they return
  - `InjectVirtualThread` is not a no-op:
    - `inject_virtual_thread.cc` explicitly enlarges/rewrites touched buffers so each virtual thread has its own copy
    - this directly affects allocation sizes and therefore shared-memory exact stats
  - `StorageRewrite` itself already recognizes `attr::virtual_thread` scopes in `storage_rewrite.cc`

## Artifacts

- No separate `/tmp` artifact set created for this narrow specialist note.

## Outcome

- Result:
  - **Recommended narrow direction:** remove `tir::transform::Simplify()` from `MakePostVectorizePipeline()` for exact cold-path extraction.
  - **Not recommended:** remove `InjectVirtualThread()` from this path.
- Key evidence:
  - `InjectVirtualThread` removal is high-risk for correctness:
    - code comment and implementation show it enlarges touched buffers per virtual thread
    - that can change shared-memory exact accounting even if some sample tasks still pass
  - `Simplify` removal is much lower-risk:
    - collectors only need structural information already produced by `VectorizeLoop + InjectVirtualThread + StorageRewrite`
    - final exact stats are simplified again in `BuildGpuCaseStats()` / `UpperBoundOverRuntimeVars()`
  - experimental result in this specialist workspace:
    - with only `Simplify()` removed, `validate.py --task-index 29` passed
    - with only `Simplify()` removed, `validate.py --task-index 383` passed
    - task `383` cold `_ensure_exact_gpu_constraints()` dropped to about `0.55s`
    - prior integrator baseline after the earlier parallel-extract/cache work was about `6.1s`
  - experimental result for `InjectVirtualThread` removal:
    - task `29/383` still happened to pass and cold init became about `0.42s`
    - but code review shows this path is not low-risk enough because virtual-thread buffer replication can change shared-memory stats

## Uncertainty

- Remaining questions:
  - the `Simplify()` removal has only been checked on the current narrow shard (`task 29`, `task 383`)
- What is not yet proven:
  - broader-task exact/concrete agreement after removing `Simplify()`
  - whether any less obvious task relies on pass-level simplification before `BuildGpuCaseStats()`

## Next Owner

- Recommended owner: `integrator`
- Recommended next step:
  - land the narrow `Simplify()` removal in `src/auto_scheduler/exact_gpu_constraints.cc`
  - hand validation to `validator` for at least the current narrow shard plus one or two extra representative tasks before broader rollout
