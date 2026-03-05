# TVM Auto-Scheduler GPU Constraint System - 진행 상황 종합 보고서

## 목표

TVM auto-scheduler에서 생성된 스케줄 파라미터(split factor, unroll value)가 측정(measurement) 전에 항상 valid한지 보장하는 제약 시스템 구축. 랜덤으로 파라미터를 채워도 `inferbound`와 GPU 하드웨어 제약을 모두 만족하도록 만드는 것이 최종 목표.

**핵심 제약**: 제약식을 만들기 위해 `get_per_store_features_from_states`를 처음 한 번만 사용 가능. 이후 파라미터 생성 시에는 TVM lowering 호출 없이 독립적으로 valid한 파라미터를 생성해야 함.

---

## 배경 지식

### State Validity 검증 과정

1. JSON record에서 `SearchTask`와 `State`를 복원
2. `task.compute_dag.apply_steps_from_state(state)` → TE schedule 생성
3. `ScheduleToModule` (C++: `driver.schedule_to_module`) → IRModule 변환
4. GPU Pass Pipeline 적용:
   ```
   InjectPrefetch → StorageFlatten(64) → NarrowDataType(32) → Simplify
   → VectorizeLoop → InjectVirtualThread → StorageRewrite → Simplify
   ```
5. `VerifyGPUCode` (C++: `tir.analysis.verify_gpu_code`) → GPU 제약 확인
6. Feature extraction (`get_per_store_features_from_states`) → all-zero이면 invalid

### GPU 하드웨어 제약 (sm_86 기준)

| 제약 | 한계값 |
|------|--------|
| `max_threads_per_block` | 1024 |
| `max_shared_memory_per_block` | 49152 bytes |
| `max_local_memory_per_block` | 2^31 - 1 bytes |
| `max_vthread` | 8 |
| `max_vector_bytes` | 16 bytes |
| `max_innermost_split_factor` | 64 |
| `warp_size` | 32 |

### 스케줄 파라미터 구조

JSON record 내 변경 가능한 파라미터:
- **SP (SplitStep)**: `["SP", stage_id, iter_id, extent, [lengths], inner_to_outer]`
  - Spatial 4-way split: `[l0, l1, l2, l3]` where `l0*l1*l2*l3 | extent`
  - Reduce 2-way split: `[l0, l1]` where `l0*l1 | extent`
- **PR (PragmaStep)**: `["PR", stage_id, iter_id, "auto_unroll_max_step$VALUE"]`
  - VALUE ∈ {0, 16, 64, 512, 1024}

파라미터 의미 (spatial 4-way):
- `l3`: vthread extent (product ≤ 8)
- `l2`: thread extent (product ∈ [warp_size, max_threads_per_block])
- `l0`: innermost split factor (≤ 64)
- `l1`: 나머지 (unconstrained)

---

## 제약 확인 과정 (플로우)

제약을 확인하는 데는 **두 가지 경로**가 있고, 서로 다른 시점·목적으로 쓰인다.

### 경로 A: “실제로 valid인지” 확인 (정답 기준)

측정 전에 이 스케줄이 정말 GPU 제약을 만족하는지 **최종 판정**할 때 사용한다.

1. **입력**: JSON `record` (스케줄 + 파라미터가 들어 있는 한 줄).
2. **Record → Task, State 복원**
   - record를 임시 JSON 파일로 쓰고 `auto_scheduler.load_records` → `recover_measure_input(..., rebuild_state=True)` 호출.
   - 결과: `SearchTask`(`task`), `State`(`state`).
3. **State → TE Schedule**
   - `task.compute_dag.apply_steps_from_state(state)` → TE `Schedule`과 인자 텐서.
4. **Schedule → TIR (GPU pipeline)**
   - `ScheduleToModule(sch, tensors, "main", {})` → IRModule.
   - 그 다음 **GPU pass pipeline** 적용 (InjectPrefetch, StorageFlatten, NarrowDataType, Simplify, VectorizeLoop, InjectVirtualThread, StorageRewrite, Simplify).
5. **GPU 제약 검사**
   - 각 PrimFunc에 대해 C++ `tir.analysis.verify_gpu_code(func, GPU_HW)` 호출.
   - 여기서 thread_per_block, shared/local memory, vthread, max_vector_bytes 등이 HW 한계를 넘으면 **invalid**.
6. **Valid 여부**
   - 위 단계에서 예외 없이 통과하고 `verify_gpu_code`가 전부 True면 **valid**.
   - 또는 feature 추출 `get_per_store_features_from_states([state], task)`를 쓰면, 결과가 all-zero이면 **invalid**, 아니면 **valid** (내부적으로 같은 pipeline + 검사 사용).

이 경로는 **매번** record → 복원 → Schedule → TIR → 검사이므로 비용이 크다. “이 파라미터 조합이 되는지”만 판단할 때 사용한다.

---

### 경로 B: 제약 “식”으로 빠르게 확인 (Provenance 공식)

파라미터를 **바꿀 때마다** full lowering 없이, “vthread / thread 쪽은 이렇게 나올 것이다”를 **식**으로 계산할 때 사용한다.

1. **입력**: 동일하게 `record`에서 복원한 `task`, `state`, 그리고 같은 `record`(step 구조·SP/PR 정보).
2. **State에 bound 채우기**
   - `state_with_bound = task.compute_dag.infer_bound_from_state(state)`.
   - 각 stage의 각 iterator에 `range.extent`가 채워진 State가 나온다.
3. **어떤 iter가 “제약에 기여하는지” 분류**
   - State의 `stages[].iters[]`를 돌면서 `annotation` 확인:
     - 4 → vthread, 6/8/10 → threadIdx.x/y/z → 이 iter들의 extent가 “제약에 기여”.
   - 이걸로 `thread_per_block_iters`, `vthread_extent_iters` 리스트와 `base_extents`(기본 extent)를 만든다.
4. **Probing: “이 iter의 extent는 어떤 SP 파라미터에 의존하는가?”**
   - record에서 SP step만 골라, 각 `(step_idx, length_pos)`마다:
     - 그 자리만 2(또는 3)로 바꾼 record로 state를 다시 만들고 `infer_bound_from_state` 호출.
     - bound가 **바뀐** (stage_idx, iter_idx)를 기록 → “이 iter의 extent는 이 (step_idx, length_pos)에 의존한다”.
   - 결과: `iter_deps[(stage_idx, iter_idx)]` = 그 iter의 extent에 기여하는 `(step_idx, length_pos)` 리스트.
5. **제약식 (symbolic) 정의**
   - 각 (stage_idx, iter_idx)에 대해:
     - `extent(iter) = Π record["i"][1][1][step_idx][4][length_pos]` for `(step_idx, length_pos) in iter_deps[iter]` (deps 없으면 `base_extents` 사용).
   - **vthread 공식**: vthread_bound_iters 각 iter의 위 extent를 곱한 값.
   - **thread 공식**: thread_bound_iters 각 iter의 위 extent를 곱한 값 (다중 커널이면 “전체 스테이지 곱”이라 커널별 값과는 다를 수 있음).
6. **다른 record(파라미터)에 대해 “식만” 계산**
   - `eval_vthread_formula(formulas, record_new)` → 새 record의 SP length만 넣어서 vthread 값 계산.
   - `eval_thread_formula(formulas, record_new)` → 같은 식으로 thread 쪽 곱 계산.
   - **제약 확인**: vthread ∈ [1, 8], thread_prod ≤ 1024 등으로 **빠르게** 필터링 (다만 shared memory 등은 이 식에 없음).

정리하면, **경로 A**가 “실제로 제약 만족 여부를 확인하는 과정”이고, **경로 B**는 “그 중 vthread/thread 부분을 공식으로 빼서, 파라미터만 바꿔 넣어 빠르게 확인하는 과정”이다.

---

### 경로 A와 B를 함께 쓸 때 (검증 스크립트)

- **제약식 확인**: ResNet-18 등으로 task/record 로드 → `build_provenance_formulas`로 경로 B의 식 구축.
- **랜덤 파라미터 검증**:
  - 같은 task, base record에서 SP/PR만 랜덤으로 바꾼 `record_mut` 여러 개 생성.
  - 각 `record_mut`에 대해:
    - **경로 B**: `eval_vthread_formula(formulas, record_mut)`, `eval_thread_formula(...)` → formula 기준 vthread/thread 값.
    - **경로 A**: `record_mut`로 task/state 복원 → Schedule → TIR (GPU pipeline) → `parse_tir_constraints`로 TIR의 max_vthread_s, per-kernel thread_per_block 추출 + `verify_gpu_code`로 valid 여부 판정.
  - 비교: formula vthread ↔ TIR max_vthread_s 일치율, “formula로 valid 예측” vs “실제 valid” 일치율.

이렇게 하면 “제약 확인이 어떤 식으로 진행되는지”를 두 경로로 나눠서 재현·검증할 수 있다.

---

## Shared memory 제약: 완벽한 추정 가능 여부

### TIR에서의 계산 방식

`verify_gpu_code.cc`는 shared scope를 가진 **모든** `Allocate` 노드를 순회하며 다음을 합산한다:

- `size = ConstantAllocationSize() * dtype.bytes() * dtype.lanes()`
- `ConstantAllocationSize()` = Allocate의 `extents` 배열 원소 전부가 `IntImm`일 때의 **곱**, 아니면 0

즉, **최종 TIR에서의 shared 사용량** = (StorageRewrite 적용 **이후** 남는 각 Allocate의 상수 크기 × elem_bytes)의 **합**이다.

### 스케줄만으로 알 수 있는 것

1. **버퍼별 크기 (공식)**  
   각 CacheRead/CacheWrite(scope="shared") stage에 대해:
   - 그 stage가 계산되는 루프 영역(compute_at으로 정해지는 scope)의 iterator들 extents의 **곱**이 곧 해당 shared 버퍼의 원시 shape이다.
   - Iterator extent는 step 시뮬레이터로 SP length들의 곱으로 **연역적으로** 표현 가능하다.
   - 따라서 **버퍼 하나당** shared 크기 = (해당 scope iterator들의 extent 곱) × elem_bytes → **완벽히 식으로 구할 수 있다.**

2. **상한 (보수적 총량)**  
   merge를 전혀 하지 않았다고 가정하면:
   - `shared_upper_bound = Σ (각 shared stage의 버퍼 크기)`
   - 이 값도 스케줄 + step 시뮬레이터만으로 **완벽히** 계산 가능하다.
   - 검증 시: `shared_upper_bound ≤ max_shared_memory_per_block` 이면 **무조건 valid** (false negative 없음).

### 완벽한 "정확값"이 어려운 이유

**StorageRewrite**가 같은 블록 내 여러 shared 버퍼를 **한 개의 할당으로 재사용**할 수 있다.

- `FindAlloc` 조건: 같은 `attach_scope`, 같은 `storage_scope`, 크기가 `match_range`(16배 범위) 안이고, flat 등 조건 만족 시 기존 StorageEntry 재사용.
- 재사용 시 해당 entry의 크기는 `max(기존 크기, 새 버퍼 크기)`로 갱신된다.
- 따라서 **실제 총 사용량** = (merge 이후 남는 각 StorageEntry의 크기)의 합이며, 이는 "버퍼별 크기 단순 합"보다 **작거나 같다**.

`attach_scope`는 TIR의 `AttrStmt`(thread_extent) / `For` 노드 포인터로 결정되므로, **스케줄만 보고는 어떤 버퍼들이 같은 scope에 묶일지**를 정확히 알 수 없다. 그래서:

- **스케줄만으로 "merge까지 반영한 정확한 총량"을 구하는 것은 불가능에 가깝다.**
- 대신 **상한**은 위처럼 완벽히 구할 수 있다.

### 정리

| 목표 | 스케줄만으로 가능? | 비고 |
|------|-------------------|------|
| **버퍼별 shared 크기 (식)** | ✅ 가능 | scope iterator extent 곱 × elem_bytes, step 시뮬레이터로 연역 도출 |
| **총량 상한 (merge 없음)** | ✅ 가능 | 모든 shared stage 버퍼 크기 합, 안전한 검증에 사용 |
| **총량 정확값 (merge 반영)** | ❌ 사실상 불가 | StorageRewrite의 attach_scope·재사용 로직이 TIR 단계에서만 결정됨 |

**실무 권장**:  
Shared 제약은 **상한 공식**만 써도 "완벽하게 안전하게" 걸러낼 수 있다. 상한이 한계 이하면 항상 valid이고, 상한이 한계를 넘으면 그때만 (경로 A로) 실제 lowering으로 재확인하면 된다.  
즉, "완벽하게 알아낸다"를 **상한을 정확히 구한다**로 두면 가능하고, **merge까지 반영한 정확한 총량**은 스케줄 단계만으로는 불가능하다.

### felix는 merge 반영 정확 총량을 알 수 있는가?

**알 수 있다.**  
felix는 **symbolic TIR**을 만든 뒤 **동일한 GPU pass pipeline**(StorageRewrite 포함)을 그대로 적용한다.  
`GenerateCodeForState(..., symbolic=true)` → `ScheduleToModule(..., var_context)` → **StorageRewrite** 등이 적용된 **이후**의 `Stmt`를 `GetConstraints(stmt)`에 넘긴다.  
그래서 `GPUConstraintsMaker`가 방문하는 `Allocate` 노드는 **이미 merge가 반영된** TIR 위의 노드들이다. merge된 경우 개수는 줄어들고, 남은 각 Allocate의 extent는 (필요 시) `max(·,·)` 같은 PrimExpr일 수 있다.  
`CountBufferSize_`는 **ConstantAllocationSize()가 아니라** extent들의 **PrimExpr 곱**을 쓰므로, merge 후의 크기(PrimExpr)도 그대로 합산한다.  
따라서 felix가 수집하는 `shared_memory_per_block` 제약은 **merge까지 반영한 정확한 총량**에 대한 식이다.  
반면 **context-only**(step 시뮬레이터만 쓰고 TIR은 concrete로 두는) 접근은 StorageRewrite를 실행하지 않으므로, merge 결과를 알 수 없고 **상한만** 완벽히 구할 수 있다.

---

## felix vs tvm-ansor: Symbolic GPU 제약 추출 비교

felix(`~/work/felix`)에서는 **그래프(스케줄)부터 symbolic하게 만든 뒤** 한 번의 TIR 순회로 GPU 제약을 **PrimExpr 부등식**으로 뽑는다. 여기서는 그 방식을 요약하고 tvm-ansor와 비교한다.

### felix 쪽 방식

1. **Symbolic TIR 생성**
   - **VarContext**: State에 붙는 컨텍스트로, Split의 `lengths`를 **상수가 아니라 SizeVar(E0, E1, …)** 로 채운다.  
     예: `InitFillTileSize`에서 `GetVarContext().GetSplitVars(extent, n_lengths, true)`로 각 Split step의 length를 SizeVar 배열로 대체.
   - **ScheduleToModule(sch, args, name, binds, var_context)**  
     `driver_api.cc`에서 `te::InferBound(sch, vcontext)`를 호출해, bound 추론 시 **vcontext**를 넘긴다.  
     `te/schedule/bound.cc` → `PassDownDomain(..., vcontext)`, `message_passing.cc`에서 `vcontext->GetSplitSizes(...)` 등으로 split size가 **symbolic(PrimExpr)** 로 유지된다.
   - 결과적으로 **TIR Stmt**의 루프 extent·Allocate extent가 **PrimExpr(Var/곱/합)** 이고, 구체 정수로 치환되지 않은 상태로 GPU pass pipeline에 들어간다.

2. **제약 추출 (한 번의 순회)**
   - **GPUConstraintsMaker** (`src/felix/constraints.cc`):  
     - `AllocateNode`: `CountBufferSize_(op->extents, ...)` 에서 **ConstantAllocationSize() 대신** `extents`의 **PrimExpr 곱**으로 크기 계산.  
       `local_memory_per_block`, `shared_memory_per_block`에 PrimExpr로 누적.
     - `BufferRealizeNode`: 마찬가지로 bounds의 extent로 PrimExpr 곱해 크기 누적.
     - `AttrStmtNode` (thread_extent / virtual_thread): `thread_per_block *= extent` (PrimExpr).
     - 커널 탈출 시 `AddConstraint_(lhs, rhs)` → `lhs <= Integer(rhs)` 를 **PrimExpr** 로 만들어 `constraints` 벡터에 push.  
       (상수로 단순화되면 0이면 errors에 넣고, 아니면 constraint로 두지 않음.)
   - 즉, **probe 없이** Stmt 한 번 순회로 `std::vector<PrimExpr> constraints` (예: `E0*E1*E2 <= 1024`, `shared_byte_expr <= 49152`) 를 얻는다.

3. **FeaturePack과의 연동**
   - `GetFeaturePack(stmt, context, hw_params, ...)` 안에서 `GetConstraints(stmt, hw_params)` 호출.
   - 나온 제약들을 `con_0`, `con_1`, … 같은 이름으로 FeaturePack에 넣고, 이후 `RunSizeSubstitution(sizes)`로 구체값 치환.

정리하면: **스케줄 → (VarContext로 symbolic split) → ScheduleToModule(..., vcontext) → symbolic TIR → GPU passes → GetConstraints(Stmt) → symbolic 제약(PrimExpr)** 이 한 번에 이뤄지고, probing은 쓰지 않는다.

### tvm-ansor 쪽 방식

1. **TIR은 항상 concrete**
   - `driver.schedule_to_module(sch, tensors, "main", {})` 에 **VarContext 인자가 없음**.  
     Split step의 `lengths`는 JSON record에서 오는 **구체 정수**만 사용되고, InferBound도 symbolic context 없이 호출된다.
   - 따라서 TIR의 extent·alloc size는 전부 **상수**이고, “그래프부터 symbolic” 경로가 없다.

2. **제약을 얻는 방법**
   - **경로 A**: 그대로 lowering → `verify_gpu_code` / `parse_tir_constraints`로 **값**만 확인. 제약 “식”은 얻지 않음.
   - **경로 B (Provenance)**:  
     `infer_bound_from_state(state)` 로 각 iterator의 extent를 **구체값**으로 채운 뒤, **probe**로 “어떤 (step_idx, length_pos)가 이 extent에 기여하는지” 역추적해,  
     `extent(iter) = Π record[step_idx][length_pos]` 형태의 **식**을 세워 thread/vthread 제약식을 만든다.  
     즉, **연역적 step 시뮬레이션**이 아니라 **귀납적 probing**으로 식을 복원한다.
   - (앞으로 계획한) **Step 시뮬레이터**: JSON step을 재생해서 iterator별 extent를 **파라미터 변수에 대한 식**으로 추적하면, probing 없이 연역적으로 식을 만들 수 있다. 다만 이건 **Python**에서 스케줄 구조만 보고 하는 것이고, TIR은 여전히 concrete lowering으로만 생성된다.

3. **Shared memory**
   - felix: symbolic TIR이므로 `CountBufferSize_`에서 extent 곱이 이미 PrimExpr → **shared 역시 symbolic 제약**으로 나온다.
   - tvm-ansor: TIR이 concrete라서 “식”을 쓰려면 step 시뮬레이터로 scope iterator extent를 **식**으로 구해야 하고, 상한 공식은 그렇게 구한 뒤 합으로 두는 방식이 필요하다.

### 비교 요약

| 항목 | felix | tvm-ansor |
|------|--------|-----------|
| **그래프/TIR이 symbolic인가** | ✅ State의 VarContext로 Split을 SizeVar로 두고, ScheduleToModule에 vcontext 전달 → TIR이 처음부터 symbolic | ❌ ScheduleToModule에 VarContext 없음 → TIR은 항상 concrete |
| **제약 형태** | `PrimExpr` 부등식 (예: `E0*E1 <= 1024`) 한 번에 수집 | Provenance: probing으로 “어떤 SP가 어떤 extent에 기여” 복원 후 식 구성. (계획) Step 시뮬: Python에서 연역적 식 구성 |
| **Shared memory 제약** | ✅ 동일한 한 번의 순회에서 PrimExpr로 합산 → symbolic 제약 | 상한만 step 시뮬로 연역 가능; merge 반영 정확값은 불가 |
| **Merge 반영 정확 총량** | ✅ 가능. GetConstraints가 받는 Stmt가 **StorageRewrite 이후** TIR이라 merge된 Allocate만 순회·합산 → 정확한 총량 식 | ❌ 불가. context-only는 StorageRewrite를 타지 않아 merge 결과를 모름 → 상한만 가능 |
| **Probe 사용** | ❌ 사용 안 함 | ✅ Provenance는 최소 probe 사용; Step 시뮬은 목표가 probe 제거 |
| **구현 위치** | C++ (InferBound+ScheduleToModule+GetConstraints) | Python (provenance, step 시뮬), 검증만 C++ verify_gpu_code |

### tvm-ansor에서 felix 스타일을 쓰려면

- **ScheduleToModule에 VarContext 전달**이 필요하다.  
  즉, C++ 쪽에서:
  1. `driver_api.cc`의 `ScheduleToModule`에 `arith::VarContextNode* vcontext` 인자 추가.
  2. `te::InferBound(sch, vcontext)` 호출로 bound를 symbolic 유지.
  3. `te/schedule/message_passing.cc` 등에서 split 시 `vcontext->GetSplitSizes(...)` 사용해 extent를 SizeVar/식으로 유지.
- 그러면 **한 번의 lowering**으로 symbolic TIR이 나오고, **GetConstraints 같은 Stmt 방문자**를 하나 두면 thread/shared/local/vthread 제약을 전부 **PrimExpr 부등식**으로 뽑을 수 있다.  
  다만 현재 tvm-ansor 저장소에는 VarContext·SizeVar·GetSplitSizes 등 felix 쪽 arith/te 확장이 없으므로, 그 방식을 쓰려면 해당 C++ 확장을 이식하거나, 대안으로 **Python step 시뮬레이터**로 연역적 식을 만드는 길을 계속 가는 것이 현실적이다.

---

## Context-only 접근 (felix와 다른 스타일)

felix는 **symbolic TIR**을 쓰고, 여기서는 그 방식을 쓰지 않는다. 대신 **lowering은 기존처럼 concrete로 두고, lowering 과정에서 “context”만 유지**하는 방식으로 제약식을 얻는 것을 목표로 한다.

### 설계 원칙

- **Symbolic TIR 사용 안 함**: ScheduleToModule에 VarContext를 넣어 TIR 전체를 SizeVar/PrimExpr로 두지 않는다. 기존처럼 구체 정수로 lowering한 TIR만 만든다.
- **아이디어는 felix에서 가져옴**: “제약 = extent/alloc 크기에 대한 부등식”이고, 그 좌변을 **파라미터(SP length 등)에 대한 식**으로 두면, 파라미터만 바꿔 넣어서 검사할 수 있다. 즉 “어떤 파라미터가 어떤 extent/크기에 기여하는지”를 알면 된다.
- **구현 차이**: 그 “기여 관계”를 **symbolic TIR을 만드는 것이 아니라**, **lowering 과정에서 context만 유지**해서 얻는다.

### “Context”가 의미하는 것

- **Context** = 파라미터(예: `record`의 SP `lengths`, PR 값)와 lowering 결과(루프 extent, shared/local alloc 크기, thread/vthread 바인딩) 사이의 **대응 관계**.
- 구체적으로:
  - 각 (stage_idx, iter_idx) iterator에 대해: `extent = Π record[step_idx][length_pos]` 인 **(step_idx, length_pos)** 집합.
  - 각 iterator의 **annotation** (vthread / threadIdx.x 등).
  - Shared stage별: 그 stage를 스코핑하는 iterator들 → 해당 shared 버퍼 크기 = 이 iterator들의 extent 곱 × elem_bytes.
- 이 context만 있으면, record를 바꿔 넣었을 때 **lowering을 다시 하지 않고** 제약식 좌변(thread_prod, vthread_prod, shared_upper 등)을 **식으로 계산**할 수 있다.

### 유지 시점: “lowering 과정에서”

- **의도**: TIR을 symbolic으로 만들지 않으면서, **lowering이 일어나는 흐름 안에서** 위 대응 관계를 채워 넣는 구조를 두자는 것.
- 두 가지 구현 경로가 있다.

#### (A) Python에서 step만 재생해 context 구성 (lowering은 그대로)

- **Lowering**: 기존과 동일. `apply_steps_from_state(state)` → `ScheduleToModule` → GPU passes. 모두 **concrete** 값으로 동작. TIR은 그대로 concrete.
- **Context**: 같은 `record`(transform_steps)를 **Python에서만** 재생(step 시뮬레이터). SP/FU/AN/RE/… 규칙에 따라 “현재 iterator 목록과 각 iterator의 extent = 어떤 SP length들의 곱인지”를 **연역적으로** 계산. annotation은 step에서 읽어서 채움.
- **관계**: “lowering 과정”과 “context 구성”이 **같은 입력(record/state)** 을 쓰고, **같은 step 순서**를 반영한다는 의미에서 “lowering 과정에서 context를 유지”한다고 볼 수 있다. 실제 C++ lowering 코드는 수정하지 않음.

#### (B) C++ lowering에 context 콜백/수집기 추가 (선택)

- **Lowering**: 그대로 concrete 값으로 계산하되, **추가로** “이 extent를 만든 split step / 이 alloc 크기를 만든 scope” 같은 정보를 **콜백이나 수집기**로 넘긴다.
- 예: `InferBound` / `ScheduleOps` 내부에서 extent를 정할 때, “이 범위는 split step i의 length j, k, … 의 곱”이라고 기록. ScheduleToModule 시그니처에 `ProvenanceCollector*` 같은 옵션 인자를 두고, 널이 아니면 채움.
- **Context**: 한 번의 concrete lowering이 끝나면, TIR + **context**(extent/alloc ↔ parameter indices 또는 식)를 얻음. Symbolic TIR은 전혀 도입하지 않음.

### 권장 방향 (tvm-ansor 현실 기준)

- **1단계**: **(A) Python step 시뮬레이터로 context만 유지**
  - Step 시뮬레이터가 record의 step을 차례로 적용하면서, 각 iterator의 `extent_expr`(어떤 (step_idx, length_pos)의 곱인지)와 `annotation`을 유지.
  - 한 번의 시뮬레이션으로 “thread_bound iters”, “vthread_bound iters”, “shared stage별 scope iters”에 대한 context를 얻고, 여기서 thread_prod / vthread_prod / shared_upper 식을 도출.
  - Lowering은 기존 그대로 concrete; Provenance는 이 context로 대체(probe 제거 가능).
- **2단계(선택)**: 나중에 C++ 쪽을 손댈 여유가 있으면 **(B)** 를 넣어서, 한 번의 lowering으로 TIR + context를 동시에 얻고, Python 시뮬레이터와의 일치를 검증하는 데 쓸 수 있다.

### felix와의 대비

| 항목 | felix | tvm-ansor (context-only) |
|------|--------|---------------------------|
| TIR 형태 | Symbolic (SizeVar 등) | Concrete (기존과 동일) |
| 제약 식 획득 | TIR 한 번 순회로 PrimExpr 수집 | Context(파라미터↔extent/크기) 유지 → Python에서 식 계산 |
| Context 유지 위치 | TIR 안에 식이 들어감 (자체가 context) | Lowering과 동일 입력으로 Python에서 context 구성; 또는 C++에서 수집기로 기록 |
| C++ 변경 | InferBound/ScheduleToModule에 VarContext 전달 | 없음(A) 또는 수집기만 추가(B) |

이렇게 하면 **felix의 “제약을 식으로 갖는다”는 아이디어**는 가져오되, **symbolic TIR 대신 lowering 과정에서 context만 유지**하는 다른 스타일로 구현할 수 있다.

### Context-only로 shared memory 제약식을 “완벽하게” 만들 수 있는가?

- **버퍼별 shared 크기 식**  
  **완벽하게 가능하다.**  
  Step 시뮬레이터 context로 “이 shared stage를 스코핑하는 iterator들”과 “각 iterator의 extent = Π (해당 SP length)”를 알 수 있다.  
  → `shared_size_i = (스코핑 iterator들의 extent 곱) × elem_bytes` 를 **파라미터에 대한 식**으로 쓸 수 있다.  
  (ComputeAt 관계만 step에서 정확히 반영하면, 어떤 iterator가 해당 stage를 스코핑하는지 연역적으로 정해진다.)

- **총 shared 상한 (merge 없는 경우)**  
  **완벽하게 가능하다.**  
  `shared_upper_bound = Σ shared_size_i` (모든 shared stage에 대해).  
  이 값이 `max_shared_memory_per_block` 이하면 **항상** valid이므로, 제약 검사용으로 “완벽한” 상한이다.

- **Merge 반영한 “정확한 총량”**  
  **Context만으로는 완벽하게 만들 수 없다.**  
  `VerifyGPUCode`가 보는 값은 **StorageRewrite 이후** TIR의 Allocate 합이다. StorageRewrite가 같은 `attach_scope`·같은 scope·비슷한 크기인 버퍼를 **한 allocation으로 재사용**하면, 실제 총량은 “버퍼별 합”이 아니라 “merge된 allocation 크기들의 합”이 된다.  
  그 merge 여부는 TIR의 `AttrStmt`(thread_extent)/`For` 노드 포인터(`attach_scope`)로 결정되므로, **step 시뮬레이터만으로는** “어떤 버퍼가 어떤 scope에 묶일지”를 TIR과 1:1로 복제하기 어렵다.  
  따라서 context-only 접근만으로는 “merge까지 반영한 정확한 총 shared” 식을 **완벽하게** 만들 수 없다.

**정리**:  
Context-only로 **shared에 대한 제약식**은  
- **버퍼별 식**과 **총량 상한**은 완벽하게 만들 수 있고,  
- **merge 반영 정확 총량**은 완벽하게는 불가하다.  
검증 목적이라면 **상한 식만으로도 충분**하다(상한 ≤ limit이면 무조건 valid).

### Merge 반영 정확한 총량을 알 수 있으려면?

merge까지 반영한 **정확한** 총 shared 사용량을 식으로 쓰려면, 아래 중 하나가 필요하다.

1. **Felix 스타일: Symbolic TIR + 실제 StorageRewrite**
   - ScheduleToModule에 **VarContext**를 넘겨 TIR의 extent/alloc 크기를 처음부터 **PrimExpr(SizeVar 곱)** 로 만든다.
   - 그 TIR에 **실제 GPU pass pipeline(StorageRewrite 포함)** 을 적용한다.
   - StorageRewrite가 symbolic TIR 위에서 merge를 수행한 **결과** TIR을 GetConstraints 같은 방문자로 한 번 순회해, 남은 각 Allocate의 크기(PrimExpr)를 합산하면 **merge 반영 정확 총량**이 식으로 나온다.
   - 필요: C++ 쪽에 VarContext·InferBound(sch, vcontext)·GetSplitSizes 등 felix 스타일 확장 이식.

2. **StorageRewrite 과정에서 merge 구조만 기록 (추천)**
   - **아이디어**: merge를 Python에서 재현하지 않고, **한 번의 concrete lowering**을 돌릴 때 StorageRewrite가 **어떤 버퍼들을 어떤 그룹으로 묶었는지**만 기록해 둔다. 크기 식은 step 시뮬레이터로 이미 구해 두므로, “어느 그룹이 묶였는지”만 알면 `총량 = Σ_그룹 max(그 그룹에 속한 stage들의 크기 식)` 으로 **정확한 총량 식**을 만들 수 있다.
   - **구현 요지**  
     - C++: StorageRewrite(또는 그 직후의 TIR 순회)에서, **최종적으로 남는 각 shared Allocate**에 대해 “이 할당이 merge한 원본 버퍼들”을 기록한다.  
       예: `StorageEntry::allocs` 에 이미 merge된 `AllocateNode*` 목록이 있으므로, pass 종료 시점에 “shared scope인 StorageEntry별로, 그 entry에 묶인 alloc 목록”을 내보내면 된다.  
       각 원본 Allocate는 TE lowering 시점의 버퍼 이름/태그와 대응되므로, **버퍼 이름 → (stage_id 등)** 매핑을 한 번 정해 두면, “그룹 = [stage_id a, b, …]” 형태로 쓸 수 있다.
     - Python: 동일 스케치에 대해 **한 번** lowering을 돌리고(아무 concrete record로도 가능), C++에서 내려준 **merge 맵**을 받는다.  
       예: `merge_groups = [[stage_2_shared, stage_5_shared], [stage_7_shared]]`  
       Step 시뮬레이터로 각 shared stage의 **크기 식** `size_i(record)` 를 이미 가지고 있으므로,  
       `total_shared(record) = Σ_그룹 max{ size_i(record) : i ∈ 그룹 }`  
       로 merge 반영 정확 총량을 **식**으로 계산할 수 있다.
   - **전제**: 같은 스케치(같은 step 구조)라면 StorageRewrite가 어떤 버퍼를 어떤 그룹으로 묶는지는 **파라미터(SP length 값)에 따라 바뀌지 않고** “스케줄 구조”로만 결정된다고 보는 것이 일반적이다.  
     (같은 attach_scope·같은 scope·같은 elem_type이면 merge 후보가 되고, 이는 step 순서/구조에 의해 정해짐.)  
     따라서 **한 번 기록한 merge 맵**을 해당 스케치의 다른 record에도 그대로 써도 된다.
   - **정리**: Symbolic TIR 없이, **merge 구조만 한 번 기록**하면 context-only + step 시뮬만으로 **merge 반영 정확 총량 식**을 만들 수 있다. C++ 쪽은 “merge 결과 내보내기”만 추가하면 된다.

3. **Python에서 StorageRewrite merge 규칙 시뮬레이션**
   - Step 시뮬레이터로 각 shared stage의 **(크기 식, scope 키)** 를 구한다.  
     **scope 키** = “이 stage가 속한 attach_scope에 해당하는 식별자”.  
     TIR의 `attach_scope`는 “이 할당이 어느 루프(AttrStmt(thread_extent) / For) 아래에 붙는가”로 정해지므로, 스케줄 쪽에서는 “어느 thread-bound 루프(또는 block 레벨) 아래에 이 stage가 compute_at 되는가”로 대응시키면 된다.  
     예: (target_stage_id, target_iter_id) 또는 “커널 내 최상위 thread scope” 같은 정규화된 키.
   - StorageRewrite와 동일한 규칙을 적용: **같은 scope 키, 같은 storage_scope**인 버퍼들을 그룹으로 묶고, 각 그룹에서 **max(크기 식)** 로 merge 크기를 두고, **총량 = Σ 그룹별 merge 크기** 로 둔다.  
     (실제 C++는 match_range·elem_type 등으로 reuse 여부를 더 제한하므로, 완전히 똑같이 하려면 그 조건도 옮겨야 한다.)
   - **한계**: TIR의 attach_scope는 **Stmt 포인터** 기준이라, 루프 순서/구조가 바뀌면 scope가 달라질 수 있다. Python에서 “같은 scope”를 스케줄만 보고 1:1로 정의하기 어렵고, 잘못 묶으면 과대/과소 추정이 될 수 있다.  
     보수적으로 쓰려면 “같은 scope로 확실한 것만 merge”하고, 나머지는 합으로 두거나, 아예 merge 시뮬을 생략하고 **상한만** 쓰는 편이 안전하다.

4. **상한만 사용 (merge 시도 안 함)**
   - `shared_upper_bound = Σ (각 shared stage 크기 식)` 으로 두고, **shared_upper_bound ≤ max_shared_memory_per_block** 만 검사한다.
   - merge를 반영하지 않으므로 **정확한 총량**은 모르지만, 상한이 limit 이하이면 항상 valid이므로 **false negative는 없다**.  
     다만 merge로 실제 총량이 더 작은 경우에도 “상한 > limit”이면 invalid로 잘라버릴 수 있어 **false positive**는 있을 수 있다.
   - 구현이 단순하고, context-only + step 시뮬만으로 가능하다.

**실무 권장**:  
- **정확한 총량 식을 원하고 C++ 소량 수정이 가능하다**면 → **(2) StorageRewrite에서 merge 구조만 기록**하는 방식이 가장 현실적이다. Symbolic TIR 없이, 한 번의 lowering으로 “어떤 버퍼가 어떤 그룹으로 묶였는지”만 넘겨 받으면 된다.  
- **정확한 총량이 꼭 필요하지 않다**면 → **(4) 상한만** 쓰는 것이 가장 단순하고 안전하다.  
- **정확한 총량이 필요하고 C++ 대규모 확장이 가능하다**면 → **(1) felix 스타일**.  
- C++을 건드리지 않으려면 **(3) merge 시뮬**을 시도할 수 있으나, scope 키 정의와 규칙이 TIR과 맞는지 검증이 필요하다.

---

## Phase 1: Python TIR 제약값 추출기 ✅ 완료

### 목표
`feature.cc`의 GPU pass pipeline을 Python에서 재현하여, TIR 문자열에서 per-kernel GPU 제약값을 직접 추출

### 구현된 함수들

#### `lower_with_gpu_passes(task, state)`
```python
_s2m = tvm.get_global_func("driver.schedule_to_module")
GPU_PASSES = tvm.transform.Sequential([
    tir.transform.InjectPrefetch(),
    tir.transform.StorageFlatten(64, False),
    tir.transform.NarrowDataType(32),
    tir.transform.Simplify(),
    tir.transform.VectorizeLoop(True),
    tir.transform.InjectVirtualThread(),
    tir.transform.StorageRewrite(),
    tir.transform.Simplify(),
])

def lower_with_gpu_passes(task, state):
    sch, tensors = task.compute_dag.apply_steps_from_state(state)
    mod = _s2m(sch, tensors, "main", {})
    return GPU_PASSES(mod)
```

#### `parse_tir_constraints(tir_str)`
TIR 문자열에서 regex로 per-kernel 제약값 추출:
- `T.launch_thread("threadIdx.x", N)` → thread extent
- `T.launch_thread("vthread*", N)` → vthread
- `T.allocate([M], "dtype", "shared")` → shared memory bytes
- `T.allocate([M], "dtype", "local")` → local memory bytes
- `for vthread_s in T.grid(...)` → vthread.s extent (ForNode에서)

#### `check_hw_limits(kernels, max_vts, hw)`
추출된 per-kernel 제약값을 HW 한계와 비교

#### `verify_state_exact(task, state)`
ScheduleToModule + GPU passes + C++ `tir.analysis.verify_gpu_code` 직접 호출

### 발견된 문제 및 수정

1. **초기 false positive rate: 8.9%**
   - 원인 1: `parse_tir_constraints`에서 `vthread.s` (ForNode에서의 vthread extent) 미처리
   - 원인 2: `max_vector_bytes` 미체크
   
2. **vthread.s 파싱 버그**: `T.grid(a, b, c)`에서 `vthread_s`가 마지막 변수가 아닌 경우 extent를 잘못 추출
   - 수정: `vars_.index('vthread_s')`로 정확한 위치의 extent 추출

3. **`tvm.lower` vs `ScheduleToModule`**: 두 방식이 유사한 결과를 내지만, 정확한 `feature.cc` 재현을 위해 `ScheduleToModule` 사용

### 검증 결과

수정 후 기존 valid/invalid 레코드 + mutated 레코드에 대해:
- **정확도: 100%** (`get_per_store_features_from_states`와 완전 일치)
- False positive: 0%, False negative: 0%

### 파일 위치
- 노트북: `gallery/test_constraint_3.ipynb` (cell-constraint)
- C++ 참조: `src/auto_scheduler/feature.cc` (line 1401-1422)
- C++ VerifyGPUCode: `src/tir/analysis/verify_gpu_code.cc`

---

## Phase 2: Probing 기반 공식 도출 🔄 부분 완료

### 목표
각 task에 대해 N+1회 full-pipeline lowering으로 split factor와 제약값 간의 symbolic 공식 도출

### 구현: `ConstraintSystem` 클래스

```python
class ConstraintSystem:
    def __init__(self, record, hw=GPU_HW)
    def build(self)                    # N+1 probing → formula derivation
    def formula_check(self, dim_vals)  # Fast constraint check using formulas
    def generate(self, rng)            # Random valid record generation
```

### Probing 방법론

1. **Base probe**: 모든 SP lengths = [1, ..., 1] → 각 커널별 base 제약값
2. **Per-dimension probe**: i번째 SP의 j번째 length를 `smallest_factor(extent)`로 설정 → 변화 관찰
3. 각 제약값에 대해 `ratio = probe_value / base_value` 계산
4. `ratio != 1`이면 해당 dimension이 해당 제약에 기여

### 공식 도출 (Linear Interpolation)

```
g(t) = 1 + (ratio - 1) * (t - 1) / max(probe_value - 1, 1)

constraint_value = base * Π_i g_i(dim_val_i)
```

각 커널별로 `shared_bytes`, `local_bytes`, `thread_per_block` 공식과 global `vthread.s` 공식 도출.

### 빌드 결과 (24 tasks)

```
T 0: ext=[4, 4, 49, 256] est=5632B    (winograd)
T 1: ext=[1, 56, 56, 64] est=3840B    (conv2d)
T 2: ext=[1, 28, 28, 128] est=14436B  (conv2d)
T10: ext=[1, 112, 112, 64] est=48636B (conv2d, 거의 shared mem limit)
...
24 systems built.
```

### 검증 결과: 공식 기반 생성

**Phase 2 공식만으로 생성 + feature extraction 검증:**
- **총 validity: 51/210 = 24.3%** ❌ (목표 95%에 미달)

Task별 상세:
```
T 0: 3/10 valid    T 7: 4/10 valid    T14: 1/10 valid    T21: 5/10 valid
T 1: 0/10 valid    T 8: 1/10 valid    T15: skip          T22: 4/10 valid
T 2: 2/10 valid    T 9: 3/10 valid    T16: 1/10 valid    T23: 1/10 valid
T 3: 0/10 valid    T10: 3/10 valid    T17: 2/10 valid
T 4: skip          T11: 1/10 valid    T18: 5/10 valid
T 5: 3/10 valid    T12: skip          T19: 4/10 valid
T 6: 6/10 valid    T13: 1/10 valid    T20: 1/10 valid
```

### 실패 원인 분석

1. **Linear interpolation의 한계**: 실제 제약은 비선형(superlinear) 관계일 수 있음
   - shared memory: `base * t1 * t2` 형태지만, StorageRewrite 이후 allocation 크기가 비선형적으로 변화
   - vthread.s: FU+SP+AN 스텝 조합으로 결정되어 단순 선형 모델로 포착 불가

2. **다수 커널 존재**: Winograd 계열 task는 4개 커널(data_pack, bgemm, inverse, output)이 각각 독립적 제약을 가짐

3. **공식이 과소추정(underestimate)**: 실제 제약값보다 낮게 추정하여 invalid 파라미터를 통과시킴

---

## Phase 3: 랜덤 파라미터 생성 🔄 진행 중

### 접근법 A: 공식 기반 프리필터 (실패)

`ConstraintSystem.generate()`:
1. 무작위로 dimension 값 채우기 (약수만 사용)
2. `formula_check(dim_vals)`로 프리필터
3. 통과 시 record 생성

결과: 24.3% validity → **실패**

### 접근법 B: Direct Lowering Verify (Brute Force)

`TaskGenerator.generate()`:
1. 무작위로 split factor 생성 (약수 기반)
2. `inject_params`로 record 생성
3. `record_to_task_and_state`로 state 복원
4. `verify_via_lowering(task, state)`로 정확 검증 (ScheduleToModule + GPU passes + VerifyGPUCode)
5. 실패 시 최대 100회 재시도

### Direct Lowering Verify 결과

```
============================================================
Phase 3: Direct lowering verify (N=10 per task)
============================================================
T 0: gen=3/10  feat=3(100%)   [50s]
T 1: gen=1/10  feat=1(100%)   [62s]
T 2: gen=4/10  feat=4(100%)   [77s]
T 3: gen=10/10 feat=10(100%)  [78s]
T 4: gen=10/10 feat=10(100%)  [78s]
T 5: gen=0/10  feat=0(0%)     [139s]   ← 완전 실패
T 6: gen=10/10 feat=10(100%)  [142s]
T 7: gen=1/10  feat=1(100%)   [204s]
T 8: gen=6/10  feat=6(100%)   [250s]
T 9: gen=0/10  feat=0(0%)     [318s]   ← 완전 실패
T10: gen=1/10  feat=1(100%)   [339s]
T11: gen=4/10  feat=4(100%)   [349s]
T12: gen=10/10 feat=10(100%)  [349s]
T13: gen=0/10  feat=0(0%)     [417s]   ← 완전 실패
T14: gen=7/10  feat=7(100%)   [427s]
T15: gen=10/10 feat=10(100%)  [427s]
T16: gen=3/10  feat=3(100%)   [481s]
T17: gen=0/10  feat=0(0%)     [597s]   ← 완전 실패
T18: gen=10/10 feat=10(100%)  [602s]
T19: gen=1/10  feat=1(100%)   [661s]
T20: gen=8/10  feat=8(100%)   [668s]
T21: gen=2/10  feat=2(100%)   [776s]
T22: gen=1/10  feat=1(100%)   [881s]
T23: gen=3/10  feat=3(100%)   [931s]

=== TOTAL (931s) ===
Generated: 105/240 (43.75%)
Feat pass: 105/105 (100.0%)
```

### 핵심 관찰

1. **정확도는 100%**: 생성에 성공한 105개 모두 feature extraction 통과
2. **생성 성공률은 43.75%**: 100회 시도로도 valid 파라미터를 못 찾는 task 존재
3. **완전 실패 task 4개**: T5, T9, T13, T17 (100회 × 10샘플 = 1000회 시도에서 모두 실패)
4. **매우 느림**: 931초 (15.5분) for 24 tasks × 10 samples

### 병목 분석

1. **`record_to_task_and_state(rec)`**: 매 시도마다 tempfile 생성/삭제 + JSON 파싱 + state 복원 → ~3-5ms per call
2. **`verify_via_lowering`**: ScheduleToModule + 8 GPU passes + VerifyGPUCode → ~3-6ms per call
3. **조합 폭발**: 100 attempts × 10 samples × 24 tasks → 최대 24,000회 lowering
4. **프리필터 미적용**: 현재 코드에서 formula 프리필터를 사용하지 않아 모든 시도가 full lowering을 수행

---

## 현재 코드 구조

### 파일 목록

| 파일 | 설명 |
|------|------|
| `gallery/test_constraint_3.ipynb` | 메인 개발 노트북 (6 cells) |
| `gallery/test_constraint_2.ipynb` | 이전 시도 (구 ConstraintSystem, InferBound 기반) |
| `src/auto_scheduler/feature.cc` | TVM feature extraction (GPU pass pipeline 참조) |
| `src/tir/analysis/verify_gpu_code.cc` | TVM GPU code verification (C++) |

### test_constraint_3.ipynb 셀 구조

| Cell | ID | 내용 |
|------|----|------|
| 0 | cell-setup | TVM 환경 설정, 상수 정의 |
| 1 | cell-load | ResNet-18 task/record 로드 |
| 2 | cell-utils | 유틸리티 함수 (`get_divisors`, `record_to_task_and_state`, `inject_params`, `validate_via_feature`) |
| 3 | cell-constraint | Phase 1 구현 (`lower_with_gpu_passes`, `parse_tir_constraints`, `extract_gpu_constraints`, `check_hw_limits`, `verify_state_exact`) + Phase 2 `ConstraintSystem` 클래스 |
| 4 | cell-build | ConstraintSystem 빌드 (24 tasks) + 진단 출력 |
| 5 | cell-test | Phase 2/3 검증: 랜덤 생성 + feature extraction 테스트 (결과: 24.3%) |

### 유틸리티 함수

```python
def get_divisors(n)           # n의 모든 약수 반환
def record_to_task_and_state(record)  # JSON → (SearchTask, State) (tempfile 기반)
def inject_params(record, assignments)  # record에 파라미터 삽입
def validate_via_feature(record)  # feature extraction으로 validity 체크
```

---

## 시도 이력 (시간순)

### 1차 시도: test_constraint_2.ipynb (InferBound 기반)
- InferBound로 shared memory 크기 추정
- Spatial split의 l2 product를 threadIdx.x로 직접 매핑
- **실패**: thread extent는 FU+SP+AN 조합으로 결정되어 단순 매핑 불가

### 2차 시도: Phase 1 - TIR 추출기
- feature.cc의 pass pipeline을 Python으로 재현
- TIR 문자열 regex 파싱
- **성공**: 100% 정확도 달성 (VerifyGPUCode와 동일)

### 3차 시도: Phase 2 - Probing + 공식 도출
- N+1회 lowering으로 linear interpolation 공식 도출
- **부분 실패**: 공식의 과소추정으로 24.3% validity

### 4차 시도: Phase 3 - Brute Force Direct Verify
- 무작위 생성 + 매번 full lowering으로 검증
- **정확도 100% but 생성률 43.75%**, 속도 매우 느림 (15.5분)

---

## 미해결 과제 및 다음 단계

### 즉시 해결 필요

1. **생성 성공률 향상** (43.75% → 95%+)
   - 공식 프리필터와 direct verify 결합 (하이브리드)
   - 제약이 타이트한 task(T5, T9, T13, T17)에 대한 특별 처리
   - 더 스마트한 파라미터 샘플링 (현재는 완전 무작위)

2. **속도 최적화**
   - `record_to_task_and_state`의 tempfile I/O 제거
   - Task 객체 캐싱 (현재 `TaskGenerator`에서 부분적으로 구현)
   - State 복원만 별도로 수행하는 경량 함수 필요

3. **공식 정확도 개선**
   - 비선형 보간 (quadratic, piecewise)
   - Multi-point probing (2점 이상)
   - 커널별 independent constraint modeling

### 계획된 Phase 4: C++ API 강화

`src/tir/analysis/verify_gpu_code.cc`에 `GetGPUConstraintValues` 추가:
- GPUCodeVerifier에 per-kernel 값 수집 로직 추가
- Python wrapper: `tvm.tir.analysis.get_gpu_constraint_values(func)`
- TIR 문자열 파싱을 C++ API로 교체 → 더 정확하고 빠른 제약값 추출

### Provenance 기반 정확 제약식 (추가 구현)

**파일**: `gallery/constraint_provenance.py`

Lowering을 상수로 한 번만 돌리되, **State(infer_bound)**에서 각 iterator의 extent가 어떤 split 파라미터 `(step_idx, length_pos)`에 의존하는지 **probing**으로 추적해, thread/vthread에 대한 **정확한 symbolic 제약식**을 복원한다.

- **입력**: `task`, `state`, `record`, `record_to_task_and_state(record)`
- **방법**:
  1. `state_with_bound = task.compute_dag.infer_bound_from_state(state)` 로 bound 채움.
  2. State의 `stages[].iters[]`에서 `annotation`이 4(vthread), 6(threadIdx.x), 8(threadIdx.y), 10(threadIdx.z)인 iter를 thread-bound로 사용.
  3. 각 SP step의 각 length 위치 `(step_idx, length_pos)`에 대해, 해당 length만 2(또는 3)로 바꾼 record로 state를 다시 만들고 infer_bound → 어떤 (stage_idx, iter_idx)의 extent가 바뀌었는지로 **의존성** 수집.
  4. `iter_deps[(stage_idx, iter_idx)]` = 그 iter의 extent에 기여하는 `(step_idx, length_pos)` 리스트.
  5. **식**: `extent(iter) = Π length[step_idx][length_pos]` for `(step_idx, length_pos) in iter_deps[iter]` (또는 deps가 비면 base_extent 사용).

- **API**:
  - `build_provenance_formulas(task, state, record, record_to_task_and_state)` → `formulas` dict.
  - `eval_thread_formula(formulas, record)` → thread 관련 extent들의 곱 (단, **다중 커널**이면 커널별로 나누어 해석 필요).
  - `eval_vthread_formula(formulas, record)` → vthread extent 곱.

- **한계**: 다중 커널(Winograd 등)에서는 `thread_per_block`이 커널마다 다르므로, 현재 `eval_thread_formula`는 “모든 thread-bound iter extent의 곱”을 주며, 검증 시에는 TIR lowering으로 커널별 값을 쓰거나, formula의 `iter_deps`/`base_extents`로 커널별 식을 따로 구성해야 함.

### 근본적 설계 질문

- **Symbolic 접근**: TIR lowering 과정에서 split factor를 symbolic variable로 남겨 제약식을 직접 도출할 수 있는가?
- **Constraint propagation**: GPU pass pipeline의 각 pass가 split factor에 대한 constraint를 어떻게 변환하는지 정적으로 분석 가능한가?
- **다수 커널 문제**: Winograd 등 multi-kernel task에서 각 커널의 제약을 독립적으로 모델링하는 것이 충분한가?

---

## 환경 정보

- TVM: custom fork (tvm-ansor), build-release
- Python: `/root/work/venv/bin/python` (3.8.10)
- GPU target: `cuda -keys=cuda,gpu -arch=sm_86`
- 테스트 네트워크: ResNet-18, batch_size=1, NHWC layout
- 테스트 데이터: `gallery/logs_json/resnet_18/resnet_18-B1.json` (2408 records, 24 unique tasks)
