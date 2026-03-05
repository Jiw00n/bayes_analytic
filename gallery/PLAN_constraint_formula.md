# 제약식 알고리즘 구현 계획

## 1. 목표

JSON record의 **스케줄 구조(step 순서)** 를 한 번 분석해서, SP length들에 대한 **제약 부등식**을 만든다.  
이후 파라미터(SP length, PR value)만 바꿔 넣으면 **lowering 없이** 제약 만족 여부를 즉시 판정할 수 있어야 한다.

만들어야 하는 제약식:

| 제약 | 좌변 (파라미터의 식) | 우변 (한계) |
|------|----------------------|-------------|
| thread_per_block | `Π (thread-bound iterator extents)` per kernel | ≤ 1024 |
| thread_x / thread_y / thread_z | 각 축별 `Π (해당 annotation iterator extents)` per kernel | ≤ max_thread_x/y/z (보통 1024,1024,64) |
| thread 축 동일 extent | 같은 이름(threadIdx.x 등)이 여러 번 나오면 extent 동일 | must equal |
| vthread | `Π (vthread-bound iterator extents)` per kernel | ≤ 8 |
| shared_memory | `Σ_그룹 max(그룹 내 shared 버퍼 크기 식)` per kernel (merge 반영 정확값) | ≤ 49152 bytes |
| local_memory | `Σ_그룹 max(그룹 내 local 버퍼 크기 식)` per kernel (merge 반영 정확값) | ≤ max_local (2^31-1) |
| max_vector_bytes | vectorize된 iterator의 extent × dtype.bytes | ≤ 16 |
| innermost_split | 각 SP의 **마지막** length `lengths[-1]` (루프 순서상 가장 안쪽 iterator의 extent) | ≤ 64 |
| divisibility | `Π lengths` for each SP | `extent`의 약수 |

---

## 2. 알아야 하는 배경 지식

### 2.1 JSON record 구조

하나의 record = `{"i": [task_key, task_config], "r": [...]}` 이고,  
`task_config[1]` (또는 `"i"[1][1]`)에 **step 배열**이 들어 있다.

각 step 종류와 JSON 형식:

| 약어 | Step | JSON | 비고 |
|------|------|------|------|
| SP | SplitStep | `["SP", stage_id, iter_id, extent, [lengths], inner_to_outer]` | `inner_to_outer`=1이면 lengths 순서가 안→밖 |
| AN | AnnotationStep | `["AN", stage_id, iter_id, annotation]` | annotation: 0=None,1=Unroll,2=Vectorize,3=Parallel,4=VThread,5=BlockX,6=ThreadX,7=BlockY,8=ThreadY,9=BlockZ,10=ThreadZ |
| FU | FuseStep | `["FU", stage_id, [fused_ids]]` | 연속 iterator들을 하나로 합침 |
| RE | ReorderStep | `["RE", stage_id, [after_ids]]` | iterator 순서 변경 (extent/annotation 불변) |
| FSP | FollowSplitStep | `["FSP", stage_id, iter_id, src_step_id, n_split]` | 다른 SP의 lengths를 참조 |
| FFSP | FollowFusedSplitStep | `["FFSP", stage_id, iter_id, [src_step_ids], level, factor_or_nparts]` | 여러 SP의 특정 level lengths의 곱을 참조 |
| PR | PragmaStep | `["PR", stage_id, iter_id, "auto_unroll_max_step$VALUE"]` | unroll 파라미터 |
| CI | ComputeInlineStep | `["CI", stage_id]` | stage를 인라인 |
| CR | ComputeRootStep | `["CR", stage_id]` | stage를 루트로 |
| CA | ComputeAtStep | `["CA", stage_id, target_stage_id, target_iter_id]` | stage를 특정 위치에 붙임 |
| CHR | CacheReadStep | `["CHR", stage_id, scope_name, [reader_stage_ids]]` | shared 등으로 캐시 읽기 stage 추가 |
| CHW | CacheWriteStep | `["CHW", stage_id, scope_name]` | 캐시 쓰기 stage 추가 |
| SA | StorageAlignStep | `["SA", stage_id, iter_id, factor, offset]` | 메모리 정렬 |

### 2.2 Split이 iterator를 만드는 방식

`ApplySplitToState` (C++ `transform_step.cc` line 835):

- **입력**: `stage_id`, `iter_id`, `lengths = [l0, l1, ..., l_{n-1}]`, `inner_to_outer`
- **출력**: 원래 1개 iterator → `n+1`개 iterator로 교체
- `inner_to_outer = True` (일반적) 일 때:
  - lengths를 **뒤에서 앞으로** 처리: `l_{n-1}, l_{n-2}, ..., l_0` 순서로 split
  - 최종 iterator 순서 (밖→안): `[outermost, ..., l0, l1, ..., l_{n-1}]`
  - 각 iterator의 **extent**:
    - `iter[k]` = `lengths[k]` (k = 0, ..., n-1)
    - `iter[outermost]` = `ceildiv(original_extent, Π lengths)`
  - 즉, **lengths에 적힌 값이 그대로 해당 iterator의 extent가 된다.**

### 2.3 Fuse가 iterator에 미치는 영향

`FuseStepNode::ApplyToState` (line 506):

- **입력**: `stage_id`, `fused_ids = [i0, i1, ...]` (연속해야 함)
- **출력**: 이 iterator들을 하나로 합침
  - 새 iterator의 `extent = Π (원래 iterator들의 extent)`
  - annotation은 kNone으로 리셋

### 2.4 Annotation이 iterator에 미치는 영향

`AnnotationStepNode::ApplyToState` (line 375):

- **입력**: `stage_id`, `iter_id`, `annotation` (정수)
- **출력**: 해당 iterator의 annotation만 변경 (extent 불변)
- annotation 값:
  - `4` = VThread → vthread-bound
  - `6` = ThreadX, `8` = ThreadY, `10` = ThreadZ → thread-bound
  - `5` = BlockX, `7` = BlockY, `9` = BlockZ → block-bound (제약에 미포함)

### 2.5 FollowSplit / FollowFusedSplit

- **FSP** (`FollowSplitStep`): `src_step_id`가 가리키는 SP의 lengths의 앞 `n_split-1`개를 그대로 가져와서 split. lengths를 직접 갖지 않고 참조.
- **FFSP** (`FollowFusedSplitStep`): `src_step_ids`가 가리키는 여러 SP들의 `lengths[level]` 값을 모두 곱해서 하나의 split factor로 사용.

이 두 step은 **자체적인 파라미터가 없고**, 참조하는 SP의 lengths에 의존한다.

### 2.6 ComputeAt과 shared stage의 관계

- `CA` step으로 shared stage가 특정 (target_stage, target_iter)에 붙는다.
- 그 stage의 **shared 버퍼 크기** = 해당 scope 내 iterator들의 extent 곱 × elem_bytes.
- "scope 내 iterator들"은 compute_at 대상 iterator **안쪽**에 있는 것들이다.

### 2.7 GPU 하드웨어 제약 (sm_86 기준)

```
max_threads_per_block = 1024
max_thread_x = 1024
max_thread_y = 1024
max_thread_z = 64
max_shared_memory_per_block = 49152 bytes
max_local_memory_per_block = 2^31 - 1 bytes
max_vthread = 8
max_vector_bytes = 16 bytes
max_innermost_split_factor = 64
warp_size = 32
```

### 2.8 VerifyGPUCode가 확인하는 것 (per kernel)

- `thread_per_block` = `Π(threadIdx.x/y/z extent × vthread extent)` ≤ 1024
- `threadIdx.x` extent ≤ max_thread_x, `threadIdx.y` ≤ max_thread_y, `threadIdx.z` ≤ max_thread_z
- 같은 이름(예: threadIdx.x)이 두 번 이상 나오면 **extent가 동일**해야 함
- `shared_memory_per_block` = `Σ(shared Allocate 크기)` ≤ 49152
- `local_memory_per_block` = `Σ(local Allocate 크기)` ≤ 2^31-1
- vthread extent ≤ 8
- `max_vector_bytes`: Allocate/Load/Store 등에서 dtype.lanes() × dtype.bytes() ≤ 16
- ForNode `vthread.s` extent ≤ 8

---

## 3. Step 시뮬레이터: 핵심 알고리즘

### 3.1 개요

JSON record의 step 배열을 **순서대로 재생**하면서,  
각 (stage_id, iter_idx)에 대해 다음을 유지한다:

```python
class IterInfo:
    extent_expr: dict  # {(step_idx, length_pos): 1} → extent = Π SP[step_idx].lengths[length_pos]
    extent_const: int  # extent_expr에 해당하지 않는 상수 부분 (ceildiv 결과 등)
    annotation: int    # 0=None, 4=VThread, 6=ThreadX, ...
```

**핵심**: 각 iterator의 extent를 "어떤 SP의 어떤 length의 곱"으로 표현한다.

### 3.2 초기 상태

- ComputeDAG의 각 stage에 대해, 초기 iterator 목록과 extent를 가져온다.
- 초기 extent는 **상수** (스케줄 적용 전이므로).
- 이 값은 `task.compute_dag.infer_bound_from_state(initial_state)` 등으로 얻거나, record의 SP step에 적힌 `extent` 필드에서 읽는다.

### 3.3 각 Step별 시뮬레이션 규칙

#### SP (SplitStep)

```
입력: stage_id, iter_id, extent, lengths=[l0,...,l_{n-1}], inner_to_outer
결과: iter_id 위치의 iterator를 n+1개로 교체

inner_to_outer = True 일 때 (일반적):
  for k in 0..n-1:
    new_iter[k].extent_expr = {(step_idx, k): 1}   # 즉, SP[step_idx].lengths[k]
    new_iter[k].annotation = kNone

  new_iter[outermost].extent_expr = {}
  new_iter[outermost].extent_const = ceildiv(original_extent, Π lengths)
  → 이 outermost의 extent는 파라미터에 **의존하지만**, 직접 제약에 걸리지 않는 경우가 많다.
  → 정확히는: extent / (l0 * l1 * ... * l_{n-1}) 인데, divisibility 제약으로 정확히 나눠떨어져야 함.
```

**중요**: lengths에 적힌 값이 **그대로** 해당 iterator의 extent가 되므로, `(step_idx, length_pos)` 하나가 iterator extent 하나에 대응된다.

#### FU (FuseStep)

```
입력: stage_id, fused_ids=[i0, i1, ...]
결과: 여러 iterator를 하나로 합침
  new_iter.extent_expr = merge(all fused iters의 extent_expr)  # 곱
  new_iter.annotation = kNone
```

#### AN (AnnotationStep)

```
입력: stage_id, iter_id, annotation
결과: 해당 iterator의 annotation만 변경
  iter.annotation = annotation
```

#### RE (ReorderStep)

```
입력: stage_id, after_ids
결과: iterator 순서만 변경 (extent, annotation 불변)
```

#### FSP (FollowSplitStep)

```
입력: stage_id, iter_id, src_step_id, n_split
결과: src_step_id가 가리키는 SP의 lengths[0..n_split-2]를 그대로 사용.
  → 새 iterator들의 extent_expr은 참조하는 SP와 동일한 (step_idx, length_pos) 쌍.
```

#### FFSP (FollowFusedSplitStep)

```
입력: stage_id, iter_id, src_step_ids, level, factor_or_nparts
결과: 여러 SP의 lengths[level] 값의 곱으로 하나의 split factor 결정.
  → 새 iterator의 extent_expr = {(src_step_ids[0], level): 1, (src_step_ids[1], level): 1, ...}
    즉, 여러 SP length의 곱.
```

#### CA (ComputeAtStep)

```
입력: stage_id, target_stage_id, target_iter_id
결과: stage의 compute_at 위치 기록.
  → shared 버퍼 크기 계산에 필요.
```

#### CHR / CHW (CacheRead / CacheWrite)

```
결과: 새 stage 추가. stage_id가 shift된다.
  → CHR의 scope_name이 "shared"이면, 이 stage가 shared stage.
  → 해당 shared stage의 elem_dtype/bytes를 기록.
```

#### CI / CR (ComputeInline / ComputeRoot)

```
CI: stage를 인라인 → 해당 stage의 iterator는 제약 계산에서 무시.
CR: stage를 루트로 → compute_at 위치가 최상위.
```

### 3.4 시뮬레이션 결과에서 제약식 도출

시뮬레이션이 끝나면 각 stage의 각 iterator에 대해 `(extent_expr, annotation)`을 알게 된다.

#### thread_per_block 식 (per kernel)

```python
thread_prod = 1
for each (stage, iter) where annotation in {4, 6, 8, 10}:  # vthread, threadIdx.x/y/z
    thread_prod *= eval_extent(iter.extent_expr, record)
# 제약: thread_prod <= 1024
```

#### vthread 식 (per kernel)

```python
vthread_prod = 1
for each (stage, iter) where annotation == 4:  # vthread
    vthread_prod *= eval_extent(iter.extent_expr, record)
# 제약: vthread_prod <= 8
```

#### shared_memory 식 (per kernel, merge 반영 정확값)

```python
for each shared_stage:
    # compute_at 위치 아래 iterator들의 extent 곱 × elem_bytes
    shared_size[stage] = Π(scope 내 iterator extent) × elem_bytes

# StorageRewrite merge 구조(한 번의 lowering에서 기록)를 반영:
# merge_groups = [[stage_a, stage_b], [stage_c], ...]  (C++에서 한 번 기록)
total_shared = Σ_그룹 max(shared_size[i] for i in 그룹)
# 제약: total_shared <= 49152
```

**merge_groups 획득**: concrete lowering 한 번을 돌려 StorageRewrite가 어떤 shared 버퍼들을 어떤 그룹으로 묶었는지 C++에서 기록해 받는다. 다만 **concrete**일 때는 merge가 파라미터에 따라 달라질 수 있으므로(§6), 해당 record와 같은 파라미터일 때만 정확하다. (Felix는 심볼릭 TIR이라 merge 구조가 스케줄 구조로만 정해져 한 번만 돌려도 됨, §7 참고.)

#### thread_x / thread_y / thread_z (축별 제약, per kernel)

VerifyGPUCode는 thread_per_block뿐 아니라 **각 축별** extent도 검사한다 (max_thread_x, max_thread_y, max_thread_z).

```python
for axis, ann in [("x", 6), ("y", 8), ("z", 10)]:  # ThreadX, ThreadY, ThreadZ
    extents = [eval_extent(iter.extent_expr, record) for (stage, iter) in thread_iters if iter.annotation == ann]
    if not extents:
        continue
    # 동일 축이 여러 개면 반드시 extent가 같아야 함 (VerifyGPUCode)
    if len(set(extents)) > 1:
        return INVALID  # "Extent of threadIdx.x does not match the bound"
    axis_extent = extents[0]
    # 제약: axis_extent <= max_thread_x / max_thread_y / max_thread_z
```

시뮬레이터에서 축별로 iterator를 묶어 두고, **같은 annotation인 iterator들의 extent가 전부 같은지** 먼저 검사한 뒤, 그 공통값이 해당 축 한계 이하인지 확인하면 된다.

#### local_memory 식 (per kernel, merge 반영 정확값)

shared와 동일한 방식. scope가 **local**인 stage에 대해:

```python
for each local_stage:
    local_size[stage] = Π(scope 내 iterator extent) × elem_bytes

# merge_groups_local (C++에서 한 번 기록)
total_local = Σ_그룹 max(local_size[i] for i in 그룹)
# 제약: total_local <= max_local_memory_per_block (2^31-1)
```

StorageRewrite는 local scope 버퍼에 대해서도 shared와 같은 merge를 수행한다. merge_groups를 shared와 함께 한 번 기록하면 된다.

#### max_vector_bytes (벡터화 제약)

VerifyGPUCode는 **dtype.lanes() × dtype.bytes()** 가 16 이하인지 Allocate/Load/Store/Cast 등에서 검사한다.  
벡터화는 **AN(annotation=2, Vectorize)** 가 붙은 iterator의 **extent**가 lane 수가 된다.

```python
for each (stage, iter) where annotation == 2:  # Vectorize
    lane_extent = eval_extent(iter.extent_expr, record)
    # 해당 stage의 연산 dtype (float32면 4, int8이면 1 등)
    elem_bytes = stage_dtype_bytes(stage)
    # 제약: lane_extent * elem_bytes <= 16
    if lane_extent * elem_bytes > 16:
        return INVALID
```

stage별 기본 dtype은 ComputeDAG/초기 op에서 가져오면 된다.  
PR(unroll)은 vector bytes 검사와 무관하고, **Vectorize annotation이 붙은 iterator의 extent**만 이 제약에 걸린다.

#### warp_size (선택적 하한/정렬)

일부 타깃에서는 thread_per_block이 **warp_size(32)의 배수**여야 하거나, 최소 32 이상이어야 한다.  
공식으로 넣으려면:

```python
thread_prod = ...  # 위와 동일
# 제약: thread_prod >= 32 (또는 thread_prod % 32 == 0)
```

필요한 경우에만 추가하면 된다.

#### innermost_split (max_innermost_split_factor ≤ 64)

**확인**: `ApplySplitToState` (transform_step.cc)에서 **inner_to_outer=True**일 때:

1. for (i=0..n-1): `l = lengths[n-1-i]`, name = `.n`, `.n-1`, …, `.1` 순으로 push  
   → outs = [iter(lengths[n-1]), iter(lengths[n-2]), …, iter(lengths[0])]
2. outermost (name `.0`) push  
   → outs = […, outermost]
3. **reverse**  
   → 최종 순서 = [outermost, iter(lengths[0]), …, iter(lengths[n-1])]

따라서 **루프 순서상 가장 안쪽** = 배열의 **마지막** iterator → extent = **lengths[n-1] = lengths.back()**.  
sketch_policy_rules.cc 1106행: `innermost_factor = ps->lengths.back()`, 1195행: `ICHECK_LE(GetIntImm(new_lengths.back()), max_innermost_split_factor)` 로 동일하게 **마지막** 요소에 제약을 건다.

```python
for each SP step (step_idx) in steps:
    lengths = steps[step_idx].lengths
    if not lengths:
        continue
    innermost_val = lengths[-1]  # 마지막 length = 가장 안쪽 iterator의 extent
    # 제약: innermost_val <= 64
```

#### divisibility (각 SP의 Π lengths | extent)

각 Split step에서 `Π lengths` 가 원래 iterator의 **extent의 약수**여야 한다 (ceildiv가 정수여야 함).

```python
for each SP step with extent E and lengths [l0, l1, ...]:
    product = l0 * l1 * ... * l_{n-1}
    # 제약: E % product == 0
```

#### eval_extent 함수

```python
def eval_extent(extent_expr, record):
    result = extent_expr.const_part
    for (step_idx, length_pos), power in extent_expr.items():
        val = record_steps[step_idx].lengths[length_pos]
        result *= val ** power
    return result
```

---

## 4. 구현 계획

### Phase A: Step 시뮬레이터 (Python)

**파일**: `gallery/constraint_formula.py`

1. **`parse_steps(record)`**: JSON record에서 step 배열을 파싱해 Python 객체 리스트로 변환.

2. **`simulate_steps(steps, dag_info)`**: step 리스트를 순서대로 재생하면서 각 (stage, iter)의 `(extent_expr, annotation)`을 유지.
   - `dag_info`: 초기 stage/iterator 정보 (stage 이름, 초기 extent, op_type 등). `task.compute_dag`에서 한 번 추출.
   - 반환: `SimContext` (각 stage의 iterator 목록 + extent_expr + annotation + compute_at 정보 + shared stage 목록)

3. **`extract_constraints(sim_context, hw_params)`**: SimContext에서 제약식 도출.
   - `thread_formula`: per-kernel thread_prod 식 (≤ 1024)
   - `thread_axis_formulas`: per-kernel thread_x / thread_y / thread_z 식 및 동일 축 extent 일치 검사
   - `vthread_formula`: per-kernel vthread_prod 식 (≤ 8)
   - `shared_formula`: per-kernel shared 정확 식 (merge 반영)
   - `local_formula`: per-kernel local 정확 식 (merge 반영)
   - `vector_bytes_checks`: Vectorize annotation 붙은 iterator의 extent × elem_bytes ≤ 16
   - `innermost_constraints`: 각 SP의 **lengths[-1]** (마지막 length) ≤ 64. inner_to_outer=True일 때 최종 iterator 순서가 [outermost, lengths[0], …, lengths[n-1]]이므로 가장 안쪽 = lengths[n-1].
   - `divisibility_constraints`: 각 SP의 Π lengths | extent
   - `warp_constraint`: (선택) thread_prod ≥ 32 또는 32의 배수

4. **`check_constraints(constraints, record)`**: 주어진 record의 파라미터로 제약식을 평가해 valid 여부 반환.

### Phase B: 검증

1. ResNet-18의 모든 task에 대해:
   - base record로 `simulate_steps` → `extract_constraints`
   - 같은 base record로 `lower_with_gpu_passes` + `verify_gpu_code` → 실제 valid 여부
   - 두 결과 비교 → 100% 일치 확인

2. 랜덤 파라미터 변형:
   - base record에서 SP lengths를 랜덤으로 바꾼 record 생성 (약수만 사용)
   - `check_constraints` vs `verify_gpu_code` 비교
   - false negative / false positive 없이 100% 일치 확인

### Phase C: Shared memory merge 구조 기록

1. `storage_rewrite.cc`에서, pass 종료 시점에 shared scope인 `StorageEntry`별로 `allocs` 목록을 Python으로 내보내는 API 추가.
2. Python에서 이 merge 맵을 받아 `shared_formula`에 반영.
3. 검증: merge 반영 shared 총량 vs TIR의 실제 shared 총량 일치 확인.

**주의**: StorageRewrite의 merge 결과는 **파라미터에 따라 달라질 수 있음** (아래 "StorageRewrite merge와 파라미터" 참고). 따라서 "한 번 기록한 merge 구조"는 해당 record와 **같은 파라미터**일 때만 정확하다. 파라미터가 바뀌면 merge 구조가 바뀌었는지 재검증하거나, 파라미터별로 merge 구조를 다시 기록하는 전략이 필요할 수 있다.

### Phase D: 파라미터 생성기 연동

1. `check_constraints`를 프리필터로 사용:
   - 랜덤으로 SP lengths 생성 (약수 기반)
   - `check_constraints`로 즉시 검증 → 통과한 것만 record에 주입
2. 기존 "brute-force lowering verify" 대비 속도 비교.

---

## 5. 구현 순서 요약

```
A1. parse_steps 구현 + 단위 테스트
A2. simulate_steps 구현 (SP, FU, AN, RE만 먼저)
A3. simulate_steps 확장 (FSP, FFSP, CA, CHR, CHW, CI, CR)
A4. extract_constraints 구현 (thread/vthread/shared/local/축별/vector_bytes/innermost/divisibility/warp)
A5. check_constraints 구현 (위 제약식 전부 평가)
B1. base record로 검증 (thread/vthread/shared/local 등 전체 제약)
B2. 랜덤 변형으로 검증
C1. C++ merge 구조 내보내기
C2. merge 반영 shared 식 검증
D1. 파라미터 생성기 프리필터 연동
```

---

## 6. 주의사항

- **StorageRewrite merge와 파라미터**: shared memory merge 시 **조건문이 있고**, **같은 step 시퀀스라도 파라미터가 다르면 merge 결과가 달라질 수 있다**. 이유: (1) `FindAlloc`에서 `const_nbits = ConstantAllocationSize() * elem_bits`로 크기를 쓰며, extent가 모두 상수일 때만 재사용 경로를 탄다. (2) 상수 경로에서는 `const_free_map_`을 **const_nbits 기준**으로 검색하고, **match_range(16배)** 안에 있는 기존 StorageEntry와 merge한다. 따라서 파라미터(SP length)가 바뀌면 할당 크기가 바뀌어, **어떤 버퍼와 merge할지**가 달라지거나 새로 할당될 수 있다. (3) `is_small_array`(const_nbits ≤ 32 등), `is_known_size`(심볼릭이면 0) 등도 분기 조건이다. 결론: "스케치당 한 번만 merge 구조 기록"은 파라미터가 고정이거나, 실험으로 merge 구조가 변하지 않음을 확인한 경우에만 안전하다.
- **stage_id shift**: CHR/CHW step이 실행되면 새 stage가 삽입되고, 이후 step의 stage_id가 바뀐다. 시뮬레이터에서 이 shift를 정확히 반영해야 한다.
- **ceildiv**: Split의 outermost extent = `ceildiv(원래 extent, Π lengths)`. divisibility 제약을 걸면 정확히 나눠떨어지므로 `extent // Π lengths`와 같다.
- **multi-kernel**: Winograd 등은 여러 커널을 생성한다. 제약은 **per-kernel**이므로, 어떤 stage가 어느 커널에 속하는지 구분이 필요하다. 단순 접근: inline 안 된 compute_root stage 단위로 커널 구분.
- **FSP/FFSP의 extent_expr**: 이 step들은 자체 lengths를 갖지 않고 다른 SP를 참조하므로, 식에서 참조 SP의 (step_idx, length_pos)를 사용해야 한다.
- **IteratorAnnotation 값**: AN step의 annotation 필드는 C++ enum `IteratorAnnotation`의 정수 값이다 (0=None, 1=Unroll, 2=Vectorize, 3=Parallel, 4=VThread, 5=BlockX, 6=ThreadX, 7=BlockY, 8=ThreadY, 9=BlockZ, 10=ThreadZ).
- **동일 축 extent**: TIR에서 같은 thread 이름(threadIdx.x 등)이 두 번 나오면 VerifyGPUCode는 두 extent가 **같아야** 한다고 검사한다. 시뮬레이터에서 같은 annotation을 가진 iterator가 여러 개면, 각각의 extent 식을 평가한 값이 동일한지 반드시 확인해야 한다.
- **max_vector_bytes**: Vectorize(annotation=2)가 붙은 iterator의 extent가 lane 수가 되며, 그 stage의 원소 dtype bytes와 곱해져 16 이하여야 한다. PR(unroll)은 이 제약과 무관하다.

---

## 7. 참고: Felix에서 shared memory / merge 처리 방식

Felix는 **심볼릭 TIR**을 쓰기 때문에, merge 후 shared memory 총량을 **파라미터에 대한 식**으로 정확히 얻는다.

### 흐름

1. **ScheduleToModule에 VarContext 전달**  
   `driver_api.cc`: `ScheduleToModule(sch, args, name, binds, vcontext)` → `InferBound(sch, vcontext)`로 루프 extent·할당 크기가 **SizeVar 등 PrimExpr**로 남음.

2. **GPU 패스(StorageRewrite 포함)를 심볼릭 TIR에 적용**  
   `utils.cc`의 `GetGPUCodeGenPasses()`: InjectPrefetch → StorageFlatten → … → **StorageRewrite** → Simplify.  
   `GenerateCodeForState`에서 이 패스를 적용한 뒤 나온 `stmt`를 사용.

3. **StorageRewrite 동작 (심볼릭일 때)**  
   - extent가 심볼릭이면 `ConstantAllocationSize()` = 0 → `is_known_size = false`.
   - **상수 경로**(const_nbits 기준 match_range 검색)는 타지 않고, **심볼릭 경로**만 사용: `sym_free_list_`에서 `attach_scope`·`scope`·`elem_type`이 같은 기존 entry와 **라운드로빈**으로 merge.
   - 따라서 **같은 스케줄 구조**면 merge **구조**(누가 누구와 묶이는지)는 고정되고, 파라미터는 **merge 대상 선택**이 아니라 **합쳐진 크기 식**에만 반영됨.

4. **제약 추출**  
   `feat_transform.cc`: `GetFeaturePack` → `GetConstraints(stmt, hw_params)`.  
   `constraints.cc`의 **GPUConstraintsMaker**가 이 **merge까지 반영된 TIR**을 방문해서, 각 `AllocateNode`의 `extents`·`dtype`으로 `alloc_size`(PrimExpr)를 만들고, shared/local별로 **합산**한다.  
   → 합이 **파라미터(SizeVar)에 대한 PrimExpr**가 되고, 이걸 제약식으로 씀.

### tvm-ansor와의 차이

| 항목 | Felix | tvm-ansor (현재) |
|------|--------|-------------------|
| TIR | 심볼릭 (VarContext → SizeVar 등) | 구체값 (concrete) |
| StorageRewrite | 심볼릭 TIR에 적용 → merge 후에도 크기가 식 | Concrete TIR에 적용 → 크기 상수, merge 구조가 const_nbits에 따라 달라질 수 있음 |
| merge 구조 | 스케줄 구조만으로 결정 (attach_scope/scope/type) | 파라미터에 따라 const_nbits·match_range로 달라질 수 있음 |
| shared 총량 | TIR 순회만으로 **정확한 식** (merge 반영) | 한 번 기록한 merge 구조 + Python 식으로 근사하거나, 파라미터 바뀔 때마다 재기록 필요 |

정리하면, Felix는 **lowering 단계부터 심볼릭**이라 StorageRewrite가 “크기 상수”에 의존하지 않고, merge 구조가 고정되고 합산 결과가 곧바로 파라미터에 대한 정확한 제약식이 된다.
