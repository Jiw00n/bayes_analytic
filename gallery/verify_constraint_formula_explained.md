# `verify_constraint_formula.py` 실행 과정 상세 설명

```
python verify_constraint_formula.py --max-tasks 24 --random-trials 500 --seed 0
```

---

## 0. 전체 목적

Ansor의 tuning record(JSON)에 담긴 split/unroll 파라미터를 바꿨을 때,
그 설정이 GPU 제약조건(shared memory, thread 수 등)을 만족하는지를
**실제 lowering 없이 수식만으로 빠르게 판별**할 수 있는지 검증한다.

- **Formula check** (`constraint_formula.py`): lowering 없이 수식 대입 → ~43μs
- **Ground truth** (`lower_with_gpu_passes` + `verify_gpu_module`): 실제 TIR lowering + VerifyGPUCode → ~38ms
- 두 결과가 일치하면 formula가 정확한 것이고, 불일치(mismatch)가 있으면 formula에 오차가 있는 것

---

## 1. 초기화 단계

### 1.1 인자 파싱

```python
--log          = ".../resnet_18-B1.json"      # Ansor tuning log (JSON lines)
--tasks-pkl    = ".../resnet_18-(1,224,224,3).pkl"  # SearchTask 객체들
--max-tasks    = 24                            # 최대 처리할 task(workload) 수
--random-trials = 500                          # task당 랜덤 변형 record 수
--seed         = 0                             # 랜덤 시드
```

### 1.2 Task Map 로딩 — `build_task_map()`

```python
task_map = build_task_map(args.tasks_pkl)
```

pkl 파일에서 `(tasks, task_weights)` 튜플을 unpickle한다.
각 `SearchTask`를 `workload_key`로 인덱싱한 딕셔너리를 만든다.

```
{ "workload_key_abc..." : SearchTask_0,
  "workload_key_def..." : SearchTask_1, ... }
```

### 1.3 Record 그룹핑 — `_load_records_grouped()`

```python
groups = _load_records_grouped(args.log)
```

JSON lines 파일을 한 줄씩 읽어서, 각 record의 `workload_key`별로 그룹핑한다.
record 구조: `{"i": [task_info, [state_info, steps]], "r": [costs, ...], ...}`

```
{ "workload_key_abc..." : [record_0, record_1, ...],
  "workload_key_def..." : [record_5, record_6, ...], ... }
```

`all_wks`로 workload key를 정렬 후 최대 24개만 선택한다.

---

## 2. Task별 메인 루프 (24회 반복)

각 workload_key에 대해 **그 그룹의 첫 번째 record**를 `base_rec`로 사용한다.

### 2.1 Base Record → Task + State 복원

```python
task, base_state = record_to_task_state(base_rec, task_map)
```

1. `base_rec`의 JSON 문자열을 `load_record_from_string`으로 파싱 → `MeasureInput` 객체 획득
2. `MeasureInput.task.workload_key`로 `task_map`에서 정확한 `SearchTask` 찾기
3. `MeasureInput.state`를 `base_state`로 사용

### 2.2 Merge Report 획득 — `get_storage_rewrite_merge_report()`

```python
merge_report = get_storage_rewrite_merge_report(task, base_state)
```

**이것이 이 시스템의 핵심 아이디어.** Base record의 concrete 파라미터로 실제 lowering을 수행하여,
StorageRewrite가 어떤 버퍼끼리 merge했는지 정보를 얻는다.

#### 내부 과정:

1. `tir_analysis.clear_storage_rewrite_report()` — 글로벌 리포트 초기화
2. `lower_with_gpu_passes(task, base_state)` — 아래 pass 순서 실행:
   ```
   InjectPrefetch → StorageFlatten(64) → NarrowDataType(32) → Simplify
   → VectorizeLoop → InjectVirtualThread → StorageRewrite → Simplify
   ```
   `StorageRewrite` 내부의 `CollectMergeReport()`가 글로벌 맵에 기록:
   ```
   g_storage_rewrite_report["main"] = {
       "shared": [["buf_A", "buf_B"], ["buf_C"]],  ← merge 그룹
       "local":  [["buf_D", "buf_E"]]
   }
   ```
   같은 리스트 안의 버퍼들은 liveness가 겹치지 않아 **하나의 allocation을 공유**(reuse)한다는 뜻.
3. `tir_analysis.get_storage_rewrite_report()` — 글로벌 맵을 Python dict로 변환

#### 결과 예시:
```python
{
    "main": {
        "shared": [["conv2d_shared", "pad_temp_shared"]],
        "local":  [["conv2d_local"]]
    }
}
```

### 2.3 제약조건 시스템 구축 — `build_system()`

```python
system = build_system(base_rec, task, hw=hw, merge_report=merge_report)
```

이 함수가 **formula 전체를 만드는 핵심**이다. 내부 4단계:

---

#### 단계 A: `parse_steps()` — Step 파싱

Record의 `["i"][1][1]` (raw step 배열)을 구조화된 딕셔너리 리스트로 변환한다.

```python
[
  {"idx": 0, "kind": "SP", "stage_id": 3, "iter_id": 0, "extent": 512,
   "lengths": [4, 8], "inner_to_outer": True},
  {"idx": 1, "kind": "AN", "stage_id": 3, "iter_id": 1, "annotation": 6},
  {"idx": 2, "kind": "FU", "stage_id": 3, "fused_ids": [0, 1]},
  ...
]
```

지원하는 step 종류:
| Kind | 의미 |
|------|------|
| `SP` | Split — 루프를 여러 레벨로 분할 |
| `AN` | Annotate — threadIdx.x, vectorize 등 바인딩 |
| `FU` | Fuse — 여러 루프 합치기 |
| `RE` | Reorder — 루프 순서 변경 |
| `FSP` | FollowSplit — 다른 SP의 인수를 따라 분할 |
| `FFSP` | FollowFusedSplit — 여러 SP의 특정 레벨 곱으로 분할 |
| `CA` | ComputeAt — 특정 stage를 다른 stage의 루프에 배치 |
| `CI` | ComputeInline — 인라이닝 |
| `CR` | ComputeRoot — 루트 레벨로 이동 |
| `CHR` | CacheRead — 읽기 캐시 생성 (shared/local) |
| `CHW` | CacheWrite — 쓰기 캐시 생성 |
| `PR` | Pragma — auto_unroll 등 |
| `SA` | StorageAlign — 메모리 정렬 |

---

#### 단계 B: `build_dag_info()` → `simulate_steps()` — 심볼릭 시뮬레이션

##### B-1: `build_dag_info(task)`

Task의 `compute_dag`에서 초기 상태(아무 변환 안 한 상태)를 가져와 각 stage의 정보를 추출한다:

```python
[
  StageInfo(op_name="placeholder", op_type=0, scope="global", dtype_bytes=4,
            iters=[IterInfo(name="i", extent_expr=Expr("const",(512,)), ...)]),
  StageInfo(op_name="pad_temp.shared", op_type=1, scope="shared", ...),
  StageInfo(op_name="conv2d", op_type=1, scope="global", ...),
  ...
]
```

##### B-2: `simulate_steps(steps, dag_info)`

Step들을 **실제 lowering 없이** 순서대로 시뮬레이션하여, 각 stage의 iterator 구조를 갱신한다.

**핵심: Split 파라미터를 `param_expr(step_idx, pos)` 심볼로 남긴다.**

예) SP step `{"idx": 5, "lengths": [4, 8]}` 처리:
```
원래 iter: i (extent = 512)
→ split 후:
  i.0 (extent = ceildiv(512, SP[5].l[0] * SP[5].l[1]))   ← outermost
  i.1 (extent = SP[5].l[0])                                ← param
  i.2 (extent = SP[5].l[1])                                ← param (innermost)
```

SP 이외의 step도 시뮬레이션:
- **FU**: 선택된 iter들의 extent를 곱한 하나의 iter로 합침
- **RE**: iter 순서 재배열
- **FSP/FFSP**: 다른 SP의 param_expr를 참조하여 split
- **CA/CI/CR**: `compute_at`, `inline` 상태만 갱신
- **CHR/CHW**: 새 stage(cache stage) 삽입, scope를 "shared" 또는 "local"로 설정
- **AN**: iterator의 annotation 설정 (threadIdx.x, vectorize 등)

결과: `SimContext` — 모든 stage의 iterator가 심볼릭 `Expr`로 표현됨

---

#### 단계 C: `_calibrate_sim_context_with_bound()` — Probing으로 수식 보정

시뮬레이터의 extent 수식은 정확하지 않을 수 있다 (ceildiv 체인에서의 오차, bound inference와의 차이).
이를 **base record의 concrete 값으로 보정**한다.

##### C-1: Base Bound 수집

```python
base_bound = _infer_bound_from_record(task, base_record)
```

Base record를 TVM의 `infer_bound_from_state`에 넣어서,
**각 (stage, iter)의 실제 extent**를 구한다.

```
base_extent = {
    (3, 0): 4,     # stage 3, iter 0의 실제 extent
    (3, 1): 8,     # stage 3, iter 1의 실제 extent
    (3, 2): 16,    # ...
    (5, 0): 32,
    ...
}
```

##### C-2: Parameter Probing (핵심!)

**각 split 파라미터를 하나씩 변경하고, 어떤 stage의 extent가 바뀌는지 관찰한다.**

모든 SP step의 모든 length position을 열거한다:
```
params = [(step_idx=5, pos=0), (step_idx=5, pos=1), (step_idx=7, pos=0), ...]
```

각 파라미터에 대해:
1. 현재 값이 2가 아니면 probe값 = 2, 2이면 probe값 = 3
2. base record를 deepcopy하고 해당 파라미터만 probe값으로 변경
3. `infer_bound_from_state`를 다시 호출
4. **어떤 (stage, iter)의 extent가 base와 달라졌는지** 기록 → `iter_deps`
5. **어떤 stage의 전체 크기(prod of extents × dtype_bytes)가 달라졌는지** 기록 → `stage_size_deps`

예)
```
SP[5].l[0]을 4→2로 변경했더니:
  - (3, 0)의 extent가 4→2로 변함 → iter_deps[(3,0)] = [(5, 0)]
  - stage 5의 총 크기가 변함     → stage_size_deps[5] = [(5, 0)]
```

##### C-3: 수식 보정

각 (stage, iter)에 대해:

1. 시뮬레이터 수식(`eval_expr`)의 base 평가값이 `base_extent`와 일치하면 → 그대로 유지
2. 불일치이면:
   - 의존하는 파라미터가 없으면(`deps = []`) → 상수로 고정: `const_expr(target_base)`
   - 의존하는 파라미터가 있으면:
     ```
     dep_expr = SP[5].l[0] * SP[7].l[1]   (의존 파라미터들의 곱)
     dep_base = eval(dep_expr, base_steps) = 4 * 8 = 32
     target_base = 128 (실제 extent)
     ```
     - `target_base % dep_base == 0`이면: `extent = (128/32) * SP[5].l[0] * SP[7].l[1]`
     - 나누어지지 않으면: `const_expr(target_base)` (fallback)

같은 로직으로 stage 전체 크기 수식(`stage_size_exprs`)도 보정한다.

---

#### 단계 D: `extract_constraints()` — 제약조건 수식 조립

SimContext와 merge_report를 종합하여 최종 제약조건을 만든다.

##### D-1: Kernel 식별

각 stage를 `compute_at` 체인을 따라 올라가서 **kernel root stage**를 찾는다.
같은 kernel root에 속하는 stage들을 그룹핑한다.

##### D-2: Thread/Memory 수식 수집

각 kernel의 stage들을 순회하면서:

- `annotation == THREAD_X`인 iter → `thread_x_exprs`에 추가
- `annotation == THREAD_Y`인 iter → `thread_y_exprs`에 추가
- `annotation == THREAD_Z`인 iter → `thread_z_exprs`에 추가
- `annotation == VTHREAD`인 iter → `vthread_exprs`에 추가
- `annotation == VECTORIZE`인 iter → `vector_checks`에 추가
- `scope == "shared"`인 stage → `shared_stage_sizes[sid]`에 크기 수식 추가
- `scope == "local"`인 stage → `local_stage_sizes[sid]`에 크기 수식 추가

Stage 크기 수식:
- `stage_size_exprs`에 보정된 수식이 있으면 그것 사용
- 없으면: `dtype_bytes × ∏(모든 iter의 extent_expr)`

##### D-3: Merge Report 반영 (Shared/Local Memory 수식)

`merge_report`에서 가져온 merge 그룹을 stage ID에 매핑한다.

`_map_merge_groups_to_stage_ids()`:
- alloc 이름(예: `"conv2d_shared"`)을 normalize하고
- stage의 `op_name`과 prefix matching으로 대응
- 매칭이 모호(ambiguous)하면 fallback 플래그 설정

매핑 결과 예시:
```python
mapped_shared = [[3, 5], [7]]   # stage 3,5는 같은 merge group, stage 7은 단독
```

**Shared memory 수식 조립:**

- **ambiguous가 아닌 경우** (정상):
  ```
  shared_expr = max(size[3], size[5]) + size[7]
  ```
  같은 merge 그룹 내 → `max` (liveness reuse로 공간 공유)
  다른 그룹 간 → `add` (동시에 존재)

- **ambiguous인 경우** (fallback):
  ```
  shared_expr = size[3] + size[5] + size[7]
  ```
  전부 합산 (보수적 추정)

##### D-4: 기타 제약조건

- **innermost_constraints**: 각 SP step의 마지막 length가
  `max_innermost_split_factor(=64)` 이하여야 함

##### D-5: 최종 제약조건 구조

```python
{
    "hw": { "max_threads_per_block": 1024, "max_shared_memory_per_block": 49152, ... },
    "steps": [ ... ],
    "kernels": {
        root_stage_id: {
            "thread_per_block_expr": Expr(...),     # threadX * threadY * threadZ * vthread
            "vthread_prod_expr": Expr(...),
            "shared_expr": Expr(...),               # max/add 조합
            "local_expr": Expr(...),
            "thread_x_exprs": [Expr(...)],
            "thread_y_exprs": [Expr(...)],
            "thread_z_exprs": [Expr(...)],
            "vector_checks": [{"expr": Expr(...), "dtype_bytes": 4}, ...],
            "shared_merge_fallback": False,
            ...
        }
    },
    "innermost_constraints": [ ... ],
}
```

---

## 3. Base Record 검증

```python
pred_base = evaluate_record(system, base_rec)["valid"]
act_base  = _verify_record_with_lowering(task_map, base_rec, hw)
```

### 3.1 Formula 검증 — `evaluate_record()` → `check_constraints()`

Record의 concrete step 값을 수식에 대입하여 위반 여부를 판정한다.

**검사 항목 (순서대로):**

| # | 검사 | 수식 | 한계 |
|---|------|------|------|
| 1 | Divisibility | `extent % (l[0] × ... × l[n-1])` | `== 0` |
| 2 | Innermost split | `l[마지막]` | `≤ 64` |
| 3 | Thread axis X mismatch | 같은 kernel 내 모든 threadIdx.x 값 | 모두 동일 |
| 4 | Thread axis X limit | `threadIdx.x` | `≤ 1024` |
| 5 | Thread axis Y/Z | 동일 | Y≤1024, Z≤64 |
| 6 | Vthread | `∏(vthread extents)` | `≤ 8` |
| 7 | Thread per block | `X × Y × Z × vthread` | `≤ 1024` |
| 8 | Shared memory | `shared_expr` 평가 | `≤ 49152` bytes |
| 9 | Local memory | `local_expr` 평가 | `≤ 2^31 - 1` bytes |
| 10 | Vector | `vectorize_extent × dtype_bytes` | `≤ 16` bytes |

하나라도 위반이면 `valid = False`.

### 3.2 Ground Truth — `_verify_record_with_lowering()`

1. `record_to_task_state(rec)` → task, state 복원
2. `lower_with_gpu_passes(task, state)` → 전체 TIR lowering 수행
   (InjectPrefetch ~ StorageRewrite ~ Simplify)
3. `verify_gpu_module(mod, hw)` → `tir.analysis.verify_gpu_code`로 실제 검증
4. lowering 중 예외(ICHECK 실패 등) 발생 시 `False` 반환

### 3.3 Base 불일치의 의미

`base(False/True)` = formula가 reject했는데 실제로는 valid
→ formula의 **stage size 수식** 또는 **thread 수식**이 base record에서조차
  실제와 다르다는 뜻. calibration probing의 한계.

---

## 4. 랜덤 변형 루프 (task당 500회)

### 4.1 `randomize_record_params(base_rec, rng)`

Base record를 deepcopy하고 **SP와 PR step의 파라미터만 랜덤 변경**한다.
Step 시퀀스(backbone)는 그대로 유지 — 같은 sketch, 다른 파라미터.

#### SP (Split) 변경 규칙:
```python
extent = 512, lengths = [l0, l1]  (2-way split)
```
1. `rem = 512`
2. `l0` ← `rem`의 약수 중 랜덤 선택 → 예: `l0 = 8`, `rem = 512 / 8 = 64`
3. `l1` ← `rem`의 약수 중 `≤ max_innermost_split_factor(64)` 인 것 랜덤 선택

**주의**: extent의 약수만 선택하므로 divisibility 제약은 자동 만족.
단, ceildiv 체인에서의 rounding은 고려 안 됨.

-> ceildiv 체인이 뭐임?

#### PR (Pragma) 변경 규칙:
```
auto_unroll_max_step$X → X ← {0, 16, 64, 512, 1024} 중 랜덤
```

### 4.2 Formula Check (측정)

```python
t0 = time.perf_counter()
pred = evaluate_record(system, rec_mut)["valid"]
t1 = time.perf_counter()
```

단순히 수식에 concrete 값 대입 → 비교. **~43μs**.

### 4.3 Ground Truth (측정)

```python
act = _verify_record_with_lowering(task_map, rec_mut, hw)
t2 = time.perf_counter()
```

전체 TIR lowering + VerifyGPUCode. **~38ms**.

### 4.4 Mismatch 기록

`pred != act`이면 mismatch. 방향은 두 가지:

| pred | act | 의미 | 위험도 |
|------|-----|------|--------|
| True | False | Formula가 valid이라 했는데 실제 invalid | ⚠️ **Unsafe** (false negative) |
| False | True | Formula가 invalid이라 했는데 실제 valid | Safe but suboptimal (false positive) |

---

## 5. 결과 출력

### Task별 출력

```
Task  4: base(True/True) random mismatch 46/500
```
- `base(True/True)`: base record는 formula/actual 모두 valid
- `46/500`: 500개 랜덤 변형 중 46개 불일치 (9.2%)

### 전체 요약

```
Base mismatch      : 3/24        ← base record 자체의 불일치 수
Random mismatch    : 129/12000   ← 전체 랜덤 변형의 불일치 수 (1.08%)
Avg formula check  : 0.000043 s  ← 수식 평가 평균 시간
Avg lower+verify   : 0.038339 s  ← 실제 lowering 평균 시간
Speedup            : 893.34x     ← 속도 비율
```

Return code: mismatch가 하나라도 있으면 `exit(1)`, 전부 일치하면 `exit(0)`.

---

## 6. 전체 데이터 흐름 요약도

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Per-Task Setup (1회)                          │
│                                                                      │
│  base_record ─┬──→ record_to_task_state() ──→ (task, state)         │
│               │                                     │                │
│               │    ┌────────────────────────────────┘                │
│               │    │                                                 │
│               │    ▼                                                 │
│               │  get_storage_rewrite_merge_report()                  │
│               │    │  1. lower_with_gpu_passes()                     │
│               │    │     (concrete lowering with StorageRewrite)      │
│               │    │  2. CollectMergeReport → merge_report           │
│               │    │     {"shared": [["buf_A","buf_B"]], ...}        │
│               │    ▼                                                 │
│               └──→ build_system(base_record, task, hw, merge_report) │
│                     │                                                │
│                     ├─ parse_steps()         → step 구조화            │
│                     ├─ build_dag_info()      → 초기 stage 정보       │
│                     ├─ simulate_steps()      → 심볼릭 시뮬레이션     │
│                     │   (SP → param_expr, FU/RE/CA/CHR 등 적용)      │
│                     ├─ calibrate_with_bound()→ probing으로 수식 보정  │
│                     │   (파라미터 하나씩 변경 → infer_bound 비교)     │
│                     └─ extract_constraints() → 최종 제약 수식         │
│                         (merge_report 반영: max 그룹 + add)          │
│                                     │                                │
│                                     ▼                                │
│                              system (제약조건 수식 세트)              │
└──────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                Per-Trial (500회 반복)                  │
          │                                                       │
          │  randomize_record_params(base_rec) → rec_mut          │
          │           │                                           │
          │     ┌─────┴──────┐                                    │
          │     ▼            ▼                                    │
          │  Formula      Ground Truth                            │
          │  evaluate_     lower_with_gpu_passes()                │
          │  record()      + verify_gpu_module()                  │
          │  (~43μs)       (~38ms)                                │
          │     │            │                                    │
          │     ▼            ▼                                    │
          │  pred (T/F)   act (T/F)                               │
          │     │            │                                    │
          │     └────┬───────┘                                    │
          │          ▼                                            │
          │   pred == act? → mismatch count                       │
          └───────────────────────────────────────────────────────┘
```

---

## 7. Mismatch가 발생하는 근본 원인

### 원인 1: Merge 구조의 파라미터 의존성

`merge_report`는 **base record의 concrete 파라미터**로 lowering한 결과다.
파라미터가 바뀌면 StorageRewrite의 `FindAlloc`에서 liveness reuse 판단이 달라질 수 있다:

- `const_nbits`가 달라짐 → `const_free_map_`에서의 범위 매칭(1/16~16배)이 달라짐
- liveness 겹침 여부가 달라짐 → reuse 여부 변경

**결과**: formula는 `max(A, B)`로 계산했는데 실제로는 `A + B`가 필요 (또는 반대)

### 원인 2: Calibration Probing의 한계

`_calibrate_sim_context_with_bound`는 **단순 곱셈 모델**(`scale × param_product`)을 가정한다.
`ceildiv` 체인에서의 rounding, 비선형 의존성은 정확히 캡처하지 못한다.

예: `ceildiv(512, sp_0) × sp_0 = 512` (항상)이지만,
    `ceildiv(512, sp_0) × sp_1`은 sp_0에 따라 비선형.

### 원인 3: 이름 매칭 모호성

`_map_merge_groups_to_stage_ids()`의 prefix matching이 여러 stage에 매칭되면
`ambiguous = True` → 전부 합산 fallback → 과대 추정 가능.
