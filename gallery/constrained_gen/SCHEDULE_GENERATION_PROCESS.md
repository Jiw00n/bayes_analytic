# Schedule Generation Process

이 문서는 `gallery/constrained_gen/modules/schedule_generator.py`를 중심으로, 이 실험 코드에서 "schedule 생성"이 실제로 어떻게 이루어지는지 아주 자세하게 정리한 설명서다.

여기서 먼저 분명히 해야 할 점이 있다.

- 이 코드가 생성하는 것은 **새로운 transform step 구조 자체**가 아니다.
- 이 코드는 이미 정해진 **하나의 sketch 구조** 안에서, `SplitStep`과 `PragmaStep`의 **파라미터 값**을 새로 만든다.

즉, 이 저장소에서 말하는 schedule generation은 정확히는 아래에 가깝다.

1. 이미 존재하는 sketch를 하나 고른다.
2. 그 sketch를 `SymbolicState`로 바꾼다.
3. `sp_*`, `ur_*` 변수에 새로운 값을 넣는다.
4. 그 값을 다시 TVM `State`로 되돌린다.
5. 실제로 lower해서 GPU 제약을 만족하는지 확인한다.

따라서 이 문서는 "Ansor가 sketch를 어떻게 발명하는가"가 아니라, **이 실험 코드가 symbolic sketch를 어떻게 concrete schedule 후보로 인스턴스화하는가**를 설명한다.


## 1. 이 문서가 다루는 코드

핵심 구현:

- `gallery/constrained_gen/modules/schedule_generator.py`
- `gallery/constrained_gen/modules/expr_nodes.py`
- `gallery/constrained_gen/modules/param_manager.py`
- `gallery/constrained_gen/modules/tvm_verify.py`
- `gallery/constrained_gen/modules/record_loader.py`

맥락상 함께 봐야 하는 코드:

- `gallery/constrained_gen/modules/symbolic_state.py`
- `gallery/constrained_gen/modules/transform_applier.py`
- `gallery/constrained_symbolic_generator.ipynb`

근거 확인용 TVM 본체 코드:

- `src/auto_scheduler/search_policy/sketch_policy_rules.cc`
- `python/tvm/auto_scheduler/search_task.py`
- `src/tir/analysis/verify_gpu_code.cc`


## 2. 한 줄 요약

`ScheduleGenerator`는 `SymbolicState`에 들어 있는 symbolic extent 식과 annotation 정보를 읽어서 하드웨어 제약식을 만들고, 그 제약을 만족하도록 `sp_*`와 `ur_*` 값을 생성한 뒤, 그 값을 다시 TVM state로 patch해서 실제 schedule 후보를 만든다.


## 3. 전체 파이프라인

이 실험 코드에서 end-to-end 흐름은 대략 아래와 같다.

```text
JSON / MeasureRecord들
  -> sketch 기준으로 그룹핑
  -> 한 sketch의 base state 선택
  -> build_symbolic_state(task.compute_dag, base_state)
  -> ScheduleGenerator(sym_state, hw_param, enabled_constraints)
       -> _preprocess()
       -> 제약식과 변수 순서 준비
  -> randomize_params() 또는 enumerate_all_params()
       -> {sp_*, ur_*} dict 생성
  -> params_to_state(task, base_inp, base_res, params)
       -> concrete auto_scheduler State 생성
  -> task.compute_dag.apply_steps_from_state(new_state)
       -> 실제 TE schedule 생성
  -> lower_with_gpu_passes(task, new_state)
  -> verify_gpu_module(mod)
```

중요한 점:

- `ScheduleGenerator` 자체는 TE schedule을 직접 만들지 않는다.
- `ScheduleGenerator`의 출력은 단지 `params` dict다.
- 실제 schedule materialization은 `params_to_state` 이후 TVM API가 담당한다.


## 4. "무엇이 고정이고 무엇이 가변인가"

이 구분을 먼저 이해해야 생성 과정을 정확하게 볼 수 있다.

### 4.1 고정되는 것

하나의 sketch 안에서는 아래가 고정된다.

- step 종류의 순서
- 어떤 stage/iter가 split/reorder/fuse 되는지
- 어떤 iter가 vectorize/thread/vthread annotation을 가지는지
- 어떤 stage가 `compute_at`, `cache_read`, `cache_write` 되는지

이 구조는 `state.transform_steps`의 **형태**에서 온다.

### 4.2 가변되는 것

이 코드가 새로 생성하는 것은 아래뿐이다.

- `SplitStep.lengths`의 값
- `PragmaStep`의 `auto_unroll_max_step$N` 값

이것이 `SymbolicState` 안에서는 각각:

- `sp_{step_idx}_{length_idx}`
- `ur_{step_idx}`

로 이름 붙는다.

즉, 이 생성기는 "새 schedule 구조 탐색기"가 아니라 **기존 sketch의 자유 파라미터 채우기**다.


## 5. 왜 sketch 단위로 생성하나

이 부분은 `record_loader.py`에 있다.

`state_sketch_fingerprint(state)`는 각 step에서 **구조적 속성만** fingerprint에 넣고, 아래 값은 제외한다.

- `SplitStep.lengths`
- `PragmaStep`의 unroll 숫자

즉, 같은 sketch라는 뜻은:

- split이 몇 단계인지
- reorder가 어떤 순서인지
- cache read/write가 있는지
- compute_at 관계가 어떤지

는 같고,

- split factor 숫자
- unroll 숫자

만 다르다는 뜻이다.

그래서 한 sketch에서 하나의 `SymbolicState`를 만들고, 그 위에서 다양한 `params`를 생성하는 것이 가능하다.

이 설계는 현재 스케줄 생성 코드의 전제가 된다.


## 6. schedule generation의 실제 시작점

실전 사용 흐름은 보통 아래와 같다.

```python
sym_state = build_symbolic_state(task.compute_dag, base_state)
gen = ScheduleGenerator(sym_state, hw_param=None, enabled_constraints=None)
params = gen.randomize_params()
new_state = params_to_state(task, base_inp, base_res, params)
sch, tensors = task.compute_dag.apply_steps_from_state(new_state)
```

여기서 역할을 나누면 다음과 같다.

- `build_symbolic_state`
  - structure를 symbolic form으로 변환
- `ScheduleGenerator`
  - symbolic variable assignment 생성
- `params_to_state`
  - assignment를 concrete TVM state로 되돌림
- `apply_steps_from_state`
  - 실제 TE schedule materialization


## 7. `ScheduleGenerator` 생성자에서 하는 일

생성자는 짧아 보이지만 실제로는 중요한 전처리를 거의 다 수행한다.

```python
def __init__(self, sym_state, hw_param=None, enabled_constraints=None):
    self.s = sym_state
    self.hw = dict(self.DEFAULT_HW_PARAM)
    if hw_param is not None:
        self.hw.update(hw_param)
    self.pm = SymParamManager(sym_state)
    ...
    self._preprocess()
```

핵심 필드는 아래와 같다.

- `self.s`
  - 입력 `SymbolicState`
- `self.hw`
  - 하드웨어 제약 파라미터
- `self.pm`
  - `SymParamManager`
- `self._enabled`
  - 활성화된 제약 종류 집합
- `self._constraints`
  - 파싱된 constraint tree 목록
- `self._var_constraints`
  - 각 symbolic variable이 관련된 constraint index 목록
- `self._var_order`
  - 변수 할당 순서

즉, 생성자 호출이 끝났다는 것은 이미 다음이 끝났다는 뜻이다.

1. 어떤 변수들이 있는지 파악
2. 어떤 제약들이 있는지 파악
3. 각 제약을 식 트리로 파싱
4. 각 변수에 어떤 제약이 걸리는지 인덱싱
5. 어떤 순서로 값을 채울지 결정


## 8. 하드웨어 파라미터는 어디서 오나

기본값은 `ScheduleGenerator.DEFAULT_HW_PARAM`이다.

```python
{
    'max_vector_bytes': 16,
    'max_shared_memory_per_block': 49152,
    'max_threads_per_block': 1024,
    'max_vthread_extent': 8,
    'warp_size': 32,
    'max_innermost_split_factor': 64,
}
```

중요한 점:

- 이 값은 `task.hardware_params`를 자동으로 읽지 않는다.
- 사용자가 `hw_param`을 넘기지 않으면 위 상수를 그대로 쓴다.

즉, TVM의 `SearchTask.hardware_params`와 자동 연동되는 구조가 아니다.

TVM Python의 `HardwareParams`는 원래 아래 같은 필드도 가진다.

- `vector_unit_bytes`
- `max_local_memory_per_block`
- `max_thread_x`
- `max_thread_y`
- `max_thread_z`

하지만 현재 `ScheduleGenerator`는 그중 일부만 모델링한다.

### 8.1 `tvm_verify`와의 관계

`modules/tvm_verify.py`의 `GPU_VERIFY_CONSTRAINTS`는 아래를 추가로 가진다.

- `max_local_memory_per_block`
- `max_thread_x`
- `max_thread_y`
- `max_thread_z`

따라서 현재 파이프라인은 두 층으로 나뉜다.

1. `ScheduleGenerator`
   - 근사적이고 빠른 symbolic 제약 필터
2. `verify_gpu_module`
   - 실제 lowered TIR 기반의 최종 검증

즉, generator 단계에서 통과했다고 해서 GPU 코드가 반드시 유효한 것은 아니다.


## 9. `enabled_constraints`는 무엇을 켜고 끄나

사용 가능한 제약 종류는 다음 여섯 가지다.

- `vectorize`
- `shared_memory`
- `max_threads`
- `min_thread`
- `vthread`
- `innermost_split`

기본값은 전부 활성화다.

```python
if enabled_constraints is None:
    self._enabled = set(self.ALL_CONSTRAINT_KINDS)
```

unknown key가 들어오면 즉시 `ValueError`다.

```python
unknown = set(enabled_constraints) - set(self.ALL_CONSTRAINT_KINDS)
if unknown:
    raise ValueError(...)
```

중요한 구현 포인트:

- `innermost_split`은 `_constraints` 리스트에 들어가지 않는다.
- 대신 `_innermost_names` 집합을 만들어 candidate generation 때 직접 잘라낸다.

즉, `enabled_constraints`는 두 방식으로 작동한다.

1. 어떤 constraint tree를 만들지 결정
2. candidate domain을 직접 제한할지 결정


## 10. `SymParamManager`가 하는 일

스케줄 생성에서 `SymParamManager`는 화려한 클래스는 아니지만 매우 중요하다. `ScheduleGenerator`는 직접 모든 파라미터 구조를 계산하지 않고, 몇 가지 핵심 helper를 여기서 가져온다.

### 10.1 `_build_sp_groups`

`sym_state.sym_map`에서 `sp_*` 이름들을 읽어서 step index별로 그룹을 만든다.

예:

```text
sym_map keys:
  sp_3_0, sp_3_1, sp_3_2, sp_8_0, sp_8_1, ur_11

sp_groups:
  3 -> [sp_3_0, sp_3_1, sp_3_2]
  8 -> [sp_8_0, sp_8_1]
```

이 그룹이 의미하는 것은 "하나의 `SplitStep`에서 나온 factor 집합"이다.

### 10.2 `_build_sp_extents`

이 함수는 step index마다 "그 split이 쪼개는 원래 extent"를 구한다.

구현은 `self.s._state.transform_steps`를 다시 읽어서:

- 해당 `step_idx`가 `SplitStep`
- `step.extent is not None`

인 경우에만 `sp_extents[step_idx] = int(step.extent)`로 저장한다.

즉, generator가 파라미터 후보를 만들 때 쓰는 "원래 split 대상 extent"는 `SymbolicState` 내부 식을 역으로 추론해서 만드는 것이 아니라, **base TVM state의 `SplitStep.extent` 값**을 그대로 참조한다.

### 10.3 `_divisors`

양의 정수의 약수 목록을 오름차순으로 돌려준다.

이 함수는 candidate generation의 핵심이다. 현재 generator는 split factor 후보를 임의의 정수가 아니라, **현재 남은 extent의 약수**로만 선택한다.

즉, 파라미터 공간은 "정수 전체"가 아니라 "약수 기반 분할 공간"이다.

### 10.4 `UNROLL_CANDIDATES`

```python
[0, 16, 64, 512, 1024]
```

이 값은 `ur_*` 변수의 후보 집합이다.

중요한 점:

- unroll은 constraint tree 기반 filtering을 거치지 않는다.
- 단순히 이 다섯 값 중 하나를 뽑는다.


## 11. 전처리 `_preprocess()` 전체 그림

`_preprocess()`는 제약 생성의 핵심 단계다.

수행 순서는 아래와 같다.

1. `sp_groups` 구성
2. `sp_extents` 구성
3. `_ur_names`, `_all_sp_names`, `_innermost_names` 구성
4. 활성화된 제약을 읽어 `_constraints` 생성
5. `_var_constraints` 인덱스 생성
6. `_var_order` 계산

정리하면:

- `sp_groups`
  - variable을 split step 단위로 묶음
- `sp_extents`
  - 각 split step의 원래 크기
- `_constraints`
  - "이 변수들이 이런 식으로 이 상한/하한을 만족해야 함"의 목록
- `_var_constraints`
  - 한 변수에 영향을 주는 제약 빠르게 찾기 위한 역인덱스
- `_var_order`
  - 실제 sampling / DFS 순서


## 12. 전처리 중 만들어지는 내부 자료구조

### 12.1 `_sp_groups`

형태:

```text
{
  step_idx: [sp_step_idx_0, sp_step_idx_1, ...]
}
```

의미:

- 같은 split step에서 나온 factor들의 묶음

### 12.2 `_sp_extents`

형태:

```text
{
  step_idx: original_split_extent
}
```

주의:

- 오직 원래 `SplitStep`에 대해서만 생긴다.
- `FollowSplitStep`, `FollowFusedSplitStep`는 새 `sp_*`를 만들지 않으므로 직접 들어오지 않는다.

### 12.3 `_ur_names`

`sym_map`에서 `ur_`로 시작하는 키만 모은 리스트다.

### 12.4 `_all_sp_names`

모든 split 변수 이름을 step 순서대로 flatten한 리스트다.

### 12.5 `_innermost_names`

활성 제약에 `innermost_split`이 있을 때만 채워진다.

각 split group에서 마지막 변수 하나만 넣는다.

예:

```text
sp_3_0, sp_3_1, sp_3_2
```

이면:

```text
sp_3_2
```

만 innermost candidate limit의 대상이 된다.

### 12.6 `_constraints`

각 제약은 dict로 저장된다.

형태:

```text
{
  'tree': ExprNode,
  'rhs': int,
  'vars': set[str],
  'kind': str,
  'desc': str,
  'is_upper': bool,
  'has_nonlinear': bool,
}
```

의미:

- `tree`
  - 좌변 식
- `rhs`
  - 우변 상수
- `vars`
  - 이 식에 등장하는 symbolic variable 집합
- `kind`
  - 제약 종류
- `desc`
  - 디버깅용 설명
- `is_upper`
  - `lhs <= rhs`인지, 아니면 `lhs >= rhs`인지
- `has_nonlinear`
  - `min`, `ceildiv` 같은 비선형 구성요소가 있는지

### 12.7 `_var_constraints`

형태:

```text
{
  var_name: [constraint_idx, ...]
}
```

즉, 변수 하나가 관련된 constraint만 빠르게 찾기 위한 역인덱스다.


## 13. 제약식은 어디서 오나

constraint tree는 아무 식이나 파싱하는 것이 아니라, `SymbolicState`에서 읽어온 symbolic extent 식에 하드웨어 상수 조건을 붙여서 만들어진다.

이 절에서는 각 제약 종류가 **정확히 무엇을 의미하는지**, **어떤 코드에서 정보를 읽는지**, **실제 TVM 규칙과 어떤 관계인지**를 나눠서 정리한다.


## 14. `vectorize` 제약

### 14.1 정보 출처

`self.s.get_vectorize_extents()`를 호출한다.

이 함수는 `SymbolicState` 안에서 annotation code `2`인 iterator들만 수집한다.

즉, vectorize 대상은 이미 sketch 구조 안에서 결정되어 있어야 한다.

### 14.2 제약식

각 vectorized iter에 대해 다음 조건을 만든다.

```text
eval(sym_extent) * dtype_bytes <= max_vector_bytes
```

구현상으로는:

1. `sym_extent`를 string으로 꺼내 파싱
2. dtype byte가 1보다 크면 `ScaleMulNode(tree, dtype_bytes)`로 감쌈
3. upper-bound constraint로 등록

### 14.3 이 제약의 의미

예를 들어:

- vectorized extent가 `sp_10_2`
- dtype이 `float32`라서 4 byte
- `max_vector_bytes = 16`

이면:

```text
sp_10_2 * 4 <= 16
```

즉, `sp_10_2 <= 4`가 된다.

### 14.4 실제 TVM verify와의 관계

`verify_gpu_code`는 단순히 annotation된 loop extent만 보는 것이 아니라, lowered TIR에서:

- vectorized dtype lanes
- `Ramp`
- `BufferLoad`
- `BufferStore`
- `Cast`

등을 모두 살핀다.

따라서 generator의 vectorize constraint는 **빠른 symbolic 근사**이고, lowered TIR 기반 verify와 동치가 아니다.


## 15. `shared_memory` 제약

### 15.1 정보 출처

`self.s.get_shared_memory_extents()`를 읽는다.

이 값은 `SymbolicState`의 `_shared_fused_extents`에서 온다. 그리고 `_shared_fused_extents`는 `TransformApplier._apply_fuse`가 아래 조건일 때만 채운다.

- fused iter extent를 계산할 수 있음
- stage 이름에 `.shared`가 포함됨

즉, shared memory 모델링은 "shared stage의 fused extent"를 기반으로 한다.

### 15.2 제약식

각 item마다:

```text
extent * dtype_bytes
```

를 만들고, 전체 합에 대해:

```text
sum_i (extent_i * dtype_bytes_i) <= max_shared_memory_per_block
```

를 만든다.

구현에서는 item별 tree를 만든 뒤 `SumNode(children)`로 합쳐서 upper-bound constraint 하나로 등록한다.

### 15.3 의미

shared memory를 사용하는 shared stage들이 여러 개 있으면, 그 총합을 block당 shared memory 한도로 제한한다.

예:

```text
A.shared: sp_3_0 * sp_3_1 * 4 byte
B.shared: sp_7_2 * 2 byte
```

이면 총합 제약은:

```text
sp_3_0*sp_3_1*4 + sp_7_2*2 <= 49152
```

### 15.4 실제 verify와의 관계

`verify_gpu_code`는 lowered TIR의 `AllocateNode`를 방문해 실제 shared allocation size를 계산한다.

즉:

- generator는 symbolic fused extent 기반 근사
- verify는 lowered allocation 기반 실제 검사

둘은 방향은 같지만 완전히 같은 모델은 아니다.


## 16. `max_threads` 제약

### 16.1 정보 출처

`self.s.get_thread_extents()`를 읽는다.

이 함수는 annotation code가 아래인 iter들을 모은다.

- `threadIdx.x` -> 6
- `threadIdx.y` -> 8
- `threadIdx.z` -> 10

### 16.2 제약식

각 thread-bound iter에 대해:

```text
extent <= max_threads_per_block
```

를 건다.

### 16.3 중요한 해석

이 이름은 `max_threads`이지만, 현재 generator가 직접 체크하는 것은 **thread extent 하나하나의 상한**이다.

즉, 다음은 generator가 직접 모델링하지 않는다.

- `threadIdx.x * threadIdx.y * threadIdx.z`의 총 곱
- `vthread`와 합쳐진 thread-per-block 효과

반면 `verify_gpu_code`는 lowered TIR에서 실제 thread extent들을 모아 `thread_per_block_` 곱을 계산하고, 전체 block thread 수가 `max_threads_per_block`을 넘는지 검사한다.

따라서 generator의 `max_threads`는 **필요조건에 가까운 약한 필터**다.


## 17. `min_thread` 제약

이 제약은 현재 코드에서 가장 "TVM Ansor의 휴리스틱"에 가깝다.

### 17.1 정보 출처

thread-bound iter를 기준으로 보되, stage 이름이 `.shared`로 끝나는 경우는 건너뛴다.

그다음 `_find_orig_op`로 원본 op를 찾는다. 이 함수는 `.shared`, `.local` 접미사를 제거한 뒤 `compute_dag.ops`와 이름을 매칭한다.

### 17.2 제약식의 의미

원본 op의 `axis + reduce_axis`의 extent를 모두 곱한 값을 `total_space`라고 하면:

- `total_space <= warp_size * 2`
  - 제약을 만들지 않음
- `total_space > warp_size * 2`
  - 각 thread extent에 대해:
    - `extent >= warp_size`

를 만든다.

즉, 충분히 큰 iteration space를 가진 stage라면 `threadIdx.x`가 최소한 warp size는 되어야 한다는 뜻이다.

### 17.3 왜 `.shared` stage는 제외하나

코드 주석에도 적혀 있듯이 `.shared` stage는 cooperative fetching 경로에서 별도로 처리된다.

TVM C++ `InitThreadBind` 규칙에서도:

- 일반 multi-level tiled stage의 thread binding
- `.shared` cache_read stage의 cooperative fetching

을 다르게 다룬다.

현재 generator는 그 정책을 따라 `.shared` stage에는 `min_thread`를 직접 걸지 않는다.

### 17.4 TVM C++와의 관계

`schedule_generator.py` 주석은 이 로직이 `InitThreadBind::check_min_thread_extent`와 같다고 적고 있고, 실제 TVM C++ 코드도 다음 흐름을 가진다.

1. 원래 op의 `root_iter_vars()` 전체 공간 크기 계산
2. `total_space_extent <= warp_size * 2`면 check 비활성
3. 아니면 fused `threadIdx.x` extent가 `warp_size`보다 작으면 invalid

즉, 이 제약은 현재 generator가 가장 직접적으로 TVM Ansor heuristic을 모사하는 부분 중 하나다.


## 18. `vthread` 제약

### 18.1 정보 출처

`self.s.get_vthread_extents()`를 읽는다.

이 함수는 annotation code `4`인 iter만 모은다.

### 18.2 제약식

각 vthread iter에 대해:

```text
extent <= max_vthread_extent
```

### 18.3 실제 TVM와의 관계

TVM C++ `InitThreadBind` 규칙도 vthread extent가 hardware max보다 크면 invalid로 본다.

또 lowered TIR 단계의 `verify_gpu_code`도:

- `AttrStmt`의 `virtual_thread`
- `ForNode`의 `vthread.s`

를 읽어 `max_vthread`를 검사한다.

이 제약은 generator와 final verify의 방향이 비교적 잘 맞는 편이다.


## 19. `innermost_split` 제약

이 제약은 다른 것과 구현 방식이 다르다.

### 19.1 대상

각 split group에서 마지막 factor 하나:

```text
sp_{step_idx}_{last}
```

### 19.2 제약 의미

```text
sp_innermost <= max_innermost_split_factor
```

기본값은 `64`다.

### 19.3 왜 `_constraints`에 안 들어가나

현재 구현은 이 제약을 tree로 등록하지 않는다.

대신 두 군데에서 직접 처리한다.

1. `_preprocess()`에서 `_innermost_names`를 만듦
2. candidate generation 시:
   - `if name in self._innermost_names`
   - `candidates = [c for c in candidates if c <= innermost_limit]`

또한 `check_innermost_split()`에서 사후 검사도 한다.

즉, 이 제약은 "constraint solving 대상"이라기보다 "domain clipping rule"에 가깝다.

### 19.4 TVM C++와의 관계

TVM C++ `MutateTileSize`도 `max_innermost_split_factor` 기준을 사용한다.

현재 generator는 mutation rule을 직접 재현하지는 않지만, innermost split factor 상한이라는 정책은 같은 방향으로 가져온다.


## 20. 모델링하지 않는 것

이 문서에서 매우 중요하다. 현재 generator는 아래를 직접 제약식으로 모델링하지 않는다.

- local memory per block
- `threadIdx.x`, `y`, `z`의 축별 상한
- 전체 thread-per-block 곱
- kernel launch 개수
- unroll이 실제 lowered IR에 미치는 영향
- vectorized memory access의 정확한 TIR legality

따라서 실제 파이프라인에서 `verify_gpu_module`가 여전히 필요하다.

즉, 현재 generator는:

- 빠른 symbolic 필터
- 완전한 legality checker는 아님

이라고 보는 것이 정확하다.


## 21. 왜 `ExprNode` 트리가 필요한가

생성기는 symbolic extent 문자열을 그냥 `eval`로 검사하지 않는다.

대신 `expr_nodes.py`의 `ExprNode` 트리로 파싱해서 다음 두 기능을 쓴다.

1. `evaluate(assignment)`
2. `interval(domains)`

이 두 기능이 필요한 이유는 다르다.

- `evaluate`
  - 특정 assignment가 있을 때 실제 값 계산
- `interval`
  - 아직 변수가 일부만 정해졌을 때 가능한 하한/상한 계산

즉, 이 생성기는 완전한 SAT/ILP solver가 아니라, **interval arithmetic과 이분 탐색으로 후보를 빠르게 줄이는 휴리스틱 탐색기**다.


## 22. `ExprNode` 종류와 의미

현재 지원 노드는 아래와 같다.

- `ConstNode`
- `VarNode`
- `MulNode`
- `AddNode`
- `SubNode`
- `MinNode`
- `CeilDivNode`
- `ScaleMulNode`
- `SumNode`

각 노드는 세 가지 연산을 지원한다.

- `interval(domains)`
- `evaluate(assignment)`
- `variables()`

### 22.1 `interval(domains)`의 의미

입력 `domains`는 각 변수의 현재 도메인이다.

예:

```text
{
  "sp_3_0": [1, 32],
  "sp_3_1": 4,
}
```

여기서:

- `int`
  - 이미 값이 고정됨
- `[lo, hi]`
  - 아직 미확정이지만 이 구간 안에 있음

### 22.2 왜 이 표현이면 충분한가

현재 symbolic extent가 만들어내는 식은 대부분 아래 조합이다.

- 변수의 곱
- 합/차
- `min`
- `ceildiv`
- 상수 배
- shared memory 총합

즉, 현재 `SymbolicState`가 생성하는 extent 표현식을 처리하기에는 이 정도 문법이면 충분하다.


## 23. `parse_expr_tree`가 지원하는 문법

지원 문법은 매우 제한적이다.

- 정수 리터럴
- `sp_X_Y` 변수
- `a*b`
- `a+b`
- `a-b`
- `(expr)`
- `min(a,b)`
- `ceil(a/(b))`
- `math.ceil(a/(b))`

즉, 이 파서는 일반 목적 symbolic parser가 아니라 **현재 시스템이 만들어내는 식만 처리하는 전용 파서**다.

그 결과 장점은 단순함이다.

- interval 계산 구현이 쉬움
- 변수 집합 추출이 쉬움
- 이분 탐색 pruning에 쓰기 쉬움


## 24. `has_nonlinear`는 왜 계산하나

`_has_nonlinear(node)`는 아래를 비선형으로 본다.

- `MinNode`
- `CeilDivNode`

그리고 composite node는 자식 중 하나라도 비선형이면 비선형으로 본다.

이 값은 constraint solving 자체를 바꾸기보다는 **변수 순서 결정**에 쓰인다.

즉, linear-ish constraint에 더 많이 걸린 그룹을 먼저 할당해 pruning 효율을 높이려는 의도다.


## 25. 변수 순서 `_compute_var_order`

이 함수는 생성기의 효율을 크게 좌우한다.

아이디어는 "제약이 빡센 변수부터 먼저 고르자"에 가깝다.

### 25.1 먼저 변수들을 종류별로 본다

constraint를 훑어서:

- shared-memory 관련 변수 집합
- thread 관련 변수 집합
- 그 외 변수 집합

을 모은다.

실제 priority는 다음 세 카테고리다.

- `cat = 0`
  - shared memory 제약에 등장하는 group
- `cat = 1`
  - max_threads / min_thread 제약에 등장하는 group
- `cat = 2`
  - 나머지

### 25.2 group 단위로 우선순위를 계산한다

split group마다 다음 튜플을 계산한다.

```text
(cat, min_nonlinear, -total_freq)
```

의미:

- `cat`
  - shared -> thread -> other 순
- `min_nonlinear`
  - linear constraint에 걸린 그룹을 먼저
- `-total_freq`
  - 여러 제약에 많이 등장하는 그룹을 먼저

주의:

`min_nonlinear`라는 이름은 약간 헷갈릴 수 있다.

- 초기값은 `True`
- 어떤 관련 constraint라도 선형이면 `False`

이고, 정렬은 ascending이므로 결과적으로:

- `False`가 `True`보다 먼저 온다
- 즉, 선형 constraint가 포함된 그룹이 먼저 온다

### 25.3 최종 flatten

정렬된 step group 순서대로 group 내부 변수들을 그대로 이어 붙여 `_var_order`를 만든다.

즉, 한 split group의 factor들은 떨어져 배치되지 않고 연속으로 배치된다.


## 26. 왜 group 단위로 remaining extent를 관리하나

split factor 생성은 변수별로 독립적이지 않다. 같은 `SplitStep`에서 나온 factor들은 원래 extent를 공유한다.

예:

- 원래 extent = `64`
- 변수 그룹 = `[sp_3_0, sp_3_1, sp_3_2]`

이면 첫 factor가 8이면, 다음 factor는 원래 64 전체에서 고르는 것이 아니라 **남은 extent** 기준으로 골라야 한다.

그래서 generator는 `group_remaining[step_idx]`를 둔다.

초기값:

```text
group_remaining[step_idx] = original_extent
```

새 factor `chosen`을 고르면:

```python
group_remaining[step_idx] = (remaining + chosen - 1) // chosen
```

현재는 candidate를 항상 `remaining`의 약수에서 뽑기 때문에 보통 사실상 `remaining / chosen`과 같은 효과를 낸다.

이 구조 덕분에 한 split group 안의 factor들이 서로 일관된 곱 구조를 유지한다.


## 27. `randomize_params()` 전체 알고리즘

이 함수가 실제 랜덤 schedule generation의 본체다.

전체 흐름을 먼저 pseudocode로 쓰면 아래와 같다.

```text
for attempt in range(max_retries):
    sym_map의 모든 sp_*를 1로 초기화
    domains 초기화
    group_remaining 초기화

    for var in _var_order:
        extent가 없으면 현재값(사실상 1) 고정
        아니면:
            candidates = divisors(remaining)
            innermost limit 적용
            현재 domain interval과 교집합
            관련 constraints로 후보 pruning
            후보가 없으면 [1] fallback
            랜덤 선택
            sym_map / result / domains 갱신
            remaining 갱신
            propagation 수행

    ur_*는 고정 후보 집합에서 랜덤 선택
    check_all()로 사후 검사
    violation 없으면 반환

실패하면 retry
```

이제 각 단계를 자세히 본다.


## 28. 단계 1: 모든 `sp_*`를 1로 초기화

```python
for name in self._all_sp_names:
    self.s.sym_map[name] = 1
```

이 동작의 의미:

- partial assignment 동안 아직 정해지지 않은 변수는 기본적으로 `1`로 간주된다.
- concrete evaluation이나 propagation 계산을 할 때 미정 변수가 1처럼 동작하게 된다.

장점:

- `evaluate()` 호출이 단순하다.
- lower bound 쪽 계산이 안정적이다.

주의:

- 이는 엄밀한 symbolic solving이 아니라 heuristic이다.


## 29. 단계 2: domain 초기화

각 `sp_*` 변수에 대해 domain을 만든다.

규칙:

- 원래 split extent를 알면 `[1, extent]`
- 모르면 `1`

예:

```text
sp_3_0 -> [1, 64]
sp_3_1 -> [1, 64]
sp_7_0 -> 1
```

중요한 해석:

- `[1, extent]`는 가능한 약수 후보의 **구간 외피**일 뿐, 실제 후보 목록이 아니다.
- 실제 후보는 나중에 `divisors(remaining)`에서 다시 만든다.
- `extent is None`인 변수는 사실상 탐색하지 않고 고정값 1로 둔다.

즉, 원래 split extent를 모르는 변수는 generator가 적극적으로 탐색하지 않는다.


## 30. 단계 3: `group_remaining` 초기화

각 split group마다 남은 extent를 기록한다.

```python
for step_idx, ext in self._sp_extents.items():
    group_remaining[step_idx] = ext
```

이 값은 같은 group의 다음 factor를 고를 때 domain을 줄이는 데 쓰인다.


## 31. 단계 4: 변수별 candidate 생성

본체 루프는 `_var_order`를 순서대로 돈다.

```python
for name in self._var_order:
    ...
```

### 31.1 extent를 모르는 변수

```python
if extent is None:
    result[name] = self.s.sym_map[name]
    domains[name] = self.s.sym_map[name]
    continue
```

현재 시점에서 `self.s.sym_map[name]`은 앞서 1로 초기화되어 있으므로, 이런 변수는 사실상 `1`로 고정된다.

즉, generator는 "원래 split extent를 아는 변수" 중심으로 탐색하도록 설계돼 있다.

### 31.2 extent를 아는 변수

가장 먼저 현재 남은 extent의 약수 목록을 만든다.

```python
remaining = group_remaining.get(step_idx, extent)
candidates = self.pm._divisors(remaining)
```

예:

- `remaining = 64`

이면:

```text
[1, 2, 4, 8, 16, 32, 64]
```

가 된다.

즉, candidate set의 기본 형태는 항상 "남은 extent의 약수들"이다.


## 32. 단계 5: innermost factor 상한 적용

현재 변수가 `_innermost_names`에 들어 있으면:

```python
candidates = [c for c in candidates if c <= innermost_limit]
```

즉, 마지막 factor는 `max_innermost_split_factor`보다 클 수 없다.

이 단계는 constraint tree를 통하지 않는 직접 domain clipping이다.


## 33. 단계 6: 현재 interval domain과 교집합

candidate는 약수 목록이지만, propagation에 의해 domain이 이미 줄어들었을 수 있다.

예:

- 후보 약수: `[1, 2, 4, 8, 16, 32, 64]`
- 현재 domain: `[4, 16]`

이면:

```text
[4, 8, 16]
```

만 남긴다.

구현은 단순히:

- `<= hi`
- `>= lo`

로 자른다.

즉, interval domain은 약수 후보 위에 덧씌워진 추가 bound다.


## 34. 단계 7: constraint 기반 후보 pruning

현재 변수에 관련된 constraint index 목록을 찾아:

```python
constraint_indices = self._var_constraints.get(name, [])
```

있으면 `_filter_by_constraints()`를 호출한다.

이 단계가 generator의 핵심 pruning 로직이다.

입력:

- 현재 변수 이름
- 현재 candidate 목록
- 관련 constraint 목록
- 전체 domains

출력:

- 더 좁아진 candidate 목록


## 35. `_filter_by_constraints()`가 하는 일

이 함수는 candidate 전체를 brute-force로 검사하지 않는다.

대신 관련 제약을:

- upper-bound 제약
- lower-bound 제약

으로 나누고, 각 제약에 대해 **이분 탐색으로 유효한 인덱스 범위**를 찾는다.

### 35.1 상한 제약

사용 함수:

- `_bisect_upper`
- `_bisect_upper_concrete`

### 35.2 하한 제약

사용 함수:

- `_bisect_lower`

### 35.3 결과 합치기

각 상한 제약이 허용하는 최대 인덱스의 최소를 취하고,
각 하한 제약이 요구하는 최소 인덱스의 최대를 취해서,

```text
candidates[min_valid_idx : max_valid_idx + 1]
```

를 반환한다.


## 36. `_bisect_upper`: interval lower-bound 기반 상한 pruning

이 함수는 질문을 이렇게 바꾼다.

> 현재 변수 값을 candidate `x`로 고정했을 때, 나머지 미정 변수들이 domain 안에서 어떤 값을 가져도 제약을 만족할 가능성이 있는가?

upper-bound 제약 `lhs <= rhs`에서 중요한 것은 `lhs`의 가능한 **최소값**이다.

왜냐하면:

- 가능한 최소값조차 `rhs`보다 크면
  - 어떤 assignment로도 만족 불가
- 가능한 최소값이 `rhs` 이하이면
  - 적어도 만족 가능성은 있음

그래서 `_bisect_upper`는:

1. `tree.interval(test_dom)`를 호출해 `(lhs_min, lhs_max)` 구함
2. `lhs_min <= rhs`인지 검사
3. 이 성질이 candidate에 대해 단조적이라고 보고 이분 탐색

즉, 이 단계는 **아직 미정 변수까지 포함한 보수적 가능성 검사**다.


## 37. `_bisect_upper_concrete`: 현재 partial assignment 기반 상한 pruning

이 함수는 interval 대신 현재 `self.s.sym_map` snapshot을 쓴다.

즉, 이미 정해진 변수들을 concrete 값으로 넣고, 현재 변수만 candidate에 따라 바꿔 보면서:

```text
tree.evaluate(sym_map_snap) <= rhs
```

가 되는 최대 인덱스를 찾는다.

이 단계가 필요한 이유는 `_bisect_upper`의 interval bound가 너무 느슨할 수 있기 때문이다.

즉:

- `_bisect_upper`
  - 가능성 기준의 coarse pruning
- `_bisect_upper_concrete`
  - 현재 partial assignment 기준의 sharper pruning

둘을 함께 써서 upper-bound pruning을 강화한다.


## 38. `_bisect_lower`: interval upper-bound 기반 하한 pruning

lower-bound 제약은 `lhs >= rhs`다.

여기서 중요한 것은 `lhs`의 가능한 **최대값**이다.

왜냐하면:

- 가능한 최대값조차 `rhs`보다 작으면
  - 어떤 assignment로도 만족 불가
- 가능한 최대값이 `rhs` 이상이면
  - 만족 가능성은 있음

그래서 `_bisect_lower`는:

1. `tree.interval(test_dom)`에서 `lhs_max` 사용
2. `lhs_max >= rhs`가 되는 최소 candidate index를 찾는다

즉, lower-bound pruning은 "이 값보다 작으면 아무리 나머지를 좋게 골라도 부족하다"를 찾는 과정이다.


## 39. `_filter_by_constraints()`의 fallback

이 함수에는 중요한 fallback이 하나 있다.

```python
if min_valid_idx > max_valid_idx:
    return [candidates[0]]
```

즉, pruning 결과 후보가 완전히 비면 곧장 실패시키지 않고 **가장 작은 후보 하나를 남긴다.**

이것은 generator의 성격을 잘 보여 준다.

- 완전한 결정적 solver가 아님
- 실패를 바로 선언하기보다 일단 진행해 보고
- 마지막 `check_all()`에서 걸러내는 전략

이 fallback은 `randomize_params()`에만 사실상 의미가 있다.


## 40. 후보가 비었을 때 `randomize_params()`가 하는 일

`_filter_by_constraints()` 이후 후보가 비면:

```python
if not candidates:
    candidates = [1]
```

로 다시 한 번 fallback한다.

즉, 랜덤 생성 경로는 아주 보수적으로 "일단 1이라도 넣고 끝까지 가 본다"는 전략을 쓴다.

이 뒤 최종 `check_all()`에서 violation이 나면 attempt 전체를 버리고 retry한다.


## 41. 단계 8: 랜덤 선택과 상태 갱신

후보가 남았으면:

```python
chosen = rng.choice(candidates)
self.s.sym_map[name] = chosen
result[name] = chosen
domains[name] = chosen
group_remaining[step_idx] = (remaining + chosen - 1) // chosen
```

의미:

- `sym_map`
  - 현재 partial assignment 반영
- `result`
  - 최종 반환용 dict
- `domains[name] = chosen`
  - 이제 이 변수는 interval이 아니라 고정값
- `group_remaining`
  - 같은 split group의 다음 factor 후보 축소


## 42. 단계 9: `_propagate_domain()`

현재 변수에 관련된 constraint가 있었으면 propagation을 수행한다.

이 함수는 다음 아이디어를 쓴다.

> 방금 변수 하나를 정했으니, 같은 제약에 걸린 다른 미정 변수들의 허용 구간도 줄일 수 있지 않을까?

즉, candidate filtering이 "현재 변수 하나"를 자르는 것이라면, propagation은 "다른 변수의 future domain"을 줄이는 단계다.


## 43. 상한 제약에서의 propagation

상한 제약 `lhs <= rhs`에서 다른 미정 변수 `other_var`에 대해:

1. `other_var = cur_hi`를 넣어 평가
   - 이미 만족하면 도메인 유지
2. `other_var = cur_lo`를 넣어 평가
   - 이것도 위반이면 사실상 매우 빡센 상황이므로 `dom[1] = cur_lo`
3. 그 사이 어디까지 허용되는지 이분 탐색
4. 허용되는 최대값으로 `dom[1]` 줄임

즉, 상한 제약 propagation은 **다른 변수의 upper bound를 줄이는 작업**이다.


## 44. 하한 제약에서의 propagation

하한 제약 `lhs >= rhs`에서:

1. `other_var = cur_lo`를 넣어 이미 만족하면 유지
2. `other_var = cur_hi`를 넣어도 여전히 미달이면
   - 다른 미정 변수 영향이 있을 수 있으므로 **도메인을 줄이지 않고 스킵**
3. 아니라면 최소한 어느 값 이상이어야 만족 가능한지 이분 탐색
4. 그 값으로 `dom[0]` 올림

즉, 하한 제약 propagation은 **다른 변수의 lower bound를 올리는 작업**이다.

중요한 nuance:

상한 제약은 비교적 공격적으로 줄이고,
하한 제약은 "다른 변수 때문에 나중에 만족될 수도 있음"을 인정해서 더 보수적으로 동작한다.


## 45. 단계 10: unroll 값 생성

모든 `sp_*`가 끝나면 `ur_*`를 채운다.

```python
for name in self._ur_names:
    chosen = rng.choice(self.pm.UNROLL_CANDIDATES)
```

후보:

```text
[0, 16, 64, 512, 1024]
```

중요한 점:

- unroll은 `_constraints`에 등장하지 않는다.
- 따라서 candidate filtering도 propagation도 받지 않는다.
- 완전히 독립적으로 샘플링된다.

즉, unroll은 현재 generator에서 **제약 기반 탐색 대상이 아니라 단순 이산 choice**다.


## 46. 단계 11: 최종 사후 검사 `check_all()`

모든 값을 뽑은 뒤:

```python
violations = self.check_all()
if not violations:
    return result
```

를 수행한다.

이 단계는 왜 필요한가?

- 중간 pruning은 interval 근사다
- fallback 때문에 사실상 약간 무리한 값이 살아남을 수 있다
- unroll은 제약 모델링 대상이 아니다

그래서 마지막에 현재 `sym_map`을 기준으로 명시적 check 함수를 다시 돌려 실제 violation 문자열이 없는지 확인한다.

### 46.1 `check_all()`이 하는 일

활성화된 constraint kind에 대해:

- `check_vectorize`
- `check_shared_memory`
- `check_max_threads`
- `check_min_thread`
- `check_vthread`
- `check_innermost_split`

를 호출해서 violation 문자열 목록을 합친다.

즉, `_constraints`는 pruning용 internal model이고, `check_*` 함수들은 최종 명시적 검사용 API다.


## 47. `randomize_params()`의 실패 방식

violation이 있으면 attempt 전체를 버리고 retry한다.

```python
for attempt in range(max_retries):
    ...
raise RuntimeError(...)
```

기본 사용 예시에서는 `max_retries=1`인 경우도 많다.

즉:

- 한 번 뽑아 보고
- 안 되면 바로 실패

라는 사용 패턴도 가능하다.


## 48. `enumerate_all_params()`는 무엇이 다른가

이 함수는 랜덤 생성 대신 DFS로 가능한 조합을 전수 열거한다.

큰 흐름은 `randomize_params()`와 매우 유사하다.

공통점:

- 같은 `_var_order`
- 같은 `domains`
- 같은 `group_remaining`
- 같은 `_filter_by_constraints`
- 같은 `_propagate_domain`
- 마지막에 `check_all()`

차이점:

- 랜덤 선택 대신 모든 candidate를 순회
- branch가 비면 바로 backtrack
- leaf에 도달한 결과만 저장
- unroll은 마지막에 Cartesian product로 결합


## 49. DFS의 중요한 차이 1: fallback이 더 엄격하다

랜덤 생성은 후보가 비면 `[1]` fallback을 넣고 계속 진행한다.

하지만 DFS는:

```python
if not candidates:
    return
```

즉, 바로 해당 branch를 버린다.

따라서 DFS 열거는 랜덤 샘플링보다 더 "solver처럼" 동작한다.


## 50. DFS의 중요한 차이 2: backtracking 상태 복원

DFS는 branch 하나를 탐색하고 돌아와야 하므로 다음 상태를 복원한다.

- `self.s.sym_map[name]`
- `group_remaining[step_idx]`
- interval `domains`
- `result`

특히 interval domain은 mutable list이므로, branch 진입 전 `saved_domains`를 따로 복사해 둔 뒤 복원한다.

즉, DFS는 단순한 재귀가 아니라 **mutable shared state를 수동 복구하는 백트래킹 구현**이다.


## 51. DFS의 중요한 차이 3: unroll은 나중에 카르테시안 곱

DFS는 SP 변수만 먼저 열거한다.

그 다음:

```python
ur_combos = itertools_product(*[UNROLL_CANDIDATES ...])
```

로 모든 unroll 조합을 곱한다.

즉, 전체 결과 공간은:

```text
valid_sp_assignments × unroll_candidates^num_ur
```

이다.

장점:

- split 제약 pruning과 unroll independent choice를 분리할 수 있다.

단점:

- `ur_*` 수가 많으면 결과 수가 급격히 커질 수 있다.


## 52. `max_results` 안전 장치

DFS는 `max_results`를 초과하면 조기 종료한다.

즉, 이 함수는 "정말 모든 조합"이 아니라 "최대 N개까지" 열거한다.

노트북에서도:

- 일정 수 이상이면 "공간이 큼"으로 판단해 스킵

하는 방식으로 사용한다.


## 53. 생성기의 출력은 정확히 무엇인가

`randomize_params()`와 `enumerate_all_params()`가 반환하는 것은 둘 다 dict다.

형태:

```text
{
  "sp_3_0": 4,
  "sp_3_1": 8,
  "sp_7_0": 2,
  "ur_11": 64,
}
```

이 dict는 아직 TVM schedule도 아니고 TVM state도 아니다.

이것은 단지:

- 특정 sketch에 대한
- 자유 파라미터의 concrete assignment

일 뿐이다.


## 54. `params_to_state()`가 하는 일

이 함수는 `modules/tvm_verify.py`에 있다.

구현 아이디어는 간단하다.

1. `base_inp`, `base_res`를 JSON record string으로 dump
2. JSON 내부 step 배열을 직접 수정
3. 다시 `load_record_from_string`으로 읽어 새 `MeasureInput` 생성
4. 그 안의 `state` 반환

즉, symbolic 파라미터 dict를 TVM state로 바꾸는 과정은 **record patching**으로 구현되어 있다.

### 54.1 `sp_*` 매핑 규칙

이름:

```text
sp_{step_idx}_{length_idx}
```

을 읽어:

- JSON step array의 `steps[step_idx]`
- step prefix가 `"SP"`인지 확인
- `s[4][length_idx]`에 값을 씀

즉, `SplitStep.lengths` 리스트의 해당 위치를 바꾸는 것이다.

### 54.2 `ur_*` 매핑 규칙

이름:

```text
ur_{step_idx}
```

을 읽어:

- `steps[step_idx]`
- step prefix가 `"PR"`인지 확인
- `s[3] = "auto_unroll_max_step$N"`

으로 덮어쓴다.

즉, unroll은 JSON record 안의 pragma 문자열을 교체해서 materialize된다.

### 54.3 왜 base record가 필요한가

generator는 구조를 만들지 않으므로, 원래 sketch 구조를 가진 concrete base record가 필요하다.

즉, `params_to_state()`는 아래 전제를 가진다.

- step sequence는 base record와 같아야 한다
- 바꿀 것은 split factor와 unroll 값뿐이다


## 55. 실제 schedule materialization

새 state를 만들었다고 끝이 아니다. 실제 schedule은 다음 호출에서 만들어진다.

```python
sch, tensors = task.compute_dag.apply_steps_from_state(new_state)
```

이 호출이 의미하는 것:

- TVM이 concrete `State`의 step sequence를 읽고
- 실제 `te::Schedule`을 적용해서
- codegen 가능한 schedule object를 만든다

즉, 진짜 schedule은 여기서 처음 물질화된다.


## 56. lower와 최종 GPU verify

`tvm_verify.py`는 그다음 단계를 담당한다.

### 56.1 `lower_with_gpu_passes`

```python
sch, tensors = task.compute_dag.apply_steps_from_state(state)
mod = schedule_to_module(...)
mod = GPU_PASSES(mod)
```

즉:

1. TVM state -> TE schedule
2. TE schedule -> IRModule
3. GPU 관련 TIR pass pipeline 적용

### 56.2 `verify_gpu_module`

각 `PrimFunc`에 대해:

```python
tir.analysis.verify_gpu_code(func, constraints)
```

를 호출한다.

이 함수는 lowered TIR를 직접 읽어서 다음 같은 실제 제약을 검사한다.

- shared memory per block
- local memory per block
- threads per block 총합
- `threadIdx.x/y/z` 축별 상한
- vthread
- vector bytes

즉, generator의 symbolic filter보다 훨씬 실제 코드에 가깝다.


## 57. generator와 final verify의 관계

이 둘을 혼동하면 안 된다.

### 57.1 generator 단계

장점:

- 빠름
- symbolic state만 있으면 됨
- candidate space를 많이 줄일 수 있음

한계:

- lowered TIR를 보지 않음
- 근사 모델임
- 일부 제약은 아예 모델링하지 않음

### 57.2 final verify 단계

장점:

- 실제 lowered IR 기준
- TVM이 실제로 생성한 코드에 대한 검증

한계:

- 더 무겁다
- symbolic pruning처럼 search 중간에 쓰기 어렵다

즉, 현재 파이프라인은:

- 앞단은 빠른 근사 pruning
- 뒷단은 정확한 실코드 검증

의 2단계 구조다.


## 58. miniature example로 보는 랜덤 생성

아래처럼 하나의 split group이 있다고 하자.

```text
step_idx = 3
sp group = [sp_3_0, sp_3_1, sp_3_2]
original extent = 64
```

그리고 어떤 symbolic extent 식에서:

```text
sp_3_2 * 4 <= 16
```

이라는 vectorize 제약이 있다고 하자.

### 58.1 초기 상태

```text
domains:
  sp_3_0 -> [1, 64]
  sp_3_1 -> [1, 64]
  sp_3_2 -> [1, 64]

group_remaining[3] = 64
```

### 58.2 첫 변수 `sp_3_0`

후보:

```text
divisors(64) = [1, 2, 4, 8, 16, 32, 64]
```

현재는 `sp_3_2` 제약이므로 `sp_3_0`에는 직접 영향이 작을 수 있다.

예를 들어 랜덤으로 `8`을 선택했다고 하자.

그러면:

```text
sp_3_0 = 8
group_remaining[3] = ceil(64 / 8) = 8
```

### 58.3 둘째 변수 `sp_3_1`

후보는 이제:

```text
divisors(8) = [1, 2, 4, 8]
```

예를 들어 `2`를 골랐다면:

```text
sp_3_1 = 2
group_remaining[3] = ceil(8 / 2) = 4
```

### 58.4 셋째 변수 `sp_3_2`

후보는:

```text
divisors(4) = [1, 2, 4]
```

vectorize 제약:

```text
sp_3_2 * 4 <= 16
```

는 사실상:

```text
sp_3_2 <= 4
```

이므로 모든 후보가 허용된다.

만약 constraint가 더 빡세서:

```text
sp_3_2 * 4 <= 8
```

이라면 `_filter_by_constraints()`는 `[1, 2]`만 남길 것이다.

즉, generator는:

- split group consistency는 `group_remaining`
- hardware consistency는 constraint filtering

으로 동시에 관리한다.


## 59. `check_all()`과 `_constraints`의 차이

둘 다 제약을 다루지만 용도가 다르다.

### `_constraints`

목적:

- search 중간 pruning
- interval arithmetic
- variable ordering

형태:

- `ExprNode` tree + metadata

### `check_all()`

목적:

- 최종 assignment 사후 검증

형태:

- concrete evaluation 후 violation 문자열 목록

즉, `_constraints`는 solver 내부 표현이고, `check_all()`은 외부 검사용 API다.


## 60. `enabled_constraints`를 끄면 실제로 무엇이 달라지나

예를 들어:

```python
gen = ScheduleGenerator(sym_state, enabled_constraints={'shared_memory', 'max_threads'})
```

이면:

- `_constraints`에는 shared_memory와 max_threads 관련 tree만 등록된다
- vectorize, min_thread, vthread는 pruning에도 check에도 쓰이지 않는다
- `innermost_split`이 빠져 있으면 `_innermost_names`도 비게 된다

즉, disabled constraint는:

- tree building
- candidate filtering
- check_all

모두에서 빠진다.


## 61. 이 생성기를 재사용할 때 주의할 점

`ScheduleGenerator`는 immutable solver가 아니다.

이 객체는 내부에서 `self.s.sym_map`을 계속 바꾼다.

즉:

- 한 번 `randomize_params()`를 호출하고 나면
- `sym_state.sym_map`은 마지막 선택값들로 덮여 있다

노트북 예시에서 trial마다 새 `ScheduleGenerator(sym_state)`를 만들고 있는 이유도 이 mutable state 성격과 관련이 있다.

물론 같은 generator를 다시 호출해도 초기화 루프가 `sp_* = 1`로 다시 맞춰 주기는 하지만, 외부에서 `sym_state`를 공유하고 있다면 이 mutation을 알고 있어야 한다.


## 62. 현재 구현의 중요한 한계와 가정

이 절은 실전 사용에서 매우 중요하다.

### 62.1 generator는 sketch 구조를 만들지 않는다

생성기의 출력이 아무리 좋아도, base sketch 자체가 나쁘면 해결되지 않는다.

즉, 이 코드는 "구조 탐색"이 아니라 "파라미터 인스턴스화"다.

### 62.2 일부 split variable은 실제로 탐색되지 않을 수 있다

`_build_sp_extents()`에서 원래 `SplitStep.extent`를 얻지 못하면, 그 변수는 domain이 고정 1이 된다.

즉, base state에 extent 정보가 없는 split group은 generator가 적극적으로 탐색하지 못한다.

### 62.3 unroll은 거의 독립 변수다

현재 unroll은:

- 제약 tree 없음
- propagation 없음
- final `check_all` 직접 제약 없음

즉, unroll이 lowered code에 미치는 영향은 generator보다 final TVM verify에 더 많이 의존한다.

### 62.4 `max_threads`는 실제 block thread 총합을 직접 모델링하지 않는다

앞서 말했듯 이는 약한 근사다.

### 62.5 shared memory 모델도 실제 allocation과 완전히 같지 않다

현재 모델은 shared stage fused extent 기반이다.

실제 TIR allocation size와는 차이가 날 수 있다.

### 62.6 parser는 현재 symbolic 식 문법에만 맞춰져 있다

`parse_expr_tree()`가 이해하지 못하는 새 형태의 symbolic 식이 생기면 generator가 깨질 수 있다.

즉, `SymbolicState`의 표현식 생성 규칙과 `ExprNode` 파서의 문법 지원 범위는 같이 진화해야 한다.


## 63. 디버깅할 때 어디를 보면 되나

실전에서 생성이 이상할 때는 아래를 순서대로 보는 것이 좋다.

1. `sym_state.sym_map`
   - 어떤 `sp_*`, `ur_*`가 있는지
2. `gen._sp_groups`
   - split group이 기대대로 묶였는지
3. `gen._sp_extents`
   - 원래 split extent를 제대로 읽었는지
4. `gen._constraints`
   - 어떤 symbolic 식이 실제 constraint로 등록됐는지
5. `gen._var_constraints`
   - 특정 변수가 어떤 제약에 걸려 있는지
6. `gen._var_order`
   - 변수 순서가 합리적인지
7. `gen.check_all(params)`
   - 위반이 정확히 무엇인지
8. `params_to_state(...)`
   - record patch가 제대로 되었는지
9. `task.compute_dag.apply_steps_from_state(new_state)`
   - TVM이 실제 schedule 적용을 받아들이는지
10. `lower_with_gpu_passes` + `verify_gpu_module`
   - lowered code 기준으로 어떤 제약이 깨지는지

노트북의 diagnostic cell처럼 `randomize_params()` 내부를 수동으로 trace하는 것도 매우 유용하다.


## 64. 이 코드베이스에서 schedule generation을 가장 정확히 정의하면

현재 저장소의 구현을 기준으로 가장 정확한 정의는 다음이다.

> schedule generation은, 하나의 sketch fingerprint에 속한 base state를 symbolic parameter space로 올린 뒤, 하드웨어 제약을 만족하는 `SplitStep`/`PragmaStep` 파라미터 assignment를 생성하고, 이를 다시 concrete TVM state로 materialize하여 실제 lowered GPU code로 검증하는 과정이다.

이 정의에서 핵심 키워드는 세 가지다.

1. `sketch fingerprint`
   - 구조는 고정
2. `symbolic parameter space`
   - `sp_*`, `ur_*`
3. `materialize + verify`
   - 생성만으로 끝나지 않고 실제 TVM state와 lowered IR로 돌아감


## 65. 최종 정리

이 실험 코드의 schedule generation은 다음 층으로 나뉜다.

### 층 1. 구조 고정

`record_loader`가 split factor와 unroll 숫자를 무시하고 sketch fingerprint를 만든다.

즉, schedule 구조는 이미 선택되어 있다.

### 층 2. symbolic modeling

`build_symbolic_state()`가 해당 sketch를 `SymbolicState`로 바꾼다.

즉, 구조 안의 자유 파라미터가 `sp_*`, `ur_*`로 노출된다.

### 층 3. symbolic constraint solving

`ScheduleGenerator`가:

- constraint tree를 만들고
- 변수 순서를 정하고
- interval pruning / propagation / sampling 또는 DFS로
- 파라미터 값을 생성한다

### 층 4. concrete schedule materialization

`params_to_state()`가 generated params를 concrete TVM state로 되돌린다.

### 층 5. real-code verification

TVM이 실제 schedule을 lower하고, `verify_gpu_code`가 실제 GPU 제약을 확인한다.

즉, 이 시스템은 한 문장으로 말하면:

**"고정된 Ansor sketch를 symbolic parameter problem으로 바꾼 뒤, 하드웨어 제약을 만족하는 concrete schedule instance를 생성하는 파이프라인"**이다.
