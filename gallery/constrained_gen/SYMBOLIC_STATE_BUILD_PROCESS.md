# SymbolicState Build Process

이 문서는 `gallery/constrained_gen/modules` 안의 구현을 기준으로, `SymbolicState`가 어떻게 만들어지는지 처음부터 끝까지 정리한 설명서다. 설명 대상은 다음 코드다.

- `modules/param_manager.py`
- `modules/symbolic_state.py`
- `modules/transform_applier.py`
- `modules/sym_types.py`
- 필요할 때 참조한 TVM 본체 코드
  - `python/tvm/auto_scheduler/compute_dag.py`
  - `src/auto_scheduler/search_policy/sketch_policy.cc`
  - `src/auto_scheduler/transform_step.cc`
  - `include/tvm/auto_scheduler/transform_step.h`

문서의 목표는 단순히 "어디서 생성된다"를 말하는 것이 아니라, 아래 질문에 모두 답하는 것이다.

1. 입력으로 무엇을 받는가?
2. 초기 상태는 어떻게 구성되는가?
3. transform step을 replay하면서 어떤 필드가 어떻게 바뀌는가?
4. 왜 일부 extent는 `None`이 되었다가 나중에 복원되는가?
5. `sym_map`, `_split_sym_products`, `_cache_read_stencil_info` 같은 보조 메타데이터는 왜 필요한가?
6. 최종 `SymbolicState`는 이후 어떤 코드가 어떻게 소비하는가?


## 1. 한 줄 요약

`SymbolicState` 생성은 다음 두 단계로 이루어진다.

1. `compute_dag.ops`를 읽어서 stage/iter 뼈대를 복사한다.
2. 주어진 `auto_scheduler State`의 `transform_steps`를 순서대로 다시 적용하면서, split factor와 unroll 값을 심볼(`sp_*`, `ur_*`)로 치환하고, stage 구조 변경을 반영한다.

즉, 이 객체는 "TVM state의 구조를 symbolic parameter가 들어간 순수 Python 상태"로 재구성한 결과물이다.


## 2. 생성 진입점

실제 진입점은 `modules/param_manager.py`의 `build_symbolic_state`다.

```python
def build_symbolic_state(compute_dag, state):
    sym = SymbolicState(compute_dag)
    applier = TransformApplier(sym)
    applier.apply_steps(state)
    return sym
```

핵심 포인트는 세 가지다.

1. `SymbolicState(compute_dag)` 단계에서는 아직 transform history가 반영되지 않는다.
2. 실제 스케줄 구조는 `TransformApplier.apply_steps(state)`에서 만들어진다.
3. 반환되는 `sym`는 immutable snapshot이 아니라 mutable 객체다. 이후 `ScheduleGenerator`와 `verify_symbolic_state`가 이 객체의 `sym_map`을 계속 읽고 수정한다.


## 3. 입력 객체의 의미

### 3.1 `compute_dag`

`compute_dag`는 원본 연산 그래프다. 여기서 `SymbolicState`는 다음 정보를 복사한다.

- stage 순서
- 각 op의 이름
- 각 op의 output dtype
- spatial axis와 reduce axis의 이름 및 초기 extent

즉, "초기 stage/iterator 뼈대"는 `compute_dag`에서 나온다.

### 3.2 `state`

`state`는 TVM auto-scheduler가 가진 구체적인 스케줄 상태다. 중요한 정보는 `state.transform_steps`다.

이 안에는 예를 들어 다음 같은 step들이 순서대로 들어 있다.

- `SplitStep`
- `ReorderStep`
- `FuseStep`
- `AnnotationStep`
- `ComputeAtStep`
- `CacheReadStep`
- `CacheWriteStep`
- `PragmaStep`

`SymbolicState`는 이 step 배열을 "다시 재생"해서 최종 symbolic 상태를 만든다.


## 4. 최종 산출물의 데이터 모델

생성 과정을 이해하려면 먼저 `sym_types.py`와 `symbolic_state.py`의 타입을 이해해야 한다.

### 4.1 `SymExpr`

`SymExpr`는 symbolic extent 표현식 래퍼다.

- concrete 값이면 `SymExpr(16)`처럼 내부 값이 `int`
- symbolic 값이면 `SymExpr("sp_3_0")`처럼 내부 값이 `str`
- 복합 symbolic 식도 문자열로 보관
  - 예: `ceil(sp_3_0/(sp_3_1))`
  - 예: `min(sp_5_0, 64)`
  - 예: `(sp_7_0 - 1)*4 + sp_8_1`

주요 helper:

- `SymExpr.ceildiv(a, b)`
- `SymExpr.mul(a, b)`
- `SymExpr.product(items)`
- `SymExpr.min(a, b)`

이 타입이 중요한 이유는 split 후 루프 extent가 단순 정수가 아니라, 나중에 파라미터 값을 넣어 평가해야 하는 식이 되기 때문이다.

### 4.2 `eval_sym_extent`

`eval_sym_extent(expr, sym_map)`는 `SymExpr` 문자열 안의 심볼 이름을 `sym_map` 값으로 치환해서 실제 정수로 평가한다.

예를 들어:

- `expr = SymExpr("sp_3_0*sp_3_1")`
- `sym_map = {"sp_3_0": 4, "sp_3_1": 8}`

이면 결과는 `32`다.

중요한 점:

- `sym_map[name]`이 `None`이면 치환되지 않는다.
- 치환 후에도 평가가 안 되면 `"EVAL_FAIL(...)"` 문자열을 돌려준다.

즉, `SymbolicState`는 "구조"와 "식"을 만들고, 실제 수치 평가는 나중에 `sym_map` 값이 채워졌을 때 수행된다.

### 4.3 `SymIter`

`SymIter`는 iterator 1개를 표현한다.

- `name`
- `extent`: `SymExpr` 또는 `None`
- `annotation`: TVM iterator annotation code
- `iter_kind`: spatial/reduce/mixed/fused 구분용 정수

`extent is None`은 의미가 크다. 이것은 "현재 시점에서는 정확한 bound를 모른다"는 뜻이다. 특히 `compute_at`, `compute_root` 이후 자주 등장한다.

### 4.4 `SymStage`

`SymStage`는 stage 하나를 표현한다.

- `op_name`
- `op_type`: `compute` 또는 `placeholder`
- `iters`
- `compute_at`
- `auto_unroll_max_step`
- `storage_offset`
- `attach_stage_id`
- `attach_iter_id`
- `dtype`

`compute_at`는 아래 상수를 쓴다.

- `CA_ROOT = 0`
- `CA_INLINED = 1`
- `CA_ITER = 2`

즉, stage는 "어디에 attach되어 있는지"까지 포함해서 저장된다.

### 4.5 `SymbolicState` 필드 전체

`SymbolicState.__init__`가 만드는 주요 필드는 아래와 같다.

- `stages`
  - 최종 symbolic schedule 구조
- `sym_map`
  - symbolic parameter 이름과 현재 값의 매핑
- `compute_dag`
  - 원본 DAG 참조
- `_state`
  - `apply_steps`가 받은 TVM state 원본
- `_ca_saved_extents`
  - `compute_at`/`compute_root` 전에 있던 symbolic extent 백업
- `_split_sym_products`
  - split step별 factor 곱
- `_cache_read_consumer`
  - cache read stage와 consumer stage 연결 정보
- `_cache_read_stencil_info`
  - cache read index 분석 결과
- `_shared_fused_extents`
  - shared stage의 fused extent

이 중 앞의 셋만 있어도 "단순한 symbolic 상태"는 만들 수 있다. 그러나 실제 구현은 이후 제약 생성과 extent 복원까지 해야 하므로 나머지 메타데이터가 필요하다.


## 5. 생성 파이프라인 전체 흐름

아래 흐름이 실제 구현의 전체 그림이다.

```text
build_symbolic_state(compute_dag, state)
  -> SymbolicState(compute_dag)
       -> compute_dag.ops를 읽어 초기 stages 구성
  -> TransformApplier(sym)
  -> applier.apply_steps(state)
       -> sym._state = state 저장
       -> state.transform_steps 순회
       -> step type별 핸들러 호출
       -> sym.stages / sym.sym_map / 메타데이터 갱신
       -> _infer_bound_final(state) 실행
  -> sym 반환
```

이 과정을 세 부분으로 나누면 아래와 같다.

1. 초기 stage/iter skeleton 생성
2. transform history replay
3. final bound 보정


## 6. 1단계: `SymbolicState.__init__`에서 하는 일

### 6.1 기본 필드 초기화

생성자는 먼저 비어 있는 컨테이너를 만든다.

- `self.stages = []`
- `self.sym_map = OrderedDict()`
- `self.compute_dag = compute_dag`
- 여러 보조 메타데이터 dict

`sym_map`이 `OrderedDict`인 이유는 이후 split/unroll 심볼이 step 순서대로 삽입되어, 파라미터 열거 순서가 안정적으로 유지되기 쉽기 때문이다.

### 6.2 `compute_dag.ops` 순회

생성자는 `for sid, op in enumerate(compute_dag.ops)`를 돌며 stage를 하나씩 만든다.

분기 기준은 `hasattr(op, "axis")`다.

- `axis`가 있으면 compute op
- 없으면 placeholder op

#### compute op인 경우

다음을 수행한다.

1. dtype 추출
2. `op.axis`를 돌며 spatial iterator 생성
3. `op.reduce_axis`를 돌며 reduce iterator 생성
4. `SymStage(op.name, "compute", iters, dtype=...)` 추가

각 axis는 다음 정보로 `SymIter`가 된다.

- 이름: `axis.var.name`
- extent: `_safe_int_extent(axis.dom.extent)`
- annotation: `0`
- iter_kind
  - spatial axis면 `0`
  - reduce axis면 `1`

즉, 초기 상태에서 annotation은 모두 기본값이고, split/fuse/reorder도 아직 반영되지 않는다.

#### placeholder op인 경우

placeholder는 루프 축이 없으므로 빈 iterator 리스트로 stage만 만든다.

```python
SymStage(op.name, "placeholder", [], dtype=dtype)
```

### 6.3 `_safe_int_extent`

초기 extent는 `_safe_int_extent`를 통해 정수로 바꾸려 시도한다.

동작은 다음과 같다.

1. `int(extent_expr)` 시도
2. `TypeError`가 나면 `tvm.arith.Analyzer().simplify(extent_expr)` 수행
3. 다시 `int(simplified)` 시도

즉, 초기 DAG의 axis extent가 간단한 산술식이어도 simplify 후 정수로 만들 수 있으면 concrete `SymExpr(int)`로 저장한다.

### 6.4 이 단계가 끝났을 때의 상태

이 시점의 `SymbolicState`는 아직 "원본 DAG 복제본"에 가깝다.

- split되지 않은 축만 있다
- annotation은 모두 기본값이다
- `sym_map`은 비어 있다
- compute_at/cache 관련 정보도 비어 있다

즉, 이 단계만으로는 스케줄 search state를 표현하지 못한다.


## 7. 2단계: `TransformApplier.apply_steps`

실제 symbolic state 생성의 핵심은 여기다.

### 7.1 state 원본 보관

`apply_steps`는 시작하자마자 아래를 수행한다.

```python
self.s._state = state
steps = state.transform_steps
```

이 원본을 저장하는 이유는 이후 두 곳에서 필요하기 때문이다.

1. `_restore_stage_extents_if_needed`
2. `SymParamManager._build_sp_extents`

즉, symbolic state는 단순 결과물만 저장하는 것이 아니라, 원본 TVM step 배열에도 다시 접근한다.

### 7.2 step dispatch

각 step마다 `type_key.split(".")[-1]`로 실제 step 종류를 얻고, 다음 핸들러로 분기한다.

| Step type | 처리 함수 | 핵심 역할 |
| --- | --- | --- |
| `AnnotationStep` | `_apply_annotation` | iterator annotation 변경 |
| `FuseStep` | `_apply_fuse` | 여러 iter를 하나로 합침 |
| `PragmaStep` | `_apply_pragma` | unroll pragma 또는 debug_skip_region 반영 |
| `ReorderStep` | `_apply_reorder` | iter 순서 재배치 |
| `SplitStep` | `_apply_split` | 새 split 변수 생성 |
| `FollowSplitStep` | `_apply_follow_split` | 기존 split의 factor를 따라감 |
| `FollowFusedSplitStep` | `_apply_follow_fused_split` | 여러 split factor의 곱을 따라감 |
| `StorageAlignStep` | `_apply_storage_align` | storage offset 기록 |
| `ComputeAtStep` | `_apply_compute_at` | attach 관계 설정, 기존 bound 제거 |
| `ComputeInlineStep` | `_apply_compute_inline` | inline 표시 |
| `ComputeRootStep` | `_apply_compute_root` | root로 올리면서 bound 제거 |
| `CacheReadStep` | `_apply_cache_read` | 새 cache read stage 삽입 |
| `CacheWriteStep` | `_apply_cache_write` | 새 cache write stage 삽입 |

이 목록에 없는 step은 현재 구현상 warning만 출력한다. 따라서 이 symbolic 시스템은 "위 표에 포함된 step 집합"을 전제로 한다.

### 7.3 마지막 final infer

모든 step을 적용한 뒤 `_infer_bound_final(state)`를 호출한다.

이 단계는 중간에 `None`으로 남아 있는 extent를 가능한 한 채운다.


## 8. Step별 상세 동작

이 절은 생성 과정에서 가장 중요하다. 각 step이 `stages`, `sym_map`, 메타데이터를 어떻게 바꾸는지 정리한다.

### 8.1 `AnnotationStep`

동작은 단순하다.

```python
self.s.stages[step.stage_id].iters[step.iter_id].annotation = int(step.annotation)
```

즉, 구조는 바뀌지 않고 annotation만 바뀐다.

이 정보는 이후 다음 함수들이 읽는다.

- `get_vectorize_extents`
- `get_thread_extents`
- `get_vthread_extents`

따라서 annotation step은 제약 생성의 입력이 된다.

### 8.2 `FuseStep`

`FuseStep`는 여러 iterator를 하나의 iterator로 합친다.

핵심 절차:

1. fused 대상 iter id 목록을 읽는다.
2. 필요하면 `_restore_stage_extents_if_needed`로 현재 stage의 `None` extent를 먼저 복원한다.
3. 새 iter 이름을 `"a@b@c@"` 형태로 만든다.
4. fused iter extent를 원래 extent들의 곱으로 만든다.
5. 대상 구간을 새 iterator 하나로 치환한다.
6. attach 정보가 있으면 `attach_iter_id`를 새 위치에 맞게 조정한다.

세부 포인트:

- fused 대상 extent가 모두 정의되어 있으면 `SymExpr.mul`로 곱을 만든다.
- 하나라도 `None`이면 새 fused iter extent도 `None`이 된다.
- iter_kind가 섞이면 새 iter_kind를 `2`로 둔다.
- stage 이름에 `.shared`가 포함되고 fused extent를 알 수 있으면 `_shared_fused_extents[sid]`에 저장한다.

마지막 항목이 중요하다. 이후 `ScheduleGenerator.build_shared_memory_constraints`는 shared stage의 fused extent를 읽어서 shared memory 사용량 제약을 만든다.

### 8.3 `PragmaStep`

현재 구현은 두 종류만 특별 취급한다.

#### `auto_unroll_max_step$N`

예:

- pragma 문자열: `auto_unroll_max_step$512`
- step index: `17`

그러면 아래가 수행된다.

```python
sym_name = "ur_17"
s.sym_map[sym_name] = 512
s.stages[sid].auto_unroll_max_step = SymExpr("ur_17")
```

즉:

- 실제 값은 `sym_map["ur_17"] = 512`
- stage 필드에는 `SymExpr("ur_17")`

이렇게 분리해 두었기 때문에 나중에 `ScheduleGenerator`가 unroll 값을 다른 후보로 바꿔 넣을 수 있다.

#### `debug_skip_region`

이 경우 현재 stage를 root로 되돌린다.

- `compute_at = CA_ROOT`
- `attach_stage_id = None`
- `attach_iter_id = None`

### 8.4 `ReorderStep`

`after_ids` 순서대로 iterator 배열을 다시 만든다.

추가로, 어떤 다른 stage가 이 stage의 특정 iter에 attach되어 있으면, reorder 후 새로운 iter index로 `attach_iter_id`를 업데이트한다.

즉, reorder는 단순히 stage 내부 순서만 바꾸는 것이 아니라 attach map의 논리적 의미도 보존하려고 한다.

### 8.5 `SplitStep`

이 step이 symbolic state 생성의 핵심이다.

#### 8.5.1 어떤 값이 심볼이 되나

`SplitStep`의 `lengths`를 순회하면서 각 factor마다 새 symbolic 이름을 만든다.

규칙:

- `sp_{step_idx}_{length_idx}`

예:

- step index가 `3`
- `lengths = [4, None, 8]`

이면:

- `sp_3_0`
- `sp_3_1`
- `sp_3_2`

가 생성된다.

그리고 `sym_map`에 아래처럼 들어간다.

- `sp_3_0 -> 4`
- `sp_3_1 -> None`
- `sp_3_2 -> 8`

즉, `None` factor도 심볼 이름은 생긴다. 값만 아직 정해지지 않은 상태다.

#### 8.5.2 원래 iter extent 확보

split 전에 원래 iterator extent가 필요하다. 구현은 우선순위를 이렇게 둔다.

1. `orig_iter.extent`가 있으면 그것 사용
2. 없고 `step.extent`가 있으면 `SymExpr(int(step.extent))`
3. 둘 다 없으면 `None`

그리고 이 함수는 split 직전에 `_restore_stage_extents_if_needed`를 부른다. 따라서 `compute_at` 등으로 인해 임시로 `None`이었던 extent도 가능한 경우 먼저 채워진 뒤 split이 수행된다.

#### 8.5.3 split factor 곱 저장

모든 symbolic factor를 곱한 식을 `_split_sym_products[(sid, step_idx)]`에 저장한다.

이 값은 이후 두 군데에서 중요하다.

1. cache read extent 복원
2. consumer split과 cache stage 축을 연결할 때

#### 8.5.4 새 iter 생성 규칙

split 결과 iterator는 항상 `len(lengths) + 1`개다.

예를 들어 원래 iter 이름이 `i`이고 factor가 두 개면 결과 이름은 대체로 아래와 비슷하다.

- `i.0`
- `i.1`
- `i.2`

구체적인 생성 방식은 `inner_to_outer` 여부에 따라 달라진다.

##### `inner_to_outer=True`

안쪽 factor부터 밖으로 벗겨내는 방식이다. 구현은 뒤에서부터 factor를 읽는다.

반복마다:

1. 현재 factor 심볼 `f`
2. 새 iter extent = `min(f, tosplit_extent)`
3. 남은 extent = `ceildiv(tosplit_extent, 새 iter extent)`

마지막에 residual iterator를 `name.0`으로 추가하고, 전체 배열을 reverse한다.

##### `inner_to_outer=False`

바깥쪽부터 `nparts`처럼 나누는 방식이다.

반복마다:

1. 현재 factor 심볼 `f`
2. 새 iter extent = `min(f, tosplit_extent)`
3. 남은 extent = `ceildiv(tosplit_extent, 새 iter extent)`

마지막 residual iterator는 `name.{len(lengths)}`로 붙는다.

#### 8.5.5 attach index 보정

원래 iter 하나가 여러 iter로 늘어났기 때문에, 이 stage에 attach된 다른 stage들의 `attach_iter_id`를 뒤로 밀어야 한다.

현재 구현은 `shift = len(lengths)`만큼 뒤 iter id를 증가시킨다.

### 8.6 `FollowSplitStep`

이 step은 독립적인 새 factor를 만들지 않고, 과거 `SplitStep`의 factor를 "따라가는" 방식이다.

구현 절차:

1. `src_step_id`로 원본 `SplitStep`를 찾는다.
2. 원본 split의 factor 중 앞의 `n_split - 1`개를 그대로 쓴다.
3. 남은 factor들은 모두 곱해서 마지막 factor 하나로 합친다.
4. 각 factor를 새 심볼로 만들지 않고, 기존 심볼 이름(`sp_{src_step_id}_k`)을 재사용한다.

즉, 이 step은 `sym_map`에 새 변수 개수를 늘리지 않는다. 이미 있는 split 변수와 식만 재조합한다.

이 동작은 TVM C++의 `FollowSplitStepNode::ExtractSplitLengths`와 대응된다. C++도 앞 factor를 그대로 가져오고, 마지막 factor는 나머지 factor의 곱으로 만든다.

### 8.7 `FollowFusedSplitStep`

이 step은 여러 source split step의 "같은 level factor"를 곱해서 하나의 fused factor로 만든다.

예를 들어:

- `src_step_ids = [10, 11]`
- `level = 1`

이면 symbolic 식은:

- `sp_10_1*sp_11_1`

이 된다.

그리고 `factor_or_nparts`에 따라 두 가지로 해석한다.

- `True`
  - inner extent = fused factor
  - outer extent = `ceildiv(total, fused factor)`
- `False`
  - outer extent = fused factor
  - inner extent = `ceildiv(total, fused factor)`

이 step 역시 새 심볼을 만들지 않고 기존 split 심볼을 식으로 조합한다.

### 8.8 `StorageAlignStep`

구조 변경은 하지 않고, 해당 stage의 `storage_offset`만 기록한다.

```python
self.s.stages[step.stage_id].storage_offset = step.offset
```

### 8.9 `ComputeAtStep`

이 step은 symbolic state 생성에서 특히 중요하다. TVM 본체도 `compute_at` 이후 bound 정보가 정확하지 않을 수 있으니 `InferBound`가 필요하다고 명시한다.

현재 Python 구현은 다음을 수행한다.

1. 현재 stage의 모든 iter를 순회
2. extent가 symbolic이고 concrete가 아니면 `_ca_saved_extents[(sid, iid)]`에 백업
3. 모든 iter의 `extent = None`
4. `compute_at = CA_ITER`
5. `attach_stage_id = target_sid`
6. `attach_iter_id = target_iid`

즉, 이 시점 이후 해당 stage는 "attach 정보는 알지만 정확한 루프 bound는 모르는 상태"가 된다.

왜 굳이 기존 symbolic extent를 `_ca_saved_extents`에 저장하는가?

이유는 나중에 bound를 복원할 때, 완전히 concrete 값으로 덮어쓰지 말고 가능하면 원래의 symbolic 의미를 살려야 하기 때문이다.

예를 들어 split으로 만들어진 symbolic extent가 compute_at 때문에 잠시 사라졌다면, 최종 복원 시 `37` 같은 concrete 숫자보다 `sp_5_1` 같은 symbolic 식을 유지하는 편이 이후 제약 생성에 훨씬 유용하다.

### 8.10 `ComputeInlineStep`

이 step은 단순하다.

- `compute_at = CA_INLINED`
- attach 정보 제거

inline된 stage는 이후 extent 수집 함수들이 기본적으로 건너뛴다.

### 8.11 `ComputeRootStep`

`ComputeAtStep`와 거의 동일하게 동작하되 root로 올린다.

1. non-concrete symbolic extent를 `_ca_saved_extents`에 백업
2. 모든 iter extent를 `None`으로 비움
3. `compute_at = CA_ROOT`
4. attach 정보 제거

즉, compute_root도 symbolic loop bound를 일시적으로 잃는 연산으로 취급된다.

### 8.12 `CacheReadStep`

이 step은 stage 개수 자체를 바꾼다. 따라서 생성 과정에서 가장 복잡한 축에 속한다.

#### 8.12.1 partial replay로 실제 stage 확인

구현은 TVM global function `auto_scheduler.ReplayStepsPartial`를 호출한다.

```python
ps_after = ReplayStepsPartial(compute_dag, state, step_idx + 1)
```

TVM C++ 구현을 보면 이 함수는:

1. `dag->init_state`에서 시작
2. 앞의 `num_steps`개 step만 적용
3. 마지막에 `dag.InferBound(new_state)` 수행

즉, "현재 step까지 적용된 bounded state"를 돌려준다.

이게 왜 중요하냐면, cache read는 실제로 새 stage를 삽입하므로 Python symbolic 코드만으로 새 stage의 정확한 축/extent를 재구성하기 어렵다. 그래서 TVM이 계산한 partial state를 참조한다.

#### 8.12.2 새 stage 삽입

현재 구현은 `added_stage_id = sid + 1`을 새 stage 위치로 가정한다.

그다음 `ps_after.stages[added_stage_id]`에서 실제 stage 정보를 읽어 새 `SymStage`를 만든 뒤 `s.stages.insert(added_stage_id, new_sym_stage)` 한다.

즉:

- 원래 target stage는 그대로 유지
- 그 바로 뒤에 cache read stage가 하나 추가

#### 8.12.3 stage id shift

새 stage가 끼어들었기 때문에, 기존 metadata key의 stage id를 전부 재정렬해야 한다.

그래서 `_shift_ca_saved_extents(added_stage_id)`를 호출한다.

이 함수는 아래 dict들의 key를 한 칸씩 민다.

- `_ca_saved_extents`
- `_split_sym_products`
- `_cache_read_consumer`
- `_cache_read_stencil_info`
- `_shared_fused_extents`

이 처리가 없으면 이후 metadata가 모두 잘못된 stage를 가리키게 된다.

#### 8.12.4 consumer 정보 저장

`step.reader_stage_ids`가 있으면 첫 번째 reader stage를 consumer로 저장한다.

```python
s._cache_read_consumer[added_stage_id] = consumer_sid
```

이 정보는 나중에 cache read stage의 extent를 "consumer split 심볼"과 연결할 때 사용된다.

#### 8.12.5 stencil 분석

`_analyze_cache_read_stencil`은 consumer op의 body를 직접 읽어서 producer load의 인덱스 식을 분석한다.

예를 들어 인덱스가 아래처럼 생겼는지 본다.

- `spatial_var`
- `reduce_var`
- `spatial_var * stride + reduce_var`
- `spatial_var + reduce_var`

분석 결과는 axis별 `(stride, sp_order, rd_order)`로 저장된다.

이 metadata의 목적은 cache read stage extent를 단순 숫자가 아니라, 예를 들어

- `sp_10_1`
- `(sp_10_0 - 1)*4 + sp_11_1`

같은 symbolic 식으로 다시 복원하기 위함이다.

#### 8.12.6 attach stage id 보정

이미 어떤 stage가 compute_at로 attach되어 있었고, 그 attach target stage id가 새 stage 뒤에 있었다면 한 칸 밀어야 한다.

구현은 모든 stage를 돌며 `attach_stage_id >= added_stage_id`이면 `+1` 한다.

#### 8.12.7 stage 이름 동기화

partial replay 결과에서 이후 stage들의 실제 op name이 바뀔 수 있으므로, `ps_after.stages[i].op.name`을 읽어 symbolic stage 이름도 맞춘다.

### 8.13 `CacheWriteStep`

cache write도 stage 삽입 step이다.

현재 Python 구현은 `sid` 위치에 새 `SymStage`를 삽입한다. 즉, cache write stage가 원래 stage 앞에 추가되는 모델이다.

핵심 절차:

1. `ReplayStepsPartial(..., step_idx + 1)`로 현재 step까지의 실제 bounded state 확보
2. `ps_after.stages[sid]`에서 새 stage 정보를 읽음
3. `s.stages.insert(sid, new_sym_stage)` 수행
4. `_shift_ca_saved_extents(sid)` 호출
5. attach된 다른 stage의 `attach_stage_id`가 `sid` 이상이면 `+1`
6. 전체 stage 이름을 `ps_after` 기준으로 다시 동기화

주의할 점:

- TVM C++ `CacheWriteStepNode::ApplyToState`는 multi-output 예외 때문에 `added_ops`가 `1` 이상일 수 있다고 적고 있다.
- 현재 Python symbolic 구현은 이 C++의 모든 세부 분기를 그대로 재현하지는 않고, symbolic 관점에서 필요한 stage 삽입과 이름 동기화에 집중한다.

따라서 문서를 읽을 때는 "TVM 원본 state를 byte-by-byte 재현한다"보다 "후속 symbolic 제약 생성에 필요한 수준으로 구조와 extent를 추상화한다"라고 이해하는 것이 정확하다.


## 9. 왜 extent가 중간에 `None`이 되고, 어떻게 다시 살아나는가

이 시스템의 난점은 `compute_at`, `compute_root`, cache 관련 step들 때문에 어떤 시점에는 iterator extent를 정확히 알 수 없다는 점이다.

이를 해결하는 장치가 두 개 있다.

1. `_restore_stage_extents_if_needed(stage_id, step_idx)`
2. `_infer_bound_final(state)`

### 9.1 `_restore_stage_extents_if_needed`

이 함수는 다음 조건에서 호출된다.

- `FuseStep` 직전
- `SplitStep` 직전
- `FollowSplitStep` 직전
- `FollowFusedSplitStep` 직전

즉, "지금 바로 extent가 필요하지만 stage 내부에 `None`이 남아 있는 경우"에 lazy 복원을 수행한다.

절차는 아래와 같다.

1. `ReplayStepsPartial(compute_dag, _state, step_idx)` 호출
2. partial bounded state에서 같은 `stage_id`의 실제 iter extents를 읽음
3. 각 `None` iter에 대해 가능한 한 symbolic 표현으로 복원 시도

복원 우선순위는 대략 이렇다.

1. `_ca_saved_extents`에 저장된 예전 symbolic extent가 있으면 그것 사용
2. cache read stage라면 consumer split symbolic product와 stencil 정보로 symbolic 매칭 시도
3. compute_at된 stage라면 target stage의 inner iter 중 같은 evaluated extent를 갖는 non-concrete symbolic extent를 찾아 재사용
4. 다 실패하면 concrete `SymExpr(real_ext)`로 채움

이 함수의 핵심 철학은 "가능하면 symbolic 의미를 보존하고, 정말 안 되면 concrete 숫자로라도 채운다"이다.

### 9.2 `_infer_bound_final`

모든 step replay가 끝난 뒤 마지막으로 전체 stage를 훑으면서 `None` extent를 다시 채운다.

기본 아이디어는 `_restore_stage_extents_if_needed`와 거의 같다. 차이는:

- 특정 stage 하나가 아니라 전체 stage를 대상으로 한다.
- 최종 state 전체를 `compute_dag.infer_bound_from_state(...)`에 넣어 bounded state를 얻는다.

TVM Python 래퍼 `ComputeDAG.infer_bound_from_state` 문서도 다음을 보장한다.

- compute_at 등으로 잃어버린 bound 정보를 채워준다.
- 반환된 state는 complete iterator extent 정보를 가진다.

즉, `SymbolicState`는 중간 step replay 동안에는 불완전할 수 있지만, 최종 반환 시점에는 가능한 한 extent가 채워진 상태가 되도록 설계돼 있다.


## 10. cache read symbolic 복원의 내부 논리

이 부분은 구현 의도를 이해하는 데 중요하다.

### 10.1 왜 cache read는 별도 분석이 필요한가

cache read stage는 단순히 "consumer split factor를 복사한 stage"가 아니다. 실제 extent는 consumer가 producer를 어떤 패턴으로 읽는지에 따라 달라진다.

예를 들어:

- `A[i, k]`를 읽으면 cache read 축이 spatial/reduce 축과 거의 직접 대응한다.
- `A[i * 4 + k]`처럼 stencil이 있으면 extent는 단순 factor 하나가 아니라 `(spatial_extent - 1) * stride + reduce_extent` 형태가 된다.

그래서 구현은 consumer op body의 `tir.ProducerLoad`를 내려가며 읽기 인덱스 식을 분석한다.

### 10.2 저장되는 정보

`_cache_read_stencil_info[cr_stage_id]`는 대략 다음 구조다.

```text
{
  cr_axis_idx: (stride, sp_order, rd_order),
  ...
}
```

의미:

- `stride`
  - `0`이면 단순히 spatial 또는 reduce 중 하나만 직접 대응
  - 양수이면 `spatial * stride + reduce` 패턴
- `sp_order`
  - consumer spatial axis 순서
- `rd_order`
  - consumer reduce axis 순서

### 10.3 복원 시 사용법

복원 함수는 consumer stage의 `_split_sym_products`를 읽어 ordered split symbolic product 목록을 만든다.

그 뒤:

- stride가 `0`이면 해당 순서의 split symbolic factor를 직접 사용
- stride가 양수면 `(sp_sym - 1) * stride + rd_sym` 형태 식을 만든다
- 그래도 안 맞으면 evaluated extent가 같은 symbolic product를 매칭한다

즉, cache read axis extent를 단순 concrete 숫자로 잃어버리지 않도록 매우 공격적으로 symbolic 역추적을 수행한다.


## 11. `sym_map`이 생성되는 규칙

`sym_map`은 symbolic state의 핵심 파라미터 저장소다.

### 11.1 split 계열

`SplitStep`에서 생성되는 이름:

- `sp_{step_idx}_{length_idx}`

예:

- `sp_5_0`
- `sp_5_1`
- `sp_12_2`

초기값:

- 원래 state에 concrete length가 있으면 그 정수
- 없으면 `None`

### 11.2 unroll 계열

`PragmaStep(auto_unroll_max_step$N)`에서 생성되는 이름:

- `ur_{step_idx}`

초기값:

- pragma 안의 concrete 값

### 11.3 follow split 계열은 새 변수 생성 안 함

`FollowSplitStep`, `FollowFusedSplitStep`는 기존 `sp_*` 이름을 조합해서 식만 만든다.

즉, 실제 파라미터 차원 수를 늘리는 step은 아니다.

### 11.4 왜 값이 `None`일 수 있나

이 시스템은 한 sketch 안의 "구조는 같고 파라미터만 다른" 여러 state를 표현하려는 목적이 있다. 따라서 생성 시점의 base state에서 비어 있거나 바뀔 여지가 있는 factor는 `None`으로 두고 symbolic 이름만 유지하는 편이 맞다.

이후:

- `verify_symbolic_state`가 다른 concrete state 값으로 잠시 채울 수 있고
- `ScheduleGenerator`가 하드웨어 제약을 만족하는 새 값으로 채울 수 있다


## 12. 생성 결과가 만족해야 하는 불변식

`build_symbolic_state`가 끝난 후 기대하는 상태는 아래와 같다.

1. `stages`의 개수와 순서는 최종 schedule 구조와 가능한 한 맞아야 한다.
2. 각 stage의 `iters`는 split/fuse/reorder/attach가 반영된 최종 순서를 가져야 한다.
3. split/unroll 파라미터는 concrete 값이 아니라 symbolic 이름으로 stage 내부 식에 연결돼 있어야 한다.
4. `sym_map`에 값을 넣으면 `eval_sym_extent`로 실제 extent 계산이 가능해야 한다.
5. vectorize/thread/vthread/shared memory 제약을 읽는 helper가 동작해야 한다.

이 객체는 이후 constraint solver 역할의 `ScheduleGenerator`가 직접 읽기 때문에, 단순 출력용 구조체가 아니라 "계산 가능한 symbolic model"이어야 한다.


## 13. 생성 직후 이 객체를 누가 어떻게 쓰는가

### 13.1 `verify_symbolic_state`

`modules/param_manager.py`의 `verify_symbolic_state`는 다음 방식으로 검증한다.

1. 현재 `sym_state.sym_map`을 백업
2. 비교 대상 concrete `state.transform_steps`를 읽어
   - `SplitStep` 길이를 `sp_*`에 대입
   - `PragmaStep` unroll 값을 `ur_*`에 대입
3. `task.compute_dag.infer_bound_from_state(state)` 실행
4. real bounded state와 symbolic state를 비교
   - stage 개수
   - iter 개수
   - iterator 이름
   - annotation
   - evaluated extent
5. 끝나면 `sym_map` 원복

중요한 해석:

- `SymbolicState`는 특정 concrete state 하나를 위한 전용 객체가 아니다.
- 같은 sketch에 속한 여러 concrete state를 설명하는 공통 symbolic skeleton으로 쓰인다.

실제로 `verify.ipynb`에서도 sketch별 첫 state로 `sym_state`를 한 번 만들고, 같은 sketch의 다른 state들을 여기에 대입해 검증한다.

### 13.2 `ScheduleGenerator`

`modules/schedule_generator.py`는 `SymbolicState`를 입력으로 받아 하드웨어 제약을 만든다.

읽는 정보 예:

- `get_vectorize_extents()`
- `get_thread_extents()`
- `get_vthread_extents()`
- `get_shared_memory_extents()`
- `sym_map`
- `_split_sym_products` 간접 정보

즉, symbolic state 생성 단계에서 annotation, shared fused extent, split symbolic 식이 정확히 보존되지 않으면 이후 파라미터 생성이 성립하지 않는다.

### 13.3 `params_to_state`

새로 만든 파라미터 dict를 TVM state로 다시 바꾸는 함수는 `modules/tvm_verify.py`의 `params_to_state`다.

이 함수는:

- `sp_*` 이름을 JSON record 안의 `SplitStep.lengths` 위치에 되써 넣고
- `ur_*` 이름을 `PragmaStep` 문자열에 되써 넣는다

즉, `SymbolicState`에서 쓰는 이름 규약이 record patching 규약과 정확히 맞아야 전체 파이프라인이 닫힌다.


## 14. 실전에서 보통 어떻게 생성하는가

저장소 안의 `verify.ipynb` 흐름을 기준으로 보면 보통 다음 순서를 따른다.

1. 측정 record들을 workload/sketch 기준으로 그룹핑
2. sketch별 첫 번째 concrete state를 하나 고름
3. 그 state로 `build_symbolic_state(task.compute_dag, base_state)` 수행
4. 같은 sketch의 다른 state들에 대해 `verify_symbolic_state`로 공통 symbolic skeleton인지 확인
5. 검증이 되면 `ScheduleGenerator(sym_state)`로 새로운 파라미터 조합 생성

즉, 이 시스템의 전제는 "한 sketch 안에서는 step 구조가 같고 파라미터만 다르다"이다.


## 15. 주의해야 할 구현상의 포인트

### 15.1 `SymbolicState`는 deep copy된 TVM state가 아니다

이 객체는 TVM `State`를 1:1로 래핑하지 않는다.

- Python 객체로 재구성한 stage/iter 모델이다.
- 일부 정보는 symbolic 식으로 대체한다.
- 일부는 metadata dict로 따로 들고 간다.

즉, 목적은 "정확한 TVM 내부 상태 재현"이 아니라 "스케줄 파라미터를 symbolic하게 다루기 위한 모델링"이다.

### 15.2 extent는 항상 즉시 완전하지 않다

중간 step replay 도중에는 `None`이 생기는 것이 정상이다.

- `compute_at`
- `compute_root`
- cache 관련 stage 삽입 직후

따라서 이 구현은 lazy restore와 final infer를 반드시 포함해야 한다.

### 15.3 지원되지 않는 step 타입이 오면 warning만 뜬다

`apply_steps`에 없는 step이 들어오면 현재는 아래처럼 끝난다.

```python
print(f"  [WARN] Unhandled step type: {tk}")
```

즉, 새 sketch나 새 TVM 기능을 다룰 때는 여기서 누락되는 step이 없는지 먼저 확인해야 한다.

### 15.4 `cache_write`는 TVM 원본 구현보다 단순화된 부분이 있다

TVM C++는 multi-output 예외를 처리한다. 현재 Python symbolic 구현은 symbolic 목적에 맞는 수준으로 새 stage 삽입과 이름 동기화를 수행한다.

따라서 multi-output cache_write가 많은 workload를 다룰 때는 이 부분이 실제 sketch와 얼마나 잘 맞는지 별도 검증이 필요하다.


## 16. 이해를 위한 정신 모델

이 시스템을 가장 쉽게 이해하는 방법은 `SymbolicState`를 세 층으로 보는 것이다.

### 층 1. 구조 복제

`compute_dag`에서 기본 stage와 iterator를 복사한다.

질문:

- 원래 어떤 op들이 있었나?
- 각 op는 spatial/reduce 축을 몇 개 가졌나?

### 층 2. symbolic replay

`transform_steps`를 순서대로 재생하면서 구조를 바꾼다.

질문:

- 어떤 iterator가 split/fuse/reorder 되었나?
- 어떤 값이 tunable parameter인가?
- stage가 어디에 attach되었나?
- cache stage가 추가되었나?

### 층 3. bound recovery

중간에 잃어버린 extent를 TVM의 `InferBound` 결과와 메타데이터를 이용해 다시 symbolic하게 복원한다.

질문:

- 현재 루프 bound를 알 수 있는가?
- 가능하면 symbolic 의미를 보존할 수 있는가?
- 안 되면 concrete 수치라도 채울 수 있는가?

이 세 층이 합쳐져야 비로소 `ScheduleGenerator`가 읽을 수 있는 symbolic schedule model이 완성된다.


## 17. 결론

`SymbolicState` 생성은 단순 생성자 호출이 아니다. 실제로는 아래 네 가지를 동시에 수행하는 파이프라인이다.

1. 원본 `ComputeDAG`로부터 stage/iter skeleton 복사
2. TVM `State.transform_steps`를 replay하여 최종 스케줄 구조 반영
3. split/unroll 같은 concrete 파라미터를 symbolic 이름으로 승격
4. `InferBound`, saved extent, cache stencil 분석을 이용해 잃어버린 bound를 symbolic하게 복원

그래서 최종 산출물은 "보기 좋은 문자열 출력용 객체"가 아니라, 다음 작업을 위한 핵심 중간 표현이 된다.

- 같은 sketch의 다른 concrete state 검증
- 하드웨어 제약 기반 파라미터 생성
- 생성된 파라미터를 다시 TVM state로 patch

정리하면, 이 저장소에서 `SymbolicState`는 "Ansor sketch를 symbolic parameter space로 들어올리는 핵심 어댑터"다.
