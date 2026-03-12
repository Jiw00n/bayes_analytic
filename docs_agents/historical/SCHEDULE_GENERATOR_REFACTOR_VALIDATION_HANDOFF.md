# Schedule Generator Refactor + Validation Handoff

## Historical Status

This file is a historical handoff record.

- Do not use this file as the startup document for new Codex sessions.
- Use `CODEX_WORKING_CONTEXT.md` for the current working model.
- Treat every statement here as date-bound and re-check it against current code before acting on it.

## 목적

이 문서는 `gallery/constrained_gen`의 최근 생성기 리팩터링과 전수 검증 결과를 다른 세션의 에이전트가 바로 이어받을 수 있도록 정리한 handoff 문서다.

핵심 범위는 두 가지다.

1. `ScheduleGenerator`를 역할별 객체로 분리한 현재 구조
2. 그 상태에서 수행한 전수 검증 결과와 아직 남아 있는 문제


## 현재 코드 구조

### 중심 facade

- `gallery/constrained_gen/modules/schedule_generator.py`

현재 `ScheduleGenerator`는 facade 역할로 정리되어 있다.

- 생성기 공용 API
- constraint/hint 포맷팅
- concrete final checker
- 위 3개를 제외한 실질 로직은 아래 4개 객체에 위임

### 분리된 4개 객체

- `gallery/constrained_gen/modules/constraint_set.py`
  - exact/projected GPU constraint build
  - symbolic constraint bundle build
  - preprocess
  - pruning-level constraint checks

- `gallery/constrained_gen/modules/var_order_planner.py`
  - grid-loop 기반 variable order planning
  - phase entry build
  - `main compute anchor` 우선 배치

- `gallery/constrained_gen/modules/domain_propagator.py`
  - domain propagation
  - candidate filtering
  - prefix snapshot에서 leftover/resolved-false 분석

- `gallery/constrained_gen/modules/param_sampler.py`
  - `randomize_params`
  - `randomize_params_prefix`
  - `enumerate_all_params`

### facade에 남겨둔 것

아직 `ScheduleGenerator` 안에 남겨둔 것은 다음이다.

- constraint formatting
- raw/projected constraint pretty-print
- concrete final checker
  - `get_concrete_final_result`
  - `check_all_final`


## 이번 리팩터링에서 한 일

### 구조 정리

- `ScheduleGenerator`의 constraint build / var-order planning / domain propagation / param sampling 로직을 4개 모듈로 분리했다.
- `ScheduleGenerator` 하단에 남아 있던 구 구현은 wrapper 위임으로 정리했다.
- 현재 facade 쪽 wrapper는 새 컴포넌트만 호출한다.

### 생성 순서

현재 var-order planner는 grid-loop aware order를 쓴다.

- `main compute anchor` grid loop를 먼저 둔다.
- 나머지 grid loop는 기존 순서를 유지한다.
- 각 grid loop 내부 phase 순서는 다음이다.
  1. `pure_product_max_threads`
  2. `pure_product_max_vthread`
  3. `split_structure_max_threads`
  4. `split_structure_max_vthread`
  5. `scaled_product_upper_bound`
  6. `non_product_direct_arm`
  7. `non_product_gate_vars`

vectorize는 early ordered phase에 넣지 않고 legacy tail에 남겨두었다.

### checker 분리

현재 checker는 의미상 두 층으로 나뉜다.

- pruning checker
  - `check_all_pruning`
  - projected / symbolic constraint 기반

- final checker
  - `check_all_final`
  - `base_input/base_result`가 있으면 concrete lowering + `verify_gpu_module` 기반
  - 없으면 exact/pruning path fallback

주의:

- `check_all()`은 아직 `check_all_pruning()` alias다.
- 따라서 생성 경로의 최종 accept는 아직 pruning semantics를 사용한다.


## 검증에 사용한 주요 스크립트

- `gallery/constrained_gen/validate_exact_gpu_constraints.py`
  - base sketch parameter를 그대로 넣고
  - pruning / final / concrete verifier를 비교

- `gallery/constrained_gen/validate_projected_gpu_generation.py`
  - random generation 결과를 concrete lowering + verifier로 확인
  - `--all-sketches` 지원 추가됨

- `gallery/constrained_gen/refresh_all_sketches_prefix_through_split_structure.py`
  - prefix snapshot 및 leftover summary 생성


## 전수 검증 결과

### Exact validation

artifact:

- `/tmp/projected_gpu_full_validation/final_validation_20260311_exact/exact_merged_summary.json`
- `/tmp/projected_gpu_full_validation/final_validation_20260311_exact/exact_merged_mismatches.jsonl`

요약:

- checked: `912`
- false_accept: `0`
- false_reject: `134`
- final_checker_mismatch: `0`

즉:

- final checker는 concrete verifier와 어긋나지 않았다.
- 하지만 pruning/exact symbolic path에는 아직 false reject가 많이 남아 있다.

root-cause aggregate:

- `custom_exact_lowering_mismatch`: `58`
- `symbolic_thread_binding_semantics_mismatch`: `53`
- `runtime_projection_upper_bound_insufficient`: `31`

root-cause combination count:

- `["custom_exact_lowering_mismatch"]`: `51`
- `["symbolic_thread_binding_semantics_mismatch"]`: `45`
- `["runtime_projection_upper_bound_insufficient"]`: `30`
- `["custom_exact_lowering_mismatch", "symbolic_thread_binding_semantics_mismatch"]`: `7`
- `["runtime_projection_upper_bound_insufficient", "symbolic_thread_binding_semantics_mismatch"]`: `1`

top violation strings:

- `vectorize term 1: runtime-projected selector upper bound ≤ max_vector_bytes`: `38`
- `shared_memory: runtime-projected shared bytes upper bound ≤ limit`: `32`
- `threads per block under blockIdx.x@s8.i0 (vthread * threadIdx.x) ≤ 1024`: `30`
- `vthread extent T_relu:ax0.1@ax1.1@ax2.1@ax3.1@ ≤ 8`: `11`
- `threads per block under blockIdx.x@s6.i0 (vthread * threadIdx.x) ≤ 1024`: `9`

### Generation validation

artifact:

- `/tmp/projected_gpu_full_validation/final_validation_20260311_generation/generation_merged_summary.json`
- `/tmp/projected_gpu_full_validation/final_validation_20260311_generation/generation_merged_invalid.jsonl`

요약:

- selected_unique_sketches: `912`
- attempts: `2736`
- randomize_success: `2733`
- randomize_fail: `3`
- concrete_invalid: `0`
- status_counts:
  - `ok`: `911`
  - `no_randomize_success`: `1`

즉:

- concrete-invalid schedule은 하나도 나오지 않았다.
- 하지만 한 sketch는 generation acceptance가 pruning-only라서 끝까지 성공하지 못했다.

### generation 실패 sketch

유일한 실패는 아래다.

- `sketch_index=117`
- task: `vm_mod_fused_nn_dense_nn_relu`
- status: `no_randomize_success`

관련 artifact:

- `/tmp/projected_gpu_full_validation/final_validation_20260311_generation/generation_shard_1_summary.json`
- `/tmp/projected_gpu_full_validation/final_validation_20260311_generation/generation_shard_1.log`

실패 이유:

- 3번 모두 `shared_memory: runtime-projected shared bytes upper bound ≤ limit: actual=65540`
- final/concrete 기준으로는 reject reason이 아니지만
- sampler가 마지막 accept에서 아직 pruning checker를 사용하기 때문에 생성 실패로 남는다.


## false reject 원인 분석

### 1. `runtime_projection_upper_bound_insufficient`

대표 증상:

- `shared_memory` projected upper bound가 실제보다 크게 나옴
- 예: projected는 reject인데 raw exact / concrete verifier는 pass

핵심 원인:

- runtime 축 제거 시 independent upper bound를 합치면서 과대추정
- 특히 `shared_memory`에서 tail / quotient-remainder correlation을 놓침

해석:

- symbolic state 자체의 오류라기보다 projection 제약식이 너무 보수적이다.

### 2. `symbolic_thread_binding_semantics_mismatch`

대표 증상:

- symbolic state에서는 `vthread * threadIdx.x`를 per-block threads로 계산
- final lowered TIR 기준 verifier에서는 그렇게 남아 있지 않음

핵심 원인:

- `max_threads` constraint source가 final lowered kernel semantics와 다름

해석:

- symbolic thread binding은 ordering/hint로는 유용하지만
- final accept/reject semantics로는 과대추정이 생길 수 있다.

### 3. `custom_exact_lowering_mismatch`

대표 증상:

- raw exact path의 `vectorize` 또는 `max_vthread`가 concrete lower path와 다름

핵심 원인:

- `src/auto_scheduler/exact_gpu_constraints.cc`의 custom symbolic lowering path와
- `gallery/constrained_gen/modules/tvm_verify.py`의 concrete lowering path가 완전히 같지 않음

해석:

- 이 경우는 projected upper bound만의 문제가 아니라 raw exact path 자체가 concrete verifier semantics와 어긋난다.


## 현재 남아 있는 가장 중요한 known issue

### `ParamSampler` 최종 accept가 아직 pruning-only다

현재 코드:

- `gallery/constrained_gen/modules/param_sampler.py`
- `_randomize_params_with_order()` 안에서 full assignment 후
  - `violations = g.check_all()`
  - 를 사용한다.

현재 `g.check_all()`은 pruning alias다.

결과:

- final checker와 concrete verifier는 pass여도
- generator는 projected/pruning false reject 때문에 후보를 버릴 수 있다.

실제 드러난 예:

- `sketch_index=117`
- generation validation에서 유일한 `no_randomize_success`

정리:

- 이번 리팩터링으로 `check_all_final()` 자체는 맞췄다.
- 하지만 sampler의 accept semantics는 아직 바꾸지 않았다.


## 코드 상태 요약

### 현재 신뢰 가능한 것

- facade split 자체는 동작한다.
- `py_compile` 통과
- 스모크 테스트:
  - `index=2`
  - `index=117`
  - 둘 다 `check_all_pruning()`은 violation이 남고
  - `check_all_final()`은 `0` violation
  - prefix snapshot도 정상 동작

### 스모크 테스트에서 확인한 값

- `index=2`
  - task: `vm_mod_fused_nn_batch_matmul_3`
  - phase count: `7`
  - prefix length through `non_product_gate_vars`: `12`
  - pruning violations: `1`
  - final violations: `0`

- `index=117`
  - task: `vm_mod_fused_nn_dense_nn_relu`
  - phase count: `7`
  - prefix length through `non_product_gate_vars`: `8`
  - pruning violations: `1`
  - final violations: `0`
  - prefix snapshot fixed values: `10`
  - leftover constraints: `0`


## 남겨 둔 설계 결정

### 아직 facade 안에 둔 것

`ScheduleGenerator`를 더 줄일 여지는 있지만, 이번 단계에서는 아래를 facade에 남겼다.

- formatting / pretty-print
- raw exact constraint string rendering
- concrete final checker

즉 “constraint build / planning / propagation / sampling”만 먼저 분리했다.

### vectorize는 early phase에서 제외

의도적으로 유지한 결정이다.

- early variable order는 thread/vthread/split 중심
- vectorize는 legacy tail / 별도 constraint로 남겨둠


## 다음 세션에서 바로 보면 좋은 파일

- 구조
  - `gallery/constrained_gen/modules/schedule_generator.py`
  - `gallery/constrained_gen/modules/constraint_set.py`
  - `gallery/constrained_gen/modules/var_order_planner.py`
  - `gallery/constrained_gen/modules/domain_propagator.py`
  - `gallery/constrained_gen/modules/param_sampler.py`

- validation logic
  - `gallery/constrained_gen/validate_exact_gpu_constraints.py`
  - `gallery/constrained_gen/validate_projected_gpu_generation.py`
  - `gallery/constrained_gen/modules/projected_gpu_validation.py`

- concrete verifier path
  - `gallery/constrained_gen/modules/tvm_verify.py`

- raw exact path
  - `gallery/constrained_gen/modules/exact_gpu_constraints.py`
  - `src/auto_scheduler/exact_gpu_constraints.cc`


## 추천 다음 작업 순서

1. `ParamSampler`의 full-assignment accept를 `check_all_final()` 또는 hybrid path로 바꿀지 결정
2. `shared_memory` false reject를 줄일 전략 결정
   - late exact check
   - tighter runtime projection
   - 3-state pruning 강화
3. `symbolic_thread_binding_semantics_mismatch` 제거
   - `max_threads` final semantics를 final lowered TIR 쪽에 맞추기
4. `custom_exact_lowering_mismatch` 조사
   - custom exact lowering vs concrete lowering differential 비교


## 한 줄 상태 요약

생성기 구조 분리는 완료됐고 final checker는 concrete verifier와 맞는다. 남은 핵심 문제는 `false reject`이며, 현재 가장 직접적인 follow-up은 `ParamSampler`가 아직 pruning-only accept를 쓰는 점을 고치는 것이다.
