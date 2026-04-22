# vthread diag: hw_param=8 → 15 탐색 성능 저하 원인 분석

## 1. 문제

### 1.1 원래 상태

`ScheduleGenerator.DEFAULT_HW_PARAM['max_vthread_extent'] = 8` 이
Ansor/constrained_gen의 기본값. 이 값은 vthread bind 축의 extent 상한으로
들어가 constraint generator가 candidate 집합을 만들 때 쓰이고,
`_build_teacher_forcing_candidate_masks` → 학습 loss의 `masked_cross_entropy`
candidate mask로 흘러들어감. 동시에 탐색(latent walk / sampling) 시
generator가 emit 가능한 스케줄 공간도 같은 값으로 제한됨.

v1.5_ori baseline은 `cfg.generator.hw_param = {}` (즉 기본 8)로 학습+탐색을
모두 수행함. task_index=1490 (conv2d winograd stage)에서 baseline의 최종
탐색 성능은 `walk/best_measured_mean_cost = 10.664` (higher=better; 값이
클수록 측정된 실행 비용이 낮음).

### 1.2 변경과 기대

`cfg.generator.hw_param = {"max_vthread_extent": 15}` 로 올림. 동기는
`vthread_extent_blindspot` 메모에 기록된 "8~15 구간이 ansor의 과보수적
cap에 가려져 탐색되지 않는데, unrolled 상황에서는 extent>8이 유리할 수
있다"는 관찰. hw=15로 올리면 학습·탐색 모두 더 넓은 vthread 공간을 보게
되므로 최소한 성능이 유지되거나 개선될 것으로 기대.

### 1.3 관측된 문제

task 1490에서 hw=15로 학습·탐색 모두 돌렸을 때 성능이 **오히려 하락**:

| 설정 | 학습 generator | 탐색 generator | best_measured_mean_cost |
|---|---|---|---|
| baseline | hw=8 | hw=8 | **10.664** |
| 원 실험 | hw=15 | hw=15 | **10.460** |

저하 폭 = **0.204**. 더 넓은 후보 공간을 보여줬는데 왜 성능이 떨어졌는지를
규명하는 것이 이 진단의 목적.

## 2. 후보 원인

hw_param이 바뀔 때 학습/탐색에서 달라지는 경로를 추적:

- **(A) 학습-time candidate mask widening** — 비-fallback 위치에서 candidate
  mask가 넓어져 `masked_cross_entropy`의 softmax 분모가 달라짐.
- **(B) recon 위치 부활** — baseline에서 `_build_teacher_forcing_candidate_masks`가
  singleton fallback을 걸었던 위치들이 hw=15에서는 정상 후보 분포가 됨.
  그 위치에서 새로 recon gradient가 흐름.

Mask를 소비하는 live objective는 recon_loss가 유일하므로 (A)+(B)가 학습
gradient 차이의 전부.

## 3. 실험

### 3.1 diag_vthread_violation.py (B 제거 테스트)

`VIOLATION_LOSS_WEIGHT=0.0`으로 "baseline에서 singleton이었으나 hw=15에서는
정상인" 위치의 recon gradient를 0으로 눌러 (B)를 제거. 학습은 여전히 hw=15.

- 구현: `_load_baseline_workload_cache`로 `..._v4_no_budget.pt` 로드,
  `singleton_base & ~singleton_cur`를 `diag_scale=0` 적용.
- 결과: 탐색 성능이 baseline으로 회복되지 않음 (사용자 보고 "많이 나빴음").
- 결론: (B)만으로는 저하의 주 원인이 아님. (A)가 여전히 살아있음을 의미.

### 3.2 Step 1 — 학습 hw=15, 탐색 hw=8

hw=15로 학습된 checkpoint를 받아 `diag_step1_explore_hw8.py`로 **탐색
generator만 hw=8로 교체**. 모델 가중치는 그대로.

- 구현: `load_bundle(checkpoint)` 후 `_override_registry_hw_param(bundle, {})`
  로 `GeneratorRegistry`만 재구성 → `run_latent_walk(bundle=bundle, ...)`.
- 결과: **10.570**. baseline(10.664)까지는 안 오지만 원 실험(10.460)에서
  0.110 회복(전체 저하의 54%).
- 해석 (초기): 학습-time과 탐색-time이 대략 반반 기여하는 것처럼 보임.

### 3.3 Step 2 — 학습 hw=8, 탐색 hw=15

baseline checkpoint(hw=8로 학습)를 `diag_step2_explore_hw15.py`로 실행해
**탐색만 hw=15**로 돌림. 재학습 불필요 (학습 경로에서 mask를 hw=8로
override하는 것과 등가임을 확인 — 학습 gradient가 mask에 의존하는 경로는
recon_loss 하나인데, baseline 학습은 이미 hw=8 mask로 수렴한 상태).

- 결과: **10.696**. baseline(10.664)과의 차이 +0.032는 측정 노이즈 범위.

## 4. 측정 노이즈 기준선

metric은 `-ln(mean_cost_seconds)`, `higher=better`. 같은 스케줄이 여러 번
측정된 이력(`checkpoints_all/1490/measure_records/*.json`에서 ≥5회 반복된
324개 스케줄)을 이용해 단일-스케줄 재측정 노이즈를 추정:

| 통계 | std dev | range (max−min) |
|---|---|---|
| p50 | 0.008 | 0.022 |
| p90 | 0.017 | 0.057 |
| max | 0.026 | 0.083 |

`best_measured_mean_cost`는 walk가 찾은 서로 다른 스케줄 중 최대값이라
엄밀한 노이즈 모델은 다르지만, **delta가 0.03~0.04 이하이면 유의성이
없다고 봐야 함** (대략 ≤2σ). 0.1 이상은 명확히 유의.

## 5. 4-코너 매트릭스

| 학습 \ 탐색 | hw=8 | hw=15 |
|---|---|---|
| **hw=8** | 10.664 | 10.696 |
| **hw=15** | 10.570 | 10.460 |

각 방향의 delta와 유의성:

| 비교 | delta | σ 추정 | 해석 |
|---|---|---|---|
| baseline vs Step 2 | **+0.032** | ~2σ | **노이즈 내 동일** |
| baseline vs Step 1 | **−0.094** | ~5σ | 유의 |
| baseline vs hw=15 | **−0.204** | ~12σ | 매우 유의 |
| Step 1 vs hw=15 | **−0.110** | ~6σ | 유의 (상호작용) |

읽는 법:
- 학습 hw=8 고정: 탐색 8→15가 노이즈 내 동일 → **탐색 widening 단독으로는
  무해**
- 학습 hw=15 고정: 탐색 8→15가 −0.110로 추가 저하 → **hw=15로 망가진
  모델이 hw=15 탐색에서 특히 더 혼란** (상호작용)
- 탐색 고정 여부와 무관하게 학습 8→15는 항상 큰 저하 → **학습이 주원인**

## 6. 결론

**저하의 원인은 학습-time (A) 하나**. 구체적으로 `masked_cross_entropy`에서
hw=15 candidate mask를 softmax 분모로 쓰는 것이 모델 품질을 떨어뜨림. 다만
**(A)가 어떤 메커니즘으로 모델을 나쁘게 만드는지는 아직 불확정**.

근거 (확실히 말할 수 있는 범위):
- hw=8로 학습된 모델은 탐색 domain이 hw=15로 넓어져도 노이즈 내 동일 성능
  (Step 2 ≈ baseline). → 탐색-time domain 확장은 본질적 문제 아님.
- hw=15로 학습된 모델은 어떤 탐색 domain에서도 baseline보다 유의하게 못함
  (Step 1, hw=15 모두). → 학습 자체가 모델을 망침.
- (B)만 제거한 diag(§3.1)도 baseline까지 회복 안 됨. → "hw=8에서 singleton
  이던 위치가 hw=15에서 정상 후보로 부활" 경로는 주범 아님. **(A)**, 즉
  기존 비-fallback 위치의 mask 분모 확장이 남은 범인.

초기에 Step 1의 "54% 회복"을 "탐색 domain 확장이 절반 기여"로 읽었으나
Step 2가 baseline 수준임이 드러나면서 이 해석은 기각. 실제 구조는:

- 학습 widening이 저하의 전부 (−0.094, 탐색 hw=8 기준)
- 학습 hw=15 + 탐색 hw=15 조합에서 추가 −0.110 상호작용 발생 (학습-망가진
  모델이 hw=15 후보를 특별히 잘못 다루는 효과)

### 메커니즘 미확정 — 기각된 가설

초기 가설 "hw=15-only 후보는 gold로 등장하지 않아 positive supervision이
0이고 모델이 이들을 능동적으로 억제하도록 학습된다"는 **training record에
vthread extent > 8 gold가 실제로 존재함이 확인되어 기각**. 존재하는 이상
그 샘플들의 해당 position은 정상 positive gradient를 받음 (`_build_teacher_forcing_candidate_masks`
의 fallback 경로에 빠지지 않음).

### 남은 가설 (검증 필요)

- **데이터 불균형**: extent > 8 gold 샘플 비율이 낮아 hw=15에서만 살아나는
  희귀 gold에 대해 high-variance gradient가 흐름. hw=8 학습에서는 같은 샘플
  들의 해당 position이 오라클 외 gold → singleton fallback → position
  weight 0으로 제거됨(train_epoch.py:107). hw=15에서 갑자기 살아나면서
  학습 안정성에 악영향 가능.
- **Softmax 분모 희석**: gold ≤ 8이 다수인 common position에서도 hw=15는
  extent > 8 distractor logit을 추가로 낮추도록 push함. 모델 용량이
  distractor 억제에 소비되면서 common domain 학습이 저하될 가능성.
- **용량 경쟁**: 같은 파라미터로 더 넓은 출력 domain을 커버해야 함 → 흔한
  domain의 정확도 희생.

이 셋을 판별하려면 다음이 필요:
- record 내 vthread extent gold 분포 (>8 비율, var별/task별 분포)
- hw=8 학습 중 fallback에 빠진 position 개수 (per-epoch)
- hw=15 학습 시 extent > 8 gold position의 loss 궤적 vs extent ≤ 8 gold
  position의 loss 궤적

## 7. 해결 방향

### 7.1 즉시 실용해

**학습은 hw=8 mask, 탐색만 hw=15**. Step 2 결과가 baseline과 노이즈 내
동일. `cfg.generator.hw_param`을 학습 시 `{}`로 두고 checkpoint 로드 후
탐색 전에 `{"max_vthread_extent": 15}`로 override. 코드 변경 최소.

다만 이 경로는 hw=8 학습 시 extent > 8 gold를 가진 record의 해당 position이
singleton fallback으로 zero-weight 되는 것을 감수함. 즉 "그 데이터를 못
배우는 대신 모델 안정성을 얻는" 트레이드오프.

### 7.2 중·장기 근본 해결 (메커니즘 확인 후 선택)

메커니즘이 확정되지 않았으므로 아래는 가설별 후보.

- **데이터 불균형 가설이 맞다면**: extent > 8 gold 샘플 부스팅 (sample
  weighting, 또는 해당 position에 position_weights 가중). hw=15 학습에서
  희귀 gold가 안정적인 gradient를 받도록.
- **분모 희석 / 용량 경쟁 가설이 맞다면**: hw=15-only 후보에 대해 gradient
  완화 (label_smoothing 비대칭 재배분, margin loss 변형, 또는 temperature
  scaling). common domain 학습 왜곡 축소.
- **공통 안전책 — self-training 부트스트랩**: 현재 모델로 hw=15 탐색 → 측정
  → 좋은 스케줄만 training set에 추가 → hw=15 mask로 재학습. 반복. extent
  > 8 영역의 gold 밀도를 높이는 효과. `walk_sample_buffer` 인프라 재활용
  가능하지만 현재는 학습 피드백 경로가 없어 별도 연결 필요.
