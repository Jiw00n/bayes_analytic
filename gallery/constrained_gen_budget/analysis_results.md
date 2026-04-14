# TVM Ansor Constrained Gen Budget 코드 구조 분석

`/root/work/tvm-ansor/gallery/constrained_gen_budget` 디렉토리 하위의 핵심 모듈인 `modules`와 `latent_model_budget`는 **규칙 기반(Rule-based) GPU 스케줄 제약 시스템**과 **Latent Variable Model(VAE) 기반의 딥러닝 분포 학습 모듈**로 나누어집니다. 이 두 시스템은 함께 결합되어 "GPU 하드웨어 제약(Shared Memory, Threads 등)을 100% 만족하면서도, 좋은 성능을 낼 가능성이 높은 스케줄 파라미터를 연속적인 잠재 공간(Latent Space)에서 탐색/생성"하는 역할을 수행합니다.

---

## 1. `modules/` (심볼릭 제약 추론 및 파라미터 샘플링 엔진)

TVM의 텐서 연산 스케줄링(Split, Unroll, Bind 등)에서 발생하는 여러 인자들을 추상적인 수식(AST)으로 모델링하고, 이것이 실제 GPU 하드웨어 한계를 넘지 않도록 제한(Constraint)을 적용하는 **규칙 엔진(Rule Engine)** 부분입니다.

### 🎯 핵심 특징 및 구성 요소
*   **Symbolic Expression AST (`sym_types.py`, `expr_nodes.py`)**  
    스케줄 단계에서 아직 결정되지 않은 파라미터(예: `sp_0_1`, `ur_2`)를 기호(Symbol)로 간주하고 덧셈, 곱셈, 나눗셈, Min, Max 등의 AST를 구축합니다.
*   **Constraint Set (`constraint_set.py`, `gpu_projection_constraints.py`)**  
    물리적 한계(예: `max_vectorize_bytes`, `max_shared_memory`, `max_threads_per_block`, `max_vthread`)들을 추출하여 상한/하한 방정식 제약으로 변환합니다. 상수 뿐만이 아니라 동적인 수식 간의 관계로 정의됩니다.
*   **Domain Propagator (`domain_propagator.py`)**  
    아직 값이 결정되지 않은 모든 변수들에 대해 범위(Domain, min~max)를 전파(Propagate)합니다. 하나의 변수 값이 확정되면 연결된 수식을 통해 다른 변수의 가능한 선택지가 동적으로 축소됩니다.
*   **Var Order Planner (`var_order_planner.py`)**  
    조건 제약식을 어기지 않고 값을 할당하려면 어떤 변수부터 샘플링해야 하는지 순서(Phase)를 계획합니다. 상호 의존성이 높은 변수 그룹(예: Threads-per-block)이나 메모리 관련 변수들의 순서를 최적화합니다.
*   **Schedule Generator & Parameter Sampler (`schedule_generator.py`, `param_sampler.py`)**  
    위의 시스템들을 통괄하여 제약을 완벽히 만족하는 파라미터를 무작위로 생성하거나(ParamSampler), 외부 시스템에 현재 변수가 할당 가능한 값들의 후보(Candidates)를 조회할 수 있는 인터페이스(ScheduleGenerator)를 제공합니다.

> [!NOTE]
> `modules` 디렉토리는 TVM의 컴파일 검증 실패(Lowering 에러)를 겪기 전에, 파이썬 레벨에서 사전 수식 검증을 통해 100% 실행 가능한 상태만을 걸러내는 "안전망(Oracle)" 역할을 수행합니다.

---

## 2. `latent_model_budget/` (VAE 기반 스케줄 잠재 공간 생성 모델)

`modules` 가 구조적으로 "가능한 파라미터의 범위"를 정해준다면, `latent_model_budget`은 "어떤 파라미터 조합이 더 빠른 실행 시간을 나타낼 것인가"를 VAE(Variational Autoencoder)를 활용해 학습하고 생성합니다.

### 🎯 핵심 특징 및 구성 요소
*   **LatentParamVAE (`model.py`)**  
    주어진 스케줄 파라미터를 고정 벡터 크기의 잠재 공간(Latent Space) $z$로 압축(Encode)하고, 다시 파라미터 시퀀스로 복원(Decode)하는 핵심 딥러닝 모델입니다. 트랜스포머 파생 아키텍처(Cross Attention, AdaLN)를 사용하며 스케줄 성능(Cost)을 예측하는 `cost_head`를 지니고 있습니다.
*   **Prefix Legality Adapter (`adapter.py`)**  
    생성 언어 모델과 `modules/`의 제약 엔진을 이어줍니다. VAE가 Decoder를 통해 파라미터를 한 스텝씩 자동회귀(Autoregressive)로 생성해 낼 때, `LegalPrefixOracle`이 제약 엔진을 쿼리하여 **불가능한(Illegal) 토큰은 마스킹(Masking)**시킵니다. 이로 인해 모델은 언제나 GPU에서 컴파일 가능한 합법적(Legal) 스케줄만 디코딩하게 됩니다.
*   **Tokenizer & Dataset (`tokenizer.py`, `dataset.py`)**  
    계층적인 스케줄 변수 이름(`sp_0_0`, `ur_1` 등)과 해당 정수 값들을 식별자(ID) 토큰화하며, 예산(Budget) 한계 데이터도 포함하여 PyTorch 데이터셋으로 만듭니다. 
*   **Side Features (`..._side_features.py`)**  
    동적인 상태 자질(Dynamic Features)과 수치형 제약 피처들을 추출해 트랜스포머의 입력으로 덧붙여 성능 예측력을 높입니다.
*   **Training & Evaluation Loop (`train.py`, `train_losses.py`, `train_eval.py`)**  
    기존에 TVM 튜너(Ansor)를 돌려 수집한 (스케줄, 성능/시간) 쌍의 기록(.json)을 학습 오답 노트처럼 사용해 모델을 학습시키는 파이프라인 스크립트입니다.

> [!IMPORTANT]
> `latent_model_budget` 내장 VAE는 일반적인 언어 모델과는 달리, 역추적-마스킹 어댑터(`adapter.py`)와 긴밀하게 결합되어 있어, 딥러닝이 단순 확률적 실수(예: Block Size > 1024 생성)를 저지르지 않도록 하드웨어 규칙을 강제받고 있습니다.

## 💡 종합 분석
현재 이 디렉토리는 기존 TVM AutoScheduler(Ansor)의 진화형 연구로 보입니다. 과거 Ansor가 순수 무작위 진화 알고리즘 기반이었다면, 이 코드는 **1단계: `modules/` 로 절대로 실패하지 않는 검색 공간 가지치기(Pruning)** 와 **2단계: `latent_model_budget/` 로 잠재 공간 단위(Latent Space Walking/Prediction) 방향 탐색**을 혼합하여 탐색 예산(Budget) 내에서 최고의 커널을 안정적으로 빠르게 찾아내기 위한 정교한 인프라로 설계되어 있습니다.
