# TVM AutoScheduler vthread 제약 검증 맹점(Blind Spot) 분석

## 📌 문제 현상 요약
TVM의 AutoScheduler(Ansor)를 통해 생성된 측정 기록(Measurement Records, `.json` 파일)에서 **`vthread`(가상 스레드)의 크기가 하드웨어 제약 조건인 8(`max_vthread_extent`)을 초과하여 9~15 사이의 값을 가지는 개체들이 버젓이 등장**하는 현상이 관찰되었습니다.

이 문서는 왜 이 제약 조건이 Evolutionary Search 도중에 걸러지지 않았는지 코드 레벨에서 원인을 추적하고 분석한 결과입니다.

---

## 🔍 원인 분석: 두 패스 간의 통합 버그 (Integration Bug)

이 문제는 서로 만든 시기와 목적이 다른 **두 가지 임계값(8 vs 16)** 이 충돌하면서 발생한 전형적인 구조적 허점입니다.

### 1. `max_vthread_extent = 8` (AutoScheduler의 의도)
* **위치**: `src/auto_scheduler/search_task.cc` 및 `VerifyGPUCode`
* **유래**: Ansor 연구진이 탐색 공간을 설계하면서 하드웨어 벤치마크를 수행한 결과, `vthread` 크기가 GPU의 Warp Size(32)의 1/4인 **8을 넘어가는 순간 Register Spilling 현상이 발생하여 성능이 급락**한다는 것을 확인했습니다.
* **동작 체계**: 이를 방지하기 위해 생성된 코드가 `max_vthread` 제약을 넘으면 `VerifyGPUCode` 검사 단계에서 에러를 던지고, Cost Model이 이에 `-inf` 점수를 부여하여 Evolutionary Search 과정에서 자연스럽게 버려지도록(Drop) 설계했습니다.

### 2. `16 미만 Unroll` (기존 컴파일러 패스의 함정)
* **위치**: `src/tir/transforms/inject_virtual_thread.cc`
* **유래**: Ansor가 탑재되기 훨씬 전인 TVM 초창기에 만들어진 AST(추상 구문 트리) 기반 컴파일러 최적화 패스입니다.
* **목적**: "작은 크기의 `vthread` 루프를 가상 스레드 형태로 다 남겨두면 코드가 복잡해지니, **16 미만인 경우에는 아예 `For` 루프를 제거(Unroll)해버리자**"며 하드코딩해 둔 컴파일 시간 최적화를 위한 매직 넘버였습니다.

### 3. 검증 로봇의 눈을 피한 '밀수(Smuggling)' 현상
Ansor가 마련한 검문소(`VerifyGPUCode`)는 **이름이 `vthread.s`인 `For` 루프 구문을 찾아내어** 크기를 검사(`VisitStmt_(const ForNode*)`)하도록 맹목적으로 구현되어 있었습니다.

1. **상태 생성**: Evolutionary Search 도중 `MutateTileSize` 등으로 크기가 9~15인 `vthread`가 만들어집니다.
2. **트리 변이**: `InjectVirtualThread` 패스가 실행되면서, 이 값들은 16 미만이므로 루프(`vthread.s`) 자체가 흔적도 없이 완전히 Unroll 처리되어 버립니다.
3. **검증 우회**: 그 직후 실행된 `VerifyGPUCode`는 코드를 아무리 뒤져도 `vthread.s`라는 이름의 `For` 루프를 찾을 수 없으므로, **"위반 사항 없음(정상)"** 으로 착각하여 코드를 통과시킵니다.
4. **결과**: Cost Model에 의해 정상적인 점수가 매겨지고, 측정(Measurement) 단계로 넘어간 해당 코드는 NVCC 단계에서 문제없이 C++ 코드로 컴파일 및 실행되므로 최종적으로 JSON 파일에 성공 기록으로 등재됩니다.

---

## 💡 결론 및 시사점

**"어차피 허용해 줄 거면 15로 하지 왜 8로 했냐?"** 라는 의문은 바로 여기서 해답을 얻게 됩니다.

Ansor 연구진은 하드웨어 성능을 위해 **철저히 8까지만 허용하고 9부터는 모두 폐기**하려는 확고한 의도를 가졌습니다. (실제로 `vthread >= 16`인 경우는 Unroll 되지 않고 껍데기가 남아 정상적으로 `VerifyGPUCode`에 걸려 폐기됩니다.)

하지만 TVM 본체의 레거시 코드(`InjectVirtualThread`)가 `vthread`라는 꼬리표를 가진 15까지의 루프를 임의로 벗겨(Unroll)버리는 바람에, 정작 Ansor의 하드웨어 제약 검증 로봇이 **9~15 구간에 대한 감독 능력을 상실하게 된 아키텍처 상의 맹점(Blind Spot)** 이 근본 원인이었습니다. 

이를 방지하려면 AST 상의 패스 실행 순서를 조작하거나, 애초에 탐색 공간의 `MutateTileSize` 단계에서 `vthread`가 초과할 수 없도록 사전에 차단하는 코드를 추가하는 등 아키텍처의 연계를 보강해야 합니다.
