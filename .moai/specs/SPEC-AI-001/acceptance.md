# SPEC-AI-001: 인수 기준

## TAG BLOCK

```yaml
spec_id: SPEC-AI-001
document_type: acceptance
status: Planned
created: 2025-01-18
updated: 2025-01-18
```

---

## 1. 기능 인수 기준

### AC-001: DINOv2 Feature Extractor

**시나리오 1: 단일 이미지 피처 추출**

```gherkin
Given DINOv2 ViT-L 모델이 로드되어 있고
  And 768x768 크기의 RGB 이미지가 준비되어 있을 때
When Feature Extractor에 이미지를 입력하면
Then 1024차원의 피처 벡터가 반환되어야 한다
  And 피처 벡터의 dtype은 float32여야 한다
  And 추론 시간은 100ms 이내여야 한다
```

**시나리오 2: 배치 이미지 피처 추출**

```gherkin
Given DINOv2 ViT-L 모델이 로드되어 있고
  And 32개의 768x768 RGB 이미지 배치가 준비되어 있을 때
When Feature Extractor에 배치를 입력하면
Then (32, 1024) 크기의 피처 텐서가 반환되어야 한다
  And GPU 메모리 사용량이 12GB 이내여야 한다
```

**시나리오 3: 모델 freeze 검증**

```gherkin
Given DINOv2 모델이 로드되어 있을 때
When 모델의 파라미터를 확인하면
Then 모든 파라미터의 requires_grad가 False여야 한다
  And 학습 시에도 backbone 가중치가 변경되지 않아야 한다
```

---

### AC-002: Projector Layer

**시나리오 1: 차원 변환**

```gherkin
Given Projector 레이어가 초기화되어 있고
  And 1024차원의 피처 벡터가 입력될 때
When Projector를 통해 변환하면
Then 256차원의 출력 벡터가 반환되어야 한다
  And 출력 벡터가 정규화되어 있어야 한다 (LayerNorm 적용)
```

**시나리오 2: 그래디언트 흐름**

```gherkin
Given Projector 레이어가 학습 모드이고
  And 입력 피처와 타겟이 준비되어 있을 때
When 역전파를 수행하면
Then Projector의 모든 가중치에 그래디언트가 계산되어야 한다
  And 그래디언트 값이 NaN이나 Inf가 아니어야 한다
```

**시나리오 3: Dropout 동작**

```gherkin
Given Projector 레이어가 있을 때
When 동일한 입력으로 학습 모드에서 여러 번 forward하면
Then 출력값이 매번 다르게 나와야 한다 (Dropout 작동)

Given Projector 레이어가 있을 때
When 동일한 입력으로 평가 모드에서 여러 번 forward하면
Then 출력값이 매번 동일해야 한다 (Dropout 비활성화)
```

---

### AC-003: Pairwise Ranking Model

**시나리오 1: 페어 스코어 계산**

```gherkin
Given PairwiseRankingModel이 로드되어 있고
  And 두 개의 768x768 RGB 이미지가 준비되어 있을 때
When 두 이미지를 모델에 입력하면
Then 각 이미지에 대한 스칼라 스코어가 반환되어야 한다
  And 스코어는 비교 가능한 값이어야 한다 (더 높은 스코어 = 더 좋은 작품)
```

**시나리오 2: 순위 예측**

```gherkin
Given PairwiseRankingModel이 학습되어 있고
  And S티어 이미지와 C티어 이미지가 준비되어 있을 때
When 두 이미지의 순위를 예측하면
Then S티어 이미지의 스코어가 C티어 이미지보다 높아야 한다
  And 예측 결과가 일관되어야 한다 (동일 입력에 동일 출력)
```

**시나리오 3: 배치 처리**

```gherkin
Given PairwiseRankingModel이 로드되어 있고
  And 32개의 이미지 쌍이 준비되어 있을 때
When 배치로 스코어를 계산하면
Then 32개의 스코어 쌍이 반환되어야 한다
  And 처리 시간이 단일 처리 * 32보다 빨라야 한다 (배치 효율성)
```

---

### AC-004: 데이터셋 분할

**시나리오 1: 80/10/10 분할**

```gherkin
Given 2,000개의 이미지 메타데이터가 있을 때
When 데이터를 80/10/10 비율로 분할하면
Then Train 셋에 약 1,600개가 포함되어야 한다
  And Validation 셋에 약 200개가 포함되어야 한다
  And Test 셋에 약 200개가 포함되어야 한다
```

**시나리오 2: Stratified 분할**

```gherkin
Given 티어별(S/A/B/C) 분포가 있는 데이터셋이 있을 때
When Stratified 분할을 수행하면
Then 각 분할의 티어 비율이 원본과 유사해야 한다
  And 티어 비율 차이가 5% 이내여야 한다
```

**시나리오 3: 테스트 셋 격리**

```gherkin
Given 데이터가 분할되어 있을 때
When 학습이 진행될 때
Then 테스트 셋의 이미지가 학습에 사용되지 않아야 한다
  And 테스트 셋의 이미지가 검증에 사용되지 않아야 한다
```

---

### AC-005: Pairwise 데이터 생성

**시나리오 1: 유효한 페어 생성**

```gherkin
Given 티어 라벨이 있는 이미지 메타데이터가 있을 때
When Pairwise 데이터를 생성하면
Then 다른 티어의 이미지 쌍만 생성되어야 한다
  And 동일 티어의 이미지 쌍은 제외되어야 한다
```

**시나리오 2: 라벨 정확성**

```gherkin
Given S티어 이미지 A와 C티어 이미지 B가 있을 때
When (A, B) 페어를 생성하면
Then 라벨이 1이어야 한다 (A > B)

Given C티어 이미지 A와 S티어 이미지 B가 있을 때
When (A, B) 페어를 생성하면
Then 라벨이 -1이어야 한다 (A < B)
```

**시나리오 3: 페어 수량**

```gherkin
Given N개의 이미지가 있고 균등한 티어 분포일 때
When 모든 가능한 페어를 생성하면
Then 페어 수가 N*(N-1)/2 이하여야 한다 (동일 티어 제외)
  And 최소 학습에 충분한 페어 수가 생성되어야 한다
```

---

### AC-006: 학습 프로세스

**시나리오 1: Loss 감소**

```gherkin
Given 학습 데이터가 준비되어 있고
  And 모델이 초기화되어 있을 때
When 10 epoch 학습을 진행하면
Then Train loss가 초기 대비 감소해야 한다
  And Validation loss가 초기 대비 감소해야 한다
```

**시나리오 2: Early Stopping**

```gherkin
Given Early stopping patience가 10으로 설정되어 있을 때
When Validation loss가 10 epoch 동안 개선되지 않으면
Then 학습이 자동으로 종료되어야 한다
  And 최적의 체크포인트가 저장되어 있어야 한다
```

**시나리오 3: 체크포인트 저장**

```gherkin
Given 학습이 진행 중일 때
When Validation accuracy가 최고값을 갱신하면
Then 모델 체크포인트가 저장되어야 한다
  And 체크포인트에 모델 가중치가 포함되어야 한다
  And 체크포인트에 optimizer 상태가 포함되어야 한다
  And 체크포인트에 epoch 정보가 포함되어야 한다
```

**시나리오 4: 학습 재개**

```gherkin
Given 저장된 체크포인트가 있을 때
When 체크포인트에서 학습을 재개하면
Then 이전 epoch부터 학습이 계속되어야 한다
  And 모델 가중치가 복원되어야 한다
  And Optimizer 상태가 복원되어야 한다
```

---

### AC-007: Pairwise Accuracy 평가

**시나리오 1: 정확도 계산**

```gherkin
Given 학습된 모델이 있고
  And 테스트 페어 데이터가 준비되어 있을 때
When Pairwise accuracy를 계산하면
Then 정확도가 0과 1 사이의 값이어야 한다
  And 정확도 = (올바른 예측 수) / (전체 페어 수)여야 한다
```

**시나리오 2: 목표 정확도 달성**

```gherkin
Given 학습이 완료된 모델이 있을 때
When 테스트 셋에서 평가를 수행하면
Then Pairwise accuracy가 60% 이상이어야 한다
```

**시나리오 3: 과적합 검증**

```gherkin
Given 학습이 완료된 모델이 있을 때
When Train, Validation, Test 정확도를 비교하면
Then Train과 Test 정확도 차이가 20% 이내여야 한다
  And Train 정확도가 80%를 크게 초과하지 않아야 한다
```

---

### AC-008: wandb 로깅

**시나리오 1: 메트릭 로깅**

```gherkin
Given wandb가 설정되어 있을 때
When 학습이 진행되면
Then 매 epoch마다 train/loss가 기록되어야 한다
  And 매 epoch마다 val/loss가 기록되어야 한다
  And 매 epoch마다 val/accuracy가 기록되어야 한다
```

**시나리오 2: 하이퍼파라미터 기록**

```gherkin
Given 학습이 시작될 때
When wandb run이 초기화되면
Then batch_size가 기록되어야 한다
  And learning_rate가 기록되어야 한다
  And 모든 config 값이 기록되어야 한다
```

**시나리오 3: 아티팩트 저장**

```gherkin
Given 학습이 완료되었을 때
When 최종 모델을 저장하면
Then wandb artifact로 모델 체크포인트가 업로드되어야 한다
```

---

## 2. 비기능 인수 기준

### AC-009: 성능 요구사항

**시나리오 1: 추론 속도**

```gherkin
Given 학습된 모델이 GPU에 로드되어 있을 때
When 단일 이미지 쌍의 스코어를 계산하면
Then 소요 시간이 100ms 이내여야 한다
```

**시나리오 2: 메모리 사용량**

```gherkin
Given 학습이 batch size 32로 진행될 때
When GPU 메모리 사용량을 측정하면
Then 12GB를 초과하지 않아야 한다
```

**시나리오 3: 학습 시간**

```gherkin
Given 전체 학습 데이터셋이 준비되어 있을 때
When 100 epoch 또는 early stopping까지 학습하면
Then 총 학습 시간이 6시간 이내여야 한다
```

---

### AC-010: 재현 가능성

**시나리오 1: 동일 결과 재현**

```gherkin
Given random seed가 42로 고정되어 있을 때
When 동일한 조건으로 학습을 두 번 실행하면
Then 두 실험의 최종 accuracy 차이가 1% 이내여야 한다
```

**시나리오 2: 결정론적 동작**

```gherkin
Given 동일한 입력 이미지가 주어질 때
When 평가 모드에서 여러 번 추론하면
Then 매번 동일한 스코어가 출력되어야 한다
```

---

### AC-011: 코드 품질

**시나리오 1: 린터 통과**

```gherkin
Given 모든 Python 파일이 작성되어 있을 때
When Ruff 린터를 실행하면
Then 경고나 오류가 없어야 한다
```

**시나리오 2: 테스트 커버리지**

```gherkin
Given 모든 테스트가 작성되어 있을 때
When pytest-cov를 실행하면
Then 코드 커버리지가 85% 이상이어야 한다
```

**시나리오 3: 타입 검사**

```gherkin
Given 모든 Python 파일에 타입 힌트가 있을 때
When mypy를 실행하면
Then 타입 오류가 없어야 한다
```

---

## 3. Definition of Done (완료 정의)

### 기능 완료 기준

- [ ] DINOv2 Feature Extractor가 구현되고 테스트를 통과함
- [ ] Projector Layer가 구현되고 테스트를 통과함
- [ ] PairwiseRankingModel이 구현되고 테스트를 통과함
- [ ] Pairwise Dataset이 구현되고 80/10/10 분할이 검증됨
- [ ] 학습 파이프라인이 구현되고 loss가 감소함
- [ ] wandb 로깅이 작동함
- [ ] 체크포인트 저장/로드가 작동함

### 성능 완료 기준

- [ ] Test Pairwise Accuracy >= 60%
- [ ] 추론 시간 < 100ms/pair
- [ ] 학습 시 GPU 메모리 < 12GB
- [ ] 학습 시간 < 6시간

### 품질 완료 기준

- [ ] Ruff 린터 경고 없음
- [ ] 테스트 커버리지 >= 85%
- [ ] 모든 EARS 요구사항에 대응하는 테스트 존재
- [ ] 코드 리뷰 완료

### 문서화 완료 기준

- [ ] README.md 업데이트
- [ ] 사용 가이드 작성
- [ ] API 문서 자동 생성 (docstring)
- [ ] wandb 실험 결과 기록

---

## 4. 검증 방법 요약

| 인수 기준 | 검증 방법 | 자동화 |
|-----------|-----------|--------|
| AC-001 ~ AC-003 | 단위 테스트 | pytest |
| AC-004 ~ AC-005 | 데이터 검증 스크립트 | pytest |
| AC-006 | 학습 로그 확인 | wandb |
| AC-007 | 평가 스크립트 | pytest |
| AC-008 | wandb 대시보드 | 수동 확인 |
| AC-009 | 성능 벤치마크 | pytest-benchmark |
| AC-010 | 재현성 테스트 | pytest |
| AC-011 | CI/CD 파이프라인 | GitHub Actions |

---

*인수 기준 버전: 1.0.0*
*작성일: 2025-01-18*
*담당 에이전트: expert-backend*
