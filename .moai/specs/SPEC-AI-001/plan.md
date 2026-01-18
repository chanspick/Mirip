# SPEC-AI-001: 구현 계획

## TAG BLOCK

```yaml
spec_id: SPEC-AI-001
document_type: plan
status: Planned
created: 2025-01-18
updated: 2025-01-18
```

---

## 1. 구현 마일스톤

### Phase 1: 기반 구조 구축 (Primary Goal)

**목표:** 프로젝트 구조 및 핵심 컴포넌트 구현

| 태스크 | 설명 | 우선순위 | 의존성 |
|--------|------|----------|--------|
| T1.1 | 디렉토리 구조 생성 | High | - |
| T1.2 | DINOv2 Feature Extractor 구현 | High | T1.1 |
| T1.3 | Projector Layer 구현 | High | T1.1 |
| T1.4 | 단위 테스트 작성 (Feature Extractor, Projector) | High | T1.2, T1.3 |

**산출물:**
- `backend/app/ml/feature_extractor.py`
- `backend/app/ml/projector.py`
- `backend/tests/test_feature_extractor.py`
- `backend/tests/test_projector.py`

---

### Phase 2: 데이터 파이프라인 (Primary Goal)

**목표:** Pairwise 학습 데이터 준비

| 태스크 | 설명 | 우선순위 | 의존성 |
|--------|------|----------|--------|
| T2.1 | Pairwise Dataset 클래스 구현 | High | T1.1 |
| T2.2 | 데이터 분할 유틸리티 구현 (80/10/10) | High | T2.1 |
| T2.3 | Pair 생성 스크립트 구현 | High | T2.1, T2.2 |
| T2.4 | SPEC-DATA-001 데이터셋 연동 | High | T2.1 |
| T2.5 | 데이터 로더 테스트 | Medium | T2.1, T2.2, T2.3 |

**산출물:**
- `backend/training/datasets/pairwise_dataset.py`
- `backend/training/datasets/data_splitter.py`
- `backend/training/scripts/generate_pairs.py`
- `backend/tests/test_pairwise_dataset.py`

---

### Phase 3: 모델 및 학습 구현 (Primary Goal)

**목표:** Pairwise Ranking Model 및 학습 루프 구현

| 태스크 | 설명 | 우선순위 | 의존성 |
|--------|------|----------|--------|
| T3.1 | PairwiseRankingModel 클래스 구현 | High | T1.2, T1.3 |
| T3.2 | MarginRankingLoss 래퍼 구현 | High | T3.1 |
| T3.3 | Trainer 모듈 구현 | High | T3.1, T3.2, T2.1 |
| T3.4 | wandb 통합 | Medium | T3.3 |
| T3.5 | 체크포인트 저장/로드 구현 | Medium | T3.3 |
| T3.6 | 학습 스크립트 작성 | High | T3.3, T3.4, T3.5 |

**산출물:**
- `backend/app/ml/ranking_model.py`
- `backend/training/losses/margin_ranking_loss.py`
- `backend/app/ml/trainer.py`
- `backend/training/scripts/train_baseline.py`
- `backend/training/config/baseline_config.yaml`

---

### Phase 4: 평가 및 검증 (Secondary Goal)

**목표:** 모델 평가 및 성능 검증

| 태스크 | 설명 | 우선순위 | 의존성 |
|--------|------|----------|--------|
| T4.1 | Evaluator 모듈 구현 | High | T3.1 |
| T4.2 | Pairwise Accuracy 계산 구현 | High | T4.1 |
| T4.3 | 평가 스크립트 작성 | High | T4.1, T4.2 |
| T4.4 | 성능 리포트 생성 | Medium | T4.3 |
| T4.5 | 실험 결과 문서화 | Low | T4.4 |

**산출물:**
- `backend/app/ml/evaluator.py`
- `backend/training/scripts/evaluate_baseline.py`
- `backend/training/results/` (실험 결과)

---

### Phase 5: 통합 및 최적화 (Final Goal)

**목표:** 전체 파이프라인 통합 및 성능 최적화

| 태스크 | 설명 | 우선순위 | 의존성 |
|--------|------|----------|--------|
| T5.1 | End-to-end 파이프라인 테스트 | High | Phase 1-4 |
| T5.2 | 성능 최적화 (추론 속도) | Medium | T5.1 |
| T5.3 | 메모리 최적화 | Medium | T5.1 |
| T5.4 | 문서화 및 사용 가이드 작성 | Low | T5.1 |
| T5.5 | 코드 리뷰 및 리팩토링 | Medium | T5.1 |

**산출물:**
- 통합 테스트 스크립트
- 성능 벤치마크 결과
- 사용자 가이드 문서

---

## 2. 기술 접근 방식

### 2.1 DINOv2 Feature Extraction 전략

**선택 이유:**
- DINOv2 ViT-L은 visual similarity에서 CLIP 대비 2배 이상 정확도 (tech.md 근거)
- Self-supervised learning으로 풍부한 시각적 피처 학습
- 미술 작품의 구조적/질감적 특성 포착에 적합

**구현 전략:**
1. Hugging Face Transformers를 통한 모델 로드
2. CLS 토큰 피처 사용 (1024차원)
3. 모든 파라미터 freeze (전이 학습)
4. fp16 mixed precision으로 메모리 최적화

### 2.2 Projector Layer 설계

**설계 원칙:**
1. 차원 축소: 1024 -> 512 -> 256
2. LayerNorm으로 학습 안정화
3. GELU 활성화 함수 (Transformer 계열과 일관성)
4. Dropout으로 과적합 방지

**하이퍼파라미터 결정 근거:**
- Hidden dimension 512: 충분한 표현력 유지
- Output dimension 256: 비교 학습에 적합한 크기
- Dropout 0.1: 적절한 정규화

### 2.3 Pairwise Ranking 학습 전략

**접근 방식:**
1. 티어(S/A/B/C) 차이가 있는 이미지 쌍만 사용
2. MarginRankingLoss로 상대적 순서 학습
3. 동일 티어 쌍은 학습에서 제외 (노이즈 방지)

**장점:**
- 절대 점수보다 상대 비교가 레이블링 품질 높음
- 불확실한 데이터 (동일 티어) 제외로 노이즈 감소
- 간단하면서도 효과적인 접근

### 2.4 학습 최적화 전략

**메모리 최적화:**
- fp16 mixed precision training
- Gradient checkpointing (필요시)
- Batch size 32 (RTX 4070 Ti Super 16GB에 적합)

**학습 안정화:**
- AdamW optimizer with weight decay
- Cosine annealing LR scheduler
- Early stopping (patience=10)

---

## 3. 아키텍처 설계

### 3.1 컴포넌트 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Pairwise   │    │   Ranking    │    │   Trainer    │       │
│  │   Dataset    │───▶│   Model      │───▶│   Module     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    Data      │    │   Feature    │    │   wandb     │       │
│  │   Splitter   │    │   Extractor  │    │   Logger    │       │
│  └──────────────┘    │   + Projector│    └──────────────┘       │
│                      └──────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 데이터 흐름

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw       │    │  Metadata   │    │   Pairs     │
│   Images    │───▶│   JSON      │───▶│   CSV       │
│   (from     │    │   (tier,    │    │   (img_a,   │
│   DATA-001) │    │   dept)     │    │   img_b,    │
└─────────────┘    └─────────────┘    │   label)    │
                                       └──────┬──────┘
                                              │
                   ┌──────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│                  Training Loop                       │
├─────────────────────────────────────────────────────┤
│  1. Load image pair from dataset                    │
│  2. Extract features with DINOv2 (frozen)           │
│  3. Project features (256-d)                        │
│  4. Compute pairwise scores                         │
│  5. Calculate MarginRankingLoss                     │
│  6. Update Projector weights                        │
│  7. Log to wandb                                    │
└─────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│               Evaluation                             │
├─────────────────────────────────────────────────────┤
│  1. Load test pairs                                 │
│  2. Compute pairwise scores                         │
│  3. Calculate pairwise accuracy                     │
│  4. Generate performance report                     │
└─────────────────────────────────────────────────────┘
```

### 3.3 클래스 다이어그램

```
┌────────────────────────────────────┐
│       DINOv2FeatureExtractor       │
├────────────────────────────────────┤
│ - model: Dinov2Model               │
│ - processor: AutoImageProcessor    │
│ - device: str                      │
├────────────────────────────────────┤
│ + __init__(device)                 │
│ + extract(image) -> Tensor         │
│ + extract_batch(images) -> Tensor  │
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│           Projector                │
├────────────────────────────────────┤
│ - projection: nn.Sequential        │
├────────────────────────────────────┤
│ + __init__(input_dim, ...)         │
│ + forward(x) -> Tensor             │
└────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────┐
│      PairwiseRankingModel          │
├────────────────────────────────────┤
│ - feature_extractor                │
│ - projector                        │
├────────────────────────────────────┤
│ + forward_single(image) -> Tensor  │
│ + forward(img_a, img_b) -> Tuple   │
│ + predict(img_a, img_b) -> int     │
└────────────────────────────────────┘
```

---

## 4. 리스크 분석

### 4.1 기술적 리스크

| 리스크 | 확률 | 영향도 | 대응 전략 |
|--------|------|--------|-----------|
| 60% 정확도 미달 | Medium | High | Projector 구조 변경, 데이터 증강, 앙상블 |
| GPU 메모리 부족 | Low | Medium | Batch size 감소, Gradient accumulation |
| 학습 불안정 | Low | Medium | LR 조정, Warmup 추가, Gradient clipping |
| 과적합 | Medium | Medium | Dropout 증가, Early stopping 강화 |

### 4.2 일정 리스크

| 리스크 | 확률 | 영향도 | 대응 전략 |
|--------|------|--------|-----------|
| SPEC-DATA-001 데이터 품질 이슈 | Low | High | 데이터 검증 선행, 필터링 강화 |
| 실험 시간 초과 | Medium | Low | 실험 우선순위 설정, 병렬 실험 |
| 예상치 못한 버그 | Medium | Medium | 단위 테스트 선행, CI/CD 구축 |

### 4.3 완화 계획

**정확도 미달 시 대응:**
1. Projector 레이어 깊이 증가 (3층 -> 4층)
2. 데이터 증강 (회전, 플립, 색상 변환)
3. Hard negative mining 적용
4. 앙상블 모델 사용

**메모리 이슈 대응:**
1. Gradient accumulation으로 effective batch size 유지
2. 이미지 해상도 축소 (768 -> 518)
3. Gradient checkpointing 적용

---

## 5. 의존성 관리

### 5.1 외부 의존성

```toml
# requirements.txt / pyproject.toml

# Core ML
torch>=2.1.0
transformers>=4.35.0
timm>=0.9.0

# Training
pytorch-lightning>=2.1.0
wandb>=0.16.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Code Quality
ruff>=0.1.0
mypy>=1.7.0
```

### 5.2 내부 의존성

| 모듈 | 의존 대상 |
|------|-----------|
| `ranking_model.py` | `feature_extractor.py`, `projector.py` |
| `trainer.py` | `ranking_model.py`, `pairwise_dataset.py` |
| `evaluator.py` | `ranking_model.py` |
| `train_baseline.py` | `trainer.py`, `baseline_config.yaml` |

### 5.3 SPEC 의존성

```
SPEC-DATA-001 (완료) ────▶ SPEC-AI-001 (현재)
                                 │
                                 ▼
                          ┌─────────────┐
                          │ SPEC-AI-002 │
                          │ (4축 평가)   │
                          └─────────────┘
                                 │
                                 ▼
                          ┌─────────────┐
                          │ SPEC-AI-003 │
                          │ (티어분류기) │
                          └─────────────┘
```

---

## 6. 테스트 전략

### 6.1 단위 테스트

| 모듈 | 테스트 항목 |
|------|-------------|
| `feature_extractor.py` | 피처 차원 검증, 배치 처리, 디바이스 호환성 |
| `projector.py` | 출력 차원 검증, 그래디언트 흐름, 드롭아웃 |
| `ranking_model.py` | forward pass, score 계산, 예측 일관성 |
| `pairwise_dataset.py` | 데이터 로딩, 라벨 정확성, 분할 검증 |

### 6.2 통합 테스트

| 테스트 | 설명 |
|--------|------|
| 학습 루프 테스트 | 1 epoch 학습 후 loss 감소 확인 |
| 체크포인트 테스트 | 저장/로드 후 모델 상태 동일성 |
| 평가 파이프라인 테스트 | 전체 평가 흐름 검증 |

### 6.3 성능 테스트

| 테스트 | 목표 |
|--------|------|
| 추론 속도 | 이미지당 < 100ms |
| 메모리 사용량 | < 12GB (학습 시) |
| 학습 시간 | < 6시간 (전체) |

---

## 7. 모니터링 및 로깅

### 7.1 wandb 메트릭

```yaml
# 학습 메트릭
train/loss: MarginRankingLoss 값
train/accuracy: 학습 셋 pairwise accuracy
train/lr: 현재 learning rate

# 검증 메트릭
val/loss: 검증 셋 loss
val/accuracy: 검증 셋 pairwise accuracy

# 시스템 메트릭
system/gpu_memory: GPU 메모리 사용량
system/epoch_time: epoch 당 소요 시간
```

### 7.2 로깅 형식

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

---

*계획 버전: 1.0.0*
*작성일: 2025-01-18*
*담당 에이전트: expert-backend*
