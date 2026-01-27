# SPEC-AI-001: DINOv2 Baseline AI 평가 모델

## TAG BLOCK

```yaml
spec_id: SPEC-AI-001
title: DINOv2 Baseline AI 평가 모델
status: Completed
priority: High
created: 2025-01-18
updated: 2025-01-18
lifecycle: spec-anchored
domain: machine-learning
assigned: expert-backend
related_specs: [SPEC-DATA-001, SPEC-BACKEND-001]
labels: [dinov2, pairwise-ranking, baseline, ml-training, feature-extraction]
estimated_complexity: Medium
target_accuracy: 60%
```

---

## 1. Environment (환경)

### 1.1 기술 스택

| 영역 | 기술 | 버전 | 용도 |
|------|------|------|------|
| Framework | PyTorch | 2.1+ | 딥러닝 프레임워크 |
| Transformers | Hugging Face Transformers | 4.35+ | DINOv2 모델 로드 |
| Vision Models | timm | 0.9+ | 비전 모델 라이브러리 |
| Training | pytorch-lightning | 2.1+ | 학습 프레임워크 |
| Experiment Tracking | wandb | 0.16+ | 실험 추적 |
| Data Processing | numpy | 1.24+ | 수치 연산 |
| Data Processing | pandas | 2.0+ | 데이터 관리 |
| Image Processing | Pillow | 10.0+ | 이미지 I/O |
| Language | Python | 3.10+ | 개발 언어 |

### 1.2 모델 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    DINOv2 Baseline Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Image A    │    │  DINOv2      │    │  Feature     │       │
│  │   (768x768)  │───▶│  ViT-L/14    │───▶│  Vector      │       │
│  └──────────────┘    │  (frozen)    │    │  (1024-d)    │       │
│                      └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────▼───────┐       │
│  │   Image B    │    │  DINOv2      │    │  Projector   │       │
│  │   (768x768)  │───▶│  ViT-L/14    │───▶│  Layer       │       │
│  └──────────────┘    │  (frozen)    │    │  (256-d)     │       │
│                      └──────────────┘    └──────┬───────┘       │
│                                                  │               │
│                      ┌──────────────┐    ┌──────▼───────┐       │
│                      │  Pairwise    │◀───│  Ranking     │       │
│                      │  Comparison  │    │  Score       │       │
│                      └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 디렉토리 구조

```
backend/
├── app/
│   └── ml/                          # ML 모듈 (기존)
│       ├── __init__.py
│       ├── feature_extractor.py     # [신규] DINOv2 피처 추출기
│       ├── projector.py             # [신규] Projector 레이어
│       ├── ranking_model.py         # [신규] Pairwise Ranking 모델
│       ├── trainer.py               # [신규] 학습 모듈
│       └── evaluator.py             # [신규] 평가 모듈
│
├── training/                        # [신규] 학습 관련
│   ├── __init__.py
│   ├── config/
│   │   └── baseline_config.yaml     # 학습 설정
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── pairwise_dataset.py      # Pairwise 데이터셋
│   │   └── data_splitter.py         # 데이터 분할 유틸리티
│   ├── losses/
│   │   ├── __init__.py
│   │   └── margin_ranking_loss.py   # MarginRankingLoss 래퍼
│   ├── scripts/
│   │   ├── train_baseline.py        # 학습 스크립트
│   │   ├── evaluate_baseline.py     # 평가 스크립트
│   │   └── generate_pairs.py        # 페어 생성 스크립트
│   └── checkpoints/                 # 모델 체크포인트 (Git 제외)
│
├── tests/
│   ├── test_feature_extractor.py
│   ├── test_projector.py
│   ├── test_ranking_model.py
│   └── test_pairwise_dataset.py
│
└── weights/                         # 모델 가중치 (Git 제외)
    └── baseline_v1/
```

### 1.4 하드웨어 요구사항

| 항목 | 스펙 | 비고 |
|------|------|------|
| GPU | RTX 4070 Ti Super 16GB | 학습 환경 |
| VRAM 사용량 | ~12GB | batch 32, fp16 |
| 예상 학습 시간 | 4-6시간 | backbone freeze |
| 디스크 공간 | 10GB+ | 체크포인트 저장 |

---

## 2. Assumptions (가정)

### 2.1 기술적 가정

| ID | 가정 | 신뢰도 | 근거 | 검증 방법 |
|----|------|--------|------|-----------|
| A-001 | DINOv2 ViT-L이 미술 작품 피처 추출에 적합 | High | visual similarity에서 CLIP 대비 2배 정확도 (tech.md) | 피처 품질 평가 |
| A-002 | Backbone freeze로 충분한 성능 달성 가능 | Medium | 전이학습 일반적 패턴 | 실험으로 검증 |
| A-003 | Pairwise ranking이 절대 점수보다 효과적 | High | 상대적 비교가 레이블링 품질 향상 | 문헌 조사 |
| A-004 | 2,000개 이미지로 60%+ 정확도 달성 가능 | Medium | 데이터 규모 추정 | 실험으로 검증 |
| A-005 | RTX 4070 Ti Super로 학습 가능 | High | VRAM 12GB 예상 (tech.md) | 메모리 프로파일링 |

### 2.2 비즈니스 가정

| ID | 가정 | 신뢰도 | 근거 |
|----|------|--------|------|
| B-001 | 60% baseline 정확도가 MVP에 충분 | Medium | 초기 버전 목표 |
| B-002 | 4축 평가(구성력, 명암/질감, 조형완성도, 주제해석력)는 후속 SPEC에서 구현 | High | 단계별 접근 |
| B-003 | Pairwise 비교가 사용자 경험에 적합 | High | A vs B 형식 직관적 |

### 2.3 데이터 가정

| ID | 가정 | 신뢰도 | 근거 |
|----|------|--------|------|
| D-001 | SPEC-DATA-001 데이터셋 사용 가능 | High | SPEC-DATA-001 완료됨 |
| D-002 | 티어 라벨 (S/A/B/C)이 유효한 품질 지표 | High | product.md 명시 |
| D-003 | 80/10/10 분할이 적절한 비율 | High | ML 표준 관행 |

### 2.4 제약 조건

- [HARD] DINOv2 ViT-L backbone은 freeze 상태로 학습
- [HARD] 테스트셋은 학습/검증에 절대 사용 금지
- [HARD] 모든 실험은 wandb로 추적
- [HARD] 재현 가능성을 위해 random seed 고정
- [SOFT] 학습 시간 6시간 이내 목표
- [SOFT] 추론 시간 이미지당 100ms 이내 목표

---

## 3. Requirements (요구사항)

### 3.1 Ubiquitous Requirements (항상 적용)

| ID | 요구사항 |
|----|----------|
| REQ-U-001 | 시스템은 **항상** DINOv2 ViT-L 모델을 frozen 상태로 사용해야 한다 |
| REQ-U-002 | 시스템은 **항상** fp16 mixed precision을 사용하여 VRAM을 최적화해야 한다 |
| REQ-U-003 | 시스템은 **항상** 학습 로그를 wandb에 기록해야 한다 |
| REQ-U-004 | 시스템은 **항상** random seed를 42로 고정하여 재현 가능성을 보장해야 한다 |
| REQ-U-005 | 시스템은 **항상** 입력 이미지를 768x768 해상도로 정규화해야 한다 |

### 3.2 Event-Driven Requirements (이벤트 기반)

| ID | 요구사항 |
|----|----------|
| REQ-E-001 | **WHEN** 이미지가 입력되면 **THEN** DINOv2를 통해 1024차원 피처 벡터를 추출해야 한다 |
| REQ-E-002 | **WHEN** 피처 벡터가 추출되면 **THEN** Projector layer를 통해 256차원으로 변환해야 한다 |
| REQ-E-003 | **WHEN** 두 이미지의 피처가 준비되면 **THEN** pairwise ranking score를 계산해야 한다 |
| REQ-E-004 | **WHEN** epoch이 완료되면 **THEN** validation set에서 pairwise accuracy를 측정해야 한다 |
| REQ-E-005 | **WHEN** validation accuracy가 최고값을 갱신하면 **THEN** 모델 체크포인트를 저장해야 한다 |
| REQ-E-006 | **WHEN** 학습이 완료되면 **THEN** test set에서 최종 pairwise accuracy를 측정해야 한다 |

### 3.3 State-Driven Requirements (상태 기반)

| ID | 요구사항 |
|----|----------|
| REQ-S-001 | **IF** GPU 메모리가 부족하면 **THEN** batch size를 자동으로 감소시켜야 한다 |
| REQ-S-002 | **IF** validation loss가 10 epoch 동안 개선되지 않으면 **THEN** early stopping을 적용해야 한다 |
| REQ-S-003 | **IF** learning rate가 최소값 이하이면 **THEN** 학습을 종료해야 한다 |
| REQ-S-004 | **IF** 이미지 해상도가 768x768 미만이면 **THEN** 패딩 또는 리사이즈를 적용해야 한다 |
| REQ-S-005 | **IF** 체크포인트가 존재하면 **THEN** 학습을 이어서 재개할 수 있어야 한다 |

### 3.4 Unwanted Requirements (금지 사항)

| ID | 요구사항 |
|----|----------|
| REQ-N-001 | 시스템은 테스트 데이터를 학습 또는 검증에 사용**하지 않아야 한다** |
| REQ-N-002 | 시스템은 DINOv2 backbone의 가중치를 업데이트**하지 않아야 한다** |
| REQ-N-003 | 시스템은 random seed 없이 학습을 실행**하지 않아야 한다** |
| REQ-N-004 | 시스템은 wandb 로깅 없이 실험을 수행**하지 않아야 한다** |
| REQ-N-005 | 시스템은 validation 없이 체크포인트를 저장**하지 않아야 한다** |

### 3.5 Optional Requirements (선택 사항)

| ID | 요구사항 |
|----|----------|
| REQ-O-001 | **가능하면** gradient accumulation을 통해 effective batch size를 증가 |
| REQ-O-002 | **가능하면** cosine annealing learning rate scheduler 적용 |
| REQ-O-003 | **가능하면** 학습 중간에 샘플 예측 결과 시각화 제공 |
| REQ-O-004 | **가능하면** ONNX 형식으로 모델 내보내기 지원 |
| REQ-O-005 | **가능하면** ensemble을 위한 다중 체크포인트 저장 |

---

## 4. Specifications (세부 명세)

### 4.1 DINOv2 Feature Extractor 명세

#### 4.1.1 모델 로드

```python
from transformers import Dinov2Model, AutoImageProcessor

class DINOv2FeatureExtractor:
    """DINOv2 ViT-L/14 기반 피처 추출기"""

    MODEL_NAME = "facebook/dinov2-large"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = Dinov2Model.from_pretrained(self.MODEL_NAME)
        self.model.to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def extract(self, image: PIL.Image.Image) -> torch.Tensor:
        """이미지에서 1024차원 피처 벡터 추출"""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS 토큰 피처 반환
        return outputs.last_hidden_state[:, 0, :]  # Shape: (1, 1024)
```

#### 4.1.2 피처 출력 명세

| 속성 | 값 | 설명 |
|------|-----|------|
| 입력 크기 | 768x768 | DINOv2 기본 입력 크기 |
| 출력 차원 | 1024 | ViT-L 히든 차원 |
| 정규화 | ImageNet mean/std | 표준 정규화 |

### 4.2 Projector Layer 명세

#### 4.2.1 아키텍처

```python
class Projector(nn.Module):
    """피처 프로젝션 레이어"""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
```

#### 4.2.2 레이어 구성

| 레이어 | 입력 차원 | 출력 차원 | 활성화 함수 |
|--------|----------|----------|------------|
| Linear 1 | 1024 | 512 | - |
| LayerNorm 1 | 512 | 512 | - |
| GELU | 512 | 512 | GELU |
| Dropout | 512 | 512 | - |
| Linear 2 | 512 | 256 | - |
| LayerNorm 2 | 256 | 256 | - |

### 4.3 Pairwise Ranking Model 명세

#### 4.3.1 전체 모델

```python
class PairwiseRankingModel(nn.Module):
    """Pairwise Ranking 기반 미술 작품 평가 모델"""

    def __init__(
        self,
        feature_extractor: DINOv2FeatureExtractor,
        projector: Projector
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.projector = projector

    def forward_single(self, image: torch.Tensor) -> torch.Tensor:
        """단일 이미지의 임베딩 계산"""
        features = self.feature_extractor.extract(image)
        return self.projector(features)

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """두 이미지의 임베딩 및 랭킹 스코어 계산"""
        emb_a = self.forward_single(image_a)
        emb_b = self.forward_single(image_b)

        # L2 norm으로 스코어 계산 (낮을수록 좋음 -> 높을수록 좋게 변환)
        score_a = -torch.norm(emb_a, dim=-1)
        score_b = -torch.norm(emb_b, dim=-1)

        return score_a, score_b
```

### 4.4 학습 명세

#### 4.4.1 데이터셋 분할

| 분할 | 비율 | 예상 샘플 수 | 용도 |
|------|------|-------------|------|
| Train | 80% | ~1,600 | 모델 학습 |
| Validation | 10% | ~200 | 하이퍼파라미터 튜닝 |
| Test | 10% | ~200 | 최종 성능 평가 |

**분할 전략:**
- Stratified split by tier (S/A/B/C 비율 유지)
- 동일 작가의 작품은 같은 분할에 배치
- Random seed 42 고정

#### 4.4.2 Pairwise 데이터 생성

```python
def generate_pairs(
    metadata_df: pd.DataFrame,
    tier_order: dict = {"S": 4, "A": 3, "B": 2, "C": 1}
) -> list[tuple[str, str, int]]:
    """
    Pairwise 학습 데이터 생성

    Returns:
        list of (image_id_a, image_id_b, label)
        label: 1 if A > B, -1 if A < B
    """
    pairs = []
    for i, row_a in metadata_df.iterrows():
        for j, row_b in metadata_df.iterrows():
            if i >= j:
                continue

            tier_a = tier_order[row_a["tier"]]
            tier_b = tier_order[row_b["tier"]]

            if tier_a > tier_b:
                pairs.append((row_a["image_id"], row_b["image_id"], 1))
            elif tier_a < tier_b:
                pairs.append((row_a["image_id"], row_b["image_id"], -1))
            # 동일 티어는 제외 (불확실한 레이블)

    return pairs
```

#### 4.4.3 Loss Function

```python
# MarginRankingLoss 사용
criterion = nn.MarginRankingLoss(margin=1.0)

# 사용법:
# loss = criterion(score_a, score_b, target)
# target: 1 if score_a should be higher, -1 if score_b should be higher
```

#### 4.4.4 학습 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| Batch size | 32 | GPU 메모리에 맞춤 |
| Learning rate | 1e-4 | Projector 학습률 |
| Optimizer | AdamW | weight decay 포함 |
| Weight decay | 0.01 | 정규화 |
| Epochs | 100 | 최대 epoch |
| Early stopping patience | 10 | 조기 종료 기준 |
| LR scheduler | CosineAnnealing | T_max=100 |
| Margin | 1.0 | MarginRankingLoss |

### 4.5 평가 명세

#### 4.5.1 Pairwise Accuracy

```python
def compute_pairwise_accuracy(
    model: PairwiseRankingModel,
    test_pairs: list[tuple[str, str, int]],
    image_loader: Callable
) -> float:
    """
    Pairwise Accuracy 계산

    Accuracy = (올바르게 순위를 매긴 쌍의 수) / (전체 쌍의 수)
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for img_id_a, img_id_b, label in test_pairs:
            img_a = image_loader(img_id_a)
            img_b = image_loader(img_id_b)

            score_a, score_b = model(img_a, img_b)

            # 예측: score_a > score_b이면 1, 아니면 -1
            pred = 1 if score_a > score_b else -1

            if pred == label:
                correct += 1
            total += 1

    return correct / total
```

#### 4.5.2 성능 목표

| 지표 | 목표값 | 설명 |
|------|--------|------|
| Test Pairwise Accuracy | >= 60% | Baseline 목표 |
| Validation Pairwise Accuracy | >= 62% | 일반화 확인 |
| Train Pairwise Accuracy | <= 80% | 과적합 방지 |

---

## 5. Traceability (추적성)

### 5.1 연관 문서

| 문서 | 경로 | 관계 |
|------|------|------|
| 기술 스택 | `.moai/project/tech.md` | DINOv2 선택 근거, 하드웨어 스펙 |
| 제품 문서 | `.moai/project/product.md` | 4축 평가, 티어 분류 기준 |
| 프로젝트 구조 | `.moai/project/structure.md` | 디렉토리 구조 참조 |
| 데이터 파이프라인 | `.moai/specs/SPEC-DATA-001/spec.md` | 학습 데이터 제공 |
| 백엔드 SPEC | `.moai/specs/SPEC-BACKEND-001/spec.md` | API 통합 참조 |

### 5.2 후속 SPEC

| SPEC ID | 제목 | 의존성 |
|---------|------|--------|
| SPEC-AI-002 | 4축 Rubric 평가 헤드 구현 | 이 SPEC 완료 후 |
| SPEC-AI-003 | 티어 분류기 구현 | SPEC-AI-002 완료 후 |
| SPEC-AI-004 | CLIP 기반 주제 해석 모듈 | 이 SPEC과 병렬 가능 |
| SPEC-API-002 | AI 진단 API 통합 | SPEC-AI-003 완료 후 |

### 5.3 Quality Gates

- [ ] 모든 EARS 요구사항이 테스트 가능
- [ ] 85% 이상 테스트 커버리지
- [ ] Test Pairwise Accuracy >= 60%
- [ ] wandb 실험 기록 완료
- [ ] Ruff 린터 경고 없음
- [ ] 체크포인트 저장 및 로드 검증

---

*SPEC 버전: 1.0.0*
*작성일: 2025-01-18*
*담당 에이전트: expert-backend*
