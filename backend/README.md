# MIRIP Backend

MIRIP (MIRror Improvement Project) 백엔드 서비스입니다.
DINOv2 기반 AI 평가 모델을 통해 아트워크 품질을 자동으로 평가합니다.

## 아키텍처 개요

```
backend/
├── app/                    # FastAPI 애플리케이션
│   ├── ml/                 # AI 모델 모듈
│   │   ├── feature_extractor.py  # DINOv2 Feature Extractor
│   │   ├── projector.py          # Feature Projector
│   │   └── ranking_model.py      # Pairwise Ranking Model
│   ├── routers/            # API 라우터
│   ├── services/           # 비즈니스 로직
│   └── models/             # Pydantic 모델
├── training/               # 학습 모듈
│   ├── config/             # 학습 설정
│   ├── datasets/           # 데이터셋 클래스
│   ├── losses/             # 손실 함수
│   ├── scripts/            # 학습 스크립트
│   ├── trainer.py          # Trainer 클래스
│   ├── evaluator.py        # Evaluator 클래스
│   └── benchmarks.py       # 성능 벤치마크
├── data_pipeline/          # 데이터 파이프라인
└── tests/                  # 테스트 코드
```

## AI 평가 모델 (SPEC-AI-001)

### 모델 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    PairwiseRankingModel                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image (768x768x3)                                             │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────┐                                   │
│  │  DINOv2FeatureExtractor │  (Frozen, facebook/dinov2-large)  │
│  │  Output: 1024-d vector  │                                   │
│  └──────────┬──────────────┘                                   │
│             │                                                   │
│             ▼                                                   │
│  ┌─────────────────────────┐                                   │
│  │      Projector          │  (Trainable, MLP)                 │
│  │  1024 → 512 → 256-d     │                                   │
│  │  + LayerNorm + GELU     │                                   │
│  └──────────┬──────────────┘                                   │
│             │                                                   │
│             ▼                                                   │
│  ┌─────────────────────────┐                                   │
│  │      Score Head         │  (Trainable, Linear)              │
│  │  256 → 64 → 1 scalar    │                                   │
│  └──────────┬──────────────┘                                   │
│             │                                                   │
│             ▼                                                   │
│      Quality Score                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트

| 컴포넌트 | 설명 | 입출력 |
|---------|------|-------|
| `DINOv2FeatureExtractor` | 사전학습된 DINOv2 모델로 이미지 특징 추출 | (B, 3, H, W) → (B, 1024) |
| `Projector` | 고차원 특징을 저차원 임베딩 공간으로 투영 | (B, 1024) → (B, 256) |
| `PairwiseRankingModel` | 두 이미지의 품질 비교 | (img1, img2) → (score1, score2) |
| `Trainer` | 모델 학습 관리 | - |
| `Evaluator` | 모델 평가 및 메트릭 계산 | - |

### 학습 파이프라인

```
Data Preparation         Training              Evaluation
      │                     │                      │
      ▼                     ▼                      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ DataSplitter │    │   Trainer    │    │    Evaluator     │
│  80/10/10    │───▶│  AdamW       │───▶│ Pairwise Acc     │
│  Stratified  │    │  Cosine LR   │    │ Target: 60%+     │
└──────────────┘    │  Early Stop  │    └──────────────────┘
       │            └──────────────┘
       ▼
┌──────────────┐
│PairwiseDataset│
│  Cross-tier  │
│    pairs     │
└──────────────┘
```

## 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. 모델 학습

```bash
# 기본 학습 실행
python -m training.scripts.train \
    --data-dir /path/to/images \
    --metadata /path/to/metadata.csv \
    --output-dir ./checkpoints \
    --epochs 100 \
    --batch-size 32

# wandb 로깅과 함께 학습
python -m training.scripts.train \
    --data-dir /path/to/images \
    --metadata /path/to/metadata.csv \
    --wandb-project mirip-ranking \
    --wandb-run-name experiment-001
```

### 3. 모델 평가

```bash
# 테스트 셋에서 평가
python -m training.scripts.evaluate \
    --checkpoint ./checkpoints/best_model.pt \
    --test-data /path/to/test_metadata.csv \
    --image-dir /path/to/images
```

### 4. 추론

```python
import torch
from app.ml.ranking_model import PairwiseRankingModel
from torchvision import transforms
from PIL import Image

# 모델 로드
model = PairwiseRankingModel()
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 로드 및 비교
img1 = transform(Image.open("image1.jpg")).unsqueeze(0)
img2 = transform(Image.open("image2.jpg")).unsqueeze(0)

with torch.no_grad():
    score1, score2 = model(img1, img2)

if score1 > score2:
    print("Image 1 has higher quality")
else:
    print("Image 2 has higher quality")
```

## 설정 옵션

### TrainingConfig

```python
from training.config import TrainingConfig

config = TrainingConfig(
    # Optimizer 설정
    learning_rate=1e-4,      # 학습률
    weight_decay=0.01,       # L2 정규화

    # Training 설정
    batch_size=32,           # 배치 크기
    max_epochs=100,          # 최대 에폭
    gradient_clip_norm=1.0,  # 그래디언트 클리핑

    # Early stopping
    early_stopping_patience=10,  # 인내 횟수

    # Scheduler
    scheduler_t_max=100,     # Cosine annealing 주기
    scheduler_eta_min=1e-6,  # 최소 학습률

    # Checkpoint
    checkpoint_dir="./checkpoints",
    save_every_n_epochs=5,

    # wandb
    wandb_enabled=True,
    wandb_project="mirip-ranking",

    # Device
    device="cuda",           # cuda 또는 cpu
    seed=42,                 # 재현성을 위한 시드
)
```

## 성능 요구사항 (AC-009)

| 항목 | 요구사항 | 비고 |
|-----|---------|-----|
| 추론 시간 | < 100ms/pair | GPU 기준 |
| GPU 메모리 | < 12GB | batch_size=32 |
| 학습 시간 | < 6시간 | 10K 샘플 기준 |
| 테스트 정확도 | >= 60% | Pairwise accuracy |

## 테스트

```bash
# 전체 테스트 실행
pytest tests/ -v

# 커버리지 리포트
pytest tests/ --cov=app/ml --cov=training --cov-report=html

# 느린 테스트 제외 (DINOv2 로드 필요)
pytest tests/ -v -m "not slow"

# 통합 테스트만 실행
pytest tests/integration/ -v
```

## API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 추가 문서

- [학습 가이드](docs/training_guide.md) - 상세한 학습 방법
- [API 레퍼런스](docs/api_reference.md) - 모델 API 상세 문서

## 라이선스

이 프로젝트는 비공개입니다. 무단 복제 및 배포를 금합니다.
