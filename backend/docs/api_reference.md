# MIRIP AI 모델 API 레퍼런스

MIRIP AI 평가 모델의 상세 API 문서입니다.

## 목차

1. [ML 모듈](#ml-모듈)
   - [DINOv2FeatureExtractor](#dinov2featureextractor)
   - [Projector](#projector)
   - [PairwiseRankingModel](#pairwiserankingmodel)
2. [학습 모듈](#학습-모듈)
   - [TrainingConfig](#trainingconfig)
   - [Trainer](#trainer)
   - [Evaluator](#evaluator)
   - [PerformanceBenchmarks](#performancebenchmarks)
3. [데이터셋 모듈](#데이터셋-모듈)
   - [PairwiseDataset](#pairwisedataset)
   - [DataSplitter](#datasplitter)
4. [유틸리티 함수](#유틸리티-함수)

---

## ML 모듈

### DINOv2FeatureExtractor

`app.ml.feature_extractor.DINOv2FeatureExtractor`

DINOv2 사전학습 모델을 사용하여 이미지에서 특징 벡터를 추출합니다.

#### 초기화

```python
DINOv2FeatureExtractor(
    model_name: str = "facebook/dinov2-large",
    device: Optional[str] = None
)
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|-----|-------|------|
| `model_name` | str | `"facebook/dinov2-large"` | Hugging Face 모델 이름 |
| `device` | Optional[str] | None | 디바이스 (None이면 자동 감지) |

**속성:**

| 속성 | 타입 | 설명 |
|-----|-----|------|
| `output_dim` | int | 출력 특징 벡터 차원 (1024) |
| `device` | str | 현재 디바이스 |

#### 메서드

##### `extract(images: torch.Tensor) -> torch.Tensor`

이미지 배치에서 특징 벡터를 추출합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `images` | torch.Tensor | 입력 이미지 배치 (B, 3, H, W) |

**반환값:**

| 타입 | 설명 |
|-----|------|
| torch.Tensor | 특징 벡터 (B, 1024) |

**예시:**

```python
from app.ml.feature_extractor import DINOv2FeatureExtractor
import torch

extractor = DINOv2FeatureExtractor(device="cuda")
images = torch.randn(4, 3, 768, 768).cuda()
features = extractor.extract(images)
print(features.shape)  # torch.Size([4, 1024])
```

##### `forward(images: torch.Tensor) -> torch.Tensor`

`extract`와 동일. nn.Module 호환성을 위해 제공됩니다.

---

### Projector

`app.ml.projector.Projector`

고차원 특징 벡터를 저차원 임베딩 공간으로 투영하는 MLP 네트워크입니다.

#### 초기화

```python
Projector(
    input_dim: int = 1024,
    hidden_dim: int = 512,
    output_dim: int = 256,
    dropout: float = 0.1
)
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|-----|-------|------|
| `input_dim` | int | 1024 | 입력 특징 차원 |
| `hidden_dim` | int | 512 | 은닉층 차원 |
| `output_dim` | int | 256 | 출력 임베딩 차원 |
| `dropout` | float | 0.1 | 드롭아웃 비율 |

**아키텍처:**

```
Linear(input_dim, hidden_dim)
    ↓
LayerNorm(hidden_dim)
    ↓
GELU()
    ↓
Dropout(dropout)
    ↓
Linear(hidden_dim, output_dim)
    ↓
LayerNorm(output_dim)
```

#### 메서드

##### `forward(x: torch.Tensor) -> torch.Tensor`

특징 벡터를 저차원으로 투영합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `x` | torch.Tensor | 입력 특징 벡터 (B, input_dim) |

**반환값:**

| 타입 | 설명 |
|-----|------|
| torch.Tensor | 투영된 임베딩 (B, output_dim) |

**예시:**

```python
from app.ml.projector import Projector
import torch

projector = Projector(input_dim=1024, output_dim=256)
features = torch.randn(4, 1024)
embeddings = projector(features)
print(embeddings.shape)  # torch.Size([4, 256])
```

---

### PairwiseRankingModel

`app.ml.ranking_model.PairwiseRankingModel`

두 이미지의 품질을 비교하는 Pairwise Ranking 모델입니다.

#### 초기화

```python
PairwiseRankingModel(
    backbone_name: str = "facebook/dinov2-large",
    projector_output_dim: int = 256,
    freeze_backbone: bool = True,
    margin: float = 1.0,
    device: Optional[str] = None
)
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|-----|-------|------|
| `backbone_name` | str | `"facebook/dinov2-large"` | DINOv2 모델 이름 |
| `projector_output_dim` | int | 256 | Projector 출력 차원 |
| `freeze_backbone` | bool | True | 백본 가중치 동결 여부 |
| `margin` | float | 1.0 | MarginRankingLoss 마진 |
| `device` | Optional[str] | None | 디바이스 |

**속성:**

| 속성 | 타입 | 설명 |
|-----|-----|------|
| `feature_extractor` | DINOv2FeatureExtractor | 특징 추출기 |
| `projector` | Projector | 투영 네트워크 |
| `score_head` | nn.Sequential | 스코어 계산 헤드 |
| `loss_fn` | nn.MarginRankingLoss | 손실 함수 |

#### 메서드

##### `forward(img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

두 이미지의 품질 스코어를 계산합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `img1` | torch.Tensor | 첫 번째 이미지 배치 (B, 3, H, W) |
| `img2` | torch.Tensor | 두 번째 이미지 배치 (B, 3, H, W) |

**반환값:**

| 타입 | 설명 |
|-----|------|
| Tuple[torch.Tensor, torch.Tensor] | (score1, score2) 스코어 튜플, 각각 (B, 1) |

##### `compute_loss(score1: torch.Tensor, score2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor`

Margin Ranking Loss를 계산합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `score1` | torch.Tensor | 첫 번째 이미지 스코어 (B, 1) |
| `score2` | torch.Tensor | 두 번째 이미지 스코어 (B, 1) |
| `labels` | torch.Tensor | 라벨 (B,), 1: img1 > img2, -1: img1 < img2 |

**반환값:**

| 타입 | 설명 |
|-----|------|
| torch.Tensor | 손실 값 (스칼라) |

##### `get_score(image: torch.Tensor) -> torch.Tensor`

단일 이미지의 품질 스코어를 계산합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `image` | torch.Tensor | 입력 이미지 (B, 3, H, W) |

**반환값:**

| 타입 | 설명 |
|-----|------|
| torch.Tensor | 품질 스코어 (B, 1) |

**예시:**

```python
from app.ml.ranking_model import PairwiseRankingModel
import torch

model = PairwiseRankingModel(device="cuda")

# 이미지 페어 비교
img1 = torch.randn(4, 3, 768, 768).cuda()
img2 = torch.randn(4, 3, 768, 768).cuda()

score1, score2 = model(img1, img2)
print(f"Score1: {score1.shape}, Score2: {score2.shape}")

# 손실 계산
labels = torch.tensor([1, -1, 1, -1]).cuda()
loss = model.compute_loss(score1, score2, labels)
print(f"Loss: {loss.item():.4f}")
```

---

## 학습 모듈

### TrainingConfig

`training.config.TrainingConfig`

학습 설정을 담는 데이터 클래스입니다.

#### 속성

| 속성 | 타입 | 기본값 | 설명 |
|-----|-----|-------|------|
| `learning_rate` | float | 1e-4 | 학습률 |
| `weight_decay` | float | 0.01 | L2 정규화 계수 |
| `batch_size` | int | 32 | 배치 크기 |
| `max_epochs` | int | 100 | 최대 에폭 수 |
| `gradient_clip_norm` | float | 1.0 | 그래디언트 클리핑 노름 |
| `early_stopping_patience` | int | 10 | Early stopping 인내 횟수 |
| `scheduler_t_max` | int | 100 | CosineAnnealing 주기 |
| `scheduler_eta_min` | float | 1e-6 | 최소 학습률 |
| `checkpoint_dir` | str | "./checkpoints" | 체크포인트 저장 디렉토리 |
| `save_every_n_epochs` | int | 5 | 체크포인트 저장 주기 |
| `wandb_enabled` | bool | False | wandb 로깅 활성화 |
| `wandb_project` | Optional[str] | None | wandb 프로젝트 이름 |
| `wandb_run_name` | Optional[str] | None | wandb 실행 이름 |
| `device` | str | "cuda" | 학습 디바이스 |
| `seed` | int | 42 | 랜덤 시드 |

**예시:**

```python
from training.config import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    max_epochs=100,
    wandb_enabled=True,
    wandb_project="mirip-ranking",
)
```

---

### Trainer

`training.trainer.Trainer`

모델 학습을 관리하는 클래스입니다.

#### 초기화

```python
Trainer(
    model: nn.Module,
    config: TrainingConfig
)
```

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `model` | nn.Module | 학습할 모델 (PairwiseRankingModel) |
| `config` | TrainingConfig | 학습 설정 |

#### 메서드

##### `fit(train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]`

모델을 학습합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `train_loader` | DataLoader | 학습 데이터로더 |
| `val_loader` | Optional[DataLoader] | 검증 데이터로더 |

**반환값:**

| 타입 | 설명 |
|-----|------|
| Dict[str, List[float]] | 학습 히스토리 (train_loss, val_loss, train_acc, val_acc) |

##### `train_epoch(dataloader: DataLoader) -> Tuple[float, float]`

한 에폭을 학습합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `dataloader` | DataLoader | 학습 데이터로더 |

**반환값:**

| 타입 | 설명 |
|-----|------|
| Tuple[float, float] | (평균 손실, 정확도) |

##### `validate(dataloader: DataLoader) -> Tuple[float, float]`

검증을 수행합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `dataloader` | DataLoader | 검증 데이터로더 |

**반환값:**

| 타입 | 설명 |
|-----|------|
| Tuple[float, float] | (평균 손실, 정확도) |

##### `save_checkpoint(filepath: str, epoch: int, accuracy: float) -> None`

체크포인트를 저장합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `filepath` | str | 저장 경로 |
| `epoch` | int | 현재 에폭 |
| `accuracy` | float | 현재 정확도 |

##### `load_checkpoint(filepath: str) -> Dict[str, Any]`

체크포인트를 로드합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `filepath` | str | 체크포인트 경로 |

**반환값:**

| 타입 | 설명 |
|-----|------|
| Dict[str, Any] | 체크포인트 딕셔너리 |

**예시:**

```python
from training.trainer import Trainer
from training.config import TrainingConfig

config = TrainingConfig(max_epochs=100)
trainer = Trainer(model=model, config=config)

# 학습 실행
history = trainer.fit(train_loader, val_loader)

# 결과 확인
print(f"최종 학습 손실: {history['train_loss'][-1]:.4f}")
print(f"최종 검증 정확도: {history['val_acc'][-1]:.2%}")
```

---

### Evaluator

`training.evaluator.Evaluator`

모델 평가를 수행하는 클래스입니다.

#### 초기화

```python
Evaluator(
    model: nn.Module,
    device: Optional[str] = None
)
```

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `model` | nn.Module | 평가할 모델 |
| `device` | Optional[str] | 디바이스 |

#### 메서드

##### `evaluate(dataloader: DataLoader) -> float`

Pairwise accuracy를 계산합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `dataloader` | DataLoader | 평가 데이터로더 |

**반환값:**

| 타입 | 설명 |
|-----|------|
| float | 정확도 (0.0 ~ 1.0) |

##### `evaluate_detailed(dataloader: DataLoader) -> Dict[str, float]`

상세 평가 결과를 반환합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `dataloader` | DataLoader | 평가 데이터로더 |

**반환값:**

| 타입 | 설명 |
|-----|------|
| Dict[str, float] | {"accuracy": float, "total_pairs": int, "correct_predictions": int} |

##### `predict_pair(img1: torch.Tensor, img2: torch.Tensor) -> Tuple[int, float, float]`

단일 페어를 예측합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `img1` | torch.Tensor | 첫 번째 이미지 (1, 3, H, W) |
| `img2` | torch.Tensor | 두 번째 이미지 (1, 3, H, W) |

**반환값:**

| 타입 | 설명 |
|-----|------|
| Tuple[int, float, float] | (예측, score1, score2), 예측: 1(img1>img2), -1(img1<img2), 0(동점) |

**예시:**

```python
from training.evaluator import Evaluator

evaluator = Evaluator(model=model, device="cuda")

# 전체 평가
accuracy = evaluator.evaluate(test_loader)
print(f"테스트 정확도: {accuracy:.2%}")

# 상세 결과
results = evaluator.evaluate_detailed(test_loader)
print(f"정확한 예측: {results['correct_predictions']}/{results['total_pairs']}")
```

---

### PerformanceBenchmarks

`training.benchmarks.PerformanceBenchmarks`

성능 벤치마크를 측정하는 클래스입니다.

#### 클래스 상수

| 상수 | 값 | 설명 |
|-----|---|------|
| `MAX_INFERENCE_TIME_SECONDS` | 0.1 | 최대 추론 시간 (100ms) |
| `MAX_MEMORY_GB` | 12.0 | 최대 GPU 메모리 (12GB) |
| `TARGET_TRAINING_HOURS` | 6.0 | 목표 학습 시간 (6시간) |

#### 초기화

```python
PerformanceBenchmarks(
    model: nn.Module,
    device: Optional[str] = None
)
```

#### 메서드

##### `measure_inference_time(num_pairs: int = 100, warmup_iterations: int = 10, image_size: int = 768) -> float`

단일 페어당 평균 추론 시간을 측정합니다.

**반환값:** 초 단위 추론 시간

##### `measure_memory_usage(batch_size: int = 32, image_size: int = 768) -> int`

추론 시 메모리 사용량을 측정합니다.

**반환값:** 바이트 단위 메모리 사용량

##### `measure_training_memory(batch_size: int = 32, image_size: int = 768) -> float`

학습 시 GPU 메모리 사용량을 측정합니다.

**반환값:** GB 단위 메모리 사용량

##### `estimate_training_time(num_epochs: int = 100, samples_per_epoch: int = 10000, batch_size: int = 32) -> float`

총 학습 시간을 추정합니다.

**반환값:** 시간 단위 예상 학습 시간

##### `validate_reproducibility(dataloader: DataLoader, seed: int = 42, num_runs: int = 2) -> Tuple[bool, float]`

재현 가능성을 검증합니다.

**반환값:** (재현 가능 여부, 정확도 차이)

##### `run_full_benchmark(batch_size: int = 32, image_size: int = 768) -> Dict[str, Any]`

전체 벤치마크를 실행합니다.

**반환값:**

```python
{
    "inference_time_per_pair": float,
    "inference_time_ms": float,
    "memory_usage_bytes": int,
    "memory_usage_mb": float,
    "memory_usage_gb": float,
    "device": str,
    "batch_size": int,
    "image_size": int,
    "meets_inference_requirement": bool,
    "meets_memory_requirement": bool,
    "requirements": {
        "max_inference_time_ms": float,
        "max_memory_gb": float,
    },
}
```

**예시:**

```python
from training.benchmarks import PerformanceBenchmarks

benchmarks = PerformanceBenchmarks(model=model, device="cuda")
report = benchmarks.run_full_benchmark()

print(f"추론 시간: {report['inference_time_ms']:.2f}ms")
print(f"메모리: {report['memory_usage_gb']:.2f}GB")
print(f"요구사항 충족: {report['meets_inference_requirement'] and report['meets_memory_requirement']}")
```

---

## 데이터셋 모듈

### PairwiseDataset

`training.datasets.pairwise_dataset.PairwiseDataset`

Pairwise 학습을 위한 데이터셋입니다.

#### 초기화

```python
PairwiseDataset(
    metadata: pd.DataFrame,
    image_dir: str,
    transform: Optional[transforms.Compose] = None,
    tier_column: str = "tier",
    filename_column: str = "filename"
)
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|-----|-------|------|
| `metadata` | pd.DataFrame | - | 메타데이터 DataFrame |
| `image_dir` | str | - | 이미지 디렉토리 경로 |
| `transform` | Optional[Compose] | None | 이미지 전처리 |
| `tier_column` | str | "tier" | 등급 컬럼명 |
| `filename_column` | str | "filename" | 파일명 컬럼명 |

#### 메서드

##### `__len__() -> int`

데이터셋 크기 (생성된 페어 수)를 반환합니다.

##### `__getitem__(idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]`

인덱스에 해당하는 페어를 반환합니다.

**반환값:**

| 타입 | 설명 |
|-----|------|
| Tuple[Tensor, Tensor, int] | (img1, img2, label), label=1: img1>img2, label=-1: img1<img2 |

**예시:**

```python
from training.datasets.pairwise_dataset import PairwiseDataset
import pandas as pd

metadata = pd.read_csv("data/metadata.csv")
dataset = PairwiseDataset(
    metadata=metadata,
    image_dir="data/images",
    transform=transform
)

print(f"총 페어 수: {len(dataset)}")
img1, img2, label = dataset[0]
print(f"이미지 형태: {img1.shape}")
print(f"라벨: {label}")
```

---

### DataSplitter

`training.datasets.data_splitter.DataSplitter`

데이터를 학습/검증/테스트 셋으로 분할합니다.

#### 초기화

```python
DataSplitter(
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_column: Optional[str] = None
)
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|-----|-------|------|
| `train_ratio` | float | 0.8 | 학습 데이터 비율 |
| `val_ratio` | float | 0.1 | 검증 데이터 비율 |
| `test_ratio` | float | 0.1 | 테스트 데이터 비율 |
| `seed` | int | 42 | 랜덤 시드 |
| `stratify_column` | Optional[str] | None | 층화 분할 컬럼 |

#### 메서드

##### `split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`

데이터를 분할합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---------|-----|------|
| `data` | pd.DataFrame | 분할할 데이터 |

**반환값:**

| 타입 | 설명 |
|-----|------|
| Tuple[DataFrame, DataFrame, DataFrame] | (train_df, val_df, test_df) |

##### `get_split_info() -> Dict[str, Any]`

분할 정보를 반환합니다.

**반환값:**

```python
{
    "train_ratio": float,
    "val_ratio": float,
    "test_ratio": float,
    "seed": int,
    "stratify_column": Optional[str],
}
```

**예시:**

```python
from training.datasets.data_splitter import DataSplitter
import pandas as pd

splitter = DataSplitter(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    stratify_column="tier"
)

data = pd.read_csv("data/metadata.csv")
train_df, val_df, test_df = splitter.split(data)

print(f"학습: {len(train_df)}, 검증: {len(val_df)}, 테스트: {len(test_df)}")
```

---

## 유틸리티 함수

### set_seed

`training.benchmarks.set_seed`

재현 가능성을 위해 모든 랜덤 시드를 설정합니다.

```python
set_seed(seed: int = 42) -> None
```

**매개변수:**

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|-----|-------|------|
| `seed` | int | 42 | 시드 값 |

**설정되는 시드:**

- Python `random`
- NumPy `np.random`
- PyTorch `torch.manual_seed`
- CUDA `torch.cuda.manual_seed_all`
- cuDNN deterministic 모드

**예시:**

```python
from training.benchmarks import set_seed

# 재현 가능성 보장
set_seed(42)

# 동일한 랜덤 결과
import torch
tensor1 = torch.randn(10, 10)

set_seed(42)
tensor2 = torch.randn(10, 10)

assert torch.allclose(tensor1, tensor2)  # True
```

---

## 타입 정의

### 공통 타입

```python
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
```

### 반환 타입 요약

| 함수/메서드 | 반환 타입 |
|-----------|---------|
| `DINOv2FeatureExtractor.extract()` | `torch.Tensor` (B, 1024) |
| `Projector.forward()` | `torch.Tensor` (B, output_dim) |
| `PairwiseRankingModel.forward()` | `Tuple[torch.Tensor, torch.Tensor]` |
| `Trainer.fit()` | `Dict[str, List[float]]` |
| `Evaluator.evaluate()` | `float` |
| `DataSplitter.split()` | `Tuple[DataFrame, DataFrame, DataFrame]` |

---

## 에러 코드

| 에러 | 원인 | 해결 방법 |
|-----|------|---------|
| `RuntimeError: CUDA out of memory` | GPU 메모리 부족 | batch_size 감소 |
| `ValueError: Invalid tier value` | 잘못된 등급 값 | S/A/B/C 중 하나 사용 |
| `FileNotFoundError: Image not found` | 이미지 파일 없음 | image_dir 경로 확인 |
| `AssertionError: Ratios must sum to 1.0` | 분할 비율 오류 | 비율 합이 1.0이 되도록 조정 |

---

## 버전 정보

- API 버전: 1.0.0
- 최소 Python 버전: 3.10
- 최소 PyTorch 버전: 2.0.0
- 최소 transformers 버전: 4.30.0
