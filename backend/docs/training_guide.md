# MIRIP AI 모델 학습 가이드

MIRIP (MIRror Improvement Project) AI 평가 모델의 학습 가이드입니다.
DINOv2 기반 Pairwise Ranking 모델을 학습하여 아트워크 품질을 자동으로 평가합니다.

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [데이터 준비](#데이터-준비)
3. [학습 프로세스](#학습-프로세스)
4. [평가 및 추론](#평가-및-추론)
5. [문제 해결](#문제-해결)

---

## 사전 요구사항

### 하드웨어 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|-----|---------|---------|
| GPU | NVIDIA GPU 8GB VRAM | NVIDIA RTX 3090/4090 24GB |
| CPU | 8코어 | 16코어 이상 |
| RAM | 16GB | 32GB 이상 |
| 저장공간 | 50GB | 100GB 이상 (SSD 권장) |

> **참고**: DINOv2-Large 모델은 약 1.1GB의 VRAM을 사용합니다.
> batch_size=32 기준으로 학습 시 약 12GB VRAM이 필요합니다.

### 소프트웨어 요구사항

```bash
# Python 버전
Python >= 3.10

# CUDA (GPU 학습 시)
CUDA >= 11.8
cuDNN >= 8.6
```

### 의존성 설치

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 기본 의존성 설치
pip install -r requirements.txt

# 개발 의존성 설치 (테스트, 린팅 등)
pip install -r requirements-dev.txt
```

#### 주요 의존성 목록

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
wandb>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## 데이터 준비

### 데이터 구조

학습 데이터는 다음과 같은 구조를 따라야 합니다:

```
data/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── metadata.csv
```

### 메타데이터 형식

`metadata.csv` 파일은 다음 컬럼을 포함해야 합니다:

| 컬럼명 | 타입 | 설명 | 예시 |
|-------|-----|------|------|
| `image_id` | string | 고유 이미지 ID | `img_001` |
| `filename` | string | 이미지 파일명 | `image_001.jpg` |
| `tier` | string | 품질 등급 (S/A/B/C) | `A` |

예시:

```csv
image_id,filename,tier
img_001,image_001.jpg,S
img_002,image_002.jpg,A
img_003,image_003.jpg,B
img_004,image_004.jpg,C
```

### 품질 등급 체계

| 등급 | 설명 | 순위 |
|-----|------|-----|
| S | 최상위 품질 (Superior) | 1 |
| A | 우수 품질 (Excellent) | 2 |
| B | 보통 품질 (Average) | 3 |
| C | 낮은 품질 (Below Average) | 4 |

> **중요**: Pairwise Ranking 학습에서는 S > A > B > C 순서가 적용됩니다.

### 데이터 분할

DataSplitter를 사용하여 데이터를 분할합니다:

```python
from training.datasets.data_splitter import DataSplitter
import pandas as pd

# 메타데이터 로드
metadata = pd.read_csv("data/metadata.csv")

# 데이터 분할 (80/10/10)
splitter = DataSplitter(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    stratify_column="tier"
)

train_df, val_df, test_df = splitter.split(metadata)

# 분할 결과 저장
train_df.to_csv("data/train_metadata.csv", index=False)
val_df.to_csv("data/val_metadata.csv", index=False)
test_df.to_csv("data/test_metadata.csv", index=False)
```

### Pairwise 데이터셋 생성

```python
from training.datasets.pairwise_dataset import PairwiseDataset
from torchvision import transforms

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 데이터셋 생성
train_dataset = PairwiseDataset(
    metadata=train_df,
    image_dir="data/images",
    transform=transform
)

print(f"학습 페어 수: {len(train_dataset)}")
```

---

## 학습 프로세스

### 기본 학습 실행

```bash
python -m training.scripts.train \
    --data-dir data/images \
    --metadata data/train_metadata.csv \
    --val-metadata data/val_metadata.csv \
    --output-dir ./checkpoints \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 1e-4
```

### 학습 설정 옵션

| 옵션 | 기본값 | 설명 |
|-----|-------|------|
| `--epochs` | 100 | 최대 학습 에폭 수 |
| `--batch-size` | 32 | 배치 크기 |
| `--learning-rate` | 1e-4 | 초기 학습률 |
| `--weight-decay` | 0.01 | L2 정규화 |
| `--gradient-clip-norm` | 1.0 | 그래디언트 클리핑 |
| `--early-stopping-patience` | 10 | Early stopping 인내 횟수 |
| `--seed` | 42 | 랜덤 시드 |

### Python API를 통한 학습

```python
from training.trainer import Trainer
from training.config import TrainingConfig
from app.ml.ranking_model import PairwiseRankingModel
from torch.utils.data import DataLoader

# 설정 생성
config = TrainingConfig(
    learning_rate=1e-4,
    weight_decay=0.01,
    batch_size=32,
    max_epochs=100,
    gradient_clip_norm=1.0,
    early_stopping_patience=10,
    scheduler_t_max=100,
    scheduler_eta_min=1e-6,
    checkpoint_dir="./checkpoints",
    save_every_n_epochs=5,
    wandb_enabled=True,
    wandb_project="mirip-ranking",
    device="cuda",
    seed=42,
)

# 모델 초기화
model = PairwiseRankingModel(
    backbone_name="facebook/dinov2-large",
    projector_output_dim=256,
    freeze_backbone=True
)

# 데이터로더 준비
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Trainer 초기화 및 학습
trainer = Trainer(model=model, config=config)
trainer.fit(train_loader, val_loader)
```

### wandb 로깅

wandb를 사용하여 실험을 추적합니다:

```bash
# wandb 로그인 (최초 1회)
wandb login

# wandb와 함께 학습 실행
python -m training.scripts.train \
    --data-dir data/images \
    --metadata data/train_metadata.csv \
    --wandb-project mirip-ranking \
    --wandb-run-name experiment-001
```

추적되는 메트릭:

- `train/loss`: 학습 손실
- `train/accuracy`: 학습 정확도
- `val/loss`: 검증 손실
- `val/accuracy`: 검증 정확도
- `learning_rate`: 현재 학습률

### 체크포인트 관리

학습 중 체크포인트가 자동으로 저장됩니다:

```
checkpoints/
├── best_model.pt          # 최고 검증 정확도 모델
├── checkpoint_epoch_5.pt  # 에폭 5 체크포인트
├── checkpoint_epoch_10.pt # 에폭 10 체크포인트
└── last_model.pt          # 마지막 에폭 모델
```

체크포인트 구조:

```python
{
    "epoch": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "best_accuracy": float,
    "config": TrainingConfig,
}
```

### 학습 재개

중단된 학습을 재개합니다:

```python
# 체크포인트에서 학습 재개
trainer = Trainer(model=model, config=config)
trainer.load_checkpoint("checkpoints/last_model.pt")
trainer.fit(train_loader, val_loader)
```

---

## 평가 및 추론

### 모델 평가

```bash
python -m training.scripts.evaluate \
    --checkpoint ./checkpoints/best_model.pt \
    --test-data data/test_metadata.csv \
    --image-dir data/images
```

### Python API를 통한 평가

```python
from training.evaluator import Evaluator
from torch.utils.data import DataLoader

# 모델 로드
model = PairwiseRankingModel()
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Evaluator 초기화
evaluator = Evaluator(model=model, device="cuda")

# 테스트 데이터로더 준비
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 평가 실행
accuracy = evaluator.evaluate(test_loader)
print(f"테스트 정확도: {accuracy:.2%}")

# 상세 결과 얻기
detailed_results = evaluator.evaluate_detailed(test_loader)
print(f"전체 페어 수: {detailed_results['total_pairs']}")
print(f"정확한 예측: {detailed_results['correct_predictions']}")
```

### 단일 이미지 페어 추론

```python
import torch
from PIL import Image
from torchvision import transforms

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
    score1, score2 = model(img1.cuda(), img2.cuda())

if score1 > score2:
    print("Image 1이 더 높은 품질입니다")
    print(f"Score 차이: {(score1 - score2).item():.4f}")
else:
    print("Image 2가 더 높은 품질입니다")
    print(f"Score 차이: {(score2 - score1).item():.4f}")
```

### 배치 추론

```python
from typing import List, Tuple

def batch_compare(
    model: PairwiseRankingModel,
    image_pairs: List[Tuple[str, str]],
    transform: transforms.Compose,
    device: str = "cuda"
) -> List[dict]:
    """
    여러 이미지 페어를 배치로 비교합니다.

    Args:
        model: 학습된 PairwiseRankingModel
        image_pairs: (이미지1 경로, 이미지2 경로) 튜플 리스트
        transform: 이미지 전처리 transform
        device: 추론 디바이스

    Returns:
        비교 결과 딕셔너리 리스트
    """
    model.eval()
    results = []

    with torch.no_grad():
        for img1_path, img2_path in image_pairs:
            img1 = transform(Image.open(img1_path)).unsqueeze(0).to(device)
            img2 = transform(Image.open(img2_path)).unsqueeze(0).to(device)

            score1, score2 = model(img1, img2)

            results.append({
                "image1": img1_path,
                "image2": img2_path,
                "score1": score1.item(),
                "score2": score2.item(),
                "winner": "image1" if score1 > score2 else "image2",
                "confidence": abs(score1 - score2).item()
            })

    return results
```

### 성능 벤치마크

```python
from training.benchmarks import PerformanceBenchmarks, set_seed

# 재현성을 위한 시드 설정
set_seed(42)

# 벤치마크 실행
benchmarks = PerformanceBenchmarks(model=model, device="cuda")
report = benchmarks.run_full_benchmark()

print(f"추론 시간: {report['inference_time_ms']:.2f}ms/pair")
print(f"메모리 사용량: {report['memory_usage_gb']:.2f}GB")
print(f"추론 시간 요구사항 충족: {report['meets_inference_requirement']}")
print(f"메모리 요구사항 충족: {report['meets_memory_requirement']}")
```

---

## 문제 해결

### 일반적인 문제

#### 1. CUDA 메모리 부족

**증상**:
```
RuntimeError: CUDA out of memory
```

**해결 방법**:

```bash
# 배치 크기 줄이기
python -m training.scripts.train --batch-size 16

# 또는 gradient accumulation 사용
python -m training.scripts.train --batch-size 8 --gradient-accumulation-steps 4
```

#### 2. DINOv2 모델 다운로드 실패

**증상**:
```
OSError: Can't load the model for 'facebook/dinov2-large'
```

**해결 방법**:

```bash
# Hugging Face 캐시 정리
rm -rf ~/.cache/huggingface/hub/models--facebook--dinov2-large

# 또는 수동으로 모델 다운로드
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-large')"
```

#### 3. 학습이 수렴하지 않음

**증상**:
- 손실이 감소하지 않거나 발산함
- 정확도가 50% 근처에서 정체됨

**해결 방법**:

```python
# 학습률 조정
config = TrainingConfig(
    learning_rate=5e-5,  # 더 작은 학습률
    weight_decay=0.001,  # 더 작은 weight decay
)

# 또는 웜업 추가
config = TrainingConfig(
    warmup_steps=500,
    learning_rate=1e-4,
)
```

#### 4. 검증 정확도가 학습 정확도보다 낮음 (과적합)

**해결 방법**:

```python
# Dropout 증가
model = PairwiseRankingModel(
    projector_dropout=0.3,  # 기본값: 0.1
)

# 또는 데이터 증강 추가
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### 5. wandb 연결 오류

**증상**:
```
wandb: Network error (ConnectionError)
```

**해결 방법**:

```bash
# 오프라인 모드로 실행
export WANDB_MODE=offline
python -m training.scripts.train ...

# 나중에 동기화
wandb sync ./wandb/latest-run
```

### 성능 최적화 팁

1. **Mixed Precision 학습**: GPU 메모리 절약 및 학습 속도 향상

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        score1, score2 = model(img1, img2)
        loss = model.compute_loss(score1, score2, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

2. **DataLoader 최적화**:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # CPU 코어 수에 맞게 조정
    pin_memory=True,    # GPU 전송 속도 향상
    prefetch_factor=2,  # 미리 가져올 배치 수
)
```

3. **이미지 캐싱**:

```python
# 메모리에 이미지 캐싱 (RAM이 충분한 경우)
class CachedPairwiseDataset(PairwiseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}

    def _load_image(self, path):
        if path not in self.cache:
            self.cache[path] = super()._load_image(path)
        return self.cache[path]
```

---

## 부록

### 권장 하이퍼파라미터

| 데이터셋 크기 | batch_size | learning_rate | epochs |
|-------------|-----------|---------------|--------|
| < 1K | 16 | 5e-5 | 200 |
| 1K - 10K | 32 | 1e-4 | 100 |
| 10K - 100K | 64 | 2e-4 | 50 |
| > 100K | 128 | 3e-4 | 30 |

### 성능 요구사항 체크리스트

- [ ] 추론 시간 < 100ms/pair (GPU)
- [ ] GPU 메모리 < 12GB (batch_size=32)
- [ ] 총 학습 시간 < 6시간 (10K 샘플)
- [ ] 테스트 정확도 >= 60%

### 참고 자료

- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
