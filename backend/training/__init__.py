# Training Module
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
"""
모델 학습 관련 모듈

- config: 학습 설정
- trainer: 모델 학습 트레이너
- datasets: 데이터셋 로더
- evaluator: 모델 평가 (Phase 4)
- benchmarks: 성능 벤치마크 및 재현 가능성 (Phase 4)
"""

from training.config import TrainingConfig
from training.trainer import Trainer
from training.evaluator import Evaluator
from training.benchmarks import PerformanceBenchmarks, set_seed

# Convenience re-exports
from training.datasets.pairwise_dataset import PairwiseDataset
from training.datasets.data_splitter import DataSplitter

__all__ = [
    "TrainingConfig",
    "Trainer",
    "Evaluator",
    "PerformanceBenchmarks",
    "set_seed",
    "PairwiseDataset",
    "DataSplitter",
]
