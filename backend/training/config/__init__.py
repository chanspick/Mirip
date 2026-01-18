# Training Config Module
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: GREEN - Implementation
"""
Training Configuration Module

학습에 필요한 모든 하이퍼파라미터와 설정을 관리합니다.

Acceptance Criteria (AC-006):
- AdamW optimizer (lr=1e-4, weight_decay=0.01)
- CosineAnnealingLR scheduler
- Early stopping (patience=10)

Example:
    >>> config = TrainingConfig(learning_rate=1e-4, batch_size=32)
    >>> print(config.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """
    학습 설정 데이터 클래스

    모든 학습 관련 하이퍼파라미터를 중앙에서 관리합니다.

    Attributes:
        learning_rate: 학습률 (기본값: 1e-4)
        weight_decay: L2 정규화 가중치 (기본값: 0.01)
        batch_size: 배치 크기 (기본값: 32)
        max_epochs: 최대 학습 에폭 수 (기본값: 100)
        early_stopping_patience: 조기 종료 인내 횟수 (기본값: 10)
        checkpoint_dir: 체크포인트 저장 디렉토리
        wandb_project: wandb 프로젝트 이름
        wandb_run_name: wandb 실행 이름
        device: 학습 디바이스 (cuda/cpu)
        seed: 랜덤 시드

    Example:
        >>> config = TrainingConfig()
        >>> print(config.learning_rate)  # 0.0001
    """

    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Training settings
    batch_size: int = 32
    max_epochs: int = 100
    gradient_clip_norm: Optional[float] = 1.0

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Scheduler settings
    scheduler_t_max: Optional[int] = None  # None이면 max_epochs 사용
    scheduler_eta_min: float = 1e-6

    # Checkpoint settings
    checkpoint_dir: str = field(default_factory=lambda: "./checkpoints")
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3

    # wandb settings
    wandb_project: Optional[str] = "mirip-ranking"
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_enabled: bool = True

    # Device settings
    device: str = "cuda"
    mixed_precision: bool = False

    # Reproducibility
    seed: int = 42

    # Data settings
    num_workers: int = 4
    pin_memory: bool = True

    def __post_init__(self) -> None:
        """초기화 후 검증 및 추가 설정"""
        # checkpoint_dir을 Path로 변환하고 생성
        self.checkpoint_dir = str(Path(self.checkpoint_dir).resolve())
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # scheduler_t_max 기본값 설정
        if self.scheduler_t_max is None:
            self.scheduler_t_max = self.max_epochs

        # 검증
        self._validate()

    def _validate(self) -> None:
        """설정값 검증"""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")

    def to_dict(self) -> dict:
        """
        설정을 딕셔너리로 변환

        Returns:
            모든 설정이 포함된 딕셔너리
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """
        딕셔너리에서 설정 생성

        Args:
            config_dict: 설정 딕셔너리

        Returns:
            TrainingConfig 인스턴스
        """
        return cls(**config_dict)

    def __repr__(self) -> str:
        """객체의 문자열 표현"""
        return (
            f"{self.__class__.__name__}("
            f"lr={self.learning_rate}, "
            f"batch_size={self.batch_size}, "
            f"max_epochs={self.max_epochs}, "
            f"patience={self.early_stopping_patience})"
        )


__all__ = ["TrainingConfig"]
