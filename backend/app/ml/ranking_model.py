# ranking_model.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: GREEN - Implementation
"""
Pairwise Ranking Model Module

Combines frozen DINOv2 feature extractor with trainable projector
to produce scalar quality scores for artwork images.

Acceptance Criteria (AC-003):
- 두 이미지 입력 시 각각 스칼라 스코어 반환
- S티어 이미지 스코어 > C티어 이미지 스코어
- 배치 처리 지원 (32개 쌍)

Architecture:
- FeatureExtractor: DINOv2-large (frozen, 1024-d output)
- Projector: MLP (1024 -> 512 -> 256, trainable)
- ScoreHead: Linear (256 -> 1, trainable)
- Loss: MarginRankingLoss (margin=1.0)

Example:
    >>> model = PairwiseRankingModel()
    >>> img1 = torch.randn(1, 3, 768, 768)
    >>> img2 = torch.randn(1, 3, 768, 768)
    >>> score1, score2 = model(img1, img2)
    >>> print(score1.shape)  # torch.Size([1, 1])
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.ml.feature_extractor import DINOv2FeatureExtractor
from app.ml.projector import Projector


class PairwiseRankingModel(nn.Module):
    """
    Pairwise Ranking Model for Artwork Quality Assessment

    두 이미지를 입력받아 각각의 품질 스코어를 반환합니다.
    높은 스코어는 더 높은 품질(더 높은 티어)을 의미합니다.

    Attributes:
        DEFAULT_PROJECTOR_HIDDEN_DIM: 기본 projector 은닉층 차원 (512)
        DEFAULT_PROJECTOR_OUTPUT_DIM: 기본 projector 출력 차원 (256)
        DEFAULT_MARGIN: MarginRankingLoss 마진 (1.0)

    Args:
        feature_extractor_model: DINOv2 모델 이름 (기본값: facebook/dinov2-large)
        projector_hidden_dim: Projector 은닉층 차원 (기본값: 512)
        projector_output_dim: Projector 출력 차원 (기본값: 256)
        dropout: Projector 드롭아웃 비율 (기본값: 0.1)
        margin: MarginRankingLoss 마진 (기본값: 1.0)

    Example:
        >>> model = PairwiseRankingModel()
        >>> img1, img2 = torch.randn(32, 3, 768, 768), torch.randn(32, 3, 768, 768)
        >>> score1, score2 = model(img1, img2)
        >>> assert score1.shape == (32, 1)
    """

    # Configuration as class attributes
    DEFAULT_PROJECTOR_HIDDEN_DIM: int = 512
    DEFAULT_PROJECTOR_OUTPUT_DIM: int = 256
    DEFAULT_MARGIN: float = 1.0

    def __init__(
        self,
        feature_extractor_model: str = "facebook/dinov2-large",
        projector_hidden_dim: int = DEFAULT_PROJECTOR_HIDDEN_DIM,
        projector_output_dim: int = DEFAULT_PROJECTOR_OUTPUT_DIM,
        dropout: float = 0.1,
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        """
        PairwiseRankingModel 초기화

        Args:
            feature_extractor_model: DINOv2 모델 이름
            projector_hidden_dim: Projector 은닉층 차원
            projector_output_dim: Projector 출력 차원
            dropout: Projector 드롭아웃 비율
            margin: MarginRankingLoss 마진
        """
        super().__init__()

        # 설정 저장
        self._margin = margin

        # Feature Extractor (DINOv2 - frozen)
        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=feature_extractor_model,
            normalize=True,
        )
        feature_dim = self.feature_extractor.output_dim

        # Projector (trainable)
        self.projector = Projector(
            input_dim=feature_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
            dropout=dropout,
            normalize=True,
        )

        # Score Head (trainable)
        # Projector 출력을 스칼라 스코어로 변환
        self.score_head = nn.Sequential(
            nn.Linear(projector_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 가중치 초기화
        self._init_score_head()

        # Loss function
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def _init_score_head(self) -> None:
        """Score head 가중치 초기화"""
        for module in self.score_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def margin(self) -> float:
        """MarginRankingLoss 마진 반환"""
        return self._margin

    def _extract_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        단일 이미지에서 스코어 추출

        Args:
            x: 입력 이미지 텐서 (B, 3, H, W)

        Returns:
            스코어 텐서 (B, 1)
        """
        # Feature extraction (frozen)
        features = self.feature_extractor(x)  # (B, feature_dim)

        # Projection (trainable)
        projected = self.projector(features)  # (B, projector_output_dim)

        # Score prediction (trainable)
        score = self.score_head(projected)  # (B, 1)

        return score

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        두 이미지의 스코어 계산

        Args:
            img1: 첫 번째 이미지 텐서 (B, 3, H, W)
            img2: 두 번째 이미지 텐서 (B, 3, H, W)

        Returns:
            (score1, score2) 튜플
            - score1: 첫 번째 이미지 스코어 (B, 1)
            - score2: 두 번째 이미지 스코어 (B, 1)
        """
        score1 = self._extract_score(img1)
        score2 = self._extract_score(img2)

        return score1, score2

    def compute_loss(
        self,
        score1: torch.Tensor,
        score2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        MarginRankingLoss 계산

        Args:
            score1: 첫 번째 이미지 스코어 (B, 1)
            score2: 두 번째 이미지 스코어 (B, 1)
            labels: 레이블 텐서 (B,)
                   1: score1 > score2 (img1이 더 높은 티어)
                   -1: score1 < score2 (img2가 더 높은 티어)

        Returns:
            Loss 스칼라 텐서

        Note:
            MarginRankingLoss: max(0, -label * (score1 - score2) + margin)
            - label=1, score1 > score2 + margin -> loss = 0
            - label=1, score1 < score2 -> loss > 0
        """
        # score1, score2를 1D로 변환 (B,)
        score1_flat = score1.squeeze(-1)
        score2_flat = score2.squeeze(-1)

        # labels를 float로 변환
        labels_float = labels.float()

        loss = self.loss_fn(score1_flat, score2_flat, labels_float)

        return loss

    def predict_score(self, image: torch.Tensor) -> torch.Tensor:
        """
        단일 이미지의 스코어 예측 (추론용)

        Args:
            image: 입력 이미지 텐서 (B, 3, H, W)

        Returns:
            스코어 텐서 (B, 1)
        """
        self.eval()
        with torch.no_grad():
            return self._extract_score(image)

    def predict(self, img1: torch.Tensor, img2: torch.Tensor) -> int:
        """
        두 이미지 비교 (추론용)

        Args:
            img1: 첫 번째 이미지 텐서 (1, 3, H, W)
            img2: 두 번째 이미지 텐서 (1, 3, H, W)

        Returns:
            1: img1 > img2 (img1이 더 높은 품질)
            -1: img1 < img2 (img2가 더 높은 품질)
            0: img1 == img2 (동일한 품질)
        """
        self.eval()
        with torch.no_grad():
            score1, score2 = self(img1, img2)

            diff = (score1 - score2).item()

            if diff > 0:
                return 1
            elif diff < 0:
                return -1
            else:
                return 0

    def to(self, device: Union[str, torch.device]) -> "PairwiseRankingModel":
        """
        모델을 지정된 디바이스로 이동

        Args:
            device: 대상 디바이스

        Returns:
            self
        """
        self.feature_extractor = self.feature_extractor.to(device)
        self.projector = self.projector.to(device)
        self.score_head = self.score_head.to(device)
        return super().to(device)

    def __repr__(self) -> str:
        """객체의 문자열 표현"""
        return (
            f"{self.__class__.__name__}("
            f"feature_dim={self.feature_extractor.output_dim}, "
            f"projector_dim={self.projector.output_dim}, "
            f"margin={self._margin})"
        )
