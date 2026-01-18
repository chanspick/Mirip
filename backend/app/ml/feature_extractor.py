# feature_extractor.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: REFACTOR - Improved Code Quality
"""
DINOv2 Feature Extractor Module

facebook/dinov2-large 모델을 사용하여 이미지에서
1024차원 특징 벡터를 추출합니다.

Acceptance Criteria (AC-001):
- Single image: 1024-d feature vector, float32, < 100ms inference
- Batch (32 images): (32, 1024) tensor, < 12GB VRAM
- All parameters requires_grad = False

Example:
    >>> extractor = DINOv2FeatureExtractor()
    >>> image = torch.randn(1, 3, 768, 768)
    >>> features = extractor(image)
    >>> print(features.shape)  # torch.Size([1, 1024])
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DINOv2FeatureExtractor(nn.Module):
    """
    DINOv2 기반 Feature Extractor

    facebook/dinov2-large 모델을 백본으로 사용하여
    이미지에서 1024차원 특징 벡터를 추출합니다.
    모든 백본 파라미터는 동결(frozen)됩니다.

    Attributes:
        DEFAULT_MODEL_NAME: 기본 DINOv2 모델 이름
        OUTPUT_DIM: 출력 특징 벡터 차원
        SUPPORTED_MODELS: 지원하는 DINOv2 모델 목록

    Args:
        model_name: Hugging Face 모델 이름
                   기본값: "facebook/dinov2-large" (1024-d output)
        normalize: L2 정규화 적용 여부 (기본값: True)

    Example:
        >>> extractor = DINOv2FeatureExtractor()
        >>> image = torch.randn(1, 3, 768, 768)
        >>> features = extractor(image)
        >>> assert features.shape == (1, 1024)
    """

    # Configuration as class attributes
    DEFAULT_MODEL_NAME: str = "facebook/dinov2-large"
    OUTPUT_DIM: int = 1024
    SUPPORTED_MODELS: dict[str, int] = {
        "facebook/dinov2-small": 384,
        "facebook/dinov2-base": 768,
        "facebook/dinov2-large": 1024,
        "facebook/dinov2-giant": 1536,
    }

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        normalize: bool = True,
    ) -> None:
        """
        DINOv2FeatureExtractor 초기화

        Args:
            model_name: Hugging Face 모델 이름
            normalize: L2 정규화 적용 여부

        Raises:
            ValueError: 지원하지 않는 모델 이름인 경우
        """
        super().__init__()

        # 설정 저장
        self._model_name = model_name
        self._normalize = normalize

        # 출력 차원 설정
        if model_name in self.SUPPORTED_MODELS:
            self._output_dim = self.SUPPORTED_MODELS[model_name]
        else:
            # 커스텀 모델의 경우 기본 차원 사용
            self._output_dim = self.OUTPUT_DIM

        # DINOv2 모델 로드
        self.model = AutoModel.from_pretrained(model_name)

        # 모든 파라미터 동결
        self._freeze_backbone()

        # 평가 모드로 설정
        self.model.eval()

    def _freeze_backbone(self) -> None:
        """백본 모델의 모든 파라미터를 동결합니다."""
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def output_dim(self) -> int:
        """출력 특징 벡터의 차원을 반환합니다."""
        return self._output_dim

    @property
    def model_name(self) -> str:
        """사용 중인 모델 이름을 반환합니다."""
        return self._model_name

    @property
    def is_frozen(self) -> bool:
        """모든 파라미터가 동결되었는지 확인합니다."""
        return all(not p.requires_grad for p in self.model.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        이미지에서 특징 벡터 추출

        Args:
            x: 입력 이미지 텐서
               Shape: (B, 3, H, W)
               값 범위: [0, 1] 또는 ImageNet 정규화 적용
               지원 크기: 224, 384, 518, 768 등

        Returns:
            특징 벡터 텐서
            Shape: (B, output_dim)
            dtype: torch.float32
            L2 정규화 적용 시 norm = 1.0

        Note:
            - 추론 시 자동으로 torch.no_grad() 컨텍스트 사용
            - 모델은 항상 eval 모드로 유지됨
        """
        # 평가 모드 보장
        self.model.eval()

        with torch.no_grad():
            # DINOv2 forward pass
            outputs = self.model(x)

            # CLS 토큰 특징 추출
            # last_hidden_state shape: (B, num_patches + 1, hidden_dim)
            # CLS token is at position 0
            features = outputs.last_hidden_state[:, 0, :]

            # L2 정규화 (선택적)
            if self._normalize:
                features = F.normalize(features, p=2, dim=1)

        return features

    def extract_features(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        특징 추출을 위한 고급 인터페이스

        Args:
            x: 입력 이미지 텐서 (B, 3, H, W)
            return_all_tokens: True이면 모든 패치 토큰도 반환

        Returns:
            return_all_tokens=False: CLS 특징 (B, output_dim)
            return_all_tokens=True: (CLS 특징, 패치 특징) 튜플
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(x)
            hidden_states = outputs.last_hidden_state

            # CLS 토큰 (position 0)
            cls_features = hidden_states[:, 0, :]
            if self._normalize:
                cls_features = F.normalize(cls_features, p=2, dim=1)

            if return_all_tokens:
                # 패치 토큰 (position 1 이후)
                patch_features = hidden_states[:, 1:, :]
                return cls_features, patch_features

            return cls_features

    def to(self, device: Union[str, torch.device]) -> "DINOv2FeatureExtractor":
        """
        모델을 지정된 디바이스로 이동

        Args:
            device: 대상 디바이스 ("cpu", "cuda", "cuda:0" 등)

        Returns:
            self (메서드 체이닝 지원)
        """
        self.model = self.model.to(device)
        return super().to(device)

    def __repr__(self) -> str:
        """객체의 문자열 표현을 반환합니다."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self._model_name}', "
            f"output_dim={self._output_dim}, "
            f"normalize={self._normalize}, "
            f"frozen={self.is_frozen})"
        )
