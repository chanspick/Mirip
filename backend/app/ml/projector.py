# projector.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: REFACTOR - Improved Code Quality
"""
Projector Module

DINOv2 특징 벡터를 저차원 임베딩 공간으로 투영합니다.

Acceptance Criteria (AC-002):
- Input 1024-d -> Output 256-d (normalized)
- Gradient flow verified (no NaN/Inf)
- Dropout active in training, inactive in eval

Architecture:
- Linear(input_dim, hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim, output_dim) -> LayerNorm

Example:
    >>> projector = Projector()
    >>> features = torch.randn(1, 1024)
    >>> output = projector(features)
    >>> print(output.shape)  # torch.Size([1, 256])
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """
    Feature Projector

    고차원 특징 벡터를 저차원 임베딩 공간으로 투영합니다.
    학습 가능한 2-layer MLP 구조로 구성됩니다.

    Attributes:
        DEFAULT_INPUT_DIM: 기본 입력 차원 (DINOv2-large: 1024)
        DEFAULT_HIDDEN_DIM: 기본 은닉층 차원 (512)
        DEFAULT_OUTPUT_DIM: 기본 출력 차원 (256)
        DEFAULT_DROPOUT: 기본 드롭아웃 비율 (0.1)

    Args:
        input_dim: 입력 특징 벡터 차원 (기본값: 1024)
        hidden_dim: 은닉층 차원 (기본값: 512)
        output_dim: 출력 임베딩 차원 (기본값: 256)
        dropout: 드롭아웃 비율 (기본값: 0.1)
        normalize: 출력 L2 정규화 여부 (기본값: True)

    Example:
        >>> projector = Projector()
        >>> features = torch.randn(32, 1024)
        >>> output = projector(features)
        >>> assert output.shape == (32, 256)
        >>> assert torch.allclose(output.norm(dim=1), torch.ones(32), atol=1e-5)
    """

    # Configuration as class attributes
    DEFAULT_INPUT_DIM: int = 1024
    DEFAULT_HIDDEN_DIM: int = 512
    DEFAULT_OUTPUT_DIM: int = 256
    DEFAULT_DROPOUT: float = 0.1

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = DEFAULT_OUTPUT_DIM,
        dropout: float = DEFAULT_DROPOUT,
        normalize: bool = True,
    ) -> None:
        """
        Projector 초기화

        Args:
            input_dim: 입력 특징 벡터 차원
            hidden_dim: 은닉층 차원
            output_dim: 출력 임베딩 차원
            dropout: 드롭아웃 비율 (0.0 ~ 1.0)
            normalize: 출력 L2 정규화 여부

        Raises:
            ValueError: 차원이 0 이하인 경우
            ValueError: 드롭아웃 비율이 범위를 벗어난 경우
        """
        super().__init__()

        # 설정 검증
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("All dimensions must be positive integers")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("Dropout must be between 0.0 and 1.0")

        # 설정 저장
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._dropout_rate = dropout
        self._normalize = normalize

        # Architecture: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> LayerNorm
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self) -> None:
        """가중치 초기화 (Xavier uniform)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def input_dim(self) -> int:
        """입력 차원을 반환합니다."""
        return self._input_dim

    @property
    def hidden_dim(self) -> int:
        """은닉층 차원을 반환합니다."""
        return self._hidden_dim

    @property
    def output_dim(self) -> int:
        """출력 차원을 반환합니다."""
        return self._output_dim

    @property
    def dropout_rate(self) -> float:
        """드롭아웃 비율을 반환합니다."""
        return self._dropout_rate

    @property
    def num_parameters(self) -> int:
        """학습 가능한 파라미터 수를 반환합니다."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        특징 벡터 투영

        Args:
            x: 입력 특징 벡터
               Shape: (B, input_dim)
               dtype: torch.float32

        Returns:
            투영된 임베딩 벡터
            Shape: (B, output_dim)
            dtype: torch.float32
            L2 정규화 적용 시 norm = 1.0

        Note:
            - train 모드: 드롭아웃 활성화 (확률적 출력)
            - eval 모드: 드롭아웃 비활성화 (결정적 출력)
        """
        x = self.layers(x)

        if self._normalize:
            x = F.normalize(x, p=2, dim=1)

        return x

    def project(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        특징 투영을 위한 고급 인터페이스

        Args:
            x: 입력 특징 벡터 (B, input_dim)
            return_intermediate: True이면 중간 은닉 표현도 반환

        Returns:
            return_intermediate=False: 투영된 임베딩 (B, output_dim)
            return_intermediate=True: (투영된 임베딩, 중간 표현) 튜플
        """
        if return_intermediate:
            # 첫 번째 선형층 + LayerNorm + GELU까지만 실행
            intermediate = self.layers[0](x)  # Linear
            intermediate = self.layers[1](intermediate)  # LayerNorm
            intermediate = self.layers[2](intermediate)  # GELU

            # 나머지 레이어 실행
            output = self.layers[3](intermediate)  # Dropout
            output = self.layers[4](output)  # Linear
            output = self.layers[5](output)  # LayerNorm

            if self._normalize:
                output = F.normalize(output, p=2, dim=1)

            return output, intermediate

        return self.forward(x)

    def freeze(self) -> None:
        """모든 파라미터를 동결합니다."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """모든 파라미터를 동결 해제합니다."""
        for param in self.parameters():
            param.requires_grad = True

    def __repr__(self) -> str:
        """객체의 문자열 표현을 반환합니다."""
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self._input_dim}, "
            f"hidden_dim={self._hidden_dim}, "
            f"output_dim={self._output_dim}, "
            f"dropout={self._dropout_rate}, "
            f"normalize={self._normalize}, "
            f"params={self.num_parameters:,})"
        )
