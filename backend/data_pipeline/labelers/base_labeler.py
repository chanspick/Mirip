# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: BaseLabeler 추상 클래스

라벨러의 기본 인터페이스를 정의합니다.
모든 라벨러는 이 추상 클래스를 상속받아 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from data_pipeline.models.metadata import Department, Tier


class LabelingResult(BaseModel):
    """
    라벨링 결과 모델

    이미지에 대한 과별 분류와 티어 분류 결과를 포함합니다.
    """

    image_id: str = Field(..., description="이미지 ID")
    department: Department = Field(..., description="분류된 과별")
    department_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="과별 분류 신뢰도 (0-1)"
    )
    tier: Tier = Field(..., description="분류된 티어")
    tier_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="티어 분류 신뢰도 (0-1)"
    )
    tier_score: float = Field(
        ..., ge=0.0, le=100.0, description="티어 점수 (0-100)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="추가 메타데이터"
    )

    model_config = {
        "str_strip_whitespace": True,
    }

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """
        고신뢰도 여부를 확인합니다.

        과별 분류와 티어 분류 모두 임계값 이상일 때 고신뢰도로 판단합니다.

        Args:
            threshold: 신뢰도 임계값 (기본값: 0.8)

        Returns:
            bool: 고신뢰도 여부
        """
        return (
            self.department_confidence >= threshold
            and self.tier_confidence >= threshold
        )

    def needs_review(self, threshold: float = 0.6) -> bool:
        """
        수동 검토가 필요한지 확인합니다.

        과별 분류 또는 티어 분류 중 하나라도 임계값 미만이면
        수동 검토가 필요한 것으로 판단합니다.

        Args:
            threshold: 신뢰도 임계값 (기본값: 0.6)

        Returns:
            bool: 수동 검토 필요 여부
        """
        return (
            self.department_confidence < threshold
            or self.tier_confidence < threshold
        )

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        라벨링 결과를 ImageMetadata 업데이트용 딕셔너리로 변환합니다.

        Returns:
            ImageMetadata 업데이트에 사용할 딕셔너리
        """
        return {
            "department": self.department,
            "tier": self.tier,
            "tier_score": self.tier_score,
            "is_manual_label": False,  # 자동 라벨링 결과
        }


class BaseLabeler(ABC):
    """
    라벨러 추상 베이스 클래스

    이미지를 분석하여 과별 및 티어 라벨을 예측하는 기능의 인터페이스를 정의합니다.
    AutoLabeler 등이 이를 구현합니다.
    """

    @abstractmethod
    def label(
        self, image_id: str, image: Union[Image.Image, np.ndarray, str]
    ) -> LabelingResult:
        """
        단일 이미지를 라벨링합니다.

        Args:
            image_id: 이미지 식별자
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            LabelingResult: 라벨링 결과
        """
        pass

    def label_batch(
        self,
        image_ids: List[str],
        images: List[Union[Image.Image, np.ndarray, str]],
    ) -> List[LabelingResult]:
        """
        여러 이미지를 배치로 라벨링합니다.

        Args:
            image_ids: 이미지 식별자 리스트
            images: 이미지 리스트

        Returns:
            LabelingResult 리스트
        """
        if len(image_ids) != len(images):
            raise ValueError(
                f"image_ids와 images의 길이가 일치하지 않습니다: "
                f"{len(image_ids)} != {len(images)}"
            )

        return [
            self.label(image_id, image)
            for image_id, image in zip(image_ids, images)
        ]

    def _load_image(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> Image.Image:
        """
        다양한 입력 형식을 PIL Image로 변환합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            PIL Image 객체
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            # 정규화된 배열(0-1)인 경우 0-255로 변환
            if image.dtype in [np.float32, np.float64]:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            return Image.fromarray(image)
        elif isinstance(image, str):
            return Image.open(image)
        else:
            raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")

    def _to_numpy(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> np.ndarray:
        """
        다양한 입력 형식을 numpy 배열로 변환합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            numpy 배열
        """
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, str):
            return np.array(Image.open(image))
        else:
            raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")
