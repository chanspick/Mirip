# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: BaseTagger 추상 클래스

분류기의 기본 인터페이스를 정의합니다.
모든 태거는 이 추상 클래스를 상속받아 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


class TagResult(BaseModel):
    """
    태깅 결과 모델

    분류 결과와 신뢰도를 포함합니다.
    """

    tag: str = Field(..., description="분류된 태그")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="추가 메타데이터"
    )

    model_config = {
        "str_strip_whitespace": True,
    }


class BaseTagger(ABC):
    """
    태거 추상 베이스 클래스

    이미지를 분석하여 분류 태그를 예측하는 기능의 인터페이스를 정의합니다.
    DepartmentTagger, TierTagger 등이 이를 구현합니다.
    """

    @abstractmethod
    def predict(self, image: Union[Image.Image, np.ndarray, str]) -> TagResult:
        """
        단일 이미지를 분류합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            TagResult: 분류 결과
        """
        pass

    def predict_batch(
        self, images: List[Union[Image.Image, np.ndarray, str]]
    ) -> List[TagResult]:
        """
        여러 이미지를 배치로 분류합니다.

        Args:
            images: 이미지 리스트

        Returns:
            TagResult 리스트
        """
        return [self.predict(img) for img in images]

    def _load_image(self, image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
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
            return Image.fromarray(image)
        elif isinstance(image, str):
            return Image.open(image)
        else:
            raise TypeError(f"지원하지 않는 이미지 타입: {type(image)}")

    def _to_numpy(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
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
