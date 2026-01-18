# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: BasePreprocessor 추상 클래스

이미지 전처리기의 기본 인터페이스를 정의합니다.
모든 전처리기는 이 추상 클래스를 상속받아 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np
from PIL import Image


class BasePreprocessor(ABC):
    """
    이미지 전처리기 추상 베이스 클래스

    다양한 이미지 전처리 기능의 인터페이스를 정의합니다.
    ResizePreprocessor, NormalizePreprocessor, AugmentPreprocessor 등이 이를 구현합니다.
    """

    @abstractmethod
    def process(self, image: Union[Image.Image, np.ndarray, str]) -> Any:
        """
        단일 이미지를 전처리합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            전처리된 이미지 (구현체에 따라 PIL Image 또는 numpy 배열)
        """
        pass

    def process_batch(
        self, images: List[Union[Image.Image, np.ndarray, str]]
    ) -> List[Any]:
        """
        여러 이미지를 배치로 전처리합니다.

        Args:
            images: 이미지 리스트 (PIL Image, numpy 배열, 또는 파일 경로)

        Returns:
            전처리된 이미지 리스트
        """
        return [self.process(img) for img in images]

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
