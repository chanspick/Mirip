# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: NormalizePreprocessor 구현

이미지 픽셀 값을 정규화하는 전처리기입니다.
다양한 정규화 방식을 지원합니다 (0-1, -1~1, ImageNet 스타일 등).
"""

from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor


class NormalizePreprocessor(BasePreprocessor):
    """
    이미지 정규화 전처리기

    픽셀 값을 정규화합니다.
    기본값은 0-1 범위로 정규화하며, 평균/표준편차 정규화도 지원합니다.
    """

    def __init__(
        self,
        output_range: Tuple[float, float] = (0.0, 1.0),
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
    ):
        """
        NormalizePreprocessor 초기화

        Args:
            output_range: 출력 값 범위 (기본값: (0.0, 1.0))
            mean: 채널별 평균 (ImageNet 스타일 정규화용)
            std: 채널별 표준편차 (ImageNet 스타일 정규화용)
        """
        self.output_range = output_range
        self.mean = mean
        self.std = std

    def process(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> np.ndarray:
        """
        이미지를 정규화합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            정규화된 numpy 배열 (float32)
        """
        # 입력을 numpy 배열로 변환
        array = self._to_numpy(image)

        # float32로 변환
        array = array.astype(np.float32)

        if self.mean is not None and self.std is not None:
            # ImageNet 스타일 정규화: (x - mean) / std
            return self._normalize_with_mean_std(array)
        else:
            # 범위 정규화
            return self._normalize_to_range(array)

    def _normalize_to_range(self, array: np.ndarray) -> np.ndarray:
        """
        지정된 범위로 정규화합니다.

        Args:
            array: 입력 배열 (0-255 범위)

        Returns:
            정규화된 배열
        """
        min_val, max_val = self.output_range

        # 0-255를 0-1로 변환
        normalized = array / 255.0

        # 지정된 범위로 스케일링
        if min_val != 0.0 or max_val != 1.0:
            normalized = normalized * (max_val - min_val) + min_val

        return normalized

    def _normalize_with_mean_std(self, array: np.ndarray) -> np.ndarray:
        """
        평균/표준편차를 사용하여 정규화합니다.

        Args:
            array: 입력 배열 (0-255 범위)

        Returns:
            정규화된 배열
        """
        # 먼저 0-1로 정규화
        array = array / 255.0

        # 채널별 평균/표준편차 적용
        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)

        return (array - mean) / std
