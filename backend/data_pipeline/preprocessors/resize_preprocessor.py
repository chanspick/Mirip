# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: ResizePreprocessor 구현

이미지 크기를 조절하는 전처리기입니다.
다양한 보간법과 종횡비 유지 옵션을 지원합니다.
"""

from typing import Tuple, Union

import numpy as np
from PIL import Image

from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor


# PIL 보간법 매핑
INTERPOLATION_MODES = {
    "NEAREST": Image.Resampling.NEAREST,
    "BILINEAR": Image.Resampling.BILINEAR,
    "BICUBIC": Image.Resampling.BICUBIC,
    "LANCZOS": Image.Resampling.LANCZOS,
}


class ResizePreprocessor(BasePreprocessor):
    """
    이미지 리사이즈 전처리기

    이미지를 지정된 크기로 조절합니다.
    종횡비 유지 옵션과 다양한 보간법을 지원합니다.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        maintain_aspect_ratio: bool = False,
        interpolation: str = "LANCZOS",
    ):
        """
        ResizePreprocessor 초기화

        Args:
            target_size: 목표 크기 (width, height)
            maintain_aspect_ratio: 종횡비 유지 여부 (기본값: False)
            interpolation: 보간법 (NEAREST, BILINEAR, BICUBIC, LANCZOS)
        """
        self.target_size = target_size
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.interpolation = INTERPOLATION_MODES.get(
            interpolation.upper(), Image.Resampling.LANCZOS
        )

    def process(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> Image.Image:
        """
        이미지를 리사이즈합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            리사이즈된 PIL Image
        """
        # 입력을 PIL Image로 변환
        pil_image = self._load_image(image)

        if self.maintain_aspect_ratio:
            return self._resize_with_aspect_ratio(pil_image)
        else:
            return pil_image.resize(self.target_size, self.interpolation)

    def _resize_with_aspect_ratio(self, image: Image.Image) -> Image.Image:
        """
        종횡비를 유지하면서 리사이즈합니다.

        Args:
            image: PIL Image

        Returns:
            종횡비가 유지된 리사이즈 이미지
        """
        original_width, original_height = image.size
        target_width, target_height = self.target_size

        # 비율 계산
        ratio = min(target_width / original_width, target_height / original_height)

        # 새 크기 계산
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        return image.resize((new_width, new_height), self.interpolation)
