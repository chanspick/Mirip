# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: AugmentPreprocessor 구현

이미지 데이터 증강 전처리기입니다.
뒤집기, 회전, 밝기/대비 조절 등 다양한 증강 기법을 지원합니다.
"""

import random
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance

from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor


class AugmentPreprocessor(BasePreprocessor):
    """
    이미지 데이터 증강 전처리기

    다양한 이미지 변환을 통해 데이터를 증강합니다.
    뒤집기, 회전, 밝기, 대비 조절 등을 지원합니다.
    """

    def __init__(
        self,
        horizontal_flip: bool = False,
        horizontal_flip_prob: float = 0.5,
        vertical_flip: bool = False,
        vertical_flip_prob: float = 0.5,
        rotation_range: Optional[Tuple[float, float]] = None,
        brightness_range: Optional[Tuple[float, float]] = None,
        contrast_range: Optional[Tuple[float, float]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        AugmentPreprocessor 초기화

        Args:
            horizontal_flip: 수평 뒤집기 활성화 (기본값: False)
            horizontal_flip_prob: 수평 뒤집기 확률 (기본값: 0.5)
            vertical_flip: 수직 뒤집기 활성화 (기본값: False)
            vertical_flip_prob: 수직 뒤집기 확률 (기본값: 0.5)
            rotation_range: 회전 범위 (min_degree, max_degree), None이면 비활성화
            brightness_range: 밝기 범위 (min_factor, max_factor), None이면 비활성화
            contrast_range: 대비 범위 (min_factor, max_factor), None이면 비활성화
            random_seed: 랜덤 시드 (재현 가능한 결과를 위해)
        """
        self.horizontal_flip = horizontal_flip
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip = vertical_flip
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

        # 랜덤 시드 설정
        if random_seed is not None:
            random.seed(random_seed)
            self._rng = random.Random(random_seed)
        else:
            self._rng = random.Random()

    def process(
        self, image: Union[Image.Image, np.ndarray, str]
    ) -> Image.Image:
        """
        이미지에 데이터 증강을 적용합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            증강된 PIL Image
        """
        # 입력을 PIL Image로 변환
        pil_image = self._load_image(image)

        # RGB 모드로 변환 (필요한 경우)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # 수평 뒤집기
        if self.horizontal_flip and self._rng.random() < self.horizontal_flip_prob:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 수직 뒤집기
        if self.vertical_flip and self._rng.random() < self.vertical_flip_prob:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # 회전
        if self.rotation_range is not None:
            angle = self._rng.uniform(self.rotation_range[0], self.rotation_range[1])
            pil_image = pil_image.rotate(
                angle, resample=Image.Resampling.BILINEAR, expand=False
            )

        # 밝기 조절
        if self.brightness_range is not None:
            factor = self._rng.uniform(
                self.brightness_range[0], self.brightness_range[1]
            )
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(factor)

        # 대비 조절
        if self.contrast_range is not None:
            factor = self._rng.uniform(self.contrast_range[0], self.contrast_range[1])
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(factor)

        return pil_image
