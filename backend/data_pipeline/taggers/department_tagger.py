# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: DepartmentTagger 구현

이미지를 과별로 분류하는 태거입니다.
시디(시각디자인), 산디(산업디자인), 공예, 회화 4개 과를 분류합니다.
"""

import random
from typing import List, Union

import numpy as np
from PIL import Image

from data_pipeline.models.metadata import Department
from data_pipeline.taggers.base_tagger import BaseTagger, TagResult


class DepartmentTagger(BaseTagger):
    """
    과별 분류 태거

    이미지를 분석하여 4개 과 중 하나로 분류합니다:
    - 시디 (시각디자인)
    - 산디 (산업디자인)
    - 공예
    - 회화
    """

    # 분류 가능한 과별 목록
    departments: List[Department] = [
        Department.VISUAL_DESIGN,
        Department.INDUSTRIAL_DESIGN,
        Department.CRAFT,
        Department.PAINTING,
    ]

    def __init__(self, use_ml_model: bool = False, model_path: str = None):
        """
        DepartmentTagger 초기화

        Args:
            use_ml_model: ML 모델 사용 여부 (기본값: False, 규칙 기반)
            model_path: ML 모델 경로 (use_ml_model=True일 때 필요)
        """
        self.use_ml_model = use_ml_model
        self.model_path = model_path
        self.model = None

        if use_ml_model and model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """
        ML 모델을 로드합니다.

        Args:
            model_path: 모델 파일 경로
        """
        # TODO: 실제 ML 모델 로드 구현
        # 현재는 placeholder
        pass

    def predict(self, image: Union[Image.Image, np.ndarray, str]) -> TagResult:
        """
        이미지를 과별로 분류합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            TagResult: 분류 결과 (과별, 신뢰도)
        """
        # 이미지를 numpy 배열로 변환
        img_array = self._to_numpy(image)

        if self.use_ml_model and self.model is not None:
            return self._predict_with_model(img_array)
        else:
            return self._predict_rule_based(img_array)

    def _predict_with_model(self, img_array: np.ndarray) -> TagResult:
        """
        ML 모델을 사용하여 예측합니다.

        Args:
            img_array: 이미지 numpy 배열

        Returns:
            TagResult: 분류 결과
        """
        # TODO: 실제 ML 모델 추론 구현
        # 현재는 규칙 기반으로 fallback
        return self._predict_rule_based(img_array)

    def _predict_rule_based(self, img_array: np.ndarray) -> TagResult:
        """
        규칙 기반으로 예측합니다.

        이미지 특성을 분석하여 과별을 추정합니다.
        실제 학습된 모델이 없을 때 사용하는 휴리스틱입니다.

        Args:
            img_array: 이미지 numpy 배열

        Returns:
            TagResult: 분류 결과
        """
        # 이미지 특성 추출
        features = self._extract_features(img_array)

        # 규칙 기반 분류
        department, confidence = self._apply_rules(features)

        return TagResult(
            tag=department.value,
            confidence=confidence,
            metadata={
                "method": "rule_based",
                "features": {
                    "brightness": float(features["brightness"]),
                    "saturation": float(features["saturation"]),
                    "contrast": float(features["contrast"]),
                },
            },
        )

    def _extract_features(self, img_array: np.ndarray) -> dict:
        """
        이미지에서 특성을 추출합니다.

        Args:
            img_array: 이미지 numpy 배열

        Returns:
            특성 딕셔너리
        """
        # RGB 평균 밝기
        brightness = np.mean(img_array)

        # 채널별 표준편차 (대비)
        contrast = np.std(img_array)

        # 색상 다양성 (채널 간 차이)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            saturation = np.mean(np.abs(r.astype(float) - g.astype(float))) + \
                         np.mean(np.abs(g.astype(float) - b.astype(float)))
        else:
            saturation = 0.0

        return {
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
        }

    def _apply_rules(self, features: dict) -> tuple:
        """
        규칙을 적용하여 과별을 분류합니다.

        Args:
            features: 추출된 특성

        Returns:
            (Department, confidence) 튜플
        """
        brightness = features["brightness"]
        saturation = features["saturation"]
        contrast = features["contrast"]

        # 간단한 휴리스틱 규칙
        # 실제로는 학습된 모델이 이 역할을 대신함

        # 채도가 높으면 시각디자인
        if saturation > 50:
            return Department.VISUAL_DESIGN, 0.6 + random.uniform(0, 0.2)

        # 대비가 높으면 회화
        if contrast > 60:
            return Department.PAINTING, 0.5 + random.uniform(0, 0.25)

        # 밝기가 중간이면 산업디자인
        if 80 < brightness < 180:
            return Department.INDUSTRIAL_DESIGN, 0.5 + random.uniform(0, 0.2)

        # 기본값: 공예
        return Department.CRAFT, 0.4 + random.uniform(0, 0.2)
