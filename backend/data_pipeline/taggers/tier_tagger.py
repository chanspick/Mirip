# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: TierTagger 구현

이미지를 대학 티어로 분류하는 태거입니다.
S, A, B, C 4개 티어와 점수(0-100)를 예측합니다.
"""

import random
from typing import List, Union

import numpy as np
from PIL import Image

from data_pipeline.models.metadata import Tier
from data_pipeline.taggers.base_tagger import BaseTagger, TagResult


class TierTagger(BaseTagger):
    """
    대학 티어 분류 태거

    이미지 품질을 분석하여 대학 합격권 티어를 분류합니다:
    - S: 최상위 (서울대, 홍대, 국민대 등) - 점수 85-100
    - A: 상위 (건국대, 동국대 등) - 점수 70-84
    - B: 중위 (단국대, 명지대 등) - 점수 50-69
    - C: 개선 필요 - 점수 0-49
    """

    # 분류 가능한 티어 목록
    tiers: List[Tier] = [Tier.S, Tier.A, Tier.B, Tier.C]

    # 티어별 점수 범위
    TIER_THRESHOLDS = {
        Tier.S: (85, 100),
        Tier.A: (70, 84),
        Tier.B: (50, 69),
        Tier.C: (0, 49),
    }

    def __init__(self, use_ml_model: bool = False, model_path: str = None):
        """
        TierTagger 초기화

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

    def score_to_tier(self, score: float) -> Tier:
        """
        점수를 티어로 변환합니다.

        Args:
            score: 0-100 점수

        Returns:
            Tier: 해당 점수의 티어
        """
        if score >= 85:
            return Tier.S
        elif score >= 70:
            return Tier.A
        elif score >= 50:
            return Tier.B
        else:
            return Tier.C

    def predict(self, image: Union[Image.Image, np.ndarray, str]) -> TagResult:
        """
        이미지를 티어로 분류합니다.

        Args:
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            TagResult: 분류 결과 (티어, 신뢰도, 점수)
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

        이미지 품질 특성을 분석하여 점수와 티어를 추정합니다.
        실제 학습된 모델이 없을 때 사용하는 휴리스틱입니다.

        Args:
            img_array: 이미지 numpy 배열

        Returns:
            TagResult: 분류 결과
        """
        # 이미지 품질 특성 추출
        quality_features = self._extract_quality_features(img_array)

        # 품질 점수 계산 (0-100)
        score = self._calculate_quality_score(quality_features)

        # 점수를 티어로 변환
        tier = self.score_to_tier(score)

        # 신뢰도 계산 (점수가 경계에 가까울수록 낮음)
        confidence = self._calculate_confidence(score)

        return TagResult(
            tag=tier.value,
            confidence=confidence,
            metadata={
                "score": round(score, 2),
                "method": "rule_based",
                "quality_features": {
                    "sharpness": float(quality_features["sharpness"]),
                    "composition": float(quality_features["composition"]),
                    "color_balance": float(quality_features["color_balance"]),
                },
            },
        )

    def _extract_quality_features(self, img_array: np.ndarray) -> dict:
        """
        이미지 품질 특성을 추출합니다.

        Args:
            img_array: 이미지 numpy 배열

        Returns:
            품질 특성 딕셔너리
        """
        # 선명도 (라플라시안 분산)
        gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
        laplacian_var = np.var(np.gradient(np.gradient(gray)))
        sharpness = min(laplacian_var / 100, 1.0)

        # 구도 (중심 vs 가장자리 밝기 비율)
        h, w = gray.shape[:2]
        center_region = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        edge_region = np.concatenate([
            gray[:h // 4, :].flatten(),
            gray[3 * h // 4 :, :].flatten(),
            gray[:, :w // 4].flatten(),
            gray[:, 3 * w // 4 :].flatten(),
        ])
        composition = abs(np.mean(center_region) - np.mean(edge_region)) / 255

        # 색상 균형 (RGB 채널 간 균형)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            r_mean = np.mean(img_array[:, :, 0])
            g_mean = np.mean(img_array[:, :, 1])
            b_mean = np.mean(img_array[:, :, 2])
            max_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
            color_balance = 1.0 - (max_diff / 255)
        else:
            color_balance = 0.5

        return {
            "sharpness": sharpness,
            "composition": composition,
            "color_balance": color_balance,
        }

    def _calculate_quality_score(self, features: dict) -> float:
        """
        품질 특성을 기반으로 점수를 계산합니다.

        Args:
            features: 품질 특성

        Returns:
            0-100 범위의 점수
        """
        # 가중 평균 점수 계산
        sharpness_score = features["sharpness"] * 35  # 최대 35점
        composition_score = features["composition"] * 35  # 최대 35점
        color_score = features["color_balance"] * 30  # 최대 30점

        base_score = sharpness_score + composition_score + color_score

        # 약간의 랜덤성 추가 (실제 모델의 불확실성 시뮬레이션)
        noise = random.uniform(-5, 5)
        final_score = max(0, min(100, base_score + noise))

        return final_score

    def _calculate_confidence(self, score: float) -> float:
        """
        점수를 기반으로 신뢰도를 계산합니다.

        점수가 티어 경계에 가까울수록 신뢰도가 낮습니다.

        Args:
            score: 품질 점수

        Returns:
            0-1 범위의 신뢰도
        """
        # 경계값들
        boundaries = [85, 70, 50]

        # 가장 가까운 경계와의 거리
        min_distance = min(abs(score - b) for b in boundaries)

        # 거리가 멀수록 신뢰도 높음 (최대 15점 거리에서 최고 신뢰도)
        confidence = min(0.95, 0.5 + (min_distance / 30))

        return round(confidence, 3)
