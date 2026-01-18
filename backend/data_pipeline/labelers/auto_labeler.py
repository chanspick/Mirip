# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: AutoLabeler 구현

DepartmentTagger와 TierTagger를 조합하여
이미지를 자동으로 라벨링하는 클래스입니다.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from data_pipeline.labelers.base_labeler import BaseLabeler, LabelingResult
from data_pipeline.models.metadata import Department, Tier
from data_pipeline.taggers.department_tagger import DepartmentTagger
from data_pipeline.taggers.tier_tagger import TierTagger


class AutoLabeler(BaseLabeler):
    """
    자동 라벨러

    DepartmentTagger와 TierTagger를 사용하여 이미지를
    자동으로 분류하고 라벨링합니다.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        use_ml_model: bool = False,
        department_model_path: Optional[str] = None,
        tier_model_path: Optional[str] = None,
    ):
        """
        AutoLabeler 초기화

        Args:
            confidence_threshold: 신뢰도 임계값 (기본값: 0.6)
            use_ml_model: ML 모델 사용 여부 (기본값: False)
            department_model_path: 과별 분류 모델 경로
            tier_model_path: 티어 분류 모델 경로
        """
        self.confidence_threshold = confidence_threshold
        self.use_ml_model = use_ml_model

        # 태거 초기화
        self.department_tagger = DepartmentTagger(
            use_ml_model=use_ml_model,
            model_path=department_model_path,
        )
        self.tier_tagger = TierTagger(
            use_ml_model=use_ml_model,
            model_path=tier_model_path,
        )

    def label(
        self, image_id: str, image: Union[Image.Image, np.ndarray, str]
    ) -> LabelingResult:
        """
        단일 이미지를 라벨링합니다.

        DepartmentTagger와 TierTagger를 사용하여
        과별 분류와 티어 분류를 수행합니다.

        Args:
            image_id: 이미지 식별자
            image: PIL Image, numpy 배열, 또는 파일 경로

        Returns:
            LabelingResult: 라벨링 결과
        """
        # 이미지를 numpy 배열로 변환
        img_array = self._to_numpy(image)

        # 정규화된 배열(0-1)인 경우 0-255로 변환
        if img_array.dtype in [np.float32, np.float64]:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)

        # 과별 분류
        dept_result = self.department_tagger.predict(img_array)
        department = Department(dept_result.tag)
        department_confidence = dept_result.confidence

        # 티어 분류
        tier_result = self.tier_tagger.predict(img_array)
        tier = Tier(tier_result.tag)
        tier_confidence = tier_result.confidence
        tier_score = tier_result.metadata.get("score", 0.0) if tier_result.metadata else 0.0

        return LabelingResult(
            image_id=image_id,
            department=department,
            department_confidence=department_confidence,
            tier=tier,
            tier_confidence=tier_confidence,
            tier_score=tier_score,
            metadata={
                "department_metadata": dept_result.metadata,
                "tier_metadata": tier_result.metadata,
            },
        )

    def get_statistics(
        self, results: List[LabelingResult]
    ) -> Dict[str, Any]:
        """
        라벨링 결과에 대한 통계를 계산합니다.

        Args:
            results: LabelingResult 리스트

        Returns:
            통계 딕셔너리
        """
        if not results:
            return {
                "total_count": 0,
                "department_distribution": {},
                "tier_distribution": {},
                "average_confidence": {
                    "department": 0.0,
                    "tier": 0.0,
                },
            }

        # 과별 분포
        dept_counts = Counter(r.department.value for r in results)
        dept_distribution = {
            dept: count / len(results)
            for dept, count in dept_counts.items()
        }

        # 티어 분포
        tier_counts = Counter(r.tier.value for r in results)
        tier_distribution = {
            tier: count / len(results)
            for tier, count in tier_counts.items()
        }

        # 평균 신뢰도
        avg_dept_confidence = sum(r.department_confidence for r in results) / len(results)
        avg_tier_confidence = sum(r.tier_confidence for r in results) / len(results)

        return {
            "total_count": len(results),
            "department_distribution": dept_distribution,
            "tier_distribution": tier_distribution,
            "average_confidence": {
                "department": round(avg_dept_confidence, 4),
                "tier": round(avg_tier_confidence, 4),
            },
        }

    def filter_by_confidence(
        self,
        results: List[LabelingResult],
        threshold: float = 0.6,
    ) -> List[LabelingResult]:
        """
        신뢰도 임계값 이상의 결과만 필터링합니다.

        과별 분류 또는 티어 분류 중 하나라도 임계값 이상이면 포함합니다.

        Args:
            results: LabelingResult 리스트
            threshold: 신뢰도 임계값 (기본값: 0.6)

        Returns:
            필터링된 LabelingResult 리스트
        """
        return [
            r for r in results
            if r.department_confidence >= threshold
            or r.tier_confidence >= threshold
        ]

    def filter_high_confidence(
        self,
        results: List[LabelingResult],
        threshold: float = 0.8,
    ) -> List[LabelingResult]:
        """
        고신뢰도 결과만 필터링합니다.

        과별 분류와 티어 분류 모두 임계값 이상인 결과만 포함합니다.

        Args:
            results: LabelingResult 리스트
            threshold: 신뢰도 임계값 (기본값: 0.8)

        Returns:
            고신뢰도 LabelingResult 리스트
        """
        return [r for r in results if r.is_high_confidence(threshold)]

    def filter_needs_review(
        self,
        results: List[LabelingResult],
        threshold: float = 0.6,
    ) -> List[LabelingResult]:
        """
        수동 검토가 필요한 결과만 필터링합니다.

        Args:
            results: LabelingResult 리스트
            threshold: 신뢰도 임계값 (기본값: 0.6)

        Returns:
            검토 필요 LabelingResult 리스트
        """
        return [r for r in results if r.needs_review(threshold)]
