# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: Tagger 테스트

TDD RED Phase: 과별 분류 및 대학 티어 분류기에 대한 테스트를 작성합니다.
BaseTagger, DepartmentTagger, TierTagger를 검증합니다.
"""

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from data_pipeline.models.metadata import Department, Tier


class TestBaseTagger:
    """BaseTagger 추상 클래스 테스트"""

    def test_base_tagger_is_abstract(self):
        """BaseTagger는 추상 클래스여야 함"""
        from data_pipeline.taggers.base_tagger import BaseTagger

        with pytest.raises(TypeError):
            BaseTagger()  # 추상 클래스 직접 인스턴스화 불가

    def test_base_tagger_has_predict_method(self):
        """predict 추상 메서드 존재 확인"""
        from data_pipeline.taggers.base_tagger import BaseTagger
        import inspect

        assert hasattr(BaseTagger, "predict")
        assert inspect.isabstract(BaseTagger)

    def test_base_tagger_has_predict_batch_method(self):
        """predict_batch 메서드 존재 확인"""
        from data_pipeline.taggers.base_tagger import BaseTagger

        assert hasattr(BaseTagger, "predict_batch")


class TestTagResult:
    """TagResult 모델 테스트"""

    def test_tag_result_creation(self):
        """TagResult 생성"""
        from data_pipeline.taggers.base_tagger import TagResult

        result = TagResult(
            tag="시디",
            confidence=0.95,
        )

        assert result.tag == "시디"
        assert result.confidence == 0.95

    def test_tag_result_with_metadata(self):
        """메타데이터가 있는 TagResult 생성"""
        from data_pipeline.taggers.base_tagger import TagResult

        result = TagResult(
            tag="S",
            confidence=0.88,
            metadata={"score": 92.5, "model_version": "1.0"},
        )

        assert result.metadata["score"] == 92.5
        assert result.metadata["model_version"] == "1.0"


class TestDepartmentTagger:
    """DepartmentTagger 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 샘플 이미지"""
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    @pytest.fixture
    def sample_numpy_array(self):
        """테스트용 numpy 배열"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    def test_department_tagger_initialization(self):
        """DepartmentTagger 초기화"""
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        tagger = DepartmentTagger()

        assert tagger is not None

    def test_department_tagger_inherits_base(self):
        """DepartmentTagger는 BaseTagger를 상속"""
        from data_pipeline.taggers.base_tagger import BaseTagger
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        tagger = DepartmentTagger()

        assert isinstance(tagger, BaseTagger)

    def test_department_tagger_predict_returns_tag_result(self, sample_image):
        """predict 메서드는 TagResult를 반환"""
        from data_pipeline.taggers.base_tagger import TagResult
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        tagger = DepartmentTagger()
        result = tagger.predict(sample_image)

        assert isinstance(result, TagResult)

    def test_department_tagger_predict_valid_department(self, sample_image):
        """예측 결과는 유효한 과별 값"""
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        tagger = DepartmentTagger()
        result = tagger.predict(sample_image)

        valid_departments = [d.value for d in Department]
        assert result.tag in valid_departments

    def test_department_tagger_predict_confidence_range(self, sample_image):
        """신뢰도는 0-1 범위"""
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        tagger = DepartmentTagger()
        result = tagger.predict(sample_image)

        assert 0.0 <= result.confidence <= 1.0

    def test_department_tagger_predict_numpy_array(self, sample_numpy_array):
        """numpy 배열 예측"""
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        tagger = DepartmentTagger()
        result = tagger.predict(sample_numpy_array)

        assert result is not None
        assert result.tag is not None

    def test_department_tagger_predict_batch(self):
        """배치 이미지 예측"""
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        tagger = DepartmentTagger()
        images = [Image.new("RGB", (224, 224)) for _ in range(3)]
        results = tagger.predict_batch(images)

        assert len(results) == 3
        assert all(r.tag is not None for r in results)

    def test_department_tagger_all_departments_predictable(self):
        """모든 과별이 예측 가능한 값인지 확인"""
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        tagger = DepartmentTagger()

        # 분류 가능한 과별 목록 확인
        assert hasattr(tagger, "departments")
        for dept in [Department.VISUAL_DESIGN, Department.INDUSTRIAL_DESIGN,
                     Department.CRAFT, Department.PAINTING]:
            assert dept in tagger.departments or dept.value in [d.value for d in tagger.departments]


class TestTierTagger:
    """TierTagger 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 샘플 이미지"""
        return Image.new("RGB", (224, 224), color=(100, 150, 200))

    def test_tier_tagger_initialization(self):
        """TierTagger 초기화"""
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()

        assert tagger is not None

    def test_tier_tagger_inherits_base(self):
        """TierTagger는 BaseTagger를 상속"""
        from data_pipeline.taggers.base_tagger import BaseTagger
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()

        assert isinstance(tagger, BaseTagger)

    def test_tier_tagger_predict_returns_tag_result(self, sample_image):
        """predict 메서드는 TagResult를 반환"""
        from data_pipeline.taggers.base_tagger import TagResult
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()
        result = tagger.predict(sample_image)

        assert isinstance(result, TagResult)

    def test_tier_tagger_predict_valid_tier(self, sample_image):
        """예측 결과는 유효한 티어 값"""
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()
        result = tagger.predict(sample_image)

        valid_tiers = [t.value for t in Tier]
        assert result.tag in valid_tiers

    def test_tier_tagger_predict_confidence_range(self, sample_image):
        """신뢰도는 0-1 범위"""
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()
        result = tagger.predict(sample_image)

        assert 0.0 <= result.confidence <= 1.0

    def test_tier_tagger_predict_includes_score(self, sample_image):
        """예측 결과에 점수 포함"""
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()
        result = tagger.predict(sample_image)

        # metadata에 score가 있어야 함
        assert result.metadata is not None
        assert "score" in result.metadata
        assert 0.0 <= result.metadata["score"] <= 100.0

    def test_tier_tagger_score_to_tier_mapping(self):
        """점수-티어 매핑 테스트"""
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()

        # 점수 범위에 따른 티어 매핑 테스트
        # S: 85-100, A: 70-84, B: 50-69, C: 0-49
        assert tagger.score_to_tier(95) == Tier.S
        assert tagger.score_to_tier(85) == Tier.S
        assert tagger.score_to_tier(75) == Tier.A
        assert tagger.score_to_tier(70) == Tier.A
        assert tagger.score_to_tier(60) == Tier.B
        assert tagger.score_to_tier(50) == Tier.B
        assert tagger.score_to_tier(40) == Tier.C
        assert tagger.score_to_tier(0) == Tier.C

    def test_tier_tagger_predict_batch(self):
        """배치 이미지 예측"""
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()
        images = [Image.new("RGB", (224, 224)) for _ in range(4)]
        results = tagger.predict_batch(images)

        assert len(results) == 4
        assert all(r.tag is not None for r in results)
        assert all("score" in r.metadata for r in results)

    def test_tier_tagger_all_tiers_in_range(self):
        """모든 티어가 유효한 범위에 있는지 확인"""
        from data_pipeline.taggers.tier_tagger import TierTagger

        tagger = TierTagger()

        # 모든 티어가 매핑 가능한지 확인
        assert hasattr(tagger, "tiers")
        for tier in [Tier.S, Tier.A, Tier.B, Tier.C]:
            assert tier in tagger.tiers or tier.value in [t.value for t in tagger.tiers]


class TestRuleBasedTagger:
    """규칙 기반 태거 테스트 (ML 모델 없이 간단한 규칙 사용)"""

    @pytest.fixture
    def sample_image_bright(self):
        """밝은 이미지"""
        return Image.new("RGB", (100, 100), color=(255, 255, 255))

    @pytest.fixture
    def sample_image_dark(self):
        """어두운 이미지"""
        return Image.new("RGB", (100, 100), color=(50, 50, 50))

    def test_rule_based_department_tagger(self, sample_image_bright):
        """규칙 기반 과별 태거 (학습 없이 사용 가능)"""
        from data_pipeline.taggers.department_tagger import DepartmentTagger

        # 규칙 기반 모드로 초기화
        tagger = DepartmentTagger(use_ml_model=False)
        result = tagger.predict(sample_image_bright)

        # 규칙 기반이어도 결과 반환
        assert result is not None
        assert result.tag is not None

    def test_rule_based_tier_tagger(self, sample_image_bright):
        """규칙 기반 티어 태거 (학습 없이 사용 가능)"""
        from data_pipeline.taggers.tier_tagger import TierTagger

        # 규칙 기반 모드로 초기화
        tagger = TierTagger(use_ml_model=False)
        result = tagger.predict(sample_image_bright)

        # 규칙 기반이어도 결과 반환
        assert result is not None
        assert result.tag is not None
        assert "score" in result.metadata
