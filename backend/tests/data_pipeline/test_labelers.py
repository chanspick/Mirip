# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: Labeler 테스트

TDD RED Phase: 자동 라벨링 시스템에 대한 테스트를 작성합니다.
BaseLabeler, AutoLabeler를 검증합니다.
"""

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest
from PIL import Image

from data_pipeline.models.metadata import Department, Tier, ImageMetadata


class TestBaseLabeler:
    """BaseLabeler 추상 클래스 테스트"""

    def test_base_labeler_is_abstract(self):
        """BaseLabeler는 추상 클래스여야 함"""
        from data_pipeline.labelers.base_labeler import BaseLabeler

        with pytest.raises(TypeError):
            BaseLabeler()  # 추상 클래스 직접 인스턴스화 불가

    def test_base_labeler_has_label_method(self):
        """label 추상 메서드 존재 확인"""
        from data_pipeline.labelers.base_labeler import BaseLabeler
        import inspect

        assert hasattr(BaseLabeler, "label")
        assert inspect.isabstract(BaseLabeler)

    def test_base_labeler_has_label_batch_method(self):
        """label_batch 메서드 존재 확인"""
        from data_pipeline.labelers.base_labeler import BaseLabeler

        assert hasattr(BaseLabeler, "label_batch")


class TestLabelingResult:
    """LabelingResult 모델 테스트"""

    def test_labeling_result_creation(self):
        """LabelingResult 생성"""
        from data_pipeline.labelers.base_labeler import LabelingResult

        result = LabelingResult(
            image_id="img_001",
            department=Department.VISUAL_DESIGN,
            department_confidence=0.85,
            tier=Tier.A,
            tier_confidence=0.78,
            tier_score=75.5,
        )

        assert result.image_id == "img_001"
        assert result.department == Department.VISUAL_DESIGN
        assert result.department_confidence == 0.85
        assert result.tier == Tier.A
        assert result.tier_confidence == 0.78
        assert result.tier_score == 75.5

    def test_labeling_result_is_high_confidence(self):
        """고신뢰도 여부 확인"""
        from data_pipeline.labelers.base_labeler import LabelingResult

        # 높은 신뢰도 (둘 다 0.8 이상)
        high_conf = LabelingResult(
            image_id="img_001",
            department=Department.VISUAL_DESIGN,
            department_confidence=0.85,
            tier=Tier.S,
            tier_confidence=0.82,
            tier_score=90.0,
        )
        assert high_conf.is_high_confidence(threshold=0.8)

        # 낮은 신뢰도 (하나라도 0.8 미만)
        low_conf = LabelingResult(
            image_id="img_002",
            department=Department.CRAFT,
            department_confidence=0.75,
            tier=Tier.B,
            tier_confidence=0.65,
            tier_score=55.0,
        )
        assert not low_conf.is_high_confidence(threshold=0.8)

    def test_labeling_result_needs_review(self):
        """수동 검토 필요 여부 확인"""
        from data_pipeline.labelers.base_labeler import LabelingResult

        result = LabelingResult(
            image_id="img_001",
            department=Department.UNKNOWN,
            department_confidence=0.5,
            tier=Tier.C,
            tier_confidence=0.4,
            tier_score=30.0,
        )

        # 낮은 신뢰도일 때 검토 필요
        assert result.needs_review(threshold=0.6)


class TestAutoLabeler:
    """AutoLabeler 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 샘플 이미지"""
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    @pytest.fixture
    def temp_image_file(self, sample_image):
        """테스트용 임시 이미지 파일"""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            sample_image.save(f, format="JPEG")
            return Path(f.name)

    def test_auto_labeler_initialization(self):
        """AutoLabeler 초기화"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()

        assert labeler is not None

    def test_auto_labeler_inherits_base(self):
        """AutoLabeler는 BaseLabeler를 상속"""
        from data_pipeline.labelers.base_labeler import BaseLabeler
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()

        assert isinstance(labeler, BaseLabeler)

    def test_auto_labeler_has_taggers(self):
        """AutoLabeler는 DepartmentTagger와 TierTagger를 가짐"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler
        from data_pipeline.taggers.department_tagger import DepartmentTagger
        from data_pipeline.taggers.tier_tagger import TierTagger

        labeler = AutoLabeler()

        assert hasattr(labeler, "department_tagger")
        assert hasattr(labeler, "tier_tagger")
        assert isinstance(labeler.department_tagger, DepartmentTagger)
        assert isinstance(labeler.tier_tagger, TierTagger)

    def test_auto_labeler_label_returns_labeling_result(self, sample_image):
        """label 메서드는 LabelingResult를 반환"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler
        from data_pipeline.labelers.base_labeler import LabelingResult

        labeler = AutoLabeler()
        result = labeler.label("img_001", sample_image)

        assert isinstance(result, LabelingResult)

    def test_auto_labeler_label_contains_all_fields(self, sample_image):
        """라벨링 결과에 모든 필드 포함"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()
        result = labeler.label("img_001", sample_image)

        assert result.image_id == "img_001"
        assert result.department is not None
        assert 0.0 <= result.department_confidence <= 1.0
        assert result.tier is not None
        assert 0.0 <= result.tier_confidence <= 1.0
        assert 0.0 <= result.tier_score <= 100.0

    def test_auto_labeler_label_numpy_array(self):
        """numpy 배열 라벨링"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = labeler.label("img_002", img_array)

        assert result.image_id == "img_002"
        assert result.department is not None
        assert result.tier is not None

    def test_auto_labeler_label_file_path(self, temp_image_file):
        """파일 경로로 라벨링"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()
        result = labeler.label("img_003", str(temp_image_file))

        assert result.image_id == "img_003"
        assert result.department is not None

    def test_auto_labeler_label_batch(self):
        """배치 이미지 라벨링"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()
        images = [Image.new("RGB", (100, 100)) for _ in range(5)]
        image_ids = [f"img_{i:03d}" for i in range(5)]

        results = labeler.label_batch(image_ids, images)

        assert len(results) == 5
        assert all(r.image_id == f"img_{i:03d}" for i, r in enumerate(results))

    def test_auto_labeler_confidence_threshold(self, sample_image):
        """신뢰도 임계값 설정"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        # 높은 임계값으로 초기화
        labeler = AutoLabeler(confidence_threshold=0.9)

        assert labeler.confidence_threshold == 0.9

    def test_auto_labeler_to_metadata(self, sample_image):
        """라벨링 결과를 ImageMetadata로 변환"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()
        result = labeler.label("img_001", sample_image)

        # 라벨링 결과를 메타데이터로 변환
        metadata_update = result.to_metadata_dict()

        assert "department" in metadata_update
        assert "tier" in metadata_update
        assert "tier_score" in metadata_update
        assert "is_manual_label" in metadata_update
        assert metadata_update["is_manual_label"] is False


class TestAutoLabelerWithPreprocessing:
    """전처리와 함께 사용하는 AutoLabeler 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 샘플 이미지"""
        return Image.new("RGB", (500, 500), color=(100, 150, 200))

    def test_auto_labeler_with_resize(self, sample_image):
        """리사이즈 전처리 후 라벨링"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        # 전처리
        resize = ResizePreprocessor(target_size=(224, 224))
        preprocessed = resize.process(sample_image)

        # 라벨링
        labeler = AutoLabeler()
        result = labeler.label("img_001", preprocessed)

        assert result.department is not None
        assert result.tier is not None

    def test_auto_labeler_with_full_pipeline(self, sample_image):
        """전체 파이프라인 (리사이즈 -> 정규화 -> 라벨링)"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        # 전처리
        resize = ResizePreprocessor(target_size=(224, 224))
        normalize = NormalizePreprocessor()

        resized = resize.process(sample_image)
        # 정규화된 이미지는 numpy 배열이므로 AutoLabeler도 지원해야 함
        normalized = normalize.process(resized)

        # 라벨링 (numpy 배열 입력)
        labeler = AutoLabeler()
        result = labeler.label("img_pipeline", normalized)

        assert result.department is not None
        assert result.tier is not None


class TestLabelerStatistics:
    """라벨러 통계 기능 테스트"""

    def test_auto_labeler_get_statistics(self):
        """라벨링 통계 조회"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()

        # 여러 이미지 라벨링
        images = [Image.new("RGB", (100, 100), color=(i * 25, i * 20, i * 15))
                  for i in range(10)]
        image_ids = [f"img_{i:03d}" for i in range(10)]

        results = labeler.label_batch(image_ids, images)

        # 통계 조회
        stats = labeler.get_statistics(results)

        assert "total_count" in stats
        assert stats["total_count"] == 10
        assert "department_distribution" in stats
        assert "tier_distribution" in stats
        assert "average_confidence" in stats

    def test_auto_labeler_filter_by_confidence(self):
        """신뢰도로 결과 필터링"""
        from data_pipeline.labelers.auto_labeler import AutoLabeler

        labeler = AutoLabeler()

        # 여러 이미지 라벨링
        images = [Image.new("RGB", (100, 100)) for _ in range(5)]
        image_ids = [f"img_{i:03d}" for i in range(5)]

        results = labeler.label_batch(image_ids, images)

        # 고신뢰도 결과만 필터링
        high_confidence = labeler.filter_by_confidence(results, threshold=0.5)

        assert isinstance(high_confidence, list)
        # 모든 결과가 지정된 임계값 이상의 신뢰도를 가져야 함
        for result in high_confidence:
            assert result.department_confidence >= 0.5 or result.tier_confidence >= 0.5
