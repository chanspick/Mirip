# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: 메타데이터 모델 테스트

TDD RED Phase: 실패하는 테스트를 먼저 작성합니다.
이 테스트는 Pydantic 모델의 유효성 검증을 검증합니다.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError


class TestDepartmentEnum:
    """과별 분류 열거형 테스트"""

    def test_department_visual_design_value(self):
        """시각디자인 값 확인"""
        from data_pipeline.models.metadata import Department

        assert Department.VISUAL_DESIGN.value == "시디"

    def test_department_industrial_design_value(self):
        """산업디자인 값 확인"""
        from data_pipeline.models.metadata import Department

        assert Department.INDUSTRIAL_DESIGN.value == "산디"

    def test_department_craft_value(self):
        """공예 값 확인"""
        from data_pipeline.models.metadata import Department

        assert Department.CRAFT.value == "공예"

    def test_department_painting_value(self):
        """회화 값 확인"""
        from data_pipeline.models.metadata import Department

        assert Department.PAINTING.value == "회화"

    def test_department_unknown_value(self):
        """미분류 값 확인"""
        from data_pipeline.models.metadata import Department

        assert Department.UNKNOWN.value == "미분류"

    def test_department_all_values(self):
        """모든 과별 값 존재 확인"""
        from data_pipeline.models.metadata import Department

        values = [d.value for d in Department]
        assert len(values) == 5
        assert "시디" in values
        assert "산디" in values
        assert "공예" in values
        assert "회화" in values
        assert "미분류" in values


class TestTierEnum:
    """대학 티어 열거형 테스트"""

    def test_tier_s_value(self):
        """S 티어 값 확인"""
        from data_pipeline.models.metadata import Tier

        assert Tier.S.value == "S"

    def test_tier_a_value(self):
        """A 티어 값 확인"""
        from data_pipeline.models.metadata import Tier

        assert Tier.A.value == "A"

    def test_tier_b_value(self):
        """B 티어 값 확인"""
        from data_pipeline.models.metadata import Tier

        assert Tier.B.value == "B"

    def test_tier_c_value(self):
        """C 티어 값 확인"""
        from data_pipeline.models.metadata import Tier

        assert Tier.C.value == "C"

    def test_tier_unknown_value(self):
        """미분류 티어 값 확인"""
        from data_pipeline.models.metadata import Tier

        assert Tier.UNKNOWN.value == "미분류"

    def test_tier_all_values(self):
        """모든 티어 값 존재 확인"""
        from data_pipeline.models.metadata import Tier

        values = [t.value for t in Tier]
        assert len(values) == 5
        assert "S" in values
        assert "A" in values
        assert "B" in values
        assert "C" in values
        assert "미분류" in values


class TestMediumEnum:
    """매체 열거형 테스트"""

    def test_medium_values_exist(self):
        """매체 값들 존재 확인"""
        from data_pipeline.models.metadata import Medium

        # 최소 7개 매체 (연필, 목탄, 수채화, 유화, 아크릴, 디지털, 혼합)
        assert len(list(Medium)) >= 7


class TestImageMetadata:
    """이미지 메타데이터 모델 테스트"""

    @pytest.fixture
    def valid_metadata_dict(self):
        """유효한 메타데이터 딕셔너리"""
        return {
            "image_id": "img_001",
            "original_filename": "test_image.jpg",
            "file_path": "/datasets/raw/test_image.jpg",
            "file_size": 1024000,
            "width": 1920,
            "height": 1080,
            "format": "JPEG",
        }

    def test_create_metadata_with_required_fields(self, valid_metadata_dict):
        """필수 필드로 메타데이터 생성"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.image_id == "img_001"
        assert metadata.original_filename == "test_image.jpg"
        assert metadata.file_path == "/datasets/raw/test_image.jpg"
        assert metadata.file_size == 1024000
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.format == "JPEG"

    def test_metadata_default_department(self, valid_metadata_dict):
        """기본 과별 분류는 UNKNOWN"""
        from data_pipeline.models.metadata import ImageMetadata, Department

        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.department == Department.UNKNOWN

    def test_metadata_default_tier(self, valid_metadata_dict):
        """기본 티어는 UNKNOWN"""
        from data_pipeline.models.metadata import ImageMetadata, Tier

        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.tier == Tier.UNKNOWN

    def test_metadata_default_tier_score(self, valid_metadata_dict):
        """기본 티어 점수는 0.0"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.tier_score == 0.0

    def test_metadata_default_is_manual_label(self, valid_metadata_dict):
        """기본 수동 라벨 여부는 False"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.is_manual_label is False

    def test_metadata_default_consent_status(self, valid_metadata_dict):
        """기본 동의 상태는 True"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.consent_status is True

    def test_metadata_default_copyright_cleared(self, valid_metadata_dict):
        """기본 저작권 확보 상태는 True"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.copyright_cleared is True

    def test_metadata_default_tags(self, valid_metadata_dict):
        """기본 태그는 빈 리스트"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.tags == []

    def test_metadata_created_at_auto_set(self, valid_metadata_dict):
        """생성 시간 자동 설정"""
        from data_pipeline.models.metadata import ImageMetadata

        before = datetime.now()
        metadata = ImageMetadata(**valid_metadata_dict)
        after = datetime.now()

        assert before <= metadata.created_at <= after

    def test_metadata_updated_at_auto_set(self, valid_metadata_dict):
        """수정 시간 자동 설정"""
        from data_pipeline.models.metadata import ImageMetadata

        before = datetime.now()
        metadata = ImageMetadata(**valid_metadata_dict)
        after = datetime.now()

        assert before <= metadata.updated_at <= after

    def test_metadata_tier_score_validation_min(self, valid_metadata_dict):
        """티어 점수 최소값 검증 (0.0 이상)"""
        from data_pipeline.models.metadata import ImageMetadata

        valid_metadata_dict["tier_score"] = -1.0

        with pytest.raises(ValidationError):
            ImageMetadata(**valid_metadata_dict)

    def test_metadata_tier_score_validation_max(self, valid_metadata_dict):
        """티어 점수 최대값 검증 (100.0 이하)"""
        from data_pipeline.models.metadata import ImageMetadata

        valid_metadata_dict["tier_score"] = 101.0

        with pytest.raises(ValidationError):
            ImageMetadata(**valid_metadata_dict)

    def test_metadata_tier_score_valid_range(self, valid_metadata_dict):
        """유효한 티어 점수 범위 (0-100)"""
        from data_pipeline.models.metadata import ImageMetadata

        valid_metadata_dict["tier_score"] = 85.5
        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.tier_score == 85.5

    def test_metadata_with_department(self, valid_metadata_dict):
        """과별 분류 설정"""
        from data_pipeline.models.metadata import ImageMetadata, Department

        valid_metadata_dict["department"] = Department.VISUAL_DESIGN
        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.department == Department.VISUAL_DESIGN

    def test_metadata_with_tier(self, valid_metadata_dict):
        """티어 설정"""
        from data_pipeline.models.metadata import ImageMetadata, Tier

        valid_metadata_dict["tier"] = Tier.S
        valid_metadata_dict["tier_score"] = 92.5
        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.tier == Tier.S
        assert metadata.tier_score == 92.5

    def test_metadata_with_tags(self, valid_metadata_dict):
        """태그 설정"""
        from data_pipeline.models.metadata import ImageMetadata

        valid_metadata_dict["tags"] = ["인물화", "수채화", "풍경"]
        metadata = ImageMetadata(**valid_metadata_dict)

        assert metadata.tags == ["인물화", "수채화", "풍경"]

    def test_metadata_missing_required_field(self):
        """필수 필드 누락 시 ValidationError"""
        from data_pipeline.models.metadata import ImageMetadata

        with pytest.raises(ValidationError):
            ImageMetadata(image_id="img_001")  # 다른 필수 필드 누락

    def test_metadata_json_serialization(self, valid_metadata_dict):
        """JSON 직렬화 테스트"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata(**valid_metadata_dict)
        json_str = metadata.model_dump_json()

        assert isinstance(json_str, str)
        assert "img_001" in json_str

    def test_metadata_dict_serialization(self, valid_metadata_dict):
        """딕셔너리 변환 테스트"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata(**valid_metadata_dict)
        data = metadata.model_dump()

        assert isinstance(data, dict)
        assert data["image_id"] == "img_001"

    def test_metadata_from_dict(self, valid_metadata_dict):
        """딕셔너리에서 모델 생성"""
        from data_pipeline.models.metadata import ImageMetadata

        metadata = ImageMetadata.model_validate(valid_metadata_dict)

        assert metadata.image_id == "img_001"

    def test_metadata_width_positive(self, valid_metadata_dict):
        """너비는 양수여야 함"""
        from data_pipeline.models.metadata import ImageMetadata

        valid_metadata_dict["width"] = 0

        with pytest.raises(ValidationError):
            ImageMetadata(**valid_metadata_dict)

    def test_metadata_height_positive(self, valid_metadata_dict):
        """높이는 양수여야 함"""
        from data_pipeline.models.metadata import ImageMetadata

        valid_metadata_dict["height"] = -100

        with pytest.raises(ValidationError):
            ImageMetadata(**valid_metadata_dict)

    def test_metadata_file_size_positive(self, valid_metadata_dict):
        """파일 크기는 양수여야 함"""
        from data_pipeline.models.metadata import ImageMetadata

        valid_metadata_dict["file_size"] = 0

        with pytest.raises(ValidationError):
            ImageMetadata(**valid_metadata_dict)
