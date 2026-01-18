# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: Collector 테스트

TDD RED Phase: 데이터 수집기에 대한 테스트를 작성합니다.
BaseCollector와 ImageCollector의 기능을 검증합니다.
"""

import tempfile
from pathlib import Path
from typing import Iterator

import pytest


class TestBaseCollector:
    """BaseCollector 추상 클래스 테스트"""

    def test_base_collector_is_abstract(self):
        """BaseCollector는 추상 클래스여야 함"""
        from data_pipeline.collectors.base_collector import BaseCollector

        with pytest.raises(TypeError):
            BaseCollector()  # 추상 클래스 직접 인스턴스화 불가

    def test_base_collector_has_collect_method(self):
        """collect 추상 메서드 존재 확인"""
        from data_pipeline.collectors.base_collector import BaseCollector
        import inspect

        assert hasattr(BaseCollector, "collect")
        assert inspect.isabstract(BaseCollector)

    def test_base_collector_has_validate_method(self):
        """validate 추상 메서드 존재 확인"""
        from data_pipeline.collectors.base_collector import BaseCollector

        assert hasattr(BaseCollector, "validate")


class TestCollectionResult:
    """CollectionResult 모델 테스트"""

    def test_collection_result_creation(self):
        """CollectionResult 생성"""
        from data_pipeline.collectors.base_collector import CollectionResult

        result = CollectionResult(
            image_id="img_001",
            source_path="/source/image.jpg",
            local_path="/local/image.jpg",
            success=True,
        )

        assert result.image_id == "img_001"
        assert result.source_path == "/source/image.jpg"
        assert result.local_path == "/local/image.jpg"
        assert result.success is True
        assert result.error_message is None

    def test_collection_result_with_error(self):
        """실패한 CollectionResult 생성"""
        from data_pipeline.collectors.base_collector import CollectionResult

        result = CollectionResult(
            image_id="img_002",
            source_path="/source/bad.jpg",
            local_path="",
            success=False,
            error_message="File not found",
        )

        assert result.success is False
        assert result.error_message == "File not found"


class TestImageCollector:
    """ImageCollector 테스트"""

    @pytest.fixture
    def temp_output_dir(self):
        """임시 출력 디렉토리"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def temp_source_dir(self, sample_jpeg_bytes):
        """테스트용 소스 디렉토리 (이미지 포함)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir)

            # 테스트 이미지 생성
            (source_dir / "test1.jpg").write_bytes(sample_jpeg_bytes)
            (source_dir / "test2.jpg").write_bytes(sample_jpeg_bytes)
            (source_dir / "subdir").mkdir()
            (source_dir / "subdir" / "test3.jpg").write_bytes(sample_jpeg_bytes)

            # 비이미지 파일
            (source_dir / "readme.txt").write_text("Not an image")

            yield source_dir

    def test_image_collector_initialization(self, temp_output_dir):
        """ImageCollector 초기화"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)

        assert collector.output_dir == temp_output_dir

    def test_image_collector_inherits_base(self, temp_output_dir):
        """ImageCollector는 BaseCollector를 상속"""
        from data_pipeline.collectors.base_collector import BaseCollector
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)

        assert isinstance(collector, BaseCollector)

    def test_image_collector_collect_returns_iterator(
        self, temp_output_dir, temp_source_dir
    ):
        """collect 메서드는 Iterator를 반환"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)
        results = collector.collect(source=str(temp_source_dir))

        assert isinstance(results, Iterator)

    def test_image_collector_collects_images(
        self, temp_output_dir, temp_source_dir
    ):
        """이미지 파일 수집"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)
        results = list(collector.collect(source=str(temp_source_dir)))

        # 3개 이미지 수집 (test1.jpg, test2.jpg, subdir/test3.jpg)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_image_collector_ignores_non_images(
        self, temp_output_dir, temp_source_dir
    ):
        """비이미지 파일 무시"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)
        results = list(collector.collect(source=str(temp_source_dir)))

        # readme.txt는 수집되지 않음
        local_paths = [r.local_path for r in results]
        assert not any("readme.txt" in p for p in local_paths)

    def test_image_collector_copies_to_output(
        self, temp_output_dir, temp_source_dir
    ):
        """이미지를 출력 디렉토리로 복사"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)
        results = list(collector.collect(source=str(temp_source_dir)))

        # 복사된 파일 존재 확인
        for result in results:
            assert Path(result.local_path).exists()

    def test_image_collector_generates_unique_ids(
        self, temp_output_dir, temp_source_dir
    ):
        """고유한 이미지 ID 생성"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)
        results = list(collector.collect(source=str(temp_source_dir)))

        ids = [r.image_id for r in results]
        assert len(ids) == len(set(ids))  # 모든 ID가 고유함

    def test_image_collector_validate_valid_image(
        self, temp_output_dir, sample_jpeg_bytes
    ):
        """유효한 이미지 검증 통과"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)

        # 임시 이미지 파일 생성
        image_path = temp_output_dir / "valid.jpg"
        image_path.write_bytes(sample_jpeg_bytes)

        assert collector.validate(str(image_path)) is True

    def test_image_collector_validate_invalid_file(self, temp_output_dir):
        """유효하지 않은 파일 검증 실패"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)

        # 텍스트 파일 생성
        text_path = temp_output_dir / "not_image.txt"
        text_path.write_text("This is not an image")

        assert collector.validate(str(text_path)) is False

    def test_image_collector_validate_nonexistent_file(self, temp_output_dir):
        """존재하지 않는 파일 검증 실패"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)

        assert collector.validate("/nonexistent/path.jpg") is False

    def test_image_collector_validate_corrupted_image(self, temp_output_dir):
        """손상된 이미지 파일 검증 실패"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)

        # 손상된 이미지 파일 (잘못된 바이트)
        corrupted_path = temp_output_dir / "corrupted.jpg"
        corrupted_path.write_bytes(b"\xff\xd8\xff\x00\x00corrupted")

        assert collector.validate(str(corrupted_path)) is False

    def test_image_collector_supported_formats(self, temp_output_dir):
        """지원되는 이미지 형식"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)

        # 지원 형식 확인
        assert ".jpg" in collector.supported_formats
        assert ".jpeg" in collector.supported_formats
        assert ".png" in collector.supported_formats

    def test_image_collector_recursive_collection(
        self, temp_output_dir, temp_source_dir
    ):
        """하위 디렉토리 재귀 수집"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir, recursive=True)
        results = list(collector.collect(source=str(temp_source_dir)))

        # subdir/test3.jpg도 수집됨
        assert len(results) == 3

    def test_image_collector_non_recursive_collection(
        self, temp_output_dir, temp_source_dir
    ):
        """비재귀 수집 (루트만)"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir, recursive=False)
        results = list(collector.collect(source=str(temp_source_dir)))

        # 루트의 test1.jpg, test2.jpg만 수집
        assert len(results) == 2

    def test_image_collector_handles_empty_directory(self, temp_output_dir):
        """빈 디렉토리 처리"""
        from data_pipeline.collectors.image_collector import ImageCollector

        with tempfile.TemporaryDirectory() as empty_dir:
            collector = ImageCollector(output_dir=temp_output_dir)
            results = list(collector.collect(source=empty_dir))

            assert len(results) == 0

    def test_image_collector_handles_invalid_source(self, temp_output_dir):
        """잘못된 소스 경로 처리"""
        from data_pipeline.collectors.image_collector import ImageCollector

        collector = ImageCollector(output_dir=temp_output_dir)

        with pytest.raises(FileNotFoundError):
            list(collector.collect(source="/nonexistent/path"))
