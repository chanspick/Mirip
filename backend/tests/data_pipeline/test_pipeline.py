# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: DataPipeline 통합 테스트

TDD RED Phase: 전체 데이터 파이프라인 통합 테스트를 작성합니다.
수집 -> 전처리 -> 태깅 -> 라벨링 -> 저장 전체 흐름을 검증합니다.
"""

import tempfile
from pathlib import Path

import pytest
from PIL import Image

from data_pipeline.models.metadata import Department, Tier


class TestDataPipeline:
    """DataPipeline 통합 테스트"""

    @pytest.fixture
    def temp_dirs(self):
        """테스트용 임시 디렉토리들"""
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                source_path = Path(source_dir)
                output_path = Path(output_dir)

                # 테스트 이미지 생성
                for i in range(5):
                    img = Image.new("RGB", (200, 200), color=(i * 50, i * 40, i * 30))
                    img.save(source_path / f"test_image_{i}.jpg", format="JPEG")

                yield {
                    "source": source_path,
                    "output": output_path,
                }

    def test_data_pipeline_initialization(self, temp_dirs):
        """DataPipeline 초기화"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
        )

        assert pipeline is not None

    def test_data_pipeline_has_components(self, temp_dirs):
        """파이프라인에 필요한 컴포넌트가 있는지 확인"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
        )

        assert hasattr(pipeline, "collector")
        assert hasattr(pipeline, "preprocessors")
        assert hasattr(pipeline, "labeler")
        assert hasattr(pipeline, "storage")

    def test_data_pipeline_run_single_image(self, temp_dirs):
        """단일 이미지 처리"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
        )

        image_path = temp_dirs["source"] / "test_image_0.jpg"
        result = pipeline.process_single(image_path)

        assert result is not None
        assert result.image_id is not None
        assert result.department is not None
        assert result.tier is not None

    def test_data_pipeline_run_all_images(self, temp_dirs):
        """전체 이미지 처리"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
        )

        results = pipeline.run()

        assert len(results) == 5  # 5개 테스트 이미지
        for result in results:
            assert result.department is not None
            assert result.tier is not None

    def test_data_pipeline_saves_metadata(self, temp_dirs):
        """메타데이터 저장 확인"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
        )

        pipeline.run()

        # 메타데이터 디렉토리 확인
        metadata_dir = temp_dirs["output"] / "metadata"
        assert metadata_dir.exists()

        # 메타데이터 파일 확인
        metadata_files = list(metadata_dir.glob("*.json"))
        assert len(metadata_files) == 5

    def test_data_pipeline_with_custom_preprocessors(self, temp_dirs):
        """커스텀 전처리기 설정"""
        from data_pipeline.pipeline import DataPipeline
        from data_pipeline.preprocessors import ResizePreprocessor, NormalizePreprocessor

        preprocessors = [
            ResizePreprocessor(target_size=(128, 128)),
            NormalizePreprocessor(),
        ]

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
            preprocessors=preprocessors,
        )

        results = pipeline.run()

        assert len(results) == 5

    def test_data_pipeline_get_statistics(self, temp_dirs):
        """파이프라인 통계 조회"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
        )

        pipeline.run()
        stats = pipeline.get_statistics()

        assert "total_processed" in stats
        assert stats["total_processed"] == 5
        assert "department_distribution" in stats
        assert "tier_distribution" in stats

    def test_data_pipeline_filter_by_confidence(self, temp_dirs):
        """신뢰도로 결과 필터링"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
        )

        results = pipeline.run()
        filtered = pipeline.filter_results(results, min_confidence=0.5)

        # 필터링된 결과는 원본 이하여야 함
        assert len(filtered) <= len(results)

    def test_data_pipeline_export_report(self, temp_dirs):
        """처리 결과 리포트 내보내기"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
        )

        pipeline.run()
        report_path = pipeline.export_report()

        assert report_path.exists()


class TestDataPipelineConfiguration:
    """DataPipeline 설정 테스트"""

    @pytest.fixture
    def temp_dirs(self):
        """테스트용 임시 디렉토리들"""
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                source_path = Path(source_dir)
                output_path = Path(output_dir)

                # 테스트 이미지 생성
                img = Image.new("RGB", (200, 200), color=(100, 150, 200))
                img.save(source_path / "test.jpg", format="JPEG")

                yield {
                    "source": source_path,
                    "output": output_path,
                }

    def test_pipeline_with_confidence_threshold(self, temp_dirs):
        """신뢰도 임계값 설정"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
            confidence_threshold=0.8,
        )

        assert pipeline.confidence_threshold == 0.8

    def test_pipeline_with_target_size(self, temp_dirs):
        """타겟 이미지 크기 설정"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
            target_size=(256, 256),
        )

        assert pipeline.target_size == (256, 256)

    def test_pipeline_with_recursive_collection(self, temp_dirs):
        """재귀적 수집 설정"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=temp_dirs["source"],
            output_dir=temp_dirs["output"],
            recursive=True,
        )

        assert pipeline.recursive is True


class TestDataPipelineEndToEnd:
    """DataPipeline 엔드투엔드 테스트"""

    @pytest.fixture
    def realistic_temp_dirs(self):
        """실제 데이터 시나리오를 시뮬레이션하는 임시 디렉토리"""
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                source_path = Path(source_dir)
                output_path = Path(output_dir)

                # 다양한 크기와 색상의 이미지 생성
                sizes = [(100, 100), (200, 150), (300, 200), (400, 300), (500, 400)]
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

                for i, (size, color) in enumerate(zip(sizes, colors)):
                    img = Image.new("RGB", size, color=color)
                    img.save(source_path / f"artwork_{i}.jpg", format="JPEG")

                yield {
                    "source": source_path,
                    "output": output_path,
                }

    def test_full_pipeline_workflow(self, realistic_temp_dirs):
        """전체 파이프라인 워크플로우 테스트"""
        from data_pipeline.pipeline import DataPipeline

        # 1. 파이프라인 초기화
        pipeline = DataPipeline(
            source_dir=realistic_temp_dirs["source"],
            output_dir=realistic_temp_dirs["output"],
            target_size=(224, 224),
            confidence_threshold=0.5,
        )

        # 2. 파이프라인 실행
        results = pipeline.run()

        # 3. 결과 검증
        assert len(results) == 5

        # 4. 모든 결과에 필수 필드가 있는지 확인
        for result in results:
            assert result.image_id is not None
            assert result.department in [d for d in Department]
            assert result.tier in [t for t in Tier]
            assert 0.0 <= result.tier_score <= 100.0

        # 5. 통계 확인
        stats = pipeline.get_statistics()
        assert stats["total_processed"] == 5

        # 6. 메타데이터 저장 확인
        metadata_dir = realistic_temp_dirs["output"] / "metadata"
        assert metadata_dir.exists()
        assert len(list(metadata_dir.glob("*.json"))) == 5

    def test_pipeline_with_needs_review_filtering(self, realistic_temp_dirs):
        """검토 필요 항목 필터링 테스트"""
        from data_pipeline.pipeline import DataPipeline

        pipeline = DataPipeline(
            source_dir=realistic_temp_dirs["source"],
            output_dir=realistic_temp_dirs["output"],
        )

        results = pipeline.run()

        # 검토 필요 항목 필터링
        needs_review = pipeline.get_items_needing_review(results)

        # needs_review는 리스트여야 함
        assert isinstance(needs_review, list)
