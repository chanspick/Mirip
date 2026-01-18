# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: Storage 테스트

TDD RED Phase: 스토리지 시스템에 대한 테스트를 작성합니다.
BaseStorage, LocalStorage, MetadataStorage를 검증합니다.
"""

import json
import tempfile
from pathlib import Path

import pytest

from data_pipeline.models.metadata import Department, Tier, ImageMetadata


class TestBaseStorage:
    """BaseStorage 추상 클래스 테스트"""

    def test_base_storage_is_abstract(self):
        """BaseStorage는 추상 클래스여야 함"""
        from data_pipeline.storage.base_storage import BaseStorage

        with pytest.raises(TypeError):
            BaseStorage()  # 추상 클래스 직접 인스턴스화 불가

    def test_base_storage_has_save_method(self):
        """save 추상 메서드 존재 확인"""
        from data_pipeline.storage.base_storage import BaseStorage
        import inspect

        assert hasattr(BaseStorage, "save")
        assert inspect.isabstract(BaseStorage)

    def test_base_storage_has_load_method(self):
        """load 메서드 존재 확인"""
        from data_pipeline.storage.base_storage import BaseStorage

        assert hasattr(BaseStorage, "load")

    def test_base_storage_has_exists_method(self):
        """exists 메서드 존재 확인"""
        from data_pipeline.storage.base_storage import BaseStorage

        assert hasattr(BaseStorage, "exists")


class TestLocalStorage:
    """LocalStorage 테스트"""

    @pytest.fixture
    def temp_storage_dir(self):
        """테스트용 임시 스토리지 디렉토리"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_local_storage_initialization(self, temp_storage_dir):
        """LocalStorage 초기화"""
        from data_pipeline.storage.local_storage import LocalStorage

        storage = LocalStorage(base_path=temp_storage_dir)

        assert storage is not None
        assert storage.base_path == temp_storage_dir

    def test_local_storage_inherits_base(self, temp_storage_dir):
        """LocalStorage는 BaseStorage를 상속"""
        from data_pipeline.storage.base_storage import BaseStorage
        from data_pipeline.storage.local_storage import LocalStorage

        storage = LocalStorage(base_path=temp_storage_dir)

        assert isinstance(storage, BaseStorage)

    def test_local_storage_save_and_load_bytes(self, temp_storage_dir):
        """바이트 데이터 저장 및 로드"""
        from data_pipeline.storage.local_storage import LocalStorage

        storage = LocalStorage(base_path=temp_storage_dir)
        data = b"test data content"

        storage.save("test_file.bin", data)
        loaded = storage.load("test_file.bin")

        assert loaded == data

    def test_local_storage_save_and_load_text(self, temp_storage_dir):
        """텍스트 데이터 저장 및 로드"""
        from data_pipeline.storage.local_storage import LocalStorage

        storage = LocalStorage(base_path=temp_storage_dir)
        text_data = "테스트 텍스트 데이터"

        storage.save("test_file.txt", text_data)
        loaded = storage.load("test_file.txt", as_text=True)

        assert loaded == text_data

    def test_local_storage_exists(self, temp_storage_dir):
        """파일 존재 여부 확인"""
        from data_pipeline.storage.local_storage import LocalStorage

        storage = LocalStorage(base_path=temp_storage_dir)
        storage.save("existing.txt", "data")

        assert storage.exists("existing.txt") is True
        assert storage.exists("nonexistent.txt") is False

    def test_local_storage_delete(self, temp_storage_dir):
        """파일 삭제"""
        from data_pipeline.storage.local_storage import LocalStorage

        storage = LocalStorage(base_path=temp_storage_dir)
        storage.save("to_delete.txt", "data")

        assert storage.exists("to_delete.txt") is True
        storage.delete("to_delete.txt")
        assert storage.exists("to_delete.txt") is False

    def test_local_storage_list_files(self, temp_storage_dir):
        """파일 목록 조회"""
        from data_pipeline.storage.local_storage import LocalStorage

        storage = LocalStorage(base_path=temp_storage_dir)
        storage.save("file1.txt", "data1")
        storage.save("file2.txt", "data2")
        storage.save("subdir/file3.txt", "data3")

        files = storage.list_files()

        assert "file1.txt" in files
        assert "file2.txt" in files

    def test_local_storage_creates_subdirectories(self, temp_storage_dir):
        """하위 디렉토리 자동 생성"""
        from data_pipeline.storage.local_storage import LocalStorage

        storage = LocalStorage(base_path=temp_storage_dir)

        storage.save("subdir/nested/file.txt", "data")

        assert storage.exists("subdir/nested/file.txt") is True


class TestMetadataStorage:
    """MetadataStorage 테스트"""

    @pytest.fixture
    def temp_storage_dir(self):
        """테스트용 임시 스토리지 디렉토리"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_metadata(self):
        """테스트용 샘플 메타데이터"""
        return ImageMetadata(
            image_id="img_001",
            original_filename="test_image.jpg",
            file_path="/path/to/image.jpg",
            width=1920,
            height=1080,
            file_size=1024000,
            format="JPEG",
            department=Department.VISUAL_DESIGN,
            tier=Tier.A,
            tier_score=75.5,
        )

    def test_metadata_storage_initialization(self, temp_storage_dir):
        """MetadataStorage 초기화"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)

        assert storage is not None

    def test_metadata_storage_save_metadata(self, temp_storage_dir, sample_metadata):
        """메타데이터 저장"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)

        storage.save_metadata(sample_metadata)

        # 파일이 생성되었는지 확인
        assert (temp_storage_dir / "metadata" / "img_001.json").exists()

    def test_metadata_storage_load_metadata(self, temp_storage_dir, sample_metadata):
        """메타데이터 로드"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)
        storage.save_metadata(sample_metadata)

        loaded = storage.load_metadata("img_001")

        assert loaded.image_id == sample_metadata.image_id
        assert loaded.department == sample_metadata.department
        assert loaded.tier == sample_metadata.tier
        assert loaded.tier_score == sample_metadata.tier_score

    def test_metadata_storage_save_batch(self, temp_storage_dir):
        """배치 메타데이터 저장"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)
        metadatas = [
            ImageMetadata(
                image_id=f"img_{i:03d}",
                original_filename=f"image_{i}.jpg",
                file_path=f"/path/to/image_{i}.jpg",
                width=1920,
                height=1080,
                file_size=1024000,
                format="JPEG",
            )
            for i in range(5)
        ]

        storage.save_batch(metadatas)

        for i in range(5):
            assert storage.exists_metadata(f"img_{i:03d}")

    def test_metadata_storage_load_all(self, temp_storage_dir):
        """모든 메타데이터 로드"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)
        for i in range(3):
            metadata = ImageMetadata(
                image_id=f"img_{i:03d}",
                original_filename=f"image_{i}.jpg",
                file_path=f"/path/to/image_{i}.jpg",
                width=1920,
                height=1080,
                file_size=1024000,
                format="JPEG",
            )
            storage.save_metadata(metadata)

        all_metadata = storage.load_all()

        assert len(all_metadata) == 3

    def test_metadata_storage_update_metadata(self, temp_storage_dir, sample_metadata):
        """메타데이터 업데이트"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)
        storage.save_metadata(sample_metadata)

        # 업데이트
        updated_data = {"tier": Tier.S, "tier_score": 95.0}
        storage.update_metadata("img_001", updated_data)

        loaded = storage.load_metadata("img_001")
        assert loaded.tier == Tier.S
        assert loaded.tier_score == 95.0

    def test_metadata_storage_delete_metadata(self, temp_storage_dir, sample_metadata):
        """메타데이터 삭제"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)
        storage.save_metadata(sample_metadata)

        assert storage.exists_metadata("img_001") is True
        storage.delete_metadata("img_001")
        assert storage.exists_metadata("img_001") is False

    def test_metadata_storage_query_by_department(self, temp_storage_dir):
        """과별로 메타데이터 조회"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)

        # 다양한 과별 메타데이터 저장
        for i, dept in enumerate([
            Department.VISUAL_DESIGN,
            Department.VISUAL_DESIGN,
            Department.INDUSTRIAL_DESIGN,
            Department.CRAFT,
        ]):
            metadata = ImageMetadata(
                image_id=f"img_{i:03d}",
                original_filename=f"image_{i}.jpg",
                file_path=f"/path/to/image_{i}.jpg",
                width=1920,
                height=1080,
                file_size=1024000,
                format="JPEG",
                department=dept,
            )
            storage.save_metadata(metadata)

        # 시디 과별만 조회
        visual_design_items = storage.query_by_department(Department.VISUAL_DESIGN)

        assert len(visual_design_items) == 2

    def test_metadata_storage_query_by_tier(self, temp_storage_dir):
        """티어로 메타데이터 조회"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)

        # 다양한 티어 메타데이터 저장
        for i, tier in enumerate([Tier.S, Tier.A, Tier.A, Tier.B, Tier.C]):
            metadata = ImageMetadata(
                image_id=f"img_{i:03d}",
                original_filename=f"image_{i}.jpg",
                file_path=f"/path/to/image_{i}.jpg",
                width=1920,
                height=1080,
                file_size=1024000,
                format="JPEG",
                tier=tier,
            )
            storage.save_metadata(metadata)

        # A 티어만 조회
        a_tier_items = storage.query_by_tier(Tier.A)

        assert len(a_tier_items) == 2

    def test_metadata_storage_get_statistics(self, temp_storage_dir):
        """스토리지 통계 조회"""
        from data_pipeline.storage.metadata_storage import MetadataStorage

        storage = MetadataStorage(base_path=temp_storage_dir)

        # 메타데이터 저장
        for i in range(10):
            metadata = ImageMetadata(
                image_id=f"img_{i:03d}",
                original_filename=f"image_{i}.jpg",
                file_path=f"/path/to/image_{i}.jpg",
                width=1920,
                height=1080,
                file_size=1024000,
                format="JPEG",
                department=Department.VISUAL_DESIGN if i % 2 == 0 else Department.CRAFT,
                tier=Tier.S if i < 3 else Tier.A,
            )
            storage.save_metadata(metadata)

        stats = storage.get_statistics()

        assert stats["total_count"] == 10
        assert "department_distribution" in stats
        assert "tier_distribution" in stats
