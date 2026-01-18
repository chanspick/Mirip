# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: Preprocessor 테스트

TDD RED Phase: 이미지 전처리기에 대한 테스트를 작성합니다.
BasePreprocessor, ResizePreprocessor, NormalizePreprocessor, AugmentPreprocessor를 검증합니다.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


class TestBasePreprocessor:
    """BasePreprocessor 추상 클래스 테스트"""

    def test_base_preprocessor_is_abstract(self):
        """BasePreprocessor는 추상 클래스여야 함"""
        from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor

        with pytest.raises(TypeError):
            BasePreprocessor()  # 추상 클래스 직접 인스턴스화 불가

    def test_base_preprocessor_has_process_method(self):
        """process 추상 메서드 존재 확인"""
        from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor
        import inspect

        assert hasattr(BasePreprocessor, "process")
        assert inspect.isabstract(BasePreprocessor)

    def test_base_preprocessor_has_process_batch_method(self):
        """process_batch 메서드 존재 확인"""
        from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor

        assert hasattr(BasePreprocessor, "process_batch")


class TestResizePreprocessor:
    """ResizePreprocessor 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 샘플 이미지 (100x100 RGB)"""
        return Image.new("RGB", (100, 100), color=(255, 128, 64))

    @pytest.fixture
    def temp_image_file(self, sample_image):
        """테스트용 임시 이미지 파일"""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            sample_image.save(f, format="JPEG")
            return Path(f.name)

    def test_resize_preprocessor_initialization(self):
        """ResizePreprocessor 초기화"""
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        preprocessor = ResizePreprocessor(target_size=(224, 224))

        assert preprocessor.target_size == (224, 224)

    def test_resize_preprocessor_inherits_base(self):
        """ResizePreprocessor는 BasePreprocessor를 상속"""
        from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        preprocessor = ResizePreprocessor(target_size=(224, 224))

        assert isinstance(preprocessor, BasePreprocessor)

    def test_resize_preprocessor_process_image(self, sample_image):
        """이미지 리사이즈 처리"""
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        preprocessor = ResizePreprocessor(target_size=(64, 64))
        result = preprocessor.process(sample_image)

        assert result.size == (64, 64)

    def test_resize_preprocessor_process_numpy_array(self):
        """numpy 배열 리사이즈 처리"""
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        preprocessor = ResizePreprocessor(target_size=(32, 32))
        input_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocessor.process(input_array)

        # ResizePreprocessor는 PIL Image를 반환
        assert result.size == (32, 32)

    def test_resize_preprocessor_process_file_path(self, temp_image_file):
        """파일 경로로 리사이즈 처리"""
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        preprocessor = ResizePreprocessor(target_size=(50, 50))
        result = preprocessor.process(str(temp_image_file))

        assert result.size == (50, 50)

    def test_resize_preprocessor_maintain_aspect_ratio(self, sample_image):
        """종횡비 유지 리사이즈"""
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        # 100x100 이미지를 50 너비로 리사이즈 (종횡비 유지)
        preprocessor = ResizePreprocessor(
            target_size=(50, 50), maintain_aspect_ratio=True
        )
        result = preprocessor.process(sample_image)

        # 종횡비 유지 시 50x50 이하 크기
        assert result.size[0] <= 50
        assert result.size[1] <= 50

    def test_resize_preprocessor_different_interpolation(self, sample_image):
        """다른 보간법 사용"""
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        preprocessor = ResizePreprocessor(
            target_size=(64, 64), interpolation="BILINEAR"
        )
        result = preprocessor.process(sample_image)

        assert result.size == (64, 64)

    def test_resize_preprocessor_process_batch(self):
        """배치 이미지 리사이즈"""
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor

        preprocessor = ResizePreprocessor(target_size=(32, 32))
        images = [Image.new("RGB", (100, 100)) for _ in range(5)]
        results = preprocessor.process_batch(images)

        assert len(results) == 5
        assert all(img.size == (32, 32) for img in results)


class TestNormalizePreprocessor:
    """NormalizePreprocessor 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 샘플 이미지"""
        return Image.new("RGB", (50, 50), color=(100, 150, 200))

    def test_normalize_preprocessor_initialization(self):
        """NormalizePreprocessor 초기화"""
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        preprocessor = NormalizePreprocessor()

        assert preprocessor is not None

    def test_normalize_preprocessor_inherits_base(self):
        """NormalizePreprocessor는 BasePreprocessor를 상속"""
        from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        preprocessor = NormalizePreprocessor()

        assert isinstance(preprocessor, BasePreprocessor)

    def test_normalize_preprocessor_default_range(self, sample_image):
        """기본 정규화 범위 (0-1)"""
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        preprocessor = NormalizePreprocessor()
        result = preprocessor.process(sample_image)

        # numpy 배열 반환, 값 범위 0-1
        assert isinstance(result, np.ndarray)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_preprocessor_custom_range(self, sample_image):
        """사용자 정의 정규화 범위 (-1 to 1)"""
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        preprocessor = NormalizePreprocessor(output_range=(-1.0, 1.0))
        result = preprocessor.process(sample_image)

        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_normalize_preprocessor_with_mean_std(self, sample_image):
        """평균/표준편차 정규화 (ImageNet 스타일)"""
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        # ImageNet 평균/표준편차
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        preprocessor = NormalizePreprocessor(mean=mean, std=std)
        result = preprocessor.process(sample_image)

        assert isinstance(result, np.ndarray)
        assert result.shape[-1] == 3  # RGB 채널

    def test_normalize_preprocessor_process_numpy_array(self):
        """numpy 배열 정규화"""
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        preprocessor = NormalizePreprocessor()
        input_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = preprocessor.process(input_array)

        assert result.dtype == np.float32 or result.dtype == np.float64
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_preprocessor_process_batch(self):
        """배치 이미지 정규화"""
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        preprocessor = NormalizePreprocessor()
        images = [Image.new("RGB", (50, 50)) for _ in range(3)]
        results = preprocessor.process_batch(images)

        assert len(results) == 3
        assert all(isinstance(r, np.ndarray) for r in results)


class TestAugmentPreprocessor:
    """AugmentPreprocessor 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 샘플 이미지"""
        # 비대칭 이미지 (변환 확인용)
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        # 왼쪽 상단에 빨간색 사각형
        for x in range(50):
            for y in range(50):
                img.putpixel((x, y), (255, 0, 0))
        return img

    def test_augment_preprocessor_initialization(self):
        """AugmentPreprocessor 초기화"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor()

        assert preprocessor is not None

    def test_augment_preprocessor_inherits_base(self):
        """AugmentPreprocessor는 BasePreprocessor를 상속"""
        from data_pipeline.preprocessors.base_preprocessor import BasePreprocessor
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor()

        assert isinstance(preprocessor, BasePreprocessor)

    def test_augment_preprocessor_horizontal_flip(self, sample_image):
        """수평 뒤집기"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor(
            horizontal_flip=True, horizontal_flip_prob=1.0
        )
        result = preprocessor.process(sample_image)

        assert result.size == sample_image.size
        # 수평 뒤집기 후 오른쪽 상단이 빨간색이어야 함
        assert result.getpixel((99, 0)) == (255, 0, 0)

    def test_augment_preprocessor_vertical_flip(self, sample_image):
        """수직 뒤집기"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor(vertical_flip=True, vertical_flip_prob=1.0)
        result = preprocessor.process(sample_image)

        assert result.size == sample_image.size
        # 수직 뒤집기 후 왼쪽 하단이 빨간색이어야 함
        assert result.getpixel((0, 99)) == (255, 0, 0)

    def test_augment_preprocessor_rotation(self, sample_image):
        """회전"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor(rotation_range=(-90, 90))
        result = preprocessor.process(sample_image)

        # 회전 후에도 이미지 반환
        assert isinstance(result, Image.Image)

    def test_augment_preprocessor_brightness(self, sample_image):
        """밝기 조절"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor(brightness_range=(0.5, 1.5))
        result = preprocessor.process(sample_image)

        assert isinstance(result, Image.Image)

    def test_augment_preprocessor_contrast(self, sample_image):
        """대비 조절"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor(contrast_range=(0.5, 1.5))
        result = preprocessor.process(sample_image)

        assert isinstance(result, Image.Image)

    def test_augment_preprocessor_no_augmentation(self, sample_image):
        """증강 없음 (기본값)"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor()
        result = preprocessor.process(sample_image)

        # 변환 없이 동일한 이미지 반환
        assert result.size == sample_image.size

    def test_augment_preprocessor_random_seed(self, sample_image):
        """랜덤 시드 설정으로 재현 가능"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor1 = AugmentPreprocessor(
            rotation_range=(-45, 45), random_seed=42
        )
        preprocessor2 = AugmentPreprocessor(
            rotation_range=(-45, 45), random_seed=42
        )

        result1 = preprocessor1.process(sample_image.copy())
        result2 = preprocessor2.process(sample_image.copy())

        # 동일한 시드로 동일한 결과
        assert list(result1.getdata()) == list(result2.getdata())

    def test_augment_preprocessor_process_batch(self):
        """배치 이미지 증강"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor(horizontal_flip=True)
        images = [Image.new("RGB", (50, 50)) for _ in range(4)]
        results = preprocessor.process_batch(images)

        assert len(results) == 4
        assert all(isinstance(img, Image.Image) for img in results)

    def test_augment_preprocessor_combined_augmentations(self, sample_image):
        """복합 증강 적용"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )

        preprocessor = AugmentPreprocessor(
            horizontal_flip=True,
            horizontal_flip_prob=0.5,
            rotation_range=(-30, 30),
            brightness_range=(0.8, 1.2),
            contrast_range=(0.8, 1.2),
        )
        result = preprocessor.process(sample_image)

        assert isinstance(result, Image.Image)


class TestPreprocessorPipeline:
    """Preprocessor 파이프라인 테스트"""

    @pytest.fixture
    def sample_image(self):
        """테스트용 샘플 이미지"""
        return Image.new("RGB", (200, 200), color=(128, 128, 128))

    def test_pipeline_resize_then_normalize(self, sample_image):
        """리사이즈 후 정규화 파이프라인"""
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        resize = ResizePreprocessor(target_size=(64, 64))
        normalize = NormalizePreprocessor()

        resized = resize.process(sample_image)
        result = normalize.process(resized)

        assert isinstance(result, np.ndarray)
        assert result.shape[:2] == (64, 64)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_pipeline_augment_resize_normalize(self, sample_image):
        """증강 -> 리사이즈 -> 정규화 파이프라인"""
        from data_pipeline.preprocessors.augment_preprocessor import (
            AugmentPreprocessor,
        )
        from data_pipeline.preprocessors.resize_preprocessor import ResizePreprocessor
        from data_pipeline.preprocessors.normalize_preprocessor import (
            NormalizePreprocessor,
        )

        augment = AugmentPreprocessor(horizontal_flip=True, horizontal_flip_prob=0.5)
        resize = ResizePreprocessor(target_size=(128, 128))
        normalize = NormalizePreprocessor()

        augmented = augment.process(sample_image)
        resized = resize.process(augmented)
        result = normalize.process(resized)

        assert isinstance(result, np.ndarray)
        assert result.shape[:2] == (128, 128)
