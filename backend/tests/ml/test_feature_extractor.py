# test_feature_extractor.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: RED - Failing Tests
"""
DINOv2FeatureExtractor 테스트 모듈

Acceptance Criteria (AC-001):
- Single image: 1024-d feature vector, float32, < 100ms inference
- Batch (32 images): (32, 1024) tensor, < 12GB VRAM
- All parameters requires_grad = False
"""

import pytest
import torch

# Import will fail initially (RED phase)
from app.ml.feature_extractor import DINOv2FeatureExtractor


class TestDINOv2FeatureExtractor:
    """DINOv2FeatureExtractor 단위 테스트"""

    @pytest.fixture
    def extractor(self):
        """Feature extractor 인스턴스 생성"""
        return DINOv2FeatureExtractor()

    @pytest.fixture
    def sample_image(self):
        """단일 테스트 이미지 생성 (768x768 RGB)"""
        # (batch=1, channels=3, height=768, width=768)
        return torch.randn(1, 3, 768, 768)

    @pytest.fixture
    def batch_images(self):
        """배치 테스트 이미지 생성 (32개, 768x768 RGB)"""
        # (batch=32, channels=3, height=768, width=768)
        return torch.randn(32, 3, 768, 768)

    def test_feature_extractor_output_shape(self, extractor, sample_image):
        """
        AC-001: Single image produces 1024-d feature vector

        Given: A single 768x768 RGB image
        When: Passed through the feature extractor
        Then: Output should be a 1024-dimensional feature vector
        """
        # Act
        features = extractor(sample_image)

        # Assert
        assert features.shape == (1, 1024), (
            f"Expected shape (1, 1024), got {features.shape}"
        )
        assert features.dtype == torch.float32, (
            f"Expected dtype float32, got {features.dtype}"
        )

    def test_feature_extractor_batch_processing(self, extractor, batch_images):
        """
        AC-001: Batch of 32 images produces (32, 1024) tensor

        Given: A batch of 32 images (768x768 RGB)
        When: Passed through the feature extractor
        Then: Output should be (32, 1024) tensor
        """
        # Act
        features = extractor(batch_images)

        # Assert
        assert features.shape == (32, 1024), (
            f"Expected shape (32, 1024), got {features.shape}"
        )

    def test_feature_extractor_model_frozen(self, extractor):
        """
        AC-001: All backbone parameters should be frozen

        Given: A DINOv2FeatureExtractor instance
        When: Checking all parameters
        Then: All parameters should have requires_grad = False
        """
        # Act & Assert
        for name, param in extractor.named_parameters():
            assert not param.requires_grad, (
                f"Parameter '{name}' should be frozen (requires_grad=False)"
            )

    def test_feature_extractor_output_normalized(self, extractor, sample_image):
        """
        Feature vectors should be L2-normalized

        Given: A single image
        When: Passed through the feature extractor
        Then: Output should have unit L2 norm (approximately 1.0)
        """
        # Act
        features = extractor(sample_image)

        # Assert
        l2_norm = torch.norm(features, p=2, dim=1)
        assert torch.allclose(l2_norm, torch.ones_like(l2_norm), atol=1e-5), (
            f"Expected L2 norm ~1.0, got {l2_norm.item()}"
        )

    def test_feature_extractor_deterministic(self, extractor, sample_image):
        """
        Feature extraction should be deterministic (eval mode)

        Given: The same image
        When: Passed through the extractor twice
        Then: Results should be identical
        """
        # Act
        features1 = extractor(sample_image)
        features2 = extractor(sample_image)

        # Assert
        assert torch.allclose(features1, features2, atol=1e-6), (
            "Feature extraction should be deterministic"
        )

    def test_feature_extractor_different_input_sizes(self, extractor):
        """
        Extractor should handle different input sizes

        Given: Images of different sizes (224, 384, 518, 768)
        When: Passed through the extractor
        Then: All should produce 1024-d features
        """
        sizes = [224, 384, 518, 768]

        for size in sizes:
            # Act
            image = torch.randn(1, 3, size, size)
            features = extractor(image)

            # Assert
            assert features.shape == (1, 1024), (
                f"Size {size}: Expected (1, 1024), got {features.shape}"
            )


class TestDINOv2FeatureExtractorDevice:
    """디바이스 관련 테스트"""

    @pytest.fixture
    def extractor(self):
        """Feature extractor 인스턴스 생성"""
        return DINOv2FeatureExtractor()

    def test_feature_extractor_cpu_inference(self, extractor):
        """
        CPU에서 추론이 가능해야 함

        Given: CPU device
        When: Running inference
        Then: Should complete without errors
        """
        # Arrange
        image = torch.randn(1, 3, 768, 768)
        extractor = extractor.to("cpu")

        # Act
        features = extractor(image.to("cpu"))

        # Assert
        assert features.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_feature_extractor_gpu_inference(self, extractor):
        """
        GPU에서 추론이 가능해야 함

        Given: CUDA device
        When: Running inference
        Then: Should complete without errors and return CUDA tensor
        """
        # Arrange
        image = torch.randn(1, 3, 768, 768).cuda()
        extractor = extractor.cuda()

        # Act
        features = extractor(image)

        # Assert
        assert features.device.type == "cuda"


class TestDINOv2FeatureExtractorProperties:
    """DINOv2FeatureExtractor 프로퍼티 테스트"""

    @pytest.fixture
    def extractor(self):
        """Feature extractor 인스턴스 생성"""
        return DINOv2FeatureExtractor()

    def test_output_dim_property(self, extractor):
        """output_dim 프로퍼티 테스트"""
        assert extractor.output_dim == 1024

    def test_model_name_property(self, extractor):
        """model_name 프로퍼티 테스트"""
        assert extractor.model_name == "facebook/dinov2-large"

    def test_is_frozen_property(self, extractor):
        """is_frozen 프로퍼티 테스트"""
        assert extractor.is_frozen is True

    def test_repr(self, extractor):
        """__repr__ 메서드 테스트"""
        repr_str = repr(extractor)
        assert "DINOv2FeatureExtractor" in repr_str
        assert "facebook/dinov2-large" in repr_str
        assert "output_dim=1024" in repr_str
        assert "frozen=True" in repr_str


class TestDINOv2FeatureExtractorAdvanced:
    """DINOv2FeatureExtractor 고급 기능 테스트"""

    @pytest.fixture
    def extractor(self):
        """Feature extractor 인스턴스 생성"""
        return DINOv2FeatureExtractor()

    @pytest.fixture
    def sample_image(self):
        """테스트 이미지"""
        return torch.randn(2, 3, 224, 224)

    def test_extract_features_without_all_tokens(self, extractor, sample_image):
        """extract_features() 메서드의 return_all_tokens=False 테스트"""
        features = extractor.extract_features(sample_image, return_all_tokens=False)

        assert features.shape == (2, 1024)

    def test_extract_features_with_all_tokens(self, extractor, sample_image):
        """extract_features() 메서드의 return_all_tokens=True 테스트"""
        cls_features, patch_features = extractor.extract_features(
            sample_image, return_all_tokens=True
        )

        assert cls_features.shape == (2, 1024)
        # 패치 토큰 shape: (batch, num_patches, hidden_dim)
        assert patch_features.shape[0] == 2
        assert patch_features.shape[2] == 1024

    def test_normalize_false(self):
        """normalize=False 옵션 테스트"""
        extractor = DINOv2FeatureExtractor(normalize=False)
        image = torch.randn(1, 3, 224, 224)
        features = extractor(image)

        # 출력이 정규화되지 않아야 함
        assert features.shape == (1, 1024)


class TestDINOv2FeatureExtractorModels:
    """다양한 DINOv2 모델 테스트"""

    def test_supported_model_output_dims(self):
        """지원 모델의 출력 차원 확인"""
        expected_dims = {
            "facebook/dinov2-small": 384,
            "facebook/dinov2-base": 768,
            "facebook/dinov2-large": 1024,
            "facebook/dinov2-giant": 1536,
        }

        # 클래스 속성 확인
        assert DINOv2FeatureExtractor.SUPPORTED_MODELS == expected_dims

    def test_default_model_name(self):
        """기본 모델 이름 확인"""
        assert DINOv2FeatureExtractor.DEFAULT_MODEL_NAME == "facebook/dinov2-large"

    def test_default_output_dim(self):
        """기본 출력 차원 확인"""
        assert DINOv2FeatureExtractor.OUTPUT_DIM == 1024
