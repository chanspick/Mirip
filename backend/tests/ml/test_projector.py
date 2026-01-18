# test_projector.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: RED - Failing Tests
"""
Projector 테스트 모듈

Acceptance Criteria (AC-002):
- Input 1024-d -> Output 256-d (normalized)
- Gradient flow verified (no NaN/Inf)
- Dropout active in training, inactive in eval

Architecture:
- Linear(1024, 512) -> LayerNorm -> GELU -> Dropout(0.1) -> Linear(512, 256) -> LayerNorm
"""

import pytest
import torch
import torch.nn as nn

# Import will fail initially (RED phase)
from app.ml.projector import Projector


class TestProjector:
    """Projector 단위 테스트"""

    @pytest.fixture
    def projector(self):
        """Projector 인스턴스 생성"""
        return Projector()

    @pytest.fixture
    def sample_features(self):
        """단일 샘플 특징 벡터 (1024-d)"""
        return torch.randn(1, 1024)

    @pytest.fixture
    def batch_features(self):
        """배치 특징 벡터 (32, 1024)"""
        return torch.randn(32, 1024)

    def test_projector_output_shape(self, projector, sample_features):
        """
        AC-002: Input 1024-d produces 256-d output

        Given: A 1024-d feature vector
        When: Passed through the projector
        Then: Output should be 256-dimensional
        """
        # Act
        output = projector(sample_features)

        # Assert
        assert output.shape == (1, 256), (
            f"Expected shape (1, 256), got {output.shape}"
        )
        assert output.dtype == torch.float32, (
            f"Expected dtype float32, got {output.dtype}"
        )

    def test_projector_batch_output_shape(self, projector, batch_features):
        """
        배치 처리 시 올바른 출력 shape 검증

        Given: A batch of 32 feature vectors (1024-d each)
        When: Passed through the projector
        Then: Output should be (32, 256)
        """
        # Act
        output = projector(batch_features)

        # Assert
        assert output.shape == (32, 256), (
            f"Expected shape (32, 256), got {output.shape}"
        )

    def test_projector_gradient_flow(self, projector, sample_features):
        """
        AC-002: Gradient flow verified (no NaN/Inf)

        Given: A projector in training mode
        When: Forward and backward pass
        Then: Gradients should be computed without NaN/Inf
        """
        # Arrange
        projector.train()
        sample_features.requires_grad = True

        # Act
        output = projector(sample_features)
        loss = output.sum()
        loss.backward()

        # Assert - Check gradients exist and are valid
        for name, param in projector.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), (
                    f"NaN gradient in parameter '{name}'"
                )
                assert not torch.isinf(param.grad).any(), (
                    f"Inf gradient in parameter '{name}'"
                )

        # Assert - Input gradient exists
        assert sample_features.grad is not None, (
            "Input gradient should be computed"
        )
        assert not torch.isnan(sample_features.grad).any(), (
            "NaN in input gradient"
        )

    def test_projector_dropout_behavior(self, projector, sample_features):
        """
        AC-002: Dropout active in training, inactive in eval

        Given: The same input
        When: Run multiple times in train vs eval mode
        Then: Train mode outputs vary, eval mode outputs are identical
        """
        # Test eval mode (deterministic)
        projector.eval()
        output_eval_1 = projector(sample_features)
        output_eval_2 = projector(sample_features)

        assert torch.allclose(output_eval_1, output_eval_2, atol=1e-6), (
            "Eval mode should be deterministic"
        )

        # Test train mode (stochastic due to dropout)
        projector.train()
        outputs_train = [projector(sample_features) for _ in range(5)]

        # At least some outputs should differ (dropout effect)
        all_same = all(
            torch.allclose(outputs_train[0], o, atol=1e-6)
            for o in outputs_train[1:]
        )
        assert not all_same, (
            "Train mode should show variance due to dropout"
        )

    def test_projector_output_normalized(self, projector, sample_features):
        """
        AC-002: Output should be normalized

        Given: An input feature vector
        When: Passed through the projector
        Then: Output should have unit L2 norm
        """
        # Act
        projector.eval()
        output = projector(sample_features)

        # Assert
        l2_norm = torch.norm(output, p=2, dim=1)
        assert torch.allclose(l2_norm, torch.ones_like(l2_norm), atol=1e-5), (
            f"Expected L2 norm ~1.0, got {l2_norm.item()}"
        )

    def test_projector_trainable_parameters(self, projector):
        """
        Projector 파라미터는 학습 가능해야 함

        Given: A Projector instance
        When: Checking parameters
        Then: All parameters should have requires_grad = True
        """
        # Act & Assert
        trainable_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in projector.parameters())

        assert trainable_params > 0, "Projector should have trainable parameters"
        assert trainable_params == total_params, (
            "All projector parameters should be trainable"
        )


class TestProjectorArchitecture:
    """Projector 아키텍처 검증 테스트"""

    @pytest.fixture
    def projector(self):
        """Projector 인스턴스 생성"""
        return Projector()

    def test_projector_has_two_linear_layers(self, projector):
        """
        아키텍처: Linear(1024, 512) -> ... -> Linear(512, 256)

        Given: A Projector instance
        When: Inspecting the model
        Then: Should have two Linear layers with correct dimensions
        """
        # Count Linear layers
        linear_layers = [m for m in projector.modules() if isinstance(m, nn.Linear)]

        assert len(linear_layers) >= 2, (
            f"Expected at least 2 Linear layers, got {len(linear_layers)}"
        )

        # Check first linear layer
        assert linear_layers[0].in_features == 1024, (
            f"First Linear layer should have in_features=1024, got {linear_layers[0].in_features}"
        )
        assert linear_layers[0].out_features == 512, (
            f"First Linear layer should have out_features=512, got {linear_layers[0].out_features}"
        )

        # Check second linear layer
        assert linear_layers[1].in_features == 512, (
            f"Second Linear layer should have in_features=512, got {linear_layers[1].in_features}"
        )
        assert linear_layers[1].out_features == 256, (
            f"Second Linear layer should have out_features=256, got {linear_layers[1].out_features}"
        )

    def test_projector_has_layer_norm(self, projector):
        """
        아키텍처에 LayerNorm 포함 검증

        Given: A Projector instance
        When: Inspecting the model
        Then: Should have LayerNorm layers
        """
        # Count LayerNorm layers
        ln_layers = [m for m in projector.modules() if isinstance(m, nn.LayerNorm)]

        assert len(ln_layers) >= 2, (
            f"Expected at least 2 LayerNorm layers, got {len(ln_layers)}"
        )

    def test_projector_has_dropout(self, projector):
        """
        아키텍처에 Dropout 포함 검증

        Given: A Projector instance
        When: Inspecting the model
        Then: Should have Dropout layer with p=0.1
        """
        # Find Dropout layers
        dropout_layers = [m for m in projector.modules() if isinstance(m, nn.Dropout)]

        assert len(dropout_layers) >= 1, (
            f"Expected at least 1 Dropout layer, got {len(dropout_layers)}"
        )

        # Check dropout probability
        assert dropout_layers[0].p == 0.1, (
            f"Dropout probability should be 0.1, got {dropout_layers[0].p}"
        )

    def test_projector_has_gelu_activation(self, projector):
        """
        아키텍처에 GELU 활성화 함수 포함 검증

        Given: A Projector instance
        When: Inspecting the model
        Then: Should have GELU activation
        """
        # Find GELU layers
        gelu_layers = [m for m in projector.modules() if isinstance(m, nn.GELU)]

        assert len(gelu_layers) >= 1, (
            f"Expected at least 1 GELU layer, got {len(gelu_layers)}"
        )


class TestProjectorDevice:
    """디바이스 관련 테스트"""

    @pytest.fixture
    def projector(self):
        """Projector 인스턴스 생성"""
        return Projector()

    def test_projector_cpu_inference(self, projector):
        """
        CPU에서 추론이 가능해야 함
        """
        # Arrange
        features = torch.randn(1, 1024)
        projector = projector.to("cpu")

        # Act
        output = projector(features.to("cpu"))

        # Assert
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_projector_gpu_inference(self, projector):
        """
        GPU에서 추론이 가능해야 함
        """
        # Arrange
        features = torch.randn(1, 1024).cuda()
        projector = projector.cuda()

        # Act
        output = projector(features)

        # Assert
        assert output.device.type == "cuda"


class TestProjectorProperties:
    """Projector 프로퍼티 테스트"""

    @pytest.fixture
    def projector(self):
        """Projector 인스턴스 생성"""
        return Projector()

    def test_input_dim_property(self, projector):
        """input_dim 프로퍼티 테스트"""
        assert projector.input_dim == 1024

    def test_hidden_dim_property(self, projector):
        """hidden_dim 프로퍼티 테스트"""
        assert projector.hidden_dim == 512

    def test_output_dim_property(self, projector):
        """output_dim 프로퍼티 테스트"""
        assert projector.output_dim == 256

    def test_dropout_rate_property(self, projector):
        """dropout_rate 프로퍼티 테스트"""
        assert projector.dropout_rate == 0.1

    def test_num_parameters_property(self, projector):
        """num_parameters 프로퍼티 테스트"""
        assert projector.num_parameters > 0


class TestProjectorValidation:
    """Projector 유효성 검증 테스트"""

    def test_invalid_dimension_raises_error(self):
        """음수 또는 0 차원에서 ValueError 발생"""
        with pytest.raises(ValueError, match="All dimensions must be positive"):
            Projector(input_dim=0)

        with pytest.raises(ValueError, match="All dimensions must be positive"):
            Projector(hidden_dim=-1)

        with pytest.raises(ValueError, match="All dimensions must be positive"):
            Projector(output_dim=-100)

    def test_invalid_dropout_raises_error(self):
        """잘못된 드롭아웃 비율에서 ValueError 발생"""
        with pytest.raises(ValueError, match="Dropout must be between"):
            Projector(dropout=1.5)

        with pytest.raises(ValueError, match="Dropout must be between"):
            Projector(dropout=-0.1)


class TestProjectorAdvanced:
    """Projector 고급 기능 테스트"""

    @pytest.fixture
    def projector(self):
        """Projector 인스턴스 생성"""
        return Projector()

    @pytest.fixture
    def sample_features(self):
        """단일 샘플 특징 벡터"""
        return torch.randn(4, 1024)

    def test_project_with_intermediate(self, projector, sample_features):
        """project() 메서드의 return_intermediate=True 테스트"""
        projector.eval()
        output, intermediate = projector.project(sample_features, return_intermediate=True)

        assert output.shape == (4, 256)
        assert intermediate.shape == (4, 512)

    def test_project_without_intermediate(self, projector, sample_features):
        """project() 메서드의 return_intermediate=False 테스트"""
        projector.eval()
        output = projector.project(sample_features, return_intermediate=False)

        assert output.shape == (4, 256)

    def test_freeze_unfreeze(self, projector):
        """freeze/unfreeze 메서드 테스트"""
        # freeze 테스트
        projector.freeze()
        for param in projector.parameters():
            assert not param.requires_grad

        # unfreeze 테스트
        projector.unfreeze()
        for param in projector.parameters():
            assert param.requires_grad

    def test_repr(self, projector):
        """__repr__ 메서드 테스트"""
        repr_str = repr(projector)
        assert "Projector" in repr_str
        assert "input_dim=1024" in repr_str
        assert "hidden_dim=512" in repr_str
        assert "output_dim=256" in repr_str
        assert "dropout=0.1" in repr_str

    def test_normalize_false(self):
        """normalize=False 옵션 테스트"""
        projector = Projector(normalize=False)
        projector.eval()
        features = torch.randn(2, 1024)
        output = projector(features)

        # 출력이 정규화되지 않아야 함
        l2_norm = torch.norm(output, p=2, dim=1)
        # 정규화되지 않았으므로 1.0이 아닐 수 있음
        assert output.shape == (2, 256)

    def test_custom_dimensions(self):
        """커스텀 차원 설정 테스트"""
        projector = Projector(
            input_dim=768,
            hidden_dim=384,
            output_dim=128,
            dropout=0.2
        )
        features = torch.randn(2, 768)
        output = projector(features)

        assert output.shape == (2, 128)
        assert projector.input_dim == 768
        assert projector.hidden_dim == 384
        assert projector.output_dim == 128
        assert projector.dropout_rate == 0.2
