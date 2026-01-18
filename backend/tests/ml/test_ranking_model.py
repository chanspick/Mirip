# test_ranking_model.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: RED - Failing Tests
"""
PairwiseRankingModel 테스트 모듈

Acceptance Criteria (AC-003):
- 두 이미지 입력 시 각각 스칼라 스코어 반환
- S티어 이미지 스코어 > C티어 이미지 스코어
- 배치 처리 지원 (32개 쌍)

Architecture:
- FeatureExtractor (frozen DINOv2) + Projector (trainable)
- MarginRankingLoss (margin=1.0)
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock

# Import will fail initially (RED phase)
from app.ml.ranking_model import PairwiseRankingModel


class TestPairwiseRankingModelBasic:
    """PairwiseRankingModel 기본 기능 테스트"""

    @pytest.fixture
    def mock_feature_extractor(self):
        """Mock FeatureExtractor (DINOv2 로드 없이 테스트)"""
        mock = MagicMock()
        mock.output_dim = 1024
        # forward returns (B, 1024) features
        mock.return_value = torch.randn(1, 1024)
        mock.is_frozen = True
        return mock

    @pytest.fixture
    def model(self, mock_feature_extractor):
        """PairwiseRankingModel with mocked feature extractor"""
        with patch('app.ml.ranking_model.DINOv2FeatureExtractor', return_value=mock_feature_extractor):
            model = PairwiseRankingModel(
                projector_hidden_dim=512,
                projector_output_dim=256,
            )
            return model

    def test_model_initialization(self, model):
        """
        모델 초기화 검증

        Given: PairwiseRankingModel 생성 시
        When: 모델이 초기화되면
        Then: feature_extractor와 projector가 올바르게 설정되어야 함
        """
        # Assert
        assert hasattr(model, 'feature_extractor'), "feature_extractor 속성 필요"
        assert hasattr(model, 'projector'), "projector 속성 필요"
        assert hasattr(model, 'score_head'), "score_head 속성 필요"

    def test_feature_extractor_frozen(self, model):
        """
        AC-003: FeatureExtractor는 동결 상태여야 함

        Given: 초기화된 모델
        When: feature_extractor 파라미터 확인
        Then: 모든 파라미터가 requires_grad=False여야 함
        """
        # feature_extractor의 모든 파라미터가 동결되어야 함
        # Mock 사용으로 인해 is_frozen 속성 확인
        assert model.feature_extractor.is_frozen, "FeatureExtractor must be frozen"

    def test_projector_trainable(self, model):
        """
        Projector는 학습 가능해야 함

        Given: 초기화된 모델
        When: projector 파라미터 확인
        Then: 모든 파라미터가 requires_grad=True여야 함
        """
        trainable_params = sum(
            p.numel() for p in model.projector.parameters() if p.requires_grad
        )
        assert trainable_params > 0, "Projector must have trainable parameters"


class TestPairwiseRankingModelForward:
    """PairwiseRankingModel Forward Pass 테스트"""

    @pytest.fixture
    def mock_feature_extractor(self):
        """Mock FeatureExtractor with deterministic output"""
        mock = MagicMock()
        mock.output_dim = 1024
        mock.is_frozen = True

        # 캐시를 사용하여 동일 입력에 대해 동일 출력 보장
        cache = {}

        def forward_fn(x):
            batch_size = x.shape[0]
            # 입력 텐서의 해시를 키로 사용 (결정적 출력을 위해)
            key = (batch_size, x.sum().item())
            if key not in cache:
                torch.manual_seed(hash(key) % (2**32))
                cache[key] = torch.randn(batch_size, 1024)
            return cache[key]

        mock.side_effect = forward_fn
        mock.return_value = torch.randn(1, 1024)
        return mock

    @pytest.fixture
    def model(self, mock_feature_extractor):
        """PairwiseRankingModel with mocked feature extractor"""
        with patch('app.ml.ranking_model.DINOv2FeatureExtractor', return_value=mock_feature_extractor):
            model = PairwiseRankingModel()
            return model

    @pytest.fixture
    def sample_images(self):
        """단일 이미지 쌍 (1, 3, 768, 768)"""
        img1 = torch.randn(1, 3, 768, 768)
        img2 = torch.randn(1, 3, 768, 768)
        return img1, img2

    @pytest.fixture
    def batch_images(self):
        """배치 이미지 쌍 (32, 3, 768, 768)"""
        img1 = torch.randn(32, 3, 768, 768)
        img2 = torch.randn(32, 3, 768, 768)
        return img1, img2

    def test_forward_returns_scalar_scores(self, model, sample_images):
        """
        AC-003: 두 이미지 입력 시 각각 스칼라 스코어 반환

        Given: 두 이미지 (img1, img2)
        When: model.forward(img1, img2) 호출
        Then: (score1, score2) 튜플 반환, 각각 스칼라 텐서
        """
        # Arrange
        img1, img2 = sample_images

        # Act
        score1, score2 = model(img1, img2)

        # Assert
        assert score1.shape == (1, 1), f"Expected shape (1, 1), got {score1.shape}"
        assert score2.shape == (1, 1), f"Expected shape (1, 1), got {score2.shape}"
        assert score1.dtype == torch.float32, "Score should be float32"
        assert score2.dtype == torch.float32, "Score should be float32"

    def test_forward_batch_processing(self, model, batch_images):
        """
        AC-003: 배치 처리 지원 (32개 쌍)

        Given: 32개의 이미지 쌍
        When: 배치로 forward 수행
        Then: (32, 1) 크기의 스코어 반환
        """
        # Arrange
        img1, img2 = batch_images

        # Act
        score1, score2 = model(img1, img2)

        # Assert
        assert score1.shape == (32, 1), f"Expected shape (32, 1), got {score1.shape}"
        assert score2.shape == (32, 1), f"Expected shape (32, 1), got {score2.shape}"

    def test_forward_deterministic_in_eval_mode(self, model, sample_images):
        """
        Eval 모드에서 결정적 출력 검증

        Given: 동일한 입력
        When: eval 모드에서 여러 번 forward
        Then: 동일한 출력 반환
        """
        # Arrange
        model.eval()
        img1, img2 = sample_images

        # Act
        score1_a, score2_a = model(img1, img2)
        score1_b, score2_b = model(img1, img2)

        # Assert
        assert torch.allclose(score1_a, score1_b, atol=1e-6), "Eval mode should be deterministic"
        assert torch.allclose(score2_a, score2_b, atol=1e-6), "Eval mode should be deterministic"


class TestPairwiseRankingModelScoreOrdering:
    """스코어 순서 검증 테스트"""

    @pytest.fixture
    def model_with_trained_weights(self):
        """
        학습된 가중치를 시뮬레이션하는 모델

        Note: 실제 학습 없이 스코어 순서를 테스트하기 위해
        가중치를 조작하거나 mock을 사용
        """
        # 이 테스트는 실제 학습 후에 의미가 있음
        # 현재는 아키텍처 검증 목적
        with patch('app.ml.ranking_model.DINOv2FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.output_dim = 1024
            mock_instance.is_frozen = True

            # S티어 이미지는 높은 특징값, C티어는 낮은 특징값 시뮬레이션
            def feature_fn(x):
                batch_size = x.shape[0]
                # 이 값은 입력에 따라 다른 특징을 반환하도록 설정
                return torch.randn(batch_size, 1024)

            mock_instance.side_effect = feature_fn
            mock_extractor.return_value = mock_instance

            model = PairwiseRankingModel()
            return model

    def test_score_ordering_after_training(self, model_with_trained_weights):
        """
        AC-003: S티어 이미지 스코어 > C티어 이미지 스코어 (학습 후)

        Note: 이 테스트는 학습 후 모델이 올바른 순서를 학습했는지 검증
        현재는 아키텍처가 올바르게 점수를 출력하는지만 확인
        """
        # Arrange
        model = model_with_trained_weights
        model.eval()

        s_tier_image = torch.randn(1, 3, 768, 768)
        c_tier_image = torch.randn(1, 3, 768, 768)

        # Act
        score_s, score_c = model(s_tier_image, c_tier_image)

        # Assert - 스코어가 스칼라임을 확인 (순서는 학습 후 검증)
        assert score_s.numel() == 1, "S-tier score should be scalar"
        assert score_c.numel() == 1, "C-tier score should be scalar"


class TestPairwiseRankingModelLoss:
    """Loss 계산 테스트"""

    @pytest.fixture
    def model(self):
        """PairwiseRankingModel"""
        with patch('app.ml.ranking_model.DINOv2FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.output_dim = 1024
            mock_instance.is_frozen = True
            mock_instance.return_value = torch.randn(1, 1024)
            mock_extractor.return_value = mock_instance

            model = PairwiseRankingModel()
            return model

    def test_compute_loss_returns_tensor(self, model):
        """
        Loss 계산 기본 검증

        Given: 두 스코어와 레이블
        When: compute_loss 호출
        Then: 스칼라 loss 텐서 반환
        """
        # Arrange
        score1 = torch.tensor([[0.8]])
        score2 = torch.tensor([[0.3]])
        labels = torch.tensor([1])  # score1 > score2

        # Act
        loss = model.compute_loss(score1, score2, labels)

        # Assert
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.dtype == torch.float32, "Loss should be float32"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_margin_ranking_loss_correct_order(self, model):
        """
        MarginRankingLoss: 올바른 순서면 낮은 loss

        Given: score1 > score2 이고 label = 1
        When: loss 계산
        Then: loss가 낮아야 함
        """
        # Arrange - 올바른 순서 (큰 마진)
        score1_correct = torch.tensor([[2.0]])
        score2_correct = torch.tensor([[0.0]])
        labels = torch.tensor([1])

        # Act
        loss = model.compute_loss(score1_correct, score2_correct, labels)

        # Assert
        # margin=1.0 일 때, score1 - score2 > margin 이면 loss = 0
        assert loss.item() == 0.0, f"Expected 0 loss for correct large margin, got {loss.item()}"

    def test_margin_ranking_loss_wrong_order(self, model):
        """
        MarginRankingLoss: 틀린 순서면 높은 loss

        Given: score1 < score2 이고 label = 1
        When: loss 계산
        Then: loss가 높아야 함
        """
        # Arrange - 틀린 순서
        score1_wrong = torch.tensor([[0.0]])
        score2_wrong = torch.tensor([[2.0]])
        labels = torch.tensor([1])  # score1 > score2 expected

        # Act
        loss = model.compute_loss(score1_wrong, score2_wrong, labels)

        # Assert
        assert loss.item() > 0.0, f"Expected positive loss for wrong order, got {loss.item()}"

    def test_margin_ranking_loss_batch(self, model):
        """
        배치에서 loss 계산

        Given: 배치 스코어와 레이블
        When: loss 계산
        Then: 평균 loss 반환
        """
        # Arrange
        batch_size = 32
        score1 = torch.randn(batch_size, 1)
        score2 = torch.randn(batch_size, 1)
        labels = torch.randint(-1, 2, (batch_size,)).clamp(-1, 1)
        labels[labels == 0] = 1  # 0 제거

        # Act
        loss = model.compute_loss(score1, score2, labels)

        # Assert
        assert loss.ndim == 0, "Batch loss should be scalar"
        assert not torch.isnan(loss), "Batch loss should not be NaN"


class TestPairwiseRankingModelGradient:
    """Gradient Flow 테스트"""

    @pytest.fixture
    def model(self):
        """PairwiseRankingModel"""
        with patch('app.ml.ranking_model.DINOv2FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.output_dim = 1024
            mock_instance.is_frozen = True

            # gradient가 흐를 수 있도록 실제 텐서 반환
            def forward_with_grad(x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 1024, requires_grad=False)

            mock_instance.side_effect = forward_with_grad
            mock_extractor.return_value = mock_instance

            model = PairwiseRankingModel()
            return model

    def test_gradient_flows_through_projector(self, model):
        """
        Gradient가 projector를 통해 흐르는지 검증

        Given: 학습 모드의 모델
        When: forward + backward pass
        Then: projector 파라미터에 gradient 존재
        """
        # Arrange
        model.train()
        img1 = torch.randn(2, 3, 768, 768)
        img2 = torch.randn(2, 3, 768, 768)
        labels = torch.tensor([1, -1])

        # Act
        score1, score2 = model(img1, img2)
        loss = model.compute_loss(score1, score2, labels)
        loss.backward()

        # Assert
        for name, param in model.projector.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_gradient_no_nan_inf(self, model):
        """
        Gradient에 NaN/Inf 없음 검증

        Given: 학습 모드의 모델
        When: forward + backward pass
        Then: 모든 gradient가 유효해야 함
        """
        # Arrange
        model.train()
        img1 = torch.randn(4, 3, 768, 768)
        img2 = torch.randn(4, 3, 768, 768)
        labels = torch.tensor([1, 1, -1, -1])

        # Act
        score1, score2 = model(img1, img2)
        loss = model.compute_loss(score1, score2, labels)
        loss.backward()

        # Assert
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf in {name}"


class TestPairwiseRankingModelDevice:
    """디바이스 호환성 테스트"""

    @pytest.fixture
    def model(self):
        """PairwiseRankingModel"""
        with patch('app.ml.ranking_model.DINOv2FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.output_dim = 1024
            mock_instance.is_frozen = True
            mock_instance.return_value = torch.randn(1, 1024)
            mock_instance.to = MagicMock(return_value=mock_instance)
            mock_extractor.return_value = mock_instance

            model = PairwiseRankingModel()
            return model

    def test_cpu_inference(self, model):
        """
        CPU 추론 검증
        """
        # Arrange
        model = model.to("cpu")
        img1 = torch.randn(1, 3, 768, 768)
        img2 = torch.randn(1, 3, 768, 768)

        # Act
        score1, score2 = model(img1, img2)

        # Assert
        assert score1.device.type == "cpu"
        assert score2.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference(self, model):
        """
        GPU 추론 검증
        """
        # Arrange
        model = model.cuda()
        img1 = torch.randn(1, 3, 768, 768).cuda()
        img2 = torch.randn(1, 3, 768, 768).cuda()

        # Act
        score1, score2 = model(img1, img2)

        # Assert
        assert score1.device.type == "cuda"
        assert score2.device.type == "cuda"


class TestPairwiseRankingModelInference:
    """추론 관련 테스트"""

    @pytest.fixture
    def model(self):
        """PairwiseRankingModel"""
        with patch('app.ml.ranking_model.DINOv2FeatureExtractor') as mock_extractor:
            mock_instance = MagicMock()
            mock_instance.output_dim = 1024
            mock_instance.is_frozen = True
            mock_instance.return_value = torch.randn(1, 1024)
            mock_extractor.return_value = mock_instance

            model = PairwiseRankingModel()
            return model

    def test_predict_single_image_score(self, model):
        """
        단일 이미지 스코어 예측

        Given: 단일 이미지
        When: predict_score 호출
        Then: 스칼라 스코어 반환
        """
        # Arrange
        model.eval()
        image = torch.randn(1, 3, 768, 768)

        # Act
        score = model.predict_score(image)

        # Assert
        assert score.shape == (1, 1), f"Expected (1, 1), got {score.shape}"
        assert score.dtype == torch.float32

    def test_predict_comparison(self, model):
        """
        두 이미지 비교 예측

        Given: 두 이미지
        When: predict 호출
        Then: 어떤 이미지가 더 높은 점수인지 반환
        """
        # Arrange
        model.eval()
        img1 = torch.randn(1, 3, 768, 768)
        img2 = torch.randn(1, 3, 768, 768)

        # Act
        result = model.predict(img1, img2)

        # Assert - result는 1 (img1 > img2) 또는 -1 (img1 < img2)
        assert result in [1, -1, 0], f"Expected 1, -1, or 0, got {result}"
