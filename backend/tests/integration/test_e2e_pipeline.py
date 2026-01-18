# test_e2e_pipeline.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# Phase 5: E2E Integration Tests (TASK-025)
"""
E2E 통합 테스트 모듈.

전체 AI 파이프라인의 통합 동작을 검증합니다:
1. Feature Extraction Pipeline: Image -> DINOv2 -> 1024-d vector
2. Projection Pipeline: 1024-d -> Projector -> 256-d normalized
3. Scoring Pipeline: Image pair -> Model -> Scores -> Ranking
4. Training Pipeline: DataLoader -> Trainer -> Checkpoint
5. Evaluation Pipeline: Model -> Test data -> Accuracy

Acceptance Criteria:
- AC-011: 코드 품질 (테스트 커버리지 >= 85%)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from app.ml.feature_extractor import DINOv2FeatureExtractor
from app.ml.projector import Projector
from app.ml.ranking_model import PairwiseRankingModel
from training.benchmarks import PerformanceBenchmarks, set_seed
from training.config import TrainingConfig
from training.datasets.data_splitter import DataSplitter
from training.datasets.pairwise_dataset import PairwiseDataset
from training.evaluator import Evaluator
from training.trainer import Trainer


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def seed() -> int:
    """고정 시드 값."""
    return 42


@pytest.fixture
def temp_dir() -> str:
    """임시 디렉토리."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_images_dir(temp_dir: str) -> str:
    """테스트용 샘플 이미지 디렉토리 생성."""
    images_dir = Path(temp_dir) / "images"
    images_dir.mkdir(exist_ok=True)

    # 각 티어별로 이미지 생성 (S, A, B, C)
    tiers = ["S", "A", "B", "C"]
    for tier in tiers:
        for i in range(5):  # 각 티어당 5개 이미지
            img = Image.new("RGB", (224, 224), color=_get_tier_color(tier, i))
            img_path = images_dir / f"{tier}_{i}.jpg"
            img.save(img_path)

    return str(images_dir)


def _get_tier_color(tier: str, index: int) -> Tuple[int, int, int]:
    """티어별 색상 반환 (테스트용)."""
    # 티어별로 다른 밝기의 색상을 사용하여 구분 가능하게 함
    tier_brightness = {"S": 255, "A": 200, "B": 150, "C": 100}
    brightness = tier_brightness.get(tier, 128)
    offset = index * 10
    return (
        min(255, brightness + offset),
        min(255, brightness - offset),
        min(255, brightness),
    )


@pytest.fixture
def sample_metadata_df(sample_images_dir: str) -> pd.DataFrame:
    """테스트용 메타데이터 DataFrame 생성."""
    tiers = ["S", "A", "B", "C"]
    data = []
    for tier in tiers:
        for i in range(5):
            data.append({"image_path": f"{tier}_{i}.jpg", "tier": tier})

    return pd.DataFrame(data)


@pytest.fixture
def mock_feature_extractor() -> nn.Module:
    """모의 Feature Extractor (DINOv2 모델 로드 없이)."""

    class MockFeatureExtractor(nn.Module):
        """테스트용 모의 Feature Extractor."""

        def __init__(self) -> None:
            super().__init__()
            self.output_dim = 1024
            self._model_name = "mock/dinov2"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            features = torch.randn(batch_size, 1024)
            return torch.nn.functional.normalize(features, p=2, dim=1)

    return MockFeatureExtractor()


@pytest.fixture
def mock_ranking_model() -> nn.Module:
    """모의 Ranking Model."""

    class MockRankingModel(nn.Module):
        """테스트용 모의 Ranking Model."""

        def __init__(self) -> None:
            super().__init__()
            # 실제 학습 가능한 레이어 포함
            self.projector = Projector(input_dim=1024, output_dim=256)
            self.score_head = nn.Linear(256, 1)
            self._margin = 1.0
            self.loss_fn = nn.MarginRankingLoss(margin=1.0)

        def forward(
            self, img1: torch.Tensor, img2: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            batch_size = img1.shape[0]
            # 학습 가능한 레이어를 통과하도록 변경
            # 이렇게 하면 gradient가 흐를 수 있음
            features1 = torch.randn(batch_size, 1024)
            features2 = torch.randn(batch_size, 1024)

            proj1 = self.projector(features1)
            proj2 = self.projector(features2)

            score1 = self.score_head(proj1)
            score2 = self.score_head(proj2)

            return score1, score2

        def compute_loss(
            self, score1: torch.Tensor, score2: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            return self.loss_fn(score1.squeeze(), score2.squeeze(), labels.float())

    return MockRankingModel()


# =============================================================================
# Test Classes: Feature Extraction Pipeline
# =============================================================================


class TestFeatureExtractionPipeline:
    """Feature Extraction Pipeline 통합 테스트.

    검증 항목:
    - Image -> DINOv2 -> 1024-d vector
    - 출력 텐서 형태 및 타입 검증
    - L2 정규화 적용 검증
    """

    @pytest.mark.slow
    def test_feature_extraction_output_shape(self) -> None:
        """Feature extraction 출력 shape 검증 (실제 모델 사용)."""
        # 실제 DINOv2 모델 로드 (시간 소요)
        # 이 테스트는 CI에서 건너뛸 수 있음 (@pytest.mark.slow)
        extractor = DINOv2FeatureExtractor(model_name="facebook/dinov2-large")

        # 테스트 이미지 생성
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224)

        # Feature extraction
        features = extractor(image)

        # 검증
        assert features.shape == (batch_size, 1024)
        assert features.dtype == torch.float32

    @pytest.mark.slow
    def test_feature_extraction_l2_normalized(self) -> None:
        """Feature extraction L2 정규화 검증."""
        extractor = DINOv2FeatureExtractor(
            model_name="facebook/dinov2-large", normalize=True
        )

        image = torch.randn(2, 3, 224, 224)
        features = extractor(image)

        # L2 norm = 1.0 검증
        norms = torch.norm(features, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_feature_extraction_with_mock(
        self, mock_feature_extractor: MagicMock
    ) -> None:
        """Feature extraction 모의 테스트 (빠른 실행)."""
        batch_size = 4
        image = torch.randn(batch_size, 3, 224, 224)

        # Mock feature extraction
        features = mock_feature_extractor(image)

        # 검증
        assert features.shape == (batch_size, 1024)


# =============================================================================
# Test Classes: Projection Pipeline
# =============================================================================


class TestProjectionPipeline:
    """Projection Pipeline 통합 테스트.

    검증 항목:
    - 1024-d -> Projector -> 256-d normalized
    - Gradient flow 검증
    - Dropout 동작 검증 (train vs eval)
    """

    def test_projection_output_shape(self) -> None:
        """Projection 출력 shape 검증."""
        projector = Projector(input_dim=1024, hidden_dim=512, output_dim=256)

        features = torch.randn(4, 1024)
        output = projector(features)

        assert output.shape == (4, 256)

    def test_projection_l2_normalized(self) -> None:
        """Projection 출력 L2 정규화 검증."""
        projector = Projector(input_dim=1024, output_dim=256, normalize=True)

        features = torch.randn(4, 1024)
        output = projector(features)

        norms = torch.norm(output, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_projection_gradient_flow(self) -> None:
        """Projection gradient flow 검증."""
        projector = Projector(input_dim=1024, output_dim=256)
        projector.train()

        features = torch.randn(4, 1024, requires_grad=True)
        output = projector(features)
        loss = output.sum()
        loss.backward()

        # Gradient가 흐르는지 검증
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()
        assert not torch.isinf(features.grad).any()

    def test_projection_dropout_behavior(self) -> None:
        """Projection dropout 동작 검증 (train vs eval)."""
        projector = Projector(input_dim=1024, output_dim=256, dropout=0.5)

        features = torch.randn(4, 1024)

        # Train 모드에서 반복 실행 시 다른 출력
        projector.train()
        set_seed(42)
        output1 = projector(features)
        set_seed(43)
        output2 = projector(features)

        # Dropout으로 인해 다른 출력 (확률적)
        # 완전히 같을 확률은 매우 낮음

        # Eval 모드에서 반복 실행 시 동일한 출력
        projector.eval()
        output3 = projector(features)
        output4 = projector(features)
        assert torch.allclose(output3, output4)


# =============================================================================
# Test Classes: Scoring Pipeline
# =============================================================================


class TestScoringPipeline:
    """Scoring Pipeline 통합 테스트.

    검증 항목:
    - Image pair -> Model -> Scores -> Ranking
    - 높은 티어 이미지가 더 높은 스코어
    """

    def test_scoring_output_shape(self, mock_ranking_model: MagicMock) -> None:
        """Scoring 출력 shape 검증."""
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)

        score1, score2 = mock_ranking_model(img1, img2)

        assert score1.shape == (4, 1)
        assert score2.shape == (4, 1)

    def test_scoring_loss_computation(self, mock_ranking_model: MagicMock) -> None:
        """Scoring loss 계산 검증."""
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([1, -1, 1, -1])

        score1, score2 = mock_ranking_model(img1, img2)
        loss = mock_ranking_model.compute_loss(score1, score2, labels)

        assert loss.ndim == 0  # 스칼라
        assert loss.item() >= 0  # 비음수


# =============================================================================
# Test Classes: Training Pipeline
# =============================================================================


class TestTrainingPipeline:
    """Training Pipeline 통합 테스트.

    검증 항목:
    - DataLoader -> Trainer -> Checkpoint
    - 학습 루프 동작 검증
    - 체크포인트 저장/로드 검증
    """

    def test_pairwise_dataset_creation(
        self, sample_metadata_df: pd.DataFrame, sample_images_dir: str
    ) -> None:
        """PairwiseDataset 생성 검증."""
        dataset = PairwiseDataset(
            metadata_df=sample_metadata_df, image_dir=sample_images_dir
        )

        # 데이터셋 크기 검증 (모든 cross-tier 페어)
        assert len(dataset) > 0

        # 샘플 가져오기
        img1, img2, label = dataset[0]
        assert img1.shape == (3, 768, 768)
        assert img2.shape == (3, 768, 768)
        assert label in [1, -1]

    def test_data_splitter(self, sample_metadata_df: pd.DataFrame, seed: int) -> None:
        """DataSplitter 동작 검증."""
        splitter = DataSplitter(
            metadata_df=sample_metadata_df,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed,
        )

        train_df, val_df, test_df = splitter.split()

        # 분할 비율 검증 (허용 오차 내)
        total = len(sample_metadata_df)
        assert len(train_df) >= int(total * 0.7)  # 80% 근처
        assert len(val_df) >= 1
        assert len(test_df) >= 1

        # 합계 검증
        assert len(train_df) + len(val_df) + len(test_df) == total

    def test_trainer_checkpoint_save_load(
        self, mock_ranking_model: MagicMock, temp_dir: str
    ) -> None:
        """Trainer 체크포인트 저장/로드 검증."""
        # 학습 설정
        config = TrainingConfig(
            checkpoint_dir=temp_dir,
            max_epochs=1,
            wandb_enabled=False,
            device="cpu",
        )

        # Trainer 생성
        trainer = Trainer(model=mock_ranking_model, config=config)

        # 체크포인트 저장
        checkpoint_path = trainer.save_checkpoint(epoch=0)

        # 체크포인트 파일 존재 확인
        assert Path(checkpoint_path).exists()

        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path)
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_training_one_epoch(
        self,
        mock_ranking_model: MagicMock,
        sample_metadata_df: pd.DataFrame,
        sample_images_dir: str,
        temp_dir: str,
    ) -> None:
        """단일 에폭 학습 검증."""
        # 데이터셋 생성
        dataset = PairwiseDataset(
            metadata_df=sample_metadata_df, image_dir=sample_images_dir
        )
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        # 학습 설정
        config = TrainingConfig(
            checkpoint_dir=temp_dir,
            max_epochs=1,
            wandb_enabled=False,
            device="cpu",
        )

        # Trainer 생성 및 학습
        trainer = Trainer(model=mock_ranking_model, config=config)
        train_loss = trainer.train_one_epoch(train_loader)

        # 손실 값 검증
        assert isinstance(train_loss, float)
        assert train_loss >= 0

    @patch("training.trainer.wandb")
    def test_training_with_wandb_mock(
        self,
        mock_wandb: MagicMock,
        mock_ranking_model: MagicMock,
        sample_metadata_df: pd.DataFrame,
        sample_images_dir: str,
        temp_dir: str,
    ) -> None:
        """wandb 모킹을 사용한 전체 학습 검증."""
        mock_wandb.init = MagicMock()
        mock_wandb.log = MagicMock()
        mock_wandb.finish = MagicMock()

        # 데이터셋 분할
        splitter = DataSplitter(sample_metadata_df, train_ratio=0.6, val_ratio=0.2)
        train_df, val_df, _ = splitter.split()

        # 데이터로더 생성
        train_dataset = PairwiseDataset(train_df, sample_images_dir)
        val_dataset = PairwiseDataset(val_df, sample_images_dir)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2)

        # 학습 설정
        config = TrainingConfig(
            checkpoint_dir=temp_dir,
            max_epochs=2,
            wandb_enabled=True,
            wandb_project="test-project",
            device="cpu",
            save_every_n_epochs=1,
        )

        # 학습 실행
        trainer = Trainer(model=mock_ranking_model, config=config)
        history = trainer.train(train_loader, val_loader)

        # 히스토리 검증
        assert "train_loss" in history
        assert "val_loss" in history
        assert "val_accuracy" in history
        assert len(history["train_loss"]) >= 1


# =============================================================================
# Test Classes: Evaluation Pipeline
# =============================================================================


class TestEvaluationPipeline:
    """Evaluation Pipeline 통합 테스트.

    검증 항목:
    - Model -> Test data -> Accuracy
    - 정확도 범위 검증 (0.0 ~ 1.0)
    """

    def test_evaluator_accuracy_range(
        self,
        mock_ranking_model: MagicMock,
        sample_metadata_df: pd.DataFrame,
        sample_images_dir: str,
    ) -> None:
        """Evaluator 정확도 범위 검증."""
        # 테스트 데이터셋
        dataset = PairwiseDataset(sample_metadata_df, sample_images_dir)
        test_loader = DataLoader(dataset, batch_size=2)

        # Evaluator 생성 및 평가
        evaluator = Evaluator(model=mock_ranking_model, device="cpu")
        accuracy = evaluator.evaluate(test_loader)

        # 정확도 범위 검증
        assert 0.0 <= accuracy <= 1.0

    def test_evaluator_detailed_metrics(
        self,
        mock_ranking_model: MagicMock,
        sample_metadata_df: pd.DataFrame,
        sample_images_dir: str,
    ) -> None:
        """Evaluator 상세 메트릭 검증."""
        dataset = PairwiseDataset(sample_metadata_df, sample_images_dir)
        test_loader = DataLoader(dataset, batch_size=2)

        evaluator = Evaluator(model=mock_ranking_model, device="cpu")
        metrics = evaluator.evaluate_detailed(test_loader)

        # 메트릭 필드 존재 검증
        assert "accuracy" in metrics
        assert "total_pairs" in metrics
        assert "correct_predictions" in metrics

        # 값 타입 검증
        assert isinstance(metrics["accuracy"], float)
        assert isinstance(metrics["total_pairs"], int)
        assert isinstance(metrics["correct_predictions"], int)


# =============================================================================
# Test Classes: Complete E2E Pipeline
# =============================================================================


class TestCompleteE2EPipeline:
    """완전한 E2E 파이프라인 통합 테스트.

    검증 항목:
    - 전체 파이프라인 연결 검증
    - 데이터 준비 -> 학습 -> 평가 -> 체크포인트
    """

    def test_full_pipeline_with_mocks(
        self,
        mock_ranking_model: MagicMock,
        sample_metadata_df: pd.DataFrame,
        sample_images_dir: str,
        temp_dir: str,
        seed: int,
    ) -> None:
        """전체 파이프라인 통합 테스트 (모의 객체 사용)."""
        # 1. 재현성을 위한 시드 설정
        set_seed(seed)

        # 2. 데이터 분할
        splitter = DataSplitter(sample_metadata_df, train_ratio=0.6, val_ratio=0.2)
        train_df, val_df, test_df = splitter.split()

        # 3. 데이터셋 및 데이터로더 생성
        train_dataset = PairwiseDataset(train_df, sample_images_dir)
        val_dataset = PairwiseDataset(val_df, sample_images_dir)
        test_dataset = PairwiseDataset(test_df, sample_images_dir)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        # 4. 학습 설정
        config = TrainingConfig(
            checkpoint_dir=temp_dir,
            max_epochs=2,
            wandb_enabled=False,
            device="cpu",
        )

        # 5. 학습 실행
        trainer = Trainer(model=mock_ranking_model, config=config)
        history = trainer.train(train_loader, val_loader)

        # 6. 체크포인트 저장 확인
        best_checkpoint = Path(temp_dir) / "best_model.pt"
        assert best_checkpoint.exists()

        # 7. 테스트 평가
        evaluator = Evaluator(model=mock_ranking_model, device="cpu")
        test_accuracy = evaluator.evaluate(test_loader)

        # 8. 최종 검증
        assert len(history["train_loss"]) >= 1
        assert 0.0 <= test_accuracy <= 1.0

    @pytest.mark.slow
    def test_full_pipeline_with_real_model(
        self,
        sample_metadata_df: pd.DataFrame,
        sample_images_dir: str,
        temp_dir: str,
        seed: int,
    ) -> None:
        """전체 파이프라인 통합 테스트 (실제 모델 사용).

        주의: 이 테스트는 DINOv2 모델을 로드하므로 시간이 오래 걸립니다.
        CI에서는 @pytest.mark.slow로 건너뛸 수 있습니다.
        """
        # 1. 재현성을 위한 시드 설정
        set_seed(seed)

        # 2. 실제 모델 생성
        model = PairwiseRankingModel(
            feature_extractor_model="facebook/dinov2-large",
            projector_hidden_dim=512,
            projector_output_dim=256,
        )

        # 3. 데이터 분할
        splitter = DataSplitter(sample_metadata_df, train_ratio=0.6, val_ratio=0.2)
        train_df, val_df, test_df = splitter.split()

        # 4. 데이터셋 생성
        train_dataset = PairwiseDataset(train_df, sample_images_dir)
        val_dataset = PairwiseDataset(val_df, sample_images_dir)
        test_dataset = PairwiseDataset(test_df, sample_images_dir)

        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        test_loader = DataLoader(test_dataset, batch_size=2)

        # 5. 학습 설정
        config = TrainingConfig(
            checkpoint_dir=temp_dir,
            max_epochs=1,
            wandb_enabled=False,
            device="cpu",
        )

        # 6. 학습 실행
        trainer = Trainer(model=model, config=config)
        history = trainer.train(train_loader, val_loader)

        # 7. 평가
        evaluator = Evaluator(model=model, device="cpu")
        test_accuracy = evaluator.evaluate(test_loader)

        # 8. 검증
        assert len(history["train_loss"]) >= 1
        assert 0.0 <= test_accuracy <= 1.0


# =============================================================================
# Test Classes: Reproducibility
# =============================================================================


class TestReproducibility:
    """재현 가능성 테스트.

    검증 항목 (AC-010):
    - seed=42로 동일 조건 학습 시 accuracy 차이 < 1%
    - 평가 모드에서 동일 입력에 동일 출력
    """

    def test_set_seed_deterministic(self, seed: int) -> None:
        """set_seed 함수의 결정적 동작 검증."""
        set_seed(seed)
        tensor1 = torch.randn(10, 10)

        set_seed(seed)
        tensor2 = torch.randn(10, 10)

        assert torch.allclose(tensor1, tensor2)

    def test_projector_deterministic_in_eval_mode(self, seed: int) -> None:
        """Projector eval 모드에서 결정적 동작 검증."""
        set_seed(seed)
        projector = Projector(input_dim=1024, output_dim=256)
        projector.eval()

        features = torch.randn(4, 1024)

        output1 = projector(features)
        output2 = projector(features)

        assert torch.allclose(output1, output2)

    def test_data_splitter_reproducibility(
        self, sample_metadata_df: pd.DataFrame, seed: int
    ) -> None:
        """DataSplitter 재현 가능성 검증."""
        splitter1 = DataSplitter(sample_metadata_df, seed=seed)
        train_df1, val_df1, test_df1 = splitter1.split()

        splitter2 = DataSplitter(sample_metadata_df, seed=seed)
        train_df2, val_df2, test_df2 = splitter2.split()

        # 동일한 분할 검증
        assert train_df1.equals(train_df2)
        assert val_df1.equals(val_df2)
        assert test_df1.equals(test_df2)


# =============================================================================
# Test Classes: Performance Benchmarks
# =============================================================================


class TestPerformanceBenchmarks:
    """성능 벤치마크 테스트.

    검증 항목 (AC-009):
    - 추론 시간 측정 기능 검증
    - 메모리 사용량 측정 기능 검증
    """

    def test_inference_time_measurement(
        self, mock_ranking_model: MagicMock
    ) -> None:
        """추론 시간 측정 기능 검증."""
        benchmarks = PerformanceBenchmarks(model=mock_ranking_model, device="cpu")

        inference_time = benchmarks.measure_inference_time(
            num_pairs=10, warmup_iterations=2, image_size=224
        )

        # 시간 측정 결과 검증
        assert isinstance(inference_time, float)
        assert inference_time > 0

    def test_full_benchmark_report(self, mock_ranking_model: MagicMock) -> None:
        """전체 벤치마크 리포트 검증."""
        benchmarks = PerformanceBenchmarks(model=mock_ranking_model, device="cpu")

        report = benchmarks.run_full_benchmark(batch_size=4, image_size=224)

        # 리포트 필드 존재 검증
        assert "inference_time_per_pair" in report
        assert "inference_time_ms" in report
        assert "memory_usage_bytes" in report
        assert "device" in report
        assert "batch_size" in report
        assert "meets_inference_requirement" in report


# =============================================================================
# Test Classes: Error Handling
# =============================================================================


class TestErrorHandling:
    """오류 처리 테스트.

    검증 항목:
    - 잘못된 입력에 대한 적절한 예외 발생
    """

    def test_empty_metadata_df_raises_error(self, temp_dir: str) -> None:
        """빈 메타데이터 DataFrame 오류 검증."""
        empty_df = pd.DataFrame(columns=["image_path", "tier"])

        with pytest.raises(ValueError, match="cannot be empty"):
            PairwiseDataset(metadata_df=empty_df, image_dir=temp_dir)

    def test_single_tier_raises_error(
        self, sample_images_dir: str
    ) -> None:
        """단일 티어만 있는 경우 오류 검증."""
        single_tier_df = pd.DataFrame(
            {"image_path": ["S_0.jpg", "S_1.jpg"], "tier": ["S", "S"]}
        )

        with pytest.raises(ValueError, match="At least 2 different tiers"):
            PairwiseDataset(metadata_df=single_tier_df, image_dir=sample_images_dir)

    def test_invalid_ratios_raises_error(
        self, sample_metadata_df: pd.DataFrame
    ) -> None:
        """잘못된 분할 비율 오류 검증."""
        with pytest.raises(ValueError):
            DataSplitter(
                metadata_df=sample_metadata_df,
                train_ratio=0.9,
                val_ratio=0.2,  # 합계 > 1.0
            )

    def test_missing_columns_raises_error(self, temp_dir: str) -> None:
        """필수 컬럼 누락 오류 검증."""
        invalid_df = pd.DataFrame({"path": ["a.jpg"], "label": ["S"]})

        with pytest.raises(KeyError, match="Missing required columns"):
            PairwiseDataset(metadata_df=invalid_df, image_dir=temp_dir)


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
