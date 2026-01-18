# test_evaluator.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: RED - Failing Tests
"""
Evaluator 테스트 모듈

Acceptance Criteria (AC-007):
- Pairwise accuracy 계산: 정확도 = (올바른 예측 수) / (전체 페어 수)
- 정확도가 0과 1 사이의 값이어야 함
- 배치 평가 지원
- 목표: 테스트 셋에서 60%+ accuracy

Example:
    >>> evaluator = Evaluator(model)
    >>> accuracy = evaluator.evaluate(test_loader)
    >>> assert 0.0 <= accuracy <= 1.0
"""

from unittest.mock import MagicMock, patch
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import will fail initially (RED phase)
from training.evaluator import Evaluator


class TestEvaluatorInitialization:
    """Evaluator 초기화 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock PairwiseRankingModel"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        # forward returns (score1, score2)
        model.return_value = (torch.randn(4, 1), torch.randn(4, 1))
        return model

    def test_evaluator_initialization(self, mock_model):
        """
        Evaluator 초기화 검증

        Given: 학습된 모델
        When: Evaluator 생성
        Then: 모델이 eval 모드로 설정되어야 함
        """
        # Act
        evaluator = Evaluator(model=mock_model)

        # Assert
        assert evaluator.model is mock_model
        mock_model.eval.assert_called()

    def test_evaluator_sets_device(self, mock_model):
        """
        Evaluator가 디바이스를 올바르게 설정

        Given: 모델과 디바이스 지정
        When: Evaluator 생성
        Then: 모델이 지정된 디바이스로 이동
        """
        # Act
        evaluator = Evaluator(model=mock_model, device="cpu")

        # Assert
        mock_model.to.assert_called()


class TestEvaluatorPairwiseAccuracy:
    """Pairwise Accuracy 계산 테스트 (AC-007)"""

    @pytest.fixture
    def mock_model(self):
        """Mock model for accuracy testing"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    @pytest.fixture
    def test_dataloader(self):
        """테스트용 DataLoader"""
        # 8개 샘플 (img1, img2, label)
        img1 = torch.randn(8, 3, 224, 224)
        img2 = torch.randn(8, 3, 224, 224)
        # labels: 1 (img1 > img2), -1 (img1 < img2)
        labels = torch.tensor([1, 1, -1, -1, 1, -1, 1, -1])

        dataset = TensorDataset(img1, img2, labels)
        return DataLoader(dataset, batch_size=4)

    def test_evaluate_returns_accuracy_between_0_and_1(self, mock_model, test_dataloader):
        """
        AC-007: 정확도가 0과 1 사이의 값이어야 함

        Given: 학습된 모델과 테스트 데이터
        When: evaluate() 호출
        Then: 0 <= accuracy <= 1
        """
        # Arrange
        # 모델이 일관된 예측을 하도록 설정
        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            score1 = torch.randn(batch_size, 1)
            score2 = torch.randn(batch_size, 1)
            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(test_dataloader)

        # Assert
        assert isinstance(accuracy, float), "Accuracy should be a float"
        assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy} should be between 0 and 1"

    def test_evaluate_calculates_correct_accuracy(self, mock_model):
        """
        AC-007: 정확도 = (올바른 예측 수) / (전체 페어 수)

        Given: 모델이 100% 정확한 예측을 하는 경우
        When: evaluate() 호출
        Then: accuracy == 1.0
        """
        # Arrange - 4개 샘플, 모두 정확한 예측
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        # labels: [1, 1, -1, -1] - img1이 더 좋으면 1, img2가 더 좋으면 -1
        labels = torch.tensor([1, 1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        # 모델이 정확한 예측을 하도록 설정
        # label=1 -> score1 > score2
        # label=-1 -> score1 < score2
        def mock_forward(img1, img2):
            # 배치 크기에 맞는 스코어 반환
            batch_size = img1.shape[0]
            # score1이 label에 따라 크거나 작도록 설정
            score1 = torch.tensor([[2.0], [2.0], [0.0], [0.0]])[:batch_size]
            score2 = torch.tensor([[1.0], [1.0], [1.0], [1.0]])[:batch_size]
            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(dataloader)

        # Assert
        assert accuracy == 1.0, f"Expected 100% accuracy, got {accuracy}"

    def test_evaluate_with_50_percent_accuracy(self, mock_model):
        """
        50% 정확도 계산 검증

        Given: 절반만 맞는 예측
        When: evaluate() 호출
        Then: accuracy == 0.5
        """
        # Arrange - 4개 샘플
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        # labels: [1, 1, -1, -1]
        labels = torch.tensor([1, 1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        # 모델이 절반만 맞추도록 설정
        # 예측: [1, -1, -1, 1] vs 정답: [1, 1, -1, -1] -> 2/4 = 50%
        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            # index 0, 2는 맞고, index 1, 3은 틀림
            score1 = torch.tensor([[2.0], [0.5], [0.5], [2.0]])[:batch_size]
            score2 = torch.tensor([[1.0], [1.0], [1.0], [1.0]])[:batch_size]
            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(dataloader)

        # Assert
        assert accuracy == 0.5, f"Expected 50% accuracy, got {accuracy}"

    def test_evaluate_returns_zero_for_all_wrong(self, mock_model):
        """
        모든 예측이 틀린 경우 accuracy = 0.0

        Given: 모델이 모두 틀린 예측
        When: evaluate() 호출
        Then: accuracy == 0.0
        """
        # Arrange
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        # 모델이 모두 틀리도록 설정
        # 예측: [-1, -1, 1, 1] vs 정답: [1, 1, -1, -1] -> 0/4 = 0%
        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            # label=1인데 score1 < score2, label=-1인데 score1 > score2
            score1 = torch.tensor([[0.5], [0.5], [2.0], [2.0]])[:batch_size]
            score2 = torch.tensor([[1.0], [1.0], [1.0], [1.0]])[:batch_size]
            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(dataloader)

        # Assert
        assert accuracy == 0.0, f"Expected 0% accuracy, got {accuracy}"


class TestEvaluatorBatchProcessing:
    """배치 처리 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    def test_evaluate_multiple_batches(self, mock_model):
        """
        여러 배치 처리 검증

        Given: 여러 배치로 나뉜 데이터
        When: evaluate() 호출
        Then: 모든 배치의 결과가 합쳐져서 정확도 계산
        """
        # Arrange - 16개 샘플, batch_size=4 -> 4 batches
        img1 = torch.randn(16, 3, 224, 224)
        img2 = torch.randn(16, 3, 224, 224)
        # 8개 맞고, 8개 틀림 -> 50%
        labels = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        batch_count = [0]

        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            batch_idx = batch_count[0]
            batch_count[0] += 1

            # 배치별로 다른 결과 반환 (전체적으로 75% 정확도)
            if batch_idx == 0:
                # 첫 번째 배치: labels=[1,1,1,1], 예측=[1,1,1,1] -> 100%
                score1 = torch.tensor([[2.0]] * batch_size)
                score2 = torch.tensor([[1.0]] * batch_size)
            elif batch_idx == 1:
                # 두 번째 배치: labels=[-1,-1,-1,-1], 예측=[-1,-1,-1,-1] -> 100%
                score1 = torch.tensor([[0.5]] * batch_size)
                score2 = torch.tensor([[1.0]] * batch_size)
            elif batch_idx == 2:
                # 세 번째 배치: labels=[1,1,1,1], 예측=[1,1,-1,-1] -> 50%
                score1 = torch.tensor([[2.0], [2.0], [0.5], [0.5]])
                score2 = torch.tensor([[1.0]] * batch_size)
            else:
                # 네 번째 배치: labels=[-1,-1,-1,-1], 예측=[-1,-1,1,1] -> 50%
                score1 = torch.tensor([[0.5], [0.5], [2.0], [2.0]])
                score2 = torch.tensor([[1.0]] * batch_size)

            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(dataloader)

        # Assert
        # 4+4+2+2 = 12 correct out of 16 -> 75%
        assert accuracy == 0.75, f"Expected 75% accuracy, got {accuracy}"

    def test_evaluate_handles_different_batch_sizes(self, mock_model):
        """
        다른 크기의 배치 처리 검증

        Given: 마지막 배치가 작은 경우
        When: evaluate() 호출
        Then: 올바르게 처리됨
        """
        # Arrange - 10개 샘플, batch_size=4 -> 3 batches (4, 4, 2)
        img1 = torch.randn(10, 3, 224, 224)
        img2 = torch.randn(10, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1, 1, 1, -1, -1, 1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            # 모두 정답으로 예측
            score1 = torch.zeros(batch_size, 1)
            score2 = torch.zeros(batch_size, 1)

            # img1이 labels에 따라 더 크거나 작도록 설정
            for i in range(batch_size):
                if i < len(labels):
                    # 정답에 맞게 스코어 설정
                    score1[i] = 2.0
                    score2[i] = 1.0

            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(dataloader)

        # Assert
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestEvaluatorInferenceMode:
    """추론 모드 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    def test_evaluate_uses_no_grad(self, mock_model):
        """
        evaluate()가 torch.no_grad() 사용 검증

        Given: Evaluator
        When: evaluate() 호출
        Then: gradient 계산이 비활성화됨
        """
        # Arrange
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        gradient_enabled = [True]

        def mock_forward(img1, img2):
            # 현재 gradient 상태 기록
            gradient_enabled[0] = torch.is_grad_enabled()
            batch_size = img1.shape[0]
            return torch.randn(batch_size, 1), torch.randn(batch_size, 1)

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        evaluator.evaluate(dataloader)

        # Assert
        assert gradient_enabled[0] is False, "Should use no_grad during evaluation"

    def test_evaluate_sets_model_to_eval_mode(self, mock_model):
        """
        evaluate()가 model.eval() 호출 검증

        Given: Evaluator
        When: evaluate() 호출
        Then: 모델이 eval 모드로 설정됨
        """
        # Arrange
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            return torch.randn(batch_size, 1), torch.randn(batch_size, 1)

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Reset the call count from initialization
        mock_model.eval.reset_mock()

        # Act
        evaluator.evaluate(dataloader)

        # Assert
        mock_model.eval.assert_called()


class TestEvaluatorDetailedMetrics:
    """상세 메트릭 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    def test_evaluate_detailed_returns_metrics_dict(self, mock_model):
        """
        상세 평가 결과 반환 검증

        Given: Evaluator
        When: evaluate_detailed() 호출
        Then: 상세 메트릭 딕셔너리 반환
        """
        # Arrange
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            score1 = torch.tensor([[2.0], [2.0], [0.5], [0.5]])[:batch_size]
            score2 = torch.tensor([[1.0], [1.0], [1.0], [1.0]])[:batch_size]
            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        metrics = evaluator.evaluate_detailed(dataloader)

        # Assert
        assert isinstance(metrics, dict), "Should return a dictionary"
        assert "accuracy" in metrics, "Should include accuracy"
        assert "total_pairs" in metrics, "Should include total_pairs"
        assert "correct_predictions" in metrics, "Should include correct_predictions"

    def test_evaluate_detailed_counts_correct(self, mock_model):
        """
        올바른 예측 수 계산 검증

        Given: 4개 중 3개 맞춤
        When: evaluate_detailed() 호출
        Then: correct_predictions == 3, total_pairs == 4
        """
        # Arrange
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        # 3/4 정답 (index 0,1,2 정답, index 3 오답)
        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            # labels: [1, 1, -1, -1]
            # 예측: [1, 1, -1, 1] -> 3/4 정답
            score1 = torch.tensor([[2.0], [2.0], [0.5], [2.0]])[:batch_size]
            score2 = torch.tensor([[1.0], [1.0], [1.0], [1.0]])[:batch_size]
            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        metrics = evaluator.evaluate_detailed(dataloader)

        # Assert
        assert metrics["total_pairs"] == 4
        assert metrics["correct_predictions"] == 3
        assert metrics["accuracy"] == 0.75


class TestEvaluatorEdgeCases:
    """엣지 케이스 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    def test_evaluate_empty_dataloader(self, mock_model):
        """
        빈 데이터로더 처리 검증

        Given: 빈 데이터로더
        When: evaluate() 호출
        Then: 예외 없이 처리됨 (0.0 또는 적절한 값)
        """
        # Arrange
        dataset = TensorDataset(
            torch.empty(0, 3, 224, 224),
            torch.empty(0, 3, 224, 224),
            torch.empty(0, dtype=torch.long)
        )
        dataloader = DataLoader(dataset, batch_size=4)
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(dataloader)

        # Assert
        assert isinstance(accuracy, float)
        assert accuracy == 0.0, "Empty dataloader should return 0.0 accuracy"

    def test_evaluate_single_sample(self, mock_model):
        """
        단일 샘플 처리 검증

        Given: 1개 샘플
        When: evaluate() 호출
        Then: 올바른 정확도 반환
        """
        # Arrange
        img1 = torch.randn(1, 3, 224, 224)
        img2 = torch.randn(1, 3, 224, 224)
        labels = torch.tensor([1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=1)

        def mock_forward(img1, img2):
            # 정답 예측
            return torch.tensor([[2.0]]), torch.tensor([[1.0]])

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(dataloader)

        # Assert
        assert accuracy == 1.0, "Single correct prediction should give 100% accuracy"

    def test_evaluate_equal_scores(self, mock_model):
        """
        동일 스코어 처리 검증

        Given: score1 == score2인 경우
        When: evaluate() 호출
        Then: 적절히 처리됨 (tie-breaking 정책)
        """
        # Arrange
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1])

        dataset = TensorDataset(img1, img2, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            # 모든 스코어가 동일
            score1 = torch.tensor([[1.0]] * batch_size)
            score2 = torch.tensor([[1.0]] * batch_size)
            return score1, score2

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        # Act
        accuracy = evaluator.evaluate(dataloader)

        # Assert
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


class TestEvaluatorPredictPair:
    """predict_pair 메서드 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    def test_predict_pair_returns_correct_tuple(self, mock_model):
        """
        predict_pair가 올바른 튜플 반환

        Given: 두 이미지
        When: predict_pair() 호출
        Then: (prediction, score1, score2) 튜플 반환
        """
        # Arrange
        def mock_forward(img1, img2):
            return torch.tensor([[2.5]]), torch.tensor([[1.0]])

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        img1 = torch.randn(1, 3, 224, 224)
        img2 = torch.randn(1, 3, 224, 224)

        # Act
        prediction, score1, score2 = evaluator.predict_pair(img1, img2)

        # Assert
        assert isinstance(prediction, int)
        assert isinstance(score1, float)
        assert isinstance(score2, float)

    def test_predict_pair_returns_1_when_img1_better(self, mock_model):
        """
        img1이 더 좋을 때 prediction=1 반환

        Given: score1 > score2
        When: predict_pair() 호출
        Then: prediction=1
        """
        # Arrange
        def mock_forward(img1, img2):
            return torch.tensor([[3.0]]), torch.tensor([[1.0]])

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        img1 = torch.randn(1, 3, 224, 224)
        img2 = torch.randn(1, 3, 224, 224)

        # Act
        prediction, score1, score2 = evaluator.predict_pair(img1, img2)

        # Assert
        assert prediction == 1
        assert score1 > score2

    def test_predict_pair_returns_minus1_when_img2_better(self, mock_model):
        """
        img2가 더 좋을 때 prediction=-1 반환

        Given: score1 < score2
        When: predict_pair() 호출
        Then: prediction=-1
        """
        # Arrange
        def mock_forward(img1, img2):
            return torch.tensor([[0.5]]), torch.tensor([[2.0]])

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        img1 = torch.randn(1, 3, 224, 224)
        img2 = torch.randn(1, 3, 224, 224)

        # Act
        prediction, score1, score2 = evaluator.predict_pair(img1, img2)

        # Assert
        assert prediction == -1
        assert score1 < score2

    def test_predict_pair_returns_0_when_equal(self, mock_model):
        """
        동점일 때 prediction=0 반환

        Given: score1 == score2
        When: predict_pair() 호출
        Then: prediction=0
        """
        # Arrange
        def mock_forward(img1, img2):
            return torch.tensor([[1.5]]), torch.tensor([[1.5]])

        mock_model.side_effect = mock_forward
        evaluator = Evaluator(model=mock_model)

        img1 = torch.randn(1, 3, 224, 224)
        img2 = torch.randn(1, 3, 224, 224)

        # Act
        prediction, score1, score2 = evaluator.predict_pair(img1, img2)

        # Assert
        assert prediction == 0
        assert score1 == score2


class TestEvaluatorRepr:
    """__repr__ 메서드 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    def test_repr_returns_string(self, mock_model):
        """
        __repr__가 문자열 반환

        Given: Evaluator 인스턴스
        When: repr() 호출
        Then: 문자열 반환
        """
        # Arrange
        evaluator = Evaluator(model=mock_model, device="cpu")

        # Act
        result = repr(evaluator)

        # Assert
        assert isinstance(result, str)
        assert "Evaluator" in result
        assert "cpu" in result

    def test_repr_contains_device_info(self, mock_model):
        """
        __repr__에 디바이스 정보 포함

        Given: 특정 디바이스로 초기화된 Evaluator
        When: repr() 호출
        Then: 디바이스 정보가 포함됨
        """
        # Arrange
        evaluator = Evaluator(model=mock_model, device="cpu")

        # Act
        result = repr(evaluator)

        # Assert
        assert "device=cpu" in result


class TestEvaluatorDetailedEdgeCases:
    """evaluate_detailed 엣지 케이스 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    def test_evaluate_detailed_empty_dataloader(self, mock_model):
        """
        빈 데이터로더로 evaluate_detailed 호출

        Given: 빈 데이터로더
        When: evaluate_detailed() 호출
        Then: 0 값의 메트릭 반환
        """
        # Arrange
        dataset = TensorDataset(
            torch.empty(0, 3, 224, 224),
            torch.empty(0, 3, 224, 224),
            torch.empty(0, dtype=torch.long)
        )
        dataloader = DataLoader(dataset, batch_size=4)
        evaluator = Evaluator(model=mock_model)

        # Act
        metrics = evaluator.evaluate_detailed(dataloader)

        # Assert
        assert metrics["accuracy"] == 0.0
        assert metrics["total_pairs"] == 0
        assert metrics["correct_predictions"] == 0
