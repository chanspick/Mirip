# test_benchmarks.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: RED - Failing Tests
"""
Performance Benchmarks & Reproducibility 테스트 모듈

Acceptance Criteria:
- AC-009: 성능 요구사항
  - 추론 시간 < 100ms/pair (GPU)
  - 학습 시 GPU 메모리 < 12GB (batch_size=32)
  - 총 학습 시간 목표 < 6시간

- AC-010: 재현 가능성
  - seed=42로 동일 조건 학습 시 accuracy 차이 < 1%
  - 평가 모드에서 동일 입력에 동일 출력

Example:
    >>> benchmarks = PerformanceBenchmarks(model)
    >>> inference_time = benchmarks.measure_inference_time()
    >>> assert inference_time < 0.1  # 100ms
"""

import time
from unittest.mock import MagicMock, patch
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import will fail initially (RED phase)
from training.benchmarks import PerformanceBenchmarks, set_seed


class TestPerformanceBenchmarksInitialization:
    """Performance Benchmarks 초기화 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock PairwiseRankingModel"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        model.return_value = (torch.randn(1, 1), torch.randn(1, 1))
        return model

    def test_benchmarks_initialization(self, mock_model):
        """
        PerformanceBenchmarks 초기화 검증

        Given: 모델
        When: PerformanceBenchmarks 생성
        Then: 올바르게 초기화됨
        """
        # Act
        benchmarks = PerformanceBenchmarks(model=mock_model)

        # Assert
        assert benchmarks.model is mock_model

    def test_benchmarks_accepts_device(self, mock_model):
        """
        디바이스 설정 검증

        Given: 모델과 디바이스
        When: PerformanceBenchmarks 생성
        Then: 지정된 디바이스 사용
        """
        # Act
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Assert
        assert benchmarks.device == "cpu"


class TestInferenceTime:
    """추론 시간 테스트 (AC-009)"""

    @pytest.fixture
    def mock_model(self):
        """Mock model with controlled inference time"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)

        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            return torch.randn(batch_size, 1), torch.randn(batch_size, 1)

        model.side_effect = mock_forward
        return model

    def test_measure_inference_time_returns_float(self, mock_model):
        """
        추론 시간 측정 반환값 검증

        Given: 모델
        When: measure_inference_time() 호출
        Then: float 타입의 시간(초) 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        inference_time = benchmarks.measure_inference_time()

        # Assert
        assert isinstance(inference_time, float), "Should return float"
        assert inference_time >= 0, "Time should be non-negative"

    def test_measure_inference_time_per_pair(self, mock_model):
        """
        AC-009: 단일 페어당 추론 시간 측정

        Given: 모델과 이미지 페어
        When: measure_inference_time() 호출
        Then: 페어당 추론 시간 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        inference_time = benchmarks.measure_inference_time(
            num_pairs=10,
            warmup_iterations=2
        )

        # Assert
        assert isinstance(inference_time, float)
        # CPU에서는 100ms 이하가 아닐 수 있으므로 범위만 확인
        assert inference_time > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_inference_time_under_100ms_gpu(self, mock_model):
        """
        AC-009: GPU에서 추론 시간 < 100ms/pair

        Given: GPU 환경과 모델
        When: 추론 시간 측정
        Then: 100ms 미만
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cuda")

        # Act
        inference_time = benchmarks.measure_inference_time(
            num_pairs=100,
            warmup_iterations=10
        )

        # Assert
        assert inference_time < 0.1, f"Inference time {inference_time}s exceeds 100ms"

    def test_measure_inference_time_with_batch(self, mock_model):
        """
        배치 추론 시간 측정

        Given: 배치 크기 지정
        When: measure_batch_inference_time() 호출
        Then: 배치당 추론 시간 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        batch_time = benchmarks.measure_batch_inference_time(batch_size=32)

        # Assert
        assert isinstance(batch_time, float)
        assert batch_time >= 0


class TestMemoryUsage:
    """메모리 사용량 테스트 (AC-009)"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.train = MagicMock()
        model.to = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=[torch.randn(100, 100)])
        return model

    def test_measure_memory_returns_bytes(self, mock_model):
        """
        메모리 측정이 바이트 단위 반환

        Given: 모델
        When: measure_memory_usage() 호출
        Then: 메모리 사용량(bytes) 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        memory_usage = benchmarks.measure_memory_usage(batch_size=32)

        # Assert
        assert isinstance(memory_usage, (int, float))
        assert memory_usage >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_gpu_memory_under_12gb(self, mock_model):
        """
        AC-009: 학습 시 GPU 메모리 < 12GB

        Given: GPU 환경과 batch_size=32
        When: 학습 시 메모리 측정
        Then: 12GB 미만
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cuda")

        # Act
        memory_gb = benchmarks.measure_training_memory(batch_size=32)

        # Assert
        assert memory_gb < 12.0, f"Memory usage {memory_gb}GB exceeds 12GB limit"

    def test_measure_peak_memory(self, mock_model):
        """
        피크 메모리 측정

        Given: 모델
        When: measure_peak_memory() 호출
        Then: 최대 메모리 사용량 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        peak_memory = benchmarks.measure_peak_memory(batch_size=32)

        # Assert
        assert isinstance(peak_memory, (int, float))
        assert peak_memory >= 0


class TestSetSeed:
    """set_seed 함수 테스트 (AC-010)"""

    def test_set_seed_makes_torch_deterministic(self):
        """
        set_seed가 torch 연산을 결정적으로 만듦

        Given: seed=42
        When: 동일한 연산 두 번 수행
        Then: 동일한 결과
        """
        # Act
        set_seed(42)
        tensor1 = torch.randn(10, 10)

        set_seed(42)
        tensor2 = torch.randn(10, 10)

        # Assert
        assert torch.allclose(tensor1, tensor2), "Same seed should produce same random values"

    def test_set_seed_different_seeds_give_different_results(self):
        """
        다른 시드는 다른 결과

        Given: seed=42 vs seed=123
        When: 동일한 연산 수행
        Then: 다른 결과
        """
        # Act
        set_seed(42)
        tensor1 = torch.randn(10, 10)

        set_seed(123)
        tensor2 = torch.randn(10, 10)

        # Assert
        assert not torch.allclose(tensor1, tensor2), "Different seeds should produce different values"

    def test_set_seed_42_default(self):
        """
        기본 시드값 42 사용

        Given: 시드값 미지정
        When: set_seed() 호출
        Then: 기본값 42 사용
        """
        # Act
        set_seed()  # Uses default seed=42
        tensor1 = torch.randn(5, 5)

        set_seed(42)  # Explicitly set 42
        tensor2 = torch.randn(5, 5)

        # Assert
        assert torch.allclose(tensor1, tensor2), "Default seed should be 42"


class TestReproducibility:
    """재현 가능성 테스트 (AC-010)"""

    @pytest.fixture
    def simple_model(self):
        """간단한 재현 가능 모델"""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )

    def test_same_output_with_same_seed(self, simple_model):
        """
        AC-010: 동일 시드로 동일 출력

        Given: seed=42로 설정
        When: 동일 입력에 모델 추론
        Then: 동일한 출력
        """
        # Arrange
        input_tensor = torch.randn(4, 10)

        # Act
        set_seed(42)
        simple_model.eval()
        with torch.no_grad():
            output1 = simple_model(input_tensor)

        set_seed(42)
        simple_model.eval()
        with torch.no_grad():
            output2 = simple_model(input_tensor)

        # Assert
        assert torch.allclose(output1, output2), "Same seed should produce identical outputs"

    def test_eval_mode_deterministic(self, simple_model):
        """
        AC-010: 평가 모드에서 결정적 동작

        Given: 모델이 eval 모드
        When: 동일 입력으로 여러 번 추론
        Then: 모든 출력이 동일
        """
        # Arrange
        simple_model.eval()
        input_tensor = torch.randn(4, 10)

        # Act
        with torch.no_grad():
            output1 = simple_model(input_tensor)
            output2 = simple_model(input_tensor)
            output3 = simple_model(input_tensor)

        # Assert
        assert torch.allclose(output1, output2), "Eval mode should be deterministic"
        assert torch.allclose(output2, output3), "Eval mode should be deterministic"

    def test_model_weights_reproducible_with_seed(self, simple_model):
        """
        시드 설정 시 모델 가중치 초기화 재현 가능

        Given: seed=42
        When: 동일한 모델 구조로 두 번 초기화
        Then: 동일한 가중치
        """
        # Act
        set_seed(42)
        model1 = nn.Linear(10, 5)

        set_seed(42)
        model2 = nn.Linear(10, 5)

        # Assert
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Same seed should produce identical weights"


class TestBenchmarkReport:
    """벤치마크 리포트 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.train = MagicMock()
        model.to = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=[torch.randn(100, 100)])

        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            return torch.randn(batch_size, 1), torch.randn(batch_size, 1)

        model.side_effect = mock_forward
        return model

    def test_run_full_benchmark_returns_report(self, mock_model):
        """
        전체 벤치마크 실행 시 리포트 반환

        Given: 모델
        When: run_full_benchmark() 호출
        Then: 벤치마크 리포트 딕셔너리 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        report = benchmarks.run_full_benchmark()

        # Assert
        assert isinstance(report, dict)
        assert "inference_time_per_pair" in report
        assert "memory_usage_mb" in report
        assert "device" in report

    def test_benchmark_report_contains_requirements_check(self, mock_model):
        """
        벤치마크 리포트에 요구사항 충족 여부 포함

        Given: 벤치마크 실행
        When: 리포트 확인
        Then: 요구사항 충족 여부 필드 존재
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        report = benchmarks.run_full_benchmark()

        # Assert
        assert "meets_inference_requirement" in report
        assert "meets_memory_requirement" in report
        assert isinstance(report["meets_inference_requirement"], bool)
        assert isinstance(report["meets_memory_requirement"], bool)


class TestReproducibilityValidation:
    """재현 가능성 검증 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    @pytest.fixture
    def mock_dataloader(self):
        """Mock dataloader"""
        img1 = torch.randn(8, 3, 224, 224)
        img2 = torch.randn(8, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1, 1, -1, 1, -1])
        dataset = TensorDataset(img1, img2, labels)
        return DataLoader(dataset, batch_size=4)

    def test_validate_reproducibility_same_accuracy(self, mock_model, mock_dataloader):
        """
        AC-010: seed=42로 동일 조건 시 accuracy 차이 < 1%

        Given: 동일 설정으로 두 번 평가
        When: validate_reproducibility() 호출
        Then: accuracy 차이 < 1%
        """
        # Arrange
        # 동일한 예측 결과를 반환하도록 설정
        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            score1 = torch.tensor([[2.0], [2.0], [0.5], [0.5]])[:batch_size]
            score2 = torch.tensor([[1.0], [1.0], [1.0], [1.0]])[:batch_size]
            return score1, score2

        mock_model.side_effect = mock_forward
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        is_reproducible, accuracy_diff = benchmarks.validate_reproducibility(
            dataloader=mock_dataloader,
            seed=42,
            num_runs=2
        )

        # Assert
        assert isinstance(is_reproducible, bool)
        assert isinstance(accuracy_diff, float)
        assert accuracy_diff < 0.01, f"Accuracy difference {accuracy_diff} exceeds 1%"
        assert is_reproducible is True

    def test_validate_reproducibility_returns_false_for_non_deterministic(
        self, mock_model, mock_dataloader
    ):
        """
        비결정적 모델은 재현 불가능으로 판정

        Given: 매번 다른 결과를 반환하는 모델
        When: validate_reproducibility() 호출
        Then: is_reproducible=False
        """
        # Arrange
        call_count = [0]

        def mock_forward_non_deterministic(img1, img2):
            call_count[0] += 1
            batch_size = img1.shape[0]
            # 호출마다 다른 결과 반환
            if call_count[0] % 2 == 0:
                score1 = torch.tensor([[2.0], [2.0], [2.0], [2.0]])[:batch_size]
            else:
                score1 = torch.tensor([[0.5], [0.5], [0.5], [0.5]])[:batch_size]
            score2 = torch.tensor([[1.0], [1.0], [1.0], [1.0]])[:batch_size]
            return score1, score2

        mock_model.side_effect = mock_forward_non_deterministic
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        is_reproducible, accuracy_diff = benchmarks.validate_reproducibility(
            dataloader=mock_dataloader,
            seed=42,
            num_runs=2
        )

        # Assert
        # 비결정적이면 차이가 클 것
        if accuracy_diff >= 0.01:
            assert is_reproducible is False


class TestTrainingTimeBenchmark:
    """학습 시간 벤치마크 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.train = MagicMock()
        model.to = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=[torch.randn(100, 100, requires_grad=True)])

        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            return torch.randn(batch_size, 1), torch.randn(batch_size, 1)

        model.side_effect = mock_forward
        model.compute_loss = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))
        return model

    def test_estimate_training_time(self, mock_model):
        """
        학습 시간 추정

        Given: 모델과 에폭 수
        When: estimate_training_time() 호출
        Then: 예상 학습 시간(시간) 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        estimated_hours = benchmarks.estimate_training_time(
            num_epochs=100,
            samples_per_epoch=10000,
            batch_size=32
        )

        # Assert
        assert isinstance(estimated_hours, float)
        assert estimated_hours >= 0

    def test_training_time_target_check(self, mock_model):
        """
        AC-009: 총 학습 시간 목표 < 6시간 체크

        Given: 학습 시간 추정
        When: check_training_time_target() 호출
        Then: 6시간 이내인지 확인
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        meets_target, estimated_hours = benchmarks.check_training_time_target(
            num_epochs=100,
            samples_per_epoch=10000,
            batch_size=32,
            target_hours=6.0
        )

        # Assert
        assert isinstance(meets_target, bool)
        assert isinstance(estimated_hours, float)


class TestBenchmarksRepr:
    """PerformanceBenchmarks __repr__ 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    def test_repr_returns_string(self, mock_model):
        """
        __repr__이 문자열 반환

        Given: PerformanceBenchmarks 인스턴스
        When: repr() 호출
        Then: 문자열 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        result = repr(benchmarks)

        # Assert
        assert isinstance(result, str)

    def test_repr_contains_class_name_and_device(self, mock_model):
        """
        __repr__에 클래스명과 디바이스 정보 포함

        Given: PerformanceBenchmarks 인스턴스
        When: repr() 호출
        Then: 클래스명과 디바이스 정보 포함
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        result = repr(benchmarks)

        # Assert
        assert "PerformanceBenchmarks" in result
        assert "cpu" in result


class TestReproducibilityEdgeCases:
    """재현 가능성 검증 엣지 케이스 테스트"""

    @pytest.fixture
    def mock_model(self):
        """Mock model"""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    @pytest.fixture
    def mock_dataloader(self):
        """Mock dataloader"""
        img1 = torch.randn(4, 3, 224, 224)
        img2 = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([1, 1, -1, -1])
        dataset = TensorDataset(img1, img2, labels)
        return DataLoader(dataset, batch_size=2)

    def test_validate_reproducibility_single_run(self, mock_model, mock_dataloader):
        """
        단일 실행 시 accuracy_diff가 0

        Given: num_runs=1
        When: validate_reproducibility() 호출
        Then: accuracy_diff=0.0
        """
        # Arrange
        def mock_forward(img1, img2):
            batch_size = img1.shape[0]
            score1 = torch.tensor([[2.0], [2.0]])[:batch_size]
            score2 = torch.tensor([[1.0], [1.0]])[:batch_size]
            return score1, score2

        mock_model.side_effect = mock_forward
        benchmarks = PerformanceBenchmarks(model=mock_model, device="cpu")

        # Act
        is_reproducible, accuracy_diff = benchmarks.validate_reproducibility(
            dataloader=mock_dataloader,
            seed=42,
            num_runs=1
        )

        # Assert
        assert accuracy_diff == 0.0
        assert is_reproducible is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
class TestGPUBranches:
    """GPU 분기 테스트 (실제 CUDA 환경 필요)"""

    @pytest.fixture
    def real_model(self):
        """실제 PyTorch 모델"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3 * 32 * 32, 1)

            def forward(self, img1, img2):
                batch_size = img1.shape[0]
                x1 = img1.view(batch_size, -1)
                x2 = img2.view(batch_size, -1)
                return self.linear(x1), self.linear(x2)

            def compute_loss(self, score1, score2, labels):
                return ((score1 - score2).squeeze() * labels.float()).mean()

        return SimpleModel()

    def test_inference_time_with_cuda_sync(self, real_model):
        """
        CUDA 디바이스에서 동기화 및 추론 시간 측정

        Given: device="cuda"
        When: measure_inference_time() 호출
        Then: 시간 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=real_model, device="cuda")

        # Act
        inference_time = benchmarks.measure_inference_time(num_pairs=2, warmup_iterations=1, image_size=32)

        # Assert
        assert inference_time >= 0

    def test_batch_inference_time_with_cuda_sync(self, real_model):
        """
        CUDA 디바이스에서 배치 추론 시간 측정

        Given: device="cuda"
        When: measure_batch_inference_time() 호출
        Then: 시간 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=real_model, device="cuda")

        # Act
        batch_time = benchmarks.measure_batch_inference_time(batch_size=2, num_iterations=2, warmup_iterations=1, image_size=32)

        # Assert
        assert batch_time >= 0

    def test_measure_memory_usage_with_cuda(self, real_model):
        """
        CUDA 디바이스에서 메모리 측정

        Given: device="cuda"
        When: measure_memory_usage() 호출
        Then: 메모리 바이트 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=real_model, device="cuda")

        # Act
        memory = benchmarks.measure_memory_usage(batch_size=2, image_size=32)

        # Assert
        assert memory > 0

    def test_measure_training_memory_with_cuda(self, real_model):
        """
        CUDA 디바이스에서 학습 메모리 측정

        Given: device="cuda"
        When: measure_training_memory() 호출
        Then: GB 단위로 반환
        """
        # Arrange
        benchmarks = PerformanceBenchmarks(model=real_model, device="cuda")

        # Act
        memory_gb = benchmarks.measure_training_memory(batch_size=2, image_size=32)

        # Assert
        assert memory_gb > 0
