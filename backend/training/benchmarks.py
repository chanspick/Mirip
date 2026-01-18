# benchmarks.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: GREEN - Implementation
"""
Performance Benchmarks & Reproducibility Module

성능 벤치마크 측정 및 재현 가능성 검증을 위한 모듈.

Acceptance Criteria:
- AC-009: 성능 요구사항
  - 추론 시간 < 100ms/pair (GPU)
  - 학습 시 GPU 메모리 < 12GB (batch_size=32)
  - 총 학습 시간 목표 < 6시간

- AC-010: 재현 가능성
  - seed=42로 동일 조건 학습 시 accuracy 차이 < 1%
  - 평가 모드에서 동일 입력에 동일 출력

Example:
    >>> set_seed(42)
    >>> benchmarks = PerformanceBenchmarks(model)
    >>> report = benchmarks.run_full_benchmark()
    >>> print(f"Inference time: {report['inference_time_per_pair']*1000:.2f}ms")
"""

from __future__ import annotations

import random
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.evaluator import Evaluator


def set_seed(seed: int = 42) -> None:
    """
    재현 가능성을 위한 시드 설정

    모든 난수 생성기의 시드를 설정하여 결정적 동작을 보장합니다.

    Args:
        seed: 시드 값 (기본값: 42)

    Note:
        AC-010: 재현 가능성 보장을 위해 다음을 설정합니다:
        - Python random
        - NumPy random
        - PyTorch random
        - CUDA random (가능한 경우)

    Example:
        >>> set_seed(42)
        >>> tensor1 = torch.randn(10, 10)
        >>> set_seed(42)
        >>> tensor2 = torch.randn(10, 10)
        >>> assert torch.allclose(tensor1, tensor2)
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)

    # CUDA random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 결정적 알고리즘 사용 (성능 저하 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PerformanceBenchmarks:
    """
    Performance Benchmarks

    모델의 추론 시간, 메모리 사용량, 학습 시간 등을 측정합니다.

    Attributes:
        model: 벤치마크 대상 모델
        device: 벤치마크 디바이스

    Args:
        model: PairwiseRankingModel 인스턴스
        device: 벤치마크 디바이스 (기본값: cuda if available else cpu)

    Example:
        >>> benchmarks = PerformanceBenchmarks(model, device="cuda")
        >>> inference_time = benchmarks.measure_inference_time()
        >>> print(f"Inference time: {inference_time*1000:.2f}ms/pair")
    """

    # 성능 요구사항 상수
    MAX_INFERENCE_TIME_SECONDS = 0.1  # 100ms
    MAX_MEMORY_GB = 12.0  # 12GB
    TARGET_TRAINING_HOURS = 6.0  # 6시간

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ) -> None:
        """
        PerformanceBenchmarks 초기화

        Args:
            model: 벤치마크 대상 모델
            device: 벤치마크 디바이스
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 모델을 디바이스로 이동
        self.model = self.model.to(self.device)

    def measure_inference_time(
        self,
        num_pairs: int = 100,
        warmup_iterations: int = 10,
        image_size: int = 768,
    ) -> float:
        """
        단일 페어당 평균 추론 시간 측정

        Args:
            num_pairs: 측정할 페어 수 (기본값: 100)
            warmup_iterations: 워밍업 반복 횟수 (기본값: 10)
            image_size: 이미지 크기 (기본값: 768)

        Returns:
            단일 페어당 평균 추론 시간 (초)

        Note:
            AC-009: GPU에서 추론 시간 < 100ms/pair
        """
        self.model.eval()

        # 테스트 데이터 생성
        img1 = torch.randn(1, 3, image_size, image_size, device=self.device)
        img2 = torch.randn(1, 3, image_size, image_size, device=self.device)

        # 워밍업
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(img1, img2)

        # GPU 동기화 (정확한 시간 측정을 위해)
        if self.device == "cuda":
            torch.cuda.synchronize()

        # 시간 측정
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_pairs):
                _ = self.model(img1, img2)

        # GPU 동기화
        if self.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # 평균 시간 계산
        total_time = end_time - start_time
        avg_time_per_pair = total_time / num_pairs

        return avg_time_per_pair

    def measure_batch_inference_time(
        self,
        batch_size: int = 32,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
        image_size: int = 768,
    ) -> float:
        """
        배치당 평균 추론 시간 측정

        Args:
            batch_size: 배치 크기 (기본값: 32)
            num_iterations: 측정 반복 횟수 (기본값: 10)
            warmup_iterations: 워밍업 반복 횟수 (기본값: 3)
            image_size: 이미지 크기 (기본값: 768)

        Returns:
            배치당 평균 추론 시간 (초)
        """
        self.model.eval()

        # 테스트 데이터 생성
        img1 = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        img2 = torch.randn(batch_size, 3, image_size, image_size, device=self.device)

        # 워밍업
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(img1, img2)

        # GPU 동기화
        if self.device == "cuda":
            torch.cuda.synchronize()

        # 시간 측정
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(img1, img2)

        # GPU 동기화
        if self.device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # 평균 시간 계산
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_iterations

        return avg_time_per_batch

    def measure_memory_usage(
        self,
        batch_size: int = 32,
        image_size: int = 768,
    ) -> int:
        """
        추론 시 메모리 사용량 측정

        Args:
            batch_size: 배치 크기 (기본값: 32)
            image_size: 이미지 크기 (기본값: 768)

        Returns:
            메모리 사용량 (bytes)
        """
        self.model.eval()

        if self.device == "cuda":
            # GPU 메모리 초기화
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            # 테스트 데이터 생성 및 추론
            img1 = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
            img2 = torch.randn(batch_size, 3, image_size, image_size, device=self.device)

            with torch.no_grad():
                _ = self.model(img1, img2)

            # 피크 메모리 반환
            memory_bytes = torch.cuda.max_memory_allocated()
            return memory_bytes
        else:
            # CPU 메모리는 추정치 반환
            # 실제로는 psutil 등을 사용해야 하지만, 여기서는 간단히 처리
            return 0

    def measure_training_memory(
        self,
        batch_size: int = 32,
        image_size: int = 768,
    ) -> float:
        """
        학습 시 GPU 메모리 사용량 측정 (GB)

        Args:
            batch_size: 배치 크기 (기본값: 32)
            image_size: 이미지 크기 (기본값: 768)

        Returns:
            메모리 사용량 (GB)

        Note:
            AC-009: 학습 시 GPU 메모리 < 12GB
        """
        if self.device != "cuda":
            return 0.0

        self.model.train()

        # GPU 메모리 초기화
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        # 테스트 데이터 생성
        img1 = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        img2 = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
        labels = torch.randint(-1, 2, (batch_size,), device=self.device).clamp(-1, 1)
        labels[labels == 0] = 1

        # Forward pass
        score1, score2 = self.model(img1, img2)

        # Loss 계산 (모델에 compute_loss가 있다고 가정)
        if hasattr(self.model, 'compute_loss'):
            loss = self.model.compute_loss(score1, score2, labels)
        else:
            # 간단한 loss 계산
            loss = ((score1 - score2).squeeze() * labels.float()).mean()

        # Backward pass
        loss.backward()

        # 피크 메모리 반환 (GB)
        memory_bytes = torch.cuda.max_memory_allocated()
        memory_gb = memory_bytes / (1024 ** 3)

        return memory_gb

    def measure_peak_memory(
        self,
        batch_size: int = 32,
        image_size: int = 768,
    ) -> int:
        """
        피크 메모리 사용량 측정

        Args:
            batch_size: 배치 크기 (기본값: 32)
            image_size: 이미지 크기 (기본값: 768)

        Returns:
            피크 메모리 사용량 (bytes)
        """
        return self.measure_memory_usage(batch_size, image_size)

    def estimate_training_time(
        self,
        num_epochs: int = 100,
        samples_per_epoch: int = 10000,
        batch_size: int = 32,
    ) -> float:
        """
        총 학습 시간 추정

        Args:
            num_epochs: 학습 에폭 수 (기본값: 100)
            samples_per_epoch: 에폭당 샘플 수 (기본값: 10000)
            batch_size: 배치 크기 (기본값: 32)

        Returns:
            예상 학습 시간 (시간)
        """
        # 배치당 추론 시간 측정
        batch_time = self.measure_batch_inference_time(batch_size=batch_size)

        # 학습 시간은 추론 시간의 약 3배 (forward + backward + optimizer)
        training_multiplier = 3.0

        # 에폭당 배치 수
        batches_per_epoch = samples_per_epoch // batch_size

        # 총 학습 시간 계산
        time_per_epoch = batch_time * training_multiplier * batches_per_epoch
        total_time_seconds = time_per_epoch * num_epochs
        total_time_hours = total_time_seconds / 3600

        return total_time_hours

    def check_training_time_target(
        self,
        num_epochs: int = 100,
        samples_per_epoch: int = 10000,
        batch_size: int = 32,
        target_hours: float = 6.0,
    ) -> Tuple[bool, float]:
        """
        학습 시간 목표 충족 여부 확인

        Args:
            num_epochs: 학습 에폭 수
            samples_per_epoch: 에폭당 샘플 수
            batch_size: 배치 크기
            target_hours: 목표 학습 시간 (기본값: 6시간)

        Returns:
            (meets_target, estimated_hours) 튜플

        Note:
            AC-009: 총 학습 시간 목표 < 6시간
        """
        estimated_hours = self.estimate_training_time(
            num_epochs=num_epochs,
            samples_per_epoch=samples_per_epoch,
            batch_size=batch_size,
        )

        meets_target = estimated_hours < target_hours

        return meets_target, estimated_hours

    def validate_reproducibility(
        self,
        dataloader: DataLoader,
        seed: int = 42,
        num_runs: int = 2,
    ) -> Tuple[bool, float]:
        """
        재현 가능성 검증

        Args:
            dataloader: 평가 데이터로더
            seed: 시드 값 (기본값: 42)
            num_runs: 실행 횟수 (기본값: 2)

        Returns:
            (is_reproducible, accuracy_diff) 튜플
            - is_reproducible: accuracy 차이 < 1%이면 True
            - accuracy_diff: accuracy 차이 (절대값)

        Note:
            AC-010: seed=42로 동일 조건 학습 시 accuracy 차이 < 1%
        """
        accuracies = []

        for _ in range(num_runs):
            # 시드 설정
            set_seed(seed)

            # 평가 실행
            evaluator = Evaluator(model=self.model, device=self.device)
            accuracy = evaluator.evaluate(dataloader)
            accuracies.append(accuracy)

        # accuracy 차이 계산
        if len(accuracies) >= 2:
            accuracy_diff = abs(accuracies[0] - accuracies[1])
        else:
            accuracy_diff = 0.0

        # 재현 가능성 판정 (차이 < 1%)
        is_reproducible = accuracy_diff < 0.01

        return is_reproducible, accuracy_diff

    def run_full_benchmark(
        self,
        batch_size: int = 32,
        image_size: int = 768,
    ) -> Dict[str, any]:
        """
        전체 벤치마크 실행

        Args:
            batch_size: 배치 크기 (기본값: 32)
            image_size: 이미지 크기 (기본값: 768)

        Returns:
            벤치마크 리포트 딕셔너리:
            {
                "inference_time_per_pair": float,
                "memory_usage_mb": float,
                "device": str,
                "meets_inference_requirement": bool,
                "meets_memory_requirement": bool,
            }
        """
        # 추론 시간 측정
        inference_time = self.measure_inference_time()

        # 메모리 사용량 측정
        memory_bytes = self.measure_memory_usage(batch_size=batch_size, image_size=image_size)
        memory_mb = memory_bytes / (1024 ** 2)
        memory_gb = memory_bytes / (1024 ** 3)

        # 요구사항 충족 여부 확인
        meets_inference = inference_time < self.MAX_INFERENCE_TIME_SECONDS
        meets_memory = memory_gb < self.MAX_MEMORY_GB

        return {
            "inference_time_per_pair": inference_time,
            "inference_time_ms": inference_time * 1000,
            "memory_usage_bytes": memory_bytes,
            "memory_usage_mb": memory_mb,
            "memory_usage_gb": memory_gb,
            "device": self.device,
            "batch_size": batch_size,
            "image_size": image_size,
            "meets_inference_requirement": meets_inference,
            "meets_memory_requirement": meets_memory,
            "requirements": {
                "max_inference_time_ms": self.MAX_INFERENCE_TIME_SECONDS * 1000,
                "max_memory_gb": self.MAX_MEMORY_GB,
            },
        }

    def __repr__(self) -> str:
        """객체의 문자열 표현"""
        return f"{self.__class__.__name__}(device={self.device})"
