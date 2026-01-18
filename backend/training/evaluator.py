# evaluator.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: GREEN - Implementation
"""
Evaluator Module

Pairwise Ranking Model 평가를 위한 Evaluator 클래스.

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

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    """
    Pairwise Ranking Model Evaluator

    학습된 모델의 Pairwise accuracy를 평가합니다.

    Attributes:
        model: 평가할 모델
        device: 평가 디바이스

    Args:
        model: PairwiseRankingModel 인스턴스
        device: 평가 디바이스 (기본값: cuda if available else cpu)

    Example:
        >>> evaluator = Evaluator(model, device="cuda")
        >>> accuracy = evaluator.evaluate(test_loader)
        >>> print(f"Test Accuracy: {accuracy:.2%}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ) -> None:
        """
        Evaluator 초기화

        Args:
            model: 평가할 모델
            device: 평가 디바이스
        """
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 모델을 디바이스로 이동하고 eval 모드로 설정
        self.model = self.model.to(self.device)
        self.model.eval()

    def evaluate(self, dataloader: DataLoader) -> float:
        """
        데이터로더의 모든 데이터에 대해 Pairwise accuracy 계산

        Args:
            dataloader: 평가 데이터로더 (img1, img2, labels)

        Returns:
            accuracy: 0.0 ~ 1.0 사이의 정확도

        Note:
            AC-007: 정확도 = (올바른 예측 수) / (전체 페어 수)
            - label=1: img1이 더 높은 품질 (score1 > score2 예측)
            - label=-1: img2가 더 높은 품질 (score1 < score2 예측)
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                img1, img2, labels = batch
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                score1, score2 = self.model(img1, img2)

                # 예측 계산: sign(score1 - score2)
                # score1 > score2 -> 1 (img1이 더 좋음)
                # score1 < score2 -> -1 (img2가 더 좋음)
                # score1 == score2 -> 0 (동점)
                predictions = torch.sign(score1 - score2).squeeze(-1)

                # 정확도 계산
                correct = (predictions == labels.float()).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

        # 빈 데이터로더 처리
        if total_samples == 0:
            return 0.0

        accuracy = total_correct / total_samples
        return accuracy

    def evaluate_detailed(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        상세 평가 결과 반환

        Args:
            dataloader: 평가 데이터로더

        Returns:
            상세 메트릭 딕셔너리:
            {
                "accuracy": 0.75,
                "total_pairs": 100,
                "correct_predictions": 75
            }
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                img1, img2, labels = batch
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                score1, score2 = self.model(img1, img2)

                # 예측 및 정확도 계산
                predictions = torch.sign(score1 - score2).squeeze(-1)
                correct = (predictions == labels.float()).sum().item()

                total_correct += correct
                total_samples += labels.size(0)

        # 빈 데이터로더 처리
        if total_samples == 0:
            return {
                "accuracy": 0.0,
                "total_pairs": 0,
                "correct_predictions": 0,
            }

        accuracy = total_correct / total_samples

        return {
            "accuracy": accuracy,
            "total_pairs": total_samples,
            "correct_predictions": int(total_correct),
        }

    def predict_pair(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[int, float, float]:
        """
        단일 페어 예측

        Args:
            img1: 첫 번째 이미지 (1, 3, H, W)
            img2: 두 번째 이미지 (1, 3, H, W)

        Returns:
            (prediction, score1, score2) 튜플
            - prediction: 1 (img1 > img2), -1 (img1 < img2), 0 (동점)
            - score1: img1의 품질 스코어
            - score2: img2의 품질 스코어
        """
        self.model.eval()

        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        with torch.no_grad():
            score1, score2 = self.model(img1, img2)

        score1_val = score1.item()
        score2_val = score2.item()

        diff = score1_val - score2_val
        if diff > 0:
            prediction = 1
        elif diff < 0:
            prediction = -1
        else:
            prediction = 0

        return prediction, score1_val, score2_val

    def __repr__(self) -> str:
        """객체의 문자열 표현"""
        return f"{self.__class__.__name__}(device={self.device})"
