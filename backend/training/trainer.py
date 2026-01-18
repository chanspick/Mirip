# trainer.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: GREEN - Implementation
"""
Trainer Module

Pairwise Ranking Model 학습을 위한 Trainer 클래스.

Acceptance Criteria (AC-006, AC-008):
- AdamW optimizer (lr=1e-4, weight_decay=0.01)
- CosineAnnealingLR scheduler
- Early stopping (patience=10)
- 체크포인트 저장 (model, optimizer, epoch)
- 학습 재개 기능
- wandb 로깅 (train/loss, val/loss, val/accuracy)

Example:
    >>> trainer = Trainer(model=model, config=config)
    >>> history = trainer.train(train_loader, val_loader)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from training.config import TrainingConfig


class Trainer:
    """
    Pairwise Ranking Model Trainer

    학습 루프, 검증, 체크포인트 관리, wandb 로깅을 처리합니다.

    Attributes:
        model: 학습할 모델
        config: 학습 설정
        optimizer: AdamW optimizer
        scheduler: CosineAnnealingLR scheduler
        device: 학습 디바이스

    Args:
        model: PairwiseRankingModel 인스턴스
        config: TrainingConfig 인스턴스
        resume_from: 재개할 체크포인트 경로 (선택)

    Example:
        >>> trainer = Trainer(model, config)
        >>> history = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        resume_from: Optional[str] = None,
    ) -> None:
        """
        Trainer 초기화

        Args:
            model: 학습할 모델
            config: 학습 설정
            resume_from: 재개할 체크포인트 경로
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # 모델을 디바이스로 이동
        self.model = self.model.to(self.device)

        # Optimizer 설정 (AdamW)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler 설정 (CosineAnnealingLR)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.scheduler_t_max or config.max_epochs,
            eta_min=config.scheduler_eta_min,
        )

        # Early stopping 상태
        self._best_val_loss = float('inf')
        self._patience_counter = 0
        self._val_losses: List[float] = []

        # 학습 상태
        self._current_epoch = 0
        self._global_step = 0

        # wandb 초기화
        self._init_wandb()

        # 체크포인트에서 재개
        if resume_from:
            self.load_checkpoint(resume_from)

    def _init_wandb(self) -> None:
        """wandb 초기화"""
        if not self.config.wandb_enabled or wandb is None:
            return

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            config=self.config.to_dict(),
        )

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        """
        단일 에폭 학습

        Args:
            train_loader: 학습 데이터로더

        Returns:
            평균 학습 loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            img1, img2, labels = batch
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            score1, score2 = self.model(img1, img2)

            # Compute loss
            loss = self.model.compute_loss(score1, score2, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self._global_step += 1

            # Step-level wandb logging
            if self.config.wandb_enabled and wandb is not None:
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "global_step": self._global_step,
                })

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        검증 수행

        Args:
            val_loader: 검증 데이터로더

        Returns:
            (검증 loss, 검증 accuracy) 튜플
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                img1, img2, labels = batch
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                score1, score2 = self.model(img1, img2)

                # Compute loss
                loss = self.model.compute_loss(score1, score2, labels)
                total_loss += loss.item()

                # Compute accuracy
                predictions = torch.sign(score1 - score2).squeeze()
                correct = (predictions == labels.float()).sum().item()
                total_correct += correct
                total_samples += labels.size(0)

        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = total_correct / max(total_samples, 1)

        return avg_loss, accuracy

    def _compute_accuracy(
        self,
        score1: torch.Tensor,
        score2: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        정확도 계산

        Args:
            score1: 첫 번째 이미지 스코어
            score2: 두 번째 이미지 스코어
            labels: 정답 레이블

        Returns:
            정확도 (0.0 ~ 1.0)
        """
        predictions = torch.sign(score1 - score2).squeeze()
        correct = (predictions == labels.float()).sum().item()
        accuracy = correct / labels.size(0)
        return accuracy

    def _check_early_stopping(self, current_val_loss: float) -> bool:
        """
        Early stopping 체크

        Args:
            current_val_loss: 현재 검증 loss

        Returns:
            True이면 학습 중단
        """
        self._val_losses.append(current_val_loss)

        if current_val_loss < self._best_val_loss - self.config.early_stopping_min_delta:
            # 개선됨
            self._best_val_loss = current_val_loss
            self._patience_counter = 0
            return False
        else:
            # 개선되지 않음
            self._patience_counter += 1
            if self._patience_counter >= self.config.early_stopping_patience:
                return True
            return False

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        전체 학습 루프

        Args:
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더

        Returns:
            학습 히스토리 딕셔너리
            {
                "train_loss": [...],
                "val_loss": [...],
                "val_accuracy": [...]
            }
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(self._current_epoch, self.config.max_epochs):
            self._current_epoch = epoch

            # Train
            train_loss = self.train_one_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_accuracy = self.validate(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            # Scheduler step
            self.scheduler.step()

            # Epoch-level wandb logging
            if self.config.wandb_enabled and wandb is not None:
                wandb.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                    "epoch": epoch,
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                })

            # 로그 출력
            print(
                f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_accuracy:.4f}"
            )

            # 체크포인트 저장
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch)

            # Best model 저장
            if val_loss < self._best_val_loss:
                self.save_checkpoint(epoch, is_best=True)

            # Early stopping 체크
            if self._check_early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # wandb 종료
        if self.config.wandb_enabled and wandb is not None:
            wandb.finish()

        return history

    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        as_artifact: bool = False,
    ) -> str:
        """
        체크포인트 저장

        Args:
            epoch: 현재 에폭
            is_best: 최고 성능 모델 여부
            as_artifact: wandb artifact로 저장 여부

        Returns:
            저장된 체크포인트 경로
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self._best_val_loss,
            "patience_counter": self._patience_counter,
            "global_step": self._global_step,
            "config": self.config.to_dict(),
        }

        # 파일명 결정
        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = Path(self.config.checkpoint_dir) / filename
        torch.save(checkpoint, checkpoint_path)

        # wandb artifact로 저장
        if as_artifact and self.config.wandb_enabled and wandb is not None:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch}",
                type="model",
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        체크포인트 로드

        Args:
            checkpoint_path: 체크포인트 파일 경로

        Returns:
            로드된 에폭 번호
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self._best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self._patience_counter = checkpoint.get("patience_counter", 0)
        self._global_step = checkpoint.get("global_step", 0)
        self._current_epoch = checkpoint["epoch"]

        return checkpoint["epoch"]

    def __repr__(self) -> str:
        """객체의 문자열 표현"""
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"epoch={self._current_epoch}, "
            f"best_val_loss={self._best_val_loss:.4f})"
        )
