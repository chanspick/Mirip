#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
"""
학습 스크립트 - Pairwise Ranking Model 학습을 위한 메인 엔트리포인트.

Usage:
    python train.py --metadata_csv data/metadata.csv --output_dir checkpoints/
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import structlog
import torch
from torch.utils.data import DataLoader

from app.ml.ranking_model import PairwiseRankingModel
from training.config import TrainingConfig
from training.trainer import Trainer
from training.datasets.pairwise_dataset import PairwiseDataset
from training.datasets.data_splitter import DataSplitter
from training.benchmarks import set_seed

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pairwise Ranking Model 학습", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata_csv", type=str, required=True, help="메타데이터 CSV 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="체크포인트 저장 디렉토리")
    parser.add_argument("--epochs", type=int, default=100, help="최대 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="mirip-ranking")
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5)
    return parser.parse_args()


def load_metadata(csv_path):
    import pandas as pd
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"파일 없음: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"image_path", "tier"}
    if required - set(df.columns):
        raise ValueError(f"필수 컬럼 누락: {required - set(df.columns)}")
    df = df[df["tier"].isin({"S", "A", "B", "C"})]
    logger.info("메타데이터 로딩 완료", samples=len(df))
    return df


def create_loaders(train_df, val_df, image_dir, batch_size, num_workers):
    train_ds = PairwiseDataset(metadata_df=train_df, image_dir=image_dir)
    val_ds = PairwiseDataset(metadata_df=val_df, image_dir=image_dir)
    
    def collate(batch):
        i1, i2, l = zip(*batch)
        return torch.stack(i1), torch.stack(i2), torch.tensor(l, dtype=torch.long)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    return train_loader, val_loader


def print_summary(args, model, train_loader, val_loader):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sep = "=" * 60
    print(f"\n{sep}")
    print("Pairwise Ranking Model 학습 설정")
    print(sep)
    print(f"디바이스: {args.device}, 시드: {args.seed}")
    print(f"전체 파라미터: {total:,}, 학습 가능: {trainable:,}")
    print(f"에폭: {args.epochs}, 배치: {args.batch_size}, lr: {args.lr}")
    print(f"학습 배치: {len(train_loader)}, 검증 배치: {len(val_loader)}")
    print(f"{sep}\n")


def main():
    args = parse_args()
    try:
        set_seed(args.seed)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = load_metadata(args.metadata_csv)
        splitter = DataSplitter(train_ratio=args.train_ratio, val_ratio=args.val_ratio, random_state=args.seed)
        train_df, val_df, test_df = splitter.split(df)
        logger.info("데이터 분할 완료", train=len(train_df), val=len(val_df), test=len(test_df))
        
        test_df.to_csv(output_dir / "test_metadata.csv", index=False)
        train_loader, val_loader = create_loaders(train_df, val_df, args.image_dir, args.batch_size, args.num_workers)
        
        model = PairwiseRankingModel(backbone_name="dinov2_vitl14", freeze_backbone=True)
        config = TrainingConfig(
            learning_rate=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
            max_epochs=args.epochs, early_stopping_patience=args.patience,
            checkpoint_dir=str(output_dir), save_every_n_epochs=args.save_every,
            wandb_project=args.wandb_project, wandb_run_name=args.wandb_run,
            wandb_enabled=not args.no_wandb, device=args.device, seed=args.seed, num_workers=args.num_workers,
        )
        
        print_summary(args, model, train_loader, val_loader)
        trainer = Trainer(model=model, config=config, resume_from=args.resume)
        logger.info("학습 시작")
        history = trainer.train(train_loader, val_loader)
        
        import json
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        val_acc = history["val_accuracy"][-1]
        print(f"\n학습 완료! 최종 정확도: {val_acc:.4f}")
        print(f"체크포인트: {output_dir}")
        return 0
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단됨")
        return 1
    except Exception as e:
        logger.error("오류 발생", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
