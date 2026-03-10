#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
"""
학습 스크립트 - Pairwise Ranking Model 학습을 위한 메인 엔트리포인트.

Usage (기존 방식 - discrete tier):
    python train.py --metadata_csv data/metadata.csv --output_dir checkpoints/

Usage (Exp 1 - tier_score 기반 pre-computed pairs):
    python train.py --pairs_csv training/data/pairs_train.csv \
                    --val_pairs_csv training/data/pairs_val.csv \
                    --image_dir ../backend/data/crawled \
                    --output_dir checkpoints/dinov2_exp1/ \
                    --lr 5e-5 --weight_decay 0.05 --batch_size 64 --epochs 50
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

    # 데이터 소스 (둘 중 하나 필수)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--metadata_csv", type=str, help="메타데이터 CSV 경로 (기존 방식)")
    data_group.add_argument("--pairs_csv", type=str, help="Pre-computed train pairs CSV 경로 (Exp 1)")

    parser.add_argument("--val_pairs_csv", type=str, default=None, help="Pre-computed val pairs CSV 경로 (--pairs_csv 사용 시 필수)")
    parser.add_argument("--output_dir", type=str, required=True, help="체크포인트 저장 디렉토리")
    parser.add_argument("--epochs", type=int, default=100, help="최대 에폭 수")
    parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4, help="학습률")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.3, help="드롭아웃 비율")
    parser.add_argument("--margin", type=float, default=0.3, help="MarginRankingLoss margin")
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


def pairwise_collate_fn(batch):
    """DataLoader용 collate 함수 - 3-tuple (모듈 레벨 - Windows 멀티프로세싱 호환)"""
    i1, i2, l = zip(*batch)
    return torch.stack(i1), torch.stack(i2), torch.tensor(l, dtype=torch.long)


def precomputed_collate_fn(batch):
    """DataLoader용 collate 함수 - 4-tuple (PrecomputedPairDataset용)"""
    i1, i2, l, sd = zip(*batch)
    return (
        torch.stack(i1),
        torch.stack(i2),
        torch.tensor(l, dtype=torch.long),
        torch.tensor(sd, dtype=torch.long),
    )


def create_loaders_legacy(train_df, val_df, image_dir, batch_size, num_workers):
    """기존 방식: metadata → PairwiseDataset"""
    train_ds = PairwiseDataset(metadata_df=train_df, image_dir=image_dir, is_train=True)
    val_ds = PairwiseDataset(metadata_df=val_df, image_dir=image_dir, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=pairwise_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=pairwise_collate_fn)
    return train_loader, val_loader


def create_loaders_precomputed(train_csv, val_csv, image_dir, batch_size, num_workers):
    """Exp 1 방식: pre-computed pairs CSV → PrecomputedPairDataset"""
    from training.datasets.precomputed_pair_dataset import PrecomputedPairDataset

    train_ds = PrecomputedPairDataset(pairs_csv=train_csv, image_dir=image_dir, is_train=True)
    val_ds = PrecomputedPairDataset(pairs_csv=val_csv, image_dir=image_dir, is_train=False)

    logger.info("Pre-computed pairs 로딩 완료", train_pairs=len(train_ds), val_pairs=len(val_ds),
                train_same_dept=train_ds.num_same_dept, val_same_dept=val_ds.num_same_dept)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=precomputed_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=precomputed_collate_fn)
    return train_loader, val_loader


def print_summary(args, model, train_loader, val_loader):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sep = "=" * 60
    mode = "Pre-computed Pairs (Exp 1)" if args.pairs_csv else "Legacy (Discrete Tier)"
    print(f"\n{sep}")
    print(f"Pairwise Ranking Model 학습 설정 [{mode}]")
    print(sep)
    print(f"디바이스: {args.device}, 시드: {args.seed}")
    print(f"전체 파라미터: {total:,}, 학습 가능: {trainable:,}")
    print(f"에폭: {args.epochs}, 배치: {args.batch_size}, lr: {args.lr}")
    print(f"weight_decay: {args.weight_decay}, dropout: {args.dropout}, margin: {args.margin}")
    print(f"학습 배치: {len(train_loader)}, 검증 배치: {len(val_loader)}")
    print(f"{sep}\n")


def main():
    args = parse_args()
    try:
        set_seed(args.seed)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 데이터 로딩 분기
        if args.pairs_csv:
            # Exp 1: Pre-computed pairs 모드
            if not args.val_pairs_csv:
                raise ValueError("--pairs_csv 사용 시 --val_pairs_csv도 필요합니다")
            if not args.image_dir:
                raise ValueError("--pairs_csv 사용 시 --image_dir이 필요합니다")

            train_loader, val_loader = create_loaders_precomputed(
                args.pairs_csv, args.val_pairs_csv, args.image_dir,
                args.batch_size, args.num_workers,
            )
        else:
            # 기존 방식: metadata CSV
            df = load_metadata(args.metadata_csv)
            splitter = DataSplitter(metadata_df=df, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
            train_df, val_df, test_df = splitter.split()
            logger.info("데이터 분할 완료", train=len(train_df), val=len(val_df), test=len(test_df))

            test_df.to_csv(output_dir / "test_metadata.csv", index=False)
            train_loader, val_loader = create_loaders_legacy(
                train_df, val_df, args.image_dir, args.batch_size, args.num_workers,
            )

        model = PairwiseRankingModel(
            feature_extractor_model="facebook/dinov2-large",
            dropout=args.dropout,
            margin=args.margin,
        )
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
        same_dept_info = ""
        if history.get("same_dept_accuracy"):
            same_dept_acc = history["same_dept_accuracy"][-1]
            same_dept_info = f", Same-Dept Acc: {same_dept_acc:.4f}"

        print(f"\n학습 완료! Val Acc: {val_acc:.4f}{same_dept_info}")
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
