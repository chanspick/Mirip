#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# build_anchors.py
# 앵커 feature 생성 스크립트
"""
앵커 Feature 생성

학습된 모델을 사용하여 각 티어별 앵커 이미지의 feature를 추출하고 저장합니다.

Usage:
    python scripts/build_anchors.py \
        --checkpoint checkpoints/dinov2_pairwise/best_model.pt \
        --metadata data/train_metadata.csv \
        --image_dir . \
        --output anchors/anchors.pt \
        --n_per_tier 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from app.ml.ranking_model import PairwiseRankingModel
from app.ml.anchor_manager import AnchorManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="앵커 Feature 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="학습된 모델 체크포인트 경로"
    )
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="메타데이터 CSV 경로"
    )
    parser.add_argument(
        "--image_dir", type=str, default=".",
        help="이미지 디렉토리"
    )
    parser.add_argument(
        "--output", type=str, default="anchors/anchors.pt",
        help="앵커 저장 경로"
    )
    parser.add_argument(
        "--n_per_tier", type=int, default=10,
        help="티어당 앵커 이미지 수"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="연산 디바이스"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> PairwiseRankingModel:
    """체크포인트에서 모델 로드"""
    print(f"Loading model from {checkpoint_path}...")

    model = PairwiseRankingModel()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 체크포인트 형식에 따라 처리
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    return model


def main():
    args = parse_args()

    # 메타데이터 로드
    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata)
    df = df[df['tier'].isin(['S', 'A', 'B', 'C'])]

    print(f"Total images: {len(df)}")
    print(f"Tier distribution:")
    for tier in ['S', 'A', 'B', 'C']:
        count = len(df[df['tier'] == tier])
        print(f"  {tier}: {count}")

    # 모델 로드
    model = load_model(args.checkpoint, args.device)

    # 앵커 매니저 생성
    anchor_manager = AnchorManager(model=model, device=args.device)

    # 앵커 빌드
    print(f"\nBuilding anchors ({args.n_per_tier} per tier)...")
    anchor_manager.build_anchors(
        metadata_df=df,
        image_dir=args.image_dir,
        n_per_tier=args.n_per_tier,
        seed=args.seed,
    )

    # 저장
    anchor_manager.save(args.output)

    print("\n✅ Anchor features saved successfully!")
    print(f"   Path: {args.output}")
    print(f"   Total anchors: {sum(len(p) for p in anchor_manager._anchor_paths.values())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
