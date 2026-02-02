#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_ranker.py
# 티어 판정 테스트 스크립트
"""
Tier Ranker 테스트

앵커 기반 티어 판정을 테스트합니다.

Usage:
    python scripts/test_ranker.py \
        --checkpoint checkpoints/dinov2_pairwise/best_model.pt \
        --anchors anchors/anchors.pt \
        --image path/to/test_image.jpg

    # 여러 이미지 테스트
    python scripts/test_ranker.py \
        --checkpoint checkpoints/dinov2_pairwise/best_model.pt \
        --anchors anchors/anchors.pt \
        --image_dir data/crawled/raw_images \
        --metadata data/train_metadata.csv \
        --n_test 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from app.ml.ranking_model import PairwiseRankingModel
from app.ml.anchor_manager import AnchorManager
from app.ml.tier_ranker import TierRanker


def parse_args():
    parser = argparse.ArgumentParser(
        description="티어 판정 테스트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--anchors", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="단일 이미지 경로")
    parser.add_argument("--image_dir", type=str, default=".", help="이미지 디렉토리")
    parser.add_argument("--metadata", type=str, default=None, help="테스트용 메타데이터")
    parser.add_argument("--n_test", type=int, default=20, help="테스트할 이미지 수")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> PairwiseRankingModel:
    model = PairwiseRankingModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()

    print("Loading model...")
    model = load_model(args.checkpoint, args.device)

    print("Loading anchors...")
    anchor_manager = AnchorManager.load(args.anchors, model, args.device)

    print("Creating ranker...")
    ranker = TierRanker(model, anchor_manager, args.device)

    # 단일 이미지 테스트
    if args.image:
        print(f"\n{'='*60}")
        print(f"Testing: {args.image}")
        print('='*60)

        result = ranker.rank(args.image)
        print(ranker.explain(result))
        return 0

    # 메타데이터 기반 다중 테스트
    if args.metadata:
        print(f"\nLoading test metadata from {args.metadata}...")
        df = pd.read_csv(args.metadata)
        df = df[df['tier'].isin(['S', 'A', 'B', 'C'])]

        # 각 티어에서 균등하게 샘플링
        test_samples = []
        n_per_tier = args.n_test // 4

        for tier in ['S', 'A', 'B', 'C']:
            tier_df = df[df['tier'] == tier]
            samples = tier_df.sample(n=min(n_per_tier, len(tier_df)), random_state=42)
            test_samples.append(samples)

        test_df = pd.concat(test_samples)

        print(f"\nTesting {len(test_df)} images...")
        print("="*60)

        correct = 0
        confusion = {t: {t2: 0 for t2 in ['S', 'A', 'B', 'C']} for t in ['S', 'A', 'B', 'C']}

        for _, row in test_df.iterrows():
            img_path = Path(args.image_dir) / row['image_path']
            true_tier = row['tier']

            if not img_path.exists():
                print(f"Warning: {img_path} not found")
                continue

            result = ranker.rank(str(img_path))
            pred_tier = result['tier']

            is_correct = pred_tier == true_tier
            correct += int(is_correct)
            confusion[true_tier][pred_tier] += 1

            mark = "✅" if is_correct else "❌"
            print(f"{mark} True: {true_tier}, Pred: {pred_tier} (conf: {result['confidence']:.2f}) - {row['image_path']}")

        # 결과 요약
        total = len(test_df)
        accuracy = correct / total if total > 0 else 0

        print("\n" + "="*60)
        print(f"ACCURACY: {correct}/{total} = {accuracy:.1%}")
        print("="*60)

        print("\nConfusion Matrix:")
        print("        ", end="")
        for t in ['S', 'A', 'B', 'C']:
            print(f"  {t:>4}", end="")
        print("  (Pred)")

        for true_t in ['S', 'A', 'B', 'C']:
            print(f"  {true_t:>4} ", end="")
            for pred_t in ['S', 'A', 'B', 'C']:
                print(f"  {confusion[true_t][pred_t]:>4}", end="")
            print()
        print("(True)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
