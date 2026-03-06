#!/usr/bin/env python3
"""
CLIP 주제 해석 점수 증강 스크립트

exam_topic이 있는 이미지에 대해:
  CLIP text encoder(exam_topic) vs CLIP image encoder(작품 이미지)
  → cosine similarity → topic_score (0~1)

주제를 잘 표현한 그림은 높은 점수, 주제에서 벗어난 그림은 낮은 점수.

산출물:
  - 각 메타데이터에 clip_topic 필드 추가
    - topic_score: cosine similarity (0~1)
    - topic_text: 사용된 주제 텍스트

사용법:
  python backend/scripts/augment_clip_topic.py
  python backend/scripts/augment_clip_topic.py --batch-size 32
  python backend/scripts/augment_clip_topic.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

BACKEND_ROOT = Path(__file__).parents[1]
PROJECT_ROOT = Path(__file__).parents[2]
METADATA_DIR = BACKEND_ROOT / "data" / "crawled" / "metadata"


def load_clip_model(device: str):
    """CLIP 모델 로드"""
    import open_clip

    # 한국어 텍스트도 어느 정도 처리 가능한 multilingual 모델 사용
    # ViT-B-32는 가벼우면서 범용적
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    return model, preprocess, tokenizer


def compute_topic_score(
    model,
    preprocess,
    tokenizer,
    image_path: Path,
    topic_text: str,
    device: str,
) -> Optional[float]:
    """단일 이미지-주제 쌍의 cosine similarity 계산"""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # 주제 텍스트 전처리 (너무 길면 잘라냄)
        topic_text = topic_text[:200].strip()
        text_tokens = tokenizer([topic_text]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)

            # L2 정규화
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # cosine similarity
            similarity = (image_features @ text_features.T).item()

        # 0~1 범위로 클램핑 (cosine similarity는 -1~1이지만 CLIP에서는 보통 0~1)
        score = max(0.0, min(1.0, (similarity + 1) / 2))  # [-1,1] → [0,1]
        return round(score, 4)

    except Exception as e:
        print(f"  ERROR {image_path.name}: {e}")
        return None


def resolve_image_path(images: list[str]) -> Optional[Path]:
    """이미지 경로 해석"""
    if not images:
        return None
    # 첫 번째 이미지 사용
    rel_path = images[0]
    # 상대 경로를 절대 경로로 변환
    abs_path = BACKEND_ROOT / rel_path
    if abs_path.exists():
        return abs_path
    # PROJECT_ROOT에서도 시도
    abs_path = PROJECT_ROOT / rel_path
    if abs_path.exists():
        return abs_path
    return None


def main():
    parser = argparse.ArgumentParser(description="CLIP 주제 해석 점수 증강")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기 (현재 1 고정)")
    parser.add_argument("--dry-run", action="store_true", help="계산만 하고 저장하지 않음")
    parser.add_argument("--limit", type=int, default=0, help="처리할 최대 건수 (0=전체)")
    args = parser.parse_args()

    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")

    # CLIP 모델 로드
    print("CLIP 모델 로드 중...")
    model, preprocess, tokenizer = load_clip_model(device)
    print("  모델 로드 완료")

    # 메타데이터 로드
    files = sorted(METADATA_DIR.glob("*.json"), key=lambda p: int(p.stem))
    print(f"총 메타데이터: {len(files)}건")

    if args.limit > 0:
        files = files[:args.limit]

    stats = {
        "total": len(files),
        "has_topic": 0,
        "has_image": 0,
        "processed": 0,
        "skipped_no_topic": 0,
        "skipped_no_image": 0,
        "errors": 0,
        "scores": [],
    }

    start_time = time.time()

    for i, filepath in enumerate(files):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            speed = stats["processed"] / elapsed if elapsed > 0 else 0
            print(f"[{i+1}/{len(files)}] 처리={stats['processed']}, "
                  f"속도={speed:.1f}건/초, "
                  f"평균={np.mean(stats['scores']):.3f}" if stats['scores'] else "")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats["errors"] += 1
            continue

        # exam_topic 확인
        parsed = data.get("interview_parsed", {})
        exam_topic = parsed.get("exam_topic") if parsed else None

        if not exam_topic or len(exam_topic.strip()) < 5:
            stats["skipped_no_topic"] += 1
            continue
        stats["has_topic"] += 1

        # 이미지 경로 확인
        images = data.get("images", [])
        image_path = resolve_image_path(images)

        if not image_path:
            stats["skipped_no_image"] += 1
            continue
        stats["has_image"] += 1

        # CLIP 점수 계산
        score = compute_topic_score(model, preprocess, tokenizer, image_path, exam_topic, device)

        if score is not None:
            data["clip_topic"] = {
                "topic_score": score,
                "topic_text": exam_topic[:200],
            }
            stats["processed"] += 1
            stats["scores"].append(score)

            if not args.dry_run:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            stats["errors"] += 1

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("CLIP 주제 해석 점수 증강 완료")
    print("=" * 60)
    print(f"총 파일: {stats['total']}건")
    print(f"주제 있음: {stats['has_topic']}건")
    print(f"이미지 있음: {stats['has_image']}건")
    print(f"처리 완료: {stats['processed']}건")
    print(f"건너뜀 (주제 없음): {stats['skipped_no_topic']}건")
    print(f"건너뜀 (이미지 없음): {stats['skipped_no_image']}건")
    print(f"오류: {stats['errors']}건")
    print(f"소요 시간: {elapsed:.1f}초")

    if stats["scores"]:
        scores = stats["scores"]
        print(f"\n점수 분포:")
        print(f"  평균: {np.mean(scores):.4f}")
        print(f"  중앙값: {np.median(scores):.4f}")
        print(f"  표준편차: {np.std(scores):.4f}")
        print(f"  최소: {min(scores):.4f}")
        print(f"  최대: {max(scores):.4f}")

        # 구간별 분포
        bins = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        for j in range(len(bins) - 1):
            count = sum(1 for s in scores if bins[j] <= s < bins[j + 1])
            pct = count / len(scores) * 100
            print(f"  [{bins[j]:.1f}, {bins[j+1]:.1f}): {count}건 ({pct:.1f}%)")

    if args.dry_run:
        print("\n[DRY RUN] 저장하지 않음")


if __name__ == "__main__":
    main()
