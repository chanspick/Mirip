#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# prepare_data.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
"""
데이터 준비 스크립트 - 이미지 디렉토리 스캔 및 메타데이터 CSV 생성.

Usage:
    python prepare_data.py --input_dir data/images --output_csv data/metadata.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import structlog
from tqdm import tqdm

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

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VALID_TIERS = {"S", "A", "B", "C"}


def parse_args():
    parser = argparse.ArgumentParser(description="데이터 준비 스크립트", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, required=True, help="이미지 디렉토리 경로")
    parser.add_argument("--output_csv", type=str, required=True, help="출력 메타데이터 CSV 경로")
    parser.add_argument("--tier_mode", type=str, default="directory", choices=["directory", "filename", "manual"], help="Tier 할당 방식")
    parser.add_argument("--default_tier", type=str, default="B", choices=list(VALID_TIERS), help="기본 Tier")
    parser.add_argument("--validate", action="store_true", help="이미지 파일 검증")
    parser.add_argument("--min_size", type=int, default=224, help="최소 이미지 크기")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    return parser.parse_args()


def scan_images(input_dir: Path) -> List[Path]:
    images = []
    for ext in VALID_EXTENSIONS:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def extract_tier_from_path(image_path: Path, mode: str, default_tier: str) -> str:
    if mode == "directory":
        for part in image_path.parts:
            part_upper = part.upper()
            if part_upper in VALID_TIERS:
                return part_upper
    elif mode == "filename":
        name = image_path.stem.upper()
        for tier in VALID_TIERS:
            if f"_{tier}_" in name or name.startswith(f"{tier}_") or name.endswith(f"_{tier}"):
                return tier
    return default_tier


def validate_image(image_path: Path, min_size: int) -> Tuple[bool, Optional[str]]:
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
            if width < min_size or height < min_size:
                return False, f"크기 미달: {width}x{height}"
            if img.mode not in ("RGB", "RGBA", "L"):
                return False, f"지원하지 않는 모드: {img.mode}"
            return True, None
    except Exception as e:
        return False, f"로드 실패: {str(e)}"


def create_metadata(images: List[Path], input_dir: Path, tier_mode: str, default_tier: str, validate: bool, min_size: int) -> List[Dict]:
    metadata = []
    skipped = []
    
    for image_path in tqdm(images, desc="이미지 처리 중"):
        if validate:
            valid, reason = validate_image(image_path, min_size)
            if not valid:
                skipped.append((str(image_path), reason))
                continue
        
        tier = extract_tier_from_path(image_path, tier_mode, default_tier)
        rel_path = image_path.relative_to(input_dir)
        
        metadata.append({
            "image_path": str(rel_path),
            "tier": tier,
            "filename": image_path.name,
        })
    
    if skipped:
        logger.warning("건너뛴 이미지", count=len(skipped))
    
    return metadata


def save_metadata(metadata: List[Dict], output_csv: Path):
    import pandas as pd
    df = pd.DataFrame(metadata)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("메타데이터 저장 완료", path=str(output_csv), samples=len(df))


def print_statistics(metadata: List[Dict]):
    if not metadata:
        print("데이터가 없습니다.")
        return
    
    tier_counts = Counter(item["tier"] for item in metadata)
    
    sep = "=" * 60
    print(f"\n{sep}")
    print("데이터 통계")
    print(sep)
    print(f"전체 이미지 수: {len(metadata):,}")
    print("-" * 60)
    print("Tier 분포:")
    for tier in ["S", "A", "B", "C"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(metadata) * 100 if metadata else 0
        bar = "#" * int(pct / 2)
        print(f"  {tier}: {count:>6,} ({pct:>5.1f}%) {bar}")
    print(f"{sep}\n")


def main():
    args = parse_args()
    try:
        input_dir = Path(args.input_dir)
        output_csv = Path(args.output_csv)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"입력 디렉토리 없음: {input_dir}")
        
        logger.info("이미지 스캔 시작", input_dir=str(input_dir))
        images = scan_images(input_dir)
        logger.info("이미지 발견", count=len(images))
        
        if not images:
            logger.warning("이미지를 찾을 수 없습니다")
            return 1
        
        metadata = create_metadata(images, input_dir, args.tier_mode, args.default_tier, args.validate, args.min_size)
        
        if not metadata:
            logger.warning("유효한 이미지가 없습니다")
            return 1
        
        save_metadata(metadata, output_csv)
        print_statistics(metadata)
        
        return 0
        
    except Exception as e:
        logger.error("오류 발생", error=str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
