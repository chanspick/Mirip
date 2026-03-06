#!/usr/bin/env python3
"""
티어 라벨 보정 스크립트

경쟁률 데이터를 활용하여 기존 티어 라벨에 soft score를 부여합니다.
같은 B티어라도 경쟁률 81:1 합격은 상위 B, 20:1은 하위 B로 구분합니다.

산출물:
  - 각 메타데이터에 tier_refined 필드 추가
    - tier_score: 0~100 연속 점수
    - tier_confidence: 보정 신뢰도
    - competition_percentile: 같은 티어 내 경쟁률 백분위

사용법:
  python backend/scripts/augment_tier_labels.py
  python backend/scripts/augment_tier_labels.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

BACKEND_ROOT = Path(__file__).parents[1]
METADATA_DIR = BACKEND_ROOT / "data" / "crawled" / "metadata"

# 대학 티어 매핑 (기존 데이터의 tier 필드 활용)
TIER_BASE_SCORES = {
    "S": 90,
    "A": 72,
    "B": 55,
    "C": 38,
}

# 티어 내 점수 범위
TIER_RANGES = {
    "S": (82, 98),
    "A": (65, 84),
    "B": (48, 68),
    "C": (30, 50),
}


def load_all_metadata() -> list[dict]:
    """모든 메타데이터 로드"""
    files = sorted(METADATA_DIR.glob("*.json"), key=lambda p: int(p.stem))
    data_list = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
                d["_path"] = str(f)
                data_list.append(d)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return data_list


def compute_tier_scores(data_list: list[dict]) -> list[dict]:
    """경쟁률 기반 티어 점수 계산"""

    # 1단계: 티어별 경쟁률 분포 수집
    tier_ratios: dict[str, list[float]] = defaultdict(list)
    for d in data_list:
        tier = d.get("tier")
        parsed = d.get("interview_parsed", {})
        ratio = parsed.get("competition_ratio") if parsed else None
        if tier and ratio and ratio > 0:
            tier_ratios[tier].append(ratio)

    # 2단계: 티어별 경쟁률 통계 계산
    tier_stats: dict[str, dict] = {}
    for tier, ratios in tier_ratios.items():
        if len(ratios) >= 5:
            tier_stats[tier] = {
                "mean": statistics.mean(ratios),
                "median": statistics.median(ratios),
                "stdev": statistics.stdev(ratios) if len(ratios) > 1 else 1.0,
                "min": min(ratios),
                "max": max(ratios),
                "count": len(ratios),
                "sorted": sorted(ratios),
            }

    print("티어별 경쟁률 통계:")
    for tier in ["S", "A", "B", "C"]:
        if tier in tier_stats:
            s = tier_stats[tier]
            print(f"  {tier}: mean={s['mean']:.1f}, median={s['median']:.1f}, "
                  f"stdev={s['stdev']:.1f}, range=[{s['min']:.1f}, {s['max']:.1f}], n={s['count']}")

    # 3단계: 각 항목에 tier_score 계산
    for d in data_list:
        tier = d.get("tier")
        parsed = d.get("interview_parsed", {})
        ratio = parsed.get("competition_ratio") if parsed else None

        if not tier or tier not in TIER_BASE_SCORES:
            d["tier_refined"] = {
                "tier_score": None,
                "tier_confidence": 0.0,
                "competition_percentile": None,
            }
            continue

        base_score = TIER_BASE_SCORES[tier]
        score_min, score_max = TIER_RANGES[tier]

        if ratio and ratio > 0 and tier in tier_stats:
            stats = tier_stats[tier]
            sorted_ratios = stats["sorted"]

            # 경쟁률 백분위 계산 (높은 경쟁률 = 높은 백분위)
            below = sum(1 for r in sorted_ratios if r <= ratio)
            percentile = below / len(sorted_ratios)

            # 백분위를 점수 범위에 매핑
            tier_score = score_min + percentile * (score_max - score_min)
            tier_score = round(max(score_min, min(score_max, tier_score)), 1)

            confidence = 0.8  # 경쟁률 있으면 높은 신뢰도
        else:
            # 경쟁률 없으면 기본 점수 사용
            tier_score = float(base_score)
            percentile = None
            confidence = 0.4  # 경쟁률 없으면 낮은 신뢰도

        d["tier_refined"] = {
            "tier_score": tier_score,
            "tier_confidence": confidence,
            "competition_percentile": round(percentile, 3) if percentile is not None else None,
        }

    return data_list


def save_results(data_list: list[dict], dry_run: bool = False) -> int:
    """결과를 메타데이터 파일에 저장"""
    saved = 0
    for d in data_list:
        if "tier_refined" not in d:
            continue

        path = d.pop("_path")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(description="티어 라벨 보정 (경쟁률 기반)")
    parser.add_argument("--dry-run", action="store_true", help="계산만 하고 저장하지 않음")
    args = parser.parse_args()

    print("메타데이터 로드 중...")
    start = time.time()
    data_list = load_all_metadata()
    print(f"  {len(data_list)}건 로드 ({time.time() - start:.1f}초)")

    print("\n티어 점수 계산 중...")
    data_list = compute_tier_scores(data_list)

    # 통계 출력
    has_score = sum(1 for d in data_list if d.get("tier_refined", {}).get("tier_score") is not None)
    has_ratio = sum(1 for d in data_list if d.get("tier_refined", {}).get("competition_percentile") is not None)

    print(f"\n결과:")
    print(f"  tier_score 생성: {has_score}/{len(data_list)}건")
    print(f"  경쟁률 기반 보정: {has_ratio}/{len(data_list)}건")
    print(f"  경쟁률 없음 (기본 점수): {has_score - has_ratio}건")

    # 티어별 점수 분포
    print("\n티어별 tier_score 분포:")
    for tier in ["S", "A", "B", "C"]:
        scores = [
            d["tier_refined"]["tier_score"]
            for d in data_list
            if d.get("tier") == tier and d.get("tier_refined", {}).get("tier_score") is not None
        ]
        if scores:
            print(f"  {tier}: mean={statistics.mean(scores):.1f}, "
                  f"min={min(scores):.1f}, max={max(scores):.1f}, n={len(scores)}")

    if args.dry_run:
        print("\n[DRY RUN] 저장하지 않음")
    else:
        saved = save_results(data_list)
        print(f"\n{saved}건 저장 완료")


if __name__ == "__main__":
    main()
