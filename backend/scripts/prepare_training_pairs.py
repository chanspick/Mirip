#!/usr/bin/env python3
"""
tier_score 기반 학습 페어 생성 스크립트

기존 이산 티어(S/A/B/C) 대신 tier_refined.tier_score (연속값 0-100) 활용.
같은 dept 내 페어 50%, cross-dept 페어 50%.
재현작 우선 샘플링, min tier_score gap >= 5.

산출물:
  - pairs_train.csv, pairs_val.csv (image_path_1, image_path_2, label 포함)
  - metadata_enhanced.csv (학습용 메타데이터)
  - pair_statistics.json (통계 리포트)

사용법:
  python backend/scripts/prepare_training_pairs.py
  python backend/scripts/prepare_training_pairs.py --dry-run
  python backend/scripts/prepare_training_pairs.py --total-pairs 100000
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

BACKEND_ROOT = Path(__file__).parents[1]
METADATA_DIR = BACKEND_ROOT / "data" / "crawled" / "metadata"
OUTPUT_DIR = BACKEND_ROOT / "training" / "data"


def load_eligible_items() -> list[dict]:
    """페어 생성 적격 항목 로드 (tier_score + image + anchor_group 필수)"""
    files = sorted(METADATA_DIR.glob("*.json"), key=lambda p: int(p.stem))
    items = []
    skipped = {"no_score": 0, "no_image": 0, "no_anchor": 0, "parse_error": 0}

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped["parse_error"] += 1
            continue

        # 필수 필드 검증
        tier_refined = d.get("tier_refined") or {}
        tier_score = tier_refined.get("tier_score")
        if not tier_score or tier_score <= 0:
            skipped["no_score"] += 1
            continue

        images = d.get("images", [])
        if not images:
            skipped["no_image"] += 1
            continue

        anchor_group = d.get("anchor_group")
        if not anchor_group:
            skipped["no_anchor"] += 1
            continue

        # exam_topic 추출
        parsed = d.get("interview_parsed") or {}
        exam_topic = parsed.get("exam_topic") or ""

        items.append({
            "post_no": d.get("post_no"),
            "image_path": images[0],  # 첫 번째 이미지 사용
            "tier": d.get("tier", ""),
            "tier_score": tier_score,
            "normalized_dept": d.get("normalized_dept", ""),
            "anchor_group": anchor_group,
            "university": d.get("university", ""),
            "work_type": d.get("work_type", "unknown"),
            "exam_topic": exam_topic[:100] if exam_topic else "",
        })

    print(f"  적격 항목: {len(items)}건")
    print(f"  제외: score 없음={skipped['no_score']}, "
          f"이미지 없음={skipped['no_image']}, "
          f"anchor 없음={skipped['no_anchor']}, "
          f"파싱 오류={skipped['parse_error']}")

    return items


def split_items_by_image(
    items: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """이미지 기준 train/val/test 분할 (데이터 누수 방지)"""
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_items = shuffled[:n_train]
    val_items = shuffled[n_train:n_train + n_val]
    test_items = shuffled[n_train + n_val:]

    return train_items, val_items, test_items


def generate_pairs(
    items: list[dict],
    total_pairs: int = 50000,
    same_dept_ratio: float = 0.5,
    min_score_gap: float = 5.0,
    max_appearances: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    tier_score 기반 페어 생성

    품질 층화 전략:
    - Tier 1: same dept + 둘 다 재현작 → 최고 품질
    - Tier 2: same dept + 재현작 1개 이상
    - Tier 3: same dept + 평소작만
    - Tier 4: cross-dept
    """
    rng = random.Random(seed)

    n_same = int(total_pairs * same_dept_ratio)
    n_cross = total_pairs - n_same

    # dept별 그룹핑
    dept_groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        dept_groups[item["normalized_dept"]].append(item)

    # === Same-dept 페어 생성 ===
    same_dept_candidates = _generate_same_dept_candidates(
        dept_groups, min_score_gap, rng
    )
    print(f"  Same-dept 후보: {len(same_dept_candidates)}건")

    # 품질별 정렬 (Tier 1 > Tier 2 > Tier 3)
    same_dept_candidates.sort(key=lambda p: -p["quality_tier"])

    # === Cross-dept 페어 생성 ===
    cross_dept_candidates = _generate_cross_dept_candidates(
        items, min_score_gap, rng
    )
    print(f"  Cross-dept 후보: {len(cross_dept_candidates)}건")

    # === 등장 횟수 제한 적용하며 선택 ===
    same_pairs = _select_with_appearance_limit(
        same_dept_candidates, n_same, max_appearances, rng
    )
    cross_pairs = _select_with_appearance_limit(
        cross_dept_candidates, n_cross, max_appearances, rng
    )

    all_pairs = same_pairs + cross_pairs
    rng.shuffle(all_pairs)

    return all_pairs


def _generate_same_dept_candidates(
    dept_groups: dict[str, list[dict]],
    min_score_gap: float,
    rng: random.Random,
) -> list[dict]:
    """같은 dept 내 페어 후보 생성"""
    candidates = []

    for dept, items in dept_groups.items():
        if len(items) < 2:
            continue

        # 모든 가능한 페어 중 tier_score gap >= min_score_gap인 것만
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                gap = abs(a["tier_score"] - b["tier_score"])

                if gap < min_score_gap:
                    continue

                # 높은 점수가 img1 (label=1)
                if a["tier_score"] > b["tier_score"]:
                    high, low = a, b
                else:
                    high, low = b, a

                # 품질 등급 계산
                quality = _compute_quality_tier(high, low)

                candidates.append({
                    "image_path_1": high["image_path"],
                    "image_path_2": low["image_path"],
                    "label": 1,
                    "tier_score_1": high["tier_score"],
                    "tier_score_2": low["tier_score"],
                    "score_gap": gap,
                    "pair_type": "same_dept",
                    "dept": dept,
                    "quality_tier": quality,
                    "post_no_1": high["post_no"],
                    "post_no_2": low["post_no"],
                })

                # 역방향도 추가
                candidates.append({
                    "image_path_1": low["image_path"],
                    "image_path_2": high["image_path"],
                    "label": -1,
                    "tier_score_1": low["tier_score"],
                    "tier_score_2": high["tier_score"],
                    "score_gap": gap,
                    "pair_type": "same_dept",
                    "dept": dept,
                    "quality_tier": quality,
                    "post_no_1": low["post_no"],
                    "post_no_2": high["post_no"],
                })

    rng.shuffle(candidates)
    return candidates


def _generate_cross_dept_candidates(
    items: list[dict],
    min_score_gap: float,
    rng: random.Random,
    max_candidates: int = 500000,
) -> list[dict]:
    """cross-dept 페어 후보 생성 (랜덤 샘플링)"""
    candidates = []
    n = len(items)

    # 전수 조합은 너무 많으므로 랜덤 샘플링
    attempts = 0
    max_attempts = max_candidates * 3

    while len(candidates) < max_candidates and attempts < max_attempts:
        attempts += 1
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue

        a, b = items[i], items[j]

        # 같은 dept면 스킵
        if a["normalized_dept"] == b["normalized_dept"]:
            continue

        gap = abs(a["tier_score"] - b["tier_score"])
        if gap < min_score_gap:
            continue

        # 높은 점수가 img1
        if a["tier_score"] > b["tier_score"]:
            high, low = a, b
        else:
            high, low = b, a

        candidates.append({
            "image_path_1": high["image_path"],
            "image_path_2": low["image_path"],
            "label": 1,
            "tier_score_1": high["tier_score"],
            "tier_score_2": low["tier_score"],
            "score_gap": gap,
            "pair_type": "cross_dept",
            "dept": f"{high['normalized_dept']}_vs_{low['normalized_dept']}",
            "quality_tier": 0,
            "post_no_1": high["post_no"],
            "post_no_2": low["post_no"],
        })

    rng.shuffle(candidates)
    return candidates


def _compute_quality_tier(high: dict, low: dict) -> int:
    """페어 품질 등급 계산 (높을수록 좋음)"""
    h_work = high.get("work_type", "unknown")
    l_work = low.get("work_type", "unknown")

    both_repro = h_work == "재현작" and l_work == "재현작"
    any_repro = h_work == "재현작" or l_work == "재현작"

    if both_repro:
        return 3  # Tier 1: 둘 다 재현작
    elif any_repro:
        return 2  # Tier 2: 재현작 1개
    else:
        return 1  # Tier 3: 평소작만 또는 unknown


def _select_with_appearance_limit(
    candidates: list[dict],
    target: int,
    max_appearances: int,
    rng: random.Random,
) -> list[dict]:
    """등장 횟수 제한을 적용하며 페어 선택"""
    selected = []
    counts: Counter = Counter()

    for pair in candidates:
        if len(selected) >= target:
            break

        p1 = pair["image_path_1"]
        p2 = pair["image_path_2"]

        if counts[p1] < max_appearances and counts[p2] < max_appearances:
            selected.append(pair)
            counts[p1] += 1
            counts[p2] += 1

    return selected


def save_pairs_csv(pairs: list[dict], path: Path) -> None:
    """페어를 CSV로 저장"""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path_1", "image_path_2", "label",
            "tier_score_1", "tier_score_2", "score_gap",
            "pair_type", "dept", "quality_tier",
        ])
        writer.writeheader()
        for p in pairs:
            writer.writerow({k: p[k] for k in writer.fieldnames})


def save_metadata_csv(items: list[dict], path: Path) -> None:
    """메타데이터를 CSV로 저장 (학습 호환 형식)"""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_path", "tier", "tier_score", "normalized_dept",
            "anchor_group", "university", "work_type", "post_no",
        ])
        writer.writeheader()
        for item in items:
            writer.writerow({k: item[k] for k in writer.fieldnames})


def compute_statistics(
    pairs: list[dict],
    items: list[dict],
    split_name: str,
) -> dict:
    """통계 계산"""
    if not pairs:
        return {"split": split_name, "total_pairs": 0}

    same_dept = [p for p in pairs if p["pair_type"] == "same_dept"]
    cross_dept = [p for p in pairs if p["pair_type"] == "cross_dept"]

    quality_dist = Counter(p["quality_tier"] for p in pairs)
    gap_values = [p["score_gap"] for p in pairs]

    # dept별 분포
    dept_dist = Counter()
    for p in same_dept:
        dept_dist[p["dept"]] += 1

    return {
        "split": split_name,
        "total_pairs": len(pairs),
        "same_dept_pairs": len(same_dept),
        "cross_dept_pairs": len(cross_dept),
        "same_dept_ratio": round(len(same_dept) / len(pairs) * 100, 1),
        "quality_distribution": {
            "tier1_both_repro": quality_dist.get(3, 0),
            "tier2_one_repro": quality_dist.get(2, 0),
            "tier3_no_repro": quality_dist.get(1, 0),
            "tier4_cross_dept": quality_dist.get(0, 0),
        },
        "score_gap": {
            "mean": round(sum(gap_values) / len(gap_values), 1),
            "min": round(min(gap_values), 1),
            "max": round(max(gap_values), 1),
        },
        "unique_images": len(set(
            p["image_path_1"] for p in pairs
        ) | set(
            p["image_path_2"] for p in pairs
        )),
        "eligible_items": len(items),
        "top_dept_pairs": dict(dept_dist.most_common(10)),
    }


def print_report(stats: dict) -> None:
    """통계 리포트 출력"""
    print(f"\n{'=' * 60}")
    print(f"PAIR GENERATION REPORT ({stats['split']})")
    print(f"{'=' * 60}")

    print(f"\n[기본 통계]")
    print(f"  적격 항목:       {stats['eligible_items']:,}건")
    print(f"  생성 페어:       {stats['total_pairs']:,}건")
    print(f"  Same-dept:       {stats['same_dept_pairs']:,}건 ({stats['same_dept_ratio']}%)")
    print(f"  Cross-dept:      {stats['cross_dept_pairs']:,}건")
    print(f"  고유 이미지:     {stats['unique_images']:,}건")

    qd = stats["quality_distribution"]
    print(f"\n[품질 층화]")
    print(f"  Tier 1 (둘 다 재현작): {qd['tier1_both_repro']:,}건")
    print(f"  Tier 2 (재현작 1개):   {qd['tier2_one_repro']:,}건")
    print(f"  Tier 3 (평소작만):     {qd['tier3_no_repro']:,}건")
    print(f"  Tier 4 (cross-dept):   {qd['tier4_cross_dept']:,}건")

    sg = stats["score_gap"]
    print(f"\n[tier_score gap]")
    print(f"  평균: {sg['mean']}, 최소: {sg['min']}, 최대: {sg['max']}")

    if stats["top_dept_pairs"]:
        print(f"\n[Same-dept 상위 10개]")
        for dept, count in stats["top_dept_pairs"].items():
            print(f"  {dept:<30} {count:>6}건")


def main():
    parser = argparse.ArgumentParser(description="tier_score 기반 학습 페어 생성")
    parser.add_argument("--dry-run", action="store_true", help="통계만 출력, 저장 안함")
    parser.add_argument("--total-pairs", type=int, default=50000, help="총 페어 수 (기본: 50000)")
    parser.add_argument("--same-dept-ratio", type=float, default=0.5, help="same-dept 비율 (기본: 0.5)")
    parser.add_argument("--min-gap", type=float, default=5.0, help="최소 tier_score 차이 (기본: 5.0)")
    parser.add_argument("--max-appearances", type=int, default=20, help="이미지당 최대 등장 횟수 (기본: 20)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="train 비율")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="val 비율")
    args = parser.parse_args()

    print("적격 항목 로드 중...")
    start = time.time()
    items = load_eligible_items()
    print(f"  소요: {time.time() - start:.1f}초")

    print(f"\n이미지 기준 train/val/test 분할 중...")
    train_items, val_items, test_items = split_items_by_image(
        items, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Train: {len(train_items)}건, Val: {len(val_items)}건, Test: {len(test_items)}건")

    # Train 페어 생성
    n_train_pairs = int(args.total_pairs * args.train_ratio)
    n_val_pairs = int(args.total_pairs * args.val_ratio)

    print(f"\nTrain 페어 생성 중 (목표: {n_train_pairs:,})...")
    train_pairs = generate_pairs(
        train_items,
        total_pairs=n_train_pairs,
        same_dept_ratio=args.same_dept_ratio,
        min_score_gap=args.min_gap,
        max_appearances=args.max_appearances,
        seed=args.seed,
    )

    print(f"\nVal 페어 생성 중 (목표: {n_val_pairs:,})...")
    val_pairs = generate_pairs(
        val_items,
        total_pairs=n_val_pairs,
        same_dept_ratio=args.same_dept_ratio,
        min_score_gap=args.min_gap,
        max_appearances=args.max_appearances,
        seed=args.seed + 1,
    )

    # 통계
    train_stats = compute_statistics(train_pairs, train_items, "train")
    val_stats = compute_statistics(val_pairs, val_items, "val")

    print_report(train_stats)
    print_report(val_stats)

    if args.dry_run:
        print(f"\n[DRY RUN] 저장하지 않음")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        save_pairs_csv(train_pairs, OUTPUT_DIR / "pairs_train.csv")
        save_pairs_csv(val_pairs, OUTPUT_DIR / "pairs_val.csv")
        save_metadata_csv(train_items, OUTPUT_DIR / "metadata_train.csv")
        save_metadata_csv(val_items, OUTPUT_DIR / "metadata_val.csv")
        save_metadata_csv(test_items, OUTPUT_DIR / "metadata_test.csv")

        # 통계 저장
        with open(OUTPUT_DIR / "pair_statistics.json", "w", encoding="utf-8") as f:
            json.dump({"train": train_stats, "val": val_stats}, f, ensure_ascii=False, indent=2)

        print(f"\n저장 완료:")
        print(f"  {OUTPUT_DIR / 'pairs_train.csv'}")
        print(f"  {OUTPUT_DIR / 'pairs_val.csv'}")
        print(f"  {OUTPUT_DIR / 'metadata_train.csv'}")
        print(f"  {OUTPUT_DIR / 'metadata_val.csv'}")
        print(f"  {OUTPUT_DIR / 'metadata_test.csv'}")
        print(f"  {OUTPUT_DIR / 'pair_statistics.json'}")


if __name__ == "__main__":
    main()
