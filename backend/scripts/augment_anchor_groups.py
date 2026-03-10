#!/usr/bin/env python3
"""
앵커 그룹 태깅 스크립트

메타데이터에 anchor_group 필드를 추가합니다.
anchor_group = "{대학명}_{normalized_dept}" 형태의 키로,
대학-학과별 앵커 비교 시스템의 기반 데이터입니다.

산출물:
  - 각 메타데이터에 anchor_group 필드 추가
  - 15건 미만 그룹은 anchor_group = null 처리
  - 그룹별 통계 리포트 출력

사용법:
  python backend/scripts/augment_anchor_groups.py
  python backend/scripts/augment_anchor_groups.py --dry-run
  python backend/scripts/augment_anchor_groups.py --min-group-size 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# normalizer 직접 임포트 (__init__.py의 pandas 의존성 우회)
BACKEND_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "normalizer",
    BACKEND_ROOT / "data_pipeline" / "crawlers" / "normalizer.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_anchor_group_key = _mod.make_anchor_group_key

METADATA_DIR = BACKEND_ROOT / "data" / "crawled" / "metadata"


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


def compute_anchor_groups(
    data_list: list[dict],
    min_group_size: int = 15,
) -> tuple[list[dict], dict]:
    """앵커 그룹 키 생성 및 유효 그룹 필터링"""

    # 1단계: 모든 항목에 anchor_group 키 생성
    group_counter: Counter = Counter()
    for d in data_list:
        university = d.get("university", "")
        normalized_dept = d.get("normalized_dept", "")
        key = make_anchor_group_key(university, normalized_dept)
        d["_anchor_group_raw"] = key
        if key:
            group_counter[key] += 1

    # 2단계: 유효 그룹 필터링 (min_group_size 이상만)
    valid_groups = {k for k, v in group_counter.items() if v >= min_group_size}

    # 3단계: anchor_group 필드 할당
    assigned = 0
    unassigned_small = 0
    unassigned_empty = 0

    for d in data_list:
        raw_key = d.pop("_anchor_group_raw")
        if not raw_key:
            d["anchor_group"] = None
            unassigned_empty += 1
        elif raw_key in valid_groups:
            d["anchor_group"] = raw_key
            assigned += 1
        else:
            d["anchor_group"] = None
            unassigned_small += 1

    # 통계 생성
    stats = {
        "total": len(data_list),
        "assigned": assigned,
        "unassigned_small_group": unassigned_small,
        "unassigned_empty_key": unassigned_empty,
        "coverage": round(assigned / len(data_list) * 100, 1) if data_list else 0,
        "total_raw_groups": len(group_counter),
        "valid_groups": len(valid_groups),
        "min_group_size": min_group_size,
        "group_details": {
            k: v for k, v in sorted(group_counter.items(), key=lambda x: -x[1])
        },
        "valid_group_details": {
            k: v
            for k, v in sorted(group_counter.items(), key=lambda x: -x[1])
            if k in valid_groups
        },
    }

    return data_list, stats


def save_results(data_list: list[dict]) -> int:
    """결과를 메타데이터 파일에 저장"""
    saved = 0
    for d in data_list:
        if "anchor_group" not in d:
            continue

        path = d.pop("_path")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        saved += 1

    return saved


def print_report(stats: dict) -> None:
    """통계 리포트 출력"""
    print("\n" + "=" * 70)
    print("ANCHOR GROUP TAGGING REPORT")
    print("=" * 70)

    print(f"\n[기본 통계]")
    print(f"  총 메타데이터:     {stats['total']:,}건")
    print(f"  그룹 할당됨:       {stats['assigned']:,}건 ({stats['coverage']}%)")
    print(f"  소규모 그룹 제외:  {stats['unassigned_small_group']:,}건")
    print(f"  키 생성 불가:      {stats['unassigned_empty_key']:,}건")

    print(f"\n[그룹 통계]")
    print(f"  전체 그룹 수:      {stats['total_raw_groups']}개")
    print(f"  유효 그룹 수:      {stats['valid_groups']}개 (>= {stats['min_group_size']}건)")

    print(f"\n[유효 그룹 목록] ({stats['valid_groups']}개)")
    print(f"  {'그룹 키':<35} {'건수':>6}")
    print(f"  {'-'*35} {'-'*6}")
    for group, count in stats["valid_group_details"].items():
        print(f"  {group:<35} {count:>6}")

    # 미달 그룹 중 상위 10개 표시
    small_groups = {
        k: v
        for k, v in stats["group_details"].items()
        if k not in stats["valid_group_details"]
    }
    if small_groups:
        print(f"\n[미달 그룹 상위 10개] (< {stats['min_group_size']}건)")
        print(f"  {'그룹 키':<35} {'건수':>6}")
        print(f"  {'-'*35} {'-'*6}")
        for group, count in list(small_groups.items())[:10]:
            print(f"  {group:<35} {count:>6}")


def main():
    parser = argparse.ArgumentParser(description="앵커 그룹 태깅")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="계산만 하고 저장하지 않음",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=15,
        help="유효 그룹 최소 건수 (기본값: 15)",
    )
    args = parser.parse_args()

    print("메타데이터 로드 중...")
    start = time.time()
    data_list = load_all_metadata()
    print(f"  {len(data_list)}건 로드 ({time.time() - start:.1f}초)")

    print(f"\n앵커 그룹 계산 중 (min_group_size={args.min_group_size})...")
    data_list, stats = compute_anchor_groups(data_list, args.min_group_size)

    print_report(stats)

    if args.dry_run:
        print(f"\n[DRY RUN] 저장하지 않음")
    else:
        saved = save_results(data_list)
        print(f"\n{saved}건 저장 완료")


if __name__ == "__main__":
    main()
