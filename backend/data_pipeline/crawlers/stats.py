# -*- coding: utf-8 -*-
"""
크롤링 데이터 통계 리포트 생성 모듈

메타데이터 JSON 파일들을 분석하여 대학별, 학과별, 티어별 분포 등
종합적인 통계 리포트를 생성합니다.

사용법:
    python -m data_pipeline.crawlers.stats
    python -m data_pipeline.crawlers.stats --metadata-dir data/crawled/metadata
    python -m data_pipeline.crawlers.stats --format markdown --output report.md
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger("crawler.stats")


# 학과 코드 -> 한글 표시명 매핑
DEPARTMENT_DISPLAY_NAMES: Dict[str, str] = {
    "visual_design": "시각디자인",
    "industrial_design": "산업디자인",
    "craft": "공예",
    "fine_art": "회화/순수미술",
}


@dataclass
class StatsReport:
    """
    통계 리포트 데이터

    Attributes:
        total_posts: 전체 게시글 수
        posts_with_images: 이미지가 있는 게시글 수
        total_images: 전체 이미지 수 (메타데이터 기준)
        actual_image_files: 실제 이미지 파일 수 (디스크 기준)
        corrupted_images: 손상된 이미지 수
        tier_distribution: 티어별 게시글 분포 {tier: count}
        department_distribution: 학과별 분포 {dept_code: count}
        work_type_distribution: 작품 유형별 분포 {type: count}
        university_ranking: 대학별 게시글 수 (내림차순 정렬)
        year_distribution: 연도별 분포 {year: count}
        admission_type_distribution: 입시 유형별 분포 {type: count}
    """

    total_posts: int = 0
    posts_with_images: int = 0
    total_images: int = 0
    actual_image_files: int = 0
    corrupted_images: int = 0
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    department_distribution: Dict[str, int] = field(default_factory=dict)
    work_type_distribution: Dict[str, int] = field(default_factory=dict)
    university_ranking: List[Tuple[str, int]] = field(default_factory=list)
    year_distribution: Dict[str, int] = field(default_factory=dict)
    admission_type_distribution: Dict[str, int] = field(default_factory=dict)


class StatsReporter:
    """
    크롤링 데이터 통계 리포트 생성기

    메타데이터 JSON 파일들을 읽어 대학별, 학과별, 티어별 분포 등
    다양한 통계 정보를 계산하고 포맷합니다.
    """

    # 지원하는 이미지 확장자
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    def __init__(
        self,
        metadata_dir: Path,
        images_dir: Optional[Path] = None,
    ) -> None:
        """
        통계 리포트 생성기를 초기화합니다.

        Args:
            metadata_dir: 메타데이터 JSON 디렉토리 경로
            images_dir: 이미지 디렉토리 경로 (실제 파일 수 계산용, 선택사항)
        """
        self.metadata_dir = Path(metadata_dir)
        self.images_dir = Path(images_dir) if images_dir else None

    def generate(self) -> StatsReport:
        """
        전체 통계를 계산합니다.

        모든 메타데이터 JSON 파일을 읽어 티어, 학과, 작품 유형, 대학,
        연도, 입시 유형별 분포를 계산합니다.

        Returns:
            StatsReport: 계산된 통계 데이터
        """
        report = StatsReport()

        # 메타데이터 파일 목록
        metadata_files = sorted(self.metadata_dir.glob("*.json"))
        report.total_posts = len(metadata_files)

        if report.total_posts == 0:
            logger.warning(
                "메타데이터 파일이 없습니다",
                dir=str(self.metadata_dir),
            )
            return report

        # 카운터 초기화
        tier_counter: Counter = Counter()
        dept_counter: Counter = Counter()
        work_type_counter: Counter = Counter()
        uni_counter: Counter = Counter()
        year_counter: Counter = Counter()
        admission_counter: Counter = Counter()

        for json_path in metadata_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "메타데이터 읽기 실패",
                    path=str(json_path),
                    error=str(e),
                )
                continue

            # 이미지 수 집계
            images = meta.get("images", [])
            image_count = len(images)
            report.total_images += image_count
            if image_count > 0:
                report.posts_with_images += 1

            # 티어 분포 (빈 문자열이나 None은 C로 처리)
            tier = meta.get("tier", "C") or "C"
            tier_counter[tier] += 1

            # 학과 분포
            dept = meta.get("department", "")
            if dept:
                dept_counter[dept] += 1
            else:
                dept_counter["미분류"] += 1

            # 작품 유형 분포
            work_type = meta.get("work_type", "unknown") or "unknown"
            work_type_counter[work_type] += 1

            # 대학별 분포 (빈 문자열은 미분류)
            university = meta.get("university", "")
            if university:
                uni_counter[university] += 1

            # 연도별 분포
            year = meta.get("year", "")
            if year:
                year_counter[year] += 1
            else:
                year_counter["미분류"] += 1

            # 입시 유형 분포
            admission = meta.get("admission_type", "")
            if admission:
                admission_counter[admission] += 1
            else:
                admission_counter["미분류"] += 1

        # 결과 설정
        report.tier_distribution = dict(tier_counter.most_common())
        report.department_distribution = dict(dept_counter.most_common())
        report.work_type_distribution = dict(work_type_counter.most_common())
        report.university_ranking = uni_counter.most_common()
        report.year_distribution = dict(sorted(year_counter.items()))
        report.admission_type_distribution = dict(
            admission_counter.most_common()
        )

        # 실제 이미지 파일 수 (디스크 기준)
        if self.images_dir and self.images_dir.exists():
            report.actual_image_files = sum(
                1
                for f in self.images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
            )

        logger.info(
            "통계 생성 완료",
            total_posts=report.total_posts,
            posts_with_images=report.posts_with_images,
            total_images=report.total_images,
        )

        return report

    def to_console(self, report: Optional[StatsReport] = None) -> str:
        """
        터미널 출력용 포맷으로 변환합니다.

        Args:
            report: 통계 리포트 (None이면 자동 생성)

        Returns:
            str: 포맷된 리포트 문자열
        """
        if report is None:
            report = self.generate()

        lines: List[str] = []
        separator = "=" * 50

        # 헤더
        lines.append("")
        lines.append(separator)
        lines.append("  크롤링 데이터 통계 리포트")
        lines.append(separator)
        lines.append("")

        # 기본 통계
        lines.append(f"총 게시글: {report.total_posts:,}개")
        lines.append(f"이미지 있는 게시글: {report.posts_with_images:,}개")
        lines.append(f"총 이미지 (메타데이터): {report.total_images:,}장")
        if report.actual_image_files > 0:
            lines.append(f"실제 이미지 파일: {report.actual_image_files:,}장")
        lines.append("")

        # 티어 분포
        lines.append(separator)
        lines.append("  티어 분포")
        lines.append(separator)

        total_tier = sum(report.tier_distribution.values()) or 1
        max_tier_count = max(report.tier_distribution.values()) if report.tier_distribution else 1
        for tier in ["S", "A", "B", "C"]:
            count = report.tier_distribution.get(tier, 0)
            pct = 100.0 * count / total_tier
            bar = self._bar(count, max_tier_count)
            lines.append(f"  {tier}: {count:>6,} ({pct:5.1f}%) {bar}")
        lines.append("")

        # 학과별 분포
        lines.append(separator)
        lines.append("  학과별 분포")
        lines.append(separator)

        for dept, count in sorted(
            report.department_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            display_name = DEPARTMENT_DISPLAY_NAMES.get(dept, dept)
            if dept != display_name and dept != "미분류":
                lines.append(f"  {dept} ({display_name}): {count:>6,}")
            else:
                lines.append(f"  {display_name}: {count:>6,}")
        lines.append("")

        # 작품 유형 분포
        lines.append(separator)
        lines.append("  작품 유형")
        lines.append(separator)

        total_work = sum(report.work_type_distribution.values()) or 1
        for work_type, count in report.work_type_distribution.items():
            pct = 100.0 * count / total_work
            lines.append(f"  {work_type}: {count:>6,} ({pct:5.1f}%)")
        lines.append("")

        # 입시 유형 분포
        lines.append(separator)
        lines.append("  입시 유형")
        lines.append(separator)

        for adm_type, count in report.admission_type_distribution.items():
            lines.append(f"  {adm_type}: {count:>6,}")
        lines.append("")

        # 연도별 분포
        lines.append(separator)
        lines.append("  연도별 분포")
        lines.append(separator)

        for year, count in sorted(report.year_distribution.items()):
            lines.append(f"  {year}: {count:>6,}")
        lines.append("")

        # 대학별 상위 20
        lines.append(separator)
        lines.append("  대학별 상위 20")
        lines.append(separator)

        for rank, (uni, count) in enumerate(
            report.university_ranking[:20], 1
        ):
            lines.append(f"  {rank:>2}. {uni}: {count:>6,}")
        lines.append("")

        lines.append(separator)

        return "\n".join(lines)

    def to_markdown(self, report: Optional[StatsReport] = None) -> str:
        """
        마크다운 포맷으로 변환합니다.

        Args:
            report: 통계 리포트 (None이면 자동 생성)

        Returns:
            str: 마크다운 포맷 리포트 문자열
        """
        if report is None:
            report = self.generate()

        lines: List[str] = []

        # 헤더
        lines.append("# 크롤링 데이터 통계 리포트")
        lines.append("")

        # 기본 통계
        lines.append("## 기본 통계")
        lines.append("")
        lines.append("| 항목 | 수치 |")
        lines.append("|------|------|")
        lines.append(f"| 총 게시글 | {report.total_posts:,}개 |")
        lines.append(
            f"| 이미지 있는 게시글 | {report.posts_with_images:,}개 |"
        )
        lines.append(
            f"| 총 이미지 (메타데이터) | {report.total_images:,}장 |"
        )
        if report.actual_image_files > 0:
            lines.append(
                f"| 실제 이미지 파일 | {report.actual_image_files:,}장 |"
            )
        lines.append("")

        # 티어 분포
        lines.append("## 티어 분포")
        lines.append("")
        lines.append("| 티어 | 수 | 비율 |")
        lines.append("|------|---:|-----:|")

        total_tier = sum(report.tier_distribution.values()) or 1
        for tier in ["S", "A", "B", "C"]:
            count = report.tier_distribution.get(tier, 0)
            pct = 100.0 * count / total_tier
            lines.append(f"| {tier} | {count:,} | {pct:.1f}% |")
        lines.append("")

        # 학과별 분포
        lines.append("## 학과별 분포")
        lines.append("")
        lines.append("| 학과 | 수 |")
        lines.append("|------|---:|")

        for dept, count in sorted(
            report.department_distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            display_name = DEPARTMENT_DISPLAY_NAMES.get(dept, dept)
            lines.append(f"| {display_name} | {count:,} |")
        lines.append("")

        # 작품 유형
        lines.append("## 작품 유형")
        lines.append("")
        lines.append("| 유형 | 수 | 비율 |")
        lines.append("|------|---:|-----:|")

        total_work = sum(report.work_type_distribution.values()) or 1
        for work_type, count in report.work_type_distribution.items():
            pct = 100.0 * count / total_work
            lines.append(f"| {work_type} | {count:,} | {pct:.1f}% |")
        lines.append("")

        # 입시 유형
        lines.append("## 입시 유형")
        lines.append("")
        lines.append("| 유형 | 수 |")
        lines.append("|------|---:|")

        for adm_type, count in report.admission_type_distribution.items():
            lines.append(f"| {adm_type} | {count:,} |")
        lines.append("")

        # 연도별 분포
        lines.append("## 연도별 분포")
        lines.append("")
        lines.append("| 연도 | 수 |")
        lines.append("|------|---:|")

        for year, count in sorted(report.year_distribution.items()):
            lines.append(f"| {year} | {count:,} |")
        lines.append("")

        # 대학별 상위 20
        lines.append("## 대학별 상위 20")
        lines.append("")
        lines.append("| 순위 | 대학 | 수 |")
        lines.append("|-----:|------|---:|")

        for rank, (uni, count) in enumerate(
            report.university_ranking[:20], 1
        ):
            lines.append(f"| {rank} | {uni} | {count:,} |")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _bar(value: int, max_value: int, width: int = 20) -> str:
        """
        간단한 텍스트 막대 그래프를 생성합니다.

        Args:
            value: 현재 값
            max_value: 최대값
            width: 막대 최대 폭 (문자 수)

        Returns:
            str: 텍스트 막대 ('#' 문자 반복)
        """
        if max_value == 0:
            return ""
        filled = int(width * value / max_value)
        return "#" * filled


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    CLI 인자를 파싱합니다.

    Args:
        args: 커맨드라인 인자 목록 (테스트 시 주입용)

    Returns:
        argparse.Namespace: 파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description="크롤링 데이터 통계 리포트 생성 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 메타데이터 디렉토리에서 통계 생성
  python -m data_pipeline.crawlers.stats

  # 특정 디렉토리에서 통계 생성
  python -m data_pipeline.crawlers.stats --metadata-dir data/crawled/metadata

  # 마크다운 형식으로 출력
  python -m data_pipeline.crawlers.stats --format markdown

  # 파일로 저장
  python -m data_pipeline.crawlers.stats --output report.md --format markdown

  # 이미지 디렉토리 포함 (실제 파일 수 계산)
  python -m data_pipeline.crawlers.stats --images-dir data/crawled/raw_images
        """,
    )

    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="data/crawled/metadata",
        help="메타데이터 JSON 디렉토리 경로 (기본값: data/crawled/metadata)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="이미지 디렉토리 경로 (실제 파일 수 계산용, 선택사항)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["console", "markdown"],
        default="console",
        help="출력 형식 (기본값: console)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (지정하지 않으면 터미널 출력)",
    )

    return parser.parse_args(args)


def main() -> None:
    """통계 리포트 메인 진입점"""
    args = parse_args()

    metadata_dir = Path(args.metadata_dir)
    images_dir = Path(args.images_dir) if args.images_dir else None

    if not metadata_dir.exists():
        print(
            f"오류: 메타데이터 디렉토리를 찾을 수 없습니다: {metadata_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    reporter = StatsReporter(metadata_dir, images_dir)
    stats = reporter.generate()

    if args.format == "markdown":
        output = reporter.to_markdown(stats)
    else:
        output = reporter.to_console(stats)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"리포트 저장 완료: {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
