# -*- coding: utf-8 -*-
"""
데이터 정제 오케스트레이터

크롤링된 데이터의 전체 정제 프로세스를 관리합니다:
1. 손상 이미지 필터링
2. 메타데이터 정규화 (대학명/학과명 - 정규식 기반)
3. 중복 이미지 제거 (perceptual hash)
4. 통계 리포트 생성

사용법:
    python -m data_pipeline.crawlers.cleaner
    python -m data_pipeline.crawlers.cleaner --data-dir data/crawled
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import structlog

from data_pipeline.crawlers.dedup import DeduplicationResult, ImageDeduplicator
from data_pipeline.crawlers.normalizer import (
    determine_tier,
    normalize_department,
    normalize_university,
)
from data_pipeline.crawlers.stats import StatsReporter

logger = structlog.get_logger("crawler.cleaner")


@dataclass
class CleaningResult:
    """
    데이터 정제 결과

    Attributes:
        total_posts: 전체 게시글 수
        valid_posts: 유효한 게시글 수 (이미지가 1개 이상)
        invalid_images: 손상된 이미지 수
        duplicates_removed: 중복 제거된 이미지 수
        normalized_count: 정규화로 변경된 메타데이터 수
    """

    total_posts: int = 0
    valid_posts: int = 0
    invalid_images: int = 0
    duplicates_removed: int = 0
    normalized_count: int = 0

    def summary(self) -> str:
        """결과 요약 문자열을 반환합니다."""
        return (
            f"전체 게시글: {self.total_posts}, "
            f"유효: {self.valid_posts}, "
            f"손상 이미지: {self.invalid_images}, "
            f"중복 제거: {self.duplicates_removed}, "
            f"정규화 변경: {self.normalized_count}"
        )


class DataCleaner:
    """
    데이터 정제 오케스트레이터

    크롤링된 이미지와 메타데이터를 정제하는 전체 파이프라인을 실행합니다.

    파이프라인 순서:
    1. 손상 이미지 검증 및 삭제
    2. 대학명/학과명/티어 정규식 기반 재정규화
    3. perceptual hash 중복 이미지 제거
    4. 메타데이터에서 삭제된 이미지 참조 정리
    5. 통계 리포트 생성
    """

    # 지원하는 이미지 확장자
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    def __init__(
        self,
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        정제기를 초기화합니다.

        Args:
            data_dir: 크롤링 데이터 기본 디렉토리 (기본값: data/crawled)
            output_dir: 정제 결과 출력 디렉토리
                        (기본값: data_dir과 동일 - 인플레이스 정제)
        """
        if data_dir:
            base = Path(data_dir)
        else:
            base = Path("data/crawled")

        self.images_dir = base / "raw_images"
        self.metadata_dir = base / "metadata"

        # 출력 디렉토리 (기본값: 인플레이스 정제)
        if output_dir:
            out_base = Path(output_dir)
            self.output_images_dir = out_base / "raw_images"
            self.output_metadata_dir = out_base / "metadata"
            self.output_images_dir.mkdir(parents=True, exist_ok=True)
            self.output_metadata_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_images_dir = self.images_dir
            self.output_metadata_dir = self.metadata_dir

        logger.info(
            "데이터 정제기 초기화",
            images_dir=str(self.images_dir),
            metadata_dir=str(self.metadata_dir),
        )

    def clean(self) -> CleaningResult:
        """
        전체 데이터 정제 파이프라인을 실행합니다.

        실행 순서:
        1. 손상 이미지 검증 및 삭제
        2. 메타데이터 정규화 (대학명/학과명 - 정규식 기반)
        3. 중복 이미지 제거 (perceptual hash)
        4. 메타데이터에서 삭제된 이미지 참조 정리
        5. 통계 리포트 생성

        Returns:
            CleaningResult: 정제 결과
        """
        result = CleaningResult()

        # 전체 게시글 수 확인
        metadata_files = list(self.metadata_dir.glob("*.json"))
        result.total_posts = len(metadata_files)

        logger.info(
            "데이터 정제 시작",
            total_posts=result.total_posts,
        )

        # === 1단계: 손상 이미지 검증 ===
        logger.info("1단계: 손상 이미지 검증 중...")
        invalid_images = self._validate_images()
        result.invalid_images = len(invalid_images)

        # 손상 이미지 삭제
        for img_path in invalid_images:
            try:
                img_path.unlink()
                logger.debug("손상 이미지 삭제", path=str(img_path))
            except OSError as e:
                logger.warning(
                    "손상 이미지 삭제 실패",
                    path=str(img_path),
                    error=str(e),
                )

        logger.info(
            "1단계 완료: 손상 이미지 처리",
            invalid=result.invalid_images,
        )

        # === 2단계: 메타데이터 정규화 ===
        logger.info("2단계: 메타데이터 정규화 중...")
        result.normalized_count = self._normalize_metadata()
        logger.info(
            "2단계 완료: 메타데이터 정규화",
            changed=result.normalized_count,
        )

        # === 3단계: 중복 이미지 제거 ===
        logger.info("3단계: 중복 이미지 제거 중...")
        dedup_result = self._deduplicate_images()
        result.duplicates_removed = dedup_result.duplicates_removed
        logger.info(
            "3단계 완료: 중복 이미지 제거",
            removed=result.duplicates_removed,
        )

        # === 4단계: 메타데이터 이미지 참조 정리 ===
        logger.info("4단계: 메타데이터 이미지 참조 정리 중...")
        self._clean_metadata_image_refs()
        logger.info("4단계 완료: 이미지 참조 정리")

        # 유효 게시글 수 계산 (이미지가 1개 이상인 게시글)
        result.valid_posts = self._count_valid_posts()

        # === 5단계: 통계 리포트 생성 ===
        logger.info("5단계: 통계 리포트 생성 중...")
        report = self._generate_report()
        logger.info("5단계 완료: 리포트 생성")

        logger.info(
            "데이터 정제 완료",
            summary=result.summary(),
        )

        print("\n" + report)

        return result

    def _validate_images(self) -> List[Path]:
        """
        이미지 파일의 무결성을 검증합니다.

        PIL의 verify() 메서드로 이미지 파일이 손상되지 않았는지 확인합니다.
        빈 파일(0바이트)도 손상으로 처리합니다.

        Returns:
            List[Path]: 손상된 이미지 파일 경로 목록
        """
        from PIL import Image

        invalid: List[Path] = []

        if not self.images_dir.exists():
            logger.warning(
                "이미지 디렉토리가 존재하지 않음",
                dir=str(self.images_dir),
            )
            return invalid

        image_files = [
            f
            for f in sorted(self.images_dir.iterdir())
            if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
        ]

        total = len(image_files)
        logger.info("이미지 검증 시작", total=total)

        for idx, img_path in enumerate(image_files):
            if (idx + 1) % 1000 == 0:
                logger.info(
                    "이미지 검증 진행 중",
                    progress=f"{idx + 1}/{total}",
                )

            # 빈 파일 체크
            try:
                if img_path.stat().st_size == 0:
                    invalid.append(img_path)
                    logger.debug("빈 이미지 파일 감지", path=str(img_path))
                    continue
            except OSError:
                invalid.append(img_path)
                continue

            # PIL 무결성 검증
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                invalid.append(img_path)
                logger.debug(
                    "손상 이미지 감지",
                    path=str(img_path),
                    error=str(e),
                )

        logger.info(
            "이미지 검증 완료",
            total=total,
            invalid=len(invalid),
        )

        return invalid

    def _normalize_metadata(self) -> int:
        """
        모든 메타데이터 JSON의 대학명과 학과명을 정규식 기반으로 재정규화합니다.

        기존 크롤러의 딕셔너리 매핑에서 놓친 대학명을 정규식 규칙으로
        표준 축약형으로 변환하고, 티어를 재결정합니다.

        Returns:
            int: 변경된 메타데이터 파일 수
        """
        changed_count = 0
        metadata_files = sorted(self.metadata_dir.glob("*.json"))
        total = len(metadata_files)

        for idx, json_path in enumerate(metadata_files):
            if (idx + 1) % 1000 == 0:
                logger.info(
                    "메타데이터 정규화 진행 중",
                    progress=f"{idx + 1}/{total}",
                )

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "메타데이터 읽기 실패",
                    path=str(json_path),
                    error=str(e),
                )
                continue

            changed = False

            # 대학명 재정규화
            old_uni = metadata.get("university", "")
            if old_uni:
                new_uni = normalize_university(old_uni)
                if new_uni != old_uni:
                    metadata["university"] = new_uni
                    changed = True
                    logger.debug(
                        "대학명 정규화 변경",
                        post_no=metadata.get("post_no"),
                        old=old_uni,
                        new=new_uni,
                    )

            # 티어 재결정 (대학명 변경 여부와 무관하게 항상 확인)
            new_tier = determine_tier(metadata.get("university", ""))
            if new_tier != metadata.get("tier", "C"):
                metadata["tier"] = new_tier
                changed = True

            # 학과명 재정규화
            old_dept = metadata.get("department", "")
            if old_dept:
                new_dept = normalize_department(old_dept)
                if new_dept != old_dept:
                    metadata["department"] = new_dept
                    changed = True

            # 변경된 경우만 파일 저장 (불필요한 I/O 방지)
            if changed:
                try:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    changed_count += 1
                except OSError as e:
                    logger.warning(
                        "메타데이터 저장 실패",
                        path=str(json_path),
                        error=str(e),
                    )

        return changed_count

    def _deduplicate_images(self) -> DeduplicationResult:
        """
        perceptual hash로 중복 이미지를 탐지하고 제거합니다.

        pHash 해밍 거리 5 이하를 중복으로 판정하며,
        각 중복 그룹에서 첫 번째 파일만 보존합니다.

        Returns:
            DeduplicationResult: 중복 제거 결과
        """
        if not self.images_dir.exists():
            logger.warning(
                "이미지 디렉토리가 존재하지 않음",
                dir=str(self.images_dir),
            )
            return DeduplicationResult()

        deduplicator = ImageDeduplicator(hash_size=8, threshold=5)
        return deduplicator.remove_duplicates(self.images_dir)

    def _clean_metadata_image_refs(self) -> None:
        """
        메타데이터에서 존재하지 않는 이미지 파일 참조를 제거합니다.

        손상 이미지 삭제 또는 중복 제거 후, 메타데이터의 images 목록에서
        더 이상 존재하지 않는 파일 경로를 정리합니다.

        이미지 파일 이름으로 존재 여부를 확인하여 경로 형식 차이에
        강건하게 동작합니다.
        """
        # 현재 존재하는 이미지 파일 이름 세트 구성
        existing_filenames: Set[str] = set()
        if self.images_dir.exists():
            existing_filenames = {
                f.name
                for f in self.images_dir.iterdir()
                if f.is_file()
                and f.suffix.lower() in self.IMAGE_EXTENSIONS
            }

        metadata_files = sorted(self.metadata_dir.glob("*.json"))
        cleaned_count = 0

        for json_path in metadata_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            images = metadata.get("images", [])
            if not images:
                continue

            # 존재하는 이미지 파일만 유지 (파일 이름으로 확인)
            valid_images: List[str] = []
            for img_ref in images:
                img_filename = Path(img_ref).name
                if img_filename in existing_filenames:
                    valid_images.append(img_ref)

            if len(valid_images) != len(images):
                metadata["images"] = valid_images
                try:
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(
                            metadata, f, ensure_ascii=False, indent=2
                        )
                    cleaned_count += 1
                    logger.debug(
                        "이미지 참조 정리",
                        post_no=metadata.get("post_no"),
                        before=len(images),
                        after=len(valid_images),
                    )
                except OSError:
                    pass

        if cleaned_count > 0:
            logger.info(
                "메타데이터 이미지 참조 정리 완료",
                cleaned_files=cleaned_count,
            )

    def _count_valid_posts(self) -> int:
        """
        이미지가 1개 이상 있는 유효한 게시글 수를 계산합니다.

        Returns:
            int: 유효한 게시글 수
        """
        count = 0
        for json_path in self.metadata_dir.glob("*.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if metadata.get("images"):
                    count += 1
            except (json.JSONDecodeError, OSError):
                pass
        return count

    def _generate_report(self) -> str:
        """
        통계 리포트를 생성합니다.

        Returns:
            str: 포맷된 통계 리포트 문자열
        """
        reporter = StatsReporter(self.metadata_dir, self.images_dir)
        stats = reporter.generate()
        return reporter.to_console(stats)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    CLI 인자를 파싱합니다.

    Args:
        args: 커맨드라인 인자 목록 (테스트 시 주입용)

    Returns:
        argparse.Namespace: 파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description="크롤링 데이터 정제 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 데이터 디렉토리에서 정제
  python -m data_pipeline.crawlers.cleaner

  # 특정 디렉토리에서 정제
  python -m data_pipeline.crawlers.cleaner --data-dir data/crawled

  # 출력 디렉토리 지정 (원본 보존)
  python -m data_pipeline.crawlers.cleaner --data-dir data/crawled --output-dir data/cleaned
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="크롤링 데이터 디렉토리 경로 (기본값: data/crawled)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="정제 결과 출력 디렉토리 (기본값: data-dir과 동일 - 인플레이스)",
    )

    return parser.parse_args(args)


def main() -> None:
    """정제기 메인 진입점"""
    args = parse_args()

    cleaner = DataCleaner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    try:
        result = cleaner.clean()
        print(f"\n정제 완료: {result.summary()}")
    except KeyboardInterrupt:
        print("\n사용자에 의해 정제가 중단되었습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
