# -*- coding: utf-8 -*-
"""
크롤링 데이터 → 학습 파이프라인 변환 모듈 (Phase 1.3 Pipeline Integration)

크롤링된 메타데이터 JSON 파일들을 학습 파이프라인(PairwiseDataset, generate_pairs)이
기대하는 CSV 포맷으로 변환합니다.

핵심 변환 로직:
    - 1 포스트 → N 이미지 행 (one-to-many 관계)
    - 크롤러 학과 코드(visual_design 등) → 한글 표시값(시디 등)으로 매핑
    - 유효하지 않은 티어/이미지 필터링
    - 학습/검증/테스트 분할 (tiered stratified split)

사용법:
    python -m data_pipeline.crawlers.converter
    python -m data_pipeline.crawlers.converter --crawled-dir data/crawled --output-dir data/processed
    python -m data_pipeline.crawlers.converter --split --train-ratio 0.8 --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger("crawler.converter")


# ============================================================
# 학과 코드 매핑: 크롤러 코드 → 학습 파이프라인 표시값
# ============================================================
DEPARTMENT_CODE_TO_DISPLAY: Dict[str, str] = {
    "visual_design": "시디",
    "industrial_design": "산디",
    "craft": "공예",
    "fine_art": "회화",
}

# 유효한 티어 값 (학습 파이프라인에서 허용하는 값)
VALID_TIERS = {"S", "A", "B", "C"}

# CSV 출력 컬럼 순서
CSV_COLUMNS = [
    "image_path",
    "tier",
    "department",
    "university",
    "year",
    "admission_type",
    "work_type",
    "post_no",
]


# ============================================================
# 결과 데이터클래스
# ============================================================


@dataclass
class ValidationReport:
    """
    변환 데이터 유효성 검증 결과

    Attributes:
        total_rows: 전체 행 수
        valid_rows: 유효한 행 수
        missing_images: 디스크에 없는 이미지 경로 목록
        invalid_tiers: 유효하지 않은 티어 값 목록
        invalid_departments: 유효하지 않은 학과 값 목록
        unique_tiers: 고유 티어 수
        tier_distribution: 티어별 이미지 분포
        department_distribution: 학과별 이미지 분포
        is_valid: 전체 유효성 여부
        errors: 오류 메시지 목록
        warnings: 경고 메시지 목록
    """

    total_rows: int = 0
    valid_rows: int = 0
    missing_images: List[str] = field(default_factory=list)
    invalid_tiers: List[str] = field(default_factory=list)
    invalid_departments: List[str] = field(default_factory=list)
    unique_tiers: int = 0
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    department_distribution: Dict[str, int] = field(default_factory=dict)
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ConversionResult:
    """
    변환 실행 결과

    Attributes:
        total_posts: 처리된 총 포스트 수
        total_images: 변환된 총 이미지 행 수
        skipped_posts: 건너뛴 포스트 수 (이미지 없음 등)
        skipped_images: 건너뛴 이미지 수 (파일 없음 등)
        filtered_invalid_tier: 유효하지 않은 티어로 필터링된 수
        output_csv_path: 출력 CSV 파일 경로
        validation: 유효성 검증 결과
        dataframe: 변환된 DataFrame
    """

    total_posts: int = 0
    total_images: int = 0
    skipped_posts: int = 0
    skipped_images: int = 0
    filtered_invalid_tier: int = 0
    output_csv_path: Optional[Path] = None
    validation: Optional[ValidationReport] = None
    dataframe: Optional[pd.DataFrame] = None


# ============================================================
# 메인 변환기
# ============================================================


class CrawledDataConverter:
    """
    크롤링 데이터를 학습 파이프라인 포맷으로 변환합니다.

    크롤러가 수집한 메타데이터 JSON 파일들을 읽어
    PairwiseDataset/generate_pairs가 기대하는 CSV 포맷으로 변환합니다.

    변환 규칙:
        - 1 포스트 → N 이미지 행 (one-to-many)
        - 이미지 파일이 실제로 존재하는 행만 포함
        - 유효한 티어(S/A/B/C)만 포함
        - 학과 코드를 한글 표시값으로 변환

    Args:
        crawled_dir: 크롤링 데이터 루트 디렉토리
                     하위에 metadata/, raw_images/ 포함
        output_dir: 변환 결과 출력 디렉토리 (기본: data/processed)
        project_root: 프로젝트 루트 디렉토리 (이미지 경로 해석용)

    Examples:
        >>> converter = CrawledDataConverter("data/crawled")
        >>> result = converter.convert()
        >>> print(result.total_images)
        500
    """

    def __init__(
        self,
        crawled_dir: str | Path,
        output_dir: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self.crawled_dir = Path(crawled_dir)
        self.metadata_dir = self.crawled_dir / "metadata"
        self.images_dir = self.crawled_dir / "raw_images"
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed")

        # 프로젝트 루트: 이미지 상대 경로 해석에 사용
        self.project_root = Path(project_root) if project_root else Path.cwd()

    def convert(self) -> ConversionResult:
        """
        전체 변환을 실행합니다.

        실행 단계:
            1. 크롤링 메타데이터 JSON 읽기
            2. 이미지 존재 여부 검증
            3. DataFrame 생성 (한 이미지 = 한 행)
            4. CSV 내보내기 (metadata.csv)
            5. 티어별/학과별 분포 검증
            6. ConversionResult 반환

        Returns:
            ConversionResult: 변환 결과 (DataFrame, 통계, 검증 결과 포함)

        Raises:
            FileNotFoundError: 메타데이터 디렉토리가 존재하지 않을 때
        """
        # 1. 메타데이터 디렉토리 존재 확인
        if not self.metadata_dir.exists():
            raise FileNotFoundError(
                f"메타데이터 디렉토리를 찾을 수 없습니다: {self.metadata_dir}"
            )

        # 2. DataFrame 및 통계 생성
        df, stats = self._build_dataframe()

        result = ConversionResult(
            total_posts=stats["total_posts"],
            total_images=len(df),
            skipped_posts=stats["skipped_posts"],
            skipped_images=stats["skipped_images"],
            filtered_invalid_tier=stats["filtered_invalid_tier"],
            dataframe=df,
        )

        logger.info(
            "변환 완료",
            total_images=result.total_images,
            total_posts=result.total_posts,
            skipped_posts=result.skipped_posts,
            skipped_images=result.skipped_images,
        )

        # 3. 빈 결과 처리
        if len(df) == 0:
            logger.warning("변환 결과가 비어 있습니다")
            result.validation = ValidationReport(
                is_valid=False,
                errors=["변환된 이미지가 없습니다"],
            )
            return result

        # 4. CSV 내보내기
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "metadata.csv"
        result.output_csv_path = self.export_csv(df, csv_path)

        # 5. 유효성 검증
        result.validation = self.validate(df)

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        크롤링 메타데이터를 pandas DataFrame으로 변환합니다.

        각 포스트의 각 이미지가 별도의 행이 됩니다(one-to-many).
        이미지가 실제 디스크에 존재하는 경우만 포함합니다.

        Returns:
            pd.DataFrame: 변환된 DataFrame
                columns: image_path, tier, department, university,
                         year, admission_type, work_type, post_no
        """
        df, _ = self._build_dataframe()
        return df

    def _build_dataframe(self) -> tuple[pd.DataFrame, Dict[str, int]]:
        """
        내부 변환 로직: 메타데이터 JSON을 읽어 DataFrame과 통계를 생성합니다.

        Returns:
            tuple: (변환된 DataFrame, 통계 딕셔너리)
                통계 키: total_posts, skipped_posts, skipped_images,
                         filtered_invalid_tier
        """
        rows: List[Dict] = []

        # 메타데이터 JSON 파일 목록
        metadata_files = sorted(self.metadata_dir.glob("*.json"))

        if not metadata_files:
            logger.warning(
                "메타데이터 파일이 없습니다",
                dir=str(self.metadata_dir),
            )
            return pd.DataFrame(columns=CSV_COLUMNS), {
                "total_posts": 0,
                "skipped_posts": 0,
                "skipped_images": 0,
                "filtered_invalid_tier": 0,
            }

        total_posts = 0
        skipped_posts = 0
        skipped_images = 0
        filtered_invalid_tier = 0

        for json_path in metadata_files:
            total_posts += 1

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "메타데이터 읽기 실패",
                    path=str(json_path),
                    error=str(e),
                )
                skipped_posts += 1
                continue

            # 이미지 목록 확인
            images = meta.get("images", [])
            if not images:
                skipped_posts += 1
                continue

            # 티어 유효성 확인 (S/A/B/C만 허용)
            tier = meta.get("tier", "")
            if tier not in VALID_TIERS:
                filtered_invalid_tier += 1
                logger.debug(
                    "유효하지 않은 티어로 건너뜀",
                    post_no=meta.get("post_no"),
                    tier=tier,
                )
                continue

            # 학과 코드 → 한글 표시값 변환
            raw_department = meta.get("department", "")
            department = DEPARTMENT_CODE_TO_DISPLAY.get(
                raw_department, "미분류"
            )

            # 공통 메타데이터 추출
            university = meta.get("university", "")
            year = meta.get("year", "")
            admission_type = meta.get("admission_type", "")
            work_type = meta.get("work_type", "")
            post_no = meta.get("post_no", "")

            # 각 이미지에 대해 별도의 행 생성
            for image_path in images:
                # 이미지 파일 존재 확인
                resolved = self._resolve_image_path(image_path)
                if not resolved.exists():
                    logger.debug(
                        "이미지 파일 없음",
                        path=str(image_path),
                        resolved=str(resolved),
                    )
                    skipped_images += 1
                    continue

                rows.append(
                    {
                        "image_path": image_path,
                        "tier": tier,
                        "department": department,
                        "university": university,
                        "year": year,
                        "admission_type": admission_type,
                        "work_type": work_type,
                        "post_no": post_no,
                    }
                )

        stats = {
            "total_posts": total_posts,
            "skipped_posts": skipped_posts,
            "skipped_images": skipped_images,
            "filtered_invalid_tier": filtered_invalid_tier,
        }

        # DataFrame 생성 로그
        logger.info(
            "메타데이터 처리 완료",
            total_posts=total_posts,
            skipped_posts=skipped_posts,
            skipped_images=skipped_images,
            filtered_invalid_tier=filtered_invalid_tier,
            result_rows=len(rows),
        )

        if not rows:
            return pd.DataFrame(columns=CSV_COLUMNS), stats

        df = pd.DataFrame(rows, columns=CSV_COLUMNS)
        return df, stats

    def export_csv(self, df: pd.DataFrame, output_path: str | Path) -> Path:
        """
        DataFrame을 CSV 파일로 내보냅니다.

        Args:
            df: 내보낼 DataFrame
            output_path: 출력 CSV 파일 경로

        Returns:
            Path: 저장된 CSV 파일의 절대 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")

        logger.info(
            "CSV 내보내기 완료",
            path=str(output_path),
            rows=len(df),
        )

        return output_path

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        변환된 데이터의 유효성을 검증합니다.

        검증 항목:
            - 모든 이미지 파일이 디스크에 존재하는지
            - 티어 값이 유효한지 (S/A/B/C)
            - 학과 값이 유효한지
            - 최소 이미지 수 확인
            - 최소 2개 이상의 다른 티어 확인 (pairwise 학습에 필요)

        Args:
            df: 검증할 DataFrame

        Returns:
            ValidationReport: 검증 결과
        """
        report = ValidationReport()
        report.total_rows = len(df)

        if len(df) == 0:
            report.is_valid = False
            report.errors.append("데이터가 비어 있습니다")
            return report

        # 1. 이미지 파일 존재 확인
        for _, row in df.iterrows():
            resolved = self._resolve_image_path(row["image_path"])
            if not resolved.exists():
                report.missing_images.append(row["image_path"])

        if report.missing_images:
            report.warnings.append(
                f"존재하지 않는 이미지 {len(report.missing_images)}개 발견"
            )

        # 2. 티어 유효성 확인
        invalid_tier_mask = ~df["tier"].isin(VALID_TIERS)
        if invalid_tier_mask.any():
            invalid_values = df.loc[invalid_tier_mask, "tier"].unique().tolist()
            report.invalid_tiers = invalid_values
            report.errors.append(
                f"유효하지 않은 티어 값 발견: {invalid_values}"
            )
            report.is_valid = False

        # 3. 학과 유효성 확인
        valid_departments = set(DEPARTMENT_CODE_TO_DISPLAY.values()) | {"미분류"}
        invalid_dept_mask = ~df["department"].isin(valid_departments)
        if invalid_dept_mask.any():
            invalid_values = (
                df.loc[invalid_dept_mask, "department"].unique().tolist()
            )
            report.invalid_departments = invalid_values
            report.warnings.append(
                f"알 수 없는 학과 값 발견: {invalid_values}"
            )

        # 4. 티어 분포 계산
        tier_counts = df["tier"].value_counts().to_dict()
        report.tier_distribution = tier_counts
        report.unique_tiers = len(tier_counts)

        # 5. 학과 분포 계산
        dept_counts = df["department"].value_counts().to_dict()
        report.department_distribution = dept_counts

        # 6. 최소 2개 이상의 다른 티어 필요 (pairwise 학습 조건)
        if report.unique_tiers < 2:
            report.errors.append(
                f"Pairwise 학습에는 최소 2개의 다른 티어가 필요합니다. "
                f"현재: {report.unique_tiers}개 ({list(tier_counts.keys())})"
            )
            report.is_valid = False

        # 7. 최소 이미지 수 확인 (실용적 최소 기준: 10장)
        if len(df) < 10:
            report.warnings.append(
                f"이미지 수가 매우 적습니다: {len(df)}장 (최소 권장: 10장)"
            )

        # 유효 행 수 계산
        report.valid_rows = len(df) - len(report.missing_images)

        logger.info(
            "유효성 검증 완료",
            total_rows=report.total_rows,
            valid_rows=report.valid_rows,
            missing_images=len(report.missing_images),
            unique_tiers=report.unique_tiers,
            is_valid=report.is_valid,
        )

        return report

    def generate_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """
        학습/검증/테스트 분할을 생성합니다.

        티어별 층화 추출(stratified split)을 사용하여
        각 분할에서 티어 분포가 유사하도록 합니다.

        Args:
            df: 분할할 DataFrame (image_path, tier 컬럼 필수)
            train_ratio: 학습 세트 비율 (기본: 0.8)
            val_ratio: 검증 세트 비율 (기본: 0.1)
            seed: 재현성을 위한 랜덤 시드 (기본: 42)

        Returns:
            Dict[str, pd.DataFrame]: {"train": ..., "val": ..., "test": ...}

        Raises:
            ValueError: 비율이 유효하지 않을 때
        """
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError(
                f"train_ratio({train_ratio}) + val_ratio({val_ratio}) = "
                f"{train_ratio + val_ratio} > 1.0"
            )

        if len(df) == 0:
            return {
                "train": pd.DataFrame(columns=CSV_COLUMNS),
                "val": pd.DataFrame(columns=CSV_COLUMNS),
                "test": pd.DataFrame(columns=CSV_COLUMNS),
            }

        # post_no 기반 분할: 같은 포스트의 이미지는 같은 분할에 배치
        # (데이터 누출 방지)
        if "post_no" in df.columns and df["post_no"].notna().all():
            return self._split_by_post(df, train_ratio, val_ratio, seed)

        # post_no 없으면 이미지 단위 분할 (기존 DataSplitter와 동일)
        return self._split_by_image(df, train_ratio, val_ratio, seed)

    def _split_by_post(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> Dict[str, pd.DataFrame]:
        """
        포스트 단위 층화 분할 (같은 포스트의 이미지는 같은 분할에 배치)

        Args:
            df: 분할할 DataFrame
            train_ratio: 학습 세트 비율
            val_ratio: 검증 세트 비율
            seed: 랜덤 시드

        Returns:
            Dict[str, pd.DataFrame]: 분할 결과
        """
        # 포스트별 대표 티어 추출 (첫 번째 이미지의 티어 사용)
        post_tiers = (
            df.groupby("post_no")["tier"]
            .first()
            .reset_index()
        )

        rng = np.random.RandomState(seed)
        post_indices = np.arange(len(post_tiers))
        rng.shuffle(post_indices)

        # 티어별 포스트 분리 후 비율에 따라 분할
        train_posts = []
        val_posts = []
        test_posts = []

        for tier in post_tiers["tier"].unique():
            tier_mask = post_tiers["tier"] == tier
            tier_post_nos = post_tiers.loc[tier_mask, "post_no"].values.tolist()
            rng.shuffle(tier_post_nos)

            n = len(tier_post_nos)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            # 최소 1개씩 보장 (가능한 경우)
            if n >= 3:
                train_end = max(train_end, 1)
                val_end = max(val_end, train_end + 1)

            train_posts.extend(tier_post_nos[:train_end])
            val_posts.extend(tier_post_nos[train_end:val_end])
            test_posts.extend(tier_post_nos[val_end:])

        # 포스트 번호로 DataFrame 필터링
        train_df = df[df["post_no"].isin(train_posts)].reset_index(drop=True)
        val_df = df[df["post_no"].isin(val_posts)].reset_index(drop=True)
        test_df = df[df["post_no"].isin(test_posts)].reset_index(drop=True)

        logger.info(
            "포스트 단위 분할 완료",
            train=len(train_df),
            val=len(val_df),
            test=len(test_df),
        )

        return {"train": train_df, "val": val_df, "test": test_df}

    def _split_by_image(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> Dict[str, pd.DataFrame]:
        """
        이미지 단위 층화 분할 (post_no 없는 경우 대체)

        Args:
            df: 분할할 DataFrame
            train_ratio: 학습 세트 비율
            val_ratio: 검증 세트 비율
            seed: 랜덤 시드

        Returns:
            Dict[str, pd.DataFrame]: 분할 결과
        """
        rng = np.random.RandomState(seed)

        train_dfs = []
        val_dfs = []
        test_dfs = []

        for tier in df["tier"].unique():
            tier_df = df[df["tier"] == tier].copy()
            indices = np.arange(len(tier_df))
            rng.shuffle(indices)

            n = len(indices)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_dfs.append(tier_df.iloc[indices[:train_end]])
            val_dfs.append(tier_df.iloc[indices[train_end:val_end]])
            test_dfs.append(tier_df.iloc[indices[val_end:]])

        train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(columns=CSV_COLUMNS)
        val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame(columns=CSV_COLUMNS)
        test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame(columns=CSV_COLUMNS)

        logger.info(
            "이미지 단위 분할 완료",
            train=len(train_df),
            val=len(val_df),
            test=len(test_df),
        )

        return {"train": train_df, "val": val_df, "test": test_df}

    def _resolve_image_path(self, image_path: str) -> Path:
        """
        이미지 경로를 절대 경로로 해석합니다.

        메타데이터 JSON에 저장된 상대 경로(data/crawled/raw_images/...)를
        디스크 상의 절대 경로로 변환합니다.

        Args:
            image_path: 메타데이터에 저장된 이미지 경로

        Returns:
            Path: 해석된 절대 경로
        """
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.project_root / path


# ============================================================
# CLI 인터페이스
# ============================================================


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    CLI 인자를 파싱합니다.

    Args:
        args: 커맨드라인 인자 목록 (테스트 시 주입용)

    Returns:
        argparse.Namespace: 파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description="크롤링 데이터를 학습 파이프라인 포맷으로 변환합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 디렉토리에서 변환
  python -m data_pipeline.crawlers.converter

  # 특정 디렉토리에서 변환
  python -m data_pipeline.crawlers.converter --crawled-dir data/crawled --output-dir data/processed

  # 분할 포함 변환
  python -m data_pipeline.crawlers.converter --split --train-ratio 0.8 --val-ratio 0.1

  # 프로젝트 루트 지정 (이미지 상대 경로 해석용)
  python -m data_pipeline.crawlers.converter --project-root /path/to/project
        """,
    )

    parser.add_argument(
        "--crawled-dir",
        type=str,
        default="data/crawled",
        help="크롤링 데이터 디렉토리 (기본값: data/crawled)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="변환 결과 출력 디렉토리 (기본값: data/processed)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="프로젝트 루트 디렉토리 (이미지 상대 경로 해석용, 기본: 현재 디렉토리)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="학습/검증/테스트 분할 생성",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="학습 세트 비율 (기본값: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="검증 세트 비율 (기본값: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성을 위한 랜덤 시드 (기본값: 42)",
    )

    return parser.parse_args(args)


def _format_report(result: ConversionResult) -> str:
    """
    변환 결과를 터미널 출력용 문자열로 포맷합니다.

    Args:
        result: 변환 결과

    Returns:
        str: 포맷된 리포트 문자열
    """
    lines: List[str] = []
    sep = "=" * 50

    lines.append("")
    lines.append(sep)
    lines.append("  크롤링 데이터 변환 결과")
    lines.append(sep)
    lines.append("")
    lines.append(f"총 포스트: {result.total_posts:,}개")
    lines.append(f"변환된 이미지: {result.total_images:,}장")
    lines.append(f"건너뛴 포스트: {result.skipped_posts:,}개")
    lines.append(f"건너뛴 이미지: {result.skipped_images:,}장")
    lines.append(f"필터링된 (무효 티어): {result.filtered_invalid_tier:,}개")

    if result.output_csv_path:
        lines.append(f"출력 CSV: {result.output_csv_path}")

    if result.validation:
        v = result.validation
        lines.append("")
        lines.append(sep)
        lines.append("  유효성 검증")
        lines.append(sep)
        lines.append(f"유효: {'예' if v.is_valid else '아니오'}")
        lines.append(f"유효 행: {v.valid_rows:,}/{v.total_rows:,}")
        lines.append(f"누락 이미지: {len(v.missing_images):,}개")
        lines.append(f"고유 티어: {v.unique_tiers}개")

        if v.tier_distribution:
            lines.append("")
            lines.append("  티어 분포:")
            for tier in ["S", "A", "B", "C"]:
                count = v.tier_distribution.get(tier, 0)
                lines.append(f"    {tier}: {count:,}")

        if v.department_distribution:
            lines.append("")
            lines.append("  학과 분포:")
            for dept, count in sorted(
                v.department_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                lines.append(f"    {dept}: {count:,}")

        if v.errors:
            lines.append("")
            lines.append("  오류:")
            for error in v.errors:
                lines.append(f"    - {error}")

        if v.warnings:
            lines.append("")
            lines.append("  경고:")
            for warning in v.warnings:
                lines.append(f"    - {warning}")

    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


def main() -> int:
    """
    CLI 메인 진입점

    Returns:
        int: 종료 코드 (0: 성공, 1: 오류)
    """
    args = parse_args()

    try:
        converter = CrawledDataConverter(
            crawled_dir=args.crawled_dir,
            output_dir=args.output_dir,
            project_root=args.project_root,
        )

        result = converter.convert()

        # 분할 생성
        if args.split and result.dataframe is not None and len(result.dataframe) > 0:
            splits = converter.generate_split(
                df=result.dataframe,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )

            # 분할 CSV 저장
            output_dir = Path(args.output_dir)
            for split_name, split_df in splits.items():
                split_path = output_dir / f"{split_name}_metadata.csv"
                converter.export_csv(split_df, split_path)
                print(f"  {split_name}: {len(split_df):,}행 -> {split_path}")

        # 결과 출력
        print(_format_report(result))

        # 유효성 검증 실패 시 경고 코드 반환
        if result.validation and not result.validation.is_valid:
            return 2

        return 0

    except FileNotFoundError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("예상치 못한 오류 발생", error=str(e))
        print(f"예상치 못한 오류: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
