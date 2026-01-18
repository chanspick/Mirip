# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: ImageCollector 구현

로컬 파일 시스템에서 이미지를 수집하는 기능을 제공합니다.
지원되는 이미지 형식을 필터링하고 출력 디렉토리로 복사합니다.
"""

import shutil
import uuid
from pathlib import Path
from typing import Iterator, List

from PIL import Image

from data_pipeline.collectors.base_collector import BaseCollector, CollectionResult


class ImageCollector(BaseCollector):
    """
    로컬 파일 시스템 이미지 수집기

    지정된 디렉토리에서 이미지 파일을 수집하여
    출력 디렉토리로 복사합니다.
    """

    # 지원되는 이미지 형식
    supported_formats: List[str] = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]

    def __init__(self, output_dir: Path, recursive: bool = True):
        """
        ImageCollector 초기화

        Args:
            output_dir: 수집된 이미지를 저장할 출력 디렉토리
            recursive: 하위 디렉토리 재귀 탐색 여부 (기본값: True)
        """
        self.output_dir = output_dir
        self.recursive = recursive

        # 출력 디렉토리가 없으면 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, source: str) -> Iterator[CollectionResult]:
        """
        소스 디렉토리에서 이미지를 수집합니다.

        Args:
            source: 소스 디렉토리 경로

        Yields:
            CollectionResult: 각 이미지의 수집 결과

        Raises:
            FileNotFoundError: 소스 디렉토리가 존재하지 않을 때
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"소스 경로가 존재하지 않습니다: {source}")

        if not source_path.is_dir():
            raise FileNotFoundError(f"소스 경로가 디렉토리가 아닙니다: {source}")

        # 이미지 파일 탐색
        if self.recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in source_path.glob(pattern):
            if not file_path.is_file():
                continue

            # 지원되는 형식 확인
            if file_path.suffix.lower() not in self.supported_formats:
                continue

            # 이미지 수집
            yield self._collect_single_image(file_path)

    def _collect_single_image(self, source_file: Path) -> CollectionResult:
        """
        단일 이미지를 수집합니다.

        Args:
            source_file: 소스 이미지 파일 경로

        Returns:
            CollectionResult: 수집 결과
        """
        # 고유 ID 생성
        image_id = f"img_{uuid.uuid4().hex[:12]}"

        # 출력 파일 경로
        output_file = self.output_dir / f"{image_id}{source_file.suffix.lower()}"

        try:
            # 이미지 유효성 검증
            if not self.validate(str(source_file)):
                return CollectionResult(
                    image_id=image_id,
                    source_path=str(source_file),
                    local_path="",
                    success=False,
                    error_message="유효하지 않은 이미지 파일입니다",
                )

            # 파일 복사
            shutil.copy2(source_file, output_file)

            return CollectionResult(
                image_id=image_id,
                source_path=str(source_file),
                local_path=str(output_file),
                success=True,
            )

        except Exception as e:
            return CollectionResult(
                image_id=image_id,
                source_path=str(source_file),
                local_path="",
                success=False,
                error_message=str(e),
            )

    def validate(self, path: str) -> bool:
        """
        이미지 파일의 유효성을 검증합니다.

        Args:
            path: 검증할 파일 경로

        Returns:
            bool: 유효한 이미지 파일이면 True
        """
        file_path = Path(path)

        # 파일 존재 확인
        if not file_path.exists():
            return False

        # 파일인지 확인
        if not file_path.is_file():
            return False

        # 지원 형식 확인
        if file_path.suffix.lower() not in self.supported_formats:
            return False

        # 이미지 로드 시도
        try:
            with Image.open(file_path) as img:
                # 이미지 데이터 검증 (실제 로드)
                img.verify()

            # verify 후 다시 열어서 실제 로드 가능한지 확인
            with Image.open(file_path) as img:
                img.load()

            return True

        except Exception:
            return False
