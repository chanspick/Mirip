# -*- coding: utf-8 -*-
"""
이미지 중복 제거 모듈

perceptual hash(pHash)를 사용하여 시각적으로 유사한 이미지를 감지하고 제거합니다.
해시 거리 5 이하를 중복으로 판정합니다.

pHash는 이미지를 주파수 영역으로 변환한 뒤 해시값을 생성하므로,
크기 변환, 약간의 색상 변화, 압축 아티팩트에도 강건합니다.

의존성:
    pip install imagehash Pillow
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger("crawler.dedup")


@dataclass
class DeduplicationResult:
    """
    중복 제거 결과 데이터

    Attributes:
        total_images: 전체 이미지 수
        unique_images: 중복 제거 후 고유 이미지 수
        duplicates_removed: 제거된 중복 이미지 수
        duplicate_groups: 중복 그룹 (해시값 -> 파일 경로 목록)
    """

    total_images: int = 0
    unique_images: int = 0
    duplicates_removed: int = 0
    duplicate_groups: Dict[str, List[str]] = field(default_factory=dict)


class ImageDeduplicator:
    """
    이미지 중복 제거기

    perceptual hash(pHash)를 사용하여 시각적으로 유사한 이미지를 감지합니다.
    해밍 거리가 threshold 이하인 이미지 쌍을 중복으로 판정합니다.

    Union-Find 알고리즘으로 전이적 중복 관계를 처리합니다:
    A와 B가 유사하고, B와 C가 유사하면 A, B, C 모두 같은 그룹으로 묶입니다.
    """

    # 지원하는 이미지 확장자
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    def __init__(self, hash_size: int = 8, threshold: int = 5) -> None:
        """
        중복 제거기를 초기화합니다.

        Args:
            hash_size: pHash 크기 (기본값: 8 -> 64비트 해시)
            threshold: 중복 판정 해밍 거리 임계값 (기본값: 5)
        """
        self.hash_size = hash_size
        self.threshold = threshold

    def compute_hash(self, image_path: Path) -> Optional[str]:
        """
        이미지의 perceptual hash를 계산합니다.

        손상된 이미지나 읽을 수 없는 파일은 None을 반환합니다.
        verify() 후 다시 열어서 해시를 계산하는 2단계 검증을 수행합니다.

        Args:
            image_path: 이미지 파일 경로

        Returns:
            Optional[str]: 16진수 해시 문자열 또는 None (실패 시)
        """
        try:
            import imagehash
            from PIL import Image
        except ImportError:
            logger.error(
                "imagehash 또는 Pillow 라이브러리가 설치되지 않음. "
                "'pip install imagehash Pillow' 명령으로 설치하세요."
            )
            return None

        try:
            # 1단계: 이미지 무결성 검증
            with Image.open(image_path) as img:
                img.verify()

            # 2단계: verify() 후 다시 열어서 해시 계산
            # (verify()는 파일 포인터를 소모하므로 재오픈 필요)
            with Image.open(image_path) as img:
                img_hash = imagehash.phash(img, hash_size=self.hash_size)
                return str(img_hash)

        except Exception as e:
            logger.warning(
                "이미지 해시 계산 실패",
                path=str(image_path),
                error=str(e),
            )
            return None

    def find_duplicates(self, image_dir: Path) -> Dict[str, List[Path]]:
        """
        디렉토리 내 이미지들의 중복 그룹을 찾습니다.

        pHash 해밍 거리가 threshold 이하인 이미지들을 같은 그룹으로 묶습니다.
        Union-Find 알고리즘을 사용하여 전이적 중복 관계도 처리합니다.

        Args:
            image_dir: 이미지가 저장된 디렉토리 경로

        Returns:
            Dict[str, List[Path]]: 중복 그룹 딕셔너리 (대표 해시 -> 파일 목록)
                                   2개 이상의 파일이 있는 그룹만 포함
        """
        try:
            import imagehash  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            logger.error(
                "imagehash 라이브러리가 필요합니다. "
                "'pip install imagehash Pillow' 명령으로 설치하세요."
            )
            return {}

        # 1단계: 모든 이미지의 해시 계산
        hash_objects: Dict[Path, object] = {}  # path -> imagehash 객체
        image_files = self._list_images(image_dir)
        total = len(image_files)

        if total == 0:
            logger.info("이미지 파일이 없습니다", dir=str(image_dir))
            return {}

        logger.info("이미지 해시 계산 시작", total=total)

        for idx, image_path in enumerate(image_files):
            if (idx + 1) % 500 == 0 or idx == 0:
                logger.info(
                    "해시 계산 진행 중",
                    progress=f"{idx + 1}/{total}",
                )

            try:
                with Image.open(image_path) as img:
                    img.verify()
                with Image.open(image_path) as img:
                    h = imagehash.phash(img, hash_size=self.hash_size)
                    hash_objects[image_path] = h
            except Exception as e:
                logger.warning(
                    "해시 계산 스킵 (손상 이미지)",
                    path=str(image_path),
                    error=str(e),
                )

        logger.info("해시 계산 완료", computed=len(hash_objects))

        if len(hash_objects) < 2:
            return {}

        # 2단계: Union-Find로 중복 그룹 구성
        paths = list(hash_objects.keys())
        parent: Dict[Path, Path] = {p: p for p in paths}

        def find(x: Path) -> Path:
            """루트 노드를 찾습니다 (경로 압축 적용)."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: Path, y: Path) -> None:
            """두 노드를 같은 그룹으로 합칩니다."""
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 모든 이미지 쌍 비교 (O(n^2))
        n = len(paths)
        comparisons = n * (n - 1) // 2
        logger.info(
            "중복 비교 시작",
            images=n,
            comparisons=comparisons,
        )

        compared = 0
        for i in range(n):
            for j in range(i + 1, n):
                distance = hash_objects[paths[i]] - hash_objects[paths[j]]
                if distance <= self.threshold:
                    union(paths[i], paths[j])

                compared += 1
                if compared % 5000000 == 0:
                    logger.info(
                        "비교 진행 중",
                        progress=f"{compared:,}/{comparisons:,}",
                    )

        # 3단계: 루트 기준으로 그룹 생성
        groups: Dict[Path, List[Path]] = defaultdict(list)
        for path in paths:
            root = find(path)
            groups[root].append(path)

        # 2개 이상인 그룹만 필터링 (중복이 있는 그룹)
        duplicate_groups: Dict[str, List[Path]] = {}
        for root, members in groups.items():
            if len(members) > 1:
                hash_key = str(hash_objects[root])
                # 파일 이름 기준 정렬 (일관된 순서 보장)
                duplicate_groups[hash_key] = sorted(
                    members, key=lambda p: p.name
                )

        total_duplicates = sum(
            len(members) - 1 for members in duplicate_groups.values()
        )
        logger.info(
            "중복 그룹 탐색 완료",
            duplicate_groups=len(duplicate_groups),
            duplicate_images=total_duplicates,
        )

        return duplicate_groups

    def remove_duplicates(
        self,
        image_dir: Path,
        keep: str = "first",
    ) -> DeduplicationResult:
        """
        중복 이미지를 탐지하고 제거합니다.

        각 중복 그룹에서 하나의 이미지만 보존하고 나머지를 삭제합니다.

        Args:
            image_dir: 이미지 디렉토리 경로
            keep: 보존 전략 ('first' = 정렬 순서 첫 번째 파일 보존)

        Returns:
            DeduplicationResult: 중복 제거 결과
        """
        result = DeduplicationResult()

        # 전체 이미지 수 확인
        all_images = self._list_images(image_dir)
        result.total_images = len(all_images)

        if result.total_images == 0:
            logger.info(
                "이미지가 없어 중복 제거를 건너뜁니다",
                dir=str(image_dir),
            )
            return result

        # 중복 그룹 탐색
        duplicate_groups = self.find_duplicates(image_dir)

        if not duplicate_groups:
            result.unique_images = result.total_images
            logger.info("중복 이미지가 없습니다")
            return result

        # 중복 제거 실행
        removed_count = 0
        for hash_key, members in duplicate_groups.items():
            # 보존할 이미지 결정 (첫 번째 파일 보존)
            keep_path = members[0]
            remove_paths = members[1:]

            for path in remove_paths:
                try:
                    path.unlink()
                    removed_count += 1
                    logger.debug(
                        "중복 이미지 삭제",
                        removed=str(path),
                        kept=str(keep_path),
                        hash=hash_key,
                    )
                except OSError as e:
                    logger.warning(
                        "이미지 삭제 실패",
                        path=str(path),
                        error=str(e),
                    )

        result.duplicates_removed = removed_count
        result.unique_images = result.total_images - removed_count
        result.duplicate_groups = {
            k: [str(p) for p in v] for k, v in duplicate_groups.items()
        }

        logger.info(
            "중복 제거 완료",
            total=result.total_images,
            unique=result.unique_images,
            removed=result.duplicates_removed,
            groups=len(duplicate_groups),
        )

        return result

    def _list_images(self, image_dir: Path) -> List[Path]:
        """
        디렉토리 내 지원되는 이미지 파일 목록을 반환합니다.

        Args:
            image_dir: 이미지 디렉토리 경로

        Returns:
            List[Path]: 이미지 파일 경로 목록 (이름순 정렬)
        """
        if not image_dir.exists():
            logger.warning(
                "이미지 디렉토리가 존재하지 않음",
                dir=str(image_dir),
            )
            return []

        images: List[Path] = []
        for f in sorted(image_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                images.append(f)

        return images
