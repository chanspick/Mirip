# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: MetadataStorage 구현

이미지 메타데이터 전용 스토리지입니다.
JSON 형식으로 메타데이터를 저장하고 관리합니다.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from data_pipeline.models.metadata import Department, ImageMetadata, Tier
from data_pipeline.storage.local_storage import LocalStorage


class MetadataStorage:
    """
    메타데이터 전용 스토리지

    ImageMetadata를 JSON 형식으로 저장하고 관리합니다.
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        MetadataStorage 초기화

        Args:
            base_path: 기본 저장 경로
        """
        self.base_path = Path(base_path)
        self.metadata_dir = self.base_path / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self._storage = LocalStorage(self.metadata_dir)

    def _get_filename(self, image_id: str) -> str:
        """
        이미지 ID에 대한 파일명을 반환합니다.

        Args:
            image_id: 이미지 ID

        Returns:
            파일명
        """
        return f"{image_id}.json"

    def save_metadata(self, metadata: ImageMetadata) -> None:
        """
        메타데이터를 저장합니다.

        Args:
            metadata: 저장할 ImageMetadata
        """
        filename = self._get_filename(metadata.image_id)
        json_data = metadata.model_dump_json(indent=2)
        self._storage.save(filename, json_data)

    def load_metadata(self, image_id: str) -> ImageMetadata:
        """
        메타데이터를 로드합니다.

        Args:
            image_id: 이미지 ID

        Returns:
            ImageMetadata 객체

        Raises:
            FileNotFoundError: 메타데이터가 존재하지 않을 때
        """
        filename = self._get_filename(image_id)
        json_data = self._storage.load(filename, as_text=True)
        return ImageMetadata.model_validate_json(json_data)

    def exists_metadata(self, image_id: str) -> bool:
        """
        메타데이터 존재 여부를 확인합니다.

        Args:
            image_id: 이미지 ID

        Returns:
            존재 여부
        """
        filename = self._get_filename(image_id)
        return self._storage.exists(filename)

    def delete_metadata(self, image_id: str) -> None:
        """
        메타데이터를 삭제합니다.

        Args:
            image_id: 이미지 ID
        """
        filename = self._get_filename(image_id)
        self._storage.delete(filename)

    def save_batch(self, metadatas: List[ImageMetadata]) -> None:
        """
        여러 메타데이터를 배치로 저장합니다.

        Args:
            metadatas: ImageMetadata 리스트
        """
        for metadata in metadatas:
            self.save_metadata(metadata)

    def load_all(self) -> List[ImageMetadata]:
        """
        모든 메타데이터를 로드합니다.

        Returns:
            ImageMetadata 리스트
        """
        metadatas = []
        files = self._storage.list_files()

        for filename in files:
            if filename.endswith(".json"):
                try:
                    json_data = self._storage.load(filename, as_text=True)
                    metadata = ImageMetadata.model_validate_json(json_data)
                    metadatas.append(metadata)
                except Exception:
                    # 손상된 파일은 건너뜀
                    continue

        return metadatas

    def update_metadata(
        self, image_id: str, updates: Dict[str, Any]
    ) -> ImageMetadata:
        """
        메타데이터를 업데이트합니다.

        Args:
            image_id: 이미지 ID
            updates: 업데이트할 필드들

        Returns:
            업데이트된 ImageMetadata
        """
        metadata = self.load_metadata(image_id)
        metadata_dict = metadata.model_dump()

        # 업데이트 적용
        for key, value in updates.items():
            if key in metadata_dict:
                metadata_dict[key] = value

        # 새 메타데이터 생성 및 저장
        updated_metadata = ImageMetadata(**metadata_dict)
        self.save_metadata(updated_metadata)

        return updated_metadata

    def query_by_department(
        self, department: Department
    ) -> List[ImageMetadata]:
        """
        과별로 메타데이터를 조회합니다.

        Args:
            department: 과별

        Returns:
            해당 과별의 ImageMetadata 리스트
        """
        all_metadata = self.load_all()
        return [m for m in all_metadata if m.department == department]

    def query_by_tier(self, tier: Tier) -> List[ImageMetadata]:
        """
        티어로 메타데이터를 조회합니다.

        Args:
            tier: 티어

        Returns:
            해당 티어의 ImageMetadata 리스트
        """
        all_metadata = self.load_all()
        return [m for m in all_metadata if m.tier == tier]

    def get_statistics(self) -> Dict[str, Any]:
        """
        저장된 메타데이터의 통계를 반환합니다.

        Returns:
            통계 딕셔너리
        """
        all_metadata = self.load_all()

        if not all_metadata:
            return {
                "total_count": 0,
                "department_distribution": {},
                "tier_distribution": {},
            }

        # 과별 분포
        dept_counts = Counter(m.department.value for m in all_metadata)
        dept_distribution = {
            dept: count / len(all_metadata)
            for dept, count in dept_counts.items()
        }

        # 티어 분포
        tier_counts = Counter(m.tier.value for m in all_metadata)
        tier_distribution = {
            tier: count / len(all_metadata)
            for tier, count in tier_counts.items()
        }

        return {
            "total_count": len(all_metadata),
            "department_distribution": dept_distribution,
            "tier_distribution": tier_distribution,
        }
