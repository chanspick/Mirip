# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: LocalStorage 구현

로컬 파일 시스템 기반 스토리지 구현입니다.
"""

from pathlib import Path
from typing import List, Union

from data_pipeline.storage.base_storage import BaseStorage


class LocalStorage(BaseStorage):
    """
    로컬 파일 시스템 스토리지

    로컬 디렉토리에 파일을 저장하고 관리합니다.
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        LocalStorage 초기화

        Args:
            base_path: 기본 저장 경로
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_full_path(self, key: str) -> Path:
        """
        전체 파일 경로를 반환합니다.

        Args:
            key: 파일 키

        Returns:
            전체 경로
        """
        return self.base_path / key

    def save(self, key: str, data: Union[bytes, str]) -> None:
        """
        데이터를 파일로 저장합니다.

        Args:
            key: 파일 키 (상대 경로)
            data: 저장할 데이터
        """
        file_path = self._get_full_path(key)

        # 상위 디렉토리 생성
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 데이터 저장
        if isinstance(data, str):
            file_path.write_text(data, encoding="utf-8")
        else:
            file_path.write_bytes(data)

    def load(self, key: str, as_text: bool = False) -> Union[bytes, str]:
        """
        파일에서 데이터를 로드합니다.

        Args:
            key: 파일 키
            as_text: 텍스트로 로드할지 여부

        Returns:
            로드된 데이터

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
        """
        file_path = self._get_full_path(key)

        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {key}")

        if as_text:
            return file_path.read_text(encoding="utf-8")
        else:
            return file_path.read_bytes()

    def exists(self, key: str) -> bool:
        """
        파일 존재 여부를 확인합니다.

        Args:
            key: 파일 키

        Returns:
            존재 여부
        """
        return self._get_full_path(key).exists()

    def delete(self, key: str) -> None:
        """
        파일을 삭제합니다.

        Args:
            key: 삭제할 파일 키
        """
        file_path = self._get_full_path(key)
        if file_path.exists():
            file_path.unlink()

    def list_files(self, prefix: str = "") -> List[str]:
        """
        저장된 파일 목록을 조회합니다.

        Args:
            prefix: 필터링할 접두사

        Returns:
            파일 키 목록 (상대 경로)
        """
        files = []
        search_path = self.base_path / prefix if prefix else self.base_path

        if search_path.exists():
            for file_path in search_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.base_path)
                    files.append(str(relative_path))

        return files

    def get_size(self, key: str) -> int:
        """
        파일 크기를 반환합니다.

        Args:
            key: 파일 키

        Returns:
            파일 크기 (바이트)
        """
        file_path = self._get_full_path(key)
        if file_path.exists():
            return file_path.stat().st_size
        return 0
