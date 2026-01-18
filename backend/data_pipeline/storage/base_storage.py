# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: BaseStorage 추상 클래스

스토리지의 기본 인터페이스를 정의합니다.
모든 스토리지는 이 추상 클래스를 상속받아 구현합니다.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union


class BaseStorage(ABC):
    """
    스토리지 추상 베이스 클래스

    데이터를 저장하고 로드하는 기능의 인터페이스를 정의합니다.
    LocalStorage, CloudStorage 등이 이를 구현합니다.
    """

    @abstractmethod
    def save(self, key: str, data: Union[bytes, str]) -> None:
        """
        데이터를 저장합니다.

        Args:
            key: 저장 키 (파일명 또는 경로)
            data: 저장할 데이터 (바이트 또는 문자열)
        """
        pass

    @abstractmethod
    def load(self, key: str, as_text: bool = False) -> Union[bytes, str]:
        """
        데이터를 로드합니다.

        Args:
            key: 로드할 키 (파일명 또는 경로)
            as_text: 텍스트로 로드할지 여부

        Returns:
            로드된 데이터 (바이트 또는 문자열)
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        데이터 존재 여부를 확인합니다.

        Args:
            key: 확인할 키

        Returns:
            존재 여부
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        데이터를 삭제합니다.

        Args:
            key: 삭제할 키
        """
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        """
        저장된 파일 목록을 조회합니다.

        Args:
            prefix: 필터링할 접두사

        Returns:
            파일 키 목록
        """
        pass
