# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: BaseCollector 추상 클래스

데이터 수집기의 기본 인터페이스를 정의합니다.
모든 수집기는 이 추상 클래스를 상속받아 구현합니다.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from pydantic import BaseModel


class CollectionResult(BaseModel):
    """
    수집 결과 모델

    각 이미지 수집 작업의 결과를 나타냅니다.
    """

    image_id: str
    source_path: str
    local_path: str
    success: bool
    error_message: Optional[str] = None

    model_config = {
        "str_strip_whitespace": True,
    }


class BaseCollector(ABC):
    """
    데이터 수집기 추상 베이스 클래스

    다양한 소스에서 이미지를 수집하는 기능의 인터페이스를 정의합니다.
    Firebase, 로컬 파일 시스템, 외부 API 등의 수집기가 이를 구현합니다.
    """

    @abstractmethod
    def collect(self, source: str) -> Iterator[CollectionResult]:
        """
        소스에서 이미지를 수집합니다.

        Args:
            source: 수집 소스 경로 또는 식별자

        Yields:
            CollectionResult: 각 이미지의 수집 결과
        """
        pass

    @abstractmethod
    def validate(self, path: str) -> bool:
        """
        파일의 유효성을 검증합니다.

        Args:
            path: 검증할 파일 경로

        Returns:
            bool: 유효한 이미지 파일이면 True
        """
        pass
