# -*- coding: utf-8 -*-
"""
SPEC-DATA-001: 메타데이터 모델

AI 진단 데이터 파이프라인을 위한 Pydantic 기반 메타데이터 스키마입니다.
이미지 메타데이터, 과별 분류, 대학 티어 정보를 정의합니다.
"""

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class Department(str, Enum):
    """
    과별 분류 열거형

    미술 입시 4개 주요 분야:
    - 시디 (시각디자인)
    - 산디 (산업디자인)
    - 공예
    - 회화
    """

    VISUAL_DESIGN = "시디"  # 시각디자인
    INDUSTRIAL_DESIGN = "산디"  # 산업디자인
    CRAFT = "공예"
    PAINTING = "회화"
    UNKNOWN = "미분류"


class Tier(str, Enum):
    """
    대학 티어 열거형

    대학 합격권 기준 4단계:
    - S: 최상위 (서울대, 홍대, 국민대 등) - 점수 85-100
    - A: 상위 (건국대, 동국대 등) - 점수 70-84
    - B: 중위 (단국대, 명지대 등) - 점수 50-69
    - C: 개선 필요 - 점수 0-49
    """

    S = "S"  # Top tier universities
    A = "A"
    B = "B"
    C = "C"
    UNKNOWN = "미분류"


class Medium(str, Enum):
    """
    매체 열거형

    작품 제작에 사용된 매체/재료
    """

    PENCIL = "pencil"  # 연필
    CHARCOAL = "charcoal"  # 목탄
    WATERCOLOR = "watercolor"  # 수채화
    OIL = "oil"  # 유화
    ACRYLIC = "acrylic"  # 아크릴
    DIGITAL = "digital"  # 디지털
    MIXED = "mixed"  # 혼합매체


class ImageMetadata(BaseModel):
    """
    이미지 메타데이터 모델

    수집된 이미지의 모든 메타정보를 저장합니다.
    Pydantic v2 기반으로 유효성 검증을 수행합니다.
    """

    # 필수 식별 정보
    image_id: str = Field(..., description="고유 이미지 ID")
    original_filename: str = Field(..., description="원본 파일명")
    file_path: str = Field(..., description="저장 경로")
    file_size: int = Field(..., gt=0, description="파일 크기 (bytes)")
    width: int = Field(..., gt=0, description="이미지 너비 (pixels)")
    height: int = Field(..., gt=0, description="이미지 높이 (pixels)")
    format: str = Field(..., description="이미지 형식 (JPEG, PNG 등)")

    # 과별 분류 정보
    department: Department = Field(
        default=Department.UNKNOWN, description="과별 분류"
    )

    # 티어 라벨 정보
    tier: Tier = Field(default=Tier.UNKNOWN, description="대학 티어")
    tier_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="티어 점수 (0-100)"
    )
    is_manual_label: bool = Field(
        default=False, description="수동 라벨 여부"
    )

    # 출처 및 권한 정보
    source: str = Field(default="", description="수집 소스")
    consent_status: bool = Field(
        default=True, description="사용 동의 여부"
    )
    copyright_cleared: bool = Field(
        default=True, description="저작권 확보 여부"
    )

    # 시간 정보
    created_at: datetime = Field(
        default_factory=datetime.now, description="생성 시간"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="수정 시간"
    )

    # 태그 정보
    tags: List[str] = Field(
        default_factory=list, description="주제 키워드 태그"
    )

    model_config = {
        "use_enum_values": False,  # Enum 객체 그대로 유지
        "str_strip_whitespace": True,
        "validate_default": True,
    }
