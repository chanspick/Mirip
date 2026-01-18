# Request Models
"""
MIRIP API 요청 모델
Pydantic v2 모델을 활용한 입력 데이터 검증
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ======================
# 공통 타입
# ======================
DepartmentType = Literal[
    "visual_design",
    "industrial_design",
    "fine_art",
    "craft",
]

LanguageType = Literal["ko", "en"]


# ======================
# 평가 요청
# ======================
class EvaluateRequest(BaseModel):
    """단일 이미지 평가 요청 (Form 데이터)"""

    # 참고: image는 UploadFile로 라우터에서 직접 처리
    department: DepartmentType = Field(
        ...,
        description="학과 (visual_design/industrial_design/fine_art/craft)",
    )
    theme: Optional[str] = Field(
        None,
        description="주제 (예: '자연과 인간의 공존')",
        max_length=200,
    )
    include_feedback: bool = Field(
        True,
        description="피드백 포함 여부",
    )
    language: LanguageType = Field(
        "ko",
        description="응답 언어",
    )


# ======================
# 비교 요청
# ======================
class CompareRequest(BaseModel):
    """복수 이미지 비교 요청 (Form 데이터)"""

    # 참고: images는 list[UploadFile]로 라우터에서 직접 처리
    department: DepartmentType = Field(
        ...,
        description="학과",
    )


# ======================
# 이력 조회 요청
# ======================
class HistoryRequest(BaseModel):
    """진단 이력 조회 요청"""

    user_id: str = Field(
        ...,
        description="사용자 ID",
        min_length=1,
        max_length=128,
    )
    limit: int = Field(
        10,
        description="조회 개수",
        ge=1,
        le=100,
    )
    offset: int = Field(
        0,
        description="시작 위치",
        ge=0,
    )
