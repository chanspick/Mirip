# Response Models
"""
MIRIP API 응답 모델
Pydantic v2 모델을 활용한 응답 데이터 구조화
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ======================
# 헬스체크 응답
# ======================
class DependencyStatus(BaseModel):
    """의존성 상태"""

    name: str
    status: str = Field(description="healthy / unhealthy / unknown")
    latency_ms: Optional[float] = Field(None, description="응답 시간 (ms)")


class HealthResponse(BaseModel):
    """헬스체크 응답"""

    status: str = Field(description="healthy / unhealthy")
    version: str = Field(description="API 버전")
    timestamp: datetime = Field(default_factory=datetime.now)
    dependencies: Optional[list[DependencyStatus]] = Field(
        None,
        description="의존성 상태 (상세 헬스체크 시)",
    )


# ======================
# 평가 응답
# ======================
class RubricScores(BaseModel):
    """4축 루브릭 점수"""

    composition: float = Field(description="구도 및 구성 (0-100)")
    technique: float = Field(description="표현 기법 (0-100)")
    creativity: float = Field(description="창의성 (0-100)")
    completeness: float = Field(description="완성도 (0-100)")


class Probability(BaseModel):
    """대학별 합격 확률"""

    university: str = Field(description="대학명")
    department: str = Field(description="학과명")
    probability: float = Field(description="합격 확률 (0-1)")


class Feedback(BaseModel):
    """피드백"""

    strengths: list[str] = Field(description="강점")
    improvements: list[str] = Field(description="개선점")
    overall: str = Field(description="종합 피드백")


class EvaluateResponse(BaseModel):
    """평가 응답"""

    evaluation_id: str = Field(description="평가 ID")
    tier: str = Field(description="등급 (S/A/B/C)")
    scores: RubricScores = Field(description="4축 점수")
    probabilities: list[Probability] = Field(description="대학별 합격 확률")
    feedback: Optional[Feedback] = Field(None, description="피드백")
    created_at: datetime = Field(default_factory=datetime.now)


# ======================
# 비교 응답
# ======================
class CompareItem(BaseModel):
    """비교 항목"""

    image_index: int = Field(description="이미지 인덱스 (0-based)")
    tier: str = Field(description="등급")
    scores: RubricScores = Field(description="점수")
    rank: int = Field(description="순위")


class CompareResponse(BaseModel):
    """비교 응답"""

    comparison_id: str = Field(description="비교 ID")
    items: list[CompareItem] = Field(description="비교 결과")
    summary: str = Field(description="비교 요약")
    created_at: datetime = Field(default_factory=datetime.now)


# ======================
# 이력 응답
# ======================
class HistoryItem(BaseModel):
    """이력 항목"""

    evaluation_id: str
    tier: str
    scores: RubricScores
    created_at: datetime
    thumbnail_url: Optional[str] = None


class HistoryResponse(BaseModel):
    """이력 응답"""

    user_id: str
    items: list[HistoryItem]
    total: int
    has_more: bool


# ======================
# 에러 응답
# ======================
class ErrorResponse(BaseModel):
    """에러 응답"""

    error: str = Field(description="에러 코드")
    message: str = Field(description="에러 메시지")
    detail: Optional[dict[str, Any]] = Field(
        None,
        description="상세 정보 (DEBUG 모드에서만 포함)",
    )
