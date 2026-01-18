# Evaluate Router
"""
단일 이미지 평가 API
SPEC-BACKEND-003에서 본격 구현 예정
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.models.request import DepartmentType, LanguageType
from app.models.response import ErrorResponse, EvaluateResponse

router = APIRouter()


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
    summary="단일 이미지 평가",
    description="제출된 이미지를 AI가 분석하여 4축 루브릭 점수, 등급, 합격 확률을 반환합니다.",
)
async def evaluate_image(
    image: UploadFile = File(..., description="평가할 이미지"),
    department: DepartmentType = Form(..., description="학과"),
    theme: str | None = Form(None, description="주제"),
    include_feedback: bool = Form(True, description="피드백 포함 여부"),
    language: LanguageType = Form("ko", description="응답 언어"),
) -> EvaluateResponse:
    """
    단일 이미지 평가 API

    - **image**: 평가할 이미지 파일 (JPEG, PNG)
    - **department**: 학과 (visual_design, industrial_design, fine_art, craft)
    - **theme**: 주제 (선택사항)
    - **include_feedback**: 피드백 포함 여부
    - **language**: 응답 언어 (ko, en)
    """
    # 이미지 형식 검증
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="지원하지 않는 이미지 형식입니다. JPEG 또는 PNG만 지원합니다.",
        )

    # TODO: SPEC-BACKEND-003에서 구현
    # 1. 이미지 전처리
    # 2. ML 모델 추론
    # 3. 피드백 생성 (OpenAI)
    # 4. 결과 반환

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="이 기능은 SPEC-BACKEND-003에서 구현 예정입니다.",
    )
