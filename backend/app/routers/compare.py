# Compare Router
"""
복수 이미지 비교 API
SPEC-BACKEND-004에서 본격 구현 예정
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.models.request import DepartmentType
from app.models.response import CompareResponse, ErrorResponse

router = APIRouter()


@router.post(
    "/compare",
    response_model=CompareResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
    summary="복수 이미지 비교",
    description="여러 이미지를 비교하여 상대적 순위와 점수를 반환합니다.",
)
async def compare_images(
    images: list[UploadFile] = File(..., description="비교할 이미지들 (2-10개)"),
    department: DepartmentType = Form(..., description="학과"),
) -> CompareResponse:
    """
    복수 이미지 비교 API

    - **images**: 비교할 이미지 파일들 (2-10개)
    - **department**: 학과
    """
    # 이미지 개수 검증
    if len(images) < 2 or len(images) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미지는 2개 이상 10개 이하로 제출해야 합니다.",
        )

    # 이미지 형식 검증
    for idx, img in enumerate(images):
        if img.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"이미지 {idx + 1}: 지원하지 않는 형식입니다.",
            )

    # TODO: SPEC-BACKEND-004에서 구현
    # 1. 모든 이미지 전처리
    # 2. ML 모델 추론 (배치)
    # 3. 상대적 순위 계산
    # 4. 비교 요약 생성

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="이 기능은 SPEC-BACKEND-004에서 구현 예정입니다.",
    )
