# Credential Router
"""
크레덴셜 API
추후 구현 예정
"""

from fastapi import APIRouter, HTTPException, status

from app.models.response import ErrorResponse

router = APIRouter()


@router.post(
    "/credentials",
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
    summary="크레덴셜 생성",
    description="평가 결과를 기반으로 인증서/크레덴셜을 생성합니다.",
)
async def create_credential():
    """
    크레덴셜 생성 API

    특정 평가 결과에 대한 인증서를 생성합니다.
    """
    # TODO: 크레덴셜 기능 구현
    # 1. 평가 결과 검증
    # 2. 인증서 생성
    # 3. Firestore에 저장
    # 4. 인증서 URL 반환

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="이 기능은 추후 구현 예정입니다.",
    )


@router.get(
    "/credentials/{credential_id}",
    responses={
        404: {"model": ErrorResponse, "description": "크레덴셜 없음"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
    summary="크레덴셜 조회",
    description="크레덴셜을 조회합니다.",
)
async def get_credential(credential_id: str):
    """
    크레덴셜 조회 API

    - **credential_id**: 크레덴셜 ID
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="이 기능은 추후 구현 예정입니다.",
    )
