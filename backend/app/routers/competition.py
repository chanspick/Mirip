# Competition Router
"""
공모전 API
추후 구현 예정
"""

from fastapi import APIRouter, HTTPException, status

from app.models.response import ErrorResponse

router = APIRouter()


@router.get(
    "/competitions",
    responses={
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
    summary="공모전 목록 조회",
    description="진행 중인 공모전 목록을 조회합니다.",
)
async def get_competitions():
    """
    공모전 목록 조회 API

    진행 중인 공모전 정보를 반환합니다.
    """
    # TODO: 공모전 기능 구현
    # 1. Firestore에서 공모전 목록 조회
    # 2. 활성 상태인 공모전만 필터링
    # 3. 결과 반환

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="이 기능은 추후 구현 예정입니다.",
    )


@router.get(
    "/competitions/{competition_id}",
    responses={
        404: {"model": ErrorResponse, "description": "공모전 없음"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
    summary="공모전 상세 조회",
    description="특정 공모전의 상세 정보를 조회합니다.",
)
async def get_competition(competition_id: str):
    """
    공모전 상세 조회 API

    - **competition_id**: 공모전 ID
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="이 기능은 추후 구현 예정입니다.",
    )
