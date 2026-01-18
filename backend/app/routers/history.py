# History Router
"""
진단 이력 조회 API
Firebase Firestore 연동 후 구현 예정
"""

from fastapi import APIRouter, HTTPException, Query, status

from app.models.response import ErrorResponse, HistoryResponse

router = APIRouter()


@router.get(
    "/history",
    response_model=HistoryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        404: {"model": ErrorResponse, "description": "사용자 없음"},
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
    summary="진단 이력 조회",
    description="사용자의 과거 진단 이력을 조회합니다.",
)
async def get_history(
    user_id: str = Query(..., description="사용자 ID", min_length=1, max_length=128),
    limit: int = Query(10, description="조회 개수", ge=1, le=100),
    offset: int = Query(0, description="시작 위치", ge=0),
) -> HistoryResponse:
    """
    진단 이력 조회 API

    - **user_id**: 사용자 고유 ID
    - **limit**: 한 번에 조회할 개수 (기본값: 10, 최대: 100)
    - **offset**: 시작 위치 (페이지네이션)
    """
    # TODO: Firebase Firestore 연동 후 구현
    # 1. user_id로 사용자 검증
    # 2. Firestore에서 진단 이력 조회
    # 3. 페이지네이션 적용
    # 4. 결과 반환

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="이 기능은 Firebase 연동 후 구현 예정입니다.",
    )
