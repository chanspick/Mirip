# FastAPI Application Entry Point
"""
MIRIP Backend API 서버
미술 입시 AI 진단 플랫폼
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.response import DependencyStatus, ErrorResponse, HealthResponse
from app.routers import compare, competition, credential, evaluate, history

# ======================
# 로깅 설정
# ======================
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if not settings.DEBUG else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# ======================
# Lifespan 이벤트
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """애플리케이션 수명주기 관리"""
    # Startup
    logger.info(
        "Starting MIRIP API Server",
        version=settings.PROJECT_VERSION,
        debug=settings.DEBUG,
    )
    yield
    # Shutdown
    logger.info("Shutting down MIRIP API Server")


# ======================
# FastAPI 애플리케이션
# ======================
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=settings.PROJECT_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)


# ======================
# 미들웨어
# ======================
# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)


# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청/응답 로깅"""
    start_time = time.time()

    # 요청 로깅
    logger.info(
        "Request received",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown",
    )

    response = await call_next(request)

    # 응답 로깅
    process_time = time.time() - start_time
    logger.info(
        "Request completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time_ms=round(process_time * 1000, 2),
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response


# ======================
# 예외 처리
# ======================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 처리"""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        exc_info=exc,
    )

    error_response = ErrorResponse(
        error="INTERNAL_ERROR",
        message="내부 서버 오류가 발생했습니다.",
        detail={"exception": str(exc)} if settings.DEBUG else None,
    )

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


# ======================
# 헬스체크 엔드포인트
# ======================
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="기본 헬스체크",
)
async def health_check() -> HealthResponse:
    """
    서버 상태 확인

    서버가 정상적으로 동작하는지 확인합니다.
    """
    return HealthResponse(
        status="healthy",
        version=settings.PROJECT_VERSION,
        timestamp=datetime.now(),
    )


@app.get(
    f"{settings.API_V1_PREFIX}/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="상세 헬스체크",
)
async def health_check_detailed() -> HealthResponse:
    """
    API 상태 및 의존성 확인

    서버 상태와 함께 Redis 등 의존성 상태를 확인합니다.
    """
    dependencies = []

    # Redis 상태 확인
    redis_status = await _check_redis_health()
    dependencies.append(redis_status)

    # 전체 상태 결정
    overall_status = "healthy" if all(
        d.status == "healthy" for d in dependencies
    ) else "unhealthy"

    return HealthResponse(
        status=overall_status,
        version=settings.PROJECT_VERSION,
        timestamp=datetime.now(),
        dependencies=dependencies,
    )


async def _check_redis_health() -> DependencyStatus:
    """Redis 헬스체크"""
    try:
        import redis.asyncio as redis

        start = time.time()
        client = redis.from_url(settings.REDIS_URL, socket_timeout=settings.REDIS_TIMEOUT)
        await client.ping()
        await client.close()
        latency = (time.time() - start) * 1000

        return DependencyStatus(
            name="redis",
            status="healthy",
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.warning("Redis health check failed", error=str(e))
        return DependencyStatus(
            name="redis",
            status="unhealthy",
            latency_ms=None,
        )


# ======================
# API 라우터 등록
# ======================
app.include_router(
    evaluate.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Evaluate"],
)
app.include_router(
    compare.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Compare"],
)
app.include_router(
    history.router,
    prefix=settings.API_V1_PREFIX,
    tags=["History"],
)
app.include_router(
    competition.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Competition"],
)
app.include_router(
    credential.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Credential"],
)


# ======================
# 루트 엔드포인트
# ======================
@app.get("/", include_in_schema=False)
async def root():
    """루트 리다이렉트"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "docs": "/docs" if settings.DEBUG else None,
    }
