# Configuration Management
"""
MIRIP Backend 설정 관리
Pydantic Settings를 활용한 환경 변수 기반 설정
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ======================
    # 서버 설정
    # ======================
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # ======================
    # API 설정
    # ======================
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "MIRIP API"
    PROJECT_VERSION: str = "0.1.0"
    PROJECT_DESCRIPTION: str = "미술 입시 AI 진단 플랫폼 API"

    # ======================
    # 모델 설정
    # ======================
    MODEL_WEIGHTS_PATH: str = "/app/weights"
    DEVICE: str = "cuda"

    # ======================
    # Redis 설정
    # ======================
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TIMEOUT: int = 5

    # ======================
    # Firebase 설정
    # ======================
    FIREBASE_CREDENTIALS_PATH: Optional[str] = None

    # ======================
    # OpenAI 설정 (LLM 피드백 생성용)
    # ======================
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ======================
    # CORS 설정
    # ======================
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """설정 인스턴스를 캐시하여 반환"""
    return Settings()


# 전역 설정 인스턴스
settings = get_settings()
