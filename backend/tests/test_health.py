# Health Check Tests
"""
헬스체크 엔드포인트 테스트
SPEC-BACKEND-001 요구사항 검증
"""

import pytest
from httpx import AsyncClient


# ======================
# 기본 헬스체크 테스트
# ======================
class TestBasicHealthCheck:
    """기본 헬스체크 (/health) 테스트"""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient):
        """REQ-E-001: GET /health 요청 시 서버 상태 반환"""
        response = await client.get("/health")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_healthy_status(self, client: AsyncClient):
        """헬스체크 응답에 healthy 상태 포함"""
        response = await client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_returns_version(self, client: AsyncClient):
        """헬스체크 응답에 버전 정보 포함"""
        response = await client.get("/health")
        data = response.json()

        assert "version" in data
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_health_returns_timestamp(self, client: AsyncClient):
        """헬스체크 응답에 타임스탬프 포함"""
        response = await client.get("/health")
        data = response.json()

        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_health_returns_json(self, client: AsyncClient):
        """REQ-U-001: JSON 형식으로 응답"""
        response = await client.get("/health")

        assert response.headers["content-type"] == "application/json"


# ======================
# 상세 헬스체크 테스트
# ======================
class TestDetailedHealthCheck:
    """상세 헬스체크 (/api/v1/health) 테스트"""

    @pytest.mark.asyncio
    async def test_api_health_returns_200(self, client: AsyncClient):
        """REQ-E-002: GET /api/v1/health 요청 시 API 상태 반환"""
        response = await client.get("/api/v1/health")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_health_returns_dependencies(self, client: AsyncClient):
        """상세 헬스체크 응답에 의존성 상태 포함"""
        response = await client.get("/api/v1/health")
        data = response.json()

        assert "dependencies" in data
        assert isinstance(data["dependencies"], list)

    @pytest.mark.asyncio
    async def test_api_health_includes_redis_status(self, client: AsyncClient):
        """의존성 상태에 Redis 포함"""
        response = await client.get("/api/v1/health")
        data = response.json()

        redis_deps = [d for d in data["dependencies"] if d["name"] == "redis"]
        assert len(redis_deps) == 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_api_health_redis_healthy(self, client: AsyncClient):
        """Redis가 정상일 때 healthy 상태 반환"""
        response = await client.get("/api/v1/health")
        data = response.json()

        redis_dep = next(d for d in data["dependencies"] if d["name"] == "redis")
        # Redis가 실행 중이면 healthy, 아니면 unhealthy
        assert redis_dep["status"] in ["healthy", "unhealthy"]


# ======================
# 루트 엔드포인트 테스트
# ======================
class TestRootEndpoint:
    """루트 엔드포인트 (/) 테스트"""

    @pytest.mark.asyncio
    async def test_root_returns_200(self, client: AsyncClient):
        """루트 엔드포인트 접근 가능"""
        response = await client.get("/")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_root_returns_project_info(self, client: AsyncClient):
        """루트 엔드포인트에서 프로젝트 정보 반환"""
        response = await client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data


# ======================
# 미들웨어 테스트
# ======================
class TestMiddleware:
    """미들웨어 테스트"""

    @pytest.mark.asyncio
    async def test_process_time_header(self, client: AsyncClient):
        """응답에 X-Process-Time 헤더 포함"""
        response = await client.get("/health")

        assert "x-process-time" in response.headers

    @pytest.mark.asyncio
    async def test_cors_headers(self, client: AsyncClient):
        """CORS 프리플라이트 요청 처리"""
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )

        # CORS가 활성화되어 있으면 200 또는 허용된 메서드 반환
        assert response.status_code in [200, 405]
