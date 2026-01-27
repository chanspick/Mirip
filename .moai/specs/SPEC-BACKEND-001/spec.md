# SPEC-BACKEND-001: MIRIP Backend FastAPI Setup

## TAG BLOCK

```yaml
spec_id: SPEC-BACKEND-001
title: MIRIP Backend FastAPI Initial Setup
status: Completed
priority: High
created: 2025-01-17
lifecycle: spec-anchored
domain: backend
assigned: expert-backend
related_specs: []
labels: [fastapi, python, docker, ml-infrastructure]
```

---

## 1. Environment (환경)

### 1.1 기술 스택

| 영역 | 기술 | 버전 | 용도 |
|------|------|------|------|
| Framework | FastAPI | 0.104+ | REST API 서버 |
| Language | Python | 3.10+ | 백엔드 언어 |
| Validation | Pydantic | 2.x | 데이터 검증 |
| Server | Uvicorn | 0.24+ | ASGI 서버 |
| Container | Docker | 24+ | 컨테이너화 |
| Cache | Redis | 7+ | API 캐싱 |
| ML Runtime | PyTorch | 2.1+ | 딥러닝 프레임워크 |

### 1.2 개발 환경

- **OS**: Windows/Linux/macOS
- **Python**: 3.10 이상 (권장 3.11)
- **Docker**: Docker Desktop 또는 Docker Engine
- **GPU**: NVIDIA GPU with CUDA 12.1 지원 (추론 서버)

### 1.3 디렉토리 구조

```
backend/
├── app/
│   ├── main.py               # FastAPI 앱 엔트리 포인트
│   ├── config.py             # Pydantic Settings 설정 관리
│   ├── routers/              # API 라우터
│   │   ├── __init__.py
│   │   ├── evaluate.py       # 단일 이미지 평가 API
│   │   ├── compare.py        # 복수 이미지 비교 API
│   │   ├── history.py        # 진단 이력 조회 API
│   │   ├── competition.py    # 공모전 API
│   │   └── credential.py     # 크레덴셜 API
│   ├── services/             # 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── inference.py      # ML 추론 서비스
│   │   ├── feedback.py       # 피드백 생성 서비스
│   │   └── storage.py        # 스토리지 관리 서비스
│   ├── models/               # Pydantic 모델
│   │   ├── __init__.py
│   │   ├── request.py        # 요청 모델
│   │   └── response.py       # 응답 모델
│   └── ml/                   # ML 모듈 (추후 구현)
│       ├── __init__.py
│       ├── feature_extractor.py
│       ├── fusion_module.py
│       ├── rubric_heads.py
│       ├── tier_classifier.py
│       └── weights/          # 모델 가중치
├── tests/                    # 테스트 디렉토리
│   ├── __init__.py
│   ├── conftest.py           # pytest fixtures
│   └── test_health.py        # 헬스체크 테스트
├── requirements.txt          # Python 의존성
├── requirements-dev.txt      # 개발 의존성
├── Dockerfile                # Docker 이미지 정의
├── docker-compose.yml        # Docker Compose 설정
└── .env.example              # 환경 변수 예시
```

---

## 2. Assumptions (가정)

### 2.1 기술적 가정

| ID | 가정 | 신뢰도 | 근거 | 검증 방법 |
|----|------|--------|------|-----------|
| A-001 | Python 3.10+ 환경 사용 가능 | High | tech.md 명시 | 버전 확인 |
| A-002 | Docker 환경 구축 가능 | High | tech.md Docker 설정 포함 | Docker 설치 확인 |
| A-003 | Redis 서버 접근 가능 | Medium | docker-compose로 제공 | 연결 테스트 |
| A-004 | CUDA 지원 GPU 사용 가능 (추론 시) | Medium | 로컬 GPU 또는 클라우드 | nvidia-smi 확인 |

### 2.2 비즈니스 가정

| ID | 가정 | 신뢰도 | 근거 |
|----|------|--------|------|
| B-001 | 초기 단계는 단일 서버 배포 | High | MVP 단계 |
| B-002 | API 버전 관리 필요 (v1) | High | 확장성 고려 |
| B-003 | 헬스체크 엔드포인트 필수 | High | 운영 모니터링 |

### 2.3 제약 조건

- [HARD] Python 3.10 이상 버전만 지원
- [HARD] FastAPI 0.104+ 사용 필수 (Pydantic v2 호환)
- [HARD] Docker 이미지는 CUDA 12.1 기반
- [SOFT] 개발 환경에서는 GPU 없이도 동작 가능

---

## 3. Requirements (요구사항)

### 3.1 Ubiquitous Requirements (항상 적용)

| ID | 요구사항 |
|----|----------|
| REQ-U-001 | 시스템은 **항상** 모든 요청에 대해 JSON 형식으로 응답해야 한다 |
| REQ-U-002 | 시스템은 **항상** 요청/응답 로깅을 수행해야 한다 |
| REQ-U-003 | 시스템은 **항상** 환경 변수를 통해 설정을 관리해야 한다 |
| REQ-U-004 | 시스템은 **항상** Pydantic 모델을 통해 입력 데이터를 검증해야 한다 |

### 3.2 Event-Driven Requirements (이벤트 기반)

| ID | 요구사항 |
|----|----------|
| REQ-E-001 | **WHEN** GET /health 요청이 들어오면 **THEN** 서버 상태와 버전 정보를 반환해야 한다 |
| REQ-E-002 | **WHEN** GET /api/v1/health 요청이 들어오면 **THEN** API 상태와 의존성 상태를 반환해야 한다 |
| REQ-E-003 | **WHEN** 애플리케이션이 시작되면 **THEN** 설정을 로드하고 로거를 초기화해야 한다 |
| REQ-E-004 | **WHEN** Docker 컨테이너가 시작되면 **THEN** 헬스체크가 성공해야 한다 |

### 3.3 State-Driven Requirements (상태 기반)

| ID | 요구사항 |
|----|----------|
| REQ-S-001 | **IF** DEBUG 모드가 활성화되어 있으면 **THEN** 상세 에러 메시지를 반환해야 한다 |
| REQ-S-002 | **IF** Redis 연결이 실패하면 **THEN** 캐시 없이 동작하고 경고 로그를 남겨야 한다 |
| REQ-S-003 | **IF** 환경 변수가 누락되면 **THEN** 기본값을 사용하거나 시작 시 오류를 발생시켜야 한다 |

### 3.4 Unwanted Requirements (금지 사항)

| ID | 요구사항 |
|----|----------|
| REQ-N-001 | 시스템은 하드코딩된 시크릿 값을 코드에 포함**하지 않아야 한다** |
| REQ-N-002 | 시스템은 스택 트레이스를 프로덕션 응답에 노출**하지 않아야 한다** |
| REQ-N-003 | 시스템은 동기 블로킹 I/O를 메인 이벤트 루프에서 실행**하지 않아야 한다** |

### 3.5 Optional Requirements (선택 사항)

| ID | 요구사항 |
|----|----------|
| REQ-O-001 | **가능하면** Swagger UI와 ReDoc 문서를 /docs와 /redoc에서 제공 |
| REQ-O-002 | **가능하면** CORS 설정을 통해 프론트엔드 개발 환경 지원 |
| REQ-O-003 | **가능하면** 요청 ID 기반 분산 추적 지원 |

---

## 4. Specifications (세부 명세)

### 4.1 FastAPI 애플리케이션 (main.py)

```python
# 예상 구조
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import evaluate, compare, competition, credential

app = FastAPI(
    title="MIRIP API",
    version="0.1.0",
    description="미술 입시 AI 진단 플랫폼 API"
)

# CORS 미들웨어
# 라우터 등록
# 헬스체크 엔드포인트
```

### 4.2 설정 관리 (config.py)

```python
# Pydantic Settings 활용
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 서버 설정
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # 모델 설정
    MODEL_WEIGHTS_PATH: str = "/app/weights"
    DEVICE: str = "cuda"

    # Redis 설정
    REDIS_URL: str = "redis://localhost:6379"

    # Firebase 설정
    FIREBASE_CREDENTIALS_PATH: str = ""

    # OpenAI 설정 (LLM 피드백 생성용)
    OPENAI_API_KEY: str = ""

    class Config:
        env_file = ".env"
```

### 4.3 Pydantic 모델

#### 요청 모델 (request.py)

| 모델명 | 필드 | 타입 | 설명 |
|--------|------|------|------|
| EvaluateRequest | image | UploadFile | 평가할 이미지 (multipart/form-data) |
| | department | str | 학과 (visual_design/industrial_design/fine_art/craft) |
| | theme | str \| None | 주제 (예: "자연과 인간의 공존") |
| | include_feedback | bool | 피드백 포함 여부 (기본값: true) |
| | language | str | 응답 언어 (기본값: "ko") |
| CompareRequest | images | list[UploadFile] | 비교할 이미지들 (2-10개) |
| | department | str | 학과 |
| HistoryRequest | user_id | str | 사용자 ID |
| | limit | int | 조회 개수 (기본값: 10) |

#### 응답 모델 (response.py)

| 모델명 | 필드 | 타입 | 설명 |
|--------|------|------|------|
| HealthResponse | status | str | "healthy" / "unhealthy" |
| | version | str | API 버전 |
| | timestamp | datetime | 응답 시간 |
| | dependencies | dict | 의존성 상태 |
| EvaluateResponse | tier | str | S/A/B/C 등급 |
| | scores | dict | 4축 점수 |
| | probabilities | dict | 대학별 합격 확률 |
| | feedback | dict | 강점/개선점 |
| ErrorResponse | error | str | 에러 코드 |
| | message | str | 에러 메시지 |
| | detail | dict | 상세 정보 (DEBUG 모드) |

### 4.4 API 엔드포인트

| 메서드 | 경로 | 설명 | 상태 |
|--------|------|------|------|
| GET | /health | 기본 헬스체크 | 이번 SPEC |
| GET | /api/v1/health | API 헬스체크 (의존성 포함) | 이번 SPEC |
| POST | /api/v1/evaluate | 단일 이미지 평가 | Placeholder |
| POST | /api/v1/compare | 복수 이미지 비교 | Placeholder |
| GET | /api/v1/history | 진단 이력 조회 | Placeholder |
| GET | /api/v1/competitions | 공모전 목록 | Placeholder |
| POST | /api/v1/credentials | 크레덴셜 생성 | Placeholder |

### 4.5 Docker 설정

#### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./weights:/app/weights:ro
      - ./logs:/app/logs
    environment:
      - DEBUG=true
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### 4.6 의존성 (requirements.txt)

```txt
# Core
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# ML (placeholder - 추후 활성화)
# torch>=2.1.0
# transformers>=4.35.0
# timm>=0.9.0

# Image Processing
# opencv-python>=4.8.0
# pillow>=10.0.0
# albumentations>=1.3.0

# Cache & Storage
redis>=5.0.0

# Utilities
python-multipart>=0.0.6
python-dotenv>=1.0.0
httpx>=0.25.0

# Logging
structlog>=23.2.0
```

---

## 5. Traceability (추적성)

### 5.1 연관 문서

| 문서 | 경로 | 관계 |
|------|------|------|
| 기술 스택 | `.moai/project/tech.md` | 기술 선택 근거 |
| 프로젝트 구조 | `.moai/project/structure.md` | 디렉토리 구조 정의 |
| 제품 문서 | `.moai/project/product.md` | 비즈니스 요구사항 |

### 5.2 후속 SPEC

| SPEC ID | 제목 | 의존성 |
|---------|------|--------|
| SPEC-BACKEND-002 | ML Inference Service | 이 SPEC 완료 후 |
| SPEC-BACKEND-003 | Evaluate API Implementation | SPEC-BACKEND-002 완료 후 |
| SPEC-BACKEND-004 | Compare API Implementation | SPEC-BACKEND-002 완료 후 |

### 5.3 Quality Gates

- [ ] 모든 EARS 요구사항이 테스트 가능
- [ ] 85% 이상 테스트 커버리지
- [ ] Ruff 린터 경고 없음
- [ ] Docker 빌드 성공
- [ ] 헬스체크 엔드포인트 동작 확인

---

*SPEC 버전: 1.0.0*
*작성일: 2025-01-17*
*담당 에이전트: expert-backend*
