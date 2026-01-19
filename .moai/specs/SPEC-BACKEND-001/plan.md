# SPEC-BACKEND-001 Implementation Plan

## TAG BLOCK

```yaml
spec_id: SPEC-BACKEND-001
document_type: plan
version: 1.0.0
created: 2025-01-17
```

---

## 1. Implementation Overview

### 1.1 목표

MIRIP 백엔드 FastAPI 프로젝트의 초기 설정을 완료하여 개발 기반을 마련한다.

### 1.2 범위

- 프로젝트 디렉토리 구조 생성
- FastAPI 애플리케이션 기본 설정
- Pydantic 기반 설정 관리
- 기본 Pydantic 모델 정의
- Docker 환경 구성
- 헬스체크 엔드포인트 구현

### 1.3 범위 외

- ML 모델 통합 (SPEC-BACKEND-002)
- 실제 API 비즈니스 로직 구현 (SPEC-BACKEND-003, 004)
- 데이터베이스 연동
- 인증/인가 시스템

---

## 2. Milestones

### Primary Goal: 프로젝트 구조 및 FastAPI 기본 설정

**우선순위: High**

| Task ID | 작업 | 산출물 |
|---------|------|--------|
| T-001 | 디렉토리 구조 생성 | backend/ 폴더 구조 |
| T-002 | requirements.txt 작성 | requirements.txt |
| T-003 | Pydantic Settings 설정 | app/config.py |
| T-004 | FastAPI 앱 생성 | app/main.py |
| T-005 | 헬스체크 엔드포인트 | GET /health, GET /api/v1/health |

**완료 조건:**
- `uvicorn app.main:app --reload` 실행 성공
- `/health` 엔드포인트 200 응답
- `/docs` Swagger UI 접근 가능

---

### Secondary Goal: Pydantic 모델 및 라우터 Placeholder

**우선순위: Medium**

| Task ID | 작업 | 산출물 |
|---------|------|--------|
| T-006 | Request 모델 정의 | app/models/request.py |
| T-007 | Response 모델 정의 | app/models/response.py |
| T-008 | Evaluate 라우터 Placeholder | app/routers/evaluate.py |
| T-009 | Compare 라우터 Placeholder | app/routers/compare.py |
| T-010 | Competition 라우터 Placeholder | app/routers/competition.py |
| T-011 | Credential 라우터 Placeholder | app/routers/credential.py |

**완료 조건:**
- 모든 라우터가 main.py에 등록
- Swagger UI에서 모든 엔드포인트 확인 가능
- Placeholder 엔드포인트가 501 Not Implemented 반환

---

### Tertiary Goal: 서비스 레이어 및 Docker 설정

**우선순위: Medium**

| Task ID | 작업 | 산출물 |
|---------|------|--------|
| T-012 | Inference 서비스 Placeholder | app/services/inference.py |
| T-013 | Feedback 서비스 Placeholder | app/services/feedback.py |
| T-014 | Storage 서비스 Placeholder | app/services/storage.py |
| T-015 | Dockerfile 작성 | Dockerfile |
| T-016 | docker-compose.yml 작성 | docker-compose.yml |
| T-017 | .env.example 작성 | .env.example |

**완료 조건:**
- `docker-compose up --build` 성공
- 컨테이너 내에서 헬스체크 성공
- Redis 컨테이너 정상 동작

---

### Final Goal: 테스트 및 문서화

**우선순위: Medium**

| Task ID | 작업 | 산출물 |
|---------|------|--------|
| T-018 | pytest 설정 | tests/conftest.py |
| T-019 | 헬스체크 테스트 작성 | tests/test_health.py |
| T-020 | ML 모듈 Placeholder | app/ml/__init__.py 등 |
| T-021 | README.md 작성 | backend/README.md |

**완료 조건:**
- `pytest` 실행 시 모든 테스트 통과
- 테스트 커버리지 85% 이상
- README에 설치 및 실행 방법 문서화

---

## 3. Technical Approach

### 3.1 FastAPI 애플리케이션 구조

```
Application Factory Pattern 사용:
- main.py에서 create_app() 함수로 앱 생성
- 설정, 미들웨어, 라우터를 순차적으로 등록
- Lifespan 컨텍스트 매니저로 시작/종료 이벤트 관리
```

### 3.2 설정 관리 전략

```
Environment-based Configuration:
- Pydantic BaseSettings 활용
- .env 파일에서 환경 변수 로드
- 개발/스테이징/프로덕션 환경 분리
- 필수 설정은 validation으로 강제
```

### 3.3 라우터 구조

```
API Versioning:
- /api/v1/ 프리픽스로 버전 관리
- 각 도메인별 라우터 분리 (evaluate, compare, competition, credential)
- 공통 의존성은 dependencies.py에서 관리
```

### 3.4 에러 처리 전략

```
Centralized Error Handling:
- HTTPException을 통한 일관된 에러 응답
- ErrorResponse 모델로 구조화된 에러 형식
- DEBUG 모드에서만 상세 정보 노출
```

---

## 4. Architecture Design

### 4.1 레이어 아키텍처

```
┌─────────────────────────────────────────┐
│              Routers (API Layer)         │
│  evaluate.py, compare.py, competition.py │
├─────────────────────────────────────────┤
│            Services (Business Layer)     │
│  inference.py, feedback.py, storage.py   │
├─────────────────────────────────────────┤
│              Models (Data Layer)         │
│      request.py, response.py             │
├─────────────────────────────────────────┤
│           ML Modules (Core Layer)        │
│  feature_extractor, fusion, rubric_heads │
└─────────────────────────────────────────┘
```

### 4.2 의존성 흐름

```
Request
   │
   ▼
Router (입력 검증, 라우팅)
   │
   ▼
Service (비즈니스 로직)
   │
   ▼
ML Module (추론 처리)
   │
   ▼
Response
```

### 4.3 설정 로딩 흐름

```
.env 파일
   │
   ▼
Pydantic Settings (검증)
   │
   ▼
settings 싱글톤
   │
   ▼
각 모듈에서 import하여 사용
```

---

## 5. Risks and Mitigations

### 5.1 기술적 리스크

| 리스크 | 영향도 | 발생 확률 | 대응 방안 |
|--------|--------|-----------|-----------|
| PyTorch CUDA 호환성 | High | Medium | CPU fallback 지원, 버전 고정 |
| Docker GPU 접근 | Medium | Medium | nvidia-container-toolkit 문서화 |
| Redis 연결 실패 | Low | Low | 캐시 비활성화 fallback |

### 5.2 의존성 리스크

| 리스크 | 영향도 | 발생 확률 | 대응 방안 |
|--------|--------|-----------|-----------|
| Pydantic v2 마이그레이션 이슈 | Medium | Low | FastAPI 0.104+ 사용으로 해결 |
| transformers 버전 충돌 | Medium | Medium | 버전 범위 명시적 지정 |

---

## 6. Implementation Order

### Phase 1: 기반 구축

```
1. backend/ 디렉토리 생성
2. requirements.txt 작성
3. app/__init__.py 생성
4. app/config.py 구현
5. app/main.py 구현 (기본 앱 + 헬스체크)
```

### Phase 2: 모델 및 라우터

```
6. app/models/request.py 구현
7. app/models/response.py 구현
8. app/routers/evaluate.py (placeholder)
9. app/routers/compare.py (placeholder)
10. app/routers/competition.py (placeholder)
11. app/routers/credential.py (placeholder)
```

### Phase 3: 서비스 및 Docker

```
12. app/services/inference.py (placeholder)
13. app/services/feedback.py (placeholder)
14. app/services/storage.py (placeholder)
15. Dockerfile 작성
16. docker-compose.yml 작성
17. .env.example 작성
```

### Phase 4: 테스트 및 마무리

```
18. tests/conftest.py 설정
19. tests/test_health.py 작성
20. app/ml/ placeholder 파일 생성
21. backend/README.md 작성
22. 최종 검증 및 테스트
```

---

## 7. Acceptance Criteria Reference

상세한 수락 기준은 `acceptance.md` 문서를 참조하세요.

### 핵심 검증 항목

- [ ] 모든 디렉토리 및 파일 생성 완료
- [ ] `uvicorn app.main:app --reload` 성공
- [ ] GET /health 응답 확인
- [ ] GET /api/v1/health 응답 확인
- [ ] Swagger UI 접근 가능
- [ ] `docker-compose up --build` 성공
- [ ] pytest 모든 테스트 통과
- [ ] 테스트 커버리지 85% 이상

---

## 8. Expert Consultation Recommendations

### 추천 전문가 상담

| 영역 | 담당 에이전트 | 상담 시점 | 상담 내용 |
|------|---------------|-----------|-----------|
| Backend | expert-backend | 구현 시작 전 | FastAPI 아키텍처 검토 |
| DevOps | expert-devops | Docker 설정 시 | 컨테이너 최적화 |
| Security | expert-security | 설정 완료 후 | 환경 변수 보안 검토 |

---

*Plan 버전: 1.0.0*
*작성일: 2025-01-17*
*연관 SPEC: SPEC-BACKEND-001*
