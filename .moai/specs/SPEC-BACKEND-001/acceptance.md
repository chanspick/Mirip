# SPEC-BACKEND-001 Acceptance Criteria

## TAG BLOCK

```yaml
spec_id: SPEC-BACKEND-001
document_type: acceptance
version: 1.0.0
created: 2025-01-17
```

---

## 1. Acceptance Criteria Overview

### 1.1 Definition of Done

SPEC-BACKEND-001은 다음 조건이 모두 충족될 때 완료로 간주합니다:

- [ ] 모든 필수 파일 및 디렉토리 생성
- [ ] FastAPI 애플리케이션 정상 실행
- [ ] 헬스체크 엔드포인트 동작 확인
- [ ] Docker 환경 구축 및 실행 성공
- [ ] 테스트 커버리지 85% 이상 달성
- [ ] 코드 품질 기준 충족 (린터 경고 없음)

### 1.2 Quality Gates

| Gate | 기준 | 측정 방법 |
|------|------|-----------|
| Test Coverage | >= 85% | pytest --cov |
| Lint | 0 warnings | ruff check |
| Type Check | 0 errors | mypy |
| Build | Success | docker-compose up --build |
| Health Check | 200 OK | curl /health |

---

## 2. Test Scenarios

### 2.1 프로젝트 구조 검증

#### TC-001: 디렉토리 구조 확인

```gherkin
Feature: 프로젝트 디렉토리 구조
  As a 개발자
  I want 표준화된 디렉토리 구조
  So that 일관된 개발 환경을 유지할 수 있다

  Scenario: 필수 디렉토리 존재 확인
    Given backend 디렉토리가 존재한다
    When 디렉토리 구조를 확인한다
    Then 다음 디렉토리가 존재해야 한다:
      | 경로 |
      | backend/app/ |
      | backend/app/routers/ |
      | backend/app/services/ |
      | backend/app/models/ |
      | backend/app/ml/ |
      | backend/tests/ |

  Scenario: 필수 파일 존재 확인
    Given backend 디렉토리가 존재한다
    When 파일 구조를 확인한다
    Then 다음 파일이 존재해야 한다:
      | 파일 |
      | backend/app/main.py |
      | backend/app/config.py |
      | backend/requirements.txt |
      | backend/Dockerfile |
      | backend/docker-compose.yml |
```

---

### 2.2 FastAPI 애플리케이션 검증

#### TC-002: 애플리케이션 시작

```gherkin
Feature: FastAPI 애플리케이션 시작
  As a 개발자
  I want FastAPI 애플리케이션이 정상 시작되길
  So that API 서비스를 제공할 수 있다

  Scenario: 개발 서버 시작 성공
    Given requirements.txt의 의존성이 설치되어 있다
    And .env 파일이 존재한다
    When uvicorn app.main:app --reload 명령을 실행한다
    Then 서버가 8000 포트에서 시작된다
    And 로그에 "Uvicorn running on http://0.0.0.0:8000" 메시지가 출력된다

  Scenario: 설정 로드 성공
    Given 환경 변수가 올바르게 설정되어 있다
    When 애플리케이션이 시작된다
    Then Settings 객체가 정상 생성된다
    And DEBUG, HOST, PORT 설정이 로드된다
```

#### TC-003: 기본 헬스체크

```gherkin
Feature: 기본 헬스체크 엔드포인트
  As a 운영자
  I want 서버 상태를 확인할 수 있길
  So that 모니터링과 로드밸런싱에 활용할 수 있다

  Scenario: GET /health 성공 응답
    Given FastAPI 서버가 실행 중이다
    When GET /health 요청을 보낸다
    Then 상태 코드 200을 반환한다
    And 응답 본문에 다음 필드가 포함된다:
      | 필드 | 타입 | 예시 |
      | status | string | "healthy" |
      | version | string | "0.1.0" |
      | timestamp | datetime | "2025-01-17T12:00:00Z" |

  Scenario: GET /api/v1/health 의존성 상태 포함
    Given FastAPI 서버가 실행 중이다
    And Redis 서버가 실행 중이다
    When GET /api/v1/health 요청을 보낸다
    Then 상태 코드 200을 반환한다
    And 응답 본문에 dependencies 필드가 포함된다
    And dependencies.redis 상태가 "connected"이다
```

---

### 2.3 설정 관리 검증

#### TC-004: Pydantic Settings 검증

```gherkin
Feature: Pydantic Settings 설정 관리
  As a 개발자
  I want 환경 변수 기반 설정 관리
  So that 환경별로 다른 설정을 적용할 수 있다

  Scenario: 기본값 적용
    Given .env 파일이 없다
    When Settings 객체를 생성한다
    Then 기본값이 적용된다:
      | 설정 | 기본값 |
      | HOST | "0.0.0.0" |
      | PORT | 8000 |
      | DEBUG | False |
      | DEVICE | "cuda" |

  Scenario: 환경 변수 오버라이드
    Given .env 파일에 DEBUG=true가 설정되어 있다
    When Settings 객체를 생성한다
    Then settings.DEBUG가 True이다

  Scenario: 필수 설정 누락 시 에러
    Given 필수 환경 변수가 누락되어 있다
    When Settings 객체를 생성한다
    Then ValidationError가 발생한다
```

---

### 2.4 Pydantic 모델 검증

#### TC-005: Request 모델 검증

```gherkin
Feature: Request 모델 데이터 검증
  As a API 사용자
  I want 입력 데이터가 검증되길
  So that 잘못된 요청을 사전에 방지할 수 있다

  Scenario: EvaluateRequest 유효한 데이터
    Given 유효한 평가 요청 데이터가 있다:
      | 필드 | 값 |
      | department | "시각디자인" |
      | target_universities | ["서울대", "홍익대"] |
    When EvaluateRequest 모델로 파싱한다
    Then 모델 객체가 정상 생성된다

  Scenario: EvaluateRequest 잘못된 학과
    Given 잘못된 학과 데이터가 있다:
      | 필드 | 값 |
      | department | "invalid_department" |
    When EvaluateRequest 모델로 파싱한다
    Then ValidationError가 발생한다
    And 에러 메시지에 "department" 필드가 포함된다
```

#### TC-006: Response 모델 검증

```gherkin
Feature: Response 모델 직렬화
  As a API 개발자
  I want 응답이 일관된 형식으로 직렬화되길
  So that 클라이언트가 예측 가능한 응답을 받을 수 있다

  Scenario: HealthResponse 직렬화
    Given HealthResponse 객체가 있다:
      | 필드 | 값 |
      | status | "healthy" |
      | version | "0.1.0" |
    When JSON으로 직렬화한다
    Then 유효한 JSON 문자열이 반환된다
    And timestamp 필드가 ISO 8601 형식이다

  Scenario: ErrorResponse 직렬화 (DEBUG 모드)
    Given DEBUG 모드가 활성화되어 있다
    And ErrorResponse 객체가 있다
    When JSON으로 직렬화한다
    Then detail 필드가 포함된다

  Scenario: ErrorResponse 직렬화 (프로덕션 모드)
    Given DEBUG 모드가 비활성화되어 있다
    And ErrorResponse 객체가 있다
    When JSON으로 직렬화한다
    Then detail 필드가 제외된다
```

---

### 2.5 라우터 Placeholder 검증

#### TC-007: Placeholder 엔드포인트 동작

```gherkin
Feature: Placeholder 엔드포인트
  As a 개발자
  I want placeholder 엔드포인트가 501을 반환하길
  So that 미구현 기능을 명확히 식별할 수 있다

  Scenario: POST /api/v1/evaluate placeholder
    Given FastAPI 서버가 실행 중이다
    When POST /api/v1/evaluate 요청을 보낸다
    Then 상태 코드 501 Not Implemented를 반환한다
    And 응답에 "Not implemented yet" 메시지가 포함된다

  Scenario: POST /api/v1/compare placeholder
    Given FastAPI 서버가 실행 중이다
    When POST /api/v1/compare 요청을 보낸다
    Then 상태 코드 501 Not Implemented를 반환한다

  Scenario: GET /api/v1/competitions placeholder
    Given FastAPI 서버가 실행 중이다
    When GET /api/v1/competitions 요청을 보낸다
    Then 상태 코드 501 Not Implemented를 반환한다
```

---

### 2.6 Docker 환경 검증

#### TC-008: Docker 빌드 및 실행

```gherkin
Feature: Docker 컨테이너 실행
  As a DevOps 엔지니어
  I want Docker로 애플리케이션을 실행할 수 있길
  So that 일관된 배포 환경을 제공할 수 있다

  Scenario: Docker 이미지 빌드 성공
    Given Dockerfile이 존재한다
    And requirements.txt가 존재한다
    When docker build -t mirip-backend . 명령을 실행한다
    Then 이미지 빌드가 성공한다
    And 이미지 크기가 5GB 미만이다

  Scenario: docker-compose 실행 성공
    Given docker-compose.yml이 존재한다
    When docker-compose up -d 명령을 실행한다
    Then api 컨테이너가 실행된다
    And redis 컨테이너가 실행된다
    And api 컨테이너의 헬스체크가 healthy이다

  Scenario: 컨테이너 헬스체크
    Given docker-compose로 컨테이너가 실행 중이다
    When curl http://localhost:8000/health 요청을 보낸다
    Then 상태 코드 200을 반환한다
```

---

### 2.7 테스트 검증

#### TC-009: pytest 실행

```gherkin
Feature: 테스트 실행
  As a 개발자
  I want 자동화된 테스트를 실행할 수 있길
  So that 코드 품질을 보장할 수 있다

  Scenario: pytest 전체 테스트 성공
    Given 테스트 파일이 tests/ 디렉토리에 있다
    When pytest 명령을 실행한다
    Then 모든 테스트가 통과한다
    And 테스트 커버리지가 85% 이상이다

  Scenario: 헬스체크 테스트
    Given test_health.py가 존재한다
    When pytest tests/test_health.py 명령을 실행한다
    Then test_health_check 테스트가 통과한다
    And test_api_health_check 테스트가 통과한다
```

---

## 3. Non-Functional Requirements Verification

### 3.1 성능 기준

| 항목 | 기준 | 측정 방법 |
|------|------|-----------|
| 서버 시작 시간 | < 5초 | 시작 로그 타임스탬프 |
| 헬스체크 응답 시간 | < 100ms | curl -w "%{time_total}" |
| Docker 빌드 시간 | < 10분 | docker build 시간 |

### 3.2 보안 기준

| 항목 | 기준 | 검증 방법 |
|------|------|-----------|
| 시크릿 하드코딩 | 없음 | 코드 리뷰, grep 검사 |
| 스택 트레이스 노출 | 프로덕션에서 없음 | DEBUG=false로 테스트 |
| CORS 설정 | 화이트리스트 방식 | 설정 파일 확인 |

### 3.3 코드 품질 기준

| 항목 | 기준 | 도구 |
|------|------|------|
| Lint | 0 errors, 0 warnings | ruff |
| Type Check | 0 errors | mypy |
| Complexity | < 10 (cyclomatic) | radon |

---

## 4. Verification Methods

### 4.1 자동화 테스트

```bash
# 전체 테스트 실행
cd backend
pytest --cov=app --cov-report=html

# 린트 검사
ruff check app/

# 타입 검사
mypy app/
```

### 4.2 수동 검증

```bash
# 서버 시작
uvicorn app.main:app --reload

# 헬스체크 확인
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/health

# Swagger UI 확인
# 브라우저에서 http://localhost:8000/docs 접속
```

### 4.3 Docker 검증

```bash
# 빌드 및 실행
docker-compose up --build -d

# 로그 확인
docker-compose logs -f api

# 헬스체크 확인
docker-compose exec api curl http://localhost:8000/health

# 정리
docker-compose down -v
```

---

## 5. Sign-off Checklist

### 5.1 기능 완료

- [ ] 모든 필수 파일 생성됨
- [ ] FastAPI 애플리케이션 정상 시작
- [ ] GET /health 200 OK 반환
- [ ] GET /api/v1/health 200 OK 반환
- [ ] Swagger UI 접근 가능
- [ ] 모든 placeholder 엔드포인트 501 반환

### 5.2 품질 완료

- [ ] pytest 모든 테스트 통과
- [ ] 테스트 커버리지 85% 이상
- [ ] ruff lint 경고 없음
- [ ] mypy 타입 에러 없음

### 5.3 배포 준비

- [ ] Docker 이미지 빌드 성공
- [ ] docker-compose 실행 성공
- [ ] 컨테이너 헬스체크 통과
- [ ] .env.example 문서화 완료
- [ ] README.md 작성 완료

---

*Acceptance Criteria 버전: 1.0.0*
*작성일: 2025-01-17*
*연관 SPEC: SPEC-BACKEND-001*
