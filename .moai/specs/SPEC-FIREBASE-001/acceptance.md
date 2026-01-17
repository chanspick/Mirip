# SPEC-FIREBASE-001: 인수 기준

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-FIREBASE-001 |
| **제목** | Firebase 연동 및 사전등록 페이지 구현 |
| **관련 SPEC** | spec.md, plan.md |

---

## 1. 인수 기준 (Acceptance Criteria)

### AC1: Firebase 초기화

**Given** 애플리케이션이 시작될 때
**When** Firebase 초기화 코드가 실행되면
**Then** Firebase가 milip-prototype 프로젝트에 성공적으로 연결되어야 한다
**And** Firestore 인스턴스가 사용 가능해야 한다
**And** 콘솔에 초기화 에러가 없어야 한다

**검증 방법:**
- [ ] 개발자 도구 콘솔에서 Firebase 초기화 확인
- [ ] Firestore 연결 상태 확인
- [ ] 네트워크 탭에서 Firebase 요청 확인

---

### AC2: 폼 필드 검증

**Given** 사용자가 사전등록 폼을 작성할 때

**Scenario 2.1: 이름 필드 검증**
**When** 이름 필드가 비어있는 상태로 제출하면
**Then** "이름을 입력해주세요" 에러 메시지가 표시되어야 한다

**Scenario 2.2: 이메일 형식 검증**
**When** 잘못된 이메일 형식(예: "test@")을 입력하면
**Then** "올바른 이메일을 입력해주세요" 에러 메시지가 표시되어야 한다

**Scenario 2.3: 유저유형 검증**
**When** 유저 유형을 선택하지 않고 제출하면
**Then** "유저 유형을 선택해주세요" 에러 메시지가 표시되어야 한다

**Scenario 2.4: 모든 필드 유효**
**When** 모든 필드가 올바르게 입력되면
**Then** 제출 버튼이 활성화되어야 한다
**And** 에러 메시지가 표시되지 않아야 한다

**검증 방법:**
- [ ] 각 필드별 빈 값 제출 테스트
- [ ] 잘못된 이메일 형식 테스트 (test, test@, @test.com)
- [ ] 올바른 이메일 형식 테스트 (test@example.com)
- [ ] 유저유형 미선택 테스트

---

### AC3: Firestore 데이터 저장

**Given** 사용자가 유효한 정보로 폼을 작성했을 때
**When** 제출 버튼을 클릭하면
**Then** 등록 데이터가 Firestore 'registrations' 컬렉션에 저장되어야 한다

**저장되어야 할 데이터:**
| 필드 | 타입 | 예시 값 |
|------|------|---------|
| name | string | "홍길동" |
| email | string | "hong@example.com" |
| userType | string | "student" |
| timestamp | Timestamp | 서버 시간 |
| registrationId | string | 자동 생성된 문서 ID |

**검증 방법:**
- [ ] Firebase Console에서 'registrations' 컬렉션 확인
- [ ] 저장된 문서의 필드 검증
- [ ] timestamp가 서버 시간으로 기록되었는지 확인
- [ ] registrationId가 고유한지 확인

---

### AC4: 성공 모달 표시

**Given** 등록 데이터가 Firestore에 성공적으로 저장되었을 때
**When** 저장이 완료되면
**Then** 성공 모달이 화면 중앙에 표시되어야 한다
**And** 모달에 "등록이 완료되었습니다" 메시지가 표시되어야 한다
**And** 모달 닫기 버튼이 작동해야 한다
**And** 모달 외부 클릭 시 모달이 닫혀야 한다

**검증 방법:**
- [ ] 모달 표시 확인
- [ ] 모달 메시지 확인
- [ ] 닫기 버튼 클릭 테스트
- [ ] 오버레이 클릭 테스트
- [ ] ESC 키 테스트

---

### AC5: 에러 처리 및 사용자 피드백

**Given** 네트워크 오류가 발생했을 때
**When** 폼 제출이 실패하면
**Then** 사용자 친화적인 에러 메시지가 표시되어야 한다
**And** 기술적 세부사항은 노출되지 않아야 한다
**And** 재시도 옵션이 제공되어야 한다

**에러 메시지 예시:**
- "네트워크 연결을 확인해주세요. 다시 시도해주세요."
- "등록 처리 중 문제가 발생했습니다. 잠시 후 다시 시도해주세요."

**검증 방법:**
- [ ] 오프라인 상태에서 제출 테스트
- [ ] Firebase 서비스 중단 시뮬레이션
- [ ] 에러 메시지 사용자 친화성 확인
- [ ] 콘솔에 스택 트레이스가 노출되지 않는지 확인

---

### AC6: 테스트 커버리지

**Given** 모든 컴포넌트와 서비스가 구현되었을 때
**When** 테스트 스위트를 실행하면
**Then** 전체 테스트 커버리지가 85% 이상이어야 한다

**커버리지 세부 목표:**
| 영역 | 목표 커버리지 |
|------|--------------|
| registrationService.js | 95% |
| RegistrationForm.js | 90% |
| Landing.js | 80% |
| firebase.js | 85% |

**검증 방법:**
- [ ] `npm test -- --coverage` 실행
- [ ] 커버리지 리포트 확인
- [ ] 미커버 라인 분석 및 테스트 추가

---

## 2. 테스트 시나리오

### 2.1 정상 플로우 테스트

#### TC-001: 사전등록 성공 플로우

```gherkin
Feature: 사전등록 성공
  As a 미술 입시생
  I want to MIRIP 서비스에 사전등록
  So that 서비스 출시 시 알림을 받을 수 있다

  Scenario: 유효한 정보로 사전등록
    Given 사용자가 랜딩 페이지에 접속한다
    And 사전등록 폼이 표시된다
    When 이름에 "홍길동"을 입력한다
    And 이메일에 "hong@example.com"을 입력한다
    And 유저유형에서 "입시생"을 선택한다
    And "사전등록 신청" 버튼을 클릭한다
    Then 로딩 상태가 표시된다
    And 성공 모달이 표시된다
    And 모달에 "등록이 완료되었습니다" 메시지가 표시된다
    And Firestore에 데이터가 저장된다
```

### 2.2 검증 실패 테스트

#### TC-002: 빈 폼 제출

```gherkin
Feature: 폼 검증
  Scenario: 모든 필드가 비어있는 상태로 제출
    Given 사용자가 사전등록 폼을 보고 있다
    When 아무것도 입력하지 않고 제출 버튼을 클릭한다
    Then 이름 필드에 "이름을 입력해주세요" 에러가 표시된다
    And 이메일 필드에 "이메일을 입력해주세요" 에러가 표시된다
    And 유저유형 필드에 "유저 유형을 선택해주세요" 에러가 표시된다
    And 폼이 제출되지 않는다
```

#### TC-003: 잘못된 이메일 형식

```gherkin
Feature: 이메일 검증
  Scenario Outline: 다양한 잘못된 이메일 형식
    Given 사용자가 사전등록 폼을 작성 중이다
    When 이메일에 "<invalid_email>"을 입력한다
    And 다른 필드로 포커스를 이동한다
    Then "올바른 이메일을 입력해주세요" 에러가 표시된다

    Examples:
      | invalid_email |
      | test          |
      | test@         |
      | @example.com  |
      | test@.com     |
      | test@com      |
```

### 2.3 에러 처리 테스트

#### TC-004: 네트워크 에러 처리

```gherkin
Feature: 네트워크 에러 처리
  Scenario: 오프라인 상태에서 제출
    Given 사용자가 유효한 정보를 입력했다
    And 네트워크가 오프라인 상태이다
    When 제출 버튼을 클릭한다
    Then 에러 메시지가 표시된다
    And 메시지에 "네트워크"라는 단어가 포함된다
    And 재시도 버튼이 표시된다
```

#### TC-005: 중복 제출 방지

```gherkin
Feature: 중복 제출 방지
  Scenario: 같은 세션에서 두 번 제출
    Given 사용자가 이미 등록을 완료했다
    When 같은 정보로 다시 제출하려고 한다
    Then 이미 등록되었다는 메시지가 표시된다
    And 폼이 비활성화된다
```

### 2.4 반응형 테스트

#### TC-006: 모바일 반응형

```gherkin
Feature: 반응형 디자인
  Scenario Outline: 다양한 화면 크기에서 레이아웃
    Given 사용자가 <device> 화면으로 접속한다
    When 랜딩 페이지가 로드되면
    Then 모든 섹션이 올바르게 표시된다
    And 폼이 사용 가능하다
    And 버튼이 클릭 가능하다

    Examples:
      | device          | width  |
      | Desktop         | 1440px |
      | Tablet          | 768px  |
      | Mobile Portrait | 375px  |
```

---

## 3. 품질 게이트

### 3.1 코드 품질

| 기준 | 목표 | 도구 |
|------|------|------|
| 린트 에러 | 0 | ESLint |
| 타입 에러 | 0 | PropTypes |
| 미사용 변수 | 0 | ESLint |
| console.log | 0 (프로덕션) | ESLint |

### 3.2 성능 기준

| 지표 | 목표 | 측정 도구 |
|------|------|----------|
| 초기 로드 시간 | < 3초 | Lighthouse |
| Firestore 쓰기 시간 | < 1초 | Network Tab |
| First Contentful Paint | < 1.5초 | Lighthouse |
| Cumulative Layout Shift | < 0.1 | Lighthouse |

### 3.3 접근성 기준

| 기준 | 목표 | 검증 방법 |
|------|------|----------|
| 키보드 네비게이션 | 모든 폼 요소 접근 가능 | 수동 테스트 |
| 폼 레이블 | 모든 입력에 label 연결 | axe 검사 |
| 색상 대비 | WCAG AA 준수 | Lighthouse |
| 에러 메시지 | 스크린 리더 호환 | NVDA 테스트 |

---

## 4. Definition of Done

### 4.1 코드 완료 조건

- [ ] 모든 요구사항(REQ-*)이 구현됨
- [ ] 모든 인수 기준(AC1-AC6)이 통과됨
- [ ] 단위 테스트 커버리지 85% 이상
- [ ] ESLint 에러 0개
- [ ] PropTypes 또는 TypeScript 타입 정의 완료
- [ ] CSS 변수만 사용 (하드코딩된 색상/크기 없음)

### 4.2 문서 완료 조건

- [ ] 코드 주석 작성 (복잡한 로직)
- [ ] README.md 업데이트 (환경 변수 설정 방법)
- [ ] .env.example 파일 생성

### 4.3 배포 완료 조건

- [ ] Firebase Hosting 배포 성공
- [ ] Firestore 보안 규칙 적용
- [ ] 환경 변수 설정 완료
- [ ] 프로덕션 빌드 에러 없음

### 4.4 검증 완료 조건

- [ ] 크로스 브라우저 테스트 (Chrome, Safari, Firefox)
- [ ] 반응형 테스트 (Desktop, Tablet, Mobile)
- [ ] 실제 Firestore에 데이터 저장 확인
- [ ] 성공/에러 모달 동작 확인

---

## 5. 체크리스트

### 5.1 구현 체크리스트

**Firebase 설정:**
- [ ] firebase.js 생성
- [ ] .env.local 설정
- [ ] Firestore 연결 확인

**서비스 레이어:**
- [ ] registrationService.js 생성
- [ ] create() 함수 구현
- [ ] 에러 처리 구현

**컴포넌트:**
- [ ] RegistrationForm 컴포넌트 생성
- [ ] 폼 검증 로직 구현
- [ ] 로딩 상태 구현
- [ ] 성공/에러 모달 연동

**랜딩 페이지:**
- [ ] Landing 페이지 컴포넌트 생성
- [ ] Hero 섹션 구현
- [ ] Problem/Solution 섹션 구현
- [ ] AI Preview 섹션 구현
- [ ] CTA 섹션 연동
- [ ] Header/Footer 연동

### 5.2 테스트 체크리스트

- [ ] registrationService.test.js
- [ ] RegistrationForm.test.js
- [ ] Landing.test.js
- [ ] 통합 테스트 (E2E)

### 5.3 배포 체크리스트

- [ ] firebase.json 설정
- [ ] firestore.rules 설정
- [ ] `npm run build` 성공
- [ ] `firebase deploy` 성공

---

*인수 기준 버전: 1.0*
*최종 업데이트: 2025-01-17*
