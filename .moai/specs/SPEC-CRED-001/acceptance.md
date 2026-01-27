# SPEC-CRED-001 인수 조건

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-CRED-001 |
| **문서 유형** | Acceptance Criteria |
| **생성일** | 2025-01-19 |
| **관련 SPEC** | SPEC-CRED-001/spec.md, SPEC-CRED-001/plan.md |

---

## 1. 인수 조건 개요

### 1.1 테스트 범위

| 영역 | 설명 | 커버리지 목표 |
|------|------|---------------|
| 데이터 모델 | Firestore 서비스 레이어 | 85% |
| 마이페이지 | 잔디밭, 히스토리, 프로필 편집 | 80% |
| 공개 프로필 | 프로필 조회, 포트폴리오 표시 | 80% |
| 포트폴리오 | CRUD, 이미지 업로드 | 85% |
| 시스템 연동 | 진단/공모전 → 활동 기록 | 80% |

### 1.2 테스트 유형

- **단위 테스트**: 서비스 함수, 유틸리티 함수
- **통합 테스트**: Firestore 연동, 컴포넌트 렌더링
- **E2E 테스트**: 사용자 시나리오 기반 플로우

---

## 2. 시나리오별 인수 조건 (Given-When-Then)

### 2.1 마이페이지 시나리오

#### AC-CRED-001: 마이페이지 접근 및 표시

```gherkin
Feature: 마이페이지 접근 및 표시
  사용자가 마이페이지에서 자신의 활동 현황을 확인한다

  Background:
    Given 사용자 "artista_kim"이 로그인 상태이다
    And 사용자에게 다음 활동 기록이 있다:
      | 날짜       | 타입              | 제목              |
      | 2025-01-19 | diagnosis         | AI 진단 완료      |
      | 2025-01-18 | competition_submit| 미립콤프 #5 출품  |
      | 2025-01-15 | portfolio_add     | 작품 추가         |

  Scenario: 마이페이지 정상 로드
    When 사용자가 "/profile" 페이지에 접근한다
    Then 프로필 카드에 "artista_kim"의 정보가 표시된다
    And 잔디밭(ActivityHeatmap)이 표시된다
    And 활동 히스토리에 최근 활동 3건이 표시된다
    And 통계 요약에 "총 활동 3회"가 표시된다

  Scenario: 비로그인 사용자 마이페이지 접근 차단
    Given 사용자가 로그아웃 상태이다
    When 사용자가 "/profile" 페이지에 접근한다
    Then 로그인 페이지로 리다이렉트된다
```

#### AC-CRED-002: 잔디밭(ActivityHeatmap) 표시

```gherkin
Feature: 잔디밭 활동 히트맵 표시
  사용자의 연간 활동을 GitHub 스타일 히트맵으로 표시한다

  Background:
    Given 사용자 "artista_kim"이 로그인 상태이다
    And 2025년도에 다음 일별 활동이 있다:
      | 날짜       | 활동 수 |
      | 2025-01-01 | 0       |
      | 2025-01-02 | 2       |
      | 2025-01-03 | 5       |
      | 2025-01-04 | 8       |

  Scenario: 잔디밭 정상 렌더링
    When 마이페이지가 로드된다
    Then 52주 x 7일 형태의 히트맵이 표시된다
    And 2025-01-01 셀은 레벨 0 (빈칸) 색상이다
    And 2025-01-02 셀은 레벨 1 (연한색) 색상이다
    And 2025-01-03 셀은 레벨 2 (중간색) 색상이다
    And 2025-01-04 셀은 레벨 4 (진한색) 색상이다

  Scenario: 잔디밭 셀 호버 시 툴팁
    When 사용자가 2025-01-03 셀에 마우스를 호버한다
    Then 툴팁에 "2025년 1월 3일: 5개의 활동"이 표시된다
```

#### AC-CRED-003: 활동 히스토리 타임라인

```gherkin
Feature: 활동 히스토리 타임라인
  사용자의 활동 기록을 시간순으로 표시한다

  Background:
    Given 사용자 "artista_kim"이 로그인 상태이다
    And 50개의 활동 기록이 있다

  Scenario: 활동 히스토리 초기 로드
    When 마이페이지가 로드된다
    Then 최신 20개의 활동이 표시된다
    And 각 활동에 타입별 아이콘이 표시된다
    And 활동이 날짜별로 그룹핑된다 (오늘, 어제, 이번 주)

  Scenario: 활동 히스토리 더보기
    Given 마이페이지가 로드된 상태이다
    When 사용자가 "더보기" 버튼을 클릭한다
    Then 추가 20개의 활동이 로드된다
    And 총 40개의 활동이 표시된다
```

---

### 2.2 공개 프로필 시나리오

#### AC-CRED-004: 공개 프로필 조회

```gherkin
Feature: 공개 프로필 조회
  다른 사용자의 공개 프로필을 조회한다

  Background:
    Given 사용자 "artista_kim"이 다음 프로필을 가진다:
      | 필드        | 값                    |
      | displayName | 김아티스트            |
      | bio         | 시각디자인 전공       |
      | tier        | A                     |
      | isPublic    | true                  |
    And 사용자에게 3개의 공개 포트폴리오 작품이 있다
    And 사용자에게 2개의 수상 이력이 있다

  Scenario: 공개 프로필 정상 조회
    When 방문자가 "/profile/artista_kim" 페이지에 접근한다
    Then 프로필 카드에 "김아티스트"가 표시된다
    And 소개에 "시각디자인 전공"이 표시된다
    And A티어 배지가 표시된다
    And 포트폴리오 그리드에 3개의 작품이 표시된다
    And 수상 이력 목록에 2개의 수상이 표시된다

  Scenario: 비공개 프로필 접근 차단
    Given 사용자 "private_user"의 isPublic이 false이다
    When 방문자가 "/profile/private_user" 페이지에 접근한다
    Then "비공개 프로필입니다" 메시지가 표시된다
    And 프로필 정보가 노출되지 않는다

  Scenario: 존재하지 않는 사용자 프로필
    When 방문자가 "/profile/nonexistent_user" 페이지에 접근한다
    Then 404 페이지가 표시된다
    And "사용자를 찾을 수 없습니다" 메시지가 표시된다
```

#### AC-CRED-005: 비공개 포트폴리오 보호

```gherkin
Feature: 비공개 포트폴리오 보호
  비공개 설정된 포트폴리오는 공개 프로필에 노출되지 않는다

  Background:
    Given 사용자 "artista_kim"이 다음 포트폴리오를 가진다:
      | 제목    | isPublic |
      | 작품 A  | true     |
      | 작품 B  | false    |
      | 작품 C  | true     |

  Scenario: 공개 프로필에서 공개 작품만 표시
    When 방문자가 "/profile/artista_kim" 페이지에 접근한다
    Then 포트폴리오 그리드에 2개의 작품만 표시된다
    And "작품 A"와 "작품 C"가 표시된다
    And "작품 B"는 표시되지 않는다

  Scenario: 소유자는 모든 작품 확인 가능
    Given 사용자 "artista_kim"이 로그인 상태이다
    When 사용자가 "/portfolio" 페이지에 접근한다
    Then 포트폴리오 그리드에 3개의 작품이 모두 표시된다
    And "작품 B"에 "비공개" 라벨이 표시된다
```

---

### 2.3 포트폴리오 관리 시나리오

#### AC-CRED-006: 포트폴리오 작품 추가

```gherkin
Feature: 포트폴리오 작품 추가
  사용자가 포트폴리오에 새 작품을 추가한다

  Background:
    Given 사용자 "artista_kim"이 로그인 상태이다
    And 사용자가 "/portfolio" 페이지에 있다

  Scenario: 작품 정상 추가
    When 사용자가 "작품 추가" 버튼을 클릭한다
    And 다음 정보를 입력한다:
      | 필드       | 값                  |
      | 제목       | 나의 첫 작품        |
      | 설명       | 시각디자인 과제물   |
      | 카테고리   | visual_design       |
      | 이미지     | test_image.jpg      |
    And "저장" 버튼을 클릭한다
    Then 작품이 포트폴리오 그리드에 추가된다
    And 성공 토스트 메시지가 표시된다
    And 활동 기록(type: portfolio_add)이 생성된다

  Scenario: 이미지 크기 초과 시 에러
    When 사용자가 15MB 크기의 이미지를 업로드 시도한다
    Then "이미지는 10MB 이하여야 합니다" 에러가 표시된다
    And 작품이 저장되지 않는다

  Scenario: 지원하지 않는 이미지 형식
    When 사용자가 "test.gif" 파일을 업로드 시도한다
    Then "JPG, PNG, WebP 형식만 지원합니다" 에러가 표시된다
```

#### AC-CRED-007: 포트폴리오 작품 편집 및 삭제

```gherkin
Feature: 포트폴리오 작품 편집 및 삭제
  사용자가 기존 포트폴리오 작품을 편집하거나 삭제한다

  Background:
    Given 사용자 "artista_kim"이 로그인 상태이다
    And 다음 포트폴리오 작품이 있다:
      | id     | 제목    | isPublic |
      | port-1 | 작품 A  | true     |

  Scenario: 작품 제목 편집
    When 사용자가 "작품 A"의 편집 버튼을 클릭한다
    And 제목을 "작품 A (수정됨)"으로 변경한다
    And "저장" 버튼을 클릭한다
    Then 작품 제목이 "작품 A (수정됨)"으로 변경된다
    And 성공 토스트 메시지가 표시된다

  Scenario: 작품 공개/비공개 토글
    When 사용자가 "작품 A"의 공개 토글을 클릭한다
    Then 작품의 isPublic이 false로 변경된다
    And "비공개로 변경되었습니다" 토스트가 표시된다

  Scenario: 작품 삭제
    When 사용자가 "작품 A"의 삭제 버튼을 클릭한다
    And 확인 다이얼로그에서 "삭제"를 클릭한다
    Then 작품이 포트폴리오에서 제거된다
    And Storage에서 이미지가 삭제된다
```

---

### 2.4 시스템 연동 시나리오

#### AC-CRED-008: AI 진단 완료 시 활동 기록 생성

```gherkin
Feature: AI 진단 완료 시 활동 기록 자동 생성
  AI 진단이 완료되면 활동 기록이 자동으로 생성된다

  Background:
    Given 사용자 "artista_kim"이 로그인 상태이다
    And 사용자의 현재 최고 티어는 "B"이다

  Scenario: 진단 완료 시 활동 기록 생성
    When 사용자가 AI 진단을 완료하고 결과가 "A" 티어이다
    Then 다음 활동 기록이 생성된다:
      | 필드   | 값                           |
      | type   | diagnosis                    |
      | title  | AI 진단 완료                 |
      | tier   | A                            |
    And 사용자의 최고 티어가 "A"로 업데이트된다
    And 잔디밭의 오늘 날짜 활동 수가 1 증가한다

  Scenario: 낮은 티어 진단 시 최고 티어 유지
    When 사용자가 AI 진단을 완료하고 결과가 "C" 티어이다
    Then 활동 기록이 생성된다
    And 사용자의 최고 티어는 "B"로 유지된다
```

#### AC-CRED-009: 공모전 출품/수상 시 활동 기록 생성

```gherkin
Feature: 공모전 출품/수상 시 활동 기록 자동 생성
  공모전 출품 및 수상 시 활동 기록이 자동으로 생성된다

  Background:
    Given 사용자 "artista_kim"이 로그인 상태이다
    And 공모전 "미립콤프 #5"가 존재한다

  Scenario: 공모전 출품 시 활동 기록 생성
    When 사용자가 "미립콤프 #5"에 작품을 출품한다
    Then 다음 활동 기록이 생성된다:
      | 필드             | 값                    |
      | type             | competition_submit    |
      | title            | 공모전 출품           |
      | competitionTitle | 미립콤프 #5          |
    And 잔디밭의 오늘 날짜 활동 수가 1 증가한다

  Scenario: 공모전 수상 시 활동 기록 및 수상 이력 생성
    When 사용자가 "미립콤프 #5"에서 "금상" (2등)을 수상한다
    Then 다음 활동 기록이 생성된다:
      | 필드             | 값                    |
      | type             | competition_award     |
      | title            | 공모전 수상           |
      | awardRank        | 2                     |
    And 다음 수상 이력이 추가된다:
      | 필드             | 값                    |
      | competitionTitle | 미립콤프 #5          |
      | awardName        | 금상                  |
      | rank             | 2                     |
    And 사용자의 awardCount가 1 증가한다
```

---

### 2.5 프로필 편집 시나리오

#### AC-CRED-010: 프로필 정보 편집

```gherkin
Feature: 프로필 정보 편집
  사용자가 자신의 프로필 정보를 편집한다

  Background:
    Given 사용자 "artista_kim"이 로그인 상태이다
    And 사용자가 마이페이지에 있다

  Scenario: 프로필 정보 수정
    When 사용자가 프로필 편집 버튼을 클릭한다
    And 다음 정보를 수정한다:
      | 필드        | 값                    |
      | displayName | 김아티스트 (수정)     |
      | bio         | 홍익대학교 시디과     |
    And "저장" 버튼을 클릭한다
    Then 프로필 정보가 업데이트된다
    And 활동 기록(type: profile_update)이 생성된다
    And 성공 토스트 메시지가 표시된다

  Scenario: 프로필 이미지 변경
    When 사용자가 프로필 이미지 변경 버튼을 클릭한다
    And 새 이미지를 선택한다
    Then 이미지가 Storage에 업로드된다
    And 프로필 이미지가 변경된다
    And 이전 이미지는 삭제된다

  Scenario: 사용자명 변경 (중복 검사)
    When 사용자가 username을 "new_username"으로 변경 시도한다
    And "new_username"이 사용 가능한 경우
    Then username이 "new_username"으로 변경된다
    And 공개 프로필 URL이 "/profile/new_username"으로 변경된다

  Scenario: 중복된 사용자명 변경 시도
    Given "existing_user" username이 이미 존재한다
    When 사용자가 username을 "existing_user"로 변경 시도한다
    Then "이미 사용 중인 사용자명입니다" 에러가 표시된다
    And username이 변경되지 않는다
```

---

## 3. 비기능 요구사항 테스트

### 3.1 성능 테스트

```gherkin
Feature: 성능 요구사항 검증

  Scenario: 마이페이지 초기 로드 시간
    Given 사용자에게 1년치 활동 데이터가 있다
    When 마이페이지를 로드한다
    Then 페이지 로드 시간이 2초 이내이다

  Scenario: 잔디밭 렌더링 시간
    Given 365일치 활동 데이터가 있다
    When 잔디밭 컴포넌트가 렌더링된다
    Then 렌더링 시간이 500ms 이내이다

  Scenario: 프로필 이미지 업로드 시간
    Given 5MB 크기의 이미지 파일이 있다
    When 이미지를 업로드한다
    Then 업로드 완료 시간이 3초 이내이다
```

### 3.2 보안 테스트

```gherkin
Feature: 보안 요구사항 검증

  Scenario: 다른 사용자 프로필 편집 차단
    Given 사용자 A가 로그인 상태이다
    When 사용자 A가 사용자 B의 프로필을 수정 시도한다
    Then 403 Forbidden 에러가 반환된다
    And 데이터가 변경되지 않는다

  Scenario: 비공개 포트폴리오 직접 URL 접근 차단
    Given 사용자 A의 비공개 포트폴리오 ID가 "port-123"이다
    When 사용자 B가 해당 포트폴리오 직접 URL에 접근한다
    Then 데이터가 반환되지 않는다

  Scenario: 인증되지 않은 활동 기록 생성 차단
    Given 사용자가 로그아웃 상태이다
    When 활동 기록 생성 API를 직접 호출한다
    Then 401 Unauthorized 에러가 반환된다
```

---

## 4. Definition of Done (완료 정의)

### 4.1 기능 완료 조건

- [ ] 모든 인수 조건(AC-CRED-001 ~ AC-CRED-010) 통과
- [ ] 단위 테스트 커버리지 80% 이상
- [ ] E2E 테스트 주요 시나리오 통과
- [ ] 크로스 브라우저 테스트 (Chrome, Firefox, Safari)
- [ ] 모바일 반응형 UI 검증

### 4.2 품질 완료 조건

- [ ] ESLint 경고 0개
- [ ] Firestore 보안 규칙 테스트 통과
- [ ] 성능 요구사항 충족 (LCP < 2s)
- [ ] 접근성 검사 (WCAG 2.1 AA 준수)

### 4.3 문서 완료 조건

- [ ] 컴포넌트 JSDoc 주석 완성
- [ ] Firestore 스키마 문서화
- [ ] API 서비스 함수 문서화
- [ ] README 업데이트 (사용법 가이드)

---

## 5. 테스트 데이터

### 5.1 시드 데이터

```javascript
// 테스트용 사용자 데이터
const testUsers = [
  {
    uid: 'test-user-1',
    username: 'artista_kim',
    displayName: '김아티스트',
    bio: '시각디자인 전공',
    tier: 'A',
    isPublic: true,
    totalActivities: 50,
    awardCount: 3,
  },
  {
    uid: 'test-user-2',
    username: 'private_user',
    displayName: '비공개 사용자',
    isPublic: false,
  },
];

// 테스트용 활동 데이터
const testActivities = [
  {
    userId: 'test-user-1',
    type: 'diagnosis',
    title: 'AI 진단 완료',
    date: '2025-01-19',
    metadata: { tier: 'A' },
  },
  // ...
];
```

---

*문서 버전: 1.0*
*최종 업데이트: 2025-01-19*
*작성: manager-spec*
