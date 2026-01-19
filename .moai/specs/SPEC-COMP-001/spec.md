# SPEC-COMP-001: 공모전 MVP

> Phase 1 Week 3-4 - 공모전 플랫폼 핵심 기능

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-COMP-001 |
| **제목** | 공모전 MVP |
| **상태** | Completed |
| **우선순위** | High |
| **Phase** | Phase 1 Week 3-4 |

---

## 1. 개요

### 목적

MIRIP 플랫폼의 핵심 기능인 공모전 시스템을 구현합니다. 사용자가 공모전을 탐색하고, 상세 정보를 확인하며, 작품을 출품할 수 있는 MVP를 개발합니다.

### 범위

- 공모전 목록 페이지 (카드 그리드, 필터, 정렬, 페이지네이션)
- 공모전 상세 페이지 (헤더, 탭 메뉴, 출품 버튼)
- 출품 페이지 (이미지 업로드, 작품 정보 폼)
- Firestore CRUD (competitions, submissions 컬렉션)
- 초기 데이터 시딩

---

## 2. EARS 요구사항

### 2.1 공모전 목록 페이지

#### REQ-C-001: 공모전 목록 조회

**When** 사용자가 `/competitions` 페이지에 접근하면,
**the system shall** Firestore에서 공모전 목록을 조회하여 카드 그리드 형태로 표시한다.

**Acceptance Criteria:**
- [ ] 공모전 카드에 썸네일, 제목, D-Day, 상금, 분야 표시
- [ ] 한 페이지에 12개 카드 표시 (4열 x 3행)
- [ ] 로딩 중 스켈레톤 UI 표시

#### REQ-C-002: 공모전 필터

**When** 사용자가 필터를 선택하면,
**the system shall** 선택된 조건에 맞는 공모전만 표시한다.

**Filter Options:**
- 분야: 전체, 시각디자인, 산업디자인, 공예, 회화
- 상태: 전체, 진행중, 마감임박, 종료
- 주최자: 검색 가능

**Acceptance Criteria:**
- [ ] 필터 변경 시 즉시 목록 갱신
- [ ] 복수 필터 조합 가능
- [ ] URL 쿼리 파라미터로 필터 상태 유지

#### REQ-C-003: 공모전 정렬

**When** 사용자가 정렬 옵션을 선택하면,
**the system shall** 선택된 기준으로 목록을 정렬한다.

**Sort Options:**
- 마감순 (D-Day 오름차순)
- 상금순 (내림차순)
- 인기순 (참가자 수 내림차순)
- 최신순 (등록일 내림차순)

**Acceptance Criteria:**
- [ ] 기본 정렬: 마감순
- [ ] 정렬 변경 시 즉시 목록 갱신

#### REQ-C-004: 페이지네이션

**When** 공모전이 12개를 초과하면,
**the system shall** 무한 스크롤 또는 페이지네이션으로 추가 로드한다.

**Acceptance Criteria:**
- [ ] 스크롤 시 다음 페이지 자동 로드 (무한 스크롤)
- [ ] 전체 개수 및 현재 위치 표시

---

### 2.2 공모전 상세 페이지

#### REQ-C-005: 공모전 상세 정보 표시

**When** 사용자가 `/competitions/:id` 페이지에 접근하면,
**the system shall** 해당 공모전의 상세 정보를 표시한다.

**Display Elements:**
- 헤더: 썸네일, 제목, D-Day 배지, 참가자 수, 상금
- 탭 메뉴: 상세정보, 출품작, 결과
- 출품 버튼

**Acceptance Criteria:**
- [ ] 공모전 ID로 Firestore에서 조회
- [ ] 존재하지 않는 ID 접근 시 404 페이지 표시
- [ ] D-Day 계산 및 배지 표시 (D-30 이내: 마감임박)

#### REQ-C-006: 탭 메뉴 네비게이션

**When** 사용자가 탭을 클릭하면,
**the system shall** 해당 탭의 콘텐츠를 표시한다.

**Tabs:**
- 상세정보: 공모전 설명, 일정, 규칙, 상금 정보
- 출품작: 현재 출품된 작품 갤러리
- 결과: 수상작 목록 (종료된 공모전만)

**Acceptance Criteria:**
- [ ] 탭 클릭 시 컨텐츠 전환
- [ ] URL 해시로 탭 상태 유지 (#info, #submissions, #results)

#### REQ-C-007: 출품 버튼

**When** 로그인한 사용자가 진행 중인 공모전에서 "출품하기" 버튼을 클릭하면,
**the system shall** 출품 페이지로 이동한다.

**Acceptance Criteria:**
- [ ] 비로그인 사용자: 로그인 모달 표시
- [ ] 마감된 공모전: 버튼 비활성화 및 "마감됨" 표시
- [ ] 이미 출품한 사용자: "출품 완료" 상태 표시

---

### 2.3 출품 페이지

#### REQ-C-008: 이미지 업로드

**When** 사용자가 `/competitions/:id/submit` 페이지에서 이미지를 업로드하면,
**the system shall** Firebase Storage에 이미지를 저장하고 미리보기를 표시한다.

**Acceptance Criteria:**
- [ ] 드래그앤드롭 및 파일 선택 지원
- [ ] 지원 형식: JPG, PNG, WebP
- [ ] 최대 파일 크기: 10MB
- [ ] 업로드 진행률 표시
- [ ] 이미지 미리보기 표시

#### REQ-C-009: 작품 정보 입력

**When** 사용자가 작품 정보를 입력하면,
**the system shall** 입력값을 검증하고 저장한다.

**Fields:**
- 작품 제목 (필수, 최대 100자)
- 작품 설명 (선택, 최대 500자)
- 사용 도구/재료 (선택)
- 제작 기간 (선택)

**Acceptance Criteria:**
- [ ] 필수 필드 미입력 시 제출 불가
- [ ] 실시간 글자 수 카운트
- [ ] 입력값 임시 저장 (localStorage)

#### REQ-C-010: 출품 제출

**When** 사용자가 모든 정보를 입력하고 "제출" 버튼을 클릭하면,
**the system shall** Firestore에 출품 정보를 저장하고 완료 화면을 표시한다.

**Acceptance Criteria:**
- [ ] 제출 전 미리보기 단계 표시
- [ ] 중복 제출 방지
- [ ] 제출 완료 시 확인 모달 표시
- [ ] 공모전 상세 페이지로 리다이렉트

---

### 2.4 Firestore 데이터 구조

#### REQ-C-011: competitions 컬렉션

```typescript
interface Competition {
  id: string;
  title: string;
  description: string;
  thumbnail: string;
  category: 'visual_design' | 'industrial_design' | 'craft' | 'fine_art';
  organizer: string;
  prize: number;
  startDate: Timestamp;
  endDate: Timestamp;
  rules: string;
  schedule: ScheduleItem[];
  participantCount: number;
  status: 'upcoming' | 'active' | 'ended';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

interface ScheduleItem {
  date: string;
  title: string;
  description?: string;
}
```

#### REQ-C-012: submissions 컬렉션

```typescript
interface Submission {
  id: string;
  competitionId: string;
  userId: string;
  title: string;
  description?: string;
  imageUrl: string;
  tools?: string;
  duration?: string;
  status: 'pending' | 'approved' | 'rejected' | 'winner';
  createdAt: Timestamp;
  updatedAt: Timestamp;
}
```

#### REQ-C-013: Firestore 보안 규칙

**Acceptance Criteria:**
- [ ] 공모전 조회: 모든 사용자 허용
- [ ] 출품 생성: 로그인 사용자만 허용
- [ ] 출품 조회: 모든 사용자 허용
- [ ] 출품 수정/삭제: 본인만 허용

---

### 2.5 초기 데이터

#### REQ-C-014: 테스트 데이터 시딩

**Acceptance Criteria:**
- [ ] 테스트 공모전 5개 생성
- [ ] 테스트 출품작 20개 생성
- [ ] 각 카테고리별 최소 1개 공모전
- [ ] 상태별 분포: 진행중 3개, 마감임박 1개, 종료 1개

---

## 3. 기술 스택

### Frontend

- React 18
- React Router 6
- CSS Modules
- Firebase SDK

### Backend

- Firebase Firestore
- Firebase Storage
- Firebase Auth (기존)

---

## 4. 파일 구조

```
my-app/src/
├── pages/
│   ├── competitions/
│   │   ├── CompetitionList/
│   │   │   ├── CompetitionList.js
│   │   │   ├── CompetitionList.module.css
│   │   │   └── index.js
│   │   ├── CompetitionDetail/
│   │   │   ├── CompetitionDetail.js
│   │   │   ├── CompetitionDetail.module.css
│   │   │   └── index.js
│   │   └── SubmitPage/
│   │       ├── SubmitPage.js
│   │       ├── SubmitPage.module.css
│   │       └── index.js
├── components/
│   ├── competitions/
│   │   ├── CompetitionCard/
│   │   ├── CompetitionFilter/
│   │   ├── CompetitionSort/
│   │   ├── CompetitionHeader/
│   │   ├── CompetitionTabs/
│   │   ├── SubmissionGallery/
│   │   └── ImageUploader/
├── services/
│   ├── competitionService.js
│   └── submissionService.js
├── hooks/
│   ├── useCompetitions.js
│   └── useSubmission.js
└── scripts/
    └── seedCompetitions.js
```

---

## 5. 마일스톤

| 단계 | 작업 | 예상 시간 |
|------|------|----------|
| 1 | Firestore 서비스 및 초기 데이터 | 2시간 |
| 2 | 공모전 목록 페이지 | 4시간 |
| 3 | 공모전 상세 페이지 리팩토링 | 3시간 |
| 4 | 출품 페이지 | 3시간 |
| 5 | 테스트 및 QA | 2시간 |

**총 예상 시간**: 14시간

---

## 6. 의존성

### 선행 작업

- [x] SPEC-UI-001: 공통 UI 컴포넌트
- [x] SPEC-FIREBASE-001: Firebase 연동
- [x] SPEC-BACKEND-001: Backend 셋업

### 후속 작업

- SPEC-EDU-001: AI 진단 페이지 (Phase 2)

---

*문서 버전: 1.0*
*작성일: 2026-01-17*
