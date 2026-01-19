# SPEC-COMP-001 구현 계획

## Phase 1: Firestore 서비스 및 데이터 모델

### 1.1 Competition 서비스
- `services/competitionService.js` 생성
- CRUD 함수: getCompetitions, getCompetitionById, createCompetition
- 필터/정렬 로직 구현

### 1.2 Submission 서비스
- `services/submissionService.js` 생성
- CRUD 함수: getSubmissions, createSubmission, updateSubmission
- 이미지 업로드 함수

### 1.3 초기 데이터 시딩
- 테스트 공모전 5개
- 테스트 출품작 20개

## Phase 2: 공모전 목록 페이지

### 2.1 CompetitionList 페이지
- 카드 그리드 레이아웃
- 반응형 디자인 (4열 → 2열 → 1열)

### 2.2 CompetitionCard 컴포넌트
- 썸네일, 제목, D-Day, 상금, 분야
- 호버 효과

### 2.3 CompetitionFilter 컴포넌트
- 분야 필터 (시각디자인, 산업디자인, 공예, 회화)
- 상태 필터 (진행중, 마감임박, 종료)

### 2.4 CompetitionSort 컴포넌트
- 정렬 옵션 드롭다운

### 2.5 무한 스크롤
- useInfiniteScroll 훅
- 로딩 인디케이터

## Phase 3: 공모전 상세 페이지

### 3.1 CompetitionDetail 리팩토링
- 기존 MiripComp.js 코드 활용
- 동적 데이터 바인딩

### 3.2 CompetitionHeader 컴포넌트
- 썸네일, 제목, D-Day 배지
- 참가자 수, 상금

### 3.3 CompetitionTabs 컴포넌트
- 상세정보, 출품작, 결과 탭
- URL 해시 연동

### 3.4 SubmissionGallery 컴포넌트
- 출품작 그리드
- 라이트박스 효과

## Phase 4: 출품 페이지

### 4.1 SubmitPage 페이지
- 스텝 위저드 (업로드 → 정보입력 → 미리보기 → 완료)

### 4.2 ImageUploader 컴포넌트
- 드래그앤드롭 지원
- 업로드 진행률
- 미리보기

### 4.3 SubmissionForm 컴포넌트
- 작품 제목, 설명, 도구, 기간
- 유효성 검사

## Phase 5: 라우팅 및 통합

### 5.1 라우터 설정
- /competitions
- /competitions/:id
- /competitions/:id/submit

### 5.2 네비게이션 연결
- 헤더 메뉴 연결
- 버튼 링크 연결

---

## 구현 순서

1. competitionService.js (Firestore CRUD)
2. submissionService.js (Firestore CRUD + Storage)
3. 시드 데이터 스크립트
4. CompetitionCard 컴포넌트
5. CompetitionList 페이지
6. CompetitionFilter, CompetitionSort 컴포넌트
7. CompetitionDetail 페이지 리팩토링
8. ImageUploader 컴포넌트
9. SubmitPage 페이지
10. 라우터 연결 및 통합 테스트
