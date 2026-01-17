# MIRIP 프로젝트 구조

> 미립(MIRIP) 전체 프로젝트 디렉토리 및 라우트 구조

---

## 1. 전체 디렉토리 구조

```
mirip/
├── landing/                      # 랜딩 페이지 (Vanilla JS)
│   ├── assets/                   # 이미지, 폰트 등 에셋
│   ├── .firebase/                # Firebase 캐시
│   ├── .firebaserc               # Firebase 프로젝트 설정
│   ├── firebase.json             # Firebase 호스팅 설정
│   ├── index.html                # 메인 HTML
│   ├── script.js                 # JavaScript 로직
│   └── styles.css                # CSS 스타일
│
├── my-app/                       # 메인 React 앱
│   ├── public/                   # 정적 자원
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/           # 공통 컴포넌트
│   │   │   │   ├── Button/
│   │   │   │   ├── Card/
│   │   │   │   ├── Modal/
│   │   │   │   ├── Loading/
│   │   │   │   ├── Header/
│   │   │   │   └── Footer/
│   │   │   │
│   │   │   ├── diagnosis/        # AI 진단 관련
│   │   │   │   ├── ImageUploader/
│   │   │   │   ├── RubricScore/
│   │   │   │   ├── TierBadge/
│   │   │   │   ├── FeedbackPanel/
│   │   │   │   └── HistoryList/
│   │   │   │
│   │   │   ├── competition/      # 공모전 관련
│   │   │   │   ├── CompetitionCard/
│   │   │   │   ├── CompetitionFilter/
│   │   │   │   ├── SubmissionForm/
│   │   │   │   ├── ResultCard/
│   │   │   │   └── GalleryGrid/
│   │   │   │
│   │   │   ├── credential/       # 크레덴셜 관련
│   │   │   │   ├── ProfileCard/
│   │   │   │   ├── AchievementList/
│   │   │   │   ├── PortfolioGrid/
│   │   │   │   └── CredentialBadge/
│   │   │   │
│   │   │   └── payment/          # 결제 관련
│   │   │       ├── PricingCard/
│   │   │       ├── CheckoutForm/
│   │   │       └── PaymentStatus/
│   │   │
│   │   ├── pages/
│   │   │   ├── Home/             # 홈 페이지
│   │   │   ├── MiripComp/        # 공모전 페이지들
│   │   │   │   ├── CompetitionList/
│   │   │   │   ├── CompetitionDetail/
│   │   │   │   ├── SubmitPage/
│   │   │   │   └── ResultPage/
│   │   │   │
│   │   │   ├── MiripEdu/         # AI 진단 페이지들
│   │   │   │   ├── DiagnosisLanding/
│   │   │   │   ├── DiagnosisPage/
│   │   │   │   ├── ResultPage/
│   │   │   │   └── HistoryPage/
│   │   │   │
│   │   │   ├── Credential/       # 크레덴셜 페이지들
│   │   │   │   ├── ProfileEdit/
│   │   │   │   ├── PublicProfile/
│   │   │   │   └── Portfolio/
│   │   │   │
│   │   │   ├── Payment/          # 결제 페이지들
│   │   │   │   ├── Pricing/
│   │   │   │   ├── Checkout/
│   │   │   │   └── Success/
│   │   │   │
│   │   │   └── PreRegister/      # 사전등록
│   │   │
│   │   ├── hooks/                # 커스텀 훅
│   │   │   ├── useAuth.js
│   │   │   ├── useFirestore.js
│   │   │   ├── useDiagnosis.js
│   │   │   ├── useCompetition.js
│   │   │   └── usePayment.js
│   │   │
│   │   ├── utils/                # 유틸리티 함수
│   │   │   ├── formatters.js
│   │   │   ├── validators.js
│   │   │   ├── imageProcessing.js
│   │   │   └── constants.js
│   │   │
│   │   ├── services/             # API 서비스
│   │   │   ├── api.js
│   │   │   ├── firebase.js
│   │   │   ├── diagnosis.js
│   │   │   ├── competition.js
│   │   │   └── payment.js
│   │   │
│   │   ├── store/                # 상태 관리 (Zustand)
│   │   │   ├── authStore.js
│   │   │   ├── diagnosisStore.js
│   │   │   └── competitionStore.js
│   │   │
│   │   ├── App.js                # 라우터 설정
│   │   ├── global.css            # 전역 스타일
│   │   └── index.js              # 엔트리 포인트
│   │
│   └── package.json              # 의존성 관리
│
├── backend/                      # FastAPI 서버
│   ├── app/
│   │   ├── main.py               # 앱 엔트리 포인트
│   │   ├── config.py             # 설정 관리
│   │   │
│   │   ├── routers/              # API 라우터
│   │   │   ├── evaluate.py       # 단일 이미지 평가
│   │   │   ├── compare.py        # 복수 이미지 비교
│   │   │   ├── competition.py    # 공모전 API
│   │   │   └── credential.py     # 크레덴셜 API
│   │   │
│   │   ├── services/             # 비즈니스 로직
│   │   │   ├── inference.py      # ML 추론 서비스
│   │   │   ├── feedback.py       # 피드백 생성
│   │   │   └── storage.py        # 스토리지 관리
│   │   │
│   │   ├── models/               # Pydantic 모델
│   │   │   ├── request.py        # 요청 모델
│   │   │   └── response.py       # 응답 모델
│   │   │
│   │   └── ml/                   # ML 모듈
│   │       ├── feature_extractor.py   # 피처 추출
│   │       ├── fusion_module.py       # 피처 융합
│   │       ├── rubric_heads.py        # 루브릭 평가 헤드
│   │       ├── tier_classifier.py     # 티어 분류기
│   │       └── weights/               # 모델 가중치
│   │
│   ├── requirements.txt          # Python 의존성
│   ├── Dockerfile                # Docker 이미지
│   └── docker-compose.yml        # Docker Compose 설정
│
├── .claude/                      # Claude Code 설정
│   └── skills/                   # MoAI-ADK 스킬 모음
│
├── .moai/                        # MoAI-ADK 프로젝트 설정
│   ├── config/                   # 설정 파일
│   ├── project/                  # 프로젝트 문서
│   ├── specs/                    # SPEC 명세서
│   └── memory/                   # 컨텍스트 메모리
│
├── .gitignore                    # Git 제외 파일
├── .mcp.json                     # MCP 서버 설정
├── CLAUDE.md                     # Claude Code 지침
├── firebase.json                 # Firebase 설정
├── package.json                  # 루트 의존성
└── README.md                     # 프로젝트 소개
```

---

## 2. 라우트 구조

### 전체 라우트 목록 (18+ 라우트)

```
/                           -> 홈 (서비스 소개)
/pre-register               -> 사전등록

# 공모전 (Phase 1)
/competitions               -> 공모전 목록
/competitions/:id           -> 공모전 상세
/competitions/:id/submit    -> 출품하기
/competitions/:id/result    -> 심사 결과

# AI 진단 (Phase 2)
/edu                        -> AI 진단 랜딩
/edu/diagnosis              -> 진단 페이지
/edu/result/:id             -> 결과 페이지
/edu/history                -> 진단 이력

# 크레덴셜 (Phase 3)
/profile                    -> 내 프로필 편집
/profile/:username          -> 공개 프로필
/portfolio                  -> 포트폴리오 관리

# 결제 (Phase 3)
/pricing                    -> 요금제 안내
/checkout                   -> 결제 진행
/checkout/success           -> 결제 완료
```

### 라우트 상세

| 경로 | 컴포넌트 | 기능 | Phase |
|-----|---------|------|-------|
| `/` | Home | 메인 대시보드, 서비스 소개 | 1 |
| `/pre-register` | PreRegister | 사전등록 폼 | 1 |
| `/competitions` | CompetitionList | 공모전 목록, 필터, 검색 | 1 |
| `/competitions/:id` | CompetitionDetail | 공모전 상세 정보, 탭 메뉴 | 1 |
| `/competitions/:id/submit` | SubmitPage | 작품 출품 폼 | 1 |
| `/competitions/:id/result` | ResultPage | 심사 결과 확인 | 1 |
| `/edu` | DiagnosisLanding | AI 진단 소개 | 2 |
| `/edu/diagnosis` | DiagnosisPage | 이미지 업로드 및 진단 | 2 |
| `/edu/result/:id` | ResultPage | 진단 결과 상세 | 2 |
| `/edu/history` | HistoryPage | 진단 이력 목록 | 2 |
| `/profile` | ProfileEdit | 프로필 편집 | 3 |
| `/profile/:username` | PublicProfile | 공개 프로필 | 3 |
| `/portfolio` | Portfolio | 포트폴리오 관리 | 3 |
| `/pricing` | Pricing | 요금제 안내 | 3 |
| `/checkout` | Checkout | 결제 진행 | 3 |
| `/checkout/success` | Success | 결제 완료 | 3 |

---

## 3. 컴포넌트 분류

### 3.1 공통 컴포넌트 (`components/common/`)

| 컴포넌트 | 용도 | 주요 Props |
|---------|------|-----------|
| `Button` | 공통 버튼 | variant, size, onClick |
| `Card` | 기본 카드 레이아웃 | children, shadow, padding |
| `Modal` | 모달 다이얼로그 | isOpen, onClose, title |
| `Loading` | 로딩 스피너 | size, color |
| `Header` | 페이지 헤더 | title, breadcrumbs |
| `Footer` | 페이지 푸터 | - |

### 3.2 진단 컴포넌트 (`components/diagnosis/`)

| 컴포넌트 | 용도 | 주요 Props |
|---------|------|-----------|
| `ImageUploader` | 이미지 업로드 영역 | onUpload, maxSize |
| `RubricScore` | 루브릭별 점수 표시 | scores, weights |
| `TierBadge` | 티어 배지 (S/A/B/C) | tier, probabilities |
| `FeedbackPanel` | 강점/개선점 피드백 | strengths, improvements |
| `HistoryList` | 진단 이력 목록 | history, onSelect |

### 3.3 공모전 컴포넌트 (`components/competition/`)

| 컴포넌트 | 용도 | 주요 Props |
|---------|------|-----------|
| `CompetitionCard` | 공모전 카드 | competition, onClick |
| `CompetitionFilter` | 필터/정렬 UI | filters, onFilterChange |
| `SubmissionForm` | 출품 작성 폼 | competitionId, onSubmit |
| `ResultCard` | 심사 결과 카드 | result, rank |
| `GalleryGrid` | 출품작 갤러리 | submissions, layout |

### 3.4 크레덴셜 컴포넌트 (`components/credential/`)

| 컴포넌트 | 용도 | 주요 Props |
|---------|------|-----------|
| `ProfileCard` | 프로필 카드 | user, editable |
| `AchievementList` | 수상 이력 목록 | achievements |
| `PortfolioGrid` | 포트폴리오 그리드 | artworks, columns |
| `CredentialBadge` | 인증 배지 | type, verified |

### 3.5 결제 컴포넌트 (`components/payment/`)

| 컴포넌트 | 용도 | 주요 Props |
|---------|------|-----------|
| `PricingCard` | 요금제 카드 | plan, features, price |
| `CheckoutForm` | 결제 정보 입력 | plan, onCheckout |
| `PaymentStatus` | 결제 상태 표시 | status, message |

---

## 4. 모듈 구성

### 4.1 hooks/ 모듈

```javascript
// hooks/useAuth.js
// Firebase Authentication 관련 커스텀 훅
// - useCurrentUser(): 현재 로그인 사용자
// - useSignIn(): 로그인 함수
// - useSignOut(): 로그아웃 함수

// hooks/useFirestore.js
// Firestore CRUD 관련 커스텀 훅
// - useDocument(collection, id): 단일 문서
// - useCollection(collection, query): 문서 목록

// hooks/useDiagnosis.js
// AI 진단 관련 커스텀 훅
// - useDiagnosis(): 진단 요청/결과 관리
// - useDiagnosisHistory(): 진단 이력 관리

// hooks/useCompetition.js
// 공모전 관련 커스텀 훅
// - useCompetitions(): 공모전 목록
// - useSubmission(): 출품 관리

// hooks/usePayment.js
// 결제 관련 커스텀 훅
// - useSubscription(): 구독 상태
// - useCheckout(): 결제 처리
```

### 4.2 utils/ 모듈

```javascript
// utils/formatters.js
// - formatDate(date): 날짜 포맷팅
// - formatCurrency(amount): 금액 포맷팅
// - formatTier(tier): 티어 표시 포맷팅

// utils/validators.js
// - validateImage(file): 이미지 유효성 검사
// - validateEmail(email): 이메일 검증
// - validateSubmission(data): 출품 데이터 검증

// utils/imageProcessing.js
// - resizeImage(file, maxSize): 이미지 리사이즈
// - convertToBase64(file): Base64 변환
// - getImageDimensions(file): 이미지 크기 확인

// utils/constants.js
// - TIERS: 티어 정의
// - DEPARTMENTS: 학과 목록
// - RUBRICS: 평가 축 정의
```

### 4.3 services/ 모듈

```javascript
// services/api.js
// - axios 인스턴스 설정
// - 인터셉터 설정 (토큰, 에러 처리)

// services/firebase.js
// - Firebase 앱 초기화
// - Auth, Firestore, Storage 인스턴스

// services/diagnosis.js
// - evaluateImage(image, options): 단일 이미지 평가
// - compareImages(images, options): 복수 이미지 비교
// - getHistory(userId, limit): 진단 이력 조회

// services/competition.js
// - getCompetitions(filters): 공모전 목록
// - getCompetitionById(id): 공모전 상세
// - submitArtwork(competitionId, data): 작품 출품

// services/payment.js
// - initiatePayment(plan): 결제 시작
// - verifyPayment(paymentKey): 결제 검증
// - getSubscriptionStatus(userId): 구독 상태 확인
```

---

## 5. 파일 명명 규칙

### JavaScript/React

- 컴포넌트: `PascalCase.js` (예: `CompetitionCard.js`)
- 커스텀 훅: `use` 접두사 + `camelCase.js` (예: `useAuth.js`)
- 유틸리티: `camelCase.js` (예: `formatters.js`)
- 서비스: `camelCase.js` (예: `diagnosis.js`)
- 스토어: `camelCase.js` + `Store` 접미사 (예: `authStore.js`)

### CSS

- CSS Module: `ComponentName.module.css`
- 전역 스타일: `global.css`

### Python (Backend)

- 모듈: `snake_case.py` (예: `feature_extractor.py`)
- 클래스: `PascalCase` (예: `InferenceEngine`)

---

## 6. 모듈 의존성 관계

```
index.js
    └── App.js
          ├── Home
          │     ├── Header
          │     ├── Button
          │     └── Footer
          │
          ├── CompetitionList
          │     ├── CompetitionCard
          │     ├── CompetitionFilter
          │     └── GalleryGrid
          │
          ├── DiagnosisPage
          │     ├── ImageUploader
          │     ├── RubricScore
          │     ├── TierBadge
          │     └── FeedbackPanel
          │
          ├── ProfileEdit
          │     ├── ProfileCard
          │     ├── AchievementList
          │     └── PortfolioGrid
          │
          └── Checkout
                ├── PricingCard
                ├── CheckoutForm
                └── PaymentStatus
```

---

*문서 버전: 2.1*
*최종 업데이트: 2025년 1월*
