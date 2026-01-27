# SPEC-FIREBASE-001: Firebase 연동 및 사전등록 페이지 구현

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-FIREBASE-001 |
| **제목** | Firebase 연동 및 사전등록 페이지 구현 |
| **상태** | Completed |
| **우선순위** | High |
| **Phase** | Phase 1 Week 2-3 |
| **생성일** | 2025-01-17 |
| **담당 도메인** | Frontend (Firebase Integration) |
| **의존성** | SPEC-UI-001 (공통 컴포넌트) |

---

## 1. 환경 (Environment)

### 1.1 기술 스택

| 기술 | 버전 | 용도 |
|------|------|------|
| React | 18.3.1 | UI 프레임워크 |
| Firebase SDK | 10.x (latest) | Firebase 서비스 연동 |
| Firebase Hosting | - | 정적 파일 배포 |
| Firebase Firestore | - | 사전등록 데이터 저장 |
| CSS Modules | - | 컴포넌트 스타일링 |

### 1.2 Firebase 프로젝트

| 항목 | 값 |
|------|-----|
| **프로젝트명** | milip-prototype |
| **프로젝트 ID** | milip-prototype |
| **상태** | 기존 프로젝트 (생성 완료) |

### 1.3 디렉토리 구조

```
my-app/src/
├── config/
│   └── firebase.js              # Firebase 초기화 설정
├── services/
│   └── registrationService.js   # Firestore 작업 서비스
├── pages/
│   └── Landing/
│       ├── Landing.js           # 랜딩 페이지 컴포넌트
│       └── Landing.module.css   # 랜딩 페이지 스타일
├── components/
│   ├── common/                  # SPEC-UI-001 컴포넌트 (기존)
│   │   ├── Button/
│   │   ├── Card/
│   │   ├── Modal/
│   │   ├── Header/
│   │   ├── Footer/
│   │   └── Loading/
│   └── features/
│       └── RegistrationForm/
│           ├── RegistrationForm.js
│           └── RegistrationForm.module.css
└── global.css                   # CSS 변수 정의 (기존)
```

### 1.4 디자인 시스템 참조

- **디자인 컨셉**: 미술관/갤러리 스타일, 미니멀 & 여백 중심
- **컬러 시스템**: global.css 정의 CSS 변수 사용
- **타이포그래피**: Noto Serif KR (제목), Pretendard (본문), Cormorant Garamond (영문)
- **기존 landing/**: 디자인 레퍼런스로 활용 (HTML -> React 변환)

---

## 2. 가정 (Assumptions)

### 2.1 기술적 가정

| ID | 가정 | 신뢰도 | 검증 방법 |
|----|------|--------|-----------|
| A-01 | Firebase 프로젝트(milip-prototype)가 이미 생성되어 있다 | High | Firebase Console 확인 |
| A-02 | Firebase SDK 10.x가 React 18과 호환된다 | High | Firebase 공식 문서 확인 |
| A-03 | Firestore 보안 규칙이 사전등록 컬렉션에 대해 쓰기 허용으로 설정 가능하다 | High | Firebase 콘솔에서 설정 |
| A-04 | SPEC-UI-001의 공통 컴포넌트가 완료되어 재사용 가능하다 | High | 코드 확인됨 |
| A-05 | 환경 변수를 통한 Firebase 설정 관리가 가능하다 (.env) | High | CRA 문서 확인 |

### 2.2 비즈니스 가정

| ID | 가정 | 신뢰도 | 틀릴 경우 영향 |
|----|------|--------|----------------|
| B-01 | 사전등록 페이지는 MVP 출시 전까지 유지된다 | High | 페이지 비활성화 필요 |
| B-02 | 수집 데이터는 이름, 이메일, 유저유형만 필요하다 | High | 스키마 확장 필요 |
| B-03 | 인증 없이 사전등록이 가능해야 한다 | High | Firebase Auth 추가 필요 |

---

## 3. 요구사항 (Requirements) - EARS 형식

### 3.1 전역 요구사항 (Ubiquitous)

| ID | 요구사항 | 검증 방법 |
|----|----------|-----------|
| REQ-U-01 | 시스템은 **항상** 환경 변수를 통해 Firebase 설정을 관리해야 한다 | .env 파일 및 코드 리뷰 |
| REQ-U-02 | 시스템은 **항상** SPEC-UI-001에서 정의한 공통 컴포넌트를 재사용해야 한다 | 코드 리뷰 |
| REQ-U-03 | 시스템은 **항상** global.css에 정의된 CSS 변수를 사용해야 한다 | 코드 리뷰 |
| REQ-U-04 | 시스템은 **항상** 반응형 디자인을 지원해야 한다 (1024px, 768px, 480px) | 반응형 테스트 |
| REQ-U-05 | 시스템은 **항상** 사용자 친화적인 에러 메시지를 표시해야 한다 | UI/UX 테스트 |

### 3.2 이벤트 기반 요구사항 (Event-Driven)

#### 3.2.1 Firebase 초기화

| ID | 요구사항 |
|----|----------|
| REQ-E-01 | **WHEN** 애플리케이션이 시작되면 **THEN** Firebase가 milip-prototype 설정으로 초기화되어야 한다 |
| REQ-E-02 | **WHEN** Firebase 초기화가 실패하면 **THEN** 콘솔에 에러가 로그되고 앱은 계속 동작해야 한다 |

#### 3.2.2 랜딩 페이지

| ID | 요구사항 |
|----|----------|
| REQ-E-03 | **WHEN** 사용자가 랜딩 페이지를 방문하면 **THEN** Hero 섹션이 MIRIP 브랜딩과 함께 표시되어야 한다 |
| REQ-E-04 | **WHEN** 사용자가 페이지를 스크롤하면 **THEN** 각 섹션이 순차적으로 표시되어야 한다 |
| REQ-E-05 | **WHEN** 사용자가 네비게이션 링크를 클릭하면 **THEN** 해당 섹션으로 스무스 스크롤되어야 한다 |

#### 3.2.3 사전등록 폼

| ID | 요구사항 |
|----|----------|
| REQ-E-06 | **WHEN** 사용자가 등록 폼을 제출하면 **THEN** 이름, 이메일, 유저유형 필드가 검증되어야 한다 |
| REQ-E-07 | **WHEN** 검증이 성공하면 **THEN** 등록 데이터가 Firestore 'registrations' 컬렉션에 저장되어야 한다 |
| REQ-E-08 | **WHEN** 저장이 성공하면 **THEN** 성공 Modal이 표시되어야 한다 |
| REQ-E-09 | **WHEN** 저장이 실패하면 **THEN** 에러 메시지가 사용자에게 표시되어야 한다 |
| REQ-E-10 | **WHEN** 폼이 제출 중이면 **THEN** 버튼이 로딩 상태로 변경되어야 한다 |

#### 3.2.4 Firestore 작업

| ID | 요구사항 |
|----|----------|
| REQ-E-11 | **WHEN** 등록 데이터가 저장되면 **THEN** registrationId가 자동 생성되어야 한다 |
| REQ-E-12 | **WHEN** 등록 데이터가 저장되면 **THEN** timestamp가 서버 시간으로 기록되어야 한다 |
| REQ-E-13 | **WHEN** 네트워크 오류가 발생하면 **THEN** 사용자에게 재시도 옵션이 제공되어야 한다 |

### 3.3 상태 기반 요구사항 (State-Driven)

| ID | 요구사항 |
|----|----------|
| REQ-S-01 | **IF** 이름 필드가 비어있으면 **THEN** "이름을 입력해주세요" 에러가 표시되어야 한다 |
| REQ-S-02 | **IF** 이메일 형식이 올바르지 않으면 **THEN** "올바른 이메일을 입력해주세요" 에러가 표시되어야 한다 |
| REQ-S-03 | **IF** 유저유형이 선택되지 않으면 **THEN** "유저 유형을 선택해주세요" 에러가 표시되어야 한다 |
| REQ-S-04 | **IF** 폼이 유효하지 않으면 **THEN** 제출 버튼이 비활성화 상태를 유지해야 한다 |
| REQ-S-05 | **IF** 제출이 진행 중이면 **THEN** 폼 입력이 비활성화되어야 한다 |

### 3.4 금지 요구사항 (Unwanted Behavior)

| ID | 요구사항 |
|----|----------|
| REQ-N-01 | 시스템은 Firebase 설정값을 코드에 하드코딩**하지 않아야 한다** |
| REQ-N-02 | 시스템은 중복 등록을 동일 세션에서 **허용하지 않아야 한다** |
| REQ-N-03 | 시스템은 Firestore에 검증되지 않은 데이터를 **저장하지 않아야 한다** |
| REQ-N-04 | 시스템은 에러 발생 시 기술적 세부사항을 사용자에게 **노출하지 않아야 한다** |
| REQ-N-05 | 시스템은 개인정보를 콘솔에 **로깅하지 않아야 한다** |

### 3.5 선택적 요구사항 (Optional)

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| REQ-O-01 | **가능하면** 이메일 중복 검사 기능을 제공한다 | Medium |
| REQ-O-02 | **가능하면** 등록 성공 후 이메일 확인 기능을 제공한다 | Low |
| REQ-O-03 | **가능하면** Google Analytics 이벤트 추적을 제공한다 | Low |

---

## 4. 명세 (Specifications)

### 4.1 Firebase 설정 (firebase.js)

#### 환경 변수

```javascript
// .env.local
REACT_APP_FIREBASE_API_KEY=xxx
REACT_APP_FIREBASE_AUTH_DOMAIN=milip-prototype.firebaseapp.com
REACT_APP_FIREBASE_PROJECT_ID=milip-prototype
REACT_APP_FIREBASE_STORAGE_BUCKET=milip-prototype.appspot.com
REACT_APP_FIREBASE_MESSAGING_SENDER_ID=xxx
REACT_APP_FIREBASE_APP_ID=xxx
```

#### 초기화 패턴

```javascript
// src/config/firebase.js
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
```

### 4.2 Registration Service

#### Firestore 스키마

```typescript
interface Registration {
  registrationId: string;     // 자동 생성 (Firestore document ID)
  name: string;               // 사용자 이름
  email: string;              // 이메일 주소
  userType: 'student' | 'parent' | 'artist' | 'organizer';
  timestamp: Timestamp;       // 서버 타임스탬프
  createdAt: string;          // ISO 8601 형식 (읽기용)
}
```

#### 서비스 인터페이스

```javascript
// src/services/registrationService.js
export const registrationService = {
  // 등록 데이터 저장
  async create(data) {},

  // 이메일 중복 검사 (Optional)
  async checkEmailExists(email) {},
};
```

### 4.3 RegistrationForm 컴포넌트

#### Props Interface

```typescript
interface RegistrationFormProps {
  onSuccess?: (registration: Registration) => void;
  onError?: (error: Error) => void;
}
```

#### 유저유형 옵션

| Value | Label |
|-------|-------|
| student | 입시생 |
| parent | 학부모 |
| artist | 신진 작가 |
| organizer | 공모전 주최자 |

#### 폼 검증 규칙

| 필드 | 검증 규칙 | 에러 메시지 |
|------|-----------|------------|
| name | required, min 2자 | "이름을 입력해주세요" / "이름은 2자 이상이어야 합니다" |
| email | required, email format | "이메일을 입력해주세요" / "올바른 이메일을 입력해주세요" |
| userType | required | "유저 유형을 선택해주세요" |

### 4.4 Landing 페이지

#### 섹션 구성

| 섹션 | 설명 | 컴포넌트 |
|------|------|----------|
| Hero | MIRIP 브랜딩, 메인 CTA | Header (common) |
| Problem | 문제점 3가지 나열 | Card (common) |
| Solution | 4단계 타임라인 | - |
| AI Preview | AI 진단 미리보기 | Card (common) |
| CTA | 사전등록 폼 | RegistrationForm |
| Footer | 푸터 정보 | Footer (common) |

#### 스타일 명세

- **Hero 섹션**: 전체 화면 높이 (100vh), 중앙 정렬
- **섹션 간격**: var(--spacing-section) (120px)
- **애니메이션**: 스크롤 시 fade-in 효과 (0.6s)
- **반응형**: 768px 이하에서 단일 컬럼 레이아웃

---

## 5. 추적성 (Traceability)

### 5.1 관련 문서

- `.moai/project/product.md` - 서비스 정의 및 디자인 철학
- `.moai/project/tech.md` - 기술 스택 및 Firebase 설정 정보
- `.moai/specs/SPEC-UI-001/` - 공통 컴포넌트 SPEC
- `landing/index.html` - 디자인 레퍼런스 (static HTML)

### 5.2 SPEC 태그

```
[SPEC-FIREBASE-001] Firebase Integration & Pre-registration
├── [FB-001-CFG] Firebase Configuration
├── [FB-001-SVC] Registration Service
├── [FB-001-FRM] RegistrationForm Component
├── [FB-001-LND] Landing Page
└── [FB-001-STY] Landing Page Styles
```

### 5.3 컴포넌트 재사용

| 공통 컴포넌트 (SPEC-UI-001) | 사용 위치 |
|----------------------------|----------|
| Button | Hero CTA, RegistrationForm 제출 |
| Card | Problem 섹션, AI Preview 섹션 |
| Modal | 등록 성공 모달 |
| Header | 랜딩 페이지 상단 네비게이션 |
| Footer | 랜딩 페이지 하단 |
| Loading | 폼 제출 중 상태 |

---

*SPEC 버전: 1.0*
*최종 업데이트: 2025-01-17*
