# SPEC-FIREBASE-001: 구현 계획

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-FIREBASE-001 |
| **제목** | Firebase 연동 및 사전등록 페이지 구현 |
| **관련 SPEC** | spec.md |

---

## 1. 마일스톤

### Primary Goal: Firebase 연동 및 핵심 기능

| 순서 | 작업 | 산출물 | 의존성 |
|------|------|--------|--------|
| 1 | Firebase SDK 설치 및 설정 | firebase.js, .env.local | 없음 |
| 2 | Registration Service 구현 | registrationService.js | firebase.js |
| 3 | RegistrationForm 컴포넌트 구현 | RegistrationForm.js, .module.css | registrationService.js, SPEC-UI-001 |
| 4 | 성공/에러 처리 구현 | Modal 연동 | RegistrationForm |

### Secondary Goal: 랜딩 페이지 구현

| 순서 | 작업 | 산출물 | 의존성 |
|------|------|--------|--------|
| 5 | Landing 페이지 컴포넌트 구조 | Landing.js | Header, Footer (SPEC-UI-001) |
| 6 | Hero 섹션 구현 | Landing.module.css | Button (SPEC-UI-001) |
| 7 | Problem/Solution 섹션 구현 | Landing.module.css | Card (SPEC-UI-001) |
| 8 | AI Preview 섹션 구현 | Landing.module.css | Card (SPEC-UI-001) |
| 9 | CTA 섹션 연동 | Landing.js | RegistrationForm |

### Final Goal: 품질 보증 및 배포 준비

| 순서 | 작업 | 산출물 | 의존성 |
|------|------|--------|--------|
| 10 | 단위 테스트 작성 | *.test.js | 컴포넌트 구현 완료 |
| 11 | 반응형 테스트 및 조정 | 스타일 수정 | 랜딩 페이지 구현 |
| 12 | Firebase Hosting 배포 테스트 | firebase.json | 전체 구현 완료 |

---

## 2. 기술 접근 방식

### 2.1 Firebase 설정 전략

```
환경 변수 기반 설정
├── .env.local (로컬 개발용, Git 제외)
├── .env.production (프로덕션용)
└── firebase.js (환경 변수 참조)
```

**설정 패턴:**
- Create React App의 `REACT_APP_` prefix 환경 변수 사용
- firebase 모듈을 통한 개별 서비스 import (tree-shaking 최적화)
- Firestore만 초기 연동 (Auth, Storage는 Phase 2)

### 2.2 서비스 레이어 패턴

```javascript
// 서비스 레이어 구조
src/services/
└── registrationService.js
    ├── create(data)       // 등록 생성
    ├── checkEmailExists() // 이메일 중복 검사 (Optional)
    └── validateData()     // 클라이언트 검증
```

**에러 처리 전략:**
- 네트워크 에러: 재시도 안내 메시지
- 검증 에러: 필드별 에러 메시지
- 서버 에러: 일반 에러 메시지 (기술 세부사항 숨김)

### 2.3 컴포넌트 아키텍처

```
Landing (페이지)
├── Header (SPEC-UI-001)
├── HeroSection (내부 컴포넌트)
├── ProblemSection (내부 컴포넌트)
├── SolutionSection (내부 컴포넌트)
├── AIPreviewSection (내부 컴포넌트)
├── CTASection
│   └── RegistrationForm (feature 컴포넌트)
│       ├── Button (SPEC-UI-001)
│       └── Modal (SPEC-UI-001)
└── Footer (SPEC-UI-001)
```

### 2.4 폼 상태 관리

```javascript
// useState를 활용한 폼 상태 관리
const [formData, setFormData] = useState({
  name: '',
  email: '',
  userType: ''
});

const [formState, setFormState] = useState({
  isSubmitting: false,
  isSubmitted: false,
  errors: {}
});
```

**검증 타이밍:**
- onChange: 개별 필드 검증 (debounced)
- onBlur: 필드 검증 실행
- onSubmit: 전체 폼 검증

---

## 3. 아키텍처 설계 방향

### 3.1 디렉토리 구조

```
my-app/src/
├── config/
│   └── firebase.js              # Firebase 초기화
├── services/
│   └── registrationService.js   # Firestore 작업
├── pages/
│   └── Landing/
│       ├── Landing.js           # 메인 페이지 컴포넌트
│       ├── Landing.module.css   # 페이지 스타일
│       └── sections/            # 섹션 컴포넌트 (선택적 분리)
│           ├── HeroSection.js
│           ├── ProblemSection.js
│           ├── SolutionSection.js
│           └── AIPreviewSection.js
├── components/
│   ├── common/                  # 기존 (SPEC-UI-001)
│   └── features/
│       └── RegistrationForm/
│           ├── RegistrationForm.js
│           ├── RegistrationForm.module.css
│           └── index.js
└── hooks/                       # (선택적)
    └── useRegistration.js       # 등록 로직 커스텀 훅
```

### 3.2 데이터 흐름

```
[사용자 입력]
     ↓
[RegistrationForm]
     ↓ (검증)
[registrationService.create()]
     ↓
[Firestore 'registrations' 컬렉션]
     ↓ (성공/실패)
[Modal 표시]
```

### 3.3 스타일 전략

**CSS Modules 명명 규칙:**
- BEM 스타일 적용: `.section`, `.section__title`, `.section--hero`
- CSS 변수 참조: `var(--color-primary)`, `var(--spacing-md)`

**반응형 브레이크포인트:**
```css
/* Desktop first approach */
@media (max-width: 1024px) { /* Tablet */ }
@media (max-width: 768px) { /* Mobile Landscape */ }
@media (max-width: 480px) { /* Mobile Portrait */ }
```

---

## 4. 리스크 및 대응 방안

### 4.1 기술 리스크

| 리스크 | 확률 | 영향 | 대응 방안 |
|--------|------|------|----------|
| Firebase 초기화 실패 | Low | High | 에러 바운더리 + 폴백 UI |
| Firestore 쓰기 권한 문제 | Medium | High | 보안 규칙 사전 테스트 |
| 환경 변수 누락 | Low | High | 빌드 시 환경 변수 검증 |
| CSS 변수 미지원 브라우저 | Low | Medium | 폴백 값 제공 |

### 4.2 의존성 리스크

| 의존성 | 리스크 | 대응 방안 |
|--------|--------|----------|
| SPEC-UI-001 미완료 | Medium | 병렬 개발, 모의 컴포넌트 사용 |
| Firebase SDK 버전 충돌 | Low | package.json 버전 고정 |

### 4.3 일정 리스크

| 리스크 | 대응 방안 |
|--------|----------|
| 디자인 변경 요청 | CSS 변수 기반 테마로 변경 용이 |
| 추가 필드 요구 | 스키마 확장 가능한 구조 설계 |

---

## 5. 테스트 전략

### 5.1 단위 테스트

| 대상 | 테스트 항목 | 도구 |
|------|------------|------|
| firebase.js | 초기화 성공/실패 | Jest |
| registrationService | CRUD 작업, 에러 처리 | Jest + Firebase Emulator |
| RegistrationForm | 입력 검증, 제출 처리 | React Testing Library |
| Landing | 렌더링, 섹션 표시 | React Testing Library |

### 5.2 통합 테스트

| 시나리오 | 검증 항목 |
|----------|----------|
| 사전등록 성공 플로우 | 폼 입력 → 제출 → Firestore 저장 → 성공 모달 |
| 검증 실패 플로우 | 잘못된 입력 → 에러 메시지 표시 |
| 네트워크 에러 플로우 | 오프라인 상태 → 에러 메시지 → 재시도 안내 |

### 5.3 커버리지 목표

- **전체 커버리지**: 85% 이상
- **핵심 로직**: 95% 이상 (registrationService, 폼 검증)
- **UI 컴포넌트**: 80% 이상

---

## 6. 배포 계획

### 6.1 Firebase Hosting 설정

```json
// firebase.json
{
  "hosting": {
    "public": "build",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [
      {
        "source": "**",
        "destination": "/index.html"
      }
    ]
  }
}
```

### 6.2 배포 단계

1. **로컬 빌드 테스트**: `npm run build`
2. **Firebase Emulator 테스트**: `firebase emulators:start`
3. **스테이징 배포**: `firebase deploy --only hosting:staging`
4. **프로덕션 배포**: `firebase deploy --only hosting`

### 6.3 Firestore 보안 규칙

```javascript
// firestore.rules
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // 사전등록 컬렉션: 쓰기만 허용 (읽기 불가)
    match /registrations/{docId} {
      allow create: if true;
      allow read, update, delete: if false;
    }
  }
}
```

---

## 7. 다음 단계

### 7.1 SPEC 완료 후 실행

```
/moai:2-run SPEC-FIREBASE-001
```

### 7.2 관련 SPEC

| SPEC | 관계 | 상태 |
|------|------|------|
| SPEC-UI-001 | 의존 (공통 컴포넌트) | Completed |
| SPEC-AUTH-001 | 후속 (Firebase Auth) | Planned |
| SPEC-ANALYTICS-001 | 후속 (이벤트 추적) | Planned |

---

*계획 버전: 1.0*
*최종 업데이트: 2025-01-17*
