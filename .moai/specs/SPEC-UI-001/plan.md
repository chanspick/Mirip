# SPEC-UI-001 구현 계획

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-UI-001 |
| **관련 SPEC** | [spec.md](./spec.md) |
| **Phase** | Phase 1 Week 1-2 |

---

## 1. 마일스톤 (우선순위 기반)

### Primary Goal: 프로젝트 기반 설정

**범위**: React 프로젝트 구조 정리 및 디자인 시스템 CSS 변수 설정

| 작업 | 설명 | 산출물 |
|------|------|--------|
| 디렉토리 구조 생성 | components/common 하위 폴더 생성 | 6개 컴포넌트 폴더 |
| global.css 설정 | CSS 변수 정의 (색상, 타이포그래피, 간격, 트랜지션) | global.css |
| 폰트 로드 설정 | Google Fonts 링크 추가 | index.html 수정 |

### Secondary Goal: 핵심 컴포넌트 개발

**범위**: Button, Card, Loading 컴포넌트 구현

| 작업 | 설명 | 의존성 |
|------|------|--------|
| Button 컴포넌트 | 3가지 variant (primary, cta, outline), 3가지 size | global.css |
| Card 컴포넌트 | 2가지 variant (basic, frame) | global.css |
| Loading 컴포넌트 | 스피너 애니메이션, 3가지 size | global.css |

### Tertiary Goal: 레이아웃 컴포넌트 개발

**범위**: Header, Footer, Modal 컴포넌트 구현

| 작업 | 설명 | 의존성 |
|------|------|--------|
| Header 컴포넌트 | 고정 네비게이션, 스크롤 감지, 모바일 대응 | Button |
| Footer 컴포넌트 | 링크 그룹, 저작권 표시 | - |
| Modal 컴포넌트 | 오버레이, 애니메이션, ESC 키 처리 | Button |

### Final Goal: 통합 및 검증

**범위**: 컴포넌트 통합 테스트 및 문서화

| 작업 | 설명 | 의존성 |
|------|------|--------|
| 스토리북/데모 페이지 | 모든 컴포넌트 상태 확인 | 모든 컴포넌트 |
| 반응형 테스트 | 1024px, 768px, 480px 브레이크포인트 | 모든 컴포넌트 |
| 접근성 검증 | 키보드 네비게이션, ARIA 속성 | Modal, Button |

---

## 2. 기술 접근 방식

### 2.1 CSS Modules 패턴

```jsx
// 컴포넌트 파일 구조
import styles from './Button.module.css';

export const Button = ({ variant = 'primary', size = 'md', ...props }) => {
  const classNames = [
    styles.button,
    styles[variant],
    styles[size]
  ].join(' ');

  return <button className={classNames} {...props} />;
};
```

### 2.2 CSS 변수 활용 패턴

```css
/* global.css에서 정의 */
:root {
  --bg-primary: #FAFAFA;
  --text-primary: #1A1A1A;
  --accent-cta: #8B0000;
  --transition-fast: 0.3s ease;
}

/* 컴포넌트에서 사용 */
.button {
  background-color: var(--text-primary);
  color: var(--white);
  transition: var(--transition-fast);
}
```

### 2.3 반응형 설계 패턴

```css
/* 모바일 퍼스트 접근 */
.container {
  padding: 16px;
}

@media (min-width: 768px) {
  .container {
    padding: 24px;
  }
}

@media (min-width: 1024px) {
  .container {
    max-width: 1200px;
    margin: 0 auto;
  }
}
```

### 2.4 애니메이션 패턴

```css
/* 트랜지션 변수 사용 */
.modal {
  opacity: 0;
  visibility: hidden;
  transition: opacity var(--transition-fast);
}

.modal.active {
  opacity: 1;
  visibility: visible;
}

/* 콘텐츠 슬라이드 인 */
.modalContent {
  transform: translateY(20px);
  transition: transform var(--transition-smooth);
}

.modal.active .modalContent {
  transform: translateY(0);
}
```

---

## 3. 아키텍처 설계

### 3.1 컴포넌트 계층 구조

```
공통 컴포넌트 (components/common/)
├── 기본 요소 (Primitives)
│   ├── Button - 인터랙션 기본 단위
│   ├── Card - 콘텐츠 컨테이너
│   └── Loading - 상태 표시
│
└── 레이아웃 요소 (Layout)
    ├── Header - 페이지 상단 네비게이션
    ├── Footer - 페이지 하단 정보
    └── Modal - 오버레이 다이얼로그
```

### 3.2 Props 설계 원칙

1. **명시적 Variant**: 스타일 변형은 variant prop으로 관리
2. **Size Scale**: sm, md, lg 3단계 크기 스케일 통일
3. **Composition**: children prop을 통한 유연한 구성
4. **Controlled/Uncontrolled**: Modal은 isOpen으로 제어형 설계

### 3.3 스타일 설계 원칙

1. **CSS 변수 우선**: 모든 색상, 간격, 트랜지션은 변수 참조
2. **BEM 네이밍**: CSS Modules 내에서 .component, .componentVariant 패턴
3. **반응형 분리**: 미디어 쿼리는 컴포넌트 CSS 파일 하단에 정의

---

## 4. 의존성 및 제약사항

### 4.1 외부 의존성

| 의존성 | 버전 | 용도 |
|--------|------|------|
| react | ^18.3.1 | UI 프레임워크 |
| react-dom | ^18.3.1 | DOM 렌더링 |
| react-router-dom | ^6.2.1 | 라우팅 (Header 링크용) |

### 4.2 기술 제약사항

| 제약 | 설명 | 대응 |
|------|------|------|
| CRA 빌드 시스템 | Webpack 설정 직접 수정 불가 | CSS Modules 기본 지원 활용 |
| 브라우저 지원 | Chrome, Firefox, Safari, Edge 최근 2버전 | CSS 변수 사용 가능 |
| 폰트 로딩 | 외부 Google Fonts 의존 | preconnect로 성능 최적화 |

### 4.3 성능 고려사항

| 항목 | 목표 | 방법 |
|------|------|------|
| 번들 크기 | 공통 컴포넌트 < 50KB | Tree shaking, 코드 분할 |
| 초기 렌더링 | FCP < 1.5s | 폰트 preload, CSS 인라인 |
| 인터랙션 | INP < 200ms | 이벤트 핸들러 최적화 |

---

## 5. 위험 요소 및 대응

### 5.1 기술 위험

| 위험 | 확률 | 영향 | 대응 |
|------|------|------|------|
| 폰트 로딩 지연 | Medium | 레이아웃 시프트 | font-display: swap 사용 |
| CSS 변수 미지원 | Low | 스타일 깨짐 | browserslist로 타겟 제한 |
| 모바일 터치 이슈 | Medium | UX 저하 | 터치 타겟 최소 44px 확보 |

### 5.2 프로젝트 위험

| 위험 | 확률 | 영향 | 대응 |
|------|------|------|------|
| 디자인 변경 | Medium | 컴포넌트 수정 필요 | CSS 변수로 변경 용이성 확보 |
| 범위 확장 | Low | 일정 지연 | 핵심 컴포넌트 우선 완료 |

---

## 6. 작업 분해 구조 (WBS)

### 6.1 Primary Goal 상세

```
1. 프로젝트 기반 설정
   1.1 디렉토리 구조
       - [ ] components/common/Button 폴더 생성
       - [ ] components/common/Card 폴더 생성
       - [ ] components/common/Modal 폴더 생성
       - [ ] components/common/Header 폴더 생성
       - [ ] components/common/Footer 폴더 생성
       - [ ] components/common/Loading 폴더 생성

   1.2 global.css 설정
       - [ ] 색상 변수 정의 (배경, 텍스트, 포인트, 티어, 루브릭)
       - [ ] 타이포그래피 변수 정의 (폰트 패밀리, 크기)
       - [ ] 간격 변수 정의 (spacing-xs ~ section)
       - [ ] 트랜지션 변수 정의 (fast, smooth)

   1.3 폰트 로드
       - [ ] index.html에 Google Fonts 링크 추가
       - [ ] preconnect 최적화 적용
```

### 6.2 Secondary Goal 상세

```
2. 핵심 컴포넌트 개발
   2.1 Button 컴포넌트
       - [ ] Button.js 작성 (variant, size, disabled, onClick props)
       - [ ] Button.module.css 작성 (primary, cta, outline, sm, md, lg)
       - [ ] hover, active, disabled 상태 스타일
       - [ ] 반응형 스타일 (터치 타겟 확보)

   2.2 Card 컴포넌트
       - [ ] Card.js 작성 (variant, padding, shadow props)
       - [ ] Card.module.css 작성 (basic, frame 변형)
       - [ ] frame 변형의 offset shadow 효과

   2.3 Loading 컴포넌트
       - [ ] Loading.js 작성 (size, color, fullScreen props)
       - [ ] Loading.module.css 작성 (스피너 회전 애니메이션)
       - [ ] 풀스크린 오버레이 모드
```

### 6.3 Tertiary Goal 상세

```
3. 레이아웃 컴포넌트 개발
   3.1 Header 컴포넌트
       - [ ] Header.js 작성 (logo, navItems, ctaButton props)
       - [ ] 스크롤 감지 훅 구현 (useScrollPosition)
       - [ ] Header.module.css 작성 (기본, scrolled 상태)
       - [ ] 네비게이션 링크 hover 밑줄 효과
       - [ ] 모바일 햄버거 메뉴 (선택적)

   3.2 Footer 컴포넌트
       - [ ] Footer.js 작성 (links, copyright, socialLinks props)
       - [ ] Footer.module.css 작성
       - [ ] 반응형 레이아웃 (그리드 → 스택)

   3.3 Modal 컴포넌트
       - [ ] Modal.js 작성 (isOpen, onClose, title props)
       - [ ] Modal.module.css 작성 (오버레이, 콘텐츠 애니메이션)
       - [ ] ESC 키 닫기 이벤트 핸들러
       - [ ] 오버레이 클릭 닫기 구현
       - [ ] 배경 스크롤 잠금 (body overflow)
```

---

## 7. 예상 산출물

### 7.1 파일 목록

```
my-app/src/
├── global.css (수정)
├── index.html (수정 - 폰트 로드)
└── components/common/
    ├── Button/
    │   ├── Button.js
    │   ├── Button.module.css
    │   └── index.js
    ├── Card/
    │   ├── Card.js
    │   ├── Card.module.css
    │   └── index.js
    ├── Modal/
    │   ├── Modal.js
    │   ├── Modal.module.css
    │   └── index.js
    ├── Header/
    │   ├── Header.js
    │   ├── Header.module.css
    │   └── index.js
    ├── Footer/
    │   ├── Footer.js
    │   ├── Footer.module.css
    │   └── index.js
    └── Loading/
        ├── Loading.js
        ├── Loading.module.css
        └── index.js
```

### 7.2 품질 기준

| 항목 | 기준 |
|------|------|
| 린트 | ESLint 경고 0개 |
| 스타일 | Prettier 포맷 적용 |
| 반응형 | 모든 브레이크포인트 정상 동작 |
| 접근성 | 키보드 네비게이션 가능 |

---

## 8. 추적성

### SPEC 참조

- **spec.md**: 요구사항 정의
- **acceptance.md**: 인수 조건

### 태그

```
[SPEC-UI-001] [plan.md]
```

---

*계획 버전: 1.0*
*최종 업데이트: 2025-01-17*
