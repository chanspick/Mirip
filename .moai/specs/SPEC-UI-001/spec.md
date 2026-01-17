# SPEC-UI-001: 디자인 시스템 기반 공통 컴포넌트 개발

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-UI-001 |
| **제목** | React 프로젝트 구조 정리 및 디자인 시스템 기반 공통 컴포넌트 개발 |
| **상태** | Planned |
| **우선순위** | High |
| **Phase** | Phase 1 Week 1-2 |
| **생성일** | 2025-01-17 |
| **담당 도메인** | Frontend (UI Components) |

---

## 1. 환경 (Environment)

### 1.1 기술 스택

| 기술 | 버전 | 용도 |
|------|------|------|
| React | 18.x | UI 프레임워크 |
| React Router | 6.x | 클라이언트 라우팅 |
| Create React App | 5.x | 빌드 도구 |
| CSS Modules | - | 컴포넌트 스타일링 |

### 1.2 디자인 시스템 참조

- **디자인 컨셉**: 미술관/갤러리 스타일, 미니멀 & 여백 중심
- **컬러 시스템**: design-system.md 정의 CSS 변수 사용
- **타이포그래피**: Noto Serif KR (제목), Pretendard (본문), Cormorant Garamond (영문), Roboto Mono (숫자)
- **간격 시스템**: 8px 기반 스케일 (xs: 8px ~ section: 120px)

### 1.3 디렉토리 구조

```
my-app/src/
├── components/
│   └── common/
│       ├── Button/
│       │   ├── Button.js
│       │   └── Button.module.css
│       ├── Card/
│       │   ├── Card.js
│       │   └── Card.module.css
│       ├── Modal/
│       │   ├── Modal.js
│       │   └── Modal.module.css
│       ├── Loading/
│       │   ├── Loading.js
│       │   └── Loading.module.css
│       ├── Header/
│       │   ├── Header.js
│       │   └── Header.module.css
│       └── Footer/
│           ├── Footer.js
│           └── Footer.module.css
├── global.css          # CSS 변수 정의
└── index.js
```

---

## 2. 가정 (Assumptions)

### 2.1 기술적 가정

| ID | 가정 | 신뢰도 | 검증 방법 |
|----|------|--------|-----------|
| A-01 | Create React App 5.x가 CSS Modules를 기본 지원한다 | High | CRA 문서 확인 |
| A-02 | Google Fonts에서 필요한 모든 폰트를 로드할 수 있다 | High | 폰트 가용성 확인됨 |
| A-03 | CSS 변수가 모든 타겟 브라우저에서 지원된다 | High | browserslist 설정 기반 |
| A-04 | 컴포넌트는 React 18의 Concurrent Features와 호환되어야 한다 | Medium | 구현 시 테스트 필요 |

### 2.2 비즈니스 가정

| ID | 가정 | 신뢰도 | 틀릴 경우 영향 |
|----|------|--------|----------------|
| B-01 | 모든 공통 컴포넌트는 Phase 2, 3에서도 재사용된다 | High | 컴포넌트 수정 필요 |
| B-02 | 디자인 시스템은 MVP 기간 동안 큰 변경이 없다 | Medium | CSS 변수 수정으로 대응 가능 |

---

## 3. 요구사항 (Requirements) - EARS 형식

### 3.1 전역 요구사항 (Ubiquitous)

| ID | 요구사항 | 검증 방법 |
|----|----------|-----------|
| REQ-U-01 | 시스템은 **항상** design-system.md에 정의된 CSS 변수를 사용해야 한다 | 코드 리뷰 |
| REQ-U-02 | 시스템은 **항상** CSS Modules을 통해 스타일을 캡슐화해야 한다 | 빌드 시 클래스명 해싱 확인 |
| REQ-U-03 | 시스템은 **항상** 반응형 브레이크포인트(1024px, 768px, 480px)를 지원해야 한다 | 반응형 테스트 |
| REQ-U-04 | 시스템은 **항상** 0.3s (fast) 또는 0.6s (smooth) 트랜지션을 사용해야 한다 | 코드 리뷰 |

### 3.2 이벤트 기반 요구사항 (Event-Driven)

#### 3.2.1 Button 컴포넌트

| ID | 요구사항 |
|----|----------|
| REQ-E-01 | **WHEN** 사용자가 Button을 클릭하면 **THEN** onClick 핸들러가 호출되어야 한다 |
| REQ-E-02 | **WHEN** 사용자가 Button에 hover하면 **THEN** 0.3s 트랜지션으로 스타일이 변경되어야 한다 |
| REQ-E-03 | **WHEN** Button이 disabled 상태이면 **THEN** 클릭 이벤트가 무시되어야 한다 |

#### 3.2.2 Modal 컴포넌트

| ID | 요구사항 |
|----|----------|
| REQ-E-04 | **WHEN** Modal이 열리면 **THEN** 배경 오버레이가 0.3s 페이드인되어야 한다 |
| REQ-E-05 | **WHEN** 사용자가 오버레이를 클릭하면 **THEN** Modal이 닫혀야 한다 |
| REQ-E-06 | **WHEN** 사용자가 ESC 키를 누르면 **THEN** Modal이 닫혀야 한다 |
| REQ-E-07 | **WHEN** Modal이 열리면 **THEN** 콘텐츠가 translateY(20px)에서 translateY(0)으로 0.6s 애니메이션되어야 한다 |

#### 3.2.3 Header 컴포넌트

| ID | 요구사항 |
|----|----------|
| REQ-E-08 | **WHEN** 사용자가 스크롤하면 **THEN** Header 배경이 반투명 + blur 효과로 변경되어야 한다 |
| REQ-E-09 | **WHEN** 네비게이션 링크에 hover하면 **THEN** 골드 밑줄이 0.3s로 확장되어야 한다 |

#### 3.2.4 Loading 컴포넌트

| ID | 요구사항 |
|----|----------|
| REQ-E-10 | **WHEN** Loading이 표시되면 **THEN** 스피너가 회전 애니메이션을 시작해야 한다 |

### 3.3 상태 기반 요구사항 (State-Driven)

| ID | 요구사항 |
|----|----------|
| REQ-S-01 | **IF** Button variant가 'primary'이면 **THEN** 검정 배경 + 흰색 텍스트로 렌더링해야 한다 |
| REQ-S-02 | **IF** Button variant가 'cta'이면 **THEN** 딥레드(#8B0000) 배경으로 렌더링해야 한다 |
| REQ-S-03 | **IF** Button variant가 'outline'이면 **THEN** 투명 배경 + 검정 테두리로 렌더링해야 한다 |
| REQ-S-04 | **IF** Card variant가 'frame'이면 **THEN** 액자 스타일 그림자를 적용해야 한다 |
| REQ-S-05 | **IF** Modal isOpen이 false이면 **THEN** Modal이 DOM에서 숨겨져야 한다 |
| REQ-S-06 | **IF** Header가 scrolled 상태이면 **THEN** 패딩이 줄어들고 배경이 적용되어야 한다 |

### 3.4 금지 요구사항 (Unwanted Behavior)

| ID | 요구사항 |
|----|----------|
| REQ-N-01 | 시스템은 인라인 스타일을 **사용하지 않아야 한다** (CSS Modules 사용) |
| REQ-N-02 | 시스템은 하드코딩된 색상값을 **사용하지 않아야 한다** (CSS 변수 사용) |
| REQ-N-03 | 시스템은 px 단위 외의 고정 크기를 제목에 **사용하지 않아야 한다** (clamp 사용) |
| REQ-N-04 | Modal이 열린 상태에서 배경 스크롤이 **허용되지 않아야 한다** |

### 3.5 선택적 요구사항 (Optional)

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| REQ-O-01 | **가능하면** Button에 loading 상태 표시 기능을 제공한다 | Low |
| REQ-O-02 | **가능하면** Card에 hover 시 그림자 확대 효과를 제공한다 | Low |
| REQ-O-03 | **가능하면** Modal에 포커스 트랩 기능을 제공한다 | Medium |

---

## 4. 명세 (Specifications)

### 4.1 Button 컴포넌트

#### Props Interface

```typescript
interface ButtonProps {
  variant?: 'primary' | 'cta' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  fullWidth?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
}
```

#### Variants 스타일 명세

| Variant | Background | Color | Border | Hover |
|---------|------------|-------|--------|-------|
| primary | --text-primary | --white | --text-primary | 반전 (투명 + 검정) |
| cta | --accent-cta | --white | --accent-cta | --accent-cta-hover |
| outline | transparent | --text-primary | --text-primary | 반전 (검정 + 흰색) |

#### Sizes 명세

| Size | Padding | Font Size |
|------|---------|-----------|
| sm | 10px 24px | 0.875rem |
| md | 16px 40px | 0.9375rem |
| lg | 18px 48px | 1rem |

### 4.2 Card 컴포넌트

#### Props Interface

```typescript
interface CardProps {
  variant?: 'basic' | 'frame';
  padding?: 'sm' | 'md' | 'lg';
  shadow?: boolean;
  children: React.ReactNode;
}
```

#### Variants 명세

| Variant | Background | Shadow | 특징 |
|---------|------------|--------|------|
| basic | --white | 1px solid --border-light | 기본 카드 |
| frame | --bg-secondary | offset shadow (20px 20px) | 액자 스타일, aspect-ratio: 3/4 |

### 4.3 Modal 컴포넌트

#### Props Interface

```typescript
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
}
```

#### 스타일 명세

- **Overlay**: rgba(26, 26, 26, 0.6), z-index: 2000
- **Content**: --white 배경, 60px 패딩, max-width: 400px
- **Animation**:
  - 오버레이: opacity 0.3s
  - 콘텐츠: translateY 0.6s cubic-bezier(0.4, 0, 0.2, 1)

### 4.4 Header 컴포넌트

#### Props Interface

```typescript
interface HeaderProps {
  logo?: React.ReactNode;
  navItems?: Array<{ label: string; href: string }>;
  ctaButton?: { label: string; onClick: () => void };
}
```

#### 스타일 명세

- **Position**: fixed, top: 0, z-index: 1000
- **Padding**: 기본 20px, scrolled 시 16px
- **Scrolled 상태**:
  - Background: rgba(250, 250, 250, 0.95)
  - backdrop-filter: blur(10px)
  - box-shadow: 0 1px 0 --border-light

### 4.5 Footer 컴포넌트

#### Props Interface

```typescript
interface FooterProps {
  links?: Array<{ label: string; href: string }>;
  copyright?: string;
  socialLinks?: Array<{ icon: React.ReactNode; href: string }>;
}
```

#### 스타일 명세

- **Background**: --bg-secondary
- **Padding**: var(--spacing-xl) 0
- **Typography**: --font-sans, --text-muted

### 4.6 Loading 컴포넌트

#### Props Interface

```typescript
interface LoadingProps {
  size?: 'sm' | 'md' | 'lg';
  color?: string;
  fullScreen?: boolean;
}
```

#### Sizes 명세

| Size | Width/Height |
|------|--------------|
| sm | 24px |
| md | 40px |
| lg | 64px |

---

## 5. 추적성 (Traceability)

### 5.1 관련 문서

- `.moai/project/product.md` - 서비스 정의 및 디자인 철학
- `.moai/project/structure.md` - 디렉토리 구조 및 컴포넌트 정의
- `.moai/project/tech.md` - 기술 스택 정보
- `.moai/project/design-system.md` - CSS 변수 및 스타일 가이드

### 5.2 SPEC 태그

```
[SPEC-UI-001] Common Components Development
├── [UI-001-BTN] Button Component
├── [UI-001-CRD] Card Component
├── [UI-001-MDL] Modal Component
├── [UI-001-HDR] Header Component
├── [UI-001-FTR] Footer Component
└── [UI-001-LDG] Loading Component
```

---

*SPEC 버전: 1.0*
*최종 업데이트: 2025-01-17*
