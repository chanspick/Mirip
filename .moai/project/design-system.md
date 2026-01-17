# MIRIP 디자인 시스템

> 미술관/갤러리 컨셉의 미니멀하고 고급스러운 디자인 시스템

---

## 디자인 철학

### 핵심 컨셉

- **미술관/갤러리** 감성
- **미니멀 & 여백** 중심
- **작품이 돋보이는** 레이아웃
- **세리프 폰트**로 고급스러움
- **얇은 라인, 섬세한 디테일**

### 디자인 원칙

1. **여백의 미**: 콘텐츠 주변에 충분한 공간 확보
2. **절제된 색상**: 모노톤 + 골드 포인트
3. **타이포그래피 위계**: 세리프(제목) + 산세리프(본문)
4. **미세한 움직임**: 부드럽고 우아한 트랜지션

---

## CSS 변수 (Design Tokens)

### 색상 (Colors)

```css
:root {
    /* 배경 (Background) */
    --bg-primary: #FAFAFA;        /* 주요 배경 - 오프화이트 */
    --bg-secondary: #F5F3F0;      /* 보조 배경 - 웜그레이 */
    --bg-warm: #E8E4DF;           /* 강조 배경 - 웜베이지 */
    --white: #FFFFFF;             /* 카드, 모달 배경 */

    /* 텍스트 (Text) */
    --text-primary: #1A1A1A;      /* 제목, 강조 텍스트 */
    --text-secondary: #333333;    /* 본문 텍스트 */
    --text-muted: #666666;        /* 보조 텍스트, 캡션 */

    /* 포인트 (Accent) */
    --accent-gold: #B8860B;       /* 주요 포인트 - 골드 */
    --accent-bronze: #8B7355;     /* 보조 포인트 - 브론즈 */

    /* CTA (Call to Action) */
    --accent-cta: #8B0000;        /* CTA 버튼 - 딥레드 */
    --accent-cta-hover: #6B0000;  /* CTA 호버 */

    /* 테두리 (Border) */
    --border-light: #D4CFC9;      /* 얇은 테두리, 구분선 */

    /* ===== 추가: AI 진단 관련 색상 ===== */

    /* 티어 컬러 (Tier Colors) */
    --tier-s: #8b5cf6;            /* S티어 - 퍼플 (서울대급) */
    --tier-a: #3b82f6;            /* A티어 - 블루 (홍대급) */
    --tier-b: #22c55e;            /* B티어 - 그린 (국민대급) */
    --tier-c: #6b7280;            /* C티어 - 그레이 (지방대급) */

    /* 루브릭 컬러 (Rubric Colors) */
    --rubric-composition: #f59e0b;    /* 구성력 - 오렌지 */
    --rubric-tone: #8b5cf6;           /* 명암/질감 - 퍼플 */
    --rubric-form: #3b82f6;           /* 조형완성도 - 블루 */
    --rubric-theme: #ec4899;          /* 주제해석 - 핑크 */

    /* 피드백 컬러 (Feedback Colors) */
    --feedback-strength: #22c55e;     /* 강점 - 그린 */
    --feedback-improve: #f59e0b;      /* 개선점 - 오렌지 */
}
```

### 색상 사용 가이드

| 용도 | 색상 | 코드 |
|-----|------|------|
| 주요 배경 | 오프화이트 | `--bg-primary` |
| 섹션 구분 배경 | 웜그레이 | `--bg-secondary` |
| 카드/모달 배경 | 순백 | `--white` |
| 제목 텍스트 | 차콜 블랙 | `--text-primary` |
| 본문 텍스트 | 다크 그레이 | `--text-secondary` |
| 캡션/보조 텍스트 | 미디엄 그레이 | `--text-muted` |
| 강조/라벨 | 골드 | `--accent-gold` |
| 보조 강조 | 브론즈 | `--accent-bronze` |
| CTA 버튼 | 딥 레드 | `--accent-cta` |
| 테두리/구분선 | 라이트 베이지 | `--border-light` |

### 티어 색상 사용 가이드

| 티어 | 설명 | 색상 코드 | 사용처 |
|------|------|----------|--------|
| S | 서울대급 | `--tier-s` (#8b5cf6) | 티어 배지, 차트 |
| A | 홍대급 | `--tier-a` (#3b82f6) | 티어 배지, 차트 |
| B | 국민대급 | `--tier-b` (#22c55e) | 티어 배지, 차트 |
| C | 지방대급 | `--tier-c` (#6b7280) | 티어 배지, 차트 |

### 루브릭 색상 사용 가이드

| 루브릭 | 설명 | 색상 코드 | 사용처 |
|--------|------|----------|--------|
| 구성력 | 화면 배치, 비례, 균형 | `--rubric-composition` (#f59e0b) | 점수 차트, 프로그레스 바 |
| 명암/질감 | 톤 표현, 재질감, 필압 | `--rubric-tone` (#8b5cf6) | 점수 차트, 프로그레스 바 |
| 조형완성도 | 형태 정확성, 마감 | `--rubric-form` (#3b82f6) | 점수 차트, 프로그레스 바 |
| 주제해석 | 주제와의 연관성, 창의성 | `--rubric-theme` (#ec4899) | 점수 차트, 프로그레스 바 |

### 피드백 색상 사용 가이드

| 유형 | 설명 | 색상 코드 | 사용처 |
|------|------|----------|--------|
| 강점 | 잘한 점 | `--feedback-strength` (#22c55e) | 피드백 카드, 아이콘 |
| 개선점 | 보완할 점 | `--feedback-improve` (#f59e0b) | 피드백 카드, 아이콘 |

---

## 타이포그래피 (Typography)

### 폰트 패밀리

```css
:root {
    /* 한글 제목 - 세리프 */
    --font-serif-kr: 'Noto Serif KR', serif;

    /* 한글/영문 본문 - 산세리프 */
    --font-sans: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;

    /* 영문 포인트 - 세리프 */
    --font-serif-en: 'Cormorant Garamond', serif;

    /* 숫자/점수 - 모노스페이스 */
    --font-mono: 'Roboto Mono', monospace;
}
```

### 폰트 로드

```html
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Noto+Serif+KR:wght@400;500;600;700&family=Pretendard:wght@300;400;500;600&family=Roboto+Mono:wght@400;500;600&display=swap" rel="stylesheet">
```

### 타이포그래피 스케일

| 요소 | 폰트 | 크기 | 두께 | 행간 | 자간 |
|-----|------|------|------|------|------|
| Hero 제목 | Noto Serif KR | `clamp(2.5rem, 5vw, 3.5rem)` | 600 | 1.3 | -0.02em |
| 섹션 제목 | Noto Serif KR | `clamp(1.75rem, 3vw, 2.5rem)` | 600 | 1.3 | -0.02em |
| 카드 제목 | Noto Serif KR | 1.25rem | 600 | 1.4 | - |
| 본문 | Pretendard | 1rem | 400 | 1.7 | - |
| 부제목 | Pretendard | 1.125rem | 400 | 1.8 | - |
| 캡션 | Pretendard | 0.875rem | 400 | 1.6 | - |
| 라벨 | Cormorant Garamond | 0.875rem | 400 | - | 0.2em |
| 로고 | Cormorant Garamond | 1.75rem | 600 | - | 0.1em |
| **점수/숫자** | Roboto Mono | 1.5rem+ | 500-600 | - | 0.05em |

### 숫자/점수 타이포그래피

AI 진단 결과의 점수 및 숫자 표시에는 `Roboto Mono` 폰트를 사용합니다.

```css
/* 점수 표시 - 큰 숫자 */
.score-large {
    font-family: var(--font-mono);
    font-size: 3rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* 점수 표시 - 중간 숫자 */
.score-medium {
    font-family: var(--font-mono);
    font-size: 1.5rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* 확률/퍼센트 */
.percentage {
    font-family: var(--font-mono);
    font-size: 1rem;
    font-weight: 400;
}
```

### 타이포그래피 예시

```css
/* Hero 제목 */
.hero-title {
    font-family: var(--font-serif-kr);
    font-size: clamp(2.5rem, 5vw, 3.5rem);
    font-weight: 600;
    line-height: 1.3;
    letter-spacing: -0.02em;
}

/* 섹션 제목 */
.section-title {
    font-family: var(--font-serif-kr);
    font-size: clamp(1.75rem, 3vw, 2.5rem);
    font-weight: 600;
    letter-spacing: -0.02em;
}

/* 섹션 라벨 (영문) */
.section-label {
    font-family: var(--font-serif-en);
    font-size: 0.875rem;
    font-weight: 400;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent-gold);
}

/* 본문 */
body {
    font-family: var(--font-sans);
    font-size: 1rem;
    line-height: 1.7;
}
```

---

## 간격 시스템 (Spacing)

### 간격 변수

```css
:root {
    /* 8px 기반 스케일 */
    --spacing-xs: 8px;
    --spacing-sm: 16px;
    --spacing-md: 24px;
    --spacing-lg: 32px;
    --spacing-xl: 48px;
    --spacing-2xl: 64px;
    --spacing-3xl: 80px;
    --spacing-section: 120px;       /* 섹션 패딩 */

    /* 컨테이너 */
    --container-max: 1200px;
    --container-padding: 24px;
}
```

### 간격 스케일

| 이름 | 값 | 용도 |
|-----|-----|------|
| xs | 8px | 인라인 요소 간격 |
| sm | 16px | 텍스트 간격 |
| md | 24px | 요소 내부 패딩 |
| lg | 32px | 컴포넌트 간격 |
| xl | 48px | 그룹 간격 |
| 2xl | 64px | 섹션 내 그룹 |
| 3xl | 80px | 그리드 갭 |
| section | 120px | 섹션 간 간격 |

### 컨테이너

```css
.container {
    max-width: var(--container-max);
    margin: 0 auto;
    padding: 0 24px;
}
```

---

## 트랜지션 (Transitions)

### 트랜지션 변수

```css
:root {
    /* 부드러운 트랜지션 (긴 애니메이션) */
    --transition-smooth: 0.6s cubic-bezier(0.4, 0, 0.2, 1);

    /* 빠른 트랜지션 (인터랙션) */
    --transition-fast: 0.3s ease;
}
```

### 애니메이션 패턴

#### Fade In (스크롤 트리거)

```css
.fade-in {
    opacity: 0;
    transform: translateY(30px);
    transition: opacity 0.8s var(--transition-smooth),
                transform 0.8s var(--transition-smooth);
}

.fade-in.visible {
    opacity: 1;
    transform: translateY(0);
}

/* 딜레이 클래스 */
.delay-1 { transition-delay: 0.1s; }
.delay-2 { transition-delay: 0.2s; }
.delay-3 { transition-delay: 0.3s; }
.delay-4 { transition-delay: 0.4s; }
```

#### 스크롤 인디케이터 펄스

```css
@keyframes scrollPulse {
    0%, 100% { opacity: 1; height: 60px; }
    50% { opacity: 0.5; height: 40px; }
}

.scroll-line {
    animation: scrollPulse 2s ease-in-out infinite;
}
```

---

## 컴포넌트 패턴

### 버튼 (Buttons)

#### Primary Button

```css
.btn {
    display: inline-block;
    font-family: var(--font-sans);
    font-size: 0.9375rem;
    font-weight: 500;
    text-decoration: none;
    letter-spacing: 0.02em;
    transition: var(--transition-fast);
    cursor: pointer;
}

.btn-primary {
    padding: 16px 40px;
    background-color: var(--text-primary);
    color: var(--white);
    border: 1px solid var(--text-primary);
}

.btn-primary:hover {
    background-color: transparent;
    color: var(--text-primary);
}
```

#### CTA Button (Submit)

```css
.btn-submit {
    width: 100%;
    padding: 16px;
    background-color: var(--accent-cta);
    color: var(--white);
    border: 1px solid var(--accent-cta);
    font-size: 1rem;
    font-weight: 500;
}

.btn-submit:hover {
    background-color: var(--accent-cta-hover);
    border-color: var(--accent-cta-hover);
}
```

#### Outline Button (Nav CTA)

```css
.nav-cta {
    padding: 10px 24px;
    border: 1px solid var(--text-primary);
    transition: var(--transition-fast);
}

.nav-cta:hover {
    background-color: var(--text-primary);
    color: var(--white);
}
```

### 카드 (Cards)

#### 기본 카드

```css
.card {
    background-color: var(--white);
    padding: 48px;
    box-shadow: 0 0 0 1px var(--border-light);
}
```

#### 프레임 카드 (작품 프레임)

```css
.frame {
    position: relative;
    aspect-ratio: 3/4;
    background-color: var(--bg-secondary);
    box-shadow:
        0 0 0 1px var(--border-light),
        20px 20px 0 0 var(--bg-warm);
}
```

#### 티어 배지 카드

```css
.tier-badge {
    display: inline-flex;
    align-items: center;
    padding: 8px 16px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-weight: 600;
}

.tier-badge--s { background-color: var(--tier-s); color: white; }
.tier-badge--a { background-color: var(--tier-a); color: white; }
.tier-badge--b { background-color: var(--tier-b); color: white; }
.tier-badge--c { background-color: var(--tier-c); color: white; }
```

### 폼 요소 (Form Elements)

#### Input / Select

```css
.form-group input,
.form-group select {
    width: 100%;
    padding: 14px 16px;
    font-family: var(--font-sans);
    font-size: 1rem;
    border: 1px solid var(--border-light);
    background-color: var(--bg-primary);
    color: var(--text-primary);
    transition: var(--transition-fast);
}

.form-group input:focus,
.form-group select:focus {
    outline: none;
    border-color: var(--accent-gold);
}

.form-group input::placeholder {
    color: var(--text-muted);
}
```

#### Label

```css
.form-group label {
    display: block;
    font-size: 0.875rem;
    font-weight: 500;
    margin-bottom: 8px;
    color: var(--text-primary);
}
```

### 모달 (Modal)

```css
.modal {
    position: fixed;
    inset: 0;
    background-color: rgba(26, 26, 26, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2000;
    opacity: 0;
    visibility: hidden;
    transition: var(--transition-fast);
}

.modal.active {
    opacity: 1;
    visibility: visible;
}

.modal-content {
    background-color: var(--white);
    padding: 60px;
    max-width: 400px;
    width: 90%;
    text-align: center;
    transform: translateY(20px);
    transition: var(--transition-smooth);
}

.modal.active .modal-content {
    transform: translateY(0);
}
```

### 네비게이션 (Navigation)

#### 고정 네비게이션

```css
.nav {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    padding: 20px 0;
    transition: var(--transition-fast);
}

.nav.scrolled {
    background-color: rgba(250, 250, 250, 0.95);
    backdrop-filter: blur(10px);
    box-shadow: 0 1px 0 var(--border-light);
    padding: 16px 0;
}
```

#### 네비게이션 링크 (밑줄 효과)

```css
.nav-link {
    position: relative;
    font-family: var(--font-sans);
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-decoration: none;
    letter-spacing: 0.02em;
    transition: var(--transition-fast);
}

.nav-link::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 0;
    width: 0;
    height: 1px;
    background-color: var(--accent-gold);
    transition: var(--transition-fast);
}

.nav-link:hover::after {
    width: 100%;
}
```

---

## 레이아웃 패턴

### 그리드 시스템

#### 2컬럼 그리드

```css
.grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 80px;
    align-items: center;
}
```

#### 3컬럼 그리드

```css
.grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 60px;
}
```

### 섹션 구조

```css
.section {
    padding: var(--section-padding) 0;
}

.section-header {
    text-align: center;
    margin-bottom: 80px;
}
```

---

## 반응형 브레이크포인트

### 브레이크포인트

| 이름 | 값 | 용도 |
|-----|-----|------|
| Wide | 1440px+ | 넓은 모니터 |
| Desktop | 1200px | 기본 데스크톱 |
| Laptop | 1024px | 태블릿 가로 |
| Tablet | 768px | 태블릿 세로 |
| Mobile | 480px | 모바일 |

### 반응형 변수 조정

```css
@media (max-width: 1024px) {
    :root {
        --section-padding: 100px;
    }
}

@media (max-width: 768px) {
    :root {
        --section-padding: 80px;
    }
}

@media (max-width: 480px) {
    html {
        font-size: 15px;
    }
}
```

### 그리드 반응형

```css
/* 태블릿 */
@media (max-width: 1024px) {
    .grid-2 {
        grid-template-columns: 1fr;
        gap: 60px;
    }
}

/* 모바일 */
@media (max-width: 768px) {
    .grid-3 {
        grid-template-columns: 1fr;
        gap: 40px;
    }
}
```

---

## React 컴포넌트 변환 가이드

### CSS Variables를 global.css에 추가

```css
/* src/global.css */
:root {
    /* 위의 모든 CSS 변수 복사 */
}
```

### CSS Module 패턴

```css
/* Component.module.css */
.title {
    font-family: var(--font-serif-kr);
    font-size: clamp(1.75rem, 3vw, 2.5rem);
    font-weight: 600;
    letter-spacing: -0.02em;
}
```

### 컴포넌트 구조 예시

```jsx
// Button.js
import styles from './Button.module.css';

export const Button = ({ variant = 'primary', children, ...props }) => {
    return (
        <button
            className={`${styles.btn} ${styles[variant]}`}
            {...props}
        >
            {children}
        </button>
    );
};
```

---

## 디자인 체크리스트

### 색상

- [ ] 배경은 `--bg-primary` 또는 `--bg-secondary` 사용
- [ ] 텍스트는 `--text-primary`, `--text-secondary`, `--text-muted` 위계 준수
- [ ] 포인트 색상은 `--accent-gold`로 제한적 사용
- [ ] CTA 버튼에만 `--accent-cta` 사용
- [ ] 티어 표시는 `--tier-*` 색상 사용
- [ ] 루브릭 차트는 `--rubric-*` 색상 사용
- [ ] 피드백은 `--feedback-*` 색상 사용

### 타이포그래피

- [ ] 제목은 Noto Serif KR 세리프 폰트
- [ ] 본문은 Pretendard 산세리프 폰트
- [ ] 영문 라벨은 Cormorant Garamond 사용
- [ ] **점수/숫자는 Roboto Mono 사용**
- [ ] 적절한 행간(1.7 이상) 유지

### 간격

- [ ] 섹션 간 120px (반응형: 100px -> 80px)
- [ ] 컴포넌트 간격 24px 이상
- [ ] 충분한 여백 확보

### 애니메이션

- [ ] 트랜지션은 0.3s 또는 0.6s
- [ ] ease 또는 cubic-bezier(0.4, 0, 0.2, 1)
- [ ] 스크롤 기반 fade-in 효과

---

*문서 버전: 2.1*
*최종 업데이트: 2025년 1월*
*기반: landing/styles.css 분석 + requirements.md Section 5*
