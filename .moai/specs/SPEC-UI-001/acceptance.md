# SPEC-UI-001 인수 조건

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-UI-001 |
| **관련 SPEC** | [spec.md](./spec.md), [plan.md](./plan.md) |
| **Phase** | Phase 1 Week 1-2 |

---

## 1. 인수 조건 (Gherkin 형식)

### 1.1 Button 컴포넌트

#### AC-BTN-01: Primary 버튼 렌더링

```gherkin
Feature: Button Primary Variant

  Scenario: Primary 버튼이 올바르게 렌더링된다
    Given Button 컴포넌트가 variant="primary"로 렌더링되면
    When 화면에 표시될 때
    Then 배경색은 #1A1A1A (--text-primary)이어야 한다
    And 텍스트 색상은 #FFFFFF (--white)이어야 한다
    And 테두리는 1px solid #1A1A1A이어야 한다

  Scenario: Primary 버튼 hover 상태
    Given Primary 버튼이 렌더링되어 있을 때
    When 사용자가 버튼에 마우스를 올리면
    Then 배경색은 transparent로 변경되어야 한다
    And 텍스트 색상은 #1A1A1A로 변경되어야 한다
    And 트랜지션은 0.3s ease로 적용되어야 한다
```

#### AC-BTN-02: CTA 버튼 렌더링

```gherkin
Feature: Button CTA Variant

  Scenario: CTA 버튼이 올바르게 렌더링된다
    Given Button 컴포넌트가 variant="cta"로 렌더링되면
    When 화면에 표시될 때
    Then 배경색은 #8B0000 (--accent-cta)이어야 한다
    And 텍스트 색상은 #FFFFFF이어야 한다

  Scenario: CTA 버튼 hover 상태
    Given CTA 버튼이 렌더링되어 있을 때
    When 사용자가 버튼에 마우스를 올리면
    Then 배경색은 #6B0000 (--accent-cta-hover)으로 변경되어야 한다
```

#### AC-BTN-03: 버튼 disabled 상태

```gherkin
Feature: Button Disabled State

  Scenario: Disabled 버튼은 클릭되지 않는다
    Given Button 컴포넌트가 disabled={true}로 렌더링되면
    When 사용자가 버튼을 클릭하면
    Then onClick 핸들러가 호출되지 않아야 한다
    And 버튼의 opacity는 0.5 이하여야 한다
    And cursor는 not-allowed여야 한다
```

#### AC-BTN-04: 버튼 크기 변형

```gherkin
Feature: Button Size Variants

  Scenario Outline: 버튼 크기가 올바르게 적용된다
    Given Button 컴포넌트가 size="<size>"로 렌더링되면
    When 화면에 표시될 때
    Then 패딩은 "<padding>"이어야 한다
    And 폰트 크기는 "<fontSize>"이어야 한다

    Examples:
      | size | padding    | fontSize   |
      | sm   | 10px 24px  | 0.875rem   |
      | md   | 16px 40px  | 0.9375rem  |
      | lg   | 18px 48px  | 1rem       |
```

---

### 1.2 Card 컴포넌트

#### AC-CRD-01: Basic 카드 렌더링

```gherkin
Feature: Card Basic Variant

  Scenario: Basic 카드가 올바르게 렌더링된다
    Given Card 컴포넌트가 variant="basic"로 렌더링되면
    When 화면에 표시될 때
    Then 배경색은 #FFFFFF (--white)이어야 한다
    And box-shadow는 "0 0 0 1px #D4CFC9"이어야 한다
    And 기본 패딩은 48px이어야 한다
```

#### AC-CRD-02: Frame 카드 렌더링

```gherkin
Feature: Card Frame Variant

  Scenario: Frame 카드가 액자 스타일로 렌더링된다
    Given Card 컴포넌트가 variant="frame"로 렌더링되면
    When 화면에 표시될 때
    Then 배경색은 #F5F3F0 (--bg-secondary)이어야 한다
    And aspect-ratio는 3/4이어야 한다
    And offset shadow "20px 20px 0 0 #E8E4DF"가 적용되어야 한다
```

---

### 1.3 Modal 컴포넌트

#### AC-MDL-01: Modal 열기/닫기

```gherkin
Feature: Modal Open/Close

  Scenario: Modal이 isOpen={true}일 때 표시된다
    Given Modal 컴포넌트가 isOpen={true}로 렌더링되면
    When 화면에 표시될 때
    Then 오버레이가 보여야 한다
    And 오버레이 배경색은 rgba(26, 26, 26, 0.6)이어야 한다
    And 콘텐츠가 중앙에 위치해야 한다

  Scenario: Modal이 isOpen={false}일 때 숨겨진다
    Given Modal 컴포넌트가 isOpen={false}로 렌더링되면
    When 화면에 표시될 때
    Then Modal이 보이지 않아야 한다
    And visibility는 hidden이어야 한다
```

#### AC-MDL-02: Modal 닫기 인터랙션

```gherkin
Feature: Modal Close Interactions

  Scenario: 오버레이 클릭으로 Modal이 닫힌다
    Given Modal이 열려있을 때
    When 사용자가 오버레이(배경) 영역을 클릭하면
    Then onClose 핸들러가 호출되어야 한다

  Scenario: ESC 키로 Modal이 닫힌다
    Given Modal이 열려있을 때
    When 사용자가 ESC 키를 누르면
    Then onClose 핸들러가 호출되어야 한다

  Scenario: Modal 콘텐츠 클릭은 닫기를 트리거하지 않는다
    Given Modal이 열려있을 때
    When 사용자가 Modal 콘텐츠 영역을 클릭하면
    Then onClose 핸들러가 호출되지 않아야 한다
```

#### AC-MDL-03: Modal 애니메이션

```gherkin
Feature: Modal Animation

  Scenario: Modal 열기 애니메이션
    Given Modal이 닫혀있을 때
    When isOpen이 true로 변경되면
    Then 오버레이가 0.3s 동안 페이드인되어야 한다
    And 콘텐츠가 translateY(20px)에서 translateY(0)으로 0.6s 동안 슬라이드되어야 한다
```

#### AC-MDL-04: Modal 배경 스크롤 잠금

```gherkin
Feature: Modal Background Scroll Lock

  Scenario: Modal이 열리면 배경 스크롤이 잠긴다
    Given 페이지가 스크롤 가능한 상태일 때
    When Modal이 열리면
    Then body의 overflow는 hidden이어야 한다

  Scenario: Modal이 닫히면 배경 스크롤이 복원된다
    Given Modal이 열려있을 때
    When Modal이 닫히면
    Then body의 overflow는 원래 상태로 복원되어야 한다
```

---

### 1.4 Header 컴포넌트

#### AC-HDR-01: Header 기본 렌더링

```gherkin
Feature: Header Basic Rendering

  Scenario: Header가 고정 위치로 렌더링된다
    Given Header 컴포넌트가 렌더링되면
    When 화면에 표시될 때
    Then position은 fixed이어야 한다
    And top은 0이어야 한다
    And z-index는 1000이어야 한다
    And 기본 패딩은 20px 0이어야 한다
```

#### AC-HDR-02: Header 스크롤 효과

```gherkin
Feature: Header Scroll Effect

  Scenario: 스크롤 시 Header 스타일이 변경된다
    Given Header가 렌더링되어 있을 때
    When 사용자가 페이지를 50px 이상 스크롤하면
    Then Header 배경색은 rgba(250, 250, 250, 0.95)이어야 한다
    And backdrop-filter: blur(10px)가 적용되어야 한다
    And box-shadow가 추가되어야 한다
    And 패딩은 16px 0으로 줄어야 한다
```

#### AC-HDR-03: 네비게이션 링크 호버 효과

```gherkin
Feature: Navigation Link Hover

  Scenario: 네비게이션 링크 hover 시 밑줄이 나타난다
    Given Header의 네비게이션 링크가 렌더링되어 있을 때
    When 사용자가 링크에 마우스를 올리면
    Then 골드색(#B8860B) 밑줄이 0.3s 동안 왼쪽에서 오른쪽으로 확장되어야 한다
    And 밑줄 높이는 1px이어야 한다
```

---

### 1.5 Footer 컴포넌트

#### AC-FTR-01: Footer 렌더링

```gherkin
Feature: Footer Rendering

  Scenario: Footer가 올바르게 렌더링된다
    Given Footer 컴포넌트가 렌더링되면
    When 화면에 표시될 때
    Then 배경색은 #F5F3F0 (--bg-secondary)이어야 한다
    And 패딩은 48px 이상이어야 한다
    And 텍스트 색상은 #666666 (--text-muted)이어야 한다
```

---

### 1.6 Loading 컴포넌트

#### AC-LDG-01: Loading 스피너

```gherkin
Feature: Loading Spinner

  Scenario: Loading 스피너가 회전 애니메이션을 표시한다
    Given Loading 컴포넌트가 렌더링되면
    When 화면에 표시될 때
    Then 원형 스피너가 보여야 한다
    And 무한 회전 애니메이션이 적용되어야 한다

  Scenario Outline: Loading 크기가 올바르게 적용된다
    Given Loading 컴포넌트가 size="<size>"로 렌더링되면
    When 화면에 표시될 때
    Then 스피너 크기는 <dimension>이어야 한다

    Examples:
      | size | dimension |
      | sm   | 24px      |
      | md   | 40px      |
      | lg   | 64px      |
```

#### AC-LDG-02: Loading 풀스크린 모드

```gherkin
Feature: Loading Fullscreen Mode

  Scenario: fullScreen 모드에서 전체 화면 오버레이가 표시된다
    Given Loading 컴포넌트가 fullScreen={true}로 렌더링되면
    When 화면에 표시될 때
    Then 오버레이가 전체 화면을 덮어야 한다
    And 스피너가 화면 중앙에 위치해야 한다
    And 오버레이 배경은 반투명이어야 한다
```

---

## 2. 전역 품질 기준

### 2.1 디자인 시스템 준수

```gherkin
Feature: Design System Compliance

  Scenario: 모든 컴포넌트가 CSS 변수를 사용한다
    Given 모든 공통 컴포넌트의 CSS 파일을 검사할 때
    When 색상, 간격, 트랜지션 값을 확인하면
    Then 하드코딩된 값이 없어야 한다
    And 모든 값은 var(--variable) 형식이어야 한다

  Scenario: 모든 컴포넌트가 CSS Modules를 사용한다
    Given 모든 공통 컴포넌트를 검사할 때
    When 스타일 적용 방식을 확인하면
    Then 인라인 스타일이 없어야 한다
    And 모든 클래스명은 styles.className 형식이어야 한다
```

### 2.2 반응형 디자인

```gherkin
Feature: Responsive Design

  Scenario Outline: 모든 컴포넌트가 브레이크포인트에서 올바르게 동작한다
    Given 모든 공통 컴포넌트가 렌더링되어 있을 때
    When 뷰포트 너비가 <width>일 때
    Then 레이아웃이 깨지지 않아야 한다
    And 텍스트가 잘리지 않아야 한다
    And 터치 타겟이 최소 44px이어야 한다

    Examples:
      | width  |
      | 1440px |
      | 1024px |
      | 768px  |
      | 480px  |
      | 375px  |
```

### 2.3 접근성

```gherkin
Feature: Accessibility

  Scenario: Button이 키보드로 접근 가능하다
    Given Button 컴포넌트가 렌더링되어 있을 때
    When 사용자가 Tab 키로 포커스를 이동하면
    Then Button이 포커스를 받아야 한다
    And 포커스 표시가 보여야 한다
    And Enter 또는 Space 키로 클릭 가능해야 한다

  Scenario: Modal이 키보드로 조작 가능하다
    Given Modal이 열려있을 때
    When 사용자가 Tab 키를 누르면
    Then 포커스가 Modal 내부에서 순환해야 한다
    And ESC 키로 닫을 수 있어야 한다
```

---

## 3. 완료 정의 (Definition of Done)

### 3.1 코드 품질

- [ ] ESLint 경고 0개
- [ ] Prettier 포맷 적용
- [ ] 불필요한 console.log 제거
- [ ] PropTypes 또는 TypeScript 타입 정의

### 3.2 기능 완성도

- [ ] 모든 variant 구현 완료
- [ ] 모든 size 구현 완료
- [ ] hover, active, disabled 상태 구현
- [ ] 반응형 스타일 적용

### 3.3 테스트

- [ ] 수동 테스트 완료 (모든 브라우저)
- [ ] 반응형 테스트 완료 (모든 브레이크포인트)
- [ ] 키보드 네비게이션 테스트 완료

### 3.4 문서화

- [ ] 각 컴포넌트 Props 주석 작성
- [ ] 사용 예시 코드 작성 (선택적)

---

## 4. 추적성

### SPEC 참조

- **spec.md**: 요구사항 정의 (REQ-* 참조)
- **plan.md**: 구현 계획

### 요구사항 매핑

| 인수 조건 | 관련 요구사항 |
|-----------|---------------|
| AC-BTN-01 | REQ-S-01 |
| AC-BTN-02 | REQ-S-02 |
| AC-BTN-03 | REQ-E-03 |
| AC-BTN-04 | REQ-E-01, REQ-E-02 |
| AC-CRD-01 | REQ-S-04 |
| AC-CRD-02 | REQ-S-04 |
| AC-MDL-01 | REQ-S-05 |
| AC-MDL-02 | REQ-E-05, REQ-E-06 |
| AC-MDL-03 | REQ-E-04, REQ-E-07 |
| AC-MDL-04 | REQ-N-04 |
| AC-HDR-01 | REQ-E-08 |
| AC-HDR-02 | REQ-E-08, REQ-S-06 |
| AC-HDR-03 | REQ-E-09 |
| AC-LDG-01 | REQ-E-10 |

### 태그

```
[SPEC-UI-001] [acceptance.md]
```

---

*인수 조건 버전: 1.0*
*최종 업데이트: 2025-01-17*
