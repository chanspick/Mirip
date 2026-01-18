# SPEC-DATA-001: 인수 조건

## TAG BLOCK

```yaml
spec_id: SPEC-DATA-001
title: AI 진단 데이터 파이프라인 - 인수 조건
status: Completed
created: 2025-01-18
updated: 2026-01-18
verified: 2026-01-18
related_spec: SPEC-DATA-001/spec.md
```

---

## 1. 인수 조건 개요

### 1.1 Definition of Done (완료 정의)

SPEC-DATA-001은 다음 조건이 모두 충족될 때 완료로 간주합니다:

- [ ] 2,000개 이상 이미지 데이터셋 구축 완료 (데이터 수집 단계에서 수행 예정)
- [ ] 모든 이미지에 대한 메타데이터 태깅 완료 (데이터 수집 단계에서 수행 예정)
- [ ] 자동 라벨링 적용 및 검증 완료 (데이터 수집 단계에서 수행 예정)
- [x] 품질 게이트 모든 항목 통과 (2026-01-18 검증 완료 - 코드 품질)
- [x] 테스트 커버리지 85% 이상 (실제: 90%, 2026-01-18 검증 완료)

**파이프라인 코드 구현 완료** (2026-01-18):
- [x] 데이터 수집기 구현 완료
- [x] 전처리기 구현 완료
- [x] 메타데이터 태거 구현 완료
- [x] 자동 라벨러 구현 완료
- [x] 스토리지 구현 완료
- [x] 통합 파이프라인 구현 완료
- [x] 단위 테스트 168개 통과

### 1.2 인수 조건 요약

| 영역 | 핵심 기준 | 목표값 |
|------|-----------|--------|
| 데이터 수집 | 이미지 수량 | >= 2,000개 |
| 전처리 | 품질 통과율 | >= 95% |
| 태깅 | 완료율 | 100% |
| 라벨링 | 정확도 | >= 85% |
| 코드 품질 | 테스트 커버리지 | >= 85% |

---

## 2. 데이터 수집 인수 조건

### AC-COL-001: 이미지 수량 목표

```gherkin
Feature: 이미지 데이터 수집
  데이터 파이프라인은 AI 학습을 위해 충분한 양의 이미지를 수집해야 한다

  Scenario: 목표 이미지 수량 달성
    Given 데이터 수집 파이프라인이 구성되어 있다
    When 모든 수집 소스에서 이미지 수집을 완료한다
    Then 전체 이미지 수량은 2,000개 이상이어야 한다
    And 유효한 이미지만 카운트에 포함되어야 한다

  Scenario: 과별 최소 분포 확보
    Given 2,000개 이상의 이미지가 수집되었다
    When 과별 분포를 분석한다
    Then 시각디자인(visual_design)은 최소 300개 이상이어야 한다
    And 산업디자인(industrial_design)은 최소 300개 이상이어야 한다
    And 회화(fine_art)는 최소 300개 이상이어야 한다
    And 공예(craft)는 최소 300개 이상이어야 한다
```

### AC-COL-002: 이미지 무결성 검증

```gherkin
Feature: 이미지 무결성 검증
  수집된 모든 이미지는 손상되지 않은 유효한 파일이어야 한다

  Scenario: 유효한 이미지 파일 검증
    Given 이미지 파일이 수집되었다
    When 무결성 검증을 수행한다
    Then 파일 헤더가 유효한 이미지 형식이어야 한다
    And 파일이 정상적으로 로드되어야 한다
    And 손상된 파일은 격리 처리되어야 한다

  Scenario: 지원 형식 확인
    Given 이미지 파일이 수집되었다
    When 파일 형식을 확인한다
    Then JPEG, PNG, WebP 형식만 허용되어야 한다
    And 지원되지 않는 형식은 변환하거나 제외해야 한다
```

### AC-COL-003: 저작권 및 동의 확인

```gherkin
Feature: 저작권 및 사용 동의 확인
  데이터셋에 포함된 모든 이미지는 사용 권한이 확보되어야 한다

  Scenario: 사용 동의 확인
    Given 이미지가 데이터셋에 추가될 예정이다
    When 저작권 및 동의 상태를 확인한다
    Then 사용자 동의가 확보된 이미지만 포함되어야 한다
    And 동의 상태가 메타데이터에 기록되어야 한다

  Scenario: 개인정보 포함 이미지 처리
    Given 이미지에 개인정보가 포함되어 있다
    When 개인정보 검사를 수행한다
    Then 얼굴이 포함된 이미지는 익명화 처리해야 한다
    And 이름, 서명 등이 포함된 이미지는 제외하거나 마스킹해야 한다
```

---

## 3. 전처리 인수 조건

### AC-PRE-001: 이미지 정규화

```gherkin
Feature: 이미지 정규화
  모든 이미지는 일관된 형식과 크기로 정규화되어야 한다

  Scenario: 크기 정규화
    Given 다양한 크기의 원본 이미지가 있다
    When 정규화 처리를 수행한다
    Then 모든 이미지의 장변은 768 픽셀이어야 한다
    And 원본 비율이 유지되어야 한다
    And LANCZOS 리샘플링이 적용되어야 한다

  Scenario: 형식 통일
    Given 다양한 형식의 이미지가 있다
    When 형식 변환을 수행한다
    Then 모든 이미지는 JPEG 형식이어야 한다
    And 품질 설정은 95 이상이어야 한다
    And RGB 색상 공간이어야 한다

  Scenario: 메타데이터 제거
    Given EXIF 메타데이터가 포함된 이미지가 있다
    When 정규화 처리를 수행한다
    Then EXIF 데이터가 제거되어야 한다
    And GPS, 카메라 정보가 포함되지 않아야 한다
```

### AC-PRE-002: 품질 필터링

```gherkin
Feature: 이미지 품질 필터링
  품질 기준에 미달하는 이미지는 필터링되어야 한다

  Scenario: 해상도 검사
    Given 이미지가 전처리 대기 중이다
    When 해상도를 검사한다
    Then 512x512 미만의 이미지는 제외되어야 한다
    And 제외 사유가 로그에 기록되어야 한다

  Scenario: 블러 검출
    Given 이미지가 전처리 대기 중이다
    When 블러 정도를 검사한다
    Then Laplacian variance가 100 미만인 이미지는 플래그되어야 한다
    And 수동 검토 대기열에 추가되어야 한다

  Scenario: 품질 통과율 확인
    Given 전체 이미지에 대해 품질 필터링을 완료했다
    When 통과율을 계산한다
    Then 전체 품질 통과율은 95% 이상이어야 한다
```

---

## 4. 메타데이터 태깅 인수 조건

### AC-TAG-001: 과별 분류

```gherkin
Feature: 과별 분류 태깅
  모든 이미지는 4개 과별 중 하나로 분류되어야 한다

  Scenario: 과별 분류 완료
    Given 전처리된 이미지가 있다
    When 과별 분류 태깅을 수행한다
    Then 모든 이미지에 department 필드가 설정되어야 한다
    And 값은 visual_design, industrial_design, fine_art, craft 중 하나여야 한다
    And department_confidence 필드가 설정되어야 한다

  Scenario: 분류 신뢰도 검증
    Given 자동 분류된 이미지가 있다
    When 신뢰도를 확인한다
    Then 신뢰도가 0.7 미만인 이미지는 수동 검토 플래그가 설정되어야 한다
```

### AC-TAG-002: 티어 라벨링

```gherkin
Feature: 대학 티어 라벨링
  모든 이미지는 S/A/B/C 티어로 라벨링되어야 한다

  Scenario: 티어 라벨 완료
    Given 태깅 대상 이미지가 있다
    When 티어 라벨링을 수행한다
    Then 모든 이미지에 tier 필드가 설정되어야 한다
    And 값은 S, A, B, C 중 하나여야 한다
    And tier_score 필드에 0-100 범위의 점수가 설정되어야 한다

  Scenario: 티어별 점수 범위 검증
    Given 티어가 라벨링된 이미지가 있다
    When 점수 범위를 검증한다
    Then S 티어는 85-100점 범위여야 한다
    And A 티어는 70-84점 범위여야 한다
    And B 티어는 50-69점 범위여야 한다
    And C 티어는 0-49점 범위여야 한다
```

### AC-TAG-003: 메타데이터 완성도

```gherkin
Feature: 메타데이터 완성도
  모든 필수 메타데이터 필드가 채워져야 한다

  Scenario: 필수 필드 완성
    Given 태깅된 이미지가 있다
    When 메타데이터 완성도를 검사한다
    Then image_id 필드가 존재해야 한다
    And file_name 필드가 존재해야 한다
    And department 필드가 존재해야 한다
    And tier 필드가 존재해야 한다
    And source 필드가 존재해야 한다
    And consent_status가 true여야 한다

  Scenario: 태깅 완료율 100%
    Given 전체 이미지에 대해 태깅을 완료했다
    When 완료율을 계산한다
    Then 메타데이터 태깅 완료율은 100%여야 한다
```

---

## 5. 자동 라벨링 인수 조건

### AC-LAB-001: 자동 라벨 생성

```gherkin
Feature: 자동 라벨 생성
  자동 라벨링 시스템은 이미지에 티어 라벨을 자동으로 부여해야 한다

  Scenario: 자동 라벨 생성
    Given 전처리된 이미지가 있다
    And 레퍼런스 임베딩 데이터가 로드되어 있다
    When 자동 라벨링을 실행한다
    Then tier 필드에 자동 라벨이 설정되어야 한다
    And tier_score 필드에 점수가 계산되어야 한다
    And is_manual_label 필드가 false로 설정되어야 한다

  Scenario: 4축 점수 계산
    Given 자동 라벨링을 수행한다
    When 루브릭 점수를 계산한다
    Then 구성력(composition) 점수가 계산되어야 한다
    And 명암/질감(texture) 점수가 계산되어야 한다
    And 조형완성도(completeness) 점수가 계산되어야 한다
    And 주제해석력(interpretation) 점수가 계산되어야 한다
```

### AC-LAB-002: 라벨 검증

```gherkin
Feature: 라벨 검증
  자동 생성된 라벨은 샘플링 검증을 통과해야 한다

  Scenario: 샘플링 검증
    Given 자동 라벨링이 완료된 데이터셋이 있다
    When 15% 샘플을 추출하여 전문가 검토를 수행한다
    Then 전문가 라벨과 1단계 이내 일치율이 85% 이상이어야 한다
    And 검증 결과가 validation_report.json에 기록되어야 한다

  Scenario: 불일치 라벨 처리
    Given 전문가 검토 결과 불일치 라벨이 발견되었다
    When 불일치 처리 프로세스를 실행한다
    Then 해당 이미지의 is_manual_label이 true로 변경되어야 한다
    And 전문가 라벨로 수정되어야 한다
    And 수정 이력이 기록되어야 한다
```

### AC-LAB-003: 라벨링 정확도

```gherkin
Feature: 라벨링 정확도
  자동 라벨링 정확도는 목표치를 충족해야 한다

  Scenario: 정확도 목표 달성
    Given 샘플 검증이 완료되었다
    When 전체 정확도를 계산한다
    Then 자동 라벨링 정확도는 85% 이상이어야 한다

  Scenario: 과별 정확도 검증
    Given 샘플 검증이 완료되었다
    When 과별 정확도를 계산한다
    Then 각 과별 정확도는 80% 이상이어야 한다
```

---

## 6. 통합 인수 조건

### AC-INT-001: 전체 파이프라인 통합

```gherkin
Feature: 전체 파이프라인 통합
  수집-전처리-태깅-라벨링 파이프라인이 원활하게 연동되어야 한다

  Scenario: End-to-End 파이프라인 실행
    Given 이미지 소스가 준비되어 있다
    When 전체 파이프라인을 실행한다
    Then 수집 단계가 성공적으로 완료되어야 한다
    And 전처리 단계가 성공적으로 완료되어야 한다
    And 태깅 단계가 성공적으로 완료되어야 한다
    And 라벨링 단계가 성공적으로 완료되어야 한다
    And 최종 데이터셋이 생성되어야 한다

  Scenario: 오류 복구
    Given 파이프라인 실행 중 오류가 발생했다
    When 재실행을 수행한다
    Then 이미 완료된 단계는 건너뛰어야 한다
    And 실패한 항목만 재처리되어야 한다
```

### AC-INT-002: 데이터셋 분할

```gherkin
Feature: 데이터셋 분할
  최종 데이터셋은 학습/검증/테스트 세트로 분할되어야 한다

  Scenario: Train/Val/Test 분할
    Given 전체 데이터셋이 준비되었다
    When 데이터셋 분할을 수행한다
    Then Train 세트는 전체의 80%여야 한다
    And Validation 세트는 전체의 10%여야 한다
    And Test 세트는 전체의 10%여야 한다

  Scenario: 분할 시 계층화
    Given 데이터셋 분할을 수행한다
    When 각 세트의 분포를 확인한다
    Then 각 세트의 과별 분포가 전체 분포와 유사해야 한다
    And 각 세트의 티어 분포가 전체 분포와 유사해야 한다
```

### AC-INT-003: 저장소 연동

```gherkin
Feature: 저장소 연동
  데이터셋은 Firebase 및 로컬 저장소에 올바르게 저장되어야 한다

  Scenario: Firestore 메타데이터 저장
    Given 메타데이터가 생성되었다
    When Firestore에 저장한다
    Then 모든 메타데이터가 올바르게 저장되어야 한다
    And 인덱스 쿼리가 정상 동작해야 한다

  Scenario: 로컬 데이터셋 저장
    Given 처리된 이미지가 있다
    When 로컬 저장소에 저장한다
    Then datasets/processed/ 디렉토리에 이미지가 저장되어야 한다
    And datasets/metadata/ 디렉토리에 JSON 파일이 저장되어야 한다
    And datasets/labels/ 디렉토리에 라벨 파일이 저장되어야 한다
```

---

## 7. 품질 게이트

### QG-001: 코드 품질

```gherkin
Feature: 코드 품질 검증
  모든 코드는 품질 기준을 충족해야 한다

  Scenario: 테스트 커버리지
    Given 모든 모듈 코드가 작성되었다
    When pytest --cov를 실행한다
    Then 테스트 커버리지가 85% 이상이어야 한다

  Scenario: 린터 검사
    Given 모든 코드가 작성되었다
    When ruff check를 실행한다
    Then 린터 경고가 0개여야 한다

  Scenario: 타입 검사
    Given 모든 코드가 작성되었다
    When mypy를 실행한다
    Then 타입 오류가 0개여야 한다
```

### QG-002: 성능 기준

```gherkin
Feature: 성능 기준 검증
  파이프라인은 성능 기준을 충족해야 한다

  Scenario: 단일 이미지 처리 시간
    Given 전처리 파이프라인이 준비되었다
    When 단일 이미지를 처리한다
    Then 처리 시간이 5초 이내여야 한다

  Scenario: 배치 처리 성능
    Given 100개 이미지 배치가 준비되었다
    When 배치 처리를 실행한다
    Then 총 처리 시간이 5분 이내여야 한다
```

---

## 8. 검증 체크리스트

### 최종 인수 검증 체크리스트

```
데이터 수집 (데이터 수집 단계에서 수행 예정):
[ ] 총 이미지 수 >= 2,000개
[ ] 과별 최소 분포 >= 15% (300개)
[ ] 저작권 확보율 100%
[ ] 개인정보 포함 이미지 0개 (또는 익명화 완료)

전처리 코드 구현 (2026-01-18 완료):
[x] ResizePreprocessor 구현 완료
[x] NormalizePreprocessor 구현 완료
[x] AugmentPreprocessor 구현 완료
[x] 품질 필터링 로직 구현 완료

메타데이터 태깅 코드 구현 (2026-01-18 완료):
[x] ImageMetadata Pydantic 모델 구현 완료
[x] Department Enum (시디/산디/공예/회화/미분류) 구현 완료
[x] Tier Enum (S/A/B/C/미분류) 구현 완료
[x] DepartmentTagger 구현 완료
[x] TierTagger 구현 완료

자동 라벨링 코드 구현 (2026-01-18 완료):
[x] AutoLabeler 구현 완료
[x] 4축 점수 계산 로직 구현 완료
[x] 신뢰도 점수 계산 구현 완료

스토리지 코드 구현 (2026-01-18 완료):
[x] LocalStorage 구현 완료
[x] MetadataStorage 구현 완료

통합 파이프라인 (2026-01-18 완료):
[x] DataPipeline 클래스 구현 완료
[x] E2E 파이프라인 동작 확인

품질 게이트 (2026-01-18 검증 완료):
[x] 테스트 커버리지 >= 85% (실제: 90%)
[x] 린터 경고 0개
[x] 168개 테스트 모두 통과
```

---

*인수 조건 버전: 1.1.0*
*작성일: 2025-01-18*
*최종 수정일: 2026-01-18*
*검증 완료일: 2026-01-18*
*SPEC 참조: SPEC-DATA-001*

---

## 9. 검증 결과 요약

### 코드 품질 검증 (2026-01-18)

| 항목 | 목표 | 실제 | 상태 |
|------|------|------|------|
| 테스트 커버리지 | >= 85% | 90% | **통과** |
| 테스트 통과율 | 100% | 168/168 (100%) | **통과** |
| 린터 경고 | 0개 | 0개 | **통과** |

### 구현 완료 컴포넌트 (2026-01-18)

| 컴포넌트 | 파일 수 | 테스트 수 | 상태 |
|----------|---------|-----------|------|
| Models | 2 | 20+ | **완료** |
| Collectors | 3 | 20+ | **완료** |
| Preprocessors | 5 | 30+ | **완료** |
| Taggers | 4 | 30+ | **완료** |
| Labelers | 3 | 25+ | **완료** |
| Storage | 4 | 25+ | **완료** |
| Pipeline | 1 | 15+ | **완료** |

### 다음 단계

파이프라인 코드 구현이 완료되었습니다. 다음 단계는 실제 데이터 수집입니다:

1. **데이터 소스 확보**: 공모전 출품작, 파트너 학원 이미지
2. **파이프라인 실행**: 수집 -> 전처리 -> 태깅 -> 라벨링
3. **품질 검증**: 샘플링 기반 전문가 검토
4. **데이터셋 완성**: train/val/test 분할
