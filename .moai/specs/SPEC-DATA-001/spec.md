# SPEC-DATA-001: AI 진단 데이터 파이프라인

## TAG BLOCK

```yaml
spec_id: SPEC-DATA-001
title: AI 진단 데이터 파이프라인
status: Completed
priority: High
created: 2025-01-18
updated: 2026-01-18
completed: 2026-01-18
lifecycle: spec-anchored
domain: data-engineering
assigned: expert-backend
related_specs: [SPEC-COMP-001, SPEC-BACKEND-001]
labels: [data-pipeline, image-processing, ml-training, metadata, labeling]
test_coverage: 90%
tests_passed: 168/168
```

---

## 1. Environment (환경)

### 1.1 기술 스택

| 영역 | 기술 | 버전 | 용도 |
|------|------|------|------|
| Framework | PyTorch | 2.1+ | 딥러닝 프레임워크 |
| Transformers | Hugging Face Transformers | 4.35+ | DINOv2, CLIP 모델 로드 |
| Image Processing | OpenCV | 4.8+ | 이미지 전처리/정규화 |
| Image Processing | Pillow | 10.0+ | 이미지 I/O |
| Image Processing | albumentations | 1.3+ | 데이터 증강 |
| Data Validation | Pydantic | 2.x | 메타데이터 검증 |
| Storage | Firebase Storage | - | 이미지 저장소 |
| Database | Firebase Firestore | - | 메타데이터 저장 |
| Language | Python | 3.10+ | 파이프라인 언어 |

### 1.2 ML 모델 아키텍처

| 모델 | 역할 | 선택 이유 |
|------|------|-----------|
| DINOv2 ViT-L | RGB 피처 추출 | visual similarity에서 CLIP 대비 2배 이상 정확도 (64% vs 28%) |
| PiDiNet | Edge 피처 추출 | HED 대비 28% 파라미터로 0.9% ODS 향상, 경량화 |
| CLIP ViT-L | 주제 해석 전용 | text-image alignment 최적화 |

### 1.3 디렉토리 구조

```
backend/
├── data_pipeline/
│   ├── __init__.py
│   ├── config.py                 # 파이프라인 설정
│   ├── collectors/               # 데이터 수집 모듈
│   │   ├── __init__.py
│   │   ├── firebase_collector.py # Firebase에서 이미지 수집
│   │   ├── local_collector.py    # 로컬 파일 시스템 수집
│   │   └── validation.py         # 이미지 유효성 검사
│   │
│   ├── preprocessors/            # 전처리 모듈
│   │   ├── __init__.py
│   │   ├── normalizer.py         # 이미지 정규화 (크기, 형식)
│   │   ├── quality_filter.py     # 품질 필터링
│   │   └── augmentor.py          # 데이터 증강
│   │
│   ├── taggers/                  # 메타데이터 태깅 모듈
│   │   ├── __init__.py
│   │   ├── department_tagger.py  # 과별 분류 태거
│   │   ├── tier_tagger.py        # 대학 티어 라벨러
│   │   ├── keyword_extractor.py  # 주제 키워드 추출
│   │   └── metadata_schema.py    # 메타데이터 스키마
│   │
│   ├── labelers/                 # 자동 라벨링 모듈
│   │   ├── __init__.py
│   │   ├── auto_labeler.py       # 자동 라벨 생성
│   │   ├── score_mapper.py       # 티어별 점수 범위 설정
│   │   └── validator.py          # 라벨 검증 (샘플링)
│   │
│   ├── storage/                  # 저장소 모듈
│   │   ├── __init__.py
│   │   ├── firebase_storage.py   # Firebase Storage 연동
│   │   └── firestore_db.py       # Firestore 메타데이터 저장
│   │
│   └── scripts/                  # 실행 스크립트
│       ├── run_collection.py     # 데이터 수집 실행
│       ├── run_preprocessing.py  # 전처리 실행
│       ├── run_tagging.py        # 태깅 실행
│       ├── run_labeling.py       # 자동 라벨링 실행
│       └── run_validation.py     # 검증 실행
│
├── datasets/                     # 로컬 데이터셋 (Git 제외)
│   ├── raw/                      # 원본 이미지
│   ├── processed/                # 전처리된 이미지
│   ├── metadata/                 # 메타데이터 JSON
│   └── labels/                   # 라벨 파일
│
└── tests/
    ├── test_collectors/
    ├── test_preprocessors/
    ├── test_taggers/
    └── test_labelers/
```

---

## 2. Assumptions (가정)

### 2.1 기술적 가정

| ID | 가정 | 신뢰도 | 근거 | 검증 방법 |
|----|------|--------|------|-----------|
| A-001 | 공모전 출품 이미지 활용 가능 | High | SPEC-COMP-001 구현 완료 예정 | 데이터 접근 권한 확인 |
| A-002 | Firebase Storage에 이미지 저장됨 | High | tech.md 명시 | Storage 버킷 확인 |
| A-003 | 이미지 평균 크기 2-5MB | Medium | 일반적인 작품 이미지 크기 | 샘플 분석 |
| A-004 | Python 3.10+ 환경 사용 가능 | High | tech.md 명시 | 버전 확인 |
| A-005 | GPU 없이도 전처리 가능 | High | 전처리는 CPU 기반 | 로컬 테스트 |

### 2.2 비즈니스 가정

| ID | 가정 | 신뢰도 | 근거 |
|----|------|--------|------|
| B-001 | 2,000개 이미지 수집 목표 달성 가능 | Medium | 공모전 + 외부 소스 활용 |
| B-002 | 4개 과별 분류 체계 유지 | High | product.md 명시 (시디/산디/공예/회화) |
| B-003 | S/A/B/C 4단계 티어 체계 | High | product.md 명시 |
| B-004 | 저작권/개인정보 이슈 해결 가능 | Medium | 사용자 동의 기반 |

### 2.3 데이터 가정

| ID | 가정 | 신뢰도 | 근거 |
|----|------|--------|------|
| D-001 | 과별 분포: 시디 40%, 산디 25%, 회화 20%, 공예 15% | Low | 입시생 분포 추정 |
| D-002 | 티어별 분포: S 5%, A 25%, B 45%, C 25% | Low | 대학 합격률 추정 |
| D-003 | 이미지 형식: JPEG 80%, PNG 20% | Medium | 일반적 이미지 형식 |

### 2.4 제약 조건

- [HARD] 저작권이 확보된 이미지만 사용
- [HARD] 개인정보(얼굴, 이름)가 포함된 이미지 제외 또는 익명화
- [HARD] 이미지 최소 해상도: 512x512 픽셀
- [HARD] 데이터셋 로컬 저장 (Git 제외)
- [SOFT] 과별 균형 분포 목표 (최소 각 15% 이상)
- [SOFT] 티어별 균형 분포 고려

---

## 3. Requirements (요구사항)

### 3.1 Ubiquitous Requirements (항상 적용)

| ID | 요구사항 |
|----|----------|
| REQ-U-001 | 시스템은 **항상** 이미지 파일에 대해 무결성 검증을 수행해야 한다 |
| REQ-U-002 | 시스템은 **항상** 메타데이터를 JSON 형식으로 저장해야 한다 |
| REQ-U-003 | 시스템은 **항상** 처리 로그를 파일로 기록해야 한다 |
| REQ-U-004 | 시스템은 **항상** 원본 이미지를 보존하고 전처리된 이미지를 별도 저장해야 한다 |
| REQ-U-005 | 시스템은 **항상** Pydantic 모델을 통해 메타데이터를 검증해야 한다 |

### 3.2 Event-Driven Requirements (이벤트 기반)

| ID | 요구사항 |
|----|----------|
| REQ-E-001 | **WHEN** 새 이미지가 수집되면 **THEN** 품질 검사 및 정규화를 자동 수행해야 한다 |
| REQ-E-002 | **WHEN** 이미지가 정규화되면 **THEN** 메타데이터 태깅 파이프라인을 트리거해야 한다 |
| REQ-E-003 | **WHEN** 태깅이 완료되면 **THEN** 자동 라벨링을 적용해야 한다 |
| REQ-E-004 | **WHEN** 라벨링이 완료되면 **THEN** Firestore에 메타데이터를 저장해야 한다 |
| REQ-E-005 | **WHEN** 품질 검사에 실패하면 **THEN** 해당 이미지를 격리하고 실패 원인을 기록해야 한다 |
| REQ-E-006 | **WHEN** 배치 처리가 완료되면 **THEN** 처리 통계 리포트를 생성해야 한다 |

### 3.3 State-Driven Requirements (상태 기반)

| ID | 요구사항 |
|----|----------|
| REQ-S-001 | **IF** 이미지 해상도가 512x512 미만이면 **THEN** 업스케일링 또는 제외 처리해야 한다 |
| REQ-S-002 | **IF** 이미지 형식이 지원되지 않으면 **THEN** JPEG로 변환하거나 제외해야 한다 |
| REQ-S-003 | **IF** 과별 분류가 불확실하면 **THEN** 수동 검토 대기열에 추가해야 한다 |
| REQ-S-004 | **IF** 티어 라벨이 수동으로 설정되었으면 **THEN** 자동 라벨링을 건너뛰어야 한다 |
| REQ-S-005 | **IF** Firebase 연결이 실패하면 **THEN** 로컬 캐시에 저장하고 재시도 큐에 추가해야 한다 |

### 3.4 Unwanted Requirements (금지 사항)

| ID | 요구사항 |
|----|----------|
| REQ-N-001 | 시스템은 저작권이 확보되지 않은 이미지를 데이터셋에 포함**하지 않아야 한다** |
| REQ-N-002 | 시스템은 개인정보가 포함된 이미지를 익명화 없이 저장**하지 않아야 한다** |
| REQ-N-003 | 시스템은 원본 이미지를 삭제하거나 수정**하지 않아야 한다** |
| REQ-N-004 | 시스템은 검증되지 않은 라벨을 최종 데이터셋에 포함**하지 않아야 한다** |
| REQ-N-005 | 시스템은 512x512 미만 해상도 이미지를 학습 데이터로 사용**하지 않아야 한다** |

### 3.5 Optional Requirements (선택 사항)

| ID | 요구사항 |
|----|----------|
| REQ-O-001 | **가능하면** CLIP 기반 주제 키워드 자동 추출 제공 |
| REQ-O-002 | **가능하면** 데이터 증강(회전, 플립, 색상 조정) 옵션 제공 |
| REQ-O-003 | **가능하면** 중복 이미지 탐지 및 제거 기능 제공 |
| REQ-O-004 | **가능하면** 대시보드를 통한 데이터셋 현황 시각화 제공 |
| REQ-O-005 | **가능하면** 외부 데이터셋 임포트 인터페이스 제공 |

---

## 4. Specifications (세부 명세)

### 4.1 데이터 수집 명세

#### 4.1.1 수집 소스

| 소스 | 우선순위 | 예상 수량 | 설명 |
|------|----------|-----------|------|
| 공모전 출품작 | High | 500-1,000 | SPEC-COMP-001 통해 수집된 이미지 |
| 사용자 업로드 | High | 300-500 | AI 진단 서비스 이용자 이미지 (동의 필요) |
| 파트너 학원 | Medium | 500-700 | 협력 학원 제공 이미지 |
| 오픈 데이터셋 | Low | 200-300 | 공개 미술 데이터셋 (저작권 확인) |

#### 4.1.2 수집 워크플로우

```
[소스] -> [다운로드] -> [무결성 검증] -> [저작권 확인] -> [raw/ 저장]
                              |
                              v
                    [실패 시 격리 + 로깅]
```

### 4.2 전처리 명세

#### 4.2.1 이미지 정규화

| 단계 | 처리 내용 | 파라미터 |
|------|-----------|----------|
| 형식 변환 | 지원 형식 통일 | JPEG (품질 95) |
| 크기 정규화 | 표준 해상도 조정 | 768x768 (장변 기준) |
| 색상 공간 | RGB 통일 | sRGB |
| 메타데이터 제거 | EXIF 등 제거 | 개인정보 보호 |

#### 4.2.2 품질 필터링

| 기준 | 임계값 | 처리 |
|------|--------|------|
| 최소 해상도 | 512x512 | 미만 시 제외 또는 업스케일링 |
| 블러 검출 | Laplacian < 100 | 흐린 이미지 제외 |
| 노이즈 검출 | SNR < 20dB | 노이즈 심한 이미지 제외 |
| 컬러 다양성 | 색상 채널 분산 | 단색/무채색 필터링 |

### 4.3 메타데이터 스키마

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime

class Department(str, Enum):
    VISUAL_DESIGN = "visual_design"      # 시각디자인
    INDUSTRIAL_DESIGN = "industrial_design"  # 산업디자인
    FINE_ART = "fine_art"                # 회화
    CRAFT = "craft"                      # 공예

class Tier(str, Enum):
    S = "S"  # 최상위 (상위 5%)
    A = "A"  # 상위 (상위 30%)
    B = "B"  # 중위 (상위 75%)
    C = "C"  # 하위 (하위 25%)

class Medium(str, Enum):
    PENCIL = "pencil"           # 연필
    CHARCOAL = "charcoal"       # 목탄
    WATERCOLOR = "watercolor"   # 수채화
    OIL = "oil"                 # 유화
    ACRYLIC = "acrylic"         # 아크릴
    DIGITAL = "digital"         # 디지털
    MIXED = "mixed"             # 혼합매체

class ImageMetadata(BaseModel):
    # 식별 정보
    image_id: str = Field(..., description="고유 이미지 ID")
    file_name: str = Field(..., description="파일명")
    file_path: str = Field(..., description="저장 경로")

    # 원본 정보
    original_width: int = Field(..., ge=1, description="원본 너비")
    original_height: int = Field(..., ge=1, description="원본 높이")
    file_size_bytes: int = Field(..., ge=1, description="파일 크기")

    # 분류 정보
    department: Department = Field(..., description="과별 분류")
    department_confidence: float = Field(0.0, ge=0.0, le=1.0, description="분류 신뢰도")

    # 티어 라벨
    tier: Tier = Field(..., description="대학 티어")
    tier_score: float = Field(..., ge=0.0, le=100.0, description="티어 점수")
    is_manual_label: bool = Field(False, description="수동 라벨 여부")

    # 상세 정보
    year: Optional[int] = Field(None, ge=2000, le=2030, description="제작 연도")
    admission_type: Optional[str] = Field(None, description="전형 유형")
    medium: Optional[Medium] = Field(None, description="매체")
    keywords: List[str] = Field(default_factory=list, description="주제 키워드")

    # 출처 정보
    source: str = Field(..., description="수집 소스")
    consent_status: bool = Field(True, description="사용 동의 여부")
    copyright_cleared: bool = Field(True, description="저작권 확보 여부")

    # 처리 정보
    processed_at: datetime = Field(default_factory=datetime.now)
    pipeline_version: str = Field("1.0.0", description="파이프라인 버전")

    class Config:
        use_enum_values = True
```

### 4.4 자동 라벨링 명세

#### 4.4.1 티어별 점수 범위

| 티어 | 점수 범위 | 대학 예시 | 설명 |
|------|-----------|-----------|------|
| S | 85-100 | 서울대, 홍대, 국민대 | 최상위 합격권 |
| A | 70-84 | 건국대, 동국대, 세종대 | 상위 합격권 |
| B | 50-69 | 중경외시, 단국대, 명지대 | 중위 합격권 |
| C | 0-49 | 기타 대학 | 개선 필요 |

#### 4.4.2 자동 라벨링 알고리즘

```python
# 자동 라벨링 로직 (의사 코드)
def auto_label(image_features: dict) -> tuple[Tier, float]:
    """
    DINOv2 피처 기반 자동 라벨링

    Args:
        image_features: 추출된 이미지 피처

    Returns:
        (티어, 점수)
    """
    # 1. 피처 정규화
    normalized_features = normalize(image_features)

    # 2. 레퍼런스 데이터와 유사도 계산
    similarity_scores = compute_similarity(
        normalized_features,
        reference_embeddings
    )

    # 3. 4축 점수 계산 (구성력, 명암/질감, 조형완성도, 주제해석력)
    rubric_scores = {
        "composition": compute_composition_score(normalized_features),
        "texture": compute_texture_score(normalized_features),
        "completeness": compute_completeness_score(normalized_features),
        "interpretation": compute_interpretation_score(normalized_features)
    }

    # 4. 가중 평균 점수 계산
    weights = {"composition": 0.25, "texture": 0.25,
               "completeness": 0.30, "interpretation": 0.20}
    final_score = weighted_average(rubric_scores, weights)

    # 5. 티어 결정
    tier = score_to_tier(final_score)

    return tier, final_score
```

### 4.5 검증 명세

#### 4.5.1 샘플링 검증

| 검증 유형 | 샘플 비율 | 검증 방법 |
|----------|-----------|-----------|
| 과별 분류 검증 | 10% | 전문가 수동 검토 |
| 티어 라벨 검증 | 15% | 전문가 수동 검토 + 교차 검증 |
| 품질 검증 | 5% | 시각적 검토 |

#### 4.5.2 검증 기준

| 지표 | 목표값 | 설명 |
|------|--------|------|
| 과별 분류 정확도 | >= 90% | 전문가와 일치율 |
| 티어 분류 정확도 | >= 85% | 전문가와 1단계 이내 일치 |
| 데이터 품질 점수 | >= 95% | 품질 기준 통과율 |

---

## 5. Traceability (추적성)

### 5.1 연관 문서

| 문서 | 경로 | 관계 |
|------|------|------|
| 기술 스택 | `.moai/project/tech.md` | ML 모델 및 기술 선택 근거 |
| 제품 문서 | `.moai/project/product.md` | 비즈니스 요구사항 |
| 프로젝트 구조 | `.moai/project/structure.md` | 디렉토리 구조 참조 |
| 공모전 SPEC | `.moai/specs/SPEC-COMP-001/spec.md` | 출품 이미지 데이터 소스 |
| 백엔드 SPEC | `.moai/specs/SPEC-BACKEND-001/spec.md` | API 연동 |

### 5.2 후속 SPEC

| SPEC ID | 제목 | 의존성 |
|---------|------|--------|
| SPEC-ML-001 | AI 진단 모델 학습 | 이 SPEC 완료 후 (데이터셋 필요) |
| SPEC-ML-002 | 루브릭 평가 헤드 구현 | SPEC-ML-001 완료 후 |
| SPEC-API-001 | AI 진단 API 통합 | SPEC-ML-001 완료 후 |

### 5.3 Quality Gates

- [x] 모든 EARS 요구사항이 테스트 가능 (2026-01-18 검증 완료)
- [x] 85% 이상 테스트 커버리지 (실제: 90%) (2026-01-18 검증 완료)
- [ ] 2,000개 이미지 수집 완료 (데이터 수집 단계에서 수행 예정)
- [ ] 메타데이터 태깅 완료율 100% (데이터 수집 단계에서 수행 예정)
- [ ] 자동 라벨링 정확도 >= 85% (데이터 수집 단계에서 수행 예정)
- [x] Ruff 린터 경고 없음 (2026-01-18 검증 완료)
- [ ] 데이터 품질 검증 통과 (데이터 수집 단계에서 수행 예정)

---

*SPEC 버전: 1.1.0*
*작성일: 2025-01-18*
*최종 수정일: 2026-01-18*
*완료일: 2026-01-18*
*담당 에이전트: expert-backend*

---

## 6. 구현 요약 (Implementation Summary)

### 6.1 구현된 컴포넌트

| 모듈 | 파일 수 | 주요 클래스/함수 |
|------|---------|------------------|
| models | 2 | ImageMetadata, Department, Tier |
| collectors | 3 | BaseCollector, ImageCollector |
| preprocessors | 5 | ResizePreprocessor, NormalizePreprocessor, AugmentPreprocessor |
| taggers | 4 | DepartmentTagger, TierTagger |
| labelers | 3 | AutoLabeler |
| storage | 4 | LocalStorage, MetadataStorage |
| pipeline | 1 | DataPipeline |
| **합계** | **24** | - |

### 6.2 테스트 결과

- 총 테스트: 168개
- 성공: 168개 (100%)
- 실패: 0개
- 커버리지: 90%

### 6.3 커밋 이력

| 커밋 해시 | 설명 |
|-----------|------|
| 1dc5351 | 데이터 파이프라인 기본 구조 및 메타데이터 모델 구현 |
| 93c89ef | 이미지 수집기 및 전처리기 구현 |
| 7aa94ea | 과별/티어 태거 및 자동 라벨러 구현 |
| e171af9 | 스토리지 및 통합 파이프라인 구현 |
| c7445bb | 단위 테스트 추가 (커버리지 90%) |
