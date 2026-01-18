# SPEC-DATA-001: 구현 계획서

## TAG BLOCK

```yaml
spec_id: SPEC-DATA-001
title: AI 진단 데이터 파이프라인 - 구현 계획
status: Completed
created: 2025-01-18
updated: 2026-01-18
completed: 2026-01-18
related_spec: SPEC-DATA-001/spec.md
```

---

## 1. 마일스톤 개요

### 우선순위 기반 구현 단계

| 마일스톤 | 우선순위 | 주요 목표 | 의존성 | 상태 | 완료일 |
|----------|----------|-----------|--------|------|--------|
| M1: 수집 인프라 | Primary | 데이터 수집 파이프라인 구축 | 없음 | **완료** | 2026-01-18 |
| M2: 전처리 시스템 | Primary | 이미지 정규화 및 품질 필터링 | M1 | **완료** | 2026-01-18 |
| M3: 메타데이터 태깅 | Secondary | 과별/티어 태깅 시스템 | M2 | **완료** | 2026-01-18 |
| M4: 자동 라벨링 | Secondary | 자동 라벨 생성 및 검증 | M3 | **완료** | 2026-01-18 |
| M5: 통합 및 검증 | Final | 파이프라인 통합 및 품질 검증 | M4 | **완료** | 2026-01-18 |

---

## 2. 마일스톤 상세 계획

### M1: 데이터 수집 인프라 (Primary Goal)

#### 목표
- 다양한 소스에서 이미지 수집 가능한 인프라 구축
- Firebase Storage 연동 및 로컬 파일 시스템 지원

#### 태스크 체크리스트

```
[x] T1.1: 수집 모듈 기본 구조 설계 (2026-01-18 완료)
    - collectors/ 디렉토리 구조 생성
    - 추상 베이스 클래스 정의 (BaseCollector)
    - 설정 관리 (config.py)

[x] T1.2: Firebase 수집기 구현 (2026-01-18 완료 - 로컬 수집기로 대체)
    - Firebase Storage 연결 설정 -> 추후 구현 예정
    - 이미지 다운로드 기능
    - 배치 다운로드 지원
    - 재시도 로직 구현

[x] T1.3: 로컬 수집기 구현 (2026-01-18 완료)
    - 로컬 디렉토리 스캔 (ImageCollector)
    - 파일 복사 및 이동
    - 중복 파일 탐지

[x] T1.4: 유효성 검증 모듈 (2026-01-18 완료)
    - 이미지 파일 무결성 검사
    - 지원 형식 확인
    - 손상 파일 탐지

[x] T1.5: 수집 스크립트 작성 (2026-01-18 완료 - 파이프라인에 통합)
    - DataPipeline 클래스로 통합
    - 배치 처리 지원
    - 진행 상황 로깅
```

#### 기술 접근 방식

```python
# collectors/base.py (추상 베이스 클래스)
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from pydantic import BaseModel

class CollectionResult(BaseModel):
    image_id: str
    source_path: str
    local_path: str
    success: bool
    error_message: Optional[str] = None

class BaseCollector(ABC):
    @abstractmethod
    def collect(self, source: str) -> Iterator[CollectionResult]:
        """이미지 수집 제너레이터"""
        pass

    @abstractmethod
    def validate(self, path: str) -> bool:
        """이미지 유효성 검증"""
        pass
```

#### 아키텍처 설계

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection Layer                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Firebase   │  │    Local     │  │   External   │       │
│  │  Collector   │  │  Collector   │  │   Collector  │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                  ┌─────────▼─────────┐                       │
│                  │    Validation     │                       │
│                  │      Module       │                       │
│                  └─────────┬─────────┘                       │
│                            │                                 │
│                  ┌─────────▼─────────┐                       │
│                  │    Raw Storage    │                       │
│                  │   (datasets/raw)  │                       │
│                  └───────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

#### 리스크 및 대응

| 리스크 | 영향도 | 대응 방안 |
|--------|--------|-----------|
| Firebase 연결 실패 | High | 재시도 로직 + 로컬 캐시 |
| 대용량 파일 메모리 이슈 | Medium | 스트리밍 다운로드 |
| 중복 파일 수집 | Low | 해시 기반 중복 탐지 |

---

### M2: 이미지 전처리 시스템 (Primary Goal)

#### 목표
- 이미지 정규화 (크기, 형식, 색상 공간)
- 품질 필터링 시스템 구축

#### 태스크 체크리스트

```
[x] T2.1: 정규화 모듈 구현 (2026-01-18 완료)
    - 크기 조정 (ResizePreprocessor)
    - 형식 변환 (NormalizePreprocessor)
    - 색상 공간 정규화 (sRGB)
    - EXIF 메타데이터 제거

[x] T2.2: 품질 필터링 모듈 (2026-01-18 완료)
    - 블러 탐지 (Laplacian variance)
    - 해상도 검사 (최소 512x512)
    - 노이즈 수준 분석
    - 컬러 다양성 검사

[x] T2.3: 배치 전처리 파이프라인 (2026-01-18 완료)
    - 병렬 처리 지원
    - 진행률 추적
    - 실패 항목 재처리

[x] T2.4: 전처리 스크립트 (2026-01-18 완료 - 파이프라인에 통합)
    - DataPipeline 클래스로 통합
    - 설정 파일 지원
    - 드라이런 모드
```

#### 기술 접근 방식

```python
# preprocessors/normalizer.py
import cv2
from PIL import Image
from pathlib import Path

class ImageNormalizer:
    def __init__(
        self,
        target_size: int = 768,
        output_format: str = "JPEG",
        quality: int = 95
    ):
        self.target_size = target_size
        self.output_format = output_format
        self.quality = quality

    def normalize(self, input_path: Path, output_path: Path) -> bool:
        """이미지 정규화 수행"""
        # 1. 이미지 로드
        img = Image.open(input_path)

        # 2. RGB 변환
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 3. 크기 조정 (장변 기준)
        width, height = img.size
        if width > height:
            new_width = self.target_size
            new_height = int(height * (self.target_size / width))
        else:
            new_height = self.target_size
            new_width = int(width * (self.target_size / height))

        img = img.resize((new_width, new_height), Image.LANCZOS)

        # 4. EXIF 제거 및 저장
        img.save(output_path, self.output_format, quality=self.quality)
        return True
```

#### 아키텍처 설계

```
┌─────────────────────────────────────────────────────────────┐
│                   Preprocessing Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  datasets/raw/         ┌──────────────────────────────────┐ │
│       │                │        Preprocessing Flow         │ │
│       │                │                                   │ │
│       ▼                │  ┌─────────┐    ┌─────────────┐  │ │
│  ┌─────────┐           │  │  Load   │───▶│   Quality   │  │ │
│  │  Image  │───────────│─▶│  Image  │    │   Filter    │  │ │
│  │  Input  │           │  └─────────┘    └──────┬──────┘  │ │
│  └─────────┘           │                        │         │ │
│                        │                        ▼         │ │
│                        │               ┌─────────────┐    │ │
│                        │               │  Normalize  │    │ │
│                        │               │   (resize,  │    │ │
│                        │               │   format)   │    │ │
│                        │               └──────┬──────┘    │ │
│                        │                      │           │ │
│                        └──────────────────────│───────────┘ │
│                                               ▼             │
│                                    datasets/processed/      │
└─────────────────────────────────────────────────────────────┘
```

---

### M3: 메타데이터 태깅 시스템 (Secondary Goal)

#### 목표
- 과별 분류 (시디/산디/공예/회화)
- 대학 티어 라벨 (S/A/B/C)
- 주제 키워드 추출
- 연도, 전형, 매체 태깅

#### 태스크 체크리스트

```
[x] T3.1: 메타데이터 스키마 구현 (2026-01-18 완료)
    - Pydantic 모델 정의 (ImageMetadata)
    - Department, Tier Enum 정의
    - JSON 직렬화/역직렬화

[x] T3.2: 과별 분류 태거 (2026-01-18 완료)
    - DepartmentTagger 구현
    - 파일명/폴더 기반 자동 분류
    - 신뢰도 점수 계산

[x] T3.3: 티어 태거 (2026-01-18 완료)
    - TierTagger 구현
    - 티어 기준 가이드라인 적용
    - 라벨 이력 관리

[ ] T3.4: 키워드 추출기 (선택 - 추후 구현)
    - CLIP 기반 주제 해석
    - 키워드 후보 생성
    - 수동 검토/수정

[x] T3.5: 메타데이터 저장 (2026-01-18 완료)
    - MetadataStorage 구현
    - JSON 파일 저장
    - Firestore 연동 (추후 구현)
```

#### 기술 접근 방식

```python
# taggers/department_tagger.py
from enum import Enum
from typing import Optional, Tuple
from pathlib import Path
import re

class DepartmentTagger:
    """과별 분류 태거"""

    # 폴더명/파일명 패턴 기반 분류 규칙
    PATTERNS = {
        "visual_design": ["시디", "시각", "visual", "vd", "graphic"],
        "industrial_design": ["산디", "산업", "industrial", "id", "product"],
        "fine_art": ["회화", "순수", "fine", "painting", "drawing"],
        "craft": ["공예", "craft", "ceramic", "textile", "jewelry"]
    }

    def tag(self, image_path: Path) -> Tuple[str, float]:
        """
        파일 경로 기반 과별 분류

        Returns:
            (department, confidence)
        """
        path_str = str(image_path).lower()

        for dept, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if pattern in path_str:
                    return dept, 0.9  # 패턴 매칭 시 높은 신뢰도

        return "unknown", 0.0  # 분류 불가 시

# taggers/metadata_schema.py - spec.md에 정의된 ImageMetadata 클래스 사용
```

---

### M4: 자동 라벨링 시스템 (Secondary Goal)

#### 목표
- 티어별 점수 범위 설정
- 자동 라벨 생성 스크립트
- 라벨 검증 (샘플링)

#### 태스크 체크리스트

```
[x] T4.1: 점수 범위 매퍼 (2026-01-18 완료)
    - 티어별 점수 범위 정의
    - 점수 → 티어 변환 로직
    - 경계값 처리

[x] T4.2: 자동 라벨러 구현 (2026-01-18 완료)
    - AutoLabeler 클래스 구현
    - 4축 점수 계산 로직 (구성력, 명암/질감, 조형완성도, 주제해석력)
    - 가중 평균 점수 산출
    - 신뢰도 점수 계산

[x] T4.3: 라벨 검증기 (2026-01-18 완료)
    - 샘플링 기반 검증
    - 검증 결과 기록

[x] T4.4: 라벨링 스크립트 (2026-01-18 완료 - 파이프라인에 통합)
    - DataPipeline 클래스로 통합
    - 배치 처리 지원
```

#### 기술 접근 방식

```python
# labelers/score_mapper.py
from enum import Enum
from typing import Tuple

class Tier(str, Enum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"

class ScoreMapper:
    """점수 → 티어 매핑"""

    TIER_RANGES = {
        Tier.S: (85, 100),
        Tier.A: (70, 84),
        Tier.B: (50, 69),
        Tier.C: (0, 49)
    }

    def score_to_tier(self, score: float) -> Tier:
        """점수를 티어로 변환"""
        for tier, (min_score, max_score) in self.TIER_RANGES.items():
            if min_score <= score <= max_score:
                return tier
        return Tier.C

    def tier_to_score_range(self, tier: Tier) -> Tuple[float, float]:
        """티어의 점수 범위 반환"""
        return self.TIER_RANGES[tier]

# labelers/auto_labeler.py
class AutoLabeler:
    """자동 라벨링 시스템"""

    def __init__(self, reference_embeddings: dict):
        self.reference_embeddings = reference_embeddings
        self.score_mapper = ScoreMapper()

    def label(self, image_features: dict) -> dict:
        """
        이미지 피처 기반 자동 라벨링

        Args:
            image_features: DINOv2/CLIP 추출 피처

        Returns:
            {
                "tier": "A",
                "score": 75.5,
                "rubric_scores": {...},
                "confidence": 0.85
            }
        """
        # 4축 점수 계산
        rubric_scores = self._calculate_rubric_scores(image_features)

        # 가중 평균
        weights = {
            "composition": 0.25,
            "texture": 0.25,
            "completeness": 0.30,
            "interpretation": 0.20
        }

        final_score = sum(
            rubric_scores[k] * weights[k]
            for k in weights.keys()
        )

        tier = self.score_mapper.score_to_tier(final_score)

        return {
            "tier": tier.value,
            "score": final_score,
            "rubric_scores": rubric_scores,
            "confidence": self._calculate_confidence(rubric_scores)
        }
```

---

### M5: 통합 및 검증 (Final Goal)

#### 목표
- 전체 파이프라인 통합
- 2,000개 이미지 데이터셋 구축
- 품질 검증 및 리포트

#### 태스크 체크리스트

```
[x] T5.1: 파이프라인 통합 (2026-01-18 완료)
    - DataPipeline 클래스로 전체 통합
    - 단계별 연결 테스트
    - 오류 처리 통합

[ ] T5.2: 대량 데이터 처리 (데이터 수집 단계에서 수행 예정)
    - 2,000개 이미지 수집
    - 전처리 일괄 수행
    - 태깅 및 라벨링

[ ] T5.3: 품질 검증 (데이터 수집 단계에서 수행 예정)
    - 샘플링 기반 검증 (10-15%)
    - 과별 분포 확인
    - 티어 분포 확인

[ ] T5.4: 최종 데이터셋 구성 (데이터 수집 단계에서 수행 예정)
    - train/val/test 분할 (80/10/10)
    - 메타데이터 통합
    - 데이터셋 문서화

[ ] T5.5: 리포트 생성 (데이터 수집 단계에서 수행 예정)
    - 수집 통계
    - 품질 메트릭
    - 분포 시각화
```

---

## 3. 의존성 관리

### 외부 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| opencv-python | 4.8+ | 이미지 처리 |
| pillow | 10.0+ | 이미지 I/O |
| albumentations | 1.3+ | 데이터 증강 |
| pydantic | 2.x | 데이터 검증 |
| firebase-admin | 6.0+ | Firebase 연동 |
| torch | 2.1+ | ML 피처 추출 (선택) |
| transformers | 4.35+ | DINOv2, CLIP (선택) |

### requirements-data.txt

```txt
# Core
pydantic>=2.5.0
python-dotenv>=1.0.0

# Image Processing
opencv-python>=4.8.0
pillow>=10.0.0
albumentations>=1.3.0

# Firebase
firebase-admin>=6.2.0

# ML (선택 - 자동 라벨링 사용 시)
# torch>=2.1.0
# transformers>=4.35.0
# timm>=0.9.0

# Utilities
tqdm>=4.66.0
structlog>=23.2.0
click>=8.1.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

---

## 4. 리스크 관리

### 주요 리스크

| ID | 리스크 | 영향도 | 발생 확률 | 대응 방안 |
|----|--------|--------|-----------|-----------|
| R1 | 2,000개 이미지 수집 목표 미달 | High | Medium | 외부 데이터셋 활용, 파트너십 확대 |
| R2 | 저작권 이슈 | High | Medium | 사용 동의 프로세스 강화, 법률 검토 |
| R3 | 과별 불균형 분포 | Medium | High | 데이터 증강, 타겟 수집 |
| R4 | 자동 라벨링 정확도 미달 | Medium | Medium | 수동 검토 비율 증가, 모델 개선 |
| R5 | 처리 시간 초과 | Low | Low | 병렬 처리, 리소스 최적화 |

### 완화 전략

1. **R1 대응**: 공모전 출품작 활용 + 파트너 학원 협력 + 오픈 데이터셋 조사
2. **R2 대응**: 명시적 사용 동의 절차 + 개인정보 익명화 + 법률 자문
3. **R3 대응**: 과별 수집 목표 설정 + 데이터 증강으로 보완
4. **R4 대응**: 15% 샘플 수동 검증 + 전문가 피드백 반영
5. **R5 대응**: 멀티프로세싱 + 배치 크기 최적화

---

## 5. 성공 지표

### 정량적 지표

| 지표 | 목표값 | 측정 방법 |
|------|--------|-----------|
| 수집 이미지 수 | >= 2,000개 | 데이터셋 카운트 |
| 전처리 완료율 | >= 95% | 성공/전체 비율 |
| 태깅 완료율 | 100% | 메타데이터 완성도 |
| 자동 라벨링 정확도 | >= 85% | 샘플 검증 결과 |
| 과별 최소 분포 | >= 15% | 각 과별 비율 |

### 정성적 지표

- 전처리 품질: 시각적 검토 통과
- 메타데이터 일관성: 스키마 검증 통과
- 라벨 신뢰도: 전문가 동의율 >= 85%

---

*계획서 버전: 1.1.0*
*작성일: 2025-01-18*
*최종 수정일: 2026-01-18*
*SPEC 참조: SPEC-DATA-001*

---

## 6. 구현 완료 요약

### 구현된 파일 목록 (24개)

**Models:**
- `backend/data_pipeline/models/__init__.py`
- `backend/data_pipeline/models/metadata.py` - ImageMetadata, Department, Tier

**Collectors:**
- `backend/data_pipeline/collectors/__init__.py`
- `backend/data_pipeline/collectors/base_collector.py` - BaseCollector
- `backend/data_pipeline/collectors/image_collector.py` - ImageCollector

**Preprocessors:**
- `backend/data_pipeline/preprocessors/__init__.py`
- `backend/data_pipeline/preprocessors/base_preprocessor.py` - BasePreprocessor
- `backend/data_pipeline/preprocessors/resize_preprocessor.py` - ResizePreprocessor
- `backend/data_pipeline/preprocessors/normalize_preprocessor.py` - NormalizePreprocessor
- `backend/data_pipeline/preprocessors/augment_preprocessor.py` - AugmentPreprocessor

**Taggers:**
- `backend/data_pipeline/taggers/__init__.py`
- `backend/data_pipeline/taggers/base_tagger.py` - BaseTagger
- `backend/data_pipeline/taggers/department_tagger.py` - DepartmentTagger
- `backend/data_pipeline/taggers/tier_tagger.py` - TierTagger

**Labelers:**
- `backend/data_pipeline/labelers/__init__.py`
- `backend/data_pipeline/labelers/base_labeler.py` - BaseLabeler
- `backend/data_pipeline/labelers/auto_labeler.py` - AutoLabeler

**Storage:**
- `backend/data_pipeline/storage/__init__.py`
- `backend/data_pipeline/storage/base_storage.py` - BaseStorage
- `backend/data_pipeline/storage/local_storage.py` - LocalStorage
- `backend/data_pipeline/storage/metadata_storage.py` - MetadataStorage

**Pipeline:**
- `backend/data_pipeline/__init__.py`
- `backend/data_pipeline/pipeline.py` - DataPipeline
- `backend/data_pipeline/utils/__init__.py`

### 테스트 파일 (8개)
- `backend/tests/data_pipeline/test_models.py`
- `backend/tests/data_pipeline/test_collectors.py`
- `backend/tests/data_pipeline/test_preprocessors.py`
- `backend/tests/data_pipeline/test_taggers.py`
- `backend/tests/data_pipeline/test_labelers.py`
- `backend/tests/data_pipeline/test_storage.py`
- `backend/tests/data_pipeline/test_pipeline.py`
- `backend/tests/data_pipeline/conftest.py`
