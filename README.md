# MIRIP - AI 미술 진단 서비스

AI 기반 미술 작품 진단 및 대학 입시 컨설팅 서비스

## 프로젝트 구조

```
Mirip/
├── my-app/                 # React 프론트엔드
│   ├── src/
│   │   ├── components/     # 공통 UI 컴포넌트
│   │   ├── pages/          # 페이지 컴포넌트
│   │   ├── services/       # API 서비스
│   │   └── config/         # 설정 (Firebase 등)
│   └── package.json
│
├── backend/                # Python 백엔드
│   ├── data_pipeline/      # AI 진단 데이터 파이프라인
│   │   ├── models/         # 메타데이터 모델
│   │   ├── collectors/     # 데이터 수집기
│   │   ├── preprocessors/  # 이미지 전처리기
│   │   ├── taggers/        # 과별/티어 태거
│   │   ├── labelers/       # 자동 라벨러
│   │   ├── storage/        # 저장소
│   │   └── pipeline.py     # 통합 파이프라인
│   └── tests/              # 테스트
│
└── .moai/                  # MoAI-ADK 프로젝트 관리
    ├── specs/              # SPEC 문서
    ├── config/             # 프로젝트 설정
    └── reports/            # 동기화 리포트
```

## 기술 스택

### 프론트엔드
- React 18
- Firebase (Authentication, Firestore, Storage)
- CSS Modules

### 백엔드 (데이터 파이프라인)
- Python 3.10+
- Pydantic 2.x (데이터 검증)
- OpenCV / Pillow (이미지 처리)
- PyTorch / Transformers (ML, 선택)

## 설치 및 실행

### 프론트엔드

```bash
cd my-app
npm install
npm start
```

### 백엔드 (데이터 파이프라인)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 데이터 파이프라인 (SPEC-DATA-001)

AI 진단 학습을 위한 데이터 파이프라인입니다.

### 주요 기능

- **데이터 수집**: 로컬/Firebase에서 이미지 수집
- **전처리**: 크기 정규화, 형식 변환, 품질 필터링
- **태깅**: 과별(시디/산디/공예/회화) 및 티어(S/A/B/C) 분류
- **자동 라벨링**: 4축 점수 계산 (구성력, 명암/질감, 조형완성도, 주제해석력)
- **저장**: 로컬 파일 시스템 및 메타데이터 JSON 저장

### 구현 현황

| 컴포넌트 | 상태 | 테스트 커버리지 |
|----------|------|-----------------|
| Models | 완료 | 90% |
| Collectors | 완료 | 90% |
| Preprocessors | 완료 | 90% |
| Taggers | 완료 | 90% |
| Labelers | 완료 | 90% |
| Storage | 완료 | 90% |
| Pipeline | 완료 | 90% |

**테스트 결과**: 168/168 통과 (100%)

## AI 평가 모델 (SPEC-AI-001)

DINOv2 기반 미술 작품 평가 모델입니다.

### 모델 아키텍처

- **Feature Extractor**: DINOv2 ViT-L (frozen backbone, 1024-d 출력)
- **Projector**: 1024 → 512 → 256 (LayerNorm, GELU, Dropout 0.1)
- **Score Head**: Linear (256 → 1)
- **Loss**: MarginRankingLoss (margin=1.0)

### 학습 설정

- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: CosineAnnealingLR
- **Early Stopping**: patience=10
- **데이터 분할**: 80/10/10 (Stratified by tier)

### 구현 현황

| 컴포넌트 | 상태 | 설명 |
|----------|------|------|
| FeatureExtractor | 완료 | DINOv2 ViT-L 모델 로드 |
| Projector | 완료 | 차원 축소 및 정규화 |
| PairwiseRankingModel | 완료 | 통합 모델 |
| PairwiseDataset | 완료 | 페어 생성 및 분할 |
| Trainer | 완료 | 학습 루프 및 체크포인트 |

**목표 정확도**: Pairwise Accuracy 60%+

## SPEC 문서

- `SPEC-DATA-001`: AI 진단 데이터 파이프라인 (완료)
- `SPEC-AI-001`: DINOv2 Baseline AI 평가 모델 (완료)
- `SPEC-COMP-001`: 공모전 시스템
- `SPEC-BACKEND-001`: 백엔드 API

## 라이선스

Private - All Rights Reserved
