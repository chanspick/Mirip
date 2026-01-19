# MIRIP 개발 로드맵

> 최종 업데이트: 2026-01-20

---

## 📊 전체 진행 현황

```
Phase 1 (Foundation)     ████████████████████ 100%  ✅ 완료
Phase B (Integration)    ████████████████████ 100%  ✅ 완료
Phase 3 (Credential)     ████████████████████ 100%  ✅ 완료
Phase A (ML Training)    ████░░░░░░░░░░░░░░░░  20%  🔧 인프라 준비
Phase C (Launch)         ░░░░░░░░░░░░░░░░░░░░   0%  ⏳ 대기
```

---

## ✅ Phase 1: Foundation (완료)

### 완료된 SPEC

| SPEC ID | 제목 | 상태 | 완료일 |
|---------|------|------|--------|
| SPEC-UI-001 | 디자인 시스템 기반 공통 컴포넌트 | ✅ Completed | 2025-01-17 |
| SPEC-FIREBASE-001 | Firebase 연동 및 사전등록 | ✅ Completed | 2025-01-17 |
| SPEC-COMP-001 | 공모전 MVP (목록/상세/출품) | ✅ Completed | 2025-01-17 |
| SPEC-BACKEND-001 | FastAPI Backend 초기 셋업 | ✅ Completed | 2025-01-17 |
| SPEC-DATA-001 | AI 진단 데이터 파이프라인 | ✅ Completed | 2026-01-18 |
| SPEC-AI-001 | DINOv2 Baseline AI 모델 | ✅ Completed | 2026-01-18 |

### 주요 산출물

- **Frontend**: Landing, Competition (List/Detail/Submit), AI Diagnosis 페이지
- **Backend**: FastAPI 프로젝트 구조, 데이터 파이프라인 모듈
- **ML**: DINOv2 Feature Extractor, Pairwise Ranking Model 구조

---

## ✅ Phase B: Service Integration (완료)

### B-1: 프론트엔드 프로토타입 ✅ 완료

| 작업 | 상태 | 설명 |
|------|------|------|
| Landing 페이지 업데이트 | ✅ 완료 | 사전등록 제거, CTA 카드 추가 |
| AI 진단 페이지 생성 | ✅ 완료 | 이미지 업로드, Mock 결과 표시 |
| 네비게이션 통합 | ✅ 완료 | 전체 페이지 네비게이션 일관성 |
| 디자인 개선 | ✅ 완료 | 프로토타입 배너 제거, UI 수정 |

### B-2: Backend API 구현 ✅ 완료

| 작업 | 상태 | 설명 |
|------|------|------|
| `/api/v1/evaluate` 엔드포인트 | ✅ 완료 | 단일 이미지 AI 평가 |
| `/api/v1/compare` 엔드포인트 | ✅ 완료 | 복수 이미지 비교 |
| DINOv2 추론 서비스 | ✅ 완료 | Feature extraction + Scoring |
| Mock 피드백 생성 | ✅ 완료 | 점수 기반 정적 피드백 |

### B-3: FE-BE 연동 ✅ 완료

| 작업 | 상태 | 설명 |
|------|------|------|
| API 클라이언트 작성 | ✅ 완료 | fetch 기반 diagnosisService |
| Mock → Real API 전환 | ✅ 완료 | DiagnosisPage API 연결 |
| 에러 핸들링 | ✅ 완료 | 네트워크 오류, 타임아웃 처리 |
| 로딩 상태 개선 | ✅ 완료 | 진행 상태별 메시지 표시 |

---

## ✅ Phase 3: Credential System (크레덴셜) - 완료

### SPEC-CRED-001: 마이페이지 + 공개 프로필 + GitHub 잔디밭

| 마일스톤 | 우선순위 | 상태 | 설명 |
|----------|----------|------|------|
| M1: 데이터 모델 및 서비스 | PRIMARY | ✅ 완료 | Firestore 스키마, 서비스 레이어 |
| M2: 마이페이지 (잔디밭) | PRIMARY | ✅ 완료 | ActivityHeatmap, ActivityTimeline |
| M3: 공개 프로필 | SECONDARY | ✅ 완료 | PublicProfile, ProfileCard, TierBadge |
| M4: 포트폴리오 관리 | SECONDARY | ✅ 완료 | Portfolio CRUD, 이미지 업로드 |
| M5: 기존 시스템 연동 | FINAL | ✅ 완료 | 진단/공모전 → 활동 기록 자동화 |

### 주요 산출물

- **컴포넌트**: ActivityHeatmap, ActivityTimeline, StreakDisplay, ProfileCard, TierBadge, AchievementList, PortfolioCard, PortfolioGrid, PortfolioUploadForm, PortfolioModal
- **서비스**: credentialService, activityService, portfolioService, awardService, integrationService
- **Hooks**: useUserProfile, useActivities, usePortfolios, useAwards, useAuth
- **페이지**: /profile (마이페이지), /profile/:username (공개 프로필), /portfolio (포트폴리오 관리)
- **테스트**: 547+ 테스트 통과 (TDD 방식)

---

## 🔧 Phase A: Multi-branch Model (로컬 학습)

> **전제조건**: RTX 4070 Ti Super 16GB GPU

### A-0: 학습 인프라 (완료) ✅

| 구성요소 | 상태 | 파일 |
|---------|------|------|
| 메인 학습 스크립트 | ✅ 완료 | `training/scripts/train.py` |
| 평가 스크립트 | ✅ 완료 | `training/scripts/evaluate.py` |
| 데이터 준비 스크립트 | ✅ 완료 | `training/scripts/prepare_data.py` |
| PairwiseRankingModel | ✅ 완료 | `app/ml/ranking_model.py` |
| Trainer (AdamW, 조기종료) | ✅ 완료 | `training/trainer.py` |
| Evaluator | ✅ 완료 | `training/evaluator.py` |

### A-1: 데이터 수집

| 작업 | 상태 | 목표 |
|------|------|------|
| 공모전 출품작 수집 | ⏳ 대기 | 500-1,000개 |
| 파트너 학원 데이터 | ⏳ 대기 | 500-700개 |
| 외부 데이터셋 | ⏳ 대기 | 200-300개 |
| **총 목표** | - | **2,000개** |

### A-2: 모델 학습

| 작업 | 상태 | 설명 |
|------|------|------|
| Pairwise 데이터 생성 | 🔧 준비됨 | `generate_pairs.py` 스크립트 완료 |
| DINOv2 Projector 학습 | 🔧 준비됨 | `train.py` 스크립트 완료 |
| 검증 및 평가 | 🔧 준비됨 | `evaluate.py` 스크립트 완료 |
| 체크포인트 저장 | 🔧 준비됨 | Trainer에 구현됨 |

### A-3: 모델 배포

| 작업 | 상태 | 설명 |
|------|------|------|
| 추론 서비스 통합 | ⏳ 대기 | 학습된 가중치 로드 |
| 성능 최적화 | ⏳ 대기 | fp16, 배치 처리 |
| A/B 테스트 | ⏳ 대기 | Mock vs Real 비교 |

---

## ⏳ Phase C: Launch (배포)

### C-1: 배포 준비

| 작업 | 상태 | 설명 |
|------|------|------|
| Firebase Hosting 설정 | ⏳ 대기 | 프론트엔드 배포 |
| Backend 배포 (Cloud Run/GCE) | ⏳ 대기 | GPU 인스턴스 설정 |
| 도메인 연결 | ⏳ 대기 | mirip.kr (예정) |
| SSL 인증서 | ⏳ 대기 | HTTPS 설정 |

### C-2: 출시 전 체크리스트

| 항목 | 상태 |
|------|------|
| [ ] 전체 기능 테스트 |
| [ ] 모바일 반응형 테스트 |
| [ ] 성능 테스트 (Lighthouse) |
| [ ] 보안 검토 |
| [ ] 개인정보처리방침 |
| [ ] 이용약관 |

---

## 📁 프로젝트 구조

```
Mirip/
├── my-app/                    # React Frontend
│   ├── src/
│   │   ├── components/        # UI 컴포넌트
│   │   │   ├── common/        # 공통 컴포넌트 (Header, Footer, Button 등)
│   │   │   ├── credential/    # 크레덴셜 컴포넌트 (Phase 3)
│   │   │   │   ├── ActivityHeatmap/   # GitHub 잔디밭 스타일
│   │   │   │   ├── ActivityTimeline/  # 활동 타임라인
│   │   │   │   ├── StreakDisplay/     # 연속 활동 표시
│   │   │   │   ├── ProfileCard/       # 프로필 카드
│   │   │   │   ├── TierBadge/         # 등급 배지 (S/A/B/C)
│   │   │   │   ├── AchievementList/   # 수상 내역
│   │   │   │   └── Portfolio*/        # 포트폴리오 관련
│   │   │   └── competitions/  # 공모전 컴포넌트
│   │   ├── pages/             # 페이지 컴포넌트
│   │   │   ├── Landing/       # 랜딩 페이지
│   │   │   ├── competitions/  # 공모전 페이지들
│   │   │   ├── diagnosis/     # AI 진단 페이지
│   │   │   ├── Profile/       # 마이페이지 (/profile)
│   │   │   ├── PublicProfile/ # 공개 프로필 (/profile/:username)
│   │   │   └── Portfolio/     # 포트폴리오 관리 (/portfolio)
│   │   ├── hooks/             # 커스텀 훅
│   │   │   ├── useUserProfile.js
│   │   │   ├── useActivities.js
│   │   │   ├── usePortfolios.js
│   │   │   ├── useAwards.js
│   │   │   └── useAuth.js
│   │   ├── services/          # API 서비스
│   │   │   ├── credentialService.js   # 사용자 프로필
│   │   │   ├── activityService.js     # 활동 기록
│   │   │   ├── portfolioService.js    # 포트폴리오
│   │   │   ├── awardService.js        # 수상 내역
│   │   │   └── integrationService.js  # 시스템 연동
│   │   ├── types/             # 타입 정의
│   │   ├── utils/             # 유틸리티
│   │   └── config/            # 설정 파일
│   └── public/
│
├── backend/                   # FastAPI Backend
│   ├── app/
│   │   ├── routers/           # API 라우터
│   │   ├── services/          # 비즈니스 로직
│   │   ├── models/            # Pydantic 모델
│   │   └── ml/                # ML 모듈
│   ├── data_pipeline/         # 데이터 파이프라인
│   └── training/              # 학습 스크립트
│
└── .moai/                     # MoAI-ADK 설정
    ├── specs/                 # SPEC 문서
    ├── config/                # 프로젝트 설정
    └── roadmap.md             # 이 문서
```

---

## 🎯 현재 우선순위

1. **즉시**: Phase A-1 데이터 수집 (목표 2,000개)
   - 이미지를 티어별(S/A/B/C) 폴더에 정리
   - `python training/scripts/prepare_data.py --input_dir data/images --output_csv data/metadata.csv --tier_mode directory`
2. **다음**: Phase A-2 모델 학습 (Pairwise Ranking)
   - `python training/scripts/train.py --metadata_csv data/metadata.csv --output_dir checkpoints/ --epochs 100 --device cuda`
3. **이후**: Phase A-3 모델 배포 및 Phase C 출시 준비

---

## 📖 학습 스크립트 사용법

### 1. 데이터 준비
```bash
# 이미지 디렉토리 구조: data/images/{S,A,B,C}/*.jpg
python training/scripts/prepare_data.py \
    --input_dir data/images \
    --output_csv data/metadata.csv \
    --tier_mode directory \
    --validate
```

### 2. 모델 학습
```bash
python training/scripts/train.py \
    --metadata_csv data/metadata.csv \
    --output_dir checkpoints/ \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --device cuda \
    --wandb_project mirip-training
```

### 3. 모델 평가
```bash
python training/scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_csv checkpoints/test_metadata.csv \
    --benchmark
```

---

## 📝 참고

### 기술 스택

| 영역 | 기술 |
|------|------|
| Frontend | React 18, CSS Modules, React Router 6 |
| Backend | FastAPI, Pydantic, Uvicorn |
| ML | PyTorch, DINOv2, Transformers |
| Database | Firebase Firestore |
| Storage | Firebase Storage |
| Deployment | Firebase Hosting, Cloud Run (예정) |

### 관련 SPEC 문서

- `.moai/specs/SPEC-UI-001/` - UI 컴포넌트
- `.moai/specs/SPEC-COMP-001/` - 공모전 시스템
- `.moai/specs/SPEC-BACKEND-001/` - Backend 셋업
- `.moai/specs/SPEC-DATA-001/` - 데이터 파이프라인
- `.moai/specs/SPEC-AI-001/` - AI 모델
- `.moai/specs/SPEC-CRED-001/` - 크레덴셜 시스템 (마이페이지, 공개 프로필, 포트폴리오)

---

## 📜 커밋 히스토리

### Phase 3 (완료)
| 커밋 | 설명 | 날짜 |
|------|------|------|
| `170849f` | feat(credential): SPEC-CRED-001 크레덴셜 시스템 구현 | 2026-01-20 |
| `dff0607` | docs: SPEC-CRED-001 완료 상태 업데이트 | 2026-01-20 |

### Phase A (진행중)
| 커밋 | 설명 | 날짜 |
|------|------|------|
| `32ec817` | feat(training): Phase A 학습 인프라 완료 | 2026-01-19 |

### Phase B (완료)
| 커밋 | 설명 | 날짜 |
|------|------|------|
| `0556b9a` | docs: Phase B 완료 - 로드맵 업데이트 | 2026-01-19 |
| `3c8c53b` | feat(frontend): Phase B-3 FE-BE 연동 구현 | 2026-01-19 |
| `7fc6bbc` | feat(backend): Phase B-2 Backend API 구현 완료 | 2026-01-19 |
| `eec0b85` | feat(frontend): Phase B-1 디자인 개선 및 로드맵 추가 | 2026-01-19 |

### Phase 1 (완료)
| 커밋 | 설명 | 날짜 |
|------|------|------|
| `c4c1c03` | feat(ai): SPEC-AI-001 DINOv2 Baseline AI 모델 구현 | 2026-01-18 |
| `ed28f99` | docs(SPEC-DATA-001): 문서 동기화 완료 | 2026-01-18 |
| `c7445bb` | test(data-pipeline): SPEC-DATA-001 단위 테스트 추가 | 2026-01-18 |
| `e171af9` | feat(data-pipeline): 스토리지 및 통합 파이프라인 구현 | 2026-01-18 |
| `a30f20c` | feat(firebase): SPEC-FIREBASE-001 Firebase 연동 | 2025-01-17 |

---

*문서 버전: 4.0.0*
*작성자: MoAI-ADK*
*마지막 업데이트: 2026-01-20*

---

## 🔄 다음 세션 이어하기

### 현재 완료된 작업
- ✅ Phase 1 (Foundation) - 6개 SPEC 완료
- ✅ Phase B (Integration) - FE-BE 연동 완료
- ✅ Phase 3 (Credential) - SPEC-CRED-001 완료 (83개 파일, 547+ 테스트)

### 다음 우선순위 작업
1. **Phase A-1: 데이터 수집** - 학습용 이미지 2,000개 수집
   ```bash
   # 이미지를 티어별 폴더에 정리 후:
   cd backend
   python training/scripts/prepare_data.py --input_dir data/images --output_csv data/metadata.csv --tier_mode directory
   ```

2. **Phase A-2: 모델 학습** - Pairwise Ranking 모델 학습
   ```bash
   python training/scripts/train.py --metadata_csv data/metadata.csv --output_dir checkpoints/ --epochs 100 --device cuda
   ```

### 로컬 개발 서버
```bash
cd my-app && npm start  # http://localhost:3000
```

### 새로 추가된 페이지
- `/profile` - 마이페이지 (GitHub 잔디밭 스타일 활동 히트맵)
- `/profile/:username` - 공개 프로필
- `/portfolio` - 포트폴리오 관리
