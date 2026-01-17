# 미립(MIRIP) 통합 개발 정의서 v2.1

> Claude Code 작업용 최종 명세서
> 사업계획서 기준 정렬 + AI 시스템 전체 스펙 포함

---

## 1. 프로젝트 개요

### 1.1 서비스 정의

| 항목 | 내용 |
|------|------|
| **플랫폼명** | 미립 (MIRIP) |
| **컨셉** | 미술 입시의 모의고사 + 성적표 |
| **핵심 가치** | "내 그림이 목표 대학 기준 어느 위치인가"를 객관적 데이터로 확인 |
| **타겟** | 미술 입시생 (연간 5만 명), 미술학원/화방/지자체 |

### 1.2 Pain Point → Solution 매핑

| # | Pain Point | Solution | 플랫폼 기능 |
|---|------------|----------|-------------|
| 1 | 평가의 주관성 | AI 합격 예측 | **미립에듀** (AI 진단) |
| 2 | 정보 파편화 | Kaggle형 공모전 플랫폼 | **미립콤프** (공모전) |
| 3 | 경력 비표준화 | 예술가용 LinkedIn | **크레덴셜** (포트폴리오) |

### 1.3 Customer Decision Journey (CDJ) 기반 설계

```
┌─────────────────────────────────────────────────────────────────────┐
│  인지 (Awareness)     →   유입 (Acquisition)   →   전환 (Conversion)  │
│  공모전 정보 큐레이션      콘테스트 참가            AI 진단 체험→구독    │
│  (무료)                   (무료/유료)             (건당/월정액)        │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│  리텐션 (Retention)                                                  │
│  프로필(크레덴셜) 축적 → 이탈 비용 증가 → Lock-in                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 타겟 사용자

| 유형 | 설명 | 핵심 니즈 |
|------|------|-----------|
| 입시생 | 미대 입시 준비 중인 학생 | 실력 진단, 합격 가능성 예측 |
| 학부모 | 입시생 자녀를 둔 부모 | 객관적 평가, 비용 효율 |
| 신진 작가 | 데뷔 준비 중인 예술가 | 포트폴리오, 공모전 정보 |
| 주최자 | 공모전/전시 운영자 | 심사 효율화, 작품 관리 |

---

## 2. 기술 스택

### 2.1 프론트엔드

| 구분 | 기술 | 버전 |
|------|------|------|
| Framework | React | 18.x |
| Routing | React Router | 6.x |
| Bundler | Create React App | 5.x |
| Styling | CSS Modules | - |
| State | React Context (→ Zustand 도입 예정) | - |
| Charts | Recharts / Chart.js | - |

### 2.2 백엔드/인프라

| 구분 | 기술 | 용도 |
|------|------|------|
| API Server | FastAPI | ML Inference + REST API |
| Hosting | Firebase Hosting | 정적 파일 서빙 |
| Database | Firebase Firestore | 유저/공모전/결과 데이터 |
| Storage | Firebase Storage | 이미지 저장 |
| Auth | Firebase Auth | 소셜 로그인 (카카오/구글) |
| Cache | Redis | API 캐싱 |
| Payments | 토스페이먼츠 / 카카오페이 | 결제 연동 |

### 2.3 ML/AI 스택

```python
# Core ML
pytorch: 2.1+
transformers: 4.35+  # DINOv2, CLIP
timm: 0.9+

# Data Processing
opencv-python: 4.8+
albumentations: 1.3+
pillow: 10.0+

# Training
pytorch-lightning: 2.1+
wandb: 0.16+  # experiment tracking

# Inference
fastapi: 0.104+
onnxruntime: 1.16+  # 최적화된 inference

# Storage
postgresql: 15+
redis: 7+  # caching
minio: latest  # 이미지 저장
```

---

## 3. 프로젝트 구조

```
mirip/
├── landing/                      # 랜딩 페이지 (Vanilla JS)
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── my-app/                       # 메인 React 앱
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   ├── diagnosis/
│   │   │   ├── competition/
│   │   │   ├── credential/
│   │   │   └── payment/
│   │   │
│   │   ├── pages/
│   │   │   ├── Home/
│   │   │   ├── MiripComp/        # 공모전
│   │   │   ├── MiripEdu/         # AI 진단
│   │   │   ├── Credential/       # 크레덴셜
│   │   │   ├── Payment/          # 결제
│   │   │   └── PreRegister/      # 사전등록
│   │   │
│   │   ├── hooks/
│   │   ├── utils/
│   │   ├── services/
│   │   ├── App.js
│   │   └── index.js
│   │
│   └── package.json
│
├── backend/                      # FastAPI 서버
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── routers/
│   │   │   ├── evaluate.py
│   │   │   ├── compare.py
│   │   │   ├── competition.py
│   │   │   └── credential.py
│   │   ├── services/
│   │   │   ├── inference.py
│   │   │   ├── feedback.py
│   │   │   └── storage.py
│   │   ├── models/
│   │   │   ├── request.py
│   │   │   └── response.py
│   │   └── ml/
│   │       ├── feature_extractor.py
│   │       ├── fusion_module.py
│   │       ├── rubric_heads.py
│   │       ├── tier_classifier.py
│   │       └── weights/
│   │
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── firebase.json
```

---

## 4. 라우트 구조

```
/                           → 홈 (서비스 소개)
/pre-register               → 사전등록

# 공모전 (Phase 1)
/competitions               → 공모전 목록
/competitions/:id           → 공모전 상세
/competitions/:id/submit    → 출품하기
/competitions/:id/result    → 심사 결과

# AI 진단 (Phase 2)
/edu                        → AI 진단 랜딩
/edu/diagnosis              → 진단 페이지
/edu/result/:id             → 결과 페이지
/edu/history                → 진단 이력

# 크레덴셜 (Phase 3)
/profile                    → 내 프로필 편집
/profile/:username          → 공개 프로필
/portfolio                  → 포트폴리오 관리

# 결제 (Phase 3)
/pricing                    → 요금제 안내
/checkout                   → 결제 진행
/checkout/success           → 결제 완료
```

---

## 5. 디자인 시스템

### 5.1 컬러 팔레트

```css
:root {
  /* 배경 */
  --bg-primary: #FAFAFA;         /* 오프화이트 - 주요 배경 */
  --bg-secondary: #F5F3F0;       /* 웜그레이 - 보조 배경 */
  
  /* 텍스트 */
  --text-primary: #1A1A1A;       /* 차콜 블랙 - 제목 텍스트 */
  --text-secondary: #666666;     /* 본문 텍스트 */
  
  /* 포인트 */
  --color-accent: #B8860B;       /* 골드 - 포인트 */
  --color-cta: #8B0000;          /* 딥 레드 - CTA 버튼 */
  
  /* 테두리 */
  --border-color: #D4CFC9;       /* 라이트 베이지 */
  
  /* 티어 컬러 */
  --tier-s: #8b5cf6;             /* 퍼플 */
  --tier-a: #3b82f6;             /* 블루 */
  --tier-b: #22c55e;             /* 그린 */
  --tier-c: #6b7280;             /* 그레이 */
  
  /* 루브릭 컬러 */
  --rubric-composition: #f59e0b;
  --rubric-tone: #8b5cf6;
  --rubric-form: #3b82f6;
  --rubric-theme: #ec4899;
  
  /* 피드백 */
  --feedback-strength: #22c55e;
  --feedback-improve: #f59e0b;
}
```

### 5.2 타이포그래피

```css
/* 한글 제목 */
font-family: 'Noto Serif KR', serif;

/* 본문 */
font-family: 'Pretendard', -apple-system, sans-serif;

/* 영문 포인트 */
font-family: 'Cormorant Garamond', serif;

/* 숫자/점수 */
font-family: 'Roboto Mono', monospace;
```

### 5.3 간격 시스템

```css
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
```

### 5.4 트랜지션

```css
--transition-fast: 0.3s ease;
--transition-slow: 0.6s cubic-bezier(0.4, 0, 0.2, 1);

/* Fade-in 애니메이션 */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
```

### 5.5 반응형 브레이크포인트

```css
/* Mobile First */
--bp-mobile: 480px;
--bp-tablet: 768px;
--bp-laptop: 1024px;
--bp-desktop: 1200px;
--bp-wide: 1440px;

/* 섹션 패딩 반응형 */
/* Desktop: 120px → Tablet: 100px → Mobile: 80px */
```

### 5.6 컴포넌트 패턴

**버튼**
- Primary: 골드(#B8860B) 배경
- CTA: 딥레드(#8B0000) 배경
- Outline: 투명 배경 + 테두리

**카드**
- 오프화이트 배경
- 라이트 베이지 테두리
- 호버 시 미세한 그림자

**폼 요소**
- 라이트 베이지 테두리
- 포커스 시 골드 테두리

---

## 6. 공모전 플랫폼 (미립콤프) - Phase 1

> **목표**: 트래픽 확보, 초기 유저 100명

### 6.1 공모전 목록 페이지

**UI 요구사항**
- 카드 그리드 레이아웃 (반응형: PC 3열, 태블릿 2열, 모바일 1열)
- 필터: 분야(시디/산디/회화/공예), 상태(진행중/마감임박/종료), 주최자
- 정렬: 마감순, 상금순, 인기순
- 무한 스크롤 또는 페이지네이션

**카드 컴포넌트**
```
┌─────────────────────────────┐
│  [썸네일 이미지]             │
├─────────────────────────────┤
│  [주최자 로고] 주최자명       │
│  공모전 제목                 │
│  분야 | 상금 000만원         │
│  D-14 | 조회 1.2K            │
└─────────────────────────────┘
```

**데이터 모델 (Firestore)**
```javascript
// Collection: competitions
{
  id: string,
  title: string,
  description: string,
  thumbnail_url: string,
  
  organizer: {
    id: string,
    name: string,
    logo_url: string,
    type: 'academy' | 'company' | 'government' | 'individual'
  },
  
  category: 'visual_design' | 'industrial_design' | 'fine_art' | 'craft',
  tags: string[],
  
  submission_start: Timestamp,
  submission_end: Timestamp,
  judging_start: Timestamp,
  result_date: Timestamp,
  
  prize: {
    total: number,
    breakdown: [
      { rank: '대상', amount: 1000000, count: 1 },
      { rank: '우수상', amount: 500000, count: 3 }
    ]
  },
  benefits: string[],
  
  requirements: {
    eligible: string,
    format: string[],
    max_size: number,
    max_submissions: number
  },
  
  judging_criteria: [
    { name: '창의성', weight: 30 },
    { name: '완성도', weight: 40 },
    { name: '주제 적합성', weight: 30 }
  ],
  
  stats: {
    views: number,
    submissions: number,
    bookmarks: number
  },
  
  status: 'draft' | 'upcoming' | 'ongoing' | 'judging' | 'completed',
  created_at: Timestamp,
  updated_at: Timestamp
}
```

### 6.2 공모전 상세 페이지

**섹션 구성**
1. 헤더: 썸네일, 제목, 주최자, D-Day 배지
2. 탭: 상세정보 | 출품작 | 결과
3. 상세정보: 소개, 일정, 상금, 출품 조건, 심사 기준
4. 출품작: 공개된 출품작 갤러리
5. 결과: 수상작 발표 (종료 후)

### 6.3 출품 페이지

**플로우**
```
이미지 업로드 → 작품 정보 입력 → 미리보기 → 제출 완료
```

**데이터 모델**
```javascript
// Collection: submissions
{
  id: string,
  competition_id: string,
  user_id: string,
  
  artwork: {
    title: string,
    description: string,
    image_urls: string[],
    thumbnail_url: string
  },
  
  medium: string,
  size: string,
  year: number,
  
  status: 'draft' | 'submitted' | 'under_review' | 'awarded' | 'rejected',
  submitted_at: Timestamp,
  
  judging: {
    scores: { [criteria]: number },
    total_score: number,
    rank: string | null,
    feedback: string | null
  },
  
  is_public: boolean,
  stats: { views: number, likes: number, comments: number }
}
```

---

## 7. AI 평가 시스템 (미립에듀) - Phase 2

> **목표**: AI 평가 모델 PoC, 유료 전환 유도

### 7.1 데이터 구성

#### 7.1.1 데이터 규모 및 수집

| 항목 | 내용 |
|------|------|
| 목표 규모 | 2,000개 (Phase 1), 5,000개+ (Phase 2) |
| 풀링 방식 | 과별 통합 (시각디자인, 산업디자인, 공예 등) |
| 과당 목표 | 400~500개 |

#### 7.1.2 메타데이터 태깅

- 과 (시디/산디/공예/회화 등)
- 대학 티어 (S/A/B/C)
- 주제 키워드
- 연도
- 전형 구분 (수시/정시)
- 매체 (연필/수채/아크릴 등)

#### 7.1.3 라벨링 전략

**Phase 1: 자동 라벨링 (MVP)**

| 티어 | 합격 점수 범위 | 탈락 점수 |
|------|----------------|-----------|
| S (서울대) | 92~96 | 85~88 |
| A (홍대) | 85~90 | 78~82 |
| B (국민대) | 78~84 | 71~75 |
| C (지방대) | 68~75 | 61~65 |

**Phase 2: 점진적 정제**
- B2B 학원 파트너와 협력하여 전문가 피드백 수집
- Hard negative mining으로 경계 케이스 재라벨링
- 이미지당 최소 3명 이상 교차 검증

---

### 7.2 모델 아키텍처

#### 7.2.1 전체 구조

```
                      ┌─────────────────┐
                      │   Input Image   │
                      └────────┬────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                              ▼
       ┌────────────────┐            ┌────────────────┐
       │  DINOv2 ViT-L  │            │    PiDiNet     │
       │   (frozen)     │            │   (frozen)     │
       └───────┬────────┘            └───────┬────────┘
               │                              │
               ▼                              ▼
       ┌────────────────┐            ┌────────────────┐
       │  RGB Projector │            │ Edge Projector │
       │   (trainable)  │            │  (trainable)   │
       └───────┬────────┘            └───────┬────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
                     ┌────────────────┐
                     │  Fusion Layer  │
                     │ (concat + MLP) │
                     └───────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
      ┌─────────┐      ┌─────────┐      ┌─────────┐
      │구성력   │      │명암/질감│      │조형완성 │
      │MLP Head │      │MLP Head │      │MLP Head │
      └─────────┘      └─────────┘      └─────────┘
```

#### 7.2.2 Backbone 선택 근거

| 모델 | 역할 | 선택 이유 |
|------|------|-----------|
| DINOv2 ViT-L | RGB 피처 추출 | visual similarity에서 CLIP 대비 2배 이상 정확도 (64% vs 28%) |
| PiDiNet | Edge 피처 추출 | HED 대비 28% 파라미터로 0.9% ODS 향상, 경량화 |
| CLIP ViT-L | 주제 해석 전용 | text-image alignment 최적화 |

#### 7.2.3 Dual-Branch Fusion

```python
class FusionModule(nn.Module):
    def __init__(self, rgb_dim=1024, edge_dim=256, fusion_dim=512):
        self.rgb_proj = nn.Linear(rgb_dim, fusion_dim)
        self.edge_proj = nn.Linear(edge_dim, fusion_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, rgb_feat, edge_feat):
        rgb = self.rgb_proj(rgb_feat)
        edge = self.edge_proj(edge_feat)
        fused = torch.cat([rgb, edge], dim=-1)
        return self.fusion_mlp(fused)
```

#### 7.2.4 평가 축 정의

| 축 | 설명 | 주요 피처 소스 |
|----|------|----------------|
| 구성력 | 화면 배치, 비례, 균형 | DINOv2 spatial features |
| 명암/질감 | 톤 표현, 재질감, 필압 | DINOv2 + PiDiNet edge |
| 조형 완성도 | 형태 정확성, 마감 | DINOv2 + PiDiNet edge |
| 주제 해석력 | 주제와의 연관성, 창의성 | CLIP text-image similarity |

#### 7.2.5 주제 해석 점수 산출

```python
def compute_theme_score(image_embedding, theme_text):
    text_embedding = clip_text_encoder(theme_text)
    similarity = F.cosine_similarity(image_embedding, text_embedding)
    score = (similarity + 1) / 2 * 100  # 0~100 스케일
    return score
```

#### 7.2.6 과별 가중치

```python
DEPARTMENT_WEIGHTS = {
    'visual_design': {      # 시각디자인
        'composition': 0.25,
        'tone_texture': 0.20,
        'form_completion': 0.20,
        'theme_interpretation': 0.35  # 주제 비중 높음
    },
    'industrial_design': {  # 산업디자인
        'composition': 0.25,
        'tone_texture': 0.25,
        'form_completion': 0.35,  # 조형 비중 높음
        'theme_interpretation': 0.15
    },
    'fine_art': {           # 회화
        'composition': 0.20,
        'tone_texture': 0.35,  # 표현력 비중 높음
        'form_completion': 0.25,
        'theme_interpretation': 0.20
    }
}
```

---

### 7.3 학습 전략

#### 7.3.1 Task 정의

- **Primary**: Pairwise Ranking (margin ranking loss)
- **Auxiliary**: Rubric별 regression (MSE loss)

#### 7.3.2 Pair 생성 전략

**기본 규칙**
- 같은 과 내에서 티어가 다른 조합만 생성
- 라벨: 티어 높은 쪽이 이김 (binary)

**Hard Negative Mining (Phase 2)**
```python
def mine_hard_negatives(embeddings, labels, margin=0.1):
    """
    임베딩 거리가 margin 이내인데 라벨이 다른 페어 우선 샘플링
    """
    hard_pairs = []
    for i, j in itertools.combinations(range(len(embeddings)), 2):
        dist = torch.norm(embeddings[i] - embeddings[j])
        if dist < margin and labels[i] != labels[j]:
            hard_pairs.append((i, j))
    return hard_pairs
```

**예상 페어 수**

| 과 | 티어별 50개 기준 | 총 페어 수 |
|----|------------------|------------|
| 시각디자인 | S vs A: 2,500 | ~10,000+ |
| | S vs B: 2,500 | |
| | A vs B: 2,500 | |

#### 7.3.3 Loss Function

```python
class CombinedLoss(nn.Module):
    def __init__(self, margin=0.2, lambda_reg=0.3):
        self.margin = margin
        self.lambda_reg = lambda_reg
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.regression_loss = nn.MSELoss()
    
    def forward(self, score_a, score_b, label, rubric_pred, rubric_gt):
        # Pairwise ranking loss
        l_rank = self.ranking_loss(score_a, score_b, label)
        
        # Rubric regression loss (auxiliary)
        l_reg = self.regression_loss(rubric_pred, rubric_gt)
        
        return l_rank + self.lambda_reg * l_reg
```

#### 7.3.4 Augmentation 정책

| 허용 | 금지 |
|------|------|
| 밝기 미세 조정 (±3%) | 좌우 반전 |
| 가우시안 노이즈 (σ < 0.01) | 회전 |
| JPEG compression artifact | Crop |
| Color jittering (±3%) | Geometric 변환 |
| Mixup on embedding level | |

#### 7.3.5 학습 설정

```yaml
backbone_freeze: true
optimizer: AdamW
learning_rate: 1e-4
weight_decay: 0.01
scheduler: CosineAnnealingLR
warmup_epochs: 2
total_epochs: 30
batch_size: 32  # pairs
gradient_accumulation: 2
mixed_precision: fp16
```

---

### 7.4 Inference 파이프라인

#### 7.4.1 티어 판정 (GMM 기반)

```python
class TierClassifier:
    def __init__(self, n_tiers=4):
        self.gmm = GaussianMixture(n_components=n_tiers)
        self.tier_names = ['S', 'A', 'B', 'C']
    
    def fit(self, embeddings, tier_labels):
        """학습 데이터로 티어별 분포 학습"""
        self.gmm.fit(embeddings)
        self.tier_mapping = self._map_components_to_tiers(tier_labels)
    
    def predict(self, embedding):
        """새 이미지의 티어 확률 분포 반환"""
        probs = self.gmm.predict_proba(embedding.reshape(1, -1))
        tier_probs = {
            self.tier_names[i]: probs[0][self.tier_mapping[i]] 
            for i in range(len(self.tier_names))
        }
        return tier_probs
    
    def get_tier_range(self, embedding, threshold=0.2):
        """확률 threshold 이상인 티어 범위 반환"""
        probs = self.predict(embedding)
        valid_tiers = [t for t, p in probs.items() if p >= threshold]
        return f"{valid_tiers[-1]}~{valid_tiers[0]}"
```

#### 7.4.2 Inference Flow

```
Input Image
     │
     ▼
┌─────────────────────────────────────────┐
│ 1. Feature Extraction                    │
│    - DINOv2: RGB embedding              │
│    - PiDiNet: Edge embedding            │
│    - CLIP: Theme embedding              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 2. Fusion & Rubric Scoring              │
│    - 구성력: 85점                        │
│    - 명암/질감: 78점                     │
│    - 조형완성도: 82점                    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 3. Theme Score (CLIP)                   │
│    - 주제 해석력: 88점                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 4. Tier Classification (GMM)            │
│    - S: 15%, A: 55%, B: 25%, C: 5%      │
│    - 판정: "A~S 범위, 중상위권"          │
└─────────────────────────────────────────┘
```

#### 7.4.3 Inference Engine 구현

```python
import torch
from transformers import AutoModel, CLIPModel, CLIPProcessor

class InferenceEngine:
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.dino = DINOv2Extractor().to(self.device)
        self.pidinet = PiDiNetExtractor().to(self.device)
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
        
        # Load trained heads
        self.fusion = FusionModule().to(self.device)
        self.rubric_heads = RubricHeads().to(self.device)
        self.tier_classifier = TierClassifier()
        
        # Load weights
        self._load_weights(config.weights_path)
    
    @torch.no_grad()
    def evaluate(self, image, department, theme):
        # 1. Feature extraction
        rgb_feat = self.dino(image)
        edge_feat = self.pidinet(image)
        
        # 2. Fusion
        fused = self.fusion(rgb_feat, edge_feat)
        
        # 3. Rubric scores
        rubric_scores = self.rubric_heads(fused)
        
        # 4. Theme score (CLIP)
        theme_score = self._compute_theme_score(image, theme)
        
        # 5. Weighted total
        weights = DEPARTMENT_WEIGHTS[department]
        weighted_total = self._compute_weighted_total(rubric_scores, theme_score, weights)
        
        # 6. Tier classification
        tier_probs = self.tier_classifier.predict(fused)
        
        return {
            'rubric_scores': rubric_scores,
            'theme_score': theme_score,
            'weighted_total': weighted_total,
            'tier_prediction': tier_probs
        }
```

---

### 7.5 API 명세

#### 7.5.1 단일 이미지 평가

```
POST /api/v1/evaluate
```

**Request**
```json
{
  "image": "base64_encoded_string",
  "department": "visual_design",
  "theme": "자연과 인간의 공존",
  "options": {
    "include_feedback": true,
    "language": "ko"
  }
}
```

**Response**
```json
{
  "success": true,
  "data": {
    "image_id": "eval_20250117_001",
    "tier_prediction": {
      "range": "A~S",
      "description": "홍대 상위권 ~ 서울대 하위권",
      "probabilities": {
        "S": 0.15,
        "A": 0.55,
        "B": 0.25,
        "C": 0.05
      }
    },
    "rubric_scores": {
      "composition": 85,
      "tone_texture": 78,
      "form_completion": 82,
      "theme_interpretation": 88
    },
    "weighted_total": 83.4,
    "confidence": 0.78,
    "feedback": {
      "strengths": [
        "화면 구성의 균형감이 우수합니다",
        "주제에 대한 독창적 해석이 돋보입니다"
      ],
      "improvements": [
        "명암 대비를 더 강화하면 입체감이 살아날 것입니다",
        "세부 묘사의 완성도를 높여보세요"
      ],
      "recommendation": "현재 수준에서 명암 표현력을 보완하면 A티어 진입 가능"
    }
  }
}
```

#### 7.5.2 복수 이미지 비교

```
POST /api/v1/compare
```

**Request**
```json
{
  "images": [
    {"id": "img_001", "image": "base64_1"},
    {"id": "img_002", "image": "base64_2"},
    {"id": "img_003", "image": "base64_3"}
  ],
  "department": "visual_design"
}
```

**Response**
```json
{
  "success": true,
  "data": {
    "comparison_type": "pairwise_ranking",
    "results": [
      {"image_id": "img_003", "rank": 1, "score": 87.2},
      {"image_id": "img_001", "rank": 2, "score": 83.4},
      {"image_id": "img_002", "rank": 3, "score": 79.1}
    ],
    "ranking_confidence": 0.85
  }
}
```

#### 7.5.3 히스토리 조회

```
GET /api/v1/history?user_id={uid}&limit=10
```

---

### 7.6 인프라 및 비용

#### 7.6.1 학습 환경

| 항목 | 스펙 |
|------|------|
| GPU | RTX 4070 Ti Super 16GB |
| 예상 학습 시간 | 4~6시간 (backbone freeze) |
| VRAM 사용량 | ~12GB (batch 32, fp16) |

#### 7.6.2 비용 구조

| 단계 | 비용 | 비고 |
|------|------|------|
| 학습 | 전기세 (~3,000원) | 1회 학습 기준 |
| Inference | 무료 | 로컬 처리 |
| 피드백 생성 | API 비용 (선택) | LLM 호출 시에만 |

#### 7.6.3 Inference 서버 요구사항

```yaml
# Minimum Requirements
cpu: 4 cores
ram: 16GB
gpu: RTX 3060 12GB (또는 동급)
storage: 50GB SSD

# Latency Target
single_image: < 500ms
batch_10: < 2s
```

#### 7.6.4 Docker 설정

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./weights:/app/weights:ro
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_WEIGHTS_PATH=/app/weights
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

### 7.7 검증 계획

#### 7.7.1 정량 평가

| 지표 | 목표 | 측정 방법 |
|------|------|-----------|
| Pairwise Accuracy | 80%+ | 테스트셋 페어 정답률 |
| Tier Classification | 75%+ | 전문가 라벨 대비 일치율 |
| Rubric SRCC | 0.7+ | 전문가 점수와 Spearman 상관 |

#### 7.7.2 정성 평가

- 미술학원 강사 3인 블라인드 테스트
- 학생 피드백 만족도 조사 (5점 척도)
- A/B 테스트: AI 피드백 vs 강사 피드백 선호도

#### 7.7.3 Ablation Study

| 실험 | 비교 대상 |
|------|-----------|
| Backbone | DINOv2 vs CLIP vs DINOv2+CLIP |
| Edge Branch | with PiDiNet vs without |
| Loss Function | Ranking only vs Ranking + Regression |
| Fusion | Early vs Late vs Cross-attention |

---

### 7.8 리스크 및 대응

| 리스크 | 영향 | 대응 |
|--------|------|------|
| 데이터 부족 | 과적합, 일반화 실패 | 과별 최소 300개 확보, augmentation 강화 |
| 라벨 노이즈 | 학습 불안정 | Phase 2에서 전문가 검수, hard negative mining |
| 과별 불균형 | 특정 과 성능 저하 | 과별 별도 head, weighted sampling |
| 주관성 이슈 | 사용자 불만 | 확률 분포로 제시, "참고용" 명시 |

---

## 8. 크레덴셜 시스템 - Phase 3

> **목표**: 리텐션, Lock-in 효과

### 8.1 프로필 데이터 모델

```javascript
// Collection: users
{
  id: string,
  email: string,
  username: string,
  display_name: string,
  avatar_url: string,
  bio: string,
  
  artist_info: {
    category: string[],
    style: string[],
    medium: string[],
    education: [{ school, major, year }],
    career: [{ title, organization, period }]
  },
  
  achievements: [
    {
      type: 'award' | 'exhibition' | 'publication',
      title: string,
      organization: string,
      date: Timestamp,
      proof_url: string,
      verified: boolean
    }
  ],
  
  diagnosis_history: [
    {
      id: string,
      date: Timestamp,
      department: string,
      tier: string,
      total_score: number
    }
  ],
  
  subscription: {
    plan: 'free' | 'basic' | 'pro',
    started_at: Timestamp,
    expires_at: Timestamp
  },
  
  created_at: Timestamp,
  updated_at: Timestamp
}
```

### 8.2 자동 이력 적재

**트리거**
- 공모전 수상 → achievements에 자동 추가 (verified: true)
- AI 진단 완료 → diagnosis_history에 추가
- 외부 전시/수상 → 수동 입력 + 증빙 업로드

---

## 9. 결제 시스템 - Phase 3

### 9.1 요금제

| 구분 | Free | Basic | Pro |
|------|------|-------|-----|
| 가격 | 0원 | 월 9,900원 | 월 29,900원 |
| AI 진단 | 1회 체험 | 월 5회 | 무제한 |
| 상세 피드백 | ❌ | ✅ | ✅ |
| 비교 진단 | ❌ | ❌ | ✅ |
| 이력 저장 | ❌ | 최근 10개 | 무제한 |

**건당 결제**
- AI 진단 1회: 9,900원
- 비교 진단 1회: 14,900원

### 9.2 결제 플로우

```
요금제 선택 → 결제 정보 입력 → 토스페이먼츠 결제 → 
→ Webhook 수신 → 구독 상태 업데이트 → 완료 페이지
```

---

## 10. 개발 로드맵

### Phase 1: 플랫폼 MVP (4주)

**Week 1-2: 기반 구축**
- [ ] Firebase 프로젝트 설정
- [ ] React 프로젝트 구조 정리
- [ ] 공통 컴포넌트 (디자인 시스템 적용)
- [ ] 사전등록 페이지

**Week 3-4: 공모전 MVP**
- [ ] 공모전 목록/상세/출품 페이지
- [ ] 공모전 CRUD API
- [ ] 초기 데이터 시딩

**완료 기준**: 사전등록 100명, 공모전 5개, 출품 10건

---

### Phase 2: AI 진단 PoC (8주)

**Week 1-2: 데이터 파이프라인**
- [ ] 데이터 수집 및 전처리
- [ ] 메타데이터 태깅 시스템
- [ ] 자동 라벨링 적용

**Week 3-4: DINOv2 Baseline**
- [ ] DINOv2 single branch 구현
- [ ] Pairwise ranking 학습
- [ ] 기본 검증

**Week 5-6: Multi-branch 확장**
- [ ] PiDiNet edge branch 추가
- [ ] Fusion module 구현
- [ ] Rubric heads 분리

**Week 7-8: 서비스 연동**
- [ ] FastAPI 서버 구축
- [ ] GMM tier classifier 구현
- [ ] FE-BE 연동
- [ ] 결과 페이지 UI

**완료 기준**: 티어 정확도 75%+, 테스트 진단 50회+

---

### Phase 3: 수익화 + 리텐션 (4주)

**Week 9-10: 크레덴셜**
- [ ] 프로필/포트폴리오 페이지
- [ ] 자동 이력 적재

**Week 11-12: 결제**
- [ ] 토스페이먼츠 연동
- [ ] 구독 관리

**완료 기준**: 유료 전환 10명, 월 매출 30만원

---

### Phase 4: 고도화 (12주)

**Week 1-4: 데이터 확장**
- [ ] 5,000개 데이터 확보
- [ ] 전문가 검수 라벨링

**Week 5-8: 모델 고도화**
- [ ] LLM 피드백 연동
- [ ] Ablation study 완료
- [ ] 티어 정확도 85%+

**Week 9-12: B2B 확장**
- [ ] 학원용 공모전 생성
- [ ] 대시보드 제공

---

## 11. 환경 변수

### Frontend (.env)
```
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_FIREBASE_API_KEY=xxx
REACT_APP_FIREBASE_AUTH_DOMAIN=xxx
REACT_APP_FIREBASE_PROJECT_ID=xxx
REACT_APP_FIREBASE_STORAGE_BUCKET=xxx
REACT_APP_TOSS_CLIENT_KEY=xxx
```

### Backend (.env)
```
HOST=0.0.0.0
PORT=8000
DEBUG=true
MODEL_WEIGHTS_PATH=/app/weights
DEVICE=cuda
OPENAI_API_KEY=xxx
REDIS_URL=redis://localhost:6379
FIREBASE_CREDENTIALS_PATH=/app/credentials/firebase.json
```

---

## 12. 체크리스트

### 개발 시작 전
- [ ] Node.js 18+, Python 3.10+
- [ ] Firebase 프로젝트 생성
- [ ] 토스페이먼츠 테스트 계정

### ML 모델 가중치
- [ ] DINOv2 ViT-L (transformers 자동)
- [ ] PiDiNet pretrained weights
- [ ] CLIP ViT-L (transformers 자동)
- [ ] Trained fusion/rubric heads

### Phase별 완료 확인
- [ ] Phase 1: 공모전 MVP 동작
- [ ] Phase 2: AI 진단 API 응답 < 3초
- [ ] Phase 3: 결제 플로우 테스트

---

## 13. 참고 자료

### ML 모델
- DINOv2: https://github.com/facebookresearch/dinov2
- PiDiNet: https://github.com/hellozhuo/pidinet
- CLIP: https://github.com/openai/CLIP

### 기술 문서
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- Firebase: https://firebase.google.com/docs
- 토스페이먼츠: https://docs.tosspayments.com/

---

**문서 버전**: 2.1
**최종 수정**: 2025-01-17
**변경 사항**:
- 디자인 시스템 사용자 제안서 기준으로 수정
- AI 프로덕트 개발부 전체 복원 (데이터/모델/학습/Inference/검증)