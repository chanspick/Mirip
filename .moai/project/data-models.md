# MIRIP 데이터 모델

> 미립(MIRIP) Firestore 컬렉션 스키마 정의

---

## 1. 데이터베이스 개요

### 사용 기술

- **Database**: Firebase Firestore (NoSQL)
- **Storage**: Firebase Storage (이미지)
- **Auth**: Firebase Authentication

### 컬렉션 목록

| 컬렉션 | 용도 | Phase |
|--------|------|-------|
| `users` | 사용자 프로필 및 구독 정보 | 1 |
| `competitions` | 공모전 정보 | 1 |
| `submissions` | 공모전 출품작 | 1 |
| `evaluations` | AI 진단 결과 | 2 |
| `subscriptions` | 결제/구독 정보 | 3 |

---

## 2. users 컬렉션

사용자 계정 및 크레덴셜 정보를 저장합니다.

### 스키마

```javascript
// Collection: users
// Document ID: Firebase Auth UID
{
  // 기본 정보
  id: string,                    // Firebase Auth UID
  email: string,                 // 이메일 주소
  username: string,              // 고유 사용자명 (URL용)
  display_name: string,          // 표시 이름
  avatar_url: string,            // 프로필 이미지 URL
  bio: string,                   // 자기소개 (최대 500자)

  // 아티스트 정보
  artist_info: {
    category: string[],          // 분야 ['visual_design', 'fine_art']
    style: string[],             // 작업 스타일
    medium: string[],            // 주 재료 ['연필', '수채', '아크릴']
    education: [
      {
        school: string,          // 학교명
        major: string,           // 전공
        year: number             // 졸업/재학 연도
      }
    ],
    career: [
      {
        title: string,           // 직함
        organization: string,    // 기관명
        period: string           // 기간 ('2020-2023')
      }
    ]
  },

  // 수상/전시 이력
  achievements: [
    {
      type: 'award' | 'exhibition' | 'publication',
      title: string,             // 수상/전시명
      organization: string,      // 주최 기관
      date: Timestamp,           // 날짜
      proof_url: string,         // 증빙 URL
      verified: boolean          // 검증 여부
    }
  ],

  // AI 진단 이력 (최근 요약)
  diagnosis_history: [
    {
      id: string,                // 진단 결과 ID
      date: Timestamp,           // 진단 일시
      department: string,        // 학과
      tier: string,              // 판정 티어
      total_score: number        // 종합 점수
    }
  ],

  // 구독 정보
  subscription: {
    plan: 'free' | 'basic' | 'pro',
    started_at: Timestamp,
    expires_at: Timestamp
  },

  // 메타 정보
  created_at: Timestamp,
  updated_at: Timestamp
}
```

### 필드 설명

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `id` | string | O | Firebase Auth UID |
| `email` | string | O | 이메일 주소 (유니크) |
| `username` | string | O | 사용자명 (유니크, URL용) |
| `display_name` | string | O | 화면 표시 이름 |
| `avatar_url` | string | X | 프로필 이미지 |
| `bio` | string | X | 자기소개 (최대 500자) |
| `artist_info` | object | X | 아티스트 상세 정보 |
| `achievements` | array | X | 수상/전시 이력 |
| `diagnosis_history` | array | X | AI 진단 이력 (최근 10개) |
| `subscription` | object | O | 구독 상태 |
| `created_at` | Timestamp | O | 가입 일시 |
| `updated_at` | Timestamp | O | 수정 일시 |

### 인덱스

- `email` (unique)
- `username` (unique)
- `subscription.plan` + `created_at`

### 보안 규칙

```javascript
match /users/{userId} {
  allow read: if request.auth != null;
  allow write: if request.auth.uid == userId;
}
```

---

## 3. competitions 컬렉션

공모전 정보를 저장합니다.

### 스키마

```javascript
// Collection: competitions
// Document ID: auto-generated
{
  id: string,
  title: string,                 // 공모전 제목
  description: string,           // 상세 설명
  thumbnail_url: string,         // 썸네일 이미지

  // 주최자 정보
  organizer: {
    id: string,                  // 주최자 user_id
    name: string,                // 주최자명
    logo_url: string,            // 로고 이미지
    type: 'academy' | 'company' | 'government' | 'individual'
  },

  // 분류
  category: 'visual_design' | 'industrial_design' | 'fine_art' | 'craft',
  tags: string[],                // 태그 목록

  // 일정
  submission_start: Timestamp,   // 접수 시작
  submission_end: Timestamp,     // 접수 마감
  judging_start: Timestamp,      // 심사 시작
  result_date: Timestamp,        // 결과 발표

  // 상금
  prize: {
    total: number,               // 총 상금
    breakdown: [
      {
        rank: string,            // '대상', '우수상'
        amount: number,          // 상금액
        count: number            // 인원
      }
    ]
  },
  benefits: string[],            // 기타 혜택

  // 출품 조건
  requirements: {
    eligible: string,            // 참가 자격
    format: string[],            // 파일 형식 ['jpg', 'png']
    max_size: number,            // 최대 파일 크기 (MB)
    max_submissions: number      // 최대 출품 수
  },

  // 심사 기준
  judging_criteria: [
    {
      name: string,              // '창의성'
      weight: number             // 가중치 (%)
    }
  ],

  // 통계
  stats: {
    views: number,               // 조회수
    submissions: number,         // 출품 수
    bookmarks: number            // 북마크 수
  },

  // 상태
  status: 'draft' | 'upcoming' | 'ongoing' | 'judging' | 'completed',

  // 메타
  created_at: Timestamp,
  updated_at: Timestamp
}
```

### 필드 설명

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `title` | string | O | 공모전 제목 (최대 100자) |
| `description` | string | O | 상세 설명 |
| `thumbnail_url` | string | O | 썸네일 이미지 URL |
| `organizer` | object | O | 주최자 정보 |
| `category` | string | O | 분야 카테고리 |
| `tags` | array | X | 태그 목록 (최대 10개) |
| `submission_start` | Timestamp | O | 접수 시작일 |
| `submission_end` | Timestamp | O | 접수 마감일 |
| `prize` | object | O | 상금 정보 |
| `requirements` | object | O | 출품 조건 |
| `judging_criteria` | array | O | 심사 기준 (합계 100) |
| `stats` | object | O | 통계 정보 |
| `status` | string | O | 공모전 상태 |

### 상태 전이

```
draft -> upcoming -> ongoing -> judging -> completed
```

### 인덱스

- `status` + `submission_end` (DESC)
- `category` + `status`
- `organizer.id` + `created_at` (DESC)
- `stats.submissions` (DESC)

---

## 4. submissions 컬렉션

공모전 출품작 정보를 저장합니다.

### 스키마

```javascript
// Collection: submissions
// Document ID: auto-generated
{
  id: string,
  competition_id: string,        // 공모전 ID (FK)
  user_id: string,               // 출품자 ID (FK)

  // 작품 정보
  artwork: {
    title: string,               // 작품 제목
    description: string,         // 작품 설명
    image_urls: string[],        // 이미지 URL 목록
    thumbnail_url: string        // 썸네일 URL
  },

  // 작품 메타
  medium: string,                // 재료/기법
  size: string,                  // 크기 ('100x80cm')
  year: number,                  // 제작 연도

  // 상태
  status: 'draft' | 'submitted' | 'under_review' | 'awarded' | 'rejected',
  submitted_at: Timestamp,       // 제출 일시

  // 심사 결과
  judging: {
    scores: {                    // 기준별 점수
      [criteria_name]: number
    },
    total_score: number,         // 총점
    rank: string | null,         // 수상 등급 ('대상', '우수상')
    feedback: string | null      // 심사평
  },

  // 공개 설정
  is_public: boolean,            // 갤러리 공개 여부

  // 통계
  stats: {
    views: number,
    likes: number,
    comments: number
  },

  // 메타
  created_at: Timestamp,
  updated_at: Timestamp
}
```

### 필드 설명

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `competition_id` | string | O | 공모전 ID |
| `user_id` | string | O | 출품자 ID |
| `artwork` | object | O | 작품 정보 |
| `artwork.title` | string | O | 작품 제목 (최대 50자) |
| `artwork.image_urls` | array | O | 이미지 URL (최대 5개) |
| `medium` | string | X | 재료/기법 |
| `size` | string | X | 작품 크기 |
| `year` | number | X | 제작 연도 |
| `status` | string | O | 출품 상태 |
| `judging` | object | X | 심사 결과 (심사 후) |
| `is_public` | boolean | O | 갤러리 공개 여부 |

### 상태 전이

```
draft -> submitted -> under_review -> awarded / rejected
```

### 인덱스

- `competition_id` + `status`
- `user_id` + `created_at` (DESC)
- `competition_id` + `judging.total_score` (DESC)

---

## 5. evaluations 컬렉션

AI 진단 결과를 저장합니다.

### 스키마

```javascript
// Collection: evaluations
// Document ID: auto-generated
{
  id: string,
  user_id: string,               // 사용자 ID (FK)

  // 입력 정보
  input: {
    image_url: string,           // 원본 이미지 URL
    thumbnail_url: string,       // 썸네일 URL
    department: string,          // 학과
    theme: string | null         // 주제 (선택)
  },

  // 티어 예측
  tier_prediction: {
    range: string,               // 'A~S'
    description: string,         // 설명
    probabilities: {
      S: number,
      A: number,
      B: number,
      C: number
    }
  },

  // 루브릭 점수
  rubric_scores: {
    composition: number,         // 구성력 (0-100)
    tone_texture: number,        // 명암/질감 (0-100)
    form_completion: number,     // 조형완성도 (0-100)
    theme_interpretation: number // 주제해석력 (0-100)
  },

  // 종합
  weighted_total: number,        // 가중 평균 점수
  confidence: number,            // 예측 신뢰도

  // 피드백
  feedback: {
    strengths: string[],         // 강점
    improvements: string[],      // 개선점
    recommendation: string       // 종합 추천
  } | null,

  // 메타
  created_at: Timestamp,
  processing_time_ms: number     // 처리 시간
}
```

### 필드 설명

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `user_id` | string | O | 사용자 ID |
| `input` | object | O | 입력 정보 |
| `tier_prediction` | object | O | 티어 예측 결과 |
| `rubric_scores` | object | O | 루브릭별 점수 |
| `weighted_total` | number | O | 종합 점수 |
| `confidence` | number | O | 신뢰도 (0-1) |
| `feedback` | object | X | 상세 피드백 (유료) |
| `created_at` | Timestamp | O | 생성 일시 |
| `processing_time_ms` | number | O | 처리 시간 (ms) |

### 인덱스

- `user_id` + `created_at` (DESC)
- `input.department` + `created_at` (DESC)

---

## 6. subscriptions 컬렉션

결제 및 구독 정보를 저장합니다.

### 스키마

```javascript
// Collection: subscriptions
// Document ID: auto-generated
{
  id: string,
  user_id: string,               // 사용자 ID (FK)

  // 요금제
  plan: 'free' | 'basic' | 'pro',

  // 결제 정보
  payment: {
    provider: 'toss' | 'kakao',  // 결제 수단
    payment_key: string,         // 결제 키
    order_id: string,            // 주문 ID
    amount: number,              // 결제 금액
    method: string               // 결제 방법 ('카드', '계좌이체')
  },

  // 기간
  started_at: Timestamp,
  expires_at: Timestamp,

  // 상태
  status: 'active' | 'cancelled' | 'expired',
  auto_renew: boolean,           // 자동 갱신 여부

  // 사용량
  usage: {
    evaluations_used: number,    // 사용한 진단 횟수
    evaluations_limit: number,   // 진단 한도
    comparisons_used: number,    // 사용한 비교 횟수
    comparisons_limit: number    // 비교 한도
  },

  // 메타
  created_at: Timestamp,
  updated_at: Timestamp
}
```

### 인덱스

- `user_id` + `status`
- `expires_at` (for cleanup jobs)

---

## 7. 컬렉션 관계도

```
users (1) ----< (N) submissions
  |
  +----< (N) evaluations
  |
  +----< (N) subscriptions

competitions (1) ----< (N) submissions
```

---

## 8. 데이터 검증 규칙

### users

- `email`: 유효한 이메일 형식
- `username`: 영문소문자, 숫자, 언더스코어만 허용 (3-30자)
- `bio`: 최대 500자
- `achievements`: 최대 50개

### competitions

- `title`: 최대 100자
- `tags`: 최대 10개
- `judging_criteria`: 가중치 합계 = 100
- `submission_end` > `submission_start`

### submissions

- `artwork.title`: 최대 50자
- `artwork.image_urls`: 최대 5개
- `judging.scores`: 0-100 범위

### evaluations

- `rubric_scores.*`: 0-100 범위
- `confidence`: 0-1 범위
- `tier_prediction.probabilities`: 합계 = 1

---

## 9. 마이그레이션 가이드

### Phase 1 -> Phase 2

1. `users` 컬렉션에 `diagnosis_history` 필드 추가
2. `evaluations` 컬렉션 생성

### Phase 2 -> Phase 3

1. `users` 컬렉션에 `subscription` 필드 추가
2. `subscriptions` 컬렉션 생성
3. 기존 사용자 `subscription.plan = 'free'` 설정

---

*문서 버전: 2.1*
*최종 업데이트: 2025년 1월*
