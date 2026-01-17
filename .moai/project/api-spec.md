# MIRIP API 명세서

> 미립(MIRIP) AI 진단 API 명세

---

## 1. API 개요

### Base URL

```
Production: https://api.mirip.ai/api/v1
Development: http://localhost:8000/api/v1
```

### 인증

- Firebase Auth 기반 JWT 토큰 인증
- Header: `Authorization: Bearer {token}`

### 응답 형식

모든 API 응답은 JSON 형식입니다.

**성공 응답:**
```json
{
  "success": true,
  "data": { ... }
}
```

**오류 응답:**
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "오류 설명"
  }
}
```

---

## 2. 단일 이미지 평가

### POST /api/v1/evaluate

단일 이미지에 대한 AI 평가를 수행합니다.

#### Request

**Headers:**
```
Authorization: Bearer {token}
Content-Type: application/json
```

**Body:**
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

**필드 설명:**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `image` | string | O | Base64 인코딩된 이미지 데이터 |
| `department` | string | O | 학과 코드 (`visual_design`, `industrial_design`, `fine_art`, `craft`) |
| `theme` | string | X | 작품 주제 (주제 해석 점수 산출용) |
| `options.include_feedback` | boolean | X | 상세 피드백 포함 여부 (기본값: false) |
| `options.language` | string | X | 피드백 언어 (`ko`, `en`, 기본값: `ko`) |

#### Response

**성공 (200):**
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

**필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `image_id` | string | 평가 결과 고유 ID |
| `tier_prediction.range` | string | 예측 티어 범위 |
| `tier_prediction.description` | string | 티어 범위 설명 |
| `tier_prediction.probabilities` | object | 티어별 확률 (합계 1.0) |
| `rubric_scores.composition` | number | 구성력 점수 (0-100) |
| `rubric_scores.tone_texture` | number | 명암/질감 점수 (0-100) |
| `rubric_scores.form_completion` | number | 조형완성도 점수 (0-100) |
| `rubric_scores.theme_interpretation` | number | 주제해석력 점수 (0-100) |
| `weighted_total` | number | 가중 평균 종합 점수 |
| `confidence` | number | 예측 신뢰도 (0-1) |
| `feedback.strengths` | array | 강점 목록 |
| `feedback.improvements` | array | 개선점 목록 |
| `feedback.recommendation` | string | 종합 추천 |

#### 오류 코드

| 코드 | 상태 | 설명 |
|------|------|------|
| `INVALID_IMAGE` | 400 | 잘못된 이미지 형식 |
| `IMAGE_TOO_LARGE` | 400 | 이미지 크기 초과 (최대 10MB) |
| `INVALID_DEPARTMENT` | 400 | 잘못된 학과 코드 |
| `QUOTA_EXCEEDED` | 429 | 일일 평가 횟수 초과 |
| `UNAUTHORIZED` | 401 | 인증 실패 |
| `SERVER_ERROR` | 500 | 서버 내부 오류 |

---

## 3. 복수 이미지 비교

### POST /api/v1/compare

여러 이미지의 상대적 순위를 비교합니다.

#### Request

**Headers:**
```
Authorization: Bearer {token}
Content-Type: application/json
```

**Body:**
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

**필드 설명:**

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `images` | array | O | 비교할 이미지 목록 (최소 2개, 최대 10개) |
| `images[].id` | string | O | 이미지 식별자 |
| `images[].image` | string | O | Base64 인코딩된 이미지 |
| `department` | string | O | 학과 코드 |

#### Response

**성공 (200):**
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

**필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `comparison_type` | string | 비교 방식 (`pairwise_ranking`) |
| `results` | array | 순위별 결과 목록 (1위부터) |
| `results[].image_id` | string | 이미지 식별자 |
| `results[].rank` | number | 순위 (1부터 시작) |
| `results[].score` | number | 종합 점수 |
| `ranking_confidence` | number | 순위 신뢰도 (0-1) |

#### 오류 코드

| 코드 | 상태 | 설명 |
|------|------|------|
| `INSUFFICIENT_IMAGES` | 400 | 이미지 개수 부족 (최소 2개) |
| `TOO_MANY_IMAGES` | 400 | 이미지 개수 초과 (최대 10개) |
| `QUOTA_EXCEEDED` | 429 | 일일 비교 횟수 초과 |
| `UNAUTHORIZED` | 401 | 인증 실패 |

---

## 4. 평가 이력 조회

### GET /api/v1/history

사용자의 평가 이력을 조회합니다.

#### Request

**Headers:**
```
Authorization: Bearer {token}
```

**Query Parameters:**

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| `user_id` | string | O | 사용자 ID |
| `limit` | number | X | 조회 개수 (기본값: 10, 최대: 50) |
| `offset` | number | X | 시작 위치 (기본값: 0) |
| `department` | string | X | 학과 필터 |

**예시:**
```
GET /api/v1/history?user_id=user123&limit=10&offset=0
```

#### Response

**성공 (200):**
```json
{
  "success": true,
  "data": {
    "total_count": 25,
    "items": [
      {
        "id": "eval_20250117_001",
        "created_at": "2025-01-17T10:30:00Z",
        "department": "visual_design",
        "tier_range": "A~S",
        "weighted_total": 83.4,
        "thumbnail_url": "https://storage.mirip.ai/thumbnails/eval_001.jpg"
      },
      {
        "id": "eval_20250116_002",
        "created_at": "2025-01-16T14:20:00Z",
        "department": "fine_art",
        "tier_range": "B~A",
        "weighted_total": 78.9,
        "thumbnail_url": "https://storage.mirip.ai/thumbnails/eval_002.jpg"
      }
    ],
    "pagination": {
      "limit": 10,
      "offset": 0,
      "has_more": true
    }
  }
}
```

**필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `total_count` | number | 전체 이력 개수 |
| `items` | array | 이력 목록 |
| `items[].id` | string | 평가 ID |
| `items[].created_at` | string | 평가 일시 (ISO 8601) |
| `items[].department` | string | 학과 코드 |
| `items[].tier_range` | string | 티어 범위 |
| `items[].weighted_total` | number | 종합 점수 |
| `items[].thumbnail_url` | string | 썸네일 이미지 URL |
| `pagination.has_more` | boolean | 추가 데이터 존재 여부 |

---

## 5. 학과 가중치 정보

### GET /api/v1/departments

학과별 평가 가중치 정보를 조회합니다.

#### Response

**성공 (200):**
```json
{
  "success": true,
  "data": {
    "departments": [
      {
        "code": "visual_design",
        "name": "시각디자인",
        "weights": {
          "composition": 0.25,
          "tone_texture": 0.20,
          "form_completion": 0.20,
          "theme_interpretation": 0.35
        }
      },
      {
        "code": "industrial_design",
        "name": "산업디자인",
        "weights": {
          "composition": 0.25,
          "tone_texture": 0.25,
          "form_completion": 0.35,
          "theme_interpretation": 0.15
        }
      },
      {
        "code": "fine_art",
        "name": "회화",
        "weights": {
          "composition": 0.20,
          "tone_texture": 0.35,
          "form_completion": 0.25,
          "theme_interpretation": 0.20
        }
      }
    ]
  }
}
```

---

## 6. 에러 처리

### 공통 오류 코드

| HTTP 상태 | 코드 | 설명 |
|----------|------|------|
| 400 | `BAD_REQUEST` | 잘못된 요청 |
| 401 | `UNAUTHORIZED` | 인증 필요 |
| 403 | `FORBIDDEN` | 권한 없음 |
| 404 | `NOT_FOUND` | 리소스 없음 |
| 429 | `RATE_LIMITED` | 요청 횟수 초과 |
| 500 | `INTERNAL_ERROR` | 서버 오류 |

### 오류 응답 예시

```json
{
  "success": false,
  "error": {
    "code": "QUOTA_EXCEEDED",
    "message": "일일 평가 횟수를 초과했습니다. 내일 다시 시도하거나 구독을 업그레이드하세요.",
    "details": {
      "current_usage": 5,
      "daily_limit": 5,
      "reset_at": "2025-01-18T00:00:00Z"
    }
  }
}
```

---

## 7. Rate Limiting

### 요청 제한

| 요금제 | 평가 (일) | 비교 (일) | API 호출 (분) |
|--------|----------|----------|--------------|
| Free | 1 | 0 | 10 |
| Basic | 5 | 0 | 60 |
| Pro | 무제한 | 무제한 | 300 |

### Rate Limit 헤더

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705500000
```

---

## 8. 웹훅 (Webhook)

### 평가 완료 웹훅

비동기 평가 완료 시 등록된 URL로 결과를 전송합니다.

**Payload:**
```json
{
  "event": "evaluation.completed",
  "data": {
    "image_id": "eval_20250117_001",
    "user_id": "user123",
    "status": "success",
    "result_url": "https://api.mirip.ai/api/v1/results/eval_20250117_001"
  },
  "timestamp": "2025-01-17T10:30:00Z"
}
```

---

*문서 버전: 2.1*
*최종 업데이트: 2025년 1월*
