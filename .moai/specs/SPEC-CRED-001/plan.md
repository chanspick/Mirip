# SPEC-CRED-001 구현 계획

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-CRED-001 |
| **문서 유형** | Implementation Plan |
| **생성일** | 2025-01-19 |
| **관련 SPEC** | SPEC-CRED-001/spec.md |

---

## 1. 마일스톤 개요

### 우선순위 기반 마일스톤

| 마일스톤 | 우선순위 | 핵심 산출물 | 의존성 |
|----------|----------|-------------|--------|
| M1: 데이터 모델 및 서비스 | **PRIMARY** | Firestore 스키마, 서비스 레이어 | 없음 |
| M2: 마이페이지 (잔디밭 + 히스토리) | **PRIMARY** | ActivityHeatmap, ActivityTimeline | M1 |
| M3: 공개 프로필 | **SECONDARY** | PublicProfile 페이지, ProfileCard | M1, M2 |
| M4: 포트폴리오 관리 | **SECONDARY** | Portfolio CRUD, 이미지 업로드 | M1 |
| M5: 기존 시스템 연동 | **FINAL** | 진단/공모전 → 활동 기록 자동화 | M1, M2 |

---

## 2. 마일스톤 상세

### M1: 데이터 모델 및 서비스 (PRIMARY)

#### 목표

- Firestore 컬렉션 스키마 정의 및 보안 규칙 설정
- 크레덴셜 관련 서비스 레이어 구현
- 커스텀 훅 구현

#### 태스크

```
M1.1 Firestore 스키마 설계
├── users 컬렉션 확장 (크레덴셜 필드 추가)
├── users/{userId}/activities 서브컬렉션 생성
├── users/{userId}/dailyActivityCounts 서브컬렉션 생성
├── users/{userId}/portfolios 서브컬렉션 생성
└── users/{userId}/awards 서브컬렉션 생성

M1.2 Firestore 보안 규칙
├── 본인 데이터 읽기/쓰기 권한
├── 공개 프로필 읽기 권한 (isPublic: true)
├── 공개 포트폴리오 읽기 권한 (isPublic: true)
└── 관리자 수상 검증 권한

M1.3 서비스 레이어 구현
├── credentialService.js (프로필 CRUD)
├── activityService.js (활동 기록)
├── portfolioService.js (포트폴리오 CRUD)
└── awardService.js (수상 이력)

M1.4 커스텀 훅 구현
├── useUserProfile.js
├── useActivities.js
├── usePortfolios.js
└── useAwards.js
```

#### 산출물

| 파일 | 설명 |
|------|------|
| `firestore.rules` | Firestore 보안 규칙 |
| `my-app/src/services/credentialService.js` | 프로필 서비스 |
| `my-app/src/services/activityService.js` | 활동 기록 서비스 |
| `my-app/src/services/portfolioService.js` | 포트폴리오 서비스 |
| `my-app/src/services/awardService.js` | 수상 이력 서비스 |
| `my-app/src/hooks/useUserProfile.js` | 프로필 훅 |
| `my-app/src/hooks/useActivities.js` | 활동 기록 훅 |
| `my-app/src/hooks/usePortfolios.js` | 포트폴리오 훅 |
| `my-app/src/hooks/useAwards.js` | 수상 이력 훅 |

---

### M2: 마이페이지 (잔디밭 + 히스토리) (PRIMARY)

#### 목표

- GitHub 스타일 활동 히트맵(잔디밭) 컴포넌트 구현
- 활동 히스토리 타임라인 구현
- 마이페이지 통합 페이지 구현

#### 태스크

```
M2.1 ActivityHeatmap 컴포넌트
├── 52주 x 7일 그리드 레이아웃
├── 활동 레벨별 색상 (5단계)
├── 셀 호버 툴팁 (날짜, 활동 수)
├── 월 레이블 표시
└── 반응형 디자인

M2.2 ActivityTimeline 컴포넌트
├── 시간순 활동 목록
├── 활동 타입별 아이콘/색상
├── 날짜 그룹핑 (오늘, 어제, 이번 주)
├── 더보기/무한 스크롤
└── 빈 상태 UI

M2.3 통계 요약 컴포넌트
├── 총 활동 수
├── 진단 횟수
├── 수상 횟수
├── 연속 활동일 (스트릭)
└── 가장 활동적인 요일

M2.4 MyPage 페이지 통합
├── ProfileCard 섹션
├── ActivityHeatmap 섹션
├── ActivityTimeline 섹션
├── 통계 요약 섹션
└── 라우트 설정 (/profile)
```

#### 산출물

| 파일 | 설명 |
|------|------|
| `my-app/src/components/credential/ActivityHeatmap/` | 잔디밭 컴포넌트 |
| `my-app/src/components/credential/ActivityTimeline/` | 타임라인 컴포넌트 |
| `my-app/src/components/credential/ActivityStats/` | 통계 요약 컴포넌트 |
| `my-app/src/pages/Credential/ProfileEdit/` | 마이페이지 |

---

### M3: 공개 프로필 (SECONDARY)

#### 목표

- ProfileCard 컴포넌트 구현
- TierBadge 컴포넌트 구현
- AchievementList 컴포넌트 구현
- 공개 프로필 페이지 구현

#### 태스크

```
M3.1 ProfileCard 컴포넌트
├── 프로필 이미지 (기본 아바타)
├── 이름, 소개 표시
├── 소셜 링크 아이콘
├── 편집 모드 (소유자)
└── 읽기 전용 모드 (방문자)

M3.2 TierBadge 컴포넌트
├── 티어별 색상/디자인
├── 크기 변형 (sm, md, lg)
├── 호버 툴팁
└── 애니메이션 효과

M3.3 AchievementList 컴포넌트
├── 수상 카드 목록
├── 공모전명, 순위, 수상일
├── 검증 배지
├── 더보기 (접기/펼치기)
└── 빈 상태 UI

M3.4 PublicProfile 페이지
├── username 파라미터 처리
├── ProfileCard 섹션
├── TierBadge 표시
├── PortfolioGrid 섹션
├── AchievementList 섹션
├── OG 메타 태그
└── 404 처리
```

#### 산출물

| 파일 | 설명 |
|------|------|
| `my-app/src/components/credential/ProfileCard/` | 프로필 카드 |
| `my-app/src/components/credential/TierBadge/` | 티어 배지 |
| `my-app/src/components/credential/AchievementList/` | 수상 이력 |
| `my-app/src/pages/Credential/PublicProfile/` | 공개 프로필 페이지 |

---

### M4: 포트폴리오 관리 (SECONDARY)

#### 목표

- PortfolioGrid 컴포넌트 구현
- 포트폴리오 CRUD 모달 구현
- 이미지 업로드 및 썸네일 생성
- 포트폴리오 관리 페이지 구현

#### 태스크

```
M4.1 PortfolioGrid 컴포넌트
├── 그리드/Masonry 레이아웃
├── 이미지 카드 (제목, 카테고리)
├── 클릭 시 상세 모달
├── 편집 모드 UI
└── 빈 상태 UI

M4.2 PortfolioModal 컴포넌트
├── 작품 상세 보기
├── 작품 추가 폼
├── 작품 편집 폼
├── 이미지 미리보기
└── 삭제 확인

M4.3 이미지 처리
├── 이미지 압축 (최대 2MB)
├── 썸네일 생성 (300x300)
├── WebP 변환 (선택적)
├── Storage 업로드
└── URL 반환

M4.4 Portfolio 페이지
├── PortfolioGrid 섹션
├── 추가/편집/삭제 버튼
├── 순서 변경 (드래그 앤 드롭)
├── 공개/비공개 토글
└── 라우트 설정 (/portfolio)
```

#### 산출물

| 파일 | 설명 |
|------|------|
| `my-app/src/components/credential/PortfolioGrid/` | 포트폴리오 그리드 |
| `my-app/src/components/credential/PortfolioModal/` | 포트폴리오 모달 |
| `my-app/src/utils/imageProcessing.js` | 이미지 처리 유틸 (확장) |
| `my-app/src/pages/Credential/Portfolio/` | 포트폴리오 관리 페이지 |

---

### M5: 기존 시스템 연동 (FINAL)

#### 목표

- AI 진단 완료 시 활동 기록 자동 생성
- 공모전 출품 시 활동 기록 자동 생성
- 공모전 수상 시 활동 기록 + 수상 이력 자동 생성
- 사용자 최고 티어 자동 업데이트

#### 태스크

```
M5.1 진단 서비스 연동
├── diagnosisService.js 수정
│   ├── evaluateImage 성공 후 활동 기록 생성
│   └── 사용자 최고 티어 업데이트
└── 테스트 케이스 추가

M5.2 공모전 서비스 연동
├── submissionService.js 수정
│   └── 출품 완료 후 활동 기록 생성
├── 수상 처리 로직 추가
│   ├── 활동 기록 생성
│   └── awards 컬렉션에 수상 이력 추가
└── 테스트 케이스 추가

M5.3 프로필 업데이트 연동
├── credentialService.js 수정
│   └── 프로필 수정 시 활동 기록 생성
└── 테스트 케이스 추가

M5.4 포트폴리오 연동
├── portfolioService.js 수정
│   └── 작품 추가 시 활동 기록 생성
└── 테스트 케이스 추가
```

#### 산출물

| 파일 | 설명 |
|------|------|
| `my-app/src/services/diagnosisService.js` | 진단 서비스 (수정) |
| `my-app/src/services/submissionService.js` | 출품 서비스 (수정) |
| `my-app/src/services/credentialService.js` | 프로필 서비스 (수정) |
| `my-app/src/services/portfolioService.js` | 포트폴리오 서비스 (수정) |

---

## 3. 기술 접근 방식

### 3.1 잔디밭 (ActivityHeatmap) 구현

```javascript
// 잔디밭 데이터 구조
const heatmapData = {
  // YYYY-MM-DD: 활동 수
  '2025-01-01': 3,
  '2025-01-02': 0,
  '2025-01-03': 5,
  // ...
};

// 레벨 계산
const getLevel = (count) => {
  if (count === 0) return 0;
  if (count <= 2) return 1;
  if (count <= 4) return 2;
  if (count <= 6) return 3;
  return 4;
};

// 색상 매핑
const colors = {
  0: '#ebedf0', // 빈칸
  1: '#9be9a8', // 연한 초록
  2: '#40c463', // 중간 초록
  3: '#30a14e', // 진한 초록
  4: '#216e39', // 가장 진한 초록
};
```

### 3.2 Firestore 쿼리 최적화

```javascript
// 잔디밭용 일별 집계 조회 (365일)
const getDailyActivityCounts = async (userId, year) => {
  const startDate = `${year}-01-01`;
  const endDate = `${year}-12-31`;

  const q = query(
    collection(db, `users/${userId}/dailyActivityCounts`),
    where('date', '>=', startDate),
    where('date', '<=', endDate),
    orderBy('date', 'asc')
  );

  const snapshot = await getDocs(q);
  return snapshot.docs.map(doc => doc.data());
};
```

### 3.3 이미지 처리 파이프라인

```javascript
// 이미지 업로드 파이프라인
const uploadPortfolioImage = async (userId, file) => {
  // 1. 압축
  const compressed = await compressImage(file, { maxSize: 2 * 1024 * 1024 });

  // 2. 썸네일 생성
  const thumbnail = await createThumbnail(compressed, { width: 300, height: 300 });

  // 3. Storage 업로드
  const imageRef = ref(storage, `portfolios/${userId}/${Date.now()}_original`);
  const thumbRef = ref(storage, `portfolios/${userId}/${Date.now()}_thumb`);

  await uploadBytes(imageRef, compressed);
  await uploadBytes(thumbRef, thumbnail);

  // 4. URL 반환
  return {
    imageUrl: await getDownloadURL(imageRef),
    thumbnailUrl: await getDownloadURL(thumbRef),
  };
};
```

### 3.4 보안 규칙

```javascript
// firestore.rules
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // 사용자 프로필
    match /users/{userId} {
      allow read: if resource.data.isPublic == true || request.auth.uid == userId;
      allow write: if request.auth.uid == userId;

      // 활동 기록
      match /activities/{activityId} {
        allow read: if request.auth.uid == userId;
        allow create: if request.auth.uid == userId;
      }

      // 일별 집계
      match /dailyActivityCounts/{date} {
        allow read: if request.auth.uid == userId;
        allow write: if request.auth.uid == userId;
      }

      // 포트폴리오
      match /portfolios/{portfolioId} {
        allow read: if resource.data.isPublic == true || request.auth.uid == userId;
        allow write: if request.auth.uid == userId;
      }

      // 수상 이력
      match /awards/{awardId} {
        allow read: if true; // 수상 이력은 공개
        allow create: if request.auth.uid == userId;
      }
    }
  }
}
```

---

## 4. 아키텍처 설계

### 4.1 컴포넌트 계층

```
pages/
├── Credential/
│   ├── ProfileEdit/        # 마이페이지 (/profile)
│   │   └── ProfileEdit.js
│   ├── PublicProfile/      # 공개 프로필 (/profile/:username)
│   │   └── PublicProfile.js
│   └── Portfolio/          # 포트폴리오 관리 (/portfolio)
│       └── Portfolio.js

components/
├── credential/
│   ├── ActivityHeatmap/    # 잔디밭
│   ├── ActivityTimeline/   # 타임라인
│   ├── ActivityStats/      # 통계 요약
│   ├── ProfileCard/        # 프로필 카드
│   ├── TierBadge/          # 티어 배지
│   ├── PortfolioGrid/      # 포트폴리오 그리드
│   ├── PortfolioModal/     # 포트폴리오 모달
│   ├── AchievementList/    # 수상 이력
│   └── CredentialBadge/    # 인증 배지
```

### 4.2 데이터 흐름

```
[사용자 행동]
     │
     ▼
[서비스 레이어] ───────────────────┐
     │                           │
     ▼                           ▼
[활동 기록 생성]           [데이터 저장]
     │                           │
     ▼                           │
[일별 집계 업데이트]              │
     │                           │
     ▼                           ▼
[Firestore] ◄────────────────────┘
     │
     ▼
[커스텀 훅] ──► [컴포넌트] ──► [UI 렌더링]
```

---

## 5. 위험 및 대응

### 5.1 식별된 위험

| 위험 | 영향 | 확률 | 대응 |
|------|------|------|------|
| 잔디밭 데이터 조회 성능 | 중 | 중 | dailyActivityCounts 별도 컬렉션으로 집계 |
| 이미지 Storage 용량 초과 | 중 | 낮 | 이미지 압축, 썸네일 생성, 무료 티어 모니터링 |
| Firestore 쿼리 비용 | 낮 | 중 | 캐싱, 페이지네이션, 인덱스 최적화 |
| username 충돌 | 낮 | 낮 | 고유성 검증, 예약어 차단 |

### 5.2 완화 전략

1. **잔디밭 성능**: dailyActivityCounts 서브컬렉션으로 사전 집계, React Query 캐싱
2. **Storage 용량**: 이미지 최대 2MB 압축, WebP 변환, 미사용 이미지 정리 스크립트
3. **쿼리 비용**: 실시간 리스너 대신 일회성 조회, 결과 캐싱, 불필요한 재조회 방지
4. **username 충돌**: 생성 시 중복 검사, admin/profile/api 등 예약어 차단

---

## 6. 품질 게이트

### 6.1 각 마일스톤 완료 조건

| 마일스톤 | 완료 조건 |
|----------|-----------|
| M1 | 모든 서비스 함수 단위 테스트 통과, Firestore 보안 규칙 테스트 통과 |
| M2 | 잔디밭 365일 데이터 정상 렌더링, 타임라인 페이지네이션 동작 |
| M3 | 공개 프로필 URL 접근 시 정상 표시, 비공개 데이터 노출 없음 |
| M4 | 포트폴리오 CRUD 전체 동작, 이미지 업로드/썸네일 생성 정상 |
| M5 | 진단/공모전 완료 시 활동 기록 자동 생성, 잔디밭 반영 확인 |

### 6.2 TRUST 5 체크리스트

- [ ] **Test-first**: 각 서비스 함수에 대한 단위 테스트 작성
- [ ] **Readable**: 컴포넌트/서비스 명명 규칙 준수, JSDoc 주석
- [ ] **Unified**: CSS Module 스타일 격리, 디자인 시스템 적용
- [ ] **Secured**: Firestore 보안 규칙, 이미지 업로드 검증
- [ ] **Trackable**: SPEC-CRED-001 태그로 커밋 추적

---

*문서 버전: 1.0*
*최종 업데이트: 2025-01-19*
*작성: manager-spec*
