# SPEC-CRED-001: 크레덴셜 시스템 (마이페이지 + 공개 프로필 + GitHub 잔디밭)

---

## 메타데이터

| 항목 | 내용 |
|------|------|
| **SPEC ID** | SPEC-CRED-001 |
| **제목** | 크레덴셜 시스템 (마이페이지 + 공개 프로필 + GitHub 잔디밭) |
| **생성일** | 2025-01-19 |
| **상태** | Planned |
| **우선순위** | HIGH (Phase 3 핵심 기능) |
| **담당** | expert-frontend, expert-backend |
| **관련 SPEC** | SPEC-COMP-001 (공모전), SPEC-AI-001 (AI 진단), SPEC-FIREBASE-001 |
| **Lifecycle** | spec-anchored |

---

## 1. 환경 (Environment)

### 1.1 기술 스택

| 영역 | 기술 | 버전 |
|------|------|------|
| Frontend | React | 18.x |
| Routing | React Router DOM | 6.x |
| Styling | CSS Modules | - |
| State | React Context / Zustand | - |
| Database | Firebase Firestore | 10.x |
| Storage | Firebase Storage | 10.x |
| Auth | Firebase Auth | 10.x |

### 1.2 기존 구현 현황

- **Firebase 구성**: `my-app/src/config/firebase.js` - Auth, Firestore, Storage 초기화 완료
- **진단 서비스**: `my-app/src/services/diagnosisService.js` - AI 평가 API 연동 완료
- **공모전 서비스**: `my-app/src/services/competitionService.js` - CRUD 및 쿼리 완료
- **공통 컴포넌트**: Button, Card, Modal, Loading, Header, Footer

### 1.3 라우트 구조

```
/profile              -> 마이페이지 (내 프로필 편집 + 잔디밭 + 히스토리)
/profile/:username    -> 공개 프로필 (프로필 카드 + 포트폴리오)
/portfolio            -> 포트폴리오 관리
```

---

## 2. 가정 (Assumptions)

### 2.1 기술적 가정

| 가정 | 신뢰도 | 근거 | 위험 |
|------|--------|------|------|
| Firebase Auth로 사용자 인증됨 | HIGH | SPEC-FIREBASE-001 구현 완료 | 낮음 |
| Firestore로 실시간 데이터 동기화 가능 | HIGH | Firebase 공식 지원 | 낮음 |
| 365일 잔디밭 데이터 효율적 저장 가능 | MEDIUM | 문서당 1MB 제한 고려 필요 | 중간 |
| Storage 이미지 용량 관리 필요 | MEDIUM | 무료 티어 제한 존재 | 중간 |

### 2.2 비즈니스 가정

| 가정 | 신뢰도 | 근거 |
|------|--------|------|
| 사용자는 GitHub 잔디밭 UI에 익숙함 | MEDIUM | 개발자 커뮤니티 표준 |
| 공개 프로필 공유로 바이럴 효과 기대 | MEDIUM | LinkedIn, GitHub 사례 |
| 활동 기록이 이탈 비용(Lock-in) 증가 | HIGH | product.md CDJ 분석 |

### 2.3 제외 범위

- **결제 시스템**: 별도 SPEC (SPEC-PAY-001)로 분리
- **경력 증명서 PDF 발급**: v2.0 이후 기능
- **소셜 로그인 연동**: SPEC-FIREBASE-001에서 처리

---

## 3. 요구사항 (Requirements) - EARS 형식

### 3.1 Ubiquitous (항상 적용)

| ID | 요구사항 |
|----|----------|
| REQ-U-001 | 시스템은 **항상** 인증된 사용자만 마이페이지(/profile)에 접근할 수 있도록 해야 한다 |
| REQ-U-002 | 시스템은 **항상** 공개 프로필(/profile/:username)은 비인증 사용자도 조회할 수 있어야 한다 |
| REQ-U-003 | 시스템은 **항상** 활동 기록 생성 시 자동으로 일별 집계 데이터를 업데이트해야 한다 |
| REQ-U-004 | 시스템은 **항상** 이미지 업로드 시 최대 10MB, JPG/PNG/WebP 형식만 허용해야 한다 |
| REQ-U-005 | 시스템은 **항상** 사용자명(username)을 URL-safe 형식으로 유지해야 한다 |

### 3.2 Event-Driven (이벤트 발생 시)

| ID | 이벤트 | 동작 |
|----|--------|------|
| REQ-E-001 | **WHEN** 사용자가 AI 진단을 완료 **THEN** 활동 기록(type: diagnosis)을 자동 생성한다 |
| REQ-E-002 | **WHEN** 사용자가 공모전에 출품 **THEN** 활동 기록(type: competition_submit)을 자동 생성한다 |
| REQ-E-003 | **WHEN** 사용자가 공모전 수상 **THEN** 활동 기록(type: competition_award)을 자동 생성하고 수상 이력에 추가한다 |
| REQ-E-004 | **WHEN** 사용자가 프로필 정보 수정 **THEN** 활동 기록(type: profile_update)을 생성한다 |
| REQ-E-005 | **WHEN** 사용자가 포트폴리오 작품 추가 **THEN** 활동 기록(type: portfolio_add)을 생성한다 |
| REQ-E-006 | **WHEN** 잔디밭 컴포넌트 마운트 **THEN** 최근 365일 활동 데이터를 조회하여 히트맵을 렌더링한다 |
| REQ-E-007 | **WHEN** 공개 프로필 URL 접근 **THEN** 해당 사용자의 프로필 카드 + 포트폴리오를 표시한다 |

### 3.3 State-Driven (상태 기반)

| ID | 조건 | 동작 |
|----|------|------|
| REQ-S-001 | **IF** 사용자가 로그인 상태 **THEN** 마이페이지에서 편집 기능을 활성화한다 |
| REQ-S-002 | **IF** 공개 프로필 소유자가 아닌 사용자 **THEN** 읽기 전용 모드로 표시한다 |
| REQ-S-003 | **IF** 포트폴리오 작품이 0개 **THEN** 빈 상태 UI와 작품 추가 안내를 표시한다 |
| REQ-S-004 | **IF** 활동 기록이 없는 날짜 **THEN** 잔디밭에서 빈 셀(레벨 0)로 표시한다 |
| REQ-S-005 | **IF** 수상 이력이 있는 사용자 **THEN** 프로필 카드에 수상 배지를 표시한다 |

### 3.4 Optional (선택적 기능)

| ID | 요구사항 | 우선순위 |
|----|----------|----------|
| REQ-O-001 | **가능하면** 활동 히스토리 타임라인에 무한 스크롤을 제공한다 | LOW |
| REQ-O-002 | **가능하면** 포트폴리오 작품에 태그 필터 기능을 제공한다 | LOW |
| REQ-O-003 | **가능하면** 프로필 공유 시 OG 메타 태그로 미리보기를 제공한다 | MEDIUM |
| REQ-O-004 | **가능하면** 잔디밭 색상 테마 선택 기능을 제공한다 | LOW |

### 3.5 Unwanted (금지 사항)

| ID | 금지 동작 |
|----|-----------|
| REQ-N-001 | 시스템은 **비공개로 설정된 포트폴리오 작품을** 공개 프로필에 노출**하지 않아야 한다** |
| REQ-N-002 | 시스템은 **다른 사용자의 프로필을** 편집**할 수 없어야 한다** |
| REQ-N-003 | 시스템은 **인증되지 않은 사용자의** 활동 기록을 생성**하지 않아야 한다** |
| REQ-N-004 | 시스템은 **삭제된 사용자의 공개 프로필을** 조회 가능**하지 않아야 한다** |

---

## 4. 명세 (Specifications)

### 4.1 데이터 모델

#### 4.1.1 users 컬렉션 확장

```typescript
// Firestore: users/{userId}
interface User {
  // 기존 필드
  uid: string;
  email: string;
  createdAt: Timestamp;

  // 크레덴셜 확장 필드
  username: string;           // URL용 고유 식별자 (예: artista_kim)
  displayName: string;        // 표시 이름
  bio: string;                // 자기소개 (최대 500자)
  profileImageUrl: string;    // 프로필 이미지 URL
  socialLinks: {
    instagram?: string;
    behance?: string;
    website?: string;
  };
  tier: 'S' | 'A' | 'B' | 'C' | null;  // 최고 진단 티어
  totalActivities: number;    // 총 활동 수
  awardCount: number;         // 총 수상 횟수
  isPublic: boolean;          // 프로필 공개 여부
  updatedAt: Timestamp;
}
```

#### 4.1.2 activities 컬렉션 (잔디밭 데이터)

```typescript
// Firestore: users/{userId}/activities/{activityId}
interface Activity {
  id: string;
  userId: string;
  type: 'diagnosis' | 'competition_submit' | 'competition_award' | 'profile_update' | 'portfolio_add';
  title: string;              // 활동 제목 (예: "AI 진단 완료", "미립콤프 #3 출품")
  description: string;        // 상세 설명
  metadata: {
    // type별 추가 데이터
    diagnosisId?: string;
    competitionId?: string;
    competitionTitle?: string;
    awardRank?: number;
    portfolioId?: string;
    tier?: string;
  };
  createdAt: Timestamp;
  date: string;               // YYYY-MM-DD (잔디밭 집계용)
}
```

#### 4.1.3 dailyActivityCounts 컬렉션 (잔디밭 집계)

```typescript
// Firestore: users/{userId}/dailyActivityCounts/{YYYY-MM-DD}
interface DailyActivityCount {
  date: string;               // YYYY-MM-DD
  count: number;              // 해당 날짜 활동 수
  types: {                    // 타입별 카운트
    diagnosis: number;
    competition_submit: number;
    competition_award: number;
    profile_update: number;
    portfolio_add: number;
  };
}
```

#### 4.1.4 portfolios 컬렉션

```typescript
// Firestore: users/{userId}/portfolios/{portfolioId}
interface Portfolio {
  id: string;
  userId: string;
  title: string;
  description: string;
  imageUrl: string;           // Storage 이미지 URL
  thumbnailUrl: string;       // 썸네일 URL
  category: 'visual_design' | 'industrial_design' | 'fine_art' | 'craft';
  tags: string[];
  isPublic: boolean;          // 공개 여부
  order: number;              // 정렬 순서
  createdAt: Timestamp;
  updatedAt: Timestamp;
}
```

#### 4.1.5 awards 컬렉션

```typescript
// Firestore: users/{userId}/awards/{awardId}
interface Award {
  id: string;
  userId: string;
  competitionId: string;
  competitionTitle: string;
  rank: number;               // 순위 (1, 2, 3...)
  awardName: string;          // 상명 (예: "대상", "금상")
  awardedAt: Timestamp;
  certificateUrl?: string;    // 증명서 이미지 URL (선택)
  isVerified: boolean;        // 검증 여부
}
```

### 4.2 컴포넌트 명세

#### 4.2.1 ActivityHeatmap (잔디밭)

```
경로: my-app/src/components/credential/ActivityHeatmap/
파일: ActivityHeatmap.js, ActivityHeatmap.module.css

Props:
- userId: string              // 사용자 ID
- year?: number               // 표시 연도 (기본: 현재 연도)
- colorScheme?: 'green' | 'blue' | 'purple'  // 색상 테마

Features:
- GitHub 스타일 52주 x 7일 히트맵
- 활동 레벨별 색상 (0: 빈칸, 1-2: 연한색, 3-5: 중간색, 6+: 진한색)
- 셀 호버 시 날짜 및 활동 수 툴팁
- 연도 선택 드롭다운 (선택적)
```

#### 4.2.2 ActivityTimeline (활동 히스토리)

```
경로: my-app/src/components/credential/ActivityTimeline/
파일: ActivityTimeline.js, ActivityTimeline.module.css

Props:
- userId: string
- limit?: number              // 표시 개수 (기본: 20)
- onLoadMore?: () => void     // 더보기 콜백

Features:
- 시간순 활동 목록 (최신순)
- 활동 타입별 아이콘 및 색상
- 날짜 그룹핑 (오늘, 어제, 이번 주 등)
- 무한 스크롤 또는 더보기 버튼
```

#### 4.2.3 ProfileCard (프로필 카드)

```
경로: my-app/src/components/credential/ProfileCard/
파일: ProfileCard.js, ProfileCard.module.css

Props:
- user: User
- editable?: boolean          // 편집 모드
- onEdit?: () => void

Features:
- 프로필 이미지 (기본 아바타)
- 이름, 소개, 소셜 링크
- 티어 배지 표시
- 활동 통계 (진단 N회, 수상 N회)
- 편집 버튼 (소유자만)
```

#### 4.2.4 PortfolioGrid (포트폴리오 그리드)

```
경로: my-app/src/components/credential/PortfolioGrid/
파일: PortfolioGrid.js, PortfolioGrid.module.css

Props:
- portfolios: Portfolio[]
- columns?: 2 | 3 | 4         // 열 개수 (기본: 3)
- editable?: boolean
- onAdd?: () => void
- onEdit?: (id: string) => void
- onDelete?: (id: string) => void

Features:
- Masonry 또는 균등 그리드 레이아웃
- 이미지 클릭 시 상세 모달
- 드래그 앤 드롭 순서 변경 (편집 모드)
- 빈 상태 UI
```

#### 4.2.5 AchievementList (수상 이력)

```
경로: my-app/src/components/credential/AchievementList/
파일: AchievementList.js, AchievementList.module.css

Props:
- awards: Award[]
- showAll?: boolean           // 전체 표시 여부

Features:
- 수상 카드 목록
- 공모전 이름, 순위, 수상일
- 검증 배지 (공식 확인됨)
- 더보기 (기본 5개 표시)
```

#### 4.2.6 TierBadge (티어 배지)

```
경로: my-app/src/components/credential/TierBadge/
파일: TierBadge.js, TierBadge.module.css

Props:
- tier: 'S' | 'A' | 'B' | 'C' | null
- size?: 'sm' | 'md' | 'lg'

Features:
- 티어별 색상 (S: 금색, A: 은색, B: 동색, C: 회색)
- 호버 시 툴팁 (티어 설명)
```

### 4.3 페이지 명세

#### 4.3.1 MyPage (마이페이지)

```
경로: my-app/src/pages/Credential/ProfileEdit/
라우트: /profile

섹션:
1. ProfileCard (편집 가능)
2. ActivityHeatmap (잔디밭)
3. ActivityTimeline (최근 활동)
4. 통계 요약 (총 진단, 총 수상, 포트폴리오 수)
5. 빠른 링크 (포트폴리오 관리, 프로필 공유)
```

#### 4.3.2 PublicProfile (공개 프로필)

```
경로: my-app/src/pages/Credential/PublicProfile/
라우트: /profile/:username

섹션:
1. ProfileCard (읽기 전용)
2. TierBadge
3. PortfolioGrid (공개 작품만)
4. AchievementList (수상 이력)
5. 프로필 공유 버튼

SEO:
- 동적 OG 메타 태그
- 시맨틱 HTML 구조
```

#### 4.3.3 PortfolioManage (포트폴리오 관리)

```
경로: my-app/src/pages/Credential/Portfolio/
라우트: /portfolio

기능:
1. 포트폴리오 목록 (그리드)
2. 작품 추가 모달
3. 작품 편집 모달
4. 공개/비공개 토글
5. 순서 변경 (드래그 앤 드롭)
```

### 4.4 서비스 명세

#### 4.4.1 credentialService.js

```javascript
// my-app/src/services/credentialService.js

// 프로필 조회
getUserProfile(userId: string): Promise<User>
getUserByUsername(username: string): Promise<User | null>

// 프로필 업데이트
updateUserProfile(userId: string, data: Partial<User>): Promise<void>
uploadProfileImage(userId: string, file: File): Promise<string>

// 사용자명 검증
checkUsernameAvailability(username: string): Promise<boolean>
```

#### 4.4.2 activityService.js

```javascript
// my-app/src/services/activityService.js

// 활동 기록 생성
createActivity(userId: string, activity: Omit<Activity, 'id' | 'createdAt'>): Promise<string>

// 활동 조회
getActivities(userId: string, options: { limit?: number, startAfter?: DocumentSnapshot }): Promise<Activity[]>
getDailyActivityCounts(userId: string, startDate: string, endDate: string): Promise<DailyActivityCount[]>

// 잔디밭 데이터 집계 (내부 함수)
incrementDailyCount(userId: string, date: string, type: ActivityType): Promise<void>
```

#### 4.4.3 portfolioService.js

```javascript
// my-app/src/services/portfolioService.js

// 포트폴리오 CRUD
createPortfolio(userId: string, data: Omit<Portfolio, 'id' | 'createdAt'>): Promise<string>
getPortfolios(userId: string, options?: { publicOnly?: boolean }): Promise<Portfolio[]>
updatePortfolio(portfolioId: string, data: Partial<Portfolio>): Promise<void>
deletePortfolio(portfolioId: string): Promise<void>

// 이미지 업로드
uploadPortfolioImage(userId: string, file: File): Promise<{ imageUrl: string, thumbnailUrl: string }>
```

#### 4.4.4 awardService.js

```javascript
// my-app/src/services/awardService.js

// 수상 이력
getAwards(userId: string): Promise<Award[]>
createAward(userId: string, award: Omit<Award, 'id'>): Promise<string>

// (관리자용) 수상 검증
verifyAward(awardId: string): Promise<void>
```

### 4.5 기존 시스템 연동

#### 4.5.1 진단 서비스 연동 (diagnosisService.js 수정)

```javascript
// evaluateImage 함수 내부에 활동 기록 추가
const result = await fetchEvaluate(imageFile, department);

// 활동 기록 생성
await activityService.createActivity(userId, {
  type: 'diagnosis',
  title: 'AI 진단 완료',
  description: `${departmentName} 진단 - ${result.tier}티어`,
  metadata: {
    diagnosisId: result.evaluationId,
    tier: result.tier,
  },
  date: new Date().toISOString().split('T')[0],
});

// 사용자 최고 티어 업데이트
await updateUserTierIfHigher(userId, result.tier);
```

#### 4.5.2 공모전 서비스 연동 (submissionService.js 수정)

```javascript
// 출품 시 활동 기록
await activityService.createActivity(userId, {
  type: 'competition_submit',
  title: '공모전 출품',
  description: `${competition.title} 출품 완료`,
  metadata: {
    competitionId: competition.id,
    competitionTitle: competition.title,
  },
  date: new Date().toISOString().split('T')[0],
});

// 수상 시 활동 기록 및 수상 이력 추가
await activityService.createActivity(userId, {
  type: 'competition_award',
  title: '공모전 수상',
  description: `${competition.title} ${awardName}`,
  metadata: {
    competitionId: competition.id,
    competitionTitle: competition.title,
    awardRank: rank,
  },
  date: new Date().toISOString().split('T')[0],
});

await awardService.createAward(userId, { ... });
```

---

## 5. 제약사항

### 5.1 기술적 제약

| 제약 | 설명 | 대응 방안 |
|------|------|-----------|
| Firestore 문서 크기 | 최대 1MB | 활동 기록을 서브컬렉션으로 분리 |
| Firestore 쿼리 제한 | 복합 쿼리 제한 | 인덱스 사전 설정 |
| Storage 무료 티어 | 5GB 제한 | 이미지 압축, 썸네일 생성 |
| 잔디밭 365일 데이터 | 최대 365개 문서 | dailyActivityCounts 별도 컬렉션 |

### 5.2 성능 요구사항

| 항목 | 목표 |
|------|------|
| 마이페이지 초기 로드 | < 2초 |
| 잔디밭 렌더링 | < 500ms |
| 프로필 이미지 업로드 | < 3초 (5MB 기준) |
| 공개 프로필 로드 | < 1.5초 |

### 5.3 보안 요구사항

- Firestore 보안 규칙으로 사용자별 데이터 접근 제어
- 공개/비공개 설정에 따른 데이터 노출 제어
- 이미지 업로드 시 파일 타입 및 크기 검증

---

## 6. 추적성

### 6.1 관련 문서

- product.md: Pain Point #3 (경력 비표준화 → 크레덴셜)
- structure.md: 라우트 및 컴포넌트 구조
- tech.md: Firebase 기술 스택

### 6.2 연동 SPEC

| SPEC | 연동 내용 |
|------|-----------|
| SPEC-AI-001 | 진단 완료 → 활동 기록 생성 |
| SPEC-COMP-001 | 출품/수상 → 활동 기록 생성 |
| SPEC-FIREBASE-001 | Auth, Firestore 인프라 |

---

*문서 버전: 1.0*
*최종 업데이트: 2025-01-19*
*작성: manager-spec*
