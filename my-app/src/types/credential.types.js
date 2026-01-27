/**
 * Credential System 타입 정의
 *
 * SPEC-CRED-001 요구사항에 따른 Firestore 스키마 타입 정의
 * JSDoc을 사용한 타입 문서화
 *
 * @module types/credential.types
 */

// ============================================
// 사용자 프로필 타입
// ============================================

/**
 * 사용자 프로필
 * Firestore Collection: users
 *
 * @typedef {Object} UserProfile
 * @property {string} uid - Firebase Auth UID
 * @property {string} username - 고유 사용자명 (URL-safe, 소문자 + 숫자 + 언더스코어)
 * @property {string} displayName - 화면에 표시되는 이름
 * @property {string} [profileImageUrl] - 프로필 이미지 URL
 * @property {string} [bio] - 자기소개 (최대 500자)
 * @property {string} tier - 사용자 티어 (S/A/B/C/Unranked)
 * @property {number} totalActivities - 총 활동 수
 * @property {number} currentStreak - 현재 연속 활동일
 * @property {number} longestStreak - 최장 연속 활동일
 * @property {Object} createdAt - Firestore Timestamp (생성일)
 * @property {Object} updatedAt - Firestore Timestamp (수정일)
 * @property {boolean} isPublic - 프로필 공개 여부 (기본값: true)
 */

/**
 * 사용자 프로필 생성 데이터
 *
 * @typedef {Object} CreateUserProfileData
 * @property {string} uid - Firebase Auth UID
 * @property {string} email - 이메일 (username 자동 생성용)
 * @property {string} [displayName] - 화면에 표시되는 이름
 */

/**
 * 사용자 프로필 업데이트 데이터
 *
 * @typedef {Object} UpdateUserProfileData
 * @property {string} [username] - 사용자명 (변경 시 중복 체크 필요)
 * @property {string} [displayName] - 화면에 표시되는 이름
 * @property {string} [profileImageUrl] - 프로필 이미지 URL
 * @property {string} [bio] - 자기소개 (최대 500자)
 * @property {boolean} [isPublic] - 프로필 공개 여부
 */

// ============================================
// 활동 타입
// ============================================

/**
 * 활동 타입 열거형
 * @typedef {'diagnosis' | 'competition_submit' | 'competition_award' | 'profile_update' | 'portfolio_add'} ActivityType
 */

/**
 * 활동 기록
 * Firestore Collection: users/{userId}/activities
 *
 * @typedef {Object} Activity
 * @property {string} id - 활동 ID
 * @property {string} userId - 사용자 UID
 * @property {ActivityType} type - 활동 타입
 * @property {string} title - 활동 제목
 * @property {string} [description] - 활동 설명
 * @property {Object} metadata - 활동 타입별 추가 데이터
 * @property {Object} createdAt - Firestore Timestamp
 */

/**
 * 활동 생성 데이터
 *
 * @typedef {Object} CreateActivityData
 * @property {ActivityType} type - 활동 타입
 * @property {string} title - 활동 제목
 * @property {string} [description] - 활동 설명
 * @property {Object} [metadata] - 추가 메타데이터
 */

/**
 * 일별 활동 수
 * Firestore Collection: users/{userId}/dailyActivities
 *
 * @typedef {Object} DailyActivityCount
 * @property {string} date - YYYY-MM-DD 형식의 날짜
 * @property {number} count - 해당 일의 활동 수
 * @property {string[]} activityIds - 해당 일의 활동 ID 목록
 */

/**
 * 활동 히트맵 데이터
 *
 * @typedef {Object} ActivityHeatmapData
 * @property {number} year - 연도
 * @property {DailyActivityCount[]} days - 일별 활동 데이터 (365일)
 * @property {number} totalActivities - 해당 연도 총 활동 수
 */

/**
 * 활동 조회 옵션
 *
 * @typedef {Object} GetActivitiesOptions
 * @property {number} [pageSize=20] - 페이지 크기
 * @property {Object} [lastDoc] - 페이지네이션용 마지막 문서
 * @property {ActivityType} [type] - 활동 타입 필터
 */

// ============================================
// 포트폴리오 타입
// ============================================

/**
 * 포트폴리오 작품
 * Firestore Collection: users/{userId}/portfolios
 *
 * @typedef {Object} Portfolio
 * @property {string} id - 포트폴리오 ID
 * @property {string} userId - 사용자 UID
 * @property {string} title - 작품 제목
 * @property {string} [description] - 작품 설명
 * @property {string} imageUrl - 원본 이미지 URL
 * @property {string} [thumbnailUrl] - 썸네일 이미지 URL
 * @property {string[]} tags - 태그 목록
 * @property {boolean} isPublic - 공개 여부 (기본값: true)
 * @property {Object} createdAt - Firestore Timestamp
 * @property {Object} updatedAt - Firestore Timestamp
 */

/**
 * 포트폴리오 생성 데이터
 *
 * @typedef {Object} CreatePortfolioData
 * @property {string} title - 작품 제목
 * @property {string} [description] - 작품 설명
 * @property {string} imageUrl - 원본 이미지 URL
 * @property {string} [thumbnailUrl] - 썸네일 이미지 URL
 * @property {string[]} [tags] - 태그 목록
 * @property {boolean} [isPublic=true] - 공개 여부
 */

/**
 * 포트폴리오 업데이트 데이터
 *
 * @typedef {Object} UpdatePortfolioData
 * @property {string} [title] - 작품 제목
 * @property {string} [description] - 작품 설명
 * @property {string} [imageUrl] - 원본 이미지 URL
 * @property {string} [thumbnailUrl] - 썸네일 이미지 URL
 * @property {string[]} [tags] - 태그 목록
 * @property {boolean} [isPublic] - 공개 여부
 */

/**
 * 포트폴리오 조회 옵션
 *
 * @typedef {Object} GetPortfoliosOptions
 * @property {number} [pageSize=12] - 페이지 크기
 * @property {Object} [lastDoc] - 페이지네이션용 마지막 문서
 * @property {boolean} [publicOnly=false] - 공개 작품만 조회
 */

// ============================================
// 수상 타입
// ============================================

/**
 * 수상 등급
 * @typedef {'대상' | '금상' | '은상' | '동상' | '입선'} AwardRank
 */

/**
 * 수상 기록
 * Firestore Collection: users/{userId}/awards
 *
 * @typedef {Object} Award
 * @property {string} id - 수상 ID
 * @property {string} competitionId - 공모전 ID
 * @property {string} competitionTitle - 공모전 제목
 * @property {AwardRank} rank - 수상 등급
 * @property {Object} awardedAt - Firestore Timestamp (수상일)
 */

/**
 * 수상 생성 데이터
 *
 * @typedef {Object} CreateAwardData
 * @property {string} competitionId - 공모전 ID
 * @property {string} competitionTitle - 공모전 제목
 * @property {AwardRank} rank - 수상 등급
 * @property {Date} [awardedAt] - 수상일 (기본값: 현재 시간)
 */

// ============================================
// 티어 관련 타입
// ============================================

/**
 * 사용자 티어
 * @typedef {'S' | 'A' | 'B' | 'C' | 'Unranked'} UserTier
 */

/**
 * 티어 계산 기준
 *
 * @typedef {Object} TierCriteria
 * @property {number} minDiagnoses - 최소 진단 횟수
 * @property {number} minAverageScore - 최소 평균 점수
 * @property {number} [minAwards] - 최소 수상 횟수 (선택)
 */

/**
 * 티어별 기준 정의
 * @type {Object.<UserTier, TierCriteria>}
 */
export const TIER_CRITERIA = {
  S: { minDiagnoses: 20, minAverageScore: 90, minAwards: 3 },
  A: { minDiagnoses: 10, minAverageScore: 75, minAwards: 1 },
  B: { minDiagnoses: 5, minAverageScore: 60 },
  C: { minDiagnoses: 1, minAverageScore: 0 },
  Unranked: { minDiagnoses: 0, minAverageScore: 0 },
};

// ============================================
// 유효성 검사용 상수
// ============================================

/**
 * Username 유효성 검사 정규식
 * 소문자, 숫자, 언더스코어만 허용, 3-30자
 */
export const USERNAME_REGEX = /^[a-z0-9_]{3,30}$/;

/**
 * Bio 최대 길이
 */
export const BIO_MAX_LENGTH = 500;

/**
 * 활동 타입 목록
 * @type {ActivityType[]}
 */
export const ACTIVITY_TYPES = [
  'diagnosis',
  'competition_submit',
  'competition_award',
  'profile_update',
  'portfolio_add',
];

/**
 * 수상 등급 목록
 * @type {AwardRank[]}
 */
export const AWARD_RANKS = ['대상', '금상', '은상', '동상', '입선'];

/**
 * 기본 페이지 크기
 */
export const DEFAULT_PAGE_SIZE = 20;

/**
 * 포트폴리오 기본 페이지 크기
 */
export const PORTFOLIO_PAGE_SIZE = 12;
