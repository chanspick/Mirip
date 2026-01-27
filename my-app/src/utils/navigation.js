/**
 * 네비게이션 유틸리티
 *
 * 앱 전체에서 사용되는 라우트 상수와 네비게이션 헬퍼 함수를 제공합니다.
 * SPEC-CRED-001 M5 요구사항에 따라 구현되었습니다.
 *
 * @module utils/navigation
 */

/**
 * 크레덴셜 관련 라우트 상수
 * @type {Object}
 */
export const CREDENTIAL_ROUTES = {
  /** 마이페이지 (로그인 사용자 전용) */
  PROFILE: '/profile',

  /** 공개 프로필 (username 기반) */
  PUBLIC_PROFILE: (username) => `/profile/${username}`,

  /** 포트폴리오 관리 페이지 */
  PORTFOLIO: '/portfolio',
};

/**
 * 공모전 관련 라우트 상수
 * @type {Object}
 */
export const COMPETITION_ROUTES = {
  /** 공모전 목록 */
  LIST: '/competitions',

  /** 공모전 상세 */
  DETAIL: (id) => `/competitions/${id}`,

  /** 공모전 출품 페이지 */
  SUBMIT: (id) => `/competitions/${id}/submit`,
};

/**
 * AI 진단 관련 라우트 상수
 * @type {Object}
 */
export const DIAGNOSIS_ROUTES = {
  /** AI 진단 페이지 */
  MAIN: '/diagnosis',
};

/**
 * 메인 라우트 상수
 * @type {Object}
 */
export const MAIN_ROUTES = {
  /** 랜딩 페이지 */
  HOME: '/',

  /** 이용약관 */
  TERMS: '/terms',

  /** 개인정보처리방침 */
  PRIVACY: '/privacy',
};

/**
 * 네비게이션 아이템 생성 헬퍼
 *
 * @param {string} label - 표시 텍스트
 * @param {string} href - 링크 경로
 * @returns {{label: string, href: string}}
 */
export const createNavItem = (label, href) => ({ label, href });

/**
 * 로그인 사용자용 기본 네비게이션 아이템
 * @type {Array<{label: string, href: string}>}
 */
export const LOGGED_IN_NAV_ITEMS = [
  { label: '공모전', href: COMPETITION_ROUTES.LIST },
  { label: 'AI 진단', href: DIAGNOSIS_ROUTES.MAIN },
  { label: '마이페이지', href: CREDENTIAL_ROUTES.PROFILE },
];

/**
 * 비로그인 사용자용 기본 네비게이션 아이템
 * @type {Array<{label: string, href: string}>}
 */
export const GUEST_NAV_ITEMS = [
  { label: '공모전', href: COMPETITION_ROUTES.LIST },
  { label: 'AI 진단', href: DIAGNOSIS_ROUTES.MAIN },
  { label: 'Why MIRIP', href: '#problem' },
  { label: 'Solution', href: '#solution' },
];

/**
 * URL에서 username 추출
 *
 * @param {string} pathname - URL 경로
 * @returns {string|null} 추출된 username 또는 null
 */
export const extractUsernameFromPath = (pathname) => {
  const match = pathname.match(/^\/profile\/([^/]+)$/);
  return match ? match[1] : null;
};

/**
 * 현재 경로가 프로필 관련 경로인지 확인
 *
 * @param {string} pathname - URL 경로
 * @returns {boolean}
 */
export const isProfileRoute = (pathname) => {
  return pathname === CREDENTIAL_ROUTES.PROFILE || pathname.startsWith('/profile/');
};

/**
 * 현재 경로가 공모전 관련 경로인지 확인
 *
 * @param {string} pathname - URL 경로
 * @returns {boolean}
 */
export const isCompetitionRoute = (pathname) => {
  return pathname === COMPETITION_ROUTES.LIST || pathname.startsWith('/competitions/');
};

const navigation = {
  CREDENTIAL_ROUTES,
  COMPETITION_ROUTES,
  DIAGNOSIS_ROUTES,
  MAIN_ROUTES,
  createNavItem,
  LOGGED_IN_NAV_ITEMS,
  GUEST_NAV_ITEMS,
  extractUsernameFromPath,
  isProfileRoute,
  isCompetitionRoute,
};

export default navigation;
