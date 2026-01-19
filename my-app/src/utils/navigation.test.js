/**
 * 네비게이션 유틸리티 테스트
 *
 * SPEC-CRED-001 M5 요구사항에 따른 네비게이션 유틸리티 단위 테스트
 *
 * @module utils/navigation.test
 */

import {
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
} from './navigation';

describe('navigation utils', () => {
  describe('CREDENTIAL_ROUTES', () => {
    it('should have correct PROFILE route', () => {
      expect(CREDENTIAL_ROUTES.PROFILE).toBe('/profile');
    });

    it('should generate correct PUBLIC_PROFILE route', () => {
      expect(CREDENTIAL_ROUTES.PUBLIC_PROFILE('testuser')).toBe('/profile/testuser');
    });

    it('should have correct PORTFOLIO route', () => {
      expect(CREDENTIAL_ROUTES.PORTFOLIO).toBe('/portfolio');
    });
  });

  describe('COMPETITION_ROUTES', () => {
    it('should have correct LIST route', () => {
      expect(COMPETITION_ROUTES.LIST).toBe('/competitions');
    });

    it('should generate correct DETAIL route', () => {
      expect(COMPETITION_ROUTES.DETAIL('comp-123')).toBe('/competitions/comp-123');
    });

    it('should generate correct SUBMIT route', () => {
      expect(COMPETITION_ROUTES.SUBMIT('comp-123')).toBe('/competitions/comp-123/submit');
    });
  });

  describe('DIAGNOSIS_ROUTES', () => {
    it('should have correct MAIN route', () => {
      expect(DIAGNOSIS_ROUTES.MAIN).toBe('/diagnosis');
    });
  });

  describe('MAIN_ROUTES', () => {
    it('should have correct HOME route', () => {
      expect(MAIN_ROUTES.HOME).toBe('/');
    });

    it('should have correct TERMS route', () => {
      expect(MAIN_ROUTES.TERMS).toBe('/terms');
    });

    it('should have correct PRIVACY route', () => {
      expect(MAIN_ROUTES.PRIVACY).toBe('/privacy');
    });
  });

  describe('createNavItem', () => {
    it('should create nav item with label and href', () => {
      const item = createNavItem('공모전', '/competitions');
      expect(item).toEqual({ label: '공모전', href: '/competitions' });
    });
  });

  describe('LOGGED_IN_NAV_ITEMS', () => {
    it('should contain 공모전 link', () => {
      expect(LOGGED_IN_NAV_ITEMS).toContainEqual(
        expect.objectContaining({ label: '공모전', href: '/competitions' })
      );
    });

    it('should contain AI 진단 link', () => {
      expect(LOGGED_IN_NAV_ITEMS).toContainEqual(
        expect.objectContaining({ label: 'AI 진단', href: '/diagnosis' })
      );
    });

    it('should contain 마이페이지 link', () => {
      expect(LOGGED_IN_NAV_ITEMS).toContainEqual(
        expect.objectContaining({ label: '마이페이지', href: '/profile' })
      );
    });
  });

  describe('GUEST_NAV_ITEMS', () => {
    it('should contain 공모전 link', () => {
      expect(GUEST_NAV_ITEMS).toContainEqual(
        expect.objectContaining({ label: '공모전', href: '/competitions' })
      );
    });

    it('should contain AI 진단 link', () => {
      expect(GUEST_NAV_ITEMS).toContainEqual(
        expect.objectContaining({ label: 'AI 진단', href: '/diagnosis' })
      );
    });

    it('should not contain 마이페이지 link', () => {
      expect(GUEST_NAV_ITEMS).not.toContainEqual(
        expect.objectContaining({ label: '마이페이지' })
      );
    });

    it('should contain anchor links for landing page sections', () => {
      expect(GUEST_NAV_ITEMS).toContainEqual(
        expect.objectContaining({ label: 'Why MIRIP', href: '#problem' })
      );
      expect(GUEST_NAV_ITEMS).toContainEqual(
        expect.objectContaining({ label: 'Solution', href: '#solution' })
      );
    });
  });

  describe('extractUsernameFromPath', () => {
    it('should extract username from public profile path', () => {
      expect(extractUsernameFromPath('/profile/testuser')).toBe('testuser');
    });

    it('should return null for my profile path', () => {
      expect(extractUsernameFromPath('/profile')).toBeNull();
    });

    it('should return null for non-profile paths', () => {
      expect(extractUsernameFromPath('/competitions')).toBeNull();
      expect(extractUsernameFromPath('/')).toBeNull();
    });

    it('should handle usernames with underscores', () => {
      expect(extractUsernameFromPath('/profile/test_user_123')).toBe('test_user_123');
    });

    it('should return null for nested paths beyond username', () => {
      expect(extractUsernameFromPath('/profile/testuser/settings')).toBeNull();
    });
  });

  describe('isProfileRoute', () => {
    it('should return true for my profile route', () => {
      expect(isProfileRoute('/profile')).toBe(true);
    });

    it('should return true for public profile routes', () => {
      expect(isProfileRoute('/profile/testuser')).toBe(true);
    });

    it('should return false for non-profile routes', () => {
      expect(isProfileRoute('/')).toBe(false);
      expect(isProfileRoute('/competitions')).toBe(false);
      expect(isProfileRoute('/diagnosis')).toBe(false);
    });
  });

  describe('isCompetitionRoute', () => {
    it('should return true for competitions list route', () => {
      expect(isCompetitionRoute('/competitions')).toBe(true);
    });

    it('should return true for competition detail routes', () => {
      expect(isCompetitionRoute('/competitions/comp-123')).toBe(true);
      expect(isCompetitionRoute('/competitions/comp-123/submit')).toBe(true);
    });

    it('should return false for non-competition routes', () => {
      expect(isCompetitionRoute('/')).toBe(false);
      expect(isCompetitionRoute('/profile')).toBe(false);
      expect(isCompetitionRoute('/diagnosis')).toBe(false);
    });
  });
});
