/**
 * useAuth 훅 테스트
 *
 * SPEC-CRED-001 M5 요구사항에 따른 인증 훅 단위 테스트
 *
 * @module hooks/useAuth.test
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import useAuth from './useAuth';
import { initializeUserProfile, getUserProfile } from '../services/credentialService';

// Mock Firebase Auth
jest.mock('firebase/auth', () => ({
  onAuthStateChanged: jest.fn(),
  signOut: jest.fn(),
  getAuth: jest.fn(() => ({})),
}));

// Mock Firebase config
jest.mock('../config/firebase', () => ({
  auth: {},
}));

// Mock Credential Service
jest.mock('../services/credentialService', () => ({
  initializeUserProfile: jest.fn(),
  getUserProfile: jest.fn(),
}));

describe('useAuth', () => {
  const mockUser = {
    uid: 'test-uid-123',
    email: 'test@example.com',
    photoURL: 'https://example.com/photo.jpg',
  };

  const mockProfile = {
    uid: 'test-uid-123',
    username: 'testuser',
    displayName: 'Test User',
    profileImageUrl: null,
    tier: 'Unranked',
    totalActivities: 0,
  };

  let authStateCallback = null;

  beforeEach(() => {
    jest.clearAllMocks();
    authStateCallback = null;

    // onAuthStateChanged를 mock하여 콜백을 캡처
    onAuthStateChanged.mockImplementation((auth, callback) => {
      authStateCallback = callback;
      return jest.fn(); // unsubscribe function
    });
  });

  describe('initial state', () => {
    it('should start with loading true', () => {
      const { result } = renderHook(() => useAuth());

      expect(result.current.loading).toBe(true);
      expect(result.current.user).toBeNull();
      expect(result.current.profile).toBeNull();
    });

    it('should set isAuthenticated to false initially', () => {
      const { result } = renderHook(() => useAuth());

      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('when user logs in', () => {
    it('should set user and initialize profile', async () => {
      initializeUserProfile.mockResolvedValue(mockProfile);

      const { result } = renderHook(() => useAuth());

      // 로그인 이벤트 시뮬레이션
      await act(async () => {
        authStateCallback(mockUser);
      });

      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });

      expect(result.current.user).toEqual(mockUser);
      expect(result.current.isAuthenticated).toBe(true);
      expect(initializeUserProfile).toHaveBeenCalledWith(mockUser.uid, mockUser.email);
    });

    it('should set profile after initialization', async () => {
      initializeUserProfile.mockResolvedValue(mockProfile);

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        authStateCallback(mockUser);
      });

      await waitFor(() => {
        expect(result.current.profile).not.toBeNull();
      });

      expect(result.current.profile).toEqual(mockProfile);
      expect(result.current.isProfileComplete).toBe(true);
    });

    it('should handle profile initialization failure gracefully', async () => {
      initializeUserProfile.mockRejectedValue(new Error('Init failed'));
      getUserProfile.mockResolvedValue(mockProfile);

      // 콘솔 에러 억제
      jest.spyOn(console, 'error').mockImplementation(() => {});

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        authStateCallback(mockUser);
      });

      await waitFor(() => {
        expect(result.current.loading).toBe(false);
      });

      // 초기화 실패해도 기존 프로필 조회 시도
      expect(getUserProfile).toHaveBeenCalledWith(mockUser.uid);
      expect(result.current.error).not.toBeNull();

      console.error.mockRestore();
    });
  });

  describe('when user logs out', () => {
    it('should clear user and profile', async () => {
      initializeUserProfile.mockResolvedValue(mockProfile);

      const { result } = renderHook(() => useAuth());

      // 먼저 로그인
      await act(async () => {
        authStateCallback(mockUser);
      });

      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(true);
      });

      // 로그아웃 이벤트 시뮬레이션
      await act(async () => {
        authStateCallback(null);
      });

      expect(result.current.user).toBeNull();
      expect(result.current.profile).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('signOut function', () => {
    it('should call firebase signOut and clear state', async () => {
      signOut.mockResolvedValue();
      initializeUserProfile.mockResolvedValue(mockProfile);

      const { result } = renderHook(() => useAuth());

      // 먼저 로그인
      await act(async () => {
        authStateCallback(mockUser);
      });

      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(true);
      });

      // signOut 호출
      await act(async () => {
        await result.current.signOut();
      });

      expect(signOut).toHaveBeenCalled();
      expect(result.current.user).toBeNull();
      expect(result.current.profile).toBeNull();
    });

    it('should handle signOut error', async () => {
      const error = new Error('Sign out failed');
      signOut.mockRejectedValue(error);
      initializeUserProfile.mockResolvedValue(mockProfile);

      // 콘솔 에러 억제
      jest.spyOn(console, 'error').mockImplementation(() => {});

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        authStateCallback(mockUser);
      });

      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(true);
      });

      // signOut 호출 - 에러 발생
      await expect(
        act(async () => {
          await result.current.signOut();
        })
      ).rejects.toThrow('Sign out failed');

      console.error.mockRestore();
    });
  });

  describe('refreshProfile function', () => {
    it('should fetch and update profile', async () => {
      initializeUserProfile.mockResolvedValue(mockProfile);
      const updatedProfile = { ...mockProfile, displayName: 'Updated Name' };
      getUserProfile.mockResolvedValue(updatedProfile);

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        authStateCallback(mockUser);
      });

      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(true);
      });

      // 프로필 새로고침
      let refreshedProfile;
      await act(async () => {
        refreshedProfile = await result.current.refreshProfile();
      });

      expect(getUserProfile).toHaveBeenCalledWith(mockUser.uid);
      expect(refreshedProfile).toEqual(updatedProfile);
      expect(result.current.profile).toEqual(updatedProfile);
    });

    it('should return null when not authenticated', async () => {
      const { result } = renderHook(() => useAuth());

      await act(async () => {
        authStateCallback(null);
      });

      let refreshedProfile;
      await act(async () => {
        refreshedProfile = await result.current.refreshProfile();
      });

      expect(refreshedProfile).toBeNull();
    });
  });

  describe('isProfileComplete', () => {
    it('should return true when profile has username', async () => {
      initializeUserProfile.mockResolvedValue(mockProfile);

      const { result } = renderHook(() => useAuth());

      await act(async () => {
        authStateCallback(mockUser);
      });

      await waitFor(() => {
        expect(result.current.profile).not.toBeNull();
      });

      expect(result.current.isProfileComplete).toBe(true);
    });

    it('should return false when profile is null', async () => {
      const { result } = renderHook(() => useAuth());

      await act(async () => {
        authStateCallback(null);
      });

      expect(result.current.isProfileComplete).toBe(false);
    });
  });
});
