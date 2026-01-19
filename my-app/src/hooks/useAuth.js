/**
 * 인증 상태 관리 훅
 *
 * Firebase Auth 상태를 추적하고, 첫 로그인 시 프로필을 초기화합니다.
 * SPEC-CRED-001 M5 요구사항에 따라 구현되었습니다.
 *
 * @module hooks/useAuth
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { onAuthStateChanged, signOut as firebaseSignOut } from 'firebase/auth';
import { auth } from '../config/firebase';
import { initializeUserProfile, getUserProfile } from '../services/credentialService';

/**
 * 인증 상태 및 프로필 초기화 훅
 *
 * @returns {Object} 인증 상태 및 관련 함수
 * @property {Object|null} user - Firebase Auth 사용자 객체
 * @property {Object|null} profile - 사용자 프로필 데이터
 * @property {boolean} loading - 초기 로딩 상태
 * @property {boolean} profileLoading - 프로필 로딩 상태
 * @property {Error|null} error - 에러 객체
 * @property {Function} signOut - 로그아웃 함수
 * @property {Function} refreshProfile - 프로필 새로고침 함수
 */
const useAuth = () => {
  // Firebase Auth 사용자 상태
  const [user, setUser] = useState(null);
  // 사용자 프로필 상태
  const [profile, setProfile] = useState(null);
  // 초기 로딩 상태 (Auth 상태 확인 중)
  const [loading, setLoading] = useState(true);
  // 프로필 로딩 상태
  const [profileLoading, setProfileLoading] = useState(false);
  // 에러 상태
  const [error, setError] = useState(null);
  // 프로필 초기화 완료 여부
  const [profileInitialized, setProfileInitialized] = useState(false);

  /**
   * 프로필 초기화 또는 조회
   * 첫 로그인 시 프로필을 생성하고, 기존 사용자는 프로필을 조회합니다.
   */
  const initializeOrFetchProfile = useCallback(async (firebaseUser) => {
    if (!firebaseUser) {
      setProfile(null);
      setProfileInitialized(false);
      return;
    }

    setProfileLoading(true);
    setError(null);

    try {
      // 프로필 초기화 또는 기존 프로필 반환
      // initializeUserProfile은 기존 프로필이 있으면 그대로 반환합니다
      const userProfile = await initializeUserProfile(
        firebaseUser.uid,
        firebaseUser.email
      );
      setProfile(userProfile);
      setProfileInitialized(true);
    } catch (err) {
      console.error('[useAuth] 프로필 초기화 실패:', err);
      setError(err);
      // 초기화 실패 시 기존 프로필 조회 시도
      try {
        const existingProfile = await getUserProfile(firebaseUser.uid);
        if (existingProfile) {
          setProfile(existingProfile);
          setProfileInitialized(true);
        }
      } catch (fetchError) {
        console.error('[useAuth] 프로필 조회도 실패:', fetchError);
      }
    } finally {
      setProfileLoading(false);
    }
  }, []);

  /**
   * 프로필 새로고침
   */
  const refreshProfile = useCallback(async () => {
    if (!user) return null;

    setProfileLoading(true);
    try {
      const userProfile = await getUserProfile(user.uid);
      setProfile(userProfile);
      return userProfile;
    } catch (err) {
      console.error('[useAuth] 프로필 새로고침 실패:', err);
      setError(err);
      return null;
    } finally {
      setProfileLoading(false);
    }
  }, [user]);

  /**
   * 로그아웃
   */
  const signOut = useCallback(async () => {
    try {
      await firebaseSignOut(auth);
      setUser(null);
      setProfile(null);
      setProfileInitialized(false);
      setError(null);
    } catch (err) {
      console.error('[useAuth] 로그아웃 실패:', err);
      setError(err);
      throw err;
    }
  }, []);

  // Auth 상태 변화 감지
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      setUser(firebaseUser);
      setLoading(false);

      // 사용자가 로그인하면 프로필 초기화
      if (firebaseUser && !profileInitialized) {
        await initializeOrFetchProfile(firebaseUser);
      } else if (!firebaseUser) {
        setProfile(null);
        setProfileInitialized(false);
      }
    });

    return () => unsubscribe();
  }, [initializeOrFetchProfile, profileInitialized]);

  // 반환값 메모이제이션
  const value = useMemo(() => ({
    user,
    profile,
    loading,
    profileLoading,
    error,
    signOut,
    refreshProfile,
    isAuthenticated: !!user,
    isProfileComplete: !!(profile && profile.username),
  }), [user, profile, loading, profileLoading, error, signOut, refreshProfile]);

  return value;
};

export default useAuth;
