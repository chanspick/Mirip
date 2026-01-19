/**
 * useUserProfile 훅
 *
 * 사용자 프로필 상태 관리 및 CRUD 작업을 제공합니다.
 * SPEC-CRED-001 요구사항에 따라 구현되었습니다.
 *
 * @module hooks/useUserProfile
 */

import { useState, useEffect, useCallback } from 'react';
import {
  getUserProfile,
  getUserProfileByUsername,
  updateUserProfile,
  checkUsernameAvailable,
  initializeUserProfile,
} from '../services/credentialService';

/**
 * 사용자 프로필 관리 훅
 * @param {string} [uid] - 조회할 사용자 UID (없으면 현재 사용자)
 * @returns {Object} 프로필 상태 및 메서드
 */
const useUserProfile = (uid) => {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [updating, setUpdating] = useState(false);

  /**
   * 프로필 로드
   */
  const loadProfile = useCallback(async () => {
    if (!uid) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const profileData = await getUserProfile(uid);
      setProfile(profileData);
    } catch (err) {
      setError(err.message);
      setProfile(null);
    } finally {
      setLoading(false);
    }
  }, [uid]);

  /**
   * 초기 로드
   */
  useEffect(() => {
    loadProfile();
  }, [loadProfile]);

  /**
   * Username으로 프로필 조회
   * @param {string} username - 사용자명
   * @returns {Promise<Object|null>} 프로필 데이터
   */
  const fetchByUsername = useCallback(async (username) => {
    setLoading(true);
    setError(null);

    try {
      const profileData = await getUserProfileByUsername(username);
      return profileData;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  /**
   * 프로필 업데이트
   * @param {Object} updateData - 업데이트할 데이터
   * @returns {Promise<boolean>} 성공 여부
   */
  const update = useCallback(async (updateData) => {
    if (!uid) {
      setError('사용자 ID가 필요합니다');
      return false;
    }

    setUpdating(true);
    setError(null);

    try {
      await updateUserProfile(uid, updateData);
      // 프로필 다시 로드
      await loadProfile();
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setUpdating(false);
    }
  }, [uid, loadProfile]);

  /**
   * Username 사용 가능 여부 확인
   * @param {string} username - 체크할 username
   * @returns {Promise<{available: boolean, error: string|null}>}
   */
  const checkUsername = useCallback(async (username) => {
    try {
      const available = await checkUsernameAvailable(username);
      return { available, error: null };
    } catch (err) {
      return { available: false, error: err.message };
    }
  }, []);

  /**
   * 프로필 초기화 (첫 로그인 시)
   * @param {string} email - 사용자 이메일
   * @returns {Promise<Object|null>} 생성된 프로필
   */
  const initialize = useCallback(async (email) => {
    if (!uid) {
      setError('사용자 ID가 필요합니다');
      return null;
    }

    setLoading(true);
    setError(null);

    try {
      const newProfile = await initializeUserProfile(uid, email);
      setProfile(newProfile);
      return newProfile;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [uid]);

  /**
   * 프로필 새로고침
   */
  const refresh = useCallback(() => {
    return loadProfile();
  }, [loadProfile]);

  /**
   * 에러 초기화
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // 상태
    profile,
    loading,
    error,
    updating,
    isAuthenticated: !!profile,

    // 메서드
    update,
    checkUsername,
    initialize,
    refresh,
    fetchByUsername,
    clearError,
  };
};

export default useUserProfile;
