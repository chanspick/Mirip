/**
 * useActivities 훅
 *
 * 활동 기록 목록 및 페이지네이션을 관리합니다.
 * SPEC-CRED-001 요구사항에 따라 구현되었습니다.
 *
 * @module hooks/useActivities
 */

import { useState, useEffect, useCallback } from 'react';
import {
  recordActivity,
  getActivities,
  getActivityHeatmapData,
  calculateStreak,
  getDailyActivityCount,
} from '../services/activityService';

/**
 * 활동 기록 관리 훅
 * @param {string} userId - 사용자 ID
 * @param {Object} [options] - 옵션
 * @param {number} [options.pageSize=20] - 페이지 크기
 * @param {string} [options.type] - 활동 타입 필터
 * @returns {Object} 활동 목록 상태 및 메서드
 */
const useActivities = (userId, options = {}) => {
  const { pageSize = 20, type } = options;

  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const [hasMore, setHasMore] = useState(false);
  const [lastDoc, setLastDoc] = useState(null);
  const [recording, setRecording] = useState(false);

  /**
   * 활동 목록 로드
   */
  const loadActivities = useCallback(async (reset = true) => {
    if (!userId) {
      setLoading(false);
      return;
    }

    if (reset) {
      setLoading(true);
      setActivities([]);
      setLastDoc(null);
    } else {
      setLoadingMore(true);
    }
    setError(null);

    try {
      const result = await getActivities(userId, {
        pageSize,
        lastDoc: reset ? null : lastDoc,
        type,
      });

      if (reset) {
        setActivities(result.activities);
      } else {
        setActivities((prev) => [...prev, ...result.activities]);
      }
      setLastDoc(result.lastDoc);
      setHasMore(result.hasMore);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [userId, pageSize, type, lastDoc]);

  /**
   * 초기 로드
   */
  useEffect(() => {
    loadActivities(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, type]);

  /**
   * 더 불러오기
   */
  const loadMore = useCallback(() => {
    if (!hasMore || loadingMore) return;
    loadActivities(false);
  }, [hasMore, loadingMore, loadActivities]);

  /**
   * 활동 기록 추가
   * @param {Object} activityData - 활동 데이터
   * @returns {Promise<Object|null>} 생성된 활동
   */
  const record = useCallback(async (activityData) => {
    if (!userId) {
      setError('사용자 ID가 필요합니다');
      return null;
    }

    setRecording(true);
    setError(null);

    try {
      const newActivity = await recordActivity(userId, activityData);
      // 목록 맨 앞에 추가
      setActivities((prev) => [newActivity, ...prev]);
      return newActivity;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setRecording(false);
    }
  }, [userId]);

  /**
   * 목록 새로고침
   */
  const refresh = useCallback(() => {
    return loadActivities(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, type]);

  /**
   * 에러 초기화
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // 상태
    activities,
    loading,
    loadingMore,
    error,
    hasMore,
    recording,

    // 메서드
    record,
    loadMore,
    refresh,
    clearError,
  };
};

/**
 * 활동 히트맵 데이터 훅
 * @param {string} userId - 사용자 ID
 * @param {number} [year] - 연도
 * @returns {Object} 히트맵 데이터 상태
 */
export const useActivityHeatmap = (userId, year) => {
  const [heatmapData, setHeatmapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadHeatmap = useCallback(async () => {
    if (!userId) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await getActivityHeatmapData(userId, year);
      setHeatmapData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId, year]);

  useEffect(() => {
    loadHeatmap();
  }, [loadHeatmap]);

  const refresh = useCallback(() => {
    return loadHeatmap();
  }, [loadHeatmap]);

  return {
    heatmapData,
    loading,
    error,
    refresh,
  };
};

/**
 * 스트릭 데이터 훅
 * @param {string} userId - 사용자 ID
 * @returns {Object} 스트릭 상태
 */
export const useStreak = (userId) => {
  const [streak, setStreak] = useState({ currentStreak: 0, longestStreak: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadStreak = useCallback(async () => {
    if (!userId) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await calculateStreak(userId);
      setStreak(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    loadStreak();
  }, [loadStreak]);

  const refresh = useCallback(() => {
    return loadStreak();
  }, [loadStreak]);

  return {
    ...streak,
    loading,
    error,
    refresh,
  };
};

/**
 * 일별 활동 수 훅
 * @param {string} userId - 사용자 ID
 * @param {string} [date] - YYYY-MM-DD 형식 날짜
 * @returns {Object} 일별 활동 상태
 */
export const useDailyActivity = (userId, date) => {
  const [dailyActivity, setDailyActivity] = useState({ date: '', count: 0, activityIds: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadDailyActivity = useCallback(async () => {
    if (!userId) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await getDailyActivityCount(userId, date);
      setDailyActivity(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId, date]);

  useEffect(() => {
    loadDailyActivity();
  }, [loadDailyActivity]);

  const refresh = useCallback(() => {
    return loadDailyActivity();
  }, [loadDailyActivity]);

  return {
    ...dailyActivity,
    loading,
    error,
    refresh,
  };
};

export default useActivities;
