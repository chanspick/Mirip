/**
 * useAwards 훅
 *
 * 수상 기록 조회 및 관리를 제공합니다.
 * SPEC-CRED-001 요구사항에 따라 구현되었습니다.
 *
 * @module hooks/useAwards
 */

import { useState, useEffect, useCallback } from 'react';
import {
  recordAward,
  getAwards,
  getPublicAwards,
} from '../services/awardService';

/**
 * 수상 기록 관리 훅
 * @param {string} userId - 사용자 ID
 * @param {Object} [options] - 옵션
 * @param {boolean} [options.publicOnly=false] - 공개 수상 기록만 조회
 * @returns {Object} 수상 기록 상태 및 메서드
 */
const useAwards = (userId, options = {}) => {
  const { publicOnly = false } = options;

  const [awards, setAwards] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [recording, setRecording] = useState(false);

  /**
   * 수상 기록 로드
   */
  const loadAwards = useCallback(async () => {
    if (!userId) {
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const fetchFn = publicOnly ? getPublicAwards : getAwards;
      const data = await fetchFn(userId);
      setAwards(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId, publicOnly]);

  /**
   * 초기 로드
   */
  useEffect(() => {
    loadAwards();
  }, [loadAwards]);

  /**
   * 수상 기록 추가
   * @param {Object} awardData - 수상 데이터
   * @returns {Promise<Object|null>} 생성된 수상 기록
   */
  const record = useCallback(async (awardData) => {
    if (!userId) {
      setError('사용자 ID가 필요합니다');
      return null;
    }

    setRecording(true);
    setError(null);

    try {
      const newAward = await recordAward(userId, awardData);
      // 목록 맨 앞에 추가 (최신순 정렬)
      setAwards((prev) => [newAward, ...prev]);
      return newAward;
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
    return loadAwards();
  }, [loadAwards]);

  /**
   * 에러 초기화
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  /**
   * 수상 등급별 수 계산
   * @returns {Object} 등급별 수상 수
   */
  const countByRank = useCallback(() => {
    const counts = {
      '대상': 0,
      '금상': 0,
      '은상': 0,
      '동상': 0,
      '입선': 0,
    };

    awards.forEach((award) => {
      if (counts[award.rank] !== undefined) {
        counts[award.rank]++;
      }
    });

    return counts;
  }, [awards]);

  /**
   * 특정 공모전의 수상 기록 찾기
   * @param {string} competitionId - 공모전 ID
   * @returns {Object|undefined} 수상 기록
   */
  const findByCompetition = useCallback((competitionId) => {
    return awards.find((a) => a.competitionId === competitionId);
  }, [awards]);

  /**
   * 최고 등급 수상 기록 반환
   * @returns {Object|null} 최고 등급 수상 기록
   */
  const getTopAward = useCallback(() => {
    const rankOrder = ['대상', '금상', '은상', '동상', '입선'];

    for (const rank of rankOrder) {
      const award = awards.find((a) => a.rank === rank);
      if (award) return award;
    }

    return null;
  }, [awards]);

  return {
    // 상태
    awards,
    loading,
    error,
    recording,
    count: awards.length,
    hasAwards: awards.length > 0,

    // 메서드
    record,
    refresh,
    clearError,
    countByRank,
    findByCompetition,
    getTopAward,
  };
};

export default useAwards;
