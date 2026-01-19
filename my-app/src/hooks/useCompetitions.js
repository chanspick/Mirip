/**
 * 공모전 관련 커스텀 훅
 *
 * @module hooks/useCompetitions
 */

import { useState, useEffect, useCallback } from 'react';
import {
  getCompetitions,
  getCompetitionById,
} from '../services/competitionService';

/**
 * 공모전 목록 조회 훅
 * @param {Object} initialFilters - 초기 필터 옵션
 * @returns {Object} 공모전 목록, 로딩 상태, 에러, 필터 변경 함수
 */
export const useCompetitionList = (initialFilters = {}) => {
  const [competitions, setCompetitions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    category: 'all',
    status: 'all',
    sortBy: 'endDate',
    sortOrder: 'asc',
    ...initialFilters,
  });
  const [lastDoc, setLastDoc] = useState(null);
  const [hasMore, setHasMore] = useState(true);

  // 초기 로드
  const fetchCompetitions = useCallback(async (reset = false) => {
    setLoading(true);
    setError(null);

    try {
      const result = await getCompetitions({
        ...filters,
        lastDoc: reset ? null : lastDoc,
      });

      if (reset) {
        setCompetitions(result.competitions);
      } else {
        setCompetitions((prev) => [...prev, ...result.competitions]);
      }
      setLastDoc(result.lastDoc);
      setHasMore(result.hasMore);
    } catch (err) {
      setError(err.message || '공모전 목록을 불러오는데 실패했습니다.');
    } finally {
      setLoading(false);
    }
  }, [filters, lastDoc]);

  // 필터 변경 시 리셋
  useEffect(() => {
    setLastDoc(null);
    fetchCompetitions(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters.category, filters.status, filters.sortBy, filters.sortOrder]);

  // 필터 변경 함수
  const updateFilters = useCallback((newFilters) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
  }, []);

  // 다음 페이지 로드
  const loadMore = useCallback(() => {
    if (!loading && hasMore) {
      fetchCompetitions(false);
    }
  }, [loading, hasMore, fetchCompetitions]);

  // 새로고침
  const refresh = useCallback(() => {
    setLastDoc(null);
    fetchCompetitions(true);
  }, [fetchCompetitions]);

  return {
    competitions,
    loading,
    error,
    filters,
    hasMore,
    updateFilters,
    loadMore,
    refresh,
  };
};

/**
 * 공모전 상세 조회 훅
 * @param {string} competitionId - 공모전 ID
 * @returns {Object} 공모전 상세, 로딩 상태, 에러
 */
export const useCompetitionDetail = (competitionId) => {
  const [competition, setCompetition] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchCompetition = async () => {
      if (!competitionId) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const result = await getCompetitionById(competitionId);
        if (!result) {
          setError('공모전을 찾을 수 없습니다.');
        } else {
          setCompetition(result);
        }
      } catch (err) {
        setError(err.message || '공모전 정보를 불러오는데 실패했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchCompetition();
  }, [competitionId]);

  return { competition, loading, error };
};

const competitionHooks = { useCompetitionList, useCompetitionDetail };
export default competitionHooks;
