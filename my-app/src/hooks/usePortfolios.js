/**
 * usePortfolios 훅
 *
 * 포트폴리오 CRUD 및 목록 관리를 제공합니다.
 * SPEC-CRED-001 요구사항에 따라 구현되었습니다.
 *
 * @module hooks/usePortfolios
 */

import { useState, useEffect, useCallback } from 'react';
import {
  addPortfolio,
  updatePortfolio,
  deletePortfolio,
  getPortfolios,
  getPublicPortfolios,
} from '../services/portfolioService';

/**
 * 포트폴리오 관리 훅
 * @param {string} userId - 사용자 ID
 * @param {Object} [options] - 옵션
 * @param {number} [options.pageSize=12] - 페이지 크기
 * @param {boolean} [options.publicOnly=false] - 공개 작품만 조회
 * @returns {Object} 포트폴리오 상태 및 메서드
 */
const usePortfolios = (userId, options = {}) => {
  const { pageSize = 12, publicOnly = false } = options;

  const [portfolios, setPortfolios] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const [hasMore, setHasMore] = useState(false);
  const [lastDoc, setLastDoc] = useState(null);
  const [processing, setProcessing] = useState(false);

  /**
   * 포트폴리오 목록 로드
   */
  const loadPortfolios = useCallback(async (reset = true) => {
    if (!userId) {
      setLoading(false);
      return;
    }

    if (reset) {
      setLoading(true);
      setPortfolios([]);
      setLastDoc(null);
    } else {
      setLoadingMore(true);
    }
    setError(null);

    try {
      const fetchFn = publicOnly ? getPublicPortfolios : getPortfolios;
      const result = await fetchFn(userId, {
        pageSize,
        lastDoc: reset ? null : lastDoc,
        publicOnly,
      });

      if (reset) {
        setPortfolios(result.portfolios);
      } else {
        setPortfolios((prev) => [...prev, ...result.portfolios]);
      }
      setLastDoc(result.lastDoc);
      setHasMore(result.hasMore);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [userId, pageSize, publicOnly, lastDoc]);

  /**
   * 초기 로드
   */
  useEffect(() => {
    loadPortfolios(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, publicOnly]);

  /**
   * 더 불러오기
   */
  const loadMore = useCallback(() => {
    if (!hasMore || loadingMore) return;
    loadPortfolios(false);
  }, [hasMore, loadingMore, loadPortfolios]);

  /**
   * 포트폴리오 추가
   * @param {Object} portfolioData - 포트폴리오 데이터
   * @returns {Promise<Object|null>} 생성된 포트폴리오
   */
  const add = useCallback(async (portfolioData) => {
    if (!userId) {
      setError('사용자 ID가 필요합니다');
      return null;
    }

    setProcessing(true);
    setError(null);

    try {
      const newPortfolio = await addPortfolio(userId, portfolioData);
      // 목록 맨 앞에 추가
      setPortfolios((prev) => [newPortfolio, ...prev]);
      return newPortfolio;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setProcessing(false);
    }
  }, [userId]);

  /**
   * 포트폴리오 수정
   * @param {string} portfolioId - 포트폴리오 ID
   * @param {Object} updateData - 업데이트 데이터
   * @returns {Promise<boolean>} 성공 여부
   */
  const update = useCallback(async (portfolioId, updateData) => {
    if (!userId) {
      setError('사용자 ID가 필요합니다');
      return false;
    }

    setProcessing(true);
    setError(null);

    try {
      await updatePortfolio(userId, portfolioId, updateData);
      // 로컬 상태 업데이트
      setPortfolios((prev) =>
        prev.map((p) =>
          p.id === portfolioId ? { ...p, ...updateData } : p
        )
      );
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setProcessing(false);
    }
  }, [userId]);

  /**
   * 포트폴리오 삭제
   * @param {string} portfolioId - 포트폴리오 ID
   * @returns {Promise<boolean>} 성공 여부
   */
  const remove = useCallback(async (portfolioId) => {
    if (!userId) {
      setError('사용자 ID가 필요합니다');
      return false;
    }

    setProcessing(true);
    setError(null);

    try {
      await deletePortfolio(userId, portfolioId);
      // 로컬 상태에서 제거
      setPortfolios((prev) => prev.filter((p) => p.id !== portfolioId));
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setProcessing(false);
    }
  }, [userId]);

  /**
   * 목록 새로고침
   */
  const refresh = useCallback(() => {
    return loadPortfolios(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId, publicOnly]);

  /**
   * 에러 초기화
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  /**
   * 특정 포트폴리오 찾기
   * @param {string} portfolioId - 포트폴리오 ID
   * @returns {Object|undefined} 포트폴리오 데이터
   */
  const findById = useCallback((portfolioId) => {
    return portfolios.find((p) => p.id === portfolioId);
  }, [portfolios]);

  return {
    // 상태
    portfolios,
    loading,
    loadingMore,
    error,
    hasMore,
    processing,
    count: portfolios.length,

    // 메서드
    add,
    update,
    remove,
    loadMore,
    refresh,
    findById,
    clearError,
  };
};

export default usePortfolios;
