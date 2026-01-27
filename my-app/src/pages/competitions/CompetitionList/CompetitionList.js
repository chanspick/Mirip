/**
 * 공모전 목록 페이지
 *
 * SPEC-COMP-001 REQ-C-001 ~ REQ-C-004 구현
 *
 * @module pages/competitions/CompetitionList
 */

import React, { useEffect, useCallback, useRef, useState, useMemo } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import CompetitionCard from '../../../components/competitions/CompetitionCard';
import CompetitionFilter from '../../../components/competitions/CompetitionFilter';
import { Header, Footer, Loading, AuthModal } from '../../../components/common';
import { useCompetitionList } from '../../../hooks/useCompetitions';
import { useAuth } from '../../../hooks';
import { GUEST_NAV_ITEMS, LOGGED_IN_NAV_ITEMS } from '../../../utils/navigation';
import styles from './CompetitionList.module.css';

/**
 * Footer 링크
 */
const FOOTER_LINKS = [
  { label: '이용약관', href: '/terms' },
  { label: '개인정보처리방침', href: '/privacy' },
];

/**
 * 스켈레톤 카드 컴포넌트
 */
const SkeletonCard = () => (
  <div className={styles.skeletonCard}>
    <div className={styles.skeletonThumbnail} />
    <div className={styles.skeletonContent}>
      <div className={styles.skeletonCategory} />
      <div className={styles.skeletonTitle} />
      <div className={styles.skeletonOrganizer} />
      <div className={styles.skeletonFooter} />
    </div>
  </div>
);

/**
 * 공모전 목록 페이지 컴포넌트
 */
const CompetitionList = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const observerRef = useRef(null);
  const loadMoreRef = useRef(null);

  // 인증 상태
  const { profile, isAuthenticated, signOut } = useAuth();
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  // 네비게이션 아이템 (로그인 여부에 따라 변경)
  const navItems = useMemo(
    () => isAuthenticated ? LOGGED_IN_NAV_ITEMS : GUEST_NAV_ITEMS,
    [isAuthenticated]
  );

  // 로그인/로그아웃 핸들러
  const handleLoginClick = useCallback(() => setIsAuthModalOpen(true), []);
  const handleLogout = useCallback(async () => {
    try {
      await signOut();
    } catch (error) {
      console.error('[CompetitionList] 로그아웃 실패:', error);
    }
  }, [signOut]);

  // URL에서 초기 필터 값 읽기
  const initialFilters = {
    category: searchParams.get('category') || 'all',
    status: searchParams.get('status') || 'all',
    sortBy: searchParams.get('sortBy') || 'endDate',
  };

  const {
    competitions,
    loading,
    error,
    filters,
    hasMore,
    updateFilters,
    loadMore,
  } = useCompetitionList(initialFilters);

  // 필터 변경 시 URL 업데이트
  const handleFilterChange = useCallback((newFilters) => {
    const updatedFilters = { ...filters, ...newFilters };
    const params = new URLSearchParams();

    if (updatedFilters.category !== 'all') {
      params.set('category', updatedFilters.category);
    }
    if (updatedFilters.status !== 'all') {
      params.set('status', updatedFilters.status);
    }
    if (updatedFilters.sortBy !== 'endDate') {
      params.set('sortBy', updatedFilters.sortBy);
    }

    setSearchParams(params);
    updateFilters(newFilters);
  }, [filters, setSearchParams, updateFilters]);

  // 무한 스크롤 설정
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !loading) {
          loadMore();
        }
      },
      { threshold: 0.1 }
    );

    observerRef.current = observer;

    if (loadMoreRef.current) {
      observer.observe(loadMoreRef.current);
    }

    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [hasMore, loading, loadMore]);

  return (
    <div className={styles.page}>
      {/* 헤더 */}
      <Header
        logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
        navItems={navItems}
        ctaButton={!isAuthenticated ? {
          label: 'AI 진단',
          onClick: () => window.location.href = '/diagnosis',
        } : null}
        user={isAuthenticated ? { photoURL: profile?.profileImageUrl, displayName: profile?.displayName } : null}
        onLoginClick={handleLoginClick}
        onLogout={handleLogout}
      />

      {/* 로그인 모달 */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
      />

      {/* 메인 콘텐츠 */}
      <main className={styles.main}>
        <div className={styles.container}>
          {/* 페이지 헤더 */}
          <header className={styles.header}>
            <span className={styles.sectionLabel}>Competition</span>
            <h1 className={styles.title}>공모전</h1>
            <p className={styles.subtitle}>
              다양한 분야의 공모전에 참여하고 실력을 뽐내보세요
            </p>
          </header>

          {/* 필터 */}
          <CompetitionFilter
            filters={filters}
            onFilterChange={handleFilterChange}
          />

          {/* 에러 상태 */}
          {error && (
            <div className={styles.error}>
              <p>{error}</p>
              <button onClick={() => window.location.reload()}>
                다시 시도
              </button>
            </div>
          )}

          {/* 공모전 그리드 */}
          <div className={styles.grid}>
            {/* 초기 로딩 스켈레톤 */}
            {loading && competitions.length === 0 && (
              <>
                {[...Array(8)].map((_, index) => (
                  <SkeletonCard key={`skeleton-${index}`} />
                ))}
              </>
            )}

            {/* 공모전 카드 목록 */}
            {competitions.map((competition) => (
              <CompetitionCard
                key={competition.id}
                competition={competition}
              />
            ))}
          </div>

          {/* 빈 상태 */}
          {!loading && competitions.length === 0 && !error && (
            <div className={styles.empty}>
              <p>조건에 맞는 공모전이 없습니다.</p>
            </div>
          )}

          {/* 더 불러오기 트리거 */}
          <div ref={loadMoreRef} className={styles.loadMoreTrigger}>
            {loading && competitions.length > 0 && (
              <Loading size="medium" />
            )}
          </div>

          {/* 모두 로드됨 */}
          {!hasMore && competitions.length > 0 && (
            <p className={styles.endMessage}>
              모든 공모전을 확인했습니다
            </p>
          )}
        </div>
      </main>

      {/* 푸터 */}
      <Footer
        links={FOOTER_LINKS}
        copyright="© 2025 MIRIP. All rights reserved."
      />
    </div>
  );
};

export default CompetitionList;
