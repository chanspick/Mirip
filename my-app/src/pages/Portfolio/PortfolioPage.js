// PortfolioPage 컴포넌트
// SPEC-CRED-001: M4 포트폴리오 관리
// 사용자 포트폴리오 관리 페이지 - 추가, 편집, 삭제, 상세 보기

import React, { useState, useCallback, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Header, Footer } from '../../components/common';
import { PortfolioGrid, PortfolioUploadForm, PortfolioModal } from '../../components/credential';
import { usePortfolios } from '../../hooks';
import { auth } from '../../config/firebase';
import styles from './PortfolioPage.module.css';

/**
 * 네비게이션 아이템
 */
const NAV_ITEMS = [
  { label: '공모전', href: '/competitions' },
  { label: 'AI 진단', href: '/diagnosis' },
  { label: '마이페이지', href: '/profile' },
];

/**
 * Footer 링크
 */
const FOOTER_LINKS = [
  { label: '이용약관', href: '/terms' },
  { label: '개인정보처리방침', href: '/privacy' },
];

/**
 * PortfolioPage 컴포넌트
 * 사용자의 포트폴리오를 관리하는 페이지
 */
const PortfolioPage = () => {
  const navigate = useNavigate();

  // 현재 로그인된 사용자 ID
  const currentUser = auth.currentUser;
  const userId = currentUser?.uid;

  // 포트폴리오 훅
  const {
    portfolios,
    loading,
    error,
    add,
    update,
    remove,
    loadMore,
    hasMore,
    refresh,
  } = usePortfolios(userId);

  // UI 상태
  const [isUploadFormOpen, setIsUploadFormOpen] = useState(false);
  const [editingPortfolio, setEditingPortfolio] = useState(null);
  const [viewingPortfolio, setViewingPortfolio] = useState(null);
  const [viewingIndex, setViewingIndex] = useState(0);
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [actionError, setActionError] = useState(null);

  // 필터 및 정렬 상태
  const [filterTag, setFilterTag] = useState(null);
  const [sortBy, setSortBy] = useState('newest');

  /**
   * AI 진단 페이지로 이동
   */
  const goToDiagnosis = useCallback(() => {
    navigate('/diagnosis');
  }, [navigate]);

  /**
   * Header CTA 버튼 설정
   */
  const ctaButtonConfig = useMemo(
    () => ({
      label: 'AI 진단',
      onClick: goToDiagnosis,
    }),
    [goToDiagnosis]
  );

  /**
   * 새 작품 추가 폼 열기
   */
  const handleOpenUploadForm = useCallback(() => {
    setEditingPortfolio(null);
    setIsUploadFormOpen(true);
    setActionError(null);
  }, []);

  /**
   * 업로드 폼 닫기
   */
  const handleCloseUploadForm = useCallback(() => {
    setIsUploadFormOpen(false);
    setEditingPortfolio(null);
    setActionError(null);
  }, []);

  /**
   * 포트폴리오 편집 시작
   */
  const handleEdit = useCallback((portfolio) => {
    setEditingPortfolio(portfolio);
    setIsUploadFormOpen(true);
    setActionError(null);
  }, []);

  /**
   * 포트폴리오 업로드/수정 제출
   */
  const handleSubmit = useCallback(async (data) => {
    setIsSubmitting(true);
    setActionError(null);

    try {
      if (editingPortfolio) {
        // 수정
        await update(editingPortfolio.id, data);
      } else {
        // 추가
        await add(data);
      }
      handleCloseUploadForm();
    } catch (err) {
      setActionError(err.message || '저장에 실패했습니다.');
    } finally {
      setIsSubmitting(false);
    }
  }, [editingPortfolio, add, update, handleCloseUploadForm]);

  /**
   * 삭제 대상 설정
   */
  const handleDeleteRequest = useCallback((portfolioId) => {
    setDeleteTarget(portfolioId);
  }, []);

  /**
   * 삭제 취소
   */
  const handleCancelDelete = useCallback(() => {
    setDeleteTarget(null);
  }, []);

  /**
   * 삭제 확인
   */
  const handleConfirmDelete = useCallback(async () => {
    if (!deleteTarget) return;

    setIsSubmitting(true);
    setActionError(null);

    try {
      await remove(deleteTarget);
      setDeleteTarget(null);
    } catch (err) {
      setActionError(err.message || '삭제에 실패했습니다.');
    } finally {
      setIsSubmitting(false);
    }
  }, [deleteTarget, remove]);

  /**
   * 포트폴리오 상세 보기 열기
   */
  const handleItemClick = useCallback((portfolio) => {
    const index = portfolios.findIndex((p) => p.id === portfolio.id);
    setViewingPortfolio(portfolio);
    setViewingIndex(index >= 0 ? index : 0);
  }, [portfolios]);

  /**
   * 상세 보기 모달 닫기
   */
  const handleCloseModal = useCallback(() => {
    setViewingPortfolio(null);
    setViewingIndex(0);
  }, []);

  /**
   * 이전 포트폴리오 보기
   */
  const handlePrev = useCallback(() => {
    if (viewingIndex > 0) {
      const newIndex = viewingIndex - 1;
      setViewingIndex(newIndex);
      setViewingPortfolio(portfolios[newIndex]);
    }
  }, [viewingIndex, portfolios]);

  /**
   * 다음 포트폴리오 보기
   */
  const handleNext = useCallback(() => {
    if (viewingIndex < portfolios.length - 1) {
      const newIndex = viewingIndex + 1;
      setViewingIndex(newIndex);
      setViewingPortfolio(portfolios[newIndex]);
    }
  }, [viewingIndex, portfolios]);

  /**
   * 필터 버튼 클릭
   */
  const handleFilterClick = useCallback(() => {
    // 필터 메뉴 토글 (간단한 구현)
    setFilterTag(filterTag ? null : '');
  }, [filterTag]);

  /**
   * 정렬 버튼 클릭
   */
  const handleSortClick = useCallback(() => {
    // 정렬 토글 (newest <-> oldest)
    setSortBy((prev) => (prev === 'newest' ? 'oldest' : 'newest'));
  }, []);

  // 로그인하지 않은 경우
  if (!userId) {
    return (
      <div className={styles.portfolioPage}>
        <Header
          logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
          navItems={NAV_ITEMS}
          ctaButton={ctaButtonConfig}
        />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.loginRequired}>
              <span className={styles.loginIcon}>{'\uD83D\uDD12'}</span>
              <h2>로그인이 필요합니다</h2>
              <p>포트폴리오를 관리하려면 로그인해주세요.</p>
              <Link to="/" className={styles.loginButton}>
                홈으로 이동
              </Link>
            </div>
          </div>
        </main>
        <Footer
          links={FOOTER_LINKS}
          copyright="© 2025 MIRIP. All rights reserved."
        />
      </div>
    );
  }

  return (
    <div className={styles.portfolioPage} data-testid="portfolio-page">
      <Header
        logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
        navItems={NAV_ITEMS}
        ctaButton={ctaButtonConfig}
      />

      <main className={styles.main}>
        <div className={styles.container}>
          {/* 페이지 헤더 */}
          <div className={styles.pageHeader}>
            <div className={styles.titleSection}>
              <h1 className={styles.pageTitle}>
                <span className={styles.titleIcon}>{'\uD83C\uDFA8'}</span>
                내 포트폴리오
              </h1>
              <span className={styles.portfolioCount} data-testid="portfolio-count">
                총 {portfolios.length}개 작품
              </span>
            </div>

            <div className={styles.actions}>
              <button
                type="button"
                className={styles.filterButton}
                onClick={handleFilterClick}
                data-testid="filter-button"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
                </svg>
                필터
              </button>
              <button
                type="button"
                className={styles.sortButton}
                onClick={handleSortClick}
                data-testid="sort-button"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="4" y1="6" x2="20" y2="6" />
                  <line x1="4" y1="12" x2="16" y2="12" />
                  <line x1="4" y1="18" x2="12" y2="18" />
                </svg>
                {sortBy === 'newest' ? '최신순' : '오래된순'}
              </button>
              <button
                type="button"
                className={styles.addButton}
                onClick={handleOpenUploadForm}
                data-testid="add-portfolio-button"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
                새 작품 추가
              </button>
            </div>
          </div>

          {/* 에러 메시지 */}
          {(error || actionError) && (
            <div className={styles.errorMessage}>
              <span className={styles.errorIcon}>{'\u26A0\uFE0F'}</span>
              {error || actionError}
            </div>
          )}

          {/* 포트폴리오 그리드 */}
          <PortfolioGrid
            portfolios={portfolios}
            isOwner={true}
            onEdit={handleEdit}
            onDelete={handleDeleteRequest}
            onItemClick={handleItemClick}
            loading={loading}
            emptyMessage="아직 등록된 작품이 없습니다. 첫 번째 작품을 추가해보세요!"
          />

          {/* 더 보기 버튼 */}
          {hasMore && !loading && (
            <div className={styles.loadMoreContainer}>
              <button
                type="button"
                className={styles.loadMoreButton}
                onClick={loadMore}
                data-testid="load-more-button"
              >
                더 보기
              </button>
            </div>
          )}
        </div>
      </main>

      <Footer
        links={FOOTER_LINKS}
        copyright="© 2025 MIRIP. All rights reserved."
      />

      {/* 업로드/편집 폼 모달 */}
      {isUploadFormOpen && (
        <div className={styles.modalOverlay} data-testid="upload-modal-overlay">
          <div className={styles.modalContent}>
            <div className={styles.modalHeader}>
              <h2 className={styles.modalTitle}>
                {editingPortfolio ? '작품 수정' : '새 작품 추가'}
              </h2>
              <button
                type="button"
                className={styles.modalCloseButton}
                onClick={handleCloseUploadForm}
                aria-label="닫기"
              >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <PortfolioUploadForm
              onSubmit={handleSubmit}
              onCancel={handleCloseUploadForm}
              initialData={editingPortfolio}
              loading={isSubmitting}
            />
          </div>
        </div>
      )}

      {/* 삭제 확인 모달 */}
      {deleteTarget && (
        <div className={styles.modalOverlay} data-testid="delete-confirm-modal">
          <div className={styles.confirmModal}>
            <div className={styles.confirmIcon}>{'\uD83D\uDDD1\uFE0F'}</div>
            <h3 className={styles.confirmTitle}>작품 삭제</h3>
            <p className={styles.confirmMessage}>
              이 작품을 정말 삭제하시겠습니까?<br />
              삭제된 작품은 복구할 수 없습니다.
            </p>
            <div className={styles.confirmButtons}>
              <button
                type="button"
                className={styles.cancelButton}
                onClick={handleCancelDelete}
                data-testid="cancel-delete-button"
              >
                취소
              </button>
              <button
                type="button"
                className={styles.deleteButton}
                onClick={handleConfirmDelete}
                disabled={isSubmitting}
                data-testid="confirm-delete-button"
              >
                {isSubmitting ? '삭제 중...' : '삭제'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 상세 보기 모달 */}
      <PortfolioModal
        isOpen={!!viewingPortfolio}
        portfolio={viewingPortfolio}
        onClose={handleCloseModal}
        portfolios={portfolios}
        currentIndex={viewingIndex}
        onPrev={viewingIndex > 0 ? handlePrev : undefined}
        onNext={viewingIndex < portfolios.length - 1 ? handleNext : undefined}
      />
    </div>
  );
};

export default PortfolioPage;
