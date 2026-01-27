// PortfolioGrid 컴포넌트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 그리드
// 반응형 그리드 레이아웃으로 포트폴리오 목록을 표시합니다

import React, { useState, useMemo } from 'react';
import PropTypes from 'prop-types';
import PortfolioCard from '../PortfolioCard';
import styles from './PortfolioGrid.module.css';

// 정렬 옵션
const SORT_OPTIONS = [
  { value: 'newest', label: '최신순' },
  { value: 'oldest', label: '오래된순' },
];

/**
 * PortfolioGrid 컴포넌트
 * 반응형 그리드 레이아웃으로 포트폴리오 목록을 표시합니다
 *
 * @param {Object} props - 컴포넌트 props
 * @param {Array} props.portfolios - 포트폴리오 배열
 * @param {boolean} [props.isOwner=false] - 소유자 모드 활성화
 * @param {function} [props.onCardClick] - 카드 클릭 콜백
 * @param {function} [props.onEdit] - 편집 콜백
 * @param {function} [props.onDelete] - 삭제 콜백
 * @param {boolean} [props.loading=false] - 로딩 상태
 * @param {boolean} [props.loadingMore=false] - 추가 로딩 상태
 * @param {boolean} [props.hasMore=false] - 더 불러올 데이터 여부
 * @param {function} [props.onLoadMore] - 더 보기 콜백
 * @param {boolean} [props.showFilter=false] - 필터 UI 표시 여부
 * @param {boolean} [props.showSort=false] - 정렬 UI 표시 여부
 * @param {string} [props.emptyMessage] - 빈 상태 메시지
 */
const PortfolioGrid = ({
  portfolios = [],
  isOwner = false,
  onCardClick,
  onEdit,
  onDelete,
  loading = false,
  loadingMore = false,
  hasMore = false,
  onLoadMore,
  showFilter = false,
  showSort = false,
  emptyMessage = '포트폴리오가 없습니다.',
}) => {
  // 필터 상태
  const [selectedTags, setSelectedTags] = useState([]);
  // 정렬 상태
  const [sortOrder, setSortOrder] = useState('newest');

  // 모든 고유 태그 추출
  const allTags = useMemo(() => {
    const tagSet = new Set();
    portfolios.forEach((portfolio) => {
      if (portfolio.tags) {
        portfolio.tags.forEach((tag) => tagSet.add(tag));
      }
    });
    return Array.from(tagSet);
  }, [portfolios]);

  // 필터링된 포트폴리오
  const filteredPortfolios = useMemo(() => {
    let result = [...portfolios];

    // 태그 필터링
    if (selectedTags.length > 0) {
      result = result.filter((portfolio) =>
        portfolio.tags?.some((tag) => selectedTags.includes(tag))
      );
    }

    // 정렬
    if (sortOrder === 'oldest') {
      result.reverse();
    }

    return result;
  }, [portfolios, selectedTags, sortOrder]);

  // 태그 필터 토글
  const handleTagToggle = (tag) => {
    setSelectedTags((prev) =>
      prev.includes(tag)
        ? prev.filter((t) => t !== tag)
        : [...prev, tag]
    );
  };

  // 정렬 변경
  const handleSortChange = (e) => {
    setSortOrder(e.target.value);
  };

  // 더 보기 클릭
  const handleLoadMore = () => {
    if (onLoadMore) {
      onLoadMore();
    }
  };

  return (
    <div className={styles.container}>
      {/* 필터 및 정렬 영역 */}
      {(showFilter || showSort) && (
        <div className={styles.toolbar}>
          {/* 태그 필터 */}
          {showFilter && allTags.length > 0 && (
            <div className={styles.filterContainer} data-testid="filter-container">
              <span className={styles.filterLabel}>태그 필터:</span>
              <div className={styles.filterTags}>
                {allTags.map((tag) => (
                  <button
                    key={tag}
                    type="button"
                    className={`${styles.filterTag} ${
                      selectedTags.includes(tag) ? styles.active : ''
                    }`}
                    onClick={() => handleTagToggle(tag)}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 정렬 */}
          {showSort && (
            <div className={styles.sortContainer}>
              <select
                className={styles.sortSelect}
                value={sortOrder}
                onChange={handleSortChange}
                data-testid="sort-select"
              >
                {SORT_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      {/* 로딩 표시 (초기 로딩) */}
      {loading && portfolios.length === 0 && (
        <div className={styles.loadingContainer} data-testid="loading-indicator">
          <div className={styles.spinner} />
          <span>포트폴리오를 불러오는 중...</span>
        </div>
      )}

      {/* 빈 상태 */}
      {!loading && filteredPortfolios.length === 0 && (
        <div className={styles.emptyState} data-testid="empty-state">
          <svg
            className={styles.emptyIcon}
            width="64"
            height="64"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
          <p className={styles.emptyMessage}>{emptyMessage}</p>
        </div>
      )}

      {/* 포트폴리오 그리드 */}
      {filteredPortfolios.length > 0 && (
        <div className={`${styles.grid} grid`} data-testid="portfolio-grid">
          {filteredPortfolios.map((portfolio) => (
            <PortfolioCard
              key={portfolio.id}
              portfolio={portfolio}
              isOwner={isOwner}
              onClick={onCardClick}
              onEdit={onEdit}
              onDelete={onDelete}
            />
          ))}
        </div>
      )}

      {/* 로딩 인디케이터 (추가 로딩 또는 데이터가 있을 때) */}
      {loading && portfolios.length > 0 && (
        <div className={styles.loadingOverlay} data-testid="loading-indicator">
          <div className={styles.spinner} />
        </div>
      )}

      {/* 더 보기 버튼 */}
      {hasMore && (
        <div className={styles.loadMoreContainer}>
          <button
            type="button"
            className={styles.loadMoreButton}
            onClick={handleLoadMore}
            disabled={loadingMore}
            data-testid="load-more-button"
          >
            {loadingMore ? (
              <>
                <span className={styles.spinnerSmall} />
                불러오는 중...
              </>
            ) : (
              '더 보기'
            )}
          </button>
        </div>
      )}
    </div>
  );
};

PortfolioGrid.propTypes = {
  /** 포트폴리오 배열 */
  portfolios: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      title: PropTypes.string.isRequired,
      imageUrl: PropTypes.string.isRequired,
      thumbnailUrl: PropTypes.string,
      description: PropTypes.string,
      tags: PropTypes.arrayOf(PropTypes.string),
      isPublic: PropTypes.bool,
    })
  ),
  /** 소유자 모드 */
  isOwner: PropTypes.bool,
  /** 카드 클릭 콜백 */
  onCardClick: PropTypes.func,
  /** 편집 콜백 */
  onEdit: PropTypes.func,
  /** 삭제 콜백 */
  onDelete: PropTypes.func,
  /** 로딩 상태 */
  loading: PropTypes.bool,
  /** 추가 로딩 상태 */
  loadingMore: PropTypes.bool,
  /** 더 불러올 데이터 여부 */
  hasMore: PropTypes.bool,
  /** 더 보기 콜백 */
  onLoadMore: PropTypes.func,
  /** 필터 UI 표시 여부 */
  showFilter: PropTypes.bool,
  /** 정렬 UI 표시 여부 */
  showSort: PropTypes.bool,
  /** 빈 상태 메시지 */
  emptyMessage: PropTypes.string,
};

export default PortfolioGrid;
