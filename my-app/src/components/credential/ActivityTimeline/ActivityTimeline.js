// ActivityTimeline 컴포넌트
// SPEC-CRED-001: M2 활동 타임라인
// 최근 활동을 세로 타임라인으로 표시합니다

import React, { useState, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import useActivities from '../../../hooks';
import styles from './ActivityTimeline.module.css';

// 활동 타입별 설정
const ACTIVITY_TYPE_CONFIG = {
  diagnosis: {
    label: 'AI 진단',
    icon: '\uD83D\uDD0D', // 돋보기
    color: '#3b82f6',
  },
  competition: {
    label: '공모전',
    icon: '\uD83C\uDFC6', // 트로피
    color: '#f59e0b',
  },
  portfolio: {
    label: '포트폴리오',
    icon: '\uD83D\uDDBC\uFE0F', // 액자
    color: '#8b5cf6',
  },
  submission: {
    label: '제출',
    icon: '\uD83D\uDCE4', // 보내기
    color: '#10b981',
  },
  award: {
    label: '수상',
    icon: '\uD83C\uDF96\uFE0F', // 메달
    color: '#ef4444',
  },
};

// 기본 필터 타입
const DEFAULT_FILTER_TYPES = ['diagnosis', 'competition', 'portfolio', 'submission', 'award'];

/**
 * 상대적 시간 포맷팅
 * @param {Date} date - 날짜 객체
 * @returns {string} 상대적 시간 문자열
 */
const formatRelativeTime = (date) => {
  const now = new Date();
  const diffMs = now - date;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);
  const diffWeek = Math.floor(diffDay / 7);
  const diffMonth = Math.floor(diffDay / 30);

  if (diffSec < 60) return '방금 전';
  if (diffMin < 60) return `${diffMin}분 전`;
  if (diffHour < 24) return `${diffHour}시간 전`;
  if (diffDay < 7) return `${diffDay}일 전`;
  if (diffWeek < 4) return `${diffWeek}주 전`;
  if (diffMonth < 12) return `${diffMonth}개월 전`;

  // 1년 이상이면 날짜로 표시
  return date.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
};

/**
 * ActivityTimeline 컴포넌트
 * 최근 활동을 세로 타임라인으로 표시합니다
 *
 * @param {Object} props - 컴포넌트 props
 * @param {string} props.userId - 사용자 ID
 * @param {number} [props.limit=10] - 초기 로드 개수
 * @param {string[]} [props.filterTypes] - 표시할 활동 타입 필터
 */
const ActivityTimeline = ({ userId, limit = 10, filterTypes }) => {
  const [selectedFilter, setSelectedFilter] = useState(null);

  // 사용 가능한 필터 타입
  const availableFilters = useMemo(() => {
    return filterTypes || DEFAULT_FILTER_TYPES;
  }, [filterTypes]);

  // useActivities 훅 호출
  const {
    activities,
    loading,
    loadingMore,
    error,
    hasMore,
    loadMore,
  } = useActivities(userId, {
    pageSize: limit,
    type: selectedFilter,
  });

  /**
   * 필터 변경 핸들러
   */
  const handleFilterChange = useCallback((type) => {
    setSelectedFilter(type === selectedFilter ? null : type);
  }, [selectedFilter]);

  /**
   * 전체 필터 선택 핸들러
   */
  const handleSelectAll = useCallback(() => {
    setSelectedFilter(undefined);
  }, []);

  // userId가 없는 경우
  if (!userId) {
    return (
      <div className={styles.container}>
        <p className={styles.emptyMessage}>사용자 정보가 필요합니다</p>
      </div>
    );
  }

  // 로딩 상태
  if (loading && activities.length === 0) {
    return (
      <div className={styles.container} data-testid="timeline-loading">
        <div className={styles.loadingContainer}>
          <div className={styles.loadingSpinner} />
          <p>활동 로딩 중...</p>
        </div>
      </div>
    );
  }

  // 에러 상태
  if (error) {
    return (
      <div className={styles.container}>
        <p className={styles.errorMessage}>{error}</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* 필터 버튼 */}
      <div className={styles.filterContainer}>
        <button
          type="button"
          className={`${styles.filterButton} ${selectedFilter === undefined || selectedFilter === null ? styles.active : ''}`}
          onClick={handleSelectAll}
          data-testid="filter-all"
        >
          전체
        </button>
        {availableFilters.map((type) => {
          const config = ACTIVITY_TYPE_CONFIG[type];
          if (!config) return null;

          return (
            <button
              key={type}
              type="button"
              className={`${styles.filterButton} ${selectedFilter === type ? styles.active : ''}`}
              onClick={() => handleFilterChange(type)}
              data-testid={`filter-${type}`}
              style={{
                '--filter-color': config.color,
              }}
            >
              <span className={styles.filterIcon}>{config.icon}</span>
              {config.label}
            </button>
          );
        })}
      </div>

      {/* 활동이 없는 경우 */}
      {activities.length === 0 && !loading && (
        <div className={styles.emptyState}>
          <span className={styles.emptyIcon}>\uD83D\uDCDD</span>
          <p>아직 활동 기록이 없습니다</p>
        </div>
      )}

      {/* 타임라인 */}
      {activities.length > 0 && (
        <div className={styles.timeline} data-testid="timeline-container">
          {activities.map((activity) => {
            const config = ACTIVITY_TYPE_CONFIG[activity.type] || {
              label: '기타',
              icon: '\uD83D\uDCCC',
              color: '#6b7280',
            };

            // createdAt이 Firestore Timestamp인 경우 Date로 변환
            const createdAtDate = activity.createdAt?.toDate
              ? activity.createdAt.toDate()
              : new Date(activity.createdAt);

            return (
              <div
                key={activity.id}
                className={styles.timelineItem}
                data-testid="timeline-item"
              >
                {/* 아이콘 */}
                <div
                  className={styles.iconWrapper}
                  style={{ backgroundColor: config.color }}
                  data-testid={`activity-icon-${activity.type}`}
                >
                  <span className={styles.icon}>{config.icon}</span>
                </div>

                {/* 콘텐츠 */}
                <div className={styles.content}>
                  <div className={styles.header}>
                    <span
                      className={styles.typeLabel}
                      style={{ color: config.color }}
                    >
                      {config.label}
                    </span>
                    <span
                      className={styles.time}
                      data-testid="activity-time"
                      title={createdAtDate.toLocaleString('ko-KR')}
                    >
                      {formatRelativeTime(createdAtDate)}
                    </span>
                  </div>
                  <h4 className={styles.title}>{activity.title}</h4>
                  {activity.description && (
                    <p className={styles.description}>{activity.description}</p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* 더 불러오기 */}
      {loadingMore && (
        <div className={styles.loadingMore} data-testid="loading-more">
          <div className={styles.loadingSpinnerSmall} />
          <span>불러오는 중...</span>
        </div>
      )}

      {hasMore && !loadingMore && (
        <button
          type="button"
          className={styles.loadMoreButton}
          onClick={loadMore}
          data-testid="load-more-button"
        >
          더 보기
        </button>
      )}
    </div>
  );
};

ActivityTimeline.propTypes = {
  /** 사용자 ID */
  userId: PropTypes.string,
  /** 초기 로드 개수 */
  limit: PropTypes.number,
  /** 표시할 활동 타입 필터 */
  filterTypes: PropTypes.arrayOf(PropTypes.string),
};

export default ActivityTimeline;
