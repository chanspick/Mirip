// StreakDisplay 컴포넌트
// SPEC-CRED-001: M2 스트릭 표시
// 현재 스트릭과 최장 스트릭 기록을 표시합니다

import React from 'react';
import PropTypes from 'prop-types';
import { useStreak } from '../../../hooks';
import styles from './StreakDisplay.module.css';

// 스트릭 강조 기준 (7일 이상이면 강조)
const HIGHLIGHT_THRESHOLD = 7;

/**
 * StreakDisplay 컴포넌트
 * 현재 스트릭과 최장 스트릭 기록을 표시합니다
 *
 * @param {Object} props - 컴포넌트 props
 * @param {string} props.userId - 사용자 ID
 * @param {boolean} [props.showLongest=true] - 최장 스트릭 표시 여부
 * @param {boolean} [props.compact=false] - 축소 모드 여부
 * @param {number} [props.totalActivities] - 총 활동 수 (선택적)
 */
const StreakDisplay = ({
  userId,
  showLongest = true,
  compact = false,
  totalActivities,
}) => {
  const { currentStreak, longestStreak, loading, error } = useStreak(userId);

  // userId가 없는 경우
  if (!userId) {
    return (
      <div className={styles.container}>
        <p className={styles.emptyMessage}>사용자 정보가 필요합니다</p>
      </div>
    );
  }

  // 로딩 상태
  if (loading) {
    return (
      <div className={styles.container} data-testid="streak-loading">
        <div className={styles.loadingContainer}>
          <div className={styles.loadingSpinner} />
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

  // 강조 여부 결정
  const isHighlighted = currentStreak >= HIGHLIGHT_THRESHOLD;
  const isActive = currentStreak > 0;

  // 컨테이너 클래스 조합
  const containerClasses = [
    styles.container,
    styles.streakContainer,
    isHighlighted ? styles.highlighted : '',
    compact ? styles.compact : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={containerClasses} data-testid="streak-container">
      {/* 현재 스트릭 */}
      <div className={styles.streakItem}>
        <div className={styles.streakHeader}>
          <span
            className={`${styles.fireEmoji} ${isActive ? '' : styles.inactive}`}
            data-testid="fire-emoji"
          >
            {'\uD83D\uDD25'}
          </span>
          <span className={styles.streakLabel}>현재 스트릭</span>
        </div>
        <div className={styles.streakValue}>
          <span className={styles.number} data-testid="current-streak-value">
            {currentStreak}
          </span>
          <span className={styles.unit}>일</span>
        </div>
      </div>

      {/* 최장 스트릭 */}
      {showLongest && (
        <div className={styles.streakItem}>
          <div className={styles.streakHeader}>
            <span className={styles.trophyEmoji}>{'\uD83C\uDFC6'}</span>
            <span className={styles.streakLabel}>최장 기록</span>
          </div>
          <div className={styles.streakValue}>
            <span className={styles.number} data-testid="longest-streak-value">
              {longestStreak}
            </span>
            <span className={styles.unit}>일</span>
          </div>
        </div>
      )}

      {/* 총 활동 수 (선택적) */}
      {totalActivities !== undefined && (
        <div className={styles.streakItem}>
          <div className={styles.streakHeader}>
            <span className={styles.chartEmoji}>{'\uD83D\uDCCA'}</span>
            <span className={styles.streakLabel}>총 활동</span>
          </div>
          <div className={styles.streakValue}>
            <span className={styles.number} data-testid="total-activities">
              {totalActivities}
            </span>
            <span className={styles.unit}>개</span>
          </div>
        </div>
      )}
    </div>
  );
};

StreakDisplay.propTypes = {
  /** 사용자 ID */
  userId: PropTypes.string,
  /** 최장 스트릭 표시 여부 */
  showLongest: PropTypes.bool,
  /** 축소 모드 여부 */
  compact: PropTypes.bool,
  /** 총 활동 수 (선택적) */
  totalActivities: PropTypes.number,
};

export default StreakDisplay;
