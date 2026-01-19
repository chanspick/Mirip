// ActivityHeatmap 컴포넌트
// SPEC-CRED-001: M2 활동 히트맵 (GitHub-style 잔디밭)
// 52주 x 7일 그리드로 연간 활동 현황을 시각화합니다

import React, { useState, useCallback, useMemo } from 'react';
import PropTypes from 'prop-types';
import { useActivityHeatmap } from '../../../hooks';
import styles from './ActivityHeatmap.module.css';

// 요일 레이블 (일요일부터 시작)
const DAY_LABELS = ['일', '월', '화', '수', '목', '금', '토'];

// 월 레이블
const MONTH_LABELS = ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'];

/**
 * 활동 수에 따른 레벨 계산
 * @param {number} count - 활동 수
 * @returns {number} 레벨 (0-4)
 */
const getActivityLevel = (count) => {
  if (count === 0) return 0;
  if (count <= 2) return 1;
  if (count <= 4) return 2;
  if (count <= 7) return 3;
  return 4;
};

/**
 * 날짜 포맷팅 (YYYY-MM-DD → YYYY년 M월 D일)
 * @param {string} dateStr - YYYY-MM-DD 형식 날짜
 * @returns {string} 포맷된 날짜
 */
const formatDateDisplay = (dateStr) => {
  const [year, month, day] = dateStr.split('-');
  return `${year}년 ${parseInt(month)}월 ${parseInt(day)}일`;
};

/**
 * ActivityHeatmap 컴포넌트
 * GitHub 스타일의 활동 히트맵 (잔디밭)을 표시합니다
 *
 * @param {Object} props - 컴포넌트 props
 * @param {string} props.userId - 사용자 ID
 * @param {number} [props.year] - 표시할 연도 (기본값: 현재 연도)
 * @param {function} [props.onCellClick] - 셀 클릭 시 콜백
 */
const ActivityHeatmap = ({ userId, year: initialYear, onCellClick }) => {
  const currentYear = new Date().getFullYear();
  const [selectedYear, setSelectedYear] = useState(initialYear || currentYear);
  const [tooltip, setTooltip] = useState(null);

  const { heatmapData, loading, error } = useActivityHeatmap(userId, selectedYear);

  /**
   * 이전 연도로 이동
   */
  const handlePrevYear = useCallback(() => {
    setSelectedYear((prev) => prev - 1);
  }, []);

  /**
   * 다음 연도로 이동
   */
  const handleNextYear = useCallback(() => {
    if (selectedYear < currentYear) {
      setSelectedYear((prev) => prev + 1);
    }
  }, [selectedYear, currentYear]);

  /**
   * 셀 마우스 진입 핸들러
   */
  const handleCellMouseEnter = useCallback((event, dayData) => {
    const rect = event.target.getBoundingClientRect();
    setTooltip({
      date: dayData.date,
      count: dayData.count,
      x: rect.left + rect.width / 2,
      y: rect.top,
    });
  }, []);

  /**
   * 셀 마우스 이탈 핸들러
   */
  const handleCellMouseLeave = useCallback(() => {
    setTooltip(null);
  }, []);

  /**
   * 셀 클릭 핸들러
   */
  const handleCellClick = useCallback(
    (dayData) => {
      if (onCellClick) {
        onCellClick({
          date: dayData.date,
          count: dayData.count,
          activityIds: dayData.activityIds,
        });
      }
    },
    [onCellClick]
  );

  /**
   * 주 단위로 데이터 그룹화
   */
  const weeks = useMemo(() => {
    if (!heatmapData?.days) return [];

    const result = [];
    const days = heatmapData.days;

    // 첫 번째 날의 요일 확인 (연도 시작)
    const firstDay = new Date(days[0]?.date);
    const startDayOfWeek = firstDay.getDay();

    // 빈 셀로 첫 주 시작 채우기
    let currentWeek = Array(startDayOfWeek).fill(null);

    days.forEach((day) => {
      currentWeek.push(day);

      if (currentWeek.length === 7) {
        result.push(currentWeek);
        currentWeek = [];
      }
    });

    // 마지막 주 처리
    if (currentWeek.length > 0) {
      while (currentWeek.length < 7) {
        currentWeek.push(null);
      }
      result.push(currentWeek);
    }

    return result;
  }, [heatmapData]);

  /**
   * 월별 레이블 위치 계산
   */
  const monthLabels = useMemo(() => {
    if (!heatmapData?.days || weeks.length === 0) return [];

    const labels = [];
    let currentMonth = -1;

    weeks.forEach((week, weekIndex) => {
      const firstValidDay = week.find((d) => d !== null);
      if (firstValidDay) {
        const month = parseInt(firstValidDay.date.split('-')[1]) - 1;
        if (month !== currentMonth) {
          labels.push({ month, weekIndex });
          currentMonth = month;
        }
      }
    });

    return labels;
  }, [heatmapData, weeks]);

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
      <div className={styles.container} data-testid="heatmap-loading">
        <div className={styles.loadingContainer}>
          <div className={styles.loadingSpinner} />
          <p>히트맵 로딩 중...</p>
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

  // 데이터가 없는 경우
  if (!heatmapData) {
    return (
      <div className={styles.container}>
        <p className={styles.emptyMessage}>활동 데이터가 없습니다</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* 헤더: 연도 선택기 */}
      <div className={styles.header}>
        <div className={styles.yearSelector} data-testid="year-selector">
          <button
            type="button"
            className={styles.yearButton}
            onClick={handlePrevYear}
            aria-label="이전 연도"
            data-testid="year-prev-button"
          >
            &lt;
          </button>
          <span className={styles.yearText}>{selectedYear}</span>
          <button
            type="button"
            className={styles.yearButton}
            onClick={handleNextYear}
            disabled={selectedYear >= currentYear}
            aria-label="다음 연도"
            data-testid="year-next-button"
          >
            &gt;
          </button>
        </div>
        <div className={styles.totalActivities}>
          총 <strong>{heatmapData.totalActivities}</strong>개 활동
        </div>
      </div>

      {/* 히트맵 그리드 */}
      <div className={styles.heatmapWrapper}>
        {/* 요일 레이블 */}
        <div className={styles.dayLabels}>
          {DAY_LABELS.map((label, index) => (
            <span
              key={label}
              className={styles.dayLabel}
              style={{ visibility: index % 2 === 0 ? 'visible' : 'hidden' }}
            >
              {label}
            </span>
          ))}
        </div>

        {/* 그리드 컨테이너 */}
        <div className={styles.gridContainer}>
          {/* 월 레이블 */}
          <div className={styles.monthLabels}>
            {monthLabels.map(({ month, weekIndex }) => (
              <span
                key={`${month}-${weekIndex}`}
                className={styles.monthLabel}
                style={{ gridColumn: weekIndex + 1 }}
              >
                {MONTH_LABELS[month]}
              </span>
            ))}
          </div>

          {/* 히트맵 그리드 */}
          <div className={styles.grid} data-testid="heatmap-grid">
            {weeks.map((week, weekIndex) => (
              <div key={weekIndex} className={styles.week}>
                {week.map((day, dayIndex) => {
                  if (!day) {
                    return (
                      <div
                        key={`empty-${weekIndex}-${dayIndex}`}
                        className={`${styles.cell} ${styles.empty}`}
                      />
                    );
                  }

                  const level = getActivityLevel(day.count);

                  return (
                    <div
                      key={day.date}
                      data-date={day.date}
                      className={`${styles.cell} ${styles[`level${level}`]}`}
                      onMouseEnter={(e) => handleCellMouseEnter(e, day)}
                      onMouseLeave={handleCellMouseLeave}
                      onClick={() => handleCellClick(day)}
                      role="button"
                      tabIndex={0}
                      aria-label={`${formatDateDisplay(day.date)}: ${day.count}개 활동`}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          handleCellClick(day);
                        }
                      }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 레벨 범례 */}
      <div className={styles.legend} data-testid="level-legend">
        <span className={styles.legendLabel}>적음</span>
        <div className={`${styles.legendCell} ${styles.level0}`} />
        <div className={`${styles.legendCell} ${styles.level1}`} />
        <div className={`${styles.legendCell} ${styles.level2}`} />
        <div className={`${styles.legendCell} ${styles.level3}`} />
        <div className={`${styles.legendCell} ${styles.level4}`} />
        <span className={styles.legendLabel}>많음</span>
      </div>

      {/* 툴팁 */}
      {tooltip && (
        <div
          role="tooltip"
          className={styles.tooltip}
          style={{
            left: tooltip.x,
            top: tooltip.y - 8,
          }}
        >
          <strong>{tooltip.date}</strong>
          <span>{tooltip.count}개 활동</span>
        </div>
      )}
    </div>
  );
};

ActivityHeatmap.propTypes = {
  /** 사용자 ID */
  userId: PropTypes.string,
  /** 표시할 연도 (기본값: 현재 연도) */
  year: PropTypes.number,
  /** 셀 클릭 시 콜백 */
  onCellClick: PropTypes.func,
};

export default ActivityHeatmap;
