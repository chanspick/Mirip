/**
 * 공모전 필터/정렬 컴포넌트
 *
 * @module components/competitions/CompetitionFilter
 */

import React from 'react';
import styles from './CompetitionFilter.module.css';

/**
 * 분야 옵션
 */
const CATEGORY_OPTIONS = [
  { value: 'all', label: '전체 분야' },
  { value: 'visual_design', label: '시각디자인' },
  { value: 'industrial_design', label: '산업디자인' },
  { value: 'craft', label: '공예' },
  { value: 'fine_art', label: '회화' },
];

/**
 * 상태 옵션
 */
const STATUS_OPTIONS = [
  { value: 'all', label: '전체 상태' },
  { value: 'active', label: '진행중' },
  { value: 'ending_soon', label: '마감임박' },
  { value: 'ended', label: '종료' },
];

/**
 * 정렬 옵션
 */
const SORT_OPTIONS = [
  { value: 'endDate', label: '마감순' },
  { value: 'prize', label: '상금순' },
  { value: 'popularity', label: '인기순' },
  { value: 'createdAt', label: '최신순' },
];

/**
 * 공모전 필터 컴포넌트
 * @param {Object} props
 * @param {Object} props.filters - 현재 필터 상태
 * @param {Function} props.onFilterChange - 필터 변경 핸들러
 */
const CompetitionFilter = ({ filters, onFilterChange }) => {
  const handleChange = (key) => (e) => {
    onFilterChange({ [key]: e.target.value });
  };

  return (
    <div className={styles.filterContainer}>
      <div className={styles.filterGroup}>
        {/* 분야 필터 */}
        <select
          className={styles.select}
          value={filters.category}
          onChange={handleChange('category')}
          aria-label="분야 필터"
        >
          {CATEGORY_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        {/* 상태 필터 */}
        <select
          className={styles.select}
          value={filters.status}
          onChange={handleChange('status')}
          aria-label="상태 필터"
        >
          {STATUS_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      {/* 정렬 */}
      <div className={styles.sortGroup}>
        <label className={styles.sortLabel}>정렬:</label>
        <select
          className={styles.select}
          value={filters.sortBy}
          onChange={handleChange('sortBy')}
          aria-label="정렬 옵션"
        >
          {SORT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default CompetitionFilter;
