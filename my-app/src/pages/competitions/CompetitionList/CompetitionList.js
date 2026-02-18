// CompetitionList 컴포넌트
// SPEC-COMP-001: 공모전 목록 페이지

import React from 'react';
import styles from './CompetitionList.module.css';

/**
 * 공모전 목록 페이지
 * - 공모전 카드 그리드
 * - 필터 및 정렬
 * - 페이지네이션
 */
const CompetitionList = () => {
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>공모전</h1>
        <p className={styles.subtitle}>다양한 분야의 공모전에 참여하고 실력을 뽐내보세요</p>
      </div>

      <div className={styles.content}>
        <div className={styles.placeholder}>
          <span className={styles.icon}>🏆</span>
          <h2>공모전 목록</h2>
          <p>곧 다양한 공모전이 등록될 예정입니다.</p>
          <p className={styles.devNote}>개발 진행 중 - SPEC-COMP-001</p>
        </div>
      </div>
    </div>
  );
};

export default CompetitionList;
