// CompetitionDetail 컴포넌트
// SPEC-COMP-001: 공모전 상세 페이지

import React from 'react';
import { useParams, Link } from 'react-router-dom';
import styles from './CompetitionDetail.module.css';

/**
 * 공모전 상세 페이지
 * - 공모전 헤더 정보
 * - 탭 메뉴 (상세정보, 출품작, 결과)
 * - 출품 버튼
 */
const CompetitionDetail = () => {
  const { id } = useParams();

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <Link to="/competitions" className={styles.backLink}>
          ← 목록으로
        </Link>
        <h1 className={styles.title}>공모전 상세</h1>
      </div>

      <div className={styles.content}>
        <div className={styles.placeholder}>
          <span className={styles.icon}>📋</span>
          <h2>공모전 #{id}</h2>
          <p>공모전 상세 정보가 여기에 표시됩니다.</p>
          <p className={styles.devNote}>개발 진행 중 - SPEC-COMP-001</p>

          <Link to={`/competitions/${id}/submit`} className={styles.submitButton}>
            출품하기
          </Link>
        </div>
      </div>
    </div>
  );
};

export default CompetitionDetail;
