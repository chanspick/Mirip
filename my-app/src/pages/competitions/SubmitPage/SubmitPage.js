// SubmitPage 컴포넌트
// SPEC-COMP-001: 공모전 출품 페이지

import React from 'react';
import { useParams, Link } from 'react-router-dom';
import styles from './SubmitPage.module.css';

/**
 * 공모전 출품 페이지
 * - 이미지 업로드
 * - 작품 정보 입력
 * - 제출 기능
 */
const SubmitPage = () => {
  const { id } = useParams();

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <Link to={`/competitions/${id}`} className={styles.backLink}>
          ← 공모전으로
        </Link>
        <h1 className={styles.title}>작품 출품</h1>
      </div>

      <div className={styles.content}>
        <div className={styles.placeholder}>
          <span className={styles.icon}>🎨</span>
          <h2>출품 페이지</h2>
          <p>공모전 #{id}에 작품을 출품합니다.</p>
          <p className={styles.devNote}>개발 진행 중 - SPEC-COMP-001</p>

          <div className={styles.uploadArea}>
            <span className={styles.uploadIcon}>📤</span>
            <p>이미지를 드래그하거나 클릭하여 업로드</p>
            <span className={styles.uploadHint}>JPG, PNG, WebP (최대 10MB)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SubmitPage;
