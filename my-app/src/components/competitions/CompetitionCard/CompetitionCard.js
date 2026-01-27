/**
 * 공모전 카드 컴포넌트
 *
 * 공모전 목록에서 사용되는 개별 카드 UI
 *
 * @module components/competitions/CompetitionCard
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './CompetitionCard.module.css';

/**
 * 분야 레이블 매핑
 */
const CATEGORY_LABELS = {
  visual_design: '시각디자인',
  industrial_design: '산업디자인',
  craft: '공예',
  fine_art: '회화',
};

/**
 * 상금 포맷팅
 * @param {number} prize - 상금
 * @returns {string} 포맷된 상금
 */
const formatPrize = (prize) => {
  if (prize >= 10000000) {
    return `${(prize / 10000000).toFixed(0)}천만원`;
  }
  if (prize >= 10000) {
    return `${(prize / 10000).toFixed(0)}만원`;
  }
  return `${prize.toLocaleString()}원`;
};

/**
 * D-Day 배지 스타일 결정
 * @param {number} dDay - D-Day
 * @param {string} status - 상태
 * @returns {string} CSS 클래스명
 */
const getDDayBadgeClass = (dDay, status) => {
  if (status === 'ended' || dDay < 0) return styles.badgeEnded;
  if (dDay <= 7) return styles.badgeUrgent;
  if (dDay <= 30) return styles.badgeSoon;
  return styles.badgeNormal;
};

/**
 * 공모전 카드 컴포넌트
 * @param {Object} props
 * @param {Object} props.competition - 공모전 데이터
 */
const CompetitionCard = ({ competition }) => {
  const navigate = useNavigate();

  const {
    id,
    title,
    thumbnail,
    category,
    organizer,
    prize,
    dDay,
    status,
    participantCount,
  } = competition;

  const handleClick = () => {
    navigate(`/competitions/${id}`);
  };

  const getDDayText = () => {
    if (status === 'ended' || dDay < 0) return '종료';
    if (dDay === 0) return 'D-Day';
    return `D-${dDay}`;
  };

  return (
    <article className={styles.card} onClick={handleClick}>
      {/* 썸네일 */}
      <div className={styles.thumbnailWrapper}>
        <img
          src={thumbnail}
          alt={title}
          className={styles.thumbnail}
          loading="lazy"
        />
        {/* D-Day 배지 */}
        <span className={`${styles.badge} ${getDDayBadgeClass(dDay, status)}`}>
          {getDDayText()}
        </span>
      </div>

      {/* 콘텐츠 */}
      <div className={styles.content}>
        {/* 분야 태그 */}
        <span className={styles.category}>
          {CATEGORY_LABELS[category] || category}
        </span>

        {/* 제목 */}
        <h3 className={styles.title}>{title}</h3>

        {/* 주최자 */}
        <p className={styles.organizer}>{organizer}</p>

        {/* 하단 정보 */}
        <div className={styles.footer}>
          <span className={styles.prize}>
            💰 {formatPrize(prize)}
          </span>
          <span className={styles.participants}>
            👥 {participantCount}명
          </span>
        </div>
      </div>
    </article>
  );
};

export default CompetitionCard;
