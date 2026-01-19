// AchievementList 컴포넌트
// SPEC-CRED-001: M3 공개 프로필 - 수상 내역 목록
// 사용자의 수상 기록을 목록으로 표시합니다

import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import styles from './AchievementList.module.css';

/**
 * 수상 등급별 아이콘
 */
const RANK_ICONS = {
  '대상': '\uD83C\uDFC6', // 트로피
  '금상': '\uD83E\uDD47', // 금메달
  '은상': '\uD83E\uDD48', // 은메달
  '동상': '\uD83E\uDD49', // 동메달
  '입선': '\uD83C\uDF96', // 메달
};

/**
 * 날짜 포맷 함수
 * @param {Object|Date} timestamp - Firestore Timestamp 또는 Date
 * @returns {string} 포맷된 날짜 문자열 (YYYY.MM.DD)
 */
const formatDate = (timestamp) => {
  if (!timestamp) return '';

  const date = timestamp.toDate ? timestamp.toDate() : new Date(timestamp);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');

  return `${year}.${month}.${day}`;
};

/**
 * AchievementList 컴포넌트
 * 사용자의 수상 기록을 목록으로 표시합니다
 *
 * @param {Object} props - 컴포넌트 props
 * @param {Array} props.awards - 수상 기록 배열
 * @param {boolean} [props.loading=false] - 로딩 상태
 * @param {boolean} [props.compact=false] - 축소 모드
 * @param {number} [props.maxItems] - 최대 표시 항목 수
 */
const AchievementList = ({
  awards = [],
  loading = false,
  compact = false,
  maxItems,
}) => {
  // 로딩 상태
  if (loading) {
    return (
      <div className={styles.container} data-testid="achievement-loading">
        <div className={styles.loadingContainer}>
          <div className={styles.loadingSpinner} />
          <p>수상 내역 로딩 중...</p>
        </div>
      </div>
    );
  }

  // 빈 상태
  if (!awards || awards.length === 0) {
    return (
      <div className={styles.container} data-testid="empty-achievements">
        <div className={styles.emptyContainer}>
          <span className={styles.emptyIcon}>{'\uD83C\uDFC5'}</span>
          <p className={styles.emptyMessage}>아직 수상 내역이 없습니다</p>
        </div>
      </div>
    );
  }

  // maxItems가 지정되면 해당 수만큼만 표시
  const displayAwards = maxItems ? awards.slice(0, maxItems) : awards;
  const hasMore = maxItems && awards.length > maxItems;

  // 리스트 클래스 조합
  const listClasses = [
    styles.list,
    compact ? styles.compact : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={styles.container}>
      <ul className={listClasses} data-testid="achievement-list">
        {displayAwards.map((award) => {
          const rankClass = `rank${award.rank}`;
          const icon = RANK_ICONS[award.rank] || '\uD83C\uDF96';

          return (
            <li
              key={award.id}
              className={`${styles.item} ${styles[rankClass]}`}
              data-testid={`achievement-item-${award.id}`}
            >
              {/* 수상 등급 아이콘 */}
              <span className={styles.rankIcon}>{icon}</span>

              {/* 수상 정보 */}
              <div className={styles.info}>
                {/* 공모전 제목 (링크) */}
                <Link
                  to={`/competitions/${award.competitionId}`}
                  className={styles.competitionTitle}
                  data-testid={`competition-link-${award.competitionId}`}
                >
                  {award.competitionTitle}
                </Link>

                {/* 수상 등급 및 날짜 */}
                <div className={styles.details}>
                  <span className={styles.rank}>{award.rank}</span>
                  <span className={styles.divider}>|</span>
                  <span className={styles.date}>{formatDate(award.awardedAt)}</span>
                </div>
              </div>
            </li>
          );
        })}
      </ul>

      {/* 더보기 버튼 */}
      {hasMore && (
        <button
          type="button"
          className={styles.showMoreButton}
          data-testid="show-more-button"
        >
          더보기 ({awards.length - maxItems}개 더)
        </button>
      )}
    </div>
  );
};

AchievementList.propTypes = {
  /** 수상 기록 배열 */
  awards: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.string.isRequired,
      competitionId: PropTypes.string.isRequired,
      competitionTitle: PropTypes.string.isRequired,
      rank: PropTypes.oneOf(['대상', '금상', '은상', '동상', '입선']).isRequired,
      awardedAt: PropTypes.oneOfType([
        PropTypes.object, // Firestore Timestamp
        PropTypes.instanceOf(Date),
      ]),
    })
  ),
  /** 로딩 상태 */
  loading: PropTypes.bool,
  /** 축소 모드 */
  compact: PropTypes.bool,
  /** 최대 표시 항목 수 */
  maxItems: PropTypes.number,
};

export default AchievementList;
