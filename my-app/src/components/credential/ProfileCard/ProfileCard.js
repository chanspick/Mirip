// ProfileCard 컴포넌트
// SPEC-CRED-001: M3 공개 프로필 - 프로필 카드
// 사용자 프로필의 축소된 카드 뷰를 표시합니다

import React from 'react';
import PropTypes from 'prop-types';
import { Link } from 'react-router-dom';
import TierBadge from '../TierBadge';
import styles from './ProfileCard.module.css';

// Bio 최대 표시 길이 (truncate 기준)
const BIO_TRUNCATE_LENGTH = 80;

/**
 * 이름에서 이니셜 추출
 * @param {string} name - 이름
 * @returns {string} 이니셜 (최대 2자)
 */
const getInitials = (name) => {
  if (!name) return '?';
  const words = name.split(' ');
  if (words.length >= 2) {
    return (words[0][0] + words[1][0]).toUpperCase();
  }
  return name.slice(0, 2).toUpperCase();
};

/**
 * ProfileCard 컴포넌트
 * 사용자 프로필의 축소된 카드 뷰를 표시합니다
 *
 * @param {Object} props - 컴포넌트 props
 * @param {Object} props.profile - UserProfile 객체
 * @param {boolean} [props.showStats=true] - 활동 통계 표시 여부
 * @param {boolean} [props.compact=false] - 축소 모드
 * @param {function} [props.onClick] - 클릭 핸들러
 */
const ProfileCard = ({
  profile,
  showStats = true,
  compact = false,
  onClick,
}) => {
  // profile이 없으면 렌더링하지 않음
  if (!profile) {
    return null;
  }

  const {
    username,
    displayName,
    profileImageUrl,
    bio,
    tier,
    totalActivities,
    currentStreak,
    longestStreak,
  } = profile;

  // Bio truncate 처리
  const isBioTruncated = bio && bio.length > BIO_TRUNCATE_LENGTH;
  const displayBio = isBioTruncated
    ? `${bio.substring(0, BIO_TRUNCATE_LENGTH)}...`
    : bio;

  // 카드 클래스 조합
  const cardClasses = [
    styles.card,
    compact ? styles.compact : '',
    onClick ? styles.clickable : '',
  ].filter(Boolean).join(' ');

  // 클릭 핸들러
  const handleClick = () => {
    if (onClick) {
      onClick();
    }
  };

  // 키보드 접근성
  const handleKeyDown = (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && onClick) {
      e.preventDefault();
      onClick();
    }
  };

  return (
    <div
      className={cardClasses}
      data-testid="profile-card"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {/* 아바타 섹션 */}
      <div className={styles.avatarSection} data-testid="profile-avatar">
        {profileImageUrl ? (
          <img
            src={profileImageUrl}
            alt={displayName}
            className={styles.avatar}
          />
        ) : (
          <div
            className={styles.avatarPlaceholder}
            data-testid="avatar-placeholder"
          >
            {getInitials(displayName)}
          </div>
        )}
      </div>

      {/* 정보 섹션 */}
      <div className={styles.infoSection}>
        {/* 이름과 티어 배지 */}
        <div className={styles.nameRow}>
          <h3 className={styles.displayName}>{displayName}</h3>
          <TierBadge tier={tier} size={compact ? 'small' : 'medium'} />
        </div>

        {/* 사용자명 */}
        <Link
          to={`/profile/${username}`}
          className={styles.username}
          data-testid="profile-link"
        >
          @{username}
        </Link>

        {/* Bio (compact 모드가 아닐 때만) */}
        {!compact && bio && (
          <p
            className={`${styles.bio} ${isBioTruncated ? styles.truncated : ''}`}
            data-testid="profile-bio"
          >
            {displayBio}
          </p>
        )}

        {/* 활동 통계 */}
        {showStats && !compact && (
          <div className={styles.stats} data-testid="profile-stats">
            <div className={styles.statItem}>
              <span className={styles.statValue}>{totalActivities || 0}</span>
              <span className={styles.statLabel}>활동</span>
            </div>
            <div className={styles.statItem}>
              <span className={styles.statValue}>{currentStreak || 0}</span>
              <span className={styles.statLabel}>연속</span>
            </div>
            <div className={styles.statItem}>
              <span className={styles.statValue}>{longestStreak || 0}</span>
              <span className={styles.statLabel}>최장</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

ProfileCard.propTypes = {
  /** UserProfile 객체 */
  profile: PropTypes.shape({
    uid: PropTypes.string,
    username: PropTypes.string.isRequired,
    displayName: PropTypes.string.isRequired,
    profileImageUrl: PropTypes.string,
    bio: PropTypes.string,
    tier: PropTypes.string,
    totalActivities: PropTypes.number,
    currentStreak: PropTypes.number,
    longestStreak: PropTypes.number,
  }),
  /** 활동 통계 표시 여부 */
  showStats: PropTypes.bool,
  /** 축소 모드 */
  compact: PropTypes.bool,
  /** 클릭 핸들러 */
  onClick: PropTypes.func,
};

export default ProfileCard;
