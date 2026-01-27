// TierBadge 컴포넌트
// SPEC-CRED-001: M3 공개 프로필 - 티어 배지
// 사용자 티어를 시각적 배지로 표시합니다

import React from 'react';
import PropTypes from 'prop-types';
import styles from './TierBadge.module.css';

/**
 * 유효한 티어 목록
 */
const VALID_TIERS = ['S', 'A', 'B', 'C', 'Unranked'];

/**
 * 티어별 설명
 */
const TIER_DESCRIPTIONS = {
  S: '최상위 티어 - 뛰어난 실력과 다수의 수상 경력',
  A: '상위 티어 - 우수한 실력과 수상 경력',
  B: '중상위 티어 - 발전 중인 실력',
  C: '일반 티어 - 시작 단계',
  Unranked: '등급 없음 - 아직 평가되지 않음',
};

/**
 * TierBadge 컴포넌트
 * 사용자 티어를 색상 코딩된 배지로 표시합니다
 *
 * @param {Object} props - 컴포넌트 props
 * @param {string} props.tier - 티어 (S/A/B/C/Unranked)
 * @param {string} [props.size='medium'] - 크기 (small/medium/large)
 * @param {boolean} [props.showLabel=true] - 레이블 표시 여부
 */
const TierBadge = ({
  tier,
  size = 'medium',
  showLabel = true,
}) => {
  // 유효한 티어인지 확인, 아니면 Unranked로 처리
  const validTier = VALID_TIERS.includes(tier) ? tier : 'Unranked';

  // 표시할 레이블 (Unranked는 "-"로 표시)
  const displayLabel = validTier === 'Unranked' ? '-' : validTier;

  // 티어별 클래스 이름
  const tierClassName = `tier${validTier}`;

  // 클래스 조합
  const badgeClasses = [
    styles.badge,
    styles[tierClassName],
    styles[size],
  ].filter(Boolean).join(' ');

  // 접근성용 레이블
  const ariaLabel = `${validTier} 티어: ${TIER_DESCRIPTIONS[validTier]}`;

  return (
    <span
      className={badgeClasses}
      data-testid="tier-badge"
      aria-label={ariaLabel}
      title={TIER_DESCRIPTIONS[validTier]}
    >
      {showLabel && (
        <span className={styles.label}>{displayLabel}</span>
      )}
    </span>
  );
};

TierBadge.propTypes = {
  /** 사용자 티어 (S/A/B/C/Unranked) */
  tier: PropTypes.oneOf(['S', 'A', 'B', 'C', 'Unranked', '']),
  /** 배지 크기 */
  size: PropTypes.oneOf(['small', 'medium', 'large']),
  /** 레이블 표시 여부 */
  showLabel: PropTypes.bool,
};

export default TierBadge;
