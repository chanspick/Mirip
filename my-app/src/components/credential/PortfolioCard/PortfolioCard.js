// PortfolioCard 컴포넌트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 카드
// 포트폴리오 작품의 카드 뷰를 표시합니다

import React from 'react';
import PropTypes from 'prop-types';
import styles from './PortfolioCard.module.css';

// 설명 최대 표시 길이 (truncate 기준)
const DESCRIPTION_TRUNCATE_LENGTH = 100;

/**
 * PortfolioCard 컴포넌트
 * 포트폴리오 작품의 카드 뷰를 표시합니다
 *
 * @param {Object} props - 컴포넌트 props
 * @param {Object} props.portfolio - 포트폴리오 객체
 * @param {boolean} [props.isOwner=false] - 소유자 여부 (편집/삭제 버튼 표시)
 * @param {function} [props.onEdit] - 편집 콜백
 * @param {function} [props.onDelete] - 삭제 콜백
 * @param {function} [props.onClick] - 클릭 콜백 (전체 크기 보기)
 */
const PortfolioCard = ({
  portfolio,
  isOwner = false,
  onEdit,
  onDelete,
  onClick,
}) => {
  // portfolio가 없으면 렌더링하지 않음
  if (!portfolio) {
    return null;
  }

  const {
    title,
    description,
    imageUrl,
    thumbnailUrl,
    tags = [],
    isPublic,
  } = portfolio;

  // 표시할 이미지 URL (썸네일 우선)
  const displayImageUrl = thumbnailUrl || imageUrl;

  // 설명 truncate 처리
  const isDescTruncated = description && description.length > DESCRIPTION_TRUNCATE_LENGTH;
  const displayDescription = isDescTruncated
    ? `${description.substring(0, DESCRIPTION_TRUNCATE_LENGTH)}...`
    : description;

  // 카드 클래스 조합
  const cardClasses = [
    styles.card,
    onClick ? styles.clickable : '',
  ].filter(Boolean).join(' ');

  // 카드 클릭 핸들러
  const handleClick = () => {
    if (onClick) {
      onClick(portfolio);
    }
  };

  // 키보드 접근성
  const handleKeyDown = (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && onClick) {
      e.preventDefault();
      onClick(portfolio);
    }
  };

  // 편집 버튼 클릭 핸들러
  const handleEditClick = (e) => {
    e.stopPropagation();
    if (onEdit) {
      onEdit(portfolio);
    }
  };

  // 삭제 버튼 클릭 핸들러
  const handleDeleteClick = (e) => {
    e.stopPropagation();
    if (onDelete) {
      onDelete(portfolio);
    }
  };

  return (
    <div
      className={cardClasses}
      data-testid="portfolio-card"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {/* 이미지 컨테이너 */}
      <div className={styles.imageContainer}>
        <img
          src={displayImageUrl}
          alt={title}
          className={styles.thumbnail}
          data-testid="portfolio-thumbnail"
          loading="lazy"
        />

        {/* 비공개 표시 (소유자에게만 표시) */}
        {isOwner && !isPublic && (
          <div className={styles.privateIndicator} data-testid="private-indicator">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
              <path d="M7 11V7a5 5 0 0 1 10 0v4" />
            </svg>
            <span>비공개</span>
          </div>
        )}

        {/* 소유자 액션 버튼 */}
        {isOwner && (
          <div className={styles.actionButtons}>
            <button
              type="button"
              className={styles.actionButton}
              onClick={handleEditClick}
              data-testid="edit-button"
              aria-label="편집"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
              </svg>
            </button>
            <button
              type="button"
              className={`${styles.actionButton} ${styles.deleteButton}`}
              onClick={handleDeleteClick}
              data-testid="delete-button"
              aria-label="삭제"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* 컨텐츠 영역 */}
      <div className={styles.content}>
        {/* 제목 */}
        <h3 className={styles.title}>{title}</h3>

        {/* 설명 */}
        {description && (
          <p
            className={`${styles.description} ${isDescTruncated ? styles.truncated : ''}`}
            data-testid="portfolio-description"
          >
            {displayDescription}
          </p>
        )}

        {/* 태그 */}
        {tags.length > 0 && (
          <div className={styles.tags} data-testid="portfolio-tags">
            {tags.map((tag, index) => (
              <span key={index} className={styles.tag}>
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

PortfolioCard.propTypes = {
  /** 포트폴리오 객체 */
  portfolio: PropTypes.shape({
    id: PropTypes.string,
    userId: PropTypes.string,
    title: PropTypes.string.isRequired,
    description: PropTypes.string,
    imageUrl: PropTypes.string.isRequired,
    thumbnailUrl: PropTypes.string,
    tags: PropTypes.arrayOf(PropTypes.string),
    isPublic: PropTypes.bool,
    createdAt: PropTypes.object,
    updatedAt: PropTypes.object,
  }),
  /** 소유자 여부 */
  isOwner: PropTypes.bool,
  /** 편집 콜백 */
  onEdit: PropTypes.func,
  /** 삭제 콜백 */
  onDelete: PropTypes.func,
  /** 클릭 콜백 */
  onClick: PropTypes.func,
};

export default PortfolioCard;
