// PortfolioModal 컴포넌트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 상세 모달
// 전체 크기 이미지 보기, 네비게이션, 키보드 지원

import React, { useEffect, useCallback } from 'react';
import PropTypes from 'prop-types';
import styles from './PortfolioModal.module.css';

/**
 * PortfolioModal 컴포넌트
 * 포트폴리오 작품의 전체 크기 보기 모달
 *
 * @param {Object} props - 컴포넌트 props
 * @param {boolean} props.isOpen - 모달 열림 상태
 * @param {Object} props.portfolio - 포트폴리오 객체
 * @param {function} props.onClose - 닫기 콜백
 * @param {Array} [props.portfolios] - 전체 포트폴리오 목록 (네비게이션용)
 * @param {number} [props.currentIndex] - 현재 인덱스
 * @param {function} [props.onPrev] - 이전 콜백
 * @param {function} [props.onNext] - 다음 콜백
 */
const PortfolioModal = ({
  isOpen,
  portfolio,
  onClose,
  portfolios,
  currentIndex = 0,
  onPrev,
  onNext,
}) => {
  // 네비게이션 가능 여부
  const hasNavigation = portfolios && portfolios.length > 1;
  const canGoPrev = hasNavigation && currentIndex > 0;
  const canGoNext = hasNavigation && currentIndex < portfolios.length - 1;

  // 키보드 이벤트 핸들러
  const handleKeyDown = useCallback(
    (e) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'Escape':
          onClose();
          break;
        case 'ArrowLeft':
          if (canGoPrev && onPrev) {
            onPrev();
          }
          break;
        case 'ArrowRight':
          if (canGoNext && onNext) {
            onNext();
          }
          break;
        default:
          break;
      }
    },
    [isOpen, onClose, canGoPrev, canGoNext, onPrev, onNext]
  );

  // 키보드 이벤트 리스너
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  // 바디 스크롤 잠금
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // 오버레이 클릭 핸들러
  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  // 모달이 닫혀있거나 포트폴리오가 없으면 렌더링하지 않음
  if (!isOpen || !portfolio) {
    return null;
  }

  const {
    title,
    description,
    imageUrl,
    tags = [],
  } = portfolio;

  return (
    <div
      className={styles.overlay}
      onClick={handleOverlayClick}
      data-testid="modal-overlay"
    >
      <div
        className={styles.modal}
        data-testid="portfolio-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        {/* 닫기 버튼 */}
        <button
          type="button"
          className={styles.closeButton}
          onClick={onClose}
          data-testid="close-button"
          aria-label="모달 닫기"
        >
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>

        {/* 이전 버튼 */}
        {hasNavigation && (
          <button
            type="button"
            className={`${styles.navButton} ${styles.prevButton}`}
            onClick={onPrev}
            disabled={!canGoPrev}
            data-testid="prev-button"
            aria-label="이전 작품"
          >
            <svg
              width="32"
              height="32"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
        )}

        {/* 다음 버튼 */}
        {hasNavigation && (
          <button
            type="button"
            className={`${styles.navButton} ${styles.nextButton}`}
            onClick={onNext}
            disabled={!canGoNext}
            data-testid="next-button"
            aria-label="다음 작품"
          >
            <svg
              width="32"
              height="32"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="9 18 15 12 9 6" />
            </svg>
          </button>
        )}

        {/* 콘텐츠 영역 */}
        <div
          className={styles.content}
          data-testid="modal-content"
          onClick={(e) => e.stopPropagation()}
        >
          {/* 이미지 영역 */}
          <div className={styles.imageContainer}>
            <img
              src={imageUrl}
              alt={title}
              className={styles.image}
              data-testid="modal-image"
            />
          </div>

          {/* 정보 영역 */}
          <div className={styles.info}>
            {/* 제목 */}
            <h2 id="modal-title" className={styles.title}>
              {title}
            </h2>

            {/* 설명 */}
            {description && (
              <p className={styles.description} data-testid="modal-description">
                {description}
              </p>
            )}

            {/* 태그 */}
            {tags.length > 0 && (
              <div className={styles.tags}>
                {tags.map((tag, index) => (
                  <span key={index} className={styles.tag}>
                    {tag}
                  </span>
                ))}
              </div>
            )}

            {/* 인덱스 표시 */}
            {hasNavigation && (
              <div className={styles.indexIndicator}>
                {currentIndex + 1} / {portfolios.length}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

PortfolioModal.propTypes = {
  /** 모달 열림 상태 */
  isOpen: PropTypes.bool.isRequired,
  /** 포트폴리오 객체 */
  portfolio: PropTypes.shape({
    id: PropTypes.string,
    title: PropTypes.string.isRequired,
    description: PropTypes.string,
    imageUrl: PropTypes.string.isRequired,
    tags: PropTypes.arrayOf(PropTypes.string),
  }),
  /** 닫기 콜백 */
  onClose: PropTypes.func.isRequired,
  /** 전체 포트폴리오 목록 */
  portfolios: PropTypes.array,
  /** 현재 인덱스 */
  currentIndex: PropTypes.number,
  /** 이전 콜백 */
  onPrev: PropTypes.func,
  /** 다음 콜백 */
  onNext: PropTypes.func,
};

export default PortfolioModal;
