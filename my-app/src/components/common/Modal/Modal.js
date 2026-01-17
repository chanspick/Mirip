// Modal 컴포넌트
// SPEC-UI-001: 공통 UI 컴포넌트 - Modal
// ESC 키 닫기, 오버레이 클릭 닫기, 바디 스크롤 잠금을 지원하는 모달

import React, { useEffect, useCallback, useId } from 'react';
import PropTypes from 'prop-types';
import styles from './Modal.module.css';

/**
 * Modal 컴포넌트
 * @param {Object} props - 컴포넌트 props
 * @param {boolean} props.isOpen - 모달 열림 상태
 * @param {Function} props.onClose - 모달 닫기 핸들러
 * @param {string} props.title - 모달 제목
 * @param {React.ReactNode} props.children - 모달 내용
 * @param {string} props.className - 추가 CSS 클래스
 */
const Modal = ({
  isOpen,
  onClose,
  title,
  children,
  className = '',
  ...props
}) => {
  // 고유 ID 생성 (접근성용)
  const titleId = useId();

  // ESC 키 핸들러
  const handleKeyDown = useCallback(
    (event) => {
      if (event.key === 'Escape' && isOpen) {
        onClose();
      }
    },
    [isOpen, onClose]
  );

  // ESC 키 이벤트 리스너
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
  const handleOverlayClick = (event) => {
    // 오버레이 자체를 클릭했을 때만 닫기
    if (event.target === event.currentTarget) {
      onClose();
    }
  };

  // 모달이 닫혀있으면 렌더링하지 않음
  if (!isOpen) {
    return null;
  }

  // 클래스 이름 조합
  const modalClasses = [
    styles.modal,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div
      className={styles.overlay}
      onClick={handleOverlayClick}
      data-testid="modal-overlay"
    >
      <div
        className={modalClasses}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? titleId : undefined}
        {...props}
      >
        {/* 모달 헤더 */}
        <div className={styles.header}>
          {title && (
            <h2 id={titleId} className={styles.title}>
              {title}
            </h2>
          )}
          <button
            className={styles.closeButton}
            onClick={onClose}
            type="button"
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
        </div>

        {/* 모달 본문 */}
        <div className={styles.content}>
          {children}
        </div>
      </div>
    </div>
  );
};

Modal.propTypes = {
  /** 모달 열림 상태 */
  isOpen: PropTypes.bool.isRequired,
  /** 모달 닫기 핸들러 */
  onClose: PropTypes.func.isRequired,
  /** 모달 제목 */
  title: PropTypes.string,
  /** 모달 내용 */
  children: PropTypes.node,
  /** 추가 CSS 클래스 */
  className: PropTypes.string,
};

export default Modal;
