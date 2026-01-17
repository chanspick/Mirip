// Button 컴포넌트
// SPEC-UI-001: 공통 UI 컴포넌트 - Button
// 다양한 스타일 variant와 사이즈를 지원하는 재사용 가능한 버튼 컴포넌트

import React from 'react';
import PropTypes from 'prop-types';
import styles from './Button.module.css';

/**
 * Button 컴포넌트
 * @param {Object} props - 컴포넌트 props
 * @param {'primary' | 'cta' | 'outline'} props.variant - 버튼 스타일 variant
 * @param {'sm' | 'md' | 'lg'} props.size - 버튼 사이즈
 * @param {boolean} props.disabled - 비활성화 상태
 * @param {boolean} props.fullWidth - 전체 너비 사용 여부
 * @param {Function} props.onClick - 클릭 이벤트 핸들러
 * @param {React.ReactNode} props.children - 버튼 내용
 * @param {string} props.className - 추가 CSS 클래스
 */
const Button = ({
  variant = 'primary',
  size = 'md',
  disabled = false,
  fullWidth = false,
  onClick,
  children,
  className = '',
  ...props
}) => {
  // 클래스 이름 조합
  const buttonClasses = [
    styles.button,
    styles[variant],
    styles[size],
    fullWidth && styles.fullWidth,
    disabled && styles.disabled,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  // 키보드 이벤트 핸들러 (접근성)
  const handleKeyDown = (event) => {
    if (disabled) return;

    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onClick && onClick(event);
    }
  };

  return (
    <button
      className={buttonClasses}
      onClick={disabled ? undefined : onClick}
      onKeyDown={handleKeyDown}
      disabled={disabled}
      type="button"
      {...props}
    >
      {children}
    </button>
  );
};

Button.propTypes = {
  /** 버튼 스타일 variant */
  variant: PropTypes.oneOf(['primary', 'cta', 'outline']),
  /** 버튼 사이즈 */
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  /** 비활성화 상태 */
  disabled: PropTypes.bool,
  /** 전체 너비 사용 여부 */
  fullWidth: PropTypes.bool,
  /** 클릭 이벤트 핸들러 */
  onClick: PropTypes.func,
  /** 버튼 내용 */
  children: PropTypes.node.isRequired,
  /** 추가 CSS 클래스 */
  className: PropTypes.string,
};

export default Button;
