// Loading 컴포넌트
// SPEC-UI-001: 공통 UI 컴포넌트 - Loading
// 다양한 사이즈와 전체 화면 모드를 지원하는 로딩 스피너

import React from 'react';
import PropTypes from 'prop-types';
import styles from './Loading.module.css';

/**
 * Loading 컴포넌트
 * @param {Object} props - 컴포넌트 props
 * @param {'sm' | 'md' | 'lg'} props.size - 스피너 사이즈
 * @param {boolean} props.fullScreen - 전체 화면 오버레이 모드
 * @param {string} props.color - 스피너 색상 (CSS 색상값)
 * @param {string} props.className - 추가 CSS 클래스
 */
const Loading = ({
  size = 'md',
  fullScreen = false,
  color,
  className = '',
  ...props
}) => {
  // 클래스 이름 조합
  const loadingClasses = [
    styles.loading,
    styles[size],
    fullScreen && styles.fullScreen,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  // 커스텀 색상 스타일
  const spinnerStyle = color
    ? { borderTopColor: color }
    : undefined;

  return (
    <div
      className={loadingClasses}
      role="status"
      aria-label="로딩 중"
      {...props}
    >
      <div className={styles.spinner} style={spinnerStyle} />
    </div>
  );
};

Loading.propTypes = {
  /** 스피너 사이즈 */
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  /** 전체 화면 오버레이 모드 */
  fullScreen: PropTypes.bool,
  /** 스피너 색상 (CSS 색상값) */
  color: PropTypes.string,
  /** 추가 CSS 클래스 */
  className: PropTypes.string,
};

export default Loading;
