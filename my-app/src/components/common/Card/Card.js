// Card 컴포넌트
// SPEC-UI-001: 공통 UI 컴포넌트 - Card
// 기본 카드 및 프레임 스타일을 지원하는 컨테이너 컴포넌트

import React from 'react';
import PropTypes from 'prop-types';
import styles from './Card.module.css';

/**
 * Card 컴포넌트
 * @param {Object} props - 컴포넌트 props
 * @param {'basic' | 'frame'} props.variant - 카드 스타일 variant
 * @param {'sm' | 'md' | 'lg'} props.padding - 내부 패딩 사이즈
 * @param {React.ReactNode} props.children - 카드 내용
 * @param {string} props.className - 추가 CSS 클래스
 */
const Card = ({
  variant = 'basic',
  padding = 'md',
  children,
  className = '',
  ...props
}) => {
  // 패딩 클래스명 생성 (camelCase로 변환)
  const paddingClass = `padding${padding.charAt(0).toUpperCase()}${padding.slice(1)}`;

  // 클래스 이름 조합
  const cardClasses = [
    styles.card,
    styles[variant],
    styles[paddingClass],
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <div className={cardClasses} {...props}>
      {children}
    </div>
  );
};

Card.propTypes = {
  /** 카드 스타일 variant */
  variant: PropTypes.oneOf(['basic', 'frame']),
  /** 내부 패딩 사이즈 */
  padding: PropTypes.oneOf(['sm', 'md', 'lg']),
  /** 카드 내용 */
  children: PropTypes.node,
  /** 추가 CSS 클래스 */
  className: PropTypes.string,
};

export default Card;
