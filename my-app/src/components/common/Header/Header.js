// Header 컴포넌트
// SPEC-UI-001: 공통 UI 컴포넌트 - Header
// 고정 위치 헤더, 스크롤 감지 효과, 네비게이션 지원

import React, { useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import styles from './Header.module.css';

/**
 * Header 컴포넌트
 * @param {Object} props - 컴포넌트 props
 * @param {React.ReactNode} props.logo - 로고 요소
 * @param {Array<{label: string, href: string}>} props.navItems - 네비게이션 아이템 배열
 * @param {{label: string, onClick: Function}} props.ctaButton - CTA 버튼 설정
 * @param {string} props.className - 추가 CSS 클래스
 */
const Header = ({
  logo,
  navItems = [],
  ctaButton,
  className = '',
  ...props
}) => {
  // 스크롤 상태 관리
  const [isScrolled, setIsScrolled] = useState(false);

  // 스크롤 이벤트 핸들러
  useEffect(() => {
    const handleScroll = () => {
      const scrolled = window.scrollY > 50;
      setIsScrolled(scrolled);
    };

    // 초기 스크롤 상태 확인
    handleScroll();

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // 클래스 이름 조합
  const headerClasses = [
    styles.header,
    isScrolled && styles.scrolled,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <header className={headerClasses} {...props}>
      <div className={styles.container}>
        {/* 로고 영역 */}
        <div className={styles.logoContainer}>
          {logo}
        </div>

        {/* 네비게이션 영역 */}
        <nav className={styles.nav} role="navigation">
          <ul className={styles.navList}>
            {navItems.map((item, index) => (
              <li key={index} className={styles.navItem}>
                <a href={item.href} className={styles.navLink}>
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        {/* CTA 버튼 영역 */}
        {ctaButton && (
          <div className={styles.ctaContainer}>
            <button
              className={styles.ctaButton}
              onClick={ctaButton.onClick}
              type="button"
            >
              {ctaButton.label}
            </button>
          </div>
        )}
      </div>
    </header>
  );
};

Header.propTypes = {
  /** 로고 요소 */
  logo: PropTypes.node,
  /** 네비게이션 아이템 배열 */
  navItems: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.string.isRequired,
      href: PropTypes.string.isRequired,
    })
  ),
  /** CTA 버튼 설정 */
  ctaButton: PropTypes.shape({
    label: PropTypes.string.isRequired,
    onClick: PropTypes.func.isRequired,
  }),
  /** 추가 CSS 클래스 */
  className: PropTypes.string,
};

export default Header;
