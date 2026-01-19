// Header 컴포넌트
// SPEC-UI-001: 공통 UI 컴포넌트 - Header
// SPEC-CRED-001 M5: 프로필 링크 추가
// 고정 위치 헤더, 스크롤 감지 효과, 네비게이션 지원

import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import PropTypes from 'prop-types';
import styles from './Header.module.css';

// 기본 프로필 이미지 (사용자 아이콘)
const DEFAULT_AVATAR = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiM2NjY2NjYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMjAgMjF2LTJhNCA0IDAgMCAwLTQtNEg4YTQgNCAwIDAgMC00IDR2MiI+PC9wYXRoPjxjaXJjbGUgY3g9IjEyIiBjeT0iNyIgcj0iNCI+PC9jaXJjbGU+PC9zdmc+';

/**
 * Header 컴포넌트
 * @param {Object} props - 컴포넌트 props
 * @param {React.ReactNode} props.logo - 로고 요소
 * @param {Array<{label: string, href: string}>} props.navItems - 네비게이션 아이템 배열
 * @param {{label: string, onClick: Function}} props.ctaButton - CTA 버튼 설정
 * @param {string} props.className - 추가 CSS 클래스
 * @param {Object} props.user - 로그인 사용자 정보 (SPEC-CRED-001 M5)
 * @param {string} props.user.photoURL - 프로필 이미지 URL
 * @param {string} props.user.displayName - 표시 이름
 */
const Header = ({
  logo,
  navItems = [],
  ctaButton,
  className = '',
  user,
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
                {item.href.startsWith('#') ? (
                  <a href={item.href} className={styles.navLink}>
                    {item.label}
                  </a>
                ) : (
                  <Link to={item.href} className={styles.navLink}>
                    {item.label}
                  </Link>
                )}
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

        {/* 사용자 프로필 영역 (SPEC-CRED-001 M5) */}
        {user && (
          <div className={styles.profileContainer}>
            <Link to="/profile" className={styles.profileLink}>
              <img
                src={user.photoURL || DEFAULT_AVATAR}
                alt="프로필"
                className={styles.profileAvatar}
              />
              <span className={styles.profileName}>마이페이지</span>
            </Link>
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
  /** 로그인 사용자 정보 (SPEC-CRED-001 M5) */
  user: PropTypes.shape({
    photoURL: PropTypes.string,
    displayName: PropTypes.string,
  }),
};

export default Header;
