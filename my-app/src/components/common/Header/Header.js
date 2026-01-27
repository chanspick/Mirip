// Header 컴포넌트
// SPEC-UI-001: 공통 UI 컴포넌트 - Header
// SPEC-CRED-001 M5: 프로필 링크 추가
// 고정 위치 헤더, 스크롤 감지 효과, 네비게이션 지원
// 로그인/로그아웃 버튼 지원

import React, { useState, useEffect, useRef } from 'react';
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
 * @param {Function} props.onLoginClick - 로그인 버튼 클릭 핸들러
 * @param {Function} props.onLogout - 로그아웃 핸들러
 */
const Header = ({
  logo,
  navItems = [],
  ctaButton,
  className = '',
  user,
  onLoginClick,
  onLogout,
  ...props
}) => {
  // 스크롤 상태 관리
  const [isScrolled, setIsScrolled] = useState(false);
  // 프로필 드롭다운 상태
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

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

  // 드롭다운 외부 클릭 감지
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsDropdownOpen(false);
      }
    };

    if (isDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);

  // 로그아웃 핸들러
  const handleLogout = () => {
    setIsDropdownOpen(false);
    if (onLogout) {
      onLogout();
    }
  };

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

        {/* 로그인/프로필 영역 */}
        {user ? (
          // 로그인 상태: 프로필 드롭다운
          <div className={styles.profileContainer} ref={dropdownRef}>
            <button
              type="button"
              className={styles.profileButton}
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              aria-expanded={isDropdownOpen}
              aria-haspopup="true"
            >
              <img
                src={user.photoURL || DEFAULT_AVATAR}
                alt="프로필"
                className={styles.profileAvatar}
              />
              <span className={styles.profileName}>
                {user.displayName || '마이페이지'}
              </span>
              <svg
                className={`${styles.dropdownArrow} ${isDropdownOpen ? styles.open : ''}`}
                width="12"
                height="12"
                viewBox="0 0 12 12"
                fill="none"
              >
                <path
                  d="M3 4.5L6 7.5L9 4.5"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>

            {/* 드롭다운 메뉴 */}
            {isDropdownOpen && (
              <div className={styles.dropdown}>
                <Link
                  to="/profile"
                  className={styles.dropdownItem}
                  onClick={() => setIsDropdownOpen(false)}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                    <circle cx="12" cy="7" r="4" />
                  </svg>
                  마이페이지
                </Link>
                <Link
                  to="/portfolio"
                  className={styles.dropdownItem}
                  onClick={() => setIsDropdownOpen(false)}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <polyline points="21 15 16 10 5 21" />
                  </svg>
                  포트폴리오
                </Link>
                <div className={styles.dropdownDivider} />
                <button
                  type="button"
                  className={styles.dropdownItem}
                  onClick={handleLogout}
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                    <polyline points="16 17 21 12 16 7" />
                    <line x1="21" y1="12" x2="9" y2="12" />
                  </svg>
                  로그아웃
                </button>
              </div>
            )}
          </div>
        ) : (
          // 비로그인 상태: 로그인 버튼
          onLoginClick && (
            <div className={styles.authContainer}>
              <button
                type="button"
                className={styles.loginButton}
                onClick={onLoginClick}
              >
                로그인
              </button>
            </div>
          )
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
  /** 로그인 버튼 클릭 핸들러 */
  onLoginClick: PropTypes.func,
  /** 로그아웃 핸들러 */
  onLogout: PropTypes.func,
};

export default Header;
