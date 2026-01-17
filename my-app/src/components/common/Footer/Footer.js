// Footer 컴포넌트
// SPEC-UI-001: 공통 UI 컴포넌트 - Footer
// 링크 목록과 저작권 정보를 표시하는 푸터 컴포넌트

import React from 'react';
import PropTypes from 'prop-types';
import styles from './Footer.module.css';

/**
 * Footer 컴포넌트
 * @param {Object} props - 컴포넌트 props
 * @param {Array<{label: string, href: string}>} props.links - 푸터 링크 배열
 * @param {string} props.copyright - 저작권 텍스트
 * @param {string} props.className - 추가 CSS 클래스
 */
const Footer = ({
  links = [],
  copyright,
  className = '',
  ...props
}) => {
  // 클래스 이름 조합
  const footerClasses = [
    styles.footer,
    className,
  ]
    .filter(Boolean)
    .join(' ');

  return (
    <footer className={footerClasses} {...props}>
      <div className={styles.container}>
        {/* 링크 영역 */}
        <nav className={styles.linkNav} role="navigation" aria-label="푸터 네비게이션">
          <ul className={styles.linkList}>
            {links.map((link, index) => (
              <li key={index} className={styles.linkItem}>
                <a href={link.href} className={styles.link}>
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        {/* 저작권 영역 */}
        <div className={styles.copyright}>
          {copyright}
        </div>
      </div>
    </footer>
  );
};

Footer.propTypes = {
  /** 푸터 링크 배열 */
  links: PropTypes.arrayOf(
    PropTypes.shape({
      label: PropTypes.string.isRequired,
      href: PropTypes.string.isRequired,
    })
  ),
  /** 저작권 텍스트 */
  copyright: PropTypes.string,
  /** 추가 CSS 클래스 */
  className: PropTypes.string,
};

export default Footer;
