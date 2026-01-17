// Footer 컴포넌트 테스트
// SPEC-UI-001: 공통 UI 컴포넌트 - Footer

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Footer from './Footer';

describe('Footer 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('footer 요소가 렌더링된다', () => {
      render(<Footer />);
      expect(screen.getByRole('contentinfo')).toBeInTheDocument();
    });

    test('footer 클래스가 적용된다', () => {
      render(<Footer data-testid="footer" />);
      expect(screen.getByTestId('footer')).toHaveClass('footer');
    });
  });

  // links 테스트
  describe('links props', () => {
    const links = [
      { label: '이용약관', href: '/terms' },
      { label: '개인정보처리방침', href: '/privacy' },
      { label: '고객센터', href: '/support' },
    ];

    test('links가 올바르게 렌더링된다', () => {
      render(<Footer links={links} />);
      expect(screen.getByText('이용약관')).toBeInTheDocument();
      expect(screen.getByText('개인정보처리방침')).toBeInTheDocument();
      expect(screen.getByText('고객센터')).toBeInTheDocument();
    });

    test('links가 올바른 href를 가진다', () => {
      render(<Footer links={links} />);
      expect(screen.getByText('이용약관').closest('a')).toHaveAttribute('href', '/terms');
      expect(screen.getByText('개인정보처리방침').closest('a')).toHaveAttribute('href', '/privacy');
      expect(screen.getByText('고객센터').closest('a')).toHaveAttribute('href', '/support');
    });

    test('links가 비어있으면 링크가 렌더링되지 않는다', () => {
      render(<Footer links={[]} data-testid="footer" />);
      const linkList = screen.getByTestId('footer').querySelector('[class*="linkList"]');
      expect(linkList.querySelectorAll('a')).toHaveLength(0);
    });

    test('links가 제공되지 않으면 기본값으로 빈 배열이 사용된다', () => {
      render(<Footer data-testid="footer" />);
      const linkList = screen.getByTestId('footer').querySelector('[class*="linkList"]');
      expect(linkList.querySelectorAll('a')).toHaveLength(0);
    });
  });

  // copyright 테스트
  describe('copyright props', () => {
    test('copyright가 올바르게 렌더링된다', () => {
      const copyright = 'Copyright 2024 MIRIP. All rights reserved.';
      render(<Footer copyright={copyright} />);
      expect(screen.getByText(copyright)).toBeInTheDocument();
    });

    test('copyright가 없으면 copyright 영역이 렌더링되지 않는다', () => {
      render(<Footer data-testid="footer" />);
      const footer = screen.getByTestId('footer');
      const copyrightArea = footer.querySelector('[class*="copyright"]');
      expect(copyrightArea).toBeEmptyDOMElement();
    });
  });

  // className 조합 테스트
  describe('className 조합', () => {
    test('추가 className을 전달할 수 있다', () => {
      render(<Footer className="custom-class" data-testid="footer" />);
      expect(screen.getByTestId('footer')).toHaveClass('custom-class');
    });
  });

  // 접근성 테스트
  describe('접근성', () => {
    test('contentinfo role이 적용되어 있다', () => {
      render(<Footer />);
      expect(screen.getByRole('contentinfo')).toBeInTheDocument();
    });

    test('navigation role이 링크 영역에 적용되어 있다', () => {
      const links = [{ label: '홈', href: '/' }];
      render(<Footer links={links} />);
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });
  });

  // 복합 테스트
  describe('복합 렌더링', () => {
    test('links와 copyright가 함께 렌더링된다', () => {
      const links = [
        { label: '이용약관', href: '/terms' },
        { label: '개인정보처리방침', href: '/privacy' },
      ];
      const copyright = 'Copyright 2024 MIRIP';

      render(<Footer links={links} copyright={copyright} />);

      expect(screen.getByText('이용약관')).toBeInTheDocument();
      expect(screen.getByText('개인정보처리방침')).toBeInTheDocument();
      expect(screen.getByText(copyright)).toBeInTheDocument();
    });
  });
});
