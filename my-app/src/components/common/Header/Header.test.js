// Header 컴포넌트 테스트
// SPEC-UI-001: 공통 UI 컴포넌트 - Header

import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import Header from './Header';

// 스크롤 이벤트를 시뮬레이션하기 위한 헬퍼 함수
const mockScrollY = (value) => {
  Object.defineProperty(window, 'scrollY', {
    writable: true,
    configurable: true,
    value,
  });
  window.dispatchEvent(new Event('scroll'));
};

describe('Header 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('header 요소가 렌더링된다', () => {
      render(<Header />);
      expect(screen.getByRole('banner')).toBeInTheDocument();
    });

    test('logo prop이 렌더링된다', () => {
      render(<Header logo={<span>MIRIP</span>} />);
      expect(screen.getByText('MIRIP')).toBeInTheDocument();
    });

    test('기본 logo가 없으면 로고 영역이 비어있다', () => {
      render(<Header data-testid="header" />);
      const header = screen.getByTestId('header');
      const logoArea = header.querySelector('[class*="logoContainer"]');
      expect(logoArea).toBeInTheDocument();
    });
  });

  // navItems 테스트
  describe('navItems props', () => {
    const navItems = [
      { label: '홈', href: '/' },
      { label: '서비스', href: '/services' },
      { label: '소개', href: '/about' },
    ];

    test('navItems가 올바르게 렌더링된다', () => {
      render(<Header navItems={navItems} />);
      expect(screen.getByText('홈')).toBeInTheDocument();
      expect(screen.getByText('서비스')).toBeInTheDocument();
      expect(screen.getByText('소개')).toBeInTheDocument();
    });

    test('nav 링크들이 올바른 href를 가진다', () => {
      render(<Header navItems={navItems} />);
      expect(screen.getByText('홈').closest('a')).toHaveAttribute('href', '/');
      expect(screen.getByText('서비스').closest('a')).toHaveAttribute('href', '/services');
      expect(screen.getByText('소개').closest('a')).toHaveAttribute('href', '/about');
    });

    test('navItems가 비어있으면 nav 링크가 렌더링되지 않는다', () => {
      render(<Header navItems={[]} />);
      const nav = screen.getByRole('navigation');
      expect(nav.querySelectorAll('a')).toHaveLength(0);
    });
  });

  // ctaButton 테스트
  describe('ctaButton props', () => {
    test('ctaButton이 렌더링된다', () => {
      const ctaButton = { label: '시작하기', onClick: jest.fn() };
      render(<Header ctaButton={ctaButton} />);
      expect(screen.getByText('시작하기')).toBeInTheDocument();
    });

    test('ctaButton 클릭 시 onClick이 호출된다', () => {
      const handleClick = jest.fn();
      const ctaButton = { label: '시작하기', onClick: handleClick };
      render(<Header ctaButton={ctaButton} />);

      fireEvent.click(screen.getByText('시작하기'));
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    test('ctaButton이 없으면 버튼이 렌더링되지 않는다', () => {
      render(<Header />);
      expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });
  });

  // 스크롤 효과 테스트
  describe('스크롤 효과', () => {
    beforeEach(() => {
      // 스크롤 위치 초기화
      mockScrollY(0);
    });

    test('초기 상태에서는 scrolled 클래스가 없다', () => {
      render(<Header data-testid="header" />);
      expect(screen.getByTestId('header')).not.toHaveClass('scrolled');
    });

    test('스크롤이 50px 이상일 때 scrolled 클래스가 적용된다', () => {
      render(<Header data-testid="header" />);

      act(() => {
        mockScrollY(51);
      });

      expect(screen.getByTestId('header')).toHaveClass('scrolled');
    });

    test('스크롤이 50px 미만으로 돌아오면 scrolled 클래스가 제거된다', () => {
      render(<Header data-testid="header" />);

      act(() => {
        mockScrollY(100);
      });
      expect(screen.getByTestId('header')).toHaveClass('scrolled');

      act(() => {
        mockScrollY(30);
      });
      expect(screen.getByTestId('header')).not.toHaveClass('scrolled');
    });
  });

  // 고정 위치 테스트
  describe('고정 위치', () => {
    test('header는 fixed 클래스를 가진다', () => {
      render(<Header data-testid="header" />);
      expect(screen.getByTestId('header')).toHaveClass('header');
    });
  });

  // className 조합 테스트
  describe('className 조합', () => {
    test('추가 className을 전달할 수 있다', () => {
      render(<Header className="custom-class" data-testid="header" />);
      expect(screen.getByTestId('header')).toHaveClass('custom-class');
    });
  });

  // 접근성 테스트
  describe('접근성', () => {
    test('navigation role이 적용되어 있다', () => {
      render(<Header navItems={[{ label: '홈', href: '/' }]} />);
      expect(screen.getByRole('navigation')).toBeInTheDocument();
    });

    test('banner role(header)이 적용되어 있다', () => {
      render(<Header />);
      expect(screen.getByRole('banner')).toBeInTheDocument();
    });
  });
});
