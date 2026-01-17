// Loading 컴포넌트 테스트
// SPEC-UI-001: 공통 UI 컴포넌트 - Loading

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Loading from './Loading';

describe('Loading 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('로딩 스피너가 렌더링된다', () => {
      render(<Loading data-testid="loading" />);
      expect(screen.getByTestId('loading')).toBeInTheDocument();
    });

    test('기본 size는 md이다', () => {
      render(<Loading data-testid="loading" />);
      const loading = screen.getByTestId('loading');
      expect(loading).toHaveClass('md');
    });

    test('스피너 요소가 포함되어 있다', () => {
      render(<Loading data-testid="loading" />);
      const spinner = screen.getByTestId('loading').querySelector('[class*="spinner"]');
      expect(spinner).toBeInTheDocument();
    });
  });

  // size 테스트
  describe('size props', () => {
    test('size="sm"이 올바르게 적용된다', () => {
      render(<Loading size="sm" data-testid="loading" />);
      expect(screen.getByTestId('loading')).toHaveClass('sm');
    });

    test('size="md"가 올바르게 적용된다', () => {
      render(<Loading size="md" data-testid="loading" />);
      expect(screen.getByTestId('loading')).toHaveClass('md');
    });

    test('size="lg"가 올바르게 적용된다', () => {
      render(<Loading size="lg" data-testid="loading" />);
      expect(screen.getByTestId('loading')).toHaveClass('lg');
    });
  });

  // fullScreen 테스트
  describe('fullScreen props', () => {
    test('fullScreen이 true일 때 fullScreen 클래스가 적용된다', () => {
      render(<Loading fullScreen data-testid="loading" />);
      expect(screen.getByTestId('loading')).toHaveClass('fullScreen');
    });

    test('fullScreen이 false일 때 fullScreen 클래스가 적용되지 않는다', () => {
      render(<Loading fullScreen={false} data-testid="loading" />);
      expect(screen.getByTestId('loading')).not.toHaveClass('fullScreen');
    });

    test('fullScreen 모드에서 overlay가 렌더링된다', () => {
      render(<Loading fullScreen data-testid="loading" />);
      const loading = screen.getByTestId('loading');
      expect(loading).toHaveClass('fullScreen');
    });
  });

  // color 테스트
  describe('color props', () => {
    test('커스텀 color가 inline style로 적용된다', () => {
      render(<Loading color="#FF0000" data-testid="loading" />);
      const spinner = screen.getByTestId('loading').querySelector('[class*="spinner"]');
      expect(spinner).toHaveStyle({ borderTopColor: '#FF0000' });
    });

    test('기본 color는 inline style이 적용되지 않는다', () => {
      render(<Loading data-testid="loading" />);
      const spinner = screen.getByTestId('loading').querySelector('[class*="spinner"]');
      // 기본 색상은 CSS 변수로 적용되므로 inline style 속성이 없거나 비어있음
      expect(spinner.style.borderTopColor).toBe('');
    });
  });

  // className 조합 테스트
  describe('className 조합', () => {
    test('여러 props가 조합되어 올바른 클래스가 적용된다', () => {
      render(
        <Loading size="lg" fullScreen data-testid="loading">
          로딩 중...
        </Loading>
      );
      const loading = screen.getByTestId('loading');
      expect(loading).toHaveClass('loading', 'lg', 'fullScreen');
    });

    test('추가 className을 전달할 수 있다', () => {
      render(<Loading className="custom-class" data-testid="loading" />);
      expect(screen.getByTestId('loading')).toHaveClass('custom-class');
    });
  });

  // 접근성 테스트
  describe('접근성', () => {
    test('role="status"가 적용되어 있다', () => {
      render(<Loading data-testid="loading" />);
      expect(screen.getByRole('status')).toBeInTheDocument();
    });

    test('aria-label이 적용되어 있다', () => {
      render(<Loading data-testid="loading" />);
      expect(screen.getByRole('status')).toHaveAttribute('aria-label', '로딩 중');
    });
  });
});
