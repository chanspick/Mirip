// Button 컴포넌트 테스트
// SPEC-UI-001: 공통 UI 컴포넌트 - Button

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import Button from './Button';

describe('Button 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('children을 올바르게 렌더링한다', () => {
      render(<Button>클릭하세요</Button>);
      expect(screen.getByRole('button')).toHaveTextContent('클릭하세요');
    });

    test('기본 variant는 primary이다', () => {
      render(<Button>버튼</Button>);
      const button = screen.getByRole('button');
      expect(button).toHaveClass('primary');
    });

    test('기본 size는 md이다', () => {
      render(<Button>버튼</Button>);
      const button = screen.getByRole('button');
      expect(button).toHaveClass('md');
    });
  });

  // variant 테스트
  describe('variant props', () => {
    test('variant="primary"가 올바르게 적용된다', () => {
      render(<Button variant="primary">Primary</Button>);
      expect(screen.getByRole('button')).toHaveClass('primary');
    });

    test('variant="cta"가 올바르게 적용된다', () => {
      render(<Button variant="cta">CTA</Button>);
      expect(screen.getByRole('button')).toHaveClass('cta');
    });

    test('variant="outline"이 올바르게 적용된다', () => {
      render(<Button variant="outline">Outline</Button>);
      expect(screen.getByRole('button')).toHaveClass('outline');
    });
  });

  // size 테스트
  describe('size props', () => {
    test('size="sm"이 올바르게 적용된다', () => {
      render(<Button size="sm">Small</Button>);
      expect(screen.getByRole('button')).toHaveClass('sm');
    });

    test('size="md"가 올바르게 적용된다', () => {
      render(<Button size="md">Medium</Button>);
      expect(screen.getByRole('button')).toHaveClass('md');
    });

    test('size="lg"가 올바르게 적용된다', () => {
      render(<Button size="lg">Large</Button>);
      expect(screen.getByRole('button')).toHaveClass('lg');
    });
  });

  // disabled 테스트
  describe('disabled props', () => {
    test('disabled가 true일 때 버튼이 비활성화된다', () => {
      render(<Button disabled>Disabled</Button>);
      expect(screen.getByRole('button')).toBeDisabled();
    });

    test('disabled가 true일 때 disabled 클래스가 적용된다', () => {
      render(<Button disabled>Disabled</Button>);
      expect(screen.getByRole('button')).toHaveClass('disabled');
    });

    test('disabled가 false일 때 버튼이 활성화된다', () => {
      render(<Button disabled={false}>Enabled</Button>);
      expect(screen.getByRole('button')).not.toBeDisabled();
    });
  });

  // fullWidth 테스트
  describe('fullWidth props', () => {
    test('fullWidth가 true일 때 fullWidth 클래스가 적용된다', () => {
      render(<Button fullWidth>Full Width</Button>);
      expect(screen.getByRole('button')).toHaveClass('fullWidth');
    });

    test('fullWidth가 false일 때 fullWidth 클래스가 적용되지 않는다', () => {
      render(<Button fullWidth={false}>Normal Width</Button>);
      expect(screen.getByRole('button')).not.toHaveClass('fullWidth');
    });
  });

  // onClick 테스트
  describe('onClick props', () => {
    test('클릭 시 onClick 핸들러가 호출된다', () => {
      const handleClick = jest.fn();
      render(<Button onClick={handleClick}>Click</Button>);

      fireEvent.click(screen.getByRole('button'));
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    test('disabled 상태에서는 onClick이 호출되지 않는다', () => {
      const handleClick = jest.fn();
      render(<Button onClick={handleClick} disabled>Disabled</Button>);

      fireEvent.click(screen.getByRole('button'));
      expect(handleClick).not.toHaveBeenCalled();
    });
  });

  // 키보드 접근성 테스트
  describe('키보드 접근성', () => {
    test('Enter 키로 버튼을 활성화할 수 있다', () => {
      const handleClick = jest.fn();
      render(<Button onClick={handleClick}>Enter Key</Button>);

      const button = screen.getByRole('button');
      fireEvent.keyDown(button, { key: 'Enter', code: 'Enter' });
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    test('Space 키로 버튼을 활성화할 수 있다', () => {
      const handleClick = jest.fn();
      render(<Button onClick={handleClick}>Space Key</Button>);

      const button = screen.getByRole('button');
      fireEvent.keyDown(button, { key: ' ', code: 'Space' });
      expect(handleClick).toHaveBeenCalledTimes(1);
    });
  });

  // className 조합 테스트
  describe('className 조합', () => {
    test('여러 props가 조합되어 올바른 클래스가 적용된다', () => {
      render(
        <Button variant="cta" size="lg" fullWidth disabled>
          Combined
        </Button>
      );
      const button = screen.getByRole('button');
      expect(button).toHaveClass('button', 'cta', 'lg', 'fullWidth', 'disabled');
    });

    test('추가 className을 전달할 수 있다', () => {
      render(<Button className="custom-class">Custom</Button>);
      const button = screen.getByRole('button');
      expect(button).toHaveClass('custom-class');
    });
  });
});
