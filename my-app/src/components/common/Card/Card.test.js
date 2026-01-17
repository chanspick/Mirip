// Card 컴포넌트 테스트
// SPEC-UI-001: 공통 UI 컴포넌트 - Card

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import Card from './Card';

describe('Card 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('children을 올바르게 렌더링한다', () => {
      render(<Card>카드 내용</Card>);
      expect(screen.getByText('카드 내용')).toBeInTheDocument();
    });

    test('기본 variant는 basic이다', () => {
      render(<Card data-testid="card">내용</Card>);
      const card = screen.getByTestId('card');
      expect(card).toHaveClass('basic');
    });

    test('기본 padding은 md이다', () => {
      render(<Card data-testid="card">내용</Card>);
      const card = screen.getByTestId('card');
      expect(card).toHaveClass('paddingMd');
    });
  });

  // variant 테스트
  describe('variant props', () => {
    test('variant="basic"이 올바르게 적용된다', () => {
      render(<Card variant="basic" data-testid="card">Basic</Card>);
      expect(screen.getByTestId('card')).toHaveClass('basic');
    });

    test('variant="frame"이 올바르게 적용된다', () => {
      render(<Card variant="frame" data-testid="card">Frame</Card>);
      expect(screen.getByTestId('card')).toHaveClass('frame');
    });
  });

  // padding 테스트
  describe('padding props', () => {
    test('padding="sm"이 올바르게 적용된다', () => {
      render(<Card padding="sm" data-testid="card">Small Padding</Card>);
      expect(screen.getByTestId('card')).toHaveClass('paddingSm');
    });

    test('padding="md"가 올바르게 적용된다', () => {
      render(<Card padding="md" data-testid="card">Medium Padding</Card>);
      expect(screen.getByTestId('card')).toHaveClass('paddingMd');
    });

    test('padding="lg"가 올바르게 적용된다', () => {
      render(<Card padding="lg" data-testid="card">Large Padding</Card>);
      expect(screen.getByTestId('card')).toHaveClass('paddingLg');
    });
  });

  // className 조합 테스트
  describe('className 조합', () => {
    test('여러 props가 조합되어 올바른 클래스가 적용된다', () => {
      render(
        <Card variant="frame" padding="lg" data-testid="card">
          Combined
        </Card>
      );
      const card = screen.getByTestId('card');
      expect(card).toHaveClass('card', 'frame', 'paddingLg');
    });

    test('추가 className을 전달할 수 있다', () => {
      render(<Card className="custom-class" data-testid="card">Custom</Card>);
      const card = screen.getByTestId('card');
      expect(card).toHaveClass('custom-class');
    });
  });

  // frame variant 스타일 테스트
  describe('frame variant 스타일', () => {
    test('frame variant는 frame 클래스를 가진다', () => {
      render(<Card variant="frame" data-testid="card">Frame Style</Card>);
      const card = screen.getByTestId('card');
      expect(card).toHaveClass('frame');
    });
  });

  // 중첩 컴포넌트 테스트
  describe('중첩 컴포넌트', () => {
    test('복잡한 children을 올바르게 렌더링한다', () => {
      render(
        <Card data-testid="card">
          <h2>제목</h2>
          <p>본문 내용</p>
          <button>버튼</button>
        </Card>
      );
      expect(screen.getByText('제목')).toBeInTheDocument();
      expect(screen.getByText('본문 내용')).toBeInTheDocument();
      expect(screen.getByText('버튼')).toBeInTheDocument();
    });
  });
});
