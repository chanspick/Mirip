// StreakDisplay 컴포넌트 테스트
// SPEC-CRED-001: M2 스트릭 표시

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import StreakDisplay from './StreakDisplay';

// useStreak 훅 모킹
jest.mock('../../../hooks', () => ({
  useStreak: jest.fn(),
}));

import { useStreak } from '../../../hooks';

describe('StreakDisplay 컴포넌트', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('로딩 상태를 올바르게 표시한다', () => {
      useStreak.mockReturnValue({
        currentStreak: 0,
        longestStreak: 0,
        loading: true,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.getByTestId('streak-loading')).toBeInTheDocument();
    });

    test('에러 상태를 올바르게 표시한다', () => {
      useStreak.mockReturnValue({
        currentStreak: 0,
        longestStreak: 0,
        loading: false,
        error: '스트릭을 불러올 수 없습니다',
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.getByText('스트릭을 불러올 수 없습니다')).toBeInTheDocument();
    });

    test('스트릭 정보가 올바르게 렌더링된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.getByTestId('streak-container')).toBeInTheDocument();
    });
  });

  // 현재 스트릭 표시 테스트
  describe('현재 스트릭 표시', () => {
    test('현재 스트릭 수가 표시된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.getByTestId('current-streak-value')).toHaveTextContent('7');
    });

    test('현재 스트릭 레이블이 표시된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.getByText('현재 스트릭')).toBeInTheDocument();
    });

    test('스트릭이 있을 때 불꽃 이모지가 표시된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.getByTestId('fire-emoji')).toBeInTheDocument();
    });

    test('스트릭이 0일 때 불꽃 이모지가 흐리게 표시된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 0,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      const fireEmoji = screen.getByTestId('fire-emoji');
      expect(fireEmoji).toHaveClass('inactive');
    });
  });

  // 최장 스트릭 표시 테스트
  describe('최장 스트릭 표시', () => {
    test('최장 스트릭 수가 표시된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.getByTestId('longest-streak-value')).toHaveTextContent('30');
    });

    test('최장 스트릭 레이블이 표시된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.getByText('최장 기록')).toBeInTheDocument();
    });
  });

  // 단위 표시 테스트
  describe('단위 표시', () => {
    test('일 단위가 표시된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      const dayLabels = screen.getAllByText('일');
      expect(dayLabels.length).toBeGreaterThan(0);
    });
  });

  // 시각적 효과 테스트
  describe('시각적 효과', () => {
    test('높은 스트릭(7일 이상)에서 강조 스타일이 적용된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 10,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      const container = screen.getByTestId('streak-container');
      expect(container).toHaveClass('highlighted');
    });

    test('낮은 스트릭에서 기본 스타일이 적용된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 3,
        longestStreak: 10,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      const container = screen.getByTestId('streak-container');
      expect(container).not.toHaveClass('highlighted');
    });
  });

  // Props 테스트
  describe('Props 처리', () => {
    test('userId가 없으면 빈 상태를 표시한다', () => {
      useStreak.mockReturnValue({
        currentStreak: 0,
        longestStreak: 0,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay />);
      expect(screen.getByText('사용자 정보가 필요합니다')).toBeInTheDocument();
    });

    test('showLongest가 false이면 최장 스트릭이 표시되지 않는다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" showLongest={false} />);
      expect(screen.queryByTestId('longest-streak-value')).not.toBeInTheDocument();
    });

    test('compact 모드에서 축소된 레이아웃이 적용된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" compact />);
      const container = screen.getByTestId('streak-container');
      expect(container).toHaveClass('compact');
    });
  });

  // 총 활동 수 표시 테스트
  describe('총 활동 수 표시', () => {
    test('totalActivities prop이 전달되면 표시된다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" totalActivities={156} />);
      expect(screen.getByTestId('total-activities')).toHaveTextContent('156');
    });

    test('totalActivities가 없으면 해당 섹션이 표시되지 않는다', () => {
      useStreak.mockReturnValue({
        currentStreak: 7,
        longestStreak: 30,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<StreakDisplay userId="test-user" />);
      expect(screen.queryByTestId('total-activities')).not.toBeInTheDocument();
    });
  });
});
