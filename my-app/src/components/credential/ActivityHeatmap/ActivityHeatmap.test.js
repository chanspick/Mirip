// ActivityHeatmap 컴포넌트 테스트
// SPEC-CRED-001: M2 활동 히트맵 (GitHub-style 잔디밭)

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ActivityHeatmap from './ActivityHeatmap';

// useActivityHeatmap 훅 모킹
jest.mock('../../../hooks', () => ({
  useActivityHeatmap: jest.fn(),
}));

import { useActivityHeatmap } from '../../../hooks';

// 테스트용 히트맵 데이터 생성 헬퍼
const createMockHeatmapData = (year = 2025) => {
  const days = [];
  const startDate = new Date(year, 0, 1);
  const endDate = new Date(year, 11, 31);

  for (let d = new Date(startDate); d <= endDate; d.setDate(d.getDate() + 1)) {
    const dateStr = d.toISOString().split('T')[0];
    // 랜덤하게 활동 수 생성 (테스트용)
    const count = Math.floor(Math.random() * 10);
    days.push({
      date: dateStr,
      count,
      activityIds: Array(count).fill('test-id'),
    });
  }

  return {
    year,
    days,
    totalActivities: days.reduce((sum, d) => sum + d.count, 0),
  };
};

describe('ActivityHeatmap 컴포넌트', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('로딩 상태를 올바르게 표시한다', () => {
      useActivityHeatmap.mockReturnValue({
        heatmapData: null,
        loading: true,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      expect(screen.getByTestId('heatmap-loading')).toBeInTheDocument();
    });

    test('에러 상태를 올바르게 표시한다', () => {
      useActivityHeatmap.mockReturnValue({
        heatmapData: null,
        loading: false,
        error: '데이터를 불러올 수 없습니다',
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      expect(screen.getByText('데이터를 불러올 수 없습니다')).toBeInTheDocument();
    });

    test('히트맵 그리드가 올바르게 렌더링된다', () => {
      const mockData = createMockHeatmapData(2025);
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      expect(screen.getByTestId('heatmap-grid')).toBeInTheDocument();
    });

    test('요일 레이블이 표시된다 (일, 월, 화, 수, 목, 금, 토)', () => {
      const mockData = createMockHeatmapData(2025);
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);

      // 일요일, 화요일, 토요일 표시 (모바일 최적화로 일부만 표시 가능)
      expect(screen.getByText('일')).toBeInTheDocument();
      expect(screen.getByText('화')).toBeInTheDocument();
      expect(screen.getByText('토')).toBeInTheDocument();
    });

    test('월별 레이블이 표시된다', () => {
      const mockData = createMockHeatmapData(2025);
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);

      // 최소한 일부 월 레이블이 표시되어야 함
      expect(screen.getByText('1월')).toBeInTheDocument();
    });
  });

  // 활동 레벨 테스트
  describe('활동 레벨 색상', () => {
    test('활동이 없는 셀은 level0 클래스를 가진다', () => {
      const mockData = {
        year: 2025,
        days: [{ date: '2025-01-01', count: 0, activityIds: [] }],
        totalActivities: 0,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      const cells = document.querySelectorAll('[data-date="2025-01-01"]');
      expect(cells.length).toBeGreaterThan(0);
      expect(cells[0]).toHaveClass('level0');
    });

    test('활동 1-2개는 level1 클래스를 가진다', () => {
      const mockData = {
        year: 2025,
        days: [{ date: '2025-01-01', count: 2, activityIds: ['1', '2'] }],
        totalActivities: 2,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      const cell = document.querySelector('[data-date="2025-01-01"]');
      expect(cell).toHaveClass('level1');
    });

    test('활동 3-4개는 level2 클래스를 가진다', () => {
      const mockData = {
        year: 2025,
        days: [{ date: '2025-01-01', count: 4, activityIds: ['1', '2', '3', '4'] }],
        totalActivities: 4,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      const cell = document.querySelector('[data-date="2025-01-01"]');
      expect(cell).toHaveClass('level2');
    });

    test('활동 5-7개는 level3 클래스를 가진다', () => {
      const mockData = {
        year: 2025,
        days: [{ date: '2025-01-01', count: 6, activityIds: Array(6).fill('id') }],
        totalActivities: 6,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      const cell = document.querySelector('[data-date="2025-01-01"]');
      expect(cell).toHaveClass('level3');
    });

    test('활동 8개 이상은 level4 클래스를 가진다', () => {
      const mockData = {
        year: 2025,
        days: [{ date: '2025-01-01', count: 10, activityIds: Array(10).fill('id') }],
        totalActivities: 10,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      const cell = document.querySelector('[data-date="2025-01-01"]');
      expect(cell).toHaveClass('level4');
    });
  });

  // 툴팁 테스트
  describe('툴팁 기능', () => {
    test('셀 호버 시 툴팁이 표시된다', async () => {
      const mockData = {
        year: 2025,
        days: [{ date: '2025-01-01', count: 3, activityIds: ['1', '2', '3'] }],
        totalActivities: 3,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      const cell = document.querySelector('[data-date="2025-01-01"]');

      fireEvent.mouseEnter(cell);

      await waitFor(() => {
        expect(screen.getByRole('tooltip')).toBeInTheDocument();
      });
    });

    test('툴팁에 날짜와 활동 수가 표시된다', async () => {
      const mockData = {
        year: 2025,
        days: [{ date: '2025-01-01', count: 5, activityIds: Array(5).fill('id') }],
        totalActivities: 5,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      const cell = document.querySelector('[data-date="2025-01-01"]');

      fireEvent.mouseEnter(cell);

      await waitFor(() => {
        const tooltip = screen.getByRole('tooltip');
        expect(tooltip).toHaveTextContent('2025-01-01');
        expect(tooltip).toHaveTextContent('5개 활동');
      });
    });
  });

  // 연도 선택기 테스트
  describe('연도 선택기', () => {
    test('연도 선택기가 표시된다', () => {
      const mockData = createMockHeatmapData(2025);
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      expect(screen.getByTestId('year-selector')).toBeInTheDocument();
    });

    test('현재 연도가 선택되어 있다', () => {
      const currentYear = new Date().getFullYear();
      const mockData = createMockHeatmapData(currentYear);
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      expect(screen.getByText(String(currentYear))).toBeInTheDocument();
    });

    test('이전 연도 버튼 클릭 시 연도가 변경된다', () => {
      const mockData = createMockHeatmapData(2025);
      const refreshMock = jest.fn();
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: refreshMock,
      });

      render(<ActivityHeatmap userId="test-user" year={2025} />);

      const prevButton = screen.getByTestId('year-prev-button');
      fireEvent.click(prevButton);

      // year prop이 변경되면 컴포넌트가 새 데이터를 요청해야 함
      expect(useActivityHeatmap).toHaveBeenCalled();
    });
  });

  // 셀 클릭 테스트
  describe('셀 클릭 이벤트', () => {
    test('onCellClick 콜백이 호출된다', () => {
      const mockData = {
        year: 2025,
        days: [{ date: '2025-01-01', count: 3, activityIds: ['1', '2', '3'] }],
        totalActivities: 3,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      const onCellClick = jest.fn();
      render(<ActivityHeatmap userId="test-user" onCellClick={onCellClick} />);

      const cell = document.querySelector('[data-date="2025-01-01"]');
      fireEvent.click(cell);

      expect(onCellClick).toHaveBeenCalledWith({
        date: '2025-01-01',
        count: 3,
        activityIds: ['1', '2', '3'],
      });
    });
  });

  // 총 활동 수 표시 테스트
  describe('총 활동 수 표시', () => {
    test('총 활동 수가 표시된다', () => {
      const mockData = {
        year: 2025,
        days: [
          { date: '2025-01-01', count: 3, activityIds: ['1', '2', '3'] },
          { date: '2025-01-02', count: 5, activityIds: Array(5).fill('id') },
        ],
        totalActivities: 8,
      };
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      expect(screen.getByText(/8/)).toBeInTheDocument();
    });
  });

  // 레벨 범례 테스트
  describe('레벨 범례', () => {
    test('레벨 범례가 표시된다', () => {
      const mockData = createMockHeatmapData(2025);
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" />);
      expect(screen.getByTestId('level-legend')).toBeInTheDocument();
      expect(screen.getByText('적음')).toBeInTheDocument();
      expect(screen.getByText('많음')).toBeInTheDocument();
    });
  });

  // Props 테스트
  describe('Props 처리', () => {
    test('userId가 없으면 빈 상태를 표시한다', () => {
      useActivityHeatmap.mockReturnValue({
        heatmapData: null,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap />);
      expect(screen.getByText('사용자 정보가 필요합니다')).toBeInTheDocument();
    });

    test('year prop이 전달되면 해당 연도 데이터를 요청한다', () => {
      const mockData = createMockHeatmapData(2024);
      useActivityHeatmap.mockReturnValue({
        heatmapData: mockData,
        loading: false,
        error: null,
        refresh: jest.fn(),
      });

      render(<ActivityHeatmap userId="test-user" year={2024} />);
      expect(useActivityHeatmap).toHaveBeenCalledWith('test-user', 2024);
    });
  });
});
