// ActivityTimeline 컴포넌트 테스트
// SPEC-CRED-001: M2 활동 타임라인

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ActivityTimeline from './ActivityTimeline';

// useActivities 훅 모킹
jest.mock('../../../hooks', () => ({
  __esModule: true,
  default: jest.fn(),
  useActivities: jest.fn(),
}));

import { useActivities } from '../../../hooks';

// 테스트용 활동 데이터
const mockActivities = [
  {
    id: '1',
    type: 'diagnosis',
    title: 'AI 진단 완료',
    description: '홍익대 실기 진단 결과',
    createdAt: { toDate: () => new Date('2025-01-15T10:30:00') },
    metadata: { score: 85 },
  },
  {
    id: '2',
    type: 'competition',
    title: '공모전 참가',
    description: '2025 미술대전 출품',
    createdAt: { toDate: () => new Date('2025-01-14T14:00:00') },
    metadata: { competitionId: 'comp-1' },
  },
  {
    id: '3',
    type: 'portfolio',
    title: '포트폴리오 업데이트',
    description: '새 작품 3개 추가',
    createdAt: { toDate: () => new Date('2025-01-13T09:15:00') },
    metadata: { portfolioId: 'port-1' },
  },
];

describe('ActivityTimeline 컴포넌트', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('로딩 상태를 올바르게 표시한다', () => {
      useActivities.mockReturnValue({
        activities: [],
        loading: true,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);
      expect(screen.getByTestId('timeline-loading')).toBeInTheDocument();
    });

    test('에러 상태를 올바르게 표시한다', () => {
      useActivities.mockReturnValue({
        activities: [],
        loading: false,
        loadingMore: false,
        error: '활동을 불러올 수 없습니다',
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);
      expect(screen.getByText('활동을 불러올 수 없습니다')).toBeInTheDocument();
    });

    test('활동이 없을 때 빈 상태 메시지를 표시한다', () => {
      useActivities.mockReturnValue({
        activities: [],
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);
      expect(screen.getByText('아직 활동 기록이 없습니다')).toBeInTheDocument();
    });

    test('활동 목록이 올바르게 렌더링된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      expect(screen.getByText('AI 진단 완료')).toBeInTheDocument();
      expect(screen.getByText('공모전 참가')).toBeInTheDocument();
      expect(screen.getByText('포트폴리오 업데이트')).toBeInTheDocument();
    });
  });

  // 활동 아이템 테스트
  describe('활동 아이템 표시', () => {
    test('활동 타입에 따른 아이콘이 표시된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      // 각 활동 타입별 아이콘 확인
      expect(screen.getByTestId('activity-icon-diagnosis')).toBeInTheDocument();
      expect(screen.getByTestId('activity-icon-competition')).toBeInTheDocument();
      expect(screen.getByTestId('activity-icon-portfolio')).toBeInTheDocument();
    });

    test('활동 설명이 표시된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      expect(screen.getByText('홍익대 실기 진단 결과')).toBeInTheDocument();
      expect(screen.getByText('2025 미술대전 출품')).toBeInTheDocument();
    });

    test('활동 시간이 상대적으로 표시된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      // 날짜/시간 정보가 표시되어야 함
      const timeElements = document.querySelectorAll('[data-testid="activity-time"]');
      expect(timeElements.length).toBe(3);
    });
  });

  // 페이지네이션 테스트
  describe('페이지네이션', () => {
    test('더 불러오기 버튼이 hasMore일 때 표시된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: true,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);
      expect(screen.getByTestId('load-more-button')).toBeInTheDocument();
    });

    test('더 불러오기 버튼 클릭 시 loadMore가 호출된다', () => {
      const loadMoreMock = jest.fn();
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: true,
        loadMore: loadMoreMock,
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      fireEvent.click(screen.getByTestId('load-more-button'));
      expect(loadMoreMock).toHaveBeenCalled();
    });

    test('loadingMore일 때 로딩 표시가 나타난다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: true,
        error: null,
        hasMore: true,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);
      expect(screen.getByTestId('loading-more')).toBeInTheDocument();
    });

    test('hasMore가 false이면 버튼이 표시되지 않는다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);
      expect(screen.queryByTestId('load-more-button')).not.toBeInTheDocument();
    });
  });

  // 필터 테스트
  describe('활동 타입 필터', () => {
    test('필터 버튼이 표시된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      expect(screen.getByTestId('filter-all')).toBeInTheDocument();
      expect(screen.getByTestId('filter-diagnosis')).toBeInTheDocument();
      expect(screen.getByTestId('filter-competition')).toBeInTheDocument();
      expect(screen.getByTestId('filter-portfolio')).toBeInTheDocument();
    });

    test('필터 클릭 시 해당 타입만 표시된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      // diagnosis 필터 클릭
      fireEvent.click(screen.getByTestId('filter-diagnosis'));

      // useActivities가 type 옵션과 함께 호출되었는지 확인
      expect(useActivities).toHaveBeenCalledWith('test-user', expect.objectContaining({
        type: 'diagnosis',
      }));
    });

    test('전체 필터 클릭 시 모든 활동이 표시된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      // 먼저 다른 필터 선택
      fireEvent.click(screen.getByTestId('filter-diagnosis'));

      // 전체 필터 클릭
      fireEvent.click(screen.getByTestId('filter-all'));

      // type이 undefined로 호출되어야 함
      expect(useActivities).toHaveBeenCalledWith('test-user', expect.objectContaining({
        type: undefined,
      }));
    });
  });

  // Props 테스트
  describe('Props 처리', () => {
    test('userId가 없으면 빈 상태를 표시한다', () => {
      useActivities.mockReturnValue({
        activities: [],
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline />);
      expect(screen.getByText('사용자 정보가 필요합니다')).toBeInTheDocument();
    });

    test('limit prop이 전달되면 해당 개수만큼 요청한다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" limit={5} />);

      expect(useActivities).toHaveBeenCalledWith('test-user', expect.objectContaining({
        pageSize: 5,
      }));
    });

    test('filterTypes prop이 전달되면 필터 버튼이 제한된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" filterTypes={['diagnosis', 'competition']} />);

      expect(screen.getByTestId('filter-diagnosis')).toBeInTheDocument();
      expect(screen.getByTestId('filter-competition')).toBeInTheDocument();
      expect(screen.queryByTestId('filter-portfolio')).not.toBeInTheDocument();
    });
  });

  // 타임라인 레이아웃 테스트
  describe('타임라인 레이아웃', () => {
    test('세로 타임라인 레이아웃이 적용된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      expect(screen.getByTestId('timeline-container')).toHaveClass('timeline');
    });

    test('타임라인 라인이 표시된다', () => {
      useActivities.mockReturnValue({
        activities: mockActivities,
        loading: false,
        loadingMore: false,
        error: null,
        hasMore: false,
        loadMore: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ActivityTimeline userId="test-user" />);

      const timelineItems = document.querySelectorAll('[data-testid="timeline-item"]');
      expect(timelineItems.length).toBe(3);
    });
  });
});
