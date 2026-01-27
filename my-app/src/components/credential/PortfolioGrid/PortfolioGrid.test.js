// PortfolioGrid 컴포넌트 테스트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 그리드
// TDD RED Phase: 실패하는 테스트 먼저 작성

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import PortfolioGrid from './PortfolioGrid';

// 테스트용 포트폴리오 데이터
const mockPortfolios = [
  {
    id: 'portfolio-001',
    userId: 'user-123',
    title: '정물화 작품',
    description: '사과와 꽃병을 그린 정물화입니다.',
    imageUrl: 'https://example.com/artwork1.jpg',
    thumbnailUrl: 'https://example.com/artwork1-thumb.jpg',
    tags: ['정물화', '수채화'],
    isPublic: true,
    createdAt: { toDate: () => new Date('2024-01-15') },
  },
  {
    id: 'portfolio-002',
    userId: 'user-123',
    title: '풍경화 작품',
    description: '산과 강을 그린 풍경화입니다.',
    imageUrl: 'https://example.com/artwork2.jpg',
    thumbnailUrl: 'https://example.com/artwork2-thumb.jpg',
    tags: ['풍경화', '유화'],
    isPublic: true,
    createdAt: { toDate: () => new Date('2024-01-14') },
  },
  {
    id: 'portfolio-003',
    userId: 'user-123',
    title: '인물화 작품',
    description: '친구의 초상화입니다.',
    imageUrl: 'https://example.com/artwork3.jpg',
    thumbnailUrl: 'https://example.com/artwork3-thumb.jpg',
    tags: ['인물화', '아크릴'],
    isPublic: false,
    createdAt: { toDate: () => new Date('2024-01-13') },
  },
];

describe('PortfolioGrid 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('그리드 컨테이너가 렌더링된다', () => {
      render(<PortfolioGrid portfolios={mockPortfolios} />);
      expect(screen.getByTestId('portfolio-grid')).toBeInTheDocument();
    });

    test('모든 포트폴리오 카드가 렌더링된다', () => {
      render(<PortfolioGrid portfolios={mockPortfolios} />);
      const cards = screen.getAllByTestId('portfolio-card');
      expect(cards).toHaveLength(3);
    });

    test('각 포트폴리오 제목이 표시된다', () => {
      render(<PortfolioGrid portfolios={mockPortfolios} />);
      expect(screen.getByText('정물화 작품')).toBeInTheDocument();
      expect(screen.getByText('풍경화 작품')).toBeInTheDocument();
      expect(screen.getByText('인물화 작품')).toBeInTheDocument();
    });

    test('빈 배열이면 그리드가 렌더링된다', () => {
      render(<PortfolioGrid portfolios={[]} />);
      expect(screen.getByTestId('portfolio-grid')).toBeInTheDocument();
    });
  });

  // 빈 상태 테스트
  describe('빈 상태', () => {
    test('포트폴리오가 없으면 빈 상태 메시지가 표시된다', () => {
      render(<PortfolioGrid portfolios={[]} />);
      expect(screen.getByTestId('empty-state')).toBeInTheDocument();
    });

    test('빈 상태에 기본 메시지가 표시된다', () => {
      render(<PortfolioGrid portfolios={[]} />);
      expect(screen.getByText(/포트폴리오가 없습니다/)).toBeInTheDocument();
    });

    test('사용자 정의 빈 상태 메시지를 표시할 수 있다', () => {
      render(
        <PortfolioGrid
          portfolios={[]}
          emptyMessage="아직 작품이 없어요!"
        />
      );
      expect(screen.getByText('아직 작품이 없어요!')).toBeInTheDocument();
    });
  });

  // 로딩 상태 테스트
  describe('로딩 상태', () => {
    test('loading이 true이면 로딩 표시가 나타난다', () => {
      render(<PortfolioGrid portfolios={[]} loading={true} />);
      expect(screen.getByTestId('loading-indicator')).toBeInTheDocument();
    });

    test('loading이 false이면 로딩 표시가 없다', () => {
      render(<PortfolioGrid portfolios={mockPortfolios} loading={false} />);
      expect(screen.queryByTestId('loading-indicator')).not.toBeInTheDocument();
    });

    test('로딩 중에도 기존 포트폴리오는 표시된다', () => {
      render(<PortfolioGrid portfolios={mockPortfolios} loading={true} />);
      expect(screen.getAllByTestId('portfolio-card')).toHaveLength(3);
    });
  });

  // 소유자 모드 테스트
  describe('소유자 모드', () => {
    test('isOwner가 true이면 카드에 편집/삭제 버튼이 표시된다', () => {
      render(<PortfolioGrid portfolios={mockPortfolios} isOwner={true} />);
      expect(screen.getAllByTestId('edit-button')).toHaveLength(3);
      expect(screen.getAllByTestId('delete-button')).toHaveLength(3);
    });

    test('isOwner가 false이면 편집/삭제 버튼이 없다', () => {
      render(<PortfolioGrid portfolios={mockPortfolios} isOwner={false} />);
      expect(screen.queryByTestId('edit-button')).not.toBeInTheDocument();
      expect(screen.queryByTestId('delete-button')).not.toBeInTheDocument();
    });
  });

  // 콜백 핸들러 테스트
  describe('콜백 핸들러', () => {
    test('카드 클릭 시 onCardClick이 호출된다', () => {
      const handleCardClick = jest.fn();
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          onCardClick={handleCardClick}
        />
      );
      const cards = screen.getAllByTestId('portfolio-card');
      fireEvent.click(cards[0]);
      expect(handleCardClick).toHaveBeenCalledWith(mockPortfolios[0]);
    });

    test('편집 버튼 클릭 시 onEdit이 호출된다', () => {
      const handleEdit = jest.fn();
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          isOwner={true}
          onEdit={handleEdit}
        />
      );
      const editButtons = screen.getAllByTestId('edit-button');
      fireEvent.click(editButtons[0]);
      expect(handleEdit).toHaveBeenCalledWith(mockPortfolios[0]);
    });

    test('삭제 버튼 클릭 시 onDelete가 호출된다', () => {
      const handleDelete = jest.fn();
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          isOwner={true}
          onDelete={handleDelete}
        />
      );
      const deleteButtons = screen.getAllByTestId('delete-button');
      fireEvent.click(deleteButtons[1]);
      expect(handleDelete).toHaveBeenCalledWith(mockPortfolios[1]);
    });
  });

  // 필터 테스트
  describe('태그 필터', () => {
    test('필터가 활성화되면 필터 UI가 표시된다', () => {
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          showFilter={true}
        />
      );
      expect(screen.getByTestId('filter-container')).toBeInTheDocument();
    });

    test('모든 고유 태그가 필터 옵션에 표시된다', () => {
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          showFilter={true}
        />
      );
      expect(screen.getByText('정물화')).toBeInTheDocument();
      expect(screen.getByText('풍경화')).toBeInTheDocument();
      expect(screen.getByText('인물화')).toBeInTheDocument();
    });

    test('필터가 비활성화되면 필터 UI가 없다', () => {
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          showFilter={false}
        />
      );
      expect(screen.queryByTestId('filter-container')).not.toBeInTheDocument();
    });
  });

  // 정렬 테스트
  describe('정렬', () => {
    test('정렬 옵션이 있으면 정렬 UI가 표시된다', () => {
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          showSort={true}
        />
      );
      expect(screen.getByTestId('sort-select')).toBeInTheDocument();
    });

    test('정렬 옵션이 없으면 정렬 UI가 없다', () => {
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          showSort={false}
        />
      );
      expect(screen.queryByTestId('sort-select')).not.toBeInTheDocument();
    });
  });

  // 더 보기 테스트
  describe('더 보기 (Load More)', () => {
    test('hasMore가 true이면 더 보기 버튼이 표시된다', () => {
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          hasMore={true}
        />
      );
      expect(screen.getByTestId('load-more-button')).toBeInTheDocument();
    });

    test('hasMore가 false이면 더 보기 버튼이 없다', () => {
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          hasMore={false}
        />
      );
      expect(screen.queryByTestId('load-more-button')).not.toBeInTheDocument();
    });

    test('더 보기 버튼 클릭 시 onLoadMore가 호출된다', () => {
      const handleLoadMore = jest.fn();
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          hasMore={true}
          onLoadMore={handleLoadMore}
        />
      );
      fireEvent.click(screen.getByTestId('load-more-button'));
      expect(handleLoadMore).toHaveBeenCalledTimes(1);
    });

    test('loadingMore가 true이면 더 보기 버튼이 비활성화된다', () => {
      render(
        <PortfolioGrid
          portfolios={mockPortfolios}
          hasMore={true}
          loadingMore={true}
        />
      );
      expect(screen.getByTestId('load-more-button')).toBeDisabled();
    });
  });

  // 반응형 그리드 테스트
  describe('그리드 레이아웃', () => {
    test('그리드 컨테이너가 grid 클래스를 가진다', () => {
      render(<PortfolioGrid portfolios={mockPortfolios} />);
      const grid = screen.getByTestId('portfolio-grid');
      expect(grid).toHaveClass('grid');
    });
  });
});
