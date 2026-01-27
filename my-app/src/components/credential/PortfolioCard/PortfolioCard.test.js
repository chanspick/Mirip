// PortfolioCard 컴포넌트 테스트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 카드
// TDD RED Phase: 실패하는 테스트 먼저 작성

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import PortfolioCard from './PortfolioCard';

// 테스트용 포트폴리오 데이터
const mockPortfolio = {
  id: 'portfolio-001',
  userId: 'user-123',
  title: '정물화 작품',
  description: '사과와 꽃병을 그린 정물화입니다. 색연필과 수채화 기법을 혼합하여 사용했습니다.',
  imageUrl: 'https://example.com/artwork.jpg',
  thumbnailUrl: 'https://example.com/artwork-thumb.jpg',
  tags: ['정물화', '수채화', '색연필'],
  isPublic: true,
  createdAt: { toDate: () => new Date('2024-01-15') },
  updatedAt: { toDate: () => new Date('2024-01-16') },
};

// 비공개 포트폴리오
const privatePortfolio = {
  ...mockPortfolio,
  id: 'portfolio-002',
  isPublic: false,
};

describe('PortfolioCard 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('포트폴리오 카드가 렌더링된다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} />);
      expect(screen.getByTestId('portfolio-card')).toBeInTheDocument();
    });

    test('제목이 표시된다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} />);
      expect(screen.getByText('정물화 작품')).toBeInTheDocument();
    });

    test('설명이 표시된다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} />);
      expect(screen.getByText(/사과와 꽃병을 그린/)).toBeInTheDocument();
    });

    test('portfolio가 없으면 렌더링되지 않는다', () => {
      render(<PortfolioCard />);
      expect(screen.queryByTestId('portfolio-card')).not.toBeInTheDocument();
    });
  });

  // 이미지 테스트
  describe('이미지 표시', () => {
    test('썸네일 이미지가 표시된다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} />);
      const image = screen.getByTestId('portfolio-thumbnail');
      expect(image).toBeInTheDocument();
      expect(image).toHaveAttribute('src', 'https://example.com/artwork-thumb.jpg');
    });

    test('썸네일이 없으면 원본 이미지가 표시된다', () => {
      const noThumbPortfolio = {
        ...mockPortfolio,
        thumbnailUrl: null,
      };
      render(<PortfolioCard portfolio={noThumbPortfolio} />);
      const image = screen.getByTestId('portfolio-thumbnail');
      expect(image).toHaveAttribute('src', 'https://example.com/artwork.jpg');
    });

    test('이미지에 alt 텍스트가 있다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} />);
      const image = screen.getByTestId('portfolio-thumbnail');
      expect(image).toHaveAttribute('alt', '정물화 작품');
    });

    test('이미지에 lazy loading이 적용된다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} />);
      const image = screen.getByTestId('portfolio-thumbnail');
      expect(image).toHaveAttribute('loading', 'lazy');
    });
  });

  // 태그 테스트
  describe('태그 표시', () => {
    test('태그들이 표시된다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} />);
      expect(screen.getByText('정물화')).toBeInTheDocument();
      expect(screen.getByText('수채화')).toBeInTheDocument();
      expect(screen.getByText('색연필')).toBeInTheDocument();
    });

    test('태그가 없으면 태그 섹션이 표시되지 않는다', () => {
      const noTagsPortfolio = {
        ...mockPortfolio,
        tags: [],
      };
      render(<PortfolioCard portfolio={noTagsPortfolio} />);
      expect(screen.queryByTestId('portfolio-tags')).not.toBeInTheDocument();
    });
  });

  // 공개/비공개 표시 테스트
  describe('공개/비공개 표시', () => {
    test('공개 포트폴리오는 공개 표시가 없다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} isOwner={true} />);
      expect(screen.queryByTestId('private-indicator')).not.toBeInTheDocument();
    });

    test('비공개 포트폴리오는 비공개 표시가 있다', () => {
      render(<PortfolioCard portfolio={privatePortfolio} isOwner={true} />);
      expect(screen.getByTestId('private-indicator')).toBeInTheDocument();
    });

    test('소유자가 아니면 비공개 표시가 없다', () => {
      render(<PortfolioCard portfolio={privatePortfolio} isOwner={false} />);
      expect(screen.queryByTestId('private-indicator')).not.toBeInTheDocument();
    });
  });

  // 소유자 액션 테스트
  describe('소유자 액션', () => {
    test('isOwner가 true이면 편집/삭제 버튼이 표시된다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} isOwner={true} />);
      expect(screen.getByTestId('edit-button')).toBeInTheDocument();
      expect(screen.getByTestId('delete-button')).toBeInTheDocument();
    });

    test('isOwner가 false이면 편집/삭제 버튼이 없다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} isOwner={false} />);
      expect(screen.queryByTestId('edit-button')).not.toBeInTheDocument();
      expect(screen.queryByTestId('delete-button')).not.toBeInTheDocument();
    });

    test('편집 버튼 클릭 시 onEdit이 호출된다', () => {
      const handleEdit = jest.fn();
      render(
        <PortfolioCard
          portfolio={mockPortfolio}
          isOwner={true}
          onEdit={handleEdit}
        />
      );
      fireEvent.click(screen.getByTestId('edit-button'));
      expect(handleEdit).toHaveBeenCalledWith(mockPortfolio);
    });

    test('삭제 버튼 클릭 시 onDelete가 호출된다', () => {
      const handleDelete = jest.fn();
      render(
        <PortfolioCard
          portfolio={mockPortfolio}
          isOwner={true}
          onDelete={handleDelete}
        />
      );
      fireEvent.click(screen.getByTestId('delete-button'));
      expect(handleDelete).toHaveBeenCalledWith(mockPortfolio);
    });
  });

  // 클릭 핸들러 테스트
  describe('클릭 핸들러', () => {
    test('카드 클릭 시 onClick이 호출된다', () => {
      const handleClick = jest.fn();
      render(<PortfolioCard portfolio={mockPortfolio} onClick={handleClick} />);
      const card = screen.getByTestId('portfolio-card');
      fireEvent.click(card);
      expect(handleClick).toHaveBeenCalledWith(mockPortfolio);
    });

    test('onClick이 있으면 카드가 clickable 클래스를 가진다', () => {
      const handleClick = jest.fn();
      render(<PortfolioCard portfolio={mockPortfolio} onClick={handleClick} />);
      const card = screen.getByTestId('portfolio-card');
      expect(card).toHaveClass('clickable');
    });

    test('onClick이 없으면 카드가 clickable 클래스를 가지지 않는다', () => {
      render(<PortfolioCard portfolio={mockPortfolio} />);
      const card = screen.getByTestId('portfolio-card');
      expect(card).not.toHaveClass('clickable');
    });

    test('액션 버튼 클릭 시 카드 클릭이 전파되지 않는다', () => {
      const handleClick = jest.fn();
      const handleEdit = jest.fn();
      render(
        <PortfolioCard
          portfolio={mockPortfolio}
          isOwner={true}
          onClick={handleClick}
          onEdit={handleEdit}
        />
      );
      fireEvent.click(screen.getByTestId('edit-button'));
      expect(handleEdit).toHaveBeenCalled();
      expect(handleClick).not.toHaveBeenCalled();
    });
  });

  // 키보드 접근성 테스트
  describe('키보드 접근성', () => {
    test('onClick이 있으면 Enter 키로 클릭할 수 있다', () => {
      const handleClick = jest.fn();
      render(<PortfolioCard portfolio={mockPortfolio} onClick={handleClick} />);
      const card = screen.getByTestId('portfolio-card');
      fireEvent.keyDown(card, { key: 'Enter' });
      expect(handleClick).toHaveBeenCalledWith(mockPortfolio);
    });

    test('onClick이 있으면 Space 키로 클릭할 수 있다', () => {
      const handleClick = jest.fn();
      render(<PortfolioCard portfolio={mockPortfolio} onClick={handleClick} />);
      const card = screen.getByTestId('portfolio-card');
      fireEvent.keyDown(card, { key: ' ' });
      expect(handleClick).toHaveBeenCalledWith(mockPortfolio);
    });

    test('onClick이 있으면 role=button과 tabIndex=0이 있다', () => {
      const handleClick = jest.fn();
      render(<PortfolioCard portfolio={mockPortfolio} onClick={handleClick} />);
      const card = screen.getByTestId('portfolio-card');
      expect(card).toHaveAttribute('role', 'button');
      expect(card).toHaveAttribute('tabIndex', '0');
    });
  });

  // 설명 truncate 테스트
  describe('설명 truncate', () => {
    test('긴 설명은 truncate된다', () => {
      const longDescPortfolio = {
        ...mockPortfolio,
        description: '아주 긴 설명입니다. '.repeat(20),
      };
      render(<PortfolioCard portfolio={longDescPortfolio} />);
      const description = screen.getByTestId('portfolio-description');
      expect(description).toHaveClass('truncated');
    });

    test('설명이 없으면 설명 섹션이 표시되지 않는다', () => {
      const noDescPortfolio = {
        ...mockPortfolio,
        description: '',
      };
      render(<PortfolioCard portfolio={noDescPortfolio} />);
      expect(screen.queryByTestId('portfolio-description')).not.toBeInTheDocument();
    });
  });
});
