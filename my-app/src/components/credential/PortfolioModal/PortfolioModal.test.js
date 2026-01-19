// PortfolioModal 컴포넌트 테스트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 상세 모달
// TDD RED Phase: 실패하는 테스트 먼저 작성

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import PortfolioModal from './PortfolioModal';

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

// 여러 포트폴리오 (네비게이션 테스트용)
const mockPortfolios = [
  mockPortfolio,
  {
    ...mockPortfolio,
    id: 'portfolio-002',
    title: '풍경화 작품',
    imageUrl: 'https://example.com/artwork2.jpg',
  },
  {
    ...mockPortfolio,
    id: 'portfolio-003',
    title: '인물화 작품',
    imageUrl: 'https://example.com/artwork3.jpg',
  },
];

describe('PortfolioModal 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('isOpen이 true이면 모달이 렌더링된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(screen.getByTestId('portfolio-modal')).toBeInTheDocument();
    });

    test('isOpen이 false이면 모달이 렌더링되지 않는다', () => {
      render(
        <PortfolioModal
          isOpen={false}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(screen.queryByTestId('portfolio-modal')).not.toBeInTheDocument();
    });

    test('portfolio가 없으면 모달이 렌더링되지 않는다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={null}
          onClose={jest.fn()}
        />
      );
      expect(screen.queryByTestId('portfolio-modal')).not.toBeInTheDocument();
    });
  });

  // 이미지 표시 테스트
  describe('이미지 표시', () => {
    test('원본 이미지가 전체 크기로 표시된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      const image = screen.getByTestId('modal-image');
      expect(image).toHaveAttribute('src', 'https://example.com/artwork.jpg');
    });

    test('이미지에 alt 텍스트가 있다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      const image = screen.getByTestId('modal-image');
      expect(image).toHaveAttribute('alt', '정물화 작품');
    });
  });

  // 포트폴리오 정보 표시 테스트
  describe('포트폴리오 정보 표시', () => {
    test('제목이 표시된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(screen.getByText('정물화 작품')).toBeInTheDocument();
    });

    test('설명이 표시된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(screen.getByText(/사과와 꽃병을 그린/)).toBeInTheDocument();
    });

    test('태그가 표시된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(screen.getByText('정물화')).toBeInTheDocument();
      expect(screen.getByText('수채화')).toBeInTheDocument();
      expect(screen.getByText('색연필')).toBeInTheDocument();
    });

    test('설명이 없으면 설명 섹션이 표시되지 않는다', () => {
      const noDescPortfolio = { ...mockPortfolio, description: '' };
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={noDescPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(screen.queryByTestId('modal-description')).not.toBeInTheDocument();
    });
  });

  // 닫기 버튼 테스트
  describe('닫기 버튼', () => {
    test('닫기 버튼이 표시된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(screen.getByTestId('close-button')).toBeInTheDocument();
    });

    test('닫기 버튼 클릭 시 onClose가 호출된다', () => {
      const handleClose = jest.fn();
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={handleClose}
        />
      );
      fireEvent.click(screen.getByTestId('close-button'));
      expect(handleClose).toHaveBeenCalledTimes(1);
    });
  });

  // 오버레이 클릭 테스트
  describe('오버레이 클릭', () => {
    test('오버레이 클릭 시 onClose가 호출된다', () => {
      const handleClose = jest.fn();
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={handleClose}
        />
      );
      fireEvent.click(screen.getByTestId('modal-overlay'));
      expect(handleClose).toHaveBeenCalledTimes(1);
    });

    test('모달 내부 클릭 시 onClose가 호출되지 않는다', () => {
      const handleClose = jest.fn();
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={handleClose}
        />
      );
      fireEvent.click(screen.getByTestId('modal-content'));
      expect(handleClose).not.toHaveBeenCalled();
    });
  });

  // 키보드 네비게이션 테스트
  describe('키보드 네비게이션', () => {
    test('ESC 키를 누르면 onClose가 호출된다', () => {
      const handleClose = jest.fn();
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={handleClose}
        />
      );
      fireEvent.keyDown(document, { key: 'Escape' });
      expect(handleClose).toHaveBeenCalledTimes(1);
    });

    test('왼쪽 화살표 키를 누르면 onPrev가 호출된다', () => {
      const handlePrev = jest.fn();
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolios[1]}
          portfolios={mockPortfolios}
          currentIndex={1}
          onClose={jest.fn()}
          onPrev={handlePrev}
        />
      );
      fireEvent.keyDown(document, { key: 'ArrowLeft' });
      expect(handlePrev).toHaveBeenCalledTimes(1);
    });

    test('오른쪽 화살표 키를 누르면 onNext가 호출된다', () => {
      const handleNext = jest.fn();
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolios[1]}
          portfolios={mockPortfolios}
          currentIndex={1}
          onClose={jest.fn()}
          onNext={handleNext}
        />
      );
      fireEvent.keyDown(document, { key: 'ArrowRight' });
      expect(handleNext).toHaveBeenCalledTimes(1);
    });
  });

  // 네비게이션 버튼 테스트
  describe('네비게이션 버튼', () => {
    test('portfolios가 있으면 네비게이션 버튼이 표시된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolios[1]}
          portfolios={mockPortfolios}
          currentIndex={1}
          onClose={jest.fn()}
          onPrev={jest.fn()}
          onNext={jest.fn()}
        />
      );
      expect(screen.getByTestId('prev-button')).toBeInTheDocument();
      expect(screen.getByTestId('next-button')).toBeInTheDocument();
    });

    test('첫 번째 항목에서 이전 버튼이 비활성화된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolios[0]}
          portfolios={mockPortfolios}
          currentIndex={0}
          onClose={jest.fn()}
          onPrev={jest.fn()}
          onNext={jest.fn()}
        />
      );
      expect(screen.getByTestId('prev-button')).toBeDisabled();
    });

    test('마지막 항목에서 다음 버튼이 비활성화된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolios[2]}
          portfolios={mockPortfolios}
          currentIndex={2}
          onClose={jest.fn()}
          onPrev={jest.fn()}
          onNext={jest.fn()}
        />
      );
      expect(screen.getByTestId('next-button')).toBeDisabled();
    });

    test('이전 버튼 클릭 시 onPrev가 호출된다', () => {
      const handlePrev = jest.fn();
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolios[1]}
          portfolios={mockPortfolios}
          currentIndex={1}
          onClose={jest.fn()}
          onPrev={handlePrev}
          onNext={jest.fn()}
        />
      );
      fireEvent.click(screen.getByTestId('prev-button'));
      expect(handlePrev).toHaveBeenCalledTimes(1);
    });

    test('다음 버튼 클릭 시 onNext가 호출된다', () => {
      const handleNext = jest.fn();
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolios[1]}
          portfolios={mockPortfolios}
          currentIndex={1}
          onClose={jest.fn()}
          onPrev={jest.fn()}
          onNext={handleNext}
        />
      );
      fireEvent.click(screen.getByTestId('next-button'));
      expect(handleNext).toHaveBeenCalledTimes(1);
    });

    test('portfolios가 없으면 네비게이션 버튼이 없다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(screen.queryByTestId('prev-button')).not.toBeInTheDocument();
      expect(screen.queryByTestId('next-button')).not.toBeInTheDocument();
    });
  });

  // 인덱스 표시 테스트
  describe('인덱스 표시', () => {
    test('현재 인덱스와 전체 개수가 표시된다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolios[1]}
          portfolios={mockPortfolios}
          currentIndex={1}
          onClose={jest.fn()}
        />
      );
      expect(screen.getByText('2 / 3')).toBeInTheDocument();
    });
  });

  // 바디 스크롤 잠금 테스트
  describe('바디 스크롤 잠금', () => {
    test('모달이 열리면 바디 스크롤이 잠긴다', () => {
      render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(document.body.style.overflow).toBe('hidden');
    });

    test('모달이 닫히면 바디 스크롤이 복원된다', () => {
      const { rerender } = render(
        <PortfolioModal
          isOpen={true}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      rerender(
        <PortfolioModal
          isOpen={false}
          portfolio={mockPortfolio}
          onClose={jest.fn()}
        />
      );
      expect(document.body.style.overflow).toBe('');
    });
  });
});
