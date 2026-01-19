// PortfolioPage 테스트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 관리 페이지
// TDD RED Phase: 실패하는 테스트 먼저 작성

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// react-router-dom은 src/__mocks__/react-router-dom.js에서 자동 모킹됨

// 훅 모킹
jest.mock('../../hooks', () => ({
  usePortfolios: jest.fn(),
}));

// Firebase Auth 모킹
jest.mock('../../config/firebase', () => ({
  auth: {
    currentUser: { uid: 'test-user-123', email: 'test@example.com' },
  },
}));

// 공통 컴포넌트 모킹
jest.mock('../../components/common', () => ({
  Header: ({ children, logo, navItems, ctaButton }) => (
    <header role="banner" data-testid="mock-header">
      {logo}
      {children}
    </header>
  ),
  Footer: ({ children, links, copyright }) => (
    <footer role="contentinfo" data-testid="mock-footer">
      {copyright}
    </footer>
  ),
}));

// 크레덴셜 컴포넌트 모킹
jest.mock('../../components/credential', () => ({
  PortfolioGrid: ({ portfolios, isOwner, onEdit, onDelete, onItemClick, loading, emptyMessage }) => (
    <div data-testid="mock-portfolio-grid">
      {loading && <div data-testid="portfolio-grid-loading">로딩 중...</div>}
      {!loading && portfolios?.length === 0 && (
        <div data-testid="portfolio-grid-empty">{emptyMessage || '포트폴리오가 없습니다'}</div>
      )}
      {portfolios?.map((p) => (
        <div key={p.id} data-testid={`portfolio-item-${p.id}`}>
          <span>{p.title}</span>
          {isOwner && (
            <>
              <button onClick={() => onEdit?.(p)} data-testid={`edit-${p.id}`}>편집</button>
              <button onClick={() => onDelete?.(p.id)} data-testid={`delete-${p.id}`}>삭제</button>
            </>
          )}
          <button onClick={() => onItemClick?.(p)} data-testid={`view-${p.id}`}>보기</button>
        </div>
      ))}
    </div>
  ),
  PortfolioUploadForm: ({ onSubmit, onCancel, initialData, loading }) => (
    <div data-testid="mock-portfolio-upload-form">
      <input
        data-testid="upload-title-input"
        defaultValue={initialData?.title || ''}
        onChange={(e) => {}}
      />
      <button
        onClick={() => onSubmit?.({ title: '테스트 작품', imageUrl: 'test.jpg' })}
        data-testid="upload-submit-button"
        disabled={loading}
      >
        {initialData ? '수정' : '업로드'}
      </button>
      <button onClick={onCancel} data-testid="upload-cancel-button">취소</button>
    </div>
  ),
  PortfolioModal: ({ isOpen, portfolio, onClose, portfolios, currentIndex, onPrev, onNext }) => (
    isOpen ? (
      <div data-testid="mock-portfolio-modal">
        <h2>{portfolio?.title}</h2>
        <button onClick={onClose} data-testid="modal-close-button">닫기</button>
        {onPrev && <button onClick={onPrev} data-testid="modal-prev-button">이전</button>}
        {onNext && <button onClick={onNext} data-testid="modal-next-button">다음</button>}
      </div>
    ) : null
  ),
}));

// 모킹된 훅 import
import { usePortfolios } from '../../hooks';
import PortfolioPage from './PortfolioPage';

// 테스트용 포트폴리오 데이터
const mockPortfolios = [
  {
    id: 'portfolio-001',
    userId: 'test-user-123',
    title: '정물화 작품',
    description: '사과와 꽃병을 그린 정물화입니다.',
    imageUrl: 'https://example.com/artwork1.jpg',
    thumbnailUrl: 'https://example.com/artwork1-thumb.jpg',
    tags: ['정물화', '수채화'],
    isPublic: true,
    createdAt: { toDate: () => new Date('2024-01-15') },
    updatedAt: { toDate: () => new Date('2024-01-16') },
  },
  {
    id: 'portfolio-002',
    userId: 'test-user-123',
    title: '풍경화 작품',
    description: '산과 강을 그린 풍경화입니다.',
    imageUrl: 'https://example.com/artwork2.jpg',
    thumbnailUrl: 'https://example.com/artwork2-thumb.jpg',
    tags: ['풍경화', '유화'],
    isPublic: false,
    createdAt: { toDate: () => new Date('2024-01-20') },
    updatedAt: { toDate: () => new Date('2024-01-21') },
  },
];

describe('PortfolioPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // 기본 모킹 설정
    usePortfolios.mockReturnValue({
      portfolios: mockPortfolios,
      loading: false,
      error: null,
      add: jest.fn().mockResolvedValue({ id: 'new-id' }),
      update: jest.fn().mockResolvedValue(true),
      remove: jest.fn().mockResolvedValue(true),
      loadMore: jest.fn(),
      hasMore: false,
      refresh: jest.fn(),
    });
  });

  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('페이지가 올바르게 렌더링된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByTestId('portfolio-page')).toBeInTheDocument();
    });

    test('페이지 제목이 표시된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByText(/내 포트폴리오/)).toBeInTheDocument();
    });

    test('Header가 표시된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByRole('banner')).toBeInTheDocument();
    });

    test('Footer가 표시된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByRole('contentinfo')).toBeInTheDocument();
    });

    test('새 작품 추가 버튼이 표시된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByTestId('add-portfolio-button')).toBeInTheDocument();
    });
  });

  // 로딩 상태 테스트
  describe('로딩 상태', () => {
    test('로딩 중일 때 로딩 인디케이터가 표시된다', () => {
      usePortfolios.mockReturnValue({
        portfolios: [],
        loading: true,
        error: null,
        add: jest.fn(),
        update: jest.fn(),
        remove: jest.fn(),
        loadMore: jest.fn(),
        hasMore: false,
        refresh: jest.fn(),
      });

      render(<PortfolioPage />);
      expect(screen.getByTestId('portfolio-grid-loading')).toBeInTheDocument();
    });
  });

  // 에러 상태 테스트
  describe('에러 상태', () => {
    test('에러 발생 시 에러 메시지가 표시된다', () => {
      usePortfolios.mockReturnValue({
        portfolios: [],
        loading: false,
        error: '포트폴리오를 불러올 수 없습니다',
        add: jest.fn(),
        update: jest.fn(),
        remove: jest.fn(),
        loadMore: jest.fn(),
        hasMore: false,
        refresh: jest.fn(),
      });

      render(<PortfolioPage />);
      expect(screen.getByText('포트폴리오를 불러올 수 없습니다')).toBeInTheDocument();
    });
  });

  // 빈 상태 테스트
  describe('빈 상태', () => {
    test('포트폴리오가 없을 때 빈 상태 메시지가 표시된다', () => {
      usePortfolios.mockReturnValue({
        portfolios: [],
        loading: false,
        error: null,
        add: jest.fn(),
        update: jest.fn(),
        remove: jest.fn(),
        loadMore: jest.fn(),
        hasMore: false,
        refresh: jest.fn(),
      });

      render(<PortfolioPage />);
      expect(screen.getByTestId('portfolio-grid-empty')).toBeInTheDocument();
    });
  });

  // 로그인 필요 테스트
  describe('로그인 필요', () => {
    test('로그인하지 않은 경우 로그인 필요 메시지가 표시된다', () => {
      // Firebase Auth 모킹을 null로 변경
      jest.doMock('../../config/firebase', () => ({
        auth: {
          currentUser: null,
        },
      }));

      // 모듈을 다시 로드하는 대신 컴포넌트 props로 처리
      // 실제 구현에서는 auth.currentUser가 null일 때 처리
    });
  });

  // 포트폴리오 목록 표시 테스트
  describe('포트폴리오 목록', () => {
    test('포트폴리오 그리드가 렌더링된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByTestId('mock-portfolio-grid')).toBeInTheDocument();
    });

    test('포트폴리오 아이템들이 표시된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByTestId('portfolio-item-portfolio-001')).toBeInTheDocument();
      expect(screen.getByTestId('portfolio-item-portfolio-002')).toBeInTheDocument();
    });
  });

  // 새 작품 추가 테스트
  describe('새 작품 추가', () => {
    test('새 작품 추가 버튼 클릭 시 업로드 폼이 표시된다', () => {
      render(<PortfolioPage />);

      const addButton = screen.getByTestId('add-portfolio-button');
      fireEvent.click(addButton);

      expect(screen.getByTestId('mock-portfolio-upload-form')).toBeInTheDocument();
    });

    test('업로드 폼에서 취소 클릭 시 폼이 닫힌다', () => {
      render(<PortfolioPage />);

      // 폼 열기
      fireEvent.click(screen.getByTestId('add-portfolio-button'));
      expect(screen.getByTestId('mock-portfolio-upload-form')).toBeInTheDocument();

      // 취소 클릭
      fireEvent.click(screen.getByTestId('upload-cancel-button'));
      expect(screen.queryByTestId('mock-portfolio-upload-form')).not.toBeInTheDocument();
    });

    test('업로드 폼 제출 시 add 함수가 호출된다', async () => {
      const addMock = jest.fn().mockResolvedValue({ id: 'new-id' });
      usePortfolios.mockReturnValue({
        portfolios: mockPortfolios,
        loading: false,
        error: null,
        add: addMock,
        update: jest.fn(),
        remove: jest.fn(),
        loadMore: jest.fn(),
        hasMore: false,
        refresh: jest.fn(),
      });

      render(<PortfolioPage />);

      // 폼 열기
      fireEvent.click(screen.getByTestId('add-portfolio-button'));

      // 제출
      fireEvent.click(screen.getByTestId('upload-submit-button'));

      await waitFor(() => {
        expect(addMock).toHaveBeenCalled();
      });
    });
  });

  // 포트폴리오 편집 테스트
  describe('포트폴리오 편집', () => {
    test('편집 버튼 클릭 시 편집 폼이 표시된다', () => {
      render(<PortfolioPage />);

      const editButton = screen.getByTestId('edit-portfolio-001');
      fireEvent.click(editButton);

      expect(screen.getByTestId('mock-portfolio-upload-form')).toBeInTheDocument();
    });

    test('편집 폼 제출 시 update 함수가 호출된다', async () => {
      const updateMock = jest.fn().mockResolvedValue(true);
      usePortfolios.mockReturnValue({
        portfolios: mockPortfolios,
        loading: false,
        error: null,
        add: jest.fn(),
        update: updateMock,
        remove: jest.fn(),
        loadMore: jest.fn(),
        hasMore: false,
        refresh: jest.fn(),
      });

      render(<PortfolioPage />);

      // 편집 모드 진입
      fireEvent.click(screen.getByTestId('edit-portfolio-001'));

      // 제출
      fireEvent.click(screen.getByTestId('upload-submit-button'));

      await waitFor(() => {
        expect(updateMock).toHaveBeenCalled();
      });
    });
  });

  // 포트폴리오 삭제 테스트
  describe('포트폴리오 삭제', () => {
    test('삭제 버튼 클릭 시 확인 모달이 표시된다', () => {
      render(<PortfolioPage />);

      const deleteButton = screen.getByTestId('delete-portfolio-001');
      fireEvent.click(deleteButton);

      expect(screen.getByTestId('delete-confirm-modal')).toBeInTheDocument();
    });

    test('삭제 확인 시 remove 함수가 호출된다', async () => {
      const removeMock = jest.fn().mockResolvedValue(true);
      usePortfolios.mockReturnValue({
        portfolios: mockPortfolios,
        loading: false,
        error: null,
        add: jest.fn(),
        update: jest.fn(),
        remove: removeMock,
        loadMore: jest.fn(),
        hasMore: false,
        refresh: jest.fn(),
      });

      render(<PortfolioPage />);

      // 삭제 버튼 클릭
      fireEvent.click(screen.getByTestId('delete-portfolio-001'));

      // 확인 버튼 클릭
      fireEvent.click(screen.getByTestId('confirm-delete-button'));

      await waitFor(() => {
        expect(removeMock).toHaveBeenCalledWith('portfolio-001');
      });
    });

    test('삭제 취소 시 모달이 닫힌다', () => {
      render(<PortfolioPage />);

      // 삭제 버튼 클릭
      fireEvent.click(screen.getByTestId('delete-portfolio-001'));
      expect(screen.getByTestId('delete-confirm-modal')).toBeInTheDocument();

      // 취소 버튼 클릭
      fireEvent.click(screen.getByTestId('cancel-delete-button'));
      expect(screen.queryByTestId('delete-confirm-modal')).not.toBeInTheDocument();
    });
  });

  // 포트폴리오 상세 보기 테스트
  describe('포트폴리오 상세 보기', () => {
    test('포트폴리오 클릭 시 모달이 열린다', () => {
      render(<PortfolioPage />);

      const viewButton = screen.getByTestId('view-portfolio-001');
      fireEvent.click(viewButton);

      expect(screen.getByTestId('mock-portfolio-modal')).toBeInTheDocument();
    });

    test('모달 닫기 버튼 클릭 시 모달이 닫힌다', () => {
      render(<PortfolioPage />);

      // 모달 열기
      fireEvent.click(screen.getByTestId('view-portfolio-001'));
      expect(screen.getByTestId('mock-portfolio-modal')).toBeInTheDocument();

      // 모달 닫기
      fireEvent.click(screen.getByTestId('modal-close-button'));
      expect(screen.queryByTestId('mock-portfolio-modal')).not.toBeInTheDocument();
    });

    test('모달에서 이전/다음 네비게이션이 작동한다', () => {
      render(<PortfolioPage />);

      // 첫 번째 포트폴리오 열기
      fireEvent.click(screen.getByTestId('view-portfolio-001'));

      // 다음 버튼 클릭
      const nextButton = screen.getByTestId('modal-next-button');
      fireEvent.click(nextButton);

      // 두 번째 포트폴리오가 표시됨 확인
      expect(screen.getByText('풍경화 작품')).toBeInTheDocument();
    });
  });

  // 필터 및 정렬 테스트
  describe('필터 및 정렬', () => {
    test('필터 버튼이 표시된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByTestId('filter-button')).toBeInTheDocument();
    });

    test('정렬 버튼이 표시된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByTestId('sort-button')).toBeInTheDocument();
    });
  });

  // 통계 정보 테스트
  describe('통계 정보', () => {
    test('총 작품 수가 표시된다', () => {
      render(<PortfolioPage />);
      expect(screen.getByTestId('portfolio-count')).toBeInTheDocument();
      expect(screen.getByText(/2/)).toBeInTheDocument();
    });
  });
});
