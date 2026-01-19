// PublicProfilePage 테스트
// SPEC-CRED-001: M3 공개 프로필

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// react-router-dom 모킹 (src/__mocks__/react-router-dom.js 자동 사용)
// useParams를 오버라이드하기 위해 별도 변수 사용
const mockUseParams = jest.fn();
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  useParams: () => mockUseParams(),
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/' }),
  Link: ({ children, to, className, ...props }) => (
    <a href={to} className={className} {...props}>{children}</a>
  ),
  BrowserRouter: ({ children }) => <div>{children}</div>,
}));

// 훅 모킹
const mockFetchByUsername = jest.fn();
jest.mock('../../hooks', () => ({
  useUserProfile: () => ({
    profile: null,
    loading: false,
    error: null,
    fetchByUsername: mockFetchByUsername,
  }),
}));

// Firebase Auth 모킹
jest.mock('../../config/firebase', () => ({
  auth: {
    currentUser: null,
  },
}));

// 서비스 모킹
jest.mock('../../services/awardService', () => ({
  getPublicAwards: jest.fn(),
}));

// 공통 컴포넌트 모킹
jest.mock('../../components/common', () => ({
  Header: ({ children, logo }) => (
    <header role="banner" data-testid="mock-header">
      {logo}
      {children}
    </header>
  ),
  Footer: ({ copyright }) => (
    <footer role="contentinfo" data-testid="mock-footer">
      {copyright}
    </footer>
  ),
}));

// 크레덴셜 컴포넌트 모킹
jest.mock('../../components/credential', () => ({
  ActivityHeatmap: ({ userId, readOnly }) => (
    <div data-testid="mock-activity-heatmap" data-readonly={readOnly?.toString()}>
      Heatmap for {userId}
    </div>
  ),
  TierBadge: ({ tier }) => (
    <span data-testid="mock-tier-badge">{tier}</span>
  ),
  ProfileCard: ({ profile }) => (
    <div data-testid="mock-profile-card">{profile?.displayName}</div>
  ),
  AchievementList: ({ awards, loading }) => (
    <div data-testid="mock-achievement-list" data-loading={loading?.toString()}>
      {awards?.length || 0} awards
    </div>
  ),
}));

// 모킹된 서비스 import
import { getPublicAwards } from '../../services/awardService';
import PublicProfilePage from './PublicProfilePage';

// 기본 모킹 데이터
const mockPublicProfile = {
  uid: 'public-user-123',
  username: 'artistuser',
  displayName: '아티스트 사용자',
  bio: '미술 작품을 공유하는 크리에이터입니다.',
  profileImageUrl: 'https://example.com/avatar.jpg',
  tier: 'A',
  totalActivities: 245,
  currentStreak: 14,
  longestStreak: 45,
  isPublic: true,
};

const mockAwards = [
  {
    id: 'award-1',
    competitionId: 'comp-1',
    competitionTitle: '2024 전국 미술대전',
    rank: '금상',
    awardedAt: { toDate: () => new Date('2024-06-15') },
  },
  {
    id: 'award-2',
    competitionId: 'comp-2',
    competitionTitle: '청소년 디자인 공모전',
    rank: '은상',
    awardedAt: { toDate: () => new Date('2024-03-20') },
  },
];

describe('PublicProfilePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockUseParams.mockReturnValue({ username: 'artistuser' });
    getPublicAwards.mockResolvedValue(mockAwards);
  });

  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('로딩 상태를 표시한다', async () => {
      mockFetchByUsername.mockImplementation(() => new Promise(() => {})); // 무한 대기

      render(<PublicProfilePage />);
      expect(screen.getByTestId('public-profile-loading')).toBeInTheDocument();
    });

    test('프로필 데이터 로드 후 페이지가 렌더링된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByTestId('public-profile-page')).toBeInTheDocument();
      });
    });

    test('존재하지 않는 사용자에 대해 404를 표시한다', async () => {
      mockFetchByUsername.mockResolvedValue(null);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByTestId('profile-not-found')).toBeInTheDocument();
      });
    });

    test('비공개 프로필에 대해 비공개 메시지를 표시한다', async () => {
      mockFetchByUsername.mockResolvedValue({
        ...mockPublicProfile,
        isPublic: false,
      });

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByTestId('profile-private')).toBeInTheDocument();
      });
    });
  });

  // 프로필 정보 표시 테스트
  describe('프로필 정보 표시', () => {
    test('displayName이 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByText('아티스트 사용자')).toBeInTheDocument();
      });
    });

    test('username이 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByText('@artistuser')).toBeInTheDocument();
      });
    });

    test('TierBadge가 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByTestId('mock-tier-badge')).toBeInTheDocument();
      });
    });

    test('bio가 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByText(/미술 작품을 공유하는 크리에이터/)).toBeInTheDocument();
      });
    });

    test('프로필 아바타가 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByTestId('public-profile-avatar')).toBeInTheDocument();
      });
    });
  });

  // 활동 통계 테스트
  describe('활동 통계', () => {
    test('총 활동 수가 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByText('245')).toBeInTheDocument();
      });
    });

    test('연속 활동 일수가 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByText('14')).toBeInTheDocument();
      });
    });
  });

  // 잔디밭 섹션 테스트
  describe('잔디밭 섹션', () => {
    test('ActivityHeatmap이 읽기 전용으로 렌더링된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        const heatmap = screen.getByTestId('mock-activity-heatmap');
        expect(heatmap).toBeInTheDocument();
        expect(heatmap).toHaveAttribute('data-readonly', 'true');
      });
    });
  });

  // 수상 내역 섹션 테스트
  describe('수상 내역 섹션', () => {
    test('AchievementList가 렌더링된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByTestId('mock-achievement-list')).toBeInTheDocument();
      });
    });

    test('수상 내역이 올바르게 전달된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(getPublicAwards).toHaveBeenCalledWith('public-user-123');
      });
    });
  });

  // 공유 기능 테스트
  describe('공유 기능', () => {
    test('공유 버튼이 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByTestId('share-button')).toBeInTheDocument();
      });
    });
  });

  // 레이아웃 테스트
  describe('레이아웃', () => {
    test('Header가 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByRole('banner')).toBeInTheDocument();
      });
    });

    test('Footer가 표시된다', async () => {
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByRole('contentinfo')).toBeInTheDocument();
      });
    });
  });

  // URL 파라미터 테스트
  describe('URL 파라미터', () => {
    test('username 파라미터로 프로필을 조회한다', async () => {
      mockUseParams.mockReturnValue({ username: 'testartist' });
      mockFetchByUsername.mockResolvedValue(mockPublicProfile);

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(mockFetchByUsername).toHaveBeenCalledWith('testartist');
      });
    });
  });

  // 에러 처리 테스트
  describe('에러 처리', () => {
    test('네트워크 에러 시 에러 메시지를 표시한다', async () => {
      mockFetchByUsername.mockRejectedValue(new Error('네트워크 오류'));

      render(<PublicProfilePage />);

      await waitFor(() => {
        expect(screen.getByTestId('profile-error')).toBeInTheDocument();
      });
    });
  });
});
