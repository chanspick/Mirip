// ProfilePage 테스트
// SPEC-CRED-001: M2 마이페이지

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// react-router-dom은 src/__mocks__/react-router-dom.js에서 자동 모킹됨

// 훅 모킹
jest.mock('../../hooks', () => ({
  useUserProfile: jest.fn(),
  useActivityHeatmap: jest.fn(),
  useStreak: jest.fn(),
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
  ActivityHeatmap: ({ userId }) => (
    <div data-testid="mock-activity-heatmap">Heatmap for {userId}</div>
  ),
  ActivityTimeline: ({ userId, limit }) => (
    <div data-testid="mock-activity-timeline">Timeline for {userId}</div>
  ),
  StreakDisplay: ({ userId, totalActivities }) => (
    <div data-testid="mock-streak-display">Streak for {userId}</div>
  ),
}));

// 모킹된 훅 import
import { useUserProfile, useActivityHeatmap, useStreak } from '../../hooks';
import ProfilePage from './ProfilePage';

// 기본 모킹 데이터
const mockProfile = {
  uid: 'test-user-123',
  username: 'testuser',
  displayName: '테스트 사용자',
  bio: '미술을 사랑하는 학생입니다.',
  profileImageUrl: null,
  tier: 'Bronze',
  totalActivities: 156,
  currentStreak: 7,
  longestStreak: 30,
  isPublic: true,
};

describe('ProfilePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // 기본 모킹 설정
    useUserProfile.mockReturnValue({
      profile: mockProfile,
      loading: false,
      error: null,
      updating: false,
      update: jest.fn(),
      refresh: jest.fn(),
    });

    useActivityHeatmap.mockReturnValue({
      heatmapData: null,
      loading: false,
      error: null,
      refresh: jest.fn(),
    });

    useStreak.mockReturnValue({
      currentStreak: 7,
      longestStreak: 30,
      loading: false,
      error: null,
      refresh: jest.fn(),
    });
  });

  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('로딩 상태를 표시한다', () => {
      useUserProfile.mockReturnValue({
        profile: null,
        loading: true,
        error: null,
        updating: false,
        update: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ProfilePage />);
      expect(screen.getByTestId('profile-loading')).toBeInTheDocument();
    });

    test('에러 상태를 표시한다', () => {
      useUserProfile.mockReturnValue({
        profile: null,
        loading: false,
        error: '프로필을 불러올 수 없습니다',
        updating: false,
        update: jest.fn(),
        refresh: jest.fn(),
      });

      render(<ProfilePage />);
      expect(screen.getByText('프로필을 불러올 수 없습니다')).toBeInTheDocument();
    });

    test('프로필 페이지가 올바르게 렌더링된다', () => {
      render(<ProfilePage />);
      expect(screen.getByTestId('profile-page')).toBeInTheDocument();
    });
  });

  // 프로필 정보 표시 테스트
  describe('프로필 정보 표시', () => {
    test('displayName이 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByText('테스트 사용자')).toBeInTheDocument();
    });

    test('username이 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByText('@testuser')).toBeInTheDocument();
    });

    test('bio가 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByText('미술을 사랑하는 학생입니다.')).toBeInTheDocument();
    });

    test('프로필 이미지 또는 기본 아바타가 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByTestId('profile-avatar')).toBeInTheDocument();
    });

    test('프로필 편집 버튼이 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByTestId('edit-profile-button')).toBeInTheDocument();
    });
  });

  // 활동 현황 섹션 테스트
  describe('활동 현황 섹션', () => {
    test('활동 현황 제목이 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByText(/활동 현황/)).toBeInTheDocument();
    });

    test('StreakDisplay 컴포넌트가 렌더링된다', () => {
      render(<ProfilePage />);
      expect(screen.getByTestId('streak-section')).toBeInTheDocument();
    });
  });

  // 잔디밭 섹션 테스트
  describe('잔디밭 섹션', () => {
    test('활동 잔디밭 제목이 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByText(/활동 잔디밭/)).toBeInTheDocument();
    });

    test('ActivityHeatmap 컴포넌트가 렌더링된다', () => {
      render(<ProfilePage />);
      expect(screen.getByTestId('heatmap-section')).toBeInTheDocument();
    });
  });

  // 최근 활동 섹션 테스트
  describe('최근 활동 섹션', () => {
    test('최근 활동 제목이 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByText(/최근 활동/)).toBeInTheDocument();
    });

    test('ActivityTimeline 컴포넌트가 렌더링된다', () => {
      render(<ProfilePage />);
      expect(screen.getByTestId('timeline-section')).toBeInTheDocument();
    });
  });

  // 프로필 편집 테스트
  describe('프로필 편집', () => {
    test('편집 버튼 클릭 시 편집 모드로 전환된다', () => {
      render(<ProfilePage />);

      const editButton = screen.getByTestId('edit-profile-button');
      fireEvent.click(editButton);

      expect(screen.getByTestId('edit-form')).toBeInTheDocument();
    });

    test('편집 취소 버튼이 작동한다', () => {
      render(<ProfilePage />);

      // 편집 모드 진입
      fireEvent.click(screen.getByTestId('edit-profile-button'));
      expect(screen.getByTestId('edit-form')).toBeInTheDocument();

      // 취소 버튼 클릭
      fireEvent.click(screen.getByTestId('cancel-edit-button'));
      expect(screen.queryByTestId('edit-form')).not.toBeInTheDocument();
    });

    test('저장 버튼 클릭 시 update 함수가 호출된다', async () => {
      const updateMock = jest.fn().mockResolvedValue(true);
      useUserProfile.mockReturnValue({
        profile: mockProfile,
        loading: false,
        error: null,
        updating: false,
        update: updateMock,
        refresh: jest.fn(),
      });

      render(<ProfilePage />);

      // 편집 모드 진입
      fireEvent.click(screen.getByTestId('edit-profile-button'));

      // displayName 수정
      const displayNameInput = screen.getByTestId('edit-displayName');
      fireEvent.change(displayNameInput, { target: { value: '새 이름' } });

      // 저장
      fireEvent.click(screen.getByTestId('save-profile-button'));

      await waitFor(() => {
        expect(updateMock).toHaveBeenCalledWith(expect.objectContaining({
          displayName: '새 이름',
        }));
      });
    });
  });

  // 레이아웃 테스트
  describe('레이아웃', () => {
    test('Header가 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByRole('banner')).toBeInTheDocument();
    });

    test('Footer가 표시된다', () => {
      render(<ProfilePage />);
      expect(screen.getByRole('contentinfo')).toBeInTheDocument();
    });
  });
});
