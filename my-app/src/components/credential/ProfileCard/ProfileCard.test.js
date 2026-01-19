// ProfileCard 컴포넌트 테스트
// SPEC-CRED-001: M3 공개 프로필 - 프로필 카드

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import ProfileCard from './ProfileCard';

// 테스트용 프로필 데이터
const mockProfile = {
  uid: 'test-user-123',
  username: 'testuser',
  displayName: 'Test User',
  profileImageUrl: 'https://example.com/avatar.jpg',
  bio: '안녕하세요! 미술을 사랑하는 학생입니다. 열심히 그림을 그리고 있습니다.',
  tier: 'A',
  totalActivities: 156,
  currentStreak: 7,
  longestStreak: 30,
};

// BrowserRouter로 래핑하는 헬퍼 함수
const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

describe('ProfileCard 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('프로필 카드가 렌더링된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} />);
      expect(screen.getByTestId('profile-card')).toBeInTheDocument();
    });

    test('displayName이 표시된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} />);
      expect(screen.getByText('Test User')).toBeInTheDocument();
    });

    test('username이 표시된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} />);
      expect(screen.getByText('@testuser')).toBeInTheDocument();
    });

    test('프로필 이미지가 표시된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} />);
      const avatar = screen.getByTestId('profile-avatar');
      expect(avatar).toBeInTheDocument();
    });
  });

  // TierBadge 통합 테스트
  describe('TierBadge 표시', () => {
    test('티어 배지가 표시된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} />);
      expect(screen.getByTestId('tier-badge')).toBeInTheDocument();
    });
  });

  // Bio 표시 테스트
  describe('Bio 표시', () => {
    test('bio가 표시된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} />);
      expect(screen.getByText(/안녕하세요!/)).toBeInTheDocument();
    });

    test('긴 bio는 truncate된다', () => {
      const longBioProfile = {
        ...mockProfile,
        bio: '아주 긴 자기소개입니다. '.repeat(20),
      };
      renderWithRouter(<ProfileCard profile={longBioProfile} />);
      const bioElement = screen.getByTestId('profile-bio');
      expect(bioElement).toHaveClass('truncated');
    });
  });

  // 활동 통계 테스트
  describe('활동 통계', () => {
    test('showStats가 true이면 통계가 표시된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} showStats={true} />);
      expect(screen.getByTestId('profile-stats')).toBeInTheDocument();
    });

    test('showStats가 false이면 통계가 표시되지 않는다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} showStats={false} />);
      expect(screen.queryByTestId('profile-stats')).not.toBeInTheDocument();
    });

    test('총 활동 수가 표시된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} showStats={true} />);
      expect(screen.getByText('156')).toBeInTheDocument();
    });
  });

  // Compact 모드 테스트
  describe('Compact 모드', () => {
    test('compact 모드에서 축소된 레이아웃이 적용된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} compact={true} />);
      const card = screen.getByTestId('profile-card');
      expect(card).toHaveClass('compact');
    });

    test('compact 모드에서도 기본 정보는 표시된다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} compact={true} />);
      expect(screen.getByText('Test User')).toBeInTheDocument();
      expect(screen.getByText('@testuser')).toBeInTheDocument();
    });
  });

  // onClick 핸들러 테스트
  describe('onClick 핸들러', () => {
    test('카드 클릭 시 onClick이 호출된다', () => {
      const handleClick = jest.fn();
      renderWithRouter(<ProfileCard profile={mockProfile} onClick={handleClick} />);
      const card = screen.getByTestId('profile-card');
      fireEvent.click(card);
      expect(handleClick).toHaveBeenCalledTimes(1);
    });

    test('onClick이 없으면 클릭해도 에러가 발생하지 않는다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} />);
      const card = screen.getByTestId('profile-card');
      expect(() => fireEvent.click(card)).not.toThrow();
    });
  });

  // 프로필 링크 테스트
  describe('프로필 링크', () => {
    test('프로필 링크가 올바른 URL을 가진다', () => {
      renderWithRouter(<ProfileCard profile={mockProfile} />);
      const link = screen.getByTestId('profile-link');
      expect(link).toHaveAttribute('href', '/profile/testuser');
    });
  });

  // 아바타 플레이스홀더 테스트
  describe('아바타 플레이스홀더', () => {
    test('프로필 이미지가 없으면 이니셜이 표시된다', () => {
      const noImageProfile = {
        ...mockProfile,
        profileImageUrl: null,
      };
      renderWithRouter(<ProfileCard profile={noImageProfile} />);
      const placeholder = screen.getByTestId('avatar-placeholder');
      expect(placeholder).toBeInTheDocument();
    });
  });

  // Props 유효성 테스트
  describe('Props 유효성', () => {
    test('profile이 없으면 카드가 렌더링되지 않는다', () => {
      renderWithRouter(<ProfileCard />);
      expect(screen.queryByTestId('profile-card')).not.toBeInTheDocument();
    });
  });
});
