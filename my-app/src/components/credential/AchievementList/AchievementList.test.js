// AchievementList 컴포넌트 테스트
// SPEC-CRED-001: M3 공개 프로필 - 수상 내역 목록

import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import '@testing-library/jest-dom';
import AchievementList from './AchievementList';

// 테스트용 수상 데이터
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
  {
    id: 'award-3',
    competitionId: 'comp-3',
    competitionTitle: '지역 예술 경진대회',
    rank: '대상',
    awardedAt: { toDate: () => new Date('2023-12-10') },
  },
];

// BrowserRouter로 래핑하는 헬퍼 함수
const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

describe('AchievementList 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('수상 목록이 렌더링된다', () => {
      renderWithRouter(<AchievementList awards={mockAwards} />);
      expect(screen.getByTestId('achievement-list')).toBeInTheDocument();
    });

    test('수상 항목이 올바르게 표시된다', () => {
      renderWithRouter(<AchievementList awards={mockAwards} />);
      expect(screen.getByText('2024 전국 미술대전')).toBeInTheDocument();
      expect(screen.getByText('청소년 디자인 공모전')).toBeInTheDocument();
      expect(screen.getByText('지역 예술 경진대회')).toBeInTheDocument();
    });

    test('수상 등급이 표시된다', () => {
      renderWithRouter(<AchievementList awards={mockAwards} />);
      expect(screen.getByText('금상')).toBeInTheDocument();
      expect(screen.getByText('은상')).toBeInTheDocument();
      expect(screen.getByText('대상')).toBeInTheDocument();
    });
  });

  // 수상 등급 아이콘 테스트
  describe('수상 등급 아이콘', () => {
    test('대상에는 트로피 아이콘이 표시된다', () => {
      const grandPrizeAward = [{
        ...mockAwards[2],
        rank: '대상',
      }];
      renderWithRouter(<AchievementList awards={grandPrizeAward} />);
      const item = screen.getByTestId('achievement-item-award-3');
      expect(item).toHaveClass('rank대상');
    });

    test('금상에는 금색 스타일이 적용된다', () => {
      const goldAward = [{
        ...mockAwards[0],
        rank: '금상',
      }];
      renderWithRouter(<AchievementList awards={goldAward} />);
      const item = screen.getByTestId('achievement-item-award-1');
      expect(item).toHaveClass('rank금상');
    });
  });

  // 날짜 표시 테스트
  describe('날짜 표시', () => {
    test('수상일이 포맷되어 표시된다', () => {
      renderWithRouter(<AchievementList awards={mockAwards} />);
      expect(screen.getByText(/2024\.06\.15/)).toBeInTheDocument();
    });
  });

  // 공모전 링크 테스트
  describe('공모전 링크', () => {
    test('공모전 제목이 링크로 표시된다', () => {
      renderWithRouter(<AchievementList awards={mockAwards} />);
      const link = screen.getByTestId('competition-link-comp-1');
      expect(link).toHaveAttribute('href', '/competitions/comp-1');
    });
  });

  // 빈 상태 테스트
  describe('빈 상태', () => {
    test('수상 내역이 없으면 빈 메시지가 표시된다', () => {
      renderWithRouter(<AchievementList awards={[]} />);
      expect(screen.getByTestId('empty-achievements')).toBeInTheDocument();
    });

    test('awards가 undefined이면 빈 메시지가 표시된다', () => {
      renderWithRouter(<AchievementList />);
      expect(screen.getByTestId('empty-achievements')).toBeInTheDocument();
    });
  });

  // 로딩 상태 테스트
  describe('로딩 상태', () => {
    test('loading이 true이면 로딩 UI가 표시된다', () => {
      renderWithRouter(<AchievementList awards={[]} loading={true} />);
      expect(screen.getByTestId('achievement-loading')).toBeInTheDocument();
    });
  });

  // Compact 모드 테스트
  describe('Compact 모드', () => {
    test('compact 모드에서 축소된 레이아웃이 적용된다', () => {
      renderWithRouter(<AchievementList awards={mockAwards} compact={true} />);
      const list = screen.getByTestId('achievement-list');
      expect(list).toHaveClass('compact');
    });
  });

  // maxItems 테스트
  describe('maxItems', () => {
    test('maxItems가 지정되면 해당 수만큼만 표시된다', () => {
      renderWithRouter(<AchievementList awards={mockAwards} maxItems={2} />);
      const items = screen.getAllByTestId(/^achievement-item-/);
      expect(items).toHaveLength(2);
    });

    test('더보기 버튼이 표시된다', () => {
      renderWithRouter(<AchievementList awards={mockAwards} maxItems={2} />);
      expect(screen.getByTestId('show-more-button')).toBeInTheDocument();
    });
  });
});
