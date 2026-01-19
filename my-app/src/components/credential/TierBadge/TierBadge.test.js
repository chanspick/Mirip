// TierBadge 컴포넌트 테스트
// SPEC-CRED-001: M3 공개 프로필 - 티어 배지

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import TierBadge from './TierBadge';

describe('TierBadge 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('티어 배지가 렌더링된다', () => {
      render(<TierBadge tier="S" />);
      expect(screen.getByTestId('tier-badge')).toBeInTheDocument();
    });

    test('티어 레이블이 표시된다', () => {
      render(<TierBadge tier="A" />);
      expect(screen.getByText('A')).toBeInTheDocument();
    });

    test('showLabel이 false이면 레이블이 표시되지 않는다', () => {
      render(<TierBadge tier="S" showLabel={false} />);
      expect(screen.queryByText('S')).not.toBeInTheDocument();
    });
  });

  // 티어별 색상 테스트
  describe('티어별 색상', () => {
    test('S 티어는 금색 스타일이 적용된다', () => {
      render(<TierBadge tier="S" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('tierS');
    });

    test('A 티어는 은색 스타일이 적용된다', () => {
      render(<TierBadge tier="A" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('tierA');
    });

    test('B 티어는 동색 스타일이 적용된다', () => {
      render(<TierBadge tier="B" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('tierB');
    });

    test('C 티어는 파란색 스타일이 적용된다', () => {
      render(<TierBadge tier="C" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('tierC');
    });

    test('Unranked 티어는 회색 스타일이 적용된다', () => {
      render(<TierBadge tier="Unranked" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('tierUnranked');
    });
  });

  // 크기 변형 테스트
  describe('크기 변형', () => {
    test('기본 크기는 medium이다', () => {
      render(<TierBadge tier="S" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('medium');
    });

    test('small 크기가 적용된다', () => {
      render(<TierBadge tier="S" size="small" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('small');
    });

    test('large 크기가 적용된다', () => {
      render(<TierBadge tier="S" size="large" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('large');
    });
  });

  // 접근성 테스트
  describe('접근성', () => {
    test('티어 설명이 aria-label로 제공된다', () => {
      render(<TierBadge tier="S" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveAttribute('aria-label');
    });

    test('S 티어는 적절한 설명을 갖는다', () => {
      render(<TierBadge tier="S" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge.getAttribute('aria-label')).toContain('S');
    });
  });

  // 유효하지 않은 티어 처리 테스트
  describe('유효하지 않은 티어 처리', () => {
    test('유효하지 않은 티어는 Unranked로 표시된다', () => {
      render(<TierBadge tier="invalid" />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('tierUnranked');
    });

    test('tier prop이 없으면 Unranked로 표시된다', () => {
      render(<TierBadge />);
      const badge = screen.getByTestId('tier-badge');
      expect(badge).toHaveClass('tierUnranked');
    });
  });

  // Unranked 특수 표시 테스트
  describe('Unranked 특수 표시', () => {
    test('Unranked는 "-"로 표시된다', () => {
      render(<TierBadge tier="Unranked" />);
      expect(screen.getByText('-')).toBeInTheDocument();
    });
  });
});
