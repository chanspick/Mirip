// Landing 컴포넌트 테스트
// SPEC-FIREBASE-001: Landing Page TDD Tests
// GREEN Phase: 테스트 통과를 위한 구현 검증

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import Landing from './Landing';

// Mock registrationService
jest.mock('../../services/registrationService', () => ({
  create: jest.fn(),
}));

// Mock scrollIntoView
const mockScrollIntoView = jest.fn();
window.HTMLElement.prototype.scrollIntoView = mockScrollIntoView;

// Mock window.scrollY
Object.defineProperty(window, 'scrollY', {
  writable: true,
  value: 0,
});

describe('Landing', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockScrollIntoView.mockClear();
    window.scrollY = 0;
  });

  describe('Hero Section', () => {
    it('Hero 섹션이 렌더링된다', () => {
      render(<Landing />);

      // 메인 타이틀 확인
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
        /당신의 작품/
      );
      expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(
        /어디까지/
      );
    });

    it('Hero 서브타이틀이 렌더링된다', () => {
      render(<Landing />);

      expect(
        screen.getByText(/대학별 합격 데이터를 학습한 AI가/)
      ).toBeInTheDocument();
      expect(screen.getByText(/객관적으로 진단합니다/)).toBeInTheDocument();
    });

    it('Hero CTA 버튼이 렌더링된다', () => {
      render(<Landing />);

      const heroSection = screen.getByTestId('hero-section');
      const ctaButton = within(heroSection).getByRole('button', {
        name: /사전등록/,
      });
      expect(ctaButton).toBeInTheDocument();
    });

    it('Hero CTA 버튼 클릭 시 CTA 섹션으로 스크롤된다', () => {
      render(<Landing />);

      const heroSection = screen.getByTestId('hero-section');
      const ctaButton = within(heroSection).getByRole('button', {
        name: /사전등록/,
      });

      fireEvent.click(ctaButton);

      expect(mockScrollIntoView).toHaveBeenCalledWith({
        behavior: 'smooth',
        block: 'start',
      });
    });
  });

  describe('Problem Section', () => {
    it('Problem 섹션 라벨과 타이틀이 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByText('The Problem')).toBeInTheDocument();
      expect(
        screen.getByRole('heading', { name: /미술 입시, 막연한 불안/ })
      ).toBeInTheDocument();
    });

    it('3개의 Problem 카드가 렌더링된다', () => {
      render(<Landing />);

      const problemSection = screen.getByTestId('problem-section');

      // 3개의 번호
      expect(within(problemSection).getByText('01')).toBeInTheDocument();
      expect(within(problemSection).getByText('02')).toBeInTheDocument();
      expect(within(problemSection).getByText('03')).toBeInTheDocument();

      // 내용 확인
      expect(
        within(problemSection).getByText(/월 80~150만원/)
      ).toBeInTheDocument();
      expect(
        within(problemSection).getByText(/지금 내 실력이 어느 정도인지/)
      ).toBeInTheDocument();
      expect(
        within(problemSection).getByText(/수상 이력과 포트폴리오/)
      ).toBeInTheDocument();
    });
  });

  describe('Solution Section', () => {
    it('Solution 섹션 라벨과 타이틀이 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByText('The Solution')).toBeInTheDocument();
      expect(
        screen.getByRole('heading', { name: /MIRIP이 연결합니다/ })
      ).toBeInTheDocument();
    });

    it('4단계 타임라인이 렌더링된다', () => {
      render(<Landing />);

      const solutionSection = screen.getByTestId('solution-section');

      // 4개의 스텝 번호
      const stepNumbers = within(solutionSection).getAllByText(/^0[1-4]$/);
      expect(stepNumbers).toHaveLength(4);

      // 타임라인 제목들
      expect(within(solutionSection).getByText('공모전')).toBeInTheDocument();
      expect(
        within(solutionSection).getByText('크레덴셜')
      ).toBeInTheDocument();
      expect(within(solutionSection).getByText('AI 진단')).toBeInTheDocument();
      expect(within(solutionSection).getByText('커리어')).toBeInTheDocument();
    });

    it('타임라인 설명 텍스트가 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByText('참여와 경쟁의 시작')).toBeInTheDocument();
      expect(screen.getByText('이력이 쌓이는 프로필')).toBeInTheDocument();
      expect(screen.getByText('대학별 합격 예측')).toBeInTheDocument();
      expect(screen.getByText('채용과 거래로 확장')).toBeInTheDocument();
    });
  });

  describe('AI Preview Section', () => {
    it('AI Preview 섹션 라벨과 타이틀이 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByText('AI Diagnosis')).toBeInTheDocument();
      expect(
        screen.getByRole('heading', { name: /AI가 본 당신의 가능성/ })
      ).toBeInTheDocument();
    });

    it('AI 진단 리포트 카드가 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByText('AI 진단 리포트')).toBeInTheDocument();
    });

    it('대학별 점수 바가 렌더링된다', () => {
      render(<Landing />);

      const aiSection = screen.getByTestId('ai-preview-section');

      expect(within(aiSection).getByText('서울대')).toBeInTheDocument();
      expect(within(aiSection).getByText('홍익대')).toBeInTheDocument();
      expect(within(aiSection).getByText('국민대')).toBeInTheDocument();

      // 점수 값 확인
      expect(within(aiSection).getByText('74')).toBeInTheDocument();
      expect(within(aiSection).getByText('81')).toBeInTheDocument();
      expect(within(aiSection).getByText('69')).toBeInTheDocument();
    });

    it('피드백 텍스트가 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByText(/구도와 명암에서 강점/)).toBeInTheDocument();
      expect(screen.getByText(/비례 보완 추천/)).toBeInTheDocument();
    });
  });

  describe('CTA Section', () => {
    it('CTA 섹션 라벨과 타이틀이 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByText('Pre-registration')).toBeInTheDocument();
      expect(
        screen.getByRole('heading', { name: /MIRIP, 곧 시작됩니다/ })
      ).toBeInTheDocument();
    });

    it('RegistrationForm이 렌더링된다', () => {
      render(<Landing />);

      // 폼 필드 확인
      expect(screen.getByLabelText(/이름/)).toBeInTheDocument();
      expect(screen.getByLabelText(/이메일/)).toBeInTheDocument();
      expect(screen.getByLabelText(/유저 유형/)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /등록하기/ })).toBeInTheDocument();
    });

    it('안내 텍스트가 렌더링된다', () => {
      render(<Landing />);

      expect(
        screen.getByText(/등록하신 정보는 서비스 출시 알림 목적으로만 사용됩니다/)
      ).toBeInTheDocument();
    });
  });

  describe('Success Modal', () => {
    it('등록 성공 시 성공 모달이 표시된다', async () => {
      const registrationService = require('../../services/registrationService');
      registrationService.create.mockResolvedValue({
        id: 'test-id',
        name: '홍길동',
        email: 'test@example.com',
        userType: 'student',
      });

      render(<Landing />);

      // 폼 작성
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      // 성공 모달 확인
      await waitFor(() => {
        expect(screen.getByText('등록이 완료되었습니다')).toBeInTheDocument();
      });

      expect(
        screen.getByText(/서비스 출시 시 가장 먼저 알려드리겠습니다/)
      ).toBeInTheDocument();
    });

    it('성공 모달 닫기 버튼이 작동한다', async () => {
      const registrationService = require('../../services/registrationService');
      registrationService.create.mockResolvedValue({
        id: 'test-id',
        name: '홍길동',
        email: 'test@example.com',
        userType: 'student',
      });

      render(<Landing />);

      // 폼 작성 및 제출
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      // 모달이 표시될 때까지 대기
      await waitFor(() => {
        expect(screen.getByText('등록이 완료되었습니다')).toBeInTheDocument();
      });

      // 닫기 버튼 클릭
      const closeButton = screen.getByRole('button', { name: /모달 닫기/ });
      fireEvent.click(closeButton);

      // 모달이 사라짐 확인
      await waitFor(() => {
        expect(screen.queryByText('등록이 완료되었습니다')).not.toBeInTheDocument();
      });
    });

    it('성공 모달은 오버레이 클릭으로 닫힌다', async () => {
      const registrationService = require('../../services/registrationService');
      registrationService.create.mockResolvedValue({
        id: 'test-id',
        name: '홍길동',
        email: 'test@example.com',
        userType: 'student',
      });

      render(<Landing />);

      // 폼 작성 및 제출
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      // 모달이 표시될 때까지 대기
      await waitFor(() => {
        expect(screen.getByText('등록이 완료되었습니다')).toBeInTheDocument();
      });

      // 오버레이 클릭
      const overlay = screen.getByTestId('modal-overlay');
      fireEvent.click(overlay);

      // 모달이 사라짐 확인
      await waitFor(() => {
        expect(screen.queryByText('등록이 완료되었습니다')).not.toBeInTheDocument();
      });
    });
  });

  describe('Navigation', () => {
    it('Header가 렌더링된다', () => {
      render(<Landing />);

      // Header와 Footer에 각각 navigation이 있으므로 최소 1개 이상 확인
      const navigations = screen.getAllByRole('navigation');
      expect(navigations.length).toBeGreaterThanOrEqual(1);
    });

    it('네비게이션 링크들이 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByRole('link', { name: /Why MIRIP/ })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /Solution/ })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: /AI Preview/ })).toBeInTheDocument();
    });

    it('Header CTA 버튼 클릭 시 CTA 섹션으로 스크롤된다', () => {
      render(<Landing />);

      // Header와 Hero 섹션에 동일한 이름의 버튼이 있으므로 첫 번째 버튼(Header)을 선택
      const ctaButtons = screen.getAllByRole('button', {
        name: /사전등록/,
      });
      const headerCtaButton = ctaButtons[0]; // Header의 CTA 버튼

      fireEvent.click(headerCtaButton);

      expect(mockScrollIntoView).toHaveBeenCalledWith({
        behavior: 'smooth',
        block: 'start',
      });
    });
  });

  describe('Footer', () => {
    it('Footer가 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByRole('contentinfo')).toBeInTheDocument();
    });

    it('저작권 텍스트가 렌더링된다', () => {
      render(<Landing />);

      expect(screen.getByText(/2025 MIRIP/)).toBeInTheDocument();
    });
  });

  describe('Smooth Scroll', () => {
    it('각 섹션에 id가 올바르게 설정된다', () => {
      render(<Landing />);

      expect(screen.getByTestId('hero-section')).toHaveAttribute('id', 'hero');
      expect(screen.getByTestId('problem-section')).toHaveAttribute('id', 'problem');
      expect(screen.getByTestId('solution-section')).toHaveAttribute('id', 'solution');
      expect(screen.getByTestId('ai-preview-section')).toHaveAttribute('id', 'ai-preview');
      expect(screen.getByTestId('cta-section')).toHaveAttribute('id', 'cta');
    });
  });

  describe('Accessibility', () => {
    it('모든 섹션에 적절한 heading 구조가 있다', () => {
      render(<Landing />);

      // h1은 하나만 있어야 함
      const h1Elements = screen.getAllByRole('heading', { level: 1 });
      expect(h1Elements).toHaveLength(1);

      // h2는 각 섹션에 있어야 함 (Problem, Solution, AI Preview, CTA)
      const h2Elements = screen.getAllByRole('heading', { level: 2 });
      expect(h2Elements.length).toBeGreaterThanOrEqual(4);
    });

    it('이미지에 대체 텍스트가 있다', () => {
      render(<Landing />);

      const images = screen.queryAllByRole('img');
      images.forEach((img) => {
        expect(img).toHaveAttribute('alt');
      });
    });
  });

  describe('Responsive Design', () => {
    it('모바일 뷰포트에서도 렌더링된다', () => {
      // 모바일 뷰포트 시뮬레이션
      window.innerWidth = 375;
      window.dispatchEvent(new Event('resize'));

      render(<Landing />);

      // 주요 요소들이 여전히 렌더링되는지 확인
      expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument();
      expect(screen.getByTestId('hero-section')).toBeInTheDocument();
      expect(screen.getByTestId('problem-section')).toBeInTheDocument();
      expect(screen.getByTestId('solution-section')).toBeInTheDocument();
      expect(screen.getByTestId('ai-preview-section')).toBeInTheDocument();
      expect(screen.getByTestId('cta-section')).toBeInTheDocument();
    });
  });
});
