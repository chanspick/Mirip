// Modal 컴포넌트 테스트
// SPEC-UI-001: 공통 UI 컴포넌트 - Modal

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Modal from './Modal';

describe('Modal 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('isOpen이 true일 때 모달이 렌더링된다', () => {
      render(
        <Modal isOpen={true} onClose={() => {}}>
          모달 내용
        </Modal>
      );
      expect(screen.getByText('모달 내용')).toBeInTheDocument();
    });

    test('isOpen이 false일 때 모달이 렌더링되지 않는다', () => {
      render(
        <Modal isOpen={false} onClose={() => {}}>
          모달 내용
        </Modal>
      );
      expect(screen.queryByText('모달 내용')).not.toBeInTheDocument();
    });

    test('title이 렌더링된다', () => {
      render(
        <Modal isOpen={true} onClose={() => {}} title="모달 제목">
          모달 내용
        </Modal>
      );
      expect(screen.getByText('모달 제목')).toBeInTheDocument();
    });

    test('children이 올바르게 렌더링된다', () => {
      render(
        <Modal isOpen={true} onClose={() => {}}>
          <p>본문 내용</p>
          <button>확인</button>
        </Modal>
      );
      expect(screen.getByText('본문 내용')).toBeInTheDocument();
      expect(screen.getByText('확인')).toBeInTheDocument();
    });
  });

  // onClose 테스트
  describe('onClose 동작', () => {
    test('오버레이 클릭 시 onClose가 호출된다', () => {
      const handleClose = jest.fn();
      render(
        <Modal isOpen={true} onClose={handleClose} data-testid="modal">
          모달 내용
        </Modal>
      );

      // 오버레이 클릭 (모달 배경)
      const overlay = screen.getByTestId('modal-overlay');
      fireEvent.click(overlay);

      expect(handleClose).toHaveBeenCalledTimes(1);
    });

    test('모달 콘텐츠 클릭 시 onClose가 호출되지 않는다', () => {
      const handleClose = jest.fn();
      render(
        <Modal isOpen={true} onClose={handleClose}>
          <div data-testid="modal-content-inner">모달 내용</div>
        </Modal>
      );

      // 모달 콘텐츠 클릭
      fireEvent.click(screen.getByTestId('modal-content-inner'));

      expect(handleClose).not.toHaveBeenCalled();
    });

    test('닫기 버튼 클릭 시 onClose가 호출된다', () => {
      const handleClose = jest.fn();
      render(
        <Modal isOpen={true} onClose={handleClose} title="제목">
          모달 내용
        </Modal>
      );

      const closeButton = screen.getByLabelText('모달 닫기');
      fireEvent.click(closeButton);

      expect(handleClose).toHaveBeenCalledTimes(1);
    });
  });

  // ESC 키 테스트
  describe('ESC 키 동작', () => {
    test('ESC 키를 누르면 onClose가 호출된다', () => {
      const handleClose = jest.fn();
      render(
        <Modal isOpen={true} onClose={handleClose}>
          모달 내용
        </Modal>
      );

      fireEvent.keyDown(document, { key: 'Escape', code: 'Escape' });

      expect(handleClose).toHaveBeenCalledTimes(1);
    });

    test('isOpen이 false일 때 ESC 키가 동작하지 않는다', () => {
      const handleClose = jest.fn();
      render(
        <Modal isOpen={false} onClose={handleClose}>
          모달 내용
        </Modal>
      );

      fireEvent.keyDown(document, { key: 'Escape', code: 'Escape' });

      expect(handleClose).not.toHaveBeenCalled();
    });
  });

  // 바디 스크롤 잠금 테스트
  describe('바디 스크롤 잠금', () => {
    test('모달이 열리면 body overflow가 hidden이 된다', () => {
      render(
        <Modal isOpen={true} onClose={() => {}}>
          모달 내용
        </Modal>
      );

      expect(document.body.style.overflow).toBe('hidden');
    });

    test('모달이 닫히면 body overflow가 복원된다', () => {
      const { rerender } = render(
        <Modal isOpen={true} onClose={() => {}}>
          모달 내용
        </Modal>
      );

      expect(document.body.style.overflow).toBe('hidden');

      rerender(
        <Modal isOpen={false} onClose={() => {}}>
          모달 내용
        </Modal>
      );

      expect(document.body.style.overflow).toBe('');
    });
  });

  // 접근성 테스트
  describe('접근성', () => {
    test('dialog role이 적용되어 있다', () => {
      render(
        <Modal isOpen={true} onClose={() => {}}>
          모달 내용
        </Modal>
      );

      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });

    test('aria-modal이 true로 설정되어 있다', () => {
      render(
        <Modal isOpen={true} onClose={() => {}}>
          모달 내용
        </Modal>
      );

      expect(screen.getByRole('dialog')).toHaveAttribute('aria-modal', 'true');
    });

    test('title이 있으면 aria-labelledby가 설정된다', () => {
      render(
        <Modal isOpen={true} onClose={() => {}} title="테스트 제목">
          모달 내용
        </Modal>
      );

      const dialog = screen.getByRole('dialog');
      const titleId = dialog.getAttribute('aria-labelledby');
      expect(titleId).toBeTruthy();
      expect(screen.getByText('테스트 제목')).toHaveAttribute('id', titleId);
    });
  });

  // className 테스트
  describe('className 조합', () => {
    test('추가 className을 전달할 수 있다', () => {
      render(
        <Modal isOpen={true} onClose={() => {}} className="custom-class">
          모달 내용
        </Modal>
      );

      expect(screen.getByRole('dialog')).toHaveClass('custom-class');
    });
  });
});
