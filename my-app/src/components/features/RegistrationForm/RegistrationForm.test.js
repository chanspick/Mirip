// RegistrationForm 컴포넌트 테스트
// SPEC-FIREBASE-001: 사전 등록 폼 컴포넌트
// TDD RED 단계: 실패하는 테스트 먼저 작성

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import RegistrationForm from './RegistrationForm';
import * as registrationService from '../../../services/registrationService';

// registrationService 모킹
jest.mock('../../../services/registrationService');

describe('RegistrationForm 컴포넌트', () => {
  // 각 테스트 전에 모킹 초기화
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // ===== 기본 렌더링 테스트 =====
  describe('기본 렌더링', () => {
    test('이름 입력 필드를 렌더링한다', () => {
      render(<RegistrationForm />);
      expect(screen.getByLabelText(/이름/)).toBeInTheDocument();
    });

    test('이메일 입력 필드를 렌더링한다', () => {
      render(<RegistrationForm />);
      expect(screen.getByLabelText(/이메일/)).toBeInTheDocument();
    });

    test('유저 유형 선택 필드를 렌더링한다', () => {
      render(<RegistrationForm />);
      expect(screen.getByLabelText(/유저 유형/)).toBeInTheDocument();
    });

    test('제출 버튼을 렌더링한다', () => {
      render(<RegistrationForm />);
      expect(screen.getByRole('button', { name: /등록하기/ })).toBeInTheDocument();
    });

    test('className prop이 적용된다', () => {
      const { container } = render(<RegistrationForm className="custom-class" />);
      expect(container.firstChild).toHaveClass('custom-class');
    });
  });

  // ===== 유저 유형 옵션 테스트 =====
  describe('유저 유형 옵션', () => {
    test('입시생 옵션이 존재한다', () => {
      render(<RegistrationForm />);
      const select = screen.getByLabelText(/유저 유형/);
      expect(select).toContainHTML('입시생');
    });

    test('학부모 옵션이 존재한다', () => {
      render(<RegistrationForm />);
      const select = screen.getByLabelText(/유저 유형/);
      expect(select).toContainHTML('학부모');
    });

    test('신진 작가 옵션이 존재한다', () => {
      render(<RegistrationForm />);
      const select = screen.getByLabelText(/유저 유형/);
      expect(select).toContainHTML('신진 작가');
    });

    test('공모전 주최자 옵션이 존재한다', () => {
      render(<RegistrationForm />);
      const select = screen.getByLabelText(/유저 유형/);
      expect(select).toContainHTML('공모전 주최자');
    });
  });

  // ===== 이름 유효성 검사 테스트 =====
  describe('이름 유효성 검사', () => {
    test('이름이 비어있으면 에러 메시지를 표시한다', async () => {
      render(<RegistrationForm />);

      // 폼 제출
      const submitButton = screen.getByRole('button', { name: /등록하기/ });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('이름을 입력해주세요')).toBeInTheDocument();
      });
    });

    test('이름이 2자 미만이면 에러 메시지를 표시한다', async () => {
      render(<RegistrationForm />);

      // 1글자만 입력
      const nameInput = screen.getByLabelText(/이름/);
      await userEvent.type(nameInput, '홍');

      // 폼 제출
      const submitButton = screen.getByRole('button', { name: /등록하기/ });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('이름은 2자 이상이어야 합니다')).toBeInTheDocument();
      });
    });

    test('이름이 2자 이상이면 에러 메시지가 표시되지 않는다', async () => {
      render(<RegistrationForm />);

      // 2글자 이상 입력
      const nameInput = screen.getByLabelText(/이름/);
      await userEvent.type(nameInput, '홍길동');

      // blur 이벤트로 유효성 검사 트리거
      fireEvent.blur(nameInput);

      await waitFor(() => {
        expect(screen.queryByText('이름을 입력해주세요')).not.toBeInTheDocument();
        expect(screen.queryByText('이름은 2자 이상이어야 합니다')).not.toBeInTheDocument();
      });
    });
  });

  // ===== 이메일 유효성 검사 테스트 =====
  describe('이메일 유효성 검사', () => {
    test('이메일이 비어있으면 에러 메시지를 표시한다', async () => {
      render(<RegistrationForm />);

      // 이름만 입력하고 폼 제출
      const nameInput = screen.getByLabelText(/이름/);
      await userEvent.type(nameInput, '홍길동');

      const submitButton = screen.getByRole('button', { name: /등록하기/ });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('이메일을 입력해주세요')).toBeInTheDocument();
      });
    });

    test('이메일 형식이 올바르지 않으면 에러 메시지를 표시한다', async () => {
      render(<RegistrationForm />);

      // 잘못된 이메일 형식 입력
      const emailInput = screen.getByLabelText(/이메일/);
      await userEvent.type(emailInput, 'invalid-email');

      // 폼 제출
      const submitButton = screen.getByRole('button', { name: /등록하기/ });
      fireEvent.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('올바른 이메일을 입력해주세요')).toBeInTheDocument();
      });
    });

    test('올바른 이메일 형식이면 에러 메시지가 표시되지 않는다', async () => {
      render(<RegistrationForm />);

      // 올바른 이메일 형식 입력
      const emailInput = screen.getByLabelText(/이메일/);
      await userEvent.type(emailInput, 'test@example.com');

      // blur 이벤트로 유효성 검사 트리거
      fireEvent.blur(emailInput);

      await waitFor(() => {
        expect(screen.queryByText('이메일을 입력해주세요')).not.toBeInTheDocument();
        expect(screen.queryByText('올바른 이메일을 입력해주세요')).not.toBeInTheDocument();
      });
    });
  });

  // ===== 유저 유형 유효성 검사 테스트 =====
  describe('유저 유형 유효성 검사', () => {
    test('유저 유형을 선택하지 않으면 에러 메시지를 표시한다', async () => {
      render(<RegistrationForm />);

      // 다른 필드는 채우고 유저 유형만 비워둠
      const nameInput = screen.getByLabelText(/이름/);
      await userEvent.type(nameInput, '홍길동');

      const emailInput = screen.getByLabelText(/이메일/);
      await userEvent.type(emailInput, 'test@example.com');

      // 폼 제출
      const submitButton = screen.getByRole('button', { name: /등록하기/ });
      fireEvent.click(submitButton);

      // 에러 메시지 확인 (에러 메시지는 span.errorMessage 클래스에서 표시됨)
      await waitFor(() => {
        const errorElements = document.querySelectorAll('[class*="errorMessage"]');
        const hasUserTypeError = Array.from(errorElements).some(
          el => el.textContent === '유저 유형을 선택해주세요'
        );
        expect(hasUserTypeError).toBe(true);
      });
    });

    test('유저 유형을 선택하면 에러 메시지가 표시되지 않는다', async () => {
      render(<RegistrationForm />);

      // 먼저 폼을 제출하여 에러 상태 발생
      const submitButton = screen.getByRole('button', { name: /등록하기/ });
      fireEvent.click(submitButton);

      // 에러 메시지가 표시됨 확인 (에러 메시지는 span.errorMessage 클래스에서 표시됨)
      await waitFor(() => {
        const errorElements = document.querySelectorAll('[class*="errorMessage"]');
        const hasUserTypeError = Array.from(errorElements).some(
          el => el.textContent === '유저 유형을 선택해주세요'
        );
        expect(hasUserTypeError).toBe(true);
      });

      // 유저 유형 선택
      const userTypeSelect = screen.getByLabelText(/유저 유형/);
      await userEvent.selectOptions(userTypeSelect, 'student');

      // 에러 메시지가 사라졌는지 확인
      await waitFor(() => {
        const errorElements = document.querySelectorAll('[class*="errorMessage"]');
        const hasUserTypeError = Array.from(errorElements).some(
          el => el.textContent === '유저 유형을 선택해주세요'
        );
        expect(hasUserTypeError).toBe(false);
      });
    });
  });

  // ===== 폼 제출 테스트 =====
  describe('폼 제출', () => {
    test('모든 필드가 유효하면 registrationService.create를 호출한다', async () => {
      registrationService.create.mockResolvedValue({
        id: 'test-id',
        name: '홍길동',
        email: 'test@example.com',
        userType: 'student',
      });

      render(<RegistrationForm />);

      // 모든 필드 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      await waitFor(() => {
        expect(registrationService.create).toHaveBeenCalledWith({
          name: '홍길동',
          email: 'test@example.com',
          userType: 'student',
        });
      });
    });

    test('유효하지 않은 필드가 있으면 registrationService.create를 호출하지 않는다', async () => {
      render(<RegistrationForm />);

      // 이름만 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      await waitFor(() => {
        expect(registrationService.create).not.toHaveBeenCalled();
      });
    });
  });

  // ===== 로딩 상태 테스트 =====
  describe('로딩 상태', () => {
    test('제출 중에는 버튼이 비활성화된다', async () => {
      // 지연된 Promise 생성
      let resolvePromise;
      registrationService.create.mockReturnValue(
        new Promise((resolve) => {
          resolvePromise = resolve;
        })
      );

      render(<RegistrationForm />);

      // 모든 필드 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      // 버튼이 비활성화되었는지 확인
      await waitFor(() => {
        expect(screen.getByRole('button')).toBeDisabled();
      });

      // Promise 해결
      resolvePromise({ id: 'test-id' });
    });

    test('제출 중에는 폼 필드가 비활성화된다', async () => {
      // 지연된 Promise 생성
      let resolvePromise;
      registrationService.create.mockReturnValue(
        new Promise((resolve) => {
          resolvePromise = resolve;
        })
      );

      render(<RegistrationForm />);

      // 모든 필드 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      // 필드들이 비활성화되었는지 확인
      await waitFor(() => {
        expect(screen.getByLabelText(/이름/)).toBeDisabled();
        expect(screen.getByLabelText(/이메일/)).toBeDisabled();
        expect(screen.getByLabelText(/유저 유형/)).toBeDisabled();
      });

      // Promise 해결
      resolvePromise({ id: 'test-id' });
    });

    test('제출 중에는 로딩 텍스트가 표시된다', async () => {
      // 지연된 Promise 생성
      let resolvePromise;
      registrationService.create.mockReturnValue(
        new Promise((resolve) => {
          resolvePromise = resolve;
        })
      );

      render(<RegistrationForm />);

      // 모든 필드 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      // 로딩 텍스트 확인
      await waitFor(() => {
        expect(screen.getByText(/등록 중.../)).toBeInTheDocument();
      });

      // Promise 해결
      resolvePromise({ id: 'test-id' });
    });
  });

  // ===== 콜백 테스트 =====
  describe('콜백 함수', () => {
    test('등록 성공 시 onSuccess 콜백이 호출된다', async () => {
      const mockOnSuccess = jest.fn();
      const registrationResult = {
        id: 'test-id',
        name: '홍길동',
        email: 'test@example.com',
        userType: 'student',
      };

      registrationService.create.mockResolvedValue(registrationResult);

      render(<RegistrationForm onSuccess={mockOnSuccess} />);

      // 모든 필드 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalledWith(registrationResult);
      });
    });

    test('등록 실패 시 onError 콜백이 호출된다', async () => {
      const mockOnError = jest.fn();
      const error = new Error('등록 저장 중 오류가 발생했습니다');

      registrationService.create.mockRejectedValue(error);

      render(<RegistrationForm onError={mockOnError} />);

      // 모든 필드 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      await waitFor(() => {
        expect(mockOnError).toHaveBeenCalledWith(error);
      });
    });
  });

  // ===== 에러 처리 테스트 =====
  describe('에러 처리', () => {
    test('서비스 에러 시 에러 메시지를 표시한다', async () => {
      registrationService.create.mockRejectedValue(
        new Error('등록 저장 중 오류가 발생했습니다')
      );

      render(<RegistrationForm />);

      // 모든 필드 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      await waitFor(() => {
        expect(screen.getByText(/등록 저장 중 오류가 발생했습니다/)).toBeInTheDocument();
      });
    });

    test('에러 발생 후 버튼이 다시 활성화된다', async () => {
      registrationService.create.mockRejectedValue(
        new Error('등록 저장 중 오류가 발생했습니다')
      );

      render(<RegistrationForm />);

      // 모든 필드 입력
      await userEvent.type(screen.getByLabelText(/이름/), '홍길동');
      await userEvent.type(screen.getByLabelText(/이메일/), 'test@example.com');
      await userEvent.selectOptions(screen.getByLabelText(/유저 유형/), 'student');

      // 폼 제출
      fireEvent.click(screen.getByRole('button', { name: /등록하기/ }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /등록하기/ })).not.toBeDisabled();
      });
    });
  });

  // ===== 입력 상태 관리 테스트 =====
  describe('입력 상태 관리', () => {
    test('이름 입력 시 상태가 업데이트된다', () => {
      render(<RegistrationForm />);

      const nameInput = screen.getByLabelText(/이름/);
      // fireEvent.change를 사용하여 한글 입력 문제 해결
      fireEvent.change(nameInput, { target: { value: '홍길동' } });

      expect(nameInput).toHaveValue('홍길동');
    });

    test('이메일 입력 시 상태가 업데이트된다', async () => {
      render(<RegistrationForm />);

      const emailInput = screen.getByLabelText(/이메일/);
      await userEvent.type(emailInput, 'test@example.com');

      expect(emailInput).toHaveValue('test@example.com');
    });

    test('유저 유형 선택 시 상태가 업데이트된다', async () => {
      render(<RegistrationForm />);

      const userTypeSelect = screen.getByLabelText(/유저 유형/);
      await userEvent.selectOptions(userTypeSelect, 'parent');

      expect(userTypeSelect).toHaveValue('parent');
    });
  });
});
