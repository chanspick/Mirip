// PortfolioUploadForm 컴포넌트 테스트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 업로드 폼
// TDD RED Phase: 실패하는 테스트 먼저 작성

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import PortfolioUploadForm from './PortfolioUploadForm';

// 테스트용 초기 데이터 (편집 모드용)
const mockInitialData = {
  id: 'portfolio-001',
  title: '정물화 작품',
  description: '사과와 꽃병을 그린 정물화입니다.',
  imageUrl: 'https://example.com/artwork.jpg',
  thumbnailUrl: 'https://example.com/artwork-thumb.jpg',
  tags: ['정물화', '수채화'],
  isPublic: true,
};

// 파일 객체 생성 헬퍼
const createMockFile = (name = 'test-image.jpg', size = 1024, type = 'image/jpeg') => {
  const file = new File(['test'], name, { type });
  Object.defineProperty(file, 'size', { value: size });
  return file;
};

describe('PortfolioUploadForm 컴포넌트', () => {
  // 기본 렌더링 테스트
  describe('기본 렌더링', () => {
    test('폼이 렌더링된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByTestId('upload-form')).toBeInTheDocument();
    });

    test('제목 입력 필드가 렌더링된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByLabelText(/제목/)).toBeInTheDocument();
    });

    test('설명 입력 필드가 렌더링된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByLabelText(/설명/)).toBeInTheDocument();
    });

    test('태그 입력 필드가 렌더링된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByLabelText(/태그/)).toBeInTheDocument();
    });

    test('공개/비공개 토글이 렌더링된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByTestId('public-toggle')).toBeInTheDocument();
    });

    test('이미지 업로드 영역이 렌더링된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByTestId('image-upload-area')).toBeInTheDocument();
    });

    test('제출 버튼이 렌더링된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByTestId('submit-button')).toBeInTheDocument();
    });

    test('취소 버튼이 렌더링된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByTestId('cancel-button')).toBeInTheDocument();
    });
  });

  // 편집 모드 테스트
  describe('편집 모드', () => {
    test('isEdit이 true이면 제출 버튼 텍스트가 수정으로 변경된다', () => {
      render(
        <PortfolioUploadForm
          onSubmit={jest.fn()}
          onCancel={jest.fn()}
          initialData={mockInitialData}
          isEdit={true}
        />
      );
      expect(screen.getByTestId('submit-button')).toHaveTextContent('수정');
    });

    test('isEdit이 false이면 제출 버튼 텍스트가 등록으로 표시된다', () => {
      render(
        <PortfolioUploadForm
          onSubmit={jest.fn()}
          onCancel={jest.fn()}
          isEdit={false}
        />
      );
      expect(screen.getByTestId('submit-button')).toHaveTextContent('등록');
    });

    test('initialData가 있으면 폼 필드가 채워진다', () => {
      render(
        <PortfolioUploadForm
          onSubmit={jest.fn()}
          onCancel={jest.fn()}
          initialData={mockInitialData}
          isEdit={true}
        />
      );
      expect(screen.getByLabelText(/제목/)).toHaveValue('정물화 작품');
      expect(screen.getByLabelText(/설명/)).toHaveValue('사과와 꽃병을 그린 정물화입니다.');
    });

    test('initialData에 태그가 있으면 태그가 표시된다', () => {
      render(
        <PortfolioUploadForm
          onSubmit={jest.fn()}
          onCancel={jest.fn()}
          initialData={mockInitialData}
          isEdit={true}
        />
      );
      expect(screen.getByText('정물화')).toBeInTheDocument();
      expect(screen.getByText('수채화')).toBeInTheDocument();
    });

    test('편집 모드에서 이미지 미리보기가 표시된다', () => {
      render(
        <PortfolioUploadForm
          onSubmit={jest.fn()}
          onCancel={jest.fn()}
          initialData={mockInitialData}
          isEdit={true}
        />
      );
      expect(screen.getByTestId('image-preview')).toBeInTheDocument();
    });
  });

  // 이미지 업로드 테스트
  describe('이미지 업로드', () => {
    test('파일 선택 시 미리보기가 표시된다', async () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      const input = screen.getByTestId('file-input');
      const file = createMockFile();

      // FileReader mock
      const mockFileReader = {
        readAsDataURL: jest.fn(),
        result: 'data:image/jpeg;base64,test',
        onload: null,
      };
      jest.spyOn(global, 'FileReader').mockImplementation(() => mockFileReader);

      fireEvent.change(input, { target: { files: [file] } });

      // Trigger onload
      mockFileReader.onload();

      await waitFor(() => {
        expect(screen.getByTestId('image-preview')).toBeInTheDocument();
      });

      jest.restoreAllMocks();
    });

    test('드래그 앤 드롭 영역이 존재한다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      const dropZone = screen.getByTestId('image-upload-area');
      expect(dropZone).toBeInTheDocument();
    });

    test('이미지가 없으면 업로드 안내 텍스트가 표시된다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      expect(screen.getByText(/이미지를 드래그하거나 클릭하여 업로드/)).toBeInTheDocument();
    });
  });

  // 태그 입력 테스트
  describe('태그 입력', () => {
    test('쉼표로 구분하여 태그를 추가할 수 있다', async () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      const tagInput = screen.getByLabelText(/태그/);

      await userEvent.type(tagInput, '정물화,');

      await waitFor(() => {
        expect(screen.getByText('정물화')).toBeInTheDocument();
      });
    });

    test('Enter 키로 태그를 추가할 수 있다', async () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      const tagInput = screen.getByLabelText(/태그/);

      await userEvent.type(tagInput, '풍경화{enter}');

      await waitFor(() => {
        expect(screen.getByText('풍경화')).toBeInTheDocument();
      });
    });

    test('태그 삭제 버튼을 클릭하면 태그가 제거된다', async () => {
      render(
        <PortfolioUploadForm
          onSubmit={jest.fn()}
          onCancel={jest.fn()}
          initialData={mockInitialData}
          isEdit={true}
        />
      );

      const removeButton = screen.getAllByTestId('remove-tag-button')[0];
      fireEvent.click(removeButton);

      await waitFor(() => {
        expect(screen.queryByText('정물화')).not.toBeInTheDocument();
      });
    });

    test('빈 태그는 추가되지 않는다', async () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      const tagInput = screen.getByLabelText(/태그/);

      await userEvent.type(tagInput, '   ,');

      const tags = screen.queryAllByTestId('tag-chip');
      expect(tags).toHaveLength(0);
    });
  });

  // 공개/비공개 토글 테스트
  describe('공개/비공개 토글', () => {
    test('기본값은 공개(true)이다', () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      const toggle = screen.getByTestId('public-toggle');
      expect(toggle).toBeChecked();
    });

    test('토글 클릭 시 상태가 변경된다', async () => {
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={jest.fn()} />);
      const toggle = screen.getByTestId('public-toggle');

      fireEvent.click(toggle);

      expect(toggle).not.toBeChecked();
    });
  });

  // 유효성 검사 테스트
  describe('유효성 검사', () => {
    test('제목이 비어있으면 에러 메시지가 표시된다', async () => {
      const handleSubmit = jest.fn();
      render(<PortfolioUploadForm onSubmit={handleSubmit} onCancel={jest.fn()} />);

      // 이미지만 선택 (제목 없이)
      const input = screen.getByTestId('file-input');
      const file = createMockFile();

      const mockFileReader = {
        readAsDataURL: jest.fn(),
        result: 'data:image/jpeg;base64,test',
        onload: null,
      };
      jest.spyOn(global, 'FileReader').mockImplementation(() => mockFileReader);
      fireEvent.change(input, { target: { files: [file] } });
      mockFileReader.onload();

      fireEvent.click(screen.getByTestId('submit-button'));

      await waitFor(() => {
        expect(screen.getByText(/제목을 입력해주세요/)).toBeInTheDocument();
      });
      expect(handleSubmit).not.toHaveBeenCalled();

      jest.restoreAllMocks();
    });

    test('이미지가 없으면 에러 메시지가 표시된다 (신규 등록 시)', async () => {
      const handleSubmit = jest.fn();
      render(<PortfolioUploadForm onSubmit={handleSubmit} onCancel={jest.fn()} />);

      // 제목만 입력 (이미지 없이)
      const titleInput = screen.getByLabelText(/제목/);
      await userEvent.type(titleInput, '테스트 작품');

      fireEvent.click(screen.getByTestId('submit-button'));

      await waitFor(() => {
        expect(screen.getByText(/이미지를 선택해주세요/)).toBeInTheDocument();
      });
      expect(handleSubmit).not.toHaveBeenCalled();
    });

    test('편집 모드에서는 기존 이미지가 있으면 새 이미지 없이 제출 가능하다', async () => {
      const handleSubmit = jest.fn();
      render(
        <PortfolioUploadForm
          onSubmit={handleSubmit}
          onCancel={jest.fn()}
          initialData={mockInitialData}
          isEdit={true}
        />
      );

      fireEvent.click(screen.getByTestId('submit-button'));

      await waitFor(() => {
        expect(handleSubmit).toHaveBeenCalled();
      });
    });
  });

  // 폼 제출 테스트
  describe('폼 제출', () => {
    test('유효한 폼 제출 시 onSubmit이 호출된다', async () => {
      const handleSubmit = jest.fn();
      render(<PortfolioUploadForm onSubmit={handleSubmit} onCancel={jest.fn()} />);

      // 이미지 선택
      const input = screen.getByTestId('file-input');
      const file = createMockFile();

      const mockFileReader = {
        readAsDataURL: jest.fn(),
        result: 'data:image/jpeg;base64,test',
        onload: null,
      };
      jest.spyOn(global, 'FileReader').mockImplementation(() => mockFileReader);
      fireEvent.change(input, { target: { files: [file] } });
      mockFileReader.onload();

      // 제목 입력
      const titleInput = screen.getByLabelText(/제목/);
      await userEvent.type(titleInput, '테스트 작품');

      // 제출
      fireEvent.click(screen.getByTestId('submit-button'));

      await waitFor(() => {
        expect(handleSubmit).toHaveBeenCalled();
      });

      jest.restoreAllMocks();
    });

    test('제출 시 폼 데이터가 올바르게 전달된다', async () => {
      const handleSubmit = jest.fn();
      render(<PortfolioUploadForm onSubmit={handleSubmit} onCancel={jest.fn()} />);

      // 이미지 선택
      const input = screen.getByTestId('file-input');
      const file = createMockFile();

      const mockFileReader = {
        readAsDataURL: jest.fn(),
        result: 'data:image/jpeg;base64,test',
        onload: null,
      };
      jest.spyOn(global, 'FileReader').mockImplementation(() => mockFileReader);
      fireEvent.change(input, { target: { files: [file] } });
      mockFileReader.onload();

      // 제목 입력
      await userEvent.type(screen.getByLabelText(/제목/), '테스트 작품');

      // 설명 입력
      await userEvent.type(screen.getByLabelText(/설명/), '테스트 설명');

      // 제출
      fireEvent.click(screen.getByTestId('submit-button'));

      await waitFor(() => {
        expect(handleSubmit).toHaveBeenCalledWith(
          expect.objectContaining({
            title: '테스트 작품',
            description: '테스트 설명',
            isPublic: true,
          }),
          expect.any(Object) // file
        );
      });

      jest.restoreAllMocks();
    });
  });

  // 취소 버튼 테스트
  describe('취소 버튼', () => {
    test('취소 버튼 클릭 시 onCancel이 호출된다', () => {
      const handleCancel = jest.fn();
      render(<PortfolioUploadForm onSubmit={jest.fn()} onCancel={handleCancel} />);

      fireEvent.click(screen.getByTestId('cancel-button'));

      expect(handleCancel).toHaveBeenCalledTimes(1);
    });
  });

  // 제출 중 상태 테스트
  describe('제출 중 상태', () => {
    test('submitting이 true이면 제출 버튼이 비활성화된다', () => {
      render(
        <PortfolioUploadForm
          onSubmit={jest.fn()}
          onCancel={jest.fn()}
          submitting={true}
        />
      );
      expect(screen.getByTestId('submit-button')).toBeDisabled();
    });

    test('submitting이 true이면 로딩 텍스트가 표시된다', () => {
      render(
        <PortfolioUploadForm
          onSubmit={jest.fn()}
          onCancel={jest.fn()}
          submitting={true}
        />
      );
      expect(screen.getByText(/처리 중/)).toBeInTheDocument();
    });
  });
});
