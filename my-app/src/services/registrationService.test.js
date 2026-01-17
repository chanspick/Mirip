/**
 * Registration Service 테스트
 *
 * Firestore에 등록 데이터를 저장하는 서비스를 테스트합니다.
 * SPEC-FIREBASE-001 요구사항에 따라 작성되었습니다.
 */

// Firebase 모듈 모킹
jest.mock('firebase/firestore', () => ({
  collection: jest.fn(),
  addDoc: jest.fn(),
  serverTimestamp: jest.fn(() => ({ _serverTimestamp: true })),
}));

// Firebase config 모킹
jest.mock('../config/firebase', () => ({
  db: { type: 'firestore-mock' },
}));

import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { create } from './registrationService';

describe('Registration Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // serverTimestamp 반환값 재설정
    serverTimestamp.mockReturnValue({ _serverTimestamp: true });
  });

  describe('create 함수', () => {
    const mockRegistrationData = {
      name: '홍길동',
      email: 'hong@example.com',
      phone: '010-1234-5678',
      company: '테스트 회사',
    };

    it('유효한 데이터로 등록을 생성해야 합니다', async () => {
      // Firestore 모킹 설정
      const mockDocRef = { id: 'test-doc-id' };
      const mockCollectionRef = { path: 'registrations' };

      collection.mockReturnValue(mockCollectionRef);
      addDoc.mockResolvedValue(mockDocRef);

      const result = await create(mockRegistrationData);

      // collection 함수가 올바른 인자로 호출되었는지 확인
      expect(collection).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'firestore-mock' }),
        'registrations'
      );

      // addDoc 함수가 올바른 데이터로 호출되었는지 확인
      expect(addDoc).toHaveBeenCalledWith(mockCollectionRef, {
        ...mockRegistrationData,
        timestamp: expect.objectContaining({ _serverTimestamp: true }),
      });

      // 결과값 확인
      expect(result).toEqual({
        id: 'test-doc-id',
        ...mockRegistrationData,
      });
    });

    it('serverTimestamp를 사용하여 타임스탬프를 추가해야 합니다', async () => {
      const mockDocRef = { id: 'test-doc-id' };
      const mockCollectionRef = { path: 'registrations' };

      collection.mockReturnValue(mockCollectionRef);
      addDoc.mockResolvedValue(mockDocRef);

      await create(mockRegistrationData);

      // serverTimestamp가 호출되었는지 확인
      expect(serverTimestamp).toHaveBeenCalled();

      // addDoc에 전달된 데이터에 timestamp가 포함되어 있는지 확인
      expect(addDoc).toHaveBeenCalledWith(
        mockCollectionRef,
        expect.objectContaining({
          timestamp: expect.objectContaining({ _serverTimestamp: true }),
        })
      );
    });

    it('필수 필드가 누락된 경우 에러를 던져야 합니다', async () => {
      const invalidData = {
        name: '홍길동',
        // email 누락
      };

      await expect(create(invalidData)).rejects.toThrow('필수 필드가 누락되었습니다');
    });

    it('이메일 형식이 잘못된 경우 에러를 던져야 합니다', async () => {
      const invalidEmailData = {
        name: '홍길동',
        email: 'invalid-email',
        phone: '010-1234-5678',
      };

      await expect(create(invalidEmailData)).rejects.toThrow('유효하지 않은 이메일 형식입니다');
    });
  });

  describe('에러 처리', () => {
    it('Firestore 에러 발생 시 적절한 에러를 던져야 합니다', async () => {
      const mockRegistrationData = {
        name: '홍길동',
        email: 'hong@example.com',
        phone: '010-1234-5678',
      };

      const mockCollectionRef = { path: 'registrations' };
      collection.mockReturnValue(mockCollectionRef);

      // Firestore 에러 시뮬레이션
      const firestoreError = new Error('Firestore connection failed');
      addDoc.mockRejectedValue(firestoreError);

      await expect(create(mockRegistrationData)).rejects.toThrow('등록 저장 중 오류가 발생했습니다');
    });

    it('빈 객체가 전달된 경우 에러를 던져야 합니다', async () => {
      await expect(create({})).rejects.toThrow('필수 필드가 누락되었습니다');
    });

    it('null이나 undefined가 전달된 경우 에러를 던져야 합니다', async () => {
      await expect(create(null)).rejects.toThrow('유효하지 않은 데이터입니다');
      await expect(create(undefined)).rejects.toThrow('유효하지 않은 데이터입니다');
    });
  });
});
