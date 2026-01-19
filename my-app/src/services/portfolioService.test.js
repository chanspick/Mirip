/**
 * Portfolio Service 테스트
 *
 * 포트폴리오 관리 서비스를 테스트합니다.
 * SPEC-CRED-001 요구사항에 따라 작성되었습니다.
 *
 * @module services/portfolioService.test
 */

// Firebase 모듈 모킹
jest.mock('firebase/firestore', () => ({
  collection: jest.fn(),
  doc: jest.fn(),
  getDoc: jest.fn(),
  getDocs: jest.fn(),
  addDoc: jest.fn(),
  updateDoc: jest.fn(),
  deleteDoc: jest.fn(),
  query: jest.fn(),
  where: jest.fn(),
  orderBy: jest.fn(),
  limit: jest.fn(),
  startAfter: jest.fn(),
  Timestamp: {
    now: jest.fn(() => ({
      _timestamp: true,
      toDate: () => new Date('2024-06-15'),
    })),
  },
}));

// Firebase config 모킹
jest.mock('../config/firebase', () => ({
  db: { type: 'firestore-mock' },
}));

import {
  collection,
  doc,
  getDoc,
  getDocs,
  addDoc,
  updateDoc,
  deleteDoc,
  query,
  where,
  orderBy,
  limit,
  startAfter,
  Timestamp,
} from 'firebase/firestore';

import {
  addPortfolio,
  updatePortfolio,
  deletePortfolio,
  getPortfolios,
  getPublicPortfolios,
} from './portfolioService';

describe('Portfolio Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Timestamp.now.mockReturnValue({
      _timestamp: true,
      toDate: () => new Date('2024-06-15'),
    });
  });

  // ============================================
  // addPortfolio 테스트
  // ============================================
  describe('addPortfolio', () => {
    const mockUserId = 'test-user-123';
    const mockPortfolioData = {
      title: '나의 작품',
      description: '시각디자인 포트폴리오',
      imageUrl: 'https://storage.example.com/image.jpg',
      thumbnailUrl: 'https://storage.example.com/thumb.jpg',
      tags: ['시각디자인', '포스터'],
      isPublic: true,
    };

    it('포트폴리오를 성공적으로 추가해야 합니다', async () => {
      const mockDocRef = { id: 'portfolio-123' };

      collection.mockReturnValue('mock-collection');
      addDoc.mockResolvedValue(mockDocRef);

      const result = await addPortfolio(mockUserId, mockPortfolioData);

      expect(addDoc).toHaveBeenCalledWith(
        'mock-collection',
        expect.objectContaining({
          userId: mockUserId,
          title: '나의 작품',
          imageUrl: 'https://storage.example.com/image.jpg',
          isPublic: true,
          createdAt: expect.any(Object),
          updatedAt: expect.any(Object),
        })
      );

      expect(result).toMatchObject({
        id: 'portfolio-123',
        title: '나의 작품',
      });
    });

    it('isPublic 기본값은 true여야 합니다', async () => {
      const mockDocRef = { id: 'portfolio-123' };
      const dataWithoutPublic = {
        title: '나의 작품',
        imageUrl: 'https://storage.example.com/image.jpg',
      };

      collection.mockReturnValue('mock-collection');
      addDoc.mockResolvedValue(mockDocRef);

      await addPortfolio(mockUserId, dataWithoutPublic);

      expect(addDoc).toHaveBeenCalledWith(
        'mock-collection',
        expect.objectContaining({
          isPublic: true,
        })
      );
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(addPortfolio()).rejects.toThrow('userId는 필수입니다');
      await expect(addPortfolio('')).rejects.toThrow('userId는 필수입니다');
    });

    it('title이 없는 경우 에러를 던져야 합니다', async () => {
      await expect(
        addPortfolio(mockUserId, { imageUrl: 'https://example.com/img.jpg' })
      ).rejects.toThrow('title은 필수입니다');
    });

    it('imageUrl이 없는 경우 에러를 던져야 합니다', async () => {
      await expect(
        addPortfolio(mockUserId, { title: '테스트' })
      ).rejects.toThrow('imageUrl은 필수입니다');
    });
  });

  // ============================================
  // updatePortfolio 테스트
  // ============================================
  describe('updatePortfolio', () => {
    const mockUserId = 'test-user-123';
    const mockPortfolioId = 'portfolio-123';

    it('포트폴리오를 성공적으로 업데이트해야 합니다', async () => {
      const updateData = {
        title: '수정된 제목',
        description: '수정된 설명',
      };

      const mockDocRef = { id: mockPortfolioId };
      const mockDocSnap = {
        exists: () => true,
        data: () => ({ userId: mockUserId, title: '원래 제목' }),
      };

      doc.mockReturnValue(mockDocRef);
      getDoc.mockResolvedValue(mockDocSnap);
      updateDoc.mockResolvedValue();

      await updatePortfolio(mockUserId, mockPortfolioId, updateData);

      expect(updateDoc).toHaveBeenCalledWith(
        mockDocRef,
        expect.objectContaining({
          title: '수정된 제목',
          description: '수정된 설명',
          updatedAt: expect.any(Object),
        })
      );
    });

    it('다른 사용자의 포트폴리오 수정 시 에러를 던져야 합니다', async () => {
      const mockDocRef = { id: mockPortfolioId };
      const mockDocSnap = {
        exists: () => true,
        data: () => ({ userId: 'other-user', title: '다른 사용자 작품' }),
      };

      doc.mockReturnValue(mockDocRef);
      getDoc.mockResolvedValue(mockDocSnap);

      await expect(
        updatePortfolio(mockUserId, mockPortfolioId, { title: '수정' })
      ).rejects.toThrow('포트폴리오 수정 권한이 없습니다');
    });

    it('존재하지 않는 포트폴리오 수정 시 에러를 던져야 합니다', async () => {
      const mockDocRef = { id: mockPortfolioId };
      const mockDocSnap = {
        exists: () => false,
      };

      doc.mockReturnValue(mockDocRef);
      getDoc.mockResolvedValue(mockDocSnap);

      await expect(
        updatePortfolio(mockUserId, mockPortfolioId, { title: '수정' })
      ).rejects.toThrow('포트폴리오를 찾을 수 없습니다');
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(updatePortfolio()).rejects.toThrow('userId는 필수입니다');
    });

    it('portfolioId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(updatePortfolio(mockUserId)).rejects.toThrow('portfolioId는 필수입니다');
    });

    it('업데이트할 데이터가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(
        updatePortfolio(mockUserId, mockPortfolioId, {})
      ).rejects.toThrow('업데이트할 데이터가 없습니다');
    });
  });

  // ============================================
  // deletePortfolio 테스트
  // ============================================
  describe('deletePortfolio', () => {
    const mockUserId = 'test-user-123';
    const mockPortfolioId = 'portfolio-123';

    it('포트폴리오를 성공적으로 삭제해야 합니다', async () => {
      const mockDocRef = { id: mockPortfolioId };
      const mockDocSnap = {
        exists: () => true,
        data: () => ({ userId: mockUserId, title: '삭제할 작품' }),
      };

      doc.mockReturnValue(mockDocRef);
      getDoc.mockResolvedValue(mockDocSnap);
      deleteDoc.mockResolvedValue();

      await deletePortfolio(mockUserId, mockPortfolioId);

      expect(deleteDoc).toHaveBeenCalledWith(mockDocRef);
    });

    it('다른 사용자의 포트폴리오 삭제 시 에러를 던져야 합니다', async () => {
      const mockDocRef = { id: mockPortfolioId };
      const mockDocSnap = {
        exists: () => true,
        data: () => ({ userId: 'other-user', title: '다른 사용자 작품' }),
      };

      doc.mockReturnValue(mockDocRef);
      getDoc.mockResolvedValue(mockDocSnap);

      await expect(
        deletePortfolio(mockUserId, mockPortfolioId)
      ).rejects.toThrow('포트폴리오 삭제 권한이 없습니다');
    });

    it('존재하지 않는 포트폴리오 삭제 시 에러를 던져야 합니다', async () => {
      const mockDocRef = { id: mockPortfolioId };
      const mockDocSnap = {
        exists: () => false,
      };

      doc.mockReturnValue(mockDocRef);
      getDoc.mockResolvedValue(mockDocSnap);

      await expect(
        deletePortfolio(mockUserId, mockPortfolioId)
      ).rejects.toThrow('포트폴리오를 찾을 수 없습니다');
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(deletePortfolio()).rejects.toThrow('userId는 필수입니다');
    });

    it('portfolioId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(deletePortfolio(mockUserId)).rejects.toThrow('portfolioId는 필수입니다');
    });
  });

  // ============================================
  // getPortfolios 테스트
  // ============================================
  describe('getPortfolios', () => {
    const mockUserId = 'test-user-123';

    it('포트폴리오 목록을 페이지네이션과 함께 반환해야 합니다', async () => {
      const mockPortfolios = [
        { id: 'p-1', title: '작품 1', isPublic: true, createdAt: { toDate: () => new Date() } },
        { id: 'p-2', title: '작품 2', isPublic: false, createdAt: { toDate: () => new Date() } },
      ];

      const mockQuerySnapshot = {
        docs: mockPortfolios.map((p) => ({
          id: p.id,
          data: () => p,
        })),
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      orderBy.mockReturnValue('mock-orderBy');
      limit.mockReturnValue('mock-limit');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getPortfolios(mockUserId, { pageSize: 12 });

      expect(orderBy).toHaveBeenCalledWith('createdAt', 'desc');
      expect(result.portfolios).toHaveLength(2);
      expect(result).toHaveProperty('lastDoc');
      expect(result).toHaveProperty('hasMore');
    });

    it('publicOnly 옵션이 적용되어야 합니다', async () => {
      const mockQuerySnapshot = {
        docs: [],
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      orderBy.mockReturnValue('mock-orderBy');
      limit.mockReturnValue('mock-limit');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      await getPortfolios(mockUserId, { publicOnly: true });

      expect(where).toHaveBeenCalledWith('isPublic', '==', true);
    });

    it('페이지네이션이 올바르게 동작해야 합니다', async () => {
      const mockLastDoc = { id: 'last-doc' };
      const mockQuerySnapshot = {
        docs: [],
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      orderBy.mockReturnValue('mock-orderBy');
      limit.mockReturnValue('mock-limit');
      startAfter.mockReturnValue('mock-startAfter');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      await getPortfolios(mockUserId, { lastDoc: mockLastDoc });

      expect(startAfter).toHaveBeenCalledWith(mockLastDoc);
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getPortfolios()).rejects.toThrow('userId는 필수입니다');
    });
  });

  // ============================================
  // getPublicPortfolios 테스트
  // ============================================
  describe('getPublicPortfolios', () => {
    const mockUserId = 'test-user-123';

    it('공개 포트폴리오만 반환해야 합니다', async () => {
      const mockPortfolios = [
        { id: 'p-1', title: '공개 작품', isPublic: true, createdAt: { toDate: () => new Date() } },
      ];

      const mockQuerySnapshot = {
        docs: mockPortfolios.map((p) => ({
          id: p.id,
          data: () => p,
        })),
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      orderBy.mockReturnValue('mock-orderBy');
      limit.mockReturnValue('mock-limit');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getPublicPortfolios(mockUserId);

      expect(where).toHaveBeenCalledWith('isPublic', '==', true);
      expect(result.portfolios).toHaveLength(1);
      expect(result.portfolios[0].isPublic).toBe(true);
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getPublicPortfolios()).rejects.toThrow('userId는 필수입니다');
    });
  });

  // ============================================
  // 에러 처리 테스트
  // ============================================
  describe('에러 처리', () => {
    it('Firestore 에러 발생 시 적절한 에러를 던져야 합니다', async () => {
      const firestoreError = new Error('Firestore connection failed');
      collection.mockReturnValue('mock-collection');
      addDoc.mockRejectedValue(firestoreError);

      await expect(
        addPortfolio('test-user', { title: '테스트', imageUrl: 'https://example.com/img.jpg' })
      ).rejects.toThrow('포트폴리오 추가 중 오류가 발생했습니다');
    });
  });
});
