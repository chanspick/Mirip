/**
 * Award Service 테스트
 *
 * 수상 기록 서비스를 테스트합니다.
 * SPEC-CRED-001 요구사항에 따라 작성되었습니다.
 *
 * @module services/awardService.test
 */

// Firebase 모듈 모킹
jest.mock('firebase/firestore', () => ({
  collection: jest.fn(),
  doc: jest.fn(),
  getDoc: jest.fn(),
  getDocs: jest.fn(),
  addDoc: jest.fn(),
  query: jest.fn(),
  where: jest.fn(),
  orderBy: jest.fn(),
  Timestamp: {
    now: jest.fn(() => ({
      _timestamp: true,
      toDate: () => new Date('2024-06-15'),
    })),
    fromDate: jest.fn((date) => ({
      toDate: () => date,
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
  query,
  where,
  orderBy,
  Timestamp,
} from 'firebase/firestore';

import {
  recordAward,
  getAwards,
  getPublicAwards,
} from './awardService';

describe('Award Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Timestamp.now.mockReturnValue({
      _timestamp: true,
      toDate: () => new Date('2024-06-15'),
    });
  });

  // ============================================
  // recordAward 테스트
  // ============================================
  describe('recordAward', () => {
    const mockUserId = 'test-user-123';
    const mockAwardData = {
      competitionId: 'comp-123',
      competitionTitle: '2024 전국 디자인 공모전',
      rank: '금상',
    };

    it('수상 기록을 성공적으로 추가해야 합니다', async () => {
      const mockDocRef = { id: 'award-123' };

      collection.mockReturnValue('mock-collection');
      addDoc.mockResolvedValue(mockDocRef);

      const result = await recordAward(mockUserId, mockAwardData);

      expect(addDoc).toHaveBeenCalledWith(
        'mock-collection',
        expect.objectContaining({
          competitionId: 'comp-123',
          competitionTitle: '2024 전국 디자인 공모전',
          rank: '금상',
          awardedAt: expect.any(Object),
        })
      );

      expect(result).toMatchObject({
        id: 'award-123',
        competitionId: 'comp-123',
        rank: '금상',
      });
    });

    it('awardedAt을 지정하면 해당 날짜로 저장해야 합니다', async () => {
      const mockDocRef = { id: 'award-123' };
      const customDate = new Date('2024-05-01');
      const awardDataWithDate = {
        ...mockAwardData,
        awardedAt: customDate,
      };

      collection.mockReturnValue('mock-collection');
      addDoc.mockResolvedValue(mockDocRef);

      await recordAward(mockUserId, awardDataWithDate);

      expect(Timestamp.fromDate).toHaveBeenCalledWith(customDate);
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(recordAward()).rejects.toThrow('userId는 필수입니다');
      await expect(recordAward('')).rejects.toThrow('userId는 필수입니다');
    });

    it('competitionId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(
        recordAward(mockUserId, { competitionTitle: '공모전', rank: '금상' })
      ).rejects.toThrow('competitionId는 필수입니다');
    });

    it('competitionTitle이 없는 경우 에러를 던져야 합니다', async () => {
      await expect(
        recordAward(mockUserId, { competitionId: 'comp-123', rank: '금상' })
      ).rejects.toThrow('competitionTitle은 필수입니다');
    });

    it('rank가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(
        recordAward(mockUserId, { competitionId: 'comp-123', competitionTitle: '공모전' })
      ).rejects.toThrow('rank는 필수입니다');
    });

    it('유효하지 않은 rank이면 에러를 던져야 합니다', async () => {
      await expect(
        recordAward(mockUserId, {
          competitionId: 'comp-123',
          competitionTitle: '공모전',
          rank: '최우수상', // 유효하지 않은 rank
        })
      ).rejects.toThrow('유효하지 않은 수상 등급입니다');
    });

    it('모든 유효한 rank를 허용해야 합니다', async () => {
      const validRanks = ['대상', '금상', '은상', '동상', '입선'];
      const mockDocRef = { id: 'award-123' };

      collection.mockReturnValue('mock-collection');
      addDoc.mockResolvedValue(mockDocRef);

      for (const rank of validRanks) {
        const result = await recordAward(mockUserId, {
          competitionId: 'comp-123',
          competitionTitle: '공모전',
          rank,
        });

        expect(result.rank).toBe(rank);
      }
    });
  });

  // ============================================
  // getAwards 테스트
  // ============================================
  describe('getAwards', () => {
    const mockUserId = 'test-user-123';

    it('수상 기록 목록을 반환해야 합니다', async () => {
      const mockAwards = [
        {
          id: 'a-1',
          competitionId: 'comp-1',
          competitionTitle: '공모전 1',
          rank: '대상',
          awardedAt: { toDate: () => new Date('2024-06-01') }
        },
        {
          id: 'a-2',
          competitionId: 'comp-2',
          competitionTitle: '공모전 2',
          rank: '금상',
          awardedAt: { toDate: () => new Date('2024-05-01') }
        },
      ];

      const mockQuerySnapshot = {
        docs: mockAwards.map((a) => ({
          id: a.id,
          data: () => a,
        })),
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      orderBy.mockReturnValue('mock-orderBy');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getAwards(mockUserId);

      expect(orderBy).toHaveBeenCalledWith('awardedAt', 'desc');
      expect(result).toHaveLength(2);
      expect(result[0]).toMatchObject({
        id: 'a-1',
        rank: '대상',
      });
    });

    it('수상 기록이 없으면 빈 배열을 반환해야 합니다', async () => {
      const mockQuerySnapshot = {
        docs: [],
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      orderBy.mockReturnValue('mock-orderBy');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getAwards(mockUserId);

      expect(result).toEqual([]);
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getAwards()).rejects.toThrow('userId는 필수입니다');
    });
  });

  // ============================================
  // getPublicAwards 테스트
  // ============================================
  describe('getPublicAwards', () => {
    const mockUserId = 'test-user-123';

    it('공개 프로필 사용자의 수상 기록을 반환해야 합니다', async () => {
      const mockUserProfile = {
        uid: mockUserId,
        isPublic: true,
      };

      const mockAwards = [
        {
          id: 'a-1',
          competitionId: 'comp-1',
          competitionTitle: '공모전 1',
          rank: '금상',
          awardedAt: { toDate: () => new Date() }
        },
      ];

      // 사용자 프로필 조회 모킹
      doc.mockReturnValue({ id: mockUserId });
      getDoc.mockResolvedValue({
        exists: () => true,
        data: () => mockUserProfile,
      });

      // 수상 기록 조회 모킹
      const mockQuerySnapshot = {
        docs: mockAwards.map((a) => ({
          id: a.id,
          data: () => a,
        })),
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      orderBy.mockReturnValue('mock-orderBy');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getPublicAwards(mockUserId);

      expect(result).toHaveLength(1);
      expect(result[0].rank).toBe('금상');
    });

    it('비공개 프로필 사용자의 경우 빈 배열을 반환해야 합니다', async () => {
      const mockUserProfile = {
        uid: mockUserId,
        isPublic: false,
      };

      doc.mockReturnValue({ id: mockUserId });
      getDoc.mockResolvedValue({
        exists: () => true,
        data: () => mockUserProfile,
      });

      const result = await getPublicAwards(mockUserId);

      expect(result).toEqual([]);
    });

    it('존재하지 않는 사용자의 경우 빈 배열을 반환해야 합니다', async () => {
      doc.mockReturnValue({ id: mockUserId });
      getDoc.mockResolvedValue({
        exists: () => false,
      });

      const result = await getPublicAwards(mockUserId);

      expect(result).toEqual([]);
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getPublicAwards()).rejects.toThrow('userId는 필수입니다');
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
        recordAward('test-user', {
          competitionId: 'comp-123',
          competitionTitle: '공모전',
          rank: '금상',
        })
      ).rejects.toThrow('수상 기록 중 오류가 발생했습니다');
    });
  });
});
