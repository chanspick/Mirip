/**
 * Activity Service 테스트
 *
 * 활동 기록 서비스를 테스트합니다.
 * SPEC-CRED-001 요구사항에 따라 작성되었습니다.
 *
 * @module services/activityService.test
 */

// Firebase 모듈 모킹
jest.mock('firebase/firestore', () => ({
  collection: jest.fn(),
  doc: jest.fn(),
  getDoc: jest.fn(),
  getDocs: jest.fn(),
  addDoc: jest.fn(),
  updateDoc: jest.fn(),
  setDoc: jest.fn(),
  query: jest.fn(),
  where: jest.fn(),
  orderBy: jest.fn(),
  limit: jest.fn(),
  startAfter: jest.fn(),
  Timestamp: {
    now: jest.fn(() => ({
      _timestamp: true,
      toDate: () => new Date('2024-06-15'),
      toMillis: () => new Date('2024-06-15').getTime(),
    })),
    fromDate: jest.fn((date) => ({
      toDate: () => date,
      toMillis: () => date.getTime(),
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
  setDoc,
  query,
  where,
  orderBy,
  limit,
  startAfter,
  Timestamp,
} from 'firebase/firestore';

import {
  recordActivity,
  getActivities,
  getActivityHeatmapData,
  calculateStreak,
  getDailyActivityCount,
} from './activityService';

describe('Activity Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Timestamp.now.mockReturnValue({
      _timestamp: true,
      toDate: () => new Date('2024-06-15'),
      toMillis: () => new Date('2024-06-15').getTime(),
    });
  });

  // ============================================
  // recordActivity 테스트
  // ============================================
  describe('recordActivity', () => {
    const mockUserId = 'test-user-123';
    const mockActivityData = {
      type: 'diagnosis',
      title: 'AI 진단 완료',
      description: '시각디자인 진단',
      metadata: { diagnosisId: 'diag-123', tier: 'A' },
    };

    it('활동을 성공적으로 기록해야 합니다', async () => {
      const mockDocRef = { id: 'activity-123' };

      collection.mockReturnValue('mock-collection');
      addDoc.mockResolvedValue(mockDocRef);
      doc.mockReturnValue({ id: mockUserId });

      // getDoc이 여러 번 호출됨:
      // 1. updateDailyActivityCount에서 일별 활동 문서 조회
      // 2. updateUserStats에서 사용자 문서 조회
      // 3. calculateStreak에서 getDocs 사용 (별도 모킹 필요 없음)
      getDoc
        .mockResolvedValueOnce({
          // 일별 활동 문서 (처음 생성되는 경우)
          exists: () => false,
        })
        .mockResolvedValueOnce({
          // 사용자 문서
          exists: () => true,
          data: () => ({ totalActivities: 5, currentStreak: 2, longestStreak: 5 }),
        });

      // calculateStreak용 getDocs 모킹
      getDocs.mockResolvedValue({ docs: [] });

      updateDoc.mockResolvedValue();
      setDoc.mockResolvedValue();

      const result = await recordActivity(mockUserId, mockActivityData);

      expect(addDoc).toHaveBeenCalled();
      expect(result).toMatchObject({
        id: 'activity-123',
        type: 'diagnosis',
        title: 'AI 진단 완료',
      });
    });

    it('사용자의 총 활동 수를 업데이트해야 합니다', async () => {
      const mockDocRef = { id: 'activity-123' };

      collection.mockReturnValue('mock-collection');
      addDoc.mockResolvedValue(mockDocRef);
      doc.mockReturnValue({ id: mockUserId });

      // getDoc이 여러 번 호출됨:
      // 1. updateDailyActivityCount에서 일별 활동 문서 조회
      // 2. updateUserStats에서 사용자 문서 조회
      getDoc
        .mockResolvedValueOnce({
          // 일별 활동 문서 (처음 생성되는 경우)
          exists: () => false,
        })
        .mockResolvedValueOnce({
          // 사용자 문서
          exists: () => true,
          data: () => ({ totalActivities: 5, currentStreak: 2, longestStreak: 5 }),
        });

      // calculateStreak용 getDocs 모킹
      getDocs.mockResolvedValue({ docs: [] });

      updateDoc.mockResolvedValue();
      setDoc.mockResolvedValue();

      await recordActivity(mockUserId, mockActivityData);

      expect(updateDoc).toHaveBeenCalled();
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(recordActivity()).rejects.toThrow('userId는 필수입니다');
      await expect(recordActivity('')).rejects.toThrow('userId는 필수입니다');
    });

    it('유효하지 않은 활동 타입이면 에러를 던져야 합니다', async () => {
      const invalidData = {
        type: 'invalid_type',
        title: '테스트',
      };

      await expect(recordActivity(mockUserId, invalidData)).rejects.toThrow('유효하지 않은 활동 타입입니다');
    });

    it('title이 없는 경우 에러를 던져야 합니다', async () => {
      const invalidData = {
        type: 'diagnosis',
      };

      await expect(recordActivity(mockUserId, invalidData)).rejects.toThrow('title은 필수입니다');
    });
  });

  // ============================================
  // getActivities 테스트
  // ============================================
  describe('getActivities', () => {
    const mockUserId = 'test-user-123';

    it('활동 목록을 페이지네이션과 함께 반환해야 합니다', async () => {
      const mockActivities = [
        { id: 'act-1', type: 'diagnosis', title: '진단 1', createdAt: { toDate: () => new Date() } },
        { id: 'act-2', type: 'portfolio_add', title: '포트폴리오 추가', createdAt: { toDate: () => new Date() } },
      ];

      const mockQuerySnapshot = {
        docs: mockActivities.map((act, idx) => ({
          id: act.id,
          data: () => act,
        })),
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      orderBy.mockReturnValue('mock-orderBy');
      limit.mockReturnValue('mock-limit');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getActivities(mockUserId, { pageSize: 20 });

      expect(orderBy).toHaveBeenCalledWith('createdAt', 'desc');
      expect(limit).toHaveBeenCalled();
      expect(result.activities).toHaveLength(2);
      expect(result).toHaveProperty('lastDoc');
      expect(result).toHaveProperty('hasMore');
    });

    it('타입 필터가 적용되어야 합니다', async () => {
      const mockQuerySnapshot = {
        docs: [],
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      orderBy.mockReturnValue('mock-orderBy');
      limit.mockReturnValue('mock-limit');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      await getActivities(mockUserId, { type: 'diagnosis' });

      expect(where).toHaveBeenCalledWith('type', '==', 'diagnosis');
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

      await getActivities(mockUserId, { lastDoc: mockLastDoc });

      expect(startAfter).toHaveBeenCalledWith(mockLastDoc);
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getActivities()).rejects.toThrow('userId는 필수입니다');
    });
  });

  // ============================================
  // getActivityHeatmapData 테스트
  // ============================================
  describe('getActivityHeatmapData', () => {
    const mockUserId = 'test-user-123';
    const mockYear = 2024;

    it('1년간의 활동 히트맵 데이터를 반환해야 합니다', async () => {
      const mockDailyDocs = [
        { id: '2024-06-01', data: () => ({ date: '2024-06-01', count: 3, activityIds: ['a1', 'a2', 'a3'] }) },
        { id: '2024-06-15', data: () => ({ date: '2024-06-15', count: 1, activityIds: ['a4'] }) },
      ];

      const mockQuerySnapshot = {
        docs: mockDailyDocs,
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getActivityHeatmapData(mockUserId, mockYear);

      expect(result).toHaveProperty('year', mockYear);
      expect(result).toHaveProperty('days');
      expect(result).toHaveProperty('totalActivities');
      expect(Array.isArray(result.days)).toBe(true);
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getActivityHeatmapData()).rejects.toThrow('userId는 필수입니다');
    });

    it('year가 없는 경우 현재 연도를 사용해야 합니다', async () => {
      const mockQuerySnapshot = { docs: [] };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getActivityHeatmapData(mockUserId);

      expect(result).toHaveProperty('year');
      expect(typeof result.year).toBe('number');
    });
  });

  // ============================================
  // calculateStreak 테스트
  // ============================================
  describe('calculateStreak', () => {
    const mockUserId = 'test-user-123';

    it('연속 활동일을 계산해야 합니다', async () => {
      // 오늘부터 3일 연속 활동
      const today = new Date('2024-06-15');
      const mockDailyDocs = [
        { id: '2024-06-15', data: () => ({ date: '2024-06-15', count: 1 }) },
        { id: '2024-06-14', data: () => ({ date: '2024-06-14', count: 2 }) },
        { id: '2024-06-13', data: () => ({ date: '2024-06-13', count: 1 }) },
        // 6월 12일은 없음 (연속 끊김)
        { id: '2024-06-11', data: () => ({ date: '2024-06-11', count: 1 }) },
      ];

      const mockQuerySnapshot = {
        docs: mockDailyDocs,
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      orderBy.mockReturnValue('mock-orderBy');
      limit.mockReturnValue('mock-limit');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await calculateStreak(mockUserId);

      expect(result).toHaveProperty('currentStreak');
      expect(result).toHaveProperty('longestStreak');
      expect(typeof result.currentStreak).toBe('number');
    });

    it('활동이 없는 경우 0을 반환해야 합니다', async () => {
      const mockQuerySnapshot = {
        docs: [],
      };

      collection.mockReturnValue('mock-collection');
      query.mockReturnValue('mock-query');
      orderBy.mockReturnValue('mock-orderBy');
      limit.mockReturnValue('mock-limit');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await calculateStreak(mockUserId);

      expect(result.currentStreak).toBe(0);
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(calculateStreak()).rejects.toThrow('userId는 필수입니다');
    });
  });

  // ============================================
  // getDailyActivityCount 테스트
  // ============================================
  describe('getDailyActivityCount', () => {
    const mockUserId = 'test-user-123';
    const mockDate = '2024-06-15';

    it('특정 날짜의 활동 수를 반환해야 합니다', async () => {
      const mockDocSnap = {
        exists: () => true,
        data: () => ({ date: mockDate, count: 5, activityIds: ['a1', 'a2', 'a3', 'a4', 'a5'] }),
      };

      doc.mockReturnValue({ id: mockDate });
      getDoc.mockResolvedValue(mockDocSnap);

      const result = await getDailyActivityCount(mockUserId, mockDate);

      expect(result).toMatchObject({
        date: mockDate,
        count: 5,
      });
      expect(result.activityIds).toHaveLength(5);
    });

    it('해당 날짜에 활동이 없으면 count 0을 반환해야 합니다', async () => {
      const mockDocSnap = {
        exists: () => false,
      };

      doc.mockReturnValue({ id: mockDate });
      getDoc.mockResolvedValue(mockDocSnap);

      const result = await getDailyActivityCount(mockUserId, mockDate);

      expect(result).toMatchObject({
        date: mockDate,
        count: 0,
        activityIds: [],
      });
    });

    it('userId가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getDailyActivityCount()).rejects.toThrow('userId는 필수입니다');
    });

    it('date가 없는 경우 오늘 날짜를 사용해야 합니다', async () => {
      const mockDocSnap = {
        exists: () => false,
      };

      doc.mockReturnValue({ id: '2024-06-15' });
      getDoc.mockResolvedValue(mockDocSnap);

      const result = await getDailyActivityCount(mockUserId);

      expect(result).toHaveProperty('date');
      expect(result).toHaveProperty('count', 0);
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
        recordActivity('test-user', { type: 'diagnosis', title: '테스트' })
      ).rejects.toThrow('활동 기록 중 오류가 발생했습니다');
    });
  });
});
