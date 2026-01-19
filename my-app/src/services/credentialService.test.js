/**
 * Credential Service 테스트
 *
 * 사용자 프로필 CRUD 서비스를 테스트합니다.
 * SPEC-CRED-001 요구사항에 따라 작성되었습니다.
 *
 * @module services/credentialService.test
 */

// Firebase 모듈 모킹
jest.mock('firebase/firestore', () => ({
  collection: jest.fn(),
  doc: jest.fn(),
  getDoc: jest.fn(),
  getDocs: jest.fn(),
  setDoc: jest.fn(),
  updateDoc: jest.fn(),
  query: jest.fn(),
  where: jest.fn(),
  Timestamp: {
    now: jest.fn(() => ({ _timestamp: true, toDate: () => new Date() })),
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
  setDoc,
  updateDoc,
  query,
  where,
  Timestamp,
} from 'firebase/firestore';

import {
  getUserProfile,
  getUserProfileByUsername,
  updateUserProfile,
  checkUsernameAvailable,
  initializeUserProfile,
} from './credentialService';

describe('Credential Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Timestamp.now.mockReturnValue({ _timestamp: true, toDate: () => new Date() });
  });

  // ============================================
  // getUserProfile 테스트
  // ============================================
  describe('getUserProfile', () => {
    const mockUid = 'test-user-123';
    const mockUserData = {
      uid: mockUid,
      username: 'testuser',
      displayName: '테스트 유저',
      tier: 'B',
      totalActivities: 10,
      currentStreak: 3,
      longestStreak: 7,
      isPublic: true,
      createdAt: { toDate: () => new Date('2024-01-01') },
      updatedAt: { toDate: () => new Date('2024-01-15') },
    };

    it('존재하는 사용자 프로필을 반환해야 합니다', async () => {
      const mockDocRef = { id: mockUid };
      const mockDocSnap = {
        exists: () => true,
        data: () => mockUserData,
        id: mockUid,
      };

      doc.mockReturnValue(mockDocRef);
      getDoc.mockResolvedValue(mockDocSnap);

      const result = await getUserProfile(mockUid);

      expect(doc).toHaveBeenCalled();
      expect(getDoc).toHaveBeenCalledWith(mockDocRef);
      expect(result).toMatchObject({
        uid: mockUid,
        username: 'testuser',
        displayName: '테스트 유저',
        tier: 'B',
      });
    });

    it('존재하지 않는 사용자의 경우 null을 반환해야 합니다', async () => {
      const mockDocSnap = {
        exists: () => false,
      };

      doc.mockReturnValue({ id: 'non-existent' });
      getDoc.mockResolvedValue(mockDocSnap);

      const result = await getUserProfile('non-existent-uid');

      expect(result).toBeNull();
    });

    it('uid가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getUserProfile()).rejects.toThrow('uid는 필수입니다');
      await expect(getUserProfile('')).rejects.toThrow('uid는 필수입니다');
      await expect(getUserProfile(null)).rejects.toThrow('uid는 필수입니다');
    });
  });

  // ============================================
  // getUserProfileByUsername 테스트
  // ============================================
  describe('getUserProfileByUsername', () => {
    const mockUsername = 'testuser';
    const mockUserData = {
      uid: 'test-user-123',
      username: mockUsername,
      displayName: '테스트 유저',
      tier: 'A',
      isPublic: true,
    };

    it('username으로 사용자 프로필을 조회해야 합니다', async () => {
      const mockQuerySnapshot = {
        empty: false,
        docs: [
          {
            id: 'test-user-123',
            data: () => mockUserData,
          },
        ],
      };

      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      collection.mockReturnValue('mock-collection');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getUserProfileByUsername(mockUsername);

      expect(where).toHaveBeenCalledWith('username', '==', mockUsername);
      expect(result).toMatchObject({
        uid: 'test-user-123',
        username: mockUsername,
      });
    });

    it('존재하지 않는 username의 경우 null을 반환해야 합니다', async () => {
      const mockQuerySnapshot = {
        empty: true,
        docs: [],
      };

      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      collection.mockReturnValue('mock-collection');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await getUserProfileByUsername('nonexistent');

      expect(result).toBeNull();
    });

    it('username이 없는 경우 에러를 던져야 합니다', async () => {
      await expect(getUserProfileByUsername()).rejects.toThrow('username은 필수입니다');
      await expect(getUserProfileByUsername('')).rejects.toThrow('username은 필수입니다');
    });
  });

  // ============================================
  // checkUsernameAvailable 테스트
  // ============================================
  describe('checkUsernameAvailable', () => {
    it('사용 가능한 username이면 true를 반환해야 합니다', async () => {
      const mockQuerySnapshot = {
        empty: true,
        docs: [],
      };

      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      collection.mockReturnValue('mock-collection');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await checkUsernameAvailable('newuser');

      expect(result).toBe(true);
    });

    it('이미 사용 중인 username이면 false를 반환해야 합니다', async () => {
      const mockQuerySnapshot = {
        empty: false,
        docs: [{ id: 'existing-user' }],
      };

      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      collection.mockReturnValue('mock-collection');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      const result = await checkUsernameAvailable('existinguser');

      expect(result).toBe(false);
    });

    it('유효하지 않은 username 형식이면 에러를 던져야 합니다', async () => {
      // 대문자 포함
      await expect(checkUsernameAvailable('TestUser')).rejects.toThrow('유효하지 않은 username 형식입니다');
      // 특수문자 포함
      await expect(checkUsernameAvailable('test-user')).rejects.toThrow('유효하지 않은 username 형식입니다');
      // 너무 짧음
      await expect(checkUsernameAvailable('ab')).rejects.toThrow('유효하지 않은 username 형식입니다');
      // 너무 김
      await expect(checkUsernameAvailable('a'.repeat(31))).rejects.toThrow('유효하지 않은 username 형식입니다');
    });
  });

  // ============================================
  // updateUserProfile 테스트
  // ============================================
  describe('updateUserProfile', () => {
    const mockUid = 'test-user-123';

    it('프로필을 성공적으로 업데이트해야 합니다', async () => {
      const updateData = {
        displayName: '새로운 이름',
        bio: '새로운 자기소개',
      };

      const mockDocRef = { id: mockUid };
      doc.mockReturnValue(mockDocRef);
      updateDoc.mockResolvedValue();

      await updateUserProfile(mockUid, updateData);

      expect(updateDoc).toHaveBeenCalledWith(
        mockDocRef,
        expect.objectContaining({
          displayName: '새로운 이름',
          bio: '새로운 자기소개',
          updatedAt: expect.any(Object),
        })
      );
    });

    it('username 변경 시 중복 체크를 수행해야 합니다', async () => {
      const mockQuerySnapshot = {
        empty: true,
        docs: [],
      };

      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      collection.mockReturnValue('mock-collection');
      getDocs.mockResolvedValue(mockQuerySnapshot);
      doc.mockReturnValue({ id: mockUid });
      updateDoc.mockResolvedValue();

      await updateUserProfile(mockUid, { username: 'newusername' });

      expect(where).toHaveBeenCalledWith('username', '==', 'newusername');
    });

    it('이미 사용 중인 username으로 변경 시 에러를 던져야 합니다', async () => {
      const mockQuerySnapshot = {
        empty: false,
        docs: [{ id: 'other-user' }],
      };

      // username 중복 체크용 모킹
      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      collection.mockReturnValue('mock-collection');
      getDocs.mockResolvedValue(mockQuerySnapshot);

      // getUserProfile 호출용 getDoc 모킹 (현재 사용자 프로필 조회)
      doc.mockReturnValue({ id: mockUid });
      getDoc.mockResolvedValue({
        exists: () => true,
        data: () => ({
          uid: mockUid,
          username: 'differentuser', // 현재 username과 다른 값
          displayName: '테스트 유저',
        }),
        id: mockUid,
      });

      await expect(
        updateUserProfile(mockUid, { username: 'existinguser' })
      ).rejects.toThrow('이미 사용 중인 username입니다');
    });

    it('bio가 500자를 초과하면 에러를 던져야 합니다', async () => {
      const longBio = 'a'.repeat(501);

      await expect(
        updateUserProfile(mockUid, { bio: longBio })
      ).rejects.toThrow('자기소개는 500자를 초과할 수 없습니다');
    });

    it('uid가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(updateUserProfile()).rejects.toThrow('uid는 필수입니다');
      await expect(updateUserProfile('')).rejects.toThrow('uid는 필수입니다');
    });

    it('업데이트할 데이터가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(updateUserProfile(mockUid, {})).rejects.toThrow('업데이트할 데이터가 없습니다');
      await expect(updateUserProfile(mockUid, null)).rejects.toThrow('업데이트할 데이터가 없습니다');
    });
  });

  // ============================================
  // initializeUserProfile 테스트
  // ============================================
  describe('initializeUserProfile', () => {
    const mockUid = 'new-user-123';
    const mockEmail = 'newuser@example.com';

    it('새 사용자 프로필을 초기화해야 합니다', async () => {
      const mockDocRef = { id: mockUid };
      const mockDocSnap = {
        exists: () => false,
      };

      doc.mockReturnValue(mockDocRef);
      getDoc.mockResolvedValue(mockDocSnap);
      setDoc.mockResolvedValue();

      // username 중복 체크 모킹
      query.mockReturnValue('mock-query');
      where.mockReturnValue('mock-where');
      collection.mockReturnValue('mock-collection');
      getDocs.mockResolvedValue({ empty: true, docs: [] });

      const result = await initializeUserProfile(mockUid, mockEmail);

      expect(setDoc).toHaveBeenCalledWith(
        mockDocRef,
        expect.objectContaining({
          uid: mockUid,
          username: expect.any(String),
          displayName: expect.any(String),
          tier: 'Unranked',
          totalActivities: 0,
          currentStreak: 0,
          longestStreak: 0,
          isPublic: true,
        })
      );

      expect(result).toMatchObject({
        uid: mockUid,
        tier: 'Unranked',
      });
    });

    it('이미 프로필이 존재하면 기존 프로필을 반환해야 합니다', async () => {
      const existingProfile = {
        uid: mockUid,
        username: 'existinguser',
        displayName: '기존 유저',
        tier: 'A',
      };

      const mockDocSnap = {
        exists: () => true,
        data: () => existingProfile,
        id: mockUid,
      };

      doc.mockReturnValue({ id: mockUid });
      getDoc.mockResolvedValue(mockDocSnap);

      const result = await initializeUserProfile(mockUid, mockEmail);

      expect(setDoc).not.toHaveBeenCalled();
      expect(result).toMatchObject(existingProfile);
    });

    it('uid가 없는 경우 에러를 던져야 합니다', async () => {
      await expect(initializeUserProfile()).rejects.toThrow('uid는 필수입니다');
      await expect(initializeUserProfile('')).rejects.toThrow('uid는 필수입니다');
    });

    it('email이 없는 경우 에러를 던져야 합니다', async () => {
      await expect(initializeUserProfile(mockUid)).rejects.toThrow('email은 필수입니다');
      await expect(initializeUserProfile(mockUid, '')).rejects.toThrow('email은 필수입니다');
    });
  });

  // ============================================
  // 에러 처리 테스트
  // ============================================
  describe('에러 처리', () => {
    it('Firestore 에러 발생 시 적절한 에러를 던져야 합니다', async () => {
      const firestoreError = new Error('Firestore connection failed');
      doc.mockReturnValue({ id: 'test' });
      getDoc.mockRejectedValue(firestoreError);

      await expect(getUserProfile('test-uid')).rejects.toThrow('프로필 조회 중 오류가 발생했습니다');
    });
  });
});
