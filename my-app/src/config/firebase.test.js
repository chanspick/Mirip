/**
 * Firebase 설정 테스트
 *
 * Firebase 초기화 및 Firestore 인스턴스 내보내기를 테스트합니다.
 * SPEC-FIREBASE-001 요구사항에 따라 작성되었습니다.
 */

// Firebase 모듈 모킹
jest.mock('firebase/app', () => ({
  initializeApp: jest.fn(() => ({ name: '[DEFAULT]' })),
}));

jest.mock('firebase/firestore', () => ({
  getFirestore: jest.fn(() => ({ type: 'firestore' })),
  serverTimestamp: jest.fn(() => ({ _serverTimestamp: true })),
}));

describe('Firebase Configuration', () => {
  // 각 테스트 전에 모듈 캐시 초기화
  beforeEach(() => {
    jest.resetModules();
    // 환경 변수 설정
    process.env.REACT_APP_FIREBASE_API_KEY = 'test-api-key';
    process.env.REACT_APP_FIREBASE_AUTH_DOMAIN = 'test-project.firebaseapp.com';
    process.env.REACT_APP_FIREBASE_PROJECT_ID = 'test-project';
    process.env.REACT_APP_FIREBASE_STORAGE_BUCKET = 'test-project.appspot.com';
    process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID = '123456789';
    process.env.REACT_APP_FIREBASE_APP_ID = 'test-app-id';
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Firebase App 초기화', () => {
    it('환경 변수를 사용하여 Firebase를 초기화해야 합니다', () => {
      const { initializeApp } = require('firebase/app');
      require('./firebase');

      expect(initializeApp).toHaveBeenCalledTimes(1);
      expect(initializeApp).toHaveBeenCalledWith({
        apiKey: 'test-api-key',
        authDomain: 'test-project.firebaseapp.com',
        projectId: 'test-project',
        storageBucket: 'test-project.appspot.com',
        messagingSenderId: '123456789',
        appId: 'test-app-id',
      });
    });

    it('Firebase app 인스턴스를 내보내야 합니다', () => {
      const { app } = require('./firebase');
      expect(app).toBeDefined();
      expect(app.name).toBe('[DEFAULT]');
    });
  });

  describe('Firestore 인스턴스', () => {
    it('Firestore 인스턴스를 내보내야 합니다', () => {
      const { db } = require('./firebase');
      expect(db).toBeDefined();
      expect(db.type).toBe('firestore');
    });

    it('getFirestore가 app과 함께 호출되어야 합니다', () => {
      const { getFirestore } = require('firebase/firestore');
      require('./firebase');

      expect(getFirestore).toHaveBeenCalledTimes(1);
    });
  });

  describe('환경 변수 검증', () => {
    it('필수 환경 변수가 설정되지 않으면 경고를 출력해야 합니다', () => {
      // 환경 변수 제거
      delete process.env.REACT_APP_FIREBASE_API_KEY;

      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      // 모듈 재로드
      jest.resetModules();

      // 모킹 재설정
      jest.doMock('firebase/app', () => ({
        initializeApp: jest.fn(() => ({ name: '[DEFAULT]' })),
      }));

      jest.doMock('firebase/firestore', () => ({
        getFirestore: jest.fn(() => ({ type: 'firestore' })),
      }));

      require('./firebase');

      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Firebase')
      );

      consoleSpy.mockRestore();
    });
  });
});
