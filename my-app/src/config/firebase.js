/**
 * Firebase 설정 파일
 *
 * Firebase SDK를 초기화하고 Firestore 인스턴스를 내보냅니다.
 * 모든 민감한 설정값은 환경 변수에서 로드됩니다.
 *
 * SPEC-FIREBASE-001 요구사항에 따라 구현되었습니다.
 *
 * @module config/firebase
 */

import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';
import { getAuth } from 'firebase/auth';

/**
 * Firebase 환경 변수 유효성 검사
 * 필수 환경 변수가 설정되지 않은 경우 경고를 출력합니다.
 */
const validateEnvironmentVariables = () => {
  const requiredEnvVars = [
    'REACT_APP_FIREBASE_API_KEY',
    'REACT_APP_FIREBASE_AUTH_DOMAIN',
    'REACT_APP_FIREBASE_PROJECT_ID',
    'REACT_APP_FIREBASE_STORAGE_BUCKET',
    'REACT_APP_FIREBASE_MESSAGING_SENDER_ID',
    'REACT_APP_FIREBASE_APP_ID',
  ];

  const missingVars = requiredEnvVars.filter(
    (envVar) => !process.env[envVar]
  );

  if (missingVars.length > 0) {
    console.warn(
      `Firebase 설정 경고: 다음 환경 변수가 설정되지 않았습니다: ${missingVars.join(', ')}`
    );
  }
};

// 환경 변수 유효성 검사 실행
validateEnvironmentVariables();

/**
 * Firebase 설정 객체
 * 환경 변수에서 모든 값을 로드합니다.
 *
 * @type {Object}
 */
const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.REACT_APP_FIREBASE_APP_ID,
};

/**
 * Firebase 앱 인스턴스
 * @type {FirebaseApp}
 */
const app = initializeApp(firebaseConfig);

/**
 * Firestore 데이터베이스 인스턴스
 * @type {Firestore}
 */
const db = getFirestore(app);

/**
 * Firebase Storage 인스턴스
 * @type {FirebaseStorage}
 */
const storage = getStorage(app);

/**
 * Firebase Auth 인스턴스
 * @type {Auth}
 */
const auth = getAuth(app);

export { app, db, storage, auth };
