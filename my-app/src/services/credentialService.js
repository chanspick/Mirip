/**
 * Credential 서비스
 *
 * 사용자 프로필 CRUD 및 관련 기능을 제공합니다.
 * SPEC-CRED-001 요구사항에 따라 구현되었습니다.
 *
 * @module services/credentialService
 */

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
import { db } from '../config/firebase';
import { USERNAME_REGEX, BIO_MAX_LENGTH } from '../types/credential.types';

const USERS_COLLECTION = 'users';

/**
 * 이메일에서 기본 username 생성
 * @param {string} email - 이메일 주소
 * @returns {string} 생성된 username
 */
const generateUsernameFromEmail = (email) => {
  // 이메일의 @ 앞부분 추출
  const localPart = email.split('@')[0];
  // 소문자로 변환하고 특수문자를 언더스코어로 대체
  let username = localPart.toLowerCase().replace(/[^a-z0-9]/g, '_');
  // 연속된 언더스코어 제거
  username = username.replace(/_+/g, '_');
  // 앞뒤 언더스코어 제거
  username = username.replace(/^_+|_+$/g, '');
  // 최소 3자, 최대 30자
  if (username.length < 3) {
    username = username + '_user';
  }
  if (username.length > 30) {
    username = username.substring(0, 30);
  }
  return username;
};

/**
 * 고유한 username 생성 (중복 시 숫자 추가)
 * @param {string} baseUsername - 기본 username
 * @returns {Promise<string>} 고유한 username
 */
const generateUniqueUsername = async (baseUsername) => {
  let username = baseUsername;
  let counter = 1;

  // eslint-disable-next-line no-constant-condition
  while (true) {
    const isAvailable = await checkUsernameAvailableInternal(username);
    if (isAvailable) {
      return username;
    }
    // 숫자를 붙여서 다시 시도
    username = `${baseUsername}${counter}`;
    if (username.length > 30) {
      username = `${baseUsername.substring(0, 30 - String(counter).length)}${counter}`;
    }
    counter++;

    // 무한 루프 방지 (최대 1000번 시도)
    if (counter > 1000) {
      throw new Error('고유한 username을 생성할 수 없습니다');
    }
  }
};

/**
 * Username 중복 체크 (내부용, 유효성 검사 없이)
 * @param {string} username - 체크할 username
 * @returns {Promise<boolean>} 사용 가능하면 true
 */
const checkUsernameAvailableInternal = async (username) => {
  const usersRef = collection(db, USERS_COLLECTION);
  const q = query(usersRef, where('username', '==', username));
  const snapshot = await getDocs(q);
  return snapshot.empty;
};

/**
 * 사용자 프로필 조회
 * @param {string} uid - Firebase Auth UID
 * @returns {Promise<Object|null>} 사용자 프로필 또는 null
 * @throws {Error} uid가 없거나 Firestore 에러 시
 */
export const getUserProfile = async (uid) => {
  if (!uid) {
    throw new Error('uid는 필수입니다');
  }

  try {
    const docRef = doc(db, USERS_COLLECTION, uid);
    const docSnap = await getDoc(docRef);

    if (!docSnap.exists()) {
      return null;
    }

    const data = docSnap.data();
    return {
      ...data,
      uid: docSnap.id,
    };
  } catch (error) {
    console.error('프로필 조회 실패:', error);
    throw new Error('프로필 조회 중 오류가 발생했습니다');
  }
};

/**
 * Username으로 사용자 프로필 조회
 * @param {string} username - 사용자명
 * @returns {Promise<Object|null>} 사용자 프로필 또는 null
 * @throws {Error} username이 없거나 Firestore 에러 시
 */
export const getUserProfileByUsername = async (username) => {
  if (!username) {
    throw new Error('username은 필수입니다');
  }

  try {
    const usersRef = collection(db, USERS_COLLECTION);
    const q = query(usersRef, where('username', '==', username));
    const snapshot = await getDocs(q);

    if (snapshot.empty) {
      return null;
    }

    const docSnap = snapshot.docs[0];
    const data = docSnap.data();
    return {
      ...data,
      uid: docSnap.id,
    };
  } catch (error) {
    console.error('username으로 프로필 조회 실패:', error);
    throw new Error('프로필 조회 중 오류가 발생했습니다');
  }
};

/**
 * Username 사용 가능 여부 확인
 * @param {string} username - 체크할 username
 * @returns {Promise<boolean>} 사용 가능하면 true
 * @throws {Error} 유효하지 않은 username 형식
 */
export const checkUsernameAvailable = async (username) => {
  if (!username || !USERNAME_REGEX.test(username)) {
    throw new Error('유효하지 않은 username 형식입니다');
  }

  try {
    return await checkUsernameAvailableInternal(username);
  } catch (error) {
    console.error('username 중복 체크 실패:', error);
    throw new Error('username 확인 중 오류가 발생했습니다');
  }
};

/**
 * 사용자 프로필 업데이트
 * @param {string} uid - Firebase Auth UID
 * @param {Object} updateData - 업데이트할 데이터
 * @returns {Promise<void>}
 * @throws {Error} 유효성 검사 실패 또는 Firestore 에러 시
 */
export const updateUserProfile = async (uid, updateData) => {
  if (!uid) {
    throw new Error('uid는 필수입니다');
  }

  if (!updateData || Object.keys(updateData).length === 0) {
    throw new Error('업데이트할 데이터가 없습니다');
  }

  // bio 길이 검증
  if (updateData.bio && updateData.bio.length > BIO_MAX_LENGTH) {
    throw new Error('자기소개는 500자를 초과할 수 없습니다');
  }

  // username 변경 시 유효성 및 중복 체크
  if (updateData.username) {
    if (!USERNAME_REGEX.test(updateData.username)) {
      throw new Error('유효하지 않은 username 형식입니다');
    }

    const isAvailable = await checkUsernameAvailableInternal(updateData.username);
    if (!isAvailable) {
      // 현재 사용자의 username인지 확인
      const currentProfile = await getUserProfile(uid);
      if (!currentProfile || currentProfile.username !== updateData.username) {
        throw new Error('이미 사용 중인 username입니다');
      }
    }
  }

  try {
    const docRef = doc(db, USERS_COLLECTION, uid);
    await updateDoc(docRef, {
      ...updateData,
      updatedAt: Timestamp.now(),
    });
  } catch (error) {
    console.error('프로필 업데이트 실패:', error);
    throw new Error('프로필 업데이트 중 오류가 발생했습니다');
  }
};

/**
 * 새 사용자 프로필 초기화 (첫 로그인 시)
 * @param {string} uid - Firebase Auth UID
 * @param {string} email - 사용자 이메일
 * @returns {Promise<Object>} 생성된 또는 기존 프로필
 * @throws {Error} 필수 값 누락 또는 Firestore 에러 시
 */
export const initializeUserProfile = async (uid, email) => {
  if (!uid) {
    throw new Error('uid는 필수입니다');
  }

  if (!email) {
    throw new Error('email은 필수입니다');
  }

  try {
    // 기존 프로필 확인
    const existingProfile = await getUserProfile(uid);
    if (existingProfile) {
      return existingProfile;
    }

    // 기본 username 생성
    const baseUsername = generateUsernameFromEmail(email);
    const username = await generateUniqueUsername(baseUsername);

    // displayName 생성 (이메일의 @ 앞부분)
    const displayName = email.split('@')[0];

    // 새 프로필 생성
    const newProfile = {
      uid,
      username,
      displayName,
      profileImageUrl: null,
      bio: '',
      tier: 'Unranked',
      totalActivities: 0,
      currentStreak: 0,
      longestStreak: 0,
      isPublic: true,
      createdAt: Timestamp.now(),
      updatedAt: Timestamp.now(),
    };

    const docRef = doc(db, USERS_COLLECTION, uid);
    await setDoc(docRef, newProfile);

    return newProfile;
  } catch (error) {
    console.error('프로필 초기화 실패:', error);
    throw new Error('프로필 초기화 중 오류가 발생했습니다');
  }
};

const credentialService = {
  getUserProfile,
  getUserProfileByUsername,
  updateUserProfile,
  checkUsernameAvailable,
  initializeUserProfile,
};

export default credentialService;
