/**
 * Award 서비스
 *
 * 수상 기록 관리 기능을 제공합니다.
 * SPEC-CRED-001 요구사항에 따라 구현되었습니다.
 *
 * @module services/awardService
 */

import {
  collection,
  doc,
  getDoc,
  getDocs,
  addDoc,
  query,
  orderBy,
  Timestamp,
} from 'firebase/firestore';
import { db } from '../config/firebase';
import { AWARD_RANKS } from '../types/credential.types';

const USERS_COLLECTION = 'users';
const AWARDS_SUBCOLLECTION = 'awards';

/**
 * 수상 기록 추가
 * @param {string} userId - 사용자 ID
 * @param {Object} awardData - 수상 데이터
 * @param {string} awardData.competitionId - 공모전 ID
 * @param {string} awardData.competitionTitle - 공모전 제목
 * @param {string} awardData.rank - 수상 등급 (대상/금상/은상/동상/입선)
 * @param {Date} [awardData.awardedAt] - 수상일 (기본값: 현재 시간)
 * @returns {Promise<Object>} 생성된 수상 데이터
 * @throws {Error} 유효성 검사 실패 또는 Firestore 에러 시
 */
export const recordAward = async (userId, awardData) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  if (!awardData || !awardData.competitionId) {
    throw new Error('competitionId는 필수입니다');
  }

  if (!awardData.competitionTitle) {
    throw new Error('competitionTitle은 필수입니다');
  }

  if (!awardData.rank) {
    throw new Error('rank는 필수입니다');
  }

  if (!AWARD_RANKS.includes(awardData.rank)) {
    throw new Error('유효하지 않은 수상 등급입니다');
  }

  try {
    const awardedAt = awardData.awardedAt
      ? Timestamp.fromDate(awardData.awardedAt)
      : Timestamp.now();

    const awardDoc = {
      competitionId: awardData.competitionId,
      competitionTitle: awardData.competitionTitle,
      rank: awardData.rank,
      awardedAt,
    };

    const awardsRef = collection(db, USERS_COLLECTION, userId, AWARDS_SUBCOLLECTION);
    const docRef = await addDoc(awardsRef, awardDoc);

    return {
      id: docRef.id,
      ...awardDoc,
    };
  } catch (error) {
    console.error('수상 기록 실패:', error);
    throw new Error('수상 기록 중 오류가 발생했습니다');
  }
};

/**
 * 수상 기록 목록 조회
 * @param {string} userId - 사용자 ID
 * @returns {Promise<Array>} 수상 기록 목록
 * @throws {Error} userId가 없거나 Firestore 에러 시
 */
export const getAwards = async (userId) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  try {
    const awardsRef = collection(db, USERS_COLLECTION, userId, AWARDS_SUBCOLLECTION);
    const q = query(awardsRef, orderBy('awardedAt', 'desc'));
    const snapshot = await getDocs(q);

    return snapshot.docs.map((docSnap) => ({
      id: docSnap.id,
      ...docSnap.data(),
    }));
  } catch (error) {
    console.error('수상 기록 조회 실패:', error);
    throw new Error('수상 기록 조회 중 오류가 발생했습니다');
  }
};

/**
 * 공개 프로필 사용자의 수상 기록 조회
 * @param {string} userId - 사용자 ID
 * @returns {Promise<Array>} 수상 기록 목록 (비공개 프로필이면 빈 배열)
 * @throws {Error} userId가 없거나 Firestore 에러 시
 */
export const getPublicAwards = async (userId) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  try {
    // 사용자 프로필 공개 여부 확인
    const userDocRef = doc(db, USERS_COLLECTION, userId);
    const userDocSnap = await getDoc(userDocRef);

    if (!userDocSnap.exists()) {
      return [];
    }

    const userData = userDocSnap.data();
    if (!userData.isPublic) {
      return [];
    }

    // 공개 프로필인 경우 수상 기록 반환
    return await getAwards(userId);
  } catch (error) {
    console.error('공개 수상 기록 조회 실패:', error);
    throw new Error('수상 기록 조회 중 오류가 발생했습니다');
  }
};

const awardService = {
  recordAward,
  getAwards,
  getPublicAwards,
};

export default awardService;
