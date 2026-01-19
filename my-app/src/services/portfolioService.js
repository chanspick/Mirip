/**
 * Portfolio 서비스
 *
 * 포트폴리오 작품 관리 기능을 제공합니다.
 * SPEC-CRED-001 요구사항에 따라 구현되었습니다.
 *
 * @module services/portfolioService
 */

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
import { db } from '../config/firebase';
import { PORTFOLIO_PAGE_SIZE } from '../types/credential.types';

const USERS_COLLECTION = 'users';
const PORTFOLIOS_SUBCOLLECTION = 'portfolios';

/**
 * 포트폴리오 추가
 * @param {string} userId - 사용자 ID
 * @param {Object} portfolioData - 포트폴리오 데이터
 * @param {string} portfolioData.title - 작품 제목
 * @param {string} [portfolioData.description] - 작품 설명
 * @param {string} portfolioData.imageUrl - 원본 이미지 URL
 * @param {string} [portfolioData.thumbnailUrl] - 썸네일 이미지 URL
 * @param {string[]} [portfolioData.tags] - 태그 목록
 * @param {boolean} [portfolioData.isPublic=true] - 공개 여부
 * @returns {Promise<Object>} 생성된 포트폴리오 데이터
 * @throws {Error} 유효성 검사 실패 또는 Firestore 에러 시
 */
export const addPortfolio = async (userId, portfolioData) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  if (!portfolioData || !portfolioData.title) {
    throw new Error('title은 필수입니다');
  }

  if (!portfolioData.imageUrl) {
    throw new Error('imageUrl은 필수입니다');
  }

  try {
    const now = Timestamp.now();

    const portfolioDoc = {
      userId,
      title: portfolioData.title,
      description: portfolioData.description || '',
      imageUrl: portfolioData.imageUrl,
      thumbnailUrl: portfolioData.thumbnailUrl || null,
      tags: portfolioData.tags || [],
      isPublic: portfolioData.isPublic !== undefined ? portfolioData.isPublic : true,
      createdAt: now,
      updatedAt: now,
    };

    const portfoliosRef = collection(db, USERS_COLLECTION, userId, PORTFOLIOS_SUBCOLLECTION);
    const docRef = await addDoc(portfoliosRef, portfolioDoc);

    return {
      id: docRef.id,
      ...portfolioDoc,
    };
  } catch (error) {
    console.error('포트폴리오 추가 실패:', error);
    throw new Error('포트폴리오 추가 중 오류가 발생했습니다');
  }
};

/**
 * 포트폴리오 업데이트
 * @param {string} userId - 사용자 ID
 * @param {string} portfolioId - 포트폴리오 ID
 * @param {Object} updateData - 업데이트할 데이터
 * @returns {Promise<void>}
 * @throws {Error} 유효성 검사 실패, 권한 없음 또는 Firestore 에러 시
 */
export const updatePortfolio = async (userId, portfolioId, updateData) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  if (!portfolioId) {
    throw new Error('portfolioId는 필수입니다');
  }

  if (!updateData || Object.keys(updateData).length === 0) {
    throw new Error('업데이트할 데이터가 없습니다');
  }

  try {
    const portfolioRef = doc(db, USERS_COLLECTION, userId, PORTFOLIOS_SUBCOLLECTION, portfolioId);
    const portfolioSnap = await getDoc(portfolioRef);

    if (!portfolioSnap.exists()) {
      throw new Error('포트폴리오를 찾을 수 없습니다');
    }

    const existingData = portfolioSnap.data();
    if (existingData.userId !== userId) {
      throw new Error('포트폴리오 수정 권한이 없습니다');
    }

    await updateDoc(portfolioRef, {
      ...updateData,
      updatedAt: Timestamp.now(),
    });
  } catch (error) {
    // 이미 에러 메시지가 설정된 경우 그대로 throw
    if (error.message.includes('찾을 수 없습니다') ||
        error.message.includes('권한이 없습니다')) {
      throw error;
    }
    console.error('포트폴리오 업데이트 실패:', error);
    throw new Error('포트폴리오 업데이트 중 오류가 발생했습니다');
  }
};

/**
 * 포트폴리오 삭제
 * @param {string} userId - 사용자 ID
 * @param {string} portfolioId - 포트폴리오 ID
 * @returns {Promise<void>}
 * @throws {Error} 유효성 검사 실패, 권한 없음 또는 Firestore 에러 시
 */
export const deletePortfolio = async (userId, portfolioId) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  if (!portfolioId) {
    throw new Error('portfolioId는 필수입니다');
  }

  try {
    const portfolioRef = doc(db, USERS_COLLECTION, userId, PORTFOLIOS_SUBCOLLECTION, portfolioId);
    const portfolioSnap = await getDoc(portfolioRef);

    if (!portfolioSnap.exists()) {
      throw new Error('포트폴리오를 찾을 수 없습니다');
    }

    const existingData = portfolioSnap.data();
    if (existingData.userId !== userId) {
      throw new Error('포트폴리오 삭제 권한이 없습니다');
    }

    await deleteDoc(portfolioRef);
  } catch (error) {
    // 이미 에러 메시지가 설정된 경우 그대로 throw
    if (error.message.includes('찾을 수 없습니다') ||
        error.message.includes('권한이 없습니다')) {
      throw error;
    }
    console.error('포트폴리오 삭제 실패:', error);
    throw new Error('포트폴리오 삭제 중 오류가 발생했습니다');
  }
};

/**
 * 포트폴리오 목록 조회
 * @param {string} userId - 사용자 ID
 * @param {Object} [options] - 조회 옵션
 * @param {number} [options.pageSize=12] - 페이지 크기
 * @param {Object} [options.lastDoc] - 페이지네이션용 마지막 문서
 * @param {boolean} [options.publicOnly=false] - 공개 작품만 조회
 * @returns {Promise<{portfolios: Array, lastDoc: Object, hasMore: boolean}>}
 * @throws {Error} userId가 없거나 Firestore 에러 시
 */
export const getPortfolios = async (userId, options = {}) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  const { pageSize = PORTFOLIO_PAGE_SIZE, lastDoc, publicOnly = false } = options;

  try {
    const portfoliosRef = collection(db, USERS_COLLECTION, userId, PORTFOLIOS_SUBCOLLECTION);
    const constraints = [];

    // 공개 여부 필터
    if (publicOnly) {
      constraints.push(where('isPublic', '==', true));
    }

    // 정렬 (최신순)
    constraints.push(orderBy('createdAt', 'desc'));

    // 페이지 크기 (+1로 다음 페이지 존재 여부 확인)
    constraints.push(limit(pageSize + 1));

    // 페이지네이션
    if (lastDoc) {
      constraints.push(startAfter(lastDoc));
    }

    const q = query(portfoliosRef, ...constraints);
    const snapshot = await getDocs(q);

    const portfolios = snapshot.docs.slice(0, pageSize).map((docSnap) => ({
      id: docSnap.id,
      ...docSnap.data(),
    }));

    const hasMore = snapshot.docs.length > pageSize;
    const newLastDoc = snapshot.docs.length > 0
      ? snapshot.docs[Math.min(snapshot.docs.length - 1, pageSize - 1)]
      : null;

    return {
      portfolios,
      lastDoc: newLastDoc,
      hasMore,
    };
  } catch (error) {
    console.error('포트폴리오 목록 조회 실패:', error);
    throw new Error('포트폴리오 목록 조회 중 오류가 발생했습니다');
  }
};

/**
 * 공개 포트폴리오 목록 조회
 * @param {string} userId - 사용자 ID
 * @param {Object} [options] - 조회 옵션
 * @param {number} [options.pageSize=12] - 페이지 크기
 * @param {Object} [options.lastDoc] - 페이지네이션용 마지막 문서
 * @returns {Promise<{portfolios: Array, lastDoc: Object, hasMore: boolean}>}
 * @throws {Error} userId가 없거나 Firestore 에러 시
 */
export const getPublicPortfolios = async (userId, options = {}) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  return getPortfolios(userId, { ...options, publicOnly: true });
};

const portfolioService = {
  addPortfolio,
  updatePortfolio,
  deletePortfolio,
  getPortfolios,
  getPublicPortfolios,
};

export default portfolioService;
