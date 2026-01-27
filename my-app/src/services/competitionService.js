/**
 * 공모전 서비스
 *
 * Firestore competitions 컬렉션 CRUD 및 쿼리 함수
 * SPEC-COMP-001 요구사항에 따라 구현
 *
 * @module services/competitionService
 */

import {
  collection,
  doc,
  getDocs,
  getDoc,
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

const COLLECTION_NAME = 'competitions';
const PAGE_SIZE = 12;

/**
 * 공모전 상태 계산
 * @param {Date} startDate - 시작일
 * @param {Date} endDate - 종료일
 * @returns {'upcoming' | 'active' | 'ending_soon' | 'ended'}
 */
const calculateStatus = (startDate, endDate) => {
  const now = new Date();
  const end = endDate instanceof Timestamp ? endDate.toDate() : new Date(endDate);
  const start = startDate instanceof Timestamp ? startDate.toDate() : new Date(startDate);

  if (now < start) return 'upcoming';
  if (now > end) return 'ended';

  // D-7 이내면 마감임박
  const daysUntilEnd = Math.ceil((end - now) / (1000 * 60 * 60 * 24));
  if (daysUntilEnd <= 7) return 'ending_soon';

  return 'active';
};

/**
 * D-Day 계산
 * @param {Date|Timestamp} endDate - 종료일
 * @returns {number} D-Day (음수: 종료됨)
 */
const calculateDDay = (endDate) => {
  const now = new Date();
  const end = endDate instanceof Timestamp ? endDate.toDate() : new Date(endDate);
  return Math.ceil((end - now) / (1000 * 60 * 60 * 24));
};

/**
 * 공모전 목록 조회
 * @param {Object} options - 조회 옵션
 * @param {string} [options.category] - 분야 필터
 * @param {string} [options.status] - 상태 필터
 * @param {string} [options.sortBy='endDate'] - 정렬 기준
 * @param {string} [options.sortOrder='asc'] - 정렬 순서
 * @param {DocumentSnapshot} [options.lastDoc] - 페이지네이션용 마지막 문서
 * @returns {Promise<{competitions: Array, lastDoc: DocumentSnapshot, hasMore: boolean}>}
 */
export const getCompetitions = async (options = {}) => {
  const {
    category,
    status,
    sortBy = 'endDate',
    sortOrder = 'asc',
    lastDoc,
  } = options;

  try {
    let q = collection(db, COLLECTION_NAME);
    const constraints = [];

    // 분야 필터
    if (category && category !== 'all') {
      constraints.push(where('category', '==', category));
    }

    // 정렬
    const sortField = sortBy === 'prize' ? 'prize' :
                      sortBy === 'popularity' ? 'participantCount' :
                      sortBy === 'createdAt' ? 'createdAt' : 'endDate';
    const order = sortBy === 'endDate' ? 'asc' : sortOrder;
    constraints.push(orderBy(sortField, order));

    // 페이지네이션
    constraints.push(limit(PAGE_SIZE + 1));
    if (lastDoc) {
      constraints.push(startAfter(lastDoc));
    }

    q = query(q, ...constraints);
    const snapshot = await getDocs(q);

    let competitions = snapshot.docs.map((doc) => {
      const data = doc.data();
      const computedStatus = calculateStatus(data.startDate, data.endDate);
      return {
        id: doc.id,
        ...data,
        status: computedStatus,
        dDay: calculateDDay(data.endDate),
      };
    });

    // 상태 필터 (클라이언트 사이드)
    if (status && status !== 'all') {
      competitions = competitions.filter((comp) => {
        if (status === 'active') return comp.status === 'active' || comp.status === 'ending_soon';
        if (status === 'ending_soon') return comp.status === 'ending_soon';
        if (status === 'ended') return comp.status === 'ended';
        return true;
      });
    }

    const hasMore = competitions.length > PAGE_SIZE;
    if (hasMore) {
      competitions = competitions.slice(0, PAGE_SIZE);
    }

    return {
      competitions,
      lastDoc: snapshot.docs[Math.min(snapshot.docs.length - 1, PAGE_SIZE - 1)],
      hasMore,
    };
  } catch (error) {
    console.error('공모전 목록 조회 실패:', error);
    throw error;
  }
};

/**
 * 공모전 상세 조회
 * @param {string} id - 공모전 ID
 * @returns {Promise<Object|null>}
 */
export const getCompetitionById = async (id) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, id);
    const docSnap = await getDoc(docRef);

    if (!docSnap.exists()) {
      return null;
    }

    const data = docSnap.data();
    return {
      id: docSnap.id,
      ...data,
      status: calculateStatus(data.startDate, data.endDate),
      dDay: calculateDDay(data.endDate),
    };
  } catch (error) {
    console.error('공모전 상세 조회 실패:', error);
    throw error;
  }
};

/**
 * 공모전 생성
 * @param {Object} competitionData - 공모전 데이터
 * @returns {Promise<string>} 생성된 문서 ID
 */
export const createCompetition = async (competitionData) => {
  try {
    const docRef = await addDoc(collection(db, COLLECTION_NAME), {
      ...competitionData,
      participantCount: 0,
      createdAt: Timestamp.now(),
      updatedAt: Timestamp.now(),
    });
    return docRef.id;
  } catch (error) {
    console.error('공모전 생성 실패:', error);
    throw error;
  }
};

/**
 * 공모전 수정
 * @param {string} id - 공모전 ID
 * @param {Object} updateData - 수정할 데이터
 */
export const updateCompetition = async (id, updateData) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, id);
    await updateDoc(docRef, {
      ...updateData,
      updatedAt: Timestamp.now(),
    });
  } catch (error) {
    console.error('공모전 수정 실패:', error);
    throw error;
  }
};

/**
 * 공모전 삭제
 * @param {string} id - 공모전 ID
 */
export const deleteCompetition = async (id) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, id);
    await deleteDoc(docRef);
  } catch (error) {
    console.error('공모전 삭제 실패:', error);
    throw error;
  }
};

/**
 * 참가자 수 증가
 * @param {string} id - 공모전 ID
 */
export const incrementParticipantCount = async (id) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, id);
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
      const currentCount = docSnap.data().participantCount || 0;
      await updateDoc(docRef, {
        participantCount: currentCount + 1,
        updatedAt: Timestamp.now(),
      });
    }
  } catch (error) {
    console.error('참가자 수 증가 실패:', error);
    throw error;
  }
};

const competitionService = {
  getCompetitions,
  getCompetitionById,
  createCompetition,
  updateCompetition,
  deleteCompetition,
  incrementParticipantCount,
};
export default competitionService;
