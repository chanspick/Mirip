/**
 * Activity 서비스
 *
 * 활동 기록 및 스트릭 계산 기능을 제공합니다.
 * SPEC-CRED-001 요구사항에 따라 구현되었습니다.
 *
 * @module services/activityService
 */

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
import { db } from '../config/firebase';
import { ACTIVITY_TYPES, DEFAULT_PAGE_SIZE } from '../types/credential.types';

const USERS_COLLECTION = 'users';
const ACTIVITIES_SUBCOLLECTION = 'activities';
const DAILY_ACTIVITIES_SUBCOLLECTION = 'dailyActivities';

/**
 * 날짜를 YYYY-MM-DD 형식으로 변환
 * @param {Date} date - 날짜 객체
 * @returns {string} YYYY-MM-DD 형식 문자열
 */
const formatDateToString = (date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

/**
 * 오늘 날짜 문자열 반환
 * @returns {string} YYYY-MM-DD 형식의 오늘 날짜
 */
const getTodayString = () => {
  return formatDateToString(new Date());
};

/**
 * 일별 활동 수 업데이트
 * @param {string} userId - 사용자 ID
 * @param {string} dateStr - YYYY-MM-DD 형식 날짜
 * @param {string} activityId - 활동 ID
 */
const updateDailyActivityCount = async (userId, dateStr, activityId) => {
  const dailyDocRef = doc(db, USERS_COLLECTION, userId, DAILY_ACTIVITIES_SUBCOLLECTION, dateStr);
  const dailyDocSnap = await getDoc(dailyDocRef);

  if (dailyDocSnap.exists()) {
    const data = dailyDocSnap.data();
    await setDoc(dailyDocRef, {
      date: dateStr,
      count: data.count + 1,
      activityIds: [...data.activityIds, activityId],
    });
  } else {
    await setDoc(dailyDocRef, {
      date: dateStr,
      count: 1,
      activityIds: [activityId],
    });
  }
};

/**
 * 사용자 통계 업데이트 (총 활동 수, 스트릭)
 * @param {string} userId - 사용자 ID
 */
const updateUserStats = async (userId) => {
  const userDocRef = doc(db, USERS_COLLECTION, userId);
  const userDocSnap = await getDoc(userDocRef);

  if (userDocSnap.exists()) {
    const userData = userDocSnap.data();
    const newTotalActivities = (userData.totalActivities || 0) + 1;

    // 스트릭 계산
    const streakData = await calculateStreak(userId);

    await updateDoc(userDocRef, {
      totalActivities: newTotalActivities,
      currentStreak: streakData.currentStreak,
      longestStreak: Math.max(userData.longestStreak || 0, streakData.currentStreak),
      updatedAt: Timestamp.now(),
    });
  }
};

/**
 * 활동 기록
 * @param {string} userId - 사용자 ID
 * @param {Object} activityData - 활동 데이터
 * @param {string} activityData.type - 활동 타입
 * @param {string} activityData.title - 활동 제목
 * @param {string} [activityData.description] - 활동 설명
 * @param {Object} [activityData.metadata] - 추가 메타데이터
 * @returns {Promise<Object>} 생성된 활동 데이터
 * @throws {Error} 유효성 검사 실패 또는 Firestore 에러 시
 */
export const recordActivity = async (userId, activityData) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  if (!activityData || !activityData.type) {
    throw new Error('활동 타입은 필수입니다');
  }

  if (!ACTIVITY_TYPES.includes(activityData.type)) {
    throw new Error('유효하지 않은 활동 타입입니다');
  }

  if (!activityData.title) {
    throw new Error('title은 필수입니다');
  }

  try {
    const now = Timestamp.now();
    const dateStr = getTodayString();

    const activityDoc = {
      userId,
      type: activityData.type,
      title: activityData.title,
      description: activityData.description || '',
      metadata: activityData.metadata || {},
      createdAt: now,
    };

    // 활동 추가
    const activitiesRef = collection(db, USERS_COLLECTION, userId, ACTIVITIES_SUBCOLLECTION);
    const docRef = await addDoc(activitiesRef, activityDoc);

    // 일별 활동 수 업데이트
    await updateDailyActivityCount(userId, dateStr, docRef.id);

    // 사용자 통계 업데이트
    await updateUserStats(userId);

    return {
      id: docRef.id,
      ...activityDoc,
    };
  } catch (error) {
    console.error('활동 기록 실패:', error);
    throw new Error('활동 기록 중 오류가 발생했습니다');
  }
};

/**
 * 활동 목록 조회
 * @param {string} userId - 사용자 ID
 * @param {Object} [options] - 조회 옵션
 * @param {number} [options.pageSize=20] - 페이지 크기
 * @param {Object} [options.lastDoc] - 페이지네이션용 마지막 문서
 * @param {string} [options.type] - 활동 타입 필터
 * @returns {Promise<{activities: Array, lastDoc: Object, hasMore: boolean}>}
 * @throws {Error} userId가 없거나 Firestore 에러 시
 */
export const getActivities = async (userId, options = {}) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  const { pageSize = DEFAULT_PAGE_SIZE, lastDoc, type } = options;

  try {
    const activitiesRef = collection(db, USERS_COLLECTION, userId, ACTIVITIES_SUBCOLLECTION);
    const constraints = [];

    // 타입 필터
    if (type) {
      constraints.push(where('type', '==', type));
    }

    // 정렬 (최신순)
    constraints.push(orderBy('createdAt', 'desc'));

    // 페이지 크기 (+1로 다음 페이지 존재 여부 확인)
    constraints.push(limit(pageSize + 1));

    // 페이지네이션
    if (lastDoc) {
      constraints.push(startAfter(lastDoc));
    }

    const q = query(activitiesRef, ...constraints);
    const snapshot = await getDocs(q);

    const activities = snapshot.docs.slice(0, pageSize).map((docSnap) => ({
      id: docSnap.id,
      ...docSnap.data(),
    }));

    const hasMore = snapshot.docs.length > pageSize;
    const newLastDoc = snapshot.docs.length > 0
      ? snapshot.docs[Math.min(snapshot.docs.length - 1, pageSize - 1)]
      : null;

    return {
      activities,
      lastDoc: newLastDoc,
      hasMore,
    };
  } catch (error) {
    console.error('활동 목록 조회 실패:', error);
    throw new Error('활동 목록 조회 중 오류가 발생했습니다');
  }
};

/**
 * 활동 히트맵 데이터 조회 (1년간)
 * @param {string} userId - 사용자 ID
 * @param {number} [year] - 조회할 연도 (기본값: 현재 연도)
 * @returns {Promise<{year: number, days: Array, totalActivities: number}>}
 * @throws {Error} userId가 없거나 Firestore 에러 시
 */
export const getActivityHeatmapData = async (userId, year) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  const targetYear = year || new Date().getFullYear();

  try {
    const startDate = `${targetYear}-01-01`;
    const endDate = `${targetYear}-12-31`;

    const dailyRef = collection(db, USERS_COLLECTION, userId, DAILY_ACTIVITIES_SUBCOLLECTION);
    const q = query(
      dailyRef,
      where('date', '>=', startDate),
      where('date', '<=', endDate)
    );

    const snapshot = await getDocs(q);

    // 일별 데이터를 Map으로 변환
    const dailyMap = new Map();
    let totalActivities = 0;

    snapshot.docs.forEach((docSnap) => {
      const data = docSnap.data();
      dailyMap.set(data.date, data);
      totalActivities += data.count;
    });

    // 365일 전체 데이터 생성
    const days = [];
    const startOfYear = new Date(targetYear, 0, 1);
    const endOfYear = new Date(targetYear, 11, 31);

    for (let d = new Date(startOfYear); d <= endOfYear; d.setDate(d.getDate() + 1)) {
      const dateStr = formatDateToString(new Date(d));
      const dayData = dailyMap.get(dateStr) || { date: dateStr, count: 0, activityIds: [] };
      days.push(dayData);
    }

    return {
      year: targetYear,
      days,
      totalActivities,
    };
  } catch (error) {
    console.error('히트맵 데이터 조회 실패:', error);
    throw new Error('히트맵 데이터 조회 중 오류가 발생했습니다');
  }
};

/**
 * 스트릭 계산
 * @param {string} userId - 사용자 ID
 * @returns {Promise<{currentStreak: number, longestStreak: number}>}
 * @throws {Error} userId가 없거나 Firestore 에러 시
 */
export const calculateStreak = async (userId) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  try {
    const dailyRef = collection(db, USERS_COLLECTION, userId, DAILY_ACTIVITIES_SUBCOLLECTION);
    const q = query(
      dailyRef,
      orderBy('date', 'desc'),
      limit(365) // 최대 1년치
    );

    const snapshot = await getDocs(q);

    if (snapshot.docs.length === 0) {
      return { currentStreak: 0, longestStreak: 0 };
    }

    // 날짜 정렬 (최신순)
    const dates = snapshot.docs
      .map((doc) => doc.data().date)
      .sort((a, b) => b.localeCompare(a));

    const today = getTodayString();
    const yesterday = formatDateToString(new Date(Date.now() - 86400000));

    let currentStreak = 0;
    let longestStreak = 0;
    let tempStreak = 0;
    let expectedDate = today;

    // 오늘 또는 어제부터 시작해야 현재 스트릭으로 인정
    const firstDate = dates[0];
    const streakStarted = firstDate === today || firstDate === yesterday;

    if (streakStarted) {
      expectedDate = firstDate;
    }

    for (const date of dates) {
      if (date === expectedDate) {
        tempStreak++;
        if (streakStarted && date === expectedDate) {
          currentStreak = tempStreak;
        }
        // 이전 날짜 계산
        const prevDate = new Date(expectedDate);
        prevDate.setDate(prevDate.getDate() - 1);
        expectedDate = formatDateToString(prevDate);
      } else if (date < expectedDate) {
        // 연속이 끊김
        longestStreak = Math.max(longestStreak, tempStreak);
        tempStreak = 1;
        if (!streakStarted) {
          currentStreak = 0;
        }
        // 새로운 시작점
        const prevDate = new Date(date);
        prevDate.setDate(prevDate.getDate() - 1);
        expectedDate = formatDateToString(prevDate);
      }
    }

    longestStreak = Math.max(longestStreak, tempStreak);

    return {
      currentStreak,
      longestStreak,
    };
  } catch (error) {
    console.error('스트릭 계산 실패:', error);
    throw new Error('스트릭 계산 중 오류가 발생했습니다');
  }
};

/**
 * 특정 날짜의 활동 수 조회
 * @param {string} userId - 사용자 ID
 * @param {string} [date] - YYYY-MM-DD 형식 날짜 (기본값: 오늘)
 * @returns {Promise<{date: string, count: number, activityIds: Array}>}
 * @throws {Error} userId가 없거나 Firestore 에러 시
 */
export const getDailyActivityCount = async (userId, date) => {
  if (!userId) {
    throw new Error('userId는 필수입니다');
  }

  const targetDate = date || getTodayString();

  try {
    const dailyDocRef = doc(db, USERS_COLLECTION, userId, DAILY_ACTIVITIES_SUBCOLLECTION, targetDate);
    const dailyDocSnap = await getDoc(dailyDocRef);

    if (!dailyDocSnap.exists()) {
      return {
        date: targetDate,
        count: 0,
        activityIds: [],
      };
    }

    return dailyDocSnap.data();
  } catch (error) {
    console.error('일별 활동 수 조회 실패:', error);
    throw new Error('일별 활동 수 조회 중 오류가 발생했습니다');
  }
};

const activityService = {
  recordActivity,
  getActivities,
  getActivityHeatmapData,
  calculateStreak,
  getDailyActivityCount,
};

export default activityService;
