/**
 * Registration Service
 *
 * Firestore에 등록 데이터를 저장하는 서비스입니다.
 * SPEC-FIREBASE-001 요구사항에 따라 구현되었습니다.
 *
 * @module services/registrationService
 */

import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import { db } from '../config/firebase';

/**
 * 이메일 형식 유효성 검사 정규식
 * @type {RegExp}
 */
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * 필수 필드 목록
 * @type {string[]}
 */
const REQUIRED_FIELDS = ['name', 'email'];

/**
 * 입력 데이터 유효성 검사
 *
 * @param {Object} data - 검증할 데이터 객체
 * @throws {Error} 데이터가 null이거나 undefined인 경우
 * @throws {Error} 필수 필드가 누락된 경우
 * @throws {Error} 이메일 형식이 유효하지 않은 경우
 */
const validateData = (data) => {
  // null/undefined 체크
  if (data === null || data === undefined) {
    throw new Error('유효하지 않은 데이터입니다');
  }

  // 필수 필드 체크
  const missingFields = REQUIRED_FIELDS.filter(
    (field) => !data[field] || data[field].trim() === ''
  );

  if (missingFields.length > 0) {
    throw new Error('필수 필드가 누락되었습니다');
  }

  // 이메일 형식 체크
  if (!EMAIL_REGEX.test(data.email)) {
    throw new Error('유효하지 않은 이메일 형식입니다');
  }
};

/**
 * 새로운 등록 데이터를 Firestore에 저장합니다.
 *
 * @param {Object} data - 저장할 등록 데이터
 * @param {string} data.name - 이름 (필수)
 * @param {string} data.email - 이메일 (필수)
 * @param {string} [data.phone] - 전화번호 (선택)
 * @param {string} [data.company] - 회사명 (선택)
 * @returns {Promise<Object>} 생성된 등록 데이터 (ID 포함)
 * @throws {Error} 유효성 검사 실패 시
 * @throws {Error} Firestore 저장 실패 시
 *
 * @example
 * const registration = await create({
 *   name: '홍길동',
 *   email: 'hong@example.com',
 *   phone: '010-1234-5678'
 * });
 * console.log(registration.id); // 'abc123...'
 */
const create = async (data) => {
  // 입력 데이터 유효성 검사
  validateData(data);

  try {
    // 저장할 데이터 준비 (서버 타임스탬프 추가)
    const registrationData = {
      ...data,
      timestamp: serverTimestamp(),
    };

    // Firestore에 저장
    const registrationsRef = collection(db, 'registrations');
    const docRef = await addDoc(registrationsRef, registrationData);

    // 생성된 문서 ID와 함께 반환
    return {
      id: docRef.id,
      ...data,
    };
  } catch (error) {
    // Firestore 에러를 사용자 친화적인 메시지로 변환
    console.error('Registration Service Error:', error);
    throw new Error('등록 저장 중 오류가 발생했습니다');
  }
};

export { create };
