/**
 * 출품 서비스
 *
 * Firestore submissions 컬렉션 CRUD 및 Storage 업로드 함수
 * SPEC-COMP-001 요구사항에 따라 구현
 * SPEC-CRED-001 M5: 활동 기록 연동 추가
 *
 * @module services/submissionService
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
  Timestamp,
} from 'firebase/firestore';
import {
  ref,
  uploadBytesResumable,
  getDownloadURL,
  deleteObject,
} from 'firebase/storage';
import { db, storage } from '../config/firebase';
import { incrementParticipantCount } from './competitionService';
import { recordSubmissionActivity } from './integrationService';

const COLLECTION_NAME = 'submissions';
const STORAGE_PATH = 'submissions';

/**
 * 이미지 업로드
 * @param {File} file - 업로드할 파일
 * @param {string} competitionId - 공모전 ID
 * @param {string} userId - 사용자 ID
 * @param {Function} onProgress - 진행률 콜백
 * @returns {Promise<string>} 다운로드 URL
 */
export const uploadImage = async (file, competitionId, userId, onProgress) => {
  // 파일 유효성 검사
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowedTypes.includes(file.type)) {
    throw new Error('지원하지 않는 파일 형식입니다. (JPG, PNG, WebP만 가능)');
  }

  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    throw new Error('파일 크기가 10MB를 초과합니다.');
  }

  // 고유 파일명 생성
  const timestamp = Date.now();
  const extension = file.name.split('.').pop();
  const fileName = `${competitionId}/${userId}/${timestamp}.${extension}`;
  const storageRef = ref(storage, `${STORAGE_PATH}/${fileName}`);

  return new Promise((resolve, reject) => {
    const uploadTask = uploadBytesResumable(storageRef, file);

    uploadTask.on(
      'state_changed',
      (snapshot) => {
        const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
        if (onProgress) {
          onProgress(progress);
        }
      },
      (error) => {
        console.error('이미지 업로드 실패:', error);
        reject(error);
      },
      async () => {
        try {
          const downloadURL = await getDownloadURL(uploadTask.snapshot.ref);
          resolve(downloadURL);
        } catch (error) {
          reject(error);
        }
      }
    );
  });
};

/**
 * 이미지 삭제
 * @param {string} imageUrl - 삭제할 이미지 URL
 */
export const deleteImage = async (imageUrl) => {
  try {
    const imageRef = ref(storage, imageUrl);
    await deleteObject(imageRef);
  } catch (error) {
    console.error('이미지 삭제 실패:', error);
    // 이미지 삭제 실패는 무시 (이미 삭제되었을 수 있음)
  }
};

/**
 * 출품작 생성
 * @param {Object} submissionData - 출품 데이터
 * @param {string} submissionData.competitionId - 공모전 ID
 * @param {string} submissionData.userId - 사용자 ID
 * @param {string} [submissionData.competitionTitle] - 공모전 제목 (활동 기록용)
 * @param {string} [submissionData.imageUrl] - 출품 이미지 URL
 * @returns {Promise<string>} 생성된 문서 ID
 */
export const createSubmission = async (submissionData) => {
  try {
    const docRef = await addDoc(collection(db, COLLECTION_NAME), {
      ...submissionData,
      status: 'pending',
      createdAt: Timestamp.now(),
      updatedAt: Timestamp.now(),
    });

    // 공모전 참가자 수 증가
    await incrementParticipantCount(submissionData.competitionId);

    // SPEC-CRED-001 M5: 활동 기록 (non-blocking)
    // userId와 competitionTitle이 있으면 활동을 기록
    if (submissionData.userId && submissionData.competitionTitle) {
      recordSubmissionActivity(submissionData.userId, {
        competitionId: submissionData.competitionId,
        competitionTitle: submissionData.competitionTitle,
        submissionId: docRef.id,
        imageUrl: submissionData.imageUrl || null,
      }).catch((err) => {
        console.warn('[SubmissionService] 활동 기록 실패 (무시됨):', err);
      });
    }

    return docRef.id;
  } catch (error) {
    console.error('출품 생성 실패:', error);
    throw error;
  }
};

/**
 * 출품작 조회 (공모전별)
 * @param {string} competitionId - 공모전 ID
 * @returns {Promise<Array>}
 */
export const getSubmissionsByCompetition = async (competitionId) => {
  try {
    const q = query(
      collection(db, COLLECTION_NAME),
      where('competitionId', '==', competitionId),
      orderBy('createdAt', 'desc')
    );
    const snapshot = await getDocs(q);

    return snapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    }));
  } catch (error) {
    console.error('출품작 목록 조회 실패:', error);
    throw error;
  }
};

/**
 * 출품작 조회 (사용자별)
 * @param {string} userId - 사용자 ID
 * @returns {Promise<Array>}
 */
export const getSubmissionsByUser = async (userId) => {
  try {
    const q = query(
      collection(db, COLLECTION_NAME),
      where('userId', '==', userId),
      orderBy('createdAt', 'desc')
    );
    const snapshot = await getDocs(q);

    return snapshot.docs.map((doc) => ({
      id: doc.id,
      ...doc.data(),
    }));
  } catch (error) {
    console.error('사용자 출품작 조회 실패:', error);
    throw error;
  }
};

/**
 * 사용자의 특정 공모전 출품 여부 확인
 * @param {string} competitionId - 공모전 ID
 * @param {string} userId - 사용자 ID
 * @returns {Promise<Object|null>} 출품 정보 또는 null
 */
export const getUserSubmission = async (competitionId, userId) => {
  try {
    const q = query(
      collection(db, COLLECTION_NAME),
      where('competitionId', '==', competitionId),
      where('userId', '==', userId)
    );
    const snapshot = await getDocs(q);

    if (snapshot.empty) {
      return null;
    }

    const doc = snapshot.docs[0];
    return {
      id: doc.id,
      ...doc.data(),
    };
  } catch (error) {
    console.error('출품 여부 확인 실패:', error);
    throw error;
  }
};

/**
 * 출품작 상세 조회
 * @param {string} id - 출품 ID
 * @returns {Promise<Object|null>}
 */
export const getSubmissionById = async (id) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, id);
    const docSnap = await getDoc(docRef);

    if (!docSnap.exists()) {
      return null;
    }

    return {
      id: docSnap.id,
      ...docSnap.data(),
    };
  } catch (error) {
    console.error('출품 상세 조회 실패:', error);
    throw error;
  }
};

/**
 * 출품작 수정
 * @param {string} id - 출품 ID
 * @param {Object} updateData - 수정할 데이터
 */
export const updateSubmission = async (id, updateData) => {
  try {
    const docRef = doc(db, COLLECTION_NAME, id);
    await updateDoc(docRef, {
      ...updateData,
      updatedAt: Timestamp.now(),
    });
  } catch (error) {
    console.error('출품 수정 실패:', error);
    throw error;
  }
};

/**
 * 출품작 삭제
 * @param {string} id - 출품 ID
 * @param {string} imageUrl - 삭제할 이미지 URL (선택)
 */
export const deleteSubmission = async (id, imageUrl) => {
  try {
    // 이미지 삭제
    if (imageUrl) {
      await deleteImage(imageUrl);
    }

    // 문서 삭제
    const docRef = doc(db, COLLECTION_NAME, id);
    await deleteDoc(docRef);
  } catch (error) {
    console.error('출품 삭제 실패:', error);
    throw error;
  }
};

const submissionService = {
  uploadImage,
  deleteImage,
  createSubmission,
  getSubmissionsByCompetition,
  getSubmissionsByUser,
  getUserSubmission,
  getSubmissionById,
  updateSubmission,
  deleteSubmission,
};
export default submissionService;
