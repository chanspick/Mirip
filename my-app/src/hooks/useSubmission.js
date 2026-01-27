/**
 * 출품 관련 커스텀 훅
 *
 * @module hooks/useSubmission
 */

import { useState, useEffect, useCallback } from 'react';
import {
  uploadImage,
  createSubmission,
  getSubmissionsByCompetition,
  getUserSubmission,
} from '../services/submissionService';

/**
 * 이미지 업로드 훅
 * @returns {Object} 업로드 함수, 진행률, 에러
 */
export const useImageUpload = () => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);

  const upload = useCallback(async (file, competitionId, userId) => {
    setUploading(true);
    setProgress(0);
    setError(null);

    try {
      const url = await uploadImage(file, competitionId, userId, (p) => {
        setProgress(p);
      });
      setImageUrl(url);
      return url;
    } catch (err) {
      setError(err.message || '이미지 업로드에 실패했습니다.');
      throw err;
    } finally {
      setUploading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setProgress(0);
    setError(null);
    setImageUrl(null);
  }, []);

  return { upload, uploading, progress, error, imageUrl, reset };
};

/**
 * 출품 생성 훅
 * @returns {Object} 출품 함수, 로딩 상태, 에러
 */
export const useCreateSubmission = () => {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);
  const [submissionId, setSubmissionId] = useState(null);

  const submit = useCallback(async (submissionData) => {
    setSubmitting(true);
    setError(null);

    try {
      const id = await createSubmission(submissionData);
      setSubmissionId(id);
      return id;
    } catch (err) {
      setError(err.message || '출품에 실패했습니다.');
      throw err;
    } finally {
      setSubmitting(false);
    }
  }, []);

  return { submit, submitting, error, submissionId };
};

/**
 * 공모전별 출품작 목록 조회 훅
 * @param {string} competitionId - 공모전 ID
 * @returns {Object} 출품작 목록, 로딩 상태, 에러
 */
export const useSubmissionList = (competitionId) => {
  const [submissions, setSubmissions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSubmissions = async () => {
      if (!competitionId) {
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const result = await getSubmissionsByCompetition(competitionId);
        setSubmissions(result);
      } catch (err) {
        setError(err.message || '출품작 목록을 불러오는데 실패했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchSubmissions();
  }, [competitionId]);

  return { submissions, loading, error };
};

/**
 * 사용자 출품 여부 확인 훅
 * @param {string} competitionId - 공모전 ID
 * @param {string} userId - 사용자 ID
 * @returns {Object} 출품 정보, 로딩 상태
 */
export const useUserSubmission = (competitionId, userId) => {
  const [submission, setSubmission] = useState(null);
  const [loading, setLoading] = useState(true);
  const [hasSubmitted, setHasSubmitted] = useState(false);

  useEffect(() => {
    const checkSubmission = async () => {
      if (!competitionId || !userId) {
        setLoading(false);
        return;
      }

      setLoading(true);

      try {
        const result = await getUserSubmission(competitionId, userId);
        setSubmission(result);
        setHasSubmitted(!!result);
      } catch (err) {
        console.error('출품 여부 확인 실패:', err);
      } finally {
        setLoading(false);
      }
    };

    checkSubmission();
  }, [competitionId, userId]);

  return { submission, loading, hasSubmitted };
};

const submissionHooks = {
  useImageUpload,
  useCreateSubmission,
  useSubmissionList,
  useUserSubmission,
};
export default submissionHooks;
