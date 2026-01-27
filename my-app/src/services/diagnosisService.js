/**
 * AI 진단 서비스
 *
 * Backend API (POST /api/v1/evaluate)와 통신하여 AI 작품 평가를 수행
 * SPEC-AI-001 요구사항에 따라 구현
 * SPEC-CRED-001 M5: 활동 기록 연동 추가
 *
 * @module services/diagnosisService
 */

import { recordDiagnosisActivity } from './integrationService';

// API 기본 URL 설정
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 기본 타임아웃 (30초)
const DEFAULT_TIMEOUT = 30000;

// Mock 모드 활성화 여부
const USE_MOCK = process.env.REACT_APP_USE_MOCK === 'true';

/**
 * API 에러 클래스
 */
export class DiagnosisAPIError extends Error {
  constructor(message, statusCode = null, details = null) {
    super(message);
    this.name = 'DiagnosisAPIError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

/**
 * 네트워크 에러 클래스
 */
export class NetworkError extends Error {
  constructor(message = '네트워크 연결을 확인해주세요.') {
    super(message);
    this.name = 'NetworkError';
  }
}

/**
 * 타임아웃 에러 클래스
 */
export class TimeoutError extends Error {
  constructor(message = '요청 시간이 초과되었습니다. 다시 시도해주세요.') {
    super(message);
    this.name = 'TimeoutError';
  }
}

/**
 * 타임아웃을 적용한 fetch 함수
 * @param {string} url - 요청 URL
 * @param {RequestInit} options - fetch 옵션
 * @param {number} timeout - 타임아웃 (ms)
 * @returns {Promise<Response>}
 */
const fetchWithTimeout = async (url, options, timeout = DEFAULT_TIMEOUT) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new TimeoutError();
    }
    throw new NetworkError();
  }
};

/**
 * API 응답을 UI 형식으로 변환
 *
 * API 응답 형식:
 * {
 *   "evaluation_id": "uuid",
 *   "tier": "A",
 *   "scores": {
 *     "composition": 75.5,
 *     "technique": 80.2,
 *     "creativity": 72.1,
 *     "completeness": 78.9
 *   },
 *   "probabilities": [
 *     {"university": "홍익대학교", "department": "시각디자인과", "probability": 0.72}
 *   ],
 *   "feedback": {
 *     "strengths": ["..."],
 *     "improvements": ["..."],
 *     "overall": "..."
 *   }
 * }
 *
 * UI 형식:
 * {
 *   "tier": "A",
 *   "universityScores": [{ "university": "홍익대", "score": 72 }],
 *   "axisScores": { "composition": 75, "color": 80, "technique": 72, "creativity": 78 },
 *   "feedback": { "strengths": [...], "improvements": [...] }
 * }
 *
 * @param {Object} apiResponse - API 응답
 * @returns {Object} UI 형식으로 변환된 결과
 */
const mapApiResponseToUIFormat = (apiResponse) => {
  const { tier, scores, probabilities, feedback, evaluation_id } = apiResponse;

  // 대학별 점수 변환 (probability 0-1 -> score 0-100)
  const universityScores = (probabilities || []).map((prob) => ({
    university: prob.university.replace('대학교', '대'), // 홍익대학교 -> 홍익대
    score: Math.round(prob.probability * 100),
    department: prob.department,
  }));

  // 4축 점수 변환
  // API: composition, technique, creativity, completeness
  // UI: composition, color, technique, creativity
  // 매핑: technique -> color, creativity -> technique, completeness -> creativity
  const axisScores = {
    composition: Math.round(scores?.composition || 0),
    color: Math.round(scores?.technique || 0), // technique -> 색채
    technique: Math.round(scores?.creativity || 0), // creativity -> 기법
    creativity: Math.round(scores?.completeness || 0), // completeness -> 창의성
  };

  return {
    evaluationId: evaluation_id,
    tier: tier || 'B',
    universityScores,
    axisScores,
    feedback: {
      strengths: feedback?.strengths || [],
      improvements: feedback?.improvements || [],
      overall: feedback?.overall || '',
    },
  };
};

/**
 * 이미지 평가 API 호출
 *
 * @param {File} imageFile - 평가할 이미지 파일
 * @param {string} department - 목표 학과 (visual_design, industrial_design, fine_art, craft)
 * @param {boolean} includeFeedback - 피드백 포함 여부
 * @param {Object} options - 추가 옵션
 * @param {number} options.timeout - 타임아웃 (기본 30초)
 * @param {function} options.onProgress - 진행 상태 콜백
 * @param {string} options.userId - 사용자 ID (활동 기록용, 선택사항)
 * @param {string} options.imageUrl - 이미지 URL (활동 기록용, 선택사항)
 * @returns {Promise<Object>} UI 형식의 평가 결과
 * @throws {DiagnosisAPIError} API 오류
 * @throws {NetworkError} 네트워크 오류
 * @throws {TimeoutError} 타임아웃 오류
 */
export const evaluateImage = async (imageFile, department, includeFeedback = true, options = {}) => {
  const { timeout = DEFAULT_TIMEOUT, onProgress, userId, imageUrl } = options;

  // Mock 모드 체크
  if (USE_MOCK) {
    console.log('[DiagnosisService] Mock 모드로 실행됨');
    const mockResult = await generateMockResult(department);

    // SPEC-CRED-001 M5: Mock 모드에서도 활동 기록 (userId가 제공된 경우)
    if (userId) {
      recordDiagnosisActivity(userId, {
        ...mockResult,
        imageUrl,
      }).catch((err) => {
        console.warn('[DiagnosisService] Mock 활동 기록 실패 (무시됨):', err);
      });
    }

    return mockResult;
  }

  // 진행 상태: 업로드 시작
  onProgress?.('uploading');

  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('department', department);
  formData.append('include_feedback', includeFeedback.toString());

  try {
    // 진행 상태: API 호출
    onProgress?.('analyzing');

    const response = await fetchWithTimeout(
      `${API_BASE_URL}/api/v1/evaluate`,
      {
        method: 'POST',
        body: formData,
        // Content-Type은 FormData일 때 자동 설정됨 (boundary 포함)
      },
      timeout
    );

    if (!response.ok) {
      let errorMessage = '평가 중 오류가 발생했습니다.';
      let errorDetails = null;

      try {
        const errorData = await response.json();
        errorMessage = errorData.detail || errorData.message || errorMessage;
        errorDetails = errorData;
      } catch {
        // JSON 파싱 실패 시 기본 메시지 사용
      }

      // 상태 코드별 에러 메시지
      if (response.status === 400) {
        errorMessage = '잘못된 요청입니다. 이미지 형식을 확인해주세요.';
      } else if (response.status === 413) {
        errorMessage = '이미지 크기가 너무 큽니다. 10MB 이하로 줄여주세요.';
      } else if (response.status === 500) {
        errorMessage = '서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.';
      } else if (response.status === 503) {
        errorMessage = '서비스가 일시적으로 사용 불가합니다. 잠시 후 다시 시도해주세요.';
      }

      throw new DiagnosisAPIError(errorMessage, response.status, errorDetails);
    }

    // 진행 상태: 결과 처리
    onProgress?.('processing');

    const apiResponse = await response.json();
    const result = mapApiResponseToUIFormat(apiResponse);

    // SPEC-CRED-001 M5: 활동 기록 (userId가 제공된 경우)
    // 실패해도 메인 플로우를 차단하지 않음 (non-blocking)
    if (userId) {
      recordDiagnosisActivity(userId, {
        ...result,
        imageUrl,
      }).catch((err) => {
        console.warn('[DiagnosisService] 활동 기록 실패 (무시됨):', err);
      });
    }

    return result;

  } catch (error) {
    // 이미 변환된 에러는 그대로 throw
    if (error instanceof DiagnosisAPIError ||
        error instanceof NetworkError ||
        error instanceof TimeoutError) {
      throw error;
    }

    // 예상치 못한 에러
    console.error('[DiagnosisService] 예상치 못한 에러:', error);
    throw new NetworkError('서버와 연결할 수 없습니다. 네트워크 상태를 확인해주세요.');
  }
};

/**
 * API 서버 상태 확인
 * @returns {Promise<boolean>} 서버가 정상이면 true
 */
export const checkApiHealth = async () => {
  try {
    const response = await fetchWithTimeout(
      `${API_BASE_URL}/health`,
      { method: 'GET' },
      5000 // 5초 타임아웃
    );
    return response.ok;
  } catch {
    return false;
  }
};

/**
 * Mock 결과 생성 (프로토타입/오프라인용)
 *
 * @param {string} department - 목표 학과
 * @returns {Promise<Object>} Mock 평가 결과
 */
export const generateMockResult = async (department) => {
  // 실제 API 호출 시뮬레이션 (1.5-2.5초 딜레이)
  const delay = 1500 + Math.random() * 1000;
  await new Promise(resolve => setTimeout(resolve, delay));

  // 학과별 대학 목록
  const universities = {
    visual_design: [
      { university: '홍익대', department: '시각디자인과' },
      { university: '서울대', department: '디자인학부' },
      { university: '국민대', department: '시각디자인학과' },
      { university: '건국대', department: '커뮤니케이션디자인학과' },
    ],
    industrial_design: [
      { university: '서울대', department: '디자인학부' },
      { university: '홍익대', department: '산업디자인학과' },
      { university: 'KAIST', department: '산업디자인학과' },
      { university: '한양대', department: '산업디자인학과' },
    ],
    fine_art: [
      { university: '서울대', department: '서양화과' },
      { university: '홍익대', department: '회화과' },
      { university: '이화여대', department: '서양화전공' },
      { university: '중앙대', department: '한국화전공' },
    ],
    craft: [
      { university: '서울대', department: '공예과' },
      { university: '홍익대', department: '도예유리과' },
      { university: '국민대', department: '도자공예학과' },
      { university: '이화여대', department: '공예전공' },
    ],
  };

  const deptUniversities = universities[department] || universities.visual_design;

  // 랜덤 티어 (S: 10%, A: 30%, B: 40%, C: 20%)
  const tierRand = Math.random();
  let tier;
  if (tierRand < 0.1) tier = 'S';
  else if (tierRand < 0.4) tier = 'A';
  else if (tierRand < 0.8) tier = 'B';
  else tier = 'C';

  // 티어에 따른 기본 점수 범위
  const tierBaseScore = {
    S: { min: 85, max: 98 },
    A: { min: 70, max: 89 },
    B: { min: 55, max: 74 },
    C: { min: 40, max: 59 },
  };

  const base = tierBaseScore[tier];

  return {
    evaluationId: `mock-${Date.now()}`,
    tier,
    universityScores: deptUniversities.map(univ => ({
      ...univ,
      score: Math.floor(Math.random() * (base.max - base.min) + base.min),
    })),
    axisScores: {
      composition: Math.floor(Math.random() * (base.max - base.min) + base.min),
      color: Math.floor(Math.random() * (base.max - base.min) + base.min),
      technique: Math.floor(Math.random() * (base.max - base.min) + base.min),
      creativity: Math.floor(Math.random() * (base.max - base.min) + base.min),
    },
    feedback: {
      strengths: [
        '구도 구성이 안정적입니다',
        '색채 조화가 돋보입니다',
        '창의적인 표현이 인상적입니다',
      ].slice(0, Math.floor(Math.random() * 2) + 2),
      improvements: [
        '디테일 표현을 좀 더 다듬어보세요',
        '명암 대비를 강조하면 입체감이 살아납니다',
        '배경 처리를 보완하면 완성도가 높아집니다',
      ].slice(0, Math.floor(Math.random() * 2) + 2),
      overall: tier === 'S' || tier === 'A'
        ? '전반적으로 우수한 작품입니다. 세부 표현을 더 다듬으면 좋겠습니다.'
        : '기본기가 잘 갖춰져 있습니다. 꾸준한 연습으로 더 발전할 수 있습니다.',
    },
  };
};

const diagnosisService = {
  evaluateImage,
  checkApiHealth,
  generateMockResult,
  DiagnosisAPIError,
  NetworkError,
  TimeoutError,
};

export default diagnosisService;
