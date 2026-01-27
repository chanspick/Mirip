/**
 * 통합 서비스
 *
 * 시스템 간 연동을 위한 헬퍼 함수들을 제공합니다.
 * SPEC-CRED-001 M5 요구사항에 따라 구현되었습니다.
 *
 * @module services/integrationService
 */

import { recordActivity } from './activityService';
import { recordAward } from './awardService';

/**
 * AI 진단 완료 후 활동 기록
 *
 * diagnosisService에서 호출됩니다.
 * 실패해도 메인 플로우를 차단하지 않습니다 (non-blocking).
 *
 * @param {string} userId - 사용자 ID
 * @param {Object} diagnosisResult - 진단 결과
 * @param {string} diagnosisResult.tier - 평가 티어 (S/A/B/C)
 * @param {number} diagnosisResult.overallScore - 종합 점수
 * @param {Object} diagnosisResult.axisScores - 축별 점수
 * @param {string} [diagnosisResult.imageUrl] - 진단 이미지 URL
 * @returns {Promise<Object|null>} 생성된 활동 또는 null (실패 시)
 */
export const recordDiagnosisActivity = async (userId, diagnosisResult) => {
  if (!userId) {
    console.warn('[IntegrationService] userId 없이 진단 활동 기록 시도');
    return null;
  }

  try {
    // 종합 점수 계산 (axisScores 평균)
    const scores = diagnosisResult.axisScores || {};
    const scoreValues = Object.values(scores).filter(v => typeof v === 'number');
    const overallScore = scoreValues.length > 0
      ? Math.round(scoreValues.reduce((a, b) => a + b, 0) / scoreValues.length)
      : 0;

    const activity = await recordActivity(userId, {
      type: 'diagnosis',
      title: 'AI 진단 완료',
      description: `작품 AI 평가 티어: ${diagnosisResult.tier || 'N/A'}, 점수: ${overallScore}점`,
      metadata: {
        tier: diagnosisResult.tier,
        overallScore,
        axisScores: diagnosisResult.axisScores,
        imageUrl: diagnosisResult.imageUrl || null,
        evaluationId: diagnosisResult.evaluationId || null,
      },
    });

    return activity;
  } catch (error) {
    // 활동 기록 실패는 메인 플로우를 차단하지 않음
    console.error('[IntegrationService] 진단 활동 기록 실패:', error);
    return null;
  }
};

/**
 * 공모전 출품 후 활동 기록
 *
 * submissionService에서 호출됩니다.
 * 실패해도 메인 플로우를 차단하지 않습니다 (non-blocking).
 *
 * @param {string} userId - 사용자 ID
 * @param {Object} submissionData - 출품 데이터
 * @param {string} submissionData.competitionId - 공모전 ID
 * @param {string} submissionData.competitionTitle - 공모전 제목
 * @param {string} submissionData.submissionId - 출품 ID
 * @param {string} [submissionData.imageUrl] - 출품 이미지 URL
 * @returns {Promise<Object|null>} 생성된 활동 또는 null (실패 시)
 */
export const recordSubmissionActivity = async (userId, submissionData) => {
  if (!userId) {
    console.warn('[IntegrationService] userId 없이 출품 활동 기록 시도');
    return null;
  }

  try {
    const activity = await recordActivity(userId, {
      type: 'competition_submit',
      title: `${submissionData.competitionTitle || '공모전'} 출품`,
      description: '공모전 작품 출품 완료',
      metadata: {
        competitionId: submissionData.competitionId,
        competitionTitle: submissionData.competitionTitle,
        submissionId: submissionData.submissionId,
        imageUrl: submissionData.imageUrl || null,
      },
    });

    return activity;
  } catch (error) {
    // 활동 기록 실패는 메인 플로우를 차단하지 않음
    console.error('[IntegrationService] 출품 활동 기록 실패:', error);
    return null;
  }
};

/**
 * 공모전 수상 기록
 *
 * 공모전 결과 발표 시 호출됩니다.
 * awards 컬렉션에 기록하고, 활동으로도 기록합니다.
 *
 * @param {string} userId - 사용자 ID
 * @param {Object} awardData - 수상 데이터
 * @param {string} awardData.competitionId - 공모전 ID
 * @param {string} awardData.competitionTitle - 공모전 제목
 * @param {string} awardData.rank - 수상 등급 (대상/금상/은상/동상/입선)
 * @param {Date} [awardData.awardedAt] - 수상일
 * @returns {Promise<{award: Object|null, activity: Object|null}>}
 */
export const recordCompetitionAward = async (userId, awardData) => {
  if (!userId) {
    console.warn('[IntegrationService] userId 없이 수상 기록 시도');
    return { award: null, activity: null };
  }

  let award = null;
  let activity = null;

  // 1. 수상 기록 (awards 컬렉션)
  try {
    award = await recordAward(userId, {
      competitionId: awardData.competitionId,
      competitionTitle: awardData.competitionTitle,
      rank: awardData.rank,
      awardedAt: awardData.awardedAt,
    });
  } catch (error) {
    console.error('[IntegrationService] 수상 기록 실패:', error);
  }

  // 2. 활동 기록
  try {
    activity = await recordActivity(userId, {
      type: 'competition_award',
      title: `${awardData.competitionTitle} ${awardData.rank}`,
      description: '공모전 수상',
      metadata: {
        competitionId: awardData.competitionId,
        competitionTitle: awardData.competitionTitle,
        rank: awardData.rank,
        awardId: award?.id || null,
      },
    });
  } catch (error) {
    console.error('[IntegrationService] 수상 활동 기록 실패:', error);
  }

  return { award, activity };
};

/**
 * 포트폴리오 추가 후 활동 기록
 *
 * portfolioService에서 호출됩니다.
 * 실패해도 메인 플로우를 차단하지 않습니다 (non-blocking).
 *
 * @param {string} userId - 사용자 ID
 * @param {Object} portfolioData - 포트폴리오 데이터
 * @param {string} portfolioData.id - 포트폴리오 ID
 * @param {string} portfolioData.title - 작품 제목
 * @param {string} [portfolioData.imageUrl] - 이미지 URL
 * @returns {Promise<Object|null>} 생성된 활동 또는 null (실패 시)
 */
export const recordPortfolioActivity = async (userId, portfolioData) => {
  if (!userId) {
    console.warn('[IntegrationService] userId 없이 포트폴리오 활동 기록 시도');
    return null;
  }

  try {
    const activity = await recordActivity(userId, {
      type: 'portfolio_add',
      title: `포트폴리오 추가: ${portfolioData.title}`,
      description: '새 작품을 포트폴리오에 추가했습니다',
      metadata: {
        portfolioId: portfolioData.id,
        title: portfolioData.title,
        imageUrl: portfolioData.imageUrl || null,
      },
    });

    return activity;
  } catch (error) {
    console.error('[IntegrationService] 포트폴리오 활동 기록 실패:', error);
    return null;
  }
};

const integrationService = {
  recordDiagnosisActivity,
  recordSubmissionActivity,
  recordCompetitionAward,
  recordPortfolioActivity,
};

export default integrationService;
