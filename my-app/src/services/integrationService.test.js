/**
 * 통합 서비스 테스트
 *
 * SPEC-CRED-001 M5 요구사항에 따른 통합 서비스 단위 테스트
 *
 * @module services/integrationService.test
 */

import {
  recordDiagnosisActivity,
  recordSubmissionActivity,
  recordCompetitionAward,
  recordPortfolioActivity,
} from './integrationService';
import { recordActivity } from './activityService';
import { recordAward } from './awardService';

// Mock dependencies
jest.mock('./activityService', () => ({
  recordActivity: jest.fn(),
}));

jest.mock('./awardService', () => ({
  recordAward: jest.fn(),
}));

describe('integrationService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // 콘솔 경고/에러 억제
    jest.spyOn(console, 'warn').mockImplementation(() => {});
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('recordDiagnosisActivity', () => {
    const mockUserId = 'test-user-123';
    const mockDiagnosisResult = {
      tier: 'A',
      evaluationId: 'eval-123',
      axisScores: {
        composition: 75,
        color: 80,
        technique: 72,
        creativity: 78,
      },
      imageUrl: 'https://example.com/image.jpg',
    };

    it('should record diagnosis activity successfully', async () => {
      const mockActivity = { id: 'activity-123', type: 'diagnosis' };
      recordActivity.mockResolvedValue(mockActivity);

      const result = await recordDiagnosisActivity(mockUserId, mockDiagnosisResult);

      expect(recordActivity).toHaveBeenCalledWith(mockUserId, expect.objectContaining({
        type: 'diagnosis',
        title: 'AI 진단 완료',
      }));
      expect(result).toEqual(mockActivity);
    });

    it('should calculate overall score from axis scores', async () => {
      recordActivity.mockResolvedValue({ id: 'activity-123' });

      await recordDiagnosisActivity(mockUserId, mockDiagnosisResult);

      // 평균 점수 계산: (75 + 80 + 72 + 78) / 4 = 76.25 -> 76
      expect(recordActivity).toHaveBeenCalledWith(
        mockUserId,
        expect.objectContaining({
          metadata: expect.objectContaining({
            overallScore: 76,
          }),
        })
      );
    });

    it('should return null when userId is not provided', async () => {
      const result = await recordDiagnosisActivity(null, mockDiagnosisResult);

      expect(result).toBeNull();
      expect(recordActivity).not.toHaveBeenCalled();
    });

    it('should return null and not throw when activity recording fails', async () => {
      recordActivity.mockRejectedValue(new Error('DB Error'));

      const result = await recordDiagnosisActivity(mockUserId, mockDiagnosisResult);

      expect(result).toBeNull();
      expect(console.error).toHaveBeenCalled();
    });

    it('should handle missing axisScores gracefully', async () => {
      recordActivity.mockResolvedValue({ id: 'activity-123' });

      await recordDiagnosisActivity(mockUserId, { tier: 'B' });

      expect(recordActivity).toHaveBeenCalledWith(
        mockUserId,
        expect.objectContaining({
          metadata: expect.objectContaining({
            overallScore: 0,
          }),
        })
      );
    });
  });

  describe('recordSubmissionActivity', () => {
    const mockUserId = 'test-user-123';
    const mockSubmissionData = {
      competitionId: 'comp-123',
      competitionTitle: '2024 미술 공모전',
      submissionId: 'sub-123',
      imageUrl: 'https://example.com/submission.jpg',
    };

    it('should record submission activity successfully', async () => {
      const mockActivity = { id: 'activity-123', type: 'competition_submit' };
      recordActivity.mockResolvedValue(mockActivity);

      const result = await recordSubmissionActivity(mockUserId, mockSubmissionData);

      expect(recordActivity).toHaveBeenCalledWith(mockUserId, expect.objectContaining({
        type: 'competition_submit',
        title: '2024 미술 공모전 출품',
      }));
      expect(result).toEqual(mockActivity);
    });

    it('should return null when userId is not provided', async () => {
      const result = await recordSubmissionActivity(null, mockSubmissionData);

      expect(result).toBeNull();
      expect(recordActivity).not.toHaveBeenCalled();
    });

    it('should return null when activity recording fails', async () => {
      recordActivity.mockRejectedValue(new Error('DB Error'));

      const result = await recordSubmissionActivity(mockUserId, mockSubmissionData);

      expect(result).toBeNull();
    });

    it('should use default title when competitionTitle is missing', async () => {
      recordActivity.mockResolvedValue({ id: 'activity-123' });

      await recordSubmissionActivity(mockUserId, {
        competitionId: 'comp-123',
        submissionId: 'sub-123',
      });

      expect(recordActivity).toHaveBeenCalledWith(
        mockUserId,
        expect.objectContaining({
          title: '공모전 출품',
        })
      );
    });
  });

  describe('recordCompetitionAward', () => {
    const mockUserId = 'test-user-123';
    const mockAwardData = {
      competitionId: 'comp-123',
      competitionTitle: '2024 미술 공모전',
      rank: '금상',
    };

    it('should record both award and activity', async () => {
      const mockAward = { id: 'award-123' };
      const mockActivity = { id: 'activity-123' };
      recordAward.mockResolvedValue(mockAward);
      recordActivity.mockResolvedValue(mockActivity);

      const result = await recordCompetitionAward(mockUserId, mockAwardData);

      expect(recordAward).toHaveBeenCalledWith(mockUserId, expect.objectContaining({
        competitionId: 'comp-123',
        rank: '금상',
      }));
      expect(recordActivity).toHaveBeenCalledWith(mockUserId, expect.objectContaining({
        type: 'competition_award',
        title: '2024 미술 공모전 금상',
      }));
      expect(result).toEqual({ award: mockAward, activity: mockActivity });
    });

    it('should return null values when userId is not provided', async () => {
      const result = await recordCompetitionAward(null, mockAwardData);

      expect(result).toEqual({ award: null, activity: null });
      expect(recordAward).not.toHaveBeenCalled();
      expect(recordActivity).not.toHaveBeenCalled();
    });

    it('should continue recording activity even if award fails', async () => {
      recordAward.mockRejectedValue(new Error('Award DB Error'));
      recordActivity.mockResolvedValue({ id: 'activity-123' });

      const result = await recordCompetitionAward(mockUserId, mockAwardData);

      expect(result.award).toBeNull();
      expect(result.activity).not.toBeNull();
    });

    it('should continue even if activity recording fails', async () => {
      recordAward.mockResolvedValue({ id: 'award-123' });
      recordActivity.mockRejectedValue(new Error('Activity DB Error'));

      const result = await recordCompetitionAward(mockUserId, mockAwardData);

      expect(result.award).not.toBeNull();
      expect(result.activity).toBeNull();
    });
  });

  describe('recordPortfolioActivity', () => {
    const mockUserId = 'test-user-123';
    const mockPortfolioData = {
      id: 'portfolio-123',
      title: '나의 작품',
      imageUrl: 'https://example.com/portfolio.jpg',
    };

    it('should record portfolio activity successfully', async () => {
      const mockActivity = { id: 'activity-123', type: 'portfolio_add' };
      recordActivity.mockResolvedValue(mockActivity);

      const result = await recordPortfolioActivity(mockUserId, mockPortfolioData);

      expect(recordActivity).toHaveBeenCalledWith(mockUserId, expect.objectContaining({
        type: 'portfolio_add',
        title: '포트폴리오 추가: 나의 작품',
      }));
      expect(result).toEqual(mockActivity);
    });

    it('should return null when userId is not provided', async () => {
      const result = await recordPortfolioActivity(null, mockPortfolioData);

      expect(result).toBeNull();
      expect(recordActivity).not.toHaveBeenCalled();
    });

    it('should return null when recording fails', async () => {
      recordActivity.mockRejectedValue(new Error('DB Error'));

      const result = await recordPortfolioActivity(mockUserId, mockPortfolioData);

      expect(result).toBeNull();
    });
  });
});
