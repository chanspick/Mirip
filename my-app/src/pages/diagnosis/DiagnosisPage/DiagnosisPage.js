/**
 * AI 진단 페이지
 *
 * MIRIP 프로토타입 - 작품 이미지를 업로드하고 AI 진단 결과를 받는 페이지
 * 디자인 시스템: Competition, Landing 페이지와 동일한 패턴 적용
 * Phase B-3: Backend API 통합 (POST /api/v1/evaluate)
 *
 * @module pages/diagnosis/DiagnosisPage
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Header, Footer, Button, Loading } from '../../../components/common';
import {
  evaluateImage,
  generateMockResult,
  checkApiHealth,
  DiagnosisAPIError,
  NetworkError,
  TimeoutError,
} from '../../../services/diagnosisService';
import styles from './DiagnosisPage.module.css';

/**
 * 네비게이션 아이템
 */
const NAV_ITEMS = [
  { label: '공모전', href: '/competitions' },
  { label: 'AI 진단', href: '/diagnosis' },
  { label: 'Why MIRIP', href: '/#problem' },
  { label: 'Solution', href: '/#solution' },
];

/**
 * Footer 링크
 */
const FOOTER_LINKS = [
  { label: '이용약관', href: '/terms' },
  { label: '개인정보처리방침', href: '/privacy' },
];

/**
 * 학과 옵션
 */
const DEPARTMENT_OPTIONS = [
  { value: 'visual_design', label: '시각디자인' },
  { value: 'industrial_design', label: '산업디자인' },
  { value: 'fine_art', label: '회화' },
  { value: 'craft', label: '공예' },
];

/**
 * 진행 상태 메시지
 */
const PROGRESS_MESSAGES = {
  uploading: '이미지를 업로드하고 있습니다...',
  analyzing: 'AI가 작품을 분석하고 있습니다...',
  processing: '결과를 처리하고 있습니다...',
  default: 'AI가 작품을 분석하고 있습니다...',
};

/**
 * 에러 타입별 메시지
 */
const getErrorMessage = (error) => {
  if (error instanceof NetworkError) {
    return {
      message: error.message,
      canRetry: true,
      type: 'network',
    };
  }
  if (error instanceof TimeoutError) {
    return {
      message: error.message,
      canRetry: true,
      type: 'timeout',
    };
  }
  if (error instanceof DiagnosisAPIError) {
    return {
      message: error.message,
      canRetry: error.statusCode >= 500, // 서버 오류만 재시도 가능
      type: 'api',
    };
  }
  return {
    message: '진단 중 오류가 발생했습니다. 다시 시도해주세요.',
    canRetry: true,
    type: 'unknown',
  };
};

/**
 * AI 진단 페이지 컴포넌트
 */
const DiagnosisPage = () => {
  // 상태 관리
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [department, setDepartment] = useState('visual_design');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [errorInfo, setErrorInfo] = useState(null); // 상세 에러 정보
  const [isDragging, setIsDragging] = useState(false);
  const [progressStatus, setProgressStatus] = useState('default'); // 진행 상태
  const [isApiAvailable, setIsApiAvailable] = useState(null); // API 가용성

  const fileInputRef = useRef(null);

  /**
   * API 서버 상태 확인 (마운트 시)
   */
  useEffect(() => {
    const checkApi = async () => {
      const available = await checkApiHealth();
      setIsApiAvailable(available);
      if (!available) {
        console.log('[DiagnosisPage] API 서버 불가 - Mock 모드로 동작합니다.');
      }
    };
    checkApi();
  }, []);

  /**
   * 파일 선택 핸들러
   */
  const handleFileSelect = useCallback((file) => {
    if (!file) return;

    // 파일 타입 검증
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setError('JPG, PNG, WebP 형식의 이미지만 업로드 가능합니다.');
      return;
    }

    // 파일 크기 검증 (10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('파일 크기는 10MB 이하여야 합니다.');
      return;
    }

    setError(null);
    setSelectedImage(file);
    setResult(null);

    // 미리보기 URL 생성
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  }, []);

  /**
   * 파일 입력 변경 핸들러
   */
  const handleInputChange = useCallback((e) => {
    const file = e.target.files?.[0];
    handleFileSelect(file);
  }, [handleFileSelect]);

  /**
   * 드래그 이벤트 핸들러
   */
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    handleFileSelect(file);
  }, [handleFileSelect]);

  /**
   * 업로드 영역 클릭 핸들러
   */
  const handleUploadClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  /**
   * 진단 시작 핸들러
   */
  const handleAnalyze = useCallback(async () => {
    if (!selectedImage) {
      setError('이미지를 먼저 업로드해주세요.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setErrorInfo(null);
    setProgressStatus('default');

    try {
      let analysisResult;

      // API가 사용 불가능하면 Mock 모드로 폴백
      if (isApiAvailable === false) {
        console.log('[DiagnosisPage] API 불가 - Mock 결과 사용');
        analysisResult = await generateMockResult(department);
      } else {
        // 실제 API 호출
        analysisResult = await evaluateImage(
          selectedImage,
          department,
          true, // includeFeedback
          {
            timeout: 30000, // 30초 타임아웃
            onProgress: (status) => {
              setProgressStatus(status);
            },
          }
        );
      }

      setResult(analysisResult);
    } catch (err) {
      console.error('[DiagnosisPage] 진단 오류:', err);

      const errorDetails = getErrorMessage(err);
      setError(errorDetails.message);
      setErrorInfo(errorDetails);

      // API 오류 시 Mock 폴백 시도 (네트워크/타임아웃 에러만)
      if (errorDetails.type === 'network' || errorDetails.type === 'timeout') {
        setIsApiAvailable(false);
      }
    } finally {
      setIsAnalyzing(false);
      setProgressStatus('default');
    }
  }, [selectedImage, department, isApiAvailable]);

  /**
   * 재시도 핸들러
   */
  const handleRetry = useCallback(() => {
    setError(null);
    setErrorInfo(null);
    handleAnalyze();
  }, [handleAnalyze]);

  /**
   * 초기화 핸들러
   */
  const handleReset = useCallback(() => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setErrorInfo(null);
    setProgressStatus('default');
  }, []);

  return (
    <div className={styles.page}>
      {/* 헤더 */}
      <Header
        logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
        navItems={NAV_ITEMS}
        ctaButton={{
          label: '공모전',
          onClick: () => window.location.href = '/competitions',
        }}
      />

      {/* 메인 콘텐츠 */}
      <main className={styles.main}>
        <div className={styles.container}>
          {/* 페이지 헤더 */}
          <header className={styles.header}>
            <span className={styles.sectionLabel}>AI Diagnosis</span>
            <h1 className={styles.title}>AI 작품 진단</h1>
            <p className={styles.subtitle}>
              작품 이미지를 업로드하면 AI가 대학별 합격 가능성을 분석합니다
            </p>
          </header>

          <div className={styles.content}>
            {/* 좌측: 업로드 영역 */}
            <div className={styles.uploadSection}>
              <div
                className={`${styles.uploadArea} ${isDragging ? styles.dragging : ''} ${previewUrl ? styles.hasImage : ''}`}
                onClick={handleUploadClick}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleInputChange}
                  className={styles.fileInput}
                />

                {previewUrl ? (
                  <div className={styles.previewContainer}>
                    <img
                      src={previewUrl}
                      alt="작품 미리보기"
                      className={styles.previewImage}
                    />
                    <button
                      className={styles.changeButton}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleReset();
                      }}
                    >
                      이미지 변경
                    </button>
                  </div>
                ) : (
                  <div className={styles.uploadPlaceholder}>
                    <div className={styles.uploadIcon}>🎨</div>
                    <p className={styles.uploadText}>
                      작품 이미지를 드래그하거나
                      <br />
                      클릭하여 업로드하세요
                    </p>
                    <span className={styles.uploadHint}>
                      JPG, PNG, WebP / 최대 10MB
                    </span>
                  </div>
                )}
              </div>

              {/* 학과 선택 */}
              <div className={styles.optionGroup}>
                <label className={styles.optionLabel}>목표 학과</label>
                <select
                  className={styles.select}
                  value={department}
                  onChange={(e) => setDepartment(e.target.value)}
                  disabled={isAnalyzing}
                >
                  {DEPARTMENT_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>

              {/* 에러 메시지 */}
              {error && (
                <div className={styles.error}>
                  <span>{error}</span>
                  {errorInfo?.canRetry && (
                    <button
                      className={styles.retryButton}
                      onClick={handleRetry}
                      type="button"
                    >
                      다시 시도
                    </button>
                  )}
                </div>
              )}

              {/* 진단 버튼 */}
              <Button
                variant="cta"
                size="lg"
                onClick={handleAnalyze}
                disabled={!selectedImage || isAnalyzing}
                className={styles.analyzeButton}
              >
                {isAnalyzing ? '분석 중...' : 'AI 진단 시작'}
              </Button>
            </div>

            {/* 우측: 결과 영역 */}
            <div className={styles.resultSection}>
              {isAnalyzing ? (
                <div className={styles.loadingState}>
                  <Loading size="large" />
                  <p className={styles.loadingText}>
                    {PROGRESS_MESSAGES[progressStatus] || PROGRESS_MESSAGES.default}
                  </p>
                  {isApiAvailable === false && (
                    <p className={styles.mockModeHint}>
                      (오프라인 모드)
                    </p>
                  )}
                </div>
              ) : result ? (
                <div className={styles.resultCard}>
                  {/* 티어 */}
                  <div className={styles.tierSection}>
                    <span className={styles.tierLabel}>종합 등급</span>
                    <span className={`${styles.tierValue} ${styles[`tier${result.tier}`]}`}>
                      {result.tier}
                    </span>
                  </div>

                  {/* 대학별 점수 */}
                  <div className={styles.scoresSection}>
                    <h3 className={styles.sectionSubtitle}>대학별 합격 예측</h3>
                    <div className={styles.scoresList}>
                      {result.universityScores.map((item) => (
                        <div key={item.university} className={styles.scoreItem}>
                          <span className={styles.scoreUniv}>{item.university}</span>
                          <div className={styles.scoreBar}>
                            <div
                              className={styles.scoreFill}
                              style={{ width: `${item.score}%` }}
                            />
                          </div>
                          <span className={styles.scoreValue}>{item.score}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 4축 점수 */}
                  <div className={styles.axisSection}>
                    <h3 className={styles.sectionSubtitle}>세부 평가</h3>
                    <div className={styles.axisGrid}>
                      <div className={styles.axisItem}>
                        <span className={styles.axisLabel}>구도</span>
                        <span className={styles.axisValue}>{result.axisScores.composition}</span>
                      </div>
                      <div className={styles.axisItem}>
                        <span className={styles.axisLabel}>색채</span>
                        <span className={styles.axisValue}>{result.axisScores.color}</span>
                      </div>
                      <div className={styles.axisItem}>
                        <span className={styles.axisLabel}>기법</span>
                        <span className={styles.axisValue}>{result.axisScores.technique}</span>
                      </div>
                      <div className={styles.axisItem}>
                        <span className={styles.axisLabel}>창의성</span>
                        <span className={styles.axisValue}>{result.axisScores.creativity}</span>
                      </div>
                    </div>
                  </div>

                  {/* 피드백 */}
                  <div className={styles.feedbackSection}>
                    <div className={styles.feedbackGroup}>
                      <h4 className={styles.feedbackTitle}>💪 강점</h4>
                      <ul className={styles.feedbackList}>
                        {result.feedback.strengths.map((item, idx) => (
                          <li key={idx}>{item}</li>
                        ))}
                      </ul>
                    </div>
                    <div className={styles.feedbackGroup}>
                      <h4 className={styles.feedbackTitle}>📈 개선점</h4>
                      <ul className={styles.feedbackList}>
                        {result.feedback.improvements.map((item, idx) => (
                          <li key={idx}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* 다시 진단 버튼 */}
                  <button className={styles.resetButton} onClick={handleReset}>
                    새 작품 진단하기
                  </button>
                </div>
              ) : (
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>🤖</div>
                  <h3 className={styles.emptyTitle}>AI 진단 대기 중</h3>
                  <p className={styles.emptyText}>
                    작품 이미지를 업로드하고
                    <br />
                    진단 버튼을 클릭하세요
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* 안내 문구 */}
          <div className={styles.notice}>
            {isApiAvailable === false ? (
              <p>
                <strong>오프라인 모드:</strong> 현재 AI 서버에 연결할 수 없어 데모 데이터로 동작합니다.
                네트워크 연결을 확인하거나 잠시 후 다시 시도해주세요.
              </p>
            ) : isApiAvailable === true ? (
              <p>
                <strong>AI 진단 서비스:</strong> DINOv2 기반 AI 모델이 작품을 분석합니다.
                결과는 참고용이며, 실제 입시 결과와 다를 수 있습니다.
              </p>
            ) : (
              <p>
                <strong>서버 연결 확인 중...</strong> 잠시만 기다려주세요.
              </p>
            )}
          </div>
        </div>
      </main>

      {/* 푸터 */}
      <Footer
        links={FOOTER_LINKS}
        copyright="© 2025 MIRIP. All rights reserved."
      />
    </div>
  );
};

export default DiagnosisPage;
