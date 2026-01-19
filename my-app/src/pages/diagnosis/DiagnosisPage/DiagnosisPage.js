/**
 * AI 진단 페이지
 *
 * MIRIP 프로토타입 - 작품 이미지를 업로드하고 AI 진단 결과를 받는 페이지
 * 디자인 시스템: Competition, Landing 페이지와 동일한 패턴 적용
 *
 * @module pages/diagnosis/DiagnosisPage
 */

import React, { useState, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Header, Footer, Button, Loading } from '../../../components/common';
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
 * 목 AI 진단 결과 (프로토타입용)
 * 실제 서비스에서는 백엔드 API 호출로 대체
 */
const generateMockResult = (department) => {
  // 학과별 대학 목록
  const universities = {
    visual_design: ['서울대', '홍익대', '국민대', '건국대'],
    industrial_design: ['서울대', '홍익대', 'KAIST', '한양대'],
    fine_art: ['서울대', '홍익대', '이화여대', '중앙대'],
    craft: ['서울대', '홍익대', '국민대', '이화여대'],
  };

  const scores = universities[department] || universities.visual_design;

  return {
    tier: ['S', 'A', 'B', 'C'][Math.floor(Math.random() * 3)],
    universityScores: scores.map(univ => ({
      university: univ,
      score: Math.floor(Math.random() * 30) + 60,
    })),
    feedback: {
      strengths: ['구도 구성이 안정적입니다', '명암 대비가 효과적입니다'],
      improvements: ['비례 표현을 보완하면 좋겠습니다', '디테일 표현을 강화해보세요'],
    },
    axisScores: {
      composition: Math.floor(Math.random() * 20) + 70,
      color: Math.floor(Math.random() * 20) + 65,
      technique: Math.floor(Math.random() * 20) + 68,
      creativity: Math.floor(Math.random() * 20) + 72,
    },
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
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);

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

    try {
      // 프로토타입: 2초 딜레이 후 목 결과 반환
      // 실제 서비스에서는 백엔드 API 호출
      await new Promise(resolve => setTimeout(resolve, 2000));

      const mockResult = generateMockResult(department);
      setResult(mockResult);
    } catch (err) {
      setError('진단 중 오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedImage, department]);

  /**
   * 초기화 핸들러
   */
  const handleReset = useCallback(() => {
    setSelectedImage(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
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
                  <p className={styles.loadingText}>AI가 작품을 분석하고 있습니다...</p>
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
            <p>
              <strong>프로토타입 안내:</strong> 현재는 목업 데이터로 동작합니다.
              실제 AI 모델 연동은 추후 업데이트될 예정입니다.
            </p>
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
