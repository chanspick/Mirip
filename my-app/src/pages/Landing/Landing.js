// Landing 페이지 컴포넌트
// SPEC-FIREBASE-001: 사전 등록 랜딩 페이지
// 6개 섹션: Hero, Problem, Solution, AI Preview, CTA, Success Modal
// TDD로 구현 - 28개 테스트 통과

import React, { useState, useRef, useCallback, useMemo } from 'react';
import { Header, Footer, Modal, Button } from '../../components/common';
import RegistrationForm from '../../components/features/RegistrationForm';
import styles from './Landing.module.css';

/**
 * Problem 카드 데이터
 * @type {Array<{number: string, text: string, emphasis: string}>}
 */
const PROBLEM_CARDS = [
  {
    number: '01',
    text: '월 80~150만원,',
    emphasis: '그러나 기준 없는 평가',
  },
  {
    number: '02',
    text: '"지금 내 실력이 어느 정도인지"',
    emphasis: '알 수 없음',
  },
  {
    number: '03',
    text: '수상 이력과 포트폴리오,',
    emphasis: '흩어지고 사라짐',
  },
];

/**
 * Solution 타임라인 데이터
 * @type {Array<{step: string, title: string, description: string}>}
 */
const SOLUTION_TIMELINE = [
  { step: '01', title: '공모전', description: '참여와 경쟁의 시작' },
  { step: '02', title: '크레덴셜', description: '이력이 쌓이는 프로필' },
  { step: '03', title: 'AI 진단', description: '대학별 합격 예측' },
  { step: '04', title: '커리어', description: '채용과 거래로 확장' },
];

/**
 * AI Preview 대학별 점수 데이터
 * @type {Array<{university: string, score: number}>}
 */
const AI_SCORES = [
  { university: '서울대', score: 74 },
  { university: '홍익대', score: 81 },
  { university: '국민대', score: 69 },
];

/**
 * 네비게이션 아이템
 * @type {Array<{label: string, href: string}>}
 */
const NAV_ITEMS = [
  { label: 'Why MIRIP', href: '#problem' },
  { label: 'Solution', href: '#solution' },
  { label: 'AI Preview', href: '#ai-preview' },
];

/**
 * Landing 페이지 컴포넌트
 * 사전 등록을 위한 랜딩 페이지로, 6개의 섹션으로 구성됩니다.
 */
/**
 * Footer 링크 상수
 * @type {Array<{label: string, href: string}>}
 */
const FOOTER_LINKS = [
  { label: '이용약관', href: '/terms' },
  { label: '개인정보처리방침', href: '/privacy' },
];

/**
 * Landing 페이지 컴포넌트
 * 사전 등록을 위한 랜딩 페이지로, 6개의 섹션으로 구성됩니다.
 *
 * @component
 * @example
 * return (
 *   <Landing />
 * )
 */
const Landing = () => {
  // 성공 모달 상태
  const [isModalOpen, setIsModalOpen] = useState(false);

  // CTA 섹션 참조 (스크롤용)
  const ctaSectionRef = useRef(null);

  /**
   * CTA 섹션으로 스크롤
   * @function
   */
  const scrollToCta = useCallback(() => {
    if (ctaSectionRef.current) {
      ctaSectionRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    }
  }, []);

  /**
   * 등록 성공 핸들러
   */
  const handleRegistrationSuccess = useCallback(() => {
    setIsModalOpen(true);
  }, []);

  /**
   * 모달 닫기 핸들러
   */
  const handleCloseModal = useCallback(() => {
    setIsModalOpen(false);
  }, []);

  /**
   * Header CTA 버튼 설정 (메모이제이션)
   */
  const ctaButtonConfig = useMemo(
    () => ({
      label: '사전등록',
      onClick: scrollToCta,
    }),
    [scrollToCta]
  );

  return (
    <div className={styles.landing}>
      {/* Header */}
      <Header
        logo={<span className={styles.logo}>MIRIP</span>}
        navItems={NAV_ITEMS}
        ctaButton={ctaButtonConfig}
      />

      {/* Hero Section */}
      <section
        id="hero"
        className={styles.hero}
        data-testid="hero-section"
      >
        <div className={styles.heroContainer}>
          <div className={styles.heroContent}>
            <h1 className={styles.heroTitle}>
              당신의 작품,
              <br />
              <span className={styles.highlight}>어디까지</span> 갈 수 있을까요?
            </h1>
            <p className={styles.heroSubtitle}>
              대학별 합격 데이터를 학습한 AI가
              <br />
              객관적으로 진단합니다
            </p>
            <Button variant="cta" size="lg" onClick={scrollToCta}>
              사전등록
            </Button>
          </div>
          <div className={styles.heroVisual}>
            <div className={styles.heroFrame}>
              <div className={styles.heroArtwork} />
            </div>
          </div>
        </div>
        <div className={styles.heroScroll}>
          <span>Scroll</span>
          <div className={styles.scrollLine} />
        </div>
      </section>

      {/* Problem Section */}
      <section
        id="problem"
        className={styles.problem}
        data-testid="problem-section"
      >
        <div className={styles.container}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionLabel}>The Problem</span>
            <h2 className={styles.sectionTitle}>미술 입시, 막연한 불안</h2>
          </div>
          <div className={styles.problemGrid}>
            {PROBLEM_CARDS.map((card) => (
              <div key={card.number} className={styles.problemItem}>
                <span className={styles.problemNumber}>{card.number}</span>
                <p className={styles.problemText}>
                  {card.text}
                  <br />
                  <strong>{card.emphasis}</strong>
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section
        id="solution"
        className={styles.solution}
        data-testid="solution-section"
      >
        <div className={styles.container}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionLabel}>The Solution</span>
            <h2 className={styles.sectionTitle}>MIRIP이 연결합니다</h2>
          </div>
          <div className={styles.timeline}>
            <div className={styles.timelineLine} />
            {SOLUTION_TIMELINE.map((item) => (
              <div key={item.step} className={styles.timelineItem}>
                <div className={styles.timelineMarker}>
                  <span className={styles.timelineStep}>{item.step}</span>
                </div>
                <div className={styles.timelineContent}>
                  <h3 className={styles.timelineTitle}>{item.title}</h3>
                  <p className={styles.timelineDesc}>{item.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* AI Preview Section */}
      <section
        id="ai-preview"
        className={styles.aiPreview}
        data-testid="ai-preview-section"
      >
        <div className={styles.container}>
          <div className={styles.sectionHeader}>
            <span className={styles.sectionLabel}>AI Diagnosis</span>
            <h2 className={styles.sectionTitle}>AI가 본 당신의 가능성</h2>
          </div>
          <div className={styles.previewWrapper}>
            <div className={styles.previewArtwork}>
              <div className={styles.artworkFrame}>
                <div className={styles.artworkImage} />
                <span className={styles.artworkLabel}>작품 이미지</span>
              </div>
            </div>
            <div className={styles.previewReport}>
              <div className={styles.reportCard}>
                <h4 className={styles.reportTitle}>AI 진단 리포트</h4>
                <div className={styles.reportScores}>
                  {AI_SCORES.map((item) => (
                    <div key={item.university} className={styles.scoreItem}>
                      <span className={styles.scoreUniv}>{item.university}</span>
                      <div className={styles.scoreBar}>
                        <div
                          className={styles.scoreFill}
                          style={{ width: `${item.score}%` }}
                          data-score={item.score}
                        />
                      </div>
                      <span className={styles.scoreValue}>{item.score}</span>
                    </div>
                  ))}
                </div>
                <div className={styles.reportFeedback}>
                  <p className={styles.feedbackText}>
                    "구도와 명암에서 강점,
                    <br />
                    비례 보완 추천"
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section
        id="cta"
        className={styles.cta}
        data-testid="cta-section"
        ref={ctaSectionRef}
      >
        <div className={styles.container}>
          <div className={styles.ctaWrapper}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionLabel}>Pre-registration</span>
              <h2 className={styles.sectionTitle}>MIRIP, 곧 시작됩니다</h2>
            </div>
            <RegistrationForm
              onSuccess={handleRegistrationSuccess}
              className={styles.ctaForm}
            />
            <p className={styles.ctaNotice}>
              등록하신 정보는 서비스 출시 알림 목적으로만 사용됩니다.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <Footer
        links={FOOTER_LINKS}
        copyright="© 2025 MIRIP. All rights reserved."
      />

      {/* Success Modal */}
      <Modal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        title="등록이 완료되었습니다"
      >
        <div className={styles.modalContent}>
          <div className={styles.modalIcon}>&#10003;</div>
          <p className={styles.modalText}>
            서비스 출시 시 가장 먼저 알려드리겠습니다.
          </p>
        </div>
      </Modal>
    </div>
  );
};

export default Landing;
