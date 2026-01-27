// Landing 페이지 컴포넌트
// MIRIP 프로토타입 버전 - 공모전 + AI 진단 연결
// SPEC-CRED-001 M5: 로그인 사용자 CTA 추가
// 5개 섹션: Hero, Problem, Solution, AI Preview, CTA (프로토타입)

import React, { useCallback, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Header, Footer, Button, AuthModal } from '../../components/common';
import { useAuth } from '../../hooks';
import { CREDENTIAL_ROUTES, GUEST_NAV_ITEMS, LOGGED_IN_NAV_ITEMS } from '../../utils/navigation';
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

// 네비게이션 아이템은 utils/navigation.js에서 가져옵니다
// GUEST_NAV_ITEMS: 비로그인 사용자용
// LOGGED_IN_NAV_ITEMS: 로그인 사용자용

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
 * MIRIP 프로토타입 버전 - 공모전과 AI 진단 기능 제공
 *
 * @component
 */
const Landing = () => {
  const navigate = useNavigate();
  // SPEC-CRED-001 M5: 인증 상태 확인
  const { profile, isAuthenticated, signOut } = useAuth();
  // 로그인 모달 상태
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  /**
   * AI 진단 페이지로 이동
   */
  const goToDiagnosis = useCallback(() => {
    navigate('/diagnosis');
  }, [navigate]);

  /**
   * 로그인 모달 열기
   */
  const handleLoginClick = useCallback(() => {
    setIsAuthModalOpen(true);
  }, []);

  /**
   * 로그인 성공 핸들러
   */
  const handleLoginSuccess = useCallback((loggedInUser) => {
    console.log('[Landing] 로그인 성공:', loggedInUser.displayName);
    // 로그인 후 필요한 추가 작업 있으면 여기서 처리
  }, []);

  /**
   * 로그아웃 핸들러
   */
  const handleLogout = useCallback(async () => {
    try {
      await signOut();
      console.log('[Landing] 로그아웃 완료');
    } catch (error) {
      console.error('[Landing] 로그아웃 실패:', error);
    }
  }, [signOut]);

  /**
   * Header CTA 버튼 설정 (메모이제이션)
   * 로그인 사용자는 CTA 버튼 대신 프로필 링크 사용
   */
  const ctaButtonConfig = useMemo(
    () => isAuthenticated ? null : ({
      label: 'AI 진단',
      onClick: goToDiagnosis,
    }),
    [goToDiagnosis, isAuthenticated]
  );

  /**
   * 네비게이션 아이템 (로그인 여부에 따라 변경)
   */
  const navItems = useMemo(
    () => isAuthenticated ? LOGGED_IN_NAV_ITEMS : GUEST_NAV_ITEMS,
    [isAuthenticated]
  );

  return (
    <div className={styles.landing}>
      {/* Header */}
      <Header
        logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
        navItems={navItems}
        ctaButton={ctaButtonConfig}
        user={isAuthenticated ? { photoURL: profile?.profileImageUrl, displayName: profile?.displayName } : null}
        onLoginClick={handleLoginClick}
        onLogout={handleLogout}
      />

      {/* 로그인 모달 */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
        onSuccess={handleLoginSuccess}
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
            <Button variant="cta" size="lg" onClick={goToDiagnosis}>
              AI 진단 시작하기
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
                <Link to="/diagnosis" className={styles.tryDiagnosisButton}>
                  내 작품 진단받기
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section - 프로토타입 기능 소개 */}
      <section
        id="cta"
        className={styles.cta}
        data-testid="cta-section"
      >
        <div className={styles.container}>
          <div className={styles.ctaWrapper}>
            <div className={styles.sectionHeader}>
              <span className={styles.sectionLabel}>Try Now</span>
              <h2 className={styles.sectionTitle}>지금 바로 체험해보세요</h2>
            </div>
            <div className={styles.ctaCards}>
              <Link to="/competitions" className={styles.ctaCard}>
                <div className={styles.ctaCardIcon}>🏆</div>
                <h3 className={styles.ctaCardTitle}>공모전</h3>
                <p className={styles.ctaCardDesc}>
                  다양한 분야의 공모전에 참여하고
                  <br />
                  실력을 뽐내보세요
                </p>
                <span className={styles.ctaCardLink}>둘러보기 →</span>
              </Link>
              <Link to="/diagnosis" className={styles.ctaCard}>
                <div className={styles.ctaCardIcon}>🤖</div>
                <h3 className={styles.ctaCardTitle}>AI 진단</h3>
                <p className={styles.ctaCardDesc}>
                  내 작품의 대학별 합격 가능성을
                  <br />
                  AI로 분석해보세요
                </p>
                <span className={styles.ctaCardLink}>진단받기 →</span>
              </Link>
              {/* SPEC-CRED-001 M5: 로그인 사용자용 추가 CTA */}
              {isAuthenticated && (
                <>
                  <Link to={CREDENTIAL_ROUTES.PROFILE} className={styles.ctaCard}>
                    <div className={styles.ctaCardIcon}>👤</div>
                    <h3 className={styles.ctaCardTitle}>마이페이지</h3>
                    <p className={styles.ctaCardDesc}>
                      나의 활동 현황과 잔디밭을
                      <br />
                      확인해보세요
                    </p>
                    <span className={styles.ctaCardLink}>보러가기 →</span>
                  </Link>
                  <Link to={CREDENTIAL_ROUTES.PORTFOLIO} className={styles.ctaCard}>
                    <div className={styles.ctaCardIcon}>🎨</div>
                    <h3 className={styles.ctaCardTitle}>포트폴리오</h3>
                    <p className={styles.ctaCardDesc}>
                      나의 작품들을 관리하고
                      <br />
                      공유해보세요
                    </p>
                    <span className={styles.ctaCardLink}>관리하기 →</span>
                  </Link>
                </>
              )}
            </div>
            <p className={styles.ctaNotice}>
              프로토타입 버전입니다. 더 나은 서비스를 위해 피드백을 기다립니다.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <Footer
        links={FOOTER_LINKS}
        copyright="© 2025 MIRIP. All rights reserved."
      />
    </div>
  );
};

export default Landing;
