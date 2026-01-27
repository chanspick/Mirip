/**
 * 공모전 상세 페이지
 *
 * SPEC-COMP-001 REQ-C-005 ~ REQ-C-010 구현
 *
 * @module pages/competitions/CompetitionDetail
 */

import React, { useState, useMemo } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Loading } from '../../../components/common';
import { useCompetitionDetail } from '../../../hooks/useCompetitions';
import styles from './CompetitionDetail.module.css';

/**
 * D-Day 계산
 * @param {Date} endDate - 마감일
 * @returns {string} D-Day 문자열
 */
const calculateDDay = (endDate) => {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const end = new Date(endDate);
  end.setHours(0, 0, 0, 0);
  const diff = Math.ceil((end - today) / (1000 * 60 * 60 * 24));

  if (diff < 0) return '마감';
  if (diff === 0) return 'D-Day';
  return `D-${diff}`;
};

/**
 * 날짜 포맷팅
 * @param {Date} date - 날짜
 * @returns {string} 포맷된 날짜
 */
const formatDate = (date) => {
  if (!date) return '';
  const d = new Date(date);
  return d.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
};

/**
 * 상금 포맷팅
 * @param {number} amount - 금액
 * @returns {string} 포맷된 금액
 */
const formatPrize = (amount) => {
  if (!amount) return '상금 미정';
  if (amount >= 10000) {
    return `${Math.floor(amount / 10000)}만원`;
  }
  return `${amount.toLocaleString()}원`;
};

/**
 * 탭 목록
 */
const TABS = [
  { id: 'overview', label: '대회 안내' },
  { id: 'rules', label: '규칙' },
  { id: 'schedule', label: '일정' },
  { id: 'prizes', label: '상금' },
];

/**
 * 공모전 상세 페이지 컴포넌트
 */
const CompetitionDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');

  const { competition, loading, error } = useCompetitionDetail(id);

  // 공모전 상태 계산
  const status = useMemo(() => {
    if (!competition) return null;
    const now = new Date();
    const start = new Date(competition.startDate);
    const end = new Date(competition.endDate);

    if (now < start) return { label: '예정', className: styles.statusUpcoming };
    if (now > end) return { label: '마감', className: styles.statusClosed };
    return { label: '진행중', className: styles.statusActive };
  }, [competition]);

  // 로딩 상태
  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <Loading size="large" />
      </div>
    );
  }

  // 에러 상태
  if (error) {
    return (
      <div className={styles.errorContainer}>
        <p>{error}</p>
        <button onClick={() => navigate('/competitions')}>
          목록으로 돌아가기
        </button>
      </div>
    );
  }

  // 공모전이 없는 경우
  if (!competition) {
    return (
      <div className={styles.notFound}>
        <h2>공모전을 찾을 수 없습니다</h2>
        <p>요청하신 공모전이 존재하지 않거나 삭제되었습니다.</p>
        <Link to="/competitions" className={styles.backLink}>
          목록으로 돌아가기
        </Link>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* 히어로 섹션 */}
      <section
        className={styles.hero}
        style={{
          backgroundImage: competition.thumbnailUrl
            ? `linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url(${competition.thumbnailUrl})`
            : 'linear-gradient(135deg, #3c7dde 0%, #4918e8 100%)',
        }}
      >
        <div className={styles.heroContent}>
          {/* 상태 배지 */}
          <span className={`${styles.status} ${status?.className}`}>
            {status?.label}
          </span>

          {/* 제목 */}
          <h1 className={styles.title}>{competition.title}</h1>

          {/* 카테고리 태그 */}
          <div className={styles.tags}>
            <span className={styles.tag}>{competition.category}</span>
            {competition.tags?.map((tag, index) => (
              <span key={index} className={styles.tag}>
                {tag}
              </span>
            ))}
          </div>

          {/* 메타 정보 */}
          <div className={styles.metaInfo}>
            <div className={styles.metaItem}>
              <span className={styles.metaIcon}>💰</span>
              <span>상금: {formatPrize(competition.prize)}</span>
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaIcon}>📅</span>
              <span>
                {formatDate(competition.startDate)} ~ {formatDate(competition.endDate)}
              </span>
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaIcon}>👥</span>
              <span>{competition.participantCount || 0}명 참여</span>
            </div>
            <div className={styles.metaItem}>
              <span className={styles.metaIcon}>⏰</span>
              <span>{calculateDDay(competition.endDate)}</span>
            </div>
          </div>

          {/* 참여 버튼 */}
          {status?.label !== '마감' && (
            <Link
              to={`/competitions/${id}/submit`}
              className={styles.submitButton}
            >
              참여하기
            </Link>
          )}
        </div>
      </section>

      {/* 탭 네비게이션 */}
      <nav className={styles.tabNav}>
        <div className={styles.tabList}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`${styles.tab} ${activeTab === tab.id ? styles.tabActive : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* 탭 콘텐츠 */}
      <main className={styles.content}>
        {/* 대회 안내 탭 */}
        {activeTab === 'overview' && (
          <div className={styles.tabContent}>
            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>대회 개요</h2>
              <div className={styles.description}>
                {competition.description || '대회 설명이 없습니다.'}
              </div>
            </section>

            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>주최 기관</h2>
              <p>{competition.organizer || '주최 기관 정보가 없습니다.'}</p>
            </section>

            {competition.requirements && (
              <section className={styles.section}>
                <h2 className={styles.sectionTitle}>참가 자격</h2>
                <p>{competition.requirements}</p>
              </section>
            )}
          </div>
        )}

        {/* 규칙 탭 */}
        {activeTab === 'rules' && (
          <div className={styles.tabContent}>
            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>대회 규칙</h2>
              <div className={styles.rulesList}>
                <ul>
                  <li>작품은 참가자 본인이 직접 제작해야 합니다.</li>
                  <li>기존에 발표되거나 수상한 작품은 제출할 수 없습니다.</li>
                  <li>제출된 작품의 저작권은 참가자에게 있습니다.</li>
                  <li>심사 결과에 대한 이의 제기는 발표 후 7일 이내 가능합니다.</li>
                </ul>
              </div>
            </section>
          </div>
        )}

        {/* 일정 탭 */}
        {activeTab === 'schedule' && (
          <div className={styles.tabContent}>
            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>대회 일정</h2>
              <div className={styles.timeline}>
                <div className={styles.timelineItem}>
                  <div className={styles.timelineDate}>
                    {formatDate(competition.startDate)}
                  </div>
                  <div className={styles.timelineContent}>
                    <h3>접수 시작</h3>
                    <p>공모전 접수가 시작됩니다.</p>
                  </div>
                </div>
                <div className={styles.timelineItem}>
                  <div className={styles.timelineDate}>
                    {formatDate(competition.endDate)}
                  </div>
                  <div className={styles.timelineContent}>
                    <h3>접수 마감</h3>
                    <p>작품 제출 마감일입니다.</p>
                  </div>
                </div>
                {competition.resultDate && (
                  <div className={styles.timelineItem}>
                    <div className={styles.timelineDate}>
                      {formatDate(competition.resultDate)}
                    </div>
                    <div className={styles.timelineContent}>
                      <h3>결과 발표</h3>
                      <p>심사 결과가 발표됩니다.</p>
                    </div>
                  </div>
                )}
              </div>
            </section>
          </div>
        )}

        {/* 상금 탭 */}
        {activeTab === 'prizes' && (
          <div className={styles.tabContent}>
            <section className={styles.section}>
              <h2 className={styles.sectionTitle}>시상 내역</h2>
              <div className={styles.prizeList}>
                <div className={styles.prizeItem}>
                  <div className={styles.prizeRank}>🥇 대상</div>
                  <div className={styles.prizeAmount}>
                    {formatPrize(competition.prize)}
                  </div>
                </div>
                <div className={styles.prizeItem}>
                  <div className={styles.prizeRank}>🥈 최우수상</div>
                  <div className={styles.prizeAmount}>
                    {formatPrize(Math.floor((competition.prize || 0) * 0.5))}
                  </div>
                </div>
                <div className={styles.prizeItem}>
                  <div className={styles.prizeRank}>🥉 우수상</div>
                  <div className={styles.prizeAmount}>
                    {formatPrize(Math.floor((competition.prize || 0) * 0.3))}
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>

      {/* 하단 CTA */}
      {status?.label !== '마감' && (
        <div className={styles.bottomCta}>
          <div className={styles.ctaContent}>
            <div className={styles.ctaInfo}>
              <span className={styles.ctaDeadline}>
                마감까지 {calculateDDay(competition.endDate)}
              </span>
              <span className={styles.ctaParticipants}>
                현재 {competition.participantCount || 0}명 참여 중
              </span>
            </div>
            <Link
              to={`/competitions/${id}/submit`}
              className={styles.ctaButton}
            >
              작품 제출하기
            </Link>
          </div>
        </div>
      )}
    </div>
  );
};

export default CompetitionDetail;
