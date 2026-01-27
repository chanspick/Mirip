// PublicProfilePage 컴포넌트
// SPEC-CRED-001: M3 공개 프로필
// 다른 사용자의 공개 프로필을 표시합니다

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { Header, Footer } from '../../components/common';
import { ActivityHeatmap, TierBadge, AchievementList } from '../../components/credential';
import { useUserProfile } from '../../hooks';
import { getPublicAwards } from '../../services/awardService';
import styles from './PublicProfilePage.module.css';

/**
 * 네비게이션 아이템
 */
const NAV_ITEMS = [
  { label: '공모전', href: '/competitions' },
  { label: 'AI 진단', href: '/diagnosis' },
  { label: '마이페이지', href: '/profile' },
];

/**
 * Footer 링크
 */
const FOOTER_LINKS = [
  { label: '이용약관', href: '/terms' },
  { label: '개인정보처리방침', href: '/privacy' },
];

/**
 * 기본 아바타 이니셜 생성
 * @param {string} name - 이름
 * @returns {string} 이니셜
 */
const getInitials = (name) => {
  if (!name) return '?';
  const words = name.split(' ');
  if (words.length >= 2) {
    return (words[0][0] + words[1][0]).toUpperCase();
  }
  return name.slice(0, 2).toUpperCase();
};

/**
 * PublicProfilePage 컴포넌트
 * 공개 프로필 페이지 - 다른 사용자의 프로필을 표시합니다
 */
const PublicProfilePage = () => {
  const { username } = useParams();
  const navigate = useNavigate();
  const { fetchByUsername } = useUserProfile();

  // 프로필 상태
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // 수상 내역 상태
  const [awards, setAwards] = useState([]);
  const [awardsLoading, setAwardsLoading] = useState(false);

  /**
   * AI 진단 페이지로 이동
   */
  const goToDiagnosis = useCallback(() => {
    navigate('/diagnosis');
  }, [navigate]);

  /**
   * Header CTA 버튼 설정
   */
  const ctaButtonConfig = useMemo(
    () => ({
      label: 'AI 진단',
      onClick: goToDiagnosis,
    }),
    [goToDiagnosis]
  );

  /**
   * 프로필 로드
   */
  useEffect(() => {
    const loadProfile = async () => {
      if (!username) {
        setLoading(false);
        setError('사용자명이 필요합니다');
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const profileData = await fetchByUsername(username);
        setProfile(profileData);

        // 공개 프로필인 경우 수상 내역 로드
        if (profileData && profileData.isPublic && profileData.uid) {
          setAwardsLoading(true);
          try {
            const awardsData = await getPublicAwards(profileData.uid);
            setAwards(awardsData || []);
          } catch (awardErr) {
            console.error('수상 내역 로드 실패:', awardErr);
            setAwards([]);
          } finally {
            setAwardsLoading(false);
          }
        }
      } catch (err) {
        console.error('프로필 로드 실패:', err);
        setError(err.message || '프로필을 불러올 수 없습니다');
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, [username, fetchByUsername]);

  /**
   * 공유 버튼 클릭 핸들러
   */
  const handleShare = useCallback(async () => {
    const shareUrl = window.location.href;
    const shareData = {
      title: `${profile?.displayName}님의 프로필`,
      text: profile?.bio || `${profile?.displayName}님의 MIRIP 프로필을 확인해보세요!`,
      url: shareUrl,
    };

    if (navigator.share && navigator.canShare?.(shareData)) {
      try {
        await navigator.share(shareData);
      } catch (err) {
        if (err.name !== 'AbortError') {
          // 사용자가 취소한 경우 무시
          console.error('공유 실패:', err);
        }
      }
    } else {
      // Web Share API 지원하지 않는 경우 클립보드에 복사
      try {
        await navigator.clipboard.writeText(shareUrl);
        alert('프로필 링크가 클립보드에 복사되었습니다!');
      } catch (err) {
        console.error('클립보드 복사 실패:', err);
      }
    }
  }, [profile]);

  // 로딩 상태
  if (loading) {
    return (
      <div className={styles.publicProfilePage}>
        <Header
          logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
          navItems={NAV_ITEMS}
          ctaButton={ctaButtonConfig}
        />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.loadingContainer} data-testid="public-profile-loading">
              <div className={styles.loadingSpinner} />
              <p>프로필 로딩 중...</p>
            </div>
          </div>
        </main>
        <Footer
          links={FOOTER_LINKS}
          copyright="© 2025 MIRIP. All rights reserved."
        />
      </div>
    );
  }

  // 에러 상태
  if (error) {
    return (
      <div className={styles.publicProfilePage}>
        <Header
          logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
          navItems={NAV_ITEMS}
          ctaButton={ctaButtonConfig}
        />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.errorContainer} data-testid="profile-error">
              <span className={styles.errorIcon}>{'\u26A0\uFE0F'}</span>
              <p className={styles.errorMessage}>{error}</p>
              <Link to="/" className={styles.homeButton}>
                홈으로 이동
              </Link>
            </div>
          </div>
        </main>
        <Footer
          links={FOOTER_LINKS}
          copyright="© 2025 MIRIP. All rights reserved."
        />
      </div>
    );
  }

  // 프로필 없음 (404)
  if (!profile) {
    return (
      <div className={styles.publicProfilePage}>
        <Header
          logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
          navItems={NAV_ITEMS}
          ctaButton={ctaButtonConfig}
        />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.notFoundContainer} data-testid="profile-not-found">
              <span className={styles.notFoundIcon}>{'\uD83D\uDD0D'}</span>
              <h2>사용자를 찾을 수 없습니다</h2>
              <p>@{username} 사용자가 존재하지 않습니다.</p>
              <Link to="/" className={styles.homeButton}>
                홈으로 이동
              </Link>
            </div>
          </div>
        </main>
        <Footer
          links={FOOTER_LINKS}
          copyright="© 2025 MIRIP. All rights reserved."
        />
      </div>
    );
  }

  // 비공개 프로필
  if (!profile.isPublic) {
    return (
      <div className={styles.publicProfilePage}>
        <Header
          logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
          navItems={NAV_ITEMS}
          ctaButton={ctaButtonConfig}
        />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.privateContainer} data-testid="profile-private">
              <span className={styles.privateIcon}>{'\uD83D\uDD12'}</span>
              <h2>비공개 프로필</h2>
              <p>@{username}님의 프로필은 비공개입니다.</p>
              <Link to="/" className={styles.homeButton}>
                홈으로 이동
              </Link>
            </div>
          </div>
        </main>
        <Footer
          links={FOOTER_LINKS}
          copyright="© 2025 MIRIP. All rights reserved."
        />
      </div>
    );
  }

  return (
    <div className={styles.publicProfilePage} data-testid="public-profile-page">
      <Header
        logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
        navItems={NAV_ITEMS}
        ctaButton={ctaButtonConfig}
      />

      <main className={styles.main}>
        <div className={styles.container}>
          {/* 프로필 헤더 섹션 */}
          <section className={styles.profileHeader}>
            {/* 아바타 */}
            <div className={styles.avatarContainer} data-testid="public-profile-avatar">
              {profile.profileImageUrl ? (
                <img
                  src={profile.profileImageUrl}
                  alt={profile.displayName}
                  className={styles.avatar}
                />
              ) : (
                <div className={styles.avatarPlaceholder}>
                  {getInitials(profile.displayName)}
                </div>
              )}
            </div>

            {/* 프로필 정보 */}
            <div className={styles.profileInfo}>
              <div className={styles.nameRow}>
                <h1 className={styles.displayName}>{profile.displayName}</h1>
                <TierBadge tier={profile.tier} size="medium" />
              </div>
              <p className={styles.username}>@{profile.username}</p>
              {profile.bio && (
                <p className={styles.bio}>{profile.bio}</p>
              )}

              {/* 공유 버튼 */}
              <button
                type="button"
                className={styles.shareButton}
                onClick={handleShare}
                data-testid="share-button"
                aria-label="프로필 공유"
              >
                <span className={styles.shareIcon}>{'\uD83D\uDD17'}</span>
                공유
              </button>
            </div>
          </section>

          {/* 활동 통계 섹션 */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>
              <span className={styles.sectionIcon}>{'\uD83D\uDCCA'}</span>
              활동 통계
            </h2>
            <div className={styles.statsGrid} data-testid="public-stats-section">
              <div className={styles.statItem}>
                <span className={styles.statValue}>{profile.totalActivities || 0}</span>
                <span className={styles.statLabel}>총 활동</span>
              </div>
              <div className={styles.statItem}>
                <span className={styles.statValue}>{profile.currentStreak || 0}</span>
                <span className={styles.statLabel}>연속 활동</span>
              </div>
              <div className={styles.statItem}>
                <span className={styles.statValue}>{profile.longestStreak || 0}</span>
                <span className={styles.statLabel}>최장 연속</span>
              </div>
              <div className={styles.statItem}>
                <span className={styles.statValue}>{awards.length}</span>
                <span className={styles.statLabel}>수상 횟수</span>
              </div>
            </div>
          </section>

          {/* 활동 잔디밭 섹션 */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>
              <span className={styles.sectionIcon}>{'\uD83C\uDF3F'}</span>
              활동 잔디밭
            </h2>
            <div className={styles.heatmapContainer} data-testid="public-heatmap-section">
              <ActivityHeatmap userId={profile.uid} readOnly={true} />
            </div>
          </section>

          {/* 수상 내역 섹션 */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>
              <span className={styles.sectionIcon}>{'\uD83C\uDFC6'}</span>
              수상 내역
            </h2>
            <div className={styles.achievementsContainer} data-testid="public-achievements-section">
              <AchievementList
                awards={awards}
                loading={awardsLoading}
                maxItems={5}
              />
            </div>
          </section>
        </div>
      </main>

      <Footer
        links={FOOTER_LINKS}
        copyright="© 2025 MIRIP. All rights reserved."
      />
    </div>
  );
};

export default PublicProfilePage;
