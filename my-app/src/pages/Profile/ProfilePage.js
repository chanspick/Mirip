// ProfilePage 컴포넌트
// SPEC-CRED-001: M2 마이페이지
// 사용자 프로필, 활동 히트맵, 타임라인을 표시합니다

import React, { useState, useCallback, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Header, Footer } from '../../components/common';
import { ActivityHeatmap, ActivityTimeline, StreakDisplay } from '../../components/credential';
import { useUserProfile } from '../../hooks';
import { auth } from '../../config/firebase';
import styles from './ProfilePage.module.css';

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
 * ProfilePage 컴포넌트
 * 마이페이지 - 사용자 프로필과 활동 현황을 표시합니다
 */
const ProfilePage = () => {
  const navigate = useNavigate();

  // 현재 로그인된 사용자 ID
  const currentUser = auth.currentUser;
  const userId = currentUser?.uid;

  // 프로필 훅
  const {
    profile,
    loading,
    error,
    updating,
    update,
  } = useUserProfile(userId);

  // 편집 모드 상태
  const [isEditing, setIsEditing] = useState(false);
  const [editForm, setEditForm] = useState({
    displayName: '',
    bio: '',
  });
  const [editError, setEditError] = useState(null);

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
   * 편집 모드 시작
   */
  const handleStartEdit = useCallback(() => {
    setEditForm({
      displayName: profile?.displayName || '',
      bio: profile?.bio || '',
    });
    setIsEditing(true);
    setEditError(null);
  }, [profile]);

  /**
   * 편집 취소
   */
  const handleCancelEdit = useCallback(() => {
    setIsEditing(false);
    setEditError(null);
  }, []);

  /**
   * 폼 입력 변경 핸들러
   */
  const handleInputChange = useCallback((e) => {
    const { name, value } = e.target;
    setEditForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  }, []);

  /**
   * 프로필 저장
   */
  const handleSaveProfile = useCallback(async () => {
    setEditError(null);

    try {
      const success = await update({
        displayName: editForm.displayName,
        bio: editForm.bio,
      });

      if (success) {
        setIsEditing(false);
      } else {
        setEditError('프로필 업데이트에 실패했습니다');
      }
    } catch (err) {
      setEditError(err.message || '프로필 업데이트 중 오류가 발생했습니다');
    }
  }, [update, editForm]);

  // 로그인하지 않은 경우
  if (!userId) {
    return (
      <div className={styles.profilePage}>
        <Header
          logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
          navItems={NAV_ITEMS}
          ctaButton={ctaButtonConfig}
        />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.loginRequired}>
              <span className={styles.loginIcon}>{'\uD83D\uDD12'}</span>
              <h2>로그인이 필요합니다</h2>
              <p>마이페이지를 보려면 로그인해주세요.</p>
              <Link to="/" className={styles.loginButton}>
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

  // 로딩 상태
  if (loading) {
    return (
      <div className={styles.profilePage}>
        <Header
          logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
          navItems={NAV_ITEMS}
          ctaButton={ctaButtonConfig}
        />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.loadingContainer} data-testid="profile-loading">
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
      <div className={styles.profilePage}>
        <Header
          logo={<Link to="/" className={styles.logo}>MIRIP</Link>}
          navItems={NAV_ITEMS}
          ctaButton={ctaButtonConfig}
        />
        <main className={styles.main}>
          <div className={styles.container}>
            <div className={styles.errorContainer}>
              <span className={styles.errorIcon}>{'\u26A0\uFE0F'}</span>
              <p className={styles.errorMessage}>{error}</p>
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
    <div className={styles.profilePage} data-testid="profile-page">
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
            <div className={styles.avatarContainer} data-testid="profile-avatar">
              {profile?.profileImageUrl ? (
                <img
                  src={profile.profileImageUrl}
                  alt={profile.displayName}
                  className={styles.avatar}
                />
              ) : (
                <div className={styles.avatarPlaceholder}>
                  {getInitials(profile?.displayName)}
                </div>
              )}
            </div>

            {/* 프로필 정보 */}
            <div className={styles.profileInfo}>
              {isEditing ? (
                // 편집 폼
                <div className={styles.editForm} data-testid="edit-form">
                  <div className={styles.formGroup}>
                    <label htmlFor="edit-displayName">이름</label>
                    <input
                      id="edit-displayName"
                      name="displayName"
                      type="text"
                      value={editForm.displayName}
                      onChange={handleInputChange}
                      className={styles.input}
                      data-testid="edit-displayName"
                      maxLength={50}
                    />
                  </div>
                  <div className={styles.formGroup}>
                    <label htmlFor="edit-bio">자기소개</label>
                    <textarea
                      id="edit-bio"
                      name="bio"
                      value={editForm.bio}
                      onChange={handleInputChange}
                      className={styles.textarea}
                      data-testid="edit-bio"
                      maxLength={500}
                      rows={3}
                    />
                    <span className={styles.charCount}>
                      {editForm.bio.length}/500
                    </span>
                  </div>
                  {editError && (
                    <p className={styles.editError}>{editError}</p>
                  )}
                  <div className={styles.editButtons}>
                    <button
                      type="button"
                      className={styles.cancelButton}
                      onClick={handleCancelEdit}
                      data-testid="cancel-edit-button"
                    >
                      취소
                    </button>
                    <button
                      type="button"
                      className={styles.saveButton}
                      onClick={handleSaveProfile}
                      disabled={updating}
                      data-testid="save-profile-button"
                    >
                      {updating ? '저장 중...' : '저장'}
                    </button>
                  </div>
                </div>
              ) : (
                // 프로필 표시
                <>
                  <h1 className={styles.displayName}>{profile?.displayName}</h1>
                  <p className={styles.username}>@{profile?.username}</p>
                  {profile?.bio && (
                    <p className={styles.bio}>{profile.bio}</p>
                  )}
                  <button
                    type="button"
                    className={styles.editButton}
                    onClick={handleStartEdit}
                    data-testid="edit-profile-button"
                  >
                    프로필 편집
                  </button>
                </>
              )}
            </div>
          </section>

          {/* 활동 현황 섹션 */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>
              <span className={styles.sectionIcon}>{'\uD83D\uDCCA'}</span>
              활동 현황
            </h2>
            <div className={styles.statsContainer} data-testid="streak-section">
              <StreakDisplay
                userId={userId}
                totalActivities={profile?.totalActivities}
              />
            </div>
          </section>

          {/* 활동 잔디밭 섹션 */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>
              <span className={styles.sectionIcon}>{'\uD83C\uDF3F'}</span>
              활동 잔디밭
            </h2>
            <div className={styles.heatmapContainer} data-testid="heatmap-section">
              <ActivityHeatmap userId={userId} />
            </div>
          </section>

          {/* 최근 활동 섹션 */}
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>
              <span className={styles.sectionIcon}>{'\uD83D\uDCCB'}</span>
              최근 활동
            </h2>
            <div className={styles.timelineContainer} data-testid="timeline-section">
              <ActivityTimeline userId={userId} limit={10} />
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

export default ProfilePage;
