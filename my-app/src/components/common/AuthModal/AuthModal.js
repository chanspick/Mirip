// AuthModal 컴포넌트
// 소셜 로그인(Google)을 지원하는 인증 모달
// SPEC-UI-001: 공통 UI 컴포넌트

import React, { useState, useCallback } from 'react';
import PropTypes from 'prop-types';
import { GoogleAuthProvider, signInWithPopup } from 'firebase/auth';
import { auth } from '../../../config/firebase';
import Modal from '../Modal';
import styles from './AuthModal.module.css';

// Google 인증 제공자
const googleProvider = new GoogleAuthProvider();

/**
 * AuthModal 컴포넌트
 * 소셜 로그인을 지원하는 인증 모달
 *
 * @param {Object} props - 컴포넌트 props
 * @param {boolean} props.isOpen - 모달 열림 상태
 * @param {Function} props.onClose - 모달 닫기 핸들러
 * @param {Function} props.onSuccess - 로그인 성공 콜백
 */
const AuthModal = ({ isOpen, onClose, onSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Google 로그인 핸들러
   */
  const handleGoogleSignIn = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;

      if (onSuccess) {
        onSuccess(user);
      }
      onClose();
    } catch (err) {
      console.error('[AuthModal] Google 로그인 실패:', err);

      // 사용자가 취소한 경우는 에러로 표시하지 않음
      if (err.code !== 'auth/popup-closed-by-user') {
        setError(getErrorMessage(err.code));
      }
    } finally {
      setLoading(false);
    }
  }, [onClose, onSuccess]);

  /**
   * 에러 코드를 한글 메시지로 변환
   */
  const getErrorMessage = (errorCode) => {
    const errorMessages = {
      'auth/popup-blocked': '팝업이 차단되었습니다. 팝업 차단을 해제해주세요.',
      'auth/network-request-failed': '네트워크 오류가 발생했습니다. 인터넷 연결을 확인해주세요.',
      'auth/too-many-requests': '너무 많은 요청이 있었습니다. 잠시 후 다시 시도해주세요.',
      'auth/cancelled-popup-request': '로그인이 취소되었습니다.',
    };

    return errorMessages[errorCode] || '로그인 중 오류가 발생했습니다. 다시 시도해주세요.';
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="로그인">
      <div className={styles.container}>
        {/* 로고 영역 */}
        <div className={styles.logoSection}>
          <h2 className={styles.logo}>MIRIP</h2>
          <p className={styles.subtitle}>당신의 작품, 어디까지 갈 수 있을까요?</p>
        </div>

        {/* 소셜 로그인 버튼 */}
        <div className={styles.socialButtons}>
          <button
            type="button"
            className={styles.googleButton}
            onClick={handleGoogleSignIn}
            disabled={loading}
            data-testid="google-signin-button"
          >
            {loading ? (
              <span className={styles.loadingSpinner} />
            ) : (
              <>
                <svg className={styles.googleIcon} viewBox="0 0 24 24">
                  <path
                    fill="#4285F4"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  />
                </svg>
                <span>Google로 계속하기</span>
              </>
            )}
          </button>

          {/* 카카오 로그인은 추후 구현 예정 */}
          {/* <button
            type="button"
            className={styles.kakaoButton}
            disabled={loading}
          >
            <span>카카오로 계속하기</span>
          </button> */}
        </div>

        {/* 에러 메시지 */}
        {error && (
          <p className={styles.errorMessage} data-testid="auth-error">
            {error}
          </p>
        )}

        {/* 이용약관 안내 */}
        <p className={styles.terms}>
          로그인 시{' '}
          <a href="/terms" target="_blank" rel="noopener noreferrer">
            이용약관
          </a>
          {' '}및{' '}
          <a href="/privacy" target="_blank" rel="noopener noreferrer">
            개인정보처리방침
          </a>
          에 동의하는 것으로 간주됩니다.
        </p>
      </div>
    </Modal>
  );
};

AuthModal.propTypes = {
  /** 모달 열림 상태 */
  isOpen: PropTypes.bool.isRequired,
  /** 모달 닫기 핸들러 */
  onClose: PropTypes.func.isRequired,
  /** 로그인 성공 콜백 */
  onSuccess: PropTypes.func,
};

export default AuthModal;
