// RegistrationForm 컴포넌트
// SPEC-FIREBASE-001: 사전 등록 폼 컴포넌트
// 이름, 이메일, 유저 유형을 수집하여 Firestore에 저장

import React, { useState } from 'react';
import PropTypes from 'prop-types';
import Button from '../../common/Button';
import styles from './RegistrationForm.module.css';
import * as registrationService from '../../../services/registrationService';

/**
 * 유저 유형 옵션 목록
 * @type {Array<{value: string, label: string}>}
 */
const USER_TYPE_OPTIONS = [
  { value: 'student', label: '입시생' },
  { value: 'parent', label: '학부모' },
  { value: 'artist', label: '신진 작가' },
  { value: 'organizer', label: '공모전 주최자' },
];

/**
 * 이메일 유효성 검사 정규식
 * @type {RegExp}
 */
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * RegistrationForm 컴포넌트
 * 사전 등록을 위한 폼 컴포넌트로, 이름, 이메일, 유저 유형을 수집합니다.
 *
 * @param {Object} props - 컴포넌트 props
 * @param {Function} props.onSuccess - 등록 성공 시 호출되는 콜백 함수
 * @param {Function} props.onError - 등록 실패 시 호출되는 콜백 함수
 * @param {string} props.className - 추가 CSS 클래스
 */
const RegistrationForm = ({ onSuccess, onError, className = '' }) => {
  // 폼 데이터 상태
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    userType: '',
  });

  // 유효성 검사 에러 상태
  const [errors, setErrors] = useState({
    name: '',
    email: '',
    userType: '',
  });

  // 로딩 상태
  const [isSubmitting, setIsSubmitting] = useState(false);

  // 서버 에러 상태
  const [serverError, setServerError] = useState('');

  /**
   * 개별 필드 유효성 검사
   * @param {string} fieldName - 필드 이름
   * @param {string} value - 필드 값
   * @returns {string} 에러 메시지 (에러가 없으면 빈 문자열)
   */
  const validateField = (fieldName, value) => {
    switch (fieldName) {
      case 'name':
        if (!value || value.trim() === '') {
          return '이름을 입력해주세요';
        }
        if (value.trim().length < 2) {
          return '이름은 2자 이상이어야 합니다';
        }
        return '';

      case 'email':
        if (!value || value.trim() === '') {
          return '이메일을 입력해주세요';
        }
        if (!EMAIL_REGEX.test(value)) {
          return '올바른 이메일을 입력해주세요';
        }
        return '';

      case 'userType':
        if (!value || value === '') {
          return '유저 유형을 선택해주세요';
        }
        return '';

      default:
        return '';
    }
  };

  /**
   * 전체 폼 유효성 검사
   * @returns {boolean} 모든 필드가 유효하면 true
   */
  const validateForm = () => {
    const newErrors = {
      name: validateField('name', formData.name),
      email: validateField('email', formData.email),
      userType: validateField('userType', formData.userType),
    };

    setErrors(newErrors);

    // 모든 에러가 비어있으면 유효함
    return Object.values(newErrors).every((error) => error === '');
  };

  /**
   * 입력 값 변경 핸들러
   * @param {React.ChangeEvent<HTMLInputElement | HTMLSelectElement>} e - 이벤트 객체
   */
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    // 입력 시 해당 필드 에러 초기화
    if (errors[name]) {
      setErrors((prev) => ({
        ...prev,
        [name]: '',
      }));
    }

    // 서버 에러 초기화
    if (serverError) {
      setServerError('');
    }
  };

  /**
   * 필드 blur 이벤트 핸들러 (실시간 유효성 검사)
   * @param {React.FocusEvent<HTMLInputElement | HTMLSelectElement>} e - 이벤트 객체
   */
  const handleBlur = (e) => {
    const { name, value } = e.target;
    const error = validateField(name, value);
    setErrors((prev) => ({
      ...prev,
      [name]: error,
    }));
  };

  /**
   * 폼 제출 핸들러
   * @param {React.FormEvent<HTMLFormElement>} e - 이벤트 객체
   */
  const handleSubmit = async (e) => {
    e.preventDefault();

    // 유효성 검사 실패 시 제출 중단
    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    setServerError('');

    try {
      // registrationService를 통해 데이터 저장
      const result = await registrationService.create({
        name: formData.name.trim(),
        email: formData.email.trim(),
        userType: formData.userType,
      });

      // 성공 콜백 호출
      if (onSuccess) {
        onSuccess(result);
      }
    } catch (error) {
      // 서버 에러 표시
      setServerError(error.message);

      // 에러 콜백 호출
      if (onError) {
        onError(error);
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  // 클래스 이름 조합
  const formClasses = [styles.form, className].filter(Boolean).join(' ');

  return (
    <form className={formClasses} onSubmit={handleSubmit} noValidate>
      {/* 이름 입력 필드 */}
      <div className={styles.formGroup}>
        <label htmlFor="name" className={styles.label}>
          이름
        </label>
        <input
          type="text"
          id="name"
          name="name"
          value={formData.name}
          onChange={handleChange}
          onBlur={handleBlur}
          disabled={isSubmitting}
          className={`${styles.input} ${errors.name ? styles.inputError : ''}`}
          placeholder="이름을 입력해주세요"
        />
        {errors.name && <span className={styles.errorMessage}>{errors.name}</span>}
      </div>

      {/* 이메일 입력 필드 */}
      <div className={styles.formGroup}>
        <label htmlFor="email" className={styles.label}>
          이메일
        </label>
        <input
          type="email"
          id="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          onBlur={handleBlur}
          disabled={isSubmitting}
          className={`${styles.input} ${errors.email ? styles.inputError : ''}`}
          placeholder="이메일을 입력해주세요"
        />
        {errors.email && <span className={styles.errorMessage}>{errors.email}</span>}
      </div>

      {/* 유저 유형 선택 필드 */}
      <div className={styles.formGroup}>
        <label htmlFor="userType" className={styles.label}>
          유저 유형
        </label>
        <select
          id="userType"
          name="userType"
          value={formData.userType}
          onChange={handleChange}
          onBlur={handleBlur}
          disabled={isSubmitting}
          className={`${styles.select} ${errors.userType ? styles.inputError : ''}`}
        >
          <option value="">유저 유형을 선택해주세요</option>
          {USER_TYPE_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        {errors.userType && <span className={styles.errorMessage}>{errors.userType}</span>}
      </div>

      {/* 서버 에러 메시지 */}
      {serverError && <div className={styles.serverError}>{serverError}</div>}

      {/* 제출 버튼 */}
      <Button
        type="submit"
        variant="cta"
        fullWidth
        disabled={isSubmitting}
      >
        {isSubmitting ? '등록 중...' : '등록하기'}
      </Button>
    </form>
  );
};

RegistrationForm.propTypes = {
  /** 등록 성공 시 호출되는 콜백 함수 */
  onSuccess: PropTypes.func,
  /** 등록 실패 시 호출되는 콜백 함수 */
  onError: PropTypes.func,
  /** 추가 CSS 클래스 */
  className: PropTypes.string,
};

export default RegistrationForm;
