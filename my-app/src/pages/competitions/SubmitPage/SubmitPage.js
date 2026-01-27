/**
 * 공모전 출품 페이지
 *
 * SPEC-COMP-001 REQ-C-012 ~ REQ-C-015 구현
 *
 * @module pages/competitions/SubmitPage
 */

import React, { useState, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Loading } from '../../../components/common';
import ImageUploader from '../../../components/competitions/ImageUploader';
import { useCompetitionDetail } from '../../../hooks/useCompetitions';
import { useImageUpload, useCreateSubmission } from '../../../hooks/useSubmission';
import styles from './SubmitPage.module.css';

/**
 * 출품 페이지 컴포넌트
 */
const SubmitPage = () => {
  const { id: competitionId } = useParams();
  const navigate = useNavigate();

  // 폼 상태
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    artistName: '',
    artistEmail: '',
    agreeTerms: false,
  });
  const [selectedImage, setSelectedImage] = useState(null);
  const [formErrors, setFormErrors] = useState({});

  // 공모전 정보 가져오기
  const { competition, loading: competitionLoading, error: competitionError } = useCompetitionDetail(competitionId);

  // 이미지 업로드 훅
  const { uploadImage, progress: uploadProgress, uploading, error: uploadError } = useImageUpload();

  // 제출 훅
  const { createSubmission, submitting, error: submitError } = useCreateSubmission();

  /**
   * 입력 변경 핸들러
   */
  const handleInputChange = useCallback((e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
    // 에러 클리어
    if (formErrors[name]) {
      setFormErrors((prev) => ({ ...prev, [name]: null }));
    }
  }, [formErrors]);

  /**
   * 이미지 선택 핸들러
   */
  const handleImageSelect = useCallback((file, previewUrl) => {
    setSelectedImage({ file, previewUrl });
    if (formErrors.image) {
      setFormErrors((prev) => ({ ...prev, image: null }));
    }
  }, [formErrors]);

  /**
   * 폼 유효성 검사
   */
  const validateForm = () => {
    const errors = {};

    if (!formData.title.trim()) {
      errors.title = '작품명을 입력해주세요.';
    } else if (formData.title.length > 100) {
      errors.title = '작품명은 100자를 초과할 수 없습니다.';
    }

    if (!formData.description.trim()) {
      errors.description = '작품 설명을 입력해주세요.';
    } else if (formData.description.length > 1000) {
      errors.description = '작품 설명은 1000자를 초과할 수 없습니다.';
    }

    if (!formData.artistName.trim()) {
      errors.artistName = '작가명을 입력해주세요.';
    }

    if (!formData.artistEmail.trim()) {
      errors.artistEmail = '이메일을 입력해주세요.';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.artistEmail)) {
      errors.artistEmail = '올바른 이메일 형식을 입력해주세요.';
    }

    if (!selectedImage?.file) {
      errors.image = '작품 이미지를 업로드해주세요.';
    }

    if (!formData.agreeTerms) {
      errors.agreeTerms = '약관에 동의해주세요.';
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  /**
   * 폼 제출 핸들러
   */
  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    try {
      // 1. 이미지 업로드
      const imageUrl = await uploadImage(selectedImage.file, competitionId);

      // 2. 출품 데이터 생성
      const submissionData = {
        competitionId,
        title: formData.title.trim(),
        description: formData.description.trim(),
        artistName: formData.artistName.trim(),
        artistEmail: formData.artistEmail.trim(),
        imageUrl,
      };

      await createSubmission(submissionData);

      // 3. 성공 시 상세 페이지로 이동
      navigate(`/competitions/${competitionId}`, {
        state: { submitSuccess: true },
      });
    } catch (err) {
      // 에러는 훅에서 처리됨
      console.error('제출 실패:', err);
    }
  };

  // 로딩 상태
  if (competitionLoading) {
    return (
      <div className={styles.loadingContainer}>
        <Loading size="large" />
      </div>
    );
  }

  // 에러 상태
  if (competitionError || !competition) {
    return (
      <div className={styles.errorContainer}>
        <h2>공모전을 찾을 수 없습니다</h2>
        <p>{competitionError || '요청하신 공모전이 존재하지 않습니다.'}</p>
        <Link to="/competitions" className={styles.backLink}>
          목록으로 돌아가기
        </Link>
      </div>
    );
  }

  // 마감된 공모전 체크
  const isExpired = new Date() > new Date(competition.endDate);
  if (isExpired) {
    return (
      <div className={styles.expiredContainer}>
        <h2>마감된 공모전입니다</h2>
        <p>이 공모전의 접수가 마감되었습니다.</p>
        <Link to={`/competitions/${competitionId}`} className={styles.backLink}>
          공모전 상세로 돌아가기
        </Link>
      </div>
    );
  }

  const displayError = uploadError || submitError;
  const isSubmitting = uploading || submitting;

  return (
    <div className={styles.container}>
      {/* 헤더 */}
      <header className={styles.header}>
        <Link to={`/competitions/${competitionId}`} className={styles.backButton}>
          ← 공모전으로 돌아가기
        </Link>
        <h1 className={styles.title}>작품 출품</h1>
        <p className={styles.competitionTitle}>{competition.title}</p>
      </header>

      {/* 출품 폼 */}
      <form onSubmit={handleSubmit} className={styles.form}>
        {/* 이미지 업로드 */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>작품 이미지 *</h2>
          <ImageUploader
            onImageSelect={handleImageSelect}
            uploadProgress={uploadProgress}
            isUploading={uploading}
            error={formErrors.image}
          />
        </section>

        {/* 작품 정보 */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>작품 정보</h2>

          <div className={styles.formGroup}>
            <label htmlFor="title" className={styles.label}>
              작품명 *
            </label>
            <input
              type="text"
              id="title"
              name="title"
              value={formData.title}
              onChange={handleInputChange}
              placeholder="작품의 제목을 입력해주세요"
              maxLength={100}
              className={`${styles.input} ${formErrors.title ? styles.inputError : ''}`}
              disabled={isSubmitting}
            />
            {formErrors.title && (
              <p className={styles.errorText}>{formErrors.title}</p>
            )}
            <p className={styles.charCount}>{formData.title.length}/100</p>
          </div>

          <div className={styles.formGroup}>
            <label htmlFor="description" className={styles.label}>
              작품 설명 *
            </label>
            <textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleInputChange}
              placeholder="작품에 대한 설명, 제작 의도, 사용한 기법 등을 입력해주세요"
              maxLength={1000}
              rows={6}
              className={`${styles.textarea} ${formErrors.description ? styles.inputError : ''}`}
              disabled={isSubmitting}
            />
            {formErrors.description && (
              <p className={styles.errorText}>{formErrors.description}</p>
            )}
            <p className={styles.charCount}>{formData.description.length}/1000</p>
          </div>
        </section>

        {/* 작가 정보 */}
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>작가 정보</h2>

          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label htmlFor="artistName" className={styles.label}>
                작가명 *
              </label>
              <input
                type="text"
                id="artistName"
                name="artistName"
                value={formData.artistName}
                onChange={handleInputChange}
                placeholder="실명 또는 예명"
                className={`${styles.input} ${formErrors.artistName ? styles.inputError : ''}`}
                disabled={isSubmitting}
              />
              {formErrors.artistName && (
                <p className={styles.errorText}>{formErrors.artistName}</p>
              )}
            </div>

            <div className={styles.formGroup}>
              <label htmlFor="artistEmail" className={styles.label}>
                이메일 *
              </label>
              <input
                type="email"
                id="artistEmail"
                name="artistEmail"
                value={formData.artistEmail}
                onChange={handleInputChange}
                placeholder="example@email.com"
                className={`${styles.input} ${formErrors.artistEmail ? styles.inputError : ''}`}
                disabled={isSubmitting}
              />
              {formErrors.artistEmail && (
                <p className={styles.errorText}>{formErrors.artistEmail}</p>
              )}
            </div>
          </div>
        </section>

        {/* 약관 동의 */}
        <section className={styles.section}>
          <div className={styles.termsBox}>
            <h3>출품 약관</h3>
            <ul>
              <li>제출한 작품은 본인이 직접 창작한 것임을 확인합니다.</li>
              <li>타인의 저작권을 침해하지 않았음을 확인합니다.</li>
              <li>주최 측의 홍보 목적으로 작품이 사용될 수 있습니다.</li>
              <li>심사 결과에 대한 이의 제기는 발표 후 7일 이내 가능합니다.</li>
            </ul>
          </div>

          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              name="agreeTerms"
              checked={formData.agreeTerms}
              onChange={handleInputChange}
              className={styles.checkbox}
              disabled={isSubmitting}
            />
            <span>위 약관에 동의합니다 *</span>
          </label>
          {formErrors.agreeTerms && (
            <p className={styles.errorText}>{formErrors.agreeTerms}</p>
          )}
        </section>

        {/* 에러 메시지 */}
        {displayError && (
          <div className={styles.submitError}>
            <p>{displayError}</p>
          </div>
        )}

        {/* 제출 버튼 */}
        <div className={styles.submitContainer}>
          <button
            type="submit"
            className={styles.submitButton}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <Loading size="small" />
                <span>{uploading ? '업로드 중...' : '제출 중...'}</span>
              </>
            ) : (
              '작품 제출하기'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SubmitPage;
