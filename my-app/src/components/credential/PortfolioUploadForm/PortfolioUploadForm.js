// PortfolioUploadForm 컴포넌트
// SPEC-CRED-001: M4 포트폴리오 관리 - 포트폴리오 업로드 폼
// 이미지 업로드, 제목, 설명, 태그, 공개 설정을 포함하는 폼

import React, { useState, useRef, useCallback, useEffect } from 'react';
import PropTypes from 'prop-types';
import styles from './PortfolioUploadForm.module.css';

/**
 * PortfolioUploadForm 컴포넌트
 * 포트폴리오 등록/수정을 위한 폼
 *
 * @param {Object} props - 컴포넌트 props
 * @param {function} props.onSubmit - 제출 콜백 (data, file)
 * @param {function} props.onCancel - 취소 콜백
 * @param {Object} [props.initialData] - 편집 모드용 초기 데이터
 * @param {boolean} [props.isEdit=false] - 편집 모드 플래그
 * @param {boolean} [props.submitting=false] - 제출 중 상태
 */
const PortfolioUploadForm = ({
  onSubmit,
  onCancel,
  initialData,
  isEdit = false,
  submitting = false,
}) => {
  // 폼 상태
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState([]);
  const [tagInput, setTagInput] = useState('');
  const [isPublic, setIsPublic] = useState(true);

  // 이미지 상태
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // 에러 상태
  const [errors, setErrors] = useState({});

  // 파일 input ref
  const fileInputRef = useRef(null);

  // 초기 데이터 설정 (편집 모드)
  useEffect(() => {
    if (initialData) {
      setTitle(initialData.title || '');
      setDescription(initialData.description || '');
      setTags(initialData.tags || []);
      setIsPublic(initialData.isPublic !== false);
      if (initialData.imageUrl || initialData.thumbnailUrl) {
        setImagePreview(initialData.thumbnailUrl || initialData.imageUrl);
      }
    }
  }, [initialData]);

  // 이미지 파일 처리
  const handleImageFile = useCallback((file) => {
    if (!file) return;

    // 이미지 파일 타입 검증
    if (!file.type.startsWith('image/')) {
      setErrors((prev) => ({ ...prev, image: '이미지 파일만 업로드 가능합니다.' }));
      return;
    }

    setImageFile(file);
    setErrors((prev) => ({ ...prev, image: null }));

    // 미리보기 생성
    const reader = new FileReader();
    reader.onload = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  }, []);

  // 파일 선택 핸들러
  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    handleImageFile(file);
  };

  // 드래그 이벤트 핸들러
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    handleImageFile(file);
  };

  // 업로드 영역 클릭
  const handleUploadAreaClick = () => {
    fileInputRef.current?.click();
  };

  // 태그 추가
  const addTag = useCallback((tag) => {
    const trimmedTag = tag.trim();
    if (trimmedTag && !tags.includes(trimmedTag)) {
      setTags((prev) => [...prev, trimmedTag]);
    }
  }, [tags]);

  // 태그 입력 핸들러
  const handleTagInputChange = (e) => {
    const value = e.target.value;
    // 쉼표가 포함되면 태그 추가
    if (value.includes(',')) {
      const parts = value.split(',');
      parts.forEach((part, index) => {
        // 마지막 요소는 입력 필드에 남김
        if (index < parts.length - 1) {
          addTag(part);
        } else {
          setTagInput(part);
        }
      });
    } else {
      setTagInput(value);
    }
  };

  // 태그 입력 키 핸들러
  const handleTagKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTag(tagInput);
      setTagInput('');
    }
  };

  // 태그 삭제
  const handleRemoveTag = (tagToRemove) => {
    setTags((prev) => prev.filter((tag) => tag !== tagToRemove));
  };

  // 유효성 검사
  const validate = () => {
    const newErrors = {};

    if (!title.trim()) {
      newErrors.title = '제목을 입력해주세요.';
    }

    // 신규 등록 시 이미지 필수
    if (!isEdit && !imageFile && !imagePreview) {
      newErrors.image = '이미지를 선택해주세요.';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // 폼 제출
  const handleSubmit = (e) => {
    e.preventDefault();

    if (!validate()) {
      return;
    }

    const formData = {
      title: title.trim(),
      description: description.trim(),
      tags,
      isPublic,
    };

    // 편집 모드에서 initialData 포함
    if (isEdit && initialData) {
      formData.id = initialData.id;
      formData.imageUrl = initialData.imageUrl;
      formData.thumbnailUrl = initialData.thumbnailUrl;
    }

    onSubmit(formData, imageFile);
  };

  return (
    <form
      className={styles.form}
      onSubmit={handleSubmit}
      data-testid="upload-form"
    >
      {/* 이미지 업로드 영역 */}
      <div className={styles.formGroup}>
        <label className={styles.label}>작품 이미지 *</label>
        <div
          className={`${styles.uploadArea} ${isDragging ? styles.dragging : ''} ${imagePreview ? styles.hasImage : ''}`}
          onClick={handleUploadAreaClick}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          data-testid="image-upload-area"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && handleUploadAreaClick()}
        >
          {imagePreview ? (
            <div className={styles.previewContainer}>
              <img
                src={imagePreview}
                alt="미리보기"
                className={styles.preview}
                data-testid="image-preview"
              />
              <div className={styles.previewOverlay}>
                <span>클릭하여 변경</span>
              </div>
            </div>
          ) : (
            <div className={styles.uploadPlaceholder}>
              <svg
                className={styles.uploadIcon}
                width="48"
                height="48"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
              <p className={styles.uploadText}>
                이미지를 드래그하거나 클릭하여 업로드
              </p>
              <span className={styles.uploadHint}>
                JPG, PNG, GIF (최대 10MB)
              </span>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className={styles.fileInput}
            data-testid="file-input"
          />
        </div>
        {errors.image && (
          <span className={styles.errorMessage}>{errors.image}</span>
        )}
      </div>

      {/* 제목 입력 */}
      <div className={styles.formGroup}>
        <label htmlFor="title" className={styles.label}>
          제목 *
        </label>
        <input
          id="title"
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className={`${styles.input} ${errors.title ? styles.inputError : ''}`}
          placeholder="작품 제목을 입력하세요"
          maxLength={100}
        />
        {errors.title && (
          <span className={styles.errorMessage}>{errors.title}</span>
        )}
      </div>

      {/* 설명 입력 */}
      <div className={styles.formGroup}>
        <label htmlFor="description" className={styles.label}>
          설명
        </label>
        <textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className={styles.textarea}
          placeholder="작품에 대한 설명을 입력하세요"
          rows={4}
          maxLength={1000}
        />
      </div>

      {/* 태그 입력 */}
      <div className={styles.formGroup}>
        <label htmlFor="tags" className={styles.label}>
          태그
        </label>
        <div className={styles.tagsContainer}>
          {tags.map((tag) => (
            <span key={tag} className={styles.tagChip} data-testid="tag-chip">
              {tag}
              <button
                type="button"
                className={styles.removeTagButton}
                onClick={() => handleRemoveTag(tag)}
                data-testid="remove-tag-button"
                aria-label={`${tag} 태그 삭제`}
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </span>
          ))}
          <input
            id="tags"
            type="text"
            value={tagInput}
            onChange={handleTagInputChange}
            onKeyDown={handleTagKeyDown}
            className={styles.tagInput}
            placeholder={tags.length === 0 ? '태그 입력 후 Enter 또는 쉼표로 구분' : ''}
          />
        </div>
      </div>

      {/* 공개/비공개 토글 */}
      <div className={styles.formGroup}>
        <div className={styles.toggleRow}>
          <label htmlFor="isPublic" className={styles.toggleLabel}>
            공개 설정
          </label>
          <label className={styles.toggle}>
            <input
              id="isPublic"
              type="checkbox"
              checked={isPublic}
              onChange={(e) => setIsPublic(e.target.checked)}
              data-testid="public-toggle"
            />
            <span className={styles.toggleSlider} />
          </label>
          <span className={styles.toggleStatus}>
            {isPublic ? '공개' : '비공개'}
          </span>
        </div>
        <p className={styles.toggleHint}>
          {isPublic
            ? '다른 사용자가 이 작품을 볼 수 있습니다.'
            : '나만 이 작품을 볼 수 있습니다.'}
        </p>
      </div>

      {/* 버튼 영역 */}
      <div className={styles.buttonGroup}>
        <button
          type="button"
          className={styles.cancelButton}
          onClick={onCancel}
          data-testid="cancel-button"
          disabled={submitting}
        >
          취소
        </button>
        <button
          type="submit"
          className={styles.submitButton}
          data-testid="submit-button"
          disabled={submitting}
        >
          {submitting ? (
            <>
              <span className={styles.spinner} />
              처리 중...
            </>
          ) : (
            isEdit ? '수정' : '등록'
          )}
        </button>
      </div>
    </form>
  );
};

PortfolioUploadForm.propTypes = {
  /** 제출 콜백 */
  onSubmit: PropTypes.func.isRequired,
  /** 취소 콜백 */
  onCancel: PropTypes.func.isRequired,
  /** 편집 모드용 초기 데이터 */
  initialData: PropTypes.shape({
    id: PropTypes.string,
    title: PropTypes.string,
    description: PropTypes.string,
    imageUrl: PropTypes.string,
    thumbnailUrl: PropTypes.string,
    tags: PropTypes.arrayOf(PropTypes.string),
    isPublic: PropTypes.bool,
  }),
  /** 편집 모드 플래그 */
  isEdit: PropTypes.bool,
  /** 제출 중 상태 */
  submitting: PropTypes.bool,
};

export default PortfolioUploadForm;
