/**
 * 이미지 업로더 컴포넌트
 *
 * SPEC-COMP-001 REQ-C-011 구현
 * - 드래그앤드롭 지원
 * - 업로드 진행률 표시
 * - 이미지 미리보기
 * - 파일 유효성 검사 (JPG, PNG, WebP / 최대 10MB)
 *
 * @module components/competitions/ImageUploader
 */

import React, { useState, useRef, useCallback } from 'react';
import styles from './ImageUploader.module.css';

/**
 * 허용된 파일 타입
 */
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

/**
 * 최대 파일 크기 (10MB)
 */
const MAX_FILE_SIZE = 10 * 1024 * 1024;

/**
 * 파일 크기 포맷팅
 * @param {number} bytes - 바이트
 * @returns {string} 포맷된 크기
 */
const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * 이미지 업로더 컴포넌트
 *
 * @param {Object} props
 * @param {Function} props.onImageSelect - 이미지 선택 시 호출되는 콜백 (file, previewUrl)
 * @param {Function} props.onUpload - 업로드 시작 시 호출되는 콜백
 * @param {number} props.uploadProgress - 업로드 진행률 (0-100)
 * @param {boolean} props.isUploading - 업로드 중 여부
 * @param {string} props.error - 에러 메시지
 * @param {string} props.currentImageUrl - 기존 이미지 URL (수정 시)
 */
const ImageUploader = ({
  onImageSelect,
  uploadProgress = 0,
  isUploading = false,
  error,
  currentImageUrl,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState(currentImageUrl || null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const fileInputRef = useRef(null);

  /**
   * 파일 유효성 검사
   * @param {File} file - 검사할 파일
   * @returns {string|null} 에러 메시지 또는 null
   */
  const validateFile = (file) => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return 'JPG, PNG, WebP 형식의 이미지만 업로드할 수 있습니다.';
    }
    if (file.size > MAX_FILE_SIZE) {
      return `파일 크기는 10MB를 초과할 수 없습니다. (현재: ${formatFileSize(file.size)})`;
    }
    return null;
  };

  /**
   * 파일 처리
   * @param {File} file - 처리할 파일
   */
  const handleFile = useCallback((file) => {
    setValidationError(null);

    const error = validateFile(file);
    if (error) {
      setValidationError(error);
      return;
    }

    // 미리보기 URL 생성
    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);
    setSelectedFile(file);

    // 콜백 호출
    if (onImageSelect) {
      onImageSelect(file, previewUrl);
    }
  }, [onImageSelect]);

  /**
   * 드래그 이벤트 핸들러
   */
  const handleDragEnter = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  /**
   * 파일 선택 핸들러
   */
  const handleFileSelect = useCallback((e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  }, [handleFile]);

  /**
   * 영역 클릭 핸들러
   */
  const handleClick = useCallback(() => {
    if (!isUploading && fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, [isUploading]);

  /**
   * 이미지 제거 핸들러
   */
  const handleRemove = useCallback((e) => {
    e.stopPropagation();
    setPreview(null);
    setSelectedFile(null);
    setValidationError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (onImageSelect) {
      onImageSelect(null, null);
    }
  }, [onImageSelect]);

  const displayError = validationError || error;

  return (
    <div className={styles.container}>
      <input
        ref={fileInputRef}
        type="file"
        accept={ALLOWED_TYPES.join(',')}
        onChange={handleFileSelect}
        className={styles.hiddenInput}
        disabled={isUploading}
      />

      <div
        className={`${styles.dropzone} ${isDragging ? styles.dragging : ''} ${preview ? styles.hasPreview : ''} ${isUploading ? styles.uploading : ''}`}
        onClick={handleClick}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        aria-label="이미지 업로드"
      >
        {preview ? (
          <div className={styles.previewContainer}>
            <img
              src={preview}
              alt="미리보기"
              className={styles.previewImage}
            />
            {!isUploading && (
              <button
                type="button"
                className={styles.removeButton}
                onClick={handleRemove}
                aria-label="이미지 제거"
              >
                ✕
              </button>
            )}
            {isUploading && (
              <div className={styles.uploadOverlay}>
                <div className={styles.progressContainer}>
                  <div
                    className={styles.progressBar}
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <span className={styles.progressText}>
                  {uploadProgress}% 업로드 중...
                </span>
              </div>
            )}
          </div>
        ) : (
          <div className={styles.placeholder}>
            <div className={styles.uploadIcon}>📷</div>
            <p className={styles.placeholderText}>
              {isDragging
                ? '이미지를 여기에 놓으세요'
                : '클릭하거나 이미지를 드래그하세요'}
            </p>
            <p className={styles.placeholderHint}>
              JPG, PNG, WebP (최대 10MB)
            </p>
          </div>
        )}
      </div>

      {selectedFile && !isUploading && (
        <div className={styles.fileInfo}>
          <span className={styles.fileName}>{selectedFile.name}</span>
          <span className={styles.fileSize}>
            {formatFileSize(selectedFile.size)}
          </span>
        </div>
      )}

      {displayError && (
        <p className={styles.error}>{displayError}</p>
      )}
    </div>
  );
};

export default ImageUploader;
