# -*- coding: utf-8 -*-
"""
크롤러 모듈

midaeipsi.com에서 미대 입시 합격작 이미지와 메타데이터를 수집하고,
수집된 데이터를 정제, 정규화, 중복 제거하는 기능을 제공합니다.

주요 구성:
    - MidaeipsiCrawler: 웹 크롤링 및 데이터 수집
    - DataCleaner: 데이터 정제 오케스트레이터
    - ImageDeduplicator: perceptual hash 기반 이미지 중복 제거
    - normalizer: 정규식 기반 대학명/학과명 정규화
    - StatsReporter: 통계 리포트 생성
    - CrawledDataConverter: 크롤링 데이터 → 학습 파이프라인 변환
"""

from data_pipeline.crawlers.cleaner import CleaningResult, DataCleaner
from data_pipeline.crawlers.config import (
    BASE_URL,
    BOARD_NAME,
    POST_RANGE,
)
from data_pipeline.crawlers.converter import (
    ConversionResult,
    CrawledDataConverter,
    ValidationReport,
)
from data_pipeline.crawlers.crawler import MidaeipsiCrawler
from data_pipeline.crawlers.dedup import DeduplicationResult, ImageDeduplicator
from data_pipeline.crawlers.normalizer import (
    determine_tier,
    extract_department_from_title,
    extract_university_from_title,
    normalize_department,
    normalize_university,
)
from data_pipeline.crawlers.parser import PostParser
from data_pipeline.crawlers.stats import StatsReport, StatsReporter

__all__ = [
    # 크롤러
    "BASE_URL",
    "BOARD_NAME",
    "POST_RANGE",
    "MidaeipsiCrawler",
    "PostParser",
    # 데이터 정제
    "DataCleaner",
    "CleaningResult",
    # 중복 제거
    "ImageDeduplicator",
    "DeduplicationResult",
    # 정규화 함수
    "normalize_university",
    "normalize_department",
    "extract_university_from_title",
    "extract_department_from_title",
    "determine_tier",
    # 통계
    "StatsReporter",
    "StatsReport",
    # 변환기 (Phase 1.3)
    "CrawledDataConverter",
    "ConversionResult",
    "ValidationReport",
]
