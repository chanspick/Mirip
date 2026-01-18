# -*- coding: utf-8 -*-
"""
메타데이터 태깅 모듈

과별 분류, 티어 라벨링, 키워드 추출 기능을 제공합니다.
"""

from data_pipeline.taggers.base_tagger import BaseTagger, TagResult
from data_pipeline.taggers.department_tagger import DepartmentTagger
from data_pipeline.taggers.tier_tagger import TierTagger

__all__ = [
    "BaseTagger",
    "TagResult",
    "DepartmentTagger",
    "TierTagger",
]
