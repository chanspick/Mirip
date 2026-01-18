# -*- coding: utf-8 -*-
"""
데이터 수집 모듈

다양한 소스에서 이미지를 수집하는 기능을 제공합니다.
"""

from data_pipeline.collectors.base_collector import BaseCollector, CollectionResult
from data_pipeline.collectors.image_collector import ImageCollector

__all__ = [
    "BaseCollector",
    "CollectionResult",
    "ImageCollector",
]
