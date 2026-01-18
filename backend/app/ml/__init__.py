# ML Modules
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
"""
ML 모듈

Phase 1 (SPEC-AI-001):
- feature_extractor: DINOv2 기반 특징 추출기
- projector: 특징 벡터 투영 레이어

Phase 3 (SPEC-AI-001):
- ranking_model: Pairwise Ranking 모델

Phase 2 (예정):
- fusion_module: 멀티브랜치 피처 융합
- rubric_heads: 4축 루브릭 평가 헤드
- tier_classifier: GMM 기반 티어 분류기
"""

from app.ml.feature_extractor import DINOv2FeatureExtractor
from app.ml.projector import Projector
from app.ml.ranking_model import PairwiseRankingModel

__all__ = [
    "DINOv2FeatureExtractor",
    "Projector",
    "PairwiseRankingModel",
]
