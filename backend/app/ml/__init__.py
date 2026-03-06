# ML Modules
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
"""
ML 모듈

Phase 1 (SPEC-AI-001):
- feature_extractor: DINOv2 기반 특징 추출기
- projector: 특징 벡터 투영 레이어
- ranking_model: Pairwise Ranking 모델

Phase 2 (Multi-Branch):
- edge_branch: PiDiNet-tiny 엣지 검출 브랜치
- fusion_module: GatedFusion 멀티브랜치 피처 융합
- rubric_heads: 4축 루브릭 평가 헤드
- multi_branch_model: Multi-Branch Ranking 모델
"""

from app.ml.feature_extractor import DINOv2FeatureExtractor
from app.ml.projector import Projector
from app.ml.ranking_model import PairwiseRankingModel
from app.ml.edge_branch import EdgeBranch
from app.ml.fusion_module import GatedFusion
from app.ml.rubric_heads import RubricHeads
from app.ml.multi_branch_model import MultiBranchRankingModel

__all__ = [
    "DINOv2FeatureExtractor",
    "Projector",
    "PairwiseRankingModel",
    "EdgeBranch",
    "GatedFusion",
    "RubricHeads",
    "MultiBranchRankingModel",
]
