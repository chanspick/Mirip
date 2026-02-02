# tier_ranker.py
# 티어 판정 모듈
"""
Tier Ranker

새 이미지를 앵커 이미지들과 비교하여 티어를 판정합니다.

Usage:
    ranker = TierRanker(model, anchor_manager, device='cuda')
    result = ranker.rank(image_tensor)
    print(result)
    # {'tier': 'B', 'confidence': 0.72, 'win_rates': {'S': 0.2, 'A': 0.4, 'B': 0.7, 'C': 1.0}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.ml.anchor_manager import AnchorManager


class TierRanker:
    """
    앵커 기반 티어 판정기

    새 이미지의 feature를 추출하고, 저장된 앵커 feature들과
    pairwise 비교하여 최종 티어를 판정합니다.
    """

    TIERS = ['S', 'A', 'B', 'C']
    TIER_VALUES = {'S': 4, 'A': 3, 'B': 2, 'C': 1}
    DEFAULT_IMAGE_SIZE = 448
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model: nn.Module,
        anchor_manager: AnchorManager,
        device: Union[str, torch.device] = 'cpu',
    ) -> None:
        """
        Args:
            model: PairwiseRankingModel 인스턴스
            anchor_manager: 앵커 feature가 로드된 AnchorManager
            device: 연산 디바이스
        """
        self.model = model
        self.anchor_manager = anchor_manager
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # 이미지 전처리
        self._transform = transforms.Compose([
            transforms.Resize((self.DEFAULT_IMAGE_SIZE, self.DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        # 앵커 feature를 디바이스로 미리 이동
        self._anchor_features: Dict[str, torch.Tensor] = {}
        for tier in self.TIERS:
            feats = anchor_manager.get_tier_features(tier)
            if feats is not None:
                self._anchor_features[tier] = feats.to(self.device)

    def _extract_feature(self, image: torch.Tensor) -> torch.Tensor:
        """이미지에서 feature 추출 (feature_extractor + projector)"""
        with torch.no_grad():
            feat = self.model.feature_extractor(image)
            feat = self.model.projector(feat)
        return feat

    def _compute_score(self, feature: torch.Tensor) -> torch.Tensor:
        """Feature에서 score 계산 (score_head)"""
        with torch.no_grad():
            score = self.model.score_head(feature)
        return score

    def rank(
        self,
        image: Union[torch.Tensor, Image.Image, str, Path],
    ) -> Dict:
        """
        이미지의 티어를 판정합니다.

        Args:
            image: 입력 이미지 (텐서, PIL Image, 또는 경로)

        Returns:
            {
                'tier': 'B',           # 최종 판정 티어
                'confidence': 0.72,    # 판정 신뢰도
                'win_rates': {         # 각 티어 앵커 대비 승률
                    'S': 0.2,
                    'A': 0.4,
                    'B': 0.7,
                    'C': 1.0
                },
                'score': 0.45,         # 원본 스코어
                'details': {...}       # 상세 비교 결과
            }
        """
        # 이미지 로드 및 전처리
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        if isinstance(image, Image.Image):
            image = self._transform(image).unsqueeze(0)

        image = image.to(self.device)

        # Feature 추출
        input_feature = self._extract_feature(image)
        input_score = self._compute_score(input_feature).item()

        # 각 티어 앵커와 비교
        win_rates = {}
        details = {}

        for tier in self.TIERS:
            if tier not in self._anchor_features:
                continue

            anchor_feats = self._anchor_features[tier]
            n_anchors = anchor_feats.shape[0]

            # 앵커들의 score 계산
            anchor_scores = self._compute_score(anchor_feats)  # (n_anchors, 1)

            # 승패 계산 (input이 anchor보다 높으면 승)
            wins = (input_score > anchor_scores.squeeze()).sum().item()
            win_rate = wins / n_anchors

            win_rates[tier] = win_rate
            details[tier] = {
                'wins': int(wins),
                'losses': n_anchors - int(wins),
                'total': n_anchors,
                'anchor_scores': anchor_scores.squeeze().tolist(),
            }

        # 티어 판정 로직
        tier, confidence = self._determine_tier(win_rates)

        return {
            'tier': tier,
            'confidence': confidence,
            'win_rates': win_rates,
            'score': input_score,
            'details': details,
        }

    def _determine_tier(self, win_rates: Dict[str, float]) -> Tuple[str, float]:
        """
        승률 기반 티어 판정

        로직:
        - S 앵커 대비 승률 >= 0.5 → S
        - A 앵커 대비 승률 >= 0.5 → A
        - B 앵커 대비 승률 >= 0.5 → B
        - 그 외 → C
        """
        # 승률 기반 판정
        if win_rates.get('S', 0) >= 0.5:
            tier = 'S'
            confidence = win_rates['S']
        elif win_rates.get('A', 0) >= 0.5:
            tier = 'A'
            confidence = win_rates['A']
        elif win_rates.get('B', 0) >= 0.5:
            tier = 'B'
            confidence = win_rates['B']
        else:
            tier = 'C'
            confidence = 1.0 - win_rates.get('C', 0)

        return tier, round(confidence, 3)

    def rank_batch(
        self,
        images: List[Union[torch.Tensor, Image.Image, str, Path]],
    ) -> List[Dict]:
        """여러 이미지를 한 번에 판정"""
        return [self.rank(img) for img in images]

    def explain(self, result: Dict) -> str:
        """판정 결과를 자연어로 설명"""
        tier = result['tier']
        confidence = result['confidence']
        win_rates = result['win_rates']

        lines = [
            f"## 티어 판정 결과: {tier}",
            f"신뢰도: {confidence:.1%}",
            "",
            "### 앵커 비교 결과:",
        ]

        for t in self.TIERS:
            if t in win_rates:
                wr = win_rates[t]
                detail = result['details'].get(t, {})
                wins = detail.get('wins', 0)
                total = detail.get('total', 0)
                lines.append(f"- vs {t}티어: {wins}/{total} 승 ({wr:.1%})")

        return "\n".join(lines)
