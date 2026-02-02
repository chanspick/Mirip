# anchor_manager.py
# 앵커 이미지 feature 관리 모듈
"""
앵커 Feature Manager

각 티어(S/A/B/C)의 대표 이미지 feature를 미리 추출하여 저장하고,
추론 시 빠르게 로드하여 사용합니다.

Usage:
    # 앵커 feature 생성 및 저장
    manager = AnchorManager(model, device='cuda')
    manager.build_anchors(metadata_df, image_dir, n_per_tier=10)
    manager.save('anchors.pt')

    # 앵커 feature 로드
    manager = AnchorManager.load('anchors.pt', model, device='cuda')
    anchor_features = manager.get_tier_features('S')
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms


class AnchorManager:
    """
    앵커 이미지 feature 관리자

    각 티어별 대표 이미지의 feature를 미리 추출하여 저장합니다.
    추론 시에는 저장된 feature만 로드하여 빠르게 비교합니다.
    """

    TIERS = ['S', 'A', 'B', 'C']
    DEFAULT_IMAGE_SIZE = 448
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model: nn.Module,
        device: Union[str, torch.device] = 'cpu',
    ) -> None:
        """
        Args:
            model: PairwiseRankingModel 인스턴스
            device: 연산 디바이스
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # 티어별 앵커 feature 저장소
        self._anchor_features: Dict[str, torch.Tensor] = {}
        self._anchor_paths: Dict[str, List[str]] = {}

        # 이미지 전처리
        self._transform = transforms.Compose([
            transforms.Resize((self.DEFAULT_IMAGE_SIZE, self.DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

    def build_anchors(
        self,
        metadata_df: pd.DataFrame,
        image_dir: Union[str, Path],
        n_per_tier: int = 10,
        selection_method: str = 'random',
        seed: int = 42,
    ) -> None:
        """
        각 티어에서 앵커 이미지를 선정하고 feature를 추출합니다.

        Args:
            metadata_df: image_path, tier 컬럼을 포함한 DataFrame
            image_dir: 이미지 디렉토리
            n_per_tier: 티어당 앵커 이미지 수
            selection_method: 'random' 또는 'centroid' (향후 확장)
            seed: 랜덤 시드
        """
        import random
        random.seed(seed)

        image_dir = Path(image_dir)

        for tier in self.TIERS:
            tier_df = metadata_df[metadata_df['tier'] == tier]

            if len(tier_df) == 0:
                print(f"Warning: No images for tier {tier}")
                continue

            # 앵커 이미지 선정
            n_select = min(n_per_tier, len(tier_df))
            selected = tier_df.sample(n=n_select, random_state=seed)

            # Feature 추출
            features_list = []
            paths_list = []

            for _, row in selected.iterrows():
                img_path = image_dir / row['image_path']
                if not img_path.exists():
                    print(f"Warning: Image not found: {img_path}")
                    continue

                # 이미지 로드 및 전처리
                img = Image.open(img_path).convert('RGB')
                img_tensor = self._transform(img).unsqueeze(0).to(self.device)

                # Feature 추출 (feature_extractor + projector)
                with torch.no_grad():
                    feat = self.model.feature_extractor(img_tensor)
                    feat = self.model.projector(feat)

                features_list.append(feat.cpu())
                paths_list.append(str(row['image_path']))

            if features_list:
                self._anchor_features[tier] = torch.cat(features_list, dim=0)
                self._anchor_paths[tier] = paths_list
                print(f"Tier {tier}: {len(features_list)} anchors extracted")

    def get_tier_features(self, tier: str) -> Optional[torch.Tensor]:
        """특정 티어의 앵커 feature 반환"""
        return self._anchor_features.get(tier)

    def get_all_features(self) -> Tuple[torch.Tensor, List[str]]:
        """
        모든 앵커 feature와 해당 티어 라벨 반환

        Returns:
            (features, tier_labels) 튜플
            - features: (N, feature_dim) 텐서
            - tier_labels: 각 feature의 티어 라벨 리스트
        """
        all_features = []
        all_labels = []

        for tier in self.TIERS:
            if tier in self._anchor_features:
                feats = self._anchor_features[tier]
                all_features.append(feats)
                all_labels.extend([tier] * feats.shape[0])

        if not all_features:
            raise ValueError("No anchor features available")

        return torch.cat(all_features, dim=0), all_labels

    def save(self, path: Union[str, Path]) -> None:
        """앵커 데이터 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'anchor_features': self._anchor_features,
            'anchor_paths': self._anchor_paths,
        }, path)
        print(f"Anchors saved to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        model: nn.Module,
        device: Union[str, torch.device] = 'cpu',
    ) -> 'AnchorManager':
        """저장된 앵커 데이터 로드"""
        manager = cls(model=model, device=device)

        data = torch.load(path, map_location='cpu')
        manager._anchor_features = data['anchor_features']
        manager._anchor_paths = data['anchor_paths']

        print(f"Anchors loaded from {path}")
        for tier, feats in manager._anchor_features.items():
            print(f"  Tier {tier}: {feats.shape[0]} anchors")

        return manager

    def __repr__(self) -> str:
        counts = {t: len(self._anchor_paths.get(t, [])) for t in self.TIERS}
        return f"AnchorManager({counts})"
