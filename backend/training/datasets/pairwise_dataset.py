# pairwise_dataset.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: GREEN - Implementation
"""
Pairwise Dataset Module

Pairwise dataset for ranking model training.
Generates pairs from images of different tiers (S, A, B, C).

Acceptance Criteria (AC-005):
- Only pairs from different tiers
- Label = 1 if tier(img1) > tier(img2), else -1
- No same-tier pairs

Tier Order:
- S=4 > A=3 > B=2 > C=1 (higher is better)

Example:
    >>> metadata_df = pd.DataFrame({
    ...     'image_path': ['s1.jpg', 'c1.jpg'],
    ...     'tier': ['S', 'C']
    ... })
    >>> dataset = PairwiseDataset(metadata_df, '/path/to/images')
    >>> img1, img2, label = dataset[0]
    >>> print(label)  # 1 (S > C) or -1 (C < S)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairwiseDataset(Dataset):
    """
    Pairwise dataset for ranking model training.

    Generates image pairs from different tiers for pairwise ranking.
    Each pair consists of two images from different tiers with a label
    indicating which image has the higher tier (better quality).

    Attributes:
        TIER_VALUES: Mapping from tier label to numeric value
        DEFAULT_IMAGE_SIZE: Default image size for transforms

    Args:
        metadata_df: DataFrame with columns ['image_path', 'tier']
        image_dir: Directory containing the images
        transform: Optional transform to apply to images.
                  If None, applies default transform.

    Raises:
        ValueError: If metadata_df is empty
        ValueError: If only one tier exists (no valid pairs possible)
        KeyError: If required columns are missing

    Example:
        >>> dataset = PairwiseDataset(metadata_df, '/data/images')
        >>> img1, img2, label = dataset[0]
        >>> print(img1.shape)  # torch.Size([3, 768, 768])
    """

    # Tier value mapping: higher value = better quality
    TIER_VALUES: dict[str, int] = {
        'S': 4,
        'A': 3,
        'B': 2,
        'C': 1
    }

    # Default image size for DINOv2 (448 for faster training)
    DEFAULT_IMAGE_SIZE: int = 448

    # ImageNet normalization stats
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        max_pairs: Optional[int] = 20000,
        max_appearances_per_image: int = 10,
        seed: int = 42,
        is_train: bool = True,
    ) -> None:
        """
        Initialize PairwiseDataset.

        Args:
            metadata_df: DataFrame with columns ['image_path', 'tier']
            image_dir: Directory containing the images
            transform: Optional transform to apply to images
            max_pairs: Maximum number of pairs to generate (None for all pairs).
                      Default 20,000 to prevent overfitting.
            max_appearances_per_image: 각 이미지가 페어에 등장할 수 있는 최대 횟수.
                      Default 10 to prevent same image being seen too many times.
            seed: Random seed for reproducible pair sampling
            is_train: True면 augmentation 포함, False면 기본 transform만

        Raises:
            ValueError: If metadata_df is empty
            ValueError: If only one tier exists
            KeyError: If required columns are missing
        """
        self._max_pairs = max_pairs
        self._max_appearances = max_appearances_per_image
        self._seed = seed
        self._is_train = is_train
        # Validate required columns
        required_columns = {'image_path', 'tier'}
        if not required_columns.issubset(metadata_df.columns):
            missing = required_columns - set(metadata_df.columns)
            raise KeyError(f"Missing required columns: {missing}")

        # Validate non-empty DataFrame
        if len(metadata_df) == 0:
            raise ValueError("metadata_df cannot be empty")

        # Validate multiple tiers exist
        unique_tiers = metadata_df['tier'].unique()
        if len(unique_tiers) < 2:
            raise ValueError(
                f"At least 2 different tiers required, got {len(unique_tiers)}: {unique_tiers}"
            )

        self._metadata_df = metadata_df.copy()
        self._image_dir = Path(image_dir)
        self._transform = transform or self._get_default_transform(is_train)

        # Generate all valid pairs
        self._pairs = self._generate_pairs()

    def _get_default_transform(self, is_train: bool = True) -> Callable:
        """
        Get default image transform.

        Args:
            is_train: True면 augmentation 포함, False면 기본 transform만

        Returns:
            Composed transform
        """
        if is_train:
            # Train: augmentation 적용
            return transforms.Compose([
                transforms.Resize((self.DEFAULT_IMAGE_SIZE + 32, self.DEFAULT_IMAGE_SIZE + 32)),
                transforms.RandomCrop(self.DEFAULT_IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD
                )
            ])
        else:
            # Val/Test: augmentation 없음
            return transforms.Compose([
                transforms.Resize((self.DEFAULT_IMAGE_SIZE, self.DEFAULT_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.IMAGENET_MEAN,
                    std=self.IMAGENET_STD
                )
            ])

    def _generate_pairs(self) -> List[Tuple[str, str, int]]:
        """
        Generate valid pairs from different tiers with optional sampling.

        Returns:
            List of tuples (img1_path, img2_path, label)
            where label = 1 if tier(img1) > tier(img2), else -1

        Note:
            If max_pairs is set, samples uniformly across tier combinations.
            Generates both orderings: (A, B) and (B, A) for each pair.
        """
        random.seed(self._seed)

        # Group images by tier
        tier_groups = self._metadata_df.groupby('tier')['image_path'].apply(list).to_dict()
        tiers = list(tier_groups.keys())

        # Collect tier combination info
        tier_combos = []
        for i, tier1 in enumerate(tiers):
            for tier2 in tiers[i + 1:]:
                images1 = tier_groups[tier1]
                images2 = tier_groups[tier2]
                tier1_value = self.TIER_VALUES.get(tier1, 0)
                tier2_value = self.TIER_VALUES.get(tier2, 0)
                # 양방향 쌍 생성을 위한 총 가능 쌍 수
                total_possible = len(images1) * len(images2) * 2
                tier_combos.append({
                    'tier1': tier1, 'tier2': tier2,
                    'images1': images1, 'images2': images2,
                    'tier1_value': tier1_value, 'tier2_value': tier2_value,
                    'total_possible': total_possible
                })

        # 전체 가능한 쌍 수 계산
        total_all_pairs = sum(c['total_possible'] for c in tier_combos)

        # max_pairs가 없거나 전체 쌍보다 크면 모든 쌍 생성
        if self._max_pairs is None or total_all_pairs <= self._max_pairs:
            return self._generate_all_pairs(tier_combos)

        # 샘플링: 이미지당 등장 횟수 제한 적용
        pairs = []
        image_counts: dict[str, int] = {}  # 이미지별 등장 횟수 추적

        # 모든 가능한 페어를 생성하고 셔플
        all_candidate_pairs = []
        for combo in tier_combos:
            for img1 in combo['images1']:
                for img2 in combo['images2']:
                    label1 = 1 if combo['tier1_value'] > combo['tier2_value'] else -1
                    all_candidate_pairs.append((img1, img2, label1))
                    label2 = 1 if combo['tier2_value'] > combo['tier1_value'] else -1
                    all_candidate_pairs.append((img2, img1, label2))

        random.shuffle(all_candidate_pairs)

        # 등장 횟수 제한을 적용하면서 페어 선택
        for img1, img2, label in all_candidate_pairs:
            count1 = image_counts.get(img1, 0)
            count2 = image_counts.get(img2, 0)

            # 두 이미지 모두 limit 이하인 경우만 추가
            if count1 < self._max_appearances and count2 < self._max_appearances:
                pairs.append((img1, img2, label))
                image_counts[img1] = count1 + 1
                image_counts[img2] = count2 + 1

                if len(pairs) >= self._max_pairs:
                    break

        random.shuffle(pairs)
        return pairs

    def _generate_all_pairs(self, tier_combos: List[dict]) -> List[Tuple[str, str, int]]:
        """Generate all pairs from tier combinations (원래 방식)."""
        pairs = []
        for combo in tier_combos:
            for img1 in combo['images1']:
                for img2 in combo['images2']:
                    label1 = 1 if combo['tier1_value'] > combo['tier2_value'] else -1
                    pairs.append((img1, img2, label1))
                    label2 = 1 if combo['tier2_value'] > combo['tier1_value'] else -1
                    pairs.append((img2, img1, label2))
        return pairs

    def _sample_pairs_from_combo(
        self, combo: dict, n_samples: int
    ) -> List[Tuple[str, str, int]]:
        """Sample n_samples pairs from a single tier combination."""
        images1 = combo['images1']
        images2 = combo['images2']
        tier1_value = combo['tier1_value']
        tier2_value = combo['tier2_value']

        pairs = []
        max_possible = len(images1) * len(images2) * 2
        n_samples = min(n_samples, max_possible)

        # 각 방향별 샘플 수
        n_per_direction = n_samples // 2

        # (tier1 -> tier2) 방향 샘플링
        label1 = 1 if tier1_value > tier2_value else -1
        for _ in range(n_per_direction):
            img1 = random.choice(images1)
            img2 = random.choice(images2)
            pairs.append((img1, img2, label1))

        # (tier2 -> tier1) 방향 샘플링
        label2 = 1 if tier2_value > tier1_value else -1
        for _ in range(n_samples - n_per_direction):
            img1 = random.choice(images2)
            img2 = random.choice(images1)
            pairs.append((img1, img2, label2))

        return pairs

    @property
    def pairs(self) -> List[Tuple[str, str, int]]:
        """
        Get all generated pairs.

        Returns:
            List of (img1_path, img2_path, label) tuples
        """
        return self._pairs

    def __len__(self) -> int:
        """
        Get the number of pairs in the dataset.

        Returns:
            Number of image pairs
        """
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a pair of images and their label.

        Args:
            idx: Index of the pair

        Returns:
            Tuple of (img1_tensor, img2_tensor, label)
            - img1_tensor: First image tensor (C, H, W)
            - img2_tensor: Second image tensor (C, H, W)
            - label: 1 if tier(img1) > tier(img2), else -1

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self._pairs):
            raise IndexError(f"Index {idx} out of range [0, {len(self._pairs)})")

        img1_path, img2_path, label = self._pairs[idx]

        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        # Apply transforms
        img1_tensor = self._transform(img1)
        img2_tensor = self._transform(img2)

        return img1_tensor, img2_tensor, label

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from disk.

        Args:
            image_path: Relative path to the image

        Returns:
            PIL Image in RGB mode

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        full_path = self._image_dir / image_path

        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")

        return Image.open(full_path).convert('RGB')

    def get_tier_distribution(self) -> dict[str, int]:
        """
        Get the distribution of images per tier.

        Returns:
            Dictionary mapping tier to count
        """
        return self._metadata_df['tier'].value_counts().to_dict()

    def get_pair_count_by_tier_combination(self) -> dict[str, int]:
        """
        Get the number of pairs for each tier combination.

        Returns:
            Dictionary mapping tier combination to count
        """
        combination_counts: dict[str, int] = {}

        for img1_path, img2_path, _ in self._pairs:
            tier1 = self._metadata_df[
                self._metadata_df['image_path'] == img1_path
            ]['tier'].values[0]
            tier2 = self._metadata_df[
                self._metadata_df['image_path'] == img2_path
            ]['tier'].values[0]

            key = f"{tier1}-{tier2}"
            combination_counts[key] = combination_counts.get(key, 0) + 1

        return combination_counts

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"{self.__class__.__name__}("
            f"num_images={len(self._metadata_df)}, "
            f"num_pairs={len(self._pairs)}, "
            f"tiers={list(self._metadata_df['tier'].unique())})"
        )
