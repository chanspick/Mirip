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

    # Default image size for DINOv2
    DEFAULT_IMAGE_SIZE: int = 768

    # ImageNet normalization stats
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        image_dir: Union[str, Path],
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize PairwiseDataset.

        Args:
            metadata_df: DataFrame with columns ['image_path', 'tier']
            image_dir: Directory containing the images
            transform: Optional transform to apply to images

        Raises:
            ValueError: If metadata_df is empty
            ValueError: If only one tier exists
            KeyError: If required columns are missing
        """
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
        self._transform = transform or self._get_default_transform()

        # Generate all valid pairs
        self._pairs = self._generate_pairs()

    def _get_default_transform(self) -> Callable:
        """
        Get default image transform.

        Returns:
            Composed transform with resize, to_tensor, and normalize
        """
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
        Generate all valid pairs from different tiers.

        Returns:
            List of tuples (img1_path, img2_path, label)
            where label = 1 if tier(img1) > tier(img2), else -1

        Note:
            Generates both orderings: (A, B) and (B, A) for each pair
        """
        pairs = []

        # Group images by tier
        tier_groups = self._metadata_df.groupby('tier')['image_path'].apply(list).to_dict()

        # Get all tier combinations
        tiers = list(tier_groups.keys())

        for i, tier1 in enumerate(tiers):
            for tier2 in tiers[i + 1:]:
                # Get images from each tier
                images1 = tier_groups[tier1]
                images2 = tier_groups[tier2]

                # Calculate label based on tier values
                tier1_value = self.TIER_VALUES.get(tier1, 0)
                tier2_value = self.TIER_VALUES.get(tier2, 0)

                # Generate pairs in both directions
                for img1 in images1:
                    for img2 in images2:
                        # (tier1, tier2) pair
                        label1 = 1 if tier1_value > tier2_value else -1
                        pairs.append((img1, img2, label1))

                        # (tier2, tier1) pair (reversed)
                        label2 = 1 if tier2_value > tier1_value else -1
                        pairs.append((img2, img1, label2))

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
