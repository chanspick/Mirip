# data_splitter.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: GREEN - Implementation
"""
Data Splitter Module

Stratified data splitter for train/val/test sets.

Acceptance Criteria (AC-004):
- 80% train, 10% validation, 10% test (default)
- Stratified by tier (S/A/B/C)
- Reproducible with seed
- Tier ratio difference < 5% between splits

Example:
    >>> splitter = DataSplitter(metadata_df, train_ratio=0.8, val_ratio=0.1, seed=42)
    >>> train_df, val_df, test_df = splitter.split()
    >>> print(len(train_df), len(val_df), len(test_df))  # ~1600, ~200, ~200
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


class DataSplitter:
    """
    Stratified data splitter for train/val/test sets.

    Uses sklearn's StratifiedShuffleSplit to ensure each split
    has similar tier distribution as the original dataset.

    Attributes:
        DEFAULT_TRAIN_RATIO: Default train set ratio (0.8)
        DEFAULT_VAL_RATIO: Default validation set ratio (0.1)
        DEFAULT_SEED: Default random seed (42)

    Args:
        metadata_df: DataFrame with columns ['image_path', 'tier']
        train_ratio: Proportion for training set (default: 0.8)
        val_ratio: Proportion for validation set (default: 0.1)
        seed: Random seed for reproducibility (default: 42)

    Raises:
        ValueError: If metadata_df is empty
        ValueError: If ratios are invalid (negative or sum > 1)
        KeyError: If required columns are missing

    Example:
        >>> splitter = DataSplitter(metadata_df, seed=42)
        >>> train_df, val_df, test_df = splitter.split()
    """

    # Default configuration
    DEFAULT_TRAIN_RATIO: float = 0.8
    DEFAULT_VAL_RATIO: float = 0.1
    DEFAULT_SEED: int = 42

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        val_ratio: float = DEFAULT_VAL_RATIO,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Initialize DataSplitter.

        Args:
            metadata_df: DataFrame with columns ['image_path', 'tier']
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            seed: Random seed for reproducibility

        Raises:
            ValueError: If metadata_df is empty
            ValueError: If ratios are invalid
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

        # Validate ratios
        if train_ratio < 0 or val_ratio < 0:
            raise ValueError("Ratios cannot be negative")

        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError(
                f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) = "
                f"{train_ratio + val_ratio} > 1.0"
            )

        self._metadata_df = metadata_df.copy()
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._seed = seed

        # Cache for split results
        self._split_cache: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None

    @property
    def train_ratio(self) -> float:
        """Get the train ratio."""
        return self._train_ratio

    @property
    def val_ratio(self) -> float:
        """Get the validation ratio."""
        return self._val_ratio

    @property
    def test_ratio(self) -> float:
        """Get the test ratio."""
        return self._test_ratio

    @property
    def seed(self) -> int:
        """Get the random seed."""
        return self._seed

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into train, validation, and test sets.

        Uses stratified sampling to ensure each split has similar
        tier distribution as the original dataset.

        Returns:
            Tuple of (train_df, val_df, test_df)

        Note:
            Results are cached after first call. Subsequent calls
            return the same split.
        """
        if self._split_cache is not None:
            return self._split_cache

        # Get stratification labels
        labels = self._metadata_df['tier'].values
        indices = np.arange(len(self._metadata_df))

        # Handle very small datasets
        n_samples = len(self._metadata_df)
        if n_samples < 5:
            # For very small datasets, do simple split
            return self._simple_split()

        # First split: separate test set
        train_val_ratio = self._train_ratio + self._val_ratio
        test_ratio_relative = self._test_ratio

        if test_ratio_relative > 0:
            try:
                splitter1 = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=test_ratio_relative,
                    random_state=self._seed
                )
                train_val_idx, test_idx = next(splitter1.split(indices, labels))
            except ValueError:
                # Fallback for cases where stratification fails
                return self._simple_split()
        else:
            train_val_idx = indices
            test_idx = np.array([], dtype=int)

        # Second split: separate train and val from train_val
        if self._val_ratio > 0 and len(train_val_idx) > 1:
            # Calculate relative val ratio within train_val
            val_ratio_relative = self._val_ratio / train_val_ratio

            train_val_labels = labels[train_val_idx]
            train_val_indices = np.arange(len(train_val_idx))

            try:
                splitter2 = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=val_ratio_relative,
                    random_state=self._seed
                )
                train_rel_idx, val_rel_idx = next(splitter2.split(train_val_indices, train_val_labels))

                train_idx = train_val_idx[train_rel_idx]
                val_idx = train_val_idx[val_rel_idx]
            except ValueError:
                # Fallback: simple split if stratification fails
                split_point = int(len(train_val_idx) * (1 - val_ratio_relative))
                np.random.seed(self._seed)
                np.random.shuffle(train_val_idx)
                train_idx = train_val_idx[:split_point]
                val_idx = train_val_idx[split_point:]
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=int)

        # Create DataFrames
        train_df = self._metadata_df.iloc[train_idx].reset_index(drop=True)
        val_df = self._metadata_df.iloc[val_idx].reset_index(drop=True)
        test_df = self._metadata_df.iloc[test_idx].reset_index(drop=True)

        # Cache results
        self._split_cache = (train_df, val_df, test_df)

        return train_df, val_df, test_df

    def _simple_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Simple split for very small datasets.

        Falls back to random split without stratification
        when dataset is too small for proper stratification.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        np.random.seed(self._seed)
        indices = np.arange(len(self._metadata_df))
        np.random.shuffle(indices)

        n = len(indices)
        train_end = int(n * self._train_ratio)
        val_end = int(n * (self._train_ratio + self._val_ratio))

        # Ensure at least 1 sample in each split if possible
        train_end = max(train_end, 1) if n >= 3 else train_end
        val_end = max(val_end, train_end + 1) if n >= 3 else val_end

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        train_df = self._metadata_df.iloc[train_idx].reset_index(drop=True)
        val_df = self._metadata_df.iloc[val_idx].reset_index(drop=True)
        test_df = self._metadata_df.iloc[test_idx].reset_index(drop=True)

        self._split_cache = (train_df, val_df, test_df)

        return train_df, val_df, test_df

    def get_split_statistics(self) -> dict:
        """
        Get statistics about the split.

        Returns:
            Dictionary with split sizes and tier distributions
        """
        train_df, val_df, test_df = self.split()

        stats = {
            'total_samples': len(self._metadata_df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_ratio_actual': len(train_df) / len(self._metadata_df),
            'val_ratio_actual': len(val_df) / len(self._metadata_df),
            'test_ratio_actual': len(test_df) / len(self._metadata_df),
            'original_tier_distribution': self._metadata_df['tier'].value_counts(normalize=True).to_dict(),
            'train_tier_distribution': train_df['tier'].value_counts(normalize=True).to_dict() if len(train_df) > 0 else {},
            'val_tier_distribution': val_df['tier'].value_counts(normalize=True).to_dict() if len(val_df) > 0 else {},
            'test_tier_distribution': test_df['tier'].value_counts(normalize=True).to_dict() if len(test_df) > 0 else {},
        }

        return stats

    def __repr__(self) -> str:
        """String representation of the DataSplitter."""
        return (
            f"{self.__class__.__name__}("
            f"n_samples={len(self._metadata_df)}, "
            f"train_ratio={self._train_ratio}, "
            f"val_ratio={self._val_ratio}, "
            f"test_ratio={self._test_ratio:.2f}, "
            f"seed={self._seed})"
        )
