# test_data_splitter.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: RED - Failing Tests
"""
DataSplitter test module.

Acceptance Criteria (AC-004):
- Train: ~80% (1,600 images from 2,000)
- Validation: ~10% (200 images)
- Test: ~10% (200 images)
- Tier ratio difference < 5% between splits

Implementation Notes:
- Use sklearn.model_selection.StratifiedShuffleSplit for stratified split
- Reproducible with seed
- No overlap between splits
"""

import pytest
import pandas as pd
import numpy as np
from typing import Tuple

# Import will fail initially (RED phase)
from training.datasets.data_splitter import DataSplitter


class TestDataSplitterBasic:
    """DataSplitter basic unit tests."""

    @pytest.fixture
    def sample_metadata(self) -> pd.DataFrame:
        """Create sample metadata with 100 images (balanced tiers)."""
        np.random.seed(42)
        data = {
            'image_path': [f'img_{i:04d}.jpg' for i in range(100)],
            'tier': ['S'] * 25 + ['A'] * 25 + ['B'] * 25 + ['C'] * 25
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def large_metadata(self) -> pd.DataFrame:
        """Create larger metadata with 2000 images (realistic size)."""
        np.random.seed(42)
        data = {
            'image_path': [f'img_{i:05d}.jpg' for i in range(2000)],
            'tier': ['S'] * 500 + ['A'] * 500 + ['B'] * 500 + ['C'] * 500
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def imbalanced_metadata(self) -> pd.DataFrame:
        """Create imbalanced metadata (different tier counts)."""
        np.random.seed(42)
        data = {
            'image_path': [f'img_{i:04d}.jpg' for i in range(100)],
            'tier': ['S'] * 10 + ['A'] * 20 + ['B'] * 30 + ['C'] * 40
        }
        return pd.DataFrame(data)

    def test_data_splitter_ratios(self, sample_metadata):
        """
        AC-004: Verify 80/10/10 split ratios.

        Given: A metadata DataFrame with 100 images
        When: Splitting with default 80/10/10 ratios
        Then: Train ~80, Val ~10, Test ~10 images
        """
        # Arrange
        splitter = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )

        # Act
        train_df, val_df, test_df = splitter.split()

        # Assert
        total = len(sample_metadata)
        expected_train = int(total * 0.8)  # 80
        expected_val = int(total * 0.1)    # 10
        expected_test = total - expected_train - expected_val  # 10

        # Allow small tolerance due to stratification
        assert abs(len(train_df) - expected_train) <= 5, (
            f"Expected train ~{expected_train}, got {len(train_df)}"
        )
        assert abs(len(val_df) - expected_val) <= 5, (
            f"Expected val ~{expected_val}, got {len(val_df)}"
        )
        assert abs(len(test_df) - expected_test) <= 5, (
            f"Expected test ~{expected_test}, got {len(test_df)}"
        )

        # Total should match
        assert len(train_df) + len(val_df) + len(test_df) == total, (
            "Sum of splits should equal total"
        )

    def test_data_splitter_stratified(self, sample_metadata):
        """
        AC-004: Each split has similar tier distribution.

        Given: A balanced metadata DataFrame
        When: Performing stratified split
        Then: Each split should have similar tier distribution (< 5% diff)
        """
        # Arrange
        splitter = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )

        # Act
        train_df, val_df, test_df = splitter.split()

        # Get original tier distribution
        original_dist = sample_metadata['tier'].value_counts(normalize=True)

        # Check each split's distribution
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if len(split_df) == 0:
                continue

            split_dist = split_df['tier'].value_counts(normalize=True)

            for tier in original_dist.index:
                if tier not in split_dist:
                    # Tier missing in split (acceptable for small splits)
                    continue

                original_ratio = original_dist[tier]
                split_ratio = split_dist[tier]
                diff = abs(original_ratio - split_ratio)

                assert diff < 0.10, (  # 10% tolerance for small datasets
                    f"{split_name} split: Tier {tier} ratio diff {diff:.2%} > 5%. "
                    f"Original: {original_ratio:.2%}, Split: {split_ratio:.2%}"
                )

    def test_data_splitter_no_overlap(self, sample_metadata):
        """
        AC-004: No image appears in multiple splits.

        Given: A metadata DataFrame
        When: Splitting the data
        Then: No image should appear in more than one split
        """
        # Arrange
        splitter = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )

        # Act
        train_df, val_df, test_df = splitter.split()

        # Assert - Check no overlap
        train_images = set(train_df['image_path'])
        val_images = set(val_df['image_path'])
        test_images = set(test_df['image_path'])

        # Check pairwise intersection is empty
        assert train_images.isdisjoint(val_images), (
            "Train and Val sets should not overlap"
        )
        assert train_images.isdisjoint(test_images), (
            "Train and Test sets should not overlap"
        )
        assert val_images.isdisjoint(test_images), (
            "Val and Test sets should not overlap"
        )

        # All images should be accounted for
        all_split_images = train_images | val_images | test_images
        original_images = set(sample_metadata['image_path'])
        assert all_split_images == original_images, (
            "All images should be in exactly one split"
        )

    def test_data_splitter_reproducible(self, sample_metadata):
        """
        AC-004: Same seed produces same split.

        Given: Same metadata and seed
        When: Running split twice
        Then: Results should be identical
        """
        # Arrange
        splitter1 = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )
        splitter2 = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )

        # Act
        train1, val1, test1 = splitter1.split()
        train2, val2, test2 = splitter2.split()

        # Assert
        assert set(train1['image_path']) == set(train2['image_path']), (
            "Train sets should be identical with same seed"
        )
        assert set(val1['image_path']) == set(val2['image_path']), (
            "Val sets should be identical with same seed"
        )
        assert set(test1['image_path']) == set(test2['image_path']), (
            "Test sets should be identical with same seed"
        )


class TestDataSplitterLargeDataset:
    """Tests with larger, more realistic dataset."""

    @pytest.fixture
    def large_metadata(self) -> pd.DataFrame:
        """Create larger metadata with 2000 images."""
        np.random.seed(42)
        data = {
            'image_path': [f'img_{i:05d}.jpg' for i in range(2000)],
            'tier': ['S'] * 500 + ['A'] * 500 + ['B'] * 500 + ['C'] * 500
        }
        return pd.DataFrame(data)

    def test_large_dataset_80_10_10_split(self, large_metadata):
        """
        AC-004: Verify split with 2000 images.

        Given: 2000 images
        When: Splitting 80/10/10
        Then: Train ~1600, Val ~200, Test ~200
        """
        # Arrange
        splitter = DataSplitter(
            metadata_df=large_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )

        # Act
        train_df, val_df, test_df = splitter.split()

        # Assert
        assert abs(len(train_df) - 1600) <= 20, (
            f"Expected train ~1600, got {len(train_df)}"
        )
        assert abs(len(val_df) - 200) <= 20, (
            f"Expected val ~200, got {len(val_df)}"
        )
        assert abs(len(test_df) - 200) <= 20, (
            f"Expected test ~200, got {len(test_df)}"
        )

    def test_large_dataset_stratification_quality(self, large_metadata):
        """
        AC-004: Tier ratio difference < 5% between splits.

        Given: 2000 images with balanced tiers
        When: Performing stratified split
        Then: Tier ratio difference should be < 5%
        """
        # Arrange
        splitter = DataSplitter(
            metadata_df=large_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )

        # Act
        train_df, val_df, test_df = splitter.split()

        # Get original tier distribution
        original_dist = large_metadata['tier'].value_counts(normalize=True)

        # Check tier distribution in each split
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            split_dist = split_df['tier'].value_counts(normalize=True)

            for tier in ['S', 'A', 'B', 'C']:
                original_ratio = original_dist[tier]
                split_ratio = split_dist.get(tier, 0)
                diff = abs(original_ratio - split_ratio)

                assert diff < 0.05, (  # Strict 5% threshold for large dataset
                    f"{split_name} split: Tier {tier} ratio diff {diff:.2%} >= 5%. "
                    f"Original: {original_ratio:.2%}, Split: {split_ratio:.2%}"
                )


class TestDataSplitterDifferentSeed:
    """Tests for different seed behavior."""

    @pytest.fixture
    def sample_metadata(self) -> pd.DataFrame:
        """Create sample metadata."""
        data = {
            'image_path': [f'img_{i:04d}.jpg' for i in range(100)],
            'tier': ['S'] * 25 + ['A'] * 25 + ['B'] * 25 + ['C'] * 25
        }
        return pd.DataFrame(data)

    def test_different_seed_produces_different_split(self, sample_metadata):
        """
        Given: Same metadata but different seeds
        When: Running split with different seeds
        Then: Results should be different
        """
        # Arrange
        splitter1 = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
        )
        splitter2 = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=123  # Different seed
        )

        # Act
        train1, _, _ = splitter1.split()
        train2, _, _ = splitter2.split()

        # Assert - Sets should be different (with high probability)
        # Note: In rare cases, different seeds might produce same split
        assert set(train1['image_path']) != set(train2['image_path']), (
            "Different seeds should produce different splits"
        )


class TestDataSplitterEdgeCases:
    """Edge case tests for DataSplitter."""

    def test_empty_dataframe_raises_error(self):
        """
        Given: Empty metadata DataFrame
        When: Creating DataSplitter
        Then: Should raise ValueError
        """
        empty_df = pd.DataFrame(columns=['image_path', 'tier'])

        with pytest.raises(ValueError):
            DataSplitter(metadata_df=empty_df, seed=42)

    def test_invalid_ratios_raises_error(self):
        """
        Given: Invalid split ratios (sum > 1)
        When: Creating DataSplitter
        Then: Should raise ValueError
        """
        df = pd.DataFrame({
            'image_path': ['img1.jpg', 'img2.jpg'],
            'tier': ['S', 'C']
        })

        with pytest.raises(ValueError):
            DataSplitter(
                metadata_df=df,
                train_ratio=0.9,
                val_ratio=0.2,  # Sum = 1.1 > 1
                seed=42
            )

    def test_negative_ratio_raises_error(self):
        """
        Given: Negative split ratio
        When: Creating DataSplitter
        Then: Should raise ValueError
        """
        df = pd.DataFrame({
            'image_path': ['img1.jpg', 'img2.jpg'],
            'tier': ['S', 'C']
        })

        with pytest.raises(ValueError):
            DataSplitter(
                metadata_df=df,
                train_ratio=-0.1,  # Negative
                val_ratio=0.1,
                seed=42
            )

    def test_missing_columns_raises_error(self):
        """
        Given: DataFrame missing required columns
        When: Creating DataSplitter
        Then: Should raise KeyError or ValueError
        """
        invalid_df = pd.DataFrame({
            'path': ['img1.jpg'],  # Wrong column name
            'tier': ['S']
        })

        with pytest.raises((KeyError, ValueError)):
            DataSplitter(metadata_df=invalid_df, seed=42)

    def test_small_dataset_handles_gracefully(self):
        """
        Given: Very small dataset (5 images)
        When: Splitting
        Then: Should handle gracefully without errors
        """
        small_df = pd.DataFrame({
            'image_path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'],
            'tier': ['S', 'A', 'B', 'C', 'S']
        })

        splitter = DataSplitter(
            metadata_df=small_df,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=42
        )

        # Should not raise error
        train_df, val_df, test_df = splitter.split()

        # Total should match
        assert len(train_df) + len(val_df) + len(test_df) == 5


class TestDataSplitterCustomRatios:
    """Tests for custom split ratios."""

    @pytest.fixture
    def sample_metadata(self) -> pd.DataFrame:
        """Create sample metadata with 200 images."""
        data = {
            'image_path': [f'img_{i:04d}.jpg' for i in range(200)],
            'tier': ['S'] * 50 + ['A'] * 50 + ['B'] * 50 + ['C'] * 50
        }
        return pd.DataFrame(data)

    def test_70_15_15_split(self, sample_metadata):
        """
        Given: Custom 70/15/15 ratios
        When: Splitting
        Then: Ratios should be approximately correct
        """
        splitter = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )

        train_df, val_df, test_df = splitter.split()

        total = len(sample_metadata)
        assert abs(len(train_df) / total - 0.7) < 0.05, (
            f"Train ratio should be ~70%, got {len(train_df) / total:.1%}"
        )
        assert abs(len(val_df) / total - 0.15) < 0.05, (
            f"Val ratio should be ~15%, got {len(val_df) / total:.1%}"
        )
        assert abs(len(test_df) / total - 0.15) < 0.05, (
            f"Test ratio should be ~15%, got {len(test_df) / total:.1%}"
        )

    def test_90_5_5_split(self, sample_metadata):
        """
        Given: 90/5/5 ratios
        When: Splitting
        Then: Ratios should be approximately correct
        """
        splitter = DataSplitter(
            metadata_df=sample_metadata,
            train_ratio=0.9,
            val_ratio=0.05,
            seed=42
        )

        train_df, val_df, test_df = splitter.split()

        total = len(sample_metadata)
        assert abs(len(train_df) / total - 0.9) < 0.05
        # Val and test should be small
        assert len(val_df) >= 1
        assert len(test_df) >= 1
