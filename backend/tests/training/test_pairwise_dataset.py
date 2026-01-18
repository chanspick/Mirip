# test_pairwise_dataset.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
# TDD Phase: RED - Failing Tests
"""
PairwiseDataset test module.

Acceptance Criteria (AC-005):
- Only pairs from different tiers
- Label = 1 if tier(A) > tier(B), else -1
- No same-tier pairs

Implementation Notes:
- Tier order: S > A > B > C (numeric: S=4, A=3, B=2, C=1)
- Image transforms: Resize(768), ToTensor(), Normalize(imagenet_stats)
- Mock data for testing (don't need real images)
"""

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import tempfile
import os

# Import will fail initially (RED phase)
from training.datasets.pairwise_dataset import PairwiseDataset


class TestPairwiseDatasetBasic:
    """PairwiseDataset basic unit tests."""

    @pytest.fixture
    def sample_metadata(self) -> pd.DataFrame:
        """Create sample metadata DataFrame with tier labels."""
        data = {
            'image_path': [
                'img_001.jpg', 'img_002.jpg', 'img_003.jpg', 'img_004.jpg',
                'img_005.jpg', 'img_006.jpg', 'img_007.jpg', 'img_008.jpg'
            ],
            'tier': ['S', 'S', 'A', 'A', 'B', 'B', 'C', 'C']
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_image_dir(self) -> str:
        """Create temporary directory with mock images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock images (8 images: 2 S, 2 A, 2 B, 2 C)
            for i in range(1, 9):
                img_path = Path(tmpdir) / f'img_{i:03d}.jpg'
                # Create a simple 768x768 RGB image
                img = Image.new('RGB', (768, 768), color=(i * 30, i * 20, i * 10))
                img.save(img_path)
            yield tmpdir

    @pytest.fixture
    def dataset(self, sample_metadata, temp_image_dir) -> PairwiseDataset:
        """Create PairwiseDataset instance."""
        return PairwiseDataset(
            metadata_df=sample_metadata,
            image_dir=temp_image_dir,
            transform=None
        )

    def test_pairwise_dataset_length(self, dataset):
        """
        AC-005: Verify dataset length.

        Given: A metadata DataFrame with 8 images (2 per tier)
        When: Creating PairwiseDataset
        Then: Length should match number of valid pairs (different tiers only)

        Note: With 2 S, 2 A, 2 B, 2 C images:
        - S-A pairs: 2*2 = 4
        - S-B pairs: 2*2 = 4
        - S-C pairs: 2*2 = 4
        - A-B pairs: 2*2 = 4
        - A-C pairs: 2*2 = 4
        - B-C pairs: 2*2 = 4
        Total: 24 pairs
        """
        # Each pair can be in both orders (A,B) and (B,A)
        # So total pairs = 24 * 2 = 48 if we include both orders
        # Or 24 if we only generate one direction
        expected_min_pairs = 24  # Minimum: one direction only

        assert len(dataset) >= expected_min_pairs, (
            f"Expected at least {expected_min_pairs} pairs, got {len(dataset)}"
        )
        assert len(dataset) > 0, "Dataset should not be empty"

    def test_pairwise_dataset_getitem(self, dataset):
        """
        AC-005: Verify item retrieval returns (img1, img2, label).

        Given: A PairwiseDataset with valid pairs
        When: Accessing an item via __getitem__
        Then: Should return tuple of (img1_tensor, img2_tensor, label)
        """
        # Act
        item = dataset[0]

        # Assert - Should return tuple of 3 elements
        assert isinstance(item, tuple), (
            f"Expected tuple, got {type(item)}"
        )
        assert len(item) == 3, (
            f"Expected 3 elements (img1, img2, label), got {len(item)}"
        )

        img1, img2, label = item

        # Check image tensors
        assert isinstance(img1, torch.Tensor), (
            f"img1 should be torch.Tensor, got {type(img1)}"
        )
        assert isinstance(img2, torch.Tensor), (
            f"img2 should be torch.Tensor, got {type(img2)}"
        )

        # Check label
        assert isinstance(label, (int, torch.Tensor)), (
            f"label should be int or Tensor, got {type(label)}"
        )

    def test_pairwise_dataset_label_values(self, dataset):
        """
        AC-005: Labels must be 1 or -1.

        Given: A PairwiseDataset with valid pairs
        When: Checking all labels
        Then: All labels should be either 1 or -1
        """
        # Check multiple items
        num_samples = min(10, len(dataset))

        for i in range(num_samples):
            _, _, label = dataset[i]

            # Convert to int if tensor
            if isinstance(label, torch.Tensor):
                label_val = label.item()
            else:
                label_val = label

            assert label_val in [1, -1], (
                f"Label at index {i} should be 1 or -1, got {label_val}"
            )

    def test_pairwise_dataset_different_tiers(self, sample_metadata, temp_image_dir):
        """
        AC-005: Pairs must be from different tiers.

        Given: A PairwiseDataset
        When: Checking all generated pairs
        Then: No pair should have images from the same tier
        """
        dataset = PairwiseDataset(
            metadata_df=sample_metadata,
            image_dir=temp_image_dir,
            transform=None
        )

        # Get all pairs
        pairs = dataset.pairs

        for idx, (img1_path, img2_path, _) in enumerate(pairs):
            # Get tiers from metadata
            tier1 = sample_metadata[
                sample_metadata['image_path'] == img1_path
            ]['tier'].values[0]
            tier2 = sample_metadata[
                sample_metadata['image_path'] == img2_path
            ]['tier'].values[0]

            assert tier1 != tier2, (
                f"Pair {idx}: Images from same tier ({tier1}). "
                f"Paths: {img1_path}, {img2_path}"
            )

    def test_pairwise_dataset_transforms(self, sample_metadata, temp_image_dir):
        """
        AC-005: Verify image transforms applied.

        Given: A PairwiseDataset with transforms
        When: Accessing an item
        Then: Images should be properly transformed (768x768, normalized)
        """
        from torchvision import transforms

        # Standard transforms for DINOv2
        transform = transforms.Compose([
            transforms.Resize((768, 768)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

        dataset = PairwiseDataset(
            metadata_df=sample_metadata,
            image_dir=temp_image_dir,
            transform=transform
        )

        # Get an item
        img1, img2, _ = dataset[0]

        # Check shape: (3, 768, 768)
        assert img1.shape == (3, 768, 768), (
            f"Expected img1 shape (3, 768, 768), got {img1.shape}"
        )
        assert img2.shape == (3, 768, 768), (
            f"Expected img2 shape (3, 768, 768), got {img2.shape}"
        )

        # Check dtype
        assert img1.dtype == torch.float32, (
            f"Expected dtype float32, got {img1.dtype}"
        )


class TestPairwiseDatasetLabelLogic:
    """Tests for label generation logic."""

    @pytest.fixture
    def metadata_with_known_tiers(self) -> pd.DataFrame:
        """Create metadata with specific tier assignments."""
        data = {
            'image_path': ['s1.jpg', 'a1.jpg', 'b1.jpg', 'c1.jpg'],
            'tier': ['S', 'A', 'B', 'C']
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def temp_dir_for_tiers(self) -> str:
        """Create temp directory with images for tier testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ['s1.jpg', 'a1.jpg', 'b1.jpg', 'c1.jpg']:
                img_path = Path(tmpdir) / name
                img = Image.new('RGB', (768, 768), color='white')
                img.save(img_path)
            yield tmpdir

    def test_label_s_vs_c_is_positive(self, metadata_with_known_tiers, temp_dir_for_tiers):
        """
        AC-005: Label = 1 if tier(A) > tier(B).

        Given: S-tier image paired with C-tier image
        When: (S, C) pair is created
        Then: Label should be 1 (S > C)
        """
        dataset = PairwiseDataset(
            metadata_df=metadata_with_known_tiers,
            image_dir=temp_dir_for_tiers,
            transform=None
        )

        # Find the S vs C pair
        for img1_path, img2_path, label in dataset.pairs:
            tier1 = metadata_with_known_tiers[
                metadata_with_known_tiers['image_path'] == img1_path
            ]['tier'].values[0]
            tier2 = metadata_with_known_tiers[
                metadata_with_known_tiers['image_path'] == img2_path
            ]['tier'].values[0]

            if tier1 == 'S' and tier2 == 'C':
                assert label == 1, (
                    f"S vs C should have label 1, got {label}"
                )
            elif tier1 == 'C' and tier2 == 'S':
                assert label == -1, (
                    f"C vs S should have label -1, got {label}"
                )

    def test_label_c_vs_s_is_negative(self, metadata_with_known_tiers, temp_dir_for_tiers):
        """
        AC-005: Label = -1 if tier(A) < tier(B).

        Given: C-tier image paired with S-tier image
        When: (C, S) pair is created
        Then: Label should be -1 (C < S)
        """
        dataset = PairwiseDataset(
            metadata_df=metadata_with_known_tiers,
            image_dir=temp_dir_for_tiers,
            transform=None
        )

        # Find the C vs S pair
        found_c_vs_s = False
        for img1_path, img2_path, label in dataset.pairs:
            tier1 = metadata_with_known_tiers[
                metadata_with_known_tiers['image_path'] == img1_path
            ]['tier'].values[0]
            tier2 = metadata_with_known_tiers[
                metadata_with_known_tiers['image_path'] == img2_path
            ]['tier'].values[0]

            if tier1 == 'C' and tier2 == 'S':
                found_c_vs_s = True
                assert label == -1, (
                    f"C vs S should have label -1, got {label}"
                )

        # At least one C vs S pair should exist if we generate both directions
        # If not found, the dataset might only generate one direction

    def test_tier_numeric_mapping(self, metadata_with_known_tiers, temp_dir_for_tiers):
        """
        Test that tier order is correct: S=4 > A=3 > B=2 > C=1.

        Given: Pairs from different tiers
        When: Labels are generated
        Then: Labels should follow tier hierarchy
        """
        dataset = PairwiseDataset(
            metadata_df=metadata_with_known_tiers,
            image_dir=temp_dir_for_tiers,
            transform=None
        )

        tier_to_value = {'S': 4, 'A': 3, 'B': 2, 'C': 1}

        for img1_path, img2_path, label in dataset.pairs:
            tier1 = metadata_with_known_tiers[
                metadata_with_known_tiers['image_path'] == img1_path
            ]['tier'].values[0]
            tier2 = metadata_with_known_tiers[
                metadata_with_known_tiers['image_path'] == img2_path
            ]['tier'].values[0]

            expected_label = 1 if tier_to_value[tier1] > tier_to_value[tier2] else -1

            assert label == expected_label, (
                f"Pair ({tier1}, {tier2}): expected label {expected_label}, got {label}"
            )


class TestPairwiseDatasetEdgeCases:
    """Edge case tests for PairwiseDataset."""

    def test_empty_dataframe_raises_error(self):
        """
        Given: Empty metadata DataFrame
        When: Creating PairwiseDataset
        Then: Should raise ValueError
        """
        empty_df = pd.DataFrame(columns=['image_path', 'tier'])

        with pytest.raises(ValueError):
            PairwiseDataset(
                metadata_df=empty_df,
                image_dir='/tmp',
                transform=None
            )

    def test_single_tier_raises_error(self):
        """
        Given: Metadata with only one tier
        When: Creating PairwiseDataset
        Then: Should raise ValueError (no valid pairs possible)
        """
        single_tier_df = pd.DataFrame({
            'image_path': ['img1.jpg', 'img2.jpg'],
            'tier': ['S', 'S']
        })

        with pytest.raises(ValueError):
            PairwiseDataset(
                metadata_df=single_tier_df,
                image_dir='/tmp',
                transform=None
            )

    def test_missing_columns_raises_error(self):
        """
        Given: DataFrame missing required columns
        When: Creating PairwiseDataset
        Then: Should raise KeyError or ValueError
        """
        invalid_df = pd.DataFrame({
            'path': ['img1.jpg'],  # Wrong column name
            'tier': ['S']
        })

        with pytest.raises((KeyError, ValueError)):
            PairwiseDataset(
                metadata_df=invalid_df,
                image_dir='/tmp',
                transform=None
            )


class TestPairwiseDatasetWithTransforms:
    """Tests for PairwiseDataset with various transforms."""

    @pytest.fixture
    def simple_metadata(self) -> pd.DataFrame:
        """Create minimal metadata for transform testing."""
        return pd.DataFrame({
            'image_path': ['img1.jpg', 'img2.jpg'],
            'tier': ['S', 'C']
        })

    @pytest.fixture
    def simple_image_dir(self) -> str:
        """Create temp directory with 2 images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ['img1.jpg', 'img2.jpg']:
                img = Image.new('RGB', (768, 768), color='blue')
                img.save(Path(tmpdir) / name)
            yield tmpdir

    def test_default_transform_applied(self, simple_metadata, simple_image_dir):
        """
        Given: PairwiseDataset without explicit transform
        When: Accessing items
        Then: Default transform should be applied (if configured)
        """
        dataset = PairwiseDataset(
            metadata_df=simple_metadata,
            image_dir=simple_image_dir,
            transform=None  # Will use default if implemented
        )

        img1, img2, _ = dataset[0]

        # Images should be tensors
        assert isinstance(img1, torch.Tensor)
        assert isinstance(img2, torch.Tensor)

    def test_custom_transform_applied(self, simple_metadata, simple_image_dir):
        """
        Given: PairwiseDataset with custom transform
        When: Accessing items
        Then: Custom transform should be applied
        """
        from torchvision import transforms

        custom_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Different size
            transforms.ToTensor()
        ])

        dataset = PairwiseDataset(
            metadata_df=simple_metadata,
            image_dir=simple_image_dir,
            transform=custom_transform
        )

        img1, img2, _ = dataset[0]

        # Check custom size
        assert img1.shape == (3, 224, 224), (
            f"Expected shape (3, 224, 224), got {img1.shape}"
        )
