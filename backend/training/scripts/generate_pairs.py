#!/usr/bin/env python
# generate_pairs.py
# SPEC-AI-001: DINOv2 Baseline AI Evaluation Model
"""
Pair Generation Script

CLI script to generate pairwise training data from image metadata.
Saves pairs to JSON/CSV for reproducibility and inspection.

Acceptance Criteria (AC-005):
- Generate pairs from metadata CSV
- Save pairs to JSON/CSV for reproducibility
- Support stratified train/val/test split

Usage:
    python -m training.scripts.generate_pairs \\
        --metadata metadata.csv \\
        --output pairs.json \\
        --format json \\
        --split-ratios 0.8 0.1 0.1 \\
        --seed 42

Example:
    # Generate pairs with default 80/10/10 split
    python -m training.scripts.generate_pairs \\
        --metadata data/metadata.csv \\
        --output data/pairs \\
        --format both

    # Generate pairs for train set only
    python -m training.scripts.generate_pairs \\
        --metadata data/train_metadata.csv \\
        --output data/train_pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.datasets.data_splitter import DataSplitter
from training.datasets.pairwise_dataset import PairwiseDataset


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate pairwise training data from image metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate pairs for entire dataset
  python -m training.scripts.generate_pairs --metadata metadata.csv --output pairs.json

  # Generate pairs with train/val/test split
  python -m training.scripts.generate_pairs --metadata metadata.csv --output pairs --split

  # Generate pairs in CSV format
  python -m training.scripts.generate_pairs --metadata metadata.csv --output pairs.csv --format csv
        """
    )

    parser.add_argument(
        "--metadata", "-m",
        type=Path,
        required=True,
        help="Path to metadata CSV file with columns [image_path, tier]"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output path for pairs (file or directory if --split)"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format (default: json)"
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="Split data into train/val/test sets"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Image directory (for validation only, not required)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """
    Load metadata from CSV file.

    Args:
        metadata_path: Path to metadata CSV file

    Returns:
        DataFrame with columns [image_path, tier]

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required_columns = {'image_path', 'tier'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    return df


def generate_pairs_from_metadata(
    metadata_df: pd.DataFrame,
    verbose: bool = False
) -> List[Tuple[str, str, int]]:
    """
    Generate pairs from metadata DataFrame.

    Args:
        metadata_df: DataFrame with columns [image_path, tier]
        verbose: Print progress information

    Returns:
        List of (img1_path, img2_path, label) tuples
    """
    # Use a dummy image_dir since we're not actually loading images
    # Just generating the pair list
    tier_values = {'S': 4, 'A': 3, 'B': 2, 'C': 1}
    pairs = []

    # Group images by tier
    tier_groups = metadata_df.groupby('tier')['image_path'].apply(list).to_dict()
    tiers = list(tier_groups.keys())

    if verbose:
        print(f"Tiers found: {tiers}")
        for tier, images in tier_groups.items():
            print(f"  {tier}: {len(images)} images")

    # Generate pairs for different tiers
    for i, tier1 in enumerate(tiers):
        for tier2 in tiers[i + 1:]:
            images1 = tier_groups[tier1]
            images2 = tier_groups[tier2]

            tier1_value = tier_values.get(tier1, 0)
            tier2_value = tier_values.get(tier2, 0)

            for img1 in images1:
                for img2 in images2:
                    # (tier1, tier2) pair
                    label1 = 1 if tier1_value > tier2_value else -1
                    pairs.append((img1, img2, label1))

                    # (tier2, tier1) pair (reversed)
                    label2 = 1 if tier2_value > tier1_value else -1
                    pairs.append((img2, img1, label2))

    if verbose:
        print(f"Generated {len(pairs)} pairs")

    return pairs


def save_pairs_json(
    pairs: List[Tuple[str, str, int]],
    output_path: Path
) -> None:
    """
    Save pairs to JSON file.

    Args:
        pairs: List of (img1_path, img2_path, label) tuples
        output_path: Output JSON file path
    """
    data = {
        "metadata": {
            "num_pairs": len(pairs),
            "format_version": "1.0"
        },
        "pairs": [
            {
                "img1": img1,
                "img2": img2,
                "label": label
            }
            for img1, img2, label in pairs
        ]
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_pairs_csv(
    pairs: List[Tuple[str, str, int]],
    output_path: Path
) -> None:
    """
    Save pairs to CSV file.

    Args:
        pairs: List of (img1_path, img2_path, label) tuples
        output_path: Output CSV file path
    """
    df = pd.DataFrame(pairs, columns=['img1', 'img2', 'label'])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()

    try:
        # Load metadata
        if args.verbose:
            print(f"Loading metadata from: {args.metadata}")

        metadata_df = load_metadata(args.metadata)

        if args.verbose:
            print(f"Loaded {len(metadata_df)} images")
            print(f"Tier distribution:")
            for tier, count in metadata_df['tier'].value_counts().items():
                print(f"  {tier}: {count}")

        if args.split:
            # Split into train/val/test
            if args.verbose:
                print(f"\nSplitting data ({args.train_ratio}/{args.val_ratio}/{1-args.train_ratio-args.val_ratio})...")

            splitter = DataSplitter(
                metadata_df=metadata_df,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed
            )

            train_df, val_df, test_df = splitter.split()

            if args.verbose:
                print(f"  Train: {len(train_df)} images")
                print(f"  Val: {len(val_df)} images")
                print(f"  Test: {len(test_df)} images")

            # Generate pairs for each split
            output_dir = args.output
            output_dir.mkdir(parents=True, exist_ok=True)

            splits = [
                ("train", train_df),
                ("val", val_df),
                ("test", test_df)
            ]

            for split_name, split_df in splits:
                if len(split_df) < 2:
                    if args.verbose:
                        print(f"\nSkipping {split_name} (not enough images)")
                    continue

                if args.verbose:
                    print(f"\nGenerating {split_name} pairs...")

                pairs = generate_pairs_from_metadata(split_df, verbose=args.verbose)

                # Save in requested format
                if args.format in ["json", "both"]:
                    json_path = output_dir / f"{split_name}_pairs.json"
                    save_pairs_json(pairs, json_path)
                    if args.verbose:
                        print(f"  Saved to: {json_path}")

                if args.format in ["csv", "both"]:
                    csv_path = output_dir / f"{split_name}_pairs.csv"
                    save_pairs_csv(pairs, csv_path)
                    if args.verbose:
                        print(f"  Saved to: {csv_path}")

            # Also save split metadata
            for split_name, split_df in splits:
                meta_path = output_dir / f"{split_name}_metadata.csv"
                split_df.to_csv(meta_path, index=False)
                if args.verbose:
                    print(f"  Saved metadata to: {meta_path}")

        else:
            # Generate pairs for entire dataset
            if args.verbose:
                print("\nGenerating pairs for entire dataset...")

            pairs = generate_pairs_from_metadata(metadata_df, verbose=args.verbose)

            # Determine output path
            output_path = args.output
            if output_path.suffix == '':
                # Add extension based on format
                if args.format == "json":
                    output_path = output_path.with_suffix('.json')
                elif args.format == "csv":
                    output_path = output_path.with_suffix('.csv')

            # Save in requested format
            if args.format in ["json", "both"]:
                json_path = output_path if output_path.suffix == '.json' else output_path.with_suffix('.json')
                save_pairs_json(pairs, json_path)
                print(f"Saved {len(pairs)} pairs to: {json_path}")

            if args.format in ["csv", "both"]:
                csv_path = output_path if output_path.suffix == '.csv' else output_path.with_suffix('.csv')
                save_pairs_csv(pairs, csv_path)
                print(f"Saved {len(pairs)} pairs to: {csv_path}")

        if args.verbose:
            print("\nDone!")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
