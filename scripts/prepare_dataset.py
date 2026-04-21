#!/usr/bin/env python3
"""
Prepare dataset: Generate metadata CSV with class mapping and train/val/test splits.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --dataset_root path/to/dataset
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import DatasetMapper
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MCI dataset: map classes and create splits'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='Alzheimer_MRI_4_classes_dataset',
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_metadata.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        help='Disable stratified splitting'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MCI DATASET PREPARATION")
    print("="*60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output file: {args.output}")
    print(f"Train/Val/Test ratios: {args.train_ratio:.2f}/{args.val_ratio:.2f}/{1-args.train_ratio-args.val_ratio:.2f}")
    print(f"Random seed: {args.seed}")
    print(f"Stratified: {not args.no_stratify}")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_root):
        print(f"\n❌ ERROR: Dataset directory not found: {args.dataset_root}")
        print("Please ensure the dataset is in the correct location.")
        return 1
    
    # Create mapper
    mapper = DatasetMapper(args.dataset_root)
    
    # Scan dataset
    print("\n📂 Scanning dataset...")
    df = mapper.scan_dataset()
    
    if len(df) == 0:
        print("\n❌ ERROR: No images found in dataset!")
        return 1
    
    # Create splits
    print("\n✂️  Creating train/val/test splits...")
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    df = mapper.create_splits(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
        stratify=not args.no_stratify,
        random_seed=args.seed
    )
    
    # Save metadata
    print("\n💾 Saving metadata...")
    mapper.save_metadata(df, args.output)
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print("="*60)
    print(f"\nMetadata saved to: {args.output}")
    print(f"Total samples: {len(df)}")
    print("\nNext steps:")
    print("  1. Review the metadata CSV to verify class mapping")
    print("  2. Train models using: python scripts/train_baseline.py")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
