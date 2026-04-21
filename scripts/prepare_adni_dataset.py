#!/usr/bin/env python3
"""
Prepare ADNI dataset: Generate metadata CSV for 3D NIfTI volumes.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DatasetMapper
import yaml

def main():
    print("="*60)
    print("ADNI 3D DATASET PREPARATION")
    print("="*60)
    
    # Load data config
    config_path = Path('configs/data_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_root = config['paths']['dataset_root']
    output_csv = config['paths']['metadata_csv']
    
    print(f"Dataset root: {dataset_root}")
    print(f"Output CSV: {output_csv}")
    
    if not os.path.exists(dataset_root):
        print(f"❌ ERROR: Dataset root '{dataset_root}' not found!")
        return
    
    # Create mapper
    mapper = DatasetMapper(dataset_root)
    
    # Scan ADNI dataset (this will use the _scan_adni_dataset method)
    df = mapper.scan_dataset()
    
    if len(df) == 0:
        print("❌ ERROR: No samples found!")
        return
    
    # Create splits
    print("\nCreating train/val/test splits...")
    df = mapper.create_splits(
        df,
        train_ratio=config['splits']['train_ratio'],
        val_ratio=config['splits']['val_ratio'],
        test_ratio=config['splits']['test_ratio'],
        stratify=config['splits']['stratify'],
        random_seed=config['splits']['random_seed']
    )
    
    # Save
    mapper.save_metadata(df, output_csv)
    
    print("\n✅ Metadata saved to:", output_csv)
    print("="*60)

if __name__ == '__main__':
    main()
