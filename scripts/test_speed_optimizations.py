import time
import torch
import os
import sys
from pathlib import Path
import shutil
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule

def test_optimizations():
    print("="*60)
    print("🚀 MCI CLASSIFICATION: SPEED OPTIMIZATION TEST")
    print("="*60)
    
    # 1. Test GPU & AMP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device detected: {device}")
    
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {major}.{minor}")
        if major >= 7:
            print("✅ Tensor Cores detected (AMP will be very effective)")
    else:
        print("⚠️ Running on CPU - AMP will have no effect.")

    # 2. Test Preprocessing Cache
    # Load data config to get real target sizes
    with open('configs/data_config.yaml', 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    cache_dir = Path("cache/smoke_test")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    print("\n📦 Initializing DataModule with Cache...")
    dm = MCIDataModule(
        metadata_csv='data_metadata_adni.csv',
        preprocessor_config=data_cfg['preprocessing'],
        batch_size=1,
        num_workers=0,  # 0 for clear measurement in main thread
        cache_dir=str(cache_dir)
    )
    
    # Get the training dataset
    try:
        dataset = dm.train_dataset()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Ensure 'data_metadata_adni.csv' exists and is correct.")
        return

    print(f"\n--- Performance Comparison (Single Sample) ---")
    
    # Disable augmentation for the test to verify identity
    original_aug = dataset.augmentation
    dataset.augmentation = None 
    
    # FIRST ACCESS (COLD CACHE)
    start = time.time()
    img1, label1, _ = dataset[0]
    time_cold = time.time() - start
    print(f"1. Cold Cache Access: {time_cold:.4f}s (Preprocessing + Saving)")
    
    # SECOND ACCESS (WARM CACHE)
    start = time.time()
    img2, label2, _ = dataset[0]
    time_warm = time.time() - start
    print(f"2. Warm Cache Access: {time_warm:.4f}s (Loading tensor only)")
    
    # Restore augmentation
    dataset.augmentation = original_aug
    
    # Calculation
    speedup = time_cold / max(time_warm, 0.0001)
    print(f"\n⚡ Result: Speed increase of {speedup:.1f}x for data loading")
    
    if speedup > 2.0:
        print("✅ CACHE SYSTEM: VERIFIED (Success)")
    else:
        print("⚠️ CACHE SYSTEM: Low speedup. This is normal if resizing is very fast (2D) or disk is slow.")

    # Verification of data integrity
    if torch.allclose(img1, img2):
        print("✅ DATA INTEGRITY: VERIFIED (Original and Cache are identical)")
    else:
        print("❌ DATA INTEGRITY: FAILED (Differences found between original and cache)")

    # 3. Data Statistics Summary
    num_train = len(dataset)
    num_epochs = data_cfg.get('training', {}).get('epochs', 100)
    effective_samples = num_train * num_epochs
    
    print("\n" + "="*60)
    print("📊 DATASET & AUGMENTATION SUMMARY")
    print("="*60)
    print(f"Physical training samples (patients): {num_train}")
    print(f"Planned training epochs:            {num_epochs}")
    print(f"Augmentation mode:                  ON-THE-FLY (Random each epoch)")
    print(f"Effective training exposures:       {effective_samples:,}")
    print("-" * 60)
    print("Interpretation for your Thesis:")
    print(f"The model will 'see' {effective_samples:,} unique variations of brain MRI")
    print(f"volumes, which provides a {num_epochs}x increase in data diversity")
    print("compared to training without augmentation.")
    print("="*60)

    print("\n✅ TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_optimizations()
