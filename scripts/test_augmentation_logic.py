import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import AugmentationPipeline

def test_3d_augmentation():
    print("Testing 3D Augmentation...")
    # Shape (C, D, H, W)
    dummy_3d = torch.randn(1, 32, 64, 64)
    aug = AugmentationPipeline(
        horizontal_flip=True,
        rotation_range=10.0,
        shift_3d_range=5.0,
        noise_std=0.01
    )
    
    augmented = aug(dummy_3d)
    
    print(f"  Original shape: {dummy_3d.shape}")
    print(f"  Augmented shape: {augmented.shape}")
    
    assert dummy_3d.shape == augmented.shape, "Shape mismatch in 3D!"
    assert not torch.equal(dummy_3d, augmented), "Augmentation did not change the tensor values!"
    print("  ✅ 3D Augmentation test passed!")

def test_2d_augmentation():
    print("\nTesting 2D Augmentation...")
    # Shape (C, H, W)
    dummy_2d = torch.randn(1, 64, 64)
    aug = AugmentationPipeline(
        horizontal_flip=True,
        rotation_range=10.0,
        noise_std=0.01
    )
    
    augmented = aug(dummy_2d)
    
    print(f"  Original shape: {dummy_2d.shape}")
    print(f"  Augmented shape: {augmented.shape}")
    
    assert dummy_2d.shape == augmented.shape, "Shape mismatch in 2D!"
    assert not torch.equal(dummy_2d, augmented), "Augmentation did not change the tensor values!"
    print("  ✅ 2D Augmentation test passed!")

if __name__ == "__main__":
    try:
        test_3d_augmentation()
        test_2d_augmentation()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
