import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import AugmentationPipeline, get_augmentation


def test_3d_augmentation():
    print("Testing 3D Augmentation...")
    dummy_3d = torch.randn(1, 32, 64, 64)
    aug = AugmentationPipeline(
        horizontal_flip=True,
        rotation_range=10.0,
        shift_3d_range=5.0,
        noise_std=0.01,
    )

    augmented = aug(dummy_3d)

    print(f"  Original shape: {dummy_3d.shape}")
    print(f"  Augmented shape: {augmented.shape}")

    assert dummy_3d.shape == augmented.shape, "Shape mismatch in 3D!"
    assert not torch.equal(dummy_3d, augmented), "Augmentation did not change the tensor values!"
    print("  3D Augmentation test passed!")


def test_2d_augmentation():
    print("\nTesting 2D Augmentation...")
    dummy_2d = torch.randn(1, 64, 64)
    aug = AugmentationPipeline(
        horizontal_flip=True,
        rotation_range=10.0,
        noise_std=0.01,
    )

    augmented = aug(dummy_2d)

    print(f"  Original shape: {dummy_2d.shape}")
    print(f"  Augmented shape: {augmented.shape}")

    assert dummy_2d.shape == augmented.shape, "Shape mismatch in 2D!"
    assert not torch.equal(dummy_2d, augmented), "Augmentation did not change the tensor values!"
    print("  2D Augmentation test passed!")


def test_brightness_changes_values():
    print("\nTesting brightness...")
    tensor = torch.zeros(1, 8, 8, 8)
    aug = AugmentationPipeline(
        horizontal_flip=False,
        rotation_range=0.0,
        shift_3d_range=0.0,
        random_brightness=0.1,
        random_contrast=0.0,
        noise_std=0.0,
        elastic_deformation=False,
    )

    with mock.patch("torch.rand", return_value=torch.tensor([1.0])):
        result = aug._random_brightness_contrast(tensor.clone())

    assert torch.allclose(result, torch.full_like(tensor, 0.1)), "Brightness delta not applied"
    print("  Brightness test passed!")


def test_contrast_scales_around_mean():
    print("\nTesting contrast...")
    tensor = torch.tensor([[[[0.0, 2.0], [0.0, 2.0]], [[0.0, 2.0], [0.0, 2.0]]]], dtype=torch.float32)
    aug = AugmentationPipeline(
        random_brightness=0.0,
        random_contrast=0.2,
    )

    with mock.patch("torch.rand", return_value=torch.tensor([1.0])):
        result = aug._random_brightness_contrast(tensor.clone())

    expected = (tensor - tensor.mean()) * 1.2 + tensor.mean()
    assert torch.allclose(result, expected), "Contrast factor not applied"
    print("  Contrast test passed!")


def test_flip_probability_zero_and_one():
    print("\nTesting flip_probability...")
    tensor = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    aug_zero = AugmentationPipeline(
        horizontal_flip=True,
        flip_probability=0.0,
        rotation_range=0.0,
        shift_3d_range=0.0,
        random_brightness=0.0,
        random_contrast=0.0,
        noise_std=0.0,
        elastic_deformation=False,
    )
    aug_one = AugmentationPipeline(
        horizontal_flip=True,
        flip_probability=1.0,
        rotation_range=0.0,
        shift_3d_range=0.0,
        random_brightness=0.0,
        random_contrast=0.0,
        noise_std=0.0,
        elastic_deformation=False,
    )

    assert torch.equal(aug_zero(tensor.clone()), tensor), "flip_probability=0 should not flip"
    assert torch.equal(
        aug_one(tensor.clone()),
        torch.flip(tensor, dims=[-1]),
    ), "flip_probability=1 should always flip"
    print("  Flip probability test passed!")


def test_elastic_preserves_shape():
    print("\nTesting elastic deformation...")
    try:
        import torchio  # noqa: F401
    except ImportError:
        print("  Skipping elastic test (TorchIO not installed)")
        return

    tensor = torch.randn(1, 16, 32, 32)
    aug = AugmentationPipeline(
        horizontal_flip=False,
        rotation_range=0.0,
        shift_3d_range=0.0,
        random_brightness=0.0,
        random_contrast=0.0,
        noise_std=0.0,
        elastic_deformation=True,
        elastic_num_control_points=5,
        elastic_max_displacement=3.0,
    )

    augmented = aug(tensor.clone())
    assert augmented.shape == tensor.shape, "Elastic deformation changed tensor shape"
    assert not torch.equal(tensor, augmented), "Elastic deformation did not modify tensor"
    print("  Elastic deformation test passed!")


def test_get_augmentation_from_yaml():
    print("\nTesting YAML integration...")
    config_path = Path(__file__).resolve().parent.parent / "configs" / "data_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    aug = get_augmentation(data_cfg, is_train=True)
    assert aug is not None, "Augmentation should be enabled in data_config.yaml"
    assert aug.random_brightness == data_cfg["augmentation"]["random_brightness"]
    assert aug.random_contrast == data_cfg["augmentation"]["random_contrast"]
    assert aug.flip_probability == data_cfg["augmentation"]["flip_probability"]
    assert aug.elastic_deformation is False
    assert aug.rotation_range == 0
    assert aug.shift_3d_range == 0
    assert aug.noise_std == data_cfg["augmentation"]["noise_std"]

    description = aug.describe_active_transforms()
    assert "brightness" in description
    assert "elastic" not in description
    print(f"  Active transforms: {description}")
    print("  YAML integration test passed!")


def test_get_augmentation_disabled_for_val():
    print("\nTesting val split has no augmentation...")
    aug = get_augmentation({"augmentation": {"enabled": True}}, is_train=False)
    assert aug is None, "Validation should not use augmentation"
    print("  Val augmentation disabled test passed!")


if __name__ == "__main__":
    try:
        test_3d_augmentation()
        test_2d_augmentation()
        test_brightness_changes_values()
        test_contrast_scales_around_mean()
        test_flip_probability_zero_and_one()
        test_elastic_preserves_shape()
        test_get_augmentation_from_yaml()
        test_get_augmentation_disabled_for_val()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
