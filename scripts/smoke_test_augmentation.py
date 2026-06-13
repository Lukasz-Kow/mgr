"""Smoke test: verify on-the-fly augmentation differs across repeated loads."""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MCIDataModule


def main() -> None:
    with open("configs/data_config.yaml", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    dm = MCIDataModule(
        metadata_csv=data_cfg["paths"]["metadata_csv"],
        preprocessor_config=data_cfg["preprocessing"],
        batch_size=1,
        num_workers=0,
        augmentation_config=data_cfg,
        cache_dir="cache/smoke_test_aug",
    )

    ds = dm.train_dataset()
    img1, _, _ = ds[0]
    img2, _, _ = ds[0]

    assert img1.shape == img2.shape, "Shape should be preserved"
    assert not (img1 == img2).all(), "Two augmented loads should differ"

    print(f"Shape: {img1.shape}")
    print(f"Max diff between two augmented loads: {(img1 - img2).abs().max().item():.6f}")
    print("Smoke test passed!")


if __name__ == "__main__":
    main()
