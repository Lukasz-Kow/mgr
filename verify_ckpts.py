import torch
import sys
from pathlib import Path

checkpoints = [
    'checkpoints/baseline/best_model.pth',
    'checkpoints/selective_net/best_model.pt',
    'checkpoints/evidential/best_model.pt',
    'checkpoints/hybrid/best_model.pt'
]

for ckpt in checkpoints:
    p = Path(ckpt)
    if not p.exists():
        print(f"Skipping {ckpt} (not found)")
        continue
    try:
        data = torch.load(p, map_location='cpu')
        print(f"Loaded {ckpt} successfully. Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    except Exception as e:
        print(f"Failed to load {ckpt}: {e}")
