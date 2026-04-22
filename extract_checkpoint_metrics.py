#!/usr/bin/env python3
"""Extract validation metrics stored inside checkpoint files."""
import torch
import os
import json

checkpoints = {
    'Baseline (SR)': 'checkpoints/baseline/best_model.pth',
    'SelectiveNet': 'checkpoints/selective_net/best_model.pt',
    'Evidential (EDL)': 'checkpoints/evidential/best_model.pt',
    'Hybrid (3D-ResNet-EDL)': 'checkpoints/hybrid/best_model.pt',
}

for name, path in checkpoints.items():
    if not os.path.exists(path):
        print(f"\n{'='*60}")
        print(f"  {name}: CHECKPOINT NOT FOUND ({path})")
        continue
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Path: {path}")
    print(f"  Size: {os.path.getsize(path)/1024/1024:.1f} MB")
    print(f"{'='*60}")
    
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        # Print all keys
        print(f"  Checkpoint keys: {list(ckpt.keys())}")
        
        # Epoch
        if 'epoch' in ckpt:
            print(f"  Best epoch: {ckpt['epoch']}")
        
        # Config info
        if 'config' in ckpt:
            cfg = ckpt['config']
            if 'training' in cfg:
                print(f"  Epochs configured: {cfg['training'].get('epochs', '?')}")
                print(f"  Learning rate: {cfg['training'].get('learning_rate', '?')}")
                print(f"  Batch size: {cfg['training'].get('batch_size', '?')}")
        
        # Validation metrics
        if 'val_metrics' in ckpt:
            m = ckpt['val_metrics']
            print(f"\n  --- Validation Metrics ---")
            for key, val in sorted(m.items()):
                if key in ['confusion_matrix']:
                    print(f"  {key}:")
                    print(f"    {val}")
                elif key in ['risk_coverage']:
                    rc = val
                    print(f"  risk_coverage: {len(rc.get('coverages', []))} points")
                elif isinstance(val, float):
                    print(f"  {key}: {val:.6f}")
                elif isinstance(val, (int, str)):
                    print(f"  {key}: {val}")
                else:
                    print(f"  {key}: {type(val).__name__}")
        else:
            print("  No val_metrics found in checkpoint")
        
        # State dict info
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
            keys = list(sd.keys())
            # Detect 2D vs 3D
            for k in keys:
                if 'conv' in k and 'weight' in k:
                    shape = sd[k].shape
                    dim = '2D' if len(shape) == 4 else '3D' if len(shape) == 5 else '?'
                    print(f"\n  Backbone type: {dim} (first conv shape: {shape})")
                    break
            print(f"  Total state dict keys: {len(keys)}")
            
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")

print(f"\n{'='*60}")
print("Done.")
