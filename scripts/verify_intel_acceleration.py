
import torch
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.optimizations import get_optimized_device, optimize_model_and_optimizer, get_amp_config

def verify_intel():
    print("="*60)
    print("🔍 VERIFYING INTEL ACCELERATION (IPEX / XPU / BFloat16)")
    print("="*60)
    
    # 1. Device Detection
    device = get_optimized_device()
    print(f"Detected Device: {device}")
    
    try:
        import intel_extension_for_pytorch as ipex
        print(f"IPEX Version: {ipex.__version__}")
    except ImportError:
        print("❌ IPEX not found! Run: pip install intel-extension-for-pytorch")
        return

    # 2. Benchmark Model
    print("\n🏃 Running Benchmark (3D Convolution)...")
    
    # Simulate a typical 3D MRI batch (B=2, C=1, D=128, H=128, W=128)
    # Using smaller size for quick test
    input_size = (2, 1, 64, 64, 64)
    x = torch.randn(input_size).to(device)
    
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv3d(32, 64, kernel_size=3, padding=1),
        torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 2)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # AMP config
    enabled, dtype = get_amp_config(device)
    print(f"AMP Enabled: {enabled}, DType: {dtype}")
    
    # Optimize
    model, optimizer = optimize_model_and_optimizer(model, optimizer, dtype=dtype)
    
    # Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            out = model(x)
            loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Measure
    print("Measuring speed...")
    torch.xpu.synchronize() if device.type == 'xpu' else None
    start_time = time.time()
    
    iters = 10
    for _ in range(iters):
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            out = model(x)
            loss = out.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.xpu.synchronize() if device.type == 'xpu' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iters
    print(f"\n✅ Benchmark Complete!")
    print(f"   Average Iteration Time: {avg_time:.4f}s")
    print(f"   Device used: {device}")
    
    if device.type == 'xpu':
        print("🚀 EXCELLENT: Your Intel Iris Xe is being used for acceleration!")
    elif device.type == 'cpu':
        print("💡 NOTE: Running on CPU, but using IPEX + BFloat16 optimizations.")
    
    print("="*60)

if __name__ == "__main__":
    verify_intel()
