
import torch
import os
import sys

# Bardziej odporny import IPEX
ipex = None
try:
    import intel_extension_for_pytorch as ipex
except (ImportError, AttributeError, RuntimeError, SystemExit) as e:
    # SystemExit is important here because some IPEX versions call os.exit on version mismatch
    print(f"\n⚠️  Warning: Intel Extension for PyTorch (IPEX) could not be initialized.")
    print(f"   Details: {e}")
    print("   Falling back to standard PyTorch (CPU) modes.\n")
    ipex = None

def get_optimized_device(config_device=None):
    """
    Detect the best available device, prioritizing CUDA, then Intel XPU, then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if ipex is not None and hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device("xpu")
    
    return torch.device("cpu")

def optimize_model_and_optimizer(model, optimizer=None, dtype=torch.float32):
    """
    Apply Intel IPEX optimizations if available.
    """
    if ipex is not None:
        try:
            if optimizer:
                model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype)
            else:
                model = ipex.optimize(model, dtype=dtype)
            print(f"✅ Intel IPEX optimizations applied (dtype={dtype})")
        except AssertionError as e:
            if "BF16 weight prepack" in str(e):
                print(f"⚠️  Hardware doesn't support BF16 prepacking. Retrying with weights_prepack=False...")
                if optimizer:
                    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype, weights_prepack=False)
                else:
                    model = ipex.optimize(model, dtype=dtype, weights_prepack=False)
                print(f"✅ Intel IPEX optimizations applied (without prepacking)")
            else:
                print(f"❌ IPEX optimization failed: {e}. Continuing without IPEX optimization.")
    return model, optimizer

def get_amp_config(device):
    """
    Get the appropriate AMP (Automatic Mixed Precision) configuration for the device.
    """
    enabled = True
    if device.type == 'cuda':
        dtype = torch.float16
    elif device.type == 'xpu':
        dtype = torch.float16 
    elif device.type == 'cpu':
        # Check if hardware actually supports bfloat16
        if ipex is not None and hasattr(ipex, 'cpu') and hasattr(ipex.cpu, 'runtime') and hasattr(ipex.cpu.runtime, 'is_bf16_supported'):
             # Some versions use this
             bf16_supported = ipex.cpu.runtime.is_bf16_supported()
        else:
             # Standard torch check
             bf16_supported = torch.cpu.is_bf16_supported() if hasattr(torch.cpu, 'is_bf16_supported') else False
        
        if bf16_supported:
            dtype = torch.bfloat16
        else:
            print("ℹ️  BFloat16 not supported by CPU, using Float32 (Standard)")
            enabled = False
            dtype = torch.float32
    else:
        enabled = False
        dtype = torch.float32
        
    return enabled, dtype
