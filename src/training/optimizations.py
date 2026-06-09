
import torch
import os
import sys

# Bardziej odporny import IPEX
ipex = None
try:
    import intel_extension_for_pytorch as ipex
except (ImportError, AttributeError, RuntimeError, SystemExit) as e:
    # Ignorujemy błąd - na układach NVIDIA (CUDA) IPEX nie jest nam potrzebny
    ipex = None

def setup_cuda_optimizations():
    """
    Apply global optimizations for Nvidia GPUs (TF32 and cuDNN benchmark).
    """
    if torch.cuda.is_available():
        # Enable TF32 for matrix multiplications (Ampere+ GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable TF32 for cuDNN
        torch.backends.cudnn.allow_tf32 = True
        # Disable cuDNN benchmark for 3D CNNs on constrained hardware (causes very long hangs on first batch)
        torch.backends.cudnn.benchmark = False
        print("[CUDA] CUDA optimizations enabled: TF32 (cuDNN benchmark disabled to prevent hanging)")

def get_optimized_device(config_device=None):
    """
    Detect the best available device, prioritizing CUDA, then Intel XPU, then CPU.
    Respects config_device='cpu' to force CPU training.
    """
    if config_device == 'cpu':
        return torch.device("cpu")

    if torch.cuda.is_available():
        setup_cuda_optimizations()
        return torch.device("cuda")
    
    if ipex is not None and hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device("xpu")
    
    return torch.device("cpu")

def optimize_model_and_optimizer(model, optimizer=None, dtype=torch.float32, device=None):
    """
    Apply hardware-specific optimizations:
    - Nvidia/CUDA: PyTorch 2.0+ torch.compile
    - Intel: IPEX optimizations
    """
    # PyTorch 2.0 compile for CUDA
    if device is not None and device.type == 'cuda' and hasattr(torch, 'compile'):
        # Triton (używany przez torch.compile) wymaga Compute Capability >= 7.0
        major, minor = torch.cuda.get_device_capability(device)
        if major >= 7:
            try:
                model = torch.compile(model)
                print("[compile] PyTorch 2.0 model compilation enabled (torch.compile)")
            except Exception as e:
                print(f"[compile WARNING] torch.compile failed or not supported: {e}")
        else:
            print(f"[compile INFO] Skipping torch.compile because CUDA Capability is {major}.{minor} (requires >= 7.0)")

    # Intel IPEX optimizations
    if ipex is not None and (device is None or device.type in ['cpu', 'xpu']):
        try:
            if optimizer:
                model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype)
            else:
                model = ipex.optimize(model, dtype=dtype)
            print(f"[IPEX] Intel IPEX optimizations applied (dtype={dtype})")
        except AssertionError as e:
            if "BF16 weight prepack" in str(e):
                print(f"[IPEX WARNING] Hardware doesn't support BF16 prepacking. Retrying with weights_prepack=False...")
                if optimizer:
                    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=dtype, weights_prepack=False)
                else:
                    model = ipex.optimize(model, dtype=dtype, weights_prepack=False)
                print(f"[IPEX] Intel IPEX optimizations applied (without prepacking)")
            else:
                print(f"[IPEX ERROR] IPEX optimization failed: {e}. Continuing without IPEX optimization.")
    
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
            print("[IPEX INFO] BFloat16 not supported by CPU, using Float32 (Standard)")
            enabled = False
            dtype = torch.float32
    else:
        enabled = False
        dtype = torch.float32
        
    return enabled, dtype
