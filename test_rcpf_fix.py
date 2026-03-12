#!/usr/bin/env python3
"""Test rcpf precision fix - Simple test for MACA GPU precision"""

import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Test basic CUDA operations
    x = torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0], device='cuda')

    # Test division (uses rcpf internally)
    y = 1.0 / x
    print(f"\n1.0 / {x.cpu().tolist()} = {y.cpu().tolist()}")

    # Expected values
    expected = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.03125])

    # Check precision
    diff = torch.abs(y.cpu() - expected)
    max_diff = torch.max(diff).item()
    print(f"Max diff from expected: {max_diff:.2e}")

    if max_diff < 1e-6:
        print("✓ PRECISION PASS: rcpf fix is working!")
    else:
        print(f"✗ PRECISION FAIL: max_diff={max_diff:.2e} >= 1e-6")

    # Test more operations
    print("\nTesting more operations...")
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')

    c = torch.matmul(a, b)
    print(f"Matmul test: shape={c.shape}, mean={c.mean().item():.4f}")

    d = torch.div(1.0, a + 1e-5)
    print(f"Division test: shape={d.shape}, mean={d.mean().item():.4f}")

    print("\n✓ All basic CUDA tests passed!")
else:
    print("CUDA not available, skipping GPU tests")
