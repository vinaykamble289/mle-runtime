#!/usr/bin/env python3
"""Test if OpType enum values are correct in C++"""

from mle_runtime import Engine, Device
import numpy as np

# Create a simple test to see if DECOMPOSITION (32) is recognized
print("Testing OpType enum values...")

# Try to load and run a PCA model
try:
    engine = Engine(Device.CPU)
    engine.load_model('test_pca.mle')
    x = np.zeros((1, 10), dtype=np.float32)
    result = engine.run([x])
    print(f"SUCCESS: PCA model ran successfully!")
    print(f"Output shape: {result[0].shape}")
except RuntimeError as e:
    print(f"FAILED: {e}")
    print("\nThis means the C++ binary doesn't have the new OpType cases.")
    print("The Python module needs to be recompiled with the updated engine.cpp")
