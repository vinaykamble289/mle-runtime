#!/usr/bin/env python3
"""
Complete workflow example: Train → Export → Load → Infer
Demonstrates the full pipeline from PyTorch to MLE runtime
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools/exporter'))
from pytorch_to_mle import MLEExporter

try:
    import mle_runtime
    HAS_RUNTIME = True
except ImportError:
    print("Warning: mle_runtime not installed. Install with: cd bindings/python && pip install -e .")
    HAS_RUNTIME = False


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for demonstration"""
    def __init__(self, input_size=128, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def train_model(epochs=5):
    """Train a simple model on synthetic data"""
    print("=" * 60)
    print("Step 1: Training PyTorch Model")
    print("=" * 60)
    
    # Create model
    model = SimpleClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Synthetic data
    batch_size = 32
    input_size = 128
    num_classes = 10
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        # Generate batch
        X = torch.randn(batch_size, input_size)
        y = torch.randint(0, num_classes, (batch_size,))
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 1 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    model.eval()
    print("✓ Training complete\n")
    return model


def export_model(model, output_path="model.mle"):
    """Export PyTorch model to .mle format"""
    print("=" * 60)
    print("Step 2: Exporting to .mle Format")
    print("=" * 60)
    
    exporter = MLEExporter()
    input_shape = (1, 128)
    
    print(f"Exporting to: {output_path}")
    print(f"Input shape: {input_shape}")
    
    start = time.perf_counter()
    exporter.export_mlp(model, input_shape, output_path)
    end = time.perf_counter()
    
    print(f"✓ Export complete in {(end - start) * 1000:.2f} ms")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.2f} KB\n")
    
    return output_path


def inspect_model(model_path):
    """Inspect .mle file"""
    print("=" * 60)
    print("Step 3: Inspecting .mle File")
    print("=" * 60)
    
    import struct
    import json
    
    with open(model_path, 'rb') as f:
        # Read header (actually 80 bytes: 2*uint32 + 7*uint64 + 16 reserved)
        header = f.read(80)
        if len(header) < 80:
            print(f"Error: File too small ({len(header)} bytes)")
            return
        # Unpack: magic(4) + version(4) + 7 uint64(56) + 16 reserved = 80 bytes
        data = struct.unpack('II QQQQQQQ 16s', header)
        magic, version, meta_off, meta_size, graph_off, graph_size, \
        weights_off, weights_size, sig_off = data[:9]
        
        print(f"Magic: 0x{magic:08x}")
        print(f"Version: {version}")
        print(f"Metadata size: {meta_size} bytes")
        print(f"Graph size: {graph_size} bytes")
        print(f"Weights size: {weights_size / 1024:.2f} KB")
        
        # Read metadata
        if meta_size > 0:
            f.seek(meta_off)
            meta_bytes = f.read(meta_size)
            try:
                metadata = json.loads(meta_bytes.decode('utf-8'))
                print(f"Model name: {metadata.get('model_name', 'N/A')}")
            except:
                print(f"Metadata: (binary, {meta_size} bytes)")
        
        # Read graph info
        f.seek(graph_off)
        num_nodes, num_tensors = struct.unpack('II', f.read(8))
        print(f"Graph nodes: {num_nodes}")
        print(f"Tensors: {num_tensors}")
    
    print("✓ Inspection complete\n")


def run_inference(model_path, pytorch_model):
    """Load .mle and run inference, compare with PyTorch"""
    if not HAS_RUNTIME:
        print("Skipping inference (mle_runtime not available)")
        return
    
    print("=" * 60)
    print("Step 4: Running Inference")
    print("=" * 60)
    
    # Try GPU first, fallback to CPU
    try:
        device = mle_runtime.Device.CUDA
        engine = mle_runtime.Engine(device)
        print("Using device: CUDA")
    except:
        device = mle_runtime.Device.CPU
        engine = mle_runtime.Engine(device)
        print("Using device: CPU")
    
    # Load model
    print(f"Loading model: {model_path}")
    start = time.perf_counter()
    engine.load_model(model_path)
    end = time.perf_counter()
    print(f"✓ Cold load time: {(end - start) * 1000:.2f} ms")
    
    # Create test input
    np.random.seed(42)
    input_data = np.random.randn(1, 128).astype(np.float32)
    
    # MLE inference
    print("\nRunning MLE inference...")
    start = time.perf_counter()
    outputs_mle = engine.run([input_data])
    end = time.perf_counter()
    mle_time = (end - start) * 1000
    output_mle = outputs_mle[0]
    
    print(f"✓ Inference time: {mle_time:.2f} ms")
    print(f"  Output shape: {output_mle.shape}")
    print(f"  Peak memory: {engine.peak_memory_usage() / 1024 / 1024:.2f} MB")
    
    # PyTorch inference for comparison
    print("\nRunning PyTorch inference for comparison...")
    pytorch_model.eval()
    with torch.no_grad():
        input_torch = torch.from_numpy(input_data)
        output_torch = pytorch_model(input_torch).numpy()
    
    # Compare outputs
    print("\nComparing outputs:")
    print(f"  MLE output (first 5): {output_mle.flatten()[:5]}")
    print(f"  PyTorch output (first 5): {output_torch.flatten()[:5]}")
    
    abs_error = np.abs(output_torch - output_mle)
    rel_error = abs_error / (np.abs(output_torch) + 1e-8)
    
    print(f"\nError metrics:")
    print(f"  Max absolute error: {np.max(abs_error):.6f}")
    print(f"  Max relative error: {np.max(rel_error):.6f}")
    print(f"  Mean absolute error: {np.mean(abs_error):.6f}")
    
    if np.max(rel_error) < 1e-3:
        print("✓ Outputs match within tolerance (< 1e-3)")
    else:
        print("⚠ Outputs differ more than expected")
    
    # Benchmark
    print("\nBenchmarking (100 iterations)...")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        engine.run([input_data])
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  P95: {np.percentile(times, 95):.2f} ms")
    print(f"  P99: {np.percentile(times, 99):.2f} ms")
    
    print("\n✓ Inference complete\n")


def main():
    print("\n" + "=" * 60)
    print("Complete Workflow: PyTorch → .mle → Inference")
    print("=" * 60 + "\n")
    
    # Step 1: Train
    model = train_model(epochs=5)
    
    # Step 2: Export
    model_path = export_model(model, "example_model.mle")
    
    # Step 3: Inspect
    inspect_model(model_path)
    
    # Step 4: Inference
    run_inference(model_path, model)
    
    print("=" * 60)
    print("Workflow Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Try with your own PyTorch model")
    print("  2. Experiment with different architectures")
    print("  3. Measure performance on your hardware")
    print("  4. Build visual pipelines in FlowForge (cd frontend && npm run dev)")
    print()


if __name__ == '__main__':
    main()
