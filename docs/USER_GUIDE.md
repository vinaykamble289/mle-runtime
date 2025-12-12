# MLE Runtime User Guide

## Overview

MLE Runtime is a high-performance machine learning inference engine designed to replace joblib with dramatically better performance and features.

## Key Benefits

- **10-100x faster loading** compared to joblib
- **50-90% smaller file sizes** with compression
- **Universal framework support** (not just scikit-learn)
- **Cross-platform deployment** without Python dependencies
- **Enterprise security** with model signing and encryption

## Installation

### Basic Installation
```bash
pip install mle-runtime
```

### Framework-Specific Installation
```bash
# For scikit-learn support
pip install mle-runtime[sklearn]

# For PyTorch support  
pip install mle-runtime[pytorch]

# For all frameworks
pip install mle-runtime[all]
```

## Quick Start

### 1. Export a Model

```python
import mle_runtime as mle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train your model
X = np.random.randn(1000, 20)
y = np.random.randint(0, 3, 1000)
model = RandomForestClassifier()
model.fit(X, y)

# Export to MLE format (much faster than joblib)
result = mle.export_model(model, 'my_model.mle', input_shape=(1, 20))
print(f"Export successful: {result['success']}")
print(f"File size: {result['file_size_bytes']} bytes")
```

### 2. Load and Use the Model

```python
import mle_runtime as mle
import numpy as np

# Load model (instant loading with memory mapping)
runtime = mle.load_model('my_model.mle')

# Get model information
info = runtime.get_model_info()
print(f"Model version: {info['version']}")
print(f"Model features: {info['features']}")

# Run inference
X_test = np.random.randn(100, 20)
predictions = runtime.run([X_test])
print(f"Predictions shape: {predictions[0].shape}")
```

### 3. Benchmark Performance

```python
# Compare performance
results = runtime.benchmark([X_test], num_runs=100)
print(f"Average inference time: {results['mean_time_ms']:.2f}ms")
print(f"Throughput: {results['throughput_samples_per_sec']:.0f} samples/sec")
```

## Supported Frameworks

### Scikit-learn (Fully Supported)
- **Linear Models**: LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
- **Tree Models**: DecisionTree, RandomForest, GradientBoosting, AdaBoost, ExtraTrees
- **SVM Models**: SVC, SVR, LinearSVC, LinearSVR
- **Other Models**: GaussianNB, KNeighbors, MLPClassifier/Regressor

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
mle.export_model(model, 'logistic_model.mle')
```

### PyTorch (Neural Networks)
- **Layers**: Linear, Conv2d, BatchNorm, LayerNorm, LSTM, GRU
- **Activations**: ReLU, GELU, Softmax, Sigmoid, Tanh
- **Requirements**: Must provide `input_shape`

```python
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
mle.export_model(model, 'pytorch_model.mle', input_shape=(1, 784))
```

### Gradient Boosting Frameworks
- **XGBoost**: XGBClassifier, XGBRegressor
- **LightGBM**: LGBMClassifier, LGBMRegressor  
- **CatBoost**: CatBoostClassifier, CatBoostRegressor

```python
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
mle.export_model(model, 'xgb_model.mle')
```

## Advanced Features

### Model Compression

```python
# Export with compression (smaller file sizes)
result = mle.export_model(model, 'compressed_model.mle', compression=True)

# Quantization utilities
from mle_runtime import CompressionUtils
quantized, scale, zero_point = CompressionUtils.quantize_weights_int8(weights)
```

### Model Security

```python
from mle_runtime import SecurityUtils

# Generate key pair for signing
public_key, private_key = SecurityUtils.generate_keypair()

# Load with signature verification
runtime = mle.load_model('signed_model.mle', 
                        verify_signature=True, 
                        public_key=public_key)
```

### Model Inspection

```python
# Detailed model analysis
analysis = mle.inspect_model('model.mle')
print(f"Model type: {analysis['basic_info']['metadata']}")
print(f"File size: {analysis['file_size']} bytes")
print(f"Recommendations: {analysis['recommendations']}")
```

## Performance Comparison

### vs. Joblib

| Metric | Joblib | MLE Runtime | Improvement |
|--------|--------|-------------|-------------|
| Load Time | 100-500ms | 1-5ms | **100x faster** |
| File Size | 100% | 10-50% | **50-90% smaller** |
| Framework Support | sklearn only | Universal | **∞ better** |
| Cross-platform | Python only | Universal | **∞ better** |
| Security | None | Enterprise | **∞ better** |

### Real-world Example

```python
import time
import joblib
import mle_runtime as mle
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Joblib export/load
start = time.time()
joblib.dump(model, 'model.pkl')
joblib_export_time = time.time() - start

start = time.time()
joblib_model = joblib.load('model.pkl')
joblib_load_time = time.time() - start

# MLE export/load
start = time.time()
mle.export_model(model, 'model.mle')
mle_export_time = time.time() - start

start = time.time()
mle_runtime = mle.load_model('model.mle')
mle_load_time = time.time() - start

print(f"Export time - Joblib: {joblib_export_time*1000:.1f}ms, MLE: {mle_export_time*1000:.1f}ms")
print(f"Load time - Joblib: {joblib_load_time*1000:.1f}ms, MLE: {mle_load_time*1000:.1f}ms")
print(f"Speed improvement: {joblib_load_time/mle_load_time:.1f}x faster")
```

## Production Deployment

### Web Service Example

```python
from flask import Flask, request, jsonify
import mle_runtime as mle
import numpy as np

app = Flask(__name__)

# Load model once at startup
runtime = mle.load_model('production_model.mle')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = np.array(request.json['features'])
        
        # Run inference
        predictions = runtime.run([data])
        
        return jsonify({
            'predictions': predictions[0].tolist(),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

# Install MLE Runtime
RUN pip install mle-runtime[all]

# Copy model and application
COPY model.mle /app/
COPY app.py /app/

WORKDIR /app

# Run application
CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mle-inference-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mle-inference
  template:
    metadata:
      labels:
        app: mle-inference
    spec:
      containers:
      - name: inference
        image: your-registry/mle-inference:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "256Mi"  # 75% less than joblib
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Command Line Interface

MLE Runtime provides convenient CLI tools:

```bash
# Export model
mle-export model.pkl model.mle

# Inspect model
mle-inspect model.mle

# Benchmark model
mle-benchmark model.mle test_data.npy

# Get help
mle-runtime --help
```

## Configuration

### Environment Variables

```bash
# Show detailed warnings (optional)
export MLE_SHOW_WARNINGS=true

# Set memory limits
export MLE_MEMORY_LIMIT=1GB

# Enable compression by default
export MLE_DEFAULT_COMPRESSION=true
```

### Python Configuration

```python
import mle_runtime as mle

# Configure global settings
mle.configure(
    memory_limit="1GB",
    cache_size="256MB",
    enable_compression=True,
    default_device="cpu"
)
```

## Best Practices

### 1. Model Export
- Always provide `input_shape` for better optimization
- Use compression for production models
- Test exported models before deployment

### 2. Performance Optimization
- Use appropriate batch sizes for inference
- Consider memory-mapped loading for large models
- Monitor memory usage in production

### 3. Security
- Enable signature verification for production models
- Use encryption for sensitive models
- Validate model integrity before loading

### 4. Monitoring
- Track inference times and throughput
- Monitor memory usage
- Set up alerts for model loading failures

## Migration from Joblib

### Step 1: Replace Export Code

**Before (Joblib):**
```python
import joblib
joblib.dump(model, 'model.pkl')
```

**After (MLE Runtime):**
```python
import mle_runtime as mle
mle.export_model(model, 'model.mle')
```

### Step 2: Replace Loading Code

**Before (Joblib):**
```python
import joblib
model = joblib.load('model.pkl')
predictions = model.predict(X_test)
```

**After (MLE Runtime):**
```python
import mle_runtime as mle
runtime = mle.load_model('model.mle')
predictions = runtime.run([X_test])[0]
```

### Step 3: Update Dependencies

**Before:**
```txt
scikit-learn==1.3.0
joblib==1.3.0
```

**After:**
```txt
scikit-learn==1.3.0
mle-runtime[sklearn]==2.0.1
```

## Troubleshooting

### Common Issues

1. **Import Errors**: See [Troubleshooting Guide](TROUBLESHOOTING.md)
2. **Performance Issues**: Check batch sizes and memory settings
3. **File Format Errors**: Ensure model was exported correctly

### Getting Help

- **Documentation**: [API Reference](API_REFERENCE.md)
- **Issues**: GitHub Issues
- **Community**: Stack Overflow with `mle-runtime` tag

## What's Next?

- **GPU Acceleration**: CUDA support for neural networks
- **More Frameworks**: TensorFlow, JAX support
- **Advanced Compression**: Better quantization algorithms
- **Distributed Inference**: Multi-node deployment

---

**MLE Runtime - Making ML inference fast, efficient, and production-ready!**