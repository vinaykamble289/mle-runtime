# MLE Runtime API Reference

## Overview

MLE Runtime is a high-performance machine learning inference engine that provides 10-100x faster model loading and 50-90% smaller file sizes compared to traditional serialization tools like joblib.

## Installation

```bash
pip install mle-runtime
```

## Quick Start

```python
import mle_runtime as mle

# Export any model
mle.export_model(your_model, 'model.mle')

# Load and run inference
runtime = mle.load_model('model.mle')
predictions = runtime.run([X_test])
```

## Core Classes

### MLERuntime

The main runtime class for loading and running MLE models.

```python
class MLERuntime:
    def __init__(self, device: str = "cpu")
    def load_model(self, path: Union[str, Path], **kwargs) -> Dict[str, Any]
    def run(self, inputs: List[np.ndarray]) -> List[np.ndarray]
    def get_model_info(self) -> Dict[str, Any]
    def benchmark(self, inputs: List[np.ndarray], num_runs: int = 100) -> Dict[str, float]
```

**Parameters:**
- `device`: Compute device ("cpu" or "cuda")

**Methods:**

#### `load_model(path, **kwargs)`
Load an MLE model file.

**Parameters:**
- `path`: Path to .mle model file
- `verify_signature`: Whether to verify digital signature (default: False)
- `public_key`: Public key for signature verification

**Returns:**
- Dictionary with model information including version, features, and compression stats

#### `run(inputs)`
Run inference on loaded model.

**Parameters:**
- `inputs`: List of numpy arrays as model inputs

**Returns:**
- List of numpy arrays as model outputs

#### `get_model_info()`
Get comprehensive model information.

**Returns:**
- Dictionary with model metadata, features, and statistics

#### `benchmark(inputs, num_runs=100)`
Benchmark model performance.

**Parameters:**
- `inputs`: Input data for benchmarking
- `num_runs`: Number of benchmark iterations

**Returns:**
- Dictionary with performance statistics (mean, std, min, max times)

### MLEFormat

File format constants and specifications.

```python
class MLEFormat:
    # Magic number and version
    MLE_MAGIC = 0x00454C4D
    MLE_VERSION = 2
    
    # Feature flags
    FEATURE_COMPRESSION = 0x00000001
    FEATURE_ENCRYPTION = 0x00000002
    FEATURE_SIGNING = 0x00000004
    
    # Compression types
    COMPRESSION_NONE = 0
    COMPRESSION_LZ4 = 1
    COMPRESSION_ZSTD = 2
```

### CompressionUtils

Utilities for model compression and quantization.

```python
class CompressionUtils:
    @staticmethod
    def quantize_weights_int8(weights: np.ndarray) -> Tuple[np.ndarray, float, int]
    
    @staticmethod
    def quantize_weights_fp16(weights: np.ndarray) -> np.ndarray
    
    @staticmethod
    def compress_data(data: bytes, compression_type: int = 0) -> bytes
```

### SecurityUtils

Utilities for model security and integrity.

```python
class SecurityUtils:
    @staticmethod
    def compute_checksum(data: bytes) -> int
    
    @staticmethod
    def compute_hash(data: bytes) -> str
    
    @staticmethod
    def generate_keypair() -> Tuple[bytes, bytes]
```

## Main Functions

### export_model()

Universal model exporter that automatically detects model type.

```python
def export_model(model, output_path, input_shape=None, **kwargs) -> dict
```

**Parameters:**
- `model`: Trained model from any supported framework
- `output_path`: Path to save .mle file
- `input_shape`: Input shape tuple (required for some models)
- `**kwargs`: Additional export options

**Returns:**
- Dictionary with export information and statistics

**Supported Frameworks:**
- Scikit-learn (all major algorithms)
- PyTorch (neural networks)
- TensorFlow/Keras (neural networks)
- XGBoost (gradient boosting)
- LightGBM (gradient boosting)
- CatBoost (gradient boosting)

**Example:**
```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train model
X = np.random.randn(1000, 20)
y = np.random.randint(0, 3, 1000)
model = RandomForestClassifier()
model.fit(X, y)

# Export to MLE format
result = mle.export_model(model, 'rf_model.mle', input_shape=(1, 20))
print(f"Export successful: {result['success']}")
print(f"File size: {result['file_size_bytes']} bytes")
print(f"Compression ratio: {result['compression_ratio']}x")
```

### load_model()

Load MLE model and return runtime instance.

```python
def load_model(path: Union[str, Path], device: str = "cpu") -> MLERuntime
```

**Parameters:**
- `path`: Path to .mle model file
- `device`: Compute device ("cpu" or "cuda")

**Returns:**
- MLERuntime instance with loaded model

**Example:**
```python
# Load model
runtime = mle.load_model('rf_model.mle')

# Get model information
info = runtime.get_model_info()
print(f"Model version: {info['version']}")
print(f"Features: {info['features']}")

# Run inference
predictions = runtime.run([X_test])
```

### inspect_model()

Inspect MLE model without loading it for inference.

```python
def inspect_model(path: Union[str, Path]) -> dict
```

**Parameters:**
- `path`: Path to .mle model file

**Returns:**
- Dictionary with comprehensive model analysis

### benchmark_model()

Benchmark model performance.

```python
def benchmark_model(model_path, inputs, num_runs=100) -> dict
```

**Parameters:**
- `model_path`: Path to .mle model file
- `inputs`: Input data for benchmarking
- `num_runs`: Number of benchmark iterations

**Returns:**
- Dictionary with performance statistics

## Framework-Specific Exporters

### Scikit-learn

```python
from mle_runtime import export_sklearn_model

result = export_sklearn_model(sklearn_model, 'model.mle', input_shape=(1, 20))
```

**Supported Models:**
- Linear models: LogisticRegression, LinearRegression, Ridge, Lasso, etc.
- Tree models: DecisionTree, RandomForest, GradientBoosting, etc.
- SVM models: SVC, SVR, LinearSVC, LinearSVR
- Other: GaussianNB, KNeighbors, MLP

### PyTorch

```python
from mle_runtime import export_pytorch_model

result = export_pytorch_model(pytorch_model, 'model.mle', input_shape=(1, 784))
```

**Requirements:**
- `input_shape` is required for PyTorch models
- Model should be in evaluation mode

### TensorFlow/Keras

```python
from mle_runtime import export_tensorflow_model

result = export_tensorflow_model(keras_model, 'model.mle')
```

### Gradient Boosting

```python
from mle_runtime import export_xgboost_model, export_lightgbm_model, export_catboost_model

# XGBoost
result = export_xgboost_model(xgb_model, 'model.mle')

# LightGBM
result = export_lightgbm_model(lgb_model, 'model.mle')

# CatBoost
result = export_catboost_model(cb_model, 'model.mle')
```

## Utility Functions

### get_version_info()

Get comprehensive version information.

```python
info = mle.get_version_info()
print(f"Version: {info['version']}")
print(f"Features: {info['features']}")
print(f"Operators: {info['operators']}")
```

### get_supported_operators()

Get list of supported operators.

```python
operators = mle.get_supported_operators()
print(f"Supported operators: {len(operators)}")
```

## Error Handling

MLE Runtime provides comprehensive error handling:

```python
try:
    runtime = mle.load_model('model.mle')
    predictions = runtime.run([X_test])
except FileNotFoundError:
    print("Model file not found")
except ValueError as e:
    print(f"Invalid model format: {e}")
except RuntimeError as e:
    print(f"Runtime error: {e}")
```

## Performance Tips

1. **Use appropriate device**: Specify "cuda" for GPU acceleration when available
2. **Batch processing**: Process multiple samples together for better throughput
3. **Memory mapping**: MLE files are memory-mapped for instant loading
4. **Compression**: Enable compression for smaller file sizes

## Advanced Features

### Model Compression

```python
# Export with compression
result = mle.export_model(model, 'model.mle', compression=True)

# Quantization
from mle_runtime import CompressionUtils
quantized, scale, zero_point = CompressionUtils.quantize_weights_int8(weights)
```

### Model Security

```python
from mle_runtime import SecurityUtils

# Generate key pair
public_key, private_key = SecurityUtils.generate_keypair()

# Load with signature verification
runtime = mle.load_model('model.mle', verify_signature=True, public_key=public_key)
```

### Model Inspection

```python
# Detailed model analysis
analysis = mle.inspect_model('model.mle')
print(f"Model type: {analysis['basic_info']['metadata']}")
print(f"File size: {analysis['file_size']} bytes")
print(f"Recommendations: {analysis['recommendations']}")
```

## Command Line Interface

MLE Runtime provides command-line tools:

```bash
# Export model
mle-export model.pkl model.mle

# Inspect model
mle-inspect model.mle

# Benchmark model
mle-benchmark model.mle data.npy
```

## Integration Examples

### Flask Web Service

```python
from flask import Flask, request, jsonify
import mle_runtime as mle
import numpy as np

app = Flask(__name__)
runtime = mle.load_model('model.mle')

@app.route('/predict', methods=['POST'])
def predict():
    data = np.array(request.json['data'])
    predictions = runtime.run([data])
    return jsonify({'predictions': predictions[0].tolist()})
```

### Batch Processing

```python
import mle_runtime as mle
import numpy as np

runtime = mle.load_model('model.mle')

# Process large dataset in batches
batch_size = 1000
for i in range(0, len(X), batch_size):
    batch = X[i:i+batch_size]
    predictions = runtime.run([batch])
    # Process predictions
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.