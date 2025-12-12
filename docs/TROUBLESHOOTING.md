# MLE Runtime Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### 1. Package Not Found
**Error:** `Could not find a version that satisfies the requirement mle-runtime`

**Solutions:**
```bash
# Update pip first
pip install --upgrade pip

# Try with explicit index
pip install --index-url https://pypi.org/simple/ mle-runtime

# Check Python version (requires 3.8+)
python --version
```

#### 2. Permission Denied
**Error:** `Permission denied` or `Access is denied`

**Solutions:**
```bash
# Install for current user only
pip install --user mle-runtime

# Use virtual environment (recommended)
python -m venv mle_env
source mle_env/bin/activate  # Linux/Mac
# or
mle_env\Scripts\activate     # Windows
pip install mle-runtime
```

#### 3. Dependency Conflicts
**Error:** Various dependency conflict warnings

**Solutions:**
```bash
# Create clean virtual environment
python -m venv clean_env
source clean_env/bin/activate  # Linux/Mac
clean_env\Scripts\activate     # Windows
pip install mle-runtime

# Install specific framework support
pip install mle-runtime[sklearn]  # for scikit-learn
pip install mle-runtime[pytorch]  # for PyTorch
pip install mle-runtime[all]      # for all frameworks
```

### Import Issues

#### 1. Module Not Found
**Error:** `ModuleNotFoundError: No module named 'mle_runtime'`

**Solutions:**
```python
# Check installation
import sys
print(sys.path)

# Verify installation
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
print(result.stdout)

# Reinstall if needed
pip uninstall mle-runtime
pip install mle-runtime
```

#### 2. Core Runtime Warning
**Warning:** `Core C++ runtime not available. Using Python fallback implementations.`

**This is normal behavior:**
- The warning indicates C++ acceleration is not available
- Python fallback implementations are used instead
- Functionality remains the same, performance may be slower
- This is expected in the current version

**To suppress the warning:**
```python
import warnings
warnings.filterwarnings('ignore', message='Core C++ runtime not available')
import mle_runtime
```

### Export Issues

#### 1. Unsupported Model Type
**Error:** `Unsupported model type: <class 'some_model'>`

**Solutions:**
```python
# Check supported frameworks
import mle_runtime as mle
print("Supported operators:", mle.get_supported_operators())

# Use framework-specific exporters
from mle_runtime import export_sklearn_model, export_pytorch_model

# For unsupported models, check if framework is installed
try:
    import sklearn
    print("Scikit-learn available")
except ImportError:
    print("Install scikit-learn: pip install scikit-learn")
```

#### 2. Input Shape Required
**Error:** `input_shape is required for PyTorch models`

**Solution:**
```python
# Always provide input_shape for PyTorch models
import mle_runtime as mle
result = mle.export_model(pytorch_model, 'model.mle', input_shape=(1, 784))

# For other frameworks, input_shape is optional but recommended
result = mle.export_model(sklearn_model, 'model.mle', input_shape=(1, 20))
```

#### 3. Model Not Fitted
**Error:** Various errors related to unfitted models

**Solution:**
```python
# Ensure model is trained before export
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.randn(100, 10)
y = np.random.randint(0, 2, 100)

model = RandomForestClassifier()
model.fit(X, y)  # Important: fit the model first

# Now export
import mle_runtime as mle
result = mle.export_model(model, 'model.mle')
```

### Loading Issues

#### 1. File Not Found
**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'model.mle'`

**Solutions:**
```python
import os
from pathlib import Path

# Check if file exists
if os.path.exists('model.mle'):
    print("File exists")
else:
    print("File not found")

# Use absolute path
model_path = Path('model.mle').absolute()
runtime = mle.load_model(str(model_path))

# List files in current directory
print("Files in current directory:", os.listdir('.'))
```

#### 2. Invalid Model Format
**Error:** `Invalid magic number` or `Unsupported model version`

**Solutions:**
```python
# Check if file is actually an MLE file
with open('model.mle', 'rb') as f:
    header = f.read(8)
    print("File header:", header.hex())

# Re-export the model if corrupted
# Original model -> export again -> new MLE file

# Check file size
import os
size = os.path.getsize('model.mle')
print(f"File size: {size} bytes")
if size < 100:
    print("File seems too small, might be corrupted")
```

#### 3. Signature Verification Failed
**Error:** `Signature verification failed`

**Solutions:**
```python
# Load without signature verification
runtime = mle.load_model('model.mle', verify_signature=False)

# Or provide correct public key
runtime = mle.load_model('model.mle', verify_signature=True, public_key=correct_key)
```

### Runtime Issues

#### 1. Inference Errors
**Error:** Various errors during `runtime.run()`

**Solutions:**
```python
# Check input format
import numpy as np

# Ensure inputs are numpy arrays
inputs = [np.array(data, dtype=np.float32)]

# Check input shape matches model expectations
info = runtime.get_model_info()
print("Model info:", info)

# Ensure correct number of inputs
predictions = runtime.run(inputs)  # inputs should be a list
```

#### 2. Memory Issues
**Error:** `MemoryError` or out of memory errors

**Solutions:**
```python
# Process in smaller batches
batch_size = 100
for i in range(0, len(X), batch_size):
    batch = X[i:i+batch_size]
    predictions = runtime.run([batch])

# Use appropriate data types
X = X.astype(np.float32)  # Use float32 instead of float64

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

### Performance Issues

#### 1. Slow Loading
**Issue:** Model loading takes too long

**Solutions:**
```python
# MLE files should load very quickly due to memory mapping
# If loading is slow, check:

# 1. File size
import os
size = os.path.getsize('model.mle')
print(f"File size: {size / 1024 / 1024:.2f} MB")

# 2. Disk speed (SSD vs HDD)
# 3. Available memory

# 4. Use compression for smaller files
result = mle.export_model(model, 'model.mle', compression=True)
```

#### 2. Slow Inference
**Issue:** Inference is slower than expected

**Solutions:**
```python
# 1. Use appropriate batch sizes
# Too small: overhead dominates
# Too large: memory issues

# 2. Check data types
inputs = [np.array(data, dtype=np.float32)]  # Use float32

# 3. Benchmark to identify bottlenecks
results = runtime.benchmark(inputs, num_runs=100)
print("Benchmark results:", results)

# 4. Consider GPU acceleration (when available)
runtime = mle.load_model('model.mle', device='cuda')
```

### Framework-Specific Issues

#### Scikit-learn Issues

```python
# 1. Check scikit-learn version
import sklearn
print("Scikit-learn version:", sklearn.__version__)

# 2. Supported models
supported_models = [
    'LogisticRegression', 'LinearRegression', 'RandomForestClassifier',
    'SVC', 'GaussianNB', 'KNeighborsClassifier', 'DecisionTreeClassifier'
]

# 3. Model must be fitted
model.fit(X_train, y_train)  # Required before export
```

#### PyTorch Issues

```python
# 1. Model must be in eval mode
model.eval()

# 2. Input shape is required
input_shape = (1, 784)  # Batch size, features

# 3. Check PyTorch version
import torch
print("PyTorch version:", torch.__version__)

# 4. Handle device issues
model = model.cpu()  # Move to CPU before export
```

#### TensorFlow Issues

```python
# 1. Check TensorFlow version
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# 2. Model should be compiled
model.compile(optimizer='adam', loss='mse')

# 3. Handle Keras vs TensorFlow models
# Both tf.keras.Model and tf.keras.Sequential are supported
```

### Debugging Tips

#### 1. Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

import mle_runtime as mle
# Now all operations will show detailed logs
```

#### 2. Check Model Information

```python
# After loading, inspect the model
runtime = mle.load_model('model.mle')
info = runtime.get_model_info()

print("Model version:", info['version'])
print("Features:", info['features'])
print("Model size:", info['model_size_bytes'])
print("Compression ratio:", info['compression_ratio'])
```

#### 3. Test with Simple Models

```python
# Create a simple test model
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

model = LogisticRegression()
model.fit(X, y)

# Test export
result = mle.export_model(model, 'test.mle', input_shape=(1, 5))
print("Export result:", result)

# Test loading
runtime = mle.load_model('test.mle')
predictions = runtime.run([X[:5]])
print("Predictions shape:", predictions[0].shape)
```

### Environment Issues

#### 1. Virtual Environment Problems

```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
fresh_env\Scripts\activate     # Windows

# Install only what you need
pip install mle-runtime[sklearn]

# Test installation
python -c "import mle_runtime; print('Success!')"
```

#### 2. Python Version Issues

```python
# Check Python version
import sys
print("Python version:", sys.version)

# MLE Runtime requires Python 3.8+
if sys.version_info < (3, 8):
    print("Please upgrade to Python 3.8 or higher")
```

#### 3. Operating System Issues

**Windows:**
```powershell
# Use PowerShell as Administrator if needed
# Check Windows version
Get-ComputerInfo | Select WindowsProductName, WindowsVersion

# Install Visual C++ Redistributable if needed
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Use Homebrew Python if system Python causes issues
brew install python
```

**Linux:**
```bash
# Install development headers
sudo apt-get install python3-dev  # Ubuntu/Debian
sudo yum install python3-devel    # CentOS/RHEL
```

### Getting Help

#### 1. Check Documentation
- [API Reference](API_REFERENCE.md)
- [Installation Guide](../INSTALLATION.md)
- [Examples](../examples/)

#### 2. Community Support
- GitHub Issues: https://github.com/mle-runtime/mle-runtime/issues
- Stack Overflow: Tag questions with `mle-runtime`

#### 3. Diagnostic Information

When reporting issues, include:

```python
import mle_runtime as mle
import sys
import platform

print("=== MLE Runtime Diagnostic Information ===")
print(f"MLE Runtime version: {mle.__version__}")
print(f"Python version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")

# Version info
try:
    info = mle.get_version_info()
    print(f"MLE features: {info['features']}")
    print(f"Operators: {info['operators']}")
except Exception as e:
    print(f"Error getting version info: {e}")

# Test basic functionality
try:
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    X = np.random.randn(10, 5)
    y = np.random.randint(0, 2, 10)
    model = LogisticRegression().fit(X, y)
    
    result = mle.export_model(model, 'diagnostic_test.mle')
    print(f"Export test: {'✅ PASS' if result['success'] else '❌ FAIL'}")
    
    runtime = mle.load_model('diagnostic_test.mle')
    predictions = runtime.run([X[:2]])
    print(f"Inference test: ✅ PASS")
    
except Exception as e:
    print(f"Basic functionality test: ❌ FAIL - {e}")

print("=== End Diagnostic Information ===")
```

This diagnostic information helps identify the root cause of issues quickly.