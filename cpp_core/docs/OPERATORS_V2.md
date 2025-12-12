# MLE Runtime v2.0 - Enhanced Operators

## Overview

MLE Runtime v2.0 introduces significant enhancements to the operator library, adding support for modern neural network architectures and improved machine learning algorithms.

## New Neural Network Operators

### Conv2D - 2D Convolutional Layer

**Purpose**: Performs 2D convolution operations for image processing and feature extraction.

**Signature**:
```cpp
void conv2d_cpu(const TensorView& input, const TensorView& weight, 
                const TensorView& bias, TensorView& output,
                uint32_t stride_h = 1, uint32_t stride_w = 1,
                uint32_t pad_h = 0, uint32_t pad_w = 0);
```

**Parameters**:
- `input`: Input tensor [N, C_in, H_in, W_in]
- `weight`: Convolution kernels [C_out, C_in, K_h, K_w]
- `bias`: Bias terms [C_out]
- `output`: Output tensor [N, C_out, H_out, W_out]
- `stride_h/w`: Convolution stride (default: 1)
- `pad_h/w`: Zero padding (default: 0)

**Example**:
```cpp
auto input = TensorView::create({1, 3, 224, 224}, DType::FP32);  // RGB image
auto weight = TensorView::create({64, 3, 7, 7}, DType::FP32);    // 64 filters, 7x7 kernel
auto bias = TensorView::create({64}, DType::FP32);
auto output = TensorView::create({1, 64, 218, 218}, DType::FP32); // With stride=1, no padding

ops::conv2d_cpu(*input, *weight, *bias, *output, 1, 1, 0, 0);
```

### MaxPool2D - 2D Max Pooling

**Purpose**: Downsamples feature maps by taking the maximum value in each pooling window.

**Signature**:
```cpp
void maxpool2d_cpu(const TensorView& input, TensorView& output,
                   uint32_t kernel_h = 2, uint32_t kernel_w = 2,
                   uint32_t stride_h = 2, uint32_t stride_w = 2,
                   uint32_t pad_h = 0, uint32_t pad_w = 0);
```

**Parameters**:
- `input`: Input tensor [N, C, H_in, W_in]
- `output`: Output tensor [N, C, H_out, W_out]
- `kernel_h/w`: Pooling window size (default: 2x2)
- `stride_h/w`: Pooling stride (default: 2)
- `pad_h/w`: Zero padding (default: 0)

### BatchNorm - Batch Normalization

**Purpose**: Normalizes layer inputs to improve training stability and convergence.

**Signature**:
```cpp
void batchnorm_cpu(const TensorView& input, const TensorView& weight,
                   const TensorView& bias, const TensorView& running_mean,
                   const TensorView& running_var, TensorView& output,
                   float eps = 1e-5f);
```

**Parameters**:
- `input`: Input tensor [N, C, ...]
- `weight`: Scale parameters (gamma) [C]
- `bias`: Shift parameters (beta) [C]
- `running_mean`: Running mean statistics [C]
- `running_var`: Running variance statistics [C]
- `output`: Normalized output tensor
- `eps`: Small constant for numerical stability

### Attention - Multi-Head Self-Attention

**Purpose**: Implements the attention mechanism for transformer architectures.

**Signature**:
```cpp
void attention_cpu(const TensorView& query, const TensorView& key, 
                   const TensorView& value, TensorView& output,
                   uint32_t num_heads = 8, float scale = 0.125f);
```

**Parameters**:
- `query`: Query tensor [batch, seq_len, d_model]
- `key`: Key tensor [batch, seq_len, d_model]
- `value`: Value tensor [batch, seq_len, d_model]
- `output`: Attention output [batch, seq_len, d_model]
- `num_heads`: Number of attention heads
- `scale`: Attention scaling factor (typically 1/√d_head)

## Enhanced Tree and SVM Operators

### Tree Ensemble (Random Forest)

**Improvements**:
- Better memory layout for tree parameters
- Optimized tree traversal algorithms
- Support for both classification and regression
- Improved numerical stability

### SVM with RBF Kernel

**New Features**:
- RBF (Radial Basis Function) kernel implementation
- Multi-class classification support
- Optimized kernel computations
- Configurable gamma parameter

**Signature**:
```cpp
void svm_cpu(const TensorView& input_tensor, const TensorView& support_vectors,
            const TensorView& dual_coef, const TensorView& intercept,
            TensorView& output_tensor, float gamma = 0.1f);
```

## Performance Characteristics

### Benchmarks (Intel i7, single-threaded)

| Operator | Input Size | Time per Inference |
|----------|------------|-------------------|
| Conv2D | 32→64 channels, 64x64 | ~2.5 ms |
| MaxPool2D | 64 channels, 128x128 | ~150 μs |
| BatchNorm | 64 channels, various spatial | ~50 μs |
| Attention | 512 seq, 768 dim, 12 heads | ~45 ms |

### Memory Usage

- **Conv2D**: O(K_h × K_w × C_in × C_out) for weights
- **MaxPool2D**: O(1) additional memory
- **BatchNorm**: O(C) for statistics
- **Attention**: O(seq_len²) for attention scores

## Integration with MLE Format

All new operators are fully integrated with the MLE file format v2.0:

- Operator types defined in `OpType` enum
- Automatic parameter loading from `.mle` files
- Support for compressed weight storage
- Integrity verification for operator parameters

## Usage Examples

### Building a Simple CNN

```cpp
// Load model
Engine engine(Device::CPU);
engine.load_model("cnn_model.mle");

// Prepare input
auto input = TensorView::create({1, 3, 224, 224}, DType::FP32);
// ... fill input data ...

// Run inference
auto outputs = engine.run({input});
```

### Custom Operator Chain

```cpp
// Manual operator chaining
auto conv_out = TensorView::create({1, 64, 112, 112}, DType::FP32);
ops::conv2d_cpu(*input, *conv_weights, *conv_bias, *conv_out, 2, 2);

auto pool_out = TensorView::create({1, 64, 56, 56}, DType::FP32);
ops::maxpool2d_cpu(*conv_out, *pool_out);

auto norm_out = TensorView::create({1, 64, 56, 56}, DType::FP32);
ops::batchnorm_cpu(*pool_out, *bn_weight, *bn_bias, *bn_mean, *bn_var, *norm_out);
```

## Testing and Validation

Run the comprehensive operator tests:

```bash
cd cpp_core/build
./test_operators_v2
```

This will execute:
- Functional correctness tests
- Performance benchmarks
- Memory usage validation
- Numerical accuracy checks

## Next Steps

The enhanced operators in v2.0 provide the foundation for:
- Modern CNN architectures (ResNet, EfficientNet)
- Transformer models (BERT, GPT)
- Advanced ensemble methods
- High-performance inference pipelines

See the SDK documentation for language-specific bindings and integration examples.