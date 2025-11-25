#include "tensor_view.h"
#include "mle_format.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace mle {
namespace ops {

// CPU MatMul: C = A @ B
void matmul_cpu(const TensorView& A, const TensorView& B, TensorView& C) {
    if (A.dtype() != DType::FP32 || B.dtype() != DType::FP32 || C.dtype() != DType::FP32) {
        throw std::runtime_error("Only FP32 supported for CPU matmul");
    }
    
    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();
    const auto& c_shape = C.shape();
    
    // Support 2D: [M, K] @ [K, N] = [M, N]
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("Only 2D matmul supported");
    }
    
    uint32_t M = a_shape[0];
    uint32_t K = a_shape[1];
    uint32_t N = b_shape[1];
    
    if (b_shape[0] != K || c_shape[0] != M || c_shape[1] != N) {
        throw std::runtime_error("Shape mismatch in matmul");
    }
    
    const float* a = static_cast<const float*>(A.data());
    const float* b = static_cast<const float*>(B.data());
    float* c = static_cast<float*>(C.data());
    
    // Naive implementation - can be optimized with BLAS
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// Add bias: C = A + b (broadcast)
void add_bias_cpu(const TensorView& A, const TensorView& bias, TensorView& C) {
    if (A.dtype() != DType::FP32) {
        throw std::runtime_error("Only FP32 supported");
    }
    
    const auto& a_shape = A.shape();
    const auto& b_shape = bias.shape();
    
    // Assume bias is 1D and broadcasts to last dimension
    if (b_shape.size() != 1) {
        throw std::runtime_error("Bias must be 1D");
    }
    
    uint32_t N = b_shape[0];
    size_t numel = A.numel();
    
    const float* a = static_cast<const float*>(A.data());
    const float* b = static_cast<const float*>(bias.data());
    float* c = static_cast<float*>(C.data());
    
    for (size_t i = 0; i < numel; ++i) {
        c[i] = a[i] + b[i % N];
    }
}

// ReLU: y = max(0, x)
void relu_cpu(const TensorView& input, TensorView& output) {
    if (input.dtype() != DType::FP32) {
        throw std::runtime_error("Only FP32 supported");
    }
    
    size_t numel = input.numel();
    const float* in = static_cast<const float*>(input.data());
    float* out = static_cast<float*>(output.data());
    
    for (size_t i = 0; i < numel; ++i) {
        out[i] = std::max(0.0f, in[i]);
    }
}

// GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_cpu(const TensorView& input, TensorView& output) {
    if (input.dtype() != DType::FP32) {
        throw std::runtime_error("Only FP32 supported");
    }
    
    size_t numel = input.numel();
    const float* in = static_cast<const float*>(input.data());
    float* out = static_cast<float*>(output.data());
    
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    
    for (size_t i = 0; i < numel; ++i) {
        float x = in[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

// Linear layer: y = xW^T + b
void linear_cpu(const TensorView& input, const TensorView& weight, 
                const TensorView& bias, TensorView& output) {
    // Weight is [out_features, in_features], need to transpose
    // For simplicity, assume input is [batch, in_features]
    
    const auto& in_shape = input.shape();
    const auto& w_shape = weight.shape();
    
    if (in_shape.size() != 2 || w_shape.size() != 2) {
        throw std::runtime_error("Linear expects 2D tensors");
    }
    
    uint32_t batch = in_shape[0];
    uint32_t in_features = in_shape[1];
    uint32_t out_features = w_shape[0];
    
    if (w_shape[1] != in_features) {
        throw std::runtime_error("Shape mismatch in linear");
    }
    
    const float* x = static_cast<const float*>(input.data());
    const float* w = static_cast<const float*>(weight.data());
    const float* b = static_cast<const float*>(bias.data());
    float* y = static_cast<float*>(output.data());
    
    // y = x @ W^T
    for (uint32_t i = 0; i < batch; ++i) {
        for (uint32_t j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < in_features; ++k) {
                sum += x[i * in_features + k] * w[j * in_features + k];
            }
            y[i * out_features + j] = sum + b[j];
        }
    }
}

} // namespace ops
} // namespace mle
