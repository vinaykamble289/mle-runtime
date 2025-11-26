#include "engine.h"
#include <stdexcept>
#include <chrono>

namespace mle {
namespace ops {
    // Forward declarations
    void linear_cpu(const TensorView& input, const TensorView& weight, 
                    const TensorView& bias, TensorView& output);
    void relu_cpu(const TensorView& input, TensorView& output);
    void gelu_cpu(const TensorView& input, TensorView& output);
    void softmax_cpu(const TensorView& input, TensorView& output);
}

Engine::Engine(Device device) : device_(device) {
    if (device == Device::CUDA) {
#ifndef ENABLE_CUDA
        throw std::runtime_error("CUDA support not compiled");
#endif
    }
}

Engine::~Engine() = default;

void Engine::load_model(const std::string& path) {
    loader_ = std::make_unique<ModelLoader>(path);
    
    // Don't pre-load tensors - they will be loaded on demand
    // Some tensors are placeholders (inputs/outputs) without data
    // Only parameter tensors (weights/biases) have actual data in the file
}

std::vector<std::shared_ptr<TensorView>> Engine::run(
    const std::vector<std::shared_ptr<TensorView>>& inputs) {
    
    if (!loader_) {
        throw std::runtime_error("No model loaded");
    }
    
    const auto& graph = loader_->graph();
    
    // Set input tensors
    for (uint32_t i = 0; i < graph.num_inputs && i < inputs.size(); ++i) {
        uint32_t tensor_id = graph.input_ids[i];
        tensor_cache_[tensor_id] = inputs[i];
    }
    
    // Execute nodes in order
    const GraphNode* nodes = reinterpret_cast<const GraphNode*>(
        reinterpret_cast<const TensorDesc*>(&graph + 1) + graph.num_tensors);
    
    for (uint32_t i = 0; i < graph.num_nodes; ++i) {
        execute_node(nodes[i]);
    }
    
    // Collect outputs
    std::vector<std::shared_ptr<TensorView>> outputs;
    for (uint32_t i = 0; i < graph.num_outputs; ++i) {
        uint32_t tensor_id = graph.output_ids[i];
        outputs.push_back(tensor_cache_[tensor_id]);
    }
    
    return outputs;
}

std::shared_ptr<TensorView> Engine::get_or_load_tensor(uint32_t tensor_id) {
    // Check cache first
    auto it = tensor_cache_.find(tensor_id);
    if (it != tensor_cache_.end()) {
        return it->second;
    }
    
    // Load from file (for parameter tensors)
    auto tensor = std::make_shared<TensorView>(loader_->get_tensor(tensor_id));
    tensor_cache_[tensor_id] = tensor;
    return tensor;
}

void Engine::execute_node(const GraphNode& node) {
    switch (node.op_type) {
        case OpType::LINEAR: {
            // LINEAR: input, weight, bias -> output
            auto input = get_or_load_tensor(node.input_ids[0]);
            auto weight = get_or_load_tensor(node.param_ids[0]);
            auto bias = get_or_load_tensor(node.param_ids[1]);
            
            // Allocate output
            const auto& in_shape = input->shape();
            std::vector<uint32_t> out_shape = {in_shape[0], weight->shape()[0]};
            auto output = TensorView::create(out_shape, DType::FP32);
            
            if (device_ == Device::CPU) {
                ops::linear_cpu(*input, *weight, *bias, *output);
            } else {
                // TODO: CUDA path
                throw std::runtime_error("CUDA not implemented yet");
            }
            
            tensor_cache_[node.output_ids[0]] = output;
            break;
        }
        
        case OpType::RELU: {
            auto input = get_or_load_tensor(node.input_ids[0]);
            auto output = TensorView::create(input->shape(), input->dtype());
            
            if (device_ == Device::CPU) {
                ops::relu_cpu(*input, *output);
            }
            
            tensor_cache_[node.output_ids[0]] = output;
            break;
        }
        
        case OpType::GELU: {
            auto input = get_or_load_tensor(node.input_ids[0]);
            auto output = TensorView::create(input->shape(), input->dtype());
            
            if (device_ == Device::CPU) {
                ops::gelu_cpu(*input, *output);
            }
            
            tensor_cache_[node.output_ids[0]] = output;
            break;
        }
        
        case OpType::SOFTMAX: {
            auto input = get_or_load_tensor(node.input_ids[0]);
            auto output = TensorView::create(input->shape(), input->dtype());
            
            if (device_ == Device::CPU) {
                ops::softmax_cpu(*input, *output);
            }
            
            tensor_cache_[node.output_ids[0]] = output;
            break;
        }
        
        default:
            char error_msg[128];
            snprintf(error_msg, sizeof(error_msg), "Unsupported op type: %d", static_cast<int>(node.op_type));
            throw std::runtime_error(error_msg);
    }
}

std::shared_ptr<TensorView> Engine::allocate_tensor(const TensorDesc& desc) {
    std::vector<uint32_t> shape(desc.shape, desc.shape + desc.ndim);
    return TensorView::create(shape, desc.dtype);
}

} // namespace mle
