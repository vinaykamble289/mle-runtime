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
    
    // Pre-load all parameter tensors into cache
    const auto& graph = loader_->graph();
    for (uint32_t i = 0; i < graph.num_tensors; ++i) {
        auto tensor = std::make_shared<TensorView>(loader_->get_tensor(i));
        tensor_cache_[i] = tensor;
    }
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

void Engine::execute_node(const GraphNode& node) {
    switch (node.op_type) {
        case OpType::LINEAR: {
            // LINEAR: input, weight, bias -> output
            auto input = tensor_cache_[node.input_ids[0]];
            auto weight = tensor_cache_[node.param_ids[0]];
            auto bias = tensor_cache_[node.param_ids[1]];
            
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
            auto input = tensor_cache_[node.input_ids[0]];
            auto output = TensorView::create(input->shape(), input->dtype());
            
            if (device_ == Device::CPU) {
                ops::relu_cpu(*input, *output);
            }
            
            tensor_cache_[node.output_ids[0]] = output;
            break;
        }
        
        case OpType::GELU: {
            auto input = tensor_cache_[node.input_ids[0]];
            auto output = TensorView::create(input->shape(), input->dtype());
            
            if (device_ == Device::CPU) {
                ops::gelu_cpu(*input, *output);
            }
            
            tensor_cache_[node.output_ids[0]] = output;
            break;
        }
        
        default:
            throw std::runtime_error("Unsupported op type");
    }
}

std::shared_ptr<TensorView> Engine::allocate_tensor(const TensorDesc& desc) {
    std::vector<uint32_t> shape(desc.shape, desc.shape + desc.ndim);
    return TensorView::create(shape, desc.dtype);
}

} // namespace mle
