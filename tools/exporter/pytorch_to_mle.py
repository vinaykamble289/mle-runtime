#!/usr/bin/env python3
"""
PyTorch to .mle exporter
Converts PyTorch models to custom .mle format
"""

import struct
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch
import torch.nn as nn
import numpy as np

# Constants from mle_format.h
MLE_MAGIC = 0x00454C4D
MLE_VERSION = 1

DTYPE_MAP = {
    torch.float32: 0,  # FP32
    torch.float16: 1,  # FP16
    torch.int8: 2,     # INT8
    torch.int32: 3,    # INT32
}

OP_TYPE_MAP = {
    'Linear': 1,
    'ReLU': 2,
    'GELU': 3,
    'Softmax': 4,
    'LayerNorm': 5,
}


class MLEExporter:
    def __init__(self):
        self.tensors: List[Dict] = []
        self.nodes: List[Dict] = []
        self.weights_data = bytearray()
        self.tensor_id_counter = 0
        
    def add_tensor(self, data: torch.Tensor, name: str = "") -> int:
        """Add tensor and return its ID"""
        tensor_id = self.tensor_id_counter
        self.tensor_id_counter += 1
        
        # Convert to numpy and get bytes
        np_data = data.detach().cpu().numpy()
        offset = len(self.weights_data)
        tensor_bytes = np_data.tobytes()
        self.weights_data.extend(tensor_bytes)
        
        # Create tensor descriptor
        shape = list(np_data.shape)
        while len(shape) < 8:
            shape.append(0)
        
        tensor_desc = {
            'id': tensor_id,
            'name': name,
            'offset': offset,
            'size': len(tensor_bytes),
            'ndim': len(np_data.shape),
            'shape': shape[:8],
            'dtype': DTYPE_MAP.get(data.dtype, 0),
        }
        
        self.tensors.append(tensor_desc)
        return tensor_id
    
    def add_node(self, op_type: str, inputs: List[int], outputs: List[int], 
                 params: List[int], attrs: Dict = None):
        """Add graph node"""
        node = {
            'op_type': OP_TYPE_MAP.get(op_type, 0),
            'num_inputs': len(inputs),
            'num_outputs': len(outputs),
            'num_params': len(params),
            'input_ids': inputs + [0] * (16 - len(inputs)),
            'output_ids': outputs + [0] * (16 - len(outputs)),
            'param_ids': params + [0] * (16 - len(params)),
            'attrs': attrs or {},
        }
        self.nodes.append(node)
    
    def export_mlp(self, model: nn.Module, input_shape: tuple, output_path: str):
        """Export simple MLP model"""
        model.eval()
        
        # Trace model to get graph
        dummy_input = torch.randn(*input_shape)
        
        # For simple MLP, manually extract layers
        graph_inputs = []
        graph_outputs = []
        
        # Add input tensor (placeholder)
        input_id = self.add_tensor(dummy_input, "input")
        graph_inputs.append(input_id)
        
        current_tensor_id = input_id
        
        # Extract layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Add weight and bias
                weight_id = self.add_tensor(module.weight, f"{name}.weight")
                bias_id = self.add_tensor(module.bias, f"{name}.bias")
                
                # Create output tensor placeholder
                output_id = self.tensor_id_counter
                self.tensor_id_counter += 1
                
                # Add LINEAR node
                self.add_node('Linear', 
                            inputs=[current_tensor_id],
                            outputs=[output_id],
                            params=[weight_id, bias_id])
                
                current_tensor_id = output_id
                
            elif isinstance(module, nn.ReLU):
                output_id = self.tensor_id_counter
                self.tensor_id_counter += 1
                
                self.add_node('ReLU',
                            inputs=[current_tensor_id],
                            outputs=[output_id],
                            params=[])
                
                current_tensor_id = output_id
            
            elif isinstance(module, nn.GELU):
                output_id = self.tensor_id_counter
                self.tensor_id_counter += 1
                
                self.add_node('GELU',
                            inputs=[current_tensor_id],
                            outputs=[output_id],
                            params=[])
                
                current_tensor_id = output_id
        
        graph_outputs.append(current_tensor_id)
        
        # Build metadata
        metadata = {
            'model_name': model.__class__.__name__,
            'framework': 'pytorch',
            'input_shapes': [list(input_shape)],
            'output_shapes': [[input_shape[0], list(model.parameters())[-1].shape[0]]],
        }
        metadata_json = json.dumps(metadata).encode('utf-8')
        
        # Build graph IR
        graph_ir = self._build_graph_ir(graph_inputs, graph_outputs)
        
        # Write .mle file
        self._write_mle(output_path, metadata_json, graph_ir)
        
        print(f"Exported model to {output_path}")
        print(f"  Tensors: {len(self.tensors)}")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Weights size: {len(self.weights_data)} bytes")
    
    def _build_graph_ir(self, graph_inputs: List[int], graph_outputs: List[int]) -> bytes:
        """Build graph IR binary"""
        ir = bytearray()
        
        # GraphIR header
        input_ids = graph_inputs + [0] * (16 - len(graph_inputs))
        output_ids = graph_outputs + [0] * (16 - len(graph_outputs))
        
        ir.extend(struct.pack('I', len(self.nodes)))  # num_nodes
        ir.extend(struct.pack('I', len(self.tensors)))  # num_tensors
        ir.extend(struct.pack('I', len(graph_inputs)))  # num_inputs
        ir.extend(struct.pack('I', len(graph_outputs)))  # num_outputs
        ir.extend(struct.pack('16I', *input_ids))
        ir.extend(struct.pack('16I', *output_ids))
        
        # TensorDesc array
        for tensor in self.tensors:
            ir.extend(struct.pack('Q', tensor['offset']))  # offset
            ir.extend(struct.pack('Q', tensor['size']))  # size
            ir.extend(struct.pack('I', tensor['ndim']))  # ndim
            ir.extend(struct.pack('8I', *tensor['shape']))  # shape
            ir.extend(struct.pack('B', tensor['dtype']))  # dtype
            ir.extend(struct.pack('3B', 0, 0, 0))  # reserved
        
        # GraphNode array
        for node in self.nodes:
            ir.extend(struct.pack('H', node['op_type']))
            ir.extend(struct.pack('H', node['num_inputs']))
            ir.extend(struct.pack('H', node['num_outputs']))
            ir.extend(struct.pack('H', node['num_params']))
            ir.extend(struct.pack('16I', *node['input_ids']))
            ir.extend(struct.pack('16I', *node['output_ids']))
            ir.extend(struct.pack('16I', *node['param_ids']))
            ir.extend(struct.pack('I', 0))  # attr_offset (unused for now)
            ir.extend(struct.pack('I', 0))  # attr_size
        
        return bytes(ir)
    
    def _write_mle(self, path: str, metadata: bytes, graph_ir: bytes):
        """Write .mle file"""
        with open(path, 'wb') as f:
            # Calculate offsets
            header_size = 64
            metadata_offset = header_size
            metadata_size = len(metadata)
            graph_offset = metadata_offset + metadata_size
            graph_size = len(graph_ir)
            weights_offset = graph_offset + graph_size
            weights_size = len(self.weights_data)
            signature_offset = 0  # No signature for now
            
            # Write header
            header = struct.pack(
                'II QQ QQ QQ Q 16s',
                MLE_MAGIC,
                MLE_VERSION,
                metadata_offset,
                metadata_size,
                graph_offset,
                graph_size,
                weights_offset,
                weights_size,
                signature_offset,
                b'\x00' * 16  # reserved
            )
            f.write(header)
            
            # Write metadata
            f.write(metadata)
            
            # Write graph IR
            f.write(graph_ir)
            
            # Write weights
            f.write(self.weights_data)


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to .mle')
    parser.add_argument('--model', type=str, help='Path to PyTorch model (.pth)')
    parser.add_argument('--out', type=str, required=True, help='Output .mle path')
    parser.add_argument('--input-shape', type=str, default='1,128', 
                       help='Input shape (comma-separated)')
    args = parser.parse_args()
    
    # Create simple test model if no model provided
    if not args.model:
        print("No model provided, creating test MLP...")
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    else:
        model = torch.load(args.model)
    
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    exporter = MLEExporter()
    exporter.export_mlp(model, input_shape, args.out)


if __name__ == '__main__':
    main()
