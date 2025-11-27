"""Debug MLE file contents"""
import struct
import numpy as np

def read_mle_file(path):
    with open(path, 'rb') as f:
        # Read header (72 bytes total)
        header = f.read(72)
        parts = struct.unpack('II QQQQQQQ 8s', header)
        
        magic = parts[0]
        version = parts[1]
        metadata_offset = parts[2]
        metadata_size = parts[3]
        graph_offset = parts[4]
        graph_ir_size = parts[5]
        weights_offset = parts[6]
        weights_size = parts[7]
        
        print(f"Magic: 0x{magic:08X}")
        print(f"Version: {version}")
        print(f"Metadata size: {metadata_size}")
        print(f"Graph IR size: {graph_ir_size}")
        print(f"Weights size: {weights_size}")
        print()
        
        # Read metadata
        metadata = f.read(metadata_size).decode('utf-8')
        print("Metadata:")
        print(metadata)
        print()
        
        # Read graph IR
        graph_ir = f.read(graph_ir_size)
        
        # Parse graph IR header (GraphIR struct)
        offset = 0
        
        # GraphIR header
        num_nodes = struct.unpack_from('<I', graph_ir, offset)[0]
        offset += 4
        num_tensors = struct.unpack_from('<I', graph_ir, offset)[0]
        offset += 4
        num_inputs = struct.unpack_from('<I', graph_ir, offset)[0]
        offset += 4
        num_outputs = struct.unpack_from('<I', graph_ir, offset)[0]
        offset += 4
        input_ids = list(struct.unpack_from('<16I', graph_ir, offset))
        offset += 64
        output_ids = list(struct.unpack_from('<16I', graph_ir, offset))
        offset += 64
        
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of tensors: {num_tensors}")
        print(f"Number of inputs: {num_inputs}")
        print(f"Number of outputs: {num_outputs}")
        print(f"Input IDs: {input_ids[:num_inputs]}")
        print(f"Output IDs: {output_ids[:num_outputs]}")
        print()
        
        # Read tensors (TensorDesc array)
        for i in range(num_tensors):
            tensor_offset = struct.unpack_from('<Q', graph_ir, offset)[0]
            offset += 8
            
            size = struct.unpack_from('<Q', graph_ir, offset)[0]
            offset += 8
            
            ndim = struct.unpack_from('<I', graph_ir, offset)[0]
            offset += 4
            
            shape = list(struct.unpack_from('<8I', graph_ir, offset))
            offset += 32
            
            dtype = struct.unpack_from('<B', graph_ir, offset)[0]
            offset += 1
            
            reserved = struct.unpack_from('<3B', graph_ir, offset)
            offset += 3
            
            print(f"Tensor {i}:")
            print(f"  offset={tensor_offset}, size={size}")
            print(f"  ndim={ndim}, shape={shape[:ndim]}, dtype={dtype}")
            print()
        
        # Read nodes (GraphNode array)
        for i in range(num_nodes):
            op_type = struct.unpack_from('<H', graph_ir, offset)[0]
            offset += 2
            
            node_num_inputs = struct.unpack_from('<H', graph_ir, offset)[0]
            offset += 2
            
            node_num_outputs = struct.unpack_from('<H', graph_ir, offset)[0]
            offset += 2
            
            node_num_params = struct.unpack_from('<H', graph_ir, offset)[0]
            offset += 2
            
            node_input_ids = list(struct.unpack_from('<16I', graph_ir, offset))
            offset += 64
            
            node_output_ids = list(struct.unpack_from('<16I', graph_ir, offset))
            offset += 64
            
            node_param_ids = list(struct.unpack_from('<16I', graph_ir, offset))
            offset += 64
            
            attr_offset = struct.unpack_from('<I', graph_ir, offset)[0]
            offset += 4
            
            attr_size = struct.unpack_from('<I', graph_ir, offset)[0]
            offset += 4
            
            print(f"Node {i}: op_type={op_type}")
            print(f"  inputs={node_input_ids[:node_num_inputs]}")
            print(f"  outputs={node_output_ids[:node_num_outputs]}")
            print(f"  params={node_param_ids[:node_num_params]}")
            print()

if __name__ == '__main__':
    print("="*60)
    print("Logistic Regression Model")
    print("="*60)
    read_mle_file('test_lr.mle')
