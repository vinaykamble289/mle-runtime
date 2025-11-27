import struct

# Read the .mle file and check the op_type
with open('test_pca.mle', 'rb') as f:
    # Read header (64 bytes)
    header = f.read(64)
    magic, version, metadata_offset, metadata_size, graph_offset, graph_size = struct.unpack('<IIQQQQ', header[:40])
    
    print(f"Magic: {hex(magic)}")
    print(f"Version: {version}")
    print(f"Graph offset: {graph_offset}")
    print(f"Graph size: {graph_size}")
    
    # Seek to graph
    f.seek(graph_offset)
    
    # Read GraphIR header
    graph_header = f.read(4 * 4 + 16 * 4 * 2)  # num_nodes, num_tensors, num_inputs, num_outputs + input/output IDs
    num_nodes, num_tensors, num_inputs, num_outputs = struct.unpack('<IIII', graph_header[:16])
    
    print(f"\nGraph info:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Tensors: {num_tensors}")
    
    # Skip tensor descriptors
    tensor_desc_size = 8 + 8 + 4 + 4*8 + 1 + 3  # offset, size, ndim, shape[8], dtype, reserved[3]
    f.seek(graph_offset + len(graph_header) + num_tensors * tensor_desc_size)
    
    # Read first node
    node_data = f.read(2 + 2 + 2 + 2 + 16*4 + 16*4 + 16*4 + 4 + 4)
    op_type = struct.unpack('<H', node_data[:2])[0]
    
    print(f"\nFirst node op_type: {op_type}")
