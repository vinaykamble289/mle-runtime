"""
MLE Engine wrapper - imports from bindings/python
"""

import sys
import os
import importlib.util

# Add bindings/python to path
_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_bindings_path = os.path.join(_root_dir, 'bindings', 'python')

# Import native module directly from file to avoid circular import
_pyd_file = None
for ext in ['.pyd', '.so', '.dylib']:
    potential_file = os.path.join(_bindings_path, f'mle_runtime{ext}')
    if os.path.exists(potential_file):
        _pyd_file = potential_file
        break

if not _pyd_file:
    # Try in build directory
    import glob
    build_files = glob.glob(os.path.join(_bindings_path, 'build', '**', 'mle_runtime*.pyd'), recursive=True)
    build_files.extend(glob.glob(os.path.join(_bindings_path, 'build', '**', 'mle_runtime*.so'), recursive=True))
    if build_files:
        _pyd_file = build_files[0]

if not _pyd_file:
    raise ImportError(
        f"Failed to find MLE native bindings in {_bindings_path}. "
        "Make sure to build the bindings first: cd bindings/python && python setup.py build_ext --inplace"
    )

# Load the module directly - use correct module name
spec = importlib.util.spec_from_file_location("mle_runtime", _pyd_file)
_native = importlib.util.module_from_spec(spec)
sys.modules['_mle_native_core'] = _native  # Store with different name to avoid conflict
spec.loader.exec_module(_native)

# Export classes
Engine = _native.Engine
GraphExecutor = _native.GraphExecutor
ModelLoader = _native.ModelLoader
Device = _native.Device
DType = _native.DType

__all__ = ['Engine', 'GraphExecutor', 'ModelLoader', 'Device', 'DType']
