"""
MLE Runtime - Python Client Library
Fast ML inference runtime with memory-mapped loading
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Union
import numpy as np

__version__ = "1.0.0"

class Device(Enum):
    """Device types for inference"""
    CPU = "cpu"
    CUDA = "cuda"

class DType(Enum):
    """Data types supported by MLE"""
    FP32 = 0
    FP16 = 1
    INT8 = 2
    INT32 = 3

class ModelMetadata:
    """Model metadata from .mle file"""
    
    def __init__(self, data: Dict[str, Any]):
        self.model_name = data.get("model_name", "")
        self.framework = data.get("framework", "")
        self.version = data.get("version", {})
        self.input_shapes = data.get("input_shapes", [])
        self.output_shapes = data.get("output_shapes", [])
        self.export_timestamp = data.get("export_timestamp")
    
    def __repr__(self):
        return (f"ModelMetadata(name='{self.model_name}', "
                f"framework='{self.framework}', "
                f"version={self.version})")

class MLEEngine:
    """
    MLE inference engine
    
    Fast ML inference with memory-mapped loading.
    10-100x faster than joblib/pickle.
    
    Example:
        >>> engine = MLEEngine(Device.CPU)
        >>> engine.load_model("model.mle")
        >>> outputs = engine.run([input_array])
        >>> print(outputs[0])
    """
    
    def __init__(self, device: Device = Device.CPU):
        """
        Create a new MLE engine
        
        Args:
            device: Device to run inference on (CPU or CUDA)
        """
        try:
            from . import _mle_core
            self._engine = _mle_core.Engine(device.value)
        except ImportError as e:
            raise ImportError(
                "Failed to import MLE native module. "
                "Make sure the C++ core is built and installed. "
                f"Error: {e}"
            )
        
        self._device = device
        self._metadata: Optional[ModelMetadata] = None
    
    def load_model(self, path: str) -> None:
        """
        Load a model from .mle file
        
        Args:
            path: Path to .mle model file
            
        Raises:
            RuntimeError: If loading fails
        """
        metadata_json = self._engine.load_model(path)
        import json
        self._metadata = ModelMetadata(json.loads(metadata_json))
    
    def run(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Run inference on input tensors
        
        Args:
            inputs: List of input tensors as numpy arrays
            
        Returns:
            List of output tensors as numpy arrays
            
        Raises:
            RuntimeError: If inference fails
        """
        if self._metadata is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Convert inputs to float32 if needed
        inputs_f32 = [
            inp.astype(np.float32) if inp.dtype != np.float32 else inp
            for inp in inputs
        ]
        
        return self._engine.run(inputs_f32)
    
    @property
    def metadata(self) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self._metadata
    
    @property
    def device(self) -> Device:
        """Get current device"""
        return self._device
    
    def peak_memory_usage(self) -> int:
        """
        Get peak memory usage in bytes
        
        Returns:
            Peak memory usage
        """
        return self._engine.peak_memory_usage()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Cleanup handled by C++ destructor
        pass

class MLEUtils:
    """Utility functions for MLE"""
    
    @staticmethod
    def inspect_model(path: str) -> ModelMetadata:
        """
        Inspect .mle file and return metadata
        
        Args:
            path: Path to .mle file
            
        Returns:
            Model metadata
        """
        from . import _mle_core
        import json
        metadata_json = _mle_core.inspect_model(path)
        return ModelMetadata(json.loads(metadata_json))
    
    @staticmethod
    def verify_model(path: str, public_key: str) -> bool:
        """
        Verify model signature
        
        Args:
            path: Path to .mle file
            public_key: ED25519 public key (hex string)
            
        Returns:
            True if signature is valid
        """
        from . import _mle_core
        return _mle_core.verify_model(path, public_key)

__all__ = [
    "Device",
    "DType",
    "ModelMetadata",
    "MLEEngine",
    "MLEUtils",
]
