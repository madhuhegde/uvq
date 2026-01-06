# UVQ TFLite Models - Usage Guide

## Model Selection

### FLOAT32 vs INT8

Choose the appropriate model based on your requirements:

| Criterion | FLOAT32 | INT8 |
|-----------|---------|------|
| **Accuracy** | Maximum (correlation: 1.000000) | Good (correlation: 0.979218) |
| **File Size** | 14.53 MB | 4.25 MB (70% smaller) |
| **Inference Speed** | Standard | Faster (INT8 operations) |
| **Use Case** | Maximum accuracy required | Size/speed critical |

**Recommendation**:
- Use **FLOAT32** (`distortion_net.tflite`) for development and validation
- Use **INT8** (`distortion_net_int8.tflite`) for production deployment

## Python Usage

### Basic Example

```python
import numpy as np
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="distortion_net.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input: 9 patches of 360×640 RGB (NHWC format)
video_patches = np.random.randn(9, 360, 640, 3).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], video_patches)
interpreter.invoke()
features = interpreter.get_tensor(output_details[0]['index'])

# Output shape: [1, 24, 24, 128]
print(f"Output shape: {features.shape}")
```

### Complete Pipeline

```python
import numpy as np
import tensorflow as tf
from pathlib import Path

class DistortionNetInference:
    """Wrapper for DistortionNet TFLite inference."""
    
    def __init__(self, model_path, use_int8=False):
        """
        Args:
            model_path: Path to TFLite model directory
            use_int8: If True, use INT8 model; otherwise use FLOAT32
        """
        model_name = "distortion_net_int8.tflite" if use_int8 else "distortion_net.tflite"
        full_path = Path(model_path) / model_name
        
        self.interpreter = tf.lite.Interpreter(model_path=str(full_path))
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Loaded model: {model_name}")
        print(f"Input shape:  {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def preprocess_patches(self, patches):
        """
        Preprocess video patches for inference.
        
        Args:
            patches: numpy array of shape [9, 360, 640, 3]
                     Values should be in a reasonable range (e.g., [-1, 1] or [0, 255])
        
        Returns:
            Preprocessed patches ready for inference
        """
        # Ensure correct shape
        assert patches.shape == (9, 360, 640, 3), f"Expected shape (9, 360, 640, 3), got {patches.shape}"
        
        # Convert to float32
        patches = patches.astype(np.float32)
        
        # Normalize if needed (depends on your input data)
        # Example: if input is [0, 255], normalize to [-1, 1]
        # patches = (patches / 127.5) - 1.0
        
        return patches
    
    def run_inference(self, patches):
        """
        Run inference on video patches.
        
        Args:
            patches: numpy array of shape [9, 360, 640, 3]
        
        Returns:
            features: numpy array of shape [1, 24, 24, 128]
        """
        # Preprocess
        patches = self.preprocess_patches(patches)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], patches)
        self.interpreter.invoke()
        features = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return features

# Example usage
if __name__ == '__main__':
    # Initialize inference
    model_path = "~/work/UVQ/uvq/models/tflite_models/uvq1.5"
    distortion_net = DistortionNetInference(model_path, use_int8=False)
    
    # Create sample input (9 patches)
    video_patches = np.random.randn(9, 360, 640, 3).astype(np.float32)
    
    # Run inference
    features = distortion_net.run_inference(video_patches)
    
    print(f"Input shape:  {video_patches.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{features.min():.4f}, {features.max():.4f}]")
```

## Input Format

### DistortionNet

**Shape**: `[9, 360, 640, 3]`

- **Batch dimension**: 9 patches (3×3 grid from a single frame)
- **Height**: 360 pixels
- **Width**: 640 pixels
- **Channels**: 3 (RGB)
- **Format**: NHWC (TensorFlow standard)

**Patch Layout**:
```
Frame divided into 3×3 grid:
┌─────┬─────┬─────┐
│  0  │  1  │  2  │
├─────┼─────┼─────┤
│  3  │  4  │  5  │
├─────┼─────┼─────┤
│  6  │  7  │  8  │
└─────┴─────┴─────┘

Each patch: 360×640×3
```

**Value Range**: Typically normalized to `[-1, 1]` or `[0, 1]` depending on preprocessing

## Output Format

### DistortionNet

**Shape**: `[1, 24, 24, 128]`

- **Batch dimension**: 1 (aggregated from 9 patches)
- **Height**: 24 (3 patches × 8 pixels each)
- **Width**: 24 (3 patches × 8 pixels each)
- **Channels**: 128 (feature channels)
- **Format**: NHWC (TensorFlow standard)

## Performance Tips

### 1. Batch Processing
```python
# Process multiple frames efficiently
frames = [frame1, frame2, frame3, ...]  # List of frames

for frame in frames:
    patches = extract_patches(frame)  # Extract 9 patches
    features = distortion_net.run_inference(patches)
    # Process features...
```

### 2. Memory Management
```python
# For large-scale processing, clear memory periodically
import gc

for i, frame in enumerate(frames):
    features = distortion_net.run_inference(extract_patches(frame))
    # Process features...
    
    if i % 100 == 0:
        gc.collect()  # Clear memory every 100 frames
```

### 3. Use INT8 for Production
```python
# INT8 model is 70% smaller and faster
distortion_net = DistortionNetInference(model_path, use_int8=True)
```

## Model Locations

All models are located in:
```
~/work/UVQ/uvq/models/tflite_models/uvq1.5/
├── distortion_net.tflite          # FLOAT32 (14.53 MB)
├── distortion_net_int8.tflite     # INT8 (4.25 MB)
├── distortion_net_6d.tflite       # Old 6D version (reference only)
└── distortion_net_int8_6d.tflite  # Old INT8 6D version (reference only)
```

**Note**: Only use `distortion_net.tflite` and `distortion_net_int8.tflite` for BSTM hardware. The `*_6d.tflite` models are kept for reference only.

## See Also

- [Overview](./overview.md) - General information about UVQ TFLite
- [Conversion](./conversion.md) - How to convert models from PyTorch
- [Verification](./verification.md) - Testing and validation
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
