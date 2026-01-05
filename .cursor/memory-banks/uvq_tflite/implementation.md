# UVQ 1.5 TFLite Implementation

This directory contains both PyTorch and TFLite implementations of UVQ 1.5 (Universal Video Quality) assessment model.

## Overview

UVQ 1.5 consists of three neural networks:
1. **ContentNet** - Extracts semantic content features from video frames
2. **DistortionNet** - Detects visual distortions using patch-based processing
3. **AggregationNet** - Combines features to produce quality scores [1-5]

## Implementations

### PyTorch Implementation
- **File:** `utils/uvq1p5.py`
- **Models:** `checkpoints/*.pth` (PyTorch state dicts)
- **Usage:** Standard PyTorch inference with GPU support

### TFLite Implementation
- **File:** `utils/uvq1p5_tflite.py`
- **Models:** `../models/tflite_models/uvq1.5/*.tflite`
- **Usage:** TensorFlow Lite inference (CPU optimized)

## Installation

### PyTorch Version
```bash
pip install torch torchvision numpy
```

### TFLite Version
```bash
pip install tensorflow==2.18.0 opencv-python numpy
```

## Usage

### PyTorch Inference

```python
from uvq1p5_pytorch.utils import uvq1p5

# Create model
model = uvq1p5.UVQ1p5()

# Run inference
results = model.infer(
    video_filename="video.mp4",
    video_length=20,
    transpose=False,
    fps=1,
    orig_fps=30.0
)

print(f"Quality Score: {results['uvq1p5_score']:.4f}")
```

### TFLite Inference

```python
from uvq1p5_pytorch.utils import uvq1p5_tflite

# Create model
model = uvq1p5_tflite.UVQ1p5TFLite()

# Run inference
results = model.infer(
    video_filename="video.mp4",
    video_length=20,
    transpose=False,
    fps=1,
    orig_fps=30.0
)

print(f"Quality Score: {results['uvq1p5_score']:.4f}")
```

## Testing

Compare PyTorch and TFLite implementations:

```bash
cd /home/madhuhegde/work/UVQ/uvq
source ~/work/UVQ/uvq_env/bin/activate

# Test TFLite only
python test_tflite_inference.py video.mp4 --tflite

# Test PyTorch only
python test_tflite_inference.py video.mp4 --pytorch

# Compare both
python test_tflite_inference.py video.mp4 --compare
```

## Model Files

### PyTorch Models
Location: `checkpoints/`
- `content_net.pth` - 15 MB
- `distortion_net.pth` - 15 MB
- `aggregation_net.pth` - 293 KB

### TFLite Models
Location: `../models/tflite_models/uvq1.5/`
- `content_net.tflite` - 14.55 MB
- `distortion_net.tflite` - 14.53 MB
- `aggregation_net.tflite` - 0.30 MB

## Model Specifications

### ContentNet
- **Input:** (1, 256, 256, 3) - RGB frames resized to 256×256 in [B, H, W, C] format
- **Output:** (1, 8, 8, 128) - Content feature maps
- **Purpose:** Extract semantic content features

### DistortionNet
- **Input:** (9, 360, 640, 3) - 9 patches per frame (3×3 grid from 1080p) in [B, H, W, C] format
- **Output:** (1, 24, 24, 128) - Distortion feature maps
- **Purpose:** Detect visual distortions using patch-based processing

### AggregationNet
- **Inputs:** 
  - Content features: (1, 8, 8, 128)
  - Distortion features: (1, 24, 24, 128)
- **Output:** (1, 1) - Quality score in range [1, 5]
- **Purpose:** Combine features to produce final quality assessment

## Preprocessing

Video frames must be preprocessed before inference:

### For ContentNet (256×256)
```python
import cv2
import numpy as np

# Resize to 256×256
frame_256 = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)

# Normalize to [-1, 1]
frame_normalized = (frame_256 / 255.0 - 0.5) * 2

# Add batch dimension (already in H, W, C format - no transpose needed)
frame_input = np.expand_dims(frame_normalized, axis=0)
# Shape: (1, 256, 256, 3) in [B, H, W, C] format
```

### For DistortionNet (9 patches from 1080p)
```python
import numpy as np

# Split 1080p frame into 3×3 patches
patches = []
for i in range(3):
    for j in range(3):
        patch = frame[i*360:(i+1)*360, j*640:(j+1)*640]
        # Normalize to [-1, 1]
        patch_normalized = (patch / 255.0 - 0.5) * 2
        # Already in H, W, C format - no transpose needed
        patches.append(patch_normalized)

# Stack patches: (9, 360, 640, 3) in [B, H, W, C] format
patches_input = np.stack(patches, axis=0)
```

## Performance Comparison

### Test Results (Gaming_360P_local.mp4, 20 seconds, 30 fps)

| Implementation | Score | Inference Time | Memory Usage |
|----------------|-------|----------------|--------------|
| **PyTorch** | 2.9702 | ~5-10s | ~2-3 GB |
| **TFLite** | 3.0319 | ~8-15s | ~500 MB |

**Difference:** 0.0617 (2.1%)

### Per-Frame Differences
- Mean: 0.1772
- Max: 0.5232
- Min: 0.0753

**Note:** Differences are expected due to:
- Numerical precision (float32 vs optimized TFLite ops)
- Different convolution implementations
- Rounding differences in intermediate computations

## Advantages

### PyTorch Implementation
- ✅ Higher precision (closer to original training)
- ✅ GPU acceleration support
- ✅ Faster inference with CUDA
- ✅ Better for research and development

### TFLite Implementation
- ✅ Lower memory footprint (~500 MB vs 2-3 GB)
- ✅ CPU-optimized (XNNPACK delegate)
- ✅ Mobile deployment ready
- ✅ No GPU required
- ✅ Smaller model files (~29 MB vs ~30 MB)
- ✅ Cross-platform compatibility

## Deployment

### Mobile Deployment (TFLite)

**Android:**
```kotlin
val interpreter = Interpreter(File(modelPath))
interpreter.run(inputArray, outputArray)
```

**iOS:**
```swift
let interpreter = try Interpreter(modelPath: modelPath)
try interpreter.invoke()
```

### Server Deployment (PyTorch)

```python
import torch
from uvq1p5_pytorch.utils import uvq1p5

model = uvq1p5.UVQ1p5()
if torch.cuda.is_available():
    model.cuda()

# Run inference...
```

## Known Issues

1. **Numerical Differences:** TFLite and PyTorch produce slightly different results (~2% difference) due to different operator implementations and numerical precision.

2. **Memory Usage:** TFLite implementation processes frames sequentially, which is slower but uses less memory than PyTorch batch processing.

3. **Dependencies:** TFLite requires TensorFlow (large dependency), while PyTorch version requires PyTorch.

## Future Improvements

- [ ] Quantization support (INT8) for TFLite models
- [ ] Batch processing optimization for TFLite
- [ ] GPU delegate support for TFLite
- [ ] Reduce numerical differences between implementations
- [ ] Mobile app examples (Android/iOS)

## License

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## References

- PyTorch Implementation: `utils/uvq1p5.py`
- TFLite Implementation: `utils/uvq1p5_tflite.py`
- Test Script: `../test_tflite_inference.py`
- TFLite Models: `../models/tflite_models/uvq1.5/`
- PyTorch Models: `checkpoints/`

## Contact

For questions or issues:
- Check the main README: `../README.md`
- Review test script: `../test_tflite_inference.py`
- See operator analysis: `~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5/OPERATOR_ANALYSIS.md`

