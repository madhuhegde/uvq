# UVQ TFLite Models - Overview

## What is UVQ TFLite?

UVQ (Universal Video Quality) TFLite models are TensorFlow Lite conversions of PyTorch-based video quality assessment models, specifically optimized for BSTM hardware deployment.

## Key Components

### DistortionNet
- **Purpose**: Detects visual distortions in video frames
- **Architecture**: EfficientNet-B0 backbone with depthwise separable convolutions
- **Input**: 9 patches (3×3 grid) of 360×640 RGB images in NHWC format
- **Output**: Aggregated feature map of shape [1, 24, 24, 128]

### ContentNet
- **Purpose**: Analyzes content characteristics of video frames
- **Input**: 256×256 RGB images in NHWC format
- **Output**: Content features for quality assessment

### AggregationNet
- **Purpose**: Combines distortion and content features to produce final quality scores
- **Input**: Features from DistortionNet and ContentNet
- **Output**: Video quality metrics

## Model Variants

### FLOAT32 Models
- Full precision (32-bit floating point)
- Maximum accuracy
- Larger file size (~14.53 MB for DistortionNet)
- Perfect match with PyTorch reference (correlation: 1.000000)

### INT8 Models
- Quantized (8-bit integer)
- 70% smaller file size (~4.25 MB for DistortionNet)
- Faster inference on hardware
- Good accuracy (correlation: >0.97 with PyTorch)
- Expected quantization error within acceptable range

## BSTM Hardware Compatibility

All models are optimized for BSTM hardware with the following constraints:

✅ **Maximum 4D tensors** - No 5D or 6D tensors allowed  
✅ **No GATHER_ND operators** - Inefficient operators eliminated  
✅ **NHWC format** - TensorFlow/TFLite standard format  
✅ **Contiguous memory layout** - Optimized for hardware execution  

## Key Innovation: 4D Aggregation

The critical innovation is the **pure 4D patch aggregation** strategy that:
- Converts 9 patches (3×3 grid) into a single spatial grid
- Uses only reshape and permute operations (no 5D/6D tensors)
- Produces identical output to the original 6D approach
- Fully compatible with BSTM hardware constraints

## Model Locations

All TFLite models are located in:
```
~/work/UVQ/uvq/models/tflite_models/uvq1.5/
```

## See Also

- [4D Aggregation Implementation](./4d-aggregation.md) - Technical details of the aggregation strategy
- [Model Usage](./usage.md) - How to use the TFLite models
- [Conversion Pipeline](./conversion.md) - How models are converted from PyTorch
- [Verification](./verification.md) - Testing and validation results
