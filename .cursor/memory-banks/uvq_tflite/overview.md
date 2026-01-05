# UVQ 1.5 TFLite Conversion - Overview

## What is UVQ 1.5?

UVQ (Universal Video Quality) 1.5 is a neural network-based video quality assessment model that predicts perceptual quality scores for videos on a scale of 1-5.

### Architecture

UVQ 1.5 consists of three neural networks:

1. **ContentNet** - Extracts semantic content features from video frames
   - Input: (1, 3, 256, 256) RGB frames
   - Output: (1, 8, 8, 128) content feature maps

2. **DistortionNet** - Detects visual distortions using patch-based processing
   - Input: (9, 3, 360, 640) - 9 patches per frame (3×3 grid)
   - Output: (1, 24, 24, 128) distortion feature maps

3. **AggregationNet** - Combines features to produce final quality scores
   - Inputs: Content + Distortion features
   - Output: (1, 1) quality score [1-5]

## TFLite Conversion Project

This project successfully converted UVQ 1.5 from PyTorch to TensorFlow Lite with two variants:

### FLOAT32 Models (Baseline)
- Full precision conversion from PyTorch
- Model size: 29.38 MB total
- Preserves maximum accuracy
- Suitable for server/desktop deployment

### INT8 Quantized Models (Optimized)
- Dynamic INT8 quantization applied
- Model size: 8.63 MB total (**70.6% smaller**)
- Inference speed: **1.30x faster**
- Accuracy: 2.44% difference (excellent)
- Ideal for mobile/edge deployment

## Key Results

| Metric | FLOAT32 | INT8 | Improvement |
|:-------|--------:|-----:|:-----------:|
| **Model Size** | 29.38 MB | 8.63 MB | **70.6% smaller** |
| **Inference Speed** | 15.235s | 11.750s | **1.30x faster** |
| **Quality Score** | 3.0440 | 3.1185 | **2.44% diff** |

## Recommendation

✅ **Deploy INT8 models for production**

The INT8 quantized models provide excellent balance:
- Significant size reduction (70.6%)
- Better performance (1.30x speedup)
- Minimal accuracy impact (2.44%)
- Lower memory footprint
- Faster model loading

## Project Structure

```
.cursor/memory-banks/uvq_tflite/
├── overview.md              # This file - project overview
├── implementation.md        # Complete implementation guide
├── usage.md                 # How to use the models
├── performance.md           # Performance analysis
├── quantization.md          # Quantization details
├── results-summary.md       # Presentation-ready tables
└── model-inventory.md       # Model file inventory
```

## Quick Links

- **Getting Started:** See [usage.md](./usage.md)
- **Implementation Details:** See [implementation.md](./implementation.md)
- **Performance Data:** See [performance.md](./performance.md)
- **Quantization Info:** See [quantization.md](./quantization.md)
- **Results Tables:** See [results-summary.md](./results-summary.md)

## See Also

- Main README: `~/work/UVQ/uvq/README.md`
- PyTorch Implementation: `uvq1p5_pytorch/utils/uvq1p5.py`
- TFLite Implementation: `uvq1p5_pytorch/utils/uvq1p5_tflite.py`
- Comparison Script: `compare_tflite_performance.py`
