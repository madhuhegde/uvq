# DistortionNet Structure and Debugging Guide

## Overview

DistortionNet is based on **EfficientNet-B0** architecture with modifications for video quality assessment. It processes 360×640 patches from 1080p video frames.

**Key Stats:**
- **Total Parameters:** 3,759,228 (~3.8M)
- **Total Layers:** 18 (16 MBConv blocks + 2 conv layers)
- **Input:** 3 × 360 × 640 (RGB patch)
- **Output:** 128 × 1 × 1 (feature vector after pooling)

---

## Architecture Breakdown

### Stage-by-Stage Structure

```
Input: [B, 3, 360, 640]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Initial Convolution                                │
│   Conv2dNormActivation: 3 → 32, kernel=3, stride=2         │
│   Output: [B, 32, 180, 320]                                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: MBConv1 (expand_ratio=1)                          │
│   Block 1: 32 → 16, kernel=3, stride=1                     │
│   Output: [B, 16, 180, 320]                                 │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: MBConv6 (expand_ratio=6)                          │
│   Block 1: 16 → 24, kernel=3, stride=2                     │
│   Block 2: 24 → 24, kernel=3, stride=1                     │
│   Output: [B, 24, 90, 160]                                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: MBConv6                                            │
│   Block 1: 24 → 40, kernel=5, stride=2                     │
│   Block 2: 40 → 40, kernel=5, stride=1                     │
│   Output: [B, 40, 45, 80]                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: MBConv6                                            │
│   Block 1: 40 → 80, kernel=3, stride=2                     │
│   Block 2: 80 → 80, kernel=3, stride=1                     │
│   Block 3: 80 → 80, kernel=3, stride=1                     │
│   Output: [B, 80, 23, 40]                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: MBConv6                                            │
│   Block 1: 80 → 112, kernel=5, stride=1                    │
│   Block 2: 112 → 112, kernel=5, stride=1                   │
│   Block 3: 112 → 112, kernel=5, stride=1                   │
│   Output: [B, 112, 23, 40]                                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 7: MBConv6                                            │
│   Block 1: 112 → 192, kernel=5, stride=2                   │
│   Block 2: 192 → 192, kernel=5, stride=1                   │
│   Block 3: 192 → 192, kernel=5, stride=1                   │
│   Block 4: 192 → 192, kernel=5, stride=1                   │
│   Output: [B, 192, 12, 20]                                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 8: MBConv6                                            │
│   Block 1: 192 → 320, kernel=3, stride=1                   │
│   Output: [B, 320, 12, 20]                                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 9: Final Convolution                                  │
│   Conv2dSamePadding: 320 → 128, kernel=2, stride=1         │
│   Output: [B, 128, 11, 19]                                  │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 10: Pooling & Permute                                 │
│   MaxPool2d: kernel=(5, 13), stride=1                      │
│   Permute: NCHW → NHWC                                      │
│   Output: [B, 1, 1, 128]                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## MBConv Block Structure

Each MBConv (Mobile Inverted Bottleneck Convolution) block contains:

```
Input: [B, in_channels, H, W]
    ↓
┌─────────────────────────────────────────┐
│ 1. Expansion (if expand_ratio > 1)     │
│    Conv2d 1×1: in → expanded            │
│    BatchNorm + SiLU                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 2. Depthwise Convolution ⚠️             │
│    Conv2d k×k: expanded → expanded      │
│    groups = expanded (DEPTHWISE)        │
│    BatchNorm + SiLU                     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 3. Squeeze-and-Excitation (SE)         │
│    GlobalAvgPool                        │
│    FC: expanded → squeeze → expanded    │
│    Sigmoid                              │
│    Channel-wise multiply                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 4. Projection                           │
│    Conv2d 1×1: expanded → out           │
│    BatchNorm (no activation)            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 5. Residual Connection (if applicable) │
│    if stride==1 and in==out:            │
│        output = input + output          │
└─────────────────────────────────────────┘
    ↓
Output: [B, out_channels, H', W']
```

**⚠️ The depthwise convolution is the problematic layer for TFLite!**

---

## Minimal Models for Debugging

I've created three minimal versions to help isolate issues:

### Option 1: Single Block (Minimal)

**File:** `distortionnet_minimal.py` → `DistortionNetSingleBlock`

```python
from uvq1p5_pytorch.utils.distortionnet_minimal import get_minimal_distortionnet

# Just one MBConv6 block
model = get_minimal_distortionnet('single')

# Input: [B, 16, H, W]
# Output: [B, 24, H, W]
```

**Use case:** Isolate the depthwise convolution issue
- Only 1 MBConv block
- Input: 16 channels
- Output: 24 channels
- Expand ratio: 6 (16 × 6 = 96 expanded channels)
- **This is the smallest model to test depthwise conv**

### Option 2: Minimal (5 Layers)

**File:** `distortionnet_minimal.py` → `DistortionNetMinimal`

```python
model = get_minimal_distortionnet('minimal')

# Input: [B, 3, 360, 640]
# Output: [B, 1, 1, 128]
```

**Use case:** Test first 3 stages
- Initial Conv: 3 → 32
- MBConv1: 32 → 16
- MBConv6: 16 → 24 (2 blocks)
- Final Conv: 24 → 128
- MaxPool + Permute

**Parameters:** ~500K (vs 3.8M in full model)

### Option 3: Medium (10 Layers)

**File:** `distortionnet_minimal.py` → `DistortionNetMedium`

```python
model = get_minimal_distortionnet('medium')

# Input: [B, 3, 360, 640]
# Output: [B, 1, 1, 128]
```

**Use case:** Test first 5 stages
- Stages 1-3 (as in Minimal)
- Stage 4: MBConv6 (24→40, 2 blocks)
- Stage 5: MBConv6 (40→80, 3 blocks)
- Final Conv: 80 → 128
- MaxPool + Permute

**Parameters:** ~1.5M (vs 3.8M in full model)

---

## Debugging Strategy

### Step 1: Identify the Problematic Stage

Test each minimal model in sequence:

```bash
# Test single block
python test_minimal_distortionnet.py --size single

# Test minimal (5 layers)
python test_minimal_distortionnet.py --size minimal

# Test medium (10 layers)
python test_minimal_distortionnet.py --size medium

# Test full model
python test_minimal_distortionnet.py --size full
```

### Step 2: Isolate the Depthwise Conv Issue

If the single block fails, the issue is in the depthwise convolution itself.

**Depthwise Conv Parameters:**
- Input: 96 channels (16 × 6 expansion)
- Output: 96 channels
- Kernel: 3×3
- Groups: 96 (each channel has its own filter)
- Stride: 1

### Step 3: Check TFLite Operators

```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Check for DEPTHWISE_CONV_2D operator
for op in interpreter._get_ops_details():
    if op['op_name'] == 'DEPTHWISE_CONV_2D':
        print(f"Found depthwise conv: {op}")
```

### Step 4: Try Alternative Approaches

If depthwise convolution is the issue:

**Option A: Use CPU-only delegate**
```python
# Don't use any hardware acceleration
interpreter = tf.lite.Interpreter(model_path="model.tflite")
```

**Option B: Use XNNPACK delegate**
```python
import tensorflow as tf

interpreter = tf.lite.Interpreter(
    model_path="model.tflite",
    experimental_delegates=[
        tf.lite.experimental.load_delegate('libxnnpack.so')
    ]
)
```

**Option C: Use GPU delegate (if available)**
```python
interpreter = tf.lite.Interpreter(
    model_path="model.tflite",
    experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')]
)
```

---

## Common TFLite Issues with DistortionNet

### Issue 1: Depthwise Conv Not Supported

**Symptom:** Error during inference with DEPTHWISE_CONV_2D operator

**Solutions:**
1. Use XNNPACK delegate (best performance)
2. Use CPU-only (most compatible)
3. Replace depthwise with normal conv (requires retraining)

### Issue 2: Memory Issues

**Symptom:** OOM or slow inference

**Solutions:**
1. Process patches one at a time instead of batch
2. Use INT8 quantization
3. Use smaller input size for testing

### Issue 3: Accuracy Degradation

**Symptom:** TFLite output differs significantly from PyTorch

**Solutions:**
1. Check input preprocessing (especially normalization)
2. Verify input format ([B, H, W, C] vs [B, C, H, W])
3. Use FLOAT32 instead of INT8 for debugging

---

## Testing Script

Create `test_minimal_distortionnet.py`:

```python
#!/usr/bin/env python3
import argparse
import torch
import numpy as np
from uvq1p5_pytorch.utils.distortionnet_minimal import get_minimal_distortionnet

def test_model(size='minimal'):
    print(f"\nTesting {size} DistortionNet...")
    
    # Create model
    model = get_minimal_distortionnet(size)
    model.eval()
    
    # Create sample input
    if size == 'single':
        x = torch.randn(1, 16, 180, 320)  # After initial conv
    else:
        x = torch.randn(1, 3, 360, 640)   # Full patch
    
    print(f"Input shape: {x.shape}")
    
    # Run inference
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=['single', 'minimal', 'medium'], 
                       default='minimal')
    args = parser.parse_args()
    
    test_model(args.size)
```

---

## Summary

### Key Points

1. **DistortionNet = EfficientNet-B0** with 16 MBConv blocks
2. **Each MBConv block contains depthwise convolution** (this is likely the issue)
3. **Minimal models available** for debugging at 3 levels
4. **The problematic operator** is DEPTHWISE_CONV_2D with groups=expanded_channels

### Recommended Debugging Approach

1. ✅ Start with `single` block to isolate depthwise conv
2. ✅ Try different TFLite delegates (XNNPACK, CPU, GPU)
3. ✅ Check if issue is in conversion or inference
4. ✅ Compare PyTorch vs TFLite outputs at each stage
5. ✅ If all else fails, consider replacing depthwise with normal conv (requires retraining)

### Files Created

- `distortionnet_minimal.py` - Minimal model implementations
- `distortionnet_no_depthwise.py` - Version without depthwise conv (requires retraining)
- `custom_nn_layers_no_depthwise.py` - Modified layers without depthwise conv

---

**What specific error are you encountering with the DistortionNet TFLite model?** This will help me provide more targeted debugging steps.

