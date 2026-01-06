# Minimal DistortionNet TFLite Conversion Format Fix

## Date
January 6, 2025

## Problem
The TFLite converter for minimal DistortionNet models (single, minimal, medium) was using PyTorch format `[B, C, H, W]` instead of TensorFlow format `[B, H, W, C]`. This was inconsistent with the main conversion script and could cause issues with TFLite inference.

## Solution
Modified the conversion script to use TensorFlow format `[B, H, W, C]` for both input and output, consistent with the main UVQ 1.5 TFLite conversion.

## Changes Made

### File: `/home/madhuhegde/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5/convert_minimal_distortionnet.py`

#### 1. Added Wrapper Class

Created `DistortionNetMinimalWrapper` class to handle format conversion:

```python
class DistortionNetMinimalWrapper(nn.Module):
    """Wrapper for minimal DistortionNet models that handles TensorFlow format.
    
    This wrapper converts input from TensorFlow format [B, H, W, C] to PyTorch
    format [B, C, H, W] before passing through the model, then converts the
    output back to TensorFlow format.
    """
    
    def __init__(self, size='minimal'):
        super().__init__()
        self.model = get_minimal_distortionnet(size)
        self.model.eval()
        self.size = size
    
    def forward(self, x):
        """
        Args:
            x: Tensor in TensorFlow format [B, H, W, C]
               For 'single': [9, 180, 320, 16]
               For 'minimal'/'medium': [9, 360, 640, 3]
        
        Returns:
            features: Tensor in TensorFlow format [B, H, W, C]
        """
        # Convert from TensorFlow format (B, H, W, C) to PyTorch format (B, C, H, W)
        # Use contiguous() to ensure memory layout is optimal for TFLite conversion
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Process through the minimal distortion net
        features = self.model(x)
        
        # The model already outputs in NHWC format due to PermuteLayerNHWC
        # So we don't need to permute again
        return features
```

**Key Points:**
- Uses `.contiguous()` after `permute()` to avoid GATHER_ND operators in TFLite
- Handles format conversion transparently
- Output is already in NHWC format from PermuteLayerNHWC in the model

#### 2. Updated Input Format in Conversion Function

**Before:**
```python
if size == 'single':
    sample_input = torch.randn(9, 16, 180, 320)  # PyTorch format: [B, C, H, W]
else:
    sample_input = torch.randn(9, 3, 360, 640)   # PyTorch format: [B, C, H, W]
```

**After:**
```python
if size == 'single':
    sample_input = torch.randn(9, 180, 320, 16)  # TensorFlow format: [B, H, W, C]
else:
    sample_input = torch.randn(9, 360, 640, 3)   # TensorFlow format: [B, H, W, C]
```

#### 3. Updated Verification Function

**Before:**
```python
if size == 'single':
    test_input = np.random.randn(9, 16, 180, 320).astype(np.float32)
else:
    test_input = np.random.randn(9, 3, 360, 640).astype(np.float32)
```

**After:**
```python
if size == 'single':
    test_input = np.random.randn(9, 180, 320, 16).astype(np.float32)
else:
    test_input = np.random.randn(9, 360, 640, 3).astype(np.float32)
```

#### 4. Updated Model Instantiation

**Before:**
```python
model = get_minimal_distortionnet(size)
model.eval()
```

**After:**
```python
model = DistortionNetMinimalWrapper(size)
model.eval()
```

## Input/Output Shapes

### Single Block Model
- **Input:** `[9, 180, 320, 16]` - 9 patches, 180x320 resolution, 16 channels (after initial conv)
- **Output:** `[9, H, W, C]` - Features in NHWC format

### Minimal Model (5 layers)
- **Input:** `[9, 360, 640, 3]` - 9 patches, 360x640 resolution, 3 RGB channels
- **Output:** `[9, H, W, 128]` - Features in NHWC format

### Medium Model (10 layers)
- **Input:** `[9, 360, 640, 3]` - 9 patches, 360x640 resolution, 3 RGB channels
- **Output:** `[9, H, W, 128]` - Features in NHWC format

## Consistency with Main Conversion

This change makes the minimal distortion net conversion consistent with the main conversion script:

### Main DistortionNet Conversion (`convert_to_tflite.py`)
```python
# Line 122
sample_input = torch.randn(9, 360, 640, 3)  # TensorFlow format: [B, H, W, C]
```

### Minimal DistortionNet Conversion (Now Fixed)
```python
sample_input = torch.randn(9, 360, 640, 3)  # TensorFlow format: [B, H, W, C]
```

## Benefits

1. **Consistency:** All conversion scripts now use the same format
2. **Compatibility:** TFLite models will have the expected input/output shapes
3. **Performance:** Using `.contiguous()` prevents GATHER_ND operators
4. **Maintainability:** Easier to understand and maintain with consistent format

## Testing

To test the changes:

```bash
cd ~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5
eval "$(micromamba shell hook --shell bash)"
micromamba activate ai_edge_torch_env

# Convert single block model
python convert_minimal_distortionnet.py --size single --output_dir ./debug_models --verify

# Convert minimal model
python convert_minimal_distortionnet.py --size minimal --output_dir ./debug_models --verify

# Convert medium model
python convert_minimal_distortionnet.py --size medium --output_dir ./debug_models --verify
```

## Related Files

- Main conversion script: `convert_to_tflite.py`
- Model wrappers: `uvq_models.py`
- Minimal models: `~/work/UVQ/uvq/uvq1p5_pytorch/utils/distortionnet_minimal.py`

## Status
✅ **COMPLETE** - All changes implemented and syntax validated

