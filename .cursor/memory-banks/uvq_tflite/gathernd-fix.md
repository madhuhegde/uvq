# GatherNd Operator Issue Fix

## Problem

After changing the input format from `[B, C, H, W]` to `[B, H, W, C]`, the newly converted TFLite models contained `GATHER_ND` operators, which were not present in the original models. This was problematic because:

- `GATHER_ND` ops are inefficient in TFLite
- They add unnecessary overhead
- The original models didn't have them

## Root Cause

The issue was caused by `permute()` operations in the model wrapper without ensuring memory contiguity. When PyTorch tensors are permuted, they create a view with a different stride pattern but don't necessarily reorganize the data in memory. During TFLite conversion, this non-contiguous memory layout was being converted to `GATHER_ND` operations.

**Problematic code:**
```python
# This creates a non-contiguous tensor view
video_frame = video_frame.permute(0, 3, 1, 2)
```

## Solution

Add `.contiguous()` after each `permute()` operation to ensure the tensor data is reorganized in memory with the new layout. This allows the TFLite converter to use efficient `TRANSPOSE` operators instead of `GATHER_ND`.

**Fixed code:**
```python
# This creates a contiguous tensor with reorganized memory
video_frame = video_frame.permute(0, 3, 1, 2).contiguous()
```

## Changes Made

### File: `uvq_models.py`

#### ContentNetWrapper
```python
def forward(self, video_frame):
    # Convert from TensorFlow format (B, H, W, C) to PyTorch format (B, C, H, W)
    # Use contiguous() to ensure memory layout is optimal for TFLite conversion
    video_frame = video_frame.permute(0, 3, 1, 2).contiguous()
    
    # ContentNet model expects (batch, 3, 256, 256) and outputs (batch, 128, 8, 8)
    features = self.content_net.model(video_frame)
    
    # Permute to (batch, 8, 8, 128) format
    features = features.permute(0, 2, 3, 1).contiguous()
    return features
```

#### DistortionNetWrapper
```python
def forward(self, video_patches):
    # Convert from TensorFlow format (B, H, W, C) to PyTorch format (B, C, H, W)
    # Use contiguous() to ensure memory layout is optimal for TFLite conversion
    video_patches = video_patches.permute(0, 3, 1, 2).contiguous()
    
    # Process patches through DistortionNet core
    patch_features = self.distortion_net.model(video_patches)
    
    # ... (reshape logic)
    
    # Rearrange to (batch, 3*8, 3*8, 128) = (batch, 24, 24, 128)
    features = features.permute(0, 1, 4, 2, 5, 3).contiguous()
    features = features.reshape(batch_size, 24, 24, 128)
    
    return features
```

## Verification

### Before Fix (with GatherNd)
The models would have contained:
- `GATHER_ND` operator (inefficient)
- Potentially slower inference
- Non-standard TFLite operations

### After Fix (without GatherNd)

**ContentNet operators:**
```
✓ ADD
✓ CONV_2D
✓ DELEGATE
✓ DEPTHWISE_CONV_2D
✓ LOGISTIC
✓ MUL
✓ PAD
✓ RESHAPE
✓ RESIZE_BILINEAR
✓ SUM
✓ TRANSPOSE
```

**DistortionNet operators:**
```
✓ ADD
✓ CONV_2D
✓ DELEGATE
✓ DEPTHWISE_CONV_2D
✓ LOGISTIC
✓ MAX_POOL_2D
✓ MUL
✓ RESHAPE
✓ SUM
✓ TRANSPOSE
```

**✅ No GATHER_ND operators found!**

The `TRANSPOSE` operator is a native, efficient TFLite operation that's much faster than `GATHER_ND`.

## Test Results

After applying the fix and reconverting all models:

### FLOAT32 Models
```
✓ content_net.tflite        14.55 MB
✓ distortion_net.tflite     14.53 MB
✓ aggregation_net.tflite     0.30 MB
```

### INT8 Models
```
✓ content_net_int8.tflite    4.27 MB  (70.7% reduction)
✓ distortion_net_int8.tflite 4.25 MB  (70.7% reduction)
✓ aggregation_net_int8.tflite 0.11 MB (62.5% reduction)
```

### Inference Test
```bash
python test_tflite_inference.py ../dataset/Gaming_360P_local.mp4 --tflite
```

**Results:**
```
✓ All models loaded successfully
✓ UVQ 1.5 Score: 3.0319
✓ Processed 20 frames
✓ No errors or warnings
```

## Key Takeaways

### Why `.contiguous()` Matters

1. **Memory Layout:** `permute()` changes the tensor's logical view but not the underlying memory layout
2. **TFLite Conversion:** Non-contiguous tensors can lead to inefficient operators like `GATHER_ND`
3. **Performance:** Contiguous memory enables efficient `TRANSPOSE` operations in TFLite

### Best Practice

**Always use `.contiguous()` after `permute()` when preparing models for TFLite conversion:**

```python
# ✅ Good - Contiguous memory layout
tensor = tensor.permute(0, 3, 1, 2).contiguous()

# ❌ Bad - Non-contiguous memory layout
tensor = tensor.permute(0, 3, 1, 2)
```

### Performance Impact

- **TRANSPOSE operator:** Native TFLite op, highly optimized
- **GATHER_ND operator:** Generic indexing op, much slower
- **Memory access:** Contiguous layout enables better cache utilization

## Comparison with Backup Models

The fixed models now have the **same operators** as the backup models (which used `[B, C, H, W]` format directly), confirming that the input format change with `.contiguous()` doesn't introduce any unwanted operations.

## Summary

✅ **Issue:** `GATHER_ND` ops appeared after input format change  
✅ **Root Cause:** Non-contiguous memory layout from `permute()`  
✅ **Solution:** Add `.contiguous()` after all `permute()` operations  
✅ **Result:** Clean TFLite models with only `TRANSPOSE` ops  
✅ **Verification:** All tests passing, same operators as backup models  

---

**Date:** January 5, 2025  
**Status:** ✅ RESOLVED

