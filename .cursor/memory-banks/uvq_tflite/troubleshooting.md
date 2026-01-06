# Troubleshooting Guide

## Common Issues and Solutions

### Conversion Issues

#### Issue: Size Mismatch Error

**Error**:
```
RuntimeError: Given groups=1, weight of size [32, 1, 3, 3] from checkpoint, 
the shape in current model is torch.Size([32, 32, 3, 3]).
```

**Cause**: Importing the wrong model variant (e.g., `distortionnet_no_depthwise` instead of `distortionnet`).

**Solution**:
```python
# Correct import
from uvq1p5_pytorch.utils import distortionnet

# Wrong import (causes error)
from uvq1p5_pytorch.utils import distortionnet_no_depthwise
```

**Fix in code**:
```python
# In uvq_models.py
from uvq1p5_pytorch.utils import distortionnet  # Use the full model with depthwise convolutions
```

---

#### Issue: GATHER_ND Operators Generated

**Error**: TFLite model contains GATHER_ND operators (incompatible with BSTM HW).

**Cause**: Missing `.contiguous()` calls after permute operations.

**Solution**: Always add `.contiguous()` after permute:
```python
# Wrong
x = x.permute(0, 3, 1, 2)

# Correct
x = x.permute(0, 3, 1, 2).contiguous()
```

**Verification**:
```python
with open('model.tflite', 'rb') as f:
    content = f.read()
if b'GATHER_ND' in content:
    print('⚠️  GATHER_ND found - add .contiguous() calls')
else:
    print('✅ No GATHER_ND')
```

---

#### Issue: 6D Tensor Error

**Error**:
```
Error: Tensor dimensions exceed 4D (found 6D tensor)
```

**Cause**: Using the old 6D aggregation approach.

**Solution**: Use the 4D aggregation approach:
```python
# Old approach (6D - INCOMPATIBLE)
features = patch_features.reshape(batch_size, 3, 3, 128, 8, 8)  # 6D!
features = features.permute(0, 1, 4, 2, 5, 3).contiguous()
features = features.reshape(batch_size, 24, 24, 128)

# New approach (4D - COMPATIBLE)
features = patch_features.reshape(3, 24, 8, 128)
features = features.permute(0, 2, 1, 3).contiguous()
features = features.reshape(1, 24, 24, 128)
```

See [4D Aggregation](./4d-aggregation.md) for details.

---

### Runtime Issues

#### Issue: Input Shape Mismatch

**Error**:
```
ValueError: Cannot set tensor: Got tensor with shape [1, 3, 360, 640] 
but expected [9, 360, 640, 3]
```

**Cause**: Input is in PyTorch format (NCHW) instead of TensorFlow format (NHWC).

**Solution**: Convert input to NHWC format:
```python
# If input is in PyTorch format [B, C, H, W]
input_nchw = torch.randn(9, 3, 360, 640)

# Convert to TensorFlow format [B, H, W, C]
input_nhwc = input_nchw.permute(0, 2, 3, 1).numpy()

# Now input_nhwc has shape [9, 360, 640, 3]
```

---

#### Issue: Model Not Found

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'distortion_net.tflite'
```

**Cause**: Model file doesn't exist or wrong path.

**Solution**: Use absolute path:
```python
from pathlib import Path

model_path = Path.home() / "work" / "UVQ" / "uvq" / "models" / "tflite_models" / "uvq1.5" / "distortion_net.tflite"
interpreter = tf.lite.Interpreter(model_path=str(model_path))
```

---

#### Issue: Low Correlation in Verification

**Error**: Correlation < 0.95 between PyTorch and TFLite.

**Possible causes**:
1. Wrong input format
2. Incorrect model loaded
3. Aggregation logic error
4. Missing preprocessing

**Debug steps**:

```python
# 1. Check input format
print(f"Input shape: {input.shape}")  # Should be [9, 360, 640, 3]
print(f"Input dtype: {input.dtype}")  # Should be float32

# 2. Check intermediate outputs
with torch.no_grad():
    # Check patch features before aggregation
    patch_features = model.distortion_net.model(input_nchw)
    print(f"Patch features shape: {patch_features.shape}")  # [9, 8, 8, 128]
    print(f"Patch features range: [{patch_features.min():.2f}, {patch_features.max():.2f}]")

# 3. Check final output
output = model(input_nhwc)
print(f"Output shape: {output.shape}")  # [1, 24, 24, 128]
print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")

# 4. Compare with TFLite
tflite_output = run_tflite_inference(input_nhwc)
correlation = np.corrcoef(output.flatten(), tflite_output.flatten())[0, 1]
print(f"Correlation: {correlation:.6f}")
```

---

### Environment Issues

#### Issue: TensorFlow Not Found

**Error**:
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solution**: Activate the correct environment:
```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate local_tf_env
```

---

#### Issue: ai_edge_torch Not Found

**Error**:
```
ModuleNotFoundError: No module named 'ai_edge_torch'
```

**Solution**: Install ai_edge_torch:
```bash
pip install ai-edge-torch
```

Or use the development version:
```bash
cd /home/madhuhegde/work/ai_edge_torch/ai-edge-torch
pip install -e .
```

---

### Model Quality Issues

#### Issue: INT8 Model Has High Error

**Observation**: INT8 model shows large absolute differences compared to PyTorch.

**Expected behavior**: This is normal for quantized models. Check correlation instead:
- **Correlation ≥ 0.95**: Good match (acceptable)
- **Correlation < 0.95**: Investigate further

**Acceptable INT8 metrics**:
```
Correlation: 0.979218 (>0.95 ✓)
Max absolute difference: 56.25 (large but acceptable)
Mean absolute difference: 4.97 (acceptable)
```

**If correlation < 0.95**:
1. Check if quantization config is correct
2. Try different quantization granularity (per-channel vs per-tensor)
3. Consider using FLOAT32 if accuracy is critical

---

#### Issue: Model Size Too Large

**Problem**: FLOAT32 model is 14.53 MB, too large for deployment.

**Solution**: Use INT8 quantized model:
```bash
python convert_to_tflite_int8.py --model distortion --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
```

**Result**: 4.25 MB (70% size reduction) with minimal accuracy loss.

---

### Hardware Compatibility Issues

#### Issue: Model Fails on BSTM Hardware

**Possible causes**:
1. GATHER_ND operators present
2. Tensors exceed 4D
3. Unsupported operations

**Debug steps**:

```python
# 1. Check for GATHER_ND
with open('distortion_net.tflite', 'rb') as f:
    content = f.read()
    
if b'GATHER_ND' in content:
    print('❌ GATHER_ND found - model incompatible')
else:
    print('✅ No GATHER_ND')

# 2. Check tensor dimensions (requires flatbuffers)
# Use analyze_tflite_ops.py script
python analyze_tflite_ops.py distortion_net.tflite

# 3. Verify using correct model
# Make sure you're using distortion_net.tflite, not distortion_net_6d.tflite
```

**Solution**: Ensure you're using the 4D aggregation model:
- ✅ Use: `distortion_net.tflite` or `distortion_net_int8.tflite`
- ❌ Don't use: `distortion_net_6d.tflite` or `distortion_net_int8_6d.tflite`

---

## Debugging Workflow

### Step 1: Identify the Issue

```bash
# Run comprehensive verification
cd /home/madhuhegde/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5
python compare_distortionnet_all.py
```

### Step 2: Check Model Properties

```python
import tensorflow as tf

# Load model
interpreter = tf.lite.Interpreter(model_path="distortion_net.tflite")
interpreter.allocate_tensors()

# Check input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape:  {input_details[0]['shape']}")
print(f"Input dtype:  {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output dtype: {output_details[0]['dtype']}")
```

### Step 3: Test with Known Input

```python
import numpy as np

# Create test input
np.random.seed(42)
test_input = np.random.randn(9, 360, 640, 3).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
```

### Step 4: Compare with PyTorch

```python
import torch
from uvq_models import create_distortion_net

# Load PyTorch model
pytorch_model = create_distortion_net()
pytorch_model.eval()

# Run inference
test_input_torch = torch.from_numpy(test_input)
with torch.no_grad():
    pytorch_output = pytorch_model(test_input_torch).numpy()

# Compare
correlation = np.corrcoef(output.flatten(), pytorch_output.flatten())[0, 1]
print(f"Correlation: {correlation:.6f}")

if correlation >= 0.99:
    print("✅ Excellent match")
elif correlation >= 0.95:
    print("✓ Good match")
else:
    print("❌ Poor match - investigate further")
```

---

## Getting Help

### Useful Commands

```bash
# Check model file exists
ls -lh ~/work/UVQ/uvq/models/tflite_models/uvq1.5/distortion_net*.tflite

# Check Python environment
which python
pip list | grep -E "(torch|tensorflow|ai-edge)"

# Run verification
cd /home/madhuhegde/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5
python compare_distortionnet_all.py

# Check for GATHER_ND
python -c "
with open('distortion_net.tflite', 'rb') as f:
    print('✅ No GATHER_ND' if b'GATHER_ND' not in f.read() else '❌ GATHER_ND found')
"
```

### Log Information to Provide

When reporting issues, include:

1. **Environment info**:
   ```bash
   python --version
   pip list | grep -E "(torch|tensorflow|ai-edge)"
   ```

2. **Model info**:
   ```bash
   ls -lh distortion_net.tflite
   ```

3. **Error message**: Full error traceback

4. **Verification results**: Output from `compare_distortionnet_all.py`

5. **Input/output shapes**: From your test code

---

## See Also

- [Overview](./overview.md) - General information about UVQ TFLite
- [Conversion](./conversion.md) - How to convert models
- [Verification](./verification.md) - Testing and validation
- [4D Aggregation](./4d-aggregation.md) - Technical details of aggregation

