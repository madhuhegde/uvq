# Model Conversion Pipeline

## Overview

This document describes how to convert UVQ PyTorch models to TFLite format with BSTM hardware compatibility.

## Prerequisites

### Environment Setup

```bash
# Activate the TensorFlow environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate local_tf_env
```

### Required Packages

- `torch` - PyTorch framework
- `tensorflow` - TensorFlow for TFLite
- `ai_edge_torch` - PyTorch to TFLite converter
- `numpy` - Numerical operations

## Conversion Scripts Location

All conversion scripts are located in:
```
/home/madhuhegde/work/ai_edge_torch/ai-edge-torch/
  ai_edge_torch/generative/examples/uvq1.5/
```

## Converting DistortionNet

### FLOAT32 Conversion

```bash
cd /home/madhuhegde/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5

# Convert DistortionNet to FLOAT32 TFLite
python convert_to_tflite.py \
  --model distortion \
  --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
```

**Output**: `distortion_net.tflite` (14.53 MB)

### INT8 Conversion

```bash
# Convert DistortionNet to INT8 TFLite
python convert_to_tflite_int8.py \
  --model distortion \
  --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
```

**Output**: `distortion_net_int8.tflite` (4.25 MB)

## Conversion Pipeline Details

### Step 1: Model Wrapping

The PyTorch model is wrapped to handle TensorFlow format (NHWC):

```python
from uvq_models import create_distortion_net

# Create wrapped model
model = create_distortion_net(model_path=None)  # Loads default checkpoint
model.eval()
```

The wrapper (`DistortionNetWrapper`) handles:
- Input conversion: NHWC → NCHW
- PyTorch model inference
- 4D patch aggregation
- Output format: NHWC

### Step 2: Sample Input Creation

```python
import torch

# Create sample input in TensorFlow format
sample_input = torch.randn(9, 360, 640, 3)  # [B, H, W, C]
```

### Step 3: FLOAT32 Conversion

```python
import ai_edge_torch

# Convert to TFLite
edge_model = ai_edge_torch.convert(
    model,
    (sample_input,),
)

# Export
edge_model.export("distortion_net.tflite")
```

### Step 4: INT8 Quantization

```python
from ai_edge_torch.generative.quantize import quant_recipes, quant_attrs

# Create quantization config
quant_config = quant_recipes.full_dynamic_recipe(
    mcfg=None,  # No model config needed
    weight_dtype=quant_attrs.Dtype.INT8,
    granularity=quant_attrs.Granularity.CHANNELWISE
)

# Convert with quantization
edge_model = ai_edge_torch.convert(
    model,
    (sample_input,),
    quant_config=quant_config
)

# Export
edge_model.export("distortion_net_int8.tflite")
```

## Key Implementation Details

### 4D Aggregation in Wrapper

The critical part ensuring BSTM compatibility is in `uvq_models.py`:

```python
class DistortionNetWrapper(nn.Module):
    def forward(self, video_patches):
        # Input: [9, 360, 640, 3] (NHWC)
        
        # Convert to PyTorch format
        video_patches = video_patches.permute(0, 3, 1, 2).contiguous()
        
        # Process through DistortionNet core
        patch_features = self.distortion_net.model(video_patches)
        # Output: [9, 8, 8, 128] (NHWC due to PermuteLayerNHWC)
        
        # Pure 4D aggregation (BSTM HW compatible)
        features = patch_features.reshape(3, 24, 8, 128)
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.reshape(1, 24, 24, 128)
        
        return features  # [1, 24, 24, 128] (NHWC)
```

### Ensuring No GATHER_ND

To prevent GATHER_ND operators:

1. **Use `.contiguous()`** after permute operations
2. **Use simple reshape/permute** instead of complex indexing
3. **Maintain NHWC format** throughout the pipeline

```python
# Good: No GATHER_ND
x = x.permute(0, 3, 1, 2).contiguous()  # Always use .contiguous()

# Bad: May generate GATHER_ND
x = x[:, :, :, [0, 1, 2]]  # Complex indexing
```

## Verification After Conversion

Always verify the converted model:

```bash
# Comprehensive verification
python compare_distortionnet_all.py
```

This script:
1. Loads PyTorch reference model
2. Loads TFLite FLOAT32 model
3. Loads TFLite INT8 model
4. Runs inference on all three
5. Compares outputs and reports metrics

Expected results:
- **FLOAT32**: Correlation ≈ 1.000000 (perfect match)
- **INT8**: Correlation ≥ 0.97 (good match)

## Checking for GATHER_ND

Verify no GATHER_ND operators exist:

```python
import tensorflow as tf

model_path = "distortion_net.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)

with open(model_path, 'rb') as f:
    content = f.read()

if b'GATHER_ND' in content:
    print('⚠️  GATHER_ND found in model')
else:
    print('✅ No GATHER_ND found in model')
```

## Troubleshooting Conversion Issues

### Issue: Size Mismatch Error

```
RuntimeError: Given groups=1, weight of size [32, 1, 3, 3], 
expected input to have 3 channels, but got 16 channels instead
```

**Solution**: Ensure you're importing the correct model variant:
```python
# Correct
from uvq1p5_pytorch.utils import distortionnet

# Wrong
from uvq1p5_pytorch.utils import distortionnet_no_depthwise
```

### Issue: GATHER_ND Operators Generated

**Solution**: Add `.contiguous()` after all permute operations:
```python
x = x.permute(0, 3, 1, 2).contiguous()  # Always add .contiguous()
```

### Issue: 6D Tensor Error

```
Error: Tensor dimensions exceed 4D
```

**Solution**: Use the 4D aggregation approach (see [4D Aggregation](./4d-aggregation.md))

## Model Checkpoint Locations

PyTorch checkpoints are loaded from:
```
~/work/UVQ/uvq/models/pytorch_models/
├── distortion_net.pth
├── content_net.pth
└── aggregation_net.pth
```

## Output Locations

TFLite models are saved to:
```
~/work/UVQ/uvq/models/tflite_models/uvq1.5/
├── distortion_net.tflite          # FLOAT32
├── distortion_net_int8.tflite     # INT8
├── content_net.tflite             # FLOAT32
├── content_net_int8.tflite        # INT8
├── aggregation_net.tflite         # FLOAT32
└── aggregation_net_int8.tflite    # INT8
```

## Conversion Script Reference

### convert_to_tflite.py

Main conversion script for FLOAT32 models.

**Arguments**:
- `--model`: Model to convert (`distortion`, `content`, `aggregation`, or `all`)
- `--output_dir`: Output directory for TFLite models

**Example**:
```bash
python convert_to_tflite.py --model all --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
```

### convert_to_tflite_int8.py

Conversion script for INT8 quantized models.

**Arguments**:
- `--model`: Model to convert (`distortion`, `content`, `aggregation`, or `all`)
- `--output_dir`: Output directory for TFLite models

**Example**:
```bash
python convert_to_tflite_int8.py --model all --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5
```

## See Also

- [Overview](./overview.md) - General information about UVQ TFLite
- [4D Aggregation](./4d-aggregation.md) - Technical details of aggregation
- [Usage](./usage.md) - How to use the converted models
- [Verification](./verification.md) - Testing and validation
- [Troubleshooting](./troubleshooting.md) - Common issues

