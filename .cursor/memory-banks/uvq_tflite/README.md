# UVQ TFLite Models - Documentation

## Quick Start

This memory bank contains comprehensive documentation for UVQ (Universal Video Quality) TFLite models optimized for BSTM hardware deployment.

## Documentation Structure

### 📖 [Overview](./overview.md)
**Start here** - High-level introduction to UVQ TFLite models, key components, and BSTM hardware compatibility.

**Topics covered**:
- What is UVQ TFLite?
- Model variants (FLOAT32 vs INT8)
- BSTM hardware requirements
- Key innovation: 4D aggregation

---

### 🔧 [4D Aggregation](./4d-aggregation.md)
**Technical deep-dive** - Detailed explanation of the pure 4D patch aggregation strategy that makes models BSTM-compatible.

**Topics covered**:
- Problem: 6D tensor incompatibility
- Solution: Pure 4D aggregation approach
- How it works (with visual diagrams)
- Implementation details
- Verification results

---

### 💻 [Usage Guide](./usage.md)
**How to use the models** - Practical guide for loading and running TFLite models in your application.

**Topics covered**:
- Model selection (FLOAT32 vs INT8)
- Python usage examples
- Input/output formats
- Performance tips
- Complete inference pipeline

---

### 🔄 [Conversion Pipeline](./conversion.md)
**How to convert models** - Step-by-step guide for converting PyTorch models to TFLite format.

**Topics covered**:
- Environment setup
- FLOAT32 conversion
- INT8 quantization
- Key implementation details
- Ensuring BSTM compatibility

---

### ✅ [Verification](./verification.md)
**Testing and validation** - Comprehensive verification process to ensure model quality and compatibility.

**Topics covered**:
- Verification results (FLOAT32 and INT8)
- BSTM hardware compatibility checklist
- Manual verification process
- Acceptance criteria
- 4D aggregation verification

---

### 🔍 [Troubleshooting](./troubleshooting.md)
**Common issues and solutions** - Debug guide for conversion, runtime, and compatibility issues.

**Topics covered**:
- Conversion issues (size mismatch, GATHER_ND, 6D tensors)
- Runtime issues (input shape, model not found)
- Model quality issues (INT8 error, model size)
- Hardware compatibility issues
- Debugging workflow

---

## Quick Reference

### Model Files

All models located in: `~/work/UVQ/uvq/models/tflite_models/uvq1.5/`

| Model | Size | Correlation | Use Case |
|-------|------|-------------|----------|
| `distortion_net.tflite` | 14.53 MB | 1.000000 | Maximum accuracy |
| `distortion_net_int8.tflite` | 4.25 MB | 0.979218 | Production (70% smaller) |

### Key Commands

```bash
# Convert to FLOAT32
python convert_to_tflite.py --model distortion --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5

# Convert to INT8
python convert_to_tflite_int8.py --model distortion --output_dir ~/work/UVQ/uvq/models/tflite_models/uvq1.5

# Verify models
python compare_distortionnet_all.py

# Check for GATHER_ND
python -c "
with open('distortion_net.tflite', 'rb') as f:
    print('✅ No GATHER_ND' if b'GATHER_ND' not in f.read() else '❌ GATHER_ND found')
"
```

### BSTM Hardware Requirements

✅ **No GATHER_ND operators** - Verified in both models  
✅ **All tensors ≤ 4D** - Pure 4D aggregation  
✅ **NHWC format** - TensorFlow/TFLite standard  
✅ **Contiguous memory** - Optimized layout  

### Model I/O

**Input**: `[9, 360, 640, 3]` (9 patches, 360×640 RGB, NHWC)  
**Output**: `[1, 24, 24, 128]` (aggregated features, NHWC)

## Navigation

- **New to UVQ TFLite?** → Start with [Overview](./overview.md)
- **Want to use models?** → See [Usage Guide](./usage.md)
- **Need to convert models?** → See [Conversion Pipeline](./conversion.md)
- **Verifying model quality?** → See [Verification](./verification.md)
- **Encountering issues?** → See [Troubleshooting](./troubleshooting.md)
- **Understanding 4D aggregation?** → See [4D Aggregation](./4d-aggregation.md)

## Status

✅ **Production Ready**  
✅ **BSTM HW Compatible**  
✅ **Fully Verified**  
✅ **Documented**  

---

**Last Updated**: January 6, 2025  
**Version**: UVQ 1.5 with 4D Aggregation  
**Models**: DistortionNet FLOAT32 & INT8
