# UVQ 1.5 TFLite Conversion & Quantization

Comprehensive documentation for UVQ 1.5 TFLite conversion, INT8 quantization, and performance analysis.

---

## 📚 Documentation Structure

### Core Documentation

| File | Description |
|:-----|:------------|
| **[overview.md](./overview.md)** | Project overview and architecture |
| **[implementation.md](./implementation.md)** | Complete implementation guide (PyTorch vs TFLite) |
| **[usage.md](./usage.md)** | How to use the models and comparison tools |

### Analysis & Results

| File | Description |
|:-----|:------------|
| **[results-summary.md](./results-summary.md)** | Presentation-ready tables (Performance, Accuracy, Model Size) |
| **[performance.md](./performance.md)** | Comprehensive performance analysis |
| **[quantization.md](./quantization.md)** | INT8 quantization details and process |
| **[model-inventory.md](./model-inventory.md)** | Model file inventory and locations |

### Input Format Migration

| File | Description |
|:-----|:------------|
| **[input-format-migration.md](./input-format-migration.md)** | Complete input format migration guide ([B, C, H, W] → [B, H, W, C]) with rationale, changes, test results, and benefits |

---

## 🎯 Quick Results

| Metric | FLOAT32 | INT8 | Improvement |
|:-------|--------:|-----:|:-----------:|
| **Model Size** | 29.38 MB | 8.63 MB | **70.6% smaller** |
| **Inference Speed** | 15.235s | 11.750s | **1.30x faster** |
| **Quality Score Diff** | Baseline | 2.44% | **Excellent** |

### ✅ Recommendation: Deploy INT8 models for production

---

## 📖 Reading Guide

### New to the Project?
1. Start with [overview.md](./overview.md) - understand the architecture
2. Check [results-summary.md](./results-summary.md) - see key metrics
3. Read [usage.md](./usage.md) - learn how to use the models

### Want to Implement?
1. Read [implementation.md](./implementation.md) - complete implementation guide
2. Check [usage.md](./usage.md) - how to run comparisons
3. Review [model-inventory.md](./model-inventory.md) - model file locations

### Need Detailed Analysis?
1. [performance.md](./performance.md) - comprehensive performance analysis
2. [quantization.md](./quantization.md) - INT8 quantization details
3. [results-summary.md](./results-summary.md) - presentation tables

### Understanding Input Format Changes?
1. [input-format-migration.md](./input-format-migration.md) - complete migration guide with rationale and results
2. [implementation.md](./implementation.md) - updated preprocessing code examples

---

## 🔗 Related Resources

### Implementation Files
- **Comparison Script:** `compare_tflite_performance.py`
- **Test Script:** `test_tflite_inference.py`
- **TFLite Implementation:** `uvq1p5_pytorch/utils/uvq1p5_tflite.py`
- **PyTorch Implementation:** `uvq1p5_pytorch/utils/uvq1p5.py`

### Model Locations
- **PyTorch Models:** `~/work/models/UVQ/uvq1.5/`
- **TFLite Models:** `models/tflite_models/uvq1.5/`

### Conversion Scripts
- **Location:** `~/work/ai_edge_torch/ai-edge-torch/ai_edge_torch/generative/examples/uvq1.5/`
- **FLOAT32 Conversion:** `convert_to_tflite.py`
- **INT8 Conversion:** `convert_to_tflite_int8.py`
- **Verification:** `verify_int8_models.py`

---

## 🛠️ Quick Commands

### Run Performance Comparison
```bash
source ~/work/UVQ/uvq_env/bin/activate
cd ~/work/UVQ/uvq
python compare_tflite_performance.py ../dataset/Gaming_360P_local.mp4 --fps 1 --iterations 5
```

### Test TFLite Inference
```bash
python test_tflite_inference.py video.mp4 --compare
```

### Check Model Sizes
```bash
ls -lh models/tflite_models/uvq1.5/
```

### Use INT8 Models in Code
```python
from uvq1p5_pytorch.utils.uvq1p5_tflite import UVQ1p5TFLite

# Load INT8 quantized models
model = UVQ1p5TFLite(use_quantized=True)

# Run inference
results = model.infer(
    video_filename='video.mp4',
    video_length=10,
    transpose=False,
    fps=1
)
```

---

## 📊 Documentation Summary

### What Each File Contains

**[overview.md](./overview.md)**
- Project introduction
- UVQ 1.5 architecture
- Key results summary
- Quick links

**[implementation.md](./implementation.md)**
- PyTorch vs TFLite implementations
- Model specifications
- Preprocessing details
- Code examples
- Deployment guides

**[usage.md](./usage.md)**
- Command-line options
- Usage examples
- Troubleshooting
- Integration guide

**[performance.md](./performance.md)**
- Detailed performance metrics
- Inference time analysis
- Accuracy comparison
- Hardware expectations
- ROI analysis

**[quantization.md](./quantization.md)**
- Quantization configuration
- FLOAT32 vs INT8 comparison
- Accuracy impact analysis
- Conversion process
- Alternative options

**[results-summary.md](./results-summary.md)**
- Performance table (speed)
- Accuracy table
- Model size table
- Presentation-ready format

**[model-inventory.md](./model-inventory.md)**
- File locations
- Model sizes
- Conversion scripts
- Documentation links

**[input-format-migration.md](./input-format-migration.md)**
- Rationale for input format change
- Complete migration summary
- All files modified
- Code before/after comparisons
- Test results with new format
- Performance benefits
- Verification checklist
- Deployment considerations

---

## 🎓 Key Concepts

### UVQ 1.5 Architecture
- **ContentNet:** Extracts semantic content features (256×256 input)
- **DistortionNet:** Detects distortions via patches (3×3 grid, 360×640 patches)
- **AggregationNet:** Combines features → quality score [1-5]

### Quantization Types
- **FLOAT32:** Full precision, 29.38 MB, baseline accuracy
- **INT8:** Dynamic quantization, 8.63 MB, 2.44% accuracy difference

### Performance Metrics
- **Speedup:** 1.30x faster with INT8
- **Size Reduction:** 70.6% smaller with INT8
- **Accuracy:** 2.44% difference (excellent)

---

## 📝 Document Conventions

Following the [documentation rules](./../rules/documentation.mdc):

- ✅ Kebab-case file names (`overview.md`, `performance.md`)
- ✅ Markdown format for all documentation
- ✅ Clear structure with tables and sections
- ✅ Cross-references between related files
- ✅ Code examples with syntax highlighting
- ✅ Organized in dedicated memory bank folder

---

## 🚀 Project Status

**Status:** ✅ **Production Ready**

- ✅ FLOAT32 models converted and tested
- ✅ INT8 models quantized and validated
- ✅ Performance benchmarks completed
- ✅ Accuracy verified (2.44% difference)
- ✅ Documentation complete
- ✅ Deployment-ready

---

**Last Updated:** January 5, 2026  
**Project:** UVQ 1.5 TFLite Conversion & INT8 Quantization  
**Recommendation:** Deploy INT8 models for production use
