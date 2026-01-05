# TFLite Performance Comparison Script

Quick guide for using `compare_tflite_performance.py` to compare FLOAT32 and INT8 quantized TFLite models.

## Quick Start

```bash
# Activate environment
source ~/work/UVQ/uvq_env/bin/activate

# Basic comparison
cd ~/work/UVQ/uvq
python compare_tflite_performance.py /path/to/video.mp4

# With custom settings
python compare_tflite_performance.py /path/to/video.mp4 --fps 2 --iterations 5
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `video_path` | Path to input video (required) | - |
| `--fps N` | Frames per second to sample | 1 |
| `--video_length N` | Video length in seconds | Auto-detected |
| `--iterations N` | Number of inference runs | 3 |
| `--skip_float32` | Skip FLOAT32 inference | False |
| `--skip_int8` | Skip INT8 inference | False |

## Examples

### Basic Comparison
```bash
python compare_tflite_performance.py video.mp4
```

### High-Frequency Sampling
```bash
python compare_tflite_performance.py video.mp4 --fps 5
```

### Multiple Iterations for Accuracy
```bash
python compare_tflite_performance.py video.mp4 --iterations 10
```

### Test Only INT8 Model
```bash
python compare_tflite_performance.py video.mp4 --skip_float32
```

### Manual Video Length
```bash
python compare_tflite_performance.py video.mp4 --video_length 30
```

## Output

The script provides:

1. **Model Information**
   - Model sizes (FLOAT32 vs INT8)
   - Size reduction percentage

2. **Performance Metrics**
   - Inference time per iteration
   - Average inference time
   - Standard deviation
   - Speedup factor
   - Time saved percentage

3. **Accuracy Metrics**
   - Overall quality score comparison
   - Per-frame score differences (mean, max, std)
   - Absolute and relative differences

4. **Recommendation**
   - Clear guidance on which model to use
   - Based on accuracy and performance trade-offs

## Sample Output

```
================================================================================
UVQ 1.5 TFLite Performance Comparison: FLOAT32 vs INT8
================================================================================

Video: Gaming_360P_local.mp4
Video duration: 19s (auto-detected)
Video FPS: 30.00
Sampling FPS: 1
Iterations: 2

Model sizes:
  FLOAT32: 29.38 MB
  INT8:    8.63 MB (70.6% reduction)

...

Summary
================================================================================

Accuracy: ✓ Excellent
  Overall score difference: 2.44%
  Mean per-frame difference: 2.85%

Performance: ✓ Faster
  Speedup: 1.30x
  INT8 time: 11.750s

Model Size: ✓ 70.6% smaller
  INT8 total: 8.63 MB

Recommendation:
  ✓ Use INT8 models for production deployment
    - Excellent accuracy preservation
    - Better or equal performance
    - Significantly smaller model size
```

## Requirements

- Python 3.x
- TensorFlow 2.18+
- OpenCV (cv2)
- NumPy
- ffmpeg and ffprobe (for video processing)

## Troubleshooting

### Video duration not detected
```bash
# Manually specify video length
python compare_tflite_performance.py video.mp4 --video_length 30
```

### Out of memory
```bash
# Reduce sampling rate
python compare_tflite_performance.py video.mp4 --fps 1
```

### Models not found
```bash
# Check models exist
ls -lh ~/work/UVQ/uvq/models/tflite_models/uvq1.5/
```

## Integration

Use the modified `UVQ1p5TFLite` class in your code:

```python
from uvq1p5_pytorch.utils.uvq1p5_tflite import UVQ1p5TFLite

# Use INT8 quantized models
model = UVQ1p5TFLite(use_quantized=True)

# Or use FLOAT32 models
model = UVQ1p5TFLite(use_quantized=False)

# Run inference (same API for both)
results = model.infer(
    video_filename='video.mp4',
    video_length=10,
    transpose=False,
    fps=1
)
```

## Performance Tips

1. **Use multiple iterations** (--iterations 5) for more accurate timing
2. **Test with representative videos** from your dataset
3. **Consider different FPS settings** to match your use case
4. **Run on target hardware** for realistic performance metrics

## Related Documentation

- **Performance Results:** `TFLITE_PERFORMANCE_COMPARISON.md`
- **Quantization Details:** `INT8_QUANTIZATION_SUMMARY.md`
- **Model Inventory:** `TFLITE_MODEL_INVENTORY.txt`
- **Quantization Analysis:** `QUANTIZATION_ANALYSIS.md`

## Support

For issues or questions:
1. Check model files exist in `models/tflite_models/uvq1.5/`
2. Verify video file is accessible and valid
3. Ensure all dependencies are installed
4. Check TensorFlow version (2.18+ recommended)

