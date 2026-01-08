#!/usr/bin/env python3
"""Test 9-patch INT8 DistortionNet model against PyTorch on 1080P video."""

import sys
import os
import math
sys.path.insert(0, '.')

from utils import probe
from uvq1p5_pytorch.utils import uvq1p5_tflite

video_path = '/home/madhuhegde/work/UVQ/dataset/Gaming_1080P-0ce6_orig.mp4'

print('=' * 70)
print('Testing 9-patch INT8 model on 1080P video')
print('=' * 70)

# Get video properties
duration = probe.get_video_duration(video_path)
video_length = math.ceil(duration)  # Process entire video
orig_fps = probe.get_r_frame_rate(video_path)

print(f'\nVideo: {os.path.basename(video_path)}')
print(f'Duration: {duration:.2f}s')
print(f'Processing: {video_length}s (full video)')
print(f'FPS: {orig_fps}')

# Test FLOAT32 9-patch model
print('\n1. Testing FLOAT32 9-patch model...')
model_float32 = uvq1p5_tflite.UVQ1p5TFLite(
    use_3patch_distortion=False,
    use_9patch_distortion=True,
    use_quantized=False
)

results_float32 = model_float32.infer(
    video_path, video_length, transpose=False, fps=1, orig_fps=orig_fps
)

score_float32 = results_float32['uvq1p5_score']
frame_scores_float32 = results_float32['per_frame_scores']
num_frames = len(frame_scores_float32)

print(f'   Score: {score_float32:.6f}')
print(f'   Frames: {num_frames}')

# Test INT8 9-patch model
print('\n2. Testing INT8 9-patch model...')
model_int8 = uvq1p5_tflite.UVQ1p5TFLite(
    use_3patch_distortion=False,
    use_9patch_distortion=True,
    use_quantized=True
)

results_int8 = model_int8.infer(
    video_path, video_length, transpose=False, fps=1, orig_fps=orig_fps
)

score_int8 = results_int8['uvq1p5_score']
frame_scores_int8 = results_int8['per_frame_scores']

print(f'   Score: {score_int8:.6f}')
print(f'   Frames: {len(frame_scores_int8)}')

# Compare
print('\n' + '=' * 70)
print('Comparison:')
print('=' * 70)

score_diff = abs(score_float32 - score_int8)
print(f'\nOverall scores:')
print(f'  FLOAT32: {score_float32:.6f}')
print(f'  INT8:    {score_int8:.6f}')
print(f'  Difference: {score_diff:.6f}')

# Frame-by-frame comparison
import numpy as np
frame_diff = np.abs(np.array(frame_scores_float32) - np.array(frame_scores_int8))
max_frame_diff = np.max(frame_diff)
mean_frame_diff = np.mean(frame_diff)

print(f'\nPer-frame comparison:')
print(f'  Max difference:  {max_frame_diff:.6f}')
print(f'  Mean difference: {mean_frame_diff:.6f}')

# Correlation
corr = np.corrcoef(frame_scores_float32, frame_scores_int8)[0, 1]
print(f'  Correlation:     {corr:.10f}')

# Sample frame scores
print(f'\nSample frame scores (first 5):')
print(f'  FLOAT32: {[f"{s:.4f}" for s in frame_scores_float32[:5]]}')
print(f'  INT8:    {[f"{s:.4f}" for s in frame_scores_int8[:5]]}')

# Verdict
print('\n' + '=' * 70)
print('Verdict:')
print('=' * 70)

tolerance = 0.1  # 0.1 tolerance for INT8
if score_diff < tolerance and corr > 0.95:
    print(f'✅ INT8 model performs well!')
    print(f'   Score difference: {score_diff:.6f} < {tolerance}')
    print(f'   Correlation: {corr:.6f} > 0.95')
else:
    print(f'⚠️  INT8 model shows some degradation')
    print(f'   Score difference: {score_diff:.6f}')
    print(f'   Correlation: {corr:.6f}')
    if score_diff < 0.2:
        print(f'   Still acceptable for most use cases')

print('\n' + '=' * 70)

