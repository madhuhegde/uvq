#!/usr/bin/env python3
"""Simple test to compare batch-9 and 9-patch TFLite models on video."""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import probe
from uvq1p5_pytorch.utils import uvq1p5_tflite


def test_models(video_path):
    """Test batch-9 and 9-patch models on a video."""
    print("=" * 70)
    print(f"Testing: {video_path}")
    print("=" * 70)
    
    # Get video properties
    duration = probe.get_video_duration(video_path)
    video_length = math.ceil(duration)
    orig_fps = probe.get_r_frame_rate(video_path)
    
    print(f"\nVideo duration: {duration:.2f}s")
    print(f"FPS: {orig_fps}")
    
    # Test batch-9 model
    print("\n1. Testing batch-9 model...")
    model_batch9 = uvq1p5_tflite.UVQ1p5TFLite(
        use_3patch_distortion=False,
        use_9patch_distortion=False
    )
    results_batch9 = model_batch9.infer(
        video_path, video_length, transpose=False, fps=1, orig_fps=orig_fps
    )
    score_batch9 = results_batch9['uvq1p5_score']
    frame_scores_batch9 = results_batch9['per_frame_scores']
    print(f"   Score: {score_batch9:.6f}")
    print(f"   Frame scores (first 5): {[f'{s:.4f}' for s in frame_scores_batch9[:5]]}")
    
    # Test 9-patch model
    print("\n2. Testing 9-patch model...")
    model_9patch = uvq1p5_tflite.UVQ1p5TFLite(
        use_3patch_distortion=False,
        use_9patch_distortion=True
    )
    results_9patch = model_9patch.infer(
        video_path, video_length, transpose=False, fps=1, orig_fps=orig_fps
    )
    score_9patch = results_9patch['uvq1p5_score']
    frame_scores_9patch = results_9patch['per_frame_scores']
    print(f"   Score: {score_9patch:.6f}")
    print(f"   Frame scores (first 5): {[f'{s:.4f}' for s in frame_scores_9patch[:5]]}")
    
    # Compare
    print("\n" + "=" * 70)
    print("Comparison:")
    print("=" * 70)
    
    diff = abs(score_batch9 - score_9patch)
    print(f"\nOverall score difference: {diff:.6f}")
    
    # Frame-by-frame comparison
    import numpy as np
    frame_diff = np.abs(np.array(frame_scores_batch9) - np.array(frame_scores_9patch))
    max_frame_diff = np.max(frame_diff)
    mean_frame_diff = np.mean(frame_diff)
    
    print(f"Max frame difference:     {max_frame_diff:.6f}")
    print(f"Mean frame difference:    {mean_frame_diff:.6f}")
    
    # Correlation
    corr = np.corrcoef(frame_scores_batch9, frame_scores_9patch)[0, 1]
    print(f"Correlation:              {corr:.10f}")
    
    # Verdict
    tolerance = 0.001  # 0.1% tolerance
    print("\n" + "=" * 70)
    if diff < tolerance and max_frame_diff < tolerance:
        print(f"✅ PASS: 9-patch model matches batch-9 model!")
        print(f"   (difference: {diff:.6f} < {tolerance})")
    else:
        print(f"❌ FAIL: Models differ")
        print(f"   (difference: {diff:.6f} >= {tolerance})")
    print("=" * 70)


def main():
    videos = [
        "/home/madhuhegde/work/UVQ/dataset/Gaming_360P_local.mp4",
        "/home/madhuhegde/work/UVQ/dataset/Gaming_1080P-0ce6_orig.mp4",
    ]
    
    for video_path in videos:
        if os.path.exists(video_path):
            test_models(video_path)
            print("\n")
        else:
            print(f"⚠️  Video not found: {video_path}\n")


if __name__ == '__main__':
    main()

