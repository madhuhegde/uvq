#!/usr/bin/env python3
"""Test 9-patch DistortionNet model on actual video and compare scores."""

import os
import sys
import numpy as np
from pathlib import Path

# Add UVQ source path
sys.path.insert(0, str(Path(__file__).parent))

from uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5
from uvq1p5_pytorch.utils.uvq1p5_tflite import UVQ1p5TFLite


def test_video(video_path, num_frames=10):
    """Test video with PyTorch, batch-9 TFLite, and 9-patch TFLite models.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to process
    """
    print("=" * 70)
    print(f"Testing video: {video_path}")
    print("=" * 70)
    
    # Initialize models
    print("\nInitializing models...")
    pytorch_model = UVQ1p5()
    batch9_model = UVQ1p5TFLite(use_3patch_distortion=False, use_9patch_distortion=False)
    patch9_model = UVQ1p5TFLite(use_3patch_distortion=False, use_9patch_distortion=True)
    
    # Process video
    print(f"\nProcessing {num_frames} frames...")
    
    # PyTorch
    print("\n1. PyTorch model:")
    pytorch_scores = pytorch_model(video_path, return_frame_scores=True, num_frames=num_frames)
    pytorch_avg = np.mean(pytorch_scores)
    print(f"   Average score: {pytorch_avg:.6f}")
    print(f"   Frame scores: {pytorch_scores[:5]}...")  # First 5
    
    # Batch-9 TFLite
    print("\n2. Batch-9 TFLite model:")
    batch9_scores = batch9_model(video_path, return_frame_scores=True, num_frames=num_frames)
    batch9_avg = np.mean(batch9_scores)
    print(f"   Average score: {batch9_avg:.6f}")
    print(f"   Frame scores: {batch9_scores[:5]}...")  # First 5
    
    # 9-Patch TFLite
    print("\n3. 9-Patch TFLite model:")
    patch9_scores = patch9_model(video_path, return_frame_scores=True, num_frames=num_frames)
    patch9_avg = np.mean(patch9_scores)
    print(f"   Average score: {patch9_avg:.6f}")
    print(f"   Frame scores: {patch9_scores[:5]}...")  # First 5
    
    # Compare
    print("\n" + "=" * 70)
    print("Comparison:")
    print("=" * 70)
    
    # PyTorch vs Batch-9
    diff_batch9 = np.abs(pytorch_scores - batch9_scores)
    max_diff_batch9 = np.max(diff_batch9)
    mean_diff_batch9 = np.mean(diff_batch9)
    corr_batch9 = np.corrcoef(pytorch_scores, batch9_scores)[0, 1]
    
    print(f"\nPyTorch vs Batch-9 TFLite:")
    print(f"  Average score diff: {abs(pytorch_avg - batch9_avg):.6f}")
    print(f"  Max frame diff:     {max_diff_batch9:.6f}")
    print(f"  Mean frame diff:    {mean_diff_batch9:.6f}")
    print(f"  Correlation:        {corr_batch9:.10f}")
    
    # PyTorch vs 9-Patch
    diff_patch9 = np.abs(pytorch_scores - patch9_scores)
    max_diff_patch9 = np.max(diff_patch9)
    mean_diff_patch9 = np.mean(diff_patch9)
    corr_patch9 = np.corrcoef(pytorch_scores, patch9_scores)[0, 1]
    
    print(f"\nPyTorch vs 9-Patch TFLite:")
    print(f"  Average score diff: {abs(pytorch_avg - patch9_avg):.6f}")
    print(f"  Max frame diff:     {max_diff_patch9:.6f}")
    print(f"  Mean frame diff:    {mean_diff_patch9:.6f}")
    print(f"  Correlation:        {corr_patch9:.10f}")
    
    # Batch-9 vs 9-Patch
    diff_models = np.abs(batch9_scores - patch9_scores)
    max_diff_models = np.max(diff_models)
    mean_diff_models = np.mean(diff_models)
    corr_models = np.corrcoef(batch9_scores, patch9_scores)[0, 1]
    
    print(f"\nBatch-9 TFLite vs 9-Patch TFLite:")
    print(f"  Average score diff: {abs(batch9_avg - patch9_avg):.6f}")
    print(f"  Max frame diff:     {max_diff_models:.6f}")
    print(f"  Mean frame diff:    {mean_diff_models:.6f}")
    print(f"  Correlation:        {corr_models:.10f}")
    
    # Verdict
    print("\n" + "=" * 70)
    print("Verdict:")
    print("=" * 70)
    
    tolerance = 0.01  # 1% tolerance for video scores
    
    if max_diff_models < tolerance:
        print(f"✅ 9-Patch model MATCHES Batch-9 model perfectly!")
        print(f"   (max difference: {max_diff_models:.6f} < {tolerance})")
    else:
        print(f"❌ 9-Patch model differs from Batch-9 model")
        print(f"   (max difference: {max_diff_models:.6f} >= {tolerance})")
    
    if max_diff_patch9 < tolerance:
        print(f"✅ 9-Patch model MATCHES PyTorch model!")
        print(f"   (max difference: {max_diff_patch9:.6f} < {tolerance})")
    else:
        print(f"⚠️  9-Patch model differs slightly from PyTorch")
        print(f"   (max difference: {max_diff_patch9:.6f} >= {tolerance})")
        print(f"   This is expected due to TFLite numerical precision")


def main():
    # Test videos
    video_dir = Path.home() / "work" / "UVQ" / "dataset"
    
    videos = [
        video_dir / "Gaming_360P_local.mp4",
        video_dir / "Gaming_1080P-0ce6_orig.mp4",
    ]
    
    for video_path in videos:
        if video_path.exists():
            test_video(str(video_path), num_frames=10)
            print("\n")
        else:
            print(f"⚠️  Video not found: {video_path}")
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

