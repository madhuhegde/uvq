#!/usr/bin/env python3
"""Test script to verify 3-patch DistortionNet integration.

This script compares the batch-9 and 3-patch DistortionNet implementations
to ensure they produce identical outputs.

Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0
"""

import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uvq1p5_pytorch.utils.uvq1p5_tflite import DistortionNetTFLite


def compare_batch9_vs_3patch():
    """Compare batch-9 and 3-patch DistortionNet implementations."""
    
    print("\n" + "="*70)
    print("Comparing Batch-9 vs 3-Patch DistortionNet Models")
    print("="*70)
    
    # Create sample input (9 patches)
    print("\n1. Creating test input...")
    np.random.seed(42)
    video_patches = np.random.randn(9, 360, 640, 3).astype(np.float32) * 2
    print(f"   Input shape: {video_patches.shape}")
    print(f"   Input range: [{video_patches.min():.2f}, {video_patches.max():.2f}]")
    
    # Test batch-9 model
    print("\n2. Testing batch-9 model...")
    try:
        model_batch9 = DistortionNetTFLite(use_3patch=False)
        output_batch9 = model_batch9(video_patches)
        print(f"   Output shape: {output_batch9.shape}")
        print(f"   Output range: [{output_batch9.min():.2f}, {output_batch9.max():.2f}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3-patch model
    print("\n3. Testing 3-patch model...")
    try:
        model_3patch = DistortionNetTFLite(use_3patch=True)
        output_3patch = model_3patch(video_patches)
        print(f"   Output shape: {output_3patch.shape}")
        print(f"   Output range: [{output_3patch.min():.2f}, {output_3patch.max():.2f}]")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Compare outputs
    print("\n4. Comparing outputs...")
    
    if output_batch9.shape != output_3patch.shape:
        print(f"   ✗ Shape mismatch! Batch-9: {output_batch9.shape}, 3-patch: {output_3patch.shape}")
        return False
    
    print(f"   ✓ Shapes match: {output_batch9.shape}")
    
    abs_diff = np.abs(output_batch9 - output_3patch)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Calculate correlation
    correlation = np.corrcoef(output_batch9.flatten(), output_3patch.flatten())[0, 1]
    
    print(f"\n   Absolute differences:")
    print(f"     Max:  {max_abs_diff:.6f}")
    print(f"     Mean: {mean_abs_diff:.6f}")
    
    print(f"\n   Correlation: {correlation:.8f}")
    
    # Check if results match
    if correlation >= 0.9999 and max_abs_diff < 0.1:
        print(f"\n   ✅ Excellent match (correlation >= 0.9999 and max_diff < 0.1)")
        return True
    elif correlation >= 0.999 and max_abs_diff < 1.0:
        print(f"\n   ✓ Good match (correlation >= 0.999 and max_diff < 1.0)")
        return True
    else:
        print(f"\n   ⚠️  Results differ more than expected")
        return False


def test_multiple_frames():
    """Test with multiple frames to ensure batching works correctly."""
    
    print("\n" + "="*70)
    print("Testing Multiple Frames")
    print("="*70)
    
    # Create sample input (3 frames, 9 patches each = 27 patches)
    print("\n1. Creating test input (3 frames)...")
    np.random.seed(123)
    video_patches = np.random.randn(27, 360, 640, 3).astype(np.float32) * 2
    print(f"   Input shape: {video_patches.shape}")
    print(f"   Number of frames: 3 (27 patches / 9 patches per frame)")
    
    # Test batch-9 model
    print("\n2. Testing batch-9 model...")
    try:
        model_batch9 = DistortionNetTFLite(use_3patch=False)
        output_batch9 = model_batch9(video_patches)
        print(f"   Output shape: {output_batch9.shape}")
        print(f"   Expected: (3, 24, 24, 128)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3-patch model
    print("\n3. Testing 3-patch model...")
    try:
        model_3patch = DistortionNetTFLite(use_3patch=True)
        output_3patch = model_3patch(video_patches)
        print(f"   Output shape: {output_3patch.shape}")
        print(f"   Expected: (3, 24, 24, 128)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Compare outputs
    print("\n4. Comparing outputs...")
    
    if output_batch9.shape != output_3patch.shape:
        print(f"   ✗ Shape mismatch! Batch-9: {output_batch9.shape}, 3-patch: {output_3patch.shape}")
        return False
    
    if output_batch9.shape != (3, 24, 24, 128):
        print(f"   ✗ Unexpected output shape: {output_batch9.shape}")
        return False
    
    abs_diff = np.abs(output_batch9 - output_3patch)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    print(f"   Max difference:  {max_abs_diff:.6f}")
    print(f"   Mean difference: {mean_abs_diff:.6f}")
    
    if max_abs_diff < 0.1:
        print(f"\n   ✅ Excellent match (max_diff < 0.1)")
        return True
    else:
        print(f"\n   ⚠️  Results differ more than expected")
        return False


def main():
    print("\n" + "="*70)
    print("3-Patch DistortionNet Integration Test")
    print("="*70)
    
    # Test 1: Single frame comparison
    test1_passed = compare_batch9_vs_3patch()
    
    # Test 2: Multiple frames
    test2_passed = test_multiple_frames()
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"Single frame test:   {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Multiple frames test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED!")
        return 1


if __name__ == '__main__':
    exit(main())

