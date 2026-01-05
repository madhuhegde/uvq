#!/usr/bin/env python3
"""Test script to compare PyTorch and TFLite UVQ 1.5 implementations.

Copyright 2025 Google LLC
Licensed under the Apache License, Version 2.0
"""

import argparse
import math
import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import probe
from uvq1p5_pytorch.utils import uvq1p5
from uvq1p5_pytorch.utils import uvq1p5_tflite


def test_inference(video_path, use_tflite=False):
    """Test inference on a video file.
    
    Args:
        video_path: Path to video file
        use_tflite: If True, use TFLite implementation; otherwise use PyTorch
    """
    print("\n" + "="*70)
    if use_tflite:
        print("Testing UVQ 1.5 TFLite Implementation")
    else:
        print("Testing UVQ 1.5 PyTorch Implementation")
    print("="*70)
    
    # Get video properties
    duration = probe.get_video_duration(video_path)
    if duration is None:
        print(f"Could not get duration for {video_path}")
        return None
    
    video_length = math.ceil(duration)
    orig_fps = probe.get_r_frame_rate(video_path)
    
    print(f"\nVideo: {video_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Length (ceil): {video_length}s")
    print(f"FPS: {orig_fps}")
    
    # Create model
    if use_tflite:
        model = uvq1p5_tflite.UVQ1p5TFLite()
    else:
        import torch
        model = uvq1p5.UVQ1p5()
        if torch.cuda.is_available():
            print("Using CUDA")
            model.cuda()
    
    # Run inference
    print("\nRunning inference...")
    try:
        results = model.infer(
            video_path,
            video_length,
            transpose=False,
            fps=1,
            orig_fps=orig_fps,
        )
        
        print("\n" + "="*70)
        print("Results:")
        print("="*70)
        print(f"UVQ 1.5 Score: {results['uvq1p5_score']:.4f}")
        print(f"Number of frames: {len(results['per_frame_scores'])}")
        print(f"Frame scores (first 5): {[f'{s:.4f}' for s in results['per_frame_scores'][:5]]}")
        print(f"Frame indices (first 5): {results['frame_indices'][:5]}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_implementations(video_path):
    """Compare PyTorch and TFLite implementations.
    
    Args:
        video_path: Path to video file
    """
    print("\n" + "="*70)
    print("Comparing PyTorch and TFLite Implementations")
    print("="*70)
    
    # Run PyTorch inference
    pytorch_results = test_inference(video_path, use_tflite=False)
    
    # Run TFLite inference
    tflite_results = test_inference(video_path, use_tflite=True)
    
    # Compare results
    if pytorch_results and tflite_results:
        print("\n" + "="*70)
        print("Comparison:")
        print("="*70)
        
        pytorch_score = pytorch_results['uvq1p5_score']
        tflite_score = tflite_results['uvq1p5_score']
        
        print(f"PyTorch Score: {pytorch_score:.4f}")
        print(f"TFLite Score:  {tflite_score:.4f}")
        print(f"Difference:    {abs(pytorch_score - tflite_score):.4f}")
        
        # Compare per-frame scores
        pytorch_frames = pytorch_results['per_frame_scores']
        tflite_frames = tflite_results['per_frame_scores']
        
        if len(pytorch_frames) == len(tflite_frames):
            import numpy as np
            diffs = np.abs(np.array(pytorch_frames) - np.array(tflite_frames))
            print(f"\nPer-frame differences:")
            print(f"  Mean: {np.mean(diffs):.4f}")
            print(f"  Max:  {np.max(diffs):.4f}")
            print(f"  Min:  {np.min(diffs):.4f}")
            
            if np.max(diffs) < 0.1:
                print("\n✓ Results match closely (max diff < 0.1)")
            elif np.max(diffs) < 0.5:
                print("\n⚠ Results have small differences (max diff < 0.5)")
            else:
                print("\n✗ Results have significant differences")
        else:
            print(f"\n✗ Frame count mismatch: PyTorch={len(pytorch_frames)}, TFLite={len(tflite_frames)}")


def main():
    parser = argparse.ArgumentParser(
        description='Test UVQ 1.5 TFLite implementation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test TFLite only
  python test_tflite_inference.py video.mp4 --tflite
  
  # Test PyTorch only
  python test_tflite_inference.py video.mp4 --pytorch
  
  # Compare both implementations
  python test_tflite_inference.py video.mp4 --compare
        """
    )
    
    parser.add_argument(
        'video',
        type=str,
        help='Path to video file'
    )
    
    parser.add_argument(
        '--tflite',
        action='store_true',
        help='Test TFLite implementation'
    )
    
    parser.add_argument(
        '--pytorch',
        action='store_true',
        help='Test PyTorch implementation'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare PyTorch and TFLite implementations'
    )
    
    args = parser.parse_args()
    
    # Default to compare if no option specified
    if not (args.tflite or args.pytorch or args.compare):
        args.compare = True
    
    if args.compare:
        compare_implementations(args.video)
    elif args.tflite:
        test_inference(args.video, use_tflite=True)
    elif args.pytorch:
        test_inference(args.video, use_tflite=False)


if __name__ == '__main__':
    main()

