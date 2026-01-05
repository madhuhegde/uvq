#!/usr/bin/env python3
"""
Compare performance between FLOAT32 and INT8 quantized TFLite models for UVQ 1.5.

This script:
1. Runs inference on a test video using both FLOAT32 and INT8 models
2. Measures inference time for each model
3. Compares quality scores and per-frame predictions
4. Reports model sizes and memory usage
5. Calculates speedup and accuracy metrics

Usage:
    python compare_tflite_performance.py <video_path> [--fps 1] [--iterations 3]
"""

import argparse
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import UVQ models
from uvq1p5_pytorch.utils.uvq1p5_tflite import UVQ1p5TFLite


def get_model_size(model_path):
    """Get model file size in MB."""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0


def get_video_info(video_path):
    """Get video duration and FPS using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,r_frame_rate',
            '-of', 'default=noprint_wrappers=1',
            video_path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        
        duration = None
        fps = None
        
        for line in output.split('\n'):
            if line.startswith('duration='):
                duration = float(line.split('=')[1])
            elif line.startswith('r_frame_rate='):
                fps_str = line.split('=')[1]
                num, den = map(int, fps_str.split('/'))
                fps = num / den
        
        return duration, fps
    except Exception as e:
        print(f"Warning: Could not get video info: {e}")
        return None, None


def run_inference(model, video_path, video_length, fps, num_iterations=1):
    """Run inference and measure time.
    
    Args:
        model: UVQ1p5TFLite model instance
        video_path: Path to video file
        video_length: Video duration in seconds
        fps: Frames per second to sample
        num_iterations: Number of times to run inference (for averaging)
    
    Returns:
        results: Inference results from last iteration
        avg_time: Average inference time in seconds
        times: List of individual inference times
    """
    times = []
    results = None
    
    for i in range(num_iterations):
        start_time = time.time()
        
        results = model.infer(
            video_filename=video_path,
            video_length=video_length,
            transpose=False,
            fps=fps,
            orig_fps=None,
            ffmpeg_path='ffmpeg'
        )
        
        end_time = time.time()
        inference_time = end_time - start_time
        times.append(inference_time)
        
        print(f"  Iteration {i+1}/{num_iterations}: {inference_time:.3f}s")
    
    avg_time = np.mean(times)
    return results, avg_time, times


def compare_results(float32_results, int8_results):
    """Compare inference results between FLOAT32 and INT8 models.
    
    Args:
        float32_results: Results from FLOAT32 model
        int8_results: Results from INT8 model
    
    Returns:
        comparison: Dictionary with comparison metrics
    """
    float32_score = float32_results['uvq1p5_score']
    int8_score = int8_results['uvq1p5_score']
    
    float32_frames = np.array(float32_results['per_frame_scores'])
    int8_frames = np.array(int8_results['per_frame_scores'])
    
    # Overall score difference
    score_diff = abs(float32_score - int8_score)
    score_rel_diff = (score_diff / float32_score) * 100
    
    # Per-frame differences
    frame_abs_diff = np.abs(float32_frames - int8_frames)
    frame_rel_diff = (frame_abs_diff / (np.abs(float32_frames) + 1e-8)) * 100
    
    comparison = {
        'overall_score_diff': score_diff,
        'overall_score_rel_diff': score_rel_diff,
        'frame_abs_diff_mean': frame_abs_diff.mean(),
        'frame_abs_diff_max': frame_abs_diff.max(),
        'frame_abs_diff_std': frame_abs_diff.std(),
        'frame_rel_diff_mean': frame_rel_diff.mean(),
        'frame_rel_diff_max': frame_rel_diff.max(),
        'frame_rel_diff_std': frame_rel_diff.std(),
    }
    
    return comparison


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def main():
    parser = argparse.ArgumentParser(
        description='Compare FLOAT32 vs INT8 TFLite models for UVQ 1.5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_tflite_performance.py /path/to/video.mp4
  
  # With custom FPS and multiple iterations
  python compare_tflite_performance.py /path/to/video.mp4 --fps 2 --iterations 5
  
  # Specify video length manually
  python compare_tflite_performance.py /path/to/video.mp4 --video_length 10
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the input video file'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second to sample for inference (default: 1)'
    )
    
    parser.add_argument(
        '--video_length',
        type=int,
        default=None,
        help='Video length in seconds (auto-detected if not provided)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='Number of inference iterations for averaging (default: 3)'
    )
    
    parser.add_argument(
        '--skip_float32',
        action='store_true',
        help='Skip FLOAT32 inference (only run INT8)'
    )
    
    parser.add_argument(
        '--skip_int8',
        action='store_true',
        help='Skip INT8 inference (only run FLOAT32)'
    )
    
    args = parser.parse_args()
    
    # Validate video path
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Get video info
    print_separator()
    print("UVQ 1.5 TFLite Performance Comparison: FLOAT32 vs INT8")
    print_separator()
    print(f"\nVideo: {args.video_path}")
    
    video_duration, video_fps = get_video_info(args.video_path)
    
    if args.video_length is None:
        if video_duration is not None:
            args.video_length = int(video_duration)
            print(f"Video duration: {args.video_length}s (auto-detected)")
        else:
            print("Error: Could not detect video duration. Please specify --video_length")
            return 1
    else:
        print(f"Video duration: {args.video_length}s (user-specified)")
    
    if video_fps is not None:
        print(f"Video FPS: {video_fps:.2f}")
    
    print(f"Sampling FPS: {args.fps}")
    print(f"Iterations: {args.iterations}")
    
    # Get model sizes
    base_dir = os.path.join(os.path.dirname(__file__), "models", "tflite_models", "uvq1.5")
    
    float32_sizes = {
        'content_net': get_model_size(os.path.join(base_dir, "content_net.tflite")),
        'distortion_net': get_model_size(os.path.join(base_dir, "distortion_net.tflite")),
        'aggregation_net': get_model_size(os.path.join(base_dir, "aggregation_net.tflite")),
    }
    
    int8_sizes = {
        'content_net': get_model_size(os.path.join(base_dir, "content_net_int8.tflite")),
        'distortion_net': get_model_size(os.path.join(base_dir, "distortion_net_int8.tflite")),
        'aggregation_net': get_model_size(os.path.join(base_dir, "aggregation_net_int8.tflite")),
    }
    
    float32_total = sum(float32_sizes.values())
    int8_total = sum(int8_sizes.values())
    size_reduction = (1 - int8_total / float32_total) * 100 if float32_total > 0 else 0
    
    print(f"\nModel sizes:")
    print(f"  FLOAT32: {float32_total:.2f} MB")
    print(f"  INT8:    {int8_total:.2f} MB ({size_reduction:.1f}% reduction)")
    
    # Run FLOAT32 inference
    float32_results = None
    float32_time = None
    
    if not args.skip_float32:
        print_separator()
        print("Running FLOAT32 inference...")
        print_separator()
        
        try:
            model_float32 = UVQ1p5TFLite(use_quantized=False)
            float32_results, float32_time, float32_times = run_inference(
                model_float32,
                args.video_path,
                args.video_length,
                args.fps,
                args.iterations
            )
            
            print(f"\nFLOAT32 Results:")
            print(f"  Average time: {float32_time:.3f}s")
            print(f"  Time std dev: {np.std(float32_times):.3f}s")
            print(f"  UVQ 1.5 score: {float32_results['uvq1p5_score']:.4f}")
            print(f"  Frames processed: {len(float32_results['per_frame_scores'])}")
            
        except Exception as e:
            print(f"\n✗ FLOAT32 inference failed: {e}")
            import traceback
            traceback.print_exc()
            if args.skip_int8:
                return 1
    
    # Run INT8 inference
    int8_results = None
    int8_time = None
    
    if not args.skip_int8:
        print_separator()
        print("Running INT8 inference...")
        print_separator()
        
        try:
            model_int8 = UVQ1p5TFLite(use_quantized=True)
            int8_results, int8_time, int8_times = run_inference(
                model_int8,
                args.video_path,
                args.video_length,
                args.fps,
                args.iterations
            )
            
            print(f"\nINT8 Results:")
            print(f"  Average time: {int8_time:.3f}s")
            print(f"  Time std dev: {np.std(int8_times):.3f}s")
            print(f"  UVQ 1.5 score: {int8_results['uvq1p5_score']:.4f}")
            print(f"  Frames processed: {len(int8_results['per_frame_scores'])}")
            
        except Exception as e:
            print(f"\n✗ INT8 inference failed: {e}")
            import traceback
            traceback.print_exc()
            if args.skip_float32:
                return 1
    
    # Compare results
    if float32_results is not None and int8_results is not None:
        print_separator()
        print("Comparison: FLOAT32 vs INT8")
        print_separator()
        
        comparison = compare_results(float32_results, int8_results)
        
        print(f"\nQuality Score Comparison:")
        print(f"  FLOAT32 score: {float32_results['uvq1p5_score']:.4f}")
        print(f"  INT8 score:    {int8_results['uvq1p5_score']:.4f}")
        print(f"  Absolute diff: {comparison['overall_score_diff']:.4f}")
        print(f"  Relative diff: {comparison['overall_score_rel_diff']:.2f}%")
        
        print(f"\nPer-Frame Score Differences:")
        print(f"  Mean abs diff: {comparison['frame_abs_diff_mean']:.4f}")
        print(f"  Max abs diff:  {comparison['frame_abs_diff_max']:.4f}")
        print(f"  Std abs diff:  {comparison['frame_abs_diff_std']:.4f}")
        print(f"  Mean rel diff: {comparison['frame_rel_diff_mean']:.2f}%")
        print(f"  Max rel diff:  {comparison['frame_rel_diff_max']:.2f}%")
        
        print(f"\nPerformance Comparison:")
        print(f"  FLOAT32 time: {float32_time:.3f}s")
        print(f"  INT8 time:    {int8_time:.3f}s")
        
        if float32_time > 0:
            speedup = float32_time / int8_time
            time_reduction = (1 - int8_time / float32_time) * 100
            print(f"  Speedup:      {speedup:.2f}x")
            print(f"  Time saved:   {time_reduction:.1f}%")
        
        print(f"\nModel Size Comparison:")
        print(f"  FLOAT32: {float32_total:.2f} MB")
        print(f"  INT8:    {int8_total:.2f} MB")
        print(f"  Reduction: {size_reduction:.1f}%")
        
        # Summary
        print_separator()
        print("Summary")
        print_separator()
        
        accuracy_status = "✓ Excellent" if comparison['overall_score_rel_diff'] < 5 else "⚠ Acceptable" if comparison['overall_score_rel_diff'] < 10 else "✗ Poor"
        performance_status = "✓ Faster" if int8_time < float32_time else "⚠ Slower"
        
        print(f"\nAccuracy: {accuracy_status}")
        print(f"  Overall score difference: {comparison['overall_score_rel_diff']:.2f}%")
        print(f"  Mean per-frame difference: {comparison['frame_rel_diff_mean']:.2f}%")
        
        print(f"\nPerformance: {performance_status}")
        if float32_time > 0:
            print(f"  Speedup: {speedup:.2f}x")
        print(f"  INT8 time: {int8_time:.3f}s")
        
        print(f"\nModel Size: ✓ {size_reduction:.1f}% smaller")
        print(f"  INT8 total: {int8_total:.2f} MB")
        
        # Recommendation
        print(f"\nRecommendation:")
        if comparison['overall_score_rel_diff'] < 5 and int8_time <= float32_time:
            print("  ✓ Use INT8 models for production deployment")
            print("    - Excellent accuracy preservation")
            print("    - Better or equal performance")
            print("    - Significantly smaller model size")
        elif comparison['overall_score_rel_diff'] < 10:
            print("  ⚠ INT8 models are acceptable for most use cases")
            print("    - Good accuracy preservation")
            print(f"    - {size_reduction:.1f}% smaller models")
            print("    - Consider accuracy requirements for your application")
        else:
            print("  ⚠ Carefully evaluate INT8 models for your use case")
            print("    - Significant accuracy difference detected")
            print("    - May not be suitable for high-precision applications")
    
    print_separator()
    print("Comparison complete!")
    print_separator()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

