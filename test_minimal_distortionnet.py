#!/usr/bin/env python3
"""Test minimal DistortionNet models for debugging TFLite issues."""

import argparse
import torch
import numpy as np
from uvq1p5_pytorch.utils.distortionnet_minimal import get_minimal_distortionnet
from uvq1p5_pytorch.utils import distortionnet

def test_model(size='minimal'):
    """Test a minimal DistortionNet model.
    
    Args:
        size: 'single', 'minimal', 'medium', or 'full'
    """
    print("\n" + "="*70)
    print(f"Testing {size.upper()} DistortionNet")
    print("="*70)
    
    # Create model
    if size == 'full':
        model = distortionnet.DistortionNetCore()
        print("Using full DistortionNet (18 layers)")
    else:
        model = get_minimal_distortionnet(size)
        print(f"Using minimal DistortionNet ({size})")
    
    model.eval()
    
    # Create sample input
    if size == 'single':
        # Single block expects 16 channels after initial processing
        x = torch.randn(1, 16, 180, 320)
        print(f"\nInput shape: {x.shape} (after initial conv)")
    else:
        # Full patch input
        x = torch.randn(1, 3, 360, 640)
        print(f"\nInput shape: {x.shape} (RGB patch)")
    
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(x)
    
    print(f"✓ Inference successful!")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel statistics:")
    print(f"  Parameters: {params:,}")
    print(f"  Size (MB): {params * 4 / (1024**2):.2f}")  # Assuming float32
    
    # Layer count
    if hasattr(model, 'features'):
        print(f"  Layers: {len(model.features)}")
    
    return model, output


def compare_models():
    """Compare outputs of different model sizes."""
    print("\n" + "="*70)
    print("Comparing Model Sizes")
    print("="*70)
    
    sizes = ['single', 'minimal', 'medium', 'full']
    results = {}
    
    for size in sizes:
        try:
            model, output = test_model(size)
            results[size] = {
                'params': sum(p.numel() for p in model.parameters()),
                'output_shape': output.shape,
                'success': True
            }
        except Exception as e:
            print(f"\n✗ {size} model failed: {e}")
            results[size] = {'success': False, 'error': str(e)}
    
    # Print comparison table
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"\n{'Model':<12} {'Parameters':<15} {'Output Shape':<20} {'Status':<10}")
    print("-" * 70)
    
    for size in sizes:
        if results[size]['success']:
            params = results[size]['params']
            shape = str(results[size]['output_shape'])
            status = "✓ OK"
        else:
            params = "N/A"
            shape = "N/A"
            status = "✗ FAILED"
        
        print(f"{size:<12} {str(params):<15} {shape:<20} {status:<10}")


def main():
    parser = argparse.ArgumentParser(
        description='Test minimal DistortionNet models for debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single MBConv block (isolate depthwise conv)
  python test_minimal_distortionnet.py --size single
  
  # Test minimal model (5 layers)
  python test_minimal_distortionnet.py --size minimal
  
  # Test medium model (10 layers)
  python test_minimal_distortionnet.py --size medium
  
  # Test full model (18 layers)
  python test_minimal_distortionnet.py --size full
  
  # Compare all models
  python test_minimal_distortionnet.py --compare
        """
    )
    
    parser.add_argument(
        '--size',
        choices=['single', 'minimal', 'medium', 'full'],
        default='minimal',
        help='Which model size to test (default: minimal)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all model sizes'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        test_model(args.size)
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == '__main__':
    main()

