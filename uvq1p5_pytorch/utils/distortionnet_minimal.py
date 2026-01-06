"""Minimal DistortionNet for debugging TFLite issues.

This is a simplified version with only the first few stages to help isolate
and debug TFLite conversion issues.

Copyright 2025 Google LLC
"""

import functools
import torch
from torch import nn
from . import custom_nn_layers

default_distortionnet_batchnorm2d = functools.partial(
    nn.BatchNorm2d, eps=0.001, momentum=0
)


class DistortionNetMinimal(nn.Module):
  """Minimal DistortionNet with only first 3 stages for debugging.
  
  This includes:
  - Stage 1: Initial Conv (3→32)
  - Stage 2: MBConv1 (32→16)
  - Stage 3: MBConv6 (16→24, 2 blocks)
  - Final Conv (24→128)
  - MaxPool + Permute
  
  Total: ~5 layers instead of 18
  """

  def __init__(self):
    super().__init__()
    
    self.features = nn.Sequential(
        # Stage 1: Initial convolution
        custom_nn_layers.Conv2dNormActivationSamePadding(
            3,
            32,
            kernel_size=3,
            stride=2,
            activation_layer=nn.SiLU,
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Stage 2: MBConv1 (expand_ratio=1, no expansion)
        custom_nn_layers.MBConvSamePadding(
            32,
            1,  # expand_ratio
            16,  # out_channels
            3,   # kernel
            1,   # stride
            0.0, # stochastic_depth_prob
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Stage 3: MBConv6 (expand_ratio=6) - Block 1
        custom_nn_layers.MBConvSamePadding(
            16,
            6,   # expand_ratio
            24,  # out_channels
            3,   # kernel
            2,   # stride
            0.0125,
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Stage 3: MBConv6 - Block 2
        custom_nn_layers.MBConvSamePadding(
            24,
            6,
            24,
            3,
            1,
            0.025,
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Final convolution (adjusted from 320→128 to 24→128)
        custom_nn_layers.Conv2dSamePadding(
            24, 128, kernel_size=2, stride=1, bias=False
        ),
    )
    
    self.avgpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=(5, 13), stride=1, padding=0),
        custom_nn_layers.PermuteLayerNHWC(),
    )

  def forward(self, x):
    x = self.features(x)
    features = self.avgpool(x)
    return features


class DistortionNetSingleBlock(nn.Module):
  """Single MBConv block for debugging - isolates depthwise convolution issue.
  
  This is the absolute minimal model:
  - Input: 16 channels
  - MBConv6: 16→24 (with depthwise conv)
  - Output: 24 channels in NHWC format
  """

  def __init__(self):
    super().__init__()
    
    # Just one MBConv6 block
    self.block = custom_nn_layers.MBConvSamePadding(
        16,
        6,   # expand_ratio (16 * 6 = 96 expanded channels)
        24,  # out_channels
        3,   # kernel
        1,   # stride
        0.0,
        norm_layer=default_distortionnet_batchnorm2d,
    )
    
    # Add permute layer to convert to NHWC format for TFLite
    self.permute = custom_nn_layers.PermuteLayerNHWC()

  def forward(self, x):
    x = self.block(x)
    x = self.permute(x)
    return x


class DistortionNetMedium(nn.Module):
  """Medium DistortionNet with first 5 stages for debugging.
  
  This includes:
  - Stages 1-3 (as in Minimal)
  - Stage 4: MBConv6 (24→40, 2 blocks)
  - Stage 5: MBConv6 (40→80, 3 blocks)
  - Final Conv (80→128)
  - MaxPool + Permute
  
  Total: ~10 layers instead of 18
  """

  def __init__(self):
    super().__init__()
    
    stochastic_depth_prob_step = 0.0125
    stochastic_depth_prob = [x * stochastic_depth_prob_step for x in range(16)]
    
    self.features = nn.Sequential(
        # Stage 1: Initial convolution
        custom_nn_layers.Conv2dNormActivationSamePadding(
            3, 32, kernel_size=3, stride=2,
            activation_layer=nn.SiLU,
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Stage 2: MBConv1
        custom_nn_layers.MBConvSamePadding(
            32, 1, 16, 3, 1, stochastic_depth_prob[0],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Stage 3: MBConv6 (16→24, 2 blocks)
        custom_nn_layers.MBConvSamePadding(
            16, 6, 24, 3, 2, stochastic_depth_prob[1],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            24, 6, 24, 3, 1, stochastic_depth_prob[2],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Stage 4: MBConv6 (24→40, 2 blocks)
        custom_nn_layers.MBConvSamePadding(
            24, 6, 40, 5, 2, stochastic_depth_prob[3],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            40, 6, 40, 5, 1, stochastic_depth_prob[4],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Stage 5: MBConv6 (40→80, 3 blocks)
        custom_nn_layers.MBConvSamePadding(
            40, 6, 80, 3, 2, stochastic_depth_prob[5],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            80, 6, 80, 3, 1, stochastic_depth_prob[6],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        custom_nn_layers.MBConvSamePadding(
            80, 6, 80, 3, 1, stochastic_depth_prob[7],
            norm_layer=default_distortionnet_batchnorm2d,
        ),
        
        # Final convolution (adjusted from 320→128 to 80→128)
        custom_nn_layers.Conv2dSamePadding(
            80, 128, kernel_size=2, stride=1, bias=False
        ),
    )
    
    self.avgpool = nn.Sequential(
        nn.MaxPool2d(kernel_size=(5, 13), stride=1, padding=0),
        custom_nn_layers.PermuteLayerNHWC(),
    )

  def forward(self, x):
    x = self.features(x)
    features = self.avgpool(x)
    return features


# Helper function to get the appropriate minimal model
def get_minimal_distortionnet(size='minimal'):
  """Get a minimal DistortionNet for debugging.
  
  Args:
      size: 'single' (1 block), 'minimal' (5 layers), 'medium' (10 layers)
  
  Returns:
      A minimal DistortionNet model
  """
  if size == 'single':
    return DistortionNetSingleBlock()
  elif size == 'minimal':
    return DistortionNetMinimal()
  elif size == 'medium':
    return DistortionNetMedium()
  else:
    raise ValueError(f"Unknown size: {size}. Choose 'single', 'minimal', or 'medium'")

