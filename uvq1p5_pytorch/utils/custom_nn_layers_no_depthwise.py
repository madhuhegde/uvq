"""Modified custom neural network layers with depthwise convolution replaced by normal convolution.

This module is identical to custom_nn_layers.py except that MBConvSamePadding uses
normal Conv2D instead of depthwise Conv2D to avoid issues with TFLite conversion.

Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Import everything from the original module
from .custom_nn_layers import *

# Override only the MBConvSamePadding class
class MBConvSamePadding(nn.Module):
  """MobileNetV2/EfficientNet style inverted residual block with normal Conv2D.

  This is a modified version that replaces depthwise convolution with normal
  convolution to avoid TFLite conversion issues. The depthwise convolution
  (groups=expanded_channels) is replaced with a normal convolution (groups=1).
  
  Note: This will increase the model size and computational cost, but may
  resolve compatibility issues with certain TFLite delegates or hardware.
  """

  def __init__(
      self,
      input_channels: int,
      expand_ratio: int,
      out_channels: int,
      kernel: int,
      stride: int,
      stochastic_depth_prob: float,
      norm_layer: Callable[..., nn.Module] = contentnet_default_batchnorm2d,
      se_layer: Callable[..., nn.Module] = SqueezeExcitation,
  ) -> None:
    super().__init__()

    if not (1 <= stride <= 2):
      raise ValueError("illegal stride value")

    self.use_res_connect = stride == 1 and input_channels == out_channels

    layers: List[nn.Module] = []
    activation_layer = nn.SiLU

    # expand
    expanded_channels = input_channels * expand_ratio
    if expanded_channels != input_channels:
      layers.append(
          Conv2dNormActivationSamePadding(
              input_channels,
              expanded_channels,
              kernel_size=1,
              stride=1,
              norm_layer=norm_layer,
              activation_layer=activation_layer,
          )
      )

    # MODIFIED: Use normal convolution instead of depthwise (groups=1 instead of groups=expanded_channels)
    layers.append(
        Conv2dNormActivationSamePadding(
            expanded_channels,
            expanded_channels,
            kernel_size=kernel,
            stride=stride,
            groups=1,  # Changed from groups=expanded_channels to groups=1
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
    )

    # squeeze and excitation
    squeeze_channels = max(1, input_channels // 4)
    layers.append(
        se_layer(
            expanded_channels,
            squeeze_channels,
            activation=partial(nn.SiLU, inplace=True),
        )
    )

    # project
    layers.append(
        Conv2dNormActivationSamePadding(
            expanded_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=None,
        )
    )

    self.block = nn.Sequential(*layers)
    self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
    self.out_channels = out_channels

  def forward(self, input):
    result = self.block(input)
    if self.use_res_connect:
      result = self.stochastic_depth(result)
      result += input
    return result

