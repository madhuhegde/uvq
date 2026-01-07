"""Utility functions for TFLite DistortionNet aggregation.

This module provides helper functions for splitting input patches and aggregating
outputs when using the 3-patch DistortionNet model.

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

import numpy as np


def split_patches_into_rows(patches: np.ndarray) -> list[np.ndarray]:
    """Split 9 input patches into 3 rows of 3 patches each.
    
    This function splits a 3x3 grid of patches into 3 rows, where each row
    contains 3 patches. This is used for sequential processing with the
    3-patch DistortionNet model.
    
    Args:
        patches: A numpy array of shape (9, H, W, C) representing 9 patches
                 arranged in a 3x3 grid. The patches are expected to be in
                 row-major order:
                 [0, 1, 2, 3, 4, 5, 6, 7, 8] corresponds to:
                 ┌─────┬─────┬─────┐
                 │  0  │  1  │  2  │  <- Row 1
                 ├─────┼─────┼─────┤
                 │  3  │  4  │  5  │  <- Row 2
                 ├─────┼─────┼─────┤
                 │  6  │  7  │  8  │  <- Row 3
                 └─────┴─────┴─────┘
    
    Returns:
        A list of 3 numpy arrays, each of shape (3, H, W, C), representing
        the patches for each row.
    
    Raises:
        ValueError: If the input does not have exactly 9 patches.
    
    Example:
        >>> patches = np.random.randn(9, 360, 640, 3)
        >>> rows = split_patches_into_rows(patches)
        >>> len(rows)
        3
        >>> rows[0].shape
        (3, 360, 640, 3)
    """
    if patches.shape[0] != 9:
        raise ValueError(f"Expected 9 patches, but got {patches.shape[0]}")
    
    row1_patches = patches[0:3]  # Patches 0, 1, 2
    row2_patches = patches[3:6]  # Patches 3, 4, 5
    row3_patches = patches[6:9]  # Patches 6, 7, 8
    
    return [row1_patches, row2_patches, row3_patches]


def aggregate_row_patches(row_patches: np.ndarray) -> np.ndarray:
    """Aggregate 3 patch features horizontally into a single row.
    
    This function takes 3 individual patch features and aggregates them
    horizontally using 4D operations to match PyTorch's aggregation logic.
    
    Args:
        row_patches: A numpy array of shape (3, 8, 8, 128) representing
                     3 patch features from a single row.
    
    Returns:
        A numpy array of shape (1, 8, 24, 128) representing the horizontally
        aggregated row features.
    
    Raises:
        ValueError: If the input does not have exactly 3 patches.
    
    Example:
        >>> patches = np.random.randn(3, 8, 8, 128)
        >>> row = aggregate_row_patches(patches)
        >>> row.shape
        (1, 8, 24, 128)
    """
    if row_patches.shape[0] != 3:
        raise ValueError(f"Expected 3 patches, but got {row_patches.shape[0]}")
    
    # Aggregate 3 patches horizontally using 4D operations
    # Input: [3, 8, 8, 128]
    # Output: [1, 8, 24, 128]
    
    # Transpose to [8, 3, 8, 128] - move patch dimension to middle
    features = np.transpose(row_patches, (1, 0, 2, 3))
    
    # Reshape to [1, 8, 24, 128] - flatten patch and width dimensions
    features = features.reshape(1, 8, 24, 128)
    
    return features


def aggregate_distortion_rows(row_outputs: list[np.ndarray]) -> np.ndarray:
    """Aggregate 3 row outputs from the 3-patch DistortionNet model into a single feature map.
    
    This function takes the outputs from 3 sequential calls to the 3-patch
    DistortionNet model (one per row) and concatenates them along the height
    dimension to form the final aggregated feature map.
    
    Args:
        row_outputs: A list of 3 numpy arrays, each of shape (1, 8, 24, 128),
                     representing the aggregated features for each row:
                     - Row 1: Features from patches 0, 1, 2
                     - Row 2: Features from patches 3, 4, 5
                     - Row 3: Features from patches 6, 7, 8
    
    Returns:
        A single numpy array of shape (1, 24, 24, 128) representing the
        fully aggregated feature map. The height dimension (24) comes from
        concatenating 3 rows of height 8 each.
    
    Raises:
        ValueError: If the number of row outputs is not exactly 3.
    
    Example:
        >>> row1 = np.random.randn(1, 8, 24, 128)
        >>> row2 = np.random.randn(1, 8, 24, 128)
        >>> row3 = np.random.randn(1, 8, 24, 128)
        >>> aggregated = aggregate_distortion_rows([row1, row2, row3])
        >>> aggregated.shape
        (1, 24, 24, 128)
    """
    if len(row_outputs) != 3:
        raise ValueError(f"Expected 3 row outputs, but got {len(row_outputs)}")
    
    # Concatenate along the height dimension (axis=1)
    # Each row output is (1, 8, 24, 128)
    # Concatenating 3 such outputs along axis=1 results in (1, 3*8, 24, 128) = (1, 24, 24, 128)
    aggregated_features = np.concatenate(row_outputs, axis=1)
    
    return aggregated_features

