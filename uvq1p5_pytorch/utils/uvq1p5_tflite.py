"""TFLite implementation of the UVQ1p5 model.

This module contains the UVQ1p5 TFLite model implementation, providing
functionality to load TFLite models and run inference for video quality assessment.

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

import os
import sys
import math
from typing import Any
import numpy as np

# Import TensorFlow Lite
try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not found. Please install: pip install tensorflow")
    raise

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utils'
        )
    )
)
import video_reader


class ContentNetTFLite:
    """TFLite implementation of ContentNet."""
    
    def __init__(self, model_path=None):
        """Initialize ContentNet TFLite model.
        
        Args:
            model_path: Path to the TFLite model file. If None, uses default path.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models", 
                "tflite_models", "uvq1.5", "content_net.tflite"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ContentNet TFLite model not found: {model_path}")
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Verify input/output shapes
        # Using TensorFlow format: [B, H, W, C]
        expected_input_shape = [1, 256, 256, 3]
        expected_output_shape = [1, 8, 8, 128]
        
        actual_input_shape = self.input_details[0]['shape'].tolist()
        actual_output_shape = self.output_details[0]['shape'].tolist()
        
        if actual_input_shape != expected_input_shape:
            raise ValueError(
                f"ContentNet input shape mismatch: expected {expected_input_shape}, "
                f"got {actual_input_shape}"
            )
        
        if actual_output_shape != expected_output_shape:
            raise ValueError(
                f"ContentNet output shape mismatch: expected {expected_output_shape}, "
                f"got {actual_output_shape}"
            )
    
    def __call__(self, video_frame):
        """Run inference on video frame.
        
        Args:
            video_frame: numpy array of shape (batch, 256, 256, 3) in range [-1, 1]
        
        Returns:
            features: numpy array of shape (batch, 8, 8, 128)
        """
        # Ensure input is float32
        video_frame = video_frame.astype(np.float32)
        
        # Process each frame in the batch
        batch_size = video_frame.shape[0]
        all_features = []
        
        for i in range(batch_size):
            frame = video_frame[i:i+1]  # Keep batch dimension
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], frame)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            features = self.interpreter.get_tensor(self.output_details[0]['index'])
            all_features.append(features)
        
        # Concatenate all features
        return np.concatenate(all_features, axis=0)


class DistortionNetTFLite:
    """TFLite implementation of DistortionNet."""
    
    def __init__(self, model_path=None, use_3patch=False):
        """Initialize DistortionNet TFLite model.
        
        Args:
            model_path: Path to the TFLite model file. If None, uses default path.
            use_3patch: If True, uses the 3-patch model and performs application-side aggregation.
                        Otherwise, uses the batch-9 model.
        """
        self.use_3patch = use_3patch
        if model_path is None:
            if self.use_3patch:
                model_name = "distortion_net_3patch.tflite"
            else:
                model_name = "distortion_net.tflite"
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models",
                "tflite_models", "uvq1.5", model_name
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DistortionNet TFLite model not found: {model_path}")
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Verify input/output shapes
        # Using TensorFlow format: [B, H, W, C]
        if self.use_3patch:
            expected_input_shape = [3, 360, 640, 3]
            expected_output_shape = [3, 8, 8, 128]  # Individual patches, not aggregated
        else:
            expected_input_shape = [9, 360, 640, 3]
            expected_output_shape = [1, 24, 24, 128]
        
        actual_input_shape = self.input_details[0]['shape'].tolist()
        actual_output_shape = self.output_details[0]['shape'].tolist()
        
        if actual_input_shape != expected_input_shape:
            raise ValueError(
                f"DistortionNet input shape mismatch: expected {expected_input_shape}, "
                f"got {actual_input_shape} for model {model_path}"
            )
        
        if actual_output_shape != expected_output_shape:
            raise ValueError(
                f"DistortionNet output shape mismatch: expected {expected_output_shape}, "
                f"got {actual_output_shape} for model {model_path}"
            )
    
    def __call__(self, video_patches):
        """Run inference on video patches.
        
        Args:
            video_patches: numpy array of shape (batch * 9, 360, 640, 3) in range [-1, 1]
                           For batch=9 model: processes all 9 patches at once
                           For 3-patch model: splits into 3 rows and aggregates
        
        Returns:
            features: numpy array of shape (batch, 24, 24, 128)
        """
        # Ensure input is float32
        video_patches = video_patches.astype(np.float32)
        
        if self.use_3patch:
            from .tflite_aggregation import split_patches_into_rows, aggregate_row_patches, aggregate_distortion_rows
            
            # Process each frame (9 patches per frame)
            num_patches = video_patches.shape[0]
            batch_size = num_patches // 9
            all_features = []
            
            for i in range(batch_size):
                # Get 9 patches for this frame
                patches = video_patches[i*9:(i+1)*9]
                
                # Split into 3 rows
                row_patches_list = split_patches_into_rows(patches)
                
                # Process each row
                row_outputs = []
                for row_patches in row_patches_list:
                    # Run inference - output is [3, 8, 8, 128] (individual patches)
                    self.interpreter.set_tensor(self.input_details[0]['index'], row_patches)
                    self.interpreter.invoke()
                    row_patch_features = self.interpreter.get_tensor(self.output_details[0]['index'])
                    
                    # Aggregate 3 patches horizontally using 4D operations
                    # [3, 8, 8, 128] -> [1, 8, 24, 128]
                    row_output = aggregate_row_patches(row_patch_features)
                    row_outputs.append(row_output)
                
                # Aggregate 3 rows vertically
                # 3 x [1, 8, 24, 128] -> [1, 24, 24, 128]
                features = aggregate_distortion_rows(row_outputs)
                all_features.append(features)
            
            # Concatenate all features
            return np.concatenate(all_features, axis=0)
        else:
            # Original batch-9 logic
            # Process each frame (9 patches per frame)
            num_patches = video_patches.shape[0]
            batch_size = num_patches // 9
            all_features = []
            
            for i in range(batch_size):
                # Get 9 patches for this frame
                patches = video_patches[i*9:(i+1)*9]
                
                # Set input tensor
                self.interpreter.set_tensor(self.input_details[0]['index'], patches)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output tensor
                features = self.interpreter.get_tensor(self.output_details[0]['index'])
                all_features.append(features)
            
            # Concatenate all features
            return np.concatenate(all_features, axis=0)


class AggregationNetTFLite:
    """TFLite implementation of AggregationNet."""
    
    def __init__(self, model_path=None):
        """Initialize AggregationNet TFLite model.
        
        Args:
            model_path: Path to the TFLite model file. If None, uses default path.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "models",
                "tflite_models", "uvq1.5", "aggregation_net.tflite"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AggregationNet TFLite model not found: {model_path}")
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Verify input/output shapes
        expected_input_shapes = [[1, 8, 8, 128], [1, 24, 24, 128]]
        expected_output_shape = [1, 1]
        
        if len(self.input_details) != 2:
            raise ValueError(
                f"AggregationNet should have 2 inputs, got {len(self.input_details)}"
            )
        
        actual_input_shapes = [
            self.input_details[0]['shape'].tolist(),
            self.input_details[1]['shape'].tolist()
        ]
        actual_output_shape = self.output_details[0]['shape'].tolist()
        
        if actual_input_shapes != expected_input_shapes:
            raise ValueError(
                f"AggregationNet input shapes mismatch: expected {expected_input_shapes}, "
                f"got {actual_input_shapes}"
            )
        
        if actual_output_shape != expected_output_shape:
            raise ValueError(
                f"AggregationNet output shape mismatch: expected {expected_output_shape}, "
                f"got {actual_output_shape}"
            )
    
    def __call__(self, content_features, distortion_features):
        """Run inference on content and distortion features.
        
        Args:
            content_features: numpy array of shape (batch, 8, 8, 128)
            distortion_features: numpy array of shape (batch, 24, 24, 128)
        
        Returns:
            quality_scores: numpy array of shape (batch, 1) in range [1, 5]
        """
        # Ensure inputs are float32
        content_features = content_features.astype(np.float32)
        distortion_features = distortion_features.astype(np.float32)
        
        # Process each frame in the batch
        batch_size = content_features.shape[0]
        all_scores = []
        
        for i in range(batch_size):
            content = content_features[i:i+1]
            distortion = distortion_features[i:i+1]
            
            # Set input tensors
            self.interpreter.set_tensor(self.input_details[0]['index'], content)
            self.interpreter.set_tensor(self.input_details[1]['index'], distortion)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            score = self.interpreter.get_tensor(self.output_details[0]['index'])
            all_scores.append(score)
        
        # Concatenate all scores
        return np.concatenate(all_scores, axis=0)


class UVQ1p5TFLite:
    """TFLite implementation of UVQ 1.5 model."""
    
    def __init__(
        self,
        content_model_path=None,
        distortion_model_path=None,
        aggregation_model_path=None,
        use_quantized=False,
        use_3patch_distortion=False
    ):
        """Initialize UVQ 1.5 TFLite model.
        
        Args:
            content_model_path: Path to ContentNet TFLite model
            distortion_model_path: Path to DistortionNet TFLite model
            aggregation_model_path: Path to AggregationNet TFLite model
            use_quantized: If True, use INT8 quantized models (_int8.tflite)
            use_3patch_distortion: If True, use 3-patch DistortionNet model with
                                   application-side aggregation (lower memory usage)
        """
        self.use_quantized = use_quantized
        self.use_3patch_distortion = use_3patch_distortion
        
        # Auto-detect quantized models if use_quantized=True and paths not provided
        if use_quantized:
            if content_model_path is None:
                content_model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "models", 
                    "tflite_models", "uvq1.5", "content_net_int8.tflite"
                )
            if distortion_model_path is None:
                distortion_model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "models",
                    "tflite_models", "uvq1.5", "distortion_net_int8.tflite"
                )
            if aggregation_model_path is None:
                aggregation_model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "models",
                    "tflite_models", "uvq1.5", "aggregation_net_int8.tflite"
                )
        
        model_type = "INT8 Quantized" if use_quantized else "FLOAT32"
        distortion_type = "3-patch" if use_3patch_distortion else "batch-9"
        print(f"Loading UVQ 1.5 TFLite models ({model_type}, DistortionNet: {distortion_type})...")
        
        self.content_net = ContentNetTFLite(content_model_path)
        print("  ✓ ContentNet loaded")
        
        self.distortion_net = DistortionNetTFLite(distortion_model_path, use_3patch=use_3patch_distortion)
        print(f"  ✓ DistortionNet loaded ({distortion_type})")
        
        self.aggregation_net = AggregationNetTFLite(aggregation_model_path)
        print("  ✓ AggregationNet loaded")
        
        print(f"UVQ 1.5 TFLite models ready! ({model_type}, DistortionNet: {distortion_type})")
    
    def preprocess_frame_for_content(self, frame):
        """Preprocess frame for ContentNet (resize to 256x256).
        
        Args:
            frame: numpy array of shape (height, width, 3) in range [-1, 1]
        
        Returns:
            preprocessed: numpy array of shape (1, 256, 256, 3) in [B, H, W, C] format
        """
        import cv2
        
        # Resize to 256x256
        frame_256 = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_CUBIC)
        
        # Add batch dimension (no transpose needed - already in H, W, C format)
        frame_256 = np.expand_dims(frame_256, axis=0)
        
        return frame_256.astype(np.float32)
    
    def preprocess_frame_for_distortion(self, frame):
        """Preprocess frame for DistortionNet (split into 3x3 patches).
        
        Args:
            frame: numpy array of shape (1080, 1920, 3) in range [-1, 1]
        
        Returns:
            patches: numpy array of shape (9, 360, 640, 3) in [B, H, W, C] format
        """
        # Split into 3x3 patches of 360x640
        patches = []
        for i in range(3):
            for j in range(3):
                # Extract patch (already in H, W, C format - no transpose needed)
                patch = frame[i*360:(i+1)*360, j*640:(j+1)*640, :]
                patches.append(patch)
        
        # Stack patches
        patches = np.stack(patches, axis=0)
        return patches.astype(np.float32)
    
    def infer(
        self,
        video_filename: str,
        video_length: int,
        transpose: bool,
        fps: int = 1,
        orig_fps: float | None = None,
        ffmpeg_path: str = "ffmpeg",
    ) -> dict[str, Any]:
        """Run UVQ 1.5 inference on a video file using TFLite models.
        
        Args:
            video_filename: Path to the video file.
            video_length: Length of the video in seconds.
            transpose: Whether to transpose the video.
            fps: Frames per second to sample for inference.
            orig_fps: Original frames per second of the video, used for frame index
                calculation.
            ffmpeg_path: Path to ffmpeg executable.
        
        Returns:
            A dictionary containing the overall UVQ 1.5 score, per-frame scores,
            and frame indices.
        """
        # Load video
        video_1080p, num_real_frames = self.load_video(
            video_filename,
            video_length,
            transpose,
            fps=fps,
            ffmpeg_path=ffmpeg_path,
        )
        
        # video_1080p shape: (num_seconds, fps, height=1080, width=1920, channels=3)
        num_seconds, read_fps, h, w, c = video_1080p.shape
        num_frames = num_seconds * read_fps
        
        # Process frames
        predictions = []
        
        for frame_idx in range(num_frames):
            # Get frame (1080, 1920, 3)
            sec_idx = frame_idx // read_fps
            fps_idx = frame_idx % read_fps
            frame = video_1080p[sec_idx, fps_idx]
            
            # Preprocess for ContentNet (256x256)
            frame_256 = self.preprocess_frame_for_content(frame)
            
            # Preprocess for DistortionNet (9 patches)
            patches = self.preprocess_frame_for_distortion(frame)
            
            # Run ContentNet
            content_features = self.content_net(frame_256)
            
            # Run DistortionNet
            distortion_features = self.distortion_net(patches)
            
            # Run AggregationNet
            quality_score = self.aggregation_net(content_features, distortion_features)
            
            predictions.append(quality_score[0, 0])
        
        # Calculate overall score
        predictions = np.array(predictions)
        video_score = float(np.mean(predictions))
        frame_scores = predictions.tolist()
        
        # Calculate frame indices
        if orig_fps:
            frame_indices = [
                int(round(i * orig_fps / fps)) for i in range(len(frame_scores))
            ]
        else:
            frame_indices = list(range(len(frame_scores)))
        
        return {
            "uvq1p5_score": video_score,
            "per_frame_scores": frame_scores,
            "frame_indices": frame_indices,
        }
    
    def load_video(
        self,
        video_filename: str,
        video_length: int,
        transpose: bool = False,
        fps: int = 1,
        ffmpeg_path: str = "ffmpeg",
    ) -> tuple[np.ndarray, int]:
        """Load and preprocess a video for UVQ 1.5 inference.
        
        Args:
            video_filename: Path to the video file.
            video_length: Length of the video in seconds.
            transpose: Whether to transpose the video.
            fps: Frames per second to sample.
            ffmpeg_path: Path to ffmpeg executable.
        
        Returns:
            A tuple containing the loaded video as a numpy array and the number of
            real frames.
        """
        video, num_real_frames = video_reader.load_video_1p5(
            video_filename,
            video_length,
            transpose,
            video_fps=fps,
            video_height=1080,
            video_width=1920,
            ffmpeg_path=ffmpeg_path,
        )
        return video, num_real_frames

