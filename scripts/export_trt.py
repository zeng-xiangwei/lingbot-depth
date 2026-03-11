#!/usr/bin/env python3
"""
TensorRT Engine Build Script for LingBot-Depth Model
合并导出：Encoder + Decoder -> Single ONNX
"""
import os
os.environ['XFORMERS_DISABLED'] = '1'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import cv2
import numpy as np

# TensorRT imports
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available, will generate ONNX only")

from mdm.model.v2 import MDMModel


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess RGB image."""
    image_np = cv2.imread(image_path)
    if image_np is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    image_np = image_np.astype(np.float32) / 255.0
    return image_np


def load_depth(depth_path: str, scale: float = 1000.0) -> np.ndarray:
    """Load depth map from PNG (16-bit)."""
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"Failed to read depth: {depth_path}")
    depth_map = depth_map.astype(np.float32) / scale
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    return depth_map


def build_preprocess_model(model, dummy_input, onnx_path):
    """
    Step 1: Export preprocessing module (image normalization + depth processing)
    Input: image (1,3,H,W), depth (1,1,H,W)
    Output: image_14 (normalized), depth_14 (processed)
    """
    print(f"Exporting preprocessing model to ONNX: {onnx_path}")
    
    base_h, base_w = dummy_input['base_h'], dummy_input['base_w']
    
    # Preprocessing wrapper - separates image normalization and depth processing
    class PreprocessWrapper(nn.Module):
        """Preprocessing only wrapper"""
        
        def __init__(self, encoder, base_h, base_w):
            super().__init__()
            self.encoder = encoder
            self.base_h = base_h
            self.base_w = base_w
            # ImageNet normalization
            self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
        def forward(self, image, depth):
            # 直接使用 base_h 和 base_w 作为 token 数量
            token_rows = self.base_h
            token_cols = self.base_w
            max_tokens = 7180  # 固定的最大 token 数量
            
            # Image preprocessing
            image_14 = F.interpolate(image, (token_rows * 14, token_cols * 14), mode="bilinear", align_corners=False)
            image_14 = (image_14 - self.image_mean) / self.image_std
            
            # Depth preprocessing
            depth_14 = F.interpolate(depth, (token_rows * 14, token_cols * 14), mode="nearest")
            
            # Handle invalid depth values
            depth_14[torch.isinf(depth_14)] = 0.0
            depth_14[torch.isnan(depth_14)] = 0.0
            dmask_14 = (depth_14 > 0.01).detach()
            depth_14 = depth_14 * dmask_14.float()
            
            # Apply log transform
            depth_14 = torch.log(depth_14)
            depth_14[~dmask_14] = 0.0
            depth_14 = torch.nan_to_num(depth_14, nan=0.0, posinf=0.0, neginf=0.0)
            
            features = self.encoder.backbone.get_intermediate_layers_mae(
                x_img=image_14, 
                x_depth=depth_14, 
                n=self.encoder.intermediate_layers, 
                return_class_token=True)

            x = features[0][0]  # shape: (1, dynamic_tokens, 1024)
            cls_token = features[0][0]
            
            # 将动态 token 数量 padding 到固定的 max_tokens
            # x shape: (1, dynamic_tokens, 1024) -> (1, max_tokens, 1024)
            current_tokens = x.shape[1]
            if current_tokens < max_tokens:
                # Padding
                padding = torch.zeros(1, max_tokens - current_tokens, x.shape[2], 
                                     dtype=x.dtype, device=x.device)
                x = torch.cat([x, padding], dim=1)
                cls_token = torch.cat([cls_token, padding], dim=1)
            elif current_tokens > max_tokens:
                # 截断
                x = x[:, :max_tokens, :]
                cls_token = cls_token[:, :max_tokens, :]
            
            print(f"x shape: {x.shape}, cls_token shape: {cls_token.shape}")
            return x, cls_token
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper = PreprocessWrapper(model.encoder, base_h, base_w)
    wrapper.to(device)
    wrapper.eval()
    
    # Dummy inputs
    dummy_image = torch.randn(1, 3, 480, 640, dtype=torch.float32, device=device)
    dummy_depth = torch.randn(1, 1, 480, 640, dtype=torch.float32, device=device)
    
    print(f"Exporting preprocessing with input shape: image={dummy_image.shape}, depth={dummy_depth.shape}")
    print(f"Token dimensions: base_h={base_h}, base_w={base_w}")
    
    # Export to ONNX with dynamic axes
    # Dynamic axes: batch dimension (0) and sequence dimension (1) are dynamic
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_depth),
        onnx_path,
        input_names=['image', 'depth'],
        output_names=['x', 'cls_token'],
        opset_version=17,
        dynamic_axes=None,
        do_constant_folding=False,
    )
    
    print(f"Preprocessing ONNX exported: {onnx_path}")
    return onnx_path


def build_encoder_model(model, dummy_input, engine_path, precision='fp16'):
    """
    Build encoder-only model to ONNX.
    Forward: image + depth -> features + cls_token
    """
    onnx_path = engine_path.replace('.engine', '.onnx')
    print(f"Exporting encoder model to ONNX: {onnx_path}")
    
    base_h, base_w = dummy_input['base_h'], dummy_input['base_w']
    
    # Encoder wrapper
    class EncoderWrapper(nn.Module):
        """Encoder only wrapper"""
        
        def __init__(self, encoder, base_h, base_w):
            super().__init__()
            self.encoder = encoder
            self.base_h = base_h
            self.base_w = base_w
            
        def forward(self, image, depth):
            # Forward through encoder
            features, cls_token, _, _ = self.encoder(
                image, depth, 
                self.base_h, self.base_w, 
                return_class_token=True, 
                remap_depth_in='log'
            )
            return features, cls_token
    
    # Create wrapper
    wrapper = EncoderWrapper(model.encoder, base_h, base_w)
    wrapper.eval()
    
    # Dummy inputs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_image = torch.randn(1, 3, 480, 640, dtype=torch.float32, device=device)
    dummy_depth = torch.randn(1, 1, 480, 640, dtype=torch.float32, device=device)
    
    print(f"Exporting encoder with input shape: image={dummy_image.shape}, depth={dummy_depth.shape}")
    print(f"Token dimensions: base_h={base_h}, base_w={base_w}")
    
    # Export to ONNX
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_depth),
        onnx_path,
        input_names=['image', 'depth'],
        output_names=['features', 'cls_token'],
        opset_version=17,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    
    print(f"Encoder ONNX exported: {onnx_path}")
    return onnx_path


def build_full_model(model, dummy_input, engine_path, precision='fp16'):
    """
    Build complete model (Encoder + Decoder) as single ONNX.
    Forward: image + depth -> depth_reg + mask
    """
    onnx_path = engine_path.replace('.engine', '.onnx')
    print(f"Exporting full model to ONNX: {onnx_path}")
    
    base_h, base_w = dummy_input['base_h'], dummy_input['base_w']
    aspect_ratio = base_w / base_h
    output_size = (480, 640)  # 原始图像尺寸
    
    # Complete model wrapper
    class FullModelWrapper(nn.Module):
        """Complete model: Encoder + Decoder in one"""
        
        def __init__(self, model, base_h, base_w, aspect_ratio, output_size):
            super().__init__()
            self.model = model
            self.base_h = base_h
            self.base_w = base_w
            self.aspect_ratio = aspect_ratio
            self.output_size = output_size
            
        def forward(self, image, depth):
            from mdm.utils.geo import normalized_view_plane_uv
            
            # ===== Encoder =====
            features, cls_token, _, _ = self.model.encoder(
                image, depth, self.base_h, self.base_w, 
                return_class_token=True, remap_depth_in='log'
            )
            
            # ===== Decoder =====
            batch_size, _, img_h, img_w = image.shape
            device, dtype = image.device, image.dtype
            
            # Add cls token to features
            features = features + cls_token[..., None, None]
            features = [features, None, None, None, None]
            
            # Dynamically create UV coordinates
            for level in range(5):
                uv = normalized_view_plane_uv(
                    width=self.base_w * 2 ** level,
                    height=self.base_h * 2 ** level,
                    aspect_ratio=self.aspect_ratio,
                    dtype=dtype,
                    device=device
                )
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
                if features[level] is None:
                    features[level] = uv
                else:
                    features[level] = torch.cat([features[level], uv], dim=1)
            
            # Forward through neck
            features = self.model.neck(features)
            
            # Forward through heads
            depth_reg = self.model.depth_head(features)[-1]
            # Upsample to full resolution
            depth_reg = F.interpolate(
                depth_reg, 
                size=self.output_size, 
                mode='bilinear', 
                align_corners=False,
            )
            # Use reshape instead of squeeze
            B, C, H, W = depth_reg.shape
            depth_reg = depth_reg.exp().reshape(B, H, W)
            
            mask = self.model.mask_head(features)[-1]
            mask = F.interpolate(
                mask, 
                size=self.output_size, 
                mode='bilinear', 
                align_corners=False,
            )
            # Use reshape instead of squeeze
            B, C, H, W = mask.shape
            mask = mask.reshape(B, H, W).sigmoid()
            
            return depth_reg, mask
    
    # Create wrapper
    wrapper = FullModelWrapper(model, base_h, base_w, aspect_ratio, output_size)
    wrapper.eval()
    
    # Dummy inputs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_image = torch.randn(1, 3, 480, 640, dtype=torch.float32, device=device)
    dummy_depth = torch.randn(1, 1, 480, 640, dtype=torch.float32, device=device)
    
    print(f"Exporting full model with input shape: image={dummy_image.shape}, depth={dummy_depth.shape}")
    
    # Export with dynamic axes
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_depth),
        onnx_path,
        input_names=['image', 'depth'],
        output_names=['depth_reg', 'mask'],
        opset_version=17,
        dynamic_axes=None,
        do_constant_folding=True,
    )
    
    print(f"Full model ONNX exported: {onnx_path}")
    return onnx_path


def build_engine_from_onnx(onnx_path, engine_path, precision='fp16'):
    """Build TensorRT engine from ONNX file."""
    print(f"Building TensorRT engine from: {onnx_path}")
    
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX Parser Error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX file")
    
    config = builder.create_builder_config()
    
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
    
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    
    # Handle different TensorRT versions for workspace size
    try:
        config.max_workspace_size = 4 * 1024 * 1024 * 1024  # TensorRT 8.x
    except AttributeError:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)  # TensorRT 9.x
    
    # Handle different TensorRT versions for building engine
    try:
        engine = builder.build_engine(network, config)  # TensorRT 8.x
    except AttributeError:
        serialized_engine = builder.build_serialized_network(network, config)  # TensorRT 9.x
        engine = trt.Runtime(logger).deserialize_cuda_engine(serialized_engine)
    
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved: {engine_path}")
    print(f"Engine size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
    
    return engine_path


def export_static(model_path: str, output_path: str, height: int = 480, width: int = 640, export_encoder: bool = True):
    """Export model for TensorRT with static shapes."""
    print(f"Loading model from: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MDMModel.from_pretrained(model_path)
    model.encoder.onnx_compatible_mode = True
    model.to(device)
    model.eval()
    
    # Pre-compute token dimensions
    aspect_ratio = width / height
    min_tokens, max_tokens = model.num_tokens_range
    resolution_level = 9
    num_tokens = int(min_tokens + (resolution_level / 9) * (max_tokens - min_tokens))
    base_h = int(round((num_tokens / aspect_ratio) ** 0.5))
    base_w = int(round((num_tokens * aspect_ratio) ** 0.5))
    
    print(f"Input: {height}x{width}")
    print(f"Tokens: {num_tokens}, base_h={base_h}, base_w={base_w}")
    
    # Dummy inputs
    dummy_input = {
        'base_h': base_h,
        'base_w': base_w,
    }
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    result = {}
    
    # Build encoder-only model
    if export_encoder:
        encoder_engine_path = str(output_dir / 'encoder.engine')
        try:
            # encoder_onnx = build_encoder_model(model, dummy_input, encoder_engine_path, 'fp16')
            encoder_onnx = build_preprocess_model(model, dummy_input, str(output_dir / 'preprocess.onnx'))
            print(f"Encoder model: {encoder_onnx}")
            result['encoder_onnx'] = encoder_onnx
        except Exception as e:
            print(f"Encoder model export failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue even if encoder fails
    
    # Build full model (merged encoder + decoder)
    full_engine_path = str(output_dir / 'full.engine')
    try:
        full_onnx = build_full_model(model, dummy_input, full_engine_path, 'fp16')
        print(f"Full model: {full_onnx}")
    except Exception as e:
        print(f"Full model build failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    
    # result['full_onnx'] = full_onnx
    # result['full_engine'] = full_engine_path if TRT_AVAILABLE else None
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT Engine for LingBot-Depth')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--output', type=str, default='result/model', help='Output prefix')
    parser.add_argument('--height', type=int, default=480, help='Input height')
    parser.add_argument('--width', type=int, default=640, help='Input width')
    parser.add_argument('--precision', type=str, default='fp16', 
                       choices=['fp16', 'fp32', 'int8'], help='Precision')
    parser.add_argument('--encoder-only', action='store_true', help='Only export encoder to ONNX/TensorRT for debugging')
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    print("=" * 60)
    export_static(args.model, args.output, args.height, args.width, export_encoder=True)    
    print("=" * 60)


if __name__ == '__main__':
    main()
