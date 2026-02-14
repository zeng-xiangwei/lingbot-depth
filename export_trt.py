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

# TensorRT imports
try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available, will generate ONNX only")

from mdm.model.v2 import MDMModel


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
            batch_size = features.shape[0]
            device = features.device
            dtype = features.dtype
            
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
            depth_reg = depth_reg.reshape(B, H, W)
            # Apply remap_depth_out (exp)
            depth_reg = depth_reg.exp()
            
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
        opset_version=16,
        do_constant_folding=True,
        dynamic_axes=None,
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


def export_static(model_path: str, output_path: str, height: int = 480, width: int = 640):
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
    
    return {
        'full_onnx': full_onnx,
        'full_engine': full_engine_path if TRT_AVAILABLE else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Build TensorRT Engine for LingBot-Depth')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--output', type=str, default='result/model', help='Output prefix')
    parser.add_argument('--height', type=int, default=480, help='Input height')
    parser.add_argument('--width', type=int, default=640, help='Input width')
    parser.add_argument('--precision', type=str, default='fp16', 
                       choices=['fp16', 'fp32', 'int8'], help='Precision')
    
    args = parser.parse_args()
    print(f"Arguments: {args}")
    
    print("=" * 60)
    export_static(args.model, args.output, args.height, args.width)
    print("=" * 60)


if __name__ == '__main__':
    main()
