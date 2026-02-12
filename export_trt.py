#!/usr/bin/env python3
"""
TensorRT Engine Build Script for LingBot-Depth Model
分阶段导出：Encoder -> Neck/Heads
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


def build_encoder_engine(model, dummy_input, engine_path, precision='fp16'):
    """
    Build TensorRT engine for encoder using ONNX as intermediate.
    Encoder forward: image + depth -> features + cls_token
    """
    onnx_path = engine_path.replace('.engine', '.onnx')
    print(f"Exporting encoder to ONNX: {onnx_path}")
    
    encoder = model.encoder
    encoder.eval()
    encoder.onnx_compatible_mode = True
    
    device = next(encoder.parameters()).device
    dummy_image = torch.randn(1, 3, 480, 640, dtype=torch.float32, device=device)
    dummy_depth = torch.randn(1, 1, 480, 640, dtype=torch.float32, device=device)
    
    base_h, base_w = dummy_input['base_h'], dummy_input['base_w']
    print(f"Exporting with base_h={base_h}, base_w={base_w}")
    
    # Export encoder - use a wrapper to ensure correct output names
    class EncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            
        def forward(self, image, depth, base_h, base_w):
            # Forward and only return features and cls_token
            features, cls_token, _, _ = self.encoder(image, depth, base_h, base_w, return_class_token=True, remap_depth_in='log')
            return features, cls_token
    
    wrapper = EncoderWrapper(encoder)
    wrapper.eval()
    
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_depth, base_h, base_w),
        onnx_path,
        input_names=['image', 'depth', 'base_h', 'base_w'],
        output_names=['features', 'cls_token'],
        opset_version=16,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    
    print(f"Encoder ONNX exported: {onnx_path}")
    
    # if TRT_AVAILABLE:
    #     return build_engine_from_onnx(onnx_path, engine_path, precision)
    return onnx_path


def build_decoder_engine(model, dummy_input, engine_path, precision='fp16'):
    """
    Build TensorRT engine for decoder (neck + heads).
    Decoder forward: features + cls_token -> depth_reg + mask
    """
    onnx_path = engine_path.replace('.engine', '.onnx')
    print(f"Exporting decoder to ONNX: {onnx_path}")
    
    base_h, base_w = dummy_input['base_h'], dummy_input['base_w']
    aspect_ratio = base_w / base_h
    
    # Dynamically create UV coordinates on the fly (avoid CPU/GPU mismatch)
    class FullDecoderWrapper(nn.Module):
        """Complete decoder that dynamically creates UV coordinates."""
        
        def __init__(self, model, base_h, base_w, aspect_ratio, output_size):
            super().__init__()
            self.model = model
            self.base_h = base_h
            self.base_w = base_w
            self.aspect_ratio = aspect_ratio
            self.output_size = output_size  # (480, 640)
            
        def forward(self, features, cls_token):
            from mdm.utils.geo import normalized_view_plane_uv
            
            batch_size = features.shape[0]
            device = features.device
            dtype = features.dtype
            
            # Add cls token to features
            features = features + cls_token[..., None, None]
            features = [features, None, None, None, None]
            
            # Dynamically create UV coordinates on the same device
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
            depth_reg = None
            mask = None
            if hasattr(self.model, 'depth_head'):
                depth_reg = self.model.depth_head(features)[-1]
                # Upsample to full resolution
                if depth_reg is not None:
                    depth_reg = F.interpolate(
                        depth_reg, 
                        size=self.output_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    # Squeeze: (B, 1, H, W) -> (B, H, W)
                    depth_reg = depth_reg.squeeze(1)
                    # Apply remap_depth_out (exp)
                    depth_reg = depth_reg.exp()
            if hasattr(self.model, 'mask_head'):
                mask = self.model.mask_head(features)[-1]
                if mask is not None:
                    mask = F.interpolate(
                        mask, 
                        size=self.output_size, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    # Apply sigmoid and squeeze
                    mask = mask.squeeze(1).sigmoid()
            
            return depth_reg, mask
    
    decoder = FullDecoderWrapper(model, base_h, base_w, aspect_ratio, output_size=(480, 640))
    decoder.eval()
    
    # Dummy inputs matching encoder output shapes
    B, C, H, W = dummy_input['features_shape']
    dummy_features = torch.randn(B, C, H, W, dtype=torch.float32, device='cuda')
    dummy_cls = torch.randn(B, C, dtype=torch.float32, device='cuda')
    
    print(f"Exporting decoder with features_shape={dummy_input['features_shape']}")
    
    torch.onnx.export(
        decoder,
        (dummy_features, dummy_cls),
        onnx_path,
        input_names=['features', 'cls_token'],
        output_names=['depth_reg', 'mask'],
        opset_version=16,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    
    print(f"Decoder ONNX exported: {onnx_path}")
    
    # if TRT_AVAILABLE:
    #     return build_engine_from_onnx(onnx_path, engine_path, precision)
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
    
    # Feature dimension from encoder
    feat_dim = model.encoder.dim_features
    print(f"Encoder feature dimension: {feat_dim}")
    
    # Dummy inputs for shape inference
    dummy_input = {
        'base_h': base_h,
        'base_w': base_w,
        'features_shape': (1, feat_dim, base_h, base_w),
    }
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Build encoder
    encoder_engine_path = str(output_dir / 'encoder.engine')
    try:
        encoder_onnx = build_encoder_engine(model, dummy_input, encoder_engine_path, 'fp16')
        print(f"Encoder: {encoder_onnx}")
    except Exception as e:
        print(f"Encoder build failed: {e}")
        import traceback
        traceback.print_exc()
        encoder_onnx = str(output_dir / 'encoder.onnx')
    
    # Build decoder
    decoder_engine_path = str(output_dir / 'decoder.engine')
    try:
        decoder_onnx = build_decoder_engine(model, dummy_input, decoder_engine_path, 'fp16')
        print(f"Decoder: {decoder_onnx}")
    except Exception as e:
        print(f"Decoder build failed: {e}")
        import traceback
        traceback.print_exc()
        decoder_onnx = str(output_dir / 'decoder.onnx')
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    
    return {
        'encoder_onnx': encoder_onnx,
        'decoder_onnx': decoder_onnx,
        'encoder_engine': encoder_engine_path if TRT_AVAILABLE else None,
        'decoder_engine': decoder_engine_path if TRT_AVAILABLE else None,
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
