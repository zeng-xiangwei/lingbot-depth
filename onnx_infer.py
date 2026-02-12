#!/usr/bin/env python3
"""
ONNX Inference Script for LingBot-Depth Model
验证 ONNX 文件是否正确

Usage:
    # 验证 Encoder
    python onnx_infer.py --encoder result/encoder_encoder.onnx --input examples/0/

    # 验证 Decoder
    python onnx_infer.py --decoder result/decoder_decoder.onnx --input examples/0/

    # 两阶段推理
    python onnx_infer.py --encoder result/encoder_encoder.onnx --decoder result/decoder_decoder.onnx --input examples/0/
"""
import os
os.environ['XFORMERS_DISABLED'] = '1'

import cv2
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional
import time
import argparse


def preprocess_image(image_path: str) -> tuple:
    """Load and preprocess RGB image."""
    image_np = cv2.imread(image_path)
    if image_np is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image_np = image_np.astype(np.float32) / 255.0
    
    # HWC to CHW
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).numpy()
    
    return image_np, image_tensor


def load_depth(depth_path: str, scale: float = 1000.0) -> np.ndarray:
    """Load depth map from PNG (16-bit)."""
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"Failed to read depth: {depth_path}")
    
    depth_map = depth_map.astype(np.float32) / scale
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    return depth_map


def depth_to_color(depth_map: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Convert depth to color visualization."""
    valid = np.isfinite(depth_map) & (depth_map > 0)
    if vmin is None:
        vmin = depth_map[valid].min() if valid.any() else 0
    if vmax is None:
        vmax = depth_map[valid].max() if valid.any() else 1
    
    normalized = np.clip((depth_map - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = [0, 0, 0]
    return colored


def check_onnx_model(onnx_path: str) -> bool:
    """Check if ONNX file is valid."""
    print(f"\n{'='*60}")
    print(f"Checking ONNX model: {onnx_path}")
    print('='*60)
    
    try:
        # Load session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get inputs
        inputs = session.get_inputs()
        print(f"\nInputs ({len(inputs)}):")
        for inp in inputs:
            dtype_name = getattr(inp, 'dtype', str(getattr(inp, 'element_dtype', 'unknown')))
            print(f"  - {inp.name}: {inp.shape}, dtype={dtype_name}")
        
        # Get outputs
        outputs = session.get_outputs()
        print(f"\nOutputs ({len(outputs)}):")
        for out in outputs:
            dtype_name = getattr(out, 'dtype', str(getattr(out, 'element_dtype', 'unknown')))
            print(f"  - {out.name}: {out.shape}, dtype={dtype_name}")
        
        # Check for unnamed tensors
        for inp in inputs:
            if not inp.name:
                print(f"  WARNING: Unnamed input tensor found!")
        for out in outputs:
            if not out.name:
                print(f"  WARNING: Unnamed output tensor found!")
        
        print(f"\n✅ ONNX model is valid!")
        return True
        
    except Exception as e:
        print(f"\n❌ ONNX model check failed: {e}")
        return False


def infer_encoder(onnx_path: str, image_np: np.ndarray, depth_np: np.ndarray) -> Dict:
    """Run Encoder inference."""
    print(f"\n{'='*60}")
    print("Running Encoder inference")
    print('='*60)
    
    # Auto-detect available providers
    available = ort.get_available_providers()
    providers = ['CUDAExecutionProvider']
    print(f"Using providers: {providers}")
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Print expected input shapes
    for inp in session.get_inputs():
        print(f"  Expected input: {inp.name} -> shape={inp.shape}")
    
    # Prepare inputs - ensure 4D shape (B, C, H, W) for ONNX
    # ONNX expects (B, H, W, C) for conv ops typically
    image_input = np.transpose(image_np, (2, 0, 1))[np.newaxis, :, :, :]  # (1, C, H, W)
    depth_input = depth_np[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    
    inputs = {
        'image': image_input.astype(np.float32),
        'depth': depth_input.astype(np.float32),
        'base_h': np.array(52, dtype=np.int64),  # 0-d array
        'base_w': np.array(69, dtype=np.int64),  # 0-d array
    }
    
    print(f"  Actual image shape: {inputs['image'].shape}")
    print(f"  Actual depth shape: {inputs['depth'].shape}")
    
    # Run
    start = time.time()
    outputs = session.run(None, inputs)
    elapsed = time.time() - start
    
    print(f"Inference time: {elapsed*1000:.2f}ms")
    
    # Print outputs
    for i, out in enumerate(session.get_outputs()):
        arr = outputs[i]
        print(f"  {out.name}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"    min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    
    return {out.name: outputs[i] for i, out in enumerate(session.get_outputs())}


def infer_decoder(onnx_path: str, features: np.ndarray, cls_token: np.ndarray) -> Dict:
    """Run Decoder inference."""
    print(f"\n{'='*60}")
    print("Running Decoder inference")
    print('='*60)
    
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    
    # Prepare inputs
    inputs = {
        'features': features,
        'cls_token': cls_token,
    }
    
    # Run
    start = time.time()
    outputs = session.run(None, inputs)
    elapsed = time.time() - start
    
    print(f"Inference time: {elapsed*1000:.2f}ms")
    
    # Print outputs
    for i, out in enumerate(session.get_outputs()):
        arr = outputs[i]
        print(f"  {out.name}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"    min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    
    return {out.name: outputs[i] for i, out in enumerate(session.get_outputs())}


def infer_two_stage(encoder_path: str, decoder_path: str, image_np: np.ndarray, depth_np: np.ndarray, output_dir: Path):
    """Run two-stage inference."""
    print(f"\n{'='*60}")
    print("Running Two-Stage Inference")
    print('='*60)
    
    # Stage 1: Encoder
    enc_outputs = infer_encoder(encoder_path, image_np, depth_np)
    
    features = enc_outputs.get('features')
    cls_token = enc_outputs.get('cls_token')
    
    if features is None or cls_token is None:
        print("❌ Encoder outputs not found! Check ONNX outputs.")
        return
    
    # Stage 2: Decoder
    dec_outputs = infer_decoder(decoder_path, features, cls_token)
    
    # Save results
    depth_pred = dec_outputs.get('depth_reg')
    if depth_pred is not None:
        # Decoder 输出已经是 (B, H, W) 格式，不需要再 squeeze
        # 如果还有 batch 维度才 squeeze
        if depth_pred.ndim == 3 and depth_pred.shape[0] == 1:
            depth_pred = depth_pred.squeeze(0)
        elif depth_pred.ndim == 2:
            pass  # 已经是 (H, W)
        else:
            depth_pred = depth_pred.squeeze()
        
        # Save as numpy
        np.save(output_dir / 'depth_onnx.npy', depth_pred)
        
        # Save as image
        depth_colored = depth_to_color(depth_pred)
        cv2.imwrite(str(output_dir / 'depth_onnx.png'), depth_colored)
        
        print(f"\n✅ Results saved to {output_dir}")
        print(f"   - depth_onnx.npy: shape={depth_pred.shape}")
        print(f"   - depth_onnx.png: shape={depth_colored.shape}")
        
        # Compare with input
        valid = depth_pred > 0
        if valid.any():
            print(f"\nDepth prediction stats:")
            print(f"   Valid pixels: {valid.sum()} ({valid.sum()/depth_pred.size*100:.1f}%)")
            print(f"   Range: {depth_pred[valid].min():.3f} - {depth_pred[valid].max():.3f} meters")


def main():
    parser = argparse.ArgumentParser(description='ONNX Inference for LingBot-Depth')
    parser.add_argument('--encoder', type=str, help='Encoder ONNX path')
    parser.add_argument('--decoder', type=str, help='Decoder ONNX path')
    parser.add_argument('--input', type=str, default='examples/0', help='Input directory')
    parser.add_argument('--output', type=str, default='result_onnx', help='Output directory')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    rgb_path = input_dir / 'rgb.png'
    depth_path = input_dir / 'raw_depth.png'
    
    if not rgb_path.exists():
        print(f"❌ Image not found: {rgb_path}")
        return
    if not depth_path.exists():
        print(f"❌ Depth not found: {depth_path}")
        return
    
    image_np, image_tensor = preprocess_image(str(rgb_path))
    depth_np = load_depth(str(depth_path))
    
    print(f"Input image: {image_np.shape[:2]}")
    print(f"Input depth range: {depth_np[depth_np>0].min():.2f} - {depth_np.max():.2f}")
    
    # Check models
    if args.encoder:
        if not check_onnx_model(args.encoder):
            return
    
    if args.decoder:
        if not check_onnx_model(args.decoder):
            return
    
    # Run inference
    if args.encoder and args.decoder:
        infer_two_stage(args.encoder, args.decoder, image_np, depth_np, output_dir)
    elif args.encoder:
        infer_encoder(args.encoder, image_np, depth_np)
    elif args.decoder:
        # Need mock features/cls_token for decoder
        features = np.random.randn(1, 1024, 52, 69).astype(np.float32)
        cls_token = np.random.randn(1, 1024).astype(np.float32)
        infer_decoder(args.decoder, features, cls_token)
    else:
        print("❌ Please specify --encoder and/or --decoder")
        print("Example: python onnx_infer.py --encoder model.onnx --input examples/0/")


if __name__ == '__main__':
    main()
