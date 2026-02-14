#!/usr/bin/env python3
"""
ONNX Inference Script for LingBot-Depth Model
验证 ONNX 文件是否正确
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
    
    return image_np


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


def check_onnx_model(onnx_path: str):
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
            print(f"  - {inp.name}: {inp.shape}")
        
        # Get outputs
        outputs = session.get_outputs()
        print(f"\nOutputs ({len(outputs)}):")
        for out in outputs:
            print(f"  - {out.name}: {out.shape}")
        
        print(f"\nONNX model is valid!")
        return session.get_inputs(), session.get_outputs()
        
    except Exception as e:
        print(f"\nONNX model check failed: {e}")
        return None, None


def infer_encoder(onnx_path: str, image_np: np.ndarray, depth_np: np.ndarray) -> Dict:
    """Run Encoder inference."""
    print(f"\n{'='*60}")
    print("Running Encoder inference")
    print('='*60)
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Print expected input shapes
    for inp in session.get_inputs():
        print(f"  Expected input: {inp.name} -> shape={inp.shape}")
    
    # Prepare inputs - ensure 4D shape (B, C, H, W) for ONNX
    # image and depth are the only inputs (base_h/base_w computed internally)
    image_input = np.transpose(image_np, (2, 0, 1))[np.newaxis, :, :, :]  # (1, C, H, W)
    depth_input = depth_np[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    
    inputs = {
        'image': image_input.astype(np.float32),
        'depth': depth_input.astype(np.float32),
    }
    
    print(f"  Input image: shape={inputs['image'].shape}, min={inputs['image'].min():.4f}, max={inputs['image'].max():.4f}, mean={inputs['image'].mean():.4f}")
    print(f"  Input depth: shape={inputs['depth'].shape}, min={inputs['depth'].min():.4f}, max={inputs['depth'].max():.4f}, mean={inputs['depth'].mean():.4f}")
    
    # Run
    start = time.time()
    outputs = session.run(None, inputs)
    elapsed = time.time() - start
    
    print(f"Inference time: {elapsed*1000:.2f}ms")
    
    # Print outputs
    for i, out in enumerate(session.get_outputs()):
        arr = outputs[i]
        print(f"  {out.name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    
    return {out.name: outputs[i] for i, out in enumerate(session.get_outputs())}


def infer_decoder(onnx_path: str, features: np.ndarray, cls_token: np.ndarray) -> Dict:
    """Run Decoder inference."""
    print(f"\n{'='*60}")
    print("Running Decoder inference")
    print('='*60)
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Print input details
    print(f"  Input features: shape={features.shape}, min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    print(f"  Input cls_token: shape={cls_token.shape}, min={cls_token.min():.4f}, max={cls_token.max():.4f}, mean={cls_token.mean():.4f}")
    
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
        print(f"  {out.name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
    
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
        print("Encoder outputs not found! Check ONNX outputs.")
        return
    
    # Stage 2: Decoder
    dec_outputs = infer_decoder(decoder_path, features, cls_token)
    
    # Save results
    depth_pred = dec_outputs.get('depth_reg')
    if depth_pred is not None:
        if depth_pred.ndim == 3 and depth_pred.shape[0] == 1:
            depth_pred = depth_pred.squeeze(0)
        elif depth_pred.ndim == 2:
            pass
        else:
            depth_pred = depth_pred.squeeze()
        
        np.save(output_dir / 'depth_onnx.npy', depth_pred)
        depth_colored = depth_to_color(depth_pred)
        cv2.imwrite(str(output_dir / 'depth_onnx.png'), depth_colored)
        
        print(f"\nResults saved to {output_dir}")
        print(f"   - depth_onnx.npy: shape={depth_pred.shape}")
        print(f"   - depth_onnx.png: shape={depth_colored.shape}")


def main():
    parser = argparse.ArgumentParser(description='ONNX Inference for LingBot-Depth')
    parser.add_argument('--encoder', type=str, help='Encoder ONNX path')
    parser.add_argument('--decoder', type=str, help='Decoder ONNX path')
    parser.add_argument('--input', type=str, default='examples/0', help='Input directory')
    parser.add_argument('--output', type=str, default='result_onnx', help='Output directory')
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    rgb_path = input_dir / 'rgb.png'
    depth_path = input_dir / 'raw_depth.png'
    
    if not rgb_path.exists():
        print(f"Image not found: {rgb_path}")
        return
    if not depth_path.exists():
        print(f"Depth not found: {depth_path}")
        return
    
    image_np = preprocess_image(str(rgb_path))
    depth_np = load_depth(str(depth_path))
    
    print(f"Input image: {image_np.shape[:2]}")
    print(f"Input depth range: {depth_np[depth_np>0].min():.2f} - {depth_np.max():.2f}")
    
    # Check models
    if args.encoder:
        check_onnx_model(args.encoder)
    
    if args.decoder:
        check_onnx_model(args.decoder)
    
    # Run inference
    if args.encoder and args.decoder:
        infer_two_stage(args.encoder, args.decoder, image_np, depth_np, output_dir)
    elif args.encoder:
        infer_encoder(args.encoder, image_np, depth_np)
    elif args.decoder:
        features = np.random.randn(1, 1024, 52, 69).astype(np.float32)
        cls_token = np.random.randn(1, 1024).astype(np.float32)
        infer_decoder(args.decoder, features, cls_token)
    else:
        print("Please specify --encoder and/or --decoder")


if __name__ == '__main__':
    main()
